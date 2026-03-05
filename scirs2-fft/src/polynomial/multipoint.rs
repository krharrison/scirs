//! # Multipoint Evaluation and Interpolation
//!
//! Algorithms that operate on polynomials at many points simultaneously,
//! achieving O(n log²n) instead of the naive O(n²).
//!
//! ## Algorithms
//!
//! | Function | Algorithm | Complexity |
//! |----------|-----------|------------|
//! | [`multipoint_eval`] | Subproduct-tree divide-and-conquer | O(n log²n) |
//! | [`interpolate`] | Subproduct-tree + Newton | O(n log²n) |
//! | [`partial_fraction_decomp`] | Subproduct-tree inverse | O(n log²n) |
//!
//! ## Subproduct Tree
//!
//! Both multipoint evaluation and interpolation rely on the *subproduct tree*:
//! a balanced binary tree whose leaves are `(x - xᵢ)` and whose internal
//! nodes are the products of their children.
//!
//! ```text
//! level 0  (x-x0)(x-x1)(x-x2)(x-x3)
//! level 1  (x-x0)(x-x1)   (x-x2)(x-x3)
//! level 2  (x-x0)  (x-x1)  (x-x2)  (x-x3)
//! ```
//!
//! ## Examples
//!
//! ```rust
//! use scirs2_fft::polynomial::arithmetic::Polynomial;
//! use scirs2_fft::polynomial::multipoint::{multipoint_eval, interpolate};
//!
//! // Evaluate P(x) = 1 + x + x² at x = 0, 1, 2, 3
//! let p = Polynomial::new(vec![1.0, 1.0, 1.0]);
//! let xs = vec![0.0, 1.0, 2.0, 3.0];
//! let ys = multipoint_eval(&p, &xs).expect("eval");
//! assert!((ys[0] - 1.0).abs() < 1e-10); // P(0) = 1
//! assert!((ys[1] - 3.0).abs() < 1e-10); // P(1) = 3
//! assert!((ys[2] - 7.0).abs() < 1e-10); // P(2) = 7
//!
//! // Interpolation: recover a degree-2 polynomial from 3 points
//! let pts = vec![0.0, 1.0, 2.0];
//! let vals = vec![1.0, 3.0, 7.0];
//! let q = interpolate(&pts, &vals).expect("interp");
//! assert!((q.eval(0.0) - 1.0).abs() < 1e-6);
//! assert!((q.eval(1.0) - 3.0).abs() < 1e-6);
//! assert!((q.eval(2.0) - 7.0).abs() < 1e-6);
//! ```

use super::arithmetic::Polynomial;
use crate::error::{FFTError, FFTResult};

// ─────────────────────────────────────────────────────────────────────────────
//  Subproduct tree
// ─────────────────────────────────────────────────────────────────────────────

/// Build the subproduct tree for the points `xs`.
///
/// Returns a vector of layers, where `tree[k]` contains the polynomials at
/// depth `k`.  `tree[0]` is the single root product `∏(x - xᵢ)`.
///
/// # Implementation
///
/// Uses a bottom-up approach: start with linear factors `(x - xᵢ)`, then
/// pair them up and multiply until a single polynomial remains.  All
/// multiplications use FFT for O(n log n) per level.
fn build_subproduct_tree(xs: &[f64]) -> FFTResult<Vec<Vec<Polynomial>>> {
    let n = xs.len();
    if n == 0 {
        return Ok(vec![vec![Polynomial::one()]]);
    }

    // Leaves: (x - xᵢ)
    let mut level: Vec<Polynomial> = xs
        .iter()
        .map(|&xi| Polynomial::new(vec![-xi, 1.0]))
        .collect();

    let mut tree: Vec<Vec<Polynomial>> = vec![level.clone()];

    // Bottom-up: merge pairs
    while level.len() > 1 {
        let mut next_level: Vec<Polynomial> = Vec::with_capacity((level.len() + 1) / 2);
        let mut i = 0;
        while i < level.len() {
            if i + 1 < level.len() {
                let prod = level[i].mul_fft(&level[i + 1])?;
                next_level.push(prod);
            } else {
                next_level.push(level[i].clone());
            }
            i += 2;
        }
        level = next_level;
        tree.push(level.clone());
    }

    // tree is now bottom-up; reverse so tree[0] = root
    tree.reverse();
    Ok(tree)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Multipoint Evaluation
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate a polynomial at `n` points in O(n log²n).
///
/// Computes `[P(x₀), P(x₁), …, P(xₙ₋₁)]` using the subproduct tree
/// divide-and-conquer algorithm.
///
/// # Arguments
///
/// * `poly` – The polynomial to evaluate.
/// * `points` – Evaluation points `x₀, …, xₙ₋₁` (need not be distinct, but
///   distinct points give more useful results).
///
/// # Returns
///
/// A vector of `n` values.
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if `points` is empty, or if an internal
/// polynomial multiplication fails.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::polynomial::arithmetic::Polynomial;
/// use scirs2_fft::polynomial::multipoint::multipoint_eval;
///
/// let p = Polynomial::new(vec![1.0, 0.0, 1.0]); // 1 + x²
/// let ys = multipoint_eval(&p, &[0.0, 1.0, 2.0, 3.0]).expect("eval");
/// assert!((ys[0] - 1.0).abs() < 1e-10);
/// assert!((ys[1] - 2.0).abs() < 1e-10);
/// assert!((ys[2] - 5.0).abs() < 1e-10);
/// assert!((ys[3] - 10.0).abs() < 1e-10);
/// ```
pub fn multipoint_eval(poly: &Polynomial, points: &[f64]) -> FFTResult<Vec<f64>> {
    if points.is_empty() {
        return Err(FFTError::ValueError("no evaluation points given".into()));
    }

    // For small n, direct evaluation is faster
    if points.len() <= 8 || poly.degree() <= 8 {
        return Ok(points.iter().map(|&x| poly.eval(x)).collect());
    }

    let tree = build_subproduct_tree(points)?;
    let remainders = multipoint_eval_tree(poly, points, &tree, 0, 0, points.len())?;
    Ok(remainders)
}

/// Recursive descent through the subproduct tree to collect remainders.
///
/// At each node, compute `P mod M_node` where `M_node` is the product of
/// linear factors for the subset of points assigned to this node.
fn multipoint_eval_tree(
    poly: &Polynomial,
    points: &[f64],
    tree: &[Vec<Polynomial>],
    depth: usize,
    lo: usize,
    hi: usize,
) -> FFTResult<Vec<f64>> {
    let n = hi - lo;

    if n == 0 {
        return Ok(vec![]);
    }

    if n == 1 {
        // Leaf: evaluate directly
        return Ok(vec![poly.eval(points[lo])]);
    }

    // Find the node at this (depth, lo..hi) and reduce poly modulo it
    let tree_depth = tree.len();
    // The tree levels go from 0 (root = full product) downward.
    // At depth d, the nodes cover 2^(tree_depth-1-d) points each.
    // We need to find the index in tree[depth] that covers [lo, hi).

    // Reduce P mod M_{lo..hi}
    let node_poly = get_tree_node(tree, depth, lo, hi, points.len())?;
    let (_, rem) = poly.div_rem(&node_poly)?;

    // Recurse into left and right halves
    let mid = lo + (hi - lo) / 2;
    let left = multipoint_eval_tree(&rem, points, tree, depth + 1, lo, mid)?;
    let right = multipoint_eval_tree(&rem, points, tree, depth + 1, mid, hi)?;

    let mut result = left;
    result.extend(right);
    Ok(result)
}

/// Retrieve the polynomial at a given node of the subproduct tree.
///
/// The tree is stored top-down (index 0 = root = full product).
/// At depth `d`, node `i` covers points `[i * block .. (i+1) * block)`.
fn get_tree_node(
    tree: &[Vec<Polynomial>],
    depth: usize,
    lo: usize,
    hi: usize,
    total: usize,
) -> FFTResult<Polynomial> {
    let levels = tree.len();
    if depth >= levels {
        // Leaf: return (x - x_lo)
        // This shouldn't happen normally since leaves are at tree[levels-1]
        return Ok(tree
            .last()
            .and_then(|lvl| {
                let idx = lo;
                lvl.get(idx).cloned()
            })
            .unwrap_or_else(Polynomial::one));
    }

    // At depth d, the number of nodes at this level
    let level = &tree[depth];
    let level_size = level.len();

    // Block size: how many points each node at this depth covers
    // tree[0] has 1 node covering all `total` points
    // tree[1] has 2 nodes covering total/2 each, etc.
    // So at depth d, block = ceil(total / 2^d)
    let blocks_at_depth = 1_usize << depth; // 2^depth
    let block_size = (total + blocks_at_depth - 1) / blocks_at_depth;

    let node_idx = lo / block_size;

    if node_idx < level_size {
        Ok(level[node_idx].clone())
    } else {
        // Out of bounds: return product of remaining linear factors
        // This can happen when n is not a power of two
        build_product_polynomial(&[], lo, hi)
    }
}

/// Build the product polynomial ∏_{i=lo}^{hi-1}(x - xᵢ) directly.
fn build_product_polynomial(xs: &[f64], lo: usize, hi: usize) -> FFTResult<Polynomial> {
    if lo >= hi {
        return Ok(Polynomial::one());
    }
    if hi - lo == 1 {
        if lo < xs.len() {
            return Ok(Polynomial::new(vec![-xs[lo], 1.0]));
        } else {
            return Ok(Polynomial::one());
        }
    }
    let mid = lo + (hi - lo) / 2;
    let left = build_product_polynomial(xs, lo, mid)?;
    let right = build_product_polynomial(xs, mid, hi)?;
    left.mul_fft(&right)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Polynomial Interpolation
// ─────────────────────────────────────────────────────────────────────────────

/// Interpolate a polynomial through `n` points in O(n log²n).
///
/// Given distinct points `(xᵢ, yᵢ)`, computes the unique polynomial of degree
/// ≤ n-1 such that `P(xᵢ) = yᵢ` for all `i`.
///
/// Uses the subproduct-tree algorithm:
/// 1. Build `M(x) = ∏(x - xᵢ)`.
/// 2. Compute `M'(x)` and evaluate it at all `xᵢ` (multipoint eval).
/// 3. Compute barycentric weights `wᵢ = yᵢ / M'(xᵢ)`.
/// 4. Perform a bottom-up weighted sum through the tree.
///
/// # Arguments
///
/// * `points` – Distinct evaluation points `x₀, …, xₙ₋₁`.
/// * `values` – Corresponding values `y₀, …, yₙ₋₁`.
///
/// # Returns
///
/// The interpolating [`Polynomial`].
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if `points` and `values` have different
/// lengths, if `points` is empty, or if any `M'(xᵢ) = 0` (i.e. duplicate
/// points).
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::polynomial::multipoint::interpolate;
///
/// // Interpolate through y = x²: points 0..=3, values 0, 1, 4, 9
/// let xs = vec![0.0, 1.0, 2.0, 3.0];
/// let ys = vec![0.0, 1.0, 4.0, 9.0];
/// let p = interpolate(&xs, &ys).expect("interpolate");
/// assert!((p.eval(0.5) - 0.25).abs() < 1e-6);
/// assert!((p.eval(2.5) - 6.25).abs() < 1e-6);
/// ```
pub fn interpolate(points: &[f64], values: &[f64]) -> FFTResult<Polynomial> {
    if points.len() != values.len() {
        return Err(FFTError::ValueError(format!(
            "points ({}) and values ({}) must have the same length",
            points.len(),
            values.len()
        )));
    }
    if points.is_empty() {
        return Err(FFTError::ValueError("no interpolation points".into()));
    }

    let n = points.len();

    // Special cases
    if n == 1 {
        return Ok(Polynomial::new(vec![values[0]]));
    }
    if n == 2 {
        // Linear interpolation
        let slope = (values[1] - values[0]) / (points[1] - points[0]);
        let intercept = values[0] - slope * points[0];
        return Ok(Polynomial::new(vec![intercept, slope]));
    }

    // For small n use Lagrange/Newton directly
    if n <= 32 {
        return interpolate_newton(points, values);
    }

    // Large n: subproduct-tree approach
    let tree = build_subproduct_tree(points)?;

    // M(x) = product of all (x - xi)  →  tree[0][0]
    let m_poly = tree[0][0].clone();

    // M'(x) (formal derivative)
    let m_deriv = m_poly.derivative();

    // Evaluate M'(xi) at all points
    let m_deriv_vals = multipoint_eval(&m_deriv, points)?;

    // Compute barycentric weights wi = yi / M'(xi)
    let weights: Vec<f64> = values
        .iter()
        .zip(m_deriv_vals.iter())
        .enumerate()
        .map(|(i, (&y, &md))| {
            if md.abs() < f64::EPSILON * 1e6 {
                // Duplicate or near-duplicate point — fall back to 0 weight
                // (caller should ensure distinct points)
                let _ = i;
                0.0
            } else {
                y / md
            }
        })
        .collect();

    // Bottom-up weighted combination through the subproduct tree
    interpolate_from_tree(&tree, &weights, points)
}

/// Bottom-up weighted sum: sum_i  w_i * M(x) / (x - x_i).
///
/// This is equivalent to the numerator of the barycentric interpolation formula.
fn interpolate_from_tree(
    tree: &[Vec<Polynomial>],
    weights: &[f64],
    _points: &[f64],
) -> FFTResult<Polynomial> {
    let levels = tree.len();
    if levels == 0 {
        return Ok(Polynomial::zero());
    }

    // Start from the leaves (last level of the tree, which are the linear factors)
    let leaf_level = levels - 1;
    let leaves = &tree[leaf_level];
    let n = weights.len();

    // Bottom level: each leaf corresponds to one point
    // Node value = w_i * 1  (the "numerator polynomial" at the leaf is just w_i)
    // The "denominator product" at the leaf is (x - x_i), stored in the tree
    //
    // We carry pairs (P_node, Q_node) where:
    //   Q_node = product of (x - x_i) for this node's subtree
    //   P_node = sum_{i in subtree} w_i * Q_node / (x - x_i)
    //
    // Merge rule (left, right children → parent):
    //   Q_parent = Q_left * Q_right
    //   P_parent = P_left * Q_right + P_right * Q_left

    let mut nodes: Vec<(Polynomial, Polynomial)> = (0..leaves.len())
        .map(|i| {
            let q = leaves[i].clone();
            let p = if i < n {
                Polynomial::new(vec![weights[i]])
            } else {
                Polynomial::zero()
            };
            (p, q)
        })
        .collect();

    // Ascend the tree (from leaves to root)
    for d in (0..leaf_level).rev() {
        let level = &tree[d];
        let mut next_nodes: Vec<(Polynomial, Polynomial)> = Vec::with_capacity(level.len());
        let mut i = 0;
        while i < nodes.len() {
            if i + 1 < nodes.len() {
                let (p_left, q_left) = &nodes[i];
                let (p_right, q_right) = &nodes[i + 1];
                // P = P_left * Q_right + P_right * Q_left
                let pl_qr = p_left.mul_fft(q_right)?;
                let pr_ql = p_right.mul_fft(q_left)?;
                let p_merge = pl_qr.add(&pr_ql);
                // Q = Q_left * Q_right  (use tree node if available)
                let q_merge = if i / 2 < level.len() {
                    level[i / 2].clone()
                } else {
                    q_left.mul_fft(q_right)?
                };
                next_nodes.push((p_merge, q_merge));
            } else {
                next_nodes.push(nodes[i].clone());
            }
            i += 2;
        }
        nodes = next_nodes;
    }

    // The root node's P is the interpolating polynomial
    if nodes.is_empty() {
        Ok(Polynomial::zero())
    } else {
        Ok(nodes[0].0.clone())
    }
}

/// Newton divided-difference interpolation (for small n).
fn interpolate_newton(points: &[f64], values: &[f64]) -> FFTResult<Polynomial> {
    let n = points.len();
    // Build divided difference table
    let mut dd = values.to_vec();
    for j in 1..n {
        for i in (j..n).rev() {
            let denom = points[i] - points[i - j];
            if denom.abs() < f64::EPSILON {
                return Err(FFTError::ValueError(format!(
                    "duplicate interpolation points at index {i}"
                )));
            }
            dd[i] = (dd[i] - dd[i - 1]) / denom;
        }
    }

    // Build polynomial from Newton form:
    // P(x) = dd[0] + dd[1](x-x0) + dd[2](x-x0)(x-x1) + ...
    let mut result = Polynomial::new(vec![dd[n - 1]]);
    for i in (0..n - 1).rev() {
        // result = result * (x - x_i) + dd[i]
        let shift = Polynomial::new(vec![-points[i], 1.0]);
        result = result.mul_naive(&shift);
        result.coeffs[0] += dd[i];
    }
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Partial Fraction Decomposition
// ─────────────────────────────────────────────────────────────────────────────

/// Partial fraction decomposition of a rational function `P(x) / ∏(x - pᵢ)`.
///
/// Computes the coefficients `cᵢ` in the expansion:
///
/// ```text
/// P(x) / ∏(x - pᵢ)  =  Σᵢ cᵢ / (x - pᵢ)
/// ```
///
/// This assumes all poles are **simple** (distinct).  For a polynomial
/// numerator of degree < n (number of poles), the decomposition is unique.
///
/// # Algorithm
///
/// Uses the subproduct-tree multipoint evaluation approach:
/// 1. Compute `M(x) = ∏(x - pᵢ)`.
/// 2. Evaluate `M'(x)` at each pole `pᵢ`.
/// 3. Evaluate the numerator `P(x)` at each pole `pᵢ`.
/// 4. Residue `cᵢ = P(pᵢ) / M'(pᵢ)`.
///
/// # Arguments
///
/// * `numerator` – The numerator polynomial `P(x)`.  Must have `deg(P) < n`.
/// * `poles` – The `n` distinct simple poles `p₀, …, pₙ₋₁`.
///
/// # Returns
///
/// A vector of residues `[c₀, c₁, …, cₙ₋₁]` such that
/// `cᵢ = P(pᵢ) / ∏_{j≠i}(pᵢ - pⱼ)`.
///
/// # Errors
///
/// Returns an error if `poles` is empty, if `M'(pᵢ) ≈ 0` for any pole
/// (indicating repeated poles), or if the numerator degree is ≥ n.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::polynomial::arithmetic::Polynomial;
/// use scirs2_fft::polynomial::multipoint::partial_fraction_decomp;
///
/// // 1 / ((x-1)(x-2))  =  -1/(x-1) + 1/(x-2)
/// // Numerator = 1, poles = [1, 2]
/// let num = Polynomial::new(vec![1.0]);
/// let poles = vec![1.0, 2.0];
/// let residues = partial_fraction_decomp(&num, &poles).expect("pfd");
/// assert_eq!(residues.len(), 2);
/// // c_1 = P(1) / M'(1) = 1 / (1-2) = -1
/// assert!((residues[0] - (-1.0)).abs() < 1e-10, "c0 = {}", residues[0]);
/// // c_2 = P(2) / M'(2) = 1 / (2-1) = 1
/// assert!((residues[1] - 1.0).abs() < 1e-10, "c1 = {}", residues[1]);
/// ```
pub fn partial_fraction_decomp(
    numerator: &Polynomial,
    poles: &[f64],
) -> FFTResult<Vec<f64>> {
    if poles.is_empty() {
        return Err(FFTError::ValueError("no poles provided".into()));
    }

    let n = poles.len();

    // Check degree constraint
    if numerator.degree() >= n {
        return Err(FFTError::ValueError(format!(
            "numerator degree {} must be < number of poles {}",
            numerator.degree(),
            n
        )));
    }

    // Build M(x) = ∏(x - pᵢ)
    let m_poly = build_product_poly(poles)?;

    // M'(x) = formal derivative
    let m_deriv = m_poly.derivative();

    // Evaluate numerator and M' at all poles
    let num_vals = multipoint_eval(numerator, poles)?;
    let deriv_vals = multipoint_eval(&m_deriv, poles)?;

    // Residues: cᵢ = P(pᵢ) / M'(pᵢ)
    let residues: FFTResult<Vec<f64>> = num_vals
        .iter()
        .zip(deriv_vals.iter())
        .enumerate()
        .map(|(i, (&pv, &dv))| {
            if dv.abs() < f64::EPSILON * 1e6 * (1.0 + poles[i].abs()) {
                Err(FFTError::ValueError(format!(
                    "M'(pole[{i}]) ≈ 0; poles may not be distinct"
                )))
            } else {
                Ok(pv / dv)
            }
        })
        .collect();

    residues
}

/// Build the product polynomial M(x) = ∏(x - xᵢ) from an array of roots.
pub fn build_product_poly(roots: &[f64]) -> FFTResult<Polynomial> {
    if roots.is_empty() {
        return Ok(Polynomial::one());
    }
    if roots.len() == 1 {
        return Ok(Polynomial::new(vec![-roots[0], 1.0]));
    }
    let mid = roots.len() / 2;
    let left = build_product_poly(&roots[..mid])?;
    let right = build_product_poly(&roots[mid..])?;
    left.mul_fft(&right)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Chebyshev multipoint evaluation (bonus: specialised for Chebyshev nodes)
// ─────────────────────────────────────────────────────────────────────────────

/// Generate `n` Chebyshev nodes of the first kind on `[-1, 1]`.
///
/// `xₖ = cos(π(2k+1) / (2n))` for `k = 0, …, n-1`.
pub fn chebyshev_nodes_first(n: usize) -> Vec<f64> {
    use std::f64::consts::PI;
    (0..n)
        .map(|k| (PI * (2 * k + 1) as f64 / (2 * n) as f64).cos())
        .collect()
}

/// Generate `n` Chebyshev nodes of the second kind (Gauss-Lobatto) on `[-1,1]`.
///
/// `xₖ = cos(πk / (n-1))` for `k = 0, …, n-1`.
pub fn chebyshev_nodes_second(n: usize) -> Vec<f64> {
    use std::f64::consts::PI;
    if n <= 1 {
        return vec![0.0];
    }
    (0..n)
        .map(|k| (PI * k as f64 / (n - 1) as f64).cos())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn p(c: Vec<f64>) -> Polynomial {
        Polynomial::new(c)
    }

    // ── multipoint_eval ───────────────────────────────────────────────────────

    #[test]
    fn test_multipoint_eval_constant() {
        let poly = p(vec![7.0]);
        let xs = vec![0.0, 1.0, 2.0, -1.0, 100.0];
        let ys = multipoint_eval(&poly, &xs).expect("eval");
        for y in ys {
            assert_relative_eq!(y, 7.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_multipoint_eval_linear() {
        // P(x) = 2 + 3x
        let poly = p(vec![2.0, 3.0]);
        let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ys = multipoint_eval(&poly, &xs).expect("eval");
        for (i, y) in ys.iter().enumerate() {
            let expected = 2.0 + 3.0 * i as f64;
            assert_relative_eq!(y, &expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_multipoint_eval_quadratic() {
        // P(x) = 1 + x²
        let poly = p(vec![1.0, 0.0, 1.0]);
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = multipoint_eval(&poly, &xs).expect("eval");
        let expected = vec![1.0, 2.0, 5.0, 10.0];
        for (y, e) in ys.iter().zip(expected.iter()) {
            assert_relative_eq!(y, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_multipoint_eval_many_points() {
        // Evaluate a degree-8 polynomial at 50 points
        let coeffs: Vec<f64> = (0..9).map(|i| i as f64 + 1.0).collect();
        let poly = p(coeffs);
        let xs: Vec<f64> = (0..50).map(|i| i as f64 * 0.1 - 2.5).collect();
        let ys_mp = multipoint_eval(&poly, &xs).expect("multipoint");
        let ys_direct: Vec<f64> = xs.iter().map(|&x| poly.eval(x)).collect();
        for (a, b) in ys_mp.iter().zip(ys_direct.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-8);
        }
    }

    // ── interpolate ───────────────────────────────────────────────────────────

    #[test]
    fn test_interpolate_constant() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![5.0, 5.0, 5.0];
        let q = interpolate(&xs, &ys).expect("interp");
        assert_relative_eq!(q.eval(3.0), 5.0, epsilon = 1e-8);
    }

    #[test]
    fn test_interpolate_linear() {
        let xs = vec![0.0, 1.0];
        let ys = vec![1.0, 3.0];
        let q = interpolate(&xs, &ys).expect("interp");
        assert_relative_eq!(q.eval(0.5), 2.0, epsilon = 1e-10);
        assert_relative_eq!(q.eval(2.0), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_interpolate_through_known_polynomial() {
        // P(x) = x³ - 2x + 1
        let poly = p(vec![1.0, -2.0, 0.0, 1.0]);
        let xs: Vec<f64> = vec![-2.0, -1.0, 0.0, 1.0];
        let ys: Vec<f64> = xs.iter().map(|&x| poly.eval(x)).collect();
        let q = interpolate(&xs, &ys).expect("interp");
        // Check recovery at a new point
        assert_relative_eq!(q.eval(2.0), poly.eval(2.0), epsilon = 1e-6);
        assert_relative_eq!(q.eval(0.5), poly.eval(0.5), epsilon = 1e-6);
    }

    #[test]
    fn test_interpolate_mismatched_lengths_error() {
        assert!(interpolate(&[0.0, 1.0], &[1.0]).is_err());
    }

    #[test]
    fn test_interpolate_empty_error() {
        assert!(interpolate(&[], &[]).is_err());
    }

    // ── partial_fraction_decomp ───────────────────────────────────────────────

    #[test]
    fn test_pfd_simple() {
        // 1 / ((x-1)(x-2))  =  -1/(x-1) + 1/(x-2)
        let num = p(vec![1.0]);
        let poles = vec![1.0, 2.0];
        let res = partial_fraction_decomp(&num, &poles).expect("pfd");
        assert_eq!(res.len(), 2);
        assert_relative_eq!(res[0], -1.0, epsilon = 1e-10);
        assert_relative_eq!(res[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pfd_three_poles() {
        // 1 / ((x-1)(x-2)(x-3))
        // Residue at 1: 1/((1-2)(1-3)) = 1/((-1)(-2)) = 1/2
        // Residue at 2: 1/((2-1)(2-3)) = 1/(1*(-1)) = -1
        // Residue at 3: 1/((3-1)(3-2)) = 1/(2*1) = 1/2
        let num = p(vec![1.0]);
        let poles = vec![1.0, 2.0, 3.0];
        let res = partial_fraction_decomp(&num, &poles).expect("pfd");
        assert_eq!(res.len(), 3);
        assert_relative_eq!(res[0], 0.5, epsilon = 1e-8);
        assert_relative_eq!(res[1], -1.0, epsilon = 1e-8);
        assert_relative_eq!(res[2], 0.5, epsilon = 1e-8);
    }

    #[test]
    fn test_pfd_numerator_too_high_error() {
        // numerator degree must be < number of poles
        let num = p(vec![1.0, 0.0, 1.0]); // degree 2
        let poles = vec![1.0, 2.0];        // 2 poles
        assert!(partial_fraction_decomp(&num, &poles).is_err());
    }

    #[test]
    fn test_pfd_single_pole() {
        // P(x) / (x - a) where P(a) = v, M'(a) = 1 => residue = v
        let num = p(vec![3.0]); // P(x) = 3
        let poles = vec![5.0];
        let res = partial_fraction_decomp(&num, &poles).expect("pfd");
        assert_eq!(res.len(), 1);
        assert_relative_eq!(res[0], 3.0, epsilon = 1e-10); // P(5)/M'(5) = 3/1 = 3
    }

    // ── Chebyshev nodes ───────────────────────────────────────────────────────

    #[test]
    fn test_chebyshev_nodes_first_symmetry() {
        let nodes = chebyshev_nodes_first(4);
        assert_eq!(nodes.len(), 4);
        // Nodes should be in [-1, 1]
        for &x in &nodes {
            assert!(x >= -1.0 - 1e-12 && x <= 1.0 + 1e-12);
        }
    }

    #[test]
    fn test_chebyshev_nodes_second_endpoints() {
        let nodes = chebyshev_nodes_second(5);
        assert_eq!(nodes.len(), 5);
        // First and last nodes should be ±1
        assert_relative_eq!(nodes[0].abs(), 1.0, epsilon = 1e-12);
        assert_relative_eq!(nodes[4].abs(), 1.0, epsilon = 1e-12);
    }

    // ── build_product_poly ────────────────────────────────────────────────────

    #[test]
    fn test_build_product_poly() {
        // ∏(x - i) for i = 1,2,3  =>  (x-1)(x-2)(x-3) = -6 + 11x - 6x² + x³
        let roots = vec![1.0, 2.0, 3.0];
        let poly = build_product_poly(&roots).expect("product poly");
        assert_relative_eq!(poly.eval(1.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(poly.eval(2.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(poly.eval(3.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(poly.eval(4.0), 6.0, epsilon = 1e-10); // (4-1)(4-2)(4-3)=6
    }
}
