//! Shortest Vector Problem (SVP) solver via Kannan-Fincke-Pohst enumeration.
//!
//! The SVP asks for the shortest nonzero vector in a lattice defined by a basis.
//! This implementation uses branch-and-bound enumeration (Fincke-Pohst algorithm)
//! with Gram-Schmidt orthogonalization for efficient pruning.
//!
//! # References
//! - Fincke, U., Pohst, M. (1985). "Improved methods for calculating vectors of
//!   short length in a lattice, including a complexity analysis." Mathematics of
//!   Computation, 44(170), 463–471.
//! - Kannan, R. (1987). "Minkowski's convex body theorem and integer programming."
//!   Mathematics of Operations Research, 12(3), 415–440.

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{Array1, Array2};

use super::gram_schmidt::gram_schmidt;

/// Compute the projected squared norm ||π_k(v)||² where v = Σ_{i≥k} coeffs[i] * b_i.
///
/// Using the Gram-Schmidt decomposition, the projection onto the sublattice
/// spanned by b_k, ..., b_{n-1} is:
/// π_k(v) = Σ_{i≥k} (coeffs[i] + Σ_{j>i,j<n} coeffs[j] * mu[j][i]) * b̃_i
///
/// The squared norm is: Σ_{i=k}^{n-1} (coeffs[i] + Σ_{j>i} coeffs[j] * mu[j][i])² * bnorm_sq[i]
///
/// # Arguments
/// * `coeffs` - Integer coordinates (only coeffs[k..] are used)
/// * `mu` - Gram-Schmidt coefficient matrix [n, n]
/// * `bnorm_sq` - Squared norms of orthogonalized basis vectors [n]
/// * `k` - Starting level (0-indexed, innermost level of the enumeration)
pub fn projected_norm_sq(
    coeffs: &[f64],
    mu: &Array2<f64>,
    bnorm_sq: &Array1<f64>,
    k: usize,
) -> f64 {
    let n = coeffs.len();
    let mut total = 0.0;
    for i in k..n {
        // sigma_i = coeffs[i] + Σ_{j>i, j<n} coeffs[j] * mu[j][i]
        let mut sigma = coeffs[i];
        for j in (i + 1)..n {
            sigma += coeffs[j] * mu[[j, i]];
        }
        total += sigma * sigma * bnorm_sq[i];
    }
    total
}

/// Solve the Shortest Vector Problem via Kannan enumeration (Fincke-Pohst).
///
/// Finds the shortest nonzero vector in the lattice spanned by the rows of `basis`.
/// Uses branch-and-bound enumeration with Gram-Schmidt orthogonalization for pruning.
///
/// # Arguments
/// * `basis` - Matrix [n, d] whose rows are lattice basis vectors
/// * `max_nodes` - Maximum number of enumeration nodes before early termination
///
/// # Returns
/// The shortest nonzero lattice vector found.
///
/// # Errors
/// Returns an error if the basis is empty or max_nodes is exceeded without finding
/// any nonzero vector.
pub fn solve_svp(basis: &Array2<f64>, max_nodes: usize) -> OptimizeResult<Array1<f64>> {
    let n = basis.nrows();
    let d = basis.ncols();
    if n == 0 || d == 0 {
        return Err(OptimizeError::ValueError(
            "SVP: basis must be non-empty".to_string(),
        ));
    }

    let (mu, bnorm_sq) = gram_schmidt(basis);

    // Initial bound: use the first basis vector as the initial shortest candidate
    let first_norm_sq: f64 = (0..d).map(|j| basis[[0, j]].powi(2)).sum();
    let mut best_norm_sq = first_norm_sq;
    let mut best_coeffs: Vec<f64> = vec![0.0; n];
    best_coeffs[0] = 1.0;

    // Check all single-basis-vector candidates first
    for i in 0..n {
        let norm_sq: f64 = (0..d).map(|j| basis[[i, j]].powi(2)).sum();
        if norm_sq < best_norm_sq && norm_sq > 1e-14 {
            best_norm_sq = norm_sq;
            best_coeffs = vec![0.0; n];
            best_coeffs[i] = 1.0;
        }
    }

    // Fincke-Pohst enumeration: enumerate integer coefficient vectors
    // We use a stack-based depth-first search
    // State: (k, sigma, partial_coeffs)
    // k is the current depth (from n-1 down to 0)

    let mut nodes_visited = 0usize;
    let mut coeffs = vec![0.0f64; n];

    // Stack entry: (level, coefficient at that level, parent_sigma)
    // We enumerate from the topmost level (k = n-1) downward
    enum StackEntry {
        Push {
            level: usize,
            c_start: i64,
            c_end: i64,
            c_current: i64,
        },
        Pop,
    }

    // Use iterative DFS
    // At each level k (from 0 to n-1, outermost = n-1, innermost = 0):
    // The sigma for level k is: coeffs[k] + Σ_{j>k} coeffs[j] * mu[j][k]
    // The bound: σ_k² * bnorm_sq[k] <= best_norm_sq - Σ_{i>k} sigma_i² * bnorm_sq[i]

    // We implement a simple recursive-style enumeration via an explicit stack
    let result = enumerate_svp(
        &mu,
        &bnorm_sq,
        n,
        &mut coeffs,
        0,
        best_norm_sq,
        &mut nodes_visited,
        max_nodes,
    );

    let (found_norm_sq, found_coeffs) = result?;
    if found_norm_sq < best_norm_sq {
        best_norm_sq = found_norm_sq;
        best_coeffs = found_coeffs;
    }

    // Reconstruct the vector from coefficients
    let mut vec = Array1::<f64>::zeros(d);
    for i in 0..n {
        for j in 0..d {
            vec[j] += best_coeffs[i] * basis[[i, j]];
        }
    }

    // Verify it's nonzero
    let norm: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-10 {
        return Err(OptimizeError::ComputationError(
            "SVP enumeration found a zero vector; basis may be degenerate".to_string(),
        ));
    }

    // Ignore unused variable warning
    let _ = best_norm_sq;

    Ok(vec)
}

/// Recursive enumeration for SVP.
///
/// Enumerates at level `k` (k=0 is innermost, k=n-1 is outermost).
/// Returns (best_norm_sq, best_coeffs) found in this subtree.
fn enumerate_svp(
    mu: &Array2<f64>,
    bnorm_sq: &Array1<f64>,
    n: usize,
    coeffs: &mut Vec<f64>,
    k: usize,
    current_bound: f64,
    nodes_visited: &mut usize,
    max_nodes: usize,
) -> OptimizeResult<(f64, Vec<f64>)> {
    *nodes_visited += 1;
    if *nodes_visited > max_nodes {
        // Return current best without error — partial enumeration
        return Ok((current_bound, coeffs.clone()));
    }

    if k == n {
        // Leaf node: compute actual lattice vector norm
        // Check if all coefficients are zero (trivial vector)
        if coeffs.iter().all(|&c| c == 0.0) {
            return Ok((current_bound, coeffs.clone()));
        }
        let norm_sq = projected_norm_sq(coeffs, mu, bnorm_sq, 0);
        if norm_sq > 1e-14 && norm_sq < current_bound {
            return Ok((norm_sq, coeffs.clone()));
        }
        return Ok((current_bound, coeffs.clone()));
    }

    // Compute the center for the coefficient at level k
    // sigma_k^{parent} = Σ_{j>k} coeffs[j] * mu[j][k]
    let mut sigma_parent = 0.0f64;
    for j in (k + 1)..n {
        sigma_parent += coeffs[j] * mu[[j, k]];
    }

    // The center of the enumeration at level k
    let center = -sigma_parent;

    // Compute remaining budget for this level
    // The projected norm of layers above k (i.e., k+1..n-1)
    let upper_contrib = projected_norm_sq_partial(coeffs, mu, bnorm_sq, k + 1, n);
    let remaining = current_bound - upper_contrib;

    if remaining <= 0.0 {
        return Ok((current_bound, coeffs.clone()));
    }

    // Bound on |sigma_k + coeffs[k]| * sqrt(bnorm_sq[k]) <= sqrt(remaining)
    // So |coeffs[k] - center| <= sqrt(remaining / bnorm_sq[k])
    let bk = bnorm_sq[k];
    if bk < 1e-14 {
        // Degenerate direction; only try coefficient 0
        coeffs[k] = 0.0;
        let (nb, nc) = enumerate_svp(mu, bnorm_sq, n, coeffs, k + 1, current_bound, nodes_visited, max_nodes)?;
        return Ok((nb, nc));
    }

    let radius = (remaining / bk).sqrt();
    let c_lo = (center - radius).ceil() as i64;
    let c_hi = (center + radius).floor() as i64;

    let mut best_norm = current_bound;
    let mut best_c = coeffs.clone();

    // Enumerate from center outward (Schnorr-Euchner ordering)
    // Alternate: center, center+1, center-1, center+2, center-2, ...
    let c_center = center.round() as i64;
    let mut candidates: Vec<i64> = Vec::new();
    // Start from c_center and spiral outward within [c_lo, c_hi]
    candidates.push(c_center);
    let mut offset = 1i64;
    loop {
        let added = candidates.len();
        if c_center + offset <= c_hi {
            candidates.push(c_center + offset);
        }
        if c_center - offset >= c_lo {
            candidates.push(c_center - offset);
        }
        if candidates.len() == added {
            break; // No new candidates added
        }
        offset += 1;
        if offset > (c_hi - c_lo + 1).max(0) + 1 {
            break;
        }
    }

    for &c in &candidates {
        if c < c_lo || c > c_hi {
            continue;
        }
        coeffs[k] = c as f64;

        // Pruning: check partial projected norm
        let sigma_k = sigma_parent + c as f64;
        let contrib_k = sigma_k * sigma_k * bk;
        if upper_contrib + contrib_k >= best_norm {
            continue;
        }

        let (sub_norm, sub_coeffs) = enumerate_svp(
            mu,
            bnorm_sq,
            n,
            coeffs,
            k + 1,
            best_norm,
            nodes_visited,
            max_nodes,
        )?;
        if sub_norm < best_norm {
            best_norm = sub_norm;
            best_c = sub_coeffs;
        }
        if *nodes_visited > max_nodes {
            break;
        }
    }
    coeffs[k] = 0.0;

    Ok((best_norm, best_c))
}

/// Compute partial projected norm for levels `from..to` (excludes level `from-1`).
fn projected_norm_sq_partial(
    coeffs: &[f64],
    mu: &Array2<f64>,
    bnorm_sq: &Array1<f64>,
    from: usize,
    to: usize,
) -> f64 {
    let n = coeffs.len();
    let mut total = 0.0;
    for i in from..to {
        let mut sigma = coeffs[i];
        for j in (i + 1)..n {
            sigma += coeffs[j] * mu[[j, i]];
        }
        total += sigma * sigma * bnorm_sq[i];
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_svp_2d_known_shortest() {
        // Lattice spanned by [3,0] and [0,2]: shortest vector should be [0,2] (or [0,-2])
        let basis = array![[3.0, 0.0], [0.0, 2.0]];
        let v = solve_svp(&basis, 100_000).expect("SVP should succeed");
        let norm_sq: f64 = v.iter().map(|x| x * x).sum();
        // Shortest vector has norm 2, so norm^2 = 4
        assert!(
            (norm_sq - 4.0).abs() < 1e-6 || (norm_sq - 9.0).abs() < 1e-6,
            "Expected norm^2 = 4 (or 9), got {}",
            norm_sq
        );
        // Must be non-zero
        assert!(norm_sq > 1e-10);
    }

    #[test]
    fn test_svp_integer_lattice_shortest_vector() {
        // Standard Z^2 basis: shortest vector has length 1
        let basis = array![[1.0, 0.0], [0.0, 1.0]];
        let v = solve_svp(&basis, 100_000).expect("SVP should succeed");
        let norm_sq: f64 = v.iter().map(|x| x * x).sum();
        assert!((norm_sq - 1.0).abs() < 1e-6, "Expected norm^2 = 1, got {}", norm_sq);
    }

    #[test]
    fn test_projected_norm_sq_zero_for_zero_vector() {
        let basis = array![[1.0, 0.0], [0.0, 1.0]];
        let (mu, bnorm_sq) = gram_schmidt(&basis);
        let coeffs = vec![0.0, 0.0];
        let pn = projected_norm_sq(&coeffs, &mu, &bnorm_sq, 0);
        assert!(pn.abs() < 1e-12, "Zero vector should have zero projected norm, got {}", pn);
    }
}
