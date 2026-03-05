//! Sparse grid (Smolyak) quadrature for multi-dimensional integration
//!
//! This module implements the Smolyak construction for high-dimensional numerical
//! integration. Instead of a full tensor product of one-dimensional quadrature rules
//! (which scales as O(n^d)), sparse grids build a carefully chosen subset of tensor
//! products that preserves polynomial exactness while requiring only
//! O(n * (log n)^{d-1}) points.
//!
//! ## Key features
//!
//! - Smolyak construction with any 1-D nested quadrature rule
//! - Built-in Clenshaw-Curtis and Gauss-Legendre base rules
//! - Configurable level for accuracy / cost trade-off
//! - Exact for total-degree polynomials up to `2*level - 1`
//!
//! ## References
//!
//! - S.A. Smolyak (1963), "Quadrature and interpolation formulas formed by
//!   tensor products of one-dimensional operators"
//! - T. Gerstner, M. Griebel (1998), "Numerical integration using sparse grids"

use crate::error::{IntegrateError, IntegrateResult};
use crate::IntegrateFloat;
use scirs2_core::ndarray::Array1;
use std::collections::HashMap;

/// Helper to convert f64 constants to generic Float type
#[inline(always)]
fn to_f<F: IntegrateFloat>(value: f64) -> F {
    F::from_f64(value).unwrap_or_else(|| F::zero())
}

// ---------------------------------------------------------------------------
// 1-D rule abstraction
// ---------------------------------------------------------------------------

/// A one-dimensional quadrature rule: nodes and weights on `[-1,1]`.
#[derive(Debug, Clone)]
pub struct OneDRule {
    /// Nodes on `[-1,1]`
    pub nodes: Vec<f64>,
    /// Corresponding weights
    pub weights: Vec<f64>,
}

/// Family of nested 1-D rules indexed by level.
pub trait OneDRuleFamily: Send + Sync {
    /// Return the 1-D rule at the given level (level >= 1).
    /// Level 1 should have 1 point, level 2 should have 3, etc.
    fn rule(&self, level: usize) -> IntegrateResult<OneDRule>;
}

/// Clenshaw-Curtis nested rule family (levels 1, 2, 3, ... give 1, 3, 5, ... points).
#[derive(Debug, Clone, Copy)]
pub struct ClenshawCurtisFamily;

impl OneDRuleFamily for ClenshawCurtisFamily {
    fn rule(&self, level: usize) -> IntegrateResult<OneDRule> {
        if level == 0 {
            return Err(IntegrateError::ValueError("Level must be >= 1".into()));
        }
        // n = 2^(level-1) for level >= 2, n = 1 for level 1
        let n = if level == 1 { 1 } else { 1 << (level - 1) };
        cc_rule_f64(n)
    }
}

/// Gauss-Legendre rule family (not nested, but usable with Smolyak).
#[derive(Debug, Clone, Copy)]
pub struct GaussLegendreFamily;

impl OneDRuleFamily for GaussLegendreFamily {
    fn rule(&self, level: usize) -> IntegrateResult<OneDRule> {
        if level == 0 {
            return Err(IntegrateError::ValueError("Level must be >= 1".into()));
        }
        // level k gives a k-point Gauss-Legendre rule
        gl_rule_f64(level)
    }
}

// ---------------------------------------------------------------------------
// Low-level rule generators (f64)
// ---------------------------------------------------------------------------

/// Generate Clenshaw-Curtis nodes and weights with `n+1` points on `[-1,1]`.
fn cc_rule_f64(n: usize) -> IntegrateResult<OneDRule> {
    if n == 0 {
        // 1-point midpoint rule
        return Ok(OneDRule {
            nodes: vec![0.0],
            weights: vec![2.0],
        });
    }
    if n == 1 {
        return Ok(OneDRule {
            nodes: vec![-1.0, 0.0, 1.0],
            weights: vec![1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0],
        });
    }

    let nf = n as f64;
    let pi = std::f64::consts::PI;
    let mut nodes = Vec::with_capacity(n + 1);
    let mut weights = Vec::with_capacity(n + 1);

    for j in 0..=n {
        let theta = j as f64 * pi / nf;
        nodes.push(theta.cos());
    }

    let half_n = n / 2;
    for j in 0..=n {
        let c_j: f64 = if j == 0 || j == n { 1.0 } else { 2.0 };
        let theta_j = j as f64 * pi / nf;
        let mut s = 0.0_f64;
        for k in 1..=half_n {
            let b_k: f64 = if k < half_n || (!n.is_multiple_of(2) && k == half_n) {
                2.0
            } else {
                1.0
            };
            let denom = (4 * k * k) as f64 - 1.0;
            s += b_k / denom * (2.0 * k as f64 * theta_j).cos();
        }
        weights.push(c_j / nf * (1.0 - s));
    }

    Ok(OneDRule { nodes, weights })
}

/// Generate Gauss-Legendre nodes and weights with `n` points on `[-1,1]`.
fn gl_rule_f64(n: usize) -> IntegrateResult<OneDRule> {
    if n == 0 {
        return Err(IntegrateError::ValueError(
            "Number of GL points must be >= 1".into(),
        ));
    }
    if n == 1 {
        return Ok(OneDRule {
            nodes: vec![0.0],
            weights: vec![2.0],
        });
    }

    let pi = std::f64::consts::PI;
    let mut nodes = vec![0.0_f64; n];
    let mut weights = vec![0.0_f64; n];

    // Golub-Welsch: eigenvalue approach via Newton iteration on Legendre polynomials
    let m = n.div_ceil(2);
    for i in 0..m {
        // Initial guess
        let mut z = ((i as f64 + 0.75) / (n as f64 + 0.5) * pi).cos();

        for _ in 0..100 {
            let mut p0 = 1.0_f64;
            let mut p1 = z;
            for k in 2..=n {
                let kf = k as f64;
                let p2 = ((2.0 * kf - 1.0) * z * p1 - (kf - 1.0) * p0) / kf;
                p0 = p1;
                p1 = p2;
            }
            // p1 = P_n(z), derivative = n * (z * P_n - P_{n-1}) / (z^2 - 1)
            let nf = n as f64;
            let dp = nf * (z * p1 - p0) / (z * z - 1.0);
            let delta = p1 / dp;
            z -= delta;
            if delta.abs() < 1e-15 {
                break;
            }
        }

        nodes[i] = -z;
        nodes[n - 1 - i] = z;

        // Weight
        let mut p0 = 1.0_f64;
        let mut p1 = z;
        for k in 2..=n {
            let kf = k as f64;
            let p2 = ((2.0 * kf - 1.0) * z * p1 - (kf - 1.0) * p0) / kf;
            p0 = p1;
            p1 = p2;
        }
        let nf = n as f64;
        let dp = nf * (z * p1 - p0) / (z * z - 1.0);
        let w = 2.0 / ((1.0 - z * z) * dp * dp);
        weights[i] = w;
        weights[n - 1 - i] = w;
    }

    Ok(OneDRule { nodes, weights })
}

// ---------------------------------------------------------------------------
// Smolyak sparse grid construction
// ---------------------------------------------------------------------------

/// Result of sparse grid quadrature
#[derive(Debug, Clone)]
pub struct SparseGridResult<F: IntegrateFloat> {
    /// Estimated value of the integral
    pub value: F,
    /// Number of grid points used
    pub n_points: usize,
    /// Smolyak level
    pub level: usize,
    /// Dimensionality
    pub dim: usize,
}

/// Options for sparse grid quadrature
#[derive(Debug, Clone)]
pub struct SparseGridOptions {
    /// Smolyak level (higher = more accurate, default 4)
    pub level: usize,
    /// Rule family to use
    pub rule_family: SparseGridRuleFamily,
}

/// Which 1-D rule family to use
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SparseGridRuleFamily {
    /// Clenshaw-Curtis (nested, good for smooth functions)
    ClenshawCurtis,
    /// Gauss-Legendre (higher polynomial exactness per point)
    GaussLegendre,
}

impl Default for SparseGridOptions {
    fn default() -> Self {
        Self {
            level: 4,
            rule_family: SparseGridRuleFamily::ClenshawCurtis,
        }
    }
}

/// Enumerate all multi-indices alpha in N^d with |alpha|_1 = q and alpha_i >= 1.
fn enumerate_multi_indices(d: usize, q: usize) -> Vec<Vec<usize>> {
    if d == 0 {
        return if q == 0 { vec![vec![]] } else { vec![] };
    }
    if d == 1 {
        return if q >= 1 { vec![vec![q]] } else { vec![] };
    }
    let mut result = Vec::new();
    // alpha_0 ranges from 1 to q - (d-1)  (since remaining d-1 components must be >= 1)
    let max_first = if q >= d { q - d + 1 } else { return result };
    for a0 in 1..=max_first {
        let sub = enumerate_multi_indices(d - 1, q - a0);
        for mut tail in sub {
            let mut row = Vec::with_capacity(d);
            row.push(a0);
            row.append(&mut tail);
            result.push(row);
        }
    }
    result
}

/// Build a sparse grid and compute the integral of `f` over `[a, b]^d`
/// using the Smolyak algorithm at the specified level.
///
/// # Arguments
///
/// * `f`       - Integrand accepting an `Array1<F>` of length `d`
/// * `ranges`  - Integration ranges `(lower, upper)` per dimension
/// * `options` - Optional sparse grid parameters
///
/// # Examples
///
/// ```
/// use scirs2_integrate::sparse_grid::{sparse_grid_quad, SparseGridOptions};
/// use scirs2_core::ndarray::Array1;
///
/// // Integrate f(x,y) = x*y over [0,1]^2 => 0.25
/// let result = sparse_grid_quad(
///     |x: &Array1<f64>| x[0] * x[1],
///     &[(0.0, 1.0), (0.0, 1.0)],
///     None,
/// ).expect("sparse_grid_quad");
/// assert!((result.value - 0.25).abs() < 1e-10);
/// ```
pub fn sparse_grid_quad<F, Func>(
    f: Func,
    ranges: &[(F, F)],
    options: Option<SparseGridOptions>,
) -> IntegrateResult<SparseGridResult<F>>
where
    F: IntegrateFloat,
    Func: Fn(&Array1<F>) -> F,
{
    let opts = options.unwrap_or_default();
    let d = ranges.len();
    if d == 0 {
        return Err(IntegrateError::ValueError(
            "At least one dimension required".into(),
        ));
    }
    let level = opts.level;
    if level == 0 {
        return Err(IntegrateError::ValueError("Level must be >= 1".into()));
    }

    // Build 1-D rules at all needed levels
    let max_level_1d = level; // max component-level is level (from Smolyak formula)
    let family: Box<dyn OneDRuleFamily> = match opts.rule_family {
        SparseGridRuleFamily::ClenshawCurtis => Box::new(ClenshawCurtisFamily),
        SparseGridRuleFamily::GaussLegendre => Box::new(GaussLegendreFamily),
    };

    let mut rules_1d = Vec::with_capacity(max_level_1d + 1);
    rules_1d.push(OneDRule {
        nodes: vec![],
        weights: vec![],
    }); // placeholder for index 0
    for lv in 1..=max_level_1d {
        rules_1d.push(family.rule(lv)?);
    }

    // Smolyak formula:
    // Q^d_l = sum_{l-d+1 <= |alpha| <= l, alpha_i >= 1}
    //         (-1)^{l - |alpha|} * C(d-1, l-|alpha|) * (Q^1_{alpha_1} x ... x Q^1_{alpha_d})
    //
    // where C(n,k) is the binomial coefficient.

    // We accumulate weighted contributions into a hashmap keyed by point coordinates
    // (discretised to avoid floating-point deduplication issues).
    let mut point_weights: HashMap<Vec<i64>, (Array1<F>, F)> = HashMap::new();
    let mut total_points_used: usize = 0;

    let q_min = if level >= d { level - d + 1 } else { 1 };
    let q_max = level;

    for q in q_min..=q_max {
        let multi_indices = enumerate_multi_indices(d, q);
        let sign: f64 = if (level - q).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        let binom = binomial_coeff(d - 1, level - q);
        let coeff = sign * binom as f64;

        for alpha in &multi_indices {
            // Build tensor product of 1-D rules at levels alpha[0], ..., alpha[d-1]
            // Map each 1-D rule from [-1,1] to [a_i, b_i]
            add_tensor_product(
                &rules_1d,
                alpha,
                ranges,
                to_f::<F>(coeff),
                &mut point_weights,
                &mut total_points_used,
            )?;
        }
    }

    // Sum up
    let mut integral = F::zero();
    for (pt, w) in point_weights.values() {
        integral += *w * f(pt);
    }

    Ok(SparseGridResult {
        value: integral,
        n_points: point_weights.len(),
        level,
        dim: d,
    })
}

/// Add the weighted tensor product contribution to the point-weight map.
fn add_tensor_product<F: IntegrateFloat>(
    rules: &[OneDRule],
    alpha: &[usize],
    ranges: &[(F, F)],
    coeff: F,
    map: &mut HashMap<Vec<i64>, (Array1<F>, F)>,
    total: &mut usize,
) -> IntegrateResult<()> {
    let d = alpha.len();
    let half = 0.5_f64;

    // Collect 1-D nodes/weights mapped to physical coordinates
    let mut dim_nodes: Vec<Vec<f64>> = Vec::with_capacity(d);
    let mut dim_weights: Vec<Vec<f64>> = Vec::with_capacity(d);

    for (i, &lv) in alpha.iter().enumerate() {
        let rule = &rules[lv];
        let (a_f64, b_f64) = (
            ranges[i]
                .0
                .to_f64()
                .ok_or_else(|| IntegrateError::ComputationError("f64 conversion".into()))?,
            ranges[i]
                .1
                .to_f64()
                .ok_or_else(|| IntegrateError::ComputationError("f64 conversion".into()))?,
        );
        let mid = (a_f64 + b_f64) * half;
        let half_len = (b_f64 - a_f64) * half;

        let mapped_nodes: Vec<f64> = rule.nodes.iter().map(|&x| mid + half_len * x).collect();
        let mapped_weights: Vec<f64> = rule.weights.iter().map(|&w| w * half_len).collect();
        dim_nodes.push(mapped_nodes);
        dim_weights.push(mapped_weights);
    }

    // Iterate over tensor product
    let sizes: Vec<usize> = dim_nodes.iter().map(|v| v.len()).collect();
    let total_size: usize = sizes.iter().product();
    *total += total_size;

    let mut indices = vec![0_usize; d];
    for _ in 0..total_size {
        // Compute point and weight
        let mut w_prod = 1.0_f64;
        let mut point_f64 = vec![0.0_f64; d];
        for k in 0..d {
            point_f64[k] = dim_nodes[k][indices[k]];
            w_prod *= dim_weights[k][indices[k]];
        }

        // Key for deduplication (round to 14 significant digits)
        let key: Vec<i64> = point_f64
            .iter()
            .map(|&x| (x * 1e14).round() as i64)
            .collect();

        let w_contrib: F = coeff * F::from_f64(w_prod).unwrap_or_else(|| F::zero());

        let entry = map.entry(key).or_insert_with(|| {
            let pt_arr = Array1::from_vec(
                point_f64
                    .iter()
                    .map(|&v| F::from_f64(v).unwrap_or_else(|| F::zero()))
                    .collect(),
            );
            (pt_arr, F::zero())
        });
        entry.1 += w_contrib;

        // Increment multi-index (lexicographic)
        let mut carry = true;
        for k in (0..d).rev() {
            if carry {
                indices[k] += 1;
                if indices[k] >= sizes[k] {
                    indices[k] = 0;
                } else {
                    carry = false;
                }
            }
        }
    }

    Ok(())
}

/// Binomial coefficient C(n, k)
fn binomial_coeff(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k); // symmetry
    let mut result: usize = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_1d_constant() {
        // integral of 1 over [0,1] = 1
        let res = sparse_grid_quad(
            |_x: &Array1<f64>| 1.0,
            &[(0.0, 1.0)],
            Some(SparseGridOptions {
                level: 2,
                ..Default::default()
            }),
        )
        .expect("sparse_grid_quad");
        assert!(
            (res.value - 1.0).abs() < 1e-14,
            "1D constant: got {}",
            res.value
        );
    }

    #[test]
    fn test_2d_polynomial() {
        // integral of x*y over [0,1]^2 = 0.25
        let res = sparse_grid_quad(
            |x: &Array1<f64>| x[0] * x[1],
            &[(0.0, 1.0), (0.0, 1.0)],
            Some(SparseGridOptions {
                level: 3,
                ..Default::default()
            }),
        )
        .expect("sparse_grid_quad");
        assert!(
            (res.value - 0.25).abs() < 1e-10,
            "2D x*y: got {}",
            res.value
        );
    }

    #[test]
    fn test_3d_polynomial() {
        // integral of x*y*z over [0,1]^3 = 0.125
        let res = sparse_grid_quad(
            |x: &Array1<f64>| x[0] * x[1] * x[2],
            &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            Some(SparseGridOptions {
                level: 3,
                ..Default::default()
            }),
        )
        .expect("sparse_grid_quad");
        assert!(
            (res.value - 0.125).abs() < 1e-10,
            "3D x*y*z: got {}",
            res.value
        );
    }

    #[test]
    fn test_2d_gaussian() {
        // integral of exp(-(x^2+y^2)) over [-3,3]^2 ≈ pi (approx 3.1415...)
        // Exact: integral of exp(-r^2) from -inf to inf = sqrt(pi), so 2D = pi
        // But on [-3,3]^2 it is very close to pi.
        let res = sparse_grid_quad(
            |x: &Array1<f64>| (-x[0] * x[0] - x[1] * x[1]).exp(),
            &[(-3.0, 3.0), (-3.0, 3.0)],
            Some(SparseGridOptions {
                level: 8,
                ..Default::default()
            }),
        )
        .expect("sparse_grid_quad");
        assert!(
            (res.value - std::f64::consts::PI).abs() < 0.02,
            "2D Gaussian: got {}, expected pi",
            res.value
        );
    }

    #[test]
    fn test_gauss_legendre_family() {
        // Same test with GL family
        let res = sparse_grid_quad(
            |x: &Array1<f64>| x[0] * x[1],
            &[(0.0, 1.0), (0.0, 1.0)],
            Some(SparseGridOptions {
                level: 3,
                rule_family: SparseGridRuleFamily::GaussLegendre,
            }),
        )
        .expect("sparse_grid_quad");
        assert!(
            (res.value - 0.25).abs() < 1e-10,
            "GL 2D x*y: got {}",
            res.value
        );
    }

    #[test]
    fn test_enumerate_multi_indices() {
        // d=2, q=3 => alpha in {(1,2), (2,1)}
        let mi = enumerate_multi_indices(2, 3);
        assert_eq!(mi.len(), 2);
    }

    #[test]
    fn test_binomial() {
        assert_eq!(binomial_coeff(5, 2), 10);
        assert_eq!(binomial_coeff(4, 0), 1);
        assert_eq!(binomial_coeff(4, 4), 1);
        assert_eq!(binomial_coeff(0, 0), 1);
        assert_eq!(binomial_coeff(3, 5), 0);
    }

    #[test]
    fn test_high_dim_constant() {
        // integral of 1 over [0,1]^5 = 1
        // Need level >= d for Smolyak to have at least one term
        let ranges: Vec<(f64, f64)> = (0..5).map(|_| (0.0, 1.0)).collect();
        let res = sparse_grid_quad(
            |_x: &Array1<f64>| 1.0,
            &ranges,
            Some(SparseGridOptions {
                level: 6,
                ..Default::default()
            }),
        )
        .expect("sparse_grid_quad");
        assert!(
            (res.value - 1.0).abs() < 1e-10,
            "5D constant: got {}",
            res.value
        );
    }

    #[test]
    fn test_non_unit_domain() {
        // integral of 1 over [2,5]^2 = 9
        let res = sparse_grid_quad(
            |_x: &Array1<f64>| 1.0,
            &[(2.0, 5.0), (2.0, 5.0)],
            Some(SparseGridOptions {
                level: 2,
                ..Default::default()
            }),
        )
        .expect("sparse_grid_quad");
        assert!(
            (res.value - 9.0).abs() < 1e-10,
            "non-unit domain: got {}",
            res.value
        );
    }

    #[test]
    fn test_invalid_level() {
        let res = sparse_grid_quad(
            |_x: &Array1<f64>| 1.0,
            &[(0.0, 1.0)],
            Some(SparseGridOptions {
                level: 0,
                ..Default::default()
            }),
        );
        assert!(res.is_err(), "level 0 should be invalid");
    }
}
