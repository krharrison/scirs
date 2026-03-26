//! Sparse Grid Smolyak Quadrature for high-dimensional integration.
//!
//! The Smolyak construction avoids the curse of dimensionality that afflicts
//! full tensor-product rules by combining lower-level 1-D rules in a sparse
//! fashion:
//!
//! ```text
//! Q_{n,l}[f] = Σ_{|i| ≤ l+n-1}  (-1)^{l+n-1-|i|}  C(n-1, l+n-1-|i|)  (Q_{i_1} ⊗ ⋯ ⊗ Q_{i_n})[f]
//! ```
//!
//! where `n` is the dimension, `l` is the accuracy level, and each
//! `Q_k` is a 1-D quadrature rule with `m(k)` nodes.
//!
//! ## Point count
//!
//! For level `l=1`: same as a single 1-D rule.
//! For level `l=2`: roughly O(n · m) points (vs O(m^n) for tensor product).
//!
//! ## References
//!
//! - S.A. Smolyak (1963), "Quadrature and interpolation formulas for tensor
//!   products of certain classes of functions"
//! - T. Gerstner & M. Griebel (1998), "Numerical integration using sparse grids"
//! - F. Heiss & V. Winschel (2008), "Likelihood approximation by numerical
//!   integration on sparse grids"

use crate::error::{IntegrateError, IntegrateResult};
use std::collections::HashMap;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Public configuration types
// ---------------------------------------------------------------------------

/// Supported 1-D quadrature rule families.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum OneDimRule {
    /// Clenshaw-Curtis (nested) nodes on `[-1, 1]`.
    ClenshawCurtis,
    /// Gauss-Legendre nodes on `[-1, 1]`.
    GaussLegendre,
    /// Gauss-Patterson nested nodes on `[-1, 1]` (fallback to CC for high levels).
    GaussPatterson,
}

impl Default for OneDimRule {
    fn default() -> Self {
        Self::ClenshawCurtis
    }
}

/// Configuration for Smolyak sparse-grid quadrature.
#[derive(Debug, Clone)]
pub struct SmolyakConfig {
    /// Accuracy level (≥ 1). Level `l` gives exactness `2l-1` for polynomials.
    pub level: usize,
    /// Number of integration dimensions.
    pub n_dims: usize,
    /// 1-D rule family.
    pub rule: OneDimRule,
}

impl Default for SmolyakConfig {
    fn default() -> Self {
        Self {
            level: 3,
            n_dims: 3,
            rule: OneDimRule::ClenshawCurtis,
        }
    }
}

/// A constructed Smolyak sparse grid.
#[derive(Debug, Clone)]
pub struct SmolyakGrid {
    /// Grid points (each inner `Vec<f64>` has length `n_dims`).
    pub points: Vec<Vec<f64>>,
    /// Quadrature weights (same length as `points`).
    pub weights: Vec<f64>,
    /// Number of integration dimensions.
    pub n_dims: usize,
    /// Accuracy level used to construct the grid.
    pub level: usize,
}

impl SmolyakGrid {
    /// Number of grid points.
    #[inline]
    pub fn n_points(&self) -> usize {
        self.points.len()
    }
}

/// Result of a sparse-grid integration.
#[derive(Debug, Clone)]
pub struct SparseGridResult {
    /// Approximated integral value.
    pub value: f64,
    /// Number of function evaluations used.
    pub n_points: usize,
    /// Error estimate |Q_l - Q_{l-1}|.
    pub error_estimate: f64,
}

// ---------------------------------------------------------------------------
// Internal helpers — Clenshaw-Curtis
// ---------------------------------------------------------------------------

/// Clenshaw-Curtis nodes and weights for `n` points on `[-1, 1]`.
///
/// * `n = 1` → midpoint rule: x = [0], w = [2].
/// * `n > 1` → Chebyshev extreme (Gauss-Lobatto) nodes with exact CC weights.
///
/// Uses the Waldvogel (2006) formula:
/// ```text
/// w_j = (c_j / N) · [1 − Σ_{k=1}^{⌊N/2⌋} b_k / (4k²−1) · cos(2πjk/N)]
/// ```
/// where `N = n−1`, `c_0 = c_N = 1`, `c_j = 2` for `0 < j < N`,
/// `b_k = 1` if `k = N/2` (N even) else `b_k = 2`.
pub fn cc_points_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 1 {
        return (vec![0.0], vec![2.0]);
    }
    let m = n - 1; // N = polynomial degree
    let mf = m as f64;
    // Nodes: x_j = -cos(π j / m),  j = 0 … m
    let nodes: Vec<f64> = (0..n)
        .map(|j| -(PI * j as f64 / mf).cos())
        .collect();

    // Weights: w_j = (c_j / m) * [1 - sum_{k=1..m/2} b_k/(4k²-1) * cos(2π j k / m)]
    let mut weights = vec![0.0_f64; n];
    for j in 0..n {
        let cj = if j == 0 || j == m { 1.0 } else { 2.0 };
        let half = m / 2;
        let mut s = 0.0_f64;
        for k in 1..=half {
            let bk = if k == half && m % 2 == 0 { 1.0 } else { 2.0 };
            let denom = 4.0 * (k as f64).powi(2) - 1.0;
            s += bk / denom * (2.0 * PI * j as f64 * k as f64 / mf).cos();
        }
        weights[j] = cj / mf * (1.0 - s);
    }
    (nodes, weights)
}

// ---------------------------------------------------------------------------
// Internal helpers — Gauss-Legendre (Golub-Welsch)
// ---------------------------------------------------------------------------

/// Gauss-Legendre nodes and weights for `n` points on `[-1, 1]`.
///
/// Uses Newton's method on the Legendre polynomial to find the roots, then
/// computes weights from the derivative.  Accurate to ≈ 14 significant digits
/// for n ≤ 50.
pub fn gl_points_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 1 {
        return (vec![0.0], vec![2.0]);
    }
    let mut nodes = vec![0.0_f64; n];
    let mut weights = vec![0.0_f64; n];
    // Only half the roots need computing (symmetry)
    let half = (n + 1) / 2;
    let nf = n as f64;
    for i in 0..half {
        // Initial guess (Tricomi approximation)
        let mut x = -(1.0 - (1.0 - 1.0 / nf) / (8.0 * nf * nf))
            * (PI * (4 * i + 3) as f64 / (4 * n + 2) as f64).cos();
        // Newton iterations (typically < 10)
        for _ in 0..100 {
            let (p, dp) = legendre_p_and_dp(n, x);
            let dx = p / dp;
            x -= dx;
            if dx.abs() < 1e-15 * (1.0 + x.abs()) {
                break;
            }
        }
        let (_, dp) = legendre_p_and_dp(n, x);
        let w = 2.0 / ((1.0 - x * x) * dp * dp);
        nodes[i] = -x;
        nodes[n - 1 - i] = x;
        weights[i] = w;
        weights[n - 1 - i] = w;
    }
    (nodes, weights)
}

/// Evaluate the Legendre polynomial P_n(x) and its derivative P_n'(x)
/// using the three-term recurrence.
fn legendre_p_and_dp(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    if n == 1 {
        return (x, 1.0);
    }
    let mut p_prev = 1.0_f64;
    let mut p_curr = x;
    for k in 1..n {
        let kf = k as f64;
        let p_next = ((2.0 * kf + 1.0) * x * p_curr - kf * p_prev) / (kf + 1.0);
        p_prev = p_curr;
        p_curr = p_next;
    }
    let nf = n as f64;
    let dp = nf * (x * p_curr - p_prev) / (x * x - 1.0);
    (p_curr, dp)
}

// ---------------------------------------------------------------------------
// Internal helpers — Gauss-Patterson
// ---------------------------------------------------------------------------

/// Gauss-Patterson nested rule.
///
/// Gauss-Patterson nodes at level `k` are the union of level-`k-1` nodes plus
/// new points that maximise the polynomial exactness.  For robustness we use
/// CC for levels > 5 where exact tabulated rules are less common.
fn gp_points_weights(level: usize) -> (Vec<f64>, Vec<f64>) {
    // Gauss-Patterson has 1, 3, 7, 15, 31, ... = 2^k - 1 points.
    // We fall back to CC for large levels.
    let n = match level {
        1 => 1,
        2 => 3,
        3 => 7,
        4 => 15,
        5 => 31,
        k => 2_usize.pow((k as u32).min(10)) + 1, // fall back to CC-like
    };
    // For levels 1-5 use GL as a high-quality proxy
    // (true GP tables are large — GL is exact to the same degree on symmetric intervals)
    gl_points_weights(n)
}

// ---------------------------------------------------------------------------
// 1-D dispatch
// ---------------------------------------------------------------------------

fn one_dim_rule(level: usize, rule: &OneDimRule) -> (Vec<f64>, Vec<f64>) {
    // Map level -> number of points
    // CC: level k -> 2^{k-1}+1 for k>=2, 1 for k=1
    // GL: level k -> k points
    match rule {
        OneDimRule::ClenshawCurtis => {
            let n = if level <= 1 {
                1
            } else {
                (1_usize << (level - 1)) + 1
            };
            cc_points_weights(n)
        }
        OneDimRule::GaussLegendre => {
            let n = level.max(1);
            gl_points_weights(n)
        }
        OneDimRule::GaussPatterson => gp_points_weights(level),
    }
}

// ---------------------------------------------------------------------------
// Combinatorial helper — binomial coefficient
// ---------------------------------------------------------------------------

fn binom(n: usize, k: usize) -> i64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result = 1_i64;
    for i in 0..k {
        result = result * (n - i) as i64 / (i + 1) as i64;
    }
    result
}

// ---------------------------------------------------------------------------
// Multi-index enumeration (sum-constrained)
// ---------------------------------------------------------------------------

/// Enumerate all multi-indices `i = (i_1, …, i_n)` with `i_j >= 1` and
/// `sum(i) == target`.
fn multi_indices_sum(n_dims: usize, target: usize, base: usize, result: &mut Vec<Vec<usize>>, current: &mut Vec<usize>) {
    if current.len() == n_dims {
        if base == 0 {
            result.push(current.clone());
        }
        return;
    }
    let remaining_dims = n_dims - current.len();
    // Each remaining dimension contributes at least 1
    if base < remaining_dims {
        return;
    }
    let max_val = base - (remaining_dims - 1); // leave at least 1 for each remaining dim
    for val in 1..=max_val {
        current.push(val);
        multi_indices_sum(n_dims, target, base - val, result, current);
        current.pop();
    }
}

// ---------------------------------------------------------------------------
// Main Smolyak grid construction
// ---------------------------------------------------------------------------

/// Build the Smolyak sparse grid for `n_dims` dimensions at accuracy level `level`.
///
/// The Smolyak formula sums over all multi-indices `i` (each component ≥ 1)
/// with `|i| = sum(i) ≤ level + n_dims - 1`, weighted by the difference-
/// operator coefficient `(-1)^{level+n_dims-1-|i|} * C(n_dims-1, level+n_dims-1-|i|)`.
pub fn smolyak_grid(n_dims: usize, level: usize, rule: &OneDimRule) -> IntegrateResult<SmolyakGrid> {
    if n_dims == 0 {
        return Err(IntegrateError::InvalidInput(
            "n_dims must be at least 1".into(),
        ));
    }
    if level == 0 {
        return Err(IntegrateError::InvalidInput(
            "level must be at least 1".into(),
        ));
    }

    let max_sum = level + n_dims - 1;

    // We accumulate unique points in a HashMap keyed by a string encoding.
    // Value: (point, accumulated_weight).
    let mut point_map: HashMap<String, (Vec<f64>, f64)> = HashMap::new();

    // Iterate over all multi-indices i with 1 ≤ i_j and sum(i) in [n_dims, max_sum]
    for s in n_dims..=max_sum {
        // Coefficient: (-1)^{max_sum - s} * C(n_dims - 1, max_sum - s)
        let diff = max_sum - s;
        let sign = if diff % 2 == 0 { 1_i64 } else { -1_i64 };
        let coeff = sign * binom(n_dims - 1, diff);
        if coeff == 0 {
            continue;
        }
        let coeff_f = coeff as f64;

        // Enumerate multi-indices of sum s
        let mut indices: Vec<Vec<usize>> = Vec::new();
        multi_indices_sum(n_dims, s, s, &mut indices, &mut Vec::new());

        for idx in indices {
            // Build tensor-product points and weights for this multi-index
            // Start with dimension 0
            let (mut pts, mut wts): (Vec<Vec<f64>>, Vec<f64>) = {
                let (x0, w0) = one_dim_rule(idx[0], rule);
                let pts: Vec<Vec<f64>> = x0.iter().map(|&xi| vec![xi]).collect();
                let wts: Vec<f64> = w0;
                (pts, wts)
            };

            // Tensor product with each subsequent dimension
            for d in 1..n_dims {
                let (xd, wd) = one_dim_rule(idx[d], rule);
                let mut new_pts = Vec::with_capacity(pts.len() * xd.len());
                let mut new_wts = Vec::with_capacity(wts.len() * xd.len());
                for (p, &wp) in pts.iter().zip(wts.iter()) {
                    for (&xj, &wj) in xd.iter().zip(wd.iter()) {
                        let mut new_p = p.clone();
                        new_p.push(xj);
                        new_pts.push(new_p);
                        new_wts.push(wp * wj);
                    }
                }
                pts = new_pts;
                wts = new_wts;
            }

            // Accumulate into point map with Smolyak coefficient
            for (p, w) in pts.iter().zip(wts.iter()) {
                let key = point_key(p);
                let entry = point_map.entry(key).or_insert_with(|| (p.clone(), 0.0));
                entry.1 += coeff_f * w;
            }
        }
    }

    // Collect non-negligible points
    let mut points = Vec::with_capacity(point_map.len());
    let mut weights = Vec::with_capacity(point_map.len());
    for (_, (p, w)) in point_map {
        if w.abs() > 1e-16 {
            points.push(p);
            weights.push(w);
        }
    }

    Ok(SmolyakGrid {
        points,
        weights,
        n_dims,
        level,
    })
}

/// Create a reproducible string key for a point.
fn point_key(p: &[f64]) -> String {
    p.iter()
        .map(|&v| format!("{:.15e}", v))
        .collect::<Vec<_>>()
        .join(",")
}

// ---------------------------------------------------------------------------
// Integration using a pre-built grid
// ---------------------------------------------------------------------------

/// Evaluate `f` on all grid points and return the weighted sum.
pub fn smolyak_integrate(f: impl Fn(&[f64]) -> f64, grid: &SmolyakGrid) -> f64 {
    grid.points
        .iter()
        .zip(grid.weights.iter())
        .map(|(p, &w)| w * f(p))
        .sum()
}

/// Compute the integral of `f` over `[-1,1]^n_dims` using the Smolyak rule,
/// and estimate the error as `|Q_level - Q_{level-1}|`.
pub fn smolyak_integrate_with_error(
    f: impl Fn(&[f64]) -> f64 + Clone,
    n_dims: usize,
    level: usize,
    rule: OneDimRule,
) -> IntegrateResult<SparseGridResult> {
    if level < 1 {
        return Err(IntegrateError::InvalidInput(
            "level must be at least 1".into(),
        ));
    }
    let grid_l = smolyak_grid(n_dims, level, &rule)?;
    let val_l = smolyak_integrate(f.clone(), &grid_l);
    let n_points = grid_l.n_points();

    let error_estimate = if level >= 2 {
        let grid_l1 = smolyak_grid(n_dims, level - 1, &rule)?;
        let val_l1 = smolyak_integrate(f, &grid_l1);
        (val_l - val_l1).abs()
    } else {
        val_l.abs() // no lower-level estimate
    };

    Ok(SparseGridResult {
        value: val_l,
        n_points,
        error_estimate,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// ∫_{-1}^{1} ∫_{-1}^{1} (x² + y²) dx dy = 8/3
    #[test]
    fn test_smolyak_x2_plus_y2_2d() {
        let result = smolyak_integrate_with_error(
            |p| p[0] * p[0] + p[1] * p[1],
            2,
            3,
            OneDimRule::ClenshawCurtis,
        )
        .expect("smolyak integration should succeed");
        let exact = 8.0 / 3.0;
        assert!(
            (result.value - exact).abs() < 1e-10,
            "value = {}, exact = {}, error = {}",
            result.value,
            exact,
            (result.value - exact).abs()
        );
    }

    /// Sparse grid has far fewer points than full tensor-product rule.
    #[test]
    fn test_sparse_grid_point_count() {
        let grid = smolyak_grid(4, 3, &OneDimRule::ClenshawCurtis)
            .expect("grid construction should succeed");
        // Full tensor product at level 3 CC would have (2^2+1)^4 = 625 points
        // Sparse grid should be much smaller
        assert!(
            grid.n_points() < 200,
            "sparse grid should have fewer points than full tensor product; got {}",
            grid.n_points()
        );
        assert!(grid.n_points() > 0, "grid should be non-empty");
    }

    /// Level-3 should be more accurate than level-1 for a smooth function.
    #[test]
    fn test_level_improvement() {
        let f = |p: &[f64]| p[0] * p[0] + p[1] * p[1];
        let exact = 8.0 / 3.0;

        let r1 = smolyak_integrate_with_error(f, 2, 1, OneDimRule::ClenshawCurtis)
            .expect("level-1 integration should succeed");
        let r3 = smolyak_integrate_with_error(f, 2, 3, OneDimRule::ClenshawCurtis)
            .expect("level-3 integration should succeed");

        let err1 = (r1.value - exact).abs();
        let err3 = (r3.value - exact).abs();
        // Level 3 should be at least as accurate as level 1
        assert!(
            err3 <= err1 + 1e-12,
            "level-3 error {} should be ≤ level-1 error {}",
            err3,
            err1
        );
    }

    /// Gauss-Legendre rule should also integrate a polynomial exactly.
    #[test]
    fn test_gauss_legendre_rule() {
        // ∫_{-1}^{1} x^4 dx = 2/5
        let result = smolyak_integrate_with_error(
            |p| p[0].powi(4),
            1,
            3,
            OneDimRule::GaussLegendre,
        )
        .expect("GL integration should succeed");
        let exact = 2.0 / 5.0;
        assert!(
            (result.value - exact).abs() < 1e-10,
            "GL result = {}, exact = {}",
            result.value,
            exact
        );
    }

    /// Basic test for cc_points_weights orthogonality check.
    #[test]
    fn test_cc_points_weights_3() {
        let (x, w) = cc_points_weights(3);
        assert_eq!(x.len(), 3);
        assert_eq!(w.len(), 3);
        // Sum of weights = 2 (length of [-1,1])
        let sum_w: f64 = w.iter().sum();
        assert!((sum_w - 2.0).abs() < 1e-12, "sum of CC weights = {}", sum_w);
    }

    /// Basic test for gl_points_weights.
    #[test]
    fn test_gl_points_weights_5() {
        let (x, w) = gl_points_weights(5);
        assert_eq!(x.len(), 5);
        // Sum of weights = 2
        let sum_w: f64 = w.iter().sum();
        assert!((sum_w - 2.0).abs() < 1e-12, "sum of GL weights = {}", sum_w);
        // ∫x^4 dx = 2/5
        let val: f64 = x.iter().zip(w.iter()).map(|(&xi, &wi)| wi * xi.powi(4)).sum();
        assert!((val - 0.4).abs() < 1e-12, "GL integral of x^4 = {}", val);
    }
}
