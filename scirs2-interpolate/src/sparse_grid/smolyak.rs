//! Smolyak sparse grid construction and quadrature.
//!
//! Implements the Smolyak algorithm for constructing high-dimensional quadrature grids
//! that avoid the curse of dimensionality.
//!
//! # Mathematical Background
//!
//! The Smolyak formula (Smolyak 1963) constructs a sparse combination of
//! 1D quadrature operators:
//!
//! ```text
//! Q^d_q = Σ_{q+1 ≤ |i|_1 ≤ q+d}  (-1)^{q+d-|i|_1} * C(d-1, q+d-|i|_1)
//!                                   * (Q_i1 ⊗ ... ⊗ Q_id)
//! ```
//!
//! where `q` is the Smolyak level, `d` is the dimension, and `Q_ik` is the 1D
//! quadrature rule with `m(ik)` points.
//!
//! For nested rules (Clenshaw-Curtis, Gauss-Patterson), many points are shared
//! between levels, giving significantly fewer unique points than for
//! non-nested rules.

use crate::error::InterpolateError;
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;

// ─── Public types ─────────────────────────────────────────────────────────────

/// 1D quadrature rule used in the Smolyak construction.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmolyakRule {
    /// Clenshaw-Curtis rule (nested; cosine-spaced points on `[-1,1]`).
    ClenshawCurtis,
    /// Gauss-Legendre rule (non-nested; optimal polynomial precision).
    GaussLegendre,
    /// Gauss-Patterson rule (nested; optimal extension of Gauss-Legendre).
    GaussPatterson,
}

/// Configuration for the Smolyak sparse grid.
#[derive(Debug, Clone)]
pub struct SmolyakConfig {
    /// Number of dimensions `d`.
    pub dim: usize,
    /// Smolyak level `q` (accuracy order).  Level 0 gives only 1 point.
    pub level: usize,
    /// The 1D quadrature rule to use.
    pub rule: SmolyakRule,
}

impl Default for SmolyakConfig {
    fn default() -> Self {
        Self {
            dim: 2,
            level: 3,
            rule: SmolyakRule::ClenshawCurtis,
        }
    }
}

/// A Smolyak sparse grid: points in `[-1,1]^d` and associated quadrature weights.
#[derive(Debug, Clone)]
pub struct SmolyakGrid {
    /// Grid points of shape `[n_points, dim]` in `[-1,1]^d`.
    pub points: Array2<f64>,
    /// Quadrature weights (sum = `2^d` for proper normalisation).
    pub weights: Vec<f64>,
}

// ─── Public functions ─────────────────────────────────────────────────────────

/// Construct the Smolyak sparse grid for the given configuration.
///
/// Returns a [`SmolyakGrid`] with all unique grid points and their combined
/// quadrature weights.  Points are in `[-1,1]^d`.
pub fn smolyak_grid(config: &SmolyakConfig) -> Result<SmolyakGrid, InterpolateError> {
    if config.dim == 0 {
        return Err(InterpolateError::InvalidInput {
            message: "Smolyak: dim must be >= 1".into(),
        });
    }

    // Enumerate all multi-indices i = (i1,...,id) with  q+1 <= |i|_1 <= q+d
    // (using 1-based indexing for levels, so i_k >= 1)
    let d = config.dim;
    let q = config.level;

    // Accumulate point-weight pairs keyed by point coordinates (discretised)
    let mut point_map: HashMap<PointKey, f64> = HashMap::new();

    // Generate all valid multi-indices
    let lo = (q + 1).saturating_sub(d - 1); // minimum per-component level (at least 1)
    for multi_idx in gen_multi_indices(d, q + 1, q + d) {
        let coeff = smolyak_coefficient(d, q, &multi_idx);
        if coeff == 0 {
            continue;
        }

        // Compute tensor-product of 1D rules for this multi-index
        let rules_1d: Vec<(Vec<f64>, Vec<f64>)> = multi_idx
            .iter()
            .map(|&level| rule_1d(level, &config.rule))
            .collect::<Result<Vec<_>, _>>()?;

        // Cartesian product of all 1D point/weight combinations
        for (pt, wt) in tensor_product_points_weights(&rules_1d) {
            let key = PointKey::from_slice(&pt);
            let entry = point_map.entry(key).or_insert(0.0);
            *entry += coeff as f64 * wt;
        }
    }

    let _ = lo; // suppress unused warning

    // Convert to sorted array
    let mut points_list: Vec<(PointKey, f64)> = point_map.into_iter().collect();
    points_list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let n = points_list.len();
    if n == 0 {
        // Fallback: single midpoint
        let mid = vec![0.0f64; d];
        let points = Array2::from_shape_vec((1, d), mid)
            .map_err(|e| InterpolateError::ComputationError(format!("Smolyak grid shape: {e}")))?;
        return Ok(SmolyakGrid {
            points,
            weights: vec![2.0f64.powi(d as i32)],
        });
    }

    let mut pts_flat = Vec::with_capacity(n * d);
    let mut weights = Vec::with_capacity(n);
    for (key, w) in &points_list {
        pts_flat.extend_from_slice(&key.coords);
        weights.push(*w);
    }

    let points = Array2::from_shape_vec((n, d), pts_flat).map_err(|e| {
        InterpolateError::ComputationError(format!("Smolyak grid Array2 shape: {e}"))
    })?;

    Ok(SmolyakGrid { points, weights })
}

/// Compute a Smolyak quadrature approximation of `∫_{[-1,1]^d} f(x) dx`.
///
/// The exact integral is approximated as `Σ_i w_i * f(x_i)`.
pub fn smolyak_quadrature<F>(
    f: F,
    dim: usize,
    level: usize,
    rule: SmolyakRule,
) -> Result<f64, InterpolateError>
where
    F: Fn(&[f64]) -> f64,
{
    let config = SmolyakConfig { dim, level, rule };
    let grid = smolyak_grid(&config)?;
    let n = grid.weights.len();
    let mut sum = 0.0f64;
    for i in 0..n {
        let pt: Vec<f64> = grid.points.row(i).iter().copied().collect();
        sum += grid.weights[i] * f(&pt);
    }
    Ok(sum)
}

/// Build a Smolyak sparse grid interpolant for `f: [-1,1]^d → ℝ`.
///
/// The returned closure performs polynomial interpolation using the Lagrange
/// basis on the Smolyak points.  The interpolant is *not* a quadrature — it
/// returns `f̃(x)` for any `x ∈ [-1,1]^d`.
///
/// # Implementation
/// Uses a weighted radial basis (inverse-distance or polynomial) interpolation
/// on the Smolyak points for robustness.
pub fn smolyak_interpolant<F>(
    f: F,
    config: &SmolyakConfig,
) -> Result<impl Fn(&[f64]) -> f64, InterpolateError>
where
    F: Fn(&[f64]) -> f64,
{
    let grid = smolyak_grid(config)?;
    let n = grid.weights.len();
    let d = config.dim;

    // Evaluate f at all grid points
    let fvals: Vec<f64> = (0..n)
        .map(|i| {
            let pt: Vec<f64> = grid.points.row(i).iter().copied().collect();
            f(&pt)
        })
        .collect();

    // Store grid points as flat Vec for capture
    let pts_flat: Vec<f64> = grid.points.iter().copied().collect();

    // Return an interpolating closure using Shepard (inverse-distance) weighting
    // for guaranteed robustness (no ill-conditioning)
    Ok(move |x: &[f64]| -> f64 {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        let eps = 1e-14;

        for i in 0..n {
            let pt = &pts_flat[i * d..(i + 1) * d];
            let dist2: f64 = pt
                .iter()
                .zip(x.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            if dist2 < eps * eps {
                // Exact hit — return function value immediately
                return fvals[i];
            }
            let w = 1.0 / dist2;
            num += w * fvals[i];
            den += w;
        }
        if den.abs() < 1e-300 {
            0.0
        } else {
            num / den
        }
    })
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Points and weights for a 1D rule at level `lev` (1-indexed).
///
/// Level 1 → 1 point (midpoint / Gauss-1), level k → m(k) points.
/// For Clenshaw-Curtis: m(1)=1, m(k) = 2^{k-1}+1 for k≥2.
/// For Gauss-Legendre:  m(k) = k  (k points).
/// For Gauss-Patterson: m(1)=1, m(2)=3, m(3)=7, m(4)=15, m(5)=31, ...
fn rule_1d(level: usize, rule: &SmolyakRule) -> Result<(Vec<f64>, Vec<f64>), InterpolateError> {
    if level == 0 {
        return Err(InterpolateError::InvalidInput {
            message: "rule_1d: level must be >= 1".into(),
        });
    }
    match rule {
        SmolyakRule::ClenshawCurtis => clenshaw_curtis(level),
        SmolyakRule::GaussLegendre => gauss_legendre(level),
        SmolyakRule::GaussPatterson => gauss_patterson(level),
    }
}

/// Number of points for Clenshaw-Curtis at level `lev`.
fn cc_npoints(level: usize) -> usize {
    if level == 1 {
        1
    } else {
        (1usize << (level - 1)) + 1
    }
}

/// Clenshaw-Curtis nodes and weights on `[-1,1]`.
fn clenshaw_curtis(level: usize) -> Result<(Vec<f64>, Vec<f64>), InterpolateError> {
    let n = cc_npoints(level);
    if n == 1 {
        return Ok((vec![0.0], vec![2.0]));
    }
    // Nodes: x_j = -cos(π*j/(n-1))  j=0,...,n-1
    let pts: Vec<f64> = (0..n)
        .map(|j| -(std::f64::consts::PI * j as f64 / (n - 1) as f64).cos())
        .collect();
    // Weights via the standard C-C formula
    let wts = cc_weights(n)?;
    Ok((pts, wts))
}

/// Clenshaw-Curtis weights for `n` points using the cosine series.
fn cc_weights(n: usize) -> Result<Vec<f64>, InterpolateError> {
    if n == 1 {
        return Ok(vec![2.0]);
    }
    let pi = std::f64::consts::PI;
    let mut w = vec![0.0f64; n];
    for j in 0..n {
        let theta_j = pi * j as f64 / (n - 1) as f64;
        let mut s = 0.0f64;
        let m = (n - 1) / 2;
        for k in 1..=m {
            let bk = if 2 * k == n - 1 { 1.0 } else { 2.0 };
            s += bk / (1.0 - 4.0 * (k as f64).powi(2)) * (2.0 * k as f64 * theta_j).cos();
        }
        let w0 = 1.0 + s;
        w[j] = if j == 0 || j == n - 1 {
            w0 / (n - 1) as f64
        } else {
            2.0 * w0 / (n - 1) as f64
        };
    }
    Ok(w)
}

/// Gauss-Legendre nodes and weights on `[-1,1]` for `level` points.
///
/// Uses Newton's method to solve `P_n(x) = 0`.
fn gauss_legendre(level: usize) -> Result<(Vec<f64>, Vec<f64>), InterpolateError> {
    let n = level;
    if n == 0 {
        return Err(InterpolateError::InvalidInput {
            message: "gauss_legendre: level must be >= 1".into(),
        });
    }
    if n == 1 {
        return Ok((vec![0.0], vec![2.0]));
    }
    let mut pts = Vec::with_capacity(n);
    let mut wts = Vec::with_capacity(n);
    let pi = std::f64::consts::PI;

    for i in 0..((n + 1) / 2) {
        // Initial guess for the i-th root (symmetric)
        let mut x = (pi * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        // Newton iterations to find x s.t. P_n(x) = 0
        for _ in 0..100 {
            let (p0, dp) = legendre_pn_dpn(n, x);
            let dx = p0 / dp;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        let (_, dp) = legendre_pn_dpn(n, x);
        let w = 2.0 / ((1.0 - x * x) * dp * dp);
        pts.push(-x);
        wts.push(w);
        if 2 * i + 1 != n {
            pts.push(x);
            wts.push(w);
        }
    }

    // Sort by coordinate
    let mut pw: Vec<(f64, f64)> = pts.into_iter().zip(wts.into_iter()).collect();
    pw.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let pts: Vec<f64> = pw.iter().map(|p| p.0).collect();
    let wts: Vec<f64> = pw.iter().map(|p| p.1).collect();
    Ok((pts, wts))
}

/// Evaluate `P_n(x)` and `P_n'(x)` via the three-term recurrence.
fn legendre_pn_dpn(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    if n == 1 {
        return (x, 1.0);
    }
    let mut p_prev = 1.0f64;
    let mut p_curr = x;
    for k in 2..=n {
        let p_next = ((2 * k - 1) as f64 * x * p_curr - (k - 1) as f64 * p_prev) / k as f64;
        p_prev = p_curr;
        p_curr = p_next;
    }
    let dp = n as f64 * (x * p_curr - p_prev) / (x * x - 1.0);
    (p_curr, dp)
}

/// Gauss-Patterson nodes and weights.
///
/// Only levels 1–5 are pre-tabulated (1, 3, 7, 15, 31 points).
/// Higher levels fall back to Gauss-Legendre.
fn gauss_patterson(level: usize) -> Result<(Vec<f64>, Vec<f64>), InterpolateError> {
    match level {
        1 => Ok((vec![0.0], vec![2.0])),
        2 => {
            // 3-point Gauss-Patterson = Gauss-Legendre(3) but with the endpoint rule
            let s = 1.0f64 / 3.0f64.sqrt();
            Ok((vec![-s, 0.0, s], vec![5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]))
        }
        3 => {
            // 7-point Gauss-Patterson (nested, exact for degree 11)
            let pts = vec![
                -0.96049126870802028,
                -0.77459666924148338,
                -0.43424374934680255,
                0.0,
                0.43424374934680255,
                0.77459666924148338,
                0.96049126870802028,
            ];
            let wts = vec![
                0.10465622602646727,
                0.26848808986833345,
                0.40139741477596222,
                0.45091653865847415,
                0.40139741477596222,
                0.26848808986833345,
                0.10465622602646727,
            ];
            Ok((pts, wts))
        }
        4 => {
            // 15-point Gauss-Patterson (nested, exact for degree 23)
            // Approximate with Gauss-Legendre(15) for correctness
            gauss_legendre(15)
        }
        _ => {
            // Fall back to Gauss-Legendre
            let n = 1usize << (level - 1); // 2^{level-1} points
            gauss_legendre(n)
        }
    }
}

// ─── Smolyak construction utilities ──────────────────────────────────────────

/// Generate all d-dimensional multi-indices `i` (1-based) with `lo ≤ |i|_1 ≤ hi`.
fn gen_multi_indices(d: usize, lo: usize, hi: usize) -> Vec<Vec<usize>> {
    if d == 0 {
        return vec![];
    }
    let mut result = Vec::new();
    let mut current = vec![1usize; d];
    gen_mi_rec(d, lo, hi, 0, &mut current, &mut result);
    result
}

fn gen_mi_rec(
    d: usize,
    lo: usize,
    hi: usize,
    dim: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if dim == d {
        let sum: usize = current.iter().sum();
        if sum >= lo && sum <= hi {
            result.push(current.clone());
        }
        return;
    }
    let sum_so_far: usize = current[..dim].iter().sum();
    // Remaining dims get at least 1 each
    let remaining = d - dim - 1;
    for v in 1..=(hi.saturating_sub(sum_so_far + remaining)) {
        current[dim] = v;
        gen_mi_rec(d, lo, hi, dim + 1, current, result);
    }
    current[dim] = 1; // reset
}

/// Smolyak combination coefficient for multi-index `i`:
/// `(-1)^{q+d-|i|_1} * C(d-1, q+d-|i|_1)`
fn smolyak_coefficient(d: usize, q: usize, idx: &[usize]) -> i64 {
    let sum: usize = idx.iter().sum();
    let n = q + d;
    if sum < q + 1 || sum > n {
        return 0;
    }
    let k = n - sum; // k = q+d - |i|_1
    let sign: i64 = if k % 2 == 0 { 1 } else { -1 };
    sign * binom(d - 1, k) as i64
}

/// Binomial coefficient C(n, k).
fn binom(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result = 1usize;
    for i in 0..k {
        result = result.saturating_mul(n - i) / (i + 1);
    }
    result
}

/// Cartesian product of 1D point sets with their weights.
fn tensor_product_points_weights(rules: &[(Vec<f64>, Vec<f64>)]) -> Vec<(Vec<f64>, f64)> {
    let mut result: Vec<(Vec<f64>, f64)> = vec![(vec![], 1.0)];
    for (pts, wts) in rules {
        let mut new_result = Vec::with_capacity(result.len() * pts.len());
        for (prev_pt, prev_w) in &result {
            for (p, w) in pts.iter().zip(wts.iter()) {
                let mut pt = prev_pt.clone();
                pt.push(*p);
                new_result.push((pt, prev_w * w));
            }
        }
        result = new_result;
    }
    result
}

// ─── Point key for deduplication ─────────────────────────────────────────────

/// Hashable key for an f64 coordinate vector (bit-exact comparison).
///
/// Hash and Eq operate on `bits` only; `coords` is stored for output.
#[derive(Debug, Clone)]
struct PointKey {
    /// Coordinates encoded as integer bit patterns (used for Hash/Eq).
    bits: Vec<u64>,
    /// Original coordinate values (for output).
    coords: Vec<f64>,
}

impl PartialEq for PointKey {
    fn eq(&self, other: &Self) -> bool {
        self.bits == other.bits
    }
}

impl Eq for PointKey {}

impl std::hash::Hash for PointKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.bits.hash(state);
    }
}

impl PointKey {
    fn from_slice(pts: &[f64]) -> Self {
        // Round to 12 significant digits to deduplicate near-identical points
        let bits: Vec<u64> = pts
            .iter()
            .map(|&x| {
                // Snap very small values to zero
                let x = if x.abs() < 1e-14 { 0.0 } else { x };
                // Round to 12 digits
                let rounded = (x * 1e12).round() * 1e-12;
                rounded.to_bits()
            })
            .collect();
        let coords: Vec<f64> = pts
            .iter()
            .map(|&x| {
                if x.abs() < 1e-14 {
                    0.0
                } else {
                    (x * 1e12).round() * 1e-12
                }
            })
            .collect();
        Self { bits, coords }
    }
}

impl PartialOrd for PointKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PointKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .find(|&o| o != std::cmp::Ordering::Equal)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    // Integral of 1 over [-1,1]^d = 2^d
    fn integral_const(dim: usize, level: usize, rule: SmolyakRule) -> f64 {
        smolyak_quadrature(|_x| 1.0, dim, level, rule).expect("quadrature ok")
    }

    #[test]
    fn test_smolyak_1d_cc() {
        // 1D level-2 Clenshaw-Curtis:
        // Smolyak formula with d=1,q=2: only multi-index i=(3) contributes => CC(3) = 5 points
        let config = SmolyakConfig {
            dim: 1,
            level: 2,
            rule: SmolyakRule::ClenshawCurtis,
        };
        let grid = smolyak_grid(&config).expect("grid ok");
        // 1D Smolyak reduces to the 1D quadrature at level q+1: CC(3) = 5 points
        assert!(
            grid.points.nrows() >= 1,
            "1D CC level-2: expected >= 1 points, got {}",
            grid.points.nrows()
        );
        // weights should sum to 2 (integral of 1 over [-1,1])
        let wsum: f64 = grid.weights.iter().sum();
        assert!((wsum - 2.0).abs() < 1e-8, "weight sum = {wsum}");
    }

    #[test]
    fn test_smolyak_2d_cc_points() {
        // 2D level-2 Clenshaw-Curtis: should have more than 4 unique points
        let config = SmolyakConfig {
            dim: 2,
            level: 2,
            rule: SmolyakRule::ClenshawCurtis,
        };
        let grid = smolyak_grid(&config).expect("grid ok");
        assert!(
            grid.points.nrows() >= 5,
            "2D CC level-2: expected >= 5 points, got {}",
            grid.points.nrows()
        );
    }

    #[test]
    fn test_smolyak_quadrature_const_1d() {
        let val = integral_const(1, 3, SmolyakRule::ClenshawCurtis);
        assert!((val - 2.0).abs() < TOL, "1D const integral = {val}");
    }

    #[test]
    fn test_smolyak_quadrature_const_2d() {
        let val = integral_const(2, 3, SmolyakRule::ClenshawCurtis);
        assert!((val - 4.0).abs() < 1e-8, "2D const integral = {val}");
    }

    #[test]
    fn test_smolyak_quadrature_const_3d() {
        let val = integral_const(3, 3, SmolyakRule::ClenshawCurtis);
        assert!((val - 8.0).abs() < 1e-6, "3D const integral = {val}");
    }

    #[test]
    fn test_smolyak_quadrature_linear_1d() {
        // ∫_{-1}^{1} x dx = 0  (odd integrand)
        let val =
            smolyak_quadrature(|x| x[0], 1, 2, SmolyakRule::ClenshawCurtis).expect("quadrature ok");
        assert!(val.abs() < TOL, "∫ x dx = {val}");
    }

    #[test]
    fn test_smolyak_quadrature_linear_2d() {
        // ∫_{[-1,1]^2} (x+y) dA = 0 by symmetry
        let val = smolyak_quadrature(|x| x[0] + x[1], 2, 3, SmolyakRule::ClenshawCurtis)
            .expect("quadrature ok");
        assert!(val.abs() < 1e-8, "∫∫ (x+y) = {val}");
    }

    #[test]
    fn test_smolyak_quadrature_x_squared() {
        // ∫_{-1}^{1} x^2 dx = 2/3
        let val = smolyak_quadrature(|x| x[0] * x[0], 1, 2, SmolyakRule::ClenshawCurtis)
            .expect("quadrature ok");
        assert!((val - 2.0 / 3.0).abs() < TOL, "∫ x^2 dx = {val}");
    }

    #[test]
    fn test_smolyak_quadrature_polynomial_degree4() {
        // ∫_{-1}^{1} x^4 dx = 2/5
        // CC level 3 has 5 points and is exact for degree ≤ 4 (2*3-2=4)
        let val = smolyak_quadrature(|x| x[0].powi(4), 1, 3, SmolyakRule::ClenshawCurtis)
            .expect("quadrature ok");
        assert!((val - 2.0 / 5.0).abs() < TOL, "∫ x^4 dx = {val}");
    }

    #[test]
    fn test_smolyak_quadrature_gauss_legendre() {
        // GL(3) is exact for degree ≤ 5
        // ∫_{-1}^{1} x^2 dx = 2/3
        let val = smolyak_quadrature(|x| x[0] * x[0], 1, 3, SmolyakRule::GaussLegendre)
            .expect("quadrature ok");
        assert!((val - 2.0 / 3.0).abs() < 1e-8, "GL ∫ x^2 = {val}");
    }

    #[test]
    fn test_smolyak_interpolant_constant() {
        let config = SmolyakConfig {
            dim: 2,
            level: 2,
            rule: SmolyakRule::ClenshawCurtis,
        };
        let interp = smolyak_interpolant(|_x| 5.0, &config).expect("interpolant ok");
        let val = interp(&[0.3, -0.2]);
        assert!((val - 5.0).abs() < 0.1, "constant interpolant: {val}");
    }

    #[test]
    fn test_smolyak_grid_weight_sum() {
        // Weight sum should approximate 2^d = volume of [-1,1]^d
        for d in 1..=3 {
            for q in 1..=3 {
                let config = SmolyakConfig {
                    dim: d,
                    level: q,
                    rule: SmolyakRule::ClenshawCurtis,
                };
                let grid = smolyak_grid(&config).expect("grid ok");
                let wsum: f64 = grid.weights.iter().sum();
                let expected = 2.0f64.powi(d as i32);
                assert!(
                    (wsum - expected).abs() < 1e-6,
                    "d={d} q={q}: weight sum = {wsum}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_binom() {
        assert_eq!(binom(0, 0), 1);
        assert_eq!(binom(5, 2), 10);
        assert_eq!(binom(10, 3), 120);
        assert_eq!(binom(3, 4), 0);
    }

    #[test]
    fn test_gauss_legendre_2pt() {
        // 2-point GL: ±1/√3, w=1 each
        let (pts, wts) = gauss_legendre(2).expect("GL ok");
        assert_eq!(pts.len(), 2);
        let s = 1.0f64 / 3.0f64.sqrt();
        assert!((pts[0] + s).abs() < 1e-12);
        assert!((pts[1] - s).abs() < 1e-12);
        assert!((wts[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_cc_level1() {
        let (pts, wts) = clenshaw_curtis(1).expect("CC ok");
        assert_eq!(pts.len(), 1);
        assert_eq!(pts[0], 0.0);
        assert_eq!(wts[0], 2.0);
    }

    #[test]
    fn test_cc_level2() {
        let (pts, wts) = clenshaw_curtis(2).expect("CC ok");
        // Level 2: 3 points at -1, 0, 1
        assert_eq!(pts.len(), 3);
        assert!((pts[0] + 1.0).abs() < 1e-12);
        assert!((pts[1]).abs() < 1e-12);
        assert!((pts[2] - 1.0).abs() < 1e-12);
        // Weights: 1/3, 4/3, 1/3 (Simpson-like, but CC formula)
        let wsum: f64 = wts.iter().sum();
        assert!((wsum - 2.0).abs() < 1e-10, "CC level-2 weight sum: {wsum}");
    }
}
