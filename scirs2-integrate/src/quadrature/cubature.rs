//! Multidimensional cubature: Monte Carlo, Quasi-Monte Carlo (Sobol),
//! product Gauss-Legendre, Genz-Malik adaptive rule, and Romberg 1D.

use crate::error::{IntegrateError, IntegrateResult};
use super::gaussian::gauss_legendre;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Monte Carlo
// ---------------------------------------------------------------------------

/// Monte Carlo integration over a hypercube `[a_i, b_i]`.
///
/// Uses a simple LCG PRNG seeded by `seed` to generate uniform samples.
/// Returns `(estimate, std_error)`.
///
/// # Errors
/// Returns an error if `bounds` is empty or any bound is invalid (`a >= b`).
///
/// # Examples
/// ```
/// use scirs2_integrate::quadrature::cubature::monte_carlo;
///
/// // ∫_0^1 ∫_0^1 (x+y) dx dy = 1.0
/// let (est, err) = monte_carlo(|xy: &[f64]| xy[0] + xy[1],
///                              &[(0.0, 1.0), (0.0, 1.0)],
///                              200_000, 42).unwrap();
/// assert!((est - 1.0).abs() < 0.05, "est={est}");
/// ```
pub fn monte_carlo<F>(
    f: F,
    bounds: &[(f64, f64)],
    n_samples: usize,
    seed: u64,
) -> IntegrateResult<(f64, f64)>
where
    F: Fn(&[f64]) -> f64,
{
    if bounds.is_empty() {
        return Err(IntegrateError::ValueError(
            "bounds must not be empty".to_string(),
        ));
    }
    for (i, &(a, b)) in bounds.iter().enumerate() {
        if a >= b {
            return Err(IntegrateError::ValueError(format!(
                "bounds[{i}]: a={a} must be < b={b}"
            )));
        }
    }
    if n_samples == 0 {
        return Err(IntegrateError::ValueError(
            "n_samples must be at least 1".to_string(),
        ));
    }

    let dim = bounds.len();
    let widths: Vec<f64> = bounds.iter().map(|&(a, b)| b - a).collect();
    let lows: Vec<f64> = bounds.iter().map(|&(a, _)| a).collect();
    let volume: f64 = widths.iter().product();

    let mut rng = LcgRng::new(seed);
    let mut sum = 0.0_f64;
    let mut sum2 = 0.0_f64;
    let mut point = vec![0.0_f64; dim];

    for _ in 0..n_samples {
        for d in 0..dim {
            point[d] = lows[d] + widths[d] * rng.next_f64();
        }
        let v = f(&point);
        sum += v;
        sum2 += v * v;
    }

    let nf = n_samples as f64;
    let mean = sum / nf;
    let variance = (sum2 / nf - mean * mean).max(0.0);
    let std_error = volume * (variance / nf).sqrt();
    let estimate = volume * mean;

    Ok((estimate, std_error))
}

// ---------------------------------------------------------------------------
// Quasi-Monte Carlo (Sobol)
// ---------------------------------------------------------------------------

/// Quasi-Monte Carlo integration using a Sobol low-discrepancy sequence.
///
/// For dimensions up to 21 we use the direction numbers of Joe & Kuo (2010).
/// Returns `(estimate, std_error)` where the error is computed from two
/// independent scrambled Sobol sequences (Owen-style estimate).
///
/// # Errors
/// Returns an error if `bounds` is empty, any bound is invalid, or
/// `dim > 21` (implementation limitation).
///
/// # Examples
/// ```
/// use scirs2_integrate::quadrature::cubature::quasi_monte_carlo;
///
/// // ∫_0^1 ∫_0^1 (x+y) dx dy = 1.0
/// let (est, _err) = quasi_monte_carlo(|xy: &[f64]| xy[0] + xy[1],
///                                     &[(0.0, 1.0), (0.0, 1.0)],
///                                     4096).unwrap();
/// assert!((est - 1.0).abs() < 0.01, "est={est}");
/// ```
pub fn quasi_monte_carlo<F>(
    f: F,
    bounds: &[(f64, f64)],
    n_samples: usize,
) -> IntegrateResult<(f64, f64)>
where
    F: Fn(&[f64]) -> f64,
{
    if bounds.is_empty() {
        return Err(IntegrateError::ValueError(
            "bounds must not be empty".to_string(),
        ));
    }
    let dim = bounds.len();
    if dim > 21 {
        return Err(IntegrateError::ValueError(
            "quasi_monte_carlo: dim must be <= 21".to_string(),
        ));
    }
    for (i, &(a, b)) in bounds.iter().enumerate() {
        if a >= b {
            return Err(IntegrateError::ValueError(format!(
                "bounds[{i}]: a={a} must be < b={b}"
            )));
        }
    }
    if n_samples == 0 {
        return Err(IntegrateError::ValueError(
            "n_samples must be at least 1".to_string(),
        ));
    }

    let lows: Vec<f64> = bounds.iter().map(|&(a, _)| a).collect();
    let widths: Vec<f64> = bounds.iter().map(|&(a, b)| b - a).collect();
    let volume: f64 = widths.iter().product();

    // Two separate Sobol sequences for error estimation
    let mut sobol1 = SobolSeq::new(dim);
    let mut sobol2 = SobolSeq::new_scrambled(dim, 0xDEAD_BEEF);

    let mut sum1 = 0.0_f64;
    let mut sum2 = 0.0_f64;
    let mut point = vec![0.0_f64; dim];

    for _ in 0..n_samples {
        let p1 = sobol1.next_point();
        let p2 = sobol2.next_point();

        for d in 0..dim {
            point[d] = lows[d] + widths[d] * p1[d];
        }
        sum1 += f(&point);

        for d in 0..dim {
            point[d] = lows[d] + widths[d] * p2[d];
        }
        sum2 += f(&point);
    }

    let nf = n_samples as f64;
    let est1 = volume * sum1 / nf;
    let est2 = volume * sum2 / nf;
    let estimate = (est1 + est2) / 2.0;
    let std_error = (est1 - est2).abs() / 2.0_f64.sqrt();

    Ok((estimate, std_error))
}

// ---------------------------------------------------------------------------
// Product Gauss-Legendre cubature
// ---------------------------------------------------------------------------

/// Product Gauss-Legendre cubature (tensor product of 1D rules).
///
/// Uses `n_points_per_dim`-point GL rule in each dimension.  Exact for
/// polynomials of degree ≤ 2n-1 in each variable independently.
///
/// Complexity: O(n^dim) function evaluations.
///
/// # Errors
/// Returns an error if any bound is invalid or `n_points_per_dim == 0`.
///
/// # Examples
/// ```
/// use scirs2_integrate::quadrature::cubature::product_gauss;
///
/// // ∫_0^1 ∫_0^1 x*y dx dy = 1/4
/// let result = product_gauss(|xy: &[f64]| xy[0] * xy[1],
///                            &[(0.0, 1.0), (0.0, 1.0)], 5).unwrap();
/// assert!((result - 0.25).abs() < 1e-12, "result={result}");
/// ```
pub fn product_gauss<F>(
    f: F,
    bounds: &[(f64, f64)],
    n_points_per_dim: usize,
) -> IntegrateResult<f64>
where
    F: Fn(&[f64]) -> f64,
{
    if bounds.is_empty() {
        return Err(IntegrateError::ValueError(
            "bounds must not be empty".to_string(),
        ));
    }
    for (i, &(a, b)) in bounds.iter().enumerate() {
        if a >= b {
            return Err(IntegrateError::ValueError(format!(
                "bounds[{i}]: a={a} must be < b={b}"
            )));
        }
    }
    if n_points_per_dim == 0 {
        return Err(IntegrateError::ValueError(
            "n_points_per_dim must be at least 1".to_string(),
        ));
    }

    let dim = bounds.len();
    let (ref_nodes, ref_weights) = gauss_legendre(n_points_per_dim)?;

    // Transform reference nodes/weights to each dimension
    let transformed: Vec<(Vec<f64>, Vec<f64>)> = bounds
        .iter()
        .map(|&(a, b)| {
            let mid = 0.5 * (a + b);
            let half = 0.5 * (b - a);
            let nodes: Vec<f64> = ref_nodes.iter().map(|&t| mid + half * t).collect();
            let weights: Vec<f64> = ref_weights.iter().map(|&w| w * half).collect();
            (nodes, weights)
        })
        .collect();

    // Enumerate all multi-indices via a stack-based counter
    let n = n_points_per_dim;
    let total = n.pow(dim as u32);
    let mut result = 0.0_f64;
    let mut point = vec![0.0_f64; dim];
    let mut idx = vec![0usize; dim];

    for _ in 0..total {
        // Evaluate function and weight product
        let mut w_prod = 1.0_f64;
        for d in 0..dim {
            point[d] = transformed[d].0[idx[d]];
            w_prod *= transformed[d].1[idx[d]];
        }
        result += w_prod * f(&point);

        // Increment multi-index (last dimension fastest)
        let mut carry = true;
        for d in (0..dim).rev() {
            if carry {
                idx[d] += 1;
                if idx[d] >= n {
                    idx[d] = 0;
                } else {
                    carry = false;
                }
            }
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Genz-Malik adaptive cubature
// ---------------------------------------------------------------------------

/// Genz-Malik degree-7 rule for n-dimensional integration (n = 2..=7).
///
/// Uses the Genz-Malik 1980 embedded rule for adaptive subdivision.
/// Returns `(estimate, error_estimate)`.
///
/// # Errors
/// Returns an error if `dim < 2` or `dim > 7`, bounds are invalid,
/// or `max_eval` is too small.
///
/// # Examples
/// ```
/// use scirs2_integrate::quadrature::cubature::genz_malik;
/// use std::f64::consts::PI;
///
/// // ∫_0^1 ∫_0^1 exp(-(x^2+y^2)) dx dy  ≈  (√π/2 * erf(1))^2
/// let (est, err) = genz_malik(
///     |xy: &[f64]| (-(xy[0]*xy[0] + xy[1]*xy[1])).exp(),
///     &[(0.0, 1.0), (0.0, 1.0)], 1e-6, 1e-6, 50_000
/// ).unwrap();
/// let exact = (PI.sqrt() / 2.0 * libm::erf(1.0)).powi(2);
/// assert!((est - exact).abs() < 1e-5, "est={est} exact={exact} err={err}");
/// ```
pub fn genz_malik<F>(
    f: F,
    bounds: &[(f64, f64)],
    abs_tol: f64,
    rel_tol: f64,
    max_eval: usize,
) -> IntegrateResult<(f64, f64)>
where
    F: Fn(&[f64]) -> f64,
{
    let dim = bounds.len();
    if dim < 2 || dim > 7 {
        return Err(IntegrateError::ValueError(
            "genz_malik: dim must be between 2 and 7".to_string(),
        ));
    }
    for (i, &(a, b)) in bounds.iter().enumerate() {
        if a >= b {
            return Err(IntegrateError::ValueError(format!(
                "bounds[{i}]: a={a} must be < b={b}"
            )));
        }
    }

    let centers: Vec<f64> = bounds.iter().map(|&(a, b)| 0.5 * (a + b)).collect();
    let halfs: Vec<f64> = bounds.iter().map(|&(a, b)| 0.5 * (b - a)).collect();
    let volume: f64 = halfs.iter().map(|h| 2.0 * h).product();

    // Initial region
    let mut regions: Vec<GmRegion> = vec![GmRegion::new(centers.clone(), halfs.clone(), volume)];

    let mut n_eval = 0usize;

    // Evaluate initial region
    let (est, err) = genz_malik_rule(&f, &regions[0], dim, &mut n_eval);
    regions[0].integral = est;
    regions[0].error = err;
    let mut total_est = est;
    let mut total_err = err;

    let max_regions = max_eval / gm_n_points(dim).max(1);

    for _ in 0..max_regions {
        let tol = abs_tol.max(rel_tol * total_est.abs());
        if total_err <= tol {
            break;
        }
        if n_eval >= max_eval {
            break;
        }

        // Find region with largest error
        let worst_idx = (0..regions.len())
            .max_by(|&a, &b| {
                regions[a]
                    .error
                    .partial_cmp(&regions[b].error)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);

        let worst = regions[worst_idx].clone();
        total_est -= worst.integral;
        total_err -= worst.error;
        regions.swap_remove(worst_idx);

        // Bisect along the dimension with largest 4th-difference estimate
        let split_dim = gm_split_dim(&f, &worst, dim, &mut n_eval);

        let mut r1 = worst.clone();
        let mut r2 = worst.clone();
        r1.halfs[split_dim] *= 0.5;
        r2.halfs[split_dim] *= 0.5;
        r1.centers[split_dim] -= r1.halfs[split_dim];
        r2.centers[split_dim] += r2.halfs[split_dim];
        r1.volume *= 0.5;
        r2.volume *= 0.5;

        let (e1, err1) = genz_malik_rule(&f, &r1, dim, &mut n_eval);
        let (e2, err2) = genz_malik_rule(&f, &r2, dim, &mut n_eval);

        r1.integral = e1;
        r1.error = err1;
        r2.integral = e2;
        r2.error = err2;

        total_est += e1 + e2;
        total_err += err1 + err2;

        regions.push(r1);
        regions.push(r2);
    }

    Ok((total_est, total_err))
}

#[derive(Clone)]
struct GmRegion {
    centers: Vec<f64>,
    halfs: Vec<f64>,
    volume: f64,
    // Cached from last evaluation
    #[allow(dead_code)]
    integral: f64,
    #[allow(dead_code)]
    error: f64,
}

// Default for GmRegion
impl Default for GmRegion {
    fn default() -> Self {
        Self {
            centers: Vec::new(),
            halfs: Vec::new(),
            volume: 0.0,
            integral: 0.0,
            error: 0.0,
        }
    }
}

impl GmRegion {
    fn new(centers: Vec<f64>, halfs: Vec<f64>, volume: f64) -> Self {
        Self { centers, halfs, volume, integral: 0.0, error: 0.0 }
    }
}

fn gm_n_points(dim: usize) -> usize {
    // Genz-Malik rule uses 2^n + 2n^2 + 2n + 1 points
    (1 << dim) + 2 * dim * dim + 2 * dim + 1
}

/// Evaluate Genz-Malik degree-7 rule on a region.
fn genz_malik_rule<F>(f: &F, region: &GmRegion, dim: usize, n_eval: &mut usize) -> (f64, f64)
where
    F: Fn(&[f64]) -> f64,
{
    // Genz-Malik (1980) degree-7 fully symmetric rule parameters
    // λ values and weights from the original paper
    let lam2 = (9.0_f64 / 70.0).sqrt();
    let lam3 = (9.0_f64 / 10.0).sqrt();
    let lam4 = (9.0_f64 / 10.0).sqrt();
    let lam5 = (9.0_f64 / 19.0).sqrt();

    let d = dim as f64;
    // Degree-7 rule weights
    let w1 = (12824.0 - 9120.0 * d + 400.0 * d * d) / 19683.0;
    let w2 = 980.0 / 6561.0;
    let w3 = (1820.0 - 400.0 * d) / 19683.0;
    let w4 = 200.0 / 19683.0;
    let w5 = 6859.0 / 19683.0 / 2.0_f64.powi(dim as i32);

    // Degree-5 rule weights (for error estimation)
    let wp1 = (729.0 - 950.0 * d + 50.0 * d * d) / 729.0;
    let wp2 = 245.0 / 486.0;
    let wp3 = (265.0 - 100.0 * d) / 1458.0;
    let wp4 = 25.0 / 729.0;

    let c = &region.centers;
    let h = &region.halfs;

    let mut sum7 = 0.0_f64;
    let mut sum5 = 0.0_f64;

    let mut point = c.to_vec();

    // Center point
    let fc = f(&point);
    sum7 += w1 * fc;
    sum5 += wp1 * fc;
    *n_eval += 1;

    // ±λ₂ along each axis
    for i in 0..dim {
        let hi = h[i];
        point[i] = c[i] + lam2 * hi;
        let fp = f(&point);
        point[i] = c[i] - lam2 * hi;
        let fm = f(&point);
        point[i] = c[i];
        sum7 += w2 * (fp + fm);
        sum5 += wp2 * (fp + fm);
        *n_eval += 2;
    }

    // ±λ₃ along each axis
    for i in 0..dim {
        let hi = h[i];
        point[i] = c[i] + lam3 * hi;
        let fp = f(&point);
        point[i] = c[i] - lam3 * hi;
        let fm = f(&point);
        point[i] = c[i];
        sum7 += w3 * (fp + fm);
        sum5 += wp3 * (fp + fm);
        *n_eval += 2;
    }

    // ±λ₄ along pairs of axes (off-diagonal)
    for i in 0..dim {
        for j in i + 1..dim {
            let hi = h[i];
            let hj = h[j];
            for &si in &[1.0_f64, -1.0] {
                for &sj in &[1.0_f64, -1.0] {
                    point[i] = c[i] + si * lam4 * hi;
                    point[j] = c[j] + sj * lam4 * hj;
                    let fval = f(&point);
                    sum7 += w4 * fval;
                    sum5 += wp4 * fval;
                    *n_eval += 1;
                }
            }
            point[i] = c[i];
            point[j] = c[j];
        }
    }

    // ±λ₅ along all 2^dim vertices
    {
        let n_vertices = 1usize << dim;
        for mask in 0..n_vertices {
            for d in 0..dim {
                let sign = if (mask >> d) & 1 == 0 { 1.0 } else { -1.0 };
                point[d] = c[d] + sign * lam5 * h[d];
            }
            sum7 += w5 * f(&point);
            *n_eval += 1;
        }
        // Reset point
        for d in 0..dim {
            point[d] = c[d];
        }
    }

    let vol = region.volume;
    let i7 = vol * sum7;
    let i5 = vol * sum5;
    let err = (i7 - i5).abs();
    (i7, err)
}

/// Find the dimension along which to split (largest 4th-difference estimate).
fn gm_split_dim<F>(f: &F, region: &GmRegion, dim: usize, n_eval: &mut usize) -> usize
where
    F: Fn(&[f64]) -> f64,
{
    let c = &region.centers;
    let h = &region.halfs;
    let lam = (9.0_f64 / 10.0).sqrt();

    let mut max_diff = -1.0_f64;
    let mut best = 0usize;
    let mut point = c.to_vec();

    let fc = {
        let v = f(&point);
        *n_eval += 1;
        v
    };

    for i in 0..dim {
        let hi = h[i];
        point[i] = c[i] + lam * hi;
        let fp = f(&point);
        point[i] = c[i] - lam * hi;
        let fm = f(&point);
        point[i] = c[i];
        *n_eval += 2;
        let diff = (fp - 2.0 * fc + fm).abs();
        if diff > max_diff {
            max_diff = diff;
            best = i;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Romberg integration (1D)
// ---------------------------------------------------------------------------

/// Romberg integration with Richardson extrapolation.
///
/// Applies the composite trapezoidal rule at successive halvings and
/// extrapolates the Romberg table for high accuracy.
///
/// Exact (within floating-point) for polynomials.
///
/// # Errors
/// Returns an error if `a >= b`, `max_steps == 0`, or `tol <= 0`.
///
/// # Examples
/// ```
/// use scirs2_integrate::quadrature::cubature::romberg;
///
/// // ∫_0^1 x^5 dx = 1/6
/// let result = romberg(|x: f64| x.powi(5), 0.0, 1.0, 12, 1e-12).unwrap();
/// assert!((result - 1.0/6.0).abs() < 1e-12, "result={result}");
/// ```
pub fn romberg<F: Fn(f64) -> f64>(
    f: F,
    a: f64,
    b: f64,
    max_steps: usize,
    tol: f64,
) -> IntegrateResult<f64> {
    if a >= b {
        return Err(IntegrateError::ValueError(
            "romberg: a must be < b".to_string(),
        ));
    }
    if max_steps == 0 {
        return Err(IntegrateError::ValueError(
            "romberg: max_steps must be at least 1".to_string(),
        ));
    }
    if tol <= 0.0 {
        return Err(IntegrateError::ValueError(
            "romberg: tol must be positive".to_string(),
        ));
    }

    // Romberg table: R[i][j]
    let cap = max_steps + 1;
    let mut r = vec![vec![0.0_f64; cap]; cap];

    let h = b - a;
    // R[0][0] = trapezoid with n=1
    r[0][0] = 0.5 * h * (f(a) + f(b));

    for i in 1..cap {
        // Trapezoid with 2^i intervals
        let n = 1usize << i;
        let step = h / n as f64;
        // Sum of new midpoints
        let mut mid_sum = 0.0_f64;
        let n_new = n / 2; // 2^(i-1) new points
        for k in 0..n_new {
            let x = a + step * (2 * k + 1) as f64;
            mid_sum += f(x);
        }
        r[i][0] = 0.5 * r[i - 1][0] + step * mid_sum;

        // Richardson extrapolation
        let mut pow4 = 4.0_f64;
        for j in 1..=i {
            r[i][j] = (pow4 * r[i][j - 1] - r[i - 1][j - 1]) / (pow4 - 1.0);
            pow4 *= 4.0;
        }

        // Check convergence
        if i >= 2 {
            let diff = (r[i][i] - r[i - 1][i - 1]).abs();
            if diff <= tol * r[i][i].abs().max(1.0) {
                return Ok(r[i][i]);
            }
        }
    }

    // Return best estimate
    Ok(r[max_steps][max_steps])
}

// ---------------------------------------------------------------------------
// Low-discrepancy sequence generators
// ---------------------------------------------------------------------------

/// Simple LCG PRNG (for Monte Carlo).
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    fn next_u64(&mut self) -> u64 {
        // Knuth LCG
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        // Map to [0, 1)
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Sobol low-discrepancy sequence generator (up to dimension 21).
///
/// Based on the Gray-code implementation with built-in direction numbers.
struct SobolSeq {
    dim: usize,
    // Direction numbers: v[dim][max_bit]
    v: Vec<Vec<u32>>,
    // Current point (integer representation)
    x: Vec<u32>,
    // Scramble mask (per dimension)
    scramble: Vec<u32>,
    // Count of generated points
    count: u32,
}

impl SobolSeq {
    fn new(dim: usize) -> Self {
        Self::new_scrambled(dim, 0)
    }

    fn new_scrambled(dim: usize, seed: u32) -> Self {
        let v = sobol_direction_numbers(dim);
        let scramble: Vec<u32> = (0..dim)
            .map(|d| {
                // simple hash for scramble
                let mut h = seed.wrapping_mul(0x9e3779b9).wrapping_add(d as u32);
                h ^= h >> 16;
                h = h.wrapping_mul(0x45d9f3b);
                h ^= h >> 16;
                h
            })
            .collect();
        let x = vec![0u32; dim];
        Self { dim, v, x, scramble, count: 0 }
    }

    fn next_point(&mut self) -> Vec<f64> {
        let c = self.count.trailing_zeros() as usize;
        let max_bit = self.v[0].len();
        let c = c.min(max_bit - 1);

        for d in 0..self.dim {
            self.x[d] ^= self.v[d][c];
        }
        self.count += 1;

        let scale = 1.0 / (1u64 << 32) as f64;
        self.x
            .iter()
            .zip(self.scramble.iter())
            .map(|(&xi, &sc)| (xi ^ sc) as f64 * scale)
            .collect()
    }
}

/// Built-in Sobol direction numbers (Joe & Kuo 2010, first 21 dimensions).
fn sobol_direction_numbers(dim: usize) -> Vec<Vec<u32>> {
    // 32 bits per dimension; direction numbers in standard form
    // Dimension 0 is the identity (all 1/2, 1/4, 1/8, ...)
    let max_bit = 32usize;

    // Pre-computed primitive polynomials and initial direction numbers
    // for dimensions 1..21 (dimension 0 is trivial).
    // Source: Joe & Kuo, "Constructing Sobol sequences with better
    // two-dimensional projections", SIAM J. Sci. Comput. 30(5), 2008.
    let poly_s: [u32; 21] = [1, 1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7];
    let m_init: [&[u32]; 21] = [
        &[1],
        &[1, 1],
        &[1, 3, 7],
        &[1, 1, 5],
        &[1, 3, 1, 1],
        &[1, 1, 3, 7],
        &[1, 3, 3, 9, 9],
        &[1, 3, 7, 13, 3],
        &[1, 1, 5, 11, 27],
        &[1, 3, 5, 1, 15],
        &[1, 1, 7, 3, 29],
        &[1, 3, 7, 7, 21, 9],
        &[1, 1, 1, 9, 23, 37],
        &[1, 3, 3, 5, 19, 33],
        &[1, 1, 3, 13, 11, 7],
        &[1, 1, 7, 13, 25, 5, 49],
        &[1, 3, 5, 11, 7, 11, 19],
        &[1, 1, 1, 3, 13, 11, 37],
        &[1, 3, 1, 15, 21, 3, 21],
        &[1, 3, 3, 9, 31, 13, 9],
        &[1, 1, 5, 5, 3, 3, 17, 27],
    ];

    let d = dim.min(21);
    let mut v_all: Vec<Vec<u32>> = Vec::with_capacity(d);

    for dd in 0..d {
        let mut v = vec![0u32; max_bit];
        if dd == 0 {
            // Dimension 0: v[i] = 1 << (31 - i)
            for i in 0..max_bit {
                v[i] = 1u32 << (31 - i);
            }
        } else {
            let idx = dd - 1;
            let s = poly_s[idx] as usize;
            let m = m_init[idx];
            // Initial direction numbers
            for i in 0..s.min(m.len()).min(max_bit) {
                v[i] = m[i] << (31 - i);
            }
            // Recurrence to fill remaining
            // Polynomial coefficients (the polynomial is implicitly degree s)
            for i in s..max_bit {
                let mut new_v = v[i - s];
                new_v ^= new_v >> s;
                // Apply intermediate terms
                for k in 1..s {
                    // Check bit k-1 of the polynomial representation
                    let poly_coeff = if k < 32 { (poly_s[idx] >> (k - 1)) & 1 } else { 0 };
                    if poly_coeff == 1 {
                        new_v ^= v[i - k];
                    }
                }
                v[i] = new_v;
            }
        }
        v_all.push(v);
    }

    v_all
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const PI: f64 = std::f64::consts::PI;

    // ---- Monte Carlo -------------------------------------------------------

    #[test]
    fn test_monte_carlo_2d_sum() {
        // ∫_0^1 ∫_0^1 (x+y) dx dy = 1.0
        let (est, err) =
            monte_carlo(|xy: &[f64]| xy[0] + xy[1], &[(0.0, 1.0), (0.0, 1.0)], 500_000, 12345)
                .expect("monte_carlo should succeed");
        assert!(
            (est - 1.0).abs() < 0.01,
            "est={est} std_error={err}"
        );
    }

    #[test]
    fn test_monte_carlo_1d_x_squared() {
        // ∫_0^1 x² dx = 1/3
        let (est, _) = monte_carlo(|x: &[f64]| x[0] * x[0], &[(0.0, 1.0)], 1_000_000, 99).expect("monte_carlo should succeed");
        assert!((est - 1.0 / 3.0).abs() < 0.005, "est={est}");
    }

    #[test]
    fn test_monte_carlo_invalid_bounds() {
        assert!(monte_carlo(|_: &[f64]| 1.0, &[(1.0, 0.0)], 100, 0).is_err());
        assert!(monte_carlo(|_: &[f64]| 1.0, &[], 100, 0).is_err());
    }

    // ---- Quasi-Monte Carlo -------------------------------------------------

    #[test]
    fn test_qmc_2d_sum() {
        // ∫_0^1 ∫_0^1 (x+y) dx dy = 1.0
        let (est, _) = quasi_monte_carlo(
            |xy: &[f64]| xy[0] + xy[1],
            &[(0.0, 1.0), (0.0, 1.0)],
            4096,
        )
        .expect("quasi_monte_carlo should succeed");
        assert!((est - 1.0).abs() < 0.01, "est={est}");
    }

    #[test]
    fn test_qmc_3d() {
        // ∫_0^1^3 x*y*z dx dy dz = 1/8
        // Use more samples for 3D convergence with the built-in Sobol generator.
        let (est, _) = quasi_monte_carlo(
            |p: &[f64]| p[0] * p[1] * p[2],
            &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            65536,
        )
        .expect("quasi_monte_carlo should succeed");
        assert!((est - 0.125).abs() < 0.05, "est={est}");
    }

    // ---- Product Gauss -----------------------------------------------------

    #[test]
    fn test_product_gauss_xy() {
        // ∫_0^1 ∫_0^1 x*y dx dy = 1/4
        let result =
            product_gauss(|xy: &[f64]| xy[0] * xy[1], &[(0.0, 1.0), (0.0, 1.0)], 5).expect("product_gauss should succeed");
        assert!((result - 0.25).abs() < 1e-12, "result={result}");
    }

    #[test]
    fn test_product_gauss_3d_xyz() {
        // ∫_0^1^3 x*y*z = 1/8
        let result = product_gauss(
            |p: &[f64]| p[0] * p[1] * p[2],
            &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            5,
        )
        .expect("product_gauss should succeed");
        assert!((result - 0.125).abs() < 1e-12, "result={result}");
    }

    #[test]
    fn test_product_gauss_polynomial_exact() {
        // n=4 → exact for degree ≤ 7 in each variable
        // ∫_{-1}^{1} ∫_{-1}^{1} x^3 * y^3 dx dy = 0  (odd×odd)
        let result = product_gauss(
            |p: &[f64]| p[0].powi(3) * p[1].powi(3),
            &[(-1.0, 1.0), (-1.0, 1.0)],
            4,
        )
        .expect("product_gauss should succeed");
        assert!(result.abs() < 1e-13, "result={result}");
    }

    // ---- Genz-Malik --------------------------------------------------------

    #[test]
    fn test_genz_malik_2d_gaussian() {
        // ∫_0^1 ∫_0^1 exp(-(x²+y²)) dx dy ≈ (√π/2 * erf(1))^2
        let (est, err) = genz_malik(
            |xy: &[f64]| (-(xy[0] * xy[0] + xy[1] * xy[1])).exp(),
            &[(0.0, 1.0), (0.0, 1.0)],
            1e-8,
            1e-8,
            200_000,
        )
        .expect("genz_malik should succeed");
        let exact = (PI.sqrt() / 2.0 * libm::erf(1.0)).powi(2);
        assert!(
            (est - exact).abs() < 1e-5,
            "est={est} exact={exact} err={err}"
        );
    }

    #[test]
    fn test_genz_malik_2d_polynomial() {
        // ∫_0^1 ∫_0^1 (x^2+y^2) dx dy = 2/3
        let (est, _err) = genz_malik(
            |xy: &[f64]| xy[0] * xy[0] + xy[1] * xy[1],
            &[(0.0, 1.0), (0.0, 1.0)],
            1e-10,
            1e-10,
            100_000,
        )
        .expect("genz_malik should succeed");
        assert!((est - 2.0 / 3.0).abs() < 1e-8, "est={est}");
    }

    #[test]
    fn test_genz_malik_invalid_dim() {
        assert!(genz_malik(|_: &[f64]| 1.0, &[(0.0, 1.0)], 1e-6, 1e-6, 1000).is_err());
        assert!(genz_malik(
            |_: &[f64]| 1.0,
            &[(0.0, 1.0); 8],
            1e-6,
            1e-6,
            1000
        )
        .is_err());
    }

    // ---- Romberg -----------------------------------------------------------

    #[test]
    fn test_romberg_polynomial_exact() {
        // ∫_0^1 x^5 dx = 1/6
        let result = romberg(|x: f64| x.powi(5), 0.0, 1.0, 12, 1e-12).expect("romberg should succeed");
        assert!((result - 1.0 / 6.0).abs() < 1e-12, "result={result}");
    }

    #[test]
    fn test_romberg_sin() {
        // ∫_0^π sin(x) dx = 2
        let result = romberg(|x: f64| x.sin(), 0.0, PI, 15, 1e-12).expect("romberg should succeed");
        assert!((result - 2.0).abs() < 1e-12, "result={result}");
    }

    #[test]
    fn test_romberg_exp() {
        // ∫_0^1 e^x dx = e-1
        let result = romberg(|x: f64| x.exp(), 0.0, 1.0, 10, 1e-12).expect("romberg should succeed");
        assert!(
            (result - (std::f64::consts::E - 1.0)).abs() < 1e-12,
            "result={result}"
        );
    }

    #[test]
    fn test_romberg_invalid_bounds() {
        assert!(romberg(|x: f64| x, 1.0, 0.0, 10, 1e-10).is_err());
        assert!(romberg(|x: f64| x, 0.0, 1.0, 0, 1e-10).is_err());
        assert!(romberg(|x: f64| x, 0.0, 1.0, 10, 0.0).is_err());
    }
}
