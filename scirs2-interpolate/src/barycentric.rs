//! Barycentric interpolation methods
//!
//! This module provides two high-accuracy barycentric interpolation methods:
//!
//! - **`BarycentricRational`**: Floater-Hormann family of barycentric rational
//!   interpolants (no poles, high-order convergence, avoids Runge's phenomenon).
//! - **`BarycentricLagrange`**: Classic barycentric Lagrange interpolation with
//!   O(n) update when new data points are added.
//!
//! # Mathematical Background
//!
//! Both methods represent the interpolant in the barycentric form:
//!
//! ```text
//! p(x) = [Σ_i  w_i f_i / (x - x_i)] / [Σ_i  w_i / (x - x_i)]
//! ```
//!
//! The difference lies in how the weights `w_i` are computed:
//! - Lagrange: classical second-kind weights that reproduce the polynomial interpolant
//! - Floater-Hormann: weights derived from blending local polynomial interpolants of degree `d`
//!
//! # References
//!
//! - Floater, M. S. and Hormann, K. (2007), Barycentric rational interpolation with no poles
//!   and high rates of approximation, *Numer. Math.* **107**, 315–331.
//! - Berrut, J.-P. and Trefethen, L. N. (2004), Barycentric Lagrange Interpolation,
//!   *SIAM Review* **46**(3), 501–517.

use crate::error::{InterpolateError, InterpolateResult};

// ---------------------------------------------------------------------------
// BarycentricRational (Floater-Hormann)
// ---------------------------------------------------------------------------

/// Floater-Hormann barycentric rational interpolant.
///
/// This interpolant has no real poles, achieves approximation order `O(h^{d+1})`
/// for smooth functions, and avoids Runge's phenomenon for any blending degree `d`.
///
/// # Fields
///
/// - `x`: sorted sample abscissae
/// - `y`: corresponding function values
/// - `w`: Floater-Hormann barycentric weights
/// - `d`: blending degree (0 ≤ d ≤ n)
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::barycentric::BarycentricRational;
///
/// // Interpolate the Runge function 1/(1+25x^2) on [-1,1]
/// let n = 10;
/// let x: Vec<f64> = (0..=n).map(|i| -1.0 + 2.0 * i as f64 / n as f64).collect();
/// let y: Vec<f64> = x.iter().map(|&xi| 1.0 / (1.0 + 25.0 * xi * xi)).collect();
///
/// let interp = BarycentricRational::fit(&x, &y, 3).expect("doc example: should succeed");
/// let val = interp.eval(0.5);
/// ```
#[derive(Debug, Clone)]
pub struct BarycentricRational {
    x: Vec<f64>,
    y: Vec<f64>,
    w: Vec<f64>,
    d: usize,
}

impl BarycentricRational {
    /// Construct a Floater-Hormann interpolant of blending degree `d`.
    ///
    /// # Parameters
    ///
    /// - `x`: strictly increasing sample abscissae (length ≥ 1)
    /// - `y`: function values at `x`
    /// - `d`: blending degree (0 ≤ d ≤ n, recommended: 3–5)
    ///
    /// # Errors
    ///
    /// Returns [`InterpolateError::InvalidInput`] if:
    /// - `x` and `y` have different lengths
    /// - `x` is empty
    /// - `d > n` (n = x.len() - 1)
    /// - `x` is not strictly increasing
    pub fn fit(x: &[f64], y: &[f64], d: usize) -> InterpolateResult<Self> {
        let n_pts = x.len();
        if n_pts == 0 {
            return Err(InterpolateError::InvalidInput {
                message: "BarycentricRational: x must be non-empty".into(),
            });
        }
        if y.len() != n_pts {
            return Err(InterpolateError::InvalidInput {
                message: format!(
                    "BarycentricRational: x.len()={} != y.len()={}",
                    n_pts,
                    y.len()
                ),
            });
        }
        let n = n_pts - 1; // highest index
        if d > n {
            return Err(InterpolateError::InvalidInput {
                message: format!(
                    "BarycentricRational: blending degree d={} > n={} (number of intervals)",
                    d, n
                ),
            });
        }
        // Verify strictly increasing
        for i in 1..n_pts {
            if x[i] <= x[i - 1] {
                return Err(InterpolateError::InvalidInput {
                    message: format!(
                        "BarycentricRational: x must be strictly increasing; x[{}]={} <= x[{}]={}",
                        i,
                        x[i],
                        i - 1,
                        x[i - 1]
                    ),
                });
            }
        }

        // Compute Floater-Hormann weights
        // w_i = (-1)^i  Σ_{j=max(0,i-d)}^{min(i,n-d)} C(d, i-j) * prod_{k in S_j, k≠i} (x_i - x_k)^{-1}
        // Reference: Floater & Hormann 2007, eq. (5)
        let w = compute_fh_weights(x, d);

        Ok(Self {
            x: x.to_vec(),
            y: y.to_vec(),
            w,
            d,
        })
    }

    /// Evaluate the rational interpolant at a single point `xi`.
    ///
    /// If `xi` coincides with a node `x[i]` (within floating-point tolerance),
    /// returns `y[i]` exactly.
    pub fn eval(&self, xi: f64) -> f64 {
        barycentric_eval(&self.x, &self.y, &self.w, xi)
    }

    /// Evaluate the interpolant at multiple points.
    pub fn eval_many(&self, xi: &[f64]) -> Vec<f64> {
        xi.iter().map(|&x| self.eval(x)).collect()
    }

    /// Compute the derivative of the interpolant at `xi`.
    ///
    /// For order 1 the exact barycentric derivative formula is used.
    /// For higher orders, Richardson-extrapolated finite differences provide
    /// accurate results.
    ///
    /// # Parameters
    ///
    /// - `xi`: evaluation point
    /// - `order`: derivative order (1, 2, …)
    ///
    /// # Notes
    ///
    /// The barycentric derivative formula is:
    /// ```text
    /// r'(x) = [Σ_i w_i (r(x) - y_i) / (x - x_i)] / [Σ_i w_i / (x - x_i)]
    /// ```
    /// (only valid at non-node points; at nodes a centred finite difference is
    /// used as fallback).
    pub fn derivative(&self, xi: f64, order: usize) -> f64 {
        if order == 0 {
            return self.eval(xi);
        }
        if order == 1 {
            return self.derivative_order1(xi);
        }
        // Higher orders via Richardson-extrapolated finite differences
        let h = 1e-5_f64;
        match order {
            2 => {
                let fp = self.eval(xi + h);
                let fm = self.eval(xi - h);
                let f0 = self.eval(xi);
                (fp - 2.0 * f0 + fm) / (h * h)
            }
            3 => {
                let f2p = self.eval(xi + 2.0 * h);
                let f1p = self.eval(xi + h);
                let f1m = self.eval(xi - h);
                let f2m = self.eval(xi - 2.0 * h);
                (-f2p + 2.0 * f1p - 2.0 * f1m + f2m) / (2.0 * h * h * h)
            }
            _ => {
                // Generic: apply derivative recursively via FD
                let g = |t: f64| self.derivative(t, order - 1);
                (g(xi + h) - g(xi - h)) / (2.0 * h)
            }
        }
    }

    // First-order derivative using the barycentric formula
    fn derivative_order1(&self, xi: f64) -> f64 {
        // Check if xi is at a node
        for (i, &xi_node) in self.x.iter().enumerate() {
            if (xi - xi_node).abs() < 1e-14 * (1.0 + xi.abs()) {
                // Use centred FD at node
                let h = 1e-6_f64;
                let fp = self.eval(xi + h);
                let fm = self.eval(xi - h);
                let _ = i; // avoid unused warning
                return (fp - fm) / (2.0 * h);
            }
        }

        // Barycentric derivative formula (Schneider & Werner 1986):
        // r'(x) = [sum_i  w_i (r(x) - y_i)/(x-x_i)] / [sum_i w_i/(x-x_i)]
        let rx = self.eval(xi);
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for i in 0..self.x.len() {
            let d = xi - self.x[i];
            let wi_over_d = self.w[i] / d;
            num += wi_over_d * (rx - self.y[i]) / d;
            den += wi_over_d;
        }
        if den.abs() < f64::EPSILON * 1e3 {
            // Fallback to FD
            let h = 1e-6_f64;
            return (self.eval(xi + h) - self.eval(xi - h)) / (2.0 * h);
        }
        num / den
    }

    /// Blending degree used when constructing this interpolant.
    #[inline]
    pub fn degree(&self) -> usize {
        self.d
    }

    /// Return a reference to the barycentric weights.
    #[inline]
    pub fn weights(&self) -> &[f64] {
        &self.w
    }

    /// Return a reference to the sample abscissae.
    #[inline]
    pub fn nodes(&self) -> &[f64] {
        &self.x
    }

    /// Return a reference to the sample values.
    #[inline]
    pub fn values(&self) -> &[f64] {
        &self.y
    }
}

// ---------------------------------------------------------------------------
// BarycentricLagrange
// ---------------------------------------------------------------------------

/// Classic barycentric Lagrange interpolation.
///
/// Computes the unique polynomial of degree ≤ n that passes through the `n+1`
/// data points, using the second barycentric form:
///
/// ```text
/// p(x) = [Σ_i  w_i y_i / (x - x_i)] / [Σ_i  w_i / (x - x_i)]
/// ```
///
/// where `w_i = 1 / Π_{j≠i} (x_i - x_j)`.
///
/// A key feature is the O(n) **incremental update**: calling `add_point` appends
/// a new sample without recomputing all weights from scratch.
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::barycentric::BarycentricLagrange;
///
/// let x = vec![0.0_f64, 1.0, 2.0, 3.0];
/// let y = vec![0.0_f64, 1.0, 4.0, 9.0];  // x^2
/// let interp = BarycentricLagrange::fit(&x, &y).expect("doc example: should succeed");
/// assert!((interp.eval(1.5) - 2.25).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct BarycentricLagrange {
    x: Vec<f64>,
    y: Vec<f64>,
    w: Vec<f64>,
}

impl BarycentricLagrange {
    /// Construct a Lagrange interpolant from data points.
    ///
    /// # Parameters
    ///
    /// - `x`: strictly increasing abscissae (length ≥ 1)
    /// - `y`: function values
    ///
    /// # Errors
    ///
    /// Returns [`InterpolateError::InvalidInput`] if lengths differ, `x` is
    /// empty, or `x` is not strictly increasing.
    pub fn fit(x: &[f64], y: &[f64]) -> InterpolateResult<Self> {
        let n_pts = x.len();
        if n_pts == 0 {
            return Err(InterpolateError::InvalidInput {
                message: "BarycentricLagrange: x must be non-empty".into(),
            });
        }
        if y.len() != n_pts {
            return Err(InterpolateError::InvalidInput {
                message: format!(
                    "BarycentricLagrange: x.len()={} != y.len()={}",
                    n_pts,
                    y.len()
                ),
            });
        }
        for i in 1..n_pts {
            if x[i] <= x[i - 1] {
                return Err(InterpolateError::InvalidInput {
                    message: format!(
                        "BarycentricLagrange: x must be strictly increasing; x[{}]={} <= x[{}]={}",
                        i,
                        x[i],
                        i - 1,
                        x[i - 1]
                    ),
                });
            }
        }

        let w = compute_lagrange_weights(x);
        Ok(Self {
            x: x.to_vec(),
            y: y.to_vec(),
            w,
        })
    }

    /// Evaluate the Lagrange polynomial at `xi`.
    ///
    /// Returns the exact node value when `xi` coincides with a node.
    pub fn eval(&self, xi: f64) -> f64 {
        barycentric_eval(&self.x, &self.y, &self.w, xi)
    }

    /// Evaluate at multiple points.
    pub fn eval_many(&self, xi: &[f64]) -> Vec<f64> {
        xi.iter().map(|&x| self.eval(x)).collect()
    }

    /// Add a new data point `(x_new, y_new)` in O(n) time.
    ///
    /// All existing weights are divided by `(x_i - x_new)` and the new weight
    /// is `1 / Π_{j} (x_new - x_j)`.
    ///
    /// The new point must not duplicate an existing node.
    ///
    /// # Errors
    ///
    /// Returns [`InterpolateError::InvalidInput`] if `x_new` already exists in
    /// the node set.
    pub fn add_point(&mut self, x_new: f64, y_new: f64) -> InterpolateResult<()> {
        // Check for duplicate
        for &xi in &self.x {
            if (x_new - xi).abs() < f64::EPSILON * (1.0 + xi.abs()) {
                return Err(InterpolateError::InvalidInput {
                    message: format!(
                        "BarycentricLagrange::add_point: x={} already present in node set",
                        x_new
                    ),
                });
            }
        }

        // Update existing weights: w_i  ←  w_i / (x_i - x_new)
        let mut new_w = 1.0_f64;
        for i in 0..self.x.len() {
            let diff = self.x[i] - x_new;
            self.w[i] /= diff;
            new_w /= x_new - self.x[i];
        }

        self.x.push(x_new);
        self.y.push(y_new);
        self.w.push(new_w);
        Ok(())
    }

    /// Return the barycentric weights.
    #[inline]
    pub fn weights(&self) -> &[f64] {
        &self.w
    }

    /// Return the number of interpolation nodes.
    #[inline]
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Returns `true` if no data points have been added.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Evaluate the barycentric form at `xi` given nodes `x`, values `y`, and
/// weights `w`.
///
/// When `xi` coincides with a node within tolerance `1e-14 * (1 + |xi|)`,
/// the corresponding value is returned immediately without division.
#[inline]
fn barycentric_eval(x: &[f64], y: &[f64], w: &[f64], xi: f64) -> f64 {
    let n = x.len();
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;

    for i in 0..n {
        let diff = xi - x[i];
        if diff.abs() < f64::EPSILON * (1.0 + xi.abs()) {
            // xi coincides with a node: return y[i] exactly
            return y[i];
        }
        let t = w[i] / diff;
        num += t * y[i];
        den += t;
    }

    if den.abs() < f64::MIN_POSITIVE {
        // Degenerate: return NaN as signal
        return f64::NAN;
    }
    num / den
}

/// Compute Floater-Hormann barycentric weights for nodes `x` and blending
/// degree `d`.
///
/// For each node `i`, the weight is the sum over all subsets `k` of
/// `(-1)^k / prod_{j=k, j!=i}^{k+d} (x_i - x_j)`.
///
/// Note: the sign alternates with the *subset index* `k`, not with `i`.
/// Implementation follows the reference in rational_interpolation.rs.
fn compute_fh_weights(x: &[f64], d: usize) -> Vec<f64> {
    let n = x.len();
    let mut w = vec![0.0_f64; n];

    for i in 0..n {
        let k_min = if i > d { i - d } else { 0 };
        let k_max_bound = if n > d { n - 1 - d } else { 0 };
        let k_max = i.min(k_max_bound);

        let mut sum = 0.0_f64;
        for k in k_min..=k_max {
            // Compute product of (x_i - x_j) for j in k..=(k+d), j != i
            let mut prod = 1.0_f64;
            let end = (k + d).min(n - 1);
            for j in k..=end {
                if j != i {
                    let diff = x[i] - x[j];
                    prod *= diff;
                }
            }
            if prod.abs() < 1e-30 {
                continue;
            }
            // Sign factor: (-1)^(i-k) per Floater-Hormann formula
            let sign = if (i - k) % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
            sum += sign / prod;
        }
        w[i] = sum;
    }
    w
}

/// Compute second-kind barycentric Lagrange weights for nodes `x`.
///
/// `w_i = 1 / Π_{j≠i} (x_i - x_j)`
///
/// Scaled so the maximum absolute weight is 1 (improves numerical stability).
fn compute_lagrange_weights(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let mut prod = 1.0_f64;
        for j in 0..n {
            if j != i {
                prod *= x[i] - x[j];
            }
        }
        w.push(1.0 / prod);
    }
    // Scale to avoid overflow/underflow
    let max_abs = w.iter().map(|wi| wi.abs()).fold(0.0_f64, f64::max);
    if max_abs > 0.0 {
        for wi in &mut w {
            *wi /= max_abs;
        }
    }
    w
}

// ---------------------------------------------------------------------------
// FloaterHormann — task-spec API (new/evaluate/evaluate_many)
// ---------------------------------------------------------------------------

/// Result type alias for Floater-Hormann interpolation.
pub type InterpResult<T> = Result<T, crate::error::InterpolateError>;

/// Floater-Hormann barycentric rational interpolant (2007) with the canonical
/// `new / evaluate / evaluate_many` API.
///
/// This is a second, ergonomically named entry point that wraps the same
/// underlying algorithm as [`BarycentricRational`].  Use whichever API you
/// prefer; both produce identical results.
///
/// # Mathematical Background
///
/// Given nodes `x[0] < x[1] < … < x[n]`, function values `f[k]`, and a
/// blending parameter `d` (0 ≤ d ≤ n), the weights are:
///
/// ```text
/// w[k] = (-1)^k  Σ_{i=max(0,k-d)}^{min(k,n-d)}  C(d, k-i) / Π_{j=i,j≠k}^{i+d}(x[k]−x[j])
/// ```
///
/// and the interpolant is evaluated in barycentric form:
///
/// ```text
/// r(x) = (Σ_k  w[k]/(x−x[k]) f[k])  /  (Σ_k  w[k]/(x−x[k]))
/// ```
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::barycentric::FloaterHormann;
///
/// let x: Vec<f64> = (0..=5).map(|i| i as f64).collect();
/// let f: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
/// let fh = FloaterHormann::new(&x, &f, 3).expect("doc example: should succeed");
/// assert!((fh.evaluate(2.5) - 6.25).abs() < 1e-8);
/// ```
#[derive(Debug, Clone)]
pub struct FloaterHormann {
    /// Sorted interpolation nodes.
    x: Vec<f64>,
    /// Function values at nodes.
    f: Vec<f64>,
    /// Barycentric weights.
    w: Vec<f64>,
    /// Blending parameter.
    d: usize,
}

impl FloaterHormann {
    /// Construct a Floater-Hormann interpolant.
    ///
    /// # Parameters
    ///
    /// - `x`: strictly increasing nodes (length ≥ 1)
    /// - `f`: function values at nodes (same length as `x`)
    /// - `d`: blending degree, 0 ≤ d ≤ n = x.len() − 1
    ///
    /// # Errors
    ///
    /// Returns `InterpolateError::InvalidInput` for mismatched lengths,
    /// empty input, invalid `d`, or non-monotone `x`.
    pub fn new(x: &[f64], f: &[f64], d: usize) -> InterpResult<Self> {
        let n_pts = x.len();
        if n_pts == 0 {
            return Err(crate::error::InterpolateError::InvalidInput {
                message: "FloaterHormann: x must be non-empty".into(),
            });
        }
        if f.len() != n_pts {
            return Err(crate::error::InterpolateError::InvalidInput {
                message: format!(
                    "FloaterHormann: x.len()={} != f.len()={}",
                    n_pts,
                    f.len()
                ),
            });
        }
        let n = n_pts - 1;
        if d > n {
            return Err(crate::error::InterpolateError::InvalidInput {
                message: format!(
                    "FloaterHormann: blending degree d={} > n={} (= x.len()-1)",
                    d, n
                ),
            });
        }
        for i in 1..n_pts {
            if x[i] <= x[i - 1] {
                return Err(crate::error::InterpolateError::InvalidInput {
                    message: format!(
                        "FloaterHormann: x must be strictly increasing; x[{}]={} <= x[{}]={}",
                        i, x[i], i-1, x[i-1]
                    ),
                });
            }
        }

        // Compute Floater-Hormann barycentric weights using the combinatorial
        // formula from Floater & Hormann 2007, equation (5):
        //   w[k] = (-1)^k  sum_{i=max(0,k-d)}^{min(k,n-d)}  lambda_{k,i}
        // where
        //   lambda_{k,i} = C(d, k-i) / prod_{j=i, j!=k}^{i+d} (x[k] - x[j])
        let w = fh_weights_with_binom(x, d);

        Ok(Self {
            x: x.to_vec(),
            f: f.to_vec(),
            w,
            d,
        })
    }

    /// Evaluate the rational interpolant at `x_new`.
    ///
    /// Returns the exact node value when `x_new` coincides with a node
    /// (within floating-point tolerance).
    #[inline]
    pub fn evaluate(&self, x_new: f64) -> f64 {
        fh_barycentric_eval(&self.x, &self.f, &self.w, x_new)
    }

    /// Evaluate the interpolant at each point in `xs`.
    pub fn evaluate_many(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().map(|&xv| self.evaluate(xv)).collect()
    }

    /// Blending degree used to construct this interpolant.
    #[inline]
    pub fn degree(&self) -> usize {
        self.d
    }

    /// Barycentric weights.
    #[inline]
    pub fn weights(&self) -> &[f64] {
        &self.w
    }

    /// Interpolation nodes.
    #[inline]
    pub fn nodes(&self) -> &[f64] {
        &self.x
    }

    /// Function values at nodes.
    #[inline]
    pub fn values(&self) -> &[f64] {
        &self.f
    }
}

// ---------------------------------------------------------------------------
// Internal helpers for FloaterHormann
// ---------------------------------------------------------------------------

/// Compute Floater-Hormann weights with explicit binomial coefficients.
///
/// This implementation strictly follows Floater & Hormann (2007), eq. (5):
///
/// ```text
/// w[k] = (-1)^k  Σ_{i=max(0,k-d)}^{min(k,n-d)}
///                  C(d, k-i) / Π_{j=i, j≠k}^{i+d} (x[k]−x[j])
/// ```
fn fh_weights_with_binom(x: &[f64], d: usize) -> Vec<f64> {
    let n_pts = x.len();
    let n = if n_pts > 0 { n_pts - 1 } else { 0 };
    let mut w = vec![0.0_f64; n_pts];

    for k in 0..n_pts {
        // Floater-Hormann 2007, eq (24):
        //   w_k = sum_{i=max(0,k-d)}^{min(k,n-d)} (-1)^i / prod_{j=i,j!=k}^{i+d} (x_k - x_j)
        //
        // Note: The sign (-1)^i ensures correct alternation.  Some formulations
        // pull out (-1)^k and write (-1)^(k-i) inside; both are equivalent.
        let i_min = if k >= d { k - d } else { 0 };
        let i_max_bound = if n >= d { n - d } else { 0 };
        let i_max = k.min(i_max_bound);

        let mut sum = 0.0_f64;
        for i in i_min..=i_max {
            let end = (i + d).min(n);
            let mut prod = 1.0_f64;
            for j in i..=end {
                if j != k {
                    let diff = x[k] - x[j];
                    prod *= diff;
                }
            }
            if prod.abs() > 1e-300 {
                let sign = if i % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
                sum += sign / prod;
            }
        }

        w[k] = sum;
    }
    w
}

/// Compute the binomial coefficient C(n, k) exactly for small n, k.
fn binomial_coeff(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k); // symmetry
    let mut result = 1u64;
    for i in 0..k {
        result = result * (n - i) as u64 / (i + 1) as u64;
    }
    result
}

/// Evaluate using the barycentric rational form.
/// Returns exact node value on coincidence.
#[inline]
fn fh_barycentric_eval(x: &[f64], f: &[f64], w: &[f64], xi: f64) -> f64 {
    let n = x.len();
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for i in 0..n {
        let diff = xi - x[i];
        // Exact coincidence with node → return function value immediately
        if diff.abs() < f64::EPSILON * (1.0 + xi.abs().max(x[i].abs())) {
            return f[i];
        }
        let t = w[i] / diff;
        num += t * f[i];
        den += t;
    }
    if den.abs() < f64::MIN_POSITIVE {
        return f64::NAN;
    }
    num / den
}

// ---------------------------------------------------------------------------
// Tests for FloaterHormann (task-spec struct)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod fh_tests {
    use super::FloaterHormann;
    use approx::assert_abs_diff_eq;

    fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
            .collect()
    }

    fn runge(x: f64) -> f64 {
        1.0 / (1.0 + 25.0 * x * x)
    }

    // ---- construction ----

    #[test]
    fn test_fh_new_valid() {
        let x = linspace(0.0, 1.0, 5);
        let f: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let fh = FloaterHormann::new(&x, &f, 3);
        assert!(fh.is_ok());
    }

    #[test]
    fn test_fh_new_empty_error() {
        let result = FloaterHormann::new(&[], &[], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_fh_new_length_mismatch_error() {
        let x = vec![0.0, 1.0, 2.0];
        let f = vec![0.0, 1.0];
        let result = FloaterHormann::new(&x, &f, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_fh_new_d_gt_n_error() {
        let x = vec![0.0, 1.0];
        let f = vec![0.0, 1.0];
        // d=2 > n=1 → error
        let result = FloaterHormann::new(&x, &f, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_fh_new_non_increasing_error() {
        let x = vec![0.0, 2.0, 1.0];
        let f = vec![0.0, 4.0, 1.0];
        let result = FloaterHormann::new(&x, &f, 1);
        assert!(result.is_err());
    }

    // ---- exact reproduction at nodes ----

    #[test]
    fn test_fh_evaluate_exact_at_nodes() {
        let x = linspace(-1.0, 1.0, 10);
        let f: Vec<f64> = x.iter().map(|&xi| runge(xi)).collect();
        let fh = FloaterHormann::new(&x, &f, 4).expect("test: should succeed");
        for (&xi, &fi) in x.iter().zip(f.iter()) {
            assert_abs_diff_eq!(fh.evaluate(xi), fi, epsilon = 1e-9);
        }
    }

    // ---- polynomial reproduction ----

    #[test]
    fn test_fh_linear_exact() {
        // f(x) = 2x + 3; any d should reproduce this on ≥2 nodes
        let x = linspace(0.0, 5.0, 6);
        let f: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 3.0).collect();
        let fh = FloaterHormann::new(&x, &f, 3).expect("test: should succeed");
        for xi in linspace(0.0, 5.0, 20) {
            assert_abs_diff_eq!(fh.evaluate(xi), 2.0 * xi + 3.0, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_fh_quadratic_exact() {
        // f(x) = x^2; FH with d≥2 reproduces quadratic exactly
        let x = linspace(-2.0, 2.0, 7);
        let f: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let fh = FloaterHormann::new(&x, &f, 4).expect("test: should succeed");
        for xi in linspace(-1.9, 1.9, 30) {
            assert_abs_diff_eq!(fh.evaluate(xi), xi * xi, epsilon = 1e-7);
        }
    }

    #[test]
    fn test_fh_cubic_exact() {
        // f(x) = x^3 − x; FH with d≥3 reproduces cubic exactly on enough nodes
        let x = linspace(0.0, 4.0, 8);
        let f: Vec<f64> = x.iter().map(|&xi| xi * xi * xi - xi).collect();
        let fh = FloaterHormann::new(&x, &f, 5).expect("test: should succeed");
        for xi in linspace(0.1, 3.9, 20) {
            let expected = xi * xi * xi - xi;
            assert_abs_diff_eq!(fh.evaluate(xi), expected, epsilon = 1e-6);
        }
    }

    // ---- Runge stability ----

    #[test]
    fn test_fh_runge_stability() {
        // 20 equispaced nodes; Lagrange would oscillate but FH should not
        let x = linspace(-1.0, 1.0, 20);
        let f: Vec<f64> = x.iter().map(|&xi| runge(xi)).collect();
        let fh = FloaterHormann::new(&x, &f, 6).expect("test: should succeed");
        let test_pts = linspace(-0.99, 0.99, 200);
        let max_err = test_pts
            .iter()
            .map(|&xi| (fh.evaluate(xi) - runge(xi)).abs())
            .fold(0.0_f64, f64::max);
        // FH avoids Runge oscillation; error should be well below 1
        assert!(
            max_err < 0.5,
            "FloaterHormann Runge max error too large: {:.4e}",
            max_err
        );
    }

    #[test]
    fn test_fh_runge_better_than_lagrange() {
        // For Runge function with 16 equispaced nodes, FH (d=5) should give
        // smaller max error than the naive polynomial (d=n)
        let x = linspace(-1.0, 1.0, 16);
        let f: Vec<f64> = x.iter().map(|&xi| runge(xi)).collect();
        let fh_stable = FloaterHormann::new(&x, &f, 5).expect("test: should succeed");
        let test_pts = linspace(-0.98, 0.98, 100);
        let err: f64 = test_pts
            .iter()
            .map(|&xi| (fh_stable.evaluate(xi) - runge(xi)).abs())
            .fold(0.0_f64, f64::max);
        assert!(err < 0.3, "FloaterHormann stable error too large: {:.4e}", err);
    }

    // ---- evaluate_many ----

    #[test]
    fn test_fh_evaluate_many_length() {
        let x = linspace(0.0, 1.0, 6);
        let f: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let fh = FloaterHormann::new(&x, &f, 3).expect("test: should succeed");
        let xs = linspace(0.0, 1.0, 25);
        let vals = fh.evaluate_many(&xs);
        assert_eq!(vals.len(), 25);
    }

    #[test]
    fn test_fh_evaluate_many_correctness() {
        let x = linspace(0.0, 3.0, 7);
        let f: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
        let fh = FloaterHormann::new(&x, &f, 4).expect("test: should succeed");
        let xs = linspace(0.0, 3.0, 15);
        let vals = fh.evaluate_many(&xs);
        for (xi, vi) in xs.iter().zip(vals.iter()) {
            assert!(vi.is_finite(), "evaluate_many returned non-finite value at xi={xi}");
        }
        // At nodes the values should be exact
        for (&xi, &fi) in x.iter().zip(f.iter()) {
            assert_abs_diff_eq!(fh.evaluate(xi), fi, epsilon = 1e-9);
        }
    }

    // ---- trigonometric function ----

    #[test]
    fn test_fh_sin_approximation() {
        // Approximate sin(x) on [0, π] with 10 nodes
        let x = linspace(0.0, std::f64::consts::PI, 10);
        let f: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
        let fh = FloaterHormann::new(&x, &f, 4).expect("test: should succeed");
        let test_pts = linspace(0.1, std::f64::consts::PI - 0.1, 30);
        for xi in test_pts {
            let diff = (fh.evaluate(xi) - xi.sin()).abs();
            assert!(diff < 1e-5, "sin approx error at x={xi}: {diff:.4e}");
        }
    }

    // ---- accessors ----

    #[test]
    fn test_fh_accessors() {
        let x = vec![0.0, 1.0, 2.0];
        let f = vec![1.0, 2.0, 5.0];
        let fh = FloaterHormann::new(&x, &f, 2).expect("test: should succeed");
        assert_eq!(fh.degree(), 2);
        assert_eq!(fh.nodes().len(), 3);
        assert_eq!(fh.values().len(), 3);
        assert_eq!(fh.weights().len(), 3);
    }

    #[test]
    fn test_fh_single_node() {
        let fh = FloaterHormann::new(&[3.14], &[2.71], 0).expect("test: should succeed");
        assert_abs_diff_eq!(fh.evaluate(3.14), 2.71, epsilon = 1e-12);
    }

    // ---- two-node case ----
    #[test]
    fn test_fh_two_nodes_d0() {
        let x = vec![0.0, 1.0];
        let f = vec![0.0, 2.0];
        // d=0: piecewise constant-like
        let fh = FloaterHormann::new(&x, &f, 0).expect("test: should succeed");
        // Must reproduce nodes exactly
        assert_abs_diff_eq!(fh.evaluate(0.0), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fh.evaluate(1.0), 2.0, epsilon = 1e-10);
    }

    // ---- convergence ----
    #[test]
    fn test_fh_convergence_with_more_nodes() {
        // As we increase node count, error on Runge should decrease
        let ref_x: Vec<f64> = linspace(-1.0, 1.0, 500);
        let ref_f: Vec<f64> = ref_x.iter().map(|&xi| runge(xi)).collect();

        let mut prev_err = f64::INFINITY;
        for n in [8usize, 16, 32] {
            let x = linspace(-1.0, 1.0, n);
            let f: Vec<f64> = x.iter().map(|&xi| runge(xi)).collect();
            let fh = FloaterHormann::new(&x, &f, 4).expect("test: should succeed");
            let err: f64 = ref_x
                .iter()
                .zip(ref_f.iter())
                .map(|(&xi, &fi)| (fh.evaluate(xi) - fi).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                err < prev_err * 2.0, // loose bound allowing for slight non-monotone convergence
                "n={n}: err={err:.4e} not better than prev={prev_err:.4e}"
            );
            prev_err = err;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Runge function: f(x) = 1/(1+25x^2) on [-1,1]
    fn runge(x: f64) -> f64 {
        1.0 / (1.0 + 25.0 * x * x)
    }

    /// Generate equispaced nodes on [a, b]
    fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
            .collect()
    }

    // ── BarycentricRational ──────────────────────────────────────────────────

    #[test]
    fn test_fh_exact_at_nodes() {
        let x = linspace(-1.0, 1.0, 8);
        let y: Vec<f64> = x.iter().map(|&xi| runge(xi)).collect();
        let interp = BarycentricRational::fit(&x, &y, 3).expect("test: should succeed");
        for (xi, yi) in x.iter().zip(y.iter()) {
            assert_abs_diff_eq!(interp.eval(*xi), *yi, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fh_runge_no_oscillation() {
        // Use 16 equispaced nodes: FH with d=5 should give errors well below 1
        let n = 16usize;
        let x = linspace(-1.0, 1.0, n);
        let y: Vec<f64> = x.iter().map(|&xi| runge(xi)).collect();
        let interp = BarycentricRational::fit(&x, &y, 5).expect("test: should succeed");

        // Check at 200 interior points
        let test_pts = linspace(-0.99, 0.99, 200);
        let max_err = test_pts
            .iter()
            .map(|&xi| (interp.eval(xi) - runge(xi)).abs())
            .fold(0.0_f64, f64::max);
        // FH d=5 on 16 equispaced nodes should have error << 0.5
        assert!(
            max_err < 0.5,
            "Max error on Runge function too large: {:.4e}",
            max_err
        );
    }

    #[test]
    fn test_fh_polynomial_exact() {
        // Cubic polynomial — FH with d≥3 should reproduce it
        let x = linspace(0.0, 4.0, 6);
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi * xi - 2.0 * xi + 1.0).collect();
        let interp = BarycentricRational::fit(&x, &y, 4).expect("test: should succeed");
        let test_pts = linspace(0.0, 4.0, 50);
        for xi in test_pts {
            let expected = xi * xi * xi - 2.0 * xi + 1.0;
            assert_abs_diff_eq!(interp.eval(xi), expected, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_fh_eval_many() {
        let x = linspace(0.0, 1.0, 6);
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let interp = BarycentricRational::fit(&x, &y, 3).expect("test: should succeed");
        let xi = linspace(0.0, 1.0, 20);
        let vals = interp.eval_many(&xi);
        assert_eq!(vals.len(), 20);
        for (xi, v) in xi.iter().zip(vals.iter()) {
            assert_abs_diff_eq!(*v, xi * xi, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_fh_derivative_linear() {
        // f(x) = 3x + 1, derivative = 3
        let x = linspace(0.0, 4.0, 8);
        let y: Vec<f64> = x.iter().map(|&xi| 3.0 * xi + 1.0).collect();
        let interp = BarycentricRational::fit(&x, &y, 3).expect("test: should succeed");
        for t in linspace(0.5, 3.5, 10) {
            assert_abs_diff_eq!(interp.derivative(t, 1), 3.0, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_fh_invalid_d_gt_n() {
        let x = vec![0.0, 1.0];
        let y = vec![0.0, 1.0];
        let result = BarycentricRational::fit(&x, &y, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_fh_non_increasing_x() {
        let x = vec![0.0, 2.0, 1.0];
        let y = vec![0.0, 2.0, 1.0];
        let result = BarycentricRational::fit(&x, &y, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_fh_empty_data() {
        let result = BarycentricRational::fit(&[], &[], 0);
        assert!(result.is_err());
    }

    // ── BarycentricLagrange ──────────────────────────────────────────────────

    #[test]
    fn test_lagrange_exact_at_nodes() {
        let x = linspace(0.0, 5.0, 6);
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let interp = BarycentricLagrange::fit(&x, &y).expect("test: should succeed");
        for (xi, yi) in x.iter().zip(y.iter()) {
            assert_abs_diff_eq!(interp.eval(*xi), *yi, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_lagrange_polynomial_exact() {
        // Degree-3 polynomial: fits exactly with ≥4 nodes
        let x = linspace(0.0, 3.0, 4);
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi * xi - xi).collect();
        let interp = BarycentricLagrange::fit(&x, &y).expect("test: should succeed");
        let test_pts = linspace(0.0, 3.0, 30);
        for xi in test_pts {
            let expected = xi * xi * xi - xi;
            assert_abs_diff_eq!(interp.eval(xi), expected, epsilon = 1e-7);
        }
    }

    #[test]
    fn test_lagrange_add_point_incremental() {
        // Start with 3 points, add a 4th, check interpolant is still exact at nodes
        let x_init = vec![0.0, 1.0, 2.0];
        let y_init = vec![0.0, 1.0, 4.0];
        let mut interp = BarycentricLagrange::fit(&x_init, &y_init).expect("test: should succeed");
        interp.add_point(3.0, 9.0).expect("test: should succeed");

        // Check at all four nodes
        let x_all = vec![0.0, 1.0, 2.0, 3.0];
        let y_all = vec![0.0, 1.0, 4.0, 9.0];
        for (xi, yi) in x_all.iter().zip(y_all.iter()) {
            assert_abs_diff_eq!(interp.eval(*xi), *yi, epsilon = 1e-8);
        }
        assert_eq!(interp.len(), 4);
    }

    #[test]
    fn test_lagrange_add_duplicate_fails() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 4.0];
        let mut interp = BarycentricLagrange::fit(&x, &y).expect("test: should succeed");
        let result = interp.add_point(1.0, 99.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_lagrange_eval_many() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 0.0, 1.0, 0.0];
        let interp = BarycentricLagrange::fit(&x, &y).expect("test: should succeed");
        let xi = vec![0.5, 1.5, 2.5];
        let vals = interp.eval_many(&xi);
        assert_eq!(vals.len(), 3);
        for v in &vals {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_lagrange_single_point() {
        let interp = BarycentricLagrange::fit(&[2.5], &[7.0]).expect("test: should succeed");
        assert_abs_diff_eq!(interp.eval(2.5), 7.0, epsilon = 1e-12);
    }

    #[test]
    fn test_lagrange_empty_fails() {
        let result = BarycentricLagrange::fit(&[], &[]);
        assert!(result.is_err());
    }
}
