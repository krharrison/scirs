//! Kriging Interpolation and Variogram Modelling
//!
//! Provides:
//! - Parametric variogram models (Spherical, Exponential, Gaussian, Linear)
//! - Empirical variogram computation from point data
//! - Weighted Least Squares (WLS) variogram model fitting
//! - Ordinary Kriging with prediction variance
//!
//! # Mathematical background
//!
//! A variogram model `γ(h)` describes spatial structure:
//!
//! **Spherical**: `γ(h) = c₀ + c·[1.5(h/a) - 0.5(h/a)³]` for `h ≤ a`, else `c₀ + c`
//!
//! **Exponential**: `γ(h) = c₀ + c·[1 - exp(-h/a)]`
//!   (`a` is the practical range; effective range ≈ 3a)
//!
//! **Gaussian**: `γ(h) = c₀ + c·[1 - exp(-(h/a)²)]`
//!
//! **Linear**: `γ(h) = c₀ + slope·h`
//!
//! **Ordinary Kriging system** (n training points):
//! ```text
//! [ Γ  1 ] [ w ]   [ γ(x₀, xᵢ) ]
//! [ 1ᵀ 0 ] [ μ ] = [     1      ]
//! ```
//! where `Γᵢⱼ = γ(xᵢ, xⱼ)`, `w` are kriging weights, and `μ` is the Lagrange multiplier.
//! Prediction variance: `σ²ₖ = Σᵢ wᵢ γ(x₀, xᵢ) + μ`.
//!
//! # References
//! - Matheron, G. (1963). Principles of Geostatistics.
//! - Cressie, N. (1993). *Statistics for Spatial Data*.
//! - Webster, R. & Oliver, M.A. (2007). *Geostatistics for Environmental Scientists*.

use super::{SpatialError, SpatialResult};

// ---------------------------------------------------------------------------
// Variogram models
// ---------------------------------------------------------------------------

/// Parametric variogram model.
///
/// Each variant encodes a covariance structure parameterised by nugget, sill, and
/// range (or slope for the linear model).
#[derive(Debug, Clone, PartialEq)]
pub enum VariogramModel {
    /// Spherical variogram.
    ///
    /// `γ(h) = c₀ + c·[1.5(h/a) - 0.5(h/a)³]` for `0 < h ≤ a`, else `c₀ + c`.
    Spherical {
        /// Nugget variance `c₀ ≥ 0`.
        nugget: f64,
        /// Partial sill `c ≥ 0`.
        sill: f64,
        /// Range `a > 0`.
        range: f64,
    },
    /// Exponential variogram.
    ///
    /// `γ(h) = c₀ + c·[1 - exp(-h/a)]`
    Exponential {
        /// Nugget variance.
        nugget: f64,
        /// Partial sill.
        sill: f64,
        /// Scale parameter `a > 0` (practical range ≈ 3a).
        range: f64,
    },
    /// Gaussian variogram.
    ///
    /// `γ(h) = c₀ + c·[1 - exp(-(h/a)²)]`
    Gaussian {
        /// Nugget variance.
        nugget: f64,
        /// Partial sill.
        sill: f64,
        /// Scale parameter `a > 0`.
        range: f64,
    },
    /// Linear variogram (unbounded).
    ///
    /// `γ(h) = c₀ + slope·h`
    Linear {
        /// Nugget variance.
        nugget: f64,
        /// Slope `≥ 0`.
        slope: f64,
    },
}

impl VariogramModel {
    /// Evaluate the variogram at lag distance `h`.
    ///
    /// Returns `γ(h)`.  At `h = 0` the value is always 0 (not the nugget),
    /// consistent with the convention `γ(0) = 0` used in kriging systems.
    pub fn evaluate(&self, h: f64) -> f64 {
        if h < 0.0 {
            // Variogram is an even function of h
            return self.evaluate(-h);
        }
        if h == 0.0 {
            return 0.0;
        }
        match self {
            VariogramModel::Spherical { nugget, sill, range } => {
                if *range <= 0.0 {
                    return *nugget + *sill;
                }
                let hr = h / range;
                if hr >= 1.0 {
                    nugget + sill
                } else {
                    nugget + sill * (1.5 * hr - 0.5 * hr.powi(3))
                }
            }
            VariogramModel::Exponential { nugget, sill, range } => {
                if *range <= 0.0 {
                    return *nugget + *sill;
                }
                nugget + sill * (1.0 - (-h / range).exp())
            }
            VariogramModel::Gaussian { nugget, sill, range } => {
                if *range <= 0.0 {
                    return *nugget + *sill;
                }
                let hr2 = (h / range).powi(2);
                nugget + sill * (1.0 - (-hr2).exp())
            }
            VariogramModel::Linear { nugget, slope } => {
                nugget + slope * h
            }
        }
    }

    /// Fit a variogram model of the given type to an empirical variogram using
    /// Weighted Least Squares.
    ///
    /// WLS weights each lag bin by `counts[i] / γ̂(lag)²` as recommended by
    /// Cressie (1985).
    ///
    /// # Arguments
    /// * `emp`        – Empirical variogram (must have ≥ 2 non-empty lag bins).
    /// * `model_type` – One of `"spherical"`, `"exponential"`, `"gaussian"`, `"linear"`.
    ///
    /// # Errors
    /// Returns [`SpatialError`] when the model type is unknown or fitting fails.
    pub fn fit_empirical(emp: &EmpiricalVariogram, model_type: &str) -> SpatialResult<Self> {
        let lags = &emp.lag_centers;
        let gamma = &emp.semivariance;
        let counts = &emp.counts;

        let n = lags.len();
        if n < 2 {
            return Err(SpatialError::InsufficientData(
                "Need at least 2 lag bins to fit a variogram".to_string(),
            ));
        }

        // Filter to bins with at least one pair
        let valid: Vec<usize> = (0..n).filter(|&i| counts[i] > 0).collect();
        if valid.len() < 2 {
            return Err(SpatialError::InsufficientData(
                "Too few non-empty lag bins for variogram fitting".to_string(),
            ));
        }

        // Initial parameter estimates
        let gamma_max = gamma
            .iter()
            .cloned()
            .filter(|g| g.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        let gamma_min = gamma
            .iter()
            .cloned()
            .filter(|g| g.is_finite())
            .fold(f64::INFINITY, f64::min);
        let lag_max = lags
            .iter()
            .cloned()
            .filter(|l| l.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);

        // Guess: nugget ≈ smallest semivariance, sill ≈ max, range ≈ 60% of max lag
        let nugget_init = gamma_min.max(0.0);
        let sill_init = (gamma_max - nugget_init).max(1e-10);
        let range_init = (0.6 * lag_max).max(1e-10);
        let slope_init = if lag_max > 0.0 {
            (gamma_max - nugget_init) / lag_max
        } else {
            1.0
        };

        match model_type.to_lowercase().as_str() {
            "spherical" => {
                let (nugget, sill, range) = wls_fit_3param(
                    &valid,
                    lags,
                    gamma,
                    counts,
                    nugget_init,
                    sill_init,
                    range_init,
                    |h, n, s, r| {
                        if r <= 0.0 { return n + s; }
                        let hr = h / r;
                        if hr >= 1.0 { n + s } else { n + s * (1.5 * hr - 0.5 * hr.powi(3)) }
                    },
                )?;
                Ok(VariogramModel::Spherical { nugget, sill, range })
            }
            "exponential" => {
                let (nugget, sill, range) = wls_fit_3param(
                    &valid,
                    lags,
                    gamma,
                    counts,
                    nugget_init,
                    sill_init,
                    range_init,
                    |h, n, s, r| {
                        if r <= 0.0 { return n + s; }
                        n + s * (1.0 - (-h / r).exp())
                    },
                )?;
                Ok(VariogramModel::Exponential { nugget, sill, range })
            }
            "gaussian" => {
                let (nugget, sill, range) = wls_fit_3param(
                    &valid,
                    lags,
                    gamma,
                    counts,
                    nugget_init,
                    sill_init,
                    range_init,
                    |h, n, s, r| {
                        if r <= 0.0 { return n + s; }
                        n + s * (1.0 - (-(h / r).powi(2)).exp())
                    },
                )?;
                Ok(VariogramModel::Gaussian { nugget, sill, range })
            }
            "linear" => {
                let (nugget, slope) = wls_fit_linear(
                    &valid,
                    lags,
                    gamma,
                    counts,
                    nugget_init,
                    slope_init,
                )?;
                Ok(VariogramModel::Linear { nugget, slope })
            }
            other => Err(SpatialError::InvalidArgument(format!(
                "Unknown variogram model '{}'. Expected one of: spherical, exponential, gaussian, linear",
                other
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// WLS optimisation helpers
// ---------------------------------------------------------------------------

/// Weighted least squares for 3-parameter variogram models.
///
/// Uses grid search followed by coordinate descent (alternating 1-D golden-section
/// search) to minimise
/// `Σ_i wᵢ (γ_emp[i] - γ_model(h_i; n, c, r))²`
/// where `wᵢ = counts[i] / γ_model(h_i)²`.
fn wls_fit_3param<F>(
    valid: &[usize],
    lags: &[f64],
    gamma: &[f64],
    counts: &[usize],
    nugget_init: f64,
    sill_init: f64,
    range_init: f64,
    model_fn: F,
) -> SpatialResult<(f64, f64, f64)>
where
    F: Fn(f64, f64, f64, f64) -> f64 + Copy,
{
    /// WLS objective
    fn objective<F2: Fn(f64, f64, f64, f64) -> f64>(
        valid: &[usize],
        lags: &[f64],
        gamma: &[f64],
        counts: &[usize],
        nugget: f64,
        sill: f64,
        range: f64,
        model_fn: F2,
    ) -> f64 {
        if nugget < 0.0 || sill <= 0.0 || range <= 0.0 {
            return f64::INFINITY;
        }
        valid.iter().fold(0.0, |acc, &i| {
            let h = lags[i];
            let g_model = model_fn(h, nugget, sill, range);
            let w = if g_model > 1e-15 {
                counts[i] as f64 / (g_model * g_model)
            } else {
                0.0
            };
            let residual = gamma[i] - g_model;
            acc + w * residual * residual
        })
    }

    // Golden-section search in 1 parameter holding others fixed
    let golden_search = |f: &dyn Fn(f64) -> f64, lo: f64, hi: f64| -> f64 {
        let phi = (5.0_f64.sqrt() - 1.0) / 2.0; // ≈ 0.618
        let mut a = lo;
        let mut b = hi;
        let mut c = b - phi * (b - a);
        let mut d = a + phi * (b - a);
        let tol = 1e-8 * (1.0 + hi - lo);
        while (b - a).abs() > tol {
            if f(c) < f(d) {
                b = d;
            } else {
                a = c;
            }
            c = b - phi * (b - a);
            d = a + phi * (b - a);
        }
        (a + b) / 2.0
    };

    let mut nugget = nugget_init.max(0.0);
    let mut sill = sill_init.max(1e-10);
    let mut range = range_init.max(1e-10);

    // Coordinate descent iterations
    for _ in 0..200 {
        let n_fix = nugget;
        let c_fix = sill;
        let r_fix = range;

        // Optimise nugget
        {
            let obj = |n: f64| {
                objective(valid, lags, gamma, counts, n, c_fix, r_fix, model_fn)
            };
            nugget = golden_search(&obj, 0.0, gamma[valid[0]].max(0.0) * 1.5 + 1e-12);
        }
        // Optimise sill
        {
            let lag_max = lags[*valid.last().unwrap_or(&0)];
            let n_fix2 = nugget;
            let obj = |c: f64| {
                objective(valid, lags, gamma, counts, n_fix2, c, r_fix, model_fn)
            };
            sill = golden_search(&obj, 1e-12, (gamma.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 1.0) * 3.0);
            let _ = lag_max;
        }
        // Optimise range
        {
            let n_fix3 = nugget;
            let c_fix3 = sill;
            let lag_max = lags[*valid.last().unwrap_or(&0)];
            let obj = |r: f64| {
                objective(valid, lags, gamma, counts, n_fix3, c_fix3, r, model_fn)
            };
            range = golden_search(&obj, 1e-12, lag_max * 5.0 + 1.0);
        }

        // Check convergence
        let diff = (nugget - n_fix).abs()
            + (sill - c_fix).abs()
            + (range - r_fix).abs();
        if diff < 1e-10 {
            break;
        }
    }

    if !nugget.is_finite() || !sill.is_finite() || !range.is_finite() {
        return Err(SpatialError::ConvergenceError(
            "Variogram WLS fitting did not produce finite parameters".to_string(),
        ));
    }

    Ok((nugget.max(0.0), sill.max(1e-10), range.max(1e-10)))
}

/// WLS fitting for the linear model (2 parameters).
fn wls_fit_linear(
    valid: &[usize],
    lags: &[f64],
    gamma: &[f64],
    counts: &[usize],
    nugget_init: f64,
    slope_init: f64,
) -> SpatialResult<(f64, f64)> {
    fn objective(
        valid: &[usize],
        lags: &[f64],
        gamma: &[f64],
        counts: &[usize],
        nugget: f64,
        slope: f64,
    ) -> f64 {
        valid.iter().fold(0.0, |acc, &i| {
            let g_model = nugget + slope * lags[i];
            let w = if g_model > 1e-15 {
                counts[i] as f64 / (g_model * g_model)
            } else {
                0.0
            };
            let r = gamma[i] - g_model;
            acc + w * r * r
        })
    }

    let golden = |f: &dyn Fn(f64) -> f64, lo: f64, hi: f64| -> f64 {
        let phi = (5.0_f64.sqrt() - 1.0) / 2.0;
        let mut a = lo;
        let mut b = hi;
        let tol = 1e-9 * (1.0 + hi - lo);
        let mut c = b - phi * (b - a);
        let mut d = a + phi * (b - a);
        while (b - a).abs() > tol {
            if f(c) < f(d) { b = d; } else { a = c; }
            c = b - phi * (b - a);
            d = a + phi * (b - a);
        }
        (a + b) / 2.0
    };

    let mut nugget = nugget_init.max(0.0);
    let mut slope = slope_init.max(0.0);
    let gamma_max = gamma.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    for _ in 0..200 {
        let s_fix = slope;
        let n_fix = nugget;
        {
            let obj = |n: f64| objective(valid, lags, gamma, counts, n, s_fix);
            nugget = golden(&obj, 0.0, gamma_max.max(1.0) * 1.5);
        }
        {
            let n_fix2 = nugget;
            let obj = |s: f64| objective(valid, lags, gamma, counts, n_fix2, s);
            let lag_max = lags[*valid.last().unwrap_or(&0)];
            slope = golden(&obj, 0.0, if lag_max > 0.0 { gamma_max / lag_max * 3.0 + 1.0 } else { 10.0 });
        }
        if (nugget - n_fix).abs() + (slope - s_fix).abs() < 1e-10 {
            break;
        }
    }

    Ok((nugget.max(0.0), slope.max(0.0)))
}

// ---------------------------------------------------------------------------
// Empirical variogram
// ---------------------------------------------------------------------------

/// Empirical (sample) variogram computed from point observations.
#[derive(Debug, Clone)]
pub struct EmpiricalVariogram {
    /// Centre of each lag bin.
    pub lag_centers: Vec<f64>,
    /// Average semivariance in each lag bin:
    /// `γ̂(h) = (1/(2N(h))) Σ_{(i,j)∈N(h)} (z_i - z_j)²`.
    pub semivariance: Vec<f64>,
    /// Number of point pairs in each lag bin.
    pub counts: Vec<usize>,
}

/// Compute an empirical variogram from irregularly scattered point observations.
///
/// # Arguments
/// * `locations` – `(x, y)` coordinates of the measurement locations.
/// * `values`    – Observed values at each location.
/// * `n_lags`    – Number of equal-width lag bins.
/// * `max_lag`   – Maximum lag distance to consider.
///
/// # Errors
/// Returns [`SpatialError`] when `n < 2`, `n_lags < 1`, or `max_lag ≤ 0`.
pub fn empirical_variogram(
    locations: &[(f64, f64)],
    values: &[f64],
    n_lags: usize,
    max_lag: f64,
) -> SpatialResult<EmpiricalVariogram> {
    let n = locations.len();
    if n < 2 {
        return Err(SpatialError::InsufficientData(
            "Empirical variogram requires at least 2 locations".to_string(),
        ));
    }
    if n != values.len() {
        return Err(SpatialError::DimensionMismatch(
            "locations and values must have the same length".to_string(),
        ));
    }
    if n_lags < 1 {
        return Err(SpatialError::InvalidArgument(
            "n_lags must be at least 1".to_string(),
        ));
    }
    if max_lag <= 0.0 || !max_lag.is_finite() {
        return Err(SpatialError::InvalidArgument(
            "max_lag must be a positive finite number".to_string(),
        ));
    }

    let lag_width = max_lag / n_lags as f64;
    let mut sums = vec![0.0_f64; n_lags];
    let mut counts = vec![0_usize; n_lags];

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = locations[i].0 - locations[j].0;
            let dy = locations[i].1 - locations[j].1;
            let h = (dx * dx + dy * dy).sqrt();
            if h > max_lag {
                continue;
            }
            // Bin index (0-based): use floor so 0 goes into bin 0
            let bin = ((h / lag_width) as usize).min(n_lags - 1);
            let sq_diff = (values[i] - values[j]).powi(2);
            sums[bin] += sq_diff;
            counts[bin] += 1;
        }
    }

    let lag_centers: Vec<f64> = (0..n_lags)
        .map(|k| (k as f64 + 0.5) * lag_width)
        .collect();

    let semivariance: Vec<f64> = (0..n_lags)
        .map(|k| {
            if counts[k] > 0 {
                sums[k] / (2.0 * counts[k] as f64)
            } else {
                0.0
            }
        })
        .collect();

    Ok(EmpiricalVariogram {
        lag_centers,
        semivariance,
        counts,
    })
}

// ---------------------------------------------------------------------------
// Ordinary Kriging
// ---------------------------------------------------------------------------

/// Ordinary Kriging interpolator.
///
/// Uses the unbiasedness constraint `Σᵢ wᵢ = 1` and solves the kriging system
/// via Gaussian elimination for the kriging weights at each prediction location.
pub struct OrdinaryKriging {
    locations: Vec<(f64, f64)>,
    values: Vec<f64>,
    variogram: VariogramModel,
}

impl OrdinaryKriging {
    /// Create a new ordinary kriging system.
    ///
    /// # Arguments
    /// * `locations` – Training point coordinates.
    /// * `values`    – Training observations.
    /// * `variogram` – Fitted variogram model.
    pub fn new(
        locations: &[(f64, f64)],
        values: &[f64],
        variogram: VariogramModel,
    ) -> Self {
        Self {
            locations: locations.to_vec(),
            values: values.to_vec(),
            variogram,
        }
    }

    /// Predict the value at a new location `(x, y)`.
    ///
    /// Returns `(prediction, kriging_variance)`.
    ///
    /// # Errors
    /// Returns [`SpatialError`] when the kriging system is singular or has fewer
    /// than 1 training point.
    pub fn predict(&self, x: f64, y: f64) -> SpatialResult<(f64, f64)> {
        let n = self.locations.len();
        if n == 0 {
            return Err(SpatialError::InsufficientData(
                "Kriging requires at least one training point".to_string(),
            ));
        }

        // Build (n+1)×(n+1) kriging matrix A and RHS vector b
        // Layout: rows/cols 0..n are the n training points; row/col n is the
        // Lagrange multiplier for the unbiasedness constraint.
        let size = n + 1;
        let mut a = vec![0.0_f64; size * size];
        let mut b = vec![0.0_f64; size];

        // Fill A: A[i][j] = γ(xᵢ, xⱼ)
        for i in 0..n {
            for j in 0..n {
                let dx = self.locations[i].0 - self.locations[j].0;
                let dy = self.locations[i].1 - self.locations[j].1;
                let h = (dx * dx + dy * dy).sqrt();
                a[i * size + j] = self.variogram.evaluate(h);
            }
            // Unbiasedness column/row
            a[i * size + n] = 1.0;
            a[n * size + i] = 1.0;
        }
        a[n * size + n] = 0.0;

        // RHS b: b[i] = γ(x₀, xᵢ), b[n] = 1
        for i in 0..n {
            let dx = x - self.locations[i].0;
            let dy = y - self.locations[i].1;
            let h = (dx * dx + dy * dy).sqrt();
            b[i] = self.variogram.evaluate(h);
        }
        b[n] = 1.0;

        // Solve A·w = b via Gaussian elimination with partial pivoting
        let w = solve_linear(size, &mut a, &mut b)?;

        let prediction: f64 = (0..n)
            .map(|i| w[i] * self.values[i])
            .sum();

        // Kriging variance: σ²ₖ = Σᵢ wᵢ γ(x₀, xᵢ) + μ
        // where μ = w[n] (the Lagrange multiplier)
        // Rebuild b[0..n] (solve modifies b in place — we need original rhs)
        let mut krig_var = w[n]; // Lagrange multiplier contribution
        for i in 0..n {
            let dx = x - self.locations[i].0;
            let dy = y - self.locations[i].1;
            let h = (dx * dx + dy * dy).sqrt();
            krig_var += w[i] * self.variogram.evaluate(h);
        }
        let krig_var = krig_var.max(0.0);

        Ok((prediction, krig_var))
    }

    /// Batch prediction at every combination of `(xs[i], ys[j])` — i.e. a grid.
    ///
    /// Returns `(predictions, variances)` each of length `xs.len() * ys.len()`,
    /// ordered row-major (outer loop over `xs`, inner loop over `ys`).
    ///
    /// # Errors
    /// Propagates errors from [`OrdinaryKriging::predict`].
    pub fn predict_grid(
        &self,
        xs: &[f64],
        ys: &[f64],
    ) -> SpatialResult<(Vec<f64>, Vec<f64>)> {
        let mut preds = Vec::with_capacity(xs.len() * ys.len());
        let mut vars = Vec::with_capacity(xs.len() * ys.len());

        for &xi in xs {
            for &yi in ys {
                let (p, v) = self.predict(xi, yi)?;
                preds.push(p);
                vars.push(v);
            }
        }

        Ok((preds, vars))
    }
}

// ---------------------------------------------------------------------------
// Linear system solver (Gaussian elimination with partial pivoting)
// ---------------------------------------------------------------------------

/// Solve `A·x = b` via Gaussian elimination with partial pivoting.
///
/// `a` is provided as a flat row-major slice of length `n*n` (modified in place).
/// `b` is the right-hand-side vector of length `n` (modified in place).
///
/// Returns the solution vector `x`.
fn solve_linear(n: usize, a: &mut [f64], b: &mut [f64]) -> SpatialResult<Vec<f64>> {
    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let pivot_row = (col..n)
            .max_by(|&r1, &r2| {
                a[r1 * n + col]
                    .abs()
                    .partial_cmp(&a[r2 * n + col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);

        if a[pivot_row * n + col].abs() < 1e-12 {
            return Err(SpatialError::ComputationError(
                "Kriging system is singular; check for duplicate locations or degenerate variogram".to_string(),
            ));
        }

        // Swap rows
        if pivot_row != col {
            for k in 0..n {
                a.swap(col * n + k, pivot_row * n + k);
            }
            b.swap(col, pivot_row);
        }

        let pivot = a[col * n + col];
        for row in (col + 1)..n {
            let factor = a[row * n + col] / pivot;
            a[row * n + col] = 0.0;
            for k in (col + 1)..n {
                a[row * n + k] -= factor * a[col * n + k];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i * n + j] * x[j];
        }
        x[i] = sum / a[i * n + i];
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------------
    // Variogram model tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_spherical_at_origin() {
        let vm = VariogramModel::Spherical { nugget: 0.0, sill: 1.0, range: 1.0 };
        assert_eq!(vm.evaluate(0.0), 0.0);
    }

    #[test]
    fn test_spherical_at_range() {
        let vm = VariogramModel::Spherical { nugget: 0.1, sill: 0.9, range: 10.0 };
        // At h = range: γ = nugget + sill
        let v = vm.evaluate(10.0);
        assert!((v - 1.0).abs() < 1e-10, "spherical at range: {}", v);
    }

    #[test]
    fn test_spherical_beyond_range_equals_sill() {
        let vm = VariogramModel::Spherical { nugget: 0.2, sill: 0.8, range: 5.0 };
        let v1 = vm.evaluate(5.0);
        let v2 = vm.evaluate(100.0);
        assert!((v1 - v2).abs() < 1e-10, "beyond range should equal sill+nugget");
    }

    #[test]
    fn test_exponential_monotone_increasing() {
        let vm = VariogramModel::Exponential { nugget: 0.0, sill: 1.0, range: 2.0 };
        let vals: Vec<f64> = (0..=10).map(|i| vm.evaluate(i as f64 * 0.5)).collect();
        for w in vals.windows(2) {
            assert!(w[1] >= w[0] - 1e-12, "Exponential should be non-decreasing");
        }
    }

    #[test]
    fn test_gaussian_monotone_increasing() {
        let vm = VariogramModel::Gaussian { nugget: 0.1, sill: 0.9, range: 3.0 };
        let vals: Vec<f64> = (0..=15).map(|i| vm.evaluate(i as f64 * 0.4)).collect();
        for w in vals.windows(2) {
            assert!(w[1] >= w[0] - 1e-12, "Gaussian should be non-decreasing");
        }
    }

    #[test]
    fn test_linear_evaluate() {
        let vm = VariogramModel::Linear { nugget: 0.5, slope: 2.0 };
        assert_eq!(vm.evaluate(0.0), 0.0);
        assert!((vm.evaluate(1.0) - 2.5).abs() < 1e-12);
        assert!((vm.evaluate(3.0) - 6.5).abs() < 1e-12);
    }

    #[test]
    fn test_variogram_negative_h() {
        let vm = VariogramModel::Exponential { nugget: 0.0, sill: 1.0, range: 1.0 };
        let v_pos = vm.evaluate(1.0);
        let v_neg = vm.evaluate(-1.0);
        assert!((v_pos - v_neg).abs() < 1e-12, "variogram must be even function");
    }

    // ---------------------------------------------------------------------------
    // Empirical variogram tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_empirical_variogram_basic() {
        let locs: Vec<(f64, f64)> = (0..10).map(|i| (i as f64, 0.0)).collect();
        let vals: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let emp = empirical_variogram(&locs, &vals, 5, 9.0)
            .expect("empirical_variogram failed");
        assert_eq!(emp.lag_centers.len(), 5);
        assert_eq!(emp.semivariance.len(), 5);
        assert_eq!(emp.counts.len(), 5);
    }

    #[test]
    fn test_empirical_variogram_counts_positive() {
        let locs: Vec<(f64, f64)> = (0..8).map(|i| (i as f64, 0.0)).collect();
        let vals: Vec<f64> = vec![1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0];
        let emp = empirical_variogram(&locs, &vals, 4, 7.0)
            .expect("empirical_variogram failed");
        let total_pairs: usize = emp.counts.iter().sum();
        // Total unique pairs = n*(n-1)/2 = 28 within max_lag
        assert!(total_pairs > 0, "should have pairs");
    }

    #[test]
    fn test_empirical_variogram_constant_values() {
        let locs: Vec<(f64, f64)> = (0..5).map(|i| (i as f64, 0.0)).collect();
        let vals = vec![3.0_f64; 5];
        let emp = empirical_variogram(&locs, &vals, 3, 4.0)
            .expect("empirical_variogram failed");
        for &sv in &emp.semivariance {
            assert!(sv.abs() < 1e-12, "constant values → zero semivariance");
        }
    }

    #[test]
    fn test_empirical_variogram_dimension_mismatch() {
        let locs: Vec<(f64, f64)> = vec![(0.0, 0.0), (1.0, 0.0)];
        let vals = vec![1.0, 2.0, 3.0]; // wrong length
        assert!(empirical_variogram(&locs, &vals, 2, 2.0).is_err());
    }

    // ---------------------------------------------------------------------------
    // Variogram fitting tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_fit_spherical_recovers_structure() {
        // Build synthetic spherical variogram data
        let nugget_true = 0.1;
        let sill_true = 1.0;
        let range_true = 5.0;
        let vm_true = VariogramModel::Spherical {
            nugget: nugget_true,
            sill: sill_true,
            range: range_true,
        };
        let lags: Vec<f64> = (1..=12).map(|i| i as f64 * 0.6).collect();
        let gamma: Vec<f64> = lags.iter().map(|&h| vm_true.evaluate(h)).collect();
        let counts: Vec<usize> = vec![20; lags.len()];
        let emp = EmpiricalVariogram {
            lag_centers: lags,
            semivariance: gamma,
            counts,
        };
        let fitted = VariogramModel::fit_empirical(&emp, "spherical")
            .expect("fit_empirical failed");
        if let VariogramModel::Spherical { nugget, sill, range } = fitted {
            assert!((nugget - nugget_true).abs() < 0.05, "nugget: {}", nugget);
            assert!((sill - sill_true).abs() < 0.1, "sill: {}", sill);
            assert!((range - range_true).abs() < 0.5, "range: {}", range);
        } else {
            panic!("Expected Spherical variant");
        }
    }

    #[test]
    fn test_fit_unknown_model_type() {
        let emp = EmpiricalVariogram {
            lag_centers: vec![1.0, 2.0],
            semivariance: vec![0.1, 0.2],
            counts: vec![10, 10],
        };
        assert!(VariogramModel::fit_empirical(&emp, "quadratic").is_err());
    }

    // ---------------------------------------------------------------------------
    // Ordinary kriging tests
    // ---------------------------------------------------------------------------

    /// Place training points on a line z(x) = 2x and predict in between.
    #[test]
    fn test_kriging_linear_trend() {
        let n = 6_usize;
        let locs: Vec<(f64, f64)> = (0..n).map(|i| (i as f64 * 2.0, 0.0)).collect();
        let vals: Vec<f64> = locs.iter().map(|&(x, _)| 2.0 * x).collect();
        let vm = VariogramModel::Linear { nugget: 0.0, slope: 0.1 };
        let kriging = OrdinaryKriging::new(&locs, &vals, vm);

        // Predict at midpoint between loc[2] and loc[3]
        let (pred, var) = kriging.predict(5.0, 0.0).expect("kriging predict failed");
        assert!(
            (pred - 10.0).abs() < 2.0,
            "prediction {} far from expected 10.0",
            pred
        );
        assert!(var >= 0.0, "variance should be non-negative");
    }

    #[test]
    fn test_kriging_at_training_point() {
        let locs: Vec<(f64, f64)> = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)];
        let vals = vec![0.0_f64, 1.0, 4.0, 9.0];
        let vm = VariogramModel::Exponential { nugget: 1e-6, sill: 5.0, range: 2.0 };
        let kriging = OrdinaryKriging::new(&locs, &vals, vm);

        // Prediction at a training point should be close to the training value
        let (pred, _) = kriging.predict(1.0, 0.0).expect("kriging predict failed");
        assert!(
            (pred - 1.0).abs() < 0.5,
            "At training point: {} ≈ 1.0",
            pred
        );
    }

    #[test]
    fn test_kriging_variance_nonnegative() {
        let locs: Vec<(f64, f64)> =
            vec![(0.0, 0.0), (5.0, 0.0), (0.0, 5.0), (5.0, 5.0)];
        let vals = vec![1.0_f64, 2.0, 3.0, 4.0];
        let vm = VariogramModel::Spherical { nugget: 0.1, sill: 0.9, range: 8.0 };
        let kriging = OrdinaryKriging::new(&locs, &vals, vm);

        let test_pts = vec![(2.5, 2.5), (1.0, 1.0), (4.9, 4.9)];
        for (x, y) in test_pts {
            let (_, var) = kriging.predict(x, y).expect("kriging predict failed");
            assert!(var >= 0.0, "variance at ({},{}) = {} < 0", x, y, var);
        }
    }

    #[test]
    fn test_predict_grid_shape() {
        let locs: Vec<(f64, f64)> =
            vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
        let vals = vec![1.0_f64, 2.0, 3.0, 4.0];
        let vm = VariogramModel::Gaussian { nugget: 0.05, sill: 1.0, range: 2.0 };
        let kriging = OrdinaryKriging::new(&locs, &vals, vm);

        let xs = vec![0.25, 0.5, 0.75];
        let ys = vec![0.25, 0.5, 0.75];
        let (preds, vars) = kriging.predict_grid(&xs, &ys).expect("predict_grid failed");
        assert_eq!(preds.len(), xs.len() * ys.len());
        assert_eq!(vars.len(), xs.len() * ys.len());
        for v in &vars {
            assert!(*v >= 0.0, "grid variance < 0");
        }
    }

    #[test]
    fn test_kriging_no_training_data_fails() {
        let vm = VariogramModel::Linear { nugget: 0.0, slope: 1.0 };
        let kriging = OrdinaryKriging::new(&[], &[], vm);
        assert!(kriging.predict(0.5, 0.5).is_err());
    }
}
