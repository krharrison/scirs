//! ARFIMA(p, d, q) model: simulation, fitting, and forecasting.
//!
//! ARFIMA (AutoRegressive Fractionally Integrated Moving Average) generalises
//! ARIMA to non-integer differencing order `d`.  When `d ∈ (0, 0.5)` the
//! process is stationary with long memory (autocorrelations decay hyperbolically).
//!
//! The model is:  `φ(L) (1-L)^d (y_t − μ) = θ(L) ε_t`
//! where `φ(L)` is the AR polynomial and `θ(L)` the MA polynomial.
//!
//! # References
//! - Granger, C. W. J. & Joyeux, R. (1980). *An introduction to long-memory
//!   time series models and fractional differencing*. J. Time Series Anal.
//! - Hosking, J. R. M. (1981). *Fractional differencing*. Biometrika.

use std::f64::consts::PI;

use scirs2_core::ndarray::{Array1, Ix1};
use scirs2_core::random::{random_normal_array, seeded_rng};

use crate::error::{Result, TimeSeriesError};

// ---------------------------------------------------------------------------
// Public-use internal helper – also consumed by estimation.rs
// ---------------------------------------------------------------------------

/// Compute the k-th binomial coefficient of the fractional differencing expansion.
///
/// `w_k = ∏_{j=1}^{k} (j - 1 - d) / j`
///
/// For positive `d` these weights are alternating in sign and decay
/// hyperbolically, truncating after threshold.
pub(crate) fn frac_diff_weight(d: f64, k: usize) -> f64 {
    if k == 0 {
        return 1.0;
    }
    let mut w = 1.0_f64;
    for j in 1..=k {
        w *= (j as f64 - 1.0 - d) / j as f64;
    }
    w
}

/// Compute all truncated fractional-differencing weights for operator `(1-L)^d`.
///
/// Weights with `|w_k| < threshold` are omitted.  The vector always contains
/// at least the k=0 weight (= 1.0).
pub(crate) fn frac_diff_weights_truncated(d: f64, max_terms: usize, threshold: f64) -> Vec<f64> {
    let mut weights = Vec::with_capacity(max_terms.min(256));
    weights.push(1.0_f64); // w_0 = 1
    for k in 1..max_terms {
        let w = frac_diff_weight(d, k);
        if w.abs() < threshold {
            break;
        }
        weights.push(w);
    }
    weights
}

// ---------------------------------------------------------------------------
// ARFIMA model struct
// ---------------------------------------------------------------------------

/// ARFIMA(p, d, q) model specification.
///
/// Fractionally Integrated ARMA: `φ(L)(1-L)^d (y_t - μ) = θ(L) ε_t`.
/// When `d ∈ (0, 0.5)` the process is stationary long-memory.
/// When `d ∈ (-0.5, 0)` the process is anti-persistent.
#[derive(Debug, Clone)]
pub struct ARFIMAModel {
    /// AR order p.
    pub p: usize,
    /// Fractional differencing parameter d; long memory when `d ∈ (0, 0.5)`.
    pub d: f64,
    /// MA order q.
    pub q: usize,
    /// AR coefficients φ₁, …, φ_p (length p).
    pub ar: Vec<f64>,
    /// MA coefficients θ₁, …, θ_q (length q).
    pub ma: Vec<f64>,
    /// Innovation standard deviation σ > 0.
    pub sigma: f64,
    /// Process mean μ.
    pub mean: f64,
    /// Whittle log-likelihood at fitted parameters (None before fitting).
    pub log_likelihood: Option<f64>,
    /// AIC at fitted parameters (None before fitting).
    pub aic: Option<f64>,
    /// BIC at fitted parameters (None before fitting).
    pub bic: Option<f64>,
}

impl ARFIMAModel {
    /// Construct an ARFIMA(p, d, q) model with zero AR/MA coefficients, σ=1, μ=0.
    pub fn new(p: usize, d: f64, q: usize) -> Self {
        Self {
            p,
            d,
            q,
            ar: vec![0.0; p],
            ma: vec![0.0; q],
            sigma: 1.0,
            mean: 0.0,
            log_likelihood: None,
            aic: None,
            bic: None,
        }
    }

    /// Return `(0.0, 0.5)` – the open interval of `d` for which long memory
    /// holds (stationary + long range dependence).
    pub fn long_memory_range() -> (f64, f64) {
        (0.0, 0.5)
    }
}

// ---------------------------------------------------------------------------
// Standalone public API functions (as required by the module spec)
// ---------------------------------------------------------------------------

/// Apply the fractional differencing operator `(1-L)^d` to `series`.
///
/// Returns a series of the same length using a truncated binomial expansion.
/// Weights with `|w_k| < threshold` are dropped (default 1e-9).
///
/// # Arguments
/// * `series` – input time series.
/// * `d`      – differencing parameter (positive → differencing, negative → integration).
/// * `threshold` – truncation threshold for the filter coefficients.
///
/// # Example
/// ```
/// use scirs2_core::ndarray::Array1;
/// use scirs2_series::long_memory::arfima::fractional_difference;
/// let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let dx = fractional_difference(&x, 0.3, 1e-9).expect("should succeed");
/// assert_eq!(dx.len(), 5);
/// ```
pub fn fractional_difference(
    series: &Array1<f64>,
    d: f64,
    threshold: f64,
) -> Result<Array1<f64>> {
    let n = series.len();
    if n == 0 {
        return Err(TimeSeriesError::InsufficientData {
            message: "fractional_difference requires a non-empty series".to_string(),
            required: 1,
            actual: 0,
        });
    }
    let weights = frac_diff_weights_truncated(d, n, threshold.max(f64::EPSILON));
    let out: Vec<f64> = (0..n)
        .map(|t| {
            weights
                .iter()
                .enumerate()
                .filter(|&(k, _)| t >= k)
                .map(|(k, &w)| w * series[t - k])
                .sum::<f64>()
        })
        .collect();
    Ok(Array1::from(out))
}

/// Simulate an ARFIMA(p, d, q) process of length `n`.
///
/// **Algorithm**:
/// 1. Generate `n + burn` innovations `ε_t ~ N(0, σ²)`.
/// 2. Apply the MA and AR filters sequentially.
/// 3. Apply the fractional integration operator `(1-L)^{-d}`.
/// 4. Discard the first `burn` observations to reduce initialisation bias.
///
/// # Arguments
/// * `model` – ARFIMA model parameters.
/// * `n`     – number of output observations.
/// * `seed`  – RNG seed for reproducibility.
pub fn arfima_simulate(model: &ARFIMAModel, n: usize, seed: u64) -> Result<Array1<f64>> {
    if n == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "n".to_string(),
            message: "simulation length must be positive".to_string(),
        });
    }
    if model.d.abs() >= 0.5 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "d".to_string(),
            message: format!(
                "fractional parameter d={} must satisfy |d| < 0.5 for stationarity",
                model.d
            ),
        });
    }
    if model.ar.len() != model.p {
        return Err(TimeSeriesError::InvalidModel(format!(
            "AR coefficient vector length {} does not match p={}",
            model.ar.len(),
            model.p
        )));
    }
    if model.ma.len() != model.q {
        return Err(TimeSeriesError::InvalidModel(format!(
            "MA coefficient vector length {} does not match q={}",
            model.ma.len(),
            model.q
        )));
    }

    let burn = 200_usize.max(n / 4);
    let total = n + burn;
    let mut rng = seeded_rng(seed);
    let innovations: Array1<f64> =
        random_normal_array(Ix1(total), 0.0, model.sigma, &mut rng);

    // Step 1: ARMA filter  φ(L) x_t = θ(L) ε_t
    let mut arma = vec![0.0_f64; total];
    for t in 0..total {
        let mut val = innovations[t];
        for (j, &phi) in model.ar.iter().enumerate() {
            if t > j {
                val += phi * arma[t - j - 1];
            }
        }
        for (j, &theta) in model.ma.iter().enumerate() {
            if t > j {
                val += theta * innovations[t - j - 1];
            }
        }
        arma[t] = val;
    }

    // Step 2: Fractional integration (1-L)^{-d} applied to ARMA output
    let int_weights = frac_diff_weights_truncated(-model.d, total, 1e-10);
    let mut integrated = vec![0.0_f64; total];
    for t in 0..total {
        let mut val = 0.0_f64;
        for (k, &w) in int_weights.iter().enumerate() {
            if t >= k {
                val += w * arma[t - k];
            }
        }
        integrated[t] = val + model.mean;
    }

    let out: Vec<f64> = integrated[burn..].to_vec();
    Ok(Array1::from(out))
}

/// Fit an ARFIMA model to `series` using Whittle maximum likelihood.
///
/// The fractional differencing parameter `d` is estimated by grid search
/// over `(-0.49, 0.49)`, minimising the Whittle contrast function in the
/// frequency domain.  For ARFIMA(0, d, 0) this is exact; for higher-order
/// models it is an approximation.
///
/// # Arguments
/// * `series` – observed time series.
/// * `p`      – AR order.
/// * `q`      – MA order.
///
/// # Returns
/// A fitted `ARFIMAModel` with `log_likelihood`, `aic`, and `bic` populated.
pub fn arfima_fit(series: &Array1<f64>, p: usize, q: usize) -> Result<ARFIMAModel> {
    let n = series.len();
    if n < 20 {
        return Err(TimeSeriesError::InsufficientData {
            message: "ARFIMA fitting requires at least 20 observations".to_string(),
            required: 20,
            actual: n,
        });
    }

    // ---- Step 1: estimate d via Whittle ----
    let d_hat = whittle_d_grid(series)?;

    // ---- Step 2: fractionally difference series ----
    let diff = fractional_difference(series, d_hat, 1e-9)?;

    // ---- Step 3: fit ARMA(p, q) to the differenced series ----
    let (ar_coeffs, ma_coeffs, sigma_hat, arma_ll) =
        fit_arma_yw(&diff, p, q)?;

    // ---- Step 4: compute information criteria ----
    let k_params = (p + q + 2) as f64; // d, sigma, ar, ma
    let aic = -2.0 * arma_ll + 2.0 * k_params;
    let bic = -2.0 * arma_ll + k_params * (n as f64).ln();

    let mut model = ARFIMAModel::new(p, d_hat, q);
    model.ar = ar_coeffs;
    model.ma = ma_coeffs;
    model.sigma = sigma_hat;
    // Estimate mean from original series
    model.mean = series.iter().sum::<f64>() / n as f64;
    model.log_likelihood = Some(arma_ll);
    model.aic = Some(aic);
    model.bic = Some(bic);

    Ok(model)
}

/// Compute h-step-ahead forecasts from a fitted ARFIMA model.
///
/// The forecast leverages the infinite MA representation of the ARFIMA process:
/// `y_{T+h} = Σ_{j=0}^{∞} ψ_j ε_{T+h-j}`.
///
/// In practice, only the known past residuals (j ≥ h) contribute; the future
/// innovations are set to zero.
///
/// # Arguments
/// * `model`  – fitted ARFIMA model.
/// * `series` – the observed series used to compute residuals.
/// * `steps`  – number of steps ahead to forecast.
///
/// # Returns
/// `Array1<f64>` of length `steps` containing point forecasts.
pub fn arfima_forecast(
    model: &ARFIMAModel,
    series: &Array1<f64>,
    steps: usize,
) -> Result<Array1<f64>> {
    let n = series.len();
    if n == 0 {
        return Err(TimeSeriesError::InsufficientData {
            message: "ARFIMA forecast requires a non-empty history".to_string(),
            required: 1,
            actual: 0,
        });
    }
    if steps == 0 {
        return Ok(Array1::from(vec![]));
    }
    if model.d.abs() >= 0.5 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "d".to_string(),
            message: format!("model d={} violates stationarity |d| < 0.5", model.d),
        });
    }

    // ---- Build psi weights of infinite MA rep via recursion ----
    // Combine ARFIMA = ARMA(phi, theta) composed with frac diff (1-L)^d.
    // (1) Compute the psi-weights of the ARFIMA model truncated to max_psi terms.
    //     ψ(L) = φ^{-1}(L) θ(L) (1-L)^{-d}
    //
    // Approach: compute the psi weights recursively by convolving the three
    // filters.  We limit to max_psi = n + steps terms.
    let max_psi = (n + steps).min(2000);

    // (1-L)^{-d} weights (fractional integration)
    let int_weights = frac_diff_weights_truncated(-model.d, max_psi, 1e-12);

    // MA filter: numerator = θ(L) applied to integration weights
    // Convolve integration weights with MA polynomial
    let mut ma_conv = int_weights.clone();
    if !model.ma.is_empty() {
        let mut result = vec![0.0_f64; max_psi];
        for (k, &iw) in int_weights.iter().enumerate() {
            if k < max_psi {
                result[k] += iw;
            }
            for (j, &theta) in model.ma.iter().enumerate() {
                if k + j + 1 < max_psi {
                    result[k + j + 1] += theta * iw;
                }
            }
        }
        ma_conv = result[..max_psi.min(result.len())].to_vec();
    }

    // AR filter: psi_k = ma_conv_k + sum_{j=1}^{p} phi_j * psi_{k-j}
    let mut psi = vec![0.0_f64; max_psi];
    for k in 0..max_psi {
        psi[k] = if k < ma_conv.len() { ma_conv[k] } else { 0.0 };
        for (j, &phi) in model.ar.iter().enumerate() {
            if k > j {
                psi[k] += phi * psi[k - j - 1];
            }
        }
    }

    // ---- Compute past residuals from the observed series ----
    // Step 1: fractionally difference
    let diff_series = fractional_difference(series, model.d, 1e-9)?;

    // Step 2: ARMA residuals on the differenced series
    let mut residuals = vec![0.0_f64; n];
    for t in 0..n {
        let mut r = diff_series[t] - model.mean * frac_diff_weight(model.d, 0);
        for (j, &phi) in model.ar.iter().enumerate() {
            if t > j {
                r -= phi * diff_series[t - j - 1];
            }
        }
        for (j, &theta) in model.ma.iter().enumerate() {
            if t > j {
                r -= theta * residuals[t - j - 1];
            }
        }
        residuals[t] = r;
    }

    // ---- Compute h-step forecasts ----
    // y_{T+h} = mu + Σ_{j=h}^{T+h-1} ψ_j * ε_{T+h-1-j}  (j indexing from 0)
    // where future ε = 0.
    let mut forecasts = vec![0.0_f64; steps];
    for h in 1..=steps {
        let mut fc = model.mean;
        // Contribution from past residuals only (future are zero)
        for j in h..=(n + h - 1) {
            let t_res = n + h - 1 - j; // index into residuals[]
            if t_res < n && j < psi.len() {
                fc += psi[j] * residuals[t_res];
            }
        }
        forecasts[h - 1] = fc;
    }

    Ok(Array1::from(forecasts))
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Whittle MLE grid search for `d ∈ (-0.49, 0.49)`.
///
/// Minimises Σ_j [ I(ω_j) / f(ω_j) + log f(ω_j) ]  where
/// `f(ω) ∝ (4 sin²(ω/2))^{-d}` is the ARFIMA(0,d,0) spectral density.
pub(crate) fn whittle_d_grid(series: &Array1<f64>) -> Result<f64> {
    let n = series.len();
    if n < 10 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Whittle estimation requires ≥ 10 observations".to_string(),
            required: 10,
            actual: n,
        });
    }
    let data: Vec<f64> = series.iter().copied().collect();
    let m_max = n / 2;

    // Pre-compute periodogram ordinates at all Fourier frequencies
    let periodogram: Vec<f64> = (1..=m_max).map(|j| periodogram_at(&data, j)).collect();

    // Two-stage grid search: coarse then fine
    let best_d = two_stage_grid_search(&periodogram, n, -0.49, 0.49, 200)?;

    Ok(best_d)
}

/// Two-stage grid search: coarse grid, then refine in the neighbourhood.
fn two_stage_grid_search(
    periodogram: &[f64],
    n: usize,
    d_lo: f64,
    d_hi: f64,
    n_grid: usize,
) -> Result<f64> {
    // Coarse pass
    let (coarse_best, _) = grid_search_whittle(periodogram, n, d_lo, d_hi, n_grid);

    // Fine pass over ±0.05 neighbourhood
    let fine_lo = (coarse_best - 0.05).max(d_lo);
    let fine_hi = (coarse_best + 0.05).min(d_hi);
    let (fine_best, _) = grid_search_whittle(periodogram, n, fine_lo, fine_hi, 100);

    Ok(fine_best)
}

/// Single-pass Whittle grid search; returns `(best_d, best_contrast)`.
fn grid_search_whittle(
    periodogram: &[f64],
    n: usize,
    d_lo: f64,
    d_hi: f64,
    n_grid: usize,
) -> (f64, f64) {
    let m_max = periodogram.len();
    let mut best_d = 0.0_f64;
    let mut best_contrast = f64::INFINITY;

    for k in 0..=n_grid {
        let d_cand = d_lo + (d_hi - d_lo) * k as f64 / n_grid as f64;
        let mut contrast = 0.0_f64;
        for j in 1..=m_max {
            let omega = 2.0 * PI * j as f64 / n as f64;
            let sin_half = (omega / 2.0).sin();
            let spec = (4.0 * sin_half * sin_half).powf(-d_cand);
            if spec > f64::EPSILON {
                contrast += periodogram[j - 1] / spec + spec.ln();
            }
        }
        if contrast < best_contrast {
            best_contrast = contrast;
            best_d = d_cand;
        }
    }

    (best_d, best_contrast)
}

/// Compute the periodogram at Fourier frequency `j * 2π/n`.
pub(crate) fn periodogram_at(series: &[f64], j: usize) -> f64 {
    let n = series.len() as f64;
    let mean: f64 = series.iter().sum::<f64>() / n;
    let omega = 2.0 * PI * j as f64 / n;
    let (mut re, mut im) = (0.0_f64, 0.0_f64);
    for (t, &v) in series.iter().enumerate() {
        let angle = omega * t as f64;
        re += (v - mean) * angle.cos();
        im += (v - mean) * angle.sin();
    }
    (re * re + im * im) / n
}

/// Fit ARMA(p, q) by Yule-Walker for AR and conditional-sum-of-squares for MA.
/// Returns `(ar_coeffs, ma_coeffs, sigma, log_likelihood)`.
fn fit_arma_yw(
    series: &Array1<f64>,
    p: usize,
    q: usize,
) -> Result<(Vec<f64>, Vec<f64>, f64, f64)> {
    let n = series.len();
    if n < (p + q + 2).max(4) {
        return Err(TimeSeriesError::InsufficientData {
            message: "Too few observations for ARMA fitting".to_string(),
            required: p + q + 2,
            actual: n,
        });
    }

    // Compute sample autocovariances
    let mean = series.iter().sum::<f64>() / n as f64;
    let max_lag = (p + q + 2).min(n - 1);
    let mut acf = vec![0.0_f64; max_lag + 1];
    for lag in 0..=max_lag {
        let mut cov = 0.0_f64;
        for t in lag..n {
            cov += (series[t] - mean) * (series[t - lag] - mean);
        }
        acf[lag] = cov / n as f64;
    }
    let variance = acf[0].max(f64::EPSILON);

    // ---- AR estimation via Yule-Walker ----
    let ar_coeffs = if p > 0 {
        yule_walker_ar(&acf, p)?
    } else {
        vec![]
    };

    // ---- MA estimation: simple moment matching using acf of AR residuals ----
    let mut ar_resid = vec![0.0_f64; n];
    for t in 0..n {
        ar_resid[t] = series[t] - mean;
        for (j, &phi) in ar_coeffs.iter().enumerate() {
            if t > j {
                ar_resid[t] -= phi * (series[t - j - 1] - mean);
            }
        }
    }

    let ma_coeffs = if q > 0 {
        // Simple approximation: use first q autocorrelations of AR residuals
        let mut resid_acf = vec![0.0_f64; q + 1];
        for lag in 0..=q {
            let mut cov = 0.0_f64;
            for t in lag..n {
                cov += ar_resid[t] * ar_resid[t - lag];
            }
            resid_acf[lag] = cov / n as f64;
        }
        // MA(q) moment matching: solve via Durbin's algorithm approximation
        durbin_ma(&resid_acf, q)
    } else {
        vec![]
    };

    // ---- Estimate sigma from residuals ----
    let mut residuals = vec![0.0_f64; n];
    for t in 0..n {
        let mut r = series[t] - mean;
        for (j, &phi) in ar_coeffs.iter().enumerate() {
            if t > j {
                r -= phi * (series[t - j - 1] - mean);
            }
        }
        for (j, &theta) in ma_coeffs.iter().enumerate() {
            if t > j {
                r -= theta * residuals[t - j - 1];
            }
        }
        residuals[t] = r;
    }

    let sigma2 = residuals.iter().map(|r| r * r).sum::<f64>() / n as f64;
    let sigma = sigma2.sqrt().max(f64::EPSILON);

    // Gaussian log-likelihood
    let ll = -0.5 * n as f64 * (2.0 * PI * sigma2).ln() - 0.5 * n as f64;

    Ok((ar_coeffs, ma_coeffs, sigma, ll))
}

/// Yule-Walker AR(p) estimation from autocovariance vector.
fn yule_walker_ar(acf: &[f64], p: usize) -> Result<Vec<f64>> {
    if acf.len() <= p {
        return Err(TimeSeriesError::InsufficientData {
            message: "Insufficient autocovariances for Yule-Walker".to_string(),
            required: p + 1,
            actual: acf.len(),
        });
    }

    // Build Toeplitz system R * phi = r
    let mut mat = vec![vec![0.0_f64; p]; p];
    let mut rhs = vec![0.0_f64; p];
    for i in 0..p {
        rhs[i] = acf[i + 1];
        for j in 0..p {
            let lag = if i >= j { i - j } else { j - i };
            mat[i][j] = acf[lag];
        }
    }

    // Solve via Gaussian elimination with partial pivoting
    solve_linear_system(&mat, &rhs)
}

/// Durbin's approximation for MA(q) coefficients from ACF.
fn durbin_ma(acf: &[f64], q: usize) -> Vec<f64> {
    if q == 0 || acf.len() <= q {
        return vec![];
    }
    let variance = acf[0].max(f64::EPSILON);
    // Simple approximation using the Wold representation relationship
    let mut ma = vec![0.0_f64; q];
    for k in 1..=q {
        if k < acf.len() {
            // Very rough: theta_k ≈ acf[k] / variance
            let r = acf[k] / variance;
            ma[k - 1] = r.clamp(-0.99, 0.99);
        }
    }
    ma
}

/// Solve a p×p linear system A x = b via Gaussian elimination with partial pivot.
pub(crate) fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
    let n = b.len();
    if n == 0 {
        return Ok(vec![]);
    }
    let mut mat: Vec<Vec<f64>> = a.to_vec();
    let mut rhs: Vec<f64> = b.to_vec();

    for col in 0..n {
        // Find pivot row
        let mut max_val = 0.0_f64;
        let mut max_row = col;
        for row in col..n {
            if mat[row][col].abs() > max_val {
                max_val = mat[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return Err(TimeSeriesError::NumericalInstability(
                "Singular or near-singular matrix in linear system".to_string(),
            ));
        }
        mat.swap(col, max_row);
        rhs.swap(col, max_row);

        let pivot = mat[col][col];
        for row in (col + 1)..n {
            let factor = mat[row][col] / pivot;
            for k in col..n {
                let sub = factor * mat[col][k];
                mat[row][k] -= sub;
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    // Back-substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        x[i] = rhs[i];
        for j in (i + 1)..n {
            x[i] -= mat[i][j] * x[j];
        }
        if mat[i][i].abs() < 1e-15 {
            return Err(TimeSeriesError::NumericalInstability(
                "Zero diagonal in back-substitution".to_string(),
            ));
        }
        x[i] /= mat[i][i];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_frac_diff_d0_identity() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let dx = fractional_difference(&x, 0.0, 1e-12)
            .expect("fractional_difference d=0 failed");
        // d=0 → only w_0=1, so output equals input
        assert_eq!(dx.len(), 5);
        assert!((dx[0] - 1.0).abs() < 1e-12);
        assert!((dx[4] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_frac_diff_d1_is_first_difference() {
        // d=1 → (1-L)^1 x_t = x_t - x_{t-1}
        let x = Array1::from(vec![1.0, 3.0, 6.0, 10.0, 15.0]);
        let dx = fractional_difference(&x, 1.0, 1e-12)
            .expect("fractional_difference d=1 failed");
        // dx[0] = x[0], dx[1] = x[1]-x[0] = 2, dx[2] = x[2]-x[1] = 3, ...
        assert!((dx[1] - 2.0).abs() < 1e-9, "d=1 at t=1: got {}", dx[1]);
        assert!((dx[2] - 3.0).abs() < 1e-9, "d=1 at t=2: got {}", dx[2]);
    }

    #[test]
    fn test_arfima_simulate_length() {
        let model = ARFIMAModel::new(0, 0.3, 0);
        let sim = arfima_simulate(&model, 200, 42)
            .expect("arfima_simulate failed");
        assert_eq!(sim.len(), 200);
    }

    #[test]
    fn test_arfima_simulate_finite() {
        let mut model = ARFIMAModel::new(1, 0.4, 1);
        model.ar = vec![0.5];
        model.ma = vec![0.2];
        let sim = arfima_simulate(&model, 100, 99)
            .expect("arfima_simulate (1,0.4,1) failed");
        for &v in sim.iter() {
            assert!(v.is_finite(), "simulation value not finite");
        }
    }

    #[test]
    fn test_arfima_simulate_invalid_d() {
        let model = ARFIMAModel::new(0, 0.6, 0);
        let result = arfima_simulate(&model, 100, 1);
        assert!(result.is_err(), "should fail for |d| >= 0.5");
    }

    #[test]
    fn test_arfima_fit_returns_valid_model() {
        // Generate from known model and check fitted d is in range
        let true_model = ARFIMAModel::new(0, 0.3, 0);
        let sim = arfima_simulate(&true_model, 200, 7)
            .expect("simulate failed");
        let fitted = arfima_fit(&sim, 0, 0).expect("arfima_fit failed");
        assert!(fitted.d.is_finite());
        assert!(fitted.d.abs() < 0.5);
        assert!(fitted.log_likelihood.is_some());
        assert!(fitted.aic.is_some());
    }

    #[test]
    fn test_arfima_forecast_length() {
        let model = ARFIMAModel::new(0, 0.3, 0);
        let series = arfima_simulate(&model, 100, 42).expect("simulate failed");
        let fc = arfima_forecast(&model, &series, 5).expect("forecast failed");
        assert_eq!(fc.len(), 5);
    }

    #[test]
    fn test_arfima_forecast_finite() {
        let model = ARFIMAModel::new(0, 0.2, 0);
        let series = arfima_simulate(&model, 100, 5).expect("simulate failed");
        let fc = arfima_forecast(&model, &series, 10).expect("forecast failed");
        for &v in fc.iter() {
            assert!(v.is_finite(), "forecast value not finite: {}", v);
        }
    }

    #[test]
    fn test_arfima_forecast_zero_steps() {
        let model = ARFIMAModel::new(0, 0.3, 0);
        let series = arfima_simulate(&model, 50, 1).expect("simulate failed");
        let fc = arfima_forecast(&model, &series, 0).expect("forecast 0 steps failed");
        assert_eq!(fc.len(), 0);
    }
}
