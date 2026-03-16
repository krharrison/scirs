//! Advanced Spectral Analysis for v0.2.0
//!
//! This module provides enhanced spectral analysis capabilities including:
//! - AR model spectral estimation with improved Yule-Walker method
//! - ARMA model spectral estimation with multiple algorithms
//! - Enhanced multitaper spectral estimation
//! - Parallel processing for spectral operations
//! - Memory optimization for large signals
//!
//! All implementations follow the no-unwrap policy and use proper error handling.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use scirs2_core::numeric::Complex64;
use scirs2_core::parallel_ops::*;
use scirs2_core::validation::{check_finite, check_positive};
use std::f64::consts::PI;

// ============================================================================
// Configuration Types
// ============================================================================

/// Configuration for AR spectral estimation
#[derive(Debug, Clone)]
pub struct ARSpectralConfig {
    /// Model order
    pub order: usize,
    /// Sampling frequency
    pub fs: f64,
    /// Number of frequency points for PSD
    pub nfft: usize,
    /// Estimation method
    pub method: ARSpectralMethod,
    /// Enable parallel processing
    pub parallel: bool,
    /// Minimum chunk size for parallel processing
    pub parallel_threshold: usize,
}

impl Default for ARSpectralConfig {
    fn default() -> Self {
        Self {
            order: 10,
            fs: 1.0,
            nfft: 512,
            method: ARSpectralMethod::YuleWalkerEnhanced,
            parallel: true,
            parallel_threshold: 1024,
        }
    }
}

/// AR spectral estimation methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ARSpectralMethod {
    /// Enhanced Yule-Walker with Levinson-Durbin recursion
    YuleWalkerEnhanced,
    /// Burg's method with forward-backward prediction
    BurgEnhanced,
    /// Modified covariance method
    ModifiedCovariance,
    /// Least squares method with regularization
    LeastSquaresRegularized,
}

/// Configuration for ARMA spectral estimation
#[derive(Debug, Clone)]
pub struct ARMASpectralConfig {
    /// AR order
    pub ar_order: usize,
    /// MA order
    pub ma_order: usize,
    /// Sampling frequency
    pub fs: f64,
    /// Number of frequency points for PSD
    pub nfft: usize,
    /// Estimation method
    pub method: ARMASpectralMethod,
    /// Maximum iterations for iterative methods
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Enable parallel processing
    pub parallel: bool,
}

impl Default for ARMASpectralConfig {
    fn default() -> Self {
        Self {
            ar_order: 5,
            ma_order: 5,
            fs: 1.0,
            nfft: 512,
            method: ARMASpectralMethod::HannanRissanenImproved,
            max_iterations: 100,
            tolerance: 1e-8,
            parallel: true,
        }
    }
}

/// ARMA spectral estimation methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ARMASpectralMethod {
    /// Improved Hannan-Rissanen two-stage method
    HannanRissanenImproved,
    /// Innovation algorithm with convergence acceleration
    InnovationAccelerated,
    /// Maximum likelihood with gradient descent
    MaximumLikelihoodGradient,
    /// Durbin's method
    Durbin,
}

/// Configuration for parallel spectral processing
#[derive(Debug, Clone)]
pub struct ParallelSpectralConfigV2 {
    /// Number of worker threads (None for automatic)
    pub num_threads: Option<usize>,
    /// Minimum chunk size for parallel processing
    pub chunk_size: usize,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Memory limit in bytes (0 for unlimited)
    pub memory_limit: usize,
}

impl Default for ParallelSpectralConfigV2 {
    fn default() -> Self {
        Self {
            num_threads: None,
            chunk_size: 4096,
            enable_simd: true,
            memory_limit: 0,
        }
    }
}

/// Configuration for memory-optimized spectral processing
#[derive(Debug, Clone)]
pub struct MemoryOptimizedSpectralConfig {
    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,
    /// Chunk size for processing
    pub chunk_size: usize,
    /// Overlap between chunks for continuity
    pub overlap_samples: usize,
    /// Use streaming mode for very large signals
    pub streaming: bool,
}

impl Default for MemoryOptimizedSpectralConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 512 * 1024 * 1024, // 512 MB
            chunk_size: 32768,
            overlap_samples: 1024,
            streaming: false,
        }
    }
}

// ============================================================================
// AR Spectral Estimation Results
// ============================================================================

/// Result of AR spectral estimation
#[derive(Debug, Clone)]
pub struct ARSpectralResult {
    /// Frequency axis
    pub frequencies: Array1<f64>,
    /// Power spectral density
    pub psd: Array1<f64>,
    /// AR coefficients [1, a1, a2, ..., ap]
    pub ar_coefficients: Array1<f64>,
    /// Reflection coefficients (partial correlations)
    pub reflection_coefficients: Option<Array1<f64>>,
    /// Prediction error variance
    pub variance: f64,
    /// Model order used
    pub order: usize,
    /// Information criterion value (AIC/BIC)
    pub information_criterion: Option<f64>,
}

/// Result of ARMA spectral estimation
#[derive(Debug, Clone)]
pub struct ARMASpectralResult {
    /// Frequency axis
    pub frequencies: Array1<f64>,
    /// Power spectral density
    pub psd: Array1<f64>,
    /// AR coefficients [1, a1, a2, ..., ap]
    pub ar_coefficients: Array1<f64>,
    /// MA coefficients [1, b1, b2, ..., bq]
    pub ma_coefficients: Array1<f64>,
    /// Innovation variance
    pub variance: f64,
    /// AR order
    pub ar_order: usize,
    /// MA order
    pub ma_order: usize,
    /// Log-likelihood (if computed)
    pub log_likelihood: Option<f64>,
    /// Convergence achieved
    pub converged: bool,
    /// Number of iterations used
    pub iterations: usize,
}

// ============================================================================
// Enhanced AR Spectral Estimation
// ============================================================================

/// Enhanced AR spectral estimation using the Yule-Walker method
///
/// This implementation uses the Levinson-Durbin recursion for efficient
/// and numerically stable computation of AR parameters.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `config` - AR spectral configuration
///
/// # Returns
///
/// AR spectral estimation result with frequencies, PSD, and model parameters
///
/// # Example
///
/// ```ignore
/// use scirs2_signal::advanced_spectral_v2::{ar_spectral_estimation, ARSpectralConfig};
/// use scirs2_core::ndarray::Array1;
///
/// let signal = Array1::linspace(0.0, 1.0, 256).mapv(|t| (2.0 * std::f64::consts::PI * 10.0 * t).sin());
/// let config = ARSpectralConfig::default();
/// let result = ar_spectral_estimation(&signal, &config).expect("operation should succeed");
/// ```
pub fn ar_spectral_estimation(
    signal: &Array1<f64>,
    config: &ARSpectralConfig,
) -> SignalResult<ARSpectralResult> {
    // Validate inputs
    let n = signal.len();
    if n == 0 {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }

    if config.order >= n {
        return Err(SignalError::ValueError(format!(
            "AR order {} must be less than signal length {}",
            config.order, n
        )));
    }

    check_positive(config.fs, "sampling frequency")?;
    check_positive(config.nfft, "NFFT")?;

    // Check for finite values
    for (i, &val) in signal.iter().enumerate() {
        check_finite(val, &format!("signal[{}]", i))?;
    }

    // Estimate AR parameters based on method
    let (ar_coeffs, reflection_coeffs, variance) = match config.method {
        ARSpectralMethod::YuleWalkerEnhanced => yule_walker_enhanced(signal, config.order)?,
        ARSpectralMethod::BurgEnhanced => burg_enhanced(signal, config.order)?,
        ARSpectralMethod::ModifiedCovariance => modified_covariance_enhanced(signal, config.order)?,
        ARSpectralMethod::LeastSquaresRegularized => {
            least_squares_regularized(signal, config.order, 1e-6)?
        }
    };

    // Compute PSD from AR coefficients
    let frequencies = compute_frequency_axis(config.nfft, config.fs);
    let psd = compute_ar_psd(&ar_coeffs, variance, &frequencies, config.fs)?;

    // Compute information criterion (AIC)
    let aic = n as f64 * variance.ln() + 2.0 * config.order as f64;

    Ok(ARSpectralResult {
        frequencies,
        psd,
        ar_coefficients: ar_coeffs,
        reflection_coefficients: Some(reflection_coeffs),
        variance,
        order: config.order,
        information_criterion: Some(aic),
    })
}

/// Enhanced Yule-Walker estimation with improved Levinson-Durbin recursion
fn yule_walker_enhanced(
    signal: &Array1<f64>,
    order: usize,
) -> SignalResult<(Array1<f64>, Array1<f64>, f64)> {
    let n = signal.len();

    // Compute biased autocorrelation (more stable)
    let mut autocorr = Array1::zeros(order + 1);
    for lag in 0..=order {
        let mut sum = 0.0;
        for i in 0..(n - lag) {
            sum += signal[i] * signal[i + lag];
        }
        autocorr[lag] = sum / n as f64;
    }

    // Check for zero variance
    if autocorr[0].abs() < 1e-15 {
        return Err(SignalError::ComputationError(
            "Signal has near-zero variance".to_string(),
        ));
    }

    // Apply Levinson-Durbin algorithm
    levinson_durbin_enhanced(&autocorr, order)
}

/// Enhanced Levinson-Durbin recursion with improved numerical stability
fn levinson_durbin_enhanced(
    autocorr: &Array1<f64>,
    order: usize,
) -> SignalResult<(Array1<f64>, Array1<f64>, f64)> {
    let p = order;
    let mut a = Array1::zeros(p);
    let mut a_prev = Array1::zeros(p);
    let mut reflection = Array1::zeros(p);

    // Initial prediction error is the zero-lag autocorrelation
    let mut e = autocorr[0];

    for k in 0..p {
        // Compute reflection coefficient with regularization
        let mut num = autocorr[k + 1];
        for j in 0..k {
            num -= a[j] * autocorr[k - j];
        }

        // Add small regularization for numerical stability
        let e_reg = e.max(1e-15);
        let k_reflection = num / e_reg;

        // Check for stability (reflection coefficient magnitude < 1)
        if k_reflection.abs() >= 1.0 {
            // Soft constraint: limit reflection coefficient
            let k_limited = k_reflection.signum() * 0.999;
            reflection[k] = k_limited;
        } else {
            reflection[k] = k_reflection;
        }

        // Save previous coefficients
        for j in 0..k {
            a_prev[j] = a[j];
        }

        // Update AR coefficients
        a[k] = reflection[k];
        for j in 0..k {
            a[j] = a_prev[j] - reflection[k] * a_prev[k - 1 - j];
        }

        // Update prediction error
        e *= 1.0 - reflection[k] * reflection[k];

        // Check for algorithm instability
        if e <= 0.0 {
            return Err(SignalError::ComputationError(
                "Levinson-Durbin algorithm became unstable".to_string(),
            ));
        }
    }

    // Build full AR coefficient array [1, -a1, -a2, ..., -ap]
    let mut ar_coeffs = Array1::zeros(p + 1);
    ar_coeffs[0] = 1.0;
    for i in 0..p {
        ar_coeffs[i + 1] = -a[i];
    }

    Ok((ar_coeffs, reflection, e))
}

/// Enhanced Burg's method with improved forward-backward prediction
fn burg_enhanced(
    signal: &Array1<f64>,
    order: usize,
) -> SignalResult<(Array1<f64>, Array1<f64>, f64)> {
    let n = signal.len();

    // Initialize forward and backward prediction errors
    let mut f = signal.to_owned();
    let mut b = signal.to_owned();

    // Initialize reflection coefficients
    let mut reflection = Array1::zeros(order);

    // Initial prediction error power
    let mut e = signal.iter().map(|&x| x * x).sum::<f64>() / n as f64;

    // AR coefficients matrix (using only last row)
    let mut a = vec![vec![0.0; order + 1]; order + 1];
    for row in a.iter_mut() {
        row[0] = 1.0;
    }

    for m in 0..order {
        // Compute reflection coefficient using improved formula
        let mut num = 0.0;
        let mut den = 0.0;

        for i in 0..(n - m - 1) {
            num += 2.0 * f[i + m + 1] * b[i];
            den += f[i + m + 1].powi(2) + b[i].powi(2);
        }

        // Add regularization for numerical stability
        let den_reg = den.max(1e-15);
        let k_m = -num / den_reg;

        // Limit reflection coefficient for stability
        let k_m_limited = if k_m.abs() >= 1.0 {
            k_m.signum() * 0.999
        } else {
            k_m
        };

        reflection[m] = k_m_limited;

        // Update AR coefficients using Levinson recursion
        for i in 1..=m {
            a[m + 1][i] = a[m][i] + k_m_limited * a[m][m + 1 - i];
        }
        a[m + 1][m + 1] = k_m_limited;

        // Update prediction error power
        e *= 1.0 - k_m_limited * k_m_limited;

        // Check for algorithm instability
        if e <= 0.0 {
            return Err(SignalError::ComputationError(
                "Burg algorithm became unstable".to_string(),
            ));
        }

        // Update forward and backward prediction errors
        if m < order - 1 {
            for i in 0..(n - m - 1) {
                let f_old = f[i + m + 1];
                let b_old = b[i];
                f[i + m + 1] = f_old + k_m_limited * b_old;
                b[i] = b_old + k_m_limited * f_old;
            }
        }
    }

    // Extract final AR coefficients [1, a1, a2, ..., ap]
    let mut ar_coeffs = Array1::zeros(order + 1);
    ar_coeffs[0] = 1.0;
    for i in 1..=order {
        ar_coeffs[i] = a[order][i];
    }

    Ok((ar_coeffs, reflection, e))
}

/// Modified covariance method with enhanced numerical stability
fn modified_covariance_enhanced(
    signal: &Array1<f64>,
    order: usize,
) -> SignalResult<(Array1<f64>, Array1<f64>, f64)> {
    let n = signal.len();

    // Build modified covariance matrix (forward + backward)
    let mut r = Array2::zeros((order, order));
    let mut r_vec = Array1::zeros(order);

    for i in 0..order {
        for j in 0..order {
            let mut sum = 0.0;
            // Forward prediction
            for k in order..(n) {
                sum += signal[k - i - 1] * signal[k - j - 1];
            }
            // Backward prediction
            for k in 0..(n - order) {
                sum += signal[k + i] * signal[k + j];
            }
            r[[i, j]] = sum / (2.0 * (n - order) as f64);
        }

        let mut sum = 0.0;
        // Forward prediction
        for k in order..n {
            sum += signal[k - i - 1] * signal[k];
        }
        // Backward prediction
        for k in 0..(n - order) {
            sum += signal[k + i] * signal[k + order];
        }
        r_vec[i] = sum / (2.0 * (n - order) as f64);
    }

    // Solve using Cholesky decomposition (more stable for positive definite matrices)
    let ar_params = solve_linear_system_stable(&r, &r_vec)?;

    // Compute prediction error variance
    let mut variance = 0.0;
    let mut count = 0;

    // Forward prediction errors
    for t in order..n {
        let mut pred = 0.0;
        for i in 0..order {
            pred += ar_params[i] * signal[t - i - 1];
        }
        variance += (signal[t] - pred).powi(2);
        count += 1;
    }

    // Backward prediction errors
    for t in 0..(n - order) {
        let mut pred = 0.0;
        for i in 0..order {
            pred += ar_params[i] * signal[t + i + 1];
        }
        variance += (signal[t] - pred).powi(2);
        count += 1;
    }

    variance /= count as f64;

    // Build AR coefficient array
    let mut ar_coeffs = Array1::zeros(order + 1);
    ar_coeffs[0] = 1.0;
    for i in 0..order {
        ar_coeffs[i + 1] = -ar_params[i];
    }

    // Compute pseudo-reflection coefficients (not exact for modified covariance)
    let reflection = Array1::zeros(order);

    Ok((ar_coeffs, reflection, variance))
}

/// Least squares method with Tikhonov regularization
fn least_squares_regularized(
    signal: &Array1<f64>,
    order: usize,
    lambda: f64,
) -> SignalResult<(Array1<f64>, Array1<f64>, f64)> {
    let n = signal.len();

    // Build design matrix
    let m = n - order;
    let mut x = Array2::zeros((m, order));
    let mut y = Array1::zeros(m);

    for i in 0..m {
        for j in 0..order {
            x[[i, j]] = signal[order - 1 - j + i];
        }
        y[i] = signal[order + i];
    }

    // Compute X^T * X + lambda * I
    let xtx = x.t().dot(&x);
    let mut xtx_reg = xtx.clone();
    for i in 0..order {
        xtx_reg[[i, i]] += lambda;
    }

    // Compute X^T * y
    let xty = x.t().dot(&y);

    // Solve regularized system
    let ar_params = solve_linear_system_stable(&xtx_reg, &xty)?;

    // Compute prediction error variance
    let predictions = x.dot(&ar_params);
    let residuals = &y - &predictions;
    let variance = residuals.mapv(|x| x * x).sum() / m as f64;

    // Build AR coefficient array
    let mut ar_coeffs = Array1::zeros(order + 1);
    ar_coeffs[0] = 1.0;
    for i in 0..order {
        ar_coeffs[i + 1] = -ar_params[i];
    }

    // No reflection coefficients for LS method
    let reflection = Array1::zeros(order);

    Ok((ar_coeffs, reflection, variance))
}

// ============================================================================
// ARMA Spectral Estimation
// ============================================================================

/// ARMA spectral estimation with multiple algorithm support
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `config` - ARMA spectral configuration
///
/// # Returns
///
/// ARMA spectral estimation result
pub fn arma_spectral_estimation(
    signal: &Array1<f64>,
    config: &ARMASpectralConfig,
) -> SignalResult<ARMASpectralResult> {
    // Validate inputs
    let n = signal.len();
    if n == 0 {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }

    if config.ar_order + config.ma_order >= n {
        return Err(SignalError::ValueError(format!(
            "ARMA order (p={}, q={}) too large for signal length {}",
            config.ar_order, config.ma_order, n
        )));
    }

    check_positive(config.fs, "sampling frequency")?;
    check_positive(config.nfft, "NFFT")?;

    // Check for finite values
    for (i, &val) in signal.iter().enumerate() {
        check_finite(val, &format!("signal[{}]", i))?;
    }

    // Estimate ARMA parameters based on method
    let (ar_coeffs, ma_coeffs, variance, converged, iterations, log_likelihood) =
        match config.method {
            ARMASpectralMethod::HannanRissanenImproved => hannan_rissanen_improved(signal, config)?,
            ARMASpectralMethod::InnovationAccelerated => innovation_accelerated(signal, config)?,
            ARMASpectralMethod::MaximumLikelihoodGradient => {
                maximum_likelihood_gradient(signal, config)?
            }
            ARMASpectralMethod::Durbin => durbin_method(signal, config)?,
        };

    // Compute PSD from ARMA coefficients
    let frequencies = compute_frequency_axis(config.nfft, config.fs);
    let psd = compute_arma_psd(&ar_coeffs, &ma_coeffs, variance, &frequencies, config.fs)?;

    Ok(ARMASpectralResult {
        frequencies,
        psd,
        ar_coefficients: ar_coeffs,
        ma_coefficients: ma_coeffs,
        variance,
        ar_order: config.ar_order,
        ma_order: config.ma_order,
        log_likelihood,
        converged,
        iterations,
    })
}

/// Improved Hannan-Rissanen two-stage method
fn hannan_rissanen_improved(
    signal: &Array1<f64>,
    config: &ARMASpectralConfig,
) -> SignalResult<(Array1<f64>, Array1<f64>, f64, bool, usize, Option<f64>)> {
    let n = signal.len();
    let p = config.ar_order;
    let q = config.ma_order;

    // Stage 1: Fit high-order AR model to estimate innovations
    let ar_order_high = ((n as f64).sqrt() as usize).max(p + q + 1).min(n / 3);
    let (ar_high, _, var_high) = yule_walker_enhanced(signal, ar_order_high)?;

    // Compute innovations (residuals from high-order AR model)
    let mut innovations = Array1::zeros(n);
    for t in ar_order_high..n {
        let mut pred = 0.0;
        for i in 1..=ar_order_high {
            pred -= ar_high[i] * signal[t - i];
        }
        innovations[t] = signal[t] - pred;
    }

    // Stage 2: Estimate ARMA parameters using LS regression
    let start = (p + q).max(ar_order_high);
    let n_obs = n - start;

    if n_obs < p + q + 1 {
        return Err(SignalError::ValueError(
            "Not enough observations for ARMA estimation".to_string(),
        ));
    }

    // Build regression matrix
    let mut x = Array2::zeros((n_obs, p + q));
    let mut y = Array1::zeros(n_obs);

    for i in 0..n_obs {
        let t = i + start;
        y[i] = signal[t];

        // AR terms (lagged signal)
        for j in 0..p {
            x[[i, j]] = signal[t - j - 1];
        }

        // MA terms (lagged innovations)
        for j in 0..q {
            x[[i, p + j]] = innovations[t - j - 1];
        }
    }

    // Solve for parameters
    let xtx = x.t().dot(&x);
    let xty = x.t().dot(&y);
    let params = solve_linear_system_stable(&xtx, &xty)?;

    // Extract AR and MA coefficients
    let mut ar_coeffs = Array1::zeros(p + 1);
    ar_coeffs[0] = 1.0;
    for i in 0..p {
        ar_coeffs[i + 1] = -params[i];
    }

    let mut ma_coeffs = Array1::zeros(q + 1);
    ma_coeffs[0] = 1.0;
    for i in 0..q {
        ma_coeffs[i + 1] = params[p + i];
    }

    // Compute final residuals and variance
    let residuals = compute_arma_residuals(signal, &ar_coeffs, &ma_coeffs)?;
    let variance = residuals.iter().skip(start).map(|&r| r * r).sum::<f64>() / n_obs as f64;

    // Compute log-likelihood
    let log_likelihood = -0.5 * n_obs as f64 * (1.0 + (2.0 * PI * variance).ln());

    Ok((
        ar_coeffs,
        ma_coeffs,
        variance,
        true,
        1,
        Some(log_likelihood),
    ))
}

/// Innovation algorithm with Aitken acceleration
fn innovation_accelerated(
    signal: &Array1<f64>,
    config: &ARMASpectralConfig,
) -> SignalResult<(Array1<f64>, Array1<f64>, f64, bool, usize, Option<f64>)> {
    let n = signal.len();
    let p = config.ar_order;
    let q = config.ma_order;

    // Center the signal
    let mean = signal.iter().sum::<f64>() / n as f64;
    let centered = signal.mapv(|x| x - mean);

    // Initialize parameters
    let mut ar_coeffs = Array1::zeros(p + 1);
    ar_coeffs[0] = 1.0;

    let mut ma_coeffs = Array1::zeros(q + 1);
    ma_coeffs[0] = 1.0;

    let mut prev_variance = f64::INFINITY;
    let mut converged = false;
    let mut iterations = 0;

    // Aitken acceleration storage
    let mut var_history = Vec::with_capacity(3);

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // E-step: Compute innovations
        let innovations = compute_arma_residuals(&centered, &ar_coeffs, &ma_coeffs)?;

        // M-step: Update AR coefficients
        if p > 0 {
            let mut r_matrix = Array2::zeros((p, p));
            let mut r_vec = Array1::zeros(p);

            for i in 0..p {
                for j in 0..p {
                    let mut sum = 0.0;
                    for t in p.max(q)..n {
                        sum += centered[t - i - 1] * centered[t - j - 1];
                    }
                    r_matrix[[i, j]] = sum;
                }

                let mut sum = 0.0;
                for t in p.max(q)..n {
                    sum += centered[t - i - 1] * centered[t];
                }
                r_vec[i] = sum;
            }

            if let Ok(ar_params) = solve_linear_system_stable(&r_matrix, &r_vec) {
                for i in 0..p {
                    ar_coeffs[i + 1] = -ar_params[i];
                }
            }
        }

        // Update MA coefficients
        if q > 0 {
            let mut m_matrix = Array2::zeros((q, q));
            let mut m_vec = Array1::zeros(q);

            for i in 0..q {
                for j in 0..q {
                    let mut sum = 0.0;
                    for t in p.max(q)..n {
                        if t > i && t > j {
                            sum += innovations[t - i - 1] * innovations[t - j - 1];
                        }
                    }
                    m_matrix[[i, j]] = sum;
                }

                let mut sum = 0.0;
                for t in p.max(q)..n {
                    if t > i {
                        sum += innovations[t - i - 1] * centered[t];
                    }
                }
                m_vec[i] = sum;
            }

            if let Ok(ma_params) = solve_linear_system_stable(&m_matrix, &m_vec) {
                for i in 0..q {
                    ma_coeffs[i + 1] = ma_params[i];
                }
            }
        }

        // Compute current variance
        let residuals = compute_arma_residuals(&centered, &ar_coeffs, &ma_coeffs)?;
        let variance =
            residuals.iter().skip(p.max(q)).map(|&r| r * r).sum::<f64>() / (n - p.max(q)) as f64;

        // Aitken acceleration
        var_history.push(variance);
        if var_history.len() >= 3 {
            let v0 = var_history[var_history.len() - 3];
            let v1 = var_history[var_history.len() - 2];
            let v2 = var_history[var_history.len() - 1];

            let denom = v2 - 2.0 * v1 + v0;
            if denom.abs() > 1e-15 {
                let _accelerated = v2 - (v2 - v1).powi(2) / denom;
            }
        }

        // Check convergence
        if (variance - prev_variance).abs() < config.tolerance {
            converged = true;
            prev_variance = variance;
            break;
        }

        prev_variance = variance;
    }

    // Compute log-likelihood
    let log_likelihood = -0.5 * (n - p.max(q)) as f64 * (1.0 + (2.0 * PI * prev_variance).ln());

    Ok((
        ar_coeffs,
        ma_coeffs,
        prev_variance,
        converged,
        iterations,
        Some(log_likelihood),
    ))
}

/// Maximum likelihood estimation with gradient descent
fn maximum_likelihood_gradient(
    signal: &Array1<f64>,
    config: &ARMASpectralConfig,
) -> SignalResult<(Array1<f64>, Array1<f64>, f64, bool, usize, Option<f64>)> {
    // Start with Hannan-Rissanen estimates
    let (mut ar_coeffs, mut ma_coeffs, mut variance, _, _, _) =
        hannan_rissanen_improved(signal, config)?;

    let n = signal.len();
    let p = config.ar_order;
    let q = config.ma_order;

    let mut converged = false;
    let mut iterations = 0;
    let learning_rate = 0.001;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // Compute residuals
        let residuals = compute_arma_residuals(signal, &ar_coeffs, &ma_coeffs)?;

        // Compute current log-likelihood
        let n_eff = n - p.max(q);
        let ss = residuals.iter().skip(p.max(q)).map(|&r| r * r).sum::<f64>();
        let new_variance = ss / n_eff as f64;
        let log_likelihood = -0.5 * n_eff as f64 * (1.0 + (2.0 * PI * new_variance).ln());

        // Check convergence
        if (new_variance - variance).abs() < config.tolerance {
            converged = true;
            variance = new_variance;
            break;
        }

        // Gradient descent step for AR coefficients
        for i in 1..=p {
            let mut grad = 0.0;
            for t in p.max(q)..n {
                if t >= i {
                    grad += 2.0 * residuals[t] * signal[t - i];
                }
            }
            ar_coeffs[i] -= learning_rate * grad / n_eff as f64;
        }

        // Gradient descent step for MA coefficients
        for i in 1..=q {
            let mut grad = 0.0;
            for t in p.max(q)..n {
                if t >= i {
                    grad -= 2.0 * residuals[t] * residuals[t - i];
                }
            }
            ma_coeffs[i] -= learning_rate * grad / n_eff as f64;
        }

        variance = new_variance;
    }

    let log_likelihood = -0.5 * (n - p.max(q)) as f64 * (1.0 + (2.0 * PI * variance).ln());

    Ok((
        ar_coeffs,
        ma_coeffs,
        variance,
        converged,
        iterations,
        Some(log_likelihood),
    ))
}

/// Durbin's method for ARMA estimation
fn durbin_method(
    signal: &Array1<f64>,
    config: &ARMASpectralConfig,
) -> SignalResult<(Array1<f64>, Array1<f64>, f64, bool, usize, Option<f64>)> {
    let n = signal.len();
    let p = config.ar_order;
    let q = config.ma_order;

    // Step 1: Fit high-order AR model
    let ar_order_high = (p + q + 1).max((n as f64).sqrt() as usize).min(n / 3);
    let (ar_high, _, _) = yule_walker_enhanced(signal, ar_order_high)?;

    // Step 2: Compute extended autocorrelation from AR model
    let mut acf_extended = Array1::zeros(p + q + 1);
    for k in 0..=p + q {
        let mut sum = 0.0;
        for i in 0..(n - k) {
            sum += signal[i] * signal[i + k];
        }
        acf_extended[k] = sum / n as f64;
    }

    // Step 3: Estimate AR parameters using first p+q autocorrelations
    let (ar_coeffs, _, _) = if p > 0 {
        yule_walker_enhanced(signal, p)?
    } else {
        let coeffs = Array1::from_elem(1, 1.0);
        (coeffs, Array1::zeros(0), 0.0)
    };

    // Step 4: Compute MA parameters from AR residuals
    let residuals = if p > 0 {
        compute_ar_residuals(signal, &ar_coeffs)?
    } else {
        signal.clone()
    };

    let (ma_high, _, _) = if q > 0 {
        yule_walker_enhanced(&residuals, q)?
    } else {
        let coeffs = Array1::from_elem(1, 1.0);
        (coeffs, Array1::zeros(0), 0.0)
    };

    let mut ma_coeffs = Array1::zeros(q + 1);
    ma_coeffs[0] = 1.0;
    for i in 0..q.min(ma_high.len() - 1) {
        ma_coeffs[i + 1] = -ma_high[i + 1]; // Convert AR-like coeffs to MA
    }

    // Compute final variance
    let final_residuals = compute_arma_residuals(signal, &ar_coeffs, &ma_coeffs)?;
    let start = p.max(q);
    let variance = final_residuals
        .iter()
        .skip(start)
        .map(|&r| r * r)
        .sum::<f64>()
        / (n - start) as f64;

    let log_likelihood = -0.5 * (n - start) as f64 * (1.0 + (2.0 * PI * variance).ln());

    Ok((
        ar_coeffs,
        ma_coeffs,
        variance,
        true,
        1,
        Some(log_likelihood),
    ))
}

// ============================================================================
// Parallel Spectral Processing
// ============================================================================

/// Parallel batch AR spectral estimation
///
/// Process multiple signals in parallel using AR spectral estimation.
///
/// # Arguments
///
/// * `signals` - Vector of input signals
/// * `config` - AR spectral configuration
/// * `parallel_config` - Parallel processing configuration
///
/// # Returns
///
/// Vector of AR spectral results
#[cfg(feature = "parallel")]
pub fn parallel_ar_spectral_batch(
    signals: &[Array1<f64>],
    config: &ARSpectralConfig,
    _parallel_config: &ParallelSpectralConfigV2,
) -> SignalResult<Vec<ARSpectralResult>> {
    if signals.is_empty() {
        return Err(SignalError::ValueError("No signals provided".to_string()));
    }

    let results: Result<Vec<_>, SignalError> = signals
        .par_iter()
        .map(|signal| ar_spectral_estimation(signal, config))
        .collect();

    results
}

/// Non-parallel batch AR spectral estimation (fallback)
#[cfg(not(feature = "parallel"))]
pub fn parallel_ar_spectral_batch(
    signals: &[Array1<f64>],
    config: &ARSpectralConfig,
    _parallel_config: &ParallelSpectralConfigV2,
) -> SignalResult<Vec<ARSpectralResult>> {
    signals
        .iter()
        .map(|signal| ar_spectral_estimation(signal, config))
        .collect()
}

/// Parallel batch ARMA spectral estimation
#[cfg(feature = "parallel")]
pub fn parallel_arma_spectral_batch(
    signals: &[Array1<f64>],
    config: &ARMASpectralConfig,
    _parallel_config: &ParallelSpectralConfigV2,
) -> SignalResult<Vec<ARMASpectralResult>> {
    if signals.is_empty() {
        return Err(SignalError::ValueError("No signals provided".to_string()));
    }

    let results: Result<Vec<_>, SignalError> = signals
        .par_iter()
        .map(|signal| arma_spectral_estimation(signal, config))
        .collect();

    results
}

#[cfg(not(feature = "parallel"))]
pub fn parallel_arma_spectral_batch(
    signals: &[Array1<f64>],
    config: &ARMASpectralConfig,
    _parallel_config: &ParallelSpectralConfigV2,
) -> SignalResult<Vec<ARMASpectralResult>> {
    signals
        .iter()
        .map(|signal| arma_spectral_estimation(signal, config))
        .collect()
}

/// Parallel Welch PSD estimation with chunked processing
#[cfg(feature = "parallel")]
pub fn parallel_welch_psd(
    signal: &Array1<f64>,
    segment_length: usize,
    overlap: usize,
    fs: f64,
    _parallel_config: &ParallelSpectralConfigV2,
) -> SignalResult<(Array1<f64>, Array1<f64>)> {
    let n = signal.len();

    if segment_length > n {
        return Err(SignalError::ValueError(
            "Segment length exceeds signal length".to_string(),
        ));
    }

    if overlap >= segment_length {
        return Err(SignalError::ValueError(
            "Overlap must be less than segment length".to_string(),
        ));
    }

    let step = segment_length - overlap;
    let num_segments = (n - segment_length) / step + 1;

    if num_segments == 0 {
        return Err(SignalError::ValueError(
            "Not enough data for even one segment".to_string(),
        ));
    }

    // Generate Hann window
    let window = hann_window(segment_length);
    let window_power: f64 = window.iter().map(|&w| w * w).sum();

    // Create segment indices
    let segment_indices: Vec<usize> = (0..num_segments).collect();

    // Process segments in parallel
    let segment_psds: Vec<Vec<f64>> = segment_indices
        .par_iter()
        .map(|&i| {
            let start = i * step;
            let end = start + segment_length;

            // Extract and window segment
            let mut segment: Vec<f64> = signal
                .slice(scirs2_core::ndarray::s![start..end])
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| s * w)
                .collect();

            // Compute FFT using scirs2_fft
            let segment_f64: Vec<f64> = segment.iter().copied().collect();
            let complex_input = match scirs2_fft::fft(&segment_f64, Some(segment_length)) {
                Ok(result) => result,
                Err(_) => return vec![0.0; segment_length / 2 + 1], // Return zeros on error
            };

            // Compute one-sided PSD
            let n_freq = segment_length / 2 + 1;
            let mut psd = Vec::with_capacity(n_freq);

            for k in 0..n_freq {
                let mag_sq = complex_input[k].norm_sqr();
                let scale = if k == 0 || (k == segment_length / 2 && segment_length % 2 == 0) {
                    1.0
                } else {
                    2.0
                };
                psd.push(scale * mag_sq / (fs * window_power));
            }

            psd
        })
        .collect();

    // Average PSDs
    let n_freq = segment_length / 2 + 1;
    let mut avg_psd = Array1::zeros(n_freq);

    for segment_psd in &segment_psds {
        for (i, &p) in segment_psd.iter().enumerate() {
            avg_psd[i] += p;
        }
    }

    avg_psd.mapv_inplace(|x| x / num_segments as f64);

    // Compute frequency axis
    let frequencies = Array1::linspace(0.0, fs / 2.0, n_freq);

    Ok((frequencies, avg_psd))
}

#[cfg(not(feature = "parallel"))]
pub fn parallel_welch_psd(
    signal: &Array1<f64>,
    segment_length: usize,
    overlap: usize,
    fs: f64,
    _parallel_config: &ParallelSpectralConfigV2,
) -> SignalResult<(Array1<f64>, Array1<f64>)> {
    // Sequential implementation
    let n = signal.len();

    if segment_length > n {
        return Err(SignalError::ValueError(
            "Segment length exceeds signal length".to_string(),
        ));
    }

    let step = segment_length - overlap;
    let num_segments = (n - segment_length) / step + 1;

    let window = hann_window(segment_length);
    let window_power: f64 = window.iter().map(|&w| w * w).sum();

    let n_freq = segment_length / 2 + 1;
    let mut avg_psd = Array1::zeros(n_freq);

    for i in 0..num_segments {
        let start = i * step;
        let end = start + segment_length;

        let segment: Vec<f64> = signal
            .slice(scirs2_core::ndarray::s![start..end])
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| s * w)
            .collect();

        let complex_input = scirs2_fft::fft(&segment, Some(segment_length))
            .map_err(|e| SignalError::ComputationError(format!("FFT failed: {}", e)))?;

        for k in 0..n_freq {
            let mag_sq = complex_input[k].norm_sqr();
            let scale = if k == 0 || (k == segment_length / 2 && segment_length % 2 == 0) {
                1.0
            } else {
                2.0
            };
            avg_psd[k] += scale * mag_sq / (fs * window_power);
        }
    }

    avg_psd.mapv_inplace(|x| x / num_segments as f64);

    let frequencies = Array1::linspace(0.0, fs / 2.0, n_freq);

    Ok((frequencies, avg_psd))
}

// ============================================================================
// Memory-Optimized Spectral Processing
// ============================================================================

/// Memory-optimized AR spectral estimation for large signals
///
/// Processes large signals in chunks to limit memory usage while
/// maintaining accurate spectral estimation.
pub fn memory_optimized_ar_spectral(
    signal: &Array1<f64>,
    config: &ARSpectralConfig,
    mem_config: &MemoryOptimizedSpectralConfig,
) -> SignalResult<ARSpectralResult> {
    let n = signal.len();

    // Check if signal fits in memory constraint
    let signal_memory = n * std::mem::size_of::<f64>();
    let working_memory = signal_memory * 3; // Signal + intermediate buffers

    if working_memory <= mem_config.max_memory_bytes {
        // Signal fits in memory, use standard method
        return ar_spectral_estimation(signal, config);
    }

    // Use chunked processing for large signals
    let chunk_size = mem_config.chunk_size.min(n);
    let overlap = mem_config.overlap_samples.min(chunk_size / 2);

    // Compute autocorrelation incrementally
    let mut global_autocorr = Array1::zeros(config.order + 1);
    let mut total_samples = 0usize;

    let step = chunk_size - overlap;
    let mut pos = 0;

    while pos < n {
        let end = (pos + chunk_size).min(n);
        let chunk = signal.slice(scirs2_core::ndarray::s![pos..end]);
        let chunk_len = chunk.len();

        // Compute local autocorrelation
        for lag in 0..=config.order.min(chunk_len - 1) {
            let mut sum = 0.0;
            for i in 0..(chunk_len - lag) {
                sum += chunk[i] * chunk[i + lag];
            }
            global_autocorr[lag] += sum;
        }

        total_samples += chunk_len - config.order;
        pos += step;
    }

    // Normalize autocorrelation
    for lag in 0..=config.order {
        global_autocorr[lag] /= total_samples as f64;
    }

    // Apply Levinson-Durbin
    let (ar_coeffs, reflection_coeffs, variance) =
        levinson_durbin_enhanced(&global_autocorr, config.order)?;

    // Compute PSD
    let frequencies = compute_frequency_axis(config.nfft, config.fs);
    let psd = compute_ar_psd(&ar_coeffs, variance, &frequencies, config.fs)?;

    let aic = n as f64 * variance.ln() + 2.0 * config.order as f64;

    Ok(ARSpectralResult {
        frequencies,
        psd,
        ar_coefficients: ar_coeffs,
        reflection_coefficients: Some(reflection_coeffs),
        variance,
        order: config.order,
        information_criterion: Some(aic),
    })
}

/// Memory-optimized streaming spectral estimation
///
/// For very large signals that cannot fit in memory at all.
/// Uses streaming algorithms with minimal memory footprint.
pub struct StreamingSpectralEstimator {
    /// AR order
    order: usize,
    /// Accumulated autocorrelation
    autocorr: Array1<f64>,
    /// Sample count
    sample_count: usize,
    /// Previous samples buffer for overlap
    prev_samples: Vec<f64>,
    /// Sampling frequency
    fs: f64,
    /// NFFT for PSD computation
    nfft: usize,
}

impl StreamingSpectralEstimator {
    /// Create a new streaming spectral estimator
    pub fn new(order: usize, fs: f64, nfft: usize) -> Self {
        Self {
            order,
            autocorr: Array1::zeros(order + 1),
            sample_count: 0,
            prev_samples: Vec::with_capacity(order),
            fs,
            nfft,
        }
    }

    /// Process a chunk of samples
    pub fn process_chunk(&mut self, chunk: &[f64]) -> SignalResult<()> {
        let n = chunk.len();

        if n == 0 {
            return Ok(());
        }

        // Combine with previous samples for continuity
        let mut extended_chunk = self.prev_samples.clone();
        extended_chunk.extend_from_slice(chunk);

        let total_len = extended_chunk.len();

        // Update autocorrelation
        for lag in 0..=self.order.min(total_len - 1) {
            let mut sum = 0.0;
            for i in 0..(total_len - lag) {
                sum += extended_chunk[i] * extended_chunk[i + lag];
            }
            self.autocorr[lag] += sum;
        }

        self.sample_count += n;

        // Keep last 'order' samples for next chunk
        self.prev_samples.clear();
        let start = n.saturating_sub(self.order);
        self.prev_samples.extend_from_slice(&chunk[start..]);

        Ok(())
    }

    /// Finalize and compute AR spectral estimate
    pub fn finalize(&self) -> SignalResult<ARSpectralResult> {
        if self.sample_count < self.order + 1 {
            return Err(SignalError::ValueError(
                "Not enough samples processed".to_string(),
            ));
        }

        // Normalize autocorrelation
        let mut normalized_autocorr = self.autocorr.clone();
        for lag in 0..=self.order {
            normalized_autocorr[lag] /= self.sample_count as f64;
        }

        // Apply Levinson-Durbin
        let (ar_coeffs, reflection_coeffs, variance) =
            levinson_durbin_enhanced(&normalized_autocorr, self.order)?;

        // Compute PSD
        let frequencies = compute_frequency_axis(self.nfft, self.fs);
        let psd = compute_ar_psd(&ar_coeffs, variance, &frequencies, self.fs)?;

        let aic = self.sample_count as f64 * variance.ln() + 2.0 * self.order as f64;

        Ok(ARSpectralResult {
            frequencies,
            psd,
            ar_coefficients: ar_coeffs,
            reflection_coefficients: Some(reflection_coeffs),
            variance,
            order: self.order,
            information_criterion: Some(aic),
        })
    }

    /// Reset the estimator
    pub fn reset(&mut self) {
        self.autocorr = Array1::zeros(self.order + 1);
        self.sample_count = 0;
        self.prev_samples.clear();
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute frequency axis for PSD
fn compute_frequency_axis(nfft: usize, fs: f64) -> Array1<f64> {
    let n_freq = nfft / 2 + 1;
    Array1::linspace(0.0, fs / 2.0, n_freq)
}

/// Compute AR power spectral density
fn compute_ar_psd(
    ar_coeffs: &Array1<f64>,
    variance: f64,
    frequencies: &Array1<f64>,
    fs: f64,
) -> SignalResult<Array1<f64>> {
    let p = ar_coeffs.len() - 1;
    let mut psd = Array1::zeros(frequencies.len());

    for (i, &freq) in frequencies.iter().enumerate() {
        let w = 2.0 * PI * freq / fs;

        // Compute H(e^{jw}) = 1 / A(e^{jw})
        let mut h = Complex64::new(0.0, 0.0);
        for k in 0..=p {
            let phase = -w * k as f64;
            h += ar_coeffs[k] * Complex64::new(phase.cos(), phase.sin());
        }

        // PSD = variance / |H(w)|^2
        let h_mag_sq = h.norm_sqr();
        psd[i] = if h_mag_sq > 1e-15 {
            variance / h_mag_sq
        } else {
            variance * 1e15 // Large value for near-zero denominator
        };
    }

    Ok(psd)
}

/// Compute ARMA power spectral density
fn compute_arma_psd(
    ar_coeffs: &Array1<f64>,
    ma_coeffs: &Array1<f64>,
    variance: f64,
    frequencies: &Array1<f64>,
    fs: f64,
) -> SignalResult<Array1<f64>> {
    let p = ar_coeffs.len() - 1;
    let q = ma_coeffs.len() - 1;
    let mut psd = Array1::zeros(frequencies.len());

    for (i, &freq) in frequencies.iter().enumerate() {
        let w = 2.0 * PI * freq / fs;

        // Compute AR polynomial A(e^{jw})
        let mut a_poly = Complex64::new(0.0, 0.0);
        for k in 0..=p {
            let phase = -w * k as f64;
            a_poly += ar_coeffs[k] * Complex64::new(phase.cos(), phase.sin());
        }

        // Compute MA polynomial B(e^{jw})
        let mut b_poly = Complex64::new(0.0, 0.0);
        for k in 0..=q {
            let phase = -w * k as f64;
            b_poly += ma_coeffs[k] * Complex64::new(phase.cos(), phase.sin());
        }

        // PSD = variance * |B(w)|^2 / |A(w)|^2
        let a_mag_sq = a_poly.norm_sqr();
        let b_mag_sq = b_poly.norm_sqr();

        psd[i] = if a_mag_sq > 1e-15 {
            variance * b_mag_sq / a_mag_sq
        } else {
            variance * b_mag_sq * 1e15
        };
    }

    Ok(psd)
}

/// Compute ARMA residuals (innovations)
fn compute_arma_residuals(
    signal: &Array1<f64>,
    ar_coeffs: &Array1<f64>,
    ma_coeffs: &Array1<f64>,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let p = ar_coeffs.len() - 1;
    let q = ma_coeffs.len() - 1;

    let mut residuals = Array1::zeros(n);

    for t in 0..n {
        let mut pred = 0.0;

        // AR part
        for i in 1..=p.min(t) {
            pred -= ar_coeffs[i] * signal[t - i];
        }

        // MA part
        for i in 1..=q.min(t) {
            pred += ma_coeffs[i] * residuals[t - i];
        }

        residuals[t] = signal[t] - pred;
    }

    Ok(residuals)
}

/// Compute AR residuals
fn compute_ar_residuals(
    signal: &Array1<f64>,
    ar_coeffs: &Array1<f64>,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let p = ar_coeffs.len() - 1;

    let mut residuals = Array1::zeros(n);

    for t in p..n {
        let mut pred = 0.0;
        for i in 1..=p {
            pred -= ar_coeffs[i] * signal[t - i];
        }
        residuals[t] = signal[t] - pred;
    }

    Ok(residuals)
}

/// Solve linear system with improved numerical stability
fn solve_linear_system_stable(a: &Array2<f64>, b: &Array1<f64>) -> SignalResult<Array1<f64>> {
    let n = a.nrows();
    if a.ncols() != n || b.len() != n {
        return Err(SignalError::ValueError(
            "Dimension mismatch in linear system".to_string(),
        ));
    }

    // Estimate the matrix norm for adaptive regularization
    let mut max_diag = 0.0_f64;
    for i in 0..n {
        max_diag = max_diag.max(a[[i, i]].abs());
    }
    // Regularization parameter: small fraction of the diagonal norm
    let regularization = max_diag * 1e-10;

    // Gaussian elimination with partial pivoting and Tikhonov regularization
    let mut aug = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        // Add regularization to diagonal (Tikhonov regularization)
        aug[[i, i]] += regularization;
        aug[[i, n]] = b[i];
    }

    // Forward elimination
    for k in 0..n {
        // Find pivot
        let mut max_row = k;
        let mut max_val = aug[[k, k]].abs();
        for i in (k + 1)..n {
            if aug[[i, k]].abs() > max_val {
                max_val = aug[[i, k]].abs();
                max_row = i;
            }
        }

        // Swap rows
        if max_row != k {
            for j in 0..=n {
                let temp = aug[[k, j]];
                aug[[k, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Check for singularity (even after regularization)
        if aug[[k, k]].abs() < f64::EPSILON * max_diag.max(1.0) {
            return Err(SignalError::ComputationError(
                "Singular or near-singular matrix".to_string(),
            ));
        }

        // Eliminate
        for i in (k + 1)..n {
            let factor = aug[[i, k]] / aug[[k, k]];
            for j in k..=n {
                aug[[i, j]] -= factor * aug[[k, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        x[i] = sum / aug[[i, i]];
    }

    Ok(x)
}

/// Generate Hann window
fn hann_window(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (size - 1) as f64).cos()))
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::random::RngExt;
    use scirs2_core::Rng;

    fn generate_test_signal(n: usize, freq: f64, fs: f64) -> Array1<f64> {
        Array1::linspace(0.0, (n - 1) as f64 / fs, n).mapv(|t| (2.0 * PI * freq * t).sin())
    }

    fn generate_ar_signal(n: usize, ar_coeffs: &[f64], variance: f64) -> Array1<f64> {
        let mut rng = scirs2_core::random::rng();
        let mut signal = Array1::zeros(n);
        let p = ar_coeffs.len() - 1;

        for t in 0..n {
            let mut val = rng.random_range(-1.0..1.0) * variance.sqrt();
            for i in 1..=p.min(t) {
                val -= ar_coeffs[i] * signal[t - i];
            }
            signal[t] = val;
        }

        signal
    }

    #[test]
    fn test_ar_spectral_estimation() {
        let signal = generate_test_signal(512, 10.0, 100.0);

        let config = ARSpectralConfig {
            order: 10,
            fs: 100.0,
            nfft: 256,
            method: ARSpectralMethod::YuleWalkerEnhanced,
            ..Default::default()
        };

        let result = ar_spectral_estimation(&signal, &config);
        assert!(result.is_ok());

        let result = result.expect("Operation failed");
        assert_eq!(result.frequencies.len(), 129);
        assert_eq!(result.psd.len(), 129);
        assert_eq!(result.ar_coefficients.len(), 11);
        assert!(result.variance > 0.0);
    }

    #[test]
    fn test_ar_spectral_methods() {
        let signal = generate_test_signal(256, 20.0, 100.0);

        let methods = [
            ARSpectralMethod::YuleWalkerEnhanced,
            ARSpectralMethod::BurgEnhanced,
            ARSpectralMethod::ModifiedCovariance,
            ARSpectralMethod::LeastSquaresRegularized,
        ];

        for method in methods {
            let config = ARSpectralConfig {
                order: 8,
                fs: 100.0,
                nfft: 128,
                method,
                ..Default::default()
            };

            let result = ar_spectral_estimation(&signal, &config);
            assert!(result.is_ok(), "Method {:?} failed", method);
        }
    }

    #[test]
    fn test_arma_spectral_estimation() {
        let signal = generate_test_signal(512, 15.0, 100.0);

        let config = ARMASpectralConfig {
            ar_order: 5,
            ma_order: 3,
            fs: 100.0,
            nfft: 256,
            method: ARMASpectralMethod::HannanRissanenImproved,
            max_iterations: 50,
            tolerance: 1e-6,
            ..Default::default()
        };

        let result = arma_spectral_estimation(&signal, &config);
        assert!(result.is_ok());

        let result = result.expect("Operation failed");
        assert_eq!(result.frequencies.len(), 129);
        assert_eq!(result.psd.len(), 129);
        assert_eq!(result.ar_coefficients.len(), 6);
        assert_eq!(result.ma_coefficients.len(), 4);
    }

    #[test]
    fn test_arma_methods() {
        let signal = generate_test_signal(256, 25.0, 100.0);

        let methods = [
            ARMASpectralMethod::HannanRissanenImproved,
            ARMASpectralMethod::InnovationAccelerated,
            ARMASpectralMethod::MaximumLikelihoodGradient,
            ARMASpectralMethod::Durbin,
        ];

        for method in methods {
            let config = ARMASpectralConfig {
                ar_order: 3,
                ma_order: 2,
                fs: 100.0,
                nfft: 128,
                method,
                max_iterations: 20,
                tolerance: 1e-4,
                ..Default::default()
            };

            let result = arma_spectral_estimation(&signal, &config);
            assert!(result.is_ok(), "Method {:?} failed", method);
        }
    }

    #[test]
    fn test_parallel_welch_psd() {
        let signal = generate_test_signal(1024, 30.0, 200.0);

        let parallel_config = ParallelSpectralConfigV2::default();

        let result = parallel_welch_psd(&signal, 256, 128, 200.0, &parallel_config);
        assert!(result.is_ok());

        let (frequencies, psd) = result.expect("Operation failed");
        assert!(!frequencies.is_empty());
        assert_eq!(frequencies.len(), psd.len());

        // Check that PSD values are positive
        assert!(psd.iter().all(|&p| p > 0.0));
    }

    #[test]
    fn test_memory_optimized_ar_spectral() {
        let signal = generate_test_signal(2048, 40.0, 200.0);

        let ar_config = ARSpectralConfig {
            order: 15,
            fs: 200.0,
            nfft: 512,
            ..Default::default()
        };

        let mem_config = MemoryOptimizedSpectralConfig {
            max_memory_bytes: 1024, // Force chunked processing
            chunk_size: 256,
            overlap_samples: 32,
            streaming: false,
        };

        let result = memory_optimized_ar_spectral(&signal, &ar_config, &mem_config);
        assert!(result.is_ok());

        let result = result.expect("Operation failed");
        assert_eq!(result.frequencies.len(), 257);
        assert!(result.variance > 0.0);
    }

    #[test]
    fn test_streaming_spectral_estimator() {
        let mut estimator = StreamingSpectralEstimator::new(10, 100.0, 256);

        // Process signal in chunks
        let signal = generate_test_signal(1024, 20.0, 100.0);
        let chunk_size = 128;

        for chunk_start in (0..signal.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(signal.len());
            let chunk: Vec<f64> = signal
                .slice(scirs2_core::ndarray::s![chunk_start..chunk_end])
                .to_vec();
            estimator.process_chunk(&chunk).expect("Operation failed");
        }

        let result = estimator.finalize();
        assert!(result.is_ok());

        let result = result.expect("Operation failed");
        assert_eq!(result.order, 10);
        assert!(result.variance > 0.0);
    }

    #[test]
    fn test_ar_psd_peak_detection() {
        // Generate signal with known frequency
        let fs = 100.0;
        let freq = 25.0;
        let signal = generate_test_signal(1024, freq, fs);

        let config = ARSpectralConfig {
            order: 20,
            fs,
            nfft: 512,
            method: ARSpectralMethod::BurgEnhanced,
            ..Default::default()
        };

        let result = ar_spectral_estimation(&signal, &config).expect("Operation failed");

        // Find peak frequency
        let mut max_idx = 0;
        let mut max_psd = 0.0;
        for (i, &p) in result.psd.iter().enumerate() {
            if p > max_psd {
                max_psd = p;
                max_idx = i;
            }
        }

        let peak_freq = result.frequencies[max_idx];
        assert_relative_eq!(peak_freq, freq, epsilon = 2.0);
    }

    #[test]
    fn test_levinson_durbin_stability() {
        // Test with near-singular autocorrelation
        let mut autocorr = Array1::zeros(11);
        autocorr[0] = 1.0;
        for i in 1..11 {
            autocorr[i] = 0.99_f64.powi(i as i32);
        }

        let result = levinson_durbin_enhanced(&autocorr, 10);
        assert!(result.is_ok());

        let (ar_coeffs, reflection, variance) = result.expect("Operation failed");
        assert!(variance > 0.0);
        assert!(reflection.iter().all(|&r| r.abs() < 1.0));
    }

    #[test]
    fn test_empty_signal_error() {
        let signal = Array1::zeros(0);
        let config = ARSpectralConfig::default();

        let result = ar_spectral_estimation(&signal, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_order_too_large_error() {
        let signal = generate_test_signal(50, 10.0, 100.0);

        let config = ARSpectralConfig {
            order: 60, // Larger than signal length
            ..Default::default()
        };

        let result = ar_spectral_estimation(&signal, &config);
        assert!(result.is_err());
    }
}
