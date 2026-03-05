//! Common types for system identification algorithms

use scirs2_core::ndarray::{Array1, Array2};

/// Result of a parametric system identification
#[derive(Debug, Clone)]
pub struct SysIdResult {
    /// AR (denominator) coefficients: a_1, a_2, ..., a_na
    /// The full polynomial is 1 + a_1*z^{-1} + a_2*z^{-2} + ...
    pub a_coeffs: Array1<f64>,
    /// B (numerator / input) coefficients: b_0, b_1, ..., b_nb
    pub b_coeffs: Array1<f64>,
    /// MA (noise model) coefficients if applicable: c_1, c_2, ..., c_nc
    /// The full polynomial is 1 + c_1*z^{-1} + ...
    pub c_coeffs: Option<Array1<f64>>,
    /// Estimated noise variance
    pub noise_variance: f64,
    /// Model fit percentage (NRMSE-based, 0..100)
    pub fit_percentage: f64,
    /// Residual (prediction error) signal
    pub residuals: Array1<f64>,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
    /// Final Prediction Error
    pub fpe: f64,
    /// Number of estimated parameters
    pub n_params: usize,
}

/// Result of subspace (N4SID) identification
#[derive(Debug, Clone)]
pub struct SubspaceIdResult {
    /// State matrix A (n x n)
    pub a: Array2<f64>,
    /// Input matrix B (n x m)
    pub b: Array2<f64>,
    /// Output matrix C (p x n)
    pub c: Array2<f64>,
    /// Feedthrough matrix D (p x m)
    pub d: Array2<f64>,
    /// Estimated state order
    pub state_order: usize,
    /// Singular values from the SVD (useful for order selection)
    pub singular_values: Array1<f64>,
    /// Model fit percentage
    pub fit_percentage: f64,
    /// Noise variance estimate
    pub noise_variance: f64,
}

/// Configuration for ARX model estimation
#[derive(Debug, Clone)]
pub struct ArxConfig {
    /// Order of the AR part (number of past outputs)
    pub na: usize,
    /// Order of the B part (number of past inputs, including b_0)
    pub nb: usize,
    /// Input delay (dead time, number of samples)
    pub nk: usize,
}

/// Configuration for ARMAX model estimation
#[derive(Debug, Clone)]
pub struct ArmaxConfig {
    /// Order of the AR part
    pub na: usize,
    /// Order of the B (input) part
    pub nb: usize,
    /// Order of the MA (noise) part
    pub nc: usize,
    /// Input delay
    pub nk: usize,
    /// Maximum iterations for the iterative estimation
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for ArmaxConfig {
    fn default() -> Self {
        Self {
            na: 2,
            nb: 2,
            nc: 2,
            nk: 1,
            max_iter: 100,
            tolerance: 1e-6,
        }
    }
}

/// Configuration for Output-Error model estimation
#[derive(Debug, Clone)]
pub struct OeConfig {
    /// Order of the B (input numerator) polynomial
    pub nb: usize,
    /// Order of the F (denominator) polynomial
    pub nf: usize,
    /// Input delay
    pub nk: usize,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for OeConfig {
    fn default() -> Self {
        Self {
            nb: 2,
            nf: 2,
            nk: 1,
            max_iter: 100,
            tolerance: 1e-6,
        }
    }
}

/// Configuration for N4SID subspace identification
#[derive(Debug, Clone)]
pub struct N4sidConfig {
    /// Desired state order (if None, determined automatically from singular values)
    pub state_order: Option<usize>,
    /// Block rows in the Hankel matrix (past/future horizon)
    pub block_rows: usize,
    /// Threshold ratio for automatic order selection (singular value ratio)
    pub sv_threshold: f64,
}

impl Default for N4sidConfig {
    fn default() -> Self {
        Self {
            state_order: None,
            block_rows: 10,
            sv_threshold: 0.01,
        }
    }
}

/// Configuration for Recursive Least Squares
#[derive(Debug, Clone)]
pub struct RlsConfig {
    /// Number of parameters (AR + B coefficients)
    pub n_params: usize,
    /// Forgetting factor (0 < lambda <= 1, typically 0.95-0.999)
    pub forgetting_factor: f64,
    /// Initial covariance scaling (large = uncertain initial guess)
    pub initial_covariance: f64,
}

impl Default for RlsConfig {
    fn default() -> Self {
        Self {
            n_params: 4,
            forgetting_factor: 0.98,
            initial_covariance: 1000.0,
        }
    }
}

/// Configuration for Prediction Error Method
#[derive(Debug, Clone)]
pub struct PemConfig {
    /// Order of the AR part (A polynomial)
    pub na: usize,
    /// Order of the B part (input polynomial)
    pub nb: usize,
    /// Order of the MA part (C polynomial)
    pub nc: usize,
    /// Input delay
    pub nk: usize,
    /// Maximum iterations for Gauss-Newton optimization
    pub max_iter: usize,
    /// Convergence tolerance for parameter change
    pub tolerance: f64,
    /// Step size damping factor for Gauss-Newton
    pub damping: f64,
}

impl Default for PemConfig {
    fn default() -> Self {
        Self {
            na: 2,
            nb: 2,
            nc: 1,
            nk: 1,
            max_iter: 50,
            tolerance: 1e-6,
            damping: 0.5,
        }
    }
}

/// Compute model fit percentage using NRMSE:
///   fit = 100 * (1 - ||y - y_hat|| / ||y - mean(y)||)
pub(crate) fn compute_fit_percentage(y: &Array1<f64>, y_hat: &Array1<f64>) -> f64 {
    let n = y.len().min(y_hat.len());
    if n == 0 {
        return 0.0;
    }

    let y_mean = y.iter().copied().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().take(n).map(|&v| (v - y_mean).powi(2)).sum();
    let ss_res: f64 = y
        .iter()
        .take(n)
        .zip(y_hat.iter().take(n))
        .map(|(&yi, &yhi)| (yi - yhi).powi(2))
        .sum();

    if ss_tot < 1e-30 {
        return if ss_res < 1e-30 { 100.0 } else { 0.0 };
    }

    let fit = 100.0 * (1.0 - (ss_res.sqrt() / ss_tot.sqrt()));
    fit.clamp(0.0, 100.0)
}

/// Compute information criteria
pub(crate) fn compute_information_criteria(
    noise_var: f64,
    n: usize,
    n_params: usize,
) -> (f64, f64, f64) {
    let nf = n as f64;
    let kf = n_params as f64;

    // Log-likelihood (Gaussian assumption)
    let log_l = -0.5 * nf * (noise_var.max(1e-30).ln() + 1.0 + (2.0 * std::f64::consts::PI).ln());

    let aic = -2.0 * log_l + 2.0 * kf;
    let bic = -2.0 * log_l + kf * nf.ln();
    let fpe = noise_var * (nf + kf) / (nf - kf).max(1.0);

    (aic, bic, fpe)
}
