//! MVDR beamformer with diagonal loading for robustness
//!
//! Diagonal loading regularises the sample covariance matrix to improve the
//! robustness of the MVDR (Capon) beamformer against:
//!
//! - Finite sample effects (small number of snapshots)
//! - Steering vector mismatch (pointing errors, array calibration)
//! - Ill-conditioned covariance matrices
//!
//! The loaded covariance is `R_loaded = R + sigma^2 * I`, where the loading
//! level `sigma^2` can be set manually or determined automatically.
//!
//! Also provides the Robust Capon Beamformer (RCB) which explicitly optimises
//! over steering vector uncertainty.
//!
//! Provides:
//! - [`mvdr_diagonal_loading`]: MVDR with configurable diagonal loading
//! - [`auto_loading_level`]: Automatic loading to achieve a target condition number
//! - [`robust_capon`]: Robust Capon Beamformer (worst-case steering uncertainty)
//!
//! Pure Rust, no unwrap(), snake_case naming.

use crate::beamforming::array::{inner_product_conj, invert_hermitian_matrix, mat_vec_mul};
use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Loading level strategy for diagonal loading
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum LoadingLevel {
    /// Fixed loading: add `sigma^2` directly to the diagonal
    Fixed(f64),
    /// Trace-normalised loading: `sigma^2 = factor * trace(R) / N`
    TraceNormalized(f64),
    /// Automatic: find loading that minimises output variance subject to a
    /// well-conditioned covariance (target condition number ~100)
    AutoMinVariance,
    /// Eigenvalue-based: `sigma^2 = alpha * lambda_min_est(R)`
    ///
    /// The minimum eigenvalue is estimated via Gershgorin disc lower bounds.
    /// The parameter `alpha` is a scaling factor (typically 1.0..10.0).
    /// When `lambda_min` is non-positive, falls back to a trace-based estimate.
    EigenvalueBased(f64),
}

impl Default for LoadingLevel {
    fn default() -> Self {
        Self::TraceNormalized(0.01)
    }
}

/// Configuration for diagonal loading
#[derive(Debug, Clone)]
pub struct DiagonalLoadingConfig {
    /// Loading level strategy
    pub loading_level: LoadingLevel,
    /// Number of array elements
    pub n_elements: usize,
    /// Element spacing in wavelengths
    pub element_spacing: f64,
}

impl Default for DiagonalLoadingConfig {
    fn default() -> Self {
        Self {
            loading_level: LoadingLevel::default(),
            n_elements: 8,
            element_spacing: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of MVDR with diagonal loading
#[derive(Debug, Clone)]
pub struct LoadedMVDRResult {
    /// Optimal weight vector
    pub weights: Vec<Complex64>,
    /// Beamformer output power
    pub output_power: f64,
    /// Actual loading value used (`sigma^2`)
    pub loading_used: f64,
    /// Condition number of the loaded covariance matrix
    pub condition_number: f64,
    /// White Noise Gain: `|w^H s|^2 / ||w||^2`
    pub white_noise_gain: f64,
}

// ---------------------------------------------------------------------------
// MVDR with diagonal loading
// ---------------------------------------------------------------------------

/// MVDR beamformer with diagonal loading
///
/// Computes the MVDR weight vector using a loaded covariance matrix:
///
/// `R_loaded = R + sigma^2 * I`
/// `w = R_loaded^{-1} a / (a^H R_loaded^{-1} a)`
///
/// # Arguments
///
/// * `covariance` - Spatial covariance matrix (N x N), complex Hermitian
/// * `steering_vector` - Steering vector for the desired look direction
/// * `config` - Diagonal loading configuration
///
/// # Returns
///
/// [`LoadedMVDRResult`] with weights, power, condition number and WNG
pub fn mvdr_diagonal_loading(
    covariance: &[Vec<Complex64>],
    steering_vector: &[Complex64],
    config: &DiagonalLoadingConfig,
) -> SignalResult<LoadedMVDRResult> {
    let n = covariance.len();
    if n == 0 {
        return Err(SignalError::ValueError(
            "Covariance matrix must not be empty".to_string(),
        ));
    }
    if steering_vector.len() != n {
        return Err(SignalError::DimensionMismatch(format!(
            "Steering vector length {} does not match covariance size {}",
            steering_vector.len(),
            n
        )));
    }

    // Determine loading level
    let loading = resolve_loading_level(&config.loading_level, covariance)?;

    // Apply loading
    let r_loaded = apply_loading(covariance, loading);

    // Invert loaded covariance
    let r_inv = invert_hermitian_matrix(&r_loaded)?;

    // Compute weights: w = R_inv * a / (a^H R_inv a)
    let r_inv_a = mat_vec_mul(&r_inv, steering_vector);
    let denom = inner_product_conj(steering_vector, &r_inv_a);

    if denom.norm() < 1e-20 {
        return Err(SignalError::ComputationError(
            "MVDR denominator near zero; covariance may be singular even after loading".to_string(),
        ));
    }

    let weights: Vec<Complex64> = r_inv_a.iter().map(|&v| v / denom).collect();

    // Output power: P = 1 / (a^H R^{-1} a) = 1 / denom.re
    let output_power = if denom.re.abs() > 1e-30 {
        1.0 / denom.re
    } else {
        0.0
    };

    // Condition number estimate via Gershgorin bounds
    let condition_number = estimate_condition_number(&r_loaded);

    // White Noise Gain: WNG = |w^H s|^2 / ||w||^2
    let response = inner_product_conj(&weights, steering_vector);
    let w_norm_sq: f64 = weights.iter().map(|w| w.norm_sqr()).sum();
    let white_noise_gain = if w_norm_sq > 1e-30 {
        response.norm_sqr() / w_norm_sq
    } else {
        0.0
    };

    Ok(LoadedMVDRResult {
        weights,
        output_power,
        loading_used: loading,
        condition_number,
        white_noise_gain,
    })
}

// ---------------------------------------------------------------------------
// Auto loading level
// ---------------------------------------------------------------------------

/// Find the diagonal loading level that achieves a target condition number
///
/// Uses a bisection search to find `sigma^2` such that
/// `cond(R + sigma^2 I) approx target_condition`.
///
/// # Arguments
///
/// * `covariance` - Covariance matrix (N x N)
/// * `target_condition` - Desired condition number (must be >= 1)
///
/// # Returns
///
/// Loading level `sigma^2`
pub fn auto_loading_level(
    covariance: &[Vec<Complex64>],
    target_condition: f64,
) -> SignalResult<f64> {
    let n = covariance.len();
    if n == 0 {
        return Err(SignalError::ValueError(
            "Covariance matrix must not be empty".to_string(),
        ));
    }
    if target_condition < 1.0 {
        return Err(SignalError::ValueError(
            "Target condition number must be >= 1.0".to_string(),
        ));
    }

    let trace_val = compute_trace(covariance);
    if trace_val <= 0.0 {
        return Ok(1e-6); // fallback for zero/negative trace
    }

    // Bisection on loading level
    let mut lo = 0.0_f64;
    let mut hi = trace_val; // Upper bound: loading = trace makes cond ~ 2

    // Check if no loading is needed
    let cond_no_load = estimate_condition_number(covariance);
    if cond_no_load <= target_condition {
        return Ok(0.0);
    }

    for _ in 0..64 {
        let mid = (lo + hi) / 2.0;
        let loaded = apply_loading(covariance, mid);
        let cond = estimate_condition_number(&loaded);

        if cond > target_condition {
            lo = mid;
        } else {
            hi = mid;
        }

        if (hi - lo) / (hi + 1e-30) < 1e-8 {
            break;
        }
    }

    Ok(hi)
}

// ---------------------------------------------------------------------------
// Robust Capon Beamformer
// ---------------------------------------------------------------------------

/// Robust Capon Beamformer (RCB)
///
/// Optimises over steering vector uncertainty to provide robust beamforming
/// when the actual steering vector deviates from the nominal one.
///
/// The uncertainty model is a spherical region:
/// `||a - a_0|| <= epsilon`
///
/// The RCB finds the worst-case steering vector within this region and
/// computes MVDR weights accordingly. For the spherical uncertainty model,
/// this is equivalent to MVDR with an appropriate diagonal loading.
///
/// Reference: Li, Stoica, Wang, "On Robust Capon Beamforming and
/// Diagonal Loading", IEEE Trans. Signal Processing, 2003.
///
/// # Arguments
///
/// * `covariance` - Spatial covariance matrix (N x N)
/// * `nominal_steering` - Nominal steering vector
/// * `uncertainty_bound` - `epsilon`: bound on steering vector error norm
/// * `config` - Diagonal loading configuration (n_elements, element_spacing used)
///
/// # Returns
///
/// [`LoadedMVDRResult`] with the robust weights
pub fn robust_capon(
    covariance: &[Vec<Complex64>],
    nominal_steering: &[Complex64],
    uncertainty_bound: f64,
    config: &DiagonalLoadingConfig,
) -> SignalResult<LoadedMVDRResult> {
    let n = covariance.len();
    if n == 0 {
        return Err(SignalError::ValueError(
            "Covariance matrix must not be empty".to_string(),
        ));
    }
    if nominal_steering.len() != n {
        return Err(SignalError::DimensionMismatch(format!(
            "Steering vector length {} does not match covariance size {}",
            nominal_steering.len(),
            n
        )));
    }
    if uncertainty_bound < 0.0 {
        return Err(SignalError::ValueError(
            "Uncertainty bound must be non-negative".to_string(),
        ));
    }

    // For the spherical uncertainty model, the RCB is equivalent to finding
    // a Lagrange multiplier mu such that:
    //   w = (R + mu * I)^{-1} a_0 / (a_0^H (R + mu * I)^{-1} a_0)
    // where mu satisfies:
    //   ||a_hat - a_0|| = epsilon
    // with a_hat = (R + mu * I)^{-1} a_0 / ... (normalised)
    //
    // When epsilon is small, mu ~ 0. When epsilon is large, mu increases.
    // We use bisection to find the appropriate mu.

    if uncertainty_bound < 1e-12 {
        // No uncertainty: standard MVDR
        let loaded_config = DiagonalLoadingConfig {
            loading_level: LoadingLevel::Fixed(0.0),
            n_elements: config.n_elements,
            element_spacing: config.element_spacing,
        };
        return mvdr_diagonal_loading(covariance, nominal_steering, &loaded_config);
    }

    // Bisection to find the right loading level
    let trace_val = compute_trace(covariance);
    let norm_a0_sq: f64 = nominal_steering.iter().map(|v| v.norm_sqr()).sum();

    let mut lo = 0.0_f64;
    let mut hi = trace_val + norm_a0_sq / (uncertainty_bound * uncertainty_bound + 1e-30);
    let mut best_mu = trace_val * 0.01; // reasonable initial guess

    for _ in 0..64 {
        let mu = (lo + hi) / 2.0;
        let r_loaded = apply_loading(covariance, mu);

        let r_inv_result = invert_hermitian_matrix(&r_loaded);
        let r_inv = match r_inv_result {
            Ok(inv) => inv,
            Err(_) => {
                lo = mu;
                continue;
            }
        };

        let r_inv_a = mat_vec_mul(&r_inv, nominal_steering);
        let denom = inner_product_conj(nominal_steering, &r_inv_a);

        if denom.norm() < 1e-20 {
            lo = mu;
            continue;
        }

        // Compute the effective steering vector: a_hat = R_inv * a0 / (a0^H R_inv a0)
        // times norm_a0 to get the correct scale
        let a_hat: Vec<Complex64> = r_inv_a
            .iter()
            .map(|&v| v / denom * norm_a0_sq.sqrt())
            .collect();

        // Error norm: ||a_hat - a_0||
        let error_norm_sq: f64 = a_hat
            .iter()
            .zip(nominal_steering.iter())
            .map(|(&ah, &a0)| (ah - a0).norm_sqr())
            .sum();
        let error_norm = error_norm_sq.sqrt();

        if error_norm > uncertainty_bound {
            hi = mu;
        } else {
            lo = mu;
        }

        best_mu = mu;

        if (hi - lo) / (hi + 1e-30) < 1e-8 {
            break;
        }
    }

    // Use the found loading level
    let loaded_config = DiagonalLoadingConfig {
        loading_level: LoadingLevel::Fixed(best_mu),
        n_elements: config.n_elements,
        element_spacing: config.element_spacing,
    };
    mvdr_diagonal_loading(covariance, nominal_steering, &loaded_config)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve loading level to a concrete `sigma^2` value
fn resolve_loading_level(level: &LoadingLevel, covariance: &[Vec<Complex64>]) -> SignalResult<f64> {
    match level {
        LoadingLevel::Fixed(sigma_sq) => {
            if *sigma_sq < 0.0 {
                Err(SignalError::ValueError(
                    "Fixed loading level must be non-negative".to_string(),
                ))
            } else {
                Ok(*sigma_sq)
            }
        }
        LoadingLevel::TraceNormalized(factor) => {
            let n = covariance.len();
            if n == 0 {
                return Err(SignalError::ValueError(
                    "Cannot compute trace of empty matrix".to_string(),
                ));
            }
            let trace_val = compute_trace(covariance);
            Ok(factor * trace_val / n as f64)
        }
        LoadingLevel::AutoMinVariance => {
            // Target a condition number of 100
            auto_loading_level(covariance, 100.0)
        }
        LoadingLevel::EigenvalueBased(alpha) => {
            let n = covariance.len();
            if n == 0 {
                return Err(SignalError::ValueError(
                    "Cannot compute eigenvalue of empty matrix".to_string(),
                ));
            }
            if *alpha < 0.0 {
                return Err(SignalError::ValueError(
                    "Eigenvalue-based alpha must be non-negative".to_string(),
                ));
            }
            let lambda_min = estimate_min_eigenvalue(covariance);
            // If estimated minimum eigenvalue is non-positive or tiny,
            // fall back to trace-based estimate
            if lambda_min <= 1e-15 {
                let trace_val = compute_trace(covariance);
                Ok(alpha * trace_val / (n as f64 * n as f64))
            } else {
                Ok(alpha * lambda_min)
            }
        }
        _ => Err(SignalError::NotImplemented(
            "Unknown loading level variant".to_string(),
        )),
    }
}

/// Apply diagonal loading: R_loaded = R + sigma^2 * I
fn apply_loading(covariance: &[Vec<Complex64>], loading: f64) -> Vec<Vec<Complex64>> {
    let mut r_loaded: Vec<Vec<Complex64>> = covariance.to_vec();
    if loading > 0.0 {
        for (i, row) in r_loaded.iter_mut().enumerate() {
            if i < row.len() {
                row[i] += Complex64::new(loading, 0.0);
            }
        }
    }
    r_loaded
}

/// Compute trace of a complex matrix
fn compute_trace(matrix: &[Vec<Complex64>]) -> f64 {
    matrix
        .iter()
        .enumerate()
        .map(|(i, row)| if i < row.len() { row[i].re } else { 0.0 })
        .sum()
}

/// Estimate the minimum eigenvalue using Gershgorin circle lower bounds
///
/// For each row i the eigenvalue lies in the disc centred at `a_ii` with
/// radius `r_i = sum_{j!=i} |a_ij|`. The minimum eigenvalue is bounded
/// below by `min_i (a_ii - r_i)`.
fn estimate_min_eigenvalue(matrix: &[Vec<Complex64>]) -> f64 {
    let m = matrix.len();
    let mut min_lower = f64::INFINITY;

    for i in 0..m {
        let center = matrix[i][i].re;
        let radius: f64 = (0..m)
            .filter(|&j| j != i)
            .map(|j| {
                if j < matrix[i].len() {
                    matrix[i][j].norm()
                } else {
                    0.0
                }
            })
            .sum();
        let lower = center - radius;
        if lower < min_lower {
            min_lower = lower;
        }
    }

    if min_lower.is_finite() {
        min_lower
    } else {
        0.0
    }
}

/// Estimate condition number of a Hermitian PSD matrix
///
/// Uses Gershgorin disc bounds when they give useful positive lower bounds.
/// Falls back to the diagonal element ratio when Gershgorin is too
/// conservative (common for matrices with strong off-diagonal structure).
fn estimate_condition_number(matrix: &[Vec<Complex64>]) -> f64 {
    let n = matrix.len();
    if n == 0 {
        return 1.0;
    }

    let mut max_eig = f64::NEG_INFINITY;
    let mut min_eig_gershgorin = f64::INFINITY;
    let mut max_diag = f64::NEG_INFINITY;
    let mut min_diag = f64::INFINITY;

    for i in 0..n {
        let center = matrix[i][i].re;
        let radius: f64 = (0..n)
            .filter(|&j| j != i)
            .map(|j| matrix[i][j].norm())
            .sum();

        let upper = center + radius;
        let lower = center - radius;

        if upper > max_eig {
            max_eig = upper;
        }
        if lower < min_eig_gershgorin {
            min_eig_gershgorin = lower;
        }
        if center > max_diag {
            max_diag = center;
        }
        if center < min_diag {
            min_diag = center;
        }
    }

    // If Gershgorin gives a useful positive lower bound, use it
    if min_eig_gershgorin > 1e-30 {
        return max_eig / min_eig_gershgorin;
    }

    // Fallback: use diagonal ratio (reasonable heuristic for PSD matrices)
    if min_diag > 1e-30 {
        return max_diag / min_diag;
    }

    f64::INFINITY
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beamforming::array::steering_vector_ula;
    use approx::assert_relative_eq;

    fn identity_covariance(n: usize) -> Vec<Vec<Complex64>> {
        let mut cov = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        for i in 0..n {
            cov[i][i] = Complex64::new(1.0, 0.0);
        }
        cov
    }

    fn make_covariance_with_interferer(
        n: usize,
        signal_angle: f64,
        interf_angle: f64,
        d: f64,
    ) -> SignalResult<Vec<Vec<Complex64>>> {
        let sv_s = steering_vector_ula(n, signal_angle, d)?;
        let sv_i = steering_vector_ula(n, interf_angle, d)?;
        let sigma_s = 1.0;
        let sigma_i = 10.0;
        let sigma_n = 0.1;

        let mut cov = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        for i in 0..n {
            for j in 0..n {
                cov[i][j] = sigma_s * sigma_s * sv_s[i] * sv_s[j].conj()
                    + sigma_i * sigma_i * sv_i[i] * sv_i[j].conj();
            }
            cov[i][i] += Complex64::new(sigma_n * sigma_n, 0.0);
        }
        Ok(cov)
    }

    #[test]
    fn test_zero_loading_matches_standard_mvdr() {
        let n = 4;
        let cov = identity_covariance(n);
        let sv = steering_vector_ula(n, 0.2, 0.5).expect("SV");

        let config = DiagonalLoadingConfig {
            loading_level: LoadingLevel::Fixed(0.0),
            n_elements: n,
            element_spacing: 0.5,
        };

        let result = mvdr_diagonal_loading(&cov, &sv, &config).expect("should succeed");

        // With identity covariance and zero loading: P = 1/N
        assert_relative_eq!(result.output_power, 1.0 / n as f64, epsilon = 1e-8);
        assert_relative_eq!(result.loading_used, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_fixed_loading_improves_condition_number() {
        // Build a moderately ill-conditioned covariance (identity + small
        // off-diagonal perturbation) so the Gershgorin condition estimate
        // is finite both before and after loading.
        let n = 4;
        let mut cov = identity_covariance(n);
        cov[0][1] = Complex64::new(0.6, 0.1);
        cov[1][0] = Complex64::new(0.6, -0.1);
        cov[0][2] = Complex64::new(0.3, 0.0);
        cov[2][0] = Complex64::new(0.3, 0.0);

        let sv = steering_vector_ula(n, 0.0, 0.5).expect("SV");
        let cond_before = estimate_condition_number(&cov);
        assert!(
            cond_before.is_finite(),
            "Precondition: cond_before should be finite"
        );

        let config = DiagonalLoadingConfig {
            loading_level: LoadingLevel::Fixed(2.0),
            n_elements: n,
            element_spacing: 0.5,
        };

        let result = mvdr_diagonal_loading(&cov, &sv, &config).expect("should succeed");

        assert!(
            result.condition_number.is_finite(),
            "Loaded condition number should be finite, got {}",
            result.condition_number
        );
        assert!(
            result.condition_number <= cond_before,
            "Loading should improve conditioning: before={:.2}, after={:.2}",
            cond_before,
            result.condition_number
        );
        assert_relative_eq!(result.loading_used, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_trace_normalized_loading() {
        let n = 4;
        let cov = identity_covariance(n);
        let sv = steering_vector_ula(n, 0.0, 0.5).expect("SV");

        let factor = 0.1;
        let config = DiagonalLoadingConfig {
            loading_level: LoadingLevel::TraceNormalized(factor),
            n_elements: n,
            element_spacing: 0.5,
        };

        let result = mvdr_diagonal_loading(&cov, &sv, &config).expect("should succeed");

        // trace(I_4) = 4, so loading = 0.1 * 4 / 4 = 0.1
        let expected_loading = factor * 4.0 / n as f64;
        assert_relative_eq!(result.loading_used, expected_loading, epsilon = 1e-12);
    }

    #[test]
    fn test_auto_loading_achieves_target_condition() {
        let n = 6;
        let cov = make_covariance_with_interferer(n, 0.0, 0.4, 0.5).expect("cov");

        let target_cond = 50.0;
        let loading = auto_loading_level(&cov, target_cond).expect("should find loading");

        // Apply loading and check condition
        let loaded = apply_loading(&cov, loading);
        let cond = estimate_condition_number(&loaded);

        // The Gershgorin-based condition estimate is an upper bound,
        // so the actual condition should be at or below target
        assert!(
            cond <= target_cond * 2.0, // allow some slack due to Gershgorin being conservative
            "Condition number {} should be near target {}",
            cond,
            target_cond
        );
    }

    #[test]
    fn test_white_noise_gain_positive_and_bounded() {
        let n = 4;
        let cov = identity_covariance(n);
        let sv = steering_vector_ula(n, 0.1, 0.5).expect("SV");

        let config = DiagonalLoadingConfig {
            loading_level: LoadingLevel::Fixed(0.01),
            n_elements: n,
            element_spacing: 0.5,
        };

        let result = mvdr_diagonal_loading(&cov, &sv, &config).expect("should succeed");

        assert!(result.white_noise_gain > 0.0, "WNG should be positive");
        assert!(
            result.white_noise_gain <= n as f64 + 1.0,
            "WNG should be bounded: got {}",
            result.white_noise_gain
        );
    }

    #[test]
    fn test_robust_capon_basic() {
        let n = 4;
        let cov = make_covariance_with_interferer(n, 0.0, 0.4, 0.5).expect("cov");
        let sv = steering_vector_ula(n, 0.0, 0.5).expect("SV");

        let config = DiagonalLoadingConfig {
            loading_level: LoadingLevel::default(),
            n_elements: n,
            element_spacing: 0.5,
        };

        let result = robust_capon(&cov, &sv, 0.5, &config).expect("should succeed");

        assert!(result.output_power > 0.0, "Output power should be positive");
        assert!(result.output_power.is_finite());
        assert!(result.white_noise_gain > 0.0);
        assert_eq!(result.weights.len(), n);
    }

    #[test]
    fn test_robust_capon_zero_uncertainty_matches_mvdr() {
        let n = 4;
        let cov = identity_covariance(n);
        let sv = steering_vector_ula(n, 0.0, 0.5).expect("SV");

        let config = DiagonalLoadingConfig {
            loading_level: LoadingLevel::default(),
            n_elements: n,
            element_spacing: 0.5,
        };

        let result = robust_capon(&cov, &sv, 0.0, &config).expect("should succeed");
        // Zero uncertainty -> standard MVDR with zero loading
        assert_relative_eq!(result.output_power, 1.0 / n as f64, epsilon = 1e-8);
    }

    #[test]
    fn test_validation_errors() {
        let empty: Vec<Vec<Complex64>> = vec![];
        let sv = vec![Complex64::new(1.0, 0.0)];
        let config = DiagonalLoadingConfig::default();

        assert!(mvdr_diagonal_loading(&empty, &sv, &config).is_err());
        assert!(auto_loading_level(&empty, 100.0).is_err());

        // Dimension mismatch
        let cov = identity_covariance(3);
        let sv2 = vec![Complex64::new(1.0, 0.0); 5];
        assert!(mvdr_diagonal_loading(&cov, &sv2, &config).is_err());
    }

    #[test]
    fn test_eigenvalue_based_loading_identity() {
        let n = 4;
        let cov = identity_covariance(n);
        let sv = steering_vector_ula(n, 0.1, 0.5).expect("SV");
        let alpha = 2.0;

        let config = DiagonalLoadingConfig {
            loading_level: LoadingLevel::EigenvalueBased(alpha),
            n_elements: n,
            element_spacing: 0.5,
        };

        let result = mvdr_diagonal_loading(&cov, &sv, &config).expect("should succeed");

        // For identity, Gershgorin lower bound = 1.0, so loading = 2.0 * 1.0 = 2.0
        assert_relative_eq!(result.loading_used, 2.0, epsilon = 1e-12);
        assert!(result.output_power > 0.0);
    }

    #[test]
    fn test_eigenvalue_based_loading_with_interferer() {
        let n = 6;
        let cov = make_covariance_with_interferer(n, 0.0, 0.4, 0.5).expect("cov");
        let sv = steering_vector_ula(n, 0.0, 0.5).expect("SV");

        let config = DiagonalLoadingConfig {
            loading_level: LoadingLevel::EigenvalueBased(1.0),
            n_elements: n,
            element_spacing: 0.5,
        };

        let result = mvdr_diagonal_loading(&cov, &sv, &config).expect("should succeed");

        assert!(result.output_power > 0.0);
        assert!(result.output_power.is_finite());
        assert!(result.loading_used > 0.0);
        assert!(result.white_noise_gain > 0.0);
    }

    #[test]
    fn test_eigenvalue_based_preserves_distortionless() {
        let n = 6;
        let cov = make_covariance_with_interferer(n, 0.0, 0.4, 0.5).expect("cov");
        let sv = steering_vector_ula(n, 0.0, 0.5).expect("SV");

        let config = DiagonalLoadingConfig {
            loading_level: LoadingLevel::EigenvalueBased(1.0),
            n_elements: n,
            element_spacing: 0.5,
        };

        let result = mvdr_diagonal_loading(&cov, &sv, &config).expect("should succeed");

        // Check distortionless constraint: w^H a = 1
        let response = inner_product_conj(&result.weights, &sv);
        assert_relative_eq!(response.re, 1.0, epsilon = 1e-6);
        assert!(response.im.abs() < 1e-6);
    }

    #[test]
    fn test_auto_loading_no_loading_needed() {
        // Identity matrix has condition number 1, so no loading needed for target >= 1
        let n = 4;
        let cov = identity_covariance(n);
        let loading = auto_loading_level(&cov, 10.0).expect("should succeed");
        assert_relative_eq!(loading, 0.0, epsilon = 1e-8);
    }
}
