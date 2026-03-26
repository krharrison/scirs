//! Types for Bayesian neural network posterior approximations.
//!
//! Provides configuration and result types for Laplace approximation
//! and SWAG (Stochastic Weight Averaging Gaussian).

// ============================================================================
// Hessian computation methods for Laplace approximation
// ============================================================================

/// Method used to approximate the Hessian of the loss for Laplace approximation.
///
/// - `GGN`: Generalized Gauss-Newton (Fisher information proxy) via squared gradients.
/// - `Diagonal`: Diagonal Hessian approximation.
/// - `KFAC`: Kronecker-factored approximate curvature.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HessianMethod {
    /// Generalized Gauss-Newton / squared-gradient Fisher approximation (default)
    #[default]
    GGN,
    /// Diagonal curvature only
    Diagonal,
    /// Kronecker-factored approximate curvature
    KFAC,
}

// ============================================================================
// LaplaceConfig
// ============================================================================

/// Configuration for Laplace approximation of a BNN posterior.
#[derive(Debug, Clone)]
pub struct LaplaceConfig {
    /// Method for computing the approximate curvature (default: GGN)
    pub hessian_method: HessianMethod,
    /// Tikhonov / prior regularization damping added to the diagonal of H.
    /// Corresponds to the prior precision λ in N(0, λ⁻¹ I). (default: 1.0)
    pub damping: f64,
    /// Finite-difference step for gradient computation (default: 1e-5)
    pub fd_step: f64,
}

impl Default for LaplaceConfig {
    fn default() -> Self {
        Self {
            hessian_method: HessianMethod::GGN,
            damping: 1.0,
            fd_step: 1e-5,
        }
    }
}

// ============================================================================
// SwagConfig
// ============================================================================

/// Configuration for the SWAG posterior estimator.
#[derive(Debug, Clone)]
pub struct SwagConfig {
    /// Number of SGD snapshot epochs to collect (default: 20)
    pub n_epochs: usize,
    /// Maximum low-rank deviation columns C (SWAG rank; default: 20)
    pub c: usize,
    /// Learning rate for the parameter updates provided to `SwagCollector` (default: 0.01)
    pub lr: f64,
}

impl Default for SwagConfig {
    fn default() -> Self {
        Self {
            n_epochs: 20,
            c: 20,
            lr: 0.01,
        }
    }
}

// ============================================================================
// BnnApproxResult
// ============================================================================

/// Summary result of a Bayesian NN approximation (Laplace or SWAG).
///
/// Contains the mean weight vector and per-parameter uncertainty (variance).
#[derive(Debug, Clone)]
pub struct BnnApproxResult {
    /// Mean weights θ* (MAP estimate for Laplace; SWA solution for SWAG)
    pub mean_weights: Vec<f64>,
    /// Per-parameter posterior variance (diagonal of the covariance)
    pub uncertainty: Vec<f64>,
    /// Optional: label describing the approximation method used
    pub method: String,
}

impl BnnApproxResult {
    /// Return per-parameter posterior standard deviations.
    pub fn std_devs(&self) -> Vec<f64> {
        self.uncertainty.iter().map(|&v| v.sqrt()).collect()
    }

    /// Return the number of parameters.
    pub fn n_params(&self) -> usize {
        self.mean_weights.len()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayesian_approx_config_default() {
        let lap = LaplaceConfig::default();
        assert_eq!(lap.hessian_method, HessianMethod::GGN);
        assert!((lap.damping - 1.0).abs() < 1e-12);

        let swag = SwagConfig::default();
        assert_eq!(swag.n_epochs, 20);
        assert_eq!(swag.c, 20);
        assert!((swag.lr - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_hessian_method_default_is_ggn() {
        let m = HessianMethod::default();
        assert_eq!(m, HessianMethod::GGN);
    }

    #[test]
    fn test_bnn_approx_result_std_devs() {
        let result = BnnApproxResult {
            mean_weights: vec![1.0, 2.0, 3.0],
            uncertainty: vec![4.0, 9.0, 16.0],
            method: "Laplace".to_string(),
        };
        let stds = result.std_devs();
        assert!((stds[0] - 2.0).abs() < 1e-12);
        assert!((stds[1] - 3.0).abs() < 1e-12);
        assert!((stds[2] - 4.0).abs() < 1e-12);
        assert_eq!(result.n_params(), 3);
    }
}
