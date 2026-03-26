//! Types for Automatic Differentiation Variational Inference (ADVI).
//!
//! Defines configuration, result, variational family, and constraint types
//! used throughout the ADVI module.

// ============================================================================
// AdviConfig
// ============================================================================

/// Configuration for the ADVI optimizer.
///
/// Controls the number of Monte Carlo gradient samples, iterations,
/// learning rate (Adam), convergence tolerance, and prior precision.
#[derive(Debug, Clone)]
pub struct AdviConfig {
    /// Number of MC gradient samples for ELBO estimation (default: 1)
    pub n_samples: usize,
    /// Maximum number of optimization iterations (default: 1000)
    pub n_iter: usize,
    /// Adam learning rate (default: 0.01)
    pub lr: f64,
    /// Convergence tolerance: stop when |ELBO_t - ELBO_{t-1}| < tol (default: 1e-4)
    pub tol: f64,
    /// Prior precision λ (Gaussian prior N(0, λ⁻¹I)); default: 1.0
    pub prior_precision: f64,
    /// Finite-difference step for gradient estimation (default: 1e-5)
    pub fd_step: f64,
    /// Random seed for reproducibility (default: 42)
    pub seed: u64,
}

impl Default for AdviConfig {
    fn default() -> Self {
        Self {
            n_samples: 1,
            n_iter: 1000,
            lr: 0.01,
            tol: 1e-4,
            prior_precision: 1.0,
            fd_step: 1e-5,
            seed: 42,
        }
    }
}

// ============================================================================
// VariationalFamily
// ============================================================================

/// The family of variational distributions used in ADVI.
///
/// - `MeanField`: product of independent Gaussians q(θ) = Π N(μᵢ, σᵢ²)
/// - `FullRank`: multivariate Gaussian q(θ) = N(μ, LL^T)
/// - `NormalizingFlow`: normalizing flow (planar/radial; not yet fully supported)
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariationalFamily {
    /// Mean-field factored Gaussian (independent per-dimension)
    MeanField,
    /// Full-rank Gaussian with Cholesky covariance
    FullRank,
    /// Normalizing flow posterior (placeholder for future extension)
    NormalizingFlow,
}

// ============================================================================
// ConstraintType
// ============================================================================

/// The constraint type applied to a parameter before variational inference.
///
/// Parameters are mapped to unconstrained real space via bijective transforms
/// before fitting; the Jacobian log-determinant adjusts the ELBO.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintType {
    /// Parameter lives on the entire real line (no transform)
    Unconstrained,
    /// Parameter must be strictly positive (log transform: η = log θ)
    Positive,
    /// Parameter lives on the simplex (softmax/additive-log-ratio)
    Simplex,
    /// Parameter lives in the bounded interval (lo, hi)
    Bounded {
        /// Lower bound (exclusive)
        lo: f64,
        /// Upper bound (exclusive)
        hi: f64,
    },
}

// ============================================================================
// AdviResult
// ============================================================================

/// Result returned by an ADVI optimization run.
///
/// Contains the variational parameters (μ, log σ) in unconstrained space
/// and the ELBO history for convergence diagnostics.
#[derive(Debug, Clone)]
pub struct AdviResult {
    /// ELBO values at each iteration
    pub elbo_history: Vec<f64>,
    /// Variational mean in unconstrained space, length = n_params
    pub mu: Vec<f64>,
    /// Log-scale parameter ω = log σ, length = n_params (σ = exp(ω))
    pub log_sigma: Vec<f64>,
    /// Whether the optimizer converged within the tolerance
    pub converged: bool,
    /// Number of iterations actually performed
    pub n_iter_performed: usize,
}

impl AdviResult {
    /// Return the posterior standard deviations σᵢ = exp(ωᵢ).
    pub fn sigma(&self) -> Vec<f64> {
        self.log_sigma.iter().map(|&w| w.exp()).collect()
    }

    /// Return the final ELBO value, or NaN if no iterations were performed.
    pub fn final_elbo(&self) -> f64 {
        self.elbo_history.last().copied().unwrap_or(f64::NAN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advi_config_default() {
        let cfg = AdviConfig::default();
        assert_eq!(cfg.n_samples, 1);
        assert_eq!(cfg.n_iter, 1000);
        assert!((cfg.lr - 0.01).abs() < 1e-12);
        assert!((cfg.tol - 1e-4).abs() < 1e-12);
        assert!((cfg.prior_precision - 1.0).abs() < 1e-12);
        assert!((cfg.fd_step - 1e-5).abs() < 1e-12);
    }

    #[test]
    fn test_variational_family_eq() {
        assert_eq!(VariationalFamily::MeanField, VariationalFamily::MeanField);
        assert_ne!(VariationalFamily::MeanField, VariationalFamily::FullRank);
    }

    #[test]
    fn test_constraint_type_variants() {
        let c = ConstraintType::Bounded { lo: 0.0, hi: 1.0 };
        match c {
            ConstraintType::Bounded { lo, hi } => {
                assert!((lo - 0.0).abs() < 1e-15);
                assert!((hi - 1.0).abs() < 1e-15);
            }
            _ => panic!("Expected Bounded variant"),
        }
    }

    #[test]
    fn test_advi_result_sigma() {
        let result = AdviResult {
            elbo_history: vec![-10.0, -5.0, -2.0],
            mu: vec![1.0, 2.0],
            log_sigma: vec![0.0, -1.0],
            converged: true,
            n_iter_performed: 3,
        };
        let sigma = result.sigma();
        assert!((sigma[0] - 1.0).abs() < 1e-12, "exp(0) = 1");
        assert!((sigma[1] - (-1.0_f64).exp()).abs() < 1e-12);
        assert!((result.final_elbo() - (-2.0)).abs() < 1e-12);
    }
}
