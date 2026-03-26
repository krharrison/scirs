//! Type definitions for distribution distance metrics.
//!
//! This module provides configuration structs, result types, and kernel
//! enumerations used across the distribution distance metric submodules.

use serde::{Deserialize, Serialize};

// ────────────────────────────────────────────────────────────────────────────
// Kernel Type
// ────────────────────────────────────────────────────────────────────────────

/// Kernel function type for Kernel Stein Discrepancy and related methods.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
#[derive(Default)]
pub enum KernelType {
    /// Radial Basis Function (Gaussian) kernel:
    /// `k(x, y) = exp(-||x-y||^2 / (2 * bandwidth^2))`
    #[default]
    Rbf,
    /// Inverse Multi-Quadric kernel:
    /// `k(x, y) = (c^2 + ||x-y||^2)^beta`  where beta < 0 (default: -0.5)
    Imq {
        /// Constant offset (default: 1.0)
        c: f64,
        /// Exponent (must be negative, default: -0.5)
        beta: f64,
    },
    /// Polynomial kernel:
    /// `k(x, y) = (alpha * <x, y> + c)^degree`
    Polynomial {
        /// Scaling factor (default: 1.0)
        alpha: f64,
        /// Constant offset (default: 1.0)
        c: f64,
        /// Polynomial degree (default: 3)
        degree: u32,
    },
}

// ────────────────────────────────────────────────────────────────────────────
// KSD Types
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for Kernel Stein Discrepancy computations.
#[derive(Debug, Clone)]
pub struct KsdConfig {
    /// Kernel type to use
    pub kernel: KernelType,
    /// Bandwidth for RBF/IMQ kernels (median heuristic if None)
    pub bandwidth: Option<f64>,
    /// Number of bootstrap resamples for p-value computation
    pub n_bootstrap: usize,
}

impl Default for KsdConfig {
    fn default() -> Self {
        Self {
            kernel: KernelType::Rbf,
            bandwidth: None,
            n_bootstrap: 1000,
        }
    }
}

/// Result from a Kernel Stein Discrepancy computation.
#[derive(Debug, Clone)]
pub struct KsdResult {
    /// The KSD statistic value
    pub statistic: f64,
    /// Bootstrap p-value (if computed)
    pub p_value: Option<f64>,
    /// Whether the null hypothesis (samples from target) is rejected
    /// at the significance level used
    pub rejected: Option<bool>,
    /// Kernel type used
    pub kernel: KernelType,
    /// Bandwidth that was used (may have been computed via median heuristic)
    pub bandwidth: f64,
}

// ────────────────────────────────────────────────────────────────────────────
// Sinkhorn Types
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for the Sinkhorn algorithm with a cost matrix.
#[derive(Debug, Clone)]
pub struct SinkhornConfig {
    /// Entropic regularization parameter (higher = more blur, faster convergence)
    pub epsilon: f64,
    /// Maximum number of Sinkhorn iterations
    pub max_iter: usize,
    /// Convergence tolerance on dual variables
    pub tol: f64,
    /// Whether to use log-domain stabilization for numerical stability
    pub log_domain: bool,
}

impl Default for SinkhornConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            max_iter: 1000,
            tol: 1e-9,
            log_domain: true,
        }
    }
}

/// Result from a Sinkhorn divergence computation.
#[derive(Debug, Clone)]
pub struct SinkhornResult {
    /// The (debiased) Sinkhorn divergence value
    pub divergence: f64,
    /// The optimal transport plan (row-major, n x m)
    pub transport_plan: Vec<Vec<f64>>,
    /// Whether the algorithm converged within max_iter
    pub converged: bool,
    /// Number of iterations actually performed
    pub iterations: usize,
}

// ────────────────────────────────────────────────────────────────────────────
// Distance Result
// ────────────────────────────────────────────────────────────────────────────

/// Method used to compute a distribution distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DistanceMethod {
    /// Total Variation distance
    TotalVariation,
    /// Hellinger distance
    Hellinger,
    /// KL divergence
    KullbackLeibler,
    /// Jensen-Shannon divergence
    JensenShannon,
    /// Chi-squared divergence
    ChiSquare,
    /// Energy distance
    Energy,
    /// Wasserstein (Earth Mover's) distance
    Wasserstein,
    /// Sliced Wasserstein distance
    SlicedWasserstein,
    /// Sinkhorn divergence
    Sinkhorn,
}

/// A generic result carrying a distance value and its method label.
#[derive(Debug, Clone)]
pub struct DistanceResult {
    /// The computed distance / divergence value
    pub value: f64,
    /// Which method was used
    pub method: DistanceMethod,
}

impl DistanceResult {
    /// Create a new distance result.
    pub fn new(value: f64, method: DistanceMethod) -> Self {
        Self { value, method }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_type_default_is_rbf() {
        assert_eq!(KernelType::default(), KernelType::Rbf);
    }

    #[test]
    fn test_sinkhorn_config_default() {
        let cfg = SinkhornConfig::default();
        assert!((cfg.epsilon - 0.1).abs() < 1e-12);
        assert_eq!(cfg.max_iter, 1000);
        assert!(cfg.log_domain);
    }

    #[test]
    fn test_ksd_config_default() {
        let cfg = KsdConfig::default();
        assert_eq!(cfg.kernel, KernelType::Rbf);
        assert!(cfg.bandwidth.is_none());
        assert_eq!(cfg.n_bootstrap, 1000);
    }

    #[test]
    fn test_distance_result_new() {
        let r = DistanceResult::new(0.42, DistanceMethod::Hellinger);
        assert!((r.value - 0.42).abs() < 1e-12);
        assert_eq!(r.method, DistanceMethod::Hellinger);
    }

    #[test]
    fn test_distance_method_non_exhaustive() {
        // Verify we can match all current variants
        let methods = [
            DistanceMethod::TotalVariation,
            DistanceMethod::Hellinger,
            DistanceMethod::KullbackLeibler,
            DistanceMethod::JensenShannon,
            DistanceMethod::ChiSquare,
            DistanceMethod::Energy,
            DistanceMethod::Wasserstein,
            DistanceMethod::SlicedWasserstein,
            DistanceMethod::Sinkhorn,
        ];
        for m in &methods {
            // Just verify Debug formatting works for all variants
            let _ = format!("{m:?}");
        }
    }
}
