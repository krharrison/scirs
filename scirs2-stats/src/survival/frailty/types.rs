//! Core types for frailty survival models.

/// Frailty distribution family for the random effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FrailtyDistribution {
    /// Gamma frailty: u ~ Gamma(1/theta, theta) with E\[u\]=1, Var(u)=theta.
    /// Analytically tractable E-step.
    Gamma,
    /// Log-normal frailty: log(u) ~ N(-sigma^2/2, sigma^2) with E\[u\]=1, Var(u)=exp(sigma^2)-1.
    /// Requires Laplace approximation in the E-step.
    LogNormal,
    /// Inverse-Gaussian frailty: u ~ IG(1, lambda) with E\[u\]=1, Var(u)=1/lambda.
    InverseGaussian,
}

/// Configuration for frailty model fitting.
#[derive(Debug, Clone)]
pub struct FrailtyConfig {
    /// Frailty distribution family (default: Gamma).
    pub distribution: FrailtyDistribution,
    /// Maximum EM iterations (default: 200).
    pub max_iterations: usize,
    /// Convergence tolerance on relative log-likelihood change (default: 1e-6).
    pub tolerance: f64,
    /// Initial frailty variance θ₀ (default: 1.0).
    pub initial_variance: f64,
}

impl Default for FrailtyConfig {
    fn default() -> Self {
        Self {
            distribution: FrailtyDistribution::Gamma,
            max_iterations: 200,
            tolerance: 1e-6,
            initial_variance: 1.0,
        }
    }
}

/// Result of fitting a shared frailty model.
#[derive(Debug, Clone)]
pub struct FrailtyResult {
    /// Estimated regression coefficients β.
    pub coefficients: Vec<f64>,
    /// Estimated frailty variance θ.
    pub frailty_variance: f64,
    /// Posterior (empirical Bayes) frailty estimates per cluster.
    pub frailty_estimates: Vec<f64>,
    /// Log-likelihood trace across EM iterations.
    pub log_likelihood_history: Vec<f64>,
    /// Whether the EM algorithm converged.
    pub converged: bool,
    /// Number of EM iterations performed.
    pub iterations: usize,
    /// Baseline cumulative hazard at observed event times: (time, H₀(t)).
    pub baseline_hazard: Vec<(f64, f64)>,
}

/// Information about a single cluster (group).
#[derive(Debug, Clone)]
pub struct ClusterInfo {
    /// Cluster identifier.
    pub cluster_id: usize,
    /// Row indices of subjects belonging to this cluster.
    pub subject_indices: Vec<usize>,
    /// Number of events in this cluster.
    pub n_events: usize,
}

impl ClusterInfo {
    /// Create a new cluster info from a set of indices and event indicators.
    pub fn new(cluster_id: usize, subject_indices: Vec<usize>, events: &[bool]) -> Self {
        let n_events = subject_indices
            .iter()
            .filter(|&&i| i < events.len() && events[i])
            .count();
        Self {
            cluster_id,
            subject_indices,
            n_events,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frailty_config_default() {
        let cfg = FrailtyConfig::default();
        assert_eq!(cfg.distribution, FrailtyDistribution::Gamma);
        assert_eq!(cfg.max_iterations, 200);
        assert!((cfg.tolerance - 1e-6).abs() < 1e-15);
        assert!((cfg.initial_variance - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_cluster_info_event_count() {
        let events = [true, false, true, false, true];
        let info = ClusterInfo::new(0, vec![0, 2, 4], &events);
        assert_eq!(info.n_events, 3);
        assert_eq!(info.cluster_id, 0);
    }

    #[test]
    fn test_cluster_info_out_of_bounds_indices() {
        let events = [true, false];
        // Index 5 is out of bounds, should be silently skipped for event counting
        let info = ClusterInfo::new(1, vec![0, 5], &events);
        assert_eq!(info.n_events, 1);
    }

    #[test]
    fn test_frailty_distribution_variants() {
        let g = FrailtyDistribution::Gamma;
        let ln = FrailtyDistribution::LogNormal;
        let ig = FrailtyDistribution::InverseGaussian;
        assert_ne!(g, ln);
        assert_ne!(g, ig);
        assert_ne!(ln, ig);
    }
}
