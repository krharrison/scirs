//! Types for multi-fidelity optimization (Hyperband / Successive Halving).
//!
//! Provides configuration, sampling strategies, and result types used by both
//! [`super::successive_halving::SuccessiveHalving`] and
//! [`super::hyperband::Hyperband`].

use crate::error::OptimizeResult;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for multi-fidelity optimization algorithms.
///
/// The budget parameters control how resources (e.g. training epochs) are
/// allocated.  `eta` is the *reduction factor*: at each successive halving
/// round the number of surviving configurations is divided by `eta` while
/// the per-configuration budget is multiplied by `eta`.
#[derive(Debug, Clone)]
pub struct MultiFidelityConfig {
    /// Maximum resource budget per configuration (e.g. epochs).
    pub max_budget: f64,
    /// Minimum resource budget per configuration.
    pub min_budget: f64,
    /// Reduction factor (default 3).
    pub eta: usize,
    /// Number of initial configurations.  When zero the algorithm computes a
    /// sensible default from the budget ratio and `eta`.
    pub n_initial: usize,
}

impl Default for MultiFidelityConfig {
    fn default() -> Self {
        Self {
            max_budget: 81.0,
            min_budget: 1.0,
            eta: 3,
            n_initial: 0, // auto-compute
        }
    }
}

impl MultiFidelityConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> OptimizeResult<()> {
        if self.max_budget <= 0.0 {
            return Err(crate::error::OptimizeError::InvalidParameter(
                "max_budget must be positive".into(),
            ));
        }
        if self.min_budget <= 0.0 {
            return Err(crate::error::OptimizeError::InvalidParameter(
                "min_budget must be positive".into(),
            ));
        }
        if self.min_budget > self.max_budget {
            return Err(crate::error::OptimizeError::InvalidParameter(
                "min_budget must not exceed max_budget".into(),
            ));
        }
        if self.eta < 2 {
            return Err(crate::error::OptimizeError::InvalidParameter(
                "eta must be >= 2".into(),
            ));
        }
        Ok(())
    }

    /// Compute `s_max = floor(log_eta(max_budget / min_budget))`.
    pub(crate) fn s_max(&self) -> usize {
        let ratio = self.max_budget / self.min_budget;
        let eta_f = self.eta as f64;
        ratio.ln().div_euclid(eta_f.ln()) as usize
    }
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

/// Strategy used to generate initial configurations.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum ConfigSampler {
    /// Uniform random sampling inside bounds.
    Random,
    /// Latin Hypercube Sampling (space-filling).
    LatinHypercube,
}

impl Default for ConfigSampler {
    fn default() -> Self {
        Self::Random
    }
}

/// Simple xorshift64 PRNG step (pure, no external deps).
pub(crate) fn xorshift64(state: &mut u64) -> u64 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    s
}

/// Return a `f64` in `[0, 1)` from the PRNG.
pub(crate) fn rand_f64(state: &mut u64) -> f64 {
    let bits = xorshift64(state);
    (bits >> 11) as f64 / ((1u64 << 53) as f64)
}

/// Generate `n` configurations inside `bounds` using the chosen sampler.
pub(crate) fn sample_configs(
    n: usize,
    bounds: &[(f64, f64)],
    sampler: &ConfigSampler,
    rng: &mut u64,
) -> Vec<Vec<f64>> {
    match sampler {
        ConfigSampler::Random => sample_random(n, bounds, rng),
        ConfigSampler::LatinHypercube => sample_lhs(n, bounds, rng),
    }
}

fn sample_random(n: usize, bounds: &[(f64, f64)], rng: &mut u64) -> Vec<Vec<f64>> {
    (0..n)
        .map(|_| {
            bounds
                .iter()
                .map(|(lo, hi)| lo + rand_f64(rng) * (hi - lo))
                .collect()
        })
        .collect()
}

fn sample_lhs(n: usize, bounds: &[(f64, f64)], rng: &mut u64) -> Vec<Vec<f64>> {
    if n == 0 {
        return Vec::new();
    }
    let d = bounds.len();
    // For each dimension build a permutation of strata indices.
    let mut configs: Vec<Vec<f64>> = vec![vec![0.0; d]; n];
    for dim in 0..d {
        let (lo, hi) = bounds[dim];
        let mut indices: Vec<usize> = (0..n).collect();
        // Fisher-Yates shuffle
        for i in (1..n).rev() {
            let j = (xorshift64(rng) as usize) % (i + 1);
            indices.swap(i, j);
        }
        for (i, &idx) in indices.iter().enumerate() {
            let low = lo + (hi - lo) * (idx as f64) / (n as f64);
            let high = lo + (hi - lo) * ((idx + 1) as f64) / (n as f64);
            configs[i][dim] = low + rand_f64(rng) * (high - low);
        }
    }
    configs
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// A single evaluation record.
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// Unique identifier for this configuration.
    pub config_id: usize,
    /// Hyperparameter values.
    pub config: Vec<f64>,
    /// Resource level used for this evaluation.
    pub budget: f64,
    /// Objective value (lower is better).
    pub objective: f64,
}

/// Aggregated result of a multi-fidelity optimization run.
#[derive(Debug, Clone)]
pub struct MultiFidelityResult {
    /// Best configuration found.
    pub best_config: Vec<f64>,
    /// Objective value of the best configuration.
    pub best_objective: f64,
    /// Total resource budget consumed across all evaluations.
    pub total_budget_used: f64,
    /// Full evaluation log.
    pub evaluations: Vec<EvaluationResult>,
    /// Number of Hyperband brackets (1 for plain Successive Halving).
    pub n_brackets: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_valid() {
        let cfg = MultiFidelityConfig::default();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.eta, 3);
        assert!(cfg.max_budget > cfg.min_budget);
    }

    #[test]
    fn test_invalid_config_eta() {
        let cfg = MultiFidelityConfig {
            eta: 1,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_invalid_config_budget() {
        let cfg = MultiFidelityConfig {
            min_budget: 100.0,
            max_budget: 10.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_s_max_computation() {
        // 81 / 1 = 81, log_3(81) = 4
        let cfg = MultiFidelityConfig::default();
        assert_eq!(cfg.s_max(), 4);
    }

    #[test]
    fn test_random_sampling_bounds() {
        let bounds = vec![(0.0, 1.0), (-5.0, 5.0)];
        let mut rng = 42u64;
        let configs = sample_configs(20, &bounds, &ConfigSampler::Random, &mut rng);
        assert_eq!(configs.len(), 20);
        for c in &configs {
            assert_eq!(c.len(), 2);
            assert!(c[0] >= 0.0 && c[0] <= 1.0);
            assert!(c[1] >= -5.0 && c[1] <= 5.0);
        }
    }

    #[test]
    fn test_lhs_fills_space() {
        let bounds = vec![(0.0, 1.0)];
        let n = 10;
        let mut rng = 123u64;
        let configs = sample_configs(n, &bounds, &ConfigSampler::LatinHypercube, &mut rng);
        assert_eq!(configs.len(), n);
        // Each sample should be in a different stratum [i/n, (i+1)/n).
        // Collect strata and verify all n strata are covered.
        let mut strata: Vec<usize> = configs
            .iter()
            .map(|c| (c[0] * n as f64).floor() as usize)
            .collect();
        strata.sort();
        strata.dedup();
        assert_eq!(strata.len(), n, "LHS must cover all {n} strata");
    }

    #[test]
    fn test_lhs_empty() {
        let bounds = vec![(0.0, 1.0)];
        let mut rng = 1u64;
        let configs = sample_configs(0, &bounds, &ConfigSampler::LatinHypercube, &mut rng);
        assert!(configs.is_empty());
    }

    #[test]
    fn test_evaluation_result_tracks_fields() {
        let er = EvaluationResult {
            config_id: 7,
            config: vec![1.0, 2.0],
            budget: 27.0,
            objective: 0.5,
        };
        assert_eq!(er.config_id, 7);
        assert!((er.budget - 27.0).abs() < f64::EPSILON);
    }
}
