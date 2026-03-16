//! Random Neural Architecture Search baseline.
//!
//! Implements the simplest NAS strategy: uniformly sample architectures
//! from the search space and evaluate each with a fitness function.
//! Serves as a strong baseline for more sophisticated methods.

use crate::error::OptimizeError;
use crate::nas::search_space::{Architecture, SearchSpace};
use scirs2_core::random::{rngs::StdRng, SeedableRng};

/// Result of a NAS search run.
#[derive(Debug, Clone)]
pub struct NASResult {
    /// Architecture with the highest fitness score found
    pub best_arch: Architecture,
    /// Fitness score of the best architecture
    pub best_score: f64,
    /// All fitness scores in the order they were evaluated
    pub all_scores: Vec<f64>,
    /// Total number of architectures evaluated
    pub n_evaluated: usize,
}

/// Trait for evaluating the fitness of an architecture.
///
/// Implementations should be deterministic for reproducibility.
pub trait ArchFitness: Send + Sync {
    /// Evaluate an architecture and return a scalar fitness score.
    ///
    /// Higher scores are better (the search maximizes this value).
    fn evaluate(&self, arch: &Architecture) -> Result<f64, OptimizeError>;
}

/// Proxy fitness: score based on closeness to a target parameter count.
///
/// Returns 0.0 when the architecture matches the target exactly,
/// with negative scores proportional to relative deviation.
pub struct ParamCountFitness {
    /// Desired number of parameters
    pub target_params: usize,
}

impl ParamCountFitness {
    /// Create a new `ParamCountFitness` with the given target.
    pub fn new(target_params: usize) -> Self {
        Self { target_params }
    }
}

impl ArchFitness for ParamCountFitness {
    fn evaluate(&self, arch: &Architecture) -> Result<f64, OptimizeError> {
        let params = arch.total_params() as f64;
        let target = self.target_params as f64;
        if target == 0.0 {
            return Ok(if params == 0.0 { 0.0 } else { -1.0 });
        }
        Ok(-(params - target).abs() / target)
    }
}

/// Proxy fitness based on FLOPs efficiency at a given spatial resolution.
pub struct FlopsFitness {
    /// Maximum FLOPs budget (architectures exceeding this are penalized)
    pub flops_budget: usize,
    /// Spatial dimension used for FLOPs estimation
    pub spatial: usize,
}

impl FlopsFitness {
    /// Create a new `FlopsFitness`.
    pub fn new(flops_budget: usize, spatial: usize) -> Self {
        Self {
            flops_budget,
            spatial,
        }
    }
}

impl ArchFitness for FlopsFitness {
    fn evaluate(&self, arch: &Architecture) -> Result<f64, OptimizeError> {
        let flops = arch.total_flops(self.spatial) as f64;
        let budget = self.flops_budget as f64;
        if budget == 0.0 {
            return Ok(0.0);
        }
        // Reward for staying under budget; penalize excess
        if flops <= budget {
            Ok(flops / budget)
        } else {
            Ok(-(flops - budget) / budget)
        }
    }
}

/// Random Neural Architecture Search.
///
/// Samples `n_trials` architectures uniformly at random from the search
/// space and returns the one with the highest fitness score.
pub struct RandomNAS {
    /// Number of random architectures to evaluate
    pub n_trials: usize,
}

impl RandomNAS {
    /// Create a new `RandomNAS` with the specified trial budget.
    pub fn new(n_trials: usize) -> Self {
        Self { n_trials }
    }

    /// Run random search over the given `space` using `fitness`.
    ///
    /// # Arguments
    /// - `space`: The architecture search space to sample from.
    /// - `fitness`: Fitness evaluator (higher = better).
    /// - `seed`: Random seed for reproducibility.
    pub fn search<F: ArchFitness>(
        &self,
        space: &SearchSpace,
        fitness: &F,
        seed: u64,
    ) -> Result<NASResult, OptimizeError> {
        use scirs2_core::random::{Rng, RngExt};

        if self.n_trials == 0 {
            return Err(OptimizeError::InvalidParameter(
                "n_trials must be at least 1".to_string(),
            ));
        }

        let mut rng = StdRng::seed_from_u64(seed);

        let mut best_score = f64::NEG_INFINITY;
        // Sample the initial best architecture before the loop
        let mut best_arch = space.sample_random(&mut rng);
        let mut all_scores = Vec::with_capacity(self.n_trials);

        for _ in 0..self.n_trials {
            let arch = space.sample_random(&mut rng);
            let score = fitness.evaluate(&arch)?;
            all_scores.push(score);
            if score > best_score {
                best_score = score;
                best_arch = arch;
            }
        }

        Ok(NASResult {
            best_arch,
            best_score,
            all_scores,
            n_evaluated: self.n_trials,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nas::search_space::SearchSpace;

    #[test]
    fn test_random_nas_returns_result() {
        let space = SearchSpace::darts_like(3);
        let fitness = ParamCountFitness::new(10_000);
        let nas = RandomNAS::new(20);

        let result = nas.search(&space, &fitness, 0).expect("search failed");

        assert_eq!(result.n_evaluated, 20);
        assert_eq!(result.all_scores.len(), 20);
        // best_score is finite
        assert!(result.best_score.is_finite());
    }

    #[test]
    fn test_random_nas_zero_trials_errors() {
        let space = SearchSpace::darts_like(3);
        let fitness = ParamCountFitness::new(10_000);
        let nas = RandomNAS::new(0);

        assert!(nas.search(&space, &fitness, 0).is_err());
    }

    #[test]
    fn test_param_count_fitness_exact_match() {
        let mut arch = Architecture::new(1, 32, 10);
        // Architecture with zero params
        let fitness = ParamCountFitness::new(0);
        let score = fitness.evaluate(&arch).expect("eval failed");
        assert_eq!(score, 0.0);

        // Arch with non-zero params vs 0 target
        use crate::nas::search_space::{ArchEdge, ArchNode, OpType};
        arch.nodes.push(ArchNode {
            id: 0,
            name: "n0".into(),
            output_channels: 32,
        });
        arch.nodes.push(ArchNode {
            id: 1,
            name: "n1".into(),
            output_channels: 32,
        });
        arch.edges.push(ArchEdge {
            from: 0,
            to: 1,
            op: OpType::Conv3x3,
        });
        let fitness2 = ParamCountFitness::new(0);
        let score2 = fitness2.evaluate(&arch).expect("eval failed");
        assert_eq!(score2, -1.0);
    }

    #[test]
    fn test_flops_fitness_under_budget() {
        use crate::nas::search_space::{ArchEdge, ArchNode, OpType};
        let mut arch = Architecture::new(1, 8, 10);
        arch.nodes.push(ArchNode {
            id: 0,
            name: "n0".into(),
            output_channels: 8,
        });
        arch.nodes.push(ArchNode {
            id: 1,
            name: "n1".into(),
            output_channels: 8,
        });
        arch.edges.push(ArchEdge {
            from: 0,
            to: 1,
            op: OpType::Skip,
        });

        let fitness = FlopsFitness::new(1_000_000, 8);
        let score = fitness.evaluate(&arch).expect("eval failed");
        // Skip has near-zero flops, so score should be between 0 and 1
        assert!(score >= 0.0 && score <= 1.0);
    }
}
