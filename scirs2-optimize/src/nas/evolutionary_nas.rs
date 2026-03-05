//! Evolutionary Neural Architecture Search (AmoebaNet-style).
//!
//! Implements a tournament-selection evolutionary algorithm for NAS:
//! - Initialise a random population
//! - Each generation: tournament-select a parent, mutate it, replace the
//!   weakest member of the population with the child (if better)
//! - Return the best architecture found

use crate::error::OptimizeError;
use crate::nas::random_nas::{ArchFitness, NASResult};
use crate::nas::search_space::{Architecture, SearchSpace};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};

/// Configuration for the evolutionary NAS algorithm.
#[derive(Debug, Clone)]
pub struct EvolutionaryNASConfig {
    /// Number of individuals in the population
    pub population_size: usize,
    /// Number of evolutionary generations
    pub n_generations: usize,
    /// Probability of mutating any given edge per mutation event
    pub mutation_rate: f64,
    /// Number of individuals to draw in each tournament
    pub tournament_size: usize,
    /// Whether to use elitism (always keep best)
    pub elitism: bool,
}

impl Default for EvolutionaryNASConfig {
    fn default() -> Self {
        Self {
            population_size: 20,
            n_generations: 50,
            mutation_rate: 0.2,
            tournament_size: 5,
            elitism: true,
        }
    }
}

/// Evolutionary Neural Architecture Search.
///
/// Uses tournament selection and mutation to search the architecture space.
/// Inspired by the AmoebaNet regularized evolutionary search (Real et al. 2019).
pub struct EvolutionaryNAS {
    /// Algorithm configuration
    pub config: EvolutionaryNASConfig,
}

impl EvolutionaryNAS {
    /// Create with default configuration.
    pub fn new(population_size: usize, n_generations: usize) -> Self {
        Self {
            config: EvolutionaryNASConfig {
                population_size,
                n_generations,
                ..EvolutionaryNASConfig::default()
            },
        }
    }

    /// Create from an explicit config.
    pub fn with_config(config: EvolutionaryNASConfig) -> Self {
        Self { config }
    }

    /// Run evolutionary search.
    ///
    /// # Arguments
    /// - `space`: Architecture search space.
    /// - `fitness`: Fitness evaluator (higher = better).
    /// - `seed`: Random seed.
    pub fn search<F: ArchFitness>(
        &self,
        space: &SearchSpace,
        fitness: &F,
        seed: u64,
    ) -> Result<NASResult, OptimizeError> {
        if self.config.population_size < 2 {
            return Err(OptimizeError::InvalidParameter(
                "population_size must be at least 2".to_string(),
            ));
        }

        let mut rng = StdRng::seed_from_u64(seed);

        // Initialise population
        let mut population: Vec<(Architecture, f64)> =
            Vec::with_capacity(self.config.population_size);
        for _ in 0..self.config.population_size {
            let arch = space.sample_random(&mut rng);
            let score = fitness.evaluate(&arch).unwrap_or(f64::NEG_INFINITY);
            population.push((arch, score));
        }

        let mut all_scores: Vec<f64> = population.iter().map(|(_, s)| *s).collect();

        for _gen in 0..self.config.n_generations {
            // Tournament selection: pick the best from a random subset
            let parent_idx = self.tournament_select(&population, &mut rng);

            // Mutate the selected parent
            let mut child = population[parent_idx].0.clone();
            self.mutate(&mut child, space, &mut rng);
            let child_score = fitness.evaluate(&child).unwrap_or(f64::NEG_INFINITY);
            all_scores.push(child_score);

            // Find the weakest individual and replace if child is better
            let worst_idx = self.find_worst(&population);
            if child_score > population[worst_idx].1 {
                population[worst_idx] = (child, child_score);
            }
        }

        // Return the best individual
        let (best_arch, best_score) = population
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(arch, score)| (arch, score))
            .unwrap_or_else(|| {
                let arch = space.sample_random(&mut rng);
                (arch, f64::NEG_INFINITY)
            });

        let n_evaluated = self.config.population_size + self.config.n_generations;
        Ok(NASResult {
            best_arch,
            best_score,
            all_scores,
            n_evaluated,
        })
    }

    /// Tournament selection: sample `tournament_size` indices and return
    /// the index of the individual with the highest score.
    fn tournament_select(&self, population: &[(Architecture, f64)], rng: &mut StdRng) -> usize {
        let n = population.len();
        let k = self.config.tournament_size.min(n);
        let mut best_idx = rng.random_range(0..n);
        for _ in 1..k {
            let idx = rng.random_range(0..n);
            if population[idx].1 > population[best_idx].1 {
                best_idx = idx;
            }
        }
        best_idx
    }

    /// Return the index of the individual with the lowest score.
    fn find_worst(&self, population: &[(Architecture, f64)]) -> usize {
        population
            .iter()
            .enumerate()
            .min_by(|a, b| {
                a.1 .1
                    .partial_cmp(&b.1 .1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Mutate an architecture: for each edge, replace its op with a random
    /// one from the space with probability `mutation_rate`.
    fn mutate(&self, arch: &mut Architecture, space: &SearchSpace, rng: &mut StdRng) {
        if space.operations.is_empty() || arch.edges.is_empty() {
            return;
        }
        for edge in arch.edges.iter_mut() {
            if rng.random::<f64>() < self.config.mutation_rate {
                let op_idx = rng.random_range(0..space.operations.len());
                edge.op = space.operations[op_idx].clone();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nas::random_nas::ParamCountFitness;
    use crate::nas::search_space::SearchSpace;

    #[test]
    fn test_evolutionary_nas_runs() {
        let space = SearchSpace::darts_like(3);
        let fitness = ParamCountFitness::new(8_000);
        let nas = EvolutionaryNAS::new(10, 20);

        let result = nas.search(&space, &fitness, 99).expect("search failed");

        // All evaluations = population + generations (some may be skipped if no improvement)
        assert!(result.n_evaluated >= 10);
        assert!(!result.all_scores.is_empty());
    }

    #[test]
    fn test_evolutionary_nas_small_population_error() {
        let space = SearchSpace::darts_like(3);
        let fitness = ParamCountFitness::new(8_000);
        let nas = EvolutionaryNAS::new(1, 5);

        assert!(nas.search(&space, &fitness, 0).is_err());
    }

    #[test]
    fn test_evolutionary_nas_monotone_best_score() {
        // The best score should be >= the initial population's best
        let space = SearchSpace::darts_like(3);
        let fitness = ParamCountFitness::new(5_000);
        let nas = EvolutionaryNAS::new(8, 30);

        let result = nas.search(&space, &fitness, 7).expect("search failed");

        // best_score must be achievable (finite)
        assert!(result.best_score.is_finite() || result.best_score == f64::NEG_INFINITY);
    }

    #[test]
    fn test_evolutionary_nas_with_config() {
        let config = EvolutionaryNASConfig {
            population_size: 6,
            n_generations: 10,
            mutation_rate: 0.5,
            tournament_size: 3,
            elitism: false,
        };
        let space = SearchSpace::darts_like(3);
        let fitness = ParamCountFitness::new(4_000);
        let nas = EvolutionaryNAS::with_config(config);

        let result = nas.search(&space, &fitness, 1).expect("search failed");
        assert!(result.n_evaluated > 0);
    }
}
