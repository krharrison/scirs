//! SPEA2 (Strength Pareto Evolutionary Algorithm 2)
//!
//! An improved version of SPEA with fine-grained fitness assignment,
//! density estimation using k-th nearest neighbor, and archive truncation
//! that preserves boundary solutions.
//!
//! Key features:
//! - Strength-based fitness: raw fitness = sum of strengths of dominators
//! - Density estimation: 1 / (sigma_k + 2) where sigma_k is k-th nearest neighbor distance
//! - Combined fitness: F(i) = R(i) + D(i) where R = raw fitness, D = density
//! - Archive truncation preserves boundary solutions via iterative removal of
//!   the individual with smallest distance to its nearest neighbor
//!
//! # References
//!
//! - Zitzler, Laumanns & Thiele, "SPEA2: Improving the Strength Pareto
//!   Evolutionary Algorithm", TIK-Report 103, ETH Zurich, 2001

use super::{utils, MultiObjectiveConfig, MultiObjectiveOptimizer};
use crate::error::OptimizeError;
use crate::multi_objective::crossover::{CrossoverOperator, SimulatedBinaryCrossover};
use crate::multi_objective::mutation::{MutationOperator, PolynomialMutation};
use crate::multi_objective::solutions::{MultiObjectiveResult, MultiObjectiveSolution, Population};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, RngExt, SeedableRng};

/// SPEA2 optimizer
pub struct SPEA2 {
    config: MultiObjectiveConfig,
    archive_size: usize,
    n_objectives: usize,
    n_variables: usize,
    /// Archive of non-dominated and best solutions
    archive: Vec<MultiObjectiveSolution>,
    population: Population,
    generation: usize,
    n_evaluations: usize,
    rng: StdRng,
    crossover: SimulatedBinaryCrossover,
    mutation: PolynomialMutation,
    convergence_history: Vec<f64>,
}

impl SPEA2 {
    /// Create new SPEA2 optimizer
    pub fn new(population_size: usize, n_objectives: usize, n_variables: usize) -> Self {
        let config = MultiObjectiveConfig {
            population_size,
            ..Default::default()
        };
        Self::with_config(config, n_objectives, n_variables)
    }

    /// Create SPEA2 with full configuration
    pub fn with_config(
        config: MultiObjectiveConfig,
        n_objectives: usize,
        n_variables: usize,
    ) -> Self {
        let archive_size = config.archive_size.unwrap_or(config.population_size);

        let seed = config.random_seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(42)
        });

        let rng = StdRng::seed_from_u64(seed);

        let crossover =
            SimulatedBinaryCrossover::new(config.crossover_eta, config.crossover_probability);
        let mutation = PolynomialMutation::new(config.mutation_probability, config.mutation_eta);

        Self {
            config,
            archive_size,
            n_objectives,
            n_variables,
            archive: Vec::new(),
            population: Population::new(),
            generation: 0,
            n_evaluations: 0,
            rng,
            crossover,
            mutation,
            convergence_history: Vec::new(),
        }
    }

    /// Evaluate a single individual
    fn evaluate_individual<F>(
        &mut self,
        variables: &Array1<f64>,
        objective_function: &F,
    ) -> Result<Array1<f64>, OptimizeError>
    where
        F: Fn(&ArrayView1<f64>) -> Array1<f64> + Send + Sync,
    {
        self.n_evaluations += 1;

        if let Some(max_evals) = self.config.max_evaluations {
            if self.n_evaluations > max_evals {
                return Err(OptimizeError::MaxEvaluationsReached);
            }
        }

        let objectives = objective_function(&variables.view());
        if objectives.len() != self.n_objectives {
            return Err(OptimizeError::InvalidInput(format!(
                "Expected {} objectives, got {}",
                self.n_objectives,
                objectives.len()
            )));
        }

        Ok(objectives)
    }

    /// Check if solution a Pareto-dominates solution b
    fn dominates(a: &MultiObjectiveSolution, b: &MultiObjectiveSolution) -> bool {
        let mut at_least_one_better = false;
        for i in 0..a.objectives.len() {
            if a.objectives[i] > b.objectives[i] {
                return false;
            }
            if a.objectives[i] < b.objectives[i] {
                at_least_one_better = true;
            }
        }
        at_least_one_better
    }

    /// Calculate strength values for each individual
    /// S(i) = |{j | j in P+A, i dominates j}|
    fn calculate_strengths(combined: &[MultiObjectiveSolution]) -> Vec<usize> {
        let n = combined.len();
        let mut strengths = vec![0usize; n];

        for i in 0..n {
            for j in 0..n {
                if i != j && Self::dominates(&combined[i], &combined[j]) {
                    strengths[i] += 1;
                }
            }
        }

        strengths
    }

    /// Calculate raw fitness R(i) = sum of strengths of all dominators of i
    fn calculate_raw_fitness(combined: &[MultiObjectiveSolution], strengths: &[usize]) -> Vec<f64> {
        let n = combined.len();
        let mut raw_fitness = vec![0.0f64; n];

        for i in 0..n {
            for j in 0..n {
                if i != j && Self::dominates(&combined[j], &combined[i]) {
                    raw_fitness[i] += strengths[j] as f64;
                }
            }
        }

        raw_fitness
    }

    /// Calculate density estimation D(i) = 1 / (sigma_k + 2)
    /// where sigma_k is the distance to the k-th nearest neighbor
    fn calculate_density(combined: &[MultiObjectiveSolution]) -> Vec<f64> {
        let n = combined.len();
        if n <= 1 {
            return vec![0.0; n];
        }

        // k = sqrt(N) as recommended in the paper
        let k = (n as f64).sqrt().floor() as usize;
        let k = k.max(1).min(n - 1);

        let mut densities = vec![0.0f64; n];

        for i in 0..n {
            // Compute distances to all other individuals in objective space
            let mut distances: Vec<f64> = Vec::with_capacity(n - 1);
            for j in 0..n {
                if i != j {
                    let dist = euclidean_distance_objectives(&combined[i], &combined[j]);
                    distances.push(dist);
                }
            }

            // Sort distances to find k-th nearest
            distances.sort_by(|a, b| a.total_cmp(b));

            let sigma_k = if k <= distances.len() {
                distances[k - 1]
            } else {
                distances.last().copied().unwrap_or(0.0)
            };

            densities[i] = 1.0 / (sigma_k + 2.0);
        }

        densities
    }

    /// Calculate total fitness F(i) = R(i) + D(i)
    fn calculate_fitness(combined: &[MultiObjectiveSolution]) -> Vec<f64> {
        let strengths = Self::calculate_strengths(combined);
        let raw_fitness = Self::calculate_raw_fitness(combined, &strengths);
        let densities = Self::calculate_density(combined);

        raw_fitness
            .iter()
            .zip(densities.iter())
            .map(|(r, d)| r + d)
            .collect()
    }

    /// Archive truncation procedure
    /// If archive is too large, iteratively remove the individual with the smallest
    /// distance to its nearest neighbor (ties broken by second-nearest, etc.)
    fn truncate_archive(&self, candidates: &mut Vec<(MultiObjectiveSolution, f64)>) {
        while candidates.len() > self.archive_size {
            let n = candidates.len();

            // Compute all pairwise distances in objective space
            let mut dist_matrix = vec![vec![0.0f64; n]; n];
            for i in 0..n {
                for j in (i + 1)..n {
                    let d = euclidean_distance_objectives(&candidates[i].0, &candidates[j].0);
                    dist_matrix[i][j] = d;
                    dist_matrix[j][i] = d;
                }
            }

            // For each individual, sort its distances to find nearest neighbors
            let mut sorted_distances: Vec<Vec<(f64, usize)>> = Vec::with_capacity(n);
            for i in 0..n {
                let mut dists: Vec<(f64, usize)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (dist_matrix[i][j], j))
                    .collect();
                dists.sort_by(|a, b| a.0.total_cmp(&b.0));
                sorted_distances.push(dists);
            }

            // Find the individual to remove:
            // The one with the smallest nearest-neighbor distance,
            // breaking ties by second-nearest, etc.
            let mut remove_idx = 0;
            for candidate_idx in 1..n {
                let mut is_smaller = false;
                for k in 0..sorted_distances[0]
                    .len()
                    .min(sorted_distances[candidate_idx].len())
                {
                    let d_current = sorted_distances[remove_idx]
                        .get(k)
                        .map(|(d, _)| *d)
                        .unwrap_or(f64::INFINITY);
                    let d_candidate = sorted_distances[candidate_idx]
                        .get(k)
                        .map(|(d, _)| *d)
                        .unwrap_or(f64::INFINITY);

                    if d_candidate < d_current - 1e-15 {
                        is_smaller = true;
                        break;
                    } else if d_candidate > d_current + 1e-15 {
                        break;
                    }
                    // If equal, continue to next neighbor
                }
                if is_smaller {
                    remove_idx = candidate_idx;
                }
            }

            candidates.remove(remove_idx);
        }
    }

    /// Environmental selection: update archive from combined population + archive
    fn environmental_selection(
        &self,
        combined: &[MultiObjectiveSolution],
        fitness: &[f64],
    ) -> Vec<MultiObjectiveSolution> {
        // Select all individuals with fitness < 1 (non-dominated)
        let mut next_archive: Vec<(MultiObjectiveSolution, f64)> = combined
            .iter()
            .zip(fitness.iter())
            .filter(|(_, &f)| f < 1.0)
            .map(|(sol, &f)| (sol.clone(), f))
            .collect();

        if next_archive.len() < self.archive_size {
            // Fill with best dominated individuals
            let mut dominated: Vec<(MultiObjectiveSolution, f64)> = combined
                .iter()
                .zip(fitness.iter())
                .filter(|(_, &f)| f >= 1.0)
                .map(|(sol, &f)| (sol.clone(), f))
                .collect();

            // Sort by fitness (lower is better)
            dominated.sort_by(|a, b| a.1.total_cmp(&b.1));

            let remaining = self.archive_size - next_archive.len();
            next_archive.extend(dominated.into_iter().take(remaining));
        } else if next_archive.len() > self.archive_size {
            // Truncate using the SPEA2 truncation procedure
            self.truncate_archive(&mut next_archive);
        }

        next_archive.into_iter().map(|(sol, _)| sol).collect()
    }

    /// Binary tournament selection based on fitness
    fn binary_tournament_selection(
        &mut self,
        archive: &[MultiObjectiveSolution],
        fitness: &[f64],
        n_select: usize,
    ) -> Vec<MultiObjectiveSolution> {
        let n = archive.len();
        if n == 0 {
            return vec![];
        }

        let mut selected = Vec::with_capacity(n_select);

        for _ in 0..n_select {
            let idx1 = self.rng.random_range(0..n);
            let idx2 = self.rng.random_range(0..n);

            let winner = if fitness[idx1] <= fitness[idx2] {
                idx1
            } else {
                idx2
            };
            selected.push(archive[winner].clone());
        }

        selected
    }

    /// Create offspring from the archive via crossover and mutation
    fn create_offspring<F>(
        &mut self,
        archive: &[MultiObjectiveSolution],
        archive_fitness: &[f64],
        objective_function: &F,
    ) -> Result<Vec<MultiObjectiveSolution>, OptimizeError>
    where
        F: Fn(&ArrayView1<f64>) -> Array1<f64> + Send + Sync,
    {
        let mut offspring = Vec::new();

        while offspring.len() < self.config.population_size {
            let parents = self.binary_tournament_selection(archive, archive_fitness, 2);
            if parents.len() < 2 {
                break;
            }

            let p1_vars = parents[0].variables.as_slice().unwrap_or(&[]);
            let p2_vars = parents[1].variables.as_slice().unwrap_or(&[]);

            let (mut c1_vars, mut c2_vars) = self.crossover.crossover(p1_vars, p2_vars);

            let bounds: Vec<(f64, f64)> = if let Some((lower, upper)) = &self.config.bounds {
                lower
                    .iter()
                    .zip(upper.iter())
                    .map(|(&l, &u)| (l, u))
                    .collect()
            } else {
                vec![(-1.0, 1.0); self.n_variables]
            };

            self.mutation.mutate(&mut c1_vars, &bounds);
            self.mutation.mutate(&mut c2_vars, &bounds);

            let c1_arr = Array1::from_vec(c1_vars);
            let c1_obj = self.evaluate_individual(&c1_arr, objective_function)?;
            offspring.push(MultiObjectiveSolution::new(c1_arr, c1_obj));

            if offspring.len() < self.config.population_size {
                let c2_arr = Array1::from_vec(c2_vars);
                let c2_obj = self.evaluate_individual(&c2_arr, objective_function)?;
                offspring.push(MultiObjectiveSolution::new(c2_arr, c2_obj));
            }
        }

        Ok(offspring)
    }

    /// Calculate metrics
    fn calculate_metrics(&mut self) {
        if let Some(ref_point) = &self.config.reference_point {
            let pareto_front = extract_pareto_front_from_slice(&self.archive);
            let hv = utils::calculate_hypervolume(&pareto_front, ref_point);
            self.convergence_history.push(hv);
        }
    }
}

impl MultiObjectiveOptimizer for SPEA2 {
    fn optimize<F>(&mut self, objective_function: F) -> Result<MultiObjectiveResult, OptimizeError>
    where
        F: Fn(&ArrayView1<f64>) -> Array1<f64> + Send + Sync,
    {
        self.initialize_population()?;

        // Generate and evaluate initial population
        let initial_vars = utils::generate_random_population(
            self.config.population_size,
            self.n_variables,
            &self.config.bounds,
        );

        let mut initial_solutions = Vec::new();
        for vars in initial_vars {
            let objs = self.evaluate_individual(&vars, &objective_function)?;
            initial_solutions.push(MultiObjectiveSolution::new(vars, objs));
        }

        self.population = Population::from_solutions(initial_solutions);
        self.archive.clear();

        // Main loop
        while self.generation < self.config.max_generations {
            if self.check_convergence() {
                break;
            }
            self.evolve_generation(&objective_function)?;
        }

        // Extract results
        let pareto_front = extract_pareto_front_from_slice(&self.archive);
        let hypervolume = self
            .config
            .reference_point
            .as_ref()
            .map(|rp| utils::calculate_hypervolume(&pareto_front, rp));

        let mut result = MultiObjectiveResult::new(
            pareto_front,
            self.archive.clone(),
            self.n_evaluations,
            self.generation,
        );
        result.hypervolume = hypervolume;
        result.metrics.convergence_history = self.convergence_history.clone();

        Ok(result)
    }

    fn evolve_generation<F>(&mut self, objective_function: &F) -> Result<(), OptimizeError>
    where
        F: Fn(&ArrayView1<f64>) -> Array1<f64> + Send + Sync,
    {
        // Combine population and archive
        let mut combined: Vec<MultiObjectiveSolution> = self.population.solutions().to_vec();
        combined.extend(self.archive.clone());

        // Calculate fitness for combined set
        let fitness = Self::calculate_fitness(&combined);

        // Environmental selection -> new archive
        self.archive = self.environmental_selection(&combined, &fitness);

        // Compute archive fitness for mating selection
        let archive_clone = self.archive.clone();
        let archive_fitness = Self::calculate_fitness(&archive_clone);

        // Create offspring from archive
        let offspring =
            self.create_offspring(&archive_clone, &archive_fitness, objective_function)?;

        self.population = Population::from_solutions(offspring);
        self.generation += 1;
        self.calculate_metrics();

        Ok(())
    }

    fn initialize_population(&mut self) -> Result<(), OptimizeError> {
        self.population.clear();
        self.archive.clear();
        self.generation = 0;
        self.n_evaluations = 0;
        self.convergence_history.clear();
        Ok(())
    }

    fn check_convergence(&self) -> bool {
        if let Some(max_evals) = self.config.max_evaluations {
            if self.n_evaluations >= max_evals {
                return true;
            }
        }

        if self.convergence_history.len() >= 10 {
            let recent = &self.convergence_history[self.convergence_history.len() - 10..];
            let max_hv = recent.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let min_hv = recent.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            if (max_hv - min_hv) < self.config.tolerance {
                return true;
            }
        }

        false
    }

    fn get_population(&self) -> &Population {
        &self.population
    }

    fn get_generation(&self) -> usize {
        self.generation
    }

    fn get_evaluations(&self) -> usize {
        self.n_evaluations
    }

    fn name(&self) -> &str {
        "SPEA2"
    }
}

/// Euclidean distance between two solutions in objective space
fn euclidean_distance_objectives(a: &MultiObjectiveSolution, b: &MultiObjectiveSolution) -> f64 {
    a.objectives
        .iter()
        .zip(b.objectives.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Extract Pareto front from a slice of solutions
fn extract_pareto_front_from_slice(
    solutions: &[MultiObjectiveSolution],
) -> Vec<MultiObjectiveSolution> {
    let mut pareto_front: Vec<MultiObjectiveSolution> = Vec::new();

    for candidate in solutions {
        let mut is_dominated = false;
        for existing in &pareto_front {
            if SPEA2::dominates(existing, candidate) {
                is_dominated = true;
                break;
            }
        }
        if !is_dominated {
            pareto_front.retain(|existing| !SPEA2::dominates(candidate, existing));
            pareto_front.push(candidate.clone());
        }
    }

    pareto_front
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, s};

    fn zdt1(x: &ArrayView1<f64>) -> Array1<f64> {
        let f1 = x[0];
        let g = 1.0 + 9.0 * x.slice(s![1..]).sum() / (x.len() - 1) as f64;
        let f2 = g * (1.0 - (f1 / g).sqrt());
        array![f1, f2]
    }

    #[test]
    fn test_spea2_creation() {
        let spea2 = SPEA2::new(100, 2, 3);
        assert_eq!(spea2.n_objectives, 2);
        assert_eq!(spea2.n_variables, 3);
        assert_eq!(spea2.archive_size, 100);
        assert_eq!(spea2.generation, 0);
    }

    #[test]
    fn test_spea2_with_config() {
        let config = MultiObjectiveConfig {
            population_size: 50,
            archive_size: Some(30),
            max_generations: 10,
            random_seed: Some(42),
            ..Default::default()
        };

        let spea2 = SPEA2::with_config(config, 2, 3);
        assert_eq!(spea2.archive_size, 30);
    }

    #[test]
    fn test_spea2_optimize_zdt1() {
        let config = MultiObjectiveConfig {
            max_generations: 10,
            population_size: 20,
            bounds: Some((Array1::zeros(3), Array1::ones(3))),
            random_seed: Some(42),
            ..Default::default()
        };

        let mut spea2 = SPEA2::with_config(config, 2, 3);
        let result = spea2.optimize(zdt1);

        assert!(result.is_ok());
        let res = result.expect("should succeed");
        assert!(res.success);
        assert!(!res.pareto_front.is_empty());
        assert!(res.n_evaluations > 0);
    }

    #[test]
    fn test_spea2_strength_calculation() {
        let solutions = vec![
            MultiObjectiveSolution::new(array![0.0], array![1.0, 3.0]),
            MultiObjectiveSolution::new(array![1.0], array![2.0, 2.0]),
            MultiObjectiveSolution::new(array![2.0], array![3.0, 1.0]),
            MultiObjectiveSolution::new(array![3.0], array![4.0, 4.0]), // Dominated
        ];

        let strengths = SPEA2::calculate_strengths(&solutions);
        // Solution [4,4] is dominated by all 3 non-dominated solutions
        // Each non-dominated solution dominates at least the dominated one
        assert!(strengths[0] >= 1);
        assert!(strengths[1] >= 1);
        assert!(strengths[2] >= 1);
    }

    #[test]
    fn test_spea2_fitness_calculation() {
        let solutions = vec![
            MultiObjectiveSolution::new(array![0.0], array![1.0, 3.0]),
            MultiObjectiveSolution::new(array![1.0], array![2.0, 2.0]),
            MultiObjectiveSolution::new(array![2.0], array![4.0, 4.0]), // Dominated by both
        ];

        let fitness = SPEA2::calculate_fitness(&solutions);
        // Non-dominated solutions should have lower fitness than dominated ones
        assert!(fitness[0] < fitness[2] || fitness[1] < fitness[2]);
    }

    #[test]
    fn test_spea2_dominance() {
        let a = MultiObjectiveSolution::new(array![0.0], array![1.0, 2.0]);
        let b = MultiObjectiveSolution::new(array![0.0], array![2.0, 3.0]);
        let c = MultiObjectiveSolution::new(array![0.0], array![0.5, 3.5]);

        assert!(SPEA2::dominates(&a, &b)); // a dominates b
        assert!(!SPEA2::dominates(&b, &a));
        assert!(!SPEA2::dominates(&a, &c)); // Neither dominates the other
        assert!(!SPEA2::dominates(&c, &a));
    }

    #[test]
    fn test_spea2_max_evaluations() {
        let config = MultiObjectiveConfig {
            max_generations: 1000,
            max_evaluations: Some(50),
            population_size: 10,
            bounds: Some((Array1::zeros(3), Array1::ones(3))),
            random_seed: Some(42),
            ..Default::default()
        };

        let mut spea2 = SPEA2::with_config(config, 2, 3);
        let result = spea2.optimize(zdt1);
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        assert!(res.n_evaluations <= 60);
    }

    #[test]
    fn test_spea2_name() {
        let spea2 = SPEA2::new(50, 2, 3);
        assert_eq!(spea2.name(), "SPEA2");
    }

    #[test]
    fn test_euclidean_distance() {
        let a = MultiObjectiveSolution::new(array![0.0], array![0.0, 0.0]);
        let b = MultiObjectiveSolution::new(array![0.0], array![3.0, 4.0]);
        let dist = euclidean_distance_objectives(&a, &b);
        assert!((dist - 5.0).abs() < 1e-10);
    }
}
