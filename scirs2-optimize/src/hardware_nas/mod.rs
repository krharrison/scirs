//! Hardware-Aware Neural Architecture Search
//!
//! Adds a latency (and parameter-count) model on top of architecture search so that
//! only architectures that fit within a hardware budget are considered.
//!
//! ## What is included
//!
//! - `LatencyTable`: maps operation names × input sizes to latency in milliseconds.
//! - `NasObjective`: single or multi-objective formulation.
//! - `HardwareNasSearcher`: random search and evolutionary search over candidate
//!   architectures, with constraint filtering and Pareto-front extraction.
//!
//! ## References
//!
//! - Cai, H. et al. (2019). "ProxylessNAS: Direct Neural Architecture Search on
//!   Target Task and Hardware". ICLR 2019.
//! - Tan, M. & Le, Q.V. (2019). "EfficientNet: Rethinking Model Scaling for
//!   Convolutional Neural Networks". ICML 2019.

use std::collections::HashMap;

use crate::darts::Operation;
use crate::error::OptimizeError;

// ──────────────────────────────────────────────────────────── LatencyTable ──

/// Lookup table mapping `(operation_name, input_size)` → latency in milliseconds.
///
/// The default constructor populates a set of representative estimates.
#[derive(Debug, Clone)]
pub struct LatencyTable {
    /// Raw latency data: key = operation name, value = base latency (ms).
    pub op_latencies: HashMap<String, f64>,
    /// Scale factor applied per input-size unit.
    pub size_scale: f64,
}

impl LatencyTable {
    /// Create a new table with default hardware latency estimates.
    ///
    /// Values are representative of a mid-range mobile CPU at 224×224 feature map.
    pub fn new() -> Self {
        let mut op_latencies = HashMap::new();
        op_latencies.insert("conv3x3".to_string(), 1.5);
        op_latencies.insert("conv5x5".to_string(), 3.0);
        op_latencies.insert("max_pool".to_string(), 0.2);
        op_latencies.insert("avg_pool".to_string(), 0.2);
        op_latencies.insert("identity".to_string(), 0.05);
        op_latencies.insert("skip_connect".to_string(), 0.05);
        op_latencies.insert("zero".to_string(), 0.0);
        Self {
            op_latencies,
            size_scale: 1e-4, // latency per unit of input_size beyond a base
        }
    }

    /// Latency for a single operation given `input_size` (e.g., H*W*C).
    ///
    /// Uses a simple linear model: latency = base + size_scale * input_size.
    pub fn latency_of(&self, op: &str, input_size: usize) -> f64 {
        let base = self.op_latencies.get(op).cloned().unwrap_or(1.0);
        base + self.size_scale * input_size as f64
    }

    /// Total latency for a sequence of `(operation_name, input_size)` pairs.
    pub fn total_latency(&self, arch: &[(String, usize)]) -> f64 {
        arch.iter().map(|(op, sz)| self.latency_of(op, *sz)).sum()
    }
}

impl Default for LatencyTable {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────── NasObjective ──

/// Optimisation objective for hardware-aware NAS.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum NasObjective {
    /// Maximise accuracy only (no latency constraint in the objective).
    Accuracy,
    /// Minimise latency only.
    Latency,
    /// Minimise FLOPs.
    FlopsCount,
    /// Minimise parameter count.
    ParamCount,
    /// Linear scalarisation: `accuracy_weight * accuracy - latency_weight * latency`.
    MultiObjective {
        /// Weight on accuracy.
        accuracy_weight: f64,
        /// Weight on latency.
        latency_weight: f64,
    },
}

impl Default for NasObjective {
    fn default() -> Self {
        NasObjective::MultiObjective {
            accuracy_weight: 1.0,
            latency_weight: 0.01,
        }
    }
}

// ──────────────────────────────────────────────────────── HardwareNasConfig ──

/// Configuration for hardware-aware NAS.
#[derive(Debug, Clone)]
pub struct HardwareNasConfig {
    /// Maximum allowed latency in milliseconds.
    pub max_latency_ms: f64,
    /// Maximum allowed number of parameters.
    pub max_params: usize,
    /// Number of random search iterations (or initial population size for evolution).
    pub n_search_iter: usize,
    /// Optimisation objective.
    pub objective: NasObjective,
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// Number of operations to include in each candidate architecture.
    pub n_ops_per_arch: usize,
    /// Input size (H*W*C) used for latency estimation.
    pub input_size: usize,
    /// Number of parameters assumed for each operation (simplified model).
    pub params_per_op: usize,
    /// Population size for evolutionary search.
    pub population_size: usize,
    /// Tournament size for evolutionary selection.
    pub tournament_size: usize,
    /// Number of generations for evolutionary search.
    pub n_generations: usize,
}

impl Default for HardwareNasConfig {
    fn default() -> Self {
        Self {
            max_latency_ms: 10.0,
            max_params: 1_000_000,
            n_search_iter: 100,
            objective: NasObjective::default(),
            seed: 42,
            n_ops_per_arch: 8,
            input_size: 224 * 224 * 3,
            params_per_op: 9 * 16 * 16, // 3×3 conv, C=16
            population_size: 20,
            tournament_size: 3,
            n_generations: 10,
        }
    }
}

// ───────────────────────────────────────────────────────── ArchCandidate ──

/// A concrete architecture candidate with performance estimates.
#[derive(Debug, Clone)]
pub struct ArchCandidate {
    /// Sequence of operations.
    pub operations: Vec<Operation>,
    /// Estimated top-1 accuracy (fraction in [0, 1]).
    pub estimated_accuracy: f64,
    /// Estimated latency in milliseconds.
    pub estimated_latency: f64,
    /// Estimated parameter count.
    pub n_params: usize,
}

impl ArchCandidate {
    /// Scalar objective value (higher is better).
    pub fn objective_value(&self, obj: &NasObjective) -> f64 {
        match obj {
            NasObjective::Accuracy => self.estimated_accuracy,
            NasObjective::Latency => -self.estimated_latency,
            NasObjective::FlopsCount => -(self.n_params as f64), // use params as proxy for FLOPs
            NasObjective::ParamCount => -(self.n_params as f64),
            NasObjective::MultiObjective {
                accuracy_weight,
                latency_weight,
            } => {
                accuracy_weight * self.estimated_accuracy - latency_weight * self.estimated_latency
            }
        }
    }
}

// ─────────────────────────────────────────────────── HardwareNasSearcher ──

/// Hardware-aware NAS searcher.
#[derive(Debug)]
pub struct HardwareNasSearcher {
    config: HardwareNasConfig,
    latency_table: LatencyTable,
    /// Internal LCG state.
    rng_state: u64,
}

impl HardwareNasSearcher {
    /// Create a new searcher.
    pub fn new(config: HardwareNasConfig, latency_table: LatencyTable) -> Self {
        let rng_state = config.seed;
        Self {
            config,
            latency_table,
            rng_state,
        }
    }

    // ── LCG random number generator ──────────────────────────────────────

    /// Advance LCG and return next u64.
    fn lcg_next(&mut self) -> u64 {
        // Knuth's multiplicative LCG
        self.rng_state = self
            .rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.rng_state
    }

    /// Sample a uniformly random `usize` in `0..n`.
    fn rand_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.lcg_next() as usize) % n
    }

    /// Sample a uniformly random f64 in `[0, 1)`.
    fn rand_f64(&mut self) -> f64 {
        (self.lcg_next() >> 11) as f64 / (1u64 << 53) as f64
    }

    // ── Architecture sampling helpers ─────────────────────────────────────

    /// Sample a random architecture (sequence of `n_ops_per_arch` operations).
    fn sample_random_arch(&mut self) -> Vec<Operation> {
        let ops = Operation::all();
        let n = self.config.n_ops_per_arch;
        (0..n).map(|_| ops[self.rand_usize(ops.len())]).collect()
    }

    /// Estimate latency for an architecture using the latency table.
    fn estimate_latency(&self, ops: &[Operation]) -> f64 {
        let pairs: Vec<(String, usize)> = ops
            .iter()
            .map(|o| (o.name().to_string(), self.config.input_size))
            .collect();
        self.latency_table.total_latency(&pairs)
    }

    /// Estimate parameter count for an architecture.
    fn estimate_params(&self, ops: &[Operation]) -> usize {
        ops.iter()
            .map(|o| match o {
                Operation::Zero | Operation::Identity | Operation::SkipConnect => 0,
                Operation::MaxPool | Operation::AvgPool => 0,
                Operation::Conv3x3 => self.config.params_per_op,
                Operation::Conv5x5 => self.config.params_per_op * 2,
            })
            .sum()
    }

    /// Check whether a candidate satisfies the hardware constraints.
    fn satisfies_constraints(&self, candidate: &ArchCandidate) -> bool {
        candidate.estimated_latency <= self.config.max_latency_ms
            && candidate.n_params <= self.config.max_params
    }

    /// Build an `ArchCandidate` for given ops using the provided accuracy estimate.
    fn build_candidate(&mut self, ops: Vec<Operation>, accuracy: f64) -> ArchCandidate {
        let latency = self.estimate_latency(&ops);
        let n_params = self.estimate_params(&ops);
        ArchCandidate {
            operations: ops,
            estimated_accuracy: accuracy,
            estimated_latency: latency,
            n_params,
        }
    }

    // ── Public search methods ─────────────────────────────────────────────

    /// Random search: sample `n_search_iter` architectures, evaluate with `eval_fn`,
    /// filter by constraints, return the best.
    ///
    /// `eval_fn` receives a slice of `Operation` and returns an accuracy estimate
    /// in `[0, 1]`.
    ///
    /// Returns an error if no candidate satisfies the hardware constraints.
    pub fn random_search(
        &mut self,
        eval_fn: impl Fn(&[Operation]) -> f64,
    ) -> Result<ArchCandidate, OptimizeError> {
        let mut best: Option<ArchCandidate> = None;
        let obj = self.config.objective.clone();

        for _ in 0..self.config.n_search_iter {
            let ops = self.sample_random_arch();
            let acc = eval_fn(&ops);
            let candidate = self.build_candidate(ops, acc);
            if !self.satisfies_constraints(&candidate) {
                continue;
            }
            match &best {
                None => best = Some(candidate),
                Some(b) => {
                    if candidate.objective_value(&obj) > b.objective_value(&obj) {
                        best = Some(candidate);
                    }
                }
            }
        }

        best.ok_or_else(|| {
            OptimizeError::ConvergenceError(
                "No architecture found satisfying hardware constraints".to_string(),
            )
        })
    }

    /// Evolutionary search: start with a random population, apply tournament
    /// selection and random mutation for `n_generations`, return the best
    /// constraint-satisfying candidate found.
    pub fn evolutionary_search(
        &mut self,
        eval_fn: impl Fn(&[Operation]) -> f64,
    ) -> Result<ArchCandidate, OptimizeError> {
        let pop_size = self.config.population_size;
        let obj = self.config.objective.clone();

        // Initialise population.
        let mut population: Vec<ArchCandidate> = (0..pop_size)
            .map(|_| {
                let ops = self.sample_random_arch();
                let acc = eval_fn(&ops);
                self.build_candidate(ops, acc)
            })
            .collect();

        let mut best: Option<ArchCandidate> = population
            .iter()
            .filter(|c| self.satisfies_constraints(c))
            .max_by(|a, b| {
                a.objective_value(&obj)
                    .partial_cmp(&b.objective_value(&obj))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned();

        for _gen in 0..self.config.n_generations {
            let mut next_pop: Vec<ArchCandidate> = Vec::with_capacity(pop_size);

            for _ in 0..pop_size {
                // Tournament selection.
                let parent = self.tournament_select(&population, &obj);
                // Mutate: randomly swap one operation.
                let child_ops = self.mutate(&parent.operations);
                let acc = eval_fn(&child_ops);
                let child = self.build_candidate(child_ops, acc);

                if self.satisfies_constraints(&child) {
                    match &best {
                        None => best = Some(child.clone()),
                        Some(b) => {
                            if child.objective_value(&obj) > b.objective_value(&obj) {
                                best = Some(child.clone());
                            }
                        }
                    }
                }
                next_pop.push(child);
            }
            population = next_pop;
        }

        best.ok_or_else(|| {
            OptimizeError::ConvergenceError(
                "Evolutionary search: no constraint-satisfying architecture found".to_string(),
            )
        })
    }

    /// Tournament selection: sample `tournament_size` candidates uniformly at
    /// random, return a clone of the one with the best objective.
    fn tournament_select(
        &mut self,
        population: &[ArchCandidate],
        obj: &NasObjective,
    ) -> ArchCandidate {
        let t = self.config.tournament_size.min(population.len()).max(1);
        let mut best_idx = self.rand_usize(population.len());
        for _ in 1..t {
            let idx = self.rand_usize(population.len());
            if population[idx].objective_value(obj) > population[best_idx].objective_value(obj) {
                best_idx = idx;
            }
        }
        population[best_idx].clone()
    }

    /// Mutation: randomly replace one operation with another candidate operation.
    fn mutate(&mut self, ops: &[Operation]) -> Vec<Operation> {
        if ops.is_empty() {
            return Vec::new();
        }
        let mut new_ops = ops.to_vec();
        let pos = self.rand_usize(new_ops.len());
        let all_ops = Operation::all();
        new_ops[pos] = all_ops[self.rand_usize(all_ops.len())];
        new_ops
    }

    /// Compute the Pareto front of a set of candidates w.r.t.
    /// `(estimated_accuracy, -estimated_latency)` (both maximised).
    ///
    /// Returns the indices of non-dominated candidates.
    pub fn pareto_front(candidates: &[ArchCandidate]) -> Vec<usize> {
        let n = candidates.len();
        let mut dominated = vec![false; n];

        for i in 0..n {
            if dominated[i] {
                continue;
            }
            for j in 0..n {
                if i == j || dominated[j] {
                    continue;
                }
                // Does j dominate i?
                let j_dom_i = candidates[j].estimated_accuracy >= candidates[i].estimated_accuracy
                    && candidates[j].estimated_latency <= candidates[i].estimated_latency
                    && (candidates[j].estimated_accuracy > candidates[i].estimated_accuracy
                        || candidates[j].estimated_latency < candidates[i].estimated_latency);
                if j_dom_i {
                    dominated[i] = true;
                    break;
                }
            }
        }

        (0..n).filter(|&i| !dominated[i]).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════ tests ═══

#[cfg(test)]
mod tests {
    use super::*;

    fn make_searcher() -> HardwareNasSearcher {
        HardwareNasSearcher::new(HardwareNasConfig::default(), LatencyTable::new())
    }

    /// A simple accuracy oracle: favour architectures with more identity/skip ops.
    fn acc_oracle(ops: &[Operation]) -> f64 {
        let light_count = ops
            .iter()
            .filter(|o| matches!(o, Operation::Identity | Operation::SkipConnect))
            .count();
        0.5 + 0.05 * light_count as f64
    }

    #[test]
    fn latency_table_default_contains_ops() {
        let lt = LatencyTable::new();
        assert!(lt.latency_of("conv3x3", 1000) > 0.0);
        assert_eq!(lt.latency_of("zero", 0), 0.0);
    }

    #[test]
    fn total_latency_sums_correctly() {
        let lt = LatencyTable::new();
        let arch = vec![("conv3x3".to_string(), 0), ("max_pool".to_string(), 0)];
        let total = lt.total_latency(&arch);
        let expected = lt.latency_of("conv3x3", 0) + lt.latency_of("max_pool", 0);
        assert!((total - expected).abs() < 1e-12);
    }

    #[test]
    fn random_search_finds_valid_candidate() {
        let mut config = HardwareNasConfig::default();
        // Use a very loose latency budget to ensure we always find something.
        config.max_latency_ms = 10_000.0;
        config.n_search_iter = 50;
        config.n_ops_per_arch = 4;
        let mut searcher = HardwareNasSearcher::new(config, LatencyTable::new());
        let result = searcher.random_search(acc_oracle);
        assert!(result.is_ok(), "Should find a valid candidate");
        let cand = result.unwrap();
        assert!(cand.estimated_latency <= 10_000.0);
    }

    #[test]
    fn pareto_front_returns_non_dominated_subset() {
        let candidates = vec![
            ArchCandidate {
                operations: vec![],
                estimated_accuracy: 0.9,
                estimated_latency: 5.0,
                n_params: 100,
            },
            ArchCandidate {
                operations: vec![],
                estimated_accuracy: 0.8,
                estimated_latency: 3.0,
                n_params: 80,
            },
            ArchCandidate {
                operations: vec![],
                estimated_accuracy: 0.7,
                estimated_latency: 8.0, // dominated by both above
                n_params: 90,
            },
        ];
        let front = HardwareNasSearcher::pareto_front(&candidates);
        assert!(
            front.contains(&0),
            "high accuracy / moderate latency should be on front"
        );
        assert!(
            front.contains(&1),
            "low latency / moderate accuracy should be on front"
        );
        assert!(
            !front.contains(&2),
            "dominated candidate should not be on front"
        );
    }

    #[test]
    fn evolutionary_search_runs() {
        let mut config = HardwareNasConfig::default();
        config.max_latency_ms = 10_000.0;
        config.population_size = 10;
        config.n_generations = 5;
        config.n_ops_per_arch = 4;
        let mut searcher = HardwareNasSearcher::new(config, LatencyTable::new());
        let result = searcher.evolutionary_search(acc_oracle);
        assert!(
            result.is_ok(),
            "Evolutionary search should find a candidate"
        );
    }

    #[test]
    fn pareto_front_single_candidate() {
        let candidates = vec![ArchCandidate {
            operations: vec![],
            estimated_accuracy: 0.85,
            estimated_latency: 4.0,
            n_params: 50,
        }];
        let front = HardwareNasSearcher::pareto_front(&candidates);
        assert_eq!(front, vec![0]);
    }
}
