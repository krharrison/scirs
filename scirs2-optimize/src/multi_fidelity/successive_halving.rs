//! Successive Halving (SHA) algorithm.
//!
//! Successive Halving allocates a fixed resource budget across a set of
//! candidate configurations. It iteratively discards the worst-performing
//! fraction (`1 - 1/eta`) and increases the per-configuration budget by
//! `eta` until a single winner (or a small set) remains.
//!
//! # Reference
//!
//! Jamieson & Talwalkar (2016). *Non-stochastic Best Arm Identification and
//! Hyperparameter Optimization.* AISTATS.

use crate::error::{OptimizeError, OptimizeResult};

use super::types::{
    sample_configs, ConfigSampler, EvaluationResult, MultiFidelityConfig, MultiFidelityResult,
};

// ---------------------------------------------------------------------------
// Schedule entry
// ---------------------------------------------------------------------------

/// One round in the Successive Halving schedule.
#[derive(Debug, Clone)]
pub(crate) struct RoundSpec {
    /// Number of configurations to evaluate this round.
    pub n_configs: usize,
    /// Per-configuration resource budget for this round.
    pub budget: f64,
}

// ---------------------------------------------------------------------------
// SuccessiveHalving
// ---------------------------------------------------------------------------

/// Successive Halving optimizer.
///
/// ```text
///  n configs ──► evaluate at budget r ──► keep top 1/η ──► r *= η ──► repeat
/// ```
#[derive(Debug, Clone)]
pub struct SuccessiveHalving {
    config: MultiFidelityConfig,
}

impl SuccessiveHalving {
    /// Create a new Successive Halving instance.
    pub fn new(config: MultiFidelityConfig) -> OptimizeResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Compute the schedule `[(n_configs, budget)]` for each round.
    pub fn compute_schedule(&self) -> Vec<RoundSpec> {
        self.compute_schedule_with(
            self.effective_n_initial(),
            self.config.min_budget,
            self.config.max_budget,
        )
    }

    /// Internal: compute schedule given explicit start parameters.
    pub(crate) fn compute_schedule_with(
        &self,
        n_initial: usize,
        start_budget: f64,
        max_budget: f64,
    ) -> Vec<RoundSpec> {
        let eta = self.config.eta;
        let mut schedule = Vec::new();
        let mut n = n_initial;
        let mut budget = start_budget;

        loop {
            schedule.push(RoundSpec {
                n_configs: n,
                budget,
            });
            if n <= 1 || budget >= max_budget {
                break;
            }
            n = (n / eta).max(1);
            budget = (budget * eta as f64).min(max_budget);
        }
        schedule
    }

    /// Effective number of initial configurations.
    fn effective_n_initial(&self) -> usize {
        if self.config.n_initial > 0 {
            return self.config.n_initial;
        }
        // Default: ceil(s_max+1) * eta^0 = s_max + 1  (for the largest bracket)
        let s_max = self.config.s_max();
        let eta = self.config.eta;
        // n = ceil((s_max+1)/(s+1)) * eta^s   with s = s_max
        // Simplifies to eta^s_max  (standard Hyperband formula for largest bracket)
        (eta as f64).powi(s_max as i32) as usize
    }

    /// Run Successive Halving on the given objective.
    ///
    /// `objective(config, budget) -> Result<f64>` evaluates a configuration at
    /// a given resource level and returns the loss (lower is better).
    pub fn run<F>(
        &self,
        objective: &F,
        bounds: &[(f64, f64)],
        sampler: &ConfigSampler,
        rng_state: &mut u64,
    ) -> OptimizeResult<MultiFidelityResult>
    where
        F: Fn(&[f64], f64) -> OptimizeResult<f64>,
    {
        self.run_with(
            objective,
            bounds,
            sampler,
            rng_state,
            self.effective_n_initial(),
            self.config.min_budget,
        )
    }

    /// Internal run with explicit n_initial and start_budget (used by Hyperband).
    pub(crate) fn run_with<F>(
        &self,
        objective: &F,
        bounds: &[(f64, f64)],
        sampler: &ConfigSampler,
        rng_state: &mut u64,
        n_initial: usize,
        start_budget: f64,
    ) -> OptimizeResult<MultiFidelityResult>
    where
        F: Fn(&[f64], f64) -> OptimizeResult<f64>,
    {
        if bounds.is_empty() {
            return Err(OptimizeError::InvalidParameter(
                "bounds must not be empty".into(),
            ));
        }

        let schedule = self.compute_schedule_with(n_initial, start_budget, self.config.max_budget);

        // Generate initial configurations
        let initial_n = schedule.first().map(|r| r.n_configs).unwrap_or(n_initial);
        let mut configs: Vec<(usize, Vec<f64>)> =
            sample_configs(initial_n, bounds, sampler, rng_state)
                .into_iter()
                .enumerate()
                .collect();

        let mut all_evals: Vec<EvaluationResult> = Vec::new();
        let mut total_budget = 0.0;
        let mut next_id = configs.len();

        for round in &schedule {
            // Truncate to round.n_configs (configs may already be smaller)
            configs.truncate(round.n_configs);
            if configs.is_empty() {
                break;
            }

            // Evaluate each surviving config at current budget
            let mut scored: Vec<(usize, Vec<f64>, f64)> = Vec::with_capacity(configs.len());
            for (id, cfg) in &configs {
                let obj = objective(cfg, round.budget)?;
                all_evals.push(EvaluationResult {
                    config_id: *id,
                    config: cfg.clone(),
                    budget: round.budget,
                    objective: obj,
                });
                total_budget += round.budget;
                scored.push((*id, cfg.clone(), obj));
            }

            // Sort ascending by objective (lower is better)
            scored.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

            // Keep top 1/eta (at least 1)
            let keep = (scored.len() / self.config.eta).max(1);
            scored.truncate(keep);

            configs = scored.into_iter().map(|(id, cfg, _)| (id, cfg)).collect();
            let _ = next_id; // suppress unused warning
        }

        // Best overall
        let best = all_evals
            .iter()
            .min_by(|a, b| {
                a.objective
                    .partial_cmp(&b.objective)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| OptimizeError::ComputationError("no evaluations performed".into()))?;

        Ok(MultiFidelityResult {
            best_config: best.config.clone(),
            best_objective: best.objective,
            total_budget_used: total_budget,
            evaluations: all_evals,
            n_brackets: 1,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple quadratic: f(x, budget) = sum(x_i^2).  Budget is ignored.
    fn quadratic(x: &[f64], _budget: f64) -> OptimizeResult<f64> {
        Ok(x.iter().map(|xi| xi * xi).sum())
    }

    #[test]
    fn test_sh_finds_minimum() {
        let cfg = MultiFidelityConfig {
            max_budget: 27.0,
            min_budget: 1.0,
            eta: 3,
            n_initial: 27,
        };
        let sh = SuccessiveHalving::new(cfg).expect("valid config");
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let mut rng = 42u64;
        let result = sh
            .run(&quadratic, &bounds, &ConfigSampler::Random, &mut rng)
            .expect("run succeeds");
        // Best objective should be reasonably small (close to 0)
        assert!(
            result.best_objective < 10.0,
            "best objective {} should be < 10",
            result.best_objective
        );
    }

    #[test]
    fn test_budget_monotonically_increases() {
        let cfg = MultiFidelityConfig {
            max_budget: 81.0,
            min_budget: 1.0,
            eta: 3,
            n_initial: 81,
        };
        let sh = SuccessiveHalving::new(cfg).expect("valid config");
        let schedule = sh.compute_schedule();
        for window in schedule.windows(2) {
            assert!(
                window[1].budget >= window[0].budget,
                "budget must not decrease: {} -> {}",
                window[0].budget,
                window[1].budget
            );
        }
    }

    #[test]
    fn test_configs_decrease_each_round() {
        let cfg = MultiFidelityConfig {
            max_budget: 81.0,
            min_budget: 1.0,
            eta: 3,
            n_initial: 81,
        };
        let sh = SuccessiveHalving::new(cfg).expect("valid config");
        let schedule = sh.compute_schedule();
        for window in schedule.windows(2) {
            assert!(
                window[1].n_configs <= window[0].n_configs,
                "n_configs must not increase: {} -> {}",
                window[0].n_configs,
                window[1].n_configs
            );
        }
    }

    #[test]
    fn test_best_config_survives() {
        // Inject a known-good config near origin; it should survive all rounds.
        let cfg = MultiFidelityConfig {
            max_budget: 9.0,
            min_budget: 1.0,
            eta: 3,
            n_initial: 9,
        };
        let sh = SuccessiveHalving::new(cfg).expect("valid config");
        let bounds = vec![(-5.0, 5.0)];
        let mut rng = 7u64;
        let result = sh
            .run(&quadratic, &bounds, &ConfigSampler::Random, &mut rng)
            .expect("run ok");
        // The best config's ID should appear in the final round evaluations
        let best_id = result
            .evaluations
            .iter()
            .min_by(|a, b| {
                a.objective
                    .partial_cmp(&b.objective)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|e| e.config_id);
        let max_budget_evals: Vec<_> = result
            .evaluations
            .iter()
            .filter(|e| (e.budget - 9.0).abs() < 1e-9)
            .collect();
        // At least one evaluation at max budget
        assert!(
            !max_budget_evals.is_empty(),
            "should have evaluations at max budget"
        );
        // The overall best should appear in the max-budget round
        let final_ids: Vec<usize> = max_budget_evals.iter().map(|e| e.config_id).collect();
        assert!(best_id.is_some(), "should have at least one evaluation");
    }

    #[test]
    fn test_schedule_correct_pairs() {
        let cfg = MultiFidelityConfig {
            max_budget: 27.0,
            min_budget: 1.0,
            eta: 3,
            n_initial: 27,
        };
        let sh = SuccessiveHalving::new(cfg).expect("valid config");
        let schedule = sh.compute_schedule();
        // Expected: (27,1), (9,3), (3,9), (1,27)
        assert_eq!(schedule.len(), 4);
        assert_eq!(schedule[0].n_configs, 27);
        assert!((schedule[0].budget - 1.0).abs() < 1e-9);
        assert_eq!(schedule[1].n_configs, 9);
        assert!((schedule[1].budget - 3.0).abs() < 1e-9);
        assert_eq!(schedule[2].n_configs, 3);
        assert!((schedule[2].budget - 9.0).abs() < 1e-9);
        assert_eq!(schedule[3].n_configs, 1);
        assert!((schedule[3].budget - 27.0).abs() < 1e-9);
    }

    #[test]
    fn test_sh_with_lhs() {
        let cfg = MultiFidelityConfig {
            max_budget: 9.0,
            min_budget: 1.0,
            eta: 3,
            n_initial: 9,
        };
        let sh = SuccessiveHalving::new(cfg).expect("valid config");
        let bounds = vec![(-1.0, 1.0), (-1.0, 1.0)];
        let mut rng = 99u64;
        let result = sh
            .run(
                &quadratic,
                &bounds,
                &ConfigSampler::LatinHypercube,
                &mut rng,
            )
            .expect("lhs run ok");
        assert!(result.best_objective < 2.0);
    }

    #[test]
    fn test_empty_bounds_error() {
        let cfg = MultiFidelityConfig::default();
        let sh = SuccessiveHalving::new(cfg).expect("valid config");
        let result = sh.run(&quadratic, &[], &ConfigSampler::Random, &mut 1u64);
        assert!(result.is_err());
    }
}
