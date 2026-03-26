//! Hyperband algorithm.
//!
//! Hyperband (Li et al., 2017) is an early-stopping-based approach to
//! hyperparameter optimization.  It extends Successive Halving by running
//! multiple brackets with different budget/configuration trade-offs, thereby
//! hedging against the unknown optimal aggressiveness of early stopping.
//!
//! # Reference
//!
//! Li, Jamieson, DeSalvo, Rostamizadeh & Talwalkar (2017).  *Hyperband: A
//! Novel Bandit-Based Approach to Hyperparameter Optimization.* JMLR 18(185).

use crate::error::{OptimizeError, OptimizeResult};

use super::successive_halving::SuccessiveHalving;
use super::types::{ConfigSampler, EvaluationResult, MultiFidelityConfig, MultiFidelityResult};

// ---------------------------------------------------------------------------
// Bracket specification
// ---------------------------------------------------------------------------

/// Configuration for a single Hyperband bracket.
#[derive(Debug, Clone)]
pub(crate) struct BracketConfig {
    /// Number of initial configurations for this bracket.
    pub n_initial: usize,
    /// Starting budget for this bracket.
    pub min_budget: f64,
    /// Maximum budget (same across all brackets).
    pub max_budget: f64,
    /// Number of successive halving rounds in this bracket.
    pub n_rounds: usize,
}

// ---------------------------------------------------------------------------
// Hyperband
// ---------------------------------------------------------------------------

/// Hyperband optimizer.
///
/// Hyperband runs `s_max + 1` brackets of Successive Halving, where
/// `s_max = floor(log_eta(R))` and `R = max_budget / min_budget`.
///
/// Each bracket `s` (from `s_max` down to `0`) trades off the number of
/// initial configurations against the starting budget:
///
/// ```text
/// bracket s:
///   n  = ceil((s_max+1)/(s+1)) * eta^s
///   r  = max_budget / eta^s
/// ```
#[derive(Debug, Clone)]
pub struct Hyperband {
    config: MultiFidelityConfig,
}

impl Hyperband {
    /// Create a new Hyperband instance.
    pub fn new(config: MultiFidelityConfig) -> OptimizeResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Compute the bracket configurations.
    pub(crate) fn compute_brackets(&self) -> Vec<BracketConfig> {
        let s_max = self.config.s_max();
        let eta = self.config.eta;
        let eta_f = eta as f64;
        let mut brackets = Vec::with_capacity(s_max + 1);

        for s in (0..=s_max).rev() {
            let n_initial =
                ((s_max + 1) as f64 / (s + 1) as f64 * eta_f.powi(s as i32)).ceil() as usize;
            let start_budget = self.config.max_budget / eta_f.powi(s as i32);
            brackets.push(BracketConfig {
                n_initial,
                min_budget: start_budget,
                max_budget: self.config.max_budget,
                n_rounds: s + 1,
            });
        }

        brackets
    }

    /// Run Hyperband.
    ///
    /// Iterates over all brackets, runs Successive Halving for each, and
    /// returns the best configuration found across all brackets.
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
        if bounds.is_empty() {
            return Err(OptimizeError::InvalidParameter(
                "bounds must not be empty".into(),
            ));
        }

        let brackets = self.compute_brackets();
        let n_brackets = brackets.len();

        let sh = SuccessiveHalving::new(self.config.clone())?;

        let mut all_evals: Vec<EvaluationResult> = Vec::new();
        let mut total_budget = 0.0;
        let mut global_best_obj = f64::INFINITY;
        let mut global_best_cfg: Vec<f64> = Vec::new();
        let mut eval_id_offset = 0usize;

        for bracket in &brackets {
            let result = sh.run_with(
                objective,
                bounds,
                sampler,
                rng_state,
                bracket.n_initial,
                bracket.min_budget,
            )?;

            // Re-number config IDs to be globally unique
            for mut e in result.evaluations {
                e.config_id += eval_id_offset;
                if e.objective < global_best_obj {
                    global_best_obj = e.objective;
                    global_best_cfg = e.config.clone();
                }
                all_evals.push(e);
            }
            eval_id_offset = all_evals.iter().map(|e| e.config_id).max().unwrap_or(0) + 1;

            total_budget += result.total_budget_used;
        }

        if global_best_cfg.is_empty() {
            return Err(OptimizeError::ComputationError(
                "no evaluations performed across brackets".into(),
            ));
        }

        Ok(MultiFidelityResult {
            best_config: global_best_cfg,
            best_objective: global_best_obj,
            total_budget_used: total_budget,
            evaluations: all_evals,
            n_brackets,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn quadratic(x: &[f64], _budget: f64) -> OptimizeResult<f64> {
        Ok(x.iter().map(|xi| xi * xi).sum())
    }

    /// Budget-aware: penalize with `1/sqrt(budget)` noise-like term.
    fn budget_aware_quadratic(x: &[f64], budget: f64) -> OptimizeResult<f64> {
        let base: f64 = x.iter().map(|xi| xi * xi).sum();
        // Higher budget => more accurate (smaller perturbation)
        Ok(base + 1.0 / budget.sqrt())
    }

    #[test]
    fn test_multiple_brackets_generated() {
        let cfg = MultiFidelityConfig {
            max_budget: 81.0,
            min_budget: 1.0,
            eta: 3,
            n_initial: 0,
        };
        let hb = Hyperband::new(cfg).expect("valid");
        let brackets = hb.compute_brackets();
        // s_max = 4, so 5 brackets
        assert_eq!(brackets.len(), 5);
    }

    #[test]
    fn test_best_across_brackets_selected() {
        let cfg = MultiFidelityConfig {
            max_budget: 27.0,
            min_budget: 1.0,
            eta: 3,
            n_initial: 0,
        };
        let hb = Hyperband::new(cfg).expect("valid");
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let mut rng = 42u64;
        let result = hb
            .run(&quadratic, &bounds, &ConfigSampler::Random, &mut rng)
            .expect("run ok");
        // The reported best should equal the minimum across all evaluations
        let true_min = result
            .evaluations
            .iter()
            .map(|e| e.objective)
            .fold(f64::INFINITY, f64::min);
        assert!(
            (result.best_objective - true_min).abs() < 1e-12,
            "best_objective {} should match minimum evaluation {}",
            result.best_objective,
            true_min
        );
    }

    #[test]
    fn test_total_budget_bounded() {
        let cfg = MultiFidelityConfig {
            max_budget: 27.0,
            min_budget: 1.0,
            eta: 3,
            n_initial: 0,
        };
        let hb = Hyperband::new(cfg).expect("valid");
        let bounds = vec![(-1.0, 1.0)];
        let mut rng = 77u64;
        let result = hb
            .run(&quadratic, &bounds, &ConfigSampler::Random, &mut rng)
            .expect("run ok");
        // Total budget should be finite and positive
        assert!(result.total_budget_used > 0.0);
        assert!(result.total_budget_used.is_finite());
    }

    #[test]
    fn test_converges_to_optimum() {
        let cfg = MultiFidelityConfig {
            max_budget: 81.0,
            min_budget: 1.0,
            eta: 3,
            n_initial: 0,
        };
        let hb = Hyperband::new(cfg).expect("valid");
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let mut rng = 12345u64;
        let result = hb
            .run(&quadratic, &bounds, &ConfigSampler::Random, &mut rng)
            .expect("run ok");
        // With enough configs, best should be reasonably close to 0
        assert!(
            result.best_objective < 5.0,
            "best objective {} should be < 5",
            result.best_objective
        );
    }

    #[test]
    fn test_eta2_vs_eta3_different_brackets() {
        let cfg2 = MultiFidelityConfig {
            max_budget: 64.0,
            min_budget: 1.0,
            eta: 2,
            n_initial: 0,
        };
        let cfg3 = MultiFidelityConfig {
            max_budget: 64.0,
            min_budget: 1.0,
            eta: 3,
            n_initial: 0,
        };
        let hb2 = Hyperband::new(cfg2).expect("valid");
        let hb3 = Hyperband::new(cfg3).expect("valid");
        let brackets2 = hb2.compute_brackets();
        let brackets3 = hb3.compute_brackets();
        // eta=2: s_max = log_2(64) = 6, so 7 brackets
        // eta=3: s_max = floor(log_3(64)) = 3, so 4 brackets
        assert_eq!(brackets2.len(), 7, "eta=2 should have 7 brackets");
        assert_eq!(brackets3.len(), 4, "eta=3 should have 4 brackets");
    }

    #[test]
    fn test_budget_aware_objective() {
        let cfg = MultiFidelityConfig {
            max_budget: 27.0,
            min_budget: 1.0,
            eta: 3,
            n_initial: 0,
        };
        let hb = Hyperband::new(cfg).expect("valid");
        let bounds = vec![(-3.0, 3.0)];
        let mut rng = 55u64;
        let result = hb
            .run(
                &budget_aware_quadratic,
                &bounds,
                &ConfigSampler::LatinHypercube,
                &mut rng,
            )
            .expect("run ok");
        assert!(result.best_objective.is_finite());
        assert!(result.n_brackets > 1);
    }

    #[test]
    fn test_empty_bounds_error() {
        let cfg = MultiFidelityConfig::default();
        let hb = Hyperband::new(cfg).expect("valid");
        let result = hb.run(&quadratic, &[], &ConfigSampler::Random, &mut 1u64);
        assert!(result.is_err());
    }

    #[test]
    fn test_bracket_budgets_reach_max() {
        let cfg = MultiFidelityConfig {
            max_budget: 81.0,
            min_budget: 1.0,
            eta: 3,
            n_initial: 0,
        };
        let hb = Hyperband::new(cfg).expect("valid");
        let brackets = hb.compute_brackets();
        for b in &brackets {
            assert!(
                (b.max_budget - 81.0).abs() < 1e-9,
                "all brackets should share the same max_budget"
            );
        }
    }

    #[test]
    fn test_n_brackets_in_result() {
        let cfg = MultiFidelityConfig {
            max_budget: 27.0,
            min_budget: 1.0,
            eta: 3,
            n_initial: 0,
        };
        let hb = Hyperband::new(cfg).expect("valid");
        let bounds = vec![(-1.0, 1.0)];
        let mut rng = 1u64;
        let result = hb
            .run(&quadratic, &bounds, &ConfigSampler::Random, &mut rng)
            .expect("run ok");
        let expected = hb.compute_brackets().len();
        assert_eq!(result.n_brackets, expected);
    }
}
