//! Integration tests for the DRO module (WS169).
//!
//! Covers Wasserstein DRO, CVaR DRO, portfolio DRO, and multi-fidelity
//! Hyperband / Successive Halving, providing the ~20 tests required by WS169.

#[cfg(test)]
mod ws169_tests {
    use crate::dro::{
        portfolio_dro, portfolio_erm, solve_cvar_dro, CvarEstimator, DroConfig, DroResult,
        DroSolver, RobustObjective, WassersteinBall, WassersteinDro,
    };
    use crate::multi_fidelity::{ConfigSampler, Hyperband, MultiFidelityConfig, SuccessiveHalving};

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Simple LCG for deterministic sample generation.
    fn lcg_next(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn make_returns(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut s = seed.wrapping_add(1);
        (0..n)
            .map(|_| (0..d).map(|_| lcg_next(&mut s) * 0.1 - 0.02).collect())
            .collect()
    }

    fn make_samples(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
        make_returns(n, d, seed)
    }

    fn objective_fn(x: &[f64], _budget: f64) -> crate::error::OptimizeResult<f64> {
        Ok(x.iter().map(|xi| xi * xi).sum())
    }

    // ── CVaR tests ───────────────────────────────────────────────────────────

    /// CVaR at α=0.9 on [0,1,...,9]: top 10% = [9] → CVaR = 9.
    #[test]
    fn test_cvar_computation() {
        let losses: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let est = CvarEstimator::new(0.9).expect("valid alpha");
        let cvar = est.compute_cvar(&losses).expect("compute ok");
        assert!((cvar - 9.0).abs() < 0.5, "CVaR_0.9([0..9]) ≈ 9, got {cvar}");
    }

    /// CVaR_α ≥ mean for any distribution (coherent risk measure property).
    #[test]
    fn test_cvar_symmetry() {
        let losses = vec![2.0, 5.0, 1.0, 8.0, 4.0, 3.0];
        let mean = losses.iter().sum::<f64>() / losses.len() as f64;
        let est = CvarEstimator::new(0.5).expect("valid");
        let cvar = est.compute_cvar(&losses).expect("ok");
        assert!(
            cvar >= mean - 1e-9,
            "CVaR ({cvar}) should be >= mean ({mean})"
        );
    }

    /// CVaR at α very close to 1 should approach the maximum loss.
    #[test]
    fn test_cvar_at_one() {
        let losses = vec![1.0, 2.0, 5.0, 10.0, 3.0];
        let max_loss = losses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let est = CvarEstimator::new(0.99).expect("valid");
        let cvar = est.compute_cvar(&losses).expect("ok");
        assert!(
            cvar >= max_loss - 1e-6,
            "CVaR at alpha≈1 should be >= max loss ({max_loss}), got {cvar}"
        );
    }

    /// CVaR at α close to 0 should approximate the mean.
    #[test]
    fn test_cvar_at_zero() {
        let losses = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = losses.iter().sum::<f64>() / losses.len() as f64;
        let est = CvarEstimator::new(0.01).expect("valid");
        let cvar = est.compute_cvar(&losses).expect("ok");
        assert!(
            (cvar - mean).abs() < 0.5,
            "CVaR at alpha≈0 should be close to mean ({mean}), got {cvar}"
        );
    }

    // ── Wasserstein DRO tests ─────────────────────────────────────────────────

    /// Wasserstein DRO solver should converge (loss should be finite and not NaN).
    #[test]
    fn test_wasserstein_dro_converges() {
        let loss_fn = |w: &[f64], x: &[f64]| -> f64 {
            -w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum::<f64>()
        };
        let grad_fn = |_w: &[f64], x: &[f64]| -> Vec<f64> { x.iter().map(|xi| -xi).collect() };
        let samples = make_samples(30, 3, 42);
        let cfg = DroConfig {
            radius: 0.05,
            max_iter: 300,
            tol: 1e-6,
            ..Default::default()
        };
        let solver = WassersteinDro::new(cfg, &loss_fn, &grad_fn).expect("valid");
        let result = solver.solve(3, &samples).expect("solve ok");
        assert!(result.primal_obj.is_finite(), "primal_obj must be finite");
        assert!(
            result.worst_case_loss.is_finite(),
            "worst_case_loss must be finite"
        );
    }

    /// Portfolio weights must sum to 1.
    #[test]
    fn test_portfolio_dro_weights_sum_to_one() {
        let returns = make_returns(50, 4, 11);
        let result = portfolio_dro(&returns, 0.05, None).expect("dro ok");
        let sum: f64 = result.optimal_weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "portfolio weights must sum to 1, got {sum}"
        );
    }

    /// Portfolio weights must all be non-negative (no short selling).
    #[test]
    fn test_portfolio_dro_positive_weights() {
        let returns = make_returns(50, 3, 99);
        let result = portfolio_dro(&returns, 0.1, None).expect("dro ok");
        for (i, &w) in result.optimal_weights.iter().enumerate() {
            assert!(w >= -1e-8, "weight[{i}] = {w} must be non-negative");
        }
    }

    /// DRO with ε=0 should give the same primal objective as plain ERM.
    #[test]
    fn test_dro_radius_zero_equals_erm() {
        let returns = make_returns(30, 2, 7);
        let dro = portfolio_dro(&returns, 0.0, None).expect("dro ok");
        let erm = portfolio_erm(&returns, None).expect("erm ok");
        assert!(
            (dro.primal_obj - erm.primal_obj).abs() < 1e-3,
            "DRO(ε=0) primal_obj ({}) should equal ERM primal_obj ({})",
            dro.primal_obj,
            erm.primal_obj
        );
    }

    /// Larger radius → more conservative (larger) worst-case loss.
    #[test]
    fn test_dro_larger_radius_more_conservative() {
        let returns = make_returns(40, 3, 55);
        let r1 = portfolio_dro(&returns, 0.001, None).expect("dro small ok");
        let r2 = portfolio_dro(&returns, 0.5, None).expect("dro large ok");
        assert!(
            r2.worst_case_loss >= r1.worst_case_loss - 1e-4,
            "larger radius ({:.2}) should yield higher worst-case loss ({:.4} >= {:.4})",
            0.5,
            r2.worst_case_loss,
            r1.worst_case_loss
        );
    }

    /// Distance from the Wasserstein ball centre to one of its own samples is 0.
    #[test]
    fn test_wasserstein_ball_center() {
        let sample = vec![1.0, 2.0, 3.0];
        let ball = WassersteinBall::new(vec![sample.clone()], 0.5).expect("valid");
        let d = ball.distance_to_point(&sample);
        assert!(
            d < 1e-10,
            "distance from centre to itself must be 0, got {d}"
        );
    }

    /// DroResult fields must all be non-NaN.
    #[test]
    fn test_dro_result_fields() {
        let returns = make_returns(20, 2, 33);
        let result = portfolio_dro(&returns, 0.05, None).expect("dro ok");
        assert!(!result.worst_case_loss.is_nan(), "worst_case_loss is NaN");
        assert!(!result.primal_obj.is_nan(), "primal_obj is NaN");
        for (i, &w) in result.optimal_weights.iter().enumerate() {
            assert!(!w.is_nan(), "weight[{i}] is NaN");
        }
    }

    /// CVaR-DRO converges (primal_obj finite, not NaN).
    #[test]
    fn test_cvar_dro_converges() {
        let result =
            solve_cvar_dro(3, &make_samples(40, 3, 17), 0.8, 0.05, None).expect("cvar dro ok");
        assert!(result.primal_obj.is_finite(), "primal_obj not finite");
        assert!(
            result.worst_case_loss.is_finite(),
            "worst_case_loss not finite"
        );
    }

    // ── Multi-Fidelity tests ──────────────────────────────────────────────────

    /// Default MultiFidelityConfig should be valid.
    #[test]
    fn test_multi_fidelity_config_default() {
        let cfg = MultiFidelityConfig::default();
        assert!(cfg.validate().is_ok(), "default config must be valid");
    }

    /// Hyperband with max_budget=81, min_budget=1, eta=3 should have 5 brackets.
    #[test]
    fn test_hyperband_bracket_count() {
        let cfg = MultiFidelityConfig {
            max_budget: 81.0,
            min_budget: 1.0,
            eta: 3,
            n_initial: 0,
        };
        let hb = Hyperband::new(cfg).expect("valid");
        // s_max = floor(log_3(81)) = 4, so 5 brackets.
        assert_eq!(hb.compute_brackets().len(), 5, "should have 5 brackets");
    }

    /// The reported best_objective must equal the minimum across all evaluations.
    #[test]
    fn test_hyperband_returns_best() {
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
            .run(&objective_fn, &bounds, &ConfigSampler::Random, &mut rng)
            .expect("run ok");
        let true_min = result
            .evaluations
            .iter()
            .map(|e| e.objective)
            .fold(f64::INFINITY, f64::min);
        assert!(
            (result.best_objective - true_min).abs() < 1e-12,
            "best_objective ({}) must equal min evaluation ({})",
            result.best_objective,
            true_min
        );
    }

    /// Successive halving must reduce the number of configs each round.
    #[test]
    fn test_successive_halving_reduces() {
        let cfg = MultiFidelityConfig {
            max_budget: 27.0,
            min_budget: 1.0,
            eta: 3,
            n_initial: 27,
        };
        let sh = SuccessiveHalving::new(cfg).expect("valid");
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

    /// Total budget consumed must be finite and positive.
    #[test]
    fn test_hyperband_budget_respected() {
        let cfg = MultiFidelityConfig {
            max_budget: 27.0,
            min_budget: 1.0,
            eta: 3,
            n_initial: 0,
        };
        let hb = Hyperband::new(cfg).expect("valid");
        let mut rng = 7u64;
        let result = hb
            .run(
                &objective_fn,
                &[(-1.0, 1.0)],
                &ConfigSampler::Random,
                &mut rng,
            )
            .expect("run ok");
        assert!(
            result.total_budget_used > 0.0 && result.total_budget_used.is_finite(),
            "total_budget_used must be positive and finite: {}",
            result.total_budget_used
        );
    }

    /// Sampled configurations must have the correct dimensionality.
    #[test]
    fn test_sample_configs_dimensionality() {
        let cfg = MultiFidelityConfig {
            max_budget: 9.0,
            min_budget: 1.0,
            eta: 3,
            n_initial: 0,
        };
        let sh = SuccessiveHalving::new(cfg).expect("valid");
        let bounds = vec![(-1.0, 1.0), (-2.0, 2.0), (0.0, 5.0)];
        let mut rng = 111u64;
        let result = sh
            .run(&objective_fn, &bounds, &ConfigSampler::Random, &mut rng)
            .expect("run ok");
        for e in &result.evaluations {
            assert_eq!(
                e.config.len(),
                3,
                "each config must have 3 params, got {}",
                e.config.len()
            );
        }
    }

    /// promote_top_fraction: keeping top 1/eta should give the correct count.
    #[test]
    fn test_promote_top_fraction() {
        // We verify through SuccessiveHalving's schedule that keep = n/eta.
        let cfg = MultiFidelityConfig {
            max_budget: 27.0,
            min_budget: 1.0,
            eta: 3,
            n_initial: 27,
        };
        let sh = SuccessiveHalving::new(cfg).expect("valid");
        let schedule = sh.compute_schedule();
        // Expect (27, 9, 3, 1)
        assert_eq!(schedule[0].n_configs, 27);
        assert_eq!(schedule[1].n_configs, 9); // 27/3
        assert_eq!(schedule[2].n_configs, 3); //  9/3
    }

    /// Hyperband with a single config should still succeed.
    #[test]
    fn test_hyperband_single_config() {
        let cfg = MultiFidelityConfig {
            max_budget: 9.0,
            min_budget: 1.0,
            eta: 3,
            n_initial: 1,
        };
        let hb = Hyperband::new(cfg).expect("valid");
        let mut rng = 1u64;
        let result = hb
            .run(
                &objective_fn,
                &[(-1.0, 1.0)],
                &ConfigSampler::Random,
                &mut rng,
            )
            .expect("single-config run ok");
        assert!(result.best_objective.is_finite());
    }
}
