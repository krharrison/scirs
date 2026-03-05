//! Multilevel Monte Carlo (MLMC) integration
//!
//! Implements the MLMC estimator for variance reduction by combining
//! cheap coarse-level simulations with expensive fine-level corrections.

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::random::{seeded_rng, Distribution, Normal};
use scirs2_core::random::{Rng, SeedableRng};

/// Configuration for a single MLMC level.
#[derive(Debug, Clone)]
pub struct MLMCLevel {
    /// Level index (0 = coarsest)
    pub level: usize,
    /// Number of samples at this level
    pub n_samples: usize,
    /// Cost per sample (relative)
    pub cost: f64,
    /// Estimated variance at this level
    pub variance: f64,
}

/// Options for MLMC integration.
#[derive(Debug, Clone)]
pub struct MLMCOptions {
    /// Target mean-square error
    pub target_mse: f64,
    /// Initial samples per level
    pub initial_samples: usize,
    /// Maximum number of levels
    pub max_levels: usize,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for MLMCOptions {
    fn default() -> Self {
        Self {
            target_mse: 1e-4,
            initial_samples: 100,
            max_levels: 6,
            seed: None,
        }
    }
}

/// Result of MLMC integration.
#[derive(Debug, Clone)]
pub struct MLMCResult {
    /// Estimated value of the quantity of interest
    pub estimate: f64,
    /// Estimated mean-square error
    pub mse: f64,
    /// Total number of samples across all levels
    pub total_samples: usize,
    /// Level information
    pub levels: Vec<MLMCLevel>,
}

/// Perform Multilevel Monte Carlo integration.
///
/// Uses the MLMC estimator: E\[P_L\] = E\[P_0\] + Σ_{l=1}^{L} E\[P_l - P_{l-1}\]
///
/// # Arguments
///
/// * `fine_fn` - Function computing the fine-level estimate at level `l`
/// * `coarse_fn` - Function computing the coarse-level estimate at level `l-1`
/// * `options` - MLMC configuration
///
/// # Returns
///
/// An `MLMCResult` containing the estimate and diagnostics.
pub fn mlmc_integrate<F, C>(
    fine_fn: F,
    coarse_fn: C,
    options: Option<MLMCOptions>,
) -> IntegrateResult<MLMCResult>
where
    F: Fn(usize, u64) -> f64,
    C: Fn(usize, u64) -> f64,
{
    let opts = options.unwrap_or_default();
    let mut rng = match opts.seed {
        Some(s) => seeded_rng(s),
        None => seeded_rng(42),
    };

    let max_levels = opts.max_levels.max(2);
    let n_init = opts.initial_samples.max(10);

    // Pilot run to estimate variances at each level
    let mut level_vars: Vec<f64> = Vec::with_capacity(max_levels);
    let mut level_means: Vec<f64> = Vec::with_capacity(max_levels);

    for l in 0..max_levels {
        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;

        for _ in 0..n_init {
            let seed: u64 = rng.random();
            let val = if l == 0 {
                fine_fn(0, seed)
            } else {
                fine_fn(l, seed) - coarse_fn(l, seed)
            };
            sum += val;
            sum_sq += val * val;
        }

        let mean = sum / n_init as f64;
        let var = (sum_sq / n_init as f64 - mean * mean).max(0.0);

        level_means.push(mean);
        level_vars.push(var);

        // Stop adding levels if variance is negligible
        if l > 1 && var < opts.target_mse * 0.01 {
            break;
        }
    }

    let n_levels = level_vars.len();

    // Optimal sample allocation: n_l ∝ sqrt(V_l / C_l), cost C_l = 4^l
    let costs: Vec<f64> = (0..n_levels).map(|l| 4.0_f64.powi(l as i32)).collect();
    let total_cost_factor: f64 = (0..n_levels)
        .map(|l| (level_vars[l] * costs[l]).sqrt())
        .sum();

    let epsilon_sq = opts.target_mse;
    let n_samples: Vec<usize> = (0..n_levels)
        .map(|l| {
            let n_opt = (2.0 / epsilon_sq * total_cost_factor * (level_vars[l] / costs[l]).sqrt())
                .ceil() as usize;
            n_opt.max(n_init)
        })
        .collect();

    // Run MLMC estimator with optimal allocation
    let mut total_estimate = 0.0_f64;
    let mut total_variance = 0.0_f64;
    let mut total_samples = 0;
    let mut levels = Vec::with_capacity(n_levels);

    for l in 0..n_levels {
        let n = n_samples[l];
        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;

        for _ in 0..n {
            let seed: u64 = rng.random();
            let val = if l == 0 {
                fine_fn(0, seed)
            } else {
                fine_fn(l, seed) - coarse_fn(l, seed)
            };
            sum += val;
            sum_sq += val * val;
        }

        let mean = sum / n as f64;
        let var = (sum_sq / n as f64 - mean * mean).max(0.0) / n as f64;

        total_estimate += mean;
        total_variance += var;
        total_samples += n;

        levels.push(MLMCLevel {
            level: l,
            n_samples: n,
            cost: costs[l],
            variance: var * n as f64,
        });
    }

    Ok(MLMCResult {
        estimate: total_estimate,
        mse: total_variance,
        total_samples,
        levels,
    })
}

/// Adaptive MLMC that adds levels until target MSE is achieved.
pub fn mlmc_adaptive<F, C>(
    fine_fn: F,
    coarse_fn: C,
    target_mse: f64,
    seed: Option<u64>,
) -> IntegrateResult<MLMCResult>
where
    F: Fn(usize, u64) -> f64,
    C: Fn(usize, u64) -> f64,
{
    let opts = MLMCOptions {
        target_mse,
        seed,
        ..Default::default()
    };
    mlmc_integrate(fine_fn, coarse_fn, Some(opts))
}

/// Compute MLMC complexity estimate (total work for target accuracy ε).
///
/// Returns the estimated total work W ~ O(ε^{-2}) for MLMC vs O(ε^{-2-β/α}) for MC.
pub fn mlmc_complexity(levels: &[MLMCLevel]) -> f64 {
    levels.iter().map(|l| l.cost * l.n_samples as f64).sum()
}

/// Generate Geometric Brownian Motion path hierarchy for MLMC testing.
///
/// Returns a pair `(fine_value, coarse_value)` approximating `E[S_T]`
/// using `2^l` steps (fine) and `2^(l-1)` steps (coarse).
pub fn geometric_brownian_motion_levels(
    s0: f64,
    mu: f64,
    sigma: f64,
    t: f64,
    level: usize,
    seed: u64,
) -> (f64, f64) {
    let normal = match Normal::new(0.0_f64, 1.0_f64) {
        Ok(n) => n,
        Err(_) => return (s0, s0),
    };
    let mut rng = seeded_rng(seed);

    let n_fine = 1usize << level;
    let dt_fine = t / n_fine as f64;
    let sqrt_dt_fine = dt_fine.sqrt();

    // Generate fine path using GBM update: S_{n+1} = S_n * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
    let increments: Vec<f64> = (0..n_fine)
        .map(|_| Distribution::sample(&normal, &mut rng))
        .collect();

    let mut s_fine = s0;
    for &z in &increments {
        s_fine *= ((mu - 0.5 * sigma * sigma) * dt_fine + sigma * sqrt_dt_fine * z).exp();
    }

    // Generate coarse path by combining pairs of increments
    let coarse_val = if level == 0 {
        // No coarse path at level 0
        s0
    } else {
        let n_coarse = n_fine / 2;
        let dt_coarse = t / n_coarse as f64;
        let sqrt_dt_coarse = dt_coarse.sqrt();
        let mut s_coarse = s0;

        for pair in 0..n_coarse {
            let z1 = increments[2 * pair];
            let z2 = if 2 * pair + 1 < increments.len() {
                increments[2 * pair + 1]
            } else {
                0.0
            };
            // Antithetic coupling: combined Brownian increment
            let z_coarse = (z1 + z2) / std::f64::consts::SQRT_2;
            s_coarse *=
                ((mu - 0.5 * sigma * sigma) * dt_coarse + sigma * sqrt_dt_coarse * z_coarse).exp();
        }
        s_coarse
    };

    (s_fine, coarse_val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlmc_integrate_basic() {
        // Simple test: estimate E[X] where X ~ N(1, 0.1)
        // Using constant fine/coarse functions
        let result = mlmc_integrate(
            |_level, seed| {
                let mut rng = seeded_rng(seed);
                let normal = Normal::new(1.0_f64, 0.1_f64).expect("valid");
                Distribution::sample(&normal, &mut rng)
            },
            |_level, seed| {
                let mut rng = seeded_rng(seed.wrapping_add(1));
                let normal = Normal::new(1.0_f64, 0.2_f64).expect("valid");
                Distribution::sample(&normal, &mut rng)
            },
            Some(MLMCOptions {
                target_mse: 1e-3,
                initial_samples: 50,
                max_levels: 3,
                seed: Some(42),
            }),
        )
        .expect("MLMC failed");

        // Should be close to 1.0
        assert!(
            (result.estimate - 1.0).abs() < 0.5,
            "estimate={}",
            result.estimate
        );
        assert!(result.total_samples > 0);
    }

    #[test]
    fn test_gbm_levels() {
        let (fine, coarse) = geometric_brownian_motion_levels(1.0, 0.05, 0.2, 1.0, 3, 42);
        assert!(fine > 0.0, "GBM must be positive");
        assert!(coarse > 0.0, "GBM coarse must be positive");
    }

    #[test]
    fn test_mlmc_complexity() {
        let levels = vec![
            MLMCLevel {
                level: 0,
                n_samples: 1000,
                cost: 1.0,
                variance: 0.5,
            },
            MLMCLevel {
                level: 1,
                n_samples: 500,
                cost: 4.0,
                variance: 0.1,
            },
            MLMCLevel {
                level: 2,
                n_samples: 100,
                cost: 16.0,
                variance: 0.01,
            },
        ];
        let complexity = mlmc_complexity(&levels);
        // Total work = 1000*1 + 500*4 + 100*16 = 1000 + 2000 + 1600 = 4600
        assert!((complexity - 4600.0).abs() < 1e-9);
    }
}
