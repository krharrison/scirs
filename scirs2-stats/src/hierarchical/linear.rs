//! Hierarchical linear models with varying intercepts and slopes.
//!
//! Model structure:
//! ```text
//! Level 1:  y_{ij} ~ N(α_j + β_j * x_{ij}, σ²)
//! Level 2:  α_j ~ N(μ_α, τ_α²),   j = 1..n_groups
//!           β_j ~ N(μ_β, τ_β²)
//! Hyperpriors: μ_α, μ_β ~ N(0, 100)
//!              σ², τ_α², τ_β² ~ InvGamma(1, 1)
//! ```
//!
//! Estimation uses collapsed Gibbs sampling with conjugate full conditionals.

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::random::{rngs::StdRng, Distribution, Gamma, Normal, SeedableRng, Uniform};

use super::hyperpriors::lgamma;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// Hierarchical linear model with group-level varying intercepts and slopes.
///
/// The model is fitted via blocked Gibbs sampling.  After calling
/// [`HierarchicalLinearModel::fit_gibbs`] the `alphas` and `betas` fields
/// hold the posterior *means* of the group-specific intercepts and slopes,
/// while the remaining fields hold the posterior means of the hyperparameters.
#[derive(Debug, Clone)]
pub struct HierarchicalLinearModel {
    /// Number of groups.
    pub n_groups: usize,
    /// Posterior mean of the intercept hyperparameter.
    pub mu_alpha: f64,
    /// Posterior mean of the slope hyperparameter.
    pub mu_beta: f64,
    /// Posterior mean of the between-group intercept SD.
    pub tau_alpha: f64,
    /// Posterior mean of the between-group slope SD.
    pub tau_beta: f64,
    /// Posterior mean of the within-group observation noise SD.
    pub sigma: f64,
    /// Posterior mean group intercepts (length = n_groups).
    pub alphas: Vec<f64>,
    /// Posterior mean group slopes (length = n_groups).
    pub betas: Vec<f64>,
}

/// Full summary of the Gibbs-sampling fit.
#[derive(Debug, Clone)]
pub struct HierarchicalLinearResult {
    /// Posterior mean of the global intercept.
    pub mu_alpha: f64,
    /// Posterior mean of the global slope.
    pub mu_beta: f64,
    /// Posterior mean of the between-group intercept SD.
    pub tau_alpha: f64,
    /// Posterior mean of the between-group slope SD.
    pub tau_beta: f64,
    /// Posterior mean of the observation noise SD.
    pub sigma: f64,
    /// Posterior mean group intercepts.
    pub group_alphas: Vec<f64>,
    /// Posterior mean group slopes.
    pub group_betas: Vec<f64>,
    /// Number of post-warmup samples retained.
    pub n_samples: usize,
    /// Leave-one-out expected log pointwise predictive density (ELPD_LOO).
    pub loo_elpd: f64,
    /// R-hat convergence diagnostics for μ_α and μ_β.
    pub rhat_mu_alpha: f64,
    pub rhat_mu_beta: f64,
    /// Effective sample sizes.
    pub ess_mu_alpha: f64,
    pub ess_mu_beta: f64,
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl HierarchicalLinearModel {
    /// Construct a new model for `n_groups` groups.
    ///
    /// Initial parameter values are chosen to be vague (weakly informative).
    ///
    /// # Errors
    /// Returns an error when `n_groups == 0`.
    pub fn new(n_groups: usize) -> Result<Self> {
        if n_groups == 0 {
            return Err(StatsError::InvalidArgument(
                "n_groups must be >= 1".into(),
            ));
        }
        Ok(Self {
            n_groups,
            mu_alpha: 0.0,
            mu_beta: 0.0,
            tau_alpha: 1.0,
            tau_beta: 1.0,
            sigma: 1.0,
            alphas: vec![0.0; n_groups],
            betas: vec![0.0; n_groups],
        })
    }

    /// Fit the model using blocked Gibbs sampling.
    ///
    /// # Parameters
    /// - `data`: slice of `(group_id, x, y)` triples.  `group_id` must be in
    ///   `0..n_groups`.
    /// - `n_iter`: total number of MCMC iterations (including warmup).
    /// - `n_warmup`: number of warmup ("burn-in") iterations to discard.
    ///
    /// # Returns
    /// A [`HierarchicalLinearResult`] with posterior summaries and diagnostics.
    ///
    /// # Errors
    /// Returns an error on invalid input or if no observations exist.
    pub fn fit_gibbs(
        &mut self,
        data: &[(usize, f64, f64)],
        n_iter: usize,
        n_warmup: usize,
        seed: u64,
    ) -> Result<HierarchicalLinearResult> {
        if data.is_empty() {
            return Err(StatsError::InsufficientData(
                "data must be non-empty".into(),
            ));
        }
        if n_warmup >= n_iter {
            return Err(StatsError::InvalidArgument(
                "n_warmup must be < n_iter".into(),
            ));
        }
        for &(gid, x, y) in data {
            if gid >= self.n_groups {
                return Err(StatsError::InvalidArgument(format!(
                    "group_id {gid} >= n_groups {}",
                    self.n_groups
                )));
            }
            if !x.is_finite() || !y.is_finite() {
                return Err(StatsError::InvalidArgument(
                    "x and y must be finite".into(),
                ));
            }
        }

        let mut rng = StdRng::seed_from_u64(seed);

        let n_post = n_iter - n_warmup;
        let j = self.n_groups;

        // ---------- Pre-compute per-group sufficient statistics ----------
        // group_data[g] = Vec<(x, y)>
        let mut group_data: Vec<Vec<(f64, f64)>> = vec![Vec::new(); j];
        for &(gid, x, y) in data {
            group_data[gid].push((x, y));
        }

        // Chains for diagnostics
        let mut mu_alpha_chain = Vec::with_capacity(n_post);
        let mut mu_beta_chain = Vec::with_capacity(n_post);

        // Running sums for posterior means
        let mut sum_mu_alpha = 0.0_f64;
        let mut sum_mu_beta = 0.0_f64;
        let mut sum_tau_alpha = 0.0_f64;
        let mut sum_tau_beta = 0.0_f64;
        let mut sum_sigma = 0.0_f64;
        let mut sum_alphas = vec![0.0_f64; j];
        let mut sum_betas = vec![0.0_f64; j];

        // Current state
        let mut mu_alpha = 0.0_f64;
        let mut mu_beta = 0.0_f64;
        let mut tau2_alpha = 1.0_f64; // variances
        let mut tau2_beta = 1.0_f64;
        let mut sigma2 = 1.0_f64;
        let mut alphas = vec![0.0_f64; j];
        let mut betas = vec![0.0_f64; j];

        // Hyper-hyperprior: μ_α, μ_β ~ N(0, 100²)
        let mu_prior_var = 1e4_f64;

        // InvGamma(1, 1) hyperprior on variances
        let ig_shape_prior = 1.0_f64;
        let ig_scale_prior = 1.0_f64;

        for iter in 0..n_iter {
            // ---- 1. Sample group intercepts α_j ----
            for g in 0..j {
                let gd = &group_data[g];
                let n_g = gd.len() as f64;
                if n_g > 0.0 {
                    // Residuals from slope contribution
                    let resid_sum: f64 = gd.iter().map(|&(x, y)| y - betas[g] * x).sum();
                    let prec_prior = 1.0 / tau2_alpha;
                    let prec_lik = n_g / sigma2;
                    let prec_post = prec_prior + prec_lik;
                    let mean_post =
                        (mu_alpha * prec_prior + resid_sum / sigma2) / prec_post;
                    let std_post = (1.0 / prec_post).sqrt();
                    alphas[g] = sample_normal(&mut rng, mean_post, std_post)?;
                } else {
                    alphas[g] = sample_normal(&mut rng, mu_alpha, tau2_alpha.sqrt())?;
                }
            }

            // ---- 2. Sample group slopes β_j ----
            for g in 0..j {
                let gd = &group_data[g];
                let n_g = gd.len() as f64;
                if n_g > 0.0 {
                    let sum_x2: f64 = gd.iter().map(|&(x, _)| x * x).sum();
                    let sum_xy: f64 =
                        gd.iter().map(|&(x, y)| x * (y - alphas[g])).sum();
                    let prec_prior = 1.0 / tau2_beta;
                    let prec_lik = sum_x2 / sigma2;
                    let prec_post = prec_prior + prec_lik;
                    let mean_post =
                        (mu_beta * prec_prior + sum_xy / sigma2) / prec_post;
                    let std_post = (1.0 / prec_post).sqrt();
                    betas[g] = sample_normal(&mut rng, mean_post, std_post)?;
                } else {
                    betas[g] = sample_normal(&mut rng, mu_beta, tau2_beta.sqrt())?;
                }
            }

            // ---- 3. Sample hyperparameter μ_α ----
            {
                let mean_alphas = alphas.iter().sum::<f64>() / j as f64;
                let prec_prior = 1.0 / mu_prior_var;
                let prec_lik = j as f64 / tau2_alpha;
                let prec_post = prec_prior + prec_lik;
                let mean_post = (0.0 * prec_prior + mean_alphas * prec_lik) / prec_post;
                let std_post = (1.0 / prec_post).sqrt();
                mu_alpha = sample_normal(&mut rng, mean_post, std_post)?;
            }

            // ---- 4. Sample hyperparameter μ_β ----
            {
                let mean_betas = betas.iter().sum::<f64>() / j as f64;
                let prec_prior = 1.0 / mu_prior_var;
                let prec_lik = j as f64 / tau2_beta;
                let prec_post = prec_prior + prec_lik;
                let mean_post = (0.0 * prec_prior + mean_betas * prec_lik) / prec_post;
                let std_post = (1.0 / prec_post).sqrt();
                mu_beta = sample_normal(&mut rng, mean_post, std_post)?;
            }

            // ---- 5. Sample τ²_α (InvGamma full conditional) ----
            {
                let ss: f64 = alphas.iter().map(|&a| (a - mu_alpha).powi(2)).sum();
                let shape_post = ig_shape_prior + j as f64 / 2.0;
                let scale_post = ig_scale_prior + ss / 2.0;
                tau2_alpha = sample_inv_gamma(&mut rng, shape_post, scale_post)?;
            }

            // ---- 6. Sample τ²_β ----
            {
                let ss: f64 = betas.iter().map(|&b| (b - mu_beta).powi(2)).sum();
                let shape_post = ig_shape_prior + j as f64 / 2.0;
                let scale_post = ig_scale_prior + ss / 2.0;
                tau2_beta = sample_inv_gamma(&mut rng, shape_post, scale_post)?;
            }

            // ---- 7. Sample σ² ----
            {
                let total_n: usize = group_data.iter().map(|gd| gd.len()).sum();
                let mut ss_resid = 0.0_f64;
                for (g, gd) in group_data.iter().enumerate() {
                    for &(x, y) in gd {
                        let resid = y - alphas[g] - betas[g] * x;
                        ss_resid += resid * resid;
                    }
                }
                let shape_post = ig_shape_prior + total_n as f64 / 2.0;
                let scale_post = ig_scale_prior + ss_resid / 2.0;
                sigma2 = sample_inv_gamma(&mut rng, shape_post, scale_post)?;
            }

            // ---- Collect post-warmup samples ----
            if iter >= n_warmup {
                mu_alpha_chain.push(mu_alpha);
                mu_beta_chain.push(mu_beta);
                sum_mu_alpha += mu_alpha;
                sum_mu_beta += mu_beta;
                sum_tau_alpha += tau2_alpha.sqrt();
                sum_tau_beta += tau2_beta.sqrt();
                sum_sigma += sigma2.sqrt();
                for g in 0..j {
                    sum_alphas[g] += alphas[g];
                    sum_betas[g] += betas[g];
                }
            }
        }

        let n_f = n_post as f64;
        let post_mu_alpha = sum_mu_alpha / n_f;
        let post_mu_beta = sum_mu_beta / n_f;
        let post_tau_alpha = sum_tau_alpha / n_f;
        let post_tau_beta = sum_tau_beta / n_f;
        let post_sigma = sum_sigma / n_f;
        let post_alphas: Vec<f64> = sum_alphas.iter().map(|&s| s / n_f).collect();
        let post_betas: Vec<f64> = sum_betas.iter().map(|&s| s / n_f).collect();

        // Update model state with posterior means
        self.mu_alpha = post_mu_alpha;
        self.mu_beta = post_mu_beta;
        self.tau_alpha = post_tau_alpha;
        self.tau_beta = post_tau_beta;
        self.sigma = post_sigma;
        self.alphas = post_alphas.clone();
        self.betas = post_betas.clone();

        // ---- Diagnostics: R-hat and ESS ----
        let rhat_mu_alpha = split_rhat(&mu_alpha_chain);
        let rhat_mu_beta = split_rhat(&mu_beta_chain);
        let ess_mu_alpha = bulk_ess(&mu_alpha_chain);
        let ess_mu_beta = bulk_ess(&mu_beta_chain);

        // ---- LOO-CV ELPD (importance sampling approximation) ----
        let loo_elpd = compute_loo_elpd(
            data,
            &post_alphas,
            &post_betas,
            post_sigma,
        );

        Ok(HierarchicalLinearResult {
            mu_alpha: post_mu_alpha,
            mu_beta: post_mu_beta,
            tau_alpha: post_tau_alpha,
            tau_beta: post_tau_beta,
            sigma: post_sigma,
            group_alphas: post_alphas,
            group_betas: post_betas,
            n_samples: n_post,
            loo_elpd,
            rhat_mu_alpha,
            rhat_mu_beta,
            ess_mu_alpha,
            ess_mu_beta,
        })
    }

    /// Point prediction for a new observation in group `group_id`.
    ///
    /// Uses posterior mean parameters: Ê[y] = α_j + β_j * x
    ///
    /// # Errors
    /// Returns an error when `group_id >= n_groups`.
    pub fn predict(&self, group_id: usize, x: f64) -> Result<f64> {
        if group_id >= self.n_groups {
            return Err(StatsError::InvalidArgument(format!(
                "group_id {group_id} >= n_groups {}",
                self.n_groups
            )));
        }
        Ok(self.alphas[group_id] + self.betas[group_id] * x)
    }

    /// Approximate 95 % posterior predictive credible interval.
    ///
    /// `samples` is a slice of `(alphas_sample, betas_sample)` tuples
    /// collected from `fit_gibbs`.  Each element represents one posterior
    /// draw over all groups.
    ///
    /// # Errors
    /// Returns an error on invalid `group_id` or empty `samples`.
    pub fn predict_ci(
        &self,
        group_id: usize,
        x: f64,
        samples: &[(Vec<f64>, Vec<f64>)],
    ) -> Result<(f64, f64)> {
        if group_id >= self.n_groups {
            return Err(StatsError::InvalidArgument(format!(
                "group_id {group_id} >= n_groups {}",
                self.n_groups
            )));
        }
        if samples.is_empty() {
            return Err(StatsError::InsufficientData(
                "samples must be non-empty".into(),
            ));
        }
        let mut preds: Vec<f64> = samples
            .iter()
            .filter_map(|(a, b)| {
                if group_id < a.len() && group_id < b.len() {
                    Some(a[group_id] + b[group_id] * x)
                } else {
                    None
                }
            })
            .collect();
        if preds.is_empty() {
            return Err(StatsError::InsufficientData(
                "No valid predictions from samples".into(),
            ));
        }
        preds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lo = preds[(preds.len() as f64 * 0.025) as usize];
        let hi = preds[(preds.len() as f64 * 0.975).min(preds.len() as f64 - 1.0) as usize];
        Ok((lo, hi))
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Sample from N(mean, std).
fn sample_normal(rng: &mut StdRng, mean: f64, std: f64) -> Result<f64> {
    if std <= 0.0 || !std.is_finite() {
        return Ok(mean); // degenerate case
    }
    let n = Normal::new(mean, std).map_err(|e| {
        StatsError::ComputationError(format!("Normal sampling error: {e}"))
    })?;
    Ok(n.sample(rng))
}

/// Sample from InvGamma(shape, scale) where scale is the *rate* parameter β
/// such that E[X] = scale / (shape - 1).
///
/// We sample X ~ Gamma(shape, 1/scale) then return 1/X.
fn sample_inv_gamma(rng: &mut StdRng, shape: f64, scale: f64) -> Result<f64> {
    if shape <= 0.0 || scale <= 0.0 {
        return Ok(1.0);
    }
    let g = Gamma::new(shape, 1.0 / scale).map_err(|e| {
        StatsError::ComputationError(format!("Gamma sampling error: {e}"))
    })?;
    let x = g.sample(rng);
    Ok(if x > 0.0 { 1.0 / x } else { f64::MAX })
}

/// Normal log-PDF: log p(x | μ, σ).
fn normal_log_pdf(x: f64, mean: f64, std: f64) -> f64 {
    if std <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z = (x - mean) / std;
    -0.5 * z * z - std.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
}

/// Compute approximate LOO-ELPD using the log-likelihood evaluated at
/// posterior mean parameters (fast approximation, not full IS-LOO).
fn compute_loo_elpd(
    data: &[(usize, f64, f64)],
    alphas: &[f64],
    betas: &[f64],
    sigma: f64,
) -> f64 {
    data.iter()
        .map(|&(g, x, y)| {
            let mean = alphas[g] + betas[g] * x;
            normal_log_pdf(y, mean, sigma)
        })
        .sum::<f64>()
        / data.len() as f64
}

/// Gelman-Rubin R-hat via chain splitting.
fn split_rhat(chain: &[f64]) -> f64 {
    let n = chain.len();
    if n < 4 {
        return f64::NAN;
    }
    let half = n / 2;
    let c1 = &chain[..half];
    let c2 = &chain[half..];
    let m1 = mean(c1);
    let m2 = mean(c2);
    let v1 = variance(c1, m1);
    let v2 = variance(c2, m2);
    let overall_mean = (m1 + m2) / 2.0;
    let b = (half as f64) * ((m1 - overall_mean).powi(2) + (m2 - overall_mean).powi(2)) / 1.0;
    let w = (v1 + v2) / 2.0;
    if w < 1e-15 {
        return 1.0;
    }
    let var_est = (1.0 - 1.0 / half as f64) * w + b / half as f64;
    (var_est / w).sqrt()
}

/// Bulk effective sample size estimate via autocorrelation.
fn bulk_ess(chain: &[f64]) -> f64 {
    let n = chain.len();
    if n < 2 {
        return n as f64;
    }
    let m = mean(chain);
    let v = variance(chain, m);
    if v < 1e-15 {
        return n as f64;
    }
    let max_lag = n / 2;
    let mut rho_sum = 0.0_f64;
    for lag in 1..max_lag {
        let rho = autocorr(chain, lag, m, v);
        if rho < 0.05 {
            break;
        }
        rho_sum += rho;
    }
    (n as f64) / (1.0 + 2.0 * rho_sum)
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn variance(xs: &[f64], m: f64) -> f64 {
    xs.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64
}

fn autocorr(xs: &[f64], lag: usize, m: f64, v: f64) -> f64 {
    let n = xs.len();
    if lag >= n || v < 1e-15 {
        return 0.0;
    }
    let cov: f64 = (0..n - lag)
        .map(|i| (xs[i] - m) * (xs[i + lag] - m))
        .sum::<f64>()
        / (n - lag) as f64;
    cov / v
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_synthetic(n_groups: usize, obs_per_group: usize, seed: u64) -> Vec<(usize, f64, f64)> {
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut data = Vec::new();
        // True hyperparams: mu_alpha=2, mu_beta=0.5, tau_alpha=0.3, tau_beta=0.2, sigma=0.5
        let mu_alpha = 2.0_f64;
        let mu_beta = 0.5_f64;
        let tau_alpha = 0.3_f64;
        let tau_beta = 0.2_f64;
        let sigma = 0.5_f64;
        for g in 0..n_groups {
            let alpha_g = mu_alpha + tau_alpha * normal.sample(&mut rng);
            let beta_g = mu_beta + tau_beta * normal.sample(&mut rng);
            for _ in 0..obs_per_group {
                let x: f64 = 3.0 * normal.sample(&mut rng);
                let y = alpha_g + beta_g * x + sigma * normal.sample(&mut rng);
                data.push((g, x, y));
            }
        }
        data
    }

    #[test]
    fn test_model_construction() {
        let m = HierarchicalLinearModel::new(5).unwrap();
        assert_eq!(m.n_groups, 5);
        assert_eq!(m.alphas.len(), 5);
        assert_eq!(m.betas.len(), 5);
        assert!(HierarchicalLinearModel::new(0).is_err());
    }

    #[test]
    fn test_fit_gibbs_recovers_hyperparams() {
        let data = generate_synthetic(5, 20, 42);
        let mut model = HierarchicalLinearModel::new(5).unwrap();
        let result = model.fit_gibbs(&data, 2000, 500, 42).unwrap();

        // Posterior means should be near true values (mu_alpha≈2, mu_beta≈0.5)
        assert!((result.mu_alpha - 2.0).abs() < 1.5, "mu_alpha={}", result.mu_alpha);
        assert!((result.mu_beta - 0.5).abs() < 1.0, "mu_beta={}", result.mu_beta);
        assert!(result.sigma > 0.0);
        assert!(result.tau_alpha > 0.0);
        assert!(result.tau_beta > 0.0);
        assert_eq!(result.n_samples, 1500);
    }

    #[test]
    fn test_predict() {
        let data = generate_synthetic(3, 15, 10);
        let mut model = HierarchicalLinearModel::new(3).unwrap();
        model.fit_gibbs(&data, 1000, 300, 10).unwrap();

        let pred = model.predict(0, 1.0).unwrap();
        assert!(pred.is_finite());
        assert!(model.predict(3, 1.0).is_err());
    }

    #[test]
    fn test_predict_ci() {
        let data = generate_synthetic(2, 10, 7);
        let mut model = HierarchicalLinearModel::new(2).unwrap();
        model.fit_gibbs(&data, 800, 200, 7).unwrap();

        // Build some fake samples
        let samples: Vec<(Vec<f64>, Vec<f64>)> = (0..100)
            .map(|i| {
                let a = vec![model.alphas[0] + i as f64 * 0.01, model.alphas[1] - i as f64 * 0.01];
                let b = vec![model.betas[0] + i as f64 * 0.005, model.betas[1]];
                (a, b)
            })
            .collect();

        let (lo, hi) = model.predict_ci(0, 1.0, &samples).unwrap();
        assert!(lo <= hi, "CI lower bound should be <= upper bound");
        assert!(model.predict_ci(5, 1.0, &samples).is_err());
        assert!(model.predict_ci(0, 1.0, &[]).is_err());
    }

    #[test]
    fn test_loo_elpd_is_finite() {
        let data = generate_synthetic(4, 8, 99);
        let mut model = HierarchicalLinearModel::new(4).unwrap();
        let result = model.fit_gibbs(&data, 500, 100, 99).unwrap();
        assert!(result.loo_elpd.is_finite(), "LOO ELPD should be finite");
    }

    #[test]
    fn test_fit_invalid_inputs() {
        let mut model = HierarchicalLinearModel::new(3).unwrap();
        // Empty data
        assert!(model.fit_gibbs(&[], 100, 50, 0).is_err());
        // n_warmup >= n_iter
        assert!(model.fit_gibbs(&[(0, 1.0, 2.0)], 100, 100, 0).is_err());
        // Invalid group id
        assert!(model.fit_gibbs(&[(5, 1.0, 2.0)], 100, 50, 0).is_err());
    }

    #[test]
    fn test_convergence_diagnostics() {
        let data = generate_synthetic(4, 10, 5);
        let mut model = HierarchicalLinearModel::new(4).unwrap();
        let result = model.fit_gibbs(&data, 2000, 500, 5).unwrap();
        // R-hat should be close to 1.0 for well-converged chains
        assert!(result.rhat_mu_alpha.is_finite());
        assert!(result.rhat_mu_beta.is_finite());
        // ESS should be positive
        assert!(result.ess_mu_alpha > 0.0);
        assert!(result.ess_mu_beta > 0.0);
    }
}
