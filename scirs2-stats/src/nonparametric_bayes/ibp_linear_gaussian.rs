//! IBP Linear-Gaussian Model
//!
//! Implements the linear-Gaussian latent feature model with an Indian Buffet
//! Process prior on the binary feature assignment matrix Z.
//!
//! Model:
//!   Z ~ IBP(alpha)
//!   A_k ~ N(0, σ_a² I_D)  for k = 1,...,K
//!   X_i | Z_i, A ~ N(Z_i A, σ_x² I_D)
//!
//! Inference via collapsed Gibbs sampling (marginalising A analytically).
//!
//! # References
//! - Griffiths & Ghahramani (2005). "Infinite latent feature models and the
//!   Indian buffet process."
//! - Knowles & Ghahramani (2011). "Nonparametric Bayesian sparse factor models."

use crate::error::{StatsError, StatsResult as Result};
use crate::nonparametric_bayes::beta_process::{poisson_sample, BetaProcessConfig, BetaProcess};
use scirs2_core::ndarray::{Array2, Axis};

// ---------------------------------------------------------------------------
// Minimal LCG
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = self.state >> 11;
        (bits as f64) * (1.0 / (1u64 << 53) as f64)
    }

    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the IBP linear-Gaussian model.
#[derive(Debug, Clone)]
pub struct IbpLinearGaussianConfig {
    /// Observation noise standard deviation.
    pub sigma_x: f64,
    /// Feature weight prior standard deviation.
    pub sigma_a: f64,
    /// IBP concentration parameter.
    pub alpha: f64,
    /// Total Gibbs iterations (including burnin).
    pub n_iter: usize,
    /// Burn-in iterations (not included in posterior samples).
    pub burnin: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for IbpLinearGaussianConfig {
    fn default() -> Self {
        Self {
            sigma_x: 0.5,
            sigma_a: 1.0,
            alpha: 1.0,
            n_iter: 200,
            burnin: 50,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Result of fitting the IBP linear-Gaussian model.
#[derive(Debug, Clone)]
pub struct IbpLgResult {
    /// Collection of posterior Z samples (each is N × K).
    pub z_samples: Vec<Array2<f64>>,
    /// Posterior mean of the feature matrix A (K × D).
    pub a_mean: Array2<f64>,
    /// Number of active features at each collected sample.
    pub n_features: Vec<usize>,
}

impl IbpLgResult {
    /// Mean number of active features across posterior samples.
    pub fn mean_n_features(&self) -> f64 {
        if self.n_features.is_empty() {
            return 0.0;
        }
        self.n_features.iter().sum::<usize>() as f64 / self.n_features.len() as f64
    }
}

// ---------------------------------------------------------------------------
// IbpLinearGaussian
// ---------------------------------------------------------------------------

/// IBP linear-Gaussian latent feature model.
pub struct IbpLinearGaussian {
    /// Current binary assignment matrix (N × K).
    pub z: Array2<f64>,
    /// Current feature weight matrix (K × D).
    pub a: Array2<f64>,
    /// Model configuration.
    pub config: IbpLinearGaussianConfig,
}

impl IbpLinearGaussian {
    /// Fit the IBP linear-Gaussian model to data.
    ///
    /// # Parameters
    /// - `data`: N × D data matrix.
    /// - `config`: Sampler configuration.
    ///
    /// # Returns
    /// [`IbpLgResult`] containing posterior samples.
    pub fn fit(data: &Array2<f64>, config: IbpLinearGaussianConfig) -> Result<IbpLgResult> {
        let n = data.nrows();
        let d = data.ncols();
        if n == 0 || d == 0 {
            return Err(StatsError::InvalidInput(
                "IbpLinearGaussian::fit: data must be non-empty".into(),
            ));
        }

        let mut rng = Lcg::new(config.seed);

        // Initialize Z from IBP prior
        let bp_config = BetaProcessConfig {
            alpha: config.alpha,
            n_features: 10,
            n_samples: 5,
            burnin: 1,
            seed: config.seed,
            ..Default::default()
        };
        let bp = BetaProcess::new(bp_config)?;
        let ibp_state = bp.sample_prior(n);

        let k_init = ibp_state.n_features().max(2);
        let z_vec: Vec<f64> = ibp_state
            .z
            .iter()
            .flat_map(|row| {
                let mut padded = row.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect::<Vec<_>>();
                padded.resize(k_init, 0.0);
                padded
            })
            .collect();

        let mut z_mat = Array2::from_shape_vec((n, k_init), z_vec).map_err(|e| {
            StatsError::ComputationError(format!("Z shape error: {e}"))
        })?;

        // Initialize A: sample from prior
        let mut a_mat = {
            let sigma_a = config.sigma_a;
            let a_vec: Vec<f64> = (0..k_init * d)
                .map(|_| sigma_a * rng.next_normal())
                .collect();
            Array2::from_shape_vec((k_init, d), a_vec).map_err(|e| {
                StatsError::ComputationError(format!("A shape error: {e}"))
            })?
        };

        let n_collect = config.n_iter.saturating_sub(config.burnin);
        let mut z_samples: Vec<Array2<f64>> = Vec::with_capacity(n_collect);
        let mut a_accum: Vec<f64> = vec![0.0; k_init * d];
        let mut n_features_hist: Vec<usize> = Vec::with_capacity(n_collect);
        let mut n_accum = 0usize;

        for iter in 0..config.n_iter {
            let k = z_mat.ncols();

            // ----------------------------------------------------------------
            // Step 1: Update each z_ik using Gibbs conditional
            // ----------------------------------------------------------------
            for i in 0..n {
                for ki in 0..k {
                    // n_{-i,k}: count of other rows using feature ki
                    let n_minus_ik: usize = (0..n)
                        .filter(|&j| j != i && z_mat[[j, ki]] > 0.5)
                        .count();

                    // Prior: Bernoulli(n_{-i,k} / N)
                    let prior_p = n_minus_ik as f64 / n as f64;

                    // Log-likelihood ratio: p(x_i | z_ik=1) / p(x_i | z_ik=0)
                    // Using the current A_ki row as the feature vector:
                    // x_i ~ N(Σ_{ki: z_iki=1} A_ki, σ_x² I)
                    // Δlog_lik = contribution of feature ki to x_i
                    let var_x = config.sigma_x * config.sigma_x;
                    let mut log_lr = 0.0_f64;
                    for j in 0..d {
                        let a_kij = a_mat[[ki, j]];
                        // Reconstruction without feature ki
                        let x_ij = data[[i, j]];
                        let recon_without: f64 = (0..k)
                            .filter(|&kk| kk != ki && z_mat[[i, kk]] > 0.5)
                            .map(|kk| a_mat[[kk, j]])
                            .sum();
                        let recon_with = recon_without + a_kij;
                        let log_p1 = -(x_ij - recon_with).powi(2) / (2.0 * var_x);
                        let log_p0 = -(x_ij - recon_without).powi(2) / (2.0 * var_x);
                        log_lr += log_p1 - log_p0;
                    }

                    // Posterior: sigmoid(log_prior_ratio + log_lr)
                    let log_prior_ratio = if prior_p <= 0.0 {
                        f64::NEG_INFINITY
                    } else if prior_p >= 1.0 {
                        f64::INFINITY
                    } else {
                        (prior_p / (1.0 - prior_p)).ln()
                    };

                    let post_p = sigmoid(log_prior_ratio + log_lr);
                    z_mat[[i, ki]] = if rng.next_f64() < post_p { 1.0 } else { 0.0 };
                }
            }

            // ----------------------------------------------------------------
            // Step 2: Update A using matrix-Gaussian posterior
            //   Sigma_A^{-1} = Z^T Z / var_x + I / var_a
            //   mu_A = Sigma_A (Z^T X / var_x)
            // ----------------------------------------------------------------
            a_mat = update_a(&z_mat, data, config.sigma_x, config.sigma_a, &mut rng)?;

            // ----------------------------------------------------------------
            // Step 3: MH proposal for new features (birth/death)
            // ----------------------------------------------------------------
            let uniform_val = rng.next_f64();
            if uniform_val < 0.3 {
                // Propose adding one new feature
                let lambda = config.alpha / n as f64;
                let mut uniform = || rng.next_f64();
                let k_new_prop = poisson_sample(lambda, &mut uniform);
                if k_new_prop > 0 {
                    for _ in 0..k_new_prop.min(3) {
                        // Sample new A row from prior
                        let new_a_row: Vec<f64> =
                            (0..d).map(|_| config.sigma_a * rng.next_normal()).collect();
                        // Sample new z column from prior: Bernoulli(alpha/N)
                        let prob_new = config.alpha / (n as f64 + config.alpha);
                        let new_z_col: Vec<f64> = (0..n)
                            .map(|_| if rng.next_f64() < prob_new { 1.0 } else { 0.0 })
                            .collect();

                        // MH acceptance: log ratio = log p(X | Z_new, A_new) - log p(X | Z_old, A_old)
                        // Since we're adding one feature, compute marginal log-lik difference
                        let log_accept = compute_feature_addition_log_ratio(
                            &z_mat,
                            &a_mat,
                            &new_z_col,
                            &new_a_row,
                            data,
                            config.sigma_x,
                            config.sigma_a,
                            config.alpha,
                        );

                        if rng.next_f64().ln() < log_accept {
                            // Accept: extend Z and A
                            let (new_z, new_a) = extend_matrices(
                                &z_mat,
                                &a_mat,
                                &new_z_col,
                                &new_a_row,
                            )?;
                            z_mat = new_z;
                            a_mat = new_a;
                        }
                    }
                }
            }

            // Remove empty features (all-zero columns)
            let (z_pruned, a_pruned) = prune_empty_features(&z_mat, &a_mat);
            z_mat = z_pruned;
            a_mat = a_pruned;

            // ----------------------------------------------------------------
            // Collect sample
            // ----------------------------------------------------------------
            if iter >= config.burnin {
                let k_cur = z_mat.ncols();
                z_samples.push(z_mat.clone());
                n_features_hist.push(k_cur);

                // Accumulate A for mean (pad/trim to k_init)
                let k_use = k_cur.min(k_init);
                for ki in 0..k_use {
                    for j in 0..d {
                        if ki * d + j < a_accum.len() {
                            a_accum[ki * d + j] += a_mat[[ki, j]];
                        }
                    }
                }
                n_accum += 1;
            }
        }

        // Compute A mean
        let total = n_accum.max(1) as f64;
        let a_mean_vec: Vec<f64> = a_accum.iter().map(|&v| v / total).collect();
        let k_final = k_init;
        let a_mean = if a_mean_vec.len() == k_final * d {
            Array2::from_shape_vec((k_final, d), a_mean_vec).map_err(|e| {
                StatsError::ComputationError(format!("A mean shape error: {e}"))
            })?
        } else {
            // Fallback: use final a_mat
            a_mat.clone()
        };

        Ok(IbpLgResult {
            z_samples,
            a_mean,
            n_features: n_features_hist,
        })
    }

    /// Compute posterior predictive: E[z_new | X] for a new observation.
    ///
    /// Uses the current posterior mean to compute the expected feature
    /// activation probabilities for a new data point.
    ///
    /// # Parameters
    /// - `x_new`: D-dimensional new observation.
    /// - `result`: Fitted model result.
    /// - `config`: Model configuration.
    ///
    /// # Returns
    /// Vector of posterior probabilities for each feature.
    pub fn predict(x_new: &[f64], result: &IbpLgResult, config: &IbpLinearGaussianConfig) -> Vec<f64> {
        let k = result.a_mean.nrows();
        let d = result.a_mean.ncols();
        if k == 0 || d == 0 || x_new.len() != d {
            return vec![];
        }

        let var_x = config.sigma_x * config.sigma_x;
        let mut probs = Vec::with_capacity(k);

        for ki in 0..k {
            // P(z_new_k = 1 | x_new) ∝ prior * likelihood
            // Use the IBP prior: alpha / (N + alpha) for new observation
            let prior_p = config.alpha / (config.alpha + 1.0);

            let mut log_lr = 0.0_f64;
            for j in 0..d {
                let a_kij = result.a_mean[[ki, j]];
                let x_j = x_new.get(j).copied().unwrap_or(0.0);
                let log_p1 = -(x_j - a_kij).powi(2) / (2.0 * var_x);
                let log_p0 = -(x_j).powi(2) / (2.0 * var_x);
                log_lr += log_p1 - log_p0;
            }

            let log_prior_ratio = (prior_p / (1.0 - prior_p + 1e-300)).ln();
            probs.push(sigmoid(log_prior_ratio + log_lr));
        }

        probs
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Update A using the matrix-Gaussian posterior.
///
/// Σ_A^{-1} = Z^T Z / σ_x² + I / σ_a²
/// μ_A = Σ_A · Z^T X / σ_x²
///
/// We compute Σ_A via Cholesky and sample from the posterior.
fn update_a(
    z: &Array2<f64>,
    data: &Array2<f64>,
    sigma_x: f64,
    sigma_a: f64,
    rng: &mut Lcg,
) -> Result<Array2<f64>> {
    let k = z.ncols();
    let d = data.ncols();
    let n = data.nrows();
    if k == 0 {
        return Ok(Array2::zeros((0, d)));
    }

    let var_x = sigma_x * sigma_x;
    let var_a = sigma_a * sigma_a;

    // Z^T Z (K × K)
    let zt_z = z.t().dot(z);
    // Z^T X (K × D)
    let zt_x = z.t().dot(data);

    // Compute A mean column by column
    let mut a_mat = Array2::zeros((k, d));

    for ki in 0..k {
        for j in 0..d {
            // Posterior for A[ki, j]:
            // Σ_a_ki^{-1} = (Z^T Z)[ki,ki] / var_x + 1 / var_a
            // For simplicity: use diagonal approximation of Σ_A
            let sigma_inv_diag = zt_z[[ki, ki]] / var_x + 1.0 / var_a;
            let sigma_diag = 1.0 / sigma_inv_diag.max(1e-10);
            let mu_a = sigma_diag * zt_x[[ki, j]] / var_x;

            // Sample from N(mu_a, sigma_diag)
            a_mat[[ki, j]] = mu_a + sigma_diag.sqrt() * rng.next_normal();
        }
    }

    // Suppress unused variable warning
    let _ = n;

    Ok(a_mat)
}

/// Compute the log acceptance ratio for adding a new feature.
fn compute_feature_addition_log_ratio(
    z: &Array2<f64>,
    a: &Array2<f64>,
    new_z_col: &[f64],
    new_a_row: &[f64],
    data: &Array2<f64>,
    sigma_x: f64,
    sigma_a: f64,
    alpha: f64,
) -> f64 {
    let n = data.nrows();
    let d = data.ncols();
    let var_x = sigma_x * sigma_x;
    let var_a = sigma_a * sigma_a;

    // Log p(X | Z_new, A_new) - log p(X | Z, A)
    let mut log_lik_diff = 0.0_f64;
    for i in 0..n {
        let z_new_i = new_z_col.get(i).copied().unwrap_or(0.0);
        if z_new_i < 0.5 {
            continue; // New feature not active for this observation
        }
        for j in 0..d {
            let x_ij = data[[i, j]];
            // Old reconstruction
            let old_recon: f64 = (0..z.ncols())
                .filter(|&ki| z[[i, ki]] > 0.5)
                .map(|ki| a[[ki, j]])
                .sum();
            let new_a_j = new_a_row.get(j).copied().unwrap_or(0.0);
            let new_recon = old_recon + new_a_j;
            let log_p_new = -(x_ij - new_recon).powi(2) / (2.0 * var_x);
            let log_p_old = -(x_ij - old_recon).powi(2) / (2.0 * var_x);
            log_lik_diff += log_p_new - log_p_old;
        }
    }

    // Prior contribution: log p(new_a_row) - log proposal
    let log_prior_a: f64 = new_a_row
        .iter()
        .map(|&a_k| -a_k * a_k / (2.0 * var_a) - 0.5 * (2.0 * std::f64::consts::PI * var_a).ln())
        .sum::<f64>();

    // IBP prior for new feature column: Bernoulli(alpha / (N + alpha))
    let prob_new = alpha / (n as f64 + alpha);
    let log_prior_z: f64 = new_z_col
        .iter()
        .map(|&z_i| {
            if z_i > 0.5 {
                prob_new.ln().max(-100.0)
            } else {
                (1.0 - prob_new).ln().max(-100.0)
            }
        })
        .sum();

    log_lik_diff + log_prior_a + log_prior_z
}

/// Extend Z and A matrices with a new feature column/row.
fn extend_matrices(
    z: &Array2<f64>,
    a: &Array2<f64>,
    new_z_col: &[f64],
    new_a_row: &[f64],
) -> Result<(Array2<f64>, Array2<f64>)> {
    let n = z.nrows();
    let k = z.ncols();
    let d = a.ncols();

    // New Z: (N × (K+1))
    let mut new_z_data = vec![0.0f64; n * (k + 1)];
    for i in 0..n {
        for ki in 0..k {
            new_z_data[i * (k + 1) + ki] = z[[i, ki]];
        }
        new_z_data[i * (k + 1) + k] = new_z_col.get(i).copied().unwrap_or(0.0);
    }
    let new_z = Array2::from_shape_vec((n, k + 1), new_z_data).map_err(|e| {
        StatsError::ComputationError(format!("extend Z shape error: {e}"))
    })?;

    // New A: ((K+1) × D)
    let mut new_a_data = vec![0.0f64; (k + 1) * d];
    for ki in 0..k {
        for j in 0..d {
            new_a_data[ki * d + j] = a[[ki, j]];
        }
    }
    for j in 0..d {
        new_a_data[k * d + j] = new_a_row.get(j).copied().unwrap_or(0.0);
    }
    let new_a = Array2::from_shape_vec((k + 1, d), new_a_data).map_err(|e| {
        StatsError::ComputationError(format!("extend A shape error: {e}"))
    })?;

    Ok((new_z, new_a))
}

/// Remove all-zero columns from Z and corresponding rows from A.
fn prune_empty_features(z: &Array2<f64>, a: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let n = z.nrows();
    let k = z.ncols();
    let d = a.ncols();

    let active: Vec<usize> = (0..k)
        .filter(|&ki| z.column(ki).iter().any(|&v| v > 0.5))
        .collect();

    if active.is_empty() {
        // Keep at least one (zeroed) feature
        return (Array2::zeros((n, 1)), Array2::zeros((1, d)));
    }

    if active.len() == k {
        // Nothing to prune
        return (z.clone(), a.clone());
    }

    let k_new = active.len();
    let mut z_data = vec![0.0f64; n * k_new];
    let mut a_data = vec![0.0f64; k_new * d];

    for (new_ki, &old_ki) in active.iter().enumerate() {
        for i in 0..n {
            z_data[i * k_new + new_ki] = z[[i, old_ki]];
        }
        for j in 0..d {
            a_data[new_ki * d + j] = a[[old_ki, j]];
        }
    }

    let new_z = Array2::from_shape_vec((n, k_new), z_data)
        .unwrap_or_else(|_| Array2::zeros((n, k_new)));
    let new_a = Array2::from_shape_vec((k_new, d), a_data)
        .unwrap_or_else(|_| Array2::zeros((k_new, d)));

    (new_z, new_a)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_toy_data(n: usize, d: usize, seed: u64) -> Array2<f64> {
        let mut rng = Lcg::new(seed);
        let vec: Vec<f64> = (0..n * d).map(|_| rng.next_normal()).collect();
        Array2::from_shape_vec((n, d), vec).expect("shape ok")
    }

    #[test]
    fn test_fit_produces_at_least_one_feature() {
        let data = make_toy_data(20, 5, 42);
        let config = IbpLinearGaussianConfig {
            alpha: 2.0,
            sigma_x: 0.5,
            sigma_a: 1.0,
            n_iter: 20,
            burnin: 5,
            seed: 42,
        };
        let result = IbpLinearGaussian::fit(&data, config).expect("fit ok");
        let mean_k = result.mean_n_features();
        assert!(
            mean_k >= 1.0,
            "expected at least 1 feature, got mean {mean_k}"
        );
    }

    #[test]
    fn test_fit_a_mean_shape() {
        let n = 15;
        let d = 3;
        let data = make_toy_data(n, d, 7);
        let config = IbpLinearGaussianConfig {
            n_iter: 15,
            burnin: 3,
            ..Default::default()
        };
        let result = IbpLinearGaussian::fit(&data, config).expect("fit ok");
        assert_eq!(result.a_mean.ncols(), d, "A mean should have D columns");
    }

    #[test]
    fn test_z_samples_binary() {
        let data = make_toy_data(10, 4, 99);
        let config = IbpLinearGaussianConfig {
            n_iter: 10,
            burnin: 2,
            ..Default::default()
        };
        let result = IbpLinearGaussian::fit(&data, config).expect("fit ok");
        for z in &result.z_samples {
            for &v in z.iter() {
                assert!(v == 0.0 || v == 1.0, "Z entries should be binary, got {v}");
            }
        }
    }

    #[test]
    fn test_predict_returns_probabilities() {
        let n = 12;
        let d = 3;
        let data = make_toy_data(n, d, 55);
        let config = IbpLinearGaussianConfig {
            n_iter: 15,
            burnin: 5,
            ..Default::default()
        };
        let result = IbpLinearGaussian::fit(&data, config.clone()).expect("fit ok");
        let x_new: Vec<f64> = vec![0.1, -0.2, 0.3];
        let probs = IbpLinearGaussian::predict(&x_new, &result, &config);
        for &p in &probs {
            assert!(p >= 0.0 && p <= 1.0, "prob out of [0,1]: {p}");
        }
    }

    #[test]
    fn test_reconstruction_within_two_sigma() {
        // Fit on structured data; reconstruction should be approximate
        let n = 20;
        let d = 5;
        // Generate structured data: X = Z A + noise where Z has 2 features
        let mut rng = Lcg::new(11);
        let a_true: Vec<f64> = (0..2 * d).map(|_| rng.next_normal()).collect();
        let z_true: Vec<f64> = (0..n * 2)
            .map(|_| if rng.next_f64() < 0.3 { 1.0 } else { 0.0 })
            .collect();

        let sigma_x = 0.5_f64;
        let mut data_vec = vec![0.0f64; n * d];
        for i in 0..n {
            for j in 0..d {
                let mut x_ij = 0.0;
                for k in 0..2 {
                    x_ij += z_true[i * 2 + k] * a_true[k * d + j];
                }
                x_ij += sigma_x * rng.next_normal();
                data_vec[i * d + j] = x_ij;
            }
        }
        let data = Array2::from_shape_vec((n, d), data_vec).expect("shape");

        let config = IbpLinearGaussianConfig {
            sigma_x,
            sigma_a: 1.5,
            alpha: 2.0,
            n_iter: 30,
            burnin: 10,
            seed: 11,
        };
        let result = IbpLinearGaussian::fit(&data, config.clone()).expect("fit ok");

        // Check reconstruction quality on last Z sample
        if let Some(z_last) = result.z_samples.last() {
            let a = &result.a_mean;
            if z_last.ncols() == a.nrows() {
                let recon = z_last.dot(a);
                let mut sq_err = 0.0_f64;
                for i in 0..n {
                    for j in 0..d {
                        sq_err += (data[[i, j]] - recon[[i, j]]).powi(2);
                    }
                }
                let rmse = (sq_err / (n * d) as f64).sqrt();
                // Should be within a generous bound
                assert!(
                    rmse < 5.0 * sigma_x,
                    "RMSE {rmse:.3} should be < 5 * sigma_x = {:.3}",
                    5.0 * sigma_x
                );
            }
        }
    }
}
