//! Online Variational Bayes via stochastic natural-gradient updates.
//!
//! Implements the algorithm of Hoffman, Blei, Wang & Paisley (2010)
//! "Online Learning for Latent Dirichlet Allocation".
//!
//! # Overview
//! The key hyper-parameters controlling the learning schedule are:
//!
//! - `kappa ∈ (0.5, 1]` – forgetting exponent.
//! - `tau > 0`            – delay (down-weights early iterations).
//!
//! The Robbins–Monro step size at iteration `t` is:
//!
//! ```text
//! ρ_t = (t + tau)^{-kappa}
//! ```
//!
//! # LDA specialisation
//! [`OnlineVb::lda_update`] performs one mini-batch E-step (per-document
//! variational topic proportions φ and γ) followed by a natural-gradient
//! M-step on the global topic–word Dirichlet parameters λ.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Online Variational Bayes learner.
#[derive(Debug, Clone)]
pub struct OnlineVbConfig {
    /// Forgetting exponent κ ∈ (0.5, 1].
    pub kappa: f64,
    /// Delay τ > 0; down-weights early iterations.
    pub tau: f64,
    /// Mini-batch size.
    pub batch_size: usize,
    /// Total number of mini-batch updates to perform.
    pub n_iter: usize,
}

impl Default for OnlineVbConfig {
    fn default() -> Self {
        Self {
            kappa: 0.7,
            tau: 1.0,
            batch_size: 64,
            n_iter: 1000,
        }
    }
}

// ---------------------------------------------------------------------------
// Core struct
// ---------------------------------------------------------------------------

/// Online Variational Bayes learner.
///
/// `lambda` stores the global variational parameters; for LDA these are the
/// K × V topic–word Dirichlet parameters.
pub struct OnlineVb {
    /// Global variational parameters (flattened; reshaped as needed by callers).
    pub lambda: Array1<f64>,
    /// Number of mini-batch updates performed so far.
    pub t: usize,
    /// Configuration.
    pub config: OnlineVbConfig,
    /// Total corpus size (used in natural-gradient scaling).
    n_total: f64,
    /// Number of topics K (set during LDA operations).
    n_topics: usize,
    /// Vocabulary size V (set during LDA operations).
    vocab_size: usize,
}

impl OnlineVb {
    /// Construct a new [`OnlineVb`] learner with given initial parameters.
    ///
    /// # Parameters
    /// - `init_lambda`: Initial flat variational parameters.
    /// - `n_total`: Total number of data points in the corpus (for scaling).
    /// - `config`: Scheduler configuration.
    pub fn new(init_lambda: Array1<f64>, n_total: f64, config: OnlineVbConfig) -> Self {
        Self {
            lambda: init_lambda,
            t: 0,
            config,
            n_total,
            n_topics: 0,
            vocab_size: 0,
        }
    }

    /// Robbins–Monro step size at the current iteration.
    ///
    /// `ρ_t = (t + τ)^{-κ}`
    pub fn step_size(&self) -> f64 {
        let t = self.t as f64;
        (t + self.config.tau).powf(-self.config.kappa)
    }

    /// Process one mini-batch of sufficient statistics.
    ///
    /// Performs a generic natural-gradient update:
    /// ```text
    /// λ ← (1 - ρ) · λ + ρ · (η + N / |B| · ss_batch)
    /// ```
    ///
    /// # Parameters
    /// - `batch_ss`: Sufficient statistics from a mini-batch, shape
    ///   `[1 × D_lambda]` (same length as `self.lambda`).
    ///
    /// # Returns
    /// An ELBO *estimate* (currently the norm of the gradient step).
    ///
    /// # Errors
    /// Returns [`StatsError::DimensionMismatch`] if `batch_ss` has the wrong
    /// second dimension.
    pub fn update_batch(&mut self, batch_ss: &Array2<f64>) -> StatsResult<f64> {
        let d_lambda = self.lambda.len();
        if batch_ss.ncols() != d_lambda {
            return Err(StatsError::DimensionMismatch(format!(
                "update_batch: batch_ss has {} cols, expected {}",
                batch_ss.ncols(),
                d_lambda
            )));
        }
        let rho = self.step_size();
        let batch_size = batch_ss.nrows() as f64;
        let scale = self.n_total / batch_size;

        // Sum sufficient statistics across batch rows.
        let mut ss_sum = Array1::<f64>::zeros(d_lambda);
        for row in batch_ss.rows() {
            for (j, &v) in row.iter().enumerate() {
                ss_sum[j] += v;
            }
        }

        // Natural-gradient target: η + scale * ss (η=1 prior for simplicity).
        let eta = 1.0_f64;
        let mut elbo_grad_norm = 0.0_f64;
        for j in 0..d_lambda {
            let lambda_tilde = eta + scale * ss_sum[j];
            let delta = rho * (lambda_tilde - self.lambda[j]);
            elbo_grad_norm += delta * delta;
            self.lambda[j] += delta;
        }
        self.t += 1;
        Ok(elbo_grad_norm.sqrt())
    }

    /// LDA-style online update for a mini-batch of documents.
    ///
    /// Performs the Hoffman et al. (2010) E-step + M-step:
    ///
    /// 1. **E-step** (per document): iterate until γ converges or for a fixed
    ///    number of inner steps, updating φ (word-topic) and γ (doc-topic).
    /// 2. **M-step**: compute expected sufficient statistics and apply a
    ///    Robbins–Monro update to `lambda` (the K × V Dirichlet params).
    ///
    /// `lambda` is interpreted as a flattened K × V array (row-major).
    ///
    /// # Parameters
    /// - `docs`: Mini-batch of documents; each `Vec<usize>` is word indices.
    /// - `vocab_size`: Vocabulary size V.
    /// - `alpha`: Symmetric Dirichlet prior on document-topic distributions.
    ///
    /// # Returns
    /// Approximate ELBO contribution from this batch.
    ///
    /// # Errors
    /// Returns an error if `vocab_size == 0`, `n_topics == 0`, or the
    /// `lambda` array length is not K × V.
    pub fn lda_update(
        &mut self,
        docs: &[Vec<usize>],
        vocab_size: usize,
        alpha: f64,
    ) -> StatsResult<f64> {
        if vocab_size == 0 {
            return Err(StatsError::InvalidArgument(
                "lda_update: vocab_size must be > 0".to_string(),
            ));
        }
        if self.lambda.is_empty() {
            return Err(StatsError::InvalidArgument(
                "lda_update: lambda is empty; initialise via new()".to_string(),
            ));
        }
        let k = self.n_topics;
        let v = vocab_size;
        if k == 0 {
            return Err(StatsError::InvalidArgument(
                "lda_update: n_topics not set; use new_lda() constructor".to_string(),
            ));
        }
        if self.lambda.len() != k * v {
            return Err(StatsError::DimensionMismatch(format!(
                "lda_update: lambda has {} elements but K×V = {k}×{v} = {}",
                self.lambda.len(),
                k * v
            )));
        }
        self.vocab_size = v;

        let rho = self.step_size();
        let n_total = self.n_total;
        let batch_size = docs.len() as f64;
        let scale = n_total / batch_size.max(1.0);
        let eta = 1.0_f64 / k as f64; // symmetric word prior

        // Compute E[log β_{kw}] = ψ(λ_{kw}) - ψ(Σ_v λ_{kv}).
        // We use a fast digamma approximation.
        let e_log_beta = compute_e_log_beta(&self.lambda, k, v);

        // Accumulate sufficient statistics: ss[k][w] = Σ_d Σ_n φ_{d,n,k} * 1[w_n = w]
        let mut ss = vec![vec![0.0_f64; v]; k];

        let inner_iter = 20usize; // E-step iterations per document
        let mut elbo = 0.0_f64;

        for doc in docs {
            if doc.is_empty() {
                continue;
            }
            // Initialise γ_d = α + N/K.
            let n_words = doc.len() as f64;
            let mut gamma_d = vec![alpha + n_words / k as f64; k];

            // φ_{d,n} (word × topic): length K per word.
            // We collapse: store φ_counts[k] = Σ_n φ_{d,n,k}.
            let mut phi_counts = vec![0.0_f64; k];

            for _inner in 0..inner_iter {
                // E[log θ_{d,k}] = ψ(γ_dk) - ψ(Σ_k γ_dk).
                let e_log_theta = digamma_vec(&gamma_d);

                // Recompute φ for each word token.
                let mut new_gamma = vec![alpha; k];
                for &w in doc.iter() {
                    if w >= v {
                        continue; // skip out-of-range
                    }
                    // φ_{n,k} ∝ exp(E[log θ_{dk}] + E[log β_{kw}])
                    let mut phi_w = vec![0.0_f64; k];
                    let mut phi_sum = 0.0_f64;
                    for ki in 0..k {
                        let val = (e_log_theta[ki] + e_log_beta[ki * v + w]).exp();
                        phi_w[ki] = val;
                        phi_sum += val;
                    }
                    if phi_sum > 0.0 {
                        for ki in 0..k {
                            phi_w[ki] /= phi_sum;
                            new_gamma[ki] += phi_w[ki];
                        }
                    } else {
                        let uniform = 1.0 / k as f64;
                        for ki in 0..k {
                            new_gamma[ki] += uniform;
                        }
                    }
                }
                gamma_d = new_gamma;
            }

            // Compute phi_counts for sufficient statistics.
            // Recompute E-step one final time to collect ss.
            let e_log_theta = digamma_vec(&gamma_d);
            for &w in doc.iter() {
                if w >= v {
                    continue;
                }
                let mut phi_w = vec![0.0_f64; k];
                let mut phi_sum = 0.0_f64;
                for ki in 0..k {
                    let val = (e_log_theta[ki] + e_log_beta[ki * v + w]).exp();
                    phi_w[ki] = val;
                    phi_sum += val;
                }
                if phi_sum > 0.0 {
                    for ki in 0..k {
                        phi_counts[ki] += phi_w[ki] / phi_sum;
                        ss[ki][w] += phi_w[ki] / phi_sum;
                    }
                } else {
                    let uniform = 1.0 / k as f64;
                    for ki in 0..k {
                        phi_counts[ki] += uniform;
                        ss[ki][w] += uniform;
                    }
                }
            }

            // Approximate per-doc ELBO contribution.
            let gamma_sum: f64 = gamma_d.iter().sum();
            let e_log_theta = digamma_vec(&gamma_d);
            for ki in 0..k {
                let e_lt = e_log_theta[ki];
                elbo += (alpha - gamma_d[ki]) * e_lt;
                elbo += lgamma(gamma_d[ki]) - lgamma(alpha);
                elbo += phi_counts[ki] * e_lt;
            }
            elbo += lgamma(k as f64 * alpha) - lgamma(gamma_sum);
        }

        // M-step: λ̃_{kw} = η + (N/|B|) * ss_{kw}; λ ← (1-ρ)·λ + ρ·λ̃
        for ki in 0..k {
            for w in 0..v {
                let lambda_tilde = eta + scale * ss[ki][w];
                let idx = ki * v + w;
                self.lambda[idx] = (1.0 - rho) * self.lambda[idx] + rho * lambda_tilde;
            }
        }
        self.t += 1;
        Ok(elbo)
    }

    /// Expected parameters E[β_{kw}] = λ_{kw} / Σ_v λ_{kv}.
    ///
    /// Returns a flat array of length K × V (row-major).
    pub fn expected_params(&self) -> Array1<f64> {
        let len = self.lambda.len();
        if len == 0 {
            return Array1::zeros(0);
        }
        let k = self.n_topics;
        let v = if k > 0 { len / k } else { len };
        let mut result = Array1::<f64>::zeros(len);
        for ki in 0..k {
            let row_sum: f64 = (0..v).map(|w| self.lambda[ki * v + w]).sum();
            let denom = if row_sum > 0.0 { row_sum } else { 1.0 };
            for w in 0..v {
                result[ki * v + w] = self.lambda[ki * v + w] / denom;
            }
        }
        result
    }
}

/// Construct an [`OnlineVb`] pre-configured for LDA-style updates.
///
/// Initialises `lambda` to `1/K + small_noise` to break symmetry.
///
/// # Parameters
/// - `n_topics`: Number of topics K.
/// - `vocab_size`: Vocabulary size V.
/// - `n_total`: Total corpus size (number of documents).
/// - `config`: Scheduler configuration.
/// - `seed`: LCG seed for noise initialisation.
pub fn new_lda(
    n_topics: usize,
    vocab_size: usize,
    n_total: f64,
    config: OnlineVbConfig,
    seed: u64,
) -> StatsResult<OnlineVb> {
    if n_topics == 0 {
        return Err(StatsError::InvalidArgument(
            "new_lda: n_topics must be > 0".to_string(),
        ));
    }
    if vocab_size == 0 {
        return Err(StatsError::InvalidArgument(
            "new_lda: vocab_size must be > 0".to_string(),
        ));
    }

    // LCG noise to break symmetry.
    let mut lcg = seed ^ 0x9e37_79b9_7f4a_7c15_u64;
    let noise_scale = 1.0 / vocab_size as f64;
    let init_val = 1.0 / n_topics as f64;
    let lambda_data: Vec<f64> = (0..n_topics * vocab_size)
        .map(|_| {
            lcg = lcg
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (lcg >> 11) as f64 / (1u64 << 53) as f64;
            init_val + noise_scale * u * 0.01
        })
        .collect();

    let lambda = Array1::from_vec(lambda_data);
    let mut vb = OnlineVb::new(lambda, n_total, config);
    vb.n_topics = n_topics;
    vb.vocab_size = vocab_size;
    Ok(vb)
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

/// Digamma function approximation (Abramowitz & Stegun 6.3.18).
fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if x < 6.0 {
        // Recurse upward: ψ(x) = ψ(x+1) - 1/x
        return digamma(x + 1.0) - 1.0 / x;
    }
    // Asymptotic series for x >= 6.
    let x_inv = 1.0 / x;
    let x_inv2 = x_inv * x_inv;
    x.ln() - 0.5 * x_inv - x_inv2 * (1.0 / 12.0 - x_inv2 * (1.0 / 120.0 - x_inv2 * (1.0 / 252.0)))
}

/// Vectorised digamma: E[log θ_k] = ψ(γ_k) - ψ(Σ_k γ_k).
fn digamma_vec(gamma: &[f64]) -> Vec<f64> {
    let sum: f64 = gamma.iter().sum();
    let psi_sum = digamma(sum);
    gamma.iter().map(|&g| digamma(g) - psi_sum).collect()
}

/// E[log β_{kw}] = ψ(λ_{kw}) - ψ(Σ_v λ_{kv}) for all k, w.
/// Returns flat array length K × V.
fn compute_e_log_beta(lambda: &Array1<f64>, k: usize, v: usize) -> Vec<f64> {
    let mut result = vec![0.0_f64; k * v];
    for ki in 0..k {
        let row_sum: f64 = (0..v).map(|w| lambda[ki * v + w]).sum();
        let psi_sum = digamma(row_sum);
        for w in 0..v {
            result[ki * v + w] = digamma(lambda[ki * v + w]) - psi_sum;
        }
    }
    result
}

/// Natural log of the Gamma function via Stirling/Lanczos approximation.
fn lgamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    if x < 0.5 {
        // Reflection formula: Γ(x)Γ(1-x) = π/sin(πx)
        let s = std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * x);
        return s.abs().ln();
    }
    // Lanczos approximation (g=7, n=9 coefficients, Numerical Recipes).
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x - 1.0;
    let mut sum = C[0];
    for (i, &c) in C[1..].iter().enumerate() {
        sum += c / (x + i as f64 + 1.0);
    }
    let t = x + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_corpus() -> Vec<Vec<usize>> {
        vec![
            vec![0, 1, 2, 0, 1],
            vec![3, 4, 5, 3, 4],
            vec![0, 2, 1, 0],
            vec![3, 5, 4, 3],
        ]
    }

    #[test]
    fn test_config_defaults() {
        let cfg = OnlineVbConfig::default();
        assert!((cfg.kappa - 0.7).abs() < 1e-10);
        assert!((cfg.tau - 1.0).abs() < 1e-10);
        assert_eq!(cfg.batch_size, 64);
        assert_eq!(cfg.n_iter, 1000);
    }

    #[test]
    fn test_step_size_decreases() {
        let cfg = OnlineVbConfig::default();
        let mut vb = OnlineVb::new(Array1::zeros(10), 100.0, cfg);
        let rho0 = vb.step_size();
        vb.t = 1;
        let rho1 = vb.step_size();
        assert!(
            rho1 < rho0,
            "step size should decrease: rho0={rho0}, rho1={rho1}"
        );
    }

    #[test]
    fn test_step_size_monotone_decrease() {
        let cfg = OnlineVbConfig::default();
        let mut vb = OnlineVb::new(Array1::zeros(6), 100.0, cfg);
        let mut prev = vb.step_size();
        for t in 1..10 {
            vb.t = t;
            let cur = vb.step_size();
            assert!(cur < prev, "step size not monotone at t={t}");
            prev = cur;
        }
    }

    #[test]
    fn test_lda_update_lambda_grows() {
        let docs = small_corpus();
        let cfg = OnlineVbConfig {
            kappa: 0.7,
            tau: 1.0,
            batch_size: 4,
            n_iter: 10,
        };
        let mut vb = new_lda(3, 6, 4.0, cfg, 42).expect("init");
        let lambda_before = vb.lambda.sum();
        let _elbo = vb.lda_update(&docs, 6, 0.1).expect("update");
        let lambda_after = vb.lambda.sum();
        // Lambda should have changed (not still identical to init).
        assert!(
            (lambda_after - lambda_before).abs() > 1e-12,
            "lambda unchanged after update"
        );
    }

    #[test]
    fn test_expected_params_sums_per_row() {
        let cfg = OnlineVbConfig::default();
        let mut vb = new_lda(3, 6, 4.0, cfg, 1).expect("init");
        let docs = small_corpus();
        let _ = vb.lda_update(&docs, 6, 0.1);
        let ep = vb.expected_params();
        let k = 3;
        let v = 6;
        for ki in 0..k {
            let row_sum: f64 = (0..v).map(|w| ep[ki * v + w]).sum();
            assert!((row_sum - 1.0).abs() < 1e-9, "topic {ki} sum = {row_sum}");
        }
    }

    #[test]
    fn test_forgetting_kappa_max() {
        // kappa=1 means maximum forgetting (ρ_t = 1/(t+τ)).
        let cfg = OnlineVbConfig {
            kappa: 1.0,
            tau: 1.0,
            ..Default::default()
        };
        let mut vb = OnlineVb::new(Array1::zeros(6), 10.0, cfg);
        let rho0 = vb.step_size();
        vb.t = 10;
        let rho10 = vb.step_size();
        assert!(rho0 > rho10);
    }

    #[test]
    fn test_update_batch_dimension_mismatch() {
        let cfg = OnlineVbConfig::default();
        let mut vb = OnlineVb::new(Array1::zeros(6), 100.0, cfg);
        let bad_batch = Array2::<f64>::zeros((2, 5)); // wrong ncols
        assert!(vb.update_batch(&bad_batch).is_err());
    }

    #[test]
    fn test_lda_vocab_zero_error() {
        let cfg = OnlineVbConfig::default();
        let mut vb = new_lda(3, 6, 10.0, cfg, 1).expect("init");
        assert!(vb.lda_update(&[vec![0, 1]], 0, 0.1).is_err());
    }

    #[test]
    fn test_lda_empty_doc_no_panic() {
        let cfg = OnlineVbConfig {
            kappa: 0.7,
            tau: 1.0,
            batch_size: 4,
            n_iter: 5,
        };
        let mut vb = new_lda(3, 6, 4.0, cfg, 42).expect("init");
        let docs = vec![vec![], vec![0usize, 1, 2]];
        assert!(vb.lda_update(&docs, 6, 0.1).is_ok());
    }

    #[test]
    fn test_update_batch_changes_lambda() {
        let cfg = OnlineVbConfig::default();
        let init = Array1::from_vec(vec![1.0_f64; 6]);
        let mut vb = OnlineVb::new(init, 100.0, cfg);
        let batch = Array2::from_shape_vec((1, 6), vec![2.0_f64; 6]).expect("shape");
        let before = vb.lambda.sum();
        let _ = vb.update_batch(&batch).expect("update");
        let after = vb.lambda.sum();
        assert!((after - before).abs() > 1e-12);
    }
}
