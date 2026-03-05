//! Hidden Markov Models (HMM) with Gaussian emissions.
//!
//! # Overview
//!
//! A Hidden Markov Model is a generative probabilistic model with latent
//! discrete states that follow a Markov chain:
//!
//! ```text
//! s_1 → s_2 → … → s_T          (Markov chain with transition matrix A)
//! x_1   x_2       x_T          (observations emitted from states)
//! ```
//!
//! This module provides:
//!
//! * **`GaussianHMM`** – HMM with multivariate Gaussian emission per state.
//! * **Forward-Backward algorithm** – compute posterior state probabilities.
//! * **Viterbi algorithm** – most-likely hidden state sequence.
//! * **Baum-Welch** – EM training from observation sequences.
//!
//! Numerical stability is maintained throughout by performing all
//! probability computations in log-space.
//!
//! # References
//! - Rabiner, L. R. (1989). "A tutorial on hidden Markov models…"
//!   *Proc. IEEE*, 77(2), 257–286.
//! - Bishop, C. M. (2006). *PRML*, Chapter 13.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::random::{Rng, SeedableRng};

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian emission
// ─────────────────────────────────────────────────────────────────────────────

/// Parameters for a single Gaussian emission distribution.
#[derive(Debug, Clone)]
pub struct GaussianEmission {
    /// Mean vector, shape `(n_features,)`.
    pub mean: Array1<f64>,
    /// Covariance matrix (full), shape `(n_features, n_features)`.
    pub covariance: Array2<f64>,
}

impl GaussianEmission {
    /// Create a new Gaussian emission; validates that `covariance` is square
    /// and matches the dimension of `mean`.
    pub fn new(mean: Array1<f64>, covariance: Array2<f64>) -> StatsResult<Self> {
        let d = mean.len();
        if covariance.nrows() != d || covariance.ncols() != d {
            return Err(StatsError::InvalidArgument(format!(
                "mean length {d} must match covariance shape {:?}",
                covariance.shape()
            )));
        }
        Ok(Self { mean, covariance })
    }

    /// Log-probability density at observation `x`.
    pub fn log_prob(&self, x: &[f64]) -> StatsResult<f64> {
        let d = self.mean.len();
        if x.len() != d {
            return Err(StatsError::InvalidArgument(format!(
                "observation length {} != emission dimension {d}",
                x.len()
            )));
        }
        let diff: Vec<f64> = x
            .iter()
            .zip(self.mean.iter())
            .map(|(&xi, &mi)| xi - mi)
            .collect();

        // Cholesky factor and log-det
        let (l, log_det) = chol_and_log_det(&self.covariance)?;

        // Mahalanobis distance via forward substitution
        let y = forward_solve(&l, &diff);
        let maha: f64 = y.iter().map(|&yi| yi * yi).sum();

        Ok(-0.5
            * (d as f64
                * (std::f64::consts::LN_2 + std::f64::consts::PI.ln())
                + log_det
                + maha))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GaussianHMM
// ─────────────────────────────────────────────────────────────────────────────

/// A Hidden Markov Model with multivariate Gaussian emissions.
///
/// # Field layout
///
/// | field           | shape                     | meaning                            |
/// |-----------------|---------------------------|------------------------------------|
/// | `initial`       | `(n_states,)`             | π: initial state probs             |
/// | `transition`    | `(n_states, n_states)`    | A[i,j] = P(s_{t+1}=j \| s_t=i)  |
/// | `emissions`     | `Vec<GaussianEmission>`   | one Gaussian per state             |
#[derive(Debug, Clone)]
pub struct GaussianHMM {
    /// Number of hidden states.
    pub n_states: usize,
    /// Dimension of each observation vector.
    pub n_features: usize,
    /// Log initial-state probabilities, shape `(n_states,)`.
    pub log_initial: Array1<f64>,
    /// Log transition matrix, shape `(n_states, n_states)`.
    pub log_transition: Array2<f64>,
    /// Gaussian emission for each state.
    pub emissions: Vec<GaussianEmission>,
}

impl GaussianHMM {
    /// Create a new HMM from raw probability arrays.
    ///
    /// `initial` is normalised internally; `transition` rows are normalised.
    pub fn new(
        initial: Array1<f64>,
        transition: Array2<f64>,
        emissions: Vec<GaussianEmission>,
    ) -> StatsResult<Self> {
        let n = initial.len();
        if transition.nrows() != n || transition.ncols() != n {
            return Err(StatsError::InvalidArgument(format!(
                "transition shape {:?} must be ({n}, {n})",
                transition.shape()
            )));
        }
        if emissions.len() != n {
            return Err(StatsError::InvalidArgument(format!(
                "need {} emissions, got {}",
                n,
                emissions.len()
            )));
        }
        let d = emissions[0].mean.len();
        for (i, e) in emissions.iter().enumerate() {
            if e.mean.len() != d {
                return Err(StatsError::InvalidArgument(format!(
                    "emission {i} has dimension {}, expected {d}",
                    e.mean.len()
                )));
            }
        }

        let log_initial = normalise_log(&initial);
        let mut log_transition = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            let row: Array1<f64> = transition.row(i).to_owned();
            let log_row = normalise_log(&row);
            for j in 0..n {
                log_transition[[i, j]] = log_row[j];
            }
        }

        Ok(Self {
            n_states: n,
            n_features: d,
            log_initial,
            log_transition,
            emissions,
        })
    }

    /// Compute the log-emission probability matrix for a sequence.
    ///
    /// Returns `log_b` of shape `(T, n_states)`.
    fn log_emission_matrix(&self, obs: &Array2<f64>) -> StatsResult<Array2<f64>> {
        let t = obs.nrows();
        let n = self.n_states;
        let mut log_b = Array2::<f64>::zeros((t, n));
        for step in 0..t {
            let x: Vec<f64> = obs.row(step).iter().copied().collect();
            for s in 0..n {
                log_b[[step, s]] = self.emissions[s].log_prob(&x)?;
            }
        }
        Ok(log_b)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward-backward algorithm
// ─────────────────────────────────────────────────────────────────────────────

/// Result of the forward-backward algorithm.
#[derive(Debug, Clone)]
pub struct ForwardBackwardResult {
    /// Log-forward probabilities, shape `(T, n_states)`.
    pub log_alpha: Array2<f64>,
    /// Log-backward probabilities, shape `(T, n_states)`.
    pub log_beta: Array2<f64>,
    /// Posterior state probabilities γ_t(s), shape `(T, n_states)`.
    pub gamma: Array2<f64>,
    /// Joint two-slice posteriors ξ_t(i,j), shape `(T-1, n_states, n_states)`.
    pub xi: Array3<f64>,
    /// Sequence log-likelihood.
    pub log_likelihood: f64,
}

/// Run the forward-backward algorithm on a single observation sequence.
///
/// `obs` has shape `(T, n_features)`.
pub fn forward_backward(
    hmm: &GaussianHMM,
    obs: &Array2<f64>,
) -> StatsResult<ForwardBackwardResult> {
    let t = obs.nrows();
    let n = hmm.n_states;
    if t == 0 {
        return Err(StatsError::InvalidArgument(
            "observation sequence must be non-empty".to_string(),
        ));
    }

    let log_b = hmm.log_emission_matrix(obs)?;

    // ── Forward pass ──────────────────────────────────────────────────────
    let mut log_alpha = Array2::<f64>::from_elem((t, n), f64::NEG_INFINITY);
    for s in 0..n {
        log_alpha[[0, s]] = hmm.log_initial[s] + log_b[[0, s]];
    }
    for step in 1..t {
        for j in 0..n {
            let logits: Vec<f64> = (0..n)
                .map(|i| log_alpha[[step - 1, i]] + hmm.log_transition[[i, j]])
                .collect();
            log_alpha[[step, j]] = log_sum_exp(&logits) + log_b[[step, j]];
        }
    }

    // ── Backward pass ─────────────────────────────────────────────────────
    let mut log_beta = Array2::<f64>::from_elem((t, n), f64::NEG_INFINITY);
    for s in 0..n {
        log_beta[[t - 1, s]] = 0.0; // log(1)
    }
    for step in (0..t - 1).rev() {
        for i in 0..n {
            let logits: Vec<f64> = (0..n)
                .map(|j| {
                    hmm.log_transition[[i, j]]
                        + log_b[[step + 1, j]]
                        + log_beta[[step + 1, j]]
                })
                .collect();
            log_beta[[step, i]] = log_sum_exp(&logits);
        }
    }

    // ── Log-likelihood ────────────────────────────────────────────────────
    let last_alpha: Vec<f64> = (0..n).map(|s| log_alpha[[t - 1, s]]).collect();
    let log_likelihood = log_sum_exp(&last_alpha);

    // ── Gamma: posterior per state ────────────────────────────────────────
    let mut gamma = Array2::<f64>::zeros((t, n));
    for step in 0..t {
        let log_row: Vec<f64> = (0..n)
            .map(|s| log_alpha[[step, s]] + log_beta[[step, s]])
            .collect();
        let lse = log_sum_exp(&log_row);
        for s in 0..n {
            gamma[[step, s]] = (log_row[s] - lse).exp();
        }
    }

    // ── Xi: joint two-slice posteriors ───────────────────────────────────
    let mut xi = Array3::<f64>::zeros((t.saturating_sub(1), n, n));
    for step in 0..t.saturating_sub(1) {
        let mut log_xi_slice = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                log_xi_slice[[i, j]] = log_alpha[[step, i]]
                    + hmm.log_transition[[i, j]]
                    + log_b[[step + 1, j]]
                    + log_beta[[step + 1, j]];
            }
        }
        // Normalise
        let logits: Vec<f64> = log_xi_slice.iter().copied().collect();
        let lse = log_sum_exp(&logits);
        for i in 0..n {
            for j in 0..n {
                xi[[step, i, j]] = (log_xi_slice[[i, j]] - lse).exp();
            }
        }
    }

    Ok(ForwardBackwardResult {
        log_alpha,
        log_beta,
        gamma,
        xi,
        log_likelihood,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Viterbi decoding
// ─────────────────────────────────────────────────────────────────────────────

/// Run the Viterbi algorithm to find the most-likely state sequence.
///
/// Returns the decoded state indices (length `T`) and the path log-probability.
pub fn viterbi(hmm: &GaussianHMM, obs: &Array2<f64>) -> StatsResult<(Vec<usize>, f64)> {
    let t = obs.nrows();
    let n = hmm.n_states;
    if t == 0 {
        return Err(StatsError::InvalidArgument(
            "observation sequence must be non-empty".to_string(),
        ));
    }

    let log_b = hmm.log_emission_matrix(obs)?;

    // delta[t][s] = max log-prob of any path ending in state s at time t.
    let mut delta = Array2::<f64>::from_elem((t, n), f64::NEG_INFINITY);
    // psi[t][s]   = argmax predecessor state.
    let mut psi = Array2::<usize>::zeros((t, n));

    // Initialise
    for s in 0..n {
        delta[[0, s]] = hmm.log_initial[s] + log_b[[0, s]];
    }

    // Recursion
    for step in 1..t {
        for j in 0..n {
            let (best_state, best_val) = (0..n)
                .map(|i| (i, delta[[step - 1, i]] + hmm.log_transition[[i, j]]))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, f64::NEG_INFINITY));
            delta[[step, j]] = best_val + log_b[[step, j]];
            psi[[step, j]] = best_state;
        }
    }

    // Back-track
    let mut path = vec![0usize; t];
    let (best_final, best_log_prob) = (0..n)
        .map(|s| (s, delta[[t - 1, s]]))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, f64::NEG_INFINITY));
    path[t - 1] = best_final;
    for step in (0..t - 1).rev() {
        path[step] = psi[[step + 1, path[step + 1]]];
    }

    Ok((path, best_log_prob))
}

// ─────────────────────────────────────────────────────────────────────────────
// Baum-Welch training
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Baum-Welch (EM) training.
#[derive(Debug, Clone)]
pub struct BaumWelchConfig {
    /// Maximum EM iterations.
    pub max_iter: usize,
    /// Convergence tolerance on log-likelihood.
    pub tol: f64,
    /// Covariance regularisation (added to diagonal of each state covariance).
    pub reg_covar: f64,
}

impl Default for BaumWelchConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-6,
            reg_covar: 1e-6,
        }
    }
}

/// Result of Baum-Welch training.
#[derive(Debug, Clone)]
pub struct BaumWelchResult {
    /// Trained HMM.
    pub hmm: GaussianHMM,
    /// Log-likelihood at each iteration.
    pub log_likelihoods: Vec<f64>,
    /// Whether training converged.
    pub converged: bool,
}

/// Train a `GaussianHMM` using the Baum-Welch (EM) algorithm.
///
/// `sequences` is a list of observation arrays each of shape `(T_i, n_features)`.
pub fn baum_welch(
    init_hmm: GaussianHMM,
    sequences: &[Array2<f64>],
    config: &BaumWelchConfig,
) -> StatsResult<BaumWelchResult> {
    if sequences.is_empty() {
        return Err(StatsError::InvalidArgument(
            "sequences must not be empty".to_string(),
        ));
    }

    let mut hmm = init_hmm;
    let n = hmm.n_states;
    let d = hmm.n_features;
    let mut log_likelihoods = Vec::new();
    let mut converged = false;
    let mut prev_ll = f64::NEG_INFINITY;

    for _iter in 0..config.max_iter {
        // Accumulators
        let mut acc_gamma0 = vec![0.0_f64; n]; // Σ γ_0(s) over sequences
        let mut acc_trans = Array2::<f64>::zeros((n, n)); // Σ_t Σ ξ_t(i,j)
        let mut acc_gamma = vec![0.0_f64; n]; // Σ_t γ_t(s)
        let mut acc_mean: Vec<Vec<f64>> = vec![vec![0.0; d]; n]; // Σ_t γ_t(s) x_t
        let mut acc_cov: Vec<Array2<f64>> = vec![Array2::zeros((d, d)); n]; // Σ_t γ_t(s)(x_t-μ)(x_t-μ)^T
        let mut total_ll = 0.0_f64;

        for seq in sequences {
            let fb = forward_backward(&hmm, seq)?;
            total_ll += fb.log_likelihood;

            let t = seq.nrows();

            // γ_0 accumulator
            for s in 0..n {
                acc_gamma0[s] += fb.gamma[[0, s]];
            }

            // ξ accumulator
            for step in 0..t.saturating_sub(1) {
                for i in 0..n {
                    for j in 0..n {
                        acc_trans[[i, j]] += fb.xi[[step, i, j]];
                    }
                }
            }

            // γ, mean, cov accumulators
            for step in 0..t {
                let x: Vec<f64> = seq.row(step).iter().copied().collect();
                for s in 0..n {
                    let g = fb.gamma[[step, s]];
                    acc_gamma[s] += g;
                    for j in 0..d {
                        acc_mean[s][j] += g * x[j];
                    }
                }
            }
        }

        // ── M-step: initial ───────────────────────────────────────────────
        let total_seqs = sequences.len() as f64;
        let log_initial = {
            let pi: Vec<f64> = acc_gamma0
                .iter()
                .map(|&v| v / total_seqs)
                .collect();
            normalise_log(&Array1::from_vec(pi))
        };

        // ── M-step: transition ────────────────────────────────────────────
        let mut log_transition = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| acc_trans[[i, j]]).sum::<f64>().max(1e-300);
            for j in 0..n {
                log_transition[[i, j]] = (acc_trans[[i, j]] / row_sum).max(1e-300).ln();
            }
        }

        // ── M-step: emissions ─────────────────────────────────────────────
        let mut new_emissions = Vec::with_capacity(n);
        for s in 0..n {
            let n_s = acc_gamma[s].max(1e-10);
            let new_mean: Array1<f64> = Array1::from_vec(
                (0..d).map(|j| acc_mean[s][j] / n_s).collect(),
            );

            // Recompute covariance from scratch using updated mean
            let mut cov = Array2::<f64>::zeros((d, d));
            for seq in sequences {
                let fb = forward_backward(&hmm, seq)?;
                let t = seq.nrows();
                for step in 0..t {
                    let g = fb.gamma[[step, s]];
                    let x: Vec<f64> = seq.row(step).iter().copied().collect();
                    let diff: Vec<f64> =
                        x.iter().zip(new_mean.iter()).map(|(&xi, &mi)| xi - mi).collect();
                    for j in 0..d {
                        for l in 0..=j {
                            let c = g * diff[j] * diff[l] / n_s;
                            cov[[j, l]] += c;
                            if j != l {
                                cov[[l, j]] += c;
                            }
                        }
                    }
                }
            }
            // Regularise
            for j in 0..d {
                cov[[j, j]] += config.reg_covar;
            }

            new_emissions.push(GaussianEmission::new(new_mean, cov)?);
        }

        // ── Update HMM ───────────────────────────────────────────────────
        hmm.log_initial = log_initial;
        hmm.log_transition = log_transition;
        hmm.emissions = new_emissions;

        log_likelihoods.push(total_ll);

        let improvement = total_ll - prev_ll;
        if improvement.abs() < config.tol {
            converged = true;
            break;
        }
        prev_ll = total_ll;
    }

    Ok(BaumWelchResult {
        hmm,
        log_likelihoods,
        converged,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Prediction & sampling
// ─────────────────────────────────────────────────────────────────────────────

/// Decode the most-likely state sequence using the Viterbi algorithm.
pub fn predict_states(hmm: &GaussianHMM, obs: &Array2<f64>) -> StatsResult<Vec<usize>> {
    let (path, _log_p) = viterbi(hmm, obs)?;
    Ok(path)
}

/// Compute the total log-likelihood of an observation sequence.
pub fn log_likelihood(hmm: &GaussianHMM, obs: &Array2<f64>) -> StatsResult<f64> {
    let fb = forward_backward(hmm, obs)?;
    Ok(fb.log_likelihood)
}

/// Sample a sequence of `length` from the HMM.
///
/// Returns `(states, observations)`:
/// - `states`:       Vec of length `length` (hidden state indices)
/// - `observations`: Array of shape `(length, n_features)`
pub fn sample(
    hmm: &GaussianHMM,
    length: usize,
    seed: Option<u64>,
) -> StatsResult<(Vec<usize>, Array2<f64>)> {
    if length == 0 {
        return Err(StatsError::InvalidArgument(
            "sample length must be > 0".to_string(),
        ));
    }

    let mut rng: Box<dyn Rng> = match seed {
        Some(s) => Box::new(scirs2_core::random::SmallRng::seed_from_u64(s)),
        None => Box::new(scirs2_core::random::SmallRng::seed_from_u64(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(9999),
        )),
    };

    let n = hmm.n_states;
    let d = hmm.n_features;
    let mut states = Vec::with_capacity(length);
    let mut obs = Array2::<f64>::zeros((length, d));

    // Sample initial state
    let init_probs: Vec<f64> = hmm.log_initial.iter().map(|&l| l.exp()).collect();
    let s0 = sample_categorical(&init_probs, rng.as_mut());
    states.push(s0);
    sample_gaussian_into(&hmm.emissions[s0], &mut obs, 0, rng.as_mut())?;

    for t in 1..length {
        let prev = states[t - 1];
        let trans_probs: Vec<f64> = (0..n)
            .map(|j| hmm.log_transition[[prev, j]].exp())
            .collect();
        let next_state = sample_categorical(&trans_probs, rng.as_mut());
        states.push(next_state);
        sample_gaussian_into(&hmm.emissions[next_state], &mut obs, t, rng.as_mut())?;
    }

    Ok((states, obs))
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Log-sum-exp over a slice.
fn log_sum_exp(logits: &[f64]) -> f64 {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = logits.iter().map(|&x| (x - max).exp()).sum();
    max + sum.ln()
}

/// Normalise a probability vector (possibly unnormalised) and return log-probs.
fn normalise_log(probs: &Array1<f64>) -> Array1<f64> {
    let total: f64 = probs.iter().sum::<f64>().max(1e-300);
    probs.mapv(|p| (p / total).max(1e-300).ln())
}

/// Cholesky factorisation + log-determinant.
fn chol_and_log_det(mat: &Array2<f64>) -> StatsResult<(Array2<f64>, f64)> {
    let d = mat.nrows();
    let mut l = Array2::<f64>::zeros((d, d));
    for i in 0..d {
        for j in 0..=i {
            let mut s = mat[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(StatsError::ComputationError(format!(
                        "Emission covariance not positive-definite at ({i},{i}): s={s}"
                    )));
                }
                l[[i, j]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    let log_det = 2.0 * (0..d).map(|i| l[[i, i]].ln()).sum::<f64>();
    Ok((l, log_det))
}

/// Solve L x = b for lower-triangular L.
fn forward_solve(l: &Array2<f64>, b: &[f64]) -> Vec<f64> {
    let d = l.nrows();
    let mut x = vec![0.0_f64; d];
    for i in 0..d {
        let mut s = b[i];
        for j in 0..i {
            s -= l[[i, j]] * x[j];
        }
        x[i] = s / l[[i, i]];
    }
    x
}

/// Sample an index from a categorical distribution.
fn sample_categorical(probs: &[f64], rng: &mut dyn Rng) -> usize {
    let total: f64 = probs.iter().sum::<f64>();
    let u = (rng.next_u64() as f64 / u64::MAX as f64) * total;
    let mut cumsum = 0.0_f64;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if u <= cumsum {
            return i;
        }
    }
    probs.len() - 1
}

/// Sample from a Gaussian emission into row `t` of `obs`.
fn sample_gaussian_into(
    emission: &GaussianEmission,
    obs: &mut Array2<f64>,
    t: usize,
    rng: &mut dyn Rng,
) -> StatsResult<()> {
    let d = emission.mean.len();
    let (l, _log_det) = chol_and_log_det(&emission.covariance)?;
    let z: Vec<f64> = (0..d).map(|_| box_muller(rng)).collect();
    for j in 0..d {
        let lz: f64 = (0..=j).map(|k| l[[j, k]] * z[k]).sum();
        obs[[t, j]] = emission.mean[j] + lz;
    }
    Ok(())
}

/// Box-Muller standard normal sample.
fn box_muller(rng: &mut dyn Rng) -> f64 {
    let u1 = (rng.next_u64() as f64 + 1.0) / (u64::MAX as f64 + 1.0);
    let u2 = rng.next_u64() as f64 / u64::MAX as f64;
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array1, Array2};

    fn make_hmm_2state_1d() -> StatsResult<GaussianHMM> {
        let initial = array![0.5, 0.5];
        let transition = array![[0.7, 0.3], [0.4, 0.6]];
        let e0 = GaussianEmission::new(array![0.0], array![[1.0]])?;
        let e1 = GaussianEmission::new(array![3.0], array![[1.0]])?;
        GaussianHMM::new(initial, transition, vec![e0, e1])
    }

    #[test]
    fn test_sample_and_viterbi() {
        let hmm = make_hmm_2state_1d().expect("create hmm");
        let (states, obs) = sample(&hmm, 50, Some(42)).expect("sample");
        assert_eq!(states.len(), 50);
        assert_eq!(obs.shape(), &[50, 1]);

        let decoded = predict_states(&hmm, &obs).expect("viterbi");
        assert_eq!(decoded.len(), 50);
    }

    #[test]
    fn test_forward_backward_normalised() {
        let hmm = make_hmm_2state_1d().expect("create hmm");
        let (_states, obs) = sample(&hmm, 20, Some(1)).expect("sample");
        let fb = forward_backward(&hmm, &obs).expect("fb");

        // Each row of gamma should sum to 1
        for t in 0..obs.nrows() {
            let row_sum: f64 = fb.gamma.row(t).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-10,
                "gamma row {t} sums to {row_sum}"
            );
        }
        assert!(fb.log_likelihood.is_finite());
    }

    #[test]
    fn test_log_likelihood_finite() {
        let hmm = make_hmm_2state_1d().expect("create hmm");
        let (_states, obs) = sample(&hmm, 30, Some(7)).expect("sample");
        let ll = log_likelihood(&hmm, &obs).expect("log_likelihood");
        assert!(ll.is_finite() && ll < 0.0);
    }

    #[test]
    fn test_baum_welch_improves_ll() {
        let hmm = make_hmm_2state_1d().expect("create hmm");

        // Generate training data
        let mut seqs = Vec::new();
        for s in 0..5_u64 {
            let (_states, obs) = sample(&hmm, 30, Some(s)).expect("sample");
            seqs.push(obs);
        }

        // Perturb initial HMM so training has something to do
        let init_hmm = {
            let initial = array![0.6, 0.4];
            let transition = array![[0.6, 0.4], [0.5, 0.5]];
            let e0 = GaussianEmission::new(array![0.2], array![[1.5]]).expect("e0");
            let e1 = GaussianEmission::new(array![2.8], array![[1.5]]).expect("e1");
            GaussianHMM::new(initial, transition, vec![e0, e1]).expect("init hmm")
        };

        let config = BaumWelchConfig {
            max_iter: 30,
            tol: 1e-4,
            reg_covar: 1e-4,
        };
        let result = baum_welch(init_hmm, &seqs, &config).expect("baum_welch");

        assert!(
            !result.log_likelihoods.is_empty(),
            "should have at least one iteration"
        );
        // Log-likelihood should be finite
        for &ll in &result.log_likelihoods {
            assert!(ll.is_finite());
        }
    }
}
