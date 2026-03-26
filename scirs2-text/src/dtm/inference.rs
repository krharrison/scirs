//! Variational inference for the Dynamic Topic Model (DTM).
//!
//! Implements:
//! - Variational Kalman filter (forward pass)
//! - RTS smoother (backward pass)
//! - Per-document E-step (Dirichlet-LDA-style variational update)
//! - Global M-step (Kalman smoother on sufficient statistics)
//! - Main `fit` routine on `DynamicTopicModel`

use crate::dtm::model::normalise_to_simplex;
use crate::dtm::{DtmConfig, DtmResult, DynamicTopicModel};
use crate::error::{Result, TextError};

// ────────────────────────────────────────────────────────────────────────────
// Variational Kalman filter & RTS smoother
// ────────────────────────────────────────────────────────────────────────────

/// Variational Kalman forward pass for a single word dimension.
///
/// The model is:
/// ```text
///   β_{t,k,w} ~ N(β_{t-1,k,w}, σ²)   (state transition)
///   y_{t,k,w} ~ N(β_{t,k,w}, obs_noise)  (observation)
/// ```
///
/// # Arguments
/// * `observations`  – observed sufficient statistics at each time step (length T)
/// * `sigma_sq`      – state transition variance σ²
/// * `obs_noise`     – observation noise variance
///
/// # Returns
/// Tuple of four vectors (all length T):
/// `(filter_means, filter_vars, pred_means, pred_vars)`
pub fn kalman_forward(
    observations: &[f64],
    sigma_sq: f64,
    obs_noise: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let t = observations.len();
    if t == 0 {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }

    let mut filter_means = vec![0.0_f64; t];
    let mut filter_vars = vec![0.0_f64; t];
    let mut pred_means = vec![0.0_f64; t];
    let mut pred_vars = vec![0.0_f64; t];

    // Initialise: broad prior
    let prior_mean = observations[0];
    let prior_var = sigma_sq + obs_noise;

    // t = 0: prediction from prior
    pred_means[0] = prior_mean;
    pred_vars[0] = prior_var;

    // t = 0: update
    let k0 = pred_vars[0] / (pred_vars[0] + obs_noise);
    filter_means[0] = pred_means[0] + k0 * (observations[0] - pred_means[0]);
    filter_vars[0] = (1.0 - k0) * pred_vars[0];

    for s in 1..t {
        // Predict
        pred_means[s] = filter_means[s - 1];
        pred_vars[s] = filter_vars[s - 1] + sigma_sq;

        // Update (Kalman gain)
        let gain = pred_vars[s] / (pred_vars[s] + obs_noise);
        filter_means[s] = pred_means[s] + gain * (observations[s] - pred_means[s]);
        filter_vars[s] = (1.0 - gain) * pred_vars[s];
    }

    (filter_means, filter_vars, pred_means, pred_vars)
}

/// RTS (Rauch-Tung-Striebel) smoother — backward pass.
///
/// # Arguments
/// * `filter_means` – Kalman filter means (length T)
/// * `filter_vars`  – Kalman filter variances (length T)
/// * `pred_means`   – Kalman prediction means (length T, from forward pass)
/// * `pred_vars`    – Kalman prediction variances (length T, from forward pass)
/// * `sigma_sq`     – state transition variance σ²
///
/// # Returns
/// `(smoother_means, smoother_vars)` each of length T.
pub fn kalman_backward(
    filter_means: &[f64],
    filter_vars: &[f64],
    pred_means: &[f64],
    pred_vars: &[f64],
    sigma_sq: f64,
) -> (Vec<f64>, Vec<f64>) {
    let t = filter_means.len();
    if t == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut smoother_means = vec![0.0_f64; t];
    let mut smoother_vars = vec![0.0_f64; t];

    // Initialise from the last filter state
    smoother_means[t - 1] = filter_means[t - 1];
    smoother_vars[t - 1] = filter_vars[t - 1];

    for s in (0..t - 1).rev() {
        let pred_var_next = pred_vars[s + 1].max(1e-15);
        // Smoother gain
        let g = filter_vars[s] / pred_var_next;
        // Update means: m_s^smooth = m_s + G*(m_{s+1}^smooth - pred_m_{s+1})
        smoother_means[s] = filter_means[s] + g * (smoother_means[s + 1] - pred_means[s + 1]);
        // Update vars:  v_s^smooth = v_s + G²*(v_{s+1}^smooth - pred_v_{s+1})
        smoother_vars[s] = filter_vars[s] + g * g * (smoother_vars[s + 1] - pred_var_next);
        // Clamp to non-negative
        smoother_vars[s] = smoother_vars[s].max(1e-15);

        let _ = sigma_sq; // used in pred_var derivation above
    }

    (smoother_means, smoother_vars)
}

// ────────────────────────────────────────────────────────────────────────────
// E-step: per-document Dirichlet-LDA variational update
// ────────────────────────────────────────────────────────────────────────────

/// Dirichlet digamma approximation: ψ(x) ≈ ln(x) - 1/(2x) for x > 0.
fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        return -1e10;
    }
    // Abramowitz & Stegun approximation (good for x > 1; recurse otherwise)
    let mut z = x;
    let mut result = 0.0_f64;
    while z < 6.0 {
        result -= 1.0 / z;
        z += 1.0;
    }
    result += z.ln() - 0.5 / z - 1.0 / (12.0 * z * z) + 1.0 / (120.0 * z * z * z * z)
        - 1.0 / (252.0 * z * z * z * z * z * z);
    result
}

/// Variational E-step for a single document using Dirichlet-LDA updates.
///
/// # Arguments
/// * `doc_counts`   – word-count vector (length V)
/// * `gamma`        – Dirichlet variational parameter (length K, updated in place)
/// * `phi`          – topic assignment probabilities per word (K × V, updated in place)
/// * `beta_t`       – topic-word distributions at this time slice (K × V)
/// * `alpha`        – Dirichlet prior concentration
/// * `max_inner`    – max inner iterations
fn e_step_doc(
    doc_counts: &[f64],
    gamma: &mut [f64],
    phi: &mut [Vec<f64>],
    beta_t: &[Vec<f64>],
    alpha: f64,
    max_inner: usize,
) {
    let k = gamma.len();
    let vocab = doc_counts.len();

    for _ in 0..max_inner {
        // Update phi_{dkw} ∝ beta[k][w] * exp(ψ(gamma[k]))
        let dg: Vec<f64> = gamma.iter().map(|&g| digamma(g)).collect();
        for w in 0..vocab {
            if doc_counts[w] <= 0.0 {
                continue;
            }
            let mut row_sum = 0.0_f64;
            for t in 0..k {
                let beta_val = if t < beta_t.len() && w < beta_t[t].len() {
                    beta_t[t][w].max(1e-15)
                } else {
                    1e-15
                };
                phi[t][w] = beta_val * dg[t].exp();
                row_sum += phi[t][w];
            }
            if row_sum > 1e-15 {
                for t in 0..k {
                    phi[t][w] /= row_sum;
                }
            }
        }

        // Update gamma[k] = alpha + Σ_w c_w * phi[k][w]
        for t in 0..k {
            let weighted: f64 = (0..vocab).map(|w| doc_counts[w] * phi[t][w]).sum();
            gamma[t] = alpha + weighted;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Main fit routine
// ────────────────────────────────────────────────────────────────────────────

impl DynamicTopicModel {
    /// Fit the DTM to a corpus organised by time slice.
    ///
    /// # Arguments
    /// * `docs_by_time` – `T` lists of documents; each document is a word-count vector of length V
    /// * `vocab_size`   – vocabulary size V
    ///
    /// # Returns
    /// A [`DtmResult`] with topic-word trajectories and doc-topic distributions.
    pub fn fit(&self, docs_by_time: &[Vec<Vec<f64>>], vocab_size: usize) -> Result<DtmResult> {
        let n_time = docs_by_time.len();
        if n_time == 0 {
            return Err(TextError::InvalidInput(
                "Empty time-slice collection".into(),
            ));
        }

        let k = self.config.n_topics;
        let v = if vocab_size > 0 {
            vocab_size
        } else {
            docs_by_time
                .iter()
                .flat_map(|slice| slice.iter())
                .map(|d| d.len())
                .max()
                .unwrap_or(1)
        };
        let sigma_sq = self.config.sigma_sq;
        let alpha = self.config.alpha;
        let obs_noise = sigma_sq * 0.1_f64; // heuristic observation noise

        // ── Initialise β trajectories: K × T × V ──────────────────────────
        // Deterministic perturbation for reproducibility
        let mut trajectories: Vec<Vec<Vec<f64>>> = (0..k)
            .map(|ki| {
                (0..n_time)
                    .map(|ti| {
                        let mut row: Vec<f64> = (0..v)
                            .map(|wi| {
                                1.0 / v as f64
                                    + ((ki * 1009 + ti * 997 + wi * 991) % 1000) as f64 * 1e-5
                            })
                            .collect();
                        normalise_to_simplex(&mut row);
                        row
                    })
                    .collect()
            })
            .collect();

        // ── Flat doc-topic storage ─────────────────────────────────────────
        // doc_topic[t][d] = gamma vector (length K)
        let mut doc_gammas: Vec<Vec<Vec<f64>>> = docs_by_time
            .iter()
            .map(|slice| {
                slice
                    .iter()
                    .map(|_| vec![alpha + 1.0_f64 / k as f64; k])
                    .collect::<Vec<_>>()
            })
            .collect();

        for _iter in 0..self.config.max_iter {
            // ── E-step ──────────────────────────────────────────────────────
            // Collect sufficient statistics for each topic-word pair across time
            // suff_stats[k][t][w] = expected count of word w in topic k at time t
            let mut suff_stats: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0_f64; v]; n_time]; k];

            for (ti, slice) in docs_by_time.iter().enumerate() {
                let beta_t: Vec<Vec<f64>> = (0..k).map(|ki| trajectories[ki][ti].clone()).collect();

                for (di, doc_counts) in slice.iter().enumerate() {
                    let mut phi = vec![vec![0.0_f64; v]; k];
                    e_step_doc(
                        doc_counts,
                        &mut doc_gammas[ti][di],
                        &mut phi,
                        &beta_t,
                        alpha,
                        5,
                    );
                    // Accumulate into suff_stats
                    for ki in 0..k {
                        for w in 0..v {
                            suff_stats[ki][ti][w] +=
                                doc_counts.get(w).copied().unwrap_or(0.0) * phi[ki][w];
                        }
                    }
                }
            }

            // ── M-step: Kalman smoother per (k, w) ──────────────────────────
            for ki in 0..k {
                for w in 0..v {
                    // Collect observations for this (k, w) across time
                    let obs: Vec<f64> = (0..n_time)
                        .map(|ti| {
                            let total: f64 = (0..v).map(|ww| suff_stats[ki][ti][ww]).sum();
                            if total > 1e-15 {
                                (suff_stats[ki][ti][w] / total).max(1e-15)
                            } else {
                                1.0 / v as f64
                            }
                        })
                        .collect();

                    let (fm, fv, pm, pv) = kalman_forward(&obs, sigma_sq, obs_noise);
                    let (sm, _sv) = kalman_backward(&fm, &fv, &pm, &pv, sigma_sq);

                    for ti in 0..n_time {
                        trajectories[ki][ti][w] = sm[ti].max(1e-15);
                    }
                }

                // Re-normalise each time slice
                for ti in 0..n_time {
                    normalise_to_simplex(&mut trajectories[ki][ti]);
                }
            }
        }

        // ── Build doc-topic matrix ─────────────────────────────────────────
        // Normalise each gamma to get θ_d = gamma / sum(gamma)
        let mut doc_topic_matrix: Vec<Vec<f64>> = Vec::new();
        for slice_gammas in &doc_gammas {
            for gamma in slice_gammas {
                let s: f64 = gamma.iter().sum();
                let theta: Vec<f64> = gamma.iter().map(|&g| g / s.max(1e-15)).collect();
                doc_topic_matrix.push(theta);
            }
        }

        Ok(DtmResult {
            topic_word_trajectories: trajectories,
            doc_topic_matrix,
        })
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtm::{DtmConfig, DynamicTopicModel};

    fn make_slice(n_docs: usize, vocab: usize, seed: usize) -> Vec<Vec<f64>> {
        (0..n_docs)
            .map(|d| {
                (0..vocab)
                    .map(|w| ((d * 3 + w * 7 + seed) % 5) as f64)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn kalman_forward_correct_shape() {
        let obs = vec![0.1_f64, 0.15, 0.12, 0.18, 0.14];
        let (fm, fv, pm, pv) = kalman_forward(&obs, 0.01, 0.001);
        assert_eq!(fm.len(), 5);
        assert_eq!(fv.len(), 5);
        assert_eq!(pm.len(), 5);
        assert_eq!(pv.len(), 5);
    }

    #[test]
    fn kalman_backward_smoother_variance_le_filter_variance() {
        let obs = vec![0.1_f64, 0.15, 0.12, 0.18, 0.14, 0.13];
        let (fm, fv, pm, pv) = kalman_forward(&obs, 0.01, 0.001);
        let (_, sv) = kalman_backward(&fm, &fv, &pm, &pv, 0.01);
        // Smoother variance should not exceed filter variance (RTS property)
        for (i, (&sv_i, &fv_i)) in sv.iter().zip(fv.iter()).enumerate() {
            assert!(
                sv_i <= fv_i + 1e-10,
                "smoother_var[{i}]={sv_i} > filter_var[{i}]={fv_i}"
            );
        }
    }

    #[test]
    fn kalman_roundtrip_recovers_trajectory() {
        // Constant trajectory: smoother should converge near the constant
        let truth = 0.2_f64;
        let obs: Vec<f64> = vec![truth; 10];
        let (fm, fv, pm, pv) = kalman_forward(&obs, 1e-4, 1e-3);
        let (sm, _) = kalman_backward(&fm, &fv, &pm, &pv, 1e-4);
        for (i, &m) in sm.iter().enumerate() {
            assert!((m - truth).abs() < 0.05, "smoother[{i}]={m}, truth={truth}");
        }
    }

    #[test]
    fn dtm_fit_trajectories_shape() {
        let config = DtmConfig {
            n_topics: 2,
            n_time_slices: 3,
            max_iter: 5,
            sigma_sq: 0.1,
            alpha: 0.1,
        };
        let model = DynamicTopicModel::new(config);
        let docs_by_time: Vec<Vec<Vec<f64>>> = (0..3).map(|t| make_slice(4, 5, t)).collect();
        let res = model.fit(&docs_by_time, 5).expect("fit failed");
        // shape K × T × V
        assert_eq!(res.topic_word_trajectories.len(), 2);
        assert_eq!(res.topic_word_trajectories[0].len(), 3);
        assert_eq!(res.topic_word_trajectories[0][0].len(), 5);
    }

    #[test]
    fn dtm_fit_doc_topic_rows_sum_to_one() {
        let config = DtmConfig {
            n_topics: 2,
            n_time_slices: 3,
            max_iter: 3,
            sigma_sq: 0.1,
            alpha: 0.1,
        };
        let model = DynamicTopicModel::new(config);
        let docs_by_time: Vec<Vec<Vec<f64>>> = (0..3).map(|t| make_slice(3, 5, t)).collect();
        let res = model.fit(&docs_by_time, 5).expect("fit failed");
        for (d, row) in res.doc_topic_matrix.iter().enumerate() {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "doc {d} topic sum = {s}");
        }
    }

    #[test]
    fn dtm_fit_trajectories_row_sums_to_one() {
        let config = DtmConfig {
            n_topics: 2,
            n_time_slices: 3,
            max_iter: 3,
            sigma_sq: 0.1,
            alpha: 0.1,
        };
        let model = DynamicTopicModel::new(config);
        let docs_by_time: Vec<Vec<Vec<f64>>> = (0..3).map(|t| make_slice(3, 5, t)).collect();
        let res = model.fit(&docs_by_time, 5).expect("fit failed");
        for (ki, topic_traj) in res.topic_word_trajectories.iter().enumerate() {
            for (ti, row) in topic_traj.iter().enumerate() {
                let s: f64 = row.iter().sum();
                assert!(
                    (s - 1.0).abs() < 1e-4,
                    "topic {ki} time {ti} word sum = {s}"
                );
            }
        }
    }
}
