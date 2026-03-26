//! Nested (two-level) frailty model.
//!
//! Extends the shared frailty model to a hierarchical setting with two levels
//! of clustering (e.g., patients within hospitals within regions):
//!
//!   λ(t | x, v_i, w_ij) = v_i · w_ij · λ₀(t) · exp(x β)
//!
//! where v_i is the outer (region) frailty and w_ij is the inner (center) frailty.
//! Both are assumed Gamma-distributed by default.

use crate::error::{StatsError, StatsResult};

use super::shared_frailty::SharedFrailtyModel;
use super::types::{ClusterInfo, FrailtyConfig, FrailtyDistribution, FrailtyResult};

// ---------------------------------------------------------------------------
// Nested frailty result
// ---------------------------------------------------------------------------

/// Result of fitting a nested frailty model.
#[derive(Debug, Clone)]
pub struct NestedFrailtyResult {
    /// Estimated regression coefficients β.
    pub coefficients: Vec<f64>,
    /// Inner (center-level) frailty variance.
    pub inner_variance: f64,
    /// Outer (region-level) frailty variance.
    pub outer_variance: f64,
    /// Inner frailty estimates (one per inner cluster).
    pub inner_frailty_estimates: Vec<f64>,
    /// Outer frailty estimates (one per outer cluster).
    pub outer_frailty_estimates: Vec<f64>,
    /// Log-likelihood trace across EM iterations.
    pub log_likelihood_history: Vec<f64>,
    /// Whether the EM algorithm converged.
    pub converged: bool,
    /// Number of EM iterations performed.
    pub iterations: usize,
    /// Baseline cumulative hazard: (time, H₀(t)).
    pub baseline_hazard: Vec<(f64, f64)>,
}

// ---------------------------------------------------------------------------
// Nested Frailty Model
// ---------------------------------------------------------------------------

/// Two-level nested frailty model.
///
/// The hazard for subject k in inner cluster j within outer cluster i is:
///   h(t | x, v_i, w_ij) = v_i · w_ij · h₀(t) · exp(x β)
///
/// Fitted via a nested EM algorithm that alternates between updating
/// outer frailties, inner frailties, regression coefficients, and variance
/// components.
#[derive(Debug, Clone)]
pub struct NestedFrailtyModel {
    config: FrailtyConfig,
}

impl NestedFrailtyModel {
    /// Create a new nested frailty model with the given configuration.
    pub fn new(config: FrailtyConfig) -> Self {
        Self { config }
    }

    /// Fit the nested frailty model.
    ///
    /// # Arguments
    /// * `times`          – observed event/censoring times (n elements)
    /// * `events`         – event indicators
    /// * `covariates`     – covariate matrix (n rows)
    /// * `inner_clusters` – inner cluster assignment for each subject (e.g., center)
    /// * `outer_clusters` – outer cluster assignment for each subject (e.g., region)
    ///
    /// # Errors
    /// Returns `StatsError` on dimension mismatch, insufficient clusters, etc.
    pub fn fit(
        &self,
        times: &[f64],
        events: &[bool],
        covariates: &[&[f64]],
        inner_clusters: &[usize],
        outer_clusters: &[usize],
    ) -> StatsResult<NestedFrailtyResult> {
        let n = times.len();

        // --- Input validation ---
        if n == 0 {
            return Err(StatsError::InvalidArgument(
                "times must not be empty".into(),
            ));
        }
        if events.len() != n
            || covariates.len() != n
            || inner_clusters.len() != n
            || outer_clusters.len() != n
        {
            return Err(StatsError::DimensionMismatch(format!(
                "All input arrays must have length {n}"
            )));
        }
        let n_events = events.iter().filter(|&&e| e).count();
        if n_events == 0 {
            return Err(StatsError::InvalidArgument("No events observed".into()));
        }
        for &t in times {
            if !t.is_finite() || t < 0.0 {
                return Err(StatsError::InvalidArgument(format!(
                    "times must be finite and non-negative; got {t}"
                )));
            }
        }

        // Build cluster structures
        let inner_infos = build_cluster_infos(inner_clusters, events)?;
        let outer_infos = build_cluster_infos(outer_clusters, events)?;

        if inner_infos.len() < 2 {
            return Err(StatsError::InvalidArgument(
                "At least two inner clusters are required".into(),
            ));
        }
        if outer_infos.len() < 2 {
            return Err(StatsError::InvalidArgument(
                "At least two outer clusters are required".into(),
            ));
        }

        // Map subject -> inner/outer cluster index
        let mut subj_to_inner = vec![0_usize; n];
        for (k, ci) in inner_infos.iter().enumerate() {
            for &idx in &ci.subject_indices {
                subj_to_inner[idx] = k;
            }
        }
        let mut subj_to_outer = vec![0_usize; n];
        for (k, ci) in outer_infos.iter().enumerate() {
            for &idx in &ci.subject_indices {
                subj_to_outer[idx] = k;
            }
        }

        // Map inner cluster -> outer cluster (majority vote)
        let inner_to_outer: Vec<usize> = inner_infos
            .iter()
            .map(|ci| {
                if ci.subject_indices.is_empty() {
                    0
                } else {
                    // Use the outer cluster of the first subject
                    subj_to_outer[ci.subject_indices[0]]
                }
            })
            .collect();

        let p = if !covariates.is_empty() {
            covariates[0].len()
        } else {
            0
        };

        // --- Initialise parameters ---
        let mut beta = vec![0.0_f64; p];
        let mut theta_inner = self.config.initial_variance;
        let mut theta_outer = self.config.initial_variance;
        let mut inner_frailties = vec![1.0_f64; inner_infos.len()];
        let mut outer_frailties = vec![1.0_f64; outer_infos.len()];
        let mut ll_history: Vec<f64> = Vec::new();
        let mut converged = false;
        let mut iterations = 0_usize;

        // Sort order
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            times[a]
                .partial_cmp(&times[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // --- Nested EM iterations ---
        for iter in 0..self.config.max_iterations {
            iterations = iter + 1;

            // Combined frailty per subject: u_j = v_{outer(j)} * w_{inner(j)}
            let combined_frailties: Vec<f64> = (0..n)
                .map(|j| outer_frailties[subj_to_outer[j]] * inner_frailties[subj_to_inner[j]])
                .collect();

            // Compute exp(x β)
            let exp_xb: Vec<f64> = (0..n)
                .map(|j| {
                    let mut lin = 0.0_f64;
                    for col in 0..p {
                        lin += covariates[j][col] * beta[col];
                    }
                    lin.exp()
                })
                .collect();

            // Cumulative hazard per inner cluster
            let inner_cum_haz = compute_cluster_cumulative_hazard(
                &inner_infos,
                &exp_xb,
                &combined_frailties,
                times,
                events,
                &order,
            );

            // E-step: update inner frailties (given outer frailties)
            update_frailties_gamma(
                &mut inner_frailties,
                &inner_infos,
                &inner_cum_haz,
                theta_inner,
            );

            // Recompute combined frailties with updated inner
            let combined_frailties2: Vec<f64> = (0..n)
                .map(|j| outer_frailties[subj_to_outer[j]] * inner_frailties[subj_to_inner[j]])
                .collect();

            // Cumulative hazard per outer cluster
            let outer_cum_haz = compute_cluster_cumulative_hazard(
                &outer_infos,
                &exp_xb,
                &combined_frailties2,
                times,
                events,
                &order,
            );

            // E-step: update outer frailties
            update_frailties_gamma(
                &mut outer_frailties,
                &outer_infos,
                &outer_cum_haz,
                theta_outer,
            );

            // M-step: update variance components
            theta_inner = moment_variance(&inner_frailties).max(1e-8);
            theta_outer = moment_variance(&outer_frailties).max(1e-8);

            // M-step: update β via shared frailty model (treating combined frailties as fixed)
            if p > 0 {
                let combined: Vec<f64> = (0..n)
                    .map(|j| outer_frailties[subj_to_outer[j]] * inner_frailties[subj_to_inner[j]])
                    .collect();
                update_beta_newton(&mut beta, times, events, covariates, &combined, &order);
            }

            // Compute log-likelihood
            let ll = nested_log_likelihood(
                times,
                events,
                covariates,
                &beta,
                &inner_frailties,
                &outer_frailties,
                &subj_to_inner,
                &subj_to_outer,
                &order,
                theta_inner,
                theta_outer,
                &inner_infos,
                &outer_infos,
            );
            ll_history.push(ll);

            // Convergence check
            if ll_history.len() >= 2 {
                let prev = ll_history[ll_history.len() - 2];
                let rel_change = if prev.abs() > 1e-12 {
                    (ll - prev).abs() / prev.abs()
                } else {
                    (ll - prev).abs()
                };
                if rel_change < self.config.tolerance {
                    converged = true;
                    break;
                }
            }
        }

        // Final baseline hazard
        let combined: Vec<f64> = (0..n)
            .map(|j| outer_frailties[subj_to_outer[j]] * inner_frailties[subj_to_inner[j]])
            .collect();
        let exp_xb: Vec<f64> = (0..n)
            .map(|j| {
                let mut lin = 0.0_f64;
                for col in 0..p {
                    lin += covariates[j][col] * beta[col];
                }
                lin.exp()
            })
            .collect();
        let baseline_hazard = compute_baseline_hazard(times, events, &exp_xb, &combined, &order);

        Ok(NestedFrailtyResult {
            coefficients: beta,
            inner_variance: theta_inner,
            outer_variance: theta_outer,
            inner_frailty_estimates: inner_frailties,
            outer_frailty_estimates: outer_frailties,
            log_likelihood_history: ll_history,
            converged,
            iterations,
            baseline_hazard,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_cluster_infos(clusters: &[usize], events: &[bool]) -> StatsResult<Vec<ClusterInfo>> {
    let max_id = clusters.iter().copied().max().unwrap_or(0);
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); max_id + 1];
    for (i, &c) in clusters.iter().enumerate() {
        buckets[c].push(i);
    }
    Ok(buckets
        .into_iter()
        .enumerate()
        .filter(|(_, indices)| !indices.is_empty())
        .map(|(id, indices)| ClusterInfo::new(id, indices, events))
        .collect())
}

fn compute_cluster_cumulative_hazard(
    cluster_infos: &[ClusterInfo],
    exp_xb: &[f64],
    combined_frailties: &[f64],
    times: &[f64],
    events: &[bool],
    order: &[usize],
) -> Vec<f64> {
    let n = times.len();

    // Compute Breslow increments
    let mut risk_sum = 0.0_f64;
    for j in 0..n {
        risk_sum += combined_frailties[j] * exp_xb[j];
    }

    let mut cum_h0_at = vec![0.0_f64; n];
    let mut cum_h0 = 0.0_f64;
    let mut risk_ptr = 0_usize;
    for &idx in order {
        while risk_ptr < order.len() && times[order[risk_ptr]] < times[idx] - 1e-15 {
            let rem = order[risk_ptr];
            risk_sum -= combined_frailties[rem] * exp_xb[rem];
            risk_ptr += 1;
        }
        if events[idx] && risk_sum > 1e-30 {
            cum_h0 += 1.0 / risk_sum;
        }
        cum_h0_at[idx] = cum_h0;
    }

    cluster_infos
        .iter()
        .map(|ci| {
            ci.subject_indices
                .iter()
                .map(|&j| exp_xb[j] * cum_h0_at[j])
                .sum::<f64>()
        })
        .collect()
}

fn update_frailties_gamma(
    frailties: &mut [f64],
    cluster_infos: &[ClusterInfo],
    cum_hazard: &[f64],
    theta: f64,
) {
    let inv_theta = 1.0 / theta.max(1e-15);
    for (k, ci) in cluster_infos.iter().enumerate() {
        let d_i = ci.n_events as f64;
        let h_i = cum_hazard[k];
        frailties[k] = (d_i + inv_theta) / (h_i + inv_theta);
    }
}

fn moment_variance(frailties: &[f64]) -> f64 {
    let k = frailties.len() as f64;
    if k < 1.0 {
        return 1.0;
    }
    let mean = frailties.iter().sum::<f64>() / k;
    frailties.iter().map(|&u| (u - mean).powi(2)).sum::<f64>() / k
}

fn update_beta_newton(
    beta: &mut [f64],
    times: &[f64],
    events: &[bool],
    covariates: &[&[f64]],
    combined_frailties: &[f64],
    order: &[usize],
) {
    let p = beta.len();
    let n = times.len();
    if p == 0 {
        return;
    }

    let exp_xb: Vec<f64> = (0..n)
        .map(|j| {
            let mut lin = 0.0_f64;
            for col in 0..p {
                lin += covariates[j][col] * beta[col];
            }
            lin.exp()
        })
        .collect();

    let mut gradient = vec![0.0_f64; p];
    let mut hessian_diag = vec![0.0_f64; p];

    let mut s0 = 0.0_f64;
    let mut s1 = vec![0.0_f64; p];

    for j in 0..n {
        let w = combined_frailties[j] * exp_xb[j];
        s0 += w;
        for col in 0..p {
            s1[col] += w * covariates[j][col];
        }
    }

    let mut risk_ptr = 0_usize;
    for &idx in order {
        while risk_ptr < order.len() && times[order[risk_ptr]] < times[idx] - 1e-15 {
            let rem = order[risk_ptr];
            let w = combined_frailties[rem] * exp_xb[rem];
            s0 -= w;
            for col in 0..p {
                s1[col] -= w * covariates[rem][col];
            }
            risk_ptr += 1;
        }
        if events[idx] && s0 > 1e-30 {
            for col in 0..p {
                let mean_x = s1[col] / s0;
                gradient[col] += covariates[idx][col] - mean_x;
                // Approximate diagonal Hessian
                hessian_diag[col] -= 1.0; // simplified
            }
        }
    }

    // Simple diagonal Newton step with damping
    let step_size = 0.3_f64;
    let ridge = 1e-3;
    for col in 0..p {
        let h = hessian_diag[col] - ridge;
        if h.abs() > 1e-30 {
            beta[col] += step_size * gradient[col] / (-h);
        }
    }
}

fn compute_baseline_hazard(
    times: &[f64],
    events: &[bool],
    exp_xb: &[f64],
    combined_frailties: &[f64],
    order: &[usize],
) -> Vec<(f64, f64)> {
    let n = times.len();
    let mut risk_sum = 0.0_f64;
    for j in 0..n {
        risk_sum += combined_frailties[j] * exp_xb[j];
    }

    let mut baseline = Vec::new();
    let mut cum_h0 = 0.0_f64;
    let mut risk_ptr = 0_usize;
    for &idx in order {
        while risk_ptr < order.len() && times[order[risk_ptr]] < times[idx] - 1e-15 {
            let rem = order[risk_ptr];
            risk_sum -= combined_frailties[rem] * exp_xb[rem];
            risk_ptr += 1;
        }
        if events[idx] && risk_sum > 1e-30 {
            cum_h0 += 1.0 / risk_sum;
            baseline.push((times[idx], cum_h0));
        }
    }
    baseline.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-15);
    baseline
}

fn lgamma(x: f64) -> f64 {
    let c = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_10,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_10,
        -176.615_029_162_140_60,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x - 1.0;
    let mut ser = c[0];
    for (i, &ci) in c[1..].iter().enumerate() {
        ser += ci / (x + i as f64 + 1.0);
    }
    let tmp = x + 7.5;
    0.5 * std::f64::consts::TAU.ln() + (x + 0.5) * tmp.ln() - tmp + ser.ln()
}

fn nested_log_likelihood(
    times: &[f64],
    events: &[bool],
    covariates: &[&[f64]],
    beta: &[f64],
    inner_frailties: &[f64],
    outer_frailties: &[f64],
    subj_to_inner: &[usize],
    subj_to_outer: &[usize],
    order: &[usize],
    theta_inner: f64,
    theta_outer: f64,
    inner_infos: &[ClusterInfo],
    outer_infos: &[ClusterInfo],
) -> f64 {
    let n = times.len();
    let p = beta.len();

    let exp_xb: Vec<f64> = (0..n)
        .map(|j| {
            let mut lin = 0.0_f64;
            for col in 0..p {
                lin += covariates[j][col] * beta[col];
            }
            lin.exp()
        })
        .collect();

    let combined: Vec<f64> = (0..n)
        .map(|j| outer_frailties[subj_to_outer[j]] * inner_frailties[subj_to_inner[j]])
        .collect();

    // Partial likelihood
    let mut ll = 0.0_f64;
    let mut s0 = 0.0_f64;
    for j in 0..n {
        s0 += combined[j] * exp_xb[j];
    }

    let mut risk_ptr = 0_usize;
    for &idx in order {
        while risk_ptr < order.len() && times[order[risk_ptr]] < times[idx] - 1e-15 {
            let rem = order[risk_ptr];
            s0 -= combined[rem] * exp_xb[rem];
            risk_ptr += 1;
        }
        if events[idx] {
            let u = combined[idx].max(1e-30);
            let mut xb = 0.0_f64;
            for col in 0..p {
                xb += covariates[idx][col] * beta[col];
            }
            ll += u.ln() + xb - s0.max(1e-30).ln();
        }
    }

    // Gamma prior contributions for inner frailties
    let inv_ti = 1.0 / theta_inner.max(1e-15);
    for (k, _ci) in inner_infos.iter().enumerate() {
        let u = inner_frailties[k].max(1e-30);
        ll += (inv_ti - 1.0) * u.ln() - u * inv_ti - lgamma(inv_ti) + inv_ti * inv_ti.ln();
    }

    // Gamma prior contributions for outer frailties
    let inv_to = 1.0 / theta_outer.max(1e-15);
    for (k, _ci) in outer_infos.iter().enumerate() {
        let u = outer_frailties[k].max(1e-30);
        ll += (inv_to - 1.0) * u.ln() - u * inv_to - lgamma(inv_to) + inv_to * inv_to.ln();
    }

    ll
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_nested_data(
        n_outer: usize,
        n_inner_per_outer: usize,
        n_per_inner: usize,
    ) -> (Vec<f64>, Vec<bool>, Vec<Vec<f64>>, Vec<usize>, Vec<usize>) {
        let mut times = Vec::new();
        let mut events = Vec::new();
        let mut covariates = Vec::new();
        let mut inner_clusters = Vec::new();
        let mut outer_clusters = Vec::new();

        let mut inner_id = 0_usize;
        for outer in 0..n_outer {
            let outer_effect = 1.0 + 0.3 * (outer as f64 * 1.5).sin();
            for inner_offset in 0..n_inner_per_outer {
                let inner_effect = 1.0 + 0.2 * (inner_id as f64 * 2.3).sin();
                for subj in 0..n_per_inner {
                    let x = ((inner_id * n_per_inner + subj) as f64 * 0.2).sin();
                    let rate = outer_effect * inner_effect * (0.3 * x).exp();
                    let pseudo_rand = 0.5
                        + 0.4
                            * ((outer * 11 + inner_offset * 7 + subj * 3) as f64 * 1.618)
                                .sin()
                                .abs();
                    let t = pseudo_rand / rate.max(0.01);
                    let event = (outer + inner_offset + subj) % 3 != 0;

                    times.push(t.max(0.01));
                    events.push(event);
                    covariates.push(vec![x]);
                    inner_clusters.push(inner_id);
                    outer_clusters.push(outer);
                }
                inner_id += 1;
            }
        }

        (times, events, covariates, inner_clusters, outer_clusters)
    }

    #[test]
    fn test_nested_frailty_basic() {
        let (times, events, cov_owned, inner_cl, outer_cl) = generate_nested_data(3, 3, 10);
        let covariates: Vec<&[f64]> = cov_owned.iter().map(|v| v.as_slice()).collect();

        let model = NestedFrailtyModel::new(FrailtyConfig {
            max_iterations: 50,
            ..FrailtyConfig::default()
        });
        let result = model
            .fit(&times, &events, &covariates, &inner_cl, &outer_cl)
            .expect("nested fit should succeed");

        assert_eq!(result.outer_frailty_estimates.len(), 3);
        assert_eq!(result.inner_frailty_estimates.len(), 9);
        assert!(result.inner_variance > 0.0);
        assert!(result.outer_variance > 0.0);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_nested_two_variance_components() {
        let (times, events, cov_owned, inner_cl, outer_cl) = generate_nested_data(4, 2, 8);
        let covariates: Vec<&[f64]> = cov_owned.iter().map(|v| v.as_slice()).collect();

        let model = NestedFrailtyModel::new(FrailtyConfig {
            max_iterations: 100,
            ..FrailtyConfig::default()
        });
        let result = model
            .fit(&times, &events, &covariates, &inner_cl, &outer_cl)
            .expect("nested fit should succeed");

        // Both variance components should be estimated
        assert!(
            result.inner_variance > 0.0,
            "Inner variance should be positive"
        );
        assert!(
            result.outer_variance > 0.0,
            "Outer variance should be positive"
        );
    }

    #[test]
    fn test_nested_empty_data_error() {
        let model = NestedFrailtyModel::new(FrailtyConfig::default());
        let result = model.fit(&[], &[], &[], &[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_nested_single_outer_cluster_error() {
        let model = NestedFrailtyModel::new(FrailtyConfig::default());
        // All subjects in same outer cluster
        let result = model.fit(
            &[1.0, 2.0, 3.0, 4.0],
            &[true, true, true, false],
            &[&[0.1][..], &[0.2][..], &[0.3][..], &[0.4][..]],
            &[0, 1, 2, 3],
            &[0, 0, 0, 0], // single outer
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_nested_baseline_hazard() {
        let (times, events, cov_owned, inner_cl, outer_cl) = generate_nested_data(3, 2, 8);
        let covariates: Vec<&[f64]> = cov_owned.iter().map(|v| v.as_slice()).collect();

        let model = NestedFrailtyModel::new(FrailtyConfig {
            max_iterations: 30,
            ..FrailtyConfig::default()
        });
        let result = model
            .fit(&times, &events, &covariates, &inner_cl, &outer_cl)
            .expect("nested fit should succeed");

        assert!(!result.baseline_hazard.is_empty());
        // Non-decreasing
        for i in 1..result.baseline_hazard.len() {
            assert!(result.baseline_hazard[i].1 >= result.baseline_hazard[i - 1].1 - 1e-10);
        }
    }
}
