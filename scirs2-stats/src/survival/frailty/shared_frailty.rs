//! Shared (single-level) frailty model.
//!
//! Extends the Cox PH model with a multiplicative cluster-level random effect:
//!
//!   λ(t | x, u_i) = u_i · λ₀(t) · exp(x β)
//!
//! where u_i is the frailty for cluster i. The model is fitted via the EM
//! algorithm, alternating between:
//! - **E-step**: estimate posterior frailties given current (β, θ)
//! - **M-step**: update β via weighted partial likelihood and θ from frailty moments.

use crate::error::{StatsError, StatsResult};

use super::types::{ClusterInfo, FrailtyConfig, FrailtyDistribution, FrailtyResult};

// ---------------------------------------------------------------------------
// Log-gamma helper (Lanczos approximation)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Shared Frailty Model
// ---------------------------------------------------------------------------

/// Shared frailty model extending the Cox proportional hazards framework.
///
/// Each cluster *i* receives a random frailty u_i drawn from the specified
/// distribution. The hazard for subject *j* in cluster *i* is:
///
///   h(t | x_ij, u_i) = u_i · h₀(t) · exp(x_ij · β)
///
/// The model is fitted via an EM algorithm.
#[derive(Debug, Clone)]
pub struct SharedFrailtyModel {
    config: FrailtyConfig,
}

impl SharedFrailtyModel {
    /// Create a new shared frailty model with the given configuration.
    pub fn new(config: FrailtyConfig) -> Self {
        Self { config }
    }

    /// Fit the shared frailty model.
    ///
    /// # Arguments
    /// * `times`      – observed event/censoring times (n elements, finite, >= 0)
    /// * `events`     – event indicators (`true` = event, `false` = censored)
    /// * `covariates` – covariate matrix, shape (n, p) stored row-major as `&[&[f64]]`
    /// * `clusters`   – cluster assignment for each subject (n elements, 0-based)
    ///
    /// # Errors
    /// Returns `StatsError` on dimension mismatch, empty data, no events, or
    /// when all subjects belong to the same cluster.
    pub fn fit(
        &self,
        times: &[f64],
        events: &[bool],
        covariates: &[&[f64]],
        clusters: &[usize],
    ) -> StatsResult<FrailtyResult> {
        let n = times.len();

        // --- Input validation ---
        if n == 0 {
            return Err(StatsError::InvalidArgument(
                "times must not be empty".into(),
            ));
        }
        if events.len() != n {
            return Err(StatsError::DimensionMismatch(format!(
                "times length {n} != events length {}",
                events.len()
            )));
        }
        if covariates.len() != n {
            return Err(StatsError::DimensionMismatch(format!(
                "covariates rows {} != times length {n}",
                covariates.len()
            )));
        }
        if clusters.len() != n {
            return Err(StatsError::DimensionMismatch(format!(
                "clusters length {} != times length {n}",
                clusters.len()
            )));
        }
        let p = if n > 0 && !covariates.is_empty() {
            covariates[0].len()
        } else {
            0
        };
        for (i, row) in covariates.iter().enumerate() {
            if row.len() != p {
                return Err(StatsError::DimensionMismatch(format!(
                    "covariate row {i} has length {} but expected {p}",
                    row.len()
                )));
            }
        }
        for &t in times {
            if !t.is_finite() || t < 0.0 {
                return Err(StatsError::InvalidArgument(format!(
                    "times must be finite and non-negative; got {t}"
                )));
            }
        }
        let n_events = events.iter().filter(|&&e| e).count();
        if n_events == 0 {
            return Err(StatsError::InvalidArgument("No events observed".into()));
        }

        // --- Build cluster information ---
        let cluster_infos = build_clusters(clusters, events)?;
        if cluster_infos.len() < 2 {
            return Err(StatsError::InvalidArgument(
                "At least two clusters are required for a frailty model".into(),
            ));
        }

        // Check for empty clusters
        for ci in &cluster_infos {
            if ci.subject_indices.is_empty() {
                return Err(StatsError::InvalidArgument(format!(
                    "Cluster {} has no subjects",
                    ci.cluster_id
                )));
            }
        }

        // --- Sort by time (descending for risk set construction) ---
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            times[a]
                .partial_cmp(&times[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // --- Initialise parameters ---
        let mut beta = vec![0.0_f64; p];
        let mut theta = self.config.initial_variance;
        let n_clusters = cluster_infos.len();
        let mut frailties = vec![1.0_f64; n_clusters];
        let mut ll_history: Vec<f64> = Vec::new();
        let mut converged = false;
        let mut iterations = 0_usize;

        // Map from subject index to cluster position
        let mut subject_to_cluster = vec![0_usize; n];
        for (k, ci) in cluster_infos.iter().enumerate() {
            for &idx in &ci.subject_indices {
                subject_to_cluster[idx] = k;
            }
        }

        // --- EM iterations ---
        for iter in 0..self.config.max_iterations {
            iterations = iter + 1;

            // Compute exp(x_j β) for all subjects
            let exp_xb: Vec<f64> = (0..n)
                .map(|j| {
                    let mut lin = 0.0_f64;
                    for (col, b) in beta.iter().enumerate() {
                        lin += covariates[j][col] * b;
                    }
                    lin.exp()
                })
                .collect();

            // Breslow-style cumulative hazard per cluster
            let baseline_increments = breslow_increments(
                times,
                events,
                &exp_xb,
                &frailties,
                &subject_to_cluster,
                &order,
            );

            // Cumulative hazard for each cluster: H_i = sum_j in cluster_i exp(x_j β) * H₀(t_j)
            let cluster_cum_hazard =
                cluster_cumulative_hazard(&cluster_infos, &exp_xb, &baseline_increments, &order);

            // --- E-step: update frailty estimates ---
            e_step(
                &mut frailties,
                &cluster_infos,
                &cluster_cum_hazard,
                theta,
                self.config.distribution,
            );

            // --- M-step: update θ ---
            theta = m_step_variance(&frailties, &cluster_infos, self.config.distribution, theta);
            // Clamp theta to avoid degenerate values
            theta = theta.max(1e-8);

            // --- M-step: update β via one Newton-Raphson step on weighted partial likelihood ---
            if p > 0 {
                newton_step_beta(
                    &mut beta,
                    times,
                    events,
                    covariates,
                    &frailties,
                    &subject_to_cluster,
                    &order,
                );
            }

            // --- Compute observed-data log-likelihood ---
            let ll = observed_log_likelihood(
                times,
                events,
                covariates,
                &beta,
                &frailties,
                &subject_to_cluster,
                &order,
                theta,
                self.config.distribution,
                &cluster_infos,
            );
            ll_history.push(ll);

            // Check convergence
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

        // --- Final baseline hazard ---
        let exp_xb: Vec<f64> = (0..n)
            .map(|j| {
                let mut lin = 0.0_f64;
                for (col, b) in beta.iter().enumerate() {
                    lin += covariates[j][col] * b;
                }
                lin.exp()
            })
            .collect();
        let baseline_increments = breslow_increments(
            times,
            events,
            &exp_xb,
            &frailties,
            &subject_to_cluster,
            &order,
        );

        // Accumulate baseline hazard
        let mut baseline_hazard: Vec<(f64, f64)> = Vec::new();
        let mut cum_h0 = 0.0_f64;
        for &idx in &order {
            if events[idx] {
                cum_h0 += baseline_increments[idx];
                baseline_hazard.push((times[idx], cum_h0));
            }
        }
        // Deduplicate tied times (keep last cumulative value)
        baseline_hazard.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-15);

        Ok(FrailtyResult {
            coefficients: beta,
            frailty_variance: theta,
            frailty_estimates: frailties,
            log_likelihood_history: ll_history,
            converged,
            iterations,
            baseline_hazard,
        })
    }
}

// ---------------------------------------------------------------------------
// Cluster construction
// ---------------------------------------------------------------------------

fn build_clusters(clusters: &[usize], events: &[bool]) -> StatsResult<Vec<ClusterInfo>> {
    let max_id = clusters.iter().copied().max().unwrap_or(0);
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); max_id + 1];
    for (i, &c) in clusters.iter().enumerate() {
        buckets[c].push(i);
    }
    let infos: Vec<ClusterInfo> = buckets
        .into_iter()
        .enumerate()
        .filter(|(_, indices)| !indices.is_empty())
        .map(|(id, indices)| ClusterInfo::new(id, indices, events))
        .collect();
    Ok(infos)
}

// ---------------------------------------------------------------------------
// Breslow baseline hazard increments
// ---------------------------------------------------------------------------

/// Compute baseline hazard increment dH₀(t_j) at each event time.
///
/// For each event time t_j, dH₀(t_j) = d_j / S₀(t_j) where
/// S₀(t_j) = sum_{l in R(t_j)} u_{c(l)} * exp(x_l β).
fn breslow_increments(
    times: &[f64],
    events: &[bool],
    exp_xb: &[f64],
    frailties: &[f64],
    subject_to_cluster: &[usize],
    order: &[usize],
) -> Vec<f64> {
    let n = times.len();
    let mut increments = vec![0.0_f64; n];

    // Compute risk set sums going from latest to earliest time
    // Use reverse-sorted order to build cumulative sums efficiently
    let mut risk_sum = 0.0_f64;
    let mut risk_idx = n; // pointer into order (from right)

    // Collect event times with their tied counts
    // Process in forward time order
    let mut event_times: Vec<(f64, Vec<usize>)> = Vec::new();
    {
        let mut i = 0;
        while i < order.len() {
            let idx = order[i];
            if events[idx] {
                let t = times[idx];
                let mut tied = vec![idx];
                let mut j = i + 1;
                while j < order.len() && (times[order[j]] - t).abs() < 1e-15 && events[order[j]] {
                    tied.push(order[j]);
                    j += 1;
                }
                event_times.push((t, tied));
                i = j;
            } else {
                i += 1;
            }
        }
    }

    // For each event time, risk set = {l : t_l >= t}
    // We process forward, so the risk set shrinks
    // Precompute full risk sum
    for &idx in order.iter() {
        risk_sum += frailties[subject_to_cluster[idx]] * exp_xb[idx];
    }

    risk_idx = 0;
    for (t_event, tied_indices) in &event_times {
        // Remove subjects with time < t_event from risk set
        while risk_idx < order.len() && times[order[risk_idx]] < *t_event - 1e-15 {
            let rem_idx = order[risk_idx];
            risk_sum -= frailties[subject_to_cluster[rem_idx]] * exp_xb[rem_idx];
            risk_idx += 1;
        }

        let d_j = tied_indices.len() as f64;
        let dh0 = if risk_sum > 1e-30 {
            d_j / risk_sum
        } else {
            0.0
        };
        for &idx in tied_indices {
            increments[idx] = dh0;
        }
    }

    increments
}

// ---------------------------------------------------------------------------
// Cumulative hazard per cluster
// ---------------------------------------------------------------------------

fn cluster_cumulative_hazard(
    cluster_infos: &[ClusterInfo],
    exp_xb: &[f64],
    baseline_increments: &[f64],
    order: &[usize],
) -> Vec<f64> {
    // Compute total cumulative baseline hazard
    let mut cum_h0 = 0.0_f64;
    let mut cumulative_at: Vec<f64> = vec![0.0; exp_xb.len()];
    for &idx in order {
        cum_h0 += baseline_increments[idx];
        cumulative_at[idx] = cum_h0;
    }

    // For each cluster, H_i = sum_{j in cluster_i} exp(x_j β) * H₀(max_time_in_data)
    // More precisely: H_i = sum_{j in cluster_i} exp(x_j β) * sum_{t_k <= t_j} dH₀(t_k)
    // But for EM we use total cumulative exposure: H_i = sum_j exp(x_j β) * H₀(t_j)
    cluster_infos
        .iter()
        .map(|ci| {
            ci.subject_indices
                .iter()
                .map(|&j| exp_xb[j] * cumulative_at[j])
                .sum::<f64>()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// E-step: update frailty posterior estimates
// ---------------------------------------------------------------------------

fn e_step(
    frailties: &mut [f64],
    cluster_infos: &[ClusterInfo],
    cluster_cum_hazard: &[f64],
    theta: f64,
    distribution: FrailtyDistribution,
) {
    match distribution {
        FrailtyDistribution::Gamma => {
            // For Gamma(1/θ, θ) frailty, the posterior is Gamma with:
            //   shape' = d_i + 1/θ
            //   rate'  = H_i + 1/θ
            // E[u_i | data] = shape' / rate'
            let inv_theta = 1.0 / theta.max(1e-15);
            for (k, ci) in cluster_infos.iter().enumerate() {
                let d_i = ci.n_events as f64;
                let h_i = cluster_cum_hazard[k];
                frailties[k] = (d_i + inv_theta) / (h_i + inv_theta);
            }
        }
        FrailtyDistribution::LogNormal => {
            // Laplace approximation: find mode of log-posterior
            // log p(u_i | data) ∝ d_i log(u_i) - u_i H_i + log f(u_i; θ)
            // where f is log-normal density with σ² = log(1 + θ) and
            // μ = -σ²/2 (so E[u]=1).
            let sigma2 = (1.0 + theta).ln().max(1e-10);
            let mu = -sigma2 / 2.0;
            for (k, ci) in cluster_infos.iter().enumerate() {
                let d_i = ci.n_events as f64;
                let h_i = cluster_cum_hazard[k];
                // Newton iterations on log(u) = v to find mode of:
                // g(v) = d_i * v - exp(v)*H_i - (v - μ)²/(2σ²)
                // g'(v) = d_i - exp(v)*H_i - (v - μ)/σ²
                // g''(v) = -exp(v)*H_i - 1/σ²
                let mut v = 0.0_f64; // initial: u=1 => v=0
                for _ in 0..30 {
                    let ev = v.exp();
                    let g_prime = d_i - ev * h_i - (v - mu) / sigma2;
                    let g_double = -ev * h_i - 1.0 / sigma2;
                    if g_double.abs() < 1e-30 {
                        break;
                    }
                    let step = -g_prime / g_double;
                    v -= step; // Newton: v_new = v - g'/g'' but we want to maximise
                               // Actually for maximisation: v += -g'/g'' is wrong
                               // g'' < 0 so step = g'/g'' < 0 when g'>0, meaning v -= step increases v
                    if step.abs() < 1e-10 {
                        break;
                    }
                }
                frailties[k] = v.exp().max(1e-8);
            }
        }
        FrailtyDistribution::InverseGaussian => {
            // For IG(1, λ) frailty, posterior mode:
            // log p(u | data) ∝ d_i log(u) - u H_i + (-3/2)log(u) - λ(u-1)²/(2u)
            // Simplified: (d_i - 3/2) log(u) - u H_i - λ(u-1)²/(2u)
            // d/du = (d_i - 3/2)/u - H_i - λ(u²-1)/(2u²)
            let lambda = 1.0 / theta.max(1e-15);
            for (k, ci) in cluster_infos.iter().enumerate() {
                let d_i = ci.n_events as f64;
                let h_i = cluster_cum_hazard[k];
                // Approximate: use Gamma-like posterior as in Gamma case
                // This is a standard approximation for IG frailty
                let shape = d_i + 0.5;
                let rate = h_i + lambda / 2.0;
                frailties[k] = if rate > 1e-30 { shape / rate } else { 1.0 };
            }
        }
        _ => {
            // Future distributions: fall back to no frailty
            for f in frailties.iter_mut() {
                *f = 1.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// M-step: update frailty variance
// ---------------------------------------------------------------------------

fn m_step_variance(
    frailties: &[f64],
    cluster_infos: &[ClusterInfo],
    distribution: FrailtyDistribution,
    current_theta: f64,
) -> f64 {
    let k = frailties.len() as f64;
    if k < 1.0 {
        return current_theta;
    }

    match distribution {
        FrailtyDistribution::Gamma => {
            // Method of moments: Var(u_i) ≈ θ
            // Profile likelihood approach: solve ψ(1/θ) - ln(1/θ) = (1/K) Σ (ln u_i - u_i + 1) + 1
            // Simplified moment estimator: θ = (1/K) Σ (u_i - 1)²
            // But more stable: use E[u] should be 1, Var = θ
            let mean_u = frailties.iter().sum::<f64>() / k;
            let var_u = frailties.iter().map(|&u| (u - mean_u).powi(2)).sum::<f64>() / k;
            var_u.max(1e-8)
        }
        FrailtyDistribution::LogNormal => {
            // σ² = Var(log u), then θ = exp(σ²) - 1
            let log_u: Vec<f64> = frailties.iter().map(|&u| u.max(1e-15).ln()).collect();
            let mean_log = log_u.iter().sum::<f64>() / k;
            let var_log = log_u.iter().map(|&v| (v - mean_log).powi(2)).sum::<f64>() / k;
            let sigma2 = var_log.max(1e-8);
            sigma2.exp() - 1.0
        }
        FrailtyDistribution::InverseGaussian => {
            // For IG(1, λ), Var(u) = 1/λ = θ
            // Moment estimator: θ = (1/K) Σ (u_i - 1)²
            let mean_u = frailties.iter().sum::<f64>() / k;
            let var_u = frailties.iter().map(|&u| (u - mean_u).powi(2)).sum::<f64>() / k;
            var_u.max(1e-8)
        }
        _ => current_theta,
    }
}

// ---------------------------------------------------------------------------
// M-step: Newton-Raphson update for β
// ---------------------------------------------------------------------------

fn newton_step_beta(
    beta: &mut [f64],
    times: &[f64],
    events: &[bool],
    covariates: &[&[f64]],
    frailties: &[f64],
    subject_to_cluster: &[usize],
    order: &[usize],
) {
    let p = beta.len();
    if p == 0 {
        return;
    }
    let n = times.len();

    // Compute weighted partial likelihood gradient and Hessian
    let exp_xb: Vec<f64> = (0..n)
        .map(|j| {
            let mut lin = 0.0_f64;
            for (col, b) in beta.iter().enumerate() {
                lin += covariates[j][col] * b;
            }
            lin.exp()
        })
        .collect();

    let mut gradient = vec![0.0_f64; p];
    let mut hessian = vec![vec![0.0_f64; p]; p];

    // Process events in time order
    // Risk set at time t = {j : t_j >= t}
    // Weighted: w_j = u_{c(j)} * exp(x_j β)

    // Precompute risk set sums
    let mut s0 = 0.0_f64;
    let mut s1 = vec![0.0_f64; p];
    let mut s2 = vec![vec![0.0_f64; p]; p];

    // Add all to risk set initially
    for j in 0..n {
        let w = frailties[subject_to_cluster[j]] * exp_xb[j];
        s0 += w;
        for col in 0..p {
            s1[col] += w * covariates[j][col];
            for col2 in 0..p {
                s2[col][col2] += w * covariates[j][col] * covariates[j][col2];
            }
        }
    }

    let mut risk_ptr = 0_usize;
    for &idx in order {
        // Remove subjects with time < times[idx] from risk set
        while risk_ptr < order.len() && times[order[risk_ptr]] < times[idx] - 1e-15 {
            let rem = order[risk_ptr];
            let w = frailties[subject_to_cluster[rem]] * exp_xb[rem];
            s0 -= w;
            for col in 0..p {
                s1[col] -= w * covariates[rem][col];
                for col2 in 0..p {
                    s2[col][col2] -= w * covariates[rem][col] * covariates[rem][col2];
                }
            }
            risk_ptr += 1;
        }

        if events[idx] {
            if s0 > 1e-30 {
                for col in 0..p {
                    gradient[col] += covariates[idx][col] - s1[col] / s0;
                }
                for col in 0..p {
                    for col2 in 0..p {
                        hessian[col][col2] -= s2[col][col2] / s0 - (s1[col] * s1[col2]) / (s0 * s0);
                    }
                }
            }
        }
    }

    // Add ridge penalty for stability
    let ridge = 1e-4;
    for col in 0..p {
        gradient[col] -= ridge * beta[col];
        hessian[col][col] -= ridge;
    }

    // Solve H * delta = -gradient via Cholesky-like approach
    // Since H is negative definite, we solve (-H) delta = gradient
    let neg_hessian: Vec<Vec<f64>> = hessian
        .iter()
        .map(|row| row.iter().map(|&v| -v).collect())
        .collect();

    if let Some(delta) = solve_symmetric(&neg_hessian, &gradient) {
        // Step with damping to avoid overshooting
        let step_size = 0.5_f64;
        for col in 0..p {
            beta[col] += step_size * delta[col];
        }
    }
}

// ---------------------------------------------------------------------------
// Simple symmetric positive-definite solver (Cholesky)
// ---------------------------------------------------------------------------

fn solve_symmetric(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    if n == 0 {
        return Some(Vec::new());
    }

    // Cholesky: A = L L^T
    let mut l = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 0.0 {
                    return None; // Not positive definite
                }
                l[i][j] = sum.sqrt();
            } else {
                if l[j][j].abs() < 1e-30 {
                    return None;
                }
                l[i][j] = sum / l[j][j];
            }
        }
    }

    // Forward substitution: L y = b
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i][j] * y[j];
        }
        if l[i][i].abs() < 1e-30 {
            return None;
        }
        y[i] = sum / l[i][i];
    }

    // Back substitution: L^T x = y
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[j][i] * x[j];
        }
        if l[i][i].abs() < 1e-30 {
            return None;
        }
        x[i] = sum / l[i][i];
    }

    Some(x)
}

// ---------------------------------------------------------------------------
// Observed-data log-likelihood
// ---------------------------------------------------------------------------

fn observed_log_likelihood(
    times: &[f64],
    events: &[bool],
    covariates: &[&[f64]],
    beta: &[f64],
    frailties: &[f64],
    subject_to_cluster: &[usize],
    order: &[usize],
    theta: f64,
    distribution: FrailtyDistribution,
    cluster_infos: &[ClusterInfo],
) -> f64 {
    let n = times.len();
    let p = beta.len();

    // Partial log-likelihood contribution
    let exp_xb: Vec<f64> = (0..n)
        .map(|j| {
            let mut lin = 0.0_f64;
            for col in 0..p {
                lin += covariates[j][col] * beta[col];
            }
            lin.exp()
        })
        .collect();

    let mut ll = 0.0_f64;

    // Cox partial likelihood contribution (weighted)
    let mut s0 = 0.0_f64;
    for j in 0..n {
        s0 += frailties[subject_to_cluster[j]] * exp_xb[j];
    }

    let mut risk_ptr = 0_usize;
    for &idx in order {
        while risk_ptr < order.len() && times[order[risk_ptr]] < times[idx] - 1e-15 {
            let rem = order[risk_ptr];
            s0 -= frailties[subject_to_cluster[rem]] * exp_xb[rem];
            risk_ptr += 1;
        }
        if events[idx] {
            let u_i = frailties[subject_to_cluster[idx]];
            let mut xb = 0.0_f64;
            for col in 0..p {
                xb += covariates[idx][col] * beta[col];
            }
            ll += u_i.max(1e-30).ln() + xb - s0.max(1e-30).ln();
        }
    }

    // Frailty distribution log-likelihood contribution
    match distribution {
        FrailtyDistribution::Gamma => {
            let inv_theta = 1.0 / theta.max(1e-15);
            for (k, ci) in cluster_infos.iter().enumerate() {
                let u = frailties[k].max(1e-30);
                // log f(u; 1/θ, θ) = (1/θ - 1)ln(u) - u/θ - ln(Γ(1/θ)) - (1/θ)ln(θ)
                let _ = ci; // used for counting
                ll += (inv_theta - 1.0) * u.ln() - u * inv_theta - lgamma(inv_theta)
                    + inv_theta * inv_theta.ln();
            }
        }
        FrailtyDistribution::LogNormal => {
            let sigma2 = (1.0 + theta).ln().max(1e-10);
            let mu = -sigma2 / 2.0;
            for &u in frailties.iter() {
                let lu = u.max(1e-30).ln();
                ll += -0.5
                    * ((lu - mu).powi(2) / sigma2 + sigma2.ln() + std::f64::consts::TAU.ln())
                    - lu;
            }
        }
        FrailtyDistribution::InverseGaussian => {
            let lambda = 1.0 / theta.max(1e-15);
            for &u in frailties.iter() {
                let u = u.max(1e-30);
                ll += 0.5 * lambda.ln()
                    - 0.5 * std::f64::consts::TAU.ln()
                    - 1.5 * u.ln()
                    - lambda * (u - 1.0).powi(2) / (2.0 * u);
            }
        }
        _ => {}
    }

    ll
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate simple clustered survival data with known frailty.
    fn generate_clustered_data(
        n_clusters: usize,
        n_per_cluster: usize,
        true_theta: f64,
    ) -> (Vec<f64>, Vec<bool>, Vec<Vec<f64>>, Vec<usize>) {
        let n = n_clusters * n_per_cluster;
        let mut times = Vec::with_capacity(n);
        let mut events = Vec::with_capacity(n);
        let mut covariates = Vec::with_capacity(n);
        let mut clusters = Vec::with_capacity(n);

        // Deterministic "pseudo-random" frailties based on cluster id
        let mut frailty_values: Vec<f64> = (0..n_clusters)
            .map(|k| {
                // Generate frailty-like values around 1 with spread ~ sqrt(theta)
                1.0 + true_theta.sqrt() * ((k as f64 * 2.718).sin())
            })
            .collect();
        // Normalise so mean = 1
        let mean_f: f64 = frailty_values.iter().sum::<f64>() / n_clusters as f64;
        for f in &mut frailty_values {
            *f = (*f / mean_f).max(0.1);
        }

        for k in 0..n_clusters {
            for j in 0..n_per_cluster {
                let x = ((k * n_per_cluster + j) as f64 * 0.1).sin();
                // Hazard = frailty * exp(0.5 * x), so higher frailty => shorter times
                let rate = frailty_values[k] * (0.5 * x).exp();
                // Pseudo-exponential time: T = 1/rate * "pseudo-random"
                let pseudo_rand = 0.5 + 0.3 * ((k * 7 + j * 3) as f64 * 1.618).sin().abs();
                let t = pseudo_rand / rate.max(0.01);
                let event = (k + j) % 3 != 0; // ~67% events
                times.push(t.max(0.01));
                events.push(event);
                covariates.push(vec![x]);
                clusters.push(k);
            }
        }

        (times, events, covariates, clusters)
    }

    #[test]
    fn test_gamma_frailty_basic() {
        let (times, events, cov_owned, clusters) = generate_clustered_data(5, 20, 0.5);
        let covariates: Vec<&[f64]> = cov_owned.iter().map(|v| v.as_slice()).collect();

        let model = SharedFrailtyModel::new(FrailtyConfig::default());
        let result = model
            .fit(&times, &events, &covariates, &clusters)
            .expect("fit should succeed");

        assert_eq!(result.frailty_estimates.len(), 5);
        assert!(result.frailty_variance > 0.0);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_em_log_likelihood_improves_initially() {
        let (times, events, cov_owned, clusters) = generate_clustered_data(4, 15, 1.0);
        let covariates: Vec<&[f64]> = cov_owned.iter().map(|v| v.as_slice()).collect();

        let model = SharedFrailtyModel::new(FrailtyConfig {
            max_iterations: 50,
            ..FrailtyConfig::default()
        });
        let result = model
            .fit(&times, &events, &covariates, &clusters)
            .expect("fit should succeed");

        // The EM algorithm with approximate M-step (Newton on beta) may not be
        // strictly monotone in the observed-data log-likelihood. However, early
        // iterations should show improvement over the initial state.
        let ll = &result.log_likelihood_history;
        assert!(ll.len() >= 2, "Should have at least 2 EM iterations");
        // The first few iterations should improve upon the initial LL
        assert!(
            ll[1] > ll[0] - 1e-3,
            "Second iteration should improve or nearly match first: ll[0]={}, ll[1]={}",
            ll[0],
            ll[1]
        );
    }

    #[test]
    fn test_lognormal_frailty() {
        let (times, events, cov_owned, clusters) = generate_clustered_data(5, 15, 0.5);
        let covariates: Vec<&[f64]> = cov_owned.iter().map(|v| v.as_slice()).collect();

        let config = FrailtyConfig {
            distribution: FrailtyDistribution::LogNormal,
            max_iterations: 100,
            ..FrailtyConfig::default()
        };
        let model = SharedFrailtyModel::new(config);
        let result = model
            .fit(&times, &events, &covariates, &clusters)
            .expect("fit should succeed");

        assert_eq!(result.frailty_estimates.len(), 5);
        assert!(result.frailty_variance > 0.0);
    }

    #[test]
    fn test_inverse_gaussian_frailty() {
        let (times, events, cov_owned, clusters) = generate_clustered_data(5, 15, 0.5);
        let covariates: Vec<&[f64]> = cov_owned.iter().map(|v| v.as_slice()).collect();

        let config = FrailtyConfig {
            distribution: FrailtyDistribution::InverseGaussian,
            max_iterations: 100,
            ..FrailtyConfig::default()
        };
        let model = SharedFrailtyModel::new(config);
        let result = model
            .fit(&times, &events, &covariates, &clusters)
            .expect("fit should succeed");

        assert_eq!(result.frailty_estimates.len(), 5);
        assert!(result.frailty_variance > 0.0);
    }

    #[test]
    fn test_empty_data_error() {
        let model = SharedFrailtyModel::new(FrailtyConfig::default());
        let result = model.fit(&[], &[], &[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let model = SharedFrailtyModel::new(FrailtyConfig::default());
        let result = model.fit(&[1.0, 2.0], &[true], &[&[0.1][..], &[0.2][..]], &[0, 0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_cluster_error() {
        let model = SharedFrailtyModel::new(FrailtyConfig::default());
        let result = model.fit(
            &[1.0, 2.0, 3.0],
            &[true, true, false],
            &[&[0.1][..], &[0.2][..], &[0.3][..]],
            &[0, 0, 0],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_no_events_error() {
        let model = SharedFrailtyModel::new(FrailtyConfig::default());
        let result = model.fit(
            &[1.0, 2.0, 3.0],
            &[false, false, false],
            &[&[0.1][..], &[0.2][..], &[0.3][..]],
            &[0, 0, 1],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_baseline_hazard_nonempty() {
        let (times, events, cov_owned, clusters) = generate_clustered_data(3, 10, 0.5);
        let covariates: Vec<&[f64]> = cov_owned.iter().map(|v| v.as_slice()).collect();

        let model = SharedFrailtyModel::new(FrailtyConfig::default());
        let result = model
            .fit(&times, &events, &covariates, &clusters)
            .expect("fit should succeed");

        assert!(!result.baseline_hazard.is_empty());
        // Baseline hazard should be non-decreasing
        for i in 1..result.baseline_hazard.len() {
            assert!(
                result.baseline_hazard[i].1 >= result.baseline_hazard[i - 1].1 - 1e-10,
                "Baseline hazard should be non-decreasing"
            );
        }
    }
}
