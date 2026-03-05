//! Comprehensive MCMC diagnostics
//!
//! This module provides a rich set of convergence and efficiency diagnostics for
//! Markov Chain Monte Carlo (MCMC) output analysis.  The implementations follow
//! the modern recommendations of Vehtari, Gelman, Simpson, Carpenter and Burkner
//! (Bayesian Analysis, 2021) wherever applicable.
//!
//! # Diagnostics provided
//!
//! ## Convergence assessment
//! * **R-hat (Gelman-Rubin diagnostic)** -- basic multi-chain convergence via
//!   between-chain and within-chain variance comparison.
//! * **Split R-hat** -- splits each chain in half before computing R-hat so that
//!   within-chain non-stationarity is also detected.
//! * **Rank-normalised split R-hat** -- robust variant following Vehtari et al.
//!   (2021) that is insensitive to heavy tails.
//!
//! ## Sampling efficiency
//! * **Effective sample size (ESS)** -- single-chain ESS via Geyer's initial
//!   positive sequence estimator.
//! * **Bulk ESS (multi-chain)** -- ESS for location summaries (rank-normalised).
//! * **Tail ESS (multi-chain)** -- ESS for tail quantile summaries.
//!
//! ## Uncertainty of estimates
//! * **Monte Carlo standard error (MCSE)** -- for the mean and for arbitrary
//!   quantiles.
//!
//! ## Trace statistics
//! * Running mean, running variance (Welford's algorithm).
//! * Split-chain R-hat trace over sliding windows.
//!
//! ## Divergence & energy diagnostics (HMC / NUTS)
//! * Count, proportion, and indices of divergent transitions.
//! * Energy-based Bayesian Fraction of Missing Information (E-BFMI).
//!
//! ## Aggregate report
//! * [`DiagnosticReport`] -- collects all of the above into a single structure
//!   with a human-readable summary.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2};

// ═══════════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════════

/// Rational approximation of the standard-normal inverse CDF (probit).
///
/// Uses Peter Acklam's algorithm, accurate to roughly 1e-9 for p in (0, 1).
fn normal_quantile(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239e0,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838e0,
        -2.549_732_539_343_734e0,
        4.374_664_141_464_968e0,
        2.938_163_982_698_783e0,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996e0,
        3.754_408_661_907_416e0,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    let p = p.clamp(1e-15, 1.0 - 1e-15);

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Rank-normalise a single-chain sample using the inverse normal CDF of
/// fractional ranks (Blom's plotting positions).
fn rank_normalize_single(samples: &[f64]) -> Vec<f64> {
    let n = samples.len();
    if n == 0 {
        return Vec::new();
    }

    let mut indexed: Vec<(usize, f64)> = samples.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < f64::EPSILON {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0; // 1-based
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }

    let nf = n as f64;
    ranks
        .iter()
        .map(|&r| {
            let p = (r - 0.375) / (nf + 0.25);
            normal_quantile(p.clamp(1e-10, 1.0 - 1e-10))
        })
        .collect()
}

/// Rank-normalise draws across multiple chains jointly.
fn rank_normalize_multi(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut all: Vec<(usize, usize, f64)> = Vec::new();
    for (c, chain) in chains.iter().enumerate() {
        for (s, &v) in chain.iter().enumerate() {
            all.push((c, s, v));
        }
    }
    let n_total = all.len();
    if n_total == 0 {
        return chains.to_vec();
    }

    all.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0f64; n_total];
    let mut i = 0;
    while i < n_total {
        let mut j = i;
        while j < n_total && (all[j].2 - all[i].2).abs() < f64::EPSILON {
            j += 1;
        }
        let avg_rank = (i + j) as f64 / 2.0 + 0.5;
        for k in i..j {
            ranks[k] = avg_rank;
        }
        i = j;
    }

    let nf = n_total as f64;
    let mut result: Vec<Vec<f64>> = chains.iter().map(|c| vec![0.0; c.len()]).collect();
    for (idx, &(c, s, _)) in all.iter().enumerate() {
        let p = (ranks[idx] - 0.375) / (nf + 0.25);
        result[c][s] = normal_quantile(p.clamp(1e-10, 1.0 - 1e-10));
    }
    result
}

/// Compute the autocorrelation function of a centred sequence using a direct
/// O(n * max_lag) approach.
fn compute_acf(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    if n < 2 {
        return vec![1.0];
    }
    let nf = n as f64;
    let mean: f64 = x.iter().sum::<f64>() / nf;
    let var: f64 = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / nf;
    if var.abs() < f64::EPSILON {
        return vec![1.0; max_lag.min(n)];
    }

    let eff_lag = max_lag.min(n);
    let mut acf = vec![0.0; eff_lag];
    for lag in 0..eff_lag {
        let mut s = 0.0;
        for i in 0..(n - lag) {
            s += (x[i] - mean) * (x[i + lag] - mean);
        }
        acf[lag] = s / (nf * var);
    }
    acf
}

/// Geyer initial-positive-sequence estimator.
/// Returns ESS = n / tau where tau is estimated from the ACF.
fn geyer_ess_from_acf(acf: &[f64], n: usize) -> f64 {
    if acf.is_empty() || n == 0 {
        return 0.0;
    }

    let max_pairs = acf.len() / 2;
    let mut tau = acf[0]; // lag-0, typically 1.0
    let mut prev_pair_sum = f64::INFINITY;

    for t in 0..max_pairs {
        let idx0 = 2 * t + 1;
        let idx1 = idx0 + 1;
        if idx1 >= acf.len() {
            break;
        }
        let pair_sum = acf[idx0] + acf[idx1];
        if pair_sum < 0.0 {
            break;
        }
        // initial monotone sequence: clip at previous pair sum
        let pair_sum = pair_sum.min(prev_pair_sum);
        tau += 2.0 * pair_sum;
        prev_pair_sum = pair_sum;
    }

    let tau = tau.max(1.0);
    n as f64 / tau
}

/// Validate that multiple chains have the same length and meet minimum
/// requirements.
fn validate_chains(
    chains: &[Vec<f64>],
    min_chains: usize,
    min_samples: usize,
) -> StatsResult<(usize, usize)> {
    if chains.len() < min_chains {
        return Err(StatsError::InsufficientData(format!(
            "requires at least {} chains, got {}",
            min_chains,
            chains.len()
        )));
    }
    let n = chains[0].len();
    if n < min_samples {
        return Err(StatsError::InsufficientData(format!(
            "requires at least {} samples per chain, got {}",
            min_samples, n
        )));
    }
    for (i, c) in chains.iter().enumerate() {
        if c.len() != n {
            return Err(StatsError::DimensionMismatch(format!(
                "chain {} has length {} but chain 0 has length {}",
                i,
                c.len(),
                n
            )));
        }
    }
    Ok((chains.len(), n))
}

// ═══════════════════════════════════════════════════════════════════
// R-hat (Gelman-Rubin convergence diagnostic)
// ═══════════════════════════════════════════════════════════════════

/// Compute the classic Gelman-Rubin R-hat for multiple chains of a single
/// parameter.
///
/// R-hat = sqrt( var_hat_plus / W ) where var_hat_plus is a mixture of
/// between-chain variance B and within-chain variance W.  Values near 1.0
/// indicate convergence; the standard threshold is R-hat < 1.01.
///
/// # Arguments
/// * `chains` -- at least 2 chains, all the same length (>= 2 samples).
pub fn r_hat(chains: &[Vec<f64>]) -> StatsResult<f64> {
    let (m_cnt, n) = validate_chains(chains, 2, 2)?;
    let m = m_cnt as f64;
    let nf = n as f64;

    let chain_means: Vec<f64> = chains.iter().map(|c| c.iter().sum::<f64>() / nf).collect();
    let grand_mean = chain_means.iter().sum::<f64>() / m;

    let b = chain_means
        .iter()
        .map(|&mu| (mu - grand_mean).powi(2))
        .sum::<f64>()
        * nf
        / (m - 1.0);

    let w: f64 = chains
        .iter()
        .zip(chain_means.iter())
        .map(|(chain, &mu)| chain.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / (nf - 1.0))
        .sum::<f64>()
        / m;

    if w <= 0.0 {
        return Ok(1.0);
    }

    let var_hat = (1.0 - 1.0 / nf) * w + b / nf;
    Ok((var_hat / w).sqrt())
}

/// Compute split R-hat for a **single** chain by splitting it in two halves.
///
/// This detects within-chain non-stationarity.
pub fn split_r_hat(chain: &[f64]) -> StatsResult<f64> {
    if chain.len() < 4 {
        return Err(StatsError::InsufficientData(
            "split R-hat requires at least 4 samples".into(),
        ));
    }
    let mid = chain.len() / 2;
    r_hat(&[chain[..mid].to_vec(), chain[mid..2 * mid].to_vec()])
}

/// Compute split R-hat for **multiple** chains (recommended default).
///
/// Each chain is split in half, yielding 2M split-chains.  R-hat is then
/// computed on those 2M chains.
pub fn split_rhat(chains: &[Vec<f64>]) -> StatsResult<f64> {
    let (_m, n) = validate_chains(chains, 2, 4)?;
    let half = n / 2;

    let mut split_chains = Vec::with_capacity(chains.len() * 2);
    for chain in chains {
        split_chains.push(chain[..half].to_vec());
        split_chains.push(chain[half..2 * half].to_vec());
    }
    rhat_from_split(&split_chains, half)
}

/// Compute **rank-normalised** split R-hat (Vehtari et al. 2021).
///
/// Same as [`split_rhat`] but first rank-normalises the draws so that the
/// diagnostic is robust to heavy-tailed distributions.
pub fn split_rhat_rank(chains: &[Vec<f64>]) -> StatsResult<f64> {
    let (_m, n) = validate_chains(chains, 2, 4)?;
    let half = n / 2;

    let mut split_chains = Vec::with_capacity(chains.len() * 2);
    for chain in chains {
        split_chains.push(chain[..half].to_vec());
        split_chains.push(chain[half..2 * half].to_vec());
    }

    let ranked = rank_normalize_multi(&split_chains);
    rhat_from_split(&ranked, half)
}

/// Internal helper: compute R-hat from already-split chains.
fn rhat_from_split(split_chains: &[Vec<f64>], n: usize) -> StatsResult<f64> {
    let m = split_chains.len() as f64;
    let nf = n as f64;

    let mut w = 0.0;
    let mut chain_means = Vec::with_capacity(split_chains.len());
    for chain in split_chains {
        let mean = chain.iter().sum::<f64>() / nf;
        chain_means.push(mean);
        let s2: f64 = chain.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (nf - 1.0);
        w += s2;
    }
    w /= m;

    let grand_mean = chain_means.iter().sum::<f64>() / m;
    let b = chain_means
        .iter()
        .map(|&mu| (mu - grand_mean).powi(2))
        .sum::<f64>()
        * nf
        / (m - 1.0);

    let var_hat = (1.0 - 1.0 / nf) * w + b / nf;

    if w.abs() < f64::EPSILON {
        return Ok(1.0);
    }

    Ok((var_hat / w).sqrt())
}

// ═══════════════════════════════════════════════════════════════════
// Autocorrelation (public)
// ═══════════════════════════════════════════════════════════════════

/// Compute the autocorrelation function of a sample sequence up to `max_lag`.
///
/// Element 0 is always 1.0.  Uses the biased normalisation (dividing by N
/// rather than N-k) which is the standard for MCMC diagnostics.
pub fn autocorrelation(samples: &[f64], max_lag: usize) -> StatsResult<Vec<f64>> {
    if samples.len() < 2 {
        return Err(StatsError::InsufficientData(
            "autocorrelation requires at least 2 samples".into(),
        ));
    }
    Ok(compute_acf(samples, max_lag + 1))
}

// ═══════════════════════════════════════════════════════════════════
// Effective Sample Size (ESS)
// ═══════════════════════════════════════════════════════════════════

/// Compute the effective sample size for a **single** chain using Geyer's
/// initial positive sequence estimator.
pub fn effective_sample_size(samples: &[f64]) -> StatsResult<f64> {
    let n = samples.len();
    if n < 4 {
        return Err(StatsError::InsufficientData(
            "ESS requires at least 4 samples".into(),
        ));
    }

    let max_lag = n / 2;
    let acf = compute_acf(samples, max_lag);
    let ess = geyer_ess_from_acf(&acf, n);
    Ok(ess.clamp(1.0, n as f64))
}

/// Compute **bulk** ESS for a single chain.
///
/// The chain is first rank-normalised so that ESS is measured for location
/// summaries (mean, median).
pub fn bulk_ess_single(samples: &[f64]) -> StatsResult<f64> {
    if samples.len() < 4 {
        return Err(StatsError::InsufficientData(
            "bulk ESS requires at least 4 samples".into(),
        ));
    }
    let ranked = rank_normalize_single(samples);
    effective_sample_size(&ranked)
}

/// Compute **tail** ESS for a single chain.
///
/// Uses the minimum of the ESS of indicators I(x <= q5) and I(x <= q95).
pub fn tail_ess_single(samples: &[f64]) -> StatsResult<f64> {
    let n = samples.len();
    if n < 20 {
        return Err(StatsError::InsufficientData(
            "tail ESS requires at least 20 samples".into(),
        ));
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q05 = sorted[((n as f64 * 0.05).floor() as usize).min(n.saturating_sub(1))];
    let q95 = sorted[((n as f64 * 0.95).floor() as usize).min(n.saturating_sub(1))];

    let ind_lo: Vec<f64> = samples
        .iter()
        .map(|&x| if x <= q05 { 1.0 } else { 0.0 })
        .collect();
    let ind_hi: Vec<f64> = samples
        .iter()
        .map(|&x| if x <= q95 { 1.0 } else { 0.0 })
        .collect();

    let ess_lo = effective_sample_size(&ind_lo)?;
    let ess_hi = effective_sample_size(&ind_hi)?;
    Ok(ess_lo.min(ess_hi))
}

/// Compute **bulk** ESS across **multiple** chains.
///
/// Draws are rank-normalised jointly, then chains are split in half and the
/// cross-chain ESS estimator is applied.
pub fn bulk_ess(chains: &[Vec<f64>]) -> StatsResult<f64> {
    let (_m, _n) = validate_chains(chains, 2, 4)?;
    let ranked = rank_normalize_multi(chains);
    ess_from_chains(&ranked)
}

/// Compute **tail** ESS across **multiple** chains.
///
/// Returns min(ESS of I(x <= q5), ESS of I(x <= q95)) computed across all
/// chains jointly.
pub fn tail_ess(chains: &[Vec<f64>]) -> StatsResult<f64> {
    let (_m, _n) = validate_chains(chains, 2, 4)?;

    let mut pooled: Vec<f64> = chains.iter().flat_map(|c| c.iter().copied()).collect();
    pooled.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n_total = pooled.len();
    let q05 = pooled[((n_total as f64 * 0.05).floor() as usize).min(n_total.saturating_sub(1))];
    let q95 = pooled[((n_total as f64 * 0.95).floor() as usize).min(n_total.saturating_sub(1))];

    let ind_lo: Vec<Vec<f64>> = chains
        .iter()
        .map(|c| {
            c.iter()
                .map(|&v| if v <= q05 { 1.0 } else { 0.0 })
                .collect()
        })
        .collect();
    let ind_hi: Vec<Vec<f64>> = chains
        .iter()
        .map(|c| {
            c.iter()
                .map(|&v| if v <= q95 { 1.0 } else { 0.0 })
                .collect()
        })
        .collect();

    let ess_lo = ess_from_chains(&ind_lo)?;
    let ess_hi = ess_from_chains(&ind_hi)?;
    Ok(ess_lo.min(ess_hi))
}

/// Internal: compute ESS from multiple (possibly transformed) chains.
///
/// Splits each chain in half, computes per-chain ACF, averages the ACFs, and
/// applies Geyer's estimator.
fn ess_from_chains(chains: &[Vec<f64>]) -> StatsResult<f64> {
    let m = chains.len();
    let n = chains[0].len();
    if n < 4 || m < 1 {
        return Ok(0.0);
    }

    let half = n / 2;
    let mut split: Vec<Vec<f64>> = Vec::with_capacity(m * 2);
    for chain in chains {
        split.push(chain[..half].to_vec());
        split.push(chain[half..2 * half].to_vec());
    }

    let m_split = split.len();
    let n_split = half;

    // Per-chain ACF
    let max_lag = n_split;
    let mut acfs: Vec<Vec<f64>> = Vec::with_capacity(m_split);
    for chain in &split {
        acfs.push(compute_acf(chain, max_lag));
    }

    // Average ACF across split chains
    let min_len = acfs.iter().map(|a| a.len()).min().unwrap_or(1);
    let mut avg_acf = vec![0.0; min_len];
    for acf in &acfs {
        for (k, &val) in acf.iter().enumerate().take(min_len) {
            avg_acf[k] += val;
        }
    }
    for v in &mut avg_acf {
        *v /= m_split as f64;
    }

    let total_draws = n_split * m_split;
    let ess = geyer_ess_from_acf(&avg_acf, total_draws);
    Ok(ess.max(1.0))
}

// ═══════════════════════════════════════════════════════════════════
// Monte Carlo Standard Error (MCSE)
// ═══════════════════════════════════════════════════════════════════

/// MCSE of the posterior mean for a **single** chain.
///
/// MCSE_mean = sd(samples) / sqrt(ESS).
pub fn mcse(samples: &[f64]) -> StatsResult<f64> {
    let n = samples.len();
    if n < 4 {
        return Err(StatsError::InsufficientData(
            "MCSE requires at least 4 samples".into(),
        ));
    }
    let nf = n as f64;
    let mean = samples.iter().sum::<f64>() / nf;
    let var = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (nf - 1.0);
    let sd = var.sqrt();
    let ess = effective_sample_size(samples)?;
    if ess <= 0.0 {
        return Ok(sd);
    }
    Ok(sd / ess.sqrt())
}

/// MCSE of the posterior mean across **multiple** chains.
///
/// MCSE_mean = sd(pooled) / sqrt(bulk_ESS).
pub fn mcse_mean(chains: &[Vec<f64>]) -> StatsResult<f64> {
    let ess = bulk_ess(chains)?;
    if ess < 1.0 {
        return Err(StatsError::InsufficientData(
            "ESS is too small to compute MCSE for the mean".into(),
        ));
    }
    let pooled: Vec<f64> = chains.iter().flat_map(|c| c.iter().copied()).collect();
    let n = pooled.len() as f64;
    let mean = pooled.iter().sum::<f64>() / n;
    let var = pooled.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    Ok(var.sqrt() / ess.sqrt())
}

/// MCSE for a given quantile across **multiple** chains.
///
/// Uses the asymptotic formula se(q_p) = sqrt( p(1-p) / (n_eff * f(q_p)^2) )
/// where f is estimated via a Gaussian kernel density at the quantile.
///
/// # Arguments
/// * `chains` -- multiple chains.
/// * `prob` -- quantile probability in [0, 1].
pub fn mcse_quantile(chains: &[Vec<f64>], prob: f64) -> StatsResult<f64> {
    if !(0.0..=1.0).contains(&prob) {
        return Err(StatsError::DomainError(
            "prob must be between 0 and 1".into(),
        ));
    }
    let ess = tail_ess(chains)?;
    if ess < 1.0 {
        return Err(StatsError::InsufficientData(
            "ESS too small for quantile MCSE".into(),
        ));
    }

    let mut pooled: Vec<f64> = chains.iter().flat_map(|c| c.iter().copied()).collect();
    pooled.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = pooled.len();
    if n == 0 {
        return Err(StatsError::InsufficientData("no samples".into()));
    }

    let q_idx = ((n as f64 * prob).floor() as usize).min(n.saturating_sub(1));
    let q_val = pooled[q_idx];

    // IQR-based bandwidth (Silverman)
    let q25 = pooled[((n as f64 * 0.25).floor() as usize).min(n.saturating_sub(1))];
    let q75 = pooled[((n as f64 * 0.75).floor() as usize).min(n.saturating_sub(1))];
    let iqr = q75 - q25;
    if iqr.abs() < f64::EPSILON {
        return Ok(0.0);
    }

    let mean_val = pooled.iter().sum::<f64>() / n as f64;
    let sd =
        (pooled.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt();
    let h = 0.9 * sd.min(iqr / 1.34) * (n as f64).powf(-0.2);
    if h.abs() < f64::EPSILON {
        return Ok(0.0);
    }

    let inv_h = 1.0 / h;
    let norm_const = inv_h / (2.0 * std::f64::consts::PI).sqrt();
    let f_q: f64 = pooled
        .iter()
        .map(|&x| {
            let u = (x - q_val) * inv_h;
            norm_const * (-0.5 * u * u).exp()
        })
        .sum::<f64>()
        / n as f64;

    if f_q.abs() < f64::EPSILON {
        return Ok(0.0);
    }

    Ok((prob * (1.0 - prob) / (ess * f_q * f_q)).sqrt())
}

// ═══════════════════════════════════════════════════════════════════
// Trace statistics
// ═══════════════════════════════════════════════════════════════════

/// Running (cumulative) mean of a single chain.
///
/// Returns an array where element `i` is the mean of `chain[0..=i]`.
pub fn running_mean(chain: &[f64]) -> StatsResult<Array1<f64>> {
    if chain.is_empty() {
        return Err(StatsError::InsufficientData(
            "running_mean requires at least 1 sample".into(),
        ));
    }
    let n = chain.len();
    let mut result = Array1::zeros(n);
    let mut cumsum = 0.0;
    for (i, &v) in chain.iter().enumerate() {
        cumsum += v;
        result[i] = cumsum / (i as f64 + 1.0);
    }
    Ok(result)
}

/// Running (cumulative) variance using Welford's online algorithm.
///
/// Element `i` is the sample variance (ddof=1) of `chain[0..=i]`.  Element 0
/// is always 0.0.
pub fn running_variance(chain: &[f64]) -> StatsResult<Array1<f64>> {
    if chain.is_empty() {
        return Err(StatsError::InsufficientData(
            "running_variance requires at least 1 sample".into(),
        ));
    }
    let n = chain.len();
    let mut result = Array1::zeros(n);
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for (i, &v) in chain.iter().enumerate() {
        let delta = v - mean;
        mean += delta / (i as f64 + 1.0);
        let delta2 = v - mean;
        m2 += delta * delta2;
        if i > 0 {
            result[i] = m2 / i as f64; // sample variance ddof=1
        }
    }
    Ok(result)
}

/// Split-chain R-hat evaluated at each window boundary.
///
/// Useful for visualising convergence progress over the course of sampling.
///
/// # Arguments
/// * `chains` -- at least 2 chains (same length).
/// * `window_size` -- samples per evaluation point.
pub fn split_rhat_trace(chains: &[Vec<f64>], window_size: usize) -> StatsResult<Vec<f64>> {
    let (_m, n) = validate_chains(chains, 2, 4)?;
    if window_size < 4 {
        return Err(StatsError::InvalidArgument(
            "window_size must be at least 4".into(),
        ));
    }

    let mut trace = Vec::new();
    let mut end = window_size;
    while end <= n {
        let sub: Vec<Vec<f64>> = chains.iter().map(|c| c[..end].to_vec()).collect();
        match split_rhat(&sub) {
            Ok(rh) => trace.push(rh),
            Err(_) => break,
        }
        end += window_size;
    }
    Ok(trace)
}

// ═══════════════════════════════════════════════════════════════════
// Divergence diagnostics (HMC / NUTS)
// ═══════════════════════════════════════════════════════════════════

/// Summary of divergence-related diagnostics.
#[derive(Debug, Clone)]
pub struct DivergenceDiagnostics {
    /// Total number of transitions examined.
    pub total_transitions: usize,
    /// Number of divergent transitions.
    pub n_divergent: usize,
    /// Proportion of divergent transitions.
    pub divergence_rate: f64,
    /// Indices of divergent transitions.
    pub divergent_indices: Vec<usize>,
}

/// Count and characterise divergent transitions.
///
/// # Arguments
/// * `divergent_flags` -- one boolean per transition; `true` = divergent.
pub fn divergence_diagnostics(divergent_flags: &[bool]) -> DivergenceDiagnostics {
    let total = divergent_flags.len();
    let mut n_div = 0usize;
    let mut indices = Vec::new();
    for (i, &d) in divergent_flags.iter().enumerate() {
        if d {
            n_div += 1;
            indices.push(i);
        }
    }
    DivergenceDiagnostics {
        total_transitions: total,
        n_divergent: n_div,
        divergence_rate: if total > 0 {
            n_div as f64 / total as f64
        } else {
            0.0
        },
        divergent_indices: indices,
    }
}

/// Energy-based Bayesian Fraction of Missing Information (E-BFMI).
///
/// E-BFMI = Var(E_n - E_{n-1}) / Var(E_n).  Values < 0.3 indicate poor
/// exploration of the energy level set (Betancourt, 2016).
pub fn energy_bfmi(energies: &[f64]) -> StatsResult<f64> {
    let n = energies.len();
    if n < 2 {
        return Err(StatsError::InsufficientData(
            "energy_bfmi requires at least 2 energy values".into(),
        ));
    }

    let mut d_energies = Vec::with_capacity(n - 1);
    for i in 0..(n - 1) {
        d_energies.push(energies[i + 1] - energies[i]);
    }
    let n_de = d_energies.len() as f64;
    let mean_de = d_energies.iter().sum::<f64>() / n_de;
    let var_de = d_energies
        .iter()
        .map(|&x| (x - mean_de).powi(2))
        .sum::<f64>()
        / (n_de - 1.0).max(1.0);

    let nf = n as f64;
    let mean_e = energies.iter().sum::<f64>() / nf;
    let var_e = energies.iter().map(|&x| (x - mean_e).powi(2)).sum::<f64>() / (nf - 1.0).max(1.0);

    if var_e.abs() < f64::EPSILON {
        return Ok(1.0);
    }
    Ok(var_de / var_e)
}

/// Detailed energy diagnostics including E-BFMI, mean energy, and energy
/// variance.
#[derive(Debug, Clone)]
pub struct EnergyDiagnostics {
    /// E-BFMI statistic.
    pub e_bfmi: f64,
    /// Mean Hamiltonian energy.
    pub mean_energy: f64,
    /// Variance of Hamiltonian energy.
    pub energy_variance: f64,
    /// Whether E-BFMI indicates potential problems (E-BFMI < 0.3).
    pub low_bfmi_warning: bool,
}

/// Compute full energy diagnostics from a vector of Hamiltonian energies.
pub fn energy_diagnostics(energies: &[f64]) -> StatsResult<EnergyDiagnostics> {
    let n = energies.len();
    if n < 2 {
        return Err(StatsError::InsufficientData(
            "energy_diagnostics requires at least 2 energy values".into(),
        ));
    }
    let bfmi = energy_bfmi(energies)?;
    let mean_e = energies.iter().sum::<f64>() / n as f64;
    let var_e =
        energies.iter().map(|&x| (x - mean_e).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0);

    Ok(EnergyDiagnostics {
        e_bfmi: bfmi,
        mean_energy: mean_e,
        energy_variance: var_e,
        low_bfmi_warning: bfmi < 0.3,
    })
}

// ═══════════════════════════════════════════════════════════════════
// Multi-parameter convenience functions
// ═══════════════════════════════════════════════════════════════════

/// Split R-hat for every parameter across multiple chains stored as
/// `Array2<f64>` (shape: n_samples x n_params).
pub fn rhat_per_parameter(chain_samples: &[&Array2<f64>]) -> StatsResult<Array1<f64>> {
    if chain_samples.len() < 2 {
        return Err(StatsError::InsufficientData(
            "rhat_per_parameter requires at least 2 chains".into(),
        ));
    }
    let (n_samples, n_params) = chain_samples[0].dim();
    for (i, cs) in chain_samples.iter().enumerate() {
        if cs.dim() != (n_samples, n_params) {
            return Err(StatsError::DimensionMismatch(format!(
                "chain {} has shape {:?} but chain 0 has shape {:?}",
                i,
                cs.dim(),
                (n_samples, n_params)
            )));
        }
    }

    let mut result = Array1::zeros(n_params);
    for p in 0..n_params {
        let pc: Vec<Vec<f64>> = chain_samples
            .iter()
            .map(|cs| cs.column(p).to_vec())
            .collect();
        result[p] = split_rhat(&pc)?;
    }
    Ok(result)
}

/// Bulk ESS for every parameter across multiple chains.
pub fn bulk_ess_per_parameter(chain_samples: &[&Array2<f64>]) -> StatsResult<Array1<f64>> {
    if chain_samples.len() < 2 {
        return Err(StatsError::InsufficientData(
            "bulk_ess_per_parameter requires at least 2 chains".into(),
        ));
    }
    let (n_samples, n_params) = chain_samples[0].dim();
    for (i, cs) in chain_samples.iter().enumerate() {
        if cs.dim() != (n_samples, n_params) {
            return Err(StatsError::DimensionMismatch(format!(
                "chain {} shape {:?} != expected {:?}",
                i,
                cs.dim(),
                (n_samples, n_params)
            )));
        }
    }

    let mut result = Array1::zeros(n_params);
    for p in 0..n_params {
        let pc: Vec<Vec<f64>> = chain_samples
            .iter()
            .map(|cs| cs.column(p).to_vec())
            .collect();
        result[p] = bulk_ess(&pc)?;
    }
    Ok(result)
}

/// Tail ESS for every parameter across multiple chains.
pub fn tail_ess_per_parameter(chain_samples: &[&Array2<f64>]) -> StatsResult<Array1<f64>> {
    if chain_samples.len() < 2 {
        return Err(StatsError::InsufficientData(
            "tail_ess_per_parameter requires at least 2 chains".into(),
        ));
    }
    let (n_samples, n_params) = chain_samples[0].dim();
    for (i, cs) in chain_samples.iter().enumerate() {
        if cs.dim() != (n_samples, n_params) {
            return Err(StatsError::DimensionMismatch(format!(
                "chain {} shape {:?} != expected {:?}",
                i,
                cs.dim(),
                (n_samples, n_params)
            )));
        }
    }

    let mut result = Array1::zeros(n_params);
    for p in 0..n_params {
        let pc: Vec<Vec<f64>> = chain_samples
            .iter()
            .map(|cs| cs.column(p).to_vec())
            .collect();
        result[p] = tail_ess(&pc)?;
    }
    Ok(result)
}

// ═══════════════════════════════════════════════════════════════════
// DiagnosticReport
// ═══════════════════════════════════════════════════════════════════

/// Per-parameter diagnostic summary.
#[derive(Debug, Clone)]
pub struct ParameterDiagnostic {
    /// Parameter index (0-based).
    pub index: usize,
    /// Optional parameter name.
    pub name: Option<String>,
    /// Split R-hat.
    pub rhat: f64,
    /// Bulk effective sample size.
    pub bulk_ess: f64,
    /// Tail effective sample size.
    pub tail_ess: f64,
    /// MCSE for the mean.
    pub mcse_mean: f64,
    /// Posterior mean.
    pub mean: f64,
    /// Posterior standard deviation.
    pub sd: f64,
}

/// Aggregate diagnostic report for a multi-parameter MCMC run.
#[derive(Debug, Clone)]
pub struct DiagnosticReport {
    /// Per-parameter diagnostics.
    pub parameters: Vec<ParameterDiagnostic>,
    /// Number of chains.
    pub n_chains: usize,
    /// Number of samples per chain.
    pub n_samples: usize,
    /// Optional divergence diagnostics.
    pub divergence: Option<DivergenceDiagnostics>,
    /// Optional energy diagnostics.
    pub energy: Option<EnergyDiagnostics>,
}

impl DiagnosticReport {
    /// Build a diagnostic report from multiple chains.
    ///
    /// # Arguments
    /// * `chain_samples` -- one `Array2<f64>` per chain, shape (n_samples, n_params).
    /// * `param_names` -- optional parameter names.
    /// * `divergent_flags` -- optional divergence flags (one per transition).
    /// * `energies` -- optional Hamiltonian energies (one per transition).
    pub fn new(
        chain_samples: &[&Array2<f64>],
        param_names: Option<&[String]>,
        divergent_flags: Option<&[bool]>,
        energies: Option<&[f64]>,
    ) -> StatsResult<Self> {
        if chain_samples.is_empty() {
            return Err(StatsError::InsufficientData(
                "at least 1 chain is required".into(),
            ));
        }
        let (n_samples, n_params) = chain_samples[0].dim();
        let n_chains = chain_samples.len();

        for (i, cs) in chain_samples.iter().enumerate() {
            if cs.dim() != (n_samples, n_params) {
                return Err(StatsError::DimensionMismatch(format!(
                    "chain {} shape {:?} != expected {:?}",
                    i,
                    cs.dim(),
                    (n_samples, n_params)
                )));
            }
        }

        let multi_chain = n_chains >= 2;
        let mut parameters = Vec::with_capacity(n_params);

        for p in 0..n_params {
            let param_chains: Vec<Vec<f64>> = chain_samples
                .iter()
                .map(|cs| cs.column(p).to_vec())
                .collect();

            // Pooled mean / sd
            let pooled: Vec<f64> = param_chains
                .iter()
                .flat_map(|c| c.iter().copied())
                .collect();
            let nt = pooled.len() as f64;
            let p_mean = pooled.iter().sum::<f64>() / nt;
            let p_sd = (pooled.iter().map(|&x| (x - p_mean).powi(2)).sum::<f64>()
                / (nt - 1.0).max(1.0))
            .sqrt();

            let (rhat_val, bess_val, tess_val, mcse_val) = if multi_chain {
                (
                    split_rhat(&param_chains).unwrap_or(f64::NAN),
                    bulk_ess(&param_chains).unwrap_or(f64::NAN),
                    tail_ess(&param_chains).unwrap_or(f64::NAN),
                    mcse_mean(&param_chains).unwrap_or(f64::NAN),
                )
            } else {
                // Single chain fallback
                let samples = &param_chains[0];
                (
                    split_r_hat(samples).unwrap_or(f64::NAN),
                    bulk_ess_single(samples).unwrap_or(f64::NAN),
                    tail_ess_single(samples).unwrap_or(f64::NAN),
                    mcse(samples).unwrap_or(f64::NAN),
                )
            };

            let name = param_names.and_then(|names| names.get(p).cloned());
            parameters.push(ParameterDiagnostic {
                index: p,
                name,
                rhat: rhat_val,
                bulk_ess: bess_val,
                tail_ess: tess_val,
                mcse_mean: mcse_val,
                mean: p_mean,
                sd: p_sd,
            });
        }

        let divergence = divergent_flags.map(divergence_diagnostics);
        let energy = energies.and_then(|e| energy_diagnostics(e).ok());

        Ok(DiagnosticReport {
            parameters,
            n_chains,
            n_samples,
            divergence,
            energy,
        })
    }

    /// Check if all parameters have R-hat below the given threshold (default 1.01).
    pub fn all_rhat_ok(&self, threshold: Option<f64>) -> bool {
        let thresh = threshold.unwrap_or(1.01);
        self.parameters
            .iter()
            .all(|p| p.rhat <= thresh || p.rhat.is_nan())
    }

    /// Check if all parameters have bulk ESS above the given minimum.
    pub fn all_bulk_ess_ok(&self, min_ess: f64) -> bool {
        self.parameters
            .iter()
            .all(|p| p.bulk_ess >= min_ess || p.bulk_ess.is_nan())
    }

    /// Check if all parameters have tail ESS above the given minimum.
    pub fn all_tail_ess_ok(&self, min_ess: f64) -> bool {
        self.parameters
            .iter()
            .all(|p| p.tail_ess >= min_ess || p.tail_ess.is_nan())
    }

    /// Check if there are any divergences.
    pub fn has_divergences(&self) -> bool {
        self.divergence
            .as_ref()
            .map_or(false, |d| d.n_divergent > 0)
    }

    /// Check if E-BFMI indicates problems.
    pub fn has_low_bfmi(&self) -> bool {
        self.energy.as_ref().map_or(false, |e| e.low_bfmi_warning)
    }

    /// Overall health check:
    /// - R-hat < 1.01 for all parameters
    /// - Bulk ESS > 100 * n_chains
    /// - Tail ESS > 100 * n_chains
    /// - No divergences
    /// - E-BFMI >= 0.3 (if provided)
    pub fn is_healthy(&self) -> bool {
        let min_ess = 100.0 * self.n_chains as f64;
        self.all_rhat_ok(None)
            && self.all_bulk_ess_ok(min_ess)
            && self.all_tail_ess_ok(min_ess)
            && !self.has_divergences()
            && !self.has_low_bfmi()
    }

    /// Format a human-readable summary.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "MCMC Diagnostic Report ({} chains, {} samples/chain)\n",
            self.n_chains, self.n_samples
        ));
        s.push_str(&format!("{:-<72}\n", ""));
        s.push_str(&format!(
            "{:<14} {:>8} {:>10} {:>10} {:>10} {:>10}\n",
            "Parameter", "R-hat", "Bulk ESS", "Tail ESS", "MCSE", "Mean"
        ));
        s.push_str(&format!("{:-<72}\n", ""));

        for p in &self.parameters {
            let default_name = format!("param[{}]", p.index);
            let name = p.name.as_deref().unwrap_or(&default_name);
            s.push_str(&format!(
                "{:<14} {:>8.4} {:>10.1} {:>10.1} {:>10.6} {:>10.4}\n",
                name, p.rhat, p.bulk_ess, p.tail_ess, p.mcse_mean, p.mean,
            ));
        }

        if let Some(ref div) = self.divergence {
            s.push_str(&format!(
                "\nDivergences: {} / {} ({:.1}%)\n",
                div.n_divergent,
                div.total_transitions,
                div.divergence_rate * 100.0,
            ));
        }
        if let Some(ref eng) = self.energy {
            s.push_str(&format!(
                "E-BFMI: {:.4}{}\n",
                eng.e_bfmi,
                if eng.low_bfmi_warning {
                    " ** WARNING: low E-BFMI **"
                } else {
                    ""
                },
            ));
        }

        s.push_str(&format!(
            "\nOverall: {}\n",
            if self.is_healthy() {
                "HEALTHY"
            } else {
                "POTENTIAL ISSUES DETECTED"
            }
        ));
        s
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // Deterministic pseudo-normal generator (Box-Muller + LCG)
    // ---------------------------------------------------------------
    fn iid_normal_chain(n: usize, mean: f64, sd: f64, seed: u64) -> Vec<f64> {
        let mut state = seed;
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u1 = ((state >> 11) as f64 / (1u64 << 53) as f64).max(1e-15);
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u2 = (state >> 11) as f64 / (1u64 << 53) as f64;
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).sin();
            out.push(mean + sd * z0);
            if out.len() < n {
                out.push(mean + sd * z1);
            }
        }
        out.truncate(n);
        out
    }

    fn ar1_chain(n: usize, rho: f64, seed: u64) -> Vec<f64> {
        let innovations = iid_normal_chain(n, 0.0, 1.0, seed);
        let sd_scale = (1.0 - rho * rho).sqrt();
        let mut chain = Vec::with_capacity(n);
        let mut x = 0.0;
        for &e in &innovations {
            x = rho * x + sd_scale * e;
            chain.push(x);
        }
        chain
    }

    // ---------------------------------------------------------------
    // R-hat tests
    // ---------------------------------------------------------------

    #[test]
    fn test_r_hat_identical_chains() {
        // When both chains are constant (all same value), W=0 and R-hat is exactly 1.0
        // by the early-return branch in r_hat.
        let chain = vec![3.14; 10];
        let rhat = r_hat(&[chain.clone(), chain]).expect("should succeed");
        assert!(
            (rhat - 1.0).abs() < 1e-10,
            "R-hat for constant identical chains should be exactly 1.0, got {rhat}"
        );
    }

    #[test]
    fn test_split_rhat_converged_iid() {
        let c1 = iid_normal_chain(2000, 0.0, 1.0, 42);
        let c2 = iid_normal_chain(2000, 0.0, 1.0, 123);
        let c3 = iid_normal_chain(2000, 0.0, 1.0, 999);
        let rhat = split_rhat(&[c1, c2, c3]).expect("split_rhat failed");
        assert!(
            rhat < 1.05,
            "converged IID R-hat should be near 1.0, got {rhat}"
        );
    }

    #[test]
    fn test_split_rhat_non_converged() {
        let c1 = iid_normal_chain(500, 0.0, 1.0, 42);
        let c2 = iid_normal_chain(500, 5.0, 1.0, 123);
        let rhat = split_rhat(&[c1, c2]).expect("split_rhat failed");
        assert!(
            rhat > 1.1,
            "non-converged R-hat should be > 1.1, got {rhat}"
        );
    }

    #[test]
    fn test_split_rhat_rank_converged() {
        let c1 = iid_normal_chain(2000, 0.0, 1.0, 42);
        let c2 = iid_normal_chain(2000, 0.0, 1.0, 77);
        let rhat = split_rhat_rank(&[c1, c2]).expect("split_rhat_rank failed");
        assert!(
            rhat < 1.05,
            "rank R-hat converged should be near 1.0, got {rhat}"
        );
    }

    #[test]
    fn test_split_r_hat_single_chain() {
        let chain: Vec<f64> = (0..200).map(|i| (i as f64 * 0.05).sin() * 2.0).collect();
        let sr = split_r_hat(&chain).expect("should succeed");
        assert!(sr < 2.0, "split R-hat for periodic chain, got {sr}");
    }

    #[test]
    fn test_split_r_hat_nonstationary() {
        let mut chain = Vec::with_capacity(200);
        for i in 0..100 {
            chain.push(i as f64 * 0.01);
        }
        for i in 0..100 {
            chain.push(10.0 + i as f64 * 0.01);
        }
        let sr = split_r_hat(&chain).expect("should succeed");
        assert!(
            sr > 1.01,
            "non-stationary split R-hat should be > 1.01, got {sr}"
        );
    }

    #[test]
    fn test_rhat_insufficient_chains() {
        let c1 = iid_normal_chain(100, 0.0, 1.0, 1);
        assert!(split_rhat(&[c1]).is_err());
    }

    #[test]
    fn test_rhat_insufficient_samples() {
        let c1 = vec![1.0, 2.0];
        let c2 = vec![1.5, 2.5];
        assert!(split_rhat(&[c1, c2]).is_err());
    }

    // ---------------------------------------------------------------
    // Autocorrelation tests
    // ---------------------------------------------------------------

    #[test]
    fn test_autocorrelation_white_noise() {
        // Use iid_normal_chain for genuine white noise with good pseudo-random properties
        let samples = iid_normal_chain(1000, 0.0, 1.0, 42);
        let acf = autocorrelation(&samples, 20).expect("should succeed");
        assert!((acf[0] - 1.0).abs() < 1e-10, "ACF at lag 0 should be 1.0");
        for lag in 1..=20 {
            assert!(
                acf[lag].abs() < 0.15,
                "ACF at lag {lag} should be near 0, got {}",
                acf[lag]
            );
        }
    }

    #[test]
    fn test_autocorrelation_ar1() {
        // Use ar1_chain helper which generates a proper AR(1) process with white noise
        // innovations from the LCG-based normal generator.
        let samples = ar1_chain(2000, 0.8, 42);
        let acf = autocorrelation(&samples, 5).expect("should succeed");
        assert!(
            acf[1] > 0.5,
            "AR(0.8) ACF at lag 1 should be > 0.5, got {}",
            acf[1]
        );
    }

    // ---------------------------------------------------------------
    // Single-chain ESS tests
    // ---------------------------------------------------------------

    #[test]
    fn test_ess_single_iid() {
        let n = 500;
        let samples: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 * 0.618033988749;
                (x - x.floor()) * 2.0 - 1.0
            })
            .collect();
        let ess = effective_sample_size(&samples).expect("should succeed");
        assert!(
            ess > n as f64 * 0.3,
            "ESS for quasi-iid should be large, got {ess} (n={n})"
        );
    }

    #[test]
    fn test_ess_single_correlated() {
        // Use ar1_chain helper for a proper AR(1) process with rho=0.95.
        // High autocorrelation should reduce ESS significantly below n.
        let n = 1000;
        let samples = ar1_chain(n, 0.95, 42);
        let ess = effective_sample_size(&samples).expect("should succeed");
        assert!(
            ess < n as f64 * 0.5,
            "ESS for AR(0.95) should be well below n/2, got {ess}"
        );
    }

    #[test]
    fn test_bulk_ess_single() {
        let n = 500;
        let samples: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 * 0.618033988749;
                (x - x.floor()) * 2.0 - 1.0
            })
            .collect();
        let bess = bulk_ess_single(&samples).expect("should succeed");
        assert!(bess > 0.0 && bess <= n as f64);
    }

    #[test]
    fn test_tail_ess_single() {
        let n = 500;
        let samples: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 * 0.618033988749;
                (x - x.floor()) * 2.0 - 1.0
            })
            .collect();
        let tess = tail_ess_single(&samples).expect("should succeed");
        assert!(tess > 0.0 && tess <= n as f64);
    }

    // ---------------------------------------------------------------
    // Multi-chain ESS tests
    // ---------------------------------------------------------------

    #[test]
    fn test_bulk_ess_multi_iid() {
        let c1 = iid_normal_chain(1000, 0.0, 1.0, 42);
        let c2 = iid_normal_chain(1000, 0.0, 1.0, 99);
        let ess = bulk_ess(&[c1, c2]).expect("bulk_ess failed");
        assert!(ess > 500.0, "bulk ESS for IID should be large, got {ess}");
    }

    #[test]
    fn test_bulk_ess_multi_autocorrelated() {
        let c1 = ar1_chain(2000, 0.95, 42);
        let c2 = ar1_chain(2000, 0.95, 99);
        let ess = bulk_ess(&[c1, c2]).expect("bulk_ess failed");
        assert!(
            ess < 2000.0,
            "bulk ESS for AR(0.95) should be reduced, got {ess}"
        );
    }

    #[test]
    fn test_tail_ess_multi_iid() {
        let c1 = iid_normal_chain(1000, 0.0, 1.0, 42);
        let c2 = iid_normal_chain(1000, 0.0, 1.0, 99);
        let ess = tail_ess(&[c1, c2]).expect("tail_ess failed");
        assert!(
            ess > 50.0,
            "tail ESS for IID should be reasonable, got {ess}"
        );
    }

    // ---------------------------------------------------------------
    // MCSE tests
    // ---------------------------------------------------------------

    #[test]
    fn test_mcse_single_basic() {
        let n = 500;
        let samples: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 * 0.618033988749;
                (x - x.floor()) * 2.0 - 1.0
            })
            .collect();
        let mcse_val = mcse(&samples).expect("should succeed");
        assert!(mcse_val > 0.0 && mcse_val < 1.0);
    }

    #[test]
    fn test_mcse_mean_multi() {
        let c1 = iid_normal_chain(2000, 0.0, 1.0, 42);
        let c2 = iid_normal_chain(2000, 0.0, 1.0, 99);
        let val = mcse_mean(&[c1, c2]).expect("mcse_mean failed");
        assert!(
            val < 0.1,
            "MCSE for IID N(0,1) with ~4000 draws should be small, got {val}"
        );
    }

    #[test]
    fn test_mcse_quantile_median() {
        let c1 = iid_normal_chain(2000, 0.0, 1.0, 42);
        let c2 = iid_normal_chain(2000, 0.0, 1.0, 99);
        let val = mcse_quantile(&[c1, c2], 0.5).expect("mcse_quantile failed");
        assert!(val < 0.5, "MCSE for median should be reasonable, got {val}");
    }

    #[test]
    fn test_mcse_quantile_invalid_prob() {
        let c1 = iid_normal_chain(100, 0.0, 1.0, 42);
        let c2 = iid_normal_chain(100, 0.0, 1.0, 99);
        assert!(mcse_quantile(&[c1, c2], 1.5).is_err());
    }

    // ---------------------------------------------------------------
    // Trace statistics tests
    // ---------------------------------------------------------------

    #[test]
    fn test_running_mean() {
        let chain = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rm = running_mean(&chain).expect("running_mean failed");
        assert!((rm[0] - 1.0).abs() < 1e-10);
        assert!((rm[1] - 1.5).abs() < 1e-10);
        assert!((rm[2] - 2.0).abs() < 1e-10);
        assert!((rm[3] - 2.5).abs() < 1e-10);
        assert!((rm[4] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_running_variance() {
        let chain = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let rv = running_variance(&chain).expect("running_variance failed");
        assert!((rv[0] - 0.0).abs() < 1e-10);
        assert!((rv[1] - 2.0).abs() < 1e-10, "var([2,4])=2.0, got {}", rv[1]);
        assert!(
            (rv[4] - 10.0).abs() < 1e-10,
            "var([2..10])=10.0, got {}",
            rv[4]
        );
    }

    #[test]
    fn test_running_mean_empty() {
        assert!(running_mean(&[]).is_err());
    }

    #[test]
    fn test_split_rhat_trace() {
        let c1 = iid_normal_chain(400, 0.0, 1.0, 42);
        let c2 = iid_normal_chain(400, 0.0, 1.0, 99);
        let trace = split_rhat_trace(&[c1, c2], 100).expect("rhat_trace failed");
        assert!(!trace.is_empty());
        for &rh in &trace {
            assert!(rh.is_finite(), "R-hat should be finite, got {rh}");
        }
    }

    // ---------------------------------------------------------------
    // Divergence diagnostics tests
    // ---------------------------------------------------------------

    #[test]
    fn test_divergence_diagnostics_none() {
        let flags = vec![false; 100];
        let diag = divergence_diagnostics(&flags);
        assert_eq!(diag.n_divergent, 0);
        assert!((diag.divergence_rate).abs() < 1e-10);
        assert!(diag.divergent_indices.is_empty());
    }

    #[test]
    fn test_divergence_diagnostics_some() {
        let mut flags = vec![false; 100];
        flags[10] = true;
        flags[20] = true;
        flags[50] = true;
        let diag = divergence_diagnostics(&flags);
        assert_eq!(diag.n_divergent, 3);
        assert!((diag.divergence_rate - 0.03).abs() < 1e-10);
        assert_eq!(diag.divergent_indices, vec![10, 20, 50]);
    }

    #[test]
    fn test_divergence_diagnostics_empty() {
        let diag = divergence_diagnostics(&[]);
        assert_eq!(diag.total_transitions, 0);
        assert_eq!(diag.n_divergent, 0);
    }

    // ---------------------------------------------------------------
    // Energy diagnostics tests
    // ---------------------------------------------------------------

    #[test]
    fn test_energy_bfmi_constant() {
        let energies = vec![5.0; 100];
        let bfmi = energy_bfmi(&energies).expect("energy_bfmi failed");
        assert!(
            (bfmi - 1.0).abs() < 1e-10,
            "constant energy -> BFMI=1.0, got {bfmi}"
        );
    }

    #[test]
    fn test_energy_bfmi_iid() {
        let energies = iid_normal_chain(1000, 10.0, 2.0, 42);
        let bfmi = energy_bfmi(&energies).expect("energy_bfmi failed");
        assert!(bfmi > 0.3, "IID energy -> BFMI > 0.3, got {bfmi}");
    }

    #[test]
    fn test_energy_bfmi_insufficient() {
        assert!(energy_bfmi(&[1.0]).is_err());
    }

    #[test]
    fn test_energy_diagnostics() {
        let energies = iid_normal_chain(500, 10.0, 2.0, 42);
        let diag = energy_diagnostics(&energies).expect("energy_diagnostics failed");
        assert!(diag.e_bfmi > 0.3);
        assert!(!diag.low_bfmi_warning);
        assert!(diag.energy_variance > 0.0);
    }

    // ---------------------------------------------------------------
    // Multi-parameter tests
    // ---------------------------------------------------------------

    #[test]
    fn test_rhat_per_parameter() {
        let n = 500;
        let chain1 = Array2::from_shape_fn((n, 3), |(i, j)| {
            let chains = [
                iid_normal_chain(n, 0.0, 1.0, 42),
                iid_normal_chain(n, 1.0, 2.0, 43),
                iid_normal_chain(n, -1.0, 0.5, 44),
            ];
            chains[j][i]
        });
        let chain2 = Array2::from_shape_fn((n, 3), |(i, j)| {
            let chains = [
                iid_normal_chain(n, 0.0, 1.0, 142),
                iid_normal_chain(n, 1.0, 2.0, 143),
                iid_normal_chain(n, -1.0, 0.5, 144),
            ];
            chains[j][i]
        });
        let rhats = rhat_per_parameter(&[&chain1, &chain2]).expect("rhat_per_parameter failed");
        assert_eq!(rhats.len(), 3);
        for p in 0..3 {
            assert!(rhats[p] < 1.1, "param {p} R-hat near 1.0, got {}", rhats[p]);
        }
    }

    // ---------------------------------------------------------------
    // DiagnosticReport tests
    // ---------------------------------------------------------------

    #[test]
    fn test_diagnostic_report_converged() {
        let n = 1000;
        let chain1 = Array2::from_shape_fn((n, 2), |(i, j)| {
            let chains = [
                iid_normal_chain(n, 0.0, 1.0, 10),
                iid_normal_chain(n, 2.0, 1.0, 11),
            ];
            chains[j][i]
        });
        let chain2 = Array2::from_shape_fn((n, 2), |(i, j)| {
            let chains = [
                iid_normal_chain(n, 0.0, 1.0, 20),
                iid_normal_chain(n, 2.0, 1.0, 21),
            ];
            chains[j][i]
        });

        let report = DiagnosticReport::new(
            &[&chain1, &chain2],
            Some(&["mu".to_string(), "sigma".to_string()]),
            None,
            None,
        )
        .expect("DiagnosticReport::new failed");

        assert_eq!(report.n_chains, 2);
        assert_eq!(report.n_samples, n);
        assert_eq!(report.parameters.len(), 2);
        assert_eq!(report.parameters[0].name.as_deref(), Some("mu"));

        let summary = report.summary();
        assert!(!summary.is_empty());
        // Should contain the parameter names
        assert!(summary.contains("mu"));
        assert!(summary.contains("sigma"));
    }

    #[test]
    fn test_diagnostic_report_with_divergences() {
        let n = 200;
        let chain1 = Array2::from_shape_fn((n, 1), |(i, _)| iid_normal_chain(n, 0.0, 1.0, 10)[i]);
        let chain2 = Array2::from_shape_fn((n, 1), |(i, _)| iid_normal_chain(n, 0.0, 1.0, 20)[i]);

        let mut div_flags = vec![false; 200];
        div_flags[50] = true;
        div_flags[100] = true;
        let energies = iid_normal_chain(200, 5.0, 1.0, 42);

        let report =
            DiagnosticReport::new(&[&chain1, &chain2], None, Some(&div_flags), Some(&energies))
                .expect("DiagnosticReport::new failed");

        assert!(report.has_divergences());
        assert!(!report.has_low_bfmi());
        assert_eq!(report.divergence.as_ref().map(|d| d.n_divergent), Some(2));

        let summary = report.summary();
        assert!(summary.contains("Divergences"));
        assert!(summary.contains("E-BFMI"));
    }

    #[test]
    fn test_diagnostic_report_single_chain_fallback() {
        let n = 500;
        let chain1 = Array2::from_shape_fn((n, 1), |(i, _)| iid_normal_chain(n, 0.0, 1.0, 10)[i]);

        let report = DiagnosticReport::new(&[&chain1], None, None, None)
            .expect("single-chain report should succeed");
        assert_eq!(report.n_chains, 1);
        assert_eq!(report.parameters.len(), 1);
        // Single-chain R-hat uses split_r_hat internally
        assert!(report.parameters[0].rhat.is_finite());
    }

    // ---------------------------------------------------------------
    // normal_quantile helper tests
    // ---------------------------------------------------------------

    #[test]
    fn test_normal_quantile() {
        assert!(normal_quantile(0.5).abs() < 1e-6, "Phi^-1(0.5)=0");
        assert!((normal_quantile(0.975) - 1.96).abs() < 0.01);
        assert!((normal_quantile(0.025) + 1.96).abs() < 0.01);
        assert!(
            (normal_quantile(0.1) + normal_quantile(0.9)).abs() < 1e-6,
            "symmetry"
        );
    }

    #[test]
    fn test_rank_normalize_single_basic() {
        let samples = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let ranked = rank_normalize_single(&samples);
        assert_eq!(ranked.len(), samples.len());
        let n = ranked.len() as f64;
        let mean: f64 = ranked.iter().sum::<f64>() / n;
        assert!(mean.abs() < 0.3, "rank-normalised mean near 0, got {mean}");
    }

    // ---------------------------------------------------------------
    // Edge cases & error paths
    // ---------------------------------------------------------------

    #[test]
    fn test_insufficient_data_errors() {
        assert!(autocorrelation(&[1.0], 5).is_err());
        assert!(effective_sample_size(&[1.0, 2.0]).is_err());
        assert!(mcse(&[1.0, 2.0]).is_err());
        assert!(split_r_hat(&[1.0, 2.0]).is_err());
        assert!(bulk_ess_single(&[1.0]).is_err());
        assert!(tail_ess_single(&(0..10).map(|i| i as f64).collect::<Vec<_>>()).is_err());
    }

    #[test]
    fn test_compute_acf_constant() {
        let x = vec![5.0; 100];
        let acf = compute_acf(&x, 50);
        for &v in &acf {
            assert!((v - 1.0).abs() < 1e-10, "constant signal ACF should be 1.0");
        }
    }
}
