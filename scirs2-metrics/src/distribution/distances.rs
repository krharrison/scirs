//! Statistical Distance Functions Between Probability Distributions
//!
//! This module provides a comprehensive set of distance and divergence measures
//! between discrete probability distributions and between empirical samples:
//!
//! - **Total Variation Distance**: `0.5 * Σ|p_i - q_i|`
//! - **Hellinger Distance**: `sqrt(1 - Σ sqrt(p_i * q_i))`
//! - **KL Divergence**: `Σ p_i * log(p_i / q_i)`
//! - **Jensen-Shannon Divergence**: symmetric, bounded variant of KL
//! - **Chi-Square Divergence**: `Σ (p_i - q_i)² / q_i`
//! - **Energy Distance**: sample-based `2E‖X-Y‖ - E‖X-X'‖ - E‖Y-Y'‖`

use super::types::{DistanceMethod, DistanceResult};
use crate::error::{MetricsError, Result};

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

/// Validate that both slices are non-empty and of equal length.
fn check_pmf(p: &[f64], q: &[f64]) -> Result<()> {
    if p.is_empty() || q.is_empty() {
        return Err(MetricsError::InvalidInput(
            "distribution arrays must not be empty".to_string(),
        ));
    }
    if p.len() != q.len() {
        return Err(MetricsError::DimensionMismatch(format!(
            "p has {} elements but q has {}",
            p.len(),
            q.len()
        )));
    }
    Ok(())
}

/// Validate that sample arrays are non-empty.
fn check_samples(x: &[f64], y: &[f64]) -> Result<()> {
    if x.is_empty() || y.is_empty() {
        return Err(MetricsError::InvalidInput(
            "sample arrays must not be empty".to_string(),
        ));
    }
    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────
// Total Variation Distance
// ────────────────────────────────────────────────────────────────────────────

/// Total Variation distance between two discrete probability distributions.
///
/// ```text
/// TV(P, Q) = 0.5 * Σ |p_i - q_i|
/// ```
///
/// The result is always in `[0, 1]` for valid probability distributions.
///
/// # Arguments
/// * `p` - first probability distribution (non-negative, should sum to ~1)
/// * `q` - second probability distribution (same length as p)
pub fn total_variation_distance(p: &[f64], q: &[f64]) -> Result<DistanceResult> {
    check_pmf(p, q)?;
    let tv = 0.5
        * p.iter()
            .zip(q.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>();
    Ok(DistanceResult::new(tv, DistanceMethod::TotalVariation))
}

// ────────────────────────────────────────────────────────────────────────────
// Hellinger Distance
// ────────────────────────────────────────────────────────────────────────────

/// Hellinger distance between two discrete probability distributions.
///
/// ```text
/// H(P, Q) = sqrt(1 - Σ sqrt(p_i * q_i))
/// ```
///
/// The result is always in `[0, 1]`.
///
/// # Arguments
/// * `p` - first probability distribution
/// * `q` - second probability distribution
pub fn hellinger_distance(p: &[f64], q: &[f64]) -> Result<DistanceResult> {
    check_pmf(p, q)?;
    let bc: f64 = p
        .iter()
        .zip(q.iter())
        .map(|(a, b)| (a.max(0.0) * b.max(0.0)).sqrt())
        .sum();
    let h = (1.0 - bc).max(0.0).sqrt();
    Ok(DistanceResult::new(h, DistanceMethod::Hellinger))
}

// ────────────────────────────────────────────────────────────────────────────
// KL Divergence
// ────────────────────────────────────────────────────────────────────────────

/// Kullback-Leibler divergence `KL(P || Q) = Σ p_i * log(p_i / q_i)`.
///
/// Returns an error if `q_i = 0` where `p_i > 0` (the divergence would be infinite).
/// Elements where `p_i = 0` contribute 0 by convention (0 * log 0 = 0).
///
/// By Gibbs' inequality, `KL(P || Q) >= 0` with equality iff P = Q.
///
/// # Arguments
/// * `p` - reference distribution
/// * `q` - approximating distribution
pub fn kl_divergence(p: &[f64], q: &[f64]) -> Result<DistanceResult> {
    check_pmf(p, q)?;
    let mut kl = 0.0f64;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi <= 0.0 {
            continue;
        }
        if qi <= 0.0 {
            return Err(MetricsError::CalculationError(
                "KL divergence is infinite: q_i = 0 where p_i > 0".to_string(),
            ));
        }
        kl += pi * (pi / qi).ln();
    }
    Ok(DistanceResult::new(kl, DistanceMethod::KullbackLeibler))
}

// ────────────────────────────────────────────────────────────────────────────
// Jensen-Shannon Divergence
// ────────────────────────────────────────────────────────────────────────────

/// Jensen-Shannon divergence (symmetric, bounded in `[0, ln 2]`).
///
/// ```text
/// JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
/// ```
/// where `M = 0.5 * (P + Q)`.
///
/// JSD is always symmetric: `JSD(P, Q) = JSD(Q, P)`.
///
/// # Arguments
/// * `p` - first distribution
/// * `q` - second distribution
pub fn jensen_shannon_divergence(p: &[f64], q: &[f64]) -> Result<DistanceResult> {
    check_pmf(p, q)?;
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(a, b)| 0.5 * (a + b)).collect();

    // KL(P || M) — M[i] > 0 whenever P[i] > 0 or Q[i] > 0
    let mut kl_pm = 0.0f64;
    for (&pi, &mi) in p.iter().zip(m.iter()) {
        if pi > 0.0 && mi > 0.0 {
            kl_pm += pi * (pi / mi).ln();
        }
    }

    let mut kl_qm = 0.0f64;
    for (&qi, &mi) in q.iter().zip(m.iter()) {
        if qi > 0.0 && mi > 0.0 {
            kl_qm += qi * (qi / mi).ln();
        }
    }

    let jsd = 0.5 * kl_pm + 0.5 * kl_qm;
    Ok(DistanceResult::new(
        jsd.max(0.0),
        DistanceMethod::JensenShannon,
    ))
}

// ────────────────────────────────────────────────────────────────────────────
// Chi-Square Divergence
// ────────────────────────────────────────────────────────────────────────────

/// Chi-square divergence: `Σ (p_i - q_i)² / q_i`.
///
/// This is the Pearson chi-squared divergence. Returns an error if
/// `q_i = 0` where `p_i > 0`.
///
/// # Arguments
/// * `p` - observed / reference distribution
/// * `q` - expected / baseline distribution
pub fn chi_square_divergence(p: &[f64], q: &[f64]) -> Result<DistanceResult> {
    check_pmf(p, q)?;
    let mut chi2 = 0.0f64;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if qi <= 0.0 {
            if pi > 1e-15 {
                return Err(MetricsError::CalculationError(
                    "chi-square divergence undefined: q_i = 0 where p_i > 0".to_string(),
                ));
            }
            continue;
        }
        let diff = pi - qi;
        chi2 += diff * diff / qi;
    }
    Ok(DistanceResult::new(chi2, DistanceMethod::ChiSquare))
}

// ────────────────────────────────────────────────────────────────────────────
// Energy Distance (sample-based)
// ────────────────────────────────────────────────────────────────────────────

/// Energy distance between two empirical distributions (sample-based).
///
/// ```text
/// E(P, Q) = 2 * E[|X - Y|] - E[|X - X'|] - E[|Y - Y'|]
/// ```
///
/// This is a metric: non-negative, symmetric, satisfies triangle inequality,
/// and equals 0 iff P = Q.
///
/// Uses U-statistic estimators. Complexity: O(n*m + n² + m²).
///
/// # Arguments
/// * `x` - samples from first distribution
/// * `y` - samples from second distribution
pub fn energy_distance(x: &[f64], y: &[f64]) -> Result<DistanceResult> {
    check_samples(x, y)?;

    let cross = cross_mean_abs(x, y);
    let within_x = within_mean_abs(x);
    let within_y = within_mean_abs(y);

    let ed = (2.0 * cross - within_x - within_y).max(0.0);
    Ok(DistanceResult::new(ed, DistanceMethod::Energy))
}

/// E[|X - Y|] via all-pairs computation.
fn cross_mean_abs(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    let m = y.len();
    if n == 0 || m == 0 {
        return 0.0;
    }
    let mut s = 0.0f64;
    for &xi in x {
        for &yj in y {
            s += (xi - yj).abs();
        }
    }
    s / (n as f64 * m as f64)
}

/// E[|X - X'|] using sorted array optimization.
fn within_mean_abs(x: &[f64]) -> f64 {
    let n = x.len();
    if n <= 1 {
        return 0.0;
    }
    let mut sorted = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut prefix = 0.0f64;
    let mut total = 0.0f64;
    for (j, &xj) in sorted.iter().enumerate() {
        total += xj * j as f64 - prefix;
        prefix += xj;
    }

    let pairs = n as f64 * (n as f64 - 1.0) / 2.0;
    if pairs > 0.0 {
        total / pairs
    } else {
        0.0
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Sample-based versions of discrete distances
// ────────────────────────────────────────────────────────────────────────────

/// Estimate the KL divergence between two continuous distributions from
/// samples using kernel density estimation.
///
/// This uses a simple histogram approach: bin both sample sets into
/// `n_bins` equal-width bins and compute discrete KL divergence on the
/// resulting histograms (with additive smoothing to avoid zeros).
///
/// # Arguments
/// * `x` - samples from distribution P
/// * `y` - samples from distribution Q
/// * `n_bins` - number of histogram bins (recommended: 50-200)
pub fn kl_divergence_samples(x: &[f64], y: &[f64], n_bins: usize) -> Result<DistanceResult> {
    check_samples(x, y)?;
    if n_bins == 0 {
        return Err(MetricsError::InvalidInput("n_bins must be > 0".to_string()));
    }

    let (p_hist, q_hist) = build_histograms(x, y, n_bins)?;
    kl_divergence(&p_hist, &q_hist)
}

/// Estimate the Jensen-Shannon divergence from samples.
///
/// Uses histogram binning with additive smoothing.
pub fn jensen_shannon_divergence_samples(
    x: &[f64],
    y: &[f64],
    n_bins: usize,
) -> Result<DistanceResult> {
    check_samples(x, y)?;
    if n_bins == 0 {
        return Err(MetricsError::InvalidInput("n_bins must be > 0".to_string()));
    }

    let (p_hist, q_hist) = build_histograms(x, y, n_bins)?;
    jensen_shannon_divergence(&p_hist, &q_hist)
}

/// Build normalized histograms from two sample sets.
///
/// Uses the combined range with additive (Laplace) smoothing.
fn build_histograms(x: &[f64], y: &[f64], n_bins: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    // Find range over combined samples
    let all_min = x
        .iter()
        .chain(y.iter())
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let all_max = x
        .iter()
        .chain(y.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    if (all_max - all_min).abs() < f64::EPSILON {
        // All values are the same
        let p = vec![1.0];
        return Ok((p.clone(), p));
    }

    let bin_width = (all_max - all_min) / n_bins as f64;
    let smooth = 1e-10; // additive smoothing

    let mut p_counts = vec![smooth; n_bins];
    let mut q_counts = vec![smooth; n_bins];

    for &xi in x {
        let bin = ((xi - all_min) / bin_width).floor() as usize;
        let bin = bin.min(n_bins - 1);
        p_counts[bin] += 1.0;
    }
    for &yi in y {
        let bin = ((yi - all_min) / bin_width).floor() as usize;
        let bin = bin.min(n_bins - 1);
        q_counts[bin] += 1.0;
    }

    // Normalize
    let p_sum: f64 = p_counts.iter().sum();
    let q_sum: f64 = q_counts.iter().sum();
    let p_hist: Vec<f64> = p_counts.iter().map(|&v| v / p_sum).collect();
    let q_hist: Vec<f64> = q_counts.iter().map(|&v| v / q_sum).collect();

    Ok((p_hist, q_hist))
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_total_variation_in_range() {
        let p = vec![0.3, 0.4, 0.3];
        let q = vec![0.1, 0.7, 0.2];
        let result = total_variation_distance(&p, &q).expect("should succeed");
        let tv = result.value;
        assert!((0.0..=1.0).contains(&tv), "TV must be in [0,1], got {tv}");
        assert_eq!(result.method, DistanceMethod::TotalVariation);
    }

    #[test]
    fn test_total_variation_identical_is_zero() {
        let p = vec![0.5, 0.3, 0.2];
        let result = total_variation_distance(&p, &p).expect("should succeed");
        assert!(
            result.value.abs() < 1e-12,
            "TV(P,P) should be 0, got {}",
            result.value
        );
    }

    #[test]
    fn test_hellinger_in_range() {
        let p = vec![0.5, 0.3, 0.2];
        let q = vec![0.2, 0.5, 0.3];
        let result = hellinger_distance(&p, &q).expect("should succeed");
        let h = result.value;
        assert!(
            (0.0..=1.0).contains(&h),
            "Hellinger must be in [0,1], got {h}"
        );
    }

    #[test]
    fn test_hellinger_identical_is_zero() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let result = hellinger_distance(&p, &p).expect("should succeed");
        assert!(
            result.value.abs() < 1e-12,
            "H(P,P) should be 0, got {}",
            result.value
        );
    }

    #[test]
    fn test_kl_nonnegative() {
        let p = vec![0.4, 0.3, 0.3];
        let q = vec![0.2, 0.5, 0.3];
        let result = kl_divergence(&p, &q).expect("should succeed");
        assert!(
            result.value >= -1e-12,
            "KL must be >= 0 (Gibbs inequality), got {}",
            result.value
        );
    }

    #[test]
    fn test_kl_identical_is_zero() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let result = kl_divergence(&p, &p).expect("should succeed");
        assert!(
            result.value.abs() < 1e-12,
            "KL(P,P) should be 0, got {}",
            result.value
        );
    }

    #[test]
    fn test_kl_zero_q_errors() {
        let p = vec![0.5, 0.5];
        let q = vec![1.0, 0.0];
        assert!(kl_divergence(&p, &q).is_err());
    }

    #[test]
    fn test_jsd_symmetric() {
        let p = vec![0.3, 0.4, 0.3];
        let q = vec![0.1, 0.7, 0.2];
        let jsd_pq = jensen_shannon_divergence(&p, &q)
            .expect("should succeed")
            .value;
        let jsd_qp = jensen_shannon_divergence(&q, &p)
            .expect("should succeed")
            .value;
        assert!(
            (jsd_pq - jsd_qp).abs() < 1e-12,
            "JSD must be symmetric: {jsd_pq} vs {jsd_qp}"
        );
    }

    #[test]
    fn test_jsd_bounded() {
        use std::f64::consts::LN_2;
        let p = vec![0.3, 0.4, 0.3];
        let q = vec![0.1, 0.8, 0.1];
        let jsd = jensen_shannon_divergence(&p, &q)
            .expect("should succeed")
            .value;
        assert!(jsd >= 0.0, "JSD >= 0, got {jsd}");
        assert!(jsd <= LN_2 + 1e-12, "JSD <= ln(2), got {jsd}");
    }

    #[test]
    fn test_chi_square_nonnegative() {
        let p = vec![0.4, 0.3, 0.3];
        let q = vec![0.2, 0.5, 0.3];
        let result = chi_square_divergence(&p, &q).expect("should succeed");
        assert!(
            result.value >= 0.0,
            "chi-squared must be >= 0, got {}",
            result.value
        );
    }

    #[test]
    fn test_chi_square_identical_is_zero() {
        let p = vec![0.3, 0.4, 0.3];
        let result = chi_square_divergence(&p, &p).expect("should succeed");
        assert!(
            result.value.abs() < 1e-12,
            "chi²(P,P) should be 0, got {}",
            result.value
        );
    }

    #[test]
    fn test_energy_distance_nonnegative() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![10.0, 11.0, 12.0];
        let result = energy_distance(&x, &y).expect("should succeed");
        assert!(
            result.value >= 0.0,
            "energy distance must be >= 0, got {}",
            result.value
        );
    }

    #[test]
    fn test_energy_distance_identical_is_zero() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = energy_distance(&x, &x).expect("should succeed");
        assert!(
            result.value.abs() < 1e-10,
            "E(P,P) should be 0, got {}",
            result.value
        );
    }

    #[test]
    fn test_energy_distance_symmetric() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        let ed_xy = energy_distance(&x, &y).expect("should succeed").value;
        let ed_yx = energy_distance(&y, &x).expect("should succeed").value;
        assert!(
            (ed_xy - ed_yx).abs() < 1e-10,
            "energy distance must be symmetric"
        );
    }

    #[test]
    fn test_kl_divergence_samples() {
        // Samples from same distribution should have small KL
        let x: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
        let y: Vec<f64> = (0..100).map(|i| i as f64 * 0.01 + 0.005).collect();
        let result = kl_divergence_samples(&x, &y, 50).expect("should succeed");
        assert!(
            result.value >= -1e-10,
            "sample KL must be >= 0, got {}",
            result.value
        );
    }

    #[test]
    fn test_jsd_samples_symmetric() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let jsd_xy = jensen_shannon_divergence_samples(&x, &y, 20)
            .expect("should succeed")
            .value;
        let jsd_yx = jensen_shannon_divergence_samples(&y, &x, 20)
            .expect("should succeed")
            .value;
        assert!(
            (jsd_xy - jsd_yx).abs() < 1e-10,
            "sample JSD must be symmetric: {jsd_xy} vs {jsd_yx}"
        );
    }

    #[test]
    fn test_dimension_mismatch_errors() {
        let p = vec![0.5, 0.5];
        let q = vec![0.3, 0.3, 0.4];
        assert!(total_variation_distance(&p, &q).is_err());
        assert!(hellinger_distance(&p, &q).is_err());
        assert!(kl_divergence(&p, &q).is_err());
        assert!(jensen_shannon_divergence(&p, &q).is_err());
        assert!(chi_square_divergence(&p, &q).is_err());
    }

    #[test]
    fn test_empty_errors() {
        let empty: Vec<f64> = vec![];
        let p = vec![1.0];
        assert!(total_variation_distance(&empty, &p).is_err());
        assert!(energy_distance(&empty, &p).is_err());
    }
}
