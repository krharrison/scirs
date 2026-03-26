//! Wasserstein Distance (Earth Mover's Distance)
//!
//! Computes optimal transport distances between probability distributions.
//!
//! # Algorithms
//!
//! - **1D Wasserstein (p=1, p=2)**: Exact computation via sorting
//! - **Earth Mover's Distance**: Equivalent to W1
//! - **Multi-dimensional Wasserstein**: Simplified Kantorovich via LP relaxation
//! - **Empirical Wasserstein**: From raw samples
//!
//! # Examples
//!
//! ```
//! use scirs2_metrics::distributional::wasserstein::{wasserstein_1d, earth_movers_distance};
//!
//! let a = vec![1.0, 2.0, 3.0];
//! let b = vec![4.0, 5.0, 6.0];
//! let dist = wasserstein_1d(&a, &b, 1).expect("should succeed");
//! assert!((dist - 3.0).abs() < 1e-10);
//!
//! let emd = earth_movers_distance(&a, &b).expect("should succeed");
//! assert!((emd - 3.0).abs() < 1e-10);
//! ```

use crate::error::{MetricsError, Result};

/// Computes the 1D Wasserstein distance of order p between two empirical distributions.
///
/// For 1D distributions, the Wasserstein distance has a closed-form solution
/// via sorting: W_p(u, v) = ( (1/n) * sum |u_sorted_i - v_sorted_i|^p )^(1/p)
///
/// When the sample sizes differ, we use the quantile-based formula by
/// interpolating along the CDF.
///
/// # Arguments
///
/// * `a` - First set of samples
/// * `b` - Second set of samples
/// * `p` - Order of the Wasserstein distance (1 or 2)
///
/// # Returns
///
/// The p-Wasserstein distance between the two empirical distributions.
pub fn wasserstein_1d(a: &[f64], b: &[f64], p: u32) -> Result<f64> {
    if a.is_empty() || b.is_empty() {
        return Err(MetricsError::InvalidInput(
            "input samples must not be empty".to_string(),
        ));
    }
    if p == 0 {
        return Err(MetricsError::InvalidInput("p must be >= 1".to_string()));
    }

    let mut a_sorted: Vec<f64> = a.to_vec();
    let mut b_sorted: Vec<f64> = b.to_vec();
    a_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    b_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));

    if a_sorted.len() == b_sorted.len() {
        // Direct pairing of sorted samples
        let n = a_sorted.len();
        let p_f64 = f64::from(p);
        let sum: f64 = a_sorted
            .iter()
            .zip(b_sorted.iter())
            .map(|(ai, bi)| (ai - bi).abs().powf(p_f64))
            .sum();
        Ok((sum / n as f64).powf(1.0 / p_f64))
    } else {
        // Use quantile interpolation for different-sized samples
        wasserstein_1d_quantile(&a_sorted, &b_sorted, p)
    }
}

/// Wasserstein distance via quantile function interpolation (for unequal sample sizes).
///
/// We merge the CDF breakpoints of both distributions and integrate the
/// |F_a^{-1}(t) - F_b^{-1}(t)|^p over [0, 1].
fn wasserstein_1d_quantile(a_sorted: &[f64], b_sorted: &[f64], p: u32) -> Result<f64> {
    let na = a_sorted.len();
    let nb = b_sorted.len();
    let p_f64 = f64::from(p);

    // Merge all quantile breakpoints from both CDFs
    let mut breakpoints: Vec<f64> = Vec::with_capacity(na + nb + 2);
    breakpoints.push(0.0);
    for i in 1..=na {
        breakpoints.push(i as f64 / na as f64);
    }
    for i in 1..=nb {
        breakpoints.push(i as f64 / nb as f64);
    }
    breakpoints.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    breakpoints.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    let mut integral = 0.0;

    for w in breakpoints.windows(2) {
        let t_left = w[0];
        let t_right = w[1];
        let t_mid = (t_left + t_right) / 2.0;
        let dt = t_right - t_left;

        let qa = quantile_from_sorted(a_sorted, t_mid);
        let qb = quantile_from_sorted(b_sorted, t_mid);

        integral += (qa - qb).abs().powf(p_f64) * dt;
    }

    Ok(integral.powf(1.0 / p_f64))
}

/// Computes the quantile function (inverse CDF) for sorted samples at a given probability t.
fn quantile_from_sorted(sorted: &[f64], t: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if t <= 0.0 {
        return sorted[0];
    }
    if t >= 1.0 {
        return sorted[n - 1];
    }

    // Index into sorted array: the i-th sample corresponds to CDF value i/n
    let idx_f = t * n as f64 - 0.5;
    let idx_low = idx_f.floor().max(0.0) as usize;
    let idx_high = (idx_low + 1).min(n - 1);

    if idx_low == idx_high {
        return sorted[idx_low];
    }

    let frac = idx_f - idx_low as f64;
    sorted[idx_low] * (1.0 - frac) + sorted[idx_high] * frac
}

/// Computes the Earth Mover's Distance (EMD) between two empirical 1D distributions.
///
/// The EMD is equivalent to the 1-Wasserstein distance for 1D distributions.
///
/// # Arguments
///
/// * `a` - First set of samples
/// * `b` - Second set of samples
///
/// # Returns
///
/// The EMD (= W1 distance).
pub fn earth_movers_distance(a: &[f64], b: &[f64]) -> Result<f64> {
    wasserstein_1d(a, b, 1)
}

/// Computes the Wasserstein distance between two weighted discrete distributions.
///
/// Given supports `x_a`, `x_b` and weight vectors `w_a`, `w_b`
/// (which must sum to 1.0 each), computes the 1D W1 distance by
/// the CDF-difference method.
///
/// # Arguments
///
/// * `x_a` - Support points of distribution a
/// * `w_a` - Weights for distribution a (must sum to ~1.0)
/// * `x_b` - Support points of distribution b
/// * `w_b` - Weights for distribution b (must sum to ~1.0)
///
/// # Returns
///
/// The 1-Wasserstein distance between the two weighted distributions.
pub fn wasserstein_1d_weighted(x_a: &[f64], w_a: &[f64], x_b: &[f64], w_b: &[f64]) -> Result<f64> {
    if x_a.is_empty() || x_b.is_empty() {
        return Err(MetricsError::InvalidInput(
            "support points must not be empty".to_string(),
        ));
    }
    if x_a.len() != w_a.len() || x_b.len() != w_b.len() {
        return Err(MetricsError::InvalidInput(
            "support and weight vectors must have the same length".to_string(),
        ));
    }

    let sum_a: f64 = w_a.iter().sum();
    let sum_b: f64 = w_b.iter().sum();
    if (sum_a - 1.0).abs() > 0.01 || (sum_b - 1.0).abs() > 0.01 {
        return Err(MetricsError::InvalidInput(
            "weights must sum to approximately 1.0".to_string(),
        ));
    }

    // Merge all support points and compute CDF difference
    let mut events: Vec<(f64, f64, f64)> = Vec::with_capacity(x_a.len() + x_b.len());
    for i in 0..x_a.len() {
        events.push((x_a[i], w_a[i], 0.0));
    }
    for i in 0..x_b.len() {
        events.push((x_b[i], 0.0, w_b[i]));
    }
    events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut cdf_a: f64 = 0.0;
    let mut cdf_b: f64 = 0.0;
    let mut distance: f64 = 0.0;
    let mut prev_x = events[0].0;

    for &(x, wa, wb) in &events {
        let dx = x - prev_x;
        distance += (cdf_a - cdf_b).abs() * dx;
        cdf_a += wa;
        cdf_b += wb;
        prev_x = x;
    }

    Ok(distance)
}

/// Computes the multi-dimensional Wasserstein distance between two sets of samples
/// using a greedy approximation (nearest-neighbor matching).
///
/// This is an approximation to the full optimal transport problem. For exact
/// solutions, use the Sinkhorn module with small epsilon.
///
/// # Arguments
///
/// * `a` - First set of samples, shape: [n, d] (flattened row-major)
/// * `b` - Second set of samples, shape: [m, d] (flattened row-major)
/// * `dim` - Dimensionality of each sample
/// * `p` - Order of the Wasserstein distance
///
/// # Returns
///
/// Approximate p-Wasserstein distance.
pub fn wasserstein_nd(a: &[f64], b: &[f64], dim: usize, p: u32) -> Result<f64> {
    if dim == 0 {
        return Err(MetricsError::InvalidInput(
            "dimension must be > 0".to_string(),
        ));
    }
    if a.is_empty() || b.is_empty() {
        return Err(MetricsError::InvalidInput(
            "input samples must not be empty".to_string(),
        ));
    }
    if a.len() % dim != 0 || b.len() % dim != 0 {
        return Err(MetricsError::InvalidInput(format!(
            "sample arrays must be divisible by dim={dim}"
        )));
    }

    let na = a.len() / dim;
    let nb = b.len() / dim;

    if na != nb {
        return Err(MetricsError::InvalidInput(
            "multi-dimensional Wasserstein requires equal sample sizes".to_string(),
        ));
    }
    let n = na;
    let p_f64 = f64::from(p);

    // Compute full cost matrix
    let mut cost = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut dist_sq = 0.0;
            for d in 0..dim {
                let diff = a[i * dim + d] - b[j * dim + d];
                dist_sq += diff * diff;
            }
            cost[i * n + j] = dist_sq.sqrt().powf(p_f64);
        }
    }

    // Greedy assignment: for each row, pick the cheapest unmatched column
    let mut used = vec![false; n];
    let mut total_cost = 0.0;

    for i in 0..n {
        let mut best_j = 0;
        let mut best_cost = f64::MAX;
        for j in 0..n {
            if !used[j] && cost[i * n + j] < best_cost {
                best_cost = cost[i * n + j];
                best_j = j;
            }
        }
        used[best_j] = true;
        total_cost += best_cost;
    }

    Ok((total_cost / n as f64).powf(1.0 / p_f64))
}

/// Computes the Wasserstein distance between two empirical distributions
/// given as raw sample vectors.
///
/// This is a convenience wrapper around `wasserstein_1d` with p=1.
///
/// # Arguments
///
/// * `samples_a` - Samples from distribution A
/// * `samples_b` - Samples from distribution B
///
/// # Returns
///
/// The 1-Wasserstein distance.
pub fn wasserstein_from_samples(samples_a: &[f64], samples_b: &[f64]) -> Result<f64> {
    wasserstein_1d(samples_a, samples_b, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasserstein_1d_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let d = wasserstein_1d(&a, &b, 1).expect("should succeed");
        assert!(
            d.abs() < 1e-10,
            "identical distributions should have distance 0, got {d}"
        );
    }

    #[test]
    fn test_wasserstein_1d_shifted() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![3.0, 4.0, 5.0, 6.0, 7.0];
        let d = wasserstein_1d(&a, &b, 1).expect("should succeed");
        assert!(
            (d - 2.0).abs() < 1e-10,
            "shift by 2 should give W1=2.0, got {d}"
        );
    }

    #[test]
    fn test_wasserstein_1d_p2() {
        let a = vec![0.0, 0.0, 0.0, 0.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let d = wasserstein_1d(&a, &b, 2).expect("should succeed");
        assert!(
            (d - 1.0).abs() < 1e-10,
            "uniform shift by 1 should give W2=1.0, got {d}"
        );
    }

    #[test]
    fn test_wasserstein_1d_empty() {
        assert!(wasserstein_1d(&[], &[1.0], 1).is_err());
        assert!(wasserstein_1d(&[1.0], &[], 1).is_err());
    }

    #[test]
    fn test_wasserstein_1d_p_zero() {
        assert!(wasserstein_1d(&[1.0], &[2.0], 0).is_err());
    }

    #[test]
    fn test_emd_equals_w1() {
        let a = vec![1.0, 3.0, 5.0];
        let b = vec![2.0, 4.0, 6.0];
        let emd = earth_movers_distance(&a, &b).expect("should succeed");
        let w1 = wasserstein_1d(&a, &b, 1).expect("should succeed");
        assert!(
            (emd - w1).abs() < 1e-10,
            "EMD should equal W1, got emd={emd}, w1={w1}"
        );
    }

    #[test]
    fn test_wasserstein_1d_unequal_sizes() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.5, 2.5, 3.5, 4.5];
        let d = wasserstein_1d(&a, &b, 1).expect("should succeed");
        assert!(
            d > 0.0,
            "different distributions should have positive distance"
        );
    }

    #[test]
    fn test_wasserstein_weighted() {
        let x_a = vec![0.0, 1.0];
        let w_a = vec![0.5, 0.5];
        let x_b = vec![1.0, 2.0];
        let w_b = vec![0.5, 0.5];
        let d = wasserstein_1d_weighted(&x_a, &w_a, &x_b, &w_b).expect("should succeed");
        assert!(
            (d - 1.0).abs() < 1e-10,
            "shift by 1 weighted should give 1.0, got {d}"
        );
    }

    #[test]
    fn test_wasserstein_weighted_bad_weights() {
        let x_a = vec![0.0];
        let w_a = vec![2.0]; // doesn't sum to 1
        let x_b = vec![1.0];
        let w_b = vec![1.0];
        assert!(wasserstein_1d_weighted(&x_a, &w_a, &x_b, &w_b).is_err());
    }

    #[test]
    fn test_wasserstein_nd_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let d = wasserstein_nd(&a, &b, 2, 1).expect("should succeed");
        assert!(d.abs() < 1e-10, "identical should give 0, got {d}");
    }

    #[test]
    fn test_wasserstein_nd_shifted() {
        let a = vec![0.0, 0.0, 1.0, 0.0];
        let b = vec![1.0, 0.0, 2.0, 0.0];
        let d = wasserstein_nd(&a, &b, 2, 1).expect("should succeed");
        assert!(
            (d - 1.0).abs() < 1e-10,
            "shift by 1 in x-dim should give W1=1.0, got {d}"
        );
    }

    #[test]
    fn test_wasserstein_from_samples() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 3.0, 4.0];
        let d = wasserstein_from_samples(&a, &b).expect("should succeed");
        assert!(
            (d - 1.0).abs() < 1e-10,
            "shift by 1 should give W1=1.0, got {d}"
        );
    }
}
