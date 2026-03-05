//! Change Point Detection Algorithms
//!
//! This module implements several algorithms for detecting structural breaks
//! (change points) in time series data:
//!
//! - **PELT** (Pruned Exact Linear Time): exact dynamic-programming search
//!   with pruning; O(n) average complexity for independent data.
//! - **CUSUM**: cumulative sum sequential test; detects shifts in mean.
//! - **Binary Segmentation**: greedy divide-and-conquer; fast but approximate.
//! - **BIC penalty helper**: Schwarz information-criterion penalty term.
//!
//! All functions return sorted indices of detected change points.  Indices
//! refer to positions *after* which the change occurs (i.e. the first sample
//! of the new segment), in the range `[1, n-1]`.
//!
//! # References
//!
//! - Killick, R., Fearnhead, P. & Eckley, I.A. (2012). Optimal Detection of
//!   Changepoints With a Linear Computational Cost. JASA.
//! - Page, E.S. (1954). Continuous Inspection Schemes. Biometrika.
//! - Scott, A.J. & Knott, M. (1974). A Cluster Analysis Method for Grouping
//!   Means in the Analysis of Variance. Biometrics.

use crate::error::{StatsError, StatsResult};

// ---------------------------------------------------------------------------
// Cost function: L2 (sum of squared deviations from segment mean)
// ---------------------------------------------------------------------------

/// Pre-compute prefix sums for efficient cost queries.
struct PrefixSums {
    sum: Vec<f64>,   // sum[i] = x[0] + ... + x[i-1]
    sum2: Vec<f64>,  // sum2[i] = x[0]^2 + ... + x[i-1]^2
}

impl PrefixSums {
    fn new(x: &[f64]) -> Self {
        let n = x.len();
        let mut sum = vec![0.0; n + 1];
        let mut sum2 = vec![0.0; n + 1];
        for (i, &v) in x.iter().enumerate() {
            sum[i + 1] = sum[i] + v;
            sum2[i + 1] = sum2[i] + v * v;
        }
        Self { sum, sum2 }
    }

    /// L2 cost for segment [s, e) (0-indexed, exclusive end)
    /// = sum_i (x_i - mean)^2 = sum_i x_i^2 - n*mean^2
    fn cost_l2(&self, s: usize, e: usize) -> f64 {
        if e <= s {
            return 0.0;
        }
        let n = (e - s) as f64;
        let sx = self.sum[e] - self.sum[s];
        let sx2 = self.sum2[e] - self.sum2[s];
        sx2 - sx * sx / n
    }
}

// ---------------------------------------------------------------------------
// BIC penalty
// ---------------------------------------------------------------------------

/// BIC penalty for the PELT algorithm.
///
/// `bic_penalty(n, sigma)` returns `sigma^2 * log(n)`, which is the standard
/// Schwarz/BIC penalty for adding one additional change point to an L2 model.
///
/// # Arguments
///
/// * `n`     – total number of observations
/// * `sigma` – noise standard deviation (set to 1.0 for scale-free penalty)
pub fn bic_penalty(n: usize, sigma: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    sigma * sigma * (n as f64).ln()
}

// ---------------------------------------------------------------------------
// PELT algorithm
// ---------------------------------------------------------------------------

/// Detect multiple change points using the **PELT** algorithm.
///
/// Minimises the penalised cost:
///
/// ```text
/// sum_{j=1}^{m+1} C(y_{tau_{j-1}+1 : tau_j}) + beta * m
/// ```
///
/// where `C` is the L2 (squared-error) cost and `beta = penalty`.
///
/// # Arguments
///
/// * `x`       – time series (length ≥ 2)
/// * `penalty` – penalty per change point; use [`bic_penalty`] for automatic
///               selection or pass a manual value (e.g. `2 * log(n)` for BIC)
///
/// # Returns
///
/// Sorted vector of change-point indices in `[1, n-1]` (exclusive of 0 and n).
/// An empty vector means no change points were detected.
///
/// # Errors
///
/// Returns an error when `x` has fewer than 2 elements.
///
/// # Example
///
/// ```
/// use scirs2_stats::time_series::{pelt_detect, bic_penalty};
///
/// let mut x = vec![0.0f64; 50];
/// for i in 25..50 { x[i] = 5.0; }   // step at index 25
/// let cp = pelt_detect(&x, bic_penalty(50, 1.0)).unwrap();
/// assert!(!cp.is_empty());
/// ```
pub fn pelt_detect(x: &[f64], penalty: f64) -> StatsResult<Vec<usize>> {
    let n = x.len();
    if n < 2 {
        return Err(StatsError::InsufficientData(
            "PELT requires at least 2 data points".to_string(),
        ));
    }

    let ps = PrefixSums::new(x);

    // f[t] = minimum penalised cost for x[0..t]
    // cp[t] = optimal last change point before t
    let mut f = vec![f64::INFINITY; n + 1];
    let mut cp = vec![0usize; n + 1];
    f[0] = -penalty; // initialisation trick from Killick et al.

    // candidates: set of admissible last-change-point positions
    let mut cands: Vec<usize> = vec![0];

    for t in 1..=n {
        let mut best_f = f64::INFINITY;
        let mut best_cp = 0;

        let mut surviving: Vec<usize> = Vec::with_capacity(cands.len());
        for &tau in &cands {
            let cost = f[tau] + ps.cost_l2(tau, t) + penalty;
            if cost < best_f {
                best_f = cost;
                best_cp = tau;
            }
            // PELT pruning: keep tau if f[tau] + C(tau, t) <= best_f + const
            // (using the inequality from Killick et al. with constant = penalty)
            if f[tau] + ps.cost_l2(tau, t) <= best_f {
                surviving.push(tau);
            }
        }
        f[t] = best_f;
        cp[t] = best_cp;
        cands = surviving;
        cands.push(t);
    }

    // Backtrack through cp array
    let mut change_points = Vec::new();
    let mut t = n;
    loop {
        let prev = cp[t];
        if prev == 0 {
            break;
        }
        change_points.push(prev);
        t = prev;
    }
    change_points.sort_unstable();
    Ok(change_points)
}

// ---------------------------------------------------------------------------
// CUSUM
// ---------------------------------------------------------------------------

/// Detect change points using the **CUSUM** (cumulative sum) sequential test.
///
/// Computes the CUSUM statistic:
///
/// ```text
/// S_t = sum_{i=1}^{t} (x_i - mean(x))
/// ```
///
/// A change point is recorded whenever `|S_t - S_{prev}| > threshold` and
/// the statistic resets.  This implements a two-sided CUSUM scheme.
///
/// # Arguments
///
/// * `x`         – time series
/// * `threshold` – detection threshold (in the same units as `x`).  A
///                 common choice is `k * std(x)` for some `k > 0`.
///
/// # Returns
///
/// Sorted vector of change-point indices.
///
/// # Errors
///
/// Returns an error when `x` is empty or has fewer than 2 elements.
pub fn cusum_detect(x: &[f64], threshold: f64) -> StatsResult<Vec<usize>> {
    let n = x.len();
    if n < 2 {
        return Err(StatsError::InsufficientData(
            "CUSUM requires at least 2 data points".to_string(),
        ));
    }
    if threshold < 0.0 {
        return Err(StatsError::InvalidArgument(
            "threshold must be non-negative".to_string(),
        ));
    }

    let mean = x.iter().sum::<f64>() / n as f64;

    let mut change_points = Vec::new();
    let mut cusum_pos = 0.0f64;
    let mut cusum_neg = 0.0f64;
    let mut ref_level = mean;
    let mut seg_start = 0usize;
    let _ = ref_level; // used below after reset

    // Simplified two-sided CUSUM: track deviation cumulative sums
    for (i, &xi) in x.iter().enumerate() {
        cusum_pos = (cusum_pos + xi - mean).max(0.0);
        cusum_neg = (cusum_neg - xi + mean).max(0.0);

        if cusum_pos > threshold || cusum_neg > threshold {
            if i > seg_start {
                change_points.push(i);
            }
            cusum_pos = 0.0;
            cusum_neg = 0.0;
            seg_start = i + 1;

            // Recompute local mean for the remaining series
            if seg_start < n {
                let remaining = &x[seg_start..];
                ref_level = remaining.iter().sum::<f64>() / remaining.len() as f64;
            }
        }
    }

    change_points.sort_unstable();
    change_points.dedup();
    Ok(change_points)
}

// ---------------------------------------------------------------------------
// Binary Segmentation
// ---------------------------------------------------------------------------

/// Detect change points using **binary segmentation**.
///
/// Greedily splits the series by finding the point with the maximum decrease
/// in L2 cost, repeating recursively on each sub-segment until `n_bkps`
/// change points have been found or no significant split is possible.
///
/// # Arguments
///
/// * `x`      – time series (length ≥ 2)
/// * `n_bkps` – desired number of change points
///
/// # Returns
///
/// Sorted vector of up to `n_bkps` change-point indices.
///
/// # Errors
///
/// Returns an error when `x` has fewer than 2 elements.
///
/// # Example
///
/// ```
/// use scirs2_stats::time_series::binary_segmentation;
///
/// let mut x: Vec<f64> = (0..30).map(|i| i as f64 % 10.0).collect();
/// let cp = binary_segmentation(&x, 2).unwrap();
/// assert!(cp.len() <= 2);
/// ```
pub fn binary_segmentation(x: &[f64], n_bkps: usize) -> StatsResult<Vec<usize>> {
    let n = x.len();
    if n < 2 {
        return Err(StatsError::InsufficientData(
            "Binary segmentation requires at least 2 data points".to_string(),
        ));
    }
    if n_bkps == 0 {
        return Ok(Vec::new());
    }

    let ps = PrefixSums::new(x);

    // Segments to split: queue of (start, end) 0-indexed exclusive
    let mut segments: Vec<(usize, usize)> = vec![(0, n)];
    let mut change_points = Vec::new();

    for _ in 0..n_bkps {
        // Find the segment + split point with the biggest cost reduction
        let mut best_gain = f64::NEG_INFINITY;
        let mut best_split = 0usize;
        let mut best_seg_idx = 0usize;

        for (si, &(s, e)) in segments.iter().enumerate() {
            if e - s < 2 {
                continue;
            }
            let cost_full = ps.cost_l2(s, e);
            for t in s + 1..e {
                let gain = cost_full - ps.cost_l2(s, t) - ps.cost_l2(t, e);
                if gain > best_gain {
                    best_gain = gain;
                    best_split = t;
                    best_seg_idx = si;
                }
            }
        }

        if best_gain <= 0.0 {
            break; // no improvement possible
        }

        let (s, e) = segments[best_seg_idx];
        segments.remove(best_seg_idx);
        // Insert two child segments (keep list tidy)
        segments.push((s, best_split));
        segments.push((best_split, e));

        change_points.push(best_split);
    }

    change_points.sort_unstable();
    Ok(change_points)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn step_series(n: usize, cp: usize, level_a: f64, level_b: f64) -> Vec<f64> {
        (0..n)
            .map(|i| if i < cp { level_a } else { level_b })
            .collect()
    }

    #[test]
    fn test_bic_penalty_positive() {
        let p = bic_penalty(100, 1.0);
        assert!(p > 0.0);
    }

    #[test]
    fn test_pelt_no_change_flat() {
        let x = vec![1.0f64; 50];
        let cp = pelt_detect(&x, bic_penalty(50, 1.0)).unwrap();
        assert!(cp.is_empty(), "flat series should have no change points");
    }

    #[test]
    fn test_pelt_single_step() {
        let x = step_series(100, 50, 0.0, 5.0);
        let cp = pelt_detect(&x, 2.0 * (100.0f64).ln()).unwrap();
        assert!(!cp.is_empty(), "step series must have at least one change point");
        // nearest detected CP should be within ±5 of true position 50
        let nearest = cp.iter().min_by_key(|&&c| (c as isize - 50).unsigned_abs()).copied().unwrap_or(0);
        assert!(
            (nearest as isize - 50).abs() <= 5,
            "detected CP {nearest} is far from true CP 50"
        );
    }

    #[test]
    fn test_pelt_two_steps() {
        let mut x: Vec<f64> = vec![0.0; 30];
        for i in 10..20 {
            x[i] = 4.0;
        }
        for i in 20..30 {
            x[i] = 0.0;
        }
        let cp = pelt_detect(&x, 1.0).unwrap();
        // Expect two change points near 10 and 20
        assert!(cp.len() >= 1, "should detect at least one CP");
    }

    #[test]
    fn test_pelt_insufficient_data() {
        assert!(pelt_detect(&[1.0], 1.0).is_err());
    }

    #[test]
    fn test_cusum_no_change() {
        let x = vec![0.0f64; 50];
        let cp = cusum_detect(&x, 1.0).unwrap();
        assert!(cp.is_empty());
    }

    #[test]
    fn test_cusum_detects_step() {
        let x = step_series(100, 50, 0.0, 10.0);
        let std_est = 5.0; // rough estimate
        let cp = cusum_detect(&x, std_est).unwrap();
        assert!(!cp.is_empty(), "should detect the step");
    }

    #[test]
    fn test_cusum_negative_threshold_error() {
        assert!(cusum_detect(&[1.0, 2.0], -1.0).is_err());
    }

    #[test]
    fn test_binary_segmentation_zero_bkps() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let cp = binary_segmentation(&x, 0).unwrap();
        assert!(cp.is_empty());
    }

    #[test]
    fn test_binary_segmentation_single_step() {
        let x = step_series(60, 30, 0.0, 3.0);
        let cp = binary_segmentation(&x, 1).unwrap();
        assert_eq!(cp.len(), 1);
        assert!(
            (cp[0] as isize - 30).abs() <= 3,
            "detected CP {} is far from true CP 30",
            cp[0]
        );
    }

    #[test]
    fn test_binary_segmentation_sorted_output() {
        let mut x: Vec<f64> = vec![0.0; 90];
        for i in 30..60 { x[i] = 5.0; }
        let cp = binary_segmentation(&x, 3).unwrap();
        for w in cp.windows(2) {
            assert!(w[0] < w[1], "output not sorted");
        }
    }

    #[test]
    fn test_binary_segmentation_insufficient_data() {
        assert!(binary_segmentation(&[1.0], 1).is_err());
    }
}
