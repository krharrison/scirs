//! t-digest: streaming quantile estimator.
//!
//! Implements the t-digest algorithm (Dunning & Ertl, 2019) for approximate
//! quantile computation over streaming data.  The digest maintains a compact
//! summary of centroids that enables accurate estimation at extreme quantiles.
//!
//! ## Usage
//!
//! ```rust
//! use scirs2_io::analytics::TDigest;
//!
//! let mut td = TDigest::new(100.0);
//! for v in 1..=100 {
//!     td.add(v as f64);
//! }
//! // Median should be near 50.
//! let median = td.quantile(0.5);
//! assert!((median - 50.0).abs() < 2.0, "median ≈ 50, got {median}");
//! ```

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Centroid
// ---------------------------------------------------------------------------

/// A weighted centroid in the t-digest.
#[derive(Debug, Clone, PartialEq)]
pub struct Centroid {
    /// The mean of the centroid.
    pub mean: f64,
    /// The total weight of points merged into this centroid.
    pub weight: f64,
}

impl Centroid {
    /// Create a new centroid with the given mean and weight.
    pub fn new(mean: f64, weight: f64) -> Self {
        Self { mean, weight }
    }
}

// ---------------------------------------------------------------------------
// TDigest
// ---------------------------------------------------------------------------

/// t-digest streaming quantile estimator.
#[derive(Debug, Clone)]
pub struct TDigest {
    /// Ordered list of centroids (sorted by mean after `compress`).
    centroids: Vec<Centroid>,
    /// Total weight of all points ever added.
    n: f64,
    /// Compression parameter δ (higher = more centroids = more accurate).
    delta: f64,
}

impl TDigest {
    /// Create a new t-digest.
    ///
    /// * `delta` – compression factor.  Typical values: 100–1000.
    ///   Higher values retain more centroids and improve accuracy at the
    ///   cost of memory.
    pub fn new(delta: f64) -> Self {
        Self {
            centroids: Vec::new(),
            n: 0.0,
            delta,
        }
    }

    /// Return the total weight (number of points added).
    pub fn total_weight(&self) -> f64 {
        self.n
    }

    /// Return the number of centroids currently stored.
    pub fn num_centroids(&self) -> usize {
        self.centroids.len()
    }

    /// Add a single value with weight 1.
    pub fn add(&mut self, x: f64) {
        self.add_weighted(x, 1.0);
    }

    /// Add a value with an explicit weight.
    pub fn add_weighted(&mut self, x: f64, w: f64) {
        self.centroids.push(Centroid::new(x, w));
        self.n += w;
        // Compress when the buffer grows too large.
        if self.centroids.len() > (10.0 * self.delta) as usize {
            self.compress();
        }
    }

    /// Compress the centroid list using the k1 scale function.
    ///
    /// Centroids are sorted by mean, then adjacent centroids are merged as
    /// long as the combined weight does not exceed the scale limit
    /// `k1(q + Δq) - k1(q) ≤ 1` (normalised by total weight).
    pub fn compress(&mut self) {
        if self.centroids.is_empty() {
            return;
        }

        // Sort by mean.
        self.centroids
            .sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap_or(std::cmp::Ordering::Equal));

        let total = self.n;
        if total == 0.0 {
            return;
        }

        let mut merged: Vec<Centroid> = Vec::with_capacity(self.centroids.len());
        let mut current = self.centroids[0].clone();
        let mut cumulative_weight = 0.0;

        for i in 1..self.centroids.len() {
            let c = &self.centroids[i];
            let q0 = (cumulative_weight + current.weight / 2.0) / total;
            let q1 = (cumulative_weight + current.weight + c.weight / 2.0) / total;
            let k_limit = k1_scale(q0, self.delta) + 1.0;
            if k1_scale(q1, self.delta) <= k_limit {
                // Merge into current centroid.
                let new_weight = current.weight + c.weight;
                current.mean =
                    (current.mean * current.weight + c.mean * c.weight) / new_weight;
                current.weight = new_weight;
            } else {
                cumulative_weight += current.weight;
                merged.push(current.clone());
                current = c.clone();
            }
        }
        merged.push(current);
        self.centroids = merged;
    }

    /// Estimate the `q`-th quantile (0 ≤ q ≤ 1).
    ///
    /// Returns `NaN` if the digest is empty.
    pub fn quantile(&self, q: f64) -> f64 {
        if self.centroids.is_empty() || self.n == 0.0 {
            return f64::NAN;
        }

        // Ensure centroids are sorted (they should be after compress, but
        // callers may not have called compress yet).
        let mut sorted = self.centroids.clone();
        sorted.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap_or(std::cmp::Ordering::Equal));

        // Edge cases.
        if q <= 0.0 {
            return sorted.first().map(|c| c.mean).unwrap_or(f64::NAN);
        }
        if q >= 1.0 {
            return sorted.last().map(|c| c.mean).unwrap_or(f64::NAN);
        }

        // Build cumulative weight at the midpoint of each centroid.
        // The midpoint of centroid i has cumulative rank `prefix + weight/2`.
        let n = sorted.len();
        let mut mid_ranks = Vec::with_capacity(n);
        let mut prefix = 0.0_f64;
        for c in &sorted {
            mid_ranks.push(prefix + c.weight / 2.0);
            prefix += c.weight;
        }

        let target = q * self.n;

        // Binary search for the bracketing centroid.
        if target <= mid_ranks[0] {
            return sorted[0].mean;
        }
        if target >= mid_ranks[n - 1] {
            return sorted[n - 1].mean;
        }

        // Find i such that mid_ranks[i-1] <= target < mid_ranks[i].
        let i = mid_ranks
            .partition_point(|&r| r <= target)
            .min(n - 1)
            .max(1);

        // Linearly interpolate between centroid i-1 and i.
        let r0 = mid_ranks[i - 1];
        let r1 = mid_ranks[i];
        let frac = if r1 - r0 > 0.0 {
            (target - r0) / (r1 - r0)
        } else {
            0.5
        };
        sorted[i - 1].mean + frac * (sorted[i].mean - sorted[i - 1].mean)
    }

    /// Estimate the cumulative distribution (probability that a random
    /// element ≤ x).
    pub fn cdf(&self, x: f64) -> f64 {
        if self.centroids.is_empty() || self.n == 0.0 {
            return f64::NAN;
        }

        let mut sorted = self.centroids.clone();
        sorted.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap_or(std::cmp::Ordering::Equal));

        if x < sorted[0].mean {
            return 0.0;
        }
        if x >= sorted[sorted.len() - 1].mean {
            return 1.0;
        }

        let mut cumulative = 0.0;
        for i in 0..sorted.len() {
            let c = &sorted[i];
            if x < c.mean {
                // Interpolate between previous centroid and this one.
                if i == 0 {
                    return 0.0;
                }
                let prev = &sorted[i - 1];
                let frac = if c.mean - prev.mean != 0.0 {
                    (x - prev.mean) / (c.mean - prev.mean)
                } else {
                    0.5
                };
                let prev_cum_mid = cumulative - prev.weight / 2.0;
                let this_cum_mid = cumulative + c.weight / 2.0;
                return (prev_cum_mid + frac * (this_cum_mid - prev_cum_mid)) / self.n;
            }
            cumulative += c.weight;
        }
        1.0
    }

    /// Merge another t-digest into this one (extends centroid list and recompresses).
    pub fn merge(&mut self, other: &TDigest) {
        for c in &other.centroids {
            self.centroids.push(c.clone());
            self.n += c.weight;
        }
        self.compress();
    }
}

/// k1 scale function: `δ / (2π) · arcsin(2q − 1)`.
///
/// Maps a quantile in `[0, 1]` to a scale value; adjacent centroids are
/// merged only while their k1 difference remains ≤ 1.
fn k1_scale(q: f64, delta: f64) -> f64 {
    let q_clamped = q.clamp(1e-10, 1.0 - 1e-10);
    delta / (2.0 * PI) * (2.0 * q_clamped - 1.0).asin()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn build_digest(n: usize) -> TDigest {
        let mut td = TDigest::new(100.0);
        for v in 1..=n {
            td.add(v as f64);
        }
        td
    }

    #[test]
    fn test_median_on_1_to_100() {
        let mut td = build_digest(100);
        td.compress();
        let median = td.quantile(0.5);
        assert!(
            (median - 50.0).abs() < 2.0,
            "Expected median ≈ 50, got {median}"
        );
    }

    #[test]
    fn test_quantile_0_is_min() {
        let mut td = build_digest(50);
        td.compress();
        let min = td.quantile(0.0);
        assert!(
            (min - 1.0).abs() < 1.0,
            "Expected min ≈ 1, got {min}"
        );
    }

    #[test]
    fn test_quantile_1_is_max() {
        let mut td = build_digest(50);
        td.compress();
        let max = td.quantile(1.0);
        assert!(
            (max - 50.0).abs() < 1.0,
            "Expected max ≈ 50, got {max}"
        );
    }

    #[test]
    fn test_merge_two_digests() {
        let mut td_a = TDigest::new(100.0);
        let mut td_b = TDigest::new(100.0);
        for v in 1..=50 {
            td_a.add(v as f64);
        }
        for v in 51..=100 {
            td_b.add(v as f64);
        }
        td_a.merge(&td_b);
        let median = td_a.quantile(0.5);
        assert!(
            (median - 50.0).abs() < 3.0,
            "Merged median expected ≈ 50, got {median}"
        );
    }

    #[test]
    fn test_cdf_at_median_is_approx_half() {
        let mut td = build_digest(100);
        td.compress();
        let cdf_at_50 = td.cdf(50.0);
        assert!(
            (cdf_at_50 - 0.5).abs() < 0.05,
            "CDF(50) expected ≈ 0.5, got {cdf_at_50}"
        );
    }

    #[test]
    fn test_empty_digest_returns_nan() {
        let td = TDigest::new(100.0);
        assert!(td.quantile(0.5).is_nan());
        assert!(td.cdf(0.0).is_nan());
    }
}
