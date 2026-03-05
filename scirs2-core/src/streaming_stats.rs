//! Online / streaming statistics via Welford's numerically stable algorithm.
//!
//! [`StreamingStats`] accumulates count, mean, variance (Bessel-corrected),
//! minimum, and maximum in a single pass over the data without storing the
//! full dataset.  Two independent accumulators can be merged in O(1) to
//! support parallel computation over disjoint partitions.
//!
//! This complements the generic [`crate::numeric::stability::WelfordVariance`]
//! by:
//!
//! * fixing the element type to `f64` for ergonomic use in pipelines,
//! * tracking `min` / `max` alongside the central moments, and
//! * providing a `merge` method suitable for map-reduce patterns.
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::streaming_stats::StreamingStats;
//!
//! let mut stats = StreamingStats::new();
//! for x in [1.0_f64, 2.0, 3.0, 4.0, 5.0] {
//!     stats.update(x);
//! }
//! assert!((stats.mean() - 3.0).abs() < 1e-12);
//! assert!((stats.variance() - 2.5).abs() < 1e-12); // sample variance
//! assert_eq!(stats.min(), 1.0);
//! assert_eq!(stats.max(), 5.0);
//! ```

// ──────────────────────────────────────────────────────────────────────────────
// StreamingStats
// ──────────────────────────────────────────────────────────────────────────────

/// Online statistics accumulator based on Welford's one-pass algorithm.
///
/// All operations are O(1) per data point (or O(n) for a batch update).
/// After processing all data the accumulator exposes:
///
/// * `count()` – number of samples seen
/// * `mean()` – arithmetic mean
/// * `variance()` – Bessel-corrected sample variance
/// * `std_dev()` – sample standard deviation
/// * `min()` / `max()` – extremes
///
/// The `merge` method combines two accumulators in O(1) using Chan's
/// parallel formula, enabling map-reduce over partitioned data.
#[derive(Debug, Clone)]
pub struct StreamingStats {
    count: u64,
    mean: f64,
    /// Running sum of squared deviations from the current mean (M₂).
    m2: f64,
    min: f64,
    max: f64,
}

impl Default for StreamingStats {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingStats {
    /// Create a new, empty accumulator.
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Incorporate one more observation using Welford's update rule.
    ///
    /// `NaN` values are silently ignored.
    pub fn update(&mut self, value: f64) {
        if value.is_nan() {
            return;
        }
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;

        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    /// Process a batch of values, equivalent to calling [`update`](Self::update)
    /// for each element but potentially more cache-friendly.
    pub fn update_batch(&mut self, values: &[f64]) {
        for &v in values {
            self.update(v);
        }
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    /// Number of non-NaN observations accumulated so far.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Arithmetic mean of all accumulated observations.
    ///
    /// Returns `0.0` when `count == 0`.
    #[inline]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Bessel-corrected sample variance (divides by `count - 1`).
    ///
    /// Returns `0.0` when `count <= 1`.
    pub fn variance(&self) -> f64 {
        if self.count <= 1 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    /// Population variance (divides by `count`).
    ///
    /// Returns `0.0` when `count == 0`.
    pub fn population_variance(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.m2 / self.count as f64
        }
    }

    /// Sample standard deviation (square root of the sample variance).
    ///
    /// Returns `0.0` when `count <= 1`.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Minimum value seen so far.
    ///
    /// Returns `f64::INFINITY` when `count == 0`.
    #[inline]
    pub fn min(&self) -> f64 {
        self.min
    }

    /// Maximum value seen so far.
    ///
    /// Returns `f64::NEG_INFINITY` when `count == 0`.
    #[inline]
    pub fn max(&self) -> f64 {
        self.max
    }

    /// Range (max − min).
    ///
    /// Returns `0.0` when fewer than two distinct values have been seen.
    pub fn range(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.max - self.min
        }
    }

    // ── Parallel merge ────────────────────────────────────────────────────────

    /// Merge two accumulators into one using Chan's parallel update formula.
    ///
    /// The result is mathematically equivalent to having accumulated all
    /// observations from both sources into a single accumulator.
    ///
    /// This enables map-reduce patterns:
    ///
    /// ```rust
    /// use scirs2_core::streaming_stats::StreamingStats;
    ///
    /// let mut a = StreamingStats::new();
    /// a.update_batch(&[1.0, 2.0, 3.0]);
    ///
    /// let mut b = StreamingStats::new();
    /// b.update_batch(&[4.0, 5.0, 6.0]);
    ///
    /// let combined = a.merge(b);
    /// assert!((combined.mean() - 3.5).abs() < 1e-12);
    /// ```
    pub fn merge(self, other: Self) -> Self {
        if self.count == 0 {
            return other;
        }
        if other.count == 0 {
            return self;
        }

        let combined_count = self.count + other.count;
        let delta = other.mean - self.mean;
        let combined_mean = (self.mean * self.count as f64 + other.mean * other.count as f64)
            / combined_count as f64;
        // Chan's parallel variance update.
        let combined_m2 = self.m2
            + other.m2
            + delta * delta * (self.count as f64 * other.count as f64) / combined_count as f64;

        Self {
            count: combined_count,
            mean: combined_mean,
            m2: combined_m2,
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Convenience: merge a slice of accumulators into one.
    ///
    /// Returns `StreamingStats::new()` when the slice is empty.
    pub fn merge_all(stats: Vec<Self>) -> Self {
        stats.into_iter().fold(Self::new(), |acc, s| acc.merge(s))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Basic update ─────────────────────────────────────────────────────────

    #[test]
    fn test_empty_accumulator() {
        let s = StreamingStats::new();
        assert_eq!(s.count(), 0);
        assert_eq!(s.mean(), 0.0);
        assert_eq!(s.variance(), 0.0);
        assert_eq!(s.std_dev(), 0.0);
        assert_eq!(s.min(), f64::INFINITY);
        assert_eq!(s.max(), f64::NEG_INFINITY);
    }

    #[test]
    fn test_single_element() {
        let mut s = StreamingStats::new();
        s.update(7.0);
        assert_eq!(s.count(), 1);
        assert!((s.mean() - 7.0).abs() < 1e-12);
        assert_eq!(s.variance(), 0.0); // Bessel: n-1 = 0 → 0
        assert_eq!(s.min(), 7.0);
        assert_eq!(s.max(), 7.0);
    }

    #[test]
    fn test_two_elements_mean() {
        let mut s = StreamingStats::new();
        s.update(3.0);
        s.update(7.0);
        assert!((s.mean() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_known_variance() {
        // [1, 2, 3, 4, 5]  sample var = 2.5
        let mut s = StreamingStats::new();
        for x in [1.0_f64, 2.0, 3.0, 4.0, 5.0] {
            s.update(x);
        }
        assert_eq!(s.count(), 5);
        assert!((s.mean() - 3.0).abs() < 1e-12);
        assert!((s.variance() - 2.5).abs() < 1e-12);
        assert_eq!(s.min(), 1.0);
        assert_eq!(s.max(), 5.0);
    }

    #[test]
    fn test_std_dev_matches_variance_sqrt() {
        let mut s = StreamingStats::new();
        for x in [2.0_f64, 4.0, 6.0, 8.0] {
            s.update(x);
        }
        assert!((s.std_dev() - s.variance().sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_update_batch_equals_sequential() {
        let values = [1.1, 2.2, 3.3, 4.4, 5.5_f64];

        let mut seq = StreamingStats::new();
        for &v in &values {
            seq.update(v);
        }

        let mut batch = StreamingStats::new();
        batch.update_batch(&values);

        assert_eq!(seq.count(), batch.count());
        assert!((seq.mean() - batch.mean()).abs() < 1e-14);
        assert!((seq.variance() - batch.variance()).abs() < 1e-14);
    }

    #[test]
    fn test_min_max_tracking() {
        let mut s = StreamingStats::new();
        let values = [-5.0_f64, 3.0, 1.0, -1.0, 4.0, 0.0];
        s.update_batch(&values);
        assert_eq!(s.min(), -5.0);
        assert_eq!(s.max(), 4.0);
    }

    #[test]
    fn test_range() {
        let mut s = StreamingStats::new();
        s.update_batch(&[10.0, 20.0, 30.0]);
        assert!((s.range() - 20.0).abs() < 1e-12);
    }

    #[test]
    fn test_nan_ignored() {
        let mut s = StreamingStats::new();
        s.update(1.0);
        s.update(f64::NAN);
        s.update(3.0);
        assert_eq!(s.count(), 2);
        assert!((s.mean() - 2.0).abs() < 1e-12);
    }

    // ── Merge ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_merge_two_halves_equals_full() {
        let values: Vec<f64> = (1..=10).map(|x| x as f64).collect();

        let mut full = StreamingStats::new();
        full.update_batch(&values);

        let mut a = StreamingStats::new();
        a.update_batch(&values[..5]);
        let mut b = StreamingStats::new();
        b.update_batch(&values[5..]);
        let merged = a.merge(b);

        assert_eq!(merged.count(), full.count());
        assert!((merged.mean() - full.mean()).abs() < 1e-12);
        assert!((merged.variance() - full.variance()).abs() < 1e-12);
        assert_eq!(merged.min(), full.min());
        assert_eq!(merged.max(), full.max());
    }

    #[test]
    fn test_merge_with_empty_left() {
        let empty = StreamingStats::new();
        let mut s = StreamingStats::new();
        s.update_batch(&[1.0, 2.0, 3.0]);
        let merged = empty.merge(s.clone());
        assert_eq!(merged.count(), s.count());
        assert!((merged.mean() - s.mean()).abs() < 1e-12);
    }

    #[test]
    fn test_merge_with_empty_right() {
        let mut s = StreamingStats::new();
        s.update_batch(&[1.0, 2.0, 3.0]);
        let empty = StreamingStats::new();
        let merged = s.clone().merge(empty);
        assert_eq!(merged.count(), s.count());
        assert!((merged.mean() - s.mean()).abs() < 1e-12);
    }

    #[test]
    fn test_merge_all_empty_slice() {
        let merged = StreamingStats::merge_all(vec![]);
        assert_eq!(merged.count(), 0);
    }

    #[test]
    fn test_merge_all_matches_sequential() {
        let partitions: Vec<Vec<f64>> = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0],
            vec![6.0, 7.0, 8.0, 9.0],
        ];

        let all_values: Vec<f64> = partitions.iter().flatten().copied().collect();
        let mut full = StreamingStats::new();
        full.update_batch(&all_values);

        let parts: Vec<StreamingStats> = partitions
            .iter()
            .map(|p| {
                let mut s = StreamingStats::new();
                s.update_batch(p);
                s
            })
            .collect();

        let merged = StreamingStats::merge_all(parts);
        assert_eq!(merged.count(), full.count());
        assert!((merged.mean() - full.mean()).abs() < 1e-12);
        assert!((merged.variance() - full.variance()).abs() < 1e-10);
    }

    // ── Numerical stability ───────────────────────────────────────────────────

    #[test]
    fn test_large_offset_stability() {
        // Welford's algorithm should remain numerically stable when all values
        // are very large and nearly equal.
        let base = 1_000_000_000.0_f64;
        let deltas = [0.1, 0.2, 0.3, 0.4, 0.5_f64];
        let mut s = StreamingStats::new();
        for &d in &deltas {
            s.update(base + d);
        }
        let expected_mean = base + 0.3;
        assert!((s.mean() - expected_mean).abs() < 1e-6);
    }

    #[test]
    fn test_population_variance() {
        // [1, 2, 3, 4, 5]  population var = 2.0
        let mut s = StreamingStats::new();
        for x in [1.0_f64, 2.0, 3.0, 4.0, 5.0] {
            s.update(x);
        }
        assert!((s.population_variance() - 2.0).abs() < 1e-12);
    }
}
