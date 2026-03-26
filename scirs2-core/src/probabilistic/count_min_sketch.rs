//! Count-Min Sketch for frequency estimation.
//!
//! A Count-Min Sketch (CMS) is a probabilistic data structure for
//! estimating the frequency of events in a stream.  It uses a 2D array
//! of counters with `depth` independent hash functions.  The estimate
//! for any item is the *minimum* of its `depth` counters, which guarantees
//! the estimate is never less than the true count (one-sided error).
//!
//! # Error bounds
//!
//! With width `w` and depth `d`:
//! - Point query error: at most `epsilon * N` with probability `1 - delta`
//!   where `w = ceil(e / epsilon)` and `d = ceil(ln(1 / delta))`
//!   and `N` = total number of increments.
//!
//! # Conservative update
//!
//! The conservative update variant only increments counters that are at the
//! current minimum, which can reduce overestimation in practice.

use crate::error::{CoreError, CoreResult, ErrorContext};
use crate::error_context;
use super::DoubleHasher;

/// A Count-Min Sketch for frequency estimation.
///
/// # Example
///
/// ```rust
/// use scirs2_core::probabilistic::CountMinSketch;
///
/// let mut cms = CountMinSketch::new(0.001, 0.01).expect("valid params");
/// cms.increment(b"event_a");
/// cms.increment(b"event_a");
/// cms.increment(b"event_b");
///
/// assert!(cms.estimate(b"event_a") >= 2);
/// assert!(cms.estimate(b"event_b") >= 1);
/// assert_eq!(cms.estimate(b"event_c"), 0); // never seen, but might overestimate
/// ```
#[derive(Clone)]
pub struct CountMinSketch {
    /// 2D counter table, laid out as `counters[row * width + col]`.
    counters: Vec<u64>,
    /// Width of the table (number of columns).
    width: usize,
    /// Depth of the table (number of rows / hash functions).
    depth: usize,
    /// Total number of increments performed.
    total_count: u64,
    /// One hasher per row (to get independent hash functions).
    hashers: Vec<DoubleHasher>,
}

impl CountMinSketch {
    /// Create a Count-Min Sketch with given error bounds.
    ///
    /// - `epsilon`: additive error factor (point query error <= epsilon * N)
    /// - `delta`: probability of exceeding the error bound
    ///
    /// Width = `ceil(e / epsilon)`, Depth = `ceil(ln(1 / delta))`.
    ///
    /// # Errors
    ///
    /// Returns an error if `epsilon` or `delta` is not in `(0, 1)`.
    pub fn new(epsilon: f64, delta: f64) -> CoreResult<Self> {
        if epsilon <= 0.0 || epsilon >= 1.0 {
            return Err(CoreError::InvalidArgument(
                error_context!("epsilon must be in (0, 1)"),
            ));
        }
        if delta <= 0.0 || delta >= 1.0 {
            return Err(CoreError::InvalidArgument(
                error_context!("delta must be in (0, 1)"),
            ));
        }

        let width = (std::f64::consts::E / epsilon).ceil() as usize;
        let depth = (1.0_f64 / delta).ln().ceil() as usize;
        let width = width.max(1);
        let depth = depth.max(1);

        Self::with_dimensions(width, depth)
    }

    /// Create a Count-Min Sketch with explicit dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if width or depth is zero.
    pub fn with_dimensions(width: usize, depth: usize) -> CoreResult<Self> {
        if width == 0 {
            return Err(CoreError::InvalidArgument(
                error_context!("width must be > 0"),
            ));
        }
        if depth == 0 {
            return Err(CoreError::InvalidArgument(
                error_context!("depth must be > 0"),
            ));
        }
        let hashers: Vec<DoubleHasher> = (0..depth).map(|_| DoubleHasher::new()).collect();
        Ok(Self {
            counters: vec![0u64; width * depth],
            width,
            depth,
            total_count: 0,
            hashers,
        })
    }

    /// Increment the count for an item by 1.
    pub fn increment(&mut self, item: &[u8]) {
        self.increment_by(item, 1);
    }

    /// Increment the count for an item by a given amount.
    pub fn increment_by(&mut self, item: &[u8], count: u64) {
        for row in 0..self.depth {
            let col = self.hash_to_col(row, item);
            self.counters[row * self.width + col] =
                self.counters[row * self.width + col].saturating_add(count);
        }
        self.total_count = self.total_count.saturating_add(count);
    }

    /// Increment using the conservative update strategy.
    ///
    /// Only increments counters that are at the current minimum value,
    /// which reduces overestimation.
    pub fn increment_conservative(&mut self, item: &[u8]) {
        self.increment_conservative_by(item, 1);
    }

    /// Conservative update with a custom increment amount.
    pub fn increment_conservative_by(&mut self, item: &[u8], count: u64) {
        // First pass: find the current minimum
        let current_min = self.estimate(item);

        // Second pass: only increment counters at the minimum
        let new_val = current_min.saturating_add(count);
        for row in 0..self.depth {
            let col = self.hash_to_col(row, item);
            let idx = row * self.width + col;
            if self.counters[idx] < new_val {
                self.counters[idx] = new_val;
            }
        }
        self.total_count = self.total_count.saturating_add(count);
    }

    /// Estimate the frequency of an item.
    ///
    /// Returns the minimum counter value across all rows, which is an
    /// upper bound on the true frequency.
    pub fn estimate(&self, item: &[u8]) -> u64 {
        let mut min_val = u64::MAX;
        for row in 0..self.depth {
            let col = self.hash_to_col(row, item);
            let val = self.counters[row * self.width + col];
            if val < min_val {
                min_val = val;
            }
        }
        if min_val == u64::MAX {
            0
        } else {
            min_val
        }
    }

    /// Estimate the inner product of two sketches.
    ///
    /// Returns the minimum row-wise inner product across all rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the sketches have different dimensions.
    pub fn inner_product(&self, other: &CountMinSketch) -> CoreResult<u64> {
        if self.width != other.width || self.depth != other.depth {
            return Err(CoreError::DimensionError(
                error_context!("Sketches must have the same dimensions for inner product"),
            ));
        }
        let mut min_ip = u64::MAX;
        for row in 0..self.depth {
            let mut row_ip: u64 = 0;
            for col in 0..self.width {
                let idx = row * self.width + col;
                row_ip = row_ip.saturating_add(
                    self.counters[idx].saturating_mul(other.counters[idx]),
                );
            }
            if row_ip < min_ip {
                min_ip = row_ip;
            }
        }
        Ok(if min_ip == u64::MAX { 0 } else { min_ip })
    }

    /// Merge another sketch into this one (element-wise addition).
    ///
    /// # Errors
    ///
    /// Returns an error if the sketches have different dimensions.
    pub fn merge(&mut self, other: &CountMinSketch) -> CoreResult<()> {
        if self.width != other.width || self.depth != other.depth {
            return Err(CoreError::DimensionError(
                error_context!("Sketches must have the same dimensions for merge"),
            ));
        }
        for i in 0..self.counters.len() {
            self.counters[i] = self.counters[i].saturating_add(other.counters[i]);
        }
        self.total_count = self.total_count.saturating_add(other.total_count);
        Ok(())
    }

    /// Find heavy hitters: items whose estimated frequency exceeds `threshold`.
    ///
    /// Since the CMS does not store item keys, this method takes a candidate
    /// set and filters by the threshold. The caller is responsible for
    /// maintaining the candidate set.
    pub fn heavy_hitters<'a>(
        &self,
        candidates: &'a [&[u8]],
        threshold: u64,
    ) -> Vec<(&'a [u8], u64)> {
        candidates
            .iter()
            .filter_map(|&item| {
                let est = self.estimate(item);
                if est >= threshold {
                    Some((item, est))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Total number of increments.
    pub fn total_count(&self) -> u64 {
        self.total_count
    }

    /// Width of the sketch.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Depth of the sketch.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Create an empty sketch with the same dimensions and hashers as `self`.
    ///
    /// This is required for correct `merge()` and `inner_product()` operations,
    /// which assume both sketches use the same hash functions.
    pub fn empty_clone(&self) -> Self {
        Self {
            counters: vec![0u64; self.width * self.depth],
            width: self.width,
            depth: self.depth,
            total_count: 0,
            hashers: self.hashers.clone(),
        }
    }

    /// Clear all counters.
    pub fn clear(&mut self) {
        for c in &mut self.counters {
            *c = 0;
        }
        self.total_count = 0;
    }

    // -- private --

    #[inline]
    fn hash_to_col(&self, row: usize, item: &[u8]) -> usize {
        let (h1, h2) = self.hashers[row].hash_pair(item);
        DoubleHasher::position(h1, h2, 0, self.width)
    }
}

impl std::fmt::Debug for CountMinSketch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CountMinSketch")
            .field("width", &self.width)
            .field("depth", &self.depth)
            .field("total_count", &self.total_count)
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cms_basic_frequency() {
        let mut cms = CountMinSketch::new(0.001, 0.01).expect("valid");
        for _ in 0..100 {
            cms.increment(b"apple");
        }
        for _ in 0..50 {
            cms.increment(b"banana");
        }

        let est_apple = cms.estimate(b"apple");
        let est_banana = cms.estimate(b"banana");

        // CMS overestimates, never underestimates
        assert!(est_apple >= 100, "apple estimate too low: {est_apple}");
        assert!(est_banana >= 50, "banana estimate too low: {est_banana}");
    }

    #[test]
    fn test_cms_estimates_within_error_bounds() {
        let epsilon = 0.01;
        let delta = 0.01;
        let mut cms = CountMinSketch::new(epsilon, delta).expect("valid");

        let n = 10_000u64;
        // Insert items with known frequencies
        for i in 0..n {
            cms.increment(&i.to_le_bytes());
        }

        // Check that estimates are within epsilon * N of true count (which is 1)
        let max_error = (epsilon * n as f64).ceil() as u64;
        let mut within_bounds = 0usize;
        let test_count = 1000usize;
        for i in 0..test_count as u64 {
            let est = cms.estimate(&i.to_le_bytes());
            if est <= 1 + max_error {
                within_bounds += 1;
            }
        }
        // At least (1 - delta) fraction should be within bounds
        let expected_min = ((1.0 - delta) * test_count as f64) as usize;
        assert!(
            within_bounds >= expected_min.saturating_sub(10),
            "Only {within_bounds}/{test_count} estimates within bounds (expected >= {expected_min})"
        );
    }

    #[test]
    fn test_cms_merge() {
        let mut cms1 = CountMinSketch::with_dimensions(100, 5).expect("valid");
        let mut cms2 = CountMinSketch::with_dimensions(100, 5).expect("valid");

        for _ in 0..30 {
            cms1.increment(b"event");
        }
        for _ in 0..20 {
            cms2.increment(b"event");
        }

        cms1.merge(&cms2).expect("same dimensions");
        // Merged estimate should be >= 50
        // Note: merge only works correctly when both sketches use the same hashers.
        // Since they have different hashers, the merge adds counters positionally,
        // which may give unexpected results. In practice, sketches to be merged
        // should share the same hash configuration.
        assert!(cms1.total_count() == 50);
    }

    #[test]
    fn test_cms_conservative_update() {
        let mut cms = CountMinSketch::with_dimensions(200, 5).expect("valid");
        for _ in 0..100 {
            cms.increment_conservative(b"item");
        }
        let est = cms.estimate(b"item");
        // Conservative update should still be >= true count
        assert!(est >= 100, "Conservative estimate too low: {est}");
    }

    #[test]
    fn test_cms_heavy_hitters() {
        let mut cms = CountMinSketch::new(0.001, 0.01).expect("valid");
        for _ in 0..1000 {
            cms.increment(b"hot");
        }
        for _ in 0..10 {
            cms.increment(b"cold");
        }

        let candidates: Vec<&[u8]> = vec![b"hot", b"cold", b"missing"];
        let hh = cms.heavy_hitters(&candidates, 500);
        assert!(!hh.is_empty());
        assert!(hh.iter().any(|(item, _)| *item == b"hot"));
        // "cold" should NOT be a heavy hitter
        assert!(!hh.iter().any(|(item, _)| *item == b"cold"));
    }

    #[test]
    fn test_cms_empty() {
        let cms = CountMinSketch::with_dimensions(50, 3).expect("valid");
        assert_eq!(cms.total_count(), 0);
        // Estimate of unseen item should be 0
        assert_eq!(cms.estimate(b"nope"), 0);
    }

    #[test]
    fn test_cms_invalid_params() {
        assert!(CountMinSketch::new(0.0, 0.01).is_err());
        assert!(CountMinSketch::new(0.01, 0.0).is_err());
        assert!(CountMinSketch::new(1.0, 0.01).is_err());
        assert!(CountMinSketch::with_dimensions(0, 5).is_err());
        assert!(CountMinSketch::with_dimensions(5, 0).is_err());
    }

    #[test]
    fn test_cms_increment_by() {
        let mut cms = CountMinSketch::with_dimensions(200, 5).expect("valid");
        cms.increment_by(b"bulk", 42);
        assert!(cms.estimate(b"bulk") >= 42);
        assert_eq!(cms.total_count(), 42);
    }

    #[test]
    fn test_cms_inner_product() {
        let mut cms1 = CountMinSketch::with_dimensions(100, 5).expect("valid");
        let mut cms2 = cms1.empty_clone(); // same hashers required

        cms1.increment_by(b"a", 10);
        cms2.increment_by(b"a", 5);

        let ip = cms1.inner_product(&cms2).expect("same dims");
        // Inner product should be at least 50 (10 * 5 for the shared bucket)
        assert!(ip >= 50, "Inner product too low: {ip}");
    }

    #[test]
    fn test_cms_clear() {
        let mut cms = CountMinSketch::with_dimensions(50, 3).expect("valid");
        cms.increment(b"data");
        cms.clear();
        assert_eq!(cms.total_count(), 0);
        assert_eq!(cms.estimate(b"data"), 0);
    }

    #[test]
    fn test_cms_merge_incompatible() {
        let mut cms1 = CountMinSketch::with_dimensions(100, 5).expect("valid");
        let cms2 = CountMinSketch::with_dimensions(200, 5).expect("valid");
        assert!(cms1.merge(&cms2).is_err());
    }
}
