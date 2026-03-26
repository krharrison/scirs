//! Bloom filter variants: standard, counting, and scalable.
//!
//! A Bloom filter is a space-efficient probabilistic set that supports
//! membership queries with a one-sided error: it may report false positives
//! but never false negatives.
//!
//! # Variants
//!
//! - [`BloomFilter`] — fixed-size bit array with optimal hash count.
//! - [`CountingBloomFilter`] — 4-bit counters per bucket, supporting `remove()`.
//! - [`ScalableBloomFilter`] — auto-growing by appending filter slices with
//!   geometrically tighter false positive rates.

use crate::error::{CoreError, CoreResult, ErrorContext};
use crate::error_context;
use super::DoubleHasher;

// ============================================================================
// BloomFilter
// ============================================================================

/// A standard Bloom filter backed by a bit vector.
///
/// # Parameters
///
/// Given expected insertions `n` and desired false positive rate `p`:
/// - Optimal bit array size: `m = -n * ln(p) / (ln 2)^2`
/// - Optimal hash count: `k = (m / n) * ln 2`
///
/// # Example
///
/// ```rust
/// use scirs2_core::probabilistic::BloomFilter;
///
/// let mut bf = BloomFilter::with_rate(1000, 0.01).expect("valid params");
/// bf.insert(b"hello");
/// bf.insert(b"world");
///
/// assert!(bf.contains(b"hello"));
/// assert!(bf.contains(b"world"));
/// // Very unlikely (but possible) that an un-inserted item returns true
/// ```
#[derive(Clone)]
pub struct BloomFilter {
    /// Bit storage (packed into u64 words).
    bits: Vec<u64>,
    /// Total number of bits `m`.
    num_bits: usize,
    /// Number of hash functions `k`.
    num_hashes: u32,
    /// Number of items inserted.
    count: usize,
    /// Double hasher for generating positions.
    hasher: DoubleHasher,
}

impl BloomFilter {
    /// Create a Bloom filter with explicit bit count and hash count.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_bits` is zero or `num_hashes` is zero.
    pub fn new(num_bits: usize, num_hashes: u32) -> CoreResult<Self> {
        if num_bits == 0 {
            return Err(CoreError::InvalidArgument(
                error_context!("num_bits must be > 0"),
            ));
        }
        if num_hashes == 0 {
            return Err(CoreError::InvalidArgument(
                error_context!("num_hashes must be > 0"),
            ));
        }
        let n_words = (num_bits + 63) / 64;
        Ok(Self {
            bits: vec![0u64; n_words],
            num_bits,
            num_hashes,
            count: 0,
            hasher: DoubleHasher::new(),
        })
    }

    /// Create a Bloom filter sized for `expected_items` with false positive rate `fpr`.
    ///
    /// # Errors
    ///
    /// Returns an error if `expected_items` is zero or `fpr` is not in `(0, 1)`.
    pub fn with_rate(expected_items: usize, fpr: f64) -> CoreResult<Self> {
        if expected_items == 0 {
            return Err(CoreError::InvalidArgument(
                error_context!("expected_items must be > 0"),
            ));
        }
        if fpr <= 0.0 || fpr >= 1.0 {
            return Err(CoreError::InvalidArgument(
                error_context!("fpr must be in (0, 1)"),
            ));
        }
        let (m, k) = optimal_params(expected_items, fpr);
        Self::new(m, k)
    }

    /// Insert an item into the filter.
    pub fn insert(&mut self, item: &[u8]) {
        let (h1, h2) = self.hasher.hash_pair(item);
        for i in 0..self.num_hashes {
            let pos = DoubleHasher::position(h1, h2, i, self.num_bits);
            self.set_bit(pos);
        }
        self.count += 1;
    }

    /// Test whether an item *may* be in the set.
    ///
    /// Returns `true` if all `k` hash positions are set (possible false positive).
    /// Returns `false` if at least one position is unset (definite negative).
    pub fn contains(&self, item: &[u8]) -> bool {
        let (h1, h2) = self.hasher.hash_pair(item);
        for i in 0..self.num_hashes {
            let pos = DoubleHasher::position(h1, h2, i, self.num_bits);
            if !self.get_bit(pos) {
                return false;
            }
        }
        true
    }

    /// Compute the union of two Bloom filters (logical OR).
    ///
    /// # Errors
    ///
    /// Returns an error if the filters have different sizes or hash counts.
    pub fn union(&self, other: &BloomFilter) -> CoreResult<BloomFilter> {
        if self.num_bits != other.num_bits || self.num_hashes != other.num_hashes {
            return Err(CoreError::DimensionError(
                error_context!("Bloom filters must have the same num_bits and num_hashes for union"),
            ));
        }
        let bits: Vec<u64> = self
            .bits
            .iter()
            .zip(other.bits.iter())
            .map(|(a, b)| a | b)
            .collect();
        Ok(BloomFilter {
            bits,
            num_bits: self.num_bits,
            num_hashes: self.num_hashes,
            count: self.count + other.count, // approximate
            hasher: self.hasher.clone(),
        })
    }

    /// Estimate the number of items in the intersection of two filters.
    ///
    /// Uses the formula: `|A ∩ B| ≈ -m/k * ln(1 - X_ab/m)  -  (-m/k * ln(1 - X_a/m))  -  (-m/k * ln(1 - X_b/m))  +  n_a + n_b`
    /// Simplified Swamidass-Baldi method:
    ///   estimate = est(A) + est(B) - est(A ∪ B)
    ///
    /// # Errors
    ///
    /// Returns an error if the filters are incompatible.
    pub fn intersection_estimate(&self, other: &BloomFilter) -> CoreResult<f64> {
        if self.num_bits != other.num_bits || self.num_hashes != other.num_hashes {
            return Err(CoreError::DimensionError(
                error_context!(
                    "Bloom filters must have the same num_bits and num_hashes for intersection estimate"
                ),
            ));
        }
        let m = self.num_bits as f64;
        let k = self.num_hashes as f64;

        let bits_a = self.count_set_bits() as f64;
        let bits_b = other.count_set_bits() as f64;

        // Union bit count
        let bits_union: usize = self
            .bits
            .iter()
            .zip(other.bits.iter())
            .map(|(a, b)| (a | b).count_ones() as usize)
            .sum();
        let bits_ab = bits_union as f64;

        // Cardinality estimates via -m/k * ln(1 - bits/m)
        let est = |x: f64| -> f64 {
            let ratio = x / m;
            if ratio >= 1.0 {
                return m; // saturated
            }
            -m / k * (1.0 - ratio).ln()
        };

        let est_a = est(bits_a);
        let est_b = est(bits_b);
        let est_ab = est(bits_ab);

        // Inclusion-exclusion: |A ∩ B| ≈ |A| + |B| - |A ∪ B|
        let intersection = est_a + est_b - est_ab;
        Ok(intersection.max(0.0))
    }

    /// Number of items inserted.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the filter is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Total number of bits in the filter.
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// Number of hash functions.
    pub fn num_hashes(&self) -> u32 {
        self.num_hashes
    }

    /// Estimated current false positive rate based on the number of bits set.
    pub fn estimated_fpr(&self) -> f64 {
        let set_bits = self.count_set_bits() as f64;
        let fill_ratio = set_bits / self.num_bits as f64;
        fill_ratio.powi(self.num_hashes as i32)
    }

    /// Create an empty Bloom filter with the same parameters and hasher as `self`.
    ///
    /// This is useful when you need two compatible filters for `union()` or
    /// `intersection_estimate()`.
    pub fn empty_clone(&self) -> Self {
        let n_words = (self.num_bits + 63) / 64;
        Self {
            bits: vec![0u64; n_words],
            num_bits: self.num_bits,
            num_hashes: self.num_hashes,
            count: 0,
            hasher: self.hasher.clone(),
        }
    }

    /// Clear all bits, resetting the filter.
    pub fn clear(&mut self) {
        for word in &mut self.bits {
            *word = 0;
        }
        self.count = 0;
    }

    // -- private helpers --

    #[inline]
    fn set_bit(&mut self, pos: usize) {
        let word = pos / 64;
        let bit = pos % 64;
        self.bits[word] |= 1u64 << bit;
    }

    #[inline]
    fn get_bit(&self, pos: usize) -> bool {
        let word = pos / 64;
        let bit = pos % 64;
        (self.bits[word] >> bit) & 1 == 1
    }

    fn count_set_bits(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }
}

impl std::fmt::Debug for BloomFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BloomFilter")
            .field("num_bits", &self.num_bits)
            .field("num_hashes", &self.num_hashes)
            .field("count", &self.count)
            .field("set_bits", &self.count_set_bits())
            .finish()
    }
}

// ============================================================================
// CountingBloomFilter
// ============================================================================

/// A counting Bloom filter with 4-bit counters per bucket.
///
/// Unlike a standard Bloom filter, a counting variant supports `remove()`
/// at the cost of 4x memory (4 bits per bucket instead of 1).
///
/// Counter overflow is handled by capping at 15 (saturating increment).
///
/// # Example
///
/// ```rust
/// use scirs2_core::probabilistic::CountingBloomFilter;
///
/// let mut cbf = CountingBloomFilter::with_rate(1000, 0.01).expect("valid params");
/// cbf.insert(b"hello");
/// assert!(cbf.contains(b"hello"));
///
/// cbf.remove(b"hello");
/// assert!(!cbf.contains(b"hello"));
/// ```
#[derive(Clone)]
pub struct CountingBloomFilter {
    /// 4-bit counters, two per byte.
    counters: Vec<u8>,
    /// Number of buckets.
    num_buckets: usize,
    /// Number of hash functions.
    num_hashes: u32,
    /// Number of items inserted (net).
    count: usize,
    /// Double hasher.
    hasher: DoubleHasher,
}

impl CountingBloomFilter {
    /// Create a counting Bloom filter with explicit bucket count and hash count.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_buckets` is zero or `num_hashes` is zero.
    pub fn new(num_buckets: usize, num_hashes: u32) -> CoreResult<Self> {
        if num_buckets == 0 {
            return Err(CoreError::InvalidArgument(
                error_context!("num_buckets must be > 0"),
            ));
        }
        if num_hashes == 0 {
            return Err(CoreError::InvalidArgument(
                error_context!("num_hashes must be > 0"),
            ));
        }
        // Two counters per byte (4 bits each).
        let n_bytes = (num_buckets + 1) / 2;
        Ok(Self {
            counters: vec![0u8; n_bytes],
            num_buckets,
            num_hashes,
            count: 0,
            hasher: DoubleHasher::new(),
        })
    }

    /// Create a counting Bloom filter sized for `expected_items` with false positive rate `fpr`.
    ///
    /// # Errors
    ///
    /// Returns an error if `expected_items` is zero or `fpr` is not in `(0, 1)`.
    pub fn with_rate(expected_items: usize, fpr: f64) -> CoreResult<Self> {
        if expected_items == 0 {
            return Err(CoreError::InvalidArgument(
                error_context!("expected_items must be > 0"),
            ));
        }
        if fpr <= 0.0 || fpr >= 1.0 {
            return Err(CoreError::InvalidArgument(
                error_context!("fpr must be in (0, 1)"),
            ));
        }
        let (m, k) = optimal_params(expected_items, fpr);
        Self::new(m, k)
    }

    /// Insert an item into the filter.
    pub fn insert(&mut self, item: &[u8]) {
        let (h1, h2) = self.hasher.hash_pair(item);
        for i in 0..self.num_hashes {
            let pos = DoubleHasher::position(h1, h2, i, self.num_buckets);
            self.increment_counter(pos);
        }
        self.count += 1;
    }

    /// Remove an item from the filter.
    ///
    /// # Warning
    ///
    /// Removing an item that was never inserted may cause false negatives
    /// for other items. Use with care.
    pub fn remove(&mut self, item: &[u8]) {
        let (h1, h2) = self.hasher.hash_pair(item);
        for i in 0..self.num_hashes {
            let pos = DoubleHasher::position(h1, h2, i, self.num_buckets);
            self.decrement_counter(pos);
        }
        self.count = self.count.saturating_sub(1);
    }

    /// Test whether an item *may* be in the set.
    pub fn contains(&self, item: &[u8]) -> bool {
        let (h1, h2) = self.hasher.hash_pair(item);
        for i in 0..self.num_hashes {
            let pos = DoubleHasher::position(h1, h2, i, self.num_buckets);
            if self.get_counter(pos) == 0 {
                return false;
            }
        }
        true
    }

    /// Number of items inserted (net of removals).
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the filter is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Number of buckets.
    pub fn num_buckets(&self) -> usize {
        self.num_buckets
    }

    /// Number of hash functions.
    pub fn num_hashes(&self) -> u32 {
        self.num_hashes
    }

    /// Clear all counters.
    pub fn clear(&mut self) {
        for b in &mut self.counters {
            *b = 0;
        }
        self.count = 0;
    }

    // -- 4-bit counter helpers --

    #[inline]
    fn get_counter(&self, pos: usize) -> u8 {
        let byte_idx = pos / 2;
        if pos % 2 == 0 {
            self.counters[byte_idx] & 0x0F
        } else {
            (self.counters[byte_idx] >> 4) & 0x0F
        }
    }

    #[inline]
    fn increment_counter(&mut self, pos: usize) {
        let byte_idx = pos / 2;
        let current = self.get_counter(pos);
        if current < 15 {
            // saturating at 15
            if pos % 2 == 0 {
                self.counters[byte_idx] = (self.counters[byte_idx] & 0xF0) | (current + 1);
            } else {
                self.counters[byte_idx] =
                    (self.counters[byte_idx] & 0x0F) | ((current + 1) << 4);
            }
        }
    }

    #[inline]
    fn decrement_counter(&mut self, pos: usize) {
        let byte_idx = pos / 2;
        let current = self.get_counter(pos);
        if current > 0 {
            if pos % 2 == 0 {
                self.counters[byte_idx] = (self.counters[byte_idx] & 0xF0) | (current - 1);
            } else {
                self.counters[byte_idx] =
                    (self.counters[byte_idx] & 0x0F) | ((current - 1) << 4);
            }
        }
    }
}

impl std::fmt::Debug for CountingBloomFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CountingBloomFilter")
            .field("num_buckets", &self.num_buckets)
            .field("num_hashes", &self.num_hashes)
            .field("count", &self.count)
            .finish()
    }
}

// ============================================================================
// ScalableBloomFilter
// ============================================================================

/// A scalable Bloom filter that auto-grows by adding filter slices.
///
/// Each new slice uses a tighter false positive rate so that the overall
/// filter FPR converges to the target.  Specifically, slice `i` uses
/// `fpr * ratio^i` where `ratio` is typically 0.5 (halving).
///
/// # Example
///
/// ```rust
/// use scirs2_core::probabilistic::ScalableBloomFilter;
///
/// let mut sbf = ScalableBloomFilter::new(0.01, 1000, 0.5).expect("valid params");
/// for i in 0..2000u64 {
///     sbf.insert(&i.to_le_bytes());
/// }
/// // All inserted items are found (no false negatives)
/// for i in 0..2000u64 {
///     assert!(sbf.contains(&i.to_le_bytes()));
/// }
/// ```
#[derive(Clone)]
pub struct ScalableBloomFilter {
    /// Sequence of Bloom filter slices.
    slices: Vec<BloomFilter>,
    /// Target false positive rate for the overall filter.
    target_fpr: f64,
    /// Expected items per slice.
    slice_capacity: usize,
    /// FPR tightening ratio (typically 0.5).
    ratio: f64,
    /// Total items inserted.
    total_count: usize,
}

impl ScalableBloomFilter {
    /// Create a scalable Bloom filter.
    ///
    /// # Parameters
    ///
    /// - `target_fpr`: desired overall false positive rate
    /// - `initial_capacity`: expected items in the first slice
    /// - `ratio`: FPR tightening ratio per slice (e.g. 0.5 means each slice has half the FPR of the previous)
    ///
    /// # Errors
    ///
    /// Returns an error if parameters are out of range.
    pub fn new(target_fpr: f64, initial_capacity: usize, ratio: f64) -> CoreResult<Self> {
        if target_fpr <= 0.0 || target_fpr >= 1.0 {
            return Err(CoreError::InvalidArgument(
                error_context!("target_fpr must be in (0, 1)"),
            ));
        }
        if initial_capacity == 0 {
            return Err(CoreError::InvalidArgument(
                error_context!("initial_capacity must be > 0"),
            ));
        }
        if ratio <= 0.0 || ratio >= 1.0 {
            return Err(CoreError::InvalidArgument(
                error_context!("ratio must be in (0, 1)"),
            ));
        }

        // First slice gets `target_fpr * (1 - ratio)` to leave room for growth.
        // The geometric series sum: fpr0 * sum(ratio^i) = fpr0 / (1-ratio) = target_fpr
        // So fpr0 = target_fpr * (1 - ratio)
        let fpr0 = target_fpr * (1.0 - ratio);
        let first_slice = BloomFilter::with_rate(initial_capacity, fpr0)?;

        Ok(Self {
            slices: vec![first_slice],
            target_fpr,
            slice_capacity: initial_capacity,
            ratio,
            total_count: 0,
        })
    }

    /// Insert an item into the filter.
    ///
    /// If the current slice is at capacity, a new slice is added with a tighter FPR.
    pub fn insert(&mut self, item: &[u8]) {
        // Check if the latest slice needs to grow
        let last_idx = self.slices.len() - 1;
        if self.slices[last_idx].len() >= self.slice_capacity {
            // Add a new slice with tighter FPR
            let slice_idx = self.slices.len();
            let slice_fpr =
                self.target_fpr * (1.0 - self.ratio) * self.ratio.powi(slice_idx as i32);
            // Clamp FPR to a reasonable minimum
            let clamped_fpr = slice_fpr.max(1e-15);
            if let Ok(new_slice) = BloomFilter::with_rate(self.slice_capacity, clamped_fpr) {
                self.slices.push(new_slice);
            }
        }
        // Insert into the latest slice
        if let Some(last) = self.slices.last_mut() {
            last.insert(item);
        }
        self.total_count += 1;
    }

    /// Test whether an item *may* be in the set.
    ///
    /// Checks all slices; returns `true` if any slice reports containment.
    pub fn contains(&self, item: &[u8]) -> bool {
        self.slices.iter().any(|s| s.contains(item))
    }

    /// Total number of items inserted.
    pub fn len(&self) -> usize {
        self.total_count
    }

    /// Whether the filter is empty.
    pub fn is_empty(&self) -> bool {
        self.total_count == 0
    }

    /// Number of slices currently allocated.
    pub fn num_slices(&self) -> usize {
        self.slices.len()
    }

    /// Clear all slices.
    pub fn clear(&mut self) {
        for s in &mut self.slices {
            s.clear();
        }
        self.total_count = 0;
    }
}

impl std::fmt::Debug for ScalableBloomFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScalableBloomFilter")
            .field("num_slices", &self.slices.len())
            .field("total_count", &self.total_count)
            .field("target_fpr", &self.target_fpr)
            .finish()
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Compute optimal Bloom filter parameters.
///
/// Returns `(m, k)` where `m` = number of bits, `k` = number of hash functions.
fn optimal_params(n: usize, p: f64) -> (usize, u32) {
    let n_f = n as f64;
    let ln2 = std::f64::consts::LN_2;

    // m = -n * ln(p) / (ln 2)^2
    let m = (-n_f * p.ln() / (ln2 * ln2)).ceil() as usize;
    let m = m.max(1); // at least 1 bit

    // k = (m / n) * ln 2
    let k = ((m as f64 / n_f) * ln2).ceil() as u32;
    let k = k.max(1); // at least 1 hash

    (m, k)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- BloomFilter tests ----

    #[test]
    fn test_bloom_no_false_negatives() {
        let mut bf = BloomFilter::with_rate(1000, 0.01).expect("valid params");
        let items: Vec<Vec<u8>> = (0..500u64).map(|i| i.to_le_bytes().to_vec()).collect();
        for item in &items {
            bf.insert(item);
        }
        // Every inserted item must be found
        for item in &items {
            assert!(bf.contains(item), "False negative detected for {:?}", item);
        }
    }

    #[test]
    fn test_bloom_fpr_within_bounds() {
        let n = 10_000usize;
        let target_fpr = 0.05;
        let mut bf = BloomFilter::with_rate(n, target_fpr).expect("valid params");

        // Insert n items
        for i in 0..n as u64 {
            bf.insert(&i.to_le_bytes());
        }

        // Test 10,000 items that were NOT inserted
        let test_count = 10_000usize;
        let mut false_positives = 0usize;
        for i in (n as u64)..(n as u64 + test_count as u64) {
            if bf.contains(&i.to_le_bytes()) {
                false_positives += 1;
            }
        }
        let observed_fpr = false_positives as f64 / test_count as f64;
        // Allow 2x the target FPR as margin
        assert!(
            observed_fpr < target_fpr * 2.0,
            "FPR too high: {observed_fpr} (target: {target_fpr})"
        );
    }

    #[test]
    fn test_bloom_union() {
        let mut bf1 = BloomFilter::new(1000, 5).expect("valid");
        let mut bf2 = bf1.empty_clone(); // same hasher

        bf1.insert(b"alpha");
        bf2.insert(b"beta");

        let combined = bf1.union(&bf2).expect("compatible");
        assert!(combined.contains(b"alpha"));
        assert!(combined.contains(b"beta"));
    }

    #[test]
    fn test_bloom_intersection_estimate() {
        let mut bf1 = BloomFilter::with_rate(1000, 0.01).expect("valid");
        let mut bf2 = bf1.empty_clone(); // same hasher required

        // Insert 100 shared items, 100 unique to each
        for i in 0..200u64 {
            bf1.insert(&i.to_le_bytes());
        }
        for i in 100..300u64 {
            bf2.insert(&i.to_le_bytes());
        }

        let est = bf1.intersection_estimate(&bf2).expect("compatible");
        // True intersection is 100 items; allow wide margin
        assert!(est > 20.0, "Intersection estimate too low: {est}");
        assert!(est < 250.0, "Intersection estimate too high: {est}");
    }

    #[test]
    fn test_bloom_empty() {
        let bf = BloomFilter::with_rate(100, 0.01).expect("valid");
        assert!(bf.is_empty());
        assert_eq!(bf.len(), 0);
        assert!(!bf.contains(b"anything"));
    }

    #[test]
    fn test_bloom_clear() {
        let mut bf = BloomFilter::with_rate(100, 0.01).expect("valid");
        bf.insert(b"hello");
        assert!(bf.contains(b"hello"));
        bf.clear();
        assert!(bf.is_empty());
        // After clear, item should not be found (deterministic)
        assert!(!bf.contains(b"hello"));
    }

    #[test]
    fn test_bloom_invalid_params() {
        assert!(BloomFilter::new(0, 5).is_err());
        assert!(BloomFilter::new(100, 0).is_err());
        assert!(BloomFilter::with_rate(0, 0.01).is_err());
        assert!(BloomFilter::with_rate(100, 0.0).is_err());
        assert!(BloomFilter::with_rate(100, 1.0).is_err());
        assert!(BloomFilter::with_rate(100, -0.5).is_err());
    }

    // ---- CountingBloomFilter tests ----

    #[test]
    fn test_counting_bloom_insert_remove_roundtrip() {
        let mut cbf = CountingBloomFilter::with_rate(1000, 0.01).expect("valid");
        cbf.insert(b"hello");
        assert!(cbf.contains(b"hello"));
        cbf.remove(b"hello");
        assert!(!cbf.contains(b"hello"));
    }

    #[test]
    fn test_counting_bloom_no_false_negatives() {
        let mut cbf = CountingBloomFilter::with_rate(1000, 0.01).expect("valid");
        for i in 0..500u64 {
            cbf.insert(&i.to_le_bytes());
        }
        for i in 0..500u64 {
            assert!(cbf.contains(&i.to_le_bytes()));
        }
    }

    #[test]
    fn test_counting_bloom_counter_overflow() {
        // Insert the same item 20 times (overflow 4-bit counter at 15)
        let mut cbf = CountingBloomFilter::new(100, 3).expect("valid");
        for _ in 0..20 {
            cbf.insert(b"overflow_test");
        }
        assert!(cbf.contains(b"overflow_test"));
        // Remove 20 times; counter was capped at 15 so it goes to 0 at remove #15
        for _ in 0..20 {
            cbf.remove(b"overflow_test");
        }
        assert!(!cbf.contains(b"overflow_test"));
    }

    #[test]
    fn test_counting_bloom_multiple_items() {
        let mut cbf = CountingBloomFilter::with_rate(1000, 0.01).expect("valid");
        cbf.insert(b"apple");
        cbf.insert(b"banana");
        cbf.insert(b"cherry");

        assert!(cbf.contains(b"apple"));
        assert!(cbf.contains(b"banana"));
        assert!(cbf.contains(b"cherry"));

        cbf.remove(b"banana");
        assert!(cbf.contains(b"apple"));
        assert!(!cbf.contains(b"banana"));
        assert!(cbf.contains(b"cherry"));
    }

    #[test]
    fn test_counting_bloom_clear() {
        let mut cbf = CountingBloomFilter::new(100, 3).expect("valid");
        cbf.insert(b"data");
        cbf.clear();
        assert!(cbf.is_empty());
        assert!(!cbf.contains(b"data"));
    }

    // ---- ScalableBloomFilter tests ----

    #[test]
    fn test_scalable_bloom_no_false_negatives() {
        let mut sbf = ScalableBloomFilter::new(0.01, 500, 0.5).expect("valid");
        for i in 0..2000u64 {
            sbf.insert(&i.to_le_bytes());
        }
        for i in 0..2000u64 {
            assert!(
                sbf.contains(&i.to_le_bytes()),
                "False negative at {i}"
            );
        }
    }

    #[test]
    fn test_scalable_bloom_grows() {
        let mut sbf = ScalableBloomFilter::new(0.01, 100, 0.5).expect("valid");
        assert_eq!(sbf.num_slices(), 1);
        for i in 0..500u64 {
            sbf.insert(&i.to_le_bytes());
        }
        // Should have grown beyond 1 slice
        assert!(sbf.num_slices() > 1, "Expected growth, got {} slices", sbf.num_slices());
    }

    #[test]
    fn test_scalable_bloom_fpr_reasonable() {
        let mut sbf = ScalableBloomFilter::new(0.05, 1000, 0.5).expect("valid");
        for i in 0..1000u64 {
            sbf.insert(&i.to_le_bytes());
        }
        let test_count = 10_000usize;
        let mut fp = 0usize;
        for i in 1000u64..(1000 + test_count as u64) {
            if sbf.contains(&i.to_le_bytes()) {
                fp += 1;
            }
        }
        let observed = fp as f64 / test_count as f64;
        assert!(
            observed < 0.15,
            "Scalable bloom FPR too high: {observed}"
        );
    }

    #[test]
    fn test_scalable_bloom_invalid_params() {
        assert!(ScalableBloomFilter::new(0.0, 100, 0.5).is_err());
        assert!(ScalableBloomFilter::new(0.01, 0, 0.5).is_err());
        assert!(ScalableBloomFilter::new(0.01, 100, 0.0).is_err());
        assert!(ScalableBloomFilter::new(0.01, 100, 1.0).is_err());
    }

    #[test]
    fn test_scalable_bloom_single_element() {
        let mut sbf = ScalableBloomFilter::new(0.01, 100, 0.5).expect("valid");
        sbf.insert(b"only_one");
        assert!(sbf.contains(b"only_one"));
        assert_eq!(sbf.len(), 1);
    }
}
