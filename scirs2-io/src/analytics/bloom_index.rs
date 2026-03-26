//! Bloom filter and column-level Bloom index for predicate pushdown.
//!
//! A Bloom filter is a space-efficient probabilistic data structure that
//! answers set-membership queries with zero false negatives and a
//! configurable false-positive rate.
//!
//! ## Usage
//!
//! ```rust
//! use scirs2_io::analytics::bloom_index::{BloomFilter, BloomColumnIndex};
//!
//! // Single filter
//! let mut bloom = BloomFilter::new(1000, 0.01);
//! bloom.insert(b"hello");
//! bloom.insert(b"world");
//! assert!(bloom.contains(b"hello"));
//!
//! // Column index for predicate pushdown
//! let mut idx = BloomColumnIndex::new();
//! idx.add_column("city", 500, 0.02);
//! idx.insert_string("city", "Berlin");
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// BloomFilter
// ---------------------------------------------------------------------------

/// Bloom filter with double-hashing (FNV-1a + DJB2).
///
/// Parameters `m` (bit-array size) and `k` (number of hash functions) are
/// derived automatically from the expected number of elements and the desired
/// false-positive rate.
#[derive(Debug, Clone)]
pub struct BloomFilter {
    /// Packed bit array (each `u64` holds 64 bits).
    bits: Vec<u64>,
    /// Total number of bits `m`.
    n_bits: usize,
    /// Number of hash functions `k`.
    n_hashes: usize,
    /// Number of elements inserted so far.
    n_elements: u64,
}

impl BloomFilter {
    /// Create a new Bloom filter optimised for `expected_elements` items and
    /// target `false_positive_rate` ∈ (0, 1).
    ///
    /// Optimal parameters:
    /// ```text
    /// m = ceil(-n · ln p / (ln 2)²)
    /// k = round((m / n) · ln 2)
    /// ```
    pub fn new(expected_elements: usize, false_positive_rate: f64) -> Self {
        let n = expected_elements.max(1) as f64;
        // Clamp fpr to a sane range
        let p = false_positive_rate.clamp(1e-15, 1.0 - 1e-15);

        let ln2 = std::f64::consts::LN_2;
        let m = (-(n * p.ln()) / (ln2 * ln2)).ceil() as usize;
        let m = m.max(64); // minimum 64 bits

        let k = ((m as f64 / n) * ln2).round() as usize;
        let k = k.clamp(1, 30);

        let n_words = (m + 63) / 64;
        Self {
            bits: vec![0u64; n_words],
            n_bits: n_words * 64, // actual capacity (rounded up)
            n_hashes: k,
            n_elements: 0,
        }
    }

    /// Insert an item into the filter.
    pub fn insert(&mut self, item: &[u8]) {
        let h1 = Self::fnv1a_hash(item);
        let h2 = Self::djb2_hash(item);
        let m = self.n_bits as u64;

        for i in 0..self.n_hashes {
            let bit_idx = h1.wrapping_add((i as u64).wrapping_mul(h2)) % m;
            let word = (bit_idx / 64) as usize;
            let bit = bit_idx % 64;
            self.bits[word] |= 1u64 << bit;
        }
        self.n_elements += 1;
    }

    /// Return `true` if the item *might* be in the set, `false` if it is
    /// *definitely absent*.
    pub fn contains(&self, item: &[u8]) -> bool {
        let h1 = Self::fnv1a_hash(item);
        let h2 = Self::djb2_hash(item);
        let m = self.n_bits as u64;

        for i in 0..self.n_hashes {
            let bit_idx = h1.wrapping_add((i as u64).wrapping_mul(h2)) % m;
            let word = (bit_idx / 64) as usize;
            let bit = bit_idx % 64;
            if self.bits[word] & (1u64 << bit) == 0 {
                return false;
            }
        }
        true
    }

    /// Estimate the current false-positive rate based on the number of
    /// inserted elements.
    ///
    /// Formula: `(1 - exp(-k · n / m))^k`
    pub fn false_positive_rate_estimate(&self) -> f64 {
        let k = self.n_hashes as f64;
        let n = self.n_elements as f64;
        let m = self.n_bits as f64;
        (1.0 - (-k * n / m).exp()).powf(k)
    }

    /// Number of elements inserted.
    pub fn n_elements(&self) -> u64 {
        self.n_elements
    }

    /// Total number of bits in the filter.
    pub fn n_bits(&self) -> usize {
        self.n_bits
    }

    /// Number of hash functions.
    pub fn n_hashes(&self) -> usize {
        self.n_hashes
    }

    // ---------------------------------------------------------------------------
    // Hash functions
    // ---------------------------------------------------------------------------

    /// FNV-1a 64-bit hash.
    fn fnv1a_hash(data: &[u8]) -> u64 {
        const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
        const PRIME: u64 = 0x100000001b3;
        let mut hash = OFFSET_BASIS;
        for &b in data {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(PRIME);
        }
        hash
    }

    /// DJB2 hash with a fixed seed for the second hash function.
    ///
    /// Using a different starting state from FNV-1a ensures the two hash
    /// functions are independent enough for double hashing.
    fn djb2_hash(data: &[u8]) -> u64 {
        let mut hash: u64 = 5381;
        for &b in data {
            hash = hash.wrapping_mul(33).wrapping_add(u64::from(b));
        }
        // Mix to improve avalanche and ensure non-zero for the second hash
        hash ^ (hash >> 32)
    }
}

// ---------------------------------------------------------------------------
// BloomColumnIndex
// ---------------------------------------------------------------------------

/// Result of a Bloom-filter predicate check.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PredicatePushdownResult {
    /// The value *might* be present — the row group cannot be skipped.
    MightContain,
    /// The value is *definitely absent* — the row group can be skipped.
    DefinitelyAbsent,
}

/// A collection of per-column Bloom filters for predicate pushdown.
///
/// At query time, before reading a row group, call [`BloomColumnIndex::check`]
/// or [`BloomColumnIndex::check_string`] for each predicate column/value pair.
/// If the result is [`PredicatePushdownResult::DefinitelyAbsent`], the entire
/// row group can be skipped.
pub struct BloomColumnIndex {
    columns: HashMap<String, BloomFilter>,
}

impl BloomColumnIndex {
    /// Create an empty column index.
    pub fn new() -> Self {
        Self {
            columns: HashMap::new(),
        }
    }

    /// Add a Bloom filter for `column_name`.
    ///
    /// Returns a mutable reference to the newly created filter so the caller
    /// can immediately start inserting values.
    pub fn add_column(
        &mut self,
        column_name: &str,
        expected_elements: usize,
        fpr: f64,
    ) -> &mut BloomFilter {
        self.columns
            .entry(column_name.to_string())
            .or_insert_with(|| BloomFilter::new(expected_elements, fpr))
    }

    /// Insert a raw byte value into the named column's filter.
    ///
    /// If no filter has been registered for this column, the call is silently
    /// ignored (the caller may wish to call [`add_column`] first).
    pub fn insert_bytes(&mut self, column_name: &str, value: &[u8]) {
        if let Some(filter) = self.columns.get_mut(column_name) {
            filter.insert(value);
        }
    }

    /// Insert a string value into the named column's filter.
    pub fn insert_string(&mut self, column_name: &str, value: &str) {
        self.insert_bytes(column_name, value.as_bytes());
    }

    /// Check whether `value` might be present in `column_name`.
    ///
    /// Returns [`PredicatePushdownResult::DefinitelyAbsent`] if:
    /// - No filter exists for `column_name`, **or**
    /// - The filter indicates the value is absent.
    ///
    /// Note: if no filter is registered for a column, returning
    /// `DefinitelyAbsent` would be incorrect (we have no information).
    /// Instead we return `MightContain` in that case to be safe.
    pub fn check(&self, column_name: &str, value: &[u8]) -> PredicatePushdownResult {
        match self.columns.get(column_name) {
            None => PredicatePushdownResult::MightContain,
            Some(filter) => {
                if filter.contains(value) {
                    PredicatePushdownResult::MightContain
                } else {
                    PredicatePushdownResult::DefinitelyAbsent
                }
            }
        }
    }

    /// Check whether a string value might be present in `column_name`.
    pub fn check_string(&self, column_name: &str, value: &str) -> PredicatePushdownResult {
        self.check(column_name, value.as_bytes())
    }

    /// Return an iterator over `(column_name, &BloomFilter)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &BloomFilter)> {
        self.columns.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Number of columns with registered filters.
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    /// Return `true` if no column filters have been registered.
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }
}

impl Default for BloomColumnIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_no_false_negatives() {
        // Guarantee: inserted items must always be found
        let mut bloom = BloomFilter::new(500, 0.01);
        let items: Vec<String> = (0..500).map(|i| format!("item_{i}")).collect();

        for item in &items {
            bloom.insert(item.as_bytes());
        }

        for item in &items {
            assert!(
                bloom.contains(item.as_bytes()),
                "false negative for {:?}",
                item
            );
        }
    }

    #[test]
    fn test_bloom_definitely_absent_for_unseen() {
        // Items never inserted should frequently be absent (though not guaranteed)
        let mut bloom = BloomFilter::new(10, 0.001); // very low fpr, small set
        bloom.insert(b"only_this");

        // With fpr ≈ 0.001, unseen items are absent with probability ~99.9%
        // We check 100 items; statistically at least some should be absent
        let unseen: Vec<String> = (1000..1100).map(|i| format!("never_{i}")).collect();
        let absent_count = unseen
            .iter()
            .filter(|s| !bloom.contains(s.as_bytes()))
            .count();

        assert!(
            absent_count > 50,
            "too many false positives: only {absent_count}/100 correctly absent"
        );
    }

    #[test]
    fn test_bloom_column_index_check() {
        let mut idx = BloomColumnIndex::new();
        idx.add_column("city", 100, 0.01);

        let cities = ["Berlin", "Paris", "Tokyo", "London"];
        for city in &cities {
            idx.insert_string("city", city);
        }

        for city in &cities {
            let result = idx.check_string("city", city);
            assert_eq!(
                result,
                PredicatePushdownResult::MightContain,
                "inserted city {city:?} should be MightContain"
            );
        }

        // Check a completely different string (very likely absent)
        // Try many candidates to ensure at least one is absent
        let mut found_absent = false;
        for i in 0..1000 {
            let candidate = format!("NOT_A_CITY_{i}_xyz_never_inserted");
            if idx.check_string("city", &candidate)
                == PredicatePushdownResult::DefinitelyAbsent
            {
                found_absent = true;
                break;
            }
        }
        assert!(found_absent, "expected at least one definitely-absent result");
    }

    #[test]
    fn test_bloom_column_index_missing_column() {
        let idx = BloomColumnIndex::new();
        // No filter registered → MightContain (safe default)
        let result = idx.check_string("nonexistent_col", "any_value");
        assert_eq!(result, PredicatePushdownResult::MightContain);
    }

    #[test]
    fn test_bloom_fpr_estimate_increases_with_load() {
        let mut bloom = BloomFilter::new(100, 0.01);
        let fpr0 = bloom.false_positive_rate_estimate();

        for i in 0u64..50 {
            bloom.insert(&i.to_le_bytes());
        }
        let fpr50 = bloom.false_positive_rate_estimate();

        for i in 50u64..200 {
            bloom.insert(&i.to_le_bytes());
        }
        let fpr200 = bloom.false_positive_rate_estimate();

        // FPR should increase monotonically as more items are inserted
        assert!(fpr0 <= fpr50, "fpr should increase: {fpr0} -> {fpr50}");
        assert!(fpr50 <= fpr200, "fpr should increase: {fpr50} -> {fpr200}");
    }

    #[test]
    fn test_bloom_new_params_reasonable() {
        let bloom = BloomFilter::new(1000, 0.01);
        // m ≈ 9586 bits → rounded up to nearest 64 ≈ 9600; k ≈ 7
        assert!(bloom.n_bits() >= 1000, "filter too small");
        assert!(bloom.n_hashes() >= 1, "need at least 1 hash");
        assert!(bloom.n_hashes() <= 30, "too many hashes");
        assert_eq!(bloom.n_elements(), 0);
    }

    #[test]
    fn test_bloom_index_default() {
        let idx = BloomColumnIndex::default();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
    }
}
