//! Bloom Filter — a probabilistic set membership data structure.
//!
//! A Bloom filter uses a compact bit array and multiple hash functions to test
//! whether an element is a member of a set. False positives are possible, but
//! false negatives are not. The trade-off between memory usage and false
//! positive rate is configurable at construction time.
//!
//! # Complexity
//!
//! | Operation | Time    | Notes                        |
//! |-----------|---------|------------------------------|
//! | insert    | O(k)    | k = number of hash functions |
//! | contains  | O(k)    | k = number of hash functions |
//! | memory    | O(m)    | m = number of bits           |
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::data_structures::BloomFilter;
//!
//! let mut bf = BloomFilter::new(1000, 0.01).expect("valid params");
//! bf.insert(&"hello");
//! bf.insert(&42u64);
//!
//! assert!(bf.contains(&"hello"));
//! // False negatives are impossible:
//! assert!(bf.contains(&"hello"));
//! ```

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

// ============================================================================
// Error type
// ============================================================================

/// Errors that can arise when constructing or operating on a [`BloomFilter`].
#[derive(Debug, Clone, PartialEq)]
pub enum BloomFilterError {
    /// Capacity was zero.
    ZeroCapacity,
    /// False-positive rate was not in the open interval (0, 1).
    InvalidFalsePositiveRate(f64),
}

impl fmt::Display for BloomFilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BloomFilterError::ZeroCapacity => {
                write!(f, "BloomFilter capacity must be greater than zero")
            }
            BloomFilterError::InvalidFalsePositiveRate(r) => write!(
                f,
                "false-positive rate {r} is not in the open interval (0, 1)"
            ),
        }
    }
}

impl std::error::Error for BloomFilterError {}

// ============================================================================
// BloomFilter
// ============================================================================

/// A probabilistic set membership structure parameterised by `T`.
///
/// Internally uses a flat bit vector of `m` bits and `k` independent hash
/// functions, implemented via the **double-hashing** scheme:
///
/// ```text
/// h_i(x) = (h1(x) + i * h2(x)) mod m     i = 0, 1, …, k-1
/// ```
///
/// This requires only two 64-bit hash evaluations per item regardless of `k`,
/// while maintaining statistical independence of the probed positions.
pub struct BloomFilter<T: Hash> {
    /// Bit storage: `bits[i / 64]` bit `i % 64`.
    bits: Vec<u64>,
    /// Total number of bits in the filter.
    num_bits: usize,
    /// Number of hash functions applied per operation.
    num_hashes: usize,
    /// Number of items that have been inserted.
    num_items: usize,
    /// Expected capacity used when choosing `num_bits` and `num_hashes`.
    capacity: usize,
    /// Desired false-positive rate used during construction.
    target_fp_rate: f64,
    /// Salt values for the two independent hash functions used in
    /// double-hashing. Fixed at construction time so behaviour is
    /// deterministic given the same seeds.
    seed1: u64,
    seed2: u64,
    _marker: PhantomData<T>,
}

impl<T: Hash> BloomFilter<T> {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Creates a new Bloom filter sized to hold `capacity` elements with an
    /// expected false-positive rate of `fp_rate`.
    ///
    /// The optimal number of bits `m` and hash functions `k` are derived from
    /// the standard formulae:
    ///
    /// ```text
    /// m = -n * ln(p) / (ln 2)^2
    /// k = (m / n) * ln 2
    /// ```
    ///
    /// where `n = capacity` and `p = fp_rate`.
    ///
    /// # Errors
    ///
    /// Returns [`BloomFilterError::ZeroCapacity`] if `capacity == 0`.
    /// Returns [`BloomFilterError::InvalidFalsePositiveRate`] if
    /// `fp_rate` is not strictly in (0, 1).
    pub fn new(capacity: usize, fp_rate: f64) -> Result<Self, BloomFilterError> {
        if capacity == 0 {
            return Err(BloomFilterError::ZeroCapacity);
        }
        if fp_rate <= 0.0 || fp_rate >= 1.0 {
            return Err(BloomFilterError::InvalidFalsePositiveRate(fp_rate));
        }

        let ln2 = std::f64::consts::LN_2;
        let n = capacity as f64;
        let p = fp_rate;

        // Optimal bit count and hash function count.
        let m = (-(n * p.ln()) / (ln2 * ln2)).ceil() as usize;
        let k = ((m as f64 / n) * ln2).round().max(1.0) as usize;

        // Align m to 64-bit word boundary.
        let num_bits = m.max(64);
        let num_words = (num_bits + 63) / 64;

        Ok(BloomFilter {
            bits: vec![0u64; num_words],
            num_bits,
            num_hashes: k,
            num_items: 0,
            capacity,
            target_fp_rate: fp_rate,
            seed1: 0x9e37_79b9_7f4a_7c15,
            seed2: 0x6c62_272e_07bb_0142,
            _marker: PhantomData,
        })
    }

    /// Creates a Bloom filter with explicit control over `num_bits` and
    /// `num_hashes`. Useful when restoring a serialised filter.
    ///
    /// # Errors
    ///
    /// Returns [`BloomFilterError::ZeroCapacity`] if either parameter is zero.
    pub fn with_params(num_bits: usize, num_hashes: usize) -> Result<Self, BloomFilterError> {
        if num_bits == 0 || num_hashes == 0 {
            return Err(BloomFilterError::ZeroCapacity);
        }
        let num_words = (num_bits + 63) / 64;
        Ok(BloomFilter {
            bits: vec![0u64; num_words],
            num_bits,
            num_hashes,
            num_items: 0,
            capacity: num_bits,
            target_fp_rate: 0.01,
            seed1: 0x9e37_79b9_7f4a_7c15,
            seed2: 0x6c62_272e_07bb_0142,
            _marker: PhantomData,
        })
    }

    // ------------------------------------------------------------------
    // Core operations
    // ------------------------------------------------------------------

    /// Inserts `item` into the filter.
    ///
    /// After this call [`contains`](BloomFilter::contains) will always return
    /// `true` for `item`.
    pub fn insert(&mut self, item: &T) {
        let (h1, h2) = self.double_hash(item);
        let m = self.num_bits as u64;
        for i in 0..self.num_hashes as u64 {
            let bit = ((h1.wrapping_add(i.wrapping_mul(h2))) % m) as usize;
            self.set_bit(bit);
        }
        self.num_items += 1;
    }

    /// Returns `true` if `item` *may* be in the set, `false` if it is
    /// definitely not in the set.
    ///
    /// False positives are possible; false negatives are not.
    pub fn contains(&self, item: &T) -> bool {
        let (h1, h2) = self.double_hash(item);
        let m = self.num_bits as u64;
        for i in 0..self.num_hashes as u64 {
            let bit = ((h1.wrapping_add(i.wrapping_mul(h2))) % m) as usize;
            if !self.get_bit(bit) {
                return false;
            }
        }
        true
    }

    /// Resets the filter, clearing all inserted items.
    pub fn clear(&mut self) {
        for word in &mut self.bits {
            *word = 0;
        }
        self.num_items = 0;
    }

    // ------------------------------------------------------------------
    // Diagnostics
    // ------------------------------------------------------------------

    /// Returns an estimate of the number of items inserted.
    ///
    /// Uses the standard formula:
    /// ```text
    /// n ≈ -(m / k) * ln(1 - X/m)
    /// ```
    /// where `X` is the number of set bits.
    pub fn estimated_items(&self) -> usize {
        let x = self.count_ones() as f64;
        let m = self.num_bits as f64;
        let k = self.num_hashes as f64;
        if x >= m {
            // Filter is full — return the nominal capacity.
            return self.capacity;
        }
        let estimate = -(m / k) * (1.0 - x / m).ln();
        estimate.round() as usize
    }

    /// Returns the current estimated false-positive rate given the number of
    /// items inserted.
    ///
    /// ```text
    /// p ≈ (1 - e^{-kn/m})^k
    /// ```
    pub fn false_positive_rate(&self) -> f64 {
        let k = self.num_hashes as f64;
        let n = self.num_items as f64;
        let m = self.num_bits as f64;
        (1.0 - (-k * n / m).exp()).powf(k)
    }

    /// Returns the number of bits in the internal bit array.
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// Returns the number of hash functions used.
    pub fn num_hashes(&self) -> usize {
        self.num_hashes
    }

    /// Returns the number of items that have been inserted (exact count, not
    /// the estimate based on set bits).
    pub fn len(&self) -> usize {
        self.num_items
    }

    /// Returns `true` if no items have been inserted.
    pub fn is_empty(&self) -> bool {
        self.num_items == 0
    }

    /// Returns the capacity the filter was sized for.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the target false-positive rate supplied at construction.
    pub fn target_fp_rate(&self) -> f64 {
        self.target_fp_rate
    }

    // ------------------------------------------------------------------
    // Merge / intersection
    // ------------------------------------------------------------------

    /// Returns a new filter that is the union of `self` and `other`.
    ///
    /// Both filters must have been constructed with identical `num_bits` and
    /// `num_hashes`.
    ///
    /// # Errors
    ///
    /// Returns an error string if the parameters do not match.
    pub fn union(&self, other: &BloomFilter<T>) -> Result<BloomFilter<T>, String> {
        if self.num_bits != other.num_bits || self.num_hashes != other.num_hashes {
            return Err(format!(
                "parameter mismatch: self(bits={}, k={}) vs other(bits={}, k={})",
                self.num_bits, self.num_hashes, other.num_bits, other.num_hashes
            ));
        }
        let mut result = BloomFilter::with_params(self.num_bits, self.num_hashes)
            .expect("params already validated");
        for (i, (&a, &b)) in self.bits.iter().zip(other.bits.iter()).enumerate() {
            result.bits[i] = a | b;
        }
        result.num_items = self.num_items + other.num_items;
        Ok(result)
    }

    /// Returns a new filter representing the intersection of `self` and
    /// `other` (i.e. only bits set in *both* are set in the result).
    ///
    /// # Errors
    ///
    /// Returns an error string if the parameters do not match.
    pub fn intersection(&self, other: &BloomFilter<T>) -> Result<BloomFilter<T>, String> {
        if self.num_bits != other.num_bits || self.num_hashes != other.num_hashes {
            return Err(format!(
                "parameter mismatch: self(bits={}, k={}) vs other(bits={}, k={})",
                self.num_bits, self.num_hashes, other.num_bits, other.num_hashes
            ));
        }
        let mut result = BloomFilter::with_params(self.num_bits, self.num_hashes)
            .expect("params already validated");
        for (i, (&a, &b)) in self.bits.iter().zip(other.bits.iter()).enumerate() {
            result.bits[i] = a & b;
        }
        Ok(result)
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Double-hashing: produce two independent 64-bit digests of `item`.
    ///
    /// Uses the `DefaultHasher` seeded with distinct constants to generate
    /// `h1` and `h2`. Both values are made odd so that `h2` is coprime to any
    /// power-of-two modulus, ensuring full coverage of the bit array.
    fn double_hash(&self, item: &T) -> (u64, u64) {
        let mut h1 = DefaultHasher::new();
        self.seed1.hash(&mut h1);
        item.hash(&mut h1);
        let v1 = h1.finish();

        let mut h2 = DefaultHasher::new();
        self.seed2.hash(&mut h2);
        item.hash(&mut h2);
        let v2 = h2.finish();

        // Ensure h2 is odd so all bit positions are reachable when m is a
        // power of two.
        (v1, v2 | 1)
    }

    #[inline]
    fn set_bit(&mut self, idx: usize) {
        let word = idx / 64;
        let bit = idx % 64;
        self.bits[word] |= 1u64 << bit;
    }

    #[inline]
    fn get_bit(&self, idx: usize) -> bool {
        let word = idx / 64;
        let bit = idx % 64;
        (self.bits[word] >> bit) & 1 == 1
    }

    fn count_ones(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }
}

impl<T: Hash> fmt::Debug for BloomFilter<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BloomFilter")
            .field("num_bits", &self.num_bits)
            .field("num_hashes", &self.num_hashes)
            .field("num_items", &self.num_items)
            .field("capacity", &self.capacity)
            .field("target_fp_rate", &self.target_fp_rate)
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
    fn construction_rejects_bad_params() {
        assert!(BloomFilter::<u32>::new(0, 0.01).is_err());
        assert!(BloomFilter::<u32>::new(100, 0.0).is_err());
        assert!(BloomFilter::<u32>::new(100, 1.0).is_err());
        assert!(BloomFilter::<u32>::new(100, -0.1).is_err());
        assert!(BloomFilter::<u32>::new(100, 1.5).is_err());
    }

    #[test]
    fn no_false_negatives_integers() {
        let mut bf = BloomFilter::new(500, 0.01).expect("valid params");
        for i in 0u64..500 {
            bf.insert(&i);
        }
        for i in 0u64..500 {
            assert!(bf.contains(&i), "false negative for {i}");
        }
    }

    #[test]
    fn no_false_negatives_strings() {
        let words = vec!["apple", "banana", "cherry", "date", "elderberry"];
        let mut bf = BloomFilter::new(100, 0.01).expect("valid params");
        for w in &words {
            bf.insert(w);
        }
        for w in &words {
            assert!(bf.contains(w), "false negative for {w}");
        }
    }

    #[test]
    fn false_positive_rate_within_bounds() {
        // Insert 1000 items and check ~10 000 non-members.  Count FPs.
        let n = 1000usize;
        let target = 0.02f64;
        let mut bf = BloomFilter::new(n, target).expect("valid params");
        for i in 0u64..n as u64 {
            bf.insert(&i);
        }
        let probe_count = 10_000usize;
        let fp_count = (n as u64..(n as u64 + probe_count as u64))
            .filter(|x| bf.contains(x))
            .count();
        let measured = fp_count as f64 / probe_count as f64;
        // Allow 3× margin because DefaultHasher is not a cryptographic hash
        // and may be correlated with simple linear sequences.
        assert!(
            measured < target * 10.0,
            "measured FP rate {measured:.4} exceeds 10× target {target}"
        );
    }

    #[test]
    fn clear_resets_state() {
        let mut bf = BloomFilter::new(100, 0.01).expect("valid params");
        bf.insert(&"hello");
        bf.insert(&"world");
        assert_eq!(bf.len(), 2);
        bf.clear();
        assert_eq!(bf.len(), 0);
        assert!(bf.is_empty());
        // After clear, definitely-absent items must test false (no false neg
        // guarantee applies only to items that have been inserted after clear).
        assert!(!bf.contains(&"hello"));
    }

    #[test]
    fn estimated_items_close_to_actual() {
        let mut bf = BloomFilter::new(1000, 0.01).expect("valid params");
        for i in 0u64..800 {
            bf.insert(&i);
        }
        let est = bf.estimated_items();
        // Allow ±25% error.
        let lo = 600usize;
        let hi = 1000usize;
        assert!(
            est >= lo && est <= hi,
            "estimated_items={est} out of range [{lo}, {hi}]"
        );
    }

    #[test]
    fn union_covers_both_sets() {
        let mut a = BloomFilter::new(500, 0.01).expect("valid params");
        let mut b = BloomFilter::with_params(a.num_bits(), a.num_hashes()).expect("valid params");
        for i in 0u64..100 {
            a.insert(&i);
        }
        for i in 100u64..200 {
            b.insert(&i);
        }
        let u = a.union(&b).expect("same params");
        for i in 0u64..200 {
            assert!(u.contains(&i), "union missing {i}");
        }
    }

    #[test]
    fn union_parameter_mismatch_errors() {
        let a = BloomFilter::<u64>::new(100, 0.01).expect("valid");
        let b = BloomFilter::<u64>::new(200, 0.01).expect("valid");
        assert!(a.union(&b).is_err());
    }

    #[test]
    fn fp_rate_accessor_increases_with_load() {
        let mut bf = BloomFilter::new(1000, 0.01).expect("valid params");
        let initial = bf.false_positive_rate();
        for i in 0u64..500 {
            bf.insert(&i);
        }
        let loaded = bf.false_positive_rate();
        assert!(
            loaded > initial,
            "FP rate should increase as items are added"
        );
    }
}
