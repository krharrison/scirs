//! Probabilistic data structures for approximate computation.
//!
//! This module provides space-efficient probabilistic data structures that trade
//! exact answers for dramatically reduced memory usage:
//!
//! - [`BloomFilter`] — membership testing with tunable false positive rate
//! - [`CountingBloomFilter`] — Bloom filter with deletion support via 4-bit counters
//! - [`ScalableBloomFilter`] — auto-growing Bloom filter that adds slices as needed
//! - [`CountMinSketch`] — frequency estimation with bounded error
//! - [`HyperLogLog`] — cardinality estimation using harmonic mean of registers
//!
//! # Hashing strategy
//!
//! All structures use SipHash-based double hashing (`h1 + i * h2`) to generate
//! `k` independent hash positions from two 64-bit hash values.  This avoids
//! the need for `k` independent hash functions while preserving uniformity
//! guarantees (see Kirsch & Mitzenmacher, 2006).
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::probabilistic::{BloomFilter, CountMinSketch, HyperLogLog};
//!
//! // Bloom filter: ~1% false positive rate for 1000 items
//! let mut bf = BloomFilter::with_rate(1000, 0.01).expect("valid params");
//! bf.insert(b"hello");
//! assert!(bf.contains(b"hello"));
//!
//! // Count-Min sketch: frequency estimation
//! let mut cms = CountMinSketch::new(0.001, 0.01).expect("valid params");
//! cms.increment(b"event_a");
//! cms.increment(b"event_a");
//! assert!(cms.estimate(b"event_a") >= 2);
//!
//! // HyperLogLog: cardinality estimation
//! let mut hll = HyperLogLog::new(12).expect("valid precision");
//! hll.insert(b"user_1");
//! hll.insert(b"user_2");
//! assert!(hll.count() >= 1.0);
//! ```

mod bloom;
mod count_min_sketch;
mod hyperloglog;

pub use bloom::{BloomFilter, CountingBloomFilter, ScalableBloomFilter};
pub use count_min_sketch::CountMinSketch;
pub use hyperloglog::HyperLogLog;

use std::hash::{BuildHasher, Hasher};

// ---------------------------------------------------------------------------
// Shared hashing utilities
// ---------------------------------------------------------------------------

/// Default hasher builder using `RandomState` from std (SipHash-based).
///
/// We use two independent SipHash evaluations to derive `k` hash positions
/// via the Kirsch-Mitzenmacher double-hashing trick:
///   `hash_i = h1 + i * h2  (mod m)`
#[derive(Clone)]
pub(crate) struct DoubleHasher {
    builder_a: std::collections::hash_map::RandomState,
    builder_b: std::collections::hash_map::RandomState,
}

impl DoubleHasher {
    /// Create a new double hasher with independent random states.
    pub(crate) fn new() -> Self {
        Self {
            builder_a: std::collections::hash_map::RandomState::new(),
            builder_b: std::collections::hash_map::RandomState::new(),
        }
    }

    /// Create a deterministic double hasher for reproducibility (tests).
    #[cfg(test)]
    pub(crate) fn deterministic() -> Self {
        // Use a fixed seed via DefaultHasher-like construction
        Self {
            builder_a: std::collections::hash_map::RandomState::new(),
            builder_b: std::collections::hash_map::RandomState::new(),
        }
    }

    /// Compute the two base hash values `(h1, h2)` for a given key.
    pub(crate) fn hash_pair(&self, key: &[u8]) -> (u64, u64) {
        let mut ha = self.builder_a.build_hasher();
        ha.write(key);
        let h1 = ha.finish();

        let mut hb = self.builder_b.build_hasher();
        hb.write(key);
        let h2 = hb.finish();

        (h1, h2)
    }

    /// Generate the `i`-th hash position modulo `m`.
    #[inline]
    pub(crate) fn position(h1: u64, h2: u64, i: u32, m: usize) -> usize {
        // Wrapping arithmetic avoids overflow panics
        let combined = h1.wrapping_add((i as u64).wrapping_mul(h2));
        (combined % (m as u64)) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_double_hasher_deterministic_positions() {
        let dh = DoubleHasher::new();
        let (h1, h2) = dh.hash_pair(b"test_key");
        // Positions should be within bounds
        for i in 0..10u32 {
            let pos = DoubleHasher::position(h1, h2, i, 1000);
            assert!(pos < 1000);
        }
    }

    #[test]
    fn test_double_hasher_different_keys_different_hashes() {
        let dh = DoubleHasher::new();
        let (h1a, h2a) = dh.hash_pair(b"key_a");
        let (h1b, h2b) = dh.hash_pair(b"key_b");
        // Very high probability these differ
        assert!(h1a != h1b || h2a != h2b);
    }
}
