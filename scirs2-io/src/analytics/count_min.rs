//! Count-Min Sketch for approximate frequency counting.
//!
//! A probabilistic data structure that provides approximate frequency counts
//! for items in a stream with guaranteed no underestimation.
//!
//! ## Guarantees
//!
//! - **No underestimate**: `query(key) ≥ true_count(key)` always holds.
//! - **Error bound**: with probability ≥ 1 − δ the overestimate is ≤ ε · N,
//!   where N is the total weight added.
//!
//! ## Usage
//!
//! ```rust
//! use scirs2_io::analytics::CountMinSketch;
//!
//! let mut cms = CountMinSketch::new_with_error(0.01, 0.01);
//! cms.update(b"apple", 5);
//! cms.update(b"banana", 3);
//! assert!(cms.query(b"apple") >= 5);
//! assert_eq!(cms.query(b"unseen"), 0);
//! ```

use crate::error::{IoError, Result};

// ---------------------------------------------------------------------------
// FNV-1a hash with seed mixing
// ---------------------------------------------------------------------------

/// Compute a 64-bit FNV-1a hash of `data`, then mix in `seed`.
fn fnv1a_with_seed(data: &[u8], seed: u64) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    let mut hash = OFFSET_BASIS;
    for &b in data {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(PRIME);
    }
    // Mix in seed using a Fibonacci multiplicative hash step.
    (hash ^ seed).wrapping_mul(0x9e3779b97f4a7c15)
}

// ---------------------------------------------------------------------------
// CountMinSketch
// ---------------------------------------------------------------------------

/// Count-Min Sketch approximate frequency table.
#[derive(Debug, Clone)]
pub struct CountMinSketch {
    /// d × w table of counters.
    table: Vec<Vec<u64>>,
    /// Number of hash functions (rows).
    d: usize,
    /// Width of each row.
    w: usize,
    /// Per-row hash seeds.
    hash_seeds: Vec<u64>,
}

impl CountMinSketch {
    /// Construct a sketch with explicit dimensions and seeds.
    ///
    /// * `d` – number of hash functions (rows).
    /// * `w` – width (number of buckets per row).
    /// * `hash_seeds` – one seed per row; length must equal `d`.
    pub fn new(d: usize, w: usize, hash_seeds: Vec<u64>) -> Self {
        assert_eq!(hash_seeds.len(), d, "hash_seeds.len() must equal d");
        Self {
            table: vec![vec![0u64; w]; d],
            d,
            w,
            hash_seeds,
        }
    }

    /// Construct a sketch for a given error bound and failure probability.
    ///
    /// * `epsilon`  – additive error bound: `ε · N` where N = total weight.
    /// * `delta`    – failure probability (probability that guarantee fails).
    ///
    /// The dimensions are set to:
    /// - `w = ⌈e / ε⌉`
    /// - `d = ⌈ln(1/δ)⌉`
    ///
    /// Seeds are chosen deterministically from a fixed sequence so that
    /// construction is reproducible.
    pub fn new_with_error(epsilon: f64, delta: f64) -> Self {
        let epsilon = epsilon.max(1e-15);
        let delta = delta.clamp(1e-15, 1.0 - 1e-15);
        let w = ((std::f64::consts::E / epsilon).ceil() as usize).max(1);
        let d = ((1.0_f64 / delta).ln().ceil() as usize).max(1);
        // Reproducible seeds derived from a simple LCG.
        let seeds: Vec<u64> = (0..d)
            .map(|i| {
                let mut s = 0x123456789abcdef0u64.wrapping_add(i as u64 * 0xdeadbeef);
                // A few LCG steps for mixing.
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                s
            })
            .collect();
        Self::new(d, w, seeds)
    }

    /// Return the width `w`.
    pub fn width(&self) -> usize {
        self.w
    }

    /// Return the depth `d`.
    pub fn depth(&self) -> usize {
        self.d
    }

    /// Add `count` to the frequency of `key`.
    pub fn update(&mut self, key: &[u8], count: u64) {
        for j in 0..self.d {
            let h = fnv1a_with_seed(key, self.hash_seeds[j]) as usize % self.w;
            self.table[j][h] = self.table[j][h].saturating_add(count);
        }
    }

    /// Estimate the frequency of `key`.
    ///
    /// Guaranteed to be ≥ the true count; never returns 0 for an unseen key
    /// unless the underlying hash collision probability is zero.
    pub fn query(&self, key: &[u8]) -> u64 {
        (0..self.d)
            .map(|j| {
                let h = fnv1a_with_seed(key, self.hash_seeds[j]) as usize % self.w;
                self.table[j][h]
            })
            .min()
            .unwrap_or(0)
    }

    /// Merge `other` into `self` by element-wise addition.
    ///
    /// Both sketches must have the same dimensions (d, w).
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions differ.
    pub fn merge(&mut self, other: &CountMinSketch) -> Result<()> {
        if self.d != other.d || self.w != other.w {
            return Err(IoError::ValidationError(format!(
                "CountMinSketch dimension mismatch: self ({}, {}) vs other ({}, {})",
                self.d, self.w, other.d, other.w
            )));
        }
        for j in 0..self.d {
            for k in 0..self.w {
                self.table[j][k] = self.table[j][k].saturating_add(other.table[j][k]);
            }
        }
        Ok(())
    }

    /// Estimate the inner product `Σ_x count_A(x) · count_B(x)` using the
    /// minimum over all rows of the dot products of matching rows.
    ///
    /// Both sketches must have the same dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions differ.
    pub fn inner_product_estimate(&self, other: &CountMinSketch) -> Result<u64> {
        if self.d != other.d || self.w != other.w {
            return Err(IoError::ValidationError(format!(
                "CountMinSketch dimension mismatch for inner product: ({}, {}) vs ({}, {})",
                self.d, self.w, other.d, other.w
            )));
        }
        let min_dot = (0..self.d)
            .map(|j| {
                self.table[j]
                    .iter()
                    .zip(other.table[j].iter())
                    .map(|(&a, &b)| a.saturating_mul(b))
                    .fold(0u64, |acc, v| acc.saturating_add(v))
            })
            .min()
            .unwrap_or(0);
        Ok(min_dot)
    }

    /// Reset all counters to zero.
    pub fn clear(&mut self) {
        for row in &mut self.table {
            row.iter_mut().for_each(|c| *c = 0);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_unseen_key_is_zero() {
        let cms = CountMinSketch::new_with_error(0.01, 0.01);
        assert_eq!(cms.query(b"ghost"), 0);
    }

    #[test]
    fn test_update_then_query() {
        let mut cms = CountMinSketch::new_with_error(0.01, 0.01);
        cms.update(b"alpha", 7);
        let q = cms.query(b"alpha");
        assert!(q >= 7, "Expected query ≥ 7, got {q}");
    }

    #[test]
    fn test_no_underestimate() {
        let mut cms = CountMinSketch::new_with_error(0.001, 0.001);
        let key = b"test_key";
        let count = 42u64;
        cms.update(key, count);
        assert!(cms.query(key) >= count);
    }

    #[test]
    fn test_merge_combined_counts() {
        let mut cms_a = CountMinSketch::new_with_error(0.01, 0.01);
        let mut cms_b = CountMinSketch::new_with_error(0.01, 0.01);
        cms_a.update(b"x", 10);
        cms_b.update(b"x", 15);
        cms_a.merge(&cms_b).expect("merge should succeed");
        assert!(cms_a.query(b"x") >= 25);
    }

    #[test]
    fn test_merge_dimension_mismatch_returns_error() {
        let mut cms_a = CountMinSketch::new(3, 10, vec![1, 2, 3]);
        let cms_b = CountMinSketch::new(4, 10, vec![1, 2, 3, 4]);
        assert!(cms_a.merge(&cms_b).is_err());
    }

    #[test]
    fn test_inner_product_positive_for_overlapping_keys() {
        let mut cms_a = CountMinSketch::new_with_error(0.01, 0.01);
        let mut cms_b = CountMinSketch::new_with_error(0.01, 0.01);
        cms_a.update(b"shared", 5);
        cms_b.update(b"shared", 3);
        let ip = cms_a.inner_product_estimate(&cms_b).expect("inner product should succeed");
        assert!(ip > 0, "Inner product should be positive for overlapping keys, got {ip}");
    }

    #[test]
    fn test_dimensions_from_epsilon_delta() {
        let epsilon = 0.01;
        let delta = 0.01;
        let cms = CountMinSketch::new_with_error(epsilon, delta);
        // w = ceil(e / epsilon) ≈ ceil(271.8) = 272
        let expected_w = ((std::f64::consts::E / epsilon).ceil() as usize).max(1);
        // d = ceil(ln(100)) ≈ ceil(4.605) = 5
        let expected_d = ((1.0_f64 / delta).ln().ceil() as usize).max(1);
        assert_eq!(cms.width(), expected_w);
        assert_eq!(cms.depth(), expected_d);
    }
}
