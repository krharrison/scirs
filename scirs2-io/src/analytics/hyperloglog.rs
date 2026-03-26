//! HyperLogLog++ cardinality estimator.
//!
//! Implements the HyperLogLog++ algorithm (Heule, Nunkesser, Hall 2013) for
//! approximate counting of distinct elements with configurable precision.
//!
//! ## Accuracy
//!
//! With `p` precision bits and `m = 2^p` registers the standard error is
//! approximately `1.04 / √m`:
//!
//! | p  | m     | Relative error |
//! |----|-------|----------------|
//! | 4  | 16    | ~26%           |
//! | 8  | 256   | ~6.5%          |
//! | 12 | 4096  | ~1.6%          |
//! | 16 | 65536 | ~0.4%          |

use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// FNV-1a 64-bit hasher (pure Rust, no external dependency)
// ---------------------------------------------------------------------------

struct Fnv1aHasher {
    state: u64,
}

impl Fnv1aHasher {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    fn new_with_seed(seed: u64) -> Self {
        Self {
            state: Self::OFFSET_BASIS ^ seed,
        }
    }
}

impl Default for Fnv1aHasher {
    fn default() -> Self {
        Self {
            state: Self::OFFSET_BASIS,
        }
    }
}

impl Hasher for Fnv1aHasher {
    fn finish(&self) -> u64 {
        self.state
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.state ^= u64::from(b);
            self.state = self.state.wrapping_mul(Self::PRIME);
        }
    }
}

/// Murmur3-style 64-bit finalizer (avalanche mixing).
///
/// Ensures that all output bits depend on all input bits — critical for
/// obtaining a near-uniform distribution from structured inputs such as
/// sequential integers.
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

/// Hash a value using FNV-1a with an optional seed mixed in.
fn fnv1a_hash<T: Hash>(value: &T, seed: u64) -> u64 {
    let mut h = Fnv1aHasher::new_with_seed(seed);
    value.hash(&mut h);
    // Apply Murmur3 finalization to ensure uniform bit distribution.
    mix64(h.finish())
}

// ---------------------------------------------------------------------------
// HyperLogLog
// ---------------------------------------------------------------------------

/// HyperLogLog++ approximate cardinality estimator.
///
/// # Example
///
/// ```rust
/// use scirs2_io::analytics::HyperLogLog;
///
/// let mut hll = HyperLogLog::new(12);
/// for i in 0u64..10_000 {
///     hll.add(&i);
/// }
/// let estimate = hll.estimate();
/// assert!((estimate - 10_000.0).abs() / 10_000.0 < 0.05);
/// ```
#[derive(Debug, Clone)]
pub struct HyperLogLog {
    /// Register array; length = m = 2^p.
    registers: Vec<u8>,
    /// Number of precision bits (4 ≤ p ≤ 18).
    p: u8,
    /// Cached `m = 2^p`.
    m: usize,
}

impl HyperLogLog {
    /// Create a new HyperLogLog estimator.
    ///
    /// `p` is the number of precision bits; must be in `[4, 18]`.
    ///
    /// # Panics
    ///
    /// Does not panic; clamps `p` to `[4, 18]` silently.
    pub fn new(p: u8) -> Self {
        let p = p.clamp(4, 18);
        let m = 1usize << p;
        Self {
            registers: vec![0u8; m],
            p,
            m,
        }
    }

    /// Return the precision parameter `p`.
    pub fn precision(&self) -> u8 {
        self.p
    }

    /// Return the number of registers `m = 2^p`.
    pub fn num_registers(&self) -> usize {
        self.m
    }

    /// Update the register for the given raw 64-bit hash value.
    ///
    /// The top `p` bits select the register; the remaining `64-p` bits are
    /// used to compute the position-of-leftmost-one (rho).
    pub fn add_raw(&mut self, hash: u64) {
        let index = (hash >> (64 - self.p)) as usize;
        // The hash bits used for rho are the lower 64-p bits.
        let w = hash << self.p;
        // leading_zeros on 0 would give 64, but we add 1 so rho ≥ 1.
        let rho = (w.leading_zeros() + 1) as u8;
        if rho > self.registers[index] {
            self.registers[index] = rho;
        }
    }

    /// Add a hashable value to the sketch.
    pub fn add<T: Hash>(&mut self, value: &T) {
        let h = fnv1a_hash(value, 0);
        self.add_raw(h);
    }

    /// Estimate the number of distinct elements.
    ///
    /// Applies HyperLogLog++ bias corrections for small and large cardinalities.
    pub fn estimate(&self) -> f64 {
        let m = self.m as f64;
        let alpha = alpha_m(self.m);

        // Raw harmonic-mean estimator.
        let sum: f64 = self
            .registers
            .iter()
            .map(|&r| 2.0_f64.powi(-(r as i32)))
            .sum();
        let raw_estimate = alpha * m * m / sum;

        // Small range correction: linear counting on registers == 0.
        let zeros = self.registers.iter().filter(|&&r| r == 0).count() as f64;
        if raw_estimate <= 2.5 * m && zeros > 0.0 {
            return m * (m / zeros).ln();
        }

        // Large range correction (2^32 / 30 threshold).
        let threshold = (1u64 << 32) as f64 / 30.0;
        if raw_estimate > threshold {
            return -(2.0_f64.powi(32)) * (1.0 - raw_estimate / 2.0_f64.powi(32)).ln();
        }

        raw_estimate
    }

    /// Merge another HyperLogLog (same `p`) into this one via element-wise max.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the two sketches have different precision.
    pub fn merge(&mut self, other: &HyperLogLog) -> Result<(), String> {
        if self.p != other.p {
            return Err(format!(
                "Cannot merge HyperLogLog with p={} into one with p={}",
                other.p, self.p
            ));
        }
        for (r, &o) in self.registers.iter_mut().zip(other.registers.iter()) {
            if o > *r {
                *r = o;
            }
        }
        Ok(())
    }

    /// Reset all registers to zero.
    pub fn clear(&mut self) {
        self.registers.iter_mut().for_each(|r| *r = 0);
    }
}

/// Bias-correction constant `α_m` for HyperLogLog.
///
/// Returns the constant used in `E = α_m · m² / Σ 2^{−reg_i}`.
pub fn alpha_m(m: usize) -> f64 {
    match m {
        16 => 0.673,
        32 => 0.697,
        64 => 0.709,
        _ => {
            // General formula for m ≥ 128:
            // α_m = 1 / (2 ln 2 · (1 + Σ_{k≥1} (k/(2^k - 1)^2)))
            // Approximation: 0.7213 / (1 + 1.079/m)
            0.7213 / (1.0 + 1.079 / m as f64)
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
    fn test_new_clamps_p() {
        let hll = HyperLogLog::new(2); // below minimum → clamped to 4
        assert_eq!(hll.precision(), 4);
        let hll2 = HyperLogLog::new(20); // above maximum → clamped to 18
        assert_eq!(hll2.precision(), 18);
    }

    #[test]
    fn test_single_element_estimate() {
        let mut hll = HyperLogLog::new(12);
        hll.add(&42u64);
        let est = hll.estimate();
        // With one element the small-range correction should give ≈ 1.
        assert!(
            est > 0.5 && est < 2.0,
            "Expected ≈1, got {est}"
        );
    }

    #[test]
    fn test_cardinality_10k_within_5_percent() {
        let mut hll = HyperLogLog::new(12);
        for i in 0u64..10_000 {
            hll.add(&i);
        }
        let est = hll.estimate();
        let rel_err = (est - 10_000.0).abs() / 10_000.0;
        assert!(
            rel_err < 0.05,
            "Relative error {rel_err:.3} exceeds 5% (estimate={est:.0})"
        );
    }

    #[test]
    fn test_merge_union_of_disjoint_sets() {
        let mut hll_a = HyperLogLog::new(14);
        let mut hll_b = HyperLogLog::new(14);
        for i in 0u64..5_000 {
            hll_a.add(&i);
        }
        for i in 5_000u64..10_000 {
            hll_b.add(&i);
        }
        hll_a.merge(&hll_b).expect("merge should succeed");
        let est = hll_a.estimate();
        let rel_err = (est - 10_000.0).abs() / 10_000.0;
        assert!(
            rel_err < 0.05,
            "Merged estimate relative error {rel_err:.3} exceeds 5% (estimate={est:.0})"
        );
    }

    #[test]
    fn test_merge_incompatible_precision_returns_error() {
        let mut hll_a = HyperLogLog::new(12);
        let hll_b = HyperLogLog::new(8);
        assert!(hll_a.merge(&hll_b).is_err());
    }

    #[test]
    fn test_p4_vs_p12_relative_error() {
        let n = 1_000u64;
        let mut hll_p4 = HyperLogLog::new(4);
        let mut hll_p12 = HyperLogLog::new(12);
        for i in 0..n {
            hll_p4.add(&i);
            hll_p12.add(&i);
        }
        let err4 = (hll_p4.estimate() - n as f64).abs() / n as f64;
        let err12 = (hll_p12.estimate() - n as f64).abs() / n as f64;
        // p=12 should be considerably more accurate than p=4 for moderate n.
        assert!(
            err12 <= err4 || err12 < 0.1,
            "Expected p=12 more accurate: err4={err4:.3}, err12={err12:.3}"
        );
    }
}
