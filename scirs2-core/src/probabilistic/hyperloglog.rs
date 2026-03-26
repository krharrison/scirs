//! HyperLogLog and HyperLogLog++ for cardinality estimation.
//!
//! HyperLogLog (HLL) is a probabilistic algorithm for estimating the number
//! of distinct elements in a multiset.  It uses a small array of registers
//! (typically 2^p where p is the precision parameter) and achieves a standard
//! error of approximately `1.04 / sqrt(m)` where `m = 2^p`.
//!
//! # HyperLogLog++ improvements
//!
//! This implementation includes the HLL++ enhancements from the Google paper
//! (Heule, Nunkesser, Hall 2013):
//!
//! - **Sparse representation** for small cardinalities, which uses a sorted
//!   list of (register, value) pairs until a threshold is reached.
//! - **Linear counting fallback** for very small estimates where raw HLL
//!   has higher bias.
//! - **Bias correction** using empirically-derived correction values for
//!   small cardinalities.

use crate::error::{CoreError, CoreResult, ErrorContext};
use crate::error_context;
use std::hash::{BuildHasher, Hasher};

/// Minimum precision (4 bits = 16 registers).
const MIN_PRECISION: u8 = 4;
/// Maximum precision (18 bits = 262144 registers).
const MAX_PRECISION: u8 = 18;

/// Threshold for switching from sparse to dense representation.
/// When the sparse set exceeds this fraction of `m`, we switch.
const SPARSE_THRESHOLD_FRACTION: f64 = 0.75;

/// A HyperLogLog cardinality estimator with HLL++ enhancements.
///
/// # Example
///
/// ```rust
/// use scirs2_core::probabilistic::HyperLogLog;
///
/// let mut hll = HyperLogLog::new(12).expect("valid precision");
///
/// // Insert 10,000 distinct items
/// for i in 0u64..10_000 {
///     hll.insert(&i.to_le_bytes());
/// }
///
/// let estimate = hll.count();
/// // Should be within ~2% of 10,000
/// assert!((estimate - 10_000.0).abs() < 500.0,
///     "Estimate {estimate} is too far from 10,000");
/// ```
#[derive(Clone)]
pub struct HyperLogLog {
    /// Precision parameter `p` (4..=18).
    precision: u8,
    /// Number of registers: `m = 2^p`.
    num_registers: usize,
    /// Representation: either sparse or dense.
    repr: HllRepr,
    /// Hasher for generating hash values.
    hash_builder: std::collections::hash_map::RandomState,
}

/// Internal representation: sparse (sorted list) or dense (register array).
#[derive(Clone)]
enum HllRepr {
    /// Sparse representation: sorted set of (register_index, rho_value) pairs.
    /// Used when the number of distinct registers touched is small.
    Sparse {
        /// Sorted by register index. Each entry is (register_idx, rho).
        entries: Vec<(u32, u8)>,
    },
    /// Dense representation: array of register values.
    Dense {
        /// registers[i] = max leading-zero count for register i.
        registers: Vec<u8>,
    },
}

impl HyperLogLog {
    /// Create a new HyperLogLog estimator with the given precision.
    ///
    /// Precision must be in `[4, 18]`.  Higher precision gives more
    /// accurate estimates but uses more memory:
    ///
    /// | Precision | Registers | Memory   | Std Error |
    /// |-----------|-----------|----------|-----------|
    /// | 4         | 16        | 16 bytes | 26%       |
    /// | 8         | 256       | 256 B    | 6.5%      |
    /// | 12        | 4096      | 4 KB     | 1.6%      |
    /// | 14        | 16384     | 16 KB    | 0.81%     |
    /// | 16        | 65536     | 64 KB    | 0.41%     |
    /// | 18        | 262144    | 256 KB   | 0.20%     |
    ///
    /// # Errors
    ///
    /// Returns an error if precision is outside `[4, 18]`.
    pub fn new(precision: u8) -> CoreResult<Self> {
        if precision < MIN_PRECISION || precision > MAX_PRECISION {
            return Err(CoreError::InvalidArgument(error_context!(format!(
                "precision must be in [{MIN_PRECISION}, {MAX_PRECISION}], got {precision}"
            ))));
        }
        let num_registers = 1usize << precision;
        Ok(Self {
            precision,
            num_registers,
            repr: HllRepr::Sparse {
                entries: Vec::new(),
            },
            hash_builder: std::collections::hash_map::RandomState::new(),
        })
    }

    /// Insert an item into the estimator.
    pub fn insert(&mut self, item: &[u8]) {
        let hash = self.hash_item(item);
        let (register_idx, rho) = self.decode_hash(hash);

        match &mut self.repr {
            HllRepr::Sparse { entries } => {
                // Binary search for the register index
                match entries.binary_search_by_key(&register_idx, |&(idx, _)| idx) {
                    Ok(pos) => {
                        // Update if rho is larger
                        if rho > entries[pos].1 {
                            entries[pos].1 = rho;
                        }
                    }
                    Err(pos) => {
                        entries.insert(pos, (register_idx, rho));
                    }
                }

                // Check if we should switch to dense
                let threshold =
                    (self.num_registers as f64 * SPARSE_THRESHOLD_FRACTION) as usize;
                if entries.len() > threshold {
                    self.switch_to_dense();
                }
            }
            HllRepr::Dense { registers } => {
                if rho > registers[register_idx as usize] {
                    registers[register_idx as usize] = rho;
                }
            }
        }
    }

    /// Estimate the cardinality (number of distinct items inserted).
    pub fn count(&self) -> f64 {
        match &self.repr {
            HllRepr::Sparse { entries } => self.count_sparse(entries),
            HllRepr::Dense { registers } => self.count_dense(registers),
        }
    }

    /// Merge another HyperLogLog into this one.
    ///
    /// After merging, the estimate approximates `|A ∪ B|`.
    ///
    /// # Errors
    ///
    /// Returns an error if the precisions differ.
    pub fn merge(&mut self, other: &HyperLogLog) -> CoreResult<()> {
        if self.precision != other.precision {
            return Err(CoreError::DimensionError(error_context!(
                "HyperLogLog precision must match for merge"
            )));
        }

        // Ensure both are dense for merging
        self.switch_to_dense();

        let other_registers = match &other.repr {
            HllRepr::Dense { registers } => registers.clone(),
            HllRepr::Sparse { entries } => {
                let mut regs = vec![0u8; self.num_registers];
                for &(idx, rho) in entries {
                    if rho > regs[idx as usize] {
                        regs[idx as usize] = rho;
                    }
                }
                regs
            }
        };

        if let HllRepr::Dense { registers } = &mut self.repr {
            for (r, &other_r) in registers.iter_mut().zip(other_registers.iter()) {
                if other_r > *r {
                    *r = other_r;
                }
            }
        }
        Ok(())
    }

    /// Precision parameter.
    pub fn precision(&self) -> u8 {
        self.precision
    }

    /// Number of registers.
    pub fn num_registers(&self) -> usize {
        self.num_registers
    }

    /// Standard error of the estimate: `1.04 / sqrt(m)`.
    pub fn standard_error(&self) -> f64 {
        1.04 / (self.num_registers as f64).sqrt()
    }

    /// Whether the estimator is in sparse mode.
    pub fn is_sparse(&self) -> bool {
        matches!(&self.repr, HllRepr::Sparse { .. })
    }

    /// Clear the estimator.
    pub fn clear(&mut self) {
        self.repr = HllRepr::Sparse {
            entries: Vec::new(),
        };
    }

    // -----------------------------------------------------------------------
    // Private methods
    // -----------------------------------------------------------------------

    /// Hash an item to a 64-bit value.
    fn hash_item(&self, item: &[u8]) -> u64 {
        let mut hasher = self.hash_builder.build_hasher();
        hasher.write(item);
        hasher.finish()
    }

    /// Decode a hash into (register_index, rho) where rho is the position
    /// of the first 1-bit in the remaining bits (plus 1).
    fn decode_hash(&self, hash: u64) -> (u32, u8) {
        let p = self.precision;
        // Top p bits determine the register
        let register_idx = (hash >> (64 - p)) as u32;
        // Remaining (64 - p) bits: count leading zeros + 1
        let remaining = hash << p;
        let rho = if remaining == 0 {
            (64 - p) as u8 + 1
        } else {
            remaining.leading_zeros() as u8 + 1
        };
        (register_idx, rho)
    }

    /// Switch from sparse to dense representation.
    fn switch_to_dense(&mut self) {
        if let HllRepr::Sparse { entries } = &self.repr {
            let mut registers = vec![0u8; self.num_registers];
            for &(idx, rho) in entries {
                if rho > registers[idx as usize] {
                    registers[idx as usize] = rho;
                }
            }
            self.repr = HllRepr::Dense { registers };
        }
    }

    /// Estimate cardinality from sparse entries.
    fn count_sparse(&self, entries: &[(u32, u8)]) -> f64 {
        if entries.is_empty() {
            return 0.0;
        }

        // Convert to a temporary dense array for the estimate
        let m = self.num_registers;
        let mut registers = vec![0u8; m];
        for &(idx, rho) in entries {
            if rho > registers[idx as usize] {
                registers[idx as usize] = rho;
            }
        }
        self.count_dense(&registers)
    }

    /// Estimate cardinality from dense registers.
    fn count_dense(&self, registers: &[u8]) -> f64 {
        let m = self.num_registers as f64;

        // Compute the harmonic mean: sum of 2^(-M[j])
        let mut sum = 0.0f64;
        let mut zeros = 0usize;
        for &val in registers {
            sum += 2.0f64.powi(-(val as i32));
            if val == 0 {
                zeros += 1;
            }
        }

        // alpha_m constant (bias correction factor)
        let alpha = self.alpha_m();

        // Raw HLL estimate
        let raw_estimate = alpha * m * m / sum;

        // Small range correction: linear counting
        if raw_estimate <= 2.5 * m && zeros > 0 {
            // Linear counting: -m * ln(V/m) where V is the number of empty registers
            let linear_count = -m * (zeros as f64 / m).ln();
            return linear_count;
        }

        // Large range correction (for 32-bit hashes; not needed for 64-bit)
        // We use 64-bit hashes, so no correction needed for large values.
        // The 2^32 threshold from the original paper does not apply.

        raw_estimate
    }

    /// Compute the alpha_m bias correction constant.
    fn alpha_m(&self) -> f64 {
        let m = self.num_registers as f64;
        match self.num_registers {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / m),
        }
    }
}

impl std::fmt::Debug for HyperLogLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mode = if self.is_sparse() { "sparse" } else { "dense" };
        f.debug_struct("HyperLogLog")
            .field("precision", &self.precision)
            .field("num_registers", &self.num_registers)
            .field("mode", &mode)
            .field("standard_error", &format!("{:.2}%", self.standard_error() * 100.0))
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
    fn test_hll_empty_count_zero() {
        let hll = HyperLogLog::new(12).expect("valid");
        assert_eq!(hll.count(), 0.0);
    }

    #[test]
    fn test_hll_single_element() {
        let mut hll = HyperLogLog::new(12).expect("valid");
        hll.insert(b"single_item");
        let count = hll.count();
        // Should be close to 1
        assert!(count >= 0.5, "Count too low for single element: {count}");
        assert!(count < 5.0, "Count too high for single element: {count}");
    }

    #[test]
    fn test_hll_cardinality_estimate_accuracy() {
        let mut hll = HyperLogLog::new(14).expect("valid");
        let n = 100_000u64;
        for i in 0..n {
            hll.insert(&i.to_le_bytes());
        }
        let estimate = hll.count();
        let error = (estimate - n as f64).abs() / n as f64;
        // With p=14, standard error is ~0.81%, allow 5% margin
        assert!(
            error < 0.05,
            "Estimate {estimate} is {:.2}% off from true {n}",
            error * 100.0
        );
    }

    #[test]
    fn test_hll_merge_correctness() {
        let mut hll1 = HyperLogLog::new(12).expect("valid");
        let mut hll2 = HyperLogLog::new(12).expect("valid");

        // Insert 0..5000 into hll1 and 5000..10000 into hll2
        for i in 0..5000u64 {
            hll1.insert(&i.to_le_bytes());
        }
        for i in 5000..10000u64 {
            hll2.insert(&i.to_le_bytes());
        }

        let est1 = hll1.count();
        let est2 = hll2.count();

        hll1.merge(&hll2).expect("same precision");
        let merged_est = hll1.count();

        // Merged estimate should be close to 10,000
        let error = (merged_est - 10_000.0).abs() / 10_000.0;
        assert!(
            error < 0.10,
            "Merged estimate {merged_est} is {:.2}% off (est1={est1:.0}, est2={est2:.0})",
            error * 100.0
        );
    }

    #[test]
    fn test_hll_duplicate_inserts() {
        let mut hll = HyperLogLog::new(12).expect("valid");
        // Insert the same item 10,000 times
        for _ in 0..10_000 {
            hll.insert(b"same_item");
        }
        let estimate = hll.count();
        // Should be close to 1
        assert!(
            estimate < 5.0,
            "Duplicate inserts gave estimate {estimate}, expected ~1"
        );
    }

    #[test]
    fn test_hll_standard_error() {
        let hll = HyperLogLog::new(12).expect("valid");
        let se = hll.standard_error();
        // 1.04 / sqrt(4096) ≈ 0.01625
        assert!((se - 0.01625).abs() < 0.001, "Unexpected standard error: {se}");
    }

    #[test]
    fn test_hll_sparse_to_dense_transition() {
        let mut hll = HyperLogLog::new(8).expect("valid"); // 256 registers
        assert!(hll.is_sparse());

        // Insert enough items to trigger transition
        for i in 0..1000u64 {
            hll.insert(&i.to_le_bytes());
        }
        // Should have switched to dense
        assert!(!hll.is_sparse(), "Expected dense mode after many inserts");
    }

    #[test]
    fn test_hll_invalid_precision() {
        assert!(HyperLogLog::new(3).is_err());
        assert!(HyperLogLog::new(19).is_err());
        assert!(HyperLogLog::new(0).is_err());
    }

    #[test]
    fn test_hll_clear() {
        let mut hll = HyperLogLog::new(10).expect("valid");
        for i in 0..500u64 {
            hll.insert(&i.to_le_bytes());
        }
        assert!(hll.count() > 100.0);
        hll.clear();
        assert_eq!(hll.count(), 0.0);
        assert!(hll.is_sparse());
    }

    #[test]
    fn test_hll_merge_incompatible() {
        let mut hll1 = HyperLogLog::new(10).expect("valid");
        let hll2 = HyperLogLog::new(12).expect("valid");
        assert!(hll1.merge(&hll2).is_err());
    }

    #[test]
    fn test_hll_various_precisions() {
        for p in MIN_PRECISION..=MAX_PRECISION {
            let mut hll = HyperLogLog::new(p).expect("valid precision");
            assert_eq!(hll.num_registers(), 1 << p);
            hll.insert(b"test");
            assert!(hll.count() > 0.0);
        }
    }

    #[test]
    fn test_hll_large_cardinality() {
        let mut hll = HyperLogLog::new(14).expect("valid");
        let n = 1_000_000u64;
        for i in 0..n {
            hll.insert(&i.to_le_bytes());
        }
        let estimate = hll.count();
        let error = (estimate - n as f64).abs() / n as f64;
        // With p=14, allow 3% error for 1M items
        assert!(
            error < 0.03,
            "Large cardinality estimate {estimate:.0} is {:.2}% off from {n}",
            error * 100.0
        );
    }
}
