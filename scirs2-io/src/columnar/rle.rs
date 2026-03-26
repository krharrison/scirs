//! Run-Length Encoding (RLE) codec for columnar data.
//!
//! RLE compresses sequences of repeated values into `(value, count)` pairs.
//! Highly effective for sorted or low-cardinality columns.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A single run in an RLE-encoded sequence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RleRun<T: Clone + PartialEq> {
    /// The repeated value.
    pub value: T,
    /// Number of consecutive occurrences.
    pub count: u32,
}

impl<T: Clone + PartialEq> RleRun<T> {
    /// Create a new run.
    pub fn new(value: T, count: u32) -> Self {
        Self { value, count }
    }
}

// ---------------------------------------------------------------------------
// Generic helpers (private)
// ---------------------------------------------------------------------------

fn encode_generic<T: Clone + PartialEq>(data: &[T]) -> Vec<RleRun<T>> {
    let mut runs: Vec<RleRun<T>> = Vec::new();
    let mut iter = data.iter();

    let first = match iter.next() {
        Some(v) => v,
        None => return runs,
    };

    let mut current_value = first.clone();
    let mut current_count: u32 = 1;

    for val in iter {
        if *val == current_value {
            current_count = current_count.saturating_add(1);
        } else {
            runs.push(RleRun::new(current_value.clone(), current_count));
            current_value = val.clone();
            current_count = 1;
        }
    }
    runs.push(RleRun::new(current_value, current_count));
    runs
}

fn decode_generic<T: Clone + PartialEq>(runs: &[RleRun<T>]) -> Vec<T> {
    let total: usize = runs.iter().map(|r| r.count as usize).sum();
    let mut out = Vec::with_capacity(total);
    for run in runs {
        for _ in 0..run.count {
            out.push(run.value.clone());
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Public encoder
// ---------------------------------------------------------------------------

/// Stateless RLE encoder/decoder for multiple data types.
pub struct RleEncoder;

impl RleEncoder {
    /// Encode a slice of `i64` values into RLE runs.
    pub fn encode_i64(data: &[i64]) -> Vec<RleRun<i64>> {
        encode_generic(data)
    }

    /// Decode RLE runs back to a `Vec<i64>`.
    pub fn decode_i64(runs: &[RleRun<i64>]) -> Vec<i64> {
        decode_generic(runs)
    }

    /// Encode a slice of `f64` values into RLE runs.
    ///
    /// Equality is tested by bit representation to avoid IEEE-754 NaN
    /// gotchas (two NaN values with identical bits are treated as equal).
    pub fn encode_f64(data: &[f64]) -> Vec<RleRun<f64>> {
        let mut runs: Vec<RleRun<f64>> = Vec::new();
        let mut iter = data.iter();

        let first = match iter.next() {
            Some(v) => v,
            None => return runs,
        };

        let mut current_value = *first;
        let mut current_bits = first.to_bits();
        let mut current_count: u32 = 1;

        for val in iter {
            let bits = val.to_bits();
            if bits == current_bits {
                current_count = current_count.saturating_add(1);
            } else {
                runs.push(RleRun::new(current_value, current_count));
                current_value = *val;
                current_bits = bits;
                current_count = 1;
            }
        }
        runs.push(RleRun::new(current_value, current_count));
        runs
    }

    /// Decode `f64` RLE runs back to a `Vec<f64>`.
    pub fn decode_f64(runs: &[RleRun<f64>]) -> Vec<f64> {
        decode_generic(runs)
    }

    /// Encode string slices into RLE runs (stores owned `String`).
    pub fn encode_str(data: &[&str]) -> Vec<RleRun<String>> {
        let owned: Vec<String> = data.iter().map(|s| s.to_string()).collect();
        encode_generic(&owned)
    }

    /// Decode `String` RLE runs back to `Vec<String>`.
    pub fn decode_str(runs: &[RleRun<String>]) -> Vec<String> {
        decode_generic(runs)
    }

    /// Encode a byte slice using byte-level RLE.
    pub fn encode_bytes(data: &[u8]) -> Vec<RleRun<u8>> {
        encode_generic(data)
    }

    /// Decode byte RLE runs back to `Vec<u8>`.
    pub fn decode_bytes(runs: &[RleRun<u8>]) -> Vec<u8> {
        decode_generic(runs)
    }

    /// Estimate the compression ratio for an `i64` slice.
    ///
    /// Returns `original_bytes / compressed_bytes`.  The compressed size is
    /// estimated as `n_runs * (8 + 4)` bytes (8 for i64 value, 4 for u32 count).
    pub fn compression_ratio_i64(data: &[i64]) -> f64 {
        if data.is_empty() {
            return 1.0;
        }
        let original_bytes = data.len() * 8;
        let runs = Self::encode_i64(data);
        let compressed_bytes = runs.len() * 12; // 8 (i64) + 4 (u32)
        if compressed_bytes == 0 {
            return f64::INFINITY;
        }
        original_bytes as f64 / compressed_bytes as f64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rle_encode_i64_all_same() {
        let data: Vec<i64> = vec![42; 100];
        let runs = RleEncoder::encode_i64(&data);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].value, 42);
        assert_eq!(runs[0].count, 100);
    }

    #[test]
    fn test_rle_encode_i64_alternating() {
        let data: Vec<i64> = (0..10).map(|i| i % 2).collect();
        // [0,1,0,1,0,1,0,1,0,1] → 10 runs (each count=1)
        let runs = RleEncoder::encode_i64(&data);
        assert_eq!(runs.len(), 10);
        for run in &runs {
            assert_eq!(run.count, 1);
        }
    }

    #[test]
    fn test_rle_decode_roundtrip_i64() {
        let original: Vec<i64> = vec![1, 1, 2, 3, 3, 3, 4, 4, 5];
        let runs = RleEncoder::encode_i64(&original);
        let decoded = RleEncoder::decode_i64(&runs);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_rle_decode_roundtrip_bytes() {
        let original: Vec<u8> = vec![0, 0, 0, 1, 2, 2, 3, 3, 3, 3];
        let runs = RleEncoder::encode_bytes(&original);
        let decoded = RleEncoder::decode_bytes(&runs);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_rle_compression_ratio_sorted() {
        // Ascending sequence — no repeated values → ratio < 1 (expansion)
        let data: Vec<i64> = (0..100).collect();
        let ratio = RleEncoder::compression_ratio_i64(&data);
        // 100 runs × 12 bytes vs 100 × 8 bytes → ratio ≈ 0.67
        assert!(
            ratio < 1.0,
            "sorted unique data should expand: ratio={ratio}"
        );

        // Constant sequence — ratio >> 1
        let constant: Vec<i64> = vec![7; 1000];
        let ratio2 = RleEncoder::compression_ratio_i64(&constant);
        assert!(
            ratio2 > 50.0,
            "constant data should compress well: ratio={ratio2}"
        );
    }

    #[test]
    fn test_rle_empty_slice() {
        let runs: Vec<RleRun<i64>> = RleEncoder::encode_i64(&[]);
        assert!(runs.is_empty());
        let decoded = RleEncoder::decode_i64(&[]);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_rle_encode_f64_roundtrip() {
        let original: Vec<f64> = vec![1.0, 1.0, 2.5, 2.5, 2.5, f64::NAN, f64::NAN];
        let runs = RleEncoder::encode_f64(&original);
        let decoded = RleEncoder::decode_f64(&runs);
        // NaN bit identity: compare bits
        assert_eq!(decoded.len(), original.len());
        for (a, b) in original.iter().zip(decoded.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn test_rle_encode_str_roundtrip() {
        let data = vec!["a", "a", "b", "c", "c"];
        let runs = RleEncoder::encode_str(&data);
        assert_eq!(runs.len(), 3);
        let decoded = RleEncoder::decode_str(&runs);
        assert_eq!(decoded, vec!["a", "a", "b", "c", "c"]);
    }
}
