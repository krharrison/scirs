//! Binary protocol encoding utilities.
//!
//! This module provides low-level primitives used in wire protocols and
//! columnar storage formats:
//!
//! - [`VarInt`] — LEB128 variable-length unsigned integers
//! - [`ZigZag`] — ZigZag encoding for signed integers (used in Protobuf, etc.)
//! - [`BitPacking`] — compact bit-packed integer arrays
//! - [`DeltaEncoding`] — delta (first-differences) encoding for sorted/correlated data
//! - [`RunLengthEncoding`] — run-length encoding for repeated values
//!
//! # Example
//! ```rust
//! use scirs2_io::protocols::{VarInt, ZigZag, BitPacking, DeltaEncoding, RunLengthEncoding};
//!
//! // Variable-length integer (LEB128)
//! let enc = VarInt::encode(300);
//! let (val, n) = VarInt::decode(&enc).expect("decode ok");
//! assert_eq!(val, 300);
//! assert_eq!(n, enc.len());
//!
//! // ZigZag signed integers
//! assert_eq!(ZigZag::encode(-1), 1);
//! assert_eq!(ZigZag::decode(1), -1);
//!
//! // Bit-packing
//! let values = vec![0u64, 3, 1, 2, 3];
//! let packed = BitPacking::pack(&values, 2); // 2 bits each
//! let unpacked = BitPacking::unpack(&packed, 2, values.len());
//! assert_eq!(unpacked, values);
//!
//! // Delta encoding
//! let data = vec![10i64, 13, 16, 19];
//! let (first, deltas) = DeltaEncoding::encode(&data);
//! assert_eq!(DeltaEncoding::decode(first, &deltas), data);
//!
//! // Run-length encoding
//! let data = vec![1i32, 1, 1, 2, 2, 3];
//! let runs = RunLengthEncoding::encode(&data);
//! assert_eq!(RunLengthEncoding::decode(&runs), data);
//! ```

use crate::error::{IoError, Result};

// ─────────────────────────────── VarInt ──────────────────────────────────────

/// LEB128 variable-length unsigned integer encoding.
///
/// Each output byte contributes 7 bits to the value; the MSB is set on all
/// bytes except the last.
pub struct VarInt;

impl VarInt {
    /// Encode a `u64` value as LEB128 bytes.
    ///
    /// A value of 0 encodes to a single `[0x00]` byte.
    pub fn encode(mut v: u64) -> Vec<u8> {
        let mut buf = Vec::new();
        loop {
            let mut byte = (v & 0x7f) as u8;
            v >>= 7;
            if v != 0 {
                byte |= 0x80;
            }
            buf.push(byte);
            if v == 0 {
                break;
            }
        }
        buf
    }

    /// Decode a LEB128 encoded value from the start of `data`.
    ///
    /// Returns `(value, bytes_consumed)`.
    pub fn decode(data: &[u8]) -> Result<(u64, usize)> {
        let mut result: u64 = 0;
        let mut shift = 0u32;
        for (i, &byte) in data.iter().enumerate() {
            if shift >= 63 && byte > 1 {
                return Err(IoError::FormatError(format!(
                    "VarInt: overflow decoding byte 0x{byte:02x} at position {i}"
                )));
            }
            result |= ((byte & 0x7f) as u64) << shift;
            shift += 7;
            if byte & 0x80 == 0 {
                return Ok((result, i + 1));
            }
        }
        Err(IoError::FormatError(
            "VarInt: unexpected end of data".into(),
        ))
    }

    /// Encode a `usize`.
    pub fn encode_usize(v: usize) -> Vec<u8> {
        Self::encode(v as u64)
    }

    /// Decode to `usize`.
    pub fn decode_usize(data: &[u8]) -> Result<(usize, usize)> {
        let (v, n) = Self::decode(data)?;
        Ok((v as usize, n))
    }

    /// Encode a sequence of u64 values (each individually LEB128-encoded).
    pub fn encode_sequence(values: &[u64]) -> Vec<u8> {
        values.iter().flat_map(|&v| Self::encode(v)).collect()
    }

    /// Decode a sequence of `n` u64 values from LEB128 bytes.
    pub fn decode_sequence(data: &[u8], n: usize) -> Result<(Vec<u64>, usize)> {
        let mut values = Vec::with_capacity(n);
        let mut pos = 0;
        for _ in 0..n {
            if pos >= data.len() {
                return Err(IoError::FormatError(
                    "VarInt: not enough bytes for sequence".into(),
                ));
            }
            let (v, consumed) = Self::decode(&data[pos..])?;
            values.push(v);
            pos += consumed;
        }
        Ok((values, pos))
    }
}

// ─────────────────────────────── ZigZag ──────────────────────────────────────

/// ZigZag encoding for signed integers.
///
/// Maps signed integers to unsigned integers so that small-magnitude values
/// (both positive and negative) map to small unsigned values, making them
/// ideal for LEB128 compression.
///
/// ```text
/// 0  → 0
/// -1 → 1
/// 1  → 2
/// -2 → 3
/// 2  → 4
/// ```
pub struct ZigZag;

impl ZigZag {
    /// Encode `v: i64` → `u64`.
    ///
    /// Formula: `(v << 1) ^ (v >> 63)`
    #[inline]
    pub fn encode(v: i64) -> u64 {
        ((v << 1) ^ (v >> 63)) as u64
    }

    /// Decode `v: u64` → `i64`.
    ///
    /// Formula: `(v >>> 1) ^ -(v & 1)`
    #[inline]
    pub fn decode(v: u64) -> i64 {
        ((v >> 1) as i64) ^ -((v & 1) as i64)
    }

    /// Encode a sequence of i64 values.
    pub fn encode_sequence(values: &[i64]) -> Vec<u64> {
        values.iter().map(|&v| Self::encode(v)).collect()
    }

    /// Decode a sequence of u64 values.
    pub fn decode_sequence(values: &[u64]) -> Vec<i64> {
        values.iter().map(|&v| Self::decode(v)).collect()
    }

    /// Encode i64 to ZigZag then LEB128.
    pub fn encode_varint(v: i64) -> Vec<u8> {
        VarInt::encode(Self::encode(v))
    }

    /// Decode LEB128 then ZigZag i64.
    pub fn decode_varint(data: &[u8]) -> Result<(i64, usize)> {
        let (u, n) = VarInt::decode(data)?;
        Ok((Self::decode(u), n))
    }
}

// ─────────────────────────────── BitPacking ──────────────────────────────────

/// Bit-packing for compact storage of small unsigned integers.
///
/// Values must fit within the given `bit_width` (1–64 bits).
pub struct BitPacking;

impl BitPacking {
    /// Pack `values` into the minimum number of bytes using `bit_width` bits per value.
    ///
    /// Values are stored LSB-first, packed consecutively with no padding between values.
    pub fn pack(values: &[u64], bit_width: u8) -> Vec<u8> {
        if bit_width == 0 || values.is_empty() {
            return Vec::new();
        }
        let total_bits = values.len() * bit_width as usize;
        let n_bytes = (total_bits + 7) / 8;
        let mut buf = vec![0u8; n_bytes];
        let mut bit_pos = 0usize;

        for &v in values {
            let mask = if bit_width == 64 { u64::MAX } else { (1u64 << bit_width) - 1 };
            let v = v & mask;
            let mut remaining = bit_width as usize;
            let mut val = v;
            let mut bp = bit_pos;

            while remaining > 0 {
                let byte_idx = bp / 8;
                let bit_offset = bp % 8;
                let bits_in_byte = (8 - bit_offset).min(remaining);
                let bits_mask = ((1u64 << bits_in_byte) - 1) as u8;
                buf[byte_idx] |= ((val as u8) & bits_mask) << bit_offset;
                val >>= bits_in_byte;
                bp += bits_in_byte;
                remaining -= bits_in_byte;
            }
            bit_pos += bit_width as usize;
        }
        buf
    }

    /// Unpack `n` values of `bit_width` bits each from a bit-packed byte slice.
    pub fn unpack(data: &[u8], bit_width: u8, n: usize) -> Vec<u64> {
        if bit_width == 0 || n == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(n);
        let mut bit_pos = 0usize;

        for _ in 0..n {
            let mut v: u64 = 0;
            let mut bits_filled = 0usize;
            let mut remaining = bit_width as usize;
            let mut bp = bit_pos;

            while remaining > 0 {
                let byte_idx = bp / 8;
                let bit_offset = bp % 8;
                let bits_available = (8 - bit_offset).min(remaining);
                let bits_mask = ((1u64 << bits_available) - 1) as u8;
                let chunk = if byte_idx < data.len() {
                    (data[byte_idx] >> bit_offset) & bits_mask
                } else {
                    0
                };
                v |= (chunk as u64) << bits_filled;
                bits_filled += bits_available;
                bp += bits_available;
                remaining -= bits_available;
            }
            out.push(v);
            bit_pos += bit_width as usize;
        }
        out
    }

    /// Compute the minimum number of bits needed to represent `max_value`.
    pub fn bit_width_for(max_value: u64) -> u8 {
        if max_value == 0 {
            return 1;
        }
        (64 - max_value.leading_zeros()) as u8
    }
}

// ─────────────────────────────── DeltaEncoding ───────────────────────────────

/// First-differences (delta) encoding for correlated integer sequences.
///
/// Stores the first value verbatim, followed by the successive differences.
/// Particularly effective for sorted or slowly-changing sequences.
pub struct DeltaEncoding;

impl DeltaEncoding {
    /// Encode `values` as `(first_value, deltas)`.
    ///
    /// If `values` is empty, returns `(0, vec![])`.
    pub fn encode(values: &[i64]) -> (i64, Vec<i64>) {
        if values.is_empty() {
            return (0, Vec::new());
        }
        let first = values[0];
        let deltas = values.windows(2).map(|w| w[1] - w[0]).collect();
        (first, deltas)
    }

    /// Reconstruct the original sequence from `first` and `deltas`.
    pub fn decode(first: i64, deltas: &[i64]) -> Vec<i64> {
        let mut out = Vec::with_capacity(deltas.len() + 1);
        out.push(first);
        let mut current = first;
        for &d in deltas {
            current = current.saturating_add(d);
            out.push(current);
        }
        out
    }

    /// Encode and serialize to bytes:
    /// `[first: i64 LE] [n_deltas: u64 LE] [deltas: i64 LE...]`
    pub fn to_bytes(values: &[i64]) -> Vec<u8> {
        let (first, deltas) = Self::encode(values);
        let mut buf =
            Vec::with_capacity(8 + 8 + deltas.len() * 8);
        buf.extend_from_slice(&first.to_le_bytes());
        buf.extend_from_slice(&(deltas.len() as u64).to_le_bytes());
        for d in &deltas {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf
    }

    /// Deserialize from bytes written by [`DeltaEncoding::to_bytes`].
    pub fn from_bytes(data: &[u8]) -> Result<Vec<i64>> {
        if data.len() < 16 {
            return Err(IoError::FormatError(
                "DeltaEncoding: insufficient bytes".into(),
            ));
        }
        let first = i64::from_le_bytes(
            data[0..8]
                .try_into()
                .map_err(|_| IoError::FormatError("DeltaEncoding: bad first".into()))?,
        );
        let n_deltas = u64::from_le_bytes(
            data[8..16]
                .try_into()
                .map_err(|_| IoError::FormatError("DeltaEncoding: bad n_deltas".into()))?,
        ) as usize;
        if data.len() < 16 + n_deltas * 8 {
            return Err(IoError::FormatError(
                "DeltaEncoding: truncated deltas".into(),
            ));
        }
        let mut deltas = Vec::with_capacity(n_deltas);
        for i in 0..n_deltas {
            let off = 16 + i * 8;
            deltas.push(i64::from_le_bytes(
                data[off..off + 8]
                    .try_into()
                    .map_err(|_| IoError::FormatError("DeltaEncoding: bad delta".into()))?,
            ));
        }
        Ok(Self::decode(first, &deltas))
    }

    /// Double-delta encoding: apply delta encoding twice for monotone sequences.
    pub fn double_delta_encode(values: &[i64]) -> (i64, Vec<i64>) {
        let (first, deltas) = Self::encode(values);
        if deltas.is_empty() {
            return (first, Vec::new());
        }
        let (delta_first, delta_deltas) = Self::encode(&deltas);
        // Pack: [first, delta_first, delta_deltas...]
        let mut packed = vec![delta_first];
        packed.extend(delta_deltas);
        (first, packed)
    }
}

// ─────────────────────────────── RunLengthEncoding ───────────────────────────

/// Run-length encoding: compress repeated consecutive values.
pub struct RunLengthEncoding;

impl RunLengthEncoding {
    /// Encode a slice into `Vec<(value, count)>` runs.
    pub fn encode<T: Eq + Clone>(data: &[T]) -> Vec<(T, usize)> {
        if data.is_empty() {
            return Vec::new();
        }
        let mut runs: Vec<(T, usize)> = Vec::new();
        let mut current = data[0].clone();
        let mut count = 1usize;

        for item in &data[1..] {
            if *item == current {
                count += 1;
            } else {
                runs.push((current.clone(), count));
                current = item.clone();
                count = 1;
            }
        }
        runs.push((current, count));
        runs
    }

    /// Decode `Vec<(value, count)>` runs back to the original sequence.
    pub fn decode<T: Clone>(runs: &[(T, usize)]) -> Vec<T> {
        runs.iter()
            .flat_map(|(v, count)| std::iter::repeat(v.clone()).take(*count))
            .collect()
    }

    /// Compression ratio: `original_len / encoded_runs`.
    pub fn compression_ratio<T: Eq + Clone>(data: &[T]) -> f64 {
        if data.is_empty() {
            return 1.0;
        }
        let runs = Self::encode(data);
        data.len() as f64 / runs.len() as f64
    }

    /// Byte-level run-length encoding: `[(byte, count)]` with `count` as u8 (max 255).
    pub fn encode_bytes(data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut current = data[0];
        let mut count = 1u8;

        for &b in &data[1..] {
            if b == current && count < 255 {
                count += 1;
            } else {
                out.push(count);
                out.push(current);
                current = b;
                count = 1;
            }
        }
        out.push(count);
        out.push(current);
        out
    }

    /// Decode byte-level run-length encoding.
    pub fn decode_bytes(data: &[u8]) -> Result<Vec<u8>> {
        if data.len() % 2 != 0 {
            return Err(IoError::FormatError(
                "RunLengthEncoding: odd number of bytes".into(),
            ));
        }
        let mut out = Vec::new();
        for chunk in data.chunks_exact(2) {
            let count = chunk[0] as usize;
            let value = chunk[1];
            for _ in 0..count {
                out.push(value);
            }
        }
        Ok(out)
    }
}

// Arrow Flight protocol
pub mod arrow_flight;
// Kafka protocol
pub mod kafka;

// ─────────────────────────────── Tests ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── VarInt tests ──

    #[test]
    fn test_varint_zero() {
        let enc = VarInt::encode(0);
        assert_eq!(enc, vec![0x00]);
        let (v, n) = VarInt::decode(&enc).expect("decode");
        assert_eq!(v, 0);
        assert_eq!(n, 1);
    }

    #[test]
    fn test_varint_single_byte() {
        for i in 0..128u64 {
            let enc = VarInt::encode(i);
            assert_eq!(enc.len(), 1);
            let (v, n) = VarInt::decode(&enc).expect("decode");
            assert_eq!(v, i);
            assert_eq!(n, 1);
        }
    }

    #[test]
    fn test_varint_multi_byte() {
        let cases: &[u64] = &[128, 255, 256, 300, 65535, 100_000, u32::MAX as u64, u64::MAX];
        for &val in cases {
            let enc = VarInt::encode(val);
            let (decoded, consumed) = VarInt::decode(&enc).expect("decode");
            assert_eq!(decoded, val, "failed for {val}");
            assert_eq!(consumed, enc.len());
        }
    }

    #[test]
    fn test_varint_sequence() {
        let vals: Vec<u64> = vec![1, 2, 300, 65536];
        let enc = VarInt::encode_sequence(&vals);
        let (decoded, _) = VarInt::decode_sequence(&enc, 4).expect("decode");
        assert_eq!(decoded, vals);
    }

    #[test]
    fn test_varint_truncated() {
        let enc = vec![0x80u8]; // continuation bit set but no next byte
        assert!(VarInt::decode(&enc).is_err());
    }

    // ── ZigZag tests ──

    #[test]
    fn test_zigzag_basic() {
        assert_eq!(ZigZag::encode(0), 0);
        assert_eq!(ZigZag::encode(-1), 1);
        assert_eq!(ZigZag::encode(1), 2);
        assert_eq!(ZigZag::encode(-2), 3);
        assert_eq!(ZigZag::encode(2), 4);
        assert_eq!(ZigZag::encode(i64::MIN), u64::MAX);
        assert_eq!(ZigZag::encode(i64::MAX), u64::MAX - 1);
    }

    #[test]
    fn test_zigzag_roundtrip() {
        let cases: &[i64] = &[0, -1, 1, -100, 100, i64::MIN, i64::MAX, -32768, 32767];
        for &v in cases {
            assert_eq!(ZigZag::decode(ZigZag::encode(v)), v, "failed for {v}");
        }
    }

    #[test]
    fn test_zigzag_varint_roundtrip() {
        let cases: &[i64] = &[0, -1, 1, -127, 127, -128, 128];
        for &v in cases {
            let enc = ZigZag::encode_varint(v);
            let (decoded, _) = ZigZag::decode_varint(&enc).expect("decode");
            assert_eq!(decoded, v);
        }
    }

    // ── BitPacking tests ──

    #[test]
    fn test_bitpacking_2bit() {
        let values = vec![0u64, 1, 2, 3, 0, 1];
        let packed = BitPacking::pack(&values, 2);
        let unpacked = BitPacking::unpack(&packed, 2, values.len());
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_bitpacking_4bit() {
        let values: Vec<u64> = (0..16).collect();
        let packed = BitPacking::pack(&values, 4);
        let unpacked = BitPacking::unpack(&packed, 4, values.len());
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_bitpacking_8bit() {
        let values: Vec<u64> = (0..256).collect();
        let packed = BitPacking::pack(&values, 8);
        let unpacked = BitPacking::unpack(&packed, 8, values.len());
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_bitpacking_1bit() {
        let values = vec![1u64, 0, 1, 1, 0, 0, 1, 0, 1];
        let packed = BitPacking::pack(&values, 1);
        let unpacked = BitPacking::unpack(&packed, 1, values.len());
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_bitpacking_width_for() {
        assert_eq!(BitPacking::bit_width_for(0), 1);
        assert_eq!(BitPacking::bit_width_for(1), 1);
        assert_eq!(BitPacking::bit_width_for(2), 2);
        assert_eq!(BitPacking::bit_width_for(3), 2);
        assert_eq!(BitPacking::bit_width_for(4), 3);
        assert_eq!(BitPacking::bit_width_for(255), 8);
        assert_eq!(BitPacking::bit_width_for(256), 9);
    }

    // ── DeltaEncoding tests ──

    #[test]
    fn test_delta_roundtrip() {
        let data = vec![5i64, 10, 15, 20, 25];
        let (first, deltas) = DeltaEncoding::encode(&data);
        let decoded = DeltaEncoding::decode(first, &deltas);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_delta_single() {
        let data = vec![42i64];
        let (first, deltas) = DeltaEncoding::encode(&data);
        assert_eq!(first, 42);
        assert!(deltas.is_empty());
        assert_eq!(DeltaEncoding::decode(first, &deltas), data);
    }

    #[test]
    fn test_delta_bytes_roundtrip() {
        let data = vec![100i64, 200, 300, 250, 400];
        let bytes = DeltaEncoding::to_bytes(&data);
        let decoded = DeltaEncoding::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_delta_empty() {
        let (first, deltas) = DeltaEncoding::encode(&[]);
        assert_eq!(first, 0);
        assert!(deltas.is_empty());
    }

    // ── RunLengthEncoding tests ──

    #[test]
    fn test_rle_basic() {
        let data = vec![1i32, 1, 1, 2, 2, 3];
        let runs = RunLengthEncoding::encode(&data);
        assert_eq!(runs, vec![(1, 3), (2, 2), (3, 1)]);
        assert_eq!(RunLengthEncoding::decode(&runs), data);
    }

    #[test]
    fn test_rle_no_repetition() {
        let data = vec![1i32, 2, 3, 4];
        let runs = RunLengthEncoding::encode(&data);
        assert_eq!(runs.len(), 4);
        assert_eq!(RunLengthEncoding::decode(&runs), data);
    }

    #[test]
    fn test_rle_all_same() {
        let data = vec![7i32; 100];
        let runs = RunLengthEncoding::encode(&data);
        assert_eq!(runs, vec![(7, 100)]);
        assert_eq!(RunLengthEncoding::decode(&runs), data);
    }

    #[test]
    fn test_rle_empty() {
        let data: Vec<i32> = Vec::new();
        let runs = RunLengthEncoding::encode(&data);
        assert!(runs.is_empty());
        assert_eq!(RunLengthEncoding::decode(&runs), data);
    }

    #[test]
    fn test_rle_bytes_roundtrip() {
        let data = vec![0xaa_u8, 0xaa, 0xbb, 0xcc, 0xcc, 0xcc];
        let encoded = RunLengthEncoding::encode_bytes(&data);
        let decoded = RunLengthEncoding::decode_bytes(&encoded).expect("decode");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_rle_compression_ratio() {
        let data = vec![42u8; 1000];
        let ratio = RunLengthEncoding::compression_ratio(&data);
        assert!(ratio > 100.0);
    }

    #[test]
    fn test_rle_string_slices() {
        let data = vec!["a", "a", "b", "c", "c"];
        let runs = RunLengthEncoding::encode(&data);
        assert_eq!(runs, vec![("a", 2), ("b", 1), ("c", 2)]);
        let decoded = RunLengthEncoding::decode(&runs);
        assert_eq!(decoded, data);
    }
}
