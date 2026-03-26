//! Delta encoding with zigzag and variable-length integer (varint) support.
//!
//! This module provides three levels of integer compression:
//!
//! 1. **Delta encoding**: stores differences between consecutive values.
//! 2. **Zigzag encoding**: maps signed → unsigned for compact varint storage.
//! 3. **LEB128-style varint**: variable-length encoding of unsigned integers.
//!
//! The combination (`delta_plus_varint_encode`) achieves good compression for
//! sorted or slowly-varying integer sequences.

use crate::error::{IoError, Result as IoResult};

// ---------------------------------------------------------------------------
// Delta encoding
// ---------------------------------------------------------------------------

/// Encode a slice of `i64` values using delta encoding.
///
/// Returns `(base, deltas)` where:
/// - `base = data[0]` (the absolute first value)
/// - `deltas[i] = data[i+1] - data[i]`
///
/// An empty slice returns `(0, vec![])`.
pub fn delta_encode_i64(data: &[i64]) -> (i64, Vec<i64>) {
    if data.is_empty() {
        return (0, Vec::new());
    }
    let base = data[0];
    let deltas: Vec<i64> = data.windows(2).map(|w| w[1] - w[0]).collect();
    (base, deltas)
}

/// Reconstruct `i64` values from a base value and deltas.
///
/// `out[0] = base`, `out[i] = out[i-1] + deltas[i-1]`.
pub fn delta_decode_i64(base: i64, deltas: &[i64]) -> Vec<i64> {
    let mut out = Vec::with_capacity(deltas.len() + 1);
    out.push(base);
    let mut prev = base;
    for &d in deltas {
        let next = prev.wrapping_add(d);
        out.push(next);
        prev = next;
    }
    out
}

/// Encode a slice of `f64` values using delta encoding.
///
/// Returns `(base, deltas)` where `deltas[i] = data[i+1] - data[i]`.
pub fn delta_encode_f64(data: &[f64]) -> (f64, Vec<f64>) {
    if data.is_empty() {
        return (0.0, Vec::new());
    }
    let base = data[0];
    let deltas: Vec<f64> = data.windows(2).map(|w| w[1] - w[0]).collect();
    (base, deltas)
}

/// Reconstruct `f64` values from a base value and deltas.
pub fn delta_decode_f64(base: f64, deltas: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(deltas.len() + 1);
    out.push(base);
    let mut prev = base;
    for &d in deltas {
        let next = prev + d;
        out.push(next);
        prev = next;
    }
    out
}

// ---------------------------------------------------------------------------
// Zigzag encoding
// ---------------------------------------------------------------------------

/// Map a signed `i64` to an unsigned `u64` using zigzag encoding.
///
/// ```text
/// zigzag_encode(0)  = 0
/// zigzag_encode(-1) = 1
/// zigzag_encode(1)  = 2
/// zigzag_encode(-2) = 3
/// ```
///
/// This ensures that small-magnitude signed integers produce small unsigned
/// values, enabling compact varint encoding.
#[inline]
pub fn zigzag_encode(n: i64) -> u64 {
    ((n << 1) ^ (n >> 63)) as u64
}

/// Decode a zigzag-encoded `u64` back to `i64`.
#[inline]
pub fn zigzag_decode(n: u64) -> i64 {
    ((n >> 1) as i64) ^ (-((n & 1) as i64))
}

// ---------------------------------------------------------------------------
// Variable-length integer encoding (LEB128)
// ---------------------------------------------------------------------------

/// Encode an unsigned 64-bit integer as LEB128 variable-length bytes.
///
/// Each byte uses 7 bits of the value and 1 continuation bit (MSB).
/// Small values encode in 1 byte; values up to 2^63 need at most 9 bytes.
pub fn varint_encode(mut value: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(10);
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte); // No continuation bit
            break;
        } else {
            buf.push(byte | 0x80); // Set continuation bit
        }
    }
    buf
}

/// Decode a LEB128 varint from a byte slice.
///
/// Returns `(decoded_value, bytes_consumed)`.
/// Returns an error if the slice is empty or if the encoding is malformed
/// (overflows 64 bits or terminates unexpectedly).
pub fn varint_decode(bytes: &[u8]) -> IoResult<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;

    for (i, &byte) in bytes.iter().enumerate() {
        let low7 = (byte & 0x7F) as u64;

        if shift >= 63 && low7 > 1 {
            return Err(IoError::ParseError(
                "varint overflow: value does not fit in u64".to_string(),
            ));
        }

        result |= low7 << shift;

        if byte & 0x80 == 0 {
            // No continuation bit: done
            return Ok((result, i + 1));
        }

        shift += 7;
        if shift > 63 {
            return Err(IoError::ParseError(
                "varint overflow: too many continuation bytes".to_string(),
            ));
        }
    }

    Err(IoError::ParseError(
        "varint decode: unexpected end of byte slice".to_string(),
    ))
}

// ---------------------------------------------------------------------------
// Combined pipeline: delta + zigzag + varint
// ---------------------------------------------------------------------------

/// Compress an `i64` slice using the full pipeline:
/// `delta → zigzag → varint`.
///
/// Wire format:
/// ```text
/// [varint(zigzag(base)), varint(zigzag(delta_0)), varint(zigzag(delta_1)), ...]
/// ```
pub fn delta_plus_varint_encode(data: &[i64]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    let (base, deltas) = delta_encode_i64(data);

    let mut out = Vec::new();
    // Encode base
    let zz_base = zigzag_encode(base);
    out.extend_from_slice(&varint_encode(zz_base));

    // Encode each delta
    for d in &deltas {
        let zz = zigzag_encode(*d);
        out.extend_from_slice(&varint_encode(zz));
    }
    out
}

/// Decompress bytes produced by [`delta_plus_varint_encode`] back to `Vec<i64>`.
pub fn delta_plus_varint_decode(bytes: &[u8]) -> IoResult<Vec<i64>> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }

    let mut pos = 0;
    let mut values: Vec<i64> = Vec::new();

    // Decode base
    let (zz_base, consumed) = varint_decode(&bytes[pos..])?;
    pos += consumed;
    let base = zigzag_decode(zz_base);
    values.push(base);

    let mut prev = base;

    // Decode deltas
    while pos < bytes.len() {
        let (zz, consumed) = varint_decode(&bytes[pos..])?;
        pos += consumed;
        let delta = zigzag_decode(zz);
        let next = prev.wrapping_add(delta);
        values.push(next);
        prev = next;
    }

    Ok(values)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_encode_monotone_sequence() {
        let data: Vec<i64> = (0..10).collect();
        let (base, deltas) = delta_encode_i64(&data);
        assert_eq!(base, 0);
        assert_eq!(deltas, vec![1i64; 9]);
    }

    #[test]
    fn test_delta_decode_roundtrip_i64() {
        let original = vec![100i64, 105, 103, 110, 108, 200];
        let (base, deltas) = delta_encode_i64(&original);
        let decoded = delta_decode_i64(base, &deltas);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_varint_encode_small() {
        // Values 0..127 must encode in exactly 1 byte
        for v in 0u64..=127 {
            let encoded = varint_encode(v);
            assert_eq!(encoded.len(), 1, "value {v} should encode in 1 byte");
            let (decoded, n) = varint_decode(&encoded).expect("decode failed");
            assert_eq!(n, 1);
            assert_eq!(decoded, v);
        }
    }

    #[test]
    fn test_varint_encode_large() {
        let cases: &[u64] = &[128, 255, 16383, 16384, u32::MAX as u64, u64::MAX / 2];
        for &v in cases {
            let encoded = varint_encode(v);
            let (decoded, _) = varint_decode(&encoded).expect("decode failed");
            assert_eq!(decoded, v, "roundtrip failed for {v}");
        }
    }

    #[test]
    fn test_varint_roundtrip_random() {
        // Deterministic pseudo-random values using a simple LCG
        let mut state: u64 = 0x123456789ABCDEF0;
        for _ in 0..200 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = state >> 1; // ensure < 2^63
            let encoded = varint_encode(v);
            let (decoded, _) = varint_decode(&encoded).expect("decode failed");
            assert_eq!(decoded, v);
        }
    }

    #[test]
    fn test_delta_varint_roundtrip() {
        let original: Vec<i64> = vec![-1000, -500, 0, 1, 1, 2, 100, 100, 100, 200];
        let encoded = delta_plus_varint_encode(&original);
        let decoded = delta_plus_varint_decode(&encoded).expect("decode failed");
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_delta_varint_empty() {
        let encoded = delta_plus_varint_encode(&[]);
        assert!(encoded.is_empty());
        let decoded = delta_plus_varint_decode(&[]).expect("decode failed");
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_zigzag_properties() {
        assert_eq!(zigzag_encode(0), 0);
        assert_eq!(zigzag_encode(-1), 1);
        assert_eq!(zigzag_encode(1), 2);
        assert_eq!(zigzag_encode(-2), 3);
        assert_eq!(zigzag_encode(i64::MAX), u64::MAX - 1);
        assert_eq!(zigzag_encode(i64::MIN), u64::MAX);

        for n in [-1000i64, -1, 0, 1, 1000, i64::MIN, i64::MAX] {
            assert_eq!(zigzag_decode(zigzag_encode(n)), n);
        }
    }

    #[test]
    fn test_delta_f64_roundtrip() {
        let data = vec![1.0f64, 1.1, 1.2, 1.3, 1.4];
        let (base, deltas) = delta_encode_f64(&data);
        let decoded = delta_decode_f64(base, &deltas);
        for (a, b) in data.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
