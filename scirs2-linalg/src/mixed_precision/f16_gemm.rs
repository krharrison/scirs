//! F16/BF16 GEMM with f32 accumulation
//!
//! This module provides tiled matrix-matrix multiplication (GEMM) using
//! 16-bit floating-point storage (either IEEE 754 half-precision or bfloat16)
//! while accumulating in full f32. This mirrors hardware tensor-core semantics
//! and is useful for throughput-sensitive workloads where storage bandwidth is
//! the bottleneck but some numeric precision must be preserved.
//!
//! # Half-precision formats supported
//!
//! * **F16** (IEEE 754-2008 binary16): 1 sign, 5 exponent, 10 mantissa bits.
//!   Range roughly ±65504, precision ~3 decimal digits.
//! * **BF16** (Brain Float16): 1 sign, 8 exponent, 7 mantissa bits.
//!   Same exponent range as f32 but much lower mantissa precision.
//!
//! # Design
//!
//! * Pure Rust — no `half` crate, no C/Fortran dependencies.
//! * Tile size 32×32 for L1-cache friendliness.
//! * All accumulation in f32 to avoid intermediate overflow/underflow.

use scirs2_core::ndarray::Array2;

use crate::error::{LinalgError, LinalgResult};

// ─────────────────────────────────────────────────────────────────────────────
// IEEE 754 half-precision (F16) type
// ─────────────────────────────────────────────────────────────────────────────

/// A 16-bit IEEE 754 half-precision floating-point value stored as a `u16`.
///
/// This type is **not** a general-purpose arithmetic type; it is intended for
/// *storage* only. Arithmetic is always performed in f32.
///
/// # Bit layout (IEEE 754-2008 binary16)
/// ```text
/// bit 15    : sign
/// bits 14-10: biased exponent (bias = 15)
/// bits  9-0 : mantissa (implicit leading 1 for normal numbers)
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F16(pub u16);

impl F16 {
    /// Not-a-Number sentinel (quiet NaN).
    pub const NAN: F16 = F16(0x7E00);
    /// Positive infinity.
    pub const INFINITY: F16 = F16(0x7C00);
    /// Negative infinity.
    pub const NEG_INFINITY: F16 = F16(0xFC00);
    /// Positive zero.
    pub const ZERO: F16 = F16(0x0000);

    /// Convert an `f32` value to `F16` (round-to-nearest, ties-to-even).
    ///
    /// Special values (NaN, ±Inf) are preserved.  Values outside the f16 range
    /// saturate to ±infinity.
    #[inline]
    pub fn from_f32(v: f32) -> F16 {
        // Reinterpret f32 bits
        let bits: u32 = v.to_bits();
        let sign: u16 = ((bits >> 16) & 0x8000) as u16;
        let exp: i32 = ((bits >> 23) & 0xFF) as i32 - 127; // unbiased
        let mant: u32 = bits & 0x007F_FFFF;

        // Handle special values
        if exp == 128 {
            // NaN or Inf
            if mant != 0 {
                // NaN — preserve quiet bit, zero lower bits
                return F16(sign | 0x7E00 | ((mant >> 13) as u16 & 0x01FF));
            } else {
                return F16(sign | 0x7C00); // Inf
            }
        }

        if exp >= 16 {
            // Overflow → infinity
            return F16(sign | 0x7C00);
        }

        if exp < -24 {
            // Too small even for subnormal → zero
            return F16(sign);
        }

        if exp < -14 {
            // Subnormal f16: exponent is -14, so we shift mantissa right
            let shift = (-14 - exp) as u32;
            // Add the implicit leading 1 into the mantissa
            let full_mant = (mant | 0x0080_0000) >> (13 + shift);
            // Round: check the bit just below
            let round_bit = (mant | 0x0080_0000) >> (12 + shift) & 1;
            F16(sign | (full_mant as u16 + round_bit as u16))
        } else {
            // Normal f16
            let h_exp = ((exp + 15) as u16) << 10;
            let h_mant = (mant >> 13) as u16;
            // Round-to-nearest (guard bit is bit 12 of the f32 mantissa)
            let guard = (mant >> 12) & 1;
            // Sticky bit: any bits below guard
            let sticky = mant & 0x0FFF;
            let round = if guard == 1 && (sticky != 0 || (h_mant & 1) != 0) {
                1u16
            } else {
                0u16
            };
            let raw = sign | h_exp | h_mant;
            // Check for rounding overflow into exponent
            let rounded = raw + round;
            F16(rounded)
        }
    }

    /// Convert an `F16` value to `f32`.
    #[inline]
    pub fn to_f32(self) -> f32 {
        let bits = self.0 as u32;
        let sign: u32 = (bits & 0x8000) << 16;
        let exp: u32 = (bits >> 10) & 0x1F;
        let mant: u32 = bits & 0x03FF;

        let result_bits = if exp == 0 {
            if mant == 0 {
                // ±0
                sign
            } else {
                // Subnormal f16 → normal f32
                // Find leading 1 bit
                let leading = mant.leading_zeros() - 22; // 32 - 10 = 22 prefix bits
                let new_mant = (mant << (leading + 1)) & 0x007F_FFFF;
                let new_exp = 127 - 14 - leading; // f32 bias - f16 subnormal exp - leading zeros
                sign | (new_exp << 23) | new_mant
            }
        } else if exp == 31 {
            // NaN or Inf
            if mant == 0 {
                sign | 0x7F80_0000 // ±Inf
            } else {
                sign | 0x7FC0_0000 | (mant << 13) // NaN
            }
        } else {
            // Normal
            let new_exp = exp + 127 - 15; // adjust bias
            sign | (new_exp << 23) | (mant << 13)
        };

        f32::from_bits(result_bits)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BF16 helpers (operates directly on f32 bit representations)
// ─────────────────────────────────────────────────────────────────────────────

/// Truncate an `f32` to bfloat16 precision (keeps upper 16 bits).
///
/// BF16 uses the same 8-bit exponent as f32 so no exponent clamping is needed.
/// The lower 16 mantissa bits are simply discarded (truncation towards zero).
///
/// Special values (NaN, ±Inf) are preserved exactly.
#[inline]
pub fn f32_to_bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    // Round-to-nearest-even: look at bit 15 (guard) and lower bits (sticky)
    let guard = (bits >> 15) & 1;
    let sticky = bits & 0x7FFF;
    let lsb = (bits >> 16) & 1;
    let round = if guard == 1 && (sticky != 0 || lsb != 0) {
        1u32
    } else {
        0u32
    };
    ((bits + (round << 16)) >> 16) as u16
}

/// Reconstruct an `f32` from bfloat16 bits (upper 16 bits of f32).
#[inline]
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tiled GEMM infrastructure
// ─────────────────────────────────────────────────────────────────────────────

/// Default tile size (in elements per dimension) used by the tiled GEMM kernels.
///
/// 32×32 fits comfortably in a typical 32 KiB L1 data cache when using f32
/// accumulators alongside one tile from each operand.
pub const DEFAULT_TILE_SIZE: usize = 32;

/// Configuration for [`MixedPrecisionGemm`].
#[derive(Debug, Clone)]
pub struct GemmConfig {
    /// Number of rows/columns per tile.  Must be ≥ 1.
    pub tile_size: usize,
}

impl Default for GemmConfig {
    fn default() -> Self {
        GemmConfig {
            tile_size: DEFAULT_TILE_SIZE,
        }
    }
}

/// Tiled mixed-precision GEMM kernel.
///
/// Instantiate with [`MixedPrecisionGemm::new`] or
/// [`MixedPrecisionGemm::with_config`], then call [`gemm_f16_f32`] or
/// [`gemm_bf16_f32`] on the instance.
///
/// [`gemm_f16_f32`]: MixedPrecisionGemm::gemm_f16_f32
/// [`gemm_bf16_f32`]: MixedPrecisionGemm::gemm_bf16_f32
#[derive(Debug, Clone)]
pub struct MixedPrecisionGemm {
    config: GemmConfig,
}

impl MixedPrecisionGemm {
    /// Create a new instance using the default configuration.
    pub fn new() -> Self {
        MixedPrecisionGemm {
            config: GemmConfig::default(),
        }
    }

    /// Create a new instance with a custom [`GemmConfig`].
    pub fn with_config(config: GemmConfig) -> Self {
        MixedPrecisionGemm { config }
    }

    /// Return the tile size currently in use.
    pub fn tile_size(&self) -> usize {
        self.config.tile_size
    }

    /// Perform `C = A × B` where `A` and `B` are stored as `F16` arrays and
    /// the accumulator is `f32`.
    ///
    /// # Errors
    /// Returns [`LinalgError::ShapeError`] when the inner dimensions of `a`
    /// and `b` do not match.
    pub fn gemm_f16_f32(&self, a: &Array2<F16>, b: &Array2<F16>) -> LinalgResult<Array2<f32>> {
        gemm_f16_f32_tiled(a, b, self.config.tile_size)
    }

    /// Perform `C = A × B` where inputs are `f32` but are temporarily
    /// downcast to BF16 storage before each multiply-accumulate, accumulating
    /// the products back in `f32`.
    ///
    /// # Errors
    /// Returns [`LinalgError::ShapeError`] when the inner dimensions of `a`
    /// and `b` do not match.
    pub fn gemm_bf16_f32(&self, a: &Array2<f32>, b: &Array2<f32>) -> LinalgResult<Array2<f32>> {
        gemm_bf16_f32_tiled(a, b, self.config.tile_size)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Free-function API (tile_size = DEFAULT_TILE_SIZE)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute `C = A × B` where `A` and `B` are stored as `F16` and the
/// accumulator is `f32`.
///
/// Uses a tiled algorithm with a tile size of [`DEFAULT_TILE_SIZE`] for
/// cache efficiency.
///
/// # Errors
/// Returns [`LinalgError::ShapeError`] if the inner dimensions do not match.
///
/// # Examples
/// ```
/// use scirs2_linalg::mixed_precision::f16_gemm::{F16, gemm_f16_f32};
/// use scirs2_core::ndarray::Array2;
///
/// // Build 2×2 matrices in F16
/// let a = Array2::from_shape_fn((2, 2), |(i, j)| {
///     F16::from_f32((i * 2 + j + 1) as f32)
/// });
/// let b = Array2::from_shape_fn((2, 2), |(i, j)| {
///     F16::from_f32((i * 2 + j + 5) as f32)
/// });
/// let c = gemm_f16_f32(&a, &b).expect("GEMM failed");
/// // [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
/// assert!((c[[0, 0]] - 19.0).abs() < 0.5);
/// assert!((c[[1, 1]] - 50.0).abs() < 0.5);
/// ```
pub fn gemm_f16_f32(a: &Array2<F16>, b: &Array2<F16>) -> LinalgResult<Array2<f32>> {
    gemm_f16_f32_tiled(a, b, DEFAULT_TILE_SIZE)
}

/// Compute `C = A × B` where the inputs are `f32` but are downcast to BF16
/// storage for the multiply step; accumulation is always in `f32`.
///
/// Uses a tiled algorithm with a tile size of [`DEFAULT_TILE_SIZE`].
///
/// # Errors
/// Returns [`LinalgError::ShapeError`] if the inner dimensions do not match.
///
/// # Examples
/// ```
/// use scirs2_linalg::mixed_precision::f16_gemm::gemm_bf16_f32;
/// use scirs2_core::ndarray::array;
///
/// let a = array![[1.0f32, 2.0], [3.0, 4.0]];
/// let b = array![[5.0f32, 6.0], [7.0, 8.0]];
/// let c = gemm_bf16_f32(&a, &b).expect("GEMM failed");
/// assert!((c[[0, 0]] - 19.0).abs() < 0.1);
/// assert!((c[[1, 1]] - 50.0).abs() < 0.1);
/// ```
pub fn gemm_bf16_f32(a: &Array2<f32>, b: &Array2<f32>) -> LinalgResult<Array2<f32>> {
    gemm_bf16_f32_tiled(a, b, DEFAULT_TILE_SIZE)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal tiled kernels
// ─────────────────────────────────────────────────────────────────────────────

/// Internal: tiled F16 → f32 GEMM.
fn gemm_f16_f32_tiled(a: &Array2<F16>, b: &Array2<F16>, tile: usize) -> LinalgResult<Array2<f32>> {
    let (m, k) = (a.nrows(), a.ncols());
    let (kb, n) = (b.nrows(), b.ncols());

    if k != kb {
        return Err(LinalgError::ShapeError(format!(
            "gemm_f16_f32: inner dimensions mismatch ({k} vs {kb})"
        )));
    }
    if tile == 0 {
        return Err(LinalgError::ValueError(
            "tile_size must be at least 1".to_string(),
        ));
    }

    let mut c = Array2::<f32>::zeros((m, n));

    // Iterate over tiles in the K (reduction) dimension first so that we can
    // keep the A-tile and B-tile in cache simultaneously.
    let mut kt = 0;
    while kt < k {
        let kt_end = (kt + tile).min(k);

        let mut it = 0;
        while it < m {
            let it_end = (it + tile).min(m);

            let mut jt = 0;
            while jt < n {
                let jt_end = (jt + tile).min(n);

                // Inner micro-kernel: accumulate into c[it..it_end, jt..jt_end]
                for i in it..it_end {
                    for j in jt..jt_end {
                        let mut acc: f32 = 0.0;
                        for kk in kt..kt_end {
                            let av = a[[i, kk]].to_f32();
                            let bv = b[[kk, j]].to_f32();
                            acc += av * bv;
                        }
                        c[[i, j]] += acc;
                    }
                }

                jt += tile;
            }
            it += tile;
        }
        kt += tile;
    }

    Ok(c)
}

/// Internal: tiled BF16 (from f32) → f32 GEMM.
fn gemm_bf16_f32_tiled(a: &Array2<f32>, b: &Array2<f32>, tile: usize) -> LinalgResult<Array2<f32>> {
    let (m, k) = (a.nrows(), a.ncols());
    let (kb, n) = (b.nrows(), b.ncols());

    if k != kb {
        return Err(LinalgError::ShapeError(format!(
            "gemm_bf16_f32: inner dimensions mismatch ({k} vs {kb})"
        )));
    }
    if tile == 0 {
        return Err(LinalgError::ValueError(
            "tile_size must be at least 1".to_string(),
        ));
    }

    let mut c = Array2::<f32>::zeros((m, n));

    let mut kt = 0;
    while kt < k {
        let kt_end = (kt + tile).min(k);

        let mut it = 0;
        while it < m {
            let it_end = (it + tile).min(m);

            let mut jt = 0;
            while jt < n {
                let jt_end = (jt + tile).min(n);

                for i in it..it_end {
                    for j in jt..jt_end {
                        let mut acc: f32 = 0.0;
                        for kk in kt..kt_end {
                            // Downcast operands to bf16 then back to f32
                            let av = bf16_bits_to_f32(f32_to_bf16_bits(a[[i, kk]]));
                            let bv = bf16_bits_to_f32(f32_to_bf16_bits(b[[kk, j]]));
                            acc += av * bv;
                        }
                        c[[i, j]] += acc;
                    }
                }

                jt += tile;
            }
            it += tile;
        }
        kt += tile;
    }

    Ok(c)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    // ── F16 conversion round-trips ──────────────────────────────────────────

    #[test]
    fn test_f16_roundtrip_small_integers() {
        for v in [0.0f32, 1.0, -1.0, 2.0, 4.0, 8.0, 100.0, -100.0, 2048.0] {
            let h = F16::from_f32(v);
            let back = h.to_f32();
            let rel_err = if v == 0.0 {
                back.abs()
            } else {
                (back - v).abs() / v.abs()
            };
            assert!(
                rel_err < 0.001,
                "F16 round-trip failed for {v}: got {back}, rel_err={rel_err}"
            );
        }
    }

    #[test]
    fn test_f16_roundtrip_range() {
        // Values in [-2048, 2048] must roundtrip within 0.1%
        for i in -20i32..=20 {
            let v = i as f32 * 100.0;
            let h = F16::from_f32(v);
            let back = h.to_f32();
            let rel_err = if v == 0.0 {
                back.abs()
            } else {
                (back - v).abs() / v.abs()
            };
            assert!(
                rel_err < 0.001,
                "F16 round-trip outside 0.1% for {v}: got {back}"
            );
        }
    }

    #[test]
    fn test_f16_special_values() {
        // Zero
        assert_eq!(F16::from_f32(0.0_f32).to_f32(), 0.0_f32);
        // Positive infinity
        let inf = F16::from_f32(f32::INFINITY);
        assert!(inf.to_f32().is_infinite() && inf.to_f32() > 0.0);
        // Negative infinity
        let neg_inf = F16::from_f32(f32::NEG_INFINITY);
        assert!(neg_inf.to_f32().is_infinite() && neg_inf.to_f32() < 0.0);
        // NaN
        assert!(F16::from_f32(f32::NAN).to_f32().is_nan());
    }

    #[test]
    fn test_f16_saturation_to_infinity() {
        // Overflow: 1e10 is well outside f16 range (~65504)
        let overflow = F16::from_f32(1.0e10_f32);
        assert!(overflow.to_f32().is_infinite());
    }

    // ── BF16 conversion ─────────────────────────────────────────────────────

    #[test]
    fn test_bf16_preserves_exponent() {
        // BF16 has the same exponent as f32, so any power-of-two is preserved.
        for exp in [0.0625f32, 0.25, 1.0, 4.0, 256.0, 1024.0] {
            let bf = bf16_bits_to_f32(f32_to_bf16_bits(exp));
            // Exponent preserved → ratio must be exactly 1 (mantissa may differ)
            let ratio = bf / exp;
            assert!(
                (ratio - 1.0).abs() < 0.01,
                "BF16 exponent not preserved for {exp}: ratio={ratio}"
            );
        }
    }

    #[test]
    fn test_bf16_roundtrip_accuracy() {
        // BF16 has 7 mantissa bits → relative error ≤ 2^-7 ≈ 0.78%
        for v in [1.0f32, 1.5, 2.0, -3.0, 100.0, -1024.0] {
            let back = bf16_bits_to_f32(f32_to_bf16_bits(v));
            let rel = (back - v).abs() / v.abs();
            assert!(rel < 0.01, "BF16 rel error too large for {v}: {rel}");
        }
    }

    // ── gemm_f16_f32 ────────────────────────────────────────────────────────

    #[test]
    fn test_gemm_f16_identity() {
        let n = 4;
        let identity = Array2::from_shape_fn((n, n), |(i, j)| {
            F16::from_f32(if i == j { 1.0 } else { 0.0 })
        });
        let a = Array2::from_shape_fn((n, n), |(i, j)| F16::from_f32((i * n + j + 1) as f32));
        let c = gemm_f16_f32(&a, &identity).expect("GEMM failed");
        for i in 0..n {
            for j in 0..n {
                let expected = a[[i, j]].to_f32();
                let got = c[[i, j]];
                let err = (got - expected).abs();
                assert!(
                    err < 1.0,
                    "Identity GEMM wrong at [{i},{j}]: expected={expected}, got={got}"
                );
            }
        }
    }

    #[test]
    fn test_gemm_f16_vs_f64_reference() {
        // [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
        let a_f16 = Array2::from_shape_fn((2, 2), |(i, j)| F16::from_f32((i * 2 + j + 1) as f32));
        let b_f16 = Array2::from_shape_fn((2, 2), |(i, j)| F16::from_f32((i * 2 + j + 5) as f32));
        let c = gemm_f16_f32(&a_f16, &b_f16).expect("GEMM failed");

        let expected = [[19.0f32, 22.0], [43.0, 50.0]];
        for i in 0..2 {
            for j in 0..2 {
                let rel = (c[[i, j]] - expected[i][j]).abs() / expected[i][j].abs();
                assert!(
                    rel < 0.005,
                    "F16 GEMM error at [{i},{j}]: expected={}, got={}, rel={rel}",
                    expected[i][j],
                    c[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_gemm_f16_larger_matrix() {
        // 8×8 matrices — verify relative error < 0.5% vs f64 reference
        let n = 8usize;
        let a_f64: Array2<f64> =
            Array2::from_shape_fn((n, n), |(i, j)| (((i * n + j) % 7) as f64) + 1.0);
        let b_f64: Array2<f64> =
            Array2::from_shape_fn((n, n), |(i, j)| (((i + j * 3) % 5) as f64) + 1.0);
        let c_ref: Array2<f64> = a_f64.dot(&b_f64);

        let a_f16 = Array2::from_shape_fn((n, n), |(i, j)| F16::from_f32(a_f64[[i, j]] as f32));
        let b_f16 = Array2::from_shape_fn((n, n), |(i, j)| F16::from_f32(b_f64[[i, j]] as f32));
        let c = gemm_f16_f32(&a_f16, &b_f16).expect("GEMM failed");

        let mut max_rel = 0.0f32;
        for i in 0..n {
            for j in 0..n {
                let got = c[[i, j]] as f64;
                let exp = c_ref[[i, j]];
                let rel = ((got - exp).abs() / exp.abs()) as f32;
                if rel > max_rel {
                    max_rel = rel;
                }
            }
        }
        assert!(
            max_rel < 0.005,
            "F16 GEMM max_rel_error={max_rel} exceeds 0.5%"
        );
    }

    #[test]
    fn test_gemm_f16_tile_size_1_vs_32() {
        // Tile size should not affect result
        let n = 6usize;
        let a = Array2::from_shape_fn((n, n), |(i, j)| F16::from_f32((i + j + 1) as f32));
        let b = Array2::from_shape_fn((n, n), |(i, j)| F16::from_f32((i * 2 + j + 1) as f32));
        let c1 = gemm_f16_f32_tiled(&a, &b, 1).expect("tile=1 failed");
        let c32 = gemm_f16_f32_tiled(&a, &b, 32).expect("tile=32 failed");
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (c1[[i, j]] - c32[[i, j]]).abs() < 1e-4,
                    "Tile-size mismatch at [{i},{j}]: tile1={} tile32={}",
                    c1[[i, j]],
                    c32[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_gemm_f16_dimension_mismatch() {
        let a = Array2::from_shape_fn((2, 3), |(i, j)| F16::from_f32((i + j) as f32));
        let b = Array2::from_shape_fn((4, 2), |(i, j)| F16::from_f32((i + j) as f32));
        assert!(gemm_f16_f32(&a, &b).is_err());
    }

    // ── gemm_bf16_f32 ───────────────────────────────────────────────────────

    #[test]
    fn test_gemm_bf16_vs_f64_reference() {
        // BF16 has more mantissa precision than F16, so error budget is 0.1%
        let a = array![[1.0f32, 2.0], [3.0, 4.0]];
        let b = array![[5.0f32, 6.0], [7.0, 8.0]];
        let c = gemm_bf16_f32(&a, &b).expect("BF16 GEMM failed");
        let expected = [[19.0f32, 22.0], [43.0, 50.0]];
        for i in 0..2 {
            for j in 0..2 {
                let rel = (c[[i, j]] - expected[i][j]).abs() / expected[i][j].abs();
                assert!(
                    rel < 0.001,
                    "BF16 GEMM error at [{i},{j}]: expected={}, got={}, rel={rel}",
                    expected[i][j],
                    c[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_gemm_bf16_larger_matrix() {
        let n = 8usize;
        let a_f64: Array2<f64> =
            Array2::from_shape_fn((n, n), |(i, j)| (((i * n + j) % 7) as f64) + 1.0);
        let b_f64: Array2<f64> =
            Array2::from_shape_fn((n, n), |(i, j)| (((i + j * 3) % 5) as f64) + 1.0);
        let c_ref: Array2<f64> = a_f64.dot(&b_f64);

        let a_f32 = Array2::from_shape_fn((n, n), |(i, j)| a_f64[[i, j]] as f32);
        let b_f32 = Array2::from_shape_fn((n, n), |(i, j)| b_f64[[i, j]] as f32);
        let c = gemm_bf16_f32(&a_f32, &b_f32).expect("BF16 GEMM failed");

        let mut max_rel = 0.0f32;
        for i in 0..n {
            for j in 0..n {
                let got = c[[i, j]] as f64;
                let exp = c_ref[[i, j]];
                let rel = ((got - exp).abs() / exp.abs()) as f32;
                if rel > max_rel {
                    max_rel = rel;
                }
            }
        }
        assert!(
            max_rel < 0.001,
            "BF16 GEMM max_rel_error={max_rel} exceeds 0.1%"
        );
    }

    #[test]
    fn test_gemm_bf16_tile_size_1_vs_32() {
        let n = 5usize;
        let a = Array2::from_shape_fn((n, n), |(i, j)| (i + j + 1) as f32);
        let b = Array2::from_shape_fn((n, n), |(i, j)| (i * 2 + j + 1) as f32);
        let c1 = gemm_bf16_f32_tiled(&a, &b, 1).expect("tile=1 failed");
        let c32 = gemm_bf16_f32_tiled(&a, &b, 32).expect("tile=32 failed");
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (c1[[i, j]] - c32[[i, j]]).abs() < 1e-4,
                    "BF16 tile mismatch [{i},{j}]: {}, {}",
                    c1[[i, j]],
                    c32[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_gemm_bf16_dimension_mismatch() {
        let a = Array2::from_shape_fn((2, 3), |(i, j)| (i + j) as f32);
        let b = Array2::from_shape_fn((4, 2), |(i, j)| (i + j) as f32);
        assert!(gemm_bf16_f32(&a, &b).is_err());
    }

    // ── MixedPrecisionGemm struct ────────────────────────────────────────────

    #[test]
    fn test_mixed_precision_gemm_struct_default() {
        let gemm = MixedPrecisionGemm::new();
        assert_eq!(gemm.tile_size(), DEFAULT_TILE_SIZE);
    }

    #[test]
    fn test_mixed_precision_gemm_struct_custom_tile() {
        let gemm = MixedPrecisionGemm::with_config(GemmConfig { tile_size: 16 });
        assert_eq!(gemm.tile_size(), 16);
        let a = Array2::from_shape_fn((4, 4), |(i, j)| F16::from_f32((i + j + 1) as f32));
        let b = Array2::from_shape_fn((4, 4), |(i, j)| F16::from_f32((i + j + 1) as f32));
        let result = gemm.gemm_f16_f32(&a, &b);
        assert!(result.is_ok());
    }
}
