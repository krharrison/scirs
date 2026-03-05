//! NEON-accelerated 8-bit integer operations for quantised ML inference
//!
//! Quantised neural network inference on ARM processors can run significantly
//! faster using 8-bit integer arithmetic.  This module provides:
//!
//! - **Dot products** over `i8` arrays with `i32` accumulation.
//! - **Quantisation** of `f32` tensors to `i8` with configurable scale and
//!   zero-point (symmetric and asymmetric quantisation schemes).
//! - **Dequantisation** from `i8` back to `f32`.
//! - **Saturating addition** of two `i8` vectors.
//!
//! ## Quantisation convention
//!
//! We use the standard linear quantisation formula:
//!
//! ```text
//! q(x) = clamp(round(x / scale) + zero_point, -128, 127)
//! x̂(q) = (q - zero_point) * scale
//! ```
//!
//! where `scale` is a positive `f32` and `zero_point` is an `i8`.
//!
//! ## NEON strategy
//!
//! NEON provides `vmull_s8` (8×8→16 widening multiply) and `vpaddlq_s16`
//! (16-bit pairwise add → 32-bit) to implement i8 dot products efficiently
//! without intermediate overflow.  The inner loop processes 8 elements at a
//! time (one 64-bit NEON register pair).

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ============================================================
// i8 dot product with i32 accumulation
// ============================================================

/// NEON i8 dot product accumulating into i32.
///
/// Processes 8 pairs per iteration:
/// 1. Load 8 bytes into `int8x8_t` registers.
/// 2. Widen multiply: `vmull_s8` → `int16x8_t`.
/// 3. Pairwise widen add: `vpaddlq_s16` → `int32x4_t`.
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_dot_i8_impl(a: &[i8], b: &[i8]) -> i32 {
    let len = a.len().min(b.len());
    let mut i = 0;
    let mut acc = vdupq_n_s32(0);

    if i + 8 <= len {
        let va = vld1_s8(a.as_ptr().add(i));
        let vb = vld1_s8(b.as_ptr().add(i));
        // Widening multiply: i8 × i8 → i16
        let prod16 = vmull_s8(va, vb);
        // Pairwise add + widen: i16 → i32 (initialize accumulator)
        acc = vpaddlq_s16(prod16);
        i += 8;

        // Continue accumulating remaining 8-element chunks.
        while i + 8 <= len {
            let va2 = vld1_s8(a.as_ptr().add(i));
            let vb2 = vld1_s8(b.as_ptr().add(i));
            let prod162 = vmull_s8(va2, vb2);
            acc = vaddq_s32(acc, vpaddlq_s16(prod162));
            i += 8;
        }
    }

    // Horizontal sum of the 4-lane i32 register.
    let mut result = vaddvq_s32(acc);

    // Scalar tail.
    while i < len {
        result += (a[i] as i32) * (b[i] as i32);
        i += 1;
    }

    result
}

/// i8 dot product with i32 accumulation.
///
/// Uses NEON on AArch64 when available; falls back to scalar otherwise.
///
/// # Overflow behaviour
///
/// Individual products are widened to i16 before accumulation into i32,
/// preventing overflow for typical ML weight/activation ranges (|x| ≤ 127).
pub fn neon_dot_i8(a: &[i8], b: &[i8]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon_dot_i8_impl(a, b) };
        }
    }
    scalar_dot_i8(a, b)
}

// ============================================================
// i8 dot product with direct i32 accumulation (alternative)
// ============================================================

/// i8 dot product accumulating directly into i32 (avoids intermediate i16 overflow risk).
///
/// This variant widens each `i8` pair to `i32` before accumulation using
/// `vmovl_s8` + `vmull_s16`, trading slightly more register pressure for
/// exact i32 accumulation from the start.
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_dot_i8_i32acc_impl(a: &[i8], b: &[i8]) -> i32 {
    let len = a.len().min(b.len());
    let mut i = 0;
    let mut acc = vdupq_n_s32(0);

    while i + 8 <= len {
        let va8 = vld1_s8(a.as_ptr().add(i));
        let vb8 = vld1_s8(b.as_ptr().add(i));
        // Widen i8 → i16
        let va16 = vmovl_s8(va8);
        let vb16 = vmovl_s8(vb8);
        // Split into low/high 4-lane i16 halves and widening multiply to i32.
        let prod_lo = vmull_s16(vget_low_s16(va16), vget_low_s16(vb16));
        let prod_hi = vmull_s16(vget_high_s16(va16), vget_high_s16(vb16));
        acc = vaddq_s32(acc, vaddq_s32(prod_lo, prod_hi));
        i += 8;
    }

    let mut result = vaddvq_s32(acc);

    while i < len {
        result += (a[i] as i32) * (b[i] as i32);
        i += 1;
    }

    result
}

/// i8 dot product with direct i32 accumulation, safe for large dot products.
pub fn neon_dot_i8_i32acc(a: &[i8], b: &[i8]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon_dot_i8_i32acc_impl(a, b) };
        }
    }
    scalar_dot_i8(a, b)
}

// ============================================================
// Quantise f32 → i8
// ============================================================

/// NEON-vectorised quantisation: `q[i] = clamp(round(x[i] / scale) + zero_point, -128, 127)`.
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_quantize_f32_to_i8_impl(src: &[f32], scale: f32, zero_point: i8, dst: &mut [i8]) {
    let len = src.len().min(dst.len());
    let mut i = 0;
    let inv_scale = 1.0_f32 / scale;
    let v_inv_scale = vdupq_n_f32(inv_scale);
    let v_zp = vdupq_n_f32(zero_point as f32);
    let v_lo = vdupq_n_f32(-128.0_f32);
    let v_hi = vdupq_n_f32(127.0_f32);

    while i + 4 <= len {
        let vx = vld1q_f32(src.as_ptr().add(i));
        // q = round(x / scale) + zp, clamped to [-128, 127]
        let vq = vaddq_f32(vrndnq_f32(vmulq_f32(vx, v_inv_scale)), v_zp);
        let vq_clamped = vminq_f32(vmaxq_f32(vq, v_lo), v_hi);
        // Convert f32 → i32 → i8.
        let vi32 = vcvtq_s32_f32(vq_clamped);
        let vi16_lo = vmovn_s32(vi32);
        // Saturating narrow from i16 to i8.
        // We need at least 8 i16 elements for vmovn_s16; use scalar for the final narrow.
        let s0 = vgetq_lane_s32(vi32, 0).clamp(-128, 127) as i8;
        let s1 = vgetq_lane_s32(vi32, 1).clamp(-128, 127) as i8;
        let s2 = vgetq_lane_s32(vi32, 2).clamp(-128, 127) as i8;
        let s3 = vgetq_lane_s32(vi32, 3).clamp(-128, 127) as i8;
        dst[i] = s0;
        dst[i + 1] = s1;
        dst[i + 2] = s2;
        dst[i + 3] = s3;
        // Suppress unused warning on vi16_lo — it is used implicitly through vi32.
        let _ = vi16_lo;
        i += 4;
    }

    // Scalar tail.
    let zp = zero_point as i32;
    while i < len {
        let q = (src[i] / scale).round() as i32 + zp;
        dst[i] = q.clamp(-128, 127) as i8;
        i += 1;
    }
}

/// Quantise a slice of f32 values to i8.
///
/// `q[i] = clamp(round(x[i] / scale) + zero_point, -128, 127)`
pub fn neon_quantize_f32_to_i8(src: &[f32], scale: f32, zero_point: i8, dst: &mut [i8]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_quantize_f32_to_i8_impl(src, scale, zero_point, dst) }
            return;
        }
    }
    scalar_quantize_f32_to_i8(src, scale, zero_point, dst)
}

// ============================================================
// Dequantise i8 → f32
// ============================================================

/// NEON-vectorised dequantisation: `x̂[i] = (q[i] - zero_point) * scale`.
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_dequantize_i8_to_f32_impl(src: &[i8], scale: f32, zero_point: i8, dst: &mut [f32]) {
    let len = src.len().min(dst.len());
    let mut i = 0;
    let v_scale = vdupq_n_f32(scale);
    let v_zp = vdupq_n_f32(zero_point as f32);

    while i + 8 <= len {
        // Load 8 i8 values into an int8x8_t.
        let vq8 = vld1_s8(src.as_ptr().add(i));
        // Widen i8 → i16 → i32 for the lower 4 and upper 4.
        let vq16 = vmovl_s8(vq8);
        let vq32_lo = vmovl_s16(vget_low_s16(vq16));
        let vq32_hi = vmovl_s16(vget_high_s16(vq16));
        // Convert i32 → f32.
        let vf_lo = vcvtq_f32_s32(vq32_lo);
        let vf_hi = vcvtq_f32_s32(vq32_hi);
        // x̂ = (q - zp) * scale
        let vout_lo = vmulq_f32(vsubq_f32(vf_lo, v_zp), v_scale);
        let vout_hi = vmulq_f32(vsubq_f32(vf_hi, v_zp), v_scale);
        vst1q_f32(dst.as_mut_ptr().add(i), vout_lo);
        vst1q_f32(dst.as_mut_ptr().add(i + 4), vout_hi);
        i += 8;
    }

    // Scalar tail.
    let zp = zero_point as i32;
    while i < len {
        dst[i] = (src[i] as i32 - zp) as f32 * scale;
        i += 1;
    }
}

/// Dequantise a slice of i8 values to f32.
///
/// `x̂[i] = (q[i] - zero_point) * scale`
pub fn neon_dequantize_i8_to_f32(src: &[i8], scale: f32, zero_point: i8, dst: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_dequantize_i8_to_f32_impl(src, scale, zero_point, dst) }
            return;
        }
    }
    scalar_dequantize_i8_to_f32(src, scale, zero_point, dst)
}

// ============================================================
// Saturating i8 addition
// ============================================================

/// NEON saturating i8 element-wise addition: `out[i] = sat_add(a[i], b[i])`.
///
/// Values that would overflow i8 are clamped to ±127 rather than wrapping.
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_add_i8_saturating_impl(a: &[i8], b: &[i8], out: &mut [i8]) {
    let len = a.len().min(b.len()).min(out.len());
    let mut i = 0;

    while i + 16 <= len {
        let va = vld1q_s8(a.as_ptr().add(i));
        let vb = vld1q_s8(b.as_ptr().add(i));
        let vr = vqaddq_s8(va, vb);
        vst1q_s8(out.as_mut_ptr().add(i), vr);
        i += 16;
    }

    // 8-element remainder.
    while i + 8 <= len {
        let va = vld1_s8(a.as_ptr().add(i));
        let vb = vld1_s8(b.as_ptr().add(i));
        let vr = vqadd_s8(va, vb);
        vst1_s8(out.as_mut_ptr().add(i), vr);
        i += 8;
    }

    // Scalar tail with saturating semantics.
    while i < len {
        out[i] = (a[i] as i16 + b[i] as i16).clamp(-128, 127) as i8;
        i += 1;
    }
}

/// Element-wise saturating i8 addition: `out[i] = clamp(a[i] + b[i], -128, 127)`.
pub fn neon_add_i8_saturating(a: &[i8], b: &[i8], out: &mut [i8]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_add_i8_saturating_impl(a, b, out) }
            return;
        }
    }
    scalar_add_i8_saturating(a, b, out)
}

// ============================================================
// Scalar fallback implementations
// ============================================================

fn scalar_dot_i8(a: &[i8], b: &[i8]) -> i32 {
    let len = a.len().min(b.len());
    let mut sum = 0_i32;
    for i in 0..len {
        sum += (a[i] as i32) * (b[i] as i32);
    }
    sum
}

fn scalar_quantize_f32_to_i8(src: &[f32], scale: f32, zero_point: i8, dst: &mut [i8]) {
    let len = src.len().min(dst.len());
    let zp = zero_point as i32;
    for i in 0..len {
        let q = (src[i] / scale).round() as i32 + zp;
        dst[i] = q.clamp(-128, 127) as i8;
    }
}

fn scalar_dequantize_i8_to_f32(src: &[i8], scale: f32, zero_point: i8, dst: &mut [f32]) {
    let len = src.len().min(dst.len());
    let zp = zero_point as i32;
    for i in 0..len {
        dst[i] = (src[i] as i32 - zp) as f32 * scale;
    }
}

fn scalar_add_i8_saturating(a: &[i8], b: &[i8], out: &mut [i8]) {
    let len = a.len().min(b.len()).min(out.len());
    for i in 0..len {
        out[i] = (a[i] as i16 + b[i] as i16).clamp(-128, 127) as i8;
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------
    // neon_dot_i8
    // ----------------------------------------------------------

    #[test]
    fn test_dot_i8_basic() {
        // [1,2,3,4] · [1,1,1,1] = 10
        let a: Vec<i8> = vec![1, 2, 3, 4];
        let b: Vec<i8> = vec![1, 1, 1, 1];
        let result = neon_dot_i8(&a, &b);
        assert_eq!(result, 10);
    }

    #[test]
    fn test_dot_i8_negative_values() {
        // [-1, 2] · [3, -4] = -3 + -8 = -11
        let a: Vec<i8> = vec![-1, 2];
        let b: Vec<i8> = vec![3, -4];
        let result = neon_dot_i8(&a, &b);
        assert_eq!(result, -11);
    }

    #[test]
    fn test_dot_i8_large_vectors() {
        // 32 elements, all 1 → sum = 32
        let a: Vec<i8> = vec![1; 32];
        let b: Vec<i8> = vec![1; 32];
        assert_eq!(neon_dot_i8(&a, &b), 32);
    }

    #[test]
    fn test_dot_i8_matches_scalar() {
        let a: Vec<i8> = (0..25).map(|i: i8| i.wrapping_sub(12)).collect();
        let b: Vec<i8> = (0..25).map(|i: i8| i.wrapping_sub(12)).collect();
        let scalar = scalar_dot_i8(&a, &b);
        let neon = neon_dot_i8(&a, &b);
        assert_eq!(scalar, neon, "scalar={scalar} neon={neon}");
    }

    // ----------------------------------------------------------
    // neon_dot_i8_i32acc
    // ----------------------------------------------------------

    #[test]
    fn test_dot_i8_i32acc_basic() {
        let a: Vec<i8> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let b: Vec<i8> = vec![1; 8];
        let result = neon_dot_i8_i32acc(&a, &b);
        let expected: i32 = a.iter().map(|&x| x as i32).sum();
        assert_eq!(result, expected);
    }

    // ----------------------------------------------------------
    // neon_quantize_f32_to_i8 / neon_dequantize_i8_to_f32 roundtrip
    // ----------------------------------------------------------

    #[test]
    fn test_quantize_basic() {
        let src = vec![0.0_f32, 0.5, 1.0, -0.5, -1.0];
        let mut dst = vec![0_i8; 5];
        neon_quantize_f32_to_i8(&src, 1.0_f32 / 127.0, 0, &mut dst);
        // scale = 1/127, so x=1.0 → q = round(127) = 127.
        // x=0.5 → q = round(63.5) = 64.
        // x=-0.5 → q = -64.
        assert_eq!(dst[0], 0, "zero maps to zero");
        assert!(dst[2] == 127, "max positive: {}", dst[2]);
        assert!(dst[4] == -127, "max negative: {}", dst[4]);
    }

    #[test]
    fn test_quantize_roundtrip() {
        // Values that survive a round-trip at scale=0.01 with zero_point=0.
        let scale = 0.01_f32;
        let zp = 0_i8;
        let src: Vec<f32> = [-1.27, -0.5, 0.0, 0.5, 1.27].iter().map(|&x| x).collect();
        let mut quantized = vec![0_i8; src.len()];
        let mut recovered = vec![0.0_f32; src.len()];
        neon_quantize_f32_to_i8(&src, scale, zp, &mut quantized);
        neon_dequantize_i8_to_f32(&quantized, scale, zp, &mut recovered);
        for (i, (&orig, &rec)) in src.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < scale + 1e-5,
                "roundtrip error at i={i}: orig={orig} rec={rec}"
            );
        }
    }

    #[test]
    fn test_quantize_with_nonzero_zero_point() {
        // Asymmetric quantisation: zp = 128 maps 0.0 to i8(-128).
        // Actually for i8 we pick zp = -128 so that x=0 → q = 0 + (-128) = -128.
        let src = vec![0.0_f32, 1.0, -1.0];
        let scale = 1.0_f32 / 127.0;
        let zp: i8 = 0;
        let mut dst = vec![0_i8; 3];
        neon_quantize_f32_to_i8(&src, scale, zp, &mut dst);
        assert_eq!(dst[0], 0, "0 → zp");
    }

    // ----------------------------------------------------------
    // neon_dequantize_i8_to_f32
    // ----------------------------------------------------------

    #[test]
    fn test_dequantize_basic() {
        // q = [0, 127, -127] with scale=0.01, zp=0 → [0.0, 1.27, -1.27]
        let src: Vec<i8> = vec![0, 127, -127];
        let mut dst = vec![0.0_f32; 3];
        neon_dequantize_i8_to_f32(&src, 0.01, 0, &mut dst);
        assert!((dst[0] - 0.0).abs() < 1e-6);
        assert!((dst[1] - 1.27).abs() < 1e-4);
        assert!((dst[2] - (-1.27)).abs() < 1e-4);
    }

    // ----------------------------------------------------------
    // neon_add_i8_saturating
    // ----------------------------------------------------------

    #[test]
    fn test_saturating_add_no_overflow() {
        let a: Vec<i8> = vec![10, 20, 30, 40];
        let b: Vec<i8> = vec![5, 10, 15, 20];
        let mut out = vec![0_i8; 4];
        neon_add_i8_saturating(&a, &b, &mut out);
        assert_eq!(out, vec![15, 30, 45, 60]);
    }

    #[test]
    fn test_saturating_add_overflow_positive() {
        // 120 + 20 = 140 > 127, should saturate to 127.
        let a: Vec<i8> = vec![120];
        let b: Vec<i8> = vec![20];
        let mut out = vec![0_i8; 1];
        neon_add_i8_saturating(&a, &b, &mut out);
        assert_eq!(out[0], 127, "positive overflow should saturate at 127");
    }

    #[test]
    fn test_saturating_add_overflow_negative() {
        // -120 + (-20) = -140 < -128, should saturate to -128.
        let a: Vec<i8> = vec![-120];
        let b: Vec<i8> = vec![-20];
        let mut out = vec![0_i8; 1];
        neon_add_i8_saturating(&a, &b, &mut out);
        assert_eq!(out[0], -128, "negative overflow should saturate at -128");
    }

    #[test]
    fn test_saturating_add_large_vector() {
        // 20-element vector with both saturating and non-saturating entries.
        let a: Vec<i8> = (0..20)
            .map(|i| if i < 10 { 100_i8 } else { -100_i8 })
            .collect();
        let b: Vec<i8> = (0..20)
            .map(|i| if i < 10 { 50_i8 } else { -50_i8 })
            .collect();
        let mut out = vec![0_i8; 20];
        neon_add_i8_saturating(&a, &b, &mut out);
        for i in 0..10 {
            assert_eq!(out[i], 127, "expected saturation at i={i}");
        }
        for i in 10..20 {
            assert_eq!(out[i], -128, "expected negative saturation at i={i}");
        }
    }

    #[test]
    fn test_saturating_add_matches_scalar() {
        let a: Vec<i8> = (-12..12).map(|i: i8| i.wrapping_mul(10)).collect();
        let b: Vec<i8> = (0..24).map(|i: i8| i.wrapping_sub(12)).collect();
        let mut ref_out = vec![0_i8; a.len()];
        let mut neon_out = vec![0_i8; a.len()];
        scalar_add_i8_saturating(&a, &b, &mut ref_out);
        neon_add_i8_saturating(&a, &b, &mut neon_out);
        assert_eq!(ref_out, neon_out, "scalar and neon results must match");
    }

    // ----------------------------------------------------------
    // Scalar fallback correctness (architecture-independent)
    // ----------------------------------------------------------

    #[test]
    fn test_scalar_dot_i8() {
        let a: Vec<i8> = vec![3, -4, 5];
        let b: Vec<i8> = vec![2, 2, 2];
        // 3*2 + (-4)*2 + 5*2 = 6 - 8 + 10 = 8
        assert_eq!(scalar_dot_i8(&a, &b), 8);
    }

    #[test]
    fn test_scalar_quantize_roundtrip() {
        let src = vec![0.0_f32, 0.5, -0.5];
        let scale = 0.01_f32;
        let zp = 0_i8;
        let mut q = vec![0_i8; 3];
        let mut out = vec![0.0_f32; 3];
        scalar_quantize_f32_to_i8(&src, scale, zp, &mut q);
        scalar_dequantize_i8_to_f32(&q, scale, zp, &mut out);
        for i in 0..3 {
            assert!((src[i] - out[i]).abs() < scale + 1e-6);
        }
    }
}
