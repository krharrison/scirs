//! NEON fused multiply-add, scaling, and extended unary arithmetic
//!
//! This module provides ARM NEON-accelerated fused-multiply-add (FMA),
//! element-wise scaling, absolute value, and negation for f32 and f64 slices.
//!
//! All public functions are safe wrappers that dispatch to the NEON intrinsic
//! path when running on AArch64 with NEON support, and fall back to scalar
//! otherwise so they are correct on every platform.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ============================================================
// Fused multiply-add: out[i] = a[i] * b[i] + c[i]
// ============================================================

/// NEON fused multiply-add for f32: `out[i] = a[i] * b[i] + c[i]`.
///
/// Uses `vfmaq_f32` which computes the result in a single fused operation,
/// preserving precision and typically executing in fewer cycles than a separate
/// multiply followed by an add.
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON. Slices must be valid for reading/
/// writing to their respective lengths.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_fmadd_f32_impl(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    let len = a.len().min(b.len()).min(c.len()).min(out.len());
    let mut i = 0;

    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        let vc = vld1q_f32(c.as_ptr().add(i));
        // vfmaq_f32(c, a, b) = a * b + c
        let vr = vfmaq_f32(vc, va, vb);
        vst1q_f32(out.as_mut_ptr().add(i), vr);
        i += 4;
    }

    // Scalar tail
    while i < len {
        out[i] = a[i] * b[i] + c[i];
        i += 1;
    }
}

/// Fused multiply-add for f32: `out[i] = a[i] * b[i] + c[i]`.
///
/// Uses NEON on AArch64 when detected; falls back to scalar otherwise.
pub fn neon_fmadd_f32(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_fmadd_f32_impl(a, b, c, out) }
            return;
        }
    }
    fallback_fmadd_f32(a, b, c, out)
}

/// NEON fused multiply-add for f64: `out[i] = a[i] * b[i] + c[i]`.
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON. Slices must be valid.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_fmadd_f64_impl(a: &[f64], b: &[f64], c: &[f64], out: &mut [f64]) {
    let len = a.len().min(b.len()).min(c.len()).min(out.len());
    let mut i = 0;

    while i + 2 <= len {
        let va = vld1q_f64(a.as_ptr().add(i));
        let vb = vld1q_f64(b.as_ptr().add(i));
        let vc = vld1q_f64(c.as_ptr().add(i));
        let vr = vfmaq_f64(vc, va, vb);
        vst1q_f64(out.as_mut_ptr().add(i), vr);
        i += 2;
    }

    while i < len {
        out[i] = a[i] * b[i] + c[i];
        i += 1;
    }
}

/// Fused multiply-add for f64: `out[i] = a[i] * b[i] + c[i]`.
pub fn neon_fmadd_f64(a: &[f64], b: &[f64], c: &[f64], out: &mut [f64]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_fmadd_f64_impl(a, b, c, out) }
            return;
        }
    }
    fallback_fmadd_f64(a, b, c, out)
}

// ============================================================
// Element-wise scalar multiply (scale)
// ============================================================

/// NEON element-wise scale for f32: `out[i] = a[i] * scale`.
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_scale_f32_impl(a: &[f32], scale: f32, out: &mut [f32]) {
    let len = a.len().min(out.len());
    let mut i = 0;
    let vscale = vdupq_n_f32(scale);

    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vr = vmulq_f32(va, vscale);
        vst1q_f32(out.as_mut_ptr().add(i), vr);
        i += 4;
    }

    while i < len {
        out[i] = a[i] * scale;
        i += 1;
    }
}

/// Element-wise scalar multiply for f32: `out[i] = a[i] * scale`.
pub fn neon_scale_f32(a: &[f32], scale: f32, out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_scale_f32_impl(a, scale, out) }
            return;
        }
    }
    fallback_scale_f32(a, scale, out)
}

/// NEON element-wise scale for f64: `out[i] = a[i] * scale`.
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_scale_f64_impl(a: &[f64], scale: f64, out: &mut [f64]) {
    let len = a.len().min(out.len());
    let mut i = 0;
    let vscale = vdupq_n_f64(scale);

    while i + 2 <= len {
        let va = vld1q_f64(a.as_ptr().add(i));
        let vr = vmulq_f64(va, vscale);
        vst1q_f64(out.as_mut_ptr().add(i), vr);
        i += 2;
    }

    while i < len {
        out[i] = a[i] * scale;
        i += 1;
    }
}

/// Element-wise scalar multiply for f64: `out[i] = a[i] * scale`.
pub fn neon_scale_f64(a: &[f64], scale: f64, out: &mut [f64]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_scale_f64_impl(a, scale, out) }
            return;
        }
    }
    fallback_scale_f64(a, scale, out)
}

// ============================================================
// Absolute value
// ============================================================

/// NEON absolute value for f32: `out[i] = |a[i]|`.
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_abs_f32_impl(a: &[f32], out: &mut [f32]) {
    let len = a.len().min(out.len());
    let mut i = 0;

    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vr = vabsq_f32(va);
        vst1q_f32(out.as_mut_ptr().add(i), vr);
        i += 4;
    }

    while i < len {
        out[i] = a[i].abs();
        i += 1;
    }
}

/// Element-wise absolute value for f32: `out[i] = |a[i]|`.
pub fn neon_abs_f32(a: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_abs_f32_impl(a, out) }
            return;
        }
    }
    fallback_abs_f32(a, out)
}

/// NEON absolute value for f64: `out[i] = |a[i]|`.
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_abs_f64_impl(a: &[f64], out: &mut [f64]) {
    let len = a.len().min(out.len());
    let mut i = 0;

    while i + 2 <= len {
        let va = vld1q_f64(a.as_ptr().add(i));
        let vr = vabsq_f64(va);
        vst1q_f64(out.as_mut_ptr().add(i), vr);
        i += 2;
    }

    while i < len {
        out[i] = a[i].abs();
        i += 1;
    }
}

/// Element-wise absolute value for f64: `out[i] = |a[i]|`.
pub fn neon_abs_f64(a: &[f64], out: &mut [f64]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_abs_f64_impl(a, out) }
            return;
        }
    }
    fallback_abs_f64(a, out)
}

// ============================================================
// Negation
// ============================================================

/// NEON element-wise negation for f32: `out[i] = -a[i]`.
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_neg_f32_impl(a: &[f32], out: &mut [f32]) {
    let len = a.len().min(out.len());
    let mut i = 0;

    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vr = vnegq_f32(va);
        vst1q_f32(out.as_mut_ptr().add(i), vr);
        i += 4;
    }

    while i < len {
        out[i] = -a[i];
        i += 1;
    }
}

/// Element-wise negation for f32: `out[i] = -a[i]`.
pub fn neon_neg_f32(a: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_neg_f32_impl(a, out) }
            return;
        }
    }
    fallback_neg_f32(a, out)
}

// ============================================================
// Scalar fallback implementations
// ============================================================

fn fallback_fmadd_f32(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    let len = a.len().min(b.len()).min(c.len()).min(out.len());
    for i in 0..len {
        out[i] = a[i] * b[i] + c[i];
    }
}

fn fallback_fmadd_f64(a: &[f64], b: &[f64], c: &[f64], out: &mut [f64]) {
    let len = a.len().min(b.len()).min(c.len()).min(out.len());
    for i in 0..len {
        out[i] = a[i] * b[i] + c[i];
    }
}

fn fallback_scale_f32(a: &[f32], scale: f32, out: &mut [f32]) {
    let len = a.len().min(out.len());
    for i in 0..len {
        out[i] = a[i] * scale;
    }
}

fn fallback_scale_f64(a: &[f64], scale: f64, out: &mut [f64]) {
    let len = a.len().min(out.len());
    for i in 0..len {
        out[i] = a[i] * scale;
    }
}

fn fallback_abs_f32(a: &[f32], out: &mut [f32]) {
    let len = a.len().min(out.len());
    for i in 0..len {
        out[i] = a[i].abs();
    }
}

fn fallback_abs_f64(a: &[f64], out: &mut [f64]) {
    let len = a.len().min(out.len());
    for i in 0..len {
        out[i] = a[i].abs();
    }
}

fn fallback_neg_f32(a: &[f32], out: &mut [f32]) {
    let len = a.len().min(out.len());
    for i in 0..len {
        out[i] = -a[i];
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------
    // neon_fmadd_f32
    // ----------------------------------------------------------

    #[test]
    fn test_fmadd_f32_basic() {
        // out[i] = a[i] * b[i] + c[i]
        // [1,2,3,4] * [2,2,2,2] + [1,1,1,1] = [3,5,7,9]
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![2.0_f32; 4];
        let c = vec![1.0_f32; 4];
        let mut out = vec![0.0_f32; 4];
        neon_fmadd_f32(&a, &b, &c, &mut out);
        assert!((out[0] - 3.0).abs() < 1e-6, "out[0]={}", out[0]);
        assert!((out[1] - 5.0).abs() < 1e-6, "out[1]={}", out[1]);
        assert!((out[2] - 7.0).abs() < 1e-6, "out[2]={}", out[2]);
        assert!((out[3] - 9.0).abs() < 1e-6, "out[3]={}", out[3]);
    }

    #[test]
    fn test_fmadd_f32_non_multiple_of_4() {
        // 5 elements to exercise the scalar tail.
        let a = vec![1.0_f32, 1.0, 1.0, 1.0, 2.0];
        let b = vec![3.0_f32; 5];
        let c = vec![0.0_f32; 5];
        let mut out = vec![0.0_f32; 5];
        neon_fmadd_f32(&a, &b, &c, &mut out);
        for i in 0..4 {
            assert!((out[i] - 3.0).abs() < 1e-6, "out[{i}]={}", out[i]);
        }
        assert!((out[4] - 6.0).abs() < 1e-6, "out[4]={}", out[4]);
    }

    // ----------------------------------------------------------
    // neon_fmadd_f64
    // ----------------------------------------------------------

    #[test]
    fn test_fmadd_f64_basic() {
        let a = vec![1.0_f64, 2.0, 3.0];
        let b = vec![4.0_f64, 5.0, 6.0];
        let c = vec![10.0_f64, 20.0, 30.0];
        let mut out = vec![0.0_f64; 3];
        neon_fmadd_f64(&a, &b, &c, &mut out);
        // [1*4+10, 2*5+20, 3*6+30] = [14, 30, 48]
        assert!((out[0] - 14.0).abs() < 1e-10);
        assert!((out[1] - 30.0).abs() < 1e-10);
        assert!((out[2] - 48.0).abs() < 1e-10);
    }

    // ----------------------------------------------------------
    // neon_scale_f32
    // ----------------------------------------------------------

    #[test]
    fn test_scale_f32_basic() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let mut out = vec![0.0_f32; 5];
        neon_scale_f32(&a, 2.5, &mut out);
        let expected = [2.5, 5.0, 7.5, 10.0, 12.5];
        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (out[i] - exp).abs() < 1e-5,
                "out[{i}]={} expected {exp}",
                out[i]
            );
        }
    }

    #[test]
    fn test_scale_f32_zero() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut out = vec![99.0_f32; 4];
        neon_scale_f32(&a, 0.0, &mut out);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    // ----------------------------------------------------------
    // neon_scale_f64
    // ----------------------------------------------------------

    #[test]
    fn test_scale_f64_basic() {
        let a = vec![1.0_f64, -2.0, 3.0];
        let mut out = vec![0.0_f64; 3];
        neon_scale_f64(&a, -1.0, &mut out);
        assert!((out[0] - (-1.0)).abs() < 1e-10);
        assert!((out[1] - 2.0).abs() < 1e-10);
        assert!((out[2] - (-3.0)).abs() < 1e-10);
    }

    // ----------------------------------------------------------
    // neon_abs_f32
    // ----------------------------------------------------------

    #[test]
    fn test_abs_f32_mixed() {
        let a = vec![-3.0_f32, -1.0, 0.0, 1.0, 3.0, -5.0, 2.0];
        let mut out = vec![0.0_f32; 7];
        neon_abs_f32(&a, &mut out);
        let expected = [3.0, 1.0, 0.0, 1.0, 3.0, 5.0, 2.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (out[i] - exp).abs() < 1e-6,
                "out[{i}]={} expected {exp}",
                out[i]
            );
        }
    }

    // ----------------------------------------------------------
    // neon_abs_f64
    // ----------------------------------------------------------

    #[test]
    fn test_abs_f64_basic() {
        let a = vec![-2.5_f64, 3.7, -0.0];
        let mut out = vec![0.0_f64; 3];
        neon_abs_f64(&a, &mut out);
        assert!((out[0] - 2.5).abs() < 1e-10);
        assert!((out[1] - 3.7).abs() < 1e-10);
        assert!(out[2].abs() < 1e-10);
    }

    // ----------------------------------------------------------
    // neon_neg_f32
    // ----------------------------------------------------------

    #[test]
    fn test_neg_f32_basic() {
        let a = vec![1.0_f32, -2.0, 0.0, 4.0, -5.0];
        let mut out = vec![0.0_f32; 5];
        neon_neg_f32(&a, &mut out);
        let expected = [-1.0, 2.0, 0.0, -4.0, 5.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (out[i] - exp).abs() < 1e-6,
                "out[{i}]={} expected {exp}",
                out[i]
            );
        }
    }

    // ----------------------------------------------------------
    // Cross-validation: fallback vs wrapper
    // ----------------------------------------------------------

    #[test]
    fn test_fmadd_f32_matches_fallback() {
        let a: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..13).map(|i| (i as f32) * 0.5).collect();
        let c: Vec<f32> = vec![1.0_f32; 13];
        let mut out_fallback = vec![0.0_f32; 13];
        let mut out_neon = vec![0.0_f32; 13];
        fallback_fmadd_f32(&a, &b, &c, &mut out_fallback);
        neon_fmadd_f32(&a, &b, &c, &mut out_neon);
        for i in 0..13 {
            assert!(
                (out_fallback[i] - out_neon[i]).abs() < 1e-5,
                "mismatch at i={i}: fallback={} neon={}",
                out_fallback[i],
                out_neon[i]
            );
        }
    }

    #[test]
    fn test_scale_f32_matches_fallback() {
        let a: Vec<f32> = (0..17).map(|i| i as f32 - 8.0).collect();
        let scale = 1.23_f32;
        let mut out_fallback = vec![0.0_f32; 17];
        let mut out_neon = vec![0.0_f32; 17];
        fallback_scale_f32(&a, scale, &mut out_fallback);
        neon_scale_f32(&a, scale, &mut out_neon);
        for i in 0..17 {
            assert!(
                (out_fallback[i] - out_neon[i]).abs() < 1e-5,
                "mismatch at i={i}"
            );
        }
    }
}
