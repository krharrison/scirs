//! ARM Scalable Vector Extension (SVE / SVE2) capability detection and dispatch
//!
//! SVE is a variable-length SIMD architecture available on ARM Neoverse N1/V1/V2
//! server processors and Apple silicon (M4 onwards for SVE2).  Unlike NEON which
//! always operates on 128-bit vectors, SVE vector length (VLEN) is implementation
//! defined at 128–2048 bits in 128-bit increments.
//!
//! ## Design
//!
//! Stable Rust does not yet expose SVE intrinsics directly via `std::arch::aarch64`.
//! This module therefore:
//!
//! 1. **Detects** SVE availability at runtime via `is_aarch64_feature_detected!`.
//! 2. **Queries** the hardware vector length via inline assembly (`rdvl` instruction).
//! 3. **Dispatches** to the highest-quality path available:
//!    - SVE (via target_feature = "sve") if the toolchain and hardware support it,
//!    - otherwise NEON (128-bit),
//!    - otherwise scalar.
//!
//! When SVE intrinsics are stabilised in `std::arch::aarch64`, the
//! `#[target_feature(enable = "sve")]` implementations can be filled in here with
//! real vector operations without changing the public API.
//!
//! ## References
//!
//! - Arm Architecture Reference Manual — SVE supplement (DDI 0584)
//! - LLVM `acle.h` SVE intrinsic definitions
//! - <https://github.com/ARM-software/acle>

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ============================================================
// Capability types
// ============================================================

/// SVE / SVE2 capability information for the current CPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SveCapabilities {
    /// Whether SVE is available (runtime-detected).
    pub has_sve: bool,
    /// Whether SVE2 is available (superset of SVE).
    pub has_sve2: bool,
    /// Vector length in *bytes* as reported by the hardware (`rdvl` instruction).
    ///
    /// On processors that do not implement SVE this is 0.
    /// On NEON-only processors this field is 0; NEON is always 16 bytes (128 bits).
    pub vector_len_bytes: usize,
    /// Whether the `dotprod` extension (UDOT/SDOT for i8→i32) is present.
    pub has_dotprod: bool,
}

impl Default for SveCapabilities {
    fn default() -> Self {
        detect_sve_capabilities()
    }
}

// ============================================================
// Runtime capability detection
// ============================================================

/// Detect SVE and SVE2 capabilities for the current CPU.
///
/// This function queries:
/// - `is_aarch64_feature_detected!("sve")` / `"sve2"` for SVE presence,
/// - the hardware vector length via the `rdvl` instruction (AArch64 only),
/// - `is_aarch64_feature_detected!("dotprod")` for the dot-product extension.
///
/// On non-AArch64 platforms all fields are `false`/`0`.
pub fn detect_sve_capabilities() -> SveCapabilities {
    #[cfg(target_arch = "aarch64")]
    {
        let has_sve = std::arch::is_aarch64_feature_detected!("sve");
        // SVE2 implies SVE.
        let has_sve2 = std::arch::is_aarch64_feature_detected!("sve2");
        let has_dotprod = std::arch::is_aarch64_feature_detected!("dotprod");

        // Query the hardware vector length in bytes.
        // `rdvl x0, #1` reads the vector-length-in-bytes into x0.
        // We only execute this if the CPU actually has SVE.
        let vector_len_bytes = if has_sve {
            unsafe { read_vector_length_bytes() }
        } else {
            0
        };

        SveCapabilities {
            has_sve,
            has_sve2,
            vector_len_bytes,
            has_dotprod,
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        SveCapabilities {
            has_sve: false,
            has_sve2: false,
            vector_len_bytes: 0,
            has_dotprod: false,
        }
    }
}

/// Read the SVE hardware vector length in bytes.
///
/// When compiled with `target_feature = "sve"` (e.g., `-C target-feature=+sve`),
/// uses the `rdvl` instruction to query the hardware VLEN directly.
///
/// When compiled without SVE support, returns 16 as a minimum lower bound
/// (NEON width) for VLEN — this value is conservative and safe since all
/// SVE implementations must support at least 128-bit vectors.
///
/// # Safety
///
/// Must only be called after runtime-detecting SVE via
/// `is_aarch64_feature_detected!("sve")`.
#[cfg(target_arch = "aarch64")]
unsafe fn read_vector_length_bytes() -> usize {
    // Only emit rdvl when the compiler can also encode it.
    #[cfg(target_feature = "sve")]
    {
        let vl: usize;
        core::arch::asm!(
            "rdvl {vl}, #1",
            vl = out(reg) vl,
            options(nostack, pure, nomem)
        );
        vl
    }
    // Without compile-time SVE we cannot emit rdvl, so we return the minimum
    // SVE vector length (128 bits = 16 bytes) as a safe sentinel.  At runtime
    // we only reach this path if `is_aarch64_feature_detected!("sve")` returned
    // true, meaning the hardware does have SVE — but the binary was not compiled
    // with +sve so we conservatively report the architectural minimum VLEN.
    #[cfg(not(target_feature = "sve"))]
    {
        16_usize // minimum SVE VLEN in bytes (128-bit)
    }
}

// ============================================================
// Public convenience functions
// ============================================================

/// Returns `true` if the CPU supports SVE instructions.
///
/// On non-AArch64 platforms this always returns `false`.
#[inline]
pub fn has_sve() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("sve")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Returns `true` if the CPU supports SVE2 instructions.
///
/// On non-AArch64 platforms this always returns `false`.
#[inline]
pub fn has_sve2() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("sve2")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

// ============================================================
// sve_add_f32
// ============================================================

/// Element-wise f32 addition using the best available SIMD path.
///
/// Priority:
/// 1. SVE (when Rust gains stable SVE intrinsics and hardware is detected)
/// 2. NEON (128-bit, stable today on AArch64)
/// 3. Scalar fallback
///
/// Today the SVE branch falls through to NEON because stable Rust does not
/// yet expose SVE vector intrinsics.  When they are stabilised, the
/// `#[target_feature(enable = "sve")]` block should be uncommented.
pub fn sve_add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_add_f32_path(a, b, out) }
            return;
        }
    }
    scalar_add_f32(a, b, out)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_add_f32_path(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len().min(b.len()).min(out.len());
    let mut i = 0;
    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        vst1q_f32(out.as_mut_ptr().add(i), vaddq_f32(va, vb));
        i += 4;
    }
    while i < len {
        out[i] = a[i] + b[i];
        i += 1;
    }
}

fn scalar_add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len().min(b.len()).min(out.len());
    for i in 0..len {
        out[i] = a[i] + b[i];
    }
}

// ============================================================
// sve_dot_f32
// ============================================================

/// Dot product using the best available SIMD path.
///
/// Falls back through SVE → NEON → scalar as above.
pub fn sve_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon_dot_f32_path(a, b) };
        }
    }
    scalar_dot_f32(a, b)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_dot_f32_path(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut i = 0;
    let mut acc = vdupq_n_f32(0.0_f32);
    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        acc = vfmaq_f32(acc, va, vb);
        i += 4;
    }
    let mut result = vaddvq_f32(acc);
    while i < len {
        result += a[i] * b[i];
        i += 1;
    }
    result
}

fn scalar_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = 0.0_f32;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

// ============================================================
// sve_sum_f32
// ============================================================

/// Horizontal sum using the best available SIMD path.
pub fn sve_sum_f32(a: &[f32]) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon_sum_f32_path(a) };
        }
    }
    scalar_sum_f32(a)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_sum_f32_path(a: &[f32]) -> f32 {
    let len = a.len();
    let mut i = 0;
    let mut acc = vdupq_n_f32(0.0_f32);
    while i + 4 <= len {
        acc = vaddq_f32(acc, vld1q_f32(a.as_ptr().add(i)));
        i += 4;
    }
    let mut result = vaddvq_f32(acc);
    while i < len {
        result += a[i];
        i += 1;
    }
    result
}

fn scalar_sum_f32(a: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for &x in a {
        sum += x;
    }
    sum
}

// ============================================================
// sve_scale_f32
// ============================================================

/// Element-wise scalar multiply using the best available SIMD path.
pub fn sve_scale_f32(a: &[f32], scale: f32, out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_scale_f32_path(a, scale, out) }
            return;
        }
    }
    scalar_scale_f32(a, scale, out)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_scale_f32_path(a: &[f32], scale: f32, out: &mut [f32]) {
    let len = a.len().min(out.len());
    let mut i = 0;
    let vscale = vdupq_n_f32(scale);
    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        vst1q_f32(out.as_mut_ptr().add(i), vmulq_f32(va, vscale));
        i += 4;
    }
    while i < len {
        out[i] = a[i] * scale;
        i += 1;
    }
}

fn scalar_scale_f32(a: &[f32], scale: f32, out: &mut [f32]) {
    let len = a.len().min(out.len());
    for i in 0..len {
        out[i] = a[i] * scale;
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------
    // Detection
    // ----------------------------------------------------------

    #[test]
    fn test_sve_detection_does_not_panic() {
        // Simply verify the detection routines complete without panicking.
        let caps = detect_sve_capabilities();
        // On non-SVE hardware has_sve must be false.
        // If SVE is present, vector_len_bytes must be a multiple of 16.
        if caps.has_sve {
            assert!(
                caps.vector_len_bytes >= 16,
                "SVE vector length must be at least 16 bytes"
            );
            assert_eq!(
                caps.vector_len_bytes % 16,
                0,
                "SVE vector length must be a multiple of 16 bytes"
            );
        } else {
            assert_eq!(caps.vector_len_bytes, 0);
        }
        // Smoke-test the convenience functions.
        assert_eq!(has_sve(), caps.has_sve);
        assert_eq!(has_sve2(), caps.has_sve2);
    }

    #[test]
    fn test_has_sve2_implies_has_sve() {
        // SVE2 is a superset; if SVE2 is present so is SVE.
        if has_sve2() {
            assert!(has_sve(), "SVE2 implies SVE");
        }
    }

    // ----------------------------------------------------------
    // sve_add_f32
    // ----------------------------------------------------------

    #[test]
    fn test_sve_add_f32_basic() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0_f32, 20.0, 30.0, 40.0, 50.0];
        let mut out = vec![0.0_f32; 5];
        sve_add_f32(&a, &b, &mut out);
        let expected = [11.0, 22.0, 33.0, 44.0, 55.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (out[i] - exp).abs() < 1e-6,
                "out[{i}]={} expected {exp}",
                out[i]
            );
        }
    }

    #[test]
    fn test_sve_add_f32_large() {
        let n = 100;
        let a = vec![1.0_f32; n];
        let b = vec![2.0_f32; n];
        let mut out = vec![0.0_f32; n];
        sve_add_f32(&a, &b, &mut out);
        for v in &out {
            assert!((*v - 3.0).abs() < 1e-6);
        }
    }

    // ----------------------------------------------------------
    // sve_dot_f32
    // ----------------------------------------------------------

    #[test]
    fn test_sve_dot_f32_basic() {
        // [1,2,3,4] · [1,1,1,1] = 10
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![1.0_f32; 4];
        let result = sve_dot_f32(&a, &b);
        assert!((result - 10.0).abs() < 1e-5, "expected 10.0 got {result}");
    }

    #[test]
    fn test_sve_dot_f32_matches_scalar() {
        let a: Vec<f32> = (0..21).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..21).map(|i| (i as f32) * 0.5).collect();
        let scalar = scalar_dot_f32(&a, &b);
        let simd = sve_dot_f32(&a, &b);
        assert!(
            (scalar - simd).abs() <= scalar.abs() * 1e-5 + 1e-5,
            "scalar={scalar} simd={simd}"
        );
    }

    // ----------------------------------------------------------
    // sve_sum_f32
    // ----------------------------------------------------------

    #[test]
    fn test_sve_sum_f32_basic() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let result = sve_sum_f32(&a);
        assert!((result - 28.0).abs() < 1e-5, "expected 28.0 got {result}");
    }

    #[test]
    fn test_sve_sum_f32_empty() {
        assert_eq!(sve_sum_f32(&[]), 0.0);
    }

    // ----------------------------------------------------------
    // sve_scale_f32
    // ----------------------------------------------------------

    #[test]
    fn test_sve_scale_f32_basic() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let mut out = vec![0.0_f32; 5];
        sve_scale_f32(&a, 3.0, &mut out);
        let expected = [3.0, 6.0, 9.0, 12.0, 15.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (out[i] - exp).abs() < 1e-5,
                "out[{i}]={} expected {exp}",
                out[i]
            );
        }
    }

    #[test]
    fn test_sve_scale_f32_matches_scalar() {
        let a: Vec<f32> = (0..17).map(|i| i as f32 - 8.0).collect();
        let scale = 2.71828_f32;
        let mut ref_out = vec![0.0_f32; 17];
        let mut simd_out = vec![0.0_f32; 17];
        scalar_scale_f32(&a, scale, &mut ref_out);
        sve_scale_f32(&a, scale, &mut simd_out);
        for i in 0..17 {
            assert!((ref_out[i] - simd_out[i]).abs() < 1e-5, "mismatch at i={i}");
        }
    }
}
