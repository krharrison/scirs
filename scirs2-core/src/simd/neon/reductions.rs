//! NEON-optimized horizontal reduction operations
//!
//! This module provides ARM NEON-accelerated horizontal reductions:
//! - Sum: accumulate all elements into a single scalar
//! - Max: horizontal maximum across all elements
//! - Min: horizontal minimum across all elements
//!
//! Each operation has both an f32 and f64 variant. All functions include
//! a safe public wrapper that dispatches to the NEON implementation when
//! available, with automatic scalar fallback.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ============================================================
// NEON horizontal sum for f32
// ============================================================

/// NEON-accelerated horizontal sum of all f32 elements.
///
/// Accumulates four NEON lanes in parallel and then does a pairwise reduction.
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON. `a` must be a valid, non-empty slice.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_sum_f32_impl(a: &[f32]) -> f32 {
    let len = a.len();
    let mut i = 0;
    // Use two accumulators to enable instruction-level parallelism.
    let mut acc0 = vdupq_n_f32(0.0_f32);
    let mut acc1 = vdupq_n_f32(0.0_f32);

    // Process 8 elements per iteration (two 128-bit NEON registers).
    while i + 8 <= len {
        let v0 = vld1q_f32(a.as_ptr().add(i));
        let v1 = vld1q_f32(a.as_ptr().add(i + 4));
        acc0 = vaddq_f32(acc0, v0);
        acc1 = vaddq_f32(acc1, v1);
        i += 8;
    }

    // Merge the two accumulators.
    let mut acc = vaddq_f32(acc0, acc1);

    // Drain the remaining 4-element chunk.
    if i + 4 <= len {
        let v = vld1q_f32(a.as_ptr().add(i));
        acc = vaddq_f32(acc, v);
        i += 4;
    }

    // Horizontal reduction of the NEON register.
    let mut result = vaddvq_f32(acc);

    // Scalar tail.
    while i < len {
        result += a[i];
        i += 1;
    }

    result
}

/// Horizontal sum of all f32 elements.
///
/// Uses NEON on AArch64 when available; falls back to scalar otherwise.
pub fn neon_sum_f32(a: &[f32]) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon_sum_f32_impl(a) };
        }
    }
    fallback_sum_f32(a)
}

// ============================================================
// NEON horizontal sum for f64
// ============================================================

/// NEON-accelerated horizontal sum of all f64 elements.
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON. `a` must be a valid, non-empty slice.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_sum_f64_impl(a: &[f64]) -> f64 {
    let len = a.len();
    let mut i = 0;
    let mut acc0 = vdupq_n_f64(0.0_f64);
    let mut acc1 = vdupq_n_f64(0.0_f64);

    // Process 4 elements per iteration (two 128-bit NEON registers).
    while i + 4 <= len {
        let v0 = vld1q_f64(a.as_ptr().add(i));
        let v1 = vld1q_f64(a.as_ptr().add(i + 2));
        acc0 = vaddq_f64(acc0, v0);
        acc1 = vaddq_f64(acc1, v1);
        i += 4;
    }

    let mut acc = vaddq_f64(acc0, acc1);

    // Drain remaining 2-element chunk.
    if i + 2 <= len {
        let v = vld1q_f64(a.as_ptr().add(i));
        acc = vaddq_f64(acc, v);
        i += 2;
    }

    // Horizontal reduction: add the two 64-bit lanes.
    let mut result = vaddvq_f64(acc);

    // Scalar tail.
    while i < len {
        result += a[i];
        i += 1;
    }

    result
}

/// Horizontal sum of all f64 elements.
///
/// Uses NEON on AArch64 when available; falls back to scalar otherwise.
pub fn neon_sum_f64(a: &[f64]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon_sum_f64_impl(a) };
        }
    }
    fallback_sum_f64(a)
}

// ============================================================
// NEON horizontal max for f32
// ============================================================

/// NEON-accelerated horizontal maximum of f32 elements.
///
/// Returns `f32::NEG_INFINITY` for an empty slice.
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON. `a` must be a valid slice.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_max_f32_impl(a: &[f32]) -> f32 {
    let len = a.len();
    let mut i = 0;
    let mut acc = vdupq_n_f32(f32::NEG_INFINITY);

    while i + 4 <= len {
        let v = vld1q_f32(a.as_ptr().add(i));
        acc = vmaxq_f32(acc, v);
        i += 4;
    }

    // Horizontal max of the 4-lane register.
    let mut result = vmaxvq_f32(acc);

    while i < len {
        if a[i] > result {
            result = a[i];
        }
        i += 1;
    }

    result
}

/// Horizontal maximum of f32 elements.
///
/// Returns `f32::NEG_INFINITY` for an empty slice.
pub fn neon_max_f32(a: &[f32]) -> f32 {
    if a.is_empty() {
        return f32::NEG_INFINITY;
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon_max_f32_impl(a) };
        }
    }
    fallback_max_f32(a)
}

// ============================================================
// NEON horizontal max for f64
// ============================================================

/// NEON-accelerated horizontal maximum of f64 elements.
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_max_f64_impl(a: &[f64]) -> f64 {
    let len = a.len();
    let mut i = 0;
    let mut acc = vdupq_n_f64(f64::NEG_INFINITY);

    while i + 2 <= len {
        let v = vld1q_f64(a.as_ptr().add(i));
        acc = vmaxq_f64(acc, v);
        i += 2;
    }

    // Horizontal max of the 2-lane f64 register.
    // Extract both lanes and compare.
    let lo = vgetq_lane_f64(acc, 0);
    let hi = vgetq_lane_f64(acc, 1);
    let mut result = if lo > hi { lo } else { hi };

    while i < len {
        if a[i] > result {
            result = a[i];
        }
        i += 1;
    }

    result
}

/// Horizontal maximum of f64 elements.
///
/// Returns `f64::NEG_INFINITY` for an empty slice.
pub fn neon_max_f64(a: &[f64]) -> f64 {
    if a.is_empty() {
        return f64::NEG_INFINITY;
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon_max_f64_impl(a) };
        }
    }
    fallback_max_f64(a)
}

// ============================================================
// NEON horizontal min for f32
// ============================================================

/// NEON-accelerated horizontal minimum of f32 elements.
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_min_f32_impl(a: &[f32]) -> f32 {
    let len = a.len();
    let mut i = 0;
    let mut acc = vdupq_n_f32(f32::INFINITY);

    while i + 4 <= len {
        let v = vld1q_f32(a.as_ptr().add(i));
        acc = vminq_f32(acc, v);
        i += 4;
    }

    let mut result = vminvq_f32(acc);

    while i < len {
        if a[i] < result {
            result = a[i];
        }
        i += 1;
    }

    result
}

/// Horizontal minimum of f32 elements.
///
/// Returns `f32::INFINITY` for an empty slice.
pub fn neon_min_f32(a: &[f32]) -> f32 {
    if a.is_empty() {
        return f32::INFINITY;
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon_min_f32_impl(a) };
        }
    }
    fallback_min_f32(a)
}

// ============================================================
// NEON horizontal min for f64
// ============================================================

/// NEON-accelerated horizontal minimum of f64 elements.
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_min_f64_impl(a: &[f64]) -> f64 {
    let len = a.len();
    let mut i = 0;
    let mut acc = vdupq_n_f64(f64::INFINITY);

    while i + 2 <= len {
        let v = vld1q_f64(a.as_ptr().add(i));
        acc = vminq_f64(acc, v);
        i += 2;
    }

    let lo = vgetq_lane_f64(acc, 0);
    let hi = vgetq_lane_f64(acc, 1);
    let mut result = if lo < hi { lo } else { hi };

    while i < len {
        if a[i] < result {
            result = a[i];
        }
        i += 1;
    }

    result
}

/// Horizontal minimum of f64 elements.
///
/// Returns `f64::INFINITY` for an empty slice.
pub fn neon_min_f64(a: &[f64]) -> f64 {
    if a.is_empty() {
        return f64::INFINITY;
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon_min_f64_impl(a) };
        }
    }
    fallback_min_f64(a)
}

// ============================================================
// Scalar fallback implementations
// ============================================================

fn fallback_sum_f32(a: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for &x in a {
        sum += x;
    }
    sum
}

fn fallback_sum_f64(a: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    for &x in a {
        sum += x;
    }
    sum
}

fn fallback_max_f32(a: &[f32]) -> f32 {
    let mut m = f32::NEG_INFINITY;
    for &x in a {
        if x > m {
            m = x;
        }
    }
    m
}

fn fallback_max_f64(a: &[f64]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for &x in a {
        if x > m {
            m = x;
        }
    }
    m
}

fn fallback_min_f32(a: &[f32]) -> f32 {
    let mut m = f32::INFINITY;
    for &x in a {
        if x < m {
            m = x;
        }
    }
    m
}

fn fallback_min_f64(a: &[f64]) -> f64 {
    let mut m = f64::INFINITY;
    for &x in a {
        if x < m {
            m = x;
        }
    }
    m
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------
    // neon_sum_f32
    // ----------------------------------------------------------

    #[test]
    fn test_sum_f32_basic() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let result = neon_sum_f32(&a);
        assert!((result - 10.0).abs() < 1e-5, "expected 10.0, got {result}");
    }

    #[test]
    fn test_sum_f32_non_multiple_of_4() {
        // 7 elements — exercises the scalar tail.
        let a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let result = neon_sum_f32(&a);
        assert!((result - 28.0).abs() < 1e-5, "expected 28.0, got {result}");
    }

    #[test]
    fn test_sum_f32_empty() {
        let a: Vec<f32> = vec![];
        let result = neon_sum_f32(&a);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_sum_f32_large() {
        // 1024 elements, each 1.0 → expected 1024.0
        let a = vec![1.0_f32; 1024];
        let result = neon_sum_f32(&a);
        assert!(
            (result - 1024.0).abs() < 1e-2,
            "expected 1024.0, got {result}"
        );
    }

    // ----------------------------------------------------------
    // neon_sum_f64
    // ----------------------------------------------------------

    #[test]
    fn test_sum_f64_basic() {
        let a = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let result = neon_sum_f64(&a);
        assert!((result - 15.0).abs() < 1e-10, "expected 15.0, got {result}");
    }

    #[test]
    fn test_sum_f64_empty() {
        let a: Vec<f64> = vec![];
        assert_eq!(neon_sum_f64(&a), 0.0);
    }

    // ----------------------------------------------------------
    // neon_max_f32
    // ----------------------------------------------------------

    #[test]
    fn test_max_f32_basic() {
        let a = vec![3.0_f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let result = neon_max_f32(&a);
        assert!((result - 9.0).abs() < 1e-6, "expected 9.0, got {result}");
    }

    #[test]
    fn test_max_f32_single() {
        let a = vec![42.0_f32];
        assert!((neon_max_f32(&a) - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_f32_negative() {
        let a = vec![-5.0_f32, -1.0, -3.0, -2.0];
        let result = neon_max_f32(&a);
        assert!(
            (result - (-1.0)).abs() < 1e-6,
            "expected -1.0, got {result}"
        );
    }

    #[test]
    fn test_max_f32_empty() {
        assert_eq!(neon_max_f32(&[]), f32::NEG_INFINITY);
    }

    // ----------------------------------------------------------
    // neon_max_f64
    // ----------------------------------------------------------

    #[test]
    fn test_max_f64_basic() {
        let a = vec![2.0_f64, 7.0, 1.0, 8.0, 2.0, 8.0];
        let result = neon_max_f64(&a);
        assert!((result - 8.0).abs() < 1e-10, "expected 8.0, got {result}");
    }

    #[test]
    fn test_max_f64_empty() {
        assert_eq!(neon_max_f64(&[]), f64::NEG_INFINITY);
    }

    // ----------------------------------------------------------
    // neon_min_f32
    // ----------------------------------------------------------

    #[test]
    fn test_min_f32_basic() {
        let a = vec![3.0_f32, 1.0, 4.0, 1.0, 5.0, 9.0];
        let result = neon_min_f32(&a);
        assert!((result - 1.0).abs() < 1e-6, "expected 1.0, got {result}");
    }

    #[test]
    fn test_min_f32_all_negative() {
        let a = vec![-10.0_f32, -5.0, -1.0, -20.0, -3.0];
        let result = neon_min_f32(&a);
        assert!(
            (result - (-20.0)).abs() < 1e-6,
            "expected -20.0, got {result}"
        );
    }

    #[test]
    fn test_min_f32_empty() {
        assert_eq!(neon_min_f32(&[]), f32::INFINITY);
    }

    // ----------------------------------------------------------
    // neon_min_f64
    // ----------------------------------------------------------

    #[test]
    fn test_min_f64_basic() {
        let a = vec![3.14_f64, 2.71, 1.41, 1.73];
        let result = neon_min_f64(&a);
        assert!((result - 1.41).abs() < 1e-6, "expected ~1.41, got {result}");
    }

    #[test]
    fn test_min_f64_empty() {
        assert_eq!(neon_min_f64(&[]), f64::INFINITY);
    }

    // ----------------------------------------------------------
    // Cross-validation: scalar fallback vs NEON path
    // ----------------------------------------------------------

    #[test]
    fn test_sum_f32_matches_scalar() {
        let a: Vec<f32> = (0..33).map(|i| i as f32 * 0.5).collect();
        let scalar = fallback_sum_f32(&a);
        let neon = neon_sum_f32(&a);
        // Tolerance accounts for different floating-point summation order.
        assert!(
            (scalar - neon).abs() <= scalar.abs() * 1e-5 + 1e-5,
            "scalar={scalar}, neon={neon}"
        );
    }

    #[test]
    fn test_max_f32_matches_scalar() {
        let a: Vec<f32> = (0..17).map(|i| (i as f32 - 8.0) * 1.5).collect();
        assert!(
            (fallback_max_f32(&a) - neon_max_f32(&a)).abs() < 1e-6,
            "scalar and neon max disagree"
        );
    }

    #[test]
    fn test_min_f32_matches_scalar() {
        let a: Vec<f32> = (0..17).map(|i| (i as f32 - 8.0) * 1.5).collect();
        assert!(
            (fallback_min_f32(&a) - neon_min_f32(&a)).abs() < 1e-6,
            "scalar and neon min disagree"
        );
    }
}
