//! Property-based tests for SIMD operations covering edge cases
//!
//! These tests use proptest to generate inputs covering:
//! - NaN propagation
//! - Infinity handling
//! - Zero and negative zero
//! - Subnormal numbers
//! - Large and small values
//! - Empty and single-element arrays
//! - Arrays with all identical values
//! - Overflow and underflow behavior

use ndarray::Array1;
use proptest::prelude::*;

use crate::simd::{
    simd_abs_f32, simd_abs_f64, simd_add_f32, simd_add_f64, simd_div_f32, simd_div_f64,
    simd_dot_f32, simd_dot_f64, simd_floor_f32, simd_floor_f64, simd_max_f32, simd_max_f64,
    simd_mean_f32, simd_mean_f64, simd_min_f32, simd_min_f64, simd_mul_f32, simd_mul_f64,
    simd_sqrt_f32, simd_sqrt_f64, simd_sub_f32, simd_sub_f64, simd_sum_f32, simd_sum_f64,
};

// ---------------------------------------------------------------------------
// Proptest strategies for generating interesting float values
// ---------------------------------------------------------------------------

/// Strategy that generates a wide range of f32 values including special cases.
/// We deliberately include subnormals, inf, -inf, NaN, zero, and negative zero.
fn interesting_f32() -> impl Strategy<Value = f32> {
    prop_oneof![
        // Normal finite values – wide dynamic range
        (-1e30f32..1e30f32),
        // Large magnitude
        Just(f32::MAX),
        Just(f32::MIN),
        // Smallest positive normal
        Just(f32::MIN_POSITIVE),
        // Subnormal values
        Just(f32::MIN_POSITIVE * 0.5),
        Just(f32::MIN_POSITIVE * 0.1),
        // Zeros
        Just(0.0f32),
        Just(-0.0f32),
        // Infinities
        Just(f32::INFINITY),
        Just(f32::NEG_INFINITY),
        // NaN
        Just(f32::NAN),
        // Edge around 1.0
        (-2.0f32..2.0f32),
        // Very small
        (f32::MIN_POSITIVE..1e-20f32),
    ]
}

/// Strategy that generates a wide range of f64 values including special cases.
fn interesting_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        // Normal finite values – wide dynamic range
        (-1e200f64..1e200f64),
        // Large magnitude
        Just(f64::MAX),
        Just(f64::MIN),
        // Smallest positive normal
        Just(f64::MIN_POSITIVE),
        // Subnormal values
        Just(f64::MIN_POSITIVE * 0.5),
        Just(f64::MIN_POSITIVE * 0.1),
        // Zeros
        Just(0.0f64),
        Just(-0.0f64),
        // Infinities
        Just(f64::INFINITY),
        Just(f64::NEG_INFINITY),
        // NaN
        Just(f64::NAN),
        // Edge around 1.0
        (-2.0f64..2.0f64),
        // Very small
        (f64::MIN_POSITIVE..1e-200f64),
    ]
}

/// Strategy for finite-only f32 values (useful for operations that must not produce NaN input).
fn finite_f32() -> impl Strategy<Value = f32> {
    prop_oneof![
        (-1e30f32..1e30f32),
        Just(f32::MAX),
        Just(f32::MIN),
        Just(f32::MIN_POSITIVE),
        Just(f32::MIN_POSITIVE * 0.5),
        Just(0.0f32),
        Just(-0.0f32),
        Just(f32::INFINITY),
        Just(f32::NEG_INFINITY),
    ]
}

/// Strategy for finite-only f64 values.
fn finite_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        (-1e200f64..1e200f64),
        Just(f64::MAX),
        Just(f64::MIN),
        Just(f64::MIN_POSITIVE),
        Just(f64::MIN_POSITIVE * 0.5),
        Just(0.0f64),
        Just(-0.0f64),
        Just(f64::INFINITY),
        Just(f64::NEG_INFINITY),
    ]
}

/// Strategy for non-negative f32 values (for sqrt tests).
fn non_negative_f32() -> impl Strategy<Value = f32> {
    prop_oneof![
        (0.0f32..1e30f32),
        Just(f32::MAX),
        Just(f32::MIN_POSITIVE),
        Just(f32::MIN_POSITIVE * 0.5),
        Just(0.0f32),
        Just(-0.0f32),
        Just(f32::INFINITY),
    ]
}

/// Strategy for non-negative f64 values (for sqrt tests).
fn non_negative_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        (0.0f64..1e200f64),
        Just(f64::MAX),
        Just(f64::MIN_POSITIVE),
        Just(f64::MIN_POSITIVE * 0.5),
        Just(0.0f64),
        Just(-0.0f64),
        Just(f64::INFINITY),
    ]
}

/// Strategy to generate a Vec<f32> with 0..=64 elements drawn from `interesting_f32`.
fn interesting_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(interesting_f32(), 0..=max_len)
}

/// Strategy to generate a Vec<f64> with 0..=64 elements drawn from `interesting_f64`.
fn interesting_f64_vec(max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(interesting_f64(), 0..=max_len)
}

/// Strategy to generate a Vec<f32> with 0..=64 finite f32 values.
fn finite_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(finite_f32(), 0..=max_len)
}

/// Strategy to generate a Vec<f64> with 0..=64 finite f64 values.
fn finite_f64_vec(max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(finite_f64(), 0..=max_len)
}

/// Strategy for f32 values that are strictly bounded (no Inf, no NaN, no MAX/MIN).
///
/// For sum and dot reduction comparisons, even `f32::MAX` and `f32::MIN` are
/// problematic: a SIMD lane may pair (`MAX`, `-MAX`) and reduce them to 0.0
/// while scalar left-to-right summation reaches a different intermediate value.
/// Restricting to ±1e30 avoids all such catastrophic-cancellation edge cases
/// while still covering subnormals, zeros, and a wide dynamic range.
fn bounded_f32() -> impl Strategy<Value = f32> {
    prop_oneof![
        (-1e30f32..1e30f32),
        Just(f32::MIN_POSITIVE),
        Just(f32::MIN_POSITIVE * 0.5),
        Just(0.0f32),
        Just(-0.0f32),
    ]
}

/// Strategy for f64 values that are strictly bounded (no Inf, no NaN, no MAX/MIN).
fn bounded_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        (-1e200f64..1e200f64),
        Just(f64::MIN_POSITIVE),
        Just(f64::MIN_POSITIVE * 0.5),
        Just(0.0f64),
        Just(-0.0f64),
    ]
}

fn bounded_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(bounded_f32(), 0..=max_len)
}

fn bounded_f64_vec(max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(bounded_f64(), 0..=max_len)
}

/// Strategy for f32 dot-product inputs: bounded so that element-wise products
/// `a_i * b_i` do not overflow (|a_i| <= 1e15 => |a_i * b_i| <= 1e30 << f32::MAX).
fn dot_bounded_f32() -> impl Strategy<Value = f32> {
    prop_oneof![
        (-1e15f32..1e15f32),
        Just(f32::MIN_POSITIVE),
        Just(f32::MIN_POSITIVE * 0.5),
        Just(0.0f32),
        Just(-0.0f32),
    ]
}

/// Strategy for f64 dot-product inputs: bounded so that element-wise products
/// do not overflow (|a_i| <= 1e150 => |a_i * b_i| <= 1e300 << f64::MAX).
fn dot_bounded_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        (-1e150f64..1e150f64),
        Just(f64::MIN_POSITIVE),
        Just(f64::MIN_POSITIVE * 0.5),
        Just(0.0f64),
        Just(-0.0f64),
    ]
}

fn dot_bounded_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(dot_bounded_f32(), 0..=max_len)
}

fn dot_bounded_f64_vec(max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(dot_bounded_f64(), 0..=max_len)
}

// ---------------------------------------------------------------------------
// Helper: scalar reference implementations
// ---------------------------------------------------------------------------

fn scalar_add_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn scalar_add_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn scalar_sub_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn scalar_sub_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn scalar_mul_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

fn scalar_mul_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

fn scalar_div_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x / y).collect()
}

fn scalar_div_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x / y).collect()
}

fn scalar_abs_f32(a: &[f32]) -> Vec<f32> {
    a.iter().map(|x| x.abs()).collect()
}

fn scalar_abs_f64(a: &[f64]) -> Vec<f64> {
    a.iter().map(|x| x.abs()).collect()
}

fn scalar_sqrt_f32(a: &[f32]) -> Vec<f32> {
    a.iter().map(|x| x.sqrt()).collect()
}

fn scalar_sqrt_f64(a: &[f64]) -> Vec<f64> {
    a.iter().map(|x| x.sqrt()).collect()
}

fn scalar_sum_f32(a: &[f32]) -> f32 {
    a.iter().copied().sum()
}

fn scalar_sum_f64(a: &[f64]) -> f64 {
    a.iter().copied().sum()
}

fn scalar_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn scalar_dot_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn scalar_min_f32(a: &[f32]) -> f32 {
    a.iter()
        .copied()
        .fold(f32::INFINITY, |acc, x| acc.min(x))
}

fn scalar_min_f64(a: &[f64]) -> f64 {
    a.iter()
        .copied()
        .fold(f64::INFINITY, |acc, x| acc.min(x))
}

fn scalar_max_f32(a: &[f32]) -> f32 {
    a.iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, x| acc.max(x))
}

fn scalar_max_f64(a: &[f64]) -> f64 {
    a.iter()
        .copied()
        .fold(f64::NEG_INFINITY, |acc, x| acc.max(x))
}

fn scalar_floor_f32(a: &[f32]) -> Vec<f32> {
    a.iter().map(|x| x.floor()).collect()
}

fn scalar_floor_f64(a: &[f64]) -> Vec<f64> {
    a.iter().map(|x| x.floor()).collect()
}

// ---------------------------------------------------------------------------
// Comparison helpers that handle NaN and IEEE edge cases correctly
// ---------------------------------------------------------------------------

/// Compare two f32 values: both NaN counts as equal; otherwise bitwise-equal or
/// within a relative/absolute epsilon.
fn f32_equivalent(a: f32, b: f32) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a == b {
        return true;
    }
    // Relative tolerance
    let diff = (a - b).abs();
    let magnitude = a.abs().max(b.abs());
    if magnitude == 0.0 {
        diff < 1e-6
    } else {
        diff / magnitude < 1e-4
    }
}

/// Compare two f64 values: both NaN counts as equal; otherwise within tolerance.
fn f64_equivalent(a: f64, b: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a == b {
        return true;
    }
    let diff = (a - b).abs();
    let magnitude = a.abs().max(b.abs());
    if magnitude == 0.0 {
        diff < 1e-10
    } else {
        diff / magnitude < 1e-10
    }
}

/// Compare two f32 sum/reduction results with an absolute tolerance derived
/// from the input slice.
///
/// When large magnitudes nearly cancel, catastrophic cancellation is expected:
/// both SIMD and scalar paths produce results correct to within the floating-point
/// round-off budget, but the order of accumulation differs.  The acceptable
/// absolute error is `n * sum(|input|) * ε` where ε = f32::EPSILON.
fn f32_sum_equivalent(simd: f32, scalar: f32, inputs: &[f32]) -> bool {
    if simd.is_nan() && scalar.is_nan() {
        return true;
    }
    if simd == scalar {
        return true;
    }
    if simd.is_infinite() || scalar.is_infinite() {
        // Both infinite with same sign is fine; different signs or finite vs inf is not.
        return simd.is_infinite() && scalar.is_infinite() && simd.signum() == scalar.signum();
    }
    let abs_input_sum: f32 = inputs.iter().map(|x| x.abs()).sum();
    let n = inputs.len() as f32;
    // Allow n * |inputs| * ε as absolute tolerance; minimum of 1e-10 to
    // handle the all-zeros case.
    let tol = (n * abs_input_sum * f32::EPSILON).max(1e-10_f32);
    (simd - scalar).abs() <= tol
}

/// Compare two f64 sum/reduction results with an absolute tolerance derived
/// from the input slice.
fn f64_sum_equivalent(simd: f64, scalar: f64, inputs: &[f64]) -> bool {
    if simd.is_nan() && scalar.is_nan() {
        return true;
    }
    if simd == scalar {
        return true;
    }
    if simd.is_infinite() || scalar.is_infinite() {
        return simd.is_infinite() && scalar.is_infinite() && simd.signum() == scalar.signum();
    }
    let abs_input_sum: f64 = inputs.iter().map(|x| x.abs()).sum();
    let n = inputs.len() as f64;
    let tol = (n * abs_input_sum * f64::EPSILON).max(1e-20_f64);
    (simd - scalar).abs() <= tol
}

/// Compare two f32 dot-product results with an absolute tolerance derived
/// from both input slices.
fn f32_dot_equivalent(simd: f32, scalar: f32, a: &[f32], b: &[f32]) -> bool {
    if simd.is_nan() && scalar.is_nan() {
        return true;
    }
    if simd == scalar {
        return true;
    }
    if simd.is_infinite() || scalar.is_infinite() {
        return simd.is_infinite() && scalar.is_infinite() && simd.signum() == scalar.signum();
    }
    // Absolute tolerance: n * sum(|a_i * b_i|) * ε
    let abs_product_sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x * y).abs()).sum();
    let n = a.len() as f32;
    let tol = (n * abs_product_sum * f32::EPSILON).max(1e-10_f32);
    (simd - scalar).abs() <= tol
}

/// Compare two f64 dot-product results with an absolute tolerance derived
/// from both input slices.
fn f64_dot_equivalent(simd: f64, scalar: f64, a: &[f64], b: &[f64]) -> bool {
    if simd.is_nan() && scalar.is_nan() {
        return true;
    }
    if simd == scalar {
        return true;
    }
    if simd.is_infinite() || scalar.is_infinite() {
        return simd.is_infinite() && scalar.is_infinite() && simd.signum() == scalar.signum();
    }
    let abs_product_sum: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x * y).abs()).sum();
    let n = a.len() as f64;
    let tol = (n * abs_product_sum * f64::EPSILON).max(1e-20_f64);
    (simd - scalar).abs() <= tol
}

/// Compare element-wise, treating NaN==NaN as equal.
fn vec_f32_equivalent(a: &[f32], b: &[f32]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| f32_equivalent(*x, *y))
}

fn vec_f64_equivalent(a: &[f64], b: &[f64]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| f64_equivalent(*x, *y))
}

// ---------------------------------------------------------------------------
// proptest! blocks
// ---------------------------------------------------------------------------

proptest! {
    // -----------------------------------------------------------------------
    // simd_add_f32 / simd_add_f64 – element-wise addition
    // -----------------------------------------------------------------------

    #[test]
    fn prop_add_f32_matches_scalar(a in interesting_f32_vec(64), b in interesting_f32_vec(64)) {
        // Make equal-length inputs
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];

        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let simd_result = simd_add_f32(&av.view(), &bv.view());
        let scalar_result = scalar_add_f32(a, b);

        prop_assert!(
            vec_f32_equivalent(simd_result.as_slice().expect("slice"), &scalar_result),
            "SIMD add f32 mismatch: simd={:?}, scalar={:?}",
            simd_result,
            scalar_result
        );
    }

    #[test]
    fn prop_add_f64_matches_scalar(a in interesting_f64_vec(64), b in interesting_f64_vec(64)) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];

        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let simd_result = simd_add_f64(&av.view(), &bv.view());
        let scalar_result = scalar_add_f64(a, b);

        prop_assert!(
            vec_f64_equivalent(simd_result.as_slice().expect("slice"), &scalar_result),
            "SIMD add f64 mismatch"
        );
    }

    // -----------------------------------------------------------------------
    // simd_sub_f32 / simd_sub_f64
    // -----------------------------------------------------------------------

    #[test]
    fn prop_sub_f32_matches_scalar(a in interesting_f32_vec(64), b in interesting_f32_vec(64)) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];

        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let simd_result = simd_sub_f32(&av.view(), &bv.view());
        let scalar_result = scalar_sub_f32(a, b);

        prop_assert!(
            vec_f32_equivalent(simd_result.as_slice().expect("slice"), &scalar_result),
            "SIMD sub f32 mismatch"
        );
    }

    #[test]
    fn prop_sub_f64_matches_scalar(a in interesting_f64_vec(64), b in interesting_f64_vec(64)) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];

        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let simd_result = simd_sub_f64(&av.view(), &bv.view());
        let scalar_result = scalar_sub_f64(a, b);

        prop_assert!(
            vec_f64_equivalent(simd_result.as_slice().expect("slice"), &scalar_result),
            "SIMD sub f64 mismatch"
        );
    }

    // -----------------------------------------------------------------------
    // simd_mul_f32 / simd_mul_f64
    // -----------------------------------------------------------------------

    #[test]
    fn prop_mul_f32_matches_scalar(a in interesting_f32_vec(64), b in interesting_f32_vec(64)) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];

        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let simd_result = simd_mul_f32(&av.view(), &bv.view());
        let scalar_result = scalar_mul_f32(a, b);

        prop_assert!(
            vec_f32_equivalent(simd_result.as_slice().expect("slice"), &scalar_result),
            "SIMD mul f32 mismatch"
        );
    }

    #[test]
    fn prop_mul_f64_matches_scalar(a in interesting_f64_vec(64), b in interesting_f64_vec(64)) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];

        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let simd_result = simd_mul_f64(&av.view(), &bv.view());
        let scalar_result = scalar_mul_f64(a, b);

        prop_assert!(
            vec_f64_equivalent(simd_result.as_slice().expect("slice"), &scalar_result),
            "SIMD mul f64 mismatch"
        );
    }

    // -----------------------------------------------------------------------
    // simd_div_f32 / simd_div_f64
    // -----------------------------------------------------------------------

    #[test]
    fn prop_div_f32_matches_scalar(a in interesting_f32_vec(64), b in interesting_f32_vec(64)) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];

        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let simd_result = simd_div_f32(&av.view(), &bv.view());
        let scalar_result = scalar_div_f32(a, b);

        prop_assert!(
            vec_f32_equivalent(simd_result.as_slice().expect("slice"), &scalar_result),
            "SIMD div f32 mismatch"
        );
    }

    #[test]
    fn prop_div_f64_matches_scalar(a in interesting_f64_vec(64), b in interesting_f64_vec(64)) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];

        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let simd_result = simd_div_f64(&av.view(), &bv.view());
        let scalar_result = scalar_div_f64(a, b);

        prop_assert!(
            vec_f64_equivalent(simd_result.as_slice().expect("slice"), &scalar_result),
            "SIMD div f64 mismatch"
        );
    }

    // -----------------------------------------------------------------------
    // simd_abs_f32 / simd_abs_f64 – unary absolute value
    // -----------------------------------------------------------------------

    #[test]
    fn prop_abs_f32_matches_scalar(a in interesting_f32_vec(64)) {
        let av = Array1::from_vec(a.clone());
        let simd_result = simd_abs_f32(&av.view());
        let scalar_result = scalar_abs_f32(&a);

        prop_assert!(
            vec_f32_equivalent(simd_result.as_slice().expect("slice"), &scalar_result),
            "SIMD abs f32 mismatch"
        );
    }

    #[test]
    fn prop_abs_f64_matches_scalar(a in interesting_f64_vec(64)) {
        let av = Array1::from_vec(a.clone());
        let simd_result = simd_abs_f64(&av.view());
        let scalar_result = scalar_abs_f64(&a);

        prop_assert!(
            vec_f64_equivalent(simd_result.as_slice().expect("slice"), &scalar_result),
            "SIMD abs f64 mismatch"
        );
    }

    // -----------------------------------------------------------------------
    // abs is always non-negative (for finite inputs)
    // -----------------------------------------------------------------------

    #[test]
    fn prop_abs_f32_non_negative(a in interesting_f32_vec(64)) {
        let av = Array1::from_vec(a.clone());
        let result = simd_abs_f32(&av.view());
        for &v in result.iter() {
            // NaN is ok to propagate; non-NaN must be non-negative
            if !v.is_nan() {
                prop_assert!(v >= 0.0, "abs result was negative: {}", v);
            }
        }
    }

    #[test]
    fn prop_abs_f64_non_negative(a in interesting_f64_vec(64)) {
        let av = Array1::from_vec(a.clone());
        let result = simd_abs_f64(&av.view());
        for &v in result.iter() {
            if !v.is_nan() {
                prop_assert!(v >= 0.0, "abs result was negative: {}", v);
            }
        }
    }

    // -----------------------------------------------------------------------
    // simd_sqrt_f32 / simd_sqrt_f64 – unary square root
    // -----------------------------------------------------------------------

    #[test]
    fn prop_sqrt_f32_matches_scalar(a in non_negative_f32_vec(64)) {
        let av = Array1::from_vec(a.clone());
        let simd_result = simd_sqrt_f32(&av.view());
        let scalar_result = scalar_sqrt_f32(&a);

        prop_assert!(
            vec_f32_equivalent(simd_result.as_slice().expect("slice"), &scalar_result),
            "SIMD sqrt f32 mismatch"
        );
    }

    #[test]
    fn prop_sqrt_f64_matches_scalar(a in non_negative_f64_vec(64)) {
        let av = Array1::from_vec(a.clone());
        let simd_result = simd_sqrt_f64(&av.view());
        let scalar_result = scalar_sqrt_f64(&a);

        prop_assert!(
            vec_f64_equivalent(simd_result.as_slice().expect("slice"), &scalar_result),
            "SIMD sqrt f64 mismatch"
        );
    }

    // -----------------------------------------------------------------------
    // sqrt(x)^2 ≈ x for non-negative finite values
    // -----------------------------------------------------------------------

    #[test]
    fn prop_sqrt_f32_squared_recovers_input(a in finite_positive_f32_vec(32)) {
        let av = Array1::from_vec(a.clone());
        let sqrt_result = simd_sqrt_f32(&av.view());
        // Re-square
        let squared: Vec<f32> = sqrt_result.iter().map(|x| x * x).collect();
        for (orig, sq) in a.iter().zip(squared.iter()) {
            if orig.is_nan() || orig.is_infinite() {
                continue;
            }
            let rel_err = if *orig == 0.0 {
                sq.abs()
            } else {
                ((sq - orig) / orig).abs()
            };
            prop_assert!(
                rel_err < 1e-4,
                "sqrt(x)^2 != x: x={}, sqrt={}, sq={}, rel_err={}",
                orig, orig.sqrt(), sq, rel_err
            );
        }
    }

    #[test]
    fn prop_sqrt_f64_squared_recovers_input(a in finite_positive_f64_vec(32)) {
        let av = Array1::from_vec(a.clone());
        let sqrt_result = simd_sqrt_f64(&av.view());
        let squared: Vec<f64> = sqrt_result.iter().map(|x| x * x).collect();
        for (orig, sq) in a.iter().zip(squared.iter()) {
            if orig.is_nan() || orig.is_infinite() {
                continue;
            }
            let rel_err = if *orig == 0.0 {
                sq.abs()
            } else {
                ((sq - orig) / orig).abs()
            };
            prop_assert!(
                rel_err < 1e-10,
                "sqrt(x)^2 != x: x={}, sq={}, rel_err={}",
                orig, sq, rel_err
            );
        }
    }

    // -----------------------------------------------------------------------
    // simd_sum_f32 / simd_sum_f64 – reductions
    // -----------------------------------------------------------------------

    #[test]
    fn prop_sum_f32_empty_is_zero(dummy in Just(())) {
        let _ = dummy;
        let empty: Vec<f32> = vec![];
        let av = Array1::from_vec(empty);
        let result = simd_sum_f32(&av.view());
        prop_assert_eq!(result, 0.0f32);
    }

    #[test]
    fn prop_sum_f64_empty_is_zero(dummy in Just(())) {
        let _ = dummy;
        let empty: Vec<f64> = vec![];
        let av = Array1::from_vec(empty);
        let result = simd_sum_f64(&av.view());
        prop_assert_eq!(result, 0.0f64);
    }

    #[test]
    fn prop_sum_f32_single_element(v in interesting_f32()) {
        let av = Array1::from_vec(vec![v]);
        let result = simd_sum_f32(&av.view());
        prop_assert!(
            f32_equivalent(result, v),
            "sum of single element should be the element itself: got {}, expected {}",
            result, v
        );
    }

    #[test]
    fn prop_sum_f64_single_element(v in interesting_f64()) {
        let av = Array1::from_vec(vec![v]);
        let result = simd_sum_f64(&av.view());
        prop_assert!(
            f64_equivalent(result, v),
            "sum of single element should be the element itself: got {}, expected {}",
            result, v
        );
    }

    #[test]
    fn prop_sum_f32_matches_scalar(a in bounded_f32_vec(64)) {
        // Uses bounded_f32_vec (no Inf/NaN, no MAX/MIN) so 64-element sums cannot
        // overflow.  Uses f32_sum_equivalent which allows absolute error proportional
        // to n * sum(|inputs|) * epsilon, accounting for different accumulation order
        // (catastrophic cancellation) between SIMD and scalar paths.
        let av = Array1::from_vec(a.clone());
        let simd_result = simd_sum_f32(&av.view());
        let scalar_result = scalar_sum_f32(&a);
        prop_assert!(
            f32_sum_equivalent(simd_result, scalar_result, &a),
            "SIMD sum f32 mismatch: simd={}, scalar={}",
            simd_result, scalar_result
        );
    }

    #[test]
    fn prop_sum_f64_matches_scalar(a in bounded_f64_vec(64)) {
        // Uses bounded_f64_vec and f64_sum_equivalent for the same reasons.
        let av = Array1::from_vec(a.clone());
        let simd_result = simd_sum_f64(&av.view());
        let scalar_result = scalar_sum_f64(&a);
        prop_assert!(
            f64_sum_equivalent(simd_result, scalar_result, &a),
            "SIMD sum f64 mismatch: simd={}, scalar={}",
            simd_result, scalar_result
        );
    }

    // -----------------------------------------------------------------------
    // NaN propagation: any NaN input to sum should produce NaN output
    // -----------------------------------------------------------------------

    #[test]
    fn prop_sum_f32_nan_propagates(prefix in prop::collection::vec(finite_f32(), 0..32)) {
        let mut data = prefix;
        data.push(f32::NAN);
        let av = Array1::from_vec(data);
        let result = simd_sum_f32(&av.view());
        prop_assert!(result.is_nan(), "NaN in input should propagate through sum");
    }

    #[test]
    fn prop_sum_f64_nan_propagates(prefix in prop::collection::vec(finite_f64(), 0..32)) {
        let mut data = prefix;
        data.push(f64::NAN);
        let av = Array1::from_vec(data);
        let result = simd_sum_f64(&av.view());
        prop_assert!(result.is_nan(), "NaN in input should propagate through sum");
    }

    // -----------------------------------------------------------------------
    // simd_dot_f32 / simd_dot_f64 – dot product
    // -----------------------------------------------------------------------

    #[test]
    fn prop_dot_f32_matches_scalar(a in dot_bounded_f32_vec(64), b in dot_bounded_f32_vec(64)) {
        // Uses dot_bounded_f32_vec: elements bounded to ±1e15 so that element-wise
        // products (up to 1e30) and their 64-element sum (up to 6.4e31) cannot
        // overflow f32::MAX (~3.4e38).  Uses f32_dot_equivalent which allows absolute
        // error proportional to n * sum(|a_i * b_i|) * epsilon.
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];

        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let simd_result = simd_dot_f32(&av.view(), &bv.view());
        let scalar_result = scalar_dot_f32(a, b);

        prop_assert!(
            f32_dot_equivalent(simd_result, scalar_result, a, b),
            "SIMD dot f32 mismatch: simd={}, scalar={}",
            simd_result, scalar_result
        );
    }

    #[test]
    fn prop_dot_f64_matches_scalar(a in dot_bounded_f64_vec(64), b in dot_bounded_f64_vec(64)) {
        // Uses dot_bounded_f64_vec: elements bounded to ±1e150 so products (up to
        // 1e300) and their 64-element sum (up to 6.4e301) are below f64::MAX (~1.8e308).
        // Uses f64_dot_equivalent for the input-scaled tolerance.
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];

        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let simd_result = simd_dot_f64(&av.view(), &bv.view());
        let scalar_result = scalar_dot_f64(a, b);

        prop_assert!(
            f64_dot_equivalent(simd_result, scalar_result, a, b),
            "SIMD dot f64 mismatch: simd={}, scalar={}",
            simd_result, scalar_result
        );
    }

    #[test]
    fn prop_dot_f32_empty_is_zero(dummy in Just(())) {
        let _ = dummy;
        let av = Array1::<f32>::zeros(0);
        let bv = Array1::<f32>::zeros(0);
        let result = simd_dot_f32(&av.view(), &bv.view());
        prop_assert_eq!(result, 0.0f32);
    }

    #[test]
    fn prop_dot_f64_empty_is_zero(dummy in Just(())) {
        let _ = dummy;
        let av = Array1::<f64>::zeros(0);
        let bv = Array1::<f64>::zeros(0);
        let result = simd_dot_f64(&av.view(), &bv.view());
        prop_assert_eq!(result, 0.0f64);
    }

    // -----------------------------------------------------------------------
    // simd_min_f32 / simd_min_f64 – reductions
    // -----------------------------------------------------------------------

    #[test]
    fn prop_min_f32_matches_scalar(a in finite_f32_vec(64)) {
        if a.is_empty() {
            // simd_min on empty: skip (undefined behavior territory)
            return Ok(());
        }
        let av = Array1::from_vec(a.clone());
        let simd_result = simd_min_f32(&av.view());
        let scalar_result = scalar_min_f32(&a);

        prop_assert!(
            f32_equivalent(simd_result, scalar_result),
            "SIMD min f32 mismatch: simd={}, scalar={}",
            simd_result, scalar_result
        );
    }

    #[test]
    fn prop_min_f64_matches_scalar(a in finite_f64_vec(64)) {
        if a.is_empty() {
            return Ok(());
        }
        let av = Array1::from_vec(a.clone());
        let simd_result = simd_min_f64(&av.view());
        let scalar_result = scalar_min_f64(&a);

        prop_assert!(
            f64_equivalent(simd_result, scalar_result),
            "SIMD min f64 mismatch: simd={}, scalar={}",
            simd_result, scalar_result
        );
    }

    // -----------------------------------------------------------------------
    // simd_max_f32 / simd_max_f64 – reductions
    // -----------------------------------------------------------------------

    #[test]
    fn prop_max_f32_matches_scalar(a in finite_f32_vec(64)) {
        if a.is_empty() {
            return Ok(());
        }
        let av = Array1::from_vec(a.clone());
        let simd_result = simd_max_f32(&av.view());
        let scalar_result = scalar_max_f32(&a);

        prop_assert!(
            f32_equivalent(simd_result, scalar_result),
            "SIMD max f32 mismatch: simd={}, scalar={}",
            simd_result, scalar_result
        );
    }

    #[test]
    fn prop_max_f64_matches_scalar(a in finite_f64_vec(64)) {
        if a.is_empty() {
            return Ok(());
        }
        let av = Array1::from_vec(a.clone());
        let simd_result = simd_max_f64(&av.view());
        let scalar_result = scalar_max_f64(&a);

        prop_assert!(
            f64_equivalent(simd_result, scalar_result),
            "SIMD max f64 mismatch: simd={}, scalar={}",
            simd_result, scalar_result
        );
    }

    // -----------------------------------------------------------------------
    // min <= max invariant for non-empty arrays
    // -----------------------------------------------------------------------

    #[test]
    fn prop_min_leq_max_f32(a in prop::collection::vec(-1e30f32..1e30f32, 1..64)) {
        let av = Array1::from_vec(a.clone());
        let min = simd_min_f32(&av.view());
        let max = simd_max_f32(&av.view());
        prop_assert!(min <= max, "min={} > max={}", min, max);
    }

    #[test]
    fn prop_min_leq_max_f64(a in prop::collection::vec(-1e200f64..1e200f64, 1..64)) {
        let av = Array1::from_vec(a.clone());
        let min = simd_min_f64(&av.view());
        let max = simd_max_f64(&av.view());
        prop_assert!(min <= max, "min={} > max={}", min, max);
    }

    // -----------------------------------------------------------------------
    // simd_mean_f32 / simd_mean_f64 – mean is in [min, max]
    // -----------------------------------------------------------------------

    #[test]
    fn prop_mean_f32_in_range(a in prop::collection::vec(-1e10f32..1e10f32, 1..64)) {
        let av = Array1::from_vec(a.clone());
        let mean = simd_mean_f32(&av.view());
        let min = simd_min_f32(&av.view());
        let max = simd_max_f32(&av.view());
        // Allow small floating-point slack
        let slack = (max - min).abs() * 1e-4 + 1e-6;
        prop_assert!(
            mean >= min - slack && mean <= max + slack,
            "mean={} not in [{}, {}]",
            mean, min, max
        );
    }

    #[test]
    fn prop_mean_f64_in_range(a in prop::collection::vec(-1e100f64..1e100f64, 1..64)) {
        let av = Array1::from_vec(a.clone());
        let mean = simd_mean_f64(&av.view());
        let min = simd_min_f64(&av.view());
        let max = simd_max_f64(&av.view());
        let slack = (max - min).abs() * 1e-10 + 1e-12;
        prop_assert!(
            mean >= min - slack && mean <= max + slack,
            "mean={} not in [{}, {}]",
            mean, min, max
        );
    }

    // -----------------------------------------------------------------------
    // All-same-value arrays: sum, mean, min, max should be consistent
    // -----------------------------------------------------------------------

    #[test]
    fn prop_all_same_f32(v in (-1e10f32..1e10f32), n in 1usize..64) {
        let data: Vec<f32> = vec![v; n];
        let av = Array1::from_vec(data.clone());

        let sum = simd_sum_f32(&av.view());
        let mean = simd_mean_f32(&av.view());
        let min = simd_min_f32(&av.view());
        let max = simd_max_f32(&av.view());

        let expected_sum = v * n as f32;
        prop_assert!(
            f32_equivalent(sum, expected_sum),
            "all-same sum: got {}, expected {}",
            sum, expected_sum
        );
        prop_assert!(
            f32_equivalent(mean, v),
            "all-same mean: got {}, expected {}",
            mean, v
        );
        prop_assert!(
            f32_equivalent(min, v),
            "all-same min: got {}, expected {}",
            min, v
        );
        prop_assert!(
            f32_equivalent(max, v),
            "all-same max: got {}, expected {}",
            max, v
        );
    }

    #[test]
    fn prop_all_same_f64(v in (-1e100f64..1e100f64), n in 1usize..64) {
        let data: Vec<f64> = vec![v; n];
        let av = Array1::from_vec(data.clone());

        let sum = simd_sum_f64(&av.view());
        let mean = simd_mean_f64(&av.view());
        let min = simd_min_f64(&av.view());
        let max = simd_max_f64(&av.view());

        let expected_sum = v * n as f64;
        prop_assert!(
            f64_equivalent(sum, expected_sum),
            "all-same sum: got {}, expected {}",
            sum, expected_sum
        );
        prop_assert!(
            f64_equivalent(mean, v),
            "all-same mean: got {}, expected {}",
            mean, v
        );
        prop_assert!(
            f64_equivalent(min, v),
            "all-same min: got {}, expected {}",
            min, v
        );
        prop_assert!(
            f64_equivalent(max, v),
            "all-same max: got {}, expected {}",
            max, v
        );
    }

    // -----------------------------------------------------------------------
    // Infinity handling: adding +Inf to any finite value -> +Inf
    // -----------------------------------------------------------------------

    #[test]
    fn prop_add_f32_inf_propagates(a in finite_f32_vec(32)) {
        if a.is_empty() {
            return Ok(());
        }
        let mut b: Vec<f32> = vec![f32::INFINITY; a.len()];
        // Mix in some finite values: first half inf, second half normal
        for i in (a.len() / 2)..a.len() {
            b[i] = 1.0;
        }
        let av = Array1::from_vec(a.clone());
        let bv = Array1::from_vec(b.clone());
        let result = simd_add_f32(&av.view(), &bv.view());
        let scalar = scalar_add_f32(&a, &b);
        prop_assert!(
            vec_f32_equivalent(result.as_slice().expect("slice"), &scalar),
            "add with Inf mismatch"
        );
    }

    // -----------------------------------------------------------------------
    // simd_floor_f32 / simd_floor_f64 – rounding
    // -----------------------------------------------------------------------

    #[test]
    fn prop_floor_f32_matches_scalar(a in interesting_f32_vec(64)) {
        let av = Array1::from_vec(a.clone());
        let simd_result = simd_floor_f32(&av.view());
        let scalar_result = scalar_floor_f32(&a);

        prop_assert!(
            vec_f32_equivalent(simd_result.as_slice().expect("slice"), &scalar_result),
            "SIMD floor f32 mismatch"
        );
    }

    #[test]
    fn prop_floor_f64_matches_scalar(a in interesting_f64_vec(64)) {
        let av = Array1::from_vec(a.clone());
        let simd_result = simd_floor_f64(&av.view());
        let scalar_result = scalar_floor_f64(&a);

        prop_assert!(
            vec_f64_equivalent(simd_result.as_slice().expect("slice"), &scalar_result),
            "SIMD floor f64 mismatch"
        );
    }

    // floor(x) <= x for all finite x
    #[test]
    fn prop_floor_f32_leq_input(a in prop::collection::vec(-1e10f32..1e10f32, 0..64)) {
        let av = Array1::from_vec(a.clone());
        let floored = simd_floor_f32(&av.view());
        for (orig, fl) in a.iter().zip(floored.iter()) {
            prop_assert!(
                *fl <= *orig,
                "floor({}) = {} violates floor(x) <= x",
                orig, fl
            );
        }
    }

    #[test]
    fn prop_floor_f64_leq_input(a in prop::collection::vec(-1e100f64..1e100f64, 0..64)) {
        let av = Array1::from_vec(a.clone());
        let floored = simd_floor_f64(&av.view());
        for (orig, fl) in a.iter().zip(floored.iter()) {
            prop_assert!(
                *fl <= *orig,
                "floor({}) = {} violates floor(x) <= x",
                orig, fl
            );
        }
    }

    // floor(x) is always an integer (difference is < 1)
    #[test]
    fn prop_floor_f32_is_integer(a in prop::collection::vec(-1e6f32..1e6f32, 0..64)) {
        let av = Array1::from_vec(a.clone());
        let floored = simd_floor_f32(&av.view());
        for fl in floored.iter() {
            let frac = fl - fl.floor();
            prop_assert!(
                frac.abs() < 1e-5,
                "floor result {} is not an integer",
                fl
            );
        }
    }

    // -----------------------------------------------------------------------
    // Subnormal number handling: abs of subnormals
    // -----------------------------------------------------------------------

    #[test]
    fn prop_abs_subnormal_f32(dummy in Just(())) {
        let _ = dummy;
        // f32::MIN_POSITIVE * 0.5 is a subnormal
        let subnormal = f32::MIN_POSITIVE * 0.5;
        let data = vec![subnormal, -subnormal, 0.0f32, -0.0f32];
        let av = Array1::from_vec(data.clone());
        let result = simd_abs_f32(&av.view());
        let expected = scalar_abs_f32(&data);
        prop_assert!(
            vec_f32_equivalent(result.as_slice().expect("slice"), &expected),
            "abs of subnormals mismatch: {:?} vs {:?}",
            result, expected
        );
    }

    #[test]
    fn prop_abs_subnormal_f64(dummy in Just(())) {
        let _ = dummy;
        let subnormal = f64::MIN_POSITIVE * 0.5;
        let data = vec![subnormal, -subnormal, 0.0f64, -0.0f64];
        let av = Array1::from_vec(data.clone());
        let result = simd_abs_f64(&av.view());
        let expected = scalar_abs_f64(&data);
        prop_assert!(
            vec_f64_equivalent(result.as_slice().expect("slice"), &expected),
            "abs of subnormals (f64) mismatch"
        );
    }

    // -----------------------------------------------------------------------
    // Negative-zero: abs(-0.0) == 0.0
    // -----------------------------------------------------------------------

    #[test]
    fn prop_abs_negative_zero_f32(dummy in Just(())) {
        let _ = dummy;
        let data = vec![-0.0f32];
        let av = Array1::from_vec(data);
        let result = simd_abs_f32(&av.view());
        prop_assert_eq!(result[0], 0.0f32);
    }

    #[test]
    fn prop_abs_negative_zero_f64(dummy in Just(())) {
        let _ = dummy;
        let data = vec![-0.0f64];
        let av = Array1::from_vec(data);
        let result = simd_abs_f64(&av.view());
        prop_assert_eq!(result[0], 0.0f64);
    }

    // -----------------------------------------------------------------------
    // Large value overflow: MAX + MAX = Inf
    // -----------------------------------------------------------------------

    #[test]
    fn prop_add_f32_overflow(dummy in Just(())) {
        let _ = dummy;
        let data_a = vec![f32::MAX];
        let data_b = vec![f32::MAX];
        let av = Array1::from_vec(data_a);
        let bv = Array1::from_vec(data_b);
        let result = simd_add_f32(&av.view(), &bv.view());
        prop_assert!(
            result[0].is_infinite(),
            "MAX + MAX should be infinite, got {}",
            result[0]
        );
    }

    #[test]
    fn prop_add_f64_overflow(dummy in Just(())) {
        let _ = dummy;
        let data_a = vec![f64::MAX];
        let data_b = vec![f64::MAX];
        let av = Array1::from_vec(data_a);
        let bv = Array1::from_vec(data_b);
        let result = simd_add_f64(&av.view(), &bv.view());
        prop_assert!(
            result[0].is_infinite(),
            "MAX + MAX should be infinite (f64), got {}",
            result[0]
        );
    }

    // -----------------------------------------------------------------------
    // Division by zero: x / 0.0 = Inf (or NaN for 0/0)
    // -----------------------------------------------------------------------

    #[test]
    fn prop_div_f32_by_zero(a in (1.0f32..1e10f32)) {
        let av = Array1::from_vec(vec![a]);
        let bv = Array1::from_vec(vec![0.0f32]);
        let result = simd_div_f32(&av.view(), &bv.view());
        prop_assert!(
            result[0].is_infinite(),
            "x/0.0 should be Inf: got {}",
            result[0]
        );
    }

    #[test]
    fn prop_div_f64_by_zero(a in (1.0f64..1e100f64)) {
        let av = Array1::from_vec(vec![a]);
        let bv = Array1::from_vec(vec![0.0f64]);
        let result = simd_div_f64(&av.view(), &bv.view());
        prop_assert!(
            result[0].is_infinite(),
            "x/0.0 should be Inf (f64): got {}",
            result[0]
        );
    }

    // 0.0 / 0.0 = NaN
    #[test]
    fn prop_div_f32_zero_over_zero(dummy in Just(())) {
        let _ = dummy;
        let av = Array1::from_vec(vec![0.0f32]);
        let bv = Array1::from_vec(vec![0.0f32]);
        let result = simd_div_f32(&av.view(), &bv.view());
        prop_assert!(
            result[0].is_nan(),
            "0.0/0.0 should be NaN: got {}",
            result[0]
        );
    }

    // -----------------------------------------------------------------------
    // sqrt(-x) = NaN for x > 0
    // -----------------------------------------------------------------------

    #[test]
    fn prop_sqrt_f32_negative_is_nan(a in (0.001f32..1e10f32)) {
        let av = Array1::from_vec(vec![-a]);
        let result = simd_sqrt_f32(&av.view());
        prop_assert!(
            result[0].is_nan(),
            "sqrt of negative should be NaN: sqrt({}) = {}",
            -a, result[0]
        );
    }

    #[test]
    fn prop_sqrt_f64_negative_is_nan(a in (0.001f64..1e100f64)) {
        let av = Array1::from_vec(vec![-a]);
        let result = simd_sqrt_f64(&av.view());
        prop_assert!(
            result[0].is_nan(),
            "sqrt of negative should be NaN (f64): sqrt({}) = {}",
            -a, result[0]
        );
    }

    // -----------------------------------------------------------------------
    // sqrt(Inf) = Inf
    // -----------------------------------------------------------------------

    #[test]
    fn prop_sqrt_f32_inf(dummy in Just(())) {
        let _ = dummy;
        let av = Array1::from_vec(vec![f32::INFINITY]);
        let result = simd_sqrt_f32(&av.view());
        prop_assert_eq!(result[0], f32::INFINITY);
    }

    #[test]
    fn prop_sqrt_f64_inf(dummy in Just(())) {
        let _ = dummy;
        let av = Array1::from_vec(vec![f64::INFINITY]);
        let result = simd_sqrt_f64(&av.view());
        prop_assert_eq!(result[0], f64::INFINITY);
    }

    // -----------------------------------------------------------------------
    // Commutativity of add: a + b == b + a (element-wise, handling NaN)
    // -----------------------------------------------------------------------

    #[test]
    fn prop_add_f32_commutative(a in finite_f32_vec(64), b in finite_f32_vec(64)) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];

        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let ab = simd_add_f32(&av.view(), &bv.view());
        let ba = simd_add_f32(&bv.view(), &av.view());

        prop_assert!(
            vec_f32_equivalent(ab.as_slice().expect("slice"), ba.as_slice().expect("slice")),
            "add f32 not commutative"
        );
    }

    // -----------------------------------------------------------------------
    // mul commutativity: a * b == b * a
    // -----------------------------------------------------------------------

    #[test]
    fn prop_mul_f32_commutative(a in finite_f32_vec(64), b in finite_f32_vec(64)) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];

        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let ab = simd_mul_f32(&av.view(), &bv.view());
        let ba = simd_mul_f32(&bv.view(), &av.view());

        prop_assert!(
            vec_f32_equivalent(ab.as_slice().expect("slice"), ba.as_slice().expect("slice")),
            "mul f32 not commutative"
        );
    }

    // -----------------------------------------------------------------------
    // Dot product commutativity: dot(a,b) == dot(b,a)
    // -----------------------------------------------------------------------

    #[test]
    fn prop_dot_f32_commutative(a in finite_f32_vec(64), b in finite_f32_vec(64)) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];

        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let ab = simd_dot_f32(&av.view(), &bv.view());
        let ba = simd_dot_f32(&bv.view(), &av.view());

        prop_assert!(
            f32_equivalent(ab, ba),
            "dot f32 not commutative: {} vs {}",
            ab, ba
        );
    }

    // -----------------------------------------------------------------------
    // Scale invariance: dot(a*c, b) == c * dot(a, b) for scalar c
    // -----------------------------------------------------------------------

    #[test]
    fn prop_dot_f32_scale_invariance(
        a in prop::collection::vec(-100.0f32..100.0f32, 1..32),
        b in prop::collection::vec(-100.0f32..100.0f32, 1..32),
        c in (-10.0f32..10.0f32),
    ) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];

        let ac: Vec<f32> = a.iter().map(|x| x * c).collect();
        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let acv = Array1::from_vec(ac.clone());

        let dot_ab = simd_dot_f32(&av.view(), &bv.view());
        let dot_acb = simd_dot_f32(&acv.view(), &bv.view());
        let lhs = dot_acb;
        let rhs = c * dot_ab;

        // dot(a*c, b) == c * dot(a, b) holds in exact arithmetic.  In floating
        // point the two paths have different rounding: one multiplies first then
        // sums, the other sums first then multiplies.  Use an absolute tolerance
        // scaled by n * sum(|a_i * c * b_i|) * epsilon to accept any rounding path.
        let n = len as f32;
        let abs_scaled_sum: f32 = ac.iter().zip(b.iter()).map(|(x, y)| (x * y).abs()).sum();
        let tol = (n * abs_scaled_sum * f32::EPSILON * 4.0).max(1e-10_f32);

        prop_assert!(
            (lhs - rhs).abs() <= tol || f32_equivalent(lhs, rhs),
            "dot scale invariance failed: dot(a*c,b)={}, c*dot(a,b)={}, diff={}, tol={}",
            lhs, rhs, (lhs - rhs).abs(), tol
        );
    }

    // -----------------------------------------------------------------------
    // Add with zero vector: a + 0 == a
    // -----------------------------------------------------------------------

    #[test]
    fn prop_add_f32_zero_identity(a in finite_f32_vec(64)) {
        let zeros: Vec<f32> = vec![0.0f32; a.len()];
        let av = Array1::from_vec(a.clone());
        let zv = Array1::from_vec(zeros);
        let result = simd_add_f32(&av.view(), &zv.view());

        prop_assert!(
            vec_f32_equivalent(result.as_slice().expect("slice"), &a),
            "a + 0 != a"
        );
    }

    // -----------------------------------------------------------------------
    // Mul by ones: a * 1 == a
    // -----------------------------------------------------------------------

    #[test]
    fn prop_mul_f32_identity(a in finite_f32_vec(64)) {
        let ones: Vec<f32> = vec![1.0f32; a.len()];
        let av = Array1::from_vec(a.clone());
        let ov = Array1::from_vec(ones);
        let result = simd_mul_f32(&av.view(), &ov.view());

        prop_assert!(
            vec_f32_equivalent(result.as_slice().expect("slice"), &a),
            "a * 1 != a"
        );
    }

    // -----------------------------------------------------------------------
    // Mul by zero: a * 0 == 0 (for finite a)
    // -----------------------------------------------------------------------

    #[test]
    fn prop_mul_f32_zero_annihilation(a in prop::collection::vec(-1e10f32..1e10f32, 0..64)) {
        let zeros: Vec<f32> = vec![0.0f32; a.len()];
        let av = Array1::from_vec(a.clone());
        let zv = Array1::from_vec(zeros);
        let result = simd_mul_f32(&av.view(), &zv.view());

        for v in result.iter() {
            prop_assert!(*v == 0.0, "a * 0 != 0: got {}", v);
        }
    }
}

// -----------------------------------------------------------------------
// Additional non-proptest unit tests for specific edge-case values
// -----------------------------------------------------------------------

#[test]
fn test_abs_all_special_values_f32() {
    let special = vec![
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0f32,
        -0.0f32,
        f32::MIN_POSITIVE,
        f32::MIN_POSITIVE * 0.5,
        -f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE * 0.5,
    ];
    let av = Array1::from_vec(special.clone());
    let result = simd_abs_f32(&av.view());
    let expected = scalar_abs_f32(&special);

    assert!(
        vec_f32_equivalent(
            result.as_slice().expect("Operation failed"),
            &expected
        ),
        "abs of special values mismatch: {:?} vs {:?}",
        result.as_slice().expect("slice"),
        expected
    );
}

#[test]
fn test_abs_all_special_values_f64() {
    let special = vec![
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0f64,
        -0.0f64,
        f64::MIN_POSITIVE,
        f64::MIN_POSITIVE * 0.5,
        -f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE * 0.5,
    ];
    let av = Array1::from_vec(special.clone());
    let result = simd_abs_f64(&av.view());
    let expected = scalar_abs_f64(&special);

    assert!(
        vec_f64_equivalent(
            result.as_slice().expect("Operation failed"),
            &expected
        ),
        "abs of special values (f64) mismatch"
    );
}

#[test]
fn test_sum_with_inf_f32() {
    // Inf + finite = Inf
    let data = vec![f32::INFINITY, 1.0, 2.0, 3.0];
    let av = Array1::from_vec(data);
    let result = simd_sum_f32(&av.view());
    assert!(result.is_infinite() && result > 0.0);
}

#[test]
fn test_sum_with_neg_inf_f32() {
    // -Inf + finite = -Inf
    let data = vec![f32::NEG_INFINITY, 1.0, 2.0, 3.0];
    let av = Array1::from_vec(data);
    let result = simd_sum_f32(&av.view());
    assert!(result.is_infinite() && result < 0.0);
}

#[test]
fn test_sum_inf_plus_neg_inf_f32() {
    // +Inf + -Inf = NaN
    let data = vec![f32::INFINITY, f32::NEG_INFINITY];
    let av = Array1::from_vec(data);
    let result = simd_sum_f32(&av.view());
    assert!(result.is_nan());
}

#[test]
fn test_add_f32_empty_arrays() {
    let av = Array1::<f32>::zeros(0);
    let bv = Array1::<f32>::zeros(0);
    let result = simd_add_f32(&av.view(), &bv.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_add_f64_empty_arrays() {
    let av = Array1::<f64>::zeros(0);
    let bv = Array1::<f64>::zeros(0);
    let result = simd_add_f64(&av.view(), &bv.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_sqrt_f32_zero() {
    let av = Array1::from_vec(vec![0.0f32]);
    let result = simd_sqrt_f32(&av.view());
    assert_eq!(result[0], 0.0f32);
}

#[test]
fn test_sqrt_f64_zero() {
    let av = Array1::from_vec(vec![0.0f64]);
    let result = simd_sqrt_f64(&av.view());
    assert_eq!(result[0], 0.0f64);
}

#[test]
fn test_sqrt_f32_negative_zero() {
    // sqrt(-0.0) = 0.0 in IEEE 754
    let av = Array1::from_vec(vec![-0.0f32]);
    let result = simd_sqrt_f32(&av.view());
    assert_eq!(result[0], 0.0f32);
}

#[test]
fn test_add_f32_single_element() {
    let a = Array1::from_vec(vec![42.0f32]);
    let b = Array1::from_vec(vec![58.0f32]);
    let result = simd_add_f32(&a.view(), &b.view());
    assert_eq!(result[0], 100.0f32);
}

#[test]
fn test_mul_f32_with_subnormals() {
    let subnormal = f32::MIN_POSITIVE * 0.5;
    let a = Array1::from_vec(vec![subnormal, subnormal]);
    let b = Array1::from_vec(vec![2.0f32, 0.5f32]);
    let result = simd_mul_f32(&a.view(), &b.view());
    let expected_0 = subnormal * 2.0;
    let expected_1 = subnormal * 0.5;
    // expected_1 may underflow to 0
    assert!(
        f32_equivalent(result[0], expected_0),
        "subnormal * 2: got {}, expected {}",
        result[0],
        expected_0
    );
    assert!(
        f32_equivalent(result[1], expected_1),
        "subnormal * 0.5: got {}, expected {}",
        result[1],
        expected_1
    );
}

#[test]
fn test_div_f32_large_values() {
    let a = Array1::from_vec(vec![f32::MAX]);
    let b = Array1::from_vec(vec![f32::MIN_POSITIVE]);
    let result = simd_div_f32(&a.view(), &b.view());
    // MAX / MIN_POSITIVE overflows to Inf
    assert!(
        result[0].is_infinite(),
        "MAX / MIN_POSITIVE should overflow to Inf"
    );
}

#[test]
fn test_floor_f32_special_values() {
    let special = vec![
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0f32,
        -0.0f32,
        1.5f32,
        -1.5f32,
        f32::MIN_POSITIVE,
    ];
    let av = Array1::from_vec(special.clone());
    let result = simd_floor_f32(&av.view());
    let expected = scalar_floor_f32(&special);

    assert!(
        vec_f32_equivalent(
            result.as_slice().expect("Operation failed"),
            &expected
        ),
        "floor of special values mismatch: {:?} vs {:?}",
        result.as_slice().expect("slice"),
        expected
    );
}

#[test]
fn test_min_max_f32_with_single_element() {
    let av = Array1::from_vec(vec![42.0f32]);
    let min = simd_min_f32(&av.view());
    let max = simd_max_f32(&av.view());
    assert_eq!(min, 42.0f32);
    assert_eq!(max, 42.0f32);
}

#[test]
fn test_dot_f32_orthogonal() {
    // Dot product of orthogonal unit vectors is 0
    let a = Array1::from_vec(vec![1.0f32, 0.0, 0.0]);
    let b = Array1::from_vec(vec![0.0f32, 1.0, 0.0]);
    let result = simd_dot_f32(&a.view(), &b.view());
    assert_eq!(result, 0.0f32);
}

#[test]
fn test_dot_f64_unit_vectors() {
    // Dot product of unit vector with itself is 1
    let scale = (3.0f64).sqrt().recip();
    let a = Array1::from_vec(vec![scale, scale, scale]);
    let result = simd_dot_f64(&a.view(), &a.view());
    assert!(
        (result - 1.0f64).abs() < 1e-10,
        "unit vector self-dot should be 1.0: got {}",
        result
    );
}

// -----------------------------------------------------------------------
// Extra strategies referenced inside proptest! block
// -----------------------------------------------------------------------

fn non_negative_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(non_negative_f32(), 0..=max_len)
}

fn non_negative_f64_vec(max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(non_negative_f64(), 0..=max_len)
}

fn finite_positive_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(
        prop_oneof![
            (0.0f32..1e30f32),
            Just(f32::MIN_POSITIVE),
            Just(f32::MIN_POSITIVE * 0.5),
        ],
        0..=max_len,
    )
}

fn finite_positive_f64_vec(max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(
        prop_oneof![
            (0.0f64..1e200f64),
            Just(f64::MIN_POSITIVE),
            Just(f64::MIN_POSITIVE * 0.5),
        ],
        0..=max_len,
    )
}
