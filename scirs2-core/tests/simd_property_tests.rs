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

use proptest::prelude::*;
use scirs2_core::ndarray::Array1;
use scirs2_core::ndarray_ext::elementwise::{
    abs_simd, add_simd, ceil_simd, div_simd, dot_simd, floor_simd, mul_simd, round_simd,
    scalar_mul_simd, sqrt_simd, sub_simd,
};
use scirs2_core::ndarray_ext::reduction::{max_simd, mean_simd, min_simd, sum_simd};

// ---------------------------------------------------------------------------
// Strategies for generating interesting float values
// ---------------------------------------------------------------------------

/// Generates a broad range of f32 values including special IEEE 754 cases.
fn interesting_f32() -> impl Strategy<Value = f32> {
    prop_oneof![
        (-1e30f32..1e30f32),
        Just(f32::MAX),
        Just(f32::MIN),
        Just(f32::MIN_POSITIVE),
        // subnormal
        Just(f32::MIN_POSITIVE * 0.5),
        Just(f32::MIN_POSITIVE * 0.1),
        // zeros
        Just(0.0f32),
        Just(-0.0f32),
        // infinities
        Just(f32::INFINITY),
        Just(f32::NEG_INFINITY),
        // NaN
        Just(f32::NAN),
        // around 1.0
        (-2.0f32..2.0f32),
    ]
}

/// Generates a broad range of f64 values including special IEEE 754 cases.
fn interesting_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        (-1e200f64..1e200f64),
        Just(f64::MAX),
        Just(f64::MIN),
        Just(f64::MIN_POSITIVE),
        // subnormal
        Just(f64::MIN_POSITIVE * 0.5),
        Just(f64::MIN_POSITIVE * 0.1),
        // zeros
        Just(0.0f64),
        Just(-0.0f64),
        // infinities
        Just(f64::INFINITY),
        Just(f64::NEG_INFINITY),
        // NaN
        Just(f64::NAN),
        // around 1.0
        (-2.0f64..2.0f64),
    ]
}

/// Finite-only f32 (no NaN, but allows Inf and subnormals).
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

/// Finite-only f64 (no NaN, but allows Inf and subnormals).
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

/// Non-negative f32 values for sqrt tests.
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

/// Non-negative f64 values for sqrt tests.
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

// Vec strategies
fn interesting_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(interesting_f32(), 0..=max_len)
}

fn interesting_f64_vec(max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(interesting_f64(), 0..=max_len)
}

fn finite_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(finite_f32(), 0..=max_len)
}

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

fn non_negative_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(non_negative_f32(), 0..=max_len)
}

fn non_negative_f64_vec(max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(non_negative_f64(), 0..=max_len)
}

fn positive_finite_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(
        prop_oneof![
            (0.0f32..1e30f32),
            Just(f32::MIN_POSITIVE),
            Just(f32::MIN_POSITIVE * 0.5),
        ],
        0..=max_len,
    )
}

fn positive_finite_f64_vec(max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(
        prop_oneof![
            (0.0f64..1e200f64),
            Just(f64::MIN_POSITIVE),
            Just(f64::MIN_POSITIVE * 0.5),
        ],
        0..=max_len,
    )
}

// ---------------------------------------------------------------------------
// Scalar reference implementations for comparison
// ---------------------------------------------------------------------------

fn ref_add_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn ref_add_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn ref_sub_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn ref_sub_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn ref_mul_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

fn ref_mul_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

fn ref_div_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x / y).collect()
}

fn ref_div_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x / y).collect()
}

fn ref_abs_f32(a: &[f32]) -> Vec<f32> {
    a.iter().map(|x| x.abs()).collect()
}

fn ref_abs_f64(a: &[f64]) -> Vec<f64> {
    a.iter().map(|x| x.abs()).collect()
}

fn ref_sqrt_f32(a: &[f32]) -> Vec<f32> {
    a.iter().map(|x| x.sqrt()).collect()
}

fn ref_sqrt_f64(a: &[f64]) -> Vec<f64> {
    a.iter().map(|x| x.sqrt()).collect()
}

fn ref_floor_f32(a: &[f32]) -> Vec<f32> {
    a.iter().map(|x| x.floor()).collect()
}

fn ref_floor_f64(a: &[f64]) -> Vec<f64> {
    a.iter().map(|x| x.floor()).collect()
}

fn ref_ceil_f32(a: &[f32]) -> Vec<f32> {
    a.iter().map(|x| x.ceil()).collect()
}

fn ref_ceil_f64(a: &[f64]) -> Vec<f64> {
    a.iter().map(|x| x.ceil()).collect()
}

fn ref_round_f32(a: &[f32]) -> Vec<f32> {
    a.iter().map(|x| x.round()).collect()
}

fn ref_round_f64(a: &[f64]) -> Vec<f64> {
    a.iter().map(|x| x.round()).collect()
}

fn ref_sum_f32(a: &[f32]) -> f32 {
    a.iter().copied().sum()
}

fn ref_sum_f64(a: &[f64]) -> f64 {
    a.iter().copied().sum()
}

fn ref_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn ref_dot_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// Comparison helpers: NaN == NaN is treated as equal
// ---------------------------------------------------------------------------

fn f32_eq_with_nan(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || (a == b)
}

fn f64_eq_with_nan(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a == b)
}

/// Relative-tolerance comparison for f32, with NaN == NaN.
fn f32_approx(a: f32, b: f32) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a == b {
        return true;
    }
    let diff = (a - b).abs();
    let mag = a.abs().max(b.abs());
    if mag == 0.0 {
        diff < 1e-5
    } else {
        diff / mag < 1e-4
    }
}

/// Relative-tolerance comparison for f64, with NaN == NaN.
fn f64_approx(a: f64, b: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a == b {
        return true;
    }
    let diff = (a - b).abs();
    let mag = a.abs().max(b.abs());
    if mag == 0.0 {
        diff < 1e-10
    } else {
        diff / mag < 1e-10
    }
}

fn vec_f32_approx(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| f32_approx(*x, *y))
}

fn vec_f64_approx(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| f64_approx(*x, *y))
}

/// Compare two f32 sum/reduction results with absolute tolerance derived from
/// the input slice.  Accounts for catastrophic cancellation when large values
/// of opposite sign are summed in a different order by SIMD vs scalar paths.
fn f32_sum_approx(simd: f32, scalar: f32, inputs: &[f32]) -> bool {
    if simd.is_nan() && scalar.is_nan() {
        return true;
    }
    if simd == scalar {
        return true;
    }
    if simd.is_infinite() || scalar.is_infinite() {
        return simd.is_infinite() && scalar.is_infinite() && simd.signum() == scalar.signum();
    }
    let abs_input_sum: f32 = inputs.iter().map(|x| x.abs()).sum();
    let n = inputs.len() as f32;
    let tol = (n * abs_input_sum * f32::EPSILON).max(1e-10_f32);
    (simd - scalar).abs() <= tol
}

/// Compare two f64 sum/reduction results with absolute tolerance derived from
/// the input slice.
fn f64_sum_approx(simd: f64, scalar: f64, inputs: &[f64]) -> bool {
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

/// Compare two f32 dot-product results with absolute tolerance derived from
/// both input slices.
fn f32_dot_approx(simd: f32, scalar: f32, a: &[f32], b: &[f32]) -> bool {
    if simd.is_nan() && scalar.is_nan() {
        return true;
    }
    if simd == scalar {
        return true;
    }
    if simd.is_infinite() || scalar.is_infinite() {
        return simd.is_infinite() && scalar.is_infinite() && simd.signum() == scalar.signum();
    }
    let abs_product_sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x * y).abs()).sum();
    let n = a.len() as f32;
    let tol = (n * abs_product_sum * f32::EPSILON).max(1e-10_f32);
    (simd - scalar).abs() <= tol
}

/// Compare two f64 dot-product results with absolute tolerance derived from
/// both input slices.
fn f64_dot_approx(simd: f64, scalar: f64, a: &[f64], b: &[f64]) -> bool {
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

// ---------------------------------------------------------------------------
// proptest! tests
// ---------------------------------------------------------------------------

proptest! {
    // -----------------------------------------------------------------------
    // add_simd: element-wise addition matches scalar reference
    // -----------------------------------------------------------------------

    #[test]
    fn prop_add_f32_matches_scalar(a in interesting_f32_vec(64), b in interesting_f32_vec(64)) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];
        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let simd = add_simd::<f32>(&av.view(), &bv.view());
        let reference = ref_add_f32(a, b);
        prop_assert!(
            vec_f32_approx(simd.as_slice().expect("simd slice"), &reference),
            "add_simd f32 mismatch: simd={:?} ref={:?}",
            &simd.as_slice().expect("slice")[..len.min(8)],
            &reference[..len.min(8)]
        );
    }

    #[test]
    fn prop_add_f64_matches_scalar(a in interesting_f64_vec(64), b in interesting_f64_vec(64)) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];
        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let simd = add_simd::<f64>(&av.view(), &bv.view());
        let reference = ref_add_f64(a, b);
        prop_assert!(
            vec_f64_approx(simd.as_slice().expect("simd slice"), &reference),
            "add_simd f64 mismatch"
        );
    }

    // -----------------------------------------------------------------------
    // sub_simd: element-wise subtraction
    // -----------------------------------------------------------------------

    #[test]
    fn prop_sub_f32_matches_scalar(a in interesting_f32_vec(64), b in interesting_f32_vec(64)) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];
        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let simd = sub_simd::<f32>(&av.view(), &bv.view());
        let reference = ref_sub_f32(a, b);
        prop_assert!(
            vec_f32_approx(simd.as_slice().expect("simd slice"), &reference),
            "sub_simd f32 mismatch"
        );
    }

    #[test]
    fn prop_sub_f64_matches_scalar(a in interesting_f64_vec(64), b in interesting_f64_vec(64)) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];
        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let simd = sub_simd::<f64>(&av.view(), &bv.view());
        let reference = ref_sub_f64(a, b);
        prop_assert!(
            vec_f64_approx(simd.as_slice().expect("simd slice"), &reference),
            "sub_simd f64 mismatch"
        );
    }

    // -----------------------------------------------------------------------
    // mul_simd: element-wise multiplication
    // -----------------------------------------------------------------------

    #[test]
    fn prop_mul_f32_matches_scalar(a in interesting_f32_vec(64), b in interesting_f32_vec(64)) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];
        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let simd = mul_simd::<f32>(&av.view(), &bv.view());
        let reference = ref_mul_f32(a, b);
        prop_assert!(
            vec_f32_approx(simd.as_slice().expect("simd slice"), &reference),
            "mul_simd f32 mismatch"
        );
    }

    #[test]
    fn prop_mul_f64_matches_scalar(a in interesting_f64_vec(64), b in interesting_f64_vec(64)) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];
        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let simd = mul_simd::<f64>(&av.view(), &bv.view());
        let reference = ref_mul_f64(a, b);
        prop_assert!(
            vec_f64_approx(simd.as_slice().expect("simd slice"), &reference),
            "mul_simd f64 mismatch"
        );
    }

    // -----------------------------------------------------------------------
    // div_simd: element-wise division
    // -----------------------------------------------------------------------

    #[test]
    fn prop_div_f32_matches_scalar(a in interesting_f32_vec(64), b in interesting_f32_vec(64)) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];
        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let simd = div_simd::<f32>(&av.view(), &bv.view());
        let reference = ref_div_f32(a, b);
        prop_assert!(
            vec_f32_approx(simd.as_slice().expect("simd slice"), &reference),
            "div_simd f32 mismatch"
        );
    }

    #[test]
    fn prop_div_f64_matches_scalar(a in interesting_f64_vec(64), b in interesting_f64_vec(64)) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];
        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let simd = div_simd::<f64>(&av.view(), &bv.view());
        let reference = ref_div_f64(a, b);
        prop_assert!(
            vec_f64_approx(simd.as_slice().expect("simd slice"), &reference),
            "div_simd f64 mismatch"
        );
    }

    // -----------------------------------------------------------------------
    // abs_simd: unary absolute value
    // -----------------------------------------------------------------------

    #[test]
    fn prop_abs_f32_matches_scalar(a in interesting_f32_vec(64)) {
        let av = Array1::from_vec(a.clone());
        let simd = abs_simd::<f32>(&av.view());
        let reference = ref_abs_f32(&a);
        prop_assert!(
            vec_f32_approx(simd.as_slice().expect("simd slice"), &reference),
            "abs_simd f32 mismatch"
        );
    }

    #[test]
    fn prop_abs_f64_matches_scalar(a in interesting_f64_vec(64)) {
        let av = Array1::from_vec(a.clone());
        let simd = abs_simd::<f64>(&av.view());
        let reference = ref_abs_f64(&a);
        prop_assert!(
            vec_f64_approx(simd.as_slice().expect("simd slice"), &reference),
            "abs_simd f64 mismatch"
        );
    }

    /// abs result is always non-negative (NaN is allowed to stay NaN)
    #[test]
    fn prop_abs_f32_result_non_negative(a in interesting_f32_vec(64)) {
        let av = Array1::from_vec(a.clone());
        let result = abs_simd::<f32>(&av.view());
        for &v in result.iter() {
            if !v.is_nan() {
                prop_assert!(v >= 0.0, "abs result was negative: {}", v);
            }
        }
    }

    #[test]
    fn prop_abs_f64_result_non_negative(a in interesting_f64_vec(64)) {
        let av = Array1::from_vec(a.clone());
        let result = abs_simd::<f64>(&av.view());
        for &v in result.iter() {
            if !v.is_nan() {
                prop_assert!(v >= 0.0, "abs result was negative: {}", v);
            }
        }
    }

    // -----------------------------------------------------------------------
    // sqrt_simd: square root
    // -----------------------------------------------------------------------

    #[test]
    fn prop_sqrt_f32_matches_scalar(a in non_negative_f32_vec(64)) {
        let av = Array1::from_vec(a.clone());
        let simd = sqrt_simd::<f32>(&av.view());
        let reference = ref_sqrt_f32(&a);
        prop_assert!(
            vec_f32_approx(simd.as_slice().expect("simd slice"), &reference),
            "sqrt_simd f32 mismatch"
        );
    }

    #[test]
    fn prop_sqrt_f64_matches_scalar(a in non_negative_f64_vec(64)) {
        let av = Array1::from_vec(a.clone());
        let simd = sqrt_simd::<f64>(&av.view());
        let reference = ref_sqrt_f64(&a);
        prop_assert!(
            vec_f64_approx(simd.as_slice().expect("simd slice"), &reference),
            "sqrt_simd f64 mismatch"
        );
    }

    /// sqrt(x)^2 ≈ x for positive finite inputs
    #[test]
    fn prop_sqrt_f32_squaring_recovers_input(a in positive_finite_f32_vec(32)) {
        let av = Array1::from_vec(a.clone());
        let sq = sqrt_simd::<f32>(&av.view());
        for (orig, s) in a.iter().zip(sq.iter()) {
            if orig.is_nan() || orig.is_infinite() {
                continue;
            }
            let recovered = s * s;
            let rel = if *orig == 0.0 {
                recovered.abs()
            } else {
                ((recovered - orig) / orig).abs()
            };
            prop_assert!(rel < 1e-4, "sqrt({})^2 = {}, rel_err = {}", orig, recovered, rel);
        }
    }

    #[test]
    fn prop_sqrt_f64_squaring_recovers_input(a in positive_finite_f64_vec(32)) {
        let av = Array1::from_vec(a.clone());
        let sq = sqrt_simd::<f64>(&av.view());
        for (orig, s) in a.iter().zip(sq.iter()) {
            if orig.is_nan() || orig.is_infinite() {
                continue;
            }
            let recovered = s * s;
            let rel = if *orig == 0.0 {
                recovered.abs()
            } else {
                ((recovered - orig) / orig).abs()
            };
            prop_assert!(rel < 1e-10, "sqrt({})^2 = {}, rel_err = {}", orig, recovered, rel);
        }
    }

    // -----------------------------------------------------------------------
    // sqrt of negative numbers -> NaN
    // -----------------------------------------------------------------------

    #[test]
    fn prop_sqrt_f32_negative_is_nan(a in (0.001f32..1e10f32)) {
        let av = Array1::from_vec(vec![-a]);
        let result = sqrt_simd::<f32>(&av.view());
        prop_assert!(result[0].is_nan(), "sqrt({}) should be NaN, got {}", -a, result[0]);
    }

    #[test]
    fn prop_sqrt_f64_negative_is_nan(a in (0.001f64..1e100f64)) {
        let av = Array1::from_vec(vec![-a]);
        let result = sqrt_simd::<f64>(&av.view());
        prop_assert!(result[0].is_nan(), "sqrt({}) should be NaN, got {}", -a, result[0]);
    }

    // -----------------------------------------------------------------------
    // floor_simd: rounding down
    // -----------------------------------------------------------------------

    #[test]
    fn prop_floor_f32_matches_scalar(a in interesting_f32_vec(64)) {
        let av = Array1::from_vec(a.clone());
        let simd = floor_simd::<f32>(&av.view());
        let reference = ref_floor_f32(&a);
        prop_assert!(
            vec_f32_approx(simd.as_slice().expect("simd slice"), &reference),
            "floor_simd f32 mismatch"
        );
    }

    #[test]
    fn prop_floor_f64_matches_scalar(a in interesting_f64_vec(64)) {
        let av = Array1::from_vec(a.clone());
        let simd = floor_simd::<f64>(&av.view());
        let reference = ref_floor_f64(&a);
        prop_assert!(
            vec_f64_approx(simd.as_slice().expect("simd slice"), &reference),
            "floor_simd f64 mismatch"
        );
    }

    /// floor(x) <= x for all finite values
    #[test]
    fn prop_floor_f32_leq_input(a in prop::collection::vec(-1e10f32..1e10f32, 0..64)) {
        let av = Array1::from_vec(a.clone());
        let floored = floor_simd::<f32>(&av.view());
        for (orig, fl) in a.iter().zip(floored.iter()) {
            prop_assert!(*fl <= *orig, "floor({}) = {} violates floor(x) <= x", orig, fl);
        }
    }

    #[test]
    fn prop_floor_f64_leq_input(a in prop::collection::vec(-1e100f64..1e100f64, 0..64)) {
        let av = Array1::from_vec(a.clone());
        let floored = floor_simd::<f64>(&av.view());
        for (orig, fl) in a.iter().zip(floored.iter()) {
            prop_assert!(*fl <= *orig, "floor({}) = {} violates floor(x) <= x", orig, fl);
        }
    }

    // -----------------------------------------------------------------------
    // ceil_simd: rounding up
    // -----------------------------------------------------------------------

    #[test]
    fn prop_ceil_f32_matches_scalar(a in interesting_f32_vec(64)) {
        let av = Array1::from_vec(a.clone());
        let simd = ceil_simd::<f32>(&av.view());
        let reference = ref_ceil_f32(&a);
        prop_assert!(
            vec_f32_approx(simd.as_slice().expect("simd slice"), &reference),
            "ceil_simd f32 mismatch"
        );
    }

    #[test]
    fn prop_ceil_f64_matches_scalar(a in interesting_f64_vec(64)) {
        let av = Array1::from_vec(a.clone());
        let simd = ceil_simd::<f64>(&av.view());
        let reference = ref_ceil_f64(&a);
        prop_assert!(
            vec_f64_approx(simd.as_slice().expect("simd slice"), &reference),
            "ceil_simd f64 mismatch"
        );
    }

    /// ceil(x) >= x for all finite values
    #[test]
    fn prop_ceil_f32_geq_input(a in prop::collection::vec(-1e10f32..1e10f32, 0..64)) {
        let av = Array1::from_vec(a.clone());
        let ceiled = ceil_simd::<f32>(&av.view());
        for (orig, c) in a.iter().zip(ceiled.iter()) {
            prop_assert!(*c >= *orig, "ceil({}) = {} violates ceil(x) >= x", orig, c);
        }
    }

    // -----------------------------------------------------------------------
    // round_simd: rounding to nearest integer
    // -----------------------------------------------------------------------

    #[test]
    fn prop_round_f32_matches_scalar(a in interesting_f32_vec(64)) {
        let av = Array1::from_vec(a.clone());
        let simd = round_simd::<f32>(&av.view());
        let reference = ref_round_f32(&a);
        prop_assert!(
            vec_f32_approx(simd.as_slice().expect("simd slice"), &reference),
            "round_simd f32 mismatch"
        );
    }

    #[test]
    fn prop_round_f64_matches_scalar(a in interesting_f64_vec(64)) {
        let av = Array1::from_vec(a.clone());
        let simd = round_simd::<f64>(&av.view());
        let reference = ref_round_f64(&a);
        prop_assert!(
            vec_f64_approx(simd.as_slice().expect("simd slice"), &reference),
            "round_simd f64 mismatch"
        );
    }

    // -----------------------------------------------------------------------
    // sum_simd: sum reduction
    // -----------------------------------------------------------------------

    #[test]
    fn prop_sum_f32_single_element(v in interesting_f32()) {
        let av = Array1::from_vec(vec![v]);
        let result = sum_simd::<f32>(&av.view());
        prop_assert!(
            f32_approx(result, v),
            "sum of single element should equal element: sum={}, v={}",
            result, v
        );
    }

    #[test]
    fn prop_sum_f64_single_element(v in interesting_f64()) {
        let av = Array1::from_vec(vec![v]);
        let result = sum_simd::<f64>(&av.view());
        prop_assert!(
            f64_approx(result, v),
            "sum of single element should equal element: sum={}, v={}",
            result, v
        );
    }

    #[test]
    fn prop_sum_f32_matches_scalar(a in bounded_f32_vec(64)) {
        // Uses bounded_f32_vec (no Inf/NaN, no MAX/MIN) so 64-element sums cannot
        // overflow.  Uses f32_sum_approx which allows absolute error proportional
        // to n * sum(|inputs|) * epsilon, accounting for different accumulation order
        // (catastrophic cancellation) between SIMD and scalar paths.
        let av = Array1::from_vec(a.clone());
        let simd = sum_simd::<f32>(&av.view());
        let reference = ref_sum_f32(&a);
        prop_assert!(
            f32_sum_approx(simd, reference, &a),
            "sum_simd f32 mismatch: simd={}, ref={}",
            simd, reference
        );
    }

    #[test]
    fn prop_sum_f64_matches_scalar(a in bounded_f64_vec(64)) {
        // Uses bounded_f64_vec and f64_sum_approx for the same reasons.
        let av = Array1::from_vec(a.clone());
        let simd = sum_simd::<f64>(&av.view());
        let reference = ref_sum_f64(&a);
        prop_assert!(
            f64_sum_approx(simd, reference, &a),
            "sum_simd f64 mismatch: simd={}, ref={}",
            simd, reference
        );
    }

    /// NaN in input propagates to sum output
    #[test]
    fn prop_sum_f32_nan_propagates(prefix in prop::collection::vec(finite_f32(), 0..31)) {
        let mut data = prefix;
        data.push(f32::NAN);
        let av = Array1::from_vec(data);
        let result = sum_simd::<f32>(&av.view());
        prop_assert!(result.is_nan(), "NaN should propagate through sum_simd f32");
    }

    #[test]
    fn prop_sum_f64_nan_propagates(prefix in prop::collection::vec(finite_f64(), 0..31)) {
        let mut data = prefix;
        data.push(f64::NAN);
        let av = Array1::from_vec(data);
        let result = sum_simd::<f64>(&av.view());
        prop_assert!(result.is_nan(), "NaN should propagate through sum_simd f64");
    }

    // -----------------------------------------------------------------------
    // dot_simd: dot product
    // -----------------------------------------------------------------------

    #[test]
    fn prop_dot_f32_matches_scalar(a in dot_bounded_f32_vec(64), b in dot_bounded_f32_vec(64)) {
        // Uses dot_bounded_f32_vec: elements bounded to ±1e15 so that element-wise
        // products (up to 1e30) and their 64-element sum (up to 6.4e31) cannot
        // overflow f32::MAX (~3.4e38).  Uses f32_dot_approx for input-scaled tolerance.
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];
        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let simd = dot_simd::<f32>(&av.view(), &bv.view());
        let reference = ref_dot_f32(a, b);
        prop_assert!(
            f32_dot_approx(simd, reference, a, b),
            "dot_simd f32 mismatch: simd={}, ref={}",
            simd, reference
        );
    }

    #[test]
    fn prop_dot_f64_matches_scalar(a in dot_bounded_f64_vec(64), b in dot_bounded_f64_vec(64)) {
        // Uses dot_bounded_f64_vec: elements bounded to ±1e150 so products (up to
        // 1e300) and their 64-element sum are below f64::MAX (~1.8e308).
        // Uses f64_dot_approx for input-scaled tolerance.
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];
        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let simd = dot_simd::<f64>(&av.view(), &bv.view());
        let reference = ref_dot_f64(a, b);
        prop_assert!(
            f64_dot_approx(simd, reference, a, b),
            "dot_simd f64 mismatch: simd={}, ref={}",
            simd, reference
        );
    }

    /// dot(a, b) == dot(b, a) commutativity
    #[test]
    fn prop_dot_f32_commutative(a in finite_f32_vec(64), b in finite_f32_vec(64)) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];
        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let ab = dot_simd::<f32>(&av.view(), &bv.view());
        let ba = dot_simd::<f32>(&bv.view(), &av.view());
        prop_assert!(f32_approx(ab, ba), "dot not commutative: {} vs {}", ab, ba);
    }

    // -----------------------------------------------------------------------
    // min_simd / max_simd reductions (element-wise variants)
    // -----------------------------------------------------------------------

    /// For non-empty arrays, min <= max
    #[test]
    fn prop_min_leq_max_f32(a in prop::collection::vec(-1e30f32..1e30f32, 1..64)) {
        let av = Array1::from_vec(a.clone());
        if let (Some(mn), Some(mx)) = (min_simd::<f32>(&av.view()), max_simd::<f32>(&av.view())) {
            prop_assert!(mn <= mx, "min={} > max={}", mn, mx);
        }
    }

    #[test]
    fn prop_min_leq_max_f64(a in prop::collection::vec(-1e200f64..1e200f64, 1..64)) {
        let av = Array1::from_vec(a.clone());
        if let (Some(mn), Some(mx)) = (min_simd::<f64>(&av.view()), max_simd::<f64>(&av.view())) {
            prop_assert!(mn <= mx, "min={} > max={}", mn, mx);
        }
    }

    // -----------------------------------------------------------------------
    // mean_simd: value in [min, max]
    // -----------------------------------------------------------------------

    #[test]
    fn prop_mean_f32_in_range(a in prop::collection::vec(-1e10f32..1e10f32, 1..64)) {
        let av = Array1::from_vec(a.clone());
        if let (Some(mean), Some(mn), Some(mx)) = (
            mean_simd::<f32>(&av.view()),
            min_simd::<f32>(&av.view()),
            max_simd::<f32>(&av.view()),
        ) {
            let slack = (mx - mn).abs() * 1e-4 + 1e-6;
            prop_assert!(
                mean >= mn - slack && mean <= mx + slack,
                "mean {} not in [{}, {}]",
                mean, mn, mx
            );
        }
    }

    #[test]
    fn prop_mean_f64_in_range(a in prop::collection::vec(-1e100f64..1e100f64, 1..64)) {
        let av = Array1::from_vec(a.clone());
        if let (Some(mean), Some(mn), Some(mx)) = (
            mean_simd::<f64>(&av.view()),
            min_simd::<f64>(&av.view()),
            max_simd::<f64>(&av.view()),
        ) {
            let slack = (mx - mn).abs() * 1e-10 + 1e-12;
            prop_assert!(
                mean >= mn - slack && mean <= mx + slack,
                "mean {} not in [{}, {}]",
                mean, mn, mx
            );
        }
    }

    // -----------------------------------------------------------------------
    // All-same-value arrays: sum == n*v, mean == v, min == v, max == v
    // -----------------------------------------------------------------------

    #[test]
    fn prop_all_same_f32_consistency(v in (-1e10f32..1e10f32), n in 1usize..64) {
        let data: Vec<f32> = vec![v; n];
        let av = Array1::from_vec(data);
        let s = sum_simd::<f32>(&av.view());
        let expected_sum = v * n as f32;
        prop_assert!(
            f32_approx(s, expected_sum),
            "all-same sum: {} != {}",
            s, expected_sum
        );
        if let Some(mean) = mean_simd::<f32>(&av.view()) {
            prop_assert!(f32_approx(mean, v), "all-same mean: {} != {}", mean, v);
        }
        if let (Some(mn), Some(mx)) = (min_simd::<f32>(&av.view()), max_simd::<f32>(&av.view())) {
            prop_assert!(f32_approx(mn, v), "all-same min: {} != {}", mn, v);
            prop_assert!(f32_approx(mx, v), "all-same max: {} != {}", mx, v);
        }
    }

    #[test]
    fn prop_all_same_f64_consistency(v in (-1e100f64..1e100f64), n in 1usize..64) {
        let data: Vec<f64> = vec![v; n];
        let av = Array1::from_vec(data);
        let s = sum_simd::<f64>(&av.view());
        let expected_sum = v * n as f64;
        prop_assert!(
            f64_approx(s, expected_sum),
            "all-same sum: {} != {}",
            s, expected_sum
        );
    }

    // -----------------------------------------------------------------------
    // add commutativity: a + b == b + a
    // -----------------------------------------------------------------------

    #[test]
    fn prop_add_f32_commutative(a in finite_f32_vec(64), b in finite_f32_vec(64)) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];
        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let ab = add_simd::<f32>(&av.view(), &bv.view());
        let ba = add_simd::<f32>(&bv.view(), &av.view());
        prop_assert!(
            vec_f32_approx(
                ab.as_slice().expect("ab slice"),
                ba.as_slice().expect("ba slice")
            ),
            "add f32 not commutative"
        );
    }

    // -----------------------------------------------------------------------
    // mul commutativity
    // -----------------------------------------------------------------------

    #[test]
    fn prop_mul_f32_commutative(a in finite_f32_vec(64), b in finite_f32_vec(64)) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];
        let av = Array1::from_vec(a.to_vec());
        let bv = Array1::from_vec(b.to_vec());
        let ab = mul_simd::<f32>(&av.view(), &bv.view());
        let ba = mul_simd::<f32>(&bv.view(), &av.view());
        prop_assert!(
            vec_f32_approx(
                ab.as_slice().expect("ab slice"),
                ba.as_slice().expect("ba slice")
            ),
            "mul f32 not commutative"
        );
    }

    // -----------------------------------------------------------------------
    // scalar_mul_simd: a * c == c * a element-wise (implicit commutativity)
    // -----------------------------------------------------------------------

    #[test]
    fn prop_scalar_mul_f32_matches_manual(
        a in prop::collection::vec(-1e10f32..1e10f32, 0..64),
        c in -1e10f32..1e10f32,
    ) {
        let av = Array1::from_vec(a.clone());
        let simd = scalar_mul_simd::<f32>(&av.view(), c);
        let reference: Vec<f32> = a.iter().map(|x| x * c).collect();
        prop_assert!(
            vec_f32_approx(simd.as_slice().expect("simd slice"), &reference),
            "scalar_mul_simd f32 mismatch"
        );
    }

    #[test]
    fn prop_scalar_mul_f64_matches_manual(
        a in prop::collection::vec(-1e100f64..1e100f64, 0..64),
        c in -1e100f64..1e100f64,
    ) {
        let av = Array1::from_vec(a.clone());
        let simd = scalar_mul_simd::<f64>(&av.view(), c);
        let reference: Vec<f64> = a.iter().map(|x| x * c).collect();
        prop_assert!(
            vec_f64_approx(simd.as_slice().expect("simd slice"), &reference),
            "scalar_mul_simd f64 mismatch"
        );
    }

    // -----------------------------------------------------------------------
    // add identity: a + 0 == a
    // -----------------------------------------------------------------------

    #[test]
    fn prop_add_f32_zero_identity(a in finite_f32_vec(64)) {
        let zeros: Vec<f32> = vec![0.0f32; a.len()];
        let av = Array1::from_vec(a.clone());
        let zv = Array1::from_vec(zeros);
        let result = add_simd::<f32>(&av.view(), &zv.view());
        prop_assert!(
            vec_f32_approx(result.as_slice().expect("result slice"), &a),
            "a + 0 != a"
        );
    }

    // -----------------------------------------------------------------------
    // mul identity: a * 1 == a
    // -----------------------------------------------------------------------

    #[test]
    fn prop_mul_f32_identity(a in finite_f32_vec(64)) {
        let ones: Vec<f32> = vec![1.0f32; a.len()];
        let av = Array1::from_vec(a.clone());
        let ov = Array1::from_vec(ones);
        let result = mul_simd::<f32>(&av.view(), &ov.view());
        prop_assert!(
            vec_f32_approx(result.as_slice().expect("result slice"), &a),
            "a * 1 != a"
        );
    }

    // -----------------------------------------------------------------------
    // mul by zero: a * 0 == 0 (for bounded finite inputs)
    // -----------------------------------------------------------------------

    #[test]
    fn prop_mul_f32_zero_annihilation(
        a in prop::collection::vec(-1e10f32..1e10f32, 0..64),
    ) {
        let zeros: Vec<f32> = vec![0.0f32; a.len()];
        let av = Array1::from_vec(a.clone());
        let zv = Array1::from_vec(zeros);
        let result = mul_simd::<f32>(&av.view(), &zv.view());
        for v in result.iter() {
            prop_assert!(*v == 0.0, "a * 0 should be 0, got {}", v);
        }
    }

    // -----------------------------------------------------------------------
    // Overflow: MAX + MAX -> Inf
    // -----------------------------------------------------------------------

    #[test]
    fn prop_add_f32_overflow(_dummy in Just(())) {
        let av = Array1::from_vec(vec![f32::MAX]);
        let bv = Array1::from_vec(vec![f32::MAX]);
        let result = add_simd::<f32>(&av.view(), &bv.view());
        prop_assert!(result[0].is_infinite(), "MAX + MAX should overflow to Inf, got {}", result[0]);
    }

    #[test]
    fn prop_add_f64_overflow(_dummy in Just(())) {
        let av = Array1::from_vec(vec![f64::MAX]);
        let bv = Array1::from_vec(vec![f64::MAX]);
        let result = add_simd::<f64>(&av.view(), &bv.view());
        prop_assert!(result[0].is_infinite(), "MAX + MAX (f64) should overflow to Inf, got {}", result[0]);
    }

    // -----------------------------------------------------------------------
    // Division by zero: positive / 0 -> +Inf
    // -----------------------------------------------------------------------

    #[test]
    fn prop_div_f32_by_zero(a in (0.001f32..1e10f32)) {
        let av = Array1::from_vec(vec![a]);
        let bv = Array1::from_vec(vec![0.0f32]);
        let result = div_simd::<f32>(&av.view(), &bv.view());
        prop_assert!(
            result[0].is_infinite() && result[0] > 0.0,
            "positive / 0.0 should be +Inf, got {}",
            result[0]
        );
    }

    #[test]
    fn prop_div_f64_by_zero(a in (0.001f64..1e100f64)) {
        let av = Array1::from_vec(vec![a]);
        let bv = Array1::from_vec(vec![0.0f64]);
        let result = div_simd::<f64>(&av.view(), &bv.view());
        prop_assert!(
            result[0].is_infinite() && result[0] > 0.0,
            "positive / 0.0 (f64) should be +Inf, got {}",
            result[0]
        );
    }
}

// ---------------------------------------------------------------------------
// Fixed-input unit tests for specific IEEE 754 edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_abs_special_values_f32() {
    let special: Vec<f32> = vec![
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        f32::MIN_POSITIVE * 0.5, // subnormal
        -f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE * 0.5,
    ];
    let av = Array1::from_vec(special.clone());
    let result = abs_simd::<f32>(&av.view());
    let reference = ref_abs_f32(&special);
    for (r, e) in result.iter().zip(reference.iter()) {
        assert!(
            f32_eq_with_nan(*r, *e) || f32_approx(*r, *e),
            "abs special f32: got {}, expected {}",
            r,
            e
        );
    }
}

#[test]
fn test_abs_special_values_f64() {
    let special: Vec<f64> = vec![
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -0.0,
        f64::MIN_POSITIVE,
        f64::MIN_POSITIVE * 0.5,
        -f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE * 0.5,
    ];
    let av = Array1::from_vec(special.clone());
    let result = abs_simd::<f64>(&av.view());
    let reference = ref_abs_f64(&special);
    for (r, e) in result.iter().zip(reference.iter()) {
        assert!(
            f64_eq_with_nan(*r, *e) || f64_approx(*r, *e),
            "abs special f64: got {}, expected {}",
            r,
            e
        );
    }
}

#[test]
fn test_abs_negative_zero_f32() {
    let av = Array1::from_vec(vec![-0.0f32]);
    let result = abs_simd::<f32>(&av.view());
    // abs(-0.0) should be 0.0 in IEEE 754
    assert_eq!(result[0], 0.0f32);
    assert!(!result[0].is_sign_negative(), "abs(-0.0) should be +0.0");
}

#[test]
fn test_abs_negative_zero_f64() {
    let av = Array1::from_vec(vec![-0.0f64]);
    let result = abs_simd::<f64>(&av.view());
    assert_eq!(result[0], 0.0f64);
    assert!(!result[0].is_sign_negative(), "abs(-0.0) should be +0.0");
}

#[test]
fn test_abs_subnormal_f32() {
    let subnormal = f32::MIN_POSITIVE * 0.5;
    let data = vec![subnormal, -subnormal];
    let av = Array1::from_vec(data.clone());
    let result = abs_simd::<f32>(&av.view());
    assert!(
        f32_approx(result[0], subnormal),
        "abs(subnormal) mismatch: {} vs {}",
        result[0],
        subnormal
    );
    assert!(
        f32_approx(result[1], subnormal),
        "abs(-subnormal) mismatch: {} vs {}",
        result[1],
        subnormal
    );
}

#[test]
fn test_abs_subnormal_f64() {
    let subnormal = f64::MIN_POSITIVE * 0.5;
    let data = vec![subnormal, -subnormal];
    let av = Array1::from_vec(data.clone());
    let result = abs_simd::<f64>(&av.view());
    assert!(f64_approx(result[0], subnormal));
    assert!(f64_approx(result[1], subnormal));
}

#[test]
fn test_sum_empty_f32() {
    let av = Array1::<f32>::zeros(0);
    let result = sum_simd::<f32>(&av.view());
    assert_eq!(result, 0.0f32, "sum of empty array should be 0");
}

#[test]
fn test_sum_empty_f64() {
    let av = Array1::<f64>::zeros(0);
    let result = sum_simd::<f64>(&av.view());
    assert_eq!(result, 0.0f64, "sum of empty array should be 0");
}

#[test]
fn test_sum_single_element_f32() {
    let av = Array1::from_vec(vec![42.0f32]);
    assert_eq!(sum_simd::<f32>(&av.view()), 42.0f32);
}

#[test]
fn test_sum_single_element_f64() {
    let av = Array1::from_vec(vec![42.0f64]);
    assert_eq!(sum_simd::<f64>(&av.view()), 42.0f64);
}

#[test]
fn test_sum_with_inf_f32() {
    let data = vec![f32::INFINITY, 1.0, 2.0, 3.0];
    let av = Array1::from_vec(data);
    let result = sum_simd::<f32>(&av.view());
    assert!(
        result.is_infinite() && result > 0.0,
        "sum with +Inf should be +Inf"
    );
}

#[test]
fn test_sum_with_neg_inf_f32() {
    let data = vec![f32::NEG_INFINITY, 1.0, 2.0, 3.0];
    let av = Array1::from_vec(data);
    let result = sum_simd::<f32>(&av.view());
    assert!(
        result.is_infinite() && result < 0.0,
        "sum with -Inf should be -Inf"
    );
}

#[test]
fn test_sum_inf_plus_neg_inf_f32() {
    // +Inf + (-Inf) = NaN
    let data = vec![f32::INFINITY, f32::NEG_INFINITY];
    let av = Array1::from_vec(data);
    let result = sum_simd::<f32>(&av.view());
    assert!(result.is_nan(), "Inf + (-Inf) should be NaN");
}

#[test]
fn test_add_empty_arrays_f32() {
    let av = Array1::<f32>::zeros(0);
    let bv = Array1::<f32>::zeros(0);
    let result = add_simd::<f32>(&av.view(), &bv.view());
    assert_eq!(
        result.len(),
        0,
        "add of empty arrays should produce empty result"
    );
}

#[test]
fn test_add_empty_arrays_f64() {
    let av = Array1::<f64>::zeros(0);
    let bv = Array1::<f64>::zeros(0);
    let result = add_simd::<f64>(&av.view(), &bv.view());
    assert_eq!(
        result.len(),
        0,
        "add of empty arrays (f64) should produce empty result"
    );
}

#[test]
fn test_sqrt_zero_f32() {
    let av = Array1::from_vec(vec![0.0f32]);
    let result = sqrt_simd::<f32>(&av.view());
    assert_eq!(result[0], 0.0f32, "sqrt(0) should be 0");
}

#[test]
fn test_sqrt_negative_zero_f32() {
    // IEEE 754: sqrt(-0.0) = 0.0 (or -0.0 depending on implementation)
    let av = Array1::from_vec(vec![-0.0f32]);
    let result = sqrt_simd::<f32>(&av.view());
    // Either 0.0 or -0.0 is acceptable, but must NOT be NaN
    assert!(!result[0].is_nan(), "sqrt(-0.0) should not be NaN");
}

#[test]
fn test_sqrt_inf_f32() {
    let av = Array1::from_vec(vec![f32::INFINITY]);
    let result = sqrt_simd::<f32>(&av.view());
    assert_eq!(result[0], f32::INFINITY, "sqrt(Inf) should be Inf");
}

#[test]
fn test_sqrt_inf_f64() {
    let av = Array1::from_vec(vec![f64::INFINITY]);
    let result = sqrt_simd::<f64>(&av.view());
    assert_eq!(result[0], f64::INFINITY, "sqrt(Inf) should be Inf (f64)");
}

#[test]
fn test_floor_special_values_f32() {
    let special: Vec<f32> = vec![
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        1.5,
        -1.5,
        0.9,
        -0.9,
        f32::MIN_POSITIVE,
    ];
    let av = Array1::from_vec(special.clone());
    let result = floor_simd::<f32>(&av.view());
    let reference = ref_floor_f32(&special);
    for (r, e) in result.iter().zip(reference.iter()) {
        assert!(
            f32_eq_with_nan(*r, *e) || f32_approx(*r, *e),
            "floor special f32: got {}, expected {}",
            r,
            e
        );
    }
}

#[test]
fn test_floor_special_values_f64() {
    let special: Vec<f64> = vec![
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -0.0,
        1.5,
        -1.5,
        0.9,
        -0.9,
    ];
    let av = Array1::from_vec(special.clone());
    let result = floor_simd::<f64>(&av.view());
    let reference = ref_floor_f64(&special);
    for (r, e) in result.iter().zip(reference.iter()) {
        assert!(
            f64_eq_with_nan(*r, *e) || f64_approx(*r, *e),
            "floor special f64: got {}, expected {}",
            r,
            e
        );
    }
}

#[test]
fn test_min_max_empty_returns_none_f32() {
    let av = Array1::<f32>::zeros(0);
    assert!(
        min_simd::<f32>(&av.view()).is_none(),
        "min of empty should be None"
    );
    assert!(
        max_simd::<f32>(&av.view()).is_none(),
        "max of empty should be None"
    );
}

#[test]
fn test_min_max_single_element_f32() {
    let av = Array1::from_vec(vec![42.0f32]);
    assert_eq!(min_simd::<f32>(&av.view()), Some(42.0f32));
    assert_eq!(max_simd::<f32>(&av.view()), Some(42.0f32));
}

#[test]
fn test_min_max_single_element_f64() {
    let av = Array1::from_vec(vec![42.0f64]);
    assert_eq!(min_simd::<f64>(&av.view()), Some(42.0f64));
    assert_eq!(max_simd::<f64>(&av.view()), Some(42.0f64));
}

#[test]
fn test_dot_orthogonal_f32() {
    let a = Array1::from_vec(vec![1.0f32, 0.0, 0.0]);
    let b = Array1::from_vec(vec![0.0f32, 1.0, 0.0]);
    let result = dot_simd::<f32>(&a.view(), &b.view());
    assert_eq!(result, 0.0f32, "dot of orthogonal vectors should be 0");
}

#[test]
fn test_dot_unit_vector_self_f64() {
    let scale = (3.0f64).sqrt().recip();
    let a = Array1::from_vec(vec![scale, scale, scale]);
    let result = dot_simd::<f64>(&a.view(), &a.view());
    assert!(
        (result - 1.0f64).abs() < 1e-10,
        "dot of unit vector with itself should be ~1.0: got {}",
        result
    );
}

#[test]
fn test_dot_empty_f32() {
    let av = Array1::<f32>::zeros(0);
    let bv = Array1::<f32>::zeros(0);
    let result = dot_simd::<f32>(&av.view(), &bv.view());
    assert_eq!(result, 0.0f32, "dot of empty arrays should be 0");
}

#[test]
fn test_mul_with_subnormals_f32() {
    let subnormal = f32::MIN_POSITIVE * 0.5;
    let av = Array1::from_vec(vec![subnormal, subnormal]);
    let bv = Array1::from_vec(vec![2.0f32, 0.5f32]);
    let result = mul_simd::<f32>(&av.view(), &bv.view());
    // subnormal * 2 may stay subnormal or become MIN_POSITIVE
    let exp0 = subnormal * 2.0;
    let exp1 = subnormal * 0.5; // may underflow to 0
    assert!(
        f32_approx(result[0], exp0),
        "subnormal * 2: got {}, expected {}",
        result[0],
        exp0
    );
    assert!(
        f32_approx(result[1], exp1),
        "subnormal * 0.5: got {}, expected {}",
        result[1],
        exp1
    );
}

#[test]
fn test_div_large_values_overflow_f32() {
    let av = Array1::from_vec(vec![f32::MAX]);
    let bv = Array1::from_vec(vec![f32::MIN_POSITIVE]);
    let result = div_simd::<f32>(&av.view(), &bv.view());
    assert!(
        result[0].is_infinite(),
        "MAX / MIN_POSITIVE should overflow to Inf"
    );
}

#[test]
fn test_div_zero_over_zero_f32() {
    let av = Array1::from_vec(vec![0.0f32]);
    let bv = Array1::from_vec(vec![0.0f32]);
    let result = div_simd::<f32>(&av.view(), &bv.view());
    assert!(result[0].is_nan(), "0/0 should be NaN, got {}", result[0]);
}

#[test]
fn test_add_single_element_f32() {
    let av = Array1::from_vec(vec![42.0f32]);
    let bv = Array1::from_vec(vec![58.0f32]);
    let result = add_simd::<f32>(&av.view(), &bv.view());
    assert_eq!(result[0], 100.0f32, "42 + 58 should be 100");
}

#[test]
fn test_mean_empty_returns_none() {
    let av = Array1::<f32>::zeros(0);
    assert!(
        mean_simd::<f32>(&av.view()).is_none(),
        "mean of empty should be None"
    );
}

#[test]
fn test_mean_single_element_f32() {
    let av = Array1::from_vec(vec![7.0f32]);
    assert_eq!(
        mean_simd::<f32>(&av.view()),
        Some(7.0f32),
        "mean of single element should equal that element"
    );
}

#[test]
fn test_ceil_and_floor_integers_f32() {
    // For integer values, ceil and floor should equal the value itself
    let data: Vec<f32> = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let av = Array1::from_vec(data.clone());
    let floored = floor_simd::<f32>(&av.view());
    let ceiled = ceil_simd::<f32>(&av.view());
    for (i, &v) in data.iter().enumerate() {
        assert_eq!(floored[i], v, "floor({}) should be {}", v, v);
        assert_eq!(ceiled[i], v, "ceil({}) should be {}", v, v);
    }
}

#[test]
fn test_round_half_values_f32() {
    // 0.5 rounds to 1.0 in standard rounding
    let av = Array1::from_vec(vec![0.5f32, -0.5f32, 1.5f32, -1.5f32]);
    let result = round_simd::<f32>(&av.view());
    let reference = ref_round_f32(&[0.5f32, -0.5f32, 1.5f32, -1.5f32]);
    for (r, e) in result.iter().zip(reference.iter()) {
        assert!(
            f32_eq_with_nan(*r, *e) || f32_approx(*r, *e),
            "round_simd mismatch: got {}, expected {}",
            r,
            e
        );
    }
}
