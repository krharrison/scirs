//! Type conversion utilities for scientific computing
//!
//! This module provides safe numeric conversions with overflow checking,
//! array type conversions, complex number helpers, precision-controlled rounding,
//! and numeric classification utilities.
//!
//! # Safe Conversions
//!
//! - [`safe_cast`] - Cast between numeric types with overflow detection
//! - [`checked_cast`] - Returns `None` if the conversion would lose data
//!
//! # Array Conversions
//!
//! - [`array_f32_to_f64`] / [`array_f64_to_f32`] - Convert float arrays
//! - [`array_int_to_f64`] / [`array_int_to_f32`] - Integer to float array conversions
//!
//! # Complex Number Utilities
//!
//! - [`complex_from_polar`] - Create complex from magnitude and angle
//! - [`complex_magnitude`] / [`complex_phase`] - Extract components
//! - [`real_part`] / [`imag_part`] - Extract real/imaginary arrays
//!
//! # Precision
//!
//! - [`round_to_decimals`] / [`round_to_sigfigs`] - Precision-controlled rounding
//!
//! # Classification
//!
//! - [`classify_float`] - Classify a float as NaN, Inf, Subnormal, Zero, or Normal
//! - [`classify_array`] - Classify all elements of an array
//! - [`count_special_values`] - Count NaN, Inf, etc. in an array

use crate::error::{CoreError, CoreResult, ErrorContext};
use ::ndarray::{Array1, Array2, ArrayView1};
use num_complex::Complex;
use num_traits::{Float, FromPrimitive, NumCast, ToPrimitive, Zero};
use std::fmt::{Debug, Display};

// ---------------------------------------------------------------------------
// Safe numeric conversions
// ---------------------------------------------------------------------------

/// Safely cast a value from one numeric type to another with overflow detection.
///
/// Returns an error if the value cannot be exactly represented in the target type.
///
/// # Example
///
/// ```
/// use scirs2_core::type_convert::safe_cast;
///
/// let val: f64 = 42.0;
/// let result: i32 = safe_cast::<f64, i32>(val).expect("cast failed");
/// assert_eq!(result, 42);
/// ```
pub fn safe_cast<S, T>(value: S) -> CoreResult<T>
where
    S: NumCast + Display + Copy,
    T: NumCast + Display,
{
    NumCast::from(value).ok_or_else(|| {
        CoreError::TypeError(ErrorContext::new(format!(
            "Cannot safely cast value {} to target type",
            value
        )))
    })
}

/// Try to cast a value, returning `None` on failure (no error).
///
/// This is useful for optional conversions where you want to handle
/// the failure yourself.
pub fn checked_cast<S, T>(value: S) -> Option<T>
where
    S: NumCast + Copy,
    T: NumCast,
{
    NumCast::from(value)
}

/// Safely convert a u64 to usize (may fail on 16-bit platforms).
pub fn u64_to_usize(value: u64) -> CoreResult<usize> {
    usize::try_from(value).map_err(|_| {
        CoreError::TypeError(ErrorContext::new(format!(
            "u64 value {} too large for usize on this platform",
            value
        )))
    })
}

/// Safely convert an i64 to usize (fails on negative values).
pub fn i64_to_usize(value: i64) -> CoreResult<usize> {
    if value < 0 {
        return Err(CoreError::TypeError(ErrorContext::new(format!(
            "Cannot convert negative i64 value {} to usize",
            value
        ))));
    }
    usize::try_from(value).map_err(|_| {
        CoreError::TypeError(ErrorContext::new(format!(
            "i64 value {} too large for usize",
            value
        )))
    })
}

/// Safely convert a usize to u32 (may fail for large values on 64-bit).
pub fn usize_to_u32(value: usize) -> CoreResult<u32> {
    u32::try_from(value).map_err(|_| {
        CoreError::TypeError(ErrorContext::new(format!(
            "usize value {} too large for u32",
            value
        )))
    })
}

/// Safely convert f64 to i64 (rounds toward zero, fails if out of range).
pub fn f64_to_i64(value: f64) -> CoreResult<i64> {
    if value.is_nan() || value.is_infinite() {
        return Err(CoreError::TypeError(ErrorContext::new(format!(
            "Cannot convert {} to i64",
            value
        ))));
    }
    let truncated = value.trunc();
    if truncated < i64::MIN as f64 || truncated > i64::MAX as f64 {
        return Err(CoreError::TypeError(ErrorContext::new(format!(
            "f64 value {} out of range for i64",
            value
        ))));
    }
    Ok(truncated as i64)
}

/// Safely convert f64 to u64 (rounds toward zero, fails if negative or out of range).
pub fn f64_to_u64(value: f64) -> CoreResult<u64> {
    if value.is_nan() || value.is_infinite() || value < 0.0 {
        return Err(CoreError::TypeError(ErrorContext::new(format!(
            "Cannot convert {} to u64",
            value
        ))));
    }
    let truncated = value.trunc();
    if truncated > u64::MAX as f64 {
        return Err(CoreError::TypeError(ErrorContext::new(format!(
            "f64 value {} out of range for u64",
            value
        ))));
    }
    Ok(truncated as u64)
}

// ---------------------------------------------------------------------------
// Array conversions
// ---------------------------------------------------------------------------

/// Convert an `Array1<f32>` to `Array1<f64>`.
///
/// This is a widening conversion and is always lossless.
#[must_use]
pub fn array_f32_to_f64(arr: &Array1<f32>) -> Array1<f64> {
    arr.mapv(|x| x as f64)
}

/// Convert an `Array1<f64>` to `Array1<f32>`.
///
/// This is a narrowing conversion and may lose precision.
/// Returns an error if any value is outside f32 range.
pub fn array_f64_to_f32(arr: &Array1<f64>) -> CoreResult<Array1<f32>> {
    let mut result = Array1::<f32>::zeros(arr.len());
    for (i, &v) in arr.iter().enumerate() {
        if v.is_finite() && (v.abs() > f32::MAX as f64) {
            return Err(CoreError::TypeError(ErrorContext::new(format!(
                "Value {} at index {} exceeds f32 range",
                v, i
            ))));
        }
        result[i] = v as f32;
    }
    Ok(result)
}

/// Convert an `Array2<f32>` to `Array2<f64>` (lossless).
#[must_use]
pub fn array2_f32_to_f64(arr: &Array2<f32>) -> Array2<f64> {
    arr.mapv(|x| x as f64)
}

/// Convert an `Array2<f64>` to `Array2<f32>` (may lose precision).
pub fn array2_f64_to_f32(arr: &Array2<f64>) -> CoreResult<Array2<f32>> {
    let mut result = Array2::<f32>::zeros(arr.dim());
    for ((r, c), &v) in arr.indexed_iter() {
        if v.is_finite() && (v.abs() > f32::MAX as f64) {
            return Err(CoreError::TypeError(ErrorContext::new(format!(
                "Value {} at [{}, {}] exceeds f32 range",
                v, r, c
            ))));
        }
        result[[r, c]] = v as f32;
    }
    Ok(result)
}

/// Convert an integer `Array1` to `Array1<f64>`.
pub fn array_int_to_f64<I>(arr: &Array1<I>) -> CoreResult<Array1<f64>>
where
    I: ToPrimitive + Copy + Debug,
{
    let mut result = Array1::<f64>::zeros(arr.len());
    for (i, &v) in arr.iter().enumerate() {
        result[i] = v.to_f64().ok_or_else(|| {
            CoreError::TypeError(ErrorContext::new(format!(
                "Cannot convert {:?} at index {} to f64",
                v, i
            )))
        })?;
    }
    Ok(result)
}

/// Convert an integer `Array1` to `Array1<f32>`.
pub fn array_int_to_f32<I>(arr: &Array1<I>) -> CoreResult<Array1<f32>>
where
    I: ToPrimitive + Copy + Debug,
{
    let mut result = Array1::<f32>::zeros(arr.len());
    for (i, &v) in arr.iter().enumerate() {
        result[i] = v.to_f32().ok_or_else(|| {
            CoreError::TypeError(ErrorContext::new(format!(
                "Cannot convert {:?} at index {} to f32",
                v, i
            )))
        })?;
    }
    Ok(result)
}

/// Convert `Array1<f64>` to `Array1<i64>`, truncating toward zero.
pub fn array_f64_to_int(arr: &Array1<f64>) -> CoreResult<Array1<i64>> {
    let mut result = Array1::<i64>::zeros(arr.len());
    for (i, &v) in arr.iter().enumerate() {
        result[i] = f64_to_i64(v).map_err(|e| {
            CoreError::TypeError(ErrorContext::new(format!(
                "Conversion failed at index {}: {}",
                i, e
            )))
        })?;
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Complex number utilities
// ---------------------------------------------------------------------------

/// Create a complex number from polar coordinates (magnitude, phase_radians).
///
/// # Example
///
/// ```
/// use scirs2_core::type_convert::complex_from_polar;
///
/// let c = complex_from_polar(2.0_f64, std::f64::consts::FRAC_PI_2);
/// assert!((c.re).abs() < 1e-10); // ~ 0
/// assert!((c.im - 2.0).abs() < 1e-10); // ~ 2
/// ```
pub fn complex_from_polar<F: Float>(magnitude: F, phase: F) -> Complex<F> {
    Complex::new(magnitude * phase.cos(), magnitude * phase.sin())
}

/// Compute the magnitude (absolute value) of a complex number.
pub fn complex_magnitude<F: Float>(c: &Complex<F>) -> F {
    (c.re * c.re + c.im * c.im).sqrt()
}

/// Compute the phase (argument) of a complex number in radians.
pub fn complex_phase<F: Float>(c: &Complex<F>) -> F {
    c.im.atan2(c.re)
}

/// Extract real parts from an array of complex numbers.
pub fn real_part<F: Float + Zero>(arr: &Array1<Complex<F>>) -> Array1<F> {
    arr.mapv(|c| c.re)
}

/// Extract imaginary parts from an array of complex numbers.
pub fn imag_part<F: Float + Zero>(arr: &Array1<Complex<F>>) -> Array1<F> {
    arr.mapv(|c| c.im)
}

/// Create an array of complex numbers from separate real and imaginary arrays.
pub fn complex_from_parts<F: Float + Zero>(
    real: &Array1<F>,
    imag: &Array1<F>,
) -> CoreResult<Array1<Complex<F>>> {
    if real.len() != imag.len() {
        return Err(CoreError::DimensionError(ErrorContext::new(format!(
            "Real array length {} != imaginary array length {}",
            real.len(),
            imag.len()
        ))));
    }
    let mut result = Array1::<Complex<F>>::zeros(real.len());
    for i in 0..real.len() {
        result[i] = Complex::new(real[i], imag[i]);
    }
    Ok(result)
}

/// Compute magnitudes for an array of complex numbers.
pub fn complex_magnitudes<F: Float + Zero>(arr: &Array1<Complex<F>>) -> Array1<F> {
    arr.mapv(|c| (c.re * c.re + c.im * c.im).sqrt())
}

/// Compute phases for an array of complex numbers.
pub fn complex_phases<F: Float + Zero>(arr: &Array1<Complex<F>>) -> Array1<F> {
    arr.mapv(|c| c.im.atan2(c.re))
}

/// Convert `Complex<f32>` array to `Complex<f64>`.
#[must_use]
pub fn complex_f32_to_f64(arr: &Array1<Complex<f32>>) -> Array1<Complex<f64>> {
    arr.mapv(|c| Complex::new(c.re as f64, c.im as f64))
}

/// Convert `Complex<f64>` array to `Complex<f32>`.
#[must_use]
pub fn complex_f64_to_f32(arr: &Array1<Complex<f64>>) -> Array1<Complex<f32>> {
    arr.mapv(|c| Complex::new(c.re as f32, c.im as f32))
}

// ---------------------------------------------------------------------------
// Precision-controlled rounding
// ---------------------------------------------------------------------------

/// Round a float to a specified number of decimal places.
///
/// # Example
///
/// ```
/// use scirs2_core::type_convert::round_to_decimals;
///
/// assert!((round_to_decimals(1.23456_f64, 2) - 1.23_f64).abs() < 1e-12_f64);
/// assert!((round_to_decimals(1.23456_f64, 4) - 1.2346_f64).abs() < 1e-12_f64);
/// ```
pub fn round_to_decimals<F: Float + FromPrimitive>(value: F, decimals: u32) -> F {
    let factor = F::from_f64(10.0f64.powi(decimals as i32)).unwrap_or_else(F::one);
    (value * factor).round() / factor
}

/// Round a float to a specified number of significant figures.
///
/// # Example
///
/// ```
/// use scirs2_core::type_convert::round_to_sigfigs;
///
/// assert!((round_to_sigfigs(12345.6789_f64, 3) - 12300.0_f64).abs() < 1e-6_f64);
/// assert!((round_to_sigfigs(0.0012345_f64, 2) - 0.0012_f64).abs() < 1e-8_f64);
/// ```
pub fn round_to_sigfigs<F: Float + FromPrimitive>(value: F, sigfigs: u32) -> F {
    if value.is_zero() || value.is_nan() || value.is_infinite() {
        return value;
    }
    let abs_val = value.abs();
    let log10_val = abs_val.log10().floor();
    let shift = F::from_u32(sigfigs).unwrap_or_else(F::one) - F::one() - log10_val;
    let factor = F::from_f64(10.0f64).unwrap_or_else(F::one).powf(shift);
    (value * factor).round() / factor
}

/// Round an array element-wise to a given number of decimal places.
pub fn round_array_to_decimals<F: Float + FromPrimitive>(
    arr: &Array1<F>,
    decimals: u32,
) -> Array1<F> {
    arr.mapv(|v| round_to_decimals(v, decimals))
}

/// Truncate a float toward zero to a given number of decimal places.
pub fn truncate_to_decimals<F: Float + FromPrimitive>(value: F, decimals: u32) -> F {
    let factor = F::from_f64(10.0f64.powi(decimals as i32)).unwrap_or_else(F::one);
    (value * factor).trunc() / factor
}

/// Ceiling to a given number of decimal places.
pub fn ceil_to_decimals<F: Float + FromPrimitive>(value: F, decimals: u32) -> F {
    let factor = F::from_f64(10.0f64.powi(decimals as i32)).unwrap_or_else(F::one);
    (value * factor).ceil() / factor
}

/// Floor to a given number of decimal places.
pub fn floor_to_decimals<F: Float + FromPrimitive>(value: F, decimals: u32) -> F {
    let factor = F::from_f64(10.0f64.powi(decimals as i32)).unwrap_or_else(F::one);
    (value * factor).floor() / factor
}

// ---------------------------------------------------------------------------
// Float classification
// ---------------------------------------------------------------------------

/// Classification of a floating-point value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatClass {
    /// Not a number
    Nan,
    /// Positive or negative infinity
    Infinite,
    /// A subnormal (denormalized) number
    Subnormal,
    /// Positive or negative zero
    Zero,
    /// A normal floating-point number
    Normal,
}

impl std::fmt::Display for FloatClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FloatClass::Nan => write!(f, "NaN"),
            FloatClass::Infinite => write!(f, "Infinite"),
            FloatClass::Subnormal => write!(f, "Subnormal"),
            FloatClass::Zero => write!(f, "Zero"),
            FloatClass::Normal => write!(f, "Normal"),
        }
    }
}

/// Classify a floating-point value.
///
/// # Example
///
/// ```
/// use scirs2_core::type_convert::{classify_float, FloatClass};
///
/// assert_eq!(classify_float(1.0_f64), FloatClass::Normal);
/// assert_eq!(classify_float(f64::NAN), FloatClass::Nan);
/// assert_eq!(classify_float(f64::INFINITY), FloatClass::Infinite);
/// assert_eq!(classify_float(0.0_f64), FloatClass::Zero);
/// ```
pub fn classify_float<F: Float>(value: F) -> FloatClass {
    if value.is_nan() {
        FloatClass::Nan
    } else if value.is_infinite() {
        FloatClass::Infinite
    } else if value.is_zero() {
        FloatClass::Zero
    } else {
        match value.classify() {
            std::num::FpCategory::Subnormal => FloatClass::Subnormal,
            _ => FloatClass::Normal,
        }
    }
}

/// Classify all elements of a 1D array.
pub fn classify_array<F: Float>(arr: &Array1<F>) -> Array1<FloatClass> {
    let classes: Vec<FloatClass> = arr.iter().map(|&v| classify_float(v)).collect();
    Array1::from_vec(classes)
}

/// Count special values in a 1D float array.
///
/// Returns (n_nan, n_inf, n_subnormal, n_zero, n_normal).
///
/// # Example
///
/// ```
/// use scirs2_core::type_convert::count_special_values;
/// use ndarray::array;
///
/// let arr = array![1.0, f64::NAN, f64::INFINITY, 0.0, -f64::INFINITY];
/// let (nan, inf, sub, zero, normal) = count_special_values(&arr);
/// assert_eq!(nan, 1);
/// assert_eq!(inf, 2);
/// assert_eq!(zero, 1);
/// assert_eq!(normal, 1);
/// ```
pub fn count_special_values<F: Float>(arr: &Array1<F>) -> (usize, usize, usize, usize, usize) {
    let mut n_nan = 0;
    let mut n_inf = 0;
    let mut n_sub = 0;
    let mut n_zero = 0;
    let mut n_normal = 0;
    for &v in arr.iter() {
        match classify_float(v) {
            FloatClass::Nan => n_nan += 1,
            FloatClass::Infinite => n_inf += 1,
            FloatClass::Subnormal => n_sub += 1,
            FloatClass::Zero => n_zero += 1,
            FloatClass::Normal => n_normal += 1,
        }
    }
    (n_nan, n_inf, n_sub, n_zero, n_normal)
}

/// Check if an array contains any NaN values.
pub fn has_nan<F: Float>(arr: &Array1<F>) -> bool {
    arr.iter().any(|v| v.is_nan())
}

/// Check if an array contains any infinite values.
pub fn has_inf<F: Float>(arr: &Array1<F>) -> bool {
    arr.iter().any(|v| v.is_infinite())
}

/// Check if all values in an array are finite (no NaN, no Inf).
pub fn all_finite<F: Float>(arr: &Array1<F>) -> bool {
    arr.iter().all(|v| v.is_finite())
}

/// Replace NaN values in an array with a specified value.
#[must_use]
pub fn nan_to_num<F: Float>(arr: &Array1<F>, nan_value: F, pos_inf: F, neg_inf: F) -> Array1<F> {
    arr.mapv(|v| {
        if v.is_nan() {
            nan_value
        } else if v.is_infinite() {
            if v > F::zero() {
                pos_inf
            } else {
                neg_inf
            }
        } else {
            v
        }
    })
}

/// Clamp array values to a range [min_val, max_val].
///
/// NaN values are preserved.
#[must_use]
pub fn clamp_array<F: Float>(arr: &Array1<F>, min_val: F, max_val: F) -> Array1<F> {
    arr.mapv(|v| {
        if v.is_nan() {
            v
        } else if v < min_val {
            min_val
        } else if v > max_val {
            max_val
        } else {
            v
        }
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ::ndarray::array;

    const EPS: f64 = 1e-10;

    #[test]
    fn test_safe_cast_f64_to_i32() {
        let result: i32 = safe_cast::<f64, i32>(42.0).expect("cast");
        assert_eq!(result, 42);
    }

    #[test]
    fn test_safe_cast_overflow() {
        let result = safe_cast::<f64, i8>(200.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_checked_cast_success() {
        let result: Option<u32> = checked_cast::<i64, u32>(100);
        assert_eq!(result, Some(100));
    }

    #[test]
    fn test_checked_cast_failure() {
        let result: Option<u32> = checked_cast::<i64, u32>(-1);
        assert_eq!(result, None);
    }

    #[test]
    fn test_u64_to_usize() {
        let r = u64_to_usize(42).expect("convert");
        assert_eq!(r, 42);
    }

    #[test]
    fn test_i64_to_usize_negative() {
        assert!(i64_to_usize(-1).is_err());
    }

    #[test]
    fn test_usize_to_u32_ok() {
        let r = usize_to_u32(100).expect("convert");
        assert_eq!(r, 100);
    }

    #[test]
    fn test_f64_to_i64_basic() {
        assert_eq!(f64_to_i64(42.9).expect("convert"), 42);
        assert_eq!(f64_to_i64(-3.7).expect("convert"), -3);
    }

    #[test]
    fn test_f64_to_i64_nan() {
        assert!(f64_to_i64(f64::NAN).is_err());
    }

    #[test]
    fn test_f64_to_u64_negative() {
        assert!(f64_to_u64(-1.0).is_err());
    }

    #[test]
    fn test_array_f32_to_f64() {
        let arr = array![1.0f32, 2.0, 3.0];
        let result = array_f32_to_f64(&arr);
        assert!((result[0] - 1.0).abs() < EPS);
        assert!((result[2] - 3.0).abs() < EPS);
    }

    #[test]
    fn test_array_f64_to_f32_ok() {
        let arr = array![1.0f64, 2.5, -3.0];
        let result = array_f64_to_f32(&arr).expect("convert");
        assert!((result[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_array_f64_to_f32_overflow() {
        let arr = array![1e39f64]; // exceeds f32 range
        assert!(array_f64_to_f32(&arr).is_err());
    }

    #[test]
    fn test_array2_conversions() {
        let arr32 = ::ndarray::Array2::<f32>::zeros((2, 3));
        let arr64 = array2_f32_to_f64(&arr32);
        assert_eq!(arr64.dim(), (2, 3));
        let back = array2_f64_to_f32(&arr64).expect("convert");
        assert_eq!(back.dim(), (2, 3));
    }

    #[test]
    fn test_array_int_to_f64() {
        let arr = array![1i32, 2, 3, -4];
        let result = array_int_to_f64(&arr).expect("convert");
        assert!((result[3] - (-4.0)).abs() < EPS);
    }

    #[test]
    fn test_array_int_to_f32() {
        let arr = array![10i64, 20, 30];
        let result = array_int_to_f32(&arr).expect("convert");
        assert!((result[1] - 20.0).abs() < 1e-5);
    }

    #[test]
    fn test_array_f64_to_int() {
        let arr = array![1.9, 2.1, -3.5];
        let result = array_f64_to_int(&arr).expect("convert");
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], -3);
    }

    #[test]
    fn test_complex_from_polar() {
        let c = complex_from_polar(1.0_f64, 0.0);
        assert!((c.re - 1.0).abs() < EPS);
        assert!(c.im.abs() < EPS);

        let c2 = complex_from_polar(2.0_f64, std::f64::consts::FRAC_PI_2);
        assert!(c2.re.abs() < EPS);
        assert!((c2.im - 2.0).abs() < EPS);
    }

    #[test]
    fn test_complex_magnitude_phase() {
        let c = Complex::new(3.0_f64, 4.0);
        assert!((complex_magnitude(&c) - 5.0).abs() < EPS);
        let c2 = Complex::new(0.0_f64, 1.0);
        assert!((complex_phase(&c2) - std::f64::consts::FRAC_PI_2).abs() < EPS);
    }

    #[test]
    fn test_real_imag_parts() {
        let arr = array![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let re = real_part(&arr);
        let im = imag_part(&arr);
        assert!((re[0] - 1.0).abs() < EPS);
        assert!((im[1] - 4.0).abs() < EPS);
    }

    #[test]
    fn test_complex_from_parts() {
        let re = array![1.0, 2.0];
        let im = array![3.0, 4.0];
        let c = complex_from_parts(&re, &im).expect("combine");
        assert!((c[0].re - 1.0).abs() < EPS);
        assert!((c[1].im - 4.0).abs() < EPS);
    }

    #[test]
    fn test_complex_from_parts_mismatch() {
        let re = array![1.0, 2.0];
        let im = array![3.0];
        assert!(complex_from_parts(&re, &im).is_err());
    }

    #[test]
    fn test_complex_magnitudes_phases() {
        let arr = array![Complex::new(3.0, 4.0), Complex::new(0.0, 1.0)];
        let mags = complex_magnitudes(&arr);
        assert!((mags[0] - 5.0).abs() < EPS);
        let phases = complex_phases(&arr);
        assert!((phases[1] - std::f64::consts::FRAC_PI_2).abs() < EPS);
    }

    #[test]
    fn test_complex_f32_f64_conversion() {
        let arr32 = array![Complex::new(1.0f32, 2.0)];
        let arr64 = complex_f32_to_f64(&arr32);
        assert!((arr64[0].re - 1.0).abs() < EPS);
        let back = complex_f64_to_f32(&arr64);
        assert!((back[0].im - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_round_to_decimals() {
        assert!((round_to_decimals(1.23456, 2) - 1.23).abs() < 1e-12);
        assert!((round_to_decimals(1.23456, 4) - 1.2346).abs() < 1e-12);
        assert!((round_to_decimals(1.23456, 0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_round_to_sigfigs() {
        assert!((round_to_sigfigs(12345.6789, 3) - 12300.0).abs() < 1.0);
        assert!((round_to_sigfigs(0.0012345, 2) - 0.0012).abs() < 1e-6);
    }

    #[test]
    fn test_round_to_sigfigs_zero() {
        assert_eq!(round_to_sigfigs(0.0_f64, 3), 0.0);
    }

    #[test]
    fn test_round_array_to_decimals() {
        let arr = array![1.234, 5.678, 9.012];
        let rounded = round_array_to_decimals(&arr, 1);
        assert!((rounded[0] - 1.2).abs() < 1e-12);
        assert!((rounded[1] - 5.7).abs() < 1e-12);
    }

    #[test]
    fn test_truncate_to_decimals() {
        assert!((truncate_to_decimals(1.239, 2) - 1.23).abs() < 1e-12);
        assert!((truncate_to_decimals(-1.239, 2) - (-1.23)).abs() < 1e-12);
    }

    #[test]
    fn test_ceil_floor_to_decimals() {
        assert!((ceil_to_decimals(1.234, 2) - 1.24).abs() < 1e-12);
        assert!((floor_to_decimals(1.239, 2) - 1.23).abs() < 1e-12);
    }

    #[test]
    fn test_classify_float() {
        assert_eq!(classify_float(1.0_f64), FloatClass::Normal);
        assert_eq!(classify_float(0.0_f64), FloatClass::Zero);
        assert_eq!(classify_float(f64::NAN), FloatClass::Nan);
        assert_eq!(classify_float(f64::INFINITY), FloatClass::Infinite);
        assert_eq!(classify_float(f64::NEG_INFINITY), FloatClass::Infinite);
    }

    #[test]
    fn test_classify_subnormal() {
        let subnormal = f64::MIN_POSITIVE / 2.0;
        assert_eq!(classify_float(subnormal), FloatClass::Subnormal);
    }

    #[test]
    fn test_classify_array() {
        let arr = array![1.0, f64::NAN, 0.0];
        let classes = classify_array(&arr);
        assert_eq!(classes[0], FloatClass::Normal);
        assert_eq!(classes[1], FloatClass::Nan);
        assert_eq!(classes[2], FloatClass::Zero);
    }

    #[test]
    fn test_count_special_values() {
        let arr = array![1.0, f64::NAN, f64::INFINITY, 0.0, -f64::INFINITY, 2.0];
        let (nan, inf, _sub, zero, normal) = count_special_values(&arr);
        assert_eq!(nan, 1);
        assert_eq!(inf, 2);
        assert_eq!(zero, 1);
        assert_eq!(normal, 2);
    }

    #[test]
    fn test_has_nan_inf() {
        let clean = array![1.0, 2.0, 3.0];
        assert!(!has_nan(&clean));
        assert!(!has_inf(&clean));
        assert!(all_finite(&clean));

        let with_nan = array![1.0, f64::NAN];
        assert!(has_nan(&with_nan));
        assert!(!all_finite(&with_nan));

        let with_inf = array![1.0, f64::INFINITY];
        assert!(has_inf(&with_inf));
    }

    #[test]
    fn test_nan_to_num() {
        let arr = array![1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        let clean = nan_to_num(&arr, 0.0, f64::MAX, f64::MIN);
        assert!((clean[0] - 1.0).abs() < EPS);
        assert!((clean[1] - 0.0).abs() < EPS);
        assert_eq!(clean[2], f64::MAX);
        assert_eq!(clean[3], f64::MIN);
    }

    #[test]
    fn test_clamp_array() {
        let arr = array![-5.0, 0.0, 5.0, 10.0, f64::NAN];
        let clamped = clamp_array(&arr, 0.0, 8.0);
        assert!((clamped[0] - 0.0).abs() < EPS);
        assert!((clamped[2] - 5.0).abs() < EPS);
        assert!((clamped[3] - 8.0).abs() < EPS);
        assert!(clamped[4].is_nan()); // NaN preserved
    }

    #[test]
    fn test_float_class_display() {
        assert_eq!(format!("{}", FloatClass::Nan), "NaN");
        assert_eq!(format!("{}", FloatClass::Normal), "Normal");
    }
}
