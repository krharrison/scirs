//! Numeric operation traits and type-dispatch helpers for CPU GPU fallback.
//!
//! This submodule defines the [`NumericOps`] and [`FloatOps`] helper traits
//! used internally by the CPU fallback implementations, along with generic
//! type-dispatch utilities.

use super::super::{GpuDataType, GpuError};
use std::any::TypeId;

pub(super) trait NumericOps: GpuDataType {
    fn zero() -> Self;
    fn one() -> Self;
    fn from_f64(val: f64) -> Self;
    fn to_f64(self) -> f64;
    fn add(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn neg(self) -> Self;
    fn min_value() -> Self;
    fn max_value() -> Self;
    fn partial_lt(self, other: Self) -> bool;
}

impl NumericOps for f32 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
    #[inline]
    fn from_f64(val: f64) -> Self {
        val as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn add(self, other: Self) -> Self {
        self + other
    }
    #[inline]
    fn mul(self, other: Self) -> Self {
        self * other
    }
    #[inline]
    fn neg(self) -> Self {
        -self
    }
    #[inline]
    fn min_value() -> Self {
        f32::NEG_INFINITY
    }
    #[inline]
    fn max_value() -> Self {
        f32::INFINITY
    }
    #[inline]
    fn partial_lt(self, other: Self) -> bool {
        self < other
    }
}

impl NumericOps for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
    #[inline]
    fn from_f64(val: f64) -> Self {
        val
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn add(self, other: Self) -> Self {
        self + other
    }
    #[inline]
    fn mul(self, other: Self) -> Self {
        self * other
    }
    #[inline]
    fn neg(self) -> Self {
        -self
    }
    #[inline]
    fn min_value() -> Self {
        f64::NEG_INFINITY
    }
    #[inline]
    fn max_value() -> Self {
        f64::INFINITY
    }
    #[inline]
    fn partial_lt(self, other: Self) -> bool {
        self < other
    }
}

macro_rules! impl_numeric_ops_int {
    ($t:ty) => {
        impl NumericOps for $t {
            #[inline]
            fn zero() -> Self {
                0
            }
            #[inline]
            fn one() -> Self {
                1
            }
            #[inline]
            fn from_f64(val: f64) -> Self {
                val as $t
            }
            #[inline]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline]
            fn add(self, other: Self) -> Self {
                self.wrapping_add(other)
            }
            #[inline]
            fn mul(self, other: Self) -> Self {
                self.wrapping_mul(other)
            }
            #[inline]
            fn neg(self) -> Self {
                self.wrapping_neg()
            }
            #[inline]
            fn min_value() -> Self {
                <$t>::MIN
            }
            #[inline]
            fn max_value() -> Self {
                <$t>::MAX
            }
            #[inline]
            fn partial_lt(self, other: Self) -> bool {
                self < other
            }
        }
    };
}

impl_numeric_ops_int!(i8);
impl_numeric_ops_int!(i16);
impl_numeric_ops_int!(i32);
impl_numeric_ops_int!(i64);
impl_numeric_ops_int!(isize);

macro_rules! impl_numeric_ops_uint {
    ($t:ty) => {
        impl NumericOps for $t {
            #[inline]
            fn zero() -> Self {
                0
            }
            #[inline]
            fn one() -> Self {
                1
            }
            #[inline]
            fn from_f64(val: f64) -> Self {
                val as $t
            }
            #[inline]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline]
            fn add(self, other: Self) -> Self {
                self.wrapping_add(other)
            }
            #[inline]
            fn mul(self, other: Self) -> Self {
                self.wrapping_mul(other)
            }
            #[inline]
            fn neg(self) -> Self {
                // For unsigned, wrapping_neg gives two's complement
                self.wrapping_neg()
            }
            #[inline]
            fn min_value() -> Self {
                <$t>::MIN
            }
            #[inline]
            fn max_value() -> Self {
                <$t>::MAX
            }
            #[inline]
            fn partial_lt(self, other: Self) -> bool {
                self < other
            }
        }
    };
}

impl_numeric_ops_uint!(u8);
impl_numeric_ops_uint!(u16);
impl_numeric_ops_uint!(u32);
impl_numeric_ops_uint!(u64);
impl_numeric_ops_uint!(usize);

/// Helper trait for floating-point specific operations (activation functions, etc.)
pub(super) trait FloatOps: NumericOps {
    fn exp(self) -> Self;
    fn tanh_val(self) -> Self;
    fn sqrt(self) -> Self;
    fn erf(self) -> Self;
    fn max(self, other: Self) -> Self;
    fn div(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
}

impl FloatOps for f32 {
    #[inline]
    fn exp(self) -> Self {
        f32::exp(self)
    }
    #[inline]
    fn tanh_val(self) -> Self {
        f32::tanh(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    #[inline]
    fn erf(self) -> Self {
        erf_f64(self as f64) as f32
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        f32::max(self, other)
    }
    #[inline]
    fn div(self, other: Self) -> Self {
        self / other
    }
    #[inline]
    fn sub(self, other: Self) -> Self {
        self - other
    }
}

impl FloatOps for f64 {
    #[inline]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline]
    fn tanh_val(self) -> Self {
        f64::tanh(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline]
    fn erf(self) -> Self {
        erf_f64(self)
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }
    #[inline]
    fn div(self, other: Self) -> Self {
        self / other
    }
    #[inline]
    fn sub(self, other: Self) -> Self {
        self - other
    }
}

/// Approximation of the error function using Abramowitz and Stegun formula 7.1.26.
/// Maximum error: 1.5e-7
pub(super) fn erf_f64(x: f64) -> f64 {
    // Return exact zero for zero input
    if x == 0.0 {
        return 0.0;
    }

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let ax = x.abs();

    // For very small |x|, use linear Taylor approximation: erf(x) ≈ (2/sqrt(pi)) * x
    // This avoids the approximation error at x=0.
    if ax < 1e-7 {
        let two_over_sqrt_pi = std::f64::consts::FRAC_2_SQRT_PI;
        return sign * two_over_sqrt_pi * ax;
    }

    // Abramowitz & Stegun formula 7.1.26 (max error: 1.5e-7)
    // Use Horner's method for numerical stability.
    let p = 0.3275911_f64;
    let a1 = 0.254829592_f64;
    let a2 = -0.284496736_f64;
    let a3 = 1.421413741_f64;
    let a4 = -1.453152027_f64;
    let a5 = 1.061405429_f64;

    let t = 1.0 / (1.0 + p * ax);
    // Evaluate polynomial via Horner's method: a1 + t*(a2 + t*(a3 + t*(a4 + t*a5)))
    let poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
    let result = 1.0 - poly * (-ax * ax).exp();

    sign * result
}

// =============================================================================
// Type-dispatch helpers
// =============================================================================

/// Dispatch a function to the appropriate numeric type based on TypeId.
/// Returns None if the type is not supported for numeric operations.
pub(super) fn dispatch_numeric<T: GpuDataType, F32Fn, F64Fn, IntFn>(
    f32_fn: F32Fn,
    f64_fn: F64Fn,
    int_fn: IntFn,
) -> Result<Vec<T>, GpuError>
where
    F32Fn: FnOnce() -> Vec<f32>,
    F64Fn: FnOnce() -> Vec<f64>,
    IntFn: FnOnce() -> Result<Vec<T>, GpuError>,
{
    let type_id = TypeId::of::<T>();

    if type_id == TypeId::of::<f32>() {
        let result = f32_fn();
        // Safe: We verified T == f32
        let result_bytes = result.as_ptr() as *const T;
        let len = result.len();
        let mut output = Vec::with_capacity(len);
        unsafe {
            std::ptr::copy_nonoverlapping(result_bytes, output.as_mut_ptr(), len);
            output.set_len(len);
        }
        // Don't drop `result` as bytes since we copied
        std::mem::forget(result);
        Ok(output)
    } else if type_id == TypeId::of::<f64>() {
        let result = f64_fn();
        let result_bytes = result.as_ptr() as *const T;
        let len = result.len();
        let mut output = Vec::with_capacity(len);
        unsafe {
            std::ptr::copy_nonoverlapping(result_bytes, output.as_mut_ptr(), len);
            output.set_len(len);
        }
        std::mem::forget(result);
        Ok(output)
    } else {
        int_fn()
    }
}

/// Reinterpret a Vec<T> as Vec<U> when T and U have the same size and alignment.
///
/// # Safety
///
/// This is only safe when T and U are the same type (verified by TypeId check
/// in the caller). The caller must ensure the reinterpretation is valid.
pub(super) unsafe fn reinterpret_vec<T, U>(v: Vec<T>) -> Vec<U> {
    debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<U>());
    debug_assert_eq!(std::mem::align_of::<T>(), std::mem::align_of::<U>());
    let mut v = std::mem::ManuallyDrop::new(v);
    let ptr = v.as_mut_ptr() as *mut U;
    let len = v.len();
    let cap = v.capacity();
    Vec::from_raw_parts(ptr, len, cap)
}
