//! Type coercion utilities for safe and lossy numeric conversions
//!
//! This module provides the [`Coerce`] trait for converting between compatible
//! numeric types together with a [`CoercionError`] type that captures source
//! and target type information for rich error messages.
//!
//! ## Safety model
//!
//! - **safe coercions** (`i8` → `i16` → `i32` → `i64`, etc.) never lose
//!   information and always succeed.
//! - **checked coercions** (e.g. `f64` → `i32`, `i64` → `u8`) may overflow or
//!   lose fractional parts; they return `Err(CoercionError)` when the value
//!   cannot be represented exactly in the target type.
//! - A convenience [`is_lossy`] function detects potential precision loss
//!   without performing the actual conversion.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::validation::coercion::{Coerce, is_lossy};
//!
//! // Safe widening – always succeeds
//! let wide: f64 = i32::coerce_from(42_i32).expect("should succeed");
//! assert_eq!(wide, 42.0_f64);
//!
//! // Checked narrowing – may fail
//! let ok: Result<i32, _> = i32::coerce_from(3.0_f64);
//! assert!(ok.is_ok());
//!
//! let fail: Result<i32, _> = i32::coerce_from(3.7_f64);
//! assert!(fail.is_err()); // fractional part
//!
//! // Lossiness probe
//! assert!(!is_lossy::<i32, f64>());  // i32 → f64 is safe
//! assert!(is_lossy::<f64, i32>());   // f64 → i32 can be lossy
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// CoercionError
// ---------------------------------------------------------------------------

/// Error produced when a type coercion cannot be performed without loss.
#[derive(Debug, Clone, PartialEq)]
pub struct CoercionError {
    /// Name of the source type.
    pub source_type: &'static str,
    /// Name of the target type.
    pub target_type: &'static str,
    /// Stringified value that failed to coerce.
    pub value: String,
    /// Human-readable reason for the failure.
    pub reason: String,
}

impl CoercionError {
    /// Construct a new `CoercionError`.
    pub fn new(
        source_type: &'static str,
        target_type: &'static str,
        value: impl fmt::Display,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            source_type,
            target_type,
            value: value.to_string(),
            reason: reason.into(),
        }
    }
}

impl fmt::Display for CoercionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Cannot coerce {} from {} to {}: {}",
            self.value, self.source_type, self.target_type, self.reason
        )
    }
}

impl std::error::Error for CoercionError {}

// ---------------------------------------------------------------------------
// Coerce trait
// ---------------------------------------------------------------------------

/// Convert a value of type `From` into type `Self`.
///
/// Implementations must return `Err(CoercionError)` when the conversion would
/// lose information (overflow, truncation of fractional part, etc.).
pub trait Coerce<From>: Sized {
    /// Try to coerce `value` into `Self`.
    ///
    /// # Errors
    ///
    /// Returns `CoercionError` when the value cannot be represented exactly.
    fn coerce_from(value: From) -> Result<Self, CoercionError>;
}

// ---------------------------------------------------------------------------
// Helper macro: safe widening coercions (always succeed)
// ---------------------------------------------------------------------------

macro_rules! impl_coerce_safe {
    // integer → same or wider integer
    ($from:ty => $to:ty) => {
        impl Coerce<$from> for $to {
            fn coerce_from(value: $from) -> Result<$to, CoercionError> {
                Ok(value as $to)
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Helper macro: checked narrowing coercions (may fail)
// ---------------------------------------------------------------------------

macro_rules! impl_coerce_checked_int {
    ($from:ty => $to:ty) => {
        impl Coerce<$from> for $to {
            fn coerce_from(value: $from) -> Result<$to, CoercionError> {
                <$to>::try_from(value).map_err(|_| {
                    CoercionError::new(
                        stringify!($from),
                        stringify!($to),
                        value,
                        format!(
                            "value {} is outside the range of {}",
                            value,
                            stringify!($to)
                        ),
                    )
                })
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Safe widening: integers
// ---------------------------------------------------------------------------

// i8 widens into everything larger
impl_coerce_safe!(i8  => i16);
impl_coerce_safe!(i8  => i32);
impl_coerce_safe!(i8  => i64);
impl_coerce_safe!(i8  => i128);
impl_coerce_safe!(i8  => f32);
impl_coerce_safe!(i8  => f64);

// i16
impl_coerce_safe!(i16 => i32);
impl_coerce_safe!(i16 => i64);
impl_coerce_safe!(i16 => i128);
impl_coerce_safe!(i16 => f32);
impl_coerce_safe!(i16 => f64);

// i32
impl_coerce_safe!(i32 => i64);
impl_coerce_safe!(i32 => i128);
impl_coerce_safe!(i32 => f64);

// i64
impl_coerce_safe!(i64 => i128);
impl_coerce_safe!(i64 => f64);

// u8
impl_coerce_safe!(u8  => u16);
impl_coerce_safe!(u8  => u32);
impl_coerce_safe!(u8  => u64);
impl_coerce_safe!(u8  => u128);
impl_coerce_safe!(u8  => i16);
impl_coerce_safe!(u8  => i32);
impl_coerce_safe!(u8  => i64);
impl_coerce_safe!(u8  => i128);
impl_coerce_safe!(u8  => f32);
impl_coerce_safe!(u8  => f64);

// u16
impl_coerce_safe!(u16 => u32);
impl_coerce_safe!(u16 => u64);
impl_coerce_safe!(u16 => u128);
impl_coerce_safe!(u16 => i32);
impl_coerce_safe!(u16 => i64);
impl_coerce_safe!(u16 => i128);
impl_coerce_safe!(u16 => f32);
impl_coerce_safe!(u16 => f64);

// u32
impl_coerce_safe!(u32 => u64);
impl_coerce_safe!(u32 => u128);
impl_coerce_safe!(u32 => i64);
impl_coerce_safe!(u32 => i128);
impl_coerce_safe!(u32 => f64);

// u64
impl_coerce_safe!(u64 => u128);
impl_coerce_safe!(u64 => i128);
impl_coerce_safe!(u64 => f64);

// f32 → f64 is exact
impl_coerce_safe!(f32 => f64);

// identity coercions
impl_coerce_safe!(i8   => i8);
impl_coerce_safe!(i16  => i16);
impl_coerce_safe!(i32  => i32);
impl_coerce_safe!(i64  => i64);
impl_coerce_safe!(i128 => i128);
impl_coerce_safe!(u8   => u8);
impl_coerce_safe!(u16  => u16);
impl_coerce_safe!(u32  => u32);
impl_coerce_safe!(u64  => u64);
impl_coerce_safe!(u128 => u128);
impl_coerce_safe!(f32  => f32);
impl_coerce_safe!(f64  => f64);

// ---------------------------------------------------------------------------
// Checked narrowing: integers via TryFrom
// ---------------------------------------------------------------------------

impl_coerce_checked_int!(i64  => i32);
impl_coerce_checked_int!(i64  => i16);
impl_coerce_checked_int!(i64  => i8);
impl_coerce_checked_int!(i64  => u8);
impl_coerce_checked_int!(i64  => u16);
impl_coerce_checked_int!(i64  => u32);
impl_coerce_checked_int!(i64  => u64);
impl_coerce_checked_int!(i32  => i16);
impl_coerce_checked_int!(i32  => i8);
impl_coerce_checked_int!(i32  => u8);
impl_coerce_checked_int!(i32  => u16);
impl_coerce_checked_int!(i32  => u32);
impl_coerce_checked_int!(i16  => i8);
impl_coerce_checked_int!(i16  => u8);
impl_coerce_checked_int!(u64  => u32);
impl_coerce_checked_int!(u64  => u16);
impl_coerce_checked_int!(u64  => u8);
impl_coerce_checked_int!(u64  => i64);
impl_coerce_checked_int!(u64  => i32);
impl_coerce_checked_int!(u64  => i16);
impl_coerce_checked_int!(u64  => i8);
impl_coerce_checked_int!(u32  => u16);
impl_coerce_checked_int!(u32  => u8);
impl_coerce_checked_int!(u32  => i32);
impl_coerce_checked_int!(u32  => i16);
impl_coerce_checked_int!(u32  => i8);
impl_coerce_checked_int!(u16  => u8);
impl_coerce_checked_int!(u16  => i16);
impl_coerce_checked_int!(u16  => i8);

// ---------------------------------------------------------------------------
// Checked: f64 → integers (must be finite, in range, and have no fractional part)
// ---------------------------------------------------------------------------

macro_rules! impl_coerce_f64_to_int {
    ($to:ty) => {
        impl Coerce<f64> for $to {
            fn coerce_from(value: f64) -> Result<$to, CoercionError> {
                if !value.is_finite() {
                    return Err(CoercionError::new(
                        "f64",
                        stringify!($to),
                        value,
                        "value is not finite (NaN or Infinity)",
                    ));
                }
                if value.fract() != 0.0 {
                    return Err(CoercionError::new(
                        "f64",
                        stringify!($to),
                        value,
                        format!("fractional part {:.17e} would be lost", value.fract()),
                    ));
                }
                let min = <$to>::MIN as f64;
                let max = <$to>::MAX as f64;
                if value < min || value > max {
                    return Err(CoercionError::new(
                        "f64",
                        stringify!($to),
                        value,
                        format!(
                            "value {value} is outside [{min}, {max}] for {}",
                            stringify!($to)
                        ),
                    ));
                }
                Ok(value as $to)
            }
        }
    };
}

impl_coerce_f64_to_int!(i8);
impl_coerce_f64_to_int!(i16);
impl_coerce_f64_to_int!(i32);
impl_coerce_f64_to_int!(i64);
impl_coerce_f64_to_int!(u8);
impl_coerce_f64_to_int!(u16);
impl_coerce_f64_to_int!(u32);
impl_coerce_f64_to_int!(u64);

// f64 → f32: checked for overflow
impl Coerce<f64> for f32 {
    fn coerce_from(value: f64) -> Result<f32, CoercionError> {
        let narrow = value as f32;
        if narrow.is_infinite() && value.is_finite() {
            return Err(CoercionError::new(
                "f64",
                "f32",
                value,
                "value overflows f32 range",
            ));
        }
        Ok(narrow)
    }
}

// ---------------------------------------------------------------------------
// Checked: f32 → integers
// ---------------------------------------------------------------------------

macro_rules! impl_coerce_f32_to_int {
    ($to:ty) => {
        impl Coerce<f32> for $to {
            fn coerce_from(value: f32) -> Result<$to, CoercionError> {
                // Delegate through f64 for a unified implementation
                let wide = value as f64;
                <$to as Coerce<f64>>::coerce_from(wide).map_err(|mut e| {
                    e.source_type = "f32";
                    e
                })
            }
        }
    };
}

impl_coerce_f32_to_int!(i8);
impl_coerce_f32_to_int!(i16);
impl_coerce_f32_to_int!(i32);
impl_coerce_f32_to_int!(i64);
impl_coerce_f32_to_int!(u8);
impl_coerce_f32_to_int!(u16);
impl_coerce_f32_to_int!(u32);
impl_coerce_f32_to_int!(u64);

// ---------------------------------------------------------------------------
// Lossiness probe
// ---------------------------------------------------------------------------

/// Returns `true` if converting a value of type `F` to type `T` may lose
/// information (e.g. due to narrowing, overflow, or fractional truncation).
///
/// This is a compile-time / type-level heuristic table, not a per-value check.
/// For per-value checks use the `Coerce` trait directly.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::validation::coercion::is_lossy;
///
/// assert!(!is_lossy::<i32, f64>());  // i32 → f64 always safe
/// assert!(is_lossy::<f64, i32>());   // f64 → i32 can lose info
/// assert!(!is_lossy::<f32, f64>());  // f32 → f64 always safe
/// assert!(is_lossy::<f64, f32>());   // f64 → f32 can overflow
/// assert!(!is_lossy::<i16, i32>());  // i16 → i32 always safe
/// assert!(is_lossy::<i32, i16>());   // i32 → i16 can overflow
/// ```
pub fn is_lossy<F: 'static, T: 'static>() -> bool {
    use std::any::TypeId;

    let from = TypeId::of::<F>();
    let to   = TypeId::of::<T>();

    // Same type: never lossy
    if from == to {
        return false;
    }

    // Float → integer: always potentially lossy
    let float_types: &[TypeId] = &[TypeId::of::<f32>(), TypeId::of::<f64>()];
    let int_types: &[TypeId] = &[
        TypeId::of::<i8>(),  TypeId::of::<i16>(), TypeId::of::<i32>(),
        TypeId::of::<i64>(), TypeId::of::<i128>(),
        TypeId::of::<u8>(),  TypeId::of::<u16>(), TypeId::of::<u32>(),
        TypeId::of::<u64>(), TypeId::of::<u128>(),
    ];
    if float_types.contains(&from) && int_types.contains(&to) {
        return true;
    }

    // f64 → f32: can overflow
    if from == TypeId::of::<f64>() && to == TypeId::of::<f32>() {
        return true;
    }

    // Narrowing integers: source byte-width > target byte-width
    fn byte_width(id: TypeId) -> Option<u8> {
        if id == TypeId::of::<i8>()   || id == TypeId::of::<u8>()   { return Some(1); }
        if id == TypeId::of::<i16>()  || id == TypeId::of::<u16>()  { return Some(2); }
        if id == TypeId::of::<i32>()  || id == TypeId::of::<u32>()  { return Some(4); }
        if id == TypeId::of::<i64>()  || id == TypeId::of::<u64>()  { return Some(8); }
        if id == TypeId::of::<i128>() || id == TypeId::of::<u128>() { return Some(16); }
        None
    }
    fn is_signed(id: TypeId) -> bool {
        id == TypeId::of::<i8>()
            || id == TypeId::of::<i16>()
            || id == TypeId::of::<i32>()
            || id == TypeId::of::<i64>()
            || id == TypeId::of::<i128>()
    }

    if let (Some(fw), Some(tw)) = (byte_width(from), byte_width(to)) {
        // Narrowing: source wider than target
        if fw > tw {
            return true;
        }
        // Signed → unsigned (same width): can lose negatives
        if fw == tw && is_signed(from) && !is_signed(to) {
            return true;
        }
        // Unsigned → signed (same width): upper half lost
        if fw == tw && !is_signed(from) && is_signed(to) {
            return true;
        }
    }

    false
}

// ---------------------------------------------------------------------------
// Convenience function
// ---------------------------------------------------------------------------

/// Coerce `value: F` to `T`, returning a `CoercionError` on failure.
///
/// This is a free-function wrapper around `T::coerce_from(value)`.
pub fn coerce<T, F>(value: F) -> Result<T, CoercionError>
where
    T: Coerce<F>,
{
    T::coerce_from(value)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- safe widens --

    #[test]
    fn test_i32_to_f64_safe() {
        let v: f64 = f64::coerce_from(42_i32).expect("should succeed");
        assert_eq!(v, 42.0);
    }

    #[test]
    fn test_i8_to_i64_safe() {
        let v: i64 = i64::coerce_from(-5_i8).expect("should succeed");
        assert_eq!(v, -5);
    }

    #[test]
    fn test_f32_to_f64_safe() {
        let v: f64 = f64::coerce_from(3.14_f32).expect("should succeed");
        assert!((v - 3.14_f64).abs() < 1e-5);
    }

    // -- checked narrowing integers --

    #[test]
    fn test_i64_to_i32_ok() {
        let v: i32 = i32::coerce_from(100_i64).expect("in range");
        assert_eq!(v, 100);
    }

    #[test]
    fn test_i64_to_i32_overflow() {
        let big: i64 = i64::from(i32::MAX) + 1;
        assert!(i32::coerce_from(big).is_err());
    }

    #[test]
    fn test_u32_to_i32_overflow() {
        // u32::MAX > i32::MAX
        assert!(i32::coerce_from(u32::MAX).is_err());
    }

    // -- f64 → int checks --

    #[test]
    fn test_f64_to_i32_ok() {
        let v: i32 = i32::coerce_from(3.0_f64).expect("exact");
        assert_eq!(v, 3);
    }

    #[test]
    fn test_f64_to_i32_fractional() {
        assert!(i32::coerce_from(3.7_f64).is_err());
    }

    #[test]
    fn test_f64_to_i32_nan() {
        assert!(i32::coerce_from(f64::NAN).is_err());
    }

    #[test]
    fn test_f64_to_i32_infinity() {
        assert!(i32::coerce_from(f64::INFINITY).is_err());
    }

    #[test]
    fn test_f64_to_i32_overflow() {
        let big: f64 = f64::from(i32::MAX) + 1.0;
        assert!(i32::coerce_from(big).is_err());
    }

    #[test]
    fn test_f64_to_u8_ok() {
        let v: u8 = u8::coerce_from(255.0_f64).expect("in range");
        assert_eq!(v, 255);
    }

    #[test]
    fn test_f64_to_u8_negative() {
        assert!(u8::coerce_from(-1.0_f64).is_err());
    }

    // -- f64 → f32 --

    #[test]
    fn test_f64_to_f32_ok() {
        let v: f32 = f32::coerce_from(1.5_f64).expect("in range");
        assert!((v - 1.5_f32).abs() < 1e-6);
    }

    #[test]
    fn test_f64_to_f32_overflow() {
        let huge = f64::MAX;
        assert!(f32::coerce_from(huge).is_err());
    }

    // -- is_lossy --

    #[test]
    fn test_is_lossy_safe() {
        assert!(!is_lossy::<i32, f64>());
        assert!(!is_lossy::<i8, i64>());
        assert!(!is_lossy::<f32, f64>());
        assert!(!is_lossy::<u8, u64>());
        assert!(!is_lossy::<u8, i32>());
    }

    #[test]
    fn test_is_lossy_unsafe() {
        assert!(is_lossy::<f64, i32>());
        assert!(is_lossy::<f64, f32>());
        assert!(is_lossy::<i32, i16>());
        assert!(is_lossy::<u32, i32>()); // same width, sign change
        assert!(is_lossy::<i32, u32>()); // same width, sign change
    }

    #[test]
    fn test_is_lossy_identity() {
        assert!(!is_lossy::<f64, f64>());
        assert!(!is_lossy::<i32, i32>());
    }

    // -- coerce convenience fn --

    #[test]
    fn test_coerce_fn() {
        let v: f64 = coerce(10_i32).expect("should succeed");
        assert_eq!(v, 10.0);
    }

    // -- CoercionError display --

    #[test]
    fn test_coercion_error_display() {
        let err = CoercionError::new("f64", "i32", 3.7, "fractional part");
        let s = err.to_string();
        assert!(s.contains("f64"));
        assert!(s.contains("i32"));
        assert!(s.contains("3.7"));
    }
}
