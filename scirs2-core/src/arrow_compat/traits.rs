//! Conversion traits for Arrow ↔ ndarray interoperability
//!
//! Defines the core traits that enable type-safe conversions between
//! Rust primitive types and Arrow array types.

use super::error::ArrowResult;
use arrow::array::ArrayRef;
use arrow::datatypes::DataType;
use ndarray::Array1;

/// Trait for types that can be converted to Arrow arrays
///
/// Implemented for primitive numeric types (f32, f64, i32, i64, etc.),
/// boolean, and String types.
pub trait ToArrowArray: Sized {
    /// Convert a slice of data to an Arrow array
    fn to_arrow_array(data: &[Self]) -> ArrowResult<ArrayRef>;

    /// Get the Arrow `DataType` for this Rust type
    fn arrow_data_type() -> DataType;
}

/// Trait for types that can be extracted from Arrow arrays
///
/// Provides both fallible extraction (which returns an error if the
/// Arrow array contains nulls or has wrong type) and nullable extraction
/// (which returns `Option<T>` for nullable columns).
pub trait FromArrowArray: Sized {
    /// Extract data from an Arrow array into an `Array1`
    ///
    /// Returns an error if the array contains null values.
    /// Use [`from_arrow_array_nullable`](FromArrowArray::from_arrow_array_nullable)
    /// for arrays that may contain nulls.
    fn from_arrow_array(array: &ArrayRef) -> ArrowResult<Array1<Self>>;

    /// Extract data from a nullable Arrow array into an `Array1<Option<Self>>`
    fn from_arrow_array_nullable(array: &ArrayRef) -> ArrowResult<Array1<Option<Self>>>;
}

/// Trait for types that support zero-copy conversion from Arrow buffers
///
/// This is only possible when the memory layout of the Rust type exactly
/// matches the Arrow buffer layout (e.g., contiguous f64 values).
pub trait ZeroCopyFromArrow: Sized {
    /// Attempt a zero-copy view of the Arrow array data.
    ///
    /// Returns `None` if zero-copy is not possible (e.g., due to null
    /// bitmap, non-contiguous data, or type mismatch).
    fn try_zero_copy_view(array: &ArrayRef) -> ArrowResult<Option<ndarray::ArrayView1<'_, Self>>>;
}
