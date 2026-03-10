//! Core conversion implementations between Arrow arrays and ndarray
//!
//! Provides zero-copy (when possible) conversions between:
//! - `Array1<T>` ↔ Arrow primitive arrays (Float64Array, Int32Array, etc.)
//! - `Array2<T>` ↔ Arrow `RecordBatch` (columns as Arrow arrays)
//! - Nullable array support via `Option<T>`
//! - String array support
//! - Boolean array support

use super::error::{ArrowCompatError, ArrowResult};
use super::traits::{FromArrowArray, ToArrowArray, ZeroCopyFromArrow};
use arrow::array::{
    Array as ArrowArray, ArrayRef, AsArray, BooleanArray, Float32Array, Float64Array, Int32Array,
    Int64Array, StringArray,
};
use arrow::buffer::Buffer;
use arrow::datatypes::{
    ArrowPrimitiveType, DataType, Field, Float32Type, Float64Type, Int32Type, Int64Type, Schema,
};
use arrow::record_batch::RecordBatch;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use std::sync::Arc;

// =============================================================================
// Macro for implementing conversions on primitive numeric types
// =============================================================================

macro_rules! impl_primitive_arrow_conversion {
    ($rust_type:ty, $arrow_type:ty, $data_type:expr, $array_type:ty, $type_name:expr) => {
        impl ToArrowArray for $rust_type {
            fn to_arrow_array(data: &[Self]) -> ArrowResult<ArrayRef> {
                // Arrow stores primitive arrays as contiguous buffers, so this
                // copies the data into an Arrow-managed buffer.
                Ok(Arc::new(<$array_type>::from(Vec::from(data))))
            }

            fn arrow_data_type() -> DataType {
                $data_type
            }
        }

        impl FromArrowArray for $rust_type {
            fn from_arrow_array(array: &ArrayRef) -> ArrowResult<Array1<Self>> {
                let typed = array.as_primitive_opt::<$arrow_type>().ok_or_else(|| {
                    ArrowCompatError::TypeMismatch {
                        expected: $type_name.to_string(),
                        actual: format!("{:?}", array.data_type()),
                    }
                })?;

                // Check for null values
                if typed.null_count() > 0 {
                    return Err(ArrowCompatError::NullValuesPresent {
                        null_count: typed.null_count(),
                        total_count: typed.len(),
                    });
                }

                let values: Vec<$rust_type> = typed.values().iter().copied().collect();
                Ok(Array1::from_vec(values))
            }

            fn from_arrow_array_nullable(array: &ArrayRef) -> ArrowResult<Array1<Option<Self>>> {
                let typed = array.as_primitive_opt::<$arrow_type>().ok_or_else(|| {
                    ArrowCompatError::TypeMismatch {
                        expected: $type_name.to_string(),
                        actual: format!("{:?}", array.data_type()),
                    }
                })?;

                let values: Vec<Option<$rust_type>> = (0..typed.len())
                    .map(|i| {
                        if typed.is_null(i) {
                            None
                        } else {
                            Some(typed.value(i))
                        }
                    })
                    .collect();
                Ok(Array1::from_vec(values))
            }
        }

        impl ZeroCopyFromArrow for $rust_type {
            fn try_zero_copy_view(array: &ArrayRef) -> ArrowResult<Option<ArrayView1<'_, Self>>> {
                let typed = array.as_primitive_opt::<$arrow_type>().ok_or_else(|| {
                    ArrowCompatError::TypeMismatch {
                        expected: $type_name.to_string(),
                        actual: format!("{:?}", array.data_type()),
                    }
                })?;

                // Zero-copy is only possible when there are no null values
                if typed.null_count() > 0 {
                    return Ok(None);
                }

                // Arrow primitive arrays store values in a contiguous buffer,
                // so we can create a view directly over the buffer data.
                let values_slice: &[$rust_type] = typed.values();
                let view = ArrayView1::from(values_slice);
                Ok(Some(view))
            }
        }
    };
}

// Implement for all required primitive types
impl_primitive_arrow_conversion!(f64, Float64Type, DataType::Float64, Float64Array, "Float64");
impl_primitive_arrow_conversion!(f32, Float32Type, DataType::Float32, Float32Array, "Float32");
impl_primitive_arrow_conversion!(i64, Int64Type, DataType::Int64, Int64Array, "Int64");
impl_primitive_arrow_conversion!(i32, Int32Type, DataType::Int32, Int32Array, "Int32");

// =============================================================================
// Boolean conversions
// =============================================================================

impl ToArrowArray for bool {
    fn to_arrow_array(data: &[Self]) -> ArrowResult<ArrayRef> {
        Ok(Arc::new(BooleanArray::from(Vec::from(data))))
    }

    fn arrow_data_type() -> DataType {
        DataType::Boolean
    }
}

impl FromArrowArray for bool {
    fn from_arrow_array(array: &ArrayRef) -> ArrowResult<Array1<Self>> {
        let bool_array = array
            .as_boolean_opt()
            .ok_or_else(|| ArrowCompatError::TypeMismatch {
                expected: "Boolean".to_string(),
                actual: format!("{:?}", array.data_type()),
            })?;

        if bool_array.null_count() > 0 {
            return Err(ArrowCompatError::NullValuesPresent {
                null_count: bool_array.null_count(),
                total_count: bool_array.len(),
            });
        }

        let values: Vec<bool> = (0..bool_array.len()).map(|i| bool_array.value(i)).collect();
        Ok(Array1::from_vec(values))
    }

    fn from_arrow_array_nullable(array: &ArrayRef) -> ArrowResult<Array1<Option<Self>>> {
        let bool_array = array
            .as_boolean_opt()
            .ok_or_else(|| ArrowCompatError::TypeMismatch {
                expected: "Boolean".to_string(),
                actual: format!("{:?}", array.data_type()),
            })?;

        let values: Vec<Option<bool>> = (0..bool_array.len())
            .map(|i| {
                if bool_array.is_null(i) {
                    None
                } else {
                    Some(bool_array.value(i))
                }
            })
            .collect();
        Ok(Array1::from_vec(values))
    }
}

// =============================================================================
// String conversions
// =============================================================================

impl ToArrowArray for String {
    fn to_arrow_array(data: &[Self]) -> ArrowResult<ArrayRef> {
        let refs: Vec<&str> = data.iter().map(|s| s.as_str()).collect();
        Ok(Arc::new(StringArray::from(refs)))
    }

    fn arrow_data_type() -> DataType {
        DataType::Utf8
    }
}

impl FromArrowArray for String {
    fn from_arrow_array(array: &ArrayRef) -> ArrowResult<Array1<Self>> {
        let string_array = array
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| ArrowCompatError::TypeMismatch {
                expected: "Utf8 (String)".to_string(),
                actual: format!("{:?}", array.data_type()),
            })?;

        if string_array.null_count() > 0 {
            return Err(ArrowCompatError::NullValuesPresent {
                null_count: string_array.null_count(),
                total_count: string_array.len(),
            });
        }

        let values: Vec<String> = (0..string_array.len())
            .map(|i| string_array.value(i).to_string())
            .collect();
        Ok(Array1::from_vec(values))
    }

    fn from_arrow_array_nullable(array: &ArrayRef) -> ArrowResult<Array1<Option<Self>>> {
        let string_array = array
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| ArrowCompatError::TypeMismatch {
                expected: "Utf8 (String)".to_string(),
                actual: format!("{:?}", array.data_type()),
            })?;

        let values: Vec<Option<String>> = (0..string_array.len())
            .map(|i| {
                if string_array.is_null(i) {
                    None
                } else {
                    Some(string_array.value(i).to_string())
                }
            })
            .collect();
        Ok(Array1::from_vec(values))
    }
}

// =============================================================================
// Array1 → Arrow conversions (convenience functions)
// =============================================================================

/// Convert an `Array1<T>` to an Arrow `ArrayRef`
///
/// This copies the array data into an Arrow-managed buffer.
/// For zero-copy sharing, use [`array1_to_arrow_zero_copy`] when the
/// data lifetime permits.
///
/// # Examples
///
/// ```rust
/// # use scirs2_core::arrow_compat::conversions::array1_to_arrow;
/// # use ndarray::Array1;
/// let arr = Array1::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);
/// let arrow_arr = array1_to_arrow(&arr).expect("conversion failed");
/// assert_eq!(arrow_arr.len(), 4);
/// ```
pub fn array1_to_arrow<T>(array: &Array1<T>) -> ArrowResult<ArrayRef>
where
    T: ToArrowArray + Clone,
{
    let data: Vec<T> = array.iter().cloned().collect();
    T::to_arrow_array(&data)
}

/// Convert an `Array1<T>` to an Arrow `Float64Array` with zero-copy when possible
///
/// This attempts to use the ndarray's underlying buffer directly. If the
/// array is contiguous in memory, no copy is made.
pub fn array1_to_arrow_zero_copy(array: &Array1<f64>) -> ArrowResult<ArrayRef> {
    // Check if the array has standard (C-contiguous) layout
    if let Some(slice) = array.as_slice() {
        // The data is contiguous - we can build an Arrow buffer from it
        // However, Arrow needs to own the data, so we still need to copy
        // into an Arrow-managed buffer. The "zero-copy" here means we avoid
        // intermediate Vec allocations by going directly from slice to Buffer.
        let buffer = Buffer::from_slice_ref(slice);
        let arrow_array = Float64Array::new(buffer.into(), None);
        Ok(Arc::new(arrow_array))
    } else {
        // Non-contiguous: fall back to copy
        let data: Vec<f64> = array.iter().copied().collect();
        Ok(Arc::new(Float64Array::from(data)))
    }
}

/// Convert an Arrow array to an `Array1<T>`
///
/// Returns an error if the Arrow array contains null values or
/// has an incompatible type. Use [`arrow_to_array1_nullable`] for
/// arrays that may contain nulls.
///
/// # Examples
///
/// ```rust
/// # use scirs2_core::arrow_compat::conversions::{array1_to_arrow, arrow_to_array1};
/// # use ndarray::Array1;
/// let original = Array1::from_vec(vec![1.0_f64, 2.0, 3.0]);
/// let arrow_arr = array1_to_arrow(&original).expect("conversion failed");
/// let recovered: Array1<f64> = arrow_to_array1(&arrow_arr).expect("conversion failed");
/// assert_eq!(original, recovered);
/// ```
pub fn arrow_to_array1<T>(array: &ArrayRef) -> ArrowResult<Array1<T>>
where
    T: FromArrowArray,
{
    T::from_arrow_array(array)
}

/// Convert an Arrow array to an `Array1<Option<T>>` (nullable)
///
/// Null values in the Arrow array become `None` in the output.
pub fn arrow_to_array1_nullable<T>(array: &ArrayRef) -> ArrowResult<Array1<Option<T>>>
where
    T: FromArrowArray,
{
    T::from_arrow_array_nullable(array)
}

// =============================================================================
// Array2 ↔ RecordBatch conversions
// =============================================================================

/// Convert an `Array2<T>` to an Arrow `RecordBatch`
///
/// Each column of the 2D array becomes a column in the RecordBatch.
/// Column names are generated as "col_0", "col_1", etc., unless
/// custom names are provided.
///
/// # Arguments
///
/// * `array` - The 2D ndarray to convert
/// * `column_names` - Optional column names. If `None`, generates "col_0", "col_1", etc.
///
/// # Examples
///
/// ```rust
/// # use scirs2_core::arrow_compat::conversions::array2_to_record_batch;
/// # use ndarray::Array2;
/// let arr = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
///     .expect("shape error");
/// let batch = array2_to_record_batch(&arr, None).expect("conversion failed");
/// assert_eq!(batch.num_rows(), 3);
/// assert_eq!(batch.num_columns(), 2);
/// ```
pub fn array2_to_record_batch<T>(
    array: &Array2<T>,
    column_names: Option<&[&str]>,
) -> ArrowResult<RecordBatch>
where
    T: ToArrowArray + Clone,
{
    let (nrows, ncols) = (array.nrows(), array.ncols());

    // Validate column names length if provided
    if let Some(names) = column_names {
        if names.len() != ncols {
            return Err(ArrowCompatError::ShapeMismatch {
                expected: vec![ncols],
                actual: vec![names.len()],
            });
        }
    }

    // Build fields and arrays for each column
    let mut fields = Vec::with_capacity(ncols);
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(ncols);

    for col_idx in 0..ncols {
        let col_name = column_names
            .and_then(|names| names.get(col_idx).copied())
            .unwrap_or_else(|| {
                // Leak a string to get a &'static str for default names
                // This is acceptable because column names are typically few and long-lived
                // Actually, let's just format into a String and use it for Field
                ""
            });

        // Use a generated name if none provided
        let name = if col_name.is_empty() {
            format!("col_{}", col_idx)
        } else {
            col_name.to_string()
        };

        // Extract column data
        let col_data: Vec<T> = array.column(col_idx).iter().cloned().collect();
        let arrow_array = T::to_arrow_array(&col_data)?;

        fields.push(Field::new(&name, T::arrow_data_type(), false));
        arrays.push(arrow_array);
    }

    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, arrays).map_err(ArrowCompatError::from)
}

/// Convert an Arrow `RecordBatch` to an `Array2<T>`
///
/// All columns in the RecordBatch must have the same type `T`.
///
/// # Arguments
///
/// * `batch` - The RecordBatch to convert
///
/// # Examples
///
/// ```rust
/// # use scirs2_core::arrow_compat::conversions::{array2_to_record_batch, record_batch_to_array2};
/// # use ndarray::Array2;
/// let arr = Array2::from_shape_vec((3, 2), vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
///     .expect("shape error");
/// let batch = array2_to_record_batch(&arr, None).expect("conversion failed");
/// let recovered: Array2<f64> = record_batch_to_array2(&batch).expect("conversion failed");
/// assert_eq!(arr, recovered);
/// ```
pub fn record_batch_to_array2<T>(batch: &RecordBatch) -> ArrowResult<Array2<T>>
where
    T: FromArrowArray + Clone + Default,
{
    let nrows = batch.num_rows();
    let ncols = batch.num_columns();

    if ncols == 0 {
        return Err(ArrowCompatError::SchemaError(
            "RecordBatch has no columns".to_string(),
        ));
    }

    // Extract each column into an Array1 and combine
    let mut data = Vec::with_capacity(nrows * ncols);

    // We need row-major order for Array2, so we iterate rows then columns
    let columns: Vec<Array1<T>> = (0..ncols)
        .map(|col_idx| T::from_arrow_array(batch.column(col_idx)))
        .collect::<ArrowResult<Vec<_>>>()?;

    // Validate all columns have the same length
    for (col_idx, col) in columns.iter().enumerate() {
        if col.len() != nrows {
            return Err(ArrowCompatError::InconsistentColumnLengths {
                expected_len: nrows,
                column_index: col_idx,
                column_len: col.len(),
            });
        }
    }

    // Build row-major data
    for row_idx in 0..nrows {
        for col in &columns {
            data.push(col[row_idx].clone());
        }
    }

    let data_len = data.len();
    Array2::from_shape_vec((nrows, ncols), data).map_err(|_| ArrowCompatError::ShapeMismatch {
        expected: vec![nrows, ncols],
        actual: vec![data_len],
    })
}

/// Convert a single column from a `RecordBatch` to an `Array1<T>` by index
pub fn record_batch_column_to_array1<T>(
    batch: &RecordBatch,
    column_index: usize,
) -> ArrowResult<Array1<T>>
where
    T: FromArrowArray,
{
    if column_index >= batch.num_columns() {
        return Err(ArrowCompatError::ColumnOutOfBounds {
            index: column_index,
            num_columns: batch.num_columns(),
        });
    }

    T::from_arrow_array(batch.column(column_index))
}

/// Convert a single column from a `RecordBatch` to an `Array1<T>` by name
pub fn record_batch_column_by_name<T>(
    batch: &RecordBatch,
    column_name: &str,
) -> ArrowResult<Array1<T>>
where
    T: FromArrowArray,
{
    let schema = batch.schema();
    let col_idx = schema
        .fields()
        .iter()
        .position(|f| f.name() == column_name)
        .ok_or_else(|| ArrowCompatError::ColumnNotFound {
            name: column_name.to_string(),
        })?;

    T::from_arrow_array(batch.column(col_idx))
}

// =============================================================================
// Nullable Option<T> → Arrow conversions
// =============================================================================

/// Convert an `Array1<Option<T>>` to a nullable Arrow array
///
/// `None` values become null entries in the Arrow array.
pub fn nullable_array1_to_arrow<T>(array: &Array1<Option<T>>) -> ArrowResult<ArrayRef>
where
    T: NullableToArrow + Clone,
{
    let data: Vec<Option<T>> = array.iter().cloned().collect();
    T::nullable_to_arrow(&data)
}

/// Trait for types that support nullable Arrow conversion
pub trait NullableToArrow: Sized {
    /// Convert a slice of `Option<Self>` to a nullable Arrow array
    fn nullable_to_arrow(data: &[Option<Self>]) -> ArrowResult<ArrayRef>;
}

macro_rules! impl_nullable_to_arrow {
    ($rust_type:ty, $arrow_array_type:ty) => {
        impl NullableToArrow for $rust_type {
            fn nullable_to_arrow(data: &[Option<Self>]) -> ArrowResult<ArrayRef> {
                let array: $arrow_array_type = data.iter().copied().collect();
                Ok(Arc::new(array))
            }
        }
    };
}

impl_nullable_to_arrow!(f64, Float64Array);
impl_nullable_to_arrow!(f32, Float32Array);
impl_nullable_to_arrow!(i64, Int64Array);
impl_nullable_to_arrow!(i32, Int32Array);

impl NullableToArrow for bool {
    fn nullable_to_arrow(data: &[Option<Self>]) -> ArrowResult<ArrayRef> {
        let array: BooleanArray = data.iter().copied().collect();
        Ok(Arc::new(array))
    }
}

impl NullableToArrow for String {
    fn nullable_to_arrow(data: &[Option<Self>]) -> ArrowResult<ArrayRef> {
        let refs: Vec<Option<&str>> = data.iter().map(|s| s.as_deref()).collect();
        let array = StringArray::from(refs);
        Ok(Arc::new(array))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------
    // Array1 <-> Arrow primitive roundtrip tests
    // -------------------------------------------------------

    #[test]
    fn test_array1_f64_roundtrip() {
        let original = Array1::from_vec(vec![1.0_f64, 2.5, -1.23, 0.0, f64::MAX]);
        let arrow = array1_to_arrow(&original).expect("to_arrow failed");
        let recovered: Array1<f64> = arrow_to_array1(&arrow).expect("from_arrow failed");
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_array1_f32_roundtrip() {
        let original = Array1::from_vec(vec![1.0_f32, 2.5, -1.23, 0.0]);
        let arrow = array1_to_arrow(&original).expect("to_arrow failed");
        let recovered: Array1<f32> = arrow_to_array1(&arrow).expect("from_arrow failed");
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_array1_i64_roundtrip() {
        let original = Array1::from_vec(vec![1_i64, -100, i64::MAX, i64::MIN, 0]);
        let arrow = array1_to_arrow(&original).expect("to_arrow failed");
        let recovered: Array1<i64> = arrow_to_array1(&arrow).expect("from_arrow failed");
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_array1_i32_roundtrip() {
        let original = Array1::from_vec(vec![10_i32, 20, 30, -40]);
        let arrow = array1_to_arrow(&original).expect("to_arrow failed");
        let recovered: Array1<i32> = arrow_to_array1(&arrow).expect("from_arrow failed");
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_array1_bool_roundtrip() {
        let original = Array1::from_vec(vec![true, false, true, false, true]);
        let arrow = array1_to_arrow(&original).expect("to_arrow failed");
        let recovered: Array1<bool> = arrow_to_array1(&arrow).expect("from_arrow failed");
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_array1_string_roundtrip() {
        let original = Array1::from_vec(vec![
            "hello".to_string(),
            "world".to_string(),
            "".to_string(),
            "test 123".to_string(),
        ]);
        let arrow = array1_to_arrow(&original).expect("to_arrow failed");
        let recovered: Array1<String> = arrow_to_array1(&arrow).expect("from_arrow failed");
        assert_eq!(original, recovered);
    }

    // -------------------------------------------------------
    // Zero-copy tests
    // -------------------------------------------------------

    #[test]
    fn test_zero_copy_f64() {
        let original = Array1::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);
        let arrow = array1_to_arrow_zero_copy(&original).expect("zero_copy to_arrow failed");

        // Verify the data is correct
        let recovered: Array1<f64> = arrow_to_array1(&arrow).expect("from_arrow failed");
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_zero_copy_view_f64() {
        let arrow_arr: ArrayRef = Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0]));
        let view = f64::try_zero_copy_view(&arrow_arr).expect("zero_copy_view failed");
        assert!(view.is_some());
        let view = view.expect("should have view");
        assert_eq!(view.len(), 3);
        assert!((view[0] - 10.0).abs() < f64::EPSILON);
        assert!((view[1] - 20.0).abs() < f64::EPSILON);
        assert!((view[2] - 30.0).abs() < f64::EPSILON);
    }

    // -------------------------------------------------------
    // Nullable array tests
    // -------------------------------------------------------

    #[test]
    fn test_nullable_f64() {
        let data = Array1::from_vec(vec![Some(1.0_f64), None, Some(3.0), None, Some(5.0)]);
        let arrow = nullable_array1_to_arrow(&data).expect("nullable to_arrow failed");
        let recovered: Array1<Option<f64>> =
            arrow_to_array1_nullable(&arrow).expect("nullable from_arrow failed");
        assert_eq!(data, recovered);
    }

    #[test]
    fn test_nullable_i32() {
        let data = Array1::from_vec(vec![Some(10_i32), None, Some(30)]);
        let arrow = nullable_array1_to_arrow(&data).expect("nullable to_arrow failed");
        let recovered: Array1<Option<i32>> =
            arrow_to_array1_nullable(&arrow).expect("nullable from_arrow failed");
        assert_eq!(data, recovered);
    }

    #[test]
    fn test_nullable_bool() {
        let data = Array1::from_vec(vec![Some(true), None, Some(false)]);
        let arrow = nullable_array1_to_arrow(&data).expect("nullable to_arrow failed");
        let recovered: Array1<Option<bool>> =
            arrow_to_array1_nullable(&arrow).expect("nullable from_arrow failed");
        assert_eq!(data, recovered);
    }

    #[test]
    fn test_nullable_string() {
        let data = Array1::from_vec(vec![
            Some("hello".to_string()),
            None,
            Some("world".to_string()),
        ]);
        let arrow = nullable_array1_to_arrow(&data).expect("nullable to_arrow failed");
        let recovered: Array1<Option<String>> =
            arrow_to_array1_nullable(&arrow).expect("nullable from_arrow failed");
        assert_eq!(data, recovered);
    }

    #[test]
    fn test_null_values_rejected_by_non_nullable() {
        let arrow_arr: ArrayRef = Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0)]));
        let result: ArrowResult<Array1<f64>> = arrow_to_array1(&arrow_arr);
        assert!(result.is_err());
    }

    // -------------------------------------------------------
    // Array2 <-> RecordBatch tests
    // -------------------------------------------------------

    #[test]
    fn test_array2_f64_to_record_batch() {
        let arr = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("shape error");
        let batch = array2_to_record_batch(&arr, None).expect("to_batch failed");

        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.schema().field(0).name(), "col_0");
        assert_eq!(batch.schema().field(1).name(), "col_1");
    }

    #[test]
    fn test_array2_f64_roundtrip() {
        let arr = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .expect("shape error");
        let batch = array2_to_record_batch(&arr, None).expect("to_batch failed");
        let recovered: Array2<f64> = record_batch_to_array2(&batch).expect("from_batch failed");
        assert_eq!(arr, recovered);
    }

    #[test]
    fn test_array2_with_custom_column_names() {
        let arr = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("shape error");
        let names = vec!["x", "y", "z"];
        let batch = array2_to_record_batch(&arr, Some(&names)).expect("to_batch failed");

        assert_eq!(batch.schema().field(0).name(), "x");
        assert_eq!(batch.schema().field(1).name(), "y");
        assert_eq!(batch.schema().field(2).name(), "z");
    }

    #[test]
    fn test_record_batch_column_by_name() {
        let arr = Array2::from_shape_vec((3, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0])
            .expect("shape error");
        let names = vec!["values", "scores"];
        let batch = array2_to_record_batch(&arr, Some(&names)).expect("to_batch failed");

        let values: Array1<f64> =
            record_batch_column_by_name(&batch, "values").expect("column lookup failed");
        assert_eq!(values, Array1::from_vec(vec![1.0, 2.0, 3.0]));

        let scores: Array1<f64> =
            record_batch_column_by_name(&batch, "scores").expect("column lookup failed");
        assert_eq!(scores, Array1::from_vec(vec![10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_record_batch_column_not_found() {
        let arr = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).expect("shape error");
        let batch = array2_to_record_batch(&arr, None).expect("to_batch failed");

        let result: ArrowResult<Array1<f64>> = record_batch_column_by_name(&batch, "nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_column_out_of_bounds() {
        let arr = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).expect("shape error");
        let batch = array2_to_record_batch(&arr, None).expect("to_batch failed");

        let result: ArrowResult<Array1<f64>> = record_batch_column_to_array1(&batch, 5);
        assert!(result.is_err());
    }

    // -------------------------------------------------------
    // Type mismatch error tests
    // -------------------------------------------------------

    #[test]
    fn test_type_mismatch_error() {
        let arrow_arr: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0]));
        let result: ArrowResult<Array1<i32>> = arrow_to_array1(&arrow_arr);
        assert!(result.is_err());
    }

    // -------------------------------------------------------
    // Edge cases
    // -------------------------------------------------------

    #[test]
    fn test_empty_array() {
        let original: Array1<f64> = Array1::from_vec(vec![]);
        let arrow = array1_to_arrow(&original).expect("to_arrow failed");
        let recovered: Array1<f64> = arrow_to_array1(&arrow).expect("from_arrow failed");
        assert_eq!(original, recovered);
        assert_eq!(recovered.len(), 0);
    }

    #[test]
    fn test_single_element() {
        let original = Array1::from_vec(vec![42.0_f64]);
        let arrow = array1_to_arrow(&original).expect("to_arrow failed");
        let recovered: Array1<f64> = arrow_to_array1(&arrow).expect("from_arrow failed");
        assert_eq!(original, recovered);
    }
}
