//! Schema utilities for Arrow interoperability
//!
//! Provides functions to create and inspect Arrow schemas from
//! ndarray shape and dtype information.

use super::error::{ArrowCompatError, ArrowResult};
use super::traits::ToArrowArray;
use arrow::datatypes::{DataType, Field, Schema};
use ndarray::{ArrayBase, Data as NdData, Dimension, Ix2};
use std::sync::Arc;

/// Create an Arrow schema for a 1D ndarray
///
/// Generates a schema with a single column named by the provided name.
///
/// # Arguments
///
/// * `column_name` - Name for the column
///
/// # Type Parameters
///
/// * `T` - The element type, must implement `ToArrowArray`
///
/// # Examples
///
/// ```rust
/// # use scirs2_core::arrow_compat::schema::ndarray_schema_1d;
/// let schema = ndarray_schema_1d::<f64>("values").expect("schema creation failed");
/// assert_eq!(schema.fields().len(), 1);
/// assert_eq!(schema.field(0).name(), "values");
/// ```
pub fn ndarray_schema_1d<T: ToArrowArray>(column_name: &str) -> ArrowResult<Arc<Schema>> {
    let field = Field::new(column_name, T::arrow_data_type(), false);
    Ok(Arc::new(Schema::new(vec![field])))
}

/// Create an Arrow schema for a 2D ndarray
///
/// Generates a schema with one column per array column. Column names
/// are either provided or auto-generated as "col_0", "col_1", etc.
///
/// # Arguments
///
/// * `ncols` - Number of columns in the array
/// * `column_names` - Optional column names
/// * `nullable` - Whether columns should be marked as nullable
///
/// # Type Parameters
///
/// * `T` - The element type, must implement `ToArrowArray`
///
/// # Examples
///
/// ```rust
/// # use scirs2_core::arrow_compat::schema::ndarray_schema_2d;
/// let schema = ndarray_schema_2d::<f64>(3, None, false).expect("schema creation failed");
/// assert_eq!(schema.fields().len(), 3);
/// assert_eq!(schema.field(0).name(), "col_0");
/// assert_eq!(schema.field(2).name(), "col_2");
///
/// let schema = ndarray_schema_2d::<f64>(2, Some(&["x", "y"]), false)
///     .expect("schema creation failed");
/// assert_eq!(schema.field(0).name(), "x");
/// assert_eq!(schema.field(1).name(), "y");
/// ```
pub fn ndarray_schema_2d<T: ToArrowArray>(
    ncols: usize,
    column_names: Option<&[&str]>,
    nullable: bool,
) -> ArrowResult<Arc<Schema>> {
    if let Some(names) = column_names {
        if names.len() != ncols {
            return Err(ArrowCompatError::ShapeMismatch {
                expected: vec![ncols],
                actual: vec![names.len()],
            });
        }
    }

    let fields: Vec<Field> = (0..ncols)
        .map(|i| {
            let name = column_names
                .and_then(|names| names.get(i).copied())
                .map(|n| n.to_string())
                .unwrap_or_else(|| format!("col_{}", i));
            Field::new(name, T::arrow_data_type(), nullable)
        })
        .collect();

    Ok(Arc::new(Schema::new(fields)))
}

/// Infer an Arrow schema from an existing ndarray
///
/// For 1D arrays, creates a single-column schema named "value".
/// For 2D arrays, creates a multi-column schema with auto-generated names.
///
/// # Examples
///
/// ```rust
/// # use scirs2_core::arrow_compat::schema::infer_schema;
/// # use ndarray::Array1;
/// let arr = Array1::from_vec(vec![1.0_f64, 2.0, 3.0]);
/// let schema = infer_schema(&arr).expect("schema inference failed");
/// assert_eq!(schema.fields().len(), 1);
/// ```
pub fn infer_schema<S, D, T>(array: &ArrayBase<S, D>) -> ArrowResult<Arc<Schema>>
where
    S: NdData<Elem = T>,
    D: Dimension,
    T: ToArrowArray,
{
    let ndim = array.ndim();
    match ndim {
        0 => {
            // Scalar - single column
            ndarray_schema_1d::<T>("value")
        }
        1 => ndarray_schema_1d::<T>("value"),
        2 => {
            let shape = array.shape();
            let ncols = shape[1];
            ndarray_schema_2d::<T>(ncols, None, false)
        }
        _ => Err(ArrowCompatError::SchemaError(format!(
            "Cannot infer Arrow schema for {}-dimensional array (only 1D and 2D supported)",
            ndim
        ))),
    }
}

/// Create an Arrow schema with metadata
///
/// Attaches arbitrary key-value metadata to the schema, useful for
/// storing ndarray-specific information like shape, strides, etc.
///
/// # Arguments
///
/// * `fields` - Schema fields
/// * `metadata` - Key-value pairs of metadata
pub fn schema_with_metadata(fields: Vec<Field>, metadata: Vec<(&str, &str)>) -> Arc<Schema> {
    let meta: std::collections::HashMap<String, String> = metadata
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    Arc::new(Schema::new_with_metadata(fields, meta))
}

/// Create an Arrow schema that encodes ndarray shape information
///
/// This is useful for roundtrip conversions where the original shape
/// must be preserved (e.g., reshaping a 3D array through Arrow).
///
/// # Arguments
///
/// * `shape` - The ndarray shape
/// * `dtype_name` - Human-readable type name
///
/// # Examples
///
/// ```rust
/// # use scirs2_core::arrow_compat::schema::ndarray_shape_schema;
/// # use arrow::datatypes::DataType;
/// let schema = ndarray_shape_schema(&[10, 20, 30], "f64");
/// let meta = schema.metadata();
/// assert_eq!(meta.get("ndarray.shape"), Some(&"10,20,30".to_string()));
/// assert_eq!(meta.get("ndarray.ndim"), Some(&"3".to_string()));
/// assert_eq!(meta.get("ndarray.dtype"), Some(&"f64".to_string()));
/// ```
pub fn ndarray_shape_schema(shape: &[usize], dtype_name: &str) -> Arc<Schema> {
    let total_elements: usize = shape.iter().product();
    let field = Field::new("data", DataType::Float64, false);

    let shape_str = shape
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(",");

    schema_with_metadata(
        vec![field],
        vec![
            ("ndarray.shape", &shape_str),
            ("ndarray.ndim", &shape.len().to_string()),
            ("ndarray.dtype", dtype_name),
            ("ndarray.total_elements", &total_elements.to_string()),
        ],
    )
}

/// Extract ndarray shape from an Arrow schema's metadata
///
/// This is the inverse of [`ndarray_shape_schema`].
///
/// # Returns
///
/// The shape as a vector of dimension sizes, or an error if the
/// metadata is missing or malformed.
pub fn extract_ndarray_shape(schema: &Schema) -> ArrowResult<Vec<usize>> {
    let metadata = schema.metadata();

    let shape_str = metadata.get("ndarray.shape").ok_or_else(|| {
        ArrowCompatError::SchemaError("Schema metadata missing 'ndarray.shape' key".to_string())
    })?;

    shape_str
        .split(',')
        .map(|s| {
            s.trim().parse::<usize>().map_err(|e| {
                ArrowCompatError::SchemaError(format!("Invalid shape dimension '{}': {}", s, e))
            })
        })
        .collect()
}

/// Validate that a RecordBatch matches an expected schema
///
/// Checks field count, names, and data types.
pub fn validate_schema(
    batch: &arrow::record_batch::RecordBatch,
    expected: &Schema,
) -> ArrowResult<()> {
    let actual = batch.schema();

    if actual.fields().len() != expected.fields().len() {
        return Err(ArrowCompatError::SchemaError(format!(
            "Field count mismatch: expected {}, got {}",
            expected.fields().len(),
            actual.fields().len()
        )));
    }

    for (i, (expected_field, actual_field)) in expected
        .fields()
        .iter()
        .zip(actual.fields().iter())
        .enumerate()
    {
        if expected_field.name() != actual_field.name() {
            return Err(ArrowCompatError::SchemaError(format!(
                "Field {} name mismatch: expected '{}', got '{}'",
                i,
                expected_field.name(),
                actual_field.name()
            )));
        }

        if expected_field.data_type() != actual_field.data_type() {
            return Err(ArrowCompatError::TypeMismatch {
                expected: format!("{:?}", expected_field.data_type()),
                actual: format!("{:?}", actual_field.data_type()),
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_schema_1d_f64() {
        let schema = ndarray_schema_1d::<f64>("temperature").expect("schema failed");
        assert_eq!(schema.fields().len(), 1);
        assert_eq!(schema.field(0).name(), "temperature");
        assert_eq!(*schema.field(0).data_type(), DataType::Float64);
        assert!(!schema.field(0).is_nullable());
    }

    #[test]
    fn test_schema_1d_i32() {
        let schema = ndarray_schema_1d::<i32>("count").expect("schema failed");
        assert_eq!(*schema.field(0).data_type(), DataType::Int32);
    }

    #[test]
    fn test_schema_1d_bool() {
        let schema = ndarray_schema_1d::<bool>("flags").expect("schema failed");
        assert_eq!(*schema.field(0).data_type(), DataType::Boolean);
    }

    #[test]
    fn test_schema_2d_auto_names() {
        let schema = ndarray_schema_2d::<f64>(4, None, false).expect("schema failed");
        assert_eq!(schema.fields().len(), 4);
        assert_eq!(schema.field(0).name(), "col_0");
        assert_eq!(schema.field(3).name(), "col_3");
    }

    #[test]
    fn test_schema_2d_custom_names() {
        let schema =
            ndarray_schema_2d::<f32>(3, Some(&["x", "y", "z"]), true).expect("schema failed");
        assert_eq!(schema.field(0).name(), "x");
        assert_eq!(schema.field(1).name(), "y");
        assert_eq!(schema.field(2).name(), "z");
        assert!(schema.field(0).is_nullable());
    }

    #[test]
    fn test_schema_2d_name_count_mismatch() {
        let result = ndarray_schema_2d::<f64>(3, Some(&["x", "y"]), false);
        assert!(result.is_err());
    }

    #[test]
    fn test_infer_schema_1d() {
        let arr = Array1::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let schema = infer_schema(&arr).expect("infer failed");
        assert_eq!(schema.fields().len(), 1);
        assert_eq!(schema.field(0).name(), "value");
        assert_eq!(*schema.field(0).data_type(), DataType::Float64);
    }

    #[test]
    fn test_infer_schema_2d() {
        let arr = Array2::from_shape_vec((2, 3), vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("shape error");
        let schema = infer_schema(&arr).expect("infer failed");
        assert_eq!(schema.fields().len(), 3);
    }

    #[test]
    fn test_shape_schema_roundtrip() {
        let shape = vec![10, 20, 30];
        let schema = ndarray_shape_schema(&shape, "f64");

        let recovered = extract_ndarray_shape(&schema).expect("extract failed");
        assert_eq!(shape, recovered);

        let meta = schema.metadata();
        assert_eq!(meta.get("ndarray.ndim"), Some(&"3".to_string()));
        assert_eq!(meta.get("ndarray.dtype"), Some(&"f64".to_string()));
        assert_eq!(
            meta.get("ndarray.total_elements"),
            Some(&"6000".to_string())
        );
    }

    #[test]
    fn test_schema_with_metadata() {
        let fields = vec![Field::new("data", DataType::Float64, false)];
        let schema = schema_with_metadata(fields, vec![("source", "scirs2"), ("version", "0.3.0")]);

        assert_eq!(schema.metadata().get("source"), Some(&"scirs2".to_string()));
        assert_eq!(schema.metadata().get("version"), Some(&"0.3.0".to_string()));
    }

    #[test]
    fn test_validate_schema_pass() {
        let arr =
            Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 3.0, 4.0]).expect("shape error");
        let batch =
            crate::arrow_compat::conversions::array2_to_record_batch(&arr, Some(&["a", "b"]))
                .expect("batch failed");

        let expected = Schema::new(vec![
            Field::new("a", DataType::Float64, false),
            Field::new("b", DataType::Float64, false),
        ]);

        validate_schema(&batch, &expected).expect("validation should pass");
    }

    #[test]
    fn test_validate_schema_field_count_mismatch() {
        let arr =
            Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 3.0, 4.0]).expect("shape error");
        let batch =
            crate::arrow_compat::conversions::array2_to_record_batch(&arr, Some(&["a", "b"]))
                .expect("batch failed");

        let expected = Schema::new(vec![Field::new("a", DataType::Float64, false)]);

        assert!(validate_schema(&batch, &expected).is_err());
    }

    #[test]
    fn test_validate_schema_type_mismatch() {
        let arr = Array2::from_shape_vec((2, 1), vec![1.0_f64, 2.0]).expect("shape error");
        let batch = crate::arrow_compat::conversions::array2_to_record_batch(&arr, Some(&["a"]))
            .expect("batch failed");

        let expected = Schema::new(vec![Field::new("a", DataType::Int32, false)]);

        assert!(validate_schema(&batch, &expected).is_err());
    }

    #[test]
    fn test_extract_shape_missing_metadata() {
        let schema = Schema::empty();
        assert!(extract_ndarray_shape(&schema).is_err());
    }
}
