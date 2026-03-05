//! Tensor conversion utilities between ONNX tensors and ndarray arrays.
//!
//! Provides bidirectional conversion between `OnnxTensor` and
//! `scirs2_core::ndarray` arrays (Array1, Array2, ArrayD).

use scirs2_core::ndarray::{Array1, Array2, ArrayD, IxDyn};

use super::error::{OnnxError, OnnxResult};
use super::types::{OnnxDataType, OnnxTensor};

// ---------------------------------------------------------------------------
// OnnxTensor -> ndarray conversions
// ---------------------------------------------------------------------------

/// Convert an OnnxTensor to a 2D f64 array.
///
/// The tensor must have exactly 2 dimensions. Data is read from
/// `double_data` (Float64) or `float_data` (Float32, promoted to f64).
///
/// # Example
/// ```
/// use scirs2_autograd::onnx::{OnnxTensor, onnx_tensor_to_array2};
///
/// let t = OnnxTensor::from_f64("w", &[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let arr = onnx_tensor_to_array2(&t).expect("conversion failed");
/// assert_eq!(arr.shape(), &[2, 3]);
/// assert!((arr[[1, 2]] - 6.0).abs() < 1e-12);
/// ```
pub fn onnx_tensor_to_array2(tensor: &OnnxTensor) -> OnnxResult<Array2<f64>> {
    if tensor.dims.len() != 2 {
        return Err(OnnxError::ShapeMismatch {
            expected: vec![0, 0], // placeholder for "2D"
            actual: tensor.dims.clone(),
        });
    }

    let rows = dim_to_usize(tensor.dims[0], &tensor.name, 0)?;
    let cols = dim_to_usize(tensor.dims[1], &tensor.name, 1)?;
    let expected_len = rows * cols;

    let data = extract_f64_data(tensor, expected_len)?;

    Array2::from_shape_vec((rows, cols), data).map_err(|e| OnnxError::ShapeMismatch {
        expected: tensor.dims.clone(),
        actual: vec![data_len_i64(tensor)],
    })
}

/// Convert an OnnxTensor to a 1D f64 array.
///
/// The tensor must have exactly 1 dimension.
///
/// # Example
/// ```
/// use scirs2_autograd::onnx::{OnnxTensor, onnx_tensor_to_array1};
///
/// let t = OnnxTensor::from_f64("b", &[3], vec![0.1, 0.2, 0.3]);
/// let arr = onnx_tensor_to_array1(&t).expect("conversion failed");
/// assert_eq!(arr.len(), 3);
/// ```
pub fn onnx_tensor_to_array1(tensor: &OnnxTensor) -> OnnxResult<Array1<f64>> {
    if tensor.dims.len() != 1 {
        return Err(OnnxError::ShapeMismatch {
            expected: vec![0], // placeholder for "1D"
            actual: tensor.dims.clone(),
        });
    }

    let len = dim_to_usize(tensor.dims[0], &tensor.name, 0)?;
    let data = extract_f64_data(tensor, len)?;

    Array1::from_shape_vec(len, data).map_err(|_e| OnnxError::ShapeMismatch {
        expected: tensor.dims.clone(),
        actual: vec![data_len_i64(tensor)],
    })
}

/// Convert an OnnxTensor to a dynamic-dimensional f64 array.
///
/// Works with tensors of any dimensionality.
pub fn onnx_tensor_to_arrayd(tensor: &OnnxTensor) -> OnnxResult<ArrayD<f64>> {
    let shape: Vec<usize> = tensor
        .dims
        .iter()
        .enumerate()
        .map(|(i, &d)| dim_to_usize(d, &tensor.name, i))
        .collect::<OnnxResult<Vec<_>>>()?;

    let expected_len: usize = shape.iter().product();
    let data = extract_f64_data(tensor, expected_len)?;

    ArrayD::from_shape_vec(IxDyn(&shape), data).map_err(|_e| OnnxError::ShapeMismatch {
        expected: tensor.dims.clone(),
        actual: vec![data_len_i64(tensor)],
    })
}

/// Convert an OnnxTensor to a 2D f32 array.
pub fn onnx_tensor_to_array2_f32(tensor: &OnnxTensor) -> OnnxResult<Array2<f32>> {
    if tensor.dims.len() != 2 {
        return Err(OnnxError::ShapeMismatch {
            expected: vec![0, 0],
            actual: tensor.dims.clone(),
        });
    }

    let rows = dim_to_usize(tensor.dims[0], &tensor.name, 0)?;
    let cols = dim_to_usize(tensor.dims[1], &tensor.name, 1)?;
    let expected_len = rows * cols;

    let data = extract_f32_data(tensor, expected_len)?;

    Array2::from_shape_vec((rows, cols), data).map_err(|_e| OnnxError::ShapeMismatch {
        expected: tensor.dims.clone(),
        actual: vec![data_len_i64(tensor)],
    })
}

/// Convert an OnnxTensor to a 1D f32 array.
pub fn onnx_tensor_to_array1_f32(tensor: &OnnxTensor) -> OnnxResult<Array1<f32>> {
    if tensor.dims.len() != 1 {
        return Err(OnnxError::ShapeMismatch {
            expected: vec![0],
            actual: tensor.dims.clone(),
        });
    }

    let len = dim_to_usize(tensor.dims[0], &tensor.name, 0)?;
    let data = extract_f32_data(tensor, len)?;

    Array1::from_shape_vec(len, data).map_err(|_e| OnnxError::ShapeMismatch {
        expected: tensor.dims.clone(),
        actual: vec![data_len_i64(tensor)],
    })
}

// ---------------------------------------------------------------------------
// ndarray -> OnnxTensor conversions
// ---------------------------------------------------------------------------

/// Convert a 2D f64 array to an OnnxTensor.
///
/// # Example
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_autograd::onnx::{array2_to_onnx_tensor, OnnxDataType};
///
/// let arr = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
///     .expect("shape error");
/// let tensor = array2_to_onnx_tensor("weight", &arr);
/// assert_eq!(tensor.data_type, OnnxDataType::Float64);
/// assert_eq!(tensor.dims, vec![2, 3]);
/// assert_eq!(tensor.double_data.len(), 6);
/// ```
pub fn array2_to_onnx_tensor(name: &str, array: &Array2<f64>) -> OnnxTensor {
    let shape = array.shape();
    let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    // Use standard-layout (C-contiguous) iteration order
    let data: Vec<f64> = array.iter().copied().collect();
    OnnxTensor::from_f64(name, &dims, data)
}

/// Convert a 1D f64 array to an OnnxTensor.
pub fn array1_to_onnx_tensor(name: &str, array: &Array1<f64>) -> OnnxTensor {
    let dims = vec![array.len() as i64];
    let data: Vec<f64> = array.iter().copied().collect();
    OnnxTensor::from_f64(name, &dims, data)
}

/// Convert a dynamic-dimensional f64 array to an OnnxTensor.
pub fn arrayd_to_onnx_tensor(name: &str, array: &ArrayD<f64>) -> OnnxTensor {
    let dims: Vec<i64> = array.shape().iter().map(|&d| d as i64).collect();
    let data: Vec<f64> = array.iter().copied().collect();
    OnnxTensor::from_f64(name, &dims, data)
}

/// Convert a 2D f32 array to an OnnxTensor.
pub fn array2_to_onnx_tensor_f32(name: &str, array: &Array2<f32>) -> OnnxTensor {
    let shape = array.shape();
    let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    let data: Vec<f32> = array.iter().copied().collect();
    OnnxTensor::from_f32(name, &dims, data)
}

/// Convert a 1D f32 array to an OnnxTensor.
pub fn array1_to_onnx_tensor_f32(name: &str, array: &Array1<f32>) -> OnnxTensor {
    let dims = vec![array.len() as i64];
    let data: Vec<f32> = array.iter().copied().collect();
    OnnxTensor::from_f32(name, &dims, data)
}

/// Export a list of named weight arrays as ONNX tensors.
///
/// This is a convenience function for exporting model weights.
///
/// # Example
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_autograd::onnx::export_weights;
///
/// let w1 = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
///     .expect("shape error");
/// let w2 = Array2::from_shape_vec((2, 1), vec![7.0, 8.0])
///     .expect("shape error");
///
/// let tensors = export_weights(&[
///     ("layer1.weight".to_string(), w1),
///     ("layer2.weight".to_string(), w2),
/// ]);
///
/// assert_eq!(tensors.len(), 2);
/// assert_eq!(tensors[0].name, "layer1.weight");
/// assert_eq!(tensors[1].name, "layer2.weight");
/// ```
pub fn export_weights(tensors: &[(String, Array2<f64>)]) -> Vec<OnnxTensor> {
    tensors
        .iter()
        .map(|(name, arr)| array2_to_onnx_tensor(name, arr))
        .collect()
}

/// Import weight tensors from a list of OnnxTensors, converting to f64 Array2.
///
/// Returns a Vec of (name, `Array2<f64>`) pairs.
/// Tensors that cannot be converted to 2D are skipped with a warning in the errors vec.
pub fn import_weights(onnx_tensors: &[OnnxTensor]) -> (Vec<(String, Array2<f64>)>, Vec<OnnxError>) {
    let mut weights = Vec::new();
    let mut errors = Vec::new();

    for tensor in onnx_tensors {
        match onnx_tensor_to_array2(tensor) {
            Ok(arr) => weights.push((tensor.name.clone(), arr)),
            Err(e) => errors.push(e),
        }
    }

    (weights, errors)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Convert a dimension value to usize, handling negative (dynamic) dimensions
fn dim_to_usize(dim: i64, tensor_name: &str, axis: usize) -> OnnxResult<usize> {
    if dim < 0 {
        return Err(OnnxError::DataTypeError(format!(
            "Tensor '{}' has dynamic dimension {} at axis {} — cannot convert to concrete array",
            tensor_name, dim, axis
        )));
    }
    Ok(dim as usize)
}

/// Extract f64 data from an OnnxTensor, promoting from f32 if necessary
fn extract_f64_data(tensor: &OnnxTensor, expected_len: usize) -> OnnxResult<Vec<f64>> {
    match tensor.data_type {
        OnnxDataType::Float64 => {
            if tensor.double_data.len() == expected_len {
                Ok(tensor.double_data.clone())
            } else if !tensor.raw_data.is_empty() {
                f64_from_raw_bytes(&tensor.raw_data, expected_len, &tensor.name)
            } else {
                Err(OnnxError::DataTypeError(format!(
                    "Tensor '{}': expected {} f64 elements, got {} in double_data and {} raw bytes",
                    tensor.name,
                    expected_len,
                    tensor.double_data.len(),
                    tensor.raw_data.len()
                )))
            }
        }
        OnnxDataType::Float32 => {
            if tensor.float_data.len() == expected_len {
                Ok(tensor.float_data.iter().map(|&v| v as f64).collect())
            } else if !tensor.raw_data.is_empty() {
                let f32_data = f32_from_raw_bytes(&tensor.raw_data, expected_len, &tensor.name)?;
                Ok(f32_data.into_iter().map(|v| v as f64).collect())
            } else {
                Err(OnnxError::DataTypeError(format!(
                    "Tensor '{}': expected {} f32 elements, got {} in float_data",
                    tensor.name,
                    expected_len,
                    tensor.float_data.len()
                )))
            }
        }
        OnnxDataType::Int32 => {
            if tensor.int32_data.len() == expected_len {
                Ok(tensor.int32_data.iter().map(|&v| v as f64).collect())
            } else {
                Err(OnnxError::DataTypeError(format!(
                    "Tensor '{}': expected {} i32 elements, got {}",
                    tensor.name,
                    expected_len,
                    tensor.int32_data.len()
                )))
            }
        }
        OnnxDataType::Int64 => {
            if tensor.int64_data.len() == expected_len {
                Ok(tensor.int64_data.iter().map(|&v| v as f64).collect())
            } else {
                Err(OnnxError::DataTypeError(format!(
                    "Tensor '{}': expected {} i64 elements, got {}",
                    tensor.name,
                    expected_len,
                    tensor.int64_data.len()
                )))
            }
        }
        other => Err(OnnxError::DataTypeError(format!(
            "Tensor '{}': cannot convert {} to f64",
            tensor.name, other
        ))),
    }
}

/// Extract f32 data from an OnnxTensor
fn extract_f32_data(tensor: &OnnxTensor, expected_len: usize) -> OnnxResult<Vec<f32>> {
    match tensor.data_type {
        OnnxDataType::Float32 => {
            if tensor.float_data.len() == expected_len {
                Ok(tensor.float_data.clone())
            } else if !tensor.raw_data.is_empty() {
                f32_from_raw_bytes(&tensor.raw_data, expected_len, &tensor.name)
            } else {
                Err(OnnxError::DataTypeError(format!(
                    "Tensor '{}': expected {} f32 elements, got {}",
                    tensor.name,
                    expected_len,
                    tensor.float_data.len()
                )))
            }
        }
        OnnxDataType::Float64 => {
            if tensor.double_data.len() == expected_len {
                Ok(tensor.double_data.iter().map(|&v| v as f32).collect())
            } else {
                Err(OnnxError::DataTypeError(format!(
                    "Tensor '{}': expected {} f64 elements, got {}",
                    tensor.name,
                    expected_len,
                    tensor.double_data.len()
                )))
            }
        }
        other => Err(OnnxError::DataTypeError(format!(
            "Tensor '{}': cannot convert {} to f32",
            tensor.name, other
        ))),
    }
}

/// Decode f64 values from little-endian raw bytes
fn f64_from_raw_bytes(raw: &[u8], expected_count: usize, name: &str) -> OnnxResult<Vec<f64>> {
    let byte_size = expected_count * 8;
    if raw.len() < byte_size {
        return Err(OnnxError::DataTypeError(format!(
            "Tensor '{}': expected {} bytes for {} f64 values, got {}",
            name,
            byte_size,
            expected_count,
            raw.len()
        )));
    }

    let mut result = Vec::with_capacity(expected_count);
    for i in 0..expected_count {
        let offset = i * 8;
        let bytes: [u8; 8] = [
            raw[offset],
            raw[offset + 1],
            raw[offset + 2],
            raw[offset + 3],
            raw[offset + 4],
            raw[offset + 5],
            raw[offset + 6],
            raw[offset + 7],
        ];
        result.push(f64::from_le_bytes(bytes));
    }
    Ok(result)
}

/// Decode f32 values from little-endian raw bytes
fn f32_from_raw_bytes(raw: &[u8], expected_count: usize, name: &str) -> OnnxResult<Vec<f32>> {
    let byte_size = expected_count * 4;
    if raw.len() < byte_size {
        return Err(OnnxError::DataTypeError(format!(
            "Tensor '{}': expected {} bytes for {} f32 values, got {}",
            name,
            byte_size,
            expected_count,
            raw.len()
        )));
    }

    let mut result = Vec::with_capacity(expected_count);
    for i in 0..expected_count {
        let offset = i * 4;
        let bytes: [u8; 4] = [
            raw[offset],
            raw[offset + 1],
            raw[offset + 2],
            raw[offset + 3],
        ];
        result.push(f32::from_le_bytes(bytes));
    }
    Ok(result)
}

/// Get the total data length as i64 for error messages
fn data_len_i64(tensor: &OnnxTensor) -> i64 {
    let len = if !tensor.double_data.is_empty() {
        tensor.double_data.len()
    } else if !tensor.float_data.is_empty() {
        tensor.float_data.len()
    } else if !tensor.int32_data.is_empty() {
        tensor.int32_data.len()
    } else if !tensor.int64_data.is_empty() {
        tensor.int64_data.len()
    } else {
        tensor.raw_data.len()
    };
    len as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array2_roundtrip() {
        let arr =
            Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("shape");
        let tensor = array2_to_onnx_tensor("w", &arr);
        let restored = onnx_tensor_to_array2(&tensor).expect("conversion");
        assert_eq!(arr, restored);
    }

    #[test]
    fn test_array1_roundtrip() {
        let arr = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let tensor = array1_to_onnx_tensor("b", &arr);
        let restored = onnx_tensor_to_array1(&tensor).expect("conversion");
        assert_eq!(arr, restored);
    }

    #[test]
    fn test_f32_to_f64_promotion() {
        let tensor = OnnxTensor::from_f32("t", &[2, 2], vec![1.0f32, 2.0, 3.0, 4.0]);
        let arr = onnx_tensor_to_array2(&tensor).expect("conversion");
        assert!((arr[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((arr[[1, 1]] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_wrong_dimensionality() {
        let tensor = OnnxTensor::from_f64("t", &[2, 3, 4], vec![0.0; 24]);
        assert!(onnx_tensor_to_array2(&tensor).is_err());
        assert!(onnx_tensor_to_array1(&tensor).is_err());
    }

    #[test]
    fn test_dynamic_dimension_error() {
        let tensor = OnnxTensor::from_f64("t", &[-1, 10], vec![]);
        assert!(onnx_tensor_to_array2(&tensor).is_err());
    }

    #[test]
    fn test_export_weights() {
        let w1 = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("shape");
        let w2 = Array2::from_shape_vec((3, 1), vec![7.0, 8.0, 9.0]).expect("shape");

        let tensors = export_weights(&[
            ("layer1.weight".to_string(), w1.clone()),
            ("layer2.weight".to_string(), w2.clone()),
        ]);

        assert_eq!(tensors.len(), 2);
        assert_eq!(tensors[0].name, "layer1.weight");
        assert_eq!(tensors[0].dims, vec![2, 3]);
        assert_eq!(tensors[1].name, "layer2.weight");

        // Roundtrip
        let (imported, errors) = import_weights(&tensors);
        assert!(errors.is_empty());
        assert_eq!(imported.len(), 2);
        assert_eq!(imported[0].1, w1);
        assert_eq!(imported[1].1, w2);
    }

    #[test]
    fn test_arrayd_conversion() {
        let arr = ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), (0..24).map(|i| i as f64).collect())
            .expect("shape");
        let tensor = arrayd_to_onnx_tensor("nd", &arr);
        let restored = onnx_tensor_to_arrayd(&tensor).expect("conversion");
        assert_eq!(arr, restored);
    }

    #[test]
    fn test_i32_to_f64_conversion() {
        let tensor = OnnxTensor::from_i32("ints", &[3], vec![10, 20, 30]);
        let arr = onnx_tensor_to_array1(&tensor).expect("conversion");
        assert!((arr[0] - 10.0).abs() < 1e-12);
        assert!((arr[2] - 30.0).abs() < 1e-12);
    }

    #[test]
    fn test_f32_array_roundtrip() {
        let arr = Array2::from_shape_vec((2, 2), vec![1.0f32, 2.0, 3.0, 4.0]).expect("shape");
        let tensor = array2_to_onnx_tensor_f32("w_f32", &arr);
        let restored = onnx_tensor_to_array2_f32(&tensor).expect("conversion");
        assert_eq!(arr, restored);
    }
}
