//! N-dimensional array operations for WASM

use crate::error::WasmError;
use crate::utils::{js_array_to_vec_f64, parse_shape, typed_array_to_vec_f64};
use scirs2_core::ndarray::{Array1, ArrayD};
use wasm_bindgen::prelude::*;

/// A wrapper around ndarray for JavaScript interop
#[wasm_bindgen]
pub struct WasmArray {
    data: ArrayD<f64>,
}

// Internal implementation for accessing data
impl WasmArray {
    pub(crate) fn from_array(data: ArrayD<f64>) -> Self {
        Self { data }
    }

    pub(crate) fn data(&self) -> &ArrayD<f64> {
        &self.data
    }
}

#[wasm_bindgen]
impl WasmArray {
    /// Create a 1D array from a JavaScript array or typed array
    #[wasm_bindgen(constructor)]
    pub fn new(data: &JsValue) -> Result<WasmArray, JsValue> {
        let vec = if data.is_array() {
            let array = js_sys::Array::from(data);
            js_array_to_vec_f64(&array)?
        } else {
            typed_array_to_vec_f64(data)?
        };

        let array = Array1::from_vec(vec).into_dyn();
        Ok(WasmArray { data: array })
    }

    /// Create an array with a specific shape from flat data
    #[wasm_bindgen]
    pub fn from_shape(shape: &JsValue, data: &JsValue) -> Result<WasmArray, JsValue> {
        let shape_vec = parse_shape(shape)?;
        let data_vec = if data.is_array() {
            let array = js_sys::Array::from(data);
            js_array_to_vec_f64(&array)?
        } else {
            typed_array_to_vec_f64(data)?
        };

        let total_size: usize = shape_vec.iter().product();
        if data_vec.len() != total_size {
            return Err(WasmError::ShapeMismatch {
                expected: vec![total_size],
                actual: vec![data_vec.len()],
            }
            .into());
        }

        let array = ArrayD::from_shape_vec(shape_vec, data_vec)
            .map_err(|e: ndarray::ShapeError| WasmError::InvalidDimensions(e.to_string()))?;

        Ok(WasmArray { data: array })
    }

    /// Create an array of zeros with the given shape
    #[wasm_bindgen]
    pub fn zeros(shape: &JsValue) -> Result<WasmArray, JsValue> {
        let shape_vec = parse_shape(shape)?;
        let array = ArrayD::zeros(shape_vec);
        Ok(WasmArray { data: array })
    }

    /// Create an array of ones with the given shape
    #[wasm_bindgen]
    pub fn ones(shape: &JsValue) -> Result<WasmArray, JsValue> {
        let shape_vec = parse_shape(shape)?;
        let array = ArrayD::ones(shape_vec);
        Ok(WasmArray { data: array })
    }

    /// Create an array filled with a constant value
    #[wasm_bindgen]
    pub fn full(shape: &JsValue, value: f64) -> Result<WasmArray, JsValue> {
        let shape_vec = parse_shape(shape)?;
        let array = ArrayD::from_elem(shape_vec, value);
        Ok(WasmArray { data: array })
    }

    /// Create an evenly spaced array (like numpy.linspace)
    #[wasm_bindgen]
    pub fn linspace(start: f64, end: f64, num: usize) -> Result<WasmArray, JsValue> {
        if num == 0 {
            return Err(WasmError::InvalidParameter("num must be > 0".to_string()).into());
        }

        let step = if num > 1 {
            (end - start) / (num - 1) as f64
        } else {
            0.0
        };

        let vec: Vec<f64> = (0..num).map(|i| start + i as f64 * step).collect();

        let array = Array1::from_vec(vec).into_dyn();
        Ok(WasmArray { data: array })
    }

    /// Create an array with evenly spaced values (like numpy.arange)
    #[wasm_bindgen]
    pub fn arange(start: f64, end: f64, step: f64) -> Result<WasmArray, JsValue> {
        if step == 0.0 {
            return Err(WasmError::InvalidParameter("step cannot be zero".to_string()).into());
        }

        if (end - start).signum() != step.signum() {
            return Err(WasmError::InvalidParameter(
                "step direction does not match range".to_string(),
            )
            .into());
        }

        let num = ((end - start) / step).abs().ceil() as usize;
        let vec: Vec<f64> = (0..num).map(|i| start + i as f64 * step).collect();

        let array = Array1::from_vec(vec).into_dyn();
        Ok(WasmArray { data: array })
    }

    /// Get the shape of the array
    #[wasm_bindgen]
    pub fn shape(&self) -> js_sys::Array {
        let shape = self.data.shape();
        let array = js_sys::Array::new_with_length(shape.len() as u32);

        for (i, &dim) in shape.iter().enumerate() {
            array.set(i as u32, JsValue::from_f64(dim as f64));
        }

        array
    }

    /// Get the number of dimensions
    #[wasm_bindgen]
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Get the total number of elements
    #[wasm_bindgen]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the array is empty
    #[wasm_bindgen]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Convert to a flat JavaScript Float64Array
    #[wasm_bindgen]
    pub fn to_array(&self) -> js_sys::Float64Array {
        let vec: Vec<f64> = self.data.iter().copied().collect();
        let array = js_sys::Float64Array::new_with_length(vec.len() as u32);
        array.copy_from(&vec);
        array
    }

    /// Convert to a JavaScript array (nested for multi-dimensional arrays)
    #[wasm_bindgen]
    pub fn to_nested_array(&self) -> JsValue {
        // For simplicity, return flat array with shape info
        // In production, implement proper nested array conversion
        let vec: Vec<f64> = self.data.iter().copied().collect();
        serde_wasm_bindgen::to_value(&vec).unwrap_or(JsValue::NULL)
    }

    /// Get a value at the specified index (flat indexing)
    #[wasm_bindgen]
    pub fn get(&self, index: usize) -> Result<f64, JsValue> {
        self.data
            .as_slice()
            .and_then(|s| s.get(index).copied())
            .ok_or_else(|| {
                WasmError::IndexOutOfBounds(format!(
                    "Index {} out of bounds for array of length {}",
                    index,
                    self.len()
                ))
                .into()
            })
    }

    /// Set a value at the specified index (flat indexing)
    #[wasm_bindgen]
    pub fn set(&mut self, index: usize, value: f64) -> Result<(), JsValue> {
        self.data
            .as_slice_mut()
            .and_then(|s| s.get_mut(index))
            .map(|v| *v = value)
            .ok_or_else(|| {
                WasmError::IndexOutOfBounds(format!(
                    "Index {} out of bounds for array of length {}",
                    index,
                    self.len()
                ))
                .into()
            })
    }

    /// Reshape the array
    #[wasm_bindgen]
    pub fn reshape(&self, new_shape: &JsValue) -> Result<WasmArray, JsValue> {
        let shape_vec = parse_shape(new_shape)?;
        let total_size: usize = shape_vec.iter().product();

        if total_size != self.len() {
            return Err(WasmError::ShapeMismatch {
                expected: vec![self.len()],
                actual: vec![total_size],
            }
            .into());
        }

        let vec: Vec<f64> = self.data.iter().copied().collect();
        let array = ArrayD::from_shape_vec(shape_vec, vec)
            .map_err(|e: ndarray::ShapeError| WasmError::InvalidDimensions(e.to_string()))?;

        Ok(WasmArray { data: array })
    }

    /// Transpose the array (2D only for now)
    #[wasm_bindgen]
    pub fn transpose(&self) -> Result<WasmArray, JsValue> {
        if self.ndim() != 2 {
            return Err(WasmError::InvalidDimensions(
                "Transpose is only supported for 2D arrays".to_string(),
            )
            .into());
        }

        let transposed = self
            .data
            .clone()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e: ndarray::ShapeError| WasmError::ComputationError(e.to_string()))?
            .t()
            .to_owned()
            .into_dyn();

        Ok(WasmArray { data: transposed })
    }

    /// Clone the array
    #[allow(clippy::should_implement_trait)]
    #[wasm_bindgen]
    pub fn clone(&self) -> WasmArray {
        WasmArray {
            data: self.data.clone(),
        }
    }
}

/// Add two arrays element-wise
#[wasm_bindgen]
pub fn add(a: &WasmArray, b: &WasmArray) -> Result<WasmArray, JsValue> {
    if a.data().shape() != b.data().shape() {
        return Err(WasmError::ShapeMismatch {
            expected: a.data().shape().to_vec(),
            actual: b.data().shape().to_vec(),
        }
        .into());
    }

    Ok(WasmArray {
        data: a.data() + b.data(),
    })
}

/// Subtract two arrays element-wise
#[wasm_bindgen]
pub fn subtract(a: &WasmArray, b: &WasmArray) -> Result<WasmArray, JsValue> {
    if a.data().shape() != b.data().shape() {
        return Err(WasmError::ShapeMismatch {
            expected: a.data().shape().to_vec(),
            actual: b.data().shape().to_vec(),
        }
        .into());
    }

    Ok(WasmArray {
        data: a.data() - b.data(),
    })
}

/// Multiply two arrays element-wise
#[wasm_bindgen]
pub fn multiply(a: &WasmArray, b: &WasmArray) -> Result<WasmArray, JsValue> {
    if a.data().shape() != b.data().shape() {
        return Err(WasmError::ShapeMismatch {
            expected: a.data().shape().to_vec(),
            actual: b.data().shape().to_vec(),
        }
        .into());
    }

    Ok(WasmArray {
        data: a.data() * b.data(),
    })
}

/// Divide two arrays element-wise
#[wasm_bindgen]
pub fn divide(a: &WasmArray, b: &WasmArray) -> Result<WasmArray, JsValue> {
    if a.data().shape() != b.data().shape() {
        return Err(WasmError::ShapeMismatch {
            expected: a.data().shape().to_vec(),
            actual: b.data().shape().to_vec(),
        }
        .into());
    }

    Ok(WasmArray {
        data: a.data() / b.data(),
    })
}

/// Compute dot product (1D) or matrix multiplication (2D)
#[wasm_bindgen]
pub fn dot(a: &WasmArray, b: &WasmArray) -> Result<WasmArray, JsValue> {
    match (a.ndim(), b.ndim()) {
        (1, 1) => {
            // 1D dot product
            let a1 = a
                .data()
                .clone()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|e: ndarray::ShapeError| WasmError::ComputationError(e.to_string()))?;
            let b1 = b
                .data()
                .clone()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|e: ndarray::ShapeError| WasmError::ComputationError(e.to_string()))?;

            let result = a1.dot(&b1);
            let array = ArrayD::from_elem(vec![], result);
            Ok(WasmArray { data: array })
        }
        (2, 2) => {
            // Matrix multiplication
            let a2 = a
                .data()
                .clone()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e: ndarray::ShapeError| WasmError::ComputationError(e.to_string()))?;
            let b2 = b
                .data()
                .clone()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e: ndarray::ShapeError| WasmError::ComputationError(e.to_string()))?;

            if a2.ncols() != b2.nrows() {
                return Err(WasmError::ShapeMismatch {
                    expected: vec![a2.nrows(), b2.ncols()],
                    actual: vec![a2.nrows(), a2.ncols(), b2.nrows(), b2.ncols()],
                }
                .into());
            }

            let result = a2.dot(&b2).into_dyn();
            Ok(WasmArray { data: result })
        }
        _ => Err(WasmError::InvalidDimensions(
            "dot only supports 1D-1D or 2D-2D arrays".to_string(),
        )
        .into()),
    }
}

/// Sum all elements in the array
#[wasm_bindgen]
pub fn sum(arr: &WasmArray) -> f64 {
    arr.data().sum()
}

/// Compute the mean of all elements
#[wasm_bindgen]
pub fn mean(arr: &WasmArray) -> f64 {
    if arr.is_empty() {
        return f64::NAN;
    }
    arr.data().sum() / arr.len() as f64
}

/// Find the minimum value
#[wasm_bindgen]
pub fn min(arr: &WasmArray) -> f64 {
    arr.data().iter().copied().fold(f64::INFINITY, f64::min)
}

/// Find the maximum value
#[wasm_bindgen]
pub fn max(arr: &WasmArray) -> f64 {
    arr.data().iter().copied().fold(f64::NEG_INFINITY, f64::max)
}
