//! Random number generation for WASM

use crate::array::WasmArray;
use crate::error::WasmError;
use crate::utils::parse_shape;
use scirs2_core::ndarray::ArrayD;
use scirs2_core::random::*;
use wasm_bindgen::prelude::*;

/// Generate random numbers from a uniform distribution [0, 1)
#[wasm_bindgen]
pub fn random_uniform(shape: &JsValue) -> Result<WasmArray, JsValue> {
    let shape_vec = parse_shape(shape)?;
    let total_size: usize = shape_vec.iter().product();

    let mut rng = thread_rng();
    let vec: Vec<f64> = (0..total_size).map(|_| rng.random::<f64>()).collect();

    let array = ArrayD::from_shape_vec(shape_vec, vec)
        .map_err(|e: ndarray::ShapeError| WasmError::ComputationError(e.to_string()))?;

    Ok(WasmArray::from_array(array))
}

/// Generate random numbers from a normal (Gaussian) distribution
#[wasm_bindgen]
pub fn random_normal(shape: &JsValue, mean: f64, std_dev: f64) -> Result<WasmArray, JsValue> {
    if std_dev <= 0.0 {
        return Err(
            WasmError::InvalidParameter("Standard deviation must be positive".to_string()).into(),
        );
    }

    let shape_vec = parse_shape(shape)?;
    let total_size: usize = shape_vec.iter().product();

    let mut rng = thread_rng();
    let dist =
        RandNormal::new(mean, std_dev).map_err(|e| WasmError::ComputationError(e.to_string()))?;

    let vec: Vec<f64> = (0..total_size).map(|_| dist.sample(&mut rng)).collect();

    let array = ArrayD::from_shape_vec(shape_vec, vec)
        .map_err(|e: ndarray::ShapeError| WasmError::ComputationError(e.to_string()))?;

    Ok(WasmArray::from_array(array))
}

/// Generate random integers in the range [low, high)
#[wasm_bindgen]
pub fn random_integers(shape: &JsValue, low: i32, high: i32) -> Result<WasmArray, JsValue> {
    if low >= high {
        return Err(WasmError::InvalidParameter("low must be less than high".to_string()).into());
    }

    let shape_vec = parse_shape(shape)?;
    let total_size: usize = shape_vec.iter().product();

    let mut rng = thread_rng();
    let dist =
        RandUniform::new(low, high).map_err(|e| WasmError::ComputationError(e.to_string()))?;

    let vec: Vec<f64> = (0..total_size)
        .map(|_| dist.sample(&mut rng) as f64)
        .collect();

    let array = ArrayD::from_shape_vec(shape_vec, vec)
        .map_err(|e: ndarray::ShapeError| WasmError::ComputationError(e.to_string()))?;

    Ok(WasmArray::from_array(array))
}

/// Generate random numbers from an exponential distribution
#[wasm_bindgen]
pub fn random_exponential(shape: &JsValue, lambda: f64) -> Result<WasmArray, JsValue> {
    if lambda <= 0.0 {
        return Err(WasmError::InvalidParameter("Lambda must be positive".to_string()).into());
    }

    let shape_vec = parse_shape(shape)?;
    let total_size: usize = shape_vec.iter().product();

    let mut rng = thread_rng();
    let dist = Exponential::new(lambda).map_err(|e| WasmError::ComputationError(e.to_string()))?;

    let vec: Vec<f64> = (0..total_size).map(|_| dist.sample(&mut rng)).collect();

    let array = ArrayD::from_shape_vec(shape_vec, vec)
        .map_err(|e: ndarray::ShapeError| WasmError::ComputationError(e.to_string()))?;

    Ok(WasmArray::from_array(array))
}

/// Set the random seed for reproducibility
#[wasm_bindgen]
pub fn set_random_seed(seed: u64) -> String {
    // Note: In WASM, setting a global seed is not straightforward
    // This is a placeholder implementation
    format!(
        "Note: Seed setting in WASM is limited. Provided seed: {}",
        seed
    )
}
