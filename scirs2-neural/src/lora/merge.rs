//! Weight merging and quantization utilities for LoRA.
//!
//! Provides functions to merge LoRA weights into the base model,
//! compute effective weights, and quantize merged weights for efficient inference.

use scirs2_core::ndarray::Array2;

use super::linear::LoRALinear;
use crate::{NeuralError, Result};

/// Merge LoRA low-rank matrices into a weight matrix.
///
/// Computes: weight + scaling * B @ A
///
/// # Arguments
///
/// * `weight` - Original weight matrix [out_features x in_features]
/// * `a` - LoRA A matrix [rank x in_features]
/// * `b` - LoRA B matrix [out_features x rank]
/// * `scaling` - Scaling factor (typically alpha/rank)
///
/// # Errors
///
/// Returns an error if matrix dimensions are incompatible.
pub fn merge_lora_weights(
    weight: &Array2<f64>,
    a: &Array2<f64>,
    b: &Array2<f64>,
    scaling: f64,
) -> Result<Array2<f64>> {
    // Validate dimensions
    let (out_features, in_features) = (weight.nrows(), weight.ncols());
    let (a_rows, a_cols) = (a.nrows(), a.ncols());
    let (b_rows, b_cols) = (b.nrows(), b.ncols());

    if a_cols != in_features {
        return Err(NeuralError::DimensionMismatch(format!(
            "A columns ({a_cols}) must match weight columns ({in_features})"
        )));
    }
    if b_rows != out_features {
        return Err(NeuralError::DimensionMismatch(format!(
            "B rows ({b_rows}) must match weight rows ({out_features})"
        )));
    }
    if b_cols != a_rows {
        return Err(NeuralError::DimensionMismatch(format!(
            "B columns ({b_cols}) must match A rows ({a_rows})"
        )));
    }

    let delta = b.dot(a) * scaling;
    Ok(weight + &delta)
}

/// Compute the effective weight of a LoRA linear layer.
///
/// Returns W + (alpha/rank) * B @ A, regardless of merge state.
///
/// # Errors
///
/// Returns an error if the layer is in an inconsistent state.
pub fn compute_effective_weight(lora: &LoRALinear) -> Result<Array2<f64>> {
    Ok(lora.effective_weight())
}

/// Simple INT8 quantized weight representation.
///
/// Stores weights as 8-bit integers with a per-tensor scale and zero-point
/// for dequantization: float_value = (int_value - zero_point) * scale
#[derive(Debug, Clone)]
pub struct QuantizedWeight {
    /// Quantized INT8 values.
    data: Vec<i8>,
    /// Number of rows.
    rows: usize,
    /// Number of columns.
    cols: usize,
    /// Scale factor for dequantization.
    scale: f64,
    /// Zero point for asymmetric quantization.
    zero_point: i8,
    /// Number of bits used (always 8 for now).
    bits: u8,
}

impl QuantizedWeight {
    /// Get the quantized data.
    pub fn data(&self) -> &[i8] {
        &self.data
    }

    /// Get the number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get the scale factor.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Get the zero point.
    pub fn zero_point(&self) -> i8 {
        self.zero_point
    }

    /// Get the number of quantization bits.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Dequantize back to f64 array.
    ///
    /// # Errors
    ///
    /// Returns an error if internal dimensions are inconsistent.
    pub fn dequantize(&self) -> Result<Array2<f64>> {
        let values: Vec<f64> = self
            .data
            .iter()
            .map(|&v| (v as f64 - self.zero_point as f64) * self.scale)
            .collect();

        Array2::from_shape_vec((self.rows, self.cols), values)
            .map_err(|e| NeuralError::ShapeMismatch(format!("Dequantization shape error: {e}")))
    }

    /// Compute the memory savings ratio compared to f64.
    pub fn compression_ratio(&self) -> f64 {
        // f64 = 8 bytes per element, i8 = 1 byte per element
        // Plus overhead for scale + zero_point (negligible for large tensors)
        1.0 / 8.0
    }

    /// Compute the quantization error (RMSE) against original weight.
    pub fn quantization_error(&self, original: &Array2<f64>) -> Result<f64> {
        let dequantized = self.dequantize()?;
        if dequantized.shape() != original.shape() {
            return Err(NeuralError::ShapeMismatch(format!(
                "Original shape {:?} != dequantized shape {:?}",
                original.shape(),
                dequantized.shape()
            )));
        }

        let n = original.len() as f64;
        let mse: f64 = dequantized
            .iter()
            .zip(original.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            / n;

        Ok(mse.sqrt())
    }
}

/// Quantize a weight matrix to INT8.
///
/// Uses symmetric quantization: the range [-max_abs, max_abs] is mapped
/// to [-127, 127] with zero_point = 0.
///
/// # Arguments
///
/// * `weight` - Weight matrix to quantize
/// * `bits` - Number of bits (currently only 8 is supported)
///
/// # Errors
///
/// Returns an error if bits != 8 or the weight is empty.
///
/// # Example
///
/// ```
/// use scirs2_neural::lora::quantize_merged_weight;
/// use scirs2_core::ndarray::Array2;
///
/// let weight = Array2::from_shape_fn((4, 4), |(i, j)| (i as f64 - j as f64) * 0.5);
/// let quantized = quantize_merged_weight(&weight, 8).expect("quantization failed");
/// assert_eq!(quantized.bits(), 8);
/// ```
pub fn quantize_merged_weight(weight: &Array2<f64>, bits: u8) -> Result<QuantizedWeight> {
    if bits != 8 {
        return Err(NeuralError::InvalidArgument(format!(
            "Only 8-bit quantization is currently supported, got {bits}"
        )));
    }

    let (rows, cols) = (weight.nrows(), weight.ncols());
    if rows == 0 || cols == 0 {
        return Err(NeuralError::InvalidArgument(
            "Cannot quantize empty weight matrix".to_string(),
        ));
    }

    // Find the maximum absolute value for symmetric quantization
    let max_abs = weight.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

    // Handle the case where all values are zero
    if max_abs < f64::EPSILON {
        return Ok(QuantizedWeight {
            data: vec![0i8; rows * cols],
            rows,
            cols,
            scale: 1.0, // arbitrary, all values are 0
            zero_point: 0,
            bits,
        });
    }

    // Symmetric quantization: scale = max_abs / 127
    let scale = max_abs / 127.0;
    let zero_point = 0i8;

    let data: Vec<i8> = weight
        .iter()
        .map(|&v| {
            let quantized = (v / scale).round();
            // Clamp to i8 range
            quantized.clamp(-128.0, 127.0) as i8
        })
        .collect();

    Ok(QuantizedWeight {
        data,
        rows,
        cols,
        scale,
        zero_point,
        bits,
    })
}

/// Merge multiple LoRA adapters into a single weight.
///
/// Useful when composing multiple LoRA modules trained for different tasks.
///
/// # Arguments
///
/// * `base_weight` - Original base weight
/// * `lora_params` - List of (A, B, scaling) tuples
///
/// # Errors
///
/// Returns an error if any dimensions are incompatible.
pub fn merge_multiple_lora(
    base_weight: &Array2<f64>,
    lora_params: &[(Array2<f64>, Array2<f64>, f64)],
) -> Result<Array2<f64>> {
    let mut result = base_weight.clone();
    for (a, b, scaling) in lora_params {
        result = merge_lora_weights(&result, a, b, *scaling)?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_merge_lora_weights() {
        let weight = Array2::<f64>::eye(4);
        let a = Array2::from_shape_fn((2, 4), |(i, j)| (i * 4 + j) as f64 * 0.1);
        let b = Array2::from_shape_fn((4, 2), |(i, j)| (i * 2 + j) as f64 * 0.1);
        let scaling = 0.5;

        let merged = merge_lora_weights(&weight, &a, &b, scaling);
        assert!(merged.is_ok());
        let merged = merged.expect("merge failed");
        assert_eq!(merged.shape(), &[4, 4]);

        // Verify: merged = eye + 0.5 * b @ a
        let expected = &weight + &(b.dot(&a) * scaling);
        for (m, e) in merged.iter().zip(expected.iter()) {
            assert!((m - e).abs() < 1e-10);
        }
    }

    #[test]
    fn test_merge_dimension_mismatch() {
        let weight = Array2::<f64>::eye(4);
        let a = Array2::<f64>::zeros((2, 3)); // wrong: should be 4 cols
        let b = Array2::<f64>::zeros((4, 2));

        assert!(merge_lora_weights(&weight, &a, &b, 1.0).is_err());
    }

    #[test]
    fn test_compute_effective_weight() {
        let weight = Array2::<f64>::eye(4);
        let config = super::super::types::LoRAConfig {
            rank: 2,
            ..Default::default()
        };
        let lora = LoRALinear::new(weight.clone(), &config).expect("creation failed");

        let effective = compute_effective_weight(&lora).expect("compute failed");
        // B is zeros, so effective should be same as original
        for (a, b) in effective.iter().zip(weight.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_quantize_int8() {
        let weight = Array2::from_shape_fn((4, 4), |(i, j)| (i as f64 - j as f64) * 0.5);
        let quantized = quantize_merged_weight(&weight, 8).expect("quantization failed");

        assert_eq!(quantized.bits(), 8);
        assert_eq!(quantized.rows(), 4);
        assert_eq!(quantized.cols(), 4);

        // Verify we have actual data
        assert_eq!(quantized.data().len(), 16);
    }

    #[test]
    fn test_quantize_dequantize_error() {
        let weight =
            Array2::from_shape_fn((8, 8), |(i, j)| ((i as f64 - 4.0) * (j as f64 - 4.0)) * 0.1);
        let quantized = quantize_merged_weight(&weight, 8).expect("quantization failed");
        let error = quantized
            .quantization_error(&weight)
            .expect("error computation failed");

        // Quantization error should be small relative to weight magnitude
        let max_abs = weight.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(
            error < max_abs * 0.05,
            "quantization error {error} too large relative to max {max_abs}"
        );
    }

    #[test]
    fn test_quantize_zero_weight() {
        let weight = Array2::<f64>::zeros((4, 4));
        let quantized = quantize_merged_weight(&weight, 8).expect("quantization failed");
        for &v in quantized.data() {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn test_quantize_unsupported_bits() {
        let weight = Array2::<f64>::eye(4);
        assert!(quantize_merged_weight(&weight, 4).is_err());
        assert!(quantize_merged_weight(&weight, 16).is_err());
    }

    #[test]
    fn test_quantize_compression_ratio() {
        let weight = Array2::<f64>::eye(4);
        let quantized = quantize_merged_weight(&weight, 8).expect("quantization failed");
        assert!((quantized.compression_ratio() - 0.125).abs() < 1e-10);
    }

    #[test]
    fn test_merge_multiple_lora() {
        let base = Array2::<f64>::eye(4);
        let a1 = Array2::from_shape_fn((2, 4), |(i, j)| (i + j) as f64 * 0.01);
        let b1 = Array2::from_shape_fn((4, 2), |(i, j)| (i + j) as f64 * 0.01);
        let a2 = Array2::from_shape_fn((2, 4), |(i, j)| (i * j) as f64 * 0.01);
        let b2 = Array2::from_shape_fn((4, 2), |(i, j)| (i * j) as f64 * 0.01);

        let params = vec![(a1.clone(), b1.clone(), 1.0), (a2.clone(), b2.clone(), 0.5)];
        let merged = merge_multiple_lora(&base, &params).expect("merge failed");

        // Verify step-by-step
        let step1 = merge_lora_weights(&base, &a1, &b1, 1.0).expect("merge1 failed");
        let step2 = merge_lora_weights(&step1, &a2, &b2, 0.5).expect("merge2 failed");

        for (a, b) in merged.iter().zip(step2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}
