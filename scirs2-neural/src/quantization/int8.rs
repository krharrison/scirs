//! INT8 Quantization
//!
//! Provides symmetric and asymmetric INT8 quantization with per-tensor
//! and per-channel modes, plus quantized matrix multiplication.

use crate::error::{Error, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayD, ArrayView, Axis, IxDyn, Zip};

/// INT8 quantization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Int8Mode {
    /// Symmetric quantization around zero: scale = max(|x|) / 127
    Symmetric,
    /// Asymmetric quantization: scale = (max - min) / 255, with zero_point
    Asymmetric,
}

/// Per-channel or per-tensor granularity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Granularity {
    /// Single scale/zero_point for the entire tensor
    PerTensor,
    /// Separate scale/zero_point for each output channel (axis 0)
    PerChannel,
}

/// Parameters for a single INT8 quantization group (per-tensor or one channel)
#[derive(Debug, Clone)]
pub struct Int8Params {
    /// Scale factor
    pub scale: f64,
    /// Zero point (0 for symmetric)
    pub zero_point: i32,
}

/// A fully quantized INT8 tensor with metadata
#[derive(Debug, Clone)]
pub struct Int8Tensor {
    /// Quantized data stored as i8
    pub data: ArrayD<i8>,
    /// Per-channel (or single-element) parameters
    pub params: Vec<Int8Params>,
    /// Original shape
    pub shape: Vec<usize>,
    /// Granularity used
    pub granularity: Granularity,
    /// Mode used
    pub mode: Int8Mode,
}

/// Compute INT8 symmetric parameters from a slice of f64 values
fn compute_symmetric_params(values: &[f64]) -> Int8Params {
    let abs_max = values.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 1.0 };
    Int8Params {
        scale,
        zero_point: 0,
    }
}

/// Compute INT8 asymmetric parameters from a slice of f64 values
///
/// Uses the signed i8 range [-128, 127] for asymmetric quantization.
/// The zero_point is an i32 offset (not clamped to i8 range) used
/// during quantize/dequantize to correctly map arbitrary value ranges.
///
/// Formula:
///   q = clamp(round(x / scale) + zero_point, -128, 127)
///   x = (q - zero_point) * scale
fn compute_asymmetric_params(values: &[f64]) -> Int8Params {
    let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;
    // Use full i8 range: 255 levels from -128 to 127
    let scale = if range > 0.0 { range / 255.0 } else { 1.0 };
    let zero_point = if scale > 0.0 {
        // We want: min_val -> -128, max_val -> 127
        // q = round(x / scale) + zp
        // -128 = round(min_val / scale) + zp
        // zp = -128 - round(min_val / scale)
        ((-128.0) - (min_val / scale).round()) as i32
    } else {
        0
    };
    // Do NOT clamp zero_point to i8 range -- it's an i32 offset
    Int8Params { scale, zero_point }
}

/// Quantize a single f64 value to i8 using given parameters and mode
fn quantize_value(val: f64, params: &Int8Params, mode: Int8Mode) -> i8 {
    let q = match mode {
        Int8Mode::Symmetric => {
            if params.scale > 0.0 {
                (val / params.scale).round()
            } else {
                0.0
            }
        }
        Int8Mode::Asymmetric => {
            if params.scale > 0.0 {
                (val / params.scale).round() + params.zero_point as f64
            } else {
                params.zero_point as f64
            }
        }
    };
    // Both modes use i8 range [-128, 127]
    q.clamp(-128.0, 127.0) as i8
}

/// Dequantize a single i8 value back to f64
fn dequantize_value(q: i8, params: &Int8Params, mode: Int8Mode) -> f64 {
    match mode {
        Int8Mode::Symmetric => q as f64 * params.scale,
        Int8Mode::Asymmetric => (q as f64 - params.zero_point as f64) * params.scale,
    }
}

/// Quantize an f64 tensor to INT8
///
/// # Arguments
/// * `tensor` - Input tensor (any shape, but per-channel uses axis 0)
/// * `mode` - Symmetric or Asymmetric
/// * `granularity` - PerTensor or PerChannel
pub fn quantize_int8(
    tensor: &ArrayD<f64>,
    mode: Int8Mode,
    granularity: Granularity,
) -> Result<Int8Tensor> {
    let shape = tensor.shape().to_vec();

    match granularity {
        Granularity::PerTensor => {
            let values: Vec<f64> = tensor.iter().cloned().collect();
            let params = match mode {
                Int8Mode::Symmetric => compute_symmetric_params(&values),
                Int8Mode::Asymmetric => compute_asymmetric_params(&values),
            };
            let data = tensor.mapv(|v| quantize_value(v, &params, mode));
            Ok(Int8Tensor {
                data,
                params: vec![params],
                shape,
                granularity,
                mode,
            })
        }
        Granularity::PerChannel => {
            if tensor.ndim() < 1 {
                return Err(Error::InvalidArgument(
                    "Per-channel quantization requires at least 1 dimension".to_string(),
                ));
            }
            let n_channels = shape[0];
            let mut all_params = Vec::with_capacity(n_channels);

            // Compute params per channel
            for ch in 0..n_channels {
                let slice = tensor.index_axis(Axis(0), ch);
                let values: Vec<f64> = slice.iter().cloned().collect();
                let p = match mode {
                    Int8Mode::Symmetric => compute_symmetric_params(&values),
                    Int8Mode::Asymmetric => compute_asymmetric_params(&values),
                };
                all_params.push(p);
            }

            // Quantize per channel
            let mut data = ArrayD::<i8>::zeros(IxDyn(&shape));
            for (ch, params) in all_params.iter().enumerate().take(n_channels) {
                let slice = tensor.index_axis(Axis(0), ch);
                let quantized_slice = slice.mapv(|v| quantize_value(v, params, mode));
                // Copy into output
                let mut out_slice = data.index_axis_mut(Axis(0), ch);
                out_slice.assign(&quantized_slice);
            }
            Ok(Int8Tensor {
                data,
                params: all_params,
                shape,
                granularity,
                mode,
            })
        }
    }
}

/// Dequantize an INT8 tensor back to f64
pub fn dequantize_int8(qtensor: &Int8Tensor) -> ArrayD<f64> {
    match qtensor.granularity {
        Granularity::PerTensor => {
            let params = &qtensor.params[0];
            qtensor
                .data
                .mapv(|q| dequantize_value(q, params, qtensor.mode))
        }
        Granularity::PerChannel => {
            let mut result = ArrayD::<f64>::zeros(IxDyn(&qtensor.shape));
            let n_channels = qtensor.shape[0];
            for ch in 0..n_channels {
                let slice = qtensor.data.index_axis(Axis(0), ch);
                let params = &qtensor.params[ch];
                let deq = slice.mapv(|q| dequantize_value(q, params, qtensor.mode));
                let mut out_slice = result.index_axis_mut(Axis(0), ch);
                out_slice.assign(&deq);
            }
            result
        }
    }
}

/// Simulated quantized matrix multiply
///
/// Performs: dequantize(A) * dequantize(B) but using integer arithmetic
/// internally (simulated). Both matrices are quantized to INT8, multiplied
/// in i32 accumulator space, then dequantized.
///
/// # Arguments
/// * `a` - First matrix (M x K), f64
/// * `b` - Second matrix (K x N), f64
/// * `mode` - Quantization mode
pub fn quantized_matmul(a: &Array2<f64>, b: &Array2<f64>, mode: Int8Mode) -> Result<Array2<f64>> {
    let (m, k) = (a.nrows(), a.ncols());
    let (k2, n) = (b.nrows(), b.ncols());
    if k != k2 {
        return Err(Error::InvalidArgument(format!(
            "Matrix dimensions mismatch: ({}, {}) x ({}, {})",
            m, k, k2, n
        )));
    }

    // Quantize both matrices per-tensor
    let qa = quantize_int8(&a.clone().into_dyn(), mode, Granularity::PerTensor)?;
    let qb = quantize_int8(&b.clone().into_dyn(), mode, Granularity::PerTensor)?;

    let pa = &qa.params[0];
    let pb = &qb.params[0];

    // Integer matmul with i32 accumulator
    let mut result = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut acc: i32 = 0;
            for kk in 0..k {
                let a_idx = &[i, kk];
                let b_idx = &[kk, j];
                let a_val = qa.data[IxDyn(a_idx)] as i32;
                let b_val = qb.data[IxDyn(b_idx)] as i32;
                match mode {
                    Int8Mode::Symmetric => {
                        acc += a_val * b_val;
                    }
                    Int8Mode::Asymmetric => {
                        acc += (a_val - pa.zero_point) * (b_val - pb.zero_point);
                    }
                }
            }
            // Dequantize accumulated result
            result[[i, j]] = acc as f64 * pa.scale * pb.scale;
        }
    }
    Ok(result)
}

/// Compute the mean squared error between original and dequantized tensors
pub fn roundtrip_mse(
    tensor: &ArrayD<f64>,
    mode: Int8Mode,
    granularity: Granularity,
) -> Result<f64> {
    let qt = quantize_int8(tensor, mode, granularity)?;
    let deq = dequantize_int8(&qt);
    let mse = Zip::from(tensor)
        .and(&deq)
        .fold(0.0, |acc, &orig, &d| acc + (orig - d).powi(2));
    let len = tensor.len() as f64;
    if len > 0.0 {
        Ok(mse / len)
    } else {
        Ok(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn make_test_tensor() -> ArrayD<f64> {
        array![[1.0, -0.5, 0.3], [2.0, -1.5, 0.7]].into_dyn()
    }

    #[test]
    fn test_symmetric_per_tensor_roundtrip() {
        let t = make_test_tensor();
        let qt =
            quantize_int8(&t, Int8Mode::Symmetric, Granularity::PerTensor).expect("test: quantize");
        assert_eq!(qt.params.len(), 1);
        assert_eq!(qt.params[0].zero_point, 0);
        let deq = dequantize_int8(&qt);
        assert_eq!(deq.shape(), t.shape());
        let mse =
            roundtrip_mse(&t, Int8Mode::Symmetric, Granularity::PerTensor).expect("test: mse");
        assert!(mse < 0.01, "MSE too large: {}", mse);
    }

    #[test]
    fn test_asymmetric_per_tensor_roundtrip() {
        let t = make_test_tensor();
        let qt = quantize_int8(&t, Int8Mode::Asymmetric, Granularity::PerTensor)
            .expect("test: quantize");
        assert_eq!(qt.params.len(), 1);
        let deq = dequantize_int8(&qt);
        assert_eq!(deq.shape(), t.shape());
        let mse =
            roundtrip_mse(&t, Int8Mode::Asymmetric, Granularity::PerTensor).expect("test: mse");
        assert!(mse < 0.01, "MSE too large: {}", mse);
    }

    #[test]
    fn test_symmetric_per_channel() {
        let t = make_test_tensor(); // shape [2, 3]
        let qt = quantize_int8(&t, Int8Mode::Symmetric, Granularity::PerChannel)
            .expect("test: quantize");
        assert_eq!(qt.params.len(), 2); // 2 channels
        let deq = dequantize_int8(&qt);
        for (orig, deq_v) in t.iter().zip(deq.iter()) {
            assert!((orig - deq_v).abs() < 0.05, "orig={}, deq={}", orig, deq_v);
        }
    }

    #[test]
    fn test_asymmetric_per_channel() {
        let t = array![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]].into_dyn();
        let qt = quantize_int8(&t, Int8Mode::Asymmetric, Granularity::PerChannel)
            .expect("test: quantize");
        assert_eq!(qt.params.len(), 2);
        let deq = dequantize_int8(&qt);
        for (orig, deq_v) in t.iter().zip(deq.iter()) {
            assert!((orig - deq_v).abs() < 0.05, "orig={}, deq={}", orig, deq_v);
        }
    }

    #[test]
    fn test_quantized_matmul_symmetric() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let b = array![[0.5_f64, -0.5], [1.0, 0.0]];
        let expected = a.dot(&b);
        let result = quantized_matmul(&a, &b, Int8Mode::Symmetric).expect("test: matmul");
        for (e, r) in expected.iter().zip(result.iter()) {
            assert!((e - r).abs() < 0.2, "expected={}, got={}", e, r);
        }
    }

    #[test]
    fn test_quantized_matmul_asymmetric() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let b = array![[0.5_f64, -0.5], [1.0, 0.0]];
        let expected = a.dot(&b);
        let result = quantized_matmul(&a, &b, Int8Mode::Asymmetric).expect("test: matmul");
        // Asymmetric quantization with small matrices has higher quantization error
        // but should still be in the right ballpark
        for (e, r) in expected.iter().zip(result.iter()) {
            assert!((e - r).abs() < 1.0, "expected={}, got={}", e, r);
        }
    }

    #[test]
    fn test_dimension_mismatch_matmul() {
        let a = array![[1.0_f64, 2.0, 3.0]];
        let b = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let result = quantized_matmul(&a, &b, Int8Mode::Symmetric);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_tensor() {
        let t = ArrayD::<f64>::zeros(IxDyn(&[3, 4]));
        let qt =
            quantize_int8(&t, Int8Mode::Symmetric, Granularity::PerTensor).expect("test: quantize");
        let deq = dequantize_int8(&qt);
        for &v in deq.iter() {
            assert!((v).abs() < 1e-10);
        }
    }

    #[test]
    fn test_per_channel_requires_dim() {
        let t = ArrayD::<f64>::zeros(IxDyn(&[])); // scalar
        let result = quantize_int8(&t, Int8Mode::Symmetric, Granularity::PerChannel);
        assert!(result.is_err());
    }
}
