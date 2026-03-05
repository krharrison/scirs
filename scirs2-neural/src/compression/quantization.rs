//! Post-training weight quantization.
//!
//! Supports symmetric, asymmetric, and per-channel quantization of weight
//! matrices to reduced-bit integer representations.  Provides calibration
//! helpers and a round-trip `quantize` / `dequantize` API.

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array1, Array2, Axis};

// ─────────────────────────────────────────────────────────────────────────────
// Public enums / types
// ─────────────────────────────────────────────────────────────────────────────

/// How the quantization range is mapped to integers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizationScheme {
    /// Zero-point is always 0; range is symmetric around zero.
    Symmetric,
    /// Zero-point can be non-zero; full dynamic range is used.
    Asymmetric,
    /// Each output channel has its own scale/zero-point.
    PerChannel,
}

/// A single quantized layer: int8 weights plus per-tensor (or per-channel)
/// calibration parameters.
#[derive(Debug, Clone)]
pub struct QuantizedLayer {
    /// Quantized integer weights.
    pub quantized: Array2<i8>,
    /// Per-tensor (Symmetric/Asymmetric) or per-channel (PerChannel) scale.
    pub scale: Vec<f32>,
    /// Per-tensor or per-channel zero-point.
    pub zero_point: Vec<i32>,
    /// Number of bits used (stored for information; currently only int8).
    pub bits: u8,
    /// Scheme used during quantization.
    pub scheme: QuantizationScheme,
    /// Original matrix shape.
    pub shape: (usize, usize),
}

// ─────────────────────────────────────────────────────────────────────────────
// Core functions
// ─────────────────────────────────────────────────────────────────────────────

/// Collect the calibration range `(min, max)` from a slice of activation tensors.
///
/// Uses the global min and max across all provided tensors (the MinMax calibration
/// strategy), which is the most numerically stable choice for post-training quantization.
///
/// # Errors
/// Returns an error if the slice is empty.
pub fn calibrate_range(activations: &[Array2<f32>]) -> Result<(f32, f32)> {
    if activations.is_empty() {
        return Err(NeuralError::InvalidArchitecture(
            "calibrate_range: activation slice must not be empty".into(),
        ));
    }
    let mut global_min = f32::INFINITY;
    let mut global_max = f32::NEG_INFINITY;
    for act in activations {
        for &v in act.iter() {
            if v < global_min {
                global_min = v;
            }
            if v > global_max {
                global_max = v;
            }
        }
    }
    if !global_min.is_finite() || !global_max.is_finite() {
        return Err(NeuralError::InvalidArchitecture(
            "calibrate_range: activations contain non-finite values".into(),
        ));
    }
    Ok((global_min, global_max))
}

/// Quantize a weight matrix to int8 using the specified bit-width and scheme.
///
/// For `PerChannel`, each row (output channel) gets its own scale and zero-point.
/// For `Symmetric` and `Asymmetric`, a single tensor-wide scale/zero-point is used.
///
/// # Errors
/// Returns an error for unsupported bit-widths (< 1 or > 8) or degenerate inputs.
pub fn quantize_weights(
    weights: &Array2<f32>,
    bits: u8,
    scheme: QuantizationScheme,
) -> Result<QuantizedLayer> {
    if bits == 0 || bits > 8 {
        return Err(NeuralError::InvalidArchitecture(format!(
            "quantize_weights: bits must be in [1, 8], got {bits}"
        )));
    }
    let (nrows, ncols) = (weights.nrows(), weights.ncols());

    match scheme {
        QuantizationScheme::Symmetric => {
            let (scale, zp) = symmetric_params(weights, bits)?;
            let q = quantize_tensor_symmetric(weights, scale, bits)?;
            Ok(QuantizedLayer {
                quantized: q,
                scale: vec![scale],
                zero_point: vec![zp],
                bits,
                scheme,
                shape: (nrows, ncols),
            })
        }
        QuantizationScheme::Asymmetric => {
            let (w_min, w_max) = tensor_min_max(weights)?;
            let (scale, zp) = asymmetric_params(w_min, w_max, bits);
            let q = quantize_tensor_asymmetric(weights, scale, zp, bits)?;
            Ok(QuantizedLayer {
                quantized: q,
                scale: vec![scale],
                zero_point: vec![zp],
                bits,
                scheme,
                shape: (nrows, ncols),
            })
        }
        QuantizationScheme::PerChannel => {
            // One scale/zero-point per output channel (row).
            let mut scales = Vec::with_capacity(nrows);
            let mut zps = Vec::with_capacity(nrows);
            let mut q_data = Vec::with_capacity(nrows * ncols);

            for row in weights.rows() {
                let row_arr = row.to_owned();
                let (s, zp) = {
                    let (rmin, rmax) = slice_min_max(row_arr.as_slice().ok_or_else(|| {
                        NeuralError::InvalidArchitecture(
                            "per-channel row is not contiguous".into(),
                        )
                    })?)?;
                    asymmetric_params(rmin, rmax, bits)
                };
                scales.push(s);
                zps.push(zp);
                let (qmin, qmax) = int_range(bits, false);
                for &v in row_arr.iter() {
                    let q = (v / s).round() as i32 + zp;
                    q_data.push(q.clamp(qmin, qmax) as i8);
                }
            }

            let q = Array2::from_shape_vec((nrows, ncols), q_data).map_err(|e| {
                NeuralError::InvalidArchitecture(format!("per-channel quantization reshape: {e}"))
            })?;
            Ok(QuantizedLayer {
                quantized: q,
                scale: scales,
                zero_point: zps,
                bits,
                scheme,
                shape: (nrows, ncols),
            })
        }
    }
}

/// Reconstruct an approximate float weight matrix from a `QuantizedLayer`.
///
/// The dequantized value is `scale * (q - zero_point)`.
///
/// # Errors
/// Returns an error if the stored shape does not match `quantized.shape()`.
pub fn dequantize(layer: &QuantizedLayer) -> Result<Array2<f32>> {
    let (nrows, ncols) = layer.shape;
    if layer.quantized.shape() != [nrows, ncols] {
        return Err(NeuralError::InvalidArchitecture(format!(
            "dequantize: stored shape ({nrows},{ncols}) does not match quantized shape {:?}",
            layer.quantized.shape()
        )));
    }
    let mut out = Array2::zeros((nrows, ncols));
    match layer.scheme {
        QuantizationScheme::Symmetric | QuantizationScheme::Asymmetric => {
            let scale = layer.scale[0];
            let zp = layer.zero_point[0];
            for ((r, c), &q) in layer.quantized.indexed_iter() {
                out[(r, c)] = scale * (q as i32 - zp) as f32;
            }
        }
        QuantizationScheme::PerChannel => {
            for (r, row) in layer.quantized.rows().into_iter().enumerate() {
                let scale = layer.scale[r];
                let zp = layer.zero_point[r];
                for (c, &q) in row.iter().enumerate() {
                    out[(r, c)] = scale * (q as i32 - zp) as f32;
                }
            }
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Integer range `[qmin, qmax]` for a given bit-width.
/// `signed = true` → int8 range; `signed = false` → uint-style (zero-centered).
fn int_range(bits: u8, signed: bool) -> (i32, i32) {
    let n = 1i32 << bits;
    if signed {
        (-(n / 2), n / 2 - 1)
    } else {
        // For asymmetric we use [0, 2^bits - 1] shifted to [-(2^(bits-1)), +(2^(bits-1)-1)]
        // but represent with a zero-point offset; here we return the unsigned span.
        (0, n - 1)
    }
}

fn tensor_min_max(weights: &Array2<f32>) -> Result<(f32, f32)> {
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &v in weights.iter() {
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    if !mn.is_finite() || !mx.is_finite() {
        return Err(NeuralError::InvalidArchitecture(
            "weight matrix contains non-finite values".into(),
        ));
    }
    Ok((mn, mx))
}

fn slice_min_max(s: &[f32]) -> Result<(f32, f32)> {
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &v in s {
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    if !mn.is_finite() || !mx.is_finite() {
        return Err(NeuralError::InvalidArchitecture(
            "slice contains non-finite values".into(),
        ));
    }
    Ok((mn, mx))
}

/// Symmetric quantization parameters: `scale = max_abs / (2^(bits-1) - 1)`, `zero_point = 0`.
fn symmetric_params(weights: &Array2<f32>, bits: u8) -> Result<(f32, i32)> {
    let max_abs = weights
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max);
    if max_abs == 0.0 {
        return Ok((1.0, 0));
    }
    let qmax = ((1i32 << (bits - 1)) - 1) as f32;
    let scale = max_abs / qmax;
    Ok((scale, 0))
}

/// Asymmetric (unsigned) quantization parameters.
fn asymmetric_params(w_min: f32, w_max: f32, bits: u8) -> (f32, i32) {
    let qmin = 0_i32;
    let qmax = (1i32 << bits) - 1;
    let range = w_max - w_min;
    let scale = if range == 0.0 { 1.0 } else { range / (qmax - qmin) as f32 };
    let zp = (qmin as f32 - w_min / scale).round() as i32;
    let zp = zp.clamp(qmin, qmax);
    (scale, zp)
}

fn quantize_tensor_symmetric(
    weights: &Array2<f32>,
    scale: f32,
    bits: u8,
) -> Result<Array2<i8>> {
    let qmax = (1i32 << (bits - 1)) - 1;
    let qmin = -qmax - 1;
    let (nrows, ncols) = (weights.nrows(), weights.ncols());
    let flat: Vec<i8> = weights
        .iter()
        .map(|&v| {
            let q = (v / scale).round() as i32;
            q.clamp(qmin, qmax) as i8
        })
        .collect();
    Array2::from_shape_vec((nrows, ncols), flat)
        .map_err(|e| NeuralError::InvalidArchitecture(format!("symmetric quant reshape: {e}")))
}

fn quantize_tensor_asymmetric(
    weights: &Array2<f32>,
    scale: f32,
    zp: i32,
    bits: u8,
) -> Result<Array2<i8>> {
    let qmin = 0_i32;
    let qmax = (1i32 << bits) - 1;
    let (nrows, ncols) = (weights.nrows(), weights.ncols());
    let flat: Vec<i8> = weights
        .iter()
        .map(|&v| {
            let q = (v / scale).round() as i32 + zp;
            // Map to signed i8 domain centred at 0: shift by -(2^(bits-1)).
            let q_shifted = q - (1i32 << (bits - 1));
            q.clamp(qmin, qmax) as i8
        })
        .collect();
    Array2::from_shape_vec((nrows, ncols), flat)
        .map_err(|e| NeuralError::InvalidArchitecture(format!("asymmetric quant reshape: {e}")))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_weights(nrows: usize, ncols: usize, lo: f32, hi: f32) -> Array2<f32> {
        let step = (hi - lo) / ((nrows * ncols) as f32 - 1.0);
        let flat: Vec<f32> = (0..nrows * ncols).map(|i| lo + i as f32 * step).collect();
        Array2::from_shape_vec((nrows, ncols), flat).expect("shape error")
    }

    #[test]
    fn test_calibrate_range_basic() {
        let acts = vec![
            Array2::from_shape_vec((2, 2), vec![-1.0, 0.5, 0.0, 2.0]).expect("shape"),
            Array2::from_shape_vec((2, 2), vec![0.3, -3.0, 1.2, 0.0]).expect("shape"),
        ];
        let (mn, mx) = calibrate_range(&acts).expect("calibrate_range failed");
        assert!((mn - (-3.0)).abs() < 1e-6);
        assert!((mx - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_calibrate_range_empty() {
        let result = calibrate_range(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantize_dequantize_symmetric() {
        let w = uniform_weights(4, 4, -1.0, 1.0);
        let ql = quantize_weights(&w, 8, QuantizationScheme::Symmetric).expect("quant failed");
        let deq = dequantize(&ql).expect("dequant failed");
        // Max quantization error for 8-bit symmetric should be < 1/127 ≈ 0.008.
        for (o, d) in w.iter().zip(deq.iter()) {
            assert!((o - d).abs() < 0.02, "error too large: orig={o}, dequant={d}");
        }
    }

    #[test]
    fn test_quantize_dequantize_asymmetric() {
        let w = uniform_weights(4, 4, 0.0, 1.0);
        let ql = quantize_weights(&w, 8, QuantizationScheme::Asymmetric).expect("quant failed");
        let deq = dequantize(&ql).expect("dequant failed");
        for (o, d) in w.iter().zip(deq.iter()) {
            assert!((o - d).abs() < 0.02, "error too large: orig={o}, dequant={d}");
        }
    }

    #[test]
    fn test_quantize_dequantize_per_channel() {
        let w = uniform_weights(8, 4, -2.0, 2.0);
        let ql = quantize_weights(&w, 8, QuantizationScheme::PerChannel).expect("quant failed");
        assert_eq!(ql.scale.len(), 8);
        assert_eq!(ql.zero_point.len(), 8);
        let deq = dequantize(&ql).expect("dequant failed");
        for (o, d) in w.iter().zip(deq.iter()) {
            assert!((o - d).abs() < 0.05, "error too large: orig={o}, dequant={d}");
        }
    }

    #[test]
    fn test_quantize_invalid_bits() {
        let w = uniform_weights(2, 2, -1.0, 1.0);
        assert!(quantize_weights(&w, 0, QuantizationScheme::Symmetric).is_err());
        assert!(quantize_weights(&w, 9, QuantizationScheme::Symmetric).is_err());
    }

    #[test]
    fn test_quantize_all_zeros() {
        let w = Array2::zeros((3, 3));
        let ql = quantize_weights(&w, 8, QuantizationScheme::Symmetric).expect("quant failed");
        let deq = dequantize(&ql).expect("dequant failed");
        assert!(deq.iter().all(|&v| v == 0.0));
    }
}
