//! INT4 Group Quantization
//!
//! Provides INT4 quantization with configurable group sizes (32, 64, 128).
//! Two INT4 values are packed per byte for memory efficiency.
//! Each group has its own scale and zero_point.

use crate::error::{Error, Result};
use scirs2_core::ndarray::{ArrayD, IxDyn, Zip};

/// Valid group sizes for INT4 quantization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupSize {
    /// 32 elements per group
    G32,
    /// 64 elements per group
    G64,
    /// 128 elements per group
    G128,
}

impl GroupSize {
    /// Get the numeric group size
    pub fn value(self) -> usize {
        match self {
            GroupSize::G32 => 32,
            GroupSize::G64 => 64,
            GroupSize::G128 => 128,
        }
    }
}

/// Parameters for one INT4 group
#[derive(Debug, Clone)]
pub struct Int4GroupParams {
    /// Scale factor for this group
    pub scale: f64,
    /// Zero point for this group (range 0..15)
    pub zero_point: u8,
}

/// Packed INT4 tensor
///
/// Two 4-bit values are packed per byte:
/// - Low nibble: even-indexed element
/// - High nibble: odd-indexed element
///
/// INT4 values are unsigned (0..15), and the zero_point handles offset.
#[derive(Debug, Clone)]
pub struct Int4Tensor {
    /// Packed data: 2 int4 values per byte
    pub packed_data: Vec<u8>,
    /// Group parameters (scale + zero_point per group)
    pub group_params: Vec<Int4GroupParams>,
    /// Group size used
    pub group_size: GroupSize,
    /// Original shape
    pub shape: Vec<usize>,
    /// Total number of elements
    pub n_elements: usize,
}

/// Pack two u4 values into a single byte
///
/// low goes into bits 0..3, high goes into bits 4..7
fn pack_nibbles(low: u8, high: u8) -> u8 {
    (low & 0x0F) | ((high & 0x0F) << 4)
}

/// Unpack a byte into two u4 values (low, high)
fn unpack_nibbles(byte: u8) -> (u8, u8) {
    (byte & 0x0F, (byte >> 4) & 0x0F)
}

/// Quantize a flat slice of f64 values to INT4 for one group
fn quantize_group(values: &[f64]) -> (Vec<u8>, Int4GroupParams) {
    let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let range = max_val - min_val;
    let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
    let zero_point = if scale > 0.0 {
        ((-min_val / scale).round() as i32).clamp(0, 15) as u8
    } else {
        0
    };

    let params = Int4GroupParams { scale, zero_point };

    // Quantize each value to 0..15
    let quantized: Vec<u8> = values
        .iter()
        .map(|&v| {
            let q = if scale > 0.0 {
                (v / scale + zero_point as f64).round() as i32
            } else {
                zero_point as i32
            };
            q.clamp(0, 15) as u8
        })
        .collect();

    (quantized, params)
}

/// Dequantize INT4 values for one group
fn dequantize_group(quantized: &[u8], params: &Int4GroupParams) -> Vec<f64> {
    quantized
        .iter()
        .map(|&q| (q as f64 - params.zero_point as f64) * params.scale)
        .collect()
}

/// Quantize an f64 tensor to INT4 with group quantization
///
/// The tensor is flattened, then divided into groups. Each group gets
/// its own scale and zero_point. Values are packed 2-per-byte.
///
/// # Arguments
/// * `tensor` - Input tensor of any shape
/// * `group_size` - Group size (32, 64, or 128)
pub fn quantize_int4(tensor: &ArrayD<f64>, group_size: GroupSize) -> Result<Int4Tensor> {
    let shape = tensor.shape().to_vec();
    let n_elements = tensor.len();
    let gs = group_size.value();

    if n_elements == 0 {
        return Ok(Int4Tensor {
            packed_data: Vec::new(),
            group_params: Vec::new(),
            group_size,
            shape,
            n_elements: 0,
        });
    }

    // Flatten
    let flat: Vec<f64> = tensor.iter().cloned().collect();

    // Number of groups (last group may be partial)
    let n_groups = n_elements.div_ceil(gs);
    let mut all_quantized: Vec<u8> = Vec::with_capacity(n_elements);
    let mut group_params: Vec<Int4GroupParams> = Vec::with_capacity(n_groups);

    for g in 0..n_groups {
        let start = g * gs;
        let end = (start + gs).min(n_elements);
        let group_values = &flat[start..end];
        let (quantized, params) = quantize_group(group_values);
        all_quantized.extend_from_slice(&quantized);
        group_params.push(params);
    }

    // Pack nibbles: 2 values per byte
    let packed_len = n_elements.div_ceil(2);
    let mut packed_data = Vec::with_capacity(packed_len);
    let mut i = 0;
    while i < all_quantized.len() {
        let low = all_quantized[i];
        let high = if i + 1 < all_quantized.len() {
            all_quantized[i + 1]
        } else {
            0 // padding for odd count
        };
        packed_data.push(pack_nibbles(low, high));
        i += 2;
    }

    Ok(Int4Tensor {
        packed_data,
        group_params,
        group_size,
        shape,
        n_elements,
    })
}

/// Dequantize an INT4 tensor back to f64
pub fn dequantize_int4(qtensor: &Int4Tensor) -> Result<ArrayD<f64>> {
    if qtensor.n_elements == 0 {
        return Ok(ArrayD::<f64>::zeros(IxDyn(&qtensor.shape)));
    }

    // Unpack nibbles
    let mut unpacked: Vec<u8> = Vec::with_capacity(qtensor.n_elements);
    for &byte in &qtensor.packed_data {
        let (low, high) = unpack_nibbles(byte);
        unpacked.push(low);
        unpacked.push(high);
    }
    // Trim to actual count
    unpacked.truncate(qtensor.n_elements);

    // Dequantize per group
    let gs = qtensor.group_size.value();
    let mut result: Vec<f64> = Vec::with_capacity(qtensor.n_elements);
    for (g, params) in qtensor.group_params.iter().enumerate() {
        let start = g * gs;
        let end = (start + gs).min(qtensor.n_elements);
        let group_quantized = &unpacked[start..end];
        let deq = dequantize_group(group_quantized, params);
        result.extend_from_slice(&deq);
    }

    let arr = ArrayD::from_shape_vec(IxDyn(&qtensor.shape), result).map_err(|e| {
        Error::InvalidArgument(format!("Failed to reshape dequantized tensor: {}", e))
    })?;
    Ok(arr)
}

/// Compute the memory size of an INT4 tensor in bytes
/// (packed data + group params overhead)
pub fn int4_memory_bytes(qtensor: &Int4Tensor) -> usize {
    let packed = qtensor.packed_data.len();
    // Each group param: 8 bytes (f64 scale) + 1 byte (zero_point)
    let params_overhead = qtensor.group_params.len() * (8 + 1);
    packed + params_overhead
}

/// Compute compression ratio vs f64 storage
pub fn int4_compression_ratio(qtensor: &Int4Tensor) -> f64 {
    let original_bytes = qtensor.n_elements * 8; // f64 = 8 bytes
    let quantized_bytes = int4_memory_bytes(qtensor);
    if quantized_bytes > 0 {
        original_bytes as f64 / quantized_bytes as f64
    } else {
        1.0
    }
}

/// Compute roundtrip MSE for INT4 quantization
pub fn int4_roundtrip_mse(tensor: &ArrayD<f64>, group_size: GroupSize) -> Result<f64> {
    let qt = quantize_int4(tensor, group_size)?;
    let deq = dequantize_int4(&qt)?;
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

    #[test]
    fn test_pack_unpack_nibbles() {
        for low in 0..16u8 {
            for high in 0..16u8 {
                let packed = pack_nibbles(low, high);
                let (l, h) = unpack_nibbles(packed);
                assert_eq!(l, low, "low mismatch for ({}, {})", low, high);
                assert_eq!(h, high, "high mismatch for ({}, {})", low, high);
            }
        }
    }

    #[test]
    fn test_int4_roundtrip_g32() {
        // Create a tensor with 64 elements (2 full groups of 32)
        let data: Vec<f64> = (0..64).map(|i| (i as f64 - 32.0) / 16.0).collect();
        let tensor = ArrayD::from_shape_vec(IxDyn(&[8, 8]), data).expect("test: create tensor");
        let qt = quantize_int4(&tensor, GroupSize::G32).expect("test: quantize");
        assert_eq!(qt.group_params.len(), 2); // 64 / 32 = 2 groups
        assert_eq!(qt.packed_data.len(), 32); // 64 / 2 = 32 packed bytes
        let deq = dequantize_int4(&qt).expect("test: dequantize");
        assert_eq!(deq.shape(), tensor.shape());
        let mse = int4_roundtrip_mse(&tensor, GroupSize::G32).expect("test: mse");
        // INT4 has lower precision, so allow larger error
        assert!(mse < 0.05, "MSE too large: {}", mse);
    }

    #[test]
    fn test_int4_roundtrip_g64() {
        let data: Vec<f64> = (0..128).map(|i| (i as f64 - 64.0) / 32.0).collect();
        let tensor = ArrayD::from_shape_vec(IxDyn(&[128]), data).expect("test: create tensor");
        let qt = quantize_int4(&tensor, GroupSize::G64).expect("test: quantize");
        assert_eq!(qt.group_params.len(), 2);
        let deq = dequantize_int4(&qt).expect("test: dequantize");
        assert_eq!(deq.shape(), tensor.shape());
    }

    #[test]
    fn test_int4_roundtrip_g128() {
        let data: Vec<f64> = (0..256).map(|i| (i as f64 - 128.0) / 64.0).collect();
        let tensor = ArrayD::from_shape_vec(IxDyn(&[16, 16]), data).expect("test: create tensor");
        let qt = quantize_int4(&tensor, GroupSize::G128).expect("test: quantize");
        assert_eq!(qt.group_params.len(), 2);
        let mse = int4_roundtrip_mse(&tensor, GroupSize::G128).expect("test: mse");
        assert!(mse < 0.05, "MSE too large: {}", mse);
    }

    #[test]
    fn test_int4_partial_group() {
        // 50 elements: 1 full group of 32 + 1 partial group of 18
        let data: Vec<f64> = (0..50).map(|i| i as f64 / 25.0 - 1.0).collect();
        let tensor = ArrayD::from_shape_vec(IxDyn(&[50]), data).expect("test: create tensor");
        let qt = quantize_int4(&tensor, GroupSize::G32).expect("test: quantize");
        assert_eq!(qt.group_params.len(), 2); // ceil(50/32) = 2
        let deq = dequantize_int4(&qt).expect("test: dequantize");
        assert_eq!(deq.len(), 50);
    }

    #[test]
    fn test_int4_compression() {
        let data: Vec<f64> = (0..1024).map(|i| i as f64 / 512.0).collect();
        let tensor = ArrayD::from_shape_vec(IxDyn(&[1024]), data).expect("test: create tensor");
        let qt = quantize_int4(&tensor, GroupSize::G128).expect("test: quantize");
        let ratio = int4_compression_ratio(&qt);
        // With INT4, expect roughly 16x compression from f64
        // minus group param overhead
        assert!(ratio > 10.0, "Compression ratio too low: {}", ratio);
    }

    #[test]
    fn test_int4_empty_tensor() {
        let tensor = ArrayD::<f64>::zeros(IxDyn(&[0]));
        let qt = quantize_int4(&tensor, GroupSize::G32).expect("test: quantize");
        assert!(qt.packed_data.is_empty());
        let deq = dequantize_int4(&qt).expect("test: dequantize");
        assert_eq!(deq.len(), 0);
    }

    #[test]
    fn test_int4_constant_tensor() {
        // A constant tensor has range=0, so scale=1.0 and all values map to zero_point.
        // The dequantized result should be very close to the original constant.
        let tensor = ArrayD::from_elem(IxDyn(&[64]), 3.15_f64);
        let qt = quantize_int4(&tensor, GroupSize::G32).expect("test: quantize");
        let deq = dequantize_int4(&qt).expect("test: dequantize");
        // With range=0, scale=1.0 and zero_point=clamp(round(-3.15), 0, 15)=0
        // so all dequantized values = (zp - zp) * scale = 0.
        // Actually for a constant tensor, min=max=3.15, range=0, scale=1.0,
        // zp = clamp(round(-3.15/1.0), 0, 15) = 0
        // q = clamp(round(3.15/1.0 + 0), 0, 15) = clamp(3, 0, 15) = 3
        // deq = (3 - 0) * 1.0 = 3.0
        // So the error is 0.15 per element
        for &v in deq.iter() {
            assert!((v - 3.15).abs() < 0.5, "expected ~3.15, got {}", v);
        }
    }

    #[test]
    fn test_int4_small_tensor() {
        let tensor = array![1.0_f64, -1.0, 0.5, -0.5].into_dyn();
        let qt = quantize_int4(&tensor, GroupSize::G32).expect("test: quantize");
        assert_eq!(qt.group_params.len(), 1); // 4 < 32, so 1 partial group
        let deq = dequantize_int4(&qt).expect("test: dequantize");
        let mse = int4_roundtrip_mse(&tensor, GroupSize::G32).expect("test: mse");
        assert!(mse < 0.1, "MSE too large: {}", mse);
    }
}
