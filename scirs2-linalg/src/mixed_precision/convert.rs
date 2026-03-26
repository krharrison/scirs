//! Batch conversion functions for F16/BF16 arrays
//!
//! Provides efficient element-wise conversion between f32/f64 and the
//! half-precision types [`F16`] and [`BF16`], for both 1-D vectors
//! ([`Array1`]) and 2-D matrices ([`Array2`]).

use scirs2_core::ndarray::{Array1, Array2};

use super::types::{BF16, F16};

// ============================================================================
// F16 <-> f32 conversions
// ============================================================================

/// Convert an `Array1<f32>` to `Array1<F16>`.
pub fn f32_to_f16_array(input: &Array1<f32>) -> Array1<F16> {
    Array1::from_iter(input.iter().map(|&v| F16::from_f32(v)))
}

/// Convert an `Array1<F16>` to `Array1<f32>`.
pub fn f16_to_f32_array(input: &Array1<F16>) -> Array1<f32> {
    Array1::from_iter(input.iter().map(|v| v.to_f32()))
}

/// Convert an `Array2<f32>` to `Array2<F16>`.
pub fn f32_to_f16_matrix(input: &Array2<f32>) -> Array2<F16> {
    let (rows, cols) = (input.nrows(), input.ncols());
    let data: Vec<F16> = input.iter().map(|&v| F16::from_f32(v)).collect();
    Array2::from_shape_vec((rows, cols), data).unwrap_or_else(|_| Array2::default((rows, cols)))
}

/// Convert an `Array2<F16>` to `Array2<f32>`.
pub fn f16_to_f32_matrix(input: &Array2<F16>) -> Array2<f32> {
    let (rows, cols) = (input.nrows(), input.ncols());
    let data: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
    Array2::from_shape_vec((rows, cols), data).unwrap_or_else(|_| Array2::zeros((rows, cols)))
}

// ============================================================================
// BF16 <-> f32 conversions
// ============================================================================

/// Convert an `Array1<f32>` to `Array1<BF16>`.
pub fn f32_to_bf16_array(input: &Array1<f32>) -> Array1<BF16> {
    Array1::from_iter(input.iter().map(|&v| BF16::from_f32(v)))
}

/// Convert an `Array1<BF16>` to `Array1<f32>`.
pub fn bf16_to_f32_array(input: &Array1<BF16>) -> Array1<f32> {
    Array1::from_iter(input.iter().map(|v| v.to_f32()))
}

/// Convert an `Array2<f32>` to `Array2<BF16>`.
pub fn f32_to_bf16_matrix(input: &Array2<f32>) -> Array2<BF16> {
    let (rows, cols) = (input.nrows(), input.ncols());
    let data: Vec<BF16> = input.iter().map(|&v| BF16::from_f32(v)).collect();
    Array2::from_shape_vec((rows, cols), data).unwrap_or_else(|_| Array2::default((rows, cols)))
}

/// Convert an `Array2<BF16>` to `Array2<f32>`.
pub fn bf16_to_f32_matrix(input: &Array2<BF16>) -> Array2<f32> {
    let (rows, cols) = (input.nrows(), input.ncols());
    let data: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
    Array2::from_shape_vec((rows, cols), data).unwrap_or_else(|_| Array2::zeros((rows, cols)))
}

// ============================================================================
// F16 <-> f64 conversions
// ============================================================================

/// Convert an `Array1<f64>` to `Array1<F16>`.
pub fn f64_to_f16_array(input: &Array1<f64>) -> Array1<F16> {
    Array1::from_iter(input.iter().map(|&v| F16::from_f64(v)))
}

/// Convert an `Array1<F16>` to `Array1<f64>`.
pub fn f16_to_f64_array(input: &Array1<F16>) -> Array1<f64> {
    Array1::from_iter(input.iter().map(|v| v.to_f64()))
}

/// Convert an `Array2<f64>` to `Array2<F16>`.
pub fn f64_to_f16_matrix(input: &Array2<f64>) -> Array2<F16> {
    let (rows, cols) = (input.nrows(), input.ncols());
    let data: Vec<F16> = input.iter().map(|&v| F16::from_f64(v)).collect();
    Array2::from_shape_vec((rows, cols), data).unwrap_or_else(|_| Array2::default((rows, cols)))
}

/// Convert an `Array2<F16>` to `Array2<f64>`.
pub fn f16_to_f64_matrix(input: &Array2<F16>) -> Array2<f64> {
    let (rows, cols) = (input.nrows(), input.ncols());
    let data: Vec<f64> = input.iter().map(|v| v.to_f64()).collect();
    Array2::from_shape_vec((rows, cols), data).unwrap_or_else(|_| Array2::zeros((rows, cols)))
}

// ============================================================================
// BF16 <-> f64 conversions
// ============================================================================

/// Convert an `Array1<f64>` to `Array1<BF16>`.
pub fn f64_to_bf16_array(input: &Array1<f64>) -> Array1<BF16> {
    Array1::from_iter(input.iter().map(|&v| BF16::from_f64(v)))
}

/// Convert an `Array1<BF16>` to `Array1<f64>`.
pub fn bf16_to_f64_array(input: &Array1<BF16>) -> Array1<f64> {
    Array1::from_iter(input.iter().map(|v| v.to_f64()))
}

/// Convert an `Array2<f64>` to `Array2<BF16>`.
pub fn f64_to_bf16_matrix(input: &Array2<f64>) -> Array2<BF16> {
    let (rows, cols) = (input.nrows(), input.ncols());
    let data: Vec<BF16> = input.iter().map(|&v| BF16::from_f64(v)).collect();
    Array2::from_shape_vec((rows, cols), data).unwrap_or_else(|_| Array2::default((rows, cols)))
}

/// Convert an `Array2<BF16>` to `Array2<f64>`.
pub fn bf16_to_f64_matrix(input: &Array2<BF16>) -> Array2<f64> {
    let (rows, cols) = (input.nrows(), input.ncols());
    let data: Vec<f64> = input.iter().map(|v| v.to_f64()).collect();
    Array2::from_shape_vec((rows, cols), data).unwrap_or_else(|_| Array2::zeros((rows, cols)))
}

// ============================================================================
// F16 <-> BF16 conversions
// ============================================================================

/// Convert an `Array1<F16>` to `Array1<BF16>` (via f32 intermediate).
pub fn f16_to_bf16_array(input: &Array1<F16>) -> Array1<BF16> {
    Array1::from_iter(input.iter().map(|v| BF16::from_f32(v.to_f32())))
}

/// Convert an `Array1<BF16>` to `Array1<F16>` (via f32 intermediate).
pub fn bf16_to_f16_array(input: &Array1<BF16>) -> Array1<F16> {
    Array1::from_iter(input.iter().map(|v| F16::from_f32(v.to_f32())))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_f32_f16_roundtrip_array() {
        let input = array![1.0f32, 2.0, -3.0, 0.5, 0.0];
        let f16_arr = f32_to_f16_array(&input);
        let back = f16_to_f32_array(&f16_arr);
        for i in 0..input.len() {
            assert!(
                (back[i] - input[i]).abs() < 1e-3,
                "mismatch at {i}: expected {}, got {}",
                input[i],
                back[i]
            );
        }
    }

    #[test]
    fn test_f32_bf16_roundtrip_array() {
        let input = array![1.0f32, -2.0, 100.0, 0.0, 0.5];
        let bf16_arr = f32_to_bf16_array(&input);
        let back = bf16_to_f32_array(&bf16_arr);
        for i in 0..input.len() {
            assert!(
                (back[i] - input[i]).abs() < 1.0,
                "mismatch at {i}: expected {}, got {}",
                input[i],
                back[i]
            );
        }
    }

    #[test]
    fn test_f64_f16_roundtrip_array() {
        let input = array![1.0f64, -1.0, 0.25, 4.0];
        let f16_arr = f64_to_f16_array(&input);
        let back = f16_to_f64_array(&f16_arr);
        for i in 0..input.len() {
            assert!(
                (back[i] - input[i]).abs() < 1e-3,
                "mismatch at {i}: expected {}, got {}",
                input[i],
                back[i]
            );
        }
    }

    #[test]
    fn test_f32_f16_roundtrip_matrix() {
        let input = Array2::from_shape_vec((2, 3), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("shape ok");
        let f16_mat = f32_to_f16_matrix(&input);
        assert_eq!(f16_mat.shape(), &[2, 3]);
        let back = f16_to_f32_matrix(&f16_mat);
        assert_eq!(back.shape(), &[2, 3]);
        for i in 0..2 {
            for j in 0..3 {
                assert!(
                    (back[[i, j]] - input[[i, j]]).abs() < 1e-3,
                    "matrix mismatch at [{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn test_f32_bf16_roundtrip_matrix() {
        let input =
            Array2::from_shape_vec((2, 2), vec![1.0f32, -2.0, 3.0, -4.0]).expect("shape ok");
        let bf16_mat = f32_to_bf16_matrix(&input);
        assert_eq!(bf16_mat.shape(), &[2, 2]);
        let back = bf16_to_f32_matrix(&bf16_mat);
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (back[[i, j]] - input[[i, j]]).abs() < 1.0,
                    "matrix mismatch at [{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn test_f64_f16_matrix() {
        let input = Array2::from_shape_vec((2, 2), vec![1.0f64, 2.0, 3.0, 4.0]).expect("shape ok");
        let f16_mat = f64_to_f16_matrix(&input);
        let back = f16_to_f64_matrix(&f16_mat);
        for i in 0..2 {
            for j in 0..2 {
                assert!((back[[i, j]] - input[[i, j]]).abs() < 1e-2);
            }
        }
    }

    #[test]
    fn test_bf16_f64_conversions() {
        let input = array![10.0f64, -20.0, 0.0, 1e10];
        let bf16_arr = f64_to_bf16_array(&input);
        let back = bf16_to_f64_array(&bf16_arr);
        // Values within bf16 precision
        assert!((back[0] - 10.0).abs() < 1.0);
        assert!((back[1] + 20.0).abs() < 1.0);
        assert_eq!(back[2], 0.0);
    }

    #[test]
    fn test_f16_bf16_cross_conversion() {
        let f16_arr = array![F16::from_f32(1.0), F16::from_f32(2.0), F16::from_f32(-0.5)];
        let bf16_arr = f16_to_bf16_array(&f16_arr);
        assert_eq!(bf16_arr.len(), 3);
        assert!((bf16_arr[0].to_f32() - 1.0).abs() < 0.01);
        assert!((bf16_arr[1].to_f32() - 2.0).abs() < 0.01);

        let back = bf16_to_f16_array(&bf16_arr);
        assert_eq!(back.len(), 3);
        assert!((back[0].to_f32() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_empty_array_conversions() {
        let empty_f32 = Array1::<f32>::zeros(0);
        let f16_arr = f32_to_f16_array(&empty_f32);
        assert_eq!(f16_arr.len(), 0);
        let back = f16_to_f32_array(&f16_arr);
        assert_eq!(back.len(), 0);
    }
}
