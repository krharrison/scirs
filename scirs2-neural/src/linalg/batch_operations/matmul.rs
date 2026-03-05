//! Matrix multiplication operations for batch processing in neural networks
//!
//! This module contains optimized functions for batch matrix multiplication operations
//! that are commonly used in neural network computations.

use scirs2_core::ndarray::{s, Array2, Array3, ArrayView2, ArrayView3};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::error::{NeuralError, Result};

/// Perform batch matrix multiplication for neural network operations.
///
/// This function multiplies batches of matrices efficiently, which is common
/// in neural network computations like batch processing of fully connected layers.
///
/// # Arguments
///
/// * `a` - First batch of matrices with shape [batch_size, m, k]
/// * `b` - Second batch of matrices with shape [batch_size, k, n]
///
/// # Returns
///
/// * Result matrix with shape [batch_size, m, n]
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array, Ix3};
/// use scirs2_neural::linalg::batch_matmul;
///
/// // Create batch of 2x2x3 matrices (batch_size=2, m=2, k=3)
/// let a = Array::from_shape_vec(
///     (2, 2, 3),
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
/// ).expect("operation should succeed");
///
/// // Create batch of 2x3x2 matrices (batch_size=2, k=3, n=2)
/// let b = Array::from_shape_vec(
///     (2, 3, 2),
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
/// ).expect("operation should succeed");
///
/// // Result should be shape [2, 2, 2]
/// let c = batch_matmul(&a.view(), &b.view()).expect("operation should succeed");
/// assert_eq!(c.shape(), &[2, 2, 2]);
/// ```
pub fn batch_matmul<F>(a: &ArrayView3<F>, b: &ArrayView3<F>) -> Result<Array3<F>>
where
    F: Float + Debug + 'static,
{
    // Get dimensions
    let batch_size = a.shape()[0];
    if batch_size != b.shape()[0] {
        return Err(NeuralError::ShapeMismatch(format!(
            "Batch size mismatch in batch_matmul: a batch_size={}, b batch_size={}",
            batch_size,
            b.shape()[0]
        )));
    }

    let m = a.shape()[1];
    let k = a.shape()[2];
    let k2 = b.shape()[1];
    let n = b.shape()[2];

    if k != k2 {
        return Err(NeuralError::ShapeMismatch(format!(
            "Inner dimensions mismatch in batch_matmul: a has k={}, b has k={}",
            k, k2
        )));
    }

    // Allocate output array
    let mut result = Array3::<F>::zeros((batch_size, m, n));

    // Perform batch matrix multiplication with SIMD optimization for f32/f64
    use std::any::TypeId;
    if TypeId::of::<F>() == TypeId::of::<f32>() && m * k + k * n >= 64 {
        // SIMD optimization for f32
        for batch_idx in 0..batch_size {
            let a_slice = a.slice(s![batch_idx, .., ..]);
            let b_slice = b.slice(s![batch_idx, .., ..]);

            // Convert to contiguous f32 vectors
            let a_vec: Vec<f32> = a_slice
                .iter()
                .map(|&v| unsafe { std::mem::transmute_copy(&v) })
                .collect();
            let b_vec: Vec<f32> = b_slice
                .iter()
                .map(|&v| unsafe { std::mem::transmute_copy(&v) })
                .collect();
            let mut c_vec = vec![0.0f32; m * n];

            // Use SIMD matrix multiplication: C = 1.0*A*B + 0.0*C
            scirs2_core::simd_ops::simd_matrix_multiply_f32(
                m, k, n, 1.0, &a_vec, &b_vec, 0.0, &mut c_vec,
            );

            // Copy back to result array
            for i in 0..m {
                for j in 0..n {
                    let val_f32 = c_vec[i * n + j];
                    result[[batch_idx, i, j]] = unsafe { std::mem::transmute_copy(&val_f32) };
                }
            }
        }
    } else if TypeId::of::<F>() == TypeId::of::<f64>() && m * k + k * n >= 64 {
        // SIMD optimization for f64
        for batch_idx in 0..batch_size {
            let a_slice = a.slice(s![batch_idx, .., ..]);
            let b_slice = b.slice(s![batch_idx, .., ..]);

            let a_vec: Vec<f64> = a_slice
                .iter()
                .map(|&v| unsafe { std::mem::transmute_copy(&v) })
                .collect();
            let b_vec: Vec<f64> = b_slice
                .iter()
                .map(|&v| unsafe { std::mem::transmute_copy(&v) })
                .collect();
            let mut c_vec = vec![0.0f64; m * n];

            scirs2_core::simd_ops::simd_matrix_multiply_f64(
                m, k, n, 1.0, &a_vec, &b_vec, 0.0, &mut c_vec,
            );

            for i in 0..m {
                for j in 0..n {
                    let val_f64 = c_vec[i * n + j];
                    result[[batch_idx, i, j]] = unsafe { std::mem::transmute_copy(&val_f64) };
                }
            }
        }
    } else {
        // Fallback to scalar implementation for other types or small matrices
        for batch_idx in 0..batch_size {
            let a_slice = a.slice(s![batch_idx, .., ..]);
            let b_slice = b.slice(s![batch_idx, .., ..]);

            for i in 0..m {
                for j in 0..n {
                    let mut sum = F::zero();

                    for l in 0..k {
                        sum = sum + a_slice[[i, l]] * b_slice[[l, j]];
                    }

                    result[[batch_idx, i, j]] = sum;
                }
            }
        }
    }

    Ok(result)
}

/// Perform batch vector-matrix multiplication for neural network operations.
///
/// This is commonly used in RNN and attention mechanisms.
///
/// # Arguments
///
/// * `v` - Batch of vectors with shape [batch_size, k]
/// * `m` - Batch of matrices with shape [batch_size, k, n]
///
/// # Returns
///
/// * Result batch of vectors with shape [batch_size, n]
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array, Ix2, Ix3};
/// use scirs2_neural::linalg::batch_vecmat;
///
/// // Create batch of 2x3 vectors (batch_size=2, k=3)
/// let v = Array::from_shape_vec(
///     (2, 3),
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
/// ).expect("operation should succeed");
///
/// // Create batch of 2x3x2 matrices (batch_size=2, k=3, n=2)
/// let m = Array::from_shape_vec(
///     (2, 3, 2),
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
/// ).expect("operation should succeed");
///
/// // Result should be shape [2, 2]
/// let result = batch_vecmat(&v.view(), &m.view()).expect("operation should succeed");
/// assert_eq!(result.shape(), &[2, 2]);
/// ```
pub fn batch_vecmat<F>(v: &ArrayView2<F>, m: &ArrayView3<F>) -> Result<Array2<F>>
where
    F: Float + Debug + 'static,
{
    // Get dimensions
    let batch_size = v.shape()[0];
    if batch_size != m.shape()[0] {
        return Err(NeuralError::ShapeMismatch(format!(
            "Batch size mismatch in batch_vecmat: v batch_size={}, m batch_size={}",
            batch_size,
            m.shape()[0]
        )));
    }

    let k = v.shape()[1];
    let k2 = m.shape()[1];
    let n = m.shape()[2];

    if k != k2 {
        return Err(NeuralError::ShapeMismatch(format!(
            "Inner dimensions mismatch in batch_vecmat: v has k={}, m has k={}",
            k, k2
        )));
    }

    // Allocate output array
    let mut result = Array2::<F>::zeros((batch_size, n));

    // Perform batch vector-matrix multiplication
    for batch_idx in 0..batch_size {
        let v_slice = v.slice(s![batch_idx, ..]);
        let m_slice = m.slice(s![batch_idx, .., ..]);

        for j in 0..n {
            let mut sum = F::zero();

            for l in 0..k {
                sum = sum + v_slice[l] * m_slice[[l, j]];
            }

            result[[batch_idx, j]] = sum;
        }
    }

    Ok(result)
}

/// Computes the gradient for batch matrix multiplication.
///
/// This function calculates the gradients with respect to inputs `a` and `b`
/// for the batch matrix multiplication operation.
///
/// # Arguments
///
/// * `a` - First batch of matrices with shape [batch_size, m, k]
/// * `b` - Second batch of matrices with shape [batch_size, k, n]
/// * `grad_output` - Gradient of the loss with respect to the output of batch_matmul,
///   with shape [batch_size, m, n]
///
/// # Returns
///
/// * Tuple of (grad_a, grad_b) where:
///   - grad_a has shape [batch_size, m, k]
///   - grad_b has shape [batch_size, k, n]
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array, Ix3};
/// use scirs2_neural::linalg::{batch_matmul, batch_matmul_backward};
///
/// // Create input matrices
/// let a = Array::from_shape_vec(
///     (2, 2, 3),
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
/// ).expect("operation should succeed");
///
/// let b = Array::from_shape_vec(
///     (2, 3, 2),
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
/// ).expect("operation should succeed");
///
/// // Assume a gradient from the next layer
/// let grad_output = Array::from_shape_vec(
///     (2, 2, 2),
///     vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
/// ).expect("operation should succeed");
///
/// // Compute gradients
/// let (grad_a, grad_b) = batch_matmul_backward(&a.view(), &b.view(), &grad_output.view()).expect("operation should succeed");
/// assert_eq!(grad_a.shape(), a.shape());
/// assert_eq!(grad_b.shape(), b.shape());
/// ```
pub fn batch_matmul_backward<F>(
    a: &ArrayView3<F>,
    b: &ArrayView3<F>,
    grad_output: &ArrayView3<F>,
) -> Result<(Array3<F>, Array3<F>)>
where
    F: Float + Debug + 'static,
{
    // Get dimensions
    let batch_size = a.shape()[0];
    if batch_size != b.shape()[0] || batch_size != grad_output.shape()[0] {
        return Err(NeuralError::ShapeMismatch(
            format!("Batch size mismatch in batch_matmul_backward: a batch_size={}, b batch_size={}, grad_output batch_size={}", 
                    batch_size, b.shape()[0], grad_output.shape()[0])
        ));
    }

    let m = a.shape()[1];
    let k = a.shape()[2];
    let k2 = b.shape()[1];
    let n = b.shape()[2];

    let m2 = grad_output.shape()[1];
    let n2 = grad_output.shape()[2];

    if k != k2 {
        return Err(NeuralError::ShapeMismatch(format!(
            "Inner dimensions mismatch in batch_matmul_backward: a has k={}, b has k={}",
            k, k2
        )));
    }

    if m != m2 || n != n2 {
        return Err(NeuralError::ShapeMismatch(
            format!("Output dimensions mismatch in batch_matmul_backward: expected [batch_size, {}, {}], got [batch_size, {}, {}]", 
                    m, n, m2, n2)
        ));
    }

    // Allocate gradient arrays
    let mut grad_a = Array3::<F>::zeros((batch_size, m, k));
    let mut grad_b = Array3::<F>::zeros((batch_size, k, n));

    // Compute gradients with SIMD optimization for f32/f64
    use std::any::TypeId;
    if TypeId::of::<F>() == TypeId::of::<f32>() && m * k + k * n >= 64 {
        // SIMD optimization for f32
        for batch_idx in 0..batch_size {
            let b_slice = b.slice(s![batch_idx, .., ..]);
            let a_slice = a.slice(s![batch_idx, .., ..]);
            let grad_output_slice = grad_output.slice(s![batch_idx, .., ..]);

            // Compute grad_a: grad_output * b^T
            let grad_out_vec: Vec<f32> = grad_output_slice
                .iter()
                .map(|&v| unsafe { std::mem::transmute_copy(&v) })
                .collect();
            let b_t_vec: Vec<f32> = (0..k)
                .flat_map(|l| {
                    (0..n).map(move |j| {
                        let val: f32 = unsafe { std::mem::transmute_copy(&b_slice[[l, j]]) };
                        val
                    })
                })
                .collect();
            let mut grad_a_vec = vec![0.0f32; m * k];

            scirs2_core::simd_ops::simd_matrix_multiply_f32(
                m,
                n,
                k,
                1.0,
                &grad_out_vec,
                &b_t_vec,
                0.0,
                &mut grad_a_vec,
            );

            for i in 0..m {
                for l in 0..k {
                    let val_f32 = grad_a_vec[i * k + l];
                    grad_a[[batch_idx, i, l]] = unsafe { std::mem::transmute_copy(&val_f32) };
                }
            }

            // Compute grad_b: a^T * grad_output
            let a_t_vec: Vec<f32> = (0..k)
                .flat_map(|l| {
                    (0..m).map(move |i| {
                        let val: f32 = unsafe { std::mem::transmute_copy(&a_slice[[i, l]]) };
                        val
                    })
                })
                .collect();
            let mut grad_b_vec = vec![0.0f32; k * n];

            scirs2_core::simd_ops::simd_matrix_multiply_f32(
                k,
                m,
                n,
                1.0,
                &a_t_vec,
                &grad_out_vec,
                0.0,
                &mut grad_b_vec,
            );

            for l in 0..k {
                for j in 0..n {
                    let val_f32 = grad_b_vec[l * n + j];
                    grad_b[[batch_idx, l, j]] = unsafe { std::mem::transmute_copy(&val_f32) };
                }
            }
        }
    } else if TypeId::of::<F>() == TypeId::of::<f64>() && m * k + k * n >= 64 {
        // SIMD optimization for f64
        for batch_idx in 0..batch_size {
            let b_slice = b.slice(s![batch_idx, .., ..]);
            let a_slice = a.slice(s![batch_idx, .., ..]);
            let grad_output_slice = grad_output.slice(s![batch_idx, .., ..]);

            // Compute grad_a: grad_output * b^T
            let grad_out_vec: Vec<f64> = grad_output_slice
                .iter()
                .map(|&v| unsafe { std::mem::transmute_copy(&v) })
                .collect();
            let b_t_vec: Vec<f64> = (0..k)
                .flat_map(|l| {
                    (0..n).map(move |j| {
                        let val: f64 = unsafe { std::mem::transmute_copy(&b_slice[[l, j]]) };
                        val
                    })
                })
                .collect();
            let mut grad_a_vec = vec![0.0f64; m * k];

            scirs2_core::simd_ops::simd_matrix_multiply_f64(
                m,
                n,
                k,
                1.0,
                &grad_out_vec,
                &b_t_vec,
                0.0,
                &mut grad_a_vec,
            );

            for i in 0..m {
                for l in 0..k {
                    let val_f64 = grad_a_vec[i * k + l];
                    grad_a[[batch_idx, i, l]] = unsafe { std::mem::transmute_copy(&val_f64) };
                }
            }

            // Compute grad_b: a^T * grad_output
            let a_t_vec: Vec<f64> = (0..k)
                .flat_map(|l| {
                    (0..m).map(move |i| {
                        let val: f64 = unsafe { std::mem::transmute_copy(&a_slice[[i, l]]) };
                        val
                    })
                })
                .collect();
            let mut grad_b_vec = vec![0.0f64; k * n];

            scirs2_core::simd_ops::simd_matrix_multiply_f64(
                k,
                m,
                n,
                1.0,
                &a_t_vec,
                &grad_out_vec,
                0.0,
                &mut grad_b_vec,
            );

            for l in 0..k {
                for j in 0..n {
                    let val_f64 = grad_b_vec[l * n + j];
                    grad_b[[batch_idx, l, j]] = unsafe { std::mem::transmute_copy(&val_f64) };
                }
            }
        }
    } else {
        // Fallback to scalar implementation for other types or small matrices
        for batch_idx in 0..batch_size {
            let a_slice = a.slice(s![batch_idx, .., ..]);
            let b_slice = b.slice(s![batch_idx, .., ..]);
            let grad_output_slice = grad_output.slice(s![batch_idx, .., ..]);

            // Compute grad_a: grad_output * b^T
            for i in 0..m {
                for l in 0..k {
                    let mut sum = F::zero();

                    for j in 0..n {
                        sum = sum + grad_output_slice[[i, j]] * b_slice[[l, j]];
                    }

                    grad_a[[batch_idx, i, l]] = sum;
                }
            }

            // Compute grad_b: a^T * grad_output
            for l in 0..k {
                for j in 0..n {
                    let mut sum = F::zero();

                    for i in 0..m {
                        sum = sum + a_slice[[i, l]] * grad_output_slice[[i, j]];
                    }

                    grad_b[[batch_idx, l, j]] = sum;
                }
            }
        }
    }

    Ok((grad_a, grad_b))
}
