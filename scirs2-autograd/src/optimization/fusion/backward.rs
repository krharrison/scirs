//! Backward pass support for fused operations
//!
//! This module implements gradient computation for fused operations.
//! Each fused operation must provide a backward pass that correctly
//! distributes gradients to all inputs involved in the fusion.

use crate::error::AutogradError;
use crate::Result;
use scirs2_core::ndarray::{Array, Axis, IxDyn};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Gradient structures
// ---------------------------------------------------------------------------

/// Gradients for fused linear operations (matmul + bias + activation)
#[derive(Debug, Clone)]
pub struct LinearGradients<F: Float> {
    /// Gradient with respect to input x
    pub grad_x: Array<F, IxDyn>,
    /// Gradient with respect to weight w
    pub grad_w: Array<F, IxDyn>,
    /// Gradient with respect to bias
    pub grad_bias: Array<F, IxDyn>,
}

/// Gradients for fused affine operations (x * scale + shift)
#[derive(Debug, Clone)]
pub struct AffineGradients<F: Float> {
    /// Gradient with respect to input x
    pub grad_x: Array<F, IxDyn>,
    /// Gradient with respect to scale
    pub grad_scale: Array<F, IxDyn>,
    /// Gradient with respect to shift
    pub grad_shift: Array<F, IxDyn>,
}

// ---------------------------------------------------------------------------
// Backward pass for fused linear operations
// ---------------------------------------------------------------------------

/// Compute gradients for fused_linear: matmul(x, w) + bias
///
/// Given upstream gradient `grad_output`, computes gradients with respect to x, w, and bias.
///
/// # Mathematical formulation
/// Forward: `y = x @ w + bias`
/// Backward:
/// - `grad_x = grad_output @ w^T`
/// - `grad_w = x^T @ grad_output`
/// - `grad_bias = sum(grad_output, axis=0)`
pub fn fused_linear_backward<F: Float + Debug + Send + Sync + 'static>(
    grad_output: &Array<F, IxDyn>,
    x: &Array<F, IxDyn>,
    w: &Array<F, IxDyn>,
) -> Result<LinearGradients<F>> {
    validate_linear_backward_shapes(grad_output, x, w)?;

    let batch = x.shape()[0];
    let in_features = x.shape()[1];
    let out_features = w.shape()[1];

    // grad_x = grad_output @ w^T
    let mut grad_x = Array::<F, _>::zeros(vec![batch, in_features]);
    for i in 0..batch {
        for k in 0..in_features {
            let mut acc = F::zero();
            for j in 0..out_features {
                acc = acc + grad_output[[i, j]] * w[[k, j]];
            }
            grad_x[[i, k]] = acc;
        }
    }

    // grad_w = x^T @ grad_output
    let mut grad_w = Array::<F, _>::zeros(vec![in_features, out_features]);
    for k in 0..in_features {
        for j in 0..out_features {
            let mut acc = F::zero();
            for i in 0..batch {
                acc = acc + x[[i, k]] * grad_output[[i, j]];
            }
            grad_w[[k, j]] = acc;
        }
    }

    // grad_bias = sum(grad_output, axis=0)
    let grad_bias = grad_output.sum_axis(Axis(0)).into_dyn();

    Ok(LinearGradients {
        grad_x: grad_x.into_dyn(),
        grad_w: grad_w.into_dyn(),
        grad_bias,
    })
}

/// Compute gradients for fused_linear_relu: max(0, matmul(x, w) + bias)
///
/// Backward pass must account for ReLU's piecewise gradient (1 if output > 0, else 0).
pub fn fused_linear_relu_backward<F: Float + Debug + Send + Sync + 'static>(
    grad_output: &Array<F, IxDyn>,
    x: &Array<F, IxDyn>,
    w: &Array<F, IxDyn>,
    output: &Array<F, IxDyn>,
) -> Result<LinearGradients<F>> {
    validate_linear_backward_shapes(grad_output, x, w)?;

    // Apply ReLU gradient mask: grad_output * (output > 0)
    let grad_masked = grad_output.mapv(|g| g)
        * &output.mapv(|v| if v > F::zero() { F::one() } else { F::zero() });

    fused_linear_backward(&grad_masked, x, w)
}

/// Compute gradients for fused_linear_sigmoid: sigmoid(matmul(x, w) + bias)
///
/// Sigmoid gradient: sigmoid(x) * (1 - sigmoid(x))
pub fn fused_linear_sigmoid_backward<F: Float + Debug + Send + Sync + 'static>(
    grad_output: &Array<F, IxDyn>,
    x: &Array<F, IxDyn>,
    w: &Array<F, IxDyn>,
    output: &Array<F, IxDyn>,
) -> Result<LinearGradients<F>> {
    validate_linear_backward_shapes(grad_output, x, w)?;

    // Sigmoid gradient: output * (1 - output)
    let sigmoid_grad = output.mapv(|y| y * (F::one() - y));
    let grad_masked = grad_output * &sigmoid_grad;

    fused_linear_backward(&grad_masked, x, w)
}

/// Compute gradients for fused_linear_tanh: tanh(matmul(x, w) + bias)
///
/// Tanh gradient: 1 - tanh^2(x)
pub fn fused_linear_tanh_backward<F: Float + Debug + Send + Sync + 'static>(
    grad_output: &Array<F, IxDyn>,
    x: &Array<F, IxDyn>,
    w: &Array<F, IxDyn>,
    output: &Array<F, IxDyn>,
) -> Result<LinearGradients<F>> {
    validate_linear_backward_shapes(grad_output, x, w)?;

    // Tanh gradient: 1 - tanh^2(x) = 1 - output^2
    let tanh_grad = output.mapv(|y| F::one() - y * y);
    let grad_masked = grad_output * &tanh_grad;

    fused_linear_backward(&grad_masked, x, w)
}

/// Compute gradients for fused_linear_gelu
///
/// GELU gradient (tanh approximation):
/// `d/dx GELU(x) ≈ 0.5 * (1 + tanh(z)) + 0.5 * x * sech^2(z) * dz/dx`
/// where `z = sqrt(2/pi) * (x + 0.044715 * x^3)`
pub fn fused_linear_gelu_backward<F: Float + Debug + Send + Sync + 'static>(
    grad_output: &Array<F, IxDyn>,
    x: &Array<F, IxDyn>,
    w: &Array<F, IxDyn>,
    linear_output: &Array<F, IxDyn>,
) -> Result<LinearGradients<F>> {
    validate_linear_backward_shapes(grad_output, x, w)?;

    let sqrt_2_over_pi = F::from(0.7978845608028654).unwrap_or(F::one());
    let coeff = F::from(0.044715).unwrap_or(F::zero());
    let half = F::from(0.5).unwrap_or(F::one());

    // GELU gradient computation
    let gelu_grad = linear_output.mapv(|val| {
        let x3 = val * val * val;
        let z = sqrt_2_over_pi * (val + coeff * x3);
        let tanh_z = z.tanh();
        let sech2_z = F::one() - tanh_z * tanh_z;
        let dz_dx =
            sqrt_2_over_pi * (F::one() + coeff * F::from(3.0).unwrap_or(F::one()) * val * val);

        half * (F::one() + tanh_z) + half * val * sech2_z * dz_dx
    });

    let grad_masked = grad_output * &gelu_grad;
    fused_linear_backward(&grad_masked, x, w)
}

// ---------------------------------------------------------------------------
// Backward pass for fused affine operations
// ---------------------------------------------------------------------------

/// Compute gradients for fused_affine: x * scale + shift
///
/// Backward:
/// - `grad_x = grad_output * scale`
/// - `grad_scale = sum(grad_output * x)`
/// - `grad_shift = sum(grad_output)`
pub fn fused_affine_backward<F: Float + Debug + Send + Sync + 'static>(
    grad_output: &Array<F, IxDyn>,
    x: &Array<F, IxDyn>,
    scale: &Array<F, IxDyn>,
) -> Result<AffineGradients<F>> {
    // grad_x = grad_output * scale
    let grad_x = if scale.len() == 1 {
        let s = *scale.iter().next().unwrap_or(&F::one());
        grad_output.mapv(|g| g * s)
    } else {
        grad_output * scale
    };

    // grad_scale = sum(grad_output * x, keeping scale's shape)
    let grad_scale = if scale.len() == 1 {
        let sum: F = (grad_output * x).iter().fold(F::zero(), |acc, &v| acc + v);
        Array::from_elem(vec![1], sum).into_dyn()
    } else if grad_output.ndim() > 1 {
        (grad_output * x).sum_axis(Axis(0)).into_dyn()
    } else {
        (grad_output * x).into_dyn()
    };

    // grad_shift = sum(grad_output, keeping shift's shape)
    let grad_shift = if scale.len() == 1 {
        let sum: F = grad_output.iter().fold(F::zero(), |acc, &v| acc + v);
        Array::from_elem(vec![1], sum).into_dyn()
    } else if grad_output.ndim() > 1 {
        grad_output.sum_axis(Axis(0)).into_dyn()
    } else {
        grad_output.clone()
    };

    Ok(AffineGradients {
        grad_x,
        grad_scale,
        grad_shift,
    })
}

// ---------------------------------------------------------------------------
// Backward pass for reduction fusions
// ---------------------------------------------------------------------------

/// Compute gradient for fused_mean
///
/// Mean gradient broadcasts equally to all elements along the reduced axis.
pub fn fused_mean_backward<F: Float + Debug + Send + Sync + 'static>(
    grad_output: &Array<F, IxDyn>,
    input_shape: &[usize],
    axis: usize,
) -> Result<Array<F, IxDyn>> {
    if axis >= input_shape.len() {
        return Err(AutogradError::ShapeMismatch(format!(
            "Axis {} out of bounds for shape {:?}",
            axis, input_shape
        )));
    }

    let count = F::from(input_shape[axis]).unwrap_or(F::one());
    let grad_per_element = F::one() / count;

    // Broadcast grad_output back to input_shape
    let mut grad_input = grad_output.clone();
    grad_input = grad_input.insert_axis(Axis(axis));

    // Repeat along axis
    let mut broadcasted = Array::<F, _>::zeros(input_shape.to_vec());
    for _ in 0..input_shape[axis] {
        broadcasted = broadcasted
            + &grad_input
                .broadcast(input_shape)
                .ok_or_else(|| AutogradError::ShapeMismatch("Broadcast failed".to_string()))?;
    }

    Ok(broadcasted.mapv(|v| v * grad_per_element))
}

/// Compute gradient for fused_softmax
///
/// Softmax gradient: `grad_input = output * (grad_output - sum(output * grad_output))`
pub fn fused_softmax_backward<F: Float + Debug + Send + Sync + 'static>(
    grad_output: &Array<F, IxDyn>,
    output: &Array<F, IxDyn>,
    axis: usize,
) -> Result<Array<F, IxDyn>> {
    if axis >= output.ndim() {
        return Err(AutogradError::ShapeMismatch(format!(
            "Axis {} out of bounds for tensor with {} dimensions",
            axis,
            output.ndim()
        )));
    }

    // sum(output * grad_output) along axis
    let output_grad_prod = output * grad_output;
    let sum_vals = output_grad_prod.sum_axis(Axis(axis));

    // Broadcast sum back and compute: output * (grad_output - sum)
    let mut grad_input = output.clone();
    for (mut lane, &sum_val) in grad_input
        .lanes_mut(Axis(axis))
        .into_iter()
        .zip(sum_vals.iter())
    {
        for (out_v, grad_v) in lane.iter_mut().zip(grad_output.iter()) {
            *out_v = *out_v * (*grad_v - sum_val);
        }
    }

    Ok(grad_input)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_linear_backward_shapes<F: Float>(
    grad_output: &Array<F, IxDyn>,
    x: &Array<F, IxDyn>,
    w: &Array<F, IxDyn>,
) -> Result<()> {
    if grad_output.ndim() != 2 || x.ndim() != 2 || w.ndim() != 2 {
        return Err(AutogradError::ShapeMismatch(
            "All tensors must be 2-D for linear backward".to_string(),
        ));
    }

    let batch = x.shape()[0];
    let out_features = w.shape()[1];

    if grad_output.shape() != [batch, out_features] {
        return Err(AutogradError::ShapeMismatch(format!(
            "grad_output shape {:?} must match [batch={}, out_features={}]",
            grad_output.shape(),
            batch,
            out_features
        )));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array as NdArray;

    fn arr2d(rows: usize, cols: usize, vals: &[f64]) -> Array<f64, IxDyn> {
        NdArray::from_shape_vec((rows, cols), vals.to_vec())
            .expect("valid shape")
            .into_dyn()
    }

    #[test]
    fn test_fused_linear_backward_basic() {
        // Simple 2x2 case
        let x = arr2d(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let w = arr2d(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let grad_output = arr2d(2, 2, &[1.0, 1.0, 1.0, 1.0]);

        let grads = fused_linear_backward(&grad_output, &x, &w).expect("backward should succeed");

        // grad_w = x^T @ grad_output
        // x^T = [[1,3],[2,4]], grad_output = [[1,1],[1,1]]
        // grad_w = [[1+3, 1+3], [2+4, 2+4]] = [[4,4],[6,6]]
        let grad_w_flat: Vec<f64> = grads.grad_w.iter().copied().collect();
        assert_eq!(grad_w_flat, vec![4.0, 4.0, 6.0, 6.0]);
    }

    #[test]
    fn test_fused_linear_relu_backward() {
        let x = arr2d(1, 2, &[1.0, -1.0]);
        let w = arr2d(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let grad_output = arr2d(1, 2, &[1.0, 1.0]);

        // Forward output (for mask): [1, -1] @ [[1,0],[0,1]] = [1, -1]
        // After ReLU: [1, 0]
        let output = arr2d(1, 2, &[1.0, 0.0]);

        let grads = fused_linear_relu_backward(&grad_output, &x, &w, &output)
            .expect("backward should succeed");

        // Gradient should be masked by ReLU: only first element has gradient
        assert!(grads.grad_x.shape() == &[1, 2]);
    }

    #[test]
    fn test_fused_linear_sigmoid_backward() {
        let x = arr2d(1, 1, &[1.0]);
        let w = arr2d(1, 1, &[1.0]);
        let grad_output = arr2d(1, 1, &[1.0]);

        // sigmoid(1) ≈ 0.731
        let output = arr2d(1, 1, &[0.7310585786]);

        let grads = fused_linear_sigmoid_backward(&grad_output, &x, &w, &output)
            .expect("backward should succeed");

        // Sigmoid gradient: y * (1 - y) ≈ 0.731 * 0.269 ≈ 0.197
        let grad_val = grads.grad_bias.iter().next().copied().unwrap_or(0.0);
        assert!((grad_val - 0.197).abs() < 0.01);
    }

    #[test]
    fn test_fused_linear_tanh_backward() {
        let x = arr2d(1, 1, &[1.0]);
        let w = arr2d(1, 1, &[1.0]);
        let grad_output = arr2d(1, 1, &[1.0]);

        // tanh(1) ≈ 0.7616
        let output = arr2d(1, 1, &[0.7615941559]);

        let grads = fused_linear_tanh_backward(&grad_output, &x, &w, &output)
            .expect("backward should succeed");

        // Tanh gradient: 1 - tanh^2(1) ≈ 1 - 0.58 ≈ 0.42
        let grad_val = grads.grad_bias.iter().next().copied().unwrap_or(0.0);
        assert!((grad_val - 0.42).abs() < 0.01);
    }

    #[test]
    fn test_fused_affine_backward() {
        let x = arr2d(1, 3, &[1.0, 2.0, 3.0]);
        let scale = arr2d(1, 3, &[2.0, 3.0, 4.0]);
        let grad_output = arr2d(1, 3, &[1.0, 1.0, 1.0]);

        let grads =
            fused_affine_backward(&grad_output, &x, &scale).expect("backward should succeed");

        // grad_x = grad_output * scale = [2, 3, 4]
        let grad_x_flat: Vec<f64> = grads.grad_x.iter().copied().collect();
        assert_eq!(grad_x_flat, vec![2.0, 3.0, 4.0]);

        // grad_scale = grad_output * x = [1, 2, 3]
        let grad_scale_flat: Vec<f64> = grads.grad_scale.iter().copied().collect();
        assert_eq!(grad_scale_flat, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_fused_affine_backward_scalar_scale() {
        let x = arr2d(1, 3, &[1.0, 2.0, 3.0]);
        let scale = arr2d(1, 1, &[2.0]);
        let grad_output = arr2d(1, 3, &[1.0, 1.0, 1.0]);

        let grads =
            fused_affine_backward(&grad_output, &x, &scale).expect("backward should succeed");

        // grad_scale = sum(grad_output * x) = sum([1,2,3]) = 6
        let grad_scale_val = grads.grad_scale.iter().next().copied().unwrap_or(0.0);
        assert_eq!(grad_scale_val, 6.0);
    }

    #[test]
    fn test_linear_backward_shape_mismatch() {
        let x = arr2d(2, 3, &[1.0; 6]);
        let w = arr2d(3, 2, &[1.0; 6]);
        let grad_output = arr2d(2, 3, &[1.0; 6]); // Wrong: should be 2x2

        let result = fused_linear_backward(&grad_output, &x, &w);
        assert!(result.is_err());
    }
}
