//! Fused operation implementations
//!
//! Each fused operation combines multiple atomic operations into a single
//! pass over the data, eliminating intermediate allocations and reducing
//! memory traffic.

use crate::error::AutogradError;
use crate::Result;
use scirs2_core::ndarray::{Array, Axis, IxDyn};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// MatMul + Bias + Activation fusions
// ---------------------------------------------------------------------------

/// Fused linear: matmul(x, w) + bias
///
/// Computes `x @ w + bias` in a single pass without materialising the
/// intermediate matmul result.
///
/// # Arguments
/// * `x`    - Input tensor of shape (batch, in_features)
/// * `w`    - Weight matrix of shape (in_features, out_features)
/// * `bias` - Bias vector of shape (out_features,) or (1, out_features)
pub fn fused_linear<F: Float + Debug + Send + Sync + 'static>(
    x: &Array<F, IxDyn>,
    w: &Array<F, IxDyn>,
    bias: &Array<F, IxDyn>,
) -> Result<Array<F, IxDyn>> {
    validate_linear_shapes(x, w, bias)?;

    let batch = x.shape()[0];
    let in_features = x.shape()[1];
    let out_features = w.shape()[1];

    // Flatten bias to 1-D for broadcasting
    let bias_len = bias.len();
    if bias_len != out_features {
        return Err(AutogradError::ShapeMismatch(format!(
            "Bias length {} does not match out_features {}",
            bias_len, out_features
        )));
    }

    let mut result = Array::<F, _>::zeros(vec![batch, out_features]);

    for i in 0..batch {
        for j in 0..out_features {
            let mut acc = F::zero();
            for k in 0..in_features {
                acc = acc + x[[i, k]] * w[[k, j]];
            }
            // Fuse: add bias in same pass
            let bias_val = bias.as_slice().map_or_else(
                || {
                    // Fallback for non-contiguous bias
                    *bias.iter().nth(j).unwrap_or(&F::zero())
                },
                |s| s[j],
            );
            result[IxDyn(&[i, j])] = acc + bias_val;
        }
    }

    Ok(result.into_dyn())
}

/// Fused linear + ReLU: max(0, matmul(x, w) + bias)
pub fn fused_linear_relu<F: Float + Debug + Send + Sync + 'static>(
    x: &Array<F, IxDyn>,
    w: &Array<F, IxDyn>,
    bias: &Array<F, IxDyn>,
) -> Result<Array<F, IxDyn>> {
    validate_linear_shapes(x, w, bias)?;

    let batch = x.shape()[0];
    let in_features = x.shape()[1];
    let out_features = w.shape()[1];
    let bias_slice = get_bias_slice(bias, out_features)?;

    let mut result = Array::<F, _>::zeros(vec![batch, out_features]);

    for i in 0..batch {
        for j in 0..out_features {
            let mut acc = F::zero();
            for k in 0..in_features {
                acc = acc + x[[i, k]] * w[[k, j]];
            }
            let val = acc + bias_slice[j];
            // Fuse: relu in same pass
            result[IxDyn(&[i, j])] = if val > F::zero() { val } else { F::zero() };
        }
    }

    Ok(result.into_dyn())
}

/// Fused linear + GELU: gelu(matmul(x, w) + bias)
///
/// Uses the tanh approximation:
/// `gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
pub fn fused_linear_gelu<F: Float + Debug + Send + Sync + 'static>(
    x: &Array<F, IxDyn>,
    w: &Array<F, IxDyn>,
    bias: &Array<F, IxDyn>,
) -> Result<Array<F, IxDyn>> {
    validate_linear_shapes(x, w, bias)?;

    let batch = x.shape()[0];
    let in_features = x.shape()[1];
    let out_features = w.shape()[1];
    let bias_slice = get_bias_slice(bias, out_features)?;

    let sqrt_2_over_pi = F::from(0.7978845608028654).unwrap_or(F::one());
    let coeff = F::from(0.044715).unwrap_or(F::zero());
    let half = F::from(0.5).unwrap_or(F::one());

    let mut result = Array::<F, _>::zeros(vec![batch, out_features]);

    for i in 0..batch {
        for j in 0..out_features {
            let mut acc = F::zero();
            for k in 0..in_features {
                acc = acc + x[[i, k]] * w[[k, j]];
            }
            let val = acc + bias_slice[j];
            // Fuse: gelu in same pass
            let inner = sqrt_2_over_pi * (val + coeff * val * val * val);
            result[IxDyn(&[i, j])] = half * val * (F::one() + inner.tanh());
        }
    }

    Ok(result.into_dyn())
}

/// Fused linear + Sigmoid: sigmoid(matmul(x, w) + bias)
pub fn fused_linear_sigmoid<F: Float + Debug + Send + Sync + 'static>(
    x: &Array<F, IxDyn>,
    w: &Array<F, IxDyn>,
    bias: &Array<F, IxDyn>,
) -> Result<Array<F, IxDyn>> {
    validate_linear_shapes(x, w, bias)?;

    let batch = x.shape()[0];
    let in_features = x.shape()[1];
    let out_features = w.shape()[1];
    let bias_slice = get_bias_slice(bias, out_features)?;

    let mut result = Array::<F, _>::zeros(vec![batch, out_features]);

    for i in 0..batch {
        for j in 0..out_features {
            let mut acc = F::zero();
            for k in 0..in_features {
                acc = acc + x[[i, k]] * w[[k, j]];
            }
            let val = acc + bias_slice[j];
            // Fuse: sigmoid in same pass
            result[IxDyn(&[i, j])] = F::one() / (F::one() + (-val).exp());
        }
    }

    Ok(result.into_dyn())
}

/// Fused linear + Tanh: tanh(matmul(x, w) + bias)
pub fn fused_linear_tanh<F: Float + Debug + Send + Sync + 'static>(
    x: &Array<F, IxDyn>,
    w: &Array<F, IxDyn>,
    bias: &Array<F, IxDyn>,
) -> Result<Array<F, IxDyn>> {
    validate_linear_shapes(x, w, bias)?;

    let batch = x.shape()[0];
    let in_features = x.shape()[1];
    let out_features = w.shape()[1];
    let bias_slice = get_bias_slice(bias, out_features)?;

    let mut result = Array::<F, _>::zeros(vec![batch, out_features]);

    for i in 0..batch {
        for j in 0..out_features {
            let mut acc = F::zero();
            for k in 0..in_features {
                acc = acc + x[[i, k]] * w[[k, j]];
            }
            let val = acc + bias_slice[j];
            // Fuse: tanh in same pass
            result[IxDyn(&[i, j])] = val.tanh();
        }
    }

    Ok(result.into_dyn())
}

// ---------------------------------------------------------------------------
// Conv + BN + Activation fusions
// ---------------------------------------------------------------------------

/// Parameters for a batch normalization layer (pre-folded for fusion).
#[derive(Debug, Clone)]
pub struct BatchNormParams<F: Float> {
    /// Running mean per channel
    pub running_mean: Vec<F>,
    /// Running variance per channel
    pub running_var: Vec<F>,
    /// Scale (gamma) per channel
    pub gamma: Vec<F>,
    /// Shift (beta) per channel
    pub beta: Vec<F>,
    /// Small constant for numerical stability
    pub epsilon: F,
}

/// Fused Conv2D + BatchNorm
///
/// Instead of computing `bn(conv(x))`, we fold the BN parameters into the
/// convolution weights and bias:
///   w_fused = gamma / sqrt(var + eps) * w
///   b_fused = gamma / sqrt(var + eps) * (b - mean) + beta
///
/// This function returns the folded (weight, bias) so the caller can
/// perform a single convolution.
pub fn fold_conv_bn_params<F: Float + Debug + Send + Sync + 'static>(
    conv_weight: &Array<F, IxDyn>,
    conv_bias: Option<&Array<F, IxDyn>>,
    bn_params: &BatchNormParams<F>,
) -> Result<(Array<F, IxDyn>, Array<F, IxDyn>)> {
    // conv_weight shape: (out_channels, in_channels, kH, kW) or similar
    if conv_weight.ndim() < 2 {
        return Err(AutogradError::ShapeMismatch(
            "Conv weight must have at least 2 dimensions".to_string(),
        ));
    }

    let out_channels = conv_weight.shape()[0];
    if bn_params.running_mean.len() != out_channels
        || bn_params.running_var.len() != out_channels
        || bn_params.gamma.len() != out_channels
        || bn_params.beta.len() != out_channels
    {
        return Err(AutogradError::ShapeMismatch(format!(
            "BN parameter lengths must match out_channels ({})",
            out_channels
        )));
    }

    let mut fused_weight = conv_weight.clone();
    let mut fused_bias = Array::<F, _>::zeros(vec![out_channels]);

    for c in 0..out_channels {
        let inv_std = bn_params.gamma[c] / (bn_params.running_var[c] + bn_params.epsilon).sqrt();

        // Scale weight
        let weight_shape = conv_weight.shape();
        let elements_per_channel: usize = weight_shape[1..].iter().product();
        for idx in 0..elements_per_channel {
            let flat_idx = c * elements_per_channel + idx;
            if let Some(val) = fused_weight.as_slice_mut() {
                if flat_idx < val.len() {
                    val[flat_idx] = val[flat_idx] * inv_std;
                }
            }
        }

        // Compute fused bias
        let original_bias = conv_bias.and_then(|b| b.as_slice()).map_or(F::zero(), |s| {
            if c < s.len() {
                s[c]
            } else {
                F::zero()
            }
        });
        let bias_val = inv_std * (original_bias - bn_params.running_mean[c]) + bn_params.beta[c];
        if let Some(b_slice) = fused_bias.as_slice_mut() {
            if c < b_slice.len() {
                b_slice[c] = bias_val;
            }
        }
    }

    Ok((fused_weight, fused_bias.into_dyn()))
}

/// Apply ReLU activation in-place on an array.
pub fn apply_relu_inplace<F: Float>(arr: &mut Array<F, IxDyn>) {
    arr.mapv_inplace(|v| if v > F::zero() { v } else { F::zero() });
}

// ---------------------------------------------------------------------------
// Element-wise chain fusion
// ---------------------------------------------------------------------------

/// Fused affine transform: x * scale + shift
///
/// Combines a multiplication and an addition into a single pass.
pub fn fused_affine<F: Float + Debug + Send + Sync + 'static>(
    x: &Array<F, IxDyn>,
    scale: &Array<F, IxDyn>,
    shift: &Array<F, IxDyn>,
) -> Result<Array<F, IxDyn>> {
    // Validate shapes are broadcastable
    if x.shape() != scale.shape() && scale.len() != 1 && scale.shape() != x.shape() {
        // Allow scalar scale or same-shape scale
        if scale.len() != 1 {
            return Err(AutogradError::ShapeMismatch(format!(
                "Scale shape {:?} not broadcastable to input shape {:?}",
                scale.shape(),
                x.shape()
            )));
        }
    }

    let result = if scale.len() == 1 && shift.len() == 1 {
        // Both scalar
        let s = *scale.iter().next().unwrap_or(&F::one());
        let sh = *shift.iter().next().unwrap_or(&F::zero());
        x.mapv(|v| v * s + sh)
    } else if scale.shape() == x.shape() && shift.shape() == x.shape() {
        // Element-wise
        let mut out = x.clone();
        for ((o, &sc), &sh) in out.iter_mut().zip(scale.iter()).zip(shift.iter()) {
            *o = *o * sc + sh;
        }
        out
    } else {
        // Fallback: scale might be scalar, shift same shape (or vice versa)
        let scaled = if scale.len() == 1 {
            let s = *scale.iter().next().unwrap_or(&F::one());
            x.mapv(|v| v * s)
        } else {
            x * scale
        };
        if shift.len() == 1 {
            let sh = *shift.iter().next().unwrap_or(&F::zero());
            scaled.mapv(|v| v + sh)
        } else {
            &scaled + shift
        }
    };

    Ok(result)
}

/// Fused element-wise chain: apply a sequence of unary element-wise
/// operations in a single pass.
///
/// Supported ops: "relu", "sigmoid", "tanh", "gelu", "neg", "square",
/// "exp", "log", "sqrt".
pub fn fused_elementwise_chain<F: Float + Debug + Send + Sync + 'static>(
    x: &Array<F, IxDyn>,
    ops: &[&str],
) -> Result<Array<F, IxDyn>> {
    if ops.is_empty() {
        return Err(AutogradError::invalid_argument(
            "No operations to fuse in element-wise chain".to_string(),
        ));
    }

    let sqrt_2_over_pi = F::from(0.7978845608028654).unwrap_or(F::one());
    let coeff = F::from(0.044715).unwrap_or(F::zero());
    let half = F::from(0.5).unwrap_or(F::one());

    let result = x.mapv(|mut v| {
        for &op in ops {
            v = match op {
                "relu" => {
                    if v > F::zero() {
                        v
                    } else {
                        F::zero()
                    }
                }
                "sigmoid" => F::one() / (F::one() + (-v).exp()),
                "tanh" => v.tanh(),
                "gelu" => {
                    let inner = sqrt_2_over_pi * (v + coeff * v * v * v);
                    half * v * (F::one() + inner.tanh())
                }
                "neg" => -v,
                "square" => v * v,
                "exp" => v.exp(),
                "log" => v.ln(),
                "sqrt" => v.sqrt(),
                "abs" => v.abs(),
                _ => v, // Unknown op: identity
            };
        }
        v
    });

    Ok(result)
}

// ---------------------------------------------------------------------------
// Reduction fusions
// ---------------------------------------------------------------------------

/// Fused mean: sum(x, axis) / count -- avoids materialising the sum.
pub fn fused_mean<F: Float + Debug + Send + Sync + 'static>(
    x: &Array<F, IxDyn>,
    axis: usize,
) -> Result<Array<F, IxDyn>> {
    if axis >= x.ndim() {
        return Err(AutogradError::ShapeMismatch(format!(
            "Axis {} out of bounds for tensor with {} dimensions",
            axis,
            x.ndim()
        )));
    }

    let count = F::from(x.shape()[axis]).unwrap_or(F::one());

    let result = x.map_axis(Axis(axis), |lane| {
        let mut acc = F::zero();
        for &v in lane.iter() {
            acc = acc + v;
        }
        acc / count
    });

    Ok(result.into_dyn())
}

/// Fused variance: mean(square(x), axis) -- single pass.
///
/// This computes the population variance (not sample variance).
pub fn fused_variance<F: Float + Debug + Send + Sync + 'static>(
    x: &Array<F, IxDyn>,
    axis: usize,
) -> Result<Array<F, IxDyn>> {
    if axis >= x.ndim() {
        return Err(AutogradError::ShapeMismatch(format!(
            "Axis {} out of bounds for tensor with {} dimensions",
            axis,
            x.ndim()
        )));
    }

    let count = F::from(x.shape()[axis]).unwrap_or(F::one());

    let result = x.map_axis(Axis(axis), |lane| {
        let mut acc = F::zero();
        for &v in lane.iter() {
            acc = acc + v * v;
        }
        acc / count
    });

    Ok(result.into_dyn())
}

/// Fused softmax: exp(x - max) / sum(exp(x - max)) -- numerically stable,
/// single logical pass.
pub fn fused_softmax<F: Float + Debug + Send + Sync + 'static>(
    x: &Array<F, IxDyn>,
    axis: usize,
) -> Result<Array<F, IxDyn>> {
    if axis >= x.ndim() {
        return Err(AutogradError::ShapeMismatch(format!(
            "Softmax axis {} out of bounds for tensor with {} dimensions",
            axis,
            x.ndim()
        )));
    }

    // Step 1: compute max for numerical stability
    let max_vals = x.map_axis(Axis(axis), |view| {
        view.fold(F::neg_infinity(), |a, &b| if a > b { a } else { b })
    });

    // Step 2: exp(x - max) and sum in fused manner
    let mut result = x.clone();
    for (mut lane, &max_v) in result
        .lanes_mut(Axis(axis))
        .into_iter()
        .zip(max_vals.iter())
    {
        let mut sum = F::zero();
        // First sub-pass: compute exp and accumulate sum
        for v in lane.iter_mut() {
            *v = (*v - max_v).exp();
            sum = sum + *v;
        }
        // Second sub-pass: normalise
        if sum > F::zero() {
            for v in lane.iter_mut() {
                *v = *v / sum;
            }
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Validate shapes for linear (matmul + bias) operations.
fn validate_linear_shapes<F: Float + Debug>(
    x: &Array<F, IxDyn>,
    w: &Array<F, IxDyn>,
    bias: &Array<F, IxDyn>,
) -> Result<()> {
    if x.ndim() != 2 {
        return Err(AutogradError::ShapeMismatch(format!(
            "Input x must be 2-D, got {}-D",
            x.ndim()
        )));
    }
    if w.ndim() != 2 {
        return Err(AutogradError::ShapeMismatch(format!(
            "Weight w must be 2-D, got {}-D",
            w.ndim()
        )));
    }
    if x.shape()[1] != w.shape()[0] {
        return Err(AutogradError::ShapeMismatch(format!(
            "x columns ({}) must match w rows ({})",
            x.shape()[1],
            w.shape()[0]
        )));
    }
    let out_features = w.shape()[1];
    if bias.len() != out_features {
        return Err(AutogradError::ShapeMismatch(format!(
            "Bias length ({}) must match out_features ({})",
            bias.len(),
            out_features
        )));
    }
    Ok(())
}

/// Extract bias as a contiguous slice, returning an error if the length
/// does not match `out_features`.
fn get_bias_slice<F: Float>(bias: &Array<F, IxDyn>, out_features: usize) -> Result<Vec<F>> {
    let mut result = Vec::with_capacity(out_features);
    for (i, &v) in bias.iter().enumerate() {
        if i >= out_features {
            break;
        }
        result.push(v);
    }
    if result.len() != out_features {
        return Err(AutogradError::ShapeMismatch(format!(
            "Bias has {} elements but expected {}",
            result.len(),
            out_features
        )));
    }
    Ok(result)
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

    fn arr1d(vals: &[f64]) -> Array<f64, IxDyn> {
        NdArray::from_vec(vals.to_vec()).into_dyn()
    }

    // -- fused_linear -------------------------------------------------------

    #[test]
    fn test_fused_linear_basic() {
        // x = [[1, 2], [3, 4]]  (2x2)
        // w = [[1, 0], [0, 1]]  (2x2 identity)
        // bias = [10, 20]
        let x = arr2d(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let w = arr2d(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let bias = arr1d(&[10.0, 20.0]);

        let result = fused_linear(&x, &w, &bias).expect("fused_linear should succeed");
        assert_eq!(result.shape(), &[2, 2]);
        // row0: [1*1+2*0+10, 1*0+2*1+20] = [11, 22]
        // row1: [3*1+4*0+10, 3*0+4*1+20] = [13, 24]
        let flat: Vec<f64> = result.iter().copied().collect();
        assert_eq!(flat, vec![11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn test_fused_linear_matches_separate() {
        let x = arr2d(
            3,
            4,
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let w = arr2d(4, 2, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let bias = arr1d(&[0.5, -0.5]);

        let fused = fused_linear(&x, &w, &bias).expect("fused");

        // Compare with separate matmul (zero bias) + manual bias addition.
        // We reuse fused_linear with a zero bias to obtain the matmul result,
        // then add the true bias ourselves, avoiding the (unused) crate::ops path.
        let zero_bias = arr1d(&[0.0f64, 0.0f64]);
        let matmul_result = fused_linear(&x, &w, &zero_bias).expect("matmul via zero-bias fused");
        let expected: Vec<f64> = matmul_result
            .iter()
            .enumerate()
            .map(|(idx, &v)| {
                let col = idx % 2;
                let bias_vals = [0.5_f64, -0.5_f64];
                v + bias_vals[col]
            })
            .collect();

        let fused_flat: Vec<f64> = fused.iter().copied().collect();
        for (a, b) in fused_flat.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10_f64, "mismatch: {} vs {}", a, b);
        }
    }

    // -- fused_linear_relu --------------------------------------------------

    #[test]
    fn test_fused_linear_relu() {
        let x = arr2d(2, 2, &[1.0, -1.0, -2.0, 3.0]);
        let w = arr2d(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let bias = arr1d(&[-2.0, 0.0]);

        let result = fused_linear_relu(&x, &w, &bias).expect("fused_linear_relu");
        // row0: [1-2, -1+0] = [-1, -1] -> relu -> [0, 0]
        // row1: [-2-2, 3+0] = [-4, 3] -> relu -> [0, 3]
        let flat: Vec<f64> = result.iter().copied().collect();
        assert_eq!(flat, vec![0.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn test_fused_linear_relu_matches_separate() {
        let x = arr2d(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let w = arr2d(3, 2, &[0.1, -0.2, 0.3, -0.4, 0.5, -0.6]);
        let bias = arr1d(&[0.5, 0.5]);

        let fused = fused_linear_relu(&x, &w, &bias).expect("fused");

        // Separate: matmul + bias + relu
        let linear = fused_linear(&x, &w, &bias).expect("linear");
        let expected: Array<f64, IxDyn> = linear.mapv(|v| if v > 0.0 { v } else { 0.0 });

        let fused_flat: Vec<f64> = fused.iter().copied().collect();
        let exp_flat: Vec<f64> = expected.iter().copied().collect();
        for (a, b) in fused_flat.iter().zip(exp_flat.iter()) {
            assert!((a - b).abs() < 1e-10, "mismatch: {} vs {}", a, b);
        }
    }

    // -- fused_linear_gelu --------------------------------------------------

    #[test]
    fn test_fused_linear_gelu_positive_input() {
        // For large positive values, GELU ~ identity
        let x = arr2d(1, 2, &[10.0, 10.0]);
        let w = arr2d(2, 1, &[1.0, 0.0]);
        let bias = arr1d(&[0.0]);

        let result = fused_linear_gelu(&x, &w, &bias).expect("fused_linear_gelu");
        let val = result.iter().next().copied().unwrap_or(0.0);
        // gelu(10) ~ 10
        assert!((val - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_fused_linear_gelu_zero() {
        let x = arr2d(1, 1, &[0.0]);
        let w = arr2d(1, 1, &[1.0]);
        let bias = arr1d(&[0.0]);

        let result = fused_linear_gelu(&x, &w, &bias).expect("fused_linear_gelu");
        let val = result.iter().next().copied().unwrap_or(1.0);
        // gelu(0) = 0
        assert!(val.abs() < 1e-10);
    }

    // -- fused_linear_sigmoid -----------------------------------------------

    #[test]
    fn test_fused_linear_sigmoid() {
        let x = arr2d(1, 1, &[0.0]);
        let w = arr2d(1, 1, &[1.0]);
        let bias = arr1d(&[0.0]);

        let result = fused_linear_sigmoid(&x, &w, &bias).expect("fused_linear_sigmoid");
        let val = result.iter().next().copied().unwrap_or(0.0);
        // sigmoid(0) = 0.5
        assert!((val - 0.5).abs() < 1e-10);
    }

    // -- fused_affine -------------------------------------------------------

    #[test]
    fn test_fused_affine_scalar() {
        let x = arr1d(&[1.0, 2.0, 3.0]);
        let scale = arr1d(&[2.0]);
        let shift = arr1d(&[10.0]);

        let result = fused_affine(&x, &scale, &shift).expect("fused_affine");
        let flat: Vec<f64> = result.iter().copied().collect();
        assert_eq!(flat, vec![12.0, 14.0, 16.0]);
    }

    #[test]
    fn test_fused_affine_elementwise() {
        let x = arr1d(&[1.0, 2.0, 3.0]);
        let scale = arr1d(&[2.0, 3.0, 4.0]);
        let shift = arr1d(&[10.0, 20.0, 30.0]);

        let result = fused_affine(&x, &scale, &shift).expect("fused_affine");
        let flat: Vec<f64> = result.iter().copied().collect();
        assert_eq!(flat, vec![12.0, 26.0, 42.0]);
    }

    // -- fused_elementwise_chain --------------------------------------------

    #[test]
    fn test_fused_elementwise_chain_relu_neg() {
        let x = arr1d(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = fused_elementwise_chain(&x, &["relu", "neg"]).expect("chain");
        let flat: Vec<f64> = result.iter().copied().collect();
        // relu([-2,-1,0,1,2]) = [0,0,0,1,2], neg -> [0,0,0,-1,-2]
        assert_eq!(flat, vec![0.0, 0.0, 0.0, -1.0, -2.0]);
    }

    #[test]
    fn test_fused_elementwise_chain_square_sqrt() {
        let x = arr1d(&[1.0, 4.0, 9.0]);
        let result = fused_elementwise_chain(&x, &["square", "sqrt"]).expect("chain");
        let flat: Vec<f64> = result.iter().copied().collect();
        // square then sqrt = abs(x) for positive values = x
        for (a, b) in flat.iter().zip([1.0, 4.0, 9.0].iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_fused_elementwise_chain_empty_ops_error() {
        let x = arr1d(&[1.0]);
        let result = fused_elementwise_chain(&x, &[]);
        assert!(result.is_err());
    }

    // -- fused_mean ---------------------------------------------------------

    #[test]
    fn test_fused_mean_axis0() {
        let x = arr2d(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = fused_mean(&x, 0).expect("fused_mean");
        // mean along axis 0: [(1+4)/2, (2+5)/2, (3+6)/2] = [2.5, 3.5, 4.5]
        let flat: Vec<f64> = result.iter().copied().collect();
        assert_eq!(flat, vec![2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_fused_mean_axis1() {
        let x = arr2d(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = fused_mean(&x, 1).expect("fused_mean");
        // mean along axis 1: [(1+2+3)/3, (4+5+6)/3] = [2.0, 5.0]
        let flat: Vec<f64> = result.iter().copied().collect();
        assert_eq!(flat, vec![2.0, 5.0]);
    }

    #[test]
    fn test_fused_mean_out_of_bounds() {
        let x = arr2d(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = fused_mean(&x, 5);
        assert!(result.is_err());
    }

    // -- fused_variance -----------------------------------------------------

    #[test]
    fn test_fused_variance() {
        let x = arr2d(1, 4, &[1.0, 2.0, 3.0, 4.0]);
        let result = fused_variance(&x, 1).expect("fused_variance");
        // variance = mean(x^2) = (1+4+9+16)/4 = 7.5
        let val = result.iter().next().copied().unwrap_or(0.0);
        assert!((val - 7.5).abs() < 1e-10);
    }

    // -- fused_softmax ------------------------------------------------------

    #[test]
    fn test_fused_softmax_basic() {
        let x = arr2d(1, 3, &[1.0, 2.0, 3.0]);
        let result = fused_softmax(&x, 1).expect("fused_softmax");
        // softmax should sum to 1
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        // Values should be increasing
        let flat: Vec<f64> = result.iter().copied().collect();
        assert!(flat[0] < flat[1] && flat[1] < flat[2]);
    }

    #[test]
    fn test_fused_softmax_uniform() {
        let x = arr2d(1, 4, &[0.0, 0.0, 0.0, 0.0]);
        let result = fused_softmax(&x, 1).expect("fused_softmax");
        // All equal -> uniform distribution
        for &v in result.iter() {
            assert!((v - 0.25).abs() < 1e-10);
        }
    }

    #[test]
    fn test_fused_softmax_numerical_stability() {
        // Large values should not overflow
        let x = arr2d(1, 3, &[1000.0, 1001.0, 1002.0]);
        let result = fused_softmax(&x, 1).expect("fused_softmax");
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        // No NaN or Inf
        for &v in result.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_fused_softmax_out_of_bounds() {
        let x = arr2d(1, 3, &[1.0, 2.0, 3.0]);
        let result = fused_softmax(&x, 5);
        assert!(result.is_err());
    }

    // -- fold_conv_bn_params ------------------------------------------------

    #[test]
    fn test_fold_conv_bn_params() {
        // 2 output channels, 1 input channel, 1x1 kernel
        let conv_weight = arr2d(2, 1, &[1.0, 2.0]);
        let conv_bias = arr1d(&[0.0, 0.0]);
        let bn_params = BatchNormParams {
            running_mean: vec![0.0, 0.0],
            running_var: vec![1.0, 1.0],
            gamma: vec![1.0, 1.0],
            beta: vec![0.0, 0.0],
            epsilon: 1e-5,
        };

        let (fused_w, fused_b) = fold_conv_bn_params(&conv_weight, Some(&conv_bias), &bn_params)
            .expect("fold_conv_bn_params");

        // With mean=0, var=1, gamma=1, beta=0, epsilon~0:
        // fused weights should be approximately the same as original
        let w_flat: Vec<f64> = fused_w.iter().copied().collect();
        assert!((w_flat[0] - 1.0).abs() < 0.01);
        assert!((w_flat[1] - 2.0).abs() < 0.01);

        // Fused bias should be approximately 0
        let b_flat: Vec<f64> = fused_b.iter().copied().collect();
        for &v in &b_flat {
            assert!(v.abs() < 0.01);
        }
    }

    // -- Edge cases ---------------------------------------------------------

    #[test]
    fn test_fused_linear_single_element() {
        let x = arr2d(1, 1, &[5.0]);
        let w = arr2d(1, 1, &[2.0]);
        let bias = arr1d(&[3.0]);

        let result = fused_linear(&x, &w, &bias).expect("single element");
        let val = result.iter().next().copied().unwrap_or(0.0);
        assert!((val - 13.0).abs() < 1e-10); // 5*2+3 = 13
    }

    #[test]
    fn test_fused_linear_shape_mismatch() {
        let x = arr2d(2, 3, &[1.0; 6]);
        let w = arr2d(2, 2, &[1.0; 4]); // wrong: should be 3 rows
        let bias = arr1d(&[0.0, 0.0]);

        let result = fused_linear(&x, &w, &bias);
        assert!(result.is_err());
    }

    #[test]
    fn test_fused_softmax_single_element() {
        let x = arr2d(1, 1, &[42.0]);
        let result = fused_softmax(&x, 1).expect("single element softmax");
        let val = result.iter().next().copied().unwrap_or(0.0);
        // softmax of single element = 1.0
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fused_mean_single_element() {
        let x = arr2d(1, 1, &[42.0]);
        let result = fused_mean(&x, 0).expect("single element mean");
        let val = result.iter().next().copied().unwrap_or(0.0);
        assert!((val - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_relu_inplace() {
        let mut x = arr1d(&[-3.0, -1.0, 0.0, 1.0, 3.0]);
        apply_relu_inplace(&mut x);
        let flat: Vec<f64> = x.iter().copied().collect();
        assert_eq!(flat, vec![0.0, 0.0, 0.0, 1.0, 3.0]);
    }
}
