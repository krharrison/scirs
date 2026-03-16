//! Tensor operations for neural network building blocks
//!
//! This module provides common tensor operations needed for building
//! neural networks with scirs2-autograd integration.

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::Debug;

/// Concatenate arrays along a specified axis
///
/// # Arguments
///
/// * `arrays` - Slice of arrays to concatenate
/// * `axis` - Axis along which to concatenate
///
/// # Returns
///
/// Concatenated array
pub fn concat<F: Float + Debug + ScalarOperand + NumAssign>(
    arrays: &[Array<F, IxDyn>],
    axis: usize,
) -> Result<Array<F, IxDyn>> {
    if arrays.is_empty() {
        return Err(NeuralError::ShapeMismatch(
            "Cannot concatenate empty array list".to_string(),
        ));
    }

    if arrays.len() == 1 {
        return Ok(arrays[0].clone());
    }

    // Use ndarray's concatenate
    use scirs2_core::ndarray::Axis;
    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();

    scirs2_core::ndarray::concatenate(Axis(axis), &views)
        .map(|a| a.into_dyn())
        .map_err(|e| NeuralError::ShapeMismatch(format!("Concatenation failed: {}", e)))
}

/// Apply dropout to input tensor
///
/// # Arguments
///
/// * `input` - Input array
/// * `rate` - Dropout probability (0.0 to 1.0)
/// * `training` - Whether in training mode
///
/// # Returns
///
/// Array with dropout applied (in training mode) or unchanged (in inference mode)
pub fn dropout<F: Float + Debug + ScalarOperand + NumAssign>(
    input: &Array<F, IxDyn>,
    rate: f32,
    training: bool,
) -> Result<Array<F, IxDyn>> {
    if !training || rate == 0.0 {
        return Ok(input.clone());
    }

    if !(0.0..1.0).contains(&rate) {
        return Err(NeuralError::InvalidArgument(format!(
            "Dropout rate must be in [0, 1), got {}",
            rate
        )));
    }

    // Use scirs2_core random for dropout mask
    use scirs2_core::random::RngExt;
    let mut rng = scirs2_core::random::rng();

    let mut output = input.clone();
    let scale = F::one()
        / F::from(1.0 - rate).ok_or_else(|| {
            NeuralError::InvalidArgument("Failed to convert scale factor".to_string())
        })?;

    let rate_f = F::from(rate).ok_or_else(|| {
        NeuralError::InvalidArgument("Failed to convert dropout rate".to_string())
    })?;

    // Apply dropout mask
    for elem in output.iter_mut() {
        let random_val = F::from(rng.random::<f32>()).ok_or_else(|| {
            NeuralError::InvalidArgument("Failed to convert random value".to_string())
        })?;
        if random_val < rate_f {
            *elem = F::zero();
        } else {
            *elem *= scale;
        }
    }

    Ok(output)
}

/// Global average pooling over spatial dimensions
///
/// # Arguments
///
/// * `input` - Input array with shape (batch, channels, height, width)
///
/// # Returns
///
/// Array with shape (batch, channels) after global average pooling
pub fn global_avg_pool<F: Float + Debug + ScalarOperand + NumAssign>(
    input: &Array<F, IxDyn>,
) -> Result<Array<F, IxDyn>> {
    if input.ndim() != 4 {
        return Err(NeuralError::ShapeMismatch(format!(
            "Expected 4D input (batch, channels, height, width), got {}D",
            input.ndim()
        )));
    }

    let shape = input.shape();
    let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);

    let mut output = Array::zeros(vec![batch, channels]);
    let spatial_size = F::from(height * width).ok_or_else(|| {
        NeuralError::ComputationError("Failed to compute spatial size".to_string())
    })?;

    for b in 0..batch {
        for c in 0..channels {
            let mut sum = F::zero();
            for h in 0..height {
                for w in 0..width {
                    let idx = [b, c, h, w];
                    if let Some(&val) = input.get(&idx[..]) {
                        sum += val;
                    }
                }
            }
            output[[b, c]] = sum / spatial_size;
        }
    }

    Ok(output.into_dyn())
}

/// Upsample using nearest neighbor interpolation
///
/// # Arguments
///
/// * `input` - Input array with shape (batch, channels, height, width)
/// * `scale_factor` - Upsampling scale factor
///
/// # Returns
///
/// Upsampled array
pub fn upsample_nearest<F: Float + Debug + ScalarOperand + NumAssign + Copy>(
    input: &Array<F, IxDyn>,
    scale_factor: usize,
) -> Result<Array<F, IxDyn>> {
    if input.ndim() != 4 {
        return Err(NeuralError::ShapeMismatch(format!(
            "Expected 4D input, got {}D",
            input.ndim()
        )));
    }

    let shape = input.shape();
    let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);

    let new_height = height * scale_factor;
    let new_width = width * scale_factor;

    let mut output = Array::zeros(vec![batch, channels, new_height, new_width]);

    for b in 0..batch {
        for c in 0..channels {
            for h in 0..new_height {
                for w in 0..new_width {
                    let src_h = h / scale_factor;
                    let src_w = w / scale_factor;
                    let val = input[[b, c, src_h, src_w]];
                    output[[b, c, h, w]] = val;
                }
            }
        }
    }

    Ok(output.into_dyn())
}

/// Upsample using bilinear interpolation
///
/// # Arguments
///
/// * `input` - Input array with shape (batch, channels, height, width)
/// * `scale_factor` - Upsampling scale factor
///
/// # Returns
///
/// Upsampled array with bilinear interpolation
pub fn upsample_bilinear<F: Float + Debug + ScalarOperand + NumAssign + Copy>(
    input: &Array<F, IxDyn>,
    scale_factor: usize,
) -> Result<Array<F, IxDyn>> {
    if input.ndim() != 4 {
        return Err(NeuralError::ShapeMismatch(format!(
            "Expected 4D input, got {}D",
            input.ndim()
        )));
    }

    let shape = input.shape();
    let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);

    let new_height = height * scale_factor;
    let new_width = width * scale_factor;

    let mut output = Array::zeros(vec![batch, channels, new_height, new_width]);

    for b in 0..batch {
        for c in 0..channels {
            for h in 0..new_height {
                for w in 0..new_width {
                    // Compute source coordinates
                    let src_h = (h as f64) / (scale_factor as f64);
                    let src_w = (w as f64) / (scale_factor as f64);

                    let h0 = src_h.floor() as usize;
                    let w0 = src_w.floor() as usize;
                    let h1 = (h0 + 1).min(height - 1);
                    let w1 = (w0 + 1).min(width - 1);

                    let dh = F::from(src_h - h0 as f64).ok_or_else(|| {
                        NeuralError::ComputationError("Failed to convert dh".to_string())
                    })?;
                    let dw = F::from(src_w - w0 as f64).ok_or_else(|| {
                        NeuralError::ComputationError("Failed to convert dw".to_string())
                    })?;

                    // Bilinear interpolation
                    let v00 = input[[b, c, h0, w0]];
                    let v01 = input[[b, c, h0, w1]];
                    let v10 = input[[b, c, h1, w0]];
                    let v11 = input[[b, c, h1, w1]];

                    let one = F::one();
                    let v0 = v00 * (one - dw) + v01 * dw;
                    let v1 = v10 * (one - dw) + v11 * dw;
                    let val = v0 * (one - dh) + v1 * dh;

                    output[[b, c, h, w]] = val;
                }
            }
        }
    }

    Ok(output.into_dyn())
}

/// Max pooling 2D
///
/// # Arguments
///
/// * `input` - Input array with shape (batch, channels, height, width)
/// * `kernel_size` - Pooling kernel size
/// * `stride` - Pooling stride
/// * `padding` - Padding size
///
/// # Returns
///
/// Max pooled array
pub fn max_pool2d<F: Float + Debug + ScalarOperand + NumAssign + Copy>(
    input: &Array<F, IxDyn>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Result<Array<F, IxDyn>> {
    if input.ndim() != 4 {
        return Err(NeuralError::ShapeMismatch(format!(
            "Expected 4D input, got {}D",
            input.ndim()
        )));
    }

    let shape = input.shape();
    let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);

    // Compute output dimensions
    let out_height = (height + 2 * padding - kernel_size) / stride + 1;
    let out_width = (width + 2 * padding - kernel_size) / stride + 1;

    let mut output = Array::from_elem(
        vec![batch, channels, out_height, out_width],
        F::neg_infinity(),
    );

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let mut max_val = F::neg_infinity();

                    for kh in 0..kernel_size {
                        for kw in 0..kernel_size {
                            let ih = oh * stride + kh;
                            let iw = ow * stride + kw;

                            if ih >= padding && iw >= padding {
                                let ih = ih - padding;
                                let iw = iw - padding;

                                if ih < height && iw < width {
                                    let val = input[[b, c, ih, iw]];
                                    if val > max_val {
                                        max_val = val;
                                    }
                                }
                            }
                        }
                    }

                    output[[b, c, oh, ow]] = max_val;
                }
            }
        }
    }

    Ok(output.into_dyn())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array4;

    #[test]
    fn test_concat() {
        let a = Array::from_shape_vec(vec![2, 3], vec![1.0f32; 6])
            .expect("Operation failed")
            .into_dyn();
        let b = Array::from_shape_vec(vec![2, 3], vec![2.0f32; 6])
            .expect("Operation failed")
            .into_dyn();

        let result = concat(&[a, b], 0).expect("Concat failed");
        assert_eq!(result.shape(), &[4, 3]);
    }

    #[test]
    fn test_global_avg_pool() {
        let input = Array4::<f32>::from_elem((2, 3, 4, 4), 2.0).into_dyn();
        let result = global_avg_pool(&input).expect("Global avg pool failed");

        assert_eq!(result.shape(), &[2, 3]);
        assert!((result[[0, 0]] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_upsample_nearest() {
        let input = Array4::<f32>::from_elem((1, 1, 2, 2), 1.0).into_dyn();
        let result = upsample_nearest(&input, 2).expect("Upsample failed");

        assert_eq!(result.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_max_pool2d() {
        let mut input = Array4::<f32>::zeros((1, 1, 4, 4));
        input[[0, 0, 1, 1]] = 5.0;
        let input = input.into_dyn();

        let result = max_pool2d(&input, 2, 2, 0).expect("Max pool failed");

        assert_eq!(result.shape(), &[1, 1, 2, 2]);
        assert_eq!(result[[0, 0, 0, 0]], 5.0);
    }
}
