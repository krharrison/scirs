//! Standalone gradient clipping utilities for neural network training
//!
//! Provides multiple gradient clipping strategies that can be used independently
//! of any particular training loop or optimizer:
//!
//! - **Norm clipping** (`clip_grad_norm`): Clips gradients by their global L2 norm
//! - **Value clipping** (`clip_grad_value`): Clips each gradient element to a range
//! - **Adaptive clipping** (`clip_grad_adaptive`): Clips using per-parameter percentile-based thresholds
//! - **AGC** (`clip_grad_agc`): Adaptive Gradient Clipping (Brock et al., 2021)
//!
//! # Example
//!
//! ```rust
//! use scirs2_neural::training::gradient_clipping::{clip_grad_norm, clip_grad_value, ClipNormType, GradientClipResult};
//! use scirs2_core::ndarray::array;
//!
//! let mut grads = vec![array![1.0_f64, 2.0, 3.0].into_dyn()];
//! let result = clip_grad_norm(&mut grads, 1.0, ClipNormType::L2).expect("clip failed");
//! println!("Global norm before clipping: {}", result.original_norm);
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{ArrayD, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign, ToPrimitive};
use std::fmt::Debug;

// ============================================================================
// Types
// ============================================================================

/// Type of norm used for gradient clipping
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipNormType {
    /// L1 norm (sum of absolute values)
    L1,
    /// L2 norm (Euclidean norm) -- the most common choice
    L2,
    /// L-infinity norm (maximum absolute value)
    LInf,
}

/// Result of a gradient clipping operation
#[derive(Debug, Clone)]
pub struct GradientClipResult {
    /// The norm of the gradients before clipping
    pub original_norm: f64,
    /// The norm of the gradients after clipping
    pub clipped_norm: f64,
    /// Whether any clipping was actually applied
    pub was_clipped: bool,
    /// Number of individual gradient tensors processed
    pub num_tensors: usize,
    /// Total number of parameters across all tensors
    pub total_params: usize,
}

/// Configuration for adaptive gradient clipping (AGC)
///
/// Based on "High-Performance Large-Scale Image Recognition Without Normalization"
/// (Brock et al., 2021). AGC clips gradients based on the ratio of gradient norm
/// to parameter norm, which is more robust than fixed-threshold clipping.
#[derive(Debug, Clone)]
pub struct AdaptiveGradClipConfig {
    /// Clipping factor lambda. Gradients are clipped when
    /// `||grad|| / max(||param||, eps) > lambda`
    pub clip_factor: f64,
    /// Small epsilon to avoid division by zero in parameter norm
    pub eps: f64,
}

impl Default for AdaptiveGradClipConfig {
    fn default() -> Self {
        Self {
            clip_factor: 0.01,
            eps: 1e-3,
        }
    }
}

// ============================================================================
// Core clipping functions
// ============================================================================

/// Clip gradients by their global norm (in-place).
///
/// Computes the global norm of all gradient tensors and, if it exceeds `max_norm`,
/// rescales every gradient element by `max_norm / global_norm`.
///
/// This is the most commonly used gradient clipping strategy (used in PyTorch's
/// `torch.nn.utils.clip_grad_norm_`).
///
/// # Arguments
///
/// * `gradients` - Mutable slice of gradient tensors to clip in-place
/// * `max_norm` - Maximum allowed norm
/// * `norm_type` - Which norm to use (L1, L2, or LInf)
///
/// # Returns
///
/// A `GradientClipResult` with the original and clipped norms.
pub fn clip_grad_norm<F>(
    gradients: &mut [ArrayD<F>],
    max_norm: f64,
    norm_type: ClipNormType,
) -> Result<GradientClipResult>
where
    F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + ToPrimitive,
{
    if max_norm < 0.0 {
        return Err(NeuralError::InvalidArgument(
            "max_norm must be non-negative".to_string(),
        ));
    }

    let total_params: usize = gradients.iter().map(|g| g.len()).sum();

    // Compute the global norm depending on norm_type
    let global_norm = compute_global_norm(gradients, norm_type)?;

    let was_clipped = global_norm > max_norm && max_norm > 0.0;

    if was_clipped {
        let clip_coef = max_norm / (global_norm + 1e-6);
        let clip_coef_f = F::from(clip_coef).ok_or_else(|| {
            NeuralError::ComputationError("Failed to convert clip coefficient".to_string())
        })?;

        for grad in gradients.iter_mut() {
            grad.mapv_inplace(|v| v * clip_coef_f);
        }
    }

    let clipped_norm = if was_clipped {
        compute_global_norm(gradients, norm_type).unwrap_or(global_norm)
    } else {
        global_norm
    };

    Ok(GradientClipResult {
        original_norm: global_norm,
        clipped_norm,
        was_clipped,
        num_tensors: gradients.len(),
        total_params,
    })
}

/// Clip gradients element-wise by value (in-place).
///
/// Each gradient element is clamped to the range `[-clip_value, clip_value]`.
/// This is a simpler but sometimes less effective strategy than norm clipping.
///
/// # Arguments
///
/// * `gradients` - Mutable slice of gradient tensors to clip in-place
/// * `clip_value` - Maximum absolute value for any gradient element
///
/// # Returns
///
/// A `GradientClipResult` with norm information (using L-inf norm for
/// consistency with the value-clipping semantics).
pub fn clip_grad_value<F>(
    gradients: &mut [ArrayD<F>],
    clip_value: f64,
) -> Result<GradientClipResult>
where
    F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + ToPrimitive,
{
    if clip_value < 0.0 {
        return Err(NeuralError::InvalidArgument(
            "clip_value must be non-negative".to_string(),
        ));
    }

    let total_params: usize = gradients.iter().map(|g| g.len()).sum();
    let original_norm = compute_global_norm(gradients, ClipNormType::LInf)?;

    let clip_f = F::from(clip_value)
        .ok_or_else(|| NeuralError::ComputationError("Failed to convert clip_value".to_string()))?;
    let neg_clip_f = F::from(-clip_value).ok_or_else(|| {
        NeuralError::ComputationError("Failed to convert negative clip_value".to_string())
    })?;

    let mut was_clipped = false;
    for grad in gradients.iter_mut() {
        grad.mapv_inplace(|v| {
            if v > clip_f {
                was_clipped = true;
                clip_f
            } else if v < neg_clip_f {
                was_clipped = true;
                neg_clip_f
            } else {
                v
            }
        });
    }

    let clipped_norm = compute_global_norm(gradients, ClipNormType::LInf)?;

    Ok(GradientClipResult {
        original_norm,
        clipped_norm,
        was_clipped,
        num_tensors: gradients.len(),
        total_params,
    })
}

/// Adaptive Gradient Clipping (AGC) as described in Brock et al. (2021).
///
/// For each parameter tensor, clips gradients based on the ratio of the gradient
/// norm to the parameter norm:
///
///   if `||grad_i|| / max(||param_i||, eps) > lambda` then
///       `grad_i *= lambda * max(||param_i||, eps) / ||grad_i||`
///
/// This approach is more robust than fixed-threshold clipping because it adapts
/// to the scale of each parameter.
///
/// # Arguments
///
/// * `gradients` - Mutable slice of gradient tensors
/// * `parameters` - Corresponding parameter tensors (same length as gradients)
/// * `config` - AGC configuration (clip_factor, eps)
///
/// # Returns
///
/// A `GradientClipResult` with the maximum original gradient norm and post-clip norm.
pub fn clip_grad_agc<F>(
    gradients: &mut [ArrayD<F>],
    parameters: &[ArrayD<F>],
    config: &AdaptiveGradClipConfig,
) -> Result<GradientClipResult>
where
    F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + ToPrimitive,
{
    if gradients.len() != parameters.len() {
        return Err(NeuralError::InvalidArgument(format!(
            "gradients length ({}) must match parameters length ({})",
            gradients.len(),
            parameters.len()
        )));
    }

    let total_params: usize = gradients.iter().map(|g| g.len()).sum();
    let original_norm = compute_global_norm(gradients, ClipNormType::L2)?;
    let mut any_clipped = false;

    for (grad, param) in gradients.iter_mut().zip(parameters.iter()) {
        let param_norm = tensor_l2_norm(param)?;
        let grad_norm = tensor_l2_norm(grad)?;

        let effective_param_norm = param_norm.max(config.eps);

        if grad_norm > config.clip_factor * effective_param_norm && grad_norm > 0.0 {
            let scale = config.clip_factor * effective_param_norm / grad_norm;
            let scale_f = F::from(scale).ok_or_else(|| {
                NeuralError::ComputationError("Failed to convert AGC scale".to_string())
            })?;
            grad.mapv_inplace(|v| v * scale_f);
            any_clipped = true;
        }
    }

    let clipped_norm = compute_global_norm(gradients, ClipNormType::L2)?;

    Ok(GradientClipResult {
        original_norm,
        clipped_norm,
        was_clipped: any_clipped,
        num_tensors: gradients.len(),
        total_params,
    })
}

/// Clip gradients using a per-parameter percentile-based adaptive threshold.
///
/// For each gradient tensor, computes the specified percentile of the absolute
/// gradient values and clips to that threshold. This is useful when gradient
/// distributions have heavy tails (e.g., in transformers).
///
/// # Arguments
///
/// * `gradients` - Mutable slice of gradient tensors
/// * `percentile` - Percentile to use for threshold (0.0 to 100.0). For example,
///   99.0 means the top 1% of gradient magnitudes are clipped.
///
/// # Returns
///
/// A `GradientClipResult` with norm information.
pub fn clip_grad_adaptive<F>(
    gradients: &mut [ArrayD<F>],
    percentile: f64,
) -> Result<GradientClipResult>
where
    F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + ToPrimitive,
{
    if !(0.0..=100.0).contains(&percentile) {
        return Err(NeuralError::InvalidArgument(
            "percentile must be between 0.0 and 100.0".to_string(),
        ));
    }

    let total_params: usize = gradients.iter().map(|g| g.len()).sum();
    let original_norm = compute_global_norm(gradients, ClipNormType::L2)?;
    let mut any_clipped = false;

    for grad in gradients.iter_mut() {
        if grad.is_empty() {
            continue;
        }

        // Collect absolute values and sort to find the percentile
        let mut abs_vals: Vec<f64> = grad.iter().filter_map(|v| v.abs().to_f64()).collect();
        abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if abs_vals.is_empty() {
            continue;
        }

        // Compute the percentile index
        let idx = ((percentile / 100.0) * (abs_vals.len() as f64 - 1.0))
            .round()
            .max(0.0) as usize;
        let idx = idx.min(abs_vals.len() - 1);
        let threshold = abs_vals[idx];

        if threshold <= 0.0 {
            continue;
        }

        let clip_f = F::from(threshold).ok_or_else(|| {
            NeuralError::ComputationError("Failed to convert percentile threshold".to_string())
        })?;
        let neg_clip_f = F::from(-threshold).ok_or_else(|| {
            NeuralError::ComputationError(
                "Failed to convert negative percentile threshold".to_string(),
            )
        })?;

        grad.mapv_inplace(|v| {
            if v > clip_f {
                any_clipped = true;
                clip_f
            } else if v < neg_clip_f {
                any_clipped = true;
                neg_clip_f
            } else {
                v
            }
        });
    }

    let clipped_norm = compute_global_norm(gradients, ClipNormType::L2)?;

    Ok(GradientClipResult {
        original_norm,
        clipped_norm,
        was_clipped: any_clipped,
        num_tensors: gradients.len(),
        total_params,
    })
}

// ============================================================================
// Helper functions
// ============================================================================

/// Compute the L2 norm of a single tensor.
fn tensor_l2_norm<F>(tensor: &ArrayD<F>) -> Result<f64>
where
    F: Float + Debug + ScalarOperand + ToPrimitive,
{
    let mut sum_sq = 0.0_f64;
    for val in tensor.iter() {
        let f = val.to_f64().unwrap_or(0.0);
        sum_sq += f * f;
    }
    Ok(sum_sq.sqrt())
}

/// Compute the global norm of a collection of gradient tensors.
fn compute_global_norm<F>(gradients: &[ArrayD<F>], norm_type: ClipNormType) -> Result<f64>
where
    F: Float + Debug + ScalarOperand + ToPrimitive,
{
    if gradients.is_empty() {
        return Ok(0.0);
    }

    match norm_type {
        ClipNormType::L1 => {
            let mut total = 0.0_f64;
            for grad in gradients {
                for val in grad.iter() {
                    total += val.abs().to_f64().unwrap_or(0.0);
                }
            }
            Ok(total)
        }
        ClipNormType::L2 => {
            let mut sum_sq = 0.0_f64;
            for grad in gradients {
                for val in grad.iter() {
                    let f = val.to_f64().unwrap_or(0.0);
                    sum_sq += f * f;
                }
            }
            Ok(sum_sq.sqrt())
        }
        ClipNormType::LInf => {
            let mut max_abs = 0.0_f64;
            for grad in gradients {
                for val in grad.iter() {
                    let abs_val = val.abs().to_f64().unwrap_or(0.0);
                    if abs_val > max_abs {
                        max_abs = abs_val;
                    }
                }
            }
            Ok(max_abs)
        }
    }
}

/// Convenience function: compute the global L2 norm of a set of gradient tensors
/// without modifying them.
pub fn grad_norm<F>(gradients: &[ArrayD<F>], norm_type: ClipNormType) -> Result<f64>
where
    F: Float + Debug + ScalarOperand + ToPrimitive,
{
    compute_global_norm(gradients, norm_type)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    fn make_grad(vals: &[f64]) -> ArrayD<f64> {
        Array::from_vec(vals.to_vec()).into_dyn()
    }

    fn make_grad_2d(rows: usize, cols: usize, vals: &[f64]) -> ArrayD<f64> {
        Array::from_shape_vec(IxDyn(&[rows, cols]), vals.to_vec())
            .expect("test array shape mismatch")
    }

    #[test]
    fn test_clip_grad_norm_l2_clips_when_above_threshold() {
        // gradient = [3, 4] => L2 norm = 5
        let mut grads = vec![make_grad(&[3.0, 4.0])];
        let result = clip_grad_norm(&mut grads, 2.5, ClipNormType::L2)
            .expect("clip_grad_norm should succeed");

        assert!(result.was_clipped);
        assert!((result.original_norm - 5.0).abs() < 1e-10);
        // After clipping, norm should be approximately 2.5
        assert!((result.clipped_norm - 2.5).abs() < 0.1);
    }

    #[test]
    fn test_clip_grad_norm_l2_no_clip_when_below_threshold() {
        let mut grads = vec![make_grad(&[1.0, 1.0])];
        let result = clip_grad_norm(&mut grads, 10.0, ClipNormType::L2)
            .expect("clip_grad_norm should succeed");

        assert!(!result.was_clipped);
        let expected_norm = (2.0_f64).sqrt();
        assert!((result.original_norm - expected_norm).abs() < 1e-10);
        assert!((result.clipped_norm - expected_norm).abs() < 1e-10);
    }

    #[test]
    fn test_clip_grad_norm_l1() {
        // gradient = [3, -4] => L1 norm = 7
        let mut grads = vec![make_grad(&[3.0, -4.0])];
        let result = clip_grad_norm(&mut grads, 3.5, ClipNormType::L1)
            .expect("clip_grad_norm should succeed");

        assert!(result.was_clipped);
        assert!((result.original_norm - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_clip_grad_norm_linf() {
        // gradient = [1, -5, 3] => LInf norm = 5
        let mut grads = vec![make_grad(&[1.0, -5.0, 3.0])];
        let result = clip_grad_norm(&mut grads, 2.0, ClipNormType::LInf)
            .expect("clip_grad_norm should succeed");

        assert!(result.was_clipped);
        assert!((result.original_norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_clip_grad_norm_negative_max_norm_errors() {
        let mut grads = vec![make_grad(&[1.0, 2.0])];
        let result = clip_grad_norm(&mut grads, -1.0, ClipNormType::L2);
        assert!(result.is_err());
    }

    #[test]
    fn test_clip_grad_norm_multiple_tensors() {
        // tensor1: [3, 0], tensor2: [0, 4] => global L2 norm = 5
        let mut grads = vec![make_grad(&[3.0, 0.0]), make_grad(&[0.0, 4.0])];
        let result = clip_grad_norm(&mut grads, 2.5, ClipNormType::L2)
            .expect("clip_grad_norm should succeed");

        assert!(result.was_clipped);
        assert!((result.original_norm - 5.0).abs() < 1e-10);
        assert_eq!(result.num_tensors, 2);
        assert_eq!(result.total_params, 4);
    }

    #[test]
    fn test_clip_grad_value_clips_both_directions() {
        let mut grads = vec![make_grad(&[10.0, -10.0, 0.5, -0.5])];
        let result = clip_grad_value(&mut grads, 1.0).expect("clip_grad_value should succeed");

        assert!(result.was_clipped);
        let vals: Vec<f64> = grads[0].iter().copied().collect();
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - (-1.0)).abs() < 1e-10);
        assert!((vals[2] - 0.5).abs() < 1e-10);
        assert!((vals[3] - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_clip_grad_value_no_clip_needed() {
        let mut grads = vec![make_grad(&[0.1, -0.1, 0.5])];
        let result = clip_grad_value(&mut grads, 1.0).expect("clip_grad_value should succeed");

        assert!(!result.was_clipped);
    }

    #[test]
    fn test_clip_grad_value_negative_clip_errors() {
        let mut grads = vec![make_grad(&[1.0])];
        let result = clip_grad_value(&mut grads, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_clip_grad_agc_basic() {
        // params with norm ~1.41, grads with norm ~7.07
        // ratio = 7.07 / 1.41 = ~5.01, which exceeds clip_factor=0.01
        let mut grads = vec![make_grad(&[5.0, 5.0])];
        let params = vec![make_grad(&[1.0, 1.0])];
        let config = AdaptiveGradClipConfig {
            clip_factor: 0.01,
            eps: 1e-3,
        };
        let result =
            clip_grad_agc(&mut grads, &params, &config).expect("clip_grad_agc should succeed");

        assert!(result.was_clipped);
        assert!(result.clipped_norm < result.original_norm);
    }

    #[test]
    fn test_clip_grad_agc_no_clip_small_grads() {
        let mut grads = vec![make_grad(&[0.001, 0.001])];
        let params = vec![make_grad(&[10.0, 10.0])];
        let config = AdaptiveGradClipConfig {
            clip_factor: 0.1,
            eps: 1e-3,
        };
        let result =
            clip_grad_agc(&mut grads, &params, &config).expect("clip_grad_agc should succeed");

        assert!(!result.was_clipped);
    }

    #[test]
    fn test_clip_grad_agc_mismatched_lengths_errors() {
        let mut grads = vec![make_grad(&[1.0]), make_grad(&[2.0])];
        let params = vec![make_grad(&[1.0])];
        let config = AdaptiveGradClipConfig::default();
        let result = clip_grad_agc(&mut grads, &params, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_clip_grad_adaptive_percentile() {
        // Values: [-100, -50, -1, 0, 1, 50, 100]
        // 90th percentile of absolute values: sorted abs = [0,1,1,50,50,100,100] => ~100
        // So clipping at 90th percentile should clip the extreme values
        let mut grads = vec![make_grad(&[-100.0, -50.0, -1.0, 0.0, 1.0, 50.0, 100.0])];
        let result =
            clip_grad_adaptive(&mut grads, 50.0).expect("clip_grad_adaptive should succeed");

        assert!(result.was_clipped);
    }

    #[test]
    fn test_clip_grad_adaptive_invalid_percentile() {
        let mut grads = vec![make_grad(&[1.0])];
        let result = clip_grad_adaptive(&mut grads, 101.0);
        assert!(result.is_err());

        let result = clip_grad_adaptive(&mut grads, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_grad_norm_computation() {
        let grads = vec![make_grad(&[3.0, 4.0])];
        let l2 = grad_norm(&grads, ClipNormType::L2).expect("grad_norm should succeed");
        assert!((l2 - 5.0).abs() < 1e-10);

        let l1 = grad_norm(&grads, ClipNormType::L1).expect("grad_norm should succeed");
        assert!((l1 - 7.0).abs() < 1e-10);

        let linf = grad_norm(&grads, ClipNormType::LInf).expect("grad_norm should succeed");
        assert!((linf - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_clip_grad_norm_empty_gradients() {
        let mut grads: Vec<ArrayD<f64>> = vec![];
        let result = clip_grad_norm(&mut grads, 1.0, ClipNormType::L2)
            .expect("clip_grad_norm should handle empty");
        assert!(!result.was_clipped);
        assert_eq!(result.num_tensors, 0);
        assert_eq!(result.total_params, 0);
    }

    #[test]
    fn test_clip_grad_norm_2d_tensor() {
        // 2x2 matrix with known norm: sqrt(1+4+9+16) = sqrt(30) ~ 5.477
        let mut grads = vec![make_grad_2d(2, 2, &[1.0, 2.0, 3.0, 4.0])];
        let result = clip_grad_norm(&mut grads, 2.0, ClipNormType::L2)
            .expect("clip_grad_norm should succeed on 2d");

        assert!(result.was_clipped);
        let expected_norm = (30.0_f64).sqrt();
        assert!((result.original_norm - expected_norm).abs() < 1e-10);
    }

    #[test]
    fn test_clip_grad_value_preserves_within_range() {
        let original = vec![0.1, -0.2, 0.3, -0.4, 0.5];
        let mut grads = vec![make_grad(&original)];
        let _ = clip_grad_value(&mut grads, 1.0).expect("clip_grad_value should succeed");

        let vals: Vec<f64> = grads[0].iter().copied().collect();
        for (orig, clipped) in original.iter().zip(vals.iter()) {
            assert!(
                (orig - clipped).abs() < 1e-10,
                "Values within range should be unchanged"
            );
        }
    }
}
