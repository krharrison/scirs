//! Gradient utilities: HashMap-based gradient clipping and accumulation helpers
//!
//! This module provides gradient manipulation functions that operate on
//! `HashMap<String, ArrayD<f64>>` parameter maps, which is the natural
//! representation when working with named model parameters.
//!
//! ## Functions
//!
//! - [`clip_grad_norm_map`]: Clip by global L2 norm (in-place, named params)
//! - [`clip_grad_value_map`]: Clip element-wise by value (in-place, named params)
//! - [`grad_norm_map`]: Compute global L2 norm without modification
//!
//! ## Types
//!
//! - [`GradientAccumulatorMap`]: Accumulate gradients over multiple mini-batches
//!   and return averaged gradients when ready.
//!
//! # Example
//!
//! ```rust
//! use scirs2_neural::training::gradient_utils::{
//!     clip_grad_norm_map, clip_grad_value_map, GradientAccumulatorMap,
//! };
//! use scirs2_core::ndarray::{Array, IxDyn};
//! use std::collections::HashMap;
//!
//! // Build a named gradient map
//! let mut grads: HashMap<String, Array<f64, IxDyn>> = HashMap::new();
//! grads.insert("layer1.weight".to_string(), Array::from_vec(vec![3.0_f64, 4.0]).into_dyn());
//! grads.insert("layer1.bias".to_string(), Array::from_vec(vec![0.0_f64]).into_dyn());
//!
//! // Clip by norm (global L2 norm = 5.0, clipped to 2.5)
//! let original_norm = clip_grad_norm_map(&mut grads, 2.5).expect("clip failed");
//! assert!((original_norm - 5.0).abs() < 1e-6);
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array, ArrayD, IxDyn};
use std::collections::HashMap;

// ============================================================================
// Public gradient clipping on HashMap<String, ArrayD<f64>>
// ============================================================================

/// Clip named gradients by their global L2 norm (in-place).
///
/// Computes the global L2 norm across all gradient tensors in the map. If it
/// exceeds `max_norm`, every gradient element is rescaled by `max_norm / norm`.
///
/// # Arguments
///
/// * `grads` - Mutable map from parameter name to gradient tensor
/// * `max_norm` - Maximum allowed global L2 norm
///
/// # Returns
///
/// The original (pre-clipping) global L2 norm.
///
/// # Errors
///
/// Returns an error if `max_norm` is negative.
///
/// # Example
///
/// ```rust
/// use scirs2_neural::training::gradient_utils::clip_grad_norm_map;
/// use scirs2_core::ndarray::Array;
/// use std::collections::HashMap;
///
/// let mut grads = HashMap::new();
/// grads.insert("w".to_string(), Array::from_vec(vec![3.0_f64, 4.0]).into_dyn());
/// let orig = clip_grad_norm_map(&mut grads, 2.5).expect("operation should succeed");
/// assert!((orig - 5.0).abs() < 1e-6);
/// ```
pub fn clip_grad_norm_map(
    grads: &mut HashMap<String, ArrayD<f64>>,
    max_norm: f64,
) -> Result<f64> {
    if max_norm < 0.0 {
        return Err(NeuralError::InvalidArgument(
            "max_norm must be non-negative".to_string(),
        ));
    }

    // Compute global L2 norm
    let mut sum_sq = 0.0_f64;
    for tensor in grads.values() {
        for &v in tensor.iter() {
            sum_sq += v * v;
        }
    }
    let global_norm = sum_sq.sqrt();

    // Clip if needed
    if global_norm > max_norm && max_norm > 0.0 {
        let clip_coef = max_norm / (global_norm + 1e-6);
        for tensor in grads.values_mut() {
            tensor.mapv_inplace(|v| v * clip_coef);
        }
    }

    Ok(global_norm)
}

/// Clip named gradients element-wise to `[-clip_value, clip_value]` (in-place).
///
/// Each gradient element is clamped independently. This is simpler than norm
/// clipping but can distort gradient directions.
///
/// # Arguments
///
/// * `grads` - Mutable map from parameter name to gradient tensor
/// * `clip_value` - Maximum absolute value for any gradient element
///
/// # Errors
///
/// Returns an error if `clip_value` is negative.
///
/// # Example
///
/// ```rust
/// use scirs2_neural::training::gradient_utils::clip_grad_value_map;
/// use scirs2_core::ndarray::Array;
/// use std::collections::HashMap;
///
/// let mut grads = HashMap::new();
/// grads.insert("w".to_string(), Array::from_vec(vec![10.0_f64, -10.0, 0.5]).into_dyn());
/// clip_grad_value_map(&mut grads, 1.0).expect("operation should succeed");
/// let vals: Vec<f64> = grads["w"].iter().copied().collect();
/// assert!((vals[0] - 1.0).abs() < 1e-10);
/// assert!((vals[1] - (-1.0)).abs() < 1e-10);
/// assert!((vals[2] - 0.5).abs() < 1e-10);
/// ```
pub fn clip_grad_value_map(
    grads: &mut HashMap<String, ArrayD<f64>>,
    clip_value: f64,
) -> Result<()> {
    if clip_value < 0.0 {
        return Err(NeuralError::InvalidArgument(
            "clip_value must be non-negative".to_string(),
        ));
    }

    for tensor in grads.values_mut() {
        tensor.mapv_inplace(|v| v.clamp(-clip_value, clip_value));
    }
    Ok(())
}

/// Compute the global L2 norm of all named gradients without modifying them.
///
/// # Arguments
///
/// * `grads` - Map from parameter name to gradient tensor
///
/// # Returns
///
/// The global L2 norm (sqrt of sum of squared elements across all tensors).
///
/// # Example
///
/// ```rust
/// use scirs2_neural::training::gradient_utils::grad_norm_map;
/// use scirs2_core::ndarray::Array;
/// use std::collections::HashMap;
///
/// let mut grads = HashMap::new();
/// grads.insert("w".to_string(), Array::from_vec(vec![3.0_f64, 4.0]).into_dyn());
/// let norm = grad_norm_map(&grads);
/// assert!((norm - 5.0).abs() < 1e-10);
/// ```
pub fn grad_norm_map(grads: &HashMap<String, ArrayD<f64>>) -> f64 {
    let mut sum_sq = 0.0_f64;
    for tensor in grads.values() {
        for &v in tensor.iter() {
            sum_sq += v * v;
        }
    }
    sum_sq.sqrt()
}

// ============================================================================
// GradientAccumulatorMap -- HashMap-based gradient accumulation
// ============================================================================

/// Accumulate gradients over multiple mini-batches and average them.
///
/// This is the HashMap-based companion to the slice-based accumulator,
/// designed for use with named parameter dictionaries. After
/// `accumulation_steps` calls to [`accumulate`](GradientAccumulatorMap::accumulate),
/// [`should_update`](GradientAccumulatorMap::should_update) returns `true` and
/// [`get_averaged_grads`](GradientAccumulatorMap::get_averaged_grads) returns the
/// mean gradient map.
///
/// # Example
///
/// ```rust
/// use scirs2_neural::training::gradient_utils::GradientAccumulatorMap;
/// use scirs2_core::ndarray::Array;
/// use std::collections::HashMap;
///
/// let mut accum = GradientAccumulatorMap::new(4);
///
/// for step in 0..4 {
///     let mut grads = HashMap::new();
///     grads.insert("w".to_string(), Array::from_vec(vec![1.0_f64, 2.0]).into_dyn());
///     accum.accumulate(&grads).expect("operation should succeed");
///     if accum.should_update() {
///         let avg = accum.get_averaged_grads().expect("operation should succeed");
///         assert!((avg["w"][[0]] - 1.0).abs() < 1e-10);
///         accum.reset();
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct GradientAccumulatorMap {
    /// Number of mini-batches to accumulate before averaging
    accumulation_steps: usize,
    /// Current number of accumulated mini-batches
    current_step: usize,
    /// Running sum of gradients for each named parameter
    accumulated_grads: HashMap<String, ArrayD<f64>>,
}

impl GradientAccumulatorMap {
    /// Create a new accumulator that averages over `steps` mini-batches.
    ///
    /// # Panics
    ///
    /// Does not panic; returns an error from `accumulate` if called with
    /// inconsistent gradient maps.
    pub fn new(steps: usize) -> Self {
        Self {
            accumulation_steps: steps.max(1),
            current_step: 0,
            accumulated_grads: HashMap::new(),
        }
    }

    /// Accumulate one mini-batch of gradients.
    ///
    /// On the first call, the shapes are recorded. Subsequent calls must
    /// provide the same parameter names and shapes.
    ///
    /// # Arguments
    ///
    /// * `grads` - Gradient map for the current mini-batch
    ///
    /// # Errors
    ///
    /// Returns an error if a parameter's shape is inconsistent with the
    /// accumulated shape.
    pub fn accumulate(&mut self, grads: &HashMap<String, ArrayD<f64>>) -> Result<()> {
        for (name, grad) in grads {
            match self.accumulated_grads.get_mut(name) {
                Some(acc) => {
                    if acc.shape() != grad.shape() {
                        return Err(NeuralError::InvalidArgument(format!(
                            "Shape mismatch for parameter '{}': accumulated {:?} vs new {:?}",
                            name,
                            acc.shape(),
                            grad.shape()
                        )));
                    }
                    // Add in-place
                    acc.zip_mut_with(grad, |a, &b| *a += b);
                }
                None => {
                    self.accumulated_grads.insert(name.clone(), grad.clone());
                }
            }
        }
        self.current_step += 1;
        Ok(())
    }

    /// Returns `true` when enough mini-batches have been accumulated.
    pub fn should_update(&self) -> bool {
        self.current_step >= self.accumulation_steps
    }

    /// Return the averaged gradient map.
    ///
    /// Divides each accumulated gradient by the number of accumulated steps.
    ///
    /// # Errors
    ///
    /// Returns an error if no gradients have been accumulated yet.
    pub fn get_averaged_grads(&self) -> Result<HashMap<String, ArrayD<f64>>> {
        if self.current_step == 0 {
            return Err(NeuralError::InvalidArgument(
                "No gradients have been accumulated yet".to_string(),
            ));
        }

        let scale = 1.0 / self.current_step as f64;
        let averaged = self
            .accumulated_grads
            .iter()
            .map(|(name, acc)| {
                let avg = acc.mapv(|v| v * scale);
                (name.clone(), avg)
            })
            .collect();

        Ok(averaged)
    }

    /// Reset the accumulator, clearing all accumulated gradients and step counter.
    pub fn reset(&mut self) {
        self.current_step = 0;
        self.accumulated_grads.clear();
    }

    /// Current number of accumulated mini-batches.
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Configured number of accumulation steps.
    pub fn accumulation_steps(&self) -> usize {
        self.accumulation_steps
    }

    /// Names of parameters currently accumulated.
    pub fn param_names(&self) -> impl Iterator<Item = &str> {
        self.accumulated_grads.keys().map(|s| s.as_str())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    fn make_grad_map(names_vals: &[(&str, Vec<f64>)]) -> HashMap<String, ArrayD<f64>> {
        names_vals
            .iter()
            .map(|(name, vals)| {
                (
                    name.to_string(),
                    Array::from_vec(vals.clone()).into_dyn(),
                )
            })
            .collect()
    }

    // ---- clip_grad_norm_map ----

    #[test]
    fn test_clip_grad_norm_map_clips_above_threshold() {
        // [3, 0] and [0, 4] => global norm = 5
        let mut grads = make_grad_map(&[("w1", vec![3.0, 0.0]), ("w2", vec![0.0, 4.0])]);
        let orig = clip_grad_norm_map(&mut grads, 2.5).expect("failed to create orig");
        assert!((orig - 5.0).abs() < 1e-6);
        // After clipping, norm should be ~2.5
        let clipped = grad_norm_map(&grads);
        assert!((clipped - 2.5).abs() < 0.1);
    }

    #[test]
    fn test_clip_grad_norm_map_no_clip_below_threshold() {
        let mut grads = make_grad_map(&[("w", vec![1.0, 1.0])]);
        let orig = clip_grad_norm_map(&mut grads, 100.0).expect("failed to create orig");
        let expected = 2.0_f64.sqrt();
        assert!((orig - expected).abs() < 1e-10);
        // Values should be unchanged
        let vals: Vec<f64> = grads["w"].iter().copied().collect();
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_clip_grad_norm_map_negative_max_norm_errors() {
        let mut grads = make_grad_map(&[("w", vec![1.0])]);
        assert!(clip_grad_norm_map(&mut grads, -1.0).is_err());
    }

    #[test]
    fn test_clip_grad_norm_map_empty_map() {
        let mut grads: HashMap<String, ArrayD<f64>> = HashMap::new();
        let orig = clip_grad_norm_map(&mut grads, 1.0).expect("failed to create orig");
        assert!((orig - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_clip_grad_norm_map_zero_max_norm() {
        // max_norm = 0 should not clip (treated as "no limit" or keep as-is)
        let mut grads = make_grad_map(&[("w", vec![3.0, 4.0])]);
        let orig = clip_grad_norm_map(&mut grads, 0.0).expect("failed to create orig");
        assert!((orig - 5.0).abs() < 1e-6);
        // Values unchanged because the condition is `global_norm > max_norm && max_norm > 0.0`
        let vals: Vec<f64> = grads["w"].iter().copied().collect();
        assert!((vals[0] - 3.0).abs() < 1e-10);
    }

    // ---- clip_grad_value_map ----

    #[test]
    fn test_clip_grad_value_map_clips_both_directions() {
        let mut grads = make_grad_map(&[("w", vec![10.0, -10.0, 0.5, -0.5])]);
        clip_grad_value_map(&mut grads, 1.0).expect("unexpected None or Err");
        let vals: Vec<f64> = grads["w"].iter().copied().collect();
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - (-1.0)).abs() < 1e-10);
        assert!((vals[2] - 0.5).abs() < 1e-10);
        assert!((vals[3] - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_clip_grad_value_map_no_op_within_range() {
        let mut grads = make_grad_map(&[("w", vec![0.1, -0.2, 0.3])]);
        clip_grad_value_map(&mut grads, 1.0).expect("unexpected None or Err");
        let vals: Vec<f64> = grads["w"].iter().copied().collect();
        assert!((vals[0] - 0.1).abs() < 1e-10);
        assert!((vals[1] - (-0.2)).abs() < 1e-10);
        assert!((vals[2] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_clip_grad_value_map_negative_clip_errors() {
        let mut grads = make_grad_map(&[("w", vec![1.0])]);
        assert!(clip_grad_value_map(&mut grads, -1.0).is_err());
    }

    #[test]
    fn test_clip_grad_value_map_zero_clip() {
        // Clips everything to zero
        let mut grads = make_grad_map(&[("w", vec![5.0, -3.0, 0.0])]);
        clip_grad_value_map(&mut grads, 0.0).expect("unexpected None or Err");
        for &v in grads["w"].iter() {
            assert!((v - 0.0).abs() < 1e-10);
        }
    }

    // ---- grad_norm_map ----

    #[test]
    fn test_grad_norm_map_single_tensor() {
        let grads = make_grad_map(&[("w", vec![3.0, 4.0])]);
        let norm = grad_norm_map(&grads);
        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_grad_norm_map_multiple_tensors() {
        // [3, 0] + [0, 4] => global L2 = 5
        let grads = make_grad_map(&[("w1", vec![3.0, 0.0]), ("w2", vec![0.0, 4.0])]);
        let norm = grad_norm_map(&grads);
        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_grad_norm_map_empty() {
        let grads: HashMap<String, ArrayD<f64>> = HashMap::new();
        assert!((grad_norm_map(&grads) - 0.0).abs() < 1e-10);
    }

    // ---- GradientAccumulatorMap ----

    #[test]
    fn test_accumulator_map_basic_flow() {
        let mut acc = GradientAccumulatorMap::new(4);

        for _ in 0..4 {
            let grads = make_grad_map(&[("w", vec![1.0, 2.0])]);
            acc.accumulate(&grads).expect("unexpected None or Err");
        }

        assert!(acc.should_update());
        let avg = acc.get_averaged_grads().expect("failed to create avg");
        // Average of 4 identical [1, 2] grads = [1, 2]
        let vals: Vec<f64> = avg["w"].iter().copied().collect();
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_accumulator_map_averaging() {
        let mut acc = GradientAccumulatorMap::new(2);

        // Step 1: grads = [2, 4]
        let g1 = make_grad_map(&[("w", vec![2.0, 4.0])]);
        acc.accumulate(&g1).expect("unexpected None or Err");
        assert!(!acc.should_update()); // need 2 steps

        // Step 2: grads = [4, 8]
        let g2 = make_grad_map(&[("w", vec![4.0, 8.0])]);
        acc.accumulate(&g2).expect("unexpected None or Err");
        assert!(acc.should_update());

        let avg = acc.get_averaged_grads().expect("failed to create avg");
        let vals: Vec<f64> = avg["w"].iter().copied().collect();
        // Average = ([2,4] + [4,8]) / 2 = [3, 6]
        assert!((vals[0] - 3.0).abs() < 1e-10);
        assert!((vals[1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_accumulator_map_reset() {
        let mut acc = GradientAccumulatorMap::new(2);
        let grads = make_grad_map(&[("w", vec![1.0])]);
        acc.accumulate(&grads).expect("unexpected None or Err");
        acc.accumulate(&grads).expect("unexpected None or Err");
        assert!(acc.should_update());

        acc.reset();
        assert!(!acc.should_update());
        assert_eq!(acc.current_step(), 0);
    }

    #[test]
    fn test_accumulator_map_get_averaged_grads_empty_errors() {
        let acc = GradientAccumulatorMap::new(4);
        assert!(acc.get_averaged_grads().is_err());
    }

    #[test]
    fn test_accumulator_map_shape_mismatch_errors() {
        let mut acc = GradientAccumulatorMap::new(4);
        let g1 = make_grad_map(&[("w", vec![1.0, 2.0])]);
        acc.accumulate(&g1).expect("unexpected None or Err");

        // Same name, different shape
        let g2: HashMap<String, ArrayD<f64>> = {
            let mut m = HashMap::new();
            m.insert(
                "w".to_string(),
                Array::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 2.0, 3.0]).expect("unexpected None or Err"),
            );
            m
        };
        assert!(acc.accumulate(&g2).is_err());
    }

    #[test]
    fn test_accumulator_map_multiple_params() {
        let mut acc = GradientAccumulatorMap::new(2);

        let g1 = make_grad_map(&[("weight", vec![1.0, 0.0]), ("bias", vec![0.5])]);
        let g2 = make_grad_map(&[("weight", vec![3.0, 2.0]), ("bias", vec![1.5])]);

        acc.accumulate(&g1).expect("unexpected None or Err");
        acc.accumulate(&g2).expect("unexpected None or Err");
        assert!(acc.should_update());

        let avg = acc.get_averaged_grads().expect("failed to create avg");
        // weight avg = [2, 1], bias avg = [1]
        let w_vals: Vec<f64> = avg["weight"].iter().copied().collect();
        let b_vals: Vec<f64> = avg["bias"].iter().copied().collect();
        assert!((w_vals[0] - 2.0).abs() < 1e-10);
        assert!((w_vals[1] - 1.0).abs() < 1e-10);
        assert!((b_vals[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_accumulator_map_step_size_one() {
        let mut acc = GradientAccumulatorMap::new(1);
        let grads = make_grad_map(&[("w", vec![5.0, 10.0])]);
        acc.accumulate(&grads).expect("unexpected None or Err");
        assert!(acc.should_update());
        let avg = acc.get_averaged_grads().expect("failed to create avg");
        let vals: Vec<f64> = avg["w"].iter().copied().collect();
        assert!((vals[0] - 5.0).abs() < 1e-10);
        assert!((vals[1] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_accumulator_map_zero_step_size_clamped_to_one() {
        // steps=0 should be clamped to 1
        let mut acc = GradientAccumulatorMap::new(0);
        assert_eq!(acc.accumulation_steps(), 1);
        let grads = make_grad_map(&[("w", vec![1.0])]);
        acc.accumulate(&grads).expect("unexpected None or Err");
        assert!(acc.should_update());
    }

    #[test]
    fn test_accumulator_map_reusable_after_reset() {
        let mut acc = GradientAccumulatorMap::new(2);

        // First accumulation cycle
        let g1 = make_grad_map(&[("w", vec![4.0])]);
        acc.accumulate(&g1).expect("unexpected None or Err");
        acc.accumulate(&g1).expect("unexpected None or Err");
        let avg1 = acc.get_averaged_grads().expect("failed to create avg1");
        assert!((avg1["w"][[0]] - 4.0).abs() < 1e-10);
        acc.reset();

        // Second accumulation cycle with different values
        let g2 = make_grad_map(&[("w", vec![8.0])]);
        acc.accumulate(&g2).expect("unexpected None or Err");
        acc.accumulate(&g2).expect("unexpected None or Err");
        let avg2 = acc.get_averaged_grads().expect("failed to create avg2");
        assert!((avg2["w"][[0]] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_accumulator_map_param_names() {
        let mut acc = GradientAccumulatorMap::new(4);
        let grads = make_grad_map(&[("layer1.weight", vec![1.0]), ("layer1.bias", vec![0.1])]);
        acc.accumulate(&grads).expect("unexpected None or Err");

        let mut names: Vec<&str> = acc.param_names().collect();
        names.sort();
        assert_eq!(names, &["layer1.bias", "layer1.weight"]);
    }

    #[test]
    fn test_clip_and_accumulate_integration() {
        // Combine clipping and accumulation in a typical training pattern
        let mut acc = GradientAccumulatorMap::new(3);

        for step in 0..3 {
            let mut grads =
                make_grad_map(&[("w", vec![10.0 * (step as f64 + 1.0), -5.0])]);
            // Clip before accumulating
            clip_grad_norm_map(&mut grads, 1.0).expect("unexpected None or Err");
            acc.accumulate(&grads).expect("unexpected None or Err");
        }

        assert!(acc.should_update());
        let avg = acc.get_averaged_grads().expect("failed to create avg");
        // Each gradient was clipped to norm 1, average should also be norm ~1
        let norm = grad_norm_map(&avg);
        assert!(norm <= 1.0 + 1e-6);
    }
}
