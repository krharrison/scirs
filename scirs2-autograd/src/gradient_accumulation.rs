//! Gradient accumulation for large effective batch sizes
//!
//! This module provides [`GradientAccumulator`], which accumulates gradients over
//! multiple forward-backward passes before applying a single optimizer step.
//! This enables training with large effective batch sizes on limited memory.
//!
//! # Motivation
//!
//! When the desired batch size exceeds available memory, gradient accumulation
//! allows splitting one logical batch into N micro-batches:
//!
//! 1. For each micro-batch, compute gradients and add to the accumulator
//! 2. After N micro-batches, divide accumulated gradients by N and apply
//! 3. Zero the accumulator and repeat
//!
//! This produces the same result as training with the full batch, because
//! gradient of a sum equals sum of gradients.
//!
//! # Example
//!
//! ```rust
//! use scirs2_autograd::gradient_accumulation::GradientAccumulator;
//! use scirs2_core::ndarray::Array1;
//!
//! // Accumulate over 4 micro-batches for an effective batch size of 4x
//! let mut accum = GradientAccumulator::<f64>::new(4);
//!
//! // Simulate 4 micro-batch gradient computations
//! for _ in 0..4 {
//!     let grad = Array1::from(vec![1.0, 2.0, 3.0]).into_dyn();
//!     accum.accumulate_single(grad);
//! }
//!
//! assert_eq!(accum.current_step(), 4);
//! assert!(accum.should_step());
//!
//! // Retrieve accumulated (averaged) gradients
//! let averaged = accum.get_averaged_gradients();
//! assert_eq!(averaged.len(), 1);
//!
//! // Reset for next accumulation cycle
//! accum.zero_grad();
//! assert_eq!(accum.current_step(), 0);
//! ```

use crate::{Float, NdArray};
use scirs2_core::ndarray::{ArrayD, Axis, IxDyn};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// GradientAccumulator
// ---------------------------------------------------------------------------

/// Accumulates gradients over multiple micro-batches.
///
/// Designed to be used in a training loop where the effective batch size
/// is larger than what fits in memory. After `accumulation_steps` calls
/// to [`GradientAccumulator::accumulate`], the accumulated gradients can be retrieved
/// (optionally averaged) and applied to an optimizer.
///
/// # Type Parameters
/// * `F` - Floating point type (f32 or f64)
pub struct GradientAccumulator<F: Float> {
    /// Number of micro-batches to accumulate before stepping
    accumulation_steps: usize,
    /// Current step counter within an accumulation cycle
    current_step: usize,
    /// Accumulated gradient buffers, keyed by parameter index
    buffers: HashMap<usize, NdArray<F>>,
    /// Optional gradient scaling factor (for mixed-precision training)
    grad_scale: F,
    /// Whether to average gradients (divide by accumulation_steps)
    average: bool,
    /// Maximum gradient norm for clipping (None = no clipping)
    max_grad_norm: Option<F>,
    /// Total number of optimizer steps taken
    total_steps: usize,
}

impl<F: Float> GradientAccumulator<F> {
    /// Create a new gradient accumulator.
    ///
    /// # Arguments
    /// * `accumulation_steps` - Number of micro-batches per effective batch.
    ///   Must be at least 1.
    pub fn new(accumulation_steps: usize) -> Self {
        let steps = if accumulation_steps == 0 {
            1
        } else {
            accumulation_steps
        };

        Self {
            accumulation_steps: steps,
            current_step: 0,
            buffers: HashMap::new(),
            grad_scale: F::one(),
            average: true,
            max_grad_norm: None,
            total_steps: 0,
        }
    }

    /// Create a gradient accumulator with custom settings.
    ///
    /// # Arguments
    /// * `accumulation_steps` - Number of micro-batches per effective batch
    /// * `average` - Whether to average gradients (divide by step count)
    /// * `grad_scale` - Scaling factor for gradients (for mixed-precision)
    /// * `max_grad_norm` - Optional max norm for gradient clipping
    pub fn with_config(
        accumulation_steps: usize,
        average: bool,
        grad_scale: F,
        max_grad_norm: Option<F>,
    ) -> Self {
        let steps = if accumulation_steps == 0 {
            1
        } else {
            accumulation_steps
        };

        Self {
            accumulation_steps: steps,
            current_step: 0,
            buffers: HashMap::new(),
            grad_scale,
            average,
            max_grad_norm,
            total_steps: 0,
        }
    }

    /// Accumulate a single gradient array into buffer index 0.
    ///
    /// All calls accumulate into the same buffer (index 0), and the step
    /// counter is incremented. Use [`GradientAccumulator::accumulate`] for multi-parameter
    /// accumulation where each call passes all parameter gradients.
    pub fn accumulate_single(&mut self, gradient: NdArray<F>) {
        self.accumulate_at(0, gradient);
        self.current_step += 1;
    }

    /// Accumulate gradients for multiple parameters.
    ///
    /// Each gradient in the slice is accumulated at the corresponding index.
    /// This should be called once per micro-batch.
    ///
    /// # Arguments
    /// * `gradients` - Slice of gradient arrays, one per parameter
    pub fn accumulate(&mut self, gradients: &[NdArray<F>]) {
        for (idx, grad) in gradients.iter().enumerate() {
            self.accumulate_at(idx, grad.clone());
        }
        self.current_step += 1;
    }

    /// Accumulate a gradient at a specific parameter index.
    fn accumulate_at(&mut self, idx: usize, gradient: NdArray<F>) {
        let scaled_grad = if self.grad_scale != F::one() {
            gradient.mapv(|v| v * self.grad_scale)
        } else {
            gradient
        };

        match self.buffers.get_mut(&idx) {
            Some(buf) => {
                if buf.shape() == scaled_grad.shape() {
                    *buf = &*buf + &scaled_grad;
                } else {
                    // Shape mismatch: replace buffer (first occurrence wins)
                    *buf = scaled_grad;
                }
            }
            None => {
                self.buffers.insert(idx, scaled_grad);
            }
        }
    }

    /// Check whether it's time to perform an optimizer step.
    ///
    /// Returns `true` when `accumulation_steps` micro-batches have been
    /// accumulated.
    pub fn should_step(&self) -> bool {
        self.current_step >= self.accumulation_steps
    }

    /// Get the current micro-batch step count.
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Get the total number of optimizer steps performed.
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Get the number of accumulation steps configured.
    pub fn accumulation_steps(&self) -> usize {
        self.accumulation_steps
    }

    /// Get the number of parameter buffers currently tracked.
    pub fn num_param_groups(&self) -> usize {
        self.buffers.len()
    }

    /// Zero out all accumulated gradients and reset the step counter.
    ///
    /// Call this after applying the accumulated gradients with an optimizer.
    pub fn zero_grad(&mut self) {
        for buf in self.buffers.values_mut() {
            buf.fill(F::zero());
        }
        self.current_step = 0;
    }

    /// Clear all buffers entirely (frees memory).
    pub fn clear(&mut self) {
        self.buffers.clear();
        self.current_step = 0;
    }

    /// Mark that an optimizer step was performed.
    ///
    /// Increments the total step counter. Call after applying gradients.
    pub fn mark_step(&mut self) {
        self.total_steps += 1;
    }

    /// Perform a step: zero_grad + mark_step combined.
    ///
    /// Convenience method for the common pattern of zeroing gradients
    /// and incrementing the step counter after optimizer update.
    pub fn finish_step(&mut self) {
        self.mark_step();
        self.zero_grad();
    }

    /// Get the raw accumulated gradients (not averaged).
    ///
    /// Returns gradients sorted by parameter index.
    pub fn get_raw_gradients(&self) -> Vec<NdArray<F>> {
        let mut indices: Vec<usize> = self.buffers.keys().copied().collect();
        indices.sort();
        indices
            .iter()
            .filter_map(|idx| self.buffers.get(idx).cloned())
            .collect()
    }

    /// Get averaged accumulated gradients.
    ///
    /// If `average` is enabled, divides each gradient by the number of
    /// accumulated steps. Otherwise returns raw accumulated gradients.
    ///
    /// Optionally applies gradient clipping if `max_grad_norm` is set.
    pub fn get_averaged_gradients(&self) -> Vec<NdArray<F>> {
        let raw = self.get_raw_gradients();

        let averaged: Vec<NdArray<F>> = if self.average && self.current_step > 0 {
            let step_f = F::from(self.current_step).unwrap_or(F::one());
            raw.into_iter().map(|g| g.mapv(|v| v / step_f)).collect()
        } else {
            raw
        };

        // Apply gradient clipping if configured
        match self.max_grad_norm {
            Some(max_norm) => self.clip_gradients(averaged, max_norm),
            None => averaged,
        }
    }

    /// Clip gradients by global norm.
    ///
    /// If the total L2 norm of all gradients exceeds `max_norm`,
    /// scale all gradients down proportionally.
    fn clip_gradients(&self, gradients: Vec<NdArray<F>>, max_norm: F) -> Vec<NdArray<F>> {
        // Compute total norm squared
        let mut total_norm_sq = F::zero();
        for g in &gradients {
            for &val in g.iter() {
                total_norm_sq += val * val;
            }
        }

        let total_norm = total_norm_sq.sqrt();

        if total_norm > max_norm && total_norm > F::zero() {
            let clip_factor = max_norm / total_norm;
            gradients
                .into_iter()
                .map(|g| g.mapv(|v| v * clip_factor))
                .collect()
        } else {
            gradients
        }
    }

    /// Set the gradient scaling factor.
    ///
    /// Useful for mixed-precision training where gradients need to be
    /// scaled up during accumulation and scaled down before stepping.
    pub fn set_grad_scale(&mut self, scale: F) {
        self.grad_scale = scale;
    }

    /// Get the current gradient scaling factor.
    pub fn grad_scale(&self) -> F {
        self.grad_scale
    }

    /// Update the number of accumulation steps.
    ///
    /// Useful for dynamic batch sizing during training.
    pub fn set_accumulation_steps(&mut self, steps: usize) {
        self.accumulation_steps = if steps == 0 { 1 } else { steps };
    }

    /// Compute the effective batch size.
    ///
    /// # Arguments
    /// * `micro_batch_size` - Size of each micro-batch
    ///
    /// # Returns
    /// `micro_batch_size * accumulation_steps`
    pub fn effective_batch_size(&self, micro_batch_size: usize) -> usize {
        micro_batch_size * self.accumulation_steps
    }

    /// Check if any gradients contain NaN or Inf.
    ///
    /// Returns `true` if any accumulated gradient buffer has non-finite values.
    pub fn has_non_finite_grads(&self) -> bool {
        for buf in self.buffers.values() {
            for &val in buf.iter() {
                if !val.is_finite() {
                    return true;
                }
            }
        }
        false
    }

    /// Get statistics about the accumulated gradients.
    pub fn grad_stats(&self) -> GradientStats<F> {
        let mut total_elements = 0usize;
        let mut total_norm_sq = F::zero();
        let mut max_abs = F::zero();
        let mut min_abs = F::infinity();

        for buf in self.buffers.values() {
            for &val in buf.iter() {
                total_elements += 1;
                total_norm_sq += val * val;
                let abs_val = val.abs();
                if abs_val > max_abs {
                    max_abs = abs_val;
                }
                if abs_val < min_abs {
                    min_abs = abs_val;
                }
            }
        }

        let total_norm = total_norm_sq.sqrt();

        GradientStats {
            total_elements,
            total_norm,
            max_abs,
            min_abs: if total_elements > 0 {
                min_abs
            } else {
                F::zero()
            },
            num_param_groups: self.buffers.len(),
            current_step: self.current_step,
        }
    }
}

// ---------------------------------------------------------------------------
// GradientStats
// ---------------------------------------------------------------------------

/// Statistics about accumulated gradients.
#[derive(Debug, Clone)]
pub struct GradientStats<F: Float> {
    /// Total number of gradient elements across all parameters
    pub total_elements: usize,
    /// L2 norm of all gradients
    pub total_norm: F,
    /// Maximum absolute gradient value
    pub max_abs: F,
    /// Minimum absolute gradient value
    pub min_abs: F,
    /// Number of parameter groups
    pub num_param_groups: usize,
    /// Current accumulation step
    pub current_step: usize,
}

impl<F: Float + std::fmt::Display> std::fmt::Display for GradientStats<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GradientStats(step={}, norm={}, max_abs={}, elements={}, groups={})",
            self.current_step,
            self.total_norm,
            self.max_abs,
            self.total_elements,
            self.num_param_groups
        )
    }
}

// ---------------------------------------------------------------------------
// VirtualBatchAccumulator
// ---------------------------------------------------------------------------

/// Higher-level accumulator that simulates larger batch sizes.
///
/// Wraps [`GradientAccumulator`] with additional bookkeeping for
/// tracking effective training progress (samples seen, epochs, etc.).
pub struct VirtualBatchAccumulator<F: Float> {
    /// Inner gradient accumulator
    inner: GradientAccumulator<F>,
    /// Micro-batch size
    micro_batch_size: usize,
    /// Total samples processed in current epoch
    samples_processed: usize,
    /// Total samples processed across all epochs
    total_samples: usize,
    /// Number of completed epochs
    completed_epochs: usize,
    /// Dataset size (for epoch tracking)
    dataset_size: Option<usize>,
}

impl<F: Float> VirtualBatchAccumulator<F> {
    /// Create a new virtual batch accumulator.
    ///
    /// # Arguments
    /// * `micro_batch_size` - Size of each micro-batch
    /// * `accumulation_steps` - Number of micro-batches per step
    /// * `dataset_size` - Optional total dataset size for epoch tracking
    pub fn new(
        micro_batch_size: usize,
        accumulation_steps: usize,
        dataset_size: Option<usize>,
    ) -> Self {
        Self {
            inner: GradientAccumulator::new(accumulation_steps),
            micro_batch_size: if micro_batch_size == 0 {
                1
            } else {
                micro_batch_size
            },
            samples_processed: 0,
            total_samples: 0,
            completed_epochs: 0,
            dataset_size,
        }
    }

    /// Accumulate gradients from a micro-batch and track progress.
    pub fn accumulate(&mut self, gradients: &[NdArray<F>]) {
        self.inner.accumulate(gradients);
        self.samples_processed += self.micro_batch_size;
        self.total_samples += self.micro_batch_size;

        // Check for epoch boundary
        if let Some(ds) = self.dataset_size {
            if ds > 0 && self.samples_processed >= ds {
                self.completed_epochs += 1;
                self.samples_processed -= ds;
            }
        }
    }

    /// Check if it's time to step.
    pub fn should_step(&self) -> bool {
        self.inner.should_step()
    }

    /// Get averaged gradients for the optimizer step.
    pub fn get_gradients(&self) -> Vec<NdArray<F>> {
        self.inner.get_averaged_gradients()
    }

    /// Perform post-step cleanup.
    pub fn finish_step(&mut self) {
        self.inner.finish_step();
    }

    /// Get the effective batch size.
    pub fn effective_batch_size(&self) -> usize {
        self.inner.effective_batch_size(self.micro_batch_size)
    }

    /// Get training progress as a fraction of the current epoch.
    pub fn epoch_progress(&self) -> f64 {
        match self.dataset_size {
            Some(ds) if ds > 0 => self.samples_processed as f64 / ds as f64,
            _ => 0.0,
        }
    }

    /// Get the number of completed epochs.
    pub fn completed_epochs(&self) -> usize {
        self.completed_epochs
    }

    /// Get total samples processed.
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// Get total optimizer steps.
    pub fn total_steps(&self) -> usize {
        self.inner.total_steps()
    }

    /// Access the inner accumulator.
    pub fn inner(&self) -> &GradientAccumulator<F> {
        &self.inner
    }

    /// Mutable access to the inner accumulator.
    pub fn inner_mut(&mut self) -> &mut GradientAccumulator<F> {
        &mut self.inner
    }

    /// Zero all gradients and reset step counter.
    pub fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{arr1, Array1};

    #[test]
    fn test_basic_accumulation() {
        let mut accum = GradientAccumulator::<f64>::new(4);
        assert_eq!(accum.accumulation_steps(), 4);
        assert_eq!(accum.current_step(), 0);
        assert!(!accum.should_step());

        for _ in 0..4 {
            let grad = arr1(&[1.0, 2.0, 3.0]).into_dyn();
            accum.accumulate_single(grad);
        }

        assert_eq!(accum.current_step(), 4);
        assert!(accum.should_step());
    }

    #[test]
    fn test_averaged_gradients() {
        let mut accum = GradientAccumulator::<f64>::new(4);

        for _ in 0..4 {
            let grad = arr1(&[4.0, 8.0, 12.0]).into_dyn();
            accum.accumulate(&[grad]);
        }

        let averaged = accum.get_averaged_gradients();
        assert_eq!(averaged.len(), 1);
        let vals = averaged[0].as_slice().unwrap_or(&[]);
        assert!((vals[0] - 4.0).abs() < 1e-10); // 16/4
        assert!((vals[1] - 8.0).abs() < 1e-10); // 32/4
        assert!((vals[2] - 12.0).abs() < 1e-10); // 48/4
    }

    #[test]
    fn test_zero_grad() {
        let mut accum = GradientAccumulator::<f64>::new(2);
        let grad = arr1(&[1.0, 2.0]).into_dyn();
        accum.accumulate(&[grad]);
        assert_eq!(accum.current_step(), 1);

        accum.zero_grad();
        assert_eq!(accum.current_step(), 0);

        let raw = accum.get_raw_gradients();
        assert_eq!(raw.len(), 1);
        for &v in raw[0].iter() {
            assert!((v - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_gradient_clipping() {
        let mut accum = GradientAccumulator::<f64>::with_config(1, false, 1.0, Some(1.0));

        // Gradient with norm = 5 (3,4 -> sqrt(9+16) = 5)
        let grad = arr1(&[3.0, 4.0]).into_dyn();
        accum.accumulate(&[grad]);

        let clipped = accum.get_averaged_gradients();
        let vals = clipped[0].as_slice().unwrap_or(&[]);
        // Clipped to norm 1: [3/5, 4/5] = [0.6, 0.8]
        assert!((vals[0] - 0.6).abs() < 1e-10);
        assert!((vals[1] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_scaling() {
        let mut accum = GradientAccumulator::<f64>::with_config(1, false, 2.0, None);

        let grad = arr1(&[1.0, 2.0]).into_dyn();
        accum.accumulate(&[grad]);

        let raw = accum.get_raw_gradients();
        let vals = raw[0].as_slice().unwrap_or(&[]);
        // Scaled by 2.0
        assert!((vals[0] - 2.0).abs() < 1e-10);
        assert!((vals[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_multi_parameter_accumulation() {
        let mut accum = GradientAccumulator::<f64>::new(2);

        // First micro-batch: two parameter groups
        let grad1 = arr1(&[1.0, 2.0]).into_dyn();
        let grad2 = arr1(&[3.0]).into_dyn();
        accum.accumulate(&[grad1, grad2]);

        // Second micro-batch
        let grad1 = arr1(&[3.0, 4.0]).into_dyn();
        let grad2 = arr1(&[5.0]).into_dyn();
        accum.accumulate(&[grad1, grad2]);

        assert!(accum.should_step());
        assert_eq!(accum.num_param_groups(), 2);

        let averaged = accum.get_averaged_gradients();
        assert_eq!(averaged.len(), 2);

        // First param: (1+3)/2=2, (2+4)/2=3
        let vals1 = averaged[0].as_slice().unwrap_or(&[]);
        assert!((vals1[0] - 2.0).abs() < 1e-10);
        assert!((vals1[1] - 3.0).abs() < 1e-10);

        // Second param: (3+5)/2=4
        let vals2 = averaged[1].as_slice().unwrap_or(&[]);
        assert!((vals2[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_finish_step() {
        let mut accum = GradientAccumulator::<f64>::new(1);
        let grad = arr1(&[1.0]).into_dyn();
        accum.accumulate(&[grad]);

        assert_eq!(accum.total_steps(), 0);
        accum.finish_step();
        assert_eq!(accum.total_steps(), 1);
        assert_eq!(accum.current_step(), 0);
    }

    #[test]
    fn test_effective_batch_size() {
        let accum = GradientAccumulator::<f64>::new(8);
        assert_eq!(accum.effective_batch_size(32), 256);
    }

    #[test]
    fn test_grad_stats() {
        let mut accum = GradientAccumulator::<f64>::new(1);
        let grad = arr1(&[3.0, 4.0]).into_dyn();
        accum.accumulate(&[grad]);

        let stats = accum.grad_stats();
        assert_eq!(stats.total_elements, 2);
        assert!((stats.total_norm - 5.0).abs() < 1e-10); // sqrt(9+16) = 5
        assert!((stats.max_abs - 4.0).abs() < 1e-10);
        assert!((stats.min_abs - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_non_finite_detection() {
        let mut accum = GradientAccumulator::<f64>::new(1);
        let grad = arr1(&[1.0, 2.0]).into_dyn();
        accum.accumulate(&[grad]);
        assert!(!accum.has_non_finite_grads());

        let bad_grad = arr1(&[f64::NAN]).into_dyn();
        accum.accumulate(&[bad_grad]);
        assert!(accum.has_non_finite_grads());
    }

    #[test]
    fn test_clear() {
        let mut accum = GradientAccumulator::<f64>::new(1);
        let grad = arr1(&[1.0]).into_dyn();
        accum.accumulate(&[grad]);
        assert_eq!(accum.num_param_groups(), 1);

        accum.clear();
        assert_eq!(accum.num_param_groups(), 0);
        assert_eq!(accum.current_step(), 0);
    }

    #[test]
    fn test_dynamic_accumulation_steps() {
        let mut accum = GradientAccumulator::<f64>::new(2);
        assert_eq!(accum.accumulation_steps(), 2);

        accum.set_accumulation_steps(4);
        assert_eq!(accum.accumulation_steps(), 4);

        // Zero accumulation steps should be clamped to 1
        accum.set_accumulation_steps(0);
        assert_eq!(accum.accumulation_steps(), 1);
    }

    #[test]
    fn test_virtual_batch_accumulator() {
        let mut vba = VirtualBatchAccumulator::<f64>::new(32, 4, Some(1000));
        assert_eq!(vba.effective_batch_size(), 128);
        assert_eq!(vba.completed_epochs(), 0);

        // Simulate training
        for _ in 0..4 {
            let grad = arr1(&[1.0, 2.0]).into_dyn();
            vba.accumulate(&[grad]);
        }

        assert!(vba.should_step());
        assert_eq!(vba.total_samples(), 128);

        let grads = vba.get_gradients();
        assert_eq!(grads.len(), 1);

        vba.finish_step();
        assert_eq!(vba.total_steps(), 1);
    }

    #[test]
    fn test_virtual_batch_epoch_tracking() {
        let mut vba = VirtualBatchAccumulator::<f64>::new(100, 1, Some(300));

        for _ in 0..3 {
            let grad = arr1(&[1.0]).into_dyn();
            vba.accumulate(&[grad]);
            vba.finish_step();
        }

        assert_eq!(vba.total_samples(), 300);
        assert_eq!(vba.completed_epochs(), 1);
    }

    #[test]
    fn test_no_average_mode() {
        let mut accum = GradientAccumulator::<f64>::with_config(2, false, 1.0, None);

        let grad = arr1(&[2.0, 4.0]).into_dyn();
        accum.accumulate(&[grad.clone()]);
        accum.accumulate(&[grad]);

        // Without averaging, we get the raw sum
        let result = accum.get_averaged_gradients();
        let vals = result[0].as_slice().unwrap_or(&[]);
        assert!((vals[0] - 4.0).abs() < 1e-10); // 2+2
        assert!((vals[1] - 8.0).abs() < 1e-10); // 4+4
    }

    #[test]
    fn test_grad_stats_display() {
        let mut accum = GradientAccumulator::<f64>::new(1);
        let grad = arr1(&[3.0, 4.0]).into_dyn();
        accum.accumulate(&[grad]);

        let stats = accum.grad_stats();
        let display = format!("{stats}");
        assert!(display.contains("GradientStats"));
        assert!(display.contains("norm="));
    }
}
