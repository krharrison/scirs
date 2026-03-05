//! # Gradient Checkpointing
//!
//! Gradient checkpointing (also called activation recomputation) is a memory-saving
//! technique for training deep neural networks. Instead of storing all intermediate
//! activations during the forward pass for use in the backward pass, only a subset
//! of activations ("checkpoints") are stored. The remaining activations are recomputed
//! on-the-fly during backpropagation.
//!
//! This trades extra computation for reduced memory usage, which is essential for
//! training very deep models on memory-constrained devices.
//!
//! ## Design
//!
//! This module provides:
//!
//! 1. [`CheckpointedLayer`] – a wrapper around any type implementing [`Layer`] that
//!    stores the input activation for later recomputation, while discarding intermediate
//!    state.
//!
//! 2. [`SegmentCheckpointer`] – manages a sequence of layers and applies segment-based
//!    checkpointing: only every `N`-th activation is stored; the rest are recomputed
//!    during the backward pass from the nearest earlier checkpoint.
//!
//! 3. [`MemoryUsageTracker`] – tracks estimated memory consumption across forward and
//!    backward passes.
//!
//! ## Example
//!
//! ```rust,no_run
//! use scirs2_neural::gradient_checkpointing::{CheckpointedLayer, SegmentCheckpointer, MemoryUsageTracker};
//! use scirs2_neural::layers::{Layer, Dense};
//! use scirs2_core::random::rng;
//!
//! let mut rng_state = rng();
//! let layer = Dense::<f32>::new(64, 32, Some("relu"), &mut rng_state).expect("layer init");
//! let checkpointed = CheckpointedLayer::new(layer);
//! ```

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::Debug;
use std::sync::{Arc, Mutex, RwLock};

// ============================================================================
// MemoryUsageTracker
// ============================================================================

/// Tracks estimated host-side memory consumption during neural network training.
///
/// The tracker accumulates byte counts for:
/// - **Stored activations**: tensors cached during the forward pass for reuse in
///   the backward pass.
/// - **Recomputed activations**: tensors that are discarded after forward and
///   reconstructed during backward.
/// - **Gradients**: gradient tensors accumulated during backpropagation.
///
/// Byte counts are estimates based on tensor element counts and a fixed element
/// size (4 bytes for `f32`, 8 bytes for `f64`).
#[derive(Debug, Default, Clone)]
pub struct MemoryUsageTracker {
    /// Total bytes currently held in stored activation buffers
    stored_activation_bytes: Arc<Mutex<u64>>,
    /// Cumulative bytes of activations that have been recomputed
    recomputed_bytes: Arc<Mutex<u64>>,
    /// Total bytes currently held in gradient buffers
    gradient_bytes: Arc<Mutex<u64>>,
    /// Peak combined memory observed (stored activations + gradients)
    peak_bytes: Arc<Mutex<u64>>,
}

impl MemoryUsageTracker {
    /// Create a new empty tracker.
    pub fn new() -> Self {
        MemoryUsageTracker {
            stored_activation_bytes: Arc::new(Mutex::new(0)),
            recomputed_bytes: Arc::new(Mutex::new(0)),
            gradient_bytes: Arc::new(Mutex::new(0)),
            peak_bytes: Arc::new(Mutex::new(0)),
        }
    }

    /// Record that `bytes` bytes of activation memory are now being stored.
    pub fn record_stored_activation(&self, bytes: u64) {
        let new_stored = {
            if let Ok(mut v) = self.stored_activation_bytes.lock() {
                *v += bytes;
                *v
            } else {
                return;
            }
        };
        // Update peak outside the stored_activation_bytes lock to avoid deadlock.
        self.update_peak_external(new_stored);
    }

    /// Record that `bytes` bytes of stored activation memory have been freed.
    pub fn release_stored_activation(&self, bytes: u64) {
        if let Ok(mut v) = self.stored_activation_bytes.lock() {
            *v = v.saturating_sub(bytes);
        }
    }

    /// Record that `bytes` bytes were recomputed (not re-stored).
    pub fn record_recomputed(&self, bytes: u64) {
        if let Ok(mut v) = self.recomputed_bytes.lock() {
            *v += bytes;
        }
    }

    /// Record that `bytes` bytes of gradient memory are now live.
    pub fn record_gradient(&self, bytes: u64) {
        let new_grad = {
            if let Ok(mut v) = self.gradient_bytes.lock() {
                *v += bytes;
                *v
            } else {
                return;
            }
        };
        // Update peak outside the gradient_bytes lock to avoid deadlock.
        self.update_peak_external(new_grad);
    }

    /// Update the peak counter with `current_component` as one addend.
    ///
    /// Acquires locks to read the complementary counter, then updates peak.
    /// Must be called **after** releasing the caller's primary lock.
    fn update_peak_external(&self, current_component: u64) {
        // Read the other counter without holding the primary lock.
        let stored = self.stored_activation_bytes.lock().map(|v| *v).unwrap_or(0);
        let grad = self.gradient_bytes.lock().map(|v| *v).unwrap_or(0);
        let combined = stored + grad;
        // Use the maximum of combined and current_component to be conservative.
        let candidate = combined.max(current_component);
        if let Ok(mut pk) = self.peak_bytes.lock() {
            if candidate > *pk {
                *pk = candidate;
            }
        }
    }

    /// Current bytes held in stored activation buffers.
    pub fn stored_activation_bytes(&self) -> u64 {
        self.stored_activation_bytes
            .lock()
            .map(|v| *v)
            .unwrap_or(0)
    }

    /// Cumulative bytes that have been recomputed (never re-stored).
    pub fn recomputed_bytes(&self) -> u64 {
        self.recomputed_bytes.lock().map(|v| *v).unwrap_or(0)
    }

    /// Current bytes held in gradient buffers.
    pub fn gradient_bytes(&self) -> u64 {
        self.gradient_bytes.lock().map(|v| *v).unwrap_or(0)
    }

    /// Peak combined activation+gradient memory observed since creation.
    pub fn peak_bytes(&self) -> u64 {
        self.peak_bytes.lock().map(|v| *v).unwrap_or(0)
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        for m in &[
            &self.stored_activation_bytes,
            &self.recomputed_bytes,
            &self.gradient_bytes,
            &self.peak_bytes,
        ] {
            if let Ok(mut v) = m.lock() {
                *v = 0;
            }
        }
    }

    /// Return a human-readable summary of current memory usage.
    pub fn summary(&self) -> String {
        format!(
            "MemoryUsage {{ stored_activations: {} KiB, recomputed: {} KiB, \
             gradients: {} KiB, peak: {} KiB }}",
            self.stored_activation_bytes() / 1024,
            self.recomputed_bytes() / 1024,
            self.gradient_bytes() / 1024,
            self.peak_bytes() / 1024,
        )
    }
}

// ============================================================================
// CheckpointedLayer
// ============================================================================

/// A wrapper layer that applies gradient checkpointing to an inner [`Layer`].
///
/// During the forward pass the input is stored (but intermediate activations
/// inside the inner layer are not retained). During the backward pass the inner
/// layer's `forward` is called again on the stored input to recompute the
/// activation needed to compute gradients, then `backward` is called with the
/// recomputed activation.
///
/// This doubles the forward computation for this layer but can significantly
/// reduce peak memory for very deep or wide layers.
///
/// # Type Parameters
///
/// - `F` – the floating-point element type (e.g., `f32` or `f64`).
/// - `L` – the inner layer type that implements [`Layer<F>`].
pub struct CheckpointedLayer<F, L>
where
    F: Float + Debug + ScalarOperand + NumAssign + 'static,
    L: Layer<F> + 'static,
{
    /// The wrapped layer
    inner: L,
    /// The input saved from the most recent forward pass
    saved_input: RwLock<Option<Array<F, IxDyn>>>,
    /// Optional shared memory tracker
    memory_tracker: Option<Arc<MemoryUsageTracker>>,
    /// Bytes-per-element for memory accounting (derived from F)
    elem_bytes: usize,
    _phantom: std::marker::PhantomData<F>,
}

impl<F, L> CheckpointedLayer<F, L>
where
    F: Float + Debug + ScalarOperand + NumAssign + 'static,
    L: Layer<F> + 'static,
{
    /// Wrap `layer` with gradient checkpointing, using `size_of::<F>()` bytes per element.
    pub fn new(layer: L) -> Self {
        CheckpointedLayer {
            inner: layer,
            saved_input: RwLock::new(None),
            memory_tracker: None,
            elem_bytes: std::mem::size_of::<F>(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Attach a shared [`MemoryUsageTracker`] to record activation memory usage.
    pub fn with_tracker(mut self, tracker: Arc<MemoryUsageTracker>) -> Self {
        self.memory_tracker = Some(tracker);
        self
    }

    /// Provide immutable access to the inner layer.
    pub fn inner(&self) -> &L {
        &self.inner
    }

    /// Provide mutable access to the inner layer.
    pub fn inner_mut(&mut self) -> &mut L {
        &mut self.inner
    }

    /// Consume the wrapper and recover the inner layer.
    pub fn into_inner(self) -> L {
        self.inner
    }

    fn tensor_bytes(arr: &Array<F, IxDyn>) -> u64 {
        arr.len() as u64 * std::mem::size_of::<F>() as u64
    }
}

impl<F, L> Layer<F> for CheckpointedLayer<F, L>
where
    F: Float + Debug + ScalarOperand + NumAssign + Send + Sync + 'static,
    L: Layer<F> + Send + Sync + 'static,
{
    /// Forward pass: run the inner layer and **store the input** for recomputation.
    ///
    /// The stored input is the only activation saved; inner layer activations are
    /// discarded after this call returns.
    fn forward(
        &self,
        input: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let output = self.inner.forward(input)?;

        // Store input for recomputation
        let bytes = Self::tensor_bytes(input);
        {
            let mut saved = self
                .saved_input
                .write()
                .map_err(|_| NeuralError::ComputationError("lock poisoned in checkpoint forward".to_string()))?;

            // Release old allocation from tracker before overwriting
            if let (Some(old), Some(tracker)) = (saved.as_ref(), &self.memory_tracker) {
                tracker.release_stored_activation(Self::tensor_bytes(old));
            }
            *saved = Some(input.clone());
        }

        if let Some(tracker) = &self.memory_tracker {
            tracker.record_stored_activation(bytes);
        }

        Ok(output)
    }

    /// Backward pass: recompute the forward activation from the saved input, then
    /// call the inner layer's backward with the recomputed activation.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::ComputationError`] if no input was saved (i.e.,
    /// `forward` was never called on this layer).
    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Retrieve saved input
        let saved = self
            .saved_input
            .read()
            .map_err(|_| NeuralError::ComputationError("lock poisoned in checkpoint backward".to_string()))?;

        let input_ref = saved.as_ref().ok_or_else(|| {
            NeuralError::ComputationError(
                "CheckpointedLayer: backward called before forward; no saved input".to_string(),
            )
        })?;

        // Recompute forward to get the activation we need
        let recomputed = self.inner.forward(input_ref)?;
        if let Some(tracker) = &self.memory_tracker {
            tracker.record_recomputed(Self::tensor_bytes(&recomputed));
        }

        // Now run backward with the recomputed activation
        let grad_input = self.inner.backward(&recomputed, grad_output)?;

        if let Some(tracker) = &self.memory_tracker {
            tracker.record_gradient(Self::tensor_bytes(&grad_input));
        }

        Ok(grad_input)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        self.inner.update(learning_rate)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        self.inner.params()
    }

    fn gradients(&self) -> Vec<Array<F, IxDyn>> {
        self.inner.gradients()
    }

    fn set_gradients(&mut self, gradients: &[Array<F, IxDyn>]) -> Result<()> {
        self.inner.set_gradients(gradients)
    }

    fn set_params(&mut self, params: &[Array<F, IxDyn>]) -> Result<()> {
        self.inner.set_params(params)
    }

    fn set_training(&mut self, training: bool) {
        self.inner.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.inner.is_training()
    }

    fn layer_type(&self) -> &str {
        "CheckpointedLayer"
    }

    fn parameter_count(&self) -> usize {
        self.inner.parameter_count()
    }

    fn layer_description(&self) -> String {
        format!("CheckpointedLayer({})", self.inner.layer_description())
    }
}

// ============================================================================
// SegmentCheckpointer
// ============================================================================

/// Applies segment-based gradient checkpointing across a sequence of boxed layers.
///
/// The sequence is divided into segments of size `checkpoint_every`. At the
/// boundary of each segment the activation is stored. Within a segment the
/// activations are discarded and recomputed during the backward pass.
///
/// # Memory Savings
///
/// For a network with `N` layers and without checkpointing you need to store
/// `N` activations. With segment size `S`, you store approximately `N / S`
/// activations but recompute `S - 1` layers within each segment, giving an
/// `O(sqrt(N))` memory reduction when `S ≈ sqrt(N)`.
pub struct SegmentCheckpointer<F>
where
    F: Float + Debug + ScalarOperand + NumAssign + Send + Sync + 'static,
{
    /// All layers in order; layers at segment boundaries are wrapped in `CheckpointedLayer`
    layers: Vec<Box<dyn Layer<F>>>,
    /// Store activation every N layers (0 = no checkpointing)
    checkpoint_every: usize,
    /// Shared memory tracker
    memory_tracker: Arc<MemoryUsageTracker>,
    /// Activation saved at the start of the current segment (forward pass)
    segment_starts: RwLock<Vec<Array<F, IxDyn>>>,
}

impl<F> SegmentCheckpointer<F>
where
    F: Float + Debug + ScalarOperand + NumAssign + Send + Sync + 'static,
{
    /// Create a new `SegmentCheckpointer`.
    ///
    /// # Arguments
    ///
    /// - `layers` – ordered sequence of boxed layers.
    /// - `checkpoint_every` – activation is checkpointed every `checkpoint_every` layers.
    ///   A value of 0 means no intermediate checkpoints (only input and output are stored).
    pub fn new(layers: Vec<Box<dyn Layer<F>>>, checkpoint_every: usize) -> Self {
        SegmentCheckpointer {
            layers,
            checkpoint_every,
            memory_tracker: Arc::new(MemoryUsageTracker::new()),
            segment_starts: RwLock::new(Vec::new()),
        }
    }

    /// Attach a shared memory tracker.
    pub fn with_tracker(mut self, tracker: Arc<MemoryUsageTracker>) -> Self {
        self.memory_tracker = tracker;
        self
    }

    /// Get a reference to the underlying memory tracker.
    pub fn memory_tracker(&self) -> &Arc<MemoryUsageTracker> {
        &self.memory_tracker
    }

    /// Return the number of layers in this checkpointer.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Return `true` if no layers have been added.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// The effective checkpoint interval (0 means disabled).
    pub fn checkpoint_every(&self) -> usize {
        self.checkpoint_every
    }

    /// Run a full forward pass through all layers, saving activations at segment boundaries.
    ///
    /// Returns the final layer output.
    pub fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        if self.layers.is_empty() {
            return Ok(input.clone());
        }

        let mut segment_starts = self
            .segment_starts
            .write()
            .map_err(|_| NeuralError::ComputationError("lock poisoned".to_string()))?;
        segment_starts.clear();

        let ckpt = if self.checkpoint_every == 0 {
            usize::MAX
        } else {
            self.checkpoint_every
        };

        let mut current = input.clone();

        for (idx, layer) in self.layers.iter().enumerate() {
            // Save checkpoint at segment boundaries
            if idx % ckpt == 0 {
                let bytes = current.len() as u64 * std::mem::size_of::<F>() as u64;
                self.memory_tracker.record_stored_activation(bytes);
                segment_starts.push(current.clone());
            }
            current = layer.forward(&current)?;
        }

        Ok(current)
    }

    /// Run a backward pass through all layers in reverse order.
    ///
    /// For layers not at a segment boundary the stored segment-start activation is
    /// used to recompute the intermediate activations needed for gradients.
    ///
    /// # Arguments
    ///
    /// - `grad_output` – gradient with respect to the final output.
    ///
    /// # Errors
    ///
    /// Returns an error if `forward` was not called first.
    pub fn backward(&self, grad_output: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        if self.layers.is_empty() {
            return Ok(grad_output.clone());
        }

        let segment_starts = self
            .segment_starts
            .read()
            .map_err(|_| NeuralError::ComputationError("lock poisoned in backward".to_string()))?;

        if segment_starts.is_empty() {
            return Err(NeuralError::ComputationError(
                "SegmentCheckpointer::backward called before forward; no saved segment starts"
                    .to_string(),
            ));
        }

        let ckpt = if self.checkpoint_every == 0 {
            usize::MAX
        } else {
            self.checkpoint_every
        };

        let n = self.layers.len();
        let mut grad = grad_output.clone();

        // Process layers in reverse order
        for idx in (0..n).rev() {
            // Determine which segment this layer belongs to
            let seg_idx = idx / ckpt;
            let seg_idx_clamped = seg_idx.min(segment_starts.len().saturating_sub(1));
            let seg_start = &segment_starts[seg_idx_clamped];

            // Recompute activation at the input of layer `idx` by running forward
            // from the segment start up to (but not including) this layer.
            let seg_offset = seg_idx_clamped * ckpt;
            let mut activation = seg_start.clone();
            for recomp_idx in seg_offset..idx {
                activation = self.layers[recomp_idx].forward(&activation)?;
                let bytes = activation.len() as u64 * std::mem::size_of::<F>() as u64;
                self.memory_tracker.record_recomputed(bytes);
            }

            // Now compute gradients through this layer
            grad = self.layers[idx].backward(&activation, &grad)?;
            let bytes = grad.len() as u64 * std::mem::size_of::<F>() as u64;
            self.memory_tracker.record_gradient(bytes);
        }

        Ok(grad)
    }

    /// Add a new layer to the sequence.
    pub fn push(&mut self, layer: Box<dyn Layer<F>>) {
        self.layers.push(layer);
    }

    /// Return immutable access to the layers slice.
    pub fn layers(&self) -> &[Box<dyn Layer<F>>] {
        &self.layers
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Dense;
    use scirs2_core::ndarray::{Array, IxDyn};
    use scirs2_core::random::rng;

    fn make_dense(in_dim: usize, out_dim: usize) -> Dense<f32> {
        let mut rng_state = rng();
        Dense::<f32>::new(in_dim, out_dim, None, &mut rng_state)
            .expect("dense layer creation failed")
    }

    fn ones_input(shape: &[usize]) -> Array<f32, IxDyn> {
        Array::<f32, IxDyn>::ones(IxDyn(shape))
    }

    #[test]
    fn test_memory_tracker_basic() {
        let tracker = MemoryUsageTracker::new();
        tracker.record_stored_activation(4096);
        assert_eq!(tracker.stored_activation_bytes(), 4096);
        tracker.release_stored_activation(1024);
        assert_eq!(tracker.stored_activation_bytes(), 3072);
        tracker.record_recomputed(8192);
        assert_eq!(tracker.recomputed_bytes(), 8192);
        tracker.record_gradient(2048);
        assert_eq!(tracker.gradient_bytes(), 2048);
    }

    #[test]
    fn test_memory_tracker_peak() {
        let tracker = MemoryUsageTracker::new();
        tracker.record_stored_activation(1000);
        tracker.record_gradient(500);
        assert!(tracker.peak_bytes() >= 1000);
    }

    #[test]
    fn test_memory_tracker_reset() {
        let tracker = MemoryUsageTracker::new();
        tracker.record_stored_activation(4096);
        tracker.record_gradient(1024);
        tracker.reset();
        assert_eq!(tracker.stored_activation_bytes(), 0);
        assert_eq!(tracker.gradient_bytes(), 0);
    }

    #[test]
    fn test_memory_tracker_summary_format() {
        let tracker = MemoryUsageTracker::new();
        let summary = tracker.summary();
        assert!(summary.contains("MemoryUsage"));
        assert!(summary.contains("KiB"));
    }

    #[test]
    fn test_checkpointed_layer_forward() {
        let dense = make_dense(8, 4);
        let ckpt = CheckpointedLayer::new(dense);
        let input = ones_input(&[2, 8]);
        let output = ckpt.forward(&input).expect("forward failed");
        assert_eq!(output.shape(), &[2, 4]);
    }

    #[test]
    fn test_checkpointed_layer_backward() {
        let dense = make_dense(8, 4);
        let ckpt = CheckpointedLayer::new(dense);
        let input = ones_input(&[2, 8]);
        let _output = ckpt.forward(&input).expect("forward");
        let grad_out = ones_input(&[2, 4]);
        let grad_in = ckpt.backward(&input, &grad_out).expect("backward");
        assert_eq!(grad_in.shape(), &[2, 8]);
    }

    #[test]
    fn test_checkpointed_layer_backward_without_forward_errors() {
        let dense = make_dense(4, 2);
        let ckpt = CheckpointedLayer::new(dense);
        let grad = ones_input(&[1, 2]);
        let input_dummy = ones_input(&[1, 4]);
        let result = ckpt.backward(&input_dummy, &grad);
        assert!(result.is_err(), "backward without forward should error");
    }

    #[test]
    fn test_checkpointed_layer_memory_tracker() {
        let tracker = Arc::new(MemoryUsageTracker::new());
        let dense = make_dense(8, 4);
        let ckpt = CheckpointedLayer::new(dense).with_tracker(tracker.clone());
        let input = ones_input(&[3, 8]);
        let _out = ckpt.forward(&input).expect("forward");
        // 3*8*4 = 96 bytes stored
        assert!(tracker.stored_activation_bytes() > 0);
    }

    #[test]
    fn test_checkpointed_layer_trait_delegation() {
        let dense = make_dense(4, 2);
        let ckpt = CheckpointedLayer::new(dense);
        assert_eq!(ckpt.layer_type(), "CheckpointedLayer");
        assert!(ckpt.layer_description().contains("CheckpointedLayer"));
        // Parameter count should be delegated: 4*2 + 2 = 10
        assert_eq!(ckpt.parameter_count(), 10);
    }

    #[test]
    fn test_segment_checkpointer_forward() {
        let layers: Vec<Box<dyn Layer<f32>>> = vec![
            Box::new(make_dense(8, 6)),
            Box::new(make_dense(6, 4)),
            Box::new(make_dense(4, 2)),
        ];
        let ckpt = SegmentCheckpointer::new(layers, 2);
        let input = ones_input(&[2, 8]);
        let output = ckpt.forward(&input).expect("forward");
        assert_eq!(output.shape(), &[2, 2]);
    }

    #[test]
    fn test_segment_checkpointer_backward() {
        let layers: Vec<Box<dyn Layer<f32>>> = vec![
            Box::new(make_dense(8, 6)),
            Box::new(make_dense(6, 4)),
            Box::new(make_dense(4, 2)),
        ];
        let ckpt = SegmentCheckpointer::new(layers, 2);
        let input = ones_input(&[2, 8]);
        let _output = ckpt.forward(&input).expect("forward");
        let grad = ones_input(&[2, 2]);
        let grad_in = ckpt.backward(&grad).expect("backward");
        assert_eq!(grad_in.shape(), &[2, 8]);
    }

    #[test]
    fn test_segment_checkpointer_backward_without_forward_errors() {
        let layers: Vec<Box<dyn Layer<f32>>> = vec![Box::new(make_dense(4, 2))];
        let ckpt = SegmentCheckpointer::new(layers, 1);
        let grad = ones_input(&[1, 2]);
        let result = ckpt.backward(&grad);
        assert!(result.is_err());
    }

    #[test]
    fn test_segment_checkpointer_tracks_memory() {
        let tracker = Arc::new(MemoryUsageTracker::new());
        let layers: Vec<Box<dyn Layer<f32>>> = vec![
            Box::new(make_dense(8, 4)),
            Box::new(make_dense(4, 2)),
        ];
        let ckpt = SegmentCheckpointer::new(layers, 1).with_tracker(tracker.clone());
        let input = ones_input(&[4, 8]);
        let _out = ckpt.forward(&input).expect("forward");
        assert!(tracker.stored_activation_bytes() > 0);
    }

    #[test]
    fn test_segment_checkpointer_empty() {
        let ckpt: SegmentCheckpointer<f32> = SegmentCheckpointer::new(vec![], 2);
        let input = ones_input(&[2, 4]);
        let out = ckpt.forward(&input).expect("empty forward");
        assert_eq!(out.shape(), input.shape());
        assert!(ckpt.is_empty());
        assert_eq!(ckpt.len(), 0);
    }

    #[test]
    fn test_segment_checkpointer_checkpoint_every() {
        let ckpt: SegmentCheckpointer<f32> = SegmentCheckpointer::new(vec![], 3);
        assert_eq!(ckpt.checkpoint_every(), 3);
    }
}
