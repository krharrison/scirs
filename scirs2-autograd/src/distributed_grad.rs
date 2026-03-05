//! Distributed gradient primitives for parallel training
//!
//! This module provides gradient synchronisation primitives for distributed
//! training across multiple workers (processes or threads):
//!
//! - **AllReduce**: Ring and tree allreduce algorithms
//! - **Gradient bucketing**: Accumulate small gradients into communication buckets
//! - **Gradient compression**: Top-k and random-k sparsification
//! - **Asynchronous updates**: Non-blocking gradient synchronisation
//! - **Pipeline parallelism**: Model split across stages
//!
//! # Example
//!
//! ```rust
//! use scirs2_autograd::distributed_grad::*;
//! use scirs2_core::ndarray::Array1;
//!
//! // Simulate a 4-worker ring allreduce
//! let local_grads: Vec<Vec<f64>> = vec![
//!     vec![1.0, 2.0, 3.0],
//!     vec![4.0, 5.0, 6.0],
//!     vec![7.0, 8.0, 9.0],
//!     vec![10.0, 11.0, 12.0],
//! ];
//! let result = simulate_ring_allreduce(&local_grads, ReduceOp::Sum)
//!     .expect("allreduce");
//! // Sum of all workers: [22, 26, 30]
//! assert!((result[0] - 22.0).abs() < 1e-10);
//! ```

use crate::error::AutogradError;
use crate::{Float, NdArray, Result};
use scirs2_core::ndarray::{Array1, ArrayD, IxDyn};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Reduce operation
// ---------------------------------------------------------------------------

/// Reduction operation for allreduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    /// Element-wise sum
    Sum,
    /// Element-wise mean (sum / num_workers)
    Mean,
    /// Element-wise max
    Max,
    /// Element-wise min
    Min,
}

impl ReduceOp {
    /// Apply this reduce operation to two vectors element-wise.
    pub fn reduce_pair<F: Float>(&self, a: &[F], b: &[F]) -> Vec<F> {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| match self {
                ReduceOp::Sum | ReduceOp::Mean => x + y,
                ReduceOp::Max => {
                    if x > y {
                        x
                    } else {
                        y
                    }
                }
                ReduceOp::Min => {
                    if x < y {
                        x
                    } else {
                        y
                    }
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Ring allreduce (simulation)
// ---------------------------------------------------------------------------

/// Simulate a ring allreduce across `n` workers.
///
/// Each worker contributes a gradient vector of the same length. The result is
/// the element-wise reduction across all workers.
///
/// The ring allreduce proceeds in two phases:
///
/// 1. **Scatter-reduce**: Each worker sends a chunk to its neighbour and reduces
///    the incoming chunk, after `n-1` steps every worker holds the fully-reduced
///    result for one chunk.
/// 2. **Allgather**: The fully-reduced chunks are propagated around the ring so
///    every worker ends up with the complete result.
///
/// # Arguments
/// * `local_grads` - One gradient vector per worker (all same length)
/// * `op` - Reduction operation
pub fn simulate_ring_allreduce(local_grads: &[Vec<f64>], op: ReduceOp) -> Result<Vec<f64>> {
    if local_grads.is_empty() {
        return Ok(Vec::new());
    }

    let n = local_grads.len();
    let len = local_grads[0].len();

    for (i, grad) in local_grads.iter().enumerate() {
        if grad.len() != len {
            return Err(AutogradError::ShapeMismatch(format!(
                "Worker {} gradient length {} != expected {}",
                i,
                grad.len(),
                len
            )));
        }
    }

    // For simulation correctness, perform a straightforward element-wise
    // reduction that mirrors what a real ring allreduce would produce.
    // The actual ring topology is an implementation detail of the transport
    // layer; the mathematical result is the same.
    let mut result = local_grads[0].clone();

    for worker_grads in local_grads.iter().skip(1) {
        for (r, &g) in result.iter_mut().zip(worker_grads.iter()) {
            *r = match op {
                ReduceOp::Sum | ReduceOp::Mean => *r + g,
                ReduceOp::Max => {
                    if g > *r {
                        g
                    } else {
                        *r
                    }
                }
                ReduceOp::Min => {
                    if g < *r {
                        g
                    } else {
                        *r
                    }
                }
            };
        }
    }

    if op == ReduceOp::Mean {
        let n_f = n as f64;
        for v in &mut result {
            *v /= n_f;
        }
    }

    Ok(result)
}

/// Simulate a tree allreduce across `n` workers.
///
/// Uses a binary-tree pattern: reduce up to root, then broadcast down.
pub fn simulate_tree_allreduce(local_grads: &[Vec<f64>], op: ReduceOp) -> Result<Vec<f64>> {
    if local_grads.is_empty() {
        return Ok(Vec::new());
    }

    let n = local_grads.len();
    let len = local_grads[0].len();

    for (i, grad) in local_grads.iter().enumerate() {
        if grad.len() != len {
            return Err(AutogradError::ShapeMismatch(format!(
                "Worker {} gradient length {} != expected {}",
                i,
                grad.len(),
                len
            )));
        }
    }

    let mut buffers: Vec<Vec<f64>> = local_grads.to_vec();

    // Phase 1: reduce (binary tree upward)
    let mut stride = 1;
    while stride < n {
        for i in (0..n).step_by(stride * 2) {
            let partner = i + stride;
            if partner < n {
                let reduced = op.reduce_pair(&buffers[i], &buffers[partner]);
                buffers[i] = reduced;
            }
        }
        stride *= 2;
    }

    // Phase 2: broadcast (binary tree downward)
    stride /= 2;
    while stride >= 1 {
        for i in (0..n).step_by(stride * 2) {
            let partner = i + stride;
            if partner < n {
                buffers[partner] = buffers[i].clone();
            }
        }
        if stride == 1 {
            break;
        }
        stride /= 2;
    }

    let mut result = buffers[0].clone();

    if op == ReduceOp::Mean {
        let n_f = n as f64;
        for v in &mut result {
            *v /= n_f;
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Gradient bucketing
// ---------------------------------------------------------------------------

/// Accumulates small gradients into communication buckets for efficiency.
///
/// Instead of sending each gradient individually, gradients are packed into
/// fixed-size buckets to amortise communication overhead.
pub struct GradientBucketer<F: Float> {
    /// Maximum bucket size in number of elements
    bucket_capacity: usize,
    /// Current bucket under construction
    current_bucket: Vec<F>,
    /// Parameter indices in the current bucket
    current_indices: Vec<usize>,
    /// Completed buckets ready for communication
    ready_buckets: Vec<GradientBucket<F>>,
    /// Total number of elements flushed
    total_flushed: usize,
}

/// A bucket of gradient elements ready for communication.
#[derive(Debug, Clone)]
pub struct GradientBucket<F: Float> {
    /// Flat gradient data
    pub data: Vec<F>,
    /// Parameter indices for reconstruction
    pub param_indices: Vec<usize>,
    /// Element count per parameter
    pub param_sizes: Vec<usize>,
}

impl<F: Float> GradientBucketer<F> {
    /// Create a new gradient bucketer with the given capacity (elements per bucket).
    pub fn new(bucket_capacity: usize) -> Self {
        let capacity = if bucket_capacity == 0 {
            1024
        } else {
            bucket_capacity
        };
        Self {
            bucket_capacity: capacity,
            current_bucket: Vec::new(),
            current_indices: Vec::new(),
            ready_buckets: Vec::new(),
            total_flushed: 0,
        }
    }

    /// Add a gradient for a parameter.
    pub fn add_gradient(&mut self, param_idx: usize, gradient: &NdArray<F>) {
        let flat: Vec<F> = gradient.iter().copied().collect();
        let flat_len = flat.len();

        if self.current_bucket.len() + flat_len > self.bucket_capacity
            && !self.current_bucket.is_empty()
        {
            self.flush_current();
        }

        self.current_bucket.extend(flat);
        self.current_indices.push(param_idx);

        // If the gradient itself exceeds bucket capacity, flush immediately
        if self.current_bucket.len() >= self.bucket_capacity {
            self.flush_current();
        }
    }

    /// Flush the current bucket to the ready queue.
    pub fn flush_current(&mut self) {
        if self.current_bucket.is_empty() {
            return;
        }

        let data = std::mem::take(&mut self.current_bucket);
        let indices = std::mem::take(&mut self.current_indices);
        self.total_flushed += data.len();

        self.ready_buckets.push(GradientBucket {
            data,
            param_indices: indices,
            param_sizes: Vec::new(), // simplified
        });
    }

    /// Flush all remaining and return all ready buckets.
    pub fn drain_buckets(&mut self) -> Vec<GradientBucket<F>> {
        self.flush_current();
        std::mem::take(&mut self.ready_buckets)
    }

    /// Number of ready (complete) buckets.
    pub fn num_ready_buckets(&self) -> usize {
        self.ready_buckets.len()
    }

    /// Number of elements in the current (incomplete) bucket.
    pub fn current_bucket_size(&self) -> usize {
        self.current_bucket.len()
    }

    /// Total number of elements flushed so far.
    pub fn total_flushed(&self) -> usize {
        self.total_flushed
    }

    /// Reset the bucketer.
    pub fn reset(&mut self) {
        self.current_bucket.clear();
        self.current_indices.clear();
        self.ready_buckets.clear();
        self.total_flushed = 0;
    }
}

// ---------------------------------------------------------------------------
// Gradient compression
// ---------------------------------------------------------------------------

/// Compression strategy for gradient communication.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionKind {
    /// No compression
    None,
    /// Keep only the top-k elements by absolute value
    TopK { ratio: f64 },
    /// Keep a random subset of elements
    RandomK { ratio: f64 },
    /// Quantize to fewer bits
    Quantize { bits: u8 },
}

/// A compressed gradient representation.
#[derive(Debug, Clone)]
pub struct CompressedGradient<F: Float> {
    /// Non-zero values
    pub values: Vec<F>,
    /// Indices of non-zero values in the original gradient
    pub indices: Vec<usize>,
    /// Original length
    pub original_len: usize,
    /// Compression kind used
    pub kind: CompressionKind,
    /// Compression ratio achieved (compressed_size / original_size)
    pub compression_ratio: f64,
}

/// Compress a gradient using top-k sparsification.
///
/// Keeps only the top-k fraction of elements by absolute value, setting
/// the rest to zero.
pub fn compress_topk<F: Float>(gradient: &NdArray<F>, ratio: f64) -> Result<CompressedGradient<F>> {
    let flat: Vec<(usize, F)> = gradient.iter().copied().enumerate().collect();
    let n = flat.len();
    let k = ((n as f64 * ratio).ceil() as usize).min(n).max(1);

    // Sort by absolute value (descending)
    let mut abs_indexed: Vec<(usize, F)> = flat.clone();
    abs_indexed.sort_by(|a, b| {
        let abs_a = if a.1 < F::zero() {
            F::zero() - a.1
        } else {
            a.1
        };
        let abs_b = if b.1 < F::zero() {
            F::zero() - b.1
        } else {
            b.1
        };
        abs_b
            .partial_cmp(&abs_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let values: Vec<F> = abs_indexed[..k].iter().map(|&(_, v)| v).collect();
    let indices: Vec<usize> = abs_indexed[..k].iter().map(|&(i, _)| i).collect();
    let compression_ratio = if n > 0 { k as f64 / n as f64 } else { 1.0 };

    Ok(CompressedGradient {
        values,
        indices,
        original_len: n,
        kind: CompressionKind::TopK { ratio },
        compression_ratio,
    })
}

/// Compress a gradient using random-k sparsification.
///
/// Keeps a random fraction of elements, scaling them by 1/ratio to maintain
/// the expected gradient value.
pub fn compress_randomk<F: Float>(
    gradient: &NdArray<F>,
    ratio: f64,
    seed: u64,
) -> Result<CompressedGradient<F>> {
    let flat: Vec<(usize, F)> = gradient.iter().copied().enumerate().collect();
    let n = flat.len();
    let k = ((n as f64 * ratio).ceil() as usize).min(n).max(1);

    // Simple deterministic pseudo-random selection using LCG
    let scale = F::from(1.0 / ratio).ok_or_else(|| {
        AutogradError::compute_error("Cannot compute random-k scale factor".into())
    })?;

    let mut rng_state = seed;

    // Generate k unique random indices
    let mut index_set = std::collections::HashSet::new();
    let mut attempts = 0;
    while index_set.len() < k && attempts < n * 3 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let idx = (rng_state >> 33) as usize % n;
        index_set.insert(idx);
        attempts += 1;
    }

    let mut indices: Vec<usize> = index_set.into_iter().collect();
    indices.sort_unstable();

    let values: Vec<F> = indices.iter().map(|&i| flat[i].1 * scale).collect();

    let compression_ratio = if n > 0 {
        indices.len() as f64 / n as f64
    } else {
        1.0
    };

    Ok(CompressedGradient {
        values,
        indices,
        original_len: n,
        kind: CompressionKind::RandomK { ratio },
        compression_ratio,
    })
}

/// Decompress a compressed gradient back to a full gradient.
pub fn decompress_gradient<F: Float>(compressed: &CompressedGradient<F>) -> Result<NdArray<F>> {
    let mut result = ArrayD::zeros(IxDyn(&[compressed.original_len]));
    for (&idx, &val) in compressed.indices.iter().zip(compressed.values.iter()) {
        if idx < compressed.original_len {
            result[IxDyn(&[idx])] = val;
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Asynchronous gradient updater
// ---------------------------------------------------------------------------

/// Status of an asynchronous gradient update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsyncStatus {
    /// The update is in progress
    Pending,
    /// The update has completed
    Completed,
    /// The update failed
    Failed,
}

/// Token for tracking asynchronous gradient updates.
#[derive(Debug, Clone, Copy)]
pub struct AsyncHandle {
    /// Unique ID for this async operation
    pub id: usize,
    /// Current status
    pub status: AsyncStatus,
    /// Worker rank that initiated the operation
    pub initiator: usize,
}

/// Manages asynchronous gradient synchronisation.
///
/// Allows non-blocking gradient updates: the training loop can continue
/// computing the next mini-batch while gradients from the previous batch
/// are being synchronised.
pub struct AsyncGradientUpdater<F: Float> {
    /// Pending updates
    pending: Vec<AsyncHandle>,
    /// Completed gradient buffers by handle ID
    completed_buffers: HashMap<usize, Vec<NdArray<F>>>,
    /// Next handle ID
    next_id: usize,
    /// Staleness tolerance (how many steps behind a gradient can be)
    staleness_limit: usize,
    /// Current step
    current_step: usize,
}

impl<F: Float> AsyncGradientUpdater<F> {
    /// Create a new async gradient updater.
    pub fn new(staleness_limit: usize) -> Self {
        Self {
            pending: Vec::new(),
            completed_buffers: HashMap::new(),
            next_id: 0,
            staleness_limit: if staleness_limit == 0 {
                1
            } else {
                staleness_limit
            },
            current_step: 0,
        }
    }

    /// Submit gradients for asynchronous synchronisation.
    ///
    /// Returns a handle for tracking the operation.
    pub fn submit(&mut self, gradients: Vec<NdArray<F>>, rank: usize) -> AsyncHandle {
        let id = self.next_id;
        self.next_id += 1;

        let handle = AsyncHandle {
            id,
            status: AsyncStatus::Pending,
            initiator: rank,
        };

        // In a real implementation, this would send gradients to other workers.
        // Here we simulate by immediately completing them.
        self.completed_buffers.insert(id, gradients);
        self.pending.push(AsyncHandle {
            status: AsyncStatus::Completed,
            ..handle
        });

        self.current_step += 1;
        handle
    }

    /// Check if an async operation has completed.
    pub fn is_complete(&self, handle: &AsyncHandle) -> bool {
        self.completed_buffers.contains_key(&handle.id)
    }

    /// Retrieve completed gradients (consumes the buffer).
    pub fn retrieve(&mut self, handle: &AsyncHandle) -> Option<Vec<NdArray<F>>> {
        self.completed_buffers.remove(&handle.id)
    }

    /// Number of pending operations.
    pub fn num_pending(&self) -> usize {
        self.pending
            .iter()
            .filter(|h| h.status == AsyncStatus::Pending)
            .count()
    }

    /// Number of completed operations.
    pub fn num_completed(&self) -> usize {
        self.completed_buffers.len()
    }

    /// Current step counter.
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Staleness limit.
    pub fn staleness_limit(&self) -> usize {
        self.staleness_limit
    }

    /// Clear all pending and completed operations.
    pub fn clear(&mut self) {
        self.pending.clear();
        self.completed_buffers.clear();
    }
}

// ---------------------------------------------------------------------------
// Pipeline parallelism
// ---------------------------------------------------------------------------

/// A stage in a pipeline-parallel model.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Stage index (0 = first layer group, N-1 = last)
    pub stage_id: usize,
    /// Worker rank assigned to this stage
    pub rank: usize,
    /// Layer indices belonging to this stage
    pub layer_indices: Vec<usize>,
    /// Input shape
    pub input_shape: Option<Vec<usize>>,
    /// Output shape
    pub output_shape: Option<Vec<usize>>,
}

impl PipelineStage {
    /// Create a new pipeline stage.
    pub fn new(stage_id: usize, rank: usize, layer_indices: Vec<usize>) -> Self {
        Self {
            stage_id,
            rank,
            layer_indices,
            input_shape: None,
            output_shape: None,
        }
    }

    /// Number of layers in this stage.
    pub fn num_layers(&self) -> usize {
        self.layer_indices.len()
    }
}

/// Configuration for pipeline parallelism.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of pipeline stages
    pub num_stages: usize,
    /// Number of micro-batches for pipelining
    pub num_micro_batches: usize,
    /// Whether to use 1F1B (one-forward-one-backward) schedule
    pub schedule_1f1b: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            num_stages: 2,
            num_micro_batches: 4,
            schedule_1f1b: true,
        }
    }
}

/// Create pipeline stages by evenly partitioning layers.
pub fn create_pipeline_stages(
    num_layers: usize,
    config: &PipelineConfig,
) -> Result<Vec<PipelineStage>> {
    if config.num_stages == 0 {
        return Err(AutogradError::OperationError(
            "num_stages must be > 0".into(),
        ));
    }
    if num_layers == 0 {
        return Ok(Vec::new());
    }

    let layers_per_stage = (num_layers + config.num_stages - 1) / config.num_stages;
    let mut stages = Vec::with_capacity(config.num_stages);

    for stage_id in 0..config.num_stages {
        let start = stage_id * layers_per_stage;
        let end = ((stage_id + 1) * layers_per_stage).min(num_layers);
        if start >= num_layers {
            break;
        }
        let indices: Vec<usize> = (start..end).collect();
        stages.push(PipelineStage::new(stage_id, stage_id, indices));
    }

    Ok(stages)
}

/// Generate a 1F1B (one-forward-one-backward) pipeline schedule.
///
/// Returns a sequence of (micro_batch, stage, is_forward) tuples representing
/// the execution order.
pub fn generate_1f1b_schedule(
    num_stages: usize,
    num_micro_batches: usize,
) -> Vec<(usize, usize, bool)> {
    let mut schedule = Vec::new();

    // Warmup phase: forward passes to fill the pipeline
    let warmup = num_stages.min(num_micro_batches);
    for mb in 0..warmup {
        for stage in 0..num_stages {
            if mb + stage < warmup + num_stages - 1 {
                schedule.push((mb, stage, true)); // forward
            }
        }
    }

    // Steady state: alternating 1 forward + 1 backward per stage
    for mb in warmup..num_micro_batches {
        for stage in 0..num_stages {
            schedule.push((mb, stage, true)); // forward
            let bwd_mb = mb - warmup + (num_stages - 1 - stage).min(mb);
            if bwd_mb < num_micro_batches {
                schedule.push((bwd_mb, stage, false)); // backward
            }
        }
    }

    // Cooldown phase: remaining backward passes
    for mb in 0..num_micro_batches {
        for stage in (0..num_stages).rev() {
            schedule.push((mb, stage, false));
        }
    }

    // Deduplicate
    let mut seen = std::collections::HashSet::new();
    schedule.retain(|entry| seen.insert(*entry));

    schedule
}

// ---------------------------------------------------------------------------
// Distributed training config
// ---------------------------------------------------------------------------

/// Configuration for distributed training.
#[derive(Debug, Clone)]
pub struct DistributedTrainingConfig {
    /// Number of workers
    pub num_workers: usize,
    /// Allreduce algorithm
    pub allreduce_algorithm: AllReduceAlgorithm,
    /// Gradient compression strategy
    pub compression: CompressionKind,
    /// Bucket size for gradient bucketing (number of elements)
    pub bucket_size: usize,
    /// Enable asynchronous gradient updates
    pub async_updates: bool,
    /// Staleness limit for async updates
    pub staleness_limit: usize,
    /// Pipeline parallelism config (None = data parallelism only)
    pub pipeline: Option<PipelineConfig>,
}

/// Allreduce algorithm choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllReduceAlgorithm {
    /// Ring allreduce (bandwidth-optimal)
    Ring,
    /// Tree allreduce (latency-optimal)
    Tree,
    /// Recursive halving-doubling
    RecursiveHalvingDoubling,
}

impl Default for DistributedTrainingConfig {
    fn default() -> Self {
        Self {
            num_workers: 1,
            allreduce_algorithm: AllReduceAlgorithm::Ring,
            compression: CompressionKind::None,
            bucket_size: 25_000_000, // 25M elements (~100MB for FP32)
            async_updates: false,
            staleness_limit: 1,
            pipeline: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Gradient synchroniser
// ---------------------------------------------------------------------------

/// Orchestrates gradient synchronisation across workers.
pub struct GradientSynchronizer<F: Float> {
    config: DistributedTrainingConfig,
    bucketer: GradientBucketer<F>,
    async_updater: AsyncGradientUpdater<F>,
    /// Total number of synchronisation rounds completed
    sync_count: usize,
    /// Total bytes communicated (estimated)
    total_bytes_communicated: usize,
}

impl<F: Float> GradientSynchronizer<F> {
    /// Create a new gradient synchroniser.
    pub fn new(config: DistributedTrainingConfig) -> Self {
        let staleness = config.staleness_limit;
        let bucket_size = config.bucket_size;
        Self {
            config,
            bucketer: GradientBucketer::new(bucket_size),
            async_updater: AsyncGradientUpdater::new(staleness),
            sync_count: 0,
            total_bytes_communicated: 0,
        }
    }

    /// Add a gradient for synchronisation.
    pub fn add_gradient(&mut self, param_idx: usize, gradient: &NdArray<F>) {
        self.bucketer.add_gradient(param_idx, gradient);
    }

    /// Synchronise all accumulated gradients.
    ///
    /// In a real distributed system, this would communicate with other workers.
    /// Here we simulate the allreduce step.
    pub fn synchronize(&mut self) -> Result<Vec<GradientBucket<F>>> {
        let buckets = self.bucketer.drain_buckets();
        self.sync_count += 1;

        for bucket in &buckets {
            self.total_bytes_communicated += bucket.data.len() * std::mem::size_of::<F>();
        }

        Ok(buckets)
    }

    /// Get the number of synchronisation rounds completed.
    pub fn sync_count(&self) -> usize {
        self.sync_count
    }

    /// Get the total estimated bytes communicated.
    pub fn total_bytes_communicated(&self) -> usize {
        self.total_bytes_communicated
    }

    /// Get the configuration.
    pub fn config(&self) -> &DistributedTrainingConfig {
        &self.config
    }

    /// Get a reference to the async updater.
    pub fn async_updater(&self) -> &AsyncGradientUpdater<F> {
        &self.async_updater
    }

    /// Get a mutable reference to the async updater.
    pub fn async_updater_mut(&mut self) -> &mut AsyncGradientUpdater<F> {
        &mut self.async_updater
    }

    /// Reset the synchroniser state.
    pub fn reset(&mut self) {
        self.bucketer.reset();
        self.async_updater.clear();
        self.sync_count = 0;
        self.total_bytes_communicated = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    // --- ReduceOp tests ---

    #[test]
    fn test_reduce_op_sum() {
        let a: Vec<f64> = vec![1.0, 2.0, 3.0];
        let b: Vec<f64> = vec![4.0, 5.0, 6.0];
        let result = ReduceOp::Sum.reduce_pair(&a, &b);
        assert!((result[0] - 5.0_f64).abs() < 1e-10);
        assert!((result[1] - 7.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_reduce_op_max() {
        let a: Vec<f64> = vec![1.0, 5.0, 3.0];
        let b: Vec<f64> = vec![4.0, 2.0, 6.0];
        let result = ReduceOp::Max.reduce_pair(&a, &b);
        assert!((result[0] - 4.0_f64).abs() < 1e-10);
        assert!((result[1] - 5.0_f64).abs() < 1e-10);
        assert!((result[2] - 6.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_reduce_op_min() {
        let a: Vec<f64> = vec![1.0, 5.0, 3.0];
        let b: Vec<f64> = vec![4.0, 2.0, 6.0];
        let result = ReduceOp::Min.reduce_pair(&a, &b);
        assert!((result[0] - 1.0_f64).abs() < 1e-10);
        assert!((result[1] - 2.0_f64).abs() < 1e-10);
        assert!((result[2] - 3.0_f64).abs() < 1e-10);
    }

    // --- Ring allreduce tests ---

    #[test]
    fn test_ring_allreduce_sum() {
        let grads = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![10.0, 11.0, 12.0],
        ];
        let result = simulate_ring_allreduce(&grads, ReduceOp::Sum).expect("allreduce");
        assert!((result[0] - 22.0).abs() < 1e-10);
        assert!((result[1] - 26.0).abs() < 1e-10);
        assert!((result[2] - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_ring_allreduce_mean() {
        let grads = vec![vec![4.0, 8.0], vec![6.0, 12.0]];
        let result = simulate_ring_allreduce(&grads, ReduceOp::Mean).expect("allreduce mean");
        assert!((result[0] - 5.0).abs() < 1e-10);
        assert!((result[1] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_ring_allreduce_empty() {
        let grads: Vec<Vec<f64>> = vec![];
        let result = simulate_ring_allreduce(&grads, ReduceOp::Sum).expect("empty");
        assert!(result.is_empty());
    }

    #[test]
    fn test_ring_allreduce_single_worker() {
        let grads = vec![vec![1.0, 2.0, 3.0]];
        let result = simulate_ring_allreduce(&grads, ReduceOp::Sum).expect("single");
        assert!((result[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ring_allreduce_length_mismatch() {
        let grads = vec![vec![1.0, 2.0], vec![3.0]];
        let result = simulate_ring_allreduce(&grads, ReduceOp::Sum);
        assert!(result.is_err());
    }

    // --- Tree allreduce tests ---

    #[test]
    fn test_tree_allreduce_sum() {
        let grads = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        let result = simulate_tree_allreduce(&grads, ReduceOp::Sum).expect("tree allreduce");
        assert!((result[0] - 16.0).abs() < 1e-10);
        assert!((result[1] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_tree_allreduce_mean() {
        let grads = vec![vec![2.0, 4.0], vec![6.0, 8.0]];
        let result = simulate_tree_allreduce(&grads, ReduceOp::Mean).expect("tree mean");
        assert!((result[0] - 4.0).abs() < 1e-10);
        assert!((result[1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_tree_allreduce_single() {
        let grads = vec![vec![42.0]];
        let result = simulate_tree_allreduce(&grads, ReduceOp::Sum).expect("single");
        assert!((result[0] - 42.0).abs() < 1e-10);
    }

    // --- GradientBucketer tests ---

    #[test]
    fn test_bucketer_basic() {
        let mut bucketer = GradientBucketer::<f64>::new(10);
        let g1 = Array1::from(vec![1.0, 2.0, 3.0]).into_dyn();
        let g2 = Array1::from(vec![4.0, 5.0, 6.0]).into_dyn();
        bucketer.add_gradient(0, &g1);
        bucketer.add_gradient(1, &g2);

        assert_eq!(bucketer.current_bucket_size(), 6);
    }

    #[test]
    fn test_bucketer_flush() {
        let mut bucketer = GradientBucketer::<f64>::new(5);
        let g1 = Array1::from(vec![1.0, 2.0, 3.0]).into_dyn();
        let g2 = Array1::from(vec![4.0, 5.0, 6.0]).into_dyn();
        bucketer.add_gradient(0, &g1);
        bucketer.add_gradient(1, &g2); // should trigger flush

        let buckets = bucketer.drain_buckets();
        assert!(buckets.len() >= 1);
    }

    #[test]
    fn test_bucketer_drain_empty() {
        let mut bucketer = GradientBucketer::<f64>::new(100);
        let buckets = bucketer.drain_buckets();
        assert!(buckets.is_empty());
    }

    #[test]
    fn test_bucketer_reset() {
        let mut bucketer = GradientBucketer::<f64>::new(100);
        let g = Array1::from(vec![1.0]).into_dyn();
        bucketer.add_gradient(0, &g);
        bucketer.reset();
        assert_eq!(bucketer.current_bucket_size(), 0);
        assert_eq!(bucketer.total_flushed(), 0);
    }

    // --- Compression tests ---

    #[test]
    fn test_topk_compression() {
        let grad = Array1::from(vec![0.1_f64, 10.0, 0.2, -20.0, 0.3]).into_dyn();
        let compressed = compress_topk(&grad, 0.4).expect("topk");
        // Top 40% = 2 elements: -20.0 and 10.0
        assert_eq!(compressed.values.len(), 2);
        assert!(compressed.compression_ratio <= 0.5);
    }

    #[test]
    fn test_topk_decompress() {
        let grad = Array1::from(vec![0.1_f64, 10.0, 0.2, -20.0, 0.3]).into_dyn();
        let compressed = compress_topk(&grad, 0.4).expect("topk");
        let decompressed = decompress_gradient(&compressed).expect("decompress");
        assert_eq!(decompressed.len(), 5);
        // Only top-k elements should be non-zero
        let nonzero_count = decompressed
            .iter()
            .filter(|&&v: &&f64| v.abs() > 1e-10)
            .count();
        assert_eq!(nonzero_count, 2);
    }

    #[test]
    fn test_randomk_compression() {
        let grad =
            Array1::from(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).into_dyn();
        let compressed = compress_randomk(&grad, 0.3, 42).expect("randomk");
        assert!(compressed.values.len() <= 4); // ~30% of 10
        assert!(compressed.values.len() >= 1);
    }

    #[test]
    fn test_randomk_decompress() {
        let grad = Array1::from(vec![1.0_f64; 100]).into_dyn();
        let compressed = compress_randomk(&grad, 0.5, 123).expect("randomk");
        let decompressed = decompress_gradient(&compressed).expect("decompress");
        assert_eq!(decompressed.len(), 100);
    }

    // --- AsyncGradientUpdater tests ---

    #[test]
    fn test_async_updater_submit() {
        let mut updater = AsyncGradientUpdater::<f64>::new(3);
        let grads = vec![Array1::from(vec![1.0, 2.0]).into_dyn()];
        let handle = updater.submit(grads, 0);
        assert!(updater.is_complete(&handle));
        assert_eq!(updater.current_step(), 1);
    }

    #[test]
    fn test_async_updater_retrieve() {
        let mut updater = AsyncGradientUpdater::<f64>::new(3);
        let grads = vec![Array1::from(vec![1.0, 2.0]).into_dyn()];
        let handle = updater.submit(grads, 0);
        let retrieved = updater.retrieve(&handle);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.as_ref().map(|v| v.len()), Some(1));
    }

    #[test]
    fn test_async_updater_clear() {
        let mut updater = AsyncGradientUpdater::<f64>::new(3);
        let grads = vec![Array1::from(vec![1.0]).into_dyn()];
        updater.submit(grads, 0);
        updater.clear();
        assert_eq!(updater.num_completed(), 0);
    }

    // --- Pipeline parallelism tests ---

    #[test]
    fn test_create_pipeline_stages() {
        let config = PipelineConfig {
            num_stages: 3,
            num_micro_batches: 4,
            schedule_1f1b: true,
        };
        let stages = create_pipeline_stages(12, &config).expect("stages");
        assert_eq!(stages.len(), 3);
        assert_eq!(stages[0].layer_indices.len(), 4);
        assert_eq!(stages[1].layer_indices.len(), 4);
        assert_eq!(stages[2].layer_indices.len(), 4);
    }

    #[test]
    fn test_create_pipeline_stages_uneven() {
        let config = PipelineConfig {
            num_stages: 3,
            num_micro_batches: 4,
            schedule_1f1b: true,
        };
        let stages = create_pipeline_stages(10, &config).expect("stages");
        assert_eq!(stages.len(), 3);
        // 10 / 3 = 4 per stage, last stage has 2
        assert_eq!(stages[0].num_layers(), 4);
        assert_eq!(stages[2].num_layers(), 2);
    }

    #[test]
    fn test_create_pipeline_stages_zero_stages() {
        let config = PipelineConfig {
            num_stages: 0,
            num_micro_batches: 4,
            schedule_1f1b: true,
        };
        let result = create_pipeline_stages(10, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_1f1b_schedule() {
        let schedule = generate_1f1b_schedule(2, 4);
        assert!(!schedule.is_empty());
        // Should contain both forward (true) and backward (false) entries
        let has_fwd = schedule.iter().any(|&(_, _, is_fwd)| is_fwd);
        let has_bwd = schedule.iter().any(|&(_, _, is_fwd)| !is_fwd);
        assert!(has_fwd);
        assert!(has_bwd);
    }

    #[test]
    fn test_pipeline_stage_num_layers() {
        let stage = PipelineStage::new(0, 0, vec![0, 1, 2, 3]);
        assert_eq!(stage.num_layers(), 4);
    }

    // --- GradientSynchronizer tests ---

    #[test]
    fn test_synchronizer_basic() {
        let config = DistributedTrainingConfig::default();
        let mut sync = GradientSynchronizer::<f64>::new(config);
        let g = Array1::from(vec![1.0, 2.0, 3.0]).into_dyn();
        sync.add_gradient(0, &g);
        let buckets = sync.synchronize().expect("sync");
        assert!(buckets.len() >= 1);
        assert_eq!(sync.sync_count(), 1);
    }

    #[test]
    fn test_synchronizer_reset() {
        let config = DistributedTrainingConfig::default();
        let mut sync = GradientSynchronizer::<f64>::new(config);
        let g = Array1::from(vec![1.0]).into_dyn();
        sync.add_gradient(0, &g);
        sync.synchronize().expect("sync");
        sync.reset();
        assert_eq!(sync.sync_count(), 0);
        assert_eq!(sync.total_bytes_communicated(), 0);
    }

    #[test]
    fn test_synchronizer_multiple_gradients() {
        let config = DistributedTrainingConfig {
            bucket_size: 5,
            ..Default::default()
        };
        let mut sync = GradientSynchronizer::<f64>::new(config);
        for i in 0..5 {
            let g = Array1::from(vec![i as f64; 3]).into_dyn();
            sync.add_gradient(i, &g);
        }
        let buckets = sync.synchronize().expect("sync");
        assert!(buckets.len() >= 1);
    }

    // --- DistributedTrainingConfig tests ---

    #[test]
    fn test_default_config() {
        let config = DistributedTrainingConfig::default();
        assert_eq!(config.num_workers, 1);
        assert_eq!(config.allreduce_algorithm, AllReduceAlgorithm::Ring);
        assert_eq!(config.compression, CompressionKind::None);
        assert!(!config.async_updates);
    }

    // --- CompressedGradient properties ---

    #[test]
    fn test_compressed_gradient_properties() {
        let grad = Array1::from(vec![1.0_f64, 0.0, 0.0, 5.0, 0.0]).into_dyn();
        let compressed = compress_topk(&grad, 0.4).expect("topk");
        assert_eq!(compressed.original_len, 5);
        assert!(compressed.compression_ratio <= 1.0);
    }
}
