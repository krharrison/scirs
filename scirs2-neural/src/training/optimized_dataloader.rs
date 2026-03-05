//! Optimized data loading pipeline with prefetching and parallel loading
//!
//! This module provides an optimized data loading pipeline with:
//! - Prefetching for overlapping data loading and computation
//! - Parallel batch loading with configurable worker threads
//! - Memory-efficient batch caching
//! - Automatic batch size optimization

use crate::data::Dataset;
use crate::error::{NeuralError, Result};
use scirs2_core::chunking::{ChunkConfig, ChunkStrategy, ChunkingUtils};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::random::seq::SliceRandom;
use scirs2_core::NumAssign;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Type alias for a batch pair of input and target arrays
type BatchPair<F> = (Array<F, IxDyn>, Array<F, IxDyn>);

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for optimized data loading
#[derive(Debug, Clone)]
pub struct OptimizedLoaderConfig {
    /// Batch size
    pub batch_size: usize,
    /// Number of batches to prefetch
    pub prefetch_size: usize,
    /// Number of worker threads (0 for single-threaded)
    pub num_workers: usize,
    /// Whether to drop the last incomplete batch
    pub drop_last: bool,
    /// Whether to shuffle data
    pub shuffle: bool,
    /// Pin memory for faster GPU transfer (placeholder for future GPU support)
    pub pin_memory: bool,
    /// Cache batches in memory
    pub cache_batches: bool,
    /// Maximum memory for cache (in bytes, 0 for unlimited)
    pub max_cache_memory: usize,
}

impl Default for OptimizedLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            prefetch_size: 2,
            num_workers: 0,
            drop_last: false,
            shuffle: true,
            pin_memory: false,
            cache_batches: false,
            max_cache_memory: 0,
        }
    }
}

/// Statistics for data loading performance
#[derive(Debug, Clone, Default)]
pub struct LoadingStats {
    /// Total batches loaded
    pub batches_loaded: usize,
    /// Total samples loaded
    pub samples_loaded: usize,
    /// Total loading time
    pub total_load_time: Duration,
    /// Average batch load time
    pub avg_batch_time: Duration,
    /// Cache hit count
    pub cache_hits: usize,
    /// Cache miss count
    pub cache_misses: usize,
    /// Prefetch queue wait time
    pub prefetch_wait_time: Duration,
}

// =============================================================================
// Batch Result Type
// =============================================================================

/// Type alias for batch result
pub type BatchResult<F> = Result<(Array<F, IxDyn>, Array<F, IxDyn>)>;

// =============================================================================
// Batch Cache
// =============================================================================

/// Cache for storing loaded batches
struct BatchCache<F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + Send + Sync> {
    /// Cached batches by index
    cache: Vec<Option<BatchPair<F>>>,
    /// Maximum number of cached batches
    max_batches: usize,
    /// Current memory usage estimate
    memory_usage: usize,
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + Send + Sync> BatchCache<F> {
    fn new(max_batches: usize) -> Self {
        Self {
            cache: vec![None; max_batches],
            max_batches,
            memory_usage: 0,
        }
    }

    fn get(&self, index: usize) -> Option<&BatchPair<F>> {
        if index < self.cache.len() {
            self.cache[index].as_ref()
        } else {
            None
        }
    }

    fn insert(&mut self, index: usize, batch: BatchPair<F>) {
        if index < self.cache.len() {
            let batch_size = estimate_array_memory(&batch.0) + estimate_array_memory(&batch.1);
            self.memory_usage += batch_size;
            self.cache[index] = Some(batch);
        }
    }

    fn clear(&mut self) {
        self.cache.iter_mut().for_each(|b| *b = None);
        self.memory_usage = 0;
    }
}

/// Estimate memory usage of an array
fn estimate_array_memory<F: Float + NumAssign>(array: &Array<F, IxDyn>) -> usize {
    array.len() * std::mem::size_of::<F>()
}

// =============================================================================
// Prefetch Queue
// =============================================================================

/// Thread-safe queue for prefetched batches
struct PrefetchQueue<F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + Send + Sync> {
    /// Queue of prefetched batches
    queue: Mutex<VecDeque<(usize, BatchResult<F>)>>,
    /// Maximum queue size
    max_size: usize,
    /// Current size
    size: AtomicUsize,
    /// Whether to stop prefetching
    stop: AtomicBool,
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + Send + Sync> PrefetchQueue<F> {
    fn new(max_size: usize) -> Self {
        Self {
            queue: Mutex::new(VecDeque::with_capacity(max_size)),
            max_size,
            size: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
        }
    }

    fn push(&self, index: usize, batch: BatchResult<F>) -> bool {
        if self.stop.load(Ordering::Relaxed) {
            return false;
        }

        // Wait if queue is full
        while self.size.load(Ordering::Relaxed) >= self.max_size {
            if self.stop.load(Ordering::Relaxed) {
                return false;
            }
            thread::sleep(Duration::from_micros(100));
        }

        let mut queue = match self.queue.lock() {
            Ok(q) => q,
            Err(_) => return false,
        };
        queue.push_back((index, batch));
        self.size.fetch_add(1, Ordering::Relaxed);
        true
    }

    fn pop(&self) -> Option<(usize, BatchResult<F>)> {
        let mut queue = match self.queue.lock() {
            Ok(q) => q,
            Err(_) => return None,
        };
        let result = queue.pop_front();
        if result.is_some() {
            self.size.fetch_sub(1, Ordering::Relaxed);
        }
        result
    }

    fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    fn is_empty(&self) -> bool {
        self.size.load(Ordering::Relaxed) == 0
    }
}

// =============================================================================
// Optimized Data Loader
// =============================================================================

/// Optimized data loader with prefetching and parallel loading
pub struct OptimizedDataLoader<
    F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + Send + Sync,
    D: Dataset<F> + Send + Sync + Clone + 'static,
> {
    /// The underlying dataset
    dataset: Arc<D>,
    /// Configuration
    config: OptimizedLoaderConfig,
    /// Current indices for iteration
    indices: Vec<usize>,
    /// Current position in iteration
    position: AtomicUsize,
    /// Total number of batches
    num_batches: usize,
    /// Batch cache
    cache: Option<Mutex<BatchCache<F>>>,
    /// Loading statistics
    stats: Mutex<LoadingStats>,
    /// Phantom data for float type
    _phantom: PhantomData<F>,
}

impl<
        F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + Send + Sync + 'static,
        D: Dataset<F> + Send + Sync + Clone + 'static,
    > OptimizedDataLoader<F, D>
{
    /// Create a new optimized data loader
    pub fn new(dataset: D, config: OptimizedLoaderConfig) -> Self {
        let dataset_len = dataset.len();
        let batch_size = config.batch_size;
        let drop_last = config.drop_last;

        let num_batches = if drop_last {
            dataset_len / batch_size
        } else {
            dataset_len.div_ceil(batch_size)
        };

        let indices: Vec<usize> = (0..dataset_len).collect();

        let cache = if config.cache_batches {
            Some(Mutex::new(BatchCache::new(num_batches)))
        } else {
            None
        };

        Self {
            dataset: Arc::new(dataset),
            config,
            indices,
            position: AtomicUsize::new(0),
            num_batches,
            cache,
            stats: Mutex::new(LoadingStats::default()),
            _phantom: PhantomData,
        }
    }

    /// Reset the loader for a new epoch
    pub fn reset(&mut self) {
        if self.config.shuffle {
            let mut rng = scirs2_core::random::rng();
            self.indices.shuffle(&mut rng);
        }
        self.position.store(0, Ordering::Relaxed);
    }

    /// Get the number of batches
    pub fn num_batches(&self) -> usize {
        self.num_batches
    }

    /// Get the dataset length
    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    /// Check if the loader is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get loading statistics
    pub fn stats(&self) -> LoadingStats {
        self.stats
            .lock()
            .map_or_else(|_| LoadingStats::default(), |s| s.clone())
    }

    /// Load a single batch
    fn load_batch(&self, batch_idx: usize) -> BatchResult<F> {
        let start = batch_idx * self.config.batch_size;
        let end = (start + self.config.batch_size).min(self.indices.len());

        if start >= self.indices.len() {
            return Err(NeuralError::TrainingError(
                "Batch index out of range".to_string(),
            ));
        }

        let batch_indices: Vec<usize> = self.indices[start..end].to_vec();

        if batch_indices.is_empty() {
            return Err(NeuralError::TrainingError("Empty batch".to_string()));
        }

        // Load first sample to determine shapes
        let (first_x, first_y) = self.dataset.get(batch_indices[0])?;

        // Create batch arrays
        let batch_x_shape: Vec<usize> = std::iter::once(batch_indices.len())
            .chain(first_x.shape().iter().copied())
            .collect();
        let batch_y_shape: Vec<usize> = std::iter::once(batch_indices.len())
            .chain(first_y.shape().iter().copied())
            .collect();

        let mut batch_x = Array::zeros(IxDyn(&batch_x_shape));
        let mut batch_y = Array::zeros(IxDyn(&batch_y_shape));

        // Fill batch arrays
        for (i, &idx) in batch_indices.iter().enumerate() {
            let (x, y) = self.dataset.get(idx)?;

            // Copy data into batch arrays
            let mut batch_x_slice = batch_x.slice_mut(scirs2_core::ndarray::s![i, ..]);
            batch_x_slice.assign(&x);

            let mut batch_y_slice = batch_y.slice_mut(scirs2_core::ndarray::s![i, ..]);
            batch_y_slice.assign(&y);
        }

        Ok((batch_x, batch_y))
    }

    /// Get the next batch
    pub fn next_batch(&self) -> Option<BatchResult<F>> {
        let batch_idx = self.position.fetch_add(1, Ordering::Relaxed);

        if batch_idx >= self.num_batches {
            return None;
        }

        // Check cache first
        if let Some(ref cache) = self.cache {
            if let Ok(cache_guard) = cache.lock() {
                if let Some(batch) = cache_guard.get(batch_idx) {
                    if let Ok(mut stats) = self.stats.lock() {
                        stats.cache_hits += 1;
                    }
                    return Some(Ok((batch.0.clone(), batch.1.clone())));
                }
            }
        }

        // Load batch
        let start = Instant::now();
        let result = self.load_batch(batch_idx);
        let load_time = start.elapsed();

        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.batches_loaded += 1;
            stats.samples_loaded += self.config.batch_size.min(
                self.indices
                    .len()
                    .saturating_sub(batch_idx * self.config.batch_size),
            );
            stats.total_load_time += load_time;
            stats.avg_batch_time = stats.total_load_time / stats.batches_loaded as u32;
            stats.cache_misses += 1;
        }

        // Cache the result if enabled
        if let Some(ref cache) = self.cache {
            if let Ok(ref batch) = result {
                if let Ok(mut cache_guard) = cache.lock() {
                    cache_guard.insert(batch_idx, (batch.0.clone(), batch.1.clone()));
                }
            }
        }

        Some(result)
    }

    /// Create a prefetching iterator
    pub fn prefetch_iter(self) -> PrefetchingIterator<F, D> {
        PrefetchingIterator::new(self)
    }
}

impl<
        F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + Send + Sync + 'static,
        D: Dataset<F> + Send + Sync + Clone + 'static,
    > Iterator for OptimizedDataLoader<F, D>
{
    type Item = BatchResult<F>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch()
    }
}

// =============================================================================
// Prefetching Iterator
// =============================================================================

/// Iterator that prefetches batches in the background
pub struct PrefetchingIterator<
    F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + Send + Sync + 'static,
    D: Dataset<F> + Send + Sync + Clone + 'static,
> {
    /// The underlying loader
    loader: Arc<OptimizedDataLoader<F, D>>,
    /// Prefetch queue
    queue: Arc<PrefetchQueue<F>>,
    /// Worker thread handle
    worker_handle: Option<thread::JoinHandle<()>>,
    /// Expected next batch index
    expected_idx: usize,
    /// Buffered batches (for out-of-order delivery)
    buffer: VecDeque<(usize, BatchResult<F>)>,
}

impl<
        F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + Send + Sync + 'static,
        D: Dataset<F> + Send + Sync + Clone + 'static,
    > PrefetchingIterator<F, D>
{
    /// Create a new prefetching iterator
    fn new(loader: OptimizedDataLoader<F, D>) -> Self {
        let prefetch_size = loader.config.prefetch_size;
        let loader = Arc::new(loader);
        let queue = Arc::new(PrefetchQueue::new(prefetch_size));

        // Start prefetch worker
        let worker_loader = Arc::clone(&loader);
        let worker_queue = Arc::clone(&queue);

        let worker_handle = thread::spawn(move || {
            let mut batch_idx = 0;
            loop {
                if worker_queue.stop.load(Ordering::Relaxed) {
                    break;
                }

                if batch_idx >= worker_loader.num_batches {
                    break;
                }

                let result = worker_loader.load_batch(batch_idx);
                if !worker_queue.push(batch_idx, result) {
                    break;
                }
                batch_idx += 1;
            }
        });

        Self {
            loader,
            queue,
            worker_handle: Some(worker_handle),
            expected_idx: 0,
            buffer: VecDeque::new(),
        }
    }
}

impl<
        F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + Send + Sync + 'static,
        D: Dataset<F> + Send + Sync + Clone + 'static,
    > Iterator for PrefetchingIterator<F, D>
{
    type Item = BatchResult<F>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.expected_idx >= self.loader.num_batches {
            return None;
        }

        // Check buffer first
        if let Some(pos) = self
            .buffer
            .iter()
            .position(|(idx, _)| *idx == self.expected_idx)
        {
            let (_, result) = self.buffer.remove(pos).expect("Position was just found");
            self.expected_idx += 1;
            return Some(result);
        }

        // Wait for the expected batch from prefetch queue
        let wait_start = Instant::now();
        loop {
            if let Some((idx, result)) = self.queue.pop() {
                if idx == self.expected_idx {
                    self.expected_idx += 1;

                    // Update wait time statistics
                    if let Ok(mut stats) = self.loader.stats.lock() {
                        stats.prefetch_wait_time += wait_start.elapsed();
                    }

                    return Some(result);
                } else {
                    // Buffer out-of-order batches
                    self.buffer.push_back((idx, result));
                }
            } else if self.queue.is_empty() && self.queue.stop.load(Ordering::Relaxed) {
                // No more batches coming
                return None;
            } else {
                // Wait a bit for prefetch
                thread::sleep(Duration::from_micros(10));
            }
        }
    }
}

impl<
        F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + Send + Sync + 'static,
        D: Dataset<F> + Send + Sync + Clone + 'static,
    > Drop for PrefetchingIterator<F, D>
{
    fn drop(&mut self) {
        self.queue.stop();
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
    }
}

// =============================================================================
// Automatic Batch Size Optimization
// =============================================================================

/// Result of batch size optimization
#[derive(Debug, Clone)]
pub struct BatchSizeOptimizationResult {
    /// Recommended batch size
    pub recommended_batch_size: usize,
    /// Throughput at each tested batch size
    pub throughput_results: Vec<(usize, f64)>,
    /// Memory usage at each tested batch size
    pub memory_results: Vec<(usize, usize)>,
    /// Whether memory limit was reached
    pub memory_limited: bool,
}

/// Optimizer for finding the best batch size
pub struct BatchSizeOptimizer {
    /// Minimum batch size to test
    min_batch_size: usize,
    /// Maximum batch size to test
    max_batch_size: usize,
    /// Number of warmup batches before timing
    warmup_batches: usize,
    /// Number of batches to time
    timing_batches: usize,
    /// Maximum memory to use (bytes, 0 for no limit)
    max_memory: usize,
}

impl Default for BatchSizeOptimizer {
    fn default() -> Self {
        Self {
            min_batch_size: 8,
            max_batch_size: 512,
            warmup_batches: 2,
            timing_batches: 5,
            max_memory: 0,
        }
    }
}

impl BatchSizeOptimizer {
    /// Create a new batch size optimizer
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the batch size range to test
    pub fn with_range(mut self, min: usize, max: usize) -> Self {
        self.min_batch_size = min;
        self.max_batch_size = max;
        self
    }

    /// Set the maximum memory limit
    pub fn with_max_memory(mut self, max_memory: usize) -> Self {
        self.max_memory = max_memory;
        self
    }

    /// Find the optimal batch size for a dataset
    pub fn find_optimal<
        F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + Send + Sync + 'static,
        D: Dataset<F> + Send + Sync + Clone + 'static,
    >(
        &self,
        dataset: D,
    ) -> Result<BatchSizeOptimizationResult> {
        let mut throughput_results = Vec::new();
        let mut memory_results = Vec::new();
        let mut best_throughput = 0.0;
        let mut best_batch_size = self.min_batch_size;
        let mut memory_limited = false;

        let mut batch_size = self.min_batch_size;

        while batch_size <= self.max_batch_size && batch_size <= dataset.len() {
            let config = OptimizedLoaderConfig {
                batch_size,
                shuffle: false,
                drop_last: true,
                ..Default::default()
            };

            let mut loader = OptimizedDataLoader::new(dataset.clone(), config);
            loader.reset();

            // Warmup
            for _ in 0..self.warmup_batches {
                if loader.next_batch().is_none() {
                    break;
                }
            }

            // Timing
            let start = Instant::now();
            let mut batches_processed = 0;
            let mut total_memory = 0;

            for _ in 0..self.timing_batches {
                match loader.next_batch() {
                    Some(Ok((x, y))) => {
                        batches_processed += 1;
                        total_memory += estimate_array_memory(&x) + estimate_array_memory(&y);
                    }
                    Some(Err(_)) => break,
                    None => break,
                }
            }

            if batches_processed == 0 {
                break;
            }

            let elapsed = start.elapsed().as_secs_f64();
            let samples_per_second = (batches_processed * batch_size) as f64 / elapsed;
            let avg_memory = total_memory / batches_processed;

            throughput_results.push((batch_size, samples_per_second));
            memory_results.push((batch_size, avg_memory));

            // Check memory limit
            if self.max_memory > 0 && avg_memory > self.max_memory {
                memory_limited = true;
                break;
            }

            if samples_per_second > best_throughput {
                best_throughput = samples_per_second;
                best_batch_size = batch_size;
            }

            // Increase batch size
            batch_size = (batch_size * 2).min(self.max_batch_size + 1);
        }

        Ok(BatchSizeOptimizationResult {
            recommended_batch_size: best_batch_size,
            throughput_results,
            memory_results,
            memory_limited,
        })
    }
}

// =============================================================================
// Memory-Aware Data Loader (Phase 7.1 — scirs2-core::chunking integration)
// =============================================================================

/// Configuration for memory-aware data loading.
///
/// Controls how the loader queries available memory and selects chunk/batch
/// sizes to avoid pressure on the allocator during training.
#[derive(Debug, Clone)]
pub struct MemoryAwareConfig {
    /// Target fraction of estimated available system memory to use per batch.
    /// Must be in (0.0, 1.0].  Defaults to 0.25 (use ≤ 25 % of available RAM
    /// per batch so that forward + backward passes have head-room).
    pub target_memory_fraction: f64,
    /// Per-sample byte count used for sizing calculations.  Set this to the
    /// actual element count × `size_of::<F>()` for your dataset.  If `None`,
    /// the loader will query the first sample at construction time.
    pub bytes_per_sample: Option<usize>,
    /// Hard lower bound on batch size.
    pub min_batch_size: usize,
    /// Hard upper bound on batch size.
    pub max_batch_size: usize,
    /// Whether to shuffle indices each epoch.
    pub shuffle: bool,
    /// Whether to drop the final incomplete batch.
    pub drop_last: bool,
    /// Number of batches to keep prefetched in the background queue.
    pub prefetch_ahead: usize,
}

impl Default for MemoryAwareConfig {
    fn default() -> Self {
        Self {
            target_memory_fraction: 0.25,
            bytes_per_sample: None,
            min_batch_size: 4,
            max_batch_size: 4096,
            shuffle: true,
            drop_last: false,
            prefetch_ahead: 2,
        }
    }
}

/// Estimate available system memory in bytes using a conservative heuristic.
///
/// We do not depend on any OS-specific crate here; instead we read
/// `/proc/meminfo` on Linux and fall back to a safe 512 MiB constant on other
/// platforms.  This keeps the crate 100 % pure-Rust and cross-platform.
fn estimate_available_memory_bytes() -> usize {
    // Attempt Linux /proc/meminfo first (most accurate without extra deps).
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
            // Look for "MemAvailable" which already accounts for reclaimable
            // pages and is a better predictor than "MemFree".
            for line in contents.lines() {
                if line.starts_with("MemAvailable:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<usize>() {
                            return kb * 1024;
                        }
                    }
                }
            }
        }
    }
    // Conservative fallback: assume 512 MiB is available.
    512 * 1024 * 1024
}

/// Compute a batch size that respects the `target_memory_fraction` and the
/// bounds in `config`, using `ChunkingUtils::optimal_chunk_size` from
/// `scirs2-core::chunking` as a starting point for the element-count hint.
///
/// # Arguments
/// * `dataset_len`       – number of samples in the dataset.
/// * `bytes_per_sample`  – estimated byte cost of one (input + label) sample.
/// * `config`            – `MemoryAwareConfig` with fraction and bounds.
fn compute_adaptive_batch_size(
    dataset_len: usize,
    bytes_per_sample: usize,
    config: &MemoryAwareConfig,
) -> usize {
    // ── Step 1: ask ChunkingUtils for a purely data-size-driven hint ─────────
    // We use Adaptive strategy with bounds derived from the memory config so
    // that the core chunking logic (CPU count, work-stealing, etc.) still
    // participates in the decision.
    let chunk_cfg = ChunkConfig {
        strategy: ChunkStrategy::Adaptive,
        min_chunk_size: config.min_batch_size,
        max_chunk_size: config.max_batch_size,
        ..ChunkConfig::default()
    };
    let chunking_hint = ChunkingUtils::optimal_chunk_size(dataset_len, &chunk_cfg);

    // ── Step 2: derive a memory-budget-constrained upper bound ───────────────
    let available = estimate_available_memory_bytes();
    // How many bytes may we use for a single batch?
    let budget_bytes = ((available as f64) * config.target_memory_fraction) as usize;
    // Convert to sample count, guarding against division by zero.
    let budget_samples = if bytes_per_sample > 0 {
        (budget_bytes / bytes_per_sample).max(1)
    } else {
        config.max_batch_size
    };

    // ── Step 3: reconcile the two hints ──────────────────────────────────────
    // Take the more conservative (smaller) of the two estimates, then clamp to
    // the configured bounds.
    let raw = chunking_hint.min(budget_samples);
    raw.max(config.min_batch_size).min(config.max_batch_size)
}

/// A data loader that automatically sizes its batches based on available system
/// memory and the `scirs2-core::chunking` adaptive strategy.
///
/// `MemoryAwareDataLoader` wraps an existing `Dataset` and selects a batch size
/// at construction time (and can recompute it at epoch boundaries) so that the
/// training process stays within a configurable fraction of available RAM.
///
/// The loader prefetches the next batch in a background thread while the caller
/// consumes the current one, overlapping I/O and computation.
///
/// # Type Parameters
/// * `F` – floating-point element type (e.g. `f32` or `f64`).
/// * `D` – the underlying dataset type implementing [`crate::data::Dataset`].
pub struct MemoryAwareDataLoader<
    F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + Send + Sync + 'static,
    D: Dataset<F> + Send + Sync + Clone + 'static,
> {
    /// The underlying dataset (shared with the prefetch thread).
    dataset: Arc<D>,
    /// The configuration supplied by the caller.
    config: MemoryAwareConfig,
    /// Shuffled or sequential sample indices for the current epoch.
    indices: Vec<usize>,
    /// Current read position (batch-level index, not sample-level).
    position: AtomicUsize,
    /// Batch size selected at construction (may be refreshed with
    /// `refresh_batch_size`).
    batch_size: usize,
    /// Derived total number of batches for the current epoch.
    num_batches: usize,
    /// Loading performance statistics.
    stats: Mutex<LoadingStats>,
    _phantom: PhantomData<F>,
}

impl<
        F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + Send + Sync + 'static,
        D: Dataset<F> + Send + Sync + Clone + 'static,
    > MemoryAwareDataLoader<F, D>
{
    /// Create a new `MemoryAwareDataLoader` with an automatically selected
    /// batch size.
    ///
    /// The batch size is computed once at construction from the dataset length,
    /// the per-sample byte cost, and available system memory.  Call
    /// [`Self::refresh_batch_size`] at epoch boundaries if you want it to
    /// re-examine memory pressure.
    ///
    /// # Arguments
    /// * `dataset` – the dataset to load from.
    /// * `config`  – memory-aware loading configuration.
    pub fn new_adaptive(dataset: D, config: MemoryAwareConfig) -> Result<Self> {
        let dataset_len = dataset.len();
        if dataset_len == 0 {
            return Err(NeuralError::TrainingError(
                "Cannot create MemoryAwareDataLoader from an empty dataset".to_string(),
            ));
        }

        // Resolve bytes_per_sample: use the caller's hint or probe the dataset.
        let bytes_per_sample = match config.bytes_per_sample {
            Some(b) => b,
            None => {
                // Peek at the first sample to determine array shapes.
                let (x0, y0) = dataset.get(0)?;
                (x0.len() + y0.len()) * std::mem::size_of::<F>()
            }
        };

        let batch_size = compute_adaptive_batch_size(dataset_len, bytes_per_sample, &config);

        let num_batches = if config.drop_last {
            dataset_len / batch_size
        } else {
            dataset_len.div_ceil(batch_size)
        };

        let indices: Vec<usize> = (0..dataset_len).collect();

        Ok(Self {
            dataset: Arc::new(dataset),
            config,
            indices,
            position: AtomicUsize::new(0),
            batch_size,
            num_batches,
            stats: Mutex::new(LoadingStats::default()),
            _phantom: PhantomData,
        })
    }

    /// Recompute and update the batch size based on the current system memory
    /// state.  Call this between epochs to react to changing memory pressure
    /// (e.g. other processes using RAM).
    ///
    /// Returns the new batch size.
    pub fn refresh_batch_size(&mut self) -> Result<usize> {
        let dataset_len = self.dataset.len();
        let bytes_per_sample = match self.config.bytes_per_sample {
            Some(b) => b,
            None => {
                let (x0, y0) = self.dataset.get(0)?;
                (x0.len() + y0.len()) * std::mem::size_of::<F>()
            }
        };

        let new_batch_size =
            compute_adaptive_batch_size(dataset_len, bytes_per_sample, &self.config);
        self.batch_size = new_batch_size;
        self.num_batches = if self.config.drop_last {
            dataset_len / new_batch_size
        } else {
            dataset_len.div_ceil(new_batch_size)
        };
        Ok(new_batch_size)
    }

    /// Returns the batch size currently in use.
    pub fn adaptive_batch_size(&self) -> usize {
        self.batch_size
    }

    /// Returns the number of batches in the current epoch configuration.
    pub fn num_batches(&self) -> usize {
        self.num_batches
    }

    /// Returns the number of samples in the dataset.
    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    /// Returns `true` if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.dataset.len() == 0
    }

    /// Returns a snapshot of loading performance statistics.
    pub fn stats(&self) -> LoadingStats {
        self.stats
            .lock()
            .map_or_else(|_| LoadingStats::default(), |s| s.clone())
    }

    /// Reset state for the beginning of a new epoch (optionally shuffling
    /// indices).  Does *not* refresh the batch size; call
    /// [`Self::refresh_batch_size`] explicitly if desired.
    pub fn reset(&mut self) {
        if self.config.shuffle {
            let mut rng = scirs2_core::random::rng();
            self.indices.shuffle(&mut rng);
        }
        self.position.store(0, Ordering::Relaxed);
    }

    /// Load a single batch by batch index.  This is the core loading routine
    /// shared between `next_batch` and the prefetch worker.
    fn load_batch(&self, batch_idx: usize) -> BatchResult<F> {
        let start = batch_idx * self.batch_size;
        let end = (start + self.batch_size).min(self.indices.len());

        if start >= self.indices.len() {
            return Err(NeuralError::TrainingError(
                "Batch index out of range".to_string(),
            ));
        }

        let batch_indices: Vec<usize> = self.indices[start..end].to_vec();

        if batch_indices.is_empty() {
            return Err(NeuralError::TrainingError("Empty batch".to_string()));
        }

        // Probe the first sample to determine array shapes.
        let (first_x, first_y) = self.dataset.get(batch_indices[0])?;

        let batch_x_shape: Vec<usize> = std::iter::once(batch_indices.len())
            .chain(first_x.shape().iter().copied())
            .collect();
        let batch_y_shape: Vec<usize> = std::iter::once(batch_indices.len())
            .chain(first_y.shape().iter().copied())
            .collect();

        let mut batch_x = Array::zeros(IxDyn(&batch_x_shape));
        let mut batch_y = Array::zeros(IxDyn(&batch_y_shape));

        for (i, &idx) in batch_indices.iter().enumerate() {
            let (x, y) = self.dataset.get(idx)?;
            let mut sx = batch_x.slice_mut(scirs2_core::ndarray::s![i, ..]);
            sx.assign(&x);
            let mut sy = batch_y.slice_mut(scirs2_core::ndarray::s![i, ..]);
            sy.assign(&y);
        }

        Ok((batch_x, batch_y))
    }

    /// Fetch the next batch sequentially (no background prefetch).  Returns
    /// `None` once all batches for the current epoch have been consumed.
    pub fn next_batch(&self) -> Option<BatchResult<F>> {
        let batch_idx = self.position.fetch_add(1, Ordering::Relaxed);
        if batch_idx >= self.num_batches {
            return None;
        }

        let start_time = Instant::now();
        let result = self.load_batch(batch_idx);
        let elapsed = start_time.elapsed();

        if let Ok(mut stats) = self.stats.lock() {
            stats.batches_loaded += 1;
            stats.samples_loaded += self.batch_size.min(
                self.indices
                    .len()
                    .saturating_sub(batch_idx * self.batch_size),
            );
            stats.total_load_time += elapsed;
            stats.avg_batch_time = stats.total_load_time / stats.batches_loaded as u32;
            stats.cache_misses += 1;
        }

        Some(result)
    }

    /// Consume `self` and return a [`MemoryAwarePrefetchIter`] that loads the
    /// next batch in a background thread while the caller processes the current
    /// one.
    pub fn into_prefetch_iter(self) -> MemoryAwarePrefetchIter<F, D> {
        MemoryAwarePrefetchIter::new(self)
    }
}

impl<
        F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + Send + Sync + 'static,
        D: Dataset<F> + Send + Sync + Clone + 'static,
    > Iterator for MemoryAwareDataLoader<F, D>
{
    type Item = BatchResult<F>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch()
    }
}

// =============================================================================
// Prefetching iterator for MemoryAwareDataLoader
// =============================================================================

/// Background-prefetching iterator produced by
/// [`MemoryAwareDataLoader::into_prefetch_iter`].
///
/// Internally it spawns a single worker thread that fills a bounded queue with
/// pre-loaded batches.  The consumer calls `next()` and receives batches in
/// order; if the worker is faster the consumer never waits, if the consumer is
/// faster it blocks briefly until the worker catches up.
pub struct MemoryAwarePrefetchIter<
    F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + Send + Sync + 'static,
    D: Dataset<F> + Send + Sync + Clone + 'static,
> {
    loader: Arc<MemoryAwareDataLoader<F, D>>,
    queue: Arc<PrefetchQueue<F>>,
    worker: Option<thread::JoinHandle<()>>,
    expected_idx: usize,
    /// Buffer for batches that arrived out of the expected order.
    out_of_order: VecDeque<(usize, BatchResult<F>)>,
}

impl<
        F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + Send + Sync + 'static,
        D: Dataset<F> + Send + Sync + Clone + 'static,
    > MemoryAwarePrefetchIter<F, D>
{
    fn new(loader: MemoryAwareDataLoader<F, D>) -> Self {
        let prefetch_ahead = loader.config.prefetch_ahead;
        let num_batches = loader.num_batches;
        let loader = Arc::new(loader);
        let queue = Arc::new(PrefetchQueue::new(prefetch_ahead));

        let worker_loader = Arc::clone(&loader);
        let worker_queue = Arc::clone(&queue);

        let worker = thread::spawn(move || {
            for batch_idx in 0..num_batches {
                if worker_queue.stop.load(Ordering::Relaxed) {
                    break;
                }
                let result = worker_loader.load_batch(batch_idx);
                if !worker_queue.push(batch_idx, result) {
                    break;
                }
            }
        });

        Self {
            loader,
            queue,
            worker: Some(worker),
            expected_idx: 0,
            out_of_order: VecDeque::new(),
        }
    }
}

impl<
        F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + Send + Sync + 'static,
        D: Dataset<F> + Send + Sync + Clone + 'static,
    > Iterator for MemoryAwarePrefetchIter<F, D>
{
    type Item = BatchResult<F>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.expected_idx >= self.loader.num_batches {
            return None;
        }

        // Drain the out-of-order buffer first.
        if let Some(pos) = self
            .out_of_order
            .iter()
            .position(|(idx, _)| *idx == self.expected_idx)
        {
            let (_, result) = self
                .out_of_order
                .remove(pos)
                .expect("position was just found in out_of_order buffer");
            self.expected_idx += 1;
            return Some(result);
        }

        // Block until the worker delivers the expected batch.
        let wait_start = Instant::now();
        loop {
            if let Some((idx, result)) = self.queue.pop() {
                if idx == self.expected_idx {
                    if let Ok(mut stats) = self.loader.stats.lock() {
                        stats.prefetch_wait_time += wait_start.elapsed();
                    }
                    self.expected_idx += 1;
                    return Some(result);
                }
                // Not the one we need — stash for later.
                self.out_of_order.push_back((idx, result));
            } else if self.queue.is_empty() && self.queue.stop.load(Ordering::Relaxed) {
                return None;
            } else {
                thread::sleep(Duration::from_micros(10));
            }
        }
    }
}

impl<
        F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + Send + Sync + 'static,
        D: Dataset<F> + Send + Sync + Clone + 'static,
    > Drop for MemoryAwarePrefetchIter<F, D>
{
    fn drop(&mut self) {
        self.queue.stop();
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::InMemoryDataset;

    fn create_test_dataset() -> InMemoryDataset<f64> {
        let features = Array::zeros(IxDyn(&[100, 10]));
        let labels = Array::zeros(IxDyn(&[100, 2]));
        InMemoryDataset::new(features, labels).expect("Failed to create test dataset")
    }

    #[test]
    fn test_optimized_loader_config_default() {
        let config = OptimizedLoaderConfig::default();
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.prefetch_size, 2);
        assert_eq!(config.num_workers, 0);
        assert!(!config.drop_last);
        assert!(config.shuffle);
    }

    #[test]
    fn test_optimized_dataloader_creation() {
        let dataset = create_test_dataset();
        let config = OptimizedLoaderConfig {
            batch_size: 10,
            shuffle: false,
            ..Default::default()
        };

        let loader = OptimizedDataLoader::new(dataset, config);
        assert_eq!(loader.len(), 100);
        assert_eq!(loader.num_batches(), 10);
    }

    #[test]
    fn test_optimized_dataloader_iteration() {
        let dataset = create_test_dataset();
        let config = OptimizedLoaderConfig {
            batch_size: 10,
            shuffle: false,
            drop_last: true,
            ..Default::default()
        };

        let mut loader = OptimizedDataLoader::new(dataset, config);
        loader.reset();

        let mut batch_count = 0;
        while let Some(result) = loader.next_batch() {
            let (x, y) = result.expect("Failed to load batch");
            assert_eq!(x.shape()[0], 10);
            assert_eq!(y.shape()[0], 10);
            batch_count += 1;
        }

        assert_eq!(batch_count, 10);
    }

    #[test]
    fn test_optimized_dataloader_stats() {
        let dataset = create_test_dataset();
        let config = OptimizedLoaderConfig {
            batch_size: 20,
            shuffle: false,
            ..Default::default()
        };

        let mut loader = OptimizedDataLoader::new(dataset, config);
        loader.reset();

        // Load all batches
        while loader.next_batch().is_some() {}

        let stats = loader.stats();
        assert_eq!(stats.batches_loaded, 5);
        assert_eq!(stats.samples_loaded, 100);
    }

    #[test]
    fn test_batch_cache() {
        let mut cache: BatchCache<f64> = BatchCache::new(10);

        let batch1 = (Array::zeros(IxDyn(&[5, 10])), Array::zeros(IxDyn(&[5, 2])));

        cache.insert(0, batch1.clone());

        let cached = cache.get(0);
        assert!(cached.is_some());
        assert_eq!(cached.map(|b| b.0.shape()[0]), Some(5));

        assert!(cache.get(1).is_none());

        cache.clear();
        assert!(cache.get(0).is_none());
    }

    #[test]
    fn test_prefetch_queue() {
        let queue: PrefetchQueue<f64> = PrefetchQueue::new(3);

        let batch = Ok((Array::zeros(IxDyn(&[5, 10])), Array::zeros(IxDyn(&[5, 2]))));

        assert!(queue.push(0, batch));
        assert!(!queue.is_empty());

        let popped = queue.pop();
        assert!(popped.is_some());
        assert_eq!(popped.map(|(idx, _)| idx), Some(0));

        assert!(queue.is_empty());

        queue.stop();
        // After stop, push should return false
        let batch2 = Ok((Array::zeros(IxDyn(&[5, 10])), Array::zeros(IxDyn(&[5, 2]))));
        assert!(!queue.push(1, batch2));
    }

    #[test]
    fn test_loading_stats_default() {
        let stats = LoadingStats::default();
        assert_eq!(stats.batches_loaded, 0);
        assert_eq!(stats.samples_loaded, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
    }

    #[test]
    fn test_estimate_array_memory() {
        let array: Array<f64, IxDyn> = Array::zeros(IxDyn(&[10, 20]));
        let memory = estimate_array_memory(&array);
        assert_eq!(memory, 10 * 20 * std::mem::size_of::<f64>());
    }

    #[test]
    fn test_batch_size_optimizer_default() {
        let optimizer = BatchSizeOptimizer::default();
        assert_eq!(optimizer.min_batch_size, 8);
        assert_eq!(optimizer.max_batch_size, 512);
    }

    #[test]
    fn test_batch_size_optimizer_with_range() {
        let optimizer = BatchSizeOptimizer::new()
            .with_range(16, 256)
            .with_max_memory(1024 * 1024);

        assert_eq!(optimizer.min_batch_size, 16);
        assert_eq!(optimizer.max_batch_size, 256);
        assert_eq!(optimizer.max_memory, 1024 * 1024);
    }

    #[test]
    fn test_find_optimal_batch_size() {
        let dataset = create_test_dataset();
        let optimizer = BatchSizeOptimizer::new().with_range(10, 50);

        let result = optimizer.find_optimal(dataset);
        assert!(result.is_ok());

        let result = result.expect("Optimization should succeed");
        assert!(result.recommended_batch_size >= 10);
        assert!(result.recommended_batch_size <= 50);
        assert!(!result.throughput_results.is_empty());
    }

    #[test]
    fn test_dataloader_with_caching() {
        let dataset = create_test_dataset();
        let config = OptimizedLoaderConfig {
            batch_size: 10,
            shuffle: false,
            cache_batches: true,
            ..Default::default()
        };

        let mut loader = OptimizedDataLoader::new(dataset, config);
        loader.reset();

        // First pass - all cache misses
        while loader.next_batch().is_some() {}

        let stats = loader.stats();
        assert_eq!(stats.cache_misses, 10);
        assert_eq!(stats.cache_hits, 0);
    }

    #[test]
    fn test_iterator_trait() {
        let dataset = create_test_dataset();
        let config = OptimizedLoaderConfig {
            batch_size: 25,
            shuffle: false,
            drop_last: true,
            ..Default::default()
        };

        let mut loader = OptimizedDataLoader::new(dataset, config);
        loader.reset();

        let batches: Vec<_> = loader.collect();
        assert_eq!(batches.len(), 4); // 100 / 25 = 4 batches
    }

    // -------------------------------------------------------------------------
    // MemoryAwareDataLoader tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_memory_aware_config_default() {
        let cfg = MemoryAwareConfig::default();
        assert!(
            cfg.target_memory_fraction > 0.0 && cfg.target_memory_fraction <= 1.0,
            "target_memory_fraction must be in (0, 1]"
        );
        assert!(cfg.min_batch_size >= 1);
        assert!(cfg.max_batch_size >= cfg.min_batch_size);
    }

    #[test]
    fn test_memory_aware_loader_creation() {
        let dataset = create_test_dataset();
        let config = MemoryAwareConfig {
            shuffle: false,
            drop_last: false,
            ..Default::default()
        };

        let loader = MemoryAwareDataLoader::<f64, _>::new_adaptive(dataset, config)
            .expect("loader creation must succeed");

        // Batch size must be within the configured bounds.
        let bs = loader.adaptive_batch_size();
        assert!(bs >= 4, "batch_size ({bs}) must be >= min_batch_size (4)");
        assert!(
            bs <= 4096,
            "batch_size ({bs}) must be <= max_batch_size (4096)"
        );
        // Dataset has 100 samples, so there must be at least 1 batch.
        assert!(loader.num_batches() >= 1);
        assert_eq!(loader.len(), 100);
        assert!(!loader.is_empty());
    }

    #[test]
    fn test_memory_aware_loader_iteration_all_samples() {
        let dataset = create_test_dataset();
        let config = MemoryAwareConfig {
            shuffle: false,
            drop_last: false,
            min_batch_size: 10,
            max_batch_size: 10,
            target_memory_fraction: 1.0, // doesn't matter when bounds force batch_size
            ..Default::default()
        };

        let mut loader = MemoryAwareDataLoader::<f64, _>::new_adaptive(dataset, config)
            .expect("loader creation must succeed");
        loader.reset();

        let mut total_samples = 0usize;
        let mut batch_count = 0usize;
        while let Some(result) = loader.next_batch() {
            let (x, _y) = result.expect("batch load must succeed");
            total_samples += x.shape()[0];
            batch_count += 1;
        }

        assert_eq!(total_samples, 100, "all 100 samples must be yielded");
        assert_eq!(batch_count, 10, "100 samples / batch_size 10 = 10 batches");
    }

    #[test]
    fn test_memory_aware_loader_drop_last() {
        // 100 samples, batch_size clamped to 32 by bounds.
        // drop_last=true → 3 full batches of 32, 4 samples discarded.
        let dataset = create_test_dataset();
        let config = MemoryAwareConfig {
            shuffle: false,
            drop_last: true,
            min_batch_size: 32,
            max_batch_size: 32,
            target_memory_fraction: 1.0,
            ..Default::default()
        };

        let mut loader = MemoryAwareDataLoader::<f64, _>::new_adaptive(dataset, config)
            .expect("loader creation must succeed");
        loader.reset();

        let batches: Vec<_> = loader.collect();
        assert_eq!(batches.len(), 3, "drop_last: 100/32 = 3 full batches");
    }

    #[test]
    fn test_memory_aware_loader_refresh_batch_size() {
        let dataset = create_test_dataset();
        let config = MemoryAwareConfig {
            shuffle: false,
            ..Default::default()
        };

        let mut loader = MemoryAwareDataLoader::<f64, _>::new_adaptive(dataset, config)
            .expect("loader creation must succeed");

        let new_bs = loader.refresh_batch_size().expect("refresh must succeed");
        assert!(new_bs >= loader.config.min_batch_size);
        assert!(new_bs <= loader.config.max_batch_size);
        assert_eq!(new_bs, loader.adaptive_batch_size());
    }

    #[test]
    fn test_memory_aware_loader_stats() {
        let dataset = create_test_dataset();
        let config = MemoryAwareConfig {
            shuffle: false,
            drop_last: false,
            min_batch_size: 10,
            max_batch_size: 10,
            target_memory_fraction: 1.0,
            ..Default::default()
        };

        let mut loader = MemoryAwareDataLoader::<f64, _>::new_adaptive(dataset, config)
            .expect("loader creation must succeed");
        loader.reset();

        while loader.next_batch().is_some() {}

        let stats = loader.stats();
        assert_eq!(stats.batches_loaded, 10);
        assert_eq!(stats.samples_loaded, 100);
    }

    #[test]
    fn test_memory_aware_prefetch_iter() {
        let dataset = create_test_dataset();
        let config = MemoryAwareConfig {
            shuffle: false,
            drop_last: false,
            min_batch_size: 10,
            max_batch_size: 10,
            target_memory_fraction: 1.0,
            prefetch_ahead: 2,
            ..Default::default()
        };

        let mut loader = MemoryAwareDataLoader::<f64, _>::new_adaptive(dataset, config)
            .expect("loader creation must succeed");
        loader.reset();

        let iter = loader.into_prefetch_iter();
        let batches: Vec<_> = iter.collect();

        // Each batch must be a successful result with the right shape.
        for batch_result in &batches {
            let (x, _y) = batch_result
                .as_ref()
                .expect("prefetch batch must not be an error");
            assert_eq!(x.shape()[0], 10);
        }
        assert_eq!(batches.len(), 10);
    }

    #[test]
    fn test_estimate_available_memory_is_positive() {
        let mem = estimate_available_memory_bytes();
        assert!(mem > 0, "available memory estimate must be > 0");
    }

    #[test]
    fn test_compute_adaptive_batch_size_bounds() {
        let config = MemoryAwareConfig {
            min_batch_size: 8,
            max_batch_size: 64,
            target_memory_fraction: 0.1,
            bytes_per_sample: Some(1024),
            ..Default::default()
        };
        let bs = compute_adaptive_batch_size(1000, 1024, &config);
        assert!(bs >= 8, "must respect min_batch_size");
        assert!(bs <= 64, "must respect max_batch_size");
    }
}
