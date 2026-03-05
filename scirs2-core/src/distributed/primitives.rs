//! Distributed computing primitives for SciRS2 Core
//!
//! This module provides low-level, local distributed computing primitives that
//! complement the cluster management infrastructure in the parent module.
//!
//! ## Overview
//!
//! - [`WorkQueue`] / [`WorkReceiver`]: Bounded, multi-producer channel-based task
//!   distribution with backpressure
//! - [`WorkerPool`]: Fixed-size thread pool for parallel task execution with result
//!   collection
//! - [`distributed_map`]: Parallel map over a `Vec<T>` using a worker pool
//! - [`distributed_map_reduce`]: Parallel map followed by serial reduce
//! - [`chunked_parallel_process`]: Slice-based chunked parallel processing
//! - [`ResourceMonitor`]: Heuristic CPU/memory resource availability monitor
//!
//! All primitives use `std::sync::mpsc` channels; no async runtime is required.

use crate::error::{CoreError, CoreResult, ErrorContext};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors specific to distributed-primitive operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistributedError {
    /// The work queue is full (bounded queue at capacity).
    QueueFull,
    /// The channel has been disconnected (all senders/receivers dropped).
    Disconnected,
    /// A worker thread panicked.
    WorkerPanic(String),
    /// The operation timed out.
    Timeout,
    /// Invalid argument (e.g. zero workers, zero chunk size).
    InvalidArgument(String),
    /// Internal mutex/lock poisoned.
    PoisonedLock,
}

impl std::fmt::Display for DistributedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistributedError::QueueFull => write!(f, "work queue is full"),
            DistributedError::Disconnected => write!(f, "channel disconnected"),
            DistributedError::WorkerPanic(msg) => write!(f, "worker panicked: {msg}"),
            DistributedError::Timeout => write!(f, "operation timed out"),
            DistributedError::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            DistributedError::PoisonedLock => write!(f, "mutex lock poisoned"),
        }
    }
}

impl std::error::Error for DistributedError {}

impl From<DistributedError> for CoreError {
    fn from(err: DistributedError) -> Self {
        CoreError::ComputationError(ErrorContext::new(err.to_string()))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WorkQueue / WorkReceiver
// ─────────────────────────────────────────────────────────────────────────────

/// Bounded, multi-producer work queue (sending half).
///
/// Items sent to a `WorkQueue` may be received by exactly one [`WorkReceiver`].
/// The queue is backed by [`std::sync::mpsc::sync_channel`], providing natural
/// backpressure: blocking `push` waits until a slot is free; non-blocking
/// `try_push` returns `Ok(false)` when the queue is at capacity.
///
/// # Clone behaviour
///
/// Cloning a `WorkQueue` creates an additional sender to the same channel —
/// multiple threads may push concurrently.
///
/// # Example
///
/// ```rust
/// use scirs2_core::distributed::primitives::WorkQueue;
///
/// let (queue, receiver) = WorkQueue::<i32>::new(8).expect("queue creation failed");
/// queue.push(42).expect("push failed");
/// let item = receiver.recv().expect("recv returned None");
/// assert_eq!(item, 42);
/// ```
#[derive(Debug, Clone)]
pub struct WorkQueue<T: Send + 'static> {
    sender: SyncSender<T>,
    /// Approximate live item count, maintained on every push/pop.
    len: Arc<Mutex<usize>>,
    capacity: usize,
}

/// The consumer half of a [`WorkQueue`].
///
/// There can only be one receiver per queue.  All blocking and non-blocking
/// receive methods decrement the internal item counter on success.
pub struct WorkReceiver<T: Send + 'static> {
    receiver: Receiver<T>,
    len: Arc<Mutex<usize>>,
}

impl<T: Send + 'static> WorkQueue<T> {
    /// Create a new bounded work queue with the given `capacity`.
    ///
    /// Returns `(WorkQueue, WorkReceiver)` on success.
    ///
    /// # Errors
    ///
    /// [`DistributedError::InvalidArgument`] when `capacity == 0`.
    pub fn new(capacity: usize) -> Result<(Self, WorkReceiver<T>), DistributedError> {
        if capacity == 0 {
            return Err(DistributedError::InvalidArgument(
                "capacity must be > 0".to_string(),
            ));
        }
        let (tx, rx) = mpsc::sync_channel::<T>(capacity);
        let len = Arc::new(Mutex::new(0usize));
        let queue = WorkQueue {
            sender: tx,
            len: Arc::clone(&len),
            capacity,
        };
        let receiver = WorkReceiver { receiver: rx, len };
        Ok((queue, receiver))
    }

    /// Blocking push.  Waits until a slot is available.
    ///
    /// # Errors
    ///
    /// [`DistributedError::Disconnected`] if the [`WorkReceiver`] has been dropped.
    pub fn push(&self, task: T) -> Result<(), DistributedError> {
        self.sender
            .send(task)
            .map_err(|_| DistributedError::Disconnected)?;
        if let Ok(mut guard) = self.len.lock() {
            *guard = guard.saturating_add(1);
        }
        Ok(())
    }

    /// Non-blocking push.
    ///
    /// Returns:
    /// - `Ok(true)` — the task was enqueued successfully.
    /// - `Ok(false)` — the queue was full; the task was **not** enqueued.
    ///
    /// # Errors
    ///
    /// [`DistributedError::Disconnected`] if the [`WorkReceiver`] has been dropped.
    pub fn try_push(&self, task: T) -> Result<bool, DistributedError> {
        match self.sender.try_send(task) {
            Ok(()) => {
                if let Ok(mut guard) = self.len.lock() {
                    *guard = guard.saturating_add(1);
                }
                Ok(true)
            }
            Err(TrySendError::Full(_)) => Ok(false),
            Err(TrySendError::Disconnected(_)) => Err(DistributedError::Disconnected),
        }
    }

    /// Approximate number of items currently in the queue.
    pub fn len(&self) -> usize {
        self.len.lock().map(|g| *g).unwrap_or(0)
    }

    /// Returns `true` if the queue contains no items.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Maximum capacity of this queue.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T: Send + 'static> WorkReceiver<T> {
    /// Blocking receive.  Returns `None` when all senders have been dropped.
    pub fn recv(&self) -> Option<T> {
        match self.receiver.recv() {
            Ok(item) => {
                if let Ok(mut guard) = self.len.lock() {
                    *guard = guard.saturating_sub(1);
                }
                Some(item)
            }
            Err(_) => None,
        }
    }

    /// Receive with a timeout.  Returns `None` on timeout or disconnection.
    pub fn recv_timeout(&self, timeout: Duration) -> Option<T> {
        match self.receiver.recv_timeout(timeout) {
            Ok(item) => {
                if let Ok(mut guard) = self.len.lock() {
                    *guard = guard.saturating_sub(1);
                }
                Some(item)
            }
            Err(RecvTimeoutError::Timeout) | Err(RecvTimeoutError::Disconnected) => None,
        }
    }

    /// Non-blocking receive.  Returns `None` if the queue is currently empty.
    pub fn try_recv(&self) -> Option<T> {
        match self.receiver.try_recv() {
            Ok(item) => {
                if let Ok(mut guard) = self.len.lock() {
                    *guard = guard.saturating_sub(1);
                }
                Some(item)
            }
            Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WorkerPool
// ─────────────────────────────────────────────────────────────────────────────

/// A fixed-size thread pool that processes tasks of type `T` and emits
/// results of type `R`.
///
/// Tasks are distributed to worker threads via an internal `sync_channel`;
/// results are collected through a standard `mpsc::channel`.
///
/// # Example
///
/// ```rust
/// use scirs2_core::distributed::primitives::WorkerPool;
/// use std::time::Duration;
///
/// let pool = WorkerPool::new(4, |x: i32| x * x).expect("pool creation failed");
/// pool.submit(7).expect("submit failed");
/// let result = pool.collect_result(Some(Duration::from_secs(5)));
/// assert_eq!(result, Some(49));
/// pool.shutdown();
/// ```
pub struct WorkerPool<T: Send + 'static, R: Send + 'static> {
    n_workers: usize,
    handles: Vec<JoinHandle<()>>,
    task_sender: SyncSender<Option<T>>,
    result_receiver: Receiver<R>,
}

impl<T: Send + 'static, R: Send + 'static> WorkerPool<T, R> {
    /// Create a new pool with `n_workers` threads.
    ///
    /// Each thread calls `worker_fn(task)` and sends the result to the
    /// pool's internal result channel.
    ///
    /// A `None` sentinel in the task channel signals workers to stop; the pool
    /// sends one sentinel per worker on [`shutdown`](WorkerPool::shutdown).
    ///
    /// # Errors
    ///
    /// [`DistributedError::InvalidArgument`] if `n_workers == 0`.
    pub fn new<F>(n_workers: usize, worker_fn: F) -> Result<Self, DistributedError>
    where
        F: Fn(T) -> R + Send + Clone + 'static,
    {
        if n_workers == 0 {
            return Err(DistributedError::InvalidArgument(
                "n_workers must be > 0".to_string(),
            ));
        }

        // Bounded task channel — provides natural backpressure to submitters.
        let buffer = n_workers.saturating_mul(4).max(4);
        let (task_tx, task_rx) = mpsc::sync_channel::<Option<T>>(buffer);
        let (result_tx, result_rx) = mpsc::channel::<R>();

        // Share the receiver so all workers compete for tasks (work-stealing
        // within a single process).
        let shared_rx = Arc::new(Mutex::new(task_rx));

        let mut handles = Vec::with_capacity(n_workers);
        for _ in 0..n_workers {
            let shared_rx = Arc::clone(&shared_rx);
            let result_tx = result_tx.clone();
            let fn_clone = worker_fn.clone();

            let handle = thread::spawn(move || loop {
                let task = {
                    // Hold the lock only while dequeuing, not while processing.
                    let guard = match shared_rx.lock() {
                        Ok(g) => g,
                        Err(_) => break, // poisoned lock → exit
                    };
                    match guard.recv() {
                        Ok(Some(t)) => t,
                        Ok(None) | Err(_) => break, // shutdown sentinel or disconnect
                    }
                };
                let result = fn_clone(task);
                if result_tx.send(result).is_err() {
                    break; // result channel closed → no point continuing
                }
            });
            handles.push(handle);
        }

        Ok(WorkerPool {
            n_workers,
            handles,
            task_sender: task_tx,
            result_receiver: result_rx,
        })
    }

    /// Number of worker threads in this pool.
    pub fn n_workers(&self) -> usize {
        self.n_workers
    }

    /// Submit a task to the pool.
    ///
    /// Blocks briefly if the internal task buffer (capacity `n_workers * 4`)
    /// is full.
    ///
    /// # Errors
    ///
    /// [`DistributedError::Disconnected`] if all workers have exited.
    pub fn submit(&self, task: T) -> Result<(), DistributedError> {
        self.task_sender
            .send(Some(task))
            .map_err(|_| DistributedError::Disconnected)
    }

    /// Collect one result.
    ///
    /// - `timeout = None` → block indefinitely.
    /// - `timeout = Some(d)` → return `None` if no result arrives within `d`.
    pub fn collect_result(&self, timeout: Option<Duration>) -> Option<R> {
        match timeout {
            None => self.result_receiver.recv().ok(),
            Some(d) => match self.result_receiver.recv_timeout(d) {
                Ok(r) => Some(r),
                Err(RecvTimeoutError::Timeout) | Err(RecvTimeoutError::Disconnected) => None,
            },
        }
    }

    /// Collect up to `expected` results, waiting at most `timeout` per result.
    ///
    /// Returns fewer items if individual receives time out.
    pub fn collect_all(&self, expected: usize, timeout: Duration) -> Vec<R> {
        let mut results = Vec::with_capacity(expected);
        for _ in 0..expected {
            match self.collect_result(Some(timeout)) {
                Some(r) => results.push(r),
                None => break,
            }
        }
        results
    }

    /// Signal all workers to stop and join their threads.
    ///
    /// In-flight tasks complete normally; tasks still in the buffer may be
    /// discarded once a worker picks up its sentinel.
    pub fn shutdown(self) {
        for _ in 0..self.n_workers {
            // Best-effort; ignore errors if workers have already exited.
            let _ = self.task_sender.send(None);
        }
        for handle in self.handles {
            let _ = handle.join();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Distributed map / map-reduce / chunked processing
// ─────────────────────────────────────────────────────────────────────────────

/// Parallel map over `data`, applying `map_fn` using `n_workers` threads.
///
/// Results are returned in **input order** (not completion order).
///
/// # Fallback
///
/// If `n_workers == 0`, it is silently clamped to 1.
///
/// # Example
///
/// ```rust
/// use scirs2_core::distributed::primitives::distributed_map;
///
/// let data: Vec<i32> = (1..=8).collect();
/// let squares = distributed_map(data, |x| x * x, 4);
/// assert_eq!(squares, vec![1, 4, 9, 16, 25, 36, 49, 64]);
/// ```
pub fn distributed_map<T, R, F>(data: Vec<T>, map_fn: F, n_workers: usize) -> Vec<R>
where
    T: Send + 'static,
    R: Send + 'static,
    F: Fn(T) -> R + Send + Clone + 'static,
{
    let workers = n_workers.max(1);
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }

    // We tag each item with its original index so we can reorder the output.
    let pool: WorkerPool<(usize, T), (usize, R)> =
        WorkerPool::new(workers, move |(idx, item)| (idx, map_fn(item))).unwrap_or_else(|_| {
            // Unreachable: workers >= 1 guarantees success.
            panic!("internal error: WorkerPool::new failed with workers >= 1")
        });

    for (idx, item) in data.into_iter().enumerate() {
        if pool.submit((idx, item)).is_err() {
            break; // workers exited early — stop submitting
        }
    }

    // Collect with a generous per-result timeout; in practice results arrive
    // quickly but we avoid hanging forever on a misbehaving function.
    let raw = pool.collect_all(n, Duration::from_secs(120));
    pool.shutdown();

    // Reorder to match input order.
    let mut results: Vec<Option<R>> = (0..n).map(|_| None).collect();
    for (idx, result) in raw {
        if idx < results.len() {
            results[idx] = Some(result);
        }
    }

    results.into_iter().flatten().collect()
}

/// Parallel map followed by serial reduce.
///
/// The map phase distributes `map_fn` across `n_workers` threads; reduction
/// is performed sequentially on the collected (ordered) results.
///
/// # Example
///
/// ```rust
/// use scirs2_core::distributed::primitives::distributed_map_reduce;
///
/// let data: Vec<i32> = (1..=100).collect();
/// let sum = distributed_map_reduce(data, |x| x as i64, |acc, r| acc + r, 0i64, 4);
/// assert_eq!(sum, 5050i64);
/// ```
pub fn distributed_map_reduce<T, R, S, F, G>(
    data: Vec<T>,
    map_fn: F,
    reduce_fn: G,
    initial: S,
    n_workers: usize,
) -> S
where
    T: Send + 'static,
    R: Send + 'static,
    S: Send + Clone + 'static,
    F: Fn(T) -> R + Send + Clone + 'static,
    G: Fn(S, R) -> S + Send + Clone + 'static,
{
    let mapped = distributed_map(data, map_fn, n_workers);
    mapped.into_iter().fold(initial, reduce_fn)
}

/// Parallel processing of `data` divided into `chunk_size` slices.
///
/// Each chunk is processed by `process_fn` on one of `n_workers` threads.
/// The flat results from all chunks are concatenated in input order.
///
/// # Type constraints
///
/// `T` must implement `Clone` because each chunk is cloned into an owned
/// `Arc<Vec<T>>` before being sent to a worker thread.
///
/// # Argument clamping
///
/// - `chunk_size == 0` is clamped to 1.
/// - `n_workers == 0` is clamped to 1.
///
/// # Example
///
/// ```rust
/// use scirs2_core::distributed::primitives::chunked_parallel_process;
///
/// let data: Vec<i32> = (1..=12).collect();
/// let doubled = chunked_parallel_process(
///     &data,
///     |chunk| chunk.iter().map(|&x| x * 2).collect(),
///     4,
///     3,
/// );
/// assert_eq!(doubled, vec![2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]);
/// ```
pub fn chunked_parallel_process<T, R, F>(
    data: &[T],
    process_fn: F,
    chunk_size: usize,
    n_workers: usize,
) -> Vec<R>
where
    T: Send + Sync + Clone + 'static,
    R: Send + 'static,
    F: Fn(&[T]) -> Vec<R> + Send + Clone + 'static,
{
    let effective_chunk = chunk_size.max(1);
    let effective_workers = n_workers.max(1);

    if data.is_empty() {
        return Vec::new();
    }

    // Build owned Vec chunks wrapped in Arc so they can be sent to workers.
    // We clone the slice data once per chunk; this is necessary because we need
    // 'static ownership to move into threads.
    let chunks: Vec<Arc<Vec<T>>> = data
        .chunks(effective_chunk)
        .map(|c| Arc::new(c.to_vec()))
        .collect();

    let n_chunks = chunks.len();

    // Workers receive (chunk_index, Arc<Vec<T>>) and return (chunk_index, Vec<R>).
    type TaskItem<T> = (usize, Arc<Vec<T>>);
    type ResultItem<R> = (usize, Vec<R>);

    let pool: WorkerPool<TaskItem<T>, ResultItem<R>> =
        WorkerPool::new(effective_workers, move |task: TaskItem<T>| {
            let (idx, chunk) = task;
            (idx, process_fn(&chunk))
        })
        .unwrap_or_else(|_| panic!("internal error: WorkerPool::new failed with workers >= 1"));

    for (idx, chunk) in chunks.into_iter().enumerate() {
        if pool.submit((idx, chunk)).is_err() {
            break;
        }
    }

    let raw = pool.collect_all(n_chunks, Duration::from_secs(120));
    pool.shutdown();

    // Reorder to match input order.
    let mut results: Vec<Option<Vec<R>>> = (0..n_chunks).map(|_| None).collect();
    for (idx, chunk_result) in raw {
        if idx < results.len() {
            results[idx] = Some(chunk_result);
        }
    }

    results.into_iter().flatten().flatten().collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// ResourceMonitor
// ─────────────────────────────────────────────────────────────────────────────

/// Heuristic resource availability monitor.
///
/// Uses [`std::thread::available_parallelism`] to detect the logical CPU count
/// and applies simple thresholds to calculate recommended parallelism.
///
/// This is a **local** resource monitor; it does not query remote cluster
/// nodes.  For cluster-level resource management see
/// [`super::ClusterManager`].
///
/// # Example
///
/// ```rust
/// use scirs2_core::distributed::primitives::ResourceMonitor;
///
/// let monitor = ResourceMonitor::new(0.8, 1_000_000_000);
/// let workers = monitor.available_workers();
/// assert!(workers >= 1);
/// let chunk = monitor.recommended_chunk_size(1_000_000);
/// assert!(chunk >= 64);
/// assert!(monitor.can_submit());
/// ```
#[derive(Debug, Clone)]
pub struct ResourceMonitor {
    /// Fraction of logical CPUs to use (0.0–1.0).
    cpu_threshold: f64,
    /// Memory threshold in bytes (reserved for future real-time monitoring).
    memory_threshold: usize,
    /// Cached logical CPU count from `available_parallelism`.
    logical_cpus: usize,
}

impl ResourceMonitor {
    /// Create a new `ResourceMonitor`.
    ///
    /// - `cpu_threshold`: fraction of CPUs (0.0–1.0) to allocate.
    ///   Values outside `[0, 1]` are clamped.
    /// - `memory_threshold`: maximum estimated memory usage (bytes) before
    ///   `can_submit` would return `false` in a future implementation.
    pub fn new(cpu_threshold: f64, memory_threshold: usize) -> Self {
        let logical_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        ResourceMonitor {
            cpu_threshold: cpu_threshold.clamp(0.0, 1.0),
            memory_threshold,
            logical_cpus,
        }
    }

    /// Number of logical CPUs detected on this machine.
    pub fn logical_cpu_count(&self) -> usize {
        self.logical_cpus
    }

    /// Recommended number of worker threads.
    ///
    /// Returns `max(1, floor(cpu_threshold × logical_cpus))`.
    pub fn available_workers(&self) -> usize {
        let n = (self.cpu_threshold * self.logical_cpus as f64).floor() as usize;
        n.max(1)
    }

    /// Recommended chunk size for `total_work` items given the current worker count.
    ///
    /// Targets 4 chunks per worker (good for work-stealing granularity) with a
    /// minimum of 64 items per chunk.
    pub fn recommended_chunk_size(&self, total_work: usize) -> usize {
        if total_work == 0 {
            return 64;
        }
        let workers = self.available_workers();
        let target_chunks = workers.saturating_mul(4).max(1);
        (total_work / target_chunks).max(64)
    }

    /// Returns `true` when the system appears to have capacity headroom.
    ///
    /// The current implementation always returns `true` because querying
    /// OS-level CPU and memory metrics portably and without unsafe code
    /// requires platform-specific system calls beyond scope here.  The API
    /// is provided for future enhancement and testing.
    pub fn can_submit(&self) -> bool {
        true
    }

    /// CPU utilisation threshold (0.0–1.0).
    pub fn cpu_threshold(&self) -> f64 {
        self.cpu_threshold
    }

    /// Memory threshold in bytes.
    pub fn memory_threshold(&self) -> usize {
        self.memory_threshold
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Extension trait
// ─────────────────────────────────────────────────────────────────────────────

/// Extension trait that adds distributed-aware parallel processing to slices.
pub trait DistributedSliceExt<T> {
    /// Divide the slice into chunks, process each chunk in parallel, and
    /// concatenate the results in input order.
    fn distributed_process<R, F>(
        &self,
        process_fn: F,
        chunk_size: usize,
        n_workers: usize,
    ) -> Vec<R>
    where
        T: Send + Sync + Clone + 'static,
        R: Send + 'static,
        F: Fn(&[T]) -> Vec<R> + Send + Clone + 'static;
}

impl<T: Send + Sync + Clone + 'static> DistributedSliceExt<T> for [T] {
    fn distributed_process<R, F>(
        &self,
        process_fn: F,
        chunk_size: usize,
        n_workers: usize,
    ) -> Vec<R>
    where
        R: Send + 'static,
        F: Fn(&[T]) -> Vec<R> + Send + Clone + 'static,
    {
        chunked_parallel_process(self, process_fn, chunk_size, n_workers)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CoreResult-returning wrappers
// ─────────────────────────────────────────────────────────────────────────────

/// Like [`distributed_map`] but returns [`CoreResult`]`<Vec<R>>`.
///
/// This wrapper always succeeds unless `n_workers == 0` (which is silently
/// clamped to 1); it exists for consistent error-handling at call sites.
pub fn try_distributed_map<T, R, F>(data: Vec<T>, map_fn: F, n_workers: usize) -> CoreResult<Vec<R>>
where
    T: Send + 'static,
    R: Send + 'static,
    F: Fn(T) -> R + Send + Clone + 'static,
{
    Ok(distributed_map(data, map_fn, n_workers))
}

/// Like [`distributed_map_reduce`] but returns [`CoreResult`]`<S>`.
///
/// This wrapper always succeeds; it exists for consistent error-handling.
pub fn try_distributed_map_reduce<T, R, S, F, G>(
    data: Vec<T>,
    map_fn: F,
    reduce_fn: G,
    initial: S,
    n_workers: usize,
) -> CoreResult<S>
where
    T: Send + 'static,
    R: Send + 'static,
    S: Send + Clone + 'static,
    F: Fn(T) -> R + Send + Clone + 'static,
    G: Fn(S, R) -> S + Send + Clone + 'static,
{
    Ok(distributed_map_reduce(
        data, map_fn, reduce_fn, initial, n_workers,
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests (≥ 20 comprehensive cases)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    // ── WorkQueue ─────────────────────────────────────────────────────────────

    #[test]
    fn test_work_queue_basic_push_recv() {
        let (queue, receiver) = WorkQueue::<i32>::new(8).expect("queue creation failed");
        queue.push(42).expect("push failed");
        let item = receiver.recv().expect("recv returned None");
        assert_eq!(item, 42);
    }

    #[test]
    fn test_work_queue_zero_capacity_is_error() {
        let result = WorkQueue::<i32>::new(0);
        assert!(matches!(result, Err(DistributedError::InvalidArgument(_))));
    }

    #[test]
    fn test_work_queue_try_push_full() {
        let (queue, _receiver) = WorkQueue::<i32>::new(2).expect("queue creation failed");
        let r1 = queue.try_push(1).expect("try_push 1 failed");
        let r2 = queue.try_push(2).expect("try_push 2 failed");
        let r3 = queue
            .try_push(3)
            .expect("try_push 3 should return false when full");
        assert!(r1, "first slot should be accepted");
        assert!(r2, "second slot should be accepted");
        assert!(!r3, "queue is full — should return false");
    }

    #[test]
    fn test_work_queue_len_and_is_empty() {
        let (queue, receiver) = WorkQueue::<u64>::new(16).expect("queue creation failed");
        assert!(queue.is_empty(), "newly created queue must be empty");
        queue.push(10).expect("push 10 failed");
        queue.push(20).expect("push 20 failed");
        assert_eq!(queue.len(), 2, "queue len should be 2 after two pushes");
        receiver.recv();
        assert_eq!(queue.len(), 1, "queue len should be 1 after one recv");
    }

    #[test]
    fn test_work_queue_capacity() {
        let (queue, _rx) = WorkQueue::<()>::new(32).expect("queue creation failed");
        assert_eq!(queue.capacity(), 32);
    }

    #[test]
    fn test_work_queue_disconnected_on_receiver_drop() {
        let (queue, receiver) = WorkQueue::<i32>::new(4).expect("queue creation failed");
        drop(receiver);
        let err = queue.push(1);
        assert!(matches!(err, Err(DistributedError::Disconnected)));
    }

    #[test]
    fn test_work_receiver_recv_timeout_returns_none() {
        let (_queue, receiver) = WorkQueue::<i32>::new(4).expect("queue creation failed");
        let result = receiver.recv_timeout(Duration::from_millis(20));
        assert!(
            result.is_none(),
            "should time out with nothing in the queue"
        );
    }

    #[test]
    fn test_work_receiver_try_recv_empty() {
        let (_queue, receiver) = WorkQueue::<i32>::new(4).expect("queue creation failed");
        assert!(
            receiver.try_recv().is_none(),
            "try_recv on empty queue must return None"
        );
    }

    #[test]
    fn test_work_queue_multiple_producers() {
        let (queue, receiver) = WorkQueue::<i32>::new(128).expect("queue creation failed");
        let ranges: Vec<(i32, i32)> = vec![(0, 10), (10, 20), (20, 30)];
        let mut handles = Vec::new();
        for (start, end) in ranges {
            let q = queue.clone();
            handles.push(std::thread::spawn(move || {
                for i in start..end {
                    q.push(i).expect("push failed");
                }
            }));
        }
        for h in handles {
            h.join().expect("producer thread panicked");
        }
        // Drain all items
        let mut items: Vec<i32> = Vec::new();
        while let Some(x) = receiver.try_recv() {
            items.push(x);
        }
        // Flush any remaining items from the channel after try_recv emptied the counter.
        // (try_recv may miss a few items that arrived just after the counter read)
        // Give a short grace period.
        while let Some(x) = receiver.recv_timeout(Duration::from_millis(10)) {
            items.push(x);
        }
        assert_eq!(
            items.len(),
            30,
            "expected 30 items from three producers, got {}",
            items.len()
        );
        items.sort_unstable();
        assert_eq!(items, (0..30).collect::<Vec<_>>());
    }

    // ── WorkerPool ────────────────────────────────────────────────────────────

    #[test]
    fn test_worker_pool_basic_square() {
        let pool = WorkerPool::new(2, |x: i32| x * 2).expect("pool creation failed");
        pool.submit(3).expect("submit failed");
        pool.submit(7).expect("submit failed");
        let mut results = pool.collect_all(2, Duration::from_secs(5));
        results.sort_unstable();
        assert_eq!(results, vec![6, 14]);
        pool.shutdown();
    }

    #[test]
    fn test_worker_pool_zero_workers_is_error() {
        let result = WorkerPool::<i32, i32>::new(0, |x| x);
        assert!(
            matches!(result, Err(DistributedError::InvalidArgument(_))),
            "zero workers must be rejected"
        );
    }

    #[test]
    fn test_worker_pool_collect_result_none_on_timeout() {
        let pool = WorkerPool::<i32, i32>::new(1, |x| x).expect("pool creation failed");
        let result = pool.collect_result(Some(Duration::from_millis(30)));
        assert!(result.is_none(), "nothing submitted → should timeout");
        pool.shutdown();
    }

    #[test]
    fn test_worker_pool_accumulates_correct_sum() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);
        let pool = WorkerPool::new(4, move |x: usize| {
            counter_clone.fetch_add(x, Ordering::Relaxed);
            x
        })
        .expect("pool creation failed");

        for i in 0..20 {
            pool.submit(i).expect("submit failed");
        }
        let _ = pool.collect_all(20, Duration::from_secs(5));
        pool.shutdown();

        // Sum 0..20 = 190
        assert_eq!(counter.load(Ordering::Relaxed), 190);
    }

    #[test]
    fn test_worker_pool_n_workers() {
        let pool = WorkerPool::new(7, |x: i32| x).expect("pool creation failed");
        assert_eq!(pool.n_workers(), 7);
        pool.shutdown();
    }

    // ── distributed_map ───────────────────────────────────────────────────────

    #[test]
    fn test_distributed_map_empty_input() {
        let result = distributed_map(Vec::<i32>::new(), |x| x * x, 4);
        assert!(result.is_empty());
    }

    #[test]
    fn test_distributed_map_preserves_order() {
        let data: Vec<i32> = (1..=16).collect();
        let result = distributed_map(data, |x| x * x, 4);
        let expected: Vec<i32> = (1..=16).map(|x| x * x).collect();
        assert_eq!(
            result, expected,
            "distributed_map must preserve input order"
        );
    }

    #[test]
    fn test_distributed_map_single_worker() {
        let data: Vec<String> = (0..10).map(|i| format!("item-{i}")).collect();
        let lens = distributed_map(data.clone(), |s| s.len(), 1);
        let expected: Vec<usize> = data.iter().map(|s| s.len()).collect();
        assert_eq!(lens, expected);
    }

    #[test]
    fn test_distributed_map_zero_workers_clamped_to_one() {
        let data: Vec<i32> = (0..5).collect();
        // n_workers=0 must not panic; it is clamped to 1 internally.
        let result = distributed_map(data, |x| x + 1, 0);
        assert_eq!(result, vec![1, 2, 3, 4, 5]);
    }

    // ── distributed_map_reduce ────────────────────────────────────────────────

    #[test]
    fn test_distributed_map_reduce_sum() {
        let data: Vec<i32> = (1..=100).collect();
        let sum = distributed_map_reduce(data, |x| x as i64, |acc, r| acc + r, 0i64, 4);
        assert_eq!(sum, 5050, "sum 1..100 must equal 5050");
    }

    #[test]
    fn test_distributed_map_reduce_factorial_small() {
        let data: Vec<u64> = (1..=5).collect();
        let product = distributed_map_reduce(data, |x| x, |acc, r| acc * r, 1u64, 2);
        assert_eq!(product, 120, "5! = 120");
    }

    #[test]
    fn test_distributed_map_reduce_string_concat_order() {
        let data: Vec<i32> = (0..5).collect();
        let result = distributed_map_reduce(
            data,
            |x| x.to_string(),
            |mut acc, r| {
                acc.push_str(&r);
                acc
            },
            String::new(),
            2,
        );
        // Map phase orders by index; reduce is serial over ordered results.
        assert_eq!(result, "01234");
    }

    // ── chunked_parallel_process ──────────────────────────────────────────────

    #[test]
    fn test_chunked_parallel_process_basic() {
        let data: Vec<i32> = (1..=12).collect();
        let doubled =
            chunked_parallel_process(&data, |chunk| chunk.iter().map(|&x| x * 2).collect(), 4, 3);
        assert_eq!(doubled, vec![2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]);
    }

    #[test]
    fn test_chunked_parallel_process_empty_input() {
        let data: Vec<f64> = Vec::new();
        let result = chunked_parallel_process(&data, |c| c.to_vec(), 4, 2);
        assert!(result.is_empty());
    }

    #[test]
    fn test_chunked_parallel_process_chunk_larger_than_data() {
        let data: Vec<i32> = (0..5).collect();
        let result = chunked_parallel_process(
            &data,
            |chunk| chunk.iter().map(|&x| x + 1).collect(),
            100,
            2,
        );
        assert_eq!(result, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_chunked_parallel_process_chunk_size_one() {
        let data: Vec<i32> = (0..10).collect();
        let result = chunked_parallel_process(&data, |chunk| vec![chunk[0] * 3], 1, 4);
        let expected: Vec<i32> = (0..10).map(|x| x * 3).collect();
        assert_eq!(result, expected);
    }

    // ── ResourceMonitor ───────────────────────────────────────────────────────

    #[test]
    fn test_resource_monitor_available_workers_full_threshold() {
        let monitor = ResourceMonitor::new(1.0, usize::MAX);
        let workers = monitor.available_workers();
        assert!(workers >= 1);
        assert_eq!(workers, monitor.logical_cpu_count());
    }

    #[test]
    fn test_resource_monitor_half_threshold() {
        let monitor = ResourceMonitor::new(0.5, usize::MAX);
        let cpus = monitor.logical_cpu_count();
        let workers = monitor.available_workers();
        let expected = ((0.5_f64 * cpus as f64).floor() as usize).max(1);
        assert_eq!(workers, expected);
    }

    #[test]
    fn test_resource_monitor_zero_threshold_still_one_worker() {
        let monitor = ResourceMonitor::new(0.0, 0);
        assert_eq!(
            monitor.available_workers(),
            1,
            "must always return at least 1"
        );
    }

    #[test]
    fn test_resource_monitor_recommended_chunk_size() {
        let monitor = ResourceMonitor::new(1.0, usize::MAX);
        let chunk = monitor.recommended_chunk_size(1_000_000);
        assert!(chunk >= 64, "chunk must be at least 64");
        let chunk_zero = monitor.recommended_chunk_size(0);
        assert_eq!(chunk_zero, 64, "zero total work → default 64");
    }

    #[test]
    fn test_resource_monitor_can_submit() {
        let monitor = ResourceMonitor::new(0.8, 1_000_000_000);
        assert!(monitor.can_submit());
    }

    #[test]
    fn test_resource_monitor_accessors() {
        let monitor = ResourceMonitor::new(0.75, 500_000);
        assert!((monitor.cpu_threshold() - 0.75).abs() < 1e-9);
        assert_eq!(monitor.memory_threshold(), 500_000);
    }

    // ── DistributedSliceExt ───────────────────────────────────────────────────

    #[test]
    fn test_distributed_slice_ext_double() {
        let data: Vec<i32> = (1..=20).collect();
        let result =
            data.distributed_process(|chunk| chunk.iter().map(|&x| x as i64 * 2).collect(), 5, 4);
        let expected: Vec<i64> = (1..=20).map(|x| x as i64 * 2).collect();
        assert_eq!(result, expected);
    }

    // ── CoreResult wrappers ───────────────────────────────────────────────────

    #[test]
    fn test_try_distributed_map() {
        let data: Vec<i32> = (1..=5).collect();
        let result = try_distributed_map(data, |x| x + 10, 2).expect("try_distributed_map failed");
        assert_eq!(result, vec![11, 12, 13, 14, 15]);
    }

    #[test]
    fn test_try_distributed_map_reduce() {
        let data: Vec<i32> = (1..=10).collect();
        let result = try_distributed_map_reduce(data, |x| x as u32, |a, b| a + b, 0u32, 2)
            .expect("try_distributed_map_reduce failed");
        assert_eq!(result, 55, "sum 1..10 = 55");
    }

    // ── DistributedError ──────────────────────────────────────────────────────

    #[test]
    fn test_distributed_error_display_messages() {
        let cases: &[(DistributedError, &str)] = &[
            (DistributedError::QueueFull, "full"),
            (DistributedError::Disconnected, "disconnect"),
            (DistributedError::Timeout, "timed out"),
            (
                DistributedError::InvalidArgument("bad arg".into()),
                "bad arg",
            ),
            (DistributedError::WorkerPanic("boom".into()), "boom"),
            (DistributedError::PoisonedLock, "poison"),
        ];
        for (err, expected_fragment) in cases {
            let msg = err.to_string();
            assert!(
                msg.contains(expected_fragment),
                "error '{msg}' should contain '{expected_fragment}'"
            );
        }
    }

    #[test]
    fn test_distributed_error_into_core_error() {
        let err: CoreError = DistributedError::QueueFull.into();
        // Verify it converts without panicking and produces a non-empty string.
        assert!(!err.to_string().is_empty());
    }
}
