//! Types for the parallel computation module.

/// Configuration for the parallel coordinator.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of worker threads to spawn.  Default: `4`.
    pub n_workers: usize,
    /// Whether to use SharedArrayBuffer for inter-thread data.
    /// Requires `COOP`/`COEP` response headers in browser deployments.
    /// Default: `false`.
    pub use_shared_memory: bool,
    /// Number of elements per work chunk.  Default: `1024`.
    pub chunk_size: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            n_workers: 4,
            use_shared_memory: false,
            chunk_size: 1024,
        }
    }
}

/// Synchronization primitives available for coordinating parallel workers.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncPrimitive {
    /// Mutual-exclusion lock.
    Mutex,
    /// Counting semaphore.
    Semaphore,
    /// All-parties rendezvous barrier.
    Barrier,
}

/// A message passed between the coordinator and a worker thread.
#[derive(Debug, Clone)]
pub struct WorkerMessage {
    /// Zero-based index of the worker that sent or will receive this message.
    pub worker_id: usize,
    /// Payload data.
    pub data: Vec<f64>,
    /// Operation the worker should perform.
    pub operation: WorkerOp,
}

/// Operations that can be dispatched to a worker.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerOp {
    /// Matrix multiplication.
    MatMul,
    /// Reduction (sum, max, …).
    Reduce,
    /// Element-wise map.
    Map,
    /// Sort.
    Sort,
}
