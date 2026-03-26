//! Async-style allocation queue with priority scheduling and memory pressure callbacks.
//!
//! `AsyncPool` wraps an `ArenaAllocator` and adds:
//!
//! * A priority queue of pending allocation requests (simulated asynchronously —
//!   processing is explicit and synchronous, but the API mirrors typical GPU
//!   stream-submission patterns).
//! * Memory-pressure callbacks fired when the fragmentation ratio exceeds a
//!   registered threshold.
//! * Simple throughput and latency tracking for profiling.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::time::Instant;

use super::arena::ArenaAllocator;
use super::types::{
    AllocError, AllocationId, AllocationStats, AsyncAllocRequest, PoolConfig, RequestHandle,
};

// ---------------------------------------------------------------------------
// Priority-queue wrapper
// ---------------------------------------------------------------------------

/// Internal entry stored in the `BinaryHeap`.
///
/// We want *higher* priority values to be dequeued first; `BinaryHeap` is a
/// max-heap, so we can rely on that directly.
#[derive(Debug)]
struct PendingRequest {
    priority: u8,
    /// Monotonically-increasing sequence number used to break ties (FIFO within
    /// the same priority level — lower sequence number wins, so we invert the
    /// comparison).
    sequence: u64,
    handle: RequestHandle,
    request: AsyncAllocRequest,
}

impl PartialEq for PendingRequest {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.sequence == other.sequence
    }
}

impl Eq for PendingRequest {}

impl PartialOrd for PendingRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PendingRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first; ties broken by lower sequence number first.
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

// ---------------------------------------------------------------------------
// Result tracking
// ---------------------------------------------------------------------------

/// Result state for a submitted request.
#[derive(Debug)]
enum RequestState {
    Pending,
    Ready(AllocationId),
    Failed(AllocError),
}

// ---------------------------------------------------------------------------
// Pressure callback entry
// ---------------------------------------------------------------------------

struct PressureCallback {
    threshold: f64,
    callback: Box<dyn Fn(f64) + Send>,
}

// ---------------------------------------------------------------------------
// Latency tracking (ring buffer of recent timings)
// ---------------------------------------------------------------------------

const LATENCY_WINDOW: usize = 256;

struct LatencyTracker {
    samples_ns: Vec<u64>,
    head: usize,
    count: usize,
    total_ops: u64,
    window_start: Instant,
    window_ops: u64,
}

impl LatencyTracker {
    fn new() -> Self {
        Self {
            samples_ns: vec![0u64; LATENCY_WINDOW],
            head: 0,
            count: 0,
            total_ops: 0,
            window_start: Instant::now(),
            window_ops: 0,
        }
    }

    fn record(&mut self, latency_ns: u64) {
        self.samples_ns[self.head] = latency_ns;
        self.head = (self.head + 1) % LATENCY_WINDOW;
        if self.count < LATENCY_WINDOW {
            self.count += 1;
        }
        self.total_ops += 1;
        self.window_ops += 1;
    }

    fn avg_latency_ns(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let sum: u64 = self.samples_ns[..self.count].iter().sum();
        sum as f64 / self.count as f64
    }

    fn throughput_ops_per_sec(&mut self) -> f64 {
        let elapsed = self.window_start.elapsed().as_secs_f64();
        if elapsed < 1e-9 {
            return 0.0;
        }
        let ops = self.window_ops;
        // Reset window every call.
        self.window_start = Instant::now();
        self.window_ops = 0;
        ops as f64 / elapsed
    }
}

// ---------------------------------------------------------------------------
// AsyncPool
// ---------------------------------------------------------------------------

/// A priority-queue–driven allocation pool with memory pressure callbacks.
pub struct AsyncPool {
    /// Underlying arena allocator.
    arena: ArenaAllocator,
    /// Priority queue of pending requests.
    pending: BinaryHeap<PendingRequest>,
    /// Results for all submitted requests.
    results: HashMap<usize, RequestState>,
    /// Maximum capacity of the pending queue.
    queue_capacity: usize,
    /// Monotonically increasing sequence counter for FIFO tie-breaking.
    sequence: u64,
    /// Next handle id.
    next_handle: usize,
    /// Memory-pressure callbacks.
    pressure_callbacks: Vec<PressureCallback>,
    /// Latency / throughput tracking.
    latency: LatencyTracker,
}

impl AsyncPool {
    /// Create a new `AsyncPool` from the given configuration.
    pub fn new(config: PoolConfig) -> Self {
        let queue_capacity = config.async_queue_size;
        let arena = ArenaAllocator::new(config);
        Self {
            arena,
            pending: BinaryHeap::new(),
            results: HashMap::new(),
            queue_capacity,
            sequence: 0,
            next_handle: 0,
            pressure_callbacks: Vec::new(),
            latency: LatencyTracker::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Queue submission
    // -----------------------------------------------------------------------

    /// Submit an allocation request to the queue.
    ///
    /// Returns `Err(AllocError::PoolFull)` if the queue is at capacity.
    pub fn enqueue(&mut self, req: AsyncAllocRequest) -> Result<RequestHandle, AllocError> {
        if self.pending.len() >= self.queue_capacity {
            return Err(AllocError::PoolFull);
        }

        let handle = RequestHandle(self.next_handle);
        self.next_handle += 1;

        let entry = PendingRequest {
            priority: req.priority,
            sequence: self.sequence,
            handle,
            request: req,
        };
        self.sequence += 1;
        self.pending.push(entry);
        self.results.insert(handle.0, RequestState::Pending);

        Ok(handle)
    }

    // -----------------------------------------------------------------------
    // Queue processing
    // -----------------------------------------------------------------------

    /// Process up to `max_allocations` pending requests, highest-priority first.
    ///
    /// Returns a `Vec` of `(RequestHandle, AllocationId)` pairs for every
    /// request that completed successfully.  Failed requests are recorded in
    /// the result map and can be queried via `get_result`.
    pub fn process_queue(&mut self, max_allocations: usize) -> Vec<(RequestHandle, AllocationId)> {
        let mut completed = Vec::new();

        for _ in 0..max_allocations {
            let entry = match self.pending.pop() {
                Some(e) => e,
                None => break,
            };

            let t0 = Instant::now();
            let result = self
                .arena
                .alloc(entry.request.size, entry.request.alignment);
            let latency_ns = t0.elapsed().as_nanos() as u64;
            self.latency.record(latency_ns);

            match result {
                Ok(id) => {
                    self.results.insert(entry.handle.0, RequestState::Ready(id));
                    completed.push((entry.handle, id));
                }
                Err(e) => {
                    self.results.insert(entry.handle.0, RequestState::Failed(e));
                }
            }
        }

        self.check_pressure();
        completed
    }

    // -----------------------------------------------------------------------
    // Result inspection
    // -----------------------------------------------------------------------

    /// Return `true` if the given request has been processed (successfully or not).
    pub fn is_ready(&self, handle: RequestHandle) -> bool {
        matches!(
            self.results.get(&handle.0),
            Some(RequestState::Ready(_)) | Some(RequestState::Failed(_))
        )
    }

    /// Return the `AllocationId` for a successfully completed request, or `None`
    /// if the request is still pending or failed.
    pub fn get_result(&self, handle: RequestHandle) -> Option<AllocationId> {
        match self.results.get(&handle.0) {
            Some(RequestState::Ready(id)) => Some(*id),
            _ => None,
        }
    }

    /// Return the error for a failed request, or `None` if pending / successful.
    pub fn get_error(&self, handle: RequestHandle) -> Option<&AllocError> {
        match self.results.get(&handle.0) {
            Some(RequestState::Failed(e)) => Some(e),
            _ => None,
        }
    }

    // -----------------------------------------------------------------------
    // Memory pressure
    // -----------------------------------------------------------------------

    /// Register a callback to be fired when the pool fragmentation exceeds `threshold`.
    ///
    /// The callback receives the current fragmentation score.
    pub fn register_pressure_callback(&mut self, threshold: f64, cb: Box<dyn Fn(f64) + Send>) {
        self.pressure_callbacks.push(PressureCallback {
            threshold,
            callback: cb,
        });
    }

    /// Evaluate all registered pressure thresholds and fire callbacks if exceeded.
    pub fn check_pressure(&self) {
        let stats = self.arena.stats();
        let score = stats.fragmentation;

        for entry in &self.pressure_callbacks {
            if score > entry.threshold {
                (entry.callback)(score);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Statistics & profiling
    // -----------------------------------------------------------------------

    /// Return the allocation throughput in operations per second since the last call.
    pub fn throughput_ops_per_sec(&mut self) -> f64 {
        self.latency.throughput_ops_per_sec()
    }

    /// Return the average allocation latency in nanoseconds over the last
    /// `min(n_completed, 256)` operations.
    pub fn avg_alloc_latency_ns(&self) -> f64 {
        self.latency.avg_latency_ns()
    }

    /// Return a snapshot of the arena statistics.
    pub fn stats(&self) -> AllocationStats {
        self.arena.stats()
    }

    /// Return the number of pending (not-yet-processed) requests.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    // -----------------------------------------------------------------------
    // Pass-through arena access
    // -----------------------------------------------------------------------

    /// Free an allocation by id in the underlying arena.
    pub fn free(&mut self, id: AllocationId) -> Result<(), AllocError> {
        self.arena.free(id)
    }

    /// Access the underlying arena (read-only).
    pub fn arena(&self) -> &ArenaAllocator {
        &self.arena
    }

    /// Access the underlying arena (mutable).
    pub fn arena_mut(&mut self) -> &mut ArenaAllocator {
        &mut self.arena
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_pool::types::{AsyncAllocRequest, PoolConfig};
    use std::sync::{Arc, Mutex};

    fn small_pool() -> AsyncPool {
        AsyncPool::new(PoolConfig {
            total_size: 1024 * 1024, // 1 MiB
            async_queue_size: 16,
            ..Default::default()
        })
    }

    #[test]
    fn test_async_pool_enqueue() {
        let mut pool = small_pool();
        let req = AsyncAllocRequest::new(256, 5);
        let handle = pool.enqueue(req).expect("enqueue");
        assert!(
            !pool.is_ready(handle),
            "should be pending before processing"
        );
        assert_eq!(pool.pending_count(), 1);
    }

    #[test]
    fn test_async_pool_priority() {
        let mut pool = small_pool();
        // Enqueue low-priority first, then high-priority.
        let low_req = AsyncAllocRequest::new(64, 1);
        let high_req = AsyncAllocRequest::new(64, 10);
        let _low_handle = pool.enqueue(low_req).expect("enqueue low");
        let high_handle = pool.enqueue(high_req).expect("enqueue high");

        // Process only 1 request — it should be the high-priority one.
        let completed = pool.process_queue(1);
        assert_eq!(completed.len(), 1, "one request should complete");
        assert_eq!(
            completed[0].0, high_handle,
            "high-priority request should complete first"
        );
    }

    #[test]
    fn test_async_pool_process() {
        let mut pool = small_pool();
        for _ in 0..5 {
            let req = AsyncAllocRequest::new(128, 5);
            pool.enqueue(req).expect("enqueue");
        }
        let completed = pool.process_queue(5);
        assert_eq!(completed.len(), 5);
        for (handle, _id) in &completed {
            assert!(pool.is_ready(*handle));
            assert!(pool.get_result(*handle).is_some());
        }
    }

    #[test]
    fn test_pressure_callback() {
        let mut pool = small_pool();
        let fired = Arc::new(Mutex::new(false));
        let fired_clone = Arc::clone(&fired);

        // Register a callback at threshold 0.0 (always fires when fragmentation > 0).
        pool.register_pressure_callback(
            -0.1, // threshold below any possible score → always fires
            Box::new(move |_score| {
                let mut f = fired_clone.lock().expect("lock");
                *f = true;
            }),
        );

        // Allocate and free to create some fragmentation.
        let req = AsyncAllocRequest::new(64, 5);
        let handle = pool.enqueue(req).expect("enqueue");
        let completed = pool.process_queue(1);
        assert_eq!(completed.len(), 1);
        pool.free(completed[0].1).expect("free");

        // Enqueue another and process — check_pressure is called in process_queue.
        let req2 = AsyncAllocRequest::new(64, 5);
        pool.enqueue(req2).expect("enqueue 2");
        pool.process_queue(1);

        // Even if fragmentation is 0, the threshold -0.1 ensures the callback fires.
        // call check_pressure directly to be sure.
        pool.check_pressure();

        let was_fired = *fired.lock().expect("lock");
        assert!(was_fired, "pressure callback should have fired");
        let _ = handle; // suppress unused warning
    }

    #[test]
    fn test_pool_config_default() {
        let config = PoolConfig::default();
        assert_eq!(config.total_size, 64 * 1024 * 1024);
        assert_eq!(config.min_block_size, 64);
        assert_eq!(config.alignment, 256);
        assert!((config.defrag_threshold - 0.4).abs() < 1e-9);
        assert_eq!(config.async_queue_size, 1024);
    }
}
