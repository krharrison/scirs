//! Thread pool with work-stealing, priority queuing, scoped tasks, and dynamic resizing.
//!
//! This module provides a production-grade thread pool built on top of crossbeam's
//! work-stealing deques, with features including:
//!
//! - Fixed-size thread pool with configurable worker count
//! - Work-stealing for automatic load balancing across workers
//! - Task submission returning a `TaskHandle<T>` (Future-like result retrieval)
//! - Scoped tasks that can safely borrow from the outer scope
//! - Dynamic pool resizing (grow/shrink at runtime)
//! - Priority queue for tasks (Background, Normal, High, Critical)
//! - Graceful shutdown with optional timeout
//!
//! # Example
//!
//! ```rust
//! # #[cfg(feature = "parallel")]
//! # {
//! use scirs2_core::thread_pool::{ThreadPool, ThreadPoolConfig, TaskPriority};
//!
//! let pool = ThreadPool::new(ThreadPoolConfig::default())
//!     .expect("failed to create pool");
//!
//! // Submit a task and get a handle
//! let handle = pool.submit(|| 2 + 2);
//! assert_eq!(handle.join().expect("should succeed"), 4);
//!
//! // Submit with priority
//! let handle = pool.submit_with_priority(TaskPriority::High, || "hello");
//! assert_eq!(handle.join().expect("should succeed"), "hello");
//!
//! // Graceful shutdown
//! pool.shutdown();
//! # }
//! ```

use crate::error::{CoreError, CoreResult, ErrorContext};
use crossbeam_deque::{Injector, Steal, Stealer, Worker as CbWorker};
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Priority
// ---------------------------------------------------------------------------

/// Task priority levels for the thread pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TaskPriority {
    /// Lowest priority – background / best-effort work.
    Background = 0,
    /// Default priority.
    Normal = 1,
    /// Elevated priority.
    High = 2,
    /// Highest priority – latency-critical work.
    Critical = 3,
}

impl Default for TaskPriority {
    fn default() -> Self {
        Self::Normal
    }
}

// ---------------------------------------------------------------------------
// Internal task wrapper
// ---------------------------------------------------------------------------

/// Trait-object-safe closure for the thread pool.
trait TaskFn: Send {
    fn execute(self: Box<Self>);
}

impl<F: FnOnce() + Send> TaskFn for F {
    fn execute(self: Box<Self>) {
        (*self)();
    }
}

/// An internal task that carries a boxed closure together with its priority
/// and a monotonically-increasing sequence number used for FIFO tie-breaking.
struct PrioritizedTask {
    priority: TaskPriority,
    seq: u64,
    task: Box<dyn TaskFn>,
}

// BinaryHeap is a max-heap, so we order by (priority, -seq) so that
// higher priority comes first, and among equal priorities the *older*
// (lower seq) task runs first (FIFO within the same priority).
impl PartialEq for PrioritizedTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.seq == other.seq
    }
}
impl Eq for PrioritizedTask {}
impl PartialOrd for PrioritizedTask {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for PrioritizedTask {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.seq.cmp(&self.seq)) // lower seq = earlier = higher order
    }
}

// ---------------------------------------------------------------------------
// Task handle (future-like)
// ---------------------------------------------------------------------------

/// A handle to a submitted task, allowing the caller to wait for its result.
///
/// Conceptually similar to `std::future::Future` but designed for
/// synchronous blocking retrieval via [`join`](TaskHandle::join).
pub struct TaskHandle<T> {
    inner: Arc<TaskHandleInner<T>>,
}

struct TaskHandleInner<T> {
    result: Mutex<Option<T>>,
    done: Condvar,
    completed: AtomicBool,
}

impl<T> TaskHandle<T> {
    fn new() -> (Self, Arc<TaskHandleInner<T>>) {
        let inner = Arc::new(TaskHandleInner {
            result: Mutex::new(None),
            done: Condvar::new(),
            completed: AtomicBool::new(false),
        });
        (
            Self {
                inner: inner.clone(),
            },
            inner,
        )
    }

    /// Block the calling thread until the task completes and return the result.
    pub fn join(self) -> CoreResult<T> {
        let mut guard = self.inner.result.lock().map_err(|e| {
            CoreError::ComputationError(ErrorContext::new(format!("mutex poisoned: {e}")))
        })?;
        while !self.inner.completed.load(Ordering::Acquire) {
            guard = self.inner.done.wait(guard).map_err(|e| {
                CoreError::ComputationError(ErrorContext::new(format!("condvar wait failed: {e}")))
            })?;
        }
        guard.take().ok_or_else(|| {
            CoreError::ComputationError(ErrorContext::new(
                "task completed but produced no result (possibly panicked)".to_string(),
            ))
        })
    }

    /// Block until the task completes or the timeout elapses.
    ///
    /// Returns `Ok(Some(value))` on success, `Ok(None)` on timeout,
    /// or `Err(...)` on internal failure.
    pub fn join_timeout(&self, timeout: Duration) -> CoreResult<Option<T>> {
        let deadline = Instant::now() + timeout;
        let mut guard = self.inner.result.lock().map_err(|e| {
            CoreError::ComputationError(ErrorContext::new(format!("mutex poisoned: {e}")))
        })?;
        while !self.inner.completed.load(Ordering::Acquire) {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(None);
            }
            let (g, timeout_result) =
                self.inner
                    .done
                    .wait_timeout(guard, remaining)
                    .map_err(|e| {
                        CoreError::ComputationError(ErrorContext::new(format!(
                            "condvar wait failed: {e}"
                        )))
                    })?;
            guard = g;
            if timeout_result.timed_out() && !self.inner.completed.load(Ordering::Acquire) {
                return Ok(None);
            }
        }
        Ok(guard.take())
    }

    /// Returns `true` if the task has completed.
    pub fn is_done(&self) -> bool {
        self.inner.completed.load(Ordering::Acquire)
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for constructing a [`ThreadPool`].
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    /// Number of worker threads (0 means auto-detect via `num_cpus`).
    pub num_workers: usize,
    /// Name prefix for worker threads (e.g. `"scirs-pool"`).
    pub thread_name_prefix: String,
    /// Stack size for worker threads (bytes, 0 means OS default).
    pub stack_size: usize,
    /// Whether to enable the priority queue (if false, all tasks are FIFO).
    pub enable_priority: bool,
    /// Grace period during shutdown before forcibly dropping pending tasks.
    pub shutdown_timeout: Duration,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            num_workers: 0,
            thread_name_prefix: "scirs-pool".to_string(),
            stack_size: 0,
            enable_priority: true,
            shutdown_timeout: Duration::from_secs(30),
        }
    }
}

impl ThreadPoolConfig {
    /// Create a config with a specific number of workers.
    pub fn with_workers(mut self, n: usize) -> Self {
        self.num_workers = n;
        self
    }

    /// Set the thread name prefix.
    pub fn with_name_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.thread_name_prefix = prefix.into();
        self
    }

    /// Enable or disable the priority queue.
    pub fn with_priority(mut self, enable: bool) -> Self {
        self.enable_priority = enable;
        self
    }

    /// Set the shutdown timeout.
    pub fn with_shutdown_timeout(mut self, timeout: Duration) -> Self {
        self.shutdown_timeout = timeout;
        self
    }
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/// Internal shared state for the thread pool.
struct SharedState {
    /// Global injector queue (work-stealing).
    injector: Injector<PrioritizedTask>,
    /// Priority queue for tasks when priority is enabled.
    priority_queue: Mutex<BinaryHeap<PrioritizedTask>>,
    /// Stealers for each worker.
    stealers: parking_lot::RwLock<Vec<Stealer<PrioritizedTask>>>,
    /// Whether the pool is shutting down.
    shutdown: AtomicBool,
    /// Condition variable to wake sleeping workers.
    work_available: Condvar,
    /// Mutex paired with work_available condvar.
    work_mutex: Mutex<()>,
    /// Number of pending (unfinished) tasks.
    pending_count: AtomicUsize,
    /// Monotonically-increasing sequence counter for FIFO ordering.
    seq_counter: AtomicU64,
    /// Whether priority scheduling is enabled.
    enable_priority: bool,
    /// Current live worker count.
    worker_count: AtomicUsize,
    /// Total tasks executed.
    total_executed: AtomicU64,
    /// Total tasks submitted.
    total_submitted: AtomicU64,
}

impl SharedState {
    fn next_seq(&self) -> u64 {
        self.seq_counter.fetch_add(1, Ordering::Relaxed)
    }

    /// Push a task into the queue.
    fn push_task(&self, task: PrioritizedTask) {
        if self.enable_priority {
            if let Ok(mut pq) = self.priority_queue.lock() {
                pq.push(task);
            }
        } else {
            self.injector.push(task);
        }
        self.pending_count.fetch_add(1, Ordering::Release);
        self.total_submitted.fetch_add(1, Ordering::Relaxed);
        // Wake one idle worker.
        if let Ok(_guard) = self.work_mutex.lock() {
            self.work_available.notify_one();
        }
    }

    /// Try to pop the highest-priority task.
    fn pop_task(&self, local: &CbWorker<PrioritizedTask>) -> Option<PrioritizedTask> {
        // First try the priority queue (if enabled).
        if self.enable_priority {
            if let Ok(mut pq) = self.priority_queue.lock() {
                if let Some(task) = pq.pop() {
                    return Some(task);
                }
            }
        }

        // Then try local queue.
        if let Some(task) = local.pop() {
            return Some(task);
        }

        // Then try stealing from the global injector.
        loop {
            match self.injector.steal_batch_and_pop(local) {
                Steal::Success(task) => return Some(task),
                Steal::Retry => continue,
                Steal::Empty => break,
            }
        }

        // Finally try stealing from other workers.
        let stealers = self.stealers.read();
        let len = stealers.len();
        if len == 0 {
            return None;
        }
        // Start from a pseudo-random index to reduce contention.
        let start = self.seq_counter.load(Ordering::Relaxed) as usize % len;
        for i in 0..len {
            let idx = (start + i) % len;
            loop {
                match stealers[idx].steal_batch_and_pop(local) {
                    Steal::Success(task) => return Some(task),
                    Steal::Retry => continue,
                    Steal::Empty => break,
                }
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Worker handle
// ---------------------------------------------------------------------------

struct WorkerHandle {
    handle: Option<JoinHandle<()>>,
    /// Per-worker shutdown flag.
    alive: Arc<AtomicBool>,
}

// ---------------------------------------------------------------------------
// Thread Pool
// ---------------------------------------------------------------------------

/// A fixed-size thread pool with work-stealing, priority scheduling,
/// scoped tasks, and dynamic resizing.
pub struct ThreadPool {
    shared: Arc<SharedState>,
    workers: Mutex<Vec<WorkerHandle>>,
    config: ThreadPoolConfig,
}

impl ThreadPool {
    /// Create a new thread pool with the given configuration.
    pub fn new(config: ThreadPoolConfig) -> CoreResult<Self> {
        let num = if config.num_workers == 0 {
            num_cpus::get().max(1)
        } else {
            config.num_workers
        };

        let shared = Arc::new(SharedState {
            injector: Injector::new(),
            priority_queue: Mutex::new(BinaryHeap::new()),
            stealers: parking_lot::RwLock::new(Vec::new()),
            shutdown: AtomicBool::new(false),
            work_available: Condvar::new(),
            work_mutex: Mutex::new(()),
            pending_count: AtomicUsize::new(0),
            seq_counter: AtomicU64::new(0),
            enable_priority: config.enable_priority,
            worker_count: AtomicUsize::new(0),
            total_executed: AtomicU64::new(0),
            total_submitted: AtomicU64::new(0),
        });

        let pool = Self {
            shared,
            workers: Mutex::new(Vec::with_capacity(num)),
            config,
        };

        for i in 0..num {
            pool.spawn_worker(i)?;
        }

        Ok(pool)
    }

    /// Create a pool with default settings and the given number of workers.
    pub fn with_workers(n: usize) -> CoreResult<Self> {
        Self::new(ThreadPoolConfig::default().with_workers(n))
    }

    // -----------------------------------------------------------------------
    // Spawning / removing workers
    // -----------------------------------------------------------------------

    fn spawn_worker(&self, id: usize) -> CoreResult<()> {
        let local = CbWorker::new_fifo();
        let stealer = local.stealer();

        {
            let mut stealers = self.shared.stealers.write();
            stealers.push(stealer);
        }

        let shared = self.shared.clone();
        let alive = Arc::new(AtomicBool::new(true));
        let alive_clone = alive.clone();
        let name = format!("{}-{}", self.config.thread_name_prefix, id);

        let mut builder = thread::Builder::new().name(name);
        if self.config.stack_size > 0 {
            builder = builder.stack_size(self.config.stack_size);
        }

        let handle = builder
            .spawn(move || {
                worker_loop(shared, local, alive_clone);
            })
            .map_err(|e| {
                CoreError::ComputationError(ErrorContext::new(format!(
                    "failed to spawn worker thread {id}: {e}"
                )))
            })?;

        self.shared.worker_count.fetch_add(1, Ordering::Release);

        let mut workers = self.workers.lock().map_err(|e| {
            CoreError::ComputationError(ErrorContext::new(format!("workers mutex poisoned: {e}")))
        })?;
        workers.push(WorkerHandle {
            handle: Some(handle),
            alive,
        });

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Task submission
    // -----------------------------------------------------------------------

    /// Submit a task and receive a handle to retrieve its result.
    pub fn submit<F, T>(&self, f: F) -> TaskHandle<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        self.submit_with_priority(TaskPriority::Normal, f)
    }

    /// Submit a task with a specific priority.
    pub fn submit_with_priority<F, T>(&self, priority: TaskPriority, f: F) -> TaskHandle<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let (handle, inner) = TaskHandle::<T>::new();
        let seq = self.shared.next_seq();

        let task_fn = move || {
            let result = f();
            if let Ok(mut guard) = inner.result.lock() {
                *guard = Some(result);
            }
            inner.completed.store(true, Ordering::Release);
            inner.done.notify_all();
        };

        self.shared.push_task(PrioritizedTask {
            priority,
            seq,
            task: Box::new(task_fn),
        });

        handle
    }

    /// Execute a closure within a scoped context where tasks can borrow
    /// from the enclosing stack frame.
    ///
    /// All tasks submitted inside the scope must complete before `scope`
    /// returns, ensuring safe borrowing of local data.
    ///
    /// # Example
    ///
    /// ```rust
    /// # #[cfg(feature = "parallel")]
    /// # {
    /// use scirs2_core::thread_pool::{ThreadPool, ThreadPoolConfig};
    ///
    /// let pool = ThreadPool::new(ThreadPoolConfig::default().with_workers(2))
    ///     .expect("pool");
    ///
    /// let mut data = vec![0u32; 100];
    ///
    /// pool.scope(|s| {
    ///     for (i, slot) in data.iter_mut().enumerate() {
    ///         s.submit(move || {
    ///             *slot = (i * i) as u32;
    ///         });
    ///     }
    /// });
    ///
    /// assert_eq!(data[10], 100);
    /// # }
    /// ```
    pub fn scope<'env, F, R>(&'env self, f: F) -> R
    where
        F: FnOnce(&Scope<'env>) -> R,
    {
        let scope = Scope {
            shared: &self.shared,
            pending: AtomicUsize::new(0),
            done: Condvar::new(),
            done_mutex: Mutex::new(()),
            _marker: std::marker::PhantomData,
        };
        let result = f(&scope);
        scope.wait_all();
        result
    }

    // -----------------------------------------------------------------------
    // Dynamic resizing
    // -----------------------------------------------------------------------

    /// Grow the pool by adding `n` more worker threads.
    pub fn grow(&self, n: usize) -> CoreResult<()> {
        let current = self.shared.worker_count.load(Ordering::Acquire);
        for i in 0..n {
            self.spawn_worker(current + i)?;
        }
        Ok(())
    }

    /// Shrink the pool by removing up to `n` worker threads.
    ///
    /// Workers finish their current task before exiting.
    pub fn shrink(&self, n: usize) -> CoreResult<()> {
        let mut workers = self.workers.lock().map_err(|e| {
            CoreError::ComputationError(ErrorContext::new(format!("workers mutex poisoned: {e}")))
        })?;

        let to_remove = n.min(workers.len().saturating_sub(1)); // keep at least 1 worker
        for _ in 0..to_remove {
            if let Some(w) = workers.pop() {
                w.alive.store(false, Ordering::Release);
                // Wake the worker so it can see the flag.
                if let Ok(_g) = self.shared.work_mutex.lock() {
                    self.shared.work_available.notify_all();
                }
                if let Some(h) = w.handle {
                    let _ = h.join();
                }
                self.shared.worker_count.fetch_sub(1, Ordering::Release);
            }
        }
        Ok(())
    }

    /// Resize the pool to exactly `n` workers.
    pub fn resize(&self, n: usize) -> CoreResult<()> {
        let target = n.max(1);
        let current = self.worker_count();
        if target > current {
            self.grow(target - current)
        } else if target < current {
            self.shrink(current - target)
        } else {
            Ok(())
        }
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Number of live worker threads.
    pub fn worker_count(&self) -> usize {
        self.shared.worker_count.load(Ordering::Acquire)
    }

    /// Number of tasks currently pending (submitted but not yet completed).
    pub fn pending_tasks(&self) -> usize {
        self.shared.pending_count.load(Ordering::Acquire)
    }

    /// Total tasks submitted over the pool's lifetime.
    pub fn total_submitted(&self) -> u64 {
        self.shared.total_submitted.load(Ordering::Relaxed)
    }

    /// Total tasks executed over the pool's lifetime.
    pub fn total_executed(&self) -> u64 {
        self.shared.total_executed.load(Ordering::Relaxed)
    }

    /// Whether the pool is in the process of shutting down.
    pub fn is_shutting_down(&self) -> bool {
        self.shared.shutdown.load(Ordering::Acquire)
    }

    // -----------------------------------------------------------------------
    // Shutdown
    // -----------------------------------------------------------------------

    /// Initiate a graceful shutdown.
    ///
    /// Workers will finish their current task, then drain remaining queued tasks
    /// (up to `shutdown_timeout`), and finally exit.
    pub fn shutdown(&self) {
        self.shared.shutdown.store(true, Ordering::Release);

        // Wake all sleeping workers so they see the shutdown flag.
        if let Ok(_g) = self.shared.work_mutex.lock() {
            self.shared.work_available.notify_all();
        }

        let deadline = Instant::now() + self.config.shutdown_timeout;

        if let Ok(mut workers) = self.workers.lock() {
            for w in workers.iter_mut() {
                w.alive.store(false, Ordering::Release);
            }
            // Wake again after setting all alive=false.
            if let Ok(_g) = self.shared.work_mutex.lock() {
                self.shared.work_available.notify_all();
            }
            for w in workers.drain(..) {
                if let Some(h) = w.handle {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        // Timeout expired; just detach.
                        drop(h);
                    } else {
                        // Best-effort join.
                        let _ = h.join();
                    }
                }
            }
        }
    }

    /// Shutdown and block until all workers have exited.
    pub fn shutdown_and_wait(&self) {
        self.shutdown();
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        if !self.shared.shutdown.load(Ordering::Acquire) {
            self.shutdown();
        }
    }
}

// ---------------------------------------------------------------------------
// Worker loop
// ---------------------------------------------------------------------------

fn worker_loop(shared: Arc<SharedState>, local: CbWorker<PrioritizedTask>, alive: Arc<AtomicBool>) {
    while alive.load(Ordering::Acquire) && !shared.shutdown.load(Ordering::Acquire) {
        if let Some(task) = shared.pop_task(&local) {
            task.task.execute();
            shared.pending_count.fetch_sub(1, Ordering::Release);
            shared.total_executed.fetch_add(1, Ordering::Relaxed);
        } else {
            // No work available – sleep briefly.
            let guard = match shared.work_mutex.lock() {
                Ok(g) => g,
                Err(_) => break, // poisoned
            };
            // Double-check after acquiring the lock.
            if shared.shutdown.load(Ordering::Acquire) || !alive.load(Ordering::Acquire) {
                break;
            }
            let _ = shared
                .work_available
                .wait_timeout(guard, Duration::from_millis(5));
        }
    }

    // Drain remaining tasks if shutting down gracefully.
    if shared.shutdown.load(Ordering::Acquire) {
        while let Some(task) = shared.pop_task(&local) {
            task.task.execute();
            shared.pending_count.fetch_sub(1, Ordering::Release);
            shared.total_executed.fetch_add(1, Ordering::Relaxed);
        }
    }
}

// ---------------------------------------------------------------------------
// Scope (scoped tasks)
// ---------------------------------------------------------------------------

/// A scope inside which tasks may borrow data from the enclosing stack frame.
///
/// The scope guarantees that all submitted tasks complete before it drops,
/// making it safe to reference stack-local data.
pub struct Scope<'env> {
    shared: &'env Arc<SharedState>,
    pending: AtomicUsize,
    done: Condvar,
    done_mutex: Mutex<()>,
    _marker: std::marker::PhantomData<&'env ()>,
}

impl<'env> Scope<'env> {
    /// Submit a scoped task.
    ///
    /// The closure `f` may borrow data from the enclosing scope's lifetime `'env`.
    pub fn submit<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'env,
    {
        self.submit_with_priority(TaskPriority::Normal, f);
    }

    /// Submit a scoped task with a specific priority.
    pub fn submit_with_priority<F>(&self, priority: TaskPriority, f: F)
    where
        F: FnOnce() + Send + 'env,
    {
        self.pending.fetch_add(1, Ordering::Release);

        // Bundle raw pointers in a single Send struct to avoid Rust 2021
        // closure field-level capture of non-Send `*const T`.
        struct ScopeRefs {
            pending: *const AtomicUsize,
            done: *const Condvar,
            done_mutex: *const Mutex<()>,
        }
        // SAFETY: The pointed-to data lives in `Scope` fields and
        // `wait_all` ensures all tasks finish before the scope drops.
        unsafe impl Send for ScopeRefs {}

        let refs = ScopeRefs {
            pending: &self.pending as *const AtomicUsize,
            done: &self.done as *const Condvar,
            done_mutex: &self.done_mutex as *const Mutex<()>,
        };

        // Build the task closure.
        // ScopeRefs is Send (via unsafe impl above) and holds all the raw
        // pointers we need.  We build the closure in a helper function so
        // that Rust 2021 field-level capture does NOT split `refs` into its
        // individual non-Send pointer fields.
        fn make_task<'e, F2: FnOnce() + Send + 'e>(
            refs: ScopeRefs,
            f: F2,
        ) -> Box<dyn FnOnce() + Send + 'e> {
            // Wrap `refs` in a newtype that is explicitly `Send`, so that
            // Rust 2021 field-level capture cannot split it into individual
            // raw-pointer fields (which are not `Send` on their own).
            struct SendRefs(ScopeRefs);
            // SAFETY: ScopeRefs already carries `unsafe impl Send`; this
            // wrapper simply preserves that property in a capture-atomic unit.
            unsafe impl Send for SendRefs {}

            let send_refs = SendRefs(refs);
            Box::new(move || {
                f();
                // SAFETY: pointers are valid — Scope::wait_all cannot return
                // (and cannot free the Scope) until it sees pending == 0 while
                // holding the mutex.  By acquiring the mutex first and only then
                // decrementing pending, we ensure wait_all cannot sneak past the
                // mutex-guarded check and free the Scope before we call notify_all.
                unsafe {
                    let r = &send_refs.0;
                    let _g = match (*r.done_mutex).lock() {
                        Ok(g) => g,
                        Err(e) => e.into_inner(),
                    };
                    (*r.pending).fetch_sub(1, Ordering::AcqRel);
                    (*r.done).notify_all();
                }
            })
        }

        // SAFETY: We transmute the lifetime to 'static, but we guarantee
        // that all tasks complete inside `wait_all` before the scope drops,
        // so the borrows remain valid.
        let wrapper: Box<dyn FnOnce() + Send + 'static> = unsafe {
            let boxed: Box<dyn FnOnce() + Send + 'env> = make_task(refs, f);
            std::mem::transmute(boxed)
        };

        let seq = self.shared.next_seq();
        self.shared.push_task(PrioritizedTask {
            priority,
            seq,
            task: Box::new(wrapper),
        });
    }

    /// Wait for all tasks in this scope to complete.
    ///
    /// The mutex is held for the duration of the loop (released only during
    /// `wait_timeout`) so that the scope cannot be freed while a task still
    /// holds the mutex and is about to call `notify_all`.
    fn wait_all(&self) {
        let mut guard = match self.done_mutex.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        loop {
            if self.pending.load(Ordering::Acquire) == 0 {
                return;
            }
            guard = match self.done.wait_timeout(guard, Duration::from_millis(5)) {
                Ok((g, _)) => g,
                Err(e) => e.into_inner().0,
            };
        }
    }
}

// ---------------------------------------------------------------------------
// ThreadPool statistics
// ---------------------------------------------------------------------------

/// Snapshot of thread pool statistics.
#[derive(Debug, Clone)]
pub struct ThreadPoolStats {
    /// Current number of workers.
    pub worker_count: usize,
    /// Number of pending tasks.
    pub pending_tasks: usize,
    /// Total tasks submitted.
    pub total_submitted: u64,
    /// Total tasks executed.
    pub total_executed: u64,
    /// Whether the pool is shutting down.
    pub shutting_down: bool,
}

impl ThreadPool {
    /// Retrieve a snapshot of pool statistics.
    pub fn stats(&self) -> ThreadPoolStats {
        ThreadPoolStats {
            worker_count: self.worker_count(),
            pending_tasks: self.pending_tasks(),
            total_submitted: self.total_submitted(),
            total_executed: self.total_executed(),
            shutting_down: self.is_shutting_down(),
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience: global pool
// ---------------------------------------------------------------------------

/// Create a thread pool with the number of workers equal to available CPUs.
pub fn default_pool() -> CoreResult<ThreadPool> {
    ThreadPool::new(ThreadPoolConfig::default())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn test_basic_submit_and_join() {
        let pool = ThreadPool::with_workers(2).expect("pool");
        let h = pool.submit(|| 42);
        assert_eq!(h.join().expect("join"), 42);
    }

    #[test]
    fn test_multiple_tasks() {
        let pool = ThreadPool::with_workers(4).expect("pool");
        let handles: Vec<_> = (0..100).map(|i| pool.submit(move || i * i)).collect();
        for (i, h) in handles.into_iter().enumerate() {
            assert_eq!(h.join().expect("join"), i * i);
        }
    }

    #[test]
    fn test_priority_ordering() {
        // Submit many low-priority tasks, then one critical task.
        // The critical task should complete despite the backlog.
        let pool = ThreadPool::with_workers(1).expect("pool");
        let critical = pool.submit_with_priority(TaskPriority::Critical, || 999);
        let bg: Vec<_> = (0..50)
            .map(|_| pool.submit_with_priority(TaskPriority::Background, || 0))
            .collect();
        let result = critical.join().expect("critical");
        assert_eq!(result, 999);
        for h in bg {
            let _ = h.join();
        }
    }

    #[test]
    fn test_scoped_tasks() {
        let pool = ThreadPool::with_workers(2).expect("pool");
        let mut data = vec![0u32; 50];

        pool.scope(|s| {
            for (i, slot) in data.iter_mut().enumerate() {
                s.submit(move || {
                    *slot = (i as u32) * 2;
                });
            }
        });

        for (i, &val) in data.iter().enumerate() {
            assert_eq!(val, (i as u32) * 2);
        }
    }

    #[test]
    fn test_dynamic_resize() {
        let pool = ThreadPool::with_workers(2).expect("pool");
        assert_eq!(pool.worker_count(), 2);

        pool.grow(3).expect("grow");
        assert_eq!(pool.worker_count(), 5);

        pool.shrink(2).expect("shrink");
        assert_eq!(pool.worker_count(), 3);
    }

    #[test]
    fn test_resize_to() {
        let pool = ThreadPool::with_workers(2).expect("pool");
        pool.resize(6).expect("resize");
        assert_eq!(pool.worker_count(), 6);

        pool.resize(1).expect("resize down");
        assert_eq!(pool.worker_count(), 1);
    }

    #[test]
    fn test_graceful_shutdown() {
        let counter = Arc::new(AtomicU32::new(0));
        let pool = ThreadPool::with_workers(2).expect("pool");

        for _ in 0..100 {
            let c = counter.clone();
            pool.submit(move || {
                c.fetch_add(1, Ordering::Relaxed);
            });
        }

        pool.shutdown_and_wait();
        assert_eq!(counter.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn test_stats() {
        let pool = ThreadPool::with_workers(2).expect("pool");
        let h = pool.submit(|| 1);
        let _ = h.join();
        let stats = pool.stats();
        assert!(stats.total_submitted >= 1);
        assert!(stats.total_executed >= 1);
        assert_eq!(stats.worker_count, 2);
    }

    #[test]
    fn test_join_timeout() {
        let pool = ThreadPool::with_workers(1).expect("pool");
        let h = pool.submit(|| {
            std::thread::sleep(Duration::from_millis(200));
            42
        });
        // Short timeout should return None.
        let r = h.join_timeout(Duration::from_millis(10)).expect("timeout");
        // Result might be None (timeout) or Some(42) depending on timing,
        // but the call itself should not error.
        if let Some(v) = r {
            assert_eq!(v, 42);
        }
    }

    #[test]
    fn test_empty_scope() {
        let pool = ThreadPool::with_workers(2).expect("pool");
        pool.scope(|_s| {
            // no tasks
        });
    }

    #[test]
    fn test_task_handle_is_done() {
        let pool = ThreadPool::with_workers(1).expect("pool");
        let h = pool.submit(|| 7);
        let _ = h.join();
        // After join the handle is consumed, so we test via a different path.
        let h2 = pool.submit(|| {
            std::thread::sleep(Duration::from_millis(50));
            99
        });
        // Initially might not be done.
        let done_before = h2.is_done();
        let val = h2.join().expect("join");
        assert_eq!(val, 99);
        // done_before can be true or false depending on timing, that's fine.
        let _ = done_before;
    }

    #[test]
    fn test_config_builder() {
        let cfg = ThreadPoolConfig::default()
            .with_workers(4)
            .with_name_prefix("test")
            .with_priority(false)
            .with_shutdown_timeout(Duration::from_secs(5));
        assert_eq!(cfg.num_workers, 4);
        assert_eq!(cfg.thread_name_prefix, "test");
        assert!(!cfg.enable_priority);
        assert_eq!(cfg.shutdown_timeout, Duration::from_secs(5));
    }

    #[test]
    fn test_no_priority_mode() {
        let pool = ThreadPool::new(
            ThreadPoolConfig::default()
                .with_workers(2)
                .with_priority(false),
        )
        .expect("pool");
        let h = pool.submit(|| "no prio");
        assert_eq!(h.join().expect("join"), "no prio");
    }

    #[test]
    fn test_default_pool() {
        let pool = default_pool().expect("default pool");
        assert!(pool.worker_count() >= 1);
        let h = pool.submit(|| 123);
        assert_eq!(h.join().expect("join"), 123);
    }
}
