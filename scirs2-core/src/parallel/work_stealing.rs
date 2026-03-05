//! Work-stealing scheduler with double-ended deques per worker.
//!
//! Each worker owns a deque: it pushes/pops from the *front*, while
//! thieves steal from the *back*.  This classic Chase-Lev strategy
//! keeps contention low when the pool is lightly loaded.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};

// ─────────────────────────────────────────────────────────────────────────────
// Task priority
// ─────────────────────────────────────────────────────────────────────────────

/// Priority level attached to each submitted task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    /// Must complete before `Normal` tasks.
    High = 2,
    /// Default priority.
    Normal = 1,
    /// Background work – stolen last.
    Low = 0,
}

// ─────────────────────────────────────────────────────────────────────────────
// Boxed closure alias
// ─────────────────────────────────────────────────────────────────────────────

type BoxedTask = Box<dyn FnOnce() + Send + 'static>;

struct PrioritisedTask {
    priority: TaskPriority,
    task: BoxedTask,
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-worker double-ended deque
// ─────────────────────────────────────────────────────────────────────────────

/// A work-stealing deque shared between one owner thread and many stealers.
///
/// The owner pushes / pops from the *front*; stealers steal from the *back*.
#[derive(Clone)]
pub struct WorkerDeque {
    inner: Arc<Mutex<VecDeque<PrioritisedTask>>>,
}

impl WorkerDeque {
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Push a task to the *front* (owner side).
    fn push_front(&self, task: PrioritisedTask) -> CoreResult<()> {
        let mut dq = self.inner.lock().map_err(|e| {
            CoreError::SchedulerError(
                ErrorContext::new(format!("deque lock poisoned: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
        dq.push_front(task);
        Ok(())
    }

    /// Pop a task from the *front* (owner side).
    fn pop_front(&self) -> CoreResult<Option<PrioritisedTask>> {
        let mut dq = self.inner.lock().map_err(|e| {
            CoreError::SchedulerError(
                ErrorContext::new(format!("deque lock poisoned: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
        Ok(dq.pop_front())
    }

    /// Steal a task from the *back* (stealer side).
    fn steal_from(&self) -> CoreResult<Option<PrioritisedTask>> {
        let mut dq = self.inner.lock().map_err(|e| {
            CoreError::SchedulerError(
                ErrorContext::new(format!("deque lock poisoned: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
        Ok(dq.pop_back())
    }

    fn is_empty(&self) -> bool {
        self.inner
            .lock()
            .map(|dq| dq.is_empty())
            .unwrap_or(true)
    }

    fn len(&self) -> usize {
        self.inner
            .lock()
            .map(|dq| dq.len())
            .unwrap_or(0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// JoinHandle – lets callers await task results
// ─────────────────────────────────────────────────────────────────────────────

/// Handle returned by [`WorkStealingPool::spawn`].  Call [`join`] to block
/// until the task finishes and retrieve its return value.
pub struct JoinHandle<R> {
    result: Arc<(Mutex<Option<thread::Result<R>>>, Condvar)>,
}

impl<R: Send + 'static> JoinHandle<R> {
    fn new() -> (Self, Arc<(Mutex<Option<thread::Result<R>>>, Condvar)>) {
        let pair = Arc::new((Mutex::new(None), Condvar::new()));
        let handle = JoinHandle {
            result: Arc::clone(&pair),
        };
        (handle, pair)
    }

    /// Block until the task completes and return its result.
    ///
    /// # Errors
    /// Returns `CoreError::SchedulerError` if the mutex is poisoned or the
    /// spawned task panicked.
    pub fn join(self) -> CoreResult<R> {
        let (lock, cvar) = &*self.result;
        let mut guard = lock.lock().map_err(|e| {
            CoreError::SchedulerError(
                ErrorContext::new(format!("join lock poisoned: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
        while guard.is_none() {
            guard = cvar.wait(guard).map_err(|e| {
                CoreError::SchedulerError(
                    ErrorContext::new(format!("condvar wait failed: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;
        }
        match guard.take() {
            Some(Ok(val)) => Ok(val),
            Some(Err(_panic)) => Err(CoreError::SchedulerError(
                ErrorContext::new("task panicked".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            )),
            None => Err(CoreError::SchedulerError(
                ErrorContext::new("result missing after wait".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            )),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared state
// ─────────────────────────────────────────────────────────────────────────────

struct PoolState {
    deques: Vec<WorkerDeque>,
    /// Pending tasks waiting to be assigned to a worker.
    global_queue: Mutex<VecDeque<PrioritisedTask>>,
    notify: Condvar,
    /// Set to `true` when the pool is being shut down.
    shutdown: AtomicBool,
    /// Number of active worker threads.
    active_workers: AtomicUsize,
}

impl PoolState {
    fn new(n_threads: usize) -> Self {
        let deques = (0..n_threads).map(|_| WorkerDeque::new()).collect();
        Self {
            deques,
            global_queue: Mutex::new(VecDeque::new()),
            notify: Condvar::new(),
            shutdown: AtomicBool::new(false),
            active_workers: AtomicUsize::new(0),
        }
    }

    /// Enqueue a task into the global queue, notifying a waiting worker.
    fn enqueue(&self, task: PrioritisedTask) -> CoreResult<()> {
        let mut q = self.global_queue.lock().map_err(|e| {
            CoreError::SchedulerError(
                ErrorContext::new(format!("global queue lock poisoned: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
        q.push_back(task);
        drop(q);
        self.notify.notify_one();
        Ok(())
    }

    /// Try to get a task: first from the global queue, then by stealing.
    fn try_get(&self, worker_id: usize) -> CoreResult<Option<PrioritisedTask>> {
        // 1. Global queue
        {
            let mut q = self.global_queue.lock().map_err(|e| {
                CoreError::SchedulerError(
                    ErrorContext::new(format!("global queue lock poisoned: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;
            if let Some(t) = q.pop_front() {
                return Ok(Some(t));
            }
        }

        // 2. Own deque (pop front)
        if let Some(t) = self.deques[worker_id].pop_front()? {
            return Ok(Some(t));
        }

        // 3. Steal from the busiest neighbour
        let n = self.deques.len();
        let best = self
            .deques
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != worker_id)
            .max_by_key(|(_, d)| d.len());

        if let Some((_, victim)) = best {
            if let Some(t) = victim.steal_from()? {
                return Ok(Some(t));
            }
        }

        // 4. Round-robin steal fallback
        for offset in 1..n {
            let victim_id = (worker_id + offset) % n;
            if let Some(t) = self.deques[victim_id].steal_from()? {
                return Ok(Some(t));
            }
        }

        Ok(None)
    }

    /// Push a high-priority task directly into a worker's front.
    fn push_to_worker(&self, worker_id: usize, task: PrioritisedTask) -> CoreResult<()> {
        self.deques[worker_id].push_front(task)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WorkStealingPool
// ─────────────────────────────────────────────────────────────────────────────

/// A thread-pool that uses per-worker double-ended deques and work-stealing.
///
/// # Example
/// ```rust
/// use scirs2_core::parallel::work_stealing::{WorkStealingPool, TaskPriority};
///
/// let pool = WorkStealingPool::new(4).expect("should succeed");
/// let handle = pool.spawn(|| 6 * 7).expect("should succeed");
/// assert_eq!(handle.join().expect("should succeed"), 42);
/// ```
pub struct WorkStealingPool {
    state: Arc<PoolState>,
    n_threads: usize,
    /// Worker thread join handles (kept so Drop can cleanly shut down).
    _workers: Vec<thread::JoinHandle<()>>,
}

impl WorkStealingPool {
    /// Create a pool with `n_threads` worker threads.
    ///
    /// # Errors
    /// Returns `CoreError::SchedulerError` if a worker thread cannot be spawned.
    pub fn new(n_threads: usize) -> CoreResult<Self> {
        if n_threads == 0 {
            return Err(CoreError::InvalidArgument(
                ErrorContext::new("n_threads must be > 0".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        let state = Arc::new(PoolState::new(n_threads));
        let mut worker_handles = Vec::with_capacity(n_threads);

        for worker_id in 0..n_threads {
            let s = Arc::clone(&state);
            let handle = thread::Builder::new()
                .name(format!("ws-worker-{worker_id}"))
                .spawn(move || Self::worker_loop(worker_id, s))
                .map_err(|e| {
                    CoreError::SchedulerError(
                        ErrorContext::new(format!("failed to spawn worker {worker_id}: {e}"))
                            .with_location(ErrorLocation::new(file!(), line!())),
                    )
                })?;
            worker_handles.push(handle);
        }

        Ok(Self {
            state,
            n_threads,
            _workers: worker_handles,
        })
    }

    /// Worker event loop – runs until the pool is shut down.
    fn worker_loop(worker_id: usize, state: Arc<PoolState>) {
        state.active_workers.fetch_add(1, Ordering::SeqCst);

        loop {
            if state.shutdown.load(Ordering::Acquire) {
                break;
            }

            let task = match state.try_get(worker_id) {
                Ok(Some(t)) => t,
                Ok(None) => {
                    // Nothing to steal – wait on the condvar
                    let lock_result = state.global_queue.lock();
                    match lock_result {
                        Err(_) => break,
                        Ok(guard) => {
                            // Double-check: still empty?
                            if guard.is_empty()
                                && state
                                    .deques
                                    .iter()
                                    .all(|d| d.is_empty())
                            {
                                let _ = state.notify.wait_timeout(
                                    guard,
                                    std::time::Duration::from_millis(5),
                                );
                            }
                        }
                    }
                    continue;
                }
                Err(_) => break,
            };

            (task.task)();
        }

        state.active_workers.fetch_sub(1, Ordering::SeqCst);
    }

    /// Submit a closure and return a [`JoinHandle`] to await its result.
    ///
    /// # Errors
    /// Returns `CoreError::SchedulerError` if the task cannot be enqueued.
    pub fn spawn<F, R>(&self, f: F) -> CoreResult<JoinHandle<R>>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.spawn_with_priority(f, TaskPriority::Normal)
    }

    /// Submit a closure with an explicit priority.
    ///
    /// `High`-priority tasks are pushed to the front of a worker's local deque;
    /// `Normal` and `Low` tasks go into the global queue.
    ///
    /// # Errors
    /// Returns `CoreError::SchedulerError` if the task cannot be enqueued.
    pub fn spawn_with_priority<F, R>(
        &self,
        f: F,
        priority: TaskPriority,
    ) -> CoreResult<JoinHandle<R>>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let (handle, result_pair) = JoinHandle::new();

        let wrapped: BoxedTask = Box::new(move || {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
            let (lock, cvar) = &*result_pair;
            if let Ok(mut guard) = lock.lock() {
                *guard = Some(outcome);
                cvar.notify_one();
            }
        });

        let pt = PrioritisedTask {
            priority,
            task: wrapped,
        };

        if priority == TaskPriority::High {
            // Route high-priority tasks to the worker with the smallest queue.
            let target = self
                .state
                .deques
                .iter()
                .enumerate()
                .min_by_key(|(_, d)| d.len())
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.state.push_to_worker(target, pt)?;
            self.state.notify.notify_one();
        } else {
            self.state.enqueue(pt)?;
        }

        Ok(handle)
    }

    /// Number of worker threads.
    pub fn n_threads(&self) -> usize {
        self.n_threads
    }

    /// Attempt to steal one task from the deque of worker `victim_id`.
    ///
    /// # Errors
    /// Returns `CoreError::SchedulerError` if the deque lock is poisoned or the
    /// worker index is out of range.
    pub fn steal_from_worker(&self, victim_id: usize) -> CoreResult<bool> {
        if victim_id >= self.n_threads {
            return Err(CoreError::IndexError(
                ErrorContext::new(format!(
                    "victim_id {victim_id} >= n_threads {}",
                    self.n_threads
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
        let maybe = self.state.deques[victim_id].steal_from()?;
        if let Some(task) = maybe {
            // Re-queue the stolen task into the global queue so another worker
            // picks it up.
            self.state.enqueue(task)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

impl Drop for WorkStealingPool {
    fn drop(&mut self) {
        self.state.shutdown.store(true, Ordering::Release);
        // Wake all workers so they can exit.
        for _ in 0..self.n_threads * 2 {
            self.state.notify.notify_one();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_basic() {
        let pool = WorkStealingPool::new(2).expect("should succeed");
        let h = pool.spawn(|| 1 + 1).expect("should succeed");
        assert_eq!(h.join().expect("should succeed"), 2);
    }

    #[test]
    fn test_pool_multiple_tasks() {
        let pool = WorkStealingPool::new(4).expect("should succeed");
        let handles: Vec<_> = (0..20u64)
            .map(|i| pool.spawn(move || i * i).expect("should succeed"))
            .collect();
        for (i, h) in handles.into_iter().enumerate() {
            assert_eq!(h.join().expect("should succeed"), (i as u64) * (i as u64));
        }
    }

    #[test]
    fn test_priority_spawn() {
        let pool = WorkStealingPool::new(2).expect("should succeed");
        let h = pool
            .spawn_with_priority(|| "high", TaskPriority::High)
            .expect("should succeed");
        assert_eq!(h.join().expect("should succeed"), "high");
    }

    #[test]
    fn test_zero_threads_error() {
        assert!(WorkStealingPool::new(0).is_err());
    }

    #[test]
    fn test_steal_from_worker_oob() {
        let pool = WorkStealingPool::new(2).expect("should succeed");
        assert!(pool.steal_from_worker(99).is_err());
    }
}
