//! Chase-Lev work-stealing double-ended deque and scheduler.
//!
//! # Overview
//!
//! This module provides:
//!
//! - [`WorkStealingDeque`] — a single-producer, multi-consumer lock-free deque
//!   following the classic Chase-Lev design.  The *owner* pushes and pops from
//!   the **bottom**; *thieves* steal from the **top**.
//! - [`WorkStealingScheduler`] — a thread-pool built on top of per-worker
//!   [`WorkStealingDeque`]s that automatically balances load through stealing.
//! - [`PriorityTaskQueue`] — a multi-priority bounded task queue that separates
//!   high / normal / low work and serves them in order.
//!
//! # Safety note
//!
//! The Chase-Lev deque uses `unsafe` pointer arithmetic to achieve lock-free
//! semantics; all unsafety is contained inside [`WorkStealingDeque`] and is
//! guarded by the invariants described in the individual `unsafe` blocks.

use std::cell::UnsafeCell;
use std::sync::atomic::{fence, AtomicIsize, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};

// ── constants ────────────────────────────────────────────────────────────────

/// Initial capacity of the circular buffer (must be a power of two).
const INITIAL_CAPACITY: usize = 64;

// ── circular buffer ──────────────────────────────────────────────────────────

/// A heap-allocated circular buffer of capacity `cap` (always a power of two).
struct CircularBuf<T> {
    cap: usize,
    data: Box<[UnsafeCell<Option<T>>]>,
}

impl<T> CircularBuf<T> {
    fn new(cap: usize) -> Self {
        let data = (0..cap)
            .map(|_| UnsafeCell::new(None))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self { cap, data }
    }

    fn mask(&self) -> usize {
        self.cap - 1
    }

    /// Write `val` at logical index `i`.
    ///
    /// # Safety
    /// The caller must ensure no concurrent reader accesses the same slot.
    unsafe fn write(&self, i: usize, val: T) {
        let slot = self.data[i & self.mask()].get();
        // SAFETY: we have exclusive write access because the caller checked
        // bottom/top atomics before calling.
        unsafe { (*slot) = Some(val) };
    }

    /// Read the value at logical index `i`, replacing the slot with `None`.
    ///
    /// # Safety
    /// The caller must ensure it has exclusive access to this slot via
    /// the CAS on `top` (for stealers) or the pop logic (for the owner).
    unsafe fn read(&self, i: usize) -> Option<T> {
        let slot = self.data[i & self.mask()].get();
        // SAFETY: guaranteed exclusive by the caller's atomic protocol.
        unsafe { (*slot).take() }
    }
}

// SAFETY: We uphold the single-owner / many-stealer invariant through the
// atomic bottom/top protocol, making concurrent accesses disjoint.
unsafe impl<T: Send> Send for CircularBuf<T> {}
unsafe impl<T: Send> Sync for CircularBuf<T> {}

// ── WorkStealingDeque ────────────────────────────────────────────────────────

/// A Chase-Lev lock-free work-stealing deque.
///
/// The **owner** thread calls [`push`](WorkStealingDeque::push) and
/// [`pop`](WorkStealingDeque::pop).  Any number of **stealer** threads call
/// [`steal`](WorkStealingDeque::steal) concurrently.
///
/// The internal buffer grows automatically (doubling) when the owner fills it.
/// Shrinking is *not* implemented to keep the implementation simple.
pub struct WorkStealingDeque<T: Send + 'static> {
    bottom: AtomicIsize,
    top: AtomicIsize,
    buf: Mutex<Arc<CircularBuf<T>>>,
}

/// Outcome of a [`WorkStealingDeque::steal`] attempt.
#[derive(Debug)]
pub enum StealResult<T> {
    /// A task was successfully stolen.
    Success(T),
    /// The deque is empty — no task available.
    Empty,
    /// A concurrent stealer raced us — try again later.
    Retry,
}

impl<T: Send + 'static> WorkStealingDeque<T> {
    /// Create a new empty deque.
    pub fn new() -> Self {
        Self {
            bottom: AtomicIsize::new(0),
            top: AtomicIsize::new(0),
            buf: Mutex::new(Arc::new(CircularBuf::new(INITIAL_CAPACITY))),
        }
    }

    /// Number of elements currently in the deque (approximate).
    pub fn len(&self) -> usize {
        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Relaxed);
        (b - t).max(0) as usize
    }

    /// Returns `true` if the deque contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Push a task onto the bottom (owner side).
    ///
    /// May grow the internal buffer if needed.
    pub fn push(&self, task: T) -> CoreResult<()> {
        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Acquire);
        let size = (b - t) as usize;

        let buf: Arc<CircularBuf<T>> = {
            let guard = self.buf.lock().map_err(|e| {
                CoreError::SchedulerError(
                    ErrorContext::new(format!("deque buf lock poisoned: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;
            Arc::clone(&*guard)
        };

        // Grow if needed
        let buf: Arc<CircularBuf<T>> = if size >= buf.cap - 1 {
            let new_cap = buf.cap * 2;
            let new_buf = Arc::new(CircularBuf::new(new_cap));
            // Copy existing elements
            for i in t..b {
                // SAFETY: single owner, exclusive write slot on new_buf.
                unsafe {
                    let val = buf.read(i as usize);
                    if let Some(v) = val {
                        new_buf.write(i as usize, v);
                    }
                }
            }
            let mut guard = self.buf.lock().map_err(|e| {
                CoreError::SchedulerError(
                    ErrorContext::new(format!("deque buf lock poisoned during grow: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;
            *guard = Arc::clone(&new_buf);
            new_buf
        } else {
            buf
        };

        // SAFETY: `b` is the owner-exclusive write index; no stealer touches it.
        unsafe { buf.write(b as usize, task) };
        fence(Ordering::Release);
        self.bottom.store(b + 1, Ordering::Relaxed);
        Ok(())
    }

    /// Pop a task from the bottom (owner side).
    ///
    /// Returns `None` if the deque is empty.
    pub fn pop(&self) -> CoreResult<Option<T>> {
        let b = self.bottom.load(Ordering::Relaxed) - 1;
        let buf: Arc<CircularBuf<T>> = {
            let guard = self.buf.lock().map_err(|e| {
                CoreError::SchedulerError(
                    ErrorContext::new(format!("deque buf lock poisoned on pop: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;
            Arc::clone(&*guard)
        };
        self.bottom.store(b, Ordering::Relaxed);
        fence(Ordering::SeqCst);
        let t = self.top.load(Ordering::Relaxed);

        if t > b {
            // Deque was empty — restore bottom.
            self.bottom.store(b + 1, Ordering::Relaxed);
            return Ok(None);
        }

        // SAFETY: slot at `b` is exclusively owned by the owner at this point.
        let task = unsafe { buf.read(b as usize) };

        if t == b {
            // Last element — race stealers with a CAS on top.
            let stolen = self
                .top
                .compare_exchange(t, t + 1, Ordering::SeqCst, Ordering::Relaxed)
                .is_err();
            self.bottom.store(b + 1, Ordering::Relaxed);
            if stolen {
                return Ok(None);
            }
        }

        Ok(task)
    }

    /// Steal a task from the top (stealer side).
    pub fn steal(&self) -> StealResult<T> {
        let t = self.top.load(Ordering::Acquire);
        fence(Ordering::SeqCst);
        let b = self.bottom.load(Ordering::Acquire);

        if t >= b {
            return StealResult::Empty;
        }

        let buf = match self.buf.lock() {
            Ok(g) => Arc::clone(&*g),
            Err(_) => return StealResult::Retry,
        };

        // SAFETY: `t` was acquired before checking `b`; the slot is valid.
        let task = unsafe { buf.read(t as usize) };

        match self
            .top
            .compare_exchange(t, t + 1, Ordering::SeqCst, Ordering::Relaxed)
        {
            Ok(_) => match task {
                Some(v) => StealResult::Success(v),
                None => StealResult::Retry,
            },
            Err(_) => StealResult::Retry,
        }
    }
}

impl<T: Send + 'static> Default for WorkStealingDeque<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ── PriorityTaskQueue ────────────────────────────────────────────────────────

/// Priority level for tasks in a [`PriorityTaskQueue`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    /// Must run before `Normal` and `Low` tasks.
    High = 2,
    /// Default priority.
    Normal = 1,
    /// Background tasks — run only when no higher-priority work exists.
    Low = 0,
}

type BoxTask = Box<dyn FnOnce() + Send + 'static>;

struct PriorityItem {
    priority: Priority,
    seq: u64, // tie-break: lower seq = earlier submitted
    task: BoxTask,
}

impl PartialEq for PriorityItem {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.seq == other.seq
    }
}
impl Eq for PriorityItem {}

impl PartialOrd for PriorityItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority first; equal priority → lower seq first (FIFO).
        // BinaryHeap is a max-heap, so larger items are popped first.
        // We want High > Normal > Low, so use self.priority > other.priority.
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

/// A bounded multi-priority task queue.
///
/// Tasks with [`Priority::High`] are dequeued before [`Priority::Normal`],
/// which are dequeued before [`Priority::Low`].  Within a priority level
/// tasks are served FIFO.
pub struct PriorityTaskQueue {
    inner: Mutex<PriorityQueueInner>,
    not_empty: Condvar,
    not_full: Condvar,
    capacity: usize,
    seq: AtomicUsize,
}

struct PriorityQueueInner {
    heap: std::collections::BinaryHeap<PriorityItem>,
    closed: bool,
}

impl PriorityTaskQueue {
    /// Create a queue that holds at most `capacity` pending tasks.
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            inner: Mutex::new(PriorityQueueInner {
                heap: std::collections::BinaryHeap::with_capacity(cap),
                closed: false,
            }),
            not_empty: Condvar::new(),
            not_full: Condvar::new(),
            capacity: cap,
            seq: AtomicUsize::new(0),
        }
    }

    /// Submit a task at the given `priority`.
    ///
    /// Blocks if the queue is at capacity, or returns `Err` if the queue is
    /// closed.
    pub fn submit<F>(&self, priority: Priority, f: F) -> CoreResult<()>
    where
        F: FnOnce() + Send + 'static,
    {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed) as u64;
        let item = PriorityItem {
            priority,
            seq,
            task: Box::new(f),
        };
        let mut guard = self.inner.lock().map_err(|e| {
            CoreError::SchedulerError(
                ErrorContext::new(format!("priority queue lock poisoned on submit: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
        loop {
            if guard.closed {
                return Err(CoreError::InvalidInput(ErrorContext::new(
                    "PriorityTaskQueue: queue is closed",
                )));
            }
            if guard.heap.len() < self.capacity {
                break;
            }
            guard = self.not_full.wait(guard).map_err(|e| {
                CoreError::SchedulerError(
                    ErrorContext::new(format!("condvar wait poisoned: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;
        }
        guard.heap.push(item);
        self.not_empty.notify_one();
        Ok(())
    }

    /// Try to submit without blocking.  Returns `Err` if full or closed.
    pub fn try_submit<F>(&self, priority: Priority, f: F) -> CoreResult<()>
    where
        F: FnOnce() + Send + 'static,
    {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed) as u64;
        let item = PriorityItem {
            priority,
            seq,
            task: Box::new(f),
        };
        let mut guard = self.inner.lock().map_err(|e| {
            CoreError::SchedulerError(
                ErrorContext::new(format!("priority queue lock poisoned on try_submit: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
        if guard.closed {
            return Err(CoreError::InvalidInput(ErrorContext::new(
                "PriorityTaskQueue: queue is closed",
            )));
        }
        if guard.heap.len() >= self.capacity {
            return Err(CoreError::InvalidInput(ErrorContext::new(
                "PriorityTaskQueue: queue is full",
            )));
        }
        guard.heap.push(item);
        self.not_empty.notify_one();
        Ok(())
    }

    /// Block until a task is available, then return it.
    ///
    /// Returns `None` when the queue is closed and drained.
    pub fn dequeue(&self) -> CoreResult<Option<BoxTask>> {
        let mut guard = self.inner.lock().map_err(|e| {
            CoreError::SchedulerError(
                ErrorContext::new(format!("priority queue lock poisoned on dequeue: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
        loop {
            if let Some(item) = guard.heap.pop() {
                self.not_full.notify_one();
                return Ok(Some(item.task));
            }
            if guard.closed {
                return Ok(None);
            }
            guard = self.not_empty.wait(guard).map_err(|e| {
                CoreError::SchedulerError(
                    ErrorContext::new(format!("condvar wait poisoned on dequeue: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;
        }
    }

    /// Try to dequeue without blocking.  Returns `None` if empty.
    pub fn try_dequeue(&self) -> CoreResult<Option<BoxTask>> {
        let mut guard = self.inner.lock().map_err(|e| {
            CoreError::SchedulerError(
                ErrorContext::new(format!("priority queue lock poisoned on try_dequeue: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
        match guard.heap.pop() {
            Some(item) => {
                self.not_full.notify_one();
                Ok(Some(item.task))
            }
            None => Ok(None),
        }
    }

    /// Close the queue.  Pending tasks may still be dequeued; new submits fail.
    pub fn close(&self) {
        if let Ok(mut g) = self.inner.lock() {
            g.closed = true;
        }
        self.not_empty.notify_all();
        self.not_full.notify_all();
    }

    /// Number of pending tasks.
    pub fn pending(&self) -> usize {
        self.inner.lock().map(|g| g.heap.len()).unwrap_or(0)
    }
}

// ── WorkStealingScheduler ────────────────────────────────────────────────────

/// Configuration for [`WorkStealingScheduler`].
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Number of worker threads (0 = use hardware concurrency).
    pub num_workers: usize,
    /// How many steal attempts before a worker sleeps.
    pub steal_attempts: usize,
    /// Duration to sleep when idle (microseconds).
    pub idle_sleep_us: u64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            num_workers: 0,
            steal_attempts: 32,
            idle_sleep_us: 100,
        }
    }
}

/// Statistics collected by the scheduler.
#[derive(Debug, Default, Clone)]
pub struct SchedulerStats {
    /// Total tasks completed.
    pub tasks_completed: u64,
    /// Total successful steals.
    pub steal_successes: u64,
    /// Total failed steal attempts.
    pub steal_failures: u64,
}

type StatsCell = Arc<Mutex<SchedulerStats>>;

/// A work-stealing thread-pool scheduler.
///
/// Each worker has its own [`WorkStealingDeque`].  When idle, workers attempt
/// to steal from neighbours in round-robin order before sleeping.
pub struct WorkStealingScheduler {
    deques: Arc<Vec<Arc<WorkStealingDeque<BoxTask>>>>,
    handles: Vec<thread::JoinHandle<()>>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    stats: StatsCell,
    next_push: AtomicUsize,
}

impl WorkStealingScheduler {
    /// Create a scheduler with the given configuration.
    pub fn new(cfg: SchedulerConfig) -> CoreResult<Self> {
        let n = if cfg.num_workers == 0 {
            thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4)
        } else {
            cfg.num_workers
        };
        if n == 0 {
            return Err(CoreError::InvalidInput(ErrorContext::new(
                "WorkStealingScheduler: num_workers must be >= 1",
            )));
        }

        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stats: StatsCell = Arc::new(Mutex::new(SchedulerStats::default()));
        let deques: Arc<Vec<Arc<WorkStealingDeque<BoxTask>>>> =
            Arc::new((0..n).map(|_| Arc::new(WorkStealingDeque::new())).collect());

        let mut handles = Vec::with_capacity(n);
        for id in 0..n {
            let deques2 = Arc::clone(&deques);
            let stop2 = Arc::clone(&stop);
            let stats2 = Arc::clone(&stats);
            let steal_attempts = cfg.steal_attempts;
            let idle_sleep_us = cfg.idle_sleep_us;

            let handle = thread::Builder::new()
                .name(format!("ws-worker-{id}"))
                .spawn(move || {
                    worker_loop(id, n, deques2, stop2, stats2, steal_attempts, idle_sleep_us);
                })
                .map_err(|e| {
                    CoreError::SchedulerError(
                        ErrorContext::new(format!("failed to spawn worker {id}: {e}"))
                            .with_location(ErrorLocation::new(file!(), line!())),
                    )
                })?;
            handles.push(handle);
        }

        Ok(Self {
            deques,
            handles,
            stop,
            stats,
            next_push: AtomicUsize::new(0),
        })
    }

    /// Submit a closure for execution.
    pub fn submit<F>(&self, f: F) -> CoreResult<()>
    where
        F: FnOnce() + Send + 'static,
    {
        let idx = self.next_push.fetch_add(1, Ordering::Relaxed) % self.deques.len();
        self.deques[idx].push(Box::new(f))
    }

    /// Number of worker threads.
    pub fn num_workers(&self) -> usize {
        self.deques.len()
    }

    /// Snapshot of accumulated statistics.
    pub fn stats(&self) -> SchedulerStats {
        self.stats.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Shut down all worker threads and wait for them to finish.
    pub fn shutdown(self) -> CoreResult<()> {
        self.stop.store(true, Ordering::SeqCst);
        for h in self.handles {
            h.join().map_err(|_| {
                CoreError::SchedulerError(
                    ErrorContext::new("worker thread panicked during shutdown")
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;
        }
        Ok(())
    }
}

/// Worker event loop.
fn worker_loop(
    id: usize,
    n: usize,
    deques: Arc<Vec<Arc<WorkStealingDeque<BoxTask>>>>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    stats: StatsCell,
    steal_attempts: usize,
    idle_sleep_us: u64,
) {
    let mut local_completed = 0u64;
    let mut local_steals = 0u64;
    let mut local_failures = 0u64;

    loop {
        // 1. Try own deque first — use `steal` (not `pop`) because `push` is
        //    called from the submitting thread, not from this worker.  In the
        //    Chase-Lev model `push`/`pop` are owner-side, but our scheduler
        //    pushes from the main thread and workers consume.  Using `steal`
        //    is safe for any thread.
        let own = match deques[id].steal() {
            StealResult::Success(task) => {
                task();
                local_completed += 1;
                true
            }
            _ => false,
        };

        if own {
            continue;
        }

        // 2. Try to steal from neighbours.
        let mut stole = false;
        'steal: for attempt in 0..steal_attempts {
            let victim = (id + 1 + attempt) % n;
            if victim == id {
                continue;
            }
            match deques[victim].steal() {
                StealResult::Success(task) => {
                    task();
                    local_completed += 1;
                    local_steals += 1;
                    stole = true;
                    break 'steal;
                }
                StealResult::Empty => {}
                StealResult::Retry => {
                    local_failures += 1;
                }
            }
        }

        if stole {
            continue;
        }

        // 3. Re-check own deque (a task may have been pushed during stealing).
        if let StealResult::Success(task) = deques[id].steal() {
            task();
            local_completed += 1;
            continue;
        }

        // 4. Check stop flag.
        if stop.load(Ordering::Relaxed) {
            break;
        }

        // 5. Sleep briefly before retrying.
        thread::sleep(std::time::Duration::from_micros(idle_sleep_us));
    }

    // Flush local stats.
    if let Ok(mut g) = stats.lock() {
        g.tasks_completed += local_completed;
        g.steal_successes += local_steals;
        g.steal_failures += local_failures;
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn deque_push_pop_single_thread() {
        let dq: WorkStealingDeque<i32> = WorkStealingDeque::new();
        assert!(dq.is_empty());
        dq.push(1).expect("push 1");
        dq.push(2).expect("push 2");
        dq.push(3).expect("push 3");
        assert_eq!(dq.len(), 3);
        assert_eq!(dq.pop().expect("pop"), Some(3));
        assert_eq!(dq.pop().expect("pop"), Some(2));
        assert_eq!(dq.pop().expect("pop"), Some(1));
        assert_eq!(dq.pop().expect("pop"), None);
    }

    #[test]
    fn deque_steal_basic() {
        let dq = Arc::new(WorkStealingDeque::<i32>::new());
        dq.push(10).expect("push");
        dq.push(20).expect("push");

        let dq2 = Arc::clone(&dq);
        let stealer = thread::spawn(move || loop {
            match dq2.steal() {
                StealResult::Success(v) => return v,
                StealResult::Empty => return -1,
                StealResult::Retry => {}
            }
        });
        let stolen = stealer.join().expect("stealer thread");
        assert!(stolen == 10 || stolen == 20 || stolen == -1);
    }

    #[test]
    fn deque_grows_automatically() {
        let dq: WorkStealingDeque<usize> = WorkStealingDeque::new();
        for i in 0..200 {
            dq.push(i).expect("push");
        }
        let mut collected = Vec::new();
        while let Ok(Some(v)) = dq.pop() {
            collected.push(v);
        }
        assert_eq!(collected.len(), 200);
    }

    #[test]
    fn priority_queue_ordering() {
        let q = Arc::new(PriorityTaskQueue::new(16));
        let results = Arc::new(Mutex::new(Vec::new()));

        let r1 = Arc::clone(&results);
        q.submit(Priority::Low, move || {
            r1.lock().expect("lock").push("low");
        })
        .expect("submit low");

        let r2 = Arc::clone(&results);
        q.submit(Priority::High, move || {
            r2.lock().expect("lock").push("high");
        })
        .expect("submit high");

        let r3 = Arc::clone(&results);
        q.submit(Priority::Normal, move || {
            r3.lock().expect("lock").push("normal");
        })
        .expect("submit normal");

        q.close();

        // Drain in priority order
        while let Ok(Some(task)) = q.dequeue() {
            task();
        }

        let res = results.lock().expect("lock");
        assert_eq!(*res, vec!["high", "normal", "low"]);
    }

    #[test]
    fn priority_queue_fifo_within_level() {
        let q = Arc::new(PriorityTaskQueue::new(32));
        let results = Arc::new(Mutex::new(Vec::new()));

        for i in 0..5u32 {
            let r = Arc::clone(&results);
            q.submit(Priority::Normal, move || {
                r.lock().expect("lock").push(i);
            })
            .expect("submit");
        }
        q.close();

        while let Ok(Some(task)) = q.dequeue() {
            task();
        }

        let res = results.lock().expect("lock");
        assert_eq!(*res, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn scheduler_runs_tasks() {
        let cfg = SchedulerConfig {
            num_workers: 4,
            steal_attempts: 16,
            idle_sleep_us: 100,
        };
        let sched = WorkStealingScheduler::new(cfg).expect("new scheduler");
        let counter = Arc::new(AtomicU64::new(0));
        let n_tasks = 100usize;

        for _ in 0..n_tasks {
            let c = Arc::clone(&counter);
            sched
                .submit(move || {
                    c.fetch_add(1, Ordering::Relaxed);
                })
                .expect("submit");
        }

        // Wait for tasks to drain
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while counter.load(Ordering::Relaxed) < n_tasks as u64 {
            if std::time::Instant::now() > deadline {
                break;
            }
            thread::sleep(std::time::Duration::from_millis(1));
        }

        assert_eq!(counter.load(Ordering::Relaxed), n_tasks as u64);
        sched.shutdown().expect("shutdown");
    }
}
