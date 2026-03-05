//! Async-style utilities using thread-based concurrency (no async/await runtime).
//!
//! This module provides composable, thread-safe concurrent utilities that mimic
//! asynchronous patterns using standard OS threads, `Mutex`/`Condvar`, and
//! optionally Rayon for data-parallelism.
//!
//! # Overview
//!
//! | Type | Description |
//! |------|-------------|
//! | [`Channel<T>`] | Bounded MPMC channel (multi-producer, multi-consumer) |
//! | [`Pipeline<T,U>`] | Composable processing pipeline with a dedicated worker thread |
//! | [`BatchProcessor<T,U>`] | Parallel batch processor with configurable batch size |
//! | [`FutureValue<T>`] | One-shot future backed by `Mutex` + `Condvar` |
//! | [`parallel_map`] | Parallel map over a slice using Rayon |
//! | [`ThrottledIterator<T>`] | Rate-limited wrapper over any `Iterator` |
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::async_utils::{Channel, FutureValue};
//! use std::sync::Arc;
//!
//! // Bounded channel
//! let ch = Arc::new(Channel::<i32>::new(8));
//! let ch2 = Arc::clone(&ch);
//!
//! let producer = std::thread::spawn(move || {
//!     for i in 0..4 {
//!         ch2.send(i).ok();
//!     }
//!     ch2.close();
//! });
//!
//! producer.join().ok();
//!
//! let received: Vec<i32> = ch.iter().collect();
//! assert_eq!(received, vec![0, 1, 2, 3]);
//!
//! // One-shot future
//! let fv = Arc::new(FutureValue::<u64>::new());
//! let fv2 = Arc::clone(&fv);
//! std::thread::spawn(move || { fv2.complete(42); }).join().ok();
//! assert_eq!(fv.get(), Some(42));
//! ```

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::error::{CoreError, CoreResult, ErrorContext};

// ============================================================================
// Error helpers
// ============================================================================

fn closed_err(label: &'static str) -> CoreError {
    CoreError::InvalidInput(ErrorContext::new(format!("{label}: channel is closed")))
}

fn full_err(label: &'static str) -> CoreError {
    CoreError::InvalidInput(ErrorContext::new(format!("{label}: channel is full")))
}

// ============================================================================
// Channel<T> — Bounded MPMC channel
// ============================================================================

/// Internal state shared between all channel handles.
struct ChannelInner<T> {
    buf: VecDeque<T>,
    capacity: usize,
    closed: bool,
}

/// A bounded, multi-producer / multi-consumer channel.
///
/// Producers block when the internal buffer is full; consumers block when it is
/// empty.  The channel becomes permanently closed via [`Channel::close`] after
/// which further sends return `Err` and further receives drain remaining items
/// before returning `None`.
pub struct Channel<T> {
    inner: Mutex<ChannelInner<T>>,
    not_empty: Condvar,
    not_full: Condvar,
    sender_count: AtomicUsize,
}

impl<T: Send> Channel<T> {
    /// Create a new channel with the given buffer `capacity` (must be ≥ 1).
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            inner: Mutex::new(ChannelInner {
                buf: VecDeque::with_capacity(capacity),
                capacity,
                closed: false,
            }),
            not_empty: Condvar::new(),
            not_full: Condvar::new(),
            sender_count: AtomicUsize::new(0),
        }
    }

    /// Send a value, **blocking** until space is available.
    ///
    /// Returns `Err` if the channel has been closed.
    pub fn send(&self, value: T) -> CoreResult<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| closed_err("Channel::send"))?;
        loop {
            if guard.closed {
                return Err(closed_err("Channel::send"));
            }
            if guard.buf.len() < guard.capacity {
                guard.buf.push_back(value);
                self.not_empty.notify_one();
                return Ok(());
            }
            guard = self
                .not_full
                .wait(guard)
                .map_err(|_| closed_err("Channel::send"))?;
        }
    }

    /// Try to send a value without blocking.
    ///
    /// Returns `Err` if the channel is full or closed.
    pub fn try_send(&self, value: T) -> CoreResult<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| closed_err("Channel::try_send"))?;
        if guard.closed {
            return Err(closed_err("Channel::try_send"));
        }
        if guard.buf.len() < guard.capacity {
            guard.buf.push_back(value);
            self.not_empty.notify_one();
            Ok(())
        } else {
            Err(full_err("Channel::try_send"))
        }
    }

    /// Receive the next value, **blocking** until one is available.
    ///
    /// Returns `None` when the channel is closed and empty.
    pub fn recv(&self) -> Option<T> {
        let mut guard = self.inner.lock().ok()?;
        loop {
            if let Some(v) = guard.buf.pop_front() {
                self.not_full.notify_one();
                return Some(v);
            }
            if guard.closed {
                return None;
            }
            guard = self.not_empty.wait(guard).ok()?;
        }
    }

    /// Try to receive a value without blocking.  Returns `None` immediately if
    /// the buffer is empty.
    pub fn try_recv(&self) -> Option<T> {
        let mut guard = self.inner.lock().ok()?;
        let v = guard.buf.pop_front()?;
        self.not_full.notify_one();
        Some(v)
    }

    /// Close the channel.  Outstanding items can still be consumed by receivers.
    pub fn close(&self) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.closed = true;
        }
        self.not_empty.notify_all();
        self.not_full.notify_all();
    }

    /// Returns `true` if the channel has been closed.
    pub fn is_closed(&self) -> bool {
        self.inner
            .lock()
            .map(|g| g.closed)
            .unwrap_or(true)
    }

    /// Number of items currently buffered.
    pub fn len(&self) -> usize {
        self.inner.lock().map(|g| g.buf.len()).unwrap_or(0)
    }

    /// Returns `true` when the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an iterator that drains the channel until it is closed and empty.
    pub fn iter(&self) -> ChannelIter<'_, T> {
        ChannelIter { channel: self }
    }

    /// Register one logical sender (for reference counting).
    pub fn register_sender(&self) {
        self.sender_count.fetch_add(1, Ordering::SeqCst);
    }

    /// Unregister a sender; when the count reaches zero the channel is closed
    /// automatically.
    pub fn unregister_sender(&self) {
        let prev = self.sender_count.fetch_sub(1, Ordering::SeqCst);
        if prev == 1 {
            self.close();
        }
    }
}

/// An iterator over a [`Channel`] that blocks until items arrive or the
/// channel is closed.
pub struct ChannelIter<'a, T: Send> {
    channel: &'a Channel<T>,
}

impl<'a, T: Send> Iterator for ChannelIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.channel.recv()
    }
}

// ============================================================================
// FutureValue<T> — one-shot future
// ============================================================================

/// A one-shot, thread-safe value container that acts like a simple future.
///
/// One thread *completes* the future; any number of threads can *wait* for it.
pub struct FutureValue<T> {
    inner: Mutex<Option<T>>,
    ready: Condvar,
    completed: AtomicBool,
}

impl<T: Clone + Send> FutureValue<T> {
    /// Create a new, incomplete `FutureValue`.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(None),
            ready: Condvar::new(),
            completed: AtomicBool::new(false),
        }
    }

    /// Complete the future with `value`.
    pub fn complete(&self, value: T) {
        if let Ok(mut guard) = self.inner.lock() {
            *guard = Some(value);
            self.completed.store(true, Ordering::Release);
        }
        self.ready.notify_all();
    }

    /// Block until the future is complete and return a clone of the value.
    /// Returns `None` if the mutex is poisoned.
    pub fn get(&self) -> Option<T> {
        let mut guard = self.inner.lock().ok()?;
        while guard.is_none() {
            guard = self.ready.wait(guard).ok()?;
        }
        guard.clone()
    }

    /// Non-blocking poll.  Returns `Some(value)` if already complete.
    pub fn try_get(&self) -> Option<T> {
        if self.completed.load(Ordering::Acquire) {
            self.inner.lock().ok()?.clone()
        } else {
            None
        }
    }

    /// Returns `true` if the future has been completed.
    pub fn is_complete(&self) -> bool {
        self.completed.load(Ordering::Acquire)
    }

    /// Block with a timeout.  Returns `None` on timeout or mutex poison.
    pub fn get_timeout(&self, timeout: Duration) -> Option<T> {
        let deadline = Instant::now() + timeout;
        let mut guard = self.inner.lock().ok()?;
        while guard.is_none() {
            let remaining = deadline.checked_duration_since(Instant::now())?;
            let (g, result) = self.ready.wait_timeout(guard, remaining).ok()?;
            guard = g;
            if result.timed_out() {
                return None;
            }
        }
        guard.clone()
    }
}

// ============================================================================
// Pipeline<T, U> — composable single-stage processing pipeline
// ============================================================================

/// A single-stage processing pipeline backed by a dedicated worker thread.
///
/// Items are sent to an internal input [`Channel`]; the worker applies a
/// user-supplied closure and places results in an output channel.
///
/// Drop the `Pipeline` to stop the worker after draining in-flight items.
pub struct Pipeline<T: Send + 'static, U: Send + 'static> {
    input: Arc<Channel<T>>,
    output: Arc<Channel<U>>,
    _handle: thread::JoinHandle<()>,
}

impl<T: Send + 'static, U: Send + 'static> Pipeline<T, U> {
    /// Create a new pipeline with `buf_size` items of internal buffering.
    ///
    /// `f` is called for each input item to produce an output item.
    pub fn new<F>(buf_size: usize, f: F) -> Self
    where
        F: Fn(T) -> U + Send + 'static,
    {
        let input: Arc<Channel<T>> = Arc::new(Channel::new(buf_size));
        let output: Arc<Channel<U>> = Arc::new(Channel::new(buf_size));
        let inp = Arc::clone(&input);
        let out = Arc::clone(&output);

        let handle = thread::spawn(move || {
            while let Some(item) = inp.recv() {
                let result = f(item);
                if out.send(result).is_err() {
                    break;
                }
            }
            out.close();
        });

        Self {
            input,
            output,
            _handle: handle,
        }
    }

    /// Send one item into the pipeline.
    pub fn send(&self, item: T) -> CoreResult<()> {
        self.input.send(item)
    }

    /// Receive one processed result (blocks until available or pipeline drains).
    pub fn recv(&self) -> Option<U> {
        self.output.recv()
    }

    /// Close the input end; the worker will finish processing buffered items.
    pub fn close_input(&self) {
        self.input.close();
    }
}

// ============================================================================
// BatchProcessor<T, U> — batched parallel processor
// ============================================================================

/// Configuration for [`BatchProcessor`].
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of items per batch.
    pub batch_size: usize,
    /// Maximum time to wait before flushing a partial batch.
    pub flush_timeout: Duration,
    /// Number of parallel worker threads.
    pub num_workers: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 64,
            flush_timeout: Duration::from_millis(10),
            num_workers: 4,
        }
    }
}

/// Result of processing a single batch.
#[derive(Debug)]
pub struct BatchResult<U> {
    /// Processed items.
    pub items: Vec<U>,
    /// Time taken to process this batch.
    pub elapsed: Duration,
    /// Batch index (monotonically increasing).
    pub batch_idx: usize,
}

/// A batched parallel processor that collects input items into batches and
/// dispatches each batch to a pool of worker threads.
pub struct BatchProcessor<T: Send + 'static, U: Send + 'static> {
    input: Arc<Channel<T>>,
    output: Arc<Channel<BatchResult<U>>>,
    config: BatchConfig,
    _workers: Vec<thread::JoinHandle<()>>,
    _dispatcher: thread::JoinHandle<()>,
}

impl<T: Send + 'static, U: Send + 'static> BatchProcessor<T, U> {
    /// Build a new `BatchProcessor`.
    ///
    /// - `config`: batching parameters
    /// - `batch_fn`: `Arc`-wrapped closure mapping `Vec<T>` → `Vec<U>`
    pub fn new<F>(config: BatchConfig, batch_fn: Arc<F>) -> Self
    where
        F: Fn(Vec<T>) -> Vec<U> + Send + Sync + 'static + ?Sized,
    {
        let input: Arc<Channel<T>> = Arc::new(Channel::new(config.batch_size * config.num_workers));
        let output: Arc<Channel<BatchResult<U>>> =
            Arc::new(Channel::new(config.num_workers * 4));

        // Intermediate channel carrying Vec<T> batches to workers
        let batch_ch: Arc<Channel<(usize, Vec<T>)>> =
            Arc::new(Channel::new(config.num_workers * 2));

        // --- Dispatcher thread: collects individual items into batches ---
        let inp_d = Arc::clone(&input);
        let batch_ch_d = Arc::clone(&batch_ch);
        let batch_size = config.batch_size;
        let flush_timeout = config.flush_timeout;

        let dispatcher = thread::spawn(move || {
            let mut current: Vec<T> = Vec::with_capacity(batch_size);
            let mut batch_idx: usize = 0;
            let mut last_flush = Instant::now();

            loop {
                match inp_d.try_recv() {
                    Some(item) => {
                        current.push(item);
                        if current.len() >= batch_size {
                            let batch = std::mem::replace(
                                &mut current,
                                Vec::with_capacity(batch_size),
                            );
                            let _ = batch_ch_d.send((batch_idx, batch));
                            batch_idx += 1;
                            last_flush = Instant::now();
                        }
                    }
                    None => {
                        if !current.is_empty() && last_flush.elapsed() >= flush_timeout {
                            let batch = std::mem::replace(
                                &mut current,
                                Vec::with_capacity(batch_size),
                            );
                            let _ = batch_ch_d.send((batch_idx, batch));
                            batch_idx += 1;
                            last_flush = Instant::now();
                        }
                        if inp_d.is_closed() {
                            break;
                        }
                        thread::sleep(Duration::from_micros(100));
                    }
                }
            }

            // Flush remaining
            if !current.is_empty() {
                let _ = batch_ch_d.send((batch_idx, current));
            }
            batch_ch_d.close();
        });

        // --- Worker threads ---
        let mut workers = Vec::with_capacity(config.num_workers);
        for _ in 0..config.num_workers {
            let batch_ch_w = Arc::clone(&batch_ch);
            let out_w = Arc::clone(&output);
            let fn_w = Arc::clone(&batch_fn);

            let worker = thread::spawn(move || {
                while let Some((idx, batch)) = batch_ch_w.recv() {
                    let start = Instant::now();
                    let items = fn_w(batch);
                    let elapsed = start.elapsed();
                    let _ = out_w.send(BatchResult {
                        items,
                        elapsed,
                        batch_idx: idx,
                    });
                }
            });
            workers.push(worker);
        }

        Self {
            input,
            output,
            config,
            _workers: workers,
            _dispatcher: dispatcher,
        }
    }

    /// Submit one item for processing.
    pub fn submit(&self, item: T) -> CoreResult<()> {
        self.input.send(item)
    }

    /// Receive the next completed batch result.  Blocks until available.
    pub fn recv_result(&self) -> Option<BatchResult<U>> {
        self.output.recv()
    }

    /// Close the input and wait for the pipeline to drain.
    pub fn shutdown(&self) {
        self.input.close();
    }

    /// Current input buffer depth.
    pub fn pending(&self) -> usize {
        self.input.len()
    }

    /// Batch configuration used by this processor.
    pub fn config(&self) -> &BatchConfig {
        &self.config
    }
}

// ============================================================================
// ParallelMap — parallel map over slice
// ============================================================================

/// Parallel map over a slice.
///
/// Uses `rayon` when the `parallel` feature is enabled; falls back to a
/// sequential map otherwise.  Returns a `Vec<U>` preserving input order.
pub fn parallel_map<T, U, F>(items: &[T], f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> U + Sync,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        items.par_iter().map(|item| f(item)).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        items.iter().map(|item| f(item)).collect()
    }
}

/// Parallel map with owned items — consumes the `Vec<T>`.
pub fn parallel_map_owned<T, U, F>(items: Vec<T>, f: F) -> Vec<U>
where
    T: Send,
    U: Send,
    F: Fn(T) -> U + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        items.into_par_iter().map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        items.into_iter().map(f).collect()
    }
}

// ============================================================================
// ThrottledIterator<T> — rate-limited iterator
// ============================================================================

/// A rate-limited iterator wrapper.
///
/// Inserts a configurable delay between successive calls to `next()`.
///
/// # Example
///
/// ```rust
/// use scirs2_core::async_utils::ThrottledIterator;
/// use std::time::Duration;
///
/// let data = vec![1, 2, 3];
/// let mut throttled = ThrottledIterator::new(data.into_iter(), Duration::from_millis(1));
/// let collected: Vec<_> = throttled.collect();
/// assert_eq!(collected, vec![1, 2, 3]);
/// ```
pub struct ThrottledIterator<I: Iterator> {
    inner: I,
    delay: Duration,
    last_yield: Option<Instant>,
}

impl<I: Iterator> ThrottledIterator<I> {
    /// Wrap `inner` with a minimum `delay` between items.
    pub fn new(inner: I, delay: Duration) -> Self {
        Self {
            inner,
            delay,
            last_yield: None,
        }
    }

    /// Change the inter-item delay at runtime.
    pub fn set_delay(&mut self, delay: Duration) {
        self.delay = delay;
    }

    /// The current inter-item delay.
    pub fn delay(&self) -> Duration {
        self.delay
    }
}

impl<I: Iterator> Iterator for ThrottledIterator<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<I::Item> {
        if let Some(last) = self.last_yield {
            let elapsed = last.elapsed();
            if elapsed < self.delay {
                thread::sleep(self.delay - elapsed);
            }
        }
        let item = self.inner.next()?;
        self.last_yield = Some(Instant::now());
        Some(item)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn channel_send_recv_basic() {
        let ch = Arc::new(Channel::<i32>::new(4));
        let ch2 = Arc::clone(&ch);

        let handle = thread::spawn(move || {
            for i in 0..4 {
                ch2.send(i).expect("send should succeed");
            }
            ch2.close();
        });
        handle.join().expect("producer join");

        let received: Vec<i32> = ch.iter().collect();
        assert_eq!(received, vec![0, 1, 2, 3]);
    }

    #[test]
    fn channel_try_send_full_returns_err() {
        let ch = Channel::<i32>::new(2);
        ch.try_send(1).expect("first");
        ch.try_send(2).expect("second");
        assert!(ch.try_send(3).is_err(), "channel should be full");
    }

    #[test]
    fn channel_closed_send_returns_err() {
        let ch = Channel::<i32>::new(4);
        ch.close();
        assert!(ch.send(1).is_err());
    }

    #[test]
    fn future_value_complete_and_get() {
        let fv = Arc::new(FutureValue::<u64>::new());
        let fv2 = Arc::clone(&fv);
        thread::spawn(move || fv2.complete(99)).join().ok();
        assert_eq!(fv.get(), Some(99));
        assert_eq!(fv.try_get(), Some(99));
    }

    #[test]
    fn future_value_timeout() {
        let fv = FutureValue::<u64>::new();
        let result = fv.get_timeout(Duration::from_millis(10));
        assert!(result.is_none(), "should time out");
    }

    #[test]
    fn pipeline_basic() {
        let p: Pipeline<i32, i32> = Pipeline::new(8, |x| x * 2);
        for i in 0..5 {
            p.send(i).expect("send");
        }
        p.close_input();
        let mut results = Vec::new();
        while let Some(v) = p.recv() {
            results.push(v);
        }
        results.sort();
        assert_eq!(results, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn throttled_iterator_basic() {
        let data = vec![10, 20, 30];
        let throttled =
            ThrottledIterator::new(data.into_iter(), Duration::from_micros(500));
        let result: Vec<_> = throttled.collect();
        assert_eq!(result, vec![10, 20, 30]);
    }

    #[test]
    fn parallel_map_order_preserved() {
        let data: Vec<i32> = (0..100).collect();
        let doubled = parallel_map(&data, |x| x * 2);
        for (i, v) in doubled.iter().enumerate() {
            assert_eq!(*v, (i as i32) * 2);
        }
    }

    #[test]
    fn batch_processor_processes_all_items() {
        let config = BatchConfig {
            batch_size: 4,
            flush_timeout: Duration::from_millis(5),
            num_workers: 2,
        };
        let fn_arc: Arc<dyn Fn(Vec<i32>) -> Vec<i32> + Send + Sync> =
            Arc::new(|batch: Vec<i32>| batch.into_iter().map(|x| x * 3).collect());

        let bp = BatchProcessor::new(config, fn_arc);
        let total_items = 12usize;
        for i in 0..total_items {
            bp.submit(i as i32).expect("submit");
        }
        bp.shutdown();

        // Collect results
        let mut collected = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        while collected.len() < total_items {
            if Instant::now() > deadline {
                break;
            }
            if let Some(res) = bp.recv_result() {
                collected.extend(res.items);
            } else {
                thread::sleep(Duration::from_millis(1));
            }
        }
        assert_eq!(collected.len(), total_items);
    }
}
