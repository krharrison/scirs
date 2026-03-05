//! Reactive stream primitives — push-based, iterator-compatible streams.
//!
//! This module provides composable stream abstractions that model data flows as
//! sequences of values with standard combinators (`map`, `filter`, `take`,
//! `skip`, `flatten`, `zip`, `merge`).  Everything is built on top of
//! `std::sync` — no async runtime is required.
//!
//! # Overview
//!
//! | Type | Description |
//! |------|-------------|
//! | [`StreamSource`] trait | Dyn-compatible core trait (next only) |
//! | [`Stream`] trait | Full combinators trait (requires `Sized`) |
//! | [`InfiniteStream<T>`] | Iterator-backed stream |
//! | [`Subject<T>`] | Broadcast subject (push-based observable) |
//! | [`WindowedStream<T>`] | Tumbling / sliding window |
//! | [`ZipStream<A,B>`] | Element-wise zip of two streams |
//! | [`MergeStream<T>`] | Round-robin merge of multiple streams |
//! | [`BackpressureBuffer<T>`] | Bounded buffer with backpressure signaling |
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::reactive::{InfiniteStream, Stream};
//!
//! let mut s = InfiniteStream::from_iter(0..10);
//! let evens: Vec<i32> = s
//!     .by_ref()
//!     .filter(|x| x % 2 == 0)
//!     .take(3)
//!     .collect_stream();
//! assert_eq!(evens, vec![0, 2, 4]);
//! ```

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::error::{CoreError, CoreResult, ErrorContext};

// ============================================================================
// StreamSource — dyn-compatible core trait
// ============================================================================

/// The dyn-compatible core trait for streams.
///
/// Implement this for any type that produces a sequence of values.  Unlike
/// [`Stream`] (which requires `Self: Sized`), `StreamSource` can be used as a
/// trait object (`Box<dyn StreamSource<Item = T>>`).
pub trait StreamSource {
    /// The type of items produced.
    type Item;

    /// Advance the stream and return the next item, or `None` when exhausted.
    fn next_item(&mut self) -> Option<Self::Item>;
}

// ============================================================================
// Stream — rich combinator trait
// ============================================================================

/// Full stream combinator trait.
///
/// Automatically implemented for any `T: StreamSource + Sized`.  Provides the
/// ergonomic adaptor API (`map`, `filter`, `take`, etc.).
///
/// For boxed, type-erased streams use [`StreamSource`] directly or wrap in
/// [`BoxedStream`].
pub trait Stream: StreamSource + Sized {
    /// Apply `f` to each item, producing a new stream.
    fn map<U, F>(self, f: F) -> MapStream<Self, F>
    where
        F: FnMut(Self::Item) -> U,
    {
        MapStream { inner: self, f }
    }

    /// Keep only items for which `pred` returns `true`.
    fn filter<P>(self, pred: P) -> FilterStream<Self, P>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        FilterStream { inner: self, pred }
    }

    /// Take at most `n` items then stop.
    fn take(self, n: usize) -> TakeStream<Self> {
        TakeStream {
            inner: self,
            remaining: n,
        }
    }

    /// Skip the first `n` items then yield the rest.
    fn skip(self, n: usize) -> SkipStream<Self> {
        SkipStream {
            inner: self,
            remaining: n,
        }
    }

    /// Flatten a stream of iterables into a flat stream.
    fn flatten(self) -> FlattenStream<Self>
    where
        Self::Item: IntoIterator,
    {
        FlattenStream {
            outer: self,
            current: None,
        }
    }

    /// Borrow this stream mutably, producing an adaptor that forwards to it.
    fn by_ref(&mut self) -> ByRefStream<'_, Self> {
        ByRefStream { inner: self }
    }

    /// Collect all remaining items into a `Vec`.
    fn collect_stream(mut self) -> Vec<Self::Item> {
        let mut out = Vec::new();
        while let Some(item) = self.next_item() {
            out.push(item);
        }
        out
    }

    /// Count remaining items (exhausts the stream).
    fn count_stream(mut self) -> usize {
        let mut n = 0;
        while self.next_item().is_some() {
            n += 1;
        }
        n
    }

    /// Apply `f` to each item for its side-effects.
    fn for_each_stream<F: FnMut(Self::Item)>(mut self, mut f: F) {
        while let Some(item) = self.next_item() {
            f(item);
        }
    }
}

// Blanket implementation: everything that is StreamSource + Sized gets Stream.
impl<S: StreamSource + Sized> Stream for S {}

// ============================================================================
// Adaptor types
// ============================================================================

/// Stream returned by [`Stream::map`].
pub struct MapStream<S, F> {
    inner: S,
    f: F,
}

impl<S: StreamSource, U, F: FnMut(S::Item) -> U> StreamSource for MapStream<S, F> {
    type Item = U;

    fn next_item(&mut self) -> Option<U> {
        self.inner.next_item().map(|item| (self.f)(item))
    }
}

/// Stream returned by [`Stream::filter`].
pub struct FilterStream<S, P> {
    inner: S,
    pred: P,
}

impl<S: StreamSource, P: FnMut(&S::Item) -> bool> StreamSource for FilterStream<S, P> {
    type Item = S::Item;

    fn next_item(&mut self) -> Option<S::Item> {
        loop {
            let item = self.inner.next_item()?;
            if (self.pred)(&item) {
                return Some(item);
            }
        }
    }
}

/// Stream returned by [`Stream::take`].
pub struct TakeStream<S> {
    inner: S,
    remaining: usize,
}

impl<S: StreamSource> StreamSource for TakeStream<S> {
    type Item = S::Item;

    fn next_item(&mut self) -> Option<S::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        self.inner.next_item()
    }
}

/// Stream returned by [`Stream::skip`].
pub struct SkipStream<S> {
    inner: S,
    remaining: usize,
}

impl<S: StreamSource> StreamSource for SkipStream<S> {
    type Item = S::Item;

    fn next_item(&mut self) -> Option<S::Item> {
        while self.remaining > 0 {
            self.inner.next_item()?;
            self.remaining -= 1;
        }
        self.inner.next_item()
    }
}

/// Stream returned by [`Stream::flatten`].
pub struct FlattenStream<S: StreamSource>
where
    S::Item: IntoIterator,
{
    outer: S,
    current: Option<<S::Item as IntoIterator>::IntoIter>,
}

impl<S: StreamSource> StreamSource for FlattenStream<S>
where
    S::Item: IntoIterator,
{
    type Item = <S::Item as IntoIterator>::Item;

    fn next_item(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(ref mut iter) = self.current {
                if let Some(item) = iter.next() {
                    return Some(item);
                }
            }
            let next_outer = self.outer.next_item()?;
            self.current = Some(next_outer.into_iter());
        }
    }
}

/// Mutable reference adaptor returned by [`Stream::by_ref`].
pub struct ByRefStream<'a, S> {
    inner: &'a mut S,
}

impl<'a, S: StreamSource> StreamSource for ByRefStream<'a, S> {
    type Item = S::Item;

    fn next_item(&mut self) -> Option<S::Item> {
        self.inner.next_item()
    }
}

// ============================================================================
// BoxedStream — type-erased stream wrapper
// ============================================================================

/// A heap-allocated, type-erased stream.
///
/// Useful when you need to store heterogeneous streams in a collection or
/// return a stream from a function without exposing the concrete type.
pub struct BoxedStream<T> {
    inner: Box<dyn StreamSource<Item = T> + Send>,
}

impl<T> BoxedStream<T> {
    /// Wrap any `StreamSource` in a heap allocation.
    pub fn new<S>(s: S) -> Self
    where
        S: StreamSource<Item = T> + Send + 'static,
    {
        Self { inner: Box::new(s) }
    }
}

impl<T> StreamSource for BoxedStream<T> {
    type Item = T;

    fn next_item(&mut self) -> Option<T> {
        self.inner.next_item()
    }
}

// ============================================================================
// InfiniteStream<T>
// ============================================================================

/// A stream backed by any Rust `Iterator`.
///
/// The stream ends when the underlying iterator is exhausted.
pub struct InfiniteStream<I: Iterator> {
    iter: I,
}

impl<I: Iterator> InfiniteStream<I> {
    /// Wrap an `Iterator` as a `Stream`.
    pub fn from_iter(iter: I) -> Self {
        Self { iter }
    }

    /// Consume the stream and return the underlying iterator.
    pub fn into_inner(self) -> I {
        self.iter
    }
}

impl<I: Iterator> StreamSource for InfiniteStream<I> {
    type Item = I::Item;

    fn next_item(&mut self) -> Option<I::Item> {
        self.iter.next()
    }
}

/// Also implement standard `Iterator` so callers can use `for` loops and
/// standard adapters interchangeably.
impl<I: Iterator> Iterator for InfiniteStream<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<I::Item> {
        StreamSource::next_item(self)
    }
}

// ============================================================================
// Subject<T> — broadcast push-based observable
// ============================================================================

/// Internal subscriber slot.
struct Subscriber<T: Clone> {
    buf: VecDeque<T>,
    closed: bool,
}

/// Shared state for a [`Subject`].
struct SubjectInner<T: Clone> {
    subscribers: Vec<Subscriber<T>>,
    completed: bool,
}

/// A broadcast subject: push values from any thread; any number of subscriber
/// handles can independently pull values.
///
/// # Example
///
/// ```rust
/// use scirs2_core::reactive::Subject;
///
/// let mut subject = Subject::<i32>::new();
/// let rx1 = subject.subscribe();
/// let rx2 = subject.subscribe();
///
/// subject.emit(1);
/// subject.emit(2);
/// subject.complete();
///
/// assert_eq!(rx1.collect_all(), vec![1, 2]);
/// assert_eq!(rx2.collect_all(), vec![1, 2]);
/// ```
pub struct Subject<T: Clone + Send + 'static> {
    inner: Arc<(Mutex<SubjectInner<T>>, Condvar)>,
}

impl<T: Clone + Send + 'static> Subject<T> {
    /// Create a new, empty `Subject`.
    pub fn new() -> Self {
        Self {
            inner: Arc::new((
                Mutex::new(SubjectInner {
                    subscribers: Vec::new(),
                    completed: false,
                }),
                Condvar::new(),
            )),
        }
    }

    /// Subscribe, returning a [`SubjectReceiver`] that receives all future
    /// emitted values.
    pub fn subscribe(&self) -> SubjectReceiver<T> {
        let (lock, _) = &*self.inner;
        let slot_idx = lock
            .lock()
            .map(|mut g| {
                let idx = g.subscribers.len();
                g.subscribers.push(Subscriber {
                    buf: VecDeque::new(),
                    closed: false,
                });
                idx
            })
            .unwrap_or(0);

        SubjectReceiver {
            inner: Arc::clone(&self.inner),
            slot: slot_idx,
        }
    }

    /// Emit one value to all current subscribers.
    pub fn emit(&self, value: T) {
        let (lock, cv) = &*self.inner;
        if let Ok(mut g) = lock.lock() {
            for sub in g.subscribers.iter_mut() {
                if !sub.closed {
                    sub.buf.push_back(value.clone());
                }
            }
        }
        cv.notify_all();
    }

    /// Complete the subject — no further values can be emitted.
    pub fn complete(&self) {
        let (lock, cv) = &*self.inner;
        if let Ok(mut g) = lock.lock() {
            g.completed = true;
            for sub in g.subscribers.iter_mut() {
                sub.closed = true;
            }
        }
        cv.notify_all();
    }

    /// Number of current subscribers.
    pub fn subscriber_count(&self) -> usize {
        let (lock, _) = &*self.inner;
        lock.lock().map(|g| g.subscribers.len()).unwrap_or(0)
    }

    /// `true` if [`complete`](Subject::complete) has been called.
    pub fn is_completed(&self) -> bool {
        let (lock, _) = &*self.inner;
        lock.lock().map(|g| g.completed).unwrap_or(false)
    }
}

/// A subscriber handle for a [`Subject`].
pub struct SubjectReceiver<T: Clone + Send + 'static> {
    inner: Arc<(Mutex<SubjectInner<T>>, Condvar)>,
    slot: usize,
}

impl<T: Clone + Send + 'static> SubjectReceiver<T> {
    /// Block until the next value arrives or the subject is completed.
    pub fn recv(&self) -> Option<T> {
        let (lock, cv) = &*self.inner;
        let mut g = lock.lock().ok()?;
        loop {
            if let Some(v) = g.subscribers.get_mut(self.slot)?.buf.pop_front() {
                return Some(v);
            }
            if g.completed
                || g.subscribers
                    .get(self.slot)
                    .map(|s| s.closed)
                    .unwrap_or(true)
            {
                return None;
            }
            g = cv.wait(g).ok()?;
        }
    }

    /// Try to receive without blocking.
    pub fn try_recv(&self) -> Option<T> {
        let (lock, _) = &*self.inner;
        let mut g = lock.lock().ok()?;
        g.subscribers.get_mut(self.slot)?.buf.pop_front()
    }

    /// Drain all buffered and future values until the subject is completed.
    pub fn collect_all(self) -> Vec<T> {
        let mut result = Vec::new();
        loop {
            match self.recv() {
                Some(v) => result.push(v),
                None => break,
            }
        }
        result
    }
}

// ============================================================================
// WindowedStream<T>
// ============================================================================

/// Window mode for [`WindowedStream`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowMode {
    /// Non-overlapping windows of exactly `size` items.
    Tumbling,
    /// Overlapping windows: advance by `step`, window has `size` items.
    Sliding { step: usize },
}

/// A stream that groups items into fixed-size windows.
///
/// - **Tumbling** windows are non-overlapping (`step = size`).
/// - **Sliding** windows advance by a configurable `step`.
///
/// Each call to `next_item` returns `Some(Vec<T>)` with exactly `size`
/// items (or `None` when the source is exhausted).
pub struct WindowedStream<S: StreamSource>
where
    S::Item: Clone,
{
    inner: S,
    window_size: usize,
    mode: WindowMode,
    buffer: VecDeque<S::Item>,
    exhausted: bool,
}

impl<S: StreamSource> WindowedStream<S>
where
    S::Item: Clone,
{
    /// Create a tumbling window of `size`.
    pub fn tumbling(inner: S, size: usize) -> Self {
        let size = size.max(1);
        Self {
            inner,
            window_size: size,
            mode: WindowMode::Tumbling,
            buffer: VecDeque::new(),
            exhausted: false,
        }
    }

    /// Create a sliding window of `size` that advances by `step`.
    pub fn sliding(inner: S, size: usize, step: usize) -> Self {
        let size = size.max(1);
        let step = step.max(1);
        Self {
            inner,
            window_size: size,
            mode: WindowMode::Sliding { step },
            buffer: VecDeque::new(),
            exhausted: false,
        }
    }

    /// Fill the internal buffer up to `target` items.
    fn fill_to(&mut self, target: usize) {
        while !self.exhausted && self.buffer.len() < target {
            match self.inner.next_item() {
                Some(item) => self.buffer.push_back(item),
                None => {
                    self.exhausted = true;
                    break;
                }
            }
        }
    }
}

impl<S: StreamSource> StreamSource for WindowedStream<S>
where
    S::Item: Clone,
{
    type Item = Vec<S::Item>;

    fn next_item(&mut self) -> Option<Vec<S::Item>> {
        self.fill_to(self.window_size);
        if self.buffer.len() < self.window_size {
            return None;
        }
        let window: Vec<S::Item> = self.buffer.iter().take(self.window_size).cloned().collect();
        match self.mode {
            WindowMode::Tumbling => {
                for _ in 0..self.window_size {
                    self.buffer.pop_front();
                }
            }
            WindowMode::Sliding { step } => {
                for _ in 0..step {
                    self.buffer.pop_front();
                }
            }
        }
        Some(window)
    }
}

// ============================================================================
// ZipStream<A, B>
// ============================================================================

/// A stream that zips two streams element-wise.
///
/// Stops as soon as either stream is exhausted.
pub struct ZipStream<A: StreamSource, B: StreamSource> {
    left: A,
    right: B,
}

impl<A: StreamSource, B: StreamSource> ZipStream<A, B> {
    /// Create a new `ZipStream`.
    pub fn new(left: A, right: B) -> Self {
        Self { left, right }
    }
}

impl<A: StreamSource, B: StreamSource> StreamSource for ZipStream<A, B> {
    type Item = (A::Item, B::Item);

    fn next_item(&mut self) -> Option<(A::Item, B::Item)> {
        let a = self.left.next_item()?;
        let b = self.right.next_item()?;
        Some((a, b))
    }
}

// ============================================================================
// MergeStream<T>
// ============================================================================

/// A stream that merges multiple source streams in round-robin order.
///
/// Each call to `next_item` tries each sub-stream in rotation, returning the
/// first non-`None` value.  Exhausted sub-streams are removed.  Returns
/// `None` when all sub-streams are exhausted.
pub struct MergeStream<T> {
    sources: Vec<BoxedStream<T>>,
    cursor: usize,
}

impl<T: Send + 'static> MergeStream<T> {
    /// Create a `MergeStream` from a vector of `BoxedStream`s.
    pub fn new(sources: Vec<BoxedStream<T>>) -> Self {
        Self { sources, cursor: 0 }
    }

    /// Convenience: build from a vector of concrete stream types.
    pub fn from_streams<S>(streams: Vec<S>) -> Self
    where
        S: StreamSource<Item = T> + Send + 'static,
    {
        let boxed = streams.into_iter().map(BoxedStream::new).collect();
        Self::new(boxed)
    }
}

impl<T: Send + 'static> StreamSource for MergeStream<T> {
    type Item = T;

    fn next_item(&mut self) -> Option<T> {
        let n = self.sources.len();
        if n == 0 {
            return None;
        }
        for attempt in 0..n {
            let idx = (self.cursor + attempt) % n;
            if let Some(item) = self.sources[idx].next_item() {
                self.cursor = (idx + 1) % self.sources.len();
                return Some(item);
            }
        }
        None
    }
}

// ============================================================================
// BackpressureBuffer<T>
// ============================================================================

/// Backpressure state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackpressureSignal {
    /// Buffer has room; producers can continue.
    Normal,
    /// Buffer is above the high-water mark; producers should slow down.
    Throttle,
    /// Buffer is full; producers must block or drop.
    Full,
}

/// A bounded buffer with backpressure signaling.
pub struct BackpressureBuffer<T: Send> {
    inner: Mutex<VecDeque<T>>,
    not_empty: Condvar,
    not_full: Condvar,
    capacity: usize,
    high_water: usize,
}

impl<T: Send> BackpressureBuffer<T> {
    /// Create a new buffer with `capacity` and `high_water_fraction` ∈ (0, 1].
    pub fn new(capacity: usize, high_water_fraction: f64) -> Self {
        let capacity = capacity.max(1);
        let high_water = ((capacity as f64) * high_water_fraction.clamp(0.0, 1.0)) as usize;
        let high_water = high_water.max(1).min(capacity);
        Self {
            inner: Mutex::new(VecDeque::with_capacity(capacity)),
            not_empty: Condvar::new(),
            not_full: Condvar::new(),
            capacity,
            high_water,
        }
    }

    /// Try to push without blocking.
    pub fn try_push(&self, item: T) -> Result<BackpressureSignal, T> {
        // If the mutex is poisoned we return the item back to the caller.
        let mut g = match self.inner.lock() {
            Ok(guard) => guard,
            Err(_) => return Err(item),
        };
        if g.len() >= self.capacity {
            return Err(item);
        }
        g.push_back(item);
        self.not_empty.notify_one();
        let signal = if g.len() >= self.high_water {
            BackpressureSignal::Throttle
        } else {
            BackpressureSignal::Normal
        };
        Ok(signal)
    }

    /// Blocking push — waits until space is available.
    pub fn push(&self, item: T) -> CoreResult<BackpressureSignal> {
        let mut g = self.inner.lock().map_err(|_| {
            CoreError::InvalidInput(ErrorContext::new("BackpressureBuffer: mutex poisoned"))
        })?;

        while g.len() >= self.capacity {
            g = self.not_full.wait(g).map_err(|_| {
                CoreError::InvalidInput(ErrorContext::new(
                    "BackpressureBuffer: condvar poisoned",
                ))
            })?;
        }
        g.push_back(item);
        self.not_empty.notify_one();
        let signal = if g.len() >= self.high_water {
            BackpressureSignal::Throttle
        } else {
            BackpressureSignal::Normal
        };
        Ok(signal)
    }

    /// Non-blocking pop.
    pub fn try_pop(&self) -> Option<T> {
        let mut g = self.inner.lock().ok()?;
        let item = g.pop_front()?;
        self.not_full.notify_one();
        Some(item)
    }

    /// Blocking pop — waits until an item is available.
    pub fn pop(&self) -> Option<T> {
        let mut g = self.inner.lock().ok()?;
        loop {
            if let Some(item) = g.pop_front() {
                self.not_full.notify_one();
                return Some(item);
            }
            g = self.not_empty.wait(g).ok()?;
        }
    }

    /// Pop with a timeout.
    pub fn pop_timeout(&self, timeout: Duration) -> Option<T> {
        let mut g = self.inner.lock().ok()?;
        loop {
            if let Some(item) = g.pop_front() {
                self.not_full.notify_one();
                return Some(item);
            }
            let (ng, result) = self.not_empty.wait_timeout(g, timeout).ok()?;
            g = ng;
            if result.timed_out() {
                return None;
            }
        }
    }

    /// Current number of buffered items.
    pub fn len(&self) -> usize {
        self.inner.lock().map(|g| g.len()).unwrap_or(0)
    }

    /// `true` if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Maximum buffer capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Current backpressure signal (without pushing anything).
    pub fn signal(&self) -> BackpressureSignal {
        let len = self.len();
        if len >= self.capacity {
            BackpressureSignal::Full
        } else if len >= self.high_water {
            BackpressureSignal::Throttle
        } else {
            BackpressureSignal::Normal
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infinite_stream_map_filter_take() {
        let mut s = InfiniteStream::from_iter(0..100i32);
        let result: Vec<i32> = Stream::by_ref(&mut s)
            .map(|x| x * 2)
            .filter(|x| x % 4 == 0)
            .take(5)
            .collect_stream();
        assert_eq!(result, vec![0, 4, 8, 12, 16]);
    }

    #[test]
    fn infinite_stream_skip_take() {
        let s = InfiniteStream::from_iter(0..20i32);
        let result: Vec<i32> = Stream::skip(s, 5).take(5).collect_stream();
        assert_eq!(result, vec![5, 6, 7, 8, 9]);
    }

    #[test]
    fn flatten_stream() {
        let nested = vec![vec![1, 2, 3], vec![4, 5], vec![6]];
        let s = InfiniteStream::from_iter(nested.into_iter());
        let result: Vec<i32> = Stream::flatten(s).collect_stream();
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn zip_stream() {
        let a = InfiniteStream::from_iter(0..5i32);
        let b = InfiniteStream::from_iter(10..15i32);
        let result: Vec<(i32, i32)> = ZipStream::new(a, b).collect_stream();
        assert_eq!(result, vec![(0, 10), (1, 11), (2, 12), (3, 13), (4, 14)]);
    }

    #[test]
    fn tumbling_window() {
        let s = InfiniteStream::from_iter(0..9i32);
        let windows: Vec<Vec<i32>> = WindowedStream::tumbling(s, 3).collect_stream();
        assert_eq!(windows, vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]]);
    }

    #[test]
    fn sliding_window() {
        let s = InfiniteStream::from_iter(0..6i32);
        let windows: Vec<Vec<i32>> = WindowedStream::sliding(s, 3, 1).collect_stream();
        assert_eq!(
            windows,
            vec![
                vec![0, 1, 2],
                vec![1, 2, 3],
                vec![2, 3, 4],
                vec![3, 4, 5],
            ]
        );
    }

    #[test]
    fn merge_stream_round_robin() {
        let s1 = InfiniteStream::from_iter(vec![1, 3, 5].into_iter());
        let s2 = InfiniteStream::from_iter(vec![2, 4, 6].into_iter());
        let result: Vec<i32> = MergeStream::from_streams(vec![s1, s2]).collect_stream();
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn subject_broadcast() {
        let subject = Subject::<i32>::new();
        let rx1 = subject.subscribe();
        let rx2 = subject.subscribe();

        subject.emit(10);
        subject.emit(20);
        subject.complete();

        assert_eq!(rx1.collect_all(), vec![10, 20]);
        assert_eq!(rx2.collect_all(), vec![10, 20]);
    }

    #[test]
    fn backpressure_buffer_basic() {
        let buf = BackpressureBuffer::<i32>::new(4, 0.75);
        assert_eq!(buf.try_push(1), Ok(BackpressureSignal::Normal));
        assert_eq!(buf.try_push(2), Ok(BackpressureSignal::Normal));
        assert_eq!(buf.try_push(3), Ok(BackpressureSignal::Throttle));
        assert_eq!(buf.try_push(4), Ok(BackpressureSignal::Throttle));
        assert!(buf.try_push(5).is_err());
        assert_eq!(buf.try_pop(), Some(1));
        assert_eq!(buf.len(), 3);
    }
}
