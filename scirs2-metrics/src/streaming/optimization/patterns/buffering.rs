//! Ring buffers and streaming buffer abstractions
//!
//! This module provides in-memory buffer primitives optimised for high-throughput
//! streaming pipelines:
//!
//! - [`RingBuffer`]       – classic fixed-capacity circular buffer with O(1) push/pop
//! - [`StreamingBuffer`]  – self-growing buffer with configurable eviction policies
//! - [`DoubleBuffer`]     – double-buffering pattern (write into back, swap to read front)
//!
//! All implementations are lock-free at the type level; callers are responsible
//! for external synchronisation when sharing across threads.

use crate::error::{MetricsError, Result};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant, SystemTime};

// ── RingBuffer ───────────────────────────────────────────────────────────────

/// A fixed-capacity circular (ring) buffer.
///
/// When the buffer is full the oldest element is overwritten by the newest one,
/// maintaining a rolling window of the last `capacity` elements.
///
/// # Type Parameters
/// * `T` – element type; must be `Clone` so that `peek_all` can copy the view.
#[derive(Debug, Clone)]
pub struct RingBuffer<T: Clone> {
    data: Vec<Option<T>>,
    head: usize, // next write position
    len: usize,
    capacity: usize,
}

impl<T: Clone> RingBuffer<T> {
    /// Create a new ring buffer with the given capacity.
    ///
    /// Returns `Err` if `capacity` is 0.
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(MetricsError::InvalidInput(
                "RingBuffer capacity must be >= 1".to_string(),
            ));
        }
        Ok(Self {
            data: vec![None; capacity],
            head: 0,
            len: 0,
            capacity,
        })
    }

    /// Push a value into the buffer.
    ///
    /// Returns the evicted element (if the buffer was already full) so the
    /// caller can react to data loss.
    pub fn push(&mut self, value: T) -> Option<T> {
        let evicted = if self.len == self.capacity {
            self.data[self.head].take()
        } else {
            self.len += 1;
            None
        };

        self.data[self.head] = Some(value);
        self.head = (self.head + 1) % self.capacity;
        evicted
    }

    /// Remove and return the oldest element, or `None` when empty.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        let tail = (self.head + self.capacity - self.len) % self.capacity;
        let value = self.data[tail].take();
        self.len -= 1;
        value
    }

    /// Peek at the oldest element without removing it.
    pub fn peek_oldest(&self) -> Option<&T> {
        if self.len == 0 {
            return None;
        }
        let tail = (self.head + self.capacity - self.len) % self.capacity;
        self.data[tail].as_ref()
    }

    /// Peek at the most recently pushed element without removing it.
    pub fn peek_newest(&self) -> Option<&T> {
        if self.len == 0 {
            return None;
        }
        let newest = (self.head + self.capacity - 1) % self.capacity;
        self.data[newest].as_ref()
    }

    /// Collect all elements in insertion order (oldest first) without removing them.
    pub fn peek_all(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.len);
        for i in 0..self.len {
            let idx = (self.head + self.capacity - self.len + i) % self.capacity;
            if let Some(v) = &self.data[idx] {
                out.push(v.clone());
            }
        }
        out
    }

    /// Number of elements currently stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` when the buffer contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// `true` when the buffer is at capacity.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// Configured maximum number of elements.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Drain all elements and return them in insertion order (oldest first).
    pub fn drain_all(&mut self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.len);
        while let Some(v) = self.pop() {
            out.push(v);
        }
        out
    }
}

impl<F: Float + std::fmt::Debug + Copy + Clone> RingBuffer<F> {
    /// Compute the arithmetic mean of buffered values.
    ///
    /// Returns `None` if the buffer is empty.
    pub fn mean(&self) -> Option<F> {
        if self.is_empty() {
            return None;
        }
        let items = self.peek_all();
        let sum = items.iter().copied().fold(F::zero(), |a, x| a + x);
        Some(sum / F::from(items.len()).expect("usize fits in F"))
    }
}

// ── StreamingBuffer ──────────────────────────────────────────────────────────

/// Eviction policy for [`StreamingBuffer`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamingEvictionPolicy {
    /// Never evict; grow until `max_size` then reject new elements.
    Reject,
    /// Drop the oldest element when full.
    DropOldest,
    /// Drop the newest (incoming) element when full.
    DropNewest,
    /// Remove elements older than the given duration from the back.
    TimeBasedExpiry(Duration),
}

/// A timestamped element stored inside a [`StreamingBuffer`].
#[derive(Debug, Clone)]
pub struct TimestampedItem<T: Clone> {
    /// The stored value.
    pub value: T,
    /// Insertion wall-clock time.
    pub inserted_at: SystemTime,
}

/// A self-managing buffer suitable for streaming workloads.
///
/// Unlike [`RingBuffer`] this buffer grows dynamically up to `max_size`.
/// When it is full the configured `eviction_policy` determines how new elements
/// are handled.
#[derive(Debug)]
pub struct StreamingBuffer<T: Clone> {
    max_size: usize,
    eviction_policy: StreamingEvictionPolicy,
    storage: VecDeque<TimestampedItem<T>>,
    total_inserted: u64,
    total_evicted: u64,
    total_rejected: u64,
}

impl<T: Clone> StreamingBuffer<T> {
    /// Create a new streaming buffer.
    ///
    /// # Arguments
    /// * `max_size`        – upper bound on the number of buffered elements
    /// * `eviction_policy` – what to do when the buffer is full
    pub fn new(max_size: usize, eviction_policy: StreamingEvictionPolicy) -> Result<Self> {
        if max_size == 0 {
            return Err(MetricsError::InvalidInput(
                "StreamingBuffer max_size must be >= 1".to_string(),
            ));
        }
        Ok(Self {
            max_size,
            eviction_policy,
            storage: VecDeque::with_capacity(max_size.min(1024)),
            total_inserted: 0,
            total_evicted: 0,
            total_rejected: 0,
        })
    }

    /// Insert a new element.
    ///
    /// Returns `Ok(true)` if the element was stored, `Ok(false)` if it was
    /// rejected due to the `Reject` eviction policy, and `Err` on unexpected
    /// policy violations.
    pub fn push(&mut self, value: T) -> Result<bool> {
        // Expire time-based entries first
        if let StreamingEvictionPolicy::TimeBasedExpiry(ttl) = &self.eviction_policy.clone() {
            let ttl = *ttl;
            self.expire_old_entries(ttl);
        }

        if self.storage.len() < self.max_size {
            self.storage.push_back(TimestampedItem {
                value,
                inserted_at: SystemTime::now(),
            });
            self.total_inserted += 1;
            return Ok(true);
        }

        // Buffer is full — apply eviction policy
        match &self.eviction_policy {
            StreamingEvictionPolicy::Reject => {
                self.total_rejected += 1;
                Ok(false)
            }
            StreamingEvictionPolicy::DropOldest => {
                self.storage.pop_front();
                self.total_evicted += 1;
                self.storage.push_back(TimestampedItem {
                    value,
                    inserted_at: SystemTime::now(),
                });
                self.total_inserted += 1;
                Ok(true)
            }
            StreamingEvictionPolicy::DropNewest => {
                // Discard the incoming element (behaves like Reject but semantically different)
                self.total_evicted += 1;
                Ok(false)
            }
            StreamingEvictionPolicy::TimeBasedExpiry(_) => {
                // Already expired; if still full, fall back to DropOldest
                self.storage.pop_front();
                self.total_evicted += 1;
                self.storage.push_back(TimestampedItem {
                    value,
                    inserted_at: SystemTime::now(),
                });
                self.total_inserted += 1;
                Ok(true)
            }
        }
    }

    /// Remove and return the oldest element.
    pub fn pop(&mut self) -> Option<TimestampedItem<T>> {
        self.storage.pop_front()
    }

    /// Peek at the oldest element without removing it.
    pub fn peek_oldest(&self) -> Option<&TimestampedItem<T>> {
        self.storage.front()
    }

    /// Drain all elements and return them in insertion order.
    pub fn drain_all(&mut self) -> Vec<TimestampedItem<T>> {
        self.storage.drain(..).collect()
    }

    /// Current number of stored elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// `true` when no elements are stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Total elements ever inserted (including evicted ones).
    #[inline]
    pub fn total_inserted(&self) -> u64 {
        self.total_inserted
    }

    /// Total elements evicted by policy.
    #[inline]
    pub fn total_evicted(&self) -> u64 {
        self.total_evicted
    }

    /// Total elements rejected (policy = `Reject`).
    #[inline]
    pub fn total_rejected(&self) -> u64 {
        self.total_rejected
    }

    fn expire_old_entries(&mut self, ttl: Duration) {
        let now = SystemTime::now();
        while let Some(front) = self.storage.front() {
            let age = now.duration_since(front.inserted_at).unwrap_or_default();
            if age > ttl {
                self.storage.pop_front();
                self.total_evicted += 1;
            } else {
                break;
            }
        }
    }
}

// ── DoubleBuffer ─────────────────────────────────────────────────────────────

/// State of a double buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DoubleBufferState {
    /// The `A` side is the write buffer.
    WritingToA,
    /// The `B` side is the write buffer.
    WritingToB,
}

/// A double-buffer: one side accumulates incoming elements while the other is
/// available for reading / processing.
///
/// Call [`DoubleBuffer::swap`] to atomically flip the sides.  The previously
/// active write buffer becomes the read buffer, and the (now-cleared) read
/// buffer becomes the new write buffer.
#[derive(Debug)]
pub struct DoubleBuffer<T: Clone> {
    buf_a: Vec<T>,
    buf_b: Vec<T>,
    state: DoubleBufferState,
    swaps: u64,
    last_swap: Instant,
}

impl<T: Clone> DoubleBuffer<T> {
    /// Create a new double buffer.
    pub fn new() -> Self {
        Self {
            buf_a: Vec::new(),
            buf_b: Vec::new(),
            state: DoubleBufferState::WritingToA,
            swaps: 0,
            last_swap: Instant::now(),
        }
    }

    /// Push an element into the current write buffer.
    pub fn push(&mut self, value: T) {
        match self.state {
            DoubleBufferState::WritingToA => self.buf_a.push(value),
            DoubleBufferState::WritingToB => self.buf_b.push(value),
        }
    }

    /// Swap write and read buffers.
    ///
    /// Returns the completed batch from the previous write buffer.  The new
    /// write buffer is reset to empty before returning.
    pub fn swap(&mut self) -> Vec<T> {
        self.swaps += 1;
        self.last_swap = Instant::now();
        match self.state {
            DoubleBufferState::WritingToA => {
                let batch = std::mem::take(&mut self.buf_a);
                self.buf_b.clear();
                self.state = DoubleBufferState::WritingToB;
                batch
            }
            DoubleBufferState::WritingToB => {
                let batch = std::mem::take(&mut self.buf_b);
                self.buf_a.clear();
                self.state = DoubleBufferState::WritingToA;
                batch
            }
        }
    }

    /// Number of elements in the current write buffer.
    pub fn write_len(&self) -> usize {
        match self.state {
            DoubleBufferState::WritingToA => self.buf_a.len(),
            DoubleBufferState::WritingToB => self.buf_b.len(),
        }
    }

    /// Total number of swaps performed.
    #[inline]
    pub fn swap_count(&self) -> u64 {
        self.swaps
    }

    /// Current buffer state (which side is the write buffer).
    #[inline]
    pub fn state(&self) -> DoubleBufferState {
        self.state
    }
}

impl<T: Clone> Default for DoubleBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ── BufferMetrics ─────────────────────────────────────────────────────────────

/// Snapshot of buffer utilisation for monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferMetrics {
    /// Current fill level (elements stored).
    pub current_size: usize,
    /// Configured maximum capacity.
    pub max_capacity: usize,
    /// Fill ratio in [0.0, 1.0].
    pub fill_ratio: f64,
    /// Elements inserted since creation.
    pub total_inserted: u64,
    /// Elements evicted since creation.
    pub total_evicted: u64,
    /// Elements rejected since creation.
    pub total_rejected: u64,
}

impl<T: Clone> StreamingBuffer<T> {
    /// Snapshot current buffer utilisation metrics.
    pub fn metrics(&self) -> BufferMetrics {
        let fill_ratio = if self.max_size > 0 {
            self.storage.len() as f64 / self.max_size as f64
        } else {
            0.0
        };
        BufferMetrics {
            current_size: self.storage.len(),
            max_capacity: self.max_size,
            fill_ratio,
            total_inserted: self.total_inserted,
            total_evicted: self.total_evicted,
            total_rejected: self.total_rejected,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_buffer_push_pop_cycle() {
        let mut rb: RingBuffer<i32> = RingBuffer::new(3).expect("valid capacity");
        assert!(rb.is_empty());
        rb.push(1);
        rb.push(2);
        rb.push(3);
        assert!(rb.is_full());
        // Overwrite oldest
        let evicted = rb.push(4);
        assert_eq!(evicted, Some(1));
        assert_eq!(rb.peek_all(), vec![2, 3, 4]);
    }

    #[test]
    fn ring_buffer_pop_order() {
        let mut rb: RingBuffer<i32> = RingBuffer::new(4).expect("valid capacity");
        rb.push(10);
        rb.push(20);
        rb.push(30);
        assert_eq!(rb.pop(), Some(10));
        assert_eq!(rb.pop(), Some(20));
        rb.push(40);
        assert_eq!(rb.pop(), Some(30));
        assert_eq!(rb.pop(), Some(40));
        assert!(rb.is_empty());
    }

    #[test]
    fn ring_buffer_mean() {
        let mut rb: RingBuffer<f64> = RingBuffer::new(4).expect("valid capacity");
        for v in [1.0_f64, 2.0, 3.0, 4.0] {
            rb.push(v);
        }
        let m = rb.mean().expect("non-empty");
        assert!((m - 2.5).abs() < 1e-12);
    }

    #[test]
    fn ring_buffer_zero_capacity_errors() {
        assert!(RingBuffer::<i32>::new(0).is_err());
    }

    #[test]
    fn streaming_buffer_drop_oldest() {
        let mut sb: StreamingBuffer<i32> =
            StreamingBuffer::new(3, StreamingEvictionPolicy::DropOldest).expect("valid");
        sb.push(1).expect("ok");
        sb.push(2).expect("ok");
        sb.push(3).expect("ok");
        // Buffer full — oldest (1) is evicted
        let inserted = sb.push(4).expect("no error");
        assert!(inserted);
        assert_eq!(sb.len(), 3);
        assert_eq!(sb.total_evicted(), 1);
        let items: Vec<i32> = sb.drain_all().into_iter().map(|i| i.value).collect();
        assert_eq!(items, vec![2, 3, 4]);
    }

    #[test]
    fn streaming_buffer_reject_policy() {
        let mut sb: StreamingBuffer<i32> =
            StreamingBuffer::new(2, StreamingEvictionPolicy::Reject).expect("valid");
        sb.push(1).expect("ok");
        sb.push(2).expect("ok");
        let inserted = sb.push(3).expect("no error");
        assert!(!inserted);
        assert_eq!(sb.total_rejected(), 1);
        assert_eq!(sb.len(), 2);
    }

    #[test]
    fn streaming_buffer_zero_size_errors() {
        assert!(
            StreamingBuffer::<i32>::new(0, StreamingEvictionPolicy::Reject).is_err()
        );
    }

    #[test]
    fn double_buffer_swap() {
        let mut db: DoubleBuffer<i32> = DoubleBuffer::new();
        db.push(1);
        db.push(2);
        let batch = db.swap();
        assert_eq!(batch, vec![1, 2]);
        assert_eq!(db.write_len(), 0);
        assert_eq!(db.swap_count(), 1);
        db.push(3);
        let batch2 = db.swap();
        assert_eq!(batch2, vec![3]);
        assert_eq!(db.swap_count(), 2);
    }

    #[test]
    fn double_buffer_empty_swap() {
        let mut db: DoubleBuffer<i32> = DoubleBuffer::new();
        let batch = db.swap();
        assert!(batch.is_empty());
    }

    #[test]
    fn buffer_metrics_fill_ratio() {
        let mut sb: StreamingBuffer<i32> =
            StreamingBuffer::new(10, StreamingEvictionPolicy::DropOldest).expect("valid");
        for i in 0..5 {
            sb.push(i).expect("ok");
        }
        let m = sb.metrics();
        assert_eq!(m.current_size, 5);
        assert_eq!(m.max_capacity, 10);
        assert!((m.fill_ratio - 0.5).abs() < 1e-12);
    }
}
