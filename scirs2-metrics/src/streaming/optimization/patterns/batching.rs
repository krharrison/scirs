//! Adaptive batching strategies for streaming data processing
//!
//! This module provides strategies for grouping incoming stream elements into
//! batches before processing, trading latency for throughput:
//!
//! - [`FixedBatcher`]   – fixed-size count-based batching
//! - [`AdaptiveBatcher`] – dynamically adjusts batch size based on observed
//!   throughput and downstream processing latency
//! - [`PriorityBatcher`] – assembles batches by element priority so that
//!   high-priority items are never delayed by low-priority fill

use crate::error::{MetricsError, Result};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use std::time::{Duration, Instant};

// ── BatchOutcome ─────────────────────────────────────────────────────────────

/// The result returned by each batcher's `push` method.
#[derive(Debug, Clone)]
pub struct BatchOutcome<T> {
    /// The assembled batch (may be partial on timeout flush).
    pub items: Vec<T>,
    /// Whether the batch was triggered by a timeout rather than reaching the
    /// target batch size.
    pub is_timeout_flush: bool,
    /// Size target that was active when this batch was emitted.
    pub target_size: usize,
}

// ── BatcherStats ─────────────────────────────────────────────────────────────

/// Aggregate statistics collected across all emitted batches.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BatcherStats {
    /// Total number of batches emitted.
    pub batches_emitted: u64,
    /// Total elements processed.
    pub total_elements: u64,
    /// Number of timeout-triggered flushes.
    pub timeout_flushes: u64,
    /// Minimum batch size seen.
    pub min_batch_size: usize,
    /// Maximum batch size seen.
    pub max_batch_size: usize,
    /// Running mean batch size (Welford online update).
    pub mean_batch_size: f64,
}

impl BatcherStats {
    /// Update statistics with a newly emitted batch of `size` elements.
    pub fn record_batch(&mut self, size: usize, is_timeout: bool) {
        self.batches_emitted += 1;
        self.total_elements += size as u64;
        if is_timeout {
            self.timeout_flushes += 1;
        }
        if self.batches_emitted == 1 {
            self.min_batch_size = size;
            self.max_batch_size = size;
        } else {
            self.min_batch_size = self.min_batch_size.min(size);
            self.max_batch_size = self.max_batch_size.max(size);
        }
        // Welford incremental mean
        let delta = size as f64 - self.mean_batch_size;
        self.mean_batch_size += delta / self.batches_emitted as f64;
    }
}

// ── FixedBatcher ─────────────────────────────────────────────────────────────

/// A simple batcher that accumulates elements until a fixed count is reached.
///
/// An optional `max_wait` duration causes the current buffer to be flushed even
/// if it has fewer than `batch_size` elements once the wall-clock threshold is
/// exceeded.
#[derive(Debug)]
pub struct FixedBatcher<T> {
    batch_size: usize,
    max_wait: Option<Duration>,
    buffer: Vec<T>,
    window_start: Instant,
    stats: BatcherStats,
}

impl<T> FixedBatcher<T> {
    /// Create a new fixed batcher.
    ///
    /// # Arguments
    /// * `batch_size` – target number of elements per batch (must be >= 1)
    /// * `max_wait`   – optional timeout after which a partial batch is emitted
    pub fn new(batch_size: usize, max_wait: Option<Duration>) -> Result<Self> {
        if batch_size == 0 {
            return Err(MetricsError::InvalidInput(
                "FixedBatcher batch_size must be >= 1".to_string(),
            ));
        }
        Ok(Self {
            batch_size,
            max_wait,
            buffer: Vec::with_capacity(batch_size),
            window_start: Instant::now(),
            stats: BatcherStats::default(),
        })
    }

    /// Push an element.  Returns a completed batch when the target is reached or
    /// the timeout expires; otherwise returns `None`.
    pub fn push(&mut self, value: T) -> Option<BatchOutcome<T>> {
        self.buffer.push(value);

        let timeout_expired = self
            .max_wait
            .map_or(false, |d| self.window_start.elapsed() >= d);

        if self.buffer.len() >= self.batch_size {
            Some(self.flush_internal(false))
        } else if timeout_expired {
            Some(self.flush_internal(true))
        } else {
            None
        }
    }

    /// Force-flush whatever is in the buffer regardless of batch size / timeout.
    pub fn flush(&mut self) -> Option<BatchOutcome<T>> {
        if self.buffer.is_empty() {
            None
        } else {
            Some(self.flush_internal(true))
        }
    }

    /// Number of elements currently buffered (not yet emitted).
    #[inline]
    pub fn buffered_len(&self) -> usize {
        self.buffer.len()
    }

    /// Cumulative statistics since construction.
    #[inline]
    pub fn stats(&self) -> &BatcherStats {
        &self.stats
    }

    fn flush_internal(&mut self, is_timeout: bool) -> BatchOutcome<T> {
        let items = std::mem::take(&mut self.buffer);
        let size = items.len();
        self.stats.record_batch(size, is_timeout);
        self.buffer = Vec::with_capacity(self.batch_size);
        self.window_start = Instant::now();
        BatchOutcome {
            items,
            is_timeout_flush: is_timeout,
            target_size: self.batch_size,
        }
    }
}

// ── AdaptiveBatcher ──────────────────────────────────────────────────────────

/// Adaptation policy used by [`AdaptiveBatcher`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationPolicy {
    /// Adjust batch size to maintain a target processing latency.
    LatencyTarget {
        /// Desired maximum latency per batch (ms).
        target_ms: f64,
    },
    /// Adjust batch size to maximise throughput while staying within a latency
    /// budget.
    ThroughputMaximisation {
        /// Hard latency ceiling (ms).
        max_latency_ms: f64,
    },
    /// Exponential moving average of observed batch sizes with a fixed
    /// smoothing factor.
    ExponentialSmoothing {
        /// Smoothing factor α ∈ (0, 1].
        alpha: f64,
    },
}

/// An adaptive batcher that adjusts its target batch size over time.
///
/// The adaptation runs every `adaptation_interval` batches.
#[derive(Debug)]
pub struct AdaptiveBatcher<T, F: Float + std::fmt::Debug> {
    min_batch_size: usize,
    max_batch_size: usize,
    current_target: usize,
    policy: AdaptationPolicy,
    adaptation_interval: u64,
    buffer: VecDeque<T>,
    stats: BatcherStats,
    /// Most recently measured processing latencies (one per batch, ms).
    latency_history: VecDeque<F>,
    latency_history_cap: usize,
    batches_since_adapt: u64,
}

impl<T, F: Float + std::fmt::Debug> AdaptiveBatcher<T, F> {
    /// Create a new adaptive batcher.
    ///
    /// # Arguments
    /// * `min_batch_size`       – smallest allowed batch size
    /// * `max_batch_size`       – largest allowed batch size
    /// * `initial_batch_size`   – starting batch size (clamped to [min, max])
    /// * `policy`               – adaptation policy
    /// * `adaptation_interval`  – number of batches between adaptations
    pub fn new(
        min_batch_size: usize,
        max_batch_size: usize,
        initial_batch_size: usize,
        policy: AdaptationPolicy,
        adaptation_interval: u64,
    ) -> Result<Self> {
        if min_batch_size == 0 {
            return Err(MetricsError::InvalidInput(
                "AdaptiveBatcher min_batch_size must be >= 1".to_string(),
            ));
        }
        if max_batch_size < min_batch_size {
            return Err(MetricsError::InvalidInput(format!(
                "max_batch_size ({max_batch_size}) < min_batch_size ({min_batch_size})"
            )));
        }
        if adaptation_interval == 0 {
            return Err(MetricsError::InvalidInput(
                "adaptation_interval must be >= 1".to_string(),
            ));
        }
        let current_target = initial_batch_size.clamp(min_batch_size, max_batch_size);
        Ok(Self {
            min_batch_size,
            max_batch_size,
            current_target,
            policy,
            adaptation_interval,
            buffer: VecDeque::new(),
            stats: BatcherStats::default(),
            latency_history: VecDeque::new(),
            latency_history_cap: 64,
            batches_since_adapt: 0,
        })
    }

    /// Ingest an element.  Returns a batch when the current target is reached.
    pub fn push(&mut self, value: T) -> Option<BatchOutcome<T>> {
        self.buffer.push_back(value);
        if self.buffer.len() >= self.current_target {
            Some(self.emit_batch(false))
        } else {
            None
        }
    }

    /// Record the processing latency for the most recently emitted batch and
    /// trigger adaptation if the interval is reached.
    pub fn record_latency(&mut self, latency_ms: F) {
        self.latency_history.push_back(latency_ms);
        while self.latency_history.len() > self.latency_history_cap {
            self.latency_history.pop_front();
        }

        self.batches_since_adapt += 1;
        if self.batches_since_adapt >= self.adaptation_interval {
            self.adapt();
            self.batches_since_adapt = 0;
        }
    }

    /// Force-flush the buffer.
    pub fn flush(&mut self) -> Option<BatchOutcome<T>> {
        if self.buffer.is_empty() {
            None
        } else {
            Some(self.emit_batch(true))
        }
    }

    /// Current target batch size.
    #[inline]
    pub fn current_target(&self) -> usize {
        self.current_target
    }

    /// Cumulative statistics since construction.
    #[inline]
    pub fn stats(&self) -> &BatcherStats {
        &self.stats
    }

    fn emit_batch(&mut self, is_timeout: bool) -> BatchOutcome<T> {
        let items: Vec<T> = self.buffer.drain(..self.current_target.min(self.buffer.len())).collect();
        let size = items.len();
        self.stats.record_batch(size, is_timeout);
        BatchOutcome {
            items,
            is_timeout_flush: is_timeout,
            target_size: self.current_target,
        }
    }

    fn adapt(&mut self) {
        if self.latency_history.is_empty() {
            return;
        }
        let n = F::from(self.latency_history.len()).expect("usize fits in F");
        let mean_lat = self.latency_history.iter().copied().fold(F::zero(), |a, x| a + x) / n;
        let mean_lat_f64 = mean_lat.to_f64().unwrap_or(f64::MAX);

        let new_target = match &self.policy {
            AdaptationPolicy::LatencyTarget { target_ms } => {
                let ratio = target_ms / mean_lat_f64.max(f64::EPSILON);
                let adjusted = (self.current_target as f64 * ratio).round() as usize;
                adjusted.clamp(self.min_batch_size, self.max_batch_size)
            }
            AdaptationPolicy::ThroughputMaximisation { max_latency_ms } => {
                if mean_lat_f64 < *max_latency_ms * 0.8 {
                    // Well below budget — grow
                    ((self.current_target as f64 * 1.25) as usize)
                        .clamp(self.min_batch_size, self.max_batch_size)
                } else if mean_lat_f64 > *max_latency_ms {
                    // Over budget — shrink
                    ((self.current_target as f64 * 0.75) as usize)
                        .clamp(self.min_batch_size, self.max_batch_size)
                } else {
                    self.current_target
                }
            }
            AdaptationPolicy::ExponentialSmoothing { alpha } => {
                let smoothed =
                    *alpha * self.stats.mean_batch_size + (1.0 - alpha) * self.current_target as f64;
                (smoothed.round() as usize).clamp(self.min_batch_size, self.max_batch_size)
            }
        };

        self.current_target = new_target;
    }
}

// ── PriorityBatcher ──────────────────────────────────────────────────────────

/// An element wrapper that carries a priority level for use in the priority
/// queue inside [`PriorityBatcher`].
#[derive(Debug, Clone)]
struct PrioritizedItem<T> {
    priority: u32,
    sequence: u64, // tie-breaker: lower sequence means older
    value: T,
}

impl<T> PartialEq for PrioritizedItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.sequence == other.sequence
    }
}
impl<T> Eq for PrioritizedItem<T> {}

impl<T> PartialOrd for PrioritizedItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for PrioritizedItem<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first; on tie prefer the older (lower sequence) item
        other
            .priority
            .cmp(&self.priority)
            .reverse()
            .then_with(|| self.sequence.cmp(&other.sequence))
    }
}

/// A batcher that fills each batch from highest to lowest priority.
///
/// When the target batch size is reached the top-priority elements are emitted.
/// All remaining elements stay in the queue for subsequent batches.
#[derive(Debug)]
pub struct PriorityBatcher<T> {
    batch_size: usize,
    queue: BinaryHeap<PrioritizedItem<T>>,
    sequence_counter: u64,
    stats: BatcherStats,
}

impl<T: Clone + std::fmt::Debug> PriorityBatcher<T> {
    /// Create a new priority batcher.
    ///
    /// # Arguments
    /// * `batch_size` – number of elements per emitted batch (must be >= 1)
    pub fn new(batch_size: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(MetricsError::InvalidInput(
                "PriorityBatcher batch_size must be >= 1".to_string(),
            ));
        }
        Ok(Self {
            batch_size,
            queue: BinaryHeap::new(),
            sequence_counter: 0,
            stats: BatcherStats::default(),
        })
    }

    /// Push an element with a numeric priority (higher = more urgent).
    ///
    /// Returns a batch when the queue reaches the target size.
    pub fn push(&mut self, value: T, priority: u32) -> Option<BatchOutcome<T>> {
        self.queue.push(PrioritizedItem {
            priority,
            sequence: self.sequence_counter,
            value,
        });
        self.sequence_counter += 1;

        if self.queue.len() >= self.batch_size {
            Some(self.drain_batch())
        } else {
            None
        }
    }

    /// Flush the highest-priority elements currently in the queue.
    pub fn flush(&mut self) -> Option<BatchOutcome<T>> {
        if self.queue.is_empty() {
            None
        } else {
            Some(self.drain_batch())
        }
    }

    /// Number of elements waiting in the priority queue.
    #[inline]
    pub fn queued_len(&self) -> usize {
        self.queue.len()
    }

    /// Cumulative statistics since construction.
    #[inline]
    pub fn stats(&self) -> &BatcherStats {
        &self.stats
    }

    fn drain_batch(&mut self) -> BatchOutcome<T> {
        let take = self.batch_size.min(self.queue.len());
        let items: Vec<T> = (0..take)
            .filter_map(|_| self.queue.pop().map(|p| p.value))
            .collect();
        let size = items.len();
        self.stats.record_batch(size, false);
        BatchOutcome {
            items,
            is_timeout_flush: false,
            target_size: self.batch_size,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_batcher_emits_at_target() {
        let mut b: FixedBatcher<i32> = FixedBatcher::new(3, None).expect("valid");
        assert!(b.push(1).is_none());
        assert!(b.push(2).is_none());
        let out = b.push(3).expect("batch emitted");
        assert_eq!(out.items, vec![1, 2, 3]);
        assert!(!out.is_timeout_flush);
        assert_eq!(b.stats().batches_emitted, 1);
    }

    #[test]
    fn fixed_batcher_flush() {
        let mut b: FixedBatcher<i32> = FixedBatcher::new(10, None).expect("valid");
        b.push(1);
        b.push(2);
        let out = b.flush().expect("partial flush");
        assert_eq!(out.items, vec![1, 2]);
        assert!(out.is_timeout_flush);
        assert!(b.flush().is_none());
    }

    #[test]
    fn fixed_batcher_zero_size_errors() {
        assert!(FixedBatcher::<i32>::new(0, None).is_err());
    }

    #[test]
    fn adaptive_batcher_basic() {
        let mut b: AdaptiveBatcher<i32, f64> = AdaptiveBatcher::new(
            1,
            100,
            4,
            AdaptationPolicy::LatencyTarget { target_ms: 10.0 },
            5,
        )
        .expect("valid");

        for i in 0..4 {
            let out = b.push(i);
            if i < 3 {
                assert!(out.is_none());
            } else {
                assert!(out.is_some());
            }
        }
        assert_eq!(b.stats().batches_emitted, 1);
    }

    #[test]
    fn adaptive_batcher_adaptation_does_not_panic() {
        let mut b: AdaptiveBatcher<i32, f64> = AdaptiveBatcher::new(
            1,
            50,
            5,
            AdaptationPolicy::ThroughputMaximisation { max_latency_ms: 20.0 },
            2,
        )
        .expect("valid");
        // Feed 20 elements, recording latency after each batch
        for i in 0..20_i32 {
            if let Some(_out) = b.push(i) {
                b.record_latency(15.0_f64);
            }
        }
        // current_target must remain in valid range
        assert!(b.current_target() >= 1);
        assert!(b.current_target() <= 50);
    }

    #[test]
    fn priority_batcher_ordering() {
        let mut b: PriorityBatcher<&str> = PriorityBatcher::new(3).expect("valid");
        assert!(b.push("low", 1).is_none());
        assert!(b.push("high", 10).is_none());
        let out = b.push("medium", 5).expect("batch emitted");
        // highest priority first
        assert_eq!(out.items[0], "high");
        assert_eq!(out.items[1], "medium");
        assert_eq!(out.items[2], "low");
    }

    #[test]
    fn priority_batcher_invalid_size() {
        assert!(PriorityBatcher::<i32>::new(0).is_err());
    }

    #[test]
    fn batcher_stats_record() {
        let mut s = BatcherStats::default();
        s.record_batch(10, false);
        s.record_batch(5, true);
        assert_eq!(s.batches_emitted, 2);
        assert_eq!(s.total_elements, 15);
        assert_eq!(s.timeout_flushes, 1);
        assert_eq!(s.min_batch_size, 5);
        assert_eq!(s.max_batch_size, 10);
        assert!((s.mean_batch_size - 7.5).abs() < 1e-10);
    }
}
