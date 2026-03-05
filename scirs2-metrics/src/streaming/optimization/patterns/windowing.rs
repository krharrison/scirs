//! Sliding and tumbling window management for streaming data
//!
//! This module provides window implementations for partitioning a continuous stream
//! into finite chunks for aggregation and metric computation:
//! - `SlidingWindow`: overlapping windows that advance by a step size
//! - `TumblingWindow`: non-overlapping windows of a fixed duration or count
//! - `SessionWindow`: event-driven windows that close after an inactivity gap

use crate::error::{MetricsError, Result};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, SystemTime};

// ── Sliding Window ───────────────────────────────────────────────────────────

/// A count-based sliding window that advances element by element.
///
/// Elements older than `capacity` are evicted automatically.  Call
/// [`SlidingWindow::push`] to ingest new data and [`SlidingWindow::view`] to
/// inspect the current contents.
#[derive(Debug, Clone)]
pub struct SlidingWindow<T> {
    capacity: usize,
    step: usize,
    buffer: VecDeque<T>,
    steps_until_flush: usize,
}

impl<T: Clone> SlidingWindow<T> {
    /// Create a new sliding window.
    ///
    /// # Arguments
    /// * `capacity` – number of elements kept in the window
    /// * `step`     – how many elements to advance before the oldest is dropped
    ///
    /// Returns `Err` when `step` is 0 or `step > capacity`.
    pub fn new(capacity: usize, step: usize) -> Result<Self> {
        if step == 0 {
            return Err(MetricsError::InvalidInput(
                "SlidingWindow step must be >= 1".to_string(),
            ));
        }
        if step > capacity {
            return Err(MetricsError::InvalidInput(format!(
                "SlidingWindow step ({step}) must be <= capacity ({capacity})"
            )));
        }
        Ok(Self {
            capacity,
            step,
            buffer: VecDeque::with_capacity(capacity),
            steps_until_flush: step,
        })
    }

    /// Push a new element into the window.
    ///
    /// Returns `true` when a full window boundary has been crossed (i.e. `step`
    /// new elements have arrived since the last boundary).
    pub fn push(&mut self, value: T) -> bool {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(value);

        self.steps_until_flush = self.steps_until_flush.saturating_sub(1);
        if self.steps_until_flush == 0 {
            self.steps_until_flush = self.step;
            true
        } else {
            false
        }
    }

    /// View the current window contents (oldest first).
    #[inline]
    pub fn view(&self) -> impl Iterator<Item = &T> {
        self.buffer.iter()
    }

    /// Number of elements currently in the window.
    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// True when the window contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Configured capacity (maximum number of retained elements).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Step size configured at construction.
    #[inline]
    pub fn step(&self) -> usize {
        self.step
    }

    /// Drain and return all elements, leaving the window empty.
    pub fn drain(&mut self) -> Vec<T> {
        self.buffer.drain(..).collect()
    }
}

impl<F: Float + std::fmt::Debug + Copy> SlidingWindow<F> {
    /// Compute the arithmetic mean of the current window contents.
    ///
    /// Returns `None` when the window is empty.
    pub fn mean(&self) -> Option<F> {
        if self.buffer.is_empty() {
            return None;
        }
        let sum = self.buffer.iter().copied().fold(F::zero(), |acc, x| acc + x);
        let n = F::from(self.buffer.len()).expect("usize fits in F");
        Some(sum / n)
    }

    /// Compute the population variance of the window.
    pub fn variance(&self) -> Option<F> {
        let mean = self.mean()?;
        let var = self
            .buffer
            .iter()
            .copied()
            .map(|x| {
                let d = x - mean;
                d * d
            })
            .fold(F::zero(), |acc, v| acc + v)
            / F::from(self.buffer.len()).expect("usize fits in F");
        Some(var)
    }
}

// ── Tumbling Window ──────────────────────────────────────────────────────────

/// A non-overlapping (tumbling) window that collects elements until a trigger
/// fires, then emits a complete `TumblingWindowBatch` and resets.
///
/// The trigger may be count-based, time-based, or a combination of both.
#[derive(Debug, Clone)]
pub struct TumblingWindow<T> {
    trigger: TumblingTrigger,
    buffer: Vec<T>,
    window_start: SystemTime,
    window_index: u64,
}

/// How a tumbling window decides when to close.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TumblingTrigger {
    /// Close after exactly `count` elements.
    Count(usize),
    /// Close after the given `duration` has elapsed since the window opened.
    Time(Duration),
    /// Close on whichever of count or time fires first.
    CountOrTime { count: usize, duration: Duration },
}

/// A completed tumbling window batch ready for downstream processing.
#[derive(Debug, Clone)]
pub struct TumblingWindowBatch<T> {
    /// Sequential index of this batch (0-based).
    pub index: u64,
    /// Elements collected in this batch.
    pub items: Vec<T>,
    /// Timestamp when the window opened.
    pub opened_at: SystemTime,
    /// Timestamp when the window closed.
    pub closed_at: SystemTime,
}

impl<T: Clone> TumblingWindow<T> {
    /// Create a new tumbling window with the given trigger.
    pub fn new(trigger: TumblingTrigger) -> Self {
        Self {
            trigger,
            buffer: Vec::new(),
            window_start: SystemTime::now(),
            window_index: 0,
        }
    }

    /// Push `value` into the current window.
    ///
    /// Returns `Some(batch)` when the trigger fires and the window is emitted,
    /// or `None` when the window is still accumulating.
    pub fn push(&mut self, value: T) -> Option<TumblingWindowBatch<T>> {
        self.buffer.push(value);
        let now = SystemTime::now();
        let elapsed = now.duration_since(self.window_start).unwrap_or_default();

        let should_close = match &self.trigger {
            TumblingTrigger::Count(n) => self.buffer.len() >= *n,
            TumblingTrigger::Time(d) => elapsed >= *d,
            TumblingTrigger::CountOrTime { count, duration } => {
                self.buffer.len() >= *count || elapsed >= *duration
            }
        };

        if should_close {
            Some(self.close_window(now))
        } else {
            None
        }
    }

    /// Force-close the current window regardless of trigger state.
    pub fn flush(&mut self) -> Option<TumblingWindowBatch<T>> {
        if self.buffer.is_empty() {
            return None;
        }
        Some(self.close_window(SystemTime::now()))
    }

    /// Number of elements buffered in the current (open) window.
    #[inline]
    pub fn buffered_len(&self) -> usize {
        self.buffer.len()
    }

    fn close_window(&mut self, closed_at: SystemTime) -> TumblingWindowBatch<T> {
        let items = std::mem::take(&mut self.buffer);
        let batch = TumblingWindowBatch {
            index: self.window_index,
            items,
            opened_at: self.window_start,
            closed_at,
        };
        self.window_index += 1;
        self.window_start = closed_at;
        batch
    }
}

// ── Session Window ───────────────────────────────────────────────────────────

/// A session window that groups elements separated by gaps shorter than
/// `inactivity_gap`.  A new session begins whenever the gap between two
/// consecutive events exceeds the threshold.
#[derive(Debug, Clone)]
pub struct SessionWindow<T> {
    inactivity_gap: Duration,
    buffer: Vec<T>,
    last_event_time: Option<SystemTime>,
    session_index: u64,
}

/// A completed session window batch.
#[derive(Debug, Clone)]
pub struct SessionWindowBatch<T> {
    /// Sequential session index (0-based).
    pub index: u64,
    /// Elements collected in this session.
    pub items: Vec<T>,
    /// Duration of the session (time from first to last event).
    pub session_duration: Duration,
}

impl<T: Clone> SessionWindow<T> {
    /// Create a new session window.
    ///
    /// # Arguments
    /// * `inactivity_gap` – gap between events that triggers a new session.
    ///
    /// Returns `Err` if `inactivity_gap` is zero.
    pub fn new(inactivity_gap: Duration) -> Result<Self> {
        if inactivity_gap.is_zero() {
            return Err(MetricsError::InvalidInput(
                "SessionWindow inactivity_gap must be > 0".to_string(),
            ));
        }
        Ok(Self {
            inactivity_gap,
            buffer: Vec::new(),
            last_event_time: None,
            session_index: 0,
        })
    }

    /// Push an event at `event_time`.
    ///
    /// If the gap since the previous event exceeds `inactivity_gap`, the
    /// previous session is closed and returned before the new event is buffered.
    pub fn push(&mut self, value: T, event_time: SystemTime) -> Option<SessionWindowBatch<T>> {
        let mut completed = None;

        if let Some(last) = self.last_event_time {
            let gap = event_time.duration_since(last).unwrap_or_default();
            if gap >= self.inactivity_gap && !self.buffer.is_empty() {
                let items = std::mem::take(&mut self.buffer);
                let session_duration = last
                    .duration_since(
                        self.last_event_time.unwrap_or(last), // same value — gives 0
                    )
                    .unwrap_or_default();
                completed = Some(SessionWindowBatch {
                    index: self.session_index,
                    items,
                    session_duration,
                });
                self.session_index += 1;
            }
        }

        self.buffer.push(value);
        self.last_event_time = Some(event_time);
        completed
    }

    /// Force-flush the current session.
    pub fn flush(&mut self) -> Option<SessionWindowBatch<T>> {
        if self.buffer.is_empty() {
            return None;
        }
        let items = std::mem::take(&mut self.buffer);
        let batch = SessionWindowBatch {
            index: self.session_index,
            items,
            session_duration: Duration::ZERO,
        };
        self.session_index += 1;
        self.last_event_time = None;
        Some(batch)
    }

    /// Number of elements in the current open session.
    #[inline]
    pub fn buffered_len(&self) -> usize {
        self.buffer.len()
    }
}

// ── Aggregate helpers ────────────────────────────────────────────────────────

/// Summary statistics computed over a completed window batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowAggregate<F: Float + std::fmt::Debug> {
    /// Number of elements in the batch.
    pub count: usize,
    /// Sum of all values.
    pub sum: F,
    /// Arithmetic mean.
    pub mean: F,
    /// Population variance.
    pub variance: F,
    /// Minimum value.
    pub min: F,
    /// Maximum value.
    pub max: F,
}

impl<F: Float + std::fmt::Debug + Copy> WindowAggregate<F> {
    /// Compute aggregate statistics from a slice of values.
    ///
    /// Returns `Err` when `values` is empty.
    pub fn from_slice(values: &[F]) -> Result<Self> {
        if values.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Cannot aggregate an empty window".to_string(),
            ));
        }
        let count = values.len();
        let n = F::from(count).expect("usize fits in F");
        let sum = values.iter().copied().fold(F::zero(), |a, x| a + x);
        let mean = sum / n;
        let variance = values
            .iter()
            .copied()
            .map(|x| {
                let d = x - mean;
                d * d
            })
            .fold(F::zero(), |a, x| a + x)
            / n;
        let min = values
            .iter()
            .copied()
            .fold(F::infinity(), |a, x| if x < a { x } else { a });
        let max = values
            .iter()
            .copied()
            .fold(F::neg_infinity(), |a, x| if x > a { x } else { a });
        Ok(Self {
            count,
            sum,
            mean,
            variance,
            min,
            max,
        })
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_window_basic() {
        let mut w: SlidingWindow<f64> = SlidingWindow::new(4, 2).expect("valid params");
        assert!(w.is_empty());
        w.push(1.0);
        w.push(2.0);
        w.push(3.0);
        w.push(4.0);
        assert_eq!(w.len(), 4);
        // Adding one more evicts the oldest
        w.push(5.0);
        assert_eq!(w.len(), 4);
        assert_eq!(w.view().copied().collect::<Vec<_>>(), vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn sliding_window_mean_variance() {
        let mut w: SlidingWindow<f64> = SlidingWindow::new(4, 1).expect("valid params");
        for v in [2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            w.push(v);
        }
        // last 4 elements: 5, 7, 9 — wait, capacity=4, so window=[5,5,7,9]
        let mean = w.mean().expect("non-empty");
        assert!((mean - 6.5).abs() < 1e-10, "mean={mean}");
    }

    #[test]
    fn sliding_window_invalid_step() {
        assert!(SlidingWindow::<f64>::new(4, 0).is_err());
        assert!(SlidingWindow::<f64>::new(4, 5).is_err());
    }

    #[test]
    fn tumbling_window_count_trigger() {
        let mut w: TumblingWindow<i32> = TumblingWindow::new(TumblingTrigger::Count(3));
        assert!(w.push(1).is_none());
        assert!(w.push(2).is_none());
        let batch = w.push(3).expect("batch emitted");
        assert_eq!(batch.items, vec![1, 2, 3]);
        assert_eq!(batch.index, 0);
        // Next window
        assert!(w.push(4).is_none());
        let batch2 = w.push(5);
        assert!(batch2.is_none()); // only 2 so far
        let batch3 = w.push(6).expect("batch emitted");
        assert_eq!(batch3.index, 1);
    }

    #[test]
    fn tumbling_window_flush() {
        let mut w: TumblingWindow<i32> = TumblingWindow::new(TumblingTrigger::Count(10));
        w.push(1);
        w.push(2);
        let batch = w.flush().expect("non-empty flush");
        assert_eq!(batch.items, vec![1, 2]);
        assert!(w.flush().is_none()); // already empty
    }

    #[test]
    fn session_window_gap_triggers_close() {
        let mut w: SessionWindow<i32> = SessionWindow::new(Duration::from_secs(5)).expect("valid");
        let t0 = SystemTime::now();
        assert!(w.push(1, t0).is_none());
        assert!(w.push(2, t0 + Duration::from_secs(1)).is_none());
        // Gap of 10s exceeds inactivity threshold — previous session closes
        let completed = w.push(3, t0 + Duration::from_secs(11));
        assert!(completed.is_some());
        let batch = completed.expect("session closed");
        assert_eq!(batch.items, vec![1, 2]);
        assert_eq!(batch.index, 0);
    }

    #[test]
    fn window_aggregate_from_slice() {
        let values = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let agg = WindowAggregate::from_slice(&values).expect("non-empty");
        assert_eq!(agg.count, 5);
        assert!((agg.mean - 3.0).abs() < 1e-12);
        assert!((agg.min - 1.0).abs() < 1e-12);
        assert!((agg.max - 5.0).abs() < 1e-12);
    }

    #[test]
    fn window_aggregate_empty_errors() {
        let result = WindowAggregate::<f64>::from_slice(&[]);
        assert!(result.is_err());
    }
}
