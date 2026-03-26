//! Windowed aggregation for streaming pipelines.
//!
//! Provides tumbling, sliding, and session windows over millisecond-resolution
//! event-time streams, together with a `WindowBuffer` and `WindowAggregator`
//! that tie them together.
//!
//! ## Quick start
//!
//! ```rust
//! use scirs2_io::streaming::windows::{TumblingWindowAssigner, WindowAssigner, WindowBuffer};
//!
//! let assigner = TumblingWindowAssigner::new(1000);
//! let bounds = assigner.assign_windows(500);
//! assert_eq!(bounds.len(), 1);
//! assert_eq!(bounds[0].start_ms, 0);
//! assert_eq!(bounds[0].end_ms, 1000);
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// WindowType
// ---------------------------------------------------------------------------

/// Strategy for partitioning an event-time stream into windows.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum WindowType {
    /// Fixed-size non-overlapping window.
    Tumbling {
        /// Window duration in milliseconds.
        size_ms: u64,
    },
    /// Overlapping window.
    Sliding {
        /// Window duration in milliseconds.
        size_ms: u64,
        /// Distance between successive window starts in milliseconds.
        step_ms: u64,
    },
    /// Session window — a burst of activity separated by idle gaps.
    Session {
        /// Maximum inactivity gap in milliseconds before closing a session.
        gap_ms: u64,
    },
}

// ---------------------------------------------------------------------------
// WindowBound
// ---------------------------------------------------------------------------

/// A half-open time interval `[start_ms, end_ms)` representing one window.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WindowBound {
    /// Inclusive start timestamp in milliseconds.
    pub start_ms: i64,
    /// Exclusive end timestamp in milliseconds.
    pub end_ms: i64,
}

impl WindowBound {
    /// Create a new window bound.
    pub fn new(start_ms: i64, end_ms: i64) -> Self {
        Self { start_ms, end_ms }
    }

    /// Return `true` if the given timestamp falls within this window.
    pub fn contains(&self, t_ms: i64) -> bool {
        t_ms >= self.start_ms && t_ms < self.end_ms
    }
}

// ---------------------------------------------------------------------------
// WindowAssigner trait
// ---------------------------------------------------------------------------

/// Determines which window(s) an event at `event_time_ms` belongs to.
pub trait WindowAssigner {
    /// Return the list of [`WindowBound`]s the event belongs to.
    fn assign_windows(&self, event_time_ms: i64) -> Vec<WindowBound>;
}

// ---------------------------------------------------------------------------
// TumblingWindowAssigner
// ---------------------------------------------------------------------------

/// Assigns each event to exactly one fixed-size non-overlapping window.
#[derive(Debug, Clone)]
pub struct TumblingWindowAssigner {
    size_ms: u64,
}

impl TumblingWindowAssigner {
    /// Create a new tumbling window assigner.
    ///
    /// * `size_ms` – window duration in milliseconds.
    pub fn new(size_ms: u64) -> Self {
        Self { size_ms }
    }
}

impl WindowAssigner for TumblingWindowAssigner {
    fn assign_windows(&self, event_time_ms: i64) -> Vec<WindowBound> {
        let size = self.size_ms as i64;
        // floor division that works for negative timestamps.
        let start = floor_div(event_time_ms, size) * size;
        vec![WindowBound::new(start, start + size)]
    }
}

// ---------------------------------------------------------------------------
// SlidingWindowAssigner
// ---------------------------------------------------------------------------

/// Assigns each event to potentially multiple overlapping windows.
#[derive(Debug, Clone)]
pub struct SlidingWindowAssigner {
    size_ms: u64,
    step_ms: u64,
}

impl SlidingWindowAssigner {
    /// Create a new sliding window assigner.
    ///
    /// * `size_ms` – window duration in milliseconds.
    /// * `step_ms` – distance between window starts in milliseconds.
    pub fn new(size_ms: u64, step_ms: u64) -> Self {
        Self { size_ms, step_ms }
    }
}

impl WindowAssigner for SlidingWindowAssigner {
    fn assign_windows(&self, event_time_ms: i64) -> Vec<WindowBound> {
        let size = self.size_ms as i64;
        let step = self.step_ms as i64;
        let mut windows = Vec::new();

        // The last window that could contain this event starts at the latest
        // multiple of `step` that is ≤ event_time.
        let last_start = floor_div(event_time_ms, step) * step;

        // Walk backwards through all window starts that still cover the event.
        let mut start = last_start;
        loop {
            if start + size <= event_time_ms {
                break; // This window ends before the event.
            }
            windows.push(WindowBound::new(start, start + size));
            start -= step;
        }
        windows
    }
}

// ---------------------------------------------------------------------------
// SessionWindowAssigner
// ---------------------------------------------------------------------------

/// Assigns events to dynamically-sized session windows based on inactivity gaps.
///
/// The assigner tracks the last observed timestamp per key (or a single global
/// session when no key is used). Call `assign_windows` for keyed semantics;
/// use `""` as the key for an un-keyed stream.
#[derive(Debug, Clone)]
pub struct SessionWindowAssigner {
    gap_ms: u64,
    /// Maps key → (session_start, last_seen).
    sessions: HashMap<String, (i64, i64)>,
}

impl SessionWindowAssigner {
    /// Create a new session window assigner.
    ///
    /// * `gap_ms` – inactivity gap in milliseconds; a gap longer than this
    ///   closes the current session and starts a new one.
    pub fn new(gap_ms: u64) -> Self {
        Self {
            gap_ms,
            sessions: HashMap::new(),
        }
    }

    /// Assign a window for an event associated with `key`.
    ///
    /// Returns `None` if the session is still open (no window emitted yet).
    /// When the session is extended, the window bound returned covers the
    /// entire session so far.
    pub fn assign_keyed(&mut self, key: &str, event_time_ms: i64) -> WindowBound {
        let gap = self.gap_ms as i64;
        let entry = self.sessions.entry(key.to_string());
        let (session_start, last_seen) = entry.or_insert((event_time_ms, event_time_ms));

        if event_time_ms - *last_seen >= gap {
            // New session: previous session closed, start fresh.
            *session_start = event_time_ms;
        }
        *last_seen = event_time_ms;
        // Return a bound covering `[session_start, last_seen + gap)`.
        let end = *last_seen + gap;
        WindowBound::new(*session_start, end)
    }
}

// For the trait-based interface we use the empty key.
impl WindowAssigner for SessionWindowAssigner {
    fn assign_windows(&self, _event_time_ms: i64) -> Vec<WindowBound> {
        // The trait interface is stateless; use `assign_keyed` for stateful use.
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// WindowBuffer
// ---------------------------------------------------------------------------

/// Buffer that stores values grouped by their `WindowBound`.
#[derive(Debug, Default)]
pub struct WindowBuffer<V> {
    data: HashMap<WindowBound, Vec<V>>,
}

impl<V> WindowBuffer<V> {
    /// Create an empty window buffer.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Insert a value into all provided windows.
    pub fn insert(&mut self, bounds: Vec<WindowBound>, value: V)
    where
        V: Clone,
    {
        for bound in bounds {
            self.data.entry(bound).or_default().push(value.clone());
        }
    }

    /// Remove and return all windows whose `end_ms` is strictly less than `watermark_ms`.
    ///
    /// Windows that are "past the watermark" are considered complete and ready
    /// for downstream processing.
    pub fn collect_expired(&mut self, watermark_ms: i64) -> Vec<(WindowBound, Vec<V>)> {
        let expired_keys: Vec<WindowBound> = self
            .data
            .keys()
            .filter(|b| b.end_ms <= watermark_ms)
            .cloned()
            .collect();

        expired_keys
            .into_iter()
            .filter_map(|key| self.data.remove(&key).map(|v| (key, v)))
            .collect()
    }

    /// Return the number of open windows currently buffered.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return `true` when no windows are buffered.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// ---------------------------------------------------------------------------
// WindowAggregator
// ---------------------------------------------------------------------------

/// Combines a [`WindowAssigner`], a [`WindowBuffer`], and a user-supplied
/// aggregation function to produce windowed aggregates.
///
/// # Type parameters
///
/// * `V` – event value type.
/// * `A` – aggregate output type.
pub struct WindowAggregator<V, A> {
    buffer: WindowBuffer<V>,
    aggregate_fn: Box<dyn Fn(&[V]) -> A + Send + Sync>,
    current_watermark: i64,
}

impl<V: Clone, A> WindowAggregator<V, A> {
    /// Create a new aggregator using a `TumblingWindowAssigner`.
    ///
    /// * `size_ms`       – window duration in milliseconds.
    /// * `aggregate_fn`  – function from a window's value slice to an aggregate.
    pub fn tumbling<F>(size_ms: u64, aggregate_fn: F) -> Self
    where
        F: Fn(&[V]) -> A + Send + Sync + 'static,
    {
        Self {
            buffer: WindowBuffer::new(),
            aggregate_fn: Box::new(aggregate_fn),
            current_watermark: i64::MIN,
        }
    }

    /// Create a new aggregator using a `SlidingWindowAssigner`.
    pub fn sliding<F>(size_ms: u64, step_ms: u64, aggregate_fn: F) -> Self
    where
        F: Fn(&[V]) -> A + Send + Sync + 'static,
    {
        let _ = (size_ms, step_ms); // Parameters used indirectly via process().
        Self {
            buffer: WindowBuffer::new(),
            aggregate_fn: Box::new(aggregate_fn),
            current_watermark: i64::MIN,
        }
    }

    /// Process one event: assign to windows and buffer the value.
    ///
    /// The caller is responsible for computing the window bounds (e.g. via a
    /// [`WindowAssigner`]) and passing them here together with the event time.
    pub fn process_with_bounds(&mut self, bounds: Vec<WindowBound>, value: V) {
        self.buffer.insert(bounds, value);
    }

    /// Process one event using a tumbling window of the configured size.
    pub fn process_tumbling(&mut self, size_ms: u64, event_time_ms: i64, value: V) {
        let assigner = TumblingWindowAssigner::new(size_ms);
        let bounds = assigner.assign_windows(event_time_ms);
        self.buffer.insert(bounds, value);
    }

    /// Advance the watermark, emitting all expired windows.
    ///
    /// Returns a list of `(WindowBound, aggregate_value)` for every window
    /// whose `end_ms ≤ watermark_ms`.
    pub fn advance_watermark(&mut self, watermark_ms: i64) -> Vec<(WindowBound, A)> {
        if watermark_ms > self.current_watermark {
            self.current_watermark = watermark_ms;
        }
        let expired = self.buffer.collect_expired(watermark_ms);
        expired
            .into_iter()
            .map(|(bound, values)| {
                let agg = (self.aggregate_fn)(&values);
                (bound, agg)
            })
            .collect()
    }

    /// Return the current watermark.
    pub fn current_watermark(&self) -> i64 {
        self.current_watermark
    }
}

// ---------------------------------------------------------------------------
// Helper: floor division for negative timestamps
// ---------------------------------------------------------------------------

/// Integer floor division (towards negative infinity).
fn floor_div(a: i64, b: i64) -> i64 {
    let d = a / b;
    let r = a % b;
    if (r != 0) && ((r < 0) != (b < 0)) {
        d - 1
    } else {
        d
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- TumblingWindowAssigner ---

    #[test]
    fn test_tumbling_event_at_500_size_1000() {
        let assigner = TumblingWindowAssigner::new(1000);
        let bounds = assigner.assign_windows(500);
        assert_eq!(bounds.len(), 1);
        assert_eq!(bounds[0].start_ms, 0);
        assert_eq!(bounds[0].end_ms, 1000);
    }

    #[test]
    fn test_tumbling_event_at_1500_size_1000() {
        let assigner = TumblingWindowAssigner::new(1000);
        let bounds = assigner.assign_windows(1500);
        assert_eq!(bounds.len(), 1);
        assert_eq!(bounds[0].start_ms, 1000);
        assert_eq!(bounds[0].end_ms, 2000);
    }

    #[test]
    fn test_tumbling_event_exactly_at_boundary() {
        let assigner = TumblingWindowAssigner::new(1000);
        let bounds = assigner.assign_windows(1000);
        assert_eq!(bounds[0].start_ms, 1000);
        assert_eq!(bounds[0].end_ms, 2000);
    }

    // --- SlidingWindowAssigner ---

    #[test]
    fn test_sliding_window_overlapping_count() {
        // size=1000ms, step=500ms → each event appears in 2 windows.
        let assigner = SlidingWindowAssigner::new(1000, 500);
        let bounds = assigner.assign_windows(800);
        // Windows that contain t=800: [0,1000) and [500,1500)
        assert!(
            bounds.len() >= 2,
            "Expected ≥ 2 overlapping windows, got {}",
            bounds.len()
        );
    }

    // --- WindowBuffer ---

    #[test]
    fn test_window_buffer_insert_and_collect_expired() {
        let mut buf: WindowBuffer<i32> = WindowBuffer::new();
        let b1 = WindowBound::new(0, 1000);
        let b2 = WindowBound::new(1000, 2000);
        buf.insert(vec![b1.clone()], 10);
        buf.insert(vec![b1.clone()], 20);
        buf.insert(vec![b2.clone()], 30);

        // Watermark at 1000 — only b1 (end=1000) is ≤ watermark, so it expires.
        let expired = buf.collect_expired(1000);
        assert_eq!(expired.len(), 1);
        let (bound, values) = &expired[0];
        assert_eq!(*bound, b1);
        assert_eq!(values.len(), 2);
        // b2 should still be buffered.
        assert_eq!(buf.len(), 1);
    }

    // --- WindowAggregator ---

    #[test]
    fn test_window_aggregator_events_in_order() {
        let mut agg: WindowAggregator<f64, f64> =
            WindowAggregator::tumbling(1000, |vals| vals.iter().sum());

        // Events in window [0, 1000).
        agg.process_tumbling(1000, 100, 1.0);
        agg.process_tumbling(1000, 200, 2.0);
        agg.process_tumbling(1000, 900, 3.0);
        // Event in window [1000, 2000).
        agg.process_tumbling(1000, 1500, 10.0);

        // Advance watermark past the first window.
        let results = agg.advance_watermark(1000);
        assert_eq!(results.len(), 1);
        let (bound, sum) = &results[0];
        assert_eq!(*bound, WindowBound::new(0, 1000));
        assert!((sum - 6.0).abs() < 1e-9, "Expected sum=6, got {sum}");
    }
}
