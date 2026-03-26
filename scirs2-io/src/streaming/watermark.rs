//! Watermark generation and late-data handling for event-time streams.
//!
//! Watermarks represent the system's estimate of how far event-time has
//! progressed.  They allow windowed operators to know when a window is
//! "complete" and can safely be emitted, even in the face of out-of-order
//! events.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_io::streaming::watermark::{WatermarkGenerator, WatermarkStrategy};
//!
//! let mut gen = WatermarkGenerator::new(
//!     WatermarkStrategy::BoundedOutOfOrder { max_lateness_ms: 500 }
//! );
//! gen.observe_event(1000);
//! gen.observe_event(1200);
//! // Watermark lags by 500 ms behind the max seen event.
//! assert_eq!(gen.current_watermark(), 700);
//! ```

use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// WatermarkStrategy
// ---------------------------------------------------------------------------

/// Configures how the watermark is derived from observed event timestamps.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum WatermarkStrategy {
    /// Watermark equals the maximum observed event time.
    ///
    /// Suitable for perfectly ordered streams.
    MonotonousTimestamps,

    /// Watermark lags behind the maximum event time by `max_lateness_ms`.
    ///
    /// Tolerates events arriving up to `max_lateness_ms` milliseconds late.
    BoundedOutOfOrder {
        /// Maximum allowed lateness in milliseconds.
        max_lateness_ms: u64,
    },

    /// Emit watermarks only at periodic intervals (not implemented as a
    /// background timer here — call `tick()` to simulate a periodic heartbeat).
    Periodic {
        /// Minimum interval between watermark emissions in milliseconds.
        interval_ms: u64,
        /// Underlying strategy used to compute the watermark value.
        strategy: Box<WatermarkStrategy>,
    },
}

// ---------------------------------------------------------------------------
// WatermarkGenerator
// ---------------------------------------------------------------------------

/// Stateful watermark generator driven by observed event timestamps.
#[derive(Debug)]
pub struct WatermarkGenerator {
    strategy: WatermarkStrategy,
    /// Current watermark value (starts at `i64::MIN`).
    current_watermark_ms: i64,
    /// Maximum event time observed so far.
    max_event_time_ms: i64,
    /// Tracks the last time a watermark was emitted (for `Periodic`).
    last_emit_processing_ms: u64,
}

impl WatermarkGenerator {
    /// Create a new generator for the given strategy.
    pub fn new(strategy: WatermarkStrategy) -> Self {
        Self {
            strategy,
            current_watermark_ms: i64::MIN,
            max_event_time_ms: i64::MIN,
            last_emit_processing_ms: 0,
        }
    }

    /// Observe an event timestamp, updating internal state.
    ///
    /// Returns `Some(new_watermark)` if the watermark advanced; `None`
    /// otherwise.
    pub fn observe_event(&mut self, event_time_ms: i64) -> Option<i64> {
        // Always update the max observed event time.
        if event_time_ms > self.max_event_time_ms {
            self.max_event_time_ms = event_time_ms;
        }

        let new_wm = self.compute_watermark();
        if new_wm > self.current_watermark_ms {
            self.current_watermark_ms = new_wm;
            Some(new_wm)
        } else {
            None
        }
    }

    /// Simulate a periodic heartbeat tick at the given processing timestamp.
    ///
    /// For `Periodic` strategies this may cause a watermark to be emitted
    /// even if no event arrived.
    pub fn tick(&mut self, processing_time_ms: u64) -> Option<i64> {
        if let WatermarkStrategy::Periodic { interval_ms, .. } = &self.strategy {
            if processing_time_ms >= self.last_emit_processing_ms + interval_ms {
                self.last_emit_processing_ms = processing_time_ms;
                let new_wm = self.compute_watermark();
                if new_wm > self.current_watermark_ms {
                    self.current_watermark_ms = new_wm;
                    return Some(new_wm);
                }
            }
        }
        None
    }

    /// Return the current watermark value.
    pub fn current_watermark(&self) -> i64 {
        self.current_watermark_ms
    }

    /// Return the maximum event time observed so far.
    pub fn max_event_time(&self) -> i64 {
        self.max_event_time_ms
    }

    // Compute the candidate watermark from current state.
    fn compute_watermark(&self) -> i64 {
        if self.max_event_time_ms == i64::MIN {
            return i64::MIN;
        }
        match &self.strategy {
            WatermarkStrategy::MonotonousTimestamps => self.max_event_time_ms,
            WatermarkStrategy::BoundedOutOfOrder { max_lateness_ms } => {
                self.max_event_time_ms - (*max_lateness_ms as i64)
            }
            WatermarkStrategy::Periodic { strategy, .. } => {
                // Delegate to the inner strategy.
                let inner_max = self.max_event_time_ms;
                match strategy.as_ref() {
                    WatermarkStrategy::MonotonousTimestamps => inner_max,
                    WatermarkStrategy::BoundedOutOfOrder { max_lateness_ms } => {
                        inner_max - (*max_lateness_ms as i64)
                    }
                    // Nested periodic — same as monotonous for simplicity.
                    _ => inner_max,
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// LateElementPolicy
// ---------------------------------------------------------------------------

/// What to do with events that arrive after the watermark has passed their
/// window's end time.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum LateElementPolicy {
    /// Silently discard late events.
    Drop,

    /// Allow events up to `grace_ms` milliseconds past the watermark.
    AllowLateness {
        /// Grace period in milliseconds.
        grace_ms: u64,
    },

    /// Route late events to a side output (application-defined sink).
    SideOutput,
}

impl LateElementPolicy {
    /// Return `true` if the event at `event_time_ms` should be processed
    /// given the current `watermark_ms`.
    pub fn should_process(&self, event_time_ms: i64, watermark_ms: i64) -> bool {
        match self {
            LateElementPolicy::Drop => event_time_ms >= watermark_ms,
            LateElementPolicy::AllowLateness { grace_ms } => {
                event_time_ms >= watermark_ms - (*grace_ms as i64)
            }
            LateElementPolicy::SideOutput => {
                // SideOutput processes everything; the caller decides routing.
                true
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TimestampedRecord
// ---------------------------------------------------------------------------

/// A value annotated with both event time and processing time.
#[derive(Debug, Clone)]
pub struct TimestampedRecord<T> {
    /// The payload.
    pub value: T,
    /// Event time assigned by the source (milliseconds since epoch).
    pub event_time_ms: i64,
    /// Processing time when the record was observed (milliseconds since epoch).
    pub processing_time_ms: i64,
}

impl<T> TimestampedRecord<T> {
    /// Create a new timestamped record, stamping processing time automatically.
    pub fn new(value: T, event_time_ms: i64) -> Self {
        let processing_time_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        Self {
            value,
            event_time_ms,
            processing_time_ms,
        }
    }

    /// Create a record with explicit processing time (useful in tests).
    pub fn with_processing_time(value: T, event_time_ms: i64, processing_time_ms: i64) -> Self {
        Self {
            value,
            event_time_ms,
            processing_time_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monotone_watermark_never_decreases() {
        let mut gen = WatermarkGenerator::new(WatermarkStrategy::MonotonousTimestamps);
        gen.observe_event(1000);
        let wm1 = gen.current_watermark();
        gen.observe_event(500); // Out of order — should NOT decrease watermark.
        let wm2 = gen.current_watermark();
        assert!(
            wm2 >= wm1,
            "Watermark must be monotone: wm1={wm1}, wm2={wm2}"
        );
    }

    #[test]
    fn test_monotone_watermark_advances_with_max_event() {
        let mut gen = WatermarkGenerator::new(WatermarkStrategy::MonotonousTimestamps);
        gen.observe_event(500);
        assert_eq!(gen.current_watermark(), 500);
        gen.observe_event(1200);
        assert_eq!(gen.current_watermark(), 1200);
        gen.observe_event(800); // Late event — watermark stays at 1200.
        assert_eq!(gen.current_watermark(), 1200);
    }

    #[test]
    fn test_bounded_out_of_order_delays_by_max_lateness() {
        let mut gen = WatermarkGenerator::new(WatermarkStrategy::BoundedOutOfOrder {
            max_lateness_ms: 500,
        });
        gen.observe_event(1000);
        assert_eq!(gen.current_watermark(), 500); // 1000 - 500
        gen.observe_event(1200);
        assert_eq!(gen.current_watermark(), 700); // 1200 - 500
    }

    #[test]
    fn test_bounded_out_of_order_watermark_never_decreases() {
        let mut gen = WatermarkGenerator::new(WatermarkStrategy::BoundedOutOfOrder {
            max_lateness_ms: 200,
        });
        gen.observe_event(1000);
        let wm1 = gen.current_watermark();
        gen.observe_event(900); // Late.
        let wm2 = gen.current_watermark();
        assert!(wm2 >= wm1);
    }

    #[test]
    fn test_late_element_policy_drop() {
        let policy = LateElementPolicy::Drop;
        assert!(!policy.should_process(500, 1000)); // 500 < 1000 → drop
        assert!(policy.should_process(1001, 1000)); // 1001 ≥ 1000 → keep
    }

    #[test]
    fn test_late_element_policy_allow_lateness() {
        let policy = LateElementPolicy::AllowLateness { grace_ms: 300 };
        // Watermark = 1000; grace = 300 → accept events ≥ 700.
        assert!(!policy.should_process(699, 1000));
        assert!(policy.should_process(700, 1000));
        assert!(policy.should_process(1200, 1000));
    }

    #[test]
    fn test_timestamped_record_preserves_values() {
        let rec = TimestampedRecord::with_processing_time(42u64, 9000, 10000);
        assert_eq!(rec.value, 42u64);
        assert_eq!(rec.event_time_ms, 9000);
        assert_eq!(rec.processing_time_ms, 10000);
    }
}
