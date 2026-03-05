//! Event-time watermarks for out-of-order stream processing.
//!
//! Watermarks represent the operator's estimate of how far event time has
//! progressed. When a watermark W is emitted, the system declares that all
//! events with timestamps ≤ W have been (or should have been) received.
//!
//! The `EventTimeTracker` advances the watermark based on the maximum observed
//! event timestamp minus a configurable out-of-order delay.

/// Watermark — a monotonically increasing event-time progress marker.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Watermark {
    /// The event-time timestamp this watermark represents.
    pub timestamp: f64,
}

impl Watermark {
    /// Create a new watermark at the given event-time timestamp.
    pub fn new(timestamp: f64) -> Self {
        Self { timestamp }
    }

    /// Returns true if this watermark is strictly more recent than `other`.
    pub fn is_after(&self, other: &Watermark) -> bool {
        self.timestamp > other.timestamp
    }

    /// Returns true if this watermark indicates the event time is at least `t`.
    pub fn covers(&self, event_time: f64) -> bool {
        self.timestamp >= event_time
    }
}

/// Tracks event time and generates periodic watermarks.
///
/// The watermark is defined as `max_observed_event_time - max_out_of_order_delay`.
/// Events whose timestamp is ≤ the current watermark are considered "late".
#[derive(Debug, Clone)]
pub struct EventTimeTracker {
    /// Largest event timestamp seen so far.
    pub max_event_time: f64,
    /// Current watermark (= max_event_time - max_out_of_order_delay).
    pub watermark: f64,
    /// Maximum tolerated out-of-order delay in seconds.
    pub max_out_of_order_delay: f64,
    /// Total watermarks emitted.
    pub watermarks_emitted: usize,
    /// Count of late (below watermark) events received.
    pub late_events: usize,
}

impl EventTimeTracker {
    /// Create a new tracker with the given maximum allowed out-of-order delay.
    pub fn new(max_delay: f64) -> Self {
        Self {
            max_event_time: f64::NEG_INFINITY,
            watermark: f64::NEG_INFINITY,
            max_out_of_order_delay: max_delay,
            watermarks_emitted: 0,
            late_events: 0,
        }
    }

    /// Process an event's timestamp. Returns a new `Watermark` if the watermark advanced.
    pub fn process_event(&mut self, event_time: f64) -> Option<Watermark> {
        if event_time <= self.watermark {
            self.late_events += 1;
        }

        if event_time > self.max_event_time {
            self.max_event_time = event_time;
        }

        let new_wm = self.max_event_time - self.max_out_of_order_delay;
        if new_wm > self.watermark {
            self.watermark = new_wm;
            self.watermarks_emitted += 1;
            Some(Watermark {
                timestamp: new_wm,
            })
        } else {
            None
        }
    }

    /// Forcibly advance the watermark to `time` (useful for periodic punctuation).
    /// Returns `Some(Watermark)` if the watermark advanced.
    pub fn advance_to(&mut self, time: f64) -> Option<Watermark> {
        if time > self.watermark {
            self.watermark = time;
            self.watermarks_emitted += 1;
            Some(Watermark { timestamp: time })
        } else {
            None
        }
    }

    /// Returns true if the given event time is considered late (below current watermark).
    pub fn is_event_late(&self, event_time: f64) -> bool {
        event_time <= self.watermark
    }

    /// Returns the current watermark, if one has been emitted.
    pub fn current_watermark(&self) -> Option<Watermark> {
        if self.watermark == f64::NEG_INFINITY {
            None
        } else {
            Some(Watermark {
                timestamp: self.watermark,
            })
        }
    }
}

/// Multi-stream watermark coordinator.
///
/// Computes the global watermark as the minimum watermark across all input
/// streams (the "bottleneck" watermark, as in Apache Flink's approach).
#[derive(Debug, Clone)]
pub struct MultiStreamWatermark {
    stream_watermarks: Vec<f64>,
    /// The current global (minimum) watermark across all streams.
    pub global_watermark: f64,
}

impl MultiStreamWatermark {
    /// Create a coordinator tracking `num_streams` independent input streams.
    pub fn new(num_streams: usize) -> Self {
        Self {
            stream_watermarks: vec![f64::NEG_INFINITY; num_streams],
            global_watermark: f64::NEG_INFINITY,
        }
    }

    /// Update the watermark for stream `stream_id`.
    ///
    /// Returns the new global (minimum) watermark if it advanced.
    pub fn update_stream(&mut self, stream_id: usize, watermark: f64) -> Option<Watermark> {
        if stream_id >= self.stream_watermarks.len() {
            return None;
        }
        self.stream_watermarks[stream_id] = watermark;

        // Global = min of all stream watermarks
        let new_global = self
            .stream_watermarks
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);

        if new_global > self.global_watermark {
            self.global_watermark = new_global;
            Some(Watermark {
                timestamp: new_global,
            })
        } else {
            None
        }
    }

    /// Returns the number of input streams being tracked.
    pub fn num_streams(&self) -> usize {
        self.stream_watermarks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watermark_advancement() {
        let mut tracker = EventTimeTracker::new(2.0);

        // Events arrive slightly out of order; watermark lags by 2.0 seconds
        let wm = tracker.process_event(10.0);
        assert!(wm.is_some());
        assert_eq!(wm.expect("watermark should advance").timestamp, 8.0);

        let wm = tracker.process_event(12.0);
        assert!(wm.is_some());
        assert_eq!(wm.expect("watermark should advance").timestamp, 10.0);

        // Duplicate / slightly lower event — watermark does not advance
        let wm = tracker.process_event(11.0);
        assert!(wm.is_none(), "Watermark should not advance for lower event");
    }

    #[test]
    fn test_late_event_detection() {
        let mut tracker = EventTimeTracker::new(0.0);

        tracker.process_event(10.0);
        // t=10 is right at the watermark: event at 9 should be late
        assert!(tracker.is_event_late(9.0));
        assert!(!tracker.is_event_late(11.0));
    }

    #[test]
    fn test_advance_to() {
        let mut tracker = EventTimeTracker::new(1.0);
        let wm = tracker.advance_to(100.0);
        assert!(wm.is_some());
        assert_eq!(wm.expect("advance_to should emit watermark").timestamp, 100.0);

        // advance_to with lower value — no new watermark
        let wm = tracker.advance_to(50.0);
        assert!(wm.is_none());
    }

    #[test]
    fn test_multi_stream_watermark() {
        let mut msw = MultiStreamWatermark::new(3);

        // Only two of three streams have advanced
        msw.update_stream(0, 10.0);
        msw.update_stream(1, 8.0);
        // Stream 2 still at -inf; global must not advance past -inf
        assert_eq!(msw.global_watermark, f64::NEG_INFINITY);

        msw.update_stream(2, 6.0);
        // Now global = min(10, 8, 6) = 6
        assert_eq!(msw.global_watermark, 6.0);
    }
}
