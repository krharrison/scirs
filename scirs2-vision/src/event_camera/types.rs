//! Core event camera data types.
//!
//! Provides [`Event`], [`EventSlice`], [`EventFrame`], and [`EventProcessingConfig`]
//! for representing and configuring Dynamic Vision Sensor data.

use scirs2_core::ndarray::Array2;

use crate::error::{Result, VisionError};

/// Polarity of a brightness change event.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Polarity {
    /// Brightness increase (log-intensity crossed threshold upward).
    On,
    /// Brightness decrease (log-intensity crossed threshold downward).
    Off,
}

impl Polarity {
    /// Returns `1.0` for [`Polarity::On`] and `-1.0` for [`Polarity::Off`].
    pub fn sign(self) -> f64 {
        match self {
            Polarity::On => 1.0,
            Polarity::Off => -1.0,
        }
    }
}

/// A single event from a Dynamic Vision Sensor (DVS).
///
/// Each event records a brightness change at a specific pixel coordinate and time.
#[derive(Clone, Copy, Debug)]
pub struct Event {
    /// Pixel x coordinate (column).
    pub x: u16,
    /// Pixel y coordinate (row).
    pub y: u16,
    /// Timestamp in seconds (microsecond precision).
    pub timestamp: f64,
    /// Polarity of the brightness change.
    pub polarity: Polarity,
}

impl Event {
    /// Creates a new event.
    pub fn new(x: u16, y: u16, timestamp: f64, polarity: Polarity) -> Self {
        Self {
            x,
            y,
            timestamp,
            polarity,
        }
    }
}

/// A collection of events within a time window, associated with a sensor resolution.
///
/// Events are stored sorted by timestamp. The slice tracks the sensor dimensions
/// and the temporal extent of the contained events.
pub struct EventSlice {
    events: Vec<Event>,
    t_start: f64,
    t_end: f64,
    width: u16,
    height: u16,
}

impl EventSlice {
    /// Creates a new `EventSlice` from a vector of events and the sensor resolution.
    ///
    /// The events are sorted by timestamp internally. Returns an error if the
    /// event list is empty or any event coordinate exceeds the given dimensions.
    pub fn new(mut events: Vec<Event>, width: u16, height: u16) -> Result<Self> {
        if events.is_empty() {
            return Err(VisionError::InvalidParameter(
                "EventSlice requires at least one event".to_string(),
            ));
        }

        // Validate coordinates
        for e in &events {
            if e.x >= width || e.y >= height {
                return Err(VisionError::InvalidParameter(format!(
                    "Event coordinate ({}, {}) exceeds sensor dimensions ({}x{})",
                    e.x, e.y, width, height
                )));
            }
        }

        // Sort by timestamp for deterministic processing
        events.sort_by(|a, b| {
            a.timestamp
                .partial_cmp(&b.timestamp)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let t_start = events.first().map(|e| e.timestamp).unwrap_or_default();
        let t_end = events.last().map(|e| e.timestamp).unwrap_or_default();

        Ok(Self {
            events,
            t_start,
            t_end,
            width,
            height,
        })
    }

    /// Returns a reference to the event list.
    pub fn events(&self) -> &[Event] {
        &self.events
    }

    /// Returns the time range `(t_start, t_end)`.
    pub fn time_range(&self) -> (f64, f64) {
        (self.t_start, self.t_end)
    }

    /// Returns the sensor width.
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Returns the sensor height.
    pub fn height(&self) -> u16 {
        self.height
    }

    /// Filters events by polarity, returning a new `EventSlice`.
    pub fn filter_by_polarity(&self, polarity: Polarity) -> Result<Self> {
        let filtered: Vec<Event> = self
            .events
            .iter()
            .filter(|e| e.polarity == polarity)
            .copied()
            .collect();

        if filtered.is_empty() {
            return Err(VisionError::InvalidParameter(format!(
                "No events with polarity {:?} found",
                polarity
            )));
        }

        Self::new(filtered, self.width, self.height)
    }

    /// Splits the event slice at a given timestamp `t`.
    ///
    /// Returns `(before, after)` where `before` contains events with `timestamp < t`
    /// and `after` contains events with `timestamp >= t`.
    /// Returns an error if either half would be empty.
    pub fn split_at_time(&self, t: f64) -> Result<(Self, Self)> {
        let before: Vec<Event> = self
            .events
            .iter()
            .filter(|e| e.timestamp < t)
            .copied()
            .collect();
        let after: Vec<Event> = self
            .events
            .iter()
            .filter(|e| e.timestamp >= t)
            .copied()
            .collect();

        if before.is_empty() {
            return Err(VisionError::InvalidParameter(
                "Split time is before all events; 'before' half would be empty".to_string(),
            ));
        }
        if after.is_empty() {
            return Err(VisionError::InvalidParameter(
                "Split time is after all events; 'after' half would be empty".to_string(),
            ));
        }

        Ok((
            Self::new(before, self.width, self.height)?,
            Self::new(after, self.width, self.height)?,
        ))
    }

    /// Returns the event rate in events per second.
    pub fn event_rate(&self) -> f64 {
        let duration = self.t_end - self.t_start;
        if duration <= 0.0 {
            return 0.0;
        }
        self.events.len() as f64 / duration
    }

    /// Returns the number of events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns `true` if there are no events.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Creates a sub-slice containing events within `[t0, t1)`.
    pub fn time_window(&self, t0: f64, t1: f64) -> Result<Self> {
        let sub: Vec<Event> = self
            .events
            .iter()
            .filter(|e| e.timestamp >= t0 && e.timestamp < t1)
            .copied()
            .collect();

        if sub.is_empty() {
            return Err(VisionError::InvalidParameter(format!(
                "No events in time window [{}, {})",
                t0, t1
            )));
        }

        Self::new(sub, self.width, self.height)
    }
}

/// An accumulated event frame — a 2D image generated from events.
pub struct EventFrame {
    /// Image data with shape `[height, width]`.
    pub data: Array2<f64>,
    /// Start of the temporal window.
    pub t_start: f64,
    /// End of the temporal window.
    pub t_end: f64,
}

/// Configuration for event processing operations.
pub struct EventProcessingConfig {
    /// Sensor width in pixels.
    pub width: u16,
    /// Sensor height in pixels.
    pub height: u16,
    /// Duration of each frame window in seconds.
    pub time_window: f64,
    /// Decay rate for exponential decay surface (tau).
    pub decay_rate: f64,
    /// Polarity threshold for noise filtering.
    pub polarity_threshold: f64,
}

impl Default for EventProcessingConfig {
    fn default() -> Self {
        Self {
            width: 240,
            height: 180,
            time_window: 0.033, // ~30 fps
            decay_rate: 0.01,   // 10 ms decay
            polarity_threshold: 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_creation() {
        let e = Event::new(10, 20, 1.0, Polarity::On);
        assert_eq!(e.x, 10);
        assert_eq!(e.y, 20);
        assert!((e.timestamp - 1.0).abs() < f64::EPSILON);
        assert_eq!(e.polarity, Polarity::On);
    }

    #[test]
    fn test_polarity_sign() {
        assert!((Polarity::On.sign() - 1.0).abs() < f64::EPSILON);
        assert!((Polarity::Off.sign() - (-1.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_event_slice_basic() {
        let events = vec![
            Event::new(0, 0, 0.002, Polarity::On),
            Event::new(1, 1, 0.001, Polarity::Off),
            Event::new(2, 2, 0.003, Polarity::On),
        ];
        let slice = EventSlice::new(events, 10, 10).expect("failed to create EventSlice");
        assert_eq!(slice.len(), 3);
        assert!(!slice.is_empty());
        // Should be sorted by timestamp
        assert!(slice.events()[0].timestamp <= slice.events()[1].timestamp);
        assert!(slice.events()[1].timestamp <= slice.events()[2].timestamp);
    }

    #[test]
    fn test_event_slice_out_of_bounds() {
        let events = vec![Event::new(10, 5, 0.0, Polarity::On)];
        let result = EventSlice::new(events, 10, 10); // x=10 is out of bounds for width=10
        assert!(result.is_err());
    }

    #[test]
    fn test_event_slice_empty() {
        let result = EventSlice::new(vec![], 10, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_filter_by_polarity() {
        let events = vec![
            Event::new(0, 0, 0.001, Polarity::On),
            Event::new(1, 1, 0.002, Polarity::Off),
            Event::new(2, 2, 0.003, Polarity::On),
        ];
        let slice = EventSlice::new(events, 10, 10).expect("failed");
        let on_slice = slice.filter_by_polarity(Polarity::On).expect("failed");
        assert_eq!(on_slice.len(), 2);
        for e in on_slice.events() {
            assert_eq!(e.polarity, Polarity::On);
        }
    }

    #[test]
    fn test_split_at_time() {
        let events = vec![
            Event::new(0, 0, 0.001, Polarity::On),
            Event::new(1, 1, 0.002, Polarity::Off),
            Event::new(2, 2, 0.003, Polarity::On),
            Event::new(3, 3, 0.004, Polarity::Off),
        ];
        let slice = EventSlice::new(events, 10, 10).expect("failed");
        let (before, after) = slice.split_at_time(0.0025).expect("failed");
        assert_eq!(before.len(), 2);
        assert_eq!(after.len(), 2);
    }

    #[test]
    fn test_event_rate() {
        let events = vec![
            Event::new(0, 0, 0.0, Polarity::On),
            Event::new(1, 1, 0.5, Polarity::Off),
            Event::new(2, 2, 1.0, Polarity::On),
        ];
        let slice = EventSlice::new(events, 10, 10).expect("failed");
        let rate = slice.event_rate();
        assert!((rate - 3.0).abs() < 1e-9); // 3 events / 1 second
    }

    #[test]
    fn test_time_window() {
        let events = vec![
            Event::new(0, 0, 0.0, Polarity::On),
            Event::new(1, 1, 0.5, Polarity::Off),
            Event::new(2, 2, 1.0, Polarity::On),
            Event::new(3, 3, 1.5, Polarity::Off),
        ];
        let slice = EventSlice::new(events, 10, 10).expect("failed");
        let sub = slice.time_window(0.3, 1.2).expect("failed");
        assert_eq!(sub.len(), 2); // events at 0.5 and 1.0
    }
}
