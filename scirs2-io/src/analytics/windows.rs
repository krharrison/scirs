//! Window specifications for streaming analytics.
//!
//! Provides tumbling (non-overlapping), sliding (overlapping), and session
//! (gap-based) window processors for time-ordered event streams.

use std::collections::VecDeque;

/// Specification of a window strategy
#[derive(Debug, Clone)]
pub enum WindowSpec {
    /// Fixed-size non-overlapping window (size in seconds)
    Tumbling(f64),
    /// Overlapping window with given size and step (in seconds)
    Sliding {
        /// Duration of the look-back interval in seconds.
        size: f64,
        /// Interval between successive window emissions in seconds.
        step: f64,
    },
    /// Session-based window that closes after a silence gap (in seconds)
    Session {
        /// Maximum inactivity gap (in seconds) before closing the session.
        gap: f64,
    },
}

/// A time-stamped record in a windowed stream
#[derive(Debug, Clone)]
pub struct Record<T: Clone> {
    /// Event timestamp in seconds.
    pub timestamp: f64,
    /// Optional partition key for keyed streams.
    pub key: Option<String>,
    /// The payload value.
    pub value: T,
}

impl<T: Clone> Record<T> {
    /// Create an un-keyed record with the given timestamp and value.
    pub fn new(timestamp: f64, value: T) -> Self {
        Self {
            timestamp,
            key: None,
            value,
        }
    }

    /// Attach a partition key to this record.
    pub fn with_key(mut self, key: String) -> Self {
        self.key = Some(key);
        self
    }
}

/// Tumbling (non-overlapping) window processor.
///
/// Partitions the event stream into consecutive, non-overlapping windows of
/// equal duration. When an event falls outside the current window boundary,
/// the completed window is emitted and a new one begins.
pub struct TumblingWindow<T: Clone + Send> {
    /// Duration of each tumbling window in seconds.
    pub window_size: f64,
    buffer: VecDeque<Record<T>>,
    window_start: Option<f64>,
    /// Total number of windows emitted so far.
    pub windows_emitted: usize,
}

impl<T: Clone + Send> TumblingWindow<T> {
    /// Create a new tumbling window of `window_size` seconds.
    pub fn new(window_size: f64) -> Self {
        assert!(window_size > 0.0, "window_size must be positive");
        Self {
            window_size,
            buffer: VecDeque::new(),
            window_start: None,
            windows_emitted: 0,
        }
    }

    /// Add a record; returns the completed window if one was closed.
    pub fn add(&mut self, record: Record<T>) -> Option<Vec<Record<T>>> {
        let ws = *self.window_start.get_or_insert(record.timestamp);
        let window_end = ws + self.window_size;

        if record.timestamp >= window_end {
            // Emit current window, start a new one
            let window: Vec<Record<T>> = self.buffer.drain(..).collect();
            self.window_start = Some(record.timestamp);
            self.buffer.push_back(record);
            self.windows_emitted += 1;
            if !window.is_empty() {
                Some(window)
            } else {
                None
            }
        } else {
            self.buffer.push_back(record);
            None
        }
    }

    /// Flush remaining buffered records as the final (possibly partial) window.
    pub fn flush(&mut self) -> Option<Vec<Record<T>>> {
        if !self.buffer.is_empty() {
            let window: Vec<Record<T>> = self.buffer.drain(..).collect();
            self.windows_emitted += 1;
            Some(window)
        } else {
            None
        }
    }

    /// Number of records currently buffered in the open window.
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }
}

/// Sliding (overlapping) window processor.
///
/// Produces a window every `step` seconds containing all events within
/// `[emit_time - size, emit_time]`. Records older than two steps before
/// the current window start are evicted from the buffer.
pub struct SlidingWindow<T: Clone + Send> {
    /// Window size in seconds (length of the look-back interval).
    pub size: f64,
    /// Step between successive window emissions in seconds.
    pub step: f64,
    buffer: VecDeque<Record<T>>,
    next_emit: Option<f64>,
    /// Total number of windows emitted so far.
    pub windows_emitted: usize,
}

impl<T: Clone + Send> SlidingWindow<T> {
    /// Create a new sliding window with the given size and step (both in seconds).
    pub fn new(size: f64, step: f64) -> Self {
        assert!(size > 0.0, "size must be positive");
        assert!(step > 0.0, "step must be positive");
        Self {
            size,
            step,
            buffer: VecDeque::new(),
            next_emit: None,
            windows_emitted: 0,
        }
    }

    /// Add a record; returns a window if the step boundary was crossed.
    pub fn add(&mut self, record: Record<T>) -> Option<Vec<Record<T>>> {
        let next_emit = self
            .next_emit
            .get_or_insert_with(|| record.timestamp + self.step);

        self.buffer.push_back(record.clone());

        if record.timestamp >= *next_emit {
            let emit_time = *next_emit;
            let window_start = emit_time - self.size;

            let window: Vec<Record<T>> = self
                .buffer
                .iter()
                .filter(|r| r.timestamp >= window_start && r.timestamp <= emit_time)
                .cloned()
                .collect();

            // Evict records too old to contribute to future windows
            let evict_before = window_start - self.step;
            while self
                .buffer
                .front()
                .map_or(false, |r| r.timestamp < evict_before)
            {
                self.buffer.pop_front();
            }

            *self.next_emit.as_mut().expect("next_emit set above") += self.step;
            self.windows_emitted += 1;
            if !window.is_empty() {
                Some(window)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Number of records currently in the internal buffer.
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }
}

/// Session window (gap-based) processor.
///
/// Groups events into sessions separated by inactivity gaps. When a new event
/// arrives more than `gap` seconds after the previous one, the old session is
/// emitted and a fresh session begins.
pub struct SessionWindow<T: Clone + Send> {
    /// Inactivity gap (in seconds) that closes a session.
    pub gap: f64,
    current_session: Vec<Record<T>>,
    last_timestamp: Option<f64>,
    /// Total number of sessions emitted so far.
    pub sessions_emitted: usize,
}

impl<T: Clone + Send> SessionWindow<T> {
    /// Create a new session window that emits a session when no event arrives within `gap` seconds.
    pub fn new(gap: f64) -> Self {
        assert!(gap > 0.0, "gap must be positive");
        Self {
            gap,
            current_session: Vec::new(),
            last_timestamp: None,
            sessions_emitted: 0,
        }
    }

    /// Add a record; returns the completed session if a gap was detected.
    pub fn add(&mut self, record: Record<T>) -> Option<Vec<Record<T>>> {
        let result = if let Some(last_ts) = self.last_timestamp {
            if record.timestamp - last_ts > self.gap && !self.current_session.is_empty() {
                let session = std::mem::take(&mut self.current_session);
                self.sessions_emitted += 1;
                Some(session)
            } else {
                None
            }
        } else {
            None
        };

        self.last_timestamp = Some(record.timestamp);
        self.current_session.push(record);
        result
    }

    /// Flush the in-progress session.
    pub fn flush(&mut self) -> Option<Vec<Record<T>>> {
        if !self.current_session.is_empty() {
            let session = std::mem::take(&mut self.current_session);
            self.sessions_emitted += 1;
            Some(session)
        } else {
            None
        }
    }

    /// Number of records in the currently open session.
    pub fn current_session_size(&self) -> usize {
        self.current_session.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tumbling_window_emits_correct_windows() {
        let mut tw = TumblingWindow::new(10.0);

        // First window: timestamps 0..9
        for i in 0..10u64 {
            let r = Record::new(i as f64, i as f64);
            let out = tw.add(r);
            if i < 9 {
                assert!(out.is_none(), "No window yet at t={}", i);
            }
        }
        // Event at t=10 crosses the window boundary
        let r = Record::new(10.0, 10.0_f64);
        let window = tw.add(r);
        assert!(window.is_some());
        let window = window.expect("window should be present");
        assert_eq!(window.len(), 10); // 10 records in [0,10)

        // Flush the partial second window
        let final_window = tw.flush();
        assert!(final_window.is_some());
        let final_window = final_window.expect("final window should be present");
        assert_eq!(final_window.len(), 1); // just t=10
    }

    #[test]
    fn test_sliding_window_overlap() {
        let mut sw = SlidingWindow::new(10.0, 5.0);
        let mut windows_seen = 0usize;

        for i in 0..20u64 {
            let r = Record::new(i as f64, i as f64);
            if let Some(w) = sw.add(r) {
                windows_seen += 1;
                // Every window must have records within [emit - 10, emit]
                let max_ts = w.iter().map(|r| r.timestamp as u64).max().unwrap_or(0);
                let min_ts = w.iter().map(|r| r.timestamp as u64).min().unwrap_or(0);
                assert!(max_ts - min_ts <= 10, "Window too wide: {} to {}", min_ts, max_ts);
            }
        }
        // With step=5 and 20 events we should see several windows
        assert!(windows_seen >= 2, "Expected multiple windows, got {}", windows_seen);
    }

    #[test]
    fn test_session_window_gap_detection() {
        let mut sw = SessionWindow::new(5.0);

        // Events 0..4 form one session
        for i in 0..5u64 {
            let r = Record::new(i as f64, i as f64);
            let out = sw.add(r);
            assert!(out.is_none(), "No session closed yet at t={}", i);
        }

        // Event at t=10 is > 5 seconds after t=4, triggering session close
        let r = Record::new(10.0, 10.0_f64);
        let session = sw.add(r);
        assert!(session.is_some(), "Session should have been emitted");
        let session = session.expect("session should be present");
        assert_eq!(session.len(), 5);

        // Flush final session
        let final_session = sw.flush();
        assert!(final_session.is_some());
        let final_session = final_session.expect("final session should be present");
        assert_eq!(final_session.len(), 1);
        assert_eq!(sw.sessions_emitted, 2);
    }
}
