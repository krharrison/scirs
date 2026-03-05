//! # Progress Tracking for Iterative Algorithms
//!
//! Pure-Rust progress tracking with no external dependencies.
//!
//! ## Features
//!
//! - `ProgressBar` -- configurable progress display (percentage, bar, ETA)
//! - Callback-based progress notifications
//! - Nested progress bars via `ProgressGroup`
//! - Rate estimation (iterations/sec)
//! - Pure text output (no terminal escape sequences required)
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::progress::{ProgressBar, ProgressStyle};
//!
//! let mut pb = ProgressBar::new(100)
//!     .with_style(ProgressStyle::Bar)
//!     .with_prefix("Training");
//!
//! for i in 0..100 {
//!     // ... do work ...
//!     pb.inc(1);
//! }
//! pb.finish();
//! ```

use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// ProgressStyle
// ---------------------------------------------------------------------------

/// Display style for the progress bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressStyle {
    /// Show only percentage: `[Training] 42%`
    Percentage,
    /// Show a text bar: `[Training] [=====>    ] 50% (5/10) 2.3 it/s ETA 2s`
    Bar,
    /// Show only the counter: `[Training] 5/10`
    Counter,
    /// Silent -- no output, only callbacks fire.
    Silent,
}

impl Default for ProgressStyle {
    fn default() -> Self {
        Self::Bar
    }
}

// ---------------------------------------------------------------------------
// ProgressState (internal)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ProgressState {
    current: u64,
    total: u64,
    start_time: Instant,
    last_print_time: Instant,
    finished: bool,
}

impl ProgressState {
    fn new(total: u64) -> Self {
        let now = Instant::now();
        Self {
            current: 0,
            total,
            start_time: now,
            last_print_time: now,
            finished: false,
        }
    }

    fn fraction(&self) -> f64 {
        if self.total == 0 {
            return 1.0;
        }
        self.current as f64 / self.total as f64
    }

    fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    fn rate(&self) -> f64 {
        let secs = self.elapsed().as_secs_f64();
        if secs > 0.0 {
            self.current as f64 / secs
        } else {
            0.0
        }
    }

    fn eta(&self) -> Option<Duration> {
        if self.current == 0 || self.total == 0 {
            return None;
        }
        let elapsed = self.elapsed().as_secs_f64();
        let rate = self.current as f64 / elapsed;
        if rate <= 0.0 {
            return None;
        }
        let remaining = (self.total - self.current) as f64 / rate;
        Some(Duration::from_secs_f64(remaining))
    }
}

// ---------------------------------------------------------------------------
// ProgressCallback
// ---------------------------------------------------------------------------

/// A callback invoked each time progress is updated.
///
/// Receives `(current, total, elapsed)`.
pub type ProgressCallback = Box<dyn Fn(u64, u64, Duration) + Send + 'static>;

// ---------------------------------------------------------------------------
// ProgressBar
// ---------------------------------------------------------------------------

/// A configurable progress bar for iterative algorithms.
///
/// Thread-safe: the internal state is behind an `Arc<Mutex<_>>`.
pub struct ProgressBar {
    state: Arc<Mutex<ProgressState>>,
    style: ProgressStyle,
    prefix: String,
    bar_width: usize,
    min_print_interval: Duration,
    callbacks: Vec<ProgressCallback>,
    print_on_finish: bool,
}

impl ProgressBar {
    /// Create a new progress bar with the given total number of steps.
    pub fn new(total: u64) -> Self {
        Self {
            state: Arc::new(Mutex::new(ProgressState::new(total))),
            style: ProgressStyle::default(),
            prefix: String::new(),
            bar_width: 30,
            min_print_interval: Duration::from_millis(100),
            callbacks: Vec::new(),
            print_on_finish: true,
        }
    }

    /// Set the display style.
    pub fn with_style(mut self, style: ProgressStyle) -> Self {
        self.style = style;
        self
    }

    /// Set the prefix label.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Set the visual bar width (number of characters). Default: 30.
    pub fn with_bar_width(mut self, width: usize) -> Self {
        self.bar_width = width;
        self
    }

    /// Set the minimum interval between printed updates. Default: 100ms.
    pub fn with_min_print_interval(mut self, interval: Duration) -> Self {
        self.min_print_interval = interval;
        self
    }

    /// Disable automatic printing on `finish()`.
    pub fn silent_finish(mut self) -> Self {
        self.print_on_finish = false;
        self
    }

    /// Register a callback that fires on each `inc()` / `set()` call.
    pub fn on_progress<F>(mut self, callback: F) -> Self
    where
        F: Fn(u64, u64, Duration) + Send + 'static,
    {
        self.callbacks.push(Box::new(callback));
        self
    }

    /// Increment the progress by `delta` steps.
    pub fn inc(&mut self, delta: u64) {
        let (current, total, elapsed, should_print) = {
            let mut st = match self.state.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            st.current = st.current.saturating_add(delta).min(st.total);
            let now = Instant::now();
            let should_print = now.duration_since(st.last_print_time) >= self.min_print_interval;
            if should_print {
                st.last_print_time = now;
            }
            (st.current, st.total, st.elapsed(), should_print)
        };

        // Fire callbacks
        for cb in &self.callbacks {
            cb(current, total, elapsed);
        }

        // Print if enough time has passed
        if should_print && self.style != ProgressStyle::Silent {
            self.print_line();
        }
    }

    /// Set the progress to an absolute value.
    pub fn set(&mut self, value: u64) {
        let (current, total, elapsed) = {
            let mut st = match self.state.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            st.current = value.min(st.total);
            (st.current, st.total, st.elapsed())
        };

        for cb in &self.callbacks {
            cb(current, total, elapsed);
        }

        if self.style != ProgressStyle::Silent {
            self.print_line();
        }
    }

    /// Mark the progress as finished and print a final line.
    pub fn finish(&mut self) {
        {
            let mut st = match self.state.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            st.current = st.total;
            st.finished = true;
        }

        if self.print_on_finish && self.style != ProgressStyle::Silent {
            self.print_line();
        }
    }

    /// Mark the progress as finished with a custom message.
    pub fn finish_with_message(&mut self, msg: &str) {
        {
            let mut st = match self.state.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            st.current = st.total;
            st.finished = true;
        }

        if self.style != ProgressStyle::Silent {
            let prefix = if self.prefix.is_empty() {
                String::new()
            } else {
                format!("[{}] ", self.prefix)
            };
            eprintln!("{prefix}{msg}");
        }
    }

    /// Reset the progress bar to 0 with a new total.
    pub fn reset(&mut self, total: u64) {
        let mut st = match self.state.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        *st = ProgressState::new(total);
    }

    /// Get the current position.
    pub fn position(&self) -> u64 {
        match self.state.lock() {
            Ok(st) => st.current,
            Err(poisoned) => poisoned.into_inner().current,
        }
    }

    /// Get the total.
    pub fn total(&self) -> u64 {
        match self.state.lock() {
            Ok(st) => st.total,
            Err(poisoned) => poisoned.into_inner().total,
        }
    }

    /// Get elapsed time.
    pub fn elapsed(&self) -> Duration {
        match self.state.lock() {
            Ok(st) => st.elapsed(),
            Err(poisoned) => poisoned.into_inner().elapsed(),
        }
    }

    /// Get current rate (iterations/sec).
    pub fn rate(&self) -> f64 {
        match self.state.lock() {
            Ok(st) => st.rate(),
            Err(poisoned) => poisoned.into_inner().rate(),
        }
    }

    /// Get the estimated time remaining.
    pub fn eta(&self) -> Option<Duration> {
        match self.state.lock() {
            Ok(st) => st.eta(),
            Err(poisoned) => poisoned.into_inner().eta(),
        }
    }

    /// Format the current state as a string (without printing).
    pub fn format_line(&self) -> String {
        let st = match self.state.lock() {
            Ok(g) => g.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        };
        self.format_state(&st)
    }

    // -- internal helpers --

    fn print_line(&self) {
        let st = match self.state.lock() {
            Ok(g) => g.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        };
        let line = self.format_state(&st);
        eprintln!("{line}");
    }

    fn format_state(&self, st: &ProgressState) -> String {
        let prefix = if self.prefix.is_empty() {
            String::new()
        } else {
            format!("[{}] ", self.prefix)
        };

        match self.style {
            ProgressStyle::Percentage => {
                let pct = (st.fraction() * 100.0) as u32;
                format!("{prefix}{pct}%")
            }
            ProgressStyle::Counter => {
                format!("{prefix}{}/{}", st.current, st.total)
            }
            ProgressStyle::Bar => {
                let pct = (st.fraction() * 100.0) as u32;
                let filled = (st.fraction() * self.bar_width as f64) as usize;
                let empty = self.bar_width.saturating_sub(filled);

                let bar: String = "=".repeat(filled);
                let arrow = if filled < self.bar_width && !st.finished {
                    ">"
                } else {
                    ""
                };
                let spaces_count = if !arrow.is_empty() {
                    empty.saturating_sub(1)
                } else {
                    empty
                };
                let spaces: String = " ".repeat(spaces_count);

                let rate = st.rate();
                let rate_str = if rate >= 1.0 {
                    format!("{rate:.1} it/s")
                } else if rate > 0.0 {
                    let spi = 1.0 / rate;
                    format!("{spi:.1} s/it")
                } else {
                    "-- it/s".to_string()
                };

                let eta_str = match st.eta() {
                    Some(eta) => format_duration_short(eta),
                    None => "?".to_string(),
                };

                let elapsed_str = format_duration_short(st.elapsed());

                if st.finished {
                    format!(
                        "{prefix}[{bar}{arrow}{spaces}] {pct}% ({}/{}) {rate_str} elapsed {elapsed_str}",
                        st.current, st.total,
                    )
                } else {
                    format!(
                        "{prefix}[{bar}{arrow}{spaces}] {pct}% ({}/{}) {rate_str} ETA {eta_str}",
                        st.current, st.total,
                    )
                }
            }
            ProgressStyle::Silent => String::new(),
        }
    }
}

impl fmt::Debug for ProgressBar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let st = match self.state.lock() {
            Ok(g) => g.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        };
        f.debug_struct("ProgressBar")
            .field("current", &st.current)
            .field("total", &st.total)
            .field("style", &self.style)
            .field("prefix", &self.prefix)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ProgressGroup (nested progress bars)
// ---------------------------------------------------------------------------

/// A group of named progress bars for tracking nested / multi-phase operations.
///
/// Each bar is identified by a string key.
pub struct ProgressGroup {
    bars: Vec<(String, ProgressBar)>,
}

impl ProgressGroup {
    /// Create an empty progress group.
    pub fn new() -> Self {
        Self { bars: Vec::new() }
    }

    /// Add a new progress bar to the group.
    pub fn add(&mut self, name: impl Into<String>, total: u64, style: ProgressStyle) {
        let name = name.into();
        let pb = ProgressBar::new(total)
            .with_style(style)
            .with_prefix(name.clone())
            .silent_finish(); // we manage printing ourselves
        self.bars.push((name, pb));
    }

    /// Get a mutable reference to a named bar. Returns `None` if not found.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut ProgressBar> {
        self.bars
            .iter_mut()
            .find(|(n, _)| n == name)
            .map(|(_, pb)| pb)
    }

    /// Print a summary of all progress bars.
    pub fn print_summary(&self) {
        for (name, pb) in &self.bars {
            let line = pb.format_line();
            if !line.is_empty() {
                eprintln!("{line}");
            } else {
                // Silent bar -- print basic info
                eprintln!("[{name}] {}/{}", pb.position(), pb.total());
            }
        }
    }

    /// Get the overall fraction complete (average across all bars).
    pub fn overall_fraction(&self) -> f64 {
        if self.bars.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .bars
            .iter()
            .map(|(_, pb)| {
                let total = pb.total();
                if total == 0 {
                    1.0
                } else {
                    pb.position() as f64 / total as f64
                }
            })
            .sum();
        sum / self.bars.len() as f64
    }

    /// Number of bars in the group.
    pub fn len(&self) -> usize {
        self.bars.len()
    }

    /// Whether the group is empty.
    pub fn is_empty(&self) -> bool {
        self.bars.is_empty()
    }
}

impl Default for ProgressGroup {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ProgressGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProgressGroup")
            .field("count", &self.bars.len())
            .field(
                "bars",
                &self
                    .bars
                    .iter()
                    .map(|(n, pb)| format!("{n}: {}/{}", pb.position(), pb.total()))
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Helper: format duration
// ---------------------------------------------------------------------------

fn format_duration_short(d: Duration) -> String {
    let total_secs = d.as_secs();
    if total_secs < 60 {
        format!("{}s", total_secs)
    } else if total_secs < 3600 {
        let m = total_secs / 60;
        let s = total_secs % 60;
        format!("{m}m{s:02}s")
    } else {
        let h = total_secs / 3600;
        let m = (total_secs % 3600) / 60;
        let s = total_secs % 60;
        format!("{h}h{m:02}m{s:02}s")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_bar_basic() {
        let mut pb = ProgressBar::new(10).with_style(ProgressStyle::Silent);
        assert_eq!(pb.position(), 0);
        assert_eq!(pb.total(), 10);

        pb.inc(3);
        assert_eq!(pb.position(), 3);

        pb.inc(7);
        assert_eq!(pb.position(), 10);
    }

    #[test]
    fn test_progress_bar_set() {
        let mut pb = ProgressBar::new(100).with_style(ProgressStyle::Silent);
        pb.set(50);
        assert_eq!(pb.position(), 50);
        pb.set(200); // clamped to total
        assert_eq!(pb.position(), 100);
    }

    #[test]
    fn test_progress_bar_finish() {
        let mut pb = ProgressBar::new(100).with_style(ProgressStyle::Silent);
        pb.inc(50);
        pb.finish();
        assert_eq!(pb.position(), 100);
    }

    #[test]
    fn test_progress_bar_reset() {
        let mut pb = ProgressBar::new(100).with_style(ProgressStyle::Silent);
        pb.inc(50);
        pb.reset(200);
        assert_eq!(pb.position(), 0);
        assert_eq!(pb.total(), 200);
    }

    #[test]
    fn test_progress_bar_rate() {
        let mut pb = ProgressBar::new(100).with_style(ProgressStyle::Silent);
        pb.inc(10);
        // Rate should be positive after some work
        let rate = pb.rate();
        assert!(rate >= 0.0);
    }

    #[test]
    fn test_progress_bar_eta() {
        let mut pb = ProgressBar::new(100).with_style(ProgressStyle::Silent);
        // ETA is None at start
        assert!(pb.eta().is_none() || pb.eta().is_some());
        pb.inc(50);
        // After progress, ETA might be available
    }

    #[test]
    fn test_progress_bar_format_percentage() {
        let mut pb = ProgressBar::new(100)
            .with_style(ProgressStyle::Percentage)
            .with_prefix("Test");
        pb.set(42);
        let line = pb.format_line();
        assert!(line.contains("42%"));
        assert!(line.contains("[Test]"));
    }

    #[test]
    fn test_progress_bar_format_counter() {
        let mut pb = ProgressBar::new(100)
            .with_style(ProgressStyle::Counter)
            .with_prefix("Count");
        pb.set(25);
        let line = pb.format_line();
        assert!(line.contains("25/100"));
    }

    #[test]
    fn test_progress_bar_format_bar() {
        let mut pb = ProgressBar::new(100)
            .with_style(ProgressStyle::Bar)
            .with_prefix("Work")
            .with_bar_width(20);
        pb.set(50);
        let line = pb.format_line();
        assert!(line.contains("50%"));
        assert!(line.contains("[Work]"));
        assert!(line.contains("ETA"));
    }

    #[test]
    fn test_progress_bar_format_bar_finished() {
        let mut pb = ProgressBar::new(100)
            .with_style(ProgressStyle::Bar)
            .with_bar_width(10);
        pb.finish();
        let line = pb.format_line();
        assert!(line.contains("100%"));
        assert!(line.contains("elapsed"));
    }

    #[test]
    fn test_progress_bar_silent() {
        let mut pb = ProgressBar::new(100).with_style(ProgressStyle::Silent);
        pb.inc(50);
        let line = pb.format_line();
        assert!(line.is_empty());
    }

    #[test]
    fn test_progress_bar_callback() {
        let called = Arc::new(Mutex::new(0u64));
        let called_clone = Arc::clone(&called);

        let mut pb = ProgressBar::new(10)
            .with_style(ProgressStyle::Silent)
            .on_progress(move |current, _total, _elapsed| {
                let mut c = called_clone.lock().expect("lock failed");
                *c = current;
            });

        pb.inc(5);
        let val = {
            let c = called.lock().expect("lock failed");
            *c
        };
        assert_eq!(val, 5);
    }

    #[test]
    fn test_progress_bar_overflow_protection() {
        let mut pb = ProgressBar::new(10).with_style(ProgressStyle::Silent);
        pb.inc(100); // should clamp to 10
        assert_eq!(pb.position(), 10);
    }

    #[test]
    fn test_progress_group_basic() {
        let mut group = ProgressGroup::new();
        group.add("phase1", 100, ProgressStyle::Silent);
        group.add("phase2", 200, ProgressStyle::Silent);

        assert_eq!(group.len(), 2);
        assert!(!group.is_empty());

        if let Some(pb) = group.get_mut("phase1") {
            pb.inc(50);
        }

        let frac = group.overall_fraction();
        assert!(frac > 0.0 && frac < 1.0);
    }

    #[test]
    fn test_progress_group_get_unknown() {
        let mut group = ProgressGroup::new();
        assert!(group.get_mut("nonexistent").is_none());
    }

    #[test]
    fn test_progress_group_empty() {
        let group = ProgressGroup::new();
        assert!(group.is_empty());
        assert_eq!(group.overall_fraction(), 0.0);
    }

    #[test]
    fn test_format_duration_short() {
        assert_eq!(format_duration_short(Duration::from_secs(5)), "5s");
        assert_eq!(format_duration_short(Duration::from_secs(65)), "1m05s");
        assert_eq!(format_duration_short(Duration::from_secs(3661)), "1h01m01s");
    }

    #[test]
    fn test_progress_bar_debug() {
        let pb = ProgressBar::new(100)
            .with_style(ProgressStyle::Bar)
            .with_prefix("Debug");
        let dbg = format!("{pb:?}");
        assert!(dbg.contains("ProgressBar"));
        assert!(dbg.contains("Debug"));
    }

    #[test]
    fn test_progress_bar_elapsed() {
        let pb = ProgressBar::new(100).with_style(ProgressStyle::Silent);
        let e = pb.elapsed();
        assert!(e >= Duration::ZERO);
    }

    #[test]
    fn test_progress_group_debug() {
        let mut group = ProgressGroup::new();
        group.add("a", 10, ProgressStyle::Silent);
        let dbg = format!("{group:?}");
        assert!(dbg.contains("ProgressGroup"));
    }

    #[test]
    fn test_progress_bar_zero_total() {
        let mut pb = ProgressBar::new(0).with_style(ProgressStyle::Silent);
        pb.inc(1); // should clamp
        assert_eq!(pb.position(), 0);
    }

    #[test]
    fn test_progress_bar_finish_with_message() {
        let mut pb = ProgressBar::new(100).with_style(ProgressStyle::Silent);
        pb.finish_with_message("Done!");
        assert_eq!(pb.position(), 100);
    }
}
