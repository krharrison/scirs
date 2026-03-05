//! Integration bridge between SciRS2 logging and the `log` crate facade.
//!
//! This module provides:
//! - `LogCrateBridge` -- forwards SciRS2 `Logger` entries to `log` macros
//! - `scirs2_time!` macro -- times an expression and logs the elapsed duration
//! - `MemoryTracker` -- lightweight memory usage tracking utility
//!
//! # Feature Gate
//!
//! The bridge to the `log` crate requires the `logging` feature to be enabled
//! on `scirs2-core`. If the feature is not enabled, the `log` forwarding
//! functionality will not produce output unless a `log` subscriber is
//! installed externally.
//!
//! # Usage
//!
//! ```rust,no_run
//! use scirs2_core::logging::bridge::{LogCrateBridge, MemoryTracker};
//!
//! // Install the bridge so SciRS2 logs flow to the `log` crate
//! LogCrateBridge::install();
//!
//! // Use the timing utility
//! let tracker = MemoryTracker::new();
//! tracker.record_allocation(1024);
//! let report = tracker.report();
//! ```

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use super::{LogEntry, LogHandler, LogLevel};

// ---------------------------------------------------------------------------
// LogCrateBridge -- forwards SciRS2 Logger entries to the `log` crate
// ---------------------------------------------------------------------------

/// A `LogHandler` that forwards SciRS2 log entries to the standard `log` crate.
///
/// Install it once via [`LogCrateBridge::install()`] and all SciRS2 internal
/// log messages will be emitted through whatever `log` subscriber the
/// application has configured (e.g., `env_logger`, `tracing-log`, etc.).
#[derive(Debug)]
pub struct LogCrateBridge;

impl LogCrateBridge {
    /// Register this bridge as a SciRS2 log handler.
    ///
    /// Safe to call multiple times; each call adds another handler, so prefer
    /// calling this exactly once at application startup.
    pub fn install() {
        let handler = Arc::new(LogCrateBridge);
        super::set_handler(handler);
    }

    /// Map SciRS2 `LogLevel` to `log::Level`.
    fn map_level(level: LogLevel) -> log::Level {
        match level {
            LogLevel::Trace => log::Level::Trace,
            LogLevel::Debug => log::Level::Debug,
            LogLevel::Info => log::Level::Info,
            LogLevel::Warn => log::Level::Warn,
            LogLevel::Error | LogLevel::Critical => log::Level::Error,
        }
    }
}

impl LogHandler for LogCrateBridge {
    fn handle(&self, entry: &LogEntry) {
        let level = Self::map_level(entry.level);
        // Use the `log` crate's logging macro to forward the message.
        // We build a record manually so we can set the target (module).
        log::log!(target: &entry.module, level, "{}", entry.message);
    }
}

// ---------------------------------------------------------------------------
// scirs2_time! macro -- performance timing
// ---------------------------------------------------------------------------

/// Times the execution of an expression and logs the elapsed duration.
///
/// # Examples
///
/// ```rust,no_run
/// use scirs2_core::scirs2_time;
///
/// let result = scirs2_time!("matrix_multiply", {
///     // expensive operation
///     42
/// });
/// assert_eq!(result, 42);
/// ```
///
/// With an explicit logger:
///
/// ```rust,no_run
/// use scirs2_core::scirs2_time;
/// use scirs2_core::logging::Logger;
///
/// let logger = Logger::new("my_module");
/// let result = scirs2_time!(logger, "decomposition", {
///     100
/// });
/// assert_eq!(result, 100);
/// ```
#[macro_export]
macro_rules! scirs2_time {
    ($label:expr, $body:expr) => {{
        let _start = std::time::Instant::now();
        let _result = $body;
        let _elapsed = _start.elapsed();
        // Use the log crate if available
        log::info!(
            target: "scirs2::timing",
            "[TIMING] {}: {:.6}s ({:.3}ms)",
            $label,
            _elapsed.as_secs_f64(),
            _elapsed.as_secs_f64() * 1000.0
        );
        _result
    }};
    ($logger:expr, $label:expr, $body:expr) => {{
        let _start = std::time::Instant::now();
        let _result = $body;
        let _elapsed = _start.elapsed();
        $logger.info(&format!(
            "[TIMING] {}: {:.6}s ({:.3}ms)",
            $label,
            _elapsed.as_secs_f64(),
            _elapsed.as_secs_f64() * 1000.0
        ));
        _result
    }};
}

// ---------------------------------------------------------------------------
// TimingGuard -- RAII-based timing
// ---------------------------------------------------------------------------

/// RAII guard that measures wall-clock time from construction to drop.
///
/// Useful for timing scopes without wrapping in a macro.
///
/// ```rust,no_run
/// use scirs2_core::logging::bridge::TimingGuard;
///
/// {
///     let _guard = TimingGuard::new("expensive_scope");
///     // ... work happens here ...
/// } // logs elapsed time on drop
/// ```
pub struct TimingGuard {
    label: String,
    start: Instant,
}

impl TimingGuard {
    /// Create a new timing guard with the given label.
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            start: Instant::now(),
        }
    }

    /// Elapsed time since construction.
    pub fn elapsed(&self) -> std::time::Duration {
        self.start.elapsed()
    }
}

impl Drop for TimingGuard {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        log::info!(
            target: "scirs2::timing",
            "[TIMING] {}: {:.6}s ({:.3}ms)",
            self.label,
            elapsed.as_secs_f64(),
            elapsed.as_secs_f64() * 1000.0
        );
    }
}

// ---------------------------------------------------------------------------
// MemoryTracker -- lightweight memory usage tracking
// ---------------------------------------------------------------------------

/// Thread-safe memory usage tracker.
///
/// Records allocations and deallocations so callers can inspect high-water-mark
/// and current usage without relying on a system allocator hook.
#[derive(Debug, Clone)]
pub struct MemoryTracker {
    inner: Arc<MemoryTrackerInner>,
}

#[derive(Debug)]
struct MemoryTrackerInner {
    /// Current live bytes (may temporarily go negative in racy scenarios, stored as i64 via AtomicU64)
    current_bytes: AtomicU64,
    /// Peak (high-water mark) bytes
    peak_bytes: AtomicU64,
    /// Total bytes ever allocated
    total_allocated: AtomicU64,
    /// Total bytes ever freed
    total_freed: AtomicU64,
    /// Number of allocation events
    allocation_count: AtomicUsize,
    /// Number of deallocation events
    deallocation_count: AtomicUsize,
}

/// A snapshot report of memory usage.
#[derive(Debug, Clone)]
pub struct MemoryReport {
    /// Current live bytes
    pub current_bytes: u64,
    /// Peak bytes seen
    pub peak_bytes: u64,
    /// Total allocated over lifetime
    pub total_allocated: u64,
    /// Total freed over lifetime
    pub total_freed: u64,
    /// Number of allocation events
    pub allocation_count: usize,
    /// Number of deallocation events
    pub deallocation_count: usize,
}

impl std::fmt::Display for MemoryReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MemoryReport {{")?;
        writeln!(
            f,
            "  current:   {} bytes ({:.2} MiB)",
            self.current_bytes,
            self.current_bytes as f64 / (1024.0 * 1024.0)
        )?;
        writeln!(
            f,
            "  peak:      {} bytes ({:.2} MiB)",
            self.peak_bytes,
            self.peak_bytes as f64 / (1024.0 * 1024.0)
        )?;
        writeln!(
            f,
            "  allocated: {} bytes ({}x)",
            self.total_allocated, self.allocation_count
        )?;
        writeln!(
            f,
            "  freed:     {} bytes ({}x)",
            self.total_freed, self.deallocation_count
        )?;
        write!(f, "}}")
    }
}

impl MemoryTracker {
    /// Create a new memory tracker with all counters at zero.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(MemoryTrackerInner {
                current_bytes: AtomicU64::new(0),
                peak_bytes: AtomicU64::new(0),
                total_allocated: AtomicU64::new(0),
                total_freed: AtomicU64::new(0),
                allocation_count: AtomicUsize::new(0),
                deallocation_count: AtomicUsize::new(0),
            }),
        }
    }

    /// Record an allocation of `bytes` bytes.
    pub fn record_allocation(&self, bytes: u64) {
        self.inner
            .total_allocated
            .fetch_add(bytes, Ordering::Relaxed);
        self.inner.allocation_count.fetch_add(1, Ordering::Relaxed);
        let new_current = self.inner.current_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
        // Update peak via CAS loop
        let mut peak = self.inner.peak_bytes.load(Ordering::Relaxed);
        while new_current > peak {
            match self.inner.peak_bytes.compare_exchange_weak(
                peak,
                new_current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => peak = actual,
            }
        }
    }

    /// Record a deallocation of `bytes` bytes.
    pub fn record_deallocation(&self, bytes: u64) {
        self.inner.total_freed.fetch_add(bytes, Ordering::Relaxed);
        self.inner
            .deallocation_count
            .fetch_add(1, Ordering::Relaxed);
        // Saturating subtract to avoid underflow
        let _ = self.inner.current_bytes.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |current| Some(current.saturating_sub(bytes)),
        );
    }

    /// Get the current live bytes.
    pub fn current_bytes(&self) -> u64 {
        self.inner.current_bytes.load(Ordering::Relaxed)
    }

    /// Get the peak (high-water mark) bytes.
    pub fn peak_bytes(&self) -> u64 {
        self.inner.peak_bytes.load(Ordering::Relaxed)
    }

    /// Take a snapshot report.
    pub fn report(&self) -> MemoryReport {
        MemoryReport {
            current_bytes: self.inner.current_bytes.load(Ordering::Relaxed),
            peak_bytes: self.inner.peak_bytes.load(Ordering::Relaxed),
            total_allocated: self.inner.total_allocated.load(Ordering::Relaxed),
            total_freed: self.inner.total_freed.load(Ordering::Relaxed),
            allocation_count: self.inner.allocation_count.load(Ordering::Relaxed),
            deallocation_count: self.inner.deallocation_count.load(Ordering::Relaxed),
        }
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.inner.current_bytes.store(0, Ordering::Relaxed);
        self.inner.peak_bytes.store(0, Ordering::Relaxed);
        self.inner.total_allocated.store(0, Ordering::Relaxed);
        self.inner.total_freed.store(0, Ordering::Relaxed);
        self.inner.allocation_count.store(0, Ordering::Relaxed);
        self.inner.deallocation_count.store(0, Ordering::Relaxed);
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_crate_bridge_level_mapping() {
        assert_eq!(
            LogCrateBridge::map_level(LogLevel::Trace),
            log::Level::Trace
        );
        assert_eq!(
            LogCrateBridge::map_level(LogLevel::Debug),
            log::Level::Debug
        );
        assert_eq!(LogCrateBridge::map_level(LogLevel::Info), log::Level::Info);
        assert_eq!(LogCrateBridge::map_level(LogLevel::Warn), log::Level::Warn);
        assert_eq!(
            LogCrateBridge::map_level(LogLevel::Error),
            log::Level::Error
        );
        assert_eq!(
            LogCrateBridge::map_level(LogLevel::Critical),
            log::Level::Error
        );
    }

    #[test]
    fn test_memory_tracker_allocation() {
        let tracker = MemoryTracker::new();
        assert_eq!(tracker.current_bytes(), 0);

        tracker.record_allocation(1024);
        assert_eq!(tracker.current_bytes(), 1024);
        assert_eq!(tracker.peak_bytes(), 1024);

        tracker.record_allocation(2048);
        assert_eq!(tracker.current_bytes(), 3072);
        assert_eq!(tracker.peak_bytes(), 3072);
    }

    #[test]
    fn test_memory_tracker_deallocation() {
        let tracker = MemoryTracker::new();
        tracker.record_allocation(1024);
        tracker.record_allocation(2048);

        tracker.record_deallocation(1024);
        assert_eq!(tracker.current_bytes(), 2048);
        // Peak should not decrease
        assert_eq!(tracker.peak_bytes(), 3072);
    }

    #[test]
    fn test_memory_tracker_deallocation_underflow_protection() {
        let tracker = MemoryTracker::new();
        tracker.record_allocation(100);
        tracker.record_deallocation(200); // More than allocated
        assert_eq!(tracker.current_bytes(), 0); // Should saturate at 0
    }

    #[test]
    fn test_memory_tracker_report() {
        let tracker = MemoryTracker::new();
        tracker.record_allocation(1000);
        tracker.record_allocation(2000);
        tracker.record_deallocation(500);

        let report = tracker.report();
        assert_eq!(report.current_bytes, 2500);
        assert_eq!(report.peak_bytes, 3000);
        assert_eq!(report.total_allocated, 3000);
        assert_eq!(report.total_freed, 500);
        assert_eq!(report.allocation_count, 2);
        assert_eq!(report.deallocation_count, 1);
    }

    #[test]
    fn test_memory_tracker_reset() {
        let tracker = MemoryTracker::new();
        tracker.record_allocation(5000);
        tracker.reset();

        let report = tracker.report();
        assert_eq!(report.current_bytes, 0);
        assert_eq!(report.peak_bytes, 0);
        assert_eq!(report.total_allocated, 0);
        assert_eq!(report.allocation_count, 0);
    }

    #[test]
    fn test_memory_tracker_clone_shares_state() {
        let tracker1 = MemoryTracker::new();
        let tracker2 = tracker1.clone();

        tracker1.record_allocation(1024);
        assert_eq!(tracker2.current_bytes(), 1024);
    }

    #[test]
    fn test_memory_report_display() {
        let tracker = MemoryTracker::new();
        tracker.record_allocation(1_048_576); // 1 MiB
        let report = tracker.report();
        let display = format!("{report}");
        assert!(display.contains("1.00 MiB"));
        assert!(display.contains("1048576"));
    }

    #[test]
    fn test_timing_guard_elapsed() {
        let guard = TimingGuard::new("test_op");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = guard.elapsed();
        assert!(elapsed.as_millis() >= 5, "Should have elapsed at least 5ms");
        // Drop will log, which is fine in tests
    }

    #[test]
    fn test_timing_guard_does_not_panic_on_drop() {
        {
            let _guard = TimingGuard::new("scope_test");
            // Work
            let _ = 1 + 1;
        }
        // If we reach here, drop did not panic
    }

    #[test]
    fn test_log_bridge_install_does_not_panic() {
        // Just ensure installing doesn't panic
        LogCrateBridge::install();
    }

    #[test]
    fn test_memory_tracker_thread_safety() {
        let tracker = MemoryTracker::new();
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let t = tracker.clone();
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        t.record_allocation(10);
                    }
                    for _ in 0..50 {
                        t.record_deallocation(10);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("Thread should not panic");
        }

        let report = tracker.report();
        assert_eq!(report.total_allocated, 4 * 100 * 10);
        assert_eq!(report.total_freed, 4 * 50 * 10);
        assert_eq!(report.allocation_count, 400);
        assert_eq!(report.deallocation_count, 200);
        assert_eq!(report.current_bytes, 4 * 50 * 10); // 4*100*10 - 4*50*10
    }
}
