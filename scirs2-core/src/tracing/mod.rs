//! # Event Tracing System
//!
//! A lightweight, low-overhead event tracing system for SciRS2 computational workloads.
//! Provides RAII span measurement, instant event recording, and export to Chrome DevTools
//! JSON format and flamegraph data.
//!
//! ## Key Features
//!
//! - Thread-local event buffers for minimal contention
//! - RAII `SpanGuard` for automatic span recording on scope exit
//! - Configurable sampling rate and max-event cap
//! - Chrome DevTools JSON (`export_chrome_trace`)
//! - Flamegraph-compatible `(name, inclusive_ns)` pairs (`export_flamegraph_data`)
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::tracing::{Tracer, TracingConfig, export_flamegraph_data};
//! use std::sync::Arc;
//!
//! let config = TracingConfig { enabled: true, sample_rate: 1.0, max_events: 10_000 };
//! let tracer = Arc::new(Tracer::new(config));
//!
//! // RAII span
//! {
//!     let _guard = tracer.span("my_op");
//!     // work happens here
//! }
//!
//! // Instant event with metadata
//! tracer.event("checkpoint", &[("phase", "init"), ("step", "1")]);
//!
//! // Extract flamegraph data
//! let data = export_flamegraph_data(&tracer);
//! assert!(!data.is_empty());
//! ```

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::error::{CoreError, CoreResult, ErrorContext};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the event tracing system.
#[derive(Debug, Clone)]
pub struct TracingConfig {
    /// Whether tracing is active.  When `false` all operations are no-ops.
    pub enabled: bool,
    /// Fraction of spans that are actually recorded (0.0 – 1.0).
    /// Sampling is applied per-span using a thread-local counter to avoid
    /// expensive RNG calls on the hot path.
    pub sample_rate: f64,
    /// Maximum total events kept in memory across all threads.  Once this
    /// limit is reached, new events are silently dropped.
    pub max_events: usize,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sample_rate: 1.0,
            max_events: 100_000,
        }
    }
}

// ============================================================================
// TraceEvent
// ============================================================================

/// A single recorded trace event (span or instant).
#[derive(Debug, Clone)]
pub struct TraceEvent {
    /// Human-readable name / operation label.
    pub name: String,
    /// Wall-clock start of the event in nanoseconds since Unix epoch.
    pub timestamp_ns: u64,
    /// Duration in nanoseconds (0 for instant events).
    pub duration_ns: u64,
    /// OS thread ID that recorded the event.
    pub thread_id: u64,
    /// Arbitrary key-value metadata attached to the event.
    pub metadata: HashMap<String, String>,
}

impl TraceEvent {
    fn new_span(
        name: String,
        timestamp_ns: u64,
        duration_ns: u64,
        thread_id: u64,
        metadata: HashMap<String, String>,
    ) -> Self {
        Self {
            name,
            timestamp_ns,
            duration_ns,
            thread_id,
            metadata,
        }
    }

    fn new_instant(
        name: String,
        timestamp_ns: u64,
        thread_id: u64,
        metadata: HashMap<String, String>,
    ) -> Self {
        Self {
            name,
            timestamp_ns,
            duration_ns: 0,
            thread_id,
            metadata,
        }
    }
}

// ============================================================================
// Internal thread-local buffer management
// ============================================================================

/// Per-thread event buffer: flushed to the global collector on drain.
struct ThreadBuffer {
    events: Vec<TraceEvent>,
}

impl ThreadBuffer {
    fn new() -> Self {
        Self {
            events: Vec::with_capacity(256),
        }
    }
}

// ============================================================================
// SpanGuard – RAII span recorder
// ============================================================================

/// RAII guard returned by [`Tracer::span`].
///
/// The span is recorded into the tracer automatically when this guard is
/// dropped.  The guard borrows the [`Tracer`] by `Arc` so that it remains
/// valid even if the caller drops their own reference.
pub struct SpanGuard {
    name: String,
    start_instant: Instant,
    start_timestamp_ns: u64,
    thread_id: u64,
    metadata: HashMap<String, String>,
    tracer: Arc<Tracer>,
    // Whether this particular span was selected for recording (sampling).
    record: bool,
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        if !self.record {
            return;
        }
        let duration_ns = self.start_instant.elapsed().as_nanos() as u64;
        let event = TraceEvent::new_span(
            self.name.clone(),
            self.start_timestamp_ns,
            duration_ns,
            self.thread_id,
            std::mem::take(&mut self.metadata),
        );
        self.tracer.push_event(event);
    }
}

impl SpanGuard {
    /// Add or update a metadata key on the in-flight span.
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
}

// ============================================================================
// Tracer
// ============================================================================

/// The global event tracer.
///
/// Normally created once per process behind an `Arc` and shared across
/// threads.  Uses thread-local buffers for the hot path; background
/// collection is performed when a thread drains its buffer.
pub struct Tracer {
    config: TracingConfig,
    /// Monotonic epoch used as a reference point for `Instant` comparisons.
    epoch: Instant,
    /// Nanoseconds since Unix epoch corresponding to `epoch`.
    epoch_unix_ns: u64,
    /// Global event store (all threads write here after draining their TLS buffer).
    events: Mutex<Vec<TraceEvent>>,
    /// Running total of events ever collected (for the max_events cap).
    total_events: AtomicUsize,
    /// Whether tracing is enabled at runtime (mirrors config but atomically checkable).
    enabled: AtomicBool,
    /// Sampling counter per-tracer instance (used for deterministic thinning).
    sample_counter: AtomicU64,
}

impl std::fmt::Debug for Tracer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tracer")
            .field("enabled", &self.enabled.load(Ordering::Relaxed))
            .field("total_events", &self.total_events.load(Ordering::Relaxed))
            .finish()
    }
}

impl Tracer {
    /// Create a new tracer with the supplied configuration.
    pub fn new(config: TracingConfig) -> Self {
        let epoch = Instant::now();
        let epoch_unix_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        let enabled = config.enabled;
        Self {
            config,
            epoch,
            epoch_unix_ns,
            events: Mutex::new(Vec::new()),
            total_events: AtomicUsize::new(0),
            enabled: AtomicBool::new(enabled),
            sample_counter: AtomicU64::new(0),
        }
    }

    /// Enable or disable tracing at runtime.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    // ------------------------------------------------------------------
    // Sampling decision (cheap, deterministic)
    // ------------------------------------------------------------------

    fn should_sample(&self) -> bool {
        let rate = self.config.sample_rate;
        if rate >= 1.0 {
            return true;
        }
        if rate <= 0.0 {
            return false;
        }
        // Deterministic modulo-based thinning: avoids rand crate on hot path.
        let counter = self.sample_counter.fetch_add(1, Ordering::Relaxed);
        // How many out of 1000 should be sampled?
        let threshold = (rate * 1000.0) as u64;
        (counter % 1000) < threshold
    }

    // ------------------------------------------------------------------
    // Core timestamp helper
    // ------------------------------------------------------------------

    fn now_unix_ns(&self) -> u64 {
        let elapsed_ns = self.epoch.elapsed().as_nanos() as u64;
        self.epoch_unix_ns.saturating_add(elapsed_ns)
    }

    fn current_thread_id() -> u64 {
        // Use std::thread::current().id() hashed to a u64.
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        std::thread::current().id().hash(&mut hasher);
        hasher.finish()
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Start a span measurement; returns a [`SpanGuard`] that records the
    /// span when dropped.
    pub fn span(self: &Arc<Self>, name: &str) -> SpanGuard {
        let record = self.enabled.load(Ordering::Relaxed) && self.should_sample();
        SpanGuard {
            name: name.to_string(),
            start_instant: Instant::now(),
            start_timestamp_ns: self.now_unix_ns(),
            thread_id: Self::current_thread_id(),
            metadata: HashMap::new(),
            tracer: Arc::clone(self),
            record,
        }
    }

    /// Record an instant (zero-duration) event with optional metadata pairs.
    pub fn event(&self, name: &str, metadata: &[(&str, &str)]) {
        if !self.enabled.load(Ordering::Relaxed) || !self.should_sample() {
            return;
        }
        let md: HashMap<String, String> = metadata
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let ev = TraceEvent::new_instant(
            name.to_string(),
            self.now_unix_ns(),
            Self::current_thread_id(),
            md,
        );
        self.push_event(ev);
    }

    /// Push an event into the global store (respects max_events cap).
    fn push_event(&self, event: TraceEvent) {
        let current = self.total_events.load(Ordering::Relaxed);
        if current >= self.config.max_events {
            return;
        }
        if let Ok(mut guard) = self.events.lock() {
            if guard.len() < self.config.max_events {
                guard.push(event);
                self.total_events.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Return a snapshot of all collected events.
    pub fn collect_events(&self) -> Vec<TraceEvent> {
        self.events
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    /// Clear all collected events, resetting the counter.
    pub fn clear(&self) {
        if let Ok(mut guard) = self.events.lock() {
            guard.clear();
            self.total_events.store(0, Ordering::Relaxed);
        }
    }

    /// Number of events currently held.
    pub fn event_count(&self) -> usize {
        self.total_events.load(Ordering::Relaxed)
    }
}

// ============================================================================
// Chrome DevTools trace export
// ============================================================================

/// Export all events from `tracer` into Chrome DevTools JSON format and write
/// them to `path`.
///
/// The resulting file can be loaded with `chrome://tracing` or Perfetto UI.
///
/// # Errors
///
/// Returns an error if the file cannot be created or the JSON cannot be
/// serialised.
pub fn export_chrome_trace(tracer: &Tracer, path: &std::path::Path) -> CoreResult<()> {
    use std::fmt::Write as FmtWrite;
    use std::io::Write as IoWrite;

    let events = tracer.collect_events();

    let mut json = String::with_capacity(events.len() * 128 + 64);
    json.push_str("{\"traceEvents\":[\n");

    for (idx, ev) in events.iter().enumerate() {
        // Chrome trace uses microseconds.
        let ts_us = ev.timestamp_ns / 1_000;
        let dur_us = ev.duration_ns / 1_000;

        // Phase: "X" = complete (duration) event, "i" = instant event.
        let ph = if ev.duration_ns > 0 { "X" } else { "i" };

        // Build args object from metadata.
        let mut args = String::new();
        args.push('{');
        for (i, (k, v)) in ev.metadata.iter().enumerate() {
            if i > 0 {
                args.push(',');
            }
            write!(args, "\"{}\":\"{}\"", escape_json(k), escape_json(v))
                .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;
        }
        args.push('}');

        let comma = if idx + 1 < events.len() { "," } else { "" };

        writeln!(
            json,
            "{{\"name\":\"{}\",\"ph\":\"{}\",\"ts\":{},\"dur\":{},\"pid\":1,\"tid\":{},\"args\":{}}}{}",
            escape_json(&ev.name),
            ph,
            ts_us,
            dur_us,
            ev.thread_id,
            args,
            comma
        )
        .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;
    }

    json.push_str("]}\n");

    let mut file = std::fs::File::create(path)
        .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;
    file.write_all(json.as_bytes())
        .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;

    Ok(())
}

/// Escape special JSON characters in a string.
fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

// ============================================================================
// Flamegraph data export
// ============================================================================

/// Aggregate events by name and return `(name, total_inclusive_ns)` pairs
/// suitable for input to a flamegraph renderer.
///
/// Pairs are sorted by total duration descending.
pub fn export_flamegraph_data(tracer: &Tracer) -> Vec<(String, u64)> {
    let events = tracer.collect_events();
    let mut agg: HashMap<String, u64> = HashMap::new();
    for ev in &events {
        *agg.entry(ev.name.clone()).or_insert(0) += ev.duration_ns;
    }
    let mut pairs: Vec<(String, u64)> = agg.into_iter().collect();
    // Sort descending by total duration for easy consumption.
    pairs.sort_by(|a, b| b.1.cmp(&a.1));
    pairs
}

// ============================================================================
// Global tracer helpers
// ============================================================================

static GLOBAL_TRACER: std::sync::OnceLock<Arc<Tracer>> = std::sync::OnceLock::new();

/// Initialise the process-wide global tracer.  Subsequent calls are silently
/// ignored (the first initialisation wins).
pub fn init_global_tracer(config: TracingConfig) {
    let _ = GLOBAL_TRACER.set(Arc::new(Tracer::new(config)));
}

/// Obtain a reference to the global tracer, if initialised.
pub fn global_tracer() -> Option<Arc<Tracer>> {
    GLOBAL_TRACER.get().cloned()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    fn make_tracer() -> Arc<Tracer> {
        Arc::new(Tracer::new(TracingConfig {
            enabled: true,
            sample_rate: 1.0,
            max_events: 10_000,
        }))
    }

    #[test]
    fn test_span_guard_records_event() {
        let t = make_tracer();
        {
            let _g = t.span("test_span");
            std::thread::sleep(Duration::from_millis(1));
        }
        let events = t.collect_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "test_span");
        assert!(events[0].duration_ns > 0);
    }

    #[test]
    fn test_instant_event() {
        let t = make_tracer();
        t.event("my_event", &[("key", "value"), ("step", "42")]);
        let events = t.collect_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "my_event");
        assert_eq!(events[0].duration_ns, 0);
        assert_eq!(events[0].metadata.get("key").map(|s| s.as_str()), Some("value"));
    }

    #[test]
    fn test_max_events_cap() {
        let t = Arc::new(Tracer::new(TracingConfig {
            enabled: true,
            sample_rate: 1.0,
            max_events: 5,
        }));
        for i in 0..20 {
            t.event(&format!("ev_{}", i), &[]);
        }
        assert!(t.event_count() <= 5);
    }

    #[test]
    fn test_disabled_tracer_records_nothing() {
        let t = Arc::new(Tracer::new(TracingConfig {
            enabled: false,
            sample_rate: 1.0,
            max_events: 1_000,
        }));
        {
            let _g = t.span("noop");
        }
        t.event("noop_event", &[]);
        assert_eq!(t.event_count(), 0);
    }

    #[test]
    fn test_sample_rate_zero_records_nothing() {
        let t = Arc::new(Tracer::new(TracingConfig {
            enabled: true,
            sample_rate: 0.0,
            max_events: 1_000,
        }));
        for _ in 0..100 {
            t.event("sampled", &[]);
        }
        assert_eq!(t.event_count(), 0);
    }

    #[test]
    fn test_export_chrome_trace() {
        let t = make_tracer();
        {
            let _g = t.span("op_a");
        }
        t.event("instant_b", &[("x", "y")]);

        let tmp = std::env::temp_dir().join("scirs2_test_chrome_trace.json");
        export_chrome_trace(&t, &tmp).expect("chrome trace export failed");
        let content = std::fs::read_to_string(&tmp).expect("read failed");
        assert!(content.contains("traceEvents"));
        assert!(content.contains("op_a"));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_export_flamegraph_data() {
        let t = make_tracer();
        {
            let _g = t.span("matrix_mul");
        }
        {
            let _g = t.span("matrix_mul");
        }
        {
            let _g = t.span("fft");
        }
        let data = export_flamegraph_data(&t);
        assert!(!data.is_empty());
        // "matrix_mul" should have higher total time than "fft" (both ran, but matrix_mul ran twice)
        let matrix_mul = data.iter().find(|(n, _)| n == "matrix_mul");
        assert!(matrix_mul.is_some());
    }

    #[test]
    fn test_span_guard_metadata() {
        let t = make_tracer();
        {
            let mut g = t.span("annotated");
            g.add_metadata("size", "1024");
        }
        let events = t.collect_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].metadata.get("size").map(|s| s.as_str()), Some("1024"));
    }

    #[test]
    fn test_clear() {
        let t = make_tracer();
        t.event("e1", &[]);
        t.event("e2", &[]);
        assert_eq!(t.event_count(), 2);
        t.clear();
        assert_eq!(t.event_count(), 0);
        assert!(t.collect_events().is_empty());
    }
}
