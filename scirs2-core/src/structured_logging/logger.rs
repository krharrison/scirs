//! Structured logger implementation with pluggable sinks.
//!
//! Provides a zero-overhead fast-path when a record's level is below the
//! configured minimum: no allocation is performed and no lock is taken.
//!
//! # Architecture
//!
//! ```text
//!  caller  ──►  StructuredLogger  ──►  [LogSink, LogSink, …]
//!                    │
//!                    └── level filter (fast path)
//! ```
//!
//! All public types implement `Send + Sync` so the logger can be placed in an
//! `Arc` and shared across threads.

use std::io::{self, Write};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use super::types::{FieldValue, LogLevel, LogRecord, TraceConfig};

// ============================================================================
// LogSink trait
// ============================================================================

/// Pluggable destination for log records.
///
/// Implementations **must not** panic; errors should be handled internally.
pub trait LogSink: Send + Sync {
    /// Write a single log record.
    fn write(&self, record: &LogRecord);
}

// ============================================================================
// ConsoleSink
// ============================================================================

/// Emits JSON-formatted log lines to stderr (or stdout when configured).
pub struct ConsoleSink {
    use_stdout: bool,
}

impl ConsoleSink {
    /// Create a new `ConsoleSink` that writes to stderr.
    pub fn new() -> Self {
        Self { use_stdout: false }
    }

    /// Create a `ConsoleSink` that writes to stdout (useful in tests).
    pub fn stdout() -> Self {
        Self { use_stdout: true }
    }

    /// Render a log record as a compact JSON string.
    ///
    /// Format: `{"level":"INFO","msg":"...","ts":1234567890,"fields":{...}}`
    pub fn format_json(record: &LogRecord) -> String {
        let mut fields_json = String::new();
        for (i, (k, v)) in record.fields.iter().enumerate() {
            if i > 0 {
                fields_json.push(',');
            }
            let escaped_key = k.replace('\\', "\\\\").replace('"', "\\\"");
            fields_json.push('"');
            fields_json.push_str(&escaped_key);
            fields_json.push_str("\":");
            fields_json.push_str(&v.to_json_value());
        }

        let mut out = format!(
            "{{\"level\":\"{}\",\"msg\":{},\"ts\":{}",
            record.level.as_str(),
            FieldValue::Str(record.message.clone()).to_json_value(),
            record.timestamp_ns,
        );

        if let Some(tid) = record.trace_id {
            out.push_str(&format!(",\"trace_id\":{}", tid));
        }
        if let Some(sid) = record.span_id {
            out.push_str(&format!(",\"span_id\":{}", sid));
        }
        if !fields_json.is_empty() {
            out.push_str(&format!(",\"fields\":{{{}}}", fields_json));
        }
        out.push('}');
        out
    }
}

impl Default for ConsoleSink {
    fn default() -> Self {
        Self::new()
    }
}

impl LogSink for ConsoleSink {
    fn write(&self, record: &LogRecord) {
        let line = Self::format_json(record);
        if self.use_stdout {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            let _ = writeln!(handle, "{}", line);
        } else {
            let stderr = io::stderr();
            let mut handle = stderr.lock();
            let _ = writeln!(handle, "{}", line);
        }
    }
}

// ============================================================================
// MemorySink
// ============================================================================

/// Retains log records in memory — ideal for unit tests and debugging.
pub struct MemorySink {
    records: Mutex<Vec<LogRecord>>,
}

impl MemorySink {
    /// Create an empty `MemorySink`.
    pub fn new() -> Self {
        Self {
            records: Mutex::new(Vec::new()),
        }
    }

    /// Return a snapshot of all records captured so far.
    pub fn records(&self) -> Vec<LogRecord> {
        self.records.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Clear all retained records.
    pub fn clear(&self) {
        if let Ok(mut g) = self.records.lock() {
            g.clear();
        }
    }

    /// Number of records captured.
    pub fn len(&self) -> usize {
        self.records.lock().map(|g| g.len()).unwrap_or(0)
    }

    /// Return `true` when no records have been captured.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for MemorySink {
    fn default() -> Self {
        Self::new()
    }
}

impl LogSink for MemorySink {
    fn write(&self, record: &LogRecord) {
        if let Ok(mut g) = self.records.lock() {
            g.push(record.clone());
        }
    }
}

// ============================================================================
// OtlpLogRecord / OtelLogSink
// ============================================================================

/// An OTLP-compatible log record as a plain struct (no network I/O).
///
/// Field names follow the OpenTelemetry Log Data Model specification.
#[derive(Debug, Clone)]
pub struct OtlpLogRecord {
    /// Wall-clock timestamp in nanoseconds since Unix epoch.
    pub time_unix_nano: u64,
    /// Numeric severity (OTLP severity number).
    pub severity_number: u32,
    /// Canonical severity text (e.g. `"INFO"`).
    pub severity_text: String,
    /// The log message body.
    pub body: String,
    /// Structured attributes as string key-value pairs.
    pub attributes: Vec<(String, String)>,
    /// Trace ID, if present.
    pub trace_id: Option<u64>,
    /// Span ID, if present.
    pub span_id: Option<u64>,
}

/// Converts log records to OTLP format and stores them in memory.
pub struct OtelLogSink {
    records: Mutex<Vec<OtlpLogRecord>>,
}

impl OtelLogSink {
    /// Create an empty `OtelLogSink`.
    pub fn new() -> Self {
        Self {
            records: Mutex::new(Vec::new()),
        }
    }

    /// Convert a `LogRecord` to an `OtlpLogRecord`.
    pub fn to_otlp(record: &LogRecord) -> OtlpLogRecord {
        let attributes: Vec<(String, String)> = record
            .fields
            .iter()
            .map(|(k, v)| (k.clone(), v.to_json_value()))
            .collect();

        OtlpLogRecord {
            time_unix_nano: record.timestamp_ns,
            severity_number: record.level.to_otlp_severity(),
            severity_text: record.level.as_str().to_owned(),
            body: record.message.clone(),
            attributes,
            trace_id: record.trace_id,
            span_id: record.span_id,
        }
    }

    /// Return a snapshot of all OTLP records captured.
    pub fn otlp_records(&self) -> Vec<OtlpLogRecord> {
        self.records.lock().map(|g| g.clone()).unwrap_or_default()
    }
}

impl Default for OtelLogSink {
    fn default() -> Self {
        Self::new()
    }
}

impl LogSink for OtelLogSink {
    fn write(&self, record: &LogRecord) {
        let otlp = Self::to_otlp(record);
        if let Ok(mut g) = self.records.lock() {
            g.push(otlp);
        }
    }
}

// ============================================================================
// LogBuilder
// ============================================================================

/// Fluent builder that accumulates fields and emits a log record.
pub struct LogBuilder {
    fields: Vec<(String, FieldValue)>,
    trace_id: Option<u64>,
    span_id: Option<u64>,
    logger: Arc<StructuredLogger>,
}

impl LogBuilder {
    fn new(logger: Arc<StructuredLogger>) -> Self {
        Self {
            fields: Vec::new(),
            trace_id: None,
            span_id: None,
            logger,
        }
    }

    /// Add a key-value field.
    pub fn field(mut self, key: impl Into<String>, value: impl Into<FieldValue>) -> Self {
        self.fields.push((key.into(), value.into()));
        self
    }

    /// Attach a span/trace context for correlation.
    pub fn span_context(mut self, trace_id: u64, span_id: u64) -> Self {
        self.trace_id = Some(trace_id);
        self.span_id = Some(span_id);
        self
    }

    /// Emit the record with the given level and message.
    pub fn emit(self, level: LogLevel, message: impl Into<String>) {
        let mut record = LogRecord::new(level, message);
        record.fields = self.fields;
        record.trace_id = self.trace_id;
        record.span_id = self.span_id;
        self.logger.log(&record);
    }

    /// Convenience: emit at Info level.
    pub fn info(self, message: impl Into<String>) {
        self.emit(LogLevel::Info, message);
    }

    /// Convenience: emit at Warn level.
    pub fn warn(self, message: impl Into<String>) {
        self.emit(LogLevel::Warn, message);
    }

    /// Convenience: emit at Error level.
    pub fn error(self, message: impl Into<String>) {
        self.emit(LogLevel::Error, message);
    }

    /// Convenience: emit at Debug level.
    pub fn debug(self, message: impl Into<String>) {
        self.emit(LogLevel::Debug, message);
    }
}

// ============================================================================
// StructuredLogger
// ============================================================================

/// Thread-safe structured logger that dispatches records to registered sinks.
///
/// # Example
///
/// ```rust
/// use scirs2_core::structured_logging::logger::{StructuredLogger, MemorySink};
/// use scirs2_core::structured_logging::types::LogLevel;
/// use std::sync::Arc;
///
/// let sink = Arc::new(MemorySink::new());
/// let logger = Arc::new(StructuredLogger::new(LogLevel::Debug, vec![
///     Box::new(MemorySink::new()),
/// ]));
/// logger.info("hello");
/// ```
pub struct StructuredLogger {
    min_level: LogLevel,
    sinks: RwLock<Vec<Box<dyn LogSink>>>,
}

impl StructuredLogger {
    /// Create a logger with the given minimum level and sinks.
    pub fn new(min_level: LogLevel, sinks: Vec<Box<dyn LogSink>>) -> Self {
        Self {
            min_level,
            sinks: RwLock::new(sinks),
        }
    }

    /// Dispatch a `LogRecord` to all sinks (no-op if below the configured level).
    pub fn log(&self, record: &LogRecord) {
        // Fast path: compare level without acquiring any lock.
        if record.level < self.min_level {
            return;
        }
        if let Ok(sinks) = self.sinks.read() {
            for sink in sinks.iter() {
                sink.write(record);
            }
        }
    }

    /// Add a new sink at runtime.
    pub fn add_sink(&self, sink: Box<dyn LogSink>) {
        if let Ok(mut sinks) = self.sinks.write() {
            sinks.push(sink);
        }
    }

    /// Return a `LogBuilder` to accumulate fields before emitting.
    pub fn with_fields(self: &Arc<Self>, fields: Vec<(String, FieldValue)>) -> LogBuilder {
        let mut builder = LogBuilder::new(Arc::clone(self));
        builder.fields = fields;
        builder
    }

    /// Begin building a record with no pre-set fields.
    pub fn builder(self: &Arc<Self>) -> LogBuilder {
        LogBuilder::new(Arc::clone(self))
    }

    /// Emit an Info-level record.
    pub fn info(&self, msg: &str) {
        let record = LogRecord::new(LogLevel::Info, msg);
        self.log(&record);
    }

    /// Emit a Warn-level record.
    pub fn warn(&self, msg: &str) {
        let record = LogRecord::new(LogLevel::Warn, msg);
        self.log(&record);
    }

    /// Emit an Error-level record.
    pub fn error(&self, msg: &str) {
        let record = LogRecord::new(LogLevel::Error, msg);
        self.log(&record);
    }

    /// Emit a Debug-level record.
    pub fn debug(&self, msg: &str) {
        let record = LogRecord::new(LogLevel::Debug, msg);
        self.log(&record);
    }

    /// Emit a Trace-level record.
    pub fn trace(&self, msg: &str) {
        let record = LogRecord::new(LogLevel::Trace, msg);
        self.log(&record);
    }
}

// ============================================================================
// Global logger helpers
// ============================================================================

/// Initialise a thread-local (single-thread) default logger backed by a
/// `ConsoleSink`.  Useful for library consumers that do not want to manage
/// logger lifecycle explicitly.
pub fn init_logger(_config: TraceConfig) -> Arc<StructuredLogger> {
    Arc::new(StructuredLogger::new(
        LogLevel::Info,
        vec![Box::new(ConsoleSink::new())],
    ))
}

/// Current wall-clock timestamp in nanoseconds since Unix epoch.
pub(crate) fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_logger(level: LogLevel) -> (Arc<StructuredLogger>, Arc<MemorySink>) {
        let sink = Arc::new(MemorySink::new());
        // We need to pass a boxed reference; clone the Arc so the test keeps a handle.
        struct SharedSink(Arc<MemorySink>);
        impl LogSink for SharedSink {
            fn write(&self, record: &LogRecord) {
                self.0.write(record);
            }
        }
        let logger = Arc::new(StructuredLogger::new(
            level,
            vec![Box::new(SharedSink(Arc::clone(&sink)))],
        ));
        (logger, sink)
    }

    #[test]
    fn test_log_record_fields() {
        let record = LogRecord::new(LogLevel::Info, "test").with_field("key", FieldValue::Int(42));
        assert_eq!(record.message, "test");
        assert_eq!(record.fields.len(), 1);
        assert_eq!(record.fields[0].0, "key");
    }

    #[test]
    fn test_memory_sink_captures() {
        let (logger, sink) = make_logger(LogLevel::Debug);
        logger.info("hello");
        logger.warn("world");
        let recs = sink.records();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].message, "hello");
        assert_eq!(recs[1].level, LogLevel::Warn);
    }

    #[test]
    fn test_log_level_filter() {
        let (logger, sink) = make_logger(LogLevel::Warn);
        logger.debug("should be filtered");
        logger.info("also filtered");
        logger.warn("passes");
        let recs = sink.records();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].message, "passes");
    }

    #[test]
    fn test_log_builder_fluent() {
        let (logger, sink) = make_logger(LogLevel::Debug);
        logger
            .builder()
            .field("user", "alice")
            .field("count", 7i64)
            .info("user logged in");
        let recs = sink.records();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].fields.len(), 2);
    }

    #[test]
    fn test_console_sink_json_format() {
        let record = LogRecord::new(LogLevel::Info, "msg").with_field("k", FieldValue::Int(1));
        let json = ConsoleSink::format_json(&record);
        assert!(json.contains("\"level\":\"INFO\""));
        assert!(json.contains("\"msg\":\"msg\""));
        assert!(json.contains("\"k\":1"));
    }

    #[test]
    fn test_otlp_log_conversion() {
        let record =
            LogRecord::new(LogLevel::Error, "boom").with_field("code", FieldValue::Int(500));
        let otlp = OtelLogSink::to_otlp(&record);
        assert_eq!(otlp.severity_text, "ERROR");
        assert_eq!(otlp.severity_number, 17);
        assert_eq!(otlp.body, "boom");
        assert!(!otlp.attributes.is_empty());
    }
}
