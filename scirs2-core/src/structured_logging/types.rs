//! Core types for structured logging and OpenTelemetry-compatible tracing.
//!
//! Defines the fundamental data structures used across all sub-modules:
//! log levels, records, field values, span contexts, and configuration types.

use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// LogLevel
// ============================================================================

/// Severity level for a log record.
///
/// Ordered from least to most severe.  Variants are intentionally listed in
/// ascending severity order so that `as u8` comparisons work correctly.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogLevel {
    /// Highly detailed diagnostic information.
    Trace = 0,
    /// Diagnostic information useful during development.
    Debug = 1,
    /// Informational messages about normal operation.
    Info = 2,
    /// Potentially harmful situations that deserve attention.
    Warn = 3,
    /// Error events that might still allow the application to run.
    Error = 4,
    /// Very severe error events that will presumably lead the application to abort.
    Fatal = 5,
}

impl LogLevel {
    /// Return the canonical uppercase string representation used in JSON output.
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Trace => "TRACE",
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERROR",
            LogLevel::Fatal => "FATAL",
        }
    }

    /// Map from OTLP severity number (1-24) to `LogLevel`.
    pub fn from_otlp_severity(sev: u32) -> Self {
        match sev {
            1..=4 => LogLevel::Trace,
            5..=8 => LogLevel::Debug,
            9..=12 => LogLevel::Info,
            13..=16 => LogLevel::Warn,
            17..=20 => LogLevel::Error,
            _ => LogLevel::Fatal,
        }
    }

    /// Return the OTLP severity number (lowest value for the band).
    pub fn to_otlp_severity(&self) -> u32 {
        match self {
            LogLevel::Trace => 1,
            LogLevel::Debug => 5,
            LogLevel::Info => 9,
            LogLevel::Warn => 13,
            LogLevel::Error => 17,
            LogLevel::Fatal => 21,
        }
    }
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ============================================================================
// FieldValue
// ============================================================================

/// A typed value that can be attached to a log record or span attribute.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum FieldValue {
    /// UTF-8 string.
    Str(String),
    /// Signed 64-bit integer.
    Int(i64),
    /// IEEE 754 double-precision float.
    Float(f64),
    /// Boolean flag.
    Bool(bool),
    /// Homogeneous array of values (recursive).
    Array(Vec<FieldValue>),
}

impl FieldValue {
    /// Return a short type label used in JSON serialisation.
    pub fn type_label(&self) -> &'static str {
        match self {
            FieldValue::Str(_) => "str",
            FieldValue::Int(_) => "int",
            FieldValue::Float(_) => "float",
            FieldValue::Bool(_) => "bool",
            FieldValue::Array(_) => "array",
        }
    }

    /// Render the value as a JSON-compatible string fragment (not quoted for
    /// container rendering — callers that need quoted output should wrap with
    /// `format!("\"{}\"", …)` if needed).
    pub fn to_json_value(&self) -> String {
        match self {
            FieldValue::Str(s) => {
                // Escape special JSON characters.
                let escaped = s
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('\n', "\\n")
                    .replace('\r', "\\r")
                    .replace('\t', "\\t");
                format!("\"{}\"", escaped)
            }
            FieldValue::Int(i) => i.to_string(),
            FieldValue::Float(f) => {
                if f.is_nan() {
                    "null".to_string()
                } else if f.is_infinite() {
                    if *f > 0.0 {
                        "1e308".to_string()
                    } else {
                        "-1e308".to_string()
                    }
                } else {
                    format!("{}", f)
                }
            }
            FieldValue::Bool(b) => {
                if *b {
                    "true".to_string()
                } else {
                    "false".to_string()
                }
            }
            FieldValue::Array(items) => {
                let inner: Vec<String> = items.iter().map(|v| v.to_json_value()).collect();
                format!("[{}]", inner.join(","))
            }
        }
    }
}

impl From<&str> for FieldValue {
    fn from(s: &str) -> Self {
        FieldValue::Str(s.to_owned())
    }
}
impl From<String> for FieldValue {
    fn from(s: String) -> Self {
        FieldValue::Str(s)
    }
}
impl From<i64> for FieldValue {
    fn from(i: i64) -> Self {
        FieldValue::Int(i)
    }
}
impl From<f64> for FieldValue {
    fn from(f: f64) -> Self {
        FieldValue::Float(f)
    }
}
impl From<bool> for FieldValue {
    fn from(b: bool) -> Self {
        FieldValue::Bool(b)
    }
}

// ============================================================================
// LogRecord
// ============================================================================

/// A complete structured log record.
///
/// Records are immutable once created; mutations should produce a new record.
#[derive(Debug, Clone)]
pub struct LogRecord {
    /// Severity level.
    pub level: LogLevel,
    /// Human-readable message.
    pub message: String,
    /// Structured key-value pairs.
    pub fields: Vec<(String, FieldValue)>,
    /// Wall-clock time in nanoseconds since Unix epoch.
    pub timestamp_ns: u64,
    /// Span ID from the enclosing tracing span, if any.
    pub span_id: Option<u64>,
    /// Trace ID from the enclosing trace, if any.
    pub trace_id: Option<u64>,
}

impl LogRecord {
    /// Construct a record stamped with the current wall-clock time.
    pub fn new(level: LogLevel, message: impl Into<String>) -> Self {
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        Self {
            level,
            message: message.into(),
            fields: Vec::new(),
            timestamp_ns,
            span_id: None,
            trace_id: None,
        }
    }

    /// Attach a single key-value field and return `self` for chaining.
    pub fn with_field(mut self, key: impl Into<String>, value: FieldValue) -> Self {
        self.fields.push((key.into(), value));
        self
    }

    /// Attach multiple key-value fields.
    pub fn with_fields(mut self, fields: Vec<(String, FieldValue)>) -> Self {
        self.fields.extend(fields);
        self
    }

    /// Attach span/trace context.
    pub fn with_span_context(mut self, trace_id: u64, span_id: u64) -> Self {
        self.trace_id = Some(trace_id);
        self.span_id = Some(span_id);
        self
    }
}

// ============================================================================
// SpanContext
// ============================================================================

/// Identifies a specific span within a distributed trace.
///
/// Immutable; cloning is intentionally cheap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpanContext {
    /// 64-bit unique identifier for the overall trace.
    pub trace_id: u64,
    /// 64-bit unique identifier for this specific span.
    pub span_id: u64,
    /// Parent span ID, absent for root spans.
    pub parent_id: Option<u64>,
    /// Arbitrary string key-value pairs propagated across service boundaries.
    pub baggage: Vec<(String, String)>,
}

impl SpanContext {
    /// Construct a root span context (no parent).
    pub fn root(trace_id: u64, span_id: u64) -> Self {
        Self {
            trace_id,
            span_id,
            parent_id: None,
            baggage: Vec::new(),
        }
    }

    /// Construct a child span context.
    pub fn child(parent: &SpanContext, span_id: u64) -> Self {
        Self {
            trace_id: parent.trace_id,
            span_id,
            parent_id: Some(parent.span_id),
            baggage: parent.baggage.clone(),
        }
    }
}

// ============================================================================
// TraceConfig
// ============================================================================

/// Configuration controlling distributed tracing behaviour.
#[derive(Debug, Clone)]
pub struct TraceConfig {
    /// Fraction of traces that are sampled (0.0 = none, 1.0 = all).
    pub sampling_rate: f64,
    /// Maximum number of spans kept in memory before dropping the oldest.
    pub max_spans: usize,
    /// Number of spans per export batch.
    pub export_batch_size: usize,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            sampling_rate: 1.0,
            max_spans: 10_000,
            export_batch_size: 512,
        }
    }
}

// ============================================================================
// OtelConfig
// ============================================================================

/// Configuration for the OpenTelemetry integration layer.
#[derive(Debug, Clone)]
pub struct OtelConfig {
    /// Logical service name embedded in every exported record.
    pub service_name: String,
    /// Arbitrary resource attributes (e.g. `[("host", "node-1")]`).
    pub resource_attrs: Vec<(String, String)>,
}

impl Default for OtelConfig {
    fn default() -> Self {
        Self {
            service_name: "scirs2".to_owned(),
            resource_attrs: Vec::new(),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Trace < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Error);
        assert!(LogLevel::Error < LogLevel::Fatal);
    }

    #[test]
    fn test_field_value_json() {
        assert_eq!(FieldValue::Int(42).to_json_value(), "42");
        assert_eq!(FieldValue::Bool(true).to_json_value(), "true");
        assert_eq!(FieldValue::Str("hi".into()).to_json_value(), "\"hi\"");
        assert_eq!(
            FieldValue::Array(vec![FieldValue::Int(1), FieldValue::Int(2)]).to_json_value(),
            "[1,2]"
        );
    }

    #[test]
    fn test_span_context_child() {
        let root = SpanContext::root(100, 1);
        let child = SpanContext::child(&root, 2);
        assert_eq!(child.trace_id, 100);
        assert_eq!(child.parent_id, Some(1));
    }

    #[test]
    fn test_trace_config_default() {
        let cfg = TraceConfig::default();
        assert!((cfg.sampling_rate - 1.0).abs() < f64::EPSILON);
        assert_eq!(cfg.export_batch_size, 512);
    }

    #[test]
    fn test_otel_config_default() {
        let cfg = OtelConfig::default();
        assert_eq!(cfg.service_name, "scirs2");
        assert!(cfg.resource_attrs.is_empty());
    }
}
