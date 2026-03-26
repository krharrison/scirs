//! # Structured Logging and OpenTelemetry-Compatible Tracing
//!
//! This module provides an OpenTelemetry 0.30+-compatible structured logging
//! and distributed tracing stack implemented entirely in pure Rust — no C
//! bindings, no async runtime requirements, no network I/O.
//!
//! ## Sub-modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`types`] | Core data types: `LogLevel`, `LogRecord`, `FieldValue`, `SpanContext`, … |
//! | [`logger`] | Structured logger with pluggable sinks (`ConsoleSink`, `MemorySink`, `OtelLogSink`) |
//! | [`tracer`] | Distributed tracing: spans, samplers, exporters, W3C propagation |
//! | [`metrics`] | OTel-compatible counter / gauge / histogram instruments + registry |
//!
//! ## Quick Start
//!
//! ```rust
//! use scirs2_core::structured_logging::{
//!     types::{LogLevel, FieldValue},
//!     logger::{StructuredLogger, MemorySink},
//!     tracer::{Tracer, InMemoryExporter, SpanStatus},
//!     metrics::MeterRegistry,
//! };
//! use std::sync::Arc;
//!
//! // -- Logger --
//! let logger = Arc::new(StructuredLogger::new(
//!     LogLevel::Info,
//!     vec![Box::new(MemorySink::new())],
//! ));
//! logger.info("service started");
//!
//! // -- Tracer --
//! let exporter = Arc::new(InMemoryExporter::new());
//! let tracer = Tracer::new(
//!     Default::default(),
//!     Arc::clone(&exporter) as Arc<dyn scirs2_core::structured_logging::tracer::SpanExporter>,
//! );
//! tracer.with_span("process_batch", |span| {
//!     span.set_attribute("batch_size", FieldValue::Int(100));
//! });
//!
//! // -- Metrics --
//! let registry = MeterRegistry::new();
//! let req_counter = registry.counter("http_requests", &[("method", "POST")]);
//! req_counter.add(1);
//! ```

pub mod logger;
pub mod metrics;
pub mod tracer;
pub mod types;

// Convenience re-exports for the most commonly needed types.
pub use logger::{
    ConsoleSink, LogBuilder, LogSink, MemorySink, OtelLogSink, OtlpLogRecord, StructuredLogger,
};
pub use metrics::{Counter, Gauge, Histogram, MeterRegistry};
pub use tracer::{
    AlwaysOnSampler, InMemoryExporter, OtlpSpan, OtlpStubExporter, ParentBasedSampler, Sampler,
    Span, SpanEvent, SpanExporter, SpanStatus, TraceContext, TraceIdRatioSampler, Tracer,
};
pub use types::{FieldValue, LogLevel, LogRecord, OtelConfig, SpanContext, TraceConfig};
