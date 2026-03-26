//! Distributed tracing with an OpenTelemetry-compatible span model.
//!
//! # Design
//!
//! - **Span** — the fundamental unit of work, carrying a `SpanContext`.
//! - **Tracer** — creates spans, manages sampling, flushes to exporters.
//! - **Sampler** — decides per-trace whether to record.
//! - **SpanExporter** — receives completed spans for persistence / forwarding.
//! - **Context propagation** — W3C `traceparent` header inject/extract.
//!
//! No async runtime is required; all operations are synchronous.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use super::types::{FieldValue, SpanContext, TraceConfig};

// ============================================================================
// Internal ID generator (LCG, not cryptographically secure)
// ============================================================================

/// Simple linear-congruential generator seeded from the system clock.
struct LcgId {
    state: u64,
}

impl LcgId {
    fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(12345)
            ^ 0xdeadbeef_cafebabe;
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        // Knuth multiplicative hash (64-bit).
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        if self.state == 0 {
            self.state = 1;
        }
        self.state
    }
}

// Module-level generator protected by a mutex.
fn next_id() -> u64 {
    use std::cell::RefCell;
    thread_local! {
        static GEN: RefCell<LcgId> = RefCell::new(LcgId::new());
    }
    GEN.with(|g| g.borrow_mut().next())
}

// ============================================================================
// SpanStatus
// ============================================================================

/// The outcome of a span.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpanStatus {
    /// No explicit status set by the caller.
    Unset,
    /// The operation completed successfully.
    Ok,
    /// The operation failed; the inner string contains a description.
    Error(String),
}

impl Default for SpanStatus {
    fn default() -> Self {
        SpanStatus::Unset
    }
}

// ============================================================================
// SpanEvent
// ============================================================================

/// A timestamped annotation recorded inside a span.
#[derive(Debug, Clone)]
pub struct SpanEvent {
    /// Name of the event (e.g. `"retry"`, `"cache.miss"`).
    pub name: String,
    /// Wall-clock time in nanoseconds since Unix epoch.
    pub timestamp_ns: u64,
    /// Arbitrary attributes attached to the event.
    pub attributes: Vec<(String, FieldValue)>,
}

impl SpanEvent {
    fn new(name: impl Into<String>, attributes: Vec<(String, FieldValue)>) -> Self {
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        Self {
            name: name.into(),
            timestamp_ns,
            attributes,
        }
    }
}

// ============================================================================
// Span
// ============================================================================

/// A single unit of work in a distributed trace.
///
/// Call `end()` when the work is complete; spans that are never ended are
/// considered still-in-flight and will be dropped without export.
#[derive(Debug, Clone)]
pub struct Span {
    /// Identifies this span within the trace hierarchy.
    pub context: SpanContext,
    /// Human-readable operation name.
    pub name: String,
    /// Wall-clock start time in nanoseconds since Unix epoch.
    pub start_ns: u64,
    /// Wall-clock end time; `None` while the span is in-flight.
    pub end_ns: Option<u64>,
    /// Timestamped annotations recorded during the span's lifetime.
    pub events: Vec<SpanEvent>,
    /// Key-value attributes describing this span.
    pub attributes: Vec<(String, FieldValue)>,
    /// Outcome of the span.
    pub status: SpanStatus,
    /// Whether this span was sampled (i.e. should be exported).
    pub(crate) sampled: bool,
}

impl Span {
    /// Set an attribute, replacing any existing entry with the same key.
    pub fn set_attribute(&mut self, key: impl Into<String>, value: FieldValue) {
        let key = key.into();
        if let Some(entry) = self.attributes.iter_mut().find(|(k, _)| k == &key) {
            entry.1 = value;
        } else {
            self.attributes.push((key, value));
        }
    }

    /// Record an event inside this span.
    pub fn add_event(&mut self, name: impl Into<String>, attrs: Vec<(String, FieldValue)>) {
        self.events.push(SpanEvent::new(name, attrs));
    }

    /// Set the outcome status of this span.
    pub fn set_status(&mut self, status: SpanStatus) {
        self.status = status;
    }

    /// Mark the span as ended with the current wall-clock time.
    pub fn end(&mut self) {
        self.end_ns = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
        );
    }

    /// Return the elapsed wall-clock duration in nanoseconds, or `None` if
    /// the span has not been ended.
    pub fn duration_ns(&self) -> Option<u64> {
        self.end_ns.map(|e| e.saturating_sub(self.start_ns))
    }
}

// ============================================================================
// Sampler trait + implementations
// ============================================================================

/// Determines whether a trace should be recorded.
pub trait Sampler: Send + Sync {
    /// Return `true` if the span identified by `trace_id` / `name` should be
    /// recorded.
    fn should_sample(&self, trace_id: u64, name: &str) -> bool;
}

/// Always records every span.
pub struct AlwaysOnSampler;

impl Sampler for AlwaysOnSampler {
    fn should_sample(&self, _trace_id: u64, _name: &str) -> bool {
        true
    }
}

/// Never records any span.
pub struct AlwaysOffSampler;

impl Sampler for AlwaysOffSampler {
    fn should_sample(&self, _trace_id: u64, _name: &str) -> bool {
        false
    }
}

/// Records a fraction of traces determined by `ratio` (0.0 – 1.0).
///
/// Sampling is deterministic per `trace_id`: the same trace always produces
/// the same decision, which is critical for head-based sampling correctness.
pub struct TraceIdRatioSampler {
    /// Fraction of traces to sample.
    ratio: f64,
}

impl TraceIdRatioSampler {
    /// Create a sampler that records `ratio` fraction of traces.
    ///
    /// `ratio` is clamped to [0.0, 1.0].
    pub fn new(ratio: f64) -> Self {
        Self {
            ratio: ratio.clamp(0.0, 1.0),
        }
    }
}

impl Sampler for TraceIdRatioSampler {
    fn should_sample(&self, trace_id: u64, _name: &str) -> bool {
        if self.ratio >= 1.0 {
            return true;
        }
        if self.ratio <= 0.0 {
            return false;
        }
        // Apply a mixing function (Murmur3 finaliser) to spread small or
        // sequential IDs across the full 64-bit range before sampling.
        let mut h = trace_id;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        h ^= h >> 33;
        // Map to [0.0, 1.0) and compare with ratio.
        let normalised = (h as f64) / (u64::MAX as f64 + 1.0);
        normalised < self.ratio
    }
}

/// Follows the parent's sampling decision; uses `root_sampler` for root spans.
pub struct ParentBasedSampler {
    root_sampler: Box<dyn Sampler>,
}

impl ParentBasedSampler {
    /// Create a `ParentBasedSampler` with the given root sampler.
    pub fn new(root_sampler: Box<dyn Sampler>) -> Self {
        Self { root_sampler }
    }
}

impl Sampler for ParentBasedSampler {
    fn should_sample(&self, trace_id: u64, name: &str) -> bool {
        // Without a parent context we fall back to the root sampler.
        self.root_sampler.should_sample(trace_id, name)
    }
}

// Helper to test parent-based decisions with an explicit parent flag.
impl ParentBasedSampler {
    /// Honour an explicit parent decision if supplied; otherwise delegate to
    /// the root sampler.
    pub fn should_sample_with_parent(
        &self,
        parent_sampled: Option<bool>,
        trace_id: u64,
        name: &str,
    ) -> bool {
        match parent_sampled {
            Some(decision) => decision,
            None => self.root_sampler.should_sample(trace_id, name),
        }
    }
}

// ============================================================================
// SpanExporter trait + implementations
// ============================================================================

/// Receives completed spans for export (file, network, memory, etc.).
pub trait SpanExporter: Send + Sync {
    /// Export a batch of completed spans.
    fn export(&self, spans: Vec<Span>);
}

/// Stores completed spans in memory — useful for tests.
pub struct InMemoryExporter {
    spans: Mutex<Vec<Span>>,
}

impl InMemoryExporter {
    /// Create an empty `InMemoryExporter`.
    pub fn new() -> Self {
        Self {
            spans: Mutex::new(Vec::new()),
        }
    }

    /// Return a copy of all exported spans.
    pub fn spans(&self) -> Vec<Span> {
        self.spans.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Clear exported spans.
    pub fn clear(&self) {
        if let Ok(mut g) = self.spans.lock() {
            g.clear();
        }
    }
}

impl Default for InMemoryExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl SpanExporter for InMemoryExporter {
    fn export(&self, spans: Vec<Span>) {
        if let Ok(mut g) = self.spans.lock() {
            g.extend(spans);
        }
    }
}

// ============================================================================
// OtlpSpan / OtlpStubExporter
// ============================================================================

/// An OTLP-compatible span as a plain struct (no network I/O).
///
/// Field names follow the OpenTelemetry Trace specification.
#[derive(Debug, Clone)]
pub struct OtlpSpan {
    /// 16-byte hex string representing the trace ID.
    pub trace_id: String,
    /// 8-byte hex string representing the span ID.
    pub span_id: String,
    /// 8-byte hex string representing the parent span ID, empty for roots.
    pub parent_span_id: String,
    /// Human-readable operation name.
    pub name: String,
    /// Start time in nanoseconds since Unix epoch.
    pub start_time_unix_nano: u64,
    /// End time in nanoseconds since Unix epoch.
    pub end_time_unix_nano: u64,
    /// Key-value attributes as JSON-encoded string pairs.
    pub attributes: Vec<(String, String)>,
    /// Span status code as string (`"UNSET"`, `"OK"`, `"ERROR"`).
    pub status_code: String,
    /// Status message (non-empty only for `"ERROR"`).
    pub status_message: String,
}

/// Serialises spans to OTLP format and retains them in memory.
pub struct OtlpStubExporter {
    spans: Mutex<Vec<OtlpSpan>>,
}

impl OtlpStubExporter {
    /// Create an empty exporter.
    pub fn new() -> Self {
        Self {
            spans: Mutex::new(Vec::new()),
        }
    }

    /// Convert a `Span` to `OtlpSpan`.
    pub fn to_otlp(span: &Span) -> OtlpSpan {
        let trace_hex = format!("{:016x}{:016x}", 0u64, span.context.trace_id);
        let span_hex = format!("{:016x}", span.context.span_id);
        let parent_hex = span
            .context
            .parent_id
            .map(|p| format!("{:016x}", p))
            .unwrap_or_default();

        let attributes: Vec<(String, String)> = span
            .attributes
            .iter()
            .map(|(k, v)| (k.clone(), v.to_json_value()))
            .collect();

        let (status_code, status_message) = match &span.status {
            SpanStatus::Unset => ("UNSET".to_owned(), String::new()),
            SpanStatus::Ok => ("OK".to_owned(), String::new()),
            SpanStatus::Error(msg) => ("ERROR".to_owned(), msg.clone()),
        };

        OtlpSpan {
            trace_id: trace_hex,
            span_id: span_hex,
            parent_span_id: parent_hex,
            name: span.name.clone(),
            start_time_unix_nano: span.start_ns,
            end_time_unix_nano: span.end_ns.unwrap_or(span.start_ns),
            attributes,
            status_code,
            status_message,
        }
    }

    /// Return all OTLP-formatted spans.
    pub fn otlp_spans(&self) -> Vec<OtlpSpan> {
        self.spans.lock().map(|g| g.clone()).unwrap_or_default()
    }
}

impl Default for OtlpStubExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl SpanExporter for OtlpStubExporter {
    fn export(&self, spans: Vec<Span>) {
        let otlp: Vec<OtlpSpan> = spans.iter().map(Self::to_otlp).collect();
        if let Ok(mut g) = self.spans.lock() {
            g.extend(otlp);
        }
    }
}

// ============================================================================
// Context propagation (W3C traceparent)
// ============================================================================

/// W3C Trace Context propagation.
///
/// Specification: <https://www.w3.org/TR/trace-context/>
pub struct TraceContext;

impl TraceContext {
    /// Inject a W3C `traceparent` header into `headers`.
    ///
    /// Format: `00-{32-hex-trace-id}-{16-hex-span-id}-{flags}`
    pub fn inject(context: &SpanContext, headers: &mut Vec<(String, String)>) {
        // Trace ID is stored as u64; expand to 128-bit (pad with zeros).
        let traceparent = format!(
            "00-{:016x}{:016x}-{:016x}-01",
            0u64, context.trace_id, context.span_id
        );
        // Remove any existing traceparent header.
        headers.retain(|(k, _)| k.to_lowercase() != "traceparent");
        headers.push(("traceparent".to_owned(), traceparent));

        // Baggage header.
        if !context.baggage.is_empty() {
            let baggage: Vec<String> = context
                .baggage
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            headers.retain(|(k, _)| k.to_lowercase() != "baggage");
            headers.push(("baggage".to_owned(), baggage.join(",")));
        }
    }

    /// Extract a `SpanContext` from a `traceparent` header value.
    ///
    /// Returns `None` if the header is absent or malformed.
    pub fn extract(headers: &[(String, String)]) -> Option<SpanContext> {
        let traceparent = headers
            .iter()
            .find(|(k, _)| k.to_lowercase() == "traceparent")
            .map(|(_, v)| v.as_str())?;

        // Expected: 00-{32 hex}-{16 hex}-{2 hex}
        let parts: Vec<&str> = traceparent.split('-').collect();
        if parts.len() != 4 {
            return None;
        }
        // Parse trace ID: take the lower 16 hex digits (our u64).
        let trace_hex = parts[1];
        if trace_hex.len() != 32 {
            return None;
        }
        let trace_id = u64::from_str_radix(&trace_hex[16..], 16).ok()?;

        let span_hex = parts[2];
        if span_hex.len() != 16 {
            return None;
        }
        let span_id = u64::from_str_radix(span_hex, 16).ok()?;

        Some(SpanContext {
            trace_id,
            span_id,
            parent_id: None,
            baggage: Vec::new(),
        })
    }
}

// ============================================================================
// Tracer
// ============================================================================

/// Creates and manages distributed tracing spans.
pub struct Tracer {
    config: TraceConfig,
    sampler: Box<dyn Sampler>,
    exporter: Arc<dyn SpanExporter>,
    /// In-flight spans indexed by span_id.
    active_spans: Arc<Mutex<HashMap<u64, Span>>>,
}

impl Tracer {
    /// Create a tracer with the given configuration and exporter.
    pub fn new(config: TraceConfig, exporter: Arc<dyn SpanExporter>) -> Self {
        Self {
            sampler: Box::new(AlwaysOnSampler),
            config,
            exporter,
            active_spans: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Replace the sampler.
    pub fn with_sampler(mut self, sampler: Box<dyn Sampler>) -> Self {
        self.sampler = sampler;
        self
    }

    /// Start a new span, optionally as a child of `parent`.
    pub fn start_span(&self, name: &str, parent: Option<&SpanContext>) -> Span {
        let (trace_id, parent_id) = match parent {
            Some(p) => (p.trace_id, Some(p.span_id)),
            None => (next_id(), None),
        };
        let span_id = next_id();

        let sampled = self.sampler.should_sample(trace_id, name);

        let context = SpanContext {
            trace_id,
            span_id,
            parent_id,
            baggage: parent.map(|p| p.baggage.clone()).unwrap_or_default(),
        };

        let start_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        let span = Span {
            context,
            name: name.to_owned(),
            start_ns,
            end_ns: None,
            events: Vec::new(),
            attributes: Vec::new(),
            status: SpanStatus::default(),
            sampled,
        };

        if sampled {
            if let Ok(mut active) = self.active_spans.lock() {
                // Cap to configured maximum.
                if active.len() >= self.config.max_spans {
                    // Drop the oldest span by start_ns.
                    if let Some(&oldest_id) = active
                        .iter()
                        .min_by_key(|(_, s)| s.start_ns)
                        .map(|(id, _)| id)
                    {
                        active.remove(&oldest_id);
                    }
                }
                active.insert(span_id, span.clone());
            }
        }

        span
    }

    /// Finalise a span and export it if it was sampled.
    pub fn finish_span(&self, mut span: Span) {
        if span.end_ns.is_none() {
            span.end();
        }
        if !span.sampled {
            return;
        }
        // Remove from active map.
        if let Ok(mut active) = self.active_spans.lock() {
            active.remove(&span.context.span_id);
        }
        self.exporter.export(vec![span]);
    }

    /// Execute `f` within a new span that is automatically ended on return.
    pub fn with_span<F, R>(&self, name: &str, f: F) -> R
    where
        F: FnOnce(&mut Span) -> R,
    {
        let mut span = self.start_span(name, None);
        let result = f(&mut span);
        self.finish_span(span);
        result
    }

    /// Execute `f` as a child of `parent`, automatically ending the span on return.
    pub fn with_child_span<F, R>(&self, name: &str, parent: &SpanContext, f: F) -> R
    where
        F: FnOnce(&mut Span) -> R,
    {
        let mut span = self.start_span(name, Some(parent));
        let result = f(&mut span);
        self.finish_span(span);
        result
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tracer() -> (Tracer, Arc<InMemoryExporter>) {
        let exporter = Arc::new(InMemoryExporter::new());
        let tracer = Tracer::new(
            TraceConfig::default(),
            Arc::clone(&exporter) as Arc<dyn SpanExporter>,
        );
        (tracer, exporter)
    }

    #[test]
    fn test_span_lifecycle() {
        let (tracer, exporter) = make_tracer();
        tracer.with_span("op", |span| {
            span.add_event("retry", vec![]);
            span.set_attribute("host", FieldValue::Str("localhost".into()));
        });
        let spans = exporter.spans();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].name, "op");
        assert_eq!(spans[0].events.len(), 1);
        assert_eq!(spans[0].attributes.len(), 1);
        assert!(spans[0].end_ns.is_some());
    }

    #[test]
    fn test_span_parent_child() {
        let (tracer, exporter) = make_tracer();
        let parent = tracer.start_span("parent", None);
        let parent_ctx = parent.context.clone();
        tracer.finish_span(parent);

        let child = tracer.start_span("child", Some(&parent_ctx));
        assert_eq!(child.context.trace_id, parent_ctx.trace_id);
        assert_eq!(child.context.parent_id, Some(parent_ctx.span_id));
        tracer.finish_span(child);

        let spans = exporter.spans();
        assert_eq!(spans.len(), 2);
    }

    #[test]
    fn test_tracer_with_span() {
        let (tracer, exporter) = make_tracer();
        let val = tracer.with_span("compute", |_span| 42);
        assert_eq!(val, 42);
        assert_eq!(exporter.spans().len(), 1);
    }

    #[test]
    fn test_in_memory_exporter() {
        let exporter = InMemoryExporter::new();
        let span = Span {
            context: SpanContext::root(1, 2),
            name: "test".into(),
            start_ns: 0,
            end_ns: Some(100),
            events: vec![],
            attributes: vec![],
            status: SpanStatus::Ok,
            sampled: true,
        };
        exporter.export(vec![span]);
        assert_eq!(exporter.spans().len(), 1);
    }

    #[test]
    fn test_always_on_sampler() {
        let s = AlwaysOnSampler;
        for i in 0..100u64 {
            assert!(s.should_sample(i, "op"));
        }
    }

    #[test]
    fn test_ratio_sampler_approx() {
        let s = TraceIdRatioSampler::new(0.5);
        let sampled = (0u64..10_000)
            .filter(|&id| s.should_sample(id, "op"))
            .count();
        // Expect roughly 5000 ± 10%
        assert!(sampled > 4000, "sampled={}", sampled);
        assert!(sampled < 6000, "sampled={}", sampled);
    }

    #[test]
    fn test_parent_based_sampler_follows() {
        let s = ParentBasedSampler::new(Box::new(AlwaysOffSampler));
        // When parent says true, follow it.
        assert!(s.should_sample_with_parent(Some(true), 0, "op"));
        // When parent says false, follow it.
        assert!(!s.should_sample_with_parent(Some(false), 0, "op"));
        // With no parent, fall back to AlwaysOffSampler.
        assert!(!s.should_sample_with_parent(None, 0, "op"));
    }

    #[test]
    fn test_traceparent_inject() {
        let ctx = SpanContext::root(0xdeadbeef, 0xcafe);
        let mut headers = Vec::new();
        TraceContext::inject(&ctx, &mut headers);
        let tp = headers
            .iter()
            .find(|(k, _)| k == "traceparent")
            .map(|(_, v)| v.as_str())
            .expect("traceparent header missing");
        // Format: 00-{32hex}-{16hex}-01
        assert!(tp.starts_with("00-"));
        assert!(tp.ends_with("-01"));
        let parts: Vec<&str> = tp.split('-').collect();
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[1].len(), 32);
        assert_eq!(parts[2].len(), 16);
    }

    #[test]
    fn test_traceparent_extract() {
        let ctx = SpanContext::root(0xdeadbeef, 0xcafe);
        let mut headers = Vec::new();
        TraceContext::inject(&ctx, &mut headers);
        let extracted = TraceContext::extract(&headers).expect("extraction failed");
        assert_eq!(extracted.trace_id, ctx.trace_id);
        assert_eq!(extracted.span_id, ctx.span_id);
    }

    #[test]
    fn test_span_status_error() {
        let (tracer, exporter) = make_tracer();
        tracer.with_span("failing", |span| {
            span.set_status(SpanStatus::Error("timeout".into()));
        });
        let spans = exporter.spans();
        assert_eq!(spans.len(), 1);
        match &spans[0].status {
            SpanStatus::Error(msg) => assert_eq!(msg, "timeout"),
            other => panic!("expected Error, got {:?}", other),
        }
    }
}
