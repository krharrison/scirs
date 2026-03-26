//! OpenTelemetry-compatible metrics: counters, gauges, histograms, and a
//! registry for named, labelled instruments.
//!
//! All types are thread-safe (backed by atomics or `Mutex`) and can be
//! shared across threads via `Arc`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

// ============================================================================
// Counter
// ============================================================================

/// A monotonically-increasing u64 counter.
///
/// Backed by an atomic so `add` is lock-free.
pub struct Counter {
    value: AtomicU64,
    name: String,
    labels: Vec<(String, String)>,
}

impl Counter {
    /// Create a counter with the given name and labels.
    pub fn new(name: impl Into<String>, labels: Vec<(String, String)>) -> Self {
        Self {
            value: AtomicU64::new(0),
            name: name.into(),
            labels,
        }
    }

    /// Add `value` to the counter.
    pub fn add(&self, value: u64) {
        self.value.fetch_add(value, Ordering::Relaxed);
    }

    /// Increment by 1 (convenience wrapper around `add`).
    pub fn inc(&self) {
        self.add(1);
    }

    /// Return the current value.
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Return the metric name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the label set.
    pub fn labels(&self) -> &[(String, String)] {
        &self.labels
    }
}

impl std::fmt::Debug for Counter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Counter")
            .field("name", &self.name)
            .field("value", &self.get())
            .finish()
    }
}

// ============================================================================
// Gauge
// ============================================================================

/// A gauge that records an instantaneous f64 measurement.
///
/// The value is stored as a bit-pattern inside a u64 atomic for lock-free
/// access.
pub struct Gauge {
    /// Stored as the IEEE 754 bit pattern of an f64.
    bits: AtomicU64,
    name: String,
}

impl Gauge {
    /// Create a gauge initialised to 0.0.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            bits: AtomicU64::new(0f64.to_bits()),
            name: name.into(),
        }
    }

    /// Set the gauge to `v`.
    pub fn set(&self, v: f64) {
        self.bits.store(v.to_bits(), Ordering::Relaxed);
    }

    /// Return the current value.
    pub fn get(&self) -> f64 {
        f64::from_bits(self.bits.load(Ordering::Relaxed))
    }

    /// Return the metric name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl std::fmt::Debug for Gauge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gauge")
            .field("name", &self.name)
            .field("value", &self.get())
            .finish()
    }
}

// ============================================================================
// Histogram
// ============================================================================

/// A histogram that distributes observations into configurable buckets.
///
/// Follows the Prometheus/OTel convention: each bucket counts observations
/// whose value is **≤ upper_bound** (i.e. cumulative).
///
/// A `+Inf` bucket is always added implicitly.
pub struct Histogram {
    /// Upper bounds of each bucket (sorted ascending, without +Inf).
    boundaries: Vec<f64>,
    /// Cumulative counts per bucket (`boundaries.len() + 1` entries, last =
    /// +Inf bucket).
    counts: Mutex<Vec<u64>>,
    /// Running sum of all recorded values.
    sum: Mutex<f64>,
    /// Total number of observations.
    count: AtomicU64,
    name: String,
    /// Raw samples retained for percentile computation (capped at 10 000).
    samples: Mutex<Vec<f64>>,
}

impl Histogram {
    /// Create a histogram with the given name and bucket boundaries.
    ///
    /// `boundaries` are sorted internally; duplicates are removed.
    pub fn new(name: impl Into<String>, boundaries: &[f64]) -> Self {
        let mut bounds: Vec<f64> = boundaries.to_vec();
        bounds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        bounds.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);
        let n = bounds.len() + 1; // +1 for +Inf bucket
        Self {
            boundaries: bounds,
            counts: Mutex::new(vec![0u64; n]),
            sum: Mutex::new(0.0),
            count: AtomicU64::new(0),
            name: name.into(),
            samples: Mutex::new(Vec::with_capacity(1024)),
        }
    }

    /// Record a single observation.
    pub fn record(&self, value: f64) {
        // Update sum.
        if let Ok(mut s) = self.sum.lock() {
            *s += value;
        }
        // Increment count.
        self.count.fetch_add(1, Ordering::Relaxed);

        // Find the correct bucket.
        if let Ok(mut counts) = self.counts.lock() {
            let mut placed = false;
            for (i, &bound) in self.boundaries.iter().enumerate() {
                if value <= bound {
                    // Increment this bucket and all subsequent (cumulative).
                    for j in i..counts.len() {
                        counts[j] += 1;
                    }
                    placed = true;
                    break;
                }
            }
            if !placed {
                // Falls into +Inf bucket only.
                if let Some(last) = counts.last_mut() {
                    *last += 1;
                }
            }
        }

        // Retain raw samples for percentile computation (cap at 10 000).
        if let Ok(mut s) = self.samples.lock() {
            if s.len() < 10_000 {
                s.push(value);
            }
        }
    }

    /// Return `(upper_bound, cumulative_count)` pairs for each bucket.
    ///
    /// The last bucket always has `f64::INFINITY` as the upper bound.
    pub fn buckets(&self) -> Vec<(f64, u64)> {
        let counts = self.counts.lock().map(|g| g.clone()).unwrap_or_default();
        let mut result = Vec::with_capacity(counts.len());
        for (i, count) in counts.iter().enumerate() {
            let bound = self.boundaries.get(i).copied().unwrap_or(f64::INFINITY);
            result.push((bound, *count));
        }
        result
    }

    /// Total number of observations recorded.
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Sum of all recorded values.
    pub fn sum(&self) -> f64 {
        self.sum.lock().map(|g| *g).unwrap_or(0.0)
    }

    /// Return the metric name.
    pub fn name(&self) -> &str {
        &self.name
    }

    // ---- Percentile helpers ----

    /// Compute an approximate percentile from retained samples.
    ///
    /// `p` should be in [0.0, 100.0].  Returns `0.0` when no samples have
    /// been recorded.
    pub fn percentile(&self, p: f64) -> f64 {
        let mut samples = match self.samples.lock() {
            Ok(g) => g.clone(),
            Err(_) => return 0.0,
        };
        if samples.is_empty() {
            return 0.0;
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((p / 100.0) * (samples.len() as f64 - 1.0))
            .round()
            .clamp(0.0, (samples.len() - 1) as f64) as usize;
        samples[idx]
    }

    /// 50th percentile (median).
    pub fn p50(&self) -> f64 {
        self.percentile(50.0)
    }

    /// 95th percentile.
    pub fn p95(&self) -> f64 {
        self.percentile(95.0)
    }

    /// 99th percentile.
    pub fn p99(&self) -> f64 {
        self.percentile(99.0)
    }
}

impl std::fmt::Debug for Histogram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Histogram")
            .field("name", &self.name)
            .field("count", &self.count())
            .field("sum", &self.sum())
            .finish()
    }
}

// ============================================================================
// MeterRegistry
// ============================================================================

/// A registry for named, labelled metric instruments.
///
/// Returns the *same* `Arc` when the same `(name, labels)` combination is
/// requested again.
pub struct MeterRegistry {
    counters: Mutex<HashMap<String, Arc<Counter>>>,
    gauges: Mutex<HashMap<String, Arc<Gauge>>>,
    histograms: Mutex<HashMap<String, Arc<Histogram>>>,
}

impl MeterRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            counters: Mutex::new(HashMap::new()),
            gauges: Mutex::new(HashMap::new()),
            histograms: Mutex::new(HashMap::new()),
        }
    }

    /// Retrieve or create a `Counter` with the given name and labels.
    pub fn counter(&self, name: &str, labels: &[(&str, &str)]) -> Arc<Counter> {
        let key = Self::make_key(name, labels);
        let mut map = self.counters.lock().unwrap_or_else(|e| e.into_inner());
        map.entry(key)
            .or_insert_with(|| {
                Arc::new(Counter::new(
                    name,
                    labels
                        .iter()
                        .map(|(k, v)| (k.to_string(), v.to_string()))
                        .collect(),
                ))
            })
            .clone()
    }

    /// Retrieve or create a `Gauge` with the given name.
    pub fn gauge(&self, name: &str) -> Arc<Gauge> {
        let mut map = self.gauges.lock().unwrap_or_else(|e| e.into_inner());
        map.entry(name.to_owned())
            .or_insert_with(|| Arc::new(Gauge::new(name)))
            .clone()
    }

    /// Retrieve or create a `Histogram` with the given name and bucket
    /// boundaries.
    ///
    /// If the histogram already exists, the original boundaries are preserved.
    pub fn histogram(&self, name: &str, boundaries: &[f64]) -> Arc<Histogram> {
        let mut map = self.histograms.lock().unwrap_or_else(|e| e.into_inner());
        map.entry(name.to_owned())
            .or_insert_with(|| Arc::new(Histogram::new(name, boundaries)))
            .clone()
    }

    fn make_key(name: &str, labels: &[(&str, &str)]) -> String {
        if labels.is_empty() {
            return name.to_owned();
        }
        let mut parts: Vec<String> = labels.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
        parts.sort();
        format!("{}{{{}}}", name, parts.join(","))
    }
}

impl Default for MeterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_counter_atomic() {
        let counter = Arc::new(Counter::new("requests", vec![]));
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let c = Arc::clone(&counter);
                thread::spawn(move || {
                    for _ in 0..250 {
                        c.add(1);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("thread panicked");
        }
        assert_eq!(counter.get(), 1000);
    }

    #[test]
    fn test_histogram_buckets() {
        let h = Histogram::new("latency", &[1.0, 5.0, 10.0]);
        h.record(0.5);
        h.record(3.0);
        h.record(7.0);
        h.record(20.0);
        assert_eq!(h.count(), 4);

        let buckets = h.buckets();
        // ≤1.0: only 0.5
        assert_eq!(buckets[0].1, 1);
        // ≤5.0: 0.5, 3.0
        assert_eq!(buckets[1].1, 2);
        // ≤10.0: 0.5, 3.0, 7.0
        assert_eq!(buckets[2].1, 3);
        // +Inf: all 4
        assert_eq!(buckets[3].1, 4);
    }

    #[test]
    fn test_histogram_percentiles() {
        let h = Histogram::new("rt", &[1.0, 10.0, 100.0]);
        for i in 1u64..=100 {
            h.record(i as f64);
        }
        let p50 = h.p50();
        let p95 = h.p95();
        let p99 = h.p99();
        assert!(p50 <= p95, "p50={} p95={}", p50, p95);
        assert!(p95 <= p99, "p95={} p99={}", p95, p99);
    }

    #[test]
    fn test_meter_registry_same_counter() {
        let reg = MeterRegistry::new();
        let c1 = reg.counter("reqs", &[("method", "GET")]);
        let c2 = reg.counter("reqs", &[("method", "GET")]);
        c1.add(5);
        // Both arcs point to the same counter.
        assert_eq!(c2.get(), 5);
    }

    #[test]
    fn test_gauge_set_get() {
        let g = Gauge::new("cpu");
        g.set(0.75);
        assert!((g.get() - 0.75).abs() < 1e-9);
    }
}
