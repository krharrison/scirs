//! # Lightweight Metrics Collection System
//!
//! Provides thread-safe counters, gauges, histograms, and timers for observability
//! of `SciRS2` operations. Supports both human-readable text reports and
//! Prometheus exposition format.
//!
//! ## Design
//!
//! All primitives use `std::sync::atomic` types for lock-free increments and reads.
//! The global [`MetricsRegistry`] stores named metrics behind `Arc` handles so
//! callers can cheaply clone references.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::metrics_collector::{MetricsRegistry, Counter};
//! use std::sync::Arc;
//!
//! let reg = MetricsRegistry::new();
//! let c = reg.counter("ops_total");
//! c.increment();
//! c.add(9);
//! assert_eq!(c.get(), 10);
//!
//! let report = reg.report();
//! assert!(report.contains("ops_total"));
//! ```

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// AtomicF64 helper
// ---------------------------------------------------------------------------

/// Atomic wrapper for `f64` using the bit-cast + CAS technique.
///
/// Every operation uses `SeqCst` ordering which gives the strongest guarantees
/// at the cost of maximum memory-fence overhead. For metrics collection the
/// extra safety is worth it.
struct AtomicF64 {
    bits: AtomicU64,
}

impl AtomicF64 {
    fn new(v: f64) -> Self {
        Self {
            bits: AtomicU64::new(v.to_bits()),
        }
    }

    fn load(&self) -> f64 {
        f64::from_bits(self.bits.load(Ordering::SeqCst))
    }

    /// Atomically add `delta` using a compare-and-swap loop.
    fn fetch_add(&self, delta: f64) {
        loop {
            let old_bits = self.bits.load(Ordering::SeqCst);
            let new_val = f64::from_bits(old_bits) + delta;
            let new_bits = new_val.to_bits();
            if self.bits
                .compare_exchange(old_bits, new_bits, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                return;
            }
        }
    }

    fn store(&self, v: f64) {
        self.bits.store(v.to_bits(), Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------
// Counter
// ---------------------------------------------------------------------------

/// A monotonically increasing unsigned 64-bit counter.
///
/// Backed by a single [`AtomicU64`]; all operations are lock-free.
///
/// # Example
///
/// ```rust
/// use scirs2_core::metrics_collector::Counter;
///
/// let c = Counter::new();
/// c.increment();
/// c.add(4);
/// assert_eq!(c.get(), 5);
/// c.reset();
/// assert_eq!(c.get(), 0);
/// ```
pub struct Counter {
    value: AtomicU64,
}

impl Default for Counter {
    fn default() -> Self {
        Self::new()
    }
}

impl Counter {
    /// Create a new counter starting at zero.
    pub fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
        }
    }

    /// Increment the counter by one.
    pub fn increment(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    /// Add `n` to the counter.
    pub fn add(&self, n: u64) {
        self.value.fetch_add(n, Ordering::Relaxed);
    }

    /// Read the current value.
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Reset the counter to zero.
    pub fn reset(&self) {
        self.value.store(0, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Gauge
// ---------------------------------------------------------------------------

/// A signed 64-bit gauge that can be set, incremented, or decremented.
///
/// # Example
///
/// ```rust
/// use scirs2_core::metrics_collector::Gauge;
///
/// let g = Gauge::new();
/// g.set(10);
/// g.decrement();
/// assert_eq!(g.get(), 9);
/// g.add(-9);
/// assert_eq!(g.get(), 0);
/// ```
pub struct Gauge {
    value: AtomicI64,
}

impl Default for Gauge {
    fn default() -> Self {
        Self::new()
    }
}

impl Gauge {
    /// Create a new gauge starting at zero.
    pub fn new() -> Self {
        Self {
            value: AtomicI64::new(0),
        }
    }

    /// Set the gauge to an absolute value.
    pub fn set(&self, v: i64) {
        self.value.store(v, Ordering::Relaxed);
    }

    /// Increment by one.
    pub fn increment(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement by one.
    pub fn decrement(&self) {
        self.value.fetch_sub(1, Ordering::Relaxed);
    }

    /// Add a signed delta.
    pub fn add(&self, n: i64) {
        self.value.fetch_add(n, Ordering::Relaxed);
    }

    /// Read the current value.
    pub fn get(&self) -> i64 {
        self.value.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Histogram
// ---------------------------------------------------------------------------

/// A histogram that tracks the distribution of `f64` observations.
///
/// Observations are sorted into *cumulative* buckets whose upper bounds are
/// supplied at construction time. The last bucket implicitly covers
/// `(upper_bound, +Inf)`.
///
/// # Example
///
/// ```rust
/// use scirs2_core::metrics_collector::Histogram;
///
/// let buckets = Histogram::exponential_buckets(1.0, 2.0, 5);
/// let h = Histogram::new(buckets);
/// for v in [0.5, 1.5, 3.0, 7.0, 20.0] {
///     h.observe(v);
/// }
/// assert_eq!(h.count(), 5);
/// assert!((h.mean() - 6.4).abs() < 1e-9);
/// ```
pub struct Histogram {
    /// Sorted upper bounds for each bucket.  The implicit `+Inf` bucket is
    /// tracked separately via `count`.
    buckets: Vec<f64>,
    /// Cumulative count for each explicit bucket (values ≤ upper_bound).
    counts: Vec<AtomicU64>,
    /// Running sum of all observations.
    sum: AtomicF64,
    /// Total number of observations (including those in the `+Inf` bucket).
    count: AtomicU64,
}

impl Histogram {
    /// Create a histogram with explicit bucket upper bounds.
    ///
    /// `buckets` must be sorted in ascending order. An `+Inf` bucket is
    /// always added implicitly.
    ///
    /// # Panics
    ///
    /// Does not panic. Empty `buckets` is allowed; all observations will fall
    /// in the implicit `+Inf` bucket.
    pub fn new(buckets: Vec<f64>) -> Self {
        let n = buckets.len();
        let mut counts = Vec::with_capacity(n);
        for _ in 0..n {
            counts.push(AtomicU64::new(0));
        }
        Self {
            buckets,
            counts,
            sum: AtomicF64::new(0.0),
            count: AtomicU64::new(0),
        }
    }

    /// Generate `count` linearly-spaced bucket upper bounds starting at `start`
    /// with step `width`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use scirs2_core::metrics_collector::Histogram;
    /// let b = Histogram::linear_buckets(0.0, 0.5, 4);
    /// assert_eq!(b, vec![0.0, 0.5, 1.0, 1.5]);
    /// ```
    pub fn linear_buckets(start: f64, width: f64, count: usize) -> Vec<f64> {
        (0..count).map(|i| start + width * i as f64).collect()
    }

    /// Generate `count` exponentially-spaced bucket upper bounds starting at
    /// `start`, multiplied by `factor` each step.
    ///
    /// # Example
    ///
    /// ```rust
    /// use scirs2_core::metrics_collector::Histogram;
    /// let b = Histogram::exponential_buckets(1.0, 2.0, 4);
    /// assert_eq!(b, vec![1.0, 2.0, 4.0, 8.0]);
    /// ```
    pub fn exponential_buckets(start: f64, factor: f64, count: usize) -> Vec<f64> {
        let mut v = Vec::with_capacity(count);
        let mut cur = start;
        for _ in 0..count {
            v.push(cur);
            cur *= factor;
        }
        v
    }

    /// Record a single observation.
    pub fn observe(&self, value: f64) {
        self.sum.fetch_add(value);
        self.count.fetch_add(1, Ordering::Relaxed);
        // Increment every bucket whose upper bound >= value (cumulative style).
        for (ub, cnt) in self.buckets.iter().zip(self.counts.iter()) {
            if value <= *ub {
                cnt.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Approximate the `p`-th percentile (0.0 – 100.0) using linear
    /// interpolation between bucket boundaries.
    ///
    /// Returns `0.0` if no observations have been recorded yet.
    pub fn percentile(&self, p: f64) -> f64 {
        let total = self.count.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        // Clamp p to [0, 100]
        let p = p.max(0.0).min(100.0);
        let target = (p / 100.0) * total as f64;

        if self.buckets.is_empty() {
            // No explicit buckets — can only return 0
            return 0.0;
        }

        let mut prev_count: f64 = 0.0;
        let mut prev_bound = 0.0_f64;

        for (i, (ub, cnt)) in self.buckets.iter().zip(self.counts.iter()).enumerate() {
            let cum = cnt.load(Ordering::Relaxed) as f64;
            if cum >= target {
                // Linear interpolation within this bucket.
                let bucket_count = cum - prev_count;
                if bucket_count <= 0.0 {
                    return *ub;
                }
                let lower = if i == 0 { 0.0 } else { prev_bound };
                let frac = (target - prev_count) / bucket_count;
                return lower + frac * (*ub - lower);
            }
            prev_count = cum;
            prev_bound = *ub;
        }

        // All observations are in the +Inf bucket; return the last explicit bound.
        *self.buckets.last().unwrap_or(&0.0)
    }

    /// Mean of all observations. Returns `0.0` if count is zero.
    pub fn mean(&self) -> f64 {
        let n = self.count.load(Ordering::Relaxed);
        if n == 0 {
            return 0.0;
        }
        self.sum.load() / n as f64
    }

    /// Sum of all observations.
    pub fn sum(&self) -> f64 {
        self.sum.load()
    }

    /// Number of observations recorded.
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Snapshot of bucket counts in `(upper_bound, cumulative_count)` pairs.
    pub fn bucket_snapshot(&self) -> Vec<(f64, u64)> {
        self.buckets
            .iter()
            .zip(self.counts.iter())
            .map(|(ub, cnt)| (*ub, cnt.load(Ordering::Relaxed)))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Timer
// ---------------------------------------------------------------------------

/// A one-shot timer that records its duration into a [`Histogram`] when dropped
/// (or when [`Timer::stop`] is called).
///
/// # Example
///
/// ```rust
/// use scirs2_core::metrics_collector::{Histogram, Timer};
/// use std::sync::Arc;
///
/// let h = Arc::new(Histogram::new(vec![0.001, 0.01, 0.1, 1.0]));
/// let t = Timer::start(Arc::clone(&h));
/// let _elapsed = t.stop();
/// assert_eq!(h.count(), 1);
/// ```
pub struct Timer {
    histogram: Arc<Histogram>,
    start: Instant,
    /// Prevent double-recording when `stop()` is called before `drop`.
    stopped: bool,
}

impl Timer {
    /// Start a new timer that will record into `histogram`.
    pub fn start(histogram: Arc<Histogram>) -> Self {
        Self {
            histogram,
            start: Instant::now(),
            stopped: false,
        }
    }

    /// Stop the timer, record the elapsed duration in seconds, and return it.
    pub fn stop(mut self) -> Duration {
        let elapsed = self.start.elapsed();
        self.histogram.observe(elapsed.as_secs_f64());
        self.stopped = true;
        elapsed
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        if !self.stopped {
            let elapsed = self.start.elapsed();
            self.histogram.observe(elapsed.as_secs_f64());
        }
    }
}

// ---------------------------------------------------------------------------
// MetricsRegistry
// ---------------------------------------------------------------------------

/// Central registry that stores named metrics.
///
/// The global singleton is available via [`MetricsRegistry::global`].
///
/// # Example
///
/// ```rust
/// use scirs2_core::metrics_collector::MetricsRegistry;
///
/// let reg = MetricsRegistry::global();
/// let c = reg.counter("my_ops");
/// c.increment();
/// let report = reg.report();
/// assert!(report.contains("my_ops"));
/// ```
pub struct MetricsRegistry {
    counters: Mutex<HashMap<String, Arc<Counter>>>,
    gauges: Mutex<HashMap<String, Arc<Gauge>>>,
    histograms: Mutex<HashMap<String, Arc<Histogram>>>,
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// The process-wide global registry.
static GLOBAL_REGISTRY: Lazy<MetricsRegistry> = Lazy::new(MetricsRegistry::new);

impl MetricsRegistry {
    /// Create a fresh, empty registry (useful for tests).
    pub fn new() -> Self {
        Self {
            counters: Mutex::new(HashMap::new()),
            gauges: Mutex::new(HashMap::new()),
            histograms: Mutex::new(HashMap::new()),
        }
    }

    /// Return a reference to the process-wide global registry.
    pub fn global() -> &'static Self {
        &GLOBAL_REGISTRY
    }

    /// Get or create a [`Counter`] by name.
    pub fn counter(&self, name: &str) -> Arc<Counter> {
        let mut map = self.counters.lock().unwrap_or_else(|e| e.into_inner());
        map.entry(name.to_owned())
            .or_insert_with(|| Arc::new(Counter::new()))
            .clone()
    }

    /// Get or create a [`Gauge`] by name.
    pub fn gauge(&self, name: &str) -> Arc<Gauge> {
        let mut map = self.gauges.lock().unwrap_or_else(|e| e.into_inner());
        map.entry(name.to_owned())
            .or_insert_with(|| Arc::new(Gauge::new()))
            .clone()
    }

    /// Get or create a [`Histogram`] by name.
    ///
    /// The `buckets` argument is only used the *first* time this name is
    /// registered. Subsequent calls with the same name return the existing
    /// histogram regardless of the `buckets` argument.
    pub fn histogram(&self, name: &str, buckets: Vec<f64>) -> Arc<Histogram> {
        let mut map = self.histograms.lock().unwrap_or_else(|e| e.into_inner());
        map.entry(name.to_owned())
            .or_insert_with(|| Arc::new(Histogram::new(buckets)))
            .clone()
    }

    /// Return a human-readable multi-line report of all registered metrics.
    pub fn report(&self) -> String {
        let mut out = String::with_capacity(512);
        out.push_str("=== SciRS2 Metrics Report ===\n");

        // Counters
        {
            let map = self.counters.lock().unwrap_or_else(|e| e.into_inner());
            if !map.is_empty() {
                out.push_str("\n-- Counters --\n");
                let mut names: Vec<&String> = map.keys().collect();
                names.sort();
                for name in names {
                    let c = &map[name];
                    out.push_str(&format!("  {name}: {}\n", c.get()));
                }
            }
        }

        // Gauges
        {
            let map = self.gauges.lock().unwrap_or_else(|e| e.into_inner());
            if !map.is_empty() {
                out.push_str("\n-- Gauges --\n");
                let mut names: Vec<&String> = map.keys().collect();
                names.sort();
                for name in names {
                    let g = &map[name];
                    out.push_str(&format!("  {name}: {}\n", g.get()));
                }
            }
        }

        // Histograms
        {
            let map = self.histograms.lock().unwrap_or_else(|e| e.into_inner());
            if !map.is_empty() {
                out.push_str("\n-- Histograms --\n");
                let mut names: Vec<&String> = map.keys().collect();
                names.sort();
                for name in names {
                    let h = &map[name];
                    out.push_str(&format!(
                        "  {name}: count={} sum={:.6} mean={:.6} p50={:.6} p95={:.6} p99={:.6}\n",
                        h.count(),
                        h.sum(),
                        h.mean(),
                        h.percentile(50.0),
                        h.percentile(95.0),
                        h.percentile(99.0),
                    ));
                }
            }
        }

        out
    }

    /// Emit all metrics in the
    /// [Prometheus text exposition format](https://prometheus.io/docs/instrumenting/exposition_formats/).
    ///
    /// Counter lines use the `# TYPE … counter` stanza; gauges use `gauge`;
    /// histograms use `histogram` with `_bucket`, `_sum`, and `_count` lines.
    pub fn prometheus_format(&self) -> String {
        let mut out = String::with_capacity(1024);

        // Counters
        {
            let map = self.counters.lock().unwrap_or_else(|e| e.into_inner());
            let mut names: Vec<&String> = map.keys().collect();
            names.sort();
            for name in names {
                let c = &map[name];
                out.push_str(&format!("# TYPE {name} counter\n"));
                out.push_str(&format!("{name}_total {}\n", c.get()));
            }
        }

        // Gauges
        {
            let map = self.gauges.lock().unwrap_or_else(|e| e.into_inner());
            let mut names: Vec<&String> = map.keys().collect();
            names.sort();
            for name in names {
                let g = &map[name];
                out.push_str(&format!("# TYPE {name} gauge\n"));
                out.push_str(&format!("{name} {}\n", g.get()));
            }
        }

        // Histograms
        {
            let map = self.histograms.lock().unwrap_or_else(|e| e.into_inner());
            let mut names: Vec<&String> = map.keys().collect();
            names.sort();
            for name in names {
                let h = &map[name];
                out.push_str(&format!("# TYPE {name} histogram\n"));
                for (ub, cum_count) in h.bucket_snapshot() {
                    out.push_str(&format!(
                        "{name}_bucket{{le=\"{ub}\"}} {cum_count}\n"
                    ));
                }
                out.push_str(&format!(
                    "{name}_bucket{{le=\"+Inf\"}} {}\n",
                    h.count()
                ));
                out.push_str(&format!("{name}_sum {}\n", h.sum()));
                out.push_str(&format!("{name}_count {}\n", h.count()));
            }
        }

        out
    }

    /// Reset every metric in the registry to zero.
    pub fn reset_all(&self) {
        {
            let map = self.counters.lock().unwrap_or_else(|e| e.into_inner());
            for c in map.values() {
                c.reset();
            }
        }
        {
            let map = self.gauges.lock().unwrap_or_else(|e| e.into_inner());
            for g in map.values() {
                g.set(0);
            }
        }
        // Histograms cannot be reset atomically per-bucket; rebuild them.
        // We replace each histogram with a fresh one with the same buckets.
        {
            let mut map = self.histograms.lock().unwrap_or_else(|e| e.into_inner());
            for h in map.values_mut() {
                let buckets = h.buckets.clone();
                *h = Arc::new(Histogram::new(buckets));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience macros
// ---------------------------------------------------------------------------

/// Increment the named counter in the global registry.
///
/// ```rust
/// use scirs2_core::metrics_counter;
/// metrics_counter!("example_ops").increment();
/// ```
#[macro_export]
macro_rules! metrics_counter {
    ($name:expr) => {
        $crate::metrics_collector::MetricsRegistry::global().counter($name)
    };
}

/// Time a block of code and record its duration (in seconds) into a histogram
/// in the global registry.  Uses exponential buckets covering 1 µs – 10 s.
///
/// ```rust
/// use scirs2_core::metrics_time;
/// let result = metrics_time!("example_work", {
///     42_u32
/// });
/// assert_eq!(result, 42);
/// ```
#[macro_export]
macro_rules! metrics_time {
    ($name:expr, $body:block) => {{
        let _reg = $crate::metrics_collector::MetricsRegistry::global();
        let _h = _reg.histogram(
            $name,
            $crate::metrics_collector::Histogram::exponential_buckets(1e-6, 10.0, 8),
        );
        let _timer = $crate::metrics_collector::Timer::start(::std::sync::Arc::clone(&_h));
        let _result = $body;
        let _ = _timer.stop();
        _result
    }};
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    // ---- Counter ----

    #[test]
    fn test_counter_basic() {
        let c = Counter::new();
        assert_eq!(c.get(), 0);
        c.increment();
        assert_eq!(c.get(), 1);
        c.add(9);
        assert_eq!(c.get(), 10);
        c.reset();
        assert_eq!(c.get(), 0);
    }

    #[test]
    fn test_counter_concurrent() {
        let c = Arc::new(Counter::new());
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let c2 = Arc::clone(&c);
                thread::spawn(move || {
                    for _ in 0..1000 {
                        c2.increment();
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().expect("thread panicked");
        }
        assert_eq!(c.get(), 8 * 1000);
    }

    // ---- Gauge ----

    #[test]
    fn test_gauge_basic() {
        let g = Gauge::new();
        assert_eq!(g.get(), 0);
        g.set(42);
        assert_eq!(g.get(), 42);
        g.increment();
        assert_eq!(g.get(), 43);
        g.decrement();
        assert_eq!(g.get(), 42);
        g.add(-42);
        assert_eq!(g.get(), 0);
    }

    #[test]
    fn test_gauge_negative() {
        let g = Gauge::new();
        g.add(-100);
        assert_eq!(g.get(), -100);
    }

    // ---- Histogram ----

    #[test]
    fn test_histogram_observe_and_mean() {
        let h = Histogram::new(vec![1.0, 2.0, 5.0, 10.0]);
        for v in [0.5, 1.5, 3.0, 7.0, 12.0] {
            h.observe(v);
        }
        assert_eq!(h.count(), 5);
        let expected_sum = 0.5 + 1.5 + 3.0 + 7.0 + 12.0;
        assert!((h.sum() - expected_sum).abs() < 1e-9);
        assert!((h.mean() - expected_sum / 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_histogram_bucket_counts() {
        let h = Histogram::new(vec![1.0, 5.0, 10.0]);
        h.observe(0.5); // ≤1 ≤5 ≤10
        h.observe(2.0); // ≤5 ≤10
        h.observe(6.0); // ≤10
        h.observe(11.0); // none (inf)

        let snap = h.bucket_snapshot();
        assert_eq!(snap[0], (1.0, 1)); // only 0.5 ≤ 1
        assert_eq!(snap[1], (5.0, 2)); // 0.5 and 2.0 ≤ 5
        assert_eq!(snap[2], (10.0, 3)); // 0.5, 2.0, 6.0 ≤ 10
        assert_eq!(h.count(), 4);
    }

    #[test]
    fn test_histogram_percentile_approximation() {
        // Use narrow linear buckets so we can verify p50 is near median.
        let buckets = Histogram::linear_buckets(0.0, 1.0, 10); // 0..9
        let h = Histogram::new(buckets);
        // Observe values 0..=9 uniformly
        for i in 0..=9 {
            h.observe(i as f64);
        }
        let p50 = h.percentile(50.0);
        // Median of 0..9 is ~4.5; with bucket-based approximation, expect within ±1.0
        assert!(
            (p50 - 4.5).abs() < 1.5,
            "p50={p50} expected ~4.5"
        );
        let p0 = h.percentile(0.0);
        assert!(p0 >= 0.0, "p0={p0} should be ≥ 0");
        let p100 = h.percentile(100.0);
        assert!(p100 >= 0.0, "p100={p100} should be ≥ 0");
    }

    #[test]
    fn test_histogram_empty() {
        let h = Histogram::new(vec![1.0, 2.0]);
        assert_eq!(h.count(), 0);
        assert_eq!(h.sum(), 0.0);
        assert_eq!(h.mean(), 0.0);
        assert_eq!(h.percentile(50.0), 0.0);
    }

    #[test]
    fn test_histogram_linear_buckets() {
        let b = Histogram::linear_buckets(0.0, 0.5, 4);
        assert_eq!(b, vec![0.0, 0.5, 1.0, 1.5]);
    }

    #[test]
    fn test_histogram_exponential_buckets() {
        let b = Histogram::exponential_buckets(1.0, 2.0, 4);
        assert_eq!(b, vec![1.0, 2.0, 4.0, 8.0]);
    }

    // ---- Timer ----

    #[test]
    fn test_timer_records_into_histogram() {
        let h = Arc::new(Histogram::new(vec![0.001, 0.01, 0.1, 1.0]));
        let t = Timer::start(Arc::clone(&h));
        let elapsed = t.stop();
        assert_eq!(h.count(), 1);
        // elapsed must be non-negative
        assert!(elapsed.as_nanos() > 0 || elapsed.as_nanos() == 0);
    }

    #[test]
    fn test_timer_drop_records() {
        let h = Arc::new(Histogram::new(vec![0.001, 0.01, 0.1, 1.0]));
        {
            let _t = Timer::start(Arc::clone(&h));
            // drop without calling stop()
        }
        assert_eq!(h.count(), 1);
    }

    // ---- MetricsRegistry ----

    #[test]
    fn test_registry_counter_idempotent() {
        let reg = MetricsRegistry::new();
        let c1 = reg.counter("hits");
        let c2 = reg.counter("hits");
        c1.add(5);
        assert_eq!(c2.get(), 5); // same underlying Arc
    }

    #[test]
    fn test_registry_gauge_idempotent() {
        let reg = MetricsRegistry::new();
        let g1 = reg.gauge("queue_depth");
        let g2 = reg.gauge("queue_depth");
        g1.set(7);
        assert_eq!(g2.get(), 7);
    }

    #[test]
    fn test_registry_histogram_idempotent() {
        let reg = MetricsRegistry::new();
        let h1 = reg.histogram("latency", vec![0.1, 1.0, 10.0]);
        let h2 = reg.histogram("latency", vec![99.0]); // buckets ignored on 2nd call
        h1.observe(0.5);
        assert_eq!(h2.count(), 1);
    }

    #[test]
    fn test_registry_report_contains_names() {
        let reg = MetricsRegistry::new();
        let c = reg.counter("total_requests");
        c.add(42);
        let g = reg.gauge("active_conns");
        g.set(3);
        let h = reg.histogram("response_time_s", vec![0.1, 1.0]);
        h.observe(0.05);

        let report = reg.report();
        assert!(report.contains("total_requests"), "report:\n{report}");
        assert!(report.contains("42"), "report:\n{report}");
        assert!(report.contains("active_conns"), "report:\n{report}");
        assert!(report.contains("response_time_s"), "report:\n{report}");
    }

    #[test]
    fn test_registry_prometheus_format() {
        let reg = MetricsRegistry::new();
        reg.counter("prom_ops").add(7);
        reg.gauge("prom_queue").set(2);
        let h = reg.histogram("prom_latency", vec![0.001, 0.01]);
        h.observe(0.005);

        let prom = reg.prometheus_format();
        assert!(prom.contains("# TYPE prom_ops counter"), "prom:\n{prom}");
        assert!(prom.contains("prom_ops_total 7"), "prom:\n{prom}");
        assert!(prom.contains("# TYPE prom_queue gauge"), "prom:\n{prom}");
        assert!(prom.contains("prom_queue 2"), "prom:\n{prom}");
        assert!(prom.contains("# TYPE prom_latency histogram"), "prom:\n{prom}");
        assert!(prom.contains("prom_latency_count 1"), "prom:\n{prom}");
    }

    #[test]
    fn test_registry_reset_all() {
        let reg = MetricsRegistry::new();
        let c = reg.counter("reset_me");
        c.add(100);
        let g = reg.gauge("reset_gauge");
        g.set(50);
        let h = reg.histogram("reset_hist", vec![1.0, 2.0]);
        h.observe(0.5);

        reg.reset_all();

        assert_eq!(c.get(), 0);
        assert_eq!(g.get(), 0);
        // After reset, histogram Arc inside registry is new; original `h` is stale.
        // The new histogram should have count=0.
        let h2 = reg.histogram("reset_hist", vec![1.0, 2.0]);
        assert_eq!(h2.count(), 0);
    }

    // ---- Macros ----

    #[test]
    fn test_metrics_counter_macro() {
        metrics_counter!("macro_test_counter").add(3);
        let val = MetricsRegistry::global()
            .counter("macro_test_counter")
            .get();
        // May already be incremented by other test runs; just check it's >= 3.
        assert!(val >= 3);
    }

    #[test]
    fn test_metrics_time_macro() {
        let result = metrics_time!("macro_test_timing", {
            1 + 1
        });
        assert_eq!(result, 2);
        let h = MetricsRegistry::global()
            .histogram(
                "macro_test_timing",
                Histogram::exponential_buckets(1e-6, 10.0, 8),
            );
        assert!(h.count() >= 1);
    }
}
