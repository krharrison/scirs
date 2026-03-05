//! # Benchmarking Utilities for SciRS2
//!
//! This module provides lightweight, pure-Rust benchmarking tools for measuring
//! function performance, comparing implementations, and tracking throughput.
//!
//! ## Features
//!
//! - `Stopwatch` for manual timing with lap support
//! - `benchmark_fn` for automated function benchmarking with statistics
//! - `compare_implementations` for A/B testing with Welch's t-test
//! - `throughput_bench` for measuring operations/second and bytes/second
//! - `memory_bench` for estimating peak memory usage
//! - Warm-up support (discard first N iterations)
//! - CSV and JSON output for results

use std::fmt;
use std::time::{Duration, Instant};

use crate::error::{CoreError, CoreResult, ErrorContext};

// ---------------------------------------------------------------------------
// Stopwatch
// ---------------------------------------------------------------------------

/// A simple stopwatch for manual timing with lap support.
///
/// # Example
///
/// ```
/// use scirs2_core::bench_utils::Stopwatch;
///
/// let mut sw = Stopwatch::new();
/// sw.start();
/// // ... work ...
/// let lap1 = sw.lap();
/// // ... more work ...
/// sw.stop();
/// let total = sw.elapsed();
/// ```
#[derive(Debug, Clone)]
pub struct Stopwatch {
    start_time: Option<Instant>,
    elapsed: Duration,
    laps: Vec<Duration>,
    running: bool,
}

impl Stopwatch {
    /// Create a new stopped stopwatch.
    pub fn new() -> Self {
        Self {
            start_time: None,
            elapsed: Duration::ZERO,
            laps: Vec::new(),
            running: false,
        }
    }

    /// Start (or resume) the stopwatch.
    pub fn start(&mut self) {
        if !self.running {
            self.start_time = Some(Instant::now());
            self.running = true;
        }
    }

    /// Stop the stopwatch.
    pub fn stop(&mut self) {
        if self.running {
            if let Some(start) = self.start_time.take() {
                self.elapsed += start.elapsed();
            }
            self.running = false;
        }
    }

    /// Record a lap time without stopping. Returns the lap duration.
    pub fn lap(&mut self) -> Duration {
        let now = Instant::now();
        let lap_duration = if let Some(start) = self.start_time {
            now.duration_since(start)
        } else {
            Duration::ZERO
        };
        self.laps.push(lap_duration);
        // Reset the start for the next lap segment
        self.start_time = Some(now);
        lap_duration
    }

    /// Reset the stopwatch to zero.
    pub fn reset(&mut self) {
        self.start_time = None;
        self.elapsed = Duration::ZERO;
        self.laps.clear();
        self.running = false;
    }

    /// Get total elapsed time. If still running, includes current segment.
    pub fn elapsed(&self) -> Duration {
        let mut total = self.elapsed;
        if self.running {
            if let Some(start) = self.start_time {
                total += start.elapsed();
            }
        }
        total
    }

    /// Get all recorded lap times.
    pub fn laps(&self) -> &[Duration] {
        &self.laps
    }

    /// Whether the stopwatch is currently running.
    pub fn is_running(&self) -> bool {
        self.running
    }
}

impl Default for Stopwatch {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// BenchmarkConfig
// ---------------------------------------------------------------------------

/// Configuration for a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warm-up iterations (discarded).
    pub warmup_iterations: usize,
    /// Number of measured iterations.
    pub iterations: usize,
}

impl BenchmarkConfig {
    /// Create config with given warm-up and iteration counts.
    pub fn new(warmup_iterations: usize, iterations: usize) -> Self {
        Self {
            warmup_iterations,
            iterations,
        }
    }
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 5,
            iterations: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// BenchmarkStats
// ---------------------------------------------------------------------------

/// Statistics produced by a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkStats {
    /// Minimum time observed.
    pub min: Duration,
    /// Maximum time observed.
    pub max: Duration,
    /// Arithmetic mean of all samples.
    pub mean: Duration,
    /// Median (50th percentile).
    pub median: Duration,
    /// 99th percentile.
    pub p99: Duration,
    /// Standard deviation in nanoseconds.
    pub std_dev_nanos: f64,
    /// Number of samples.
    pub sample_count: usize,
    /// All sample durations (sorted).
    pub samples: Vec<Duration>,
}

impl BenchmarkStats {
    /// Compute statistics from a **non-empty** vector of durations.
    fn from_samples(mut durations: Vec<Duration>) -> CoreResult<Self> {
        if durations.is_empty() {
            return Err(CoreError::ValueError(ErrorContext::new(
                "Cannot compute benchmark stats from zero samples",
            )));
        }
        durations.sort();

        let n = durations.len();
        let min = durations[0];
        let max = durations[n - 1];

        let total_nanos: u128 = durations.iter().map(|d| d.as_nanos()).sum();
        let mean_nanos = total_nanos / n as u128;
        let mean = Duration::from_nanos(mean_nanos as u64);

        let median = if n % 2 == 0 {
            let a = durations[n / 2 - 1].as_nanos();
            let b = durations[n / 2].as_nanos();
            Duration::from_nanos(((a + b) / 2) as u64)
        } else {
            durations[n / 2]
        };

        let p99_idx = ((n as f64) * 0.99).ceil() as usize;
        let p99 = durations[p99_idx.min(n - 1)];

        // Std dev
        let mean_f = mean_nanos as f64;
        let variance: f64 = durations
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean_f;
                diff * diff
            })
            .sum::<f64>()
            / (n.max(1) as f64);
        let std_dev_nanos = variance.sqrt();

        Ok(Self {
            min,
            max,
            mean,
            median,
            p99,
            std_dev_nanos,
            sample_count: n,
            samples: durations,
        })
    }

    /// Format stats as a CSV row (header: "min_ns,max_ns,mean_ns,median_ns,p99_ns,std_dev_ns,n").
    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{},{},{:.2},{}",
            self.min.as_nanos(),
            self.max.as_nanos(),
            self.mean.as_nanos(),
            self.median.as_nanos(),
            self.p99.as_nanos(),
            self.std_dev_nanos,
            self.sample_count,
        )
    }

    /// CSV header matching `to_csv_row`.
    pub fn csv_header() -> &'static str {
        "min_ns,max_ns,mean_ns,median_ns,p99_ns,std_dev_ns,n"
    }

    /// Format stats as a JSON string.
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"min_ns":{},"max_ns":{},"mean_ns":{},"median_ns":{},"p99_ns":{},"std_dev_ns":{:.2},"n":{}}}"#,
            self.min.as_nanos(),
            self.max.as_nanos(),
            self.mean.as_nanos(),
            self.median.as_nanos(),
            self.p99.as_nanos(),
            self.std_dev_nanos,
            self.sample_count,
        )
    }
}

impl fmt::Display for BenchmarkStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "min={:?}  max={:?}  mean={:?}  median={:?}  p99={:?}  std_dev={:.0}ns  n={}",
            self.min,
            self.max,
            self.mean,
            self.median,
            self.p99,
            self.std_dev_nanos,
            self.sample_count,
        )
    }
}

// ---------------------------------------------------------------------------
// benchmark_fn
// ---------------------------------------------------------------------------

/// Run a function `N` times (after warm-up) and return performance statistics.
///
/// The function under test receives no arguments and may return any type
/// (the result is discarded via `std::hint::black_box`).
///
/// # Example
///
/// ```
/// use scirs2_core::bench_utils::{benchmark_fn, BenchmarkConfig};
///
/// let stats = benchmark_fn(
///     &BenchmarkConfig::new(3, 50),
///     || {
///         let v: Vec<f64> = (0..1000).map(|i| (i as f64).sqrt()).collect();
///         v
///     },
/// ).expect("benchmark should succeed");
///
/// assert!(stats.sample_count == 50);
/// ```
pub fn benchmark_fn<F, R>(config: &BenchmarkConfig, mut func: F) -> CoreResult<BenchmarkStats>
where
    F: FnMut() -> R,
{
    if config.iterations == 0 {
        return Err(CoreError::ValueError(ErrorContext::new(
            "iterations must be > 0",
        )));
    }

    // Warm-up
    for _ in 0..config.warmup_iterations {
        std::hint::black_box(func());
    }

    // Measured runs
    let mut durations = Vec::with_capacity(config.iterations);
    for _ in 0..config.iterations {
        let start = Instant::now();
        std::hint::black_box(func());
        durations.push(start.elapsed());
    }

    BenchmarkStats::from_samples(durations)
}

// ---------------------------------------------------------------------------
// compare_implementations (Welch's t-test)
// ---------------------------------------------------------------------------

/// Result of comparing two implementations.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Stats for implementation A.
    pub stats_a: BenchmarkStats,
    /// Stats for implementation B.
    pub stats_b: BenchmarkStats,
    /// Welch's t-statistic (positive means B is faster).
    pub t_statistic: f64,
    /// Approximate two-sided p-value.
    pub p_value: f64,
    /// Speedup ratio (mean_a / mean_b). > 1 means B is faster.
    pub speedup: f64,
    /// Whether the difference is statistically significant at alpha = 0.05.
    pub significant: bool,
}

impl fmt::Display for ComparisonResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let faster = if self.speedup > 1.0 { "B" } else { "A" };
        let ratio = if self.speedup > 1.0 {
            self.speedup
        } else if self.speedup > 0.0 {
            1.0 / self.speedup
        } else {
            f64::NAN
        };
        write!(
            f,
            "{faster} is {ratio:.2}x faster  t={:.3}  p={:.4}  sig={}",
            self.t_statistic, self.p_value, self.significant
        )
    }
}

/// Compare two implementations using Welch's t-test on their execution times.
///
/// Returns statistics for both plus a significance assessment.
pub fn compare_implementations<FA, FB, RA, RB>(
    config: &BenchmarkConfig,
    mut func_a: FA,
    mut func_b: FB,
) -> CoreResult<ComparisonResult>
where
    FA: FnMut() -> RA,
    FB: FnMut() -> RB,
{
    let stats_a = benchmark_fn(config, &mut func_a)?;
    let stats_b = benchmark_fn(config, &mut func_b)?;

    let n_a = stats_a.sample_count as f64;
    let n_b = stats_b.sample_count as f64;
    let mean_a = stats_a.mean.as_nanos() as f64;
    let mean_b = stats_b.mean.as_nanos() as f64;
    let var_a = stats_a.std_dev_nanos * stats_a.std_dev_nanos;
    let var_b = stats_b.std_dev_nanos * stats_b.std_dev_nanos;

    let se = ((var_a / n_a) + (var_b / n_b)).sqrt();
    let t_statistic = if se > 0.0 {
        (mean_a - mean_b) / se
    } else {
        0.0
    };

    // Approximate p-value using the normal distribution (valid for large N).
    let p_value = approx_two_sided_p(t_statistic);

    let speedup = if mean_b > 0.0 {
        mean_a / mean_b
    } else {
        f64::NAN
    };

    Ok(ComparisonResult {
        stats_a,
        stats_b,
        t_statistic,
        p_value,
        speedup,
        significant: p_value < 0.05,
    })
}

/// Approximate two-sided p-value from a t-statistic using the standard
/// normal CDF (good approximation when df > 30).
fn approx_two_sided_p(t: f64) -> f64 {
    // Abramowitz & Stegun approximation for the standard normal CDF
    let x = t.abs();
    let b1 = 0.319_381_530;
    let b2 = -0.356_563_782;
    let b3 = 1.781_477_937;
    let b4 = -1.821_255_978;
    let b5 = 1.330_274_429;
    let p_coeff = 0.231_641_9;

    let t_val = 1.0 / (1.0 + p_coeff * x);
    let t2 = t_val * t_val;
    let t3 = t2 * t_val;
    let t4 = t3 * t_val;
    let t5 = t4 * t_val;

    let pdf = (-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf = 1.0 - pdf * (b1 * t_val + b2 * t2 + b3 * t3 + b4 * t4 + b5 * t5);
    let one_tail = 1.0 - cdf;
    (2.0 * one_tail).min(1.0).max(0.0)
}

// ---------------------------------------------------------------------------
// throughput_bench
// ---------------------------------------------------------------------------

/// Result from a throughput benchmark.
#[derive(Debug, Clone)]
pub struct ThroughputResult {
    /// Underlying timing statistics.
    pub stats: BenchmarkStats,
    /// Operations per second.
    pub ops_per_sec: f64,
    /// Bytes processed per second (if bytes_per_op was provided).
    pub bytes_per_sec: Option<f64>,
}

impl fmt::Display for ThroughputResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2} ops/sec", self.ops_per_sec)?;
        if let Some(bps) = self.bytes_per_sec {
            let (val, unit) = humanize_bytes_per_sec(bps);
            write!(f, "  {val:.2} {unit}")?;
        }
        Ok(())
    }
}

fn humanize_bytes_per_sec(bps: f64) -> (f64, &'static str) {
    if bps >= 1e9 {
        (bps / 1e9, "GB/s")
    } else if bps >= 1e6 {
        (bps / 1e6, "MB/s")
    } else if bps >= 1e3 {
        (bps / 1e3, "KB/s")
    } else {
        (bps, "B/s")
    }
}

/// Measure throughput (ops/sec, optionally bytes/sec).
///
/// * `bytes_per_op` -- if `Some(n)`, also report bytes/sec based on n bytes per invocation.
pub fn throughput_bench<F, R>(
    config: &BenchmarkConfig,
    func: F,
    bytes_per_op: Option<usize>,
) -> CoreResult<ThroughputResult>
where
    F: FnMut() -> R,
{
    let stats = benchmark_fn(config, func)?;
    let mean_secs = stats.mean.as_secs_f64();
    let ops_per_sec = if mean_secs > 0.0 {
        1.0 / mean_secs
    } else {
        f64::INFINITY
    };
    let bytes_per_sec = bytes_per_op.map(|b| b as f64 * ops_per_sec);

    Ok(ThroughputResult {
        stats,
        ops_per_sec,
        bytes_per_sec,
    })
}

// ---------------------------------------------------------------------------
// memory_bench
// ---------------------------------------------------------------------------

/// Result from a memory benchmark.
#[derive(Debug, Clone)]
pub struct MemoryBenchResult {
    /// Estimated peak memory increase (bytes) during execution.
    ///
    /// Note: this is a best-effort estimate using the system allocator info
    /// where available. On platforms without allocator introspection it
    /// returns `None`.
    pub peak_memory_bytes: Option<usize>,
    /// Timing statistics.
    pub stats: BenchmarkStats,
}

/// Measure execution time and attempt to estimate peak memory usage for `func`.
///
/// Memory estimation is best-effort. On most platforms we measure the
/// difference in resident set size before and after execution. This is
/// inherently approximate because the OS may not reclaim freed pages
/// immediately.
pub fn memory_bench<F, R>(config: &BenchmarkConfig, mut func: F) -> CoreResult<MemoryBenchResult>
where
    F: FnMut() -> R,
{
    // Get baseline RSS
    let baseline_rss = current_rss_bytes();

    // Warm-up
    for _ in 0..config.warmup_iterations {
        std::hint::black_box(func());
    }

    let mut durations = Vec::with_capacity(config.iterations);
    let mut max_rss_delta: Option<usize> = None;

    for _ in 0..config.iterations {
        let before_rss = current_rss_bytes();
        let start = Instant::now();
        std::hint::black_box(func());
        let elapsed = start.elapsed();
        let after_rss = current_rss_bytes();

        durations.push(elapsed);

        if let (Some(before), Some(after)) = (before_rss, after_rss) {
            let delta = after.saturating_sub(before);
            max_rss_delta = Some(max_rss_delta.map_or(delta, |prev: usize| prev.max(delta)));
        }
    }

    let stats = BenchmarkStats::from_samples(durations)?;

    // Fall back to overall delta if per-iteration deltas were zero
    let peak_memory_bytes = match max_rss_delta {
        Some(0) => {
            // Try overall measurement
            let end_rss = current_rss_bytes();
            match (baseline_rss, end_rss) {
                (Some(b), Some(e)) => {
                    let delta = e.saturating_sub(b);
                    if delta > 0 {
                        Some(delta)
                    } else {
                        Some(0)
                    }
                }
                _ => None,
            }
        }
        other => other,
    };

    Ok(MemoryBenchResult {
        peak_memory_bytes,
        stats,
    })
}

/// Try to obtain the current resident set size in bytes.
/// Returns `None` if not available on this platform.
fn current_rss_bytes() -> Option<usize> {
    // On macOS, use libc getrusage when the cross_platform feature is enabled
    #[cfg(all(target_os = "macos", feature = "cross_platform"))]
    {
        return macos_rss();
    }

    // On Linux, read /proc/self/statm (no external deps needed)
    #[cfg(target_os = "linux")]
    {
        return linux_rss();
    }

    // Fallback: not available
    #[allow(unreachable_code)]
    None
}

#[cfg(all(target_os = "macos", feature = "cross_platform"))]
fn macos_rss() -> Option<usize> {
    // Use the rusage approach which is simpler and doesn't require mach bindings
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if ret == 0 {
        // ru_maxrss is in bytes on macOS
        Some(usage.ru_maxrss as usize)
    } else {
        None
    }
}

#[cfg(target_os = "linux")]
fn linux_rss() -> Option<usize> {
    use std::fs;
    let statm = fs::read_to_string("/proc/self/statm").ok()?;
    let rss_pages: usize = statm.split_whitespace().nth(1)?.parse().ok()?;
    let page_size = 4096_usize; // typical
    Some(rss_pages * page_size)
}

// ---------------------------------------------------------------------------
// BenchmarkReport -- CSV / JSON output
// ---------------------------------------------------------------------------

/// A named benchmark result for inclusion in a report.
#[derive(Debug, Clone)]
pub struct NamedBenchmark {
    /// Human-readable label.
    pub name: String,
    /// The statistics.
    pub stats: BenchmarkStats,
}

/// A collection of named benchmark results with CSV / JSON export.
#[derive(Debug, Clone, Default)]
pub struct BenchmarkReport {
    /// All benchmark entries.
    pub entries: Vec<NamedBenchmark>,
}

impl BenchmarkReport {
    /// Create an empty report.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a named result.
    pub fn add(&mut self, name: impl Into<String>, stats: BenchmarkStats) {
        self.entries.push(NamedBenchmark {
            name: name.into(),
            stats,
        });
    }

    /// Serialize the report as CSV.
    pub fn to_csv(&self) -> String {
        let mut out = format!("name,{}\n", BenchmarkStats::csv_header());
        for entry in &self.entries {
            out.push_str(&format!("{},{}\n", entry.name, entry.stats.to_csv_row()));
        }
        out
    }

    /// Serialize the report as JSON.
    pub fn to_json(&self) -> String {
        let items: Vec<String> = self
            .entries
            .iter()
            .map(|e| format!(r#"{{"name":"{}","stats":{}}}"#, e.name, e.stats.to_json()))
            .collect();
        format!("[{}]", items.join(","))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stopwatch_basic() {
        let mut sw = Stopwatch::new();
        assert!(!sw.is_running());

        sw.start();
        assert!(sw.is_running());

        std::thread::sleep(Duration::from_millis(10));
        sw.stop();

        assert!(!sw.is_running());
        assert!(sw.elapsed() >= Duration::from_millis(5));
    }

    #[test]
    fn test_stopwatch_lap() {
        let mut sw = Stopwatch::new();
        sw.start();
        std::thread::sleep(Duration::from_millis(5));
        let lap1 = sw.lap();
        assert!(lap1 >= Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(5));
        let _lap2 = sw.lap();
        sw.stop();
        assert_eq!(sw.laps().len(), 2);
    }

    #[test]
    fn test_stopwatch_reset() {
        let mut sw = Stopwatch::new();
        sw.start();
        std::thread::sleep(Duration::from_millis(5));
        sw.stop();
        assert!(sw.elapsed() > Duration::ZERO);
        sw.reset();
        assert_eq!(sw.elapsed(), Duration::ZERO);
        assert!(sw.laps().is_empty());
    }

    #[test]
    fn test_benchmark_fn_basic() {
        let config = BenchmarkConfig::new(2, 20);
        let stats = benchmark_fn(&config, || {
            let mut sum = 0u64;
            for i in 0..100 {
                sum += i;
            }
            sum
        })
        .expect("benchmark_fn should succeed");

        assert_eq!(stats.sample_count, 20);
        assert!(stats.min <= stats.mean);
        assert!(stats.mean <= stats.max);
        assert!(stats.median <= stats.max);
    }

    #[test]
    fn test_benchmark_fn_zero_iterations_error() {
        let config = BenchmarkConfig::new(0, 0);
        let result = benchmark_fn(&config, || 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_benchmark_stats_csv_json() {
        let config = BenchmarkConfig::new(0, 10);
        let stats =
            benchmark_fn(&config, || std::hint::black_box(42)).expect("benchmark should succeed");

        let csv = stats.to_csv_row();
        assert!(csv.contains(','));

        let json = stats.to_json();
        assert!(json.starts_with('{'));
        assert!(json.contains("min_ns"));
    }

    #[test]
    fn test_compare_implementations() {
        let config = BenchmarkConfig::new(2, 30);
        let result = compare_implementations(
            &config,
            || {
                let mut v = 0u64;
                for i in 0..100 {
                    v += i;
                }
                v
            },
            || {
                let mut v = 0u64;
                for i in 0..100 {
                    v += i;
                }
                v
            },
        )
        .expect("compare should succeed");

        assert_eq!(result.stats_a.sample_count, 30);
        assert_eq!(result.stats_b.sample_count, 30);
        // For nearly identical functions, speedup should be close to 1
        assert!(result.speedup > 0.0);
    }

    #[test]
    fn test_throughput_bench() {
        let config = BenchmarkConfig::new(2, 20);
        let result = throughput_bench(
            &config,
            || {
                let v: Vec<u8> = vec![0u8; 1024];
                std::hint::black_box(v);
            },
            Some(1024),
        )
        .expect("throughput bench should succeed");

        assert!(result.ops_per_sec > 0.0);
        assert!(result.bytes_per_sec.is_some());
    }

    #[test]
    fn test_memory_bench() {
        let config = BenchmarkConfig::new(1, 5);
        let result = memory_bench(&config, || {
            let v: Vec<u8> = vec![0u8; 1024 * 1024]; // 1 MiB
            std::hint::black_box(v);
        })
        .expect("memory bench should succeed");

        assert_eq!(result.stats.sample_count, 5);
        // peak_memory_bytes may or may not be available depending on platform
    }

    #[test]
    fn test_benchmark_report() {
        let config = BenchmarkConfig::new(0, 5);
        let stats =
            benchmark_fn(&config, || std::hint::black_box(42)).expect("benchmark should succeed");

        let mut report = BenchmarkReport::new();
        report.add("test_func", stats);

        let csv = report.to_csv();
        assert!(csv.contains("test_func"));
        assert!(csv.contains("min_ns"));

        let json = report.to_json();
        assert!(json.contains("test_func"));
    }

    #[test]
    fn test_approx_two_sided_p() {
        // For t=0, p should be ~1.0
        let p0 = approx_two_sided_p(0.0);
        assert!((p0 - 1.0).abs() < 0.1);

        // For large |t|, p should be very small
        let p_large = approx_two_sided_p(5.0);
        assert!(p_large < 0.001);
    }

    #[test]
    fn test_throughput_display() {
        let config = BenchmarkConfig::new(0, 5);
        let result = throughput_bench(&config, || std::hint::black_box(42), Some(1024))
            .expect("should succeed");

        let display = format!("{result}");
        assert!(display.contains("ops/sec"));
    }

    #[test]
    fn test_stopwatch_resume() {
        let mut sw = Stopwatch::new();
        sw.start();
        std::thread::sleep(Duration::from_millis(5));
        sw.stop();
        let e1 = sw.elapsed();

        sw.start(); // resume
        std::thread::sleep(Duration::from_millis(5));
        sw.stop();
        let e2 = sw.elapsed();

        assert!(e2 >= e1);
    }
}
