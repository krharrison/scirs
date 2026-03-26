//! Benchmark harness for performance regression detection.
//!
//! This module provides a lightweight, self-contained micro-benchmark harness
//! that records wall-clock timings and can compare a new run against a stored
//! baseline to detect performance regressions.
//!
//! ## Design goals
//!
//! - **No heavy framework dependency** — uses only `std::time::Instant`.
//! - **Serialisable** — baselines can be persisted as JSON files and loaded
//!   back for comparison across CI runs.
//! - **Composable** — individual `BenchmarkResult`s are collected into a
//!   `BenchmarkBaseline`, which can be diffed by the `RegressionDetector`.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::validation::benchmark_harness::{
//!     BenchmarkHarness, BenchmarkBaseline, RegressionDetector,
//! };
//!
//! // Record timings for a function.
//! let mut harness = BenchmarkHarness::new("vec_sum");
//! harness.run(1000, || {
//!     let v: Vec<f64> = (0..1000).map(|i| i as f64).collect();
//!     let _: f64 = v.iter().sum();
//! });
//! let result = harness.finish();
//! println!("mean: {:.0} ns", result.mean_ns);
//!
//! // Build a baseline and check for regressions.
//! let mut baseline = BenchmarkBaseline::new("my_crate");
//! baseline.add(result.clone());
//!
//! let mut current = BenchmarkBaseline::new("my_crate");
//! // Simulate a 5 % slower result.
//! let mut slower = result.clone();
//! slower.mean_ns *= 1.05;
//! current.add(slower);
//!
//! let detector = RegressionDetector::new(0.10); // 10 % threshold
//! assert!(!detector.has_regressions(&baseline, &current));
//! ```

use std::time::Instant;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// BenchmarkResult
// ---------------------------------------------------------------------------

/// Timing statistics for a single named benchmark.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BenchmarkResult {
    /// Human-readable name for this benchmark.
    pub name: String,
    /// Mean wall-clock time per iteration in nanoseconds.
    pub mean_ns: f64,
    /// Standard deviation of per-iteration times in nanoseconds.
    pub std_dev_ns: f64,
    /// Minimum observed time in nanoseconds.
    pub min_ns: u64,
    /// Maximum observed time in nanoseconds.
    pub max_ns: u64,
    /// Number of iterations used to compute these statistics.
    pub num_iterations: usize,
}

impl BenchmarkResult {
    /// Coefficient of variation (std_dev / mean).  Returns 0.0 when mean is
    /// zero.
    pub fn cv(&self) -> f64 {
        if self.mean_ns == 0.0 {
            0.0
        } else {
            self.std_dev_ns / self.mean_ns
        }
    }
}

impl std::fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: mean={:.1}ns ±{:.1}ns (min={}, max={}, n={})",
            self.name, self.mean_ns, self.std_dev_ns, self.min_ns, self.max_ns, self.num_iterations,
        )
    }
}

// ---------------------------------------------------------------------------
// BenchmarkBaseline
// ---------------------------------------------------------------------------

/// A collection of [`BenchmarkResult`]s representing either a stored baseline
/// or a fresh measurement run.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BenchmarkBaseline {
    /// Name of the crate or component being benchmarked.
    pub crate_name: String,
    /// Git commit hash at the time of recording, if available.
    pub git_hash: Option<String>,
    /// ISO 8601 timestamp string (e.g. `"2026-01-01T00:00:00Z"`).
    pub timestamp: String,
    /// Individual benchmark results.
    pub results: Vec<BenchmarkResult>,
}

impl BenchmarkBaseline {
    /// Create an empty baseline for `crate_name`.
    ///
    /// `git_hash` is populated from the `GIT_HASH` environment variable when
    /// present; otherwise `None` is stored.
    pub fn new(crate_name: impl Into<String>) -> Self {
        let git_hash = std::env::var("GIT_HASH").ok();
        // Use a simple static timestamp fallback when chrono is unavailable.
        let timestamp = chrono_timestamp_or_placeholder();
        Self {
            crate_name: crate_name.into(),
            git_hash,
            timestamp,
            results: Vec::new(),
        }
    }

    /// Append a [`BenchmarkResult`] to this baseline.
    pub fn add(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    /// Return a reference to the [`BenchmarkResult`] with the given name, or
    /// `None` if not present.
    pub fn find(&self, name: &str) -> Option<&BenchmarkResult> {
        self.results.iter().find(|r| r.name == name)
    }

    /// Persist the baseline to a JSON file at `path`.
    ///
    /// # Errors
    ///
    /// Propagates I/O and serialisation errors as [`std::io::Error`].
    #[cfg(feature = "serde")]
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self).map_err(|e| std::io::Error::other(e))?;
        std::fs::write(path, json)
    }

    /// Load a baseline from a JSON file at `path`.
    ///
    /// # Errors
    ///
    /// Propagates I/O and deserialisation errors as [`std::io::Error`].
    #[cfg(feature = "serde")]
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(|e| std::io::Error::other(e))
    }

    /// Return a summary line for each result in Markdown table form.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str(&format!("## Benchmark Baseline: `{}`\n\n", self.crate_name));
        if let Some(ref h) = self.git_hash {
            md.push_str(&format!("Git hash: `{h}`  \n"));
        }
        md.push_str(&format!("Timestamp: {}  \n\n", self.timestamp));
        md.push_str("| Benchmark | Mean (ns) | Std Dev (ns) | Min (ns) | Max (ns) | N |\n");
        md.push_str("|-----------|-----------|--------------|----------|----------|---|\n");
        for r in &self.results {
            md.push_str(&format!(
                "| {} | {:.1} | {:.1} | {} | {} | {} |\n",
                r.name, r.mean_ns, r.std_dev_ns, r.min_ns, r.max_ns, r.num_iterations
            ));
        }
        md
    }
}

// ---------------------------------------------------------------------------
// RegressionEntry
// ---------------------------------------------------------------------------

/// Describes a single benchmark that regressed beyond the configured threshold.
#[derive(Debug, Clone)]
pub struct RegressionEntry {
    /// Name of the regressed benchmark.
    pub name: String,
    /// Mean time in the baseline run (nanoseconds).
    pub baseline_mean_ns: f64,
    /// Mean time in the current run (nanoseconds).
    pub current_mean_ns: f64,
    /// Relative change: `(current − baseline) / baseline`.
    ///
    /// A value of `0.10` means the benchmark got 10 % slower.
    pub relative_change: f64,
}

impl RegressionEntry {
    /// How much slower the benchmark became, expressed as a percentage string
    /// (e.g. `"+12.3%"`).
    pub fn change_pct_str(&self) -> String {
        format!("{:+.1}%", self.relative_change * 100.0)
    }
}

// ---------------------------------------------------------------------------
// RegressionDetector
// ---------------------------------------------------------------------------

/// Compares a fresh benchmark run against a stored baseline and identifies
/// benchmarks that regressed beyond the configured relative threshold.
#[derive(Debug, Clone)]
pub struct RegressionDetector {
    /// Minimum relative slowdown before a result is considered a regression.
    ///
    /// E.g. `0.05` means a 5 % slow-down triggers a regression report.
    pub threshold: f64,
}

impl RegressionDetector {
    /// Create a new [`RegressionDetector`] with the given threshold.
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    /// Return all benchmarks present in both `baseline` and `current` whose
    /// mean time increased by more than `self.threshold`.
    pub fn find_regressions(
        &self,
        baseline: &BenchmarkBaseline,
        current: &BenchmarkBaseline,
    ) -> Vec<RegressionEntry> {
        let mut entries = Vec::new();
        for cur in &current.results {
            if let Some(base) = baseline.find(&cur.name) {
                if base.mean_ns > 0.0 {
                    let rel = (cur.mean_ns - base.mean_ns) / base.mean_ns;
                    if rel > self.threshold {
                        entries.push(RegressionEntry {
                            name: cur.name.clone(),
                            baseline_mean_ns: base.mean_ns,
                            current_mean_ns: cur.mean_ns,
                            relative_change: rel,
                        });
                    }
                }
            }
        }
        entries
    }

    /// Returns `true` when at least one regression is found.
    pub fn has_regressions(
        &self,
        baseline: &BenchmarkBaseline,
        current: &BenchmarkBaseline,
    ) -> bool {
        !self.find_regressions(baseline, current).is_empty()
    }

    /// Format the regression list as a Markdown table.
    pub fn report(&self, regressions: &[RegressionEntry]) -> String {
        if regressions.is_empty() {
            return "No performance regressions detected.\n".to_string();
        }
        let mut md = String::new();
        md.push_str("## Performance Regressions\n\n");
        md.push_str(&format!("Threshold: {:.1}%\n\n", self.threshold * 100.0));
        md.push_str("| Benchmark | Baseline (ns) | Current (ns) | Change |\n");
        md.push_str("|-----------|---------------|--------------|--------|\n");
        for r in regressions {
            md.push_str(&format!(
                "| {} | {:.1} | {:.1} | {} |\n",
                r.name,
                r.baseline_mean_ns,
                r.current_mean_ns,
                r.change_pct_str(),
            ));
        }
        md
    }
}

// ---------------------------------------------------------------------------
// BenchmarkHarness
// ---------------------------------------------------------------------------

/// Simple timing harness.
///
/// Call [`BenchmarkHarness::run`] with a closure to record per-iteration
/// nanosecond timings, then call [`BenchmarkHarness::finish`] to get a
/// [`BenchmarkResult`] with mean, standard deviation, min, and max.
pub struct BenchmarkHarness {
    name: String,
    /// Raw per-iteration timings in nanoseconds.
    timings: Vec<u64>,
}

impl BenchmarkHarness {
    /// Create a new harness for a benchmark named `name`.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            timings: Vec::new(),
        }
    }

    /// Execute `f` for exactly `iterations` iterations, recording the
    /// wall-clock time of each call.
    ///
    /// Calling [`Self::run`] multiple times discards previous timings.
    pub fn run(&mut self, iterations: usize, f: impl Fn()) -> &mut Self {
        self.timings.clear();
        self.timings.reserve(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            f();
            self.timings.push(start.elapsed().as_nanos() as u64);
        }
        self
    }

    /// Compute summary statistics from the recorded timings and return a
    /// [`BenchmarkResult`].
    ///
    /// Returns a zeroed result when no timings have been recorded.
    pub fn finish(&self) -> BenchmarkResult {
        if self.timings.is_empty() {
            return BenchmarkResult {
                name: self.name.clone(),
                mean_ns: 0.0,
                std_dev_ns: 0.0,
                min_ns: 0,
                max_ns: 0,
                num_iterations: 0,
            };
        }

        let n = self.timings.len();
        let sum: u64 = self.timings.iter().sum();
        let mean = sum as f64 / n as f64;

        let variance = self
            .timings
            .iter()
            .map(|&t| {
                let diff = t as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / n as f64;
        let std_dev = variance.sqrt();

        let min = *self.timings.iter().min().unwrap_or(&0);
        let max = *self.timings.iter().max().unwrap_or(&0);

        BenchmarkResult {
            name: self.name.clone(),
            mean_ns: mean,
            std_dev_ns: std_dev,
            min_ns: min,
            max_ns: max,
            num_iterations: n,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Return an ISO 8601-like timestamp using [`chrono`] when it is compiled in,
/// or a static placeholder otherwise.
fn chrono_timestamp_or_placeholder() -> String {
    // chrono is always available as a workspace dep in scirs2-core.
    use chrono::Utc;
    Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // BenchmarkHarness
    // -----------------------------------------------------------------------

    #[test]
    fn test_harness_records_timings() {
        let mut h = BenchmarkHarness::new("noop");
        h.run(50, || {});
        let r = h.finish();
        assert_eq!(r.name, "noop");
        assert_eq!(r.num_iterations, 50);
        assert!(r.mean_ns >= 0.0);
        assert!(r.min_ns <= r.max_ns);
    }

    #[test]
    fn test_harness_empty_finish() {
        let h = BenchmarkHarness::new("empty");
        let r = h.finish();
        assert_eq!(r.num_iterations, 0);
        assert_eq!(r.mean_ns, 0.0);
        assert_eq!(r.min_ns, 0);
        assert_eq!(r.max_ns, 0);
    }

    #[test]
    fn test_harness_run_resets_previous_timings() {
        let mut h = BenchmarkHarness::new("reset");
        h.run(200, || {});
        h.run(10, || {});
        let r = h.finish();
        assert_eq!(r.num_iterations, 10);
    }

    #[test]
    fn test_harness_std_dev_nonnegative() {
        let mut h = BenchmarkHarness::new("std_dev");
        h.run(100, || {
            // a tiny bit of work so timings aren't all zero
            let _: u64 = (0_u64..100).sum();
        });
        let r = h.finish();
        assert!(r.std_dev_ns >= 0.0);
    }

    #[test]
    fn test_harness_cv_zero_mean() {
        let r = BenchmarkResult {
            name: "x".to_string(),
            mean_ns: 0.0,
            std_dev_ns: 0.0,
            min_ns: 0,
            max_ns: 0,
            num_iterations: 1,
        };
        assert_eq!(r.cv(), 0.0);
    }

    // -----------------------------------------------------------------------
    // BenchmarkBaseline
    // -----------------------------------------------------------------------

    #[test]
    fn test_baseline_add_and_find() {
        let mut bl = BenchmarkBaseline::new("my_crate");
        let r = BenchmarkResult {
            name: "foo".to_string(),
            mean_ns: 100.0,
            std_dev_ns: 5.0,
            min_ns: 90,
            max_ns: 120,
            num_iterations: 50,
        };
        bl.add(r.clone());
        let found = bl.find("foo");
        assert!(found.is_some());
        assert!((found.unwrap().mean_ns - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_baseline_find_missing() {
        let bl = BenchmarkBaseline::new("x");
        assert!(bl.find("nonexistent").is_none());
    }

    #[test]
    fn test_baseline_markdown_format() {
        let mut bl = BenchmarkBaseline::new("scirs2-test");
        bl.add(BenchmarkResult {
            name: "bench_a".to_string(),
            mean_ns: 250.5,
            std_dev_ns: 12.3,
            min_ns: 220,
            max_ns: 310,
            num_iterations: 100,
        });
        let md = bl.to_markdown();
        assert!(md.contains("## Benchmark Baseline: `scirs2-test`"));
        assert!(md.contains("bench_a"));
        assert!(md.contains("250.5"));
    }

    // -----------------------------------------------------------------------
    // Save/load roundtrip (requires serde feature)
    // -----------------------------------------------------------------------

    #[cfg(feature = "serde")]
    #[test]
    fn test_baseline_save_load_roundtrip() {
        use std::env::temp_dir;

        let mut bl = BenchmarkBaseline::new("roundtrip_crate");
        bl.git_hash = Some("abc123".to_string());
        bl.add(BenchmarkResult {
            name: "test_bench".to_string(),
            mean_ns: 42.0,
            std_dev_ns: 1.5,
            min_ns: 38,
            max_ns: 50,
            num_iterations: 200,
        });

        let path = temp_dir().join("scirs2_core_benchmark_baseline_test.json");
        bl.save(&path).expect("save failed");

        let loaded = BenchmarkBaseline::load(&path).expect("load failed");
        assert_eq!(loaded.crate_name, "roundtrip_crate");
        assert_eq!(loaded.git_hash.as_deref(), Some("abc123"));
        assert_eq!(loaded.results.len(), 1);
        assert!((loaded.results[0].mean_ns - 42.0).abs() < 1e-9);

        // cleanup
        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // RegressionDetector
    // -----------------------------------------------------------------------

    fn make_baseline(name: &str, mean_ns: f64) -> BenchmarkBaseline {
        let mut bl = BenchmarkBaseline::new("test");
        bl.add(BenchmarkResult {
            name: name.to_string(),
            mean_ns,
            std_dev_ns: 1.0,
            min_ns: (mean_ns * 0.9) as u64,
            max_ns: (mean_ns * 1.1) as u64,
            num_iterations: 100,
        });
        bl
    }

    #[test]
    fn test_regression_none_when_no_change() {
        let baseline = make_baseline("bench", 100.0);
        let current = make_baseline("bench", 100.0);
        let det = RegressionDetector::new(0.05);
        assert!(!det.has_regressions(&baseline, &current));
        assert!(det.find_regressions(&baseline, &current).is_empty());
    }

    #[test]
    fn test_regression_detected_above_threshold() {
        let baseline = make_baseline("bench", 100.0);
        let current = make_baseline("bench", 115.0); // +15 %
        let det = RegressionDetector::new(0.10); // 10 % threshold
        assert!(det.has_regressions(&baseline, &current));
        let regs = det.find_regressions(&baseline, &current);
        assert_eq!(regs.len(), 1);
        assert!((regs[0].relative_change - 0.15).abs() < 1e-9);
    }

    #[test]
    fn test_regression_not_detected_below_threshold() {
        let baseline = make_baseline("bench", 100.0);
        let current = make_baseline("bench", 104.0); // +4 %
        let det = RegressionDetector::new(0.10); // 10 % threshold
        assert!(!det.has_regressions(&baseline, &current));
    }

    #[test]
    fn test_regression_improvement_not_flagged() {
        // A benchmark that got faster should never be flagged.
        let baseline = make_baseline("bench", 100.0);
        let current = make_baseline("bench", 80.0); // 20 % faster
        let det = RegressionDetector::new(0.05);
        assert!(!det.has_regressions(&baseline, &current));
    }

    #[test]
    fn test_regression_missing_benchmark_skipped() {
        let baseline = make_baseline("bench_a", 100.0);
        let current = make_baseline("bench_b", 200.0); // different name
        let det = RegressionDetector::new(0.05);
        assert!(!det.has_regressions(&baseline, &current));
    }

    #[test]
    fn test_regression_report_empty() {
        let det = RegressionDetector::new(0.05);
        let report = det.report(&[]);
        assert!(report.contains("No performance regressions"));
    }

    #[test]
    fn test_regression_report_markdown() {
        let det = RegressionDetector::new(0.10);
        let entries = vec![RegressionEntry {
            name: "slow_bench".to_string(),
            baseline_mean_ns: 100.0,
            current_mean_ns: 150.0,
            relative_change: 0.5,
        }];
        let report = det.report(&entries);
        assert!(report.contains("## Performance Regressions"));
        assert!(report.contains("slow_bench"));
        assert!(report.contains("+50.0%"));
    }
}
