//! # GPU-Agnostic Performance Profiling Module
//!
//! This module provides GPU-agnostic performance profiling infrastructure for tracking
//! kernel execution times, estimating memory bandwidth, counting FLOPS, and generating
//! HTML reports. Since actual GPU instrumentation requires hardware-specific APIs, this
//! module uses wall-clock timing as a portable proxy that works across CPU, GPU, and
//! simulated GPU backends.
//!
//! ## Features
//!
//! - Named kernel start/stop timing via wall-clock
//! - Memory bandwidth estimation from transfer size and elapsed time
//! - FLOPS counting for common operations (matmul, elementwise, reduction)
//! - HTML report generation with summary tables and per-kernel breakdowns
//! - Thread-safe: all interior state uses `RwLock` or `Mutex`
//!
//! ## Feature Gate
//!
//! This module is gated behind the `gpu-profiling` feature in `scirs2-core`.
//!
//! ## Example
//!
//! ```rust
//! # #[cfg(feature = "gpu-profiling")]
//! # {
//! use scirs2_core::profiling::gpu_profiler::{GpuProfiler, KernelKind};
//!
//! let profiler = GpuProfiler::new();
//!
//! // Time a kernel
//! profiler.start_kernel("matmul_256x256").expect("start failed");
//! // ... simulate work ...
//! profiler.stop_kernel("matmul_256x256").expect("stop failed");
//!
//! // Record FLOPS for a matrix multiply (M=N=K=256)
//! let flops = 2 * 256usize * 256 * 256; // 2*M*N*K
//! profiler.record_flops("matmul_256x256", flops as u64, KernelKind::MatMul);
//!
//! // Estimate memory bandwidth (1 MiB transfer)
//! profiler.record_memory_transfer("matmul_256x256", 1 << 20);
//!
//! let html = profiler.generate_html_report();
//! assert!(html.contains("matmul_256x256"));
//! # }
//! ```

use std::collections::HashMap;
use std::fmt;
use std::sync::{Mutex, RwLock};
use std::time::{Duration, Instant};

/// Category of GPU kernel for FLOPS accounting and display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelKind {
    /// Matrix multiplication (GEMM-family operations)
    MatMul,
    /// Element-wise operations (add, mul, relu, etc.)
    ElementWise,
    /// Reduction operations (sum, max, mean, etc.)
    Reduction,
    /// Memory copy / data movement
    MemCopy,
    /// Convolution operations
    Convolution,
    /// Attention (softmax + weighted sum)
    Attention,
    /// Custom / catch-all kernel
    Custom,
}

impl fmt::Display for KernelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelKind::MatMul => write!(f, "MatMul"),
            KernelKind::ElementWise => write!(f, "ElementWise"),
            KernelKind::Reduction => write!(f, "Reduction"),
            KernelKind::MemCopy => write!(f, "MemCopy"),
            KernelKind::Convolution => write!(f, "Convolution"),
            KernelKind::Attention => write!(f, "Attention"),
            KernelKind::Custom => write!(f, "Custom"),
        }
    }
}

/// Statistics accumulated for a single named kernel across all invocations.
#[derive(Debug, Clone)]
pub struct KernelStats {
    /// Human-readable kernel name
    pub name: String,
    /// Kernel category
    pub kind: KernelKind,
    /// Number of times this kernel was invoked
    pub invocations: u64,
    /// Total elapsed wall-clock time across all invocations
    pub total_elapsed: Duration,
    /// Minimum elapsed time across all invocations
    pub min_elapsed: Duration,
    /// Maximum elapsed time across all invocations
    pub max_elapsed: Duration,
    /// Total floating-point operations recorded
    pub total_flops: u64,
    /// Total bytes transferred (for bandwidth estimation)
    pub total_bytes_transferred: u64,
}

impl KernelStats {
    fn new(name: &str, kind: KernelKind) -> Self {
        KernelStats {
            name: name.to_string(),
            kind,
            invocations: 0,
            total_elapsed: Duration::ZERO,
            min_elapsed: Duration::MAX,
            max_elapsed: Duration::ZERO,
            total_flops: 0,
            total_bytes_transferred: 0,
        }
    }

    /// Average elapsed time per invocation. Returns `None` if never invoked.
    pub fn average_elapsed(&self) -> Option<Duration> {
        if self.invocations == 0 {
            None
        } else {
            Some(self.total_elapsed / self.invocations as u32)
        }
    }

    /// Estimated TFLOPS throughput (tera-FLOPS per second).
    ///
    /// Returns `None` if no FLOPS or no elapsed time have been recorded.
    pub fn tflops(&self) -> Option<f64> {
        let secs = self.total_elapsed.as_secs_f64();
        if secs == 0.0 || self.total_flops == 0 {
            None
        } else {
            Some(self.total_flops as f64 / secs / 1e12)
        }
    }

    /// Estimated memory bandwidth in GiB/s.
    ///
    /// Returns `None` if no bytes transferred or no elapsed time.
    pub fn bandwidth_gibs(&self) -> Option<f64> {
        let secs = self.total_elapsed.as_secs_f64();
        if secs == 0.0 || self.total_bytes_transferred == 0 {
            None
        } else {
            // bytes / secs / (2^30)
            Some(self.total_bytes_transferred as f64 / secs / (1u64 << 30) as f64)
        }
    }
}

/// An in-progress kernel timing entry (a kernel that has been started but not yet stopped).
struct ActiveKernel {
    start: Instant,
    /// `KernelKind` to assign when the stats entry is created on first invocation.
    kind: KernelKind,
}

/// Error type returned by `GpuProfiler` operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuProfilerError {
    /// A kernel was started twice without being stopped first.
    KernelAlreadyActive(String),
    /// A kernel was stopped without having been started.
    KernelNotActive(String),
    /// The internal lock was poisoned (another thread panicked while holding it).
    LockPoisoned,
}

impl fmt::Display for GpuProfilerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuProfilerError::KernelAlreadyActive(name) => {
                write!(f, "kernel '{}' was already started", name)
            }
            GpuProfilerError::KernelNotActive(name) => {
                write!(f, "kernel '{}' was not active (call start_kernel first)", name)
            }
            GpuProfilerError::LockPoisoned => write!(f, "internal profiler lock was poisoned"),
        }
    }
}

impl std::error::Error for GpuProfilerError {}

/// GPU-agnostic performance profiler.
///
/// Tracks wall-clock time for named kernels, accumulates FLOPS counts and memory
/// transfer sizes, and produces HTML reports. All operations are thread-safe.
///
/// # Notes on Wall-Clock Timing
///
/// Real GPU profiling requires synchronization barriers (e.g., `cudaDeviceSynchronize`)
/// around each kernel to get accurate per-kernel GPU time. This profiler measures
/// *host-side* wall-clock duration instead, which includes kernel launch overhead and
/// any CPU-side serialization. For pure simulation or CPU-fallback backends this is
/// identical to the actual compute time.
pub struct GpuProfiler {
    /// Per-kernel accumulated statistics, keyed by kernel name.
    stats: RwLock<HashMap<String, KernelStats>>,
    /// Currently active (started but not stopped) kernels.
    active: Mutex<HashMap<String, ActiveKernel>>,
    /// Ordered list of kernel names in insertion order (for stable HTML output).
    order: Mutex<Vec<String>>,
    /// Optional profiler-level label (e.g., model name or experiment tag).
    label: String,
}

impl GpuProfiler {
    /// Create a new profiler with an empty state.
    pub fn new() -> Self {
        GpuProfiler {
            stats: RwLock::new(HashMap::new()),
            active: Mutex::new(HashMap::new()),
            order: Mutex::new(Vec::new()),
            label: String::from("GpuProfiler"),
        }
    }

    /// Create a new profiler with a descriptive label shown in the HTML report header.
    pub fn with_label(label: impl Into<String>) -> Self {
        GpuProfiler {
            stats: RwLock::new(HashMap::new()),
            active: Mutex::new(HashMap::new()),
            order: Mutex::new(Vec::new()),
            label: label.into(),
        }
    }

    /// Start timing a named kernel.
    ///
    /// The first time a kernel is started its `KernelKind` is recorded as `Custom`.
    /// Use [`start_kernel_with_kind`](GpuProfiler::start_kernel_with_kind) to specify
    /// the kind on the first invocation.
    ///
    /// # Errors
    ///
    /// Returns [`GpuProfilerError::KernelAlreadyActive`] if the kernel was already started.
    pub fn start_kernel(&self, name: &str) -> Result<(), GpuProfilerError> {
        self.start_kernel_with_kind(name, KernelKind::Custom)
    }

    /// Start timing a named kernel, specifying its [`KernelKind`] on first use.
    ///
    /// The `kind` parameter is only applied when the kernel stats entry is first created.
    /// Subsequent calls with a different kind are silently ignored for the kind field.
    ///
    /// # Errors
    ///
    /// Returns [`GpuProfilerError::KernelAlreadyActive`] if the kernel was already started.
    pub fn start_kernel_with_kind(
        &self,
        name: &str,
        kind: KernelKind,
    ) -> Result<(), GpuProfilerError> {
        let mut active = self
            .active
            .lock()
            .map_err(|_| GpuProfilerError::LockPoisoned)?;
        if active.contains_key(name) {
            return Err(GpuProfilerError::KernelAlreadyActive(name.to_string()));
        }
        active.insert(
            name.to_string(),
            ActiveKernel {
                start: Instant::now(),
                kind,
            },
        );

        // Ensure an entry exists in `order` and `stats` so that the HTML report
        // lists this kernel even if it was never successfully stopped.
        let mut order = self
            .order
            .lock()
            .map_err(|_| GpuProfilerError::LockPoisoned)?;
        let mut stats = self
            .stats
            .write()
            .map_err(|_| GpuProfilerError::LockPoisoned)?;
        if !stats.contains_key(name) {
            stats.insert(name.to_string(), KernelStats::new(name, kind));
            order.push(name.to_string());
        }
        Ok(())
    }

    /// Stop timing a named kernel and record the elapsed duration.
    ///
    /// # Errors
    ///
    /// Returns [`GpuProfilerError::KernelNotActive`] if the kernel was not started.
    pub fn stop_kernel(&self, name: &str) -> Result<Duration, GpuProfilerError> {
        let elapsed = {
            let mut active = self
                .active
                .lock()
                .map_err(|_| GpuProfilerError::LockPoisoned)?;
            let entry = active
                .remove(name)
                .ok_or_else(|| GpuProfilerError::KernelNotActive(name.to_string()))?;
            entry.start.elapsed()
        };

        let mut stats = self
            .stats
            .write()
            .map_err(|_| GpuProfilerError::LockPoisoned)?;
        let entry = stats
            .entry(name.to_string())
            .or_insert_with(|| KernelStats::new(name, KernelKind::Custom));
        entry.invocations += 1;
        entry.total_elapsed += elapsed;
        if elapsed < entry.min_elapsed {
            entry.min_elapsed = elapsed;
        }
        if elapsed > entry.max_elapsed {
            entry.max_elapsed = elapsed;
        }
        Ok(elapsed)
    }

    /// Record a FLOP count for a named kernel.
    ///
    /// This call does **not** require the kernel to be active; it can be used to
    /// annotate after the fact. The `kind` parameter updates the kernel's stored
    /// kind only if an entry already exists (it does not create a new entry).
    pub fn record_flops(&self, name: &str, flops: u64, kind: KernelKind) {
        if let Ok(mut stats) = self.stats.write() {
            let entry = stats
                .entry(name.to_string())
                .or_insert_with(|| KernelStats::new(name, kind));
            entry.total_flops += flops;
            entry.kind = kind;
        }
    }

    /// Record bytes transferred to/from device memory for a named kernel.
    ///
    /// Used to compute estimated memory bandwidth. The value is accumulated
    /// across multiple calls so you can call this once per buffer in a kernel.
    pub fn record_memory_transfer(&self, name: &str, bytes: u64) {
        if let Ok(mut stats) = self.stats.write() {
            let entry = stats
                .entry(name.to_string())
                .or_insert_with(|| KernelStats::new(name, KernelKind::MemCopy));
            entry.total_bytes_transferred += bytes;
        }
    }

    /// Return a snapshot of the accumulated statistics for a specific kernel.
    ///
    /// Returns `None` if the kernel has never been started.
    pub fn kernel_stats(&self, name: &str) -> Option<KernelStats> {
        self.stats.read().ok()?.get(name).cloned()
    }

    /// Return a snapshot of all accumulated statistics, in insertion order.
    pub fn all_stats(&self) -> Vec<KernelStats> {
        let order = match self.order.lock() {
            Ok(o) => o.clone(),
            Err(_) => return Vec::new(),
        };
        let stats = match self.stats.read() {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        order
            .iter()
            .filter_map(|name| stats.get(name).cloned())
            .collect()
    }

    /// Reset all accumulated statistics and clear any active kernels.
    ///
    /// Useful between benchmark runs to start fresh without allocating a new profiler.
    pub fn reset(&self) {
        if let Ok(mut active) = self.active.lock() {
            active.clear();
        }
        if let Ok(mut stats) = self.stats.write() {
            stats.clear();
        }
        if let Ok(mut order) = self.order.lock() {
            order.clear();
        }
    }

    /// Total number of FLOPS recorded across all kernels.
    pub fn total_flops(&self) -> u64 {
        self.stats
            .read()
            .map(|s| s.values().map(|k| k.total_flops).sum())
            .unwrap_or(0)
    }

    /// Total wall-clock time across all kernels.
    pub fn total_elapsed(&self) -> Duration {
        self.stats
            .read()
            .map(|s| s.values().map(|k| k.total_elapsed).sum())
            .unwrap_or(Duration::ZERO)
    }

    /// Generate an HTML performance report string.
    ///
    /// The report includes:
    /// - A header with the profiler label and aggregate totals
    /// - A sortable summary table with per-kernel rows showing invocations,
    ///   total time, average time, estimated TFLOPS, and estimated bandwidth
    pub fn generate_html_report(&self) -> String {
        let all_stats = self.all_stats();
        let total_elapsed = self.total_elapsed();
        let total_flops = self.total_flops();

        let mut html = String::new();

        // ---- HTML boilerplate ----
        html.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
        html.push_str("  <meta charset=\"UTF-8\">\n");
        html.push_str("  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n");
        html.push_str(&format!("  <title>GPU Profiler Report – {}</title>\n", self.label));
        html.push_str("  <style>\n");
        html.push_str(Self::css());
        html.push_str("  </style>\n</head>\n<body>\n");

        // ---- Header ----
        html.push_str(&format!("  <h1>GPU Profiler Report: {}</h1>\n", self.label));
        html.push_str("  <div class=\"summary\">\n");
        html.push_str(&format!(
            "    <p><strong>Total wall-clock time:</strong> {:.3} ms</p>\n",
            total_elapsed.as_secs_f64() * 1000.0
        ));
        html.push_str(&format!(
            "    <p><strong>Total FLOPS recorded:</strong> {:.3} GFLOP</p>\n",
            total_flops as f64 / 1e9
        ));
        html.push_str(&format!(
            "    <p><strong>Unique kernels:</strong> {}</p>\n",
            all_stats.len()
        ));

        // Overall TFLOPS
        let secs = total_elapsed.as_secs_f64();
        if secs > 0.0 && total_flops > 0 {
            html.push_str(&format!(
                "    <p><strong>Aggregate TFLOPS:</strong> {:.4}</p>\n",
                total_flops as f64 / secs / 1e12
            ));
        }
        html.push_str("  </div>\n");

        // ---- Kernel table ----
        html.push_str("  <h2>Per-Kernel Statistics</h2>\n");
        html.push_str("  <table>\n");
        html.push_str("    <thead><tr>");
        for col in &[
            "Kernel",
            "Kind",
            "Invocations",
            "Total (ms)",
            "Avg (ms)",
            "Min (ms)",
            "Max (ms)",
            "TFLOPS",
            "BW (GiB/s)",
        ] {
            html.push_str(&format!("<th>{}</th>", col));
        }
        html.push_str("</tr></thead>\n    <tbody>\n");

        for ks in &all_stats {
            let avg_ms = ks
                .average_elapsed()
                .map(|d| format!("{:.4}", d.as_secs_f64() * 1000.0))
                .unwrap_or_else(|| "–".to_string());

            let min_ms = if ks.min_elapsed == Duration::MAX {
                "–".to_string()
            } else {
                format!("{:.4}", ks.min_elapsed.as_secs_f64() * 1000.0)
            };

            let max_ms = format!("{:.4}", ks.max_elapsed.as_secs_f64() * 1000.0);

            let tflops = ks
                .tflops()
                .map(|t| format!("{:.4}", t))
                .unwrap_or_else(|| "–".to_string());

            let bw = ks
                .bandwidth_gibs()
                .map(|b| format!("{:.4}", b))
                .unwrap_or_else(|| "–".to_string());

            html.push_str("      <tr>");
            html.push_str(&format!("<td class=\"name\">{}</td>", escape_html(&ks.name)));
            html.push_str(&format!("<td>{}</td>", ks.kind));
            html.push_str(&format!("<td>{}</td>", ks.invocations));
            html.push_str(&format!(
                "<td>{:.4}</td>",
                ks.total_elapsed.as_secs_f64() * 1000.0
            ));
            html.push_str(&format!("<td>{}</td>", avg_ms));
            html.push_str(&format!("<td>{}</td>", min_ms));
            html.push_str(&format!("<td>{}</td>", max_ms));
            html.push_str(&format!("<td>{}</td>", tflops));
            html.push_str(&format!("<td>{}</td>", bw));
            html.push_str("</tr>\n");
        }

        html.push_str("    </tbody>\n  </table>\n");

        // ---- Time-share bar chart (text-based) ----
        if !all_stats.is_empty() && total_elapsed.as_nanos() > 0 {
            html.push_str("  <h2>Relative Time Share</h2>\n");
            html.push_str("  <table class=\"barchart\">\n");
            for ks in &all_stats {
                let pct = ks.total_elapsed.as_secs_f64() / total_elapsed.as_secs_f64() * 100.0;
                html.push_str("    <tr>");
                html.push_str(&format!(
                    "<td class=\"bcname\">{}</td>",
                    escape_html(&ks.name)
                ));
                html.push_str(&format!(
                    "<td><div class=\"bar\" style=\"width:{:.1}%\">&nbsp;</div></td>",
                    pct.min(100.0)
                ));
                html.push_str(&format!("<td class=\"pct\">{:.1}%</td>", pct));
                html.push_str("</tr>\n");
            }
            html.push_str("  </table>\n");
        }

        html.push_str("</body>\n</html>\n");
        html
    }

    /// Minimal CSS used in the HTML report.
    fn css() -> &'static str {
        r#"
    body { font-family: monospace; background: #1a1a2e; color: #e0e0e0; padding: 1em 2em; }
    h1 { color: #a6e3ff; border-bottom: 1px solid #333; }
    h2 { color: #7ec8e3; margin-top: 1.5em; }
    .summary { background: #16213e; border-left: 4px solid #0f3460; padding: 0.5em 1em;
               margin-bottom: 1em; border-radius: 4px; }
    table { border-collapse: collapse; width: 100%; margin-top: 0.5em; }
    th { background: #0f3460; color: #a6e3ff; padding: 6px 12px; text-align: left; }
    td { padding: 5px 12px; border-bottom: 1px solid #333; }
    tr:hover { background: #16213e; }
    td.name { font-weight: bold; color: #80ffdb; }
    table.barchart td.bcname { width: 25%; font-size: 0.9em; }
    .bar { background: linear-gradient(90deg, #0f3460, #a6e3ff); height: 18px;
           min-width: 2px; border-radius: 2px; }
    table.barchart td { padding: 3px 8px; }
    td.pct { width: 6em; text-align: right; color: #a6e3ff; }
"#
    }
}

impl Default for GpuProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Escape HTML special characters to prevent injection in the report.
fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ============================================================================
// FLOPS estimation helpers
// ============================================================================

/// Estimate the number of FLOPs for a general matrix multiply C = A*B.
///
/// Uses the formula `2 * M * N * K` (multiply-add counted as 2 FLOP).
#[inline]
pub fn matmul_flops(m: usize, n: usize, k: usize) -> u64 {
    2u64 * m as u64 * n as u64 * k as u64
}

/// Estimate FLOPs for a 2-D convolution forward pass.
///
/// Formula: `2 * Cout * H_out * W_out * Cin * Kh * Kw`.
#[inline]
pub fn conv2d_flops(
    c_out: usize,
    h_out: usize,
    w_out: usize,
    c_in: usize,
    kh: usize,
    kw: usize,
) -> u64 {
    2u64 * c_out as u64 * h_out as u64 * w_out as u64 * c_in as u64 * kh as u64 * kw as u64
}

/// Estimate FLOPs for a scaled dot-product attention layer (single head).
///
/// Forward pass approximation: `4 * seq_len^2 * d_model`.
#[inline]
pub fn attention_flops(seq_len: usize, d_model: usize) -> u64 {
    4u64 * seq_len as u64 * seq_len as u64 * d_model as u64
}

/// Estimate memory bytes for a matrix of shape `[rows, cols]` with `elem_bytes` bytes per element.
#[inline]
pub fn matrix_bytes(rows: usize, cols: usize, elem_bytes: usize) -> u64 {
    rows as u64 * cols as u64 * elem_bytes as u64
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_basic_start_stop() {
        let profiler = GpuProfiler::new();
        profiler.start_kernel("kernel_a").expect("start failed");
        thread::sleep(Duration::from_millis(5));
        let elapsed = profiler.stop_kernel("kernel_a").expect("stop failed");
        assert!(elapsed >= Duration::from_millis(5));

        let stats = profiler.kernel_stats("kernel_a").expect("stats missing");
        assert_eq!(stats.invocations, 1);
        assert!(stats.total_elapsed >= Duration::from_millis(5));
    }

    #[test]
    fn test_multiple_invocations() {
        let profiler = GpuProfiler::new();
        for _ in 0..3 {
            profiler.start_kernel("kernel_b").expect("start");
            profiler.stop_kernel("kernel_b").expect("stop");
        }
        let stats = profiler.kernel_stats("kernel_b").expect("stats");
        assert_eq!(stats.invocations, 3);
        assert!(stats.min_elapsed <= stats.max_elapsed);
    }

    #[test]
    fn test_double_start_error() {
        let profiler = GpuProfiler::new();
        profiler.start_kernel("dup").expect("first start ok");
        let err = profiler.start_kernel("dup").expect_err("should error on double start");
        assert!(matches!(err, GpuProfilerError::KernelAlreadyActive(_)));
        // Clean up
        profiler.stop_kernel("dup").expect("stop ok");
    }

    #[test]
    fn test_stop_without_start_error() {
        let profiler = GpuProfiler::new();
        let err = profiler
            .stop_kernel("ghost")
            .expect_err("should error on stop without start");
        assert!(matches!(err, GpuProfilerError::KernelNotActive(_)));
    }

    #[test]
    fn test_flops_and_bandwidth() {
        let profiler = GpuProfiler::new();
        profiler
            .start_kernel_with_kind("gemm", KernelKind::MatMul)
            .expect("start");
        thread::sleep(Duration::from_millis(2));
        profiler.stop_kernel("gemm").expect("stop");

        let flops = matmul_flops(256, 256, 256);
        profiler.record_flops("gemm", flops, KernelKind::MatMul);
        profiler.record_memory_transfer("gemm", matrix_bytes(256, 256, 4) * 3);

        let stats = profiler.kernel_stats("gemm").expect("stats");
        assert!(stats.total_flops > 0);
        assert!(stats.total_bytes_transferred > 0);
        // TFLOPS should be positive (very small for a test though)
        assert!(stats.tflops().is_some());
        assert!(stats.bandwidth_gibs().is_some());
    }

    #[test]
    fn test_html_report_contains_kernel_name() {
        let profiler = GpuProfiler::with_label("test_experiment");
        profiler.start_kernel("my_custom_kernel").expect("start");
        profiler.stop_kernel("my_custom_kernel").expect("stop");
        let html = profiler.generate_html_report();
        assert!(html.contains("my_custom_kernel"), "kernel name absent from HTML report");
        assert!(html.contains("test_experiment"), "label absent from HTML report");
    }

    #[test]
    fn test_reset_clears_all_state() {
        let profiler = GpuProfiler::new();
        profiler.start_kernel("k1").expect("start");
        profiler.stop_kernel("k1").expect("stop");
        assert_eq!(profiler.all_stats().len(), 1);
        profiler.reset();
        assert_eq!(profiler.all_stats().len(), 0);
        assert_eq!(profiler.total_flops(), 0);
    }

    #[test]
    fn test_matmul_flops_formula() {
        // 2 * M * N * K
        assert_eq!(matmul_flops(4, 4, 4), 2 * 4 * 4 * 4);
    }

    #[test]
    fn test_conv2d_flops_formula() {
        assert_eq!(conv2d_flops(8, 14, 14, 3, 3, 3), 2 * 8 * 14 * 14 * 3 * 3 * 3);
    }

    #[test]
    fn test_all_stats_insertion_order() {
        let profiler = GpuProfiler::new();
        for name in &["alpha", "beta", "gamma"] {
            profiler.start_kernel(name).expect("start");
            profiler.stop_kernel(name).expect("stop");
        }
        let stats = profiler.all_stats();
        assert_eq!(stats[0].name, "alpha");
        assert_eq!(stats[1].name, "beta");
        assert_eq!(stats[2].name, "gamma");
    }

    #[test]
    fn test_html_escape() {
        assert_eq!(escape_html("<script>"), "&lt;script&gt;");
        assert_eq!(escape_html("a & b"), "a &amp; b");
    }
}
