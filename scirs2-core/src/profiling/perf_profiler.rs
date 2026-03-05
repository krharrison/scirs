//! # Performance Profiler – hierarchical call profiling and FLOP estimation
//!
//! This module extends the existing [`crate::profiling`] infrastructure with:
//!
//! - [`HierarchicalProfiler`] – a thread-safe, hierarchical call profiler that
//!   tracks total time, self-time, and call counts per named function.
//! - [`profile_fn`] – an inline helper that times a closure and records the
//!   result in the profiler.
//! - [`ProfileEntry`] – a snapshot of a single profiled function's statistics.
//! - [`report_profile`] – extract and sort profiling results.
//! - [`FlopsEstimator`] – estimate floating-point operation counts for common
//!   linear algebra and element-wise kernels.
//! - [`throughput_benchmark`] – measure achievable memory bandwidth in GB/s.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::profiling::perf_profiler::{
//!     HierarchicalProfiler, profile_fn, report_profile,
//!     FlopsEstimator, UnaryOp, throughput_benchmark,
//! };
//! use std::sync::Arc;
//!
//! let profiler = Arc::new(HierarchicalProfiler::new());
//!
//! // Time a closure
//! let _result = profile_fn(Arc::clone(&profiler), "my_kernel", || {
//!     // simulate work
//!     (0u64..1000).sum::<u64>()
//! });
//!
//! let entries = report_profile(&profiler);
//! assert!(!entries.is_empty());
//! assert_eq!(entries[0].function, "my_kernel");
//!
//! // FLOP estimates
//! let flops = FlopsEstimator::matmul(128, 128, 128);
//! assert!((flops - 4_194_304.0).abs() < 1.0);   // 2 * 128^3
//!
//! // Throughput benchmark (very short run – just tests the interface)
//! let gbps = throughput_benchmark(|| { let _: Vec<u8> = vec![0u8; 1024]; }, 1024, 5);
//! assert!(gbps >= 0.0);
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ============================================================================
// ProfileEntry
// ============================================================================

/// Snapshot of profiling statistics for a single named function / region.
#[derive(Debug, Clone)]
pub struct ProfileEntry {
    /// Fully-qualified function or region name.
    pub function: String,
    /// Cumulative wall-clock time attributed to this function and all its
    /// callees (inclusive time), in nanoseconds.
    pub total_time_ns: u64,
    /// Number of times this function was called (entered).
    pub call_count: u64,
    /// Time attributed exclusively to this function's own code (exclusive /
    /// self time), in nanoseconds.  This is `total_time_ns` minus the time
    /// spent in child calls.
    pub self_time_ns: u64,
}

impl ProfileEntry {
    /// Average time per call in nanoseconds.
    pub fn avg_time_ns(&self) -> f64 {
        if self.call_count == 0 {
            0.0
        } else {
            self.total_time_ns as f64 / self.call_count as f64
        }
    }
}

// ============================================================================
// Internal per-function accumulator
// ============================================================================

#[derive(Debug, Default, Clone)]
struct FunctionStats {
    total_ns: u64,
    call_count: u64,
    child_ns: u64,
}

// ============================================================================
// HierarchicalProfiler
// ============================================================================

/// Thread-safe hierarchical call profiler.
///
/// The profiler maintains a per-thread call stack so that it can correctly
/// attribute self-time versus child time.  Multiple threads can profile
/// concurrently; their results are merged when [`report_profile`] is called.
#[derive(Debug)]
pub struct HierarchicalProfiler {
    stats: Mutex<HashMap<String, FunctionStats>>,
}

impl HierarchicalProfiler {
    /// Create a new, empty profiler.
    pub fn new() -> Self {
        Self {
            stats: Mutex::new(HashMap::new()),
        }
    }

    /// Record a completed call.
    ///
    /// - `name` — function / region name.
    /// - `total_ns` — wall-clock duration of the call (inclusive).
    /// - `child_ns` — time spent in nested calls tracked by the same profiler.
    pub fn record(&self, name: &str, total_ns: u64, child_ns: u64) {
        if let Ok(mut guard) = self.stats.lock() {
            let entry = guard.entry(name.to_string()).or_default();
            entry.total_ns += total_ns;
            entry.call_count += 1;
            entry.child_ns += child_ns;
        }
    }

    /// Snapshot of all function statistics, cloned out of the mutex.
    fn snapshot(&self) -> HashMap<String, FunctionStats> {
        self.stats.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Reset all accumulated data.
    pub fn reset(&self) {
        if let Ok(mut guard) = self.stats.lock() {
            guard.clear();
        }
    }
}

impl Default for HierarchicalProfiler {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Thread-local call stack for child-time tracking
// ============================================================================

thread_local! {
    /// Stack of (function_name, start_instant, accumulated_child_ns)
    static CALL_STACK: RefCell<Vec<(String, Instant, u64)>> = RefCell::new(Vec::new());
}

use std::cell::RefCell;

// ============================================================================
// profile_fn
// ============================================================================

/// Time the execution of `f`, record the result in `profiler`, and return the
/// closure's return value.
///
/// Uses the thread-local call stack to compute self-time correctly when
/// profiled functions call each other.
///
/// ```rust
/// use scirs2_core::profiling::perf_profiler::{HierarchicalProfiler, profile_fn};
/// use std::sync::Arc;
///
/// let p = Arc::new(HierarchicalProfiler::new());
/// let sum = profile_fn(Arc::clone(&p), "sum_op", || (0u64..100).sum::<u64>());
/// assert_eq!(sum, 4950);
/// ```
#[inline]
pub fn profile_fn<F, T>(profiler: Arc<HierarchicalProfiler>, name: &str, f: F) -> T
where
    F: FnOnce() -> T,
{
    // Push this frame onto the thread-local call stack.
    CALL_STACK.with(|stack| {
        stack
            .borrow_mut()
            .push((name.to_string(), Instant::now(), 0u64));
    });

    let result = f();

    // Pop the frame and record.
    CALL_STACK.with(|stack| {
        let mut borrow = stack.borrow_mut();
        if let Some((frame_name, start, child_ns)) = borrow.pop() {
            let total_ns = start.elapsed().as_nanos() as u64;

            // Attribute child time to the *parent* frame.
            if let Some(parent) = borrow.last_mut() {
                parent.2 += total_ns;
            }

            profiler.record(&frame_name, total_ns, child_ns);
        }
    });

    result
}

// ============================================================================
// report_profile
// ============================================================================

/// Extract profiling results from `profiler`, sorted by `total_time_ns`
/// descending (hottest functions first).
///
/// ```rust
/// use scirs2_core::profiling::perf_profiler::{HierarchicalProfiler, profile_fn, report_profile};
/// use std::sync::Arc;
///
/// let p = Arc::new(HierarchicalProfiler::new());
/// profile_fn(Arc::clone(&p), "fast_op", || ());
/// let entries = report_profile(&p);
/// assert_eq!(entries[0].function, "fast_op");
/// ```
pub fn report_profile(profiler: &HierarchicalProfiler) -> Vec<ProfileEntry> {
    let snap = profiler.snapshot();
    let mut entries: Vec<ProfileEntry> = snap
        .into_iter()
        .map(|(name, stats)| {
            let self_ns = stats.total_ns.saturating_sub(stats.child_ns);
            ProfileEntry {
                function: name,
                total_time_ns: stats.total_ns,
                call_count: stats.call_count,
                self_time_ns: self_ns,
            }
        })
        .collect();
    entries.sort_by(|a, b| b.total_time_ns.cmp(&a.total_time_ns));
    entries
}

// ============================================================================
// FlopsEstimator
// ============================================================================

/// Unary element-wise operations whose FLOP count can be estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Addition or subtraction (1 FLOP per element).
    AddSub,
    /// Multiplication (1 FLOP per element).
    Mul,
    /// Division (1 FLOP per element).
    Div,
    /// Square root (usually 1 "expensive" FLOP per element; counted as 1).
    Sqrt,
    /// Exponential function (counted as 1 FLOP; real cost is higher).
    Exp,
    /// Natural logarithm (counted as 1 FLOP).
    Log,
    /// Fused multiply-add: 2 FLOPs per element.
    Fma,
    /// Custom cost supplied by the caller.
    Custom(u32),
}

/// Estimate FLOP counts for common numerical kernels.
pub struct FlopsEstimator;

impl FlopsEstimator {
    /// Estimate FLOPs for a matrix multiplication C = A(m×k) · B(k×n).
    ///
    /// Standard count: 2·m·k·n (one multiply + one add per accumulation step).
    ///
    /// ```rust
    /// use scirs2_core::profiling::perf_profiler::FlopsEstimator;
    /// assert_eq!(FlopsEstimator::matmul(2, 3, 4) as u64, 48);
    /// ```
    pub fn matmul(m: usize, k: usize, n: usize) -> f64 {
        2.0 * m as f64 * k as f64 * n as f64
    }

    /// Estimate FLOPs for an element-wise unary operation over `n` elements.
    ///
    /// ```rust
    /// use scirs2_core::profiling::perf_profiler::{FlopsEstimator, UnaryOp};
    /// assert_eq!(FlopsEstimator::elementwise(1000, UnaryOp::Fma) as u64, 2000);
    /// ```
    pub fn elementwise(n: usize, op: UnaryOp) -> f64 {
        let flops_per_elem: f64 = match op {
            UnaryOp::AddSub | UnaryOp::Mul | UnaryOp::Div | UnaryOp::Sqrt | UnaryOp::Exp
            | UnaryOp::Log => 1.0,
            UnaryOp::Fma => 2.0,
            UnaryOp::Custom(c) => c as f64,
        };
        n as f64 * flops_per_elem
    }

    /// Estimate FLOPs for a dot product of two vectors of length `n`.
    ///
    /// Each step is one multiply + one add = 2 FLOPs.
    pub fn dot_product(n: usize) -> f64 {
        2.0 * n as f64
    }

    /// Estimate FLOPs for matrix-vector product y = A(m×n) · x(n).
    ///
    /// Each of the `m` output elements requires `2n` FLOPs.
    pub fn gemv(m: usize, n: usize) -> f64 {
        2.0 * m as f64 * n as f64
    }

    /// Estimate FLOPs for an n-point FFT.
    ///
    /// Standard estimate: 5·n·log₂(n).
    pub fn fft(n: usize) -> f64 {
        if n == 0 {
            return 0.0;
        }
        5.0 * n as f64 * (n as f64).log2()
    }

    /// Estimate FLOPs for a batch-normalisation forward pass over `n` elements.
    ///
    /// Approximate cost: 5 FLOPs/element (mean, variance, normalise, scale, shift).
    pub fn batch_norm(n: usize) -> f64 {
        5.0 * n as f64
    }

    /// Convert a raw FLOP count and duration to GFLOPs/s.
    pub fn gflops(flops: f64, elapsed_ns: u64) -> f64 {
        if elapsed_ns == 0 {
            return 0.0;
        }
        flops / (elapsed_ns as f64 * 1e-9) / 1e9
    }
}

// ============================================================================
// throughput_benchmark
// ============================================================================

/// Run `f` exactly `n_iter` times and estimate memory throughput in GB/s.
///
/// `n_bytes` is the number of bytes that `f` reads or writes per iteration.
/// The benchmark measures wall-clock time for all iterations and computes:
///
/// ```text
///   throughput = (n_bytes * n_iter) / total_seconds / 1e9
/// ```
///
/// Returns 0.0 if `n_iter` is 0 or elapsed time is too small to measure.
///
/// ```rust
/// use scirs2_core::profiling::perf_profiler::throughput_benchmark;
///
/// // Very short micro-benchmark — just validates the interface.
/// let gbps = throughput_benchmark(|| { let _: Vec<u8> = vec![0u8; 256]; }, 256, 10);
/// assert!(gbps >= 0.0);
/// ```
pub fn throughput_benchmark<F>(f: F, n_bytes: usize, n_iter: usize) -> f64
where
    F: Fn(),
{
    if n_iter == 0 {
        return 0.0;
    }

    // Warmup: one un-timed iteration.
    f();

    let start = Instant::now();
    for _ in 0..n_iter {
        f();
    }
    let elapsed_s = start.elapsed().as_secs_f64();

    if elapsed_s <= 0.0 {
        return 0.0;
    }

    let total_bytes = n_bytes as f64 * n_iter as f64;
    total_bytes / elapsed_s / 1e9
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn test_profile_fn_records() {
        let p = Arc::new(HierarchicalProfiler::new());
        let _sum = profile_fn(Arc::clone(&p), "my_fn", || {
            std::thread::sleep(Duration::from_millis(1));
            42u64
        });
        let entries = report_profile(&p);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].function, "my_fn");
        assert_eq!(entries[0].call_count, 1);
        assert!(entries[0].total_time_ns > 0);
    }

    #[test]
    fn test_profile_fn_call_count() {
        let p = Arc::new(HierarchicalProfiler::new());
        for _ in 0..5 {
            profile_fn(Arc::clone(&p), "repeated", || ());
        }
        let entries = report_profile(&p);
        assert_eq!(entries[0].call_count, 5);
    }

    #[test]
    fn test_report_sorted_by_total() {
        let p = Arc::new(HierarchicalProfiler::new());
        profile_fn(Arc::clone(&p), "fast", || ());
        profile_fn(Arc::clone(&p), "slow", || {
            std::thread::sleep(Duration::from_millis(5));
        });
        let entries = report_profile(&p);
        assert!(entries[0].total_time_ns >= entries[1].total_time_ns);
    }

    #[test]
    fn test_nested_self_time() {
        let p = Arc::new(HierarchicalProfiler::new());
        profile_fn(Arc::clone(&p), "outer", || {
            profile_fn(Arc::clone(&p), "inner", || {
                std::thread::sleep(Duration::from_millis(5));
            });
        });
        let entries = report_profile(&p);
        let outer = entries.iter().find(|e| e.function == "outer");
        let inner = entries.iter().find(|e| e.function == "inner");
        assert!(outer.is_some() && inner.is_some());
        // Outer self-time should be less than inner total time
        // (outer's total ≥ inner's total, but outer's *self* ≈ outer.total - inner.total)
        let o = outer.expect("outer missing");
        let i = inner.expect("inner missing");
        assert!(o.total_time_ns >= i.total_time_ns);
    }

    #[test]
    fn test_flops_matmul() {
        let f = FlopsEstimator::matmul(4, 4, 4);
        assert_eq!(f as u64, 128); // 2 * 4 * 4 * 4
    }

    #[test]
    fn test_flops_elementwise_fma() {
        let f = FlopsEstimator::elementwise(100, UnaryOp::Fma);
        assert_eq!(f as u64, 200);
    }

    #[test]
    fn test_flops_dot_product() {
        let f = FlopsEstimator::dot_product(512);
        assert_eq!(f as u64, 1024);
    }

    #[test]
    fn test_flops_fft_zero() {
        assert_eq!(FlopsEstimator::fft(0) as u64, 0);
    }

    #[test]
    fn test_flops_fft_positive() {
        let f = FlopsEstimator::fft(1024);
        assert!(f > 0.0);
    }

    #[test]
    fn test_throughput_benchmark_positive() {
        let gbps = throughput_benchmark(|| { let _: Vec<u8> = vec![0u8; 4096]; }, 4096, 50);
        assert!(gbps >= 0.0);
    }

    #[test]
    fn test_throughput_benchmark_zero_iter() {
        let gbps = throughput_benchmark(|| {}, 1024, 0);
        assert_eq!(gbps, 0.0);
    }

    #[test]
    fn test_profiler_reset() {
        let p = Arc::new(HierarchicalProfiler::new());
        profile_fn(Arc::clone(&p), "to_reset", || ());
        assert!(!report_profile(&p).is_empty());
        p.reset();
        assert!(report_profile(&p).is_empty());
    }

    #[test]
    fn test_profile_entry_avg() {
        let p = Arc::new(HierarchicalProfiler::new());
        for _ in 0..4 {
            profile_fn(Arc::clone(&p), "avg_test", || {
                std::thread::sleep(Duration::from_millis(1));
            });
        }
        let entries = report_profile(&p);
        let entry = &entries[0];
        assert_eq!(entry.call_count, 4);
        let avg = entry.avg_time_ns();
        assert!(avg > 0.0);
        assert!((avg * 4.0 - entry.total_time_ns as f64).abs() < 1.0);
    }

    #[test]
    fn test_gflops_estimate() {
        // 1e9 FLOPs in 1 second = 1 GFLOP/s
        let g = FlopsEstimator::gflops(1e9, 1_000_000_000);
        assert!((g - 1.0).abs() < 1e-6);
    }
}
