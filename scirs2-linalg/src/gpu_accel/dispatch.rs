//! Adaptive CPU/GPU dispatch for matrix operations
//!
//! This module provides high-level wrappers that automatically select between a
//! naive CPU path (for small matrices) and the tiled GPU-style kernel (for large
//! matrices) based on a configurable operation-count threshold.  It also exposes
//! utility primitives (`gpu_transpose`, `gpu_axpy`) that complement GEMM in a
//! typical linear algebra workload.
//!
//! ## Dispatch heuristic
//!
//! The dispatcher counts the total number of floating-point multiply-add
//! operations (`m * n * k`) in a GEMM and switches to the tiled GPU kernel
//! when that product exceeds [`GpuMatrixConfig::gpu_threshold`] (default 10⁶).
//! Below the threshold the function falls through to a simple three-loop
//! reference implementation that avoids tile-size alignment overhead and
//! reduces instruction overhead for small problems.

use crate::gpu_accel::gemm::tiled_gemm_f64;
use crate::gpu_accel::types::{GpuBackendKind, GpuError, GpuMatrixConfig, GpuResult};

// ─── Adaptive GEMM dispatcher ─────────────────────────────────────────────────

/// Dispatcher that encapsulates backend selection logic.
///
/// Create an instance with [`GpuDispatcher::new`], then call
/// [`GpuDispatcher::dispatch_gemm`] (or the convenience free functions).
#[derive(Clone, Debug, Default)]
pub struct GpuDispatcher {
    config: GpuMatrixConfig,
}

impl GpuDispatcher {
    /// Create a dispatcher from a [`GpuMatrixConfig`].
    pub fn new(config: GpuMatrixConfig) -> Self {
        Self { config }
    }

    /// Determine which backend would be used for a GEMM of size `m × k × n`.
    ///
    /// Returns the resolved [`GpuBackendKind`] — useful for diagnostics.
    pub fn resolve_backend(&self, m: usize, n: usize, k: usize) -> GpuBackendKind {
        let ops = m.saturating_mul(n).saturating_mul(k);
        if ops < self.config.gpu_threshold {
            GpuBackendKind::Cpu
        } else {
            match self.config.backend {
                GpuBackendKind::OxiBlasGpu => {
                    // OxiBLAS GPU integration hook — falls back to Simulated
                    // until the GPU runtime is wired in.
                    #[cfg(any(
                        feature = "cuda",
                        feature = "opencl",
                        feature = "rocm",
                        feature = "metal"
                    ))]
                    {
                        GpuBackendKind::OxiBlasGpu
                    }
                    #[cfg(not(any(
                        feature = "cuda",
                        feature = "opencl",
                        feature = "rocm",
                        feature = "metal"
                    )))]
                    {
                        GpuBackendKind::Simulated
                    }
                }
                other => other,
            }
        }
    }

    /// Run adaptive GEMM: `C = A · B`, selecting backend automatically.
    ///
    /// - `a`: row-major slice of shape `m × k`.
    /// - `b`: row-major slice of shape `k × n`.
    /// - Returns a row-major `Vec<f64>` of shape `m × n`.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::DimensionMismatch`] if slice lengths are inconsistent
    /// with the provided `m`, `n`, `k`.
    pub fn dispatch_gemm(
        &self,
        a: &[f64],
        b: &[f64],
        m: usize,
        n: usize,
        k: usize,
    ) -> GpuResult<Vec<f64>> {
        validate_gemm_dims(a, b, m, n, k)?;

        let backend = self.resolve_backend(m, n, k);
        let mut c = vec![0.0_f64; m * n];

        match backend {
            GpuBackendKind::Cpu => {
                naive_gemm_f64(a, b, &mut c, m, n, k);
            }
            GpuBackendKind::Simulated | GpuBackendKind::OxiBlasGpu => {
                tiled_gemm_f64(a, b, &mut c, m, n, k, 1.0, self.config.tile_size);
            }
        }

        Ok(c)
    }
}

// ─── Free-function convenience wrappers ───────────────────────────────────────

/// Adaptive GEMM: `C = A · B` with automatic backend selection.
///
/// Uses the default [`GpuMatrixConfig`].  For custom tile sizes or explicit
/// backend choice, construct a [`GpuDispatcher`] directly.
///
/// # Errors
///
/// Returns [`GpuError::DimensionMismatch`] if slice lengths do not match
/// the declared dimensions.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::gpu_accel::adaptive_gemm;
/// use scirs2_linalg::gpu_accel::GpuMatrixConfig;
///
/// let a = vec![1.0_f64, 0.0, 0.0, 1.0]; // 2×2 identity
/// let b = vec![3.0_f64, 4.0, 5.0, 6.0];
/// let c = adaptive_gemm(&a, &b, 2, 2, 2, &GpuMatrixConfig::default()).unwrap();
/// assert!((c[0] - 3.0).abs() < 1e-12);
/// ```
pub fn adaptive_gemm(
    a: &[f64],
    b: &[f64],
    m: usize,
    n: usize,
    k: usize,
    config: &GpuMatrixConfig,
) -> GpuResult<Vec<f64>> {
    GpuDispatcher::new(config.clone()).dispatch_gemm(a, b, m, n, k)
}

/// Convenience wrapper: `C = A · B` with default config.
///
/// # Errors
///
/// Returns [`GpuError::DimensionMismatch`] on invalid dimensions.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::gpu_accel::gpu_matmul;
///
/// let a = vec![1.0_f64, 2.0, 3.0, 4.0]; // 2×2
/// let b = vec![0.0_f64, 1.0, 1.0, 0.0]; // swap cols
/// let c = gpu_matmul(&a, &b, 2, 2, 2).unwrap();
/// // [[1,2],[3,4]] * [[0,1],[1,0]] = [[2,1],[4,3]]
/// assert!((c[0] - 2.0).abs() < 1e-12);
/// assert!((c[1] - 1.0).abs() < 1e-12);
/// ```
pub fn gpu_matmul(a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> GpuResult<Vec<f64>> {
    adaptive_gemm(a, b, m, n, k, &GpuMatrixConfig::default())
}

/// Cache-oblivious in-place matrix transpose.
///
/// Transposes a `rows × cols` row-major matrix.  The algorithm recursively
/// splits the larger dimension until tiles fit in cache, delivering good
/// performance for all matrix aspect ratios.
///
/// Returns a new `Vec<f64>` of length `rows * cols` laid out as a `cols × rows`
/// row-major matrix.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::gpu_accel::gpu_transpose;
///
/// // [[1,2,3],[4,5,6]] → [[1,4],[2,5],[3,6]]
/// let a = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let t = gpu_transpose(&a, 2, 3);
/// assert_eq!(t.len(), 6);
/// assert!((t[0] - 1.0).abs() < 1e-12); // (0,0)
/// assert!((t[1] - 4.0).abs() < 1e-12); // (0,1) in transposed = row 1 col 0 in original
/// ```
pub fn gpu_transpose(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; rows * cols];
    // Cache-oblivious recursive transpose
    // src_cols = cols (stride of the input matrix)
    // dst_cols = rows (stride of the transposed output matrix: cols × rows)
    transpose_recursive(a, &mut out, 0, rows, 0, cols, cols, rows);
    out
}

/// BLAS-1 `y += alpha * x`.
///
/// # Panics
///
/// Panics (debug) if `x.len() != y.len()`.  In release builds the shorter
/// length is used silently.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::gpu_accel::gpu_axpy;
///
/// let x = vec![1.0_f64, 2.0, 3.0];
/// let mut y = vec![4.0_f64, 5.0, 6.0];
/// gpu_axpy(2.0, &x, &mut y);
/// assert!((y[0] - 6.0).abs() < 1e-12);
/// assert!((y[1] - 9.0).abs() < 1e-12);
/// assert!((y[2] - 12.0).abs() < 1e-12);
/// ```
pub fn gpu_axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len(), "gpu_axpy: x and y must have equal length");
    let n = x.len().min(y.len());
    for i in 0..n {
        y[i] += alpha * x[i];
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Validate that `a.len() == m*k` and `b.len() == k*n`.
fn validate_gemm_dims(a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> GpuResult<()> {
    let a_expected = m.checked_mul(k).ok_or_else(|| GpuError::SizeOverflow {
        detail: format!("m={m} * k={k} overflows usize"),
    })?;
    let b_expected = k.checked_mul(n).ok_or_else(|| GpuError::SizeOverflow {
        detail: format!("k={k} * n={n} overflows usize"),
    })?;

    if a.len() != a_expected {
        return Err(GpuError::DimensionMismatch {
            expected: a_expected,
            got: a.len(),
            context: format!(
                "adaptive_gemm: A slice should have {a_expected} elements for {m}×{k}"
            ),
        });
    }
    if b.len() != b_expected {
        return Err(GpuError::DimensionMismatch {
            expected: b_expected,
            got: b.len(),
            context: format!(
                "adaptive_gemm: B slice should have {b_expected} elements for {k}×{n}"
            ),
        });
    }
    Ok(())
}

/// Simple three-loop GEMM for small matrices (no tiling overhead).
fn naive_gemm_f64(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for kk in 0..k {
            let a_ik = a[i * k + kk];
            if a_ik == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += a_ik * b[kk * n + j];
            }
        }
    }
}

/// Cache-oblivious recursive transpose.
///
/// Recursively splits whichever dimension is larger until both tile dimensions
/// are ≤ `CACHE_TILE` (32), then copies element-by-element.
fn transpose_recursive(
    src: &[f64],
    dst: &mut [f64],
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
    src_cols: usize, // = original cols
    dst_cols: usize, // = original rows  (output is transposed)
) {
    const CACHE_TILE: usize = 32;
    let rows = row_end - row_start;
    let cols = col_end - col_start;

    if rows <= CACHE_TILE && cols <= CACHE_TILE {
        // Base case: copy tile
        for i in row_start..row_end {
            for j in col_start..col_end {
                // src[i, j] → dst[j, i]
                dst[j * dst_cols + i] = src[i * src_cols + j];
            }
        }
        return;
    }

    if rows >= cols {
        let mid = row_start + rows / 2;
        transpose_recursive(
            src, dst, row_start, mid, col_start, col_end, src_cols, dst_cols,
        );
        transpose_recursive(
            src, dst, mid, row_end, col_start, col_end, src_cols, dst_cols,
        );
    } else {
        let mid = col_start + cols / 2;
        transpose_recursive(
            src, dst, row_start, row_end, col_start, mid, src_cols, dst_cols,
        );
        transpose_recursive(
            src, dst, row_start, row_end, mid, col_end, src_cols, dst_cols,
        );
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive reference GEMM for correctness checks.
    fn ref_gemm(a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
        let mut c = vec![0.0_f64; m * n];
        for i in 0..m {
            for kk in 0..k {
                for j in 0..n {
                    c[i * n + j] += a[i * k + kk] * b[kk * n + j];
                }
            }
        }
        c
    }

    #[test]
    fn test_adaptive_gemm_small_uses_cpu_path() {
        // 2×2×2 → 8 ops, well below the default 10⁶ threshold
        let a = vec![1.0_f64, 0.0, 0.0, 1.0];
        let b = vec![5.0_f64, 6.0, 7.0, 8.0];
        let cfg = GpuMatrixConfig::default();
        let dispatcher = GpuDispatcher::new(cfg);
        assert_eq!(dispatcher.resolve_backend(2, 2, 2), GpuBackendKind::Cpu);
        let c = dispatcher.dispatch_gemm(&a, &b, 2, 2, 2).unwrap();
        assert!((c[0] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_adaptive_gemm_correct_result() {
        // 3×3 known multiplication
        let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b: Vec<f64> = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let expected = ref_gemm(&a, &b, 3, 3, 3);
        let got = adaptive_gemm(&a, &b, 3, 3, 3, &GpuMatrixConfig::default()).unwrap();
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-9, "adaptive_gemm mismatch at {i}");
        }
    }

    #[test]
    fn test_gpu_matmul_rectangular() {
        let m = 3;
        let k = 4;
        let n = 2;
        let a: Vec<f64> = (0..m * k).map(|i| i as f64 + 1.0).collect();
        let b: Vec<f64> = (0..k * n).map(|i| i as f64 * 0.5 + 0.5).collect();
        let expected = ref_gemm(&a, &b, m, n, k);
        let got = gpu_matmul(&a, &b, m, n, k).unwrap();
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-9, "matmul rect mismatch at {i}");
        }
    }

    #[test]
    fn test_gpu_matmul_associativity() {
        // (A*B)*C == A*(B*C) for 4×4 matrices
        let n = 4;
        let a: Vec<f64> = (0..n * n).map(|i| (i + 1) as f64).collect();
        let b: Vec<f64> = (0..n * n).map(|i| ((n * n - i) as f64) * 0.1).collect();
        let c_mat: Vec<f64> = (0..n * n).map(|i| (i % 3) as f64 + 0.5).collect();

        let ab = gpu_matmul(&a, &b, n, n, n).unwrap();
        let abc = gpu_matmul(&ab, &c_mat, n, n, n).unwrap();

        let bc = gpu_matmul(&b, &c_mat, n, n, n).unwrap();
        let abc2 = gpu_matmul(&a, &bc, n, n, n).unwrap();

        for (i, (&v1, &v2)) in abc.iter().zip(abc2.iter()).enumerate() {
            assert!((v1 - v2).abs() < 1e-8, "associativity failure at {i}");
        }
    }

    #[test]
    fn test_gpu_transpose_square() {
        // [[1,2],[3,4]] ᵀ = [[1,3],[2,4]]
        let a = vec![1.0_f64, 2.0, 3.0, 4.0];
        let t = gpu_transpose(&a, 2, 2);
        assert!((t[0] - 1.0).abs() < 1e-12);
        assert!((t[1] - 3.0).abs() < 1e-12);
        assert!((t[2] - 2.0).abs() < 1e-12);
        assert!((t[3] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_gpu_transpose_rectangular() {
        // [[1,2,3],[4,5,6]] (2×3) → [[1,4],[2,5],[3,6]] (3×2)
        let a = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = gpu_transpose(&a, 2, 3);
        assert_eq!(t.len(), 6);
        let expected = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        for (i, (&got, &exp)) in t.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-12, "transpose mismatch at {i}");
        }
    }

    #[test]
    fn test_gpu_transpose_double_returns_original() {
        let n = 10;
        let a: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
        let at = gpu_transpose(&a, n, n);
        let att = gpu_transpose(&at, n, n);
        for (i, (&orig, &roundtrip)) in a.iter().zip(att.iter()).enumerate() {
            assert!((orig - roundtrip).abs() < 1e-12, "double-transpose at {i}");
        }
    }

    #[test]
    fn test_gpu_axpy() {
        let x = vec![1.0_f64, 2.0, 3.0];
        let mut y = vec![10.0_f64, 20.0, 30.0];
        gpu_axpy(3.0, &x, &mut y);
        assert!((y[0] - 13.0).abs() < 1e-12);
        assert!((y[1] - 26.0).abs() < 1e-12);
        assert!((y[2] - 39.0).abs() < 1e-12);
    }

    #[test]
    fn test_gpu_axpy_zero_alpha() {
        let x = vec![99.0_f64; 5];
        let mut y = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        gpu_axpy(0.0, &x, &mut y);
        assert!((y[0] - 1.0).abs() < 1e-12);
        assert!((y[4] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_adaptive_gemm_dimension_mismatch() {
        let a = vec![1.0_f64; 6]; // 2×3
        let b = vec![1.0_f64; 8]; // 4×2 — k mismatch (3 ≠ 4)
        let result = adaptive_gemm(&a, &b, 2, 2, 3, &GpuMatrixConfig::default());
        // b should be 3*2=6, not 8
        assert!(result.is_err());
    }

    #[test]
    fn test_gpu_dispatcher_resolve_backend_small() {
        let cfg = GpuMatrixConfig {
            gpu_threshold: 1_000_000,
            ..Default::default()
        };
        let d = GpuDispatcher::new(cfg);
        assert_eq!(d.resolve_backend(10, 10, 10), GpuBackendKind::Cpu);
    }

    #[test]
    fn test_gpu_dispatcher_resolve_backend_large() {
        let cfg = GpuMatrixConfig {
            backend: GpuBackendKind::Simulated,
            gpu_threshold: 100,
            ..Default::default()
        };
        let d = GpuDispatcher::new(cfg);
        // 20*20*20 = 8000 > 100
        assert_eq!(d.resolve_backend(20, 20, 20), GpuBackendKind::Simulated);
    }

    #[test]
    fn test_gpu_identity_matmul() {
        let n = 5;
        let eye: Vec<f64> = (0..n)
            .flat_map(|i| (0..n).map(move |j| if i == j { 1.0 } else { 0.0 }))
            .collect();
        let b: Vec<f64> = (0..n * n).map(|i| (i as f64) * 1.7 + 0.3).collect();
        let c = gpu_matmul(&eye, &b, n, n, n).unwrap();
        for (i, (&got, &exp)) in c.iter().zip(b.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-12, "identity matmul at {i}");
        }
    }

    #[test]
    fn test_adaptive_gemm_gpu_path_large() {
        // Force GPU (tiled) path by using a threshold of 0
        let cfg = GpuMatrixConfig {
            backend: GpuBackendKind::Simulated,
            gpu_threshold: 0,
            tile_size: 8,
            ..Default::default()
        };
        let m = 16;
        let k = 12;
        let n = 14;
        let a: Vec<f64> = (0..m * k).map(|i| i as f64 * 0.01 + 0.1).collect();
        let b: Vec<f64> = (0..k * n).map(|i| i as f64 * 0.02 + 0.2).collect();
        let expected = ref_gemm(&a, &b, m, n, k);
        let got = adaptive_gemm(&a, &b, m, n, k, &cfg).unwrap();
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-8, "gpu-path mismatch at {i}");
        }
    }
}
