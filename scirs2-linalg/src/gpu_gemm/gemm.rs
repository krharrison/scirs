//! GPU-accelerated GEMM with CPU cache-blocked fallback
//!
//! Implements general matrix multiplication (GEMM) with three-level cache blocking
//! for optimal L1/L2/L3 cache utilization. Falls back gracefully when GPU hardware
//! is unavailable.
//!
//! ## Algorithm: 3-level Cache-Blocked GEMM
//!
//! The classic loop-order for cache efficiency is:
//! ```text
//! for kb in 0..K/Kb:
//!   for mb in 0..M/Mb:
//!     pack A block into L2-local buffer
//!     for nb in 0..N/Nb:
//!       pack B block into L1-local buffer
//!       micro-kernel: update C[mb*Mb..(mb+1)*Mb, nb*Nb..(nb+1)*Nb]
//! ```
//!
//! Block sizes are tuned for typical x86-64 cache hierarchy:
//! - L1 (32 KB): Nb = 64, Mb = 64
//! - L2 (256 KB): Mb = 128
//! - L3 (8 MB): Kb = 256

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array2, Array3, ArrayView2, Axis};

// ─── Cache-blocking constants (tuned for typical x86-64 / AArch64) ─────────
/// L1-cache block size for N dimension (fits two Nb×Kb tiles in L1)
const NB_L1: usize = 64;
/// L1-cache block size for M dimension
const MB_L1: usize = 64;
/// L2-cache block size for the K (contraction) dimension
const KB_L2: usize = 256;

// ─── Public types ────────────────────────────────────────────────────────────

/// Backend selection for GEMM operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GemmBackend {
    /// Pure Rust cache-blocked GEMM (always available).
    CpuNaive,
    /// OxiBLAS CPU GEMM (used when the `linalg` feature of `scirs2-core` is active).
    CpuBlas,
    /// OxiBLAS GPU backend (requires a CUDA/OpenCL/ROCm/Metal feature flag).
    GpuOxiblas,
}

/// Configuration for GEMM operations.
#[derive(Clone, Debug)]
pub struct GemmConfig {
    /// Which backend to use (default: `CpuNaive`).
    pub backend: GemmBackend,
    /// Cache-block size for the inner loops (default: `MB_L1`).
    pub block_size: usize,
    /// Transpose A before multiplying (default: `false`).
    pub transpose_a: bool,
    /// Transpose B before multiplying (default: `false`).
    pub transpose_b: bool,
    /// Scalar α in `C = α·A·B + β·C` (default: `1.0`).
    pub alpha: f64,
    /// Scalar β in `C = α·A·B + β·C` (default: `0.0`).
    pub beta: f64,
}

impl Default for GemmConfig {
    fn default() -> Self {
        Self {
            backend: GemmBackend::CpuNaive,
            block_size: MB_L1,
            transpose_a: false,
            transpose_b: false,
            alpha: 1.0,
            beta: 0.0,
        }
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// General matrix multiply: `C = α·op(A)·op(B) + β·C`.
///
/// - `A`: `[m, k]` (or `[k, m]` if `config.transpose_a = true`)
/// - `B`: `[k, n]` (or `[n, k]` if `config.transpose_b = true`)
/// - `c`: optional initial `[m, n]` matrix; treated as zeros when `None`.
///
/// Returns the `[m, n]` result matrix.
///
/// # Errors
///
/// Returns [`LinalgError::DimensionError`] if the inner dimensions do not match.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::gpu_gemm::{gemm, GemmConfig};
/// use scirs2_core::ndarray::array;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 1.0]]; // identity
/// let b = array![[3.0_f64, 4.0], [5.0, 6.0]];
/// let c = gemm(&a, &b, None, &GemmConfig::default()).unwrap();
/// assert!((c[[0, 0]] - 3.0).abs() < 1e-12);
/// ```
pub fn gemm(
    a: &Array2<f64>,
    b: &Array2<f64>,
    c: Option<&Array2<f64>>,
    config: &GemmConfig,
) -> LinalgResult<Array2<f64>> {
    // Resolve effective A and B after optional transpositions
    let a_eff: Array2<f64>;
    let b_eff: Array2<f64>;

    let a_ref: &Array2<f64> = if config.transpose_a {
        // t() reverses strides producing Fortran order; into_standard_layout
        // ensures the copy is C-contiguous (row-major) for as_slice() to work.
        let rows = a.ncols();
        let cols = a.nrows();
        a_eff = Array2::from_shape_fn((rows, cols), |(i, j)| a[[j, i]]);
        &a_eff
    } else {
        a
    };

    let b_ref: &Array2<f64> = if config.transpose_b {
        let rows = b.ncols();
        let cols = b.nrows();
        b_eff = Array2::from_shape_fn((rows, cols), |(i, j)| b[[j, i]]);
        &b_eff
    } else {
        b
    };

    let (m, k_a) = (a_ref.nrows(), a_ref.ncols());
    let (k_b, n) = (b_ref.nrows(), b_ref.ncols());

    if k_a != k_b {
        return Err(LinalgError::DimensionError(format!(
            "GEMM: inner dimensions must match: A has k={k_a}, B has k={k_b}"
        )));
    }
    let k = k_a;

    // Select backend
    match config.backend {
        GemmBackend::GpuOxiblas => {
            // GPU path: fall back to CPU if no GPU features compiled in.
            #[cfg(any(
                feature = "cuda",
                feature = "opencl",
                feature = "rocm",
                feature = "metal"
            ))]
            {
                // GPU dispatch is performed via the parent gpu::operations module.
                // For now delegate to the cache-blocked CPU path until the full
                // OxiBLAS GPU integration is wired up.
                gemm_cpu_blocked(a_ref, b_ref, c, m, n, k, config)
            }
            #[cfg(not(any(
                feature = "cuda",
                feature = "opencl",
                feature = "rocm",
                feature = "metal"
            )))]
            {
                // No GPU hardware available – fall through to CpuNaive
                gemm_cpu_blocked(a_ref, b_ref, c, m, n, k, config)
            }
        }
        GemmBackend::CpuBlas | GemmBackend::CpuNaive => {
            gemm_cpu_blocked(a_ref, b_ref, c, m, n, k, config)
        }
    }
}

/// Batched GEMM: `C_i = α·A_i·B_i + β·C_i` for every slice in the batch.
///
/// - `a`: `[batch, m, k]`
/// - `b`: `[batch, k, n]`
///
/// Returns `[batch, m, n]`.
///
/// # Errors
///
/// Returns [`LinalgError::DimensionError`] if batch sizes or inner dims mismatch.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::gpu_gemm::{batched_gemm, GemmConfig};
/// use scirs2_core::ndarray::Array3;
///
/// let a = Array3::<f64>::zeros((1, 2, 2));
/// let b = Array3::<f64>::from_elem((1, 2, 2), 3.0);
/// let c = batched_gemm(&a, &b, &GemmConfig::default()).unwrap();
/// assert_eq!(c.shape(), &[1, 2, 2]);
/// ```
pub fn batched_gemm(
    a: &Array3<f64>,
    b: &Array3<f64>,
    config: &GemmConfig,
) -> LinalgResult<Array3<f64>> {
    let (batch_a, m, k_a) = (a.shape()[0], a.shape()[1], a.shape()[2]);
    let (batch_b, k_b, n) = (b.shape()[0], b.shape()[1], b.shape()[2]);

    if batch_a != batch_b {
        return Err(LinalgError::DimensionError(format!(
            "Batched GEMM: batch sizes must match: got {batch_a} and {batch_b}"
        )));
    }
    if k_a != k_b {
        return Err(LinalgError::DimensionError(format!(
            "Batched GEMM: inner dimensions must match: A has k={k_a}, B has k={k_b}"
        )));
    }

    let batch = batch_a;
    let mut result = Array3::<f64>::zeros((batch, m, n));

    for i in 0..batch {
        let a_slice: Array2<f64> = a.index_axis(Axis(0), i).to_owned();
        let b_slice: Array2<f64> = b.index_axis(Axis(0), i).to_owned();
        let c_slice = gemm(&a_slice, &b_slice, None, config)?;
        result.index_axis_mut(Axis(0), i).assign(&c_slice);
    }

    Ok(result)
}

/// Symmetric GEMM: computes `C = A · A^T`.
///
/// The result is always symmetric and positive semi-definite.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::gpu_gemm::{symm_gemm, GemmConfig};
/// use scirs2_core::ndarray::array;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let c = symm_gemm(&a, &GemmConfig::default()).unwrap();
/// // c should equal a * a^T
/// assert!((c[[0, 1]] - c[[1, 0]]).abs() < 1e-12);
/// ```
pub fn symm_gemm(a: &Array2<f64>, config: &GemmConfig) -> LinalgResult<Array2<f64>> {
    let mut cfg = config.clone();
    cfg.transpose_b = true; // Force B = A^T
    gemm(a, a, None, &cfg)
}

// ─── Internal: 3-level cache-blocked GEMM ───────────────────────────────────

/// Cache-blocked GEMM for f64 arrays.
///
/// Blocking order (outer→inner): K, M, N
/// This keeps the packed A-block resident in L2 and the packed B-panel in L1.
fn gemm_cpu_blocked(
    a: &Array2<f64>,
    b: &Array2<f64>,
    c_init: Option<&Array2<f64>>,
    m: usize,
    n: usize,
    k: usize,
    config: &GemmConfig,
) -> LinalgResult<Array2<f64>> {
    // Determine block sizes from config or defaults
    let mb = config.block_size.max(1);
    let nb = mb; // keep square micro-tiles by default
    let kb = KB_L2;

    // Initialise output matrix
    let mut c: Array2<f64> = match c_init {
        Some(c0) => {
            if c0.nrows() != m || c0.ncols() != n {
                return Err(LinalgError::DimensionError(format!(
                    "GEMM: initial C has shape [{}, {}], expected [{m}, {n}]",
                    c0.nrows(),
                    c0.ncols()
                )));
            }
            if config.beta == 0.0 {
                Array2::<f64>::zeros((m, n))
            } else {
                c0.mapv(|v| v * config.beta)
            }
        }
        None => Array2::<f64>::zeros((m, n)),
    };

    // Ensure C-contiguous layout for raw slice access.
    // If the input is already C-contiguous, as_slice() returns Some without copying.
    // Otherwise we materialise a C-order copy (the copy cost is dominated by GEMM
    // for any reasonably sized matrix).
    let a_owned: Array2<f64>;
    let a_c: &Array2<f64> = if a.is_standard_layout() {
        a
    } else {
        a_owned = Array2::from_shape_fn((a.nrows(), a.ncols()), |(i, j)| a[[i, j]]);
        &a_owned
    };

    let b_owned: Array2<f64>;
    let b_c: &Array2<f64> = if b.is_standard_layout() {
        b
    } else {
        b_owned = Array2::from_shape_fn((b.nrows(), b.ncols()), |(i, j)| b[[i, j]]);
        &b_owned
    };

    let a_slice = a_c.as_slice().ok_or_else(|| {
        LinalgError::ComputationError(
            "A matrix could not be converted to a contiguous slice".to_string(),
        )
    })?;
    let b_slice = b_c.as_slice().ok_or_else(|| {
        LinalgError::ComputationError(
            "B matrix could not be converted to a contiguous slice".to_string(),
        )
    })?;
    let c_slice = c.as_slice_mut().ok_or_else(|| {
        LinalgError::ComputationError("C matrix is not contiguous in memory".to_string())
    })?;

    let alpha = config.alpha;

    // 3-level blocking: K outer, M middle, N inner
    let mut kb_start = 0;
    while kb_start < k {
        let kb_end = (kb_start + kb).min(k);
        let kb_size = kb_end - kb_start;

        let mut mb_start = 0;
        while mb_start < m {
            let mb_end = (mb_start + mb).min(m);

            // Pack A block [mb_start..mb_end, kb_start..kb_end] into a local buffer
            let mb_size = mb_end - mb_start;
            let mut a_pack = vec![0.0_f64; mb_size * kb_size];
            for i in 0..mb_size {
                for p in 0..kb_size {
                    a_pack[i * kb_size + p] = a_slice[(mb_start + i) * k + (kb_start + p)];
                }
            }

            let mut nb_start = 0;
            while nb_start < n {
                let nb_end = (nb_start + nb).min(n);
                let nb_size = nb_end - nb_start;

                // Pack B panel [kb_start..kb_end, nb_start..nb_end] into a local buffer
                let mut b_pack = vec![0.0_f64; kb_size * nb_size];
                for p in 0..kb_size {
                    for j in 0..nb_size {
                        b_pack[p * nb_size + j] = b_slice[(kb_start + p) * n + (nb_start + j)];
                    }
                }

                // Micro-kernel: update C[mb_start..mb_end, nb_start..nb_end]
                micro_kernel(
                    &a_pack, &b_pack, c_slice, mb_size, nb_size, kb_size, mb_start, nb_start, n,
                    alpha,
                );

                nb_start += nb;
            }
            mb_start += mb;
        }
        kb_start += kb;
    }

    Ok(c)
}

/// Micro-kernel: accumulates `α · A_pack · B_pack` into the appropriate tile of C.
#[inline(always)]
fn micro_kernel(
    a_pack: &[f64],
    b_pack: &[f64],
    c: &mut [f64],
    mb: usize,
    nb: usize,
    kb: usize,
    c_row_offset: usize,
    c_col_offset: usize,
    c_stride: usize,
    alpha: f64,
) {
    for i in 0..mb {
        for p in 0..kb {
            let a_ip = a_pack[i * kb + p] * alpha;
            if a_ip == 0.0 {
                continue;
            }
            for j in 0..nb {
                let c_idx = (c_row_offset + i) * c_stride + (c_col_offset + j);
                c[c_idx] += a_ip * b_pack[p * nb + j];
            }
        }
    }
}

// ─── Utility: view-based GEMM helper ─────────────────────────────────────────

/// GEMM accepting `ArrayView2` references.
///
/// Convenience wrapper that converts views to owned arrays then calls [`gemm`].
pub fn gemm_view(
    a: &ArrayView2<f64>,
    b: &ArrayView2<f64>,
    c: Option<&ArrayView2<f64>>,
    config: &GemmConfig,
) -> LinalgResult<Array2<f64>> {
    let a_owned = a.to_owned();
    let b_owned = b.to_owned();
    let c_owned = c.map(|v| v.to_owned());
    gemm(&a_owned, &b_owned, c_owned.as_ref(), config)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2, Array3};

    // Helper: naive reference GEMM
    fn naive_gemm(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        let (m, k) = (a.nrows(), a.ncols());
        let n = b.ncols();
        let mut c = Array2::<f64>::zeros((m, n));
        for i in 0..m {
            for p in 0..k {
                for j in 0..n {
                    c[[i, j]] += a[[i, p]] * b[[p, j]];
                }
            }
        }
        c
    }

    #[test]
    fn test_gemm_identity() {
        let eye = Array2::<f64>::eye(3);
        let b = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let c = gemm(&eye, &b, None, &GemmConfig::default()).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(c[[i, j]], b[[i, j]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_gemm_transpose_a() {
        // A^T * B
        let a = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]]; // 3x2
        let b = array![[1.0_f64, 0.0], [0.0, 1.0], [1.0, 1.0]]; // 3x2
                                                                // A^T is 2x3, B is 3x2 → result is 2x2
        let config = GemmConfig {
            transpose_a: true,
            ..Default::default()
        };
        let c = gemm(&a, &b, None, &config).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        // Expected: A^T * B = [[1,3,5],[2,4,6]] * [[1,0],[0,1],[1,1]]
        // Row 0: [1*1+3*0+5*1, 1*0+3*1+5*1] = [6, 8]
        // Row 1: [2*1+4*0+6*1, 2*0+4*1+6*1] = [8, 10]
        assert_abs_diff_eq!(c[[0, 0]], 6.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c[[0, 1]], 8.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c[[1, 0]], 8.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c[[1, 1]], 10.0, epsilon = 1e-12);
    }

    #[test]
    fn test_gemm_transpose_b() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        // B stored as rows that become columns after transpose:
        // B = [[5,7],[6,8]]  →  B^T = [[5,6],[7,8]]
        // A * B^T = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //         = [[5+14, 6+16], [15+28, 18+32]]
        //         = [[19, 22], [43, 50]]
        let b = array![[5.0_f64, 7.0], [6.0, 8.0]];
        let config = GemmConfig {
            transpose_b: true,
            ..Default::default()
        };
        let c = gemm(&a, &b, None, &config).unwrap();
        assert_abs_diff_eq!(c[[0, 0]], 19.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c[[0, 1]], 22.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c[[1, 0]], 43.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c[[1, 1]], 50.0, epsilon = 1e-12);
    }

    #[test]
    fn test_gemm_alpha_beta() {
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let b = array![[3.0_f64, 0.0], [0.0, 3.0]];
        let c_init = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let config = GemmConfig {
            alpha: 2.0,
            beta: 0.5,
            ..GemmConfig::default()
        };
        let c = gemm(&a, &b, Some(&c_init), &config).unwrap();
        // C = 2*I*3*I + 0.5*C_init = 6*I + 0.5*[[1,2],[3,4]]
        assert_abs_diff_eq!(c[[0, 0]], 6.0 + 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(c[[0, 1]], 0.0 + 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c[[1, 0]], 0.0 + 1.5, epsilon = 1e-12);
        assert_abs_diff_eq!(c[[1, 1]], 6.0 + 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_gemm_blocked_vs_naive() {
        use scirs2_core::ndarray::Array2;
        let m = 70;
        let k = 90;
        let n = 80;
        // Fill with deterministic values
        let a: Array2<f64> =
            Array2::from_shape_fn((m, k), |(i, j)| ((i * k + j) as f64) / (m * k) as f64);
        let b: Array2<f64> =
            Array2::from_shape_fn((k, n), |(i, j)| ((i * n + j) as f64) / (k * n) as f64);
        let expected = naive_gemm(&a, &b);
        let got = gemm(&a, &b, None, &GemmConfig::default()).unwrap();
        for i in 0..m {
            for j in 0..n {
                assert_abs_diff_eq!(got[[i, j]], expected[[i, j]], epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_gemm_non_square() {
        let a = Array2::<f64>::from_shape_fn((3, 5), |(i, j)| (i + j) as f64);
        let b = Array2::<f64>::from_shape_fn((5, 4), |(i, j)| (i * j) as f64);
        let got = gemm(&a, &b, None, &GemmConfig::default()).unwrap();
        let expected = naive_gemm(&a, &b);
        assert_eq!(got.shape(), &[3, 4]);
        for i in 0..3 {
            for j in 0..4 {
                assert_abs_diff_eq!(got[[i, j]], expected[[i, j]], epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_gemm_dimension_mismatch() {
        let a = Array2::<f64>::zeros((3, 4));
        let b = Array2::<f64>::zeros((5, 2)); // k=5 ≠ 4
        assert!(gemm(&a, &b, None, &GemmConfig::default()).is_err());
    }

    #[test]
    fn test_batched_gemm_shape() {
        let a = Array3::<f64>::from_shape_fn((4, 3, 5), |(b, i, j)| (b + i + j) as f64);
        let b = Array3::<f64>::from_shape_fn((4, 5, 2), |(b, i, j)| (b + i * j) as f64);
        let c = batched_gemm(&a, &b, &GemmConfig::default()).unwrap();
        assert_eq!(c.shape(), &[4, 3, 2]);
    }

    #[test]
    fn test_batched_gemm_result() {
        let batch = 3;
        let a =
            Array3::<f64>::from_shape_fn((batch, 2, 2), |(b, i, j)| (b * 4 + i * 2 + j + 1) as f64);
        let b =
            Array3::<f64>::from_shape_fn((batch, 2, 2), |(b, i, j)| (b * 4 + i * 2 + j + 1) as f64);
        let c_batched = batched_gemm(&a, &b, &GemmConfig::default()).unwrap();
        // Compare each slice against individual gemm
        for i in 0..batch {
            let a_slice = a.index_axis(Axis(0), i).to_owned();
            let b_slice = b.index_axis(Axis(0), i).to_owned();
            let c_single = gemm(&a_slice, &b_slice, None, &GemmConfig::default()).unwrap();
            let c_slice = c_batched.index_axis(Axis(0), i).to_owned();
            for r in 0..2 {
                for col in 0..2 {
                    assert_abs_diff_eq!(c_slice[[r, col]], c_single[[r, col]], epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_batched_gemm_batch_mismatch() {
        let a = Array3::<f64>::zeros((3, 2, 2));
        let b = Array3::<f64>::zeros((4, 2, 2));
        assert!(batched_gemm(&a, &b, &GemmConfig::default()).is_err());
    }

    #[test]
    fn test_symm_gemm_symmetry() {
        let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]]; // 2x3
        let c = symm_gemm(&a, &GemmConfig::default()).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        // Result should be symmetric
        assert_abs_diff_eq!(c[[0, 1]], c[[1, 0]], epsilon = 1e-12);
    }

    #[test]
    fn test_symm_gemm_psd() {
        // A * A^T must be positive semi-definite: diagonal ≥ 0
        let a = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let c = symm_gemm(&a, &GemmConfig::default()).unwrap();
        // Check diagonal elements are ≥ 0
        for i in 0..c.nrows() {
            assert!(c[[i, i]] >= 0.0);
        }
    }

    #[test]
    fn test_symm_gemm_values() {
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let c = symm_gemm(&a, &GemmConfig::default()).unwrap();
        // I * I^T = I
        assert_abs_diff_eq!(c[[0, 0]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c[[1, 1]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c[[0, 1]], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_gemm_large_block_boundary() {
        // Test matrices that span multiple cache blocks
        let m = 130;
        let k = 270;
        let n = 140;
        let a: Array2<f64> = Array2::from_shape_fn((m, k), |(i, j)| ((i + j) as f64) * 0.001);
        let b: Array2<f64> = Array2::from_shape_fn((k, n), |(i, j)| ((i * 2 + j) as f64) * 0.001);
        let got = gemm(&a, &b, None, &GemmConfig::default()).unwrap();
        let expected = naive_gemm(&a, &b);
        for i in 0..m {
            for j in 0..n {
                assert_abs_diff_eq!(got[[i, j]], expected[[i, j]], epsilon = 1e-6);
            }
        }
    }
}
