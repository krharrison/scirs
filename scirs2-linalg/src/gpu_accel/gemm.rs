//! GPU-style tiled GEMM kernels
//!
//! This module provides general matrix-multiply (GEMM) routines that model the
//! tiled execution strategy found on real GPU hardware.  All computation runs
//! on the CPU but the data layout (`GpuMatrixBuffer`) and the tile-loop
//! structure mirror the CUDA / OpenCL / Metal kernel organisation so that the
//! same logic can be trivially offloaded when OxiBLAS GPU bindings are wired in.
//!
//! ## Algorithm: 32×32 Tiled GEMM
//!
//! The matrices A (m×k), B (k×n), and C (m×n) are divided into square tiles of
//! edge length `TILE` (default 32).  For each pair of tiles (i_tile, j_tile) in
//! C, all matching k-tiles are multiplied and accumulated:
//!
//! ```text
//! for i_tile in 0..ceil(m/T):
//!   for j_tile in 0..ceil(n/T):
//!     for k_tile in 0..ceil(k/T):
//!       A_tile = A[i_tile*T..(i_tile+1)*T, k_tile*T..(k_tile+1)*T]
//!       B_tile = B[k_tile*T..(k_tile+1)*T, j_tile*T..(j_tile+1)*T]
//!       C[i_tile, j_tile] += A_tile * B_tile   (local inner product)
//! ```
//!
//! This gives O(n³/T) tile-level data accesses with each tile fitting in L1.

use crate::gpu_accel::types::{GpuError, GpuMatrixBuffer, GpuResult};

// Default tile edge length — matches `warp_size` in `detect_gpu_capabilities`.
const DEFAULT_TILE: usize = 32;

// ─── Single-precision GEMM ────────────────────────────────────────────────────

/// Single-precision GEMM: `C = α·A·B + β·C` (f32 operands, f32 result).
///
/// Uses a 32×32 tiled inner loop to model on-device shared-memory tiling.
///
/// # Errors
///
/// Returns [`GpuError::DimensionMismatch`] if dimensions are incompatible.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::gpu_accel::{GpuMatrixBuffer, gpu_sgemm};
///
/// let a = GpuMatrixBuffer::from_slice(&[1.0_f32, 0.0, 0.0, 1.0], 2, 2).unwrap(); // I
/// let b = GpuMatrixBuffer::from_slice(&[3.0_f32, 4.0, 5.0, 6.0], 2, 2).unwrap();
/// let mut c = GpuMatrixBuffer::<f32>::zeros(2, 2);
/// gpu_sgemm(&a, &b, 1.0_f32, 0.0_f32, &mut c).unwrap();
/// assert!((c.as_slice()[0] - 3.0).abs() < 1e-6);
/// ```
pub fn gpu_sgemm(
    a: &GpuMatrixBuffer<f32>,
    b: &GpuMatrixBuffer<f32>,
    alpha: f32,
    beta: f32,
    c: &mut GpuMatrixBuffer<f32>,
) -> GpuResult<()> {
    let (m, k_a) = (a.rows, a.cols);
    let (k_b, n) = (b.rows, b.cols);

    if k_a != k_b {
        return Err(GpuError::DimensionMismatch {
            expected: k_a,
            got: k_b,
            context: format!("gpu_sgemm: A is {m}×{k_a} but B is {k_b}×{n}; inner dims must agree"),
        });
    }
    if c.rows != m || c.cols != n {
        return Err(GpuError::DimensionMismatch {
            expected: m * n,
            got: c.rows * c.cols,
            context: format!("gpu_sgemm: C must be {m}×{n} but is {}×{}", c.rows, c.cols),
        });
    }

    let k = k_a;
    // Scale C by beta first.
    if beta == 0.0 {
        c.data.iter_mut().for_each(|v| *v = 0.0);
    } else if (beta - 1.0).abs() > f32::EPSILON {
        c.data.iter_mut().for_each(|v| *v *= beta);
    }

    tiled_gemm_f32(&a.data, &b.data, &mut c.data, m, n, k, alpha, DEFAULT_TILE);
    Ok(())
}

/// Double-precision GEMM: `C = α·A·B + β·C` (f64 operands, f64 result).
///
/// Same tiled algorithm as [`gpu_sgemm`] but with `f64` arithmetic.
///
/// # Errors
///
/// Returns [`GpuError::DimensionMismatch`] if dimensions are incompatible.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::gpu_accel::{GpuMatrixBuffer, gpu_dgemm};
///
/// let a = GpuMatrixBuffer::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], 2, 2).unwrap();
/// let b = GpuMatrixBuffer::from_slice(&[7.0_f64, 8.0, 9.0, 10.0], 2, 2).unwrap();
/// let mut c = GpuMatrixBuffer::<f64>::zeros(2, 2);
/// gpu_dgemm(&a, &b, 1.0, 0.0, &mut c).unwrap();
/// assert!((c.as_slice()[0] - 7.0).abs() < 1e-12);
/// ```
pub fn gpu_dgemm(
    a: &GpuMatrixBuffer<f64>,
    b: &GpuMatrixBuffer<f64>,
    alpha: f64,
    beta: f64,
    c: &mut GpuMatrixBuffer<f64>,
) -> GpuResult<()> {
    let (m, k_a) = (a.rows, a.cols);
    let (k_b, n) = (b.rows, b.cols);

    if k_a != k_b {
        return Err(GpuError::DimensionMismatch {
            expected: k_a,
            got: k_b,
            context: format!("gpu_dgemm: A is {m}×{k_a} but B is {k_b}×{n}; inner dims must agree"),
        });
    }
    if c.rows != m || c.cols != n {
        return Err(GpuError::DimensionMismatch {
            expected: m * n,
            got: c.rows * c.cols,
            context: format!("gpu_dgemm: C must be {m}×{n} but is {}×{}", c.rows, c.cols),
        });
    }

    let k = k_a;
    if beta == 0.0 {
        c.data.iter_mut().for_each(|v| *v = 0.0);
    } else if (beta - 1.0).abs() > f64::EPSILON {
        c.data.iter_mut().for_each(|v| *v *= beta);
    }

    tiled_gemm_f64(&a.data, &b.data, &mut c.data, m, n, k, alpha, DEFAULT_TILE);
    Ok(())
}

/// Batched GEMM: `C_i = A_i · B_i` for every index in the batch.
///
/// Uses `std::thread::scope` to parallelise across batch elements on the
/// CPU while preserving the GPU batched-GEMM calling convention.
///
/// # Errors
///
/// - [`GpuError::DimensionMismatch`] when the batch lengths differ.
/// - [`GpuError::DimensionMismatch`] when any pair has incompatible inner dims.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::gpu_accel::{GpuMatrixBuffer, gpu_batched_gemm};
///
/// let eye = GpuMatrixBuffer::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], 2, 2).unwrap();
/// let mat = GpuMatrixBuffer::from_slice(&[3.0_f64, 4.0, 5.0, 6.0], 2, 2).unwrap();
/// let results = gpu_batched_gemm(&[eye], &[mat]).unwrap();
/// assert_eq!(results.len(), 1);
/// assert!((results[0].as_slice()[0] - 3.0).abs() < 1e-12);
/// ```
pub fn gpu_batched_gemm(
    a_batch: &[GpuMatrixBuffer<f64>],
    b_batch: &[GpuMatrixBuffer<f64>],
) -> GpuResult<Vec<GpuMatrixBuffer<f64>>> {
    if a_batch.len() != b_batch.len() {
        return Err(GpuError::DimensionMismatch {
            expected: a_batch.len(),
            got: b_batch.len(),
            context: "gpu_batched_gemm: batch lengths must match".to_string(),
        });
    }

    // Validate dimensions before launching threads.
    for (idx, (a, b)) in a_batch.iter().zip(b_batch.iter()).enumerate() {
        if a.cols != b.rows {
            return Err(GpuError::DimensionMismatch {
                expected: a.cols,
                got: b.rows,
                context: format!(
                    "gpu_batched_gemm batch[{idx}]: A cols ({}) != B rows ({})",
                    a.cols, b.rows
                ),
            });
        }
    }

    let batch = a_batch.len();
    // Allocate output buffers.
    let mut results: Vec<GpuMatrixBuffer<f64>> = a_batch
        .iter()
        .zip(b_batch.iter())
        .map(|(a, b)| GpuMatrixBuffer::<f64>::zeros(a.rows, b.cols))
        .collect();

    // Parallel execution using a shared output array split into disjoint
    // per-element slots.  We use `split_at_mut` recursively to hand out
    // exclusive mutable references without raw pointer unsafety.
    parallel_batch_gemm(a_batch, b_batch, &mut results);

    Ok(results)
}

// ─── Upload / download helpers ────────────────────────────────────────────────

/// Copy a flat row-major slice into a [`GpuMatrixBuffer<f64>`].
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::gpu_accel::upload_matrix;
///
/// let buf = upload_matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2);
/// assert_eq!(buf.rows, 2);
/// ```
pub fn upload_matrix(data: &[f64], rows: usize, cols: usize) -> GpuMatrixBuffer<f64> {
    GpuMatrixBuffer {
        data: data.to_vec(),
        rows,
        cols,
    }
}

/// Extract the flat row-major data from a [`GpuMatrixBuffer<f64>`].
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::gpu_accel::{upload_matrix, download_matrix};
///
/// let buf = upload_matrix(&[5.0, 6.0, 7.0, 8.0], 2, 2);
/// let out = download_matrix(&buf);
/// assert_eq!(out, vec![5.0, 6.0, 7.0, 8.0]);
/// ```
pub fn download_matrix(buf: &GpuMatrixBuffer<f64>) -> Vec<f64> {
    buf.data.clone()
}

// ─── Internal parallelism helper ─────────────────────────────────────────────

/// Execute batched GEMM in parallel using scoped threads.
///
/// Uses `split_at_mut` recursively to distribute the output slice across
/// threads without raw pointer casting.
fn parallel_batch_gemm(
    a_batch: &[GpuMatrixBuffer<f64>],
    b_batch: &[GpuMatrixBuffer<f64>],
    results: &mut [GpuMatrixBuffer<f64>],
) {
    let batch = a_batch.len();
    if batch == 0 {
        return;
    }
    if batch == 1 {
        let m = a_batch[0].rows;
        let k = a_batch[0].cols;
        let n = b_batch[0].cols;
        tiled_gemm_f64(
            &a_batch[0].data,
            &b_batch[0].data,
            &mut results[0].data,
            m,
            n,
            k,
            1.0,
            DEFAULT_TILE,
        );
        return;
    }

    // Split into two halves and process each half in a scoped thread.
    let mid = batch / 2;
    let (a_lo, a_hi) = a_batch.split_at(mid);
    let (b_lo, b_hi) = b_batch.split_at(mid);
    let (r_lo, r_hi) = results.split_at_mut(mid);

    std::thread::scope(|scope| {
        let left = scope.spawn(|| parallel_batch_gemm(a_lo, b_lo, r_lo));
        let right = scope.spawn(|| parallel_batch_gemm(a_hi, b_hi, r_hi));
        left.join().ok();
        right.join().ok();
    });
}

// ─── Internal tiled kernels ───────────────────────────────────────────────────

/// Core tiled GEMM for f32: accumulates `alpha * A * B` into `c_flat`.
///
/// `c_flat` is assumed to have already been scaled by `beta` (or zeroed).
pub(crate) fn tiled_gemm_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    tile: usize,
) {
    let tile = tile.max(1);
    let m_tiles = m.div_ceil(tile);
    let n_tiles = n.div_ceil(tile);
    let k_tiles = k.div_ceil(tile);

    for it in 0..m_tiles {
        let i_start = it * tile;
        let i_end = (i_start + tile).min(m);

        for jt in 0..n_tiles {
            let j_start = jt * tile;
            let j_end = (j_start + tile).min(n);

            for kt in 0..k_tiles {
                let kk_start = kt * tile;
                let kk_end = (kk_start + tile).min(k);

                // Inner micro-kernel: update C[i_start..i_end, j_start..j_end]
                for i in i_start..i_end {
                    for kk in kk_start..kk_end {
                        let a_ik = a[i * k + kk] * alpha;
                        if a_ik == 0.0 {
                            continue;
                        }
                        for j in j_start..j_end {
                            c[i * n + j] += a_ik * b[kk * n + j];
                        }
                    }
                }
            }
        }
    }
}

/// Core tiled GEMM for f64: accumulates `alpha * A * B` into `c_flat`.
pub(crate) fn tiled_gemm_f64(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    tile: usize,
) {
    let tile = tile.max(1);
    let m_tiles = m.div_ceil(tile);
    let n_tiles = n.div_ceil(tile);
    let k_tiles = k.div_ceil(tile);

    for it in 0..m_tiles {
        let i_start = it * tile;
        let i_end = (i_start + tile).min(m);

        for jt in 0..n_tiles {
            let j_start = jt * tile;
            let j_end = (j_start + tile).min(n);

            for kt in 0..k_tiles {
                let kk_start = kt * tile;
                let kk_end = (kk_start + tile).min(k);

                for i in i_start..i_end {
                    for kk in kk_start..kk_end {
                        let a_ik = a[i * k + kk] * alpha;
                        if a_ik == 0.0 {
                            continue;
                        }
                        for j in j_start..j_end {
                            c[i * n + j] += a_ik * b[kk * n + j];
                        }
                    }
                }
            }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a flat row-major matrix where element (i,j) = i*cols + j + 1 (f32).
    fn sequential_f32(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols).map(|idx| (idx + 1) as f32).collect()
    }

    /// Build a flat row-major matrix where element (i,j) = i*cols + j + 1 (f64).
    fn sequential_f64(rows: usize, cols: usize) -> Vec<f64> {
        (0..rows * cols).map(|idx| (idx + 1) as f64).collect()
    }

    /// Naive reference GEMM for correctness checks.
    fn naive_gemm(a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
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
    fn test_gpu_sgemm_identity() {
        let eye_data = vec![1.0_f32, 0.0, 0.0, 1.0];
        let b_data = vec![3.0_f32, 4.0, 5.0, 6.0];
        let a = GpuMatrixBuffer::from_slice(&eye_data, 2, 2).unwrap();
        let b = GpuMatrixBuffer::from_slice(&b_data, 2, 2).unwrap();
        let mut c = GpuMatrixBuffer::<f32>::zeros(2, 2);
        gpu_sgemm(&a, &b, 1.0, 0.0, &mut c).unwrap();
        for (i, (&got, &exp)) in c.as_slice().iter().zip(b_data.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-6, "sgemm identity mismatch at {i}");
        }
    }

    #[test]
    fn test_gpu_sgemm_2x2_known_result() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        // C = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        let a = GpuMatrixBuffer::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let b = GpuMatrixBuffer::from_slice(&[5.0_f32, 6.0, 7.0, 8.0], 2, 2).unwrap();
        let mut c = GpuMatrixBuffer::<f32>::zeros(2, 2);
        gpu_sgemm(&a, &b, 1.0, 0.0, &mut c).unwrap();
        let expected = [19.0_f32, 22.0, 43.0, 50.0];
        for (i, (&got, &exp)) in c.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "sgemm 2x2 mismatch at {i}: {got} vs {exp}"
            );
        }
    }

    #[test]
    fn test_gpu_dgemm_4x4_vs_naive() {
        let a_data = sequential_f64(4, 4);
        let b_data = sequential_f64(4, 4);
        let expected = naive_gemm(&a_data, &b_data, 4, 4, 4);

        let a = GpuMatrixBuffer::from_slice(&a_data, 4, 4).unwrap();
        let b = GpuMatrixBuffer::from_slice(&b_data, 4, 4).unwrap();
        let mut c = GpuMatrixBuffer::<f64>::zeros(4, 4);
        gpu_dgemm(&a, &b, 1.0, 0.0, &mut c).unwrap();

        for (i, (&got, &exp)) in c.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-9, "dgemm 4x4 mismatch at {i}");
        }
    }

    #[test]
    fn test_gpu_dgemm_alpha_beta() {
        // C = 2*A*B + 3*C_init
        let a = GpuMatrixBuffer::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], 2, 2).unwrap();
        let b = GpuMatrixBuffer::from_slice(&[5.0_f64, 0.0, 0.0, 5.0], 2, 2).unwrap();
        let mut c = GpuMatrixBuffer::from_slice(&[1.0_f64, 1.0, 1.0, 1.0], 2, 2).unwrap();
        gpu_dgemm(&a, &b, 2.0, 3.0, &mut c).unwrap();
        // Diagonal: 2*1*5 + 3*1 = 13; Off-diagonal: 2*0 + 3*1 = 3
        assert!(
            (c.as_slice()[0] - 13.0).abs() < 1e-12,
            "dgemm alpha/beta diag"
        );
        assert!(
            (c.as_slice()[1] - 3.0).abs() < 1e-12,
            "dgemm alpha/beta off-diag"
        );
    }

    #[test]
    fn test_gpu_sgemm_dimension_mismatch() {
        let a = GpuMatrixBuffer::<f32>::zeros(2, 3);
        let b = GpuMatrixBuffer::<f32>::zeros(4, 2); // k=4 ≠ 3
        let mut c = GpuMatrixBuffer::<f32>::zeros(2, 2);
        let result = gpu_sgemm(&a, &b, 1.0, 0.0, &mut c);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GpuError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn test_gpu_dgemm_dimension_mismatch() {
        let a = GpuMatrixBuffer::<f64>::zeros(3, 5);
        let b = GpuMatrixBuffer::<f64>::zeros(4, 3);
        let mut c = GpuMatrixBuffer::<f64>::zeros(3, 3);
        assert!(gpu_dgemm(&a, &b, 1.0, 0.0, &mut c).is_err());
    }

    #[test]
    fn test_gpu_batched_gemm_correctness() {
        // Two 2×2 identity-times-matrix batches
        let eye: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0];
        let mat_a: Vec<f64> = vec![3.0, 4.0, 5.0, 6.0];
        let mat_b: Vec<f64> = vec![7.0, 8.0, 9.0, 10.0];

        let a_batch = vec![
            GpuMatrixBuffer::from_slice(&eye, 2, 2).unwrap(),
            GpuMatrixBuffer::from_slice(&eye, 2, 2).unwrap(),
        ];
        let b_batch = vec![
            GpuMatrixBuffer::from_slice(&mat_a, 2, 2).unwrap(),
            GpuMatrixBuffer::from_slice(&mat_b, 2, 2).unwrap(),
        ];

        let results = gpu_batched_gemm(&a_batch, &b_batch).unwrap();
        assert_eq!(results.len(), 2);
        // I * mat_a == mat_a
        for (&got, &exp) in results[0].as_slice().iter().zip(mat_a.iter()) {
            assert!((got - exp).abs() < 1e-12, "batched[0] mismatch");
        }
        for (&got, &exp) in results[1].as_slice().iter().zip(mat_b.iter()) {
            assert!((got - exp).abs() < 1e-12, "batched[1] mismatch");
        }
    }

    #[test]
    fn test_gpu_batched_gemm_batch_mismatch() {
        let a_batch = vec![GpuMatrixBuffer::<f64>::zeros(2, 2)];
        let b_batch: Vec<GpuMatrixBuffer<f64>> = vec![];
        assert!(gpu_batched_gemm(&a_batch, &b_batch).is_err());
    }

    #[test]
    fn test_upload_download_roundtrip() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let buf = upload_matrix(&data, 2, 3);
        let out = download_matrix(&buf);
        assert_eq!(data, out);
    }

    #[test]
    fn test_gpu_dgemm_zero_matrix() {
        let a = GpuMatrixBuffer::<f64>::zeros(3, 3);
        let b_data: Vec<f64> = (1..=9).map(|v| v as f64).collect();
        let b = GpuMatrixBuffer::from_slice(&b_data, 3, 3).unwrap();
        let mut c = GpuMatrixBuffer::<f64>::zeros(3, 3);
        gpu_dgemm(&a, &b, 1.0, 0.0, &mut c).unwrap();
        assert!(c.as_slice().iter().all(|&v| v == 0.0), "0*B must be 0");
    }

    #[test]
    fn test_tiled_vs_naive_larger_matrix() {
        let m = 50;
        let k = 60;
        let n = 55;
        let a: Vec<f64> = (0..m * k).map(|i| (i as f64) * 0.01).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i as f64) * 0.01 + 0.5).collect();
        let expected = naive_gemm(&a, &b, m, n, k);

        let mut c_tiled = vec![0.0_f64; m * n];
        tiled_gemm_f64(&a, &b, &mut c_tiled, m, n, k, 1.0, 32);

        for (i, (&got, &exp)) in c_tiled.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-6, "tiled vs naive mismatch at {i}");
        }
    }

    #[test]
    fn test_f32_f64_consistency() {
        // Same computation in f32 and f64 should give comparable results for
        // well-scaled matrices.
        let n = 8;
        let a_f64: Vec<f64> = (0..n * n).map(|i| ((i % 5) as f64) * 0.1 + 0.1).collect();
        let b_f64: Vec<f64> = (0..n * n).map(|i| ((i % 7) as f64) * 0.1 + 0.2).collect();
        let a_f32: Vec<f32> = a_f64.iter().map(|&v| v as f32).collect();
        let b_f32: Vec<f32> = b_f64.iter().map(|&v| v as f32).collect();

        let a64 = GpuMatrixBuffer::from_slice(&a_f64, n, n).unwrap();
        let b64 = GpuMatrixBuffer::from_slice(&b_f64, n, n).unwrap();
        let mut c64 = GpuMatrixBuffer::<f64>::zeros(n, n);
        gpu_dgemm(&a64, &b64, 1.0, 0.0, &mut c64).unwrap();

        let a32 = GpuMatrixBuffer::from_slice(&a_f32, n, n).unwrap();
        let b32 = GpuMatrixBuffer::from_slice(&b_f32, n, n).unwrap();
        let mut c32 = GpuMatrixBuffer::<f32>::zeros(n, n);
        gpu_sgemm(&a32, &b32, 1.0, 0.0, &mut c32).unwrap();

        for (i, (&v64, &v32)) in c64.as_slice().iter().zip(c32.as_slice().iter()).enumerate() {
            let diff = (v64 - v32 as f64).abs();
            assert!(diff < 1e-3, "f32/f64 consistency at {i}: diff = {diff}");
        }
    }

    #[test]
    fn test_gpu_batched_gemm_3_matrices() {
        let n = 3;
        let a_data: Vec<f64> = (0..n * n).map(|i| (i + 1) as f64).collect();
        let b_data: Vec<f64> = (0..n * n).map(|i| (n * n - i) as f64).collect();

        let a_batch: Vec<GpuMatrixBuffer<f64>> = (0..3)
            .map(|_| GpuMatrixBuffer::from_slice(&a_data, n, n).unwrap())
            .collect();
        let b_batch: Vec<GpuMatrixBuffer<f64>> = (0..3)
            .map(|_| GpuMatrixBuffer::from_slice(&b_data, n, n).unwrap())
            .collect();

        let results = gpu_batched_gemm(&a_batch, &b_batch).unwrap();
        assert_eq!(results.len(), 3);

        let expected = naive_gemm(&a_data, &b_data, n, n, n);
        for (bi, res) in results.iter().enumerate() {
            for (i, (&got, &exp)) in res.as_slice().iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-9,
                    "batched[{bi}][{i}]: {got} vs {exp}"
                );
            }
        }
    }
}
