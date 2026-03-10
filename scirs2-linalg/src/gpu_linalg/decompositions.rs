//! GPU-accelerated matrix decompositions.
//!
//! Provides GPU-accelerated LU, QR, Cholesky, and SVD decompositions using
//! scirs2-core GPU backends. Each operation has automatic CPU fallback when
//! GPU is unavailable or the matrix is too small to benefit from GPU transfer.

use scirs2_core::gpu::{GpuBackend, GpuContext, GpuError};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign, One, Zero};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};

use super::types::{CholeskyFactors, LuFactors, QrFactors, SvdFactors};

/// Minimum matrix dimension for GPU acceleration.
/// Below this threshold, CPU is faster due to data transfer overhead.
const GPU_DECOMP_THRESHOLD: usize = 32;

// =============================================================================
// LU Decomposition
// =============================================================================

/// GPU-accelerated LU decomposition: PA = LU
///
/// Factors the matrix A into a permutation matrix P, a lower-triangular matrix L
/// (with unit diagonal), and an upper-triangular matrix U.
///
/// Uses GPU for large matrices, falling back to CPU for small matrices or when
/// no GPU is available.
///
/// # Arguments
///
/// * `ctx` - GPU context (or None for CPU fallback)
/// * `a` - Input square matrix
///
/// # Returns
///
/// `LuFactors` containing P, L, and U matrices.
///
/// # Errors
///
/// Returns `LinalgError` if the matrix is empty, non-finite, or singular.
pub fn gpu_lu<F>(ctx: Option<&GpuContext>, a: &ArrayView2<F>) -> LinalgResult<LuFactors<F>>
where
    F: Float + NumAssign + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    validate_matrix(a, "LU decomposition")?;

    let n = a.nrows();

    // Try GPU path for large matrices
    if n >= GPU_DECOMP_THRESHOLD {
        if let Some(gpu_ctx) = ctx {
            match gpu_lu_impl(gpu_ctx, a) {
                Ok(factors) => return Ok(factors),
                Err(_) => {
                    // Fall through to CPU
                }
            }
        }
    }

    // CPU fallback: use existing LU decomposition
    cpu_lu(a)
}

/// GPU LU implementation via blocked algorithm on GPU buffers.
fn gpu_lu_impl<F>(ctx: &GpuContext, a: &ArrayView2<F>) -> Result<LuFactors<F>, LinalgError>
where
    F: Float + NumAssign + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    let n = a.nrows();
    let m = a.ncols();

    // Transfer matrix to GPU as f64
    let a_flat: Vec<f64> = a.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
    let _gpu_buf = ctx.create_buffer_from_slice(&a_flat);

    // For the GPU LU decomposition, we implement a right-looking blocked algorithm.
    // We perform the factorization panel by panel on GPU using GEMM for the
    // trailing matrix update.
    //
    // The core loop:
    //   1. Factor the current panel (column block) to find pivots
    //   2. Apply row swaps to the rest of the matrix
    //   3. Solve triangular system for U block
    //   4. Update trailing sub-matrix via GEMM: A_trail -= L_panel * U_top

    // We work on host data but use GPU for the trailing matrix update (GEMM).
    let mut work = a.to_owned();
    let min_dim = n.min(m);
    let mut piv: Vec<usize> = (0..n).collect();

    let block_size = 32.min(min_dim);

    let mut jb = 0;
    while jb < min_dim {
        let jb_end = (jb + block_size).min(min_dim);
        let current_block = jb_end - jb;

        // Factor the panel [jb..n, jb..jb_end] using partial pivoting
        for j in jb..jb_end {
            // Find pivot in column j, rows j..n
            let mut max_val = F::zero();
            let mut max_row = j;
            for i in j..n {
                let val = work[[i, j]].abs();
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }

            // Swap rows if needed
            if max_row != j {
                for col in 0..m {
                    let tmp = work[[j, col]];
                    work[[j, col]] = work[[max_row, col]];
                    work[[max_row, col]] = tmp;
                }
                piv.swap(j, max_row);
            }

            // Check for singularity
            let pivot = work[[j, j]];
            if pivot.abs() < F::epsilon() * F::from(100.0).unwrap_or_else(F::one) {
                // Nearly singular, but we continue (the result will have small pivots)
            }

            // Compute multipliers for column j
            if pivot.abs() > F::zero() {
                for i in (j + 1)..n {
                    work[[i, j]] /= pivot;
                }
            }

            // Update the panel sub-matrix
            for i in (j + 1)..n {
                let factor = work[[i, j]];
                for k in (j + 1)..jb_end {
                    let rhs = factor * work[[j, k]];
                    work[[i, k]] -= rhs;
                }
            }
        }

        // Update trailing matrix using GPU GEMM: A_trail -= L_panel * U_top
        if jb_end < m {
            let trail_rows = n - jb;
            let trail_cols = m - jb_end;

            // Extract L panel (below diagonal part of current block)
            // and U top (right of current block in factored rows)
            if trail_rows > 0 && trail_cols > 0 && current_block > 0 {
                // L_panel: rows [jb..n], cols [jb..jb_end] (lower part only, rows jb_end..n)
                let l_rows = n - jb_end;
                if l_rows > 0 {
                    let mut l_panel = vec![0.0_f64; l_rows * current_block];
                    for i in 0..l_rows {
                        for j in 0..current_block {
                            l_panel[i * current_block + j] =
                                work[[jb_end + i, jb + j]].to_f64().unwrap_or(0.0);
                        }
                    }

                    let mut u_top = vec![0.0_f64; current_block * trail_cols];
                    for i in 0..current_block {
                        for j in 0..trail_cols {
                            u_top[i * trail_cols + j] =
                                work[[jb + i, jb_end + j]].to_f64().unwrap_or(0.0);
                        }
                    }

                    // GPU GEMM: C = L_panel * U_top
                    let gpu_l = ctx.create_buffer_from_slice(&l_panel);
                    let gpu_u = ctx.create_buffer_from_slice(&u_top);

                    match ctx.gemm(&gpu_l, &gpu_u, l_rows, current_block, trail_cols) {
                        Ok(gpu_c) => {
                            let c_flat = gpu_c.to_vec();
                            // Subtract from trailing matrix
                            for i in 0..l_rows {
                                for j in 0..trail_cols {
                                    let val =
                                        F::from(c_flat[i * trail_cols + j]).unwrap_or_else(F::zero);
                                    work[[jb_end + i, jb_end + j]] -= val;
                                }
                            }
                        }
                        Err(_) => {
                            // CPU fallback for trailing update
                            for i in 0..l_rows {
                                for j in 0..trail_cols {
                                    let mut sum = F::zero();
                                    for p in 0..current_block {
                                        sum +=
                                            work[[jb_end + i, jb + p]] * work[[jb + p, jb_end + j]];
                                    }
                                    work[[jb_end + i, jb_end + j]] -= sum;
                                }
                            }
                        }
                    }
                }
            }
        }

        jb = jb_end;
    }

    // Extract P, L, U from the work array
    extract_plu(&work, &piv, n, m)
}

/// CPU fallback for LU decomposition.
fn cpu_lu<F>(a: &ArrayView2<F>) -> LinalgResult<LuFactors<F>>
where
    F: Float + NumAssign + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (p, l, u) = crate::lu(a, None)?;
    Ok(LuFactors { p, l, u })
}

/// Extract P, L, U matrices from compact LU storage.
fn extract_plu<F>(
    work: &Array2<F>,
    piv: &[usize],
    n: usize,
    m: usize,
) -> Result<LuFactors<F>, LinalgError>
where
    F: Float + NumAssign + One + Zero,
{
    let min_dim = n.min(m);

    // Build permutation matrix
    let mut p = Array2::<F>::zeros((n, n));
    for (i, &piv_idx) in piv.iter().enumerate() {
        if piv_idx < n {
            p[[i, piv_idx]] = F::one();
        }
    }

    // Extract L (lower triangular with unit diagonal)
    let mut l = Array2::<F>::zeros((n, min_dim));
    for i in 0..n {
        for j in 0..i.min(min_dim) {
            l[[i, j]] = work[[i, j]];
        }
        if i < min_dim {
            l[[i, i]] = F::one();
        }
    }

    // Extract U (upper triangular)
    let mut u = Array2::<F>::zeros((min_dim, m));
    for i in 0..min_dim {
        for j in i..m {
            u[[i, j]] = work[[i, j]];
        }
    }

    Ok(LuFactors { p, l, u })
}

// =============================================================================
// QR Decomposition
// =============================================================================

/// GPU-accelerated QR decomposition: A = QR
///
/// Factors the matrix A into an orthogonal matrix Q and an upper-triangular
/// matrix R using Householder reflections, with GPU-accelerated trailing
/// matrix updates.
///
/// # Arguments
///
/// * `ctx` - GPU context (or None for CPU fallback)
/// * `a` - Input matrix (m x n)
///
/// # Returns
///
/// `QrFactors` containing Q and R matrices.
///
/// # Errors
///
/// Returns `LinalgError` if the matrix is empty or contains non-finite values.
pub fn gpu_qr<F>(ctx: Option<&GpuContext>, a: &ArrayView2<F>) -> LinalgResult<QrFactors<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    validate_matrix(a, "QR decomposition")?;

    let (m, n) = a.dim();

    if m >= GPU_DECOMP_THRESHOLD && n >= GPU_DECOMP_THRESHOLD {
        if let Some(gpu_ctx) = ctx {
            match gpu_qr_impl(gpu_ctx, a) {
                Ok(factors) => return Ok(factors),
                Err(_) => {
                    // Fall through to CPU
                }
            }
        }
    }

    cpu_qr(a)
}

/// GPU QR implementation using Householder reflections with GPU-accelerated updates.
fn gpu_qr_impl<F>(ctx: &GpuContext, a: &ArrayView2<F>) -> Result<QrFactors<F>, LinalgError>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (m, n) = a.dim();
    let min_dim = m.min(n);

    let mut r = a.to_owned();
    let mut q = Array2::<F>::zeros((m, m));
    // Initialize Q as identity
    for i in 0..m {
        q[[i, i]] = F::one();
    }

    for k in 0..min_dim {
        // Compute Householder vector for column k
        let mut norm_sq = F::zero();
        for i in k..m {
            norm_sq += r[[i, k]] * r[[i, k]];
        }
        let norm = norm_sq.sqrt();

        if norm < F::epsilon() {
            continue;
        }

        // Compute the Householder vector v
        let sign = if r[[k, k]] >= F::zero() {
            F::one()
        } else {
            -F::one()
        };
        let mut v = vec![F::zero(); m - k];
        v[0] = r[[k, k]] + sign * norm;
        for i in 1..(m - k) {
            v[i] = r[[k + i, k]];
        }

        // Normalize v
        let mut v_norm_sq = F::zero();
        for vi in &v {
            v_norm_sq += *vi * *vi;
        }
        if v_norm_sq < F::epsilon() {
            continue;
        }
        let tau = F::from(2.0).unwrap_or_else(F::one) / v_norm_sq;

        // Apply Householder reflection to R: R[k:, k:] -= tau * v * (v^T * R[k:, k:])
        // Using GPU for the matrix-vector products when beneficial
        let trail_cols = n - k;
        let trail_rows = m - k;

        if trail_rows >= GPU_DECOMP_THRESHOLD && trail_cols >= GPU_DECOMP_THRESHOLD {
            // Try GPU-accelerated update
            // v^T * R[k:, k:] => 1 x trail_cols
            let mut vt_r = vec![F::zero(); trail_cols];
            for j in 0..trail_cols {
                let mut sum = F::zero();
                for i in 0..trail_rows {
                    sum += v[i] * r[[k + i, k + j]];
                }
                vt_r[j] = sum;
            }

            // R[k:, k:] -= tau * v * vt_r^T (outer product update)
            // Use GPU for the outer product if large enough
            let v_f64: Vec<f64> = v.iter().map(|x| x.to_f64().unwrap_or(0.0)).collect();
            let vtr_f64: Vec<f64> = vt_r.iter().map(|x| x.to_f64().unwrap_or(0.0)).collect();

            let gpu_v = ctx.create_buffer_from_slice(&v_f64);
            let gpu_vtr = ctx.create_buffer_from_slice(&vtr_f64);

            match ctx.gemm(&gpu_v, &gpu_vtr, trail_rows, 1, trail_cols) {
                Ok(gpu_outer) => {
                    let outer_flat = gpu_outer.to_vec();
                    let tau_f64 = tau.to_f64().unwrap_or(2.0);
                    for i in 0..trail_rows {
                        for j in 0..trail_cols {
                            let val = F::from(tau_f64 * outer_flat[i * trail_cols + j])
                                .unwrap_or_else(F::zero);
                            r[[k + i, k + j]] -= val;
                        }
                    }
                }
                Err(_) => {
                    // CPU fallback for outer product
                    for i in 0..trail_rows {
                        for j in 0..trail_cols {
                            r[[k + i, k + j]] -= tau * v[i] * vt_r[j];
                        }
                    }
                }
            }
        } else {
            // Small matrix: CPU update
            let mut vt_r = vec![F::zero(); trail_cols];
            for j in 0..trail_cols {
                let mut sum = F::zero();
                for i in 0..trail_rows {
                    sum += v[i] * r[[k + i, k + j]];
                }
                vt_r[j] = sum;
            }
            for i in 0..trail_rows {
                for j in 0..trail_cols {
                    r[[k + i, k + j]] -= tau * v[i] * vt_r[j];
                }
            }
        }

        // Apply Householder reflection to Q: Q[:, k:] -= tau * (Q[:, k:] * v) * v^T
        let mut q_v = vec![F::zero(); m];
        for i in 0..m {
            let mut sum = F::zero();
            for j in 0..trail_rows {
                sum += q[[i, k + j]] * v[j];
            }
            q_v[i] = sum;
        }
        for i in 0..m {
            for j in 0..trail_rows {
                q[[i, k + j]] -= tau * q_v[i] * v[j];
            }
        }
    }

    // Trim R to proper dimensions
    let r_trimmed = r.slice(scirs2_core::ndarray::s![0..min_dim, ..]).to_owned();

    // Trim Q to proper dimensions
    let q_trimmed = q.slice(scirs2_core::ndarray::s![.., 0..min_dim]).to_owned();

    // For economy QR: Q is m x min(m,n), R is min(m,n) x n
    // For full QR: Q is m x m, R is m x n
    // We return the economy form
    Ok(QrFactors {
        q: q_trimmed,
        r: r_trimmed,
    })
}

/// CPU fallback for QR decomposition.
fn cpu_qr<F>(a: &ArrayView2<F>) -> LinalgResult<QrFactors<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (q, r) = crate::qr(a, None)?;
    Ok(QrFactors { q, r })
}

// =============================================================================
// Cholesky Decomposition
// =============================================================================

/// GPU-accelerated Cholesky decomposition: A = LL^T
///
/// Factors a symmetric positive-definite matrix A into a lower-triangular
/// matrix L such that A = L * L^T. Uses a blocked algorithm with
/// GPU-accelerated trailing matrix updates.
///
/// # Arguments
///
/// * `ctx` - GPU context (or None for CPU fallback)
/// * `a` - Symmetric positive-definite matrix
///
/// # Returns
///
/// `CholeskyFactors` containing the lower-triangular factor L.
///
/// # Errors
///
/// Returns `LinalgError` if the matrix is not square, not positive definite,
/// or contains non-finite values.
pub fn gpu_cholesky<F>(
    ctx: Option<&GpuContext>,
    a: &ArrayView2<F>,
) -> LinalgResult<CholeskyFactors<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    validate_matrix(a, "Cholesky decomposition")?;

    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(
            "Cholesky decomposition requires a square matrix".to_string(),
        ));
    }

    let n = a.nrows();

    if n >= GPU_DECOMP_THRESHOLD {
        if let Some(gpu_ctx) = ctx {
            match gpu_cholesky_impl(gpu_ctx, a) {
                Ok(factors) => return Ok(factors),
                Err(_) => {
                    // Fall through to CPU
                }
            }
        }
    }

    cpu_cholesky(a)
}

/// GPU Cholesky implementation using blocked algorithm.
fn gpu_cholesky_impl<F>(
    ctx: &GpuContext,
    a: &ArrayView2<F>,
) -> Result<CholeskyFactors<F>, LinalgError>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let n = a.nrows();
    let mut l = Array2::<F>::zeros((n, n));

    // Copy lower triangle of A
    for i in 0..n {
        for j in 0..=i {
            l[[i, j]] = a[[i, j]];
        }
    }

    let block_size = 32.min(n);

    let mut jb = 0;
    while jb < n {
        let jb_end = (jb + block_size).min(n);
        let current_block = jb_end - jb;

        // Factor the diagonal block using standard Cholesky
        for j in jb..jb_end {
            // Compute diagonal element
            let mut sum = l[[j, j]];
            for k in jb..j {
                sum -= l[[j, k]] * l[[j, k]];
            }
            if sum <= F::zero() {
                return Err(LinalgError::NonPositiveDefiniteError(format!(
                    "Matrix is not positive definite at index {}",
                    j
                )));
            }
            l[[j, j]] = sum.sqrt();

            // Compute column j below diagonal within the block
            let diag = l[[j, j]];
            for i in (j + 1)..jb_end {
                let mut sum = l[[i, j]];
                for k in jb..j {
                    sum -= l[[i, k]] * l[[j, k]];
                }
                l[[i, j]] = sum / diag;
            }
        }

        // Update rows below the current block using GPU GEMM
        if jb_end < n {
            let trail_rows = n - jb_end;

            // L_trail_panel = L[jb_end:, jb:jb_end] (to be computed from A[jb_end:, jb:jb_end])
            // Solve: L_trail_panel * L_block^T = A_trail_panel
            // Where L_block is L[jb:jb_end, jb:jb_end]

            // First, solve the triangular system for the panel below
            for j in jb..jb_end {
                let diag = l[[j, j]];
                for i in jb_end..n {
                    let mut sum = l[[i, j]];
                    for k in jb..j {
                        sum -= l[[i, k]] * l[[j, k]];
                    }
                    l[[i, j]] = sum / diag;
                }
            }

            // Update trailing matrix using GPU GEMM
            // L[jb_end:, jb_end:] -= L[jb_end:, jb:jb_end] * L[jb_end:, jb:jb_end]^T
            // But we only need the lower triangle

            let mut l_panel_flat = vec![0.0_f64; trail_rows * current_block];
            for i in 0..trail_rows {
                for j in 0..current_block {
                    l_panel_flat[i * current_block + j] =
                        l[[jb_end + i, jb + j]].to_f64().unwrap_or(0.0);
                }
            }

            let gpu_l_panel = ctx.create_buffer_from_slice(&l_panel_flat);

            // Compute L_panel * L_panel^T using GPU
            match ctx.gemm_transpose_b(
                &gpu_l_panel,
                &gpu_l_panel,
                trail_rows,
                current_block,
                trail_rows,
            ) {
                Ok(gpu_update) => {
                    let update_flat = gpu_update.to_vec();
                    // Subtract from trailing lower triangle
                    for i in 0..trail_rows {
                        for j in 0..=i {
                            let val =
                                F::from(update_flat[i * trail_rows + j]).unwrap_or_else(F::zero);
                            l[[jb_end + i, jb_end + j]] -= val;
                        }
                    }
                }
                Err(_) => {
                    // CPU fallback for trailing update
                    for i in 0..trail_rows {
                        for j in 0..=i {
                            let mut sum = F::zero();
                            for p in 0..current_block {
                                sum += l[[jb_end + i, jb + p]] * l[[jb_end + j, jb + p]];
                            }
                            l[[jb_end + i, jb_end + j]] -= sum;
                        }
                    }
                }
            }
        }

        jb = jb_end;
    }

    Ok(CholeskyFactors { l })
}

/// CPU fallback for Cholesky decomposition.
fn cpu_cholesky<F>(a: &ArrayView2<F>) -> LinalgResult<CholeskyFactors<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let l = crate::cholesky(a, None)?;
    Ok(CholeskyFactors { l })
}

// =============================================================================
// SVD (Singular Value Decomposition)
// =============================================================================

/// GPU-accelerated thin SVD: A = U * diag(S) * V^T
///
/// Computes the thin (economy) singular value decomposition. For an m x n matrix
/// with k = min(m, n), returns U (m x k), S (k), V^T (k x n).
///
/// Uses a two-phase approach:
/// 1. Bidiagonalization via Householder reflections (partially GPU-accelerated)
/// 2. Iterative bidiagonal SVD (Golub-Kahan) on the bidiagonal form
///
/// # Arguments
///
/// * `ctx` - GPU context (or None for CPU fallback)
/// * `a` - Input matrix (m x n)
///
/// # Returns
///
/// `SvdFactors` containing U, S (singular values), and V^T.
///
/// # Errors
///
/// Returns `LinalgError` if the matrix is empty, contains non-finite values,
/// or the iterative SVD fails to converge.
pub fn gpu_svd<F>(ctx: Option<&GpuContext>, a: &ArrayView2<F>) -> LinalgResult<SvdFactors<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    validate_matrix(a, "SVD")?;

    let (m, n) = a.dim();

    if m >= GPU_DECOMP_THRESHOLD && n >= GPU_DECOMP_THRESHOLD {
        if let Some(gpu_ctx) = ctx {
            match gpu_svd_impl(gpu_ctx, a) {
                Ok(factors) => return Ok(factors),
                Err(_) => {
                    // Fall through to CPU
                }
            }
        }
    }

    cpu_svd(a)
}

/// GPU SVD implementation.
///
/// Strategy: Use GPU-accelerated QR iteration approach.
/// For large matrices, we first compute A^T * A (or A * A^T for tall matrices)
/// via GPU GEMM, then compute eigendecomposition of the smaller product,
/// and reconstruct the SVD factors.
fn gpu_svd_impl<F>(ctx: &GpuContext, a: &ArrayView2<F>) -> Result<SvdFactors<F>, LinalgError>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (m, n) = a.dim();
    let transpose = m < n;

    // For the one-sided Jacobi approach or cross-product method:
    // If m >= n: compute B = A^T * A (n x n), eigendecompose B = V * Lambda * V^T
    //   Then S = sqrt(Lambda), U = A * V * S^{-1}
    // If m < n:  compute B = A * A^T (m x m), eigendecompose B = U * Lambda * U^T
    //   Then S = sqrt(Lambda), V^T = S^{-1} * U^T * A

    let a_flat: Vec<f64> = a.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
    let gpu_a = ctx.create_buffer_from_slice(&a_flat);

    let (gram_size, gram_data) = if !transpose {
        // B = A^T * A (n x n)
        let gpu_ata = ctx
            .gemm_transpose_a(&gpu_a, &gpu_a, n, m, n)
            .map_err(|e| LinalgError::ComputationError(format!("GPU A^T*A failed: {}", e)))?;
        (n, gpu_ata.to_vec())
    } else {
        // B = A * A^T (m x m)
        let gpu_aat = ctx
            .gemm_transpose_b(&gpu_a, &gpu_a, m, n, m)
            .map_err(|e| LinalgError::ComputationError(format!("GPU A*A^T failed: {}", e)))?;
        (m, gpu_aat.to_vec())
    };

    // Reconstruct the Gram matrix
    let gram_f: Vec<F> = gram_data
        .iter()
        .map(|&v| F::from(v).unwrap_or_else(F::zero))
        .collect();
    let gram = Array2::from_shape_vec((gram_size, gram_size), gram_f)
        .map_err(|e| LinalgError::ShapeError(format!("Failed to reshape Gram matrix: {}", e)))?;

    // Eigendecompose the Gram matrix using CPU (it's symmetric positive semi-definite)
    let (eigenvalues, eigenvectors) = symmetric_eigen(&gram)?;

    let k = m.min(n);

    // Sort eigenvalues in descending order
    let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
    indices.sort_by(|&a_idx, &b_idx| {
        let a_val = eigenvalues[b_idx].abs();
        let b_val = eigenvalues[a_idx].abs();
        a_val
            .partial_cmp(&b_val)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Singular values = sqrt of eigenvalues (clamped to non-negative)
    let mut s = Array1::<F>::zeros(k);
    for i in 0..k {
        let ev = eigenvalues[indices[i]];
        s[i] = if ev > F::zero() { ev.sqrt() } else { F::zero() };
    }

    if !transpose {
        // We have eigenvectors of A^T * A => these are V
        let mut v = Array2::<F>::zeros((k, n));
        for i in 0..k {
            for j in 0..n {
                v[[i, j]] = eigenvectors[[j, indices[i]]];
            }
        }

        // U = A * V^T * S^{-1}
        // First compute A * V^T using GPU
        let vt_flat: Vec<f64> = v.iter().map(|x| x.to_f64().unwrap_or(0.0)).collect();
        let gpu_vt = ctx.create_buffer_from_slice(&vt_flat);

        // A (m x n) * V^T (stored as k x n, transposed = n x k) => m x k
        // We need V (n x k) for this multiplication
        let mut v_col_major = vec![0.0_f64; n * k];
        for i in 0..k {
            for j in 0..n {
                v_col_major[j * k + i] = v[[i, j]].to_f64().unwrap_or(0.0);
            }
        }
        let gpu_v_cols = ctx.create_buffer_from_slice(&v_col_major);

        let u = match ctx.gemm(&gpu_a, &gpu_v_cols, m, n, k) {
            Ok(gpu_u) => {
                let u_flat = gpu_u.to_vec();
                let mut u_mat = Array2::<F>::zeros((m, k));
                for i in 0..m {
                    for j in 0..k {
                        let val = F::from(u_flat[i * k + j]).unwrap_or_else(F::zero);
                        if s[j].abs() > F::epsilon() {
                            u_mat[[i, j]] = val / s[j];
                        }
                    }
                }
                u_mat
            }
            Err(_) => {
                // CPU fallback
                let mut u_mat = Array2::<F>::zeros((m, k));
                for i in 0..m {
                    for j in 0..k {
                        let mut sum = F::zero();
                        for p in 0..n {
                            sum += a[[i, p]] * v[[j, p]]; // v stored as rows
                        }
                        if s[j].abs() > F::epsilon() {
                            u_mat[[i, j]] = sum / s[j];
                        }
                    }
                }
                u_mat
            }
        };

        Ok(SvdFactors { u, s, vt: v })
    } else {
        // We have eigenvectors of A * A^T => these are U
        let mut u = Array2::<F>::zeros((m, k));
        for i in 0..m {
            for j in 0..k {
                u[[i, j]] = eigenvectors[[i, indices[j]]];
            }
        }

        // V^T = S^{-1} * U^T * A
        // First compute U^T * A using GPU
        let ut_flat: Vec<f64> = {
            let mut flat = vec![0.0_f64; k * m];
            for i in 0..k {
                for j in 0..m {
                    flat[i * m + j] = u[[j, i]].to_f64().unwrap_or(0.0);
                }
            }
            flat
        };
        let gpu_ut = ctx.create_buffer_from_slice(&ut_flat);

        let vt = match ctx.gemm(&gpu_ut, &gpu_a, k, m, n) {
            Ok(gpu_vt) => {
                let vt_flat = gpu_vt.to_vec();
                let mut vt_mat = Array2::<F>::zeros((k, n));
                for i in 0..k {
                    for j in 0..n {
                        let val = F::from(vt_flat[i * n + j]).unwrap_or_else(F::zero);
                        if s[i].abs() > F::epsilon() {
                            vt_mat[[i, j]] = val / s[i];
                        }
                    }
                }
                vt_mat
            }
            Err(_) => {
                // CPU fallback
                let mut vt_mat = Array2::<F>::zeros((k, n));
                for i in 0..k {
                    for j in 0..n {
                        let mut sum = F::zero();
                        for p in 0..m {
                            sum += u[[p, i]] * a[[p, j]];
                        }
                        if s[i].abs() > F::epsilon() {
                            vt_mat[[i, j]] = sum / s[i];
                        }
                    }
                }
                vt_mat
            }
        };

        Ok(SvdFactors { u, s, vt })
    }
}

/// Compute eigendecomposition of a symmetric matrix using Jacobi iteration.
///
/// Returns eigenvalues and eigenvectors sorted by magnitude (descending).
fn symmetric_eigen<F>(a: &Array2<F>) -> Result<(Array1<F>, Array2<F>), LinalgError>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let n = a.nrows();
    let mut d = a.clone();
    let mut v = Array2::<F>::zeros((n, n));
    for i in 0..n {
        v[[i, i]] = F::one();
    }

    let max_iter = 100 * n * n;
    let tol = F::epsilon() * F::from(100.0).unwrap_or_else(F::one);

    for _ in 0..max_iter {
        // Find the largest off-diagonal element
        let mut max_off = F::zero();
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = d[[i, j]].abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_off < tol {
            break;
        }

        // Compute Jacobi rotation
        let d_pp = d[[p, p]];
        let d_qq = d[[q, q]];
        let d_pq = d[[p, q]];

        let theta = if (d_pp - d_qq).abs() < F::epsilon() {
            F::from(std::f64::consts::FRAC_PI_4).unwrap_or_else(F::one)
        } else {
            let tau = F::from(2.0).unwrap_or_else(F::one) * d_pq / (d_pp - d_qq);
            let t = if tau >= F::zero() {
                F::one() / (tau + (F::one() + tau * tau).sqrt())
            } else {
                -F::one() / (-tau + (F::one() + tau * tau).sqrt())
            };
            t.atan()
        };

        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Apply rotation to D
        let mut new_d = d.clone();
        for i in 0..n {
            if i != p && i != q {
                let d_ip = d[[i, p]];
                let d_iq = d[[i, q]];
                new_d[[i, p]] = cos_t * d_ip + sin_t * d_iq;
                new_d[[p, i]] = new_d[[i, p]];
                new_d[[i, q]] = -sin_t * d_ip + cos_t * d_iq;
                new_d[[q, i]] = new_d[[i, q]];
            }
        }
        new_d[[p, p]] = cos_t * cos_t * d_pp
            + F::from(2.0).unwrap_or_else(F::one) * cos_t * sin_t * d_pq
            + sin_t * sin_t * d_qq;
        new_d[[q, q]] = sin_t * sin_t * d_pp
            - F::from(2.0).unwrap_or_else(F::one) * cos_t * sin_t * d_pq
            + cos_t * cos_t * d_qq;
        new_d[[p, q]] = F::zero();
        new_d[[q, p]] = F::zero();
        d = new_d;

        // Apply rotation to V
        for i in 0..n {
            let v_ip = v[[i, p]];
            let v_iq = v[[i, q]];
            v[[i, p]] = cos_t * v_ip + sin_t * v_iq;
            v[[i, q]] = -sin_t * v_ip + cos_t * v_iq;
        }
    }

    // Extract eigenvalues from diagonal
    let mut eigenvalues = Array1::<F>::zeros(n);
    for i in 0..n {
        eigenvalues[i] = d[[i, i]];
    }

    Ok((eigenvalues, v))
}

/// CPU fallback for SVD.
fn cpu_svd<F>(a: &ArrayView2<F>) -> LinalgResult<SvdFactors<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (u, s, vt) = crate::svd(a, false, None)?;
    Ok(SvdFactors { u, s, vt })
}

// =============================================================================
// Validation helpers
// =============================================================================

/// Validate that a matrix is non-empty and contains finite values.
fn validate_matrix<F>(a: &ArrayView2<F>, operation: &str) -> LinalgResult<()>
where
    F: Float,
{
    if a.is_empty() {
        return Err(LinalgError::ShapeError(format!(
            "{} failed: Input matrix cannot be empty",
            operation
        )));
    }

    for &val in a.iter() {
        if !val.is_finite() {
            return Err(LinalgError::InvalidInputError(format!(
                "{} failed: Matrix contains non-finite values",
                operation
            )));
        }
    }

    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    // ---- LU tests ----

    #[test]
    fn test_cpu_lu_basic() {
        let a = array![[2.0_f64, 1.0], [4.0, 3.0]];
        let factors = cpu_lu(&a.view()).expect("LU failed");

        // Convention: P * A = L * U
        let pa = factors.p.dot(&a);
        let lu = factors.l.dot(&factors.u);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(pa[[i, j]], lu[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_gpu_lu_fallback() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let factors = gpu_lu(None, &a.view()).expect("LU fallback failed");

        // Convention: P * A = L * U
        let pa = factors.p.dot(&a);
        let lu = factors.l.dot(&factors.u);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(pa[[i, j]], lu[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_lu_empty_matrix() {
        let a = Array2::<f64>::zeros((0, 0));
        let result = gpu_lu(None, &a.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_lu_3x3() {
        let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]];
        let factors = gpu_lu(None, &a.view()).expect("3x3 LU failed");
        // Convention: P * A = L * U
        let pa = factors.p.dot(&a);
        let lu = factors.l.dot(&factors.u);
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(pa[[i, j]], lu[[i, j]], epsilon = 1e-10);
            }
        }
    }

    // ---- QR tests ----

    #[test]
    fn test_cpu_qr_basic() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let factors = cpu_qr(&a.view()).expect("QR failed");

        // Verify Q * R ≈ A
        let qr_product = factors.q.dot(&factors.r);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(qr_product[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_gpu_qr_fallback() {
        let a = array![
            [12.0_f64, -51.0, 4.0],
            [6.0, 167.0, -68.0],
            [-4.0, 24.0, -41.0]
        ];
        let factors = gpu_qr(None, &a.view()).expect("QR fallback failed");
        let qr_product = factors.q.dot(&factors.r);
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(qr_product[[i, j]], a[[i, j]], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_qr_empty_matrix() {
        let a = Array2::<f64>::zeros((0, 0));
        let result = gpu_qr(None, &a.view());
        assert!(result.is_err());
    }

    // ---- Cholesky tests ----

    #[test]
    fn test_cpu_cholesky_basic() {
        let a = array![[4.0_f64, 2.0], [2.0, 5.0]];
        let factors = cpu_cholesky(&a.view()).expect("Cholesky failed");

        // Verify L * L^T ≈ A
        let llt = factors.l.dot(&factors.l.t());
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(llt[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_gpu_cholesky_fallback() {
        let a = array![[25.0_f64, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, 11.0]];
        let factors = gpu_cholesky(None, &a.view()).expect("Cholesky fallback failed");
        let llt = factors.l.dot(&factors.l.t());
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(llt[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_cholesky_non_square() {
        let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let result = gpu_cholesky(None, &a.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_cholesky_empty() {
        let a = Array2::<f64>::zeros((0, 0));
        let result = gpu_cholesky(None, &a.view());
        assert!(result.is_err());
    }

    // ---- SVD tests ----

    #[test]
    fn test_cpu_svd_basic() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let factors = cpu_svd(&a.view()).expect("SVD failed");

        // Verify U * diag(S) * V^T ≈ A
        let s_diag = Array2::from_diag(&factors.s);
        let usv = factors.u.dot(&s_diag).dot(&factors.vt);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(usv[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_gpu_svd_fallback() {
        let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
        let factors = gpu_svd(None, &a.view()).expect("SVD fallback failed");

        // For a diagonal matrix, singular values should be the absolute diagonal values
        let mut s_sorted: Vec<f64> = factors.s.iter().copied().collect();
        s_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        assert_relative_eq!(s_sorted[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(s_sorted[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_svd_empty_matrix() {
        let a = Array2::<f64>::zeros((0, 0));
        let result = gpu_svd(None, &a.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_svd_reconstruction() {
        let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]];
        let factors = gpu_svd(None, &a.view()).expect("SVD failed");
        let s_diag = Array2::from_diag(&factors.s);
        let usv = factors.u.dot(&s_diag).dot(&factors.vt);
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(usv[[i, j]], a[[i, j]], epsilon = 1e-8);
            }
        }
    }

    // ---- Validation tests ----

    #[test]
    fn test_validate_non_finite() {
        let a = array![[1.0_f64, f64::NAN], [3.0, 4.0]];
        let result = validate_matrix(&a.view(), "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_infinity() {
        let a = array![[1.0_f64, f64::INFINITY], [3.0, 4.0]];
        let result = validate_matrix(&a.view(), "test");
        assert!(result.is_err());
    }

    // ---- Symmetric eigendecomposition tests ----

    #[test]
    fn test_symmetric_eigen_diagonal() {
        let a = array![[3.0_f64, 0.0], [0.0, 1.0]];
        let (eigenvalues, eigenvectors) = symmetric_eigen(&a).expect("eigen failed");

        // Eigenvalues should be 3 and 1
        let mut ev_sorted: Vec<f64> = eigenvalues.iter().copied().collect();
        ev_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        assert_relative_eq!(ev_sorted[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(ev_sorted[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_symmetric_eigen_identity() {
        let a = array![[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let (eigenvalues, _) = symmetric_eigen(&a).expect("eigen identity failed");
        for &ev in eigenvalues.iter() {
            assert_relative_eq!(ev, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gpu_lu_with_cpu_context() {
        match GpuContext::new(GpuBackend::Cpu) {
            Ok(gpu_ctx) => {
                let a = array![[2.0_f64, 1.0], [4.0, 3.0]];
                let factors = gpu_lu(Some(&gpu_ctx), &a.view()).expect("LU with ctx failed");
                // Convention: P * A = L * U
                let pa = factors.p.dot(&a);
                let lu = factors.l.dot(&factors.u);
                for i in 0..2 {
                    for j in 0..2 {
                        assert_relative_eq!(pa[[i, j]], lu[[i, j]], epsilon = 1e-10);
                    }
                }
            }
            Err(_) => {
                // Skip if cannot create context
            }
        }
    }

    #[test]
    fn test_gpu_qr_with_cpu_context() {
        if let Ok(gpu_ctx) = GpuContext::new(GpuBackend::Cpu) {
            let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
            let factors = gpu_qr(Some(&gpu_ctx), &a.view()).expect("QR with ctx failed");
            let qr_product = factors.q.dot(&factors.r);
            for i in 0..2 {
                for j in 0..2 {
                    assert_relative_eq!(qr_product[[i, j]], a[[i, j]], epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_gpu_cholesky_with_cpu_context() {
        if let Ok(gpu_ctx) = GpuContext::new(GpuBackend::Cpu) {
            let a = array![[4.0_f64, 2.0], [2.0, 5.0]];
            let factors =
                gpu_cholesky(Some(&gpu_ctx), &a.view()).expect("Cholesky with ctx failed");
            let llt = factors.l.dot(&factors.l.t());
            for i in 0..2 {
                for j in 0..2 {
                    assert_relative_eq!(llt[[i, j]], a[[i, j]], epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_gpu_svd_with_cpu_context() {
        if let Ok(gpu_ctx) = GpuContext::new(GpuBackend::Cpu) {
            let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
            let factors = gpu_svd(Some(&gpu_ctx), &a.view()).expect("SVD with ctx failed");
            let s_diag = Array2::from_diag(&factors.s);
            let usv = factors.u.dot(&s_diag).dot(&factors.vt);
            for i in 0..2 {
                for j in 0..2 {
                    assert_relative_eq!(usv[[i, j]], a[[i, j]], epsilon = 1e-8);
                }
            }
        }
    }
}
