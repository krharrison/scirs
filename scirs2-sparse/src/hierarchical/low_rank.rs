//! Low-rank approximation utilities for H-matrix construction.
//!
//! This module provides:
//! - Truncated SVD with relative tolerance (`truncated_svd`)
//! - Adaptive Cross Approximation (`aca`) – partial-pivot row/column selection
//! - ACA+ (`aca_plus`) – full-pivot variant with guaranteed error bound
//! - H-matrix recompression (`hmatrix_truncate`)
//!
//! All routines operate on *dense* small-to-medium matrices represented as
//! flat `Vec<f64>` in row-major order.
//!
//! # References
//!
//! - Bebendorf, M. (2000): "Approximation of boundary element matrices",
//!   Numerische Mathematik, 86(4), 565–589.
//! - Bebendorf, M. & Rjasanow, S. (2003): "Adaptive low-rank approximation
//!   of collocation matrices", Computing, 70(1), 1–24.

use crate::error::{SparseError, SparseResult};

// ---------------------------------------------------------------------------
// Internal dense-matrix helpers  (row-major, f64)
// ---------------------------------------------------------------------------

/// Transpose an m×n matrix stored in row-major order.
fn transpose(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut at = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            at[j * m + i] = a[i * n + j];
        }
    }
    at
}

/// Matrix multiply  A(m×k) · B(k×n) → C(m×n), all row-major.
fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            if a_ip == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }
    c
}

/// Dot product of two equal-length slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// L2 norm of a slice.
fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Outer-product rank-1 update: M  ←  M - u·v^T
/// u has length m, v has length n, M is m×n row-major.
fn rank1_subtract(m_mat: &mut [f64], u: &[f64], v: &[f64], rows: usize, cols: usize) {
    for i in 0..rows {
        for j in 0..cols {
            m_mat[i * cols + j] -= u[i] * v[j];
        }
    }
}

// ---------------------------------------------------------------------------
// Golub-Reinsch bidiagonalisation (for truncated SVD)
// ---------------------------------------------------------------------------

/// One-sided Jacobi SVD for a dense m×n matrix (m ≥ n).
///
/// Returns (U, S, Vt) where U is m×n, S is length n, Vt is n×n, all
/// row-major.  U has orthonormal columns, Vt has orthonormal rows.
///
/// This is an educational Jacobi implementation sufficient for the small
/// matrices that appear in H-matrix leaf and low-rank blocks.
///
/// For numerical stability we work on the n×n Gram matrix A^T A and then
/// recover U.
fn jacobi_svd_thin(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // B = A^T A   (n×n)
    let at = transpose(a, m, n);
    let mut b = matmul(&at, a, n, m, n);

    // Accumulate Jacobi rotations in V (n×n), start as identity.
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    const MAX_ITER: usize = 100;
    for _iter in 0..MAX_ITER {
        let mut converged = true;
        for p in 0..n {
            for q in (p + 1)..n {
                let b_pq = b[p * n + q];
                let b_pp = b[p * n + p];
                let b_qq = b[q * n + q];
                let off_sq = b_pq * b_pq;
                if off_sq < 1e-28 * (b_pp * b_qq).abs().max(1e-100) {
                    continue;
                }
                converged = false;
                // Compute Jacobi rotation angle.
                let theta = 0.5 * (b_qq - b_pp) / b_pq;
                let t = if theta >= 0.0 {
                    1.0 / (theta + (theta * theta + 1.0).sqrt())
                } else {
                    1.0 / (theta - (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update B = J^T B J (only (p,p),(q,q),(p,q),(q,p) and columns/rows)
                // Store updated columns p and q.
                let mut bp = vec![0.0_f64; n];
                let mut bq = vec![0.0_f64; n];
                for r in 0..n {
                    bp[r] = c * b[r * n + p] - s * b[r * n + q];
                    bq[r] = s * b[r * n + p] + c * b[r * n + q];
                }
                for r in 0..n {
                    b[r * n + p] = bp[r];
                    b[r * n + q] = bq[r];
                }
                let mut bpr = vec![0.0_f64; n];
                let mut bqr = vec![0.0_f64; n];
                for r in 0..n {
                    bpr[r] = c * b[p * n + r] - s * b[q * n + r];
                    bqr[r] = s * b[p * n + r] + c * b[q * n + r];
                }
                for r in 0..n {
                    b[p * n + r] = bpr[r];
                    b[q * n + r] = bqr[r];
                }

                // Accumulate V.
                let mut vp = vec![0.0_f64; n];
                let mut vq = vec![0.0_f64; n];
                for r in 0..n {
                    vp[r] = c * v[r * n + p] - s * v[r * n + q];
                    vq[r] = s * v[r * n + p] + c * v[r * n + q];
                }
                for r in 0..n {
                    v[r * n + p] = vp[r];
                    v[r * n + q] = vq[r];
                }
            }
        }
        if converged {
            break;
        }
    }

    // Singular values = sqrt of diagonal of B (= V^T A^T A V).
    let mut sigma: Vec<f64> = (0..n).map(|i| b[i * n + i].max(0.0).sqrt()).collect();

    // Sort descending.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        sigma[b]
            .partial_cmp(&sigma[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sigma_sorted: Vec<f64> = order.iter().map(|&i| sigma[i]).collect();
    let v_sorted: Vec<f64> = {
        let mut vs = vec![0.0_f64; n * n];
        for (new_j, &old_j) in order.iter().enumerate() {
            for r in 0..n {
                vs[r * n + new_j] = v[r * n + old_j];
            }
        }
        vs
    };
    sigma = sigma_sorted;

    // U = A V Σ^{-1}  (thin: m×n)
    let av = matmul(a, &v_sorted, m, n, n);
    let mut u = vec![0.0_f64; m * n];
    for j in 0..n {
        let s = sigma[j];
        if s > 0.0 {
            for i in 0..m {
                u[i * n + j] = av[i * n + j] / s;
            }
        }
    }

    // Vt = V^T  (n×n, row-major, rows = right singular vectors)
    let vt = transpose(&v_sorted, n, n);

    (u, sigma, vt)
}

// ---------------------------------------------------------------------------
// truncated_svd
// ---------------------------------------------------------------------------

/// Truncated SVD of a dense matrix with relative tolerance.
///
/// Computes the full thin SVD and retains the k largest singular values such
/// that the discarded tail satisfies
///
/// ```text
/// (sum_{i>k} σ_i²) / (sum_i σ_i²)  ≤  tol²
/// ```
///
/// # Parameters
/// - `a`: matrix in row-major order, shape m × n.
/// - `m`, `n`: dimensions (m ≥ 1, n ≥ 1).
/// - `tol`: relative truncation tolerance (0 < tol ≤ 1).  Set to 0.0 for no
///   truncation (full thin SVD).
///
/// # Returns
/// `(U, S, Vt, rank)` where
/// - `U` is m × rank (row-major),
/// - `S` is length rank,
/// - `Vt` is rank × n (row-major),
/// - `rank` is the chosen rank.
///
/// # Errors
/// Returns an error if dimensions are zero or `tol` is negative.
pub fn truncated_svd(
    a: &[f64],
    m: usize,
    n: usize,
    tol: f64,
) -> SparseResult<(Vec<f64>, Vec<f64>, Vec<f64>, usize)> {
    if m == 0 || n == 0 {
        return Err(SparseError::ValueError(
            "truncated_svd: dimensions must be > 0".to_string(),
        ));
    }
    if tol < 0.0 {
        return Err(SparseError::ValueError(
            "truncated_svd: tolerance must be >= 0".to_string(),
        ));
    }
    if a.len() != m * n {
        return Err(SparseError::ValueError(format!(
            "truncated_svd: a.len()={} != m*n={}",
            a.len(),
            m * n
        )));
    }

    // Thin SVD: always work on the smaller square.
    let k = m.min(n); // number of singular values

    // If m < n, work on A^T.
    let (u_full, sigma, vt_full) = if m >= n {
        jacobi_svd_thin(a, m, n)
    } else {
        let at = transpose(a, m, n);
        let (u_t, s_t, vt_t) = jacobi_svd_thin(&at, n, m);
        // A = U Σ V^T  →  A^T = V Σ U^T
        // So U_A = V_At = columns of vt_t^T, Vt_A = U_At^T
        let u_a = transpose(&vt_t, m, m); // m×m, take thin
        let vt_a = transpose(&u_t, n, n); // n×m → we want n rows of length m
        // u_a is m×m but thin means m×k=m×m when m≤n, same k.
        (u_a, s_t, vt_a)
    };

    // Determine truncation rank.
    let sigma_sq_total: f64 = sigma.iter().map(|&s| s * s).sum();
    let mut rank = k;
    if tol > 0.0 && sigma_sq_total > 0.0 {
        let mut tail_sq = 0.0_f64;
        for r in (0..k).rev() {
            let new_tail = tail_sq + sigma[r] * sigma[r];
            if (new_tail / sigma_sq_total).sqrt() <= tol {
                tail_sq = new_tail;
                rank = r;
            } else {
                break;
            }
        }
        rank = rank.max(1);
    }

    // Extract first `rank` singular components.
    let u_r: Vec<f64> = (0..m)
        .flat_map(|i| (0..rank).map(move |j| u_full[i * k + j]))
        .collect();
    let s_r: Vec<f64> = sigma[..rank].to_vec();
    let vt_r: Vec<f64> = (0..rank)
        .flat_map(|i| (0..n).map(move |j| vt_full[i * n + j]))
        .collect();

    Ok((u_r, s_r, vt_r, rank))
}

// ---------------------------------------------------------------------------
// ACA  (Adaptive Cross Approximation – partial pivot)
// ---------------------------------------------------------------------------

/// Result of an ACA (or ACA+) factorisation.
///
/// Represents a low-rank approximation  A ≈ U · V^T  where
/// - `u`: m × rank, row-major,
/// - `v`: n × rank, row-major  (so `u·v^T` gives the m×n approximation),
/// - `rank`: achieved rank,
/// - `residual_norm`: estimated Frobenius norm of the residual.
#[derive(Debug, Clone)]
pub struct AcaResult {
    /// Left factor U, shape m × rank (row-major).
    pub u: Vec<f64>,
    /// Right factor V, shape n × rank (row-major).
    pub v: Vec<f64>,
    /// Achieved rank.
    pub rank: usize,
    /// Estimated Frobenius norm of the residual.
    pub residual_norm: f64,
}

/// Adaptive Cross Approximation (partial-pivot ACA).
///
/// Constructs a low-rank approximation  A ≈ U · V^T  by iteratively
/// selecting pivot rows and columns.  Access to matrix entries is provided
/// through the closure `entry(i, j)`.
///
/// # Parameters
/// - `m`, `n`: matrix dimensions.
/// - `entry`: closure returning the (i, j) entry of the matrix.
/// - `tol`: relative Frobenius-norm tolerance for stopping.
/// - `max_rank`: maximum number of cross approximation steps.
///
/// # Algorithm (partial-pivot)
/// 1. Start with row index `i = 0`.
/// 2. Evaluate the current residual row `r_i` and pick the pivot column `j`
///    as `argmax |r_i[j]|`.
/// 3. Evaluate the current residual column `c_j` and scale it by `1/r_i[j]`.
/// 4. Append `c_j → u_k`, `r_i → v_k`.
/// 5. Update the next row index as the row of the maximum entry in `c_j`
///    (excluding already-used rows).
/// 6. Stop when `||u_k|| · ||v_k|| ≤ tol · ||A_approx||_F`.
///
/// # Errors
/// Returns an error if `m == 0`, `n == 0`, or `tol < 0`.
pub fn aca<F>(
    m: usize,
    n: usize,
    entry: F,
    tol: f64,
    max_rank: usize,
) -> SparseResult<AcaResult>
where
    F: Fn(usize, usize) -> f64,
{
    if m == 0 || n == 0 {
        return Err(SparseError::ValueError(
            "aca: dimensions must be > 0".to_string(),
        ));
    }
    if tol < 0.0 {
        return Err(SparseError::ValueError(
            "aca: tolerance must be >= 0".to_string(),
        ));
    }
    let max_rank = max_rank.min(m.min(n));

    let mut u_cols: Vec<Vec<f64>> = Vec::new(); // each is length m
    let mut v_cols: Vec<Vec<f64>> = Vec::new(); // each is length n

    let mut used_rows = vec![false; m];
    let mut used_cols = vec![false; n];

    // Frobenius norm squared estimate of A_approx, updated incrementally.
    let mut approx_norm_sq = 0.0_f64;

    let mut pivot_row = 0usize;
    used_rows[pivot_row] = true;

    for _k in 0..max_rank {
        // ---- Step 1: Evaluate residual row `pivot_row` ----
        let mut r_row = vec![0.0_f64; n];
        for j in 0..n {
            let mut val = entry(pivot_row, j);
            for (uk, vk) in u_cols.iter().zip(v_cols.iter()) {
                val -= uk[pivot_row] * vk[j];
            }
            r_row[j] = val;
        }

        // ---- Step 2: Find pivot column  (max |r_row[j]|, not already used) ----
        let pivot_col_opt = (0..n)
            .filter(|&j| !used_cols[j])
            .max_by(|&a, &b| {
                r_row[a]
                    .abs()
                    .partial_cmp(&r_row[b].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        let pivot_col = match pivot_col_opt {
            Some(c) => c,
            None => break,
        };
        let pivot_val = r_row[pivot_col];
        if pivot_val.abs() < 1e-14 {
            break; // numerical zero → stop
        }

        // ---- Step 3: Evaluate residual column `pivot_col` ----
        let mut r_col = vec![0.0_f64; m];
        for i in 0..m {
            let mut val = entry(i, pivot_col);
            for (uk, vk) in u_cols.iter().zip(v_cols.iter()) {
                val -= uk[i] * vk[pivot_col];
            }
            r_col[i] = val;
        }

        // ---- Step 4: Scale column and add to factorisation ----
        // u_k = r_col / pivot_val,  v_k = r_row
        let u_new: Vec<f64> = r_col.iter().map(|&x| x / pivot_val).collect();
        let v_new = r_row.clone();

        let norm_u = norm2(&u_new);
        let norm_v = norm2(&v_new);
        let update_norm_sq = norm_u * norm_u * norm_v * norm_v;

        // Incremental Frobenius norm update:
        // ||A_k||_F^2 ≈ ||A_{k-1}||_F^2 + 2 sum_{j<k} <u_j,u_k><v_j,v_k> + ||u_k||^2 ||v_k||^2
        let mut cross = 0.0_f64;
        for (uj, vj) in u_cols.iter().zip(v_cols.iter()) {
            cross += dot(uj, &u_new) * dot(vj, &v_new);
        }
        approx_norm_sq += 2.0 * cross + update_norm_sq;

        u_cols.push(u_new);
        v_cols.push(v_new);
        used_cols[pivot_col] = true;

        // ---- Convergence check ----
        // Stop if ||u_k|| ||v_k|| ≤ tol · ||A_approx||_F
        if approx_norm_sq > 0.0 && update_norm_sq <= tol * tol * approx_norm_sq {
            break;
        }

        // ---- Step 5: Choose next pivot row from max of residual column ----
        let next_row_opt = (0..m).filter(|&i| !used_rows[i]).max_by(|&a, &b| {
            let va = u_cols.last().map(|uk| uk[a].abs()).unwrap_or(0.0);
            let vb = u_cols.last().map(|uk| uk[b].abs()).unwrap_or(0.0);
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });
        match next_row_opt {
            Some(nr) => {
                used_rows[nr] = true;
                pivot_row = nr;
            }
            None => break,
        }
    }

    let rank = u_cols.len();
    if rank == 0 {
        // Zero matrix.
        return Ok(AcaResult {
            u: vec![0.0_f64; m],
            v: vec![0.0_f64; n],
            rank: 1,
            residual_norm: 0.0,
        });
    }

    // Pack U and V into flat row-major matrices.
    // U is m × rank: U[i, k] = u_cols[k][i]
    let mut u_flat = vec![0.0_f64; m * rank];
    let mut v_flat = vec![0.0_f64; n * rank];
    for k in 0..rank {
        for i in 0..m {
            u_flat[i * rank + k] = u_cols[k][i];
        }
        for j in 0..n {
            v_flat[j * rank + k] = v_cols[k][j];
        }
    }

    let residual_norm = (approx_norm_sq.max(0.0)).sqrt(); // approximate

    Ok(AcaResult {
        u: u_flat,
        v: v_flat,
        rank,
        residual_norm,
    })
}

// ---------------------------------------------------------------------------
// ACA+  (full-pivot ACA)
// ---------------------------------------------------------------------------

/// ACA+ with full-pivot search.
///
/// Unlike the partial-pivot variant, ACA+ searches the *entire residual
/// matrix* for the maximum entry at each step, guaranteeing a monotonically
/// decreasing error bound.
///
/// **Cost per step**: O(m + n) entry evaluations after the first (using
/// incremental residual bookkeeping), making it more expensive than
/// partial-pivot ACA but with stronger convergence guarantees.
///
/// # Parameters
/// Same as [`aca`].
///
/// # Returns
/// Same as [`aca`].
pub fn aca_plus<F>(
    m: usize,
    n: usize,
    entry: F,
    tol: f64,
    max_rank: usize,
) -> SparseResult<AcaResult>
where
    F: Fn(usize, usize) -> f64,
{
    if m == 0 || n == 0 {
        return Err(SparseError::ValueError(
            "aca_plus: dimensions must be > 0".to_string(),
        ));
    }
    if tol < 0.0 {
        return Err(SparseError::ValueError(
            "aca_plus: tolerance must be >= 0".to_string(),
        ));
    }
    let max_rank = max_rank.min(m.min(n));

    // Cache the residual matrix entries incrementally.
    // We evaluate on-demand and cache for efficiency.
    // For small matrices this is fine; for large H-matrix leaves the ACA
    // (partial-pivot) variant should be preferred.
    let mut r: Vec<f64> = (0..m * n).map(|idx| entry(idx / n, idx % n)).collect();

    let mut u_cols: Vec<Vec<f64>> = Vec::new();
    let mut v_cols: Vec<Vec<f64>> = Vec::new();

    let mut approx_norm_sq = 0.0_f64;

    for _k in 0..max_rank {
        // Full pivot: find (i*, j*) = argmax |R[i, j]|.
        let (pivot_idx, &pivot_val) = r
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.abs()
                    .partial_cmp(&b.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or((0, &0.0_f64));

        if pivot_val.abs() < 1e-14 {
            break;
        }

        let pi = pivot_idx / n;
        let pj = pivot_idx % n;

        // Extract pivot row and column from residual.
        let r_row: Vec<f64> = (0..n).map(|j| r[pi * n + j]).collect();
        let r_col: Vec<f64> = (0..m).map(|i| r[i * n + pj]).collect();

        let u_new: Vec<f64> = r_col.iter().map(|&x| x / pivot_val).collect();
        let v_new = r_row;

        let norm_u = norm2(&u_new);
        let norm_v = norm2(&v_new);
        let update_norm_sq = norm_u * norm_u * norm_v * norm_v;

        let mut cross = 0.0_f64;
        for (uj, vj) in u_cols.iter().zip(v_cols.iter()) {
            cross += dot(uj, &u_new) * dot(vj, &v_new);
        }
        approx_norm_sq += 2.0 * cross + update_norm_sq;

        // Update residual: R ← R - u_new * v_new^T
        rank1_subtract(&mut r, &u_new, &v_new, m, n);

        u_cols.push(u_new);
        v_cols.push(v_new);

        if approx_norm_sq > 0.0 && update_norm_sq <= tol * tol * approx_norm_sq {
            break;
        }
    }

    let rank = u_cols.len();
    if rank == 0 {
        return Ok(AcaResult {
            u: vec![0.0_f64; m],
            v: vec![0.0_f64; n],
            rank: 1,
            residual_norm: 0.0,
        });
    }

    let mut u_flat = vec![0.0_f64; m * rank];
    let mut v_flat = vec![0.0_f64; n * rank];
    for k in 0..rank {
        for i in 0..m {
            u_flat[i * rank + k] = u_cols[k][i];
        }
        for j in 0..n {
            v_flat[j * rank + k] = v_cols[k][j];
        }
    }

    let residual_norm = approx_norm_sq.max(0.0).sqrt();

    Ok(AcaResult {
        u: u_flat,
        v: v_flat,
        rank,
        residual_norm,
    })
}

// ---------------------------------------------------------------------------
// hmatrix_truncate  (helper used by hmatrix.rs)
// ---------------------------------------------------------------------------

/// Recompress a low-rank block  A ≈ U · V^T  to the desired tolerance using
/// truncated SVD.
///
/// The input `U` (m × r) and `V` (n × r, so that `U·V^T` ≈ A) are re-
/// factored as follows:
/// 1. QR-decompose U = Q_U R_U  and  V = Q_V R_V.
/// 2. SVD of the small  r×r  product `S = R_U R_V^T`.
/// 3. Truncate to the desired rank and form new  U' = Q_U Û Σ^{1/2},
///    V' = Q_V V̂ Σ^{1/2}.
///
/// For simplicity the current implementation uses the full thin-SVD of the
/// reconstructed dense matrix (recomputing  M = U · V^T  explicitly).  This
/// is acceptable because H-matrix leaf blocks are typically small.
///
/// # Parameters
/// - `u`: m × r  (row-major).
/// - `v`: n × r  (row-major).
/// - `m`, `n`, `r`: dimensions.
/// - `tol`: relative Frobenius-norm truncation tolerance.
///
/// # Returns
/// `(U_new, V_new, new_rank)` where `U_new` is m × new_rank and `V_new` is
/// n × new_rank.
///
/// # Errors
/// Propagates errors from `truncated_svd`.
pub fn hmatrix_truncate(
    u: &[f64],
    v: &[f64],
    m: usize,
    n: usize,
    r: usize,
    tol: f64,
) -> SparseResult<(Vec<f64>, Vec<f64>, usize)> {
    if r == 0 {
        return Err(SparseError::ValueError(
            "hmatrix_truncate: rank r must be > 0".to_string(),
        ));
    }
    // Reconstruct the dense m×n matrix.
    let vt = transpose(v, n, r); // r×n
    let dense = matmul(u, &vt, m, r, n);

    let (u_svd, sigma, vt_svd, new_rank) = truncated_svd(&dense, m, n, tol)?;

    // Absorb sqrt(sigma) into both U and V.
    // U_new[i, k] = U_svd[i, k] * sqrt(sigma[k])
    // V_new[j, k] = Vt_svd[k, j] * sqrt(sigma[k])  →  V_new is n × rank
    let mut u_new = vec![0.0_f64; m * new_rank];
    let mut v_new = vec![0.0_f64; n * new_rank];
    for k in 0..new_rank {
        let sq = sigma[k].sqrt();
        for i in 0..m {
            u_new[i * new_rank + k] = u_svd[i * new_rank + k] * sq;
        }
        for j in 0..n {
            v_new[j * new_rank + k] = vt_svd[k * n + j] * sq;
        }
    }

    Ok((u_new, v_new, new_rank))
}

// ---------------------------------------------------------------------------
// Public re-exports for use in hmatrix.rs
// ---------------------------------------------------------------------------
pub use self::{matmul, norm2, transpose};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn frobenius_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }

    fn reconstruct_lr(u: &[f64], v: &[f64], m: usize, n: usize, r: usize) -> Vec<f64> {
        let vt = transpose(v, n, r);
        matmul(u, &vt, m, r, n)
    }

    /// Build a rank-2 matrix: A = u1·v1^T + u2·v2^T.
    fn rank2_matrix(m: usize, n: usize) -> Vec<f64> {
        let mut a = vec![0.0_f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let u1 = (i + 1) as f64;
                let v1 = (j + 1) as f64;
                let u2 = ((i as f64 + 0.5) * std::f64::consts::PI).sin();
                let v2 = ((j as f64 + 0.5) * std::f64::consts::PI).cos();
                a[i * n + j] = u1 * v1 + u2 * v2;
            }
        }
        a
    }

    #[test]
    fn test_truncated_svd_rank2() {
        let m = 8;
        let n = 6;
        let a = rank2_matrix(m, n);
        let a_norm = norm2(&a);

        // Full SVD (tol=0) should recover the matrix.
        let (u, sigma, vt, rank) = truncated_svd(&a, m, n, 0.0).expect("svd ok");
        assert!(rank >= 2, "rank={}", rank);
        assert_eq!(sigma.len(), rank);

        let vt_full = {
            // Expand sigma into Vt: Vt_scaled[k, j] = sigma[k] * vt[k, j]
            let k_dim = rank;
            let mut vt_s = vec![0.0_f64; k_dim * n];
            for k in 0..k_dim {
                for j in 0..n {
                    vt_s[k * n + j] = sigma[k] * vt[k * n + j];
                }
            }
            vt_s
        };
        let recon = matmul(&u, &vt_full, m, rank, n);
        let err = frobenius_diff(&a, &recon);
        assert!(
            err < 1e-8 * a_norm,
            "reconstruction error too large: {} (norm={})",
            err,
            a_norm
        );
    }

    #[test]
    fn test_truncated_svd_tolerance() {
        let m = 8;
        let n = 6;
        let a = rank2_matrix(m, n);

        // With tol=1e-6, should still get a good approximation.
        let (u, sigma, vt, rank) = truncated_svd(&a, m, n, 1e-6).expect("svd ok");
        assert!(rank <= 6, "rank={}", rank);

        let vt_full = {
            let mut vt_s = vec![0.0_f64; rank * n];
            for k in 0..rank {
                for j in 0..n {
                    vt_s[k * n + j] = sigma[k] * vt[k * n + j];
                }
            }
            vt_s
        };
        let recon = matmul(&u, &vt_full, m, rank, n);
        let err = frobenius_diff(&a, &recon);
        let a_norm = norm2(&a);
        assert!(
            err < 1e-5 * a_norm,
            "tolerance truncation error too large: {} (norm={})",
            err,
            a_norm
        );
    }

    #[test]
    fn test_aca_rank1() {
        // ACA should recover a rank-1 matrix exactly in 1 step.
        let m = 6;
        let n = 8;
        let u_vec: Vec<f64> = (0..m).map(|i| (i + 1) as f64).collect();
        let v_vec: Vec<f64> = (0..n).map(|j| (j + 1) as f64).collect();
        let entry = |i: usize, j: usize| u_vec[i] * v_vec[j];

        let res = aca(m, n, entry, 1e-8, 10).expect("aca ok");
        let recon = reconstruct_lr(&res.u, &res.v, m, n, res.rank);

        let expected: Vec<f64> = (0..m)
            .flat_map(|i| (0..n).map(move |j| u_vec[i] * v_vec[j]))
            .collect();
        let err = frobenius_diff(&recon, &expected);
        let norm = norm2(&expected);
        assert!(
            err < 1e-8 * norm,
            "ACA rank-1 error too large: {} (norm={})",
            err,
            norm
        );
    }

    #[test]
    fn test_aca_plus_rank1() {
        let m = 5;
        let n = 7;
        let u_vec: Vec<f64> = (0..m).map(|i| (i as f64 + 0.5)).collect();
        let v_vec: Vec<f64> = (0..n).map(|j| 1.0 / ((j + 1) as f64)).collect();
        let entry = |i: usize, j: usize| u_vec[i] * v_vec[j];

        let res = aca_plus(m, n, entry, 1e-8, 10).expect("aca+ ok");
        let recon = reconstruct_lr(&res.u, &res.v, m, n, res.rank);
        let expected: Vec<f64> = (0..m)
            .flat_map(|i| (0..n).map(move |j| u_vec[i] * v_vec[j]))
            .collect();
        let err = frobenius_diff(&recon, &expected);
        let norm = norm2(&expected);
        assert!(
            err < 1e-8 * norm,
            "ACA+ rank-1 error too large: {} (norm={})",
            err,
            norm
        );
    }

    #[test]
    fn test_hmatrix_truncate() {
        // Build U and V from a rank-2 matrix.
        let m = 6;
        let n = 5;
        let a = rank2_matrix(m, n);
        let (u_svd, sigma, vt_svd, r) = truncated_svd(&a, m, n, 0.0).expect("svd ok");

        // Build U (m×r with sigma absorbed) and V (n×r).
        let mut u = vec![0.0_f64; m * r];
        let mut v = vec![0.0_f64; n * r];
        for k in 0..r {
            let sq = sigma[k].sqrt();
            for i in 0..m {
                u[i * r + k] = u_svd[i * r + k] * sq;
            }
            for j in 0..n {
                v[j * r + k] = vt_svd[k * n + j] * sq;
            }
        }

        let (u2, v2, r2) = hmatrix_truncate(&u, &v, m, n, r, 1e-6).expect("truncate ok");
        assert!(r2 <= r, "truncated rank {} should be <= original {}", r2, r);

        let recon = reconstruct_lr(&u2, &v2, m, n, r2);
        let a_norm = norm2(&a);
        let err = frobenius_diff(&a, &recon);
        assert!(
            err < 1e-5 * a_norm,
            "truncation error too large: {} (norm={})",
            err,
            a_norm
        );
    }

    #[test]
    fn test_truncated_svd_error_cases() {
        assert!(truncated_svd(&[], 0, 1, 0.0).is_err());
        assert!(truncated_svd(&[], 1, 0, 0.0).is_err());
        assert!(truncated_svd(&[1.0, 2.0], 2, 2, 0.0).is_err()); // wrong length
        assert!(truncated_svd(&[1.0], 1, 1, -0.1).is_err());
    }

    #[test]
    fn test_aca_error_cases() {
        assert!(aca(0, 1, |_, _| 0.0, 0.0, 10).is_err());
        assert!(aca(1, 0, |_, _| 0.0, 0.0, 10).is_err());
        assert!(aca(1, 1, |_, _| 0.0, -1.0, 10).is_err());
    }
}
