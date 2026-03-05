//! Low-level linear algebra operations for Hierarchical Tucker decomposition.
//!
//! All routines are pure-Rust and allocation-aware.  They avoid `unwrap()`
//! and return `FFTResult` for recoverable errors.

use crate::error::{FFTError, FFTResult};

// ============================================================================
// Dense matrix routines
// ============================================================================

/// Compute  C = A · B  for column-major matrices.
///
/// A: m×k,  B: k×n,  C: m×n  (all stored in row-major order here).
pub fn matmul(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> FFTResult<Vec<f64>> {
    if a.len() != m * k {
        return Err(FFTError::ValueError(
            format!("matmul: a has {} elements, expected {}", a.len(), m * k),
        ));
    }
    if b.len() != k * n {
        return Err(FFTError::ValueError(
            format!("matmul: b has {} elements, expected {}", b.len(), k * n),
        ));
    }
    let mut c = vec![0.0_f64; m * n];
    for i in 0..m {
        for l in 0..k {
            let a_il = a[i * k + l];
            for j in 0..n {
                c[i * n + j] += a_il * b[l * n + j];
            }
        }
    }
    Ok(c)
}

/// Compute the Frobenius norm of a slice.
pub fn frobenius_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Truncated SVD via one-sided Jacobi method.
///
/// Decomposes the m×n matrix A (row-major) into U (m×r), S (r), V^T (r×n)
/// where r ≤ `max_rank`.
///
/// Uses the classical one-sided Jacobi SVD for correctness.
/// For large matrices a Golub-Reinsch Householder bidiagonalization
/// would be more efficient, but this suffices for HT decomposition.
pub fn truncated_svd(
    a: &[f64],
    m: usize,
    n: usize,
    max_rank: usize,
) -> FFTResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if a.len() != m * n {
        return Err(FFTError::ValueError(
            format!("truncated_svd: a has {} elements, expected {}", a.len(), m * n),
        ));
    }

    let r_full = m.min(n);
    let r = max_rank.min(r_full);

    // Use Householder bidiagonalization + QR iteration.
    let (u_full, s_full, vt_full) = bidiag_svd(a, m, n)?;

    // Truncate to top-r singular values.
    let mut u = vec![0.0_f64; m * r];
    for i in 0..m {
        for k in 0..r {
            u[i * r + k] = u_full[i * r_full + k];
        }
    }
    let s: Vec<f64> = s_full[..r].to_vec();
    let mut vt = vec![0.0_f64; r * n];
    for k in 0..r {
        for j in 0..n {
            vt[k * n + j] = vt_full[k * n + j];
        }
    }

    Ok((u, s, vt))
}

/// Full SVD via Golub-Reinsch bidiagonalization + implicit QR.
///
/// Returns (U, S, V^T) where U is m×m, S is r (diag), V^T is n×n.
pub fn bidiag_svd(a: &[f64], m: usize, n: usize) -> FFTResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if m == 0 || n == 0 {
        return Err(FFTError::ValueError("bidiag_svd: zero-dimension matrix".into()));
    }

    let r_full = m.min(n);

    // --- Step 1: Householder bidiagonalization ---
    // We decompose A = U_b * B * V_b^T where B is bidiagonal.
    let (mut ub, mut b_diag, mut b_super, mut vbt) = householder_bidiag(a, m, n)?;

    // --- Step 2: Implicit QR on B^T·B until convergence ---
    let max_iter = 1000 * r_full;
    for _ in 0..max_iter {
        // Convergence: check if all super-diagonal elements are negligible.
        let mut converged = true;
        for k in 0..b_super.len() {
            if b_super[k].abs() > 1e-14 * (b_diag[k].abs() + b_diag[k + 1].abs()) {
                converged = false;
                break;
            }
        }
        if converged { break; }

        // Golub-Kahan step.
        golub_kahan_step(&mut b_diag, &mut b_super, &mut ub, &mut vbt, m, n)?;
    }

    // Force non-negative singular values.
    for k in 0..r_full {
        if b_diag[k] < 0.0 {
            b_diag[k] = -b_diag[k];
            // Flip sign of corresponding column in U.
            for i in 0..m {
                ub[i * r_full + k] = -ub[i * r_full + k];
            }
        }
    }

    // Sort singular values in descending order.
    let mut order: Vec<usize> = (0..r_full).collect();
    order.sort_by(|&a, &b| b_diag[b].partial_cmp(&b_diag[a]).unwrap_or(std::cmp::Ordering::Equal));

    let s: Vec<f64> = order.iter().map(|&k| b_diag[k]).collect();
    let mut u: Vec<f64> = Vec::with_capacity(m * r_full);
    for i in 0..m {
        for &k in order.iter() {
            u.push(ub[i * r_full + k]);
        }
    }
    let mut vt: Vec<f64> = Vec::with_capacity(r_full * n);
    for &k in order.iter() {
        for j in 0..n {
            vt.push(vbt[k * n + j]);
        }
    }

    Ok((u, s, vt))
}

/// Householder bidiagonalization of an m×n matrix A (m ≥ n for simplicity).
///
/// Returns (U, diag, super_diag, V^T) where U is m×r, V^T is r×n, r = min(m,n).
fn householder_bidiag(
    a: &[f64],
    m: usize,
    n: usize,
) -> FFTResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let r = m.min(n);
    let mut mat = a.to_vec(); // m×n working copy

    // Accumulate U (m×m) and V (n×n) as identity matrices.
    let mut u_mat = eye(m);
    let mut v_mat = eye(n);

    let mut diag = vec![0.0_f64; r];
    let mut super_d = vec![0.0_f64; r.saturating_sub(1)];

    for k in 0..r {
        // Left Householder: eliminate column k below diagonal.
        {
            let col_len = m - k;
            let mut v: Vec<f64> = (0..col_len).map(|i| mat[(k + i) * n + k]).collect();
            let sigma = householder_vec(&mut v);
            diag[k] = if mat[k * n + k] >= 0.0 { -sigma } else { sigma };

            if sigma.abs() > 1e-15 {
                // Apply H = I - 2 v v^T to columns k..n of mat (rows k..m).
                apply_householder_left(&mut mat, m, n, k, &v, sigma);
                // Accumulate into U.
                apply_householder_right_rect(&mut u_mat, m, m, k, &v, sigma);
            }
        }

        // Right Householder: eliminate row k to the right of super-diagonal.
        if k < n - 1 && k < r - 1 {
            let row_len = n - k - 1;
            let mut v: Vec<f64> = (0..row_len).map(|j| mat[k * n + (k + 1 + j)]).collect();
            let sigma = householder_vec(&mut v);
            super_d[k] = if mat[k * n + k + 1] >= 0.0 { -sigma } else { sigma };

            if sigma.abs() > 1e-15 {
                // Apply H to rows k..m of mat (cols k+1..n).
                apply_householder_right(&mut mat, m, n, k, &v, sigma);
                // Accumulate into V.
                apply_householder_right_rect(&mut v_mat, n, n, k + 1, &v, sigma);
            }
        }
    }

    // U is m×m; we only need the first r columns.
    let mut u_r = vec![0.0_f64; m * r];
    for i in 0..m {
        for j in 0..r {
            u_r[i * r + j] = u_mat[i * m + j];
        }
    }

    // V is n×n; we need V^T restricted to first r rows.
    let mut vt_r = vec![0.0_f64; r * n];
    for k in 0..r {
        for j in 0..n {
            vt_r[k * n + j] = v_mat[j * n + k];
        }
    }

    Ok((u_r, diag, super_d, vt_r))
}

/// Compute the Householder vector for a column `v` (modifies in place).
/// Returns sigma = ||v||.
fn householder_vec(v: &mut Vec<f64>) -> f64 {
    let sigma: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if sigma < 1e-15 { return 0.0; }
    let sign = if v[0] >= 0.0 { 1.0 } else { -1.0 };
    v[0] += sign * sigma;
    let norm2: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm2 > 1e-15 {
        for x in v.iter_mut() { *x /= norm2; }
    }
    sigma
}

/// Apply left Householder reflection to submatrix mat[k..m, k..n].
fn apply_householder_left(mat: &mut Vec<f64>, m: usize, n: usize, k: usize, v: &[f64], _sigma: f64) {
    // H * A_sub: for each column j, A_sub[:,j] -= 2*(v^T * A_sub[:,j]) * v
    for j in k..n {
        let dot: f64 = v.iter().enumerate().map(|(i, vi)| vi * mat[(k + i) * n + j]).sum();
        for (i, vi) in v.iter().enumerate() {
            mat[(k + i) * n + j] -= 2.0 * dot * vi;
        }
    }
}

/// Apply right Householder reflection to submatrix mat[k..m, k+1..n].
fn apply_householder_right(mat: &mut Vec<f64>, m: usize, n: usize, k: usize, v: &[f64], _sigma: f64) {
    // A_sub * H: for each row i, A_sub[i,:] -= 2*(A_sub[i,:] * v) * v^T
    for i in k..m {
        let dot: f64 = v
            .iter()
            .enumerate()
            .map(|(j, vj)| vj * mat[i * n + (k + 1 + j)])
            .sum();
        for (j, vj) in v.iter().enumerate() {
            mat[i * n + (k + 1 + j)] -= 2.0 * dot * vj;
        }
    }
}

/// Apply right Householder reflection to an accumulation matrix (square).
/// Used to accumulate U and V.
fn apply_householder_right_rect(
    mat: &mut Vec<f64>,
    rows: usize,
    cols: usize,
    start_col: usize,
    v: &[f64],
    _sigma: f64,
) {
    for i in 0..rows {
        let dot: f64 = v
            .iter()
            .enumerate()
            .map(|(j, vj)| vj * mat[i * cols + (start_col + j)])
            .sum();
        for (j, vj) in v.iter().enumerate() {
            mat[i * cols + (start_col + j)] -= 2.0 * dot * vj;
        }
    }
}

/// One step of the Golub-Kahan implicit QR algorithm on the bidiagonal matrix.
fn golub_kahan_step(
    diag: &mut Vec<f64>,
    super_d: &mut Vec<f64>,
    u: &mut Vec<f64>,
    vt: &mut Vec<f64>,
    m: usize,
    n: usize,
) -> FFTResult<()> {
    let r = diag.len();
    if r == 0 { return Ok(()); }

    // Wilkinson shift from the bottom 2×2 block of B^T · B.
    let t = if r >= 2 {
        let d = diag[r - 1];
        let e = super_d.last().copied().unwrap_or(0.0);
        let d2 = diag[r - 2];
        let e2 = if r >= 3 { super_d.get(r - 3).copied().unwrap_or(0.0) } else { 0.0 };
        let tr = d2 * d2 + e2 * e2 + d * d + e * e;
        let det = (d2 * d2 + e2 * e2) * (d * d + e * e) - e2 * e2 * d * d;
        let mu1 = tr * 0.5 + (tr * tr * 0.25 - det).max(0.0).sqrt();
        let mu2 = tr * 0.5 - (tr * tr * 0.25 - det).max(0.0).sqrt();
        let shift1 = (mu1 - d * d - e * e).abs();
        let shift2 = (mu2 - d * d - e * e).abs();
        if shift1 < shift2 { mu1 } else { mu2 }
    } else {
        0.0
    };

    // Chase the bulge.
    let mut f = diag[0] * diag[0] - t;
    let mut g = diag[0] * super_d.first().copied().unwrap_or(0.0);

    for k in 0..r.saturating_sub(1) {
        // Right Givens rotation to zero g from row k.
        let (c, s) = givens(f, g);

        // Apply to bidiagonal B: columns k, k+1.
        if k > 0 {
            let sd = super_d[k - 1];
            super_d[k - 1] = c * sd + s * diag[k];
            diag[k] = -s * sd + c * diag[k];
        } else {
            diag[k] = c * diag[k] + s * (super_d.get(k).copied().unwrap_or(0.0));
        }

        // Bulge: f_new = diag[k] * ... (simplified update for bidiagonal)
        // Exact update of the 2×2 block.
        if k < super_d.len() {
            let d_k = diag[k];
            let e_k = super_d[k];
            let d_k1 = diag[k + 1];

            // Apply right Givens (c, s) to columns k, k+1.
            let new_dk = c * d_k + s * e_k;
            let new_ek = -s * d_k + c * e_k;  // this becomes the new off-diagonal before left Given
            let new_dk1 = d_k1; // only the super-diag changes at this step

            f = new_dk;
            g = s * new_dk1;
            super_d[k] = s * new_ek + c * (new_dk1 * 0.0); // approximate

            // Update V^T.
            let r_full = n.min(m);
            if k + 1 < r_full {
                for i in 0..n {
                    let vk = vt[k * n + i];
                    let vk1 = vt[(k + 1) * n + i];
                    vt[k * n + i] = c * vk + s * vk1;
                    vt[(k + 1) * n + i] = -s * vk + c * vk1;
                }
            }

            // Left Givens rotation to re-bidiagonalize.
            let (c2, s2) = givens(f, g);
            let _ = (c2, s2); // will be used below if not simplified

            // Left rotation on B: rows k, k+1.
            let new_ek2 = c2 * new_ek + s2 * d_k1;
            let new_dk1_2 = -s2 * new_ek + c2 * d_k1;
            super_d[k] = new_ek2;
            diag[k + 1] = new_dk1_2;
            diag[k] = new_dk;

            if k + 1 < r - 1 {
                let e_next = super_d[k + 1];
                f = new_dk1_2;
                g = s2 * e_next;
                super_d[k + 1] = c2 * e_next;
            }

            // Update U.
            for i in 0..m {
                let uk = u[i * r + k];
                let uk1 = u[i * r + k + 1];
                u[i * r + k] = c2 * uk + s2 * uk1;
                u[i * r + k + 1] = -s2 * uk + c2 * uk1;
            }
        }
    }

    Ok(())
}

/// Compute Givens rotation (c, s) such that [c s; -s c] [f; g] = [r; 0].
fn givens(f: f64, g: f64) -> (f64, f64) {
    if g.abs() < 1e-15 {
        return (1.0, 0.0);
    }
    if f.abs() < 1e-15 {
        return (0.0, 1.0);
    }
    let r = (f * f + g * g).sqrt();
    (f / r, g / r)
}

/// Identity matrix of size n (row-major).
fn eye(n: usize) -> Vec<f64> {
    let mut m = vec![0.0_f64; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// Compute the n-mode unfolding (matricization) of a tensor.
///
/// The n-mode unfolding of a tensor T of shape `shape` arranges the
/// mode-n fibers as columns of a matrix of size  (shape[n]) × (prod of rest).
///
/// Row index: i_n
/// Column index: (i_{n+1}, i_{n+2}, …, i_{d-1}, i_0, …, i_{n-1})  (row-major)
pub fn n_mode_unfolding(tensor: &[f64], shape: &[usize], mode: usize) -> FFTResult<(Vec<f64>, usize, usize)> {
    let d = shape.len();
    if mode >= d {
        return Err(FFTError::ValueError(
            format!("n_mode_unfolding: mode {mode} ≥ d={d}"),
        ));
    }

    let n_total: usize = shape.iter().product();
    if tensor.len() != n_total {
        return Err(FFTError::ValueError(
            format!("n_mode_unfolding: tensor length {} ≠ {}", tensor.len(), n_total),
        ));
    }

    let n_rows = shape[mode];
    let n_cols = n_total / n_rows;
    let mut mat = vec![0.0_f64; n_rows * n_cols];

    // Compute strides.
    let mut strides = vec![1usize; d];
    for k in (0..d - 1).rev() {
        strides[k] = strides[k + 1] * shape[k + 1];
    }

    // Fill the unfolding.
    for flat_idx in 0..n_total {
        // Decode multi-index.
        let mut multi_idx = vec![0usize; d];
        let mut rem = flat_idx;
        for k in 0..d {
            multi_idx[k] = rem / strides[k];
            rem %= strides[k];
        }

        let row = multi_idx[mode];

        // Column index: remaining indices in order (mode+1, …, d-1, 0, …, mode-1).
        let mut col = 0usize;
        let mut col_stride = 1usize;
        // Process indices in reverse order of column priority.
        let col_order: Vec<usize> = (mode + 1..d).chain(0..mode).collect();
        let col_strides: Vec<usize> = {
            let mut cs = vec![1usize; d - 1];
            let mut acc = 1usize;
            for (i, &mo) in col_order.iter().rev().enumerate() {
                let idx = col_order.len() - 1 - i;
                cs[idx] = acc;
                acc *= shape[mo];
            }
            cs
        };
        let _ = col_stride; // suppress warning
        for (i, &mo) in col_order.iter().enumerate() {
            col += multi_idx[mo] * col_strides[i];
        }

        mat[row * n_cols + col] = tensor[flat_idx];
    }

    Ok((mat, n_rows, n_cols))
}

/// Fold a matrix back into a tensor (inverse of n_mode_unfolding).
pub fn n_mode_folding(mat: &[f64], shape: &[usize], mode: usize) -> FFTResult<Vec<f64>> {
    let d = shape.len();
    if mode >= d {
        return Err(FFTError::ValueError(
            format!("n_mode_folding: mode {mode} ≥ d={d}"),
        ));
    }

    let n_total: usize = shape.iter().product();
    let n_rows = shape[mode];
    let n_cols = n_total / n_rows;

    if mat.len() != n_rows * n_cols {
        return Err(FFTError::ValueError(
            format!("n_mode_folding: mat length {} ≠ {}", mat.len(), n_rows * n_cols),
        ));
    }

    let mut tensor = vec![0.0_f64; n_total];

    let mut strides = vec![1usize; d];
    for k in (0..d - 1).rev() {
        strides[k] = strides[k + 1] * shape[k + 1];
    }

    for flat_idx in 0..n_total {
        let mut multi_idx = vec![0usize; d];
        let mut rem = flat_idx;
        for k in 0..d {
            multi_idx[k] = rem / strides[k];
            rem %= strides[k];
        }

        let row = multi_idx[mode];
        let col_order: Vec<usize> = (mode + 1..d).chain(0..mode).collect();
        let col_strides: Vec<usize> = {
            let mut cs = vec![1usize; d - 1];
            let mut acc = 1usize;
            for (i, &mo) in col_order.iter().rev().enumerate() {
                let idx = col_order.len() - 1 - i;
                cs[idx] = acc;
                acc *= shape[mo];
            }
            cs
        };
        let mut col = 0usize;
        for (i, &mo) in col_order.iter().enumerate() {
            col += multi_idx[mo] * col_strides[i];
        }

        tensor[flat_idx] = mat[row * n_cols + col];
    }

    Ok(tensor)
}

/// Compute the n-mode product T ×_n U where U is r×n_mode.
///
/// Result shape: shape with shape[mode] replaced by r.
pub fn n_mode_product(
    tensor: &[f64],
    shape: &[usize],
    mode: usize,
    u: &[f64],
    r: usize,
) -> FFTResult<Vec<f64>> {
    // Unfold T along mode.
    let (mat, n_rows, n_cols) = n_mode_unfolding(tensor, shape, mode)?;
    if u.len() != r * n_rows {
        return Err(FFTError::ValueError(
            format!("n_mode_product: U has {} elements, expected {}×{}={}", u.len(), r, n_rows, r * n_rows),
        ));
    }

    // Product: result_mat = U · mat,  size r × n_cols.
    let result_mat = matmul(u, r, n_rows, &mat, n_cols)?;

    // Fold back.
    let mut new_shape = shape.to_vec();
    new_shape[mode] = r;
    n_mode_folding(&result_mat, &new_shape, mode)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_identity() {
        let a = vec![1.0, 0.0, 0.0, 1.0]; // 2×2 identity
        let b = vec![3.0, 4.0, 5.0, 6.0]; // 2×2
        let c = matmul(&a, 2, 2, &b, 2).expect("failed to create c");
        assert!((c[0] - 3.0).abs() < 1e-12);
        assert!((c[3] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_n_mode_unfolding_roundtrip() {
        let shape = vec![2, 3, 4];
        let n = shape.iter().product::<usize>();
        let tensor: Vec<f64> = (0..n).map(|i| i as f64).collect();

        for mode in 0..3 {
            let (mat, rows, cols) = n_mode_unfolding(&tensor, &shape, mode).expect("unexpected None or Err");
            assert_eq!(rows, shape[mode]);
            assert_eq!(cols, n / shape[mode]);
            assert_eq!(mat.len(), n);

            // Fold back and check roundtrip.
            let recovered = n_mode_folding(&mat, &shape, mode).expect("failed to create recovered");
            for (a, b) in tensor.iter().zip(recovered.iter()) {
                assert!((a - b).abs() < 1e-12, "roundtrip failed at a={a} b={b}");
            }
        }
    }

    #[test]
    fn test_frobenius_norm() {
        let v = vec![3.0_f64, 4.0];
        assert!((frobenius_norm(&v) - 5.0).abs() < 1e-12);
    }
}
