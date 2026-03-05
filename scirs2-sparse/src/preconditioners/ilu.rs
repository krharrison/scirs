//! Incomplete LU factorization preconditioners
//!
//! Two variants are provided:
//!
//! - **[`ILU0`]**: ILU with zero fill-in — the factorization retains exactly
//!   the sparsity pattern of A, making it the cheapest ILU variant.
//!
//! - **[`ILUT`]**: ILU with threshold dropping — entries of L and U are
//!   dropped if their magnitude is below `threshold × ||row||_2`, subject
//!   to a maximum fill-per-row budget controlled by `fill_factor`.
//!
//! Both variants implement `apply(&self, r: &[f64]) -> Vec<f64>` which
//! performs the forward/backward substitution L U x = r.
//!
//! # References
//!
//! - Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed.
//!   SIAM.  §10.3 (ILU(0)), §10.4 (ILUT).

use crate::error::{SparseError, SparseResult};

// ---------------------------------------------------------------------------
// Internal CSR helpers
// ---------------------------------------------------------------------------

/// Find column `col` in the sorted slice `indices[start..end]`.
/// Returns `Some(pos)` where `indices[pos] == col`, or `None`.
#[inline]
fn find_col(indices: &[usize], start: usize, end: usize, col: usize) -> Option<usize> {
    for pos in start..end {
        if indices[pos] == col {
            return Some(pos);
        }
        if indices[pos] > col {
            return None;
        }
    }
    None
}

/// Forward solve: (unit lower-triangular L) y = b.
/// L is stored in CSR with explicit 1s on the diagonal (they can be absent —
/// the solve assumes a unit diagonal regardless).
fn forward_solve_unit(
    l_data: &[f64],
    l_indices: &[usize],
    l_indptr: &[usize],
    b: &[f64],
    n: usize,
) -> Vec<f64> {
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let start = l_indptr[i];
        let end = l_indptr[i + 1];
        let mut acc = b[i];
        for pos in start..end {
            let j = l_indices[pos];
            if j < i {
                acc -= l_data[pos] * y[j];
            }
        }
        y[i] = acc;
    }
    y
}

/// Backward solve: (upper-triangular U) x = y.
/// The diagonal entry is the *last* in each row (rows are stored with
/// columns in ascending order and the diagonal is the final non-zero).
fn backward_solve(
    u_data: &[f64],
    u_indices: &[usize],
    u_indptr: &[usize],
    y: &[f64],
    n: usize,
) -> SparseResult<Vec<f64>> {
    let mut x = vec![0.0f64; n];
    for ii in 0..n {
        let i = n - 1 - ii;
        let start = u_indptr[i];
        let end = u_indptr[i + 1];

        let mut diag = 0.0f64;
        let mut sum = y[i];
        for pos in start..end {
            let j = u_indices[pos];
            if j == i {
                diag = u_data[pos];
            } else if j > i {
                sum -= u_data[pos] * x[j];
            }
        }
        if diag.abs() < 1e-300 {
            return Err(SparseError::SingularMatrix(format!(
                "zero diagonal at row {i} in backward solve"
            )));
        }
        x[i] = sum / diag;
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// ILU(0) — Zero fill-in
// ---------------------------------------------------------------------------

/// ILU(0) preconditioner — incomplete LU with zero fill-in.
///
/// The factorization A ≈ L U has the same sparsity pattern as A.
/// For each row i:
///   for j < i in row i:  a_{ij} /= a_{jj}
///                         for k in row i with k > j: a_{ik} -= a_{ij} * a_{jk}
///                         (modification only if k is already in row i's pattern)
pub struct ILU0 {
    /// Unit-lower-triangular factor (diagonal = 1, stored explicitly as 0 / absent).
    pub l: (Vec<usize>, Vec<usize>, Vec<f64>), // (indptr, indices, data)
    /// Upper-triangular factor (diagonal included).
    pub u: (Vec<usize>, Vec<usize>, Vec<f64>),
    n: usize,
}

impl ILU0 {
    /// Compute ILU(0) from a square CSR matrix.
    ///
    /// # Arguments
    ///
    /// * `indptr`  – CSR row pointer array (length n+1).
    /// * `indices` – CSR column index array.
    /// * `data`    – CSR value array.
    /// * `n`       – Matrix dimension.
    pub fn factor(
        indptr: &[usize],
        indices: &[usize],
        data: &[f64],
        n: usize,
    ) -> SparseResult<Self> {
        if indptr.len() != n + 1 {
            return Err(SparseError::InconsistentData {
                reason: format!("indptr length {} != n+1={}", indptr.len(), n + 1),
            });
        }

        // Work on a mutable copy of the data.
        let mut a = data.to_vec();

        // Perform ILU(0) in-place on the working copy `a`.
        // For each row i we modify entries a[pos] for the columns in row i.
        for i in 1..n {
            let row_start = indptr[i];
            let row_end = indptr[i + 1];

            // For each j < i in row i:
            for pos_j in row_start..row_end {
                let j = indices[pos_j];
                if j >= i {
                    break; // Columns are sorted; we've passed the sub-diagonal.
                }

                // Pivot: find a_{jj}.
                let pivot_pos =
                    find_col(indices, indptr[j], indptr[j + 1], j).ok_or_else(|| {
                        SparseError::SingularMatrix(format!(
                            "ILU(0): missing diagonal at row {j}"
                        ))
                    })?;
                let a_jj = a[pivot_pos];
                if a_jj.abs() < 1e-300 {
                    return Err(SparseError::SingularMatrix(format!(
                        "ILU(0): zero diagonal at row {j}"
                    )));
                }

                // Multiplier: a_{ij} /= a_{jj}
                a[pos_j] /= a_jj;
                let mult = a[pos_j];

                // Update row i: a_{ik} -= mult * a_{jk}  for k > j in row i ∩ row j.
                for pos_k in pos_j + 1..row_end {
                    let k = indices[pos_k];
                    if let Some(jk_pos) = find_col(indices, indptr[j], indptr[j + 1], k) {
                        a[pos_k] -= mult * a[jk_pos];
                    }
                }
            }
        }

        // Split into L and U from the in-place factorization.
        let mut l_indptr = vec![0usize; n + 1];
        let mut u_indptr = vec![0usize; n + 1];

        for i in 0..n {
            let row_start = indptr[i];
            let row_end = indptr[i + 1];
            for pos in row_start..row_end {
                let j = indices[pos];
                if j < i {
                    l_indptr[i + 1] += 1;
                } else {
                    u_indptr[i + 1] += 1;
                }
            }
        }
        for i in 0..n {
            l_indptr[i + 1] += l_indptr[i];
            u_indptr[i + 1] += u_indptr[i];
        }

        let l_nnz = l_indptr[n];
        let u_nnz = u_indptr[n];
        let mut l_indices = vec![0usize; l_nnz];
        let mut l_data = vec![0.0f64; l_nnz];
        let mut u_indices = vec![0usize; u_nnz];
        let mut u_data = vec![0.0f64; u_nnz];

        let mut l_cur = l_indptr[..n].to_vec();
        let mut u_cur = u_indptr[..n].to_vec();

        for i in 0..n {
            let row_start = indptr[i];
            let row_end = indptr[i + 1];
            for pos in row_start..row_end {
                let j = indices[pos];
                if j < i {
                    let dst = l_cur[i];
                    l_indices[dst] = j;
                    l_data[dst] = a[pos];
                    l_cur[i] += 1;
                } else {
                    let dst = u_cur[i];
                    u_indices[dst] = j;
                    u_data[dst] = a[pos];
                    u_cur[i] += 1;
                }
            }
        }

        Ok(Self {
            l: (l_indptr, l_indices, l_data),
            u: (u_indptr, u_indices, u_data),
            n,
        })
    }

    /// Apply the preconditioner: solve L U x = r via forward/backward substitution.
    pub fn apply(&self, r: &[f64]) -> SparseResult<Vec<f64>> {
        let (l_ip, l_idx, l_dat) = &self.l;
        let (u_ip, u_idx, u_dat) = &self.u;
        let y = forward_solve_unit(l_dat, l_idx, l_ip, r, self.n);
        backward_solve(u_dat, u_idx, u_ip, &y, self.n)
    }

    /// Return the matrix dimension.
    pub fn size(&self) -> usize {
        self.n
    }
}

// ---------------------------------------------------------------------------
// ILUT — Incomplete LU with threshold dropping
// ---------------------------------------------------------------------------

/// ILUT preconditioner — incomplete LU with dual threshold and fill-factor control.
///
/// For each row i the standard ILU elimination is performed but entries are
/// dropped if `|a_{ij}| < threshold * ||row||_2`.  Additionally, no more
/// than `fill_factor` entries are kept in each of the L and U rows.
pub struct ILUT {
    /// Lower-triangular factor.
    pub l: (Vec<usize>, Vec<usize>, Vec<f64>),
    /// Upper-triangular factor.
    pub u: (Vec<usize>, Vec<usize>, Vec<f64>),
    /// Drop threshold (relative to row norm).
    pub threshold: f64,
    /// Maximum fill per L/U row.
    pub fill_factor: usize,
    n: usize,
}

impl ILUT {
    /// Compute ILUT from a square CSR matrix.
    ///
    /// # Arguments
    ///
    /// * `indptr`, `indices`, `data` – CSR representation of A.
    /// * `n`          – Matrix dimension.
    /// * `threshold`  – Relative drop tolerance τ.
    /// * `fill_factor`– Maximum non-zeros per row in L and U separately.
    pub fn factor(
        indptr: &[usize],
        indices: &[usize],
        data: &[f64],
        n: usize,
        threshold: f64,
        fill_factor: usize,
    ) -> SparseResult<Self> {
        if indptr.len() != n + 1 {
            return Err(SparseError::InconsistentData {
                reason: format!("indptr length {} != n+1={}", indptr.len(), n + 1),
            });
        }

        let fill = fill_factor.max(1);

        let mut l_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let mut u_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];

        for i in 0..n {
            // Extract row i into a dense workspace (length n).
            let mut w = vec![0.0f64; n];
            let row_start = indptr[i];
            let row_end = indptr[i + 1];
            for pos in row_start..row_end {
                w[indices[pos]] = data[pos];
            }

            // Row norm (for drop tolerance).
            let row_norm: f64 = w.iter().map(|v| v * v).sum::<f64>().sqrt();
            let drop_tol = threshold * row_norm;

            // Elimination step: for each j < i where w[j] != 0:
            for j in 0..i {
                if w[j] == 0.0 {
                    continue;
                }
                // Get the diagonal of U[j,j].
                let u_jj = u_rows[j]
                    .iter()
                    .find(|&&(c, _)| c == j)
                    .map(|&(_, v)| v)
                    .unwrap_or(0.0);
                if u_jj.abs() < 1e-300 {
                    continue;
                }
                w[j] /= u_jj;
                let mult = w[j];

                // w[k] -= mult * U[j,k]  for k > j
                for &(k, u_jk) in u_rows[j].iter().filter(|&&(c, _)| c > j) {
                    w[k] -= mult * u_jk;
                }

                // Apply drop to w[j] (the L multiplier).
                if w[j].abs() < drop_tol {
                    w[j] = 0.0;
                }
            }

            // Apply drop to upper entries and split.
            let mut l_row: Vec<(usize, f64)> = Vec::new();
            let mut u_row: Vec<(usize, f64)> = Vec::new();

            for (col, &val) in w.iter().enumerate() {
                if val == 0.0 {
                    continue;
                }
                if col < i {
                    if val.abs() >= drop_tol {
                        l_row.push((col, val));
                    }
                } else if col == i {
                    u_row.push((col, val)); // Diagonal always kept.
                } else if val.abs() >= drop_tol {
                    u_row.push((col, val));
                }
            }

            // Apply fill-in limit: keep `fill` largest-magnitude off-diagonal entries.
            if l_row.len() > fill {
                l_row.sort_unstable_by(|a, b| {
                    b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal)
                });
                l_row.truncate(fill);
                l_row.sort_unstable_by_key(|&(c, _)| c);
            }

            // For U, separate diagonal from off-diagonal when applying limit.
            let diag_entry = u_row.iter().find(|&&(c, _)| c == i).copied();
            let mut u_off: Vec<(usize, f64)> = u_row.into_iter().filter(|&(c, _)| c != i).collect();
            if u_off.len() > fill {
                u_off.sort_unstable_by(|a, b| {
                    b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal)
                });
                u_off.truncate(fill);
                u_off.sort_unstable_by_key(|&(c, _)| c);
            }
            let mut u_full: Vec<(usize, f64)> = Vec::new();
            if let Some(d) = diag_entry {
                u_full.push(d);
            }
            u_full.extend(u_off);
            u_full.sort_unstable_by_key(|&(c, _)| c);

            l_rows[i] = l_row;
            u_rows[i] = u_full;
        }

        // Convert to CSR.
        let (l_indptr, l_indices, l_data) = rows_to_csr(&l_rows, n);
        let (u_indptr, u_indices, u_data) = rows_to_csr(&u_rows, n);

        Ok(Self {
            l: (l_indptr, l_indices, l_data),
            u: (u_indptr, u_indices, u_data),
            threshold,
            fill_factor: fill,
            n,
        })
    }

    /// Apply the preconditioner: solve L U x = r.
    pub fn apply(&self, r: &[f64]) -> SparseResult<Vec<f64>> {
        let (l_ip, l_idx, l_dat) = &self.l;
        let (u_ip, u_idx, u_dat) = &self.u;
        let y = forward_solve_unit(l_dat, l_idx, l_ip, r, self.n);
        backward_solve(u_dat, u_idx, u_ip, &y, self.n)
    }

    /// Return the matrix dimension.
    pub fn size(&self) -> usize {
        self.n
    }
}

// ---------------------------------------------------------------------------
// Helper: convert a Vec-of-rows to CSR
// ---------------------------------------------------------------------------

fn rows_to_csr(rows: &[Vec<(usize, f64)>], n: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut indptr = vec![0usize; n + 1];
    for (i, row) in rows.iter().enumerate() {
        indptr[i + 1] = indptr[i] + row.len();
    }
    let nnz = indptr[n];
    let mut col_indices = vec![0usize; nnz];
    let mut values = vec![0.0f64; nnz];
    for (i, row) in rows.iter().enumerate() {
        let start = indptr[i];
        for (k, &(col, val)) in row.iter().enumerate() {
            col_indices[start + k] = col;
            values[start + k] = val;
        }
    }
    (indptr, col_indices, values)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 4×4 diagonally dominant tridiagonal system.
    fn test_matrix() -> (Vec<usize>, Vec<usize>, Vec<f64>, usize) {
        let n = 4usize;
        // A = [4,-1, 0, 0]
        //     [-1, 4,-1, 0]
        //     [ 0,-1, 4,-1]
        //     [ 0, 0,-1, 4]
        let indptr = vec![0, 2, 5, 8, 10];
        let indices = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let data = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        (indptr, indices, data, n)
    }

    fn matvec(indptr: &[usize], indices: &[usize], data: &[f64], x: &[f64], n: usize) -> Vec<f64> {
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            for pos in indptr[i]..indptr[i + 1] {
                y[i] += data[pos] * x[indices[pos]];
            }
        }
        y
    }

    #[test]
    fn test_ilu0_factor_and_apply() {
        let (ip, idx, dat, n) = test_matrix();
        let ilu = ILU0::factor(&ip, &idx, &dat, n).expect("ILU0 factor failed");

        assert_eq!(ilu.size(), n);

        let b = vec![1.0, 2.0, 3.0, 4.0];
        let x = ilu.apply(&b).expect("ILU0 apply failed");
        assert_eq!(x.len(), n);

        // Verify LUx ≈ b (ILU(0) is exact on tridiagonal since there's no fill-in).
        let lux = matvec(&ip, &idx, &dat, &x, n);
        for (bi, li) in b.iter().zip(lux.iter()) {
            assert!((bi - li).abs() < 1e-8, "LUx ≈ b failed: {bi} vs {li}");
        }
    }

    #[test]
    fn test_ilu0_reduces_residual() {
        let (ip, idx, dat, n) = test_matrix();
        let ilu = ILU0::factor(&ip, &idx, &dat, n).expect("ILU0 factor");

        let b = vec![3.0, 2.0, 2.0, 3.0];
        let prec_b = ilu.apply(&b).expect("apply");

        // Apply A to get residual A*(M^{-1} b) ≈ b.
        let ab = matvec(&ip, &idx, &dat, &prec_b, n);
        let res: f64 = b.iter().zip(ab.iter()).map(|(bi, ai)| (bi - ai).powi(2)).sum::<f64>().sqrt();
        assert!(
            res < 1e-6,
            "ILU(0) exact on tridiagonal, residual should be ~0: {res}"
        );
    }

    #[test]
    fn test_ilut_factor_and_apply() {
        let (ip, idx, dat, n) = test_matrix();
        let ilut = ILUT::factor(&ip, &idx, &dat, n, 1e-4, 4).expect("ILUT factor failed");

        assert_eq!(ilut.size(), n);

        let b = vec![1.0, 2.0, 3.0, 4.0];
        let x = ilut.apply(&b).expect("ILUT apply failed");
        assert_eq!(x.len(), n);
        assert!(x.iter().all(|v| v.is_finite()), "ILUT result must be finite");
    }

    #[test]
    fn test_ilut_high_threshold_finite() {
        let (ip, idx, dat, n) = test_matrix();
        // Very high threshold — most entries dropped, but result should still be finite.
        let ilut = ILUT::factor(&ip, &idx, &dat, n, 1.0, 2).expect("ILUT high threshold");
        let b = vec![1.0; n];
        let x = ilut.apply(&b).expect("apply");
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_ilu0_singular_diagonal_error() {
        let n = 2;
        let indptr = vec![0, 2, 4];
        let indices = vec![0, 1, 0, 1];
        let data = vec![0.0, 1.0, 1.0, 2.0]; // zero diagonal at row 0
        let result = ILU0::factor(&indptr, &indices, &data, n);
        assert!(result.is_err(), "should fail on zero diagonal");
    }
}
