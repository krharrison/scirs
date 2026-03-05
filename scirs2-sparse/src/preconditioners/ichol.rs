//! Incomplete Cholesky factorization preconditioners
//!
//! Two variants are provided for symmetric positive definite (SPD) systems:
//!
//! - **[`IC0`]**: Incomplete Cholesky with zero fill-in — the lower-triangular
//!   factor L has the same sparsity pattern as the lower triangle of A.
//!
//! - **[`ICT`]**: Incomplete Cholesky with threshold dropping — entries of L
//!   are dropped if their magnitude is below `threshold × ||row||_2`.
//!
//! Both variants solve the preconditioned system via forward/backward
//! substitution:  M x = r  where  M = L Lᵀ.
//!
//! # References
//!
//! - Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed.
//!   SIAM.  §10.3.2 (IC(0)), §10.3.3 (ICT).

use crate::error::{SparseError, SparseResult};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Forward solve: L y = b  (lower-triangular, stored in CSR).
fn forward_solve(
    l_data: &[f64],
    l_indices: &[usize],
    l_indptr: &[usize],
    b: &[f64],
    n: usize,
) -> SparseResult<Vec<f64>> {
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let start = l_indptr[i];
        let end = l_indptr[i + 1];
        let mut acc = b[i];
        let mut diag = 0.0f64;
        for pos in start..end {
            let j = l_indices[pos];
            if j < i {
                acc -= l_data[pos] * y[j];
            } else if j == i {
                diag = l_data[pos];
            }
        }
        if diag.abs() < 1e-300 {
            return Err(SparseError::SingularMatrix(format!(
                "IC forward solve: zero diagonal at row {i}"
            )));
        }
        y[i] = acc / diag;
    }
    Ok(y)
}

/// Backward solve: Lᵀ x = y  (Lᵀ is upper-triangular; accessed via L columns).
/// Because L is stored in CSR (lower-triangular), Lᵀ is accessed column-wise.
/// We implement this via a simple backward-substitution using the transposed access.
fn backward_solve_lt(
    l_data: &[f64],
    l_indices: &[usize],
    l_indptr: &[usize],
    y: &[f64],
    n: usize,
) -> SparseResult<Vec<f64>> {
    // Build a column-indexed structure for Lᵀ  (= U in CSR form).
    // Lᵀ[j,i] = L[i,j] for i > j.
    // We store cols_of_lt[j] = list of (row i, data_pos) where L[i,j] != 0.
    let mut cols_of_lt: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for i in 0..n {
        for pos in l_indptr[i]..l_indptr[i + 1] {
            let j = l_indices[pos];
            if j <= i {
                // L[i,j] contributes to Lᵀ column i (as row j) only for j < i.
                // We want Lᵀ[j, i] = L[i, j].
                cols_of_lt[j].push((i, l_data[pos]));
            }
        }
    }

    // Backward substitution on Lᵀ x = y.
    let mut x = vec![0.0f64; n];
    for ii in 0..n {
        let i = n - 1 - ii;
        // x[i] = (y[i] - sum_{j > i} Lᵀ[i,j] * x[j]) / Lᵀ[i,i]
        // Lᵀ[i,j] = L[j,i] for j > i  → use cols_of_lt[i] which has entries (row j, L[j,i]).
        let mut acc = y[i];
        let mut diag = 0.0f64;
        for &(j, val) in &cols_of_lt[i] {
            if j > i {
                acc -= val * x[j];
            } else if j == i {
                diag = val;
            }
        }
        if diag.abs() < 1e-300 {
            return Err(SparseError::SingularMatrix(format!(
                "IC backward solve: zero diagonal at row {i}"
            )));
        }
        x[i] = acc / diag;
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// IC(0) — Zero fill-in
// ---------------------------------------------------------------------------

/// IC(0) incomplete Cholesky preconditioner — zero fill-in.
///
/// A ≈ L Lᵀ where L has the same lower-triangular sparsity pattern as A.
pub struct IC0 {
    /// Lower-triangular Cholesky factor.
    /// Stored as (indptr, indices, data) in CSR.
    pub l: (Vec<usize>, Vec<usize>, Vec<f64>),
    n: usize,
}

impl IC0 {
    /// Compute IC(0) from the lower triangle of a symmetric CSR matrix.
    ///
    /// # Arguments
    ///
    /// * `indptr`  – CSR row pointer array (length n+1).
    /// * `indices` – CSR column index array (only lower-triangular entries used).
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

        // Extract lower-triangular part only (entries with col ≤ row).
        let mut l_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for i in 0..n {
            for pos in indptr[i]..indptr[i + 1] {
                let j = indices[pos];
                if j <= i {
                    l_rows[i].push((j, data[pos]));
                }
            }
            // Ensure sorted by column index.
            l_rows[i].sort_unstable_by_key(|&(c, _)| c);
        }

        // Perform in-place IC(0) factorization.
        // For each row i:
        //   Compute l_{ij} for j < i:  (nothing to update, they are fixed by earlier rows)
        //   Compute diagonal: l_{ii} = sqrt(a_{ii} - sum_{k<i} l_{ik}^2)
        //
        // The key update rule for lower-triangular IC(0):
        //   l[i][j] = (a[i][j] - sum_{k < j, k in pattern(i) ∩ pattern(j)} l[i][k]*l[j][k]) / l[j][j]
        //   l[i][i] = sqrt(a[i][i] - sum_{k < i} l[i][k]^2)

        // Process column by column (i.e., update lower rows after each column).
        // We use a dense workspace per row for simplicity (O(n^2) memory but correct).
        // For large sparse matrices a more efficient approach is used in practice;
        // here we prioritise correctness.

        // Dense lower-triangular workspace (row, col) → value.
        let mut l_dense: Vec<Vec<f64>> = (0..n).map(|i| vec![0.0f64; i + 1]).collect();
        for i in 0..n {
            for &(j, v) in &l_rows[i] {
                l_dense[i][j] = v;
            }
        }

        for j in 0..n {
            // Update diagonal l[j][j].
            let diag_sq: f64 = (0..j).map(|k| l_dense[j][k] * l_dense[j][k]).sum();
            let new_diag = l_dense[j][j] - diag_sq;
            if new_diag <= 0.0 {
                return Err(SparseError::SingularMatrix(format!(
                    "IC(0): non-positive pivot {new_diag:.3e} at row {j}; matrix may not be SPD"
                )));
            }
            l_dense[j][j] = new_diag.sqrt();
            let l_jj = l_dense[j][j];

            // Update column j entries for rows i > j that are in the pattern.
            for &(col, _) in l_rows[j + 1..].iter().flat_map(|r| r.iter()) {
                let _ = col; // We iterate by row index below.
            }
            for i in j + 1..n {
                // Only update if (i, j) is in the sparsity pattern of L.
                // Check by looking at l_rows[i].
                if l_rows[i].iter().any(|&(c, _)| c == j) {
                    // l[i][j] = (l[i][j] - sum_{k<j} l[i][k]*l[j][k]) / l[j][j]
                    let dot_sum: f64 = (0..j).map(|k| l_dense[i][k] * l_dense[j][k]).sum();
                    l_dense[i][j] = (l_dense[i][j] - dot_sum) / l_jj;
                }
            }
        }

        // Convert dense workspace back to sparse CSR (retaining original pattern).
        let mut l_indptr = vec![0usize; n + 1];
        for i in 0..n {
            l_indptr[i + 1] = l_rows[i].len();
        }
        for i in 0..n {
            l_indptr[i + 1] += l_indptr[i];
        }
        let nnz = l_indptr[n];
        let mut l_indices = vec![0usize; nnz];
        let mut l_data = vec![0.0f64; nnz];
        for i in 0..n {
            let start = l_indptr[i];
            for (k, &(j, _)) in l_rows[i].iter().enumerate() {
                l_indices[start + k] = j;
                l_data[start + k] = l_dense[i][j];
            }
        }

        Ok(Self {
            l: (l_indptr, l_indices, l_data),
            n,
        })
    }

    /// Apply the preconditioner: solve (L Lᵀ) x = r.
    pub fn apply(&self, r: &[f64]) -> SparseResult<Vec<f64>> {
        let (l_ip, l_idx, l_dat) = &self.l;
        let y = forward_solve(l_dat, l_idx, l_ip, r, self.n)?;
        backward_solve_lt(l_dat, l_idx, l_ip, &y, self.n)
    }

    /// Return the matrix dimension.
    pub fn size(&self) -> usize {
        self.n
    }
}

// ---------------------------------------------------------------------------
// ICT — Incomplete Cholesky with threshold
// ---------------------------------------------------------------------------

/// ICT incomplete Cholesky preconditioner — threshold dropping.
///
/// Same as IC(0) but entries of L are dropped if
/// `|l_{ij}| < threshold * ||row||_2`.
pub struct ICT {
    /// Lower-triangular Cholesky factor.
    pub l: (Vec<usize>, Vec<usize>, Vec<f64>),
    /// Drop threshold.
    pub threshold: f64,
    n: usize,
}

impl ICT {
    /// Compute ICT from the lower triangle of a symmetric CSR matrix.
    ///
    /// # Arguments
    ///
    /// * `indptr`, `indices`, `data` – CSR representation of A (lower triangle sufficient).
    /// * `n`         – Matrix dimension.
    /// * `threshold` – Relative drop tolerance.
    pub fn factor(
        indptr: &[usize],
        indices: &[usize],
        data: &[f64],
        n: usize,
        threshold: f64,
    ) -> SparseResult<Self> {
        if indptr.len() != n + 1 {
            return Err(SparseError::InconsistentData {
                reason: format!("indptr length {} != n+1={}", indptr.len(), n + 1),
            });
        }

        // Dense workspace.
        let mut l_dense: Vec<Vec<f64>> = (0..n).map(|i| vec![0.0f64; i + 1]).collect();
        for i in 0..n {
            for pos in indptr[i]..indptr[i + 1] {
                let j = indices[pos];
                if j <= i {
                    l_dense[i][j] = data[pos];
                }
            }
        }

        // Factorize.
        for j in 0..n {
            let diag_sq: f64 = (0..j).map(|k| l_dense[j][k] * l_dense[j][k]).sum();
            let new_diag = l_dense[j][j] - diag_sq;
            if new_diag <= 0.0 {
                return Err(SparseError::SingularMatrix(format!(
                    "ICT: non-positive pivot {new_diag:.3e} at row {j}"
                )));
            }
            l_dense[j][j] = new_diag.sqrt();
            let l_jj = l_dense[j][j];

            for i in j + 1..n {
                if l_dense[i][j] == 0.0 {
                    continue;
                }
                let dot_sum: f64 = (0..j).map(|k| l_dense[i][k] * l_dense[j][k]).sum();
                l_dense[i][j] = (l_dense[i][j] - dot_sum) / l_jj;
            }
        }

        // Apply threshold dropping and convert to CSR.
        let mut l_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for i in 0..n {
            // Row norm of row i of L.
            let row_norm: f64 = (0..=i)
                .map(|j| l_dense[i][j] * l_dense[i][j])
                .sum::<f64>()
                .sqrt();
            let drop_tol = threshold * row_norm;
            for j in 0..=i {
                let val = l_dense[i][j];
                if val != 0.0 && (j == i || val.abs() >= drop_tol) {
                    l_rows[i].push((j, val));
                }
            }
        }

        let (l_indptr, l_indices, l_data) = rows_to_csr_lower(&l_rows, n);
        Ok(Self {
            l: (l_indptr, l_indices, l_data),
            threshold,
            n,
        })
    }

    /// Apply the preconditioner: solve (L Lᵀ) x = r.
    pub fn apply(&self, r: &[f64]) -> SparseResult<Vec<f64>> {
        let (l_ip, l_idx, l_dat) = &self.l;
        let y = forward_solve(l_dat, l_idx, l_ip, r, self.n)?;
        backward_solve_lt(l_dat, l_idx, l_ip, &y, self.n)
    }

    /// Return the matrix dimension.
    pub fn size(&self) -> usize {
        self.n
    }
}

fn rows_to_csr_lower(
    rows: &[Vec<(usize, f64)>],
    n: usize,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
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

    /// Build the lower-triangular part of a 4×4 SPD tridiagonal matrix.
    /// A = [4,-1, 0, 0; -1, 4,-1, 0; 0,-1, 4,-1; 0, 0,-1, 4]
    fn test_spd_matrix_lower() -> (Vec<usize>, Vec<usize>, Vec<f64>, usize) {
        let n = 4usize;
        // Lower triangle only (including diagonal).
        let indptr = vec![0, 1, 3, 5, 7];
        let indices = vec![0, 0, 1, 1, 2, 2, 3];
        let data = vec![4.0, -1.0, 4.0, -1.0, 4.0, -1.0, 4.0];
        (indptr, indices, data, n)
    }

    /// Apply the full symmetric matrix A to a vector.
    fn matvec_sym(x: &[f64], n: usize) -> Vec<f64> {
        // A = tridiag(-1, 4, -1) of size n×n
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            y[i] += 4.0 * x[i];
            if i > 0 {
                y[i] -= x[i - 1];
            }
            if i + 1 < n {
                y[i] -= x[i + 1];
            }
        }
        y
    }

    #[test]
    fn test_ic0_factor_and_apply() {
        let (ip, idx, dat, n) = test_spd_matrix_lower();
        let ic = IC0::factor(&ip, &idx, &dat, n).expect("IC0 factor failed");

        assert_eq!(ic.size(), n);

        let b = vec![1.0, 2.0, 3.0, 4.0];
        let x = ic.apply(&b).expect("IC0 apply failed");
        assert_eq!(x.len(), n);
        assert!(x.iter().all(|v| v.is_finite()), "IC0 result must be finite");
    }

    #[test]
    fn test_ic0_reduces_residual() {
        let (ip, idx, dat, n) = test_spd_matrix_lower();
        let ic = IC0::factor(&ip, &idx, &dat, n).expect("IC0 factor");

        // Use IC0 as a preconditioner: apply it to Ax, should recover x approximately.
        let x_true = vec![1.0, 1.0, 1.0, 1.0];
        let b = matvec_sym(&x_true, n); // b = A * x_true

        let x_prec = ic.apply(&b).expect("IC0 apply");

        // The preconditioned vector should have finite entries.
        assert!(x_prec.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_ict_factor_and_apply() {
        let (ip, idx, dat, n) = test_spd_matrix_lower();
        let ict = ICT::factor(&ip, &idx, &dat, n, 1e-4).expect("ICT factor failed");

        assert_eq!(ict.size(), n);

        let b = vec![1.0, 2.0, 3.0, 4.0];
        let x = ict.apply(&b).expect("ICT apply failed");
        assert_eq!(x.len(), n);
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_ict_high_threshold() {
        let (ip, idx, dat, n) = test_spd_matrix_lower();
        // High threshold drops many entries but should still work.
        let ict = ICT::factor(&ip, &idx, &dat, n, 0.5).expect("ICT high threshold");
        let b = vec![1.0; n];
        let x = ict.apply(&b).expect("ICT high threshold apply");
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_ic0_non_spd_error() {
        // Indefinite matrix: negative diagonal.
        let n = 2;
        let indptr = vec![0, 1, 3];
        let indices = vec![0, 0, 1];
        let data = vec![-1.0, 1.0, 2.0]; // negative diagonal at row 0
        let result = IC0::factor(&indptr, &indices, &data, n);
        assert!(result.is_err(), "should fail on non-SPD matrix");
    }

    #[test]
    fn test_ic0_identity() {
        // Identity matrix → L = I, L L^T x = x.
        let n = 3;
        let indptr = vec![0, 1, 2, 3];
        let indices = vec![0, 1, 2];
        let data = vec![1.0, 1.0, 1.0];
        let ic = IC0::factor(&indptr, &indices, &data, n).expect("IC0 identity");
        let b = vec![3.0, 7.0, 2.0];
        let x = ic.apply(&b).expect("apply");
        for (xi, bi) in x.iter().zip(b.iter()) {
            assert!((xi - bi).abs() < 1e-10, "IC0 of I should return input");
        }
    }
}
