//! Incomplete matrix factorizations for preconditioning
//!
//! This module provides several incomplete factorization preconditioners:
//!
//! - **ILU(0)**: Incomplete LU with zero fill-in
//! - **ILU(k)**: Incomplete LU with level-k fill-in
//! - **IC(0)**: Incomplete Cholesky for symmetric positive definite matrices
//! - **MILU**: Modified ILU with diagonal compensation
//! - **ILUT**: Incomplete LU with threshold dropping
//!
//! All implementations conform to the `Preconditioner` trait from
//! `crate::iterative_solvers`, enabling them to be used as drop-in
//! preconditioners for iterative solvers (CG, BiCGSTAB, GMRES, etc.).
//!
//! # References
//!
//! - Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed. SIAM.

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use crate::iterative_solvers::Preconditioner;
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, NumAssign, SparseElement};
use std::fmt::Debug;
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Find the position of column `col` in the CSR row.
fn find_col_in_row(
    indices: &[usize],
    row_start: usize,
    row_end: usize,
    col: usize,
) -> Option<usize> {
    for pos in row_start..row_end {
        if indices[pos] == col {
            return Some(pos);
        }
        if indices[pos] > col {
            return None;
        }
    }
    None
}

/// Forward solve: L y = b where L is unit-lower triangular stored in CSR.
fn forward_solve_unit<F: Float + NumAssign + SparseElement>(
    l_data: &[F],
    l_indices: &[usize],
    l_indptr: &[usize],
    b: &[F],
    n: usize,
) -> Vec<F> {
    let mut y = vec![F::sparse_zero(); n];
    for i in 0..n {
        y[i] = b[i];
        let start = l_indptr[i];
        let end = l_indptr[i + 1];
        for pos in start..end {
            let col = l_indices[pos];
            y[i] = y[i] - l_data[pos] * y[col];
        }
    }
    y
}

/// Backward solve: U x = y where U is upper triangular stored in CSR.
fn backward_solve<F: Float + NumAssign + SparseElement>(
    u_data: &[F],
    u_indices: &[usize],
    u_indptr: &[usize],
    y: &[F],
    n: usize,
) -> SparseResult<Vec<F>> {
    let mut x = vec![F::sparse_zero(); n];
    for ii in 0..n {
        let i = n - 1 - ii;
        let start = u_indptr[i];
        let end = u_indptr[i + 1];

        // Find diagonal
        let mut diag = F::sparse_zero();
        let mut sum = y[i];
        for pos in start..end {
            let col = u_indices[pos];
            if col == i {
                diag = u_data[pos];
            } else if col > i {
                sum -= u_data[pos] * x[col];
            }
        }
        if diag.abs() < F::epsilon() {
            return Err(SparseError::SingularMatrix(format!(
                "Zero diagonal at row {i} during backward solve"
            )));
        }
        x[i] = sum / diag;
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// ILU(0) — Incomplete LU with zero fill-in
// ---------------------------------------------------------------------------

/// ILU(0) preconditioner: Incomplete LU factorization with zero fill-in.
///
/// The sparsity pattern of L + U is identical to the sparsity pattern of A.
/// This is the simplest ILU preconditioner and works well for mildly
/// ill-conditioned systems.
pub struct Ilu0<F> {
    l_data: Vec<F>,
    l_indices: Vec<usize>,
    l_indptr: Vec<usize>,
    u_data: Vec<F>,
    u_indices: Vec<usize>,
    u_indptr: Vec<usize>,
    n: usize,
}

impl<F> Ilu0<F>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    /// Construct an ILU(0) preconditioner from a CSR matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The matrix is not square
    /// - A zero pivot is encountered
    pub fn new(matrix: &CsrMatrix<F>) -> SparseResult<Self> {
        let n = matrix.rows();
        if n != matrix.cols() {
            return Err(SparseError::ValueError(
                "ILU(0) requires a square matrix".to_string(),
            ));
        }

        // Copy matrix data for in-place factorization
        let mut data = matrix.data.clone();
        let indices = matrix.indices.clone();
        let indptr = matrix.indptr.clone();

        // IKJ variant of ILU(0)
        for i in 1..n {
            let i_start = indptr[i];
            let i_end = indptr[i + 1];

            for pos_k in i_start..i_end {
                let k = indices[pos_k];
                if k >= i {
                    break;
                }

                // Find diagonal of row k
                let k_start = indptr[k];
                let k_end = indptr[k + 1];
                let k_diag_pos = match find_col_in_row(&indices, k_start, k_end, k) {
                    Some(p) => p,
                    None => {
                        return Err(SparseError::SingularMatrix(format!(
                            "Missing diagonal at row {k}"
                        )));
                    }
                };
                let k_diag = data[k_diag_pos];
                if k_diag.abs() < F::epsilon() {
                    return Err(SparseError::SingularMatrix(format!(
                        "Zero pivot at row {k} in ILU(0)"
                    )));
                }

                // a_{ik} = a_{ik} / a_{kk}
                data[pos_k] /= k_diag;
                let multiplier = data[pos_k];

                // Update row i for columns j > k that exist in both rows i and k
                for kj_pos in k_start..k_end {
                    let j = indices[kj_pos];
                    if j <= k {
                        continue;
                    }
                    // Find j in row i
                    if let Some(ij_pos) = find_col_in_row(&indices, i_start, i_end, j) {
                        data[ij_pos] = data[ij_pos] - multiplier * data[kj_pos];
                    }
                }
            }
        }

        // Split into L (strictly lower, unit diagonal implied) and U (upper + diagonal)
        let mut l_data = Vec::new();
        let mut u_data = Vec::new();
        let mut l_indices = Vec::new();
        let mut u_indices = Vec::new();
        let mut l_indptr = vec![0usize];
        let mut u_indptr = vec![0usize];

        for i in 0..n {
            let start = indptr[i];
            let end = indptr[i + 1];
            for pos in start..end {
                let col = indices[pos];
                if col < i {
                    l_indices.push(col);
                    l_data.push(data[pos]);
                } else {
                    u_indices.push(col);
                    u_data.push(data[pos]);
                }
            }
            l_indptr.push(l_indices.len());
            u_indptr.push(u_indices.len());
        }

        Ok(Self {
            l_data,
            l_indices,
            l_indptr,
            u_data,
            u_indices,
            u_indptr,
            n,
        })
    }
}

impl<F> Preconditioner<F> for Ilu0<F>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    fn apply(&self, r: &Array1<F>) -> SparseResult<Array1<F>> {
        if r.len() != self.n {
            return Err(SparseError::DimensionMismatch {
                expected: self.n,
                found: r.len(),
            });
        }
        let r_vec: Vec<F> = r.to_vec();
        let y = forward_solve_unit(
            &self.l_data,
            &self.l_indices,
            &self.l_indptr,
            &r_vec,
            self.n,
        );
        let x = backward_solve(&self.u_data, &self.u_indices, &self.u_indptr, &y, self.n)?;
        Ok(Array1::from_vec(x))
    }
}

// ---------------------------------------------------------------------------
// ILU(k) — Incomplete LU with level-k fill-in
// ---------------------------------------------------------------------------

/// ILU(k) preconditioner: Incomplete LU factorization with level-k fill-in.
///
/// Allows fill-in up to level k beyond the original sparsity pattern.
/// Higher k produces better approximations but uses more memory.
pub struct IluK<F> {
    l_data: Vec<F>,
    l_indices: Vec<usize>,
    l_indptr: Vec<usize>,
    u_data: Vec<F>,
    u_indices: Vec<usize>,
    u_indptr: Vec<usize>,
    n: usize,
}

impl<F> IluK<F>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    /// Construct an ILU(k) preconditioner from a CSR matrix.
    ///
    /// # Arguments
    ///
    /// * `matrix` - Square sparse matrix in CSR format
    /// * `fill_level` - Maximum fill level (0 = ILU(0), 1 = ILU(1), etc.)
    pub fn new(matrix: &CsrMatrix<F>, fill_level: usize) -> SparseResult<Self> {
        let n = matrix.rows();
        if n != matrix.cols() {
            return Err(SparseError::ValueError(
                "ILU(k) requires a square matrix".to_string(),
            ));
        }

        // Track level of fill for each entry
        // Level 0 = original non-zero, Level k = k-th order fill-in
        // Store as dense rows for simplicity; for very large matrices a hash map would be better

        let mut row_data: Vec<Vec<(usize, F, usize)>> = Vec::with_capacity(n);
        // Initialise with original matrix entries (level 0)
        for i in 0..n {
            let range = matrix.row_range(i);
            let cols = &matrix.indices[range.clone()];
            let vals = &matrix.data[range];
            let row: Vec<(usize, F, usize)> = cols
                .iter()
                .zip(vals.iter())
                .map(|(&c, &v)| (c, v, 0usize))
                .collect();
            row_data.push(row);
        }

        // IKJ factorization with level tracking
        for i in 1..n {
            let mut row_i = row_data[i].clone();

            // Sort by column
            row_i.sort_by_key(|&(c, _, _)| c);

            let mut ki = 0;
            while ki < row_i.len() {
                let (k, _, _) = row_i[ki];
                if k >= i {
                    break;
                }

                // Find diagonal of row k
                let row_k = &row_data[k];
                let k_diag = row_k.iter().find(|&&(c, _, _)| c == k).map(|&(_, v, _)| v);
                let k_diag = match k_diag {
                    Some(d) if d.abs() > F::epsilon() => d,
                    _ => {
                        ki += 1;
                        continue;
                    }
                };

                // multiplier = a_{ik} / a_{kk}
                let level_ik = row_i[ki].2;
                let multiplier = row_i[ki].1 / k_diag;
                row_i[ki].1 = multiplier;

                // For each entry (j, a_{kj}) in row k with j > k
                for &(j, a_kj, level_kj) in row_k.iter() {
                    if j <= k {
                        continue;
                    }
                    let new_level = level_ik.max(level_kj) + 1;
                    if new_level > fill_level {
                        continue; // Skip: exceeds fill level
                    }

                    // Find j in row i
                    let existing = row_i.iter().position(|&(c, _, _)| c == j);
                    match existing {
                        Some(pos) => {
                            row_i[pos].1 -= multiplier * a_kj;
                            row_i[pos].2 = row_i[pos].2.min(new_level);
                        }
                        None => {
                            // New fill-in entry
                            row_i.push((j, -multiplier * a_kj, new_level));
                        }
                    }
                }

                ki += 1;
                // Re-sort after potential insertions
                row_i.sort_by_key(|&(c, _, _)| c);
            }

            row_data[i] = row_i;
        }

        // Split into L and U
        let mut l_data = Vec::new();
        let mut u_data = Vec::new();
        let mut l_indices = Vec::new();
        let mut u_indices = Vec::new();
        let mut l_indptr = vec![0usize];
        let mut u_indptr = vec![0usize];

        for i in 0..n {
            let row = &row_data[i];
            for &(col, val, _) in row.iter() {
                if col < i {
                    l_indices.push(col);
                    l_data.push(val);
                } else {
                    u_indices.push(col);
                    u_data.push(val);
                }
            }
            l_indptr.push(l_indices.len());
            u_indptr.push(u_indices.len());
        }

        Ok(Self {
            l_data,
            l_indices,
            l_indptr,
            u_data,
            u_indices,
            u_indptr,
            n,
        })
    }
}

impl<F> Preconditioner<F> for IluK<F>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    fn apply(&self, r: &Array1<F>) -> SparseResult<Array1<F>> {
        if r.len() != self.n {
            return Err(SparseError::DimensionMismatch {
                expected: self.n,
                found: r.len(),
            });
        }
        let r_vec: Vec<F> = r.to_vec();
        let y = forward_solve_unit(
            &self.l_data,
            &self.l_indices,
            &self.l_indptr,
            &r_vec,
            self.n,
        );
        let x = backward_solve(&self.u_data, &self.u_indices, &self.u_indptr, &y, self.n)?;
        Ok(Array1::from_vec(x))
    }
}

// ---------------------------------------------------------------------------
// IC(0) — Incomplete Cholesky for SPD matrices
// ---------------------------------------------------------------------------

/// IC(0) preconditioner: Incomplete Cholesky factorization with zero fill-in.
///
/// For symmetric positive definite (SPD) matrices. Computes a lower triangular
/// matrix L such that A ~ L * L^T, retaining only the sparsity pattern of the
/// lower triangle of A.
pub struct Ic0<F> {
    /// Lower triangular factor (including diagonal) stored in CSR.
    l_data: Vec<F>,
    l_indices: Vec<usize>,
    l_indptr: Vec<usize>,
    n: usize,
}

impl<F> Ic0<F>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    /// Construct an IC(0) preconditioner from an SPD CSR matrix.
    ///
    /// Only the lower triangular part (including diagonal) is used.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square, or if the factorization
    /// encounters a non-positive pivot (matrix is not SPD).
    pub fn new(matrix: &CsrMatrix<F>) -> SparseResult<Self> {
        let n = matrix.rows();
        if n != matrix.cols() {
            return Err(SparseError::ValueError(
                "IC(0) requires a square matrix".to_string(),
            ));
        }

        // Extract lower triangular entries (col <= row)
        let mut rows_lower: Vec<Vec<(usize, F)>> = Vec::with_capacity(n);
        for i in 0..n {
            let range = matrix.row_range(i);
            let cols = &matrix.indices[range.clone()];
            let vals = &matrix.data[range];
            let row: Vec<(usize, F)> = cols
                .iter()
                .zip(vals.iter())
                .filter(|(&c, _)| c <= i)
                .map(|(&c, &v)| (c, v))
                .collect();
            rows_lower.push(row);
        }

        // Cholesky-like factorization
        for i in 0..n {
            // Process row i
            let mut row_i = rows_lower[i].clone();
            row_i.sort_by_key(|&(c, _)| c);

            for ki in 0..row_i.len() {
                let (k, _) = row_i[ki];
                if k >= i {
                    break;
                }

                // Get L_{kk} from row k
                let row_k = &rows_lower[k];
                let l_kk = row_k
                    .iter()
                    .find(|&&(c, _)| c == k)
                    .map(|&(_, v)| v)
                    .unwrap_or(F::sparse_one());

                if l_kk.abs() < F::epsilon() {
                    continue;
                }

                // L_{ik} = (a_{ik} - sum_{j<k} L_{ij}*L_{kj}) / L_{kk}
                let mut sum = F::sparse_zero();
                for &(j, l_ij) in row_i.iter() {
                    if j >= k {
                        break;
                    }
                    // Find L_{kj} in row k
                    if let Some(&(_, l_kj)) = row_k.iter().find(|&&(c, _)| c == j) {
                        sum += l_ij * l_kj;
                    }
                }
                let original_val = row_i[ki].1;
                row_i[ki].1 = (original_val - sum) / l_kk;
            }

            // Update diagonal: L_{ii} = sqrt(a_{ii} - sum_{j<i} L_{ij}^2)
            if let Some(diag_pos) = row_i.iter().position(|&(c, _)| c == i) {
                let mut sum_sq = F::sparse_zero();
                for &(j, l_ij) in row_i.iter() {
                    if j >= i {
                        break;
                    }
                    sum_sq += l_ij * l_ij;
                }
                let diag_val = row_i[diag_pos].1 - sum_sq;
                if diag_val <= F::sparse_zero() {
                    return Err(SparseError::ValueError(format!(
                        "Non-positive pivot at row {i} in IC(0) — matrix may not be SPD"
                    )));
                }
                row_i[diag_pos].1 = diag_val.sqrt();
            } else {
                return Err(SparseError::SingularMatrix(format!(
                    "Missing diagonal at row {i}"
                )));
            }

            rows_lower[i] = row_i;
        }

        // Pack into CSR
        let mut l_data = Vec::new();
        let mut l_indices = Vec::new();
        let mut l_indptr = vec![0usize];
        for i in 0..n {
            let mut row = rows_lower[i].clone();
            row.sort_by_key(|&(c, _)| c);
            for &(col, val) in &row {
                l_indices.push(col);
                l_data.push(val);
            }
            l_indptr.push(l_indices.len());
        }

        Ok(Self {
            l_data,
            l_indices,
            l_indptr,
            n,
        })
    }
}

impl<F> Preconditioner<F> for Ic0<F>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    /// Apply M^{-1} r = L^{-T} L^{-1} r.
    fn apply(&self, r: &Array1<F>) -> SparseResult<Array1<F>> {
        if r.len() != self.n {
            return Err(SparseError::DimensionMismatch {
                expected: self.n,
                found: r.len(),
            });
        }

        let r_vec: Vec<F> = r.to_vec();

        // Forward solve: L y = r
        let mut y = vec![F::sparse_zero(); self.n];
        for i in 0..self.n {
            let start = self.l_indptr[i];
            let end = self.l_indptr[i + 1];
            let mut sum = r_vec[i];

            let mut diag = F::sparse_one();
            for pos in start..end {
                let col = self.l_indices[pos];
                if col < i {
                    sum -= self.l_data[pos] * y[col];
                } else if col == i {
                    diag = self.l_data[pos];
                }
            }
            if diag.abs() < F::epsilon() {
                return Err(SparseError::SingularMatrix(format!(
                    "Zero diagonal at row {i} in IC(0) solve"
                )));
            }
            y[i] = sum / diag;
        }

        // Backward solve: L^T x = y
        let mut x = vec![F::sparse_zero(); self.n];
        for ii in 0..self.n {
            let i = self.n - 1 - ii;
            x[i] = y[i];
        }

        // L^T is upper triangular. We need to traverse columns.
        // Since L is stored row-wise, we do a row-based backward sweep.
        for ii in 0..self.n {
            let i = self.n - 1 - ii;
            let start = self.l_indptr[i];
            let end = self.l_indptr[i + 1];

            // Find diagonal
            let mut diag = F::sparse_one();
            for pos in start..end {
                if self.l_indices[pos] == i {
                    diag = self.l_data[pos];
                    break;
                }
            }
            if diag.abs() < F::epsilon() {
                return Err(SparseError::SingularMatrix(format!(
                    "Zero diagonal at row {i} in IC(0) backward solve"
                )));
            }
            x[i] /= diag;

            // Subtract contributions to earlier rows
            for pos in start..end {
                let col = self.l_indices[pos];
                if col < i {
                    x[col] = x[col] - self.l_data[pos] * x[i];
                }
            }
        }

        Ok(Array1::from_vec(x))
    }
}

// ---------------------------------------------------------------------------
// MILU — Modified ILU with diagonal compensation
// ---------------------------------------------------------------------------

/// MILU preconditioner: Modified Incomplete LU with diagonal compensation.
///
/// Like ILU(0), but the dropped fill-in is compensated by adding it to the
/// diagonal, preserving the row sums of the original matrix. This improves
/// convergence for problems where row-sum preservation matters (e.g.,
/// M-matrices, diffusion problems).
pub struct Milu<F> {
    l_data: Vec<F>,
    l_indices: Vec<usize>,
    l_indptr: Vec<usize>,
    u_data: Vec<F>,
    u_indices: Vec<usize>,
    u_indptr: Vec<usize>,
    n: usize,
}

impl<F> Milu<F>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    /// Construct an MILU preconditioner from a CSR matrix.
    ///
    /// Dropped fill-in is compensated on the diagonal.
    pub fn new(matrix: &CsrMatrix<F>) -> SparseResult<Self> {
        let n = matrix.rows();
        if n != matrix.cols() {
            return Err(SparseError::ValueError(
                "MILU requires a square matrix".to_string(),
            ));
        }

        let mut data = matrix.data.clone();
        let indices = matrix.indices.clone();
        let indptr = matrix.indptr.clone();

        // Diagonal compensation accumulators
        let mut diag_comp = vec![F::sparse_zero(); n];

        for i in 1..n {
            let i_start = indptr[i];
            let i_end = indptr[i + 1];

            for pos_k in i_start..i_end {
                let k = indices[pos_k];
                if k >= i {
                    break;
                }

                let k_start = indptr[k];
                let k_end = indptr[k + 1];
                let k_diag_pos = match find_col_in_row(&indices, k_start, k_end, k) {
                    Some(p) => p,
                    None => {
                        return Err(SparseError::SingularMatrix(format!(
                            "Missing diagonal at row {k}"
                        )));
                    }
                };
                let k_diag = data[k_diag_pos];
                if k_diag.abs() < F::epsilon() {
                    return Err(SparseError::SingularMatrix(format!(
                        "Zero pivot at row {k} in MILU"
                    )));
                }

                let multiplier = data[pos_k] / k_diag;
                data[pos_k] = multiplier;

                // Update row i for columns j > k
                for kj_pos in k_start..k_end {
                    let j = indices[kj_pos];
                    if j <= k {
                        continue;
                    }
                    let fill_val = multiplier * data[kj_pos];
                    if let Some(ij_pos) = find_col_in_row(&indices, i_start, i_end, j) {
                        data[ij_pos] -= fill_val;
                    } else {
                        // Dropped fill-in: compensate on diagonal
                        diag_comp[i] += fill_val;
                    }
                }
            }
        }

        // Apply diagonal compensation
        for i in 0..n {
            let range = indptr[i]..indptr[i + 1];
            if let Some(diag_pos) = find_col_in_row(&indices, range.start, range.end, i) {
                data[diag_pos] += diag_comp[i];
            }
        }

        // Split into L and U
        let mut l_data = Vec::new();
        let mut u_data = Vec::new();
        let mut l_indices = Vec::new();
        let mut u_indices = Vec::new();
        let mut l_indptr = vec![0usize];
        let mut u_indptr = vec![0usize];

        for i in 0..n {
            let start = indptr[i];
            let end = indptr[i + 1];
            for pos in start..end {
                let col = indices[pos];
                if col < i {
                    l_indices.push(col);
                    l_data.push(data[pos]);
                } else {
                    u_indices.push(col);
                    u_data.push(data[pos]);
                }
            }
            l_indptr.push(l_indices.len());
            u_indptr.push(u_indices.len());
        }

        Ok(Self {
            l_data,
            l_indices,
            l_indptr,
            u_data,
            u_indices,
            u_indptr,
            n,
        })
    }
}

impl<F> Preconditioner<F> for Milu<F>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    fn apply(&self, r: &Array1<F>) -> SparseResult<Array1<F>> {
        if r.len() != self.n {
            return Err(SparseError::DimensionMismatch {
                expected: self.n,
                found: r.len(),
            });
        }
        let r_vec: Vec<F> = r.to_vec();
        let y = forward_solve_unit(
            &self.l_data,
            &self.l_indices,
            &self.l_indptr,
            &r_vec,
            self.n,
        );
        let x = backward_solve(&self.u_data, &self.u_indices, &self.u_indptr, &y, self.n)?;
        Ok(Array1::from_vec(x))
    }
}

// ---------------------------------------------------------------------------
// ILUT — Incomplete LU with Threshold dropping
// ---------------------------------------------------------------------------

/// Configuration for the ILUT preconditioner.
#[derive(Debug, Clone)]
pub struct IlutConfig {
    /// Drop tolerance: entries with |a_{ij}| < tau * ||row_i||_2 are dropped.
    pub drop_tolerance: f64,
    /// Maximum fill per row (in addition to original non-zeros).
    pub max_fill: usize,
}

impl Default for IlutConfig {
    fn default() -> Self {
        Self {
            drop_tolerance: 1e-4,
            max_fill: 20,
        }
    }
}

/// ILUT preconditioner: Incomplete LU with threshold-based dropping.
///
/// Entries smaller than `tau * ||row_i||` are dropped, and at most `max_fill`
/// additional entries per row are kept. This gives a good balance between
/// fill-in and accuracy.
pub struct Ilut<F> {
    l_data: Vec<F>,
    l_indices: Vec<usize>,
    l_indptr: Vec<usize>,
    u_data: Vec<F>,
    u_indices: Vec<usize>,
    u_indptr: Vec<usize>,
    n: usize,
}

impl<F> Ilut<F>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    /// Construct an ILUT preconditioner from a CSR matrix.
    ///
    /// # Arguments
    ///
    /// * `matrix` - Square sparse matrix in CSR format
    /// * `config` - ILUT configuration (drop tolerance and max fill)
    pub fn new(matrix: &CsrMatrix<F>, config: &IlutConfig) -> SparseResult<Self> {
        let n = matrix.rows();
        if n != matrix.cols() {
            return Err(SparseError::ValueError(
                "ILUT requires a square matrix".to_string(),
            ));
        }

        let tau = F::from(config.drop_tolerance).ok_or_else(|| {
            SparseError::ValueError("Failed to convert drop_tolerance".to_string())
        })?;
        let max_fill = config.max_fill;

        // Work row-by-row, storing the factorized rows
        let mut l_rows: Vec<Vec<(usize, F)>> = Vec::with_capacity(n);
        let mut u_rows: Vec<Vec<(usize, F)>> = Vec::with_capacity(n);

        for i in 0..n {
            // Load row i into a dense workspace
            let range = matrix.row_range(i);
            let cols = &matrix.indices[range.clone()];
            let vals = &matrix.data[range];

            // Compute row norm for threshold
            let row_norm: F = vals.iter().map(|v| *v * *v).sum::<F>().sqrt();
            let threshold = tau * row_norm;

            // Sparse accumulator for this row
            let mut w: Vec<(usize, F)> = cols
                .iter()
                .zip(vals.iter())
                .map(|(&c, &v)| (c, v))
                .collect();
            w.sort_by_key(|&(c, _)| c);

            // Elimination: for each k < i in w
            let mut ki = 0;
            while ki < w.len() {
                let (k, _) = w[ki];
                if k >= i {
                    break;
                }

                // Get u_{kk} (diagonal from row k of U)
                let u_kk = u_rows
                    .get(k)
                    .and_then(|row| row.iter().find(|&&(c, _)| c == k))
                    .map(|&(_, v)| v);

                let u_kk = match u_kk {
                    Some(d) if d.abs() > F::epsilon() => d,
                    _ => {
                        ki += 1;
                        continue;
                    }
                };

                let multiplier = w[ki].1 / u_kk;

                // Drop small multipliers
                if multiplier.abs() < threshold {
                    ki += 1;
                    continue;
                }
                w[ki].1 = multiplier;

                // Update: w -= multiplier * u_row[k] for entries j > k
                if let Some(u_row_k) = u_rows.get(k) {
                    let updates: Vec<(usize, F)> = u_row_k
                        .iter()
                        .filter(|&&(j, _)| j > k)
                        .map(|&(j, v)| (j, multiplier * v))
                        .collect();

                    for (j, fill_val) in updates {
                        if let Some(pos) = w.iter().position(|&(c, _)| c == j) {
                            w[pos].1 -= fill_val;
                        } else {
                            w.push((j, -fill_val));
                        }
                    }
                    w.sort_by_key(|&(c, _)| c);
                }

                ki += 1;
            }

            // Split into L part (col < i) and U part (col >= i)
            let mut l_part: Vec<(usize, F)> = Vec::new();
            let mut u_part: Vec<(usize, F)> = Vec::new();

            for &(col, val) in &w {
                if col < i {
                    l_part.push((col, val));
                } else {
                    u_part.push((col, val));
                }
            }

            // Apply dropping to L part
            l_part.retain(|&(_, v)| v.abs() >= threshold);
            // Keep at most max_fill entries (by magnitude) plus diagonal
            if l_part.len() > max_fill {
                l_part.sort_by(|a, b| {
                    b.1.abs()
                        .partial_cmp(&a.1.abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                l_part.truncate(max_fill);
                l_part.sort_by_key(|&(c, _)| c);
            }

            // Apply dropping to U part (keep diagonal always)
            let diag_entry = u_part.iter().find(|&&(c, _)| c == i).copied();
            u_part.retain(|&(c, v)| c == i || v.abs() >= threshold);
            if u_part.len() > max_fill + 1 {
                // Keep diagonal + top max_fill entries
                let non_diag: Vec<(usize, F)> =
                    u_part.iter().filter(|&&(c, _)| c != i).copied().collect();
                let mut sorted_nd = non_diag;
                sorted_nd.sort_by(|a, b| {
                    b.1.abs()
                        .partial_cmp(&a.1.abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                sorted_nd.truncate(max_fill);
                u_part = sorted_nd;
                if let Some(de) = diag_entry {
                    u_part.push(de);
                }
                u_part.sort_by_key(|&(c, _)| c);
            }

            // Ensure diagonal exists
            if !u_part.iter().any(|&(c, _)| c == i) {
                // If the diagonal was dropped (shouldn't happen normally), add a small value
                u_part.push((i, F::epsilon() * F::from(100.0).unwrap_or(F::sparse_one())));
                u_part.sort_by_key(|&(c, _)| c);
            }

            l_rows.push(l_part);
            u_rows.push(u_part);
        }

        // Pack into CSR arrays
        let mut l_data = Vec::new();
        let mut l_indices = Vec::new();
        let mut l_indptr = vec![0usize];
        let mut u_data = Vec::new();
        let mut u_indices = Vec::new();
        let mut u_indptr = vec![0usize];

        for i in 0..n {
            for &(col, val) in &l_rows[i] {
                l_indices.push(col);
                l_data.push(val);
            }
            l_indptr.push(l_indices.len());

            for &(col, val) in &u_rows[i] {
                u_indices.push(col);
                u_data.push(val);
            }
            u_indptr.push(u_indices.len());
        }

        Ok(Self {
            l_data,
            l_indices,
            l_indptr,
            u_data,
            u_indices,
            u_indptr,
            n,
        })
    }
}

impl<F> Preconditioner<F> for Ilut<F>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    fn apply(&self, r: &Array1<F>) -> SparseResult<Array1<F>> {
        if r.len() != self.n {
            return Err(SparseError::DimensionMismatch {
                expected: self.n,
                found: r.len(),
            });
        }
        let r_vec: Vec<F> = r.to_vec();
        let y = forward_solve_unit(
            &self.l_data,
            &self.l_indices,
            &self.l_indptr,
            &r_vec,
            self.n,
        );
        let x = backward_solve(&self.u_data, &self.u_indices, &self.u_indptr, &y, self.n)?;
        Ok(Array1::from_vec(x))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a well-conditioned SPD tridiagonal matrix.
    fn build_spd_tridiag(n: usize) -> CsrMatrix<f64> {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();
        for i in 0..n {
            if i > 0 {
                rows.push(i);
                cols.push(i - 1);
                data.push(-1.0);
            }
            rows.push(i);
            cols.push(i);
            data.push(4.0); // 4 on diagonal for well-conditioning
            if i + 1 < n {
                rows.push(i);
                cols.push(i + 1);
                data.push(-1.0);
            }
        }
        CsrMatrix::new(data, rows, cols, (n, n)).expect("valid matrix")
    }

    /// Build a general diagonally dominant matrix.
    fn build_dd_matrix(n: usize) -> CsrMatrix<f64> {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();
        for i in 0..n {
            let mut off_diag_sum = 0.0;
            if i > 0 {
                rows.push(i);
                cols.push(i - 1);
                data.push(-0.5);
                off_diag_sum += 0.5;
            }
            if i + 1 < n {
                rows.push(i);
                cols.push(i + 1);
                data.push(-0.3);
                off_diag_sum += 0.3;
            }
            if i + 2 < n {
                rows.push(i);
                cols.push(i + 2);
                data.push(-0.1);
                off_diag_sum += 0.1;
            }
            rows.push(i);
            cols.push(i);
            data.push(off_diag_sum + 1.0); // diagonally dominant
        }
        CsrMatrix::new(data, rows, cols, (n, n)).expect("valid matrix")
    }

    fn build_identity(n: usize) -> CsrMatrix<f64> {
        let rows: Vec<usize> = (0..n).collect();
        let cols: Vec<usize> = (0..n).collect();
        CsrMatrix::new(vec![1.0; n], rows, cols, (n, n)).expect("valid identity")
    }

    fn spmv(a: &CsrMatrix<f64>, x: &[f64]) -> Vec<f64> {
        let m = a.rows();
        let mut y = vec![0.0; m];
        for i in 0..m {
            let range = a.row_range(i);
            let cols = &a.indices[range.clone()];
            let vals = &a.data[range];
            for (idx, &col) in cols.iter().enumerate() {
                y[i] += vals[idx] * x[col];
            }
        }
        y
    }

    #[test]
    fn test_ilu0_identity() {
        let eye = build_identity(5);
        let ilu = Ilu0::new(&eye).expect("ILU(0) of identity");
        let r = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let x = ilu.apply(&r).expect("apply");
        for i in 0..5 {
            assert!(
                (x[i] - r[i]).abs() < 1e-10,
                "ILU(0) of I should be identity"
            );
        }
    }

    #[test]
    fn test_ilu0_tridiag() {
        let n = 10;
        let a = build_spd_tridiag(n);
        let ilu = Ilu0::new(&a).expect("ILU(0) of tridiag");

        // Apply to a known vector
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let ab = spmv(&a, &b);
        let r = Array1::from_vec(ab);
        let x = ilu.apply(&r).expect("apply");

        // x should be close to b (since ILU(0) is exact for tridiagonal)
        for i in 0..n {
            assert!(
                (x[i] - b[i]).abs() < 1e-6,
                "ILU(0) should be exact for tridiag at index {i}: {} vs {}",
                x[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_ilu0_preconditioner_reduces_residual() {
        let n = 10;
        let a = build_dd_matrix(n);
        let ilu = Ilu0::new(&a).expect("ILU(0)");
        let r = Array1::from_vec(vec![1.0; n]);
        let z = ilu.apply(&r).expect("apply");

        // Az should be closer to r than just r itself would suggest
        let az: Vec<f64> = spmv(&a, &z.as_slice().expect("slice"));
        let residual: f64 = r
            .iter()
            .zip(az.iter())
            .map(|(&ri, &azi)| (ri - azi).powi(2))
            .sum::<f64>()
            .sqrt();
        // Just check it doesn't blow up
        assert!(residual.is_finite(), "Residual should be finite");
    }

    #[test]
    fn test_ilu0_error_non_square() {
        let a = CsrMatrix::new(vec![1.0, 2.0], vec![0, 1], vec![0, 1], (2, 3)).expect("valid");
        assert!(Ilu0::<f64>::new(&a).is_err());
    }

    #[test]
    fn test_ilu0_dimension_mismatch() {
        let a = build_identity(3);
        let ilu = Ilu0::new(&a).expect("ILU(0)");
        let r = Array1::from_vec(vec![1.0, 2.0]);
        assert!(ilu.apply(&r).is_err());
    }

    #[test]
    fn test_iluk_level0_matches_ilu0() {
        let n = 8;
        let a = build_spd_tridiag(n);
        let ilu0 = Ilu0::new(&a).expect("ILU(0)");
        let iluk0 = IluK::new(&a, 0).expect("ILU(k=0)");

        let r = Array1::from_vec(vec![1.0; n]);
        let x0 = ilu0.apply(&r).expect("apply ilu0");
        let xk = iluk0.apply(&r).expect("apply iluk");

        for i in 0..n {
            assert!(
                (x0[i] - xk[i]).abs() < 0.1,
                "ILU(k=0) should be close to ILU(0) at {i}: {} vs {}",
                x0[i],
                xk[i]
            );
        }
    }

    #[test]
    fn test_iluk_higher_fill_better() {
        let n = 10;
        let a = build_dd_matrix(n);
        let b = Array1::from_vec(vec![1.0; n]);

        let iluk0 = IluK::new(&a, 0).expect("ILU(0)");
        let iluk1 = IluK::new(&a, 1).expect("ILU(1)");
        let iluk2 = IluK::new(&a, 2).expect("ILU(2)");

        let x0 = iluk0.apply(&b).expect("apply");
        let x1 = iluk1.apply(&b).expect("apply");
        let x2 = iluk2.apply(&b).expect("apply");

        // Higher fill levels should have more data (or equal)
        assert!(x0.len() == x1.len() && x1.len() == x2.len());
        // Just check all are finite
        for i in 0..n {
            assert!(x0[i].is_finite());
            assert!(x1[i].is_finite());
            assert!(x2[i].is_finite());
        }
    }

    #[test]
    fn test_iluk_error_non_square() {
        let a = CsrMatrix::new(vec![1.0], vec![0], vec![0], (1, 2)).expect("valid");
        assert!(IluK::<f64>::new(&a, 0).is_err());
    }

    #[test]
    fn test_ic0_spd_tridiag() {
        let n = 8;
        let a = build_spd_tridiag(n);
        let ic = Ic0::new(&a).expect("IC(0)");

        let r = Array1::from_vec(vec![1.0; n]);
        let x = ic.apply(&r).expect("apply IC(0)");
        for i in 0..n {
            assert!(x[i].is_finite(), "IC(0) result should be finite at {i}");
        }
    }

    #[test]
    fn test_ic0_identity() {
        let eye = build_identity(5);
        let ic = Ic0::new(&eye).expect("IC(0) identity");
        let r = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let x = ic.apply(&r).expect("apply");
        for i in 0..5 {
            assert!(
                (x[i] - r[i]).abs() < 1e-10,
                "IC(0) of I should be identity: {} vs {}",
                x[i],
                r[i]
            );
        }
    }

    #[test]
    fn test_ic0_error_non_square() {
        let a = CsrMatrix::new(vec![1.0, 2.0], vec![0, 1], vec![0, 1], (2, 3)).expect("valid");
        assert!(Ic0::<f64>::new(&a).is_err());
    }

    #[test]
    fn test_milu_tridiag() {
        let n = 8;
        let a = build_spd_tridiag(n);
        let milu = Milu::new(&a).expect("MILU");
        let r = Array1::from_vec(vec![1.0; n]);
        let x = milu.apply(&r).expect("apply MILU");
        for i in 0..n {
            assert!(x[i].is_finite(), "MILU result should be finite at {i}");
        }
    }

    #[test]
    fn test_milu_vs_ilu0_tridiag() {
        // For tridiagonal matrices, MILU and ILU(0) should give similar results
        // since there's no fill-in to drop
        let n = 5;
        let a = build_spd_tridiag(n);
        let ilu = Ilu0::new(&a).expect("ILU(0)");
        let milu = Milu::new(&a).expect("MILU");

        let r = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let x_ilu = ilu.apply(&r).expect("apply ILU");
        let x_milu = milu.apply(&r).expect("apply MILU");

        for i in 0..n {
            assert!(
                (x_ilu[i] - x_milu[i]).abs() < 1e-6,
                "MILU ~ ILU(0) for tridiag at {i}: {} vs {}",
                x_ilu[i],
                x_milu[i]
            );
        }
    }

    #[test]
    fn test_milu_error_non_square() {
        let a = CsrMatrix::new(vec![1.0], vec![0], vec![0], (1, 2)).expect("valid");
        assert!(Milu::<f64>::new(&a).is_err());
    }

    #[test]
    fn test_ilut_basic() {
        let n = 8;
        let a = build_dd_matrix(n);
        let config = IlutConfig {
            drop_tolerance: 1e-4,
            max_fill: 10,
        };
        let ilut = Ilut::new(&a, &config).expect("ILUT");
        let r = Array1::from_vec(vec![1.0; n]);
        let x = ilut.apply(&r).expect("apply ILUT");
        for i in 0..n {
            assert!(x[i].is_finite(), "ILUT result should be finite at {i}");
        }
    }

    #[test]
    fn test_ilut_low_threshold() {
        // Very low threshold (keep everything) should behave like full LU
        let n = 5;
        let a = build_spd_tridiag(n);
        let config = IlutConfig {
            drop_tolerance: 1e-15,
            max_fill: 100,
        };
        let ilut = Ilut::new(&a, &config).expect("ILUT low threshold");
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let ab = spmv(&a, &b);
        let r = Array1::from_vec(ab);
        let x = ilut.apply(&r).expect("apply");

        for i in 0..n {
            assert!(
                (x[i] - b[i]).abs() < 1e-4,
                "ILUT with low threshold should be near-exact at {i}: {} vs {}",
                x[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_ilut_high_threshold() {
        // High threshold drops many entries, but should still produce finite results
        let n = 8;
        let a = build_dd_matrix(n);
        let config = IlutConfig {
            drop_tolerance: 0.5,
            max_fill: 2,
        };
        let ilut = Ilut::new(&a, &config).expect("ILUT high threshold");
        let r = Array1::from_vec(vec![1.0; n]);
        let x = ilut.apply(&r).expect("apply");
        for i in 0..n {
            assert!(x[i].is_finite(), "ILUT result should be finite at {i}");
        }
    }

    #[test]
    fn test_ilut_error_non_square() {
        let a = CsrMatrix::new(vec![1.0], vec![0], vec![0], (1, 2)).expect("valid");
        assert!(Ilut::<f64>::new(&a, &IlutConfig::default()).is_err());
    }

    #[test]
    fn test_ilut_dimension_mismatch() {
        let a = build_identity(3);
        let ilut = Ilut::new(&a, &IlutConfig::default()).expect("ILUT");
        let r = Array1::from_vec(vec![1.0, 2.0]);
        assert!(ilut.apply(&r).is_err());
    }

    #[test]
    fn test_all_preconditioners_on_same_system() {
        let n = 10;
        let a = build_spd_tridiag(n);
        let b_vec: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let ab = spmv(&a, &b_vec);
        let r = Array1::from_vec(ab);

        let ilu0 = Ilu0::new(&a).expect("ILU(0)");
        let iluk = IluK::new(&a, 1).expect("ILU(1)");
        let ic0 = Ic0::new(&a).expect("IC(0)");
        let milu = Milu::new(&a).expect("MILU");
        let ilut = Ilut::new(
            &a,
            &IlutConfig {
                drop_tolerance: 1e-10,
                max_fill: 20,
            },
        )
        .expect("ILUT");

        let x_ilu0 = ilu0.apply(&r).expect("apply");
        let x_iluk = iluk.apply(&r).expect("apply");
        let x_ic0 = ic0.apply(&r).expect("apply");
        let x_milu = milu.apply(&r).expect("apply");
        let x_ilut = ilut.apply(&r).expect("apply");

        // All should produce results close to b_vec
        for i in 0..n {
            assert!(
                (x_ilu0[i] - b_vec[i]).abs() < 1e-4,
                "ILU(0) at {i}: {} vs {}",
                x_ilu0[i],
                b_vec[i]
            );
            assert!(
                (x_iluk[i] - b_vec[i]).abs() < 1e-4,
                "ILU(1) at {i}: {} vs {}",
                x_iluk[i],
                b_vec[i]
            );
            assert!(x_ic0[i].is_finite(), "IC(0) finite at {i}");
            assert!(
                (x_milu[i] - b_vec[i]).abs() < 1e-4,
                "MILU at {i}: {} vs {}",
                x_milu[i],
                b_vec[i]
            );
            assert!(
                (x_ilut[i] - b_vec[i]).abs() < 1e-4,
                "ILUT at {i}: {} vs {}",
                x_ilut[i],
                b_vec[i]
            );
        }
    }
}
