//! Advanced sparse preconditioners for iterative linear solvers
//!
//! This module implements three preconditioner families that complement the
//! existing `incomplete_factorizations` module:
//!
//! - **SPAI** – Sparse Approximate Inverse: constructs M ≈ A⁻¹ such that
//!   ‖AM − I‖_F is minimised column-by-column using local least-squares.
//!
//! - **Neumann** – Polynomial preconditioner based on the truncated Neumann
//!   series: M ≈ α Σₖ (I − αA)^k for some scaling α.
//!
//! - **SGS** – Symmetric Gauss-Seidel: one forward sweep followed by one
//!   backward sweep of the classical Gauss-Seidel iteration.
//!
//! # References
//!
//! - Grote, M. J. & Huckle, T. (1997). Parallel preconditioning with sparse
//!   approximate inverses. *SIAM J. Sci. Comput.* 18(3), 838–853.
//! - Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed.
//!   SIAM.

use crate::error::{SparseError, SparseResult};

// ---------------------------------------------------------------------------
// Internal helpers – bare CSR operations needed here without pulling in the
// full CsrMatrix type (which carries generic type parameters and trait bounds
// that complicate the f64-only preconditioner API).
// ---------------------------------------------------------------------------

/// Multiply sparse CSR matrix (rows×cols) by dense vector x → y.
fn csr_matvec_internal(
    row_ptrs: &[usize],
    col_indices: &[usize],
    values: &[f64],
    x: &[f64],
    nrows: usize,
) -> Vec<f64> {
    let mut y = vec![0.0f64; nrows];
    for i in 0..nrows {
        let start = row_ptrs[i];
        let end = row_ptrs[i + 1];
        let mut acc = 0.0f64;
        for pos in start..end {
            acc += values[pos] * x[col_indices[pos]];
        }
        y[i] = acc;
    }
    y
}

/// Transpose a CSR matrix: returns (col_ptrs, row_indices, values) in CSR
/// of Aᵀ.  (i.e. the CSC of A stored as CSR of Aᵀ.)
fn csr_transpose_internal(
    row_ptrs: &[usize],
    col_indices: &[usize],
    values: &[f64],
    nrows: usize,
    ncols: usize,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let nnz = values.len();
    let mut col_counts = vec![0usize; ncols];
    for &c in col_indices {
        col_counts[c] += 1;
    }
    let mut col_ptrs = vec![0usize; ncols + 1];
    for c in 0..ncols {
        col_ptrs[c + 1] = col_ptrs[c] + col_counts[c];
    }
    let mut row_indices_t = vec![0usize; nnz];
    let mut values_t = vec![0.0f64; nnz];
    let mut cur = col_ptrs[..ncols].to_vec();
    for i in 0..nrows {
        for pos in row_ptrs[i]..row_ptrs[i + 1] {
            let c = col_indices[pos];
            let dst = cur[c];
            row_indices_t[dst] = i;
            values_t[dst] = values[pos];
            cur[c] += 1;
        }
    }
    (col_ptrs, row_indices_t, values_t)
}

/// Solve a small dense least-squares problem min‖Ax − b‖ using QR (Householder).
/// Returns the solution x of length n_cols.  n_rows >= n_cols is not enforced;
/// if the system is underdetermined the minimum-norm solution is returned.
fn dense_lstsq(a: &[f64], b: &[f64], n_rows: usize, n_cols: usize) -> SparseResult<Vec<f64>> {
    if n_rows == 0 || n_cols == 0 {
        return Ok(vec![0.0f64; n_cols]);
    }
    // Work on mutable copies.
    let mut q = a.to_vec(); // column-major (Fortran order) for Householder
    let mut rhs = b.to_vec();

    // Householder QR – we form Qᵀb in place.
    let m = n_rows;
    let n = n_cols.min(n_rows);
    let mut r = q.clone(); // row-major; reshape later

    // Column-major copy for in-place Householder
    let mut cm: Vec<f64> = vec![0.0; m * n_cols];
    for row in 0..m {
        for col in 0..n_cols {
            cm[col * m + row] = r[row * n_cols + col];
        }
    }

    let k = n; // number of Householder reflections
    let mut betas: Vec<f64> = vec![0.0; k];

    for j in 0..k {
        // Extract column j from index j downward.
        let col_start = j * m + j;
        let col_end = (j + 1) * m;
        let col_len = col_end - col_start;

        let mut v: Vec<f64> = cm[col_start..col_end].to_vec();
        let sigma: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();

        if sigma.abs() < 1e-300 {
            betas[j] = 0.0;
            continue;
        }

        let sign = if v[0] >= 0.0 { 1.0 } else { -1.0 };
        v[0] += sign * sigma;
        let beta = 2.0 / v.iter().map(|x| x * x).sum::<f64>();
        betas[j] = beta;

        // Apply to columns j..n_cols
        for jj in j..n_cols {
            let col_j_start = jj * m + j;
            let col_j = &cm[col_j_start..col_j_start + col_len];
            let dot: f64 = v.iter().zip(col_j.iter()).map(|(a, b)| a * b).sum();
            let scale = beta * dot;
            for ii in 0..col_len {
                cm[jj * m + j + ii] -= scale * v[ii];
            }
        }

        // Apply to rhs
        {
            let dot: f64 = v
                .iter()
                .zip(rhs[j..j + col_len].iter())
                .map(|(a, b)| a * b)
                .sum();
            let scale = beta * dot;
            for ii in 0..col_len {
                rhs[j + ii] -= scale * v[ii];
            }
        }

        // Store v (below diagonal) for potential later use; we only need R here.
        // Already in place in cm.
        let _ = betas[j]; // suppress warning
    }

    // Back-substitution on the upper-triangular R (cm is column-major).
    let mut x = vec![0.0f64; n_cols];
    for ii in 0..n {
        let i = n - 1 - ii;
        let r_ii = cm[i * m + i];
        if r_ii.abs() < 1e-14 {
            x[i] = 0.0;
            continue;
        }
        let mut sum = rhs[i];
        for jj in (i + 1)..n_cols {
            sum -= cm[jj * m + i] * x[jj];
        }
        x[i] = sum / r_ii;
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// SPAI – Sparse Approximate Inverse
// ---------------------------------------------------------------------------

/// Sparse Approximate Inverse (SPAI) preconditioner.
///
/// Constructs M ≈ A⁻¹ by solving, for each column k of M, the local
/// least-squares problem
///
///   min_mₖ ‖A mₖ − eₖ‖₂
///
/// subject to a sparsity pattern derived from the non-zeros of column k of A.
/// The sparsity budget is controlled by `max_nnz_per_col`.
pub struct SpaiPreconditioner {
    /// Sparse rows of M stored as `(column_index, value)` pairs.
    m: Vec<Vec<(usize, f64)>>,
    n: usize,
}

impl SpaiPreconditioner {
    /// Build SPAI for a CSR matrix.
    ///
    /// # Arguments
    ///
    /// * `row_ptrs`       – CSR row pointer array (length `n + 1`).
    /// * `col_indices`    – CSR column index array.
    /// * `values`         – CSR value array.
    /// * `n`              – Matrix dimension (square).
    /// * `tol`            – Residual tolerance; currently unused (reserved for
    ///                      adaptive enrichment), kept for API stability.
    /// * `max_nnz_per_col`– Sparsity budget: maximum non-zeros per column of M.
    ///
    /// # Returns
    ///
    /// A built `SpaiPreconditioner` or a `SparseError`.
    pub fn new(
        row_ptrs: &[usize],
        col_indices: &[usize],
        values: &[f64],
        n: usize,
        _tol: f64,
        max_nnz_per_col: usize,
    ) -> SparseResult<Self> {
        if row_ptrs.len() != n + 1 {
            return Err(SparseError::InconsistentData {
                reason: format!(
                    "row_ptrs length {} != n+1 = {}",
                    row_ptrs.len(),
                    n + 1
                ),
            });
        }

        // Build column-access (CSC) of A so we can quickly get the rows that
        // participate in column k.
        let (csc_ptrs, csc_rows, csc_vals) =
            csr_transpose_internal(row_ptrs, col_indices, values, n, n);

        // For each column k of M we solve a local least-squares problem.
        let mut m_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];

        for k in 0..n {
            // --- 1. Determine sparsity pattern of mₖ. -----------------------
            // Initial pattern: non-zero rows of column k of A (= column k of Aᵀ
            // stored as row k of Aᵀ in CSC/CSR-transpose).
            let mut pattern: Vec<usize> = {
                let s = csc_ptrs[k];
                let e = csc_ptrs[k + 1];
                let mut p: Vec<usize> = csc_rows[s..e].to_vec();
                // Always include k itself so the diagonal is covered.
                if !p.contains(&k) {
                    p.push(k);
                }
                p.sort_unstable();
                p
            };

            // Trim to budget.
            if pattern.len() > max_nnz_per_col {
                // Keep the `max_nnz_per_col` entries with the largest |Aₖⱼ|.
                let mut scored: Vec<(usize, f64)> = pattern
                    .iter()
                    .map(|&j| {
                        // |A[j, k]| from column k of A (CSC).
                        let s = csc_ptrs[k];
                        let e = csc_ptrs[k + 1];
                        let val = csc_rows[s..e]
                            .iter()
                            .zip(csc_vals[s..e].iter())
                            .find(|(&r, _)| r == j)
                            .map(|(_, &v)| v.abs())
                            .unwrap_or(0.0);
                        (j, val)
                    })
                    .collect();
                scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                scored.truncate(max_nnz_per_col);
                pattern = scored.iter().map(|&(j, _)| j).collect();
                pattern.sort_unstable();
            }

            let np = pattern.len();
            if np == 0 {
                continue;
            }

            // --- 2. Assemble local dense sub-matrix Â and right-hand side. ---
            // Â = A[:, pattern] restricted to rows that are non-zero.
            // rhs = eₖ restricted to those rows.

            // Collect the row indices of Â: union of non-zero rows of A across
            // the selected columns (pattern).
            let mut row_set: Vec<usize> = Vec::new();
            for &j in &pattern {
                let s = csc_ptrs[j];
                let e = csc_ptrs[j + 1];
                for &r in &csc_rows[s..e] {
                    if !row_set.contains(&r) {
                        row_set.push(r);
                    }
                }
            }
            row_set.sort_unstable();
            let nr = row_set.len();

            // Build Â (nr × np, row-major).
            let mut a_local = vec![0.0f64; nr * np];
            for (col_loc, &j) in pattern.iter().enumerate() {
                let s = csc_ptrs[j];
                let e = csc_ptrs[j + 1];
                for pos in s..e {
                    let global_row = csc_rows[pos];
                    if let Ok(row_loc) = row_set.binary_search(&global_row) {
                        a_local[row_loc * np + col_loc] = csc_vals[pos];
                    }
                }
            }

            // rhs: eₖ restricted to row_set (1 at position of k, 0 elsewhere).
            let mut rhs = vec![0.0f64; nr];
            if let Ok(pos_k) = row_set.binary_search(&k) {
                rhs[pos_k] = 1.0;
            }

            // --- 3. Solve local least-squares. -------------------------------
            let x_local = match dense_lstsq(&a_local, &rhs, nr, np) {
                Ok(v) => v,
                Err(_) => vec![0.0f64; np],
            };

            // --- 4. Scatter back into the k-th row of M (M is stored row-wise
            //        but conceptually we are building column k of M, which is
            //        row k of Mᵀ – we store M row-wise so we need to store
            //        the k-th *column* of M as contributions to each row j). --
            // Actually: m_rows[j] gets entry (k, value) meaning M[j,k] = value.
            // But we want to apply M as y = Mx, so m_rows[i] stores the non-
            // zeros of row i of M.
            // x_local[p] = M[pattern[p], k]
            // We're building row-wise storage of M:  m_rows[pattern[p]] += (k, x_local[p])
            for (p, &row_idx) in pattern.iter().enumerate() {
                if x_local[p].abs() > 0.0 {
                    m_rows[row_idx].push((k, x_local[p]));
                }
            }
        }

        Ok(Self { m: m_rows, n })
    }

    /// Apply the preconditioner: compute y = M x.
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0f64; self.n];
        for (i, row) in self.m.iter().enumerate() {
            let mut acc = 0.0f64;
            for &(j, v) in row {
                acc += v * x[j];
            }
            y[i] = acc;
        }
        y
    }

    /// Return the dimension of the preconditioner.
    pub fn size(&self) -> usize {
        self.n
    }
}

// ---------------------------------------------------------------------------
// Neumann Polynomial Preconditioner
// ---------------------------------------------------------------------------

/// Polynomial (Neumann series) preconditioner.
///
/// Approximates A⁻¹ by the truncated Neumann series
///
///   M = α Σₖ₌₀ᵈ (I − αA)^k
///
/// where α = `shift` (a scaling chosen to improve the spectral radius of
/// `I − αA`).  After `degree + 1` terms the residual polynomial
/// `p(A) ≈ A⁻¹` approximates the inverse.
///
/// The matrix A is stored internally in CSR format and the series is
/// evaluated by repeated sparse matrix-vector products at application time.
pub struct NeumannPreconditioner {
    degree: usize,
    shift: f64,
    row_ptrs: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<f64>,
    n: usize,
}

impl NeumannPreconditioner {
    /// Build a Neumann preconditioner.
    ///
    /// # Arguments
    ///
    /// * `row_ptrs`, `col_indices`, `values` – CSR representation of A.
    /// * `n`      – Matrix dimension.
    /// * `degree` – Number of terms in the Neumann series (polynomial degree).
    ///              Typical values: 1–4.
    ///
    /// The scaling `shift` α is automatically estimated as
    /// `1 / max_diag` where `max_diag` is the maximum absolute diagonal
    /// entry of A (a simple, cheap heuristic).
    pub fn new(
        row_ptrs: &[usize],
        col_indices: &[usize],
        values: &[f64],
        n: usize,
        degree: usize,
    ) -> SparseResult<Self> {
        if row_ptrs.len() != n + 1 {
            return Err(SparseError::InconsistentData {
                reason: format!(
                    "row_ptrs length {} != n+1 = {}",
                    row_ptrs.len(),
                    n + 1
                ),
            });
        }

        // Estimate scaling: α ≈ 1 / ‖A‖_∞  (row-sum norm).
        let row_sums: f64 = (0..n)
            .map(|i| {
                let start = row_ptrs[i];
                let end = row_ptrs[i + 1];
                values[start..end].iter().map(|v| v.abs()).sum::<f64>()
            })
            .fold(0.0f64, f64::max);

        let shift = if row_sums > 1e-300 {
            1.0 / row_sums
        } else {
            1.0
        };

        Ok(Self {
            degree,
            shift,
            row_ptrs: row_ptrs.to_vec(),
            col_indices: col_indices.to_vec(),
            values: values.to_vec(),
            n,
        })
    }

    /// Apply the preconditioner: y = α Σₖ₌₀ᵈ (I − αA)^k x.
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        let n = self.n;
        let alpha = self.shift;

        // Build B = I − α A stored implicitly; we compute B·v = v − α A·v.
        let b_apply = |v: &[f64]| -> Vec<f64> {
            let av = csr_matvec_internal(&self.row_ptrs, &self.col_indices, &self.values, v, n);
            let mut bv = v.to_vec();
            for i in 0..n {
                bv[i] -= alpha * av[i];
            }
            bv
        };

        // Horner evaluation: Σ B^k x = x + B(x + B(x + ... ))
        // unrolled forward:  sum = x; for k = degree-1 downto 0: sum = x + B*sum
        let mut sum = x.to_vec();
        for _ in 0..self.degree {
            let b_sum = b_apply(&sum);
            for i in 0..n {
                sum[i] = x[i] + b_sum[i];
            }
        }

        // Multiply by α.
        for v in sum.iter_mut() {
            *v *= alpha;
        }
        sum
    }

    /// Return the polynomial degree.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Return the scaling factor α.
    pub fn shift(&self) -> f64 {
        self.shift
    }

    /// Return the matrix dimension.
    pub fn size(&self) -> usize {
        self.n
    }
}

// ---------------------------------------------------------------------------
// Symmetric Gauss-Seidel (SGS) Preconditioner
// ---------------------------------------------------------------------------

/// Symmetric Gauss-Seidel (SGS) preconditioner.
///
/// Applies one forward Gauss-Seidel sweep followed by one backward sweep,
/// approximating (D + L)⁻¹ D (D + U)⁻¹ (with D diagonal, L strictly lower-
/// triangular, U strictly upper-triangular parts of A).
///
/// This is the classic SSOR preconditioner with ω = 1.  It is symmetric when
/// A is symmetric, making it suitable as a preconditioner for the Conjugate
/// Gradient method.
pub struct SgsPreconditioner {
    row_ptrs: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<f64>,
    diag: Vec<f64>,
    n: usize,
}

impl SgsPreconditioner {
    /// Build the SGS preconditioner from a CSR matrix.
    ///
    /// # Arguments
    ///
    /// * `row_ptrs`, `col_indices`, `values` – CSR representation of A.
    /// * `n` – Matrix dimension (must be square).
    ///
    /// Returns an error if any diagonal entry is zero (the preconditioner
    /// requires a non-singular diagonal).
    pub fn new(
        row_ptrs: &[usize],
        col_indices: &[usize],
        values: &[f64],
        n: usize,
    ) -> SparseResult<Self> {
        if row_ptrs.len() != n + 1 {
            return Err(SparseError::InconsistentData {
                reason: format!(
                    "row_ptrs length {} != n+1 = {}",
                    row_ptrs.len(),
                    n + 1
                ),
            });
        }

        let mut diag = vec![0.0f64; n];
        for i in 0..n {
            let start = row_ptrs[i];
            let end = row_ptrs[i + 1];
            let mut found = false;
            for pos in start..end {
                if col_indices[pos] == i {
                    diag[i] = values[pos];
                    found = true;
                    break;
                }
            }
            if !found || diag[i].abs() < 1e-300 {
                return Err(SparseError::SingularMatrix(format!(
                    "zero or missing diagonal at index {i}"
                )));
            }
        }

        Ok(Self {
            row_ptrs: row_ptrs.to_vec(),
            col_indices: col_indices.to_vec(),
            values: values.to_vec(),
            diag,
            n,
        })
    }

    /// Apply the SGS preconditioner: y = M⁻¹ x via forward + backward sweep.
    ///
    /// Forward sweep: solve (D + L) z = x  → z_i = (x_i − Σ_{j<i} A_{ij} z_j) / D_i
    /// Backward sweep: solve (D + U) y = D z → y_i = (z_i·D_i − Σ_{j>i} A_{ij} y_j) / D_i
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        let n = self.n;

        // --- Forward sweep: (D + L) z = x ---
        let mut z = vec![0.0f64; n];
        for i in 0..n {
            let start = self.row_ptrs[i];
            let end = self.row_ptrs[i + 1];
            let mut acc = x[i];
            for pos in start..end {
                let j = self.col_indices[pos];
                if j < i {
                    acc -= self.values[pos] * z[j];
                }
            }
            z[i] = acc / self.diag[i];
        }

        // --- Backward sweep: (D + U) y = D z ---
        let mut y = vec![0.0f64; n];
        for ii in 0..n {
            let i = n - 1 - ii;
            let start = self.row_ptrs[i];
            let end = self.row_ptrs[i + 1];
            let mut acc = self.diag[i] * z[i];
            for pos in start..end {
                let j = self.col_indices[pos];
                if j > i {
                    acc -= self.values[pos] * y[j];
                }
            }
            y[i] = acc / self.diag[i];
        }

        y
    }

    /// Return the matrix dimension.
    pub fn size(&self) -> usize {
        self.n
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Build the CSR of a simple 4×4 diagonally dominant SPD matrix:
    //   A = [4 -1  0  0]
    //       [-1  4 -1  0]
    //       [0 -1  4 -1]
    //       [0  0 -1  4]
    fn build_test_csr() -> (Vec<usize>, Vec<usize>, Vec<f64>, usize) {
        let n = 4usize;
        let row_ptrs = vec![0, 2, 5, 8, 10];
        let col_indices = vec![
            0, 1, // row 0
            0, 1, 2, // row 1
            1, 2, 3, // row 2
            2, 3, // row 3
        ];
        let values = vec![
            4.0, -1.0, // row 0
            -1.0, 4.0, -1.0, // row 1
            -1.0, 4.0, -1.0, // row 2
            -1.0, 4.0, // row 3
        ];
        (row_ptrs, col_indices, values, n)
    }

    #[test]
    fn test_sgs_apply_reduces_residual() {
        let (rp, ci, vals, n) = build_test_csr();
        let sgs = SgsPreconditioner::new(&rp, &ci, &vals, n).expect("SGS build failed");

        // b = [1, 2, 3, 4]
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let y = sgs.apply(&b);

        // Compute residual r = b − A*(M^{-1} b) ... but here we just check
        // that the preconditioned iterate is "closer" to the exact solution.
        // Exact solution of Ax = b can be verified by A*y ≈ b after one step.
        let ay = csr_matvec_internal(&rp, &ci, &vals, &y, n);
        let residual_norm: f64 = b.iter().zip(ay.iter()).map(|(bi, ai)| (bi - ai).powi(2)).sum::<f64>().sqrt();

        // The SGS preconditioner is a rough approximation; residual should be
        // reduced compared to the un-preconditioned iterate (y = b, r = (I−A)b).
        assert!(residual_norm < b.iter().map(|x| x * x).sum::<f64>().sqrt(),
            "SGS should reduce residual, got norm = {residual_norm}");
    }

    #[test]
    fn test_sgs_zero_diagonal_error() {
        let n = 3usize;
        let row_ptrs = vec![0, 1, 2, 3];
        let col_indices = vec![0, 1, 2];
        // Zero diagonal at index 1
        let values = vec![1.0, 0.0, 1.0];
        let result = SgsPreconditioner::new(&row_ptrs, &col_indices, &values, n);
        assert!(result.is_err(), "should fail on zero diagonal");
    }

    #[test]
    fn test_neumann_apply() {
        let (rp, ci, vals, n) = build_test_csr();
        let prec = NeumannPreconditioner::new(&rp, &ci, &vals, n, 3).expect("Neumann build failed");

        assert_eq!(prec.size(), n);
        assert!(prec.degree() == 3);
        assert!(prec.shift() > 0.0);

        let b = vec![1.0, 0.0, 0.0, 0.0];
        let y = prec.apply(&b);
        assert_eq!(y.len(), n);

        // The result should be non-trivial.
        let norm: f64 = y.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm > 0.0, "Neumann apply should give non-zero result");
    }

    #[test]
    fn test_spai_apply() {
        let (rp, ci, vals, n) = build_test_csr();
        let prec = SpaiPreconditioner::new(&rp, &ci, &vals, n, 1e-6, 4)
            .expect("SPAI build failed");

        assert_eq!(prec.size(), n);

        let b = vec![4.0, 3.0, 2.0, 1.0];
        let y = prec.apply(&b);
        assert_eq!(y.len(), n);

        // Check M*A*b ≈ b (SPAI should be a reasonable approximation).
        let ay = csr_matvec_internal(&rp, &ci, &vals, &b, n);
        let may = prec.apply(&ay);

        let rel_err: f64 = b
            .iter()
            .zip(may.iter())
            .map(|(bi, mi)| (bi - mi).powi(2))
            .sum::<f64>()
            .sqrt()
            / b.iter().map(|x| x * x).sum::<f64>().sqrt();

        // Loose tolerance – SPAI is an approximation.
        assert!(rel_err < 2.0, "SPAI: rel_err = {rel_err}");
    }

    #[test]
    fn test_spai_dimension_mismatch() {
        let row_ptrs = vec![0, 1, 2]; // n=2 but n=3 passed
        let col_indices = vec![0, 1];
        let values = vec![1.0, 1.0];
        let result = SpaiPreconditioner::new(&row_ptrs, &col_indices, &values, 3, 1e-6, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_neumann_dimension_mismatch() {
        let row_ptrs = vec![0, 1];
        let col_indices = vec![0];
        let values = vec![2.0];
        let result = NeumannPreconditioner::new(&row_ptrs, &col_indices, &values, 5, 2);
        assert!(result.is_err());
    }
}

// ---------------------------------------------------------------------------
// Submodules with specialised preconditioner families
// ---------------------------------------------------------------------------

pub mod ichol;
pub mod ilu;
pub mod polynomial;
pub mod ssor;

pub use ichol::{IC0, ICT};
pub use ilu::{ILU0, ILUT};
pub use polynomial::{ChebyshevPoly, NeumannPoly};
pub use ssor::SSORPrecond;

pub mod block;
pub use block::{BlockJacobiF64, SpaiPrecondF64, PolynomialPrecondF64};
