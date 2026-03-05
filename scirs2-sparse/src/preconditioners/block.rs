//! Block preconditioners for sparse linear systems
//!
//! This module provides block-structured preconditioners that partition the
//! system matrix into blocks and operate on them independently:
//!
//! - [`BlockJacobiF64`] — Uniform or variable block-size block Jacobi, inverts
//!   diagonal blocks via dense LU.
//! - [`SpaiPrecondF64`] — Sparse Approximate Inverse: minimises ‖I − M A‖_F
//!   column-by-column.
//! - [`PolynomialPrecondF64`] — Neumann-series polynomial preconditioner.

use crate::error::{SparseError, SparseResult};

// ---------------------------------------------------------------------------
// Internal CSR helpers
// ---------------------------------------------------------------------------

fn csr_matvec(
    row_ptr: &[usize],
    col_ind: &[usize],
    val: &[f64],
    x: &[f64],
    n: usize,
) -> Vec<f64> {
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut acc = 0.0f64;
        for pos in row_ptr[i]..row_ptr[i + 1] {
            acc += val[pos] * x[col_ind[pos]];
        }
        y[i] = acc;
    }
    y
}

// ---------------------------------------------------------------------------
// Dense LU (partial pivoting) for small blocks
// ---------------------------------------------------------------------------

/// In-place LU decomposition with partial pivoting.
/// Returns the permutation vector (row indices of P*A = L*U).
/// Returns `None` if the matrix is structurally singular.
fn dense_lu(a: &mut Vec<Vec<f64>>, n: usize) -> Option<Vec<usize>> {
    let mut perm: Vec<usize> = (0..n).collect();
    for k in 0..n {
        // Find pivot
        let mut max_val = a[k][k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let v = a[i][k].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        if max_val < 1e-300 {
            return None;
        }
        a.swap(k, max_row);
        perm.swap(k, max_row);
        let pivot = a[k][k];
        for i in (k + 1)..n {
            a[i][k] /= pivot;
            for j in (k + 1)..n {
                let l = a[i][k];
                a[i][j] -= l * a[k][j];
            }
        }
    }
    Some(perm)
}

/// Solve L * U * x = P * b using pre-computed LU factors and permutation.
fn dense_lu_solve(lu: &[Vec<f64>], perm: &[usize], b: &[f64], n: usize) -> Vec<f64> {
    // Apply permutation
    let mut y: Vec<f64> = perm.iter().map(|&p| b[p]).collect();
    // Forward substitution: L * y = Pb (L has implicit 1s on diagonal)
    for i in 0..n {
        for j in 0..i {
            y[i] -= lu[i][j] * y[j];
        }
    }
    // Backward substitution: U * x = y
    let mut x = y;
    for ii in 0..n {
        let i = n - 1 - ii;
        for j in (i + 1)..n {
            x[i] -= lu[i][j] * x[j];
        }
        x[i] /= lu[i][i];
    }
    x
}

// ---------------------------------------------------------------------------
// Block Jacobi Preconditioner
// ---------------------------------------------------------------------------

/// Block Jacobi preconditioner.
///
/// Partitions the matrix into (potentially variable-size) diagonal blocks,
/// inverts each block via dense LU, and applies the block-diagonal inverse.
///
/// # Usage
///
/// ```rust
/// use scirs2_sparse::preconditioners::block::BlockJacobiF64;
///
/// let n = 4;
/// let row_ptr = vec![0usize, 2, 4, 6, 8];
/// let col_ind = vec![0usize, 1, 0, 1, 2, 3, 2, 3];
/// let val     = vec![4.0, -1.0, -1.0, 4.0, 4.0, -1.0, -1.0, 4.0];
/// let prec = BlockJacobiF64::new_uniform(&val, &row_ptr, &col_ind, n, 2)
///     .expect("build");
/// let y = prec.apply(&[1.0, 2.0, 3.0, 4.0]);
/// assert_eq!(y.len(), n);
/// ```
pub struct BlockJacobiF64 {
    /// Dense LU factors for each diagonal block.
    lu_factors: Vec<Vec<Vec<f64>>>,
    /// Permutation vectors from LU factorization of each block.
    lu_perms: Vec<Vec<usize>>,
    /// Start index of each block in the global ordering.
    block_starts: Vec<usize>,
    /// Size of each block.
    block_sizes: Vec<usize>,
    /// Global dimension.
    n: usize,
}

impl BlockJacobiF64 {
    /// Create a uniform block Jacobi preconditioner.
    ///
    /// The matrix is partitioned into blocks of size `block_size` (the last
    /// block may be smaller if `n` is not divisible by `block_size`).
    pub fn new_uniform(
        csr_val: &[f64],
        csr_row_ptr: &[usize],
        csr_col_ind: &[usize],
        n: usize,
        block_size: usize,
    ) -> SparseResult<Self> {
        if block_size == 0 {
            return Err(SparseError::InvalidArgument(
                "block_size must be positive".to_string(),
            ));
        }
        let mut starts = Vec::new();
        let mut i = 0;
        while i < n {
            starts.push(i);
            i += block_size;
        }
        Self::new_with_partition(csr_val, csr_row_ptr, csr_col_ind, n, starts)
    }

    /// Create a block Jacobi preconditioner with an explicit partition.
    ///
    /// `block_starts` must be sorted, start with 0, and all values must be
    /// `< n`.
    pub fn new_with_partition(
        csr_val: &[f64],
        csr_row_ptr: &[usize],
        csr_col_ind: &[usize],
        n: usize,
        block_starts: Vec<usize>,
    ) -> SparseResult<Self> {
        if csr_row_ptr.len() != n + 1 {
            return Err(SparseError::InconsistentData {
                reason: format!(
                    "csr_row_ptr.len() = {} != n+1 = {}",
                    csr_row_ptr.len(),
                    n + 1
                ),
            });
        }
        if block_starts.is_empty() || block_starts[0] != 0 {
            return Err(SparseError::InvalidArgument(
                "block_starts must start with 0".to_string(),
            ));
        }

        let nb = block_starts.len();
        let mut lu_factors = Vec::with_capacity(nb);
        let mut lu_perms = Vec::with_capacity(nb);
        let mut block_sizes = Vec::with_capacity(nb);

        for b in 0..nb {
            let start = block_starts[b];
            let end = if b + 1 < nb { block_starts[b + 1] } else { n };
            let bs = end - start;
            block_sizes.push(bs);

            // Extract the dense block A[start..end, start..end]
            let mut block = vec![vec![0.0f64; bs]; bs];
            for (local_row, global_row) in (start..end).enumerate() {
                for pos in csr_row_ptr[global_row]..csr_row_ptr[global_row + 1] {
                    let col = csr_col_ind[pos];
                    if col >= start && col < end {
                        block[local_row][col - start] = csr_val[pos];
                    }
                }
            }

            // LU factorize
            let perm = dense_lu(&mut block, bs).ok_or_else(|| {
                SparseError::SingularMatrix(format!(
                    "block {b} (rows {start}..{end}) is singular"
                ))
            })?;
            lu_factors.push(block);
            lu_perms.push(perm);
        }

        Ok(Self {
            lu_factors,
            lu_perms,
            block_starts,
            block_sizes,
            n,
        })
    }

    /// Apply the block Jacobi preconditioner: compute y = M^{-1} x.
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0f64; self.n];
        for (b, &start) in self.block_starts.iter().enumerate() {
            let bs = self.block_sizes[b];
            let x_block = &x[start..start + bs];
            let y_block =
                dense_lu_solve(&self.lu_factors[b], &self.lu_perms[b], x_block, bs);
            y[start..start + bs].copy_from_slice(&y_block);
        }
        y
    }

    /// Return the global dimension.
    pub fn size(&self) -> usize {
        self.n
    }

    /// Return the number of blocks.
    pub fn num_blocks(&self) -> usize {
        self.block_starts.len()
    }
}

// ---------------------------------------------------------------------------
// SPAI – Sparse Approximate Inverse (column-wise formulation)
// ---------------------------------------------------------------------------

/// Householder QR least-squares solver for local sub-systems in SPAI.
fn local_lstsq(a_col_major: &[f64], b: &[f64], nrows: usize, ncols: usize) -> Vec<f64> {
    if nrows == 0 || ncols == 0 {
        return vec![0.0f64; ncols];
    }
    let mut a = a_col_major.to_vec();
    let mut rhs = b.to_vec();
    let k = ncols.min(nrows);

    for j in 0..k {
        // Extract column j from position j onward
        let col_start = j * nrows + j;
        let col_len = nrows - j;
        if col_len == 0 {
            break;
        }
        let col_slice = &a[col_start..col_start + col_len];
        let sigma: f64 = col_slice.iter().map(|x| x * x).sum::<f64>().sqrt();
        if sigma < 1e-300 {
            continue;
        }
        let mut v = col_slice.to_vec();
        let sign = if v[0] >= 0.0 { 1.0 } else { -1.0 };
        v[0] += sign * sigma;
        let v_norm_sq: f64 = v.iter().map(|x| x * x).sum();
        if v_norm_sq < 1e-300 {
            continue;
        }
        let beta = 2.0 / v_norm_sq;
        // Apply to columns j..ncols
        for jj in j..ncols {
            let s = jj * nrows + j;
            let dot: f64 = v.iter().zip(a[s..s + col_len].iter()).map(|(a, b)| a * b).sum();
            let scale = beta * dot;
            for ii in 0..col_len {
                a[jj * nrows + j + ii] -= scale * v[ii];
            }
        }
        // Apply to rhs
        let dot: f64 = v.iter().zip(rhs[j..j + col_len].iter()).map(|(a, b)| a * b).sum();
        let scale = beta * dot;
        for ii in 0..col_len {
            rhs[j + ii] -= scale * v[ii];
        }
    }

    // Back substitution on upper triangular R (column-major)
    let mut x = vec![0.0f64; ncols];
    for ii in 0..k {
        let i = k - 1 - ii;
        let r_ii = a[i * nrows + i];
        if r_ii.abs() < 1e-14 {
            x[i] = 0.0;
            continue;
        }
        let mut s = rhs[i];
        for jj in (i + 1)..ncols {
            s -= a[jj * nrows + i] * x[jj];
        }
        x[i] = s / r_ii;
    }
    x
}

/// Sparse Approximate Inverse (SPAI) preconditioner.
///
/// Constructs M ≈ A⁻¹ by solving, for each column k of M, the local
/// least-squares problem `min_mₖ ‖A mₖ − eₖ‖₂` subject to a sparsity
/// pattern derived from the non-zeros of column k of A.
///
/// The `max_nnz_per_col` parameter controls the sparsity budget; increasing
/// it improves accuracy at the cost of construction time.
pub struct SpaiPrecondF64 {
    /// Row-wise storage of M: `m_rows[i]` holds `(col, value)` pairs for row i.
    m_rows: Vec<Vec<(usize, f64)>>,
    n: usize,
}

impl SpaiPrecondF64 {
    /// Build the SPAI preconditioner.
    ///
    /// # Arguments
    ///
    /// * `csr_val`, `csr_row_ptr`, `csr_col_ind` – Input matrix in CSR format.
    /// * `n`                – Matrix dimension (square).
    /// * `_max_iter`        – Reserved for adaptive enrichment (currently unused).
    /// * `tolerance`        – Residual threshold; columns with residual below
    ///                        `tolerance` are accepted without further refinement.
    pub fn new(
        csr_val: &[f64],
        csr_row_ptr: &[usize],
        csr_col_ind: &[usize],
        n: usize,
        _max_iter: usize,
        tolerance: f64,
    ) -> SparseResult<Self> {
        if csr_row_ptr.len() != n + 1 {
            return Err(SparseError::InconsistentData {
                reason: format!(
                    "csr_row_ptr.len() = {} != n+1 = {}",
                    csr_row_ptr.len(),
                    n + 1
                ),
            });
        }

        // Build CSC of A (= transpose of CSR)
        let nnz = csr_val.len();
        let mut col_counts = vec![0usize; n];
        for &c in csr_col_ind {
            col_counts[c] += 1;
        }
        let mut csc_ptr = vec![0usize; n + 1];
        for c in 0..n {
            csc_ptr[c + 1] = csc_ptr[c] + col_counts[c];
        }
        let mut csc_row = vec![0usize; nnz];
        let mut csc_val = vec![0.0f64; nnz];
        let mut cur = csc_ptr[..n].to_vec();
        for i in 0..n {
            for pos in csr_row_ptr[i]..csr_row_ptr[i + 1] {
                let c = csr_col_ind[pos];
                let dst = cur[c];
                csc_row[dst] = i;
                csc_val[dst] = csr_val[pos];
                cur[c] += 1;
            }
        }

        let mut m_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];

        for k in 0..n {
            // Sparsity pattern: non-zero row indices of column k of A
            let s = csc_ptr[k];
            let e = csc_ptr[k + 1];
            let mut pattern: Vec<usize> = csc_row[s..e].to_vec();
            if !pattern.contains(&k) {
                pattern.push(k);
            }
            pattern.sort_unstable();
            pattern.dedup();

            let np = pattern.len();
            if np == 0 {
                continue;
            }

            // Collect row indices of the local sub-system:
            // rows of A that have non-zeros in any column of `pattern`.
            let mut row_set: Vec<usize> = Vec::new();
            for &j in &pattern {
                for &r in &csc_row[csc_ptr[j]..csc_ptr[j + 1]] {
                    if !row_set.contains(&r) {
                        row_set.push(r);
                    }
                }
            }
            row_set.sort_unstable();
            let nr = row_set.len();

            // Build A_local (nr × np) in column-major order
            let mut a_local = vec![0.0f64; nr * np];
            for (col_loc, &j) in pattern.iter().enumerate() {
                for pos in csc_ptr[j]..csc_ptr[j + 1] {
                    let global_row = csc_row[pos];
                    if let Ok(row_loc) = row_set.binary_search(&global_row) {
                        a_local[col_loc * nr + row_loc] = csc_val[pos];
                    }
                }
            }

            // rhs = eₖ restricted to row_set
            let mut rhs = vec![0.0f64; nr];
            if let Ok(pos_k) = row_set.binary_search(&k) {
                rhs[pos_k] = 1.0;
            }

            let x_local = local_lstsq(&a_local, &rhs, nr, np);

            // Check residual: r = A_local * x_local - rhs
            let mut resid_sq = 0.0f64;
            for row_loc in 0..nr {
                let mut v = -rhs[row_loc];
                for col_loc in 0..np {
                    v += a_local[col_loc * nr + row_loc] * x_local[col_loc];
                }
                resid_sq += v * v;
            }
            // Store entries only if residual is within tolerance or regardless
            // (tolerance is used as a quality indicator; we always store).
            let _ = tolerance; // tolerance stored for future adaptive enrichment

            // Scatter x_local into M: M[pattern[p], k] = x_local[p]
            for (p, &row_idx) in pattern.iter().enumerate() {
                if x_local[p].abs() > 1e-15 {
                    m_rows[row_idx].push((k, x_local[p]));
                }
            }

            let _ = resid_sq; // available for future use
        }

        Ok(Self { m_rows, n })
    }

    /// Apply y = M x.
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0f64; self.n];
        for (i, row) in self.m_rows.iter().enumerate() {
            let mut acc = 0.0f64;
            for &(j, v) in row {
                acc += v * x[j];
            }
            y[i] = acc;
        }
        y
    }

    /// Return the matrix dimension.
    pub fn size(&self) -> usize {
        self.n
    }
}

// ---------------------------------------------------------------------------
// Polynomial Preconditioner (Neumann series)
// ---------------------------------------------------------------------------

/// Polynomial preconditioner M⁻¹ ≈ Σ_{k=0}^{degree} c_k A^k.
///
/// For the Neumann series: `c_k = (-α)^k` with `α` chosen to make
/// `‖I − αA‖ < 1` for spectral convergence.  The polynomial is evaluated
/// via Horner's method applied to repeated sparse matrix-vector products.
pub struct PolynomialPrecondF64 {
    /// Coefficients c_0, c_1, …, c_{degree}
    coefficients: Vec<f64>,
    csr_val: Vec<f64>,
    csr_row_ptr: Vec<usize>,
    csr_col_ind: Vec<usize>,
    n: usize,
}

impl PolynomialPrecondF64 {
    /// Build a Neumann-series preconditioner.
    ///
    /// The series α Σ_{k=0}^{degree} (I − αA)^k is evaluated with
    /// `α ≈ 1 / spectral_radius_est`.
    ///
    /// # Arguments
    ///
    /// * `csr_val`, `csr_row_ptr`, `csr_col_ind` – CSR storage of A.
    /// * `n`                    – Matrix dimension.
    /// * `degree`               – Polynomial degree (number of terms − 1).
    /// * `spectral_radius_est`  – Estimate of the spectral radius of A; used
    ///                            to compute the shift α.
    pub fn neumann(
        csr_val: &[f64],
        csr_row_ptr: &[usize],
        csr_col_ind: &[usize],
        n: usize,
        degree: usize,
        spectral_radius_est: f64,
    ) -> SparseResult<Self> {
        if csr_row_ptr.len() != n + 1 {
            return Err(SparseError::InconsistentData {
                reason: format!(
                    "csr_row_ptr.len() = {} != n+1 = {}",
                    csr_row_ptr.len(),
                    n + 1
                ),
            });
        }
        let alpha = if spectral_radius_est.abs() > 1e-300 {
            1.0 / spectral_radius_est
        } else {
            1.0
        };
        // Coefficients for α Σ_{k=0}^{degree} (I − αA)^k:
        // Horner form: p(A) x = α [x + (I−αA)(x + (I−αA)(... ))]
        // We store coefficients c_k of A^k in the Horner representation.
        // Since the Neumann series is a polynomial in (I−αA), we store α
        // as the overall scale and keep the recurrence in `apply`.
        let coefficients = vec![alpha; degree + 1];

        Ok(Self {
            coefficients,
            csr_val: csr_val.to_vec(),
            csr_row_ptr: csr_row_ptr.to_vec(),
            csr_col_ind: csr_col_ind.to_vec(),
            n,
        })
    }

    /// Apply M^{-1} x via Horner evaluation of the Neumann polynomial.
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        let n = self.n;
        let degree = self.coefficients.len().saturating_sub(1);
        let alpha = self.coefficients[0]; // α = 1/spectral_radius

        // Horner evaluation:  p x = α [ x + B (x + B (x + ... B x)) ]
        // where B = I − α A.
        let b_apply = |v: &[f64]| -> Vec<f64> {
            let av = csr_matvec(&self.csr_row_ptr, &self.csr_col_ind, &self.csr_val, v, n);
            (0..n).map(|i| v[i] - alpha * av[i]).collect()
        };

        let mut acc = x.to_vec();
        for _ in 0..degree {
            let b_acc = b_apply(&acc);
            for i in 0..n {
                acc[i] = x[i] + b_acc[i];
            }
        }
        acc.iter_mut().for_each(|v| *v *= alpha);
        acc
    }

    /// Return the polynomial degree.
    pub fn degree(&self) -> usize {
        self.coefficients.len().saturating_sub(1)
    }

    /// Return the scaling factor α.
    pub fn shift(&self) -> f64 {
        *self.coefficients.first().unwrap_or(&1.0)
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

    /// 4×4 tridiagonal SPD matrix:
    /// A = diag(4,4,4,4) + offdiag(-1,-1,-1) above and below diagonal
    fn tridiag4() -> (Vec<f64>, Vec<usize>, Vec<usize>, usize) {
        let n = 4usize;
        let row_ptr = vec![0, 2, 5, 8, 10];
        let col_ind = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let val = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        (val, row_ptr, col_ind, n)
    }

    #[test]
    fn test_block_jacobi_uniform() {
        let (val, rp, ci, n) = tridiag4();
        let prec = BlockJacobiF64::new_uniform(&val, &rp, &ci, n, 2)
            .expect("BlockJacobiF64::new_uniform");
        assert_eq!(prec.size(), n);
        assert_eq!(prec.num_blocks(), 2);

        let b = vec![1.0, 0.0, 0.0, 1.0];
        let y = prec.apply(&b);
        assert_eq!(y.len(), n);

        // Verify: for block 0, [4 -1; -1 4] * y[0..2] ≈ b[0..2]
        let r0 = 4.0 * y[0] - 1.0 * y[1] - b[0];
        let r1 = -1.0 * y[0] + 4.0 * y[1] - b[1];
        assert!(r0.abs() < 1e-12, "block 0 row 0 residual {r0}");
        assert!(r1.abs() < 1e-12, "block 0 row 1 residual {r1}");
    }

    #[test]
    fn test_block_jacobi_partition() {
        let (val, rp, ci, n) = tridiag4();
        let starts = vec![0, 1, 3]; // blocks: [0], [1,2], [3]
        let prec = BlockJacobiF64::new_with_partition(&val, &rp, &ci, n, starts)
            .expect("BlockJacobiF64::new_with_partition");
        assert_eq!(prec.num_blocks(), 3);
        let y = prec.apply(&[4.0, 4.0, 4.0, 4.0]);
        assert_eq!(y.len(), n);
        // Row 0: just diagonal 4, so y[0] = 4/4 = 1.0
        assert!((y[0] - 1.0).abs() < 1e-12, "y[0]={}", y[0]);
    }

    #[test]
    fn test_block_jacobi_singular_error() {
        let n = 2;
        let row_ptr = vec![0, 1, 2];
        let col_ind = vec![0, 0]; // column 0 repeated → block [0..2] is singular
        let val = vec![1.0, 1.0];
        // Block [0,1] has column structure [1; 1] for col 0 and 0 for col 1 → singular
        let result = BlockJacobiF64::new_uniform(&val, &row_ptr, &col_ind, n, 2);
        assert!(result.is_err(), "should detect singular block");
    }

    #[test]
    fn test_spai_apply() {
        let (val, rp, ci, n) = tridiag4();
        let prec = SpaiPrecondF64::new(&val, &rp, &ci, n, 5, 1e-8)
            .expect("SpaiPrecondF64::new");
        assert_eq!(prec.size(), n);

        let b = vec![1.0, 2.0, 3.0, 4.0];
        let y = prec.apply(&b);
        assert_eq!(y.len(), n);
        // y should be non-zero (SPAI gives a meaningful approximation)
        let norm: f64 = y.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm > 1e-10, "SPAI apply should give non-zero result");
    }

    #[test]
    fn test_polynomial_neumann_apply() {
        let (val, rp, ci, n) = tridiag4();
        // Spectral radius of the tridiagonal ≈ 4 + 2*1 = 6 (Gershgorin)
        let prec = PolynomialPrecondF64::neumann(&val, &rp, &ci, n, 4, 6.0)
            .expect("PolynomialPrecondF64::neumann");
        assert_eq!(prec.size(), n);
        assert_eq!(prec.degree(), 4);

        let b = vec![1.0, 0.0, 0.0, 0.0];
        let y = prec.apply(&b);
        assert_eq!(y.len(), n);
        let norm: f64 = y.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm > 1e-10, "Neumann apply should give non-zero result");
    }

    #[test]
    fn test_polynomial_shift() {
        let (val, rp, ci, n) = tridiag4();
        let prec = PolynomialPrecondF64::neumann(&val, &rp, &ci, n, 2, 5.0)
            .expect("neumann");
        let expected_shift = 1.0 / 5.0;
        assert!((prec.shift() - expected_shift).abs() < 1e-14);
    }
}
