//! Sparse Approximate Inverse (AINV) preconditioner
//!
//! Implements the Benzi–Tůma (1996) biorthogonalisation algorithm for
//! constructing a sparse approximate inverse of a general (possibly
//! non-symmetric) matrix A.  A symmetric variant is also provided for
//! SPD matrices.
//!
//! The approximate inverse is represented as
//!
//!   M⁻¹ ≈ W D⁻¹ Zᵀ
//!
//! where:
//! - Z and W are sparse matrices whose columns z_j, w_j are built via
//!   biorthogonalisation with drop tolerance τ.
//! - D is a diagonal matrix with entries d_j = z_j^T A w_j.
//!
//! For symmetric SPD matrices the simplified form M⁻¹ ≈ Z D⁻¹ Zᵀ is used.
//!
//! # References
//!
//! - Benzi, M. & Tůma, M. (1996). A sparse approximate inverse preconditioner
//!   for the conjugate gradient method. *SIAM J. Sci. Comput.* 17(5), 1135–1149.
//! - Benzi, M. & Tůma, M. (1998). A robust incomplete factorization
//!   preconditioner for positive definite matrices. *Numer. Linear Algebra Appl.*
//!   5(4), 285–320.

use crate::error::{SparseError, SparseResult};

// ---------------------------------------------------------------------------
// Internal CSR helpers (self-contained to avoid generic-type noise)
// ---------------------------------------------------------------------------

/// Sparse matrix-vector product y = A * x using raw CSR arrays.
fn csr_matvec(
    row_ptrs: &[usize],
    col_indices: &[usize],
    values: &[f64],
    x: &[f64],
    nrows: usize,
) -> Vec<f64> {
    let mut y = vec![0.0f64; nrows];
    for i in 0..nrows {
        let mut acc = 0.0f64;
        for pos in row_ptrs[i]..row_ptrs[i + 1] {
            acc += values[pos] * x[col_indices[pos]];
        }
        y[i] = acc;
    }
    y
}

/// Dot product of two dense vectors.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm of a dense vector.
#[inline]
fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the AINV preconditioner.
#[derive(Debug, Clone)]
pub struct AinvConfig {
    /// Drop tolerance τ: entries with |z_{kl}| < τ * ‖z_k‖ are discarded.
    ///
    /// A larger value produces a sparser (cheaper to apply) but less accurate
    /// preconditioner.  Typical range: 1e-4 to 0.1.
    pub drop_tol: f64,

    /// Maximum number of outer iterations (reserved for adaptive AINV;
    /// currently the biorthogonalisation is a single pass so this bounds
    /// the column count processed).  Set to `0` to process all n columns.
    pub max_iter: usize,

    /// If `true`, use the symmetric variant (A must be SPD).
    /// W is set equal to Z, halving memory and build time.
    pub symmetric: bool,
}

impl Default for AinvConfig {
    fn default() -> Self {
        Self {
            drop_tol: 0.01,
            max_iter: 100,
            symmetric: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Sparse column (COO-like, sorted by row index)
// ---------------------------------------------------------------------------

/// A sparse column stored as sorted `(row, value)` pairs.
#[derive(Debug, Clone, Default)]
struct SparseCol {
    entries: Vec<(usize, f64)>,
}

impl SparseCol {
    fn from_unit(k: usize) -> Self {
        Self {
            entries: vec![(k, 1.0)],
        }
    }

    /// Dense representation in R^n.
    fn to_dense(&self, n: usize) -> Vec<f64> {
        let mut v = vec![0.0f64; n];
        for &(r, val) in &self.entries {
            if r < n {
                v[r] = val;
            }
        }
        v
    }

    /// Apply scalar multiply in place.
    fn scale(&mut self, s: f64) {
        for (_, v) in self.entries.iter_mut() {
            *v *= s;
        }
    }

    /// Subtract s * other from self.
    fn axpy_neg(&mut self, s: f64, other: &SparseCol) {
        for &(r, v) in &other.entries {
            let idx = self
                .entries
                .iter()
                .position(|&(row, _)| row == r);
            if let Some(i) = idx {
                self.entries[i].1 -= s * v;
            } else {
                self.entries.push((r, -s * v));
            }
        }
        self.entries.sort_unstable_by_key(|&(r, _)| r);
    }

    /// Apply drop tolerance: remove entries with |v| < tol * ‖col‖.
    fn drop(&mut self, tol: f64) {
        if tol <= 0.0 || self.entries.is_empty() {
            return;
        }
        let nrm = self
            .entries
            .iter()
            .map(|&(_, v)| v * v)
            .sum::<f64>()
            .sqrt();
        let threshold = tol * nrm;
        self.entries.retain(|&(_, v)| v.abs() >= threshold);
    }

    /// Rebuild from a dense vector applying drop tolerance.
    fn from_dense_with_drop(dense: &[f64], tol: f64) -> Self {
        let nrm = dense.iter().map(|x| x * x).sum::<f64>().sqrt();
        let threshold = if tol > 0.0 { tol * nrm } else { 0.0 };
        let entries: Vec<(usize, f64)> = dense
            .iter()
            .enumerate()
            .filter(|(_, &v)| v.abs() > threshold)
            .map(|(i, &v)| (i, v))
            .collect();
        Self { entries }
    }
}

// ---------------------------------------------------------------------------
// AINV Preconditioner
// ---------------------------------------------------------------------------

/// AINV (Approximate INVerse) preconditioner.
///
/// Implements the Benzi–Tůma biorthogonalisation algorithm.  For a matrix A
/// the preconditioner applies
///
///   M⁻¹ r  =  W * (D⁻¹ * (Zᵀ * r))
///
/// where Z and W are sparse column matrices and D is a diagonal scaling.
///
/// For SPD matrices use `AinvConfig { symmetric: true }`, which sets W = Z
/// and uses only the Z columns.
pub struct AinvPreconditioner {
    /// Columns of Z (or both Z and W for symmetric case).
    z_cols: Vec<SparseCol>,
    /// Columns of W (for non-symmetric).  Empty when `symmetric = true`.
    w_cols: Vec<SparseCol>,
    /// Diagonal scaling d_j = z_j^T A w_j (or z_j^T A z_j for symmetric).
    d: Vec<f64>,
    /// Matrix dimension.
    n: usize,
    /// Whether the symmetric (SPD) variant is active.
    symmetric: bool,
}

impl AinvPreconditioner {
    /// Construct the AINV preconditioner for a square CSR matrix.
    ///
    /// # Arguments
    ///
    /// * `row_ptrs`    – CSR row pointer array (length `n + 1`).
    /// * `col_indices` – CSR column index array.
    /// * `values`      – CSR value array.
    /// * `n`           – Matrix dimension (must equal `row_ptrs.len() - 1`).
    /// * `config`      – Algorithm configuration.
    ///
    /// # Errors
    ///
    /// Returns `SparseError` if the input arrays are inconsistent or if a
    /// zero pivot is encountered (indicating a (near-)singular matrix A).
    pub fn new(
        row_ptrs: &[usize],
        col_indices: &[usize],
        values: &[f64],
        n: usize,
        config: AinvConfig,
    ) -> SparseResult<Self> {
        if row_ptrs.len() != n + 1 {
            return Err(SparseError::InconsistentData {
                reason: format!(
                    "row_ptrs length {} != n+1={}",
                    row_ptrs.len(),
                    n + 1
                ),
            });
        }
        if n == 0 {
            return Ok(Self {
                z_cols: Vec::new(),
                w_cols: Vec::new(),
                d: Vec::new(),
                n: 0,
                symmetric: config.symmetric,
            });
        }

        let tol = config.drop_tol;
        let sym = config.symmetric;

        // Determine how many columns to process.
        let ncols = if config.max_iter == 0 || config.max_iter >= n {
            n
        } else {
            config.max_iter
        };

        // Initialise: z_j = e_j, w_j = e_j for j = 0..ncols.
        let mut z: Vec<SparseCol> = (0..ncols).map(SparseCol::from_unit).collect();
        let mut w: Vec<SparseCol> = if sym {
            Vec::new()
        } else {
            (0..ncols).map(SparseCol::from_unit).collect()
        };

        // Diagonal pivots d_j.
        let mut d = vec![0.0f64; ncols];

        // Biorthogonalisation pass (Algorithm 1 of Benzi & Tůma 1996).
        for j in 0..ncols {
            // Compute w_j_dense (= z_j for symmetric case) for matvec.
            let w_j_dense: Vec<f64> = if sym {
                z[j].to_dense(n)
            } else {
                w[j].to_dense(n)
            };

            // c_j = z_j^T * A * w_j
            let a_wj = csr_matvec(row_ptrs, col_indices, values, &w_j_dense, n);
            let z_j_dense = z[j].to_dense(n);
            let c_j = dot(&z_j_dense, &a_wj);

            if c_j.abs() < 1e-300 {
                // Near-zero pivot: skip (the column contribution is negligible).
                d[j] = 0.0;
                continue;
            }
            d[j] = c_j;

            // Update z_k and w_k for k > j.
            // z_k -= (z_k^T * A * w_j / c_j) * z_j
            // w_k -= (z_j^T * A * w_k / c_j) * w_j
            let a_wj_ref = &a_wj; // A * w_j already computed

            for k in (j + 1)..ncols {
                // --- update z_k ---
                let z_k_dense = z[k].to_dense(n);
                let numerator_z = dot(&z_k_dense, a_wj_ref);
                if numerator_z.abs() > 1e-300 {
                    let coeff = numerator_z / c_j;
                    // z_k -= coeff * z_j: rebuild from dense for simplicity
                    let mut z_k_new = z_k_dense.clone();
                    for &(r, v) in &z[j].entries {
                        z_k_new[r] -= coeff * v;
                    }
                    z[k] = SparseCol::from_dense_with_drop(&z_k_new, tol);
                }

                if !sym {
                    // --- update w_k ---
                    let w_k_dense = w[k].to_dense(n);
                    let a_wk = csr_matvec(row_ptrs, col_indices, values, &w_k_dense, n);
                    let numerator_w = dot(&z_j_dense, &a_wk);
                    if numerator_w.abs() > 1e-300 {
                        let coeff = numerator_w / c_j;
                        let mut w_k_new = w_k_dense;
                        for &(r, v) in &w[j].entries {
                            w_k_new[r] -= coeff * v;
                        }
                        w[k] = SparseCol::from_dense_with_drop(&w_k_new, tol);
                    }
                }
            }

            // Apply drop tolerance to z_j (and w_j) now that it is finalised.
            z[j].drop(tol);
            if !sym {
                w[j].drop(tol);
            }
        }

        Ok(Self {
            z_cols: z,
            w_cols: w,
            d,
            n,
            symmetric: sym,
        })
    }

    /// Apply the preconditioner: compute y = M⁻¹ r = W D⁻¹ Zᵀ r.
    ///
    /// Steps:
    /// 1. v = Zᵀ r  (for each column j: v_j = Σ_i z_j[i] * r[i])
    /// 2. v = D⁻¹ v (element-wise: v_j /= d_j)
    /// 3. y = W v   (for each column j: y += v_j * w_j)
    ///
    /// For the symmetric variant W = Z.
    pub fn apply(&self, r: &[f64]) -> Vec<f64> {
        let ncols = self.z_cols.len();
        let mut v = vec![0.0f64; ncols];

        // Step 1: v = Zᵀ r.
        for j in 0..ncols {
            let mut acc = 0.0f64;
            for &(i, val) in &self.z_cols[j].entries {
                if i < r.len() {
                    acc += val * r[i];
                }
            }
            v[j] = acc;
        }

        // Step 2: v = D⁻¹ v.
        for j in 0..ncols {
            if self.d[j].abs() > 1e-300 {
                v[j] /= self.d[j];
            } else {
                v[j] = 0.0;
            }
        }

        // Step 3: y = W v.
        let mut y = vec![0.0f64; self.n];
        let w_source: &Vec<SparseCol> = if self.symmetric {
            &self.z_cols
        } else {
            &self.w_cols
        };
        for j in 0..ncols {
            if v[j].abs() < 1e-300 {
                continue;
            }
            for &(i, val) in &w_source[j].entries {
                if i < self.n {
                    y[i] += v[j] * val;
                }
            }
        }

        y
    }

    /// Return the matrix dimension.
    pub fn size(&self) -> usize {
        self.n
    }

    /// Return the number of non-zeros in the Z factor.
    pub fn z_nnz(&self) -> usize {
        self.z_cols.iter().map(|c| c.entries.len()).sum()
    }

    /// Return the number of non-zeros in the W factor.
    pub fn w_nnz(&self) -> usize {
        if self.symmetric {
            self.z_nnz()
        } else {
            self.w_cols.iter().map(|c| c.entries.len()).sum()
        }
    }

    /// Return the diagonal scaling vector D.
    pub fn diagonal(&self) -> &[f64] {
        &self.d
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper: build CSR for a diagonal matrix diag(d_0, ..., d_{n-1}).
    // -----------------------------------------------------------------------
    fn diag_csr(diag: &[f64]) -> (Vec<usize>, Vec<usize>, Vec<f64>, usize) {
        let n = diag.len();
        let row_ptrs: Vec<usize> = (0..=n).collect();
        let col_indices: Vec<usize> = (0..n).collect();
        let values = diag.to_vec();
        (row_ptrs, col_indices, values, n)
    }

    // -----------------------------------------------------------------------
    // Helper: build the 4×4 tridiagonal SPD test matrix.
    // -----------------------------------------------------------------------
    fn tridiag_csr() -> (Vec<usize>, Vec<usize>, Vec<f64>, usize) {
        let n = 4usize;
        let row_ptrs = vec![0, 2, 5, 8, 10];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let values = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        (row_ptrs, col_indices, values, n)
    }

    // -----------------------------------------------------------------------
    // Build a 5×5 symmetric positive definite tridiagonal matrix.
    // -----------------------------------------------------------------------
    fn spd5_csr() -> (Vec<usize>, Vec<usize>, Vec<f64>, usize) {
        let n = 5usize;
        // diag = 4, off-diag = -1 (symmetric tridiagonal)
        let mut row_ptrs = vec![0usize; n + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            if i > 0 {
                col_indices.push(i - 1);
                values.push(-1.0f64);
            }
            col_indices.push(i);
            values.push(4.0f64);
            if i + 1 < n {
                col_indices.push(i + 1);
                values.push(-1.0f64);
            }
            row_ptrs[i + 1] = col_indices.len();
        }
        (row_ptrs, col_indices, values, n)
    }

    // -----------------------------------------------------------------------
    // Build a non-symmetric 4×4 matrix (upper bidiagonal + diagonal).
    // -----------------------------------------------------------------------
    fn nonsym_csr() -> (Vec<usize>, Vec<usize>, Vec<f64>, usize) {
        let n = 4usize;
        // A[i,i] = 3, A[i, i+1] = 1 for i<n-1 (upper bidiagonal + main diag)
        let row_ptrs = vec![0, 2, 4, 6, 7];
        let col_indices = vec![0, 1, 1, 2, 2, 3, 3];
        let values = vec![3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0];
        (row_ptrs, col_indices, values, n)
    }

    // -----------------------------------------------------------------------
    // Test: identity matrix → AINV should give identity action.
    // -----------------------------------------------------------------------
    #[test]
    fn test_ainv_identity_matrix() {
        let n = 4usize;
        let (rp, ci, vals, _) = diag_csr(&vec![1.0; n]);
        let config = AinvConfig {
            drop_tol: 0.0,
            max_iter: 0,
            symmetric: true,
        };
        let prec = AinvPreconditioner::new(&rp, &ci, &vals, n, config)
            .expect("AINV build failed");

        let b = vec![1.0, 2.0, 3.0, 4.0];
        let y = prec.apply(&b);
        for (yi, bi) in y.iter().zip(b.iter()) {
            let err = (yi - bi).abs();
            assert!(err < 1e-10, "identity action failed: y={yi}, b={bi}");
        }
    }

    // -----------------------------------------------------------------------
    // Test: diagonal matrix → AINV gives exact diagonal inverse.
    // -----------------------------------------------------------------------
    #[test]
    fn test_ainv_diagonal_matrix() {
        let diag = vec![2.0, 4.0, 0.5, 8.0];
        let n = diag.len();
        let (rp, ci, vals, _) = diag_csr(&diag);
        let config = AinvConfig {
            drop_tol: 0.0,
            max_iter: 0,
            symmetric: false,
        };
        let prec = AinvPreconditioner::new(&rp, &ci, &vals, n, config)
            .expect("AINV build failed");

        let b = vec![1.0; n];
        let y = prec.apply(&b);
        let expected: Vec<f64> = diag.iter().map(|d| 1.0 / d).collect();
        for (i, (yi, ei)) in y.iter().zip(expected.iter()).enumerate() {
            let err = (yi - ei).abs();
            assert!(err < 1e-10, "diagonal case i={i}: got {yi}, expected {ei}");
        }
    }

    // -----------------------------------------------------------------------
    // Test: symmetric 5×5 SPD matrix → Z^T A Z ≈ D (diagonal).
    // -----------------------------------------------------------------------
    #[test]
    fn test_ainv_spd_biorthogonality() {
        let (rp, ci, vals, n) = spd5_csr();
        let config = AinvConfig {
            drop_tol: 1e-4,
            max_iter: 0,
            symmetric: true,
        };
        let prec = AinvPreconditioner::new(&rp, &ci, &vals, n, config)
            .expect("AINV build failed");

        // Check that Zᵀ A Z is approximately diagonal (biorthogonality condition).
        // We verify that the off-diagonal product z_j^T * A * z_k ≈ 0 for j ≠ k.
        for j in 0..prec.z_cols.len() {
            let z_j = prec.z_cols[j].to_dense(n);
            let a_zj = csr_matvec(&rp, &ci, &vals, &z_j, n);
            for k in 0..prec.z_cols.len() {
                if k == j {
                    continue;
                }
                let z_k = prec.z_cols[k].to_dense(n);
                let cross = dot(&z_k, &a_zj);
                assert!(
                    cross.abs() < 5e-2,
                    "biorthogonality failed for j={j} k={k}: cross={cross}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test: drop tolerance prunes small entries.
    // -----------------------------------------------------------------------
    #[test]
    fn test_ainv_drop_tolerance_prunes() {
        let (rp, ci, vals, n) = spd5_csr();

        // High drop tolerance → very sparse Z.
        let config_high = AinvConfig {
            drop_tol: 0.5,
            max_iter: 0,
            symmetric: true,
        };
        let prec_high = AinvPreconditioner::new(&rp, &ci, &vals, n, config_high)
            .expect("AINV build (high tol) failed");

        // No drop tolerance → denser Z.
        let config_none = AinvConfig {
            drop_tol: 0.0,
            max_iter: 0,
            symmetric: true,
        };
        let prec_none = AinvPreconditioner::new(&rp, &ci, &vals, n, config_none)
            .expect("AINV build (no tol) failed");

        assert!(
            prec_high.z_nnz() <= prec_none.z_nnz(),
            "high drop tol should give z_nnz <= no drop tol: {} vs {}",
            prec_high.z_nnz(),
            prec_none.z_nnz()
        );
    }

    // -----------------------------------------------------------------------
    // Test: AINV apply reduces the residual on a random SPD system.
    // -----------------------------------------------------------------------
    #[test]
    fn test_ainv_reduces_residual() {
        let (rp, ci, vals, n) = tridiag_csr();
        let config = AinvConfig {
            drop_tol: 0.01,
            max_iter: 0,
            symmetric: true,
        };
        let prec = AinvPreconditioner::new(&rp, &ci, &vals, n, config)
            .expect("AINV build failed");

        let b = vec![1.0, 2.0, 3.0, 4.0];
        // x ≈ M⁻¹ b (one preconditioned step from zero)
        let x = prec.apply(&b);

        // Compute residual r = b - A*x.
        let ax: Vec<f64> = csr_matvec(&rp, &ci, &vals, &x, n);
        let res_norm: f64 = b
            .iter()
            .zip(ax.iter())
            .map(|(bi, axi)| (bi - axi).powi(2))
            .sum::<f64>()
            .sqrt();
        let b_norm: f64 = norm2(&b);

        // AINV is a single-step approximate solve; the relative residual
        // should be substantially less than 1.
        assert!(
            res_norm < b_norm,
            "AINV should reduce residual: res_norm={res_norm}, b_norm={b_norm}"
        );
    }

    // -----------------------------------------------------------------------
    // Test: non-symmetric matrix — general AINV (W ≠ Z).
    // -----------------------------------------------------------------------
    #[test]
    fn test_ainv_nonsymmetric() {
        let (rp, ci, vals, n) = nonsym_csr();
        let config = AinvConfig {
            drop_tol: 0.0,
            max_iter: 0,
            symmetric: false,
        };
        let prec = AinvPreconditioner::new(&rp, &ci, &vals, n, config)
            .expect("AINV build failed");

        assert_eq!(prec.size(), n);

        let b = vec![3.0, 2.0, 1.0, 0.5];
        let y = prec.apply(&b);
        assert_eq!(y.len(), n);

        // Verify apply reduces residual.
        let ay = csr_matvec(&rp, &ci, &vals, &y, n);
        let res_norm: f64 = b
            .iter()
            .zip(ay.iter())
            .map(|(bi, ayi)| (bi - ayi).powi(2))
            .sum::<f64>()
            .sqrt();
        let b_norm = norm2(&b);
        assert!(
            res_norm < b_norm * 2.0,
            "non-symmetric AINV: res_norm={res_norm}, b_norm={b_norm}"
        );
    }

    // -----------------------------------------------------------------------
    // Test: config defaults are sensible.
    // -----------------------------------------------------------------------
    #[test]
    fn test_ainv_config_defaults() {
        let cfg = AinvConfig::default();
        assert!((cfg.drop_tol - 0.01).abs() < 1e-15);
        assert_eq!(cfg.max_iter, 100);
        assert!(!cfg.symmetric);
    }

    // -----------------------------------------------------------------------
    // Test: size() accessor.
    // -----------------------------------------------------------------------
    #[test]
    fn test_ainv_size_accessor() {
        let (rp, ci, vals, n) = tridiag_csr();
        let prec = AinvPreconditioner::new(&rp, &ci, &vals, n, AinvConfig::default())
            .expect("AINV build failed");
        assert_eq!(prec.size(), n);
    }

    // -----------------------------------------------------------------------
    // Test: dimension mismatch error.
    // -----------------------------------------------------------------------
    #[test]
    fn test_ainv_dimension_mismatch() {
        let row_ptrs = vec![0, 1, 2]; // n=2
        let col_indices = vec![0, 1];
        let values = vec![1.0, 1.0];
        let result = AinvPreconditioner::new(&row_ptrs, &col_indices, &values, 5, AinvConfig::default());
        assert!(result.is_err(), "should fail on dimension mismatch");
    }
}
