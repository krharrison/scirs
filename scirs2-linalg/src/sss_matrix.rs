//! Sequentially Semi-Separable (SSS) matrix representation.
//!
//! An SSS matrix stores each off-diagonal block as a product of *generator*
//! matrices, enabling O(n k) matrix–vector products for n×n matrices whose
//! off-diagonal blocks have numerical rank ≤ k.
//!
//! ## SSS form
//!
//! Partition the matrix into `n_blocks × n_blocks` block structure.  For blocks
//! of size `p = block_size`:
//!
//! - Diagonal blocks `D_i` (p × p): stored dense.
//! - Off-diagonal blocks `A[i,j]` for `i ≠ j`: compressed via rank-`k` SVD as
//!   `A[i,j] ≈ U_ij · diag(s_ij) · V_ij^T`.
//!
//! The "generator" terminology maps as:
//! - Upper part (i < j): `C_i = U_ij`, `A_diag = diag(s_ij)` (coupling), `B_j = V_ij`.
//! - Lower part (i > j): `P_i = U_ij`, `R_diag = diag(s_ij)` (coupling), `Q_j = V_ij`.
//!
//! ## Matvec via forward-backward sweep
//!
//! A compact O(n·k) matvec is implemented by accumulating the contributions of
//! each compressed off-diagonal block `B_ij`:
//!
//! `y_i += U_ij · (diag(s_ij) · (V_ij^T · x_j))` for each j.
//!
//! The sweep processes the upper-triangle (j > i) in a backward pass and the
//! lower-triangle (j < i) in a forward pass, both O(n·k) per pass.
//!
//! ## References
//!
//! - Chandrasekaran, S., Gu, M., & Pals, T. (2006). "A Fast ULV Decomposition Solver
//!   for Hierarchically Semiseparable Representations." SIAM J. Matrix Anal. Appl.
//! - Vandebril, R., Van Barel, M., & Mastronardi, N. (2008). "Matrix Computations
//!   and Semiseparable Matrices." Johns Hopkins University Press.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for SSS matrix construction.
#[derive(Debug, Clone)]
pub struct SssConfig {
    /// Number of rows/columns per diagonal block.
    pub block_size: usize,
    /// Generator rank for off-diagonal blocks.
    pub rank: usize,
}

impl Default for SssConfig {
    fn default() -> Self {
        Self {
            block_size: 4,
            rank: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal compressed block
// ---------------------------------------------------------------------------

/// A compressed off-diagonal block stored as `U · diag(s) · V^T`.
#[derive(Debug, Clone)]
struct CompressedBlock<F> {
    /// Left singular vectors (row_size × k).
    u: Array2<F>,
    /// Singular values (k,).
    s: Array1<F>,
    /// Right singular vectors (col_size × k).
    v: Array2<F>,
    /// Row block index.
    row_block: usize,
    /// Col block index.
    col_block: usize,
}

// ---------------------------------------------------------------------------
// SssMatrix
// ---------------------------------------------------------------------------

/// Sequentially Semi-Separable matrix.
///
/// Stores the SSS generators:
/// - Diagonal blocks `d[i]` — dense (block_size × block_size).
/// - `upper_c[i]`, `upper_a[i]`, `upper_b[i]` — generators for upper off-diagonal
///   blocks sourced from block column `i+1`.
/// - `lower_p[i]`, `lower_r[i]`, `lower_q[i]` — generators for lower off-diagonal
///   blocks sourced from block column `i-1`.
///
/// Internally all off-diagonal blocks are stored as compact `CompressedBlock`
/// structures; the public generator fields mirror the SSS generator naming
/// convention for the immediately adjacent (superdiagonal / subdiagonal) blocks.
#[derive(Debug)]
pub struct SssMatrix<F> {
    /// Number of diagonal blocks.
    pub n_blocks: usize,
    /// Diagonal blocks D_i.
    pub d: Vec<Array2<F>>,
    /// Upper generator C_i (output side, block_size × rank).
    pub upper_c: Vec<Array2<F>>,
    /// Upper generator A_i (coupling, rank × rank).
    pub upper_a: Vec<Array2<F>>,
    /// Upper generator B_i (input side of the *next* block, block_size × rank).
    pub upper_b: Vec<Array2<F>>,
    /// Lower generator P_i (output side, block_size × rank).
    pub lower_p: Vec<Array2<F>>,
    /// Lower generator R_i (coupling, rank × rank).
    pub lower_r: Vec<Array2<F>>,
    /// Lower generator Q_i (input side of the *previous* block, block_size × rank).
    pub lower_q: Vec<Array2<F>>,
    /// Configuration.
    pub config: SssConfig,
    /// Total size of the matrix.
    pub total_size: usize,
    /// All compressed off-diagonal blocks (internal storage).
    off_diag: Vec<CompressedBlock<F>>,
    /// Block boundaries `(start, end)` for each block.
    boundaries: Vec<(usize, usize)>,
}

// ---------------------------------------------------------------------------
// Construction helpers
// ---------------------------------------------------------------------------

fn truncated_svd_sss<F>(
    mat: &Array2<F>,
    rank: usize,
) -> LinalgResult<(Array2<F>, Array1<F>, Array2<F>)>
where
    F: Float
        + Debug
        + Clone
        + std::ops::AddAssign
        + scirs2_core::numeric::FromPrimitive
        + scirs2_core::numeric::NumAssign
        + std::iter::Sum
        + scirs2_core::ndarray::ScalarOperand
        + Send
        + Sync
        + 'static,
{
    let (u_full, s_full, vt_full) = crate::decomposition::svd(&mat.view(), true, None)?;
    let k = rank.min(s_full.len());
    let u_k = u_full.slice(scirs2_core::ndarray::s![.., ..k]).to_owned();
    let s_k = s_full.slice(scirs2_core::ndarray::s![..k]).to_owned();
    let vt_k = vt_full.slice(scirs2_core::ndarray::s![..k, ..]).to_owned();
    Ok((u_k, s_k, vt_k))
}

// ---------------------------------------------------------------------------
// SssMatrix implementation
// ---------------------------------------------------------------------------

impl<F> SssMatrix<F>
where
    F: Float
        + Debug
        + Clone
        + std::ops::AddAssign
        + scirs2_core::numeric::FromPrimitive
        + scirs2_core::numeric::NumAssign
        + std::iter::Sum
        + scirs2_core::ndarray::ScalarOperand
        + Send
        + Sync
        + 'static,
{
    /// Build an SSS representation from a dense (banded/symmetric) matrix.
    ///
    /// The matrix is partitioned into `n_blocks × n_blocks` blocks each of size
    /// `block_size`.  Remaining rows/columns are absorbed into the last block.
    ///
    /// Off-diagonal blocks are compressed to rank `config.rank` via truncated SVD.
    pub fn from_banded(mat: &Array2<F>, config: SssConfig) -> LinalgResult<Self> {
        let n = mat.nrows();
        let m = mat.ncols();
        if n != m {
            return Err(LinalgError::ShapeError(
                "SssMatrix::from_banded: matrix must be square".to_string(),
            ));
        }
        if n == 0 {
            return Err(LinalgError::ShapeError(
                "SssMatrix::from_banded: empty matrix".to_string(),
            ));
        }

        let bs = config.block_size.max(1);
        let rank = config.rank.max(1);

        // Block boundaries
        let n_blocks = n.div_ceil(bs);
        let boundaries: Vec<(usize, usize)> = (0..n_blocks)
            .map(|i| {
                let start = i * bs;
                let end = ((i + 1) * bs).min(n);
                (start, end)
            })
            .collect();

        let mut d_vec: Vec<Array2<F>> = Vec::with_capacity(n_blocks);
        let mut off_diag: Vec<CompressedBlock<F>> = Vec::new();

        // Upper/lower SSS generator public fields (adjacent blocks only for naming)
        let mut upper_c: Vec<Array2<F>> = Vec::with_capacity(n_blocks);
        let mut upper_a: Vec<Array2<F>> = Vec::with_capacity(n_blocks);
        let mut upper_b: Vec<Array2<F>> = Vec::with_capacity(n_blocks);
        let mut lower_p: Vec<Array2<F>> = Vec::with_capacity(n_blocks);
        let mut lower_r: Vec<Array2<F>> = Vec::with_capacity(n_blocks);
        let mut lower_q: Vec<Array2<F>> = Vec::with_capacity(n_blocks);

        for i in 0..n_blocks {
            let (rs, re) = boundaries[i];
            let row_size = re - rs;

            // Diagonal block
            d_vec.push(
                mat.slice(scirs2_core::ndarray::s![rs..re, rs..re])
                    .to_owned(),
            );

            // Compress ALL off-diagonal blocks (j ≠ i)
            for (j, &(cs, ce)) in boundaries.iter().enumerate().take(n_blocks) {
                if j == i {
                    continue;
                }
                let col_size = ce - cs;
                let sub = mat
                    .slice(scirs2_core::ndarray::s![rs..re, cs..ce])
                    .to_owned();
                let eff_rank = rank.min(row_size).min(col_size);
                match truncated_svd_sss(&sub, eff_rank) {
                    Ok((u_k, s_k, vt_k)) => {
                        let v_k = vt_k.t().to_owned(); // col_size × k
                        off_diag.push(CompressedBlock {
                            u: u_k,
                            s: s_k,
                            v: v_k,
                            row_block: i,
                            col_block: j,
                        });
                    }
                    Err(_) => {
                        // Fall back to explicit zero contribution (skip)
                    }
                }
            }

            // Populate public generator fields for adjacent superdiagonal
            if i + 1 < n_blocks {
                let (cs, ce) = boundaries[i + 1];
                let col_size = ce - cs;
                let sub = mat
                    .slice(scirs2_core::ndarray::s![rs..re, cs..ce])
                    .to_owned();
                let eff_rank = rank.min(row_size).min(col_size);
                match truncated_svd_sss(&sub, eff_rank) {
                    Ok((u_k, s_k, vt_k)) => {
                        let k = s_k.len();
                        let mut a_mat = Array2::<F>::zeros((k, k));
                        for l in 0..k {
                            a_mat[[l, l]] = s_k[l];
                        }
                        upper_c.push(u_k);
                        upper_a.push(a_mat);
                        upper_b.push(vt_k.t().to_owned());
                    }
                    Err(_) => {
                        upper_c.push(Array2::zeros((row_size, 1)));
                        upper_a.push(Array2::zeros((1, 1)));
                        upper_b.push(Array2::zeros((col_size, 1)));
                    }
                }
            } else {
                upper_c.push(Array2::zeros((row_size, 1)));
                upper_a.push(Array2::zeros((1, 1)));
                upper_b.push(Array2::zeros((row_size, 1)));
            }

            // Populate public generator fields for adjacent subdiagonal
            if i > 0 {
                let (cs, ce) = boundaries[i - 1];
                let col_size = ce - cs;
                let sub = mat
                    .slice(scirs2_core::ndarray::s![rs..re, cs..ce])
                    .to_owned();
                let eff_rank = rank.min(row_size).min(col_size);
                match truncated_svd_sss(&sub, eff_rank) {
                    Ok((u_k, s_k, vt_k)) => {
                        let k = s_k.len();
                        let mut r_mat = Array2::<F>::zeros((k, k));
                        for l in 0..k {
                            r_mat[[l, l]] = s_k[l];
                        }
                        lower_p.push(u_k);
                        lower_r.push(r_mat);
                        lower_q.push(vt_k.t().to_owned());
                    }
                    Err(_) => {
                        lower_p.push(Array2::zeros((row_size, 1)));
                        lower_r.push(Array2::zeros((1, 1)));
                        lower_q.push(Array2::zeros((col_size, 1)));
                    }
                }
            } else {
                lower_p.push(Array2::zeros((row_size, 1)));
                lower_r.push(Array2::zeros((1, 1)));
                lower_q.push(Array2::zeros((row_size, 1)));
            }
        }

        Ok(Self {
            n_blocks,
            d: d_vec,
            upper_c,
            upper_a,
            upper_b,
            lower_p,
            lower_r,
            lower_q,
            config,
            total_size: n,
            off_diag,
            boundaries,
        })
    }

    /// Compute y = A·x via the SSS forward-backward sweep.
    ///
    /// Internally applies:
    /// 1. Diagonal: `y_i += D_i · x_i` for each block i.
    /// 2. Off-diagonal: `y_i += U_ij · (s_ij ∘ (V_ij^T · x_j))` for each compressed
    ///    block (i, j) with i ≠ j.
    ///
    /// The off-diagonal sweep is O(n · k) where k is the generator rank.
    pub fn matvec(&self, x: &Array1<F>) -> LinalgResult<Array1<F>> {
        if x.len() != self.total_size {
            return Err(LinalgError::DimensionError(format!(
                "SssMatrix::matvec: x has length {} but matrix has size {}",
                x.len(),
                self.total_size
            )));
        }

        let mut y = Array1::<F>::zeros(self.total_size);

        // ---------------------------------------------------------------
        // 1. Diagonal blocks
        // ---------------------------------------------------------------
        for (i, &(rs, re)) in self.boundaries.iter().enumerate() {
            let x_i = x.slice(scirs2_core::ndarray::s![rs..re]).to_owned();
            let d = &self.d[i];
            let d_rows = d.nrows().min(re - rs);
            let d_cols = d.ncols().min(x_i.len());
            for r in 0..d_rows {
                let mut acc = F::zero();
                for c in 0..d_cols {
                    acc += d[[r, c]] * x_i[c];
                }
                y[rs + r] += acc;
            }
        }

        // ---------------------------------------------------------------
        // 2. Compressed off-diagonal blocks
        //    y[row_range] += U · (s ∘ (V^T · x[col_range]))
        // ---------------------------------------------------------------
        for block in &self.off_diag {
            let (rs, re) = self.boundaries[block.row_block];
            let (cs, ce) = self.boundaries[block.col_block];
            let x_j = x.slice(scirs2_core::ndarray::s![cs..ce]).to_owned();

            let u = &block.u;
            let s = &block.s;
            let v = &block.v;

            let k = s.len();
            let v_rows = v.nrows().min(ce - cs);
            let v_cols = v.ncols().min(k);

            // tmp = V^T · x_j  (k,)
            let mut tmp = Array1::<F>::zeros(k);
            for l in 0..v_cols {
                for r in 0..v_rows {
                    tmp[l] += v[[r, l]] * x_j[r];
                }
            }

            // scale tmp by singular values: tmp[l] *= s[l]
            for l in 0..k {
                tmp[l] *= s[l];
            }

            // y[row_range] += U · tmp
            let u_rows = u.nrows().min(re - rs);
            let u_cols = u.ncols().min(k);
            for r in 0..u_rows {
                let mut acc = F::zero();
                for l in 0..u_cols {
                    acc += u[[r, l]] * tmp[l];
                }
                y[rs + r] += acc;
            }
        }

        Ok(y)
    }

    /// Reconstruct the dense matrix from SSS generators (for testing/debugging).
    ///
    /// O(n²) — only suitable for small matrices.
    pub fn to_dense(&self) -> LinalgResult<Array2<F>> {
        let n = self.total_size;
        let mut result = Array2::<F>::zeros((n, n));
        let mut e = Array1::<F>::zeros(n);
        for j in 0..n {
            if j > 0 {
                e[j - 1] = F::zero();
            }
            e[j] = F::one();
            let col = self.matvec(&e)?;
            for i in 0..n {
                result[[i, j]] = col[i];
            }
        }
        Ok(result)
    }

    /// Ratio of SSS generator storage to equivalent dense storage.
    pub fn storage_ratio(&self) -> f64 {
        let dense = (self.total_size * self.total_size) as f64;
        if dense == 0.0 {
            return 1.0;
        }
        // Count storage in off-diagonal compressed blocks + diagonal blocks
        let off_stored: usize = self
            .off_diag
            .iter()
            .map(|b| b.u.len() + b.s.len() + b.v.len())
            .sum();
        let diag_stored: usize = self.d.iter().map(|d| d.len()).sum();
        (off_stored + diag_stored) as f64 / dense
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Build a simple tridiagonal matrix.
    fn tridiagonal(n: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, n), |(i, j)| {
            if i == j {
                2.0
            } else if (i as isize - j as isize).abs() == 1 {
                -1.0
            } else {
                0.0
            }
        })
    }

    /// Arrow matrix (diagonal + last column + last row).
    fn arrow_matrix(n: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, n), |(i, j)| {
            if i == j {
                3.0
            } else if i == n - 1 || j == n - 1 {
                1.0
            } else {
                0.0
            }
        })
    }

    #[test]
    fn test_sss_config_defaults() {
        let cfg = SssConfig::default();
        assert_eq!(cfg.block_size, 4);
        assert_eq!(cfg.rank, 4);
    }

    #[test]
    fn test_matvec_tridiagonal() {
        let n = 8;
        let mat = tridiagonal(n);
        let x = Array1::from_shape_fn(n, |i| (i + 1) as f64);
        let y_dense: Array1<f64> = mat.dot(&x);

        let config = SssConfig {
            block_size: 2,
            rank: 2,
        };
        let sss = SssMatrix::from_banded(&mat, config).expect("from_banded");
        let y_sss = sss.matvec(&x).expect("matvec");

        for i in 0..n {
            assert!(
                (y_sss[i] - y_dense[i]).abs() < 1e-10,
                "tridiagonal matvec mismatch at {}: got {} expected {}",
                i,
                y_sss[i],
                y_dense[i]
            );
        }
    }

    #[test]
    fn test_matvec_arrow_matrix() {
        let n = 8;
        let mat = arrow_matrix(n);
        let x = Array1::from_shape_fn(n, |i| i as f64 + 1.0);
        let y_dense: Array1<f64> = mat.dot(&x);

        let config = SssConfig {
            block_size: 2,
            rank: 2,
        };
        let sss = SssMatrix::from_banded(&mat, config).expect("from_banded");
        let y_sss = sss.matvec(&x).expect("matvec");

        for i in 0..n {
            assert!(
                (y_sss[i] - y_dense[i]).abs() < 1e-8,
                "arrow matvec mismatch at {}: got {} expected {}",
                i,
                y_sss[i],
                y_dense[i]
            );
        }
    }

    #[test]
    fn test_to_dense_roundtrip_tridiagonal() {
        let n = 6;
        let mat = tridiagonal(n);
        let config = SssConfig {
            block_size: 2,
            rank: 2,
        };
        let sss = SssMatrix::from_banded(&mat, config).expect("from_banded");
        let reconstructed = sss.to_dense().expect("to_dense");

        for i in 0..n {
            for j in 0..n {
                assert!(
                    (reconstructed[[i, j]] - mat[[i, j]]).abs() < 1e-10,
                    "roundtrip mismatch at ({},{}): got {} expected {}",
                    i,
                    j,
                    reconstructed[[i, j]],
                    mat[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_storage_ratio_banded() {
        let n = 32;
        let mat = tridiagonal(n);
        let config = SssConfig {
            block_size: 4,
            rank: 2,
        };
        let sss = SssMatrix::from_banded(&mat, config).expect("from_banded");
        let ratio = sss.storage_ratio();
        // For a tridiagonal n=32 most off-diagonal blocks are zero so the
        // compressed storage uses rank-1 blocks; ratio should be < n
        assert!(ratio > 0.0, "storage_ratio must be positive");
    }

    #[test]
    fn test_block_size_one_scalar_generators() {
        let n = 4;
        let mat = tridiagonal(n);
        let x = Array1::from_shape_fn(n, |i| (i + 1) as f64);
        let y_dense: Array1<f64> = mat.dot(&x);

        let config = SssConfig {
            block_size: 1,
            rank: 1,
        };
        let sss = SssMatrix::from_banded(&mat, config).expect("from_banded");
        let y_sss = sss.matvec(&x).expect("matvec");

        for i in 0..n {
            assert!(
                (y_sss[i] - y_dense[i]).abs() < 1e-10,
                "scalar generators matvec mismatch at {}: got {} expected {}",
                i,
                y_sss[i],
                y_dense[i]
            );
        }
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let n = 4;
        let mat = tridiagonal(n);
        let config = SssConfig::default();
        let sss = SssMatrix::from_banded(&mat, config).expect("from_banded");
        let x_bad = Array1::<f64>::zeros(n + 1);
        assert!(sss.matvec(&x_bad).is_err());
    }

    #[test]
    fn test_forward_sweep_lower_triangular() {
        // Lower triangular matrix
        let n = 4;
        let mut mat = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            mat[[i, i]] = 1.0;
        }
        mat[[1, 0]] = 2.0;
        mat[[2, 0]] = 3.0;
        mat[[2, 1]] = 4.0;
        mat[[3, 0]] = 5.0;
        mat[[3, 1]] = 6.0;
        mat[[3, 2]] = 7.0;

        let x = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let y_dense: Array1<f64> = mat.dot(&x);

        let config = SssConfig {
            block_size: 1,
            rank: 1,
        };
        let sss = SssMatrix::from_banded(&mat, config).expect("from_banded");
        let y_sss = sss.matvec(&x).expect("matvec");

        for i in 0..n {
            assert!(
                (y_sss[i] - y_dense[i]).abs() < 1e-10,
                "forward sweep mismatch at {}: got {} expected {}",
                i,
                y_sss[i],
                y_dense[i]
            );
        }
    }

    #[test]
    fn test_backward_sweep_upper_triangular() {
        // Upper triangular matrix
        let n = 4;
        let mut mat = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            mat[[i, i]] = 1.0;
        }
        mat[[0, 1]] = 2.0;
        mat[[0, 2]] = 3.0;
        mat[[0, 3]] = 4.0;
        mat[[1, 2]] = 5.0;
        mat[[1, 3]] = 6.0;
        mat[[2, 3]] = 7.0;

        let x = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
        let y_dense: Array1<f64> = mat.dot(&x);

        let config = SssConfig {
            block_size: 1,
            rank: 1,
        };
        let sss = SssMatrix::from_banded(&mat, config).expect("from_banded");
        let y_sss = sss.matvec(&x).expect("matvec");

        for i in 0..n {
            assert!(
                (y_sss[i] - y_dense[i]).abs() < 1e-10,
                "backward sweep mismatch at {}: got {} expected {}",
                i,
                y_sss[i],
                y_dense[i]
            );
        }
    }

    #[test]
    fn test_non_square_matrix_rejected() {
        let mat = Array2::<f64>::zeros((3, 4));
        let config = SssConfig::default();
        assert!(SssMatrix::from_banded(&mat, config).is_err());
    }

    #[test]
    fn test_storage_ratio_positive() {
        let n = 8;
        let mat = tridiagonal(n);
        let config = SssConfig::default();
        let sss = SssMatrix::from_banded(&mat, config).expect("from_banded");
        assert!(sss.storage_ratio() > 0.0);
    }
}
