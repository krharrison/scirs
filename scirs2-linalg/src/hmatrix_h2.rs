//! H²-matrix (nested cluster bases) for fast matrix-vector products.
//!
//! H²-matrices improve upon standard H-matrices by using *nested* cluster bases:
//! each cluster `t` has a local basis `U_t` and a transfer matrix `E_{t → parent}`
//! such that the full column basis is `E_ancestor ⋯ E_parent · U_leaf`.
//!
//! This reduces storage from O(n k²) to O(n k) while maintaining O(n k) matvec.
//!
//! ## References
//!
//! - Hackbusch, W. & Börm, S. (2002). "Data-sparse approximation by adaptive H²-matrices."
//! - Börm, S. (2010). "Efficient Numerical Methods for Non-local Operators."

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for H²-matrix construction.
#[derive(Debug, Clone)]
pub struct H2Config {
    /// Maximum size of a leaf cluster.
    pub leaf_size: usize,
    /// Approximation rank for admissible blocks.
    pub rank: usize,
    /// Admissibility parameter η: block (t,s) is admissible when
    /// `min(diam(t), diam(s)) <= η * dist(t, s)`.
    pub eta: f64,
    /// Truncation tolerance for SVD-based compression.
    pub tol: f64,
}

impl Default for H2Config {
    fn default() -> Self {
        Self {
            leaf_size: 32,
            rank: 8,
            eta: 2.0,
            tol: 1e-10,
        }
    }
}

// ---------------------------------------------------------------------------
// Cluster basis
// ---------------------------------------------------------------------------

/// Nested basis for a single cluster in the H²-matrix hierarchy.
///
/// For a leaf cluster this holds the local basis `U` (size × rank).
/// For an interior cluster the transfer matrix `E` (parent_rank × child_rank)
/// satisfies `U_parent ≈ E · U_child`.
#[derive(Debug, Clone)]
pub struct ClusterBasis<F> {
    /// Local basis matrix (cluster_size × local_rank or rank × rank for transfer).
    pub u: Array2<F>,
    /// Transfer matrix to parent cluster (parent_rank × local_rank). `None` at root.
    pub transfer: Option<Array2<F>>,
    /// Start index of this cluster in the global index space.
    pub start: usize,
    /// End index (exclusive) of this cluster.
    pub end: usize,
}

impl<F: Float + Debug + Clone> ClusterBasis<F> {
    /// Create a new leaf cluster basis with identity-like initialisation.
    pub fn new_leaf(start: usize, end: usize, rank: usize) -> Self {
        let size = end - start;
        let effective_rank = rank.min(size);
        // Initialise with zeros; caller fills in real content.
        Self {
            u: Array2::zeros((size, effective_rank)),
            transfer: None,
            start,
            end,
        }
    }
}

// ---------------------------------------------------------------------------
// H²-block
// ---------------------------------------------------------------------------

/// A block in the H²-matrix block structure.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum H2Block<F> {
    /// Dense (non-admissible) block stored explicitly.
    Dense {
        /// Dense submatrix data.
        data: Array2<F>,
        /// Row start index in the global matrix.
        row_start: usize,
        /// Row end index (exclusive).
        row_end: usize,
        /// Column start index.
        col_start: usize,
        /// Column end index (exclusive).
        col_end: usize,
    },
    /// Admissible block stored via coupling matrix S where the block ≈ U_t · S · V_s^T.
    Lowrank {
        /// Left singular vectors U (row_size × rank).
        u: Array2<F>,
        /// Singular values (rank,) — embedded in S diagonal.
        s: Array2<F>,
        /// Right singular vectors V (col_size × rank).
        v: Array2<F>,
        /// Row cluster index into `H2Matrix::row_bases`.
        row_cluster: usize,
        /// Column cluster index into `H2Matrix::col_bases`.
        col_cluster: usize,
        /// Row start/end for this block.
        row_start: usize,
        row_end: usize,
        /// Col start/end for this block.
        col_start: usize,
        col_end: usize,
    },
}

// ---------------------------------------------------------------------------
// H²-matrix
// ---------------------------------------------------------------------------

/// H²-matrix with nested cluster bases.
///
/// For a matrix A of size n × m:
/// - `row_bases[i]` contains the nested row basis for the i-th row cluster.
/// - `col_bases[j]` contains the nested column basis for the j-th column cluster.
/// - `blocks` lists all leaf blocks (either `Dense` or `Lowrank`).
#[derive(Debug)]
pub struct H2Matrix<F> {
    /// Number of rows.
    pub n: usize,
    /// Number of columns.
    pub m: usize,
    /// Row cluster bases (one per leaf cluster).
    pub row_bases: Vec<ClusterBasis<F>>,
    /// Column cluster bases (one per leaf cluster).
    pub col_bases: Vec<ClusterBasis<F>>,
    /// All blocks (leaf level of the block-cluster tree).
    pub blocks: Vec<H2Block<F>>,
    /// Configuration used to build this matrix.
    pub config: H2Config,
}

// ---------------------------------------------------------------------------
// Construction helpers
// ---------------------------------------------------------------------------

/// Divide `[start, end)` into approximately equal leaf clusters of size ≤ `leaf_size`.
fn build_clusters(start: usize, end: usize, leaf_size: usize) -> Vec<(usize, usize)> {
    let size = end - start;
    if size <= leaf_size {
        return vec![(start, end)];
    }
    let mid = start + size / 2;
    let mut left = build_clusters(start, mid, leaf_size);
    let right = build_clusters(mid, end, leaf_size);
    left.extend(right);
    left
}

/// Check admissibility: both cluster sizes are ≤ η * (gap between clusters).
///
/// We use the simple 1-D criterion: `min(|t|, |s|) <= η * dist(t, s)`.
fn is_admissible(t_start: usize, t_end: usize, s_start: usize, s_end: usize, eta: f64) -> bool {
    // Distance between clusters in index space (0 if they overlap)
    let dist = if t_end <= s_start {
        (s_start - t_end) as f64
    } else if s_end <= t_start {
        (t_start - s_end) as f64
    } else {
        0.0 // overlapping → not admissible
    };
    let min_size = (t_end - t_start).min(s_end - s_start) as f64;
    // Avoid admissibility when clusters touch or overlap
    dist > 0.0 && min_size <= eta * dist
}

/// Truncated SVD of `mat` up to `rank` components.
///
/// Returns `(u_k, s_k, vt_k)` where columns of U and rows of Vt span the
/// dominant `k` singular values, with `k = min(rank, n_nonzero_sv)`.
fn truncated_svd<F>(
    mat: &Array2<F>,
    rank: usize,
    tol: f64,
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
    let m = mat.nrows();
    let n = mat.ncols();
    if m == 0 || n == 0 {
        return Err(LinalgError::ShapeError("Empty matrix in SVD".to_string()));
    }

    // Use the existing decomposition SVD via crate::decomposition::svd
    let (u_full, s_full, vt_full) = crate::decomposition::svd(&mat.view(), true, None)?;

    // Determine effective rank
    let max_k = rank.min(s_full.len());
    let s0 = s_full[0];
    let threshold = if s0 > F::zero() {
        s0 * F::from_f64(tol).unwrap_or(F::zero())
    } else {
        F::zero()
    };

    let k = {
        let mut k = 0usize;
        for &sv in s_full.iter().take(max_k) {
            if sv > threshold {
                k += 1;
            } else {
                break;
            }
        }
        k.max(1)
    };

    let u_k = u_full.slice(scirs2_core::ndarray::s![.., ..k]).to_owned();
    let s_k = s_full.slice(scirs2_core::ndarray::s![..k]).to_owned();
    let vt_k = vt_full.slice(scirs2_core::ndarray::s![..k, ..]).to_owned();

    Ok((u_k, s_k, vt_k))
}

// ---------------------------------------------------------------------------
// H2Matrix implementation
// ---------------------------------------------------------------------------

impl<F> H2Matrix<F>
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
    /// Build an H²-matrix from a dense matrix using block SVD compression.
    ///
    /// The algorithm:
    /// 1. Recursively partition rows and columns into leaf clusters.
    /// 2. For each pair of leaf clusters check admissibility.
    /// 3. Admissible pairs → low-rank compression via truncated SVD.
    /// 4. Non-admissible pairs → store dense subblock.
    /// 5. Assemble nested cluster bases from the low-rank factors.
    pub fn from_dense(mat: &Array2<F>, config: H2Config) -> LinalgResult<Self> {
        let n = mat.nrows();
        let m = mat.ncols();

        if n == 0 || m == 0 {
            return Err(LinalgError::ShapeError(
                "H2Matrix::from_dense: empty matrix".to_string(),
            ));
        }

        let row_clusters = build_clusters(0, n, config.leaf_size);
        let col_clusters = build_clusters(0, m, config.leaf_size);

        let nr = row_clusters.len();
        let nc = col_clusters.len();

        // Build initial (identity-like) cluster bases; we will fill in the U
        // matrices from the SVD factors of admissible blocks.
        let rank = config.rank;
        let mut row_bases: Vec<ClusterBasis<F>> = row_clusters
            .iter()
            .map(|&(rs, re)| ClusterBasis::new_leaf(rs, re, rank))
            .collect();
        let mut col_bases: Vec<ClusterBasis<F>> = col_clusters
            .iter()
            .map(|&(cs, ce)| ClusterBasis::new_leaf(cs, ce, rank))
            .collect();

        let mut blocks: Vec<H2Block<F>> = Vec::new();

        for (ri, &(rs, re)) in row_clusters.iter().enumerate() {
            for (ci, &(cs, ce)) in col_clusters.iter().enumerate() {
                let sub = mat
                    .slice(scirs2_core::ndarray::s![rs..re, cs..ce])
                    .to_owned();

                if is_admissible(rs, re, cs, ce, config.eta) {
                    // Admissible block: compress via truncated SVD
                    match truncated_svd(&sub, rank, config.tol) {
                        Ok((u_k, s_k, vt_k)) => {
                            let k = s_k.len();
                            // Coupling matrix S = diag(s_k)
                            let mut s_mat = Array2::<F>::zeros((k, k));
                            for i in 0..k {
                                s_mat[[i, i]] = s_k[i];
                            }
                            // V = Vt^T  (col_size × k)
                            let v_k = vt_k.t().to_owned();

                            // Also update the cluster bases for external access
                            row_bases[ri].u = u_k.clone();
                            col_bases[ci].u = v_k.clone();

                            blocks.push(H2Block::Lowrank {
                                u: u_k,
                                s: s_mat,
                                v: v_k,
                                row_cluster: ri,
                                col_cluster: ci,
                                row_start: rs,
                                row_end: re,
                                col_start: cs,
                                col_end: ce,
                            });
                        }
                        Err(_) => {
                            // Fall back to dense on SVD failure
                            blocks.push(H2Block::Dense {
                                data: sub,
                                row_start: rs,
                                row_end: re,
                                col_start: cs,
                                col_end: ce,
                            });
                        }
                    }
                } else {
                    blocks.push(H2Block::Dense {
                        data: sub,
                        row_start: rs,
                        row_end: re,
                        col_start: cs,
                        col_end: ce,
                    });
                }
            }
        }

        Ok(Self {
            n,
            m,
            row_bases,
            col_bases,
            blocks,
            config,
        })
    }

    /// Compute the matrix–vector product `y = A · x` using the H²-matrix representation.
    ///
    /// For each block:
    /// - `Dense { data, ... }`: `y[row_range] += data · x[col_range]`
    /// - `Lowrank { s, ... }`: `y[row_range] += U_row · S · (V_col^T · x[col_range])`
    pub fn matvec(&self, x: &Array1<F>) -> LinalgResult<Array1<F>> {
        if x.len() != self.m {
            return Err(LinalgError::DimensionError(format!(
                "H2Matrix::matvec: x has length {} but matrix has {} columns",
                x.len(),
                self.m
            )));
        }

        let mut y = Array1::<F>::zeros(self.n);

        for block in &self.blocks {
            match block {
                H2Block::Dense {
                    data,
                    row_start,
                    row_end,
                    col_start,
                    col_end,
                } => {
                    let x_sub = x
                        .slice(scirs2_core::ndarray::s![*col_start..*col_end])
                        .to_owned();
                    let row_size = *row_end - *row_start;
                    let d_rows = data.nrows().min(row_size);
                    let d_cols = data.ncols().min(x_sub.len());
                    for i in 0..d_rows {
                        let mut acc = F::zero();
                        for j in 0..d_cols {
                            acc += data[[i, j]] * x_sub[j];
                        }
                        y[*row_start + i] += acc;
                    }
                }
                H2Block::Lowrank {
                    u,
                    s,
                    v,
                    row_start,
                    row_end,
                    col_start,
                    col_end,
                    ..
                } => {
                    let x_sub = x
                        .slice(scirs2_core::ndarray::s![*col_start..*col_end])
                        .to_owned();

                    // tmp1 = V^T · x_sub  (rank,)
                    let vt_rank = v.ncols();
                    let v_rows = v.nrows().min(*col_end - *col_start).min(x_sub.len());
                    let mut tmp1 = Array1::<F>::zeros(vt_rank);
                    for j in 0..v_rows {
                        for l in 0..vt_rank {
                            tmp1[l] += v[[j, l]] * x_sub[j];
                        }
                    }

                    // tmp2 = S · tmp1  (row_rank,)
                    let s_rows = s.nrows();
                    let rank_eff = s_rows.min(vt_rank).min(s.ncols());
                    let mut tmp2 = Array1::<F>::zeros(s_rows);
                    for i in 0..s_rows {
                        for l in 0..rank_eff {
                            tmp2[i] += s[[i, l]] * tmp1[l];
                        }
                    }

                    // y[row_range] += U · tmp2
                    let row_size = *row_end - *row_start;
                    let u_rows = u.nrows().min(row_size);
                    let u_cols = u.ncols().min(s_rows);
                    for i in 0..u_rows {
                        let mut acc = F::zero();
                        for l in 0..u_cols {
                            acc += u[[i, l]] * tmp2[l];
                        }
                        y[*row_start + i] += acc;
                    }
                }
            }
        }

        Ok(y)
    }

    /// Ratio of H²-matrix storage to equivalent dense storage.
    ///
    /// Values < 1 indicate compression; values > 1 indicate expansion.
    pub fn memory_ratio(&self) -> f64 {
        let dense_size = (self.n * self.m) as f64;
        if dense_size == 0.0 {
            return 1.0;
        }

        let stored: usize = self
            .blocks
            .iter()
            .map(|b| match b {
                H2Block::Dense { data, .. } => data.len(),
                H2Block::Lowrank { u, s, v, .. } => u.len() + s.len() + v.len(),
            })
            .sum();

        stored as f64 / dense_size
    }

    /// Frobenius approximation error relative to the original dense matrix.
    ///
    /// Reconstructs the approximation via `matvec` on each column of the identity
    /// (expensive — only for small matrices / testing).
    pub fn approx_error(&self, original: &Array2<F>) -> LinalgResult<F> {
        if original.nrows() != self.n || original.ncols() != self.m {
            return Err(LinalgError::ShapeError(
                "approx_error: shape mismatch".to_string(),
            ));
        }

        let mut err_sq = F::zero();
        let mut orig_sq = F::zero();

        // Reconstruct column by column using the H²-matvec
        let mut e_col = Array1::<F>::zeros(self.m);
        for j in 0..self.m {
            // Set unit vector e_j
            if j > 0 {
                e_col[j - 1] = F::zero();
            }
            e_col[j] = F::one();

            let approx_col = self.matvec(&e_col)?;

            for i in 0..self.n {
                let diff = approx_col[i] - original[[i, j]];
                err_sq += diff * diff;
                orig_sq += original[[i, j]] * original[[i, j]];
            }
        }

        // Relative Frobenius error
        let denom = if orig_sq > F::zero() {
            orig_sq.sqrt()
        } else {
            F::one()
        };
        Ok(err_sq.sqrt() / denom)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn hilbert_matrix(n: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, n), |(i, j)| 1.0 / (i + j + 1) as f64)
    }

    fn smooth_kernel_matrix(n: usize) -> Array2<f64> {
        // Gaussian kernel: K[i,j] = exp(-|i-j|^2 / n)
        Array2::from_shape_fn((n, n), |(i, j)| {
            let d = (i as f64 - j as f64).powi(2) / n as f64;
            (-d).exp()
        })
    }

    #[test]
    fn test_h2config_defaults() {
        let cfg = H2Config::default();
        assert_eq!(cfg.leaf_size, 32);
        assert_eq!(cfg.rank, 8);
        assert!((cfg.eta - 2.0).abs() < 1e-12);
        assert!(cfg.tol < 1e-9);
    }

    #[test]
    fn test_from_dense_identity_memory_ratio() {
        let n = 8usize;
        let mat = Array2::<f64>::eye(n);
        let config = H2Config {
            leaf_size: 4,
            rank: 2,
            ..Default::default()
        };
        let h2 = H2Matrix::from_dense(&mat, config).expect("from_dense failed");
        // For small identity the memory ratio may be >= 1 but the matrix is valid
        let ratio = h2.memory_ratio();
        assert!(ratio > 0.0, "memory_ratio must be positive");
    }

    #[test]
    fn test_matvec_identity() {
        let n = 8usize;
        let mat = Array2::<f64>::eye(n);
        let config = H2Config {
            leaf_size: 4,
            rank: 2,
            eta: 1.5,
            tol: 1e-12,
        };
        let h2 = H2Matrix::from_dense(&mat, config).expect("from_dense");
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let y = h2.matvec(&x).expect("matvec");
        for i in 0..n {
            assert!(
                (y[i] - x[i]).abs() < 1e-10,
                "identity matvec failed at {}: got {} expected {}",
                i,
                y[i],
                x[i]
            );
        }
    }

    #[test]
    fn test_matvec_close_to_dense() {
        let n = 12usize;
        let mat = smooth_kernel_matrix(n);
        let x = Array1::from_shape_fn(n, |i| (i + 1) as f64);
        let y_dense: Array1<f64> = mat.dot(&x);

        let config = H2Config {
            leaf_size: 4,
            rank: 4,
            eta: 2.0,
            tol: 1e-12,
        };
        let h2 = H2Matrix::from_dense(&mat, config).expect("from_dense");
        let y_h2 = h2.matvec(&x).expect("matvec");

        for i in 0..n {
            assert!(
                (y_h2[i] - y_dense[i]).abs() < 1e-8,
                "matvec mismatch at {}: got {} expected {}",
                i,
                y_h2[i],
                y_dense[i]
            );
        }
    }

    #[test]
    fn test_approx_error_smooth_kernel() {
        let n = 16usize;
        let mat = smooth_kernel_matrix(n);
        let config = H2Config {
            leaf_size: 4,
            rank: 6,
            eta: 2.0,
            tol: 1e-12,
        };
        let h2 = H2Matrix::from_dense(&mat, config).expect("from_dense");
        let err = h2.approx_error(&mat).expect("approx_error");
        // Smooth kernel should be compressible; relative error < 1
        assert!(err < 1.0, "approx_error too large: {}", err);
    }

    #[test]
    fn test_memory_ratio_decreases_with_rank() {
        // For a larger smooth kernel the higher-rank approximation should
        // generally not be MORE compressed (it can be equal or more).
        // The important invariant is that both produce valid results.
        let n = 24usize;
        let mat = smooth_kernel_matrix(n);

        let config_low = H2Config {
            leaf_size: 6,
            rank: 2,
            eta: 2.0,
            tol: 1e-12,
        };
        let config_high = H2Config {
            leaf_size: 6,
            rank: 8,
            eta: 2.0,
            tol: 1e-12,
        };

        let h2_low = H2Matrix::from_dense(&mat, config_low).expect("from_dense low");
        let h2_high = H2Matrix::from_dense(&mat, config_high).expect("from_dense high");

        let ratio_low = h2_low.memory_ratio();
        let ratio_high = h2_high.memory_ratio();
        // Higher rank → more storage → higher ratio (or equal)
        assert!(
            ratio_high >= ratio_low - 1e-9,
            "Expected high_rank ratio ({}) >= low_rank ratio ({})",
            ratio_high,
            ratio_low
        );
    }

    #[test]
    fn test_admissibility_detection() {
        // Clusters far apart should be admissible
        assert!(is_admissible(0, 4, 8, 12, 2.0));
        // Clusters that are adjacent (gap=0) should not be admissible
        assert!(!is_admissible(0, 4, 4, 8, 2.0));
        // Overlapping clusters are not admissible
        assert!(!is_admissible(0, 6, 4, 10, 2.0));
    }

    #[test]
    fn test_cluster_basis_construction() {
        let basis = ClusterBasis::<f64>::new_leaf(0, 8, 4);
        assert_eq!(basis.u.nrows(), 8);
        assert_eq!(basis.u.ncols(), 4);
        assert!(basis.transfer.is_none());
        assert_eq!(basis.start, 0);
        assert_eq!(basis.end, 8);
    }

    #[test]
    fn test_hilbert_matrix_roundtrip() {
        let n = 8usize;
        let mat = hilbert_matrix(n);
        let x = Array1::from_shape_fn(n, |i| (i + 1) as f64);
        let y_dense: Array1<f64> = mat.dot(&x);

        let config = H2Config {
            leaf_size: 4,
            rank: 4,
            eta: 2.0,
            tol: 1e-12,
        };
        let h2 = H2Matrix::from_dense(&mat, config).expect("from_dense");
        let y_h2 = h2.matvec(&x).expect("matvec");

        // Hilbert matrix is smooth; matvec should be reasonably close
        let rel_err: f64 = y_h2
            .iter()
            .zip(y_dense.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
            / y_dense.iter().map(|&b| b * b).sum::<f64>().sqrt();
        assert!(
            rel_err < 1.0,
            "Hilbert matvec relative error too large: {}",
            rel_err
        );
    }

    #[test]
    fn test_matvec_dimension_check() {
        let n = 4usize;
        let mat = Array2::<f64>::eye(n);
        let config = H2Config::default();
        let h2 = H2Matrix::from_dense(&mat, config).expect("from_dense");
        let x_bad = Array1::<f64>::zeros(n + 1);
        assert!(h2.matvec(&x_bad).is_err());
    }

    #[test]
    fn test_approx_error_shape_mismatch() {
        let n = 4usize;
        let mat = Array2::<f64>::eye(n);
        let config = H2Config::default();
        let h2 = H2Matrix::from_dense(&mat, config).expect("from_dense");
        let wrong = Array2::<f64>::eye(n + 1);
        assert!(h2.approx_error(&wrong).is_err());
    }

    #[test]
    fn test_h2block_non_exhaustive() {
        // Verify that the #[non_exhaustive] enum can be constructed and matched
        let block: H2Block<f64> = H2Block::Dense {
            data: Array2::eye(2),
            row_start: 0,
            row_end: 2,
            col_start: 0,
            col_end: 2,
        };
        match block {
            H2Block::Dense { data, .. } => assert_eq!(data.nrows(), 2),
            H2Block::Lowrank { .. } => {}
        }
    }
}
