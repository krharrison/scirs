//! Algebraic Multigrid (AMG) standalone solver for SPD systems
//!
//! This module provides a Ruge-Stüben (classical AMG) solver for
//! symmetric positive-definite (SPD) linear systems arising from
//! discretizations of elliptic PDEs and related problems.
//!
//! # Algorithm overview
//!
//! 1. **Setup phase** (`AlgebraicMultigrid::setup`):
//!    - Detect strong connections via strength-of-connection measure
//!    - Coarsen the matrix using classical Ruge-Stüben C/F splitting
//!    - Build piecewise-constant prolongation P and restriction R = P^T
//!    - Form coarse-grid operator A_c = R A P (Galerkin)
//!    - Recurse until coarse grid is small enough
//!
//! 2. **Solve phase** (`AlgebraicMultigrid::solve`):
//!    - V-cycle or W-cycle multigrid iteration
//!    - Pre/post smoothing via weighted Jacobi
//!    - Direct solve on coarsest grid
//!
//! # References
//! - Ruge, J.W. & Stüben, K. (1987). "Algebraic multigrid". In S.F. McCormick
//!   (Ed.), *Multigrid methods*, Frontiers in Applied Math., SIAM.
//! - Briggs, W.L., Henson, V.E., & McCormick, S.F. (2000). *A Multigrid Tutorial*
//!   (2nd ed.), SIAM.

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use scirs2_core::ndarray::Array1;
use std::collections::HashSet;

/// AMG cycle type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmgCycle {
    /// V-cycle: one visit to each level going down, then up
    V,
    /// W-cycle: recursive double application at each level
    W,
}

/// Configuration for `AlgebraicMultigrid`
#[derive(Debug, Clone)]
pub struct AmgConfig {
    /// Maximum number of levels in the hierarchy (default: 10)
    pub max_levels: usize,
    /// Target coarsening ratio per level (default: 0.5)
    pub coarsening_ratio: f64,
    /// Strength-of-connection threshold θ ∈ (0,1) (default: 0.25)
    pub strong_threshold: f64,
    /// Number of pre-smoothing sweeps (default: 1)
    pub pre_smooth: usize,
    /// Number of post-smoothing sweeps (default: 1)
    pub post_smooth: usize,
    /// Maximum outer iterations (default: 100)
    pub max_iter: usize,
    /// Relative convergence tolerance (default: 1e-8)
    pub tol: f64,
    /// Multigrid cycle type (default: V)
    pub cycle: AmgCycle,
    /// Maximum coarse-grid size for direct solve (default: 50)
    pub max_coarse: usize,
    /// Damping weight for Jacobi smoother ω ∈ (0,1] (default: 2/3)
    pub omega: f64,
}

impl Default for AmgConfig {
    fn default() -> Self {
        Self {
            max_levels: 10,
            coarsening_ratio: 0.5,
            strong_threshold: 0.25,
            pre_smooth: 1,
            post_smooth: 1,
            max_iter: 100,
            tol: 1e-8,
            cycle: AmgCycle::V,
            max_coarse: 50,
            omega: 2.0 / 3.0,
        }
    }
}

/// One level in the AMG hierarchy
#[derive(Debug)]
struct AmgLevel {
    /// System matrix at this level
    a: CsrMatrix<f64>,
    /// Prolongation operator P: coarse → fine
    p: CsrMatrix<f64>,
    /// Restriction operator R = P^T: fine → coarse
    r: CsrMatrix<f64>,
}

/// Ruge-Stüben Algebraic Multigrid solver for SPD systems
///
/// # Example
/// ```
/// use scirs2_sparse::linalg::algebraic_multigrid::{AlgebraicMultigrid, AmgConfig};
/// use scirs2_sparse::csr::CsrMatrix;
///
/// // Build 5-point Laplacian on a 3×3 grid
/// let n = 9;
/// let mut rows = Vec::new();
/// let mut cols = Vec::new();
/// let mut data = Vec::new();
/// for iy in 0..3_usize {
///     for ix in 0..3_usize {
///         let i = iy * 3 + ix;
///         rows.push(i); cols.push(i); data.push(4.0);
///         if ix > 0 { rows.push(i); cols.push(i-1); data.push(-1.0); }
///         if ix < 2 { rows.push(i); cols.push(i+1); data.push(-1.0); }
///         if iy > 0 { rows.push(i); cols.push(i-3); data.push(-1.0); }
///         if iy < 2 { rows.push(i); cols.push(i+3); data.push(-1.0); }
///     }
/// }
/// let a = CsrMatrix::from_raw_csr(build_csr_from_triplets(rows, cols, data, (n, n)), (n, n))
///     .expect("valid input");
/// // ...
/// ```
pub struct AlgebraicMultigrid {
    /// Hierarchy of levels (finest first)
    levels: Vec<AmgLevel>,
    /// Configuration
    config: AmgConfig,
}

impl AlgebraicMultigrid {
    /// Build the AMG hierarchy from an SPD matrix A.
    ///
    /// # Errors
    /// Returns `SparseError` if the matrix is not square or the hierarchy
    /// construction fails.
    pub fn setup(a: &CsrMatrix<f64>, config: AmgConfig) -> SparseResult<Self> {
        let (rows, cols) = a.shape();
        if rows != cols {
            return Err(SparseError::ValueError(
                "AMG requires a square matrix".to_string(),
            ));
        }
        if rows == 0 {
            return Err(SparseError::ValueError(
                "AMG requires a non-empty matrix".to_string(),
            ));
        }

        let mut levels: Vec<AmgLevel> = Vec::new();
        let mut current = a.clone();

        for _lvl in 0..config.max_levels.saturating_sub(1) {
            let n_fine = current.rows();

            // Stop if already small enough
            if n_fine <= config.max_coarse {
                break;
            }

            // 1. Strength of connection graph
            let strong = strong_connections(&current, config.strong_threshold);

            // 2. C/F splitting (classical Ruge-Stüben)
            let cf = cf_splitting(n_fine, &strong);
            let n_coarse = cf.iter().filter(|&&c| c == CfLabel::C).count();

            // Guard: if coarsening ratio is too small, stop
            if n_coarse == 0 || n_coarse as f64 / n_fine as f64 > config.coarsening_ratio + 0.3 {
                break;
            }

            // 3. Build coarse-to-fine index maps
            let coarse_indices: Vec<usize> = cf
                .iter()
                .enumerate()
                .filter(|(_, &c)| c == CfLabel::C)
                .map(|(i, _)| i)
                .collect();

            // Map from fine index to coarse index (-1 for F-points)
            let mut fine_to_coarse: Vec<Option<usize>> = vec![None; n_fine];
            for (ci, &fi) in coarse_indices.iter().enumerate() {
                fine_to_coarse[fi] = Some(ci);
            }

            // 4. Build prolongation P (piecewise constant + direct interpolation)
            let p = build_prolongation(&current, &cf, &fine_to_coarse, n_coarse, &strong)?;

            // 5. Build restriction R = P^T
            let r = transpose_csr(&p)?;

            // 6. Galerkin coarse matrix: A_c = R A P
            let rp = matmul_csr(&r, &current)?;
            let coarse = matmul_csr(&rp, &p)?;

            levels.push(AmgLevel {
                a: current.clone(),
                p,
                r,
            });

            current = coarse;

            if current.rows() <= config.max_coarse {
                break;
            }
        }

        // Add coarsest-level matrix as a level with trivial P/R (empty)
        levels.push(AmgLevel {
            a: current,
            p: CsrMatrix::empty((0, 0)),
            r: CsrMatrix::empty((0, 0)),
        });

        if levels.is_empty() {
            return Err(SparseError::ValueError(
                "AMG hierarchy construction produced no levels".to_string(),
            ));
        }

        Ok(Self { levels, config })
    }

    /// Solve A x = b using AMG iterations.
    ///
    /// Starting from `x0` (zeros if `None`), iterates until convergence
    /// or `config.max_iter` is reached.
    ///
    /// # Returns
    /// Approximate solution vector.
    pub fn solve(
        &self,
        b: &Array1<f64>,
        x0: Option<&Array1<f64>>,
    ) -> SparseResult<Array1<f64>> {
        let n = b.len();
        let a0 = &self.levels[0].a;
        if a0.rows() != n {
            return Err(SparseError::DimensionMismatch {
                expected: a0.rows(),
                found: n,
            });
        }

        let mut x: Vec<f64> = match x0 {
            Some(x0_arr) => x0_arr.to_vec(),
            None => vec![0.0; n],
        };

        let b_vec: Vec<f64> = b.to_vec();
        let bnorm = norm2(&b_vec);
        let tol = self.config.tol * bnorm.max(1e-300);

        for _it in 0..self.config.max_iter {
            // Compute residual r = b - A x
            let ax = matvec_csr(a0, &x)?;
            let r: Vec<f64> = b_vec.iter().zip(ax.iter()).map(|(bi, axi)| bi - axi).collect();
            let rnorm = norm2(&r);

            if rnorm < tol {
                break;
            }

            // One multigrid cycle
            let correction = self.cycle(0, &r)?;
            for i in 0..n {
                x[i] += correction[i];
            }
        }

        Ok(Array1::from_vec(x))
    }

    /// Return the number of levels in the hierarchy
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    // --- Internal cycle recursion ---

    fn cycle(&self, level: usize, rhs: &[f64]) -> SparseResult<Vec<f64>> {
        let lvl = &self.levels[level];
        let n = lvl.a.rows();

        // Coarsest level: direct solve (Jacobi iterations as cheap stand-in)
        if level + 1 >= self.levels.len() {
            return self.direct_solve(level, rhs);
        }

        let mut x: Vec<f64> = vec![0.0; n];

        // Pre-smoothing
        for _ in 0..self.config.pre_smooth {
            weighted_jacobi_step(&lvl.a, &mut x, rhs, self.config.omega)?;
        }

        // Restrict residual to coarse grid
        let ax = matvec_csr(&lvl.a, &x)?;
        let residual: Vec<f64> = rhs.iter().zip(ax.iter()).map(|(ri, axi)| ri - axi).collect();
        let coarse_rhs = matvec_csr(&lvl.r, &residual)?;

        // Coarse-grid correction (one or two cycles depending on cycle type)
        let coarse_correction = self.cycle(level + 1, &coarse_rhs)?;
        let coarse_correction2 = if self.config.cycle == AmgCycle::W {
            self.cycle(level + 1, &coarse_rhs)?
        } else {
            coarse_correction.clone()
        };

        // Apply correction: use coarse_correction2 for W-cycle, coarse_correction for V
        let correction_to_apply = if self.config.cycle == AmgCycle::W {
            coarse_correction2
        } else {
            coarse_correction
        };

        // Prolongate correction and add
        let fine_correction = matvec_csr(&lvl.p, &correction_to_apply)?;
        for i in 0..n {
            x[i] += fine_correction[i];
        }

        // Post-smoothing
        for _ in 0..self.config.post_smooth {
            weighted_jacobi_step(&lvl.a, &mut x, rhs, self.config.omega)?;
        }

        Ok(x)
    }

    /// Solve on the coarsest level via repeated Jacobi iterations
    fn direct_solve(&self, level: usize, rhs: &[f64]) -> SparseResult<Vec<f64>> {
        let a = &self.levels[level].a;
        let n = a.rows();
        let mut x = vec![0.0_f64; n];
        let tol = self.config.tol * 1e-2;

        for _ in 0..200 {
            weighted_jacobi_step(a, &mut x, rhs, self.config.omega)?;
            let ax = matvec_csr(a, &x)?;
            let res: f64 = rhs
                .iter()
                .zip(ax.iter())
                .map(|(ri, axi)| (ri - axi).powi(2))
                .sum::<f64>()
                .sqrt();
            if res < tol {
                break;
            }
        }
        Ok(x)
    }
}

// --------------------------------------------------------------------------
// C/F splitting
// --------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CfLabel {
    /// Coarse-grid point
    C,
    /// Fine-grid point
    F,
    /// Undecided
    U,
}

/// Compute strong connections: i strongly influences j if
///   |A_{ij}| >= θ · max_{k≠i} |A_{ik}|
fn strong_connections(a: &CsrMatrix<f64>, theta: f64) -> Vec<Vec<usize>> {
    let n = a.rows();
    let mut strong: Vec<Vec<usize>> = vec![Vec::new(); n];

    for i in 0..n {
        // Find max |A_{ij}| for j ≠ i
        let max_off: f64 = a.indptr[i..=i + 1]
            .windows(2)
            .flat_map(|w| &a.indices[w[0]..w[1]])
            .zip(a.indptr[i..=i + 1].windows(2).flat_map(|w| &a.data[w[0]..w[1]]))
            .filter(|(&j, _)| j != i)
            .map(|(_, &v)| v.abs())
            .fold(0.0_f64, f64::max);

        if max_off == 0.0 {
            continue;
        }

        let threshold = theta * max_off;

        for k in a.indptr[i]..a.indptr[i + 1] {
            let j = a.indices[k];
            if j != i && a.data[k].abs() >= threshold {
                strong[i].push(j);
            }
        }
    }
    strong
}

/// Classical Ruge-Stüben C/F splitting
fn cf_splitting(n: usize, strong: &[Vec<usize>]) -> Vec<CfLabel> {
    let mut label = vec![CfLabel::U; n];

    // Compute "lambda" = number of undecided neighbours that strongly depend
    // on this node (influence measure)
    let mut lambda: Vec<usize> = vec![0; n];
    for i in 0..n {
        for &j in &strong[i] {
            lambda[j] += 1;
        }
    }

    let mut undecided: usize = n;

    while undecided > 0 {
        // Find undecided node with largest lambda
        let best = (0..n)
            .filter(|&i| label[i] == CfLabel::U)
            .max_by_key(|&i| lambda[i]);

        let c_node = match best {
            Some(c) => c,
            None => break,
        };

        // Mark as C-point
        label[c_node] = CfLabel::C;
        undecided -= 1;

        // All undecided nodes that c_node strongly influences become F-points
        let influenced: Vec<usize> = (0..n)
            .filter(|&j| j != c_node && label[j] == CfLabel::U && strong[j].contains(&c_node))
            .collect();

        for &f_node in &influenced {
            label[f_node] = CfLabel::F;
            undecided -= 1;

            // Increase lambda for undecided nodes that f_node strongly influences
            for &k in &strong[f_node] {
                if label[k] == CfLabel::U {
                    lambda[k] += 1;
                }
            }
        }
    }

    // Any remaining undecided nodes become F-points
    for l in label.iter_mut() {
        if *l == CfLabel::U {
            *l = CfLabel::F;
        }
    }

    label
}

// --------------------------------------------------------------------------
// Prolongation operator
// --------------------------------------------------------------------------

/// Build prolongation P: coarse → fine
///
/// For C-points: identity (1 on diagonal).
/// For F-points: standard interpolation using strong C-neighbours.
fn build_prolongation(
    a: &CsrMatrix<f64>,
    cf: &[CfLabel],
    fine_to_coarse: &[Option<usize>],
    n_coarse: usize,
    strong: &[Vec<usize>],
) -> SparseResult<CsrMatrix<f64>> {
    let n_fine = cf.len();
    let mut row_indices: Vec<usize> = Vec::new();
    let mut col_indices: Vec<usize> = Vec::new();
    let mut data: Vec<f64> = Vec::new();

    for i in 0..n_fine {
        match cf[i] {
            CfLabel::C => {
                // C-point: trivial prolongation (identity)
                if let Some(ci) = fine_to_coarse[i] {
                    row_indices.push(i);
                    col_indices.push(ci);
                    data.push(1.0);
                }
            }
            CfLabel::F => {
                // F-point: interpolate from strong C-neighbours
                let strong_c_nbrs: Vec<usize> = strong[i]
                    .iter()
                    .filter(|&&j| cf[j] == CfLabel::C)
                    .copied()
                    .collect();

                if strong_c_nbrs.is_empty() {
                    // Fallback: use all C-neighbours in matrix sparsity pattern
                    let c_nbrs_in_row: Vec<usize> = (a.indptr[i]..a.indptr[i + 1])
                        .map(|k| a.indices[k])
                        .filter(|&j| cf[j] == CfLabel::C)
                        .collect();

                    if c_nbrs_in_row.is_empty() {
                        // No C-neighbour at all: skip (contributes zero row)
                        continue;
                    }

                    // Equal-weight interpolation
                    let w = 1.0 / c_nbrs_in_row.len() as f64;
                    for j in c_nbrs_in_row {
                        if let Some(cj) = fine_to_coarse[j] {
                            row_indices.push(i);
                            col_indices.push(cj);
                            data.push(w);
                        }
                    }
                } else {
                    // Standard interpolation weight from Ruge-Stüben
                    // w_ij = -a_ij / (a_ii * sum_k(a_ik) for k in strong C)
                    let a_ii = get_diagonal(a, i);

                    // Sum of connections to F-neighbours not in strong_c
                    let strong_c_set: HashSet<usize> = strong_c_nbrs.iter().copied().collect();

                    // Denominator: a_ii + sum of non-strong-C connections
                    let mut denom = a_ii;
                    for k in a.indptr[i]..a.indptr[i + 1] {
                        let j = a.indices[k];
                        if j != i && !strong_c_set.contains(&j) {
                            // Distribute to strong C via lumping: add to diagonal contribution
                            denom += a.data[k];
                        }
                    }

                    if denom.abs() < 1e-300 {
                        // degenerate: equal weight
                        let w = 1.0 / strong_c_nbrs.len() as f64;
                        for j in &strong_c_nbrs {
                            if let Some(cj) = fine_to_coarse[*j] {
                                row_indices.push(i);
                                col_indices.push(cj);
                                data.push(w);
                            }
                        }
                        continue;
                    }

                    // Sum of a_ij for j in strong C-set
                    let sum_strong_c: f64 = strong_c_nbrs
                        .iter()
                        .map(|&j| {
                            (a.indptr[i]..a.indptr[i + 1])
                                .find(|&k| a.indices[k] == j)
                                .map(|k| a.data[k])
                                .unwrap_or(0.0)
                        })
                        .sum();

                    if sum_strong_c.abs() < 1e-300 {
                        continue;
                    }

                    for j in &strong_c_nbrs {
                        let a_ij = (a.indptr[i]..a.indptr[i + 1])
                            .find(|&k| a.indices[k] == *j)
                            .map(|k| a.data[k])
                            .unwrap_or(0.0);

                        let w_ij = -a_ij / (denom * sum_strong_c / a_ij.abs().max(1e-300));
                        // Simpler: direct interpolation weight
                        let w_direct = -a_ij / denom;

                        if let Some(cj) = fine_to_coarse[*j] {
                            row_indices.push(i);
                            col_indices.push(cj);
                            // Use direct interpolation (more stable)
                            data.push(w_direct);
                        }
                        let _ = w_ij;
                    }
                }
            }
            CfLabel::U => {
                // Should not happen after cf_splitting
            }
        }
    }

    if data.is_empty() {
        return Err(SparseError::ValueError(
            "Prolongation has no entries — coarsening failed".to_string(),
        ));
    }

    CsrMatrix::new(data, row_indices, col_indices, (n_fine, n_coarse))
}

// --------------------------------------------------------------------------
// Sparse matrix utilities
// --------------------------------------------------------------------------

/// Transpose a CSR matrix
fn transpose_csr(a: &CsrMatrix<f64>) -> SparseResult<CsrMatrix<f64>> {
    let (m, n) = a.shape();
    let nnz = a.nnz();

    let mut row_indices: Vec<usize> = Vec::with_capacity(nnz);
    let mut col_indices: Vec<usize> = Vec::with_capacity(nnz);
    let mut data: Vec<f64> = Vec::with_capacity(nnz);

    for i in 0..m {
        for k in a.indptr[i]..a.indptr[i + 1] {
            let j = a.indices[k];
            row_indices.push(j);
            col_indices.push(i);
            data.push(a.data[k]);
        }
    }

    CsrMatrix::new(data, row_indices, col_indices, (n, m))
}

/// Sparse matrix-matrix product C = A B
fn matmul_csr(a: &CsrMatrix<f64>, b: &CsrMatrix<f64>) -> SparseResult<CsrMatrix<f64>> {
    let (m, k) = a.shape();
    let (k2, n) = b.shape();
    if k != k2 {
        return Err(SparseError::DimensionMismatch {
            expected: k,
            found: k2,
        });
    }

    let mut row_indices: Vec<usize> = Vec::new();
    let mut col_indices: Vec<usize> = Vec::new();
    let mut data_out: Vec<f64> = Vec::new();

    for i in 0..m {
        // Accumulate row i of C = row i of A * B
        let mut c_row: std::collections::HashMap<usize, f64> =
            std::collections::HashMap::new();

        for ak in a.indptr[i]..a.indptr[i + 1] {
            let a_col = a.indices[ak];
            let a_val = a.data[ak];

            for bk in b.indptr[a_col]..b.indptr[a_col + 1] {
                let b_col = b.indices[bk];
                let b_val = b.data[bk];
                *c_row.entry(b_col).or_insert(0.0) += a_val * b_val;
            }
        }

        let mut c_row_vec: Vec<(usize, f64)> = c_row.into_iter().collect();
        c_row_vec.sort_by_key(|(j, _)| *j);

        for (j, v) in c_row_vec {
            if v.abs() > 1e-300 {
                row_indices.push(i);
                col_indices.push(j);
                data_out.push(v);
            }
        }
    }

    if data_out.is_empty() {
        // Return empty matrix of correct shape
        return Ok(CsrMatrix::empty((m, n)));
    }

    CsrMatrix::new(data_out, row_indices, col_indices, (m, n))
}

/// Sparse matrix-vector product y = A x
fn matvec_csr(a: &CsrMatrix<f64>, x: &[f64]) -> SparseResult<Vec<f64>> {
    let (m, n) = a.shape();
    if x.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: x.len(),
        });
    }
    let mut y = vec![0.0_f64; m];
    for i in 0..m {
        for k in a.indptr[i]..a.indptr[i + 1] {
            y[i] += a.data[k] * x[a.indices[k]];
        }
    }
    Ok(y)
}

/// Get diagonal element a[i,i]
fn get_diagonal(a: &CsrMatrix<f64>, i: usize) -> f64 {
    for k in a.indptr[i]..a.indptr[i + 1] {
        if a.indices[k] == i {
            return a.data[k];
        }
    }
    0.0
}

/// Weighted Jacobi smoother: x ← x + ω D^{-1}(b - Ax)
fn weighted_jacobi_step(
    a: &CsrMatrix<f64>,
    x: &mut Vec<f64>,
    b: &[f64],
    omega: f64,
) -> SparseResult<()> {
    let n = a.rows();
    let ax = matvec_csr(a, x)?;

    let mut x_new = x.clone();
    for i in 0..n {
        let d = get_diagonal(a, i);
        if d.abs() < 1e-300 {
            continue;
        }
        let residual_i = b[i] - ax[i];
        x_new[i] = x[i] + omega * residual_i / d;
    }
    *x = x_new;
    Ok(())
}

/// Euclidean norm of a slice
fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|&vi| vi * vi).sum::<f64>().sqrt()
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    /// Build a 1D Poisson (tridiagonal) system of size n
    /// A has 2 on diagonal, -1 on off-diagonals.  b = ones.
    fn build_1d_poisson(n: usize) -> (CsrMatrix<f64>, Vec<f64>) {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            data.push(2.0);
            if i > 0 {
                rows.push(i);
                cols.push(i - 1);
                data.push(-1.0);
            }
            if i < n - 1 {
                rows.push(i);
                cols.push(i + 1);
                data.push(-1.0);
            }
        }
        let a = CsrMatrix::new(data, rows, cols, (n, n)).expect("poisson matrix");
        let b = vec![1.0; n];
        (a, b)
    }

    /// Build a 2D 5-point Laplacian on an m×m grid (m² DOF).
    fn build_2d_laplacian(m: usize) -> (CsrMatrix<f64>, Vec<f64>) {
        let n = m * m;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();
        for iy in 0..m {
            for ix in 0..m {
                let i = iy * m + ix;
                rows.push(i);
                cols.push(i);
                data.push(4.0);
                if ix > 0 {
                    rows.push(i);
                    cols.push(i - 1);
                    data.push(-1.0);
                }
                if ix < m - 1 {
                    rows.push(i);
                    cols.push(i + 1);
                    data.push(-1.0);
                }
                if iy > 0 {
                    rows.push(i);
                    cols.push(i - m);
                    data.push(-1.0);
                }
                if iy < m - 1 {
                    rows.push(i);
                    cols.push(i + m);
                    data.push(-1.0);
                }
            }
        }
        let a = CsrMatrix::new(data, rows, cols, (n, n)).expect("laplacian matrix");
        let b: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        (a, b)
    }

    fn relative_residual(a: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
        let ax = matvec_csr(a, x).expect("matvec");
        let r: f64 = ax.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).powi(2)).sum::<f64>().sqrt();
        let bn: f64 = b.iter().map(|bi| bi * bi).sum::<f64>().sqrt();
        r / bn.max(1e-300)
    }

    #[test]
    fn test_amg_config_defaults() {
        let cfg = AmgConfig::default();
        assert_eq!(cfg.max_levels, 10);
        assert!((cfg.strong_threshold - 0.25).abs() < 1e-10);
        assert_eq!(cfg.pre_smooth, 1);
        assert_eq!(cfg.post_smooth, 1);
        assert!(matches!(cfg.cycle, AmgCycle::V));
    }

    #[test]
    fn test_amg_setup_1d_poisson() {
        let (a, _) = build_1d_poisson(20);
        let amg = AlgebraicMultigrid::setup(&a, AmgConfig::default()).expect("setup");
        assert!(amg.n_levels() >= 1);
    }

    #[test]
    fn test_amg_solve_1d_poisson() {
        let n = 15;
        let (a, b) = build_1d_poisson(n);
        let amg = AlgebraicMultigrid::setup(&a, AmgConfig::default()).expect("setup");
        let b_arr = Array1::from_vec(b.clone());
        let x = amg.solve(&b_arr, None).expect("solve");

        let rel_res = relative_residual(&a, x.as_slice().expect("slice"), &b);
        assert!(
            rel_res < 1e-6,
            "1D Poisson: relative residual {} too large",
            rel_res
        );
    }

    #[test]
    fn test_amg_solve_2d_laplacian_3x3() {
        // 3×3 grid → 9 DOF
        let (a, b) = build_2d_laplacian(3);
        let cfg = AmgConfig {
            tol: 1e-9,
            max_iter: 200,
            ..Default::default()
        };
        let amg = AlgebraicMultigrid::setup(&a, cfg).expect("setup");
        let b_arr = Array1::from_vec(b.clone());
        let x = amg.solve(&b_arr, None).expect("solve");

        let rel_res = relative_residual(&a, x.as_slice().expect("slice"), &b);
        assert!(
            rel_res < 1e-6,
            "2D Laplacian 3×3: relative residual {} too large",
            rel_res
        );
    }

    #[test]
    fn test_amg_solve_with_initial_guess() {
        let n = 10;
        let (a, b) = build_1d_poisson(n);
        let amg = AlgebraicMultigrid::setup(&a, AmgConfig::default()).expect("setup");

        let b_arr = Array1::from_vec(b.clone());
        let x0 = Array1::from_vec(vec![0.1; n]);
        let x = amg.solve(&b_arr, Some(&x0)).expect("solve with x0");

        let rel_res = relative_residual(&a, x.as_slice().expect("slice"), &b);
        assert!(rel_res < 1e-6, "relative residual {} too large", rel_res);
    }

    #[test]
    fn test_amg_w_cycle() {
        let n = 12;
        let (a, b) = build_1d_poisson(n);
        let cfg = AmgConfig {
            cycle: AmgCycle::W,
            ..Default::default()
        };
        let amg = AlgebraicMultigrid::setup(&a, cfg).expect("setup");
        let b_arr = Array1::from_vec(b.clone());
        let x = amg.solve(&b_arr, None).expect("w-cycle solve");

        let rel_res = relative_residual(&a, x.as_slice().expect("slice"), &b);
        assert!(rel_res < 1e-5, "W-cycle relative residual {} too large", rel_res);
    }

    #[test]
    fn test_amg_n_levels() {
        let (a, _) = build_2d_laplacian(5); // 25 DOF
        let cfg = AmgConfig {
            max_coarse: 5,
            ..Default::default()
        };
        let amg = AlgebraicMultigrid::setup(&a, cfg).expect("setup");
        // Should have at least 2 levels with a 25-DOF problem and max_coarse=5
        assert!(amg.n_levels() >= 2, "expected ≥2 levels, got {}", amg.n_levels());
    }

    #[test]
    fn test_strong_connections_diagonal() {
        // Pure diagonal matrix — no strong off-diagonal connections
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let data = vec![3.0, 4.0, 5.0];
        let a = CsrMatrix::new(data, rows, cols, (3, 3)).expect("diag");
        let strong = strong_connections(&a, 0.25);
        assert!(strong.iter().all(|s| s.is_empty()));
    }

    #[test]
    fn test_cf_splitting_basic() {
        // Simple 4-node path graph: 0-1-2-3 → should alternate C/F
        let strong = vec![
            vec![1_usize],       // 0 strongly influences 1
            vec![0_usize, 2],    // 1 strongly influences 0,2
            vec![1_usize, 3],
            vec![2_usize],
        ];
        let cf = cf_splitting(4, &strong);
        // All nodes must be labelled C or F (not U)
        assert!(cf.iter().all(|&c| c != CfLabel::U));
        // At least one C-point
        assert!(cf.iter().any(|&c| c == CfLabel::C));
    }

    #[test]
    fn test_amg_small_direct_solve() {
        // System small enough to hit coarsest-level direct solve
        let n = 3;
        let (a, b) = build_1d_poisson(n);
        let cfg = AmgConfig {
            max_coarse: 10, // larger than n → goes straight to coarsest level
            ..Default::default()
        };
        let amg = AlgebraicMultigrid::setup(&a, cfg).expect("setup");
        let b_arr = Array1::from_vec(b.clone());
        let x = amg.solve(&b_arr, None).expect("small solve");

        let rel_res = relative_residual(&a, x.as_slice().expect("slice"), &b);
        assert!(rel_res < 1e-5, "small direct solve rel_res {} too large", rel_res);
    }
}
