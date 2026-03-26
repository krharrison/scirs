//! Distributed Algebraic Multigrid (AMG) setup and V-cycle.
//!
//! Implements a distributed RS coarsening + direct interpolation AMG hierarchy
//! with a simulated (shared-memory) communication pattern.  In a real MPI
//! deployment the `simulate_*` steps would be replaced by actual point-to-point
//! or AllGather calls.

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};

use super::halo_exchange::distributed_spmv;
use super::partition::{DistributedCsr, RowPartition};

// ─────────────────────────────────────────────────────────────────────────────
// DistAMGConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the distributed AMG hierarchy.
#[derive(Debug, Clone)]
pub struct DistAMGConfig {
    /// Number of logical workers (default 4).
    pub n_workers: usize,
    /// Maximum number of AMG levels (default 4).
    pub max_levels: usize,
    /// Target coarsening ratio: stop coarsening when
    /// `n_coarse / n_fine >= coarsening_ratio` (default 0.25).
    pub coarsening_ratio: f64,
    /// Number of pre/post-smoother iterations on each level (default 2).
    pub smoother_iters: usize,
}

impl Default for DistAMGConfig {
    fn default() -> Self {
        Self {
            n_workers: 4,
            max_levels: 4,
            coarsening_ratio: 0.25,
            smoother_iters: 2,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DistAMGLevel
// ─────────────────────────────────────────────────────────────────────────────

/// A single level in the distributed AMG hierarchy.
#[derive(Debug, Clone)]
pub struct DistAMGLevel {
    /// The (local) fine-level matrix for this level, represented in global
    /// row/column numbering.
    pub local_matrix: CsrMatrix<f64>,
    /// Partition metadata for this level's rows.
    pub partition: RowPartition,
    /// Local prolongation operator P (n_fine_local × n_coarse).
    pub interpolation: CsrMatrix<f64>,
    /// Local restriction operator R = P^T (n_coarse × n_fine_local).
    pub restriction: CsrMatrix<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// DistAMGHierarchy
// ─────────────────────────────────────────────────────────────────────────────

/// The full distributed AMG hierarchy.
#[derive(Debug, Clone)]
pub struct DistAMGHierarchy {
    /// AMG levels from fine (index 0) to coarsest-1.
    pub levels: Vec<DistAMGLevel>,
    /// The coarsest-level matrix (solved exactly / directly).
    pub coarsest_matrix: CsrMatrix<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// distributed_rs_coarsening
// ─────────────────────────────────────────────────────────────────────────────

/// Perform Ruge-Stüben (RS) coarsening independently on each partition.
///
/// Returns a `Vec<Vec<bool>>` (one per worker) where `true` means the row is
/// a *coarse* point and `false` means it is a *fine* point.
///
/// Algorithm per worker:
/// 1. Compute the row-wise maximum off-diagonal |a_ij|.
/// 2. Mark a connection as *strong* if |a_{ij}| ≥ 0.25 * max_j|a_ij|.
/// 3. RS pass 1: greedily mark a local row as C if it has at least one strong
///    connection to an undecided row; mark neighbours of new C-points as F.
pub fn distributed_rs_coarsening(partitions: &[DistributedCsr]) -> Vec<Vec<bool>> {
    partitions
        .iter()
        .map(|dcsr| rs_coarsen_local(&dcsr.local_matrix))
        .collect()
}

/// RS coarsening for a single local matrix (in local row indices).
fn rs_coarsen_local(mat: &CsrMatrix<f64>) -> Vec<bool> {
    let n = mat.rows();
    // is_coarse[i]: None = undecided, Some(true) = C, Some(false) = F
    let mut status: Vec<Option<bool>> = vec![None; n];

    // Step 1: compute per-row max |off-diagonal|.
    let max_off_diag: Vec<f64> = (0..n)
        .map(|i| {
            let start = mat.indptr[i];
            let end = mat.indptr[i + 1];
            let mut m = 0.0_f64;
            for k in start..end {
                if mat.indices[k] != i {
                    m = m.max(mat.data[k].abs());
                }
            }
            m
        })
        .collect();

    // Step 2: greedy C/F selection.
    for i in 0..n {
        if status[i].is_some() {
            continue;
        }
        let start = mat.indptr[i];
        let end = mat.indptr[i + 1];
        let threshold = 0.25 * max_off_diag[i];

        // Check whether row i has any strong connection to an undecided node.
        // Note: column indices may be global (>= n) for off-partition entries;
        // those are treated as always-undecided (None) for the local C/F decision.
        let has_strong = (start..end).any(|k| {
            let j = mat.indices[k];
            if j == i {
                return false;
            }
            let strong_val = mat.data[k].abs() >= threshold;
            let neighbour_status = if j < n { status[j] } else { None };
            strong_val && neighbour_status != Some(false)
        });

        if has_strong {
            status[i] = Some(true); // C-point
                                    // Mark strongly-connected undecided neighbours as F.
            for k in start..end {
                let j = mat.indices[k];
                if j < n && j != i && status[j].is_none() && mat.data[k].abs() >= threshold {
                    status[j] = Some(false);
                }
            }
        } else {
            status[i] = Some(true); // isolated → C-point
        }
    }

    status.into_iter().map(|s| s.unwrap_or(true)).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// build_distributed_interpolation
// ─────────────────────────────────────────────────────────────────────────────

/// Build local prolongation matrices P for each partition.
///
/// Direct interpolation formula: for each fine point *i* and each strong
/// coarse neighbour *j*:
///   w_{ij} = -a_{ij} / (a_{ii} * Σ_{k∈C_i} a_{ik} / a_{kk})
///
/// If a fine point has no coarse neighbours it is treated as a trivially
/// prolonged point (zero interpolation — it will be smoothed away).
pub fn build_distributed_interpolation(
    partitions: &[DistributedCsr],
    coarse_masks: &[Vec<bool>],
) -> SparseResult<Vec<CsrMatrix<f64>>> {
    if partitions.len() != coarse_masks.len() {
        return Err(SparseError::DimensionMismatch {
            expected: partitions.len(),
            found: coarse_masks.len(),
        });
    }

    partitions
        .iter()
        .zip(coarse_masks.iter())
        .map(|(dcsr, mask)| build_local_interpolation(&dcsr.local_matrix, mask))
        .collect()
}

/// Build P for one local matrix (local row/column indexing).
fn build_local_interpolation(
    mat: &CsrMatrix<f64>,
    coarse_mask: &[bool],
) -> SparseResult<CsrMatrix<f64>> {
    let n = mat.rows();

    // Assign compact coarse indices.
    let mut coarse_idx: Vec<Option<usize>> = vec![None; n];
    let mut n_coarse = 0usize;
    for (i, &is_c) in coarse_mask.iter().enumerate() {
        if is_c {
            coarse_idx[i] = Some(n_coarse);
            n_coarse += 1;
        }
    }

    if n_coarse == 0 {
        // No coarse points — return empty P.
        return CsrMatrix::from_triplets(n, 1, vec![], vec![], vec![]);
    }

    // Pre-compute diagonal.
    let diagonal: Vec<f64> = (0..n)
        .map(|i| {
            let start = mat.indptr[i];
            let end = mat.indptr[i + 1];
            (start..end)
                .find(|&k| mat.indices[k] == i)
                .map(|k| mat.data[k])
                .unwrap_or(1.0)
        })
        .collect();

    let mut p_rows: Vec<usize> = Vec::new();
    let mut p_cols: Vec<usize> = Vec::new();
    let mut p_vals: Vec<f64> = Vec::new();

    for i in 0..n {
        if coarse_mask[i] {
            // C-point: P[i, coarse_idx[i]] = 1.
            p_rows.push(i);
            p_cols.push(coarse_idx[i].unwrap_or(0));
            p_vals.push(1.0);
        } else {
            // F-point: direct interpolation from strong coarse neighbours.
            let start = mat.indptr[i];
            let end = mat.indptr[i + 1];
            let a_ii = diagonal[i];
            if a_ii.abs() < f64::EPSILON * 1e6 {
                continue; // degenerate row — skip
            }

            // Collect strong coarse neighbours (threshold = 0.25 * max_off_diag).
            let max_off = (start..end)
                .filter(|&k| mat.indices[k] != i)
                .map(|k| mat.data[k].abs())
                .fold(0.0_f64, f64::max);
            let threshold = 0.25 * max_off;

            let coarse_nbrs: Vec<(usize, f64)> = (start..end)
                .filter_map(|k| {
                    let j = mat.indices[k];
                    if j < n && j != i && coarse_mask[j] && mat.data[k].abs() >= threshold {
                        Some((j, mat.data[k]))
                    } else {
                        None
                    }
                })
                .collect();

            if coarse_nbrs.is_empty() {
                continue;
            }

            // Σ_{k∈C_i} a_{ik} / a_{kk}
            let sum_ratio: f64 = coarse_nbrs
                .iter()
                .map(|&(j, a_ij)| {
                    let a_jj = diagonal[j];
                    if a_jj.abs() < f64::EPSILON * 1e6 {
                        0.0
                    } else {
                        a_ij / a_jj
                    }
                })
                .sum();

            let denom = if sum_ratio.abs() < f64::EPSILON * 1e6 {
                1.0
            } else {
                a_ii * sum_ratio
            };

            for (j, a_ij) in coarse_nbrs {
                let w = -a_ij / denom;
                if let Some(ci) = coarse_idx[j] {
                    p_rows.push(i);
                    p_cols.push(ci);
                    p_vals.push(w);
                }
            }
        }
    }

    CsrMatrix::from_triplets(n, n_coarse.max(1), p_rows, p_cols, p_vals)
}

// ─────────────────────────────────────────────────────────────────────────────
// Coarse matrix: R A P
// ─────────────────────────────────────────────────────────────────────────────

/// Compute A_c = R * A * P where R = P^T.
///
/// All matrices use local row/column indices.
fn triple_product(a: &CsrMatrix<f64>, p: &CsrMatrix<f64>) -> SparseResult<CsrMatrix<f64>> {
    // R = P^T  (n_coarse × n_fine)
    let r = p.transpose();
    // B = A * P  (n_fine × n_coarse)
    let b = sparse_matmul(a, p)?;
    // A_c = R * B  (n_coarse × n_coarse)
    sparse_matmul(&r, &b)
}

/// Sparse matrix multiplication C = A * B (all in CSR, T=f64).
fn sparse_matmul(a: &CsrMatrix<f64>, b: &CsrMatrix<f64>) -> SparseResult<CsrMatrix<f64>> {
    let (m, k_a) = a.shape();
    let (k_b, n) = b.shape();
    if k_a != k_b {
        return Err(SparseError::DimensionMismatch {
            expected: k_a,
            found: k_b,
        });
    }

    // Dense temporary row accumulator.
    let mut c_rows: Vec<usize> = Vec::new();
    let mut c_cols: Vec<usize> = Vec::new();
    let mut c_vals: Vec<f64> = Vec::new();
    let mut row_buf: Vec<f64> = vec![0.0; n];
    let mut nz_cols: Vec<usize> = Vec::new();

    for i in 0..m {
        let a_start = a.indptr[i];
        let a_end = a.indptr[i + 1];

        nz_cols.clear();

        for ka in a_start..a_end {
            let ka_col = a.indices[ka];
            let a_val = a.data[ka];
            let b_start = b.indptr[ka_col];
            let b_end = b.indptr[ka_col + 1];
            for kb in b_start..b_end {
                let j = b.indices[kb];
                if row_buf[j] == 0.0 {
                    nz_cols.push(j);
                }
                row_buf[j] += a_val * b.data[kb];
            }
        }

        nz_cols.sort_unstable();
        for &j in &nz_cols {
            let v = row_buf[j];
            if v.abs() > f64::EPSILON * 1e-3 {
                c_rows.push(i);
                c_cols.push(j);
                c_vals.push(v);
            }
            row_buf[j] = 0.0; // reset
        }
    }

    CsrMatrix::from_triplets(m, n, c_rows, c_cols, c_vals)
}

// ─────────────────────────────────────────────────────────────────────────────
// build_distributed_amg
// ─────────────────────────────────────────────────────────────────────────────

/// Build the full distributed AMG hierarchy from an initial set of partitions.
pub fn build_distributed_amg(
    partitions: &[DistributedCsr],
    config: &DistAMGConfig,
) -> SparseResult<DistAMGHierarchy> {
    if partitions.is_empty() {
        return Err(SparseError::ValueError(
            "Cannot build AMG hierarchy from empty partition list".to_string(),
        ));
    }

    // Assemble the global fine matrix from partitions.
    let global_fine = assemble_global_matrix(partitions)?;

    let mut levels: Vec<DistAMGLevel> = Vec::new();
    let mut current_mat = global_fine;

    for _lvl in 0..config.max_levels.saturating_sub(1) {
        let n = current_mat.rows();
        if n <= 4 {
            break;
        }

        // RS coarsening on local (single-worker) view.
        let coarse_mask = rs_coarsen_local(&current_mat);
        let n_coarse = coarse_mask.iter().filter(|&&c| c).count();

        // Stop if coarsening ratio not achieved.
        if n_coarse == 0 || (n_coarse as f64) / (n as f64) > config.coarsening_ratio + 0.05 {
            break;
        }

        // Build P, R, coarse matrix.
        let p = build_local_interpolation(&current_mat, &coarse_mask)?;
        let a_c = triple_product(&current_mat, &p)?;

        // Wrap in DistAMGLevel (single global partition for simplicity).
        let partition = RowPartition {
            worker_id: 0,
            local_rows: (0..n).collect(),
            n_global_rows: n,
        };
        let r = p.transpose();
        levels.push(DistAMGLevel {
            local_matrix: current_mat,
            partition,
            interpolation: p,
            restriction: r,
        });

        current_mat = a_c;
    }

    Ok(DistAMGHierarchy {
        levels,
        coarsest_matrix: current_mat,
    })
}

/// Assemble the global CSR matrix by gathering all owned rows from partitions.
fn assemble_global_matrix(partitions: &[DistributedCsr]) -> SparseResult<CsrMatrix<f64>> {
    if partitions.is_empty() {
        return CsrMatrix::from_triplets(0, 0, vec![], vec![], vec![]);
    }

    let n_global = partitions[0].partition.n_global_rows;
    let n_cols = partitions
        .iter()
        .map(|d| d.local_matrix.cols())
        .max()
        .unwrap_or(n_global);

    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();

    for dcsr in partitions {
        let mat = &dcsr.local_matrix;
        for (local_row, &global_row) in dcsr.partition.local_rows.iter().enumerate() {
            let start = mat.indptr[local_row];
            let end = mat.indptr[local_row + 1];
            for k in start..end {
                rows.push(global_row);
                cols.push(mat.indices[k]);
                vals.push(mat.data[k]);
            }
        }
    }

    CsrMatrix::from_triplets(n_global, n_cols, rows, cols, vals)
}

// ─────────────────────────────────────────────────────────────────────────────
// dist_vcycle
// ─────────────────────────────────────────────────────────────────────────────

/// Perform one AMG V-cycle: pre-smooth → restrict → coarse solve → prolongate
/// → post-smooth.
///
/// Returns the approximate solution `x` given right-hand side `rhs`.
pub fn dist_vcycle(
    hierarchy: &DistAMGHierarchy,
    rhs: &[f64],
    config: &DistAMGConfig,
) -> SparseResult<Vec<f64>> {
    if hierarchy.levels.is_empty() {
        // Only coarsest level — direct solve (Jacobi).
        return jacobi_solve(&hierarchy.coarsest_matrix, rhs, 50);
    }

    vcycle_recursive(hierarchy, 0, rhs, config)
}

/// Recursive V-cycle implementation.
fn vcycle_recursive(
    hierarchy: &DistAMGHierarchy,
    level: usize,
    rhs: &[f64],
    config: &DistAMGConfig,
) -> SparseResult<Vec<f64>> {
    let n = rhs.len();

    if level >= hierarchy.levels.len() {
        // Coarsest level — direct (Jacobi) solve.
        return jacobi_solve(&hierarchy.coarsest_matrix, rhs, 50);
    }

    let lvl = &hierarchy.levels[level];
    let mat = &lvl.local_matrix;

    // ── Pre-smooth ────────────────────────────────────────────────────────────
    let mut x = jacobi_smooth(mat, rhs, &vec![0.0; n], config.smoother_iters)?;

    // ── Compute residual r = rhs - A*x ───────────────────────────────────────
    let ax = mat.dot(&x)?;
    let residual: Vec<f64> = rhs
        .iter()
        .zip(ax.iter())
        .map(|(&b, &ax_i)| b - ax_i)
        .collect();

    // ── Restrict: r_c = R * r ────────────────────────────────────────────────
    let r = &lvl.restriction;
    let rhs_coarse = csr_matvec(r, &residual)?;

    // ── Coarse correction ─────────────────────────────────────────────────────
    let e_coarse = vcycle_recursive(hierarchy, level + 1, &rhs_coarse, config)?;

    // ── Prolongate: x += P * e_c ─────────────────────────────────────────────
    let p = &lvl.interpolation;
    let e_fine = csr_matvec(p, &e_coarse)?;

    for (xi, ei) in x.iter_mut().zip(e_fine.iter()) {
        *xi += ei;
    }

    // ── Post-smooth ───────────────────────────────────────────────────────────
    x = jacobi_smooth(mat, rhs, &x, config.smoother_iters)?;

    Ok(x)
}

// ─────────────────────────────────────────────────────────────────────────────
// Smoother helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Jacobi smoother: x_new[i] = (b[i] - Σ_{j≠i} a_{ij}*x[j]) / a_{ii}
fn jacobi_smooth(
    mat: &CsrMatrix<f64>,
    rhs: &[f64],
    x0: &[f64],
    iters: usize,
) -> SparseResult<Vec<f64>> {
    let n = mat.rows();
    let mut x = x0.to_vec();

    // Pre-compute diagonal.
    let diag: Vec<f64> = (0..n)
        .map(|i| {
            let start = mat.indptr[i];
            let end = mat.indptr[i + 1];
            (start..end)
                .find(|&k| mat.indices[k] == i)
                .map(|k| mat.data[k])
                .unwrap_or(1.0)
        })
        .collect();

    for _ in 0..iters {
        let ax = mat.dot(&x)?;
        for i in 0..n {
            let d = diag[i];
            if d.abs() > f64::EPSILON * 1e6 {
                let off_diag = ax[i] - d * x[i];
                x[i] = (rhs[i] - off_diag) / d;
            }
        }
    }
    Ok(x)
}

/// Simple Jacobi iterative solve (used at coarsest level).
fn jacobi_solve(mat: &CsrMatrix<f64>, rhs: &[f64], iters: usize) -> SparseResult<Vec<f64>> {
    jacobi_smooth(mat, rhs, &vec![0.0; rhs.len()], iters)
}

/// Dense CSR matrix-vector product (for small operators).
fn csr_matvec(mat: &CsrMatrix<f64>, x: &[f64]) -> SparseResult<Vec<f64>> {
    if x.len() < mat.cols() {
        return Err(SparseError::DimensionMismatch {
            expected: mat.cols(),
            found: x.len(),
        });
    }
    let n = mat.rows();
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let start = mat.indptr[i];
        let end = mat.indptr[i + 1];
        let mut acc = 0.0_f64;
        for k in start..end {
            let j = mat.indices[k];
            if j < x.len() {
                acc += mat.data[k] * x[j];
            }
        }
        y[i] = acc;
    }
    Ok(y)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::partition::{create_distributed_csr, partition_rows, PartitionConfig};

    /// Build a symmetric positive definite tridiagonal n×n matrix.
    /// Diagonal = 2, off-diagonal = -1.
    fn tridiag(n: usize) -> CsrMatrix<f64> {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(2.0_f64);
            if i > 0 {
                rows.push(i);
                cols.push(i - 1);
                vals.push(-1.0);
                rows.push(i - 1);
                cols.push(i);
                vals.push(-1.0);
            }
        }
        CsrMatrix::from_triplets(n, n, rows, cols, vals).expect("tridiag")
    }

    fn make_partitions(mat: &CsrMatrix<f64>, n_workers: usize) -> Vec<DistributedCsr> {
        let config = PartitionConfig {
            n_workers,
            ..Default::default()
        };
        let rps = partition_rows(mat.rows(), &config);
        rps.iter()
            .map(|rp| create_distributed_csr(mat, rp).expect("create_distributed_csr"))
            .collect()
    }

    #[test]
    fn test_rs_coarsening_reduces_size() {
        let n = 20;
        let mat = tridiag(n);
        let parts = make_partitions(&mat, 2);
        let masks = distributed_rs_coarsening(&parts);

        assert_eq!(masks.len(), 2);
        // Each worker's mask should have fewer C-points than total rows.
        for mask in &masks {
            let n_coarse = mask.iter().filter(|&&c| c).count();
            let n_fine_local = mask.len();
            assert!(
                n_coarse < n_fine_local,
                "Expected coarsening; got n_coarse={n_coarse} of {n_fine_local}"
            );
        }
    }

    #[test]
    fn test_build_amg_two_level() {
        let n = 20;
        let mat = tridiag(n);
        let parts = make_partitions(&mat, 2);
        let config = DistAMGConfig {
            n_workers: 2,
            max_levels: 2,
            coarsening_ratio: 0.6, // slightly relaxed for n=20
            smoother_iters: 1,
        };
        let hierarchy =
            build_distributed_amg(&parts, &config).expect("build_distributed_amg failed");

        // Should have at least one level.
        assert!(
            !hierarchy.levels.is_empty(),
            "Expected at least one AMG level"
        );

        // Coarsest matrix should be smaller than fine level.
        let n_fine = hierarchy.levels[0].local_matrix.rows();
        let n_coarse = hierarchy.coarsest_matrix.rows();
        assert!(
            n_coarse < n_fine,
            "Coarsest ({n_coarse}) should be smaller than fine ({n_fine})"
        );
    }

    #[test]
    fn test_vcycle_reduces_residual() {
        let n = 20;
        let mat = tridiag(n);
        let rhs: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        let parts = make_partitions(&mat, 2);
        let config = DistAMGConfig {
            n_workers: 2,
            max_levels: 3,
            coarsening_ratio: 0.7,
            smoother_iters: 2,
        };

        let hierarchy =
            build_distributed_amg(&parts, &config).expect("build_distributed_amg failed");

        let x = dist_vcycle(&hierarchy, &rhs, &config).expect("dist_vcycle failed");

        // Compute residual ||b - Ax||.
        let ax = if hierarchy.levels.is_empty() {
            hierarchy.coarsest_matrix.dot(&x).expect("coarsest dot")
        } else {
            hierarchy.levels[0]
                .local_matrix
                .dot(&x)
                .expect("level 0 dot")
        };

        let residual_norm: f64 = rhs
            .iter()
            .zip(ax.iter())
            .map(|(&b, &ax_i)| (b - ax_i).powi(2))
            .sum::<f64>()
            .sqrt();

        let rhs_norm: f64 = rhs.iter().map(|&b| b * b).sum::<f64>().sqrt();
        let relative = residual_norm / rhs_norm;

        assert!(
            relative < 1.0,
            "V-cycle should reduce relative residual below 1.0; got {relative}"
        );
    }

    #[test]
    fn test_sparse_matmul_identity() {
        // A * I = A
        let n = 5;
        let mat = tridiag(n);
        // Build n×n identity in CSR.
        let i_rows: Vec<usize> = (0..n).collect();
        let i_cols: Vec<usize> = (0..n).collect();
        let i_vals: Vec<f64> = vec![1.0; n];
        let identity = CsrMatrix::from_triplets(n, n, i_rows, i_cols, i_vals).expect("identity");

        let result = sparse_matmul(&mat, &identity).expect("matmul");

        for i in 0..n {
            for j in 0..n {
                let expected = mat.get(i, j);
                let got = result.get(i, j);
                assert!(
                    (expected - got).abs() < 1e-10,
                    "mismatch at ({i},{j}): {expected} vs {got}"
                );
            }
        }
    }

    #[test]
    fn test_build_interpolation_coarse_points_identity() {
        let n = 6;
        let mat = tridiag(n);
        // Manually coarsen: every even row is C.
        let coarse_mask: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
        let p = build_local_interpolation(&mat, &coarse_mask).expect("build_local_interpolation");
        // Each C-point should map to exactly one coarse column with weight 1.
        for (i, &is_c) in coarse_mask.iter().enumerate() {
            if is_c {
                let start = p.indptr[i];
                let end = p.indptr[i + 1];
                assert_eq!(end - start, 1, "C-point {i} should have exactly 1 entry");
                assert!(
                    (p.data[start] - 1.0).abs() < 1e-10,
                    "C-point {i} interpolation weight should be 1.0"
                );
            }
        }
    }
}
