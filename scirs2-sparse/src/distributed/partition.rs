//! Row partitioning for distributed sparse matrices.
//!
//! Provides [`partition_rows`] and [`create_distributed_csr`] for splitting a
//! [`CsrMatrix<f64>`] across multiple logical workers, including identification
//! of halo (ghost) rows needed for halo-exchange SpMV.

use std::collections::HashSet;

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};

// ─────────────────────────────────────────────────────────────────────────────
// PartitionMethod
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy used to assign rows to workers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum PartitionMethod {
    /// Contiguous blocks of rows: worker *i* owns rows `[i*n/p, (i+1)*n/p)`.
    #[default]
    Contiguous,
    /// Round-robin: row *r* goes to worker `r % p`.
    RoundRobin,
    /// Greedy by row NNZ: balance work by assigning rows greedily so that each
    /// worker gets approximately the same number of non-zeros.
    GraphBased,
}

// ─────────────────────────────────────────────────────────────────────────────
// PartitionConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for row partitioning.
#[derive(Debug, Clone)]
pub struct PartitionConfig {
    /// Number of workers (partitions) to create (default 4).
    pub n_workers: usize,
    /// Number of halo rows to include on each side (default 0 — halo exchange
    /// based on column references, not geometric proximity).
    pub overlap: usize,
    /// Row assignment strategy (default [`PartitionMethod::Contiguous`]).
    pub method: PartitionMethod,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            n_workers: 4,
            overlap: 0,
            method: PartitionMethod::Contiguous,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RowPartition
// ─────────────────────────────────────────────────────────────────────────────

/// Describes which rows a single worker owns.
#[derive(Debug, Clone)]
pub struct RowPartition {
    /// Worker identifier (0-based).
    pub worker_id: usize,
    /// Global row indices owned by this worker, in ascending order.
    pub local_rows: Vec<usize>,
    /// Total number of rows in the global matrix.
    pub n_global_rows: usize,
}

impl RowPartition {
    /// Number of rows owned by this worker.
    #[inline]
    pub fn n_local(&self) -> usize {
        self.local_rows.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DistributedCsr
// ─────────────────────────────────────────────────────────────────────────────

/// A worker's local view of a distributed CSR matrix.
///
/// `local_matrix` rows correspond to the global rows listed in
/// `partition.local_rows` (index 0 = `partition.local_rows[0]`, etc.).
/// `ghost_rows` holds the global row indices of off-partition rows referenced
/// by at least one non-zero in the local rows.
#[derive(Debug, Clone)]
pub struct DistributedCsr {
    /// Local CSR matrix (only owned rows; columns are global indices).
    pub local_matrix: CsrMatrix<f64>,
    /// Row ownership information.
    pub partition: RowPartition,
    /// Global row indices that appear as column targets in the local rows but
    /// are owned by other workers — i.e. halo / ghost rows.
    pub ghost_rows: Vec<usize>,
}

// ─────────────────────────────────────────────────────────────────────────────
// partition_rows
// ─────────────────────────────────────────────────────────────────────────────

/// Partition `n_rows` global rows across `config.n_workers` workers.
///
/// Returns a [`Vec<RowPartition>`] of length `config.n_workers`.
pub fn partition_rows(n_rows: usize, config: &PartitionConfig) -> Vec<RowPartition> {
    let p = config.n_workers.max(1);

    match config.method {
        PartitionMethod::Contiguous => (0..p)
            .map(|w| {
                let start = w * n_rows / p;
                let end = (w + 1) * n_rows / p;
                RowPartition {
                    worker_id: w,
                    local_rows: (start..end).collect(),
                    n_global_rows: n_rows,
                }
            })
            .collect(),
        PartitionMethod::RoundRobin => {
            let mut bins: Vec<Vec<usize>> = vec![Vec::new(); p];
            for r in 0..n_rows {
                bins[r % p].push(r);
            }
            bins.into_iter()
                .enumerate()
                .map(|(w, rows)| RowPartition {
                    worker_id: w,
                    local_rows: rows,
                    n_global_rows: n_rows,
                })
                .collect()
        }
        PartitionMethod::GraphBased => {
            // Partition_rows does not have NNZ information — return contiguous
            // blocks as a fallback; create_distributed_csr with GraphBased
            // re-balances by NNZ.
            (0..p)
                .map(|w| {
                    let start = w * n_rows / p;
                    let end = (w + 1) * n_rows / p;
                    RowPartition {
                        worker_id: w,
                        local_rows: (start..end).collect(),
                        n_global_rows: n_rows,
                    }
                })
                .collect()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// create_distributed_csr
// ─────────────────────────────────────────────────────────────────────────────

/// Build a [`DistributedCsr`] for one worker from the global CSR matrix.
///
/// The function extracts the rows listed in `partition.local_rows`, re-indexes
/// them into a compact local matrix (rows 0..n_local, global column indices
/// preserved), and identifies the ghost rows.
pub fn create_distributed_csr(
    global_matrix: &CsrMatrix<f64>,
    partition: &RowPartition,
) -> SparseResult<DistributedCsr> {
    let n_local = partition.local_rows.len();
    let n_cols = global_matrix.cols();
    let n_global_rows = global_matrix.rows();

    // Build a set of owned global rows for fast ghost detection.
    let owned_set: HashSet<usize> = partition.local_rows.iter().copied().collect();

    // Collect triplets for the local matrix and accumulate ghost rows.
    let mut row_indices: Vec<usize> = Vec::new();
    let mut col_indices: Vec<usize> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    let mut ghost_set: HashSet<usize> = HashSet::new();

    for (local_row, &global_row) in partition.local_rows.iter().enumerate() {
        if global_row >= n_global_rows {
            return Err(SparseError::ValueError(format!(
                "Global row {global_row} out of bounds (n_rows={n_global_rows})"
            )));
        }
        let row_start = global_matrix.indptr[global_row];
        let row_end = global_matrix.indptr[global_row + 1];

        for idx in row_start..row_end {
            let col = global_matrix.indices[idx];
            let val = global_matrix.data[idx];

            row_indices.push(local_row);
            col_indices.push(col);
            values.push(val);

            // column `col` corresponds to a row in the global matrix; if it is
            // outside the owned set it becomes a ghost row.
            if col < n_global_rows && !owned_set.contains(&col) {
                ghost_set.insert(col);
            }
        }
    }

    let local_matrix = CsrMatrix::from_triplets(n_local, n_cols, row_indices, col_indices, values)?;

    let mut ghost_rows: Vec<usize> = ghost_set.into_iter().collect();
    ghost_rows.sort_unstable();

    Ok(DistributedCsr {
        local_matrix,
        partition: partition.clone(),
        ghost_rows,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// NNZ-balanced partitioning helper (used by halo_exchange & dist_amg)
// ─────────────────────────────────────────────────────────────────────────────

/// Partition `global_matrix` by NNZ balance into `n_workers` pieces.
///
/// Returns [`DistributedCsr`] per worker.  Uses the `GraphBased` strategy
/// (greedy NNZ balance with contiguous row blocks for cache friendliness).
pub fn partition_matrix_nnz(
    global_matrix: &CsrMatrix<f64>,
    n_workers: usize,
) -> SparseResult<Vec<DistributedCsr>> {
    let n_rows = global_matrix.rows();
    let p = n_workers.max(1);

    // Compute per-row NNZ.
    let row_nnz: Vec<usize> = (0..n_rows)
        .map(|r| global_matrix.indptr[r + 1] - global_matrix.indptr[r])
        .collect();
    let total_nnz: usize = row_nnz.iter().sum();
    let target = (total_nnz + p - 1) / p;

    // Greedy contiguous block assignment.
    let mut partitions_rows: Vec<Vec<usize>> = vec![Vec::new(); p];
    let mut worker = 0usize;
    let mut acc = 0usize;

    for r in 0..n_rows {
        partitions_rows[worker].push(r);
        acc += row_nnz[r];
        if acc >= target && worker + 1 < p {
            worker += 1;
            acc = 0;
        }
    }

    let result: SparseResult<Vec<DistributedCsr>> = partitions_rows
        .into_iter()
        .enumerate()
        .map(|(w, rows)| {
            let rp = RowPartition {
                worker_id: w,
                local_rows: rows,
                n_global_rows: n_rows,
            };
            create_distributed_csr(global_matrix, &rp)
        })
        .collect();

    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 100×100 identity-like banded (tridiagonal) matrix.
    fn tridiag_100() -> CsrMatrix<f64> {
        let n = 100usize;
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
        CsrMatrix::from_triplets(n, n, rows, cols, vals).expect("tridiag_100 construction")
    }

    #[test]
    fn test_contiguous_row_count_sums_to_n() {
        let config = PartitionConfig {
            n_workers: 4,
            ..Default::default()
        };
        let parts = partition_rows(100, &config);
        assert_eq!(parts.len(), 4);
        let total: usize = parts.iter().map(|p| p.n_local()).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_contiguous_first_partition_rows() {
        let config = PartitionConfig {
            n_workers: 4,
            ..Default::default()
        };
        let parts = partition_rows(100, &config);
        // worker 0 should own rows [0..25)
        assert_eq!(parts[0].local_rows, (0..25).collect::<Vec<_>>());
        assert_eq!(parts[1].local_rows, (25..50).collect::<Vec<_>>());
        assert_eq!(parts[2].local_rows, (50..75).collect::<Vec<_>>());
        assert_eq!(parts[3].local_rows, (75..100).collect::<Vec<_>>());
    }

    #[test]
    fn test_round_robin_all_rows_assigned() {
        let config = PartitionConfig {
            n_workers: 4,
            method: PartitionMethod::RoundRobin,
            ..Default::default()
        };
        let parts = partition_rows(100, &config);
        let total: usize = parts.iter().map(|p| p.n_local()).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_create_distributed_csr_ghost_rows() {
        let mat = tridiag_100();
        let config = PartitionConfig {
            n_workers: 4,
            ..Default::default()
        };
        let partitions = partition_rows(100, &config);
        // Worker 1 owns rows [25..50].
        let dcsr =
            create_distributed_csr(&mat, &partitions[1]).expect("create_distributed_csr failed");
        // Ghost rows should include row 24 and row 50 (neighbours of boundary rows).
        assert!(
            dcsr.ghost_rows.contains(&24),
            "Expected row 24 as ghost, got {:?}",
            dcsr.ghost_rows
        );
        assert!(
            dcsr.ghost_rows.contains(&50),
            "Expected row 50 as ghost, got {:?}",
            dcsr.ghost_rows
        );
    }

    #[test]
    fn test_distributed_csr_local_matrix_nnz() {
        let mat = tridiag_100();
        let config = PartitionConfig {
            n_workers: 4,
            ..Default::default()
        };
        let partitions = partition_rows(100, &config);
        let dcsr =
            create_distributed_csr(&mat, &partitions[0]).expect("create_distributed_csr failed");
        // Worker 0 owns rows 0..25; interior rows have 3 nnz, boundary rows have 2.
        // Row 0: 2 nnz; rows 1..24: 3 nnz each; row 24: 3 nnz (has row 25 as ghost)
        // Actually row 24 (last in partition 0) still references col 25, so 3 nnz.
        // Row 0 references only (0,0) and (0,1) => 2 nnz.
        // Total: 2 + 23*3 + 3 = 2 + 69 + 3 = 74
        assert_eq!(dcsr.local_matrix.nnz(), 2 + 23 * 3 + 3);
    }

    #[test]
    fn test_partition_matrix_nnz_balanced() {
        let mat = tridiag_100();
        let dcsrs = partition_matrix_nnz(&mat, 4).expect("partition_matrix_nnz failed");
        assert_eq!(dcsrs.len(), 4);
        let total_rows: usize = dcsrs.iter().map(|d| d.partition.n_local()).sum();
        assert_eq!(total_rows, 100);
    }
}
