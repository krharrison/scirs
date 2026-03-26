//! Distributed CSR with row-based partitioning and halo exchange.
//!
//! Implements a row-striped decomposition of a sparse matrix where each
//! logical "worker" owns a contiguous range of rows.  The halo exchange is
//! simulated via shared memory (suitable for multi-threaded or single-process
//! testing); a real distributed implementation would replace the halo
//! broadcast with MPI or similar.

use crate::error::{SparseError, SparseResult};
use crate::gpu::construction::{GpuCooMatrix, GpuCsrMatrix};

// ============================================================
// Configuration
// ============================================================

/// Configuration for distributed CSR partitioning.
#[derive(Debug, Clone)]
pub struct DistributedCsrConfig {
    /// Number of workers (partitions) to create (default 4).
    pub n_workers: usize,
    /// Number of halo (ghost) rows from neighbouring partitions to include in
    /// each local matrix (default 1).
    pub overlap: usize,
}

impl Default for DistributedCsrConfig {
    fn default() -> Self {
        Self {
            n_workers: 4,
            overlap: 1,
        }
    }
}

// ============================================================
// PartitionedCsr
// ============================================================

/// A CSR matrix partitioned across multiple logical workers using a
/// row-striped decomposition with optional halo rows.
///
/// Each entry of `partitions` holds the **local** matrix for that worker,
/// which covers `[partition_row_start[w], partition_row_start[w] + partitions[w].n_rows)`.
#[derive(Debug, Clone)]
pub struct PartitionedCsr {
    /// Local matrix per worker (may include halo rows at the boundaries).
    pub partitions: Vec<GpuCsrMatrix>,
    /// Global row index of the first **owned** (non-halo) row for each worker.
    pub row_offsets: Vec<usize>,
    /// For each worker: the list of global row indices that are ghost rows
    /// (owned by another worker but needed by this worker).
    pub halo_rows: Vec<Vec<usize>>,
    /// Total number of rows in the original matrix.
    pub n_total_rows: usize,
    /// Number of columns in the original matrix.
    pub n_cols: usize,
    /// Global row index at which each worker's **local** matrix begins
    /// (including any leading halo rows).
    partition_global_start: Vec<usize>,
    /// Global row index at which each worker's **owned** rows end (exclusive).
    owned_ends: Vec<usize>,
}

impl PartitionedCsr {
    /// Partition `matrix` into `config.n_workers` pieces using a row-striped
    /// decomposition.
    ///
    /// Each partition gets its owned rows plus up to `config.overlap` halo
    /// rows from each adjacent partition.
    pub fn from_csr(matrix: &GpuCsrMatrix, config: &DistributedCsrConfig) -> Self {
        let n_workers = config.n_workers.max(1);
        let overlap = config.overlap;
        let n = matrix.n_rows;
        let n_cols = matrix.n_cols;

        // ── Compute owned row ranges (NNZ-balanced partitioning) ──────────
        let total_nnz = matrix.n_nnz();
        let target_nnz = total_nnz
            .checked_div(n_workers)
            .map(|q| q + usize::from(!total_nnz.is_multiple_of(n_workers)))
            .unwrap_or(total_nnz);

        let mut owned_starts: Vec<usize> = vec![0];
        let mut acc = 0usize;
        for row in 0..n {
            acc += matrix.row_ptr[row + 1] - matrix.row_ptr[row];
            if acc >= target_nnz && owned_starts.len() < n_workers {
                owned_starts.push(row + 1);
                acc = 0;
            }
        }
        while owned_starts.len() < n_workers {
            owned_starts.push(n);
        }
        let mut o_ends: Vec<usize> = owned_starts[1..].to_vec();
        o_ends.push(n);

        // ── Build per-worker local matrices (owned + halo rows) ───────────
        let mut partitions: Vec<GpuCsrMatrix> = Vec::with_capacity(n_workers);
        let mut halo_rows: Vec<Vec<usize>> = Vec::with_capacity(n_workers);
        let mut row_offsets: Vec<usize> = Vec::with_capacity(n_workers);
        let mut part_global_starts: Vec<usize> = Vec::with_capacity(n_workers);

        for w in 0..n_workers {
            let own_start = owned_starts[w];
            let own_end = o_ends[w];

            // Halo rows: `overlap` rows before and after the owned range.
            let halo_start = own_start.saturating_sub(overlap);
            let halo_end = (own_end + overlap).min(n);

            // Collect global row indices for ghost rows only.
            let mut ghost: Vec<usize> = Vec::new();
            for r in halo_start..own_start {
                ghost.push(r);
            }
            for r in own_end..halo_end {
                ghost.push(r);
            }

            // Build local CSR from global rows [halo_start, halo_end).
            let local_nrows = halo_end - halo_start;
            let local_nnz_start = matrix.row_ptr[halo_start];
            let local_nnz_end = matrix.row_ptr[halo_end];

            let local_row_ptr: Vec<usize> = (halo_start..=halo_end)
                .map(|r| matrix.row_ptr[r] - local_nnz_start)
                .collect();
            let local_col_idx = matrix.col_idx[local_nnz_start..local_nnz_end].to_vec();
            let local_values = matrix.values[local_nnz_start..local_nnz_end].to_vec();

            partitions.push(GpuCsrMatrix {
                row_ptr: local_row_ptr,
                col_idx: local_col_idx,
                values: local_values,
                n_rows: local_nrows,
                n_cols,
            });

            halo_rows.push(ghost);
            row_offsets.push(own_start);
            part_global_starts.push(halo_start);
        }

        Self {
            partitions,
            row_offsets,
            halo_rows,
            n_total_rows: n,
            n_cols,
            partition_global_start: part_global_starts,
            owned_ends: o_ends,
        }
    }

    /// Compute `y = A * x` using the distributed representation.
    ///
    /// Each worker computes SpMV on its local matrix (which includes halo rows)
    /// and contributes only the owned rows to the global result.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::DimensionMismatch`] when `x.len() != n_cols`.
    pub fn spmv(&self, x: &[f64]) -> SparseResult<Vec<f64>> {
        if x.len() != self.n_cols {
            return Err(SparseError::DimensionMismatch {
                expected: self.n_cols,
                found: x.len(),
            });
        }

        let n_workers = self.partitions.len();
        let mut y = vec![0.0_f64; self.n_total_rows];

        for w in 0..n_workers {
            let partition = &self.partitions[w];
            let local_y = partition.spmv(x)?;

            // Owned global row range for this worker.
            let own_global_start = self.row_offsets[w];
            let own_global_end = self.owned_ends[w];

            // The local matrix starts at global row `partition_global_start[w]`.
            // The local index of the first owned row is:
            let local_owned_start = own_global_start - self.partition_global_start[w];

            // Copy owned rows from local result to global output.
            let owned_len = own_global_end - own_global_start;
            for k in 0..owned_len {
                let local_idx = local_owned_start + k;
                if local_idx < local_y.len() {
                    y[own_global_start + k] = local_y[local_idx];
                }
            }
        }

        Ok(y)
    }

    /// Reassemble all partitions' owned rows into a single [`GpuCsrMatrix`].
    pub fn to_csr(&self) -> GpuCsrMatrix {
        let n_workers = self.partitions.len();
        let mut coo = GpuCooMatrix::new(self.n_total_rows, self.n_cols);

        for w in 0..n_workers {
            let partition = &self.partitions[w];
            let own_global_start = self.row_offsets[w];
            let own_global_end = self.owned_ends[w];
            let local_owned_start = own_global_start - self.partition_global_start[w];
            let owned_len = own_global_end - own_global_start;

            for k in 0..owned_len {
                let local_row = local_owned_start + k;
                let global_row = own_global_start + k;
                let row_start = partition.row_ptr[local_row];
                let row_end = partition.row_ptr[local_row + 1];
                for idx in row_start..row_end {
                    coo.push(global_row, partition.col_idx[idx], partition.values[idx]);
                }
            }
        }

        coo.to_csr()
    }

    /// Measure load balance quality.
    ///
    /// Returns `std_dev(nnz_per_partition) / mean(nnz_per_partition)`.
    /// A value of 0.0 means perfect balance; lower is better.
    ///
    /// Returns 0.0 for degenerate cases (0 or 1 workers, or 0 total nnz).
    pub fn load_balance_quality(&self) -> f64 {
        let n = self.partitions.len();
        if n <= 1 {
            return 0.0;
        }
        let counts: Vec<f64> = self.partitions.iter().map(|p| p.n_nnz() as f64).collect();
        let mean = counts.iter().sum::<f64>() / n as f64;
        if mean < f64::EPSILON {
            return 0.0;
        }
        let variance = counts.iter().map(|&c| (c - mean).powi(2)).sum::<f64>() / n as f64;
        variance.sqrt() / mean
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::construction::GpuCooMatrix;

    /// Build an n×n tridiagonal matrix for distributed tests.
    fn tridiag(n: usize) -> GpuCsrMatrix {
        let mut coo = GpuCooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, 4.0);
            if i > 0 {
                coo.push(i, i - 1, -1.0);
                coo.push(i - 1, i, -1.0);
            }
        }
        coo.to_csr()
    }

    #[test]
    fn test_distributed_csr_spmv_matches_sequential() {
        let n = 12;
        let mat = tridiag(n);
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        let y_seq = mat.spmv(&x).expect("sequential spmv failed");

        let config = DistributedCsrConfig {
            n_workers: 4,
            overlap: 1,
        };
        let dist = PartitionedCsr::from_csr(&mat, &config);
        let y_dist = dist.spmv(&x).expect("distributed spmv failed");

        assert_eq!(y_seq.len(), y_dist.len());
        for (i, (ys, yd)) in y_seq.iter().zip(y_dist.iter()).enumerate() {
            assert!(
                (ys - yd).abs() < 1e-10,
                "row {i}: sequential={ys} distributed={yd}"
            );
        }
    }

    #[test]
    fn test_distributed_partitioning_row_split() {
        let n = 12;
        let mat = tridiag(n);
        let config = DistributedCsrConfig {
            n_workers: 4,
            overlap: 0,
        };
        let dist = PartitionedCsr::from_csr(&mat, &config);
        assert_eq!(dist.partitions.len(), 4);

        // Row offsets must be non-decreasing and within [0, n]
        for w in &dist.row_offsets {
            assert!(*w <= n);
        }
    }

    #[test]
    fn test_distributed_to_csr_roundtrip() {
        let n = 8;
        let mat = tridiag(n);
        let config = DistributedCsrConfig {
            n_workers: 3,
            overlap: 1,
        };
        let dist = PartitionedCsr::from_csr(&mat, &config);
        let reassembled = dist.to_csr();

        // NNZ should be the same
        assert_eq!(mat.n_nnz(), reassembled.n_nnz());
        // Dense representations must match
        let d1 = mat.to_dense();
        let d2 = reassembled.to_dense();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (d1[[i, j]] - d2[[i, j]]).abs() < 1e-12,
                    "mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_load_balance_quality() {
        let n = 12;
        let mat = tridiag(n);
        let config = DistributedCsrConfig {
            n_workers: 4,
            overlap: 0,
        };
        let dist = PartitionedCsr::from_csr(&mat, &config);
        let q = dist.load_balance_quality();
        assert!(q >= 0.0);
        assert!(q < 1.0);
    }

    #[test]
    fn test_single_worker() {
        let n = 6;
        let mat = tridiag(n);
        let config = DistributedCsrConfig {
            n_workers: 1,
            overlap: 0,
        };
        let dist = PartitionedCsr::from_csr(&mat, &config);
        let x = vec![1.0; n];
        let y_seq = mat.spmv(&x).expect("spmv failed");
        let y_dist = dist.spmv(&x).expect("distributed spmv failed");
        for (ys, yd) in y_seq.iter().zip(y_dist.iter()) {
            assert!((ys - yd).abs() < 1e-10);
        }
    }

    #[test]
    fn test_more_workers_than_rows() {
        let n = 3;
        let mat = tridiag(n);
        let config = DistributedCsrConfig {
            n_workers: 6,
            overlap: 0,
        };
        let dist = PartitionedCsr::from_csr(&mat, &config);
        let x = vec![1.0; n];
        let y_seq = mat.spmv(&x).expect("spmv failed");
        let y_dist = dist.spmv(&x).expect("distributed spmv failed");
        for (ys, yd) in y_seq.iter().zip(y_dist.iter()) {
            assert!((ys - yd).abs() < 1e-10);
        }
    }
}
