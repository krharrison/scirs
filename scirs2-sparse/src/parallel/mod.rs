//! Parallel sparse matrix operations
//!
//! This module provides parallelized implementations of common sparse matrix operations:
//!
//! - **SpMV**: Parallel sparse matrix-vector multiplication (via row partitioning)
//! - **SpMM**: Parallel sparse matrix-matrix multiplication (row-parallel gustavson)
//! - **Sparse addition**: Parallel CSR matrix addition
//! - **Parallel ILU**: Incomplete LU factorization with level scheduling
//! - **Colored Gauss-Seidel**: Graph-colored parallel Gauss-Seidel iteration
//! - **RowPartitioner**: Load-balanced row partitioning for parallel operations

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use scirs2_core::numeric::SparseElement;
use scirs2_core::parallel_ops::*;
use std::fmt::Debug;

// ============================================================
// RowPartitioner
// ============================================================

/// Row-based partitioning strategy for load-balanced parallel sparse operations.
///
/// Distributes rows of a sparse matrix among worker threads so that each thread
/// processes approximately the same number of non-zeros (NNZ-balanced), rather
/// than the same number of rows.
#[derive(Debug, Clone)]
pub struct RowPartitioner {
    /// Partition boundaries: `partitions[i]..partitions[i+1]` is the row range
    /// for partition `i`.
    pub partitions: Vec<usize>,
    /// Number of partitions (threads).
    pub num_partitions: usize,
}

impl RowPartitioner {
    /// Create a new NNZ-balanced row partitioner.
    ///
    /// Splits `nrows` rows into `num_partitions` groups such that the total
    /// number of non-zeros per group is approximately equal.
    ///
    /// # Arguments
    ///
    /// * `indptr` - CSR row-pointer array of length `nrows + 1`.
    /// * `nrows`  - Number of rows in the matrix.
    /// * `num_partitions` - Number of partitions to create (≥ 1).
    ///
    /// # Errors
    ///
    /// Returns `SparseError::ValueError` if `num_partitions` is 0.
    pub fn new(indptr: &[usize], nrows: usize, num_partitions: usize) -> SparseResult<Self> {
        if num_partitions == 0 {
            return Err(SparseError::ValueError(
                "num_partitions must be at least 1".to_string(),
            ));
        }
        if indptr.len() < nrows + 1 {
            return Err(SparseError::ValueError(
                "indptr length must be at least nrows + 1".to_string(),
            ));
        }

        let total_nnz = indptr[nrows];
        // Target NNZ per partition (ceiling division).
        let target = (total_nnz + num_partitions - 1) / num_partitions;

        let mut partitions = vec![0usize; num_partitions + 1];
        let mut part = 0usize;
        let mut accumulated = 0usize;

        for row in 0..nrows {
            let row_nnz = indptr[row + 1] - indptr[row];
            accumulated += row_nnz;
            if accumulated >= target && part + 1 < num_partitions {
                part += 1;
                partitions[part] = row + 1;
                accumulated = 0;
            }
        }
        // Last partition always ends at nrows.
        partitions[num_partitions] = nrows;

        // Fill any empty tail partitions with nrows.
        for i in (part + 1)..num_partitions {
            partitions[i] = nrows;
        }

        Ok(Self {
            partitions,
            num_partitions,
        })
    }

    /// Return the row range for partition `i`.
    pub fn range(&self, i: usize) -> std::ops::Range<usize> {
        self.partitions[i]..self.partitions[i + 1]
    }

    /// Auto-choose number of partitions based on matrix size.
    pub fn auto(indptr: &[usize], nrows: usize) -> SparseResult<Self> {
        let nthreads = get_num_threads().max(1);
        Self::new(indptr, nrows, nthreads)
    }
}

// ============================================================
// parallel_spmv
// ============================================================

/// Parallel sparse matrix-vector multiplication: `y = A * x`.
///
/// Uses row partitioning and `scirs2-core`'s parallel infrastructure.
/// Each partition computes its rows independently; results are written
/// into disjoint regions of the output vector.
///
/// # Arguments
///
/// * `a` - CSR matrix of shape `(m, n)`.
/// * `x` - Dense vector of length `n`.
///
/// # Returns
///
/// Dense result vector of length `m`.
///
/// # Errors
///
/// Returns `SparseError::DimensionMismatch` if `x.len() != a.cols()`.
pub fn parallel_spmv(a: &CsrMatrix<f64>, x: &[f64]) -> SparseResult<Vec<f64>> {
    let (m, n) = a.shape();
    if x.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: x.len(),
        });
    }

    let partitioner = RowPartitioner::auto(&a.indptr, m)?;
    let num_parts = partitioner.num_partitions;

    // Build chunk descriptors (start_row, end_row).
    let ranges: Vec<(usize, usize)> = (0..num_parts)
        .map(|i| {
            let r = partitioner.range(i);
            (r.start, r.end)
        })
        .collect();

    // Parallel computation: each partition returns a Vec<(usize, f64)> of (row, value) pairs.
    let chunks: Vec<Vec<(usize, f64)>> = parallel_map(&ranges, |(start, end)| {
        let mut partial = Vec::with_capacity(end - start);
        for row in *start..*end {
            let mut sum = 0.0f64;
            for j in a.indptr[row]..a.indptr[row + 1] {
                sum += a.data[j] * x[a.indices[j]];
            }
            partial.push((row, sum));
        }
        partial
    });

    // Assemble output (each row appears in exactly one chunk).
    let mut y = vec![0.0f64; m];
    for chunk in chunks {
        for (row, val) in chunk {
            y[row] = val;
        }
    }
    Ok(y)
}

// ============================================================
// parallel_spmm  (sparse × sparse → sparse, CSR output)
// ============================================================

/// Parallel sparse matrix-matrix multiplication: `C = A * B` (CSR × CSR → CSR).
///
/// Implements the row-parallel Gustavson algorithm: each row of `C` is computed
/// independently from the corresponding row of `A` and the rows of `B`.
///
/// # Arguments
///
/// * `a` - Left operand, CSR matrix of shape `(m, k)`.
/// * `b` - Right operand, CSR matrix of shape `(k, n)`.
///
/// # Errors
///
/// Returns `SparseError::DimensionMismatch` if `a.cols() != b.rows()`.
pub fn parallel_spmm(a: &CsrMatrix<f64>, b: &CsrMatrix<f64>) -> SparseResult<CsrMatrix<f64>> {
    let (m, k) = a.shape();
    let (brows, n) = b.shape();
    if k != brows {
        return Err(SparseError::DimensionMismatch {
            expected: k,
            found: brows,
        });
    }

    // Partition rows of A for parallel computation.
    let partitioner = RowPartitioner::auto(&a.indptr, m)?;
    let num_parts = partitioner.num_partitions;

    let ranges: Vec<(usize, usize)> = (0..num_parts)
        .map(|i| {
            let r = partitioner.range(i);
            (r.start, r.end)
        })
        .collect();

    // Each partition computes its rows of C.
    // Returns Vec<(row_idx, Vec<(col, val)>)>.
    let chunks: Vec<Vec<(usize, Vec<(usize, f64)>)>> = parallel_map(&ranges, |(start, end)| {
        let mut rows_out = Vec::with_capacity(end - start);
        // Per-row dense accumulator (length n) reused across rows in this chunk.
        let mut acc = vec![0.0f64; n];
        let mut marker = vec![false; n];
        let mut nz_cols: Vec<usize> = Vec::new();

        for row in *start..*end {
            // Reset accumulator for this row.
            for &col in &nz_cols {
                acc[col] = 0.0;
                marker[col] = false;
            }
            nz_cols.clear();

            // For each non-zero in row of A, scatter-accumulate into acc.
            for ja in a.indptr[row]..a.indptr[row + 1] {
                let a_col = a.indices[ja];
                let a_val = a.data[ja];
                for jb in b.indptr[a_col]..b.indptr[a_col + 1] {
                    let b_col = b.indices[jb];
                    let b_val = b.data[jb];
                    acc[b_col] += a_val * b_val;
                    if !marker[b_col] {
                        marker[b_col] = true;
                        nz_cols.push(b_col);
                    }
                }
            }

            // Gather non-zeros for this row.
            let mut row_nz: Vec<(usize, f64)> = nz_cols
                .iter()
                .filter_map(|&col| {
                    let v = acc[col];
                    if v != 0.0 {
                        Some((col, v))
                    } else {
                        None
                    }
                })
                .collect();
            row_nz.sort_unstable_by_key(|&(col, _)| col);
            rows_out.push((row, row_nz));
        }
        rows_out
    });

    // Assemble CSR result.
    let mut indptr = vec![0usize; m + 1];
    // First pass: count NNZ per row.
    let mut all_rows: Vec<(usize, Vec<(usize, f64)>)> = chunks.into_iter().flatten().collect();
    all_rows.sort_unstable_by_key(|&(row, _)| row);
    for (row, ref nz) in &all_rows {
        indptr[row + 1] = nz.len();
    }
    for i in 1..=m {
        indptr[i] += indptr[i - 1];
    }
    let total_nnz = indptr[m];
    let mut indices = vec![0usize; total_nnz];
    let mut data = vec![0.0f64; total_nnz];
    for (row, nz) in all_rows {
        let start = indptr[row];
        for (k2, (col, val)) in nz.into_iter().enumerate() {
            indices[start + k2] = col;
            data[start + k2] = val;
        }
    }

    CsrMatrix::from_raw_csr(data, indptr, indices, (m, n))
}

// ============================================================
// parallel_csr_add
// ============================================================

/// Parallel sparse matrix addition: `C = A + B` (CSR + CSR → CSR).
///
/// Processes each row independently in parallel using a scatter-accumulate approach.
///
/// # Arguments
///
/// * `a` - CSR matrix of shape `(m, n)`.
/// * `b` - CSR matrix of the same shape `(m, n)`.
///
/// # Errors
///
/// Returns `SparseError::ShapeMismatch` if shapes differ.
pub fn parallel_csr_add(a: &CsrMatrix<f64>, b: &CsrMatrix<f64>) -> SparseResult<CsrMatrix<f64>> {
    let (am, an) = a.shape();
    let (bm, bn) = b.shape();
    if am != bm || an != bn {
        return Err(SparseError::ShapeMismatch {
            expected: (am, an),
            found: (bm, bn),
        });
    }
    let m = am;
    let n = an;

    let partitioner = RowPartitioner::auto(&a.indptr, m)?;
    let num_parts = partitioner.num_partitions;
    let ranges: Vec<(usize, usize)> = (0..num_parts)
        .map(|i| {
            let r = partitioner.range(i);
            (r.start, r.end)
        })
        .collect();

    let chunks: Vec<Vec<(usize, Vec<(usize, f64)>)>> = parallel_map(&ranges, |(start, end)| {
        let mut rows_out = Vec::with_capacity(end - start);
        let mut acc = vec![0.0f64; n];
        let mut marker = vec![false; n];
        let mut nz_cols: Vec<usize> = Vec::new();

        for row in *start..*end {
            for &col in &nz_cols {
                acc[col] = 0.0;
                marker[col] = false;
            }
            nz_cols.clear();

            // Accumulate A row.
            for ja in a.indptr[row]..a.indptr[row + 1] {
                let col = a.indices[ja];
                acc[col] += a.data[ja];
                if !marker[col] {
                    marker[col] = true;
                    nz_cols.push(col);
                }
            }
            // Accumulate B row.
            for jb in b.indptr[row]..b.indptr[row + 1] {
                let col = b.indices[jb];
                acc[col] += b.data[jb];
                if !marker[col] {
                    marker[col] = true;
                    nz_cols.push(col);
                }
            }
            let mut row_nz: Vec<(usize, f64)> = nz_cols
                .iter()
                .filter_map(|&col| {
                    let v = acc[col];
                    if v != 0.0 {
                        Some((col, v))
                    } else {
                        None
                    }
                })
                .collect();
            row_nz.sort_unstable_by_key(|&(col, _)| col);
            rows_out.push((row, row_nz));
        }
        rows_out
    });

    // Assemble.
    let mut all_rows: Vec<(usize, Vec<(usize, f64)>)> = chunks.into_iter().flatten().collect();
    all_rows.sort_unstable_by_key(|&(row, _)| row);

    let mut indptr = vec![0usize; m + 1];
    for (row, ref nz) in &all_rows {
        indptr[row + 1] = nz.len();
    }
    for i in 1..=m {
        indptr[i] += indptr[i - 1];
    }
    let total_nnz = indptr[m];
    let mut indices_out = vec![0usize; total_nnz];
    let mut data_out = vec![0.0f64; total_nnz];
    for (row, nz) in all_rows {
        let start = indptr[row];
        for (k, (col, val)) in nz.into_iter().enumerate() {
            indices_out[start + k] = col;
            data_out[start + k] = val;
        }
    }

    CsrMatrix::from_raw_csr(data_out, indptr, indices_out, (m, n))
}

// ============================================================
// ILUFactor + parallel_ilu_factor
// ============================================================

/// Result of an ILU(0) factorization with level scheduling information.
#[derive(Debug, Clone)]
pub struct ILUFactor {
    /// Lower triangular factor L (stored as CSR, diagonal = 1).
    pub l: CsrMatrix<f64>,
    /// Upper triangular factor U (stored as CSR).
    pub u: CsrMatrix<f64>,
    /// Row permutation used during factorization (identity if no pivoting).
    pub perm: Vec<usize>,
    /// Level sets for parallel triangular solve scheduling.
    /// `level_sets[k]` contains rows that can be processed concurrently at level `k`.
    pub level_sets: Vec<Vec<usize>>,
}

impl ILUFactor {
    /// Apply L^{-1} * x (forward solve) in parallel using level scheduling.
    pub fn forward_solve(&self, b: &[f64]) -> SparseResult<Vec<f64>> {
        let n = b.len();
        if self.l.rows() != n {
            return Err(SparseError::DimensionMismatch {
                expected: self.l.rows(),
                found: n,
            });
        }
        // Apply permutation.
        let mut x: Vec<f64> = (0..n).map(|i| b[self.perm[i]]).collect();

        // Forward substitution level by level.
        for level in &self.level_sets {
            // Within a level, rows are independent.
            let updates: Vec<(usize, f64)> = parallel_map(level, |&row| {
                let mut s = x[row];
                for j in self.l.indptr[row]..self.l.indptr[row + 1] {
                    let col = self.l.indices[j];
                    if col < row {
                        s -= self.l.data[j] * x[col];
                    }
                }
                (row, s)
            });
            for (row, val) in updates {
                x[row] = val;
            }
        }
        Ok(x)
    }

    /// Apply U^{-1} * x (backward solve) in parallel using reversed level scheduling.
    pub fn backward_solve(&self, b: &[f64]) -> SparseResult<Vec<f64>> {
        let n = b.len();
        if self.u.rows() != n {
            return Err(SparseError::DimensionMismatch {
                expected: self.u.rows(),
                found: n,
            });
        }
        let mut x = b.to_vec();
        // Backward substitution in reversed level order.
        for level in self.level_sets.iter().rev() {
            let updates: Vec<(usize, f64)> = parallel_map(level, |&row| {
                let mut s = x[row];
                let mut diag = 1.0f64;
                for j in self.u.indptr[row]..self.u.indptr[row + 1] {
                    let col = self.u.indices[j];
                    if col == row {
                        diag = self.u.data[j];
                    } else if col > row {
                        s -= self.u.data[j] * x[col];
                    }
                }
                (row, s / diag)
            });
            for (row, val) in updates {
                x[row] = val;
            }
        }
        Ok(x)
    }
}

/// Build level sets from a dependency DAG (used for level-scheduled ILU solve).
fn build_level_sets(n: usize, deps: &[Vec<usize>]) -> Vec<Vec<usize>> {
    // deps[i] = list of rows that row i depends on (i.e., row i must wait for these).
    let mut level_of = vec![0usize; n];
    for i in 0..n {
        for &d in &deps[i] {
            if level_of[d] + 1 > level_of[i] {
                level_of[i] = level_of[d] + 1;
            }
        }
    }
    let max_level = level_of.iter().copied().max().unwrap_or(0);
    let mut sets: Vec<Vec<usize>> = vec![Vec::new(); max_level + 1];
    for i in 0..n {
        sets[level_of[i]].push(i);
    }
    sets
}

/// Parallel incomplete LU factorization (ILU(0)) with level scheduling.
///
/// Performs ILU(0) factorization, which keeps the same sparsity pattern as
/// the original matrix. Returns an `ILUFactor` containing L, U, and level
/// sets for parallel triangular solve.
///
/// # Arguments
///
/// * `a` - Square CSR matrix of shape `(n, n)`.
///
/// # Errors
///
/// Returns an error if the matrix is not square or has a zero diagonal entry.
pub fn parallel_ilu_factor(a: &CsrMatrix<f64>) -> SparseResult<ILUFactor> {
    let (m, n) = a.shape();
    if m != n {
        return Err(SparseError::ValueError(
            "ILU factorization requires a square matrix".to_string(),
        ));
    }

    // Copy matrix values into a mutable work array indexed by (row, col).
    // We use a dense-per-row approach for simplicity with the CSR sparsity pattern.
    let nnz = a.nnz();
    let mut work_data = a.data.clone();
    let indptr = a.indptr.clone();
    let indices = a.indices.clone();

    // Build a fast lookup: given (row, col), return position in work_data.
    // Since columns within a row are arbitrary, build a per-row HashMap.
    let mut row_col_to_pos: Vec<std::collections::HashMap<usize, usize>> =
        vec![std::collections::HashMap::new(); n];
    for row in 0..n {
        for j in indptr[row]..indptr[row + 1] {
            row_col_to_pos[row].insert(indices[j], j);
        }
    }

    // ILU(0) factorization (sequential, modifies work_data in place).
    let perm: Vec<usize> = (0..n).collect();

    for i in 1..n {
        // For each non-zero (i, k) with k < i, compute multiplier.
        for ji in indptr[i]..indptr[i + 1] {
            let k = indices[ji];
            if k >= i {
                break; // Assuming sorted columns within each row.
            }
            // Find u_kk.
            let u_kk_pos = match row_col_to_pos[k].get(&k) {
                Some(&pos) => pos,
                None => continue, // No diagonal in row k — skip.
            };
            let u_kk = work_data[u_kk_pos];
            if u_kk.abs() < 1e-300 {
                return Err(SparseError::SingularMatrix(format!(
                    "Near-zero pivot encountered at row {}",
                    k
                )));
            }
            let multiplier = work_data[ji] / u_kk;
            work_data[ji] = multiplier;

            // Update row i for each (k, j) non-zero in U part of row k.
            for jk in indptr[k]..indptr[k + 1] {
                let j = indices[jk];
                if j <= k {
                    continue; // Only U part (j > k).
                }
                if let Some(&pos_ij) = row_col_to_pos[i].get(&j) {
                    work_data[pos_ij] -= multiplier * work_data[jk];
                }
                // If (i, j) not in sparsity pattern, drop (ILU(0) rule).
            }
        }
    }

    // Split work_data into L and U.
    let mut l_indptr = vec![0usize; n + 1];
    let mut u_indptr = vec![0usize; n + 1];
    let mut l_indices: Vec<usize> = Vec::with_capacity(nnz);
    let mut l_data: Vec<f64> = Vec::with_capacity(nnz);
    let mut u_indices: Vec<usize> = Vec::with_capacity(nnz);
    let mut u_data: Vec<f64> = Vec::with_capacity(nnz);

    // Dependency list for level scheduling: row i depends on row k if l[i,k] != 0.
    let mut deps: Vec<Vec<usize>> = vec![Vec::new(); n];

    for row in 0..n {
        for j in indptr[row]..indptr[row + 1] {
            let col = indices[j];
            let val = work_data[j];
            if col < row {
                // L part (strictly lower).
                l_indices.push(col);
                l_data.push(val);
                l_indptr[row + 1] += 1;
                deps[row].push(col);
            } else if col == row {
                // Diagonal: goes to U, L diagonal is 1.
                l_indices.push(col);
                l_data.push(1.0);
                l_indptr[row + 1] += 1;
                u_indices.push(col);
                u_data.push(val);
                u_indptr[row + 1] += 1;
            } else {
                // U part (strictly upper).
                u_indices.push(col);
                u_data.push(val);
                u_indptr[row + 1] += 1;
            }
        }
    }
    for i in 1..=n {
        l_indptr[i] += l_indptr[i - 1];
        u_indptr[i] += u_indptr[i - 1];
    }

    let l = CsrMatrix::from_raw_csr(l_data, l_indptr, l_indices, (n, n))?;
    let u = CsrMatrix::from_raw_csr(u_data, u_indptr, u_indices, (n, n))?;
    let level_sets = build_level_sets(n, &deps);

    Ok(ILUFactor {
        l,
        u,
        perm,
        level_sets,
    })
}

// ============================================================
// ColoredGaussSeidel
// ============================================================

/// Graph-coloring-based parallel Gauss-Seidel preconditioner / smoother.
///
/// In standard Gauss-Seidel (GS), rows must be processed sequentially because
/// updating `x[i]` may read `x[j]` for `j != i` (data dependency). Graph
/// coloring assigns each row a "color" such that rows of the same color share
/// no off-diagonal non-zeros—they are therefore independent and can be updated
/// concurrently.
///
/// This struct stores the coloring and provides a parallel GS sweep method.
#[derive(Debug, Clone)]
pub struct ColoredGaussSeidel {
    /// `color[i]` = color index for row `i`. Colors are `0..num_colors`.
    pub color: Vec<usize>,
    /// `color_sets[c]` = sorted list of rows with color `c`.
    pub color_sets: Vec<Vec<usize>>,
    /// Total number of colors used.
    pub num_colors: usize,
}

impl ColoredGaussSeidel {
    /// Build a coloring from a CSR matrix using a sequential greedy distance-1 graph coloring.
    ///
    /// Row `i` and row `j` receive the same color only if they share no off-diagonal
    /// non-zero column index — meaning they are independent in a Gauss-Seidel sweep.
    ///
    /// # Arguments
    ///
    /// * `a` - CSR matrix whose adjacency graph determines the coloring.
    ///
    /// # Returns
    ///
    /// A `ColoredGaussSeidel` instance containing the coloring.
    pub fn from_matrix<T>(a: &CsrMatrix<T>) -> SparseResult<Self>
    where
        T: Clone + Copy + SparseElement + scirs2_core::numeric::Zero + std::cmp::PartialEq + Debug,
    {
        let (n, _) = a.shape();
        let mut color = vec![usize::MAX; n];

        // Greedy coloring in natural order.
        // For each row, collect the colors of already-colored neighbors, then
        // assign the smallest unused color.
        let mut forbidden: Vec<bool> = Vec::new();

        for row in 0..n {
            // Mark colors used by already-colored neighbors.
            let neighbor_colors: Vec<usize> = (a.indptr[row]..a.indptr[row + 1])
                .filter_map(|j| {
                    let nbr = a.indices[j];
                    if nbr != row && color[nbr] != usize::MAX {
                        Some(color[nbr])
                    } else {
                        None
                    }
                })
                .collect();

            // Ensure forbidden is large enough.
            let max_needed = neighbor_colors.iter().copied().max().map(|c| c + 1).unwrap_or(0);
            if forbidden.len() < max_needed {
                forbidden.resize(max_needed, false);
            }
            for &c in &neighbor_colors {
                forbidden[c] = true;
            }

            // Find the first unused color.
            let chosen = (0..).find(|&c| c >= forbidden.len() || !forbidden[c]).unwrap_or(0);
            color[row] = chosen;

            // Unmark forbidden entries.
            for &c in &neighbor_colors {
                forbidden[c] = false;
            }
        }

        // Compute max color.
        let num_colors = color.iter().copied().filter(|&c| c != usize::MAX).max().map(|c| c + 1).unwrap_or(0);
        let mut color_sets: Vec<Vec<usize>> = vec![Vec::new(); num_colors];
        for (row, &c) in color.iter().enumerate() {
            if c < num_colors {
                color_sets[c].push(row);
            }
        }

        Ok(Self {
            color,
            color_sets,
            num_colors,
        })
    }

    /// Perform one parallel Gauss-Seidel sweep solving `A * x ≈ b`.
    ///
    /// Each color set is processed in sequence; within a color set, rows are
    /// updated concurrently since they are mutually independent.
    ///
    /// # Arguments
    ///
    /// * `a`       - System matrix (CSR).
    /// * `b`       - Right-hand side vector.
    /// * `x`       - Current iterate; updated in place.
    /// * `omega`   - Relaxation parameter (1.0 = standard GS, 0 < ω < 2 for SOR).
    ///
    /// # Errors
    ///
    /// Returns an error on shape mismatch or zero diagonal.
    pub fn sweep(
        &self,
        a: &CsrMatrix<f64>,
        b: &[f64],
        x: &mut Vec<f64>,
        omega: f64,
    ) -> SparseResult<()> {
        let (n, nc) = a.shape();
        if n != nc {
            return Err(SparseError::ValueError(
                "Matrix must be square for Gauss-Seidel".to_string(),
            ));
        }
        if b.len() != n || x.len() != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: b.len().min(x.len()),
            });
        }

        for color_set in &self.color_sets {
            // Compute new values for all rows in this color set (read x, which is not
            // yet modified for this color).
            let updates: Vec<(usize, f64)> = parallel_map(color_set, |&row| {
                let mut sigma = b[row];
                let mut a_ii = 0.0f64;
                for j in a.indptr[row]..a.indptr[row + 1] {
                    let col = a.indices[j];
                    let val = a.data[j];
                    if col == row {
                        a_ii = val;
                    } else {
                        sigma -= val * x[col];
                    }
                }
                let x_new = if a_ii.abs() > 1e-300 {
                    sigma / a_ii
                } else {
                    x[row] // No update if zero diagonal.
                };
                (row, x[row] + omega * (x_new - x[row]))
            });

            // Apply updates (all independent within this color).
            for (row, val) in updates {
                x[row] = val;
            }
        }

        Ok(())
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn small_csr() -> CsrMatrix<f64> {
        // 4×4 matrix:
        //  2 -1  0  0
        // -1  3 -1  0
        //  0 -1  3 -1
        //  0  0 -1  2
        let rows = vec![0, 0, 1, 1, 1, 2, 2, 2, 3, 3];
        let cols = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let vals = vec![2.0, -1.0, -1.0, 3.0, -1.0, -1.0, 3.0, -1.0, -1.0, 2.0];
        CsrMatrix::new(vals, rows, cols, (4, 4)).expect("build small_csr")
    }

    #[test]
    fn test_row_partitioner() {
        let a = small_csr();
        let p = RowPartitioner::new(&a.indptr, 4, 2).expect("partition");
        assert_eq!(p.num_partitions, 2);
        let r0 = p.range(0);
        let r1 = p.range(1);
        assert!(r0.start < r0.end);
        assert_eq!(r1.end, 4);
        assert_eq!(r0.end, r1.start);
    }

    #[test]
    fn test_parallel_spmv() {
        let a = small_csr();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = parallel_spmv(&a, &x).expect("spmv");
        // Reference: a * x
        // row 0: 2*1 + (-1)*2 = 0
        // row 1: (-1)*1 + 3*2 + (-1)*3 = -1 + 6 - 3 = 2
        // row 2: (-1)*2 + 3*3 + (-1)*4 = -2 + 9 - 4 = 3
        // row 3: (-1)*3 + 2*4 = -3 + 8 = 5
        assert_relative_eq!(y[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(y[1], 2.0, epsilon = 1e-12);
        assert_relative_eq!(y[2], 3.0, epsilon = 1e-12);
        assert_relative_eq!(y[3], 5.0, epsilon = 1e-12);
    }

    #[test]
    fn test_parallel_csr_add() {
        let a = small_csr();
        let b = small_csr();
        let c = parallel_csr_add(&a, &b).expect("add");
        // C = A + B = 2A
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let ya = parallel_spmv(&a, &x).expect("ya");
        let yc = parallel_spmv(&c, &x).expect("yc");
        for i in 0..4 {
            assert_relative_eq!(yc[i], 2.0 * ya[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_parallel_spmm() {
        let a = small_csr();
        // C = A * A  (square)
        let c = parallel_spmm(&a, &a).expect("spmm");
        // Verify via SpMV: (A*A)*x = A*(A*x)
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let ax = parallel_spmv(&a, &x).expect("ax");
        let aax_ref = parallel_spmv(&a, &ax).expect("aax_ref");
        let yc = parallel_spmv(&c, &x).expect("yc");
        for i in 0..4 {
            assert_relative_eq!(yc[i], aax_ref[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_parallel_ilu_factor() {
        // Simple diagonally dominant 4×4 matrix.
        let rows = vec![0, 0, 1, 1, 2, 2, 3, 3];
        let cols = vec![0, 1, 0, 1, 1, 2, 2, 3];
        let vals = vec![4.0, -1.0, -1.0, 4.0, -1.0, 4.0, -1.0, 4.0];
        let a = CsrMatrix::new(vals, rows, cols, (4, 4)).expect("build ilu test matrix");
        let ilu = parallel_ilu_factor(&a).expect("ilu factor");
        assert!(ilu.level_sets.len() > 0);
        // Solve L * U * x = b for b = ones.
        let b = vec![1.0; 4];
        let ly = ilu.forward_solve(&b).expect("forward solve");
        let x = ilu.backward_solve(&ly).expect("backward solve");
        // Verify: A * x ≈ b
        let ax = parallel_spmv(&a, &x).expect("verify");
        for i in 0..4 {
            assert_relative_eq!(ax[i], b[i], epsilon = 1e-10);
        }
    }
}
