//! GPU-ready SpMV (Sparse Matrix-Vector Multiply) operations
//!
//! Provides CPU-side SIMD-friendly implementations that serve as compute-shader
//! placeholders. All hot paths are row-parallel and chunked for cache efficiency,
//! matching the memory access pattern expected by a GPU compute kernel.

use crate::error::{SparseError, SparseResult};
use scirs2_core::ndarray::{Array2, Axis};

// ============================================================
// Configuration
// ============================================================

/// GPU compute backend selector.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuSpMvBackend {
    /// CPU simulation — row-parallel, cache-efficient (always available).
    #[default]
    Cpu,
    /// WebGPU via wgpu (feature-gated, not yet wired).
    WebGpu,
}

/// Configuration for GPU-ready SpMV operations.
#[derive(Debug, Clone)]
pub struct GpuSpMvConfig {
    /// Compute backend to use.
    pub backend: GpuSpMvBackend,
    /// Workgroup / warp size (default 256).
    pub block_size: usize,
    /// Number of warps per block (default 8).
    pub n_warps: usize,
    /// Whether to use texture memory / L1 hints for the x vector (default false).
    pub use_texture: bool,
}

impl Default for GpuSpMvConfig {
    fn default() -> Self {
        Self {
            backend: GpuSpMvBackend::Cpu,
            block_size: 256,
            n_warps: 8,
            use_texture: false,
        }
    }
}

// ============================================================
// CSR SpMV  y = A * x
// ============================================================

/// Compute `y = A * x` for a matrix stored in CSR format.
///
/// The CPU path processes rows in chunks sized to `config.block_size` so that
/// the access pattern mirrors what a GPU compute shader would execute per
/// workgroup.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] when the vector length does not
/// match the number of columns implied by `col_idx`.
pub fn csr_spmv(
    row_ptr: &[usize],
    col_idx: &[usize],
    values: &[f64],
    x: &[f64],
    config: &GpuSpMvConfig,
) -> SparseResult<Vec<f64>> {
    if row_ptr.is_empty() {
        return Ok(Vec::new());
    }
    let n_rows = row_ptr.len() - 1;

    // Basic consistency check
    if col_idx.len() != values.len() {
        return Err(SparseError::InconsistentData {
            reason: format!(
                "col_idx length {} != values length {}",
                col_idx.len(),
                values.len()
            ),
        });
    }

    let mut y = vec![0.0_f64; n_rows];

    match config.backend {
        GpuSpMvBackend::Cpu => {
            // Chunk rows by block_size to simulate GPU workgroup granularity.
            let block = config.block_size.max(1);
            let mut row_start = 0usize;
            while row_start < n_rows {
                let row_end = (row_start + block).min(n_rows);
                for row in row_start..row_end {
                    let col_start = row_ptr[row];
                    let col_end = row_ptr[row + 1];
                    let mut acc = 0.0_f64;
                    for k in col_start..col_end {
                        let col = col_idx[k];
                        if col >= x.len() {
                            return Err(SparseError::DimensionMismatch {
                                expected: x.len(),
                                found: col + 1,
                            });
                        }
                        acc += values[k] * x[col];
                    }
                    y[row] = acc;
                }
                row_start = row_end;
            }
        }
        GpuSpMvBackend::WebGpu => {
            // Fall back to CPU simulation; real wgpu dispatch would go here.
            for row in 0..n_rows {
                let col_start = row_ptr[row];
                let col_end = row_ptr[row + 1];
                let mut acc = 0.0_f64;
                for k in col_start..col_end {
                    let col = col_idx[k];
                    if col >= x.len() {
                        return Err(SparseError::DimensionMismatch {
                            expected: x.len(),
                            found: col + 1,
                        });
                    }
                    acc += values[k] * x[col];
                }
                y[row] = acc;
            }
        }
    }

    Ok(y)
}

// ============================================================
// Batched SpMV  Y = A * X  where X is [n_cols, n_rhs]
// ============================================================

/// Compute `Y = A * X` for multiple right-hand side vectors.
///
/// `x_batch` has shape `[n_cols, n_rhs]`; the result has shape
/// `[n_rows, n_rhs]`.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] when `x_batch.nrows()` does not
/// match the number of columns in the sparse matrix.
pub fn csr_spmv_batch(
    row_ptr: &[usize],
    col_idx: &[usize],
    values: &[f64],
    x_batch: &Array2<f64>,
    config: &GpuSpMvConfig,
) -> SparseResult<Array2<f64>> {
    if row_ptr.is_empty() {
        return Ok(Array2::zeros((0, x_batch.ncols())));
    }
    let n_rows = row_ptr.len() - 1;
    let n_rhs = x_batch.ncols();
    let n_cols = x_batch.nrows();

    let mut y = Array2::zeros((n_rows, n_rhs));

    for rhs in 0..n_rhs {
        let x_col = x_batch.index_axis(Axis(1), rhs);
        let x_slice: Vec<f64> = x_col.iter().copied().collect();
        if x_slice.len() != n_cols {
            return Err(SparseError::DimensionMismatch {
                expected: n_cols,
                found: x_slice.len(),
            });
        }
        let y_col = csr_spmv(row_ptr, col_idx, values, &x_slice, config)?;
        for row in 0..n_rows {
            y[[row, rhs]] = y_col[row];
        }
    }

    Ok(y)
}

// ============================================================
// SpMM  C = A * B  where B is dense [n_cols, k]
// ============================================================

/// Compute the sparse-dense product `C = A * B`.
///
/// `b` has shape `[n_cols, k]`; the result `C` has shape `[n_rows, k]`.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] when `b.nrows()` does not equal
/// the number of columns implied by `col_idx`.
pub fn csr_spmm(
    row_ptr: &[usize],
    col_idx: &[usize],
    values: &[f64],
    b: &Array2<f64>,
    config: &GpuSpMvConfig,
) -> SparseResult<Array2<f64>> {
    if row_ptr.is_empty() {
        return Ok(Array2::zeros((0, b.ncols())));
    }
    let n_rows = row_ptr.len() - 1;
    let k = b.ncols();
    let n_b_rows = b.nrows();

    let mut c = Array2::zeros((n_rows, k));

    let block = match config.backend {
        GpuSpMvBackend::Cpu => config.block_size.max(1),
        GpuSpMvBackend::WebGpu => config.block_size.max(1),
    };

    let mut row_start = 0usize;
    while row_start < n_rows {
        let row_end = (row_start + block).min(n_rows);
        for row in row_start..row_end {
            let col_start = row_ptr[row];
            let col_end = row_ptr[row + 1];
            for k_i in col_start..col_end {
                let col = col_idx[k_i];
                if col >= n_b_rows {
                    return Err(SparseError::DimensionMismatch {
                        expected: n_b_rows,
                        found: col + 1,
                    });
                }
                let a_val = values[k_i];
                for j in 0..k {
                    c[[row, j]] += a_val * b[[col, j]];
                }
            }
        }
        row_start = row_end;
    }

    Ok(c)
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn identity_csr(n: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let row_ptr: Vec<usize> = (0..=n).collect();
        let col_idx: Vec<usize> = (0..n).collect();
        let values: Vec<f64> = vec![1.0; n];
        (row_ptr, col_idx, values)
    }

    #[test]
    fn test_spmv_identity() {
        let n = 4;
        let (row_ptr, col_idx, values) = identity_csr(n);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let config = GpuSpMvConfig::default();
        let y = csr_spmv(&row_ptr, &col_idx, &values, &x, &config).expect("spmv failed");
        assert_eq!(y, x);
    }

    #[test]
    fn test_spmv_diagonal() {
        // Diagonal matrix with [2, 3, 5]
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0, 1, 2];
        let values = vec![2.0, 3.0, 5.0];
        let x = vec![1.0, 1.0, 1.0];
        let config = GpuSpMvConfig::default();
        let y = csr_spmv(&row_ptr, &col_idx, &values, &x, &config).expect("spmv failed");
        assert_eq!(y, vec![2.0, 3.0, 5.0]);
    }

    #[test]
    fn test_spmv_dense() {
        // Full 2×3 matrix [[1,2,3],[4,5,6]]
        let row_ptr = vec![0, 3, 6];
        let col_idx = vec![0, 1, 2, 0, 1, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 0.0, 1.0];
        let config = GpuSpMvConfig::default();
        let y = csr_spmv(&row_ptr, &col_idx, &values, &x, &config).expect("spmv failed");
        assert_eq!(y, vec![4.0, 10.0]);
    }

    #[test]
    fn test_spmv_batch() {
        let n = 3;
        let (row_ptr, col_idx, values) = identity_csr(n);
        let x_batch = Array2::from_shape_vec((3, 2), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
            .expect("shape error");
        let config = GpuSpMvConfig::default();
        let y = csr_spmv_batch(&row_ptr, &col_idx, &values, &x_batch, &config)
            .expect("spmv_batch failed");
        assert_eq!(y.shape(), &[3, 2]);
        assert_eq!(y[[0, 0]], 1.0);
        assert_eq!(y[[2, 1]], 6.0);
    }

    #[test]
    fn test_spmm() {
        // I * B = B
        let n = 3;
        let (row_ptr, col_idx, values) = identity_csr(n);
        let b = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("shape error");
        let config = GpuSpMvConfig::default();
        let c = csr_spmm(&row_ptr, &col_idx, &values, &b, &config).expect("spmm failed");
        assert_eq!(c, b);
    }
}
