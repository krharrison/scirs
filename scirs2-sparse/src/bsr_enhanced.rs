//! Enhanced Block Sparse Row (BSR) format with flat block storage and Block LU factorization
//!
//! This module provides an enhanced BSR format with:
//! - Flat contiguous block storage for better cache performance
//! - Block matrix-vector multiplication
//! - Direct conversion to/from CsrArray
//! - Block LU factorization for block-structured systems

use crate::csr_array::CsrArray;
use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, SparseElement};
use std::fmt::Debug;
use std::ops::Div;

/// Enhanced Block Sparse Row format with flat contiguous storage
///
/// Unlike the standard `BsrArray` which uses `Vec<Vec<Vec<T>>>`, this format
/// stores all block data in a single flat `Vec<T>` for better cache locality
/// and SIMD-friendliness. Each block of size `(br, bc)` is stored contiguously
/// in row-major order.
///
/// Memory layout for block `k`:
///   `data[k * br * bc .. (k+1) * br * bc]` is the block in row-major order
#[derive(Debug, Clone)]
pub struct EnhancedBsr<T> {
    /// Flat contiguous block data. Block `k` occupies
    /// `data[k * block_area .. (k+1) * block_area]` in row-major order.
    data: Vec<T>,
    /// Block column indices for each stored block.
    /// `block_col_indices[k]` is the block-column index of block `k`.
    block_col_indices: Vec<usize>,
    /// Block row pointers. Block row `i` contains blocks
    /// `block_row_ptr[i] .. block_row_ptr[i+1]`.
    block_row_ptr: Vec<usize>,
    /// Number of scalar rows in the matrix
    nrows: usize,
    /// Number of scalar columns in the matrix
    ncols: usize,
    /// Block row size
    br: usize,
    /// Block column size
    bc: usize,
    /// Number of block rows
    n_block_rows: usize,
    /// Number of block columns
    n_block_cols: usize,
}

impl<T> EnhancedBsr<T>
where
    T: Float + SparseElement + Debug + Copy + 'static,
{
    /// Create a new empty EnhancedBsr
    ///
    /// # Arguments
    /// * `nrows` - Number of scalar rows
    /// * `ncols` - Number of scalar columns
    /// * `br` - Block row size
    /// * `bc` - Block column size
    pub fn new(nrows: usize, ncols: usize, br: usize, bc: usize) -> SparseResult<Self> {
        if br == 0 || bc == 0 {
            return Err(SparseError::ValueError(
                "Block dimensions must be positive".to_string(),
            ));
        }
        if nrows == 0 || ncols == 0 {
            return Err(SparseError::ValueError(
                "Matrix dimensions must be positive".to_string(),
            ));
        }

        let n_block_rows = (nrows + br - 1) / br;
        let n_block_cols = (ncols + bc - 1) / bc;

        Ok(Self {
            data: Vec::new(),
            block_col_indices: Vec::new(),
            block_row_ptr: vec![0; n_block_rows + 1],
            nrows,
            ncols,
            br,
            bc,
            n_block_rows,
            n_block_cols,
        })
    }

    /// Create an EnhancedBsr from raw BSR components
    ///
    /// # Arguments
    /// * `data` - Flat block data (all blocks contiguous, each block in row-major order)
    /// * `block_col_indices` - Block column index for each stored block
    /// * `block_row_ptr` - Block row pointers
    /// * `nrows` - Number of scalar rows
    /// * `ncols` - Number of scalar columns
    /// * `br` - Block row size
    /// * `bc` - Block column size
    pub fn from_raw(
        data: Vec<T>,
        block_col_indices: Vec<usize>,
        block_row_ptr: Vec<usize>,
        nrows: usize,
        ncols: usize,
        br: usize,
        bc: usize,
    ) -> SparseResult<Self> {
        if br == 0 || bc == 0 {
            return Err(SparseError::ValueError(
                "Block dimensions must be positive".to_string(),
            ));
        }
        if nrows == 0 || ncols == 0 {
            return Err(SparseError::ValueError(
                "Matrix dimensions must be positive".to_string(),
            ));
        }

        let n_block_rows = (nrows + br - 1) / br;
        let n_block_cols = (ncols + bc - 1) / bc;
        let block_area = br * bc;
        let n_blocks = block_col_indices.len();

        if block_row_ptr.len() != n_block_rows + 1 {
            return Err(SparseError::DimensionMismatch {
                expected: n_block_rows + 1,
                found: block_row_ptr.len(),
            });
        }
        if data.len() != n_blocks * block_area {
            return Err(SparseError::DimensionMismatch {
                expected: n_blocks * block_area,
                found: data.len(),
            });
        }

        // Validate block_row_ptr is non-decreasing and last value matches n_blocks
        for i in 0..n_block_rows {
            if block_row_ptr[i] > block_row_ptr[i + 1] {
                return Err(SparseError::InconsistentData {
                    reason: "block_row_ptr must be non-decreasing".to_string(),
                });
            }
        }
        if block_row_ptr[n_block_rows] != n_blocks {
            return Err(SparseError::DimensionMismatch {
                expected: n_blocks,
                found: block_row_ptr[n_block_rows],
            });
        }

        // Validate block column indices are in range
        for &bc_idx in &block_col_indices {
            if bc_idx >= n_block_cols {
                return Err(SparseError::ValueError(format!(
                    "Block column index {} out of bounds (max {})",
                    bc_idx,
                    n_block_cols - 1
                )));
            }
        }

        Ok(Self {
            data,
            block_col_indices,
            block_row_ptr,
            nrows,
            ncols,
            br,
            bc,
            n_block_rows,
            n_block_cols,
        })
    }

    /// Create from COO-style triplets
    ///
    /// # Arguments
    /// * `rows` - Row indices of non-zero elements
    /// * `cols` - Column indices of non-zero elements
    /// * `values` - Values of non-zero elements
    /// * `nrows` - Number of rows
    /// * `ncols` - Number of columns
    /// * `br` - Block row size
    /// * `bc` - Block column size
    pub fn from_triplets(
        rows: &[usize],
        cols: &[usize],
        values: &[T],
        nrows: usize,
        ncols: usize,
        br: usize,
        bc: usize,
    ) -> SparseResult<Self> {
        if rows.len() != cols.len() || rows.len() != values.len() {
            return Err(SparseError::InconsistentData {
                reason: "rows, cols, values must have the same length".to_string(),
            });
        }
        if br == 0 || bc == 0 {
            return Err(SparseError::ValueError(
                "Block dimensions must be positive".to_string(),
            ));
        }

        let n_block_rows = (nrows + br - 1) / br;
        let n_block_cols = (ncols + bc - 1) / bc;
        let block_area = br * bc;

        // Accumulate elements into blocks using a map
        let mut block_map: std::collections::BTreeMap<(usize, usize), Vec<T>> =
            std::collections::BTreeMap::new();

        for (idx, (&r, &c)) in rows.iter().zip(cols.iter()).enumerate() {
            if r >= nrows || c >= ncols {
                return Err(SparseError::IndexOutOfBounds {
                    index: (r, c),
                    shape: (nrows, ncols),
                });
            }
            let block_r = r / br;
            let block_c = c / bc;
            let local_r = r % br;
            let local_c = c % bc;

            let block = block_map
                .entry((block_r, block_c))
                .or_insert_with(|| vec![T::sparse_zero(); block_area]);
            block[local_r * bc + local_c] = block[local_r * bc + local_c] + values[idx];
        }

        // Convert BTreeMap to BSR arrays (BTreeMap is already sorted by key)
        let mut data = Vec::new();
        let mut block_col_indices = Vec::new();
        let mut block_row_ptr = vec![0usize; n_block_rows + 1];

        let mut current_block_row = 0;
        let mut count = 0usize;

        for (&(br_idx, bc_idx), block_data) in &block_map {
            // Fill in any empty block rows before this one
            while current_block_row < br_idx {
                current_block_row += 1;
                block_row_ptr[current_block_row] = count;
            }

            data.extend_from_slice(block_data);
            block_col_indices.push(bc_idx);
            count += 1;
        }

        // Fill remaining block row pointers
        for i in (current_block_row + 1)..=n_block_rows {
            block_row_ptr[i] = count;
        }

        Ok(Self {
            data,
            block_col_indices,
            block_row_ptr,
            nrows,
            ncols,
            br,
            bc,
            n_block_rows,
            n_block_cols,
        })
    }

    /// Get the matrix shape
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Get the block size
    pub fn block_size(&self) -> (usize, usize) {
        (self.br, self.bc)
    }

    /// Get the number of stored blocks
    pub fn num_blocks(&self) -> usize {
        self.block_col_indices.len()
    }

    /// Get the number of non-zero scalar elements
    pub fn nnz(&self) -> usize {
        self.data
            .iter()
            .filter(|v| !SparseElement::is_zero(*v))
            .count()
    }

    /// Access a single block's data as a slice (row-major)
    pub fn block_data(&self, block_index: usize) -> Option<&[T]> {
        let block_area = self.br * self.bc;
        let start = block_index * block_area;
        let end = start + block_area;
        if end <= self.data.len() {
            Some(&self.data[start..end])
        } else {
            None
        }
    }

    /// Get an element at scalar position (i, j)
    pub fn get(&self, i: usize, j: usize) -> T {
        if i >= self.nrows || j >= self.ncols {
            return T::sparse_zero();
        }
        let block_r = i / self.br;
        let block_c = j / self.bc;
        let local_r = i % self.br;
        let local_c = j % self.bc;
        let block_area = self.br * self.bc;

        for k in self.block_row_ptr[block_r]..self.block_row_ptr[block_r + 1] {
            if self.block_col_indices[k] == block_c {
                return self.data[k * block_area + local_r * self.bc + local_c];
            }
        }
        T::sparse_zero()
    }

    /// Block matrix-vector multiplication: y = A * x
    ///
    /// Optimized for the flat block storage layout, processing one block at a time.
    pub fn matvec(&self, x: &[T]) -> SparseResult<Vec<T>> {
        if x.len() != self.ncols {
            return Err(SparseError::DimensionMismatch {
                expected: self.ncols,
                found: x.len(),
            });
        }

        let mut y = vec![T::sparse_zero(); self.nrows];
        let block_area = self.br * self.bc;

        for block_r in 0..self.n_block_rows {
            for k in self.block_row_ptr[block_r]..self.block_row_ptr[block_r + 1] {
                let block_c = self.block_col_indices[k];
                let block_start = k * block_area;

                for local_r in 0..self.br {
                    let row = block_r * self.br + local_r;
                    if row >= self.nrows {
                        break;
                    }
                    let mut sum = T::sparse_zero();
                    for local_c in 0..self.bc {
                        let col = block_c * self.bc + local_c;
                        if col < self.ncols {
                            sum =
                                sum + self.data[block_start + local_r * self.bc + local_c] * x[col];
                        }
                    }
                    y[row] = y[row] + sum;
                }
            }
        }

        Ok(y)
    }

    /// Convert to CsrArray
    pub fn to_csr(&self) -> SparseResult<CsrArray<T>>
    where
        T: Float + SparseElement + Div<Output = T> + 'static,
    {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        let block_area = self.br * self.bc;

        for block_r in 0..self.n_block_rows {
            for k in self.block_row_ptr[block_r]..self.block_row_ptr[block_r + 1] {
                let block_c = self.block_col_indices[k];
                let block_start = k * block_area;

                for local_r in 0..self.br {
                    let row = block_r * self.br + local_r;
                    if row >= self.nrows {
                        break;
                    }
                    for local_c in 0..self.bc {
                        let col = block_c * self.bc + local_c;
                        if col >= self.ncols {
                            break;
                        }
                        let v = self.data[block_start + local_r * self.bc + local_c];
                        if !SparseElement::is_zero(&v) {
                            rows.push(row);
                            cols.push(col);
                            vals.push(v);
                        }
                    }
                }
            }
        }

        CsrArray::from_triplets(&rows, &cols, &vals, (self.nrows, self.ncols), false)
    }

    /// Create from a CsrArray
    pub fn from_csr(csr: &CsrArray<T>, br: usize, bc: usize) -> SparseResult<Self>
    where
        T: Float + SparseElement + Div<Output = T> + 'static,
    {
        let (nrows, ncols) = csr.shape();
        let (row_indices, col_indices, values) = csr.find();

        let rows_vec: Vec<usize> = row_indices.to_vec();
        let cols_vec: Vec<usize> = col_indices.to_vec();
        let vals_vec: Vec<T> = values.to_vec();

        Self::from_triplets(&rows_vec, &cols_vec, &vals_vec, nrows, ncols, br, bc)
    }

    /// Convert to dense Array2
    pub fn to_dense(&self) -> Array2<T> {
        let mut result = Array2::zeros((self.nrows, self.ncols));
        let block_area = self.br * self.bc;

        for block_r in 0..self.n_block_rows {
            for k in self.block_row_ptr[block_r]..self.block_row_ptr[block_r + 1] {
                let block_c = self.block_col_indices[k];
                let block_start = k * block_area;

                for local_r in 0..self.br {
                    let row = block_r * self.br + local_r;
                    if row >= self.nrows {
                        break;
                    }
                    for local_c in 0..self.bc {
                        let col = block_c * self.bc + local_c;
                        if col >= self.ncols {
                            break;
                        }
                        result[[row, col]] = self.data[block_start + local_r * self.bc + local_c];
                    }
                }
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Block LU factorization
// ---------------------------------------------------------------------------

/// Result of a Block LU factorization
#[derive(Debug, Clone)]
pub struct BlockLUResult<T> {
    /// Lower triangular block factor (flat block data, block_size x block_size blocks)
    pub l_data: Vec<T>,
    pub l_block_col_indices: Vec<usize>,
    pub l_block_row_ptr: Vec<usize>,
    /// Upper triangular block factor
    pub u_data: Vec<T>,
    pub u_block_col_indices: Vec<usize>,
    pub u_block_row_ptr: Vec<usize>,
    /// Block size (must be square blocks for LU)
    pub block_size: usize,
    /// Matrix dimension (number of block rows = number of block cols)
    pub n_blocks: usize,
    /// Scalar dimension
    pub n: usize,
}

/// Perform in-place dense LU factorization of a small block (no pivoting).
/// Returns L and U as separate flat vectors (row-major, size bs*bs each).
/// L has unit diagonal, U has the pivots on the diagonal.
fn dense_block_lu<T>(block: &[T], bs: usize) -> SparseResult<(Vec<T>, Vec<T>)>
where
    T: Float + SparseElement + Debug + Copy + 'static,
{
    let mut a = block.to_vec();

    for k in 0..bs {
        let pivot = a[k * bs + k];
        if pivot.abs() < T::from(1e-14).unwrap_or(T::sparse_zero()) {
            return Err(SparseError::SingularMatrix(format!(
                "Zero pivot at position ({k}, {k}) in block LU"
            )));
        }
        for i in (k + 1)..bs {
            let factor = a[i * bs + k] / pivot;
            a[i * bs + k] = factor; // Store L factor in lower part
            for j in (k + 1)..bs {
                let ukj = a[k * bs + j];
                a[i * bs + j] = a[i * bs + j] - factor * ukj;
            }
        }
    }

    // Extract L and U
    let mut l = vec![T::sparse_zero(); bs * bs];
    let mut u = vec![T::sparse_zero(); bs * bs];

    for i in 0..bs {
        for j in 0..bs {
            if i > j {
                l[i * bs + j] = a[i * bs + j];
            } else if i == j {
                l[i * bs + j] = T::sparse_one();
                u[i * bs + j] = a[i * bs + j];
            } else {
                u[i * bs + j] = a[i * bs + j];
            }
        }
    }

    Ok((l, u))
}

/// Multiply two dense blocks: C = A * B (row-major, bs x bs)
fn dense_block_mul<T>(a: &[T], b: &[T], bs: usize) -> Vec<T>
where
    T: Float + SparseElement + Copy + 'static,
{
    let mut c = vec![T::sparse_zero(); bs * bs];
    for i in 0..bs {
        for k in 0..bs {
            let a_ik = a[i * bs + k];
            if !SparseElement::is_zero(&a_ik) {
                for j in 0..bs {
                    c[i * bs + j] = c[i * bs + j] + a_ik * b[k * bs + j];
                }
            }
        }
    }
    c
}

/// Subtract two dense blocks: C = A - B (row-major, bs x bs)
fn dense_block_sub<T>(a: &[T], b: &[T], bs: usize) -> Vec<T>
where
    T: Float + SparseElement + Copy + 'static,
{
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai - bi).collect()
}

/// Solve L * X = B for X, where L is lower triangular with unit diagonal (dense blocks, bs x bs).
fn dense_block_lower_solve<T>(l: &[T], b: &[T], bs: usize) -> Vec<T>
where
    T: Float + SparseElement + Copy + 'static,
{
    // X = L^{-1} B, column by column
    let mut x = b.to_vec();
    for col in 0..bs {
        for i in 0..bs {
            let mut sum = x[i * bs + col];
            for k in 0..i {
                sum = sum - l[i * bs + k] * x[k * bs + col];
            }
            x[i * bs + col] = sum; // L has unit diagonal
        }
    }
    x
}

/// Solve X * U = B for X, where U is upper triangular (dense blocks, bs x bs).
/// Column-by-column approach: X_ij = (B_ij - sum_{k<j} X_ik * U_kj) / U_jj
fn dense_block_upper_solve_right<T>(u: &[T], b: &[T], bs: usize) -> SparseResult<Vec<T>>
where
    T: Float + SparseElement + Copy + Debug + 'static,
{
    // X * U = B, solve for X
    // Process columns of X from left to right
    let mut x = vec![T::sparse_zero(); bs * bs];

    for j in 0..bs {
        let diag = u[j * bs + j];
        if diag.abs() < T::from(1e-14).unwrap_or(T::sparse_zero()) {
            return Err(SparseError::SingularMatrix(
                "Zero diagonal in upper triangular block".to_string(),
            ));
        }
        for i in 0..bs {
            let mut sum = b[i * bs + j];
            for k in 0..j {
                sum = sum - x[i * bs + k] * u[k * bs + j];
            }
            x[i * bs + j] = sum / diag;
        }
    }
    Ok(x)
}

/// Perform Block LU factorization of an EnhancedBsr matrix.
///
/// The matrix must be square with square blocks (br == bc) and the matrix
/// dimension must be exactly `n_block_rows * block_size`.
///
/// This implements the block IKJ variant of LU factorization:
///   For each block column k:
///     L(k,k), U(k,k) = LU( A(k,k) )
///     For i > k: L(i,k) = A(i,k) * U(k,k)^{-1}
///     For j > k: U(k,j) = L(k,k)^{-1} * A(k,j)
///     For i > k, j > k: A(i,j) -= L(i,k) * U(k,j)
pub fn block_lu<T>(bsr: &EnhancedBsr<T>) -> SparseResult<BlockLUResult<T>>
where
    T: Float + SparseElement + Debug + Copy + 'static,
{
    let (nrows, ncols) = bsr.shape();
    if nrows != ncols {
        return Err(SparseError::ValueError(
            "Block LU requires a square matrix".to_string(),
        ));
    }
    if bsr.br != bsr.bc {
        return Err(SparseError::ValueError(
            "Block LU requires square blocks (br == bc)".to_string(),
        ));
    }

    let bs = bsr.br;
    let n_blk = bsr.n_block_rows;
    let block_area = bs * bs;

    if nrows != n_blk * bs {
        return Err(SparseError::ValueError(
            "Matrix dimension must be exactly n_block_rows * block_size for Block LU".to_string(),
        ));
    }

    // Convert to a dense block representation for the factorization
    // blocks[i][j] stores the (i,j)-th block (or None if zero block)
    let mut blocks: Vec<Vec<Option<Vec<T>>>> = vec![vec![None; n_blk]; n_blk];

    for br_idx in 0..n_blk {
        for k in bsr.block_row_ptr[br_idx]..bsr.block_row_ptr[br_idx + 1] {
            let bc_idx = bsr.block_col_indices[k];
            let start = k * block_area;
            let end = start + block_area;
            blocks[br_idx][bc_idx] = Some(bsr.data[start..end].to_vec());
        }
    }

    let zero_block = vec![T::sparse_zero(); block_area];

    // Block LU factorization (IKJ variant)
    let mut l_blocks: Vec<Vec<Option<Vec<T>>>> = vec![vec![None; n_blk]; n_blk];
    let mut u_blocks: Vec<Vec<Option<Vec<T>>>> = vec![vec![None; n_blk]; n_blk];

    for k in 0..n_blk {
        // Factor the diagonal block: L_kk, U_kk = LU(A_kk)
        let a_kk = blocks[k][k].as_ref().unwrap_or(&zero_block);
        let (l_kk, u_kk) = dense_block_lu(a_kk, bs)?;
        l_blocks[k][k] = Some(l_kk.clone());
        u_blocks[k][k] = Some(u_kk.clone());

        // Compute L(i,k) for i > k: L(i,k) = A(i,k) * U(k,k)^{-1}
        for i in (k + 1)..n_blk {
            let a_ik = blocks[i][k].as_ref().unwrap_or(&zero_block);
            let has_nonzero = a_ik.iter().any(|v| !SparseElement::is_zero(v));
            if has_nonzero {
                let l_ik = dense_block_upper_solve_right(&u_kk, a_ik, bs)?;
                l_blocks[i][k] = Some(l_ik);
            }
        }

        // Compute U(k,j) for j > k: U(k,j) = L(k,k)^{-1} * A(k,j)
        for j in (k + 1)..n_blk {
            let a_kj = blocks[k][j].as_ref().unwrap_or(&zero_block);
            let has_nonzero = a_kj.iter().any(|v| !SparseElement::is_zero(v));
            if has_nonzero {
                let u_kj = dense_block_lower_solve(&l_kk, a_kj, bs);
                u_blocks[k][j] = Some(u_kj);
            }
        }

        // Update: A(i,j) -= L(i,k) * U(k,j) for i > k, j > k
        for i in (k + 1)..n_blk {
            let l_ik = match l_blocks[i][k].as_ref() {
                Some(b) => b.clone(),
                None => continue,
            };
            for j in (k + 1)..n_blk {
                let u_kj = match u_blocks[k][j].as_ref() {
                    Some(b) => b,
                    None => continue,
                };
                let product = dense_block_mul(&l_ik, u_kj, bs);
                let a_ij = blocks[i][j].get_or_insert_with(|| vec![T::sparse_zero(); block_area]);
                let updated = dense_block_sub(a_ij, &product, bs);
                *a_ij = updated;
            }
        }
    }

    // Convert L and U block arrays to flat BSR storage
    let mut l_data = Vec::new();
    let mut l_col_idx = Vec::new();
    let mut l_row_ptr = vec![0usize; n_blk + 1];

    let mut u_data = Vec::new();
    let mut u_col_idx = Vec::new();
    let mut u_row_ptr = vec![0usize; n_blk + 1];

    for i in 0..n_blk {
        for j in 0..n_blk {
            if let Some(ref blk) = l_blocks[i][j] {
                if blk.iter().any(|v| !SparseElement::is_zero(v)) {
                    l_data.extend_from_slice(blk);
                    l_col_idx.push(j);
                }
            }
        }
        l_row_ptr[i + 1] = l_col_idx.len();
    }

    for i in 0..n_blk {
        for j in 0..n_blk {
            if let Some(ref blk) = u_blocks[i][j] {
                if blk.iter().any(|v| !SparseElement::is_zero(v)) {
                    u_data.extend_from_slice(blk);
                    u_col_idx.push(j);
                }
            }
        }
        u_row_ptr[i + 1] = u_col_idx.len();
    }

    Ok(BlockLUResult {
        l_data,
        l_block_col_indices: l_col_idx,
        l_block_row_ptr: l_row_ptr,
        u_data,
        u_block_col_indices: u_col_idx,
        u_block_row_ptr: u_row_ptr,
        block_size: bs,
        n_blocks: n_blk,
        n: nrows,
    })
}

/// Solve Ax = b using a precomputed Block LU factorization.
///
/// Forward-substitutes with L, then back-substitutes with U.
pub fn block_lu_solve<T>(lu: &BlockLUResult<T>, b: &[T]) -> SparseResult<Vec<T>>
where
    T: Float + SparseElement + Debug + Copy + 'static,
{
    let n = lu.n;
    let bs = lu.block_size;
    let nb = lu.n_blocks;
    let block_area = bs * bs;

    if b.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: b.len(),
        });
    }

    // Forward substitution: L y = b
    // L's diagonal blocks are lower triangular with unit diagonal (not identity).
    // We must solve L_ii * y_i_block = (b_i - sum_{j<i} L_ij * y_j) for each block row i.
    let mut y = b.to_vec();
    for i in 0..nb {
        // y_i = y_i - sum_{j < i} L_{ij} * y_j
        for k_idx in lu.l_block_row_ptr[i]..lu.l_block_row_ptr[i + 1] {
            let j = lu.l_block_col_indices[k_idx];
            if j >= i {
                continue; // Only lower triangular part (off-diagonal)
            }
            let blk_start = k_idx * block_area;
            let blk = &lu.l_data[blk_start..blk_start + block_area];
            for lr in 0..bs {
                let row = i * bs + lr;
                if row >= n {
                    break;
                }
                let mut sum = T::sparse_zero();
                for lc in 0..bs {
                    let col = j * bs + lc;
                    if col < n {
                        sum = sum + blk[lr * bs + lc] * y[col];
                    }
                }
                y[row] = y[row] - sum;
            }
        }

        // Solve the diagonal block: L_ii * y_i_new = y_i_current
        // L_ii is lower triangular with unit diagonal
        let mut diag_blk: Option<&[T]> = None;
        for k_idx in lu.l_block_row_ptr[i]..lu.l_block_row_ptr[i + 1] {
            if lu.l_block_col_indices[k_idx] == i {
                let blk_start = k_idx * block_area;
                diag_blk = Some(&lu.l_data[blk_start..blk_start + block_area]);
                break;
            }
        }
        if let Some(diag) = diag_blk {
            // Forward solve within the block (unit lower triangular)
            for lr in 0..bs {
                let row = i * bs + lr;
                if row >= n {
                    break;
                }
                for lc in 0..lr {
                    let col = i * bs + lc;
                    if col < n {
                        y[row] = y[row] - diag[lr * bs + lc] * y[col];
                    }
                }
                // Unit diagonal: no division needed
            }
        }
    }

    // Back substitution: U x = y
    let mut x = y;
    for i in (0..nb).rev() {
        // x_i = x_i - sum_{j > i} U_{ij} * x_j
        for k_idx in lu.u_block_row_ptr[i]..lu.u_block_row_ptr[i + 1] {
            let j = lu.u_block_col_indices[k_idx];
            if j <= i {
                continue; // Only upper triangular off-diagonal
            }
            let blk_start = k_idx * block_area;
            let blk = &lu.u_data[blk_start..blk_start + block_area];
            for lr in 0..bs {
                let row = i * bs + lr;
                if row >= n {
                    break;
                }
                let mut sum = T::sparse_zero();
                for lc in 0..bs {
                    let col = j * bs + lc;
                    if col < n {
                        sum = sum + blk[lr * bs + lc] * x[col];
                    }
                }
                x[row] = x[row] - sum;
            }
        }

        // Solve U_{ii} * x_i_block = x_i_block (diagonal block)
        // Find diagonal block
        let mut diag_blk: Option<&[T]> = None;
        for k_idx in lu.u_block_row_ptr[i]..lu.u_block_row_ptr[i + 1] {
            if lu.u_block_col_indices[k_idx] == i {
                let blk_start = k_idx * block_area;
                diag_blk = Some(&lu.u_data[blk_start..blk_start + block_area]);
                break;
            }
        }
        let diag = diag_blk.ok_or_else(|| {
            SparseError::SingularMatrix("Missing diagonal block in U factor".to_string())
        })?;

        // Solve the small upper triangular system for this block
        for lr in (0..bs).rev() {
            let row = i * bs + lr;
            if row >= n {
                continue;
            }
            let d = diag[lr * bs + lr];
            if d.abs() < T::from(1e-14).unwrap_or(T::sparse_zero()) {
                return Err(SparseError::SingularMatrix(format!(
                    "Zero diagonal at block ({i},{i}), local row {lr}"
                )));
            }
            for lc in (lr + 1)..bs {
                let col = i * bs + lc;
                if col < n {
                    x[row] = x[row] - diag[lr * bs + lc] * x[col];
                }
            }
            x[row] = x[row] / d;
        }
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_enhanced_bsr_from_triplets() {
        // 4x4 matrix with 2x2 blocks:
        // [1 2 0 0]
        // [3 4 0 0]
        // [0 0 5 6]
        // [0 0 7 8]
        let rows = vec![0, 0, 1, 1, 2, 2, 3, 3];
        let cols = vec![0, 1, 0, 1, 2, 3, 2, 3];
        let vals: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let bsr =
            EnhancedBsr::from_triplets(&rows, &cols, &vals, 4, 4, 2, 2).expect("should succeed");

        assert_eq!(bsr.shape(), (4, 4));
        assert_eq!(bsr.block_size(), (2, 2));
        assert_eq!(bsr.num_blocks(), 2);
        assert_eq!(bsr.nnz(), 8);

        assert_relative_eq!(bsr.get(0, 0), 1.0);
        assert_relative_eq!(bsr.get(0, 1), 2.0);
        assert_relative_eq!(bsr.get(1, 0), 3.0);
        assert_relative_eq!(bsr.get(1, 1), 4.0);
        assert_relative_eq!(bsr.get(2, 2), 5.0);
        assert_relative_eq!(bsr.get(3, 3), 8.0);
        assert_relative_eq!(bsr.get(0, 2), 0.0);
    }

    #[test]
    fn test_enhanced_bsr_matvec() {
        // Same block diagonal matrix
        let rows = vec![0, 0, 1, 1, 2, 2, 3, 3];
        let cols = vec![0, 1, 0, 1, 2, 3, 2, 3];
        let vals: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let bsr =
            EnhancedBsr::from_triplets(&rows, &cols, &vals, 4, 4, 2, 2).expect("should succeed");
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = bsr.matvec(&x).expect("matvec should succeed");

        // y[0] = 1*1 + 2*2 = 5
        // y[1] = 3*1 + 4*2 = 11
        // y[2] = 5*3 + 6*4 = 39
        // y[3] = 7*3 + 8*4 = 53
        assert_relative_eq!(y[0], 5.0);
        assert_relative_eq!(y[1], 11.0);
        assert_relative_eq!(y[2], 39.0);
        assert_relative_eq!(y[3], 53.0);
    }

    #[test]
    fn test_enhanced_bsr_csr_roundtrip() {
        let rows = vec![0, 0, 1, 1, 2, 2, 3, 3];
        let cols = vec![0, 1, 0, 1, 2, 3, 2, 3];
        let vals: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let bsr =
            EnhancedBsr::from_triplets(&rows, &cols, &vals, 4, 4, 2, 2).expect("should succeed");
        let csr = bsr.to_csr().expect("to_csr should succeed");

        // Roundtrip: csr -> bsr -> dense should match
        let bsr2 = EnhancedBsr::from_csr(&csr, 2, 2).expect("from_csr should succeed");
        let dense1 = bsr.to_dense();
        let dense2 = bsr2.to_dense();

        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(dense1[[i, j]], dense2[[i, j]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_enhanced_bsr_to_dense() {
        // 6x6 with 3x3 blocks, some off-diagonal blocks
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals: Vec<f64> = Vec::new();

        // Block (0,0): identity * 2
        for i in 0..3 {
            rows.push(i);
            cols.push(i);
            vals.push(2.0);
        }
        // Block (0,1): ones
        for i in 0..3 {
            for j in 3..6 {
                rows.push(i);
                cols.push(j);
                vals.push(1.0);
            }
        }
        // Block (1,1): identity * 3
        for i in 3..6 {
            rows.push(i);
            cols.push(i);
            vals.push(3.0);
        }

        let bsr =
            EnhancedBsr::from_triplets(&rows, &cols, &vals, 6, 6, 3, 3).expect("should succeed");
        let dense = bsr.to_dense();

        assert_relative_eq!(dense[[0, 0]], 2.0);
        assert_relative_eq!(dense[[1, 1]], 2.0);
        assert_relative_eq!(dense[[2, 2]], 2.0);
        assert_relative_eq!(dense[[0, 3]], 1.0);
        assert_relative_eq!(dense[[3, 3]], 3.0);
        assert_relative_eq!(dense[[5, 5]], 3.0);
        assert_relative_eq!(dense[[3, 0]], 0.0);
    }

    #[test]
    fn test_block_lu_identity() {
        // 4x4 identity with 2x2 blocks
        let rows = vec![0, 1, 2, 3];
        let cols = vec![0, 1, 2, 3];
        let vals: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0];

        let bsr =
            EnhancedBsr::from_triplets(&rows, &cols, &vals, 4, 4, 2, 2).expect("should succeed");
        let lu = block_lu(&bsr).expect("block_lu should succeed");

        assert_eq!(lu.block_size, 2);
        assert_eq!(lu.n_blocks, 2);

        // Solve A x = b with b = [1, 2, 3, 4]
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let x = block_lu_solve(&lu, &b).expect("solve should succeed");

        for i in 0..4 {
            assert_relative_eq!(x[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_block_lu_dense_matrix() {
        // 4x4 dense block diagonal:
        // [[2, 1, 0, 0],
        //  [1, 3, 0, 0],
        //  [0, 0, 4, 1],
        //  [0, 0, 1, 2]]
        let rows = vec![0, 0, 1, 1, 2, 2, 3, 3];
        let cols = vec![0, 1, 0, 1, 2, 3, 2, 3];
        let vals: Vec<f64> = vec![2.0, 1.0, 1.0, 3.0, 4.0, 1.0, 1.0, 2.0];

        let bsr =
            EnhancedBsr::from_triplets(&rows, &cols, &vals, 4, 4, 2, 2).expect("should succeed");
        let lu = block_lu(&bsr).expect("block_lu should succeed");

        let b = vec![3.0, 4.0, 5.0, 3.0];
        let x = block_lu_solve(&lu, &b).expect("solve should succeed");

        // Verify: A x = b
        let y = bsr.matvec(&x).expect("matvec should succeed");
        for i in 0..4 {
            assert_relative_eq!(y[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_block_lu_full_coupling() {
        // 4x4 fully coupled matrix with 2x2 blocks
        // [[4, 1, 1, 0],
        //  [1, 4, 0, 1],
        //  [1, 0, 4, 1],
        //  [0, 1, 1, 4]]
        let rows = vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];
        let cols = vec![0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3];
        let vals: Vec<f64> = vec![4.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 4.0];

        let bsr =
            EnhancedBsr::from_triplets(&rows, &cols, &vals, 4, 4, 2, 2).expect("should succeed");
        let lu = block_lu(&bsr).expect("block_lu should succeed");

        let b = vec![1.0, 2.0, 3.0, 4.0];
        let x = block_lu_solve(&lu, &b).expect("solve should succeed");

        let y = bsr.matvec(&x).expect("matvec should succeed");
        for i in 0..4 {
            assert_relative_eq!(y[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_enhanced_bsr_empty_blocks() {
        // Matrix where some block rows are entirely zero
        let rows = vec![2, 2, 3, 3];
        let cols = vec![2, 3, 2, 3];
        let vals: Vec<f64> = vec![5.0, 6.0, 7.0, 8.0];

        let bsr =
            EnhancedBsr::from_triplets(&rows, &cols, &vals, 4, 4, 2, 2).expect("should succeed");

        assert_eq!(bsr.num_blocks(), 1);
        assert_relative_eq!(bsr.get(0, 0), 0.0);
        assert_relative_eq!(bsr.get(2, 2), 5.0);
        assert_relative_eq!(bsr.get(3, 3), 8.0);
    }

    #[test]
    fn test_enhanced_bsr_rectangular() {
        // 4x6 matrix with 2x3 blocks
        let rows = vec![0, 0, 0, 1, 1, 1];
        let cols = vec![0, 1, 2, 0, 1, 2];
        let vals: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let bsr =
            EnhancedBsr::from_triplets(&rows, &cols, &vals, 4, 6, 2, 3).expect("should succeed");
        assert_eq!(bsr.shape(), (4, 6));
        assert_eq!(bsr.block_size(), (2, 3));
        assert_relative_eq!(bsr.get(0, 0), 1.0);
        assert_relative_eq!(bsr.get(1, 2), 6.0);

        let x = vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let y = bsr.matvec(&x).expect("matvec");
        assert_relative_eq!(y[0], 6.0); // 1+2+3
        assert_relative_eq!(y[1], 15.0); // 4+5+6
        assert_relative_eq!(y[2], 0.0);
        assert_relative_eq!(y[3], 0.0);
    }
}
