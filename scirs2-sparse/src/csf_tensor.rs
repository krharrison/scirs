//! Compressed Sparse Fiber (CSF) format for sparse tensors
//!
//! CSF is a hierarchical compressed format for sparse tensors, analogous to
//! CSR/CSC for matrices but generalized to arbitrary dimensions. Each level
//! in the hierarchy compresses one tensor mode.
//!
//! For an N-dimensional tensor, CSF uses N levels:
//! - Level 0: the coarsest (outermost) mode
//! - Level N-1: the finest (innermost) mode, where values are stored
//!
//! Each level `l` has:
//! - `fptr[l]`: fiber pointers (like `indptr` in CSR)
//! - `fids[l]`: fiber indices (like `indices` in CSR)
//!
//! The values are associated with the leaf-level fibers.
//!
//! This provides excellent compression for tensors with hierarchical sparsity
//! patterns (e.g., most real-world tensors).

use crate::csr_array::CsrArray;
use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use crate::tensor_sparse::SparseTensor;
use scirs2_core::numeric::{Float, SparseElement};
use std::fmt::Debug;
use std::ops::Div;

/// Compressed Sparse Fiber (CSF) tensor format
///
/// Stores a sparse tensor using a hierarchical compressed structure.
/// The modes are ordered from coarsest (level 0) to finest (last level).
/// Values are stored only at the leaf level.
#[derive(Debug, Clone)]
pub struct CsfTensor<T> {
    /// Fiber pointers for each level.
    /// `fptr[l]` has length = (number of fibers at level l) + 1
    fptr: Vec<Vec<usize>>,
    /// Fiber indices for each level.
    /// `fids[l]` contains the mode-l coordinate for each fiber.
    fids: Vec<Vec<usize>>,
    /// Values at the leaf level
    values: Vec<T>,
    /// Tensor shape
    shape: Vec<usize>,
    /// Mode ordering (which tensor mode corresponds to each CSF level)
    mode_order: Vec<usize>,
}

impl<T> CsfTensor<T>
where
    T: Float + SparseElement + Debug + Copy + 'static,
{
    /// Create a CSF tensor from a COO-format SparseTensor.
    ///
    /// The `mode_order` specifies which tensor mode corresponds to each CSF level.
    /// For example, for a 3D tensor with `mode_order = [0, 1, 2]`, level 0
    /// compresses mode 0, level 1 compresses mode 1, and values are indexed by mode 2.
    ///
    /// # Arguments
    /// * `tensor` - The sparse tensor in COO format
    /// * `mode_order` - Ordering of modes (must be a permutation of 0..ndim)
    pub fn from_sparse_tensor(tensor: &SparseTensor<T>, mode_order: &[usize]) -> SparseResult<Self>
    where
        T: std::iter::Sum,
    {
        let ndim = tensor.ndim();
        if mode_order.len() != ndim {
            return Err(SparseError::ValueError(format!(
                "mode_order length {} must equal tensor ndim {}",
                mode_order.len(),
                ndim
            )));
        }

        // Validate mode_order is a permutation
        let mut sorted_modes = mode_order.to_vec();
        sorted_modes.sort();
        for (i, &m) in sorted_modes.iter().enumerate() {
            if m != i {
                return Err(SparseError::ValueError(
                    "mode_order must be a permutation of 0..ndim".to_string(),
                ));
            }
        }

        let nnz = tensor.nnz();
        if nnz == 0 {
            // Empty tensor
            let fptr = vec![vec![0, 0]; ndim];
            let fids = vec![Vec::new(); ndim];
            return Ok(Self {
                fptr,
                fids,
                values: Vec::new(),
                shape: tensor.shape.clone(),
                mode_order: mode_order.to_vec(),
            });
        }

        // Build sorted list of non-zero entries, sorted by the CSF level order
        // Each entry: (coordinates_in_mode_order, value)
        let mut entries: Vec<(Vec<usize>, T)> = (0..nnz)
            .map(|i| {
                let coords: Vec<usize> = mode_order.iter().map(|&m| tensor.indices[m][i]).collect();
                (coords, tensor.values[i])
            })
            .collect();

        // Sort entries lexicographically by coordinates
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        // Build CSF structure level by level
        let mut fptr: Vec<Vec<usize>> = Vec::with_capacity(ndim);
        let mut fids: Vec<Vec<usize>> = Vec::with_capacity(ndim);
        let mut values: Vec<T> = Vec::new();

        // Level 0 (root level): unique values in the first coordinate
        let mut level0_ids: Vec<usize> = Vec::new();
        let mut level0_ptr: Vec<usize> = vec![0];

        // We build all levels simultaneously by tracking coordinate groups
        // at each level.
        //
        // Algorithm: Process entries in sorted order. At each level, track
        // the current coordinate prefix. When it changes, we close the
        // current fiber and start a new one.

        // Initialize per-level structures
        for _ in 0..ndim {
            fptr.push(Vec::new());
            fids.push(Vec::new());
        }

        // For each level, we track the current prefix
        let mut prev_prefix: Vec<Option<usize>> = vec![None; ndim];
        let mut level_counts: Vec<usize> = vec![0; ndim];

        // Process each level from 0 to ndim-1
        // We use a recursive-like approach: for each entry, check which levels
        // need new fibers.
        for (entry_idx, (coords, val)) in entries.iter().enumerate() {
            // Determine the first level where the prefix changes
            let mut change_level = ndim; // no change needed
            for l in 0..ndim {
                if prev_prefix[l] != Some(coords[l]) {
                    change_level = l;
                    break;
                }
            }

            // Close fibers at levels below change_level (from deepest to change_level)
            // and open new fibers from change_level downward

            for l in change_level..ndim {
                // At level l, we need to start a new coordinate
                fids[l].push(coords[l]);

                // If this is not the leaf level, record pointer for next level
                if l < ndim - 1 {
                    // The pointer for level l+1 is the current size of fids[l+1]
                    if fptr[l].is_empty() || l == change_level {
                        // Only push a new pointer when we actually start a new fiber at this level
                        fptr[l].push(fids[l].len() - 1);
                    }
                }

                prev_prefix[l] = Some(coords[l]);
            }

            // Store value at leaf level
            values.push(*val);
        }

        // Now build proper fptr structures
        // Reset and rebuild using the sorted entries
        fptr = vec![Vec::new(); ndim];
        fids = vec![Vec::new(); ndim];
        values = Vec::new();

        // Rebuild properly using group detection
        self::build_csf_levels(&entries, &mut fptr, &mut fids, &mut values, ndim);

        Ok(Self {
            fptr,
            fids,
            values,
            shape: tensor.shape.clone(),
            mode_order: mode_order.to_vec(),
        })
    }

    /// Get the tensor shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the number of stored non-zero values
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get the mode ordering
    pub fn mode_order(&self) -> &[usize] {
        &self.mode_order
    }

    /// Get the fiber pointers at a given level
    pub fn fiber_pointers(&self, level: usize) -> Option<&[usize]> {
        self.fptr.get(level).map(|v| v.as_slice())
    }

    /// Get the fiber indices at a given level
    pub fn fiber_indices(&self, level: usize) -> Option<&[usize]> {
        self.fids.get(level).map(|v| v.as_slice())
    }

    /// Get the stored values
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Look up a value by coordinate
    pub fn get(&self, coords: &[usize]) -> T {
        if coords.len() != self.ndim() {
            return T::sparse_zero();
        }

        // Reorder coordinates according to mode_order
        let ordered_coords: Vec<usize> = self.mode_order.iter().map(|&m| coords[m]).collect();

        // Navigate the CSF tree
        self.search_tree(&ordered_coords, 0, 0)
    }

    /// Recursive tree search for a coordinate
    fn search_tree(&self, ordered_coords: &[usize], level: usize, fiber_idx: usize) -> T {
        let ndim = self.ndim();

        if level == ndim - 1 {
            // Leaf level: search fids for the coordinate
            let start = if level == 0 {
                0
            } else {
                self.fptr[level - 1].get(fiber_idx).copied().unwrap_or(0)
            };
            let end = if level == 0 {
                self.fids[level].len()
            } else {
                self.fptr[level - 1]
                    .get(fiber_idx + 1)
                    .copied()
                    .unwrap_or(self.fids[level].len())
            };

            for i in start..end {
                if i < self.fids[level].len() && self.fids[level][i] == ordered_coords[level] {
                    let val_idx = i; // leaf fids index maps directly to values
                    if val_idx < self.values.len() {
                        return self.values[val_idx];
                    }
                }
            }
            return T::sparse_zero();
        }

        // Non-leaf level: find the matching fiber and recurse
        let start = if level == 0 {
            0
        } else {
            self.fptr[level - 1].get(fiber_idx).copied().unwrap_or(0)
        };
        let end = if level == 0 {
            self.fids[level].len()
        } else {
            self.fptr[level - 1]
                .get(fiber_idx + 1)
                .copied()
                .unwrap_or(self.fids[level].len())
        };

        for i in start..end {
            if i < self.fids[level].len() && self.fids[level][i] == ordered_coords[level] {
                return self.search_tree(ordered_coords, level + 1, i);
            }
        }

        T::sparse_zero()
    }

    /// Convert back to a COO-format SparseTensor
    pub fn to_sparse_tensor(&self) -> SparseResult<SparseTensor<T>>
    where
        T: std::iter::Sum,
    {
        let ndim = self.ndim();
        let mut indices: Vec<Vec<usize>> = vec![Vec::new(); ndim];
        let mut values: Vec<T> = Vec::new();

        // Build inverse mode order
        let mut inv_mode_order = vec![0usize; ndim];
        for (csf_level, &tensor_mode) in self.mode_order.iter().enumerate() {
            inv_mode_order[tensor_mode] = csf_level;
        }

        // Traverse the CSF tree to extract all non-zero entries
        let mut coord_stack: Vec<usize> = vec![0; ndim];
        self.traverse_tree(
            0,
            0,
            &mut coord_stack,
            &mut indices,
            &mut values,
            &inv_mode_order,
        );

        SparseTensor::new(indices, values, self.shape.clone())
    }

    /// Recursive traversal to extract entries
    fn traverse_tree(
        &self,
        level: usize,
        fiber_idx: usize,
        coord_stack: &mut Vec<usize>,
        indices: &mut Vec<Vec<usize>>,
        values: &mut Vec<T>,
        inv_mode_order: &[usize],
    ) {
        let ndim = self.ndim();

        let start = if level == 0 {
            0
        } else {
            self.fptr[level - 1].get(fiber_idx).copied().unwrap_or(0)
        };
        let end = if level == 0 {
            self.fids[level].len()
        } else {
            self.fptr[level - 1]
                .get(fiber_idx + 1)
                .copied()
                .unwrap_or(self.fids[level].len())
        };

        for i in start..end {
            if i >= self.fids[level].len() {
                break;
            }
            coord_stack[level] = self.fids[level][i];

            if level == ndim - 1 {
                // Leaf: emit the entry
                if i < self.values.len() {
                    for mode in 0..ndim {
                        let csf_level = inv_mode_order[mode];
                        indices[mode].push(coord_stack[csf_level]);
                    }
                    values.push(self.values[i]);
                }
            } else {
                self.traverse_tree(level + 1, i, coord_stack, indices, values, inv_mode_order);
            }
        }
    }

    /// Tensor contraction along a specified mode with a vector.
    ///
    /// Computes the mode-n product with a vector: result = T x_n v
    /// The result is a tensor of one lower dimension.
    ///
    /// # Arguments
    /// * `mode` - The tensor mode to contract
    /// * `vector` - The vector to contract with (length = `shape[mode]`)
    pub fn contract_vector(&self, mode: usize, vector: &[T]) -> SparseResult<SparseTensor<T>>
    where
        T: std::iter::Sum,
    {
        let ndim = self.ndim();
        if mode >= ndim {
            return Err(SparseError::ValueError(format!(
                "Mode {} exceeds tensor dimensions {}",
                mode, ndim
            )));
        }
        if vector.len() != self.shape[mode] {
            return Err(SparseError::DimensionMismatch {
                expected: self.shape[mode],
                found: vector.len(),
            });
        }
        if ndim < 2 {
            return Err(SparseError::ValueError(
                "Cannot contract a 1D tensor to 0D".to_string(),
            ));
        }

        // Convert to COO, perform contraction, return result
        let coo = self.to_sparse_tensor()?;

        // Build result tensor: shape without the contracted mode
        let new_shape: Vec<usize> = (0..ndim)
            .filter(|&m| m != mode)
            .map(|m| self.shape[m])
            .collect();
        let new_ndim = new_shape.len();

        // Accumulate contracted values
        let mut result_map: std::collections::HashMap<Vec<usize>, T> =
            std::collections::HashMap::new();

        for i in 0..coo.nnz() {
            let mode_idx = coo.indices[mode][i];
            let scale = vector[mode_idx];

            if SparseElement::is_zero(&scale) {
                continue;
            }

            let key: Vec<usize> = (0..ndim)
                .filter(|&m| m != mode)
                .map(|m| coo.indices[m][i])
                .collect();

            let entry = result_map.entry(key).or_insert(T::sparse_zero());
            *entry = *entry + coo.values[i] * scale;
        }

        // Build result tensor
        let mut new_indices: Vec<Vec<usize>> = vec![Vec::new(); new_ndim];
        let mut new_values: Vec<T> = Vec::new();

        for (key, val) in &result_map {
            if !SparseElement::is_zero(val) {
                for (d, &k) in key.iter().enumerate() {
                    new_indices[d].push(k);
                }
                new_values.push(*val);
            }
        }

        if new_values.is_empty() {
            // Return empty tensor
            new_indices = vec![Vec::new(); new_ndim];
        }

        SparseTensor::new(new_indices, new_values, new_shape)
    }

    /// Mode-n product with a matrix: result = T x_n M
    ///
    /// The result has the same number of dimensions as the input tensor,
    /// but the size of mode `n` changes from `shape[n]` to `M.nrows`.
    ///
    /// # Arguments
    /// * `mode` - The tensor mode to multiply along
    /// * `matrix` - The matrix (nrows x `shape[mode]`)
    pub fn mode_n_product(&self, mode: usize, matrix: &CsrArray<T>) -> SparseResult<SparseTensor<T>>
    where
        T: Float + SparseElement + Div<Output = T> + std::iter::Sum + 'static,
    {
        let ndim = self.ndim();
        if mode >= ndim {
            return Err(SparseError::ValueError(format!(
                "Mode {} exceeds tensor dimensions {}",
                mode, ndim
            )));
        }
        let (mat_rows, mat_cols) = matrix.shape();
        if mat_cols != self.shape[mode] {
            return Err(SparseError::DimensionMismatch {
                expected: self.shape[mode],
                found: mat_cols,
            });
        }

        let coo = self.to_sparse_tensor()?;

        let mut new_shape = self.shape.clone();
        new_shape[mode] = mat_rows;

        // For each non-zero in the tensor, distribute it across matrix rows
        let mut result_map: std::collections::HashMap<Vec<usize>, T> =
            std::collections::HashMap::new();

        for i in 0..coo.nnz() {
            let mode_idx = coo.indices[mode][i];
            let tensor_val = coo.values[i];

            // Multiply by each non-zero in column mode_idx of the matrix
            for new_mode_idx in 0..mat_rows {
                let m_val = matrix.get(new_mode_idx, mode_idx);
                if SparseElement::is_zero(&m_val) {
                    continue;
                }

                let mut key: Vec<usize> = (0..ndim).map(|m| coo.indices[m][i]).collect();
                key[mode] = new_mode_idx;

                let entry = result_map.entry(key).or_insert(T::sparse_zero());
                *entry = *entry + tensor_val * m_val;
            }
        }

        let mut new_indices: Vec<Vec<usize>> = vec![Vec::new(); ndim];
        let mut new_values: Vec<T> = Vec::new();

        for (key, val) in &result_map {
            if !SparseElement::is_zero(val) {
                for (d, &k) in key.iter().enumerate() {
                    new_indices[d].push(k);
                }
                new_values.push(*val);
            }
        }

        if new_values.is_empty() {
            // Return an empty tensor with valid indices
            return Ok(SparseTensor {
                indices: (0..ndim).map(|_| Vec::new()).collect(),
                values: Vec::new(),
                shape: new_shape,
            });
        }

        SparseTensor::new(new_indices, new_values, new_shape)
    }

    /// Get memory usage estimate (bytes)
    pub fn memory_usage(&self) -> usize {
        let mut total = 0usize;
        for fp in &self.fptr {
            total += fp.len() * std::mem::size_of::<usize>();
        }
        for fi in &self.fids {
            total += fi.len() * std::mem::size_of::<usize>();
        }
        total += self.values.len() * std::mem::size_of::<T>();
        total += self.shape.len() * std::mem::size_of::<usize>();
        total += self.mode_order.len() * std::mem::size_of::<usize>();
        total
    }
}

/// Build CSF level structures from sorted entries
///
/// The CSF structure uses `fptr[level]` to index into `fids[level+1]`.
/// For each fiber at level `l` (identified by `fids[l][i]`), its children
/// in `fids[l+1]` span the range `fptr[l][i]..fptr[l][i+1]`.
///
/// `fptr[l]` has exactly `fids[l].len() + 1` entries.
/// Values are stored at the leaf level, indexed in parallel with `fids[ndim-1]`.
fn build_csf_levels<T: Copy>(
    entries: &[(Vec<usize>, T)],
    fptr: &mut Vec<Vec<usize>>,
    fids: &mut Vec<Vec<usize>>,
    values: &mut Vec<T>,
    ndim: usize,
) {
    if entries.is_empty() || ndim == 0 {
        return;
    }

    // Initialize fptr and fids for all levels
    for l in 0..ndim {
        fptr[l] = Vec::new();
        fids[l] = Vec::new();
    }

    // Use recursive grouping (no sentinel pushed inside recursion)
    build_level(entries, fptr, fids, values, 0, ndim);

    // Now add the sentinel pointer to each non-leaf level.
    // fptr[l] should have fids[l].len() + 1 entries.
    // build_level pushes one entry per fiber (the start pointer), so we
    // need to append the final sentinel = fids[l+1].len() for each non-leaf level.
    for l in 0..(ndim - 1) {
        fptr[l].push(fids[l + 1].len());
    }
}

/// Groups of entries at one CSF level: (coordinate, list of (coords, value) entries).
type LevelGroups<T> = Vec<(usize, Vec<(Vec<usize>, T)>)>;

/// Recursively build one level of the CSF structure.
///
/// For each group at this level, push the coordinate to `fids[level]`,
/// push the start pointer to `fptr[level]`, then recurse.
/// Does NOT push a sentinel; the caller handles that.
fn build_level<T: Copy>(
    entries: &[(Vec<usize>, T)],
    fptr: &mut Vec<Vec<usize>>,
    fids: &mut Vec<Vec<usize>>,
    values: &mut Vec<T>,
    level: usize,
    ndim: usize,
) {
    if entries.is_empty() {
        return;
    }

    if level == ndim - 1 {
        // Leaf level: just store fids and values
        for (coords, val) in entries {
            fids[level].push(coords[level]);
            values.push(*val);
        }
        return;
    }

    // Group entries by coordinate at this level
    let mut groups: LevelGroups<T> = Vec::new();
    let mut current_coord = entries[0].0[level];
    let mut current_group: Vec<(Vec<usize>, T)> = Vec::new();

    for (coords, val) in entries {
        if coords[level] != current_coord {
            groups.push((current_coord, std::mem::take(&mut current_group)));
            current_coord = coords[level];
        }
        current_group.push((coords.clone(), *val));
    }
    groups.push((current_coord, current_group));

    // For this level, record fids and pointers into next level
    for (coord, group) in &groups {
        fids[level].push(*coord);
        // Pointer: where the children of this fiber start in the next level's fids
        fptr[level].push(fids[level + 1].len());
        build_level(group, fptr, fids, values, level + 1, ndim);
    }
    // No sentinel here -- the caller (build_csf_levels) adds it after all recursion is done.
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_tensor_3d() -> SparseTensor<f64> {
        // 2x3x4 tensor with 5 non-zeros
        let indices = vec![
            vec![0, 0, 0, 1, 1], // mode 0
            vec![0, 1, 2, 0, 2], // mode 1
            vec![0, 1, 3, 2, 0], // mode 2
        ];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = vec![2, 3, 4];
        SparseTensor::new(indices, values, shape).expect("failed to create tensor")
    }

    #[test]
    fn test_csf_from_sparse_tensor() {
        let tensor = create_test_tensor_3d();
        let csf = CsfTensor::from_sparse_tensor(&tensor, &[0, 1, 2]).expect("CSF creation failed");

        assert_eq!(csf.ndim(), 3);
        assert_eq!(csf.nnz(), 5);
        assert_eq!(csf.shape(), &[2, 3, 4]);
        assert_eq!(csf.mode_order(), &[0, 1, 2]);
    }

    #[test]
    fn test_csf_roundtrip() {
        let tensor = create_test_tensor_3d();
        let csf = CsfTensor::from_sparse_tensor(&tensor, &[0, 1, 2]).expect("CSF creation failed");

        let recovered = csf.to_sparse_tensor().expect("to_sparse_tensor failed");

        assert_eq!(recovered.nnz(), tensor.nnz());

        // Check all values match
        for i in 0..tensor.nnz() {
            let coords: Vec<usize> = (0..3).map(|d| tensor.indices[d][i]).collect();
            let orig_val = tensor.get(&coords);
            let rec_val = recovered.get(&coords);
            assert_relative_eq!(orig_val, rec_val, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_csf_get() {
        let tensor = create_test_tensor_3d();
        let csf = CsfTensor::from_sparse_tensor(&tensor, &[0, 1, 2]).expect("CSF creation failed");

        assert_relative_eq!(csf.get(&[0, 0, 0]), 1.0, epsilon = 1e-12);
        assert_relative_eq!(csf.get(&[0, 1, 1]), 2.0, epsilon = 1e-12);
        assert_relative_eq!(csf.get(&[0, 2, 3]), 3.0, epsilon = 1e-12);
        assert_relative_eq!(csf.get(&[1, 0, 2]), 4.0, epsilon = 1e-12);
        assert_relative_eq!(csf.get(&[1, 2, 0]), 5.0, epsilon = 1e-12);
        assert_relative_eq!(csf.get(&[0, 0, 1]), 0.0, epsilon = 1e-12); // zero
    }

    #[test]
    fn test_csf_different_mode_order() {
        let tensor = create_test_tensor_3d();

        // Create with reversed mode order
        let csf = CsfTensor::from_sparse_tensor(&tensor, &[2, 1, 0]).expect("CSF creation failed");
        assert_eq!(csf.nnz(), 5);
        assert_eq!(csf.mode_order(), &[2, 1, 0]);

        // Should still look up correctly by original coordinates
        assert_relative_eq!(csf.get(&[0, 0, 0]), 1.0, epsilon = 1e-12);
        assert_relative_eq!(csf.get(&[1, 2, 0]), 5.0, epsilon = 1e-12);

        // Roundtrip should work
        let recovered = csf.to_sparse_tensor().expect("roundtrip failed");
        for i in 0..tensor.nnz() {
            let coords: Vec<usize> = (0..3).map(|d| tensor.indices[d][i]).collect();
            assert_relative_eq!(tensor.get(&coords), recovered.get(&coords), epsilon = 1e-12);
        }
    }

    #[test]
    fn test_csf_contract_vector() {
        // Simple 2x3 tensor contracted along mode 1
        let indices = vec![
            vec![0, 0, 1], // mode 0
            vec![0, 1, 0], // mode 1
        ];
        let values = vec![1.0, 2.0, 3.0];
        let shape = vec![2, 3];
        let tensor = SparseTensor::new(indices, values, shape).expect("create tensor");

        let csf = CsfTensor::from_sparse_tensor(&tensor, &[0, 1]).expect("CSF creation");

        let vector = vec![1.0, 2.0, 0.0]; // contract mode 1
        let result = csf.contract_vector(1, &vector).expect("contract_vector");

        // Result should be 1D tensor of shape [2]
        assert_eq!(result.shape, vec![2]);

        // result[0] = 1*1 + 2*2 = 5
        // result[1] = 3*1 = 3
        assert_relative_eq!(result.get(&[0]), 5.0, epsilon = 1e-12);
        assert_relative_eq!(result.get(&[1]), 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_csf_mode_n_product() {
        // 2x3 tensor, mode-1 product with a 2x3 matrix
        let indices = vec![
            vec![0, 0, 1], // mode 0
            vec![0, 2, 1], // mode 1
        ];
        let values = vec![1.0, 3.0, 2.0];
        let shape = vec![2, 3];
        let tensor = SparseTensor::new(indices, values, shape).expect("create tensor");
        let csf = CsfTensor::from_sparse_tensor(&tensor, &[0, 1]).expect("CSF");

        // 2x3 matrix
        let m_rows = vec![0, 0, 1, 1];
        let m_cols = vec![0, 2, 0, 1];
        let m_vals = vec![1.0, 1.0, 0.5, 2.0];
        let matrix =
            CsrArray::from_triplets(&m_rows, &m_cols, &m_vals, (2, 3), false).expect("matrix");

        let result = csf.mode_n_product(1, &matrix).expect("mode_n_product");

        // Result shape: [2, 2] (mode 1 changes from 3 to 2)
        assert_eq!(result.shape, vec![2, 2]);

        // result[0,0] = tensor[0,0]*M[0,0] + tensor[0,2]*M[0,2] = 1*1 + 3*1 = 4
        // result[0,1] = tensor[0,0]*M[1,0] + tensor[0,1]*M[1,1] = 1*0.5 + 0*2 = 0.5
        // result[1,0] = tensor[1,1]*M[0,1] = 0 (M[0,1] = 0)
        // result[1,1] = tensor[1,1]*M[1,1] = 2*2 = 4
        assert_relative_eq!(result.get(&[0, 0]), 4.0, epsilon = 1e-12);
        assert_relative_eq!(result.get(&[0, 1]), 0.5, epsilon = 1e-12);
        assert_relative_eq!(result.get(&[1, 1]), 4.0, epsilon = 1e-12);
    }

    #[test]
    fn test_csf_memory_usage() {
        let tensor = create_test_tensor_3d();
        let csf = CsfTensor::from_sparse_tensor(&tensor, &[0, 1, 2]).expect("CSF creation failed");
        let mem = csf.memory_usage();
        assert!(mem > 0);
    }

    #[test]
    fn test_csf_empty_tensor() {
        let indices = vec![Vec::<usize>::new(), Vec::<usize>::new()];
        let values: Vec<f64> = Vec::new();
        let shape = vec![3, 4];
        let tensor = SparseTensor::new(indices, values, shape).expect("empty tensor");
        let csf = CsfTensor::from_sparse_tensor(&tensor, &[0, 1]).expect("CSF");
        assert_eq!(csf.nnz(), 0);
    }

    #[test]
    fn test_csf_3d_roundtrip_with_permutation() {
        // Larger 3x4x5 tensor
        let indices = vec![
            vec![0, 0, 1, 2, 2, 2], // mode 0
            vec![0, 3, 1, 0, 2, 3], // mode 1
            vec![0, 4, 2, 1, 3, 4], // mode 2
        ];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![3, 4, 5];
        let tensor = SparseTensor::new(indices, values, shape).expect("tensor");

        for perm in &[[0, 1, 2], [1, 0, 2], [2, 0, 1], [0, 2, 1]] {
            let csf = CsfTensor::from_sparse_tensor(&tensor, perm).expect("CSF");
            assert_eq!(csf.nnz(), 6);

            let recovered = csf.to_sparse_tensor().expect("roundtrip");
            assert_eq!(recovered.nnz(), 6);

            for i in 0..tensor.nnz() {
                let coords: Vec<usize> = (0..3).map(|d| tensor.indices[d][i]).collect();
                assert_relative_eq!(tensor.get(&coords), recovered.get(&coords), epsilon = 1e-12,);
            }
        }
    }
}
