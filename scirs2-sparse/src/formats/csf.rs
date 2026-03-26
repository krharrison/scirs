//! Compressed Sparse Fiber (CSF) format for arbitrary-order sparse tensors
//!
//! This module provides a mode-generic CSF format that works for 2D (matrices),
//! 3D, or higher-order tensors. CSF generalises the CSR representation to
//! arbitrary dimensions using hierarchical index arrays.
//!
//! For an N-dimensional tensor, CSF uses N levels of compressed indices:
//! - `fib_ptr[mode]` — fiber pointers (analogous to `indptr` in CSR)
//! - `fib_idx[mode]` — fiber indices (analogous to `indices` in CSR)
//! - Values are stored at the leaf level.
//!
//! This is a standalone implementation with a clean public API for direct
//! COO-style construction, element access, fiber extraction, and mode-n
//! matricization.
//!
//! # References
//!
//! - Smith, S. & Karypis, G. (2015). "Tensor-Matrix Products with a Compressed
//!   Sparse Tensor." IPDPS Workshop on Irregular Applications.
//! - Li, J., et al. (2018). "HiCOO: Hierarchical Storage of Sparse Tensors."
//!   SC'18.

use crate::error::{SparseError, SparseResult};
use scirs2_core::numeric::{SparseElement, Zero};
use std::fmt::Debug;

/// Compressed Sparse Fiber (CSF) tensor.
///
/// Stores a sparse tensor of arbitrary order using hierarchical compressed
/// fiber arrays. Modes are ordered as given by `mode_order`.
#[derive(Debug, Clone)]
pub struct CsfTensor<T> {
    /// Shape of the tensor (one entry per mode).
    pub shape: Vec<usize>,
    /// Mode ordering: `mode_order[level]` = which original mode is at this CSF level.
    pub mode_order: Vec<usize>,
    /// Fiber pointers for each non-leaf level.
    /// `fib_ptr[l]` has length = `fib_idx[l].len() + 1`.
    /// Children of fiber `i` at level `l` are `fib_idx[l+1][fib_ptr[l][i]..fib_ptr[l][i+1]]`.
    pub fib_ptr: Vec<Vec<usize>>,
    /// Fiber indices for each level.
    /// `fib_idx[l]` stores the coordinate values at CSF level `l`.
    pub fib_idx: Vec<Vec<usize>>,
    /// Values at the leaf level. `values[i]` corresponds to `fib_idx[ndim-1][i]`.
    pub values: Vec<T>,
}

/// A COO-style entry for tensor construction.
#[derive(Debug, Clone)]
struct CooEntry<T: Copy> {
    coords: Vec<usize>,
    value: T,
}

impl<T> CsfTensor<T>
where
    T: Clone + Copy + Zero + SparseElement + Debug,
{
    /// Construct a CSF tensor from COO-style data (coordinate lists + values).
    ///
    /// # Arguments
    ///
    /// * `indices` - A slice of index arrays, one per mode. `indices[m][i]` is the
    ///   mode-m coordinate of the i-th non-zero.
    /// * `values` - Non-zero values. Length must match `indices[0].len()`.
    /// * `shape` - Shape of the tensor.
    /// * `mode_order` - Permutation of `0..ndim` specifying which tensor mode
    ///   goes at each CSF level. Pass `None` for natural order.
    pub fn from_coo(
        indices: &[Vec<usize>],
        values: &[T],
        shape: &[usize],
        mode_order: Option<&[usize]>,
    ) -> SparseResult<Self> {
        let ndim = shape.len();
        if indices.len() != ndim {
            return Err(SparseError::ValueError(format!(
                "indices length {} != ndim {}",
                indices.len(),
                ndim
            )));
        }
        let nnz = values.len();
        if ndim > 0 && indices[0].len() != nnz {
            return Err(SparseError::ValueError(
                "indices and values length mismatch".to_string(),
            ));
        }

        let order: Vec<usize> = match mode_order {
            Some(o) => {
                if o.len() != ndim {
                    return Err(SparseError::ValueError(
                        "mode_order length must match ndim".to_string(),
                    ));
                }
                let mut sorted = o.to_vec();
                sorted.sort_unstable();
                for (i, &v) in sorted.iter().enumerate() {
                    if v != i {
                        return Err(SparseError::ValueError(
                            "mode_order must be a permutation of 0..ndim".to_string(),
                        ));
                    }
                }
                o.to_vec()
            }
            None => (0..ndim).collect(),
        };

        if nnz == 0 {
            let fib_ptr = if ndim > 1 {
                (0..ndim - 1).map(|_| vec![0usize]).collect()
            } else {
                Vec::new()
            };
            let fib_idx = (0..ndim).map(|_| Vec::new()).collect();
            return Ok(Self {
                shape: shape.to_vec(),
                mode_order: order,
                fib_ptr,
                fib_idx,
                values: Vec::new(),
            });
        }

        // Build sorted COO entries in mode_order
        let mut entries: Vec<CooEntry<T>> = (0..nnz)
            .map(|i| {
                let coords: Vec<usize> = order.iter().map(|&m| indices[m][i]).collect();
                CooEntry {
                    coords,
                    value: values[i],
                }
            })
            .collect();
        entries.sort_by(|a, b| a.coords.cmp(&b.coords));

        // Build hierarchical structure
        let mut fib_ptr: Vec<Vec<usize>> = Vec::new();
        let mut fib_idx: Vec<Vec<usize>> = Vec::new();
        let mut leaf_values: Vec<T> = Vec::new();

        for _ in 0..ndim {
            fib_idx.push(Vec::new());
        }
        for _ in 0..ndim.saturating_sub(1) {
            fib_ptr.push(Vec::new());
        }

        Self::build_levels(
            &entries,
            &mut fib_ptr,
            &mut fib_idx,
            &mut leaf_values,
            0,
            ndim,
        );

        // Add sentinel to each fib_ptr level
        for l in 0..ndim.saturating_sub(1) {
            fib_ptr[l].push(fib_idx[l + 1].len());
        }

        Ok(Self {
            shape: shape.to_vec(),
            mode_order: order,
            fib_ptr,
            fib_idx,
            values: leaf_values,
        })
    }

    /// Recursively build CSF levels from sorted entries.
    fn build_levels(
        entries: &[CooEntry<T>],
        fib_ptr: &mut Vec<Vec<usize>>,
        fib_idx: &mut Vec<Vec<usize>>,
        values: &mut Vec<T>,
        level: usize,
        ndim: usize,
    ) {
        if entries.is_empty() {
            return;
        }

        if level == ndim - 1 {
            // Leaf level
            for entry in entries {
                fib_idx[level].push(entry.coords[level]);
                values.push(entry.value);
            }
            return;
        }

        // Group by coordinate at this level
        let mut group_start = 0usize;
        while group_start < entries.len() {
            let coord = entries[group_start].coords[level];
            let mut group_end = group_start + 1;
            while group_end < entries.len() && entries[group_end].coords[level] == coord {
                group_end += 1;
            }

            fib_idx[level].push(coord);
            fib_ptr[level].push(fib_idx[level + 1].len());
            Self::build_levels(
                &entries[group_start..group_end],
                fib_ptr,
                fib_idx,
                values,
                level + 1,
                ndim,
            );

            group_start = group_end;
        }
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Number of stored non-zeros.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Look up a value by original tensor coordinates.
    ///
    /// Returns `T::sparse_zero()` if the entry is not stored.
    pub fn get(&self, indices: &[usize]) -> Option<T> {
        let ndim = self.ndim();
        if indices.len() != ndim {
            return None;
        }

        // Map to mode_order
        let ordered: Vec<usize> = self.mode_order.iter().map(|&m| indices[m]).collect();
        self.search_tree(&ordered, 0, 0)
    }

    /// Search the CSF tree for a coordinate.
    fn search_tree(&self, ordered: &[usize], level: usize, fiber_idx: usize) -> Option<T> {
        let ndim = self.ndim();

        // Determine the range of children for this fiber
        let (start, end) = if level == 0 {
            (0, self.fib_idx[0].len())
        } else {
            let s = self.fib_ptr[level - 1].get(fiber_idx).copied().unwrap_or(0);
            let e = self.fib_ptr[level - 1]
                .get(fiber_idx + 1)
                .copied()
                .unwrap_or(self.fib_idx[level].len());
            (s, e)
        };

        let target = ordered[level];

        // Binary search within the range (indices are sorted)
        let range = &self.fib_idx[level][start..end];
        match range.binary_search(&target) {
            Ok(pos) => {
                let abs_pos = start + pos;
                if level == ndim - 1 {
                    // Leaf level
                    self.values.get(abs_pos).copied()
                } else {
                    self.search_tree(ordered, level + 1, abs_pos)
                }
            }
            Err(_) => Some(T::sparse_zero()),
        }
    }

    /// Extract a fiber from the tensor.
    ///
    /// A fiber is a 1D slice obtained by fixing all indices except one mode.
    /// Returns `(index, value)` pairs for the free mode.
    ///
    /// # Arguments
    ///
    /// * `free_mode` - The mode to leave free (tensor mode index, not CSF level).
    /// * `fixed_indices` - Values for all other modes. Length = `ndim - 1`.
    ///   The order corresponds to modes `0, 1, ..., free_mode-1, free_mode+1, ..., ndim-1`.
    pub fn fiber(
        &self,
        free_mode: usize,
        fixed_indices: &[usize],
    ) -> SparseResult<Vec<(usize, T)>> {
        let ndim = self.ndim();
        if free_mode >= ndim {
            return Err(SparseError::ValueError(format!(
                "free_mode {} >= ndim {}",
                free_mode, ndim
            )));
        }
        if fixed_indices.len() != ndim - 1 {
            return Err(SparseError::ValueError(format!(
                "fixed_indices length {} != ndim-1 = {}",
                fixed_indices.len(),
                ndim - 1
            )));
        }

        // Reconstruct full coordinates for each element and check
        let mut result = Vec::new();
        self.collect_fiber(0, 0, free_mode, fixed_indices, &mut Vec::new(), &mut result);

        Ok(result)
    }

    /// Recursively collect fiber entries.
    fn collect_fiber(
        &self,
        level: usize,
        fiber_idx: usize,
        free_mode: usize,
        fixed_indices: &[usize],
        coord_stack: &mut Vec<usize>,
        result: &mut Vec<(usize, T)>,
    ) {
        let ndim = self.ndim();
        let (start, end) = if level == 0 {
            (0, self.fib_idx[0].len())
        } else {
            let s = self.fib_ptr[level - 1].get(fiber_idx).copied().unwrap_or(0);
            let e = self.fib_ptr[level - 1]
                .get(fiber_idx + 1)
                .copied()
                .unwrap_or(self.fib_idx[level].len());
            (s, e)
        };

        let current_mode = self.mode_order[level];

        for i in start..end {
            if i >= self.fib_idx[level].len() {
                break;
            }
            let coord = self.fib_idx[level][i];

            if current_mode == free_mode {
                // This mode is free — iterate over all values
                coord_stack.push(coord);
                if level == ndim - 1 {
                    // Check fixed coords match
                    if self.check_fixed_coords(coord_stack, free_mode, fixed_indices) {
                        if let Some(&val) = self.values.get(i) {
                            result.push((coord, val));
                        }
                    }
                } else {
                    self.collect_fiber(level + 1, i, free_mode, fixed_indices, coord_stack, result);
                }
                coord_stack.pop();
            } else {
                // This mode is fixed — find the expected coordinate
                let fixed_idx = self.fixed_index_for_mode(current_mode, free_mode);
                if let Some(fidx) = fixed_idx {
                    if fidx < fixed_indices.len() && coord == fixed_indices[fidx] {
                        coord_stack.push(coord);
                        if level == ndim - 1 {
                            if self.check_fixed_coords(coord_stack, free_mode, fixed_indices) {
                                if let Some(&val) = self.values.get(i) {
                                    // The free mode coordinate is determined by which
                                    // branch we came from. We need to find it.
                                    let free_coord = self.find_free_coord(coord_stack, free_mode);
                                    if let Some(fc) = free_coord {
                                        result.push((fc, val));
                                    }
                                }
                            }
                        } else {
                            self.collect_fiber(
                                level + 1,
                                i,
                                free_mode,
                                fixed_indices,
                                coord_stack,
                                result,
                            );
                        }
                        coord_stack.pop();
                    }
                    // else: coordinate doesn't match, skip
                }
            }
        }
    }

    /// Get the index into fixed_indices for a given mode.
    fn fixed_index_for_mode(&self, mode: usize, free_mode: usize) -> Option<usize> {
        if mode == free_mode {
            return None;
        }
        let mut idx = 0usize;
        for m in 0..self.ndim() {
            if m == free_mode {
                continue;
            }
            if m == mode {
                return Some(idx);
            }
            idx += 1;
        }
        None
    }

    /// Check that the coord_stack matches the fixed indices.
    fn check_fixed_coords(
        &self,
        coord_stack: &[usize],
        free_mode: usize,
        fixed_indices: &[usize],
    ) -> bool {
        let mut fix_idx = 0usize;
        for (level, &coord) in coord_stack.iter().enumerate() {
            if level >= self.mode_order.len() {
                break;
            }
            let mode = self.mode_order[level];
            if mode == free_mode {
                continue;
            }
            if fix_idx >= fixed_indices.len() || coord != fixed_indices[fix_idx] {
                return false;
            }
            fix_idx += 1;
        }
        true
    }

    /// Find the coordinate of the free mode in the coord_stack.
    fn find_free_coord(&self, coord_stack: &[usize], free_mode: usize) -> Option<usize> {
        for (level, &coord) in coord_stack.iter().enumerate() {
            if level < self.mode_order.len() && self.mode_order[level] == free_mode {
                return Some(coord);
            }
        }
        None
    }

    /// Mode-n matricization: unfold the tensor along mode `n`.
    ///
    /// Returns a matrix (in COO form as `(rows, cols, vals)`) where:
    /// - rows correspond to the free mode's indices
    /// - columns correspond to the lexicographic combination of all other mode indices
    ///
    /// The column index for a multi-index `(i_0, ..., i_{n-1}, i_{n+1}, ..., i_{N-1})`
    /// is computed as a mixed-radix number.
    pub fn matricize(&self, mode: usize) -> SparseResult<(Vec<usize>, Vec<usize>, Vec<T>)> {
        let ndim = self.ndim();
        if mode >= ndim {
            return Err(SparseError::ValueError(format!(
                "mode {} >= ndim {}",
                mode, ndim
            )));
        }

        // Compute column strides for all modes except `mode`
        let other_modes: Vec<usize> = (0..ndim).filter(|&m| m != mode).collect();
        let mut col_strides: Vec<usize> = Vec::with_capacity(other_modes.len());
        let mut stride = 1usize;
        for &m in other_modes.iter().rev() {
            col_strides.push(stride);
            stride = stride.saturating_mul(self.shape[m]);
        }
        col_strides.reverse();

        // Extract all entries by traversing the tree
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        let mut coord_stack: Vec<usize> = vec![0; ndim];

        self.traverse_for_matricize(
            0,
            0,
            &mut coord_stack,
            mode,
            &other_modes,
            &col_strides,
            &mut rows,
            &mut cols,
            &mut vals,
        );

        Ok((rows, cols, vals))
    }

    /// Traverse CSF tree collecting entries for matricization.
    fn traverse_for_matricize(
        &self,
        level: usize,
        fiber_idx: usize,
        coord_stack: &mut Vec<usize>,
        mode: usize,
        other_modes: &[usize],
        col_strides: &[usize],
        rows: &mut Vec<usize>,
        cols: &mut Vec<usize>,
        vals: &mut Vec<T>,
    ) {
        let ndim = self.ndim();
        let (start, end) = if level == 0 {
            (0, self.fib_idx[0].len())
        } else {
            let s = self.fib_ptr[level - 1].get(fiber_idx).copied().unwrap_or(0);
            let e = self.fib_ptr[level - 1]
                .get(fiber_idx + 1)
                .copied()
                .unwrap_or(self.fib_idx[level].len());
            (s, e)
        };

        for i in start..end {
            if i >= self.fib_idx[level].len() {
                break;
            }
            coord_stack[level] = self.fib_idx[level][i];

            if level == ndim - 1 {
                // Emit entry
                if let Some(&val) = self.values.get(i) {
                    // Build the original-mode coordinates
                    let mut orig_coords = vec![0usize; ndim];
                    for (l, &c) in coord_stack.iter().enumerate().take(ndim) {
                        orig_coords[self.mode_order[l]] = c;
                    }

                    let row = orig_coords[mode];
                    let mut col = 0usize;
                    for (idx, &m) in other_modes.iter().enumerate() {
                        col += orig_coords[m] * col_strides[idx];
                    }

                    rows.push(row);
                    cols.push(col);
                    vals.push(val);
                }
            } else {
                self.traverse_for_matricize(
                    level + 1,
                    i,
                    coord_stack,
                    mode,
                    other_modes,
                    col_strides,
                    rows,
                    cols,
                    vals,
                );
            }
        }
    }

    /// Memory usage estimate in bytes.
    pub fn memory_usage(&self) -> usize {
        let mut total = 0usize;
        for fp in &self.fib_ptr {
            total += fp.len() * std::mem::size_of::<usize>();
        }
        for fi in &self.fib_idx {
            total += fi.len() * std::mem::size_of::<usize>();
        }
        total += self.values.len() * std::mem::size_of::<T>();
        total += self.shape.len() * std::mem::size_of::<usize>();
        total += self.mode_order.len() * std::mem::size_of::<usize>();
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_csf_3d_construction_and_access() {
        // 2x3x4 tensor with 5 non-zeros
        let indices = vec![
            vec![0, 0, 0, 1, 1], // mode 0
            vec![0, 1, 2, 0, 2], // mode 1
            vec![0, 1, 3, 2, 0], // mode 2
        ];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = vec![2, 3, 4];

        let csf = CsfTensor::from_coo(&indices, &values, &shape, None).expect("csf");
        assert_eq!(csf.ndim(), 3);
        assert_eq!(csf.nnz(), 5);

        assert_relative_eq!(csf.get(&[0, 0, 0]).unwrap_or(0.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(csf.get(&[0, 1, 1]).unwrap_or(0.0), 2.0, epsilon = 1e-12);
        assert_relative_eq!(csf.get(&[0, 2, 3]).unwrap_or(0.0), 3.0, epsilon = 1e-12);
        assert_relative_eq!(csf.get(&[1, 0, 2]).unwrap_or(0.0), 4.0, epsilon = 1e-12);
        assert_relative_eq!(csf.get(&[1, 2, 0]).unwrap_or(0.0), 5.0, epsilon = 1e-12);
        // Zero entry
        assert_relative_eq!(csf.get(&[0, 0, 1]).unwrap_or(0.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_csf_fiber_extraction() {
        // 3x3 matrix as a 2D tensor
        //  [1  0  2]
        //  [0  3  0]
        //  [4  0  5]
        let indices = vec![
            vec![0, 0, 1, 2, 2], // mode 0 (rows)
            vec![0, 2, 1, 0, 2], // mode 1 (cols)
        ];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = vec![3, 3];

        let csf = CsfTensor::from_coo(&indices, &values, &shape, None).expect("csf");

        // Extract row 0 (free_mode=1, fixed: mode0=0)
        let fiber = csf.fiber(1, &[0]).expect("fiber");
        // Should have entries (0, 1.0) and (2, 2.0)
        assert_eq!(fiber.len(), 2);
        let fiber_map: std::collections::HashMap<usize, f64> = fiber.into_iter().collect();
        assert_relative_eq!(*fiber_map.get(&0).unwrap_or(&0.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(*fiber_map.get(&2).unwrap_or(&0.0), 2.0, epsilon = 1e-12);

        // Extract column 2 (free_mode=0, fixed: mode1=2)
        let fiber = csf.fiber(0, &[2]).expect("fiber");
        let fiber_map: std::collections::HashMap<usize, f64> = fiber.into_iter().collect();
        assert_relative_eq!(*fiber_map.get(&0).unwrap_or(&0.0), 2.0, epsilon = 1e-12);
        assert_relative_eq!(*fiber_map.get(&2).unwrap_or(&0.0), 5.0, epsilon = 1e-12);
    }

    #[test]
    fn test_csf_matricize() {
        // 2x3x2 tensor
        let indices = vec![
            vec![0, 0, 1, 1], // mode 0
            vec![0, 1, 0, 2], // mode 1
            vec![0, 1, 0, 1], // mode 2
        ];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 3, 2];

        let csf = CsfTensor::from_coo(&indices, &values, &shape, None).expect("csf");

        // Matricize along mode 0: result is 2 x (3*2) = 2 x 6
        let (rows, cols, vals) = csf.matricize(0).expect("matricize");
        assert_eq!(rows.len(), 4);

        // Check that all entries are accounted for
        for ((&r, &c), &v) in rows.iter().zip(cols.iter()).zip(vals.iter()) {
            assert!(r < 2);
            assert!(c < 6);
            assert!(v != 0.0);
        }
    }

    #[test]
    fn test_csf_empty() {
        let indices: Vec<Vec<usize>> = vec![Vec::new(), Vec::new()];
        let values: Vec<f64> = Vec::new();
        let shape = vec![3, 4];
        let csf = CsfTensor::from_coo(&indices, &values, &shape, None).expect("csf");
        assert_eq!(csf.nnz(), 0);
        assert_eq!(csf.ndim(), 2);
    }

    #[test]
    fn test_csf_with_mode_order() {
        let indices = vec![
            vec![0, 0, 1], // mode 0
            vec![0, 1, 0], // mode 1
        ];
        let values = vec![1.0, 2.0, 3.0];
        let shape = vec![2, 2];

        let csf = CsfTensor::from_coo(&indices, &values, &shape, Some(&[1, 0])).expect("csf");
        assert_eq!(csf.nnz(), 3);

        // Access should still work with original coordinates
        assert_relative_eq!(csf.get(&[0, 0]).unwrap_or(0.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(csf.get(&[0, 1]).unwrap_or(0.0), 2.0, epsilon = 1e-12);
        assert_relative_eq!(csf.get(&[1, 0]).unwrap_or(0.0), 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_csf_memory_usage() {
        let indices = vec![vec![0, 1], vec![0, 1]];
        let values = vec![1.0, 2.0];
        let shape = vec![2, 2];
        let csf = CsfTensor::from_coo(&indices, &values, &shape, None).expect("csf");
        assert!(csf.memory_usage() > 0);
    }

    #[test]
    fn test_csf_invalid_mode_order() {
        let indices = vec![vec![0], vec![0]];
        let values = vec![1.0];
        let shape = vec![2, 2];
        assert!(CsfTensor::from_coo(&indices, &values, &shape, Some(&[0, 0])).is_err());
        assert!(CsfTensor::from_coo(&indices, &values, &shape, Some(&[0])).is_err());
    }
}
