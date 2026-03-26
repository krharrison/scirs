//! Sparse tensor operations for gradient computation
//!
//! This module provides multi-dimensional sparse tensor support for gradient
//! computation, extending the flat [`SparseGrad`] to handle shaped tensors.

use super::{EmbeddingGrad, GradientRepr, SparseGrad};
use crate::error::AutogradError;
use crate::Result;
use std::collections::HashMap;
use std::fmt;

/// A multi-dimensional sparse tensor for gradient representation.
///
/// Unlike [`SparseGrad`] which uses flat indices, this supports shaped tensors
/// with per-dimension indexing, useful for convolutional and attention gradients.
#[derive(Debug, Clone)]
pub struct SparseTensor {
    /// Shape of the tensor (e.g., [batch, height, width, channels])
    shape: Vec<usize>,
    /// Flat indices of non-zero entries (row-major order)
    indices: Vec<usize>,
    /// Values at those indices
    values: Vec<f64>,
}

impl SparseTensor {
    /// Create a new sparse tensor.
    ///
    /// # Arguments
    /// * `shape` - The dimensions of the tensor
    /// * `indices` - Flat (row-major) indices of non-zero entries
    /// * `values` - Values at those indices
    ///
    /// # Errors
    /// Returns error if indices and values have different lengths or indices are out of bounds.
    pub fn new(shape: Vec<usize>, indices: Vec<usize>, values: Vec<f64>) -> Result<Self> {
        if indices.len() != values.len() {
            return Err(AutogradError::invalid_argument(format!(
                "SparseTensor: indices length ({}) != values length ({})",
                indices.len(),
                values.len()
            )));
        }
        let total: usize = shape.iter().product();
        for &idx in &indices {
            if idx >= total {
                return Err(AutogradError::invalid_argument(format!(
                    "SparseTensor: index {} out of bounds for total size {}",
                    idx, total
                )));
            }
        }
        Ok(Self {
            shape,
            indices,
            values,
        })
    }

    /// Create a zero sparse tensor with the given shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        Self {
            shape,
            indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Create from a dense tensor (flattened) with a threshold.
    pub fn from_dense(shape: Vec<usize>, data: &[f64], threshold: f64) -> Result<Self> {
        let total: usize = shape.iter().product();
        if data.len() != total {
            return Err(AutogradError::shape_error(format!(
                "SparseTensor from_dense: data length {} != product of shape {:?} ({})",
                data.len(),
                shape,
                total
            )));
        }
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for (i, &v) in data.iter().enumerate() {
            if v.abs() >= threshold {
                indices.push(i);
                values.push(v);
            }
        }
        Ok(Self {
            shape,
            indices,
            values,
        })
    }

    /// Get the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the total number of elements.
    pub fn total_size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Get the flat indices.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Get the values.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Convert a flat index to multi-dimensional coordinates.
    pub fn flat_to_coords(&self, flat_idx: usize) -> Vec<usize> {
        let mut coords = vec![0usize; self.shape.len()];
        let mut remaining = flat_idx;
        for i in (0..self.shape.len()).rev() {
            coords[i] = remaining % self.shape[i];
            remaining /= self.shape[i];
        }
        coords
    }

    /// Convert multi-dimensional coordinates to a flat index.
    ///
    /// # Errors
    /// Returns error if coordinates are out of bounds.
    pub fn coords_to_flat(&self, coords: &[usize]) -> Result<usize> {
        if coords.len() != self.shape.len() {
            return Err(AutogradError::invalid_argument(format!(
                "SparseTensor: coords dimension {} != tensor dimension {}",
                coords.len(),
                self.shape.len()
            )));
        }
        let mut flat = 0usize;
        let mut stride = 1usize;
        for i in (0..self.shape.len()).rev() {
            if coords[i] >= self.shape[i] {
                return Err(AutogradError::invalid_argument(format!(
                    "SparseTensor: coord[{}]={} out of bounds for dim size {}",
                    i, coords[i], self.shape[i]
                )));
            }
            flat += coords[i] * stride;
            stride *= self.shape[i];
        }
        Ok(flat)
    }

    /// Convert to a flat dense vector.
    pub fn to_dense(&self) -> Vec<f64> {
        let total = self.total_size();
        let mut dense = vec![0.0; total];
        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            dense[idx] += val;
        }
        dense
    }

    /// Convert to a flat SparseGrad.
    pub fn to_sparse_grad(&self) -> SparseGrad {
        // Use the internal constructor path - we know indices are valid
        let mut sg = SparseGrad::from_dense(&self.to_dense(), f64::EPSILON);
        // Re-sparsify to ensure clean indices
        sg.sparsify(f64::EPSILON);
        sg
    }

    /// Accumulate another sparse tensor into this one.
    ///
    /// # Errors
    /// Returns error if shapes don't match.
    pub fn accumulate(&mut self, other: &SparseTensor) -> Result<()> {
        if self.shape != other.shape {
            return Err(AutogradError::shape_error(format!(
                "SparseTensor accumulate: shape mismatch ({:?} vs {:?})",
                self.shape, other.shape
            )));
        }
        self.indices.extend_from_slice(&other.indices);
        self.values.extend_from_slice(&other.values);
        self.consolidate();
        Ok(())
    }

    /// Sort and merge duplicate indices.
    fn consolidate(&mut self) {
        if self.indices.is_empty() {
            return;
        }
        let mut map: HashMap<usize, f64> = HashMap::new();
        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            *map.entry(idx).or_insert(0.0) += val;
        }
        let mut pairs: Vec<(usize, f64)> = map.into_iter().collect();
        pairs.sort_by_key(|&(idx, _)| idx);
        self.indices = pairs.iter().map(|&(idx, _)| idx).collect();
        self.values = pairs.iter().map(|&(_, val)| val).collect();
    }

    /// Reshape to a different shape (same total size).
    ///
    /// # Errors
    /// Returns error if the new shape has a different total size.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self> {
        let old_total: usize = self.shape.iter().product();
        let new_total: usize = new_shape.iter().product();
        if old_total != new_total {
            return Err(AutogradError::shape_error(format!(
                "SparseTensor reshape: total size mismatch ({} vs {})",
                old_total, new_total
            )));
        }
        Ok(Self {
            shape: new_shape,
            indices: self.indices.clone(),
            values: self.values.clone(),
        })
    }

    /// Slice along the first dimension, returning a new sparse tensor.
    ///
    /// # Errors
    /// Returns error if indices are out of range.
    pub fn slice_first_dim(&self, start: usize, end: usize) -> Result<Self> {
        if self.shape.is_empty() {
            return Err(AutogradError::invalid_argument(
                "SparseTensor: cannot slice 0-dimensional tensor".to_string(),
            ));
        }
        if start >= self.shape[0] || end > self.shape[0] || start >= end {
            return Err(AutogradError::invalid_argument(format!(
                "SparseTensor: invalid slice [{}..{}) for dim 0 of size {}",
                start, end, self.shape[0]
            )));
        }
        let inner_size: usize = self.shape[1..].iter().product();
        let flat_start = start * inner_size;
        let flat_end = end * inner_size;
        let mut new_indices = Vec::new();
        let mut new_values = Vec::new();
        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            if idx >= flat_start && idx < flat_end {
                new_indices.push(idx - flat_start);
                new_values.push(val);
            }
        }
        let mut new_shape = self.shape.clone();
        new_shape[0] = end - start;
        Ok(Self {
            shape: new_shape,
            indices: new_indices,
            values: new_values,
        })
    }
}

impl fmt::Display for SparseTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SparseTensor(shape={:?}, nnz={}, density={:.4})",
            self.shape,
            self.nnz(),
            if self.total_size() > 0 {
                self.nnz() as f64 / self.total_size() as f64
            } else {
                0.0
            }
        )
    }
}

/// Sparse gradient accumulator for batched operations.
///
/// Efficiently accumulates sparse gradients from multiple micro-batches,
/// useful for gradient accumulation in large-batch training.
#[derive(Debug)]
pub struct SparseGradAccumulator {
    /// Accumulated gradients per parameter name
    grads: HashMap<String, GradientRepr>,
    /// Number of accumulated steps
    steps: usize,
}

impl SparseGradAccumulator {
    /// Create a new sparse gradient accumulator.
    pub fn new() -> Self {
        Self {
            grads: HashMap::new(),
            steps: 0,
        }
    }

    /// Accumulate a sparse gradient for a named parameter.
    ///
    /// # Errors
    /// Returns error on shape mismatch with previously accumulated gradient.
    pub fn accumulate_sparse(&mut self, name: &str, grad: SparseGrad) -> Result<()> {
        let repr = GradientRepr::Sparse(grad);
        match self.grads.get_mut(name) {
            Some(existing) => {
                existing.accumulate(&repr)?;
            }
            None => {
                self.grads.insert(name.to_string(), repr);
            }
        }
        Ok(())
    }

    /// Accumulate a dense gradient for a named parameter.
    ///
    /// # Errors
    /// Returns error on shape mismatch.
    pub fn accumulate_dense(&mut self, name: &str, grad: Vec<f64>) -> Result<()> {
        let repr = GradientRepr::Dense(grad);
        match self.grads.get_mut(name) {
            Some(existing) => {
                existing.accumulate(&repr)?;
            }
            None => {
                self.grads.insert(name.to_string(), repr);
            }
        }
        Ok(())
    }

    /// Mark the end of a micro-batch step.
    pub fn step(&mut self) {
        self.steps += 1;
    }

    /// Get the number of accumulated steps.
    pub fn num_steps(&self) -> usize {
        self.steps
    }

    /// Get the averaged gradient for a parameter (divides by number of steps).
    ///
    /// Returns None if no gradient has been accumulated for this parameter.
    pub fn averaged_grad(&self, name: &str) -> Option<Vec<f64>> {
        if self.steps == 0 {
            return None;
        }
        self.grads.get(name).map(|g| {
            let dense = g.to_dense();
            let scale = 1.0 / self.steps as f64;
            dense.into_iter().map(|v| v * scale).collect()
        })
    }

    /// Get the raw (un-averaged) gradient for a parameter.
    pub fn raw_grad(&self, name: &str) -> Option<&GradientRepr> {
        self.grads.get(name)
    }

    /// Get all parameter names that have gradients.
    pub fn param_names(&self) -> Vec<&str> {
        self.grads.keys().map(|s| s.as_str()).collect()
    }

    /// Clear all accumulated gradients and reset the step counter.
    pub fn clear(&mut self) {
        self.grads.clear();
        self.steps = 0;
    }
}

impl Default for SparseGradAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Simulates an embedding lookup forward + backward pass with sparse gradients.
///
/// # Arguments
/// * `embedding_table` - Flattened embedding table [num_embeddings * embedding_dim]
/// * `num_embeddings` - Vocabulary size
/// * `embedding_dim` - Embedding dimension
/// * `lookup_indices` - Indices to look up
/// * `upstream_grad` - Gradient flowing back for each looked-up embedding [len(lookup_indices) * embedding_dim]
///
/// # Returns
/// An `EmbeddingGrad` with sparse gradients only for the accessed rows.
///
/// # Errors
/// Returns error on dimension mismatches or out-of-bounds indices.
pub fn embedding_backward(
    _embedding_table: &[f64],
    num_embeddings: usize,
    embedding_dim: usize,
    lookup_indices: &[usize],
    upstream_grad: &[f64],
) -> Result<EmbeddingGrad> {
    if upstream_grad.len() != lookup_indices.len() * embedding_dim {
        return Err(AutogradError::shape_error(format!(
            "embedding_backward: upstream_grad length {} != {} * {} = {}",
            upstream_grad.len(),
            lookup_indices.len(),
            embedding_dim,
            lookup_indices.len() * embedding_dim
        )));
    }
    let mut eg = EmbeddingGrad::new(num_embeddings, embedding_dim);
    for (i, &idx) in lookup_indices.iter().enumerate() {
        let grad_slice = &upstream_grad[i * embedding_dim..(i + 1) * embedding_dim];
        eg.accumulate_row(idx, grad_slice)?;
    }
    Ok(eg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_tensor_new() {
        let st = SparseTensor::new(vec![3, 4], vec![0, 5, 11], vec![1.0, 2.0, 3.0]).expect("valid");
        assert_eq!(st.shape(), &[3, 4]);
        assert_eq!(st.total_size(), 12);
        assert_eq!(st.nnz(), 3);
    }

    #[test]
    fn test_sparse_tensor_coords() {
        let st = SparseTensor::zeros(vec![2, 3, 4]);
        // flat index 0 -> [0, 0, 0]
        assert_eq!(st.flat_to_coords(0), vec![0, 0, 0]);
        // flat index 5 -> [0, 1, 1]
        assert_eq!(st.flat_to_coords(5), vec![0, 1, 1]);
        // flat index 23 -> [1, 2, 3]
        assert_eq!(st.flat_to_coords(23), vec![1, 2, 3]);
        // round-trip
        assert_eq!(st.coords_to_flat(&[1, 2, 3]).expect("ok"), 23);
    }

    #[test]
    fn test_sparse_tensor_from_dense() {
        let data = vec![0.0, 1.0, 0.0, 0.0, 2.0, 0.0];
        let st = SparseTensor::from_dense(vec![2, 3], &data, 0.5).expect("valid");
        assert_eq!(st.nnz(), 2);
        let dense = st.to_dense();
        assert!((dense[1] - 1.0).abs() < 1e-12);
        assert!((dense[4] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_sparse_tensor_accumulate() {
        let mut st1 = SparseTensor::new(vec![2, 3], vec![0, 2], vec![1.0, 2.0]).expect("valid");
        let st2 = SparseTensor::new(vec![2, 3], vec![2, 5], vec![3.0, 4.0]).expect("valid");
        st1.accumulate(&st2).expect("ok");
        let dense = st1.to_dense();
        assert!((dense[0] - 1.0).abs() < 1e-12);
        assert!((dense[2] - 5.0).abs() < 1e-12); // 2 + 3
        assert!((dense[5] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_sparse_tensor_reshape() {
        let st = SparseTensor::new(vec![2, 3], vec![0, 5], vec![1.0, 2.0]).expect("valid");
        let reshaped = st.reshape(vec![3, 2]).expect("ok");
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped.nnz(), 2);
        // Flat indices stay the same
        assert_eq!(reshaped.indices(), &[0, 5]);
    }

    #[test]
    fn test_sparse_tensor_slice() {
        let st = SparseTensor::new(vec![4, 3], vec![0, 3, 9], vec![1.0, 2.0, 3.0]).expect("valid");
        let sliced = st.slice_first_dim(1, 3).expect("ok");
        assert_eq!(sliced.shape(), &[2, 3]);
        // Only index 3 (row 1) maps to local index 0 within the slice
        assert_eq!(sliced.nnz(), 1);
        assert_eq!(sliced.indices(), &[0]);
        assert!((sliced.values()[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_sparse_grad_accumulator() {
        let mut acc = SparseGradAccumulator::new();
        let g1 = SparseGrad::new(vec![0, 2], vec![1.0, 2.0], 5).expect("valid");
        let g2 = SparseGrad::new(vec![1], vec![3.0], 5).expect("valid");
        acc.accumulate_sparse("w", g1).expect("ok");
        acc.step();
        acc.accumulate_sparse("w", g2).expect("ok");
        acc.step();
        let avg = acc.averaged_grad("w").expect("has grad");
        // (1+0)/2=0.5, (0+3)/2=1.5, (2+0)/2=1.0
        assert!((avg[0] - 0.5).abs() < 1e-12);
        assert!((avg[1] - 1.5).abs() < 1e-12);
        assert!((avg[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_sparse_grad_accumulator_mixed() {
        let mut acc = SparseGradAccumulator::new();
        acc.accumulate_dense("b", vec![1.0, 2.0]).expect("ok");
        let sg = SparseGrad::new(vec![0], vec![3.0], 2).expect("valid");
        acc.accumulate_sparse("b", sg).expect("ok");
        acc.step();
        let avg = acc.averaged_grad("b").expect("has grad");
        assert!((avg[0] - 4.0).abs() < 1e-12);
        assert!((avg[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_embedding_backward() {
        // 5 embeddings of dim 3
        let table = vec![0.0; 15];
        let indices = vec![1, 3, 1]; // look up rows 1, 3, 1
                                     // upstream grad: 3 vectors of dim 3
        let upstream = vec![
            1.0, 0.0, 0.0, // grad for lookup[0] = row 1
            0.0, 1.0, 0.0, // grad for lookup[1] = row 3
            0.0, 0.0, 1.0, // grad for lookup[2] = row 1 (accumulates)
        ];
        let eg = embedding_backward(&table, 5, 3, &indices, &upstream).expect("ok");
        assert_eq!(eg.num_updated_rows(), 2); // only rows 1 and 3
        let row1 = eg.row_grad(1).expect("exists");
        assert!((row1[0] - 1.0).abs() < 1e-12);
        assert!((row1[2] - 1.0).abs() < 1e-12); // accumulated
        let row3 = eg.row_grad(3).expect("exists");
        assert!((row3[1] - 1.0).abs() < 1e-12);
        assert!(eg.row_grad(0).is_none()); // row 0 not accessed
    }

    #[test]
    fn test_sparse_tensor_shape_mismatch() {
        let st1 = SparseTensor::new(vec![2, 3], vec![0], vec![1.0]).expect("valid");
        let st2 = SparseTensor::new(vec![3, 2], vec![0], vec![1.0]).expect("valid");
        assert!(st1.reshape(vec![3, 3]).is_err()); // different total
        let mut st1_mut = st1;
        assert!(st1_mut.accumulate(&st2).is_err()); // shape mismatch
    }

    #[test]
    fn test_sparse_tensor_display() {
        let st = SparseTensor::new(vec![10, 10], vec![0, 50], vec![1.0, 2.0]).expect("valid");
        let s = format!("{}", st);
        assert!(s.contains("shape=[10, 10]"));
        assert!(s.contains("nnz=2"));
    }
}
