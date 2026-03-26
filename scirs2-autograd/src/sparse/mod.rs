//! Sparse gradient support for automatic differentiation
//!
//! This module provides sparse gradient representations that are efficient for
//! operations where most gradient entries are zero (e.g., embedding lookups,
//! one-hot encodings, sparse attention). Instead of storing dense gradient
//! vectors, only the non-zero indices and their corresponding values are kept.
//!
//! # Key Types
//!
//! - [`SparseGrad`] - Core sparse gradient representation with indices + values
//! - [`SparseVariable`] - A variable that tracks sparse gradients
//! - [`EmbeddingGrad`] - Specialized sparse gradient for embedding layers
//!
//! # Example
//!
//! ```rust
//! use scirs2_autograd::sparse::{SparseGrad, SparseVariable};
//!
//! // Create a sparse gradient with only a few non-zero entries
//! let grad = SparseGrad::new(vec![0, 5, 12], vec![1.0, -0.5, 2.3], 100)
//!     .expect("valid sparse grad");
//!
//! // Convert to dense for final application
//! let dense = grad.to_dense();
//! assert_eq!(dense.len(), 100);
//! assert!((dense[5] - (-0.5)).abs() < 1e-10);
//! ```

pub mod sparse_tensor;

use crate::error::AutogradError;
use crate::Result;
use std::collections::HashMap;
use std::fmt;

/// A sparse gradient representation storing only non-zero entries.
///
/// This is useful for operations like embedding lookups where only a small
/// subset of parameters receive gradient updates in each step.
#[derive(Debug, Clone)]
pub struct SparseGrad {
    /// Indices of non-zero gradient entries (sorted, no duplicates after accumulation)
    indices: Vec<usize>,
    /// Values corresponding to each index
    values: Vec<f64>,
    /// Total shape (number of elements in the dense representation)
    shape: usize,
}

impl SparseGrad {
    /// Create a new sparse gradient.
    ///
    /// # Arguments
    /// * `indices` - Indices of non-zero entries
    /// * `values` - Values at those indices
    /// * `shape` - Total number of elements in the dense representation
    ///
    /// # Errors
    /// Returns error if indices and values have different lengths, or if any index is out of bounds.
    pub fn new(indices: Vec<usize>, values: Vec<f64>, shape: usize) -> Result<Self> {
        if indices.len() != values.len() {
            return Err(AutogradError::invalid_argument(format!(
                "SparseGrad: indices length ({}) != values length ({})",
                indices.len(),
                values.len()
            )));
        }
        for &idx in &indices {
            if idx >= shape {
                return Err(AutogradError::invalid_argument(format!(
                    "SparseGrad: index {} out of bounds for shape {}",
                    idx, shape
                )));
            }
        }
        let mut sg = Self {
            indices,
            values,
            shape,
        };
        sg.consolidate();
        Ok(sg)
    }

    /// Create an empty sparse gradient with the given shape.
    pub fn zeros(shape: usize) -> Self {
        Self {
            indices: Vec::new(),
            values: Vec::new(),
            shape,
        }
    }

    /// Create a sparse gradient from a dense vector, dropping entries below threshold.
    ///
    /// # Arguments
    /// * `dense` - Dense gradient vector
    /// * `threshold` - Minimum absolute value to keep (entries below this are dropped)
    pub fn from_dense(dense: &[f64], threshold: f64) -> Self {
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for (i, &v) in dense.iter().enumerate() {
            if v.abs() >= threshold {
                indices.push(i);
                values.push(v);
            }
        }
        Self {
            indices,
            values,
            shape: dense.len(),
        }
    }

    /// Convert to a dense gradient vector.
    pub fn to_dense(&self) -> Vec<f64> {
        let mut dense = vec![0.0; self.shape];
        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            dense[idx] += val;
        }
        dense
    }

    /// Get the indices of non-zero entries.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Get the values of non-zero entries.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Get the total shape (dense size).
    pub fn shape(&self) -> usize {
        self.shape
    }

    /// Number of non-zero entries (nnz).
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Sparsity ratio (fraction of zero entries).
    pub fn sparsity(&self) -> f64 {
        if self.shape == 0 {
            return 1.0;
        }
        1.0 - (self.nnz() as f64 / self.shape as f64)
    }

    /// Drop gradient entries with absolute value below the given threshold.
    pub fn sparsify(&mut self, threshold: f64) {
        let mut new_indices = Vec::new();
        let mut new_values = Vec::new();
        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            if val.abs() >= threshold {
                new_indices.push(idx);
                new_values.push(val);
            }
        }
        self.indices = new_indices;
        self.values = new_values;
    }

    /// Sort indices and merge duplicates by summing their values.
    fn consolidate(&mut self) {
        if self.indices.is_empty() {
            return;
        }
        // Group by index, sum values
        let mut map: HashMap<usize, f64> = HashMap::new();
        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            *map.entry(idx).or_insert(0.0) += val;
        }
        let mut pairs: Vec<(usize, f64)> = map.into_iter().collect();
        pairs.sort_by_key(|&(idx, _)| idx);
        self.indices = pairs.iter().map(|&(idx, _)| idx).collect();
        self.values = pairs.iter().map(|&(_, val)| val).collect();
    }

    /// Accumulate (add) another sparse gradient into this one.
    ///
    /// Both gradients must have the same shape.
    ///
    /// # Errors
    /// Returns error if shapes don't match.
    pub fn accumulate(&mut self, other: &SparseGrad) -> Result<()> {
        if self.shape != other.shape {
            return Err(AutogradError::shape_error(format!(
                "SparseGrad accumulate: shape mismatch ({} vs {})",
                self.shape, other.shape
            )));
        }
        self.indices.extend_from_slice(&other.indices);
        self.values.extend_from_slice(&other.values);
        self.consolidate();
        Ok(())
    }

    /// Merge two sparse gradients into a new one (sparse + sparse).
    ///
    /// # Errors
    /// Returns error if shapes don't match.
    pub fn merge(a: &SparseGrad, b: &SparseGrad) -> Result<SparseGrad> {
        if a.shape != b.shape {
            return Err(AutogradError::shape_error(format!(
                "SparseGrad merge: shape mismatch ({} vs {})",
                a.shape, b.shape
            )));
        }
        let mut merged = a.clone();
        merged.accumulate(b)?;
        Ok(merged)
    }

    /// Add a sparse gradient to a dense gradient vector in-place.
    ///
    /// # Errors
    /// Returns error if the dense vector length doesn't match the sparse shape.
    pub fn add_to_dense(&self, dense: &mut [f64]) -> Result<()> {
        if dense.len() != self.shape {
            return Err(AutogradError::shape_error(format!(
                "SparseGrad add_to_dense: dense length {} != sparse shape {}",
                dense.len(),
                self.shape
            )));
        }
        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            dense[idx] += val;
        }
        Ok(())
    }

    /// Scale all gradient values by a scalar.
    pub fn scale(&mut self, factor: f64) {
        for v in &mut self.values {
            *v *= factor;
        }
    }

    /// Return a scaled copy of this gradient.
    pub fn scaled(&self, factor: f64) -> Self {
        Self {
            indices: self.indices.clone(),
            values: self.values.iter().map(|&v| v * factor).collect(),
            shape: self.shape,
        }
    }

    /// Clip gradient values to a maximum norm (per-element clipping).
    pub fn clip(&mut self, max_abs: f64) {
        for v in &mut self.values {
            if *v > max_abs {
                *v = max_abs;
            } else if *v < -max_abs {
                *v = -max_abs;
            }
        }
    }

    /// Compute the L2 norm of the sparse gradient.
    pub fn l2_norm(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Compute the L1 norm of the sparse gradient.
    pub fn l1_norm(&self) -> f64 {
        self.values.iter().map(|v| v.abs()).sum()
    }
}

impl fmt::Display for SparseGrad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SparseGrad(nnz={}, shape={}, sparsity={:.1}%)",
            self.nnz(),
            self.shape,
            self.sparsity() * 100.0
        )
    }
}

/// Gradient type that can be either sparse or dense.
#[derive(Debug, Clone)]
pub enum GradientRepr {
    /// Dense gradient stored as a flat vector
    Dense(Vec<f64>),
    /// Sparse gradient with only non-zero entries
    Sparse(SparseGrad),
}

impl GradientRepr {
    /// Convert to dense representation.
    pub fn to_dense(&self) -> Vec<f64> {
        match self {
            GradientRepr::Dense(d) => d.clone(),
            GradientRepr::Sparse(s) => s.to_dense(),
        }
    }

    /// Convert to sparse representation with given threshold.
    pub fn to_sparse(&self, threshold: f64) -> SparseGrad {
        match self {
            GradientRepr::Dense(d) => SparseGrad::from_dense(d, threshold),
            GradientRepr::Sparse(s) => {
                let mut result = s.clone();
                result.sparsify(threshold);
                result
            }
        }
    }

    /// Check if the representation is sparse.
    pub fn is_sparse(&self) -> bool {
        matches!(self, GradientRepr::Sparse(_))
    }

    /// Get the total number of elements (dense shape).
    pub fn shape(&self) -> usize {
        match self {
            GradientRepr::Dense(d) => d.len(),
            GradientRepr::Sparse(s) => s.shape(),
        }
    }

    /// Accumulate another gradient into this one.
    ///
    /// # Errors
    /// Returns error on shape mismatch.
    pub fn accumulate(&mut self, other: &GradientRepr) -> Result<()> {
        match (self, other) {
            (GradientRepr::Sparse(ref mut a), GradientRepr::Sparse(b)) => {
                a.accumulate(b)?;
            }
            (GradientRepr::Dense(ref mut d), GradientRepr::Sparse(s)) => {
                s.add_to_dense(d)?;
            }
            (GradientRepr::Dense(ref mut d), GradientRepr::Dense(other_d)) => {
                if d.len() != other_d.len() {
                    return Err(AutogradError::shape_error(format!(
                        "GradientRepr accumulate: dense length mismatch ({} vs {})",
                        d.len(),
                        other_d.len()
                    )));
                }
                for (a, b) in d.iter_mut().zip(other_d.iter()) {
                    *a += b;
                }
            }
            (me @ GradientRepr::Sparse(_), GradientRepr::Dense(other_d)) => {
                // Promote to dense
                let mut dense = me.to_dense();
                if dense.len() != other_d.len() {
                    return Err(AutogradError::shape_error(format!(
                        "GradientRepr accumulate: dense length mismatch ({} vs {})",
                        dense.len(),
                        other_d.len()
                    )));
                }
                for (a, b) in dense.iter_mut().zip(other_d.iter()) {
                    *a += b;
                }
                *me = GradientRepr::Dense(dense);
            }
        }
        Ok(())
    }
}

/// A variable that tracks sparse gradients during backward pass.
///
/// This is useful for embedding layers and other operations where
/// only a small subset of parameters are accessed per forward pass.
#[derive(Debug, Clone)]
pub struct SparseVariable {
    /// The parameter values (dense, since we need all of them for forward pass)
    data: Vec<f64>,
    /// Accumulated sparse gradient
    grad: Option<GradientRepr>,
    /// Name for identification
    name: String,
}

impl SparseVariable {
    /// Create a new sparse variable.
    pub fn new(data: Vec<f64>, name: impl Into<String>) -> Self {
        Self {
            data,
            grad: None,
            name: name.into(),
        }
    }

    /// Get the parameter data.
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Get mutable access to the parameter data.
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Get the name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the accumulated gradient.
    pub fn grad(&self) -> Option<&GradientRepr> {
        self.grad.as_ref()
    }

    /// Accumulate a sparse gradient from a backward pass.
    ///
    /// # Errors
    /// Returns error on shape mismatch.
    pub fn accumulate_sparse_grad(&mut self, grad: SparseGrad) -> Result<()> {
        if grad.shape() != self.data.len() {
            return Err(AutogradError::shape_error(format!(
                "SparseVariable '{}': gradient shape {} != data length {}",
                self.name,
                grad.shape(),
                self.data.len()
            )));
        }
        match &mut self.grad {
            Some(existing) => {
                existing.accumulate(&GradientRepr::Sparse(grad))?;
            }
            None => {
                self.grad = Some(GradientRepr::Sparse(grad));
            }
        }
        Ok(())
    }

    /// Accumulate a dense gradient.
    ///
    /// # Errors
    /// Returns error on shape mismatch.
    pub fn accumulate_dense_grad(&mut self, grad: Vec<f64>) -> Result<()> {
        if grad.len() != self.data.len() {
            return Err(AutogradError::shape_error(format!(
                "SparseVariable '{}': gradient length {} != data length {}",
                self.name,
                grad.len(),
                self.data.len()
            )));
        }
        match &mut self.grad {
            Some(existing) => {
                existing.accumulate(&GradientRepr::Dense(grad))?;
            }
            None => {
                self.grad = Some(GradientRepr::Dense(grad));
            }
        }
        Ok(())
    }

    /// Zero out accumulated gradients.
    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    /// Apply accumulated gradient with a learning rate (SGD step).
    ///
    /// After applying, the gradient is zeroed out.
    pub fn apply_grad(&mut self, lr: f64) {
        if let Some(ref grad) = self.grad {
            match grad {
                GradientRepr::Dense(d) => {
                    for (param, g) in self.data.iter_mut().zip(d.iter()) {
                        *param -= lr * g;
                    }
                }
                GradientRepr::Sparse(s) => {
                    for (&idx, &val) in s.indices().iter().zip(s.values().iter()) {
                        self.data[idx] -= lr * val;
                    }
                }
            }
        }
        self.grad = None;
    }

    /// Get the dense gradient vector, or None if no gradient has been accumulated.
    pub fn dense_grad(&self) -> Option<Vec<f64>> {
        self.grad.as_ref().map(|g| g.to_dense())
    }
}

impl fmt::Display for SparseVariable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let grad_info = match &self.grad {
            Some(GradientRepr::Dense(d)) => format!("dense({})", d.len()),
            Some(GradientRepr::Sparse(s)) => format!("{}", s),
            None => "no grad".to_string(),
        };
        write!(
            f,
            "SparseVariable('{}', size={}, grad={})",
            self.name,
            self.data.len(),
            grad_info
        )
    }
}

/// Sparse gradient for embedding layers.
///
/// Only updates the rows that were accessed during the forward pass,
/// making gradient computation much more efficient for large embedding tables.
#[derive(Debug, Clone)]
pub struct EmbeddingGrad {
    /// Number of embedding vectors (vocabulary size)
    num_embeddings: usize,
    /// Dimension of each embedding vector
    embedding_dim: usize,
    /// Map from row index to accumulated gradient vector for that row
    row_grads: HashMap<usize, Vec<f64>>,
}

impl EmbeddingGrad {
    /// Create a new embedding gradient tracker.
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        Self {
            num_embeddings,
            embedding_dim,
            row_grads: HashMap::new(),
        }
    }

    /// Accumulate a gradient for a specific embedding row.
    ///
    /// # Errors
    /// Returns error if row index is out of bounds or grad dimension mismatches.
    pub fn accumulate_row(&mut self, row_idx: usize, grad: &[f64]) -> Result<()> {
        if row_idx >= self.num_embeddings {
            return Err(AutogradError::invalid_argument(format!(
                "EmbeddingGrad: row index {} out of bounds (num_embeddings={})",
                row_idx, self.num_embeddings
            )));
        }
        if grad.len() != self.embedding_dim {
            return Err(AutogradError::shape_error(format!(
                "EmbeddingGrad: grad dim {} != embedding_dim {}",
                grad.len(),
                self.embedding_dim
            )));
        }
        let entry = self
            .row_grads
            .entry(row_idx)
            .or_insert_with(|| vec![0.0; self.embedding_dim]);
        for (a, &b) in entry.iter_mut().zip(grad.iter()) {
            *a += b;
        }
        Ok(())
    }

    /// Get the number of rows that received gradients.
    pub fn num_updated_rows(&self) -> usize {
        self.row_grads.len()
    }

    /// Get the gradient for a specific row, or None if that row was not accessed.
    pub fn row_grad(&self, row_idx: usize) -> Option<&[f64]> {
        self.row_grads.get(&row_idx).map(|v| v.as_slice())
    }

    /// Iterate over all (row_index, gradient) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&usize, &Vec<f64>)> {
        self.row_grads.iter()
    }

    /// Convert to a flat SparseGrad where the total shape is num_embeddings * embedding_dim.
    pub fn to_sparse_grad(&self) -> SparseGrad {
        let total = self.num_embeddings * self.embedding_dim;
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for (&row_idx, row_grad) in &self.row_grads {
            let base = row_idx * self.embedding_dim;
            for (j, &val) in row_grad.iter().enumerate() {
                if val.abs() > f64::EPSILON {
                    indices.push(base + j);
                    values.push(val);
                }
            }
        }
        // This is safe because we validated indices during accumulate_row
        SparseGrad {
            indices,
            values,
            shape: total,
        }
    }

    /// Convert to a full dense gradient matrix (flattened).
    pub fn to_dense(&self) -> Vec<f64> {
        let total = self.num_embeddings * self.embedding_dim;
        let mut dense = vec![0.0; total];
        for (&row_idx, row_grad) in &self.row_grads {
            let base = row_idx * self.embedding_dim;
            for (j, &val) in row_grad.iter().enumerate() {
                dense[base + j] = val;
            }
        }
        dense
    }

    /// Apply the sparse gradient to an embedding table with a learning rate.
    ///
    /// Only updates the rows that received gradients.
    ///
    /// # Errors
    /// Returns error if the embedding table dimensions don't match.
    pub fn apply_to_embedding(&self, embedding: &mut [f64], lr: f64) -> Result<()> {
        let expected = self.num_embeddings * self.embedding_dim;
        if embedding.len() != expected {
            return Err(AutogradError::shape_error(format!(
                "EmbeddingGrad apply: embedding size {} != expected {}",
                embedding.len(),
                expected
            )));
        }
        for (&row_idx, row_grad) in &self.row_grads {
            let base = row_idx * self.embedding_dim;
            for (j, &g) in row_grad.iter().enumerate() {
                embedding[base + j] -= lr * g;
            }
        }
        Ok(())
    }

    /// Clear all accumulated gradients.
    pub fn zero_grad(&mut self) {
        self.row_grads.clear();
    }
}

impl fmt::Display for EmbeddingGrad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sparsity = if self.num_embeddings > 0 {
            1.0 - (self.num_updated_rows() as f64 / self.num_embeddings as f64)
        } else {
            1.0
        };
        write!(
            f,
            "EmbeddingGrad(vocab={}, dim={}, updated={}, sparsity={:.1}%)",
            self.num_embeddings,
            self.embedding_dim,
            self.num_updated_rows(),
            sparsity * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_grad_new_and_to_dense() {
        let sg =
            SparseGrad::new(vec![0, 3, 7], vec![1.0, 2.0, 3.0], 10).expect("valid sparse grad");
        let dense = sg.to_dense();
        assert_eq!(dense.len(), 10);
        assert!((dense[0] - 1.0).abs() < 1e-12);
        assert!((dense[3] - 2.0).abs() < 1e-12);
        assert!((dense[7] - 3.0).abs() < 1e-12);
        assert!((dense[1]).abs() < 1e-12);
    }

    #[test]
    fn test_sparse_grad_from_dense_roundtrip() {
        let dense_orig = vec![0.0, 1.5, 0.0, -2.0, 0.0, 0.0, 3.25, 0.0];
        let sg = SparseGrad::from_dense(&dense_orig, 1e-10);
        assert_eq!(sg.nnz(), 3);
        let dense_back = sg.to_dense();
        for (a, b) in dense_orig.iter().zip(dense_back.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_sparse_grad_accumulate() {
        let mut sg1 = SparseGrad::new(vec![0, 2], vec![1.0, 2.0], 5).expect("valid");
        let sg2 = SparseGrad::new(vec![2, 4], vec![3.0, 4.0], 5).expect("valid");
        sg1.accumulate(&sg2).expect("accumulate ok");
        let dense = sg1.to_dense();
        assert!((dense[0] - 1.0).abs() < 1e-12);
        assert!((dense[2] - 5.0).abs() < 1e-12); // 2.0 + 3.0
        assert!((dense[4] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_sparse_grad_merge() {
        let sg1 = SparseGrad::new(vec![1], vec![10.0], 5).expect("valid");
        let sg2 = SparseGrad::new(vec![1, 3], vec![5.0, 7.0], 5).expect("valid");
        let merged = SparseGrad::merge(&sg1, &sg2).expect("merge ok");
        let dense = merged.to_dense();
        assert!((dense[1] - 15.0).abs() < 1e-12);
        assert!((dense[3] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_sparse_add_to_dense() {
        let sg = SparseGrad::new(vec![0, 2], vec![1.0, 3.0], 4).expect("valid");
        let mut dense = vec![10.0, 20.0, 30.0, 40.0];
        sg.add_to_dense(&mut dense).expect("add ok");
        assert!((dense[0] - 11.0).abs() < 1e-12);
        assert!((dense[2] - 33.0).abs() < 1e-12);
        assert!((dense[1] - 20.0).abs() < 1e-12);
    }

    #[test]
    fn test_sparse_grad_sparsify() {
        let mut sg =
            SparseGrad::new(vec![0, 1, 2, 3], vec![1.0, 0.001, 2.0, 0.0001], 10).expect("valid");
        sg.sparsify(0.01);
        assert_eq!(sg.nnz(), 2);
        assert_eq!(sg.indices(), &[0, 2]);
    }

    #[test]
    fn test_sparse_grad_shape_mismatch() {
        let sg1 = SparseGrad::new(vec![0], vec![1.0], 5).expect("valid");
        let sg2 = SparseGrad::new(vec![0], vec![1.0], 10).expect("valid");
        assert!(SparseGrad::merge(&sg1, &sg2).is_err());
    }

    #[test]
    fn test_sparse_grad_index_out_of_bounds() {
        assert!(SparseGrad::new(vec![10], vec![1.0], 5).is_err());
    }

    #[test]
    fn test_sparse_grad_norms() {
        let sg = SparseGrad::new(vec![0, 1, 2], vec![3.0, 4.0, 0.0], 10).expect("valid");
        assert!((sg.l2_norm() - 5.0).abs() < 1e-12);
        assert!((sg.l1_norm() - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_sparse_grad_scale_and_clip() {
        let mut sg = SparseGrad::new(vec![0, 1], vec![2.0, -3.0], 5).expect("valid");
        sg.scale(2.0);
        assert!((sg.values()[0] - 4.0).abs() < 1e-12);
        assert!((sg.values()[1] - (-6.0)).abs() < 1e-12);
        sg.clip(5.0);
        assert!((sg.values()[0] - 4.0).abs() < 1e-12);
        assert!((sg.values()[1] - (-5.0)).abs() < 1e-12);
    }

    #[test]
    fn test_sparse_grad_duplicate_indices() {
        let sg = SparseGrad::new(vec![1, 1, 1], vec![1.0, 2.0, 3.0], 5).expect("valid");
        assert_eq!(sg.nnz(), 1);
        assert!((sg.values()[0] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_gradient_repr_accumulate_sparse_sparse() {
        let s1 = SparseGrad::new(vec![0], vec![1.0], 3).expect("valid");
        let s2 = SparseGrad::new(vec![2], vec![5.0], 3).expect("valid");
        let mut g1 = GradientRepr::Sparse(s1);
        g1.accumulate(&GradientRepr::Sparse(s2)).expect("ok");
        let dense = g1.to_dense();
        assert!((dense[0] - 1.0).abs() < 1e-12);
        assert!((dense[2] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_gradient_repr_accumulate_dense_sparse() {
        let s = SparseGrad::new(vec![1], vec![10.0], 3).expect("valid");
        let mut g = GradientRepr::Dense(vec![1.0, 2.0, 3.0]);
        g.accumulate(&GradientRepr::Sparse(s)).expect("ok");
        let dense = g.to_dense();
        assert!((dense[1] - 12.0).abs() < 1e-12);
    }

    #[test]
    fn test_sparse_variable_accumulate_and_apply() {
        let mut var = SparseVariable::new(vec![10.0, 20.0, 30.0, 40.0, 50.0], "test");
        let grad = SparseGrad::new(vec![1, 3], vec![2.0, 4.0], 5).expect("valid");
        var.accumulate_sparse_grad(grad).expect("ok");
        var.apply_grad(0.1); // lr = 0.1
        assert!((var.data()[0] - 10.0).abs() < 1e-12);
        assert!((var.data()[1] - 19.8).abs() < 1e-12); // 20 - 0.1*2
        assert!((var.data()[3] - 39.6).abs() < 1e-12); // 40 - 0.1*4
    }

    #[test]
    fn test_sparse_variable_multiple_accumulate() {
        let mut var = SparseVariable::new(vec![0.0; 5], "test");
        let g1 = SparseGrad::new(vec![0], vec![1.0], 5).expect("valid");
        let g2 = SparseGrad::new(vec![0, 2], vec![2.0, 3.0], 5).expect("valid");
        var.accumulate_sparse_grad(g1).expect("ok");
        var.accumulate_sparse_grad(g2).expect("ok");
        let dense = var.dense_grad().expect("has grad");
        assert!((dense[0] - 3.0).abs() < 1e-12); // 1 + 2
        assert!((dense[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_embedding_grad_basic() {
        let mut eg = EmbeddingGrad::new(1000, 64);
        let grad_row = vec![0.1; 64];
        eg.accumulate_row(5, &grad_row).expect("ok");
        eg.accumulate_row(100, &grad_row).expect("ok");
        eg.accumulate_row(5, &grad_row).expect("ok"); // double-accumulate
        assert_eq!(eg.num_updated_rows(), 2);
        let row5 = eg.row_grad(5).expect("should exist");
        assert!((row5[0] - 0.2).abs() < 1e-12); // accumulated twice
    }

    #[test]
    fn test_embedding_grad_to_sparse() {
        let mut eg = EmbeddingGrad::new(10, 3);
        eg.accumulate_row(2, &[1.0, 0.0, 2.0]).expect("ok");
        let sg = eg.to_sparse_grad();
        assert_eq!(sg.shape(), 30); // 10 * 3
        let dense = sg.to_dense();
        assert!((dense[6] - 1.0).abs() < 1e-12); // row 2, col 0
        assert!((dense[8] - 2.0).abs() < 1e-12); // row 2, col 2
    }

    #[test]
    fn test_embedding_grad_apply() {
        let mut eg = EmbeddingGrad::new(3, 2);
        eg.accumulate_row(1, &[10.0, 20.0]).expect("ok");
        let mut embedding = vec![0.0; 6]; // 3x2
        eg.apply_to_embedding(&mut embedding, 0.1).expect("ok");
        assert!((embedding[2] - (-1.0)).abs() < 1e-12); // row 1, col 0: 0 - 0.1*10
        assert!((embedding[3] - (-2.0)).abs() < 1e-12); // row 1, col 1: 0 - 0.1*20
        assert!((embedding[0]).abs() < 1e-12); // row 0 untouched
    }

    #[test]
    fn test_sparse_grad_zeros() {
        let sg = SparseGrad::zeros(100);
        assert_eq!(sg.nnz(), 0);
        assert_eq!(sg.shape(), 100);
        assert!((sg.sparsity() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_display_formats() {
        let sg = SparseGrad::new(vec![0, 1], vec![1.0, 2.0], 100).expect("valid");
        let display = format!("{}", sg);
        assert!(display.contains("nnz=2"));
        assert!(display.contains("shape=100"));

        let var = SparseVariable::new(vec![0.0; 10], "weights");
        let display = format!("{}", var);
        assert!(display.contains("weights"));
        assert!(display.contains("size=10"));

        let eg = EmbeddingGrad::new(1000, 64);
        let display = format!("{}", eg);
        assert!(display.contains("vocab=1000"));
    }
}
