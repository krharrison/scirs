//! Locally Linear Embedding (LLE) for non-linear dimensionality reduction
//!
//! LLE is a non-linear dimensionality reduction method that assumes the data lies
//! on a low-dimensional manifold that is locally linear. It preserves local
//! neighborhood structure in the embedding space.
//!
//! ## Algorithm Overview
//!
//! 1. **k-NN computation**: Find k nearest neighbors for each point
//! 2. **Weight computation**: Solve least-squares for reconstruction weights
//! 3. **Embedding**: Find eigenvectors of (I-W)^T(I-W) corresponding to smallest eigenvalues
//!
//! ## Variants
//!
//! - **Standard LLE**: Classic locally linear embedding
//! - **Modified LLE (MLLE)**: Uses multiple weight vectors for robustness
//! - **Hessian LLE (HLLE)**: Uses Hessian of the local geometry

use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::validation::{check_positive, checkshape};
use scirs2_linalg::{eigh, solve, svd};
use std::collections::BinaryHeap;

use crate::error::{Result, TransformError};

/// LLE method variant
#[derive(Debug, Clone, PartialEq)]
pub enum LLEMethod {
    /// Standard LLE
    Standard,
    /// Modified LLE with multiple weight vectors
    Modified,
    /// Hessian LLE using local Hessian estimation
    Hessian,
}

/// Locally Linear Embedding (LLE) dimensionality reduction
///
/// LLE finds a low-dimensional embedding that preserves local linear structure.
/// Each point is reconstructed from its neighbors with fixed weights, and the
/// embedding preserves these reconstruction weights.
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_transform::LLE;
/// use scirs2_core::ndarray::Array2;
///
/// let data = Array2::<f64>::zeros((50, 10));
/// let mut lle = LLE::new(10, 2);
/// let embedding = lle.fit_transform(&data).expect("should succeed");
/// assert_eq!(embedding.shape(), &[50, 2]);
/// ```
#[derive(Debug, Clone)]
pub struct LLE {
    /// Number of neighbors to use
    n_neighbors: usize,
    /// Number of components in the embedding
    n_components: usize,
    /// Regularization parameter
    reg: f64,
    /// Method variant
    method: LLEMethod,
    /// The embedding
    embedding: Option<Array2<f64>>,
    /// Reconstruction weights
    weights: Option<Array2<f64>>,
    /// Training data for out-of-sample extension
    training_data: Option<Array2<f64>>,
    /// Reconstruction error
    reconstruction_error: Option<f64>,
}

impl LLE {
    /// Creates a new LLE instance
    ///
    /// # Arguments
    /// * `n_neighbors` - Number of neighbors to use
    /// * `n_components` - Number of dimensions in the embedding
    pub fn new(n_neighbors: usize, n_components: usize) -> Self {
        LLE {
            n_neighbors,
            n_components,
            reg: 1e-3,
            method: LLEMethod::Standard,
            embedding: None,
            weights: None,
            training_data: None,
            reconstruction_error: None,
        }
    }

    /// Set the regularization parameter
    pub fn with_regularization(mut self, reg: f64) -> Self {
        self.reg = reg;
        self
    }

    /// Set the LLE method variant
    pub fn with_method(mut self, method: &str) -> Self {
        self.method = match method {
            "modified" | "mlle" => LLEMethod::Modified,
            "hessian" | "hlle" => LLEMethod::Hessian,
            _ => LLEMethod::Standard,
        };
        self
    }

    /// Set the LLE method variant (typed)
    pub fn with_method_type(mut self, method: LLEMethod) -> Self {
        self.method = method;
        self
    }

    /// Find k nearest neighbors for each point
    fn find_neighbors<S>(&self, x: &ArrayBase<S, Ix2>) -> (Array2<usize>, Array2<f64>)
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n_samples = x.shape()[0];
        let mut indices = Array2::zeros((n_samples, self.n_neighbors));
        let mut distances = Array2::zeros((n_samples, self.n_neighbors));

        for i in 0..n_samples {
            let mut heap: BinaryHeap<(std::cmp::Reverse<i64>, usize)> = BinaryHeap::new();

            for j in 0..n_samples {
                if i != j {
                    let mut dist = 0.0;
                    for k in 0..x.shape()[1] {
                        let diff: f64 = NumCast::from(x[[i, k]]).unwrap_or(0.0)
                            - NumCast::from(x[[j, k]]).unwrap_or(0.0);
                        dist += diff * diff;
                    }
                    dist = dist.sqrt();

                    let dist_fixed = (dist * 1e9) as i64;
                    heap.push((std::cmp::Reverse(dist_fixed), j));
                }
            }

            for j in 0..self.n_neighbors {
                if let Some((std::cmp::Reverse(dist_fixed), idx)) = heap.pop() {
                    indices[[i, j]] = idx;
                    distances[[i, j]] = dist_fixed as f64 / 1e9;
                }
            }
        }

        (indices, distances)
    }

    /// Compute reconstruction weights (standard LLE)
    fn compute_weights<S>(
        &self,
        x: &ArrayBase<S, Ix2>,
        neighbors: &Array2<usize>,
    ) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];
        let k = self.n_neighbors;

        let mut weights = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            // Create local covariance matrix
            let mut c = Array2::zeros((k, k));
            let xi = x.index_axis(Axis(0), i);

            for j in 0..k {
                let neighbor_j = neighbors[[i, j]];
                let xj = x.index_axis(Axis(0), neighbor_j);

                for l in 0..k {
                    let neighbor_l = neighbors[[i, l]];
                    let xl = x.index_axis(Axis(0), neighbor_l);

                    let mut dot = 0.0;
                    for m in 0..n_features {
                        let diff_j: f64 = NumCast::from(xi[m]).unwrap_or(0.0)
                            - NumCast::from(xj[m]).unwrap_or(0.0);
                        let diff_l: f64 = NumCast::from(xi[m]).unwrap_or(0.0)
                            - NumCast::from(xl[m]).unwrap_or(0.0);
                        dot += diff_j * diff_l;
                    }
                    c[[j, l]] = dot;
                }
            }

            // Add regularization to diagonal
            let trace: f64 = (0..k).map(|j| c[[j, j]]).sum();
            let reg_value = self.reg * trace / k as f64;
            for j in 0..k {
                c[[j, j]] += reg_value;
            }

            // Solve C * w = 1 for weights
            let ones = Array1::ones(k);
            let w = match solve(&c.view(), &ones.view(), None) {
                Ok(solution) => solution,
                Err(_) => Array1::from_elem(k, 1.0 / k as f64),
            };

            // Normalize weights to sum to 1
            let w_sum = w.sum();
            let w_normalized = if w_sum.abs() > 1e-10 {
                w / w_sum
            } else {
                Array1::from_elem(k, 1.0 / k as f64)
            };

            for j in 0..k {
                let neighbor = neighbors[[i, j]];
                weights[[i, neighbor]] = w_normalized[j];
            }
        }

        Ok(weights)
    }

    /// Compute weights for Modified LLE (MLLE)
    ///
    /// Modified LLE uses multiple weight vectors to provide more robust embeddings.
    /// For each point, it computes the SVD of the local neighborhood and uses
    /// the bottom singular vectors to form a more stable weight matrix.
    fn compute_weights_modified<S>(
        &self,
        x: &ArrayBase<S, Ix2>,
        neighbors: &Array2<usize>,
    ) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];
        let k = self.n_neighbors;
        let d = self.n_components;

        // Number of extra weight vectors for MLLE
        let n_extra = k.saturating_sub(d + 1).min(d);
        if n_extra == 0 {
            // Fall back to standard LLE
            return self.compute_weights(x, neighbors);
        }

        let mut weights = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            // Get the local neighborhood centered at xi
            let mut local_data = Array2::zeros((k, n_features));
            let xi = x.index_axis(Axis(0), i);

            for j in 0..k {
                let neighbor_j = neighbors[[i, j]];
                for m in 0..n_features {
                    let val_i: f64 = NumCast::from(xi[m]).unwrap_or(0.0);
                    let val_j: f64 = NumCast::from(x[[neighbor_j, m]]).unwrap_or(0.0);
                    local_data[[j, m]] = val_j - val_i;
                }
            }

            // SVD of the centered local neighborhood
            let svd_result = svd::<f64>(&local_data.view(), true, None);
            let (u, _s, _vt) = match svd_result {
                Ok(result) => result,
                Err(_) => {
                    // Fallback to standard weights for this point
                    let ones = Array1::from_elem(k, 1.0 / k as f64);
                    for j in 0..k {
                        weights[[i, neighbors[[i, j]]]] = ones[j];
                    }
                    continue;
                }
            };

            // Use the last (k - d) singular vectors of U
            // These span the "null space" and give the weight vectors
            let start_col = d.min(u.shape()[1].saturating_sub(1));
            let n_weight_vecs = (u.shape()[1] - start_col).min(n_extra + 1).max(1);

            // Average the squared singular vectors to form the weight vector
            let mut w = Array1::zeros(k);
            for j in 0..n_weight_vecs {
                let col_idx = start_col + j;
                if col_idx < u.shape()[1] {
                    for r in 0..k {
                        w[r] += u[[r, col_idx]] * u[[r, col_idx]];
                    }
                }
            }

            // Normalize
            let w_sum = w.sum();
            if w_sum > 1e-10 {
                w.mapv_inplace(|v| v / w_sum);
            } else {
                w = Array1::from_elem(k, 1.0 / k as f64);
            }

            for j in 0..k {
                weights[[i, neighbors[[i, j]]]] = w[j];
            }
        }

        Ok(weights)
    }

    /// Compute weights for Hessian LLE (HLLE)
    ///
    /// Hessian LLE estimates the local Hessian of the manifold and uses it
    /// to construct a weight matrix that better captures local curvature.
    fn compute_weights_hessian<S>(
        &self,
        x: &ArrayBase<S, Ix2>,
        neighbors: &Array2<usize>,
    ) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];
        let k = self.n_neighbors;
        let d = self.n_components;

        // HLLE requires k > d * (d + 3) / 2
        let min_k = d * (d + 3) / 2 + 1;
        if k < min_k {
            return Err(TransformError::InvalidInput(format!(
                "Hessian LLE requires n_neighbors >= {} for n_components={}, got {}",
                min_k, d, k
            )));
        }

        // Number of Hessian components
        let dp = d * (d + 1) / 2;

        let mut weights = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            // Get centered local neighborhood
            let mut local_data = Array2::zeros((k, n_features));
            let xi = x.index_axis(Axis(0), i);

            for j in 0..k {
                let neighbor_j = neighbors[[i, j]];
                for m in 0..n_features {
                    let val_i: f64 = NumCast::from(xi[m]).unwrap_or(0.0);
                    let val_j: f64 = NumCast::from(x[[neighbor_j, m]]).unwrap_or(0.0);
                    local_data[[j, m]] = val_j - val_i;
                }
            }

            // SVD of local neighborhood to get the tangent space
            let (u, _s, _vt) = match svd::<f64>(&local_data.view(), true, None) {
                Ok(result) => result,
                Err(_) => {
                    let ones = Array1::from_elem(k, 1.0 / k as f64);
                    for j in 0..k {
                        weights[[i, neighbors[[i, j]]]] = ones[j];
                    }
                    continue;
                }
            };

            // Take the first d columns of U as the tangent coordinates
            let mut tangent = Array2::zeros((k, d));
            let max_d = d.min(u.shape()[1]);
            for j in 0..max_d {
                for r in 0..k {
                    tangent[[r, j]] = u[[r, j]];
                }
            }

            // Build the Hessian estimator matrix
            // Columns: [1, t1, t2, ..., td, t1*t1, t1*t2, ..., td*td]
            let n_cols = 1 + d + dp;
            let mut h_mat = Array2::zeros((k, n_cols));

            for r in 0..k {
                h_mat[[r, 0]] = 1.0; // constant
                for j in 0..max_d {
                    h_mat[[r, 1 + j]] = tangent[[r, j]]; // linear
                }

                // Quadratic terms
                let mut col = 1 + d;
                for j in 0..max_d {
                    for l in j..max_d {
                        h_mat[[r, col]] = tangent[[r, j]] * tangent[[r, l]];
                        col += 1;
                    }
                }
            }

            // QR decomposition via Gram-Schmidt to get the null space
            // We want the projection onto the null space of H^T
            let (q, _r) = self.qr_decomposition(&h_mat)?;

            // The weight vector comes from the columns of Q beyond the first n_cols
            let mut w = Array1::zeros(k);
            let start_col = n_cols.min(q.shape()[1]);
            let mut count = 0;
            for col in start_col..q.shape()[1] {
                for r in 0..k {
                    w[r] += q[[r, col]] * q[[r, col]];
                }
                count += 1;
            }

            if count == 0 {
                // Fallback
                w = Array1::from_elem(k, 1.0 / k as f64);
            } else {
                let w_sum = w.sum();
                if w_sum > 1e-10 {
                    w.mapv_inplace(|v| v / w_sum);
                } else {
                    w = Array1::from_elem(k, 1.0 / k as f64);
                }
            }

            for j in 0..k {
                weights[[i, neighbors[[i, j]]]] = w[j];
            }
        }

        Ok(weights)
    }

    /// Simple QR decomposition using modified Gram-Schmidt
    fn qr_decomposition(&self, a: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
        let (m, n) = a.dim();
        let mut q = a.clone();
        let mut r = Array2::zeros((n, n));

        for j in 0..n {
            // Normalize the j-th column
            let mut norm = 0.0;
            for i in 0..m {
                norm += q[[i, j]] * q[[i, j]];
            }
            norm = norm.sqrt();

            r[[j, j]] = norm;
            if norm > 1e-14 {
                for i in 0..m {
                    q[[i, j]] /= norm;
                }
            }

            // Orthogonalize remaining columns against q_j
            for k in (j + 1)..n {
                let mut dot = 0.0;
                for i in 0..m {
                    dot += q[[i, j]] * q[[i, k]];
                }
                r[[j, k]] = dot;
                for i in 0..m {
                    q[[i, k]] -= dot * q[[i, j]];
                }
            }
        }

        Ok((q, r))
    }

    /// Compute the embedding from reconstruction weights
    fn compute_embedding(&self, weights: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = weights.shape()[0];

        // Construct the cost matrix M = (I - W)^T (I - W)
        let mut m = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in 0..n_samples {
                let mut sum = 0.0;

                if i == j {
                    sum += 1.0 - 2.0 * weights[[i, j]] + weights.column(j).dot(&weights.column(j));
                } else {
                    sum += -weights[[i, j]] - weights[[j, i]]
                        + weights.column(i).dot(&weights.column(j));
                }

                m[[i, j]] = sum;
            }
        }

        // Find the eigenvectors corresponding to the smallest eigenvalues
        let (eigenvalues, eigenvectors) =
            eigh(&m.view(), None).map_err(|e| TransformError::LinalgError(e))?;

        // Sort eigenvalues and eigenvectors
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.sort_by(|&i, &j| {
            eigenvalues[i]
                .partial_cmp(&eigenvalues[j])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Skip the first eigenvector (corresponding to eigenvalue ~0)
        // and take the next n_components eigenvectors
        let mut embedding = Array2::zeros((n_samples, self.n_components));
        for j in 0..self.n_components {
            let idx = indices[j + 1]; // Skip first eigenvector
            for i in 0..n_samples {
                embedding[[i, j]] = eigenvectors[[i, idx]];
            }
        }

        // Compute reconstruction error
        let recon_error: f64 = (0..self.n_components)
            .map(|j| {
                let idx = indices[j + 1];
                eigenvalues[idx].max(0.0)
            })
            .sum();

        // Store as side effect via return
        // (We'll compute it in fit())
        let _ = recon_error;

        Ok(embedding)
    }

    /// Compute reconstruction error
    fn compute_reconstruction_error(&self, weights: &Array2<f64>, embedding: &Array2<f64>) -> f64 {
        let n_samples = weights.shape()[0];
        let n_components = embedding.shape()[1];

        let mut total_error = 0.0;

        for i in 0..n_samples {
            for d in 0..n_components {
                let mut reconstructed = 0.0;
                for j in 0..n_samples {
                    reconstructed += weights[[i, j]] * embedding[[j, d]];
                }
                let diff = embedding[[i, d]] - reconstructed;
                total_error += diff * diff;
            }
        }

        total_error / n_samples as f64
    }

    /// Fits the LLE model to the input data
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let (n_samples, n_features) = x.dim();

        check_positive(self.n_neighbors, "n_neighbors")?;
        check_positive(self.n_components, "n_components")?;
        checkshape(x, &[n_samples, n_features], "x")?;

        if n_samples <= self.n_neighbors {
            return Err(TransformError::InvalidInput(format!(
                "n_neighbors={} must be < n_samples={}",
                self.n_neighbors, n_samples
            )));
        }

        if self.n_components >= n_samples {
            return Err(TransformError::InvalidInput(format!(
                "n_components={} must be < n_samples={}",
                self.n_components, n_samples
            )));
        }

        let x_f64 = x.mapv(|v| NumCast::from(v).unwrap_or(0.0));

        // Step 1: Find k nearest neighbors
        let (neighbors, _distances) = self.find_neighbors(&x_f64.view());

        // Step 2: Compute reconstruction weights based on method
        let weights = match &self.method {
            LLEMethod::Standard => self.compute_weights(&x_f64.view(), &neighbors)?,
            LLEMethod::Modified => self.compute_weights_modified(&x_f64.view(), &neighbors)?,
            LLEMethod::Hessian => self.compute_weights_hessian(&x_f64.view(), &neighbors)?,
        };

        // Step 3: Compute embedding from weights
        let embedding = self.compute_embedding(&weights)?;

        // Compute reconstruction error
        let recon_error = self.compute_reconstruction_error(&weights, &embedding);

        self.embedding = Some(embedding);
        self.weights = Some(weights);
        self.training_data = Some(x_f64);
        self.reconstruction_error = Some(recon_error);

        Ok(())
    }

    /// Transforms the input data using the fitted LLE model
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        if self.embedding.is_none() {
            return Err(TransformError::NotFitted(
                "LLE model has not been fitted".to_string(),
            ));
        }

        let training_data = self
            .training_data
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Training data not available".to_string()))?;

        let x_f64 = x.mapv(|v| NumCast::from(v).unwrap_or(0.0));

        if self.is_same_data(&x_f64, training_data) {
            return self
                .embedding
                .as_ref()
                .cloned()
                .ok_or_else(|| TransformError::NotFitted("Embedding not available".to_string()));
        }

        self.transform_new_data(&x_f64)
    }

    /// Fits the LLE model and transforms the data
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Returns the embedding
    pub fn embedding(&self) -> Option<&Array2<f64>> {
        self.embedding.as_ref()
    }

    /// Returns the reconstruction weights
    pub fn reconstruction_weights(&self) -> Option<&Array2<f64>> {
        self.weights.as_ref()
    }

    /// Returns the reconstruction error
    pub fn reconstruction_error(&self) -> Option<f64> {
        self.reconstruction_error
    }

    /// Check if the input data is the same as training data
    fn is_same_data(&self, x: &Array2<f64>, training_data: &Array2<f64>) -> bool {
        if x.dim() != training_data.dim() {
            return false;
        }
        let (n_samples, n_features) = x.dim();
        for i in 0..n_samples {
            for j in 0..n_features {
                if (x[[i, j]] - training_data[[i, j]]).abs() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    /// Transform new data using out-of-sample extension
    fn transform_new_data(&self, x_new: &Array2<f64>) -> Result<Array2<f64>> {
        let training_data = self
            .training_data
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Training data not available".to_string()))?;
        let training_embedding = self
            .embedding
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Embedding not available".to_string()))?;

        let (n_new, n_features) = x_new.dim();

        if n_features != training_data.ncols() {
            return Err(TransformError::InvalidInput(format!(
                "Input features {} must match training features {}",
                n_features,
                training_data.ncols()
            )));
        }

        let mut new_embedding = Array2::zeros((n_new, self.n_components));

        for i in 0..n_new {
            let new_coords =
                self.compute_new_point_embedding(&x_new.row(i), training_data, training_embedding)?;

            for j in 0..self.n_components {
                new_embedding[[i, j]] = new_coords[j];
            }
        }

        Ok(new_embedding)
    }

    /// Compute embedding coordinates for a single new point
    fn compute_new_point_embedding(
        &self,
        x_new: &scirs2_core::ndarray::ArrayView1<f64>,
        training_data: &Array2<f64>,
        training_embedding: &Array2<f64>,
    ) -> Result<Array1<f64>> {
        let n_training = training_data.nrows();
        let n_features = training_data.ncols();

        // Find k nearest neighbors in training data
        let mut distances: Vec<(f64, usize)> = Vec::with_capacity(n_training);
        for j in 0..n_training {
            let mut dist_sq = 0.0;
            for k in 0..n_features {
                let diff = x_new[k] - training_data[[j, k]];
                dist_sq += diff * diff;
            }
            distances.push((dist_sq.sqrt(), j));
        }

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let k = self.n_neighbors.min(n_training);
        let neighbor_indices: Vec<usize> =
            distances.into_iter().take(k).map(|(_, idx)| idx).collect();

        // Compute reconstruction weights
        let weights =
            self.compute_reconstruction_weights_for_point(x_new, training_data, &neighbor_indices)?;

        // Compute embedding as weighted combination
        let mut new_coords = Array1::zeros(self.n_components);
        for (i, &neighbor_idx) in neighbor_indices.iter().enumerate() {
            for dim in 0..self.n_components {
                new_coords[dim] += weights[i] * training_embedding[[neighbor_idx, dim]];
            }
        }

        Ok(new_coords)
    }

    /// Compute reconstruction weights for a single point given its neighbors
    fn compute_reconstruction_weights_for_point(
        &self,
        x_point: &scirs2_core::ndarray::ArrayView1<f64>,
        training_data: &Array2<f64>,
        neighbor_indices: &[usize],
    ) -> Result<Array1<f64>> {
        let k = neighbor_indices.len();
        let n_features = training_data.ncols();

        let mut c = Array2::zeros((k, k));

        for i in 0..k {
            let neighbor_i = neighbor_indices[i];
            for j in 0..k {
                let neighbor_j = neighbor_indices[j];

                let mut dot = 0.0;
                for m in 0..n_features {
                    let diff_i = x_point[m] - training_data[[neighbor_i, m]];
                    let diff_j = x_point[m] - training_data[[neighbor_j, m]];
                    dot += diff_i * diff_j;
                }
                c[[i, j]] = dot;
            }
        }

        let trace: f64 = (0..k).map(|i| c[[i, i]]).sum();
        let reg_value = self.reg * trace / k as f64;
        for i in 0..k {
            c[[i, i]] += reg_value;
        }

        let ones = Array1::ones(k);
        let w = match solve(&c.view(), &ones.view(), None) {
            Ok(solution) => solution,
            Err(_) => Array1::from_elem(k, 1.0 / k as f64),
        };

        let w_sum = w.sum();
        let w_normalized = if w_sum.abs() > 1e-10 {
            w / w_sum
        } else {
            Array1::from_elem(k, 1.0 / k as f64)
        };

        Ok(w_normalized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_lle_basic() {
        let n_points = 20;
        let mut data = Vec::new();

        for i in 0..n_points {
            let t = 1.5 * std::f64::consts::PI * (1.0 + 2.0 * i as f64 / n_points as f64);
            let x = t * t.cos();
            let y = 10.0 * i as f64 / n_points as f64;
            let z = t * t.sin();
            data.extend_from_slice(&[x, y, z]);
        }

        let x = Array::from_shape_vec((n_points, 3), data).expect("Failed to create array");

        let mut lle = LLE::new(5, 2);
        let embedding = lle.fit_transform(&x).expect("LLE fit_transform failed");

        assert_eq!(embedding.shape(), &[n_points, 2]);
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_lle_regularization() {
        let x: Array2<f64> = Array::eye(10) * 2.0;

        let mut lle = LLE::new(3, 2).with_regularization(0.01);
        let result = lle.fit_transform(&x);

        assert!(result.is_ok());
        let embedding = result.expect("LLE fit_transform failed");
        assert_eq!(embedding.shape(), &[10, 2]);
    }

    #[test]
    fn test_lle_modified() {
        let n_points = 20;
        let mut data = Vec::new();
        for i in 0..n_points {
            let t = i as f64 / n_points as f64 * 2.0 * std::f64::consts::PI;
            data.extend_from_slice(&[t.cos(), t.sin(), t * 0.1]);
        }

        let x = Array::from_shape_vec((n_points, 3), data).expect("Failed to create array");

        let mut lle = LLE::new(5, 2).with_method("modified");
        let embedding = lle.fit_transform(&x).expect("MLLE fit_transform failed");

        assert_eq!(embedding.shape(), &[n_points, 2]);
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_lle_hessian() {
        let n_points = 25;
        let mut data = Vec::new();
        for i in 0..n_points {
            let t = i as f64 / n_points as f64;
            data.extend_from_slice(&[t, t * 2.0, t * 3.0, t * t]);
        }

        let x = Array::from_shape_vec((n_points, 4), data).expect("Failed to create array");

        // HLLE requires k >= d*(d+3)/2 + 1 = 2*(2+3)/2 + 1 = 6
        let mut lle = LLE::new(7, 2).with_method("hessian");
        let embedding = lle.fit_transform(&x).expect("HLLE fit_transform failed");

        assert_eq!(embedding.shape(), &[n_points, 2]);
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_lle_invalid_params() {
        let x: Array2<f64> = Array::eye(5);

        let mut lle = LLE::new(10, 2);
        assert!(lle.fit(&x).is_err());

        let mut lle = LLE::new(2, 10);
        assert!(lle.fit(&x).is_err());
    }

    #[test]
    fn test_lle_reconstruction_error() {
        let n_points = 20;
        let mut data = Vec::new();
        for i in 0..n_points {
            let t = i as f64 / n_points as f64;
            data.extend_from_slice(&[t, t * 2.0, t * 3.0]);
        }

        let x = Array::from_shape_vec((n_points, 3), data).expect("Failed to create array");

        let mut lle = LLE::new(5, 2);
        let _ = lle.fit_transform(&x).expect("LLE fit_transform failed");

        let error = lle.reconstruction_error();
        assert!(error.is_some());
        let error_val = error.expect("Error should exist");
        assert!(error_val >= 0.0);
        assert!(error_val.is_finite());
    }

    #[test]
    fn test_lle_out_of_sample() {
        let n_points = 20;
        let mut data = Vec::new();
        for i in 0..n_points {
            let t = i as f64 / n_points as f64;
            data.extend_from_slice(&[t, t * 2.0, t * 3.0]);
        }

        let x = Array::from_shape_vec((n_points, 3), data).expect("Failed to create array");

        let mut lle = LLE::new(5, 2);
        lle.fit(&x).expect("LLE fit failed");

        let x_new = Array::from_shape_vec((2, 3), vec![0.25, 0.5, 0.75, 0.75, 1.5, 2.25])
            .expect("Failed to create test array");

        let new_embedding = lle.transform(&x_new).expect("LLE transform failed");
        assert_eq!(new_embedding.shape(), &[2, 2]);
        for val in new_embedding.iter() {
            assert!(val.is_finite());
        }
    }
}
