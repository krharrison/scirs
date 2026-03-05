//! Laplacian Eigenmaps for Nonlinear Dimensionality Reduction
//!
//! Laplacian Eigenmaps (Belkin & Niyogi, 2003) embeds data into a low-dimensional
//! space by preserving local neighborhood structure via the graph Laplacian.
//!
//! ## Algorithm
//!
//! 1. Construct a weighted graph (k-NN or heat kernel)
//! 2. Compute the graph Laplacian (normalized or unnormalized)
//! 3. Solve the generalized eigenvalue problem: L f = lambda D f
//! 4. Embed using eigenvectors corresponding to smallest non-zero eigenvalues
//!
//! ## Features
//!
//! - Heat kernel and k-NN graph construction
//! - Normalized and unnormalized Laplacian variants
//! - Out-of-sample extension via Nystrom approximation
//! - Automatic bandwidth selection for the heat kernel

use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::validation::{check_positive, checkshape};
use scirs2_linalg::eigh;

use crate::error::{Result, TransformError};

/// Graph construction method for Laplacian Eigenmaps
#[derive(Debug, Clone, PartialEq)]
pub enum GraphMethod {
    /// k-nearest neighbors graph with optional heat kernel weights
    KNN {
        /// Number of neighbors
        k: usize,
        /// If true, use heat kernel weights; otherwise binary weights
        heat_kernel: bool,
    },
    /// Epsilon-ball neighborhood graph
    EpsilonBall {
        /// Neighborhood radius
        epsilon: f64,
    },
    /// Full heat kernel (connect all points with Gaussian weights)
    FullHeatKernel,
}

/// Laplacian type
#[derive(Debug, Clone, PartialEq)]
pub enum LaplacianType {
    /// Unnormalized Laplacian: L = D - W
    Unnormalized,
    /// Normalized Laplacian (symmetric): L_sym = D^{-1/2} L D^{-1/2} = I - D^{-1/2} W D^{-1/2}
    NormalizedSymmetric,
    /// Normalized Laplacian (random walk): L_rw = D^{-1} L = I - D^{-1} W
    NormalizedRandomWalk,
}

/// Laplacian Eigenmaps for nonlinear dimensionality reduction
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_transform::reduction::laplacian_eigenmaps::{LaplacianEigenmaps, GraphMethod};
/// use scirs2_core::ndarray::Array2;
///
/// let data = Array2::<f64>::zeros((50, 10));
/// let mut le = LaplacianEigenmaps::new(2, GraphMethod::KNN { k: 10, heat_kernel: true });
/// let embedding = le.fit_transform(&data).expect("should succeed");
/// assert_eq!(embedding.shape(), &[50, 2]);
/// ```
#[derive(Debug, Clone)]
pub struct LaplacianEigenmaps {
    /// Number of components in the embedding
    n_components: usize,
    /// Method for constructing the graph
    graph_method: GraphMethod,
    /// Type of Laplacian to use
    laplacian_type: LaplacianType,
    /// Heat kernel bandwidth parameter (sigma). Auto-selected if None.
    sigma: Option<f64>,
    /// The embedding vectors
    embedding: Option<Array2<f64>>,
    /// Training data (for out-of-sample extension)
    training_data: Option<Array2<f64>>,
    /// Affinity (weight) matrix
    affinity_matrix: Option<Array2<f64>>,
    /// Eigenvalues from the decomposition
    eigenvalues: Option<Array1<f64>>,
    /// Eigenvectors from the decomposition
    eigenvectors: Option<Array2<f64>>,
    /// Degree vector
    degrees: Option<Array1<f64>>,
}

impl LaplacianEigenmaps {
    /// Create a new LaplacianEigenmaps instance
    ///
    /// # Arguments
    /// * `n_components` - Number of dimensions in the embedding
    /// * `graph_method` - Method for constructing the neighborhood graph
    pub fn new(n_components: usize, graph_method: GraphMethod) -> Self {
        LaplacianEigenmaps {
            n_components,
            graph_method,
            laplacian_type: LaplacianType::NormalizedSymmetric,
            sigma: None,
            embedding: None,
            training_data: None,
            affinity_matrix: None,
            eigenvalues: None,
            eigenvectors: None,
            degrees: None,
        }
    }

    /// Set the Laplacian type
    pub fn with_laplacian_type(mut self, laplacian_type: LaplacianType) -> Self {
        self.laplacian_type = laplacian_type;
        self
    }

    /// Set the heat kernel bandwidth (sigma)
    pub fn with_sigma(mut self, sigma: f64) -> Self {
        self.sigma = Some(sigma);
        self
    }

    /// Get the embedding
    pub fn embedding(&self) -> Option<&Array2<f64>> {
        self.embedding.as_ref()
    }

    /// Get the affinity matrix
    pub fn affinity_matrix(&self) -> Option<&Array2<f64>> {
        self.affinity_matrix.as_ref()
    }

    /// Get the eigenvalues
    pub fn eigenvalues(&self) -> Option<&Array1<f64>> {
        self.eigenvalues.as_ref()
    }

    /// Compute pairwise squared Euclidean distances
    fn compute_sq_distances(x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let d = x.ncols();
        let mut dist_sq = Array2::zeros((n, n));

        for i in 0..n {
            for j in (i + 1)..n {
                let mut sq = 0.0;
                for k in 0..d {
                    let diff = x[[i, k]] - x[[j, k]];
                    sq += diff * diff;
                }
                dist_sq[[i, j]] = sq;
                dist_sq[[j, i]] = sq;
            }
        }

        dist_sq
    }

    /// Estimate sigma using the median heuristic on neighbor distances
    fn estimate_sigma(dist_sq: &Array2<f64>, k: usize) -> f64 {
        let n = dist_sq.nrows();
        let mut kth_distances = Vec::with_capacity(n);

        for i in 0..n {
            let mut row_dists: Vec<f64> = (0..n)
                .filter(|&j| j != i)
                .map(|j| dist_sq[[i, j]])
                .collect();
            row_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let k_idx = (k - 1).min(row_dists.len().saturating_sub(1));
            kth_distances.push(row_dists[k_idx]);
        }

        kth_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_sq = kth_distances[kth_distances.len() / 2];

        // sigma = sqrt(median_kth_distance_sq)
        let sigma = median_sq.sqrt();
        if sigma < 1e-15 {
            1.0
        } else {
            sigma
        }
    }

    /// Construct the affinity (weight) matrix
    fn construct_affinity(&self, dist_sq: &Array2<f64>, sigma: f64) -> Result<Array2<f64>> {
        let n = dist_sq.nrows();
        let sigma_sq = sigma * sigma;
        let mut w: Array2<f64> = Array2::zeros((n, n));

        match &self.graph_method {
            GraphMethod::KNN { k, heat_kernel } => {
                let k_val = *k;
                if k_val >= n {
                    return Err(TransformError::InvalidInput(format!(
                        "k={} must be < n_samples={}",
                        k_val, n
                    )));
                }

                for i in 0..n {
                    // Find k-nearest neighbors
                    let mut neighbors: Vec<(f64, usize)> = (0..n)
                        .filter(|&j| j != i)
                        .map(|j| (dist_sq[[i, j]], j))
                        .collect();
                    neighbors
                        .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                    for idx in 0..k_val.min(neighbors.len()) {
                        let (d_sq, j) = neighbors[idx];
                        let weight = if *heat_kernel {
                            (-d_sq / (2.0 * sigma_sq)).exp()
                        } else {
                            1.0
                        };
                        // Make symmetric
                        w[[i, j]] = w[[i, j]].max(weight);
                        w[[j, i]] = w[[j, i]].max(weight);
                    }
                }
            }
            GraphMethod::EpsilonBall { epsilon } => {
                let eps_sq = epsilon * epsilon;
                for i in 0..n {
                    for j in (i + 1)..n {
                        if dist_sq[[i, j]] <= eps_sq {
                            let weight = (-dist_sq[[i, j]] / (2.0 * sigma_sq)).exp();
                            w[[i, j]] = weight;
                            w[[j, i]] = weight;
                        }
                    }
                }
            }
            GraphMethod::FullHeatKernel => {
                for i in 0..n {
                    for j in (i + 1)..n {
                        let weight = (-dist_sq[[i, j]] / (2.0 * sigma_sq)).exp();
                        w[[i, j]] = weight;
                        w[[j, i]] = weight;
                    }
                }
            }
        }

        Ok(w)
    }

    /// Compute the degree vector
    fn compute_degrees(w: &Array2<f64>) -> Array1<f64> {
        let n = w.nrows();
        let mut d = Array1::zeros(n);
        for i in 0..n {
            d[i] = w.row(i).sum();
        }
        d
    }

    /// Compute the graph Laplacian and solve the eigenvalue problem
    fn compute_embedding(
        &self,
        w: &Array2<f64>,
        degrees: &Array1<f64>,
    ) -> Result<(Array1<f64>, Array2<f64>)> {
        let n = w.nrows();

        // Check for isolated nodes
        for i in 0..n {
            if degrees[i] < 1e-15 {
                return Err(TransformError::ComputationError(format!(
                    "Node {} is isolated (zero degree). Increase k or epsilon.",
                    i
                )));
            }
        }

        match &self.laplacian_type {
            LaplacianType::Unnormalized => {
                // L = D - W
                // Solve L f = lambda D f  =>  D^{-1} L f = lambda f
                // Equivalently, solve D^{-1/2} L D^{-1/2} g = lambda g, then f = D^{-1/2} g
                let mut l_sym = Array2::zeros((n, n));
                for i in 0..n {
                    let d_i_inv_sqrt = 1.0 / degrees[i].sqrt();
                    for j in 0..n {
                        let d_j_inv_sqrt = 1.0 / degrees[j].sqrt();
                        if i == j {
                            l_sym[[i, j]] = 1.0 - w[[i, j]] / degrees[i];
                        } else {
                            l_sym[[i, j]] = -w[[i, j]] * d_i_inv_sqrt * d_j_inv_sqrt;
                        }
                    }
                }

                let (eigenvalues, eigenvectors) =
                    eigh(&l_sym.view(), None).map_err(TransformError::LinalgError)?;

                // Transform back: f = D^{-1/2} g
                let mut f_vecs = eigenvectors.clone();
                for i in 0..n {
                    let d_inv_sqrt = 1.0 / degrees[i].sqrt();
                    for j in 0..n {
                        f_vecs[[i, j]] *= d_inv_sqrt;
                    }
                }

                Ok((eigenvalues, f_vecs))
            }
            LaplacianType::NormalizedSymmetric => {
                // L_sym = I - D^{-1/2} W D^{-1/2}
                let mut l_sym = Array2::zeros((n, n));
                for i in 0..n {
                    let d_i_inv_sqrt = 1.0 / degrees[i].sqrt();
                    for j in 0..n {
                        let d_j_inv_sqrt = 1.0 / degrees[j].sqrt();
                        if i == j {
                            l_sym[[i, j]] = 1.0 - w[[i, j]] * d_i_inv_sqrt * d_j_inv_sqrt;
                        } else {
                            l_sym[[i, j]] = -w[[i, j]] * d_i_inv_sqrt * d_j_inv_sqrt;
                        }
                    }
                }

                eigh(&l_sym.view(), None).map_err(TransformError::LinalgError)
            }
            LaplacianType::NormalizedRandomWalk => {
                // L_rw = I - D^{-1} W
                // This is not symmetric, so we solve via the symmetric form
                // L_sym = D^{1/2} L_rw D^{-1/2}
                // The eigenvectors of L_rw are D^{-1/2} * eigenvectors of L_sym
                let mut l_sym = Array2::zeros((n, n));
                for i in 0..n {
                    let d_i_inv_sqrt = 1.0 / degrees[i].sqrt();
                    for j in 0..n {
                        let d_j_inv_sqrt = 1.0 / degrees[j].sqrt();
                        if i == j {
                            l_sym[[i, j]] = 1.0 - w[[i, j]] * d_i_inv_sqrt * d_j_inv_sqrt;
                        } else {
                            l_sym[[i, j]] = -w[[i, j]] * d_i_inv_sqrt * d_j_inv_sqrt;
                        }
                    }
                }

                let (eigenvalues, eigenvectors) =
                    eigh(&l_sym.view(), None).map_err(TransformError::LinalgError)?;

                // Transform: f = D^{-1/2} * eigvecs
                let mut f_vecs = eigenvectors.clone();
                for i in 0..n {
                    let d_inv_sqrt = 1.0 / degrees[i].sqrt();
                    for j in 0..n {
                        f_vecs[[i, j]] *= d_inv_sqrt;
                    }
                }

                Ok((eigenvalues, f_vecs))
            }
        }
    }

    /// Fit the Laplacian Eigenmaps model
    ///
    /// # Arguments
    /// * `x` - Input data, shape (n_samples, n_features)
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let (n_samples, n_features) = x.dim();

        check_positive(self.n_components, "n_components")?;
        checkshape(x, &[n_samples, n_features], "x")?;

        if self.n_components >= n_samples {
            return Err(TransformError::InvalidInput(format!(
                "n_components={} must be < n_samples={}",
                self.n_components, n_samples
            )));
        }

        let x_f64 = x.mapv(|v| NumCast::from(v).unwrap_or(0.0));

        // Compute pairwise distances
        let dist_sq = Self::compute_sq_distances(&x_f64);

        // Determine sigma
        let sigma = self.sigma.unwrap_or_else(|| {
            let k = match &self.graph_method {
                GraphMethod::KNN { k, .. } => *k,
                _ => (n_samples as f64).sqrt().ceil() as usize,
            };
            Self::estimate_sigma(&dist_sq, k.min(n_samples - 1).max(1))
        });

        // Construct affinity matrix
        let w = self.construct_affinity(&dist_sq, sigma)?;

        // Compute degrees
        let degrees = Self::compute_degrees(&w);

        // Compute embedding
        let (eigenvalues, eigenvectors) = self.compute_embedding(&w, &degrees)?;

        // Sort by eigenvalue (ascending) and select the n_components smallest non-zero
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.sort_by(|&i, &j| {
            eigenvalues[i]
                .partial_cmp(&eigenvalues[j])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Skip the first eigenvector (constant, eigenvalue ~ 0)
        let mut embedding = Array2::zeros((n_samples, self.n_components));
        let mut selected_eigenvalues = Array1::zeros(self.n_components);

        for j in 0..self.n_components {
            let idx = indices[j + 1]; // Skip first (trivial) eigenvector
            selected_eigenvalues[j] = eigenvalues[idx];
            for i in 0..n_samples {
                embedding[[i, j]] = eigenvectors[[i, idx]];
            }
        }

        self.embedding = Some(embedding);
        self.training_data = Some(x_f64);
        self.affinity_matrix = Some(w);
        self.eigenvalues = Some(selected_eigenvalues);
        self.eigenvectors = Some(eigenvectors);
        self.degrees = Some(degrees);

        Ok(())
    }

    /// Transform data using the fitted model
    ///
    /// For the training data, returns the stored embedding.
    /// For new data, uses Nystrom approximation.
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let training_data = self
            .training_data
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Model not fitted".to_string()))?;

        let x_f64 = x.mapv(|v| NumCast::from(v).unwrap_or(0.0));

        if self.is_same_data(&x_f64, training_data) {
            return self
                .embedding
                .as_ref()
                .cloned()
                .ok_or_else(|| TransformError::NotFitted("Embedding not available".to_string()));
        }

        self.nystrom_extension(&x_f64)
    }

    /// Fit and transform in one step
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Nystrom out-of-sample extension
    ///
    /// Approximates the embedding of new points using the Nystrom method:
    /// For each new point, compute its affinity to the training points and
    /// project using the learned eigenvectors.
    fn nystrom_extension(&self, x_new: &Array2<f64>) -> Result<Array2<f64>> {
        let training_data = self
            .training_data
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Training data not available".to_string()))?;
        let training_embedding = self
            .embedding
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Embedding not available".to_string()))?;
        let eigenvalues = self
            .eigenvalues
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Eigenvalues not available".to_string()))?;
        let degrees = self
            .degrees
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Degrees not available".to_string()))?;

        let n_new = x_new.nrows();
        let n_train = training_data.nrows();
        let n_features = training_data.ncols();

        if x_new.ncols() != n_features {
            return Err(TransformError::InvalidInput(format!(
                "Input features {} must match training features {}",
                x_new.ncols(),
                n_features
            )));
        }

        // Determine sigma for affinity computation
        let sigma = self.sigma.unwrap_or(1.0);
        let sigma_sq = sigma * sigma;

        // Compute affinity between new points and training points
        let mut w_new = Array2::zeros((n_new, n_train));
        for i in 0..n_new {
            for j in 0..n_train {
                let mut dist_sq = 0.0;
                for k in 0..n_features {
                    let diff = x_new[[i, k]] - training_data[[j, k]];
                    dist_sq += diff * diff;
                }
                w_new[[i, j]] = (-dist_sq / (2.0 * sigma_sq)).exp();
            }
        }

        // Nystrom extension: f_new = D_new^{-1} W_new * embedding / eigenvalue
        let mut new_embedding = Array2::zeros((n_new, self.n_components));

        for i in 0..n_new {
            let d_new_i: f64 = w_new.row(i).sum();
            if d_new_i < 1e-15 {
                continue;
            }

            for j in 0..self.n_components {
                let eig_val = eigenvalues[j];
                if eig_val.abs() < 1e-15 {
                    continue;
                }

                let mut sum = 0.0;
                for k in 0..n_train {
                    // Normalized weight
                    let w_norm = w_new[[i, k]] / (d_new_i.sqrt() * degrees[k].sqrt());
                    sum += w_norm * training_embedding[[k, j]];
                }
                new_embedding[[i, j]] = sum / eig_val;
            }
        }

        Ok(new_embedding)
    }

    /// Check if two data matrices are the same
    fn is_same_data(&self, x: &Array2<f64>, training_data: &Array2<f64>) -> bool {
        if x.dim() != training_data.dim() {
            return false;
        }
        let (n, m) = x.dim();
        for i in 0..n {
            for j in 0..m {
                if (x[[i, j]] - training_data[[i, j]]).abs() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    fn make_swiss_roll(n: usize) -> Array2<f64> {
        let mut data = Vec::with_capacity(n * 3);
        for i in 0..n {
            let t = 1.5 * std::f64::consts::PI * (1.0 + 2.0 * i as f64 / n as f64);
            let x = t * t.cos();
            let y = 10.0 * i as f64 / n as f64;
            let z = t * t.sin();
            data.extend_from_slice(&[x, y, z]);
        }
        Array::from_shape_vec((n, 3), data).expect("Failed to create swiss roll")
    }

    #[test]
    fn test_laplacian_eigenmaps_knn() {
        let data = make_swiss_roll(30);
        let mut le = LaplacianEigenmaps::new(
            2,
            GraphMethod::KNN {
                k: 7,
                heat_kernel: true,
            },
        );
        let embedding = le.fit_transform(&data).expect("LE fit_transform failed");

        assert_eq!(embedding.shape(), &[30, 2]);
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_laplacian_eigenmaps_knn_binary() {
        let data = make_swiss_roll(25);
        let mut le = LaplacianEigenmaps::new(
            2,
            GraphMethod::KNN {
                k: 5,
                heat_kernel: false,
            },
        );
        let embedding = le.fit_transform(&data).expect("LE fit_transform failed");

        assert_eq!(embedding.shape(), &[25, 2]);
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_laplacian_eigenmaps_full_heat() {
        let data = make_swiss_roll(20);
        let mut le = LaplacianEigenmaps::new(2, GraphMethod::FullHeatKernel).with_sigma(5.0);
        let embedding = le.fit_transform(&data).expect("LE fit_transform failed");

        assert_eq!(embedding.shape(), &[20, 2]);
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_laplacian_eigenmaps_epsilon_ball() {
        // Create data where points are close enough for epsilon-ball
        let mut data_vec = Vec::new();
        for i in 0..20 {
            let t = i as f64 / 20.0;
            data_vec.extend_from_slice(&[t, t * 2.0, t * 3.0]);
        }
        let data = Array::from_shape_vec((20, 3), data_vec).expect("Failed");

        let mut le = LaplacianEigenmaps::new(2, GraphMethod::EpsilonBall { epsilon: 0.5 });
        let embedding = le.fit_transform(&data).expect("LE fit_transform failed");

        assert_eq!(embedding.shape(), &[20, 2]);
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_laplacian_eigenmaps_unnormalized() {
        let data = make_swiss_roll(25);
        let mut le = LaplacianEigenmaps::new(
            2,
            GraphMethod::KNN {
                k: 7,
                heat_kernel: true,
            },
        )
        .with_laplacian_type(LaplacianType::Unnormalized);
        let embedding = le.fit_transform(&data).expect("LE fit_transform failed");

        assert_eq!(embedding.shape(), &[25, 2]);
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_laplacian_eigenmaps_random_walk() {
        let data = make_swiss_roll(25);
        let mut le = LaplacianEigenmaps::new(
            2,
            GraphMethod::KNN {
                k: 7,
                heat_kernel: true,
            },
        )
        .with_laplacian_type(LaplacianType::NormalizedRandomWalk);
        let embedding = le.fit_transform(&data).expect("LE fit_transform failed");

        assert_eq!(embedding.shape(), &[25, 2]);
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_laplacian_eigenmaps_eigenvalues() {
        let data = make_swiss_roll(20);
        let mut le = LaplacianEigenmaps::new(
            3,
            GraphMethod::KNN {
                k: 5,
                heat_kernel: true,
            },
        );
        le.fit(&data).expect("LE fit failed");

        let eigenvalues = le.eigenvalues().expect("Eigenvalues should exist");
        assert_eq!(eigenvalues.len(), 3);

        // Eigenvalues should be non-negative (from Laplacian)
        for &ev in eigenvalues.iter() {
            assert!(ev >= -1e-10, "Eigenvalue should be >= 0, got {}", ev);
        }
    }

    #[test]
    fn test_laplacian_eigenmaps_out_of_sample() {
        let data = make_swiss_roll(30);
        let mut le = LaplacianEigenmaps::new(
            2,
            GraphMethod::KNN {
                k: 7,
                heat_kernel: true,
            },
        );
        le.fit(&data).expect("LE fit failed");

        let new_data =
            Array::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .expect("Failed");

        let new_embedding = le.transform(&new_data).expect("LE transform failed");
        assert_eq!(new_embedding.shape(), &[3, 2]);
        for val in new_embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_laplacian_eigenmaps_custom_sigma() {
        let data = make_swiss_roll(20);
        let mut le = LaplacianEigenmaps::new(
            2,
            GraphMethod::KNN {
                k: 5,
                heat_kernel: true,
            },
        )
        .with_sigma(2.0);
        let embedding = le.fit_transform(&data).expect("LE fit_transform failed");

        assert_eq!(embedding.shape(), &[20, 2]);
    }

    #[test]
    fn test_laplacian_eigenmaps_invalid_params() {
        let data = make_swiss_roll(5);

        // n_components >= n_samples
        let mut le = LaplacianEigenmaps::new(
            10,
            GraphMethod::KNN {
                k: 3,
                heat_kernel: true,
            },
        );
        assert!(le.fit(&data).is_err());
    }

    #[test]
    fn test_laplacian_eigenmaps_not_fitted() {
        let le = LaplacianEigenmaps::new(
            2,
            GraphMethod::KNN {
                k: 5,
                heat_kernel: true,
            },
        );
        let data = make_swiss_roll(10);
        assert!(le.transform(&data).is_err());
    }

    #[test]
    fn test_laplacian_eigenmaps_affinity_matrix() {
        let data = make_swiss_roll(15);
        let mut le = LaplacianEigenmaps::new(
            2,
            GraphMethod::KNN {
                k: 5,
                heat_kernel: true,
            },
        );
        le.fit(&data).expect("LE fit failed");

        let w = le.affinity_matrix().expect("Affinity should exist");
        assert_eq!(w.shape(), &[15, 15]);

        // Affinity should be symmetric
        for i in 0..15 {
            for j in 0..15 {
                assert!(
                    (w[[i, j]] - w[[j, i]]).abs() < 1e-10,
                    "Affinity not symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }

        // Diagonal should be zero (no self-loops)
        for i in 0..15 {
            assert!(w[[i, i]].abs() < 1e-10, "Diagonal should be zero");
        }
    }
}
