//! Diffusion Maps for Nonlinear Dimensionality Reduction
//!
//! Diffusion Maps (Coifman & Lafon, 2006) embed data points based on the
//! connectivity of the underlying manifold, using a diffusion process on
//! a graph constructed from the data.
//!
//! ## Algorithm
//!
//! 1. Construct an anisotropic kernel from the data
//! 2. Normalize to form a Markov chain transition matrix
//! 3. Eigendecompose the transition matrix
//! 4. Embed using eigenvectors scaled by eigenvalues^t (diffusion time)
//!
//! ## Features
//!
//! - Anisotropic diffusion kernel (alpha parameter controls density normalization)
//! - Multi-scale analysis via diffusion time parameter
//! - Automatic dimensionality selection via spectral gap analysis
//! - Out-of-sample extension via Nystrom approximation

use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::validation::{check_positive, checkshape};
use scirs2_linalg::eigh;

use crate::error::{Result, TransformError};

/// Diffusion Maps for nonlinear dimensionality reduction
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_transform::reduction::diffusion_maps::DiffusionMaps;
/// use scirs2_core::ndarray::Array2;
///
/// let data = Array2::<f64>::zeros((50, 10));
/// let mut dm = DiffusionMaps::new(2);
/// let embedding = dm.fit_transform(&data).expect("should succeed");
/// assert_eq!(embedding.shape(), &[50, 2]);
/// ```
#[derive(Debug, Clone)]
pub struct DiffusionMaps {
    /// Number of components in the embedding
    n_components: usize,
    /// Kernel bandwidth parameter (epsilon). Auto-selected if None.
    epsilon: Option<f64>,
    /// Anisotropy parameter (alpha). Controls density normalization.
    /// alpha = 0: classical normalization (Laplacian Eigenmaps)
    /// alpha = 0.5: Fokker-Planck normalization
    /// alpha = 1: Laplace-Beltrami operator (geometry only)
    alpha: f64,
    /// Diffusion time parameter (t). Controls the scale of the diffusion.
    diffusion_time: f64,
    /// The embedding vectors
    embedding: Option<Array2<f64>>,
    /// Training data
    training_data: Option<Array2<f64>>,
    /// Eigenvalues of the diffusion operator
    eigenvalues: Option<Array1<f64>>,
    /// Eigenvectors of the diffusion operator
    eigenvectors: Option<Array2<f64>>,
    /// Kernel bandwidth used
    epsilon_used: Option<f64>,
    /// Spectral gap ratios for dimensionality selection
    spectral_gaps: Option<Array1<f64>>,
    /// Transition matrix row sums (for Nystrom)
    row_sums: Option<Array1<f64>>,
}

impl DiffusionMaps {
    /// Create a new DiffusionMaps instance
    ///
    /// # Arguments
    /// * `n_components` - Number of dimensions in the embedding
    pub fn new(n_components: usize) -> Self {
        DiffusionMaps {
            n_components,
            epsilon: None,
            alpha: 0.5,
            diffusion_time: 1.0,
            embedding: None,
            training_data: None,
            eigenvalues: None,
            eigenvectors: None,
            epsilon_used: None,
            spectral_gaps: None,
            row_sums: None,
        }
    }

    /// Set the kernel bandwidth (epsilon)
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = Some(epsilon);
        self
    }

    /// Set the anisotropy parameter (alpha)
    ///
    /// - alpha = 0: graph Laplacian normalization
    /// - alpha = 0.5: Fokker-Planck diffusion
    /// - alpha = 1.0: Laplace-Beltrami (geometry-only, density-independent)
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the diffusion time (t)
    ///
    /// Larger t emphasizes larger-scale structure; smaller t captures local detail.
    pub fn with_diffusion_time(mut self, t: f64) -> Self {
        self.diffusion_time = t;
        self
    }

    /// Get the embedding
    pub fn embedding(&self) -> Option<&Array2<f64>> {
        self.embedding.as_ref()
    }

    /// Get the eigenvalues of the diffusion operator
    pub fn eigenvalues(&self) -> Option<&Array1<f64>> {
        self.eigenvalues.as_ref()
    }

    /// Get the spectral gaps (ratio of consecutive eigenvalues)
    pub fn spectral_gaps(&self) -> Option<&Array1<f64>> {
        self.spectral_gaps.as_ref()
    }

    /// Get the epsilon (bandwidth) used
    pub fn epsilon_used(&self) -> Option<f64> {
        self.epsilon_used
    }

    /// Automatic dimensionality selection via spectral gap
    ///
    /// Finds the largest spectral gap (ratio lambda_{i} / lambda_{i+1})
    /// which indicates the intrinsic dimensionality of the data.
    ///
    /// # Arguments
    /// * `max_dim` - Maximum dimensionality to consider
    ///
    /// # Returns
    /// * Suggested number of dimensions
    pub fn suggest_dimensionality(&self, max_dim: usize) -> Result<usize> {
        let eigenvalues = self
            .eigenvalues
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Model not fitted".to_string()))?;

        let n_eigs = eigenvalues.len().min(max_dim);
        if n_eigs < 2 {
            return Ok(1);
        }

        let mut max_gap = 0.0;
        let mut best_dim = 1;

        for i in 0..(n_eigs - 1) {
            let current = eigenvalues[i].abs();
            let next = eigenvalues[i + 1].abs();

            if next > 1e-15 {
                let gap = current / next;
                if gap > max_gap {
                    max_gap = gap;
                    best_dim = i + 1;
                }
            } else {
                // Eigenvalue drops to essentially zero
                best_dim = i + 1;
                break;
            }
        }

        Ok(best_dim.max(1))
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

    /// Estimate epsilon using the median heuristic
    fn estimate_epsilon(dist_sq: &Array2<f64>) -> f64 {
        let n = dist_sq.nrows();
        let mut all_dists: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);

        for i in 0..n {
            for j in (i + 1)..n {
                all_dists.push(dist_sq[[i, j]]);
            }
        }

        all_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = all_dists[all_dists.len() / 2];
        if median < 1e-15 {
            1.0
        } else {
            median
        }
    }

    /// Construct the anisotropic diffusion kernel
    ///
    /// Step 1: Compute the isotropic kernel K(x,y) = exp(-||x-y||^2 / epsilon)
    /// Step 2: Normalize by density: K^(alpha)(x,y) = K(x,y) / (p(x)*p(y))^alpha
    ///         where p(x) = sum_y K(x,y) is a density estimate
    /// Step 3: Normalize rows to form Markov transition matrix
    fn construct_diffusion_operator(
        &self,
        dist_sq: &Array2<f64>,
        epsilon: f64,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        let n = dist_sq.nrows();

        // Step 1: Isotropic Gaussian kernel
        let mut k = Array2::zeros((n, n));
        for i in 0..n {
            k[[i, i]] = 1.0; // self-similarity
            for j in (i + 1)..n {
                let val = (-dist_sq[[i, j]] / epsilon).exp();
                k[[i, j]] = val;
                k[[j, i]] = val;
            }
        }

        // Step 2: Anisotropic normalization
        // Compute density estimate: q(x) = sum_y K(x, y)
        let mut q = Array1::zeros(n);
        for i in 0..n {
            q[i] = k.row(i).sum();
        }

        // Check for zero densities
        for i in 0..n {
            if q[i] < 1e-15 {
                return Err(TransformError::ComputationError(format!(
                    "Point {} has zero density. Increase epsilon.",
                    i
                )));
            }
        }

        // Normalize: K_alpha(x,y) = K(x,y) / (q(x) * q(y))^alpha
        if self.alpha.abs() > 1e-15 {
            for i in 0..n {
                let qi_alpha = q[i].powf(self.alpha);
                for j in 0..n {
                    let qj_alpha = q[j].powf(self.alpha);
                    k[[i, j]] /= qi_alpha * qj_alpha;
                }
            }
        }

        // Step 3: Row-normalize to get the Markov transition matrix
        let mut row_sums = Array1::zeros(n);
        for i in 0..n {
            row_sums[i] = k.row(i).sum();
        }

        // Instead of normalizing K directly, we work with the symmetric form:
        // D^{-1/2} K D^{-1/2} which has the same eigenvalues
        // and eigenvectors related by phi = D^{-1/2} psi
        let mut k_sym = Array2::zeros((n, n));
        for i in 0..n {
            let d_i_inv_sqrt = 1.0 / row_sums[i].sqrt();
            for j in 0..n {
                let d_j_inv_sqrt = 1.0 / row_sums[j].sqrt();
                k_sym[[i, j]] = k[[i, j]] * d_i_inv_sqrt * d_j_inv_sqrt;
            }
        }

        Ok((k_sym, row_sums))
    }

    /// Fit the Diffusion Maps model
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

        if self.diffusion_time < 0.0 {
            return Err(TransformError::InvalidInput(
                "diffusion_time must be non-negative".to_string(),
            ));
        }

        let x_f64 = x.mapv(|v| NumCast::from(v).unwrap_or(0.0));

        // Compute pairwise squared distances
        let dist_sq = Self::compute_sq_distances(&x_f64);

        // Determine epsilon
        let epsilon = self
            .epsilon
            .unwrap_or_else(|| Self::estimate_epsilon(&dist_sq));

        // Construct the symmetric diffusion operator
        let (k_sym, row_sums) = self.construct_diffusion_operator(&dist_sq, epsilon)?;

        // Eigendecomposition
        let (eigenvalues, eigenvectors) =
            eigh(&k_sym.view(), None).map_err(TransformError::LinalgError)?;

        // Sort eigenvalues in descending order
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.sort_by(|&i, &j| {
            eigenvalues[j]
                .partial_cmp(&eigenvalues[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // The first eigenvalue/eigenvector is trivial (constant, eigenvalue = 1)
        // We skip it and take the next n_components
        let n_eigs = self.n_components.min(n_samples - 1);
        let mut selected_eigenvalues = Array1::zeros(n_eigs);
        let mut selected_eigenvectors = Array2::zeros((n_samples, n_eigs));

        for j in 0..n_eigs {
            let idx = indices[j + 1]; // Skip first (trivial) eigenvector
            selected_eigenvalues[j] = eigenvalues[idx].max(0.0);

            // Transform back from symmetric form: phi = D^{-1/2} psi
            for i in 0..n_samples {
                let d_inv_sqrt = 1.0 / row_sums[i].sqrt();
                selected_eigenvectors[[i, j]] = eigenvectors[[i, idx]] * d_inv_sqrt;
            }
        }

        // Compute diffusion coordinates: Psi_t(x) = lambda^t * phi(x)
        let mut embedding = Array2::zeros((n_samples, n_eigs));
        for i in 0..n_samples {
            for j in 0..n_eigs {
                let scale = selected_eigenvalues[j].powf(self.diffusion_time);
                embedding[[i, j]] = scale * selected_eigenvectors[[i, j]];
            }
        }

        // Compute spectral gaps
        let mut spectral_gaps = Array1::zeros(n_eigs.saturating_sub(1));
        for j in 0..(n_eigs.saturating_sub(1)) {
            let next = selected_eigenvalues[j + 1].abs();
            if next > 1e-15 {
                spectral_gaps[j] = selected_eigenvalues[j] / next;
            } else {
                spectral_gaps[j] = f64::INFINITY;
            }
        }

        self.embedding = Some(embedding);
        self.training_data = Some(x_f64);
        self.eigenvalues = Some(selected_eigenvalues);
        self.eigenvectors = Some(selected_eigenvectors);
        self.epsilon_used = Some(epsilon);
        self.spectral_gaps = Some(spectral_gaps);
        self.row_sums = Some(row_sums);

        Ok(())
    }

    /// Transform data using the fitted model
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

    /// Nystrom out-of-sample extension for diffusion maps
    fn nystrom_extension(&self, x_new: &Array2<f64>) -> Result<Array2<f64>> {
        let training_data = self
            .training_data
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Training data not available".to_string()))?;
        let eigenvectors = self
            .eigenvectors
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Eigenvectors not available".to_string()))?;
        let eigenvalues = self
            .eigenvalues
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Eigenvalues not available".to_string()))?;
        let row_sums = self
            .row_sums
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Row sums not available".to_string()))?;

        let epsilon = self
            .epsilon_used
            .ok_or_else(|| TransformError::NotFitted("Epsilon not available".to_string()))?;

        let n_new = x_new.nrows();
        let n_train = training_data.nrows();
        let n_features = training_data.ncols();
        let n_eigs = eigenvalues.len();

        if x_new.ncols() != n_features {
            return Err(TransformError::InvalidInput(format!(
                "Input features {} must match training features {}",
                x_new.ncols(),
                n_features
            )));
        }

        // Compute kernel between new and training points
        let mut k_new = Array2::zeros((n_new, n_train));
        for i in 0..n_new {
            for j in 0..n_train {
                let mut dist_sq = 0.0;
                for d in 0..n_features {
                    let diff = x_new[[i, d]] - training_data[[j, d]];
                    dist_sq += diff * diff;
                }
                k_new[[i, j]] = (-dist_sq / epsilon).exp();
            }
        }

        // Apply anisotropic normalization if needed
        if self.alpha.abs() > 1e-15 {
            // We need the density estimate for new points
            let mut q_new = Array1::zeros(n_new);
            for i in 0..n_new {
                q_new[i] = k_new.row(i).sum();
            }

            for i in 0..n_new {
                let qi_alpha = if q_new[i] > 1e-15 {
                    q_new[i].powf(self.alpha)
                } else {
                    1.0
                };
                for j in 0..n_train {
                    let qj_alpha = row_sums[j].powf(self.alpha);
                    k_new[[i, j]] /= qi_alpha * qj_alpha;
                }
            }
        }

        // Normalize rows
        let mut new_row_sums = Array1::zeros(n_new);
        for i in 0..n_new {
            new_row_sums[i] = k_new.row(i).sum();
        }

        // Nystrom extension: phi_new = (1/lambda) * sum_j (K_new(i,j) / d_new(i)) * phi_train(j)
        let mut new_embedding = Array2::zeros((n_new, n_eigs));
        for i in 0..n_new {
            let d_new = new_row_sums[i];
            if d_new < 1e-15 {
                continue;
            }

            for c in 0..n_eigs {
                let lambda = eigenvalues[c];
                if lambda.abs() < 1e-15 {
                    continue;
                }

                let mut sum = 0.0;
                for j in 0..n_train {
                    let p_ij = k_new[[i, j]] / d_new;
                    sum += p_ij * eigenvectors[[j, c]];
                }

                // Apply diffusion time scaling
                let scale = lambda.powf(self.diffusion_time);
                new_embedding[[i, c]] = scale * sum / lambda;
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

    fn make_manifold_data(n: usize) -> Array2<f64> {
        let mut data = Vec::with_capacity(n * 3);
        for i in 0..n {
            let t = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            let r = 2.0 + 0.5 * (3.0 * t).sin();
            data.push(r * t.cos());
            data.push(r * t.sin());
            data.push(0.5 * t);
        }
        Array::from_shape_vec((n, 3), data).expect("Failed to create data")
    }

    #[test]
    fn test_diffusion_maps_basic() {
        let data = make_manifold_data(30);
        let mut dm = DiffusionMaps::new(2);
        let embedding = dm.fit_transform(&data).expect("DM fit_transform failed");

        assert_eq!(embedding.shape(), &[30, 2]);
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_diffusion_maps_custom_epsilon() {
        let data = make_manifold_data(25);
        let mut dm = DiffusionMaps::new(2).with_epsilon(2.0);
        let embedding = dm.fit_transform(&data).expect("DM fit_transform failed");

        assert_eq!(embedding.shape(), &[25, 2]);
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_diffusion_maps_alpha_zero() {
        // alpha=0 corresponds to graph Laplacian normalization
        let data = make_manifold_data(20);
        let mut dm = DiffusionMaps::new(2).with_alpha(0.0);
        let embedding = dm.fit_transform(&data).expect("DM fit_transform failed");

        assert_eq!(embedding.shape(), &[20, 2]);
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_diffusion_maps_alpha_one() {
        // alpha=1 corresponds to Laplace-Beltrami operator
        let data = make_manifold_data(20);
        let mut dm = DiffusionMaps::new(2).with_alpha(1.0);
        let embedding = dm.fit_transform(&data).expect("DM fit_transform failed");

        assert_eq!(embedding.shape(), &[20, 2]);
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_diffusion_maps_large_time() {
        let data = make_manifold_data(20);
        let mut dm = DiffusionMaps::new(2).with_diffusion_time(5.0);
        let embedding = dm.fit_transform(&data).expect("DM fit_transform failed");

        assert_eq!(embedding.shape(), &[20, 2]);
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_diffusion_maps_eigenvalues() {
        let data = make_manifold_data(25);
        let mut dm = DiffusionMaps::new(5);
        dm.fit(&data).expect("DM fit failed");

        let eigenvalues = dm.eigenvalues().expect("Eigenvalues should exist");
        assert_eq!(eigenvalues.len(), 5);

        // Eigenvalues should be positive and decreasing
        for i in 0..eigenvalues.len() {
            assert!(
                eigenvalues[i] >= -1e-10,
                "Eigenvalue {} should be >= 0, got {}",
                i,
                eigenvalues[i]
            );
            if i > 0 {
                assert!(
                    eigenvalues[i] <= eigenvalues[i - 1] + 1e-10,
                    "Eigenvalues should be decreasing"
                );
            }
        }
    }

    #[test]
    fn test_diffusion_maps_spectral_gaps() {
        let data = make_manifold_data(25);
        let mut dm = DiffusionMaps::new(5);
        dm.fit(&data).expect("DM fit failed");

        let gaps = dm.spectral_gaps().expect("Spectral gaps should exist");
        assert_eq!(gaps.len(), 4);

        // Gaps should be positive
        for &gap in gaps.iter() {
            if gap.is_finite() {
                assert!(gap > 0.0, "Spectral gap should be positive, got {}", gap);
            }
        }
    }

    #[test]
    fn test_diffusion_maps_suggest_dimensionality() {
        let data = make_manifold_data(25);
        let mut dm = DiffusionMaps::new(10);
        dm.fit(&data).expect("DM fit failed");

        let suggested = dm.suggest_dimensionality(10).expect("Suggestion failed");
        assert!(suggested >= 1);
        assert!(suggested <= 10);
    }

    #[test]
    fn test_diffusion_maps_out_of_sample() {
        let data = make_manifold_data(30);
        let mut dm = DiffusionMaps::new(2);
        dm.fit(&data).expect("DM fit failed");

        let new_data =
            Array::from_shape_vec((3, 3), vec![1.0, 2.0, 0.5, -1.0, 2.0, 1.0, 0.0, -2.0, 1.5])
                .expect("Failed");

        let new_embedding = dm.transform(&new_data).expect("DM transform failed");
        assert_eq!(new_embedding.shape(), &[3, 2]);
        for val in new_embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_diffusion_maps_epsilon_used() {
        let data = make_manifold_data(20);
        let mut dm = DiffusionMaps::new(2);
        dm.fit(&data).expect("DM fit failed");

        let eps = dm.epsilon_used().expect("Epsilon should be set");
        assert!(eps > 0.0);
        assert!(eps.is_finite());
    }

    #[test]
    fn test_diffusion_maps_invalid_params() {
        let data = make_manifold_data(5);

        let mut dm = DiffusionMaps::new(10);
        assert!(dm.fit(&data).is_err());
    }

    #[test]
    fn test_diffusion_maps_not_fitted() {
        let dm = DiffusionMaps::new(2);
        let data = make_manifold_data(10);
        assert!(dm.transform(&data).is_err());
    }

    #[test]
    fn test_diffusion_maps_linear_data() {
        // For purely linear data, DM should embed well in 1D
        let mut data_vec = Vec::new();
        for i in 0..20 {
            let t = i as f64 / 20.0;
            data_vec.extend_from_slice(&[t, 2.0 * t, 3.0 * t]);
        }
        let data = Array::from_shape_vec((20, 3), data_vec).expect("Failed");

        let mut dm = DiffusionMaps::new(3);
        let embedding = dm.fit_transform(&data).expect("DM fit_transform failed");

        assert_eq!(embedding.shape(), &[20, 3]);
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_diffusion_maps_multi_scale() {
        // Compare embeddings at different diffusion times
        let data = make_manifold_data(25);

        let mut dm1 = DiffusionMaps::new(2).with_diffusion_time(0.5);
        let emb1 = dm1.fit_transform(&data).expect("DM t=0.5 failed");

        let mut dm2 = DiffusionMaps::new(2).with_diffusion_time(5.0);
        let emb2 = dm2.fit_transform(&data).expect("DM t=5.0 failed");

        assert_eq!(emb1.shape(), &[25, 2]);
        assert_eq!(emb2.shape(), &[25, 2]);

        // The embeddings should be different due to different time scales
        let mut any_diff = false;
        for i in 0..25 {
            for j in 0..2 {
                if (emb1[[i, j]] - emb2[[i, j]]).abs() > 1e-10 {
                    any_diff = true;
                    break;
                }
            }
            if any_diff {
                break;
            }
        }
        assert!(
            any_diff,
            "Different diffusion times should give different embeddings"
        );
    }
}
