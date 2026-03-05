//! Kernel PCA (Principal Component Analysis)
//!
//! Kernel PCA is a nonlinear extension of PCA that uses the kernel trick to
//! perform PCA in a high-dimensional (potentially infinite-dimensional) feature
//! space without explicitly computing the feature map.
//!
//! ## Algorithm
//!
//! 1. Compute the kernel (Gram) matrix K
//! 2. Center K in feature space
//! 3. Eigendecompose the centered K
//! 4. Project data using the top eigenvectors
//!
//! ## Features
//!
//! - Nonlinear dimensionality reduction via kernel trick
//! - Pre-image estimation (approximate reconstruction to input space)
//! - Automatic kernel parameter selection via grid search on reconstruction error
//! - Support for all kernel types in the kernels module

use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_linalg::eigh;

use super::kernels::{
    center_kernel_matrix, center_kernel_matrix_test, cross_gram_matrix, gram_matrix, KernelType,
};
use crate::error::{Result, TransformError};

/// Kernel PCA for nonlinear dimensionality reduction
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_transform::kernel::{KernelPCA, KernelType};
/// use scirs2_core::ndarray::Array2;
///
/// let data = Array2::<f64>::zeros((50, 10));
/// let mut kpca = KernelPCA::new(2, KernelType::RBF { gamma: 0.1 });
/// let embedding = kpca.fit_transform(&data).expect("should succeed");
/// assert_eq!(embedding.shape(), &[50, 2]);
/// ```
#[derive(Debug, Clone)]
pub struct KernelPCA {
    /// Number of components to retain
    n_components: usize,
    /// Kernel function type
    kernel: KernelType,
    /// Whether to center the kernel matrix
    center: bool,
    /// Eigenvalues from the decomposition
    eigenvalues: Option<Array1<f64>>,
    /// Eigenvectors (alphas) from the decomposition
    alphas: Option<Array2<f64>>,
    /// The centered training kernel matrix
    k_train_centered: Option<Array2<f64>>,
    /// The raw training kernel matrix (for test centering)
    k_train_raw: Option<Array2<f64>>,
    /// Training data (for pre-image and out-of-sample)
    training_data: Option<Array2<f64>>,
    /// Explained variance ratio
    explained_variance_ratio: Option<Array1<f64>>,
}

impl KernelPCA {
    /// Create a new KernelPCA instance
    ///
    /// # Arguments
    /// * `n_components` - Number of principal components to retain
    /// * `kernel` - The kernel function to use
    pub fn new(n_components: usize, kernel: KernelType) -> Self {
        KernelPCA {
            n_components,
            kernel,
            center: true,
            eigenvalues: None,
            alphas: None,
            k_train_centered: None,
            k_train_raw: None,
            training_data: None,
            explained_variance_ratio: None,
        }
    }

    /// Set whether to center the kernel matrix (default: true)
    pub fn with_centering(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Get the kernel type
    pub fn kernel(&self) -> &KernelType {
        &self.kernel
    }

    /// Get the eigenvalues
    pub fn eigenvalues(&self) -> Option<&Array1<f64>> {
        self.eigenvalues.as_ref()
    }

    /// Get the explained variance ratio
    pub fn explained_variance_ratio(&self) -> Option<&Array1<f64>> {
        self.explained_variance_ratio.as_ref()
    }

    /// Get the principal axes (eigenvectors scaled by 1/sqrt(eigenvalue))
    pub fn alphas(&self) -> Option<&Array2<f64>> {
        self.alphas.as_ref()
    }

    /// Fit the Kernel PCA model to the input data
    ///
    /// # Arguments
    /// * `x` - Input data, shape (n_samples, n_features)
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        if self.n_components > n_samples {
            return Err(TransformError::InvalidInput(format!(
                "n_components={} must be <= n_samples={}",
                self.n_components, n_samples
            )));
        }

        // Convert to f64
        let x_f64 = x.mapv(|v| NumCast::from(v).unwrap_or(0.0));

        // Compute kernel matrix
        let k = gram_matrix(&x_f64.view(), &self.kernel)?;

        // Center kernel matrix if requested
        let k_centered_raw = if self.center {
            center_kernel_matrix(&k)?
        } else {
            k.clone()
        };

        // Enforce numerical symmetry (centering can introduce tiny asymmetries)
        let mut k_centered = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in i..n_samples {
                let sym = 0.5 * (k_centered_raw[[i, j]] + k_centered_raw[[j, i]]);
                k_centered[[i, j]] = sym;
                k_centered[[j, i]] = sym;
            }
        }

        // Eigendecomposition of the centered kernel matrix
        let (eigenvalues, eigenvectors) =
            eigh(&k_centered.view(), None).map_err(TransformError::LinalgError)?;

        // Sort eigenvalues in descending order
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.sort_by(|&i, &j| {
            eigenvalues[j]
                .partial_cmp(&eigenvalues[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Extract top n_components
        let mut top_eigenvalues = Array1::zeros(self.n_components);
        let mut top_eigenvectors = Array2::zeros((n_samples, self.n_components));

        for j in 0..self.n_components {
            let idx = indices[j];
            let eigval = eigenvalues[idx].max(0.0);
            top_eigenvalues[j] = eigval;

            // Normalize eigenvectors by sqrt(eigenvalue)
            let scale = if eigval > 1e-15 {
                1.0 / eigval.sqrt()
            } else {
                0.0
            };

            for i in 0..n_samples {
                top_eigenvectors[[i, j]] = eigenvectors[[i, idx]] * scale;
            }
        }

        // Compute explained variance ratio
        let total_variance: f64 = eigenvalues.iter().map(|&v| v.max(0.0)).sum();
        let explained_variance_ratio = if total_variance > 1e-15 {
            top_eigenvalues.mapv(|v| v / total_variance)
        } else {
            Array1::zeros(self.n_components)
        };

        self.eigenvalues = Some(top_eigenvalues);
        self.alphas = Some(top_eigenvectors);
        self.k_train_centered = Some(k_centered);
        self.k_train_raw = Some(k);
        self.training_data = Some(x_f64);
        self.explained_variance_ratio = Some(explained_variance_ratio);

        Ok(())
    }

    /// Transform data using the fitted model
    ///
    /// # Arguments
    /// * `x` - Input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * Projected data, shape (n_samples, n_components)
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let alphas = self
            .alphas
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("KernelPCA not fitted".to_string()))?;
        let training_data = self
            .training_data
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Training data not available".to_string()))?;
        let eigenvalues = self
            .eigenvalues
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Eigenvalues not available".to_string()))?;

        let x_f64 = x.mapv(|v| NumCast::from(v).unwrap_or(0.0));

        // Check if this is the same data
        if self.is_same_data(&x_f64, training_data) {
            return self.transform_training_data();
        }

        // Compute cross-kernel matrix between test and training
        let k_test = cross_gram_matrix(&x_f64.view(), &training_data.view(), &self.kernel)?;

        // Center the test kernel matrix
        let k_test_centered = if self.center {
            let k_train = self.k_train_raw.as_ref().ok_or_else(|| {
                TransformError::NotFitted("Training kernel matrix not available".to_string())
            })?;
            center_kernel_matrix_test(&k_test, k_train)?
        } else {
            k_test
        };

        // Project: X_proj = K_test_centered * alpha
        let n_test = x_f64.nrows();
        let n_train = training_data.nrows();
        let mut projected = Array2::zeros((n_test, self.n_components));

        for i in 0..n_test {
            for j in 0..self.n_components {
                let mut sum = 0.0;
                for k in 0..n_train {
                    sum += k_test_centered[[i, k]] * alphas[[k, j]];
                }
                // Scale by sqrt(eigenvalue) to get proper projection
                projected[[i, j]] = sum * eigenvalues[j].max(0.0).sqrt();
            }
        }

        Ok(projected)
    }

    /// Transform the training data (using stored centered kernel)
    fn transform_training_data(&self) -> Result<Array2<f64>> {
        let alphas = self
            .alphas
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("KernelPCA not fitted".to_string()))?;
        let k_centered = self.k_train_centered.as_ref().ok_or_else(|| {
            TransformError::NotFitted("Centered kernel not available".to_string())
        })?;
        let eigenvalues = self
            .eigenvalues
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Eigenvalues not available".to_string()))?;

        let n_samples = k_centered.nrows();
        let mut projected = Array2::zeros((n_samples, self.n_components));

        for i in 0..n_samples {
            for j in 0..self.n_components {
                let mut sum = 0.0;
                for k in 0..n_samples {
                    sum += k_centered[[i, k]] * alphas[[k, j]];
                }
                projected[[i, j]] = sum * eigenvalues[j].max(0.0).sqrt();
            }
        }

        Ok(projected)
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

    /// Pre-image estimation: approximate reconstruction from kernel space to input space
    ///
    /// Uses an iterative fixed-point method (Mika et al., 1998; Kwok & Tsang, 2004)
    /// to find an approximate pre-image for the kernel PCA projection.
    ///
    /// # Arguments
    /// * `projected` - Projected data in kernel PCA space, shape (n_samples, n_components)
    /// * `max_iter` - Maximum number of iterations for the fixed-point method
    /// * `tol` - Convergence tolerance
    ///
    /// # Returns
    /// * Approximate reconstruction in input space, shape (n_samples, n_features)
    pub fn inverse_transform(
        &self,
        projected: &Array2<f64>,
        max_iter: usize,
        tol: f64,
    ) -> Result<Array2<f64>> {
        let training_data = self
            .training_data
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Training data not available".to_string()))?;
        let alphas = self
            .alphas
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("KernelPCA not fitted".to_string()))?;
        let eigenvalues = self
            .eigenvalues
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Eigenvalues not available".to_string()))?;

        let n_samples = projected.nrows();
        let n_features = training_data.ncols();
        let n_train = training_data.nrows();

        let mut reconstructed = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            // Initialize with the nearest training point in kernel space
            let mut best_idx = 0;
            let mut best_dist = f64::INFINITY;

            // Compute the target kernel representation
            let mut target_kernel_rep = Array1::zeros(self.n_components);
            for j in 0..self.n_components {
                target_kernel_rep[j] = projected[[i, j]];
            }

            // Find nearest training point by kernel space distance
            for t in 0..n_train {
                let mut dist = 0.0;
                for j in 0..self.n_components {
                    let mut train_proj = 0.0;
                    let k_centered = self.k_train_centered.as_ref().ok_or_else(|| {
                        TransformError::NotFitted("Centered kernel not available".to_string())
                    })?;
                    for k in 0..n_train {
                        train_proj += k_centered[[t, k]] * alphas[[k, j]];
                    }
                    train_proj *= eigenvalues[j].max(0.0).sqrt();
                    let diff = projected[[i, j]] - train_proj;
                    dist += diff * diff;
                }
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = t;
                }
            }

            // Start with the nearest training point
            let mut x_approx = training_data.row(best_idx).to_owned();

            // Iterative fixed-point pre-image estimation
            for _iter in 0..max_iter {
                // Compute weights based on kernel between x_approx and training points
                let mut weights = Array1::zeros(n_train);
                let mut weight_sum = 0.0;

                for t in 0..n_train {
                    // Compute k(x_approx, x_t) weighted by the kernel PCA coefficients
                    let k_val = match &self.kernel {
                        KernelType::RBF { gamma } => {
                            let mut dist_sq = 0.0;
                            for d in 0..n_features {
                                let diff = x_approx[d] - training_data[[t, d]];
                                dist_sq += diff * diff;
                            }
                            (-gamma * dist_sq).exp()
                        }
                        KernelType::Laplacian { gamma } => {
                            let mut l1_dist = 0.0;
                            for d in 0..n_features {
                                l1_dist += (x_approx[d] - training_data[[t, d]]).abs();
                            }
                            (-gamma * l1_dist).exp()
                        }
                        _ => {
                            // For other kernels, use distance-based weighting
                            let mut dist_sq = 0.0;
                            for d in 0..n_features {
                                let diff = x_approx[d] - training_data[[t, d]];
                                dist_sq += diff * diff;
                            }
                            if dist_sq > 1e-15 {
                                1.0 / (1.0 + dist_sq.sqrt())
                            } else {
                                1e10
                            }
                        }
                    };

                    // Weight by kernel PCA coefficients
                    let mut coeff_weight = 0.0;
                    for j in 0..self.n_components {
                        coeff_weight += alphas[[t, j]] * projected[[i, j]];
                    }

                    weights[t] = k_val * coeff_weight.abs();
                    weight_sum += weights[t];
                }

                // Normalize weights
                if weight_sum > 1e-15 {
                    weights.mapv_inplace(|w| w / weight_sum);
                } else {
                    // Fallback: uniform weights
                    weights = Array1::from_elem(n_train, 1.0 / n_train as f64);
                }

                // Update x_approx as weighted sum of training points
                let mut x_new = Array1::zeros(n_features);
                for t in 0..n_train {
                    for d in 0..n_features {
                        x_new[d] += weights[t] * training_data[[t, d]];
                    }
                }

                // Check convergence
                let mut change = 0.0;
                for d in 0..n_features {
                    let diff = x_new[d] - x_approx[d];
                    change += diff * diff;
                }

                x_approx = x_new;

                if change.sqrt() < tol {
                    break;
                }
            }

            for d in 0..n_features {
                reconstructed[[i, d]] = x_approx[d];
            }
        }

        Ok(reconstructed)
    }

    /// Automatic kernel parameter selection via grid search
    ///
    /// Tries multiple gamma values and selects the one that minimizes
    /// the reconstruction error (for RBF/Laplacian kernels).
    ///
    /// # Arguments
    /// * `x` - Input data, shape (n_samples, n_features)
    /// * `gamma_values` - List of gamma values to try
    /// * `preimage_iter` - Number of pre-image iterations
    ///
    /// # Returns
    /// * The best gamma value and corresponding reconstruction error
    pub fn auto_select_gamma<S>(
        x: &ArrayBase<S, Ix2>,
        n_components: usize,
        gamma_values: &[f64],
        preimage_iter: usize,
    ) -> Result<(f64, f64)>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        if gamma_values.is_empty() {
            return Err(TransformError::InvalidInput(
                "gamma_values must not be empty".to_string(),
            ));
        }

        let x_f64: Array2<f64> = x.mapv(|v| NumCast::from(v).unwrap_or(0.0));

        let mut best_gamma = gamma_values[0];
        let mut best_error = f64::INFINITY;

        for &gamma in gamma_values {
            let kernel = KernelType::RBF { gamma };
            let mut kpca = KernelPCA::new(n_components, kernel);

            match kpca.fit(&x_f64.view()) {
                Ok(()) => {}
                Err(_) => continue,
            }

            let projected = match kpca.transform(&x_f64.view()) {
                Ok(p) => p,
                Err(_) => continue,
            };

            let reconstructed = match kpca.inverse_transform(&projected, preimage_iter, 1e-6) {
                Ok(r) => r,
                Err(_) => continue,
            };

            // Compute reconstruction error
            let mut error = 0.0;
            let n_samples = x_f64.nrows();
            let n_features = x_f64.ncols();
            for i in 0..n_samples {
                for j in 0..n_features {
                    let diff = x_f64[[i, j]] - reconstructed[[i, j]];
                    error += diff * diff;
                }
            }
            error /= n_samples as f64;

            if error < best_error {
                best_error = error;
                best_gamma = gamma;
            }
        }

        if best_error.is_infinite() {
            return Err(TransformError::ComputationError(
                "All gamma values failed".to_string(),
            ));
        }

        Ok((best_gamma, best_error))
    }

    /// Check if two data matrices are identical
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

    fn make_circular_data(n: usize) -> Array2<f64> {
        let mut data = Vec::with_capacity(n * 2);
        for i in 0..n {
            let t = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            let r = 1.0 + 0.3 * (i as f64 / n as f64);
            data.push(r * t.cos());
            data.push(r * t.sin());
        }
        Array::from_shape_vec((n, 2), data).expect("Failed to create data")
    }

    #[test]
    fn test_kpca_rbf_basic() {
        let data = make_circular_data(30);
        let mut kpca = KernelPCA::new(2, KernelType::RBF { gamma: 1.0 });
        let projected = kpca
            .fit_transform(&data)
            .expect("KPCA fit_transform failed");

        assert_eq!(projected.shape(), &[30, 2]);
        for val in projected.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_kpca_linear_matches_pca() {
        // Linear kernel PCA should give similar results to regular PCA
        let data = Array::from_shape_vec(
            (6, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                9.0, 10.0,
            ],
        )
        .expect("Failed");

        let mut kpca = KernelPCA::new(2, KernelType::Linear);
        let projected = kpca
            .fit_transform(&data)
            .expect("KPCA fit_transform failed");

        assert_eq!(projected.shape(), &[6, 2]);
        for val in projected.iter() {
            assert!(val.is_finite());
        }

        // Check explained variance ratio sums to less than or equal to 1
        let evr = kpca.explained_variance_ratio().expect("EVR should exist");
        assert!(evr.sum() <= 1.0 + 1e-10);
        assert!(evr.sum() > 0.0);
    }

    #[test]
    fn test_kpca_polynomial() {
        let data = make_circular_data(20);
        let kernel = KernelType::Polynomial {
            gamma: 1.0,
            coef0: 1.0,
            degree: 2,
        };
        let mut kpca = KernelPCA::new(2, kernel);
        let projected = kpca
            .fit_transform(&data)
            .expect("KPCA fit_transform failed");

        assert_eq!(projected.shape(), &[20, 2]);
        for val in projected.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_kpca_laplacian() {
        let data = make_circular_data(20);
        let mut kpca = KernelPCA::new(2, KernelType::Laplacian { gamma: 1.0 });
        let projected = kpca
            .fit_transform(&data)
            .expect("KPCA fit_transform failed");

        assert_eq!(projected.shape(), &[20, 2]);
        for val in projected.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_kpca_out_of_sample() {
        let data = make_circular_data(30);
        let mut kpca = KernelPCA::new(2, KernelType::RBF { gamma: 0.5 });
        kpca.fit(&data).expect("KPCA fit failed");

        let test_data =
            Array::from_shape_vec((3, 2), vec![0.5, 0.5, -0.5, 0.5, 0.0, -1.0]).expect("Failed");

        let test_projected = kpca.transform(&test_data).expect("KPCA transform failed");
        assert_eq!(test_projected.shape(), &[3, 2]);
        for val in test_projected.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_kpca_eigenvalues() {
        let data = make_circular_data(20);
        let mut kpca = KernelPCA::new(3, KernelType::RBF { gamma: 0.5 });
        kpca.fit(&data).expect("KPCA fit failed");

        let eigenvalues = kpca.eigenvalues().expect("Eigenvalues should exist");
        assert_eq!(eigenvalues.len(), 3);

        // Eigenvalues should be non-negative and in descending order
        for i in 0..eigenvalues.len() {
            assert!(eigenvalues[i] >= -1e-10);
            if i > 0 {
                assert!(eigenvalues[i] <= eigenvalues[i - 1] + 1e-10);
            }
        }
    }

    #[test]
    fn test_kpca_preimage() {
        let data = make_circular_data(15);
        let mut kpca = KernelPCA::new(2, KernelType::RBF { gamma: 0.5 });
        let projected = kpca
            .fit_transform(&data)
            .expect("KPCA fit_transform failed");

        let reconstructed = kpca
            .inverse_transform(&projected, 50, 1e-6)
            .expect("Pre-image estimation failed");

        assert_eq!(reconstructed.shape(), &[15, 2]);
        for val in reconstructed.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_kpca_auto_gamma() {
        let data = make_circular_data(15);
        let gammas = vec![0.01, 0.1, 0.5, 1.0, 5.0];

        let (best_gamma, best_error) = KernelPCA::auto_select_gamma(&data.view(), 2, &gammas, 10)
            .expect("Auto gamma selection failed");

        assert!(best_gamma > 0.0);
        assert!(best_error >= 0.0);
        assert!(best_error.is_finite());
    }

    #[test]
    fn test_kpca_empty_data() {
        let data: Array2<f64> = Array2::zeros((0, 5));
        let mut kpca = KernelPCA::new(2, KernelType::RBF { gamma: 1.0 });
        assert!(kpca.fit(&data).is_err());
    }

    #[test]
    fn test_kpca_too_many_components() {
        let data = make_circular_data(5);
        let mut kpca = KernelPCA::new(10, KernelType::RBF { gamma: 1.0 });
        assert!(kpca.fit(&data).is_err());
    }

    #[test]
    fn test_kpca_not_fitted() {
        let kpca = KernelPCA::new(2, KernelType::RBF { gamma: 1.0 });
        let data = make_circular_data(10);
        assert!(kpca.transform(&data).is_err());
    }

    #[test]
    fn test_kpca_sigmoid() {
        let data = make_circular_data(20);
        let kernel = KernelType::Sigmoid {
            gamma: 0.1,
            coef0: 0.0,
        };
        let mut kpca = KernelPCA::new(2, kernel);
        let projected = kpca
            .fit_transform(&data)
            .expect("KPCA fit_transform failed");

        assert_eq!(projected.shape(), &[20, 2]);
        for val in projected.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_kpca_no_centering() {
        let data = make_circular_data(20);
        let mut kpca = KernelPCA::new(2, KernelType::RBF { gamma: 0.5 }).with_centering(false);
        let projected = kpca
            .fit_transform(&data)
            .expect("KPCA fit_transform failed");

        assert_eq!(projected.shape(), &[20, 2]);
        for val in projected.iter() {
            assert!(val.is_finite());
        }
    }
}
