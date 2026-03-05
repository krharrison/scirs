//! Linear and Quadratic Discriminant Analysis
//!
//! Provides Fisher's Linear Discriminant Analysis (LDA) for dimensionality reduction
//! and classification, as well as Quadratic Discriminant Analysis (QDA) which relaxes
//! the shared covariance assumption.
//!
//! # Algorithms
//!
//! - **Fisher's LDA**: Finds directions that maximize between-class to within-class
//!   variance ratio. Supports SVD and eigenvalue-based solvers.
//! - **Regularized LDA**: Adds Tikhonov regularization to within-class scatter for
//!   ill-conditioned problems.
//! - **Multi-class LDA**: Extends to K classes with up to K-1 discriminant directions.
//! - **QDA**: Per-class covariance matrices for non-linear decision boundaries.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_linalg::svd;
use std::collections::HashMap;

use crate::error::{Result, TransformError};

const EPSILON: f64 = 1e-10;

/// Solver method for LDA
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LdaSolver {
    /// SVD-based solver (recommended for numerical stability)
    Svd,
    /// Eigenvalue-based solver
    Eigen,
}

/// Linear Discriminant Analysis (LDA) for dimensionality reduction
///
/// Finds projection directions that maximize between-class separation
/// relative to within-class scatter. The maximum number of discriminant
/// directions is min(n_classes - 1, n_features).
///
/// # Examples
///
/// ```
/// use scirs2_transform::reduction::lda::{LinearDiscriminantAnalysis, LdaSolver};
/// use scirs2_core::ndarray::{Array1, Array2};
///
/// let x = Array2::from_shape_vec(
///     (6, 3),
///     vec![1.0, 2.0, 0.5,  1.1, 2.1, 0.6,  0.9, 1.9, 0.4,
///          5.0, 4.0, 3.5,  5.1, 4.1, 3.6,  4.9, 3.9, 3.4],
/// ).expect("should succeed");
/// let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);
///
/// let mut lda = LinearDiscriminantAnalysis::new(1, LdaSolver::Svd);
/// lda.fit(&x, &y).expect("should succeed");
/// let projected = lda.transform(&x).expect("should succeed");
/// assert_eq!(projected.shape(), &[6, 1]);
/// ```
#[derive(Debug, Clone)]
pub struct LinearDiscriminantAnalysis {
    /// Number of components to keep
    n_components: usize,
    /// Solver method
    solver: LdaSolver,
    /// Regularization parameter (0 = no regularization)
    reg_param: f64,
    /// Learned projection matrix, shape (n_components, n_features)
    components_: Option<Array2<f64>>,
    /// Class means, shape (n_classes, n_features)
    class_means_: Option<Array2<f64>>,
    /// Global mean, shape (n_features,)
    global_mean_: Option<Array1<f64>>,
    /// Explained variance ratio
    explained_variance_ratio_: Option<Array1<f64>>,
    /// Class priors
    priors_: Option<Array1<f64>>,
    /// Number of features at fit time
    n_features_in_: Option<usize>,
    /// Unique class labels
    classes_: Option<Vec<i64>>,
}

impl LinearDiscriminantAnalysis {
    /// Create a new LDA
    ///
    /// # Arguments
    /// * `n_components` - Number of discriminant directions to keep
    /// * `solver` - Solver method (Svd or Eigen)
    pub fn new(n_components: usize, solver: LdaSolver) -> Self {
        LinearDiscriminantAnalysis {
            n_components,
            solver,
            reg_param: 0.0,
            components_: None,
            class_means_: None,
            global_mean_: None,
            explained_variance_ratio_: None,
            priors_: None,
            n_features_in_: None,
            classes_: None,
        }
    }

    /// Set regularization parameter for regularized LDA
    ///
    /// Adds `reg_param * I` to the within-class scatter matrix to handle
    /// singular or near-singular matrices.
    pub fn with_regularization(mut self, reg_param: f64) -> Self {
        self.reg_param = reg_param.max(0.0);
        self
    }

    /// Fit the LDA model
    ///
    /// # Arguments
    /// * `x` - Training data, shape (n_samples, n_features)
    /// * `y` - Class labels, shape (n_samples,)
    pub fn fit<L: Copy + Into<i64> + Eq + std::hash::Hash>(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<L>,
    ) -> Result<()> {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        if n_samples != y.len() {
            return Err(TransformError::InvalidInput(format!(
                "x has {} samples but y has {} elements",
                n_samples,
                y.len()
            )));
        }

        // Group samples by class
        let mut class_groups: HashMap<i64, Vec<usize>> = HashMap::new();
        for (i, &label) in y.iter().enumerate() {
            let key: i64 = label.into();
            class_groups.entry(key).or_default().push(i);
        }

        let mut classes: Vec<i64> = class_groups.keys().copied().collect();
        classes.sort();
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(TransformError::InvalidInput(
                "At least 2 classes required for LDA".to_string(),
            ));
        }

        let max_components = (n_classes - 1).min(n_features);
        if self.n_components > max_components {
            return Err(TransformError::InvalidInput(format!(
                "n_components={} must be <= min(n_classes-1, n_features)={}",
                self.n_components, max_components
            )));
        }

        // Compute global mean
        let global_mean = x.mean_axis(Axis(0)).ok_or_else(|| {
            TransformError::ComputationError("Failed to compute mean".to_string())
        })?;

        // Compute class means, priors, and scatter matrices
        let mut class_means = Array2::zeros((n_classes, n_features));
        let mut priors = Array1::zeros(n_classes);
        let mut sw = Array2::<f64>::zeros((n_features, n_features));
        let mut sb = Array2::<f64>::zeros((n_features, n_features));

        for (c, &class_label) in classes.iter().enumerate() {
            let indices = &class_groups[&class_label];
            let n_c = indices.len() as f64;
            priors[c] = n_c / n_samples as f64;

            // Compute class mean
            for &i in indices {
                for j in 0..n_features {
                    class_means[[c, j]] += x[[i, j]];
                }
            }
            for j in 0..n_features {
                class_means[[c, j]] /= n_c;
            }

            // Within-class scatter: sum of (x_i - mean_c)(x_i - mean_c)^T
            for &i in indices {
                for j in 0..n_features {
                    let dj = x[[i, j]] - class_means[[c, j]];
                    for k in j..=n_features.saturating_sub(1).min(n_features - 1) {
                        let dk = x[[i, k]] - class_means[[c, k]];
                        sw[[j, k]] += dj * dk;
                        if j != k {
                            sw[[k, j]] += dj * dk;
                        }
                    }
                }
            }

            // Between-class scatter: n_c * (mean_c - global_mean)(mean_c - global_mean)^T
            for j in 0..n_features {
                let dj = class_means[[c, j]] - global_mean[j];
                for k in j..n_features {
                    let dk = class_means[[c, k]] - global_mean[k];
                    let val = n_c * dj * dk;
                    sb[[j, k]] += val;
                    if j != k {
                        sb[[k, j]] += val;
                    }
                }
            }
        }

        // Apply regularization
        if self.reg_param > 0.0 {
            for i in 0..n_features {
                sw[[i, i]] += self.reg_param;
            }
        }

        // Solve the generalized eigenvalue problem: Sb * w = lambda * Sw * w
        let (components, eigenvalues) = match self.solver {
            LdaSolver::Svd => self.solve_svd(&sw, &sb, n_features)?,
            LdaSolver::Eigen => self.solve_eigen(&sw, &sb, n_features)?,
        };

        // Compute explained variance ratio
        let total_eigen = eigenvalues.iter().sum::<f64>();
        let explained_variance_ratio = if total_eigen > EPSILON {
            eigenvalues.mapv(|e| e / total_eigen)
        } else {
            Array1::from_elem(self.n_components, 1.0 / self.n_components as f64)
        };

        self.components_ = Some(components);
        self.class_means_ = Some(class_means);
        self.global_mean_ = Some(global_mean);
        self.explained_variance_ratio_ = Some(explained_variance_ratio);
        self.priors_ = Some(priors);
        self.n_features_in_ = Some(n_features);
        self.classes_ = Some(classes);

        Ok(())
    }

    /// SVD-based solver
    fn solve_svd(
        &self,
        sw: &Array2<f64>,
        sb: &Array2<f64>,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        // Compute Sw^{-1/2} via SVD
        let (u_sw, s_sw, vt_sw) =
            svd::<f64>(&sw.view(), true, None).map_err(TransformError::LinalgError)?;

        let mut sw_inv_sqrt = Array2::<f64>::zeros((n_features, n_features));
        for i in 0..n_features {
            if s_sw[i] > EPSILON {
                let s_inv = 1.0 / s_sw[i].sqrt();
                for j in 0..n_features {
                    for k in 0..n_features {
                        sw_inv_sqrt[[j, k]] += u_sw[[j, i]] * s_inv * vt_sw[[i, k]];
                    }
                }
            }
        }

        // Compute Sw^{-1/2} Sb Sw^{-1/2}
        // First: temp = Sb * Sw^{-1/2}^T
        let mut temp = Array2::<f64>::zeros((n_features, n_features));
        for i in 0..n_features {
            for j in 0..n_features {
                for k in 0..n_features {
                    temp[[i, j]] += sb[[i, k]] * sw_inv_sqrt[[j, k]]; // Sw^{-1/2}^T = Sw^{-1/2} (symmetric)
                }
            }
        }

        // Then: result = Sw^{-1/2} * temp
        let mut transformed_sb = Array2::<f64>::zeros((n_features, n_features));
        for i in 0..n_features {
            for j in 0..n_features {
                for k in 0..n_features {
                    transformed_sb[[i, j]] += sw_inv_sqrt[[i, k]] * temp[[k, j]];
                }
            }
        }

        // SVD of transformed Sb to get eigenvectors
        let (u_sb, s_sb, _vt_sb) =
            svd::<f64>(&transformed_sb.view(), true, None).map_err(TransformError::LinalgError)?;

        // Project back: W = Sw^{-1/2} * U_sb
        let mut components = Array2::zeros((self.n_components, n_features));
        let mut eigenvalues = Array1::zeros(self.n_components);

        for i in 0..self.n_components {
            eigenvalues[i] = s_sb[i].max(0.0);
            for j in 0..n_features {
                let mut val = 0.0;
                for k in 0..n_features {
                    val += sw_inv_sqrt[[j, k]] * u_sb[[k, i]];
                }
                components[[i, j]] = val;
            }

            // Normalize
            let norm: f64 = components.row(i).iter().map(|&v| v * v).sum::<f64>().sqrt();
            if norm > EPSILON {
                for j in 0..n_features {
                    components[[i, j]] /= norm;
                }
            }
        }

        Ok((components, eigenvalues))
    }

    /// Eigenvalue-based solver
    fn solve_eigen(
        &self,
        sw: &Array2<f64>,
        sb: &Array2<f64>,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        // Regularize Sw
        let mut sw_reg = sw.clone();
        for i in 0..n_features {
            sw_reg[[i, i]] += EPSILON;
        }

        // Compute Sw^{-1}
        let (u_sw, s_sw, vt_sw) =
            svd::<f64>(&sw_reg.view(), true, None).map_err(TransformError::LinalgError)?;

        let mut sw_inv = Array2::<f64>::zeros((n_features, n_features));
        for i in 0..n_features {
            if s_sw[i] > EPSILON {
                let s_inv = 1.0 / s_sw[i];
                for j in 0..n_features {
                    for k in 0..n_features {
                        sw_inv[[j, k]] += u_sw[[j, i]] * s_inv * vt_sw[[i, k]];
                    }
                }
            }
        }

        // Compute Sw^{-1} * Sb
        let mut m = Array2::<f64>::zeros((n_features, n_features));
        for i in 0..n_features {
            for j in 0..n_features {
                for k in 0..n_features {
                    m[[i, j]] += sw_inv[[i, k]] * sb[[k, j]];
                }
            }
        }

        // Symmetrize: (M + M^T) / 2
        let mut sym = Array2::<f64>::zeros((n_features, n_features));
        for i in 0..n_features {
            for j in 0..n_features {
                sym[[i, j]] = (m[[i, j]] + m[[j, i]]) / 2.0;
            }
        }

        // Eigendecomposition
        let (eig_vals, eig_vecs) =
            scirs2_linalg::eigh::<f64>(&sym.view(), None).or_else(|_| -> Result<_> {
                // Fallback to SVD
                let (u, s, _vt) =
                    svd::<f64>(&m.view(), true, None).map_err(TransformError::LinalgError)?;
                Ok((s, u))
            })?;

        // Sort eigenvalues descending
        let mut indices: Vec<usize> = (0..n_features).collect();
        indices.sort_by(|&a, &b| {
            eig_vals[b]
                .partial_cmp(&eig_vals[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut components = Array2::zeros((self.n_components, n_features));
        let mut eigenvalues = Array1::zeros(self.n_components);

        for i in 0..self.n_components {
            let idx = indices[i];
            eigenvalues[i] = eig_vals[idx].max(0.0);

            let mut norm = 0.0;
            for j in 0..n_features {
                components[[i, j]] = eig_vecs[[j, idx]];
                norm += components[[i, j]] * components[[i, j]];
            }
            norm = norm.sqrt();

            if norm > EPSILON {
                for j in 0..n_features {
                    components[[i, j]] /= norm;
                }
            }
        }

        Ok((components, eigenvalues))
    }

    /// Project data onto discriminant directions
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let components = self
            .components_
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("LDA has not been fitted".to_string()))?;

        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];
        let n_features_in = self.n_features_in_.unwrap_or(0);

        if n_features != n_features_in {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, expected {}",
                n_features, n_features_in
            )));
        }

        let mut transformed = Array2::zeros((n_samples, self.n_components));
        for i in 0..n_samples {
            for j in 0..self.n_components {
                let mut dot = 0.0;
                for k in 0..n_features {
                    dot += x[[i, k]] * components[[j, k]];
                }
                transformed[[i, j]] = dot;
            }
        }

        Ok(transformed)
    }

    /// Fit and transform
    pub fn fit_transform<L: Copy + Into<i64> + Eq + std::hash::Hash>(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<L>,
    ) -> Result<Array2<f64>> {
        self.fit(x, y)?;
        self.transform(x)
    }

    /// Predict class labels for new data using nearest-centroid in projected space
    pub fn predict<L: Copy + Into<i64> + Eq + std::hash::Hash>(
        &self,
        x: &Array2<f64>,
    ) -> Result<Array1<i64>> {
        let components = self
            .components_
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("LDA has not been fitted".to_string()))?;
        let class_means = self
            .class_means_
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("LDA has not been fitted".to_string()))?;
        let classes = self
            .classes_
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("LDA has not been fitted".to_string()))?;

        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];
        let n_classes = classes.len();

        // Project class means
        let mut projected_means = Array2::zeros((n_classes, self.n_components));
        for c in 0..n_classes {
            for j in 0..self.n_components {
                let mut dot = 0.0;
                for k in 0..n_features {
                    dot += class_means[[c, k]] * components[[j, k]];
                }
                projected_means[[c, j]] = dot;
            }
        }

        // Nearest centroid classification
        let mut predictions = Array1::zeros(n_samples);
        for i in 0..n_samples {
            // Project sample
            let mut projected = Array1::zeros(self.n_components);
            for j in 0..self.n_components {
                let mut dot = 0.0;
                for k in 0..n_features {
                    dot += x[[i, k]] * components[[j, k]];
                }
                projected[j] = dot;
            }

            // Find nearest class centroid
            let mut best_class = 0;
            let mut best_dist = f64::MAX;
            for c in 0..n_classes {
                let mut dist = 0.0;
                for j in 0..self.n_components {
                    let d = projected[j] - projected_means[[c, j]];
                    dist += d * d;
                }
                if dist < best_dist {
                    best_dist = dist;
                    best_class = c;
                }
            }

            predictions[i] = classes[best_class];
        }

        Ok(predictions)
    }

    /// Get the projection components
    pub fn components(&self) -> Option<&Array2<f64>> {
        self.components_.as_ref()
    }

    /// Get the class means
    pub fn class_means(&self) -> Option<&Array2<f64>> {
        self.class_means_.as_ref()
    }

    /// Get the explained variance ratio
    pub fn explained_variance_ratio(&self) -> Option<&Array1<f64>> {
        self.explained_variance_ratio_.as_ref()
    }

    /// Get class priors
    pub fn priors(&self) -> Option<&Array1<f64>> {
        self.priors_.as_ref()
    }

    /// Get the unique classes
    pub fn classes(&self) -> Option<&Vec<i64>> {
        self.classes_.as_ref()
    }
}

/// Quadratic Discriminant Analysis (QDA)
///
/// Unlike LDA, QDA does not assume shared covariance across classes.
/// Each class has its own covariance matrix, leading to quadratic decision
/// boundaries.
///
/// QDA is primarily a classifier and does not produce a linear projection
/// for dimensionality reduction.
///
/// # Examples
///
/// ```
/// use scirs2_transform::reduction::lda::QuadraticDiscriminantAnalysis;
/// use scirs2_core::ndarray::{Array1, Array2};
///
/// let x = Array2::from_shape_vec(
///     (6, 2),
///     vec![1.0, 2.0, 1.1, 2.1, 0.9, 1.9,
///          5.0, 4.0, 5.1, 4.1, 4.9, 3.9],
/// ).expect("should succeed");
/// let y = Array1::from_vec(vec![0_i64, 0, 0, 1, 1, 1]);
///
/// let mut qda = QuadraticDiscriminantAnalysis::new();
/// qda.fit(&x, &y).expect("should succeed");
/// let preds = qda.predict(&x).expect("should succeed");
/// assert_eq!(preds.len(), 6);
/// ```
#[derive(Debug, Clone)]
pub struct QuadraticDiscriminantAnalysis {
    /// Regularization parameter
    reg_param: f64,
    /// Class means, shape (n_classes, n_features)
    class_means_: Option<Array2<f64>>,
    /// Class covariance inverses, indexed by class
    class_cov_inv_: Option<Vec<Array2<f64>>>,
    /// Log determinants of class covariances
    class_log_det_: Option<Vec<f64>>,
    /// Class priors
    priors_: Option<Array1<f64>>,
    /// Unique classes
    classes_: Option<Vec<i64>>,
    /// Number of features
    n_features_in_: Option<usize>,
}

impl QuadraticDiscriminantAnalysis {
    /// Create a new QDA instance
    pub fn new() -> Self {
        QuadraticDiscriminantAnalysis {
            reg_param: 1e-6,
            class_means_: None,
            class_cov_inv_: None,
            class_log_det_: None,
            priors_: None,
            classes_: None,
            n_features_in_: None,
        }
    }

    /// Set regularization parameter
    pub fn with_regularization(mut self, reg_param: f64) -> Self {
        self.reg_param = reg_param.max(0.0);
        self
    }

    /// Fit the QDA model
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<i64>) -> Result<()> {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        if n_samples != y.len() {
            return Err(TransformError::InvalidInput(format!(
                "x has {} samples, y has {}",
                n_samples,
                y.len()
            )));
        }

        // Group by class
        let mut class_groups: HashMap<i64, Vec<usize>> = HashMap::new();
        for (i, &label) in y.iter().enumerate() {
            class_groups.entry(label).or_default().push(i);
        }

        let mut classes: Vec<i64> = class_groups.keys().copied().collect();
        classes.sort();
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(TransformError::InvalidInput(
                "At least 2 classes required".to_string(),
            ));
        }

        let mut class_means = Array2::zeros((n_classes, n_features));
        let mut priors = Array1::zeros(n_classes);
        let mut cov_invs = Vec::with_capacity(n_classes);
        let mut log_dets = Vec::with_capacity(n_classes);

        for (c, &class_label) in classes.iter().enumerate() {
            let indices = &class_groups[&class_label];
            let n_c = indices.len();
            priors[c] = n_c as f64 / n_samples as f64;

            if n_c < n_features + 1 {
                return Err(TransformError::InvalidInput(format!(
                    "Class {} has {} samples, need at least {} for QDA with {} features",
                    class_label,
                    n_c,
                    n_features + 1,
                    n_features
                )));
            }

            // Compute class mean
            for &i in indices {
                for j in 0..n_features {
                    class_means[[c, j]] += x[[i, j]];
                }
            }
            for j in 0..n_features {
                class_means[[c, j]] /= n_c as f64;
            }

            // Compute class covariance
            let mut cov = Array2::<f64>::zeros((n_features, n_features));
            for &i in indices {
                for j in 0..n_features {
                    let dj = x[[i, j]] - class_means[[c, j]];
                    for k in j..n_features {
                        let dk = x[[i, k]] - class_means[[c, k]];
                        let val = dj * dk;
                        cov[[j, k]] += val;
                        if j != k {
                            cov[[k, j]] += val;
                        }
                    }
                }
            }
            // Normalize by n_c - 1 (unbiased)
            let denom = (n_c - 1) as f64;
            cov.mapv_inplace(|v| v / denom);

            // Add regularization
            for i in 0..n_features {
                cov[[i, i]] += self.reg_param;
            }

            // Compute inverse and log-determinant via SVD
            let (u, s, vt) =
                svd::<f64>(&cov.view(), true, None).map_err(TransformError::LinalgError)?;

            let mut log_det = 0.0;
            let mut cov_inv = Array2::<f64>::zeros((n_features, n_features));

            for i in 0..n_features {
                if s[i] > EPSILON {
                    log_det += s[i].ln();
                    let s_inv = 1.0 / s[i];
                    for j in 0..n_features {
                        for k in 0..n_features {
                            cov_inv[[j, k]] += u[[j, i]] * s_inv * vt[[i, k]];
                        }
                    }
                }
            }

            cov_invs.push(cov_inv);
            log_dets.push(log_det);
        }

        self.class_means_ = Some(class_means);
        self.class_cov_inv_ = Some(cov_invs);
        self.class_log_det_ = Some(log_dets);
        self.priors_ = Some(priors);
        self.classes_ = Some(classes);
        self.n_features_in_ = Some(n_features);

        Ok(())
    }

    /// Predict class labels for new data
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<i64>> {
        let class_means = self
            .class_means_
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("QDA has not been fitted".to_string()))?;
        let cov_invs = self
            .class_cov_inv_
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("QDA has not been fitted".to_string()))?;
        let log_dets = self
            .class_log_det_
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("QDA has not been fitted".to_string()))?;
        let priors = self
            .priors_
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("QDA has not been fitted".to_string()))?;
        let classes = self
            .classes_
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("QDA has not been fitted".to_string()))?;

        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];
        let n_classes = classes.len();

        let n_features_in = self.n_features_in_.unwrap_or(0);
        if n_features != n_features_in {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, expected {}",
                n_features, n_features_in
            )));
        }

        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut best_class = 0;
            let mut best_score = f64::NEG_INFINITY;

            for c in 0..n_classes {
                // Compute (x - mu_c)^T Sigma_c^{-1} (x - mu_c)
                let mut diff = Array1::zeros(n_features);
                for j in 0..n_features {
                    diff[j] = x[[i, j]] - class_means[[c, j]];
                }

                let mut mahal = 0.0;
                for j in 0..n_features {
                    let mut temp = 0.0;
                    for k in 0..n_features {
                        temp += cov_invs[c][[j, k]] * diff[k];
                    }
                    mahal += diff[j] * temp;
                }

                // Log posterior: log(prior_c) - 0.5*log(det) - 0.5*mahal
                let score = priors[c].ln() - 0.5 * log_dets[c] - 0.5 * mahal;

                if score > best_score {
                    best_score = score;
                    best_class = c;
                }
            }

            predictions[i] = classes[best_class];
        }

        Ok(predictions)
    }

    /// Compute posterior probabilities for each class
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let class_means = self
            .class_means_
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("QDA has not been fitted".to_string()))?;
        let cov_invs = self
            .class_cov_inv_
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("QDA has not been fitted".to_string()))?;
        let log_dets = self
            .class_log_det_
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("QDA has not been fitted".to_string()))?;
        let priors = self
            .priors_
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("QDA has not been fitted".to_string()))?;
        let classes = self
            .classes_
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("QDA has not been fitted".to_string()))?;

        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];
        let n_classes = classes.len();

        let mut probas = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let mut log_scores = Array1::zeros(n_classes);
            let mut max_log = f64::NEG_INFINITY;

            for c in 0..n_classes {
                let mut diff = Array1::zeros(n_features);
                for j in 0..n_features {
                    diff[j] = x[[i, j]] - class_means[[c, j]];
                }

                let mut mahal = 0.0;
                for j in 0..n_features {
                    let mut temp = 0.0;
                    for k in 0..n_features {
                        temp += cov_invs[c][[j, k]] * diff[k];
                    }
                    mahal += diff[j] * temp;
                }

                log_scores[c] = priors[c].ln() - 0.5 * log_dets[c] - 0.5 * mahal;
                if log_scores[c] > max_log {
                    max_log = log_scores[c];
                }
            }

            // Log-sum-exp for numerical stability
            let mut sum_exp = 0.0;
            for c in 0..n_classes {
                sum_exp += (log_scores[c] - max_log).exp();
            }
            let log_sum = max_log + sum_exp.ln();

            for c in 0..n_classes {
                probas[[i, c]] = (log_scores[c] - log_sum).exp();
            }
        }

        Ok(probas)
    }

    /// Get class means
    pub fn class_means(&self) -> Option<&Array2<f64>> {
        self.class_means_.as_ref()
    }

    /// Get class priors
    pub fn priors(&self) -> Option<&Array1<f64>> {
        self.priors_.as_ref()
    }

    /// Get unique classes
    pub fn classes(&self) -> Option<&Vec<i64>> {
        self.classes_.as_ref()
    }
}

impl Default for QuadraticDiscriminantAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array;

    fn make_two_class_data() -> (Array2<f64>, Array1<i64>) {
        let x = Array::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.1, 2.1, 0.9, 1.9, 5.0, 4.0, 5.1, 4.1, 4.9, 3.9],
        )
        .expect("test data");
        let y = Array::from_vec(vec![0_i64, 0, 0, 1, 1, 1]);
        (x, y)
    }

    fn make_three_class_data() -> (Array2<f64>, Array1<i64>) {
        let x = Array::from_shape_vec(
            (9, 3),
            vec![
                1.0, 2.0, 0.5, 1.1, 2.1, 0.6, 0.9, 1.9, 0.4, 5.0, 4.0, 3.5, 5.1, 4.1, 3.6, 4.9,
                3.9, 3.4, 9.0, 8.0, 7.5, 9.1, 8.1, 7.6, 8.9, 7.9, 7.4,
            ],
        )
        .expect("test data");
        let y = Array::from_vec(vec![0_i64, 0, 0, 1, 1, 1, 2, 2, 2]);
        (x, y)
    }

    #[test]
    fn test_lda_two_class_svd() {
        let (x, y) = make_two_class_data();

        let mut lda = LinearDiscriminantAnalysis::new(1, LdaSolver::Svd);
        let projected = lda.fit_transform(&x, &y).expect("fit_transform");
        assert_eq!(projected.shape(), &[6, 1]);

        // Classes should be separated in projected space
        let class0_mean = (projected[[0, 0]] + projected[[1, 0]] + projected[[2, 0]]) / 3.0;
        let class1_mean = (projected[[3, 0]] + projected[[4, 0]] + projected[[5, 0]]) / 3.0;
        assert!((class0_mean - class1_mean).abs() > 1.0);

        // Explained variance should be 1.0 for single component
        let evr = lda.explained_variance_ratio().expect("evr");
        assert_abs_diff_eq!(evr[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lda_two_class_eigen() {
        let (x, y) = make_two_class_data();

        let mut lda = LinearDiscriminantAnalysis::new(1, LdaSolver::Eigen);
        let projected = lda.fit_transform(&x, &y).expect("fit_transform");
        assert_eq!(projected.shape(), &[6, 1]);
        assert!(projected.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_lda_three_class() {
        let (x, y) = make_three_class_data();

        let mut lda = LinearDiscriminantAnalysis::new(2, LdaSolver::Svd);
        let projected = lda.fit_transform(&x, &y).expect("fit_transform");
        assert_eq!(projected.shape(), &[9, 2]);

        let evr = lda.explained_variance_ratio().expect("evr");
        assert_eq!(evr.len(), 2);
        assert_abs_diff_eq!(evr.sum(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lda_regularized() {
        let (x, y) = make_two_class_data();

        let mut lda = LinearDiscriminantAnalysis::new(1, LdaSolver::Svd).with_regularization(0.1);
        let projected = lda.fit_transform(&x, &y).expect("fit_transform");
        assert_eq!(projected.shape(), &[6, 1]);
        assert!(projected.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_lda_predict() {
        let (x, y) = make_two_class_data();

        let mut lda = LinearDiscriminantAnalysis::new(1, LdaSolver::Svd);
        lda.fit(&x, &y).expect("fit");

        let preds = lda.predict::<i64>(&x).expect("predict");
        assert_eq!(preds.len(), 6);

        // Training data should be correctly classified
        for i in 0..3 {
            assert_eq!(preds[i], 0, "sample {} should be class 0", i);
        }
        for i in 3..6 {
            assert_eq!(preds[i], 1, "sample {} should be class 1", i);
        }
    }

    #[test]
    fn test_lda_errors() {
        let x = Array::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("test data");

        // Single class
        let y_single = Array::from_vec(vec![0_i64, 0, 0, 0]);
        let mut lda = LinearDiscriminantAnalysis::new(1, LdaSolver::Svd);
        assert!(lda.fit(&x, &y_single).is_err());

        // Length mismatch
        let y_short = Array::from_vec(vec![0_i64, 1]);
        let mut lda2 = LinearDiscriminantAnalysis::new(1, LdaSolver::Svd);
        assert!(lda2.fit(&x, &y_short).is_err());

        // Too many components
        let y = Array::from_vec(vec![0_i64, 0, 1, 1]);
        let mut lda3 = LinearDiscriminantAnalysis::new(2, LdaSolver::Svd);
        assert!(lda3.fit(&x, &y).is_err());
    }

    #[test]
    fn test_lda_not_fitted() {
        let x = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("test data");
        let lda = LinearDiscriminantAnalysis::new(1, LdaSolver::Svd);
        assert!(lda.transform(&x).is_err());
    }

    #[test]
    fn test_qda_basic() {
        // Need more samples than features for each class
        let x = Array::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 1.1, 2.1, 0.9, 1.9, 1.2, 2.2, 5.0, 4.0, 5.1, 4.1, 4.9, 3.9, 5.2, 4.2,
            ],
        )
        .expect("test data");
        let y = Array::from_vec(vec![0_i64, 0, 0, 0, 1, 1, 1, 1]);

        let mut qda = QuadraticDiscriminantAnalysis::new();
        qda.fit(&x, &y).expect("fit");

        let preds = qda.predict(&x).expect("predict");
        assert_eq!(preds.len(), 8);

        // Training data should be classified correctly
        for i in 0..4 {
            assert_eq!(preds[i], 0);
        }
        for i in 4..8 {
            assert_eq!(preds[i], 1);
        }
    }

    #[test]
    fn test_qda_predict_proba() {
        let x = Array::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 1.1, 2.1, 0.9, 1.9, 1.2, 2.2, 5.0, 4.0, 5.1, 4.1, 4.9, 3.9, 5.2, 4.2,
            ],
        )
        .expect("test data");
        let y = Array::from_vec(vec![0_i64, 0, 0, 0, 1, 1, 1, 1]);

        let mut qda = QuadraticDiscriminantAnalysis::new();
        qda.fit(&x, &y).expect("fit");

        let probas = qda.predict_proba(&x).expect("predict_proba");
        assert_eq!(probas.shape(), &[8, 2]);

        // Probabilities should sum to 1 for each sample
        for i in 0..8 {
            let sum: f64 = (0..2).map(|c| probas[[i, c]]).sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
        }

        // Class 0 samples should have higher probability for class 0
        for i in 0..4 {
            assert!(probas[[i, 0]] > probas[[i, 1]]);
        }
    }

    #[test]
    fn test_qda_errors() {
        let x = Array::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("test data");

        // Single class
        let y = Array::from_vec(vec![0_i64, 0, 0, 0]);
        let mut qda = QuadraticDiscriminantAnalysis::new();
        assert!(qda.fit(&x, &y).is_err());
    }

    #[test]
    fn test_qda_not_fitted() {
        let x = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("test data");
        let qda = QuadraticDiscriminantAnalysis::new();
        assert!(qda.predict(&x).is_err());
    }

    #[test]
    fn test_qda_regularized() {
        let x = Array::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 1.1, 2.1, 0.9, 1.9, 1.2, 2.2, 5.0, 4.0, 5.1, 4.1, 4.9, 3.9, 5.2, 4.2,
            ],
        )
        .expect("test data");
        let y = Array::from_vec(vec![0_i64, 0, 0, 0, 1, 1, 1, 1]);

        let mut qda = QuadraticDiscriminantAnalysis::new().with_regularization(1.0);
        qda.fit(&x, &y).expect("fit");

        let preds = qda.predict(&x).expect("predict");
        assert_eq!(preds.len(), 8);
    }
}
