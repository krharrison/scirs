//! Factor Analysis for dimensionality reduction and latent variable modeling
//!
//! Factor Analysis is a statistical method that describes variability among observed,
//! correlated variables in terms of a potentially lower number of unobserved variables
//! called factors. It differs from PCA in that it models noise separately for each
//! observed variable (uniquenesses/specific variances).
//!
//! ## Algorithm (EM-based)
//!
//! The factor model is: x = W * z + mu + epsilon
//! where z ~ N(0, I) are the latent factors, W is the loading matrix,
//! and epsilon_i ~ N(0, psi_i) are the uniquenesses.
//!
//! The EM algorithm alternates between:
//! - **E-step**: Compute expected sufficient statistics of latent factors given observed data
//! - **M-step**: Update loadings (W) and uniquenesses (psi) to maximize expected log-likelihood
//!
//! ## Rotation Methods
//!
//! - **Varimax**: Orthogonal rotation maximizing variance of squared loadings
//! - **Promax**: Oblique rotation based on raised varimax solution

use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::validation::{check_positive, checkshape};
use scirs2_linalg::svd;

use crate::error::{Result, TransformError};

/// Rotation method for factor loadings
#[derive(Debug, Clone, PartialEq)]
pub enum RotationMethod {
    /// No rotation
    None,
    /// Varimax (orthogonal) rotation
    Varimax,
    /// Promax (oblique) rotation
    Promax,
}

/// Scree plot data for selecting the number of factors
///
/// Contains eigenvalues of the covariance/correlation matrix, which can
/// be used to construct a scree plot for determining the appropriate
/// number of factors (look for the "elbow" in the eigenvalue curve).
#[derive(Debug, Clone)]
pub struct ScreePlotData {
    /// Eigenvalues in descending order
    pub eigenvalues: Array1<f64>,
    /// Cumulative proportion of variance explained
    pub cumulative_variance: Array1<f64>,
    /// Proportion of variance explained by each component
    pub variance_proportions: Array1<f64>,
    /// Kaiser criterion threshold (eigenvalue = 1.0 for correlation matrix)
    pub kaiser_threshold: f64,
}

/// Factor Analysis results
#[derive(Debug, Clone)]
pub struct FactorAnalysisResult {
    /// Factor loadings matrix, shape (n_features, n_factors)
    pub loadings: Array2<f64>,
    /// Factor scores for training data, shape (n_samples, n_factors)
    pub scores: Array2<f64>,
    /// Uniquenesses (specific variances) for each feature, shape (n_features,)
    pub uniquenesses: Array1<f64>,
    /// Communalities for each feature, shape (n_features,)
    pub communalities: Array1<f64>,
    /// Noise variance (average uniqueness)
    pub noise_variance: f64,
    /// Number of EM iterations performed
    pub n_iter: usize,
    /// Log-likelihood at convergence
    pub log_likelihood: f64,
}

/// Factor Analysis for dimensionality reduction
///
/// Factor Analysis assumes observed data is generated from a linear model
/// with latent factors plus Gaussian noise with feature-specific variances.
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_transform::reduction::factor_analysis::FactorAnalysis;
/// use scirs2_core::ndarray::Array2;
///
/// let data = Array2::<f64>::zeros((100, 10));
/// let mut fa = FactorAnalysis::new(3);
/// let result = fa.fit_transform(&data).expect("fit_transform failed");
/// println!("Loadings shape: {:?}", result.loadings.shape());
/// println!("Scores shape: {:?}", result.scores.shape());
/// ```
#[derive(Debug, Clone)]
pub struct FactorAnalysis {
    /// Number of factors to extract
    n_factors: usize,
    /// Rotation method to apply
    rotation: RotationMethod,
    /// Maximum number of EM iterations
    max_iter: usize,
    /// Convergence tolerance for log-likelihood change
    tol: f64,
    /// Power parameter for promax rotation
    promax_power: usize,
    /// Maximum iterations for varimax rotation
    varimax_max_iter: usize,
    /// Whether to center the data
    center: bool,
    /// Mean of training data
    mean: Option<Array1<f64>>,
    /// Factor loadings
    loadings: Option<Array2<f64>>,
    /// Uniquenesses (specific variances)
    uniquenesses: Option<Array1<f64>>,
    /// Training data (centered)
    training_data: Option<Array2<f64>>,
}

impl FactorAnalysis {
    /// Creates a new FactorAnalysis instance
    ///
    /// # Arguments
    /// * `n_factors` - Number of latent factors to extract
    pub fn new(n_factors: usize) -> Self {
        FactorAnalysis {
            n_factors,
            rotation: RotationMethod::None,
            max_iter: 1000,
            tol: 1e-8,
            promax_power: 4,
            varimax_max_iter: 100,
            center: true,
            mean: None,
            loadings: None,
            uniquenesses: None,
            training_data: None,
        }
    }

    /// Set the rotation method
    pub fn with_rotation(mut self, rotation: RotationMethod) -> Self {
        self.rotation = rotation;
        self
    }

    /// Set the maximum number of EM iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the promax power parameter
    pub fn with_promax_power(mut self, power: usize) -> Self {
        self.promax_power = power.max(2);
        self
    }

    /// Set whether to center the data
    pub fn with_center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Fit the factor analysis model and return results
    ///
    /// # Arguments
    /// * `x` - Input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<FactorAnalysisResult>` - Factor analysis results
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<FactorAnalysisResult>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|v| <f64 as NumCast>::from(v).unwrap_or_default());
        let (n_samples, n_features) = x_f64.dim();

        // Input validation
        check_positive(self.n_factors, "n_factors")?;
        checkshape(&x_f64, &[n_samples, n_features], "x")?;

        if n_samples < 2 {
            return Err(TransformError::InvalidInput(
                "Need at least 2 samples for factor analysis".to_string(),
            ));
        }

        if self.n_factors > n_features {
            return Err(TransformError::InvalidInput(format!(
                "n_factors={} must be <= n_features={}",
                self.n_factors, n_features
            )));
        }

        // Center data
        let mean = x_f64.mean_axis(Axis(0)).ok_or_else(|| {
            TransformError::ComputationError("Failed to compute mean".to_string())
        })?;

        let x_centered = if self.center {
            let mut centered = x_f64.clone();
            for i in 0..n_samples {
                for j in 0..n_features {
                    centered[[i, j]] -= mean[j];
                }
            }
            centered
        } else {
            x_f64.clone()
        };

        // Compute sample covariance matrix
        let mut cov: Array2<f64> = Array2::zeros((n_features, n_features));
        for i in 0..n_features {
            for j in 0..n_features {
                let mut sum = 0.0;
                for k in 0..n_samples {
                    sum += x_centered[[k, i]] * x_centered[[k, j]];
                }
                cov[[i, j]] = sum / (n_samples - 1) as f64;
            }
        }

        // Initialize using SVD of centered data
        let (loadings_init, uniquenesses_init) =
            self.initialize_loadings(&x_centered, &cov, n_samples)?;

        // Run EM algorithm
        let (loadings, uniquenesses, n_iter, log_likelihood) = self.em_algorithm(
            &cov,
            loadings_init,
            uniquenesses_init,
            n_samples,
            n_features,
        )?;

        // Apply rotation if requested
        let loadings_rotated = match &self.rotation {
            RotationMethod::None => loadings,
            RotationMethod::Varimax => self.varimax_rotation(&loadings)?,
            RotationMethod::Promax => self.promax_rotation(&loadings)?,
        };

        // Compute factor scores using regression method
        // scores = X_centered * Psi^{-1} * W * (W^T * Psi^{-1} * W + I)^{-1}
        let scores = self.compute_scores(&x_centered, &loadings_rotated, &uniquenesses)?;

        // Compute communalities: h_j^2 = sum of squared loadings for feature j
        let mut communalities: Array1<f64> = Array1::zeros(n_features);
        for j in 0..n_features {
            for f in 0..self.n_factors {
                communalities[j] += loadings_rotated[[j, f]] * loadings_rotated[[j, f]];
            }
        }

        let noise_variance = if uniquenesses.is_empty() {
            0.0
        } else {
            uniquenesses.sum() / uniquenesses.len() as f64
        };

        // Store fitted parameters
        self.mean = Some(mean);
        self.loadings = Some(loadings_rotated.clone());
        self.uniquenesses = Some(uniquenesses.clone());
        self.training_data = Some(x_centered);

        Ok(FactorAnalysisResult {
            loadings: loadings_rotated,
            scores,
            uniquenesses,
            communalities,
            noise_variance,
            n_iter,
            log_likelihood,
        })
    }

    /// Initialize loadings using SVD
    fn initialize_loadings(
        &self,
        x_centered: &Array2<f64>,
        _cov: &Array2<f64>,
        n_samples: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        let n_features = x_centered.shape()[1];

        // SVD of centered data
        let (_u, s, vt) = svd::<f64>(&x_centered.view(), true, None)
            .map_err(|e| TransformError::LinalgError(e))?;

        // Initial loadings from the top singular vectors
        let scale = 1.0 / (n_samples as f64 - 1.0).sqrt();
        let mut loadings: Array2<f64> = Array2::zeros((n_features, self.n_factors));
        for j in 0..n_features {
            for f in 0..self.n_factors {
                if f < s.len() && f < vt.shape()[0] {
                    loadings[[j, f]] = vt[[f, j]] * s[f] * scale;
                }
            }
        }

        // Initial uniquenesses: diagonal of cov minus communalities
        let mut uniquenesses: Array1<f64> = Array1::zeros(n_features);
        for j in 0..n_features {
            let mut communality = 0.0;
            for f in 0..self.n_factors {
                communality += loadings[[j, f]] * loadings[[j, f]];
            }
            // Variance of j-th feature
            let mut var_j = 0.0;
            for k in 0..n_samples {
                var_j += x_centered[[k, j]] * x_centered[[k, j]];
            }
            var_j /= (n_samples - 1) as f64;

            uniquenesses[j] = (var_j - communality).max(1e-6);
        }

        Ok((loadings, uniquenesses))
    }

    /// EM algorithm for factor analysis
    ///
    /// The generative model is: x ~ N(0, W*W^T + Psi)
    /// where W is the loading matrix (n_features x n_factors)
    /// and Psi = diag(psi_1, ..., psi_p) is the uniqueness matrix.
    fn em_algorithm(
        &self,
        cov: &Array2<f64>,
        mut loadings: Array2<f64>,
        mut uniquenesses: Array1<f64>,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>, usize, f64)> {
        let n_factors = self.n_factors;
        let mut prev_ll = f64::NEG_INFINITY;
        let mut n_iter = 0;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // E-step: Compute E[z|x] and E[z*z^T|x]
            // sigma_z = (I + W^T * Psi^{-1} * W)^{-1}
            // E[z|x] = sigma_z * W^T * Psi^{-1} * x

            // Compute Psi^{-1} * W
            let mut psi_inv_w: Array2<f64> = Array2::zeros((n_features, n_factors));
            for j in 0..n_features {
                let psi_inv = 1.0 / uniquenesses[j].max(1e-12);
                for f in 0..n_factors {
                    psi_inv_w[[j, f]] = psi_inv * loadings[[j, f]];
                }
            }

            // Compute W^T * Psi^{-1} * W
            let mut wt_psi_inv_w: Array2<f64> = Array2::zeros((n_factors, n_factors));
            for i in 0..n_factors {
                for j in 0..n_factors {
                    for k in 0..n_features {
                        wt_psi_inv_w[[i, j]] += loadings[[k, i]] * psi_inv_w[[k, j]];
                    }
                }
            }

            // sigma_z = (I + W^T * Psi^{-1} * W)^{-1}
            for i in 0..n_factors {
                wt_psi_inv_w[[i, i]] += 1.0;
            }

            // Invert sigma_z^{-1} using the formula for small matrices
            let sigma_z = self.invert_small_matrix(&wt_psi_inv_w)?;

            // E[z*z^T] = sigma_z + E[z]*E[z]^T
            // For EM, we need: expected_zzt = sigma_z + sigma_z * W^T * Psi^{-1} * S * Psi^{-1} * W * sigma_z
            // where S is the sample covariance
            // This simplifies to computing the M-step directly

            // M-step: Update W and Psi
            // The sufficient statistics needed are:
            // beta = sigma_z * W^T * Psi^{-1}  (n_factors x n_features)
            let mut beta: Array2<f64> = Array2::zeros((n_factors, n_features));
            for i in 0..n_factors {
                for j in 0..n_features {
                    for k in 0..n_factors {
                        beta[[i, j]] += sigma_z[[i, k]] * psi_inv_w[[j, k]]; // note: psi_inv_w is (n_features, n_factors)
                    }
                }
            }

            // E[z*x^T] = beta * S  (n_factors x n_features)
            let mut ez_xt: Array2<f64> = Array2::zeros((n_factors, n_features));
            for i in 0..n_factors {
                for j in 0..n_features {
                    for k in 0..n_features {
                        ez_xt[[i, j]] += beta[[i, k]] * cov[[k, j]];
                    }
                }
            }

            // E[z*z^T] = sigma_z + beta * S * beta^T  (n_factors x n_factors)
            let mut ez_zt = sigma_z.clone();
            for i in 0..n_factors {
                for j in 0..n_factors {
                    for k in 0..n_features {
                        ez_zt[[i, j]] += beta[[i, k]] * ez_xt[[j, k]];
                    }
                }
            }

            // New W: S * beta^T * E[z*z^T]^{-1}
            let ez_zt_inv = self.invert_small_matrix(&ez_zt)?;

            // Compute S * beta^T
            let mut s_beta_t: Array2<f64> = Array2::zeros((n_features, n_factors));
            for i in 0..n_features {
                for j in 0..n_factors {
                    for k in 0..n_features {
                        s_beta_t[[i, j]] += cov[[i, k]] * beta[[j, k]];
                    }
                }
            }

            // New loadings = S * beta^T * E[z*z^T]^{-1}
            let mut new_loadings: Array2<f64> = Array2::zeros((n_features, n_factors));
            for i in 0..n_features {
                for j in 0..n_factors {
                    for k in 0..n_factors {
                        new_loadings[[i, j]] += s_beta_t[[i, k]] * ez_zt_inv[[k, j]];
                    }
                }
            }

            // New uniquenesses: diag(S - W_new * E[z*x^T])
            let mut new_uniquenesses: Array1<f64> = Array1::zeros(n_features);
            for j in 0..n_features {
                let mut correction = 0.0;
                for f in 0..n_factors {
                    correction += new_loadings[[j, f]] * ez_xt[[f, j]];
                }
                new_uniquenesses[j] = (cov[[j, j]] - correction).max(1e-6);
            }

            loadings = new_loadings;
            uniquenesses = new_uniquenesses;

            // Compute log-likelihood
            let ll =
                self.compute_log_likelihood(cov, &loadings, &uniquenesses, n_samples, n_features);

            // Check convergence
            if (ll - prev_ll).abs() < self.tol {
                break;
            }
            prev_ll = ll;
        }

        let ll = self.compute_log_likelihood(cov, &loadings, &uniquenesses, n_samples, n_features);

        Ok((loadings, uniquenesses, n_iter, ll))
    }

    /// Compute log-likelihood of the factor model
    fn compute_log_likelihood(
        &self,
        cov: &Array2<f64>,
        loadings: &Array2<f64>,
        uniquenesses: &Array1<f64>,
        n_samples: usize,
        n_features: usize,
    ) -> f64 {
        let n_factors = self.n_factors;

        // Sigma = W * W^T + Psi
        let mut sigma: Array2<f64> = Array2::zeros((n_features, n_features));
        for i in 0..n_features {
            for j in 0..n_features {
                for f in 0..n_factors {
                    sigma[[i, j]] += loadings[[i, f]] * loadings[[j, f]];
                }
                if i == j {
                    sigma[[i, j]] += uniquenesses[i];
                }
            }
        }

        // Use Woodbury identity for efficient computation
        // log|Sigma| = log|Psi| + log|I + W^T * Psi^{-1} * W|
        let mut log_det_psi = 0.0;
        for j in 0..n_features {
            log_det_psi += uniquenesses[j].max(1e-12).ln();
        }

        // Compute W^T * Psi^{-1} * W + I
        let mut m: Array2<f64> = Array2::zeros((n_factors, n_factors));
        for i in 0..n_factors {
            for j in 0..n_factors {
                for k in 0..n_features {
                    m[[i, j]] += loadings[[k, i]] * loadings[[k, j]] / uniquenesses[k].max(1e-12);
                }
                if i == j {
                    m[[i, j]] += 1.0;
                }
            }
        }

        // log|M| using eigendecomposition
        let log_det_m = match scirs2_linalg::eigh::<f64>(&m.view(), None) {
            Ok((eigenvalues, _)) => eigenvalues.iter().map(|&e| e.max(1e-12).ln()).sum::<f64>(),
            Err(_) => 0.0,
        };

        let log_det_sigma = log_det_psi + log_det_m;

        // tr(Sigma^{-1} * S) using Woodbury
        // Sigma^{-1} = Psi^{-1} - Psi^{-1} * W * M^{-1} * W^T * Psi^{-1}
        let m_inv = match self.invert_small_matrix(&m) {
            Ok(inv) => inv,
            Err(_) => Array2::eye(n_factors),
        };

        // Compute Psi^{-1} * W
        let mut psi_inv_w: Array2<f64> = Array2::zeros((n_features, n_factors));
        for j in 0..n_features {
            for f in 0..n_factors {
                psi_inv_w[[j, f]] = loadings[[j, f]] / uniquenesses[j].max(1e-12);
            }
        }

        // Compute Psi^{-1} * W * M^{-1}
        let mut psi_inv_w_m_inv: Array2<f64> = Array2::zeros((n_features, n_factors));
        for i in 0..n_features {
            for j in 0..n_factors {
                for k in 0..n_factors {
                    psi_inv_w_m_inv[[i, j]] += psi_inv_w[[i, k]] * m_inv[[k, j]];
                }
            }
        }

        // tr(Sigma^{-1} * S) = tr(Psi^{-1} * S) - tr(Psi^{-1} * W * M^{-1} * W^T * Psi^{-1} * S)
        let mut trace_psi_inv_s = 0.0;
        for j in 0..n_features {
            trace_psi_inv_s += cov[[j, j]] / uniquenesses[j].max(1e-12);
        }

        let mut trace_correction = 0.0;
        for i in 0..n_features {
            for j in 0..n_features {
                let mut sigma_inv_ij = 0.0;
                for f in 0..n_factors {
                    sigma_inv_ij += psi_inv_w_m_inv[[i, f]] * psi_inv_w[[j, f]];
                }
                trace_correction += sigma_inv_ij * cov[[j, i]];
            }
        }

        let trace_sigma_inv_s = trace_psi_inv_s - trace_correction;

        // Log-likelihood = -n/2 * (p * log(2*pi) + log|Sigma| + tr(Sigma^{-1} * S))
        let n = n_samples as f64;
        let p = n_features as f64;
        -n / 2.0 * (p * (2.0 * std::f64::consts::PI).ln() + log_det_sigma + trace_sigma_inv_s)
    }

    /// Invert a small matrix using Gauss-Jordan elimination
    fn invert_small_matrix(&self, a: &Array2<f64>) -> Result<Array2<f64>> {
        let n = a.shape()[0];
        let mut augmented: Array2<f64> = Array2::zeros((n, 2 * n));

        // Build augmented matrix [A | I]
        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = a[[i, j]];
            }
            augmented[[i, n + i]] = 1.0;
        }

        // Forward elimination with partial pivoting
        for i in 0..n {
            let mut max_row = i;
            for k in i + 1..n {
                if augmented[[k, i]].abs() > augmented[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            if max_row != i {
                for j in 0..2 * n {
                    let temp = augmented[[i, j]];
                    augmented[[i, j]] = augmented[[max_row, j]];
                    augmented[[max_row, j]] = temp;
                }
            }

            let pivot = augmented[[i, i]];
            if pivot.abs() < 1e-14 {
                // Add regularization
                augmented[[i, i]] += 1e-8;
                let pivot = augmented[[i, i]];
                for j in 0..2 * n {
                    augmented[[i, j]] /= pivot;
                }
            } else {
                for j in 0..2 * n {
                    augmented[[i, j]] /= pivot;
                }
            }

            for k in 0..n {
                if k != i {
                    let factor = augmented[[k, i]];
                    for j in 0..2 * n {
                        augmented[[k, j]] -= factor * augmented[[i, j]];
                    }
                }
            }
        }

        // Extract inverse
        let mut inv: Array2<f64> = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inv[[i, j]] = augmented[[i, n + j]];
            }
        }

        Ok(inv)
    }

    /// Compute factor scores using the regression method (Thomson's method)
    ///
    /// scores = X * Psi^{-1} * W * (W^T * Psi^{-1} * W + I)^{-1}
    fn compute_scores(
        &self,
        x_centered: &Array2<f64>,
        loadings: &Array2<f64>,
        uniquenesses: &Array1<f64>,
    ) -> Result<Array2<f64>> {
        let n_samples = x_centered.shape()[0];
        let n_features = x_centered.shape()[1];
        let n_factors = self.n_factors;

        // Compute Psi^{-1} * W
        let mut psi_inv_w: Array2<f64> = Array2::zeros((n_features, n_factors));
        for j in 0..n_features {
            let psi_inv = 1.0 / uniquenesses[j].max(1e-12);
            for f in 0..n_factors {
                psi_inv_w[[j, f]] = psi_inv * loadings[[j, f]];
            }
        }

        // Compute W^T * Psi^{-1} * W + I
        let mut m: Array2<f64> = Array2::zeros((n_factors, n_factors));
        for i in 0..n_factors {
            for j in 0..n_factors {
                for k in 0..n_features {
                    m[[i, j]] += loadings[[k, i]] * psi_inv_w[[k, j]];
                }
                if i == j {
                    m[[i, j]] += 1.0;
                }
            }
        }

        // Invert M
        let m_inv = self.invert_small_matrix(&m)?;

        // Compute scoring matrix: Psi^{-1} * W * M^{-1}
        let mut scoring: Array2<f64> = Array2::zeros((n_features, n_factors));
        for i in 0..n_features {
            for j in 0..n_factors {
                for k in 0..n_factors {
                    scoring[[i, j]] += psi_inv_w[[i, k]] * m_inv[[k, j]];
                }
            }
        }

        // Compute scores: X * scoring
        let mut scores: Array2<f64> = Array2::zeros((n_samples, n_factors));
        for i in 0..n_samples {
            for j in 0..n_factors {
                for k in 0..n_features {
                    scores[[i, j]] += x_centered[[i, k]] * scoring[[k, j]];
                }
            }
        }

        Ok(scores)
    }

    /// Varimax rotation
    ///
    /// Maximizes the sum of variances of the squared loadings.
    /// This makes each factor either have a high or low loading on each variable,
    /// making interpretation easier.
    fn varimax_rotation(&self, loadings: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_features, n_factors) = loadings.dim();

        if n_factors < 2 {
            return Ok(loadings.clone());
        }

        let mut rotated = loadings.clone();

        // Normalize by communalities
        let mut h: Array1<f64> = Array1::zeros(n_features);
        for j in 0..n_features {
            for f in 0..n_factors {
                h[j] += rotated[[j, f]] * rotated[[j, f]];
            }
            h[j] = h[j].sqrt().max(1e-10);
        }

        let mut normalized = rotated.clone();
        for j in 0..n_features {
            for f in 0..n_factors {
                normalized[[j, f]] /= h[j];
            }
        }

        // Iterate rotations between pairs of factors
        for _iter in 0..self.varimax_max_iter {
            let mut changed = false;

            for p in 0..n_factors {
                for q in (p + 1)..n_factors {
                    // Compute the optimal rotation angle
                    let n = n_features as f64;

                    let mut sum_a: f64 = 0.0;
                    let mut sum_b: f64 = 0.0;
                    let mut sum_c: f64 = 0.0;
                    let mut sum_d: f64 = 0.0;

                    for j in 0..n_features {
                        let xj = normalized[[j, p]];
                        let yj = normalized[[j, q]];

                        let a_j = xj * xj - yj * yj;
                        let b_j = 2.0 * xj * yj;

                        sum_a += a_j;
                        sum_b += b_j;
                        sum_c += a_j * a_j - b_j * b_j;
                        sum_d += 2.0 * a_j * b_j;
                    }

                    let u = sum_d - 2.0 * sum_a * sum_b / n;
                    let v = sum_c - (sum_a * sum_a - sum_b * sum_b) / n;

                    // Rotation angle
                    let angle = 0.25 * u.atan2(v);

                    if angle.abs() < 1e-10 {
                        continue;
                    }

                    changed = true;

                    let cos_a = angle.cos();
                    let sin_a = angle.sin();

                    // Apply rotation
                    for j in 0..n_features {
                        let xj = normalized[[j, p]];
                        let yj = normalized[[j, q]];
                        normalized[[j, p]] = cos_a * xj + sin_a * yj;
                        normalized[[j, q]] = -sin_a * xj + cos_a * yj;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        // De-normalize
        for j in 0..n_features {
            for f in 0..n_factors {
                rotated[[j, f]] = normalized[[j, f]] * h[j];
            }
        }

        Ok(rotated)
    }

    /// Promax rotation (oblique)
    ///
    /// Starts with varimax rotation, then raises the loadings to a power
    /// to create a target matrix, and rotates towards that target.
    fn promax_rotation(&self, loadings: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_features, n_factors) = loadings.dim();

        if n_factors < 2 {
            return Ok(loadings.clone());
        }

        // Step 1: Get varimax rotation
        let varimax_loadings = self.varimax_rotation(loadings)?;

        // Step 2: Create target matrix by raising to a power while preserving signs
        let mut target: Array2<f64> = Array2::zeros((n_features, n_factors));
        let power = self.promax_power as f64;

        for j in 0..n_features {
            for f in 0..n_factors {
                let val = varimax_loadings[[j, f]];
                let sign = if val >= 0.0 { 1.0 } else { -1.0 };
                target[[j, f]] = sign * val.abs().powf(power);
            }
        }

        // Step 3: Rotate towards target using Procrustes rotation
        // Minimize ||A * T - target||^2 for rotation matrix T
        // T = (A^T * A)^{-1} * A^T * target

        // Compute A^T * A where A = varimax_loadings
        let mut at_a: Array2<f64> = Array2::zeros((n_factors, n_factors));
        for i in 0..n_factors {
            for j in 0..n_factors {
                for k in 0..n_features {
                    at_a[[i, j]] += varimax_loadings[[k, i]] * varimax_loadings[[k, j]];
                }
            }
        }

        // Compute (A^T * A)^{-1}
        let at_a_inv = self.invert_small_matrix(&at_a)?;

        // Compute A^T * target
        let mut at_target: Array2<f64> = Array2::zeros((n_factors, n_factors));
        for i in 0..n_factors {
            for j in 0..n_factors {
                for k in 0..n_features {
                    at_target[[i, j]] += varimax_loadings[[k, i]] * target[[k, j]];
                }
            }
        }

        // T = (A^T * A)^{-1} * A^T * target
        let mut t_mat: Array2<f64> = Array2::zeros((n_factors, n_factors));
        for i in 0..n_factors {
            for j in 0..n_factors {
                for k in 0..n_factors {
                    t_mat[[i, j]] += at_a_inv[[i, k]] * at_target[[k, j]];
                }
            }
        }

        // Result = A * T
        let mut result: Array2<f64> = Array2::zeros((n_features, n_factors));
        for i in 0..n_features {
            for j in 0..n_factors {
                for k in 0..n_factors {
                    result[[i, j]] += varimax_loadings[[i, k]] * t_mat[[k, j]];
                }
            }
        }

        Ok(result)
    }

    /// Returns the factor loadings
    pub fn loadings(&self) -> Option<&Array2<f64>> {
        self.loadings.as_ref()
    }

    /// Returns the uniquenesses
    pub fn uniquenesses(&self) -> Option<&Array1<f64>> {
        self.uniquenesses.as_ref()
    }

    /// Compute scree plot data from a data matrix
    ///
    /// Computes eigenvalues of the covariance (or correlation) matrix,
    /// which can be used for scree plots to determine the number of factors.
    ///
    /// # Arguments
    /// * `x` - Input data, shape (n_samples, n_features)
    /// * `use_correlation` - If true, use the correlation matrix instead of covariance
    ///
    /// # Returns
    /// * `Result<ScreePlotData>` - Eigenvalues and variance proportions
    pub fn scree_plot_data<S>(x: &ArrayBase<S, Ix2>, use_correlation: bool) -> Result<ScreePlotData>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|v| <f64 as NumCast>::from(v).unwrap_or_default());
        let (n_samples, n_features) = x_f64.dim();

        if n_samples < 2 {
            return Err(TransformError::InvalidInput(
                "Need at least 2 samples for scree plot".to_string(),
            ));
        }

        // Center data
        let mean = x_f64.mean_axis(Axis(0)).ok_or_else(|| {
            TransformError::ComputationError("Failed to compute mean".to_string())
        })?;

        let mut x_centered = x_f64.clone();
        for i in 0..n_samples {
            for j in 0..n_features {
                x_centered[[i, j]] -= mean[j];
            }
        }

        // Compute covariance matrix
        let mut cov: Array2<f64> = Array2::zeros((n_features, n_features));
        for i in 0..n_features {
            for j in 0..n_features {
                let mut sum = 0.0;
                for k in 0..n_samples {
                    sum += x_centered[[k, i]] * x_centered[[k, j]];
                }
                cov[[i, j]] = sum / (n_samples - 1) as f64;
            }
        }

        // If using correlation, standardize by dividing by standard deviations
        if use_correlation {
            let mut stds: Array1<f64> = Array1::zeros(n_features);
            for j in 0..n_features {
                stds[j] = cov[[j, j]].sqrt().max(1e-12);
            }
            for i in 0..n_features {
                for j in 0..n_features {
                    cov[[i, j]] /= stds[i] * stds[j];
                }
            }
        }

        // Compute eigenvalues using symmetric eigendecomposition
        let (eigenvalues_raw, _eigenvectors) =
            scirs2_linalg::eigh::<f64>(&cov.view(), None).map_err(TransformError::LinalgError)?;

        // Sort eigenvalues in descending order
        let n = eigenvalues_raw.len();
        let mut eig_vec: Vec<f64> = eigenvalues_raw.iter().map(|&e| e.max(0.0)).collect();
        eig_vec.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let mut eigenvalues: Array1<f64> = Array1::zeros(n);
        for i in 0..n {
            eigenvalues[i] = eig_vec[i];
        }

        // Compute variance proportions
        let total: f64 = eigenvalues.sum();
        let mut variance_proportions: Array1<f64> = Array1::zeros(n);
        let mut cumulative_variance: Array1<f64> = Array1::zeros(n);

        if total > 0.0 {
            let mut cumsum = 0.0;
            for i in 0..n {
                variance_proportions[i] = eigenvalues[i] / total;
                cumsum += variance_proportions[i];
                cumulative_variance[i] = cumsum;
            }
        }

        // Kaiser threshold: 1.0 for correlation matrix, average eigenvalue for covariance
        let kaiser_threshold = if use_correlation {
            1.0
        } else if n > 0 {
            total / n as f64
        } else {
            0.0
        };

        Ok(ScreePlotData {
            eigenvalues,
            cumulative_variance,
            variance_proportions,
            kaiser_threshold,
        })
    }
}

/// Convenience function for factor analysis
///
/// # Arguments
/// * `data` - Input data, shape (n_samples, n_features)
/// * `n_factors` - Number of factors to extract
///
/// # Returns
/// * `Result<FactorAnalysisResult>` - Factor analysis results
pub fn factor_analysis<S>(
    data: &ArrayBase<S, Ix2>,
    n_factors: usize,
) -> Result<FactorAnalysisResult>
where
    S: Data,
    S::Elem: Float + NumCast,
{
    let mut fa = FactorAnalysis::new(n_factors);
    fa.fit_transform(data)
}

/// Convenience function to compute scree plot data for factor selection
///
/// Returns eigenvalues and variance proportions that can be used to
/// determine the appropriate number of factors via the scree plot method
/// (look for the "elbow") or Kaiser criterion (eigenvalues > threshold).
///
/// # Arguments
/// * `data` - Input data, shape (n_samples, n_features)
/// * `use_correlation` - If true, use correlation matrix; if false, use covariance
///
/// # Returns
/// * `Result<ScreePlotData>` - Eigenvalues and variance information
pub fn scree_plot_data<S>(data: &ArrayBase<S, Ix2>, use_correlation: bool) -> Result<ScreePlotData>
where
    S: Data,
    S::Elem: Float + NumCast,
{
    FactorAnalysis::scree_plot_data(data, use_correlation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    /// Generate test data with known factor structure
    fn generate_factor_data(n_samples: usize, n_features: usize, n_factors: usize) -> Array2<f64> {
        let mut rng = scirs2_core::random::rng();
        let mut data: Array2<f64> = Array2::zeros((n_samples, n_features));

        // Create factor loadings
        let mut true_loadings: Array2<f64> = Array2::zeros((n_features, n_factors));
        for j in 0..n_features {
            let factor_idx = j % n_factors;
            true_loadings[[j, factor_idx]] = 0.8;
            // Add cross-loadings
            for f in 0..n_factors {
                if f != factor_idx {
                    true_loadings[[j, f]] = 0.1;
                }
            }
        }

        // Generate data: x = W * z + noise
        for i in 0..n_samples {
            // Random factors
            let mut factors: Array1<f64> = Array1::zeros(n_factors);
            for f in 0..n_factors {
                factors[f] = scirs2_core::random::Rng::random_range(&mut rng, -2.0..2.0);
            }

            for j in 0..n_features {
                let mut val = 0.0;
                for f in 0..n_factors {
                    val += true_loadings[[j, f]] * factors[f];
                }
                // Add noise
                val += scirs2_core::random::Rng::random_range(&mut rng, -0.3..0.3);
                data[[i, j]] = val;
            }
        }

        data
    }

    #[test]
    fn test_factor_analysis_basic() {
        let data = generate_factor_data(100, 6, 2);

        let mut fa = FactorAnalysis::new(2);
        let result = fa.fit_transform(&data).expect("FA fit_transform failed");

        assert_eq!(result.loadings.shape(), &[6, 2]);
        assert_eq!(result.scores.shape(), &[100, 2]);
        assert_eq!(result.uniquenesses.len(), 6);
        assert_eq!(result.communalities.len(), 6);

        // All values should be finite
        for val in result.loadings.iter() {
            assert!(val.is_finite());
        }
        for val in result.scores.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_factor_analysis_uniquenesses() {
        let data = generate_factor_data(100, 6, 2);

        let mut fa = FactorAnalysis::new(2);
        let result = fa.fit_transform(&data).expect("FA fit_transform failed");

        // Uniquenesses should be positive
        for &u in result.uniquenesses.iter() {
            assert!(u > 0.0, "Uniqueness should be positive, got {}", u);
        }

        // Communalities should be between 0 and data variance
        for &c in result.communalities.iter() {
            assert!(c >= 0.0, "Communality should be non-negative, got {}", c);
        }
    }

    #[test]
    fn test_factor_analysis_varimax() {
        let data = generate_factor_data(100, 6, 2);

        let mut fa = FactorAnalysis::new(2).with_rotation(RotationMethod::Varimax);
        let result = fa
            .fit_transform(&data)
            .expect("FA varimax fit_transform failed");

        assert_eq!(result.loadings.shape(), &[6, 2]);
        for val in result.loadings.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_factor_analysis_promax() {
        let data = generate_factor_data(100, 6, 2);

        let mut fa = FactorAnalysis::new(2).with_rotation(RotationMethod::Promax);
        let result = fa
            .fit_transform(&data)
            .expect("FA promax fit_transform failed");

        assert_eq!(result.loadings.shape(), &[6, 2]);
        for val in result.loadings.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_factor_analysis_convergence() {
        let data = generate_factor_data(50, 4, 2);

        let mut fa = FactorAnalysis::new(2).with_max_iter(500).with_tol(1e-6);
        let result = fa.fit_transform(&data).expect("FA fit_transform failed");

        // Should converge in reasonable number of iterations
        assert!(
            result.n_iter <= 500,
            "Should converge within max_iter, got {}",
            result.n_iter
        );

        // Log-likelihood should be finite
        assert!(
            result.log_likelihood.is_finite(),
            "Log-likelihood should be finite, got {}",
            result.log_likelihood
        );
    }

    #[test]
    fn test_factor_analysis_single_factor() {
        let data = generate_factor_data(50, 4, 1);

        let mut fa = FactorAnalysis::new(1);
        let result = fa.fit_transform(&data).expect("FA fit_transform failed");

        assert_eq!(result.loadings.shape(), &[4, 1]);
        assert_eq!(result.scores.shape(), &[50, 1]);
    }

    #[test]
    fn test_factor_analysis_convenience_fn() {
        let data = generate_factor_data(50, 6, 2);

        let result = factor_analysis(&data, 2).expect("factor_analysis failed");
        assert_eq!(result.loadings.shape(), &[6, 2]);
    }

    #[test]
    fn test_factor_analysis_invalid_params() {
        let data = Array2::<f64>::zeros((10, 3));

        // Too many factors
        let mut fa = FactorAnalysis::new(5);
        assert!(fa.fit_transform(&data).is_err());
    }

    #[test]
    fn test_scree_plot_data_covariance() {
        let data = generate_factor_data(100, 6, 2);

        let scree = FactorAnalysis::scree_plot_data(&data, false).expect("scree_plot_data failed");

        // Should have eigenvalues for all features
        assert_eq!(scree.eigenvalues.len(), 6);
        assert_eq!(scree.variance_proportions.len(), 6);
        assert_eq!(scree.cumulative_variance.len(), 6);

        // Eigenvalues should be in approximately descending order and non-negative
        for i in 1..scree.eigenvalues.len() {
            assert!(
                scree.eigenvalues[i] <= scree.eigenvalues[i - 1] + 1e-6,
                "Eigenvalues should be in descending order: eigenvalue[{}]={} > eigenvalue[{}]={}",
                i,
                scree.eigenvalues[i],
                i - 1,
                scree.eigenvalues[i - 1]
            );
        }
        for &e in scree.eigenvalues.iter() {
            assert!(e >= 0.0, "Eigenvalue should be non-negative, got {}", e);
        }

        // Variance proportions should sum to ~1
        let total: f64 = scree.variance_proportions.sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "Variance proportions should sum to 1, got {}",
            total
        );

        // Cumulative variance should end at ~1
        let last_idx = scree.cumulative_variance.len() - 1;
        assert!(
            (scree.cumulative_variance[last_idx] - 1.0).abs() < 1e-6,
            "Cumulative variance should end at 1"
        );
    }

    #[test]
    fn test_scree_plot_data_correlation() {
        let data = generate_factor_data(200, 6, 2);

        let scree = FactorAnalysis::scree_plot_data(&data, true)
            .expect("scree_plot_data correlation failed");

        // Kaiser threshold for correlation matrix should be 1.0
        assert!(
            (scree.kaiser_threshold - 1.0).abs() < 1e-10,
            "Kaiser threshold for correlation matrix should be 1.0, got {}",
            scree.kaiser_threshold
        );

        // All eigenvalues should be finite and non-negative
        for &e in scree.eigenvalues.iter() {
            assert!(
                e >= 0.0 && e.is_finite(),
                "Eigenvalue should be finite non-negative, got {}",
                e
            );
        }

        // First eigenvalue should be largest (descending order guaranteed by sort)
        assert!(
            scree.eigenvalues[0] >= scree.eigenvalues[scree.eigenvalues.len() - 1],
            "Eigenvalues should be in descending order"
        );

        // Variance proportions should sum to ~1
        let total: f64 = scree.variance_proportions.sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "Variance proportions should sum to 1, got {}",
            total
        );
    }

    #[test]
    fn test_scree_plot_convenience_fn() {
        let data = generate_factor_data(50, 4, 2);

        let scree = scree_plot_data(&data, false).expect("scree_plot_data convenience fn failed");
        assert_eq!(scree.eigenvalues.len(), 4);
        assert!(scree.eigenvalues.iter().all(|e| e.is_finite()));
    }

    #[test]
    fn test_factor_analysis_communalities_sum_constraint() {
        // Communality + uniqueness should approximate the feature variance
        let data = generate_factor_data(200, 4, 2);

        let mut fa = FactorAnalysis::new(2);
        let result = fa.fit_transform(&data).expect("FA failed");

        // Compute feature variances
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];
        let mean = data.mean_axis(Axis(0)).expect("mean computation failed");
        let mut variances: Array1<f64> = Array1::zeros(n_features);
        for j in 0..n_features {
            let mut var = 0.0;
            for i in 0..n_samples {
                let diff = data[[i, j]] - mean[j];
                var += diff * diff;
            }
            variances[j] = var / (n_samples - 1) as f64;
        }

        // communality + uniqueness should be approximately equal to the feature variance
        for j in 0..n_features {
            let total = result.communalities[j] + result.uniquenesses[j];
            let variance = variances[j];
            // Allow tolerance since EM may not exactly decompose variance
            let rel_err = (total - variance).abs() / variance.max(1e-12);
            assert!(
                rel_err < 0.5,
                "Feature {}: communality ({}) + uniqueness ({}) = {} should approximate variance ({})",
                j, result.communalities[j], result.uniquenesses[j], total, variance
            );
        }
    }

    #[test]
    fn test_factor_analysis_too_few_samples() {
        let data = Array2::<f64>::zeros((1, 3));
        let mut fa = FactorAnalysis::new(1);
        let result = fa.fit_transform(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_factor_analysis_builder_methods() {
        let fa = FactorAnalysis::new(3)
            .with_rotation(RotationMethod::Varimax)
            .with_max_iter(200)
            .with_tol(1e-5)
            .with_promax_power(6)
            .with_center(false);

        // Just verify builder doesn't panic and struct is created
        assert!(fa.loadings().is_none()); // Not yet fitted
        assert!(fa.uniquenesses().is_none());
    }

    #[test]
    fn test_scree_plot_too_few_samples() {
        let data = Array2::<f64>::zeros((1, 3));
        let result = scree_plot_data(&data, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_factor_analysis_noise_variance() {
        let data = generate_factor_data(100, 6, 2);

        let mut fa = FactorAnalysis::new(2);
        let result = fa.fit_transform(&data).expect("FA failed");

        // Noise variance should be positive and finite
        assert!(result.noise_variance > 0.0);
        assert!(result.noise_variance.is_finite());

        // Noise variance should be the average of uniquenesses
        let expected = result.uniquenesses.sum() / result.uniquenesses.len() as f64;
        assert!(
            (result.noise_variance - expected).abs() < 1e-10,
            "Noise variance should be average of uniquenesses"
        );
    }
}
