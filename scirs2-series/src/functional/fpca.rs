//! Functional Principal Component Analysis (FPCA)
//!
//! Computes functional principal components by eigendecomposition of
//! the empirical covariance operator, represented in a basis function space.
//!
//! # Methods
//!
//! - [`FPCA`]: Standard FPCA via basis representation and eigen-decomposition
//! - [`BivariateFPCA`]: FPCA for bivariate functional data
//! - [`MultilevelFPCA`]: Multilevel FPCA decomposing between-/within-subject variation
//! - [`VarianceExplained`]: Cumulative variance explanation utility

use crate::error::{Result, TimeSeriesError};
use crate::functional::basis::{evaluate_basis_matrix, BasisSystem, BSplineBasis};
use crate::functional::smoothing::{FunctionalData, PenalizedLeastSquares, solve_linear_system};
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_linalg::eigh;

// ============================================================
// VarianceExplained
// ============================================================

/// Cumulative variance explanation summary for FPCA.
#[derive(Debug, Clone)]
pub struct VarianceExplained {
    /// Eigenvalues (sorted descending)
    pub eigenvalues: Array1<f64>,
    /// Proportion of variance explained by each component
    pub proportion: Array1<f64>,
    /// Cumulative proportion of variance explained
    pub cumulative: Array1<f64>,
}

impl VarianceExplained {
    /// Compute from a set of eigenvalues
    pub fn from_eigenvalues(eigenvalues: &Array1<f64>) -> Self {
        let total: f64 = eigenvalues.iter().map(|&v| v.max(0.0)).sum();
        let k = eigenvalues.len();
        let mut proportion = Array1::zeros(k);
        let mut cumulative = Array1::zeros(k);
        if total > 0.0 {
            let mut cum = 0.0;
            for (i, &ev) in eigenvalues.iter().enumerate() {
                let p = ev.max(0.0) / total;
                proportion[i] = p;
                cum += p;
                cumulative[i] = cum;
            }
        }
        Self {
            eigenvalues: eigenvalues.clone(),
            proportion,
            cumulative,
        }
    }

    /// Number of components needed to explain at least `threshold` (0..1) of variance
    pub fn n_components_for_threshold(&self, threshold: f64) -> usize {
        for (i, &c) in self.cumulative.iter().enumerate() {
            if c >= threshold {
                return i + 1;
            }
        }
        self.eigenvalues.len()
    }
}

// ============================================================
// FPCAResult
// ============================================================

/// Result of functional PCA
#[derive(Debug, Clone)]
pub struct FPCAResult<B: BasisSystem + Clone> {
    /// Eigenvalues of the covariance operator (variance explained per component)
    pub eigenvalues: Array1<f64>,
    /// Functional principal component functions (eigenfunctions)
    pub eigenfunctions: Vec<FunctionalData<B>>,
    /// Scores matrix: (n_obs × n_components)
    /// score[i, k] = ⟨x_i - mean, ξ_k⟩
    pub scores: Array2<f64>,
    /// Mean function coefficients
    pub mean_coefficients: Array1<f64>,
    /// Variance explained summary
    pub variance_explained: VarianceExplained,
    /// Number of components retained
    pub n_components: usize,
}

impl<B: BasisSystem + Clone> FPCAResult<B> {
    /// Reconstruct the i-th observation using the first `n_comp` components
    pub fn reconstruct(
        &self,
        obs_idx: usize,
        n_comp: usize,
        basis: &B,
    ) -> Result<FunctionalData<B>> {
        if obs_idx >= self.scores.nrows() {
            return Err(TimeSeriesError::InvalidInput(format!(
                "obs_idx {} out of range (n_obs={})",
                obs_idx,
                self.scores.nrows()
            )));
        }
        let n_comp = n_comp.min(self.n_components);
        let k = basis.n_basis();
        let mut coeff = self.mean_coefficients.clone();
        for c in 0..n_comp {
            let score = self.scores[[obs_idx, c]];
            let ef_coeff = &self.eigenfunctions[c].coefficients;
            for j in 0..k {
                coeff[j] += score * ef_coeff[j];
            }
        }
        FunctionalData::new(basis.clone(), coeff)
    }

    /// Project new observations onto principal components
    /// `x_new`: (n_new × n_basis) coefficient matrix (already in basis representation)
    pub fn transform(&self, x_new_coeff: &Array2<f64>) -> Result<Array2<f64>> {
        let n_new = x_new_coeff.nrows();
        let k = x_new_coeff.ncols();
        if k != self.mean_coefficients.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.mean_coefficients.len(),
                actual: k,
            });
        }
        let mut scores = Array2::zeros((n_new, self.n_components));
        for i in 0..n_new {
            let centered: Vec<f64> = (0..k)
                .map(|j| x_new_coeff[[i, j]] - self.mean_coefficients[j])
                .collect();
            let centered_arr = Array1::from(centered);
            for c in 0..self.n_components {
                scores[[i, c]] = centered_arr.dot(&self.eigenfunctions[c].coefficients);
            }
        }
        Ok(scores)
    }
}

// ============================================================
// FPCA: Standard Functional PCA
// ============================================================

/// Configuration for Functional PCA
#[derive(Debug, Clone)]
pub struct FPCAConfig {
    /// Maximum number of principal components (None = keep all with positive eigenvalue)
    pub n_components: Option<usize>,
    /// Smoothing parameter for individual curve estimation (None = GCV)
    pub smooth_lambda: Option<f64>,
    /// Number of interior knots for B-spline representation
    pub n_interior_knots: usize,
    /// B-spline order
    pub spline_order: usize,
    /// Regularization for covariance matrix (ridge penalty to avoid singularity)
    pub cov_regularization: f64,
    /// Whether to center observations before decomposition
    pub center: bool,
}

impl Default for FPCAConfig {
    fn default() -> Self {
        Self {
            n_components: None,
            smooth_lambda: None,
            n_interior_knots: 15,
            spline_order: 4,
            cov_regularization: 1e-8,
            center: true,
        }
    }
}

/// Functional Principal Component Analysis (FPCA)
///
/// Given a sample of functional observations {x_i(t)}, computes the
/// functional principal components (Karhunen-Loève expansion) by:
///
/// 1. Smoothing each curve in a basis space to obtain coefficients c_i
/// 2. Computing the sample covariance matrix in basis space: Σ_c = (1/(n-1)) C^T C
/// 3. Solving the generalized eigenvalue problem: Σ_c G v = λ v
///    where G is the Gram matrix of the basis
/// 4. Transforming eigenvectors back to functional eigenfunctions
#[derive(Debug, Clone)]
pub struct FPCA {
    /// Configuration
    pub config: FPCAConfig,
}

impl FPCA {
    /// Create FPCA with default configuration
    pub fn new() -> Self {
        Self {
            config: FPCAConfig::default(),
        }
    }

    /// Create FPCA with custom configuration
    pub fn with_config(config: FPCAConfig) -> Self {
        Self { config }
    }

    /// Fit FPCA to a collection of discrete functional observations.
    ///
    /// # Arguments
    /// - `t_list`: observation times for each curve (can differ between curves)
    /// - `y_list`: observed values for each curve
    ///
    /// Returns an [`FPCAResult`] containing eigenfunctions, scores, and eigenvalues.
    pub fn fit(
        &self,
        t_list: &[Array1<f64>],
        y_list: &[Array1<f64>],
    ) -> Result<FPCAResult<BSplineBasis>> {
        let n = t_list.len();
        if n < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "FPCA requires at least 2 observations".to_string(),
                required: 2,
                actual: n,
            });
        }
        if y_list.len() != n {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: y_list.len(),
            });
        }

        // Build common basis
        let basis = BSplineBasis::uniform(self.config.n_interior_knots, self.config.spline_order)?;
        let k = basis.n_basis();

        // Step 1: Smooth each curve to get basis coefficients
        let pls = PenalizedLeastSquares {
            lambda: self.config.smooth_lambda,
            penalty_order: 2,
            n_lambda_grid: 30,
            lambda_min: 1e-10,
            lambda_max: 1e4,
        };
        let mut coeff_matrix = Array2::zeros((n, k));
        for (i, (t, y)) in t_list.iter().zip(y_list.iter()).enumerate() {
            let fd = pls.fit(basis.clone(), t, y)?;
            for j in 0..k {
                coeff_matrix[[i, j]] = fd.coefficients[j];
            }
        }

        // Step 2: Compute mean coefficient vector
        let mut mean_coeff = Array1::zeros(k);
        if self.config.center {
            for j in 0..k {
                let col_sum: f64 = (0..n).map(|i| coeff_matrix[[i, j]]).sum();
                mean_coeff[j] = col_sum / n as f64;
            }
        }

        // Step 3: Center the coefficient matrix
        let mut c_centered = coeff_matrix.clone();
        if self.config.center {
            for i in 0..n {
                for j in 0..k {
                    c_centered[[i, j]] -= mean_coeff[j];
                }
            }
        }

        // Step 4: Sample covariance in basis space (k × k)
        // Σ_c = (1/(n-1)) C^T C
        let n_f = if n > 1 { (n - 1) as f64 } else { 1.0 };
        let cov = c_centered.t().dot(&c_centered) / n_f;

        // Step 5: Gram matrix of basis
        let gram = basis.gram_matrix()?;

        // Step 6: Solve generalized eigenvalue problem Σ_c G v = λ v
        // Equivalent: G^{1/2} Σ_c G^{1/2} u = λ u, v = G^{-1/2} u
        // For simplicity, use regularized approach: eigendecompose G^{1/2} Σ_c G^{1/2}
        let eig_result = solve_generalized_eig(&cov, &gram, self.config.cov_regularization)?;
        let (eigenvalues_all, eigenvecs_all) = eig_result;

        // Step 7: Determine number of components to retain
        let n_comp = match self.config.n_components {
            Some(nc) => nc.min(eigenvalues_all.len()),
            None => eigenvalues_all
                .iter()
                .filter(|&&ev| ev > 0.0)
                .count()
                .max(1),
        };

        // Step 8: Extract eigenfunctions
        let mut eigenfunctions = Vec::with_capacity(n_comp);
        for c in 0..n_comp {
            let eig_coeff = eigenvecs_all.column(c).to_owned();
            // Normalize in L2: ||ξ||^2 = v^T G v
            let norm_sq = eig_coeff.dot(&gram.dot(&eig_coeff));
            let norm = norm_sq.max(0.0).sqrt();
            let normalized = if norm > 1e-15 {
                eig_coeff.mapv(|x| x / norm)
            } else {
                eig_coeff
            };
            eigenfunctions.push(FunctionalData::new(basis.clone(), normalized)?);
        }

        // Step 9: Compute scores: s_{i,c} = c_i^T G ξ_c (in basis space)
        let mut scores = Array2::zeros((n, n_comp));
        for i in 0..n {
            let ci: Array1<f64> = c_centered.row(i).to_owned();
            let g_ci = gram.dot(&ci);
            for c in 0..n_comp {
                scores[[i, c]] = g_ci.dot(&eigenfunctions[c].coefficients);
            }
        }

        let eigenvalues = Array1::from_vec(eigenvalues_all[..n_comp].to_vec());
        let variance_explained = VarianceExplained::from_eigenvalues(&eigenvalues);

        Ok(FPCAResult {
            eigenvalues,
            eigenfunctions,
            scores,
            mean_coefficients: mean_coeff,
            variance_explained,
            n_components: n_comp,
        })
    }
}

impl Default for FPCA {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// BivariateFPCA
// ============================================================

/// Result of bivariate FPCA
#[derive(Debug, Clone)]
pub struct BivariateFPCAResult<B: BasisSystem + Clone> {
    /// Eigenvalues
    pub eigenvalues: Array1<f64>,
    /// Eigenfunctions for the first component of the bivariate process
    pub eigenfunctions_1: Vec<FunctionalData<B>>,
    /// Eigenfunctions for the second component of the bivariate process
    pub eigenfunctions_2: Vec<FunctionalData<B>>,
    /// Scores: (n_obs × n_components)
    pub scores: Array2<f64>,
    /// Mean coefficients for component 1
    pub mean_coeff_1: Array1<f64>,
    /// Mean coefficients for component 2
    pub mean_coeff_2: Array1<f64>,
    /// Variance explained
    pub variance_explained: VarianceExplained,
    /// Number of components
    pub n_components: usize,
}

/// Bivariate Functional PCA for paired functional observations (x(t), y(t)).
///
/// Simultaneously diagonalizes the joint covariance operator of two functional
/// processes, finding directions of maximal joint variation.
#[derive(Debug, Clone)]
pub struct BivariateFPCA {
    /// Configuration (shared between both components)
    pub config: FPCAConfig,
}

impl BivariateFPCA {
    /// Create a new BivariateFPCA
    pub fn new() -> Self {
        Self {
            config: FPCAConfig::default(),
        }
    }

    /// Fit bivariate FPCA to paired observations.
    ///
    /// # Arguments
    /// - `t_list_1`, `y_list_1`: times and values for first process
    /// - `t_list_2`, `y_list_2`: times and values for second process
    pub fn fit(
        &self,
        t_list_1: &[Array1<f64>],
        y_list_1: &[Array1<f64>],
        t_list_2: &[Array1<f64>],
        y_list_2: &[Array1<f64>],
    ) -> Result<BivariateFPCAResult<BSplineBasis>> {
        let n = t_list_1.len();
        if n < 2 || t_list_2.len() != n {
            return Err(TimeSeriesError::InsufficientData {
                message: "BivariateFPCA requires at least 2 paired observations".to_string(),
                required: 2,
                actual: n,
            });
        }

        let basis = BSplineBasis::uniform(self.config.n_interior_knots, self.config.spline_order)?;
        let k = basis.n_basis();

        let pls = PenalizedLeastSquares {
            lambda: self.config.smooth_lambda,
            penalty_order: 2,
            n_lambda_grid: 30,
            lambda_min: 1e-10,
            lambda_max: 1e4,
        };

        // Smooth both sets of curves
        let mut coeff1 = Array2::zeros((n, k));
        let mut coeff2 = Array2::zeros((n, k));
        for i in 0..n {
            let fd1 = pls.fit(basis.clone(), &t_list_1[i], &y_list_1[i])?;
            let fd2 = pls.fit(basis.clone(), &t_list_2[i], &y_list_2[i])?;
            for j in 0..k {
                coeff1[[i, j]] = fd1.coefficients[j];
                coeff2[[i, j]] = fd2.coefficients[j];
            }
        }

        // Center both
        let mut mean1 = Array1::zeros(k);
        let mut mean2 = Array1::zeros(k);
        for j in 0..k {
            mean1[j] = (0..n).map(|i| coeff1[[i, j]]).sum::<f64>() / n as f64;
            mean2[j] = (0..n).map(|i| coeff2[[i, j]]).sum::<f64>() / n as f64;
        }
        for i in 0..n {
            for j in 0..k {
                coeff1[[i, j]] -= mean1[j];
                coeff2[[i, j]] -= mean2[j];
            }
        }

        // Build block covariance matrix (2k × 2k)
        let n_f = if n > 1 { (n - 1) as f64 } else { 1.0 };
        // [C11, C12; C21, C22] where Cij = (1/(n-1)) Xi^T Xj
        let c11 = coeff1.t().dot(&coeff1) / n_f;
        let c12 = coeff1.t().dot(&coeff2) / n_f;
        let c21 = coeff2.t().dot(&coeff1) / n_f;
        let c22 = coeff2.t().dot(&coeff2) / n_f;

        let mut block_cov = Array2::zeros((2 * k, 2 * k));
        for i in 0..k {
            for j in 0..k {
                block_cov[[i, j]] = c11[[i, j]];
                block_cov[[i, j + k]] = c12[[i, j]];
                block_cov[[i + k, j]] = c21[[i, j]];
                block_cov[[i + k, j + k]] = c22[[i, j]];
            }
        }

        // Block Gram matrix
        let gram = basis.gram_matrix()?;
        let mut block_gram = Array2::zeros((2 * k, 2 * k));
        for i in 0..k {
            for j in 0..k {
                block_gram[[i, j]] = gram[[i, j]];
                block_gram[[i + k, j + k]] = gram[[i, j]];
            }
        }

        let (eigenvalues_all, eigenvecs_all) =
            solve_generalized_eig(&block_cov, &block_gram, self.config.cov_regularization)?;

        let n_comp = match self.config.n_components {
            Some(nc) => nc.min(eigenvalues_all.len()),
            None => eigenvalues_all.iter().filter(|&&ev| ev > 0.0).count().max(1),
        };

        let mut eigenfunctions_1 = Vec::with_capacity(n_comp);
        let mut eigenfunctions_2 = Vec::with_capacity(n_comp);
        for c in 0..n_comp {
            let ev = eigenvecs_all.column(c).to_owned();
            let v1: Array1<f64> = ev.slice(s![..k]).to_owned();
            let v2: Array1<f64> = ev.slice(s![k..]).to_owned();

            // Normalize joint eigenfunction
            let norm1_sq = v1.dot(&gram.dot(&v1));
            let norm2_sq = v2.dot(&gram.dot(&v2));
            let joint_norm = (norm1_sq + norm2_sq).sqrt().max(1e-15);
            let nv1 = v1.mapv(|x| x / joint_norm);
            let nv2 = v2.mapv(|x| x / joint_norm);
            eigenfunctions_1.push(FunctionalData::new(basis.clone(), nv1)?);
            eigenfunctions_2.push(FunctionalData::new(basis.clone(), nv2)?);
        }

        // Compute bivariate scores
        let mut scores = Array2::zeros((n, n_comp));
        for i in 0..n {
            let ci1: Array1<f64> = coeff1.row(i).to_owned();
            let ci2: Array1<f64> = coeff2.row(i).to_owned();
            let gci1 = gram.dot(&ci1);
            let gci2 = gram.dot(&ci2);
            for c in 0..n_comp {
                let s1 = gci1.dot(&eigenfunctions_1[c].coefficients);
                let s2 = gci2.dot(&eigenfunctions_2[c].coefficients);
                scores[[i, c]] = s1 + s2;
            }
        }

        let eigenvalues = Array1::from_vec(eigenvalues_all[..n_comp].to_vec());
        let variance_explained = VarianceExplained::from_eigenvalues(&eigenvalues);

        Ok(BivariateFPCAResult {
            eigenvalues,
            eigenfunctions_1,
            eigenfunctions_2,
            scores,
            mean_coeff_1: mean1,
            mean_coeff_2: mean2,
            variance_explained,
            n_components: n_comp,
        })
    }
}

impl Default for BivariateFPCA {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// MultilevelFPCA
// ============================================================

/// Result of multilevel FPCA
#[derive(Debug, Clone)]
pub struct MultilevelFPCAResult<B: BasisSystem + Clone> {
    /// Between-subject eigenvalues (level 1)
    pub eigenvalues_between: Array1<f64>,
    /// Within-subject eigenvalues (level 2)
    pub eigenvalues_within: Array1<f64>,
    /// Between-subject eigenfunctions (level 1)
    pub eigenfunctions_between: Vec<FunctionalData<B>>,
    /// Within-subject eigenfunctions (level 2)
    pub eigenfunctions_within: Vec<FunctionalData<B>>,
    /// Between-subject scores (n_subjects × n_comp_between)
    pub scores_between: Array2<f64>,
    /// Within-subject scores (total_obs × n_comp_within)
    pub scores_within: Array2<f64>,
    /// Subject-level mean function coefficients
    pub subject_mean_coefficients: Array2<f64>,
    /// Grand mean coefficients
    pub grand_mean_coefficients: Array1<f64>,
}

/// Multilevel Functional PCA (MFPCA)
///
/// Decomposes functional variation into between-subject (level 1) and
/// within-subject (level 2) components.
///
/// For subject i with observations j = 1..n_i:
/// X_{ij}(t) = μ(t) + η_i(t) + ε_{ij}(t)
///
/// where μ is the grand mean, η_i is the subject-specific deviation,
/// and ε_{ij} captures within-subject variation.
#[derive(Debug, Clone)]
pub struct MultilevelFPCA {
    /// Configuration
    pub config: FPCAConfig,
    /// Number of between-subject components
    pub n_comp_between: Option<usize>,
    /// Number of within-subject components
    pub n_comp_within: Option<usize>,
}

impl MultilevelFPCA {
    /// Create MultilevelFPCA with default parameters
    pub fn new() -> Self {
        Self {
            config: FPCAConfig::default(),
            n_comp_between: None,
            n_comp_within: None,
        }
    }

    /// Fit MFPCA to grouped functional data.
    ///
    /// # Arguments
    /// - `subjects`: A slice of subject groups, each containing a slice of (t, y) pairs.
    ///   `subjects[i][j]` = (t_ij, y_ij) for subject i, replicate j.
    pub fn fit(
        &self,
        subjects: &[Vec<(Array1<f64>, Array1<f64>)>],
    ) -> Result<MultilevelFPCAResult<BSplineBasis>> {
        let n_subjects = subjects.len();
        if n_subjects < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "MultilevelFPCA requires at least 2 subjects".to_string(),
                required: 2,
                actual: n_subjects,
            });
        }

        let basis =
            BSplineBasis::uniform(self.config.n_interior_knots, self.config.spline_order)?;
        let k = basis.n_basis();
        let pls = PenalizedLeastSquares {
            lambda: self.config.smooth_lambda,
            penalty_order: 2,
            n_lambda_grid: 20,
            lambda_min: 1e-10,
            lambda_max: 1e4,
        };

        // Step 1: Smooth all curves
        let mut all_coeffs: Vec<Array1<f64>> = Vec::new();
        let mut subject_indices: Vec<usize> = Vec::new();
        let mut n_per_subject: Vec<usize> = Vec::new();

        for (i, subject_data) in subjects.iter().enumerate() {
            if subject_data.is_empty() {
                return Err(TimeSeriesError::InvalidInput(format!(
                    "Subject {} has no observations",
                    i
                )));
            }
            for (t, y) in subject_data.iter() {
                let fd = pls.fit(basis.clone(), t, y)?;
                all_coeffs.push(fd.coefficients);
                subject_indices.push(i);
            }
            n_per_subject.push(subject_data.len());
        }

        let n_total = all_coeffs.len();
        let mut coeff_matrix = Array2::zeros((n_total, k));
        for (i, c) in all_coeffs.iter().enumerate() {
            for j in 0..k {
                coeff_matrix[[i, j]] = c[j];
            }
        }

        // Step 2: Grand mean
        let grand_mean: Array1<f64> = coeff_matrix
            .mean_axis(Axis(0))
            .ok_or_else(|| TimeSeriesError::ComputationError("mean computation failed".to_string()))?;

        // Step 3: Subject-level means (η_i = mean of within-subject curves)
        let mut subject_means = Array2::zeros((n_subjects, k));
        for i in 0..n_subjects {
            let indices: Vec<usize> = subject_indices
                .iter()
                .enumerate()
                .filter(|(_, &s)| s == i)
                .map(|(idx, _)| idx)
                .collect();
            for j in 0..k {
                let sum: f64 = indices.iter().map(|&idx| coeff_matrix[[idx, j]]).sum();
                subject_means[[i, j]] = sum / indices.len() as f64;
            }
        }

        // Step 4: Between-subject deviations (n_subjects × k)
        let mut between_devs = Array2::zeros((n_subjects, k));
        for i in 0..n_subjects {
            for j in 0..k {
                between_devs[[i, j]] = subject_means[[i, j]] - grand_mean[j];
            }
        }

        // Step 5: Within-subject deviations (n_total × k)
        let mut within_devs = Array2::zeros((n_total, k));
        for (obs_idx, &subj_idx) in subject_indices.iter().enumerate() {
            for j in 0..k {
                within_devs[[obs_idx, j]] =
                    coeff_matrix[[obs_idx, j]] - subject_means[[subj_idx, j]];
            }
        }

        // Step 6: FPCA on between-subject covariance
        let gram = basis.gram_matrix()?;
        let ns_f = if n_subjects > 1 {
            (n_subjects - 1) as f64
        } else {
            1.0
        };
        let cov_between = between_devs.t().dot(&between_devs) / ns_f;

        let (ev_between, vecs_between) =
            solve_generalized_eig(&cov_between, &gram, self.config.cov_regularization)?;
        let n_comp_b = match self.n_comp_between {
            Some(nc) => nc.min(ev_between.len()),
            None => ev_between.iter().filter(|&&v| v > 0.0).count().max(1),
        };

        // Step 7: FPCA on within-subject covariance
        let nw_f = if n_total > n_subjects {
            (n_total - n_subjects) as f64
        } else {
            1.0
        };
        let cov_within = within_devs.t().dot(&within_devs) / nw_f;
        let (ev_within, vecs_within) =
            solve_generalized_eig(&cov_within, &gram, self.config.cov_regularization)?;
        let n_comp_w = match self.n_comp_within {
            Some(nc) => nc.min(ev_within.len()),
            None => ev_within.iter().filter(|&&v| v > 0.0).count().max(1),
        };

        // Build eigenfunctions
        let mut ef_between = Vec::with_capacity(n_comp_b);
        for c in 0..n_comp_b {
            let v = vecs_between.column(c).to_owned();
            let norm = v.dot(&gram.dot(&v)).max(0.0).sqrt().max(1e-15);
            let nv = v.mapv(|x| x / norm);
            ef_between.push(FunctionalData::new(basis.clone(), nv)?);
        }

        let mut ef_within = Vec::with_capacity(n_comp_w);
        for c in 0..n_comp_w {
            let v = vecs_within.column(c).to_owned();
            let norm = v.dot(&gram.dot(&v)).max(0.0).sqrt().max(1e-15);
            let nv = v.mapv(|x| x / norm);
            ef_within.push(FunctionalData::new(basis.clone(), nv)?);
        }

        // Compute between-subject scores (n_subjects × n_comp_b)
        let mut scores_between = Array2::zeros((n_subjects, n_comp_b));
        for i in 0..n_subjects {
            let dev: Array1<f64> = between_devs.row(i).to_owned();
            let gdev = gram.dot(&dev);
            for c in 0..n_comp_b {
                scores_between[[i, c]] = gdev.dot(&ef_between[c].coefficients);
            }
        }

        // Compute within-subject scores (n_total × n_comp_w)
        let mut scores_within = Array2::zeros((n_total, n_comp_w));
        for i in 0..n_total {
            let dev: Array1<f64> = within_devs.row(i).to_owned();
            let gdev = gram.dot(&dev);
            for c in 0..n_comp_w {
                scores_within[[i, c]] = gdev.dot(&ef_within[c].coefficients);
            }
        }

        Ok(MultilevelFPCAResult {
            eigenvalues_between: Array1::from_vec(ev_between[..n_comp_b].to_vec()),
            eigenvalues_within: Array1::from_vec(ev_within[..n_comp_w].to_vec()),
            eigenfunctions_between: ef_between,
            eigenfunctions_within: ef_within,
            scores_between,
            scores_within,
            subject_mean_coefficients: subject_means,
            grand_mean_coefficients: grand_mean,
        })
    }
}

impl Default for MultilevelFPCA {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Generalized eigenvalue problem solver
// ============================================================

/// Solve the generalized eigenvalue problem A v = λ B v
/// where B is symmetric positive definite, returning (eigenvalues, eigenvectors)
/// in descending order of eigenvalue.
///
/// Uses Cholesky factorization to convert to standard eigenvalue problem:
/// B = L L^T, then solve L^{-1} A L^{-T} u = λ u, v = L^{-T} u
fn solve_generalized_eig(
    a: &Array2<f64>,
    b: &Array2<f64>,
    reg: f64,
) -> Result<(Vec<f64>, Array2<f64>)> {
    let k = a.nrows();
    assert_eq!(k, b.nrows());

    // Regularize B to ensure positive definiteness
    let mut b_reg = b.clone();
    for i in 0..k {
        b_reg[[i, i]] += reg;
    }

    // Cholesky decomposition of B_reg: B_reg = L L^T
    let l = cholesky_lower(&b_reg)?;
    let l_inv = lower_triangular_inv(&l)?;

    // Transform: M = L^{-1} A L^{-T}
    let a_l_inv_t = a.dot(&l_inv.t());
    let m = l_inv.dot(&a_l_inv_t);

    // Symmetrize M (small numerical asymmetry may appear)
    let m_sym = (&m + &m.t()) / 2.0;

    // Standard symmetric eigenvalue problem
    let (eigenvalues_raw, eigenvecs_raw) = eigh(&m_sym.view(), None).map_err(|e| {
        TimeSeriesError::ComputationError(format!("Eigenvalue decomposition failed: {}", e))
    })?;

    // Back-transform: v = L^{-T} u
    let eigenvecs = l_inv.t().dot(&eigenvecs_raw);

    // Sort by descending eigenvalue
    let mut indexed: Vec<(usize, f64)> = eigenvalues_raw
        .iter()
        .enumerate()
        .map(|(i, &ev)| (i, ev))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let sorted_eigenvalues: Vec<f64> = indexed.iter().map(|(_, ev)| *ev).collect();
    let mut sorted_eigenvecs = Array2::zeros((k, k));
    for (new_col, &(old_col, _)) in indexed.iter().enumerate() {
        let col = eigenvecs.column(old_col).to_owned();
        for i in 0..k {
            sorted_eigenvecs[[i, new_col]] = col[i];
        }
    }

    Ok((sorted_eigenvalues, sorted_eigenvecs))
}

/// Compute lower-triangular Cholesky factor L such that A = L L^T
fn cholesky_lower(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    let mut l = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for p in 0..j {
                sum -= l[[i, p]] * l[[j, p]];
            }
            if i == j {
                if sum <= 0.0 {
                    // Matrix is not positive definite; add small ridge
                    sum = 1e-12;
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    Ok(l)
}

/// Invert a lower triangular matrix by forward substitution
fn lower_triangular_inv(l: &Array2<f64>) -> Result<Array2<f64>> {
    let n = l.nrows();
    let mut inv = Array2::zeros((n, n));
    for j in 0..n {
        inv[[j, j]] = 1.0 / l[[j, j]].max(1e-15).min(-1e-15).abs().max(1e-15);
        // Actually: handle the sign correctly
        inv[[j, j]] = if l[[j, j]].abs() > 1e-15 {
            1.0 / l[[j, j]]
        } else {
            0.0
        };
        for i in (j + 1)..n {
            let mut sum = 0.0;
            for p in j..i {
                sum += l[[i, p]] * inv[[p, j]];
            }
            inv[[i, j]] = if l[[i, i]].abs() > 1e-15 {
                -sum / l[[i, i]]
            } else {
                0.0
            };
        }
    }
    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn make_fpca_data(n_obs: usize) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
        let n_t = 50;
        let t: Array1<f64> = Array1::from_vec(
            (0..n_t).map(|i| i as f64 / (n_t - 1) as f64 * 2.0 * PI).collect(),
        );
        let mut t_list = Vec::new();
        let mut y_list = Vec::new();
        for i in 0..n_obs {
            let phase = i as f64 * 0.3;
            let amplitude = 1.0 + (i as f64 / n_obs as f64) * 0.5;
            let y: Array1<f64> = t.mapv(|ti| amplitude * (ti + phase).sin());
            t_list.push(t.clone());
            y_list.push(y);
        }
        (t_list, y_list)
    }

    #[test]
    fn test_fpca_basic() {
        let (t_list, y_list) = make_fpca_data(15);
        let fpca = FPCA::with_config(FPCAConfig {
            n_components: Some(3),
            smooth_lambda: Some(1e-4),
            n_interior_knots: 8,
            ..Default::default()
        });
        let result = fpca.fit(&t_list, &y_list).expect("FPCA fit failed");
        assert_eq!(result.n_components, 3);
        assert_eq!(result.scores.nrows(), 15);
        assert_eq!(result.scores.ncols(), 3);
    }

    #[test]
    fn test_fpca_variance_explained() {
        let (t_list, y_list) = make_fpca_data(20);
        let fpca = FPCA::with_config(FPCAConfig {
            n_components: Some(5),
            smooth_lambda: Some(1e-4),
            n_interior_knots: 8,
            ..Default::default()
        });
        let result = fpca.fit(&t_list, &y_list).expect("FPCA fit");
        // Cumulative variance at last component should be <= 1
        let last = result.variance_explained.cumulative[result.n_components - 1];
        assert!(last <= 1.0 + 1e-10);
    }

    #[test]
    fn test_fpca_reconstruct() {
        let (t_list, y_list) = make_fpca_data(12);
        let fpca = FPCA::with_config(FPCAConfig {
            n_components: Some(3),
            smooth_lambda: Some(1e-4),
            n_interior_knots: 6,
            ..Default::default()
        });
        let result = fpca.fit(&t_list, &y_list).expect("FPCA fit");
        let basis = BSplineBasis::uniform(6, 4).expect("basis");
        let recon = result.reconstruct(0, 3, &basis).expect("reconstruct");
        let val = recon.eval(1.0).expect("eval");
        assert!(val.is_finite());
    }

    #[test]
    fn test_variance_explained_threshold() {
        let ev = Array1::from_vec(vec![4.0, 2.0, 1.0, 0.5]);
        let ve = VarianceExplained::from_eigenvalues(&ev);
        let n = ve.n_components_for_threshold(0.8);
        assert!(n >= 1 && n <= 4);
    }
}
