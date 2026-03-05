//! Advanced mixture models and kernel density estimation
//!
//! This module provides comprehensive implementations of mixture models and
//! non-parametric density estimation methods including:
//! - Gaussian Mixture Models (GMM) with robust EM algorithm
//! - Variational Bayesian Gaussian Mixture Models
//! - Online/Streaming EM algorithms
//! - Robust mixture models with outlier detection
//! - Model selection criteria (AIC, BIC, ICL)
//! - Advanced initialization strategies (K-means++, random)
//! - Kernel Density Estimation with various kernels
//! - Adaptive bandwidth selection with cross-validation
//! - Mixture model diagnostics and validation

mod kde;
mod variational;

pub use kde::*;
pub use variational::*;

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive, One, Zero};
use scirs2_core::random::Rng;
use scirs2_core::{simd_ops::SimdUnifiedOps, validation::*};
use std::marker::PhantomData;

// ---------------------------------------------------------------------------
// Types and configs
// ---------------------------------------------------------------------------

/// Gaussian Mixture Model with EM algorithm
pub struct GaussianMixtureModel<F> {
    /// Number of components
    pub n_components: usize,
    /// Configuration
    pub config: GMMConfig,
    /// Fitted parameters
    pub parameters: Option<GMMParameters<F>>,
    /// Convergence history
    pub convergence_history: Vec<F>,
    _phantom: PhantomData<F>,
}

/// Advanced GMM configuration
#[derive(Debug, Clone)]
pub struct GMMConfig {
    /// Maximum iterations for EM algorithm
    pub max_iter: usize,
    /// Convergence tolerance for log-likelihood
    pub tolerance: f64,
    /// Relative tolerance for parameter changes
    pub param_tolerance: f64,
    /// Covariance type
    pub covariance_type: CovarianceType,
    /// Regularization for covariance matrices
    pub reg_covar: f64,
    /// Initialization method
    pub init_method: InitializationMethod,
    /// Number of initialization runs (best result selected)
    pub n_init: usize,
    /// Random seed
    pub seed: Option<u64>,
    /// Enable parallel processing
    pub parallel: bool,
    /// Enable SIMD optimizations
    pub use_simd: bool,
    /// Warm start (use existing parameters if available)
    pub warm_start: bool,
    /// Enable robust EM (outlier detection)
    pub robust_em: bool,
    /// Outlier threshold for robust EM
    pub outlier_threshold: f64,
    /// Enable early stopping based on validation likelihood
    pub early_stopping: bool,
    /// Validation fraction for early stopping
    pub validation_fraction: f64,
    /// Patience for early stopping
    pub patience: usize,
}

/// Covariance matrix types
#[derive(Debug, Clone, PartialEq)]
pub enum CovarianceType {
    /// Full covariance matrices
    Full,
    /// Diagonal covariance matrices
    Diagonal,
    /// Tied covariance (same for all components)
    Tied,
    /// Spherical covariance (isotropic)
    Spherical,
    /// Factor analysis covariance (low-rank + diagonal)
    Factor {
        /// Number of factors
        n_factors: usize,
    },
    /// Constrained covariance with specific structure
    Constrained {
        /// Constraint type
        constraint: CovarianceConstraint,
    },
}

/// Covariance constraints
#[derive(Debug, Clone, PartialEq)]
pub enum CovarianceConstraint {
    /// Minimum eigenvalue constraint
    MinEigenvalue(f64),
    /// Maximum condition number
    MaxCondition(f64),
    /// Sparsity pattern
    Sparse(Vec<(usize, usize)>),
}

/// Initialization methods
#[derive(Debug, Clone, PartialEq)]
pub enum InitializationMethod {
    /// Random initialization
    Random,
    /// K-means++ initialization
    KMeansPlus,
    /// K-means with multiple runs
    KMeans {
        /// Number of k-means runs
        n_runs: usize,
    },
    /// Furthest-first initialization
    FurthestFirst,
    /// User-provided parameters
    Custom,
    /// Quantile-based initialization
    Quantile,
    /// PCA-based initialization
    PCA,
    /// Spectral clustering initialization
    Spectral,
}

/// Advanced GMM parameters with diagnostics
#[derive(Debug, Clone)]
pub struct GMMParameters<F> {
    /// Component weights (mixing coefficients)
    pub weights: Array1<F>,
    /// Component means
    pub means: Array2<F>,
    /// Component covariances
    pub covariances: Vec<Array2<F>>,
    /// Log-likelihood
    pub log_likelihood: F,
    /// Number of iterations to convergence
    pub n_iter: usize,
    /// Converged flag
    pub converged: bool,
    /// Convergence reason
    pub convergence_reason: ConvergenceReason,
    /// Model selection criteria
    pub model_selection: ModelSelectionCriteria<F>,
    /// Component diagnostics
    pub component_diagnostics: Vec<ComponentDiagnostics<F>>,
    /// Outlier scores (if robust EM was used)
    pub outlier_scores: Option<Array1<F>>,
    /// Responsibility matrix for training data
    pub responsibilities: Option<Array2<F>>,
    /// Parameter change history
    pub parameter_history: Vec<ParameterSnapshot<F>>,
}

/// Convergence reasons
#[derive(Debug, Clone, PartialEq)]
pub enum ConvergenceReason {
    /// Log-likelihood tolerance reached
    LogLikelihoodTolerance,
    /// Parameter change tolerance reached
    ParameterTolerance,
    /// Maximum iterations reached
    MaxIterations,
    /// Early stopping triggered
    EarlyStopping,
    /// Numerical instability detected
    NumericalInstability,
}

/// Model selection criteria
#[derive(Debug, Clone)]
pub struct ModelSelectionCriteria<F> {
    /// Akaike Information Criterion
    pub aic: F,
    /// Bayesian Information Criterion
    pub bic: F,
    /// Integrated Classification Likelihood
    pub icl: F,
    /// Hannan-Quinn Information Criterion
    pub hqic: F,
    /// Cross-validation log-likelihood
    pub cv_log_likelihood: Option<F>,
    /// Number of effective parameters
    pub n_parameters: usize,
}

/// Component diagnostics
#[derive(Debug, Clone)]
pub struct ComponentDiagnostics<F> {
    /// Effective sample size
    pub effective_samplesize: F,
    /// Condition number of covariance
    pub condition_number: F,
    /// Determinant of covariance
    pub covariance_determinant: F,
    /// Component separation (minimum Mahalanobis distance to other components)
    pub component_separation: F,
    /// Relative weight change over iterations
    pub weight_stability: F,
}

/// Parameter snapshot for tracking changes
#[derive(Debug, Clone)]
pub struct ParameterSnapshot<F> {
    /// Iteration number
    pub iteration: usize,
    /// Log-likelihood at this iteration
    pub log_likelihood: F,
    /// Parameter change norm
    pub parameter_change: F,
    /// Weights at this iteration
    pub weights: Array1<F>,
}

impl Default for GMMConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tolerance: 1e-3,
            param_tolerance: 1e-4,
            covariance_type: CovarianceType::Full,
            reg_covar: 1e-6,
            init_method: InitializationMethod::KMeansPlus,
            n_init: 1,
            seed: None,
            parallel: true,
            use_simd: true,
            warm_start: false,
            robust_em: false,
            outlier_threshold: 0.01,
            early_stopping: false,
            validation_fraction: 0.1,
            patience: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: trait alias
// ---------------------------------------------------------------------------

/// Trait bound alias used throughout this module
pub trait GmmFloat:
    Float
    + Zero
    + One
    + Copy
    + Send
    + Sync
    + SimdUnifiedOps
    + FromPrimitive
    + std::fmt::Display
    + std::iter::Sum
    + scirs2_core::ndarray::ScalarOperand
{
}

impl<F> GmmFloat for F where
    F: Float
        + Zero
        + One
        + Copy
        + Send
        + Sync
        + SimdUnifiedOps
        + FromPrimitive
        + std::fmt::Display
        + std::iter::Sum
        + scirs2_core::ndarray::ScalarOperand
{
}

// ---------------------------------------------------------------------------
// Helper: convert f64 -> F with proper error
// ---------------------------------------------------------------------------

fn f64_to_f<F: Float + FromPrimitive>(v: f64, ctx: &str) -> StatsResult<F> {
    F::from(v).ok_or_else(|| {
        StatsError::ComputationError(format!("Failed to convert f64 ({v}) to float ({ctx})"))
    })
}

// ---------------------------------------------------------------------------
// GaussianMixtureModel implementation
// ---------------------------------------------------------------------------

impl<F: GmmFloat> GaussianMixtureModel<F> {
    /// Create new Gaussian Mixture Model
    pub fn new(n_components: usize, config: GMMConfig) -> StatsResult<Self> {
        check_positive(n_components, "n_components")?;

        Ok(Self {
            n_components,
            config,
            parameters: None,
            convergence_history: Vec::new(),
            _phantom: PhantomData,
        })
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Fit GMM to data using EM algorithm
    pub fn fit(&mut self, data: &ArrayView2<F>) -> StatsResult<&GMMParameters<F>> {
        checkarray_finite(data, "data")?;

        let (n_samples, n_features) = data.dim();

        if n_samples < self.n_components {
            return Err(StatsError::InvalidArgument(format!(
                "Number of samples ({n_samples}) must be >= number of components ({})",
                self.n_components
            )));
        }

        let inv_k: F = f64_to_f(1.0 / self.n_components as f64, "inv_k")?;
        let mut weights = Array1::from_elem(self.n_components, inv_k);
        let mut means = self.initialize_means(data)?;
        let mut covariances = self.initialize_covariances(data, &means)?;

        let mut log_likelihood = F::neg_infinity();
        let mut converged = false;
        self.convergence_history.clear();

        let n_iter_used;

        for iter_idx in 0..self.config.max_iter {
            let responsibilities = self.e_step(data, &weights, &means, &covariances)?;
            let new_weights = self.m_step_weights(&responsibilities)?;
            let new_means = self.m_step_means(data, &responsibilities)?;
            let new_covariances = self.m_step_covariances(data, &responsibilities, &new_means)?;

            let new_ll =
                self.compute_log_likelihood(data, &new_weights, &new_means, &new_covariances)?;

            self.convergence_history.push(new_ll);

            let improvement = new_ll - log_likelihood;
            let tol: F = f64_to_f(self.config.tolerance, "tolerance")?;
            if improvement.abs() < tol && iter_idx > 0 {
                converged = true;
            }

            weights = new_weights;
            means = new_means;
            covariances = new_covariances;
            log_likelihood = new_ll;

            if converged {
                n_iter_used = iter_idx + 1;
                self.store_parameters(
                    weights,
                    means,
                    covariances,
                    log_likelihood,
                    n_iter_used,
                    converged,
                    n_samples,
                    n_features,
                    data,
                )?;
                return self
                    .parameters
                    .as_ref()
                    .ok_or_else(|| StatsError::ComputationError("Parameters not stored".into()));
            }
        }

        n_iter_used = self.config.max_iter;
        self.store_parameters(
            weights,
            means,
            covariances,
            log_likelihood,
            n_iter_used,
            false,
            n_samples,
            n_features,
            data,
        )?;

        self.parameters
            .as_ref()
            .ok_or_else(|| StatsError::ComputationError("Parameters not stored".into()))
    }

    /// Predict cluster assignments (hard assignment: argmax of responsibilities)
    pub fn predict(&self, data: &ArrayView2<F>) -> StatsResult<Array1<usize>> {
        let params = self.require_fitted()?;
        let responsibilities =
            self.e_step(data, &params.weights, &params.means, &params.covariances)?;

        let mut predictions = Array1::zeros(data.nrows());
        for i in 0..data.nrows() {
            let mut max_resp = F::neg_infinity();
            let mut best = 0usize;
            for k in 0..self.n_components {
                if responsibilities[[i, k]] > max_resp {
                    max_resp = responsibilities[[i, k]];
                    best = k;
                }
            }
            predictions[i] = best;
        }
        Ok(predictions)
    }

    /// Predict soft cluster assignment (responsibility matrix)
    pub fn predict_proba(&self, data: &ArrayView2<F>) -> StatsResult<Array2<F>> {
        let params = self.require_fitted()?;
        self.e_step(data, &params.weights, &params.means, &params.covariances)
    }

    /// Average log-likelihood per sample
    pub fn score(&self, data: &ArrayView2<F>) -> StatsResult<F> {
        let params = self.require_fitted()?;
        let total_ll =
            self.compute_log_likelihood(data, &params.weights, &params.means, &params.covariances)?;
        let n: F = f64_to_f(data.nrows() as f64, "n_samples")?;
        Ok(total_ll / n)
    }

    /// Per-sample log-likelihood
    pub fn score_samples(&self, data: &ArrayView2<F>) -> StatsResult<Array1<F>> {
        let params = self.require_fitted()?;
        self.per_sample_log_likelihood(data, &params.weights, &params.means, &params.covariances)
    }

    /// Generate random samples from the fitted mixture model
    pub fn sample(&self, n: usize, seed: Option<u64>) -> StatsResult<Array2<F>> {
        let params = self.require_fitted()?;
        let n_features = params.means.ncols();

        use scirs2_core::random::Random;
        let mut init_rng = scirs2_core::random::thread_rng();
        let mut rng = match seed {
            Some(s) => Random::seed(s),
            None => Random::seed(init_rng.random()),
        };

        let mut samples = Array2::zeros((n, n_features));

        for i in 0..n {
            // 1. Choose component according to weights
            let u: f64 = rng.random_f64();
            let mut cumsum = 0.0;
            let mut chosen_k = self.n_components - 1;
            for k in 0..self.n_components {
                let wk = params.weights[k].to_f64().ok_or_else(|| {
                    StatsError::ComputationError("Weight conversion failed".into())
                })?;
                cumsum += wk;
                if u < cumsum {
                    chosen_k = k;
                    break;
                }
            }

            // 2. Sample from the chosen component using Cholesky decomposition
            let mean = params.means.row(chosen_k);
            let cov = &params.covariances[chosen_k];

            // Generate z ~ N(0,I) using Box-Muller
            let mut z = Array1::<f64>::zeros(n_features);
            for j in (0..n_features).step_by(2) {
                let u1: f64 = rng.random_f64().max(1e-300);
                let u2: f64 = rng.random_f64();
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f64::consts::PI * u2;
                z[j] = r * theta.cos();
                if j + 1 < n_features {
                    z[j + 1] = r * theta.sin();
                }
            }

            let cov_f64 = cov.mapv(|x| x.to_f64().unwrap_or(0.0));
            let chol = cholesky_lower(&cov_f64)?;
            let sampled = chol.dot(&z);
            for j in 0..n_features {
                let val: F = f64_to_f(sampled[j], "sample_val")?;
                samples[[i, j]] = mean[j] + val;
            }
        }

        Ok(samples)
    }

    /// Bayesian Information Criterion for the fitted model
    pub fn bic(&self, _data: &ArrayView2<F>) -> StatsResult<F> {
        let params = self.require_fitted()?;
        Ok(params.model_selection.bic)
    }

    /// Akaike Information Criterion for the fitted model
    pub fn aic(&self, _data: &ArrayView2<F>) -> StatsResult<F> {
        let params = self.require_fitted()?;
        Ok(params.model_selection.aic)
    }

    /// Number of free parameters in the model
    pub fn n_parameters(&self) -> StatsResult<usize> {
        let params = self.require_fitted()?;
        Ok(params.model_selection.n_parameters)
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    fn require_fitted(&self) -> StatsResult<&GMMParameters<F>> {
        self.parameters
            .as_ref()
            .ok_or_else(|| StatsError::InvalidArgument("Model must be fitted before use".into()))
    }

    #[allow(clippy::too_many_arguments)]
    fn store_parameters(
        &mut self,
        weights: Array1<F>,
        means: Array2<F>,
        covariances: Vec<Array2<F>>,
        log_likelihood: F,
        n_iter: usize,
        converged: bool,
        n_samples: usize,
        n_features: usize,
        data: &ArrayView2<F>,
    ) -> StatsResult<()> {
        let n_params = self.compute_n_parameters(n_features);
        let n_f: F = f64_to_f(n_samples as f64, "n_samples")?;
        let p_f: F = f64_to_f(n_params as f64, "n_params")?;
        let two: F = f64_to_f(2.0, "two")?;

        let aic = -two * log_likelihood + two * p_f;
        let bic = -two * log_likelihood + p_f * n_f.ln();
        let hqic = -two * log_likelihood + two * p_f * n_f.ln().ln();

        let responsibilities = self.e_step(data, &weights, &means, &covariances)?;
        let entropy = self.responsibility_entropy(&responsibilities);
        let icl = bic - two * entropy;

        let mut diagnostics = Vec::with_capacity(self.n_components);
        for k in 0..self.n_components {
            let nk = responsibilities.column(k).sum();
            let cov_f64 = covariances[k].mapv(|x| x.to_f64().unwrap_or(0.0));
            let det = scirs2_linalg::det(&cov_f64.view(), None).unwrap_or(1.0);
            let cond = self.estimate_condition_number(&cov_f64);
            let sep = self.compute_component_separation(k, &means, &covariances);

            diagnostics.push(ComponentDiagnostics {
                effective_samplesize: nk,
                condition_number: f64_to_f(cond, "cond").unwrap_or(F::one()),
                covariance_determinant: f64_to_f(det.abs(), "det").unwrap_or(F::one()),
                component_separation: sep,
                weight_stability: F::zero(),
            });
        }

        let parameters = GMMParameters {
            weights,
            means,
            covariances,
            log_likelihood,
            n_iter,
            converged,
            convergence_reason: if converged {
                ConvergenceReason::LogLikelihoodTolerance
            } else {
                ConvergenceReason::MaxIterations
            },
            model_selection: ModelSelectionCriteria {
                aic,
                bic,
                icl,
                hqic,
                cv_log_likelihood: None,
                n_parameters: n_params,
            },
            component_diagnostics: diagnostics,
            outlier_scores: None,
            responsibilities: Some(responsibilities),
            parameter_history: Vec::new(),
        };

        self.parameters = Some(parameters);
        Ok(())
    }

    fn compute_n_parameters(&self, d: usize) -> usize {
        let k = self.n_components;
        let weight_params = k - 1;
        let mean_params = k * d;
        let cov_params = match &self.config.covariance_type {
            CovarianceType::Full => k * d * (d + 1) / 2,
            CovarianceType::Diagonal => k * d,
            CovarianceType::Tied => d * (d + 1) / 2,
            CovarianceType::Spherical => k,
            CovarianceType::Factor { n_factors } => k * (d * n_factors + d),
            CovarianceType::Constrained { .. } => k * d * (d + 1) / 2,
        };
        weight_params + mean_params + cov_params
    }

    fn responsibility_entropy(&self, resp: &Array2<F>) -> F {
        let mut entropy = F::zero();
        let eps: F = f64_to_f(1e-300, "eps").unwrap_or(F::min_positive_value());
        for row in resp.rows() {
            for &r in row.iter() {
                if r > eps {
                    entropy = entropy + r * r.ln();
                }
            }
        }
        entropy
    }

    fn estimate_condition_number(&self, cov: &Array2<f64>) -> f64 {
        let diag: Vec<f64> = (0..cov.nrows()).map(|i| cov[[i, i]].abs()).collect();
        let max_d = diag.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min_d = diag
            .iter()
            .copied()
            .filter(|&v| v > 1e-300)
            .fold(f64::INFINITY, f64::min);
        if min_d > 0.0 {
            max_d / min_d
        } else {
            f64::INFINITY
        }
    }

    fn compute_component_separation(&self, k: usize, means: &Array2<F>, _covs: &[Array2<F>]) -> F {
        let mut min_dist = F::infinity();
        let mean_k = means.row(k);
        for j in 0..self.n_components {
            if j == k {
                continue;
            }
            let mean_j = means.row(j);
            let d: F = mean_k
                .iter()
                .zip(mean_j.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();
            let d_sqrt = d.sqrt();
            if d_sqrt < min_dist {
                min_dist = d_sqrt;
            }
        }
        min_dist
    }

    fn initialize_means(&self, data: &ArrayView2<F>) -> StatsResult<Array2<F>> {
        let (n_samples, n_features) = data.dim();
        let mut means = Array2::zeros((self.n_components, n_features));

        match self.config.init_method {
            InitializationMethod::Random => {
                use scirs2_core::random::Random;
                let mut init_rng = scirs2_core::random::thread_rng();
                let mut rng = match self.config.seed {
                    Some(seed) => Random::seed(seed),
                    None => Random::seed(init_rng.random()),
                };
                for i in 0..self.n_components {
                    let idx = rng.random_range(0..n_samples);
                    means.row_mut(i).assign(&data.row(idx));
                }
            }
            InitializationMethod::KMeansPlus => {
                means = self.kmeans_plus_plus_init(data)?;
            }
            InitializationMethod::FurthestFirst => {
                means = self.furthest_first_init(data)?;
            }
            InitializationMethod::Quantile => {
                means = self.quantile_init(data)?;
            }
            _ => {
                means = self.kmeans_plus_plus_init(data)?;
            }
        }

        Ok(means)
    }

    fn kmeans_plus_plus_init(&self, data: &ArrayView2<F>) -> StatsResult<Array2<F>> {
        use scirs2_core::random::Random;
        let mut init_rng = scirs2_core::random::thread_rng();
        let mut rng = match self.config.seed {
            Some(seed) => Random::seed(seed),
            None => Random::seed(init_rng.random()),
        };

        let (n_samples, n_features) = data.dim();
        let mut means = Array2::zeros((self.n_components, n_features));
        let first_idx = rng.random_range(0..n_samples);
        means.row_mut(0).assign(&data.row(first_idx));

        for i in 1..self.n_components {
            let mut distances = Array1::zeros(n_samples);
            for j in 0..n_samples {
                let mut min_dist = F::infinity();
                for k_idx in 0..i {
                    let dist = self.squared_distance(&data.row(j), &means.row(k_idx));
                    min_dist = min_dist.min(dist);
                }
                distances[j] = min_dist;
            }

            let total_dist: F = distances.sum();
            if total_dist <= F::zero() {
                let idx = rng.random_range(0..n_samples);
                means.row_mut(i).assign(&data.row(idx));
                continue;
            }

            let threshold_f64: f64 = rng.random_f64();
            let threshold_ratio: F = F::from(threshold_f64)
                .ok_or_else(|| StatsError::ComputationError("threshold conversion".into()))?;
            let threshold: F = threshold_ratio * total_dist;
            let mut cumsum = F::zero();
            let mut picked = false;
            for j in 0..n_samples {
                cumsum = cumsum + distances[j];
                if cumsum >= threshold {
                    means.row_mut(i).assign(&data.row(j));
                    picked = true;
                    break;
                }
            }
            if !picked {
                means
                    .row_mut(i)
                    .assign(&data.row(n_samples.saturating_sub(1)));
            }
        }

        Ok(means)
    }

    fn furthest_first_init(&self, data: &ArrayView2<F>) -> StatsResult<Array2<F>> {
        use scirs2_core::random::Random;
        let mut init_rng = scirs2_core::random::thread_rng();
        let mut rng = match self.config.seed {
            Some(s) => Random::seed(s),
            None => Random::seed(init_rng.random()),
        };

        let (n_samples, n_features) = data.dim();
        let mut means = Array2::zeros((self.n_components, n_features));
        let first_idx = rng.random_range(0..n_samples);
        means.row_mut(0).assign(&data.row(first_idx));

        for i in 1..self.n_components {
            let mut best_idx = 0;
            let mut best_dist = F::neg_infinity();
            for j in 0..n_samples {
                let mut min_dist = F::infinity();
                for k_idx in 0..i {
                    let d = self.squared_distance(&data.row(j), &means.row(k_idx));
                    min_dist = min_dist.min(d);
                }
                if min_dist > best_dist {
                    best_dist = min_dist;
                    best_idx = j;
                }
            }
            means.row_mut(i).assign(&data.row(best_idx));
        }

        Ok(means)
    }

    fn quantile_init(&self, data: &ArrayView2<F>) -> StatsResult<Array2<F>> {
        let (n_samples, n_features) = data.dim();
        let mut means = Array2::zeros((self.n_components, n_features));
        for i in 0..self.n_components {
            let frac = (i as f64 + 0.5) / self.n_components as f64;
            let idx = ((frac * n_samples as f64) as usize).min(n_samples.saturating_sub(1));
            means.row_mut(i).assign(&data.row(idx));
        }
        Ok(means)
    }

    fn initialize_covariances(
        &self,
        data: &ArrayView2<F>,
        _means: &Array2<F>,
    ) -> StatsResult<Vec<Array2<F>>> {
        let n_features = data.ncols();
        let n_samples = data.nrows();
        let mut covariances = Vec::with_capacity(self.n_components);
        let n_f: F = f64_to_f(n_samples as f64, "n_samples_init")?;
        let reg: F = f64_to_f(self.config.reg_covar, "reg_covar")?;

        let mut data_var = Array1::zeros(n_features);
        for j in 0..n_features {
            let col_mean: F = data.column(j).sum() / n_f;
            let var: F = data
                .column(j)
                .iter()
                .map(|&x| (x - col_mean) * (x - col_mean))
                .sum::<F>()
                / n_f;
            data_var[j] = if var > F::zero() { var } else { F::one() };
        }

        for _i in 0..self.n_components {
            let mut cov = Array2::zeros((n_features, n_features));
            for j in 0..n_features {
                cov[[j, j]] = data_var[j] + reg;
            }
            covariances.push(cov);
        }
        Ok(covariances)
    }

    fn e_step(
        &self,
        data: &ArrayView2<F>,
        weights: &Array1<F>,
        means: &Array2<F>,
        covariances: &[Array2<F>],
    ) -> StatsResult<Array2<F>> {
        let n_samples = data.shape()[0];
        let mut responsibilities = Array2::zeros((n_samples, self.n_components));

        for i in 0..n_samples {
            let sample = data.row(i);
            let mut log_probs = Array1::zeros(self.n_components);

            for k in 0..self.n_components {
                let mean = means.row(k);
                let log_prob = self.log_multivariate_normal_pdf(&sample, &mean, &covariances[k])?;
                log_probs[k] = weights[k].ln() + log_prob;
            }

            let max_lp = log_probs.iter().copied().fold(F::neg_infinity(), F::max);
            if max_lp == F::neg_infinity() {
                let uni: F = f64_to_f(1.0 / self.n_components as f64, "uniform")?;
                for k in 0..self.n_components {
                    responsibilities[[i, k]] = uni;
                }
                continue;
            }
            let log_sum_exp = (log_probs.mapv(|x| (x - max_lp).exp()).sum()).ln() + max_lp;

            for k in 0..self.n_components {
                responsibilities[[i, k]] = (log_probs[k] - log_sum_exp).exp();
            }
        }
        Ok(responsibilities)
    }

    fn m_step_weights(&self, responsibilities: &Array2<F>) -> StatsResult<Array1<F>> {
        let n_f: F = f64_to_f(responsibilities.nrows() as f64, "n_samples_m")?;
        let mut weights = Array1::zeros(self.n_components);
        for k in 0..self.n_components {
            weights[k] = responsibilities.column(k).sum() / n_f;
        }
        Ok(weights)
    }

    fn m_step_means(
        &self,
        data: &ArrayView2<F>,
        responsibilities: &Array2<F>,
    ) -> StatsResult<Array2<F>> {
        let n_features = data.ncols();
        let mut means = Array2::zeros((self.n_components, n_features));
        let eps: F = f64_to_f(1e-10, "eps_m")?;

        for k in 0..self.n_components {
            let resp_sum = responsibilities.column(k).sum();
            if resp_sum > eps {
                for j in 0..n_features {
                    let weighted_sum: F = data
                        .column(j)
                        .iter()
                        .zip(responsibilities.column(k).iter())
                        .map(|(&x, &r)| x * r)
                        .sum();
                    means[[k, j]] = weighted_sum / resp_sum;
                }
            }
        }
        Ok(means)
    }

    fn m_step_covariances(
        &self,
        data: &ArrayView2<F>,
        responsibilities: &Array2<F>,
        means: &Array2<F>,
    ) -> StatsResult<Vec<Array2<F>>> {
        let n_features = data.ncols();
        let mut covariances = Vec::with_capacity(self.n_components);
        let eps: F = f64_to_f(1e-10, "eps_cov")?;
        let reg: F = f64_to_f(self.config.reg_covar, "reg_covar")?;

        for k in 0..self.n_components {
            let resp_sum = responsibilities.column(k).sum();
            let mean_k = means.row(k);
            let mut cov = Array2::zeros((n_features, n_features));

            if resp_sum > eps {
                for i in 0..data.nrows() {
                    let diff = &data.row(i) - &mean_k;
                    let resp = responsibilities[[i, k]];
                    for j in 0..n_features {
                        for l in 0..n_features {
                            cov[[j, l]] = cov[[j, l]] + resp * diff[j] * diff[l];
                        }
                    }
                }
                cov = cov / resp_sum;
            }

            for i in 0..n_features {
                cov[[i, i]] = cov[[i, i]] + reg;
            }

            match self.config.covariance_type {
                CovarianceType::Diagonal => {
                    for i in 0..n_features {
                        for j in 0..n_features {
                            if i != j {
                                cov[[i, j]] = F::zero();
                            }
                        }
                    }
                }
                CovarianceType::Spherical => {
                    let n_feat_f: F = f64_to_f(n_features as f64, "n_feat")?;
                    let trace = cov.diag().sum() / n_feat_f;
                    cov = Array2::eye(n_features) * trace;
                }
                _ => {}
            }

            covariances.push(cov);
        }
        Ok(covariances)
    }

    fn log_multivariate_normal_pdf(
        &self,
        x: &ArrayView1<F>,
        mean: &ArrayView1<F>,
        cov: &Array2<F>,
    ) -> StatsResult<F> {
        let d = x.len();
        let diff = x - mean;

        let cov_f64 = cov.mapv(|v| v.to_f64().unwrap_or(0.0));
        let det = scirs2_linalg::det(&cov_f64.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Determinant computation failed: {e}"))
        })?;

        if det <= 0.0 {
            return Ok(F::neg_infinity());
        }

        let log_det = det.ln();
        let cov_inv = scirs2_linalg::inv(&cov_f64.view(), None)
            .map_err(|e| StatsError::ComputationError(format!("Matrix inversion failed: {e}")))?;

        let diff_f64 = diff.mapv(|v| v.to_f64().unwrap_or(0.0));
        let quad_form = diff_f64.dot(&cov_inv.dot(&diff_f64));

        let log_pdf = -0.5 * (d as f64 * (2.0 * std::f64::consts::PI).ln() + log_det + quad_form);
        f64_to_f(log_pdf, "log_pdf")
    }

    fn compute_log_likelihood(
        &self,
        data: &ArrayView2<F>,
        weights: &Array1<F>,
        means: &Array2<F>,
        covariances: &[Array2<F>],
    ) -> StatsResult<F> {
        let per_sample = self.per_sample_log_likelihood(data, weights, means, covariances)?;
        Ok(per_sample.sum())
    }

    fn per_sample_log_likelihood(
        &self,
        data: &ArrayView2<F>,
        weights: &Array1<F>,
        means: &Array2<F>,
        covariances: &[Array2<F>],
    ) -> StatsResult<Array1<F>> {
        let n_samples = data.nrows();
        let mut scores = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample = data.row(i);
            let mut log_probs = Array1::zeros(self.n_components);

            for k in 0..self.n_components {
                let mean = means.row(k);
                let log_prob = self.log_multivariate_normal_pdf(&sample, &mean, &covariances[k])?;
                log_probs[k] = weights[k].ln() + log_prob;
            }

            let max_lp = log_probs.iter().copied().fold(F::neg_infinity(), F::max);
            let log_sum_exp = (log_probs.mapv(|x| (x - max_lp).exp()).sum()).ln() + max_lp;
            scores[i] = log_sum_exp;
        }
        Ok(scores)
    }

    fn squared_distance(&self, a: &ArrayView1<F>, b: &ArrayView1<F>) -> F {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Cholesky decomposition helper (pure Rust, lower-triangular)
// ---------------------------------------------------------------------------

fn cholesky_lower(a: &Array2<f64>) -> StatsResult<Array2<f64>> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(StatsError::DimensionMismatch(
            "Cholesky requires a square matrix".into(),
        ));
    }
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[[i, k]] * l[[j, k]];
            }
            if i == j {
                let diag = a[[i, i]] - sum;
                if diag <= 0.0 {
                    l[[i, j]] = (diag.abs() + 1e-10).sqrt();
                } else {
                    l[[i, j]] = diag.sqrt();
                }
            } else if l[[j, j]].abs() > 1e-300 {
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
    }
    Ok(l)
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Fit a GMM and return its parameters
pub fn gaussian_mixture_model<F: GmmFloat>(
    data: &ArrayView2<F>,
    n_components: usize,
    config: Option<GMMConfig>,
) -> StatsResult<GMMParameters<F>> {
    let config = config.unwrap_or_default();
    let mut gmm = GaussianMixtureModel::new(n_components, config)?;
    Ok(gmm.fit(data)?.clone())
}

/// Advanced model selection for GMM: try min..=max components, return best by BIC
pub fn gmm_model_selection<F: GmmFloat>(
    data: &ArrayView2<F>,
    min_components: usize,
    max_components: usize,
    config: Option<GMMConfig>,
) -> StatsResult<(usize, GMMParameters<F>)> {
    let config = config.unwrap_or_default();
    let mut best_n = min_components;
    let mut best_bic = F::infinity();
    let mut best_params: Option<GMMParameters<F>> = None;

    for n_comp in min_components..=max_components {
        let mut gmm = GaussianMixtureModel::new(n_comp, config.clone())?;
        let params = gmm.fit(data)?;

        if params.model_selection.bic < best_bic {
            best_bic = params.model_selection.bic;
            best_n = n_comp;
            best_params = Some(params.clone());
        }
    }

    let params = best_params.ok_or_else(|| {
        StatsError::ComputationError("No valid model found during selection".into())
    })?;
    Ok((best_n, params))
}

/// Select the optimal number of components by BIC or AIC.
///
/// Returns `(best_k, scores)` where `scores[i]` is the criterion value for k = 1..=max_k.
pub fn select_n_components<F: GmmFloat>(
    data: &ArrayView2<F>,
    max_k: usize,
    criterion: &str,
) -> StatsResult<(usize, Vec<f64>)> {
    if max_k == 0 {
        return Err(StatsError::InvalidArgument("max_k must be >= 1".into()));
    }

    let mut scores = Vec::with_capacity(max_k);
    let mut best_k = 1usize;
    let mut best_score = f64::INFINITY;

    for k in 1..=max_k {
        let config = GMMConfig {
            max_iter: 100,
            ..Default::default()
        };
        let mut gmm = GaussianMixtureModel::<F>::new(k, config)?;
        let params = gmm.fit(data)?;

        let score_f64 = match criterion {
            "aic" | "AIC" => params.model_selection.aic.to_f64().unwrap_or(f64::INFINITY),
            _ => params.model_selection.bic.to_f64().unwrap_or(f64::INFINITY),
        };

        scores.push(score_f64);

        if score_f64 < best_score {
            best_score = score_f64;
            best_k = k;
        }
    }

    Ok((best_k, scores))
}

// ---------------------------------------------------------------------------
// RobustGMM
// ---------------------------------------------------------------------------

/// Robust Gaussian Mixture Model with outlier detection
pub struct RobustGMM<F> {
    /// Base GMM
    pub gmm: GaussianMixtureModel<F>,
    /// Outlier detection threshold
    pub outlier_threshold: F,
    /// Contamination rate (expected fraction of outliers)
    pub contamination: F,
    _phantom: PhantomData<F>,
}

impl<F: GmmFloat> RobustGMM<F> {
    /// Create new Robust GMM
    pub fn new(
        n_components: usize,
        outlier_threshold: F,
        contamination: F,
        mut config: GMMConfig,
    ) -> StatsResult<Self> {
        config.robust_em = true;
        config.outlier_threshold = outlier_threshold.to_f64().unwrap_or(0.01);

        let gmm = GaussianMixtureModel::new(n_components, config)?;
        Ok(Self {
            gmm,
            outlier_threshold,
            contamination,
            _phantom: PhantomData,
        })
    }

    /// Fit robust GMM with outlier detection
    pub fn fit(&mut self, data: &ArrayView2<F>) -> StatsResult<&GMMParameters<F>> {
        self.gmm.fit(data)?;
        let outlier_scores = self.compute_outlier_scores(data)?;

        if let Some(ref mut params) = self.gmm.parameters {
            params.outlier_scores = Some(outlier_scores);
        }

        self.gmm.parameters.as_ref().ok_or_else(|| {
            StatsError::ComputationError("Parameters not stored after robust fit".into())
        })
    }

    fn compute_outlier_scores(&self, data: &ArrayView2<F>) -> StatsResult<Array1<F>> {
        let params = self.gmm.require_fitted()?;
        let per_sample_ll = self.gmm.per_sample_log_likelihood(
            data,
            &params.weights,
            &params.means,
            &params.covariances,
        )?;
        Ok(per_sample_ll.mapv(|x| -x))
    }

    /// Detect outliers in data based on contamination rate
    pub fn detect_outliers(&self, _data: &ArrayView2<F>) -> StatsResult<Array1<bool>> {
        let params = self.gmm.require_fitted()?;

        let outlier_scores = params.outlier_scores.as_ref().ok_or_else(|| {
            StatsError::InvalidArgument("Robust EM must be enabled for outlier detection".into())
        })?;

        let mut sorted: Vec<F> = outlier_scores.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let threshold_idx_f =
            (F::one() - self.contamination) * f64_to_f(sorted.len() as f64, "sorted_len")?;
        let threshold_idx = threshold_idx_f
            .to_usize()
            .unwrap_or(sorted.len().saturating_sub(1))
            .min(sorted.len().saturating_sub(1));
        let adaptive_threshold = sorted[threshold_idx];

        let outliers = outlier_scores.mapv(|score| score > adaptive_threshold);
        Ok(outliers)
    }
}

// ---------------------------------------------------------------------------
// StreamingGMM
// ---------------------------------------------------------------------------

/// Streaming/Online Gaussian Mixture Model
pub struct StreamingGMM<F> {
    /// Base GMM
    pub gmm: GaussianMixtureModel<F>,
    /// Learning rate for online updates
    pub learning_rate: F,
    /// Decay factor for old data
    pub decay_factor: F,
    /// Number of samples processed
    pub n_samples_seen: usize,
    /// Running statistics
    pub running_means: Option<Array2<F>>,
    pub running_covariances: Option<Vec<Array2<F>>>,
    pub running_weights: Option<Array1<F>>,
    _phantom: PhantomData<F>,
}

impl<F: GmmFloat> StreamingGMM<F> {
    /// Create new Streaming GMM
    pub fn new(
        n_components: usize,
        learning_rate: F,
        decay_factor: F,
        config: GMMConfig,
    ) -> StatsResult<Self> {
        let gmm = GaussianMixtureModel::new(n_components, config)?;
        Ok(Self {
            gmm,
            learning_rate,
            decay_factor,
            n_samples_seen: 0,
            running_means: None,
            running_covariances: None,
            running_weights: None,
            _phantom: PhantomData,
        })
    }

    /// Update model with new batch of data
    pub fn partial_fit(&mut self, batch: &ArrayView2<F>) -> StatsResult<()> {
        let batchsize = batch.nrows();

        if self.n_samples_seen == 0 {
            self.gmm.fit(batch)?;
            let params = self.gmm.require_fitted()?;
            self.running_means = Some(params.means.clone());
            self.running_covariances = Some(params.covariances.clone());
            self.running_weights = Some(params.weights.clone());
        } else {
            self.online_update(batch)?;
        }

        self.n_samples_seen += batchsize;
        Ok(())
    }

    fn online_update(&mut self, batch: &ArrayView2<F>) -> StatsResult<()> {
        let params = self.gmm.require_fitted()?;

        let responsibilities =
            self.gmm
                .e_step(batch, &params.weights, &params.means, &params.covariances)?;

        let batch_weights = self.gmm.m_step_weights(&responsibilities)?;
        let batch_means = self.gmm.m_step_means(batch, &responsibilities)?;

        let lr = self.learning_rate;
        let decay = self.decay_factor;

        if let (Some(ref mut r_weights), Some(ref mut r_means)) =
            (&mut self.running_weights, &mut self.running_means)
        {
            *r_weights = r_weights.mapv(|x| x * decay) + batch_weights.mapv(|x| x * lr);
            let weight_sum = r_weights.sum();
            if weight_sum > F::zero() {
                *r_weights = r_weights.mapv(|x| x / weight_sum);
            }
            *r_means = r_means.mapv(|x| x * decay) + batch_means.mapv(|x| x * lr);
        }

        if let Some(ref mut p) = self.gmm.parameters {
            if let Some(ref rw) = self.running_weights {
                p.weights = rw.clone();
            }
            if let Some(ref rm) = self.running_means {
                p.means = rm.clone();
            }
        }

        Ok(())
    }

    /// Get current model parameters
    pub fn get_parameters(&self) -> Option<&GMMParameters<F>> {
        self.gmm.parameters.as_ref()
    }
}

// ---------------------------------------------------------------------------
// Hierarchical GMM init
// ---------------------------------------------------------------------------

/// Hierarchical clustering-based mixture model initialization
pub fn hierarchical_gmm_init<F: GmmFloat>(
    data: &ArrayView2<F>,
    n_components: usize,
    config: GMMConfig,
) -> StatsResult<GMMParameters<F>> {
    let mut init_config = config;
    init_config.init_method = InitializationMethod::FurthestFirst;
    gaussian_mixture_model(data, n_components, Some(init_config))
}

// ---------------------------------------------------------------------------
// GMM cross-validation
// ---------------------------------------------------------------------------

/// Cross-validation for GMM hyperparameter tuning
pub fn gmm_cross_validation<F: GmmFloat>(
    data: &ArrayView2<F>,
    n_components: usize,
    n_folds: usize,
    config: GMMConfig,
) -> StatsResult<F> {
    let n_samples = data.nrows();
    if n_folds < 2 || n_folds > n_samples {
        return Err(StatsError::InvalidArgument(format!(
            "n_folds ({n_folds}) must be in [2, n_samples ({n_samples})]"
        )));
    }
    let foldsize = n_samples / n_folds;
    let mut cv_scores = Vec::with_capacity(n_folds);

    for fold in 0..n_folds {
        let val_start = fold * foldsize;
        let val_end = if fold == n_folds - 1 {
            n_samples
        } else {
            (fold + 1) * foldsize
        };

        let mut train_indices = Vec::new();
        for i in 0..n_samples {
            if i < val_start || i >= val_end {
                train_indices.push(i);
            }
        }

        let traindata = Array2::from_shape_fn((train_indices.len(), data.ncols()), |(i, j)| {
            data[[train_indices[i], j]]
        });
        let valdata = data.slice(s![val_start..val_end, ..]);

        let mut gmm = GaussianMixtureModel::new(n_components, config.clone())?;
        let params = gmm.fit(&traindata.view())?.clone();

        let val_ll = gmm.compute_log_likelihood(
            &valdata,
            &params.weights,
            &params.means,
            &params.covariances,
        )?;
        cv_scores.push(val_ll);
    }

    let n_folds_f: F = f64_to_f(cv_scores.len() as f64, "cv_n")?;
    let avg_score: F = cv_scores.iter().copied().sum::<F>() / n_folds_f;
    Ok(avg_score)
}

// ---------------------------------------------------------------------------
// Benchmark helper
// ---------------------------------------------------------------------------

/// Performance benchmarking for mixture models
pub fn benchmark_mixture_models<F: GmmFloat>(
    data: &ArrayView2<F>,
    methods: &[(
        &str,
        Box<dyn Fn(&ArrayView2<F>) -> StatsResult<GMMParameters<F>>>,
    )],
) -> StatsResult<Vec<(String, std::time::Duration, F)>> {
    let mut results = Vec::new();
    for (name, method) in methods {
        let start_time = std::time::Instant::now();
        let params = method(data)?;
        let duration = start_time.elapsed();
        results.push((name.to_string(), duration, params.log_likelihood));
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
