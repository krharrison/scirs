//! Variational Bayesian Gaussian Mixture Model

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive, One, Zero};
use scirs2_core::random::Rng;
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::marker::PhantomData;

/// Variational Bayesian Gaussian Mixture Model
pub struct VariationalGMM<F> {
    /// Maximum number of components
    pub max_components: usize,
    /// Configuration
    pub config: VariationalGMMConfig,
    /// Fitted parameters
    pub parameters: Option<VariationalGMMParameters<F>>,
    /// Lower bound history
    pub lower_bound_history: Vec<F>,
    _phantom: PhantomData<F>,
}

/// Configuration for Variational GMM
#[derive(Debug, Clone)]
pub struct VariationalGMMConfig {
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Concentration parameter for Dirichlet prior
    pub alpha: f64,
    /// Degrees of freedom for Wishart prior
    pub nu: f64,
    /// Prior mean
    pub mean_prior: Option<Vec<f64>>,
    /// Prior precision matrix
    pub precision_prior: Option<Vec<Vec<f64>>>,
    /// Enable automatic relevance determination
    pub ard: bool,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for VariationalGMMConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tolerance: 1e-6,
            alpha: 1.0,
            nu: 1.0,
            mean_prior: None,
            precision_prior: None,
            ard: true,
            seed: None,
        }
    }
}

/// Variational GMM parameters
#[derive(Debug, Clone)]
pub struct VariationalGMMParameters<F> {
    /// Component weights (posterior Dirichlet parameters)
    pub weight_concentration: Array1<F>,
    /// Component means (posterior normal parameters)
    pub mean_precision: Array1<F>,
    /// Means
    pub means: Array2<F>,
    /// Component precisions (posterior Wishart parameters)
    pub degrees_of_freedom: Array1<F>,
    /// Scale matrices
    pub scale_matrices: Array3<F>,
    /// Lower bound
    pub lower_bound: F,
    /// Effective number of components
    pub effective_components: usize,
    /// Number of iterations
    pub n_iter: usize,
    /// Converged flag
    pub converged: bool,
}

/// Variational GMM result
#[derive(Debug, Clone)]
pub struct VariationalGMMResult<F> {
    /// Lower bound value
    pub lower_bound: F,
    /// Effective number of components
    pub effective_components: usize,
    /// Predictive probabilities
    pub responsibilities: Array2<F>,
    /// Component weights
    pub weights: Array1<F>,
}

impl<F> VariationalGMM<F>
where
    F: Float
        + FromPrimitive
        + SimdUnifiedOps
        + Send
        + Sync
        + std::fmt::Debug
        + std::fmt::Display
        + std::iter::Sum<F>,
{
    /// Create new Variational GMM
    pub fn new(max_components: usize, config: VariationalGMMConfig) -> Self {
        Self {
            max_components,
            config,
            parameters: None,
            lower_bound_history: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Fit Variational GMM to data
    pub fn fit(&mut self, data: &ArrayView2<F>) -> StatsResult<VariationalGMMResult<F>> {
        let (_n_samples, n_features) = data.dim();

        let alpha_f: F = F::from(self.config.alpha)
            .ok_or_else(|| StatsError::ComputationError("alpha conversion failed".into()))?;
        let nu_f: F = F::from(self.config.nu)
            .ok_or_else(|| StatsError::ComputationError("nu conversion failed".into()))?;
        let n_feat_f: F = F::from(n_features)
            .ok_or_else(|| StatsError::ComputationError("n_features conversion failed".into()))?;

        let mut weight_concentration = Array1::from_elem(self.max_components, alpha_f);
        let mean_precision_val = F::one();
        let mut mean_precision = Array1::from_elem(self.max_components, mean_precision_val);
        let mut means = self.initialize_means(data)?;
        let mut degrees_of_freedom = Array1::from_elem(self.max_components, nu_f + n_feat_f);
        let mut scale_matrices = Array3::zeros((self.max_components, n_features, n_features));
        for k in 0..self.max_components {
            for i in 0..n_features {
                scale_matrices[[k, i, i]] = F::one();
            }
        }

        let mut lower_bound = F::neg_infinity();
        let mut converged = false;
        let tol: F = F::from(self.config.tolerance)
            .ok_or_else(|| StatsError::ComputationError("tolerance conversion failed".into()))?;

        for iteration in 0..self.config.max_iter {
            let responsibilities = self.compute_responsibilities(
                data,
                &means,
                &scale_matrices,
                &degrees_of_freedom,
                &weight_concentration,
            )?;

            let (new_wc, new_mp, new_means, new_dof, new_sm) =
                self.update_parameters(data, &responsibilities)?;

            let new_lb =
                self.compute_lower_bound(data, &responsibilities, &new_wc, &new_means, &new_sm)?;

            if iteration > 0 && (new_lb - lower_bound).abs() < tol {
                converged = true;
            }

            weight_concentration = new_wc;
            mean_precision = new_mp;
            means = new_means;
            degrees_of_freedom = new_dof;
            scale_matrices = new_sm;
            lower_bound = new_lb;
            self.lower_bound_history.push(lower_bound);

            if converged {
                break;
            }
        }

        let effective_components = self.compute_effective_components(&weight_concentration);
        let responsibilities = self.compute_responsibilities(
            data,
            &means,
            &scale_matrices,
            &degrees_of_freedom,
            &weight_concentration,
        )?;
        let weights = self.compute_weights(&weight_concentration);

        let parameters = VariationalGMMParameters {
            weight_concentration,
            mean_precision,
            means,
            degrees_of_freedom,
            scale_matrices,
            lower_bound,
            effective_components,
            n_iter: self.lower_bound_history.len(),
            converged,
        };
        self.parameters = Some(parameters);

        Ok(VariationalGMMResult {
            lower_bound,
            effective_components,
            responsibilities,
            weights,
        })
    }

    fn initialize_means(&self, data: &ArrayView2<F>) -> StatsResult<Array2<F>> {
        let (n_samples, n_features) = data.dim();
        let mut means = Array2::zeros((self.max_components, n_features));
        use scirs2_core::random::Random;
        let mut init_rng = scirs2_core::random::thread_rng();
        let mut rng = match self.config.seed {
            Some(seed) => Random::seed(seed),
            None => Random::seed(init_rng.random()),
        };
        for i in 0..self.max_components {
            let idx = rng.random_range(0..n_samples);
            means.row_mut(i).assign(&data.row(idx));
        }
        Ok(means)
    }

    fn compute_responsibilities(
        &self,
        data: &ArrayView2<F>,
        means: &Array2<F>,
        scale_matrices: &Array3<F>,
        degrees_of_freedom: &Array1<F>,
        weight_concentration: &Array1<F>,
    ) -> StatsResult<Array2<F>> {
        let n_samples = data.shape()[0];
        let mut responsibilities = Array2::zeros((n_samples, self.max_components));

        for i in 0..n_samples {
            let mut log_probs = Array1::zeros(self.max_components);
            for k in 0..self.max_components {
                let log_weight = weight_concentration[k].ln();
                let log_ll = self.compute_log_likelihood_component(
                    &data.row(i),
                    &means.row(k),
                    &scale_matrices.slice(s![k, .., ..]),
                    degrees_of_freedom[k],
                )?;
                log_probs[k] = log_weight + log_ll;
            }
            let log_sum = self.log_sum_exp(&log_probs);
            for k in 0..self.max_components {
                responsibilities[[i, k]] = (log_probs[k] - log_sum).exp();
            }
        }
        Ok(responsibilities)
    }

    fn update_parameters(
        &self,
        data: &ArrayView2<F>,
        responsibilities: &Array2<F>,
    ) -> StatsResult<(Array1<F>, Array1<F>, Array2<F>, Array1<F>, Array3<F>)> {
        let (n_samples, n_features) = data.dim();

        let alpha_f: F = F::from(self.config.alpha)
            .ok_or_else(|| StatsError::ComputationError("alpha conversion".into()))?;
        let nu_f: F = F::from(self.config.nu)
            .ok_or_else(|| StatsError::ComputationError("nu conversion".into()))?;
        let n_feat_f: F = F::from(n_features)
            .ok_or_else(|| StatsError::ComputationError("n_features conversion".into()))?;
        let small: F = F::from(0.1)
            .ok_or_else(|| StatsError::ComputationError("constant conversion".into()))?;

        let mut weight_concentration = Array1::from_elem(self.max_components, alpha_f);
        let mean_precision = Array1::ones(self.max_components);
        let mut means = Array2::zeros((self.max_components, n_features));
        let mut degrees_of_freedom = Array1::from_elem(self.max_components, nu_f + n_feat_f);
        let mut scale_matrices = Array3::zeros((self.max_components, n_features, n_features));

        for k in 0..self.max_components {
            let nk: F = responsibilities.column(k).sum();
            weight_concentration[k] = weight_concentration[k] + nk;

            if nk > F::zero() {
                for j in 0..n_features {
                    let mut weighted_sum = F::zero();
                    for i in 0..n_samples {
                        weighted_sum = weighted_sum + responsibilities[[i, k]] * data[[i, j]];
                    }
                    means[[k, j]] = weighted_sum / nk;
                }
                degrees_of_freedom[k] = nu_f + nk;
                for i in 0..n_features {
                    scale_matrices[[k, i, i]] = F::one() + small * nk;
                }
            }
        }

        Ok((
            weight_concentration,
            mean_precision,
            means,
            degrees_of_freedom,
            scale_matrices,
        ))
    }

    fn compute_lower_bound(
        &self,
        data: &ArrayView2<F>,
        responsibilities: &Array2<F>,
        weight_concentration: &Array1<F>,
        means: &Array2<F>,
        scale_matrices: &Array3<F>,
    ) -> StatsResult<F> {
        let n_samples = data.shape()[0];
        let mut lower_bound = F::zero();
        let ten: F = F::from(10.0)
            .ok_or_else(|| StatsError::ComputationError("constant conversion".into()))?;
        let small_kl: F = F::from(0.01)
            .ok_or_else(|| StatsError::ComputationError("constant conversion".into()))?;

        for i in 0..n_samples {
            for k in 0..self.max_components {
                if responsibilities[[i, k]] > F::zero() {
                    let log_ll = self.compute_log_likelihood_component(
                        &data.row(i),
                        &means.row(k),
                        &scale_matrices.slice(s![k, .., ..]),
                        ten,
                    )?;
                    lower_bound = lower_bound + responsibilities[[i, k]] * log_ll;
                }
            }
        }

        for k in 0..self.max_components {
            let w = weight_concentration[k];
            if w > F::zero() {
                lower_bound = lower_bound - w * w.ln() * small_kl;
            }
        }

        Ok(lower_bound)
    }

    fn compute_effective_components(&self, wc: &Array1<F>) -> usize {
        let total: F = wc.sum();
        let threshold = F::from(0.01).unwrap_or(F::zero());
        wc.iter().filter(|&&w| w / total > threshold).count()
    }

    fn compute_weights(&self, wc: &Array1<F>) -> Array1<F> {
        let total: F = wc.sum();
        wc.mapv(|w| w / total)
    }

    fn compute_log_likelihood_component(
        &self,
        point: &ArrayView1<F>,
        mean: &ArrayView1<F>,
        _scale_matrix: &scirs2_core::ndarray::ArrayBase<
            scirs2_core::ndarray::ViewRepr<&F>,
            scirs2_core::ndarray::Dim<[usize; 2]>,
        >,
        _degrees_of_freedom: F,
    ) -> StatsResult<F> {
        let half: F = F::from(0.5)
            .ok_or_else(|| StatsError::ComputationError("constant conversion".into()))?;
        let mut sum_sq = F::zero();
        for (x, m) in point.iter().zip(mean.iter()) {
            let diff = *x - *m;
            sum_sq = sum_sq + diff * diff;
        }
        Ok(-half * sum_sq)
    }

    fn log_sum_exp(&self, logvalues: &Array1<F>) -> F {
        let max_val = logvalues.iter().fold(F::neg_infinity(), |a, &b| a.max(b));
        if max_val == F::neg_infinity() {
            return F::neg_infinity();
        }
        let sum: F = logvalues.iter().map(|&x| (x - max_val).exp()).sum();
        max_val + sum.ln()
    }
}
