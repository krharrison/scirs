//! Python bindings for MCMC and Gaussian Process modules

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use scirs2_numpy::{IntoPyArray, PyArray1, PyArray2};
use scirs2_core::ndarray::{Array1, Array2};

// =============================================================================
// MCMC Diagnostics
// =============================================================================

/// Compute R-hat (Gelman-Rubin) convergence diagnostic.
///
/// R-hat measures convergence of multiple MCMC chains to a common distribution.
/// Values close to 1.0 indicate convergence. Values > 1.05 are typically
/// considered problematic.
///
/// Parameters:
///     chains: List of chains, each chain is a list of floats
///
/// Returns:
///     R-hat statistic (float)
#[pyfunction]
pub fn r_hat_py(chains: Vec<Vec<f64>>) -> PyResult<f64> {
    if chains.is_empty() {
        return Err(PyValueError::new_err("chains must not be empty"));
    }
    scirs2_stats::mcmc::diagnostics::r_hat(&chains)
        .map_err(|e| PyRuntimeError::new_err(format!("r_hat failed: {}", e)))
}

/// Compute split R-hat for a single chain.
///
/// Splits the chain in half and computes R-hat between the two halves,
/// which can detect non-stationarity within a single chain.
///
/// Parameters:
///     chain: List of floats representing a single MCMC chain
///
/// Returns:
///     Split R-hat statistic (float)
#[pyfunction]
pub fn split_r_hat_py(chain: Vec<f64>) -> PyResult<f64> {
    if chain.is_empty() {
        return Err(PyValueError::new_err("chain must not be empty"));
    }
    scirs2_stats::mcmc::diagnostics::split_r_hat(&chain)
        .map_err(|e| PyRuntimeError::new_err(format!("split_r_hat failed: {}", e)))
}

/// Compute effective sample size (ESS).
///
/// ESS estimates the number of independent samples equivalent to the
/// correlated MCMC samples.
///
/// Parameters:
///     samples: List of floats representing MCMC samples
///
/// Returns:
///     Effective sample size (float)
#[pyfunction]
pub fn effective_sample_size_py(samples: Vec<f64>) -> PyResult<f64> {
    if samples.is_empty() {
        return Err(PyValueError::new_err("samples must not be empty"));
    }
    scirs2_stats::mcmc::diagnostics::effective_sample_size(&samples)
        .map_err(|e| PyRuntimeError::new_err(format!("effective_sample_size failed: {}", e)))
}

/// Compute bulk ESS for a single chain.
///
/// Parameters:
///     samples: List of floats representing MCMC samples
///
/// Returns:
///     Bulk ESS (float)
#[pyfunction]
pub fn bulk_ess_py(samples: Vec<f64>) -> PyResult<f64> {
    if samples.is_empty() {
        return Err(PyValueError::new_err("samples must not be empty"));
    }
    scirs2_stats::mcmc::diagnostics::bulk_ess_single(&samples)
        .map_err(|e| PyRuntimeError::new_err(format!("bulk_ess failed: {}", e)))
}

/// Compute tail ESS for a single chain.
///
/// Parameters:
///     samples: List of floats representing MCMC samples
///
/// Returns:
///     Tail ESS (float)
#[pyfunction]
pub fn tail_ess_py(samples: Vec<f64>) -> PyResult<f64> {
    if samples.is_empty() {
        return Err(PyValueError::new_err("samples must not be empty"));
    }
    scirs2_stats::mcmc::diagnostics::tail_ess_single(&samples)
        .map_err(|e| PyRuntimeError::new_err(format!("tail_ess failed: {}", e)))
}

/// Compute Monte Carlo Standard Error (MCSE) of the mean.
///
/// Parameters:
///     samples: List of floats representing MCMC samples
///
/// Returns:
///     MCSE of the mean (float)
#[pyfunction]
pub fn mcse_py(samples: Vec<f64>) -> PyResult<f64> {
    if samples.is_empty() {
        return Err(PyValueError::new_err("samples must not be empty"));
    }
    scirs2_stats::mcmc::diagnostics::mcse(&samples)
        .map_err(|e| PyRuntimeError::new_err(format!("mcse failed: {}", e)))
}

/// Compute autocorrelation of MCMC samples up to a given lag.
///
/// Parameters:
///     samples: List of floats representing MCMC samples
///     max_lag: Maximum lag to compute (default: len(samples) // 4)
///
/// Returns:
///     List of autocorrelation values for lags 0..max_lag
#[pyfunction]
#[pyo3(signature = (samples, max_lag = None))]
pub fn autocorrelation_py(samples: Vec<f64>, max_lag: Option<usize>) -> PyResult<Vec<f64>> {
    if samples.is_empty() {
        return Err(PyValueError::new_err("samples must not be empty"));
    }
    let lag = max_lag.unwrap_or(samples.len() / 4);
    scirs2_stats::mcmc::diagnostics::autocorrelation(&samples, lag)
        .map_err(|e| PyRuntimeError::new_err(format!("autocorrelation failed: {}", e)))
}

/// Compute energy-BFMI diagnostic for HMC/NUTS.
///
/// BFMI (Bayesian Fraction of Missing Information) measures mixing efficiency.
/// Values < 0.3 suggest problems.
///
/// Parameters:
///     energies: List of Hamiltonian energy values from HMC/NUTS
///
/// Returns:
///     BFMI value (float)
#[pyfunction]
pub fn energy_bfmi_py(energies: Vec<f64>) -> PyResult<f64> {
    if energies.is_empty() {
        return Err(PyValueError::new_err("energies must not be empty"));
    }
    scirs2_stats::mcmc::diagnostics::energy_bfmi(&energies)
        .map_err(|e| PyRuntimeError::new_err(format!("energy_bfmi failed: {}", e)))
}

// =============================================================================
// NUTS Sampler
// =============================================================================

/// No-U-Turn Sampler (NUTS) for Bayesian inference.
///
/// NUTS is a self-tuning variant of Hamiltonian Monte Carlo that
/// automatically adapts the step size and number of leapfrog steps.
///
/// Parameters:
///     log_prob_grad_fn: Python callable(theta: list[float]) -> (log_prob: float, grad: list[float])
///     initial: Initial parameter vector (list of floats)
///     n_samples: Number of posterior samples to draw
///     warmup_steps: Number of warmup (adaptation) steps (default: 500)
///     step_size: Initial leapfrog step size (default: 0.1)
///     max_tree_depth: Maximum binary tree depth (default: 10)
///     target_accept: Target acceptance probability for adaptation (default: 0.8)
///
/// Returns:
///     Dict with:
///     - 'samples': 2D array of shape (n_samples, dim)
///     - 'log_probs': 1D array of log probabilities
///     - 'divergences': list of bool flags
///     - 'accept_probs': list of acceptance probabilities
#[pyfunction]
#[pyo3(signature = (log_prob_grad_fn, initial, n_samples, warmup_steps=500, step_size=0.1, max_tree_depth=10, target_accept=0.8))]
pub fn nuts_sample_py(
    py: Python,
    log_prob_grad_fn: &Bound<'_, PyAny>,
    initial: Vec<f64>,
    n_samples: usize,
    warmup_steps: usize,
    step_size: f64,
    max_tree_depth: usize,
    target_accept: f64,
) -> PyResult<Py<PyAny>> {
    use scirs2_stats::{NutsConfig, NutsSampler};

    if initial.is_empty() {
        return Err(PyValueError::new_err("initial must not be empty"));
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let config = NutsConfig {
        step_size,
        max_tree_depth,
        target_accept,
        adapt_step_size: true,
        warmup_steps,
        max_delta_h: 1000.0,
    };

    let fn_obj = log_prob_grad_fn.clone().unbind();

    let log_prob_grad = move |theta: &[f64]| -> (f64, Vec<f64>) {
        Python::attach(|py| {
            let theta_list: Vec<f64> = theta.to_vec();
            let result = fn_obj
                .bind(py)
                .call1((theta_list,))
                .expect("Failed to call log_prob_grad_fn");
            let tup: (f64, Vec<f64>) = result
                .extract()
                .expect("log_prob_grad_fn must return (float, list[float])");
            tup
        })
    };

    let mut sampler = NutsSampler::new(config);
    let nuts_samples = sampler
        .sample(log_prob_grad, &initial, n_samples)
        .map_err(|e| PyRuntimeError::new_err(format!("NUTS sampling failed: {}", e)))?;

    let dim = initial.len();
    let mut samples_flat: Vec<f64> = Vec::with_capacity(n_samples * dim);
    let mut log_probs: Vec<f64> = Vec::with_capacity(n_samples);
    let mut divergences: Vec<bool> = Vec::with_capacity(n_samples);
    let mut accept_probs: Vec<f64> = Vec::with_capacity(n_samples);

    for s in &nuts_samples {
        samples_flat.extend_from_slice(&s.position);
        log_probs.push(s.log_prob);
        divergences.push(s.divergent);
        accept_probs.push(s.acceptance_prob);
    }

    let samples_arr = Array2::from_shape_vec((n_samples, dim), samples_flat)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to reshape samples: {}", e)))?;

    let log_probs_arr = Array1::from_vec(log_probs);
    let accept_arr = Array1::from_vec(accept_probs);

    let dict = PyDict::new(py);
    dict.set_item("samples", samples_arr.into_pyarray(py))?;
    dict.set_item("log_probs", log_probs_arr.into_pyarray(py))?;
    dict.set_item("divergences", divergences)?;
    dict.set_item("accept_probs", accept_arr.into_pyarray(py))?;

    Ok(dict.into())
}

/// Metropolis-Hastings sampler with random walk proposal.
///
/// Parameters:
///     log_density_fn: Python callable(theta: list[float]) -> float
///     initial: Initial parameter vector
///     n_samples: Number of samples to draw
///     step_size: Random walk proposal std deviation (default: 0.5)
///     thin: Keep every `thin`-th sample (default: 1)
///
/// Returns:
///     Dict with:
///     - 'samples': 2D array of shape (n_samples, dim)
///     - 'acceptance_rate': fraction of proposals accepted
#[pyfunction]
#[pyo3(signature = (log_density_fn, initial, n_samples, step_size=0.5, thin=1))]
pub fn metropolis_sample_py(
    py: Python,
    log_density_fn: &Bound<'_, PyAny>,
    initial: Vec<f64>,
    n_samples: usize,
    step_size: f64,
    thin: usize,
) -> PyResult<Py<PyAny>> {
    use scirs2_stats::mcmc::metropolis::{
        CustomTarget, MetropolisHastings, RandomWalkProposal,
    };

    if initial.is_empty() {
        return Err(PyValueError::new_err("initial must not be empty"));
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }
    let thin = thin.max(1);

    let fn_obj = log_density_fn.clone().unbind();
    let dim = initial.len();

    let target = CustomTarget::new(dim, move |x: &Array1<f64>| -> f64 {
        let x_vec: Vec<f64> = x.to_vec();
        Python::attach(|py| {
            fn_obj
                .bind(py)
                .call1((x_vec,))
                .and_then(|r| r.extract::<f64>())
                .unwrap_or(f64::NEG_INFINITY)
        })
    })
    .map_err(|e| PyRuntimeError::new_err(format!("Failed to create target: {}", e)))?;

    let proposal = RandomWalkProposal::new(step_size)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create proposal: {}", e)))?;

    let initial_arr = Array1::from_vec(initial);

    let mut sampler = MetropolisHastings::new(target, proposal, initial_arr)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create sampler: {}", e)))?;

    let mut rng = scirs2_core::random::thread_rng();
    let raw_samples = sampler
        .sample_thinned(n_samples, thin, &mut rng)
        .map_err(|e| PyRuntimeError::new_err(format!("Metropolis sampling failed: {}", e)))?;

    let acceptance_rate = sampler.acceptance_rate();
    let total = raw_samples.nrows();

    let dict = PyDict::new(py);
    dict.set_item("samples", raw_samples.into_pyarray(py))?;
    dict.set_item("acceptance_rate", acceptance_rate)?;
    dict.set_item("n_samples", total)?;

    Ok(dict.into())
}

// =============================================================================
// Gaussian Process Regression
// =============================================================================

/// Gaussian Process Regressor with RBF (Squared Exponential) kernel.
///
/// A non-parametric, probabilistic model suitable for regression problems
/// that also provides uncertainty estimates.
///
/// Attributes:
///     length_scale: Length scale of RBF kernel
///     alpha: Noise level added to diagonal of covariance matrix
///     normalize_y: Whether to normalize target values
#[pyclass(name = "GaussianProcessRegressor")]
pub struct PyGaussianProcessRegressor {
    x_train: Option<Vec<Vec<f64>>>,
    y_train: Option<Vec<f64>>,
    alpha: f64,
    normalize_y: bool,
    length_scale: f64,
}

#[pymethods]
impl PyGaussianProcessRegressor {
    #[new]
    #[pyo3(signature = (length_scale=1.0, alpha=1e-10, normalize_y=true))]
    fn new(length_scale: f64, alpha: f64, normalize_y: bool) -> PyResult<Self> {
        if length_scale <= 0.0 {
            return Err(PyValueError::new_err("length_scale must be positive"));
        }
        if alpha < 0.0 {
            return Err(PyValueError::new_err("alpha must be non-negative"));
        }
        Ok(PyGaussianProcessRegressor {
            x_train: None,
            y_train: None,
            alpha,
            normalize_y,
            length_scale,
        })
    }

    /// Fit the Gaussian Process to training data.
    ///
    /// Parameters:
    ///     x: Training inputs, shape (n_samples, n_features) as list of lists
    ///     y: Training targets, shape (n_samples,) as list of floats
    fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<()> {
        use scirs2_stats::gaussian_process::{GaussianProcessRegressor, SquaredExponential};

        let n = x.len();
        if n == 0 {
            return Err(PyValueError::new_err("x must not be empty"));
        }
        if y.len() != n {
            return Err(PyValueError::new_err("x and y must have the same length"));
        }
        let n_features = x[0].len();
        if n_features == 0 {
            return Err(PyValueError::new_err("x must have at least one feature"));
        }

        // Validate and convert to Array2
        let mut x_flat: Vec<f64> = Vec::with_capacity(n * n_features);
        for row in &x {
            if row.len() != n_features {
                return Err(PyValueError::new_err(
                    "All rows of x must have the same number of features",
                ));
            }
            x_flat.extend_from_slice(row);
        }
        let x_arr = Array2::from_shape_vec((n, n_features), x_flat)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create x array: {}", e)))?;
        let y_arr = Array1::from_vec(y.clone());

        // Validate the model fits without error
        let kernel = SquaredExponential::new(self.length_scale, 1.0);
        let mut gpr = GaussianProcessRegressor::with_options(kernel, self.alpha, self.normalize_y);
        gpr.fit(&x_arr, &y_arr)
            .map_err(|e| PyRuntimeError::new_err(format!("GP fit failed: {}", e)))?;

        self.x_train = Some(x);
        self.y_train = Some(y);
        Ok(())
    }

    /// Predict using the fitted Gaussian Process.
    ///
    /// Parameters:
    ///     x: Test inputs, shape (n_samples, n_features) as list of lists
    ///     return_std: If True, also return standard deviations (default: False)
    ///
    /// Returns:
    ///     If return_std=False: 1D numpy array of mean predictions
    ///     If return_std=True: tuple (mean, std) of 1D numpy arrays
    #[pyo3(signature = (x, return_std=false))]
    fn predict(
        &self,
        py: Python,
        x: Vec<Vec<f64>>,
        return_std: bool,
    ) -> PyResult<Py<PyAny>> {
        let (gpr, n_features) = self.rebuild_gpr()?;
        let x_test_arr = self.build_test_array(&x, n_features)?;

        if return_std {
            let (mean, std) = gpr
                .predict_with_std(&x_test_arr)
                .map_err(|e| PyRuntimeError::new_err(format!("GP predict_with_std failed: {}", e)))?;
            let mean_py = mean.into_pyarray(py);
            let std_py = std.into_pyarray(py);
            let tup = pyo3::types::PyTuple::new(py, [mean_py, std_py])
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create tuple: {}", e)))?;
            Ok(tup.into())
        } else {
            let mean = gpr
                .predict(&x_test_arr)
                .map_err(|e| PyRuntimeError::new_err(format!("GP predict failed: {}", e)))?;
            Ok(mean.into_pyarray(py).into())
        }
    }

    /// Score the model using R^2 coefficient of determination.
    ///
    /// Parameters:
    ///     x: Test inputs
    ///     y: True target values
    ///
    /// Returns:
    ///     R^2 score (float)
    fn score(&self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<f64> {
        let (gpr, n_features) = self.rebuild_gpr()?;
        let x_test_arr = self.build_test_array(&x, n_features)?;
        let y_test_arr = Array1::from_vec(y);

        gpr.score(&x_test_arr, &y_test_arr)
            .map_err(|e| PyRuntimeError::new_err(format!("GP score failed: {}", e)))
    }

    /// Get the log marginal likelihood of the training data.
    ///
    /// Returns:
    ///     Log marginal likelihood (float)
    fn log_marginal_likelihood(&self) -> PyResult<f64> {
        let (gpr, _) = self.rebuild_gpr()?;
        gpr.log_marginal_likelihood()
            .map_err(|e| PyRuntimeError::new_err(format!("log_marginal_likelihood failed: {}", e)))
    }

    /// Get the length scale parameter.
    #[getter]
    fn get_length_scale(&self) -> f64 {
        self.length_scale
    }

    /// Get the alpha (noise) parameter.
    #[getter]
    fn get_alpha(&self) -> f64 {
        self.alpha
    }

    /// Get whether y normalization is enabled.
    #[getter]
    fn get_normalize_y(&self) -> bool {
        self.normalize_y
    }

    /// Check if the model has been fitted.
    fn is_fitted(&self) -> bool {
        self.x_train.is_some()
    }
}

impl PyGaussianProcessRegressor {
    /// Helper: rebuild the underlying GaussianProcessRegressor and return it along with n_features.
    fn rebuild_gpr(
        &self,
    ) -> PyResult<(
        scirs2_stats::gaussian_process::GaussianProcessRegressor<
            scirs2_stats::gaussian_process::SquaredExponential,
        >,
        usize,
    )> {
        use scirs2_stats::gaussian_process::{GaussianProcessRegressor, SquaredExponential};

        let x_train = self
            .x_train
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Model not fitted yet. Call fit() first."))?;
        let y_train = self
            .y_train
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Model not fitted yet. Call fit() first."))?;

        let n_train = x_train.len();
        let n_features = x_train[0].len();

        let mut x_flat: Vec<f64> = Vec::with_capacity(n_train * n_features);
        for row in x_train {
            x_flat.extend_from_slice(row);
        }
        let x_train_arr = Array2::from_shape_vec((n_train, n_features), x_flat)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create train array: {}", e)))?;
        let y_train_arr = Array1::from_vec(y_train.clone());

        let kernel = SquaredExponential::new(self.length_scale, 1.0);
        let mut gpr = GaussianProcessRegressor::with_options(kernel, self.alpha, self.normalize_y);
        gpr.fit(&x_train_arr, &y_train_arr)
            .map_err(|e| PyRuntimeError::new_err(format!("GP fit failed: {}", e)))?;

        Ok((gpr, n_features))
    }

    /// Helper: convert list-of-lists to Array2 and validate shape.
    fn build_test_array(&self, x: &[Vec<f64>], n_features: usize) -> PyResult<Array2<f64>> {
        let n_test = x.len();
        if n_test == 0 {
            return Err(PyValueError::new_err("x must not be empty"));
        }
        let n_feat_test = x[0].len();
        if n_feat_test != n_features {
            return Err(PyValueError::new_err(format!(
                "Test x has {} features but training x has {}",
                n_feat_test, n_features
            )));
        }
        let mut flat: Vec<f64> = Vec::with_capacity(n_test * n_features);
        for row in x {
            if row.len() != n_features {
                return Err(PyValueError::new_err(
                    "All rows of test x must have the same number of features",
                ));
            }
            flat.extend_from_slice(row);
        }
        Array2::from_shape_vec((n_test, n_features), flat)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create test array: {}", e)))
    }
}

/// Python module registration function for MCMC and GP bindings
pub fn register_mcmc_gp_module(m: &Bound<'_, pyo3::PyModule>) -> pyo3::PyResult<()> {
    // MCMC diagnostics
    m.add_function(wrap_pyfunction!(r_hat_py, m)?)?;
    m.add_function(wrap_pyfunction!(split_r_hat_py, m)?)?;
    m.add_function(wrap_pyfunction!(effective_sample_size_py, m)?)?;
    m.add_function(wrap_pyfunction!(bulk_ess_py, m)?)?;
    m.add_function(wrap_pyfunction!(tail_ess_py, m)?)?;
    m.add_function(wrap_pyfunction!(mcse_py, m)?)?;
    m.add_function(wrap_pyfunction!(autocorrelation_py, m)?)?;
    m.add_function(wrap_pyfunction!(energy_bfmi_py, m)?)?;

    // MCMC samplers
    m.add_function(wrap_pyfunction!(nuts_sample_py, m)?)?;
    m.add_function(wrap_pyfunction!(metropolis_sample_py, m)?)?;

    // Gaussian Process
    m.add_class::<PyGaussianProcessRegressor>()?;

    Ok(())
}
