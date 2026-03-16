//! Gaussian Process surrogate model for Bayesian optimization.
//!
//! Provides a self-contained GP regression implementation with:
//! - Multiple kernel functions: RBF, Matern (1/2, 3/2, 5/2), Rational Quadratic
//! - Composite kernels: Sum and Product
//! - Efficient Cholesky-based prediction for mean and variance
//! - Hyperparameter optimization via type-II maximum likelihood (marginal likelihood)
//!
//! The GP is specifically designed as a surrogate model for Bayesian optimization,
//! prioritising numerically robust prediction of both mean and uncertainty.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::{Rng, RngExt};

use crate::error::{OptimizeError, OptimizeResult};

// ---------------------------------------------------------------------------
// Kernel trait & implementations
// ---------------------------------------------------------------------------

/// Trait for covariance (kernel) functions used by the GP surrogate.
pub trait SurrogateKernel: Send + Sync {
    /// Evaluate the kernel between two input vectors.
    fn eval(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64;

    /// Compute the full covariance matrix for a set of inputs.
    fn covariance_matrix(&self, x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let mut k = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..=i {
                let kij = self.eval(&x.row(i), &x.row(j));
                k[[i, j]] = kij;
                if i != j {
                    k[[j, i]] = kij;
                }
            }
        }
        k
    }

    /// Compute the cross-covariance matrix between two sets of inputs.
    fn cross_covariance(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64> {
        let n1 = x1.nrows();
        let n2 = x2.nrows();
        let mut k = Array2::zeros((n1, n2));
        for i in 0..n1 {
            for j in 0..n2 {
                k[[i, j]] = self.eval(&x1.row(i), &x2.row(j));
            }
        }
        k
    }

    /// Return current hyperparameters as a flat vector (log-scale).
    fn get_log_params(&self) -> Vec<f64>;

    /// Set hyperparameters from a flat vector (log-scale).
    fn set_log_params(&mut self, params: &[f64]);

    /// Number of hyperparameters.
    fn n_params(&self) -> usize {
        self.get_log_params().len()
    }

    /// Clone the kernel into a boxed trait object.
    fn clone_box(&self) -> Box<dyn SurrogateKernel>;

    /// Name of the kernel (for debug display).
    fn name(&self) -> &str;
}

impl Clone for Box<dyn SurrogateKernel> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// ---------------------------------------------------------------------------
// Squared Exponential (RBF) kernel
// ---------------------------------------------------------------------------

/// Squared Exponential / RBF kernel.
///
/// k(x, x') = sigma^2 * exp(-0.5 * ||x - x'||^2 / length_scale^2)
#[derive(Debug, Clone)]
pub struct RbfKernel {
    /// Length scale
    pub length_scale: f64,
    /// Signal variance
    pub signal_variance: f64,
}

impl RbfKernel {
    /// Create a new RBF kernel.
    pub fn new(length_scale: f64, signal_variance: f64) -> Self {
        Self {
            length_scale: length_scale.max(1e-10),
            signal_variance: signal_variance.max(1e-10),
        }
    }
}

impl Default for RbfKernel {
    fn default() -> Self {
        Self::new(1.0, 1.0)
    }
}

impl SurrogateKernel for RbfKernel {
    fn eval(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        let sq_dist = squared_distance(x1, x2);
        self.signal_variance * (-0.5 * sq_dist / (self.length_scale * self.length_scale)).exp()
    }

    fn get_log_params(&self) -> Vec<f64> {
        vec![self.length_scale.ln(), self.signal_variance.ln()]
    }

    fn set_log_params(&mut self, params: &[f64]) {
        if params.len() >= 2 {
            self.length_scale = params[0].exp().max(1e-10);
            self.signal_variance = params[1].exp().max(1e-10);
        }
    }

    fn clone_box(&self) -> Box<dyn SurrogateKernel> {
        Box::new(self.clone())
    }

    fn name(&self) -> &str {
        "RBF"
    }
}

// ---------------------------------------------------------------------------
// Matern kernel family
// ---------------------------------------------------------------------------

/// Variant of the Matern kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaternVariant {
    /// nu = 1/2 (exponential kernel)
    OneHalf,
    /// nu = 3/2
    ThreeHalves,
    /// nu = 5/2
    FiveHalves,
}

/// Matern kernel with selectable smoothness parameter nu.
///
/// - nu = 1/2: k = sigma^2 * exp(-r / l)  (once-differentiable)
/// - nu = 3/2: k = sigma^2 * (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)
/// - nu = 5/2: k = sigma^2 * (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)
#[derive(Debug, Clone)]
pub struct MaternKernel {
    /// Smoothness parameter
    pub variant: MaternVariant,
    /// Length scale
    pub length_scale: f64,
    /// Signal variance
    pub signal_variance: f64,
}

impl MaternKernel {
    pub fn new(variant: MaternVariant, length_scale: f64, signal_variance: f64) -> Self {
        Self {
            variant,
            length_scale: length_scale.max(1e-10),
            signal_variance: signal_variance.max(1e-10),
        }
    }

    /// Create a Matern-1/2 kernel.
    pub fn one_half(length_scale: f64, signal_variance: f64) -> Self {
        Self::new(MaternVariant::OneHalf, length_scale, signal_variance)
    }

    /// Create a Matern-3/2 kernel.
    pub fn three_halves(length_scale: f64, signal_variance: f64) -> Self {
        Self::new(MaternVariant::ThreeHalves, length_scale, signal_variance)
    }

    /// Create a Matern-5/2 kernel.
    pub fn five_halves(length_scale: f64, signal_variance: f64) -> Self {
        Self::new(MaternVariant::FiveHalves, length_scale, signal_variance)
    }
}

impl Default for MaternKernel {
    fn default() -> Self {
        Self::new(MaternVariant::FiveHalves, 1.0, 1.0)
    }
}

impl SurrogateKernel for MaternKernel {
    fn eval(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        let r = squared_distance(x1, x2).sqrt();
        let l = self.length_scale;
        let sv = self.signal_variance;

        match self.variant {
            MaternVariant::OneHalf => sv * (-r / l).exp(),
            MaternVariant::ThreeHalves => {
                let sqrt3_r_l = 3.0_f64.sqrt() * r / l;
                sv * (1.0 + sqrt3_r_l) * (-sqrt3_r_l).exp()
            }
            MaternVariant::FiveHalves => {
                let sqrt5_r_l = 5.0_f64.sqrt() * r / l;
                let r2_l2 = r * r / (l * l);
                sv * (1.0 + sqrt5_r_l + 5.0 * r2_l2 / 3.0) * (-sqrt5_r_l).exp()
            }
        }
    }

    fn get_log_params(&self) -> Vec<f64> {
        vec![self.length_scale.ln(), self.signal_variance.ln()]
    }

    fn set_log_params(&mut self, params: &[f64]) {
        if params.len() >= 2 {
            self.length_scale = params[0].exp().max(1e-10);
            self.signal_variance = params[1].exp().max(1e-10);
        }
    }

    fn clone_box(&self) -> Box<dyn SurrogateKernel> {
        Box::new(self.clone())
    }

    fn name(&self) -> &str {
        match self.variant {
            MaternVariant::OneHalf => "Matern12",
            MaternVariant::ThreeHalves => "Matern32",
            MaternVariant::FiveHalves => "Matern52",
        }
    }
}

// ---------------------------------------------------------------------------
// Rational Quadratic kernel
// ---------------------------------------------------------------------------

/// Rational Quadratic kernel.
///
/// k(x, x') = sigma^2 * (1 + ||x - x'||^2 / (2 * alpha * l^2))^(-alpha)
///
/// This can be seen as a scale mixture of RBF kernels with different length
/// scales. The parameter `alpha` controls the relative weighting of large-scale
/// vs small-scale variations.
#[derive(Debug, Clone)]
pub struct RationalQuadraticKernel {
    /// Length scale
    pub length_scale: f64,
    /// Signal variance
    pub signal_variance: f64,
    /// Shape parameter (alpha); larger alpha => closer to RBF
    pub alpha: f64,
}

impl RationalQuadraticKernel {
    pub fn new(length_scale: f64, signal_variance: f64, alpha: f64) -> Self {
        Self {
            length_scale: length_scale.max(1e-10),
            signal_variance: signal_variance.max(1e-10),
            alpha: alpha.max(1e-10),
        }
    }
}

impl Default for RationalQuadraticKernel {
    fn default() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }
}

impl SurrogateKernel for RationalQuadraticKernel {
    fn eval(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        let sq_dist = squared_distance(x1, x2);
        let base = 1.0 + sq_dist / (2.0 * self.alpha * self.length_scale * self.length_scale);
        self.signal_variance * base.powf(-self.alpha)
    }

    fn get_log_params(&self) -> Vec<f64> {
        vec![
            self.length_scale.ln(),
            self.signal_variance.ln(),
            self.alpha.ln(),
        ]
    }

    fn set_log_params(&mut self, params: &[f64]) {
        if params.len() >= 3 {
            self.length_scale = params[0].exp().max(1e-10);
            self.signal_variance = params[1].exp().max(1e-10);
            self.alpha = params[2].exp().max(1e-10);
        }
    }

    fn clone_box(&self) -> Box<dyn SurrogateKernel> {
        Box::new(self.clone())
    }

    fn name(&self) -> &str {
        "RationalQuadratic"
    }
}

// ---------------------------------------------------------------------------
// Composite kernels: Sum and Product
// ---------------------------------------------------------------------------

/// Sum of two kernels: k(x,x') = k1(x,x') + k2(x,x')
#[derive(Clone)]
pub struct SumKernel {
    pub kernel1: Box<dyn SurrogateKernel>,
    pub kernel2: Box<dyn SurrogateKernel>,
}

impl SumKernel {
    pub fn new(k1: Box<dyn SurrogateKernel>, k2: Box<dyn SurrogateKernel>) -> Self {
        Self {
            kernel1: k1,
            kernel2: k2,
        }
    }
}

impl SurrogateKernel for SumKernel {
    fn eval(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        self.kernel1.eval(x1, x2) + self.kernel2.eval(x1, x2)
    }

    fn get_log_params(&self) -> Vec<f64> {
        let mut p = self.kernel1.get_log_params();
        p.extend(self.kernel2.get_log_params());
        p
    }

    fn set_log_params(&mut self, params: &[f64]) {
        let n1 = self.kernel1.n_params();
        if params.len() >= n1 {
            self.kernel1.set_log_params(&params[..n1]);
        }
        if params.len() > n1 {
            self.kernel2.set_log_params(&params[n1..]);
        }
    }

    fn clone_box(&self) -> Box<dyn SurrogateKernel> {
        Box::new(self.clone())
    }

    fn name(&self) -> &str {
        "Sum"
    }
}

/// Product of two kernels: k(x,x') = k1(x,x') * k2(x,x')
#[derive(Clone)]
pub struct ProductKernel {
    pub kernel1: Box<dyn SurrogateKernel>,
    pub kernel2: Box<dyn SurrogateKernel>,
}

impl ProductKernel {
    pub fn new(k1: Box<dyn SurrogateKernel>, k2: Box<dyn SurrogateKernel>) -> Self {
        Self {
            kernel1: k1,
            kernel2: k2,
        }
    }
}

impl SurrogateKernel for ProductKernel {
    fn eval(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        self.kernel1.eval(x1, x2) * self.kernel2.eval(x1, x2)
    }

    fn get_log_params(&self) -> Vec<f64> {
        let mut p = self.kernel1.get_log_params();
        p.extend(self.kernel2.get_log_params());
        p
    }

    fn set_log_params(&mut self, params: &[f64]) {
        let n1 = self.kernel1.n_params();
        if params.len() >= n1 {
            self.kernel1.set_log_params(&params[..n1]);
        }
        if params.len() > n1 {
            self.kernel2.set_log_params(&params[n1..]);
        }
    }

    fn clone_box(&self) -> Box<dyn SurrogateKernel> {
        Box::new(self.clone())
    }

    fn name(&self) -> &str {
        "Product"
    }
}

// ---------------------------------------------------------------------------
// Gaussian Process Surrogate
// ---------------------------------------------------------------------------

/// Configuration for the GP surrogate.
#[derive(Clone)]
pub struct GpSurrogateConfig {
    /// Noise variance added to the diagonal for numerical stability.
    pub noise_variance: f64,
    /// Whether to optimise kernel hyperparameters via marginal likelihood.
    pub optimize_hyperparams: bool,
    /// Number of random restarts for hyperparameter optimization.
    pub n_restarts: usize,
    /// Maximum number of L-BFGS iterations per restart.
    pub max_opt_iters: usize,
}

impl Default for GpSurrogateConfig {
    fn default() -> Self {
        Self {
            noise_variance: 1e-6,
            optimize_hyperparams: true,
            n_restarts: 3,
            max_opt_iters: 50,
        }
    }
}

/// Gaussian Process surrogate model for Bayesian optimization.
///
/// Maintains training data and the fitted model (Cholesky factor + alpha vector)
/// for efficient prediction.
pub struct GpSurrogate {
    /// Kernel function
    kernel: Box<dyn SurrogateKernel>,
    /// Configuration
    config: GpSurrogateConfig,
    /// Training inputs (n_train x n_dims)
    x_train: Option<Array2<f64>>,
    /// Training targets (n_train,)
    y_train: Option<Array1<f64>>,
    /// Mean of training targets (for standardization)
    y_mean: f64,
    /// Std of training targets (for standardization)
    y_std: f64,
    /// Lower-triangular Cholesky factor of K + noise*I
    l_factor: Option<Array2<f64>>,
    /// Alpha = L^T \ (L \ y_centered)
    alpha: Option<Array1<f64>>,
}

impl GpSurrogate {
    /// Create a new GP surrogate with the given kernel.
    pub fn new(kernel: Box<dyn SurrogateKernel>, config: GpSurrogateConfig) -> Self {
        Self {
            kernel,
            config,
            x_train: None,
            y_train: None,
            y_mean: 0.0,
            y_std: 1.0,
            l_factor: None,
            alpha: None,
        }
    }

    /// Create a GP surrogate with an RBF kernel and default configuration.
    pub fn default_rbf() -> Self {
        Self::new(Box::new(RbfKernel::default()), GpSurrogateConfig::default())
    }

    /// Fit the GP to training data.
    ///
    /// Optionally optimises kernel hyperparameters via marginal likelihood maximisation.
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> OptimizeResult<()> {
        if x.nrows() != y.len() {
            return Err(OptimizeError::InvalidInput(format!(
                "x has {} rows but y has {} elements",
                x.nrows(),
                y.len()
            )));
        }
        if x.nrows() == 0 {
            return Err(OptimizeError::InvalidInput(
                "Cannot fit GP with zero training samples".to_string(),
            ));
        }

        self.x_train = Some(x.clone());
        self.y_train = Some(y.clone());

        // Standardise targets
        self.y_mean = y.iter().sum::<f64>() / y.len() as f64;
        let variance = y.iter().map(|&v| (v - self.y_mean).powi(2)).sum::<f64>() / y.len() as f64;
        self.y_std = if variance > 1e-12 {
            variance.sqrt()
        } else {
            1.0
        };

        // Optimise hyperparameters if requested
        if self.config.optimize_hyperparams && x.nrows() >= 3 {
            self.optimize_hyperparameters()?;
        }

        // Compute Cholesky factor and alpha with current kernel
        self.update_model()
    }

    /// Add new observations incrementally and refit the model.
    pub fn update(&mut self, x_new: &Array2<f64>, y_new: &Array1<f64>) -> OptimizeResult<()> {
        if x_new.nrows() != y_new.len() {
            return Err(OptimizeError::InvalidInput(
                "x_new and y_new must have same number of rows".to_string(),
            ));
        }

        match (&self.x_train, &self.y_train) {
            (Some(xt), Some(yt)) => {
                let mut x_all = Array2::zeros((xt.nrows() + x_new.nrows(), xt.ncols()));
                for i in 0..xt.nrows() {
                    for j in 0..xt.ncols() {
                        x_all[[i, j]] = xt[[i, j]];
                    }
                }
                for i in 0..x_new.nrows() {
                    for j in 0..x_new.ncols() {
                        x_all[[xt.nrows() + i, j]] = x_new[[i, j]];
                    }
                }
                let mut y_all = Array1::zeros(yt.len() + y_new.len());
                for i in 0..yt.len() {
                    y_all[i] = yt[i];
                }
                for i in 0..y_new.len() {
                    y_all[yt.len() + i] = y_new[i];
                }
                self.fit(&x_all, &y_all)
            }
            _ => self.fit(x_new, y_new),
        }
    }

    /// Predict the GP mean at test points.
    pub fn predict_mean(&self, x_test: &Array2<f64>) -> OptimizeResult<Array1<f64>> {
        let (mean, _) = self.predict(x_test)?;
        Ok(mean)
    }

    /// Predict the GP variance at test points.
    pub fn predict_variance(&self, x_test: &Array2<f64>) -> OptimizeResult<Array1<f64>> {
        let (_, var) = self.predict(x_test)?;
        Ok(var)
    }

    /// Predict both mean and variance at test points.
    pub fn predict(&self, x_test: &Array2<f64>) -> OptimizeResult<(Array1<f64>, Array1<f64>)> {
        let x_train = self.x_train.as_ref().ok_or_else(|| {
            OptimizeError::ComputationError("GP must be fitted before prediction".to_string())
        })?;
        let alpha = self
            .alpha
            .as_ref()
            .ok_or_else(|| OptimizeError::ComputationError("GP model not fitted".to_string()))?;
        let l_factor = self
            .l_factor
            .as_ref()
            .ok_or_else(|| OptimizeError::ComputationError("GP model not fitted".to_string()))?;

        // K(X_test, X_train)
        let k_star = self.kernel.cross_covariance(x_test, x_train);

        // Standardised mean: k_star @ alpha
        let mean_std = k_star.dot(alpha);

        // De-standardise
        let mean = mean_std.mapv(|v| v * self.y_std + self.y_mean);

        // Variance: k(x*, x*) - k_star @ K^{-1} @ k_star^T
        // Using: v = L \ k_star^T, variance = k_self - ||v||^2
        let n_test = x_test.nrows();
        let mut variance = Array1::zeros(n_test);

        for i in 0..n_test {
            let k_self = self.kernel.eval(&x_test.row(i), &x_test.row(i));

            // Solve L v = k_star[i, :] using forward substitution
            let k_col = k_star.row(i).to_owned();
            let v = forward_solve(l_factor, &k_col)?;

            let v_sq_sum: f64 = v.iter().map(|&vi| vi * vi).sum();
            let var = (k_self - v_sq_sum).max(0.0);
            variance[i] = var * self.y_std * self.y_std;
        }

        Ok((mean, variance))
    }

    /// Predict mean and standard deviation at a single point.
    pub fn predict_single(&self, x: &ArrayView1<f64>) -> OptimizeResult<(f64, f64)> {
        let x_mat = x
            .to_owned()
            .into_shape_with_order((1, x.len()))
            .map_err(|e| OptimizeError::ComputationError(format!("Shape error: {}", e)))?;
        let (mean, var) = self.predict(&x_mat)?;
        Ok((mean[0], var[0].max(0.0).sqrt()))
    }

    /// Compute the log marginal likelihood of the current model.
    pub fn log_marginal_likelihood(&self) -> OptimizeResult<f64> {
        let y_train = self
            .y_train
            .as_ref()
            .ok_or_else(|| OptimizeError::ComputationError("GP must be fitted".to_string()))?;
        let l_factor = self
            .l_factor
            .as_ref()
            .ok_or_else(|| OptimizeError::ComputationError("GP model not fitted".to_string()))?;
        let alpha = self
            .alpha
            .as_ref()
            .ok_or_else(|| OptimizeError::ComputationError("GP model not fitted".to_string()))?;

        let y_std = &self.standardize_y(y_train);
        let n = y_std.len() as f64;

        // -0.5 * y^T alpha
        let data_fit = -0.5 * y_std.dot(alpha);

        // -sum(log(diag(L)))
        let log_det: f64 = l_factor.diag().iter().map(|&v| v.abs().ln()).sum();

        // -0.5 * n * log(2 pi)
        let norm = -0.5 * n * (2.0 * std::f64::consts::PI).ln();

        Ok(data_fit - log_det + norm)
    }

    /// Return a reference to the current kernel.
    pub fn kernel(&self) -> &dyn SurrogateKernel {
        self.kernel.as_ref()
    }

    /// Return a mutable reference to the kernel.
    pub fn kernel_mut(&mut self) -> &mut dyn SurrogateKernel {
        self.kernel.as_mut()
    }

    /// Number of training samples.
    pub fn n_train(&self) -> usize {
        self.x_train.as_ref().map_or(0, |x| x.nrows())
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Standardise y-values: (y - mean) / std
    fn standardize_y(&self, y: &Array1<f64>) -> Array1<f64> {
        y.mapv(|v| (v - self.y_mean) / self.y_std)
    }

    /// Recompute Cholesky factor and alpha from current kernel + data.
    fn update_model(&mut self) -> OptimizeResult<()> {
        let x_train = self
            .x_train
            .as_ref()
            .ok_or_else(|| OptimizeError::ComputationError("No training data".to_string()))?;
        let y_train = self
            .y_train
            .as_ref()
            .ok_or_else(|| OptimizeError::ComputationError("No training data".to_string()))?;

        let y_std = self.standardize_y(y_train);

        // Build covariance matrix
        let mut k = self.kernel.covariance_matrix(x_train);
        let n = k.nrows();

        // Add noise to diagonal
        for i in 0..n {
            k[[i, i]] += self.config.noise_variance;
        }

        // Cholesky decomposition with jitter fallback
        let l = match cholesky(&k) {
            Ok(l) => l,
            Err(_) => {
                // Try increasing jitter
                let jitters = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2];
                let mut result = Err(OptimizeError::ComputationError(
                    "Cholesky failed with all jitter levels".to_string(),
                ));
                for &jitter in &jitters {
                    for i in 0..n {
                        k[[i, i]] += jitter;
                    }
                    match cholesky(&k) {
                        Ok(l) => {
                            result = Ok(l);
                            break;
                        }
                        Err(_) => continue,
                    }
                }
                result?
            }
        };

        // Solve L alpha1 = y_std (forward substitution)
        let alpha1 = forward_solve(&l, &y_std)?;
        // Solve L^T alpha = alpha1 (backward substitution)
        let alpha = backward_solve_transpose(&l, &alpha1)?;

        self.l_factor = Some(l);
        self.alpha = Some(alpha);

        Ok(())
    }

    /// Optimise kernel hyperparameters by maximising the log marginal likelihood.
    ///
    /// Uses a simple coordinate-wise golden-section search on each log-parameter
    /// with random restarts. This avoids depending on external optimisers and
    /// keeps the implementation self-contained.
    fn optimize_hyperparameters(&mut self) -> OptimizeResult<()> {
        let x_train = self
            .x_train
            .as_ref()
            .ok_or_else(|| OptimizeError::ComputationError("No training data".to_string()))?
            .clone();
        let y_train = self
            .y_train
            .as_ref()
            .ok_or_else(|| OptimizeError::ComputationError("No training data".to_string()))?
            .clone();
        let y_std = self.standardize_y(&y_train);

        let n_params = self.kernel.n_params();
        if n_params == 0 {
            return Ok(());
        }

        let mut best_params = self.kernel.get_log_params();
        let mut best_lml = f64::NEG_INFINITY;

        // Evaluate current params
        if let Ok(lml) = self.eval_lml_at_params(&best_params, &x_train, &y_std) {
            best_lml = lml;
        }

        let mut rng = scirs2_core::random::rng();

        // Random restarts
        for restart in 0..self.config.n_restarts {
            let init_params: Vec<f64> = if restart == 0 {
                best_params.clone()
            } else {
                (0..n_params).map(|_| rng.random_range(-2.0..2.0)).collect()
            };

            // Coordinate-wise optimization
            let mut current_params = init_params;
            for _iter in 0..self.config.max_opt_iters {
                let mut improved = false;
                for p in 0..n_params {
                    let original = current_params[p];

                    // Try a few steps in each direction
                    let steps = [0.1, 0.3, 1.0, -0.1, -0.3, -1.0];
                    let mut best_step_lml =
                        match self.eval_lml_at_params(&current_params, &x_train, &y_std) {
                            Ok(v) => v,
                            Err(_) => f64::NEG_INFINITY,
                        };
                    let mut best_step_val = original;

                    for &step in &steps {
                        current_params[p] = original + step;
                        // Clamp to reasonable range
                        current_params[p] = current_params[p].clamp(-5.0, 5.0);

                        if let Ok(lml) = self.eval_lml_at_params(&current_params, &x_train, &y_std)
                        {
                            if lml > best_step_lml {
                                best_step_lml = lml;
                                best_step_val = current_params[p];
                                improved = true;
                            }
                        }
                    }
                    current_params[p] = best_step_val;
                }
                if !improved {
                    break;
                }
            }

            // Evaluate final
            if let Ok(lml) = self.eval_lml_at_params(&current_params, &x_train, &y_std) {
                if lml > best_lml {
                    best_lml = lml;
                    best_params = current_params;
                }
            }
        }

        // Set best params
        self.kernel.set_log_params(&best_params);

        Ok(())
    }

    /// Evaluate the log marginal likelihood at given (log) kernel parameters
    /// without mutating the surrogate.
    fn eval_lml_at_params(
        &self,
        log_params: &[f64],
        x_train: &Array2<f64>,
        y_std: &Array1<f64>,
    ) -> OptimizeResult<f64> {
        let mut kernel = self.kernel.clone();
        kernel.set_log_params(log_params);

        let mut k = kernel.covariance_matrix(x_train);
        let n = k.nrows();
        for i in 0..n {
            k[[i, i]] += self.config.noise_variance;
        }

        let l = cholesky(&k)?;
        let alpha1 = forward_solve(&l, y_std)?;
        let alpha = backward_solve_transpose(&l, &alpha1)?;

        let n_f = n as f64;
        let data_fit = -0.5 * y_std.dot(&alpha);
        let log_det: f64 = l.diag().iter().map(|&v| v.abs().ln()).sum();
        let norm = -0.5 * n_f * (2.0 * std::f64::consts::PI).ln();

        Ok(data_fit - log_det + norm)
    }
}

// ---------------------------------------------------------------------------
// Linear algebra helpers (pure Rust, no unwrap)
// ---------------------------------------------------------------------------

/// Squared Euclidean distance between two vectors.
fn squared_distance(x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
    let mut s = 0.0;
    for i in 0..x1.len() {
        let d = x1[i] - x2[i];
        s += d * d;
    }
    s
}

/// Cholesky decomposition: A = L L^T  (lower triangular).
fn cholesky(a: &Array2<f64>) -> OptimizeResult<Array2<f64>> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(OptimizeError::ComputationError(
            "Cholesky: matrix must be square".to_string(),
        ));
    }
    let mut l = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0;
            for k in 0..j {
                s += l[[i, k]] * l[[j, k]];
            }
            if i == j {
                let diag = a[[i, i]] - s;
                if diag <= 0.0 {
                    return Err(OptimizeError::ComputationError(format!(
                        "Cholesky: matrix not positive-definite (diag[{}] = {:.6e})",
                        i, diag
                    )));
                }
                l[[i, j]] = diag.sqrt();
            } else {
                l[[i, j]] = (a[[i, j]] - s) / l[[j, j]];
            }
        }
    }
    Ok(l)
}

/// Forward substitution: solve L x = b where L is lower-triangular.
fn forward_solve(l: &Array2<f64>, b: &Array1<f64>) -> OptimizeResult<Array1<f64>> {
    let n = l.nrows();
    let mut x = Array1::zeros(n);
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..i {
            s += l[[i, j]] * x[j];
        }
        let diag = l[[i, i]];
        if diag.abs() < 1e-15 {
            return Err(OptimizeError::ComputationError(
                "Forward solve: near-zero diagonal".to_string(),
            ));
        }
        x[i] = (b[i] - s) / diag;
    }
    Ok(x)
}

/// Backward substitution: solve L^T x = b where L is lower-triangular.
fn backward_solve_transpose(l: &Array2<f64>, b: &Array1<f64>) -> OptimizeResult<Array1<f64>> {
    let n = l.nrows();
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut s = 0.0;
        for j in (i + 1)..n {
            s += l[[j, i]] * x[j]; // L^T[i,j] = L[j,i]
        }
        let diag = l[[i, i]];
        if diag.abs() < 1e-15 {
            return Err(OptimizeError::ComputationError(
                "Backward solve: near-zero diagonal".to_string(),
            ));
        }
        x[i] = (b[i] - s) / diag;
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn make_train_data() -> (Array2<f64>, Array1<f64>) {
        // f(x) = sin(x)  sampled at 5 points
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).expect("shape ok");
        let y = array![0.0, 0.841, 0.909, 0.141, -0.757];
        (x, y)
    }

    #[test]
    fn test_rbf_kernel_symmetry() {
        let k = RbfKernel::default();
        let a = array![1.0, 2.0];
        let b = array![3.0, 4.0];
        assert!((k.eval(&a.view(), &b.view()) - k.eval(&b.view(), &a.view())).abs() < 1e-14);
    }

    #[test]
    fn test_rbf_kernel_self_covariance() {
        let k = RbfKernel::new(1.0, 2.0);
        let a = array![1.0, 2.0];
        // Self-covariance should be signal_variance
        assert!((k.eval(&a.view(), &a.view()) - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_matern_variants() {
        let a = array![0.0];
        let b = array![1.0];

        for variant in &[
            MaternVariant::OneHalf,
            MaternVariant::ThreeHalves,
            MaternVariant::FiveHalves,
        ] {
            let k = MaternKernel::new(*variant, 1.0, 1.0);
            let val = k.eval(&a.view(), &b.view());
            assert!(val > 0.0 && val < 1.0, "Matern({:?}) = {}", variant, val);
            // Self-covariance = signal_variance
            assert!((k.eval(&a.view(), &a.view()) - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn test_rational_quadratic_kernel() {
        let k = RationalQuadraticKernel::new(1.0, 1.0, 1.0);
        let a = array![0.0];
        let b = array![1.0];
        let val = k.eval(&a.view(), &b.view());
        // Should be (1 + 0.5)^{-1} = 2/3
        assert!((val - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_rational_quadratic_approaches_rbf() {
        // As alpha -> infinity, RQ -> RBF
        let rbf = RbfKernel::new(1.0, 1.0);
        let rq = RationalQuadraticKernel::new(1.0, 1.0, 1e6);
        let a = array![0.0, 1.0];
        let b = array![2.0, 3.0];

        let rbf_val = rbf.eval(&a.view(), &b.view());
        let rq_val = rq.eval(&a.view(), &b.view());
        assert!(
            (rbf_val - rq_val).abs() < 1e-4,
            "RBF={}, RQ(alpha=1e6)={}",
            rbf_val,
            rq_val
        );
    }

    #[test]
    fn test_sum_kernel() {
        let k1 = Box::new(RbfKernel::new(1.0, 1.0));
        let k2 = Box::new(MaternKernel::five_halves(1.0, 0.5));
        let sum = SumKernel::new(k1.clone(), k2.clone());

        let a = array![1.0];
        let b = array![2.0];
        let expected = k1.eval(&a.view(), &b.view()) + k2.eval(&a.view(), &b.view());
        assert!((sum.eval(&a.view(), &b.view()) - expected).abs() < 1e-14);
    }

    #[test]
    fn test_product_kernel() {
        let k1 = Box::new(RbfKernel::new(1.0, 1.0));
        let k2 = Box::new(MaternKernel::five_halves(1.0, 0.5));
        let prod = ProductKernel::new(k1.clone(), k2.clone());

        let a = array![1.0];
        let b = array![2.0];
        let expected = k1.eval(&a.view(), &b.view()) * k2.eval(&a.view(), &b.view());
        assert!((prod.eval(&a.view(), &b.view()) - expected).abs() < 1e-14);
    }

    #[test]
    fn test_gp_fit_predict_basic() {
        let (x, y) = make_train_data();
        let mut gp = GpSurrogate::new(
            Box::new(RbfKernel::default()),
            GpSurrogateConfig {
                optimize_hyperparams: false,
                noise_variance: 1e-4,
                ..Default::default()
            },
        );
        gp.fit(&x, &y).expect("fit should succeed");

        // Predict at training points -> should be close to training values
        let (mean, var) = gp.predict(&x).expect("predict should succeed");
        for i in 0..y.len() {
            assert!(
                (mean[i] - y[i]).abs() < 0.15,
                "mean[{}]={:.4} vs y[{}]={:.4}",
                i,
                mean[i],
                i,
                y[i]
            );
            // Variance should be small at training points
            assert!(
                var[i] < 0.5,
                "var[{}]={:.4} should be small at training point",
                i,
                var[i]
            );
        }
    }

    #[test]
    fn test_gp_uncertainty_away_from_data() {
        let (x, y) = make_train_data();
        let mut gp = GpSurrogate::new(
            Box::new(RbfKernel::default()),
            GpSurrogateConfig {
                optimize_hyperparams: false,
                noise_variance: 1e-4,
                ..Default::default()
            },
        );
        gp.fit(&x, &y).expect("fit should succeed");

        // Predict far from training data
        let x_far = Array2::from_shape_vec((1, 1), vec![10.0]).expect("shape ok");
        let (_, var_far) = gp.predict(&x_far).expect("predict ok");

        // Predict at training data
        let x_near = Array2::from_shape_vec((1, 1), vec![2.0]).expect("shape ok");
        let (_, var_near) = gp.predict(&x_near).expect("predict ok");

        // Uncertainty should be higher far from data
        assert!(
            var_far[0] > var_near[0],
            "var_far={:.4} should be > var_near={:.4}",
            var_far[0],
            var_near[0]
        );
    }

    #[test]
    fn test_gp_predict_single() {
        let (x, y) = make_train_data();
        let mut gp = GpSurrogate::default_rbf();
        gp.config.optimize_hyperparams = false;
        gp.config.noise_variance = 1e-4;
        gp.fit(&x, &y).expect("fit ok");

        let point = array![1.5];
        let (mean, std) = gp.predict_single(&point.view()).expect("predict_single ok");
        assert!(mean.is_finite());
        assert!(std >= 0.0);
    }

    #[test]
    fn test_gp_log_marginal_likelihood() {
        let (x, y) = make_train_data();
        let mut gp = GpSurrogate::new(
            Box::new(RbfKernel::default()),
            GpSurrogateConfig {
                optimize_hyperparams: false,
                noise_variance: 1e-4,
                ..Default::default()
            },
        );
        gp.fit(&x, &y).expect("fit ok");

        let lml = gp.log_marginal_likelihood().expect("lml ok");
        assert!(lml.is_finite(), "LML should be finite, got {}", lml);
    }

    #[test]
    fn test_gp_update_incremental() {
        let (x, y) = make_train_data();
        let mut gp = GpSurrogate::new(
            Box::new(RbfKernel::default()),
            GpSurrogateConfig {
                optimize_hyperparams: false,
                noise_variance: 1e-4,
                ..Default::default()
            },
        );
        gp.fit(&x, &y).expect("fit ok");
        assert_eq!(gp.n_train(), 5);

        // Add one more point
        let x_new = Array2::from_shape_vec((1, 1), vec![5.0]).expect("shape ok");
        let y_new = array![-0.959];
        gp.update(&x_new, &y_new).expect("update ok");
        assert_eq!(gp.n_train(), 6);
    }

    #[test]
    fn test_gp_hyperparameter_optimization() {
        let (x, y) = make_train_data();
        let mut gp = GpSurrogate::new(
            Box::new(RbfKernel::default()),
            GpSurrogateConfig {
                optimize_hyperparams: true,
                n_restarts: 2,
                max_opt_iters: 20,
                noise_variance: 1e-4,
            },
        );
        gp.fit(&x, &y).expect("fit with optimization ok");

        // Just verify it completes without error and produces finite predictions
        let x_test = Array2::from_shape_vec((1, 1), vec![1.5]).expect("shape ok");
        let (mean, var) = gp.predict(&x_test).expect("predict ok");
        assert!(mean[0].is_finite());
        assert!(var[0].is_finite());
    }

    #[test]
    fn test_cholesky_positive_definite() {
        let a = Array2::from_shape_vec((2, 2), vec![4.0, 2.0, 2.0, 3.0]).expect("shape ok");
        let l = cholesky(&a).expect("should succeed");
        // Verify L L^T = A
        let reconstructed = l.dot(&l.t());
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (reconstructed[[i, j]] - a[[i, j]]).abs() < 1e-10,
                    "Mismatch at [{},{}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_cholesky_non_pd_fails() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 10.0, 10.0, 1.0]).expect("shape ok");
        assert!(cholesky(&a).is_err());
    }

    #[test]
    fn test_kernel_log_params_roundtrip() {
        let mut k = RbfKernel::new(2.5, 0.3);
        let params = k.get_log_params();
        k.set_log_params(&params);
        assert!((k.length_scale - 2.5).abs() < 1e-10);
        assert!((k.signal_variance - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_matern_kernel_covariance_matrix() {
        let k = MaternKernel::three_halves(1.0, 1.0);
        let x = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).expect("shape ok");
        let cov = k.covariance_matrix(&x);
        assert_eq!(cov.nrows(), 3);
        assert_eq!(cov.ncols(), 3);
        // Symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (cov[[i, j]] - cov[[j, i]]).abs() < 1e-14,
                    "Not symmetric at [{},{}]",
                    i,
                    j
                );
            }
        }
        // Positive diagonal
        for i in 0..3 {
            assert!(cov[[i, i]] > 0.0);
        }
    }

    #[test]
    fn test_gp_multidimensional() {
        // 2D function: f(x,y) = x^2 + y^2
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, 0.5, 0.5],
        )
        .expect("shape ok");
        let y = array![0.0, 1.0, 1.0, 1.0, 1.0, 0.5];

        let mut gp = GpSurrogate::new(
            Box::new(RbfKernel::default()),
            GpSurrogateConfig {
                optimize_hyperparams: false,
                noise_variance: 1e-4,
                ..Default::default()
            },
        );
        gp.fit(&x, &y).expect("fit ok");

        // Predict at origin (should be close to 0)
        let x_test = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).expect("shape ok");
        let (mean, _) = gp.predict(&x_test).expect("predict ok");
        assert!(
            mean[0].abs() < 0.3,
            "Prediction at origin should be close to 0, got {}",
            mean[0]
        );
    }
}
