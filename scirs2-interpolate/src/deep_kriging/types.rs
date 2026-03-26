//! Types for Deep Kriging and GP Surrogate modules.
//!
//! This module defines configuration types, kernel specifications,
//! acquisition functions, and result containers for neural-basis kriging
//! and Gaussian process surrogate modelling.

// ---------------------------------------------------------------------------
// Activation function
// ---------------------------------------------------------------------------

/// Activation function used in MLP layers.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Activation {
    /// Rectified Linear Unit: max(0, x)
    ReLU,
    /// Hyperbolic tangent
    Tanh,
    /// Logistic sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Exponential Linear Unit: x if x > 0 else alpha*(exp(x)-1)
    ELU { alpha: f64 },
}

impl Activation {
    /// Apply the activation function element-wise.
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::Tanh => x.tanh(),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::ELU { alpha } => {
                if x > 0.0 {
                    x
                } else {
                    alpha * (x.exp() - 1.0)
                }
            }
        }
    }

    /// Derivative of the activation function.
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            Activation::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            Activation::ELU { alpha } => {
                if x > 0.0 {
                    1.0
                } else {
                    alpha * x.exp()
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Deep Kriging config
// ---------------------------------------------------------------------------

/// Configuration for Neural Basis Kriging (Deep Kriging).
///
/// An MLP learns nonlinear basis functions that are fed into ordinary kriging.
#[derive(Debug, Clone)]
pub struct DeepKrigingConfig {
    /// Sizes of hidden layers in the MLP (e.g. `[32, 16]`).
    pub hidden_layers: Vec<usize>,
    /// Learning rate for gradient descent on MLP weights.
    pub learning_rate: f64,
    /// Number of training epochs (alternating optimisation steps).
    pub epochs: usize,
    /// Activation function used between layers.
    pub activation: Activation,
    /// Dimension of the basis output (last MLP layer).
    pub basis_dim: usize,
    /// Seed for reproducible weight initialisation.
    pub seed: u64,
}

impl Default for DeepKrigingConfig {
    fn default() -> Self {
        Self {
            hidden_layers: vec![32, 16],
            learning_rate: 0.01,
            epochs: 100,
            activation: Activation::Tanh,
            basis_dim: 8,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel types for GP surrogate
// ---------------------------------------------------------------------------

/// Kernel (covariance function) type for Gaussian process surrogate.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum KernelType {
    /// Squared exponential (RBF) kernel:
    /// k(x, x') = variance * exp(-||x-x'||^2 / (2 * lengthscale^2))
    SquaredExponential {
        /// Characteristic lengthscale.
        lengthscale: f64,
        /// Signal variance.
        variance: f64,
    },
    /// Matern kernel with smoothness parameter nu.
    /// Supported nu values: 0.5 (exponential), 1.5, 2.5.
    Matern {
        /// Smoothness parameter (0.5, 1.5, or 2.5).
        nu: f64,
        /// Characteristic lengthscale.
        lengthscale: f64,
        /// Signal variance.
        variance: f64,
    },
    /// Rational quadratic kernel:
    /// k(x, x') = variance * (1 + ||x-x'||^2 / (2 * alpha * lengthscale^2))^(-alpha)
    RationalQuadratic {
        /// Scale mixture parameter.
        alpha: f64,
        /// Characteristic lengthscale.
        lengthscale: f64,
        /// Signal variance.
        variance: f64,
    },
}

impl Default for KernelType {
    fn default() -> Self {
        KernelType::SquaredExponential {
            lengthscale: 1.0,
            variance: 1.0,
        }
    }
}

impl KernelType {
    /// Evaluate the kernel between two points.
    pub fn evaluate(&self, x: &[f64], xp: &[f64]) -> f64 {
        let sq_dist: f64 = x
            .iter()
            .zip(xp.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();

        match self {
            KernelType::SquaredExponential {
                lengthscale,
                variance,
            } => {
                let l2 = lengthscale * lengthscale;
                variance * (-sq_dist / (2.0 * l2)).exp()
            }
            KernelType::Matern {
                nu,
                lengthscale,
                variance,
            } => {
                let r = sq_dist.sqrt() / lengthscale;
                if r < 1e-12 {
                    return *variance;
                }
                if (*nu - 0.5).abs() < 1e-6 {
                    // Matern 1/2 = exponential
                    variance * (-r).exp()
                } else if (*nu - 1.5).abs() < 1e-6 {
                    // Matern 3/2
                    let s3 = 3.0_f64.sqrt() * r;
                    variance * (1.0 + s3) * (-s3).exp()
                } else if (*nu - 2.5).abs() < 1e-6 {
                    // Matern 5/2
                    let s5 = 5.0_f64.sqrt() * r;
                    variance * (1.0 + s5 + s5 * s5 / 3.0) * (-s5).exp()
                } else {
                    // Fall back to squared exponential for unsupported nu
                    variance * (-sq_dist / (2.0 * lengthscale * lengthscale)).exp()
                }
            }
            KernelType::RationalQuadratic {
                alpha,
                lengthscale,
                variance,
            } => {
                let l2 = lengthscale * lengthscale;
                variance * (1.0 + sq_dist / (2.0 * alpha * l2)).powf(-alpha)
            }
        }
    }

    /// Return the signal variance of the kernel.
    pub fn signal_variance(&self) -> f64 {
        match self {
            KernelType::SquaredExponential { variance, .. } => *variance,
            KernelType::Matern { variance, .. } => *variance,
            KernelType::RationalQuadratic { variance, .. } => *variance,
        }
    }

    /// Return the lengthscale of the kernel.
    pub fn lengthscale(&self) -> f64 {
        match self {
            KernelType::SquaredExponential { lengthscale, .. } => *lengthscale,
            KernelType::Matern { lengthscale, .. } => *lengthscale,
            KernelType::RationalQuadratic { lengthscale, .. } => *lengthscale,
        }
    }
}

// ---------------------------------------------------------------------------
// Acquisition functions
// ---------------------------------------------------------------------------

/// Acquisition function for Bayesian optimisation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum AcquisitionFunction {
    /// Expected Improvement: EI(x) = sigma * [z*Phi(z) + phi(z)]
    EI,
    /// Probability of Improvement.
    PI,
    /// Upper Confidence Bound with exploration weight kappa.
    UCB(f64),
    /// Lower Confidence Bound with exploration weight kappa.
    LCB(f64),
}

impl Default for AcquisitionFunction {
    fn default() -> Self {
        AcquisitionFunction::EI
    }
}

// ---------------------------------------------------------------------------
// GP Surrogate config
// ---------------------------------------------------------------------------

/// Configuration for the Gaussian process surrogate model.
#[derive(Debug, Clone)]
pub struct GPSurrogateConfig {
    /// Covariance kernel.
    pub kernel: KernelType,
    /// Observation noise variance (added to diagonal of K).
    pub noise: f64,
    /// Whether to optimise kernel hyperparameters via marginal likelihood.
    pub optimize_hyperparams: bool,
    /// Number of random restarts for hyperparameter optimisation.
    pub n_restarts: usize,
    /// Maximum number of optimisation iterations per restart.
    pub max_opt_iterations: usize,
}

impl Default for GPSurrogateConfig {
    fn default() -> Self {
        Self {
            kernel: KernelType::default(),
            noise: 1e-6,
            optimize_hyperparams: false,
            n_restarts: 3,
            max_opt_iterations: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of a GP surrogate prediction.
#[derive(Debug, Clone)]
pub struct SurrogateResult {
    /// Predictive means at query points.
    pub predictions: Vec<f64>,
    /// Predictive variances at query points.
    pub variances: Vec<f64>,
    /// Optimised hyperparameters (kernel parameters + noise).
    pub hyperparameters: Vec<f64>,
}
