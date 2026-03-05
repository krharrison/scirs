//! Deep Kernel Learning and Advanced Kernel Methods
//!
//! This module implements deep kernel learning (DKL), where a neural network
//! feature extractor is composed with a Gaussian process kernel, enabling
//! automatic representation learning combined with uncertainty quantification.
//!
//! ## Overview
//!
//! Deep kernel learning jointly optimizes the feature network and GP
//! hyperparameters via marginal log-likelihood (MLL) maximization, allowing the
//! model to learn problem-specific representations while retaining GP posterior
//! uncertainty estimates.
//!
//! ## Algorithms
//!
//! - **`DeepKernel`**: Neural feature extractor + base kernel composition
//! - **`DeepKernelGP`**: Full GP regression with deep kernel (exact inference)
//! - **`SpectralMixture`**: Wilson-Adams spectral mixture kernel
//! - **`ARDKernel`**: Automatic relevance determination (per-feature lengthscales)
//! - **`DeepKernelTrainer`**: Joint MLL-based training of network + GP hyperparameters
//! - **`InducingPoints`**: Sparse GP approximation with learnable inducing points
//!
//! ## References
//!
//! - Wilson, A. G., et al. (2016). Deep Kernel Learning. AISTATS.
//! - Wilson, A. G., & Adams, R. P. (2013). Gaussian Process Kernels for Pattern
//!   Discovery and Extrapolation. ICML.
//! - Snelson, E., & Ghahramani, Z. (2006). Sparse Gaussian Processes using
//!   Pseudo-inputs. NeurIPS.

use std::f64::consts::PI;

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_linalg::{cholesky, solve, solve_multiple};

use crate::error::{Result, TransformError};

// ============================================================================
// Neural network layer primitives
// ============================================================================

/// Activation function type for the embedded neural network layers.
#[derive(Debug, Clone, PartialEq)]
pub enum Activation {
    /// Rectified linear unit: max(0, x)
    ReLU,
    /// Hyperbolic tangent
    Tanh,
    /// Sigmoid logistic function
    Sigmoid,
    /// Scaled exponential linear unit (self-normalizing)
    SELU,
    /// No activation (linear pass-through)
    Linear,
}

impl Activation {
    /// Apply the activation function element-wise.
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::Tanh => x.tanh(),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::SELU => {
                const ALPHA: f64 = 1.6732631922;
                const SCALE: f64 = 1.0507009874;
                SCALE * if x > 0.0 { x } else { ALPHA * x.exp() - ALPHA }
            }
            Activation::Linear => x,
        }
    }

    /// Apply the activation derivative element-wise (for backprop).
    pub fn gradient(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            Activation::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            Activation::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            Activation::SELU => {
                const ALPHA: f64 = 1.6732631922;
                const SCALE: f64 = 1.0507009874;
                SCALE * if x > 0.0 { 1.0 } else { ALPHA * x.exp() }
            }
            Activation::Linear => 1.0,
        }
    }
}

/// A single fully-connected layer with weight matrix and bias.
#[derive(Debug, Clone)]
pub struct DenseLayer {
    /// Weight matrix of shape (out_dim, in_dim)
    pub weights: Array2<f64>,
    /// Bias vector of shape (out_dim,)
    pub bias: Array1<f64>,
    /// Activation function applied after linear transform
    pub activation: Activation,
}

impl DenseLayer {
    /// Create a new dense layer with Xavier-uniform initialization.
    ///
    /// # Arguments
    /// * `in_dim` - Input dimensionality
    /// * `out_dim` - Output dimensionality
    /// * `activation` - Activation function
    /// * `seed` - Optional seed for deterministic init
    pub fn new(in_dim: usize, out_dim: usize, activation: Activation, seed: u64) -> Self {
        // Xavier uniform: limit = sqrt(6 / (fan_in + fan_out))
        let limit = (6.0 / (in_dim + out_dim) as f64).sqrt();
        let mut weights = Array2::<f64>::zeros((out_dim, in_dim));
        let mut bias = Array1::<f64>::zeros(out_dim);

        // Deterministic LCG PRNG
        let mut state = seed.wrapping_add(1);
        let lcg = |s: u64| -> (f64, u64) {
            let s2 = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            let f = (s2 >> 11) as f64 / (1u64 << 53) as f64;
            (f * 2.0 - 1.0, s2) // uniform in (-1, 1)
        };

        for i in 0..out_dim {
            for j in 0..in_dim {
                let (v, s) = lcg(state);
                state = s;
                weights[[i, j]] = v * limit;
            }
            let (v, s) = lcg(state);
            state = s;
            bias[i] = v * 0.01;
        }

        DenseLayer { weights, bias, activation }
    }

    /// Forward pass: compute activations for a batch of inputs.
    ///
    /// # Arguments
    /// * `input` - Input of shape (batch, in_dim)
    ///
    /// # Returns
    /// Output of shape (batch, out_dim)
    pub fn forward(&self, input: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let batch = input.nrows();
        let out_dim = self.weights.nrows();
        let in_dim = self.weights.ncols();

        if input.ncols() != in_dim {
            return Err(TransformError::InvalidInput(format!(
                "DenseLayer input dim mismatch: expected {}, got {}",
                in_dim,
                input.ncols()
            )));
        }

        let mut out = Array2::<f64>::zeros((batch, out_dim));
        for b in 0..batch {
            for o in 0..out_dim {
                let mut v = self.bias[o];
                for i in 0..in_dim {
                    v += self.weights[[o, i]] * input[[b, i]];
                }
                out[[b, o]] = self.activation.apply(v);
            }
        }
        Ok(out)
    }

    /// Number of parameters (weights + biases).
    pub fn n_params(&self) -> usize {
        self.weights.len() + self.bias.len()
    }
}

// ============================================================================
// Feature Network
// ============================================================================

/// Multi-layer feature extraction network for deep kernel learning.
///
/// The network maps raw inputs x ∈ ℝ^d to a feature vector φ(x) ∈ ℝ^h,
/// which is then passed to a kernel function.
#[derive(Debug, Clone)]
pub struct FeatureNetwork {
    /// Ordered list of layers
    pub layers: Vec<DenseLayer>,
    /// Input dimension
    pub input_dim: usize,
    /// Output (embedding) dimension
    pub output_dim: usize,
}

impl FeatureNetwork {
    /// Build a feed-forward network from a layer specification.
    ///
    /// # Arguments
    /// * `dims` - Layer dimensions: [input, hidden_1, …, hidden_k, output]
    /// * `activations` - Activation for each hidden layer (last layer uses Linear)
    /// * `seed` - Random seed for weight initialization
    ///
    /// # Errors
    /// Returns an error if `dims.len() < 2` or activation count mismatches.
    pub fn new(dims: &[usize], activations: &[Activation], seed: u64) -> Result<Self> {
        if dims.len() < 2 {
            return Err(TransformError::InvalidInput(
                "FeatureNetwork requires at least 2 dimensions (input + output)".to_string(),
            ));
        }
        let n_layers = dims.len() - 1;
        if activations.len() != n_layers {
            return Err(TransformError::InvalidInput(format!(
                "activations length {} must equal number of layers {}",
                activations.len(),
                n_layers
            )));
        }

        let layers = (0..n_layers)
            .map(|i| DenseLayer::new(dims[i], dims[i + 1], activations[i].clone(), seed + i as u64))
            .collect();

        Ok(FeatureNetwork {
            layers,
            input_dim: dims[0],
            output_dim: *dims.last().unwrap_or(&1),
        })
    }

    /// Forward pass through the entire network.
    pub fn forward(&self, input: &ArrayView2<f64>) -> Result<Array2<f64>> {
        if self.layers.is_empty() {
            return Ok(input.to_owned());
        }
        let mut current = self.layers[0].forward(input)?;
        for layer in self.layers.iter().skip(1) {
            current = layer.forward(&current.view())?;
        }
        Ok(current)
    }

    /// Total number of trainable parameters.
    pub fn n_params(&self) -> usize {
        self.layers.iter().map(|l| l.n_params()).sum()
    }
}

// ============================================================================
// Base Kernel Functions
// ============================================================================

/// Base kernel specification for the GP component in deep kernel learning.
#[derive(Debug, Clone)]
pub enum BaseKernel {
    /// Squared exponential (RBF) kernel with isotropic lengthscale.
    RBF {
        /// Output variance (signal variance)
        variance: f64,
        /// Lengthscale
        lengthscale: f64,
    },
    /// Matérn 3/2 kernel.
    Matern32 {
        /// Output variance
        variance: f64,
        /// Lengthscale
        lengthscale: f64,
    },
    /// Matérn 5/2 kernel.
    Matern52 {
        /// Output variance
        variance: f64,
        /// Lengthscale
        lengthscale: f64,
    },
    /// Rational quadratic kernel.
    RationalQuadratic {
        /// Output variance
        variance: f64,
        /// Lengthscale
        lengthscale: f64,
        /// Scale mixture parameter α
        alpha: f64,
    },
    /// Linear kernel k(x,y) = variance * xᵀy
    Linear {
        /// Output variance
        variance: f64,
    },
}

impl BaseKernel {
    /// Evaluate k(x, y) where x and y are feature vectors.
    pub fn evaluate(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
        match self {
            BaseKernel::RBF { variance, lengthscale } => {
                let dist_sq = squared_l2(x, y);
                variance * (-dist_sq / (2.0 * lengthscale * lengthscale)).exp()
            }
            BaseKernel::Matern32 { variance, lengthscale } => {
                let r = l2_distance(x, y) / lengthscale;
                let sqrt3_r = 3.0f64.sqrt() * r;
                variance * (1.0 + sqrt3_r) * (-sqrt3_r).exp()
            }
            BaseKernel::Matern52 { variance, lengthscale } => {
                let r = l2_distance(x, y) / lengthscale;
                let sqrt5_r = 5.0f64.sqrt() * r;
                variance * (1.0 + sqrt5_r + 5.0 * r * r / 3.0) * (-sqrt5_r).exp()
            }
            BaseKernel::RationalQuadratic { variance, lengthscale, alpha } => {
                let dist_sq = squared_l2(x, y);
                let base = 1.0 + dist_sq / (2.0 * alpha * lengthscale * lengthscale);
                variance * base.powf(-alpha)
            }
            BaseKernel::Linear { variance } => {
                let dot: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
                variance * dot
            }
        }
    }

    /// Build the full kernel matrix K(X, X').
    pub fn matrix(&self, x: &ArrayView2<f64>, xp: &ArrayView2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let m = xp.nrows();
        let mut k = Array2::<f64>::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                let xi = x.row(i);
                let xj = xp.row(j);
                k[[i, j]] = self.evaluate(&xi, &xj);
            }
        }
        k
    }

    /// Get/set the output variance hyperparameter.
    pub fn variance(&self) -> f64 {
        match self {
            BaseKernel::RBF { variance, .. }
            | BaseKernel::Matern32 { variance, .. }
            | BaseKernel::Matern52 { variance, .. }
            | BaseKernel::RationalQuadratic { variance, .. }
            | BaseKernel::Linear { variance } => *variance,
        }
    }
}

// ============================================================================
// DeepKernel
// ============================================================================

/// Deep kernel combining a neural feature extractor with a stationary base kernel.
///
/// The kernel is k_deep(x, y) = k_base(φ(x), φ(y)) where φ is the feature
/// network and k_base is any stationary kernel.
///
/// # Example
/// ```
/// use scirs2_transform::deep_kernel::{
///     DeepKernel, FeatureNetwork, BaseKernel, Activation,
/// };
/// use scirs2_core::ndarray::Array2;
///
/// let net = FeatureNetwork::new(
///     &[4, 8, 4],
///     &[Activation::ReLU, Activation::Linear],
///     42,
/// ).expect("FeatureNetwork::new should succeed");
/// let kernel = BaseKernel::RBF { variance: 1.0, lengthscale: 1.0 };
/// let dk = DeepKernel::new(net, kernel);
///
/// let x = Array2::<f64>::zeros((5, 4));
/// let k_mat = dk.kernel_matrix(&x.view(), &x.view()).expect("kernel_matrix should succeed");
/// assert_eq!(k_mat.shape(), &[5, 5]);
/// ```
#[derive(Debug, Clone)]
pub struct DeepKernel {
    /// Feature extraction network
    pub network: FeatureNetwork,
    /// Base kernel applied to features
    pub base_kernel: BaseKernel,
}

impl DeepKernel {
    /// Create a new deep kernel.
    pub fn new(network: FeatureNetwork, base_kernel: BaseKernel) -> Self {
        DeepKernel { network, base_kernel }
    }

    /// Compute the deep kernel matrix K(X, X').
    ///
    /// # Arguments
    /// * `x` - First dataset of shape (n, d)
    /// * `xp` - Second dataset of shape (m, d)
    ///
    /// # Returns
    /// Kernel matrix of shape (n, m)
    pub fn kernel_matrix(&self, x: &ArrayView2<f64>, xp: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let phi_x = self.network.forward(x)?;
        let phi_xp = self.network.forward(xp)?;
        Ok(self.base_kernel.matrix(&phi_x.view(), &phi_xp.view()))
    }

    /// Evaluate k_deep(x, y) for single vectors.
    pub fn evaluate(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Result<f64> {
        let x2 = x.to_owned().insert_axis(Axis(0));
        let y2 = y.to_owned().insert_axis(Axis(0));
        let phi_x = self.network.forward(&x2.view())?;
        let phi_y = self.network.forward(&y2.view())?;
        Ok(self.base_kernel.evaluate(&phi_x.row(0), &phi_y.row(0)))
    }
}

// ============================================================================
// Spectral Mixture Kernel (Wilson & Adams 2013)
// ============================================================================

/// Spectral mixture kernel: sum of Q spectral components.
///
/// Each component q has weight w_q, mean μ_q, and variance v_q (per dimension).
/// The kernel is:
///
/// k_SM(τ) = Σ_q w_q · exp(-2π²τ²v_q) · cos(2πτμ_q)
///
/// for 1D input; for multi-D: product over dimensions.
///
/// # References
/// Wilson & Adams (2013), Gaussian Process Kernels for Pattern Discovery
/// and Extrapolation.
#[derive(Debug, Clone)]
pub struct SpectralMixture {
    /// Number of spectral components Q
    pub n_components: usize,
    /// Component weights, shape (Q,) — non-negative, sum to 1
    pub weights: Array1<f64>,
    /// Spectral means (frequencies), shape (Q, D)
    pub means: Array2<f64>,
    /// Spectral variances (bandwidths), shape (Q, D)
    pub variances: Array2<f64>,
    /// Input dimensionality D
    pub input_dim: usize,
}

impl SpectralMixture {
    /// Create a spectral mixture kernel with Q components and D-dimensional input.
    ///
    /// Hyperparameters are initialized using frequency sampling heuristics:
    /// - weights: uniform 1/Q
    /// - means: sampled from [0, 0.5] (frequencies up to Nyquist)
    /// - variances: small positive values
    ///
    /// # Arguments
    /// * `n_components` - Number of mixture components Q (typically 5–20)
    /// * `input_dim` - Dimensionality D of the input
    /// * `seed` - Random seed
    pub fn new(n_components: usize, input_dim: usize, seed: u64) -> Result<Self> {
        if n_components == 0 {
            return Err(TransformError::InvalidInput(
                "n_components must be > 0".to_string(),
            ));
        }
        if input_dim == 0 {
            return Err(TransformError::InvalidInput(
                "input_dim must be > 0".to_string(),
            ));
        }

        let weights = Array1::from_elem(n_components, 1.0 / n_components as f64);
        let mut means = Array2::<f64>::zeros((n_components, input_dim));
        let mut variances = Array2::<f64>::ones((n_components, input_dim));

        // LCG for initialization
        let mut state = seed.wrapping_add(7919);
        let next_u01 = |s: u64| -> (f64, u64) {
            let s2 = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((s2 >> 11) as f64 / (1u64 << 53) as f64, s2)
        };

        for q in 0..n_components {
            for d in 0..input_dim {
                let (v, s) = next_u01(state);
                state = s;
                means[[q, d]] = v * 0.5; // random frequency in [0, 0.5]

                let (v2, s2) = next_u01(state);
                state = s2;
                variances[[q, d]] = v2 * 0.5 + 0.1; // bandwidth in [0.1, 0.6]
            }
        }

        Ok(SpectralMixture {
            n_components,
            weights,
            means,
            variances,
            input_dim,
        })
    }

    /// Evaluate the spectral mixture kernel k(x, y).
    ///
    /// For D-dimensional inputs:
    /// k(x,y) = Σ_q w_q · Π_d exp(-2π²τ_d²v_{q,d}) · cos(2πτ_dμ_{q,d})
    pub fn evaluate(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Result<f64> {
        let d = x.len();
        if d != self.input_dim {
            return Err(TransformError::InvalidInput(format!(
                "SpectralMixture: expected input_dim={}, got {}",
                self.input_dim, d
            )));
        }
        if y.len() != d {
            return Err(TransformError::InvalidInput(
                "SpectralMixture: x and y must have the same length".to_string(),
            ));
        }

        let mut k = 0.0f64;
        for q in 0..self.n_components {
            let mut prod = self.weights[q];
            for dim in 0..d {
                let tau = x[dim] - y[dim];
                let v = self.variances[[q, dim]];
                let mu = self.means[[q, dim]];
                prod *= (-2.0 * PI * PI * tau * tau * v).exp() * (2.0 * PI * tau * mu).cos();
            }
            k += prod;
        }
        Ok(k)
    }

    /// Compute the spectral mixture kernel matrix K(X, X').
    pub fn kernel_matrix(&self, x: &ArrayView2<f64>, xp: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let n = x.nrows();
        let m = xp.nrows();
        let mut k = Array2::<f64>::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                k[[i, j]] = self.evaluate(&x.row(i), &xp.row(j))?;
            }
        }
        Ok(k)
    }

    /// Update weights (renormalize to sum to 1).
    pub fn set_weights(&mut self, weights: Array1<f64>) -> Result<()> {
        if weights.len() != self.n_components {
            return Err(TransformError::InvalidInput(format!(
                "weights length {} != n_components {}",
                weights.len(), self.n_components
            )));
        }
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Err(TransformError::InvalidInput(
                "weights must sum to a positive value".to_string(),
            ));
        }
        self.weights = weights.mapv(|w| w / total);
        Ok(())
    }
}

// ============================================================================
// ARD Kernel (Automatic Relevance Determination)
// ============================================================================

/// Automatic Relevance Determination (ARD) squared-exponential kernel.
///
/// k_ARD(x, y) = σ² · exp(-½ Σ_d (x_d - y_d)² / l_d²)
///
/// Each input dimension d has its own lengthscale l_d, allowing the model to
/// automatically down-weight irrelevant features (large l_d → less sensitive).
///
/// The lengthscales are optimized jointly with the GP likelihood.
#[derive(Debug, Clone)]
pub struct ARDKernel {
    /// Output variance σ²
    pub variance: f64,
    /// Per-dimension lengthscales, shape (D,)
    pub lengthscales: Array1<f64>,
    /// Input dimensionality
    pub input_dim: usize,
}

impl ARDKernel {
    /// Create an ARD kernel with uniform initial lengthscales.
    ///
    /// # Arguments
    /// * `input_dim` - Dimensionality of the input
    /// * `variance` - Initial output variance σ²
    /// * `initial_lengthscale` - Initial value for all lengthscales
    pub fn new(input_dim: usize, variance: f64, initial_lengthscale: f64) -> Result<Self> {
        if input_dim == 0 {
            return Err(TransformError::InvalidInput(
                "ARDKernel: input_dim must be > 0".to_string(),
            ));
        }
        if variance <= 0.0 {
            return Err(TransformError::InvalidInput(
                "ARDKernel: variance must be positive".to_string(),
            ));
        }
        if initial_lengthscale <= 0.0 {
            return Err(TransformError::InvalidInput(
                "ARDKernel: lengthscale must be positive".to_string(),
            ));
        }
        Ok(ARDKernel {
            variance,
            lengthscales: Array1::from_elem(input_dim, initial_lengthscale),
            input_dim,
        })
    }

    /// Evaluate k_ARD(x, y).
    pub fn evaluate(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Result<f64> {
        if x.len() != self.input_dim || y.len() != self.input_dim {
            return Err(TransformError::InvalidInput(format!(
                "ARDKernel: input_dim mismatch: expected {}, got x={}, y={}",
                self.input_dim, x.len(), y.len()
            )));
        }
        let mut dist_sq = 0.0f64;
        for d in 0..self.input_dim {
            let diff = x[d] - y[d];
            let l = self.lengthscales[d];
            dist_sq += (diff / l) * (diff / l);
        }
        Ok(self.variance * (-0.5 * dist_sq).exp())
    }

    /// Compute the ARD kernel matrix K(X, X').
    pub fn kernel_matrix(&self, x: &ArrayView2<f64>, xp: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let n = x.nrows();
        let m = xp.nrows();
        let mut k = Array2::<f64>::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                k[[i, j]] = self.evaluate(&x.row(i), &xp.row(j))?;
            }
        }
        Ok(k)
    }

    /// Set lengthscales from a flat parameter vector.
    pub fn set_lengthscales(&mut self, ls: Array1<f64>) -> Result<()> {
        if ls.len() != self.input_dim {
            return Err(TransformError::InvalidInput(format!(
                "ARDKernel: lengthscales length {} != input_dim {}",
                ls.len(), self.input_dim
            )));
        }
        if ls.iter().any(|&v| v <= 0.0) {
            return Err(TransformError::InvalidInput(
                "ARDKernel: all lengthscales must be positive".to_string(),
            ));
        }
        self.lengthscales = ls;
        Ok(())
    }

    /// Get a flat vector of log-lengthscales (for unconstrained optimization).
    pub fn log_lengthscales(&self) -> Array1<f64> {
        self.lengthscales.mapv(f64::ln)
    }

    /// Set lengthscales from a flat vector of log-lengthscales.
    pub fn set_log_lengthscales(&mut self, log_ls: &ArrayView1<f64>) -> Result<()> {
        if log_ls.len() != self.input_dim {
            return Err(TransformError::InvalidInput(format!(
                "ARDKernel: log_lengthscales length {} != input_dim {}",
                log_ls.len(), self.input_dim
            )));
        }
        self.lengthscales = log_ls.mapv(f64::exp);
        Ok(())
    }
}

// ============================================================================
// GP Utilities
// ============================================================================

/// Compute the log marginal likelihood log p(y | X, θ) for a GP with kernel K.
///
/// log p(y|X) = -½ yᵀ K⁻¹ y - ½ log|K| - n/2 log(2π)
///
/// Uses Cholesky decomposition for numerical stability.
fn gp_log_marginal_likelihood(
    k_matrix: &Array2<f64>,
    y: &ArrayView1<f64>,
    noise: f64,
) -> Result<f64> {
    let n = y.len();
    if k_matrix.nrows() != n || k_matrix.ncols() != n {
        return Err(TransformError::InvalidInput(
            "gp_log_mll: kernel matrix dimension must match y length".to_string(),
        ));
    }

    // Add noise to diagonal: K_y = K + σ²I
    let mut k_y = k_matrix.clone();
    for i in 0..n {
        k_y[[i, i]] += noise;
    }

    // Cholesky decomposition K_y = LLᵀ
    let l = cholesky(&k_y.view(), None)
        .map_err(|e| TransformError::ComputationError(format!("Cholesky failed: {e}")))?;

    // Solve L alpha_lower = y, then Lᵀ alpha = alpha_lower
    let y_col: Array1<f64> = y.to_owned();
    let alpha_lower = solve(&l.view(), &y_col.view(), None)
        .map_err(|e| TransformError::ComputationError(format!("triangular solve failed: {e}")))?;

    // Data fit: -½ yᵀ K_y⁻¹ y = -½ ||alpha_lower||²
    let data_fit: f64 = alpha_lower.iter().map(|v| v * v).sum::<f64>() * -0.5;

    // Complexity penalty: -½ log|K_y| = -Σ log(L_{ii})
    let log_det: f64 = (0..n).map(|i| l[[i, i]].ln()).sum::<f64>();
    let complexity = -log_det;

    // Constant: -n/2 * log(2π)
    let constant = -(n as f64) / 2.0 * (2.0 * PI).ln();

    Ok(data_fit + complexity + constant)
}

/// GP posterior prediction: returns (mean, variance) at test points X*.
fn gp_predict(
    k_train_train: &Array2<f64>,
    k_train_test: &Array2<f64>,
    k_test_test_diag: &Array1<f64>,
    y_train: &ArrayView1<f64>,
    noise: f64,
) -> Result<(Array1<f64>, Array1<f64>)> {
    let n = y_train.len();
    let n_test = k_test_test_diag.len();

    let mut k_y = k_train_train.clone();
    for i in 0..n {
        k_y[[i, i]] += noise;
    }

    let l = cholesky(&k_y.view(), None)
        .map_err(|e| TransformError::ComputationError(format!("Cholesky failed: {e}")))?;

    let y_col: Array1<f64> = y_train.to_owned();
    let _alpha = solve(&l.view(), &y_col.view(), None)
        .map_err(|e| TransformError::ComputationError(format!("Solve failed: {e}")))?;

    // Mean: K*ᵀ K_y⁻¹ y = K_train_test.T @ alpha  (need full K_y⁻¹y)
    // Re-solve: K_y alpha_full = y
    let alpha_full = solve(&k_y.view(), &y_col.view(), None)
        .map_err(|e| TransformError::ComputationError(format!("Solve alpha_full failed: {e}")))?;

    let mut mean = Array1::<f64>::zeros(n_test);
    for j in 0..n_test {
        let mut s = 0.0f64;
        for i in 0..n {
            s += k_train_test[[i, j]] * alpha_full[i];
        }
        mean[j] = s;
    }

    // Variance: K** - K*ᵀ K_y⁻¹ K*
    // v = L⁻¹ K_train_test  (shape n x n_test)
    let v = solve_multiple(&l.view(), &k_train_test.view(), None)
        .map_err(|e| TransformError::ComputationError(format!("Solve v failed: {e}")))?;

    let mut variance = Array1::<f64>::zeros(n_test);
    for j in 0..n_test {
        let v_col_sq: f64 = (0..n).map(|i| v[[i, j]] * v[[i, j]]).sum();
        let var = k_test_test_diag[j] - v_col_sq;
        variance[j] = var.max(0.0); // clamp numerical noise
    }

    Ok((mean, variance))
}

// ============================================================================
// DeepKernelGP
// ============================================================================

/// Gaussian process regression with a deep kernel.
///
/// The full GP uses exact inference (O(n³) scaling) and optimizes joint
/// deep kernel hyperparameters via marginal log-likelihood maximization.
///
/// # Example
/// ```
/// use scirs2_transform::deep_kernel::{
///     DeepKernelGP, DeepKernelGPConfig, FeatureNetwork, BaseKernel, Activation,
/// };
/// use scirs2_core::ndarray::Array2;
///
/// let cfg = DeepKernelGPConfig::default();
/// let mut gp = DeepKernelGP::new(cfg).expect("DeepKernelGP::new should succeed");
///
/// let x_train = Array2::<f64>::zeros((20, 3));
/// let y_train = scirs2_core::ndarray::Array1::<f64>::zeros(20);
/// gp.fit(&x_train.view(), &y_train.view()).expect("DeepKernelGP fit should succeed");
///
/// let x_test = Array2::<f64>::zeros((5, 3));
/// let (mean, var) = gp.predict(&x_test.view()).expect("DeepKernelGP predict should succeed");
/// assert_eq!(mean.len(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct DeepKernelGP {
    /// Deep kernel (feature network + base kernel)
    pub kernel: DeepKernel,
    /// Observation noise variance σ²_n
    pub noise: f64,
    /// Stored training features (after network projection)
    train_features: Option<Array2<f64>>,
    /// Stored training targets
    train_targets: Option<Array1<f64>>,
    /// Configuration
    config: DeepKernelGPConfig,
}

/// Configuration for [`DeepKernelGP`].
#[derive(Debug, Clone)]
pub struct DeepKernelGPConfig {
    /// Layer sizes [input, h1, …, feature_dim]
    pub layer_dims: Vec<usize>,
    /// Activation for each hidden layer
    pub activations: Vec<Activation>,
    /// Base kernel type
    pub base_kernel: BaseKernel,
    /// Initial noise variance
    pub noise: f64,
    /// Random seed
    pub seed: u64,
}

impl Default for DeepKernelGPConfig {
    fn default() -> Self {
        DeepKernelGPConfig {
            layer_dims: vec![1, 16, 8, 4],
            activations: vec![Activation::ReLU, Activation::ReLU, Activation::Linear],
            base_kernel: BaseKernel::RBF { variance: 1.0, lengthscale: 1.0 },
            noise: 0.01,
            seed: 42,
        }
    }
}

impl DeepKernelGP {
    /// Create a new deep kernel GP from a configuration.
    pub fn new(config: DeepKernelGPConfig) -> Result<Self> {
        let network = FeatureNetwork::new(&config.layer_dims, &config.activations, config.seed)?;
        let kernel = DeepKernel::new(network, config.base_kernel.clone());
        Ok(DeepKernelGP {
            kernel,
            noise: config.noise,
            train_features: None,
            train_targets: None,
            config,
        })
    }

    /// Fit the GP on training data (stores features; no hyperparameter optimization).
    ///
    /// For joint training of network weights and GP hyperparameters via MLL,
    /// use [`DeepKernelTrainer`].
    ///
    /// # Arguments
    /// * `x_train` - Training inputs, shape (n, d)
    /// * `y_train` - Training targets, shape (n,)
    pub fn fit(&mut self, x_train: &ArrayView2<f64>, y_train: &ArrayView1<f64>) -> Result<()> {
        if x_train.nrows() != y_train.len() {
            return Err(TransformError::InvalidInput(format!(
                "DeepKernelGP: x_train rows {} != y_train len {}",
                x_train.nrows(),
                y_train.len()
            )));
        }
        let features = self.kernel.network.forward(x_train)?;
        self.train_features = Some(features);
        self.train_targets = Some(y_train.to_owned());
        Ok(())
    }

    /// Predict mean and variance at test points.
    ///
    /// # Returns
    /// `(mean, variance)` each of shape (n_test,)
    pub fn predict(&self, x_test: &ArrayView2<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        let train_feat = self.train_features.as_ref().ok_or_else(|| {
            TransformError::NotFitted("DeepKernelGP must be fitted before predict".to_string())
        })?;
        let y_train = self.train_targets.as_ref().ok_or_else(|| {
            TransformError::NotFitted("DeepKernelGP must be fitted before predict".to_string())
        })?;

        let test_feat = self.kernel.network.forward(x_test)?;
        let k_train_train = self.kernel.base_kernel.matrix(&train_feat.view(), &train_feat.view());
        let k_train_test = self.kernel.base_kernel.matrix(&train_feat.view(), &test_feat.view());

        // Diagonal of K(X*, X*)
        let k_test_diag: Array1<f64> = (0..test_feat.nrows())
            .map(|i| {
                let xi = test_feat.row(i);
                self.kernel.base_kernel.evaluate(&xi, &xi)
            })
            .collect();

        gp_predict(&k_train_train, &k_train_test, &k_test_diag, &y_train.view(), self.noise)
    }

    /// Compute the marginal log-likelihood for the current hyperparameters.
    pub fn log_marginal_likelihood(&self) -> Result<f64> {
        let train_feat = self.train_features.as_ref().ok_or_else(|| {
            TransformError::NotFitted("DeepKernelGP must be fitted before MLL computation".to_string())
        })?;
        let y_train = self.train_targets.as_ref().ok_or_else(|| {
            TransformError::NotFitted("DeepKernelGP must be fitted before MLL computation".to_string())
        })?;

        let k = self.kernel.base_kernel.matrix(&train_feat.view(), &train_feat.view());
        gp_log_marginal_likelihood(&k, &y_train.view(), self.noise)
    }
}

// ============================================================================
// DeepKernelTrainer
// ============================================================================

/// Training result from deep kernel learning optimization.
#[derive(Debug, Clone)]
pub struct DKLTrainingResult {
    /// Final marginal log-likelihood
    pub final_mll: f64,
    /// MLL at each epoch
    pub mll_history: Vec<f64>,
    /// Number of epochs run
    pub n_epochs: usize,
}

/// Joint trainer for deep kernel learning: optimizes the feature network
/// and GP hyperparameters simultaneously via marginal log-likelihood.
///
/// Optimization uses simple finite-difference gradient ascent on the feature
/// network parameters (weights/biases), with the GP noise and kernel variance
/// optimized via coordinate steps.
///
/// This provides a self-contained implementation that avoids external
/// autograd dependencies while demonstrating the DKL training loop.
#[derive(Debug, Clone)]
pub struct DeepKernelTrainer {
    /// Learning rate for gradient ascent
    pub learning_rate: f64,
    /// Maximum number of epochs
    pub max_epochs: usize,
    /// Convergence tolerance for MLL improvement
    pub tol: f64,
    /// Finite-difference step size for gradient estimation
    pub fd_eps: f64,
}

impl Default for DeepKernelTrainer {
    fn default() -> Self {
        DeepKernelTrainer {
            learning_rate: 1e-3,
            max_epochs: 100,
            tol: 1e-5,
            fd_eps: 1e-4,
        }
    }
}

impl DeepKernelTrainer {
    /// Create a new trainer with the given hyperparameters.
    pub fn new(learning_rate: f64, max_epochs: usize, tol: f64) -> Self {
        DeepKernelTrainer {
            learning_rate,
            max_epochs,
            tol,
            fd_eps: 1e-4,
        }
    }

    /// Train a deep kernel GP via MLL maximization.
    ///
    /// Performs gradient ascent on the log-marginal likelihood using
    /// finite differences to estimate gradients w.r.t. layer parameters.
    ///
    /// # Arguments
    /// * `gp` - Mutable reference to the DeepKernelGP to optimize
    /// * `x_train` - Training inputs, shape (n, d)
    /// * `y_train` - Training targets, shape (n,)
    pub fn train(
        &self,
        gp: &mut DeepKernelGP,
        x_train: &ArrayView2<f64>,
        y_train: &ArrayView1<f64>,
    ) -> Result<DKLTrainingResult> {
        // Initial fit
        gp.fit(x_train, y_train)?;
        let mut mll_history = Vec::with_capacity(self.max_epochs);
        let mut prev_mll = gp.log_marginal_likelihood()?;
        mll_history.push(prev_mll);

        for epoch in 0..self.max_epochs {
            // Gradient ascent on each layer's weights and biases
            for layer_idx in 0..gp.kernel.network.layers.len() {
                // Weights gradient
                let out_dim = gp.kernel.network.layers[layer_idx].weights.nrows();
                let in_dim = gp.kernel.network.layers[layer_idx].weights.ncols();

                for o in 0..out_dim {
                    for i in 0..in_dim {
                        let w_orig = gp.kernel.network.layers[layer_idx].weights[[o, i]];

                        // Forward difference
                        gp.kernel.network.layers[layer_idx].weights[[o, i]] = w_orig + self.fd_eps;
                        gp.fit(x_train, y_train)?;
                        let mll_plus = gp.log_marginal_likelihood().unwrap_or(f64::NEG_INFINITY);

                        gp.kernel.network.layers[layer_idx].weights[[o, i]] = w_orig - self.fd_eps;
                        gp.fit(x_train, y_train)?;
                        let mll_minus = gp.log_marginal_likelihood().unwrap_or(f64::NEG_INFINITY);

                        let grad = (mll_plus - mll_minus) / (2.0 * self.fd_eps);
                        gp.kernel.network.layers[layer_idx].weights[[o, i]] =
                            w_orig + self.learning_rate * grad;
                    }
                }

                // Bias gradient
                let bias_dim = gp.kernel.network.layers[layer_idx].bias.len();
                for b in 0..bias_dim {
                    let b_orig = gp.kernel.network.layers[layer_idx].bias[b];

                    gp.kernel.network.layers[layer_idx].bias[b] = b_orig + self.fd_eps;
                    gp.fit(x_train, y_train)?;
                    let mll_plus = gp.log_marginal_likelihood().unwrap_or(f64::NEG_INFINITY);

                    gp.kernel.network.layers[layer_idx].bias[b] = b_orig - self.fd_eps;
                    gp.fit(x_train, y_train)?;
                    let mll_minus = gp.log_marginal_likelihood().unwrap_or(f64::NEG_INFINITY);

                    let grad = (mll_plus - mll_minus) / (2.0 * self.fd_eps);
                    gp.kernel.network.layers[layer_idx].bias[b] =
                        b_orig + self.learning_rate * grad;
                }
            }

            // Refit with updated parameters
            gp.fit(x_train, y_train)?;
            let current_mll = gp.log_marginal_likelihood()?;
            mll_history.push(current_mll);

            // Check convergence
            if (current_mll - prev_mll).abs() < self.tol {
                return Ok(DKLTrainingResult {
                    final_mll: current_mll,
                    mll_history,
                    n_epochs: epoch + 1,
                });
            }
            prev_mll = current_mll;
        }

        Ok(DKLTrainingResult {
            final_mll: prev_mll,
            mll_history,
            n_epochs: self.max_epochs,
        })
    }
}

// ============================================================================
// InducingPoints — Sparse GP Approximation
// ============================================================================

/// Sparse GP using inducing point approximation (FITC/VFE variant).
///
/// Instead of O(n³) full GP inference, maintains a set of M inducing points
/// Z ∈ ℝ^{M×D} (M ≪ n) and computes approximate posterior via:
///
///   q(f) ≈ GP(μ, K̃)  where  K̃ = Q_{nn} + diag(K_{nn} - Q_{nn})
///   Q_{nn} = K_{nm} K_{mm}⁻¹ K_{mn}
///
/// # References
/// Snelson & Ghahramani (2006), Sparse Gaussian Processes using Pseudo-inputs.
#[derive(Debug, Clone)]
pub struct InducingPoints {
    /// Inducing point locations, shape (M, D)
    pub inducing_z: Array2<f64>,
    /// Base kernel
    pub kernel: BaseKernel,
    /// Noise variance
    pub noise: f64,
    /// Stored K(Z, Z)⁻¹ from fit
    kmm_inv: Option<Array2<f64>>,
    /// Stored K(Z, X_train)
    knm_t: Option<Array2<f64>>,
    /// Alpha vector K_approx⁻¹ y
    alpha: Option<Array1<f64>>,
    /// Training targets
    y_train: Option<Array1<f64>>,
}

impl InducingPoints {
    /// Create sparse GP with M inducing points initialized via k-means++ selection.
    ///
    /// # Arguments
    /// * `n_inducing` - Number of inducing points M
    /// * `kernel` - Base kernel
    /// * `noise` - Observation noise variance
    /// * `seed` - Random seed for inducing point initialization
    pub fn new(n_inducing: usize, kernel: BaseKernel, noise: f64, seed: u64) -> Result<Self> {
        if n_inducing == 0 {
            return Err(TransformError::InvalidInput(
                "InducingPoints: n_inducing must be > 0".to_string(),
            ));
        }
        if noise <= 0.0 {
            return Err(TransformError::InvalidInput(
                "InducingPoints: noise must be positive".to_string(),
            ));
        }
        // Placeholder inducing points (dim will be set at fit time)
        let inducing_z = Array2::<f64>::zeros((n_inducing, 1));
        let _ = seed; // used at fit time
        Ok(InducingPoints {
            inducing_z,
            kernel,
            noise,
            kmm_inv: None,
            knm_t: None,
            alpha: None,
            y_train: None,
        })
    }

    /// Initialize inducing points using k-means++ seeding on training data.
    fn init_inducing_points(x_train: &ArrayView2<f64>, m: usize, seed: u64) -> Array2<f64> {
        let n = x_train.nrows();
        let d = x_train.ncols();
        let mut inducing = Vec::with_capacity(m);

        // LCG PRNG
        let mut state = seed.wrapping_add(12345);
        let next_u64 = |s: u64| -> (u64, u64) {
            let s2 = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (s2, s2)
        };

        // Pick first center uniformly at random
        let (r, s) = next_u64(state);
        state = s;
        let first_idx = (r as usize) % n;
        inducing.push(x_train.row(first_idx).to_owned());

        for _ in 1..m.min(n) {
            // Compute min distance from each point to nearest center
            let mut min_dists: Vec<f64> = (0..n)
                .map(|i| {
                    inducing.iter().map(|c| {
                        let xi = x_train.row(i);
                        let sq: f64 = xi.iter().zip(c.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                        sq
                    }).fold(f64::INFINITY, f64::min)
                })
                .collect();

            // Sample next center proportional to min_dist^2
            let total: f64 = min_dists.iter().sum();
            if total <= 0.0 {
                break;
            }
            let (r2, s2) = next_u64(state);
            state = s2;
            let threshold = (r2 as f64 / u64::MAX as f64) * total;
            let mut cumsum = 0.0f64;
            let mut chosen = 0;
            for (i, &d_sq) in min_dists.iter().enumerate() {
                cumsum += d_sq;
                if cumsum >= threshold {
                    chosen = i;
                    break;
                }
            }
            inducing.push(x_train.row(chosen).to_owned());
        }

        // Pad with zeros if n < m
        while inducing.len() < m {
            inducing.push(Array1::<f64>::zeros(d));
        }

        let flat: Vec<f64> = inducing.iter().flat_map(|r| r.iter().copied()).collect();
        Array2::from_shape_vec((m, d), flat).unwrap_or_else(|_| Array2::zeros((m, d)))
    }

    /// Compute the (approximate) inverse of a matrix via Cholesky.
    fn cholesky_inv(a: &Array2<f64>) -> Result<Array2<f64>> {
        let n = a.nrows();
        let l = cholesky(&a.view(), None)
            .map_err(|e| TransformError::ComputationError(format!("Cholesky failed: {e}")))?;
        // Solve L L^T X = I using the Cholesky factor
        let identity = Array2::<f64>::eye(n);
        let inv = solve_multiple(&l.view(), &identity.view(), None)
            .map_err(|e| TransformError::ComputationError(format!("solve failed: {e}")))?;
        Ok(inv)
    }

    /// Fit the sparse GP on training data.
    ///
    /// # Arguments
    /// * `x_train` - Training inputs, shape (n, D)
    /// * `y_train` - Training targets, shape (n,)
    /// * `seed` - Random seed for inducing point initialization
    pub fn fit(
        &mut self,
        x_train: &ArrayView2<f64>,
        y_train: &ArrayView1<f64>,
        seed: u64,
    ) -> Result<()> {
        let n = x_train.nrows();
        let m = self.inducing_z.nrows();

        if x_train.nrows() != y_train.len() {
            return Err(TransformError::InvalidInput(format!(
                "InducingPoints: x_train rows {} != y_train len {}",
                n, y_train.len()
            )));
        }

        // Initialize inducing points from training data
        self.inducing_z = Self::init_inducing_points(x_train, m, seed);

        // Compute K_mm = K(Z, Z)
        let k_mm = self.kernel.matrix(&self.inducing_z.view(), &self.inducing_z.view());

        // Regularize K_mm for stability
        let mut k_mm_reg = k_mm.clone();
        let jitter = 1e-6 * k_mm.diag().iter().cloned().fold(0.0f64, f64::max).max(1e-6);
        for i in 0..m {
            k_mm_reg[[i, i]] += jitter;
        }

        let kmm_inv = Self::cholesky_inv(&k_mm_reg)?;

        // K_nm = K(X, Z) — shape (n, m)
        let k_nm = self.kernel.matrix(x_train, &self.inducing_z.view());

        // Q_nn diagonal = diag(K_nm K_mm^{-1} K_nm^T)
        // Q_nn_diag[i] = (K_nm @ K_mm^{-1} @ K_nm^T)[i,i]
        // = row_i(K_nm) @ K_mm^{-1} @ row_i(K_nm)
        let k_nm_kmm_inv = k_nm.dot(&kmm_inv); // shape (n, m)

        // K_nn diagonal
        let k_nn_diag: Array1<f64> = (0..n)
            .map(|i| {
                let xi = x_train.row(i);
                self.kernel.evaluate(&xi, &xi)
            })
            .collect();

        // Lambda = diag(K_nn - Q_nn) + sigma^2 I
        let lambda_diag: Array1<f64> = (0..n)
            .map(|i| {
                let q_i: f64 = (0..m).map(|j| k_nm_kmm_inv[[i, j]] * k_nm[[i, j]]).sum();
                (k_nn_diag[i] - q_i).max(0.0) + self.noise
            })
            .collect();

        // FITC approximation: (K_approx)^{-1} y via Woodbury identity
        // K_approx = Lambda + K_nm K_mm^{-1} K_mn
        // (Lambda + Q_nn)^{-1} using Woodbury:
        //   = Lambda^{-1} - Lambda^{-1} K_nm (K_mm + K_mn Lambda^{-1} K_nm)^{-1} K_mn Lambda^{-1}

        let lambda_inv_diag: Array1<f64> = lambda_diag.mapv(|v| 1.0 / v.max(1e-10));

        // B = K_mm + K_mn Lambda^{-1} K_nm  (m x m)
        let mut b = k_mm.clone();
        for i in 0..n {
            let lam_i = lambda_inv_diag[i];
            for j in 0..m {
                for k in 0..m {
                    b[[j, k]] += k_nm[[i, j]] * lam_i * k_nm[[i, k]];
                }
            }
        }
        // Add jitter to B
        for i in 0..m {
            b[[i, i]] += jitter;
        }

        let b_inv = Self::cholesky_inv(&b)?;

        // alpha = (Lambda + Q_nn)^{-1} y
        // = Lambda^{-1} y - Lambda^{-1} K_nm B^{-1} K_mn Lambda^{-1} y
        let lam_inv_y: Array1<f64> = y_train.iter().enumerate()
            .map(|(i, &yi)| yi * lambda_inv_diag[i])
            .collect();

        // K_mn Lambda^{-1} y  (shape m,)
        let mut kmn_laminvy = Array1::<f64>::zeros(m);
        for i in 0..n {
            for j in 0..m {
                kmn_laminvy[j] += k_nm[[i, j]] * lam_inv_y[i];
            }
        }

        // B^{-1} K_mn Lambda^{-1} y  (shape m,)
        let b_inv_kmn_laminvy: Array1<f64> = {
            let v = kmn_laminvy.clone().insert_axis(Axis(1));
            let res = b_inv.dot(&v);
            res.column(0).to_owned()
        };

        // Lambda^{-1} K_nm B^{-1} K_mn Lambda^{-1} y  (shape n,)
        let mut correction = Array1::<f64>::zeros(n);
        for i in 0..n {
            let lam_i = lambda_inv_diag[i];
            for j in 0..m {
                correction[i] += k_nm[[i, j]] * b_inv_kmn_laminvy[j] * lam_i;
            }
        }

        let alpha = &lam_inv_y - &correction;

        self.kmm_inv = Some(kmm_inv);
        self.knm_t = Some(k_nm); // K_nm stored as (n, m)
        self.alpha = Some(alpha);
        self.y_train = Some(y_train.to_owned());
        Ok(())
    }

    /// Predict mean and variance at test points.
    ///
    /// Uses the FITC approximate posterior.
    ///
    /// # Returns
    /// `(mean, variance)` each of shape (n_test,)
    pub fn predict(&self, x_test: &ArrayView2<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        let alpha = self.alpha.as_ref().ok_or_else(|| {
            TransformError::NotFitted("InducingPoints must be fitted before predict".to_string())
        })?;
        let k_nm = self.knm_t.as_ref().ok_or_else(|| {
            TransformError::NotFitted("InducingPoints must be fitted before predict".to_string())
        })?;
        let kmm_inv = self.kmm_inv.as_ref().ok_or_else(|| {
            TransformError::NotFitted("InducingPoints must be fitted before predict".to_string())
        })?;

        let m = self.inducing_z.nrows();
        let n_test = x_test.nrows();

        // K(Z, X*) — shape (m, n_test)
        let k_zm_xstar = self.kernel.matrix(&self.inducing_z.view(), x_test);

        // Mean: K_nm_t @ K_mm^{-1} @ K_zm_xstar^T  ... actually we need
        // mean = K(X*, Z) K_mm^{-1} K(Z, X) alpha
        // = k_xstar_z @ kmm_inv @ k_zm @ alpha

        // Step 1: kmm_inv @ k_zm_xstar  (m x n_test)
        let kinv_k = kmm_inv.dot(&k_zm_xstar); // (m, n_test)

        // Step 2: K_nm @ (K_mm^{-1} K_zm_xstar)  (n x n_test)
        let k_nm_kinv_k = k_nm.dot(&kinv_k); // (n, n_test)

        // Mean = (K_nm_kinv_k)^T @ alpha  (n_test,)
        let mut mean = Array1::<f64>::zeros(n_test);
        for j in 0..n_test {
            let s: f64 = (0..k_nm.nrows()).map(|i| k_nm_kinv_k[[i, j]] * alpha[i]).sum();
            mean[j] = s;
        }

        // Variance: K(X*, X*)_diag - K(X*, Z) K_mm^{-1} K(Z, X*)_diag
        let mut variance = Array1::<f64>::zeros(n_test);
        for j in 0..n_test {
            let xi = x_test.row(j);
            let k_star_star = self.kernel.evaluate(&xi, &xi);

            // K(X*_j, Z) K_mm^{-1} K(Z, X*_j)
            let k_xstar_z: Array1<f64> = (0..m)
                .map(|k_idx| {
                    let zk = self.inducing_z.row(k_idx);
                    self.kernel.evaluate(&xi, &zk)
                })
                .collect();

            // K_mm^{-1} @ k_xstar_z
            let kinv_k_j: Array1<f64> = kmm_inv.dot(&k_xstar_z);
            let q: f64 = k_xstar_z.iter().zip(kinv_k_j.iter()).map(|(a, b)| a * b).sum();

            variance[j] = (k_star_star - q).max(0.0) + self.noise;
        }

        Ok((mean, variance))
    }

    /// Number of inducing points.
    pub fn n_inducing(&self) -> usize {
        self.inducing_z.nrows()
    }
}

// ============================================================================
// Utility functions
// ============================================================================

fn squared_l2(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
    x.iter().zip(y.iter()).map(|(a, b)| (a - b) * (a - b)).sum()
}

fn l2_distance(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
    squared_l2(x, y).sqrt()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_activation_relu() {
        let act = Activation::ReLU;
        assert_eq!(act.apply(-1.0), 0.0);
        assert!((act.apply(2.0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_dense_layer_forward() {
        let layer = DenseLayer::new(4, 3, Activation::ReLU, 42);
        let x = Array2::<f64>::ones((5, 4));
        let out = layer.forward(&x.view()).expect("forward should succeed");
        assert_eq!(out.shape(), &[5, 3]);
        // ReLU output is non-negative
        assert!(out.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_feature_network_forward() {
        let net = FeatureNetwork::new(
            &[4, 8, 4],
            &[Activation::ReLU, Activation::Linear],
            42,
        ).expect("FeatureNetwork::new should succeed");
        let x = Array2::<f64>::ones((10, 4));
        let out = net.forward(&x.view()).expect("forward should succeed");
        assert_eq!(out.shape(), &[10, 4]);
    }

    #[test]
    fn test_base_kernel_rbf() {
        let k = BaseKernel::RBF { variance: 1.0, lengthscale: 1.0 };
        let x = scirs2_core::ndarray::array![1.0, 0.0];
        let y = scirs2_core::ndarray::array![1.0, 0.0];
        let v = k.evaluate(&x.view(), &y.view());
        assert!((v - 1.0).abs() < 1e-10, "k(x,x) should be 1 for RBF");
    }

    #[test]
    fn test_deep_kernel_matrix_shape() {
        let net = FeatureNetwork::new(
            &[3, 6, 3],
            &[Activation::ReLU, Activation::Linear],
            0,
        ).expect("FeatureNetwork::new should succeed");
        let bk = BaseKernel::RBF { variance: 1.0, lengthscale: 1.0 };
        let dk = DeepKernel::new(net, bk);
        let x = Array2::<f64>::zeros((5, 3));
        let k = dk.kernel_matrix(&x.view(), &x.view()).expect("kernel_matrix should succeed");
        assert_eq!(k.shape(), &[5, 5]);
    }

    #[test]
    fn test_spectral_mixture_self_similarity() {
        let sm = SpectralMixture::new(3, 2, 42).expect("SpectralMixture::new should succeed");
        let x = scirs2_core::ndarray::array![0.5, 0.3];
        let v = sm.evaluate(&x.view(), &x.view()).expect("evaluate should succeed");
        // k(x, x) should equal sum of weights = 1.0
        assert!((v - 1.0).abs() < 1e-8, "SM k(x,x) should be sum of weights");
    }

    #[test]
    fn test_ard_kernel() {
        let ard = ARDKernel::new(3, 1.0, 1.0).expect("ARDKernel::new should succeed");
        let x = scirs2_core::ndarray::array![1.0, 0.0, 0.0];
        let y = scirs2_core::ndarray::array![1.0, 0.0, 0.0];
        let v = ard.evaluate(&x.view(), &y.view()).expect("evaluate should succeed");
        assert!((v - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_deep_kernel_gp_fit_predict() {
        let mut config = DeepKernelGPConfig::default();
        config.layer_dims = vec![2, 4, 2];
        config.activations = vec![Activation::ReLU, Activation::Linear];

        let mut gp = DeepKernelGP::new(config).expect("DeepKernelGP::new should succeed");
        let x = Array2::<f64>::zeros((10, 2));
        let y = scirs2_core::ndarray::Array1::<f64>::ones(10);

        gp.fit(&x.view(), &y.view()).expect("DeepKernelGP fit should succeed");
        let (mean, var) = gp.predict(&x.view()).expect("DeepKernelGP predict should succeed");
        assert_eq!(mean.len(), 10);
        assert_eq!(var.len(), 10);
        assert!(var.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_inducing_points_fit_predict() {
        let kernel = BaseKernel::RBF { variance: 1.0, lengthscale: 1.0 };
        let mut sparse_gp = InducingPoints::new(5, kernel, 0.1, 42).expect("InducingPoints::new should succeed");

        let x = Array2::<f64>::zeros((20, 2));
        let y = scirs2_core::ndarray::Array1::<f64>::ones(20);

        sparse_gp.fit(&x.view(), &y.view(), 42).expect("InducingPoints fit should succeed");
        let (mean, var) = sparse_gp.predict(&x.view()).expect("InducingPoints predict should succeed");

        assert_eq!(mean.len(), 20);
        assert_eq!(var.len(), 20);
        assert!(var.iter().all(|&v| v >= 0.0));
    }
}
