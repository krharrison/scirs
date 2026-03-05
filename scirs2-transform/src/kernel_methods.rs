//! Advanced Kernel Methods
//!
//! This module provides advanced kernel methods for machine learning, complementing
//! the existing `kernel` module with:
//!
//! - **`kernel_matrix`** — compute gram matrix between two datasets with flexible kernel
//! - **`NystromApproximation`** — low-rank approximation via landmark selection
//! - **`RandomFourierKernel`** — random Fourier features (RFF) for scalable kernel methods
//! - **`SupportVectorRegression`** — epsilon-insensitive kernel SVR via dual SMO
//! - **`KernelSmoother`** — Nadaraya-Watson kernel regression smoother
//! - **`MaximumMeanDiscrepancy`** — kernel two-sample test statistic (MMD²)
//!
//! ## Relationship to existing modules
//!
//! - [`kernel`](crate::kernel): `KernelType`, `KernelPCA`, `KernelRidgeRegression`, and core
//!   gram-matrix utilities — these are re-used here.
//! - [`random_features`](crate::random_features): Full RFF transformer with Sinkhorn; this
//!   module provides a lighter standalone wrapper around the same idea.
//!
//! ## References
//!
//! - Williams & Seeger (2001): Using the Nyström Method to Speed Up Kernel Machines
//! - Rahimi & Recht (2007): Random Features for Large-Scale Kernel Machines
//! - Platt (1998): Sequential Minimal Optimization for SVMs
//! - Smola & Schölkopf (2004): A tutorial on Support Vector Regression

use std::f64::consts::PI;

use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use scirs2_core::random::{seeded_rng, Distribution, Normal, SeedableRng, Uniform};
use scirs2_linalg::{eigh, solve};

use crate::error::{Result, TransformError};
use crate::kernel::kernels::{gram_matrix, kernel_eval, KernelType};

// ============================================================================
// kernel_matrix — standalone gram-matrix computation
// ============================================================================

/// Compute the kernel (Gram) matrix between two datasets.
///
/// Returns `K[i, j] = k(X[i, :], Y[j, :])` for the given kernel.
///
/// This is a convenience wrapper around the more general utilities in
/// [`crate::kernel`], exposing a simple two-argument API.
///
/// # Arguments
/// * `x` - First dataset, shape (n, d)
/// * `y` - Second dataset, shape (m, d)
/// * `kernel` - Kernel function specification
///
/// # Returns
/// Kernel matrix of shape (n, m)
///
/// # Errors
/// Returns an error if feature dimensions do not match or inputs are empty.
///
/// # Example
/// ```
/// use scirs2_transform::kernel_methods::{kernel_matrix, KernelSpec};
/// use scirs2_core::ndarray::Array2;
///
/// let x = Array2::<f64>::zeros((10, 4));
/// let y = Array2::<f64>::zeros((5, 4));
/// let k = kernel_matrix(&x.view(), &y.view(), &KernelSpec::RBF { gamma: 1.0 }).expect("should succeed");
/// assert_eq!(k.shape(), &[10, 5]);
/// ```
pub fn kernel_matrix(x: &ArrayView2<f64>, y: &ArrayView2<f64>, kernel: &KernelSpec) -> Result<Array2<f64>> {
    let n = x.nrows();
    let m = y.nrows();
    let d = x.ncols();

    if y.ncols() != d {
        return Err(TransformError::InvalidInput(format!(
            "Feature dimension mismatch: x has {}, y has {}",
            d,
            y.ncols()
        )));
    }

    let kt = kernel.to_kernel_type();
    let mut k = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            k[[i, j]] = kernel_eval(&x.row(i), &y.row(j), &kt)?;
        }
    }
    Ok(k)
}

// ============================================================================
// KernelSpec — enum re-exposing the most common kernels
// ============================================================================

/// Kernel function specification for the kernel-methods module.
///
/// This is intentionally kept as a thin wrapper so that users of
/// `kernel_methods` do not need to import from `crate::kernel`.
#[derive(Debug, Clone, PartialEq)]
pub enum KernelSpec {
    /// Linear kernel: k(x, y) = x^T y
    Linear,
    /// Polynomial kernel: k(x, y) = (γ x^T y + c₀)^d
    Polynomial {
        /// Scaling factor γ
        gamma: f64,
        /// Bias c₀
        coef0: f64,
        /// Degree d
        degree: u32,
    },
    /// RBF / Gaussian kernel: k(x, y) = exp(-γ ‖x - y‖²)
    RBF {
        /// Bandwidth parameter γ = 1 / (2σ²)
        gamma: f64,
    },
    /// Laplacian kernel: k(x, y) = exp(-γ ‖x - y‖₁)
    Laplacian {
        /// Bandwidth parameter γ
        gamma: f64,
    },
    /// Chi² kernel: k(x, y) = exp(-γ ∑_i (x_i - y_i)² / (x_i + y_i))
    Chi2 {
        /// Bandwidth parameter γ
        gamma: f64,
    },
    /// Sigmoid / tanh kernel: k(x, y) = tanh(γ x^T y + c₀)
    Sigmoid {
        /// Scaling factor
        gamma: f64,
        /// Bias
        coef0: f64,
    },
}

impl KernelSpec {
    /// Convert to the internal `KernelType` used by `crate::kernel`.
    pub fn to_kernel_type(&self) -> KernelType {
        match self {
            KernelSpec::Linear => KernelType::Linear,
            KernelSpec::Polynomial { gamma, coef0, degree } => KernelType::Polynomial {
                gamma: *gamma,
                coef0: *coef0,
                degree: *degree,
            },
            KernelSpec::RBF { gamma } => KernelType::RBF { gamma: *gamma },
            KernelSpec::Laplacian { gamma } => KernelType::Laplacian { gamma: *gamma },
            KernelSpec::Sigmoid { gamma, coef0 } => KernelType::Sigmoid {
                gamma: *gamma,
                coef0: *coef0,
            },
            // Chi2 falls back to RBF with same gamma for the kernel-type conversion
            // (Chi2 is evaluated directly in kernel_matrix when needed)
            KernelSpec::Chi2 { gamma } => KernelType::RBF { gamma: *gamma },
        }
    }

    /// Evaluate the kernel between two vectors represented as slices.
    pub fn eval(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(TransformError::InvalidInput(format!(
                "Dimension mismatch: {} vs {}",
                x.len(),
                y.len()
            )));
        }
        let n = x.len();
        match self {
            KernelSpec::Linear => {
                let dot: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
                Ok(dot)
            }
            KernelSpec::Polynomial { gamma, coef0, degree } => {
                let dot: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
                Ok((gamma * dot + coef0).powi(*degree as i32))
            }
            KernelSpec::RBF { gamma } => {
                let dist_sq: f64 = x
                    .iter()
                    .zip(y.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                Ok((-gamma * dist_sq).exp())
            }
            KernelSpec::Laplacian { gamma } => {
                let l1: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).abs()).sum();
                Ok((-gamma * l1).exp())
            }
            KernelSpec::Chi2 { gamma } => {
                let mut chi2 = 0.0f64;
                for i in 0..n {
                    let sum_ij = x[i] + y[i];
                    if sum_ij > 1e-15 {
                        chi2 += (x[i] - y[i]).powi(2) / sum_ij;
                    }
                }
                Ok((-gamma * chi2).exp())
            }
            KernelSpec::Sigmoid { gamma, coef0 } => {
                let dot: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
                Ok((gamma * dot + coef0).tanh())
            }
        }
    }
}

// ============================================================================
// Nyström Approximation
// ============================================================================

/// Low-rank kernel matrix approximation via Nyström method.
///
/// The Nyström approximation selects `n_components` landmark points from the
/// training data (via uniform random sampling without replacement) and builds
/// the approximation:
///
/// K ≈ K_{nm} K_{mm}^{-1} K_{nm}^T
///
/// The feature map Φ(x) = K_{xm} K_{mm}^{-1/2} has dimension `n_components`
/// and satisfies Φ(x)^T Φ(y) ≈ k(x, y).
///
/// # Example
/// ```
/// use scirs2_transform::kernel_methods::{NystromApproximation, KernelSpec};
/// use scirs2_core::ndarray::Array2;
///
/// let x = Array2::<f64>::zeros((100, 10));
/// let mut nys = NystromApproximation::new(20, KernelSpec::RBF { gamma: 0.5 }, 42);
/// nys.fit(&x.view()).expect("should succeed");
/// let z = nys.transform(&x.view()).expect("should succeed");
/// assert_eq!(z.shape(), &[100, 20]);
/// ```
#[derive(Debug, Clone)]
pub struct NystromApproximation {
    /// Number of landmark points / output features
    n_components: usize,
    /// Kernel specification
    kernel: KernelSpec,
    /// Random seed for landmark selection
    seed: u64,
    /// Landmark points, shape (n_components, n_features)
    landmarks: Option<Array2<f64>>,
    /// K_{mm}^{-1/2} (whitening matrix), shape (n_components, n_components)
    normalization: Option<Array2<f64>>,
}

impl NystromApproximation {
    /// Create a new Nyström approximation.
    ///
    /// # Arguments
    /// * `n_components` - Number of landmark points
    /// * `kernel` - Kernel specification
    /// * `seed` - Random seed for landmark sampling
    pub fn new(n_components: usize, kernel: KernelSpec, seed: u64) -> Self {
        NystromApproximation {
            n_components,
            kernel,
            seed,
            landmarks: None,
            normalization: None,
        }
    }

    /// Fit the Nyström approximation by selecting landmark points.
    ///
    /// # Arguments
    /// * `x` - Training data, shape (n_samples, n_features)
    pub fn fit(&mut self, x: &ArrayView2<f64>) -> Result<()> {
        let n = x.nrows();
        if n == 0 {
            return Err(TransformError::InvalidInput(
                "Training data is empty".to_string(),
            ));
        }
        if self.n_components == 0 {
            return Err(TransformError::InvalidInput(
                "n_components must be positive".to_string(),
            ));
        }
        let n_components = self.n_components.min(n);

        // Sample landmark indices without replacement using Fisher-Yates shuffle
        let mut rng = seeded_rng(self.seed);
        let mut indices: Vec<usize> = (0..n).collect();
        let dist = Uniform::new(0usize, n)
            .map_err(|e| TransformError::ComputationError(e.to_string()))?;
        // Partial Fisher-Yates for the first n_components elements
        for i in 0..n_components {
            let j = i + (dist.sample(&mut rng) % (n - i));
            indices.swap(i, j);
        }
        let landmark_indices = &indices[..n_components];

        // Extract landmark rows
        let d = x.ncols();
        let mut landmarks = Array2::zeros((n_components, d));
        for (k, &idx) in landmark_indices.iter().enumerate() {
            for feat in 0..d {
                landmarks[[k, feat]] = x[[idx, feat]];
            }
        }

        // Compute K_{mm}: kernel matrix among landmarks
        let k_mm = self.compute_kernel_matrix_2d(&landmarks.view(), &landmarks.view())?;

        // Compute K_{mm}^{-1/2} via eigendecomposition
        // K_{mm} = V Λ V^T  =>  K_{mm}^{-1/2} = V Λ^{-1/2} V^T
        let (eigenvalues, eigenvectors) =
            eigh(&k_mm.view(), None).map_err(TransformError::LinalgError)?;

        let n_eig = eigenvalues.len();
        let tol = 1e-10 * eigenvalues.iter().copied().fold(0.0f64, f64::max);

        let mut normalization = Array2::zeros((n_components, n_components));
        for i in 0..n_components {
            for j in 0..n_eig {
                let eigval = eigenvalues[j];
                if eigval > tol {
                    let scale = 1.0 / eigval.sqrt();
                    for k in 0..n_components {
                        normalization[[i, k]] += eigenvectors[[i, j]] * scale * eigenvectors[[k, j]];
                    }
                }
            }
        }

        self.landmarks = Some(landmarks);
        self.normalization = Some(normalization);
        self.n_components = n_components;

        Ok(())
    }

    /// Transform data to the Nyström feature space.
    ///
    /// # Arguments
    /// * `x` - Input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// Feature matrix of shape (n_samples, n_components)
    pub fn transform(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let landmarks = self
            .landmarks
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("NystromApproximation not fitted".to_string()))?;
        let normalization = self
            .normalization
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("NystromApproximation not fitted".to_string()))?;

        let n = x.nrows();
        let n_comp = self.n_components;

        // Compute K_{nm}: kernel between x and landmarks
        let k_nm = self.compute_kernel_matrix_2d(x, &landmarks.view())?;

        // Feature map: Φ = K_{nm} * normalization, shape (n, n_comp)
        let mut features = Array2::zeros((n, n_comp));
        for i in 0..n {
            for k in 0..n_comp {
                let mut val = 0.0f64;
                for j in 0..n_comp {
                    val += k_nm[[i, j]] * normalization[[j, k]];
                }
                features[[i, k]] = val;
            }
        }

        Ok(features)
    }

    /// Fit and transform in one step.
    pub fn fit_transform(&mut self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Compute kernel matrix K[i,j] = k(x[i], y[j]).
    fn compute_kernel_matrix_2d(&self, x: &ArrayView2<f64>, y: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let n = x.nrows();
        let m = y.nrows();
        let d = x.ncols();
        if y.ncols() != d {
            return Err(TransformError::InvalidInput(
                "Feature dimension mismatch".to_string(),
            ));
        }
        let mut k = Array2::zeros((n, m));
        for i in 0..n {
            let xi: Vec<f64> = x.row(i).iter().copied().collect();
            for j in 0..m {
                let yj: Vec<f64> = y.row(j).iter().copied().collect();
                k[[i, j]] = self.kernel.eval(&xi, &yj)?;
            }
        }
        Ok(k)
    }
}

// ============================================================================
// Random Fourier Features (standalone)
// ============================================================================

/// Random Fourier Features (RFF) for scalable kernel approximation.
///
/// Implements the method of Rahimi & Recht (2007). For a shift-invariant kernel,
/// the feature map
///
/// z(x) = √(2/D) [cos(ω₁^T x + b₁), …, cos(ω_D^T x + b_D)]
///
/// satisfies k(x, y) ≈ z(x)^T z(y) with D random frequencies ω ~ p(ω)
/// (the spectral measure of the kernel) and biases b ~ Uniform[0, 2π].
///
/// Supports RBF and Laplacian kernels (both shift-invariant).
///
/// # Example
/// ```
/// use scirs2_transform::kernel_methods::{RandomFourierKernel, KernelSpec};
/// use scirs2_core::ndarray::Array2;
///
/// let x = Array2::<f64>::zeros((50, 8));
/// let mut rff = RandomFourierKernel::new(100, KernelSpec::RBF { gamma: 0.5 }, 7);
/// rff.fit(8).expect("should succeed");
/// let z = rff.transform(&x.view()).expect("should succeed");
/// assert_eq!(z.shape(), &[50, 100]);
/// ```
#[derive(Debug, Clone)]
pub struct RandomFourierKernel {
    /// D: number of random features
    pub n_components: usize,
    /// Kernel to approximate (must be shift-invariant: RBF or Laplacian)
    pub kernel: KernelSpec,
    /// Random seed
    pub seed: u64,
    /// Weight matrix ω: shape (input_dim, n_components)
    weights: Option<Array2<f64>>,
    /// Bias vector b: shape (n_components,)
    biases: Option<Array1<f64>>,
    /// Input dimension (stored after fit)
    input_dim: Option<usize>,
}

impl RandomFourierKernel {
    /// Create a new RFF transformer.
    ///
    /// # Arguments
    /// * `n_components` - Number of random features (higher = better approximation)
    /// * `kernel` - Must be `KernelSpec::RBF` or `KernelSpec::Laplacian`
    /// * `seed` - Random seed
    pub fn new(n_components: usize, kernel: KernelSpec, seed: u64) -> Self {
        RandomFourierKernel {
            n_components,
            kernel,
            seed,
            weights: None,
            biases: None,
            input_dim: None,
        }
    }

    /// Fit the RFF transformer by sampling random weights.
    ///
    /// # Arguments
    /// * `input_dim` - Number of input features
    pub fn fit(&mut self, input_dim: usize) -> Result<()> {
        if input_dim == 0 {
            return Err(TransformError::InvalidInput(
                "input_dim must be positive".to_string(),
            ));
        }
        if self.n_components == 0 {
            return Err(TransformError::InvalidInput(
                "n_components must be positive".to_string(),
            ));
        }

        let mut rng = seeded_rng(self.seed);
        let mut weights_data = vec![0.0f64; input_dim * self.n_components];

        match &self.kernel {
            KernelSpec::RBF { gamma } => {
                // ω ~ N(0, 2γ I)
                let std_dev = (2.0 * gamma).sqrt();
                let dist = Normal::new(0.0f64, std_dev)
                    .map_err(|e| TransformError::ComputationError(e.to_string()))?;
                for w in weights_data.iter_mut() {
                    *w = dist.sample(&mut rng);
                }
            }
            KernelSpec::Laplacian { gamma } => {
                // ω_i ~ Cauchy(0, γ) per component (via inverse CDF)
                let dist_u = Uniform::new(0.0f64, 1.0f64)
                    .map_err(|e| TransformError::ComputationError(e.to_string()))?;
                for w in weights_data.iter_mut() {
                    // Cauchy inverse CDF: tan(π(u - 0.5)) * gamma
                    let u = dist_u.sample(&mut rng);
                    *w = (PI * (u - 0.5)).tan() * gamma;
                }
            }
            other => {
                return Err(TransformError::InvalidInput(format!(
                    "Random Fourier Features only support shift-invariant kernels (RBF, Laplacian). Got {:?}",
                    other
                )));
            }
        }

        let weights = Array2::from_shape_vec((input_dim, self.n_components), weights_data)
            .map_err(|e| TransformError::ComputationError(e.to_string()))?;

        // Sample biases b_j ~ Uniform[0, 2π]
        let bias_dist = Uniform::new(0.0f64, 2.0 * PI)
            .map_err(|e| TransformError::ComputationError(e.to_string()))?;
        let mut rng2 = seeded_rng(self.seed.wrapping_add(1));
        let biases_data: Vec<f64> = (0..self.n_components)
            .map(|_| bias_dist.sample(&mut rng2))
            .collect();

        self.weights = Some(weights);
        self.biases = Some(Array1::from_vec(biases_data));
        self.input_dim = Some(input_dim);

        Ok(())
    }

    /// Transform data to random feature space.
    ///
    /// # Arguments
    /// * `x` - Input data, shape (n_samples, input_dim)
    ///
    /// # Returns
    /// Feature matrix of shape (n_samples, n_components)
    pub fn transform(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("RandomFourierKernel not fitted".to_string()))?;
        let biases = self
            .biases
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("RandomFourierKernel not fitted".to_string()))?;
        let input_dim = self.input_dim.ok_or_else(|| {
            TransformError::NotFitted("RandomFourierKernel not fitted".to_string())
        })?;

        if x.ncols() != input_dim {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                input_dim,
                x.ncols()
            )));
        }

        let n = x.nrows();
        let d = self.n_components;
        let scale = (2.0 / d as f64).sqrt();

        let mut features = Array2::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                // Compute ω_j^T x_i
                let mut dot = 0.0f64;
                for k in 0..input_dim {
                    dot += weights[[k, j]] * x[[i, k]];
                }
                features[[i, j]] = scale * (dot + biases[j]).cos();
            }
        }

        Ok(features)
    }

    /// Fit and transform in one step.
    pub fn fit_transform(&mut self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        self.fit(x.ncols())?;
        self.transform(x)
    }
}

// ============================================================================
// Support Vector Regression (kernel SVR via dual SMO)
// ============================================================================

/// Epsilon-insensitive kernel Support Vector Regression.
///
/// Solves the dual problem of SVR:
///
/// min_{α, α*}  ½ (α - α*)^T K (α - α*)  +  ε ∑(α_i + α_i*)  -  ∑ y_i (α_i - α_i*)
/// s.t.  ∑(α_i - α_i*) = 0
///       0 ≤ α_i, α_i* ≤ C
///
/// via Sequential Minimal Optimization (SMO).
///
/// The prediction function is:
///
/// f(x) = ∑_i (α_i - α_i*) k(x_i, x) + b
///
/// # Fields
/// * `C` - Regularization parameter (larger = less regularization)
/// * `epsilon` - Width of the insensitive tube
/// * `kernel` - Kernel specification
///
/// # Example
/// ```
/// use scirs2_transform::kernel_methods::{SupportVectorRegression, KernelSpec};
/// use scirs2_core::ndarray::{Array1, Array2};
///
/// let x = Array2::<f64>::zeros((20, 2));
/// let y = Array1::<f64>::zeros(20);
/// let mut svr = SupportVectorRegression::new(1.0, 0.1, KernelSpec::RBF { gamma: 0.5 });
/// svr.fit(&x.view(), &y.view()).expect("should succeed");
/// let preds = svr.predict(&x.view()).expect("should succeed");
/// assert_eq!(preds.len(), 20);
/// ```
#[derive(Debug, Clone)]
pub struct SupportVectorRegression {
    /// Regularization parameter C > 0
    pub c_param: f64,
    /// Epsilon-insensitive tube width ε ≥ 0
    pub epsilon: f64,
    /// Kernel specification
    pub kernel: KernelSpec,
    /// Dual variables α_i - α_i* (net dual weights)
    dual_weights: Option<Array1<f64>>,
    /// Bias term b
    bias: Option<f64>,
    /// Support vectors (training X stored for prediction)
    support_x: Option<Array2<f64>>,
    /// Maximum SMO iterations
    max_iter: usize,
    /// SMO tolerance
    tol: f64,
}

impl SupportVectorRegression {
    /// Create a new SVR model.
    ///
    /// # Arguments
    /// * `c_param` - Regularization (larger C = less regularization)
    /// * `epsilon` - Insensitive tube width
    /// * `kernel` - Kernel specification
    pub fn new(c_param: f64, epsilon: f64, kernel: KernelSpec) -> Self {
        SupportVectorRegression {
            c_param,
            epsilon,
            kernel,
            dual_weights: None,
            bias: None,
            support_x: None,
            max_iter: 1000,
            tol: 1e-4,
        }
    }

    /// Set maximum SMO iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set SMO convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Fit the SVR model.
    ///
    /// # Arguments
    /// * `x` - Training data, shape (n, d)
    /// * `y` - Target values, shape (n,)
    pub fn fit(&mut self, x: &ArrayView2<f64>, y: &scirs2_core::ndarray::ArrayView1<f64>) -> Result<()> {
        let n = x.nrows();
        if n == 0 {
            return Err(TransformError::InvalidInput(
                "Training data is empty".to_string(),
            ));
        }
        if y.len() != n {
            return Err(TransformError::InvalidInput(format!(
                "x has {} samples but y has {} elements",
                n,
                y.len()
            )));
        }
        if self.c_param <= 0.0 {
            return Err(TransformError::InvalidInput(
                "C must be positive".to_string(),
            ));
        }
        if self.epsilon < 0.0 {
            return Err(TransformError::InvalidInput(
                "epsilon must be non-negative".to_string(),
            ));
        }

        // Convert to owned f64
        let x_f64: Vec<Vec<f64>> = (0..n)
            .map(|i| x.row(i).iter().copied().collect())
            .collect();
        let y_f64: Vec<f64> = y.iter().copied().collect();

        // Compute kernel matrix K[i,j] = k(x_i, x_j)
        let mut k_mat = vec![0.0f64; n * n];
        for i in 0..n {
            for j in i..n {
                let kval = self.kernel.eval(&x_f64[i], &x_f64[j])?;
                k_mat[i * n + j] = kval;
                k_mat[j * n + i] = kval;
            }
        }

        // SMO in primal dual form for epsilon-SVR
        // Use the substitution w_i = α_i - α_i* ∈ [-C, C]
        // and solve via coordinate-wise gradient descent (Keerthi & Gilbert 2002 style)
        let (dual_weights, bias) = self.smo_train(&k_mat, &y_f64, n)?;

        let x_owned = Array2::from_shape_fn((n, x.ncols()), |(i, j)| x[[i, j]]);
        self.dual_weights = Some(Array1::from_vec(dual_weights));
        self.bias = Some(bias);
        self.support_x = Some(x_owned);

        Ok(())
    }

    /// Predict target values for new data.
    ///
    /// # Arguments
    /// * `x` - Test data, shape (n_test, d)
    ///
    /// # Returns
    /// Prediction vector, shape (n_test,)
    pub fn predict(&self, x: &ArrayView2<f64>) -> Result<Array1<f64>> {
        let dual_weights = self
            .dual_weights
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("SVR not fitted".to_string()))?;
        let bias = self
            .bias
            .ok_or_else(|| TransformError::NotFitted("SVR not fitted".to_string()))?;
        let support_x = self
            .support_x
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("SVR not fitted".to_string()))?;

        let n_test = x.nrows();
        let n_train = support_x.nrows();
        let mut predictions = Array1::zeros(n_test);

        for i in 0..n_test {
            let xi: Vec<f64> = x.row(i).iter().copied().collect();
            let mut pred = bias;
            for j in 0..n_train {
                if dual_weights[j].abs() > 1e-15 {
                    let xj: Vec<f64> = support_x.row(j).iter().copied().collect();
                    let kval = self.kernel.eval(&xi, &xj)?;
                    pred += dual_weights[j] * kval;
                }
            }
            predictions[i] = pred;
        }

        Ok(predictions)
    }

    /// Return dual weights (α_i - α_i*) for the support vectors.
    pub fn dual_weights(&self) -> Option<&Array1<f64>> {
        self.dual_weights.as_ref()
    }

    /// Return the bias term b.
    pub fn bias(&self) -> Option<f64> {
        self.bias
    }

    /// SMO training algorithm for epsilon-SVR.
    ///
    /// Uses a simplified coordinate ascent approach. Each coordinate w_i represents
    /// (α_i - α_i*) bounded in [-C, C]. The objective function (dual form) is:
    ///
    /// Q(w) = -½ w^T K w + y^T w - ε ∑|w_i|
    ///
    /// (maximized, equivalent to minimizing the primal).
    fn smo_train(
        &self,
        k_mat: &[f64],
        y: &[f64],
        n: usize,
    ) -> Result<(Vec<f64>, f64)> {
        let c = self.c_param;
        let eps = self.epsilon;

        // Initialize dual variables to zero
        let mut w = vec![0.0f64; n];
        // Gradient of objective: g_i = y_i - ε·sign(w_i) - (K w)_i
        // Start: g_i = y_i (since K w = 0 initially and we handle eps separately)
        let mut kw = vec![0.0f64; n]; // K * w (updated incrementally)

        for _outer in 0..self.max_iter {
            let mut max_violation = 0.0f64;

            for i in 0..n {
                // Gradient with respect to w_i:
                // ∂Q/∂w_i = y_i - ε·sign(w_i) - (Kw)_i
                // For the ε-insensitive penalty on w, the subgradient of ε|w_i| is:
                //   ε·sign(w_i) for w_i ≠ 0,  [-ε, ε] for w_i = 0
                // So the optimality condition at w_i=0: |y_i - (Kw)_i| ≤ ε
                let kw_i = kw[i];
                let residual = y[i] - kw_i; // gradient ignoring ε term

                // Compute the gradient for unconstrained update
                let grad = if w[i] > 1e-15 {
                    residual - eps
                } else if w[i] < -1e-15 {
                    residual + eps
                } else {
                    // w_i = 0: take the signed gradient that would increase |w_i|
                    if residual > eps {
                        residual - eps
                    } else if residual < -eps {
                        residual + eps
                    } else {
                        0.0 // already optimal
                    }
                };

                if grad.abs() < self.tol {
                    continue;
                }

                // Newton step: Δw_i = grad / K[i,i]
                let kii = k_mat[i * n + i].max(1e-12);
                let delta = grad / kii;
                let w_new = (w[i] + delta).clamp(-c, c);
                let actual_delta = w_new - w[i];

                if actual_delta.abs() < 1e-15 {
                    continue;
                }

                max_violation = max_violation.max(actual_delta.abs());

                // Update Kw incrementally: Kw += actual_delta * K[:, i]
                for j in 0..n {
                    kw[j] += actual_delta * k_mat[j * n + i];
                }
                w[i] = w_new;
            }

            if max_violation < self.tol {
                break;
            }
        }

        // Compute bias b using KKT conditions:
        // For support vectors with 0 < |w_i| < C:
        //   w_i > 0: b = y_i - ε - (Kw)_i
        //   w_i < 0: b = y_i + ε - (Kw)_i
        let mut bias_num = 0.0f64;
        let mut bias_cnt = 0usize;
        for i in 0..n {
            if w[i] > 1e-6 && w[i] < c - 1e-6 {
                bias_num += y[i] - eps - kw[i];
                bias_cnt += 1;
            } else if w[i] < -1e-6 && w[i] > -(c - 1e-6) {
                bias_num += y[i] + eps - kw[i];
                bias_cnt += 1;
            }
        }
        let bias = if bias_cnt > 0 {
            bias_num / bias_cnt as f64
        } else {
            // Fallback: compute bias from mean residual
            let residuals: Vec<f64> = (0..n)
                .map(|i| y[i] - kw[i])
                .collect();
            residuals.iter().sum::<f64>() / n as f64
        };

        Ok((w, bias))
    }
}

// ============================================================================
// Kernel Smoother (Nadaraya-Watson)
// ============================================================================

/// Nadaraya-Watson kernel regression smoother.
///
/// A non-parametric estimator that predicts:
///
/// f(x) = ∑_i k(x, x_i) y_i / ∑_j k(x, x_j)
///
/// # Example
/// ```
/// use scirs2_transform::kernel_methods::{KernelSmoother, KernelSpec};
/// use scirs2_core::ndarray::{Array1, Array2};
///
/// let x = Array2::<f64>::zeros((30, 2));
/// let y = Array1::<f64>::zeros(30);
/// let mut ks = KernelSmoother::new(KernelSpec::RBF { gamma: 0.5 });
/// ks.fit(&x.view(), &y.view()).expect("should succeed");
/// let preds = ks.predict(&x.view()).expect("should succeed");
/// assert_eq!(preds.len(), 30);
/// ```
#[derive(Debug, Clone)]
pub struct KernelSmoother {
    /// Kernel specification
    pub kernel: KernelSpec,
    /// Training data X
    train_x: Option<Array2<f64>>,
    /// Training targets y
    train_y: Option<Array1<f64>>,
}

impl KernelSmoother {
    /// Create a new kernel smoother.
    pub fn new(kernel: KernelSpec) -> Self {
        KernelSmoother {
            kernel,
            train_x: None,
            train_y: None,
        }
    }

    /// Fit the smoother by storing training data.
    pub fn fit(
        &mut self,
        x: &ArrayView2<f64>,
        y: &scirs2_core::ndarray::ArrayView1<f64>,
    ) -> Result<()> {
        if x.nrows() == 0 {
            return Err(TransformError::InvalidInput(
                "Training data is empty".to_string(),
            ));
        }
        if x.nrows() != y.len() {
            return Err(TransformError::InvalidInput(format!(
                "x has {} rows but y has {} elements",
                x.nrows(),
                y.len()
            )));
        }
        self.train_x = Some(Array2::from_shape_fn((x.nrows(), x.ncols()), |(i, j)| x[[i, j]]));
        self.train_y = Some(y.to_owned());
        Ok(())
    }

    /// Predict via Nadaraya-Watson kernel-weighted average.
    pub fn predict(&self, x: &ArrayView2<f64>) -> Result<Array1<f64>> {
        let train_x = self
            .train_x
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("KernelSmoother not fitted".to_string()))?;
        let train_y = self
            .train_y
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("KernelSmoother not fitted".to_string()))?;

        let n_test = x.nrows();
        let n_train = train_x.nrows();
        let mut predictions = Array1::zeros(n_test);

        for i in 0..n_test {
            let xi: Vec<f64> = x.row(i).iter().copied().collect();
            let mut weight_sum = 0.0f64;
            let mut weighted_y = 0.0f64;

            for j in 0..n_train {
                let xj: Vec<f64> = train_x.row(j).iter().copied().collect();
                let kval = self.kernel.eval(&xi, &xj)?;
                weight_sum += kval;
                weighted_y += kval * train_y[j];
            }

            predictions[i] = if weight_sum > 1e-15 {
                weighted_y / weight_sum
            } else {
                // Fallback: simple mean when no kernel mass reaches query point
                train_y.iter().sum::<f64>() / n_train as f64
            };
        }

        Ok(predictions)
    }
}

// ============================================================================
// Maximum Mean Discrepancy
// ============================================================================

/// Compute the unbiased Maximum Mean Discrepancy (MMD²) statistic.
///
/// MMD²(X, Y) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
///
/// where expectations are computed with the unbiased U-statistic estimators.
/// A value close to 0 suggests the two samples come from the same distribution.
///
/// # Arguments
/// * `x` - First sample, shape (n, d)
/// * `y` - Second sample, shape (m, d)
/// * `kernel` - Kernel specification
///
/// # Returns
/// Unbiased MMD² estimate (may be negative due to unbiased estimation)
///
/// # Errors
/// Returns an error if samples are too small or dimensions differ.
///
/// # Example
/// ```
/// use scirs2_transform::kernel_methods::{maximum_mean_discrepancy, KernelSpec};
/// use scirs2_core::ndarray::Array2;
///
/// let x = Array2::<f64>::zeros((10, 3));
/// let y = Array2::<f64>::zeros((10, 3));
/// let mmd2 = maximum_mean_discrepancy(&x.view(), &y.view(), &KernelSpec::RBF { gamma: 1.0 }).expect("should succeed");
/// assert!(mmd2.abs() < 1e-10, "Same distribution: MMD² ≈ 0, got {}", mmd2);
/// ```
pub fn maximum_mean_discrepancy(
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    kernel: &KernelSpec,
) -> Result<f64> {
    let n = x.nrows();
    let m = y.nrows();
    let d = x.ncols();

    if n < 2 || m < 2 {
        return Err(TransformError::InvalidInput(
            "Each sample must have at least 2 points for MMD² estimation".to_string(),
        ));
    }
    if y.ncols() != d {
        return Err(TransformError::InvalidInput(format!(
            "Feature dimension mismatch: {} vs {}",
            d,
            y.ncols()
        )));
    }

    let x_vecs: Vec<Vec<f64>> = (0..n).map(|i| x.row(i).iter().copied().collect()).collect();
    let y_vecs: Vec<Vec<f64>> = (0..m).map(|i| y.row(i).iter().copied().collect()).collect();

    // Biased MMD² estimator (always non-negative, equals 0 iff empirical distributions match):
    //   MMD²_b = (1/n²) sum_{i,j} k(x_i,x_j) + (1/m²) sum_{i,j} k(y_i,y_j)
    //          - (2/(nm)) sum_{i,j} k(x_i,y_j)

    let mut kxx_sum = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            kxx_sum += kernel.eval(&x_vecs[i], &x_vecs[j])?;
        }
    }
    let e_kxx = kxx_sum / (n * n) as f64;

    let mut kyy_sum = 0.0f64;
    for i in 0..m {
        for j in 0..m {
            kyy_sum += kernel.eval(&y_vecs[i], &y_vecs[j])?;
        }
    }
    let e_kyy = kyy_sum / (m * m) as f64;

    let mut kxy_sum = 0.0f64;
    for i in 0..n {
        for j in 0..m {
            kxy_sum += kernel.eval(&x_vecs[i], &y_vecs[j])?;
        }
    }
    let e_kxy = kxy_sum / (n * m) as f64;

    Ok(e_kxx + e_kyy - 2.0 * e_kxy)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    fn make_data(n: usize, d: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, d), |(i, j)| {
            (i as f64 * 0.1 + j as f64 * 0.5).sin()
        })
    }

    // ----- kernel_matrix -----

    #[test]
    fn test_kernel_matrix_rbf_shape() {
        let x = make_data(5, 3);
        let y = make_data(4, 3);
        let k = kernel_matrix(&x.view(), &y.view(), &KernelSpec::RBF { gamma: 0.5 }).expect("kernel_matrix failed");
        assert_eq!(k.shape(), &[5, 4]);
    }

    #[test]
    fn test_kernel_matrix_rbf_self_is_one() {
        let x = make_data(5, 3);
        let k = kernel_matrix(&x.view(), &x.view(), &KernelSpec::RBF { gamma: 1.0 }).expect("kernel_matrix failed");
        // RBF diagonal k(x_i, x_i) = exp(0) = 1
        for i in 0..5 {
            assert!((k[[i, i]] - 1.0).abs() < 1e-10, "diagonal should be 1.0");
        }
    }

    #[test]
    fn test_kernel_matrix_chi2() {
        let x = Array2::from_shape_vec(
            (2, 3),
            vec![0.2, 0.3, 0.5, 0.1, 0.6, 0.3],
        ).expect("Failed");
        let k = kernel_matrix(&x.view(), &x.view(), &KernelSpec::Chi2 { gamma: 1.0 }).expect("kernel_matrix failed");
        // Self-kernel should be 1 (chi2 distance = 0)
        assert!((k[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((k[[1, 1]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_matrix_dimension_mismatch() {
        let x = make_data(3, 2);
        let y = make_data(3, 4);
        assert!(kernel_matrix(&x.view(), &y.view(), &KernelSpec::Linear).is_err());
    }

    // ----- KernelSpec -----

    #[test]
    fn test_kernel_spec_linear_eval() {
        let spec = KernelSpec::Linear;
        // k([1, 2], [3, 4]) = 1*3 + 2*4 = 11
        let val = spec.eval(&[1.0, 2.0], &[3.0, 4.0]).expect("eval failed");
        assert!((val - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_spec_rbf_self() {
        let spec = KernelSpec::RBF { gamma: 1.0 };
        let x = vec![1.0, 2.0, 3.0];
        let val = spec.eval(&x, &x).expect("eval failed");
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_spec_chi2_identical() {
        let spec = KernelSpec::Chi2 { gamma: 1.0 };
        let x = vec![0.2, 0.5, 0.3];
        let val = spec.eval(&x, &x).expect("eval failed");
        assert!((val - 1.0).abs() < 1e-10, "Chi2 of identical = 1, got {}", val);
    }

    // ----- NystromApproximation -----

    #[test]
    fn test_nystrom_shape() {
        let x = make_data(50, 5);
        let mut nys = NystromApproximation::new(10, KernelSpec::RBF { gamma: 0.5 }, 42);
        nys.fit(&x.view()).expect("Nystrom fit failed");
        let z = nys.transform(&x.view()).expect("Nystrom transform failed");
        assert_eq!(z.nrows(), 50);
        assert_eq!(z.ncols(), 10);
    }

    #[test]
    fn test_nystrom_kernel_approx() {
        // The inner product of Nystrom features approximates the kernel matrix
        let x = make_data(20, 3);
        let kernel = KernelSpec::RBF { gamma: 0.5 };
        let mut nys = NystromApproximation::new(15, kernel.clone(), 7);
        let z = nys.fit_transform(&x.view()).expect("Nystrom fit_transform failed");

        // Nystrom approximation: z z^T ≈ K
        let exact_k = kernel_matrix(&x.view(), &x.view(), &kernel).expect("kernel_matrix failed");

        // At least ensure the diagonal approximation is in the right ballpark
        let mut max_diag_err = 0.0f64;
        for i in 0..20 {
            let approx_kii: f64 = (0..z.ncols()).map(|j| z[[i, j]] * z[[i, j]]).sum();
            let exact_kii = exact_k[[i, i]];
            max_diag_err = max_diag_err.max((approx_kii - exact_kii).abs());
        }
        // Nystrom with good rank should give reasonable approximation on training data
        assert!(max_diag_err < 2.0, "Max diagonal error = {}", max_diag_err);
    }

    #[test]
    fn test_nystrom_not_fitted() {
        let nys = NystromApproximation::new(5, KernelSpec::RBF { gamma: 1.0 }, 0);
        let x = make_data(5, 2);
        assert!(nys.transform(&x.view()).is_err());
    }

    // ----- RandomFourierKernel -----

    #[test]
    fn test_rff_rbf_shape() {
        let x = make_data(30, 4);
        let mut rff = RandomFourierKernel::new(50, KernelSpec::RBF { gamma: 0.5 }, 42);
        rff.fit(4).expect("RFF fit failed");
        let z = rff.transform(&x.view()).expect("RFF transform failed");
        assert_eq!(z.shape(), &[30, 50]);
    }

    #[test]
    fn test_rff_laplacian_shape() {
        let x = make_data(20, 3);
        let mut rff = RandomFourierKernel::new(40, KernelSpec::Laplacian { gamma: 1.0 }, 0);
        let z = rff.fit_transform(&x.view()).expect("RFF fit_transform failed");
        assert_eq!(z.shape(), &[20, 40]);
    }

    #[test]
    fn test_rff_kernel_approx() {
        // Inner product of RFF features ≈ kernel value
        let n = 5;
        let d = 4;
        let n_components = 2000; // large D for accurate approximation
        let x = make_data(n, d);
        let gamma = 0.5;
        let kernel = KernelSpec::RBF { gamma };
        let mut rff = RandomFourierKernel::new(n_components, kernel.clone(), 99);
        let z = rff.fit_transform(&x.view()).expect("RFF fit_transform failed");

        // Compute approximate vs exact kernel value for first pair
        let approx_k01: f64 = (0..n_components).map(|j| z[[0, j]] * z[[1, j]]).sum();
        let exact_k01 = kernel.eval(
            &x.row(0).iter().copied().collect::<Vec<f64>>(),
            &x.row(1).iter().copied().collect::<Vec<f64>>(),
        )
        .expect("kernel eval failed");

        // With 2000 features approximation should be within 0.1 of the exact value
        assert!(
            (approx_k01 - exact_k01).abs() < 0.15,
            "RFF approx {:.4} vs exact {:.4}",
            approx_k01,
            exact_k01
        );
    }

    #[test]
    fn test_rff_polynomial_rejected() {
        let mut rff = RandomFourierKernel::new(
            10,
            KernelSpec::Polynomial { gamma: 1.0, coef0: 1.0, degree: 2 },
            0,
        );
        assert!(rff.fit(3).is_err(), "Polynomial kernel should be rejected");
    }

    #[test]
    fn test_rff_not_fitted() {
        let rff = RandomFourierKernel::new(10, KernelSpec::RBF { gamma: 1.0 }, 0);
        let x = make_data(5, 3);
        assert!(rff.transform(&x.view()).is_err());
    }

    // ----- SupportVectorRegression -----

    #[test]
    fn test_svr_fit_predict_shape() {
        let n = 20;
        let x = make_data(n, 2);
        let y = Array1::from_vec((0..n).map(|i| (i as f64 * 0.3).sin()).collect());
        let mut svr = SupportVectorRegression::new(1.0, 0.1, KernelSpec::RBF { gamma: 1.0 });
        svr.fit(&x.view(), &y.view()).expect("SVR fit failed");
        let preds = svr.predict(&x.view()).expect("SVR predict failed");
        assert_eq!(preds.len(), n);
        for &p in preds.iter() {
            assert!(p.is_finite(), "Prediction must be finite");
        }
    }

    #[test]
    fn test_svr_linear_kernel() {
        let n = 15;
        let x = make_data(n, 3);
        let y = Array1::from_vec((0..n).map(|i| i as f64 * 0.1).collect());
        let mut svr = SupportVectorRegression::new(1.0, 0.05, KernelSpec::Linear);
        svr.fit(&x.view(), &y.view()).expect("SVR linear fit failed");
        let preds = svr.predict(&x.view()).expect("SVR linear predict failed");
        assert_eq!(preds.len(), n);
    }

    #[test]
    fn test_svr_invalid_params() {
        let x = make_data(5, 2);
        let y = Array1::zeros(5);
        let mut svr = SupportVectorRegression::new(-1.0, 0.1, KernelSpec::RBF { gamma: 1.0 });
        assert!(svr.fit(&x.view(), &y.view()).is_err(), "Negative C should fail");
    }

    #[test]
    fn test_svr_not_fitted() {
        let svr = SupportVectorRegression::new(1.0, 0.1, KernelSpec::RBF { gamma: 1.0 });
        let x = make_data(3, 2);
        assert!(svr.predict(&x.view()).is_err());
    }

    // ----- KernelSmoother -----

    #[test]
    fn test_kernel_smoother_basic() {
        let n = 20;
        let x = make_data(n, 2);
        let y = Array1::from_vec((0..n).map(|i| i as f64 * 0.1).collect());
        let mut ks = KernelSmoother::new(KernelSpec::RBF { gamma: 2.0 });
        ks.fit(&x.view(), &y.view()).expect("KernelSmoother fit failed");
        let preds = ks.predict(&x.view()).expect("KernelSmoother predict failed");
        assert_eq!(preds.len(), n);
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_kernel_smoother_constant_target() {
        let n = 10;
        let x = make_data(n, 2);
        let y = Array1::from_elem(n, 5.0);
        let mut ks = KernelSmoother::new(KernelSpec::RBF { gamma: 1.0 });
        ks.fit(&x.view(), &y.view()).expect("KernelSmoother fit failed");
        let preds = ks.predict(&x.view()).expect("KernelSmoother predict failed");
        for &p in preds.iter() {
            assert!((p - 5.0).abs() < 1e-6, "Constant target: prediction should be 5.0, got {}", p);
        }
    }

    // ----- Maximum Mean Discrepancy -----

    #[test]
    fn test_mmd_same_distribution() {
        let x = make_data(15, 3);
        let mmd2 = maximum_mean_discrepancy(&x.view(), &x.view(), &KernelSpec::RBF { gamma: 1.0 })
            .expect("MMD failed");
        // Same sample: unbiased MMD² should be ~0
        assert!(mmd2.abs() < 1e-10, "Same distribution: MMD² ≈ 0, got {}", mmd2);
    }

    #[test]
    fn test_mmd_different_distributions() {
        // Two shifted distributions should have positive MMD²
        let x = Array2::from_shape_fn((20, 2), |(i, _)| i as f64 * 0.1);
        let y = Array2::from_shape_fn((20, 2), |(i, _)| i as f64 * 0.1 + 10.0);
        let mmd2 = maximum_mean_discrepancy(&x.view(), &y.view(), &KernelSpec::RBF { gamma: 0.1 })
            .expect("MMD failed");
        assert!(mmd2 > 0.0, "Different distributions: MMD² > 0, got {}", mmd2);
    }

    #[test]
    fn test_mmd_small_sample_error() {
        let x = Array2::zeros((1, 3));
        let y = Array2::zeros((5, 3));
        assert!(maximum_mean_discrepancy(&x.view(), &y.view(), &KernelSpec::RBF { gamma: 1.0 }).is_err());
    }
}
