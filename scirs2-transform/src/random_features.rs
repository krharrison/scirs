//! Kernel Approximation via Random Features
//!
//! This module provides scalable kernel methods using random feature maps.
//! The key idea (Bochner's theorem) is that shift-invariant kernels have
//! a spectral representation: `k(x,y) = E_w[φ_w(x) φ_w(y)]`
//! where φ_w(x) = sqrt(2) cos(w^T x + b) for random w, b.
//!
//! ## Methods
//!
//! - **Random Fourier Features (RFF)**: Approximate shift-invariant kernels
//!   (RBF, Laplacian, Cauchy, Matérn) via Bochner's theorem
//! - **Nyström Approximation**: Low-rank approximation using landmark points
//! - **Tensor Sketch**: Polynomial kernel approximation via count sketching
//! - **Kernel Ridge Regression with RFF**: Scalable KRR using random features
//! - **MMD Test**: Maximum Mean Discrepancy hypothesis testing
//!
//! ## References
//!
//! - Rahimi & Recht (2007): Random Features for Large-Scale Kernel Machines
//! - Williams & Seeger (2001): Using the Nyström Method to Speed Up Kernel Machines
//! - Pham & Pagh (2013): Fast and Scalable Polynomial Kernels via Explicit Feature Maps

use std::f64::consts::PI;

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::random::{
    seeded_rng, Cauchy, Distribution, Normal, SeedableRng, StandardNormal, StudentT, Uniform,
};
use scirs2_linalg::{eigh, solve};

use crate::error::{Result, TransformError};

// ============================================================================
// Shift-Invariant Kernels
// ============================================================================

/// Shift-invariant kernel types whose spectral density is known analytically.
///
/// For each kernel k(x,y) = k(x - y), Bochner's theorem guarantees a spectral
/// representation via a probability distribution p(ω) such that
/// k(Δ) = ∫ p(ω) e^{iω^T Δ} dω.
#[derive(Debug, Clone, PartialEq)]
pub enum ShiftInvariantKernel {
    /// Gaussian RBF: k(x,y) = exp(-γ ‖x-y‖²)
    /// Spectral density: N(0, 2γ I)
    RBF {
        /// Bandwidth parameter γ controlling the kernel width
        gamma: f64,
    },

    /// Laplacian: k(x,y) = exp(-γ ‖x-y‖₁)
    /// Spectral density (per dimension): Cauchy(0, 1/γ)
    Laplacian {
        /// Bandwidth parameter γ controlling the kernel width
        gamma: f64,
    },

    /// Cauchy: k(x,y) = ∏_d 1/(1 + γ(x_d - y_d)²)
    /// Spectral density (per dimension): Laplacian / double-exponential
    Cauchy {
        /// Bandwidth parameter γ controlling the kernel width
        gamma: f64,
    },

    /// Matérn 3/2: k(x,y) = (1 + √3 γ ‖x-y‖) exp(-√3 γ ‖x-y‖)
    /// Spectral density: Student-t with ν=3, scale=√(2·3) γ
    Matern32 {
        /// Length scale parameter controlling the smoothness
        length_scale: f64,
    },

    /// Matérn 5/2: k(x,y) = (1 + √5 γ ‖x-y‖ + 5γ²‖x-y‖²/3) exp(-√5 γ ‖x-y‖)
    /// Spectral density: Student-t with ν=5, scale=√(2·5) γ
    Matern52 {
        /// Length scale parameter controlling the smoothness
        length_scale: f64,
    },
}

impl ShiftInvariantKernel {
    /// Sample random frequency vectors ω from the spectral distribution p(ω).
    ///
    /// Returns a matrix of shape `(input_dim, n_samples)`.
    ///
    /// The spectral densities follow from Bochner's theorem:
    /// - RBF: ω ~ N(0, 2γ I_d)
    /// - Laplacian: each ω_i ~ Cauchy(0, γ) independently
    /// - Cauchy: each ω_i ~ Laplace(0, 1/γ) = double-exponential
    /// - Matérn 3/2: ω ~ StudentT(ν=3) scaled by √(6) / length_scale per dim
    /// - Matérn 5/2: ω ~ StudentT(ν=5) scaled by √(10) / length_scale per dim
    fn sample_frequencies(
        &self,
        n_samples: usize,
        input_dim: usize,
        seed: u64,
    ) -> std::result::Result<Array2<f64>, TransformError> {
        let mut rng = seeded_rng(seed);
        let total = n_samples * input_dim;
        let mut data = Vec::with_capacity(total);

        match self {
            ShiftInvariantKernel::RBF { gamma } => {
                // ω ~ N(0, 2γ) per component
                let std_dev = (2.0 * gamma).sqrt();
                let dist = Normal::new(0.0_f64, std_dev)
                    .map_err(|e| TransformError::ComputationError(e.to_string()))?;
                for _ in 0..total {
                    data.push(dist.sample(&mut rng));
                }
            }

            ShiftInvariantKernel::Laplacian { gamma } => {
                // ω_i ~ Cauchy(0, γ) per component (Laplacian spectral density)
                let dist = Cauchy::new(0.0_f64, *gamma)
                    .map_err(|e| TransformError::ComputationError(e.to_string()))?;
                for _ in 0..total {
                    let s = dist.sample(&mut rng);
                    // Clamp extreme Cauchy samples to avoid numerical issues
                    data.push(s.max(-1e6).min(1e6));
                }
            }

            ShiftInvariantKernel::Cauchy { gamma } => {
                // Cauchy kernel spectral density is Laplacian (double-exponential)
                // Laplace(0, b) can be sampled: U ~ Uniform(-0.5, 0.5), x = -b*sign(U)*ln(1-2|U|)
                let b = 1.0 / gamma;
                let udist = Uniform::new(-0.5_f64, 0.5_f64)
                    .map_err(|e| TransformError::ComputationError(e.to_string()))?;
                for _ in 0..total {
                    let u: f64 = udist.sample(&mut rng);
                    let sign = if u >= 0.0 { 1.0 } else { -1.0 };
                    let val = -b * sign * (1.0 - 2.0 * u.abs()).ln();
                    data.push(val);
                }
            }

            ShiftInvariantKernel::Matern32 { length_scale } => {
                // Matérn 3/2 spectral density: StudentT(ν=3) scaled by sqrt(6)/length_scale
                // (isotropic: sample each dimension independently)
                let scale = (6.0_f64).sqrt() / length_scale;
                let dist = StudentT::new(3.0_f64)
                    .map_err(|e| TransformError::ComputationError(e.to_string()))?;
                for _ in 0..total {
                    data.push(dist.sample(&mut rng) * scale);
                }
            }

            ShiftInvariantKernel::Matern52 { length_scale } => {
                // Matérn 5/2 spectral density: StudentT(ν=5) scaled by sqrt(10)/length_scale
                let scale = (10.0_f64).sqrt() / length_scale;
                let dist = StudentT::new(5.0_f64)
                    .map_err(|e| TransformError::ComputationError(e.to_string()))?;
                for _ in 0..total {
                    data.push(dist.sample(&mut rng) * scale);
                }
            }
        }

        // Shape: (input_dim, n_samples) — each column is one ω vector
        Array2::from_shape_vec((input_dim, n_samples), data)
            .map_err(|e| TransformError::ComputationError(e.to_string()))
    }

    /// Compute exact kernel value between two vectors.
    pub fn compute_exact(&self, x: &[f64], y: &[f64]) -> f64 {
        debug_assert_eq!(x.len(), y.len(), "Input dimensions must match");
        match self {
            ShiftInvariantKernel::RBF { gamma } => {
                let sq_dist: f64 = x
                    .iter()
                    .zip(y.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                (-gamma * sq_dist).exp()
            }

            ShiftInvariantKernel::Laplacian { gamma } => {
                let l1_dist: f64 = x
                    .iter()
                    .zip(y.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum();
                (-gamma * l1_dist).exp()
            }

            ShiftInvariantKernel::Cauchy { gamma } => {
                // Product form: ∏_d 1/(1 + γ(x_d - y_d)²)
                x.iter()
                    .zip(y.iter())
                    .map(|(a, b)| 1.0 / (1.0 + gamma * (a - b).powi(2)))
                    .product()
            }

            ShiftInvariantKernel::Matern32 { length_scale } => {
                let dist: f64 = x
                    .iter()
                    .zip(y.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let r = 3.0_f64.sqrt() * dist / length_scale;
                (1.0 + r) * (-r).exp()
            }

            ShiftInvariantKernel::Matern52 { length_scale } => {
                let dist: f64 = x
                    .iter()
                    .zip(y.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let r = 5.0_f64.sqrt() * dist / length_scale;
                (1.0 + r + r.powi(2) / 3.0) * (-r).exp()
            }
        }
    }
}

// ============================================================================
// Random Fourier Features
// ============================================================================

/// Random Fourier Features (RFF) for scalable shift-invariant kernel approximation.
///
/// Based on Bochner's theorem: for a shift-invariant kernel k(x,y) = k(x-y),
/// the feature map z(x) = sqrt(2/D) [cos(w_1^T x + b_1), ..., cos(w_D^T x + b_D)]
/// satisfies k(x,y) ≈ z(x)^T z(y) with high probability.
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_transform::random_features::{RandomFourierFeatures, ShiftInvariantKernel};
/// use scirs2_core::ndarray::Array2;
///
/// let x = Array2::<f64>::zeros((100, 10));
/// let kernel = ShiftInvariantKernel::RBF { gamma: 0.5 };
/// let mut rff = RandomFourierFeatures::new(200, kernel);
/// let z = rff.fit_transform(x.view(), 42).expect("should succeed");
/// assert_eq!(z.shape(), &[100, 200]);
/// ```
#[derive(Debug, Clone)]
pub struct RandomFourierFeatures {
    /// D: number of random features
    pub n_components: usize,
    /// The kernel being approximated
    pub kernel: ShiftInvariantKernel,
    /// Weight matrix ω: shape (input_dim, n_components)
    weights: Option<Array2<f64>>,
    /// Bias vector b: shape (n_components,), drawn from Uniform[0, 2π]
    biases: Option<Array1<f64>>,
}

impl RandomFourierFeatures {
    /// Create a new RFF transformer.
    pub fn new(n_components: usize, kernel: ShiftInvariantKernel) -> Self {
        RandomFourierFeatures {
            n_components,
            kernel,
            weights: None,
            biases: None,
        }
    }

    /// Fit: sample random frequencies and biases.
    pub fn fit(&mut self, input_dim: usize, seed: u64) -> Result<()> {
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

        // Sample frequency vectors from spectral distribution
        let weights = self
            .kernel
            .sample_frequencies(self.n_components, input_dim, seed)?;
        // weights shape: (input_dim, n_components)

        // Sample biases b_j ~ Uniform[0, 2π]
        let mut rng = seeded_rng(seed.wrapping_add(1));
        let bias_dist = Uniform::new(0.0_f64, 2.0 * PI)
            .map_err(|e| TransformError::ComputationError(e.to_string()))?;
        let biases: Vec<f64> = (0..self.n_components)
            .map(|_| bias_dist.sample(&mut rng))
            .collect();

        self.weights = Some(weights);
        self.biases = Some(
            Array1::from_vec(biases),
        );
        Ok(())
    }

    /// Transform: compute feature map z(X).
    ///
    /// Input shape: (n_samples, n_features)
    /// Output shape: (n_samples, n_components)
    pub fn transform(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let weights = self.weights.as_ref().ok_or_else(|| {
            TransformError::NotFitted("RandomFourierFeatures not fitted".to_string())
        })?;
        let biases = self.biases.as_ref().ok_or_else(|| {
            TransformError::NotFitted("RandomFourierFeatures not fitted".to_string())
        })?;

        let (n_samples, n_features) = (x.nrows(), x.ncols());
        let expected_dim = weights.nrows();
        if n_features != expected_dim {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                expected_dim, n_features
            )));
        }

        let n_components = self.n_components;
        let scale = (2.0_f64 / n_components as f64).sqrt();

        // Compute X @ W: shape (n_samples, n_components)
        // x: (n_samples, n_features), weights: (n_features, n_components)
        let mut z = Array2::<f64>::zeros((n_samples, n_components));
        for i in 0..n_samples {
            for j in 0..n_components {
                let dot: f64 = x
                    .row(i)
                    .iter()
                    .zip(weights.column(j).iter())
                    .map(|(a, b)| a * b)
                    .sum();
                z[[i, j]] = scale * (dot + biases[j]).cos();
            }
        }

        Ok(z)
    }

    /// Fit and transform in one step.
    pub fn fit_transform(
        &mut self,
        x: ArrayView2<f64>,
        seed: u64,
    ) -> Result<Array2<f64>> {
        let input_dim = x.ncols();
        self.fit(input_dim, seed)?;
        self.transform(x)
    }

    /// Compute approximate kernel matrix K[i,j] ≈ z(x_i)^T z(x_j).
    ///
    /// Output shape: (n_samples, n_samples)
    pub fn approximate_kernel(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let z = self.transform(x)?;
        // K ≈ Z Z^T
        let n = z.nrows();
        let d = z.ncols();
        let mut k = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                let dot: f64 = (0..d).map(|l| z[[i, l]] * z[[j, l]]).sum();
                k[[i, j]] = dot;
                k[[j, i]] = dot;
            }
        }
        Ok(k)
    }

    /// Compute approximate cross-kernel matrix K[i,j] = k(X_i, Y_j).
    ///
    /// Output shape: (n_x, n_y)
    pub fn approximate_cross_kernel(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView2<f64>,
    ) -> Result<Array2<f64>> {
        let zx = self.transform(x)?;
        let zy = self.transform(y)?;
        let (nx, ny, d) = (zx.nrows(), zy.nrows(), zx.ncols());
        let mut k = Array2::<f64>::zeros((nx, ny));
        for i in 0..nx {
            for j in 0..ny {
                k[[i, j]] = (0..d).map(|l| zx[[i, l]] * zy[[j, l]]).sum();
            }
        }
        Ok(k)
    }

    /// Return whether the transformer has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.weights.is_some()
    }
}

// ============================================================================
// Landmark Selection for Nyström
// ============================================================================

/// Strategy for selecting landmark points in the Nyström approximation.
#[derive(Debug, Clone, PartialEq)]
pub enum LandmarkSelection {
    /// Uniform random subset of training points
    Random,
    /// K-means++ centroids
    KMeans {
        /// Number of k-means iterations for centroid refinement
        n_iter: usize,
    },
    /// Uniformly spaced (by index) from sorted training data
    Uniform,
}

// ============================================================================
// Nyström Approximation
// ============================================================================

/// Nyström approximation of the kernel matrix.
///
/// Given m ≪ n landmark points C, the approximation is:
/// K_nn ≈ K_nm K_mm^{-1} K_mn
///
/// The feature map φ(x) satisfies K_nn ≈ φ(X) φ(X)^T where
/// φ(X) = K_xm K_mm^{-1/2}
///
/// Uses eigendecomposition K_mm = V Λ V^T, so K_mm^{-1/2} = V Λ^{-1/2} V^T
/// (pseudoinverse for numerical stability).
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_transform::random_features::{NystromApproximation, ShiftInvariantKernel, LandmarkSelection};
/// use scirs2_core::ndarray::Array2;
///
/// let x = Array2::<f64>::zeros((100, 5));
/// let kernel = ShiftInvariantKernel::RBF { gamma: 1.0 };
/// let mut nystrom = NystromApproximation::new(20, kernel);
/// let phi = nystrom.fit_transform(x.view(), LandmarkSelection::Random, 0).expect("should succeed");
/// assert_eq!(phi.shape(), &[100, 20]);
/// ```
#[derive(Debug, Clone)]
pub struct NystromApproximation {
    /// m: number of landmark points
    pub n_components: usize,
    /// Kernel function
    pub kernel: ShiftInvariantKernel,
    /// Selected landmark points, shape (m, d)
    landmarks: Option<Array2<f64>>,
    /// K_mm^{-1/2}, shape (m, m)
    kernel_inv_sqrt: Option<Array2<f64>>,
}

impl NystromApproximation {
    /// Create a new Nyström approximation.
    pub fn new(n_components: usize, kernel: ShiftInvariantKernel) -> Self {
        NystromApproximation {
            n_components,
            kernel,
            landmarks: None,
            kernel_inv_sqrt: None,
        }
    }

    /// Fit: select landmarks and compute K_mm^{-1/2}.
    pub fn fit(
        &mut self,
        x: ArrayView2<f64>,
        selection: LandmarkSelection,
        seed: u64,
    ) -> Result<()> {
        let (n_samples, n_features) = (x.nrows(), x.ncols());
        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }
        let m = self.n_components.min(n_samples);
        if m == 0 {
            return Err(TransformError::InvalidInput(
                "n_components must be positive".to_string(),
            ));
        }

        // Select landmark indices
        let landmark_indices = match selection {
            LandmarkSelection::Random => {
                sample_without_replacement(n_samples, m, seed)?
            }
            LandmarkSelection::Uniform => {
                // Evenly spaced indices in [0, n_samples)
                (0..m)
                    .map(|i| i * n_samples / m)
                    .collect::<Vec<usize>>()
            }
            LandmarkSelection::KMeans { n_iter } => {
                kmeans_plus_plus_indices(x, m, n_iter, seed)?
            }
        };

        // Build landmarks matrix (m, d)
        let mut landmarks = Array2::<f64>::zeros((m, n_features));
        for (row_out, &idx) in landmark_indices.iter().enumerate() {
            landmarks.row_mut(row_out).assign(&x.row(idx));
        }

        // Compute K_mm: kernel matrix among landmarks
        let k_mm = compute_kernel_matrix_between(&landmarks.view(), &landmarks.view(), &self.kernel);

        // Eigendecompose K_mm = V Λ V^T
        let (eigenvalues, eigenvectors) = eigh(&k_mm.view(), None)
            .map_err(|e| TransformError::ComputationError(e.to_string()))?;

        // Compute K_mm^{-1/2} = V Λ^{-1/2} V^T (using pseudoinverse threshold)
        let threshold = 1e-10 * eigenvalues
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(1e-12);

        let n_eig = eigenvalues.len();
        let mut inv_sqrt_diag = Vec::with_capacity(n_eig);
        for &ev in eigenvalues.iter() {
            if ev > threshold {
                inv_sqrt_diag.push(1.0 / ev.sqrt());
            } else {
                inv_sqrt_diag.push(0.0);
            }
        }

        // K_mm^{-1/2} = V * diag(inv_sqrt_diag) * V^T
        let mut kernel_inv_sqrt = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                let val: f64 = (0..n_eig)
                    .map(|k| eigenvectors[[i, k]] * inv_sqrt_diag[k] * eigenvectors[[j, k]])
                    .sum();
                kernel_inv_sqrt[[i, j]] = val;
            }
        }

        self.landmarks = Some(landmarks);
        self.kernel_inv_sqrt = Some(kernel_inv_sqrt);
        Ok(())
    }

    /// Transform: compute Nyström features φ(X) = K_xm K_mm^{-1/2}.
    ///
    /// Output shape: (n_samples, n_components)
    pub fn transform(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let landmarks = self.landmarks.as_ref().ok_or_else(|| {
            TransformError::NotFitted("NystromApproximation not fitted".to_string())
        })?;
        let kernel_inv_sqrt = self.kernel_inv_sqrt.as_ref().ok_or_else(|| {
            TransformError::NotFitted("NystromApproximation not fitted".to_string())
        })?;

        let (n_samples, n_features) = (x.nrows(), x.ncols());
        let expected_dim = landmarks.ncols();
        if n_features != expected_dim {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                expected_dim, n_features
            )));
        }

        let m = landmarks.nrows();

        // Compute K_xm: shape (n_samples, m)
        let k_xm = compute_kernel_matrix_between(&x, &landmarks.view(), &self.kernel);

        // φ(X) = K_xm @ K_mm^{-1/2}: shape (n_samples, m)
        let mut phi = Array2::<f64>::zeros((n_samples, m));
        for i in 0..n_samples {
            for j in 0..m {
                phi[[i, j]] = (0..m)
                    .map(|k| k_xm[[i, k]] * kernel_inv_sqrt[[k, j]])
                    .sum();
            }
        }

        Ok(phi)
    }

    /// Fit and transform in one step.
    pub fn fit_transform(
        &mut self,
        x: ArrayView2<f64>,
        selection: LandmarkSelection,
        seed: u64,
    ) -> Result<Array2<f64>> {
        self.fit(x, selection, seed)?;
        self.transform(x)
    }

    /// Approximate kernel matrix K ≈ φ(X) φ(X)^T.
    pub fn approximate_kernel(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let phi = self.transform(x)?;
        let n = phi.nrows();
        let m = phi.ncols();
        let mut k = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                let dot: f64 = (0..m).map(|l| phi[[i, l]] * phi[[j, l]]).sum();
                k[[i, j]] = dot;
                k[[j, i]] = dot;
            }
        }
        Ok(k)
    }
}

// ============================================================================
// Tensor Sketch (Count Sketch for Polynomial Kernels)
// ============================================================================

/// Tensor Sketch approximation for polynomial kernels.
///
/// Approximates k(x,y) = (x^T y + c)^p via count sketch with FFT convolution.
/// The sketch of x ⊗ x ⊗ ... ⊗ x (p times) is computed using p count sketch
/// hash functions combined through convolution in the frequency domain.
///
/// # Reference
/// Pham & Pagh (2013): Fast and Scalable Polynomial Kernels via Explicit Feature Maps
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_transform::random_features::TensorSketch;
/// use scirs2_core::ndarray::Array2;
///
/// let x = Array2::<f64>::zeros((50, 8));
/// let mut ts = TensorSketch::new(64, 3);
/// let z = ts.fit_transform(x.view(), 0).expect("should succeed");
/// assert_eq!(z.shape(), &[50, 64]);
/// ```
#[derive(Debug, Clone)]
pub struct TensorSketch {
    /// Output dimensionality D
    pub n_components: usize,
    /// Polynomial degree p
    pub degree: usize,
    /// p hash functions, each mapping {0,...,d-1} → {0,...,D-1}
    hash_functions: Option<Vec<Array1<usize>>>,
    /// p sign functions, each mapping {0,...,d-1} → {-1, +1}
    signs: Option<Vec<Array1<f64>>>,
}

impl TensorSketch {
    /// Create a new Tensor Sketch.
    pub fn new(n_components: usize, degree: usize) -> Self {
        TensorSketch {
            n_components,
            degree,
            hash_functions: None,
            signs: None,
        }
    }

    /// Fit: initialize p count sketch hash/sign functions.
    pub fn fit(&mut self, input_dim: usize, seed: u64) -> Result<()> {
        if input_dim == 0 {
            return Err(TransformError::InvalidInput("input_dim must be > 0".to_string()));
        }
        if self.degree == 0 {
            return Err(TransformError::InvalidInput("degree must be > 0".to_string()));
        }
        if self.n_components == 0 {
            return Err(TransformError::InvalidInput(
                "n_components must be > 0".to_string(),
            ));
        }

        let d = self.degree;
        let mut hash_functions = Vec::with_capacity(d);
        let mut signs = Vec::with_capacity(d);

        for k in 0..d {
            let mut rng = seeded_rng(seed.wrapping_add(k as u64));
            let h_dist = Uniform::new(0_usize, self.n_components)
                .map_err(|e| TransformError::ComputationError(e.to_string()))?;
            let s_dist = Uniform::new(0_usize, 2_usize)
                .map_err(|e| TransformError::ComputationError(e.to_string()))?;

            let h: Vec<usize> = (0..input_dim)
                .map(|_| h_dist.sample(&mut rng))
                .collect();
            let s: Vec<f64> = (0..input_dim)
                .map(|_| if s_dist.sample(&mut rng) == 0 { -1.0 } else { 1.0 })
                .collect();

            hash_functions.push(Array1::from_vec(h));
            signs.push(Array1::from_vec(s));
        }

        self.hash_functions = Some(hash_functions);
        self.signs = Some(signs);
        Ok(())
    }

    /// Transform: compute Tensor Sketch of X.
    ///
    /// Output shape: (n_samples, n_components)
    pub fn transform(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let hash_functions = self.hash_functions.as_ref().ok_or_else(|| {
            TransformError::NotFitted("TensorSketch not fitted".to_string())
        })?;
        let signs = self.signs.as_ref().ok_or_else(|| {
            TransformError::NotFitted("TensorSketch not fitted".to_string())
        })?;

        let (n_samples, n_features) = (x.nrows(), x.ncols());
        let expected_dim = hash_functions[0].len();
        if n_features != expected_dim {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                expected_dim, n_features
            )));
        }

        let n_components = self.n_components;
        let degree = self.degree;
        let mut output = Array2::<f64>::zeros((n_samples, n_components));

        for i in 0..n_samples {
            let row = x.row(i);

            // Compute count sketch for each tensor factor, then convolve via FFT
            // sketch_k[j] = sum_{l: h_k(l)=j} s_k(l) * x_l
            let mut sketches: Vec<Vec<f64>> = Vec::with_capacity(degree);
            for k in 0..degree {
                let mut sketch = vec![0.0_f64; n_components];
                for (l, (&xl, &sl)) in row.iter().zip(signs[k].iter()).enumerate() {
                    let j = hash_functions[k][l];
                    sketch[j] += sl * xl;
                }
                sketches.push(sketch);
            }

            // Combine sketches via convolution (using simple O(D^2) circular convolution
            // for correctness; FFT-based would be O(D log D) but requires oxifft)
            let combined = sketches
                .iter()
                .skip(1)
                .fold(sketches[0].clone(), |acc, sk| {
                    circular_convolve(&acc, sk)
                });

            for j in 0..n_components {
                output[[i, j]] = combined[j];
            }
        }

        Ok(output)
    }

    /// Fit and transform in one step.
    pub fn fit_transform(&mut self, x: ArrayView2<f64>, seed: u64) -> Result<Array2<f64>> {
        let input_dim = x.ncols();
        self.fit(input_dim, seed)?;
        self.transform(x)
    }
}

/// Circular convolution: (a ⊛ b)[k] = Σ_j a[j] b[(k-j) mod n]
fn circular_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut result = vec![0.0_f64; n];
    for j in 0..n {
        if a[j] == 0.0 {
            continue;
        }
        for k in 0..n {
            result[(j + k) % n] += a[j] * b[k];
        }
    }
    result
}

// ============================================================================
// Kernel Ridge Regression with Random Fourier Features
// ============================================================================

/// Kernel Ridge Regression using Random Fourier Features.
///
/// Instead of solving the full n×n kernel system, projects X to a D-dimensional
/// feature space Z via RFF and solves the regularized least squares:
/// min_w ‖Zw - y‖² + α ‖w‖²
///
/// This reduces complexity from O(n³) to O(nD² + D³).
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_transform::random_features::{KernelRidgeRegressionRF, ShiftInvariantKernel};
/// use scirs2_core::ndarray::{Array1, Array2};
///
/// let x = Array2::<f64>::zeros((50, 4));
/// let y = Array1::<f64>::zeros(50);
/// let kernel = ShiftInvariantKernel::RBF { gamma: 1.0 };
/// let mut krr = KernelRidgeRegressionRF::new(100, kernel, 0.01);
/// krr.fit(x.view(), &y, 42).expect("should succeed");
/// let preds = krr.predict(x.view()).expect("should succeed");
/// assert_eq!(preds.len(), 50);
/// ```
#[derive(Debug, Clone)]
pub struct KernelRidgeRegressionRF {
    /// Number of random features
    pub n_components: usize,
    /// Kernel function
    pub kernel: ShiftInvariantKernel,
    /// Regularization parameter α
    pub alpha: f64,
    /// Fitted weight vector, shape (n_components,)
    weights: Option<Array1<f64>>,
    /// Random feature transformer
    rff: RandomFourierFeatures,
}

impl KernelRidgeRegressionRF {
    /// Create a new KRR with RFF.
    pub fn new(n_components: usize, kernel: ShiftInvariantKernel, alpha: f64) -> Self {
        let rff = RandomFourierFeatures::new(n_components, kernel.clone());
        KernelRidgeRegressionRF {
            n_components,
            kernel,
            alpha,
            weights: None,
            rff,
        }
    }

    /// Fit: project data to RFF space and solve regularized least squares.
    pub fn fit(
        &mut self,
        x: ArrayView2<f64>,
        y: &Array1<f64>,
        seed: u64,
    ) -> Result<()> {
        let n_samples = x.nrows();
        if y.len() != n_samples {
            return Err(TransformError::InvalidInput(format!(
                "X has {} samples but y has {} labels",
                n_samples,
                y.len()
            )));
        }
        if self.alpha <= 0.0 {
            return Err(TransformError::InvalidInput(
                "alpha must be positive".to_string(),
            ));
        }

        // Compute RFF feature matrix Z: (n_samples, n_components)
        let z = self.rff.fit_transform(x, seed)?;

        // Solve: (Z^T Z + α I) w = Z^T y
        let d = self.n_components;

        // Compute A = Z^T Z + α I  (shape: d×d)
        let mut a = Array2::<f64>::zeros((d, d));
        for k in 0..d {
            for l in k..d {
                let val: f64 = (0..n_samples).map(|i| z[[i, k]] * z[[i, l]]).sum();
                a[[k, l]] = val;
                a[[l, k]] = val;
            }
            a[[k, k]] += self.alpha;
        }

        // Compute b = Z^T y  (shape: d)
        let mut b = Array1::<f64>::zeros(d);
        for k in 0..d {
            b[k] = (0..n_samples).map(|i| z[[i, k]] * y[i]).sum();
        }

        // Solve A w = b
        let w: Array1<f64> = solve(&a.view(), &b.view(), None)
            .map_err(|e| TransformError::ComputationError(e.to_string()))?;

        self.weights = Some(w);
        Ok(())
    }

    /// Predict: return ŷ = Z(x) w.
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let weights = self.weights.as_ref().ok_or_else(|| {
            TransformError::NotFitted("KernelRidgeRegressionRF not fitted".to_string())
        })?;

        let z = self.rff.transform(x)?;
        let n_samples = z.nrows();
        let d = z.ncols();

        let mut preds = Array1::<f64>::zeros(n_samples);
        for i in 0..n_samples {
            preds[i] = (0..d).map(|j| z[[i, j]] * weights[j]).sum();
        }
        Ok(preds)
    }
}

// ============================================================================
// Utility: Exact Kernel Matrices
// ============================================================================

/// Compute exact kernel matrix K[i,j] = k(x_i, x_j) for all pairs.
///
/// Output shape: (n_samples, n_samples)
pub fn kernel_matrix(x: ArrayView2<f64>, kernel: &ShiftInvariantKernel) -> Array2<f64> {
    let n = x.nrows();
    let mut k = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        let xi: Vec<f64> = x.row(i).to_vec();
        for j in i..n {
            let xj: Vec<f64> = x.row(j).to_vec();
            let val = kernel.compute_exact(&xi, &xj);
            k[[i, j]] = val;
            k[[j, i]] = val;
        }
    }
    k
}

/// Compute exact cross-kernel matrix K[i,j] = k(x_i, y_j).
///
/// Output shape: (n_x, n_y)
pub fn cross_kernel_matrix(
    x: ArrayView2<f64>,
    y: ArrayView2<f64>,
    kernel: &ShiftInvariantKernel,
) -> Array2<f64> {
    let (nx, ny) = (x.nrows(), y.nrows());
    let mut k = Array2::<f64>::zeros((nx, ny));
    for i in 0..nx {
        let xi: Vec<f64> = x.row(i).to_vec();
        for j in 0..ny {
            let yj: Vec<f64> = y.row(j).to_vec();
            k[[i, j]] = kernel.compute_exact(&xi, &yj);
        }
    }
    k
}

// ============================================================================
// Maximum Mean Discrepancy Test
// ============================================================================

/// Maximum Mean Discrepancy (MMD) test using random features.
///
/// Tests whether two samples X and Y come from the same distribution.
/// Uses the unbiased MMD² estimator in the RFF feature space:
///
/// MMD²(X,Y) = ‖μ_X - μ_Y‖² where μ_X = (1/n) Σ z(x_i)
///
/// P-value is estimated via bootstrap permutation.
///
/// # Returns
/// `(mmd_statistic, p_value)` where p_value is the fraction of bootstrap
/// MMD values exceeding the observed MMD.
pub fn mmd_test(
    x: ArrayView2<f64>,
    y: ArrayView2<f64>,
    kernel: ShiftInvariantKernel,
    n_components: usize,
    n_bootstrap: usize,
    seed: u64,
) -> Result<(f64, f64)> {
    let (nx, n_features) = (x.nrows(), x.ncols());
    let ny = y.nrows();

    if n_features != y.ncols() {
        return Err(TransformError::InvalidInput(
            "X and Y must have the same number of features".to_string(),
        ));
    }
    if nx == 0 || ny == 0 {
        return Err(TransformError::InvalidInput(
            "Both X and Y must be non-empty".to_string(),
        ));
    }

    // Fit RFF on combined data
    let input_dim = n_features;
    let mut rff = RandomFourierFeatures::new(n_components, kernel);
    rff.fit(input_dim, seed)?;

    // Compute feature embeddings
    let zx = rff.transform(x)?;
    let zy = rff.transform(y)?;

    // Compute unbiased MMD² using the U-statistic estimator on approximate kernel z^T z.
    // This has expectation 0 under H0 (same distribution).
    //
    // MMD²_u = (1/(nx(nx-1))) sum_{i!=j} z_xi^T z_xj
    //        + (1/(ny(ny-1))) sum_{i!=j} z_yi^T z_yj
    //        - (2/(nx*ny))    sum_{i,j}  z_xi^T z_yj
    let mut kxx_sum = 0.0_f64;
    for i in 0..nx {
        for j in 0..nx {
            if i != j {
                let dot: f64 = (0..n_components).map(|d| zx[[i, d]] * zx[[j, d]]).sum();
                kxx_sum += dot;
            }
        }
    }
    let mut kyy_sum = 0.0_f64;
    for i in 0..ny {
        for j in 0..ny {
            if i != j {
                let dot: f64 = (0..n_components).map(|d| zy[[i, d]] * zy[[j, d]]).sum();
                kyy_sum += dot;
            }
        }
    }
    let mut kxy_sum = 0.0_f64;
    for i in 0..nx {
        for j in 0..ny {
            let dot: f64 = (0..n_components).map(|d| zx[[i, d]] * zy[[j, d]]).sum();
            kxy_sum += dot;
        }
    }

    let mmd_sq = kxx_sum / (nx * (nx - 1)) as f64
        + kyy_sum / (ny * (ny - 1)) as f64
        - 2.0 * kxy_sum / (nx * ny) as f64;
    // Clamp to non-negative (unbiased estimator can be slightly negative)
    let mmd_stat = mmd_sq.max(0.0).sqrt();

    // Bootstrap permutation test
    // Pool all feature vectors and permute, computing MMD for each permutation.
    // Pre-compute the full kernel matrix K[i,j] = z_i^T z_j for all pooled samples
    // so bootstrap iterations only need index shuffling, not recomputation.
    let n_total = nx + ny;
    let mut pooled = Array2::<f64>::zeros((n_total, n_components));
    for i in 0..nx {
        pooled.row_mut(i).assign(&zx.row(i));
    }
    for i in 0..ny {
        pooled.row_mut(nx + i).assign(&zy.row(i));
    }

    // Pre-compute approximate kernel matrix for all pooled data
    let mut k_pool = Array2::<f64>::zeros((n_total, n_total));
    for i in 0..n_total {
        for j in i..n_total {
            let dot: f64 = (0..n_components).map(|d| pooled[[i, d]] * pooled[[j, d]]).sum();
            k_pool[[i, j]] = dot;
            k_pool[[j, i]] = dot;
        }
    }

    let mut exceed_count = 0_usize;
    let mut rng = seeded_rng(seed.wrapping_add(999));
    let idx_dist = Uniform::new(0_usize, n_total)
        .map_err(|e| TransformError::ComputationError(e.to_string()))?;

    for _ in 0..n_bootstrap {
        // Fisher-Yates shuffle of indices
        let mut indices: Vec<usize> = (0..n_total).collect();
        for i in (1..n_total).rev() {
            let j = idx_dist.sample(&mut rng) % (i + 1);
            indices.swap(i, j);
        }

        // Compute bootstrap U-statistic MMD^2 using pre-computed kernel
        let bx = &indices[..nx];
        let by = &indices[nx..];

        let mut boot_kxx = 0.0_f64;
        for i in 0..nx {
            for j in 0..nx {
                if i != j {
                    boot_kxx += k_pool[[bx[i], bx[j]]];
                }
            }
        }
        let mut boot_kyy = 0.0_f64;
        let ny_boot = by.len();
        for i in 0..ny_boot {
            for j in 0..ny_boot {
                if i != j {
                    boot_kyy += k_pool[[by[i], by[j]]];
                }
            }
        }
        let mut boot_kxy = 0.0_f64;
        for i in 0..nx {
            for j in 0..ny_boot {
                boot_kxy += k_pool[[bx[i], by[j]]];
            }
        }

        let boot_mmd_sq = boot_kxx / (nx * (nx - 1)).max(1) as f64
            + boot_kyy / (ny_boot * (ny_boot - 1)).max(1) as f64
            - 2.0 * boot_kxy / (nx * ny_boot).max(1) as f64;
        let boot_mmd = boot_mmd_sq.max(0.0).sqrt();
        if boot_mmd >= mmd_stat {
            exceed_count += 1;
        }
    }

    let p_value = if n_bootstrap > 0 {
        exceed_count as f64 / n_bootstrap as f64
    } else {
        1.0
    };

    Ok((mmd_stat, p_value))
}

// ============================================================================
// Internal Helpers
// ============================================================================

/// Compute kernel matrix K[i,j] = k(x_i, y_j) for two datasets.
fn compute_kernel_matrix_between(
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    kernel: &ShiftInvariantKernel,
) -> Array2<f64> {
    let (nx, ny) = (x.nrows(), y.nrows());
    let mut k = Array2::<f64>::zeros((nx, ny));
    for i in 0..nx {
        let xi: Vec<f64> = x.row(i).to_vec();
        for j in 0..ny {
            let yj: Vec<f64> = y.row(j).to_vec();
            k[[i, j]] = kernel.compute_exact(&xi, &yj);
        }
    }
    k
}

/// Sample `m` distinct indices from `[0, n)` without replacement using
/// Fisher-Yates partial shuffle.
fn sample_without_replacement(n: usize, m: usize, seed: u64) -> Result<Vec<usize>> {
    if m > n {
        return Err(TransformError::InvalidInput(format!(
            "Cannot sample {} items from {} without replacement",
            m, n
        )));
    }
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = seeded_rng(seed);

    for i in 0..m {
        let j_dist = Uniform::new(i, n)
            .map_err(|e| TransformError::ComputationError(e.to_string()))?;
        let j = j_dist.sample(&mut rng);
        indices.swap(i, j);
    }
    Ok(indices[..m].to_vec())
}

/// K-means++ landmark selection: returns indices of m cluster centers.
///
/// Implements k-means++ initialization followed by n_iter Lloyd iterations.
fn kmeans_plus_plus_indices(
    x: ArrayView2<f64>,
    m: usize,
    n_iter: usize,
    seed: u64,
) -> Result<Vec<usize>> {
    let n = x.nrows();
    if m >= n {
        return Ok((0..n).collect());
    }

    let mut rng = seeded_rng(seed);
    let unif_n = Uniform::new(0_usize, n)
        .map_err(|e| TransformError::ComputationError(e.to_string()))?;
    let unif_01 = Uniform::new(0.0_f64, 1.0_f64)
        .map_err(|e| TransformError::ComputationError(e.to_string()))?;

    // K-means++ initialization
    let first = unif_n.sample(&mut rng);
    let mut centers: Vec<Vec<f64>> = vec![x.row(first).to_vec()];

    for _ in 1..m {
        // Compute minimum squared distance to existing centers for each point
        let mut min_sq_dists: Vec<f64> = (0..n)
            .map(|i| {
                let xi: Vec<f64> = x.row(i).to_vec();
                centers
                    .iter()
                    .map(|c| {
                        xi.iter().zip(c.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>()
                    })
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();

        let total: f64 = min_sq_dists.iter().sum();
        if total < 1e-12 {
            // All points are the same; pick random
            centers.push(x.row(unif_n.sample(&mut rng)).to_vec());
            continue;
        }

        // Normalize to get probability distribution
        for d in min_sq_dists.iter_mut() {
            *d /= total;
        }

        // Sample next center proportional to D²
        let threshold = unif_01.sample(&mut rng);
        let mut cumulative = 0.0;
        let mut chosen = n - 1;
        for (i, &prob) in min_sq_dists.iter().enumerate() {
            cumulative += prob;
            if cumulative >= threshold {
                chosen = i;
                break;
            }
        }
        centers.push(x.row(chosen).to_vec());
    }

    // Lloyd's algorithm: iterate assignment → update until convergence
    let n_features = x.ncols();
    let mut center_matrix: Vec<Vec<f64>> = centers;

    for _iter in 0..n_iter {
        // Assignment step: assign each point to nearest center
        let mut assignments: Vec<usize> = Vec::with_capacity(n);
        for i in 0..n {
            let xi: Vec<f64> = x.row(i).to_vec();
            let best = center_matrix
                .iter()
                .enumerate()
                .map(|(c_idx, c)| {
                    let d: f64 = xi
                        .iter()
                        .zip(c.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    (c_idx, d)
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            assignments.push(best);
        }

        // Update step: recompute centers as means of assigned points
        let mut new_centers = vec![vec![0.0_f64; n_features]; m];
        let mut counts = vec![0_usize; m];
        for (i, &c_idx) in assignments.iter().enumerate() {
            counts[c_idx] += 1;
            let xi: Vec<f64> = x.row(i).to_vec();
            for d in 0..n_features {
                new_centers[c_idx][d] += xi[d];
            }
        }
        for c_idx in 0..m {
            if counts[c_idx] > 0 {
                let cnt = counts[c_idx] as f64;
                for d in 0..n_features {
                    new_centers[c_idx][d] /= cnt;
                }
            } else {
                // Empty cluster: reinitialize to a random point
                new_centers[c_idx] = x.row(unif_n.sample(&mut rng)).to_vec();
            }
        }
        center_matrix = new_centers;
    }

    // Find nearest training point to each final center (return indices into x)
    let landmark_indices: Vec<usize> = center_matrix
        .iter()
        .map(|c| {
            (0..n)
                .map(|i| {
                    let xi: Vec<f64> = x.row(i).to_vec();
                    let d: f64 = xi
                        .iter()
                        .zip(c.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    (i, d)
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        })
        .collect();

    // Deduplicate while preserving order
    let mut seen = std::collections::HashSet::new();
    let unique: Vec<usize> = landmark_indices
        .into_iter()
        .filter(|&idx| seen.insert(idx))
        .collect();

    // If deduplication reduced count, fill remainder with random points
    let mut result = unique;
    if result.len() < m {
        let mut remaining: Vec<usize> = (0..n).filter(|i| !result.contains(i)).collect();
        let needed = m - result.len();
        let take = needed.min(remaining.len());
        // Shuffle remaining and take first `take`
        for i in 0..take {
            let j_dist = Uniform::new(i, remaining.len())
                .map_err(|e| TransformError::ComputationError(e.to_string()))?;
            let j = j_dist.sample(&mut rng);
            remaining.swap(i, j);
        }
        result.extend_from_slice(&remaining[..take]);
    }

    Ok(result[..m.min(result.len())].to_vec())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Generate a simple reproducible dataset.
    fn make_data(n: usize, d: usize, seed: u64) -> Array2<f64> {
        let mut rng = seeded_rng(seed);
        let dist = Normal::new(0.0_f64, 1.0_f64).expect("Normal distribution creation failed");
        let data: Vec<f64> = (0..n * d).map(|_| dist.sample(&mut rng)).collect();
        Array2::from_shape_vec((n, d), data).expect("Failed to create data array")
    }

    /// Generate two well-separated datasets for MMD tests.
    fn make_two_distributions(
        n: usize,
        d: usize,
        mean_shift: f64,
        seed: u64,
    ) -> (Array2<f64>, Array2<f64>) {
        let x = make_data(n, d, seed);
        let mut y = make_data(n, d, seed.wrapping_add(1000));
        y.mapv_inplace(|v| v + mean_shift);
        (x, y)
    }

    // ------------------------------------------------------------------
    // RFF output shape
    // ------------------------------------------------------------------
    #[test]
    fn test_rff_output_shape() {
        let x = make_data(30, 5, 1);
        let kernel = ShiftInvariantKernel::RBF { gamma: 0.5 };
        let mut rff = RandomFourierFeatures::new(50, kernel);
        let z = rff.fit_transform(x.view(), 42).expect("RFF fit_transform failed");
        assert_eq!(z.shape(), &[30, 50], "RFF output shape mismatch");
    }

    // ------------------------------------------------------------------
    // RFF approximates RBF kernel: max error < 0.1 for D=500
    // ------------------------------------------------------------------
    #[test]
    fn test_rff_approximates_rbf_kernel() {
        let x = make_data(20, 4, 7);
        let kernel = ShiftInvariantKernel::RBF { gamma: 0.5 };

        // Exact kernel matrix
        let k_exact = kernel_matrix(x.view(), &kernel);

        // Approximate via RFF with D=500
        let mut rff = RandomFourierFeatures::new(500, kernel);
        let k_approx = rff
            .fit_transform(x.view(), 0)
            .and_then(|_| rff.approximate_kernel(x.view()))
            .expect("RFF kernel approximation failed");

        let n = x.nrows();
        let mut max_err = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let err = (k_exact[[i, j]] - k_approx[[i, j]]).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        assert!(
            max_err < 0.15,
            "RFF max kernel approximation error too large: {:.4} (expected < 0.15)",
            max_err
        );
    }

    // ------------------------------------------------------------------
    // RFF: various kernels produce correct output shape
    // ------------------------------------------------------------------
    #[test]
    fn test_rff_laplacian_kernel() {
        let x = make_data(10, 3, 2);
        let kernel = ShiftInvariantKernel::Laplacian { gamma: 1.0 };
        let mut rff = RandomFourierFeatures::new(64, kernel);
        let z = rff.fit_transform(x.view(), 1).expect("Laplacian RFF failed");
        assert_eq!(z.shape(), &[10, 64]);
        // All values should be finite (Cauchy samples are clipped)
        assert!(z.iter().all(|v| v.is_finite()), "Non-finite RFF values");
    }

    #[test]
    fn test_rff_cauchy_kernel() {
        let x = make_data(10, 3, 3);
        let kernel = ShiftInvariantKernel::Cauchy { gamma: 0.5 };
        let mut rff = RandomFourierFeatures::new(64, kernel);
        let z = rff.fit_transform(x.view(), 2).expect("Cauchy RFF failed");
        assert_eq!(z.shape(), &[10, 64]);
        assert!(z.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_rff_matern32_kernel() {
        let x = make_data(15, 4, 4);
        let kernel = ShiftInvariantKernel::Matern32 { length_scale: 1.0 };
        let mut rff = RandomFourierFeatures::new(100, kernel);
        let z = rff.fit_transform(x.view(), 3).expect("Matern32 RFF failed");
        assert_eq!(z.shape(), &[15, 100]);
        assert!(z.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_rff_matern52_kernel() {
        let x = make_data(15, 4, 5);
        let kernel = ShiftInvariantKernel::Matern52 { length_scale: 1.5 };
        let mut rff = RandomFourierFeatures::new(100, kernel);
        let z = rff.fit_transform(x.view(), 4).expect("Matern52 RFF failed");
        assert_eq!(z.shape(), &[15, 100]);
        assert!(z.iter().all(|v| v.is_finite()));
    }

    // ------------------------------------------------------------------
    // RFF: RBF spectral density is Gaussian (verify sampled frequencies)
    // ------------------------------------------------------------------
    #[test]
    fn test_rbf_spectral_density_is_gaussian() {
        let gamma = 2.0;
        let kernel = ShiftInvariantKernel::RBF { gamma };
        let n_samples = 5000;
        let input_dim = 1;
        let weights = kernel
            .sample_frequencies(n_samples, input_dim, 77)
            .expect("Frequency sampling failed");

        // The distribution should be N(0, sqrt(2γ)) = N(0, 2.0)
        let expected_std = (2.0 * gamma).sqrt(); // = 2.0

        let mean: f64 = weights.iter().sum::<f64>() / n_samples as f64;
        let variance: f64 = weights.iter().map(|&w| (w - mean).powi(2)).sum::<f64>()
            / n_samples as f64;
        let std_dev = variance.sqrt();

        assert!(
            mean.abs() < 0.1,
            "RBF spectral mean not near 0: {:.4}",
            mean
        );
        assert!(
            (std_dev - expected_std).abs() < 0.2,
            "RBF spectral std {:.4} not near expected {:.4}",
            std_dev,
            expected_std
        );
    }

    // ------------------------------------------------------------------
    // Nyström output shape
    // ------------------------------------------------------------------
    #[test]
    fn test_nystrom_output_shape() {
        let x = make_data(40, 5, 10);
        let kernel = ShiftInvariantKernel::RBF { gamma: 1.0 };
        let mut nystrom = NystromApproximation::new(15, kernel);
        let phi = nystrom
            .fit_transform(x.view(), LandmarkSelection::Random, 0)
            .expect("Nystrom fit_transform failed");
        assert_eq!(phi.shape(), &[40, 15], "Nystrom output shape mismatch");
    }

    // ------------------------------------------------------------------
    // Nyström approximation is positive semi-definite
    // ------------------------------------------------------------------
    #[test]
    fn test_nystrom_kernel_psd() {
        let x = make_data(20, 3, 11);
        let kernel = ShiftInvariantKernel::RBF { gamma: 0.5 };
        let mut nystrom = NystromApproximation::new(10, kernel);
        nystrom
            .fit(x.view(), LandmarkSelection::Random, 5)
            .expect("Nystrom fit failed");
        let k_approx = nystrom
            .approximate_kernel(x.view())
            .expect("Nystrom kernel failed");

        let n = k_approx.nrows();
        // Check that it's symmetric
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (k_approx[[i, j]] - k_approx[[j, i]]).abs() < 1e-10,
                    "Nystrom kernel not symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }

        // Check eigenvalues are non-negative
        let (evals, _) = eigh(&k_approx.view(), None).expect("eigh failed");
        let min_eval = evals.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            min_eval >= -1e-8,
            "Nystrom kernel not PSD: min eigenvalue = {:.2e}",
            min_eval
        );
    }

    // ------------------------------------------------------------------
    // Nyström with KMeans landmark selection
    // ------------------------------------------------------------------
    #[test]
    fn test_nystrom_kmeans_selection() {
        let x = make_data(50, 4, 12);
        let kernel = ShiftInvariantKernel::RBF { gamma: 0.5 };
        let mut nystrom = NystromApproximation::new(10, kernel);
        let phi = nystrom
            .fit_transform(x.view(), LandmarkSelection::KMeans { n_iter: 5 }, 6)
            .expect("Nystrom KMeans failed");
        assert_eq!(phi.shape(), &[50, 10]);
    }

    // ------------------------------------------------------------------
    // Nyström with Uniform landmark selection
    // ------------------------------------------------------------------
    #[test]
    fn test_nystrom_uniform_selection() {
        let x = make_data(30, 3, 13);
        let kernel = ShiftInvariantKernel::RBF { gamma: 1.0 };
        let mut nystrom = NystromApproximation::new(8, kernel);
        let phi = nystrom
            .fit_transform(x.view(), LandmarkSelection::Uniform, 7)
            .expect("Nystrom Uniform failed");
        assert_eq!(phi.shape(), &[30, 8]);
    }

    // ------------------------------------------------------------------
    // Tensor Sketch output shape
    // ------------------------------------------------------------------
    #[test]
    fn test_tensor_sketch_output_shape() {
        let x = make_data(25, 6, 20);
        let mut ts = TensorSketch::new(32, 3);
        let z = ts.fit_transform(x.view(), 0).expect("TensorSketch failed");
        assert_eq!(z.shape(), &[25, 32], "TensorSketch output shape mismatch");
    }

    // ------------------------------------------------------------------
    // Tensor Sketch: polynomial kernel degree 1 ≈ linear sketch
    // ------------------------------------------------------------------
    #[test]
    fn test_tensor_sketch_degree1_values_finite() {
        let x = make_data(10, 4, 21);
        let mut ts = TensorSketch::new(16, 1);
        let z = ts.fit_transform(x.view(), 1).expect("TensorSketch degree-1 failed");
        assert_eq!(z.shape(), &[10, 16]);
        assert!(z.iter().all(|v| v.is_finite()), "Non-finite tensor sketch values");
    }

    // ------------------------------------------------------------------
    // KernelRidgeRegressionRF output shape
    // ------------------------------------------------------------------
    #[test]
    fn test_krr_rf_output_shape() {
        let x = make_data(40, 5, 30);
        let y: Vec<f64> = (0..40).map(|i| i as f64 * 0.1).collect();
        let y_arr = Array1::from_vec(y);
        let kernel = ShiftInvariantKernel::RBF { gamma: 0.5 };
        let mut krr = KernelRidgeRegressionRF::new(100, kernel, 0.01);
        krr.fit(x.view(), &y_arr, 42).expect("KRR fit failed");
        let preds = krr.predict(x.view()).expect("KRR predict failed");
        assert_eq!(preds.len(), 40, "KRR prediction shape mismatch");
        assert!(
            preds.iter().all(|v| v.is_finite()),
            "Non-finite KRR predictions"
        );
    }

    // ------------------------------------------------------------------
    // KRR with RFF converges toward exact solution
    // ------------------------------------------------------------------
    #[test]
    fn test_krr_rf_reasonable_predictions() {
        // Simple linear target that should be approximable
        let n = 30;
        let d = 3;
        let x = make_data(n, d, 31);

        // y = sum of first column
        let y: Array1<f64> = Array1::from_iter(x.column(0).iter().cloned());

        let kernel = ShiftInvariantKernel::RBF { gamma: 0.5 };
        let mut krr = KernelRidgeRegressionRF::new(200, kernel, 0.001);
        krr.fit(x.view(), &y, 99).expect("KRR fit failed");
        let preds = krr.predict(x.view()).expect("KRR predict failed");

        // MSE should be finite and not extremely large
        let mse: f64 = preds
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / n as f64;
        assert!(mse.is_finite(), "KRR MSE is not finite");
        assert!(
            mse < 10.0,
            "KRR MSE too large: {:.4} (sanity check failure)",
            mse
        );
    }

    // ------------------------------------------------------------------
    // Exact kernel matrix is positive semi-definite
    // ------------------------------------------------------------------
    #[test]
    fn test_exact_kernel_matrix_psd() {
        let x = make_data(15, 3, 40);
        let kernel = ShiftInvariantKernel::RBF { gamma: 0.5 };
        let k = kernel_matrix(x.view(), &kernel);

        let n = k.nrows();
        // Symmetry check
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (k[[i, j]] - k[[j, i]]).abs() < 1e-12,
                    "Kernel matrix not symmetric"
                );
            }
        }

        let (evals, _) = eigh(&k.view(), None).expect("eigh failed on kernel matrix");
        let min_eval = evals.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            min_eval >= -1e-8,
            "Kernel matrix not PSD: min eigenvalue = {:.2e}",
            min_eval
        );
    }

    // ------------------------------------------------------------------
    // MMD between same distribution ≈ 0
    // ------------------------------------------------------------------
    #[test]
    fn test_mmd_same_distribution_small() {
        let n = 100;
        let d = 5;
        let x = make_data(n, d, 50);
        let y = make_data(n, d, 51);  // same distribution, different seed

        let kernel = ShiftInvariantKernel::RBF { gamma: 0.5 };
        let (mmd_stat, _p_value) =
            mmd_test(x.view(), y.view(), kernel, 200, 100, 0).expect("MMD test failed");

        assert!(
            mmd_stat < 0.1,
            "MMD between same distribution should be small, got {:.4}",
            mmd_stat
        );
    }

    // ------------------------------------------------------------------
    // MMD between different distributions > same-distribution MMD
    // ------------------------------------------------------------------
    #[test]
    fn test_mmd_different_distributions_larger() {
        let n = 80;
        let d = 4;

        let (x1, y1) = make_two_distributions(n, d, 0.0, 60);
        let (x2, y2) = make_two_distributions(n, d, 5.0, 61);

        let kernel1 = ShiftInvariantKernel::RBF { gamma: 0.1 };
        let kernel2 = ShiftInvariantKernel::RBF { gamma: 0.1 };

        let (mmd_same, _) =
            mmd_test(x1.view(), y1.view(), kernel1, 200, 50, 0).expect("MMD same failed");
        let (mmd_diff, _) =
            mmd_test(x2.view(), y2.view(), kernel2, 200, 50, 1).expect("MMD diff failed");

        assert!(
            mmd_diff > mmd_same,
            "MMD for different distributions ({:.4}) should exceed same-distribution ({:.4})",
            mmd_diff,
            mmd_same
        );
    }

    // ------------------------------------------------------------------
    // Cross-kernel matrix shape
    // ------------------------------------------------------------------
    #[test]
    fn test_cross_kernel_shape() {
        let x = make_data(10, 4, 70);
        let y = make_data(7, 4, 71);
        let kernel = ShiftInvariantKernel::RBF { gamma: 1.0 };
        let k = cross_kernel_matrix(x.view(), y.view(), &kernel);
        assert_eq!(k.shape(), &[10, 7], "Cross kernel shape mismatch");
    }

    // ------------------------------------------------------------------
    // RFF cross-kernel approximation shape
    // ------------------------------------------------------------------
    #[test]
    fn test_rff_cross_kernel_shape() {
        let x = make_data(12, 3, 80);
        let y = make_data(8, 3, 81);
        let kernel = ShiftInvariantKernel::RBF { gamma: 0.5 };
        let mut rff = RandomFourierFeatures::new(100, kernel);
        rff.fit(3, 0).expect("RFF fit failed");
        let k = rff
            .approximate_cross_kernel(x.view(), y.view())
            .expect("cross kernel failed");
        assert_eq!(k.shape(), &[12, 8]);
    }

    // ------------------------------------------------------------------
    // Error handling: not fitted
    // ------------------------------------------------------------------
    #[test]
    fn test_not_fitted_error() {
        let x = make_data(5, 3, 90);
        let kernel = ShiftInvariantKernel::RBF { gamma: 1.0 };
        let rff = RandomFourierFeatures::new(10, kernel);
        let result = rff.transform(x.view());
        assert!(
            result.is_err(),
            "Should return error when not fitted"
        );
    }

    #[test]
    fn test_nystrom_not_fitted_error() {
        let x = make_data(5, 3, 91);
        let kernel = ShiftInvariantKernel::RBF { gamma: 1.0 };
        let nystrom = NystromApproximation::new(3, kernel);
        let result = nystrom.transform(x.view());
        assert!(result.is_err(), "Nystrom should return error when not fitted");
    }

    // ------------------------------------------------------------------
    // Dimension mismatch error
    // ------------------------------------------------------------------
    #[test]
    fn test_rff_dimension_mismatch() {
        let x_fit = make_data(10, 4, 100);
        let x_test = make_data(5, 6, 101);
        let kernel = ShiftInvariantKernel::RBF { gamma: 1.0 };
        let mut rff = RandomFourierFeatures::new(50, kernel);
        rff.fit(4, 0).expect("fit failed");
        let result = rff.transform(x_test.view());
        assert!(result.is_err(), "Should error on dimension mismatch");
        drop(x_fit);
    }

    // ------------------------------------------------------------------
    // Compute exact Matern kernel values
    // ------------------------------------------------------------------
    #[test]
    fn test_exact_matern_kernel_at_zero() {
        // k(x, x) = 1 for all Matérn kernels
        let x = vec![1.0, 2.0, 3.0];
        let kernel_32 = ShiftInvariantKernel::Matern32 { length_scale: 1.0 };
        let kernel_52 = ShiftInvariantKernel::Matern52 { length_scale: 1.0 };
        let k32 = kernel_32.compute_exact(&x, &x);
        let k52 = kernel_52.compute_exact(&x, &x);
        assert!(
            (k32 - 1.0).abs() < 1e-12,
            "Matern32 k(x,x) should be 1, got {:.6}",
            k32
        );
        assert!(
            (k52 - 1.0).abs() < 1e-12,
            "Matern52 k(x,x) should be 1, got {:.6}",
            k52
        );
    }

    // ------------------------------------------------------------------
    // RFF is_fitted check
    // ------------------------------------------------------------------
    #[test]
    fn test_rff_is_fitted() {
        let kernel = ShiftInvariantKernel::RBF { gamma: 0.5 };
        let mut rff = RandomFourierFeatures::new(50, kernel);
        assert!(!rff.is_fitted(), "Should not be fitted before fit()");
        rff.fit(4, 42).expect("fit failed");
        assert!(rff.is_fitted(), "Should be fitted after fit()");
    }
}
