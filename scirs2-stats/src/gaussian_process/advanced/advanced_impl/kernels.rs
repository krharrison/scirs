//! Advanced GP kernels: NTK, Arc-Cosine, Spectral Mixture, Additive, ARD.

use std::f64::consts::PI;

// ============================================================
// AdvancedKernel trait
// ============================================================

/// Generic kernel trait operating on row-slices `&[f64]`.
pub trait AdvancedKernel: Clone + Send + Sync + std::fmt::Debug {
    /// Evaluate k(x₁, x₂).
    fn call(&self, x1: &[f64], x2: &[f64]) -> f64;

    /// Build the full covariance matrix K(A, B).
    fn matrix(
        &self,
        a: &scirs2_core::ndarray::Array2<f64>,
        b: &scirs2_core::ndarray::Array2<f64>,
    ) -> scirs2_core::ndarray::Array2<f64> {
        use scirs2_core::ndarray::Array2;
        let na = a.nrows();
        let nb = b.nrows();
        let mut k = Array2::<f64>::zeros((na, nb));
        for i in 0..na {
            let xi: Vec<f64> = a.row(i).iter().copied().collect();
            for j in 0..nb {
                let xj: Vec<f64> = b.row(j).iter().copied().collect();
                k[[i, j]] = self.call(&xi, &xj);
            }
        }
        k
    }

    /// Number of hyperparameters (log-space).
    fn n_params(&self) -> usize;

    /// Get log-space hyperparameters.
    fn get_log_params(&self) -> Vec<f64>;

    /// Set from log-space hyperparameters.
    fn set_log_params(&mut self, params: &[f64]);
}

// ============================================================
// Neural Tangent Kernel
// ============================================================

/// Neural Tangent Kernel (NTK) for an infinite-width MLP.
///
/// The arc-cosine kernel of degree 1 is used as a building block, then
/// composed iteratively for depth L:
///
/// Σ⁽⁰⁾(x, x') = σ_w² / d · xᵀx' + σ_b²
/// K_ntk⁽ˡ⁾ = K_ntk⁽ˡ⁻¹⁾ · σ_w² J(θ) + Σ⁽ˡ⁾
///
/// where θ = arccos(Σ⁽ˡ⁻¹⁾ / (‖x‖ ‖x'‖)) and J(θ) = (sin θ + (π−θ) cos θ) / π.
#[derive(Debug, Clone)]
pub struct NeuralTangentKernel {
    /// Network depth (number of weight layers).
    pub depth: usize,
    /// Variance of weights (σ_w²).
    pub weight_variance: f64,
    /// Variance of biases (σ_b²).
    pub bias_variance: f64,
}

impl NeuralTangentKernel {
    /// Create a new NTK.
    pub fn new(depth: usize, weight_variance: f64, bias_variance: f64) -> Self {
        Self {
            depth,
            weight_variance,
            bias_variance,
        }
    }
}

impl AdvancedKernel for NeuralTangentKernel {
    fn call(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let w = self.weight_variance;
        let b = self.bias_variance;
        let d = x1.len() as f64;
        let dot: f64 = x1.iter().zip(x2).map(|(&a, &c)| a * c).sum();
        let sigma0 = w / d * dot + b;
        let mut sigma = sigma0;
        let mut k_ntk = sigma0;
        let n1: f64 = x1.iter().map(|&a| a * a).sum::<f64>().sqrt();
        let n2: f64 = x2.iter().map(|&a| a * a).sum::<f64>().sqrt();
        for _layer in 1..self.depth {
            let denom = (n1 * n2).max(1e-14);
            let cos_theta = (sigma / denom).clamp(-1.0, 1.0);
            let theta = cos_theta.acos();
            let j = (theta.sin() + (PI - theta) * theta.cos()) / PI;
            let sigma_new = w * sigma * j + b;
            k_ntk = k_ntk * (w * j) + sigma_new;
            sigma = sigma_new;
        }
        k_ntk
    }

    fn n_params(&self) -> usize { 2 }

    fn get_log_params(&self) -> Vec<f64> {
        vec![self.weight_variance.ln(), self.bias_variance.ln()]
    }

    fn set_log_params(&mut self, params: &[f64]) {
        if params.len() >= 2 {
            self.weight_variance = params[0].exp();
            self.bias_variance = params[1].exp();
        }
    }
}

// ============================================================
// Arc-Cosine Kernel
// ============================================================

/// Arc-cosine kernel of degree `n` (Cho & Saul 2009).
///
/// k_n(x, x') = (1/π) ‖x‖ⁿ ‖x'‖ⁿ J_n(θ)
#[derive(Debug, Clone)]
pub struct ArcCosineKernel {
    /// Degree (0, 1, or 2).
    pub degree: u32,
    /// Output variance scale.
    pub variance: f64,
}

impl ArcCosineKernel {
    /// Create a new arc-cosine kernel.
    pub fn new(degree: u32, variance: f64) -> Self {
        Self { degree, variance }
    }

    fn j_function(&self, theta: f64) -> f64 {
        match self.degree {
            0 => PI - theta,
            1 => theta.sin() + (PI - theta) * theta.cos(),
            2 => {
                let s = theta.sin();
                3.0 * s * theta.cos() + (PI - theta) * (1.0 + 2.0 * theta.cos().powi(2))
            }
            n => {
                let jnm2 = ArcCosineKernel::new(n - 2, 1.0).j_function(theta);
                let jnm1 = ArcCosineKernel::new(n - 1, 1.0).j_function(theta);
                (n - 1) as f64 * theta.cos() * jnm1 - (n - 2) as f64 * jnm2
            }
        }
    }
}

impl AdvancedKernel for ArcCosineKernel {
    fn call(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let dot: f64 = x1.iter().zip(x2).map(|(&a, &c)| a * c).sum();
        let n1: f64 = x1.iter().map(|&a| a * a).sum::<f64>().sqrt();
        let n2: f64 = x2.iter().map(|&a| a * a).sum::<f64>().sqrt();
        let norm_prod = n1 * n2;
        let cos_theta = if norm_prod < 1e-14 { 1.0 } else { (dot / norm_prod).clamp(-1.0, 1.0) };
        let theta = cos_theta.acos();
        let n = self.degree as f64;
        self.variance / PI * norm_prod.powf(n) * self.j_function(theta)
    }

    fn n_params(&self) -> usize { 1 }

    fn get_log_params(&self) -> Vec<f64> { vec![self.variance.ln()] }

    fn set_log_params(&mut self, params: &[f64]) {
        if !params.is_empty() { self.variance = params[0].exp(); }
    }
}

// ============================================================
// Spectral Mixture Kernel
// ============================================================

/// One component of the spectral mixture kernel.
#[derive(Debug, Clone)]
pub struct SpectralMixtureComponent {
    /// Weight (mixture proportion).
    pub weight: f64,
    /// Mean frequency vector (length = input dimension).
    pub mean: Vec<f64>,
    /// Variance vector (per dimension, in frequency space).
    pub variance: Vec<f64>,
}

/// Spectral Mixture Kernel (Wilson & Adams 2013).
///
/// k(τ) = Σ_q w_q · Π_d cos(2π τ_d μ_{q,d}) · exp(−2π² τ_d² v_{q,d})
#[derive(Debug, Clone)]
pub struct SpectralMixtureKernel {
    /// Mixture components.
    pub components: Vec<SpectralMixtureComponent>,
}

impl SpectralMixtureKernel {
    /// Create a new spectral mixture kernel.
    pub fn new(components: Vec<SpectralMixtureComponent>) -> Self {
        Self { components }
    }

    /// Initialize with `q` components for `d`-dimensional input.
    pub fn random_init(q: usize, d: usize, rng_seed: u64) -> Self {
        let mut state = rng_seed.wrapping_add(0xdeadbeef);
        let mut rand_pos = || -> f64 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let components = (0..q)
            .map(|_| SpectralMixtureComponent {
                weight: 1.0 / q as f64,
                mean: (0..d).map(|_| rand_pos()).collect(),
                variance: (0..d).map(|_| (rand_pos() + 0.1).max(0.1)).collect(),
            })
            .collect();
        Self { components }
    }
}

impl AdvancedKernel for SpectralMixtureKernel {
    fn call(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let tau: Vec<f64> = x1.iter().zip(x2).map(|(&a, &b)| a - b).collect();
        self.components.iter().map(|c| {
            let cos_part: f64 = tau.iter().zip(c.mean.iter())
                .map(|(&t, &mu)| (2.0 * PI * t * mu).cos())
                .product();
            let exp_part: f64 = tau.iter().zip(c.variance.iter())
                .map(|(&t, &v)| -2.0 * PI * PI * t * t * v)
                .sum::<f64>()
                .exp();
            c.weight * cos_part * exp_part
        }).sum()
    }

    fn n_params(&self) -> usize {
        self.components.iter().map(|c| 1 + 2 * c.mean.len()).sum()
    }

    fn get_log_params(&self) -> Vec<f64> {
        let mut p = Vec::new();
        for c in &self.components {
            p.push(c.weight.ln());
            p.extend(c.mean.iter().copied());
            p.extend(c.variance.iter().map(|&v| v.ln()));
        }
        p
    }

    fn set_log_params(&mut self, params: &[f64]) {
        let d = if self.components.is_empty() { return; } else { self.components[0].mean.len() };
        let stride = 1 + 2 * d;
        for (i, c) in self.components.iter_mut().enumerate() {
            let base = i * stride;
            if base + stride > params.len() { break; }
            c.weight = params[base].exp();
            c.mean = params[(base + 1)..(base + 1 + d)].to_vec();
            c.variance = params[(base + 1 + d)..(base + 1 + 2 * d)].iter().map(|&v| v.exp()).collect();
        }
    }
}

// ============================================================
// Additive Kernel
// ============================================================

/// Additive kernel: k(x, x') = Σ_d k_d(x_d, x'_d).
#[derive(Debug, Clone)]
pub struct AdditiveKernel {
    /// Per-dimension length scales.
    pub length_scales: Vec<f64>,
    /// Global output variance.
    pub variance: f64,
}

impl AdditiveKernel {
    /// Create a new additive RBF-based kernel.
    pub fn new(length_scales: Vec<f64>, variance: f64) -> Self {
        Self { length_scales, variance }
    }

    /// Isotropic version for `d` dimensions.
    pub fn isotropic(d: usize, length_scale: f64, variance: f64) -> Self {
        Self { length_scales: vec![length_scale; d], variance }
    }
}

impl AdvancedKernel for AdditiveKernel {
    fn call(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let d = self.length_scales.len().min(x1.len()).min(x2.len());
        let sum: f64 = (0..d).map(|i| {
            let diff = x1[i] - x2[i];
            let ls = self.length_scales[i];
            (-0.5 * diff * diff / (ls * ls)).exp()
        }).sum();
        self.variance * sum / d as f64
    }

    fn n_params(&self) -> usize { self.length_scales.len() + 1 }

    fn get_log_params(&self) -> Vec<f64> {
        let mut p: Vec<f64> = self.length_scales.iter().map(|&l| l.ln()).collect();
        p.push(self.variance.ln());
        p
    }

    fn set_log_params(&mut self, params: &[f64]) {
        let d = self.length_scales.len();
        for i in 0..d.min(params.len()) { self.length_scales[i] = params[i].exp(); }
        if params.len() > d { self.variance = params[d].exp(); }
    }
}

// ============================================================
// ARD Kernel
// ============================================================

/// Squared-Exponential kernel with Automatic Relevance Determination (ARD).
///
/// k(x, x') = σ² exp(−½ Σ_d (x_d − x'_d)² / ℓ_d²)
#[derive(Debug, Clone)]
pub struct ARDKernel {
    /// Per-dimension length scales.
    pub length_scales: Vec<f64>,
    /// Output variance.
    pub variance: f64,
}

impl ARDKernel {
    /// Create an ARD kernel.
    pub fn new(length_scales: Vec<f64>, variance: f64) -> Self {
        Self { length_scales, variance }
    }

    /// Isotropic ARD kernel for `d` dimensions.
    pub fn isotropic(d: usize, length_scale: f64, variance: f64) -> Self {
        Self::new(vec![length_scale; d], variance)
    }
}

impl AdvancedKernel for ARDKernel {
    fn call(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let d = self.length_scales.len().min(x1.len()).min(x2.len());
        let sq: f64 = (0..d).map(|i| {
            let diff = x1[i] - x2[i];
            let ls = self.length_scales[i];
            diff * diff / (ls * ls)
        }).sum();
        self.variance * (-0.5 * sq).exp()
    }

    fn n_params(&self) -> usize { self.length_scales.len() + 1 }

    fn get_log_params(&self) -> Vec<f64> {
        let mut p: Vec<f64> = self.length_scales.iter().map(|&l| l.ln()).collect();
        p.push(self.variance.ln());
        p
    }

    fn set_log_params(&mut self, params: &[f64]) {
        let d = self.length_scales.len();
        for i in 0..d.min(params.len()) { self.length_scales[i] = params[i].exp(); }
        if params.len() > d { self.variance = params[d].exp(); }
    }
}
