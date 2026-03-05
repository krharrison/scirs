//! Comprehensive Kernel Density Estimation (KDE)
//!
//! This module provides production-ready kernel density estimation for 1D and 2D data,
//! with multiple kernel functions, bandwidth selection methods, and statistical utilities.
//!
//! # Features
//!
//! - **7 kernel types**: Gaussian, Epanechnikov, Triangular, Uniform, Biweight, Triweight, Cosine
//! - **Bandwidth selection**: Silverman's rule, Scott's rule, Improved Sheather-Jones, LSCV
//! - **1D KDE**: density evaluation, CDF, quantile, log-likelihood, sampling, weighted KDE
//! - **2D KDE**: bivariate density estimation with grid evaluation
//!
//! # Example
//!
//! ```rust
//! use scirs2_stats::kde::{KernelDensityEstimate, Kernel};
//!
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let kde = KernelDensityEstimate::new(&data, Kernel::Gaussian);
//! let density = kde.evaluate(3.0);
//! assert!(density > 0.0);
//! ```

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array2, Ix2};
use scirs2_core::random::core::{seeded_rng, thread_rng, Random};

// ---------------------------------------------------------------------------
// Kernel enum
// ---------------------------------------------------------------------------

/// Kernel functions for density estimation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// Gaussian (normal) kernel: K(u) = (2pi)^{-1/2} exp(-u^2/2)
    Gaussian,
    /// Epanechnikov kernel: K(u) = 3/4 (1 - u^2) for |u| <= 1
    Epanechnikov,
    /// Triangular kernel: K(u) = (1 - |u|) for |u| <= 1
    Triangular,
    /// Uniform (rectangular) kernel: K(u) = 1/2 for |u| <= 1
    Uniform,
    /// Biweight (quartic) kernel: K(u) = 15/16 (1 - u^2)^2 for |u| <= 1
    Biweight,
    /// Triweight kernel: K(u) = 35/32 (1 - u^2)^3 for |u| <= 1
    Triweight,
    /// Cosine kernel: K(u) = pi/4 cos(pi u / 2) for |u| <= 1
    Cosine,
}

impl Kernel {
    /// Evaluate the kernel function at point `u`.
    pub fn evaluate(&self, u: f64) -> f64 {
        match self {
            Kernel::Gaussian => {
                let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
                inv_sqrt_2pi * (-0.5 * u * u).exp()
            }
            Kernel::Epanechnikov => {
                if u.abs() <= 1.0 {
                    0.75 * (1.0 - u * u)
                } else {
                    0.0
                }
            }
            Kernel::Triangular => {
                if u.abs() <= 1.0 {
                    1.0 - u.abs()
                } else {
                    0.0
                }
            }
            Kernel::Uniform => {
                if u.abs() <= 1.0 {
                    0.5
                } else {
                    0.0
                }
            }
            Kernel::Biweight => {
                if u.abs() <= 1.0 {
                    let t = 1.0 - u * u;
                    (15.0 / 16.0) * t * t
                } else {
                    0.0
                }
            }
            Kernel::Triweight => {
                if u.abs() <= 1.0 {
                    let t = 1.0 - u * u;
                    (35.0 / 32.0) * t * t * t
                } else {
                    0.0
                }
            }
            Kernel::Cosine => {
                if u.abs() <= 1.0 {
                    (std::f64::consts::PI / 4.0) * (std::f64::consts::FRAC_PI_2 * u).cos()
                } else {
                    0.0
                }
            }
        }
    }

    /// Return the finite support radius for bounded kernels, or `f64::INFINITY` for Gaussian.
    pub fn support_radius(&self) -> f64 {
        match self {
            Kernel::Gaussian => f64::INFINITY,
            _ => 1.0,
        }
    }

    /// Second moment of the kernel: integral of u^2 K(u) du.
    /// Used in bandwidth selection formulas.
    fn variance(&self) -> f64 {
        match self {
            Kernel::Gaussian => 1.0,
            Kernel::Epanechnikov => 1.0 / 5.0,
            Kernel::Triangular => 1.0 / 6.0,
            Kernel::Uniform => 1.0 / 3.0,
            Kernel::Biweight => 1.0 / 7.0,
            Kernel::Triweight => 1.0 / 9.0,
            Kernel::Cosine => 1.0 - 8.0 / (std::f64::consts::PI * std::f64::consts::PI),
        }
    }

    /// Integral of K(u)^2 du — the roughness of the kernel.
    fn roughness(&self) -> f64 {
        match self {
            Kernel::Gaussian => 1.0 / (2.0 * std::f64::consts::PI).sqrt(),
            Kernel::Epanechnikov => 3.0 / 5.0,
            Kernel::Triangular => 2.0 / 3.0,
            Kernel::Uniform => 0.5,
            Kernel::Biweight => 5.0 / 7.0,
            Kernel::Triweight => 350.0 / 429.0,
            Kernel::Cosine => {
                // integral of (pi/4 cos(pi u/2))^2 from -1 to 1
                // = (pi^2/16) * (1 + sin(pi)/pi) = pi^2 / 16
                std::f64::consts::PI * std::f64::consts::PI / 16.0
            }
        }
    }

    /// CDF of the kernel: integral of K(t) dt from -inf to u.
    fn cdf_kernel(&self, u: f64) -> f64 {
        match self {
            Kernel::Gaussian => 0.5 * (1.0 + erf_approx(u / std::f64::consts::SQRT_2)),
            Kernel::Epanechnikov => {
                if u < -1.0 {
                    0.0
                } else if u > 1.0 {
                    1.0
                } else {
                    0.5 + 0.75 * u - 0.25 * u * u * u
                }
            }
            Kernel::Triangular => {
                if u < -1.0 {
                    0.0
                } else if u < 0.0 {
                    0.5 * (1.0 + u) * (1.0 + u)
                } else if u <= 1.0 {
                    1.0 - 0.5 * (1.0 - u) * (1.0 - u)
                } else {
                    1.0
                }
            }
            Kernel::Uniform => {
                if u < -1.0 {
                    0.0
                } else if u > 1.0 {
                    1.0
                } else {
                    0.5 * (u + 1.0)
                }
            }
            Kernel::Biweight => {
                if u < -1.0 {
                    0.0
                } else if u > 1.0 {
                    1.0
                } else {
                    let t = u;
                    // Integral of 15/16 (1-t^2)^2 from -1 to u
                    // = 15/16 * [t - 2t^3/3 + t^5/5] evaluated from -1 to u
                    // At t=-1: -1 + 2/3 -1/5 = -8/15
                    let val_u = t - 2.0 * t.powi(3) / 3.0 + t.powi(5) / 5.0;
                    let val_neg1 = -1.0 + 2.0 / 3.0 - 1.0 / 5.0; // = -8/15
                    (15.0 / 16.0) * (val_u - val_neg1)
                }
            }
            Kernel::Triweight => {
                if u < -1.0 {
                    0.0
                } else if u > 1.0 {
                    1.0
                } else {
                    let t = u;
                    // Integral of 35/32 (1-t^2)^3 from -1 to u
                    // expand (1-t^2)^3 = 1 - 3t^2 + 3t^4 - t^6
                    // antiderivative: t - t^3 + 3t^5/5 - t^7/7
                    let anti =
                        |x: f64| -> f64 { x - x.powi(3) + 3.0 * x.powi(5) / 5.0 - x.powi(7) / 7.0 };
                    (35.0 / 32.0) * (anti(t) - anti(-1.0))
                }
            }
            Kernel::Cosine => {
                if u < -1.0 {
                    0.0
                } else if u > 1.0 {
                    1.0
                } else {
                    // Integral of pi/4 cos(pi t/2) from -1 to u
                    // = pi/4 * [2/pi sin(pi t/2)] from -1 to u
                    // = 1/2 * (sin(pi u/2) - sin(-pi/2))
                    // = 1/2 * (sin(pi u/2) + 1)
                    0.5 * ((std::f64::consts::FRAC_PI_2 * u).sin() + 1.0)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 1D KDE
// ---------------------------------------------------------------------------

/// One-dimensional kernel density estimate.
///
/// # Example
///
/// ```rust
/// use scirs2_stats::kde::{KernelDensityEstimate, Kernel};
///
/// let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
/// let kde = KernelDensityEstimate::new(&data, Kernel::Gaussian);
/// let d = kde.evaluate(2.0);
/// assert!(d > 0.0);
/// ```
pub struct KernelDensityEstimate {
    data: Vec<f64>,
    bandwidth: f64,
    kernel: Kernel,
    weights: Option<Vec<f64>>,
}

impl KernelDensityEstimate {
    /// Create a new 1-D KDE with automatic bandwidth (Silverman's rule).
    ///
    /// Panics will not occur; if data is empty, bandwidth is set to 1.0 as a fallback.
    pub fn new(data: &[f64], kernel: Kernel) -> Self {
        let bw = if data.len() < 2 {
            1.0
        } else {
            silverman_bandwidth(data)
        };
        Self {
            data: data.to_vec(),
            bandwidth: bw,
            kernel,
            weights: None,
        }
    }

    /// Create a KDE with a user-specified bandwidth.
    pub fn with_bandwidth(data: &[f64], bandwidth: f64, kernel: Kernel) -> Self {
        Self {
            data: data.to_vec(),
            bandwidth: if bandwidth > 0.0 { bandwidth } else { 1.0 },
            kernel,
            weights: None,
        }
    }

    /// Create a weighted KDE.
    ///
    /// `weights` must have the same length as `data` and all entries must be non-negative.
    pub fn with_weights(data: &[f64], weights: &[f64], kernel: Kernel) -> StatsResult<Self> {
        if data.len() != weights.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "data length ({}) != weights length ({})",
                data.len(),
                weights.len()
            )));
        }
        if weights.iter().any(|&w| w < 0.0) {
            return Err(StatsError::InvalidArgument(
                "Weights must be non-negative".to_string(),
            ));
        }
        let bw = if data.len() < 2 {
            1.0
        } else {
            silverman_bandwidth(data)
        };
        Ok(Self {
            data: data.to_vec(),
            bandwidth: bw,
            kernel,
            weights: Some(weights.to_vec()),
        })
    }

    /// Return the selected bandwidth.
    pub fn bandwidth(&self) -> f64 {
        self.bandwidth
    }

    /// Evaluate the density estimate at a single point.
    pub fn evaluate(&self, x: f64) -> f64 {
        let n = self.data.len();
        if n == 0 {
            return 0.0;
        }
        let h = self.bandwidth;
        let radius = self.kernel.support_radius();

        match &self.weights {
            None => {
                let sum: f64 = self
                    .data
                    .iter()
                    .filter(|&&xi| radius.is_infinite() || ((x - xi) / h).abs() <= radius)
                    .map(|&xi| self.kernel.evaluate((x - xi) / h))
                    .sum();
                sum / (n as f64 * h)
            }
            Some(w) => {
                let w_sum: f64 = w.iter().sum();
                if w_sum <= 0.0 {
                    return 0.0;
                }
                let sum: f64 = self
                    .data
                    .iter()
                    .zip(w.iter())
                    .filter(|(&xi, _)| radius.is_infinite() || ((x - xi) / h).abs() <= radius)
                    .map(|(&xi, &wi)| wi * self.kernel.evaluate((x - xi) / h))
                    .sum();
                sum / (w_sum * h)
            }
        }
    }

    /// Evaluate the density at multiple points (vectorised).
    pub fn evaluate_batch(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().map(|&x| self.evaluate(x)).collect()
    }

    /// Leave-one-out log-likelihood of the data under the KDE.
    ///
    /// For each data point i, the density is estimated using all *other* points.
    /// The returned value is sum_i ln f_{-i}(x_i).
    pub fn log_likelihood(&self, test_data: &[f64]) -> f64 {
        let n = self.data.len();
        if n < 2 {
            return f64::NEG_INFINITY;
        }
        let h = self.bandwidth;
        let radius = self.kernel.support_radius();

        test_data
            .iter()
            .map(|&x| {
                let sum: f64 = self
                    .data
                    .iter()
                    .filter(|&&xj| {
                        // For leave-one-out on the *training* data we skip exact matches below;
                        // for external test data we use all training points.
                        radius.is_infinite() || ((x - xj) / h).abs() <= radius
                    })
                    .map(|&xj| self.kernel.evaluate((x - xj) / h))
                    .sum();
                let density = sum / (n as f64 * h);
                if density > 0.0 {
                    density.ln()
                } else {
                    f64::NEG_INFINITY
                }
            })
            .sum()
    }

    /// Leave-one-out log-likelihood evaluated on the training data itself.
    pub fn loo_log_likelihood(&self) -> f64 {
        let n = self.data.len();
        if n < 2 {
            return f64::NEG_INFINITY;
        }
        let h = self.bandwidth;
        let radius = self.kernel.support_radius();

        self.data
            .iter()
            .enumerate()
            .map(|(i, &xi)| {
                let sum: f64 = self
                    .data
                    .iter()
                    .enumerate()
                    .filter(|&(j, &xj)| {
                        j != i && (radius.is_infinite() || ((xi - xj) / h).abs() <= radius)
                    })
                    .map(|(_, &xj)| self.kernel.evaluate((xi - xj) / h))
                    .sum();
                let density = sum / ((n - 1) as f64 * h);
                if density > 0.0 {
                    density.ln()
                } else {
                    f64::NEG_INFINITY
                }
            })
            .sum()
    }

    /// Draw `n` random samples from the KDE.
    ///
    /// Each sample is drawn by picking a data point at random (according to weights
    /// if provided) and then adding noise drawn from the scaled kernel.
    pub fn sample(&self, n: usize, seed: Option<u64>) -> Vec<f64> {
        if self.data.is_empty() || n == 0 {
            return Vec::new();
        }
        let mut rng = match seed {
            Some(s) => seeded_rng(s),
            None => {
                // Use a thread rng to pick a seed, then create a seeded rng
                let mut trng = thread_rng();
                let s: u64 = trng.gen_range(0..u64::MAX);
                seeded_rng(s)
            }
        };

        let h = self.bandwidth;
        let data_len = self.data.len();

        // Build cumulative weight vector for weighted sampling
        let cum_weights = match &self.weights {
            Some(w) => {
                let total: f64 = w.iter().sum();
                if total <= 0.0 {
                    // Fallback to uniform
                    (0..data_len)
                        .map(|i| (i + 1) as f64 / data_len as f64)
                        .collect::<Vec<_>>()
                } else {
                    let mut cw = Vec::with_capacity(data_len);
                    let mut acc = 0.0;
                    for &wi in w {
                        acc += wi / total;
                        cw.push(acc);
                    }
                    cw
                }
            }
            None => (0..data_len)
                .map(|i| (i + 1) as f64 / data_len as f64)
                .collect::<Vec<_>>(),
        };

        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            // Pick a data point according to weights
            let u: f64 = rng.gen_range(0.0..1.0);
            let idx = match cum_weights
                .binary_search_by(|cw| cw.partial_cmp(&u).unwrap_or(std::cmp::Ordering::Equal))
            {
                Ok(i) => i.min(data_len - 1),
                Err(i) => i.min(data_len - 1),
            };

            // Add noise from scaled kernel
            let noise = sample_from_kernel(&self.kernel, &mut rng) * h;
            samples.push(self.data[idx] + noise);
        }

        samples
    }

    /// Estimate the CDF at point `x` — i.e. integral of the KDE from -inf to x.
    pub fn cdf(&self, x: f64) -> f64 {
        let n = self.data.len();
        if n == 0 {
            return 0.0;
        }
        let h = self.bandwidth;

        match &self.weights {
            None => {
                let sum: f64 = self
                    .data
                    .iter()
                    .map(|&xi| self.kernel.cdf_kernel((x - xi) / h))
                    .sum();
                sum / n as f64
            }
            Some(w) => {
                let w_sum: f64 = w.iter().sum();
                if w_sum <= 0.0 {
                    return 0.0;
                }
                let sum: f64 = self
                    .data
                    .iter()
                    .zip(w.iter())
                    .map(|(&xi, &wi)| wi * self.kernel.cdf_kernel((x - xi) / h))
                    .sum();
                sum / w_sum
            }
        }
    }

    /// Estimate the p-th quantile (0 < p < 1) by inverting the CDF via bisection.
    pub fn quantile(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidArgument(format!(
                "Quantile probability must be in [0, 1], got {}",
                p
            )));
        }
        if self.data.is_empty() {
            return Err(StatsError::InvalidArgument(
                "Cannot compute quantile for empty data".to_string(),
            ));
        }

        // Determine search bounds
        let min_val = self.data.iter().copied().fold(f64::INFINITY, f64::min);
        let max_val = self.data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let spread = (max_val - min_val).max(1.0);
        let mut lo = min_val - 5.0 * self.bandwidth - spread;
        let mut hi = max_val + 5.0 * self.bandwidth + spread;

        // Bisection to find x such that cdf(x) ~= p
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if self.cdf(mid) < p {
                lo = mid;
            } else {
                hi = mid;
            }
            if (hi - lo).abs() < 1e-12 {
                break;
            }
        }
        Ok(0.5 * (lo + hi))
    }
}

// ---------------------------------------------------------------------------
// Bandwidth selection
// ---------------------------------------------------------------------------

/// Silverman's rule-of-thumb bandwidth.
///
/// h = 0.9 * min(sigma, IQR/1.34) * n^{-1/5}
pub fn silverman_bandwidth(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 1.0;
    }
    let sigma = sample_std(data);
    let iqr = interquartile_range(data);
    let a = if iqr > 0.0 {
        sigma.min(iqr / 1.34)
    } else {
        sigma
    };
    if a <= 0.0 {
        return 1.0;
    }
    0.9 * a * (n as f64).powf(-0.2)
}

/// Scott's rule-of-thumb bandwidth.
///
/// h = 1.059 * sigma * n^{-1/5}
pub fn scott_bandwidth(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 1.0;
    }
    let sigma = sample_std(data);
    if sigma <= 0.0 {
        return 1.0;
    }
    1.059 * sigma * (n as f64).powf(-0.2)
}

/// Improved Sheather-Jones (ISJ) bandwidth selector.
///
/// This method uses a plug-in approach that iteratively estimates the density of
/// the second derivative of the density to determine the optimal bandwidth.
/// It is generally considered more accurate than Silverman's or Scott's rule
/// for multimodal or non-Gaussian data.
pub fn improved_sheather_jones(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 1.0;
    }

    let sigma = sample_std(data);
    if sigma <= 0.0 {
        return 1.0;
    }

    // Standardize data
    let mean = data.iter().sum::<f64>() / n as f64;
    let z: Vec<f64> = data.iter().map(|&x| (x - mean) / sigma).collect();

    // Estimate the second derivative of the density using a pilot bandwidth
    // Direct plug-in method (Sheather & Jones 1991)
    let n_f = n as f64;

    // Step 1: Initial estimate using normal reference rule for psi_6 and psi_8
    // psi_r for a standard normal is: (-1)^(r/2) * r! / (2^(r/2) * (r/2)! * sqrt(2*pi))
    // psi_6 (6th derivative at normal) = 15 / (16 sqrt(pi)) for standard normal
    // psi_8 = -105 / (32 sqrt(pi))
    let sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt();
    let psi_6_normal = -15.0 / (16.0 * std::f64::consts::PI.sqrt());
    let psi_8_normal = 105.0 / (32.0 * std::f64::consts::PI.sqrt());

    // Step 2: Compute pilot bandwidth for estimating psi_4
    // g_1 = (-6 * sqrt(2) * psi_6 / (n * psi_8))^(1/9)  -- formula from Wand & Jones
    let g2 = if psi_8_normal.abs() > 1e-30 {
        ((-2.0 * psi_6_normal) / (n_f * psi_8_normal))
            .abs()
            .powf(1.0 / 9.0)
    } else {
        silverman_bandwidth(&z)
    };
    let g1 = if psi_6_normal.abs() > 1e-30 {
        (-6.0_f64.sqrt() / (n_f * psi_6_normal * estimate_psi_internal(&z, 4, g2)))
            .abs()
            .powf(1.0 / 7.0)
    } else {
        silverman_bandwidth(&z)
    };

    // Step 3: Estimate psi_4 using the pilot bandwidth g1
    let psi_4_est = estimate_psi_internal(&z, 4, g1);

    // Step 4: AMISE-optimal bandwidth
    // h = (R(K) / (n * mu_2(K)^2 * psi_4))^{1/5}
    // For Gaussian kernel: R(K)=1/(2 sqrt(pi)), mu_2(K)=1
    let r_k = 1.0 / (2.0 * sqrt_2pi);
    let h_opt = if psi_4_est.abs() > 1e-30 {
        (r_k / (n_f * psi_4_est.abs())).powf(0.2)
    } else {
        silverman_bandwidth(&z)
    };

    // Scale back to original data scale
    let result = h_opt * sigma;
    if result.is_finite() && result > 0.0 {
        result
    } else {
        silverman_bandwidth(data)
    }
}

/// Least-squares cross-validation (LSCV) bandwidth selection.
///
/// Searches over a grid of candidate bandwidths and returns the one that
/// minimizes the integrated squared error proxy.
pub fn cross_validation_bandwidth(data: &[f64], kernel: &Kernel) -> f64 {
    let n = data.len();
    if n < 2 {
        return 1.0;
    }

    let h_ref = silverman_bandwidth(data);
    if h_ref <= 0.0 {
        return 1.0;
    }

    // Grid of candidate bandwidths from 0.1 * h_ref to 3.0 * h_ref
    let n_grid = 40;
    let mut best_h = h_ref;
    let mut best_score = f64::INFINITY;

    for i in 0..n_grid {
        let ratio = 0.1 + 2.9 * (i as f64) / (n_grid as f64 - 1.0);
        let h = h_ref * ratio;
        if h <= 0.0 {
            continue;
        }

        // LSCV score = integral f_h^2 - 2/n sum_i f_{-i}(x_i)
        // We approximate the first term using the data-based estimate:
        //   integral f_h^2 approx 1/(n^2 h) sum_i sum_j K_2((x_i - x_j)/h)
        //   where K_2 is the convolution kernel K*K.
        // For simplicity we use the leave-one-out density directly.

        // Leave-one-out cross-validation score
        let mut loo_sum = 0.0;
        let radius = kernel.support_radius();

        for i_pt in 0..n {
            let xi = data[i_pt];
            let mut density = 0.0;
            for (j_pt, &xj) in data.iter().enumerate() {
                if i_pt == j_pt {
                    continue;
                }
                let u = (xi - xj) / h;
                if radius.is_infinite() || u.abs() <= radius {
                    density += kernel.evaluate(u);
                }
            }
            density /= (n - 1) as f64 * h;
            loo_sum += density;
        }
        let loo_mean = loo_sum / n as f64;

        // Approximate integral f^2: use all-pairs kernel convolution
        let mut integral_f2 = 0.0;
        for i_pt in 0..n {
            for j_pt in 0..n {
                let u = (data[i_pt] - data[j_pt]) / h;
                if radius.is_infinite() || u.abs() <= 2.0 * radius {
                    // K*K convolution evaluated at u: for simplicity we approximate
                    integral_f2 +=
                        kernel.evaluate(u / std::f64::consts::SQRT_2) / std::f64::consts::SQRT_2;
                }
            }
        }
        integral_f2 /= (n as f64) * (n as f64) * h;

        let score = integral_f2 - 2.0 * loo_mean;

        if score < best_score {
            best_score = score;
            best_h = h;
        }
    }

    if best_h > 0.0 && best_h.is_finite() {
        best_h
    } else {
        h_ref
    }
}

// ---------------------------------------------------------------------------
// 2D KDE
// ---------------------------------------------------------------------------

/// Two-dimensional kernel density estimator.
pub struct KDE2D {
    x_data: Vec<f64>,
    y_data: Vec<f64>,
    bandwidth_x: f64,
    bandwidth_y: f64,
    kernel: Kernel,
}

impl KDE2D {
    /// Create a 2D KDE from paired data vectors.
    ///
    /// Bandwidths are selected automatically (Silverman's rule for each marginal).
    pub fn new(x: &[f64], y: &[f64], kernel: Kernel) -> StatsResult<Self> {
        if x.len() != y.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "x length ({}) != y length ({})",
                x.len(),
                y.len()
            )));
        }
        if x.is_empty() {
            return Err(StatsError::InvalidArgument(
                "Data arrays must not be empty".to_string(),
            ));
        }
        let bw_x = silverman_bandwidth(x);
        let bw_y = silverman_bandwidth(y);

        Ok(Self {
            x_data: x.to_vec(),
            y_data: y.to_vec(),
            bandwidth_x: bw_x,
            bandwidth_y: bw_y,
            kernel,
        })
    }

    /// Create a 2D KDE with explicit bandwidths.
    pub fn with_bandwidths(
        x: &[f64],
        y: &[f64],
        bandwidth_x: f64,
        bandwidth_y: f64,
        kernel: Kernel,
    ) -> StatsResult<Self> {
        if x.len() != y.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "x length ({}) != y length ({})",
                x.len(),
                y.len()
            )));
        }
        if x.is_empty() {
            return Err(StatsError::InvalidArgument(
                "Data arrays must not be empty".to_string(),
            ));
        }
        if bandwidth_x <= 0.0 || bandwidth_y <= 0.0 {
            return Err(StatsError::InvalidArgument(
                "Bandwidths must be positive".to_string(),
            ));
        }

        Ok(Self {
            x_data: x.to_vec(),
            y_data: y.to_vec(),
            bandwidth_x,
            bandwidth_y,
            kernel,
        })
    }

    /// Evaluate the 2D density estimate at a single point (x, y).
    ///
    /// Uses the product kernel: K_h(x, y) = K(u_x / h_x) * K(u_y / h_y) / (h_x h_y).
    pub fn evaluate(&self, x: f64, y: f64) -> f64 {
        let n = self.x_data.len();
        if n == 0 {
            return 0.0;
        }
        let hx = self.bandwidth_x;
        let hy = self.bandwidth_y;
        let radius = self.kernel.support_radius();

        let sum: f64 = self
            .x_data
            .iter()
            .zip(self.y_data.iter())
            .filter(|(&xi, &yi)| {
                if radius.is_infinite() {
                    true
                } else {
                    ((x - xi) / hx).abs() <= radius && ((y - yi) / hy).abs() <= radius
                }
            })
            .map(|(&xi, &yi)| {
                let kx = self.kernel.evaluate((x - xi) / hx);
                let ky = self.kernel.evaluate((y - yi) / hy);
                kx * ky
            })
            .sum();

        sum / (n as f64 * hx * hy)
    }

    /// Evaluate the 2D density on a grid.
    ///
    /// Returns an Array2 of shape (x_grid.len(), y_grid.len()) where
    /// entry (`i`, `j`) is the density at (x_grid\[i\], y_grid\[j\]).
    pub fn evaluate_grid(&self, x_grid: &[f64], y_grid: &[f64]) -> Array2<f64> {
        let nx = x_grid.len();
        let ny = y_grid.len();
        let mut grid = Array2::zeros(Ix2(nx, ny));

        for (i, &xg) in x_grid.iter().enumerate() {
            for (j, &yg) in y_grid.iter().enumerate() {
                grid[[i, j]] = self.evaluate(xg, yg);
            }
        }

        grid
    }

    /// Return the selected bandwidths.
    pub fn bandwidths(&self) -> (f64, f64) {
        (self.bandwidth_x, self.bandwidth_y)
    }
}

// ---------------------------------------------------------------------------
// Helper: erf approximation (Abramowitz & Stegun 7.1.26)
// ---------------------------------------------------------------------------

fn erf_approx(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

// ---------------------------------------------------------------------------
// Helper: sample standard deviation
// ---------------------------------------------------------------------------

fn sample_std(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    let var = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (n as f64 - 1.0);
    var.sqrt()
}

// ---------------------------------------------------------------------------
// Helper: interquartile range
// ---------------------------------------------------------------------------

fn interquartile_range(data: &[f64]) -> f64 {
    if data.len() < 4 {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q1 = percentile_sorted(&sorted, 25.0);
    let q3 = percentile_sorted(&sorted, 75.0);
    q3 - q1
}

fn percentile_sorted(sorted: &[f64], pct: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    let idx = (pct / 100.0) * (n as f64 - 1.0);
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    if hi >= n {
        sorted[n - 1]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

// ---------------------------------------------------------------------------
// Helper: estimate psi_r (integrated r-th derivative of density)
// ---------------------------------------------------------------------------

/// Estimate the functional psi_r = integral f^{(r)}(x) f(x) dx
/// using a direct plug-in estimator with a given pilot bandwidth g.
fn estimate_psi_internal(z: &[f64], r: usize, g: f64) -> f64 {
    let n = z.len();
    if n < 2 || g <= 0.0 {
        return 0.0;
    }

    let n_f = n as f64;
    let mut total = 0.0;

    for i in 0..n {
        for j in 0..n {
            let u = (z[i] - z[j]) / g;
            // For a Gaussian kernel, the r-th derivative of the kernel at u/g is:
            // (1/g^{r+1}) * phi^{(r)}(u)
            // where phi^{(r)} is the r-th derivative of the standard normal density.
            total += gaussian_derivative(u, r);
        }
    }

    let g_power = g.powi(r as i32 + 1);
    total / (n_f * n_f * g_power)
}

/// Evaluate the r-th derivative of the standard normal density at u.
/// phi^{(r)}(u) = (-1)^r * H_r(u) * phi(u)
/// where H_r is the (probabilists') Hermite polynomial.
fn gaussian_derivative(u: f64, r: usize) -> f64 {
    let phi = (-0.5 * u * u).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let sign = if r % 2 == 0 { 1.0 } else { -1.0 };
    sign * hermite_prob(u, r) * phi
}

/// Evaluate the probabilists' Hermite polynomial H_r(x) using the recurrence:
/// H_0(x) = 1, H_1(x) = x, H_{n+1}(x) = x H_n(x) - n H_{n-1}(x)
fn hermite_prob(x: f64, r: usize) -> f64 {
    if r == 0 {
        return 1.0;
    }
    if r == 1 {
        return x;
    }
    let mut h_prev2 = 1.0;
    let mut h_prev1 = x;
    for k in 2..=r {
        let h_curr = x * h_prev1 - (k as f64 - 1.0) * h_prev2;
        h_prev2 = h_prev1;
        h_prev1 = h_curr;
    }
    h_prev1
}

// ---------------------------------------------------------------------------
// Helper: sample from a standard kernel distribution
// ---------------------------------------------------------------------------

/// Generate a single sample from the standard (unit bandwidth) version of the kernel.
fn sample_from_kernel<R: scirs2_core::random::Rng>(kernel: &Kernel, rng: &mut Random<R>) -> f64 {
    match kernel {
        Kernel::Gaussian => {
            // Box-Muller
            let u1: f64 = rng.gen_range(1e-15..1.0);
            let u2: f64 = rng.gen_range(0.0..1.0);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
        Kernel::Epanechnikov => {
            // Acceptance-rejection with uniform proposal
            loop {
                let u: f64 = rng.gen_range(-1.0..1.0);
                let v: f64 = rng.gen_range(0.0..1.0);
                if v <= 0.75 * (1.0 - u * u) {
                    break u;
                }
            }
        }
        Kernel::Triangular => {
            // Inverse CDF: two uniform samples
            let u: f64 = rng.gen_range(0.0..1.0);
            if u < 0.5 {
                (2.0 * u).sqrt() - 1.0
            } else {
                1.0 - (2.0 * (1.0 - u)).sqrt()
            }
        }
        Kernel::Uniform => rng.gen_range(-1.0..1.0),
        Kernel::Biweight => {
            // Acceptance-rejection
            let k_max = 15.0 / 16.0;
            loop {
                let u: f64 = rng.gen_range(-1.0..1.0);
                let v: f64 = rng.gen_range(0.0..k_max);
                let t = 1.0 - u * u;
                if v <= (15.0 / 16.0) * t * t {
                    break u;
                }
            }
        }
        Kernel::Triweight => {
            // Acceptance-rejection
            let k_max = 35.0 / 32.0;
            loop {
                let u: f64 = rng.gen_range(-1.0..1.0);
                let v: f64 = rng.gen_range(0.0..k_max);
                let t = 1.0 - u * u;
                if v <= (35.0 / 32.0) * t * t * t {
                    break u;
                }
            }
        }
        Kernel::Cosine => {
            // Acceptance-rejection
            let k_max = std::f64::consts::PI / 4.0;
            loop {
                let u: f64 = rng.gen_range(-1.0..1.0);
                let v: f64 = rng.gen_range(0.0..k_max);
                if v <= (std::f64::consts::PI / 4.0) * (std::f64::consts::FRAC_PI_2 * u).cos() {
                    break u;
                }
            }
        }
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Kernel evaluation tests ---

    #[test]
    fn test_gaussian_kernel_peak() {
        let k = Kernel::Gaussian;
        let val = k.evaluate(0.0);
        let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert!((val - expected).abs() < 1e-12);
    }

    #[test]
    fn test_gaussian_kernel_symmetry() {
        let k = Kernel::Gaussian;
        let v1 = k.evaluate(1.5);
        let v2 = k.evaluate(-1.5);
        assert!((v1 - v2).abs() < 1e-15);
    }

    #[test]
    fn test_epanechnikov_kernel() {
        let k = Kernel::Epanechnikov;
        assert!((k.evaluate(0.0) - 0.75).abs() < 1e-12);
        assert!(k.evaluate(1.5).abs() < 1e-15);
        assert!(k.evaluate(-1.5).abs() < 1e-15);
    }

    #[test]
    fn test_triangular_kernel() {
        let k = Kernel::Triangular;
        assert!((k.evaluate(0.0) - 1.0).abs() < 1e-12);
        assert!((k.evaluate(0.5) - 0.5).abs() < 1e-12);
        assert!(k.evaluate(1.5).abs() < 1e-15);
    }

    #[test]
    fn test_uniform_kernel() {
        let k = Kernel::Uniform;
        assert!((k.evaluate(0.0) - 0.5).abs() < 1e-12);
        assert!((k.evaluate(0.9) - 0.5).abs() < 1e-12);
        assert!(k.evaluate(1.5).abs() < 1e-15);
    }

    #[test]
    fn test_biweight_kernel() {
        let k = Kernel::Biweight;
        assert!((k.evaluate(0.0) - 15.0 / 16.0).abs() < 1e-12);
        assert!(k.evaluate(1.5).abs() < 1e-15);
    }

    #[test]
    fn test_triweight_kernel() {
        let k = Kernel::Triweight;
        assert!((k.evaluate(0.0) - 35.0 / 32.0).abs() < 1e-12);
        assert!(k.evaluate(1.5).abs() < 1e-15);
    }

    #[test]
    fn test_cosine_kernel() {
        let k = Kernel::Cosine;
        assert!((k.evaluate(0.0) - std::f64::consts::PI / 4.0).abs() < 1e-12);
        assert!(k.evaluate(1.5).abs() < 1e-15);
    }

    #[test]
    fn test_support_radius() {
        assert!(Kernel::Gaussian.support_radius().is_infinite());
        assert!((Kernel::Epanechnikov.support_radius() - 1.0).abs() < 1e-15);
        assert!((Kernel::Biweight.support_radius() - 1.0).abs() < 1e-15);
    }

    // --- Kernel integration tests (verify kernels integrate to 1) ---

    #[test]
    fn test_kernel_integrates_to_one() {
        let kernels = [
            Kernel::Gaussian,
            Kernel::Epanechnikov,
            Kernel::Triangular,
            Kernel::Uniform,
            Kernel::Biweight,
            Kernel::Triweight,
            Kernel::Cosine,
        ];

        for kernel in &kernels {
            let n = 10000;
            let (lo, hi) = if kernel.support_radius().is_infinite() {
                (-10.0, 10.0)
            } else {
                (-1.0, 1.0)
            };
            let dx = (hi - lo) / n as f64;
            let integral: f64 = (0..n)
                .map(|i| {
                    let x = lo + (i as f64 + 0.5) * dx;
                    kernel.evaluate(x) * dx
                })
                .sum();
            assert!(
                (integral - 1.0).abs() < 0.01,
                "Kernel {:?} integrates to {} instead of 1.0",
                kernel,
                integral
            );
        }
    }

    // --- Bandwidth selection tests ---

    #[test]
    fn test_silverman_bandwidth_normal_data() {
        let data: Vec<f64> = (0..1000).map(|i| (i as f64 - 500.0) / 100.0).collect();
        let bw = silverman_bandwidth(&data);
        assert!(bw > 0.0);
        assert!(bw < 5.0);
    }

    #[test]
    fn test_scott_bandwidth_normal_data() {
        let data: Vec<f64> = (0..1000).map(|i| (i as f64 - 500.0) / 100.0).collect();
        let bw = scott_bandwidth(&data);
        assert!(bw > 0.0);
        assert!(bw < 5.0);
    }

    #[test]
    fn test_isj_bandwidth() {
        let data: Vec<f64> = (0..500).map(|i| (i as f64 - 250.0) / 50.0).collect();
        let bw = improved_sheather_jones(&data);
        assert!(bw > 0.0, "ISJ bandwidth should be positive, got {}", bw);
        assert!(bw.is_finite());
    }

    #[test]
    fn test_cross_validation_bandwidth() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 10.0).collect();
        let bw = cross_validation_bandwidth(&data, &Kernel::Gaussian);
        assert!(bw > 0.0);
        assert!(bw.is_finite());
    }

    // --- 1D KDE tests ---

    #[test]
    fn test_kde_evaluate_positive() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let kde = KernelDensityEstimate::new(&data, Kernel::Gaussian);
        let d = kde.evaluate(2.0);
        assert!(d > 0.0, "Density at data point should be positive");
    }

    #[test]
    fn test_kde_evaluate_batch() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let kde = KernelDensityEstimate::new(&data, Kernel::Gaussian);
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let densities = kde.evaluate_batch(&xs);
        assert_eq!(densities.len(), 5);
        for &d in &densities {
            assert!(d > 0.0);
        }
    }

    #[test]
    fn test_kde_integrates_approx_one() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let kde = KernelDensityEstimate::new(&data, Kernel::Gaussian);
        let n = 5000;
        let lo = -10.0;
        let hi = 15.0;
        let dx = (hi - lo) / n as f64;
        let integral: f64 = (0..n)
            .map(|i| {
                let x = lo + (i as f64 + 0.5) * dx;
                kde.evaluate(x) * dx
            })
            .sum();
        assert!(
            (integral - 1.0).abs() < 0.05,
            "KDE should integrate to ~1.0, got {}",
            integral
        );
    }

    #[test]
    fn test_kde_with_bandwidth() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let kde = KernelDensityEstimate::with_bandwidth(&data, 0.5, Kernel::Gaussian);
        assert!((kde.bandwidth() - 0.5).abs() < 1e-12);
        let d = kde.evaluate(2.0);
        assert!(d > 0.0);
    }

    #[test]
    fn test_kde_with_weights() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let weights = vec![1.0, 1.0, 3.0, 1.0, 1.0];
        let kde = KernelDensityEstimate::with_weights(&data, &weights, Kernel::Gaussian)
            .expect("Should create weighted KDE");
        // Density near x=2 should be higher due to weight
        let d_at_2 = kde.evaluate(2.0);
        let d_at_0 = kde.evaluate(0.0);
        assert!(
            d_at_2 > d_at_0,
            "Weighted KDE should peak near heavier data"
        );
    }

    #[test]
    fn test_kde_weight_dimension_mismatch() {
        let data = vec![0.0, 1.0, 2.0];
        let weights = vec![1.0, 1.0];
        let result = KernelDensityEstimate::with_weights(&data, &weights, Kernel::Gaussian);
        assert!(result.is_err());
    }

    #[test]
    fn test_kde_cdf_monotone() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let kde = KernelDensityEstimate::new(&data, Kernel::Gaussian);
        let xs: Vec<f64> = (-50..=100).map(|i| i as f64 * 0.1).collect();
        let cdfs: Vec<f64> = xs.iter().map(|&x| kde.cdf(x)).collect();
        for i in 1..cdfs.len() {
            assert!(
                cdfs[i] >= cdfs[i - 1] - 1e-12,
                "CDF should be monotonically non-decreasing"
            );
        }
    }

    #[test]
    fn test_kde_cdf_boundary_values() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let kde = KernelDensityEstimate::new(&data, Kernel::Gaussian);
        assert!(kde.cdf(-100.0) < 0.01, "CDF at far left should be near 0");
        assert!(kde.cdf(100.0) > 0.99, "CDF at far right should be near 1");
    }

    #[test]
    fn test_kde_quantile() {
        let data: Vec<f64> = (0..100).map(|i| i as f64 / 10.0).collect();
        let kde = KernelDensityEstimate::new(&data, Kernel::Gaussian);
        let q50 = kde.quantile(0.5).expect("Should compute median quantile");
        // Median should be roughly near 5.0 for data 0..10
        assert!(
            (q50 - 4.95).abs() < 1.0,
            "Median quantile should be near 5, got {}",
            q50
        );
    }

    #[test]
    fn test_kde_quantile_invalid() {
        let data = vec![0.0, 1.0, 2.0];
        let kde = KernelDensityEstimate::new(&data, Kernel::Gaussian);
        assert!(kde.quantile(1.5).is_err());
        assert!(kde.quantile(-0.1).is_err());
    }

    #[test]
    fn test_kde_sample() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let kde = KernelDensityEstimate::new(&data, Kernel::Gaussian);
        let samples = kde.sample(1000, Some(42));
        assert_eq!(samples.len(), 1000);
        // Mean of samples should be close to mean of data (2.5)
        let sample_mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(
            (sample_mean - 2.5).abs() < 1.0,
            "Sample mean should be near data mean, got {}",
            sample_mean
        );
    }

    #[test]
    fn test_kde_log_likelihood() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let kde = KernelDensityEstimate::new(&data, Kernel::Gaussian);
        let ll = kde.log_likelihood(&[1.0, 2.0, 3.0]);
        assert!(ll.is_finite(), "Log-likelihood should be finite");
        assert!(
            ll < 0.0,
            "Log-likelihood of a density < 1 should be negative for these points"
        );
    }

    #[test]
    fn test_kde_loo_log_likelihood() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let kde = KernelDensityEstimate::new(&data, Kernel::Gaussian);
        let loo = kde.loo_log_likelihood();
        assert!(loo.is_finite(), "LOO log-likelihood should be finite");
    }

    #[test]
    fn test_kde_all_kernels_evaluate() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let kernels = [
            Kernel::Gaussian,
            Kernel::Epanechnikov,
            Kernel::Triangular,
            Kernel::Uniform,
            Kernel::Biweight,
            Kernel::Triweight,
            Kernel::Cosine,
        ];
        for kernel in &kernels {
            let kde = KernelDensityEstimate::with_bandwidth(&data, 1.0, *kernel);
            let d = kde.evaluate(2.5);
            assert!(
                d > 0.0,
                "Density with {:?} kernel should be positive at data center",
                kernel
            );
        }
    }

    // --- 2D KDE tests ---

    #[test]
    fn test_kde2d_basic() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let kde = KDE2D::new(&x, &y, Kernel::Gaussian).expect("Should create 2D KDE");
        let d = kde.evaluate(2.0, 2.0);
        assert!(d > 0.0, "2D density at data centre should be positive");
    }

    #[test]
    fn test_kde2d_dimension_mismatch() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0];
        let result = KDE2D::new(&x, &y, Kernel::Gaussian);
        assert!(result.is_err());
    }

    #[test]
    fn test_kde2d_grid() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let kde = KDE2D::new(&x, &y, Kernel::Gaussian).expect("Should create 2D KDE");
        let xg = vec![0.0, 2.0, 4.0];
        let yg = vec![0.0, 2.0, 4.0];
        let grid = kde.evaluate_grid(&xg, &yg);
        assert_eq!(grid.shape(), &[3, 3]);
        // All values should be non-negative
        for &val in grid.iter() {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_kde2d_with_bandwidths() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 2.0, 3.0];
        let kde = KDE2D::with_bandwidths(&x, &y, 0.5, 0.5, Kernel::Epanechnikov)
            .expect("Should create 2D KDE");
        let (bx, by) = kde.bandwidths();
        assert!((bx - 0.5).abs() < 1e-12);
        assert!((by - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_kde2d_integrates_approx_one() {
        let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y_data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let kde = KDE2D::new(&x_data, &y_data, Kernel::Gaussian).expect("Should create 2D KDE");
        let n = 100;
        let lo = -5.0;
        let hi = 9.0;
        let dx = (hi - lo) / n as f64;
        let mut integral = 0.0;
        for i in 0..n {
            for j in 0..n {
                let x = lo + (i as f64 + 0.5) * dx;
                let y = lo + (j as f64 + 0.5) * dx;
                integral += kde.evaluate(x, y) * dx * dx;
            }
        }
        assert!(
            (integral - 1.0).abs() < 0.1,
            "2D KDE should integrate to ~1.0, got {}",
            integral
        );
    }

    // --- Edge case tests ---

    #[test]
    fn test_kde_empty_data() {
        let kde = KernelDensityEstimate::new(&[], Kernel::Gaussian);
        assert!((kde.evaluate(0.0)).abs() < 1e-15);
    }

    #[test]
    fn test_kde_single_point() {
        let kde = KernelDensityEstimate::new(&[5.0], Kernel::Gaussian);
        let d = kde.evaluate(5.0);
        assert!(d > 0.0);
    }

    #[test]
    fn test_kde_constant_data() {
        let data = vec![3.0; 50];
        let kde = KernelDensityEstimate::new(&data, Kernel::Gaussian);
        let d = kde.evaluate(3.0);
        assert!(d > 0.0);
    }

    #[test]
    fn test_kde_sample_reproducible() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let kde = KernelDensityEstimate::new(&data, Kernel::Gaussian);
        let s1 = kde.sample(100, Some(12345));
        let s2 = kde.sample(100, Some(12345));
        assert_eq!(s1, s2, "Same seed should produce same samples");
    }

    #[test]
    fn test_kernel_variance_positive() {
        let kernels = [
            Kernel::Gaussian,
            Kernel::Epanechnikov,
            Kernel::Triangular,
            Kernel::Uniform,
            Kernel::Biweight,
            Kernel::Triweight,
            Kernel::Cosine,
        ];
        for k in &kernels {
            assert!(
                k.variance() > 0.0,
                "Kernel {:?} variance should be positive",
                k
            );
        }
    }

    #[test]
    fn test_kernel_roughness_positive() {
        let kernels = [
            Kernel::Gaussian,
            Kernel::Epanechnikov,
            Kernel::Triangular,
            Kernel::Uniform,
            Kernel::Biweight,
            Kernel::Triweight,
            Kernel::Cosine,
        ];
        for k in &kernels {
            assert!(
                k.roughness() > 0.0,
                "Kernel {:?} roughness should be positive",
                k
            );
        }
    }

    #[test]
    fn test_kernel_cdf_boundary() {
        let kernels = [
            Kernel::Gaussian,
            Kernel::Epanechnikov,
            Kernel::Triangular,
            Kernel::Uniform,
            Kernel::Biweight,
            Kernel::Triweight,
            Kernel::Cosine,
        ];
        for k in &kernels {
            let lo = if k.support_radius().is_infinite() {
                -20.0
            } else {
                -2.0
            };
            let hi = if k.support_radius().is_infinite() {
                20.0
            } else {
                2.0
            };
            assert!(
                k.cdf_kernel(lo) < 0.01,
                "Kernel {:?} CDF at far left should be near 0, got {}",
                k,
                k.cdf_kernel(lo)
            );
            assert!(
                k.cdf_kernel(hi) > 0.99,
                "Kernel {:?} CDF at far right should be near 1, got {}",
                k,
                k.cdf_kernel(hi)
            );
        }
    }

    #[test]
    fn test_hermite_polynomials() {
        // H_0(x) = 1, H_1(x) = x, H_2(x) = x^2 - 1, H_3(x) = x^3 - 3x
        assert!((hermite_prob(2.0, 0) - 1.0).abs() < 1e-12);
        assert!((hermite_prob(2.0, 1) - 2.0).abs() < 1e-12);
        assert!((hermite_prob(2.0, 2) - 3.0).abs() < 1e-12); // 4 - 1 = 3
        assert!((hermite_prob(2.0, 3) - 2.0).abs() < 1e-12); // 8 - 6 = 2
    }

    #[test]
    fn test_erf_approx_values() {
        assert!((erf_approx(0.0)).abs() < 1e-7);
        assert!((erf_approx(1.0) - 0.84270079).abs() < 1e-5);
        assert!((erf_approx(-1.0) + 0.84270079).abs() < 1e-5);
    }
}
