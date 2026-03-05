//! Fast Independent Component Analysis (FastICA)
//!
//! Implements the FastICA algorithm for blind source separation and
//! independent component analysis. Supports both deflation and symmetric
//! approaches, multiple non-linearity functions, and whitening preprocessing.
//!
//! # Algorithm
//!
//! FastICA finds a linear transformation W such that the components
//! s = W * x are as statistically independent as possible, by maximizing
//! non-Gaussianity (measured via negentropy approximations).
//!
//! # References
//!
//! - Hyvarinen, A. (1999). Fast and Robust Fixed-Point Algorithms for
//!   Independent Component Analysis. IEEE Trans. Neural Networks, 10(3).

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_linalg::svd;

use crate::error::{Result, TransformError};

const EPSILON: f64 = 1e-10;

/// Non-linearity function used in the FastICA iteration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonLinearity {
    /// G(u) = log(cosh(u)), g(u) = tanh(u)
    /// Good general-purpose choice, robust to outliers
    LogCosh,
    /// G(u) = -exp(-u^2/2), g(u) = u*exp(-u^2/2)
    /// Good for super-Gaussian sources
    Exp,
    /// G(u) = u^4/4, g(u) = u^3
    /// Simple but sensitive to outliers
    Cube,
}

/// FastICA algorithm approach
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IcaAlgorithm {
    /// Extract components one by one (deflation approach)
    /// Uses Gram-Schmidt orthogonalization after each component
    Deflation,
    /// Extract all components simultaneously
    /// More stable but more expensive per iteration
    Symmetric,
}

/// FastICA for Independent Component Analysis
///
/// Finds independent components by maximizing non-Gaussianity using
/// fixed-point iteration.
///
/// # Examples
///
/// ```
/// use scirs2_transform::reduction::fastica::{FastICA, NonLinearity, IcaAlgorithm};
/// use scirs2_core::ndarray::Array2;
///
/// // Create mixed signals
/// let n = 100;
/// let mut data = Vec::new();
/// for i in 0..n {
///     let t = i as f64 / n as f64 * 10.0;
///     let s1 = t.sin();
///     let s2 = ((2.0 * t).sin()).signum();
///     // Mix signals
///     data.push(0.6 * s1 + 0.4 * s2);
///     data.push(0.3 * s1 + 0.7 * s2);
/// }
/// let x = Array2::from_shape_vec((n, 2), data).expect("should succeed");
///
/// let mut ica = FastICA::new(2);
/// ica.fit(&x).expect("should succeed");
/// let sources = ica.transform(&x).expect("should succeed");
/// assert_eq!(sources.shape(), &[n, 2]);
/// ```
#[derive(Debug, Clone)]
pub struct FastICA {
    /// Number of independent components to extract
    n_components: usize,
    /// Non-linearity function
    non_linearity: NonLinearity,
    /// Algorithm (deflation or symmetric)
    algorithm: IcaAlgorithm,
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
    /// Whether to perform whitening
    whiten: bool,
    /// Unmixing matrix W, shape (n_components, n_features)
    unmixing_: Option<Array2<f64>>,
    /// Mixing matrix A = W^{-1} (pseudo-inverse), shape (n_features, n_components)
    mixing_: Option<Array2<f64>>,
    /// Whitening matrix, shape (n_components, n_features)
    whitening_: Option<Array2<f64>>,
    /// Mean of training data
    mean_: Option<Array1<f64>>,
    /// Number of features at fit time
    n_features_in_: Option<usize>,
    /// Number of iterations taken
    n_iter_: Option<usize>,
    /// Kurtosis of each component (for ordering)
    kurtosis_: Option<Array1<f64>>,
}

impl FastICA {
    /// Create a new FastICA instance
    ///
    /// # Arguments
    /// * `n_components` - Number of independent components to extract
    pub fn new(n_components: usize) -> Self {
        FastICA {
            n_components,
            non_linearity: NonLinearity::LogCosh,
            algorithm: IcaAlgorithm::Deflation,
            max_iter: 200,
            tol: 1e-4,
            whiten: true,
            unmixing_: None,
            mixing_: None,
            whitening_: None,
            mean_: None,
            n_features_in_: None,
            n_iter_: None,
            kurtosis_: None,
        }
    }

    /// Set the non-linearity function
    pub fn with_non_linearity(mut self, nl: NonLinearity) -> Self {
        self.non_linearity = nl;
        self
    }

    /// Set the algorithm (deflation or symmetric)
    pub fn with_algorithm(mut self, alg: IcaAlgorithm) -> Self {
        self.algorithm = alg;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter.max(1);
        self
    }

    /// Set convergence tolerance
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol.max(1e-15);
        self
    }

    /// Set whether to whiten the data
    pub fn with_whiten(mut self, whiten: bool) -> Self {
        self.whiten = whiten;
        self
    }

    /// Fit the FastICA model
    ///
    /// # Arguments
    /// * `x` - Data matrix, shape (n_samples, n_features)
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        if n_samples < 2 {
            return Err(TransformError::InvalidInput(
                "At least 2 samples required".to_string(),
            ));
        }

        if self.n_components > n_features {
            return Err(TransformError::InvalidInput(format!(
                "n_components={} must be <= n_features={}",
                self.n_components, n_features
            )));
        }

        // Center data
        let mean = x.mean_axis(Axis(0)).ok_or_else(|| {
            TransformError::ComputationError("Failed to compute mean".to_string())
        })?;

        let mut x_centered = Array2::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                x_centered[[i, j]] = x[[i, j]] - mean[j];
            }
        }

        // Whitening
        let x_white = if self.whiten {
            let (whitening_matrix, whitened) = self.whiten_data(&x_centered)?;
            self.whitening_ = Some(whitening_matrix);
            whitened
        } else {
            if n_features != self.n_components {
                return Err(TransformError::InvalidInput(
                    "Without whitening, n_components must equal n_features".to_string(),
                ));
            }
            x_centered.clone()
        };

        let n_white = x_white.shape()[1];

        // Run FastICA
        let (w, n_iter) = match self.algorithm {
            IcaAlgorithm::Deflation => self.fastica_deflation(&x_white, n_white)?,
            IcaAlgorithm::Symmetric => self.fastica_symmetric(&x_white, n_white)?,
        };

        // Compute unmixing and mixing matrices
        let unmixing = if self.whiten {
            let whitening = self.whitening_.as_ref().ok_or_else(|| {
                TransformError::ComputationError("Whitening matrix not available".to_string())
            })?;
            // W_total = W_ica * W_whiten
            mat_mul(&w, whitening, self.n_components, n_white, n_features)
        } else {
            w
        };

        // Mixing matrix = pseudo-inverse of unmixing
        let mixing = pseudo_inverse(&unmixing)?;

        // Compute kurtosis of each component for ordering
        let sources = self.apply_unmixing(&x_centered, &unmixing)?;
        let kurtosis = compute_kurtosis(&sources)?;

        // Order components by absolute kurtosis (descending)
        let mut order: Vec<usize> = (0..self.n_components).collect();
        order.sort_by(|&a, &b| {
            kurtosis[b]
                .abs()
                .partial_cmp(&kurtosis[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut ordered_unmixing = Array2::zeros(unmixing.raw_dim());
        let mut ordered_mixing = Array2::zeros(mixing.raw_dim());
        let mut ordered_kurtosis = Array1::zeros(self.n_components);

        for (new_i, &old_i) in order.iter().enumerate() {
            for j in 0..n_features {
                ordered_unmixing[[new_i, j]] = unmixing[[old_i, j]];
            }
            for j in 0..n_features {
                ordered_mixing[[j, new_i]] = mixing[[j, old_i]];
            }
            ordered_kurtosis[new_i] = kurtosis[old_i];
        }

        self.unmixing_ = Some(ordered_unmixing);
        self.mixing_ = Some(ordered_mixing);
        self.mean_ = Some(mean);
        self.n_features_in_ = Some(n_features);
        self.n_iter_ = Some(n_iter);
        self.kurtosis_ = Some(ordered_kurtosis);

        Ok(())
    }

    /// Whiten data: decorrelate and scale to unit variance
    fn whiten_data(&self, x: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        // Compute covariance matrix
        let mut cov = Array2::<f64>::zeros((n_features, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                for k in j..n_features {
                    let val = x[[i, j]] * x[[i, k]];
                    cov[[j, k]] += val;
                    if j != k {
                        cov[[k, j]] += val;
                    }
                }
            }
        }
        cov.mapv_inplace(|v| v / n_samples as f64);

        // SVD of covariance
        let (u, s, _vt) =
            svd::<f64>(&cov.view(), true, None).map_err(TransformError::LinalgError)?;

        // Whitening matrix: D^{-1/2} * U^T (keeping top n_components)
        let mut whitening = Array2::zeros((self.n_components, n_features));
        for i in 0..self.n_components {
            if s[i] > EPSILON {
                let scale = 1.0 / s[i].sqrt();
                for j in 0..n_features {
                    whitening[[i, j]] = u[[j, i]] * scale;
                }
            }
        }

        // Apply whitening
        let mut whitened = Array2::zeros((n_samples, self.n_components));
        for i in 0..n_samples {
            for j in 0..self.n_components {
                let mut val = 0.0;
                for k in 0..n_features {
                    val += whitening[[j, k]] * x[[i, k]];
                }
                whitened[[i, j]] = val;
            }
        }

        Ok((whitening, whitened))
    }

    /// FastICA deflation approach: extract components one by one
    fn fastica_deflation(&self, x: &Array2<f64>, n_dim: usize) -> Result<(Array2<f64>, usize)> {
        let n_samples = x.shape()[0];
        let mut w_all = Array2::zeros((self.n_components, n_dim));
        let mut total_iter = 0;

        for p in 0..self.n_components {
            // Initialize w_p randomly using a deterministic seed based on component index
            let mut w = Array1::zeros(n_dim);
            for i in 0..n_dim {
                // Deterministic pseudo-random initialization
                let seed = (p * 1000 + i * 7 + 31) as f64;
                w[i] = (seed * 0.618033988749895).fract() * 2.0 - 1.0;
            }

            // Normalize
            let norm = w.iter().map(|&v| v * v).sum::<f64>().sqrt();
            if norm > EPSILON {
                w.mapv_inplace(|v| v / norm);
            }

            let mut converged = false;

            for iter in 0..self.max_iter {
                let w_old = w.clone();

                // Compute w^T * x for all samples
                let mut wx = Array1::zeros(n_samples);
                for i in 0..n_samples {
                    let mut dot = 0.0;
                    for j in 0..n_dim {
                        dot += w[j] * x[[i, j]];
                    }
                    wx[i] = dot;
                }

                // Apply non-linearity
                let (g_wx, g_prime_wx) = self.apply_nonlinearity(&wx);

                // Update: w_new = E{x * g(w^T x)} - E{g'(w^T x)} * w
                let mut w_new = Array1::zeros(n_dim);
                let mean_g_prime: f64 = g_prime_wx.iter().sum::<f64>() / n_samples as f64;

                for j in 0..n_dim {
                    let mut eg = 0.0;
                    for i in 0..n_samples {
                        eg += x[[i, j]] * g_wx[i];
                    }
                    eg /= n_samples as f64;
                    w_new[j] = eg - mean_g_prime * w[j];
                }

                // Orthogonalize against previous components (Gram-Schmidt)
                for q in 0..p {
                    let mut dot = 0.0;
                    for j in 0..n_dim {
                        dot += w_new[j] * w_all[[q, j]];
                    }
                    for j in 0..n_dim {
                        w_new[j] -= dot * w_all[[q, j]];
                    }
                }

                // Normalize
                let new_norm = w_new.iter().map(|&v| v * v).sum::<f64>().sqrt();
                if new_norm < EPSILON {
                    break;
                }
                w_new.mapv_inplace(|v| v / new_norm);

                // Check convergence (1 - |w_new . w_old| < tol)
                let dot: f64 = w_new.iter().zip(w_old.iter()).map(|(&a, &b)| a * b).sum();
                w = w_new;

                if 1.0 - dot.abs() < self.tol {
                    converged = true;
                    total_iter = total_iter.max(iter + 1);
                    break;
                }
            }

            if !converged {
                total_iter = self.max_iter;
            }

            for j in 0..n_dim {
                w_all[[p, j]] = w[j];
            }
        }

        Ok((w_all, total_iter))
    }

    /// FastICA symmetric approach: extract all components simultaneously
    fn fastica_symmetric(&self, x: &Array2<f64>, n_dim: usize) -> Result<(Array2<f64>, usize)> {
        let n_samples = x.shape()[0];

        // Initialize W randomly (deterministic)
        let mut w = Array2::zeros((self.n_components, n_dim));
        for i in 0..self.n_components {
            for j in 0..n_dim {
                let seed = (i * 997 + j * 13 + 42) as f64;
                w[[i, j]] = (seed * 0.618033988749895).fract() * 2.0 - 1.0;
            }
        }

        // Symmetric orthogonalization
        w = symmetric_orthogonalize(&w)?;

        let mut n_iter = 0;

        for iter in 0..self.max_iter {
            let w_old = w.clone();
            let mut w_new = Array2::zeros((self.n_components, n_dim));

            for p in 0..self.n_components {
                // Compute w_p^T * x
                let mut wx = Array1::zeros(n_samples);
                for i in 0..n_samples {
                    let mut dot = 0.0;
                    for j in 0..n_dim {
                        dot += w[[p, j]] * x[[i, j]];
                    }
                    wx[i] = dot;
                }

                let (g_wx, g_prime_wx) = self.apply_nonlinearity(&wx);

                let mean_g_prime: f64 = g_prime_wx.iter().sum::<f64>() / n_samples as f64;

                for j in 0..n_dim {
                    let mut eg = 0.0;
                    for i in 0..n_samples {
                        eg += x[[i, j]] * g_wx[i];
                    }
                    eg /= n_samples as f64;
                    w_new[[p, j]] = eg - mean_g_prime * w[[p, j]];
                }
            }

            // Symmetric orthogonalization
            w_new = symmetric_orthogonalize(&w_new)?;

            // Check convergence
            let mut max_change = 0.0_f64;
            for p in 0..self.n_components {
                let mut dot = 0.0;
                for j in 0..n_dim {
                    dot += w_new[[p, j]] * w_old[[p, j]];
                }
                let change = 1.0 - dot.abs();
                if change > max_change {
                    max_change = change;
                }
            }

            w = w_new;
            n_iter = iter + 1;

            if max_change < self.tol {
                break;
            }
        }

        Ok((w, n_iter))
    }

    /// Apply non-linearity function and its derivative
    fn apply_nonlinearity(&self, u: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let n = u.len();
        let mut g = Array1::zeros(n);
        let mut g_prime = Array1::zeros(n);

        match self.non_linearity {
            NonLinearity::LogCosh => {
                for i in 0..n {
                    let clamped = u[i].clamp(-20.0, 20.0);
                    g[i] = clamped.tanh();
                    g_prime[i] = 1.0 - g[i] * g[i];
                }
            }
            NonLinearity::Exp => {
                for i in 0..n {
                    let exp_val = (-u[i] * u[i] / 2.0).exp();
                    g[i] = u[i] * exp_val;
                    g_prime[i] = (1.0 - u[i] * u[i]) * exp_val;
                }
            }
            NonLinearity::Cube => {
                for i in 0..n {
                    g[i] = u[i] * u[i] * u[i];
                    g_prime[i] = 3.0 * u[i] * u[i];
                }
            }
        }

        (g, g_prime)
    }

    /// Apply unmixing matrix to centered data
    fn apply_unmixing(
        &self,
        x_centered: &Array2<f64>,
        unmixing: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let n_samples = x_centered.shape()[0];
        let n_features = x_centered.shape()[1];

        let mut result = Array2::zeros((n_samples, self.n_components));
        for i in 0..n_samples {
            for j in 0..self.n_components {
                let mut val = 0.0;
                for k in 0..n_features {
                    val += unmixing[[j, k]] * x_centered[[i, k]];
                }
                result[[i, j]] = val;
            }
        }

        Ok(result)
    }

    /// Transform data to independent components
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let unmixing = self
            .unmixing_
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("FastICA has not been fitted".to_string()))?;
        let mean = self
            .mean_
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("FastICA has not been fitted".to_string()))?;

        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];
        let n_features_in = self.n_features_in_.unwrap_or(0);

        if n_features != n_features_in {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, expected {}",
                n_features, n_features_in
            )));
        }

        // Center
        let mut x_centered = Array2::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                x_centered[[i, j]] = x[[i, j]] - mean[j];
            }
        }

        self.apply_unmixing(&x_centered, unmixing)
    }

    /// Fit and transform
    pub fn fit_transform(&mut self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Inverse transform: recover mixed signals from independent components
    pub fn inverse_transform(&self, sources: &Array2<f64>) -> Result<Array2<f64>> {
        let mixing = self
            .mixing_
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("FastICA has not been fitted".to_string()))?;
        let mean = self
            .mean_
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("FastICA has not been fitted".to_string()))?;

        let n_samples = sources.shape()[0];
        let n_components = sources.shape()[1];
        let n_features = mixing.shape()[0];

        if n_components != self.n_components {
            return Err(TransformError::InvalidInput(format!(
                "sources has {} components, expected {}",
                n_components, self.n_components
            )));
        }

        let mut result = Array2::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                let mut val = mean[j];
                for k in 0..n_components {
                    val += mixing[[j, k]] * sources[[i, k]];
                }
                result[[i, j]] = val;
            }
        }

        Ok(result)
    }

    /// Get the unmixing matrix (W)
    pub fn unmixing_matrix(&self) -> Option<&Array2<f64>> {
        self.unmixing_.as_ref()
    }

    /// Get the mixing matrix (A = W^-1)
    pub fn mixing_matrix(&self) -> Option<&Array2<f64>> {
        self.mixing_.as_ref()
    }

    /// Get the whitening matrix
    pub fn whitening_matrix(&self) -> Option<&Array2<f64>> {
        self.whitening_.as_ref()
    }

    /// Get the number of iterations taken
    pub fn n_iter(&self) -> Option<usize> {
        self.n_iter_
    }

    /// Get kurtosis of each component
    pub fn kurtosis(&self) -> Option<&Array1<f64>> {
        self.kurtosis_.as_ref()
    }
}

/// Symmetric orthogonalization: W = (W W^T)^{-1/2} W
fn symmetric_orthogonalize(w: &Array2<f64>) -> Result<Array2<f64>> {
    let n = w.shape()[0];
    let m = w.shape()[1];

    // Compute W * W^T
    let mut wwt = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0.0;
            for k in 0..m {
                dot += w[[i, k]] * w[[j, k]];
            }
            wwt[[i, j]] = dot;
        }
    }

    // SVD of W*W^T
    let (u, s, vt) = svd::<f64>(&wwt.view(), true, None).map_err(TransformError::LinalgError)?;

    // (W W^T)^{-1/2} = U * S^{-1/2} * V^T
    let mut inv_sqrt = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        if s[i] > EPSILON {
            let scale = 1.0 / s[i].sqrt();
            for j in 0..n {
                for k in 0..n {
                    inv_sqrt[[j, k]] += u[[j, i]] * scale * vt[[i, k]];
                }
            }
        }
    }

    // Result = inv_sqrt * W
    let mut result = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let mut val = 0.0;
            for k in 0..n {
                val += inv_sqrt[[i, k]] * w[[k, j]];
            }
            result[[i, j]] = val;
        }
    }

    Ok(result)
}

/// Matrix multiplication: C = A * B
fn mat_mul(a: &Array2<f64>, b: &Array2<f64>, m: usize, k: usize, n: usize) -> Array2<f64> {
    let mut c = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0;
            for l in 0..k {
                val += a[[i, l]] * b[[l, j]];
            }
            c[[i, j]] = val;
        }
    }
    c
}

/// Compute pseudo-inverse via SVD
fn pseudo_inverse(a: &Array2<f64>) -> Result<Array2<f64>> {
    let m = a.shape()[0];
    let n = a.shape()[1];

    let (u, s, vt) = svd::<f64>(&a.view(), true, None).map_err(TransformError::LinalgError)?;

    let min_dim = m.min(n);
    let mut result = Array2::<f64>::zeros((n, m));

    for i in 0..min_dim {
        if s[i] > EPSILON {
            let s_inv = 1.0 / s[i];
            for j in 0..n {
                for k in 0..m {
                    result[[j, k]] += vt[[i, j]] * s_inv * u[[k, i]];
                }
            }
        }
    }

    Ok(result)
}

/// Compute excess kurtosis for each column of a matrix
fn compute_kurtosis(x: &Array2<f64>) -> Result<Array1<f64>> {
    let n_samples = x.shape()[0];
    let n_cols = x.shape()[1];

    if n_samples < 4 {
        return Ok(Array1::zeros(n_cols));
    }

    let n = n_samples as f64;
    let mut kurtosis = Array1::zeros(n_cols);

    for j in 0..n_cols {
        let col_mean = x.column(j).iter().sum::<f64>() / n;

        let mut m2 = 0.0;
        let mut m4 = 0.0;
        for i in 0..n_samples {
            let d = x[[i, j]] - col_mean;
            let d2 = d * d;
            m2 += d2;
            m4 += d2 * d2;
        }
        m2 /= n;
        m4 /= n;

        if m2 > EPSILON {
            kurtosis[j] = m4 / (m2 * m2) - 3.0; // Excess kurtosis
        }
    }

    Ok(kurtosis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    fn make_mixed_signals(n: usize) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        // Create source signals
        let mut sources_data = Vec::new();
        for i in 0..n {
            let t = i as f64 / n as f64 * 10.0;
            sources_data.push(t.sin()); // Sinusoidal
            sources_data.push(((2.0 * t).sin()).signum()); // Square wave
        }
        let sources = Array::from_shape_vec((n, 2), sources_data).expect("test data");

        // Mixing matrix
        let mixing = Array::from_shape_vec((2, 2), vec![0.6, 0.4, 0.3, 0.7]).expect("test data");

        // Mixed signals: X = S * A^T
        let mut mixed = Array2::zeros((n, 2));
        for i in 0..n {
            for j in 0..2 {
                let mut val = 0.0;
                for k in 0..2 {
                    val += sources[[i, k]] * mixing[[k, j]];
                }
                mixed[[i, j]] = val;
            }
        }

        (sources, mixing, mixed)
    }

    #[test]
    fn test_fastica_deflation_logcosh() {
        let (_sources, _mixing, mixed) = make_mixed_signals(200);

        let mut ica = FastICA::new(2)
            .with_non_linearity(NonLinearity::LogCosh)
            .with_algorithm(IcaAlgorithm::Deflation);

        ica.fit(&mixed).expect("fit");
        let recovered = ica.transform(&mixed).expect("transform");
        assert_eq!(recovered.shape(), &[200, 2]);

        // Check that components are independent (low correlation)
        let corr = column_correlation(&recovered, 0, 1);
        assert!(
            corr.abs() < 0.3,
            "Components should be independent, correlation = {}",
            corr
        );
    }

    #[test]
    fn test_fastica_symmetric_exp() {
        let (_sources, _mixing, mixed) = make_mixed_signals(200);

        let mut ica = FastICA::new(2)
            .with_non_linearity(NonLinearity::Exp)
            .with_algorithm(IcaAlgorithm::Symmetric);

        ica.fit(&mixed).expect("fit");
        let recovered = ica.transform(&mixed).expect("transform");
        assert_eq!(recovered.shape(), &[200, 2]);
    }

    #[test]
    fn test_fastica_cube() {
        let (_sources, _mixing, mixed) = make_mixed_signals(200);

        let mut ica = FastICA::new(2).with_non_linearity(NonLinearity::Cube);

        ica.fit(&mixed).expect("fit");
        let recovered = ica.transform(&mixed).expect("transform");
        assert_eq!(recovered.shape(), &[200, 2]);
    }

    #[test]
    fn test_fastica_inverse_transform() {
        let (_sources, _mixing, mixed) = make_mixed_signals(100);

        let mut ica = FastICA::new(2);
        ica.fit(&mixed).expect("fit");
        let components = ica.transform(&mixed).expect("transform");
        let reconstructed = ica
            .inverse_transform(&components)
            .expect("inverse_transform");

        assert_eq!(reconstructed.shape(), &[100, 2]);

        // Reconstruction should be close to original
        let mut max_error = 0.0_f64;
        for i in 0..100 {
            for j in 0..2 {
                let err = (mixed[[i, j]] - reconstructed[[i, j]]).abs();
                if err > max_error {
                    max_error = err;
                }
            }
        }
        assert!(
            max_error < 0.5,
            "Max reconstruction error {} should be small",
            max_error
        );
    }

    #[test]
    fn test_fastica_kurtosis_ordering() {
        let (_sources, _mixing, mixed) = make_mixed_signals(200);

        let mut ica = FastICA::new(2);
        ica.fit(&mixed).expect("fit");

        let kurtosis = ica.kurtosis().expect("kurtosis");
        assert_eq!(kurtosis.len(), 2);

        // Components should be ordered by absolute kurtosis (descending)
        assert!(
            kurtosis[0].abs() >= kurtosis[1].abs(),
            "Components should be ordered by |kurtosis|: |{}| >= |{}|",
            kurtosis[0],
            kurtosis[1]
        );
    }

    #[test]
    fn test_fastica_matrices() {
        let (_sources, _mixing, mixed) = make_mixed_signals(100);

        let mut ica = FastICA::new(2);
        ica.fit(&mixed).expect("fit");

        assert!(ica.unmixing_matrix().is_some());
        assert!(ica.mixing_matrix().is_some());
        assert!(ica.whitening_matrix().is_some());
        assert!(ica.n_iter().is_some());
    }

    #[test]
    fn test_fastica_errors() {
        let x = Array::from_shape_vec((1, 2), vec![1.0, 2.0]).expect("test data");
        let mut ica = FastICA::new(2);
        assert!(ica.fit(&x).is_err()); // too few samples

        let x2 = Array::from_shape_vec(
            (5, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .expect("test data");
        let mut ica2 = FastICA::new(5);
        assert!(ica2.fit(&x2).is_err()); // n_components > n_features
    }

    #[test]
    fn test_fastica_not_fitted() {
        let x = Array::from_shape_vec(
            (5, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .expect("test data");
        let ica = FastICA::new(2);
        assert!(ica.transform(&x).is_err());
    }

    #[test]
    fn test_fastica_no_whiten() {
        let (_sources, _mixing, mixed) = make_mixed_signals(100);

        let mut ica = FastICA::new(2).with_whiten(false);
        ica.fit(&mixed).expect("fit");
        let recovered = ica.transform(&mixed).expect("transform");
        assert_eq!(recovered.shape(), &[100, 2]);
    }

    /// Compute Pearson correlation between two columns
    fn column_correlation(x: &Array2<f64>, col_a: usize, col_b: usize) -> f64 {
        let n = x.shape()[0] as f64;
        let a = x.column(col_a);
        let b = x.column(col_b);

        let a_mean = a.iter().sum::<f64>() / n;
        let b_mean = b.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut a_var = 0.0;
        let mut b_var = 0.0;

        for i in 0..x.shape()[0] {
            let da = a[i] - a_mean;
            let db = b[i] - b_mean;
            cov += da * db;
            a_var += da * da;
            b_var += db * db;
        }

        if a_var * b_var > EPSILON {
            cov / (a_var * b_var).sqrt()
        } else {
            0.0
        }
    }
}
