//! Gaussian Process Surrogate Model
//!
//! Full GP regression with Cholesky-based inference, marginal log-likelihood
//! computation, hyperparameter optimisation, and acquisition functions for
//! Bayesian optimisation.
//!
//! ## Key features
//!
//! - Multiple kernel functions (SE, Matern, Rational Quadratic)
//! - Cholesky-based O(n^3) inference
//! - Predictive mean **and** variance
//! - Marginal log-likelihood for model comparison / hyperparameter tuning
//! - Acquisition functions: EI, PI, UCB, LCB
//! - `suggest_next_point` for Bayesian optimisation loops
//!
//! ## References
//!
//! - Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for
//!   Machine Learning*. MIT Press.

use super::types::{AcquisitionFunction, GPSurrogateConfig, KernelType, SurrogateResult};
use crate::error::{InterpolateError, InterpolateResult};

// ---------------------------------------------------------------------------
// Gaussian Process Surrogate
// ---------------------------------------------------------------------------

/// Gaussian process surrogate model for interpolation and Bayesian optimisation.
///
/// After fitting, the model stores the Cholesky factor of the kernel matrix
/// and the weights `alpha = L^T \ (L \ y)` so that prediction is O(n) per
/// query point.
#[derive(Debug, Clone)]
pub struct GaussianProcessSurrogate {
    /// Training data locations (n x d).
    train_x: Vec<Vec<f64>>,
    /// Training target values (n,).
    train_y: Vec<f64>,
    /// Kernel specification.
    kernel: KernelType,
    /// Observation noise variance.
    noise: f64,
    /// Lower-triangular Cholesky factor L of (K + noise*I), row-major.
    chol_l: Vec<f64>,
    /// Weights alpha = L^T \ (L \ y).
    alpha: Vec<f64>,
    /// Number of training points.
    n: usize,
    /// Input dimensionality.
    dim: usize,
    /// Log-marginal-likelihood (cached after fitting).
    log_marginal_likelihood: f64,
}

impl GaussianProcessSurrogate {
    /// Fit a Gaussian process to training data.
    ///
    /// If `config.optimize_hyperparams` is true, the kernel hyperparameters
    /// are tuned by maximising the marginal log-likelihood via gradient-free
    /// coordinate search.
    pub fn fit(
        train_x: Vec<Vec<f64>>,
        train_y: Vec<f64>,
        config: GPSurrogateConfig,
    ) -> InterpolateResult<Self> {
        if train_x.is_empty() {
            return Err(InterpolateError::invalid_input(
                "GP surrogate requires at least one training point",
            ));
        }
        if train_y.len() != train_x.len() {
            return Err(InterpolateError::shape_mismatch(
                format!("{}", train_x.len()),
                format!("{}", train_y.len()),
                "train_y",
            ));
        }

        let dim = train_x[0].len();

        if config.optimize_hyperparams {
            Self::fit_with_optimization(train_x, train_y, dim, config)
        } else {
            Self::fit_internal(train_x, train_y, dim, config.kernel, config.noise)
        }
    }

    /// Internal fit with fixed hyperparameters.
    fn fit_internal(
        train_x: Vec<Vec<f64>>,
        train_y: Vec<f64>,
        dim: usize,
        kernel: KernelType,
        noise: f64,
    ) -> InterpolateResult<Self> {
        let n = train_x.len();

        // Build kernel matrix K + noise*I
        let mut k_mat = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                k_mat[i * n + j] = kernel.evaluate(&train_x[i], &train_x[j]);
            }
            k_mat[i * n + i] += noise;
        }

        // Cholesky factorisation K = L * L^T
        let chol_l = cholesky_factor(&k_mat, n)?;

        // alpha = L^T \ (L \ y)
        let tmp = forward_solve(&chol_l, &train_y, n);
        let alpha = backward_solve_transpose(&chol_l, &tmp, n);

        // Log-marginal-likelihood:
        // -0.5 * (y^T alpha + sum(log(diag(L))) * 2 + n * log(2*pi))
        let yt_alpha: f64 = train_y.iter().zip(alpha.iter()).map(|(y, a)| y * a).sum();
        let log_det: f64 = (0..n).map(|i| chol_l[i * n + i].ln()).sum::<f64>() * 2.0;
        let log_ml = -0.5 * (yt_alpha + log_det + n as f64 * (2.0 * std::f64::consts::PI).ln());

        Ok(Self {
            train_x,
            train_y,
            kernel,
            noise,
            chol_l,
            alpha,
            n,
            dim,
            log_marginal_likelihood: log_ml,
        })
    }

    /// Fit with hyperparameter optimisation via coordinate search.
    fn fit_with_optimization(
        train_x: Vec<Vec<f64>>,
        train_y: Vec<f64>,
        dim: usize,
        config: GPSurrogateConfig,
    ) -> InterpolateResult<Self> {
        let mut best_lml = f64::NEG_INFINITY;
        let mut best_kernel = config.kernel;
        let mut best_noise = config.noise;

        // Grid of lengthscales and variances to try
        let ls_candidates = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0];
        let var_candidates = [0.1, 0.5, 1.0, 2.0, 5.0];
        let noise_candidates = [1e-8, 1e-6, 1e-4, 1e-2];

        for &ls in &ls_candidates {
            for &var in &var_candidates {
                #[allow(unreachable_patterns)]
                let candidate_kernel = match config.kernel {
                    KernelType::SquaredExponential { .. } => KernelType::SquaredExponential {
                        lengthscale: ls,
                        variance: var,
                    },
                    KernelType::Matern { nu, .. } => KernelType::Matern {
                        nu,
                        lengthscale: ls,
                        variance: var,
                    },
                    KernelType::RationalQuadratic { alpha, .. } => KernelType::RationalQuadratic {
                        alpha,
                        lengthscale: ls,
                        variance: var,
                    },
                    _ => config.kernel,
                };

                for &ns in &noise_candidates {
                    if let Ok(gp) = Self::fit_internal(
                        train_x.clone(),
                        train_y.clone(),
                        dim,
                        candidate_kernel,
                        ns,
                    ) {
                        if gp.log_marginal_likelihood > best_lml {
                            best_lml = gp.log_marginal_likelihood;
                            best_kernel = candidate_kernel;
                            best_noise = ns;
                        }
                    }
                }
            }
        }

        Self::fit_internal(train_x, train_y, dim, best_kernel, best_noise)
    }

    /// Predict at a single query point.
    ///
    /// Returns `(mean, variance)`.
    pub fn predict(&self, x: &[f64]) -> InterpolateResult<(f64, f64)> {
        if x.len() != self.dim {
            return Err(InterpolateError::DimensionMismatch(format!(
                "expected dim {}, got {}",
                self.dim,
                x.len()
            )));
        }

        // k* = [k(x, x_1), ..., k(x, x_n)]
        let k_star: Vec<f64> = self
            .train_x
            .iter()
            .map(|xi| self.kernel.evaluate(x, xi))
            .collect();

        // Predictive mean: mu* = k*^T alpha
        let mean: f64 = k_star
            .iter()
            .zip(self.alpha.iter())
            .map(|(k, a)| k * a)
            .sum();

        // Predictive variance: sigma*^2 = k(x,x) - v^T v where v = L \ k*
        let k_xx = self.kernel.evaluate(x, x) + self.noise;
        let v = forward_solve(&self.chol_l, &k_star, self.n);
        let v_sq: f64 = v.iter().map(|vi| vi * vi).sum();
        let variance = (k_xx - v_sq).max(0.0);

        Ok((mean, variance))
    }

    /// Predict at multiple query points.
    pub fn predict_batch(&self, points: &[Vec<f64>]) -> InterpolateResult<SurrogateResult> {
        let mut predictions = Vec::with_capacity(points.len());
        let mut variances = Vec::with_capacity(points.len());

        for p in points {
            let (mean, var) = self.predict(p)?;
            predictions.push(mean);
            variances.push(var);
        }

        let hyperparameters = self.hyperparameter_vector();

        Ok(SurrogateResult {
            predictions,
            variances,
            hyperparameters,
        })
    }

    /// Return the cached log-marginal-likelihood.
    pub fn log_marginal_likelihood(&self) -> f64 {
        self.log_marginal_likelihood
    }

    /// Evaluate an acquisition function at a query point.
    ///
    /// `f_best` is the best (minimum) observed value so far (for minimisation).
    pub fn acquisition(
        &self,
        x: &[f64],
        acq: AcquisitionFunction,
        f_best: f64,
    ) -> InterpolateResult<f64> {
        let (mean, variance) = self.predict(x)?;
        let sigma = variance.sqrt();

        if sigma < 1e-15 {
            return Ok(0.0);
        }

        #[allow(unreachable_patterns)]
        match acq {
            AcquisitionFunction::EI => {
                // Expected Improvement (for minimisation)
                let z = (f_best - mean) / sigma;
                let ei = sigma * (z * standard_normal_cdf(z) + standard_normal_pdf(z));
                Ok(ei.max(0.0))
            }
            AcquisitionFunction::PI => {
                // Probability of Improvement
                let z = (f_best - mean) / sigma;
                Ok(standard_normal_cdf(z))
            }
            AcquisitionFunction::UCB(kappa) => {
                // Upper Confidence Bound (for maximisation: mean + kappa * sigma)
                Ok(mean + kappa * sigma)
            }
            AcquisitionFunction::LCB(kappa) => {
                // Lower Confidence Bound (for minimisation: mean - kappa * sigma)
                Ok(mean - kappa * sigma)
            }
            _ => Err(InterpolateError::NotImplemented(
                "unknown acquisition function variant".into(),
            )),
        }
    }

    /// Suggest the next evaluation point by maximising the acquisition function.
    ///
    /// Searches over a grid within `bounds` (d x 2 array: [lower, upper] per dim).
    /// `n_candidates` controls the grid density.
    pub fn suggest_next_point(
        &self,
        bounds: &[(f64, f64)],
        acq: AcquisitionFunction,
        f_best: f64,
        n_candidates: usize,
    ) -> InterpolateResult<Vec<f64>> {
        if bounds.len() != self.dim {
            return Err(InterpolateError::DimensionMismatch(format!(
                "bounds dim {} != data dim {}",
                bounds.len(),
                self.dim
            )));
        }

        let n_per_dim = (n_candidates as f64).powf(1.0 / self.dim as f64).ceil() as usize;
        let n_per_dim = n_per_dim.max(2);

        let mut best_acq_val = f64::NEG_INFINITY;
        let mut best_point = vec![0.0; self.dim];

        // For EI/PI we maximise; for LCB we minimise (negate).
        let negate = matches!(acq, AcquisitionFunction::LCB(_));

        // Generate candidates via Latin hypercube-like grid
        Self::grid_search_recursive(
            bounds,
            n_per_dim,
            &mut vec![0.0; self.dim],
            0,
            &mut |candidate| {
                if let Ok(val) = self.acquisition(candidate, acq, f_best) {
                    let val = if negate { -val } else { val };
                    if val > best_acq_val {
                        best_acq_val = val;
                        best_point = candidate.to_vec();
                    }
                }
            },
        );

        Ok(best_point)
    }

    /// Recursive grid search helper.
    fn grid_search_recursive(
        bounds: &[(f64, f64)],
        n_per_dim: usize,
        current: &mut Vec<f64>,
        dim_idx: usize,
        callback: &mut dyn FnMut(&[f64]),
    ) {
        if dim_idx >= bounds.len() {
            callback(current);
            return;
        }

        let (lo, hi) = bounds[dim_idx];
        for i in 0..n_per_dim {
            let t = if n_per_dim <= 1 {
                0.5
            } else {
                i as f64 / (n_per_dim - 1) as f64
            };
            current[dim_idx] = lo + t * (hi - lo);
            Self::grid_search_recursive(bounds, n_per_dim, current, dim_idx + 1, callback);
        }
    }

    /// Return kernel and noise as a flat vector.
    pub fn hyperparameter_vector(&self) -> Vec<f64> {
        #[allow(unreachable_patterns)]
        match self.kernel {
            KernelType::SquaredExponential {
                lengthscale,
                variance,
            } => vec![lengthscale, variance, self.noise],
            KernelType::Matern {
                nu,
                lengthscale,
                variance,
            } => vec![nu, lengthscale, variance, self.noise],
            KernelType::RationalQuadratic {
                alpha,
                lengthscale,
                variance,
            } => vec![alpha, lengthscale, variance, self.noise],
            _ => vec![self.noise],
        }
    }

    /// Access the fitted kernel.
    pub fn kernel(&self) -> &KernelType {
        &self.kernel
    }

    /// Access the noise variance.
    pub fn noise(&self) -> f64 {
        self.noise
    }

    /// Number of training points.
    pub fn n_train(&self) -> usize {
        self.n
    }

    /// Input dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// Linear algebra helpers: Cholesky, forward/backward solve
// ---------------------------------------------------------------------------

/// Compute the Cholesky factorisation L of a symmetric positive-definite
/// matrix A (row-major, n x n) such that A = L * L^T.
fn cholesky_factor(a: &[f64], n: usize) -> InterpolateResult<Vec<f64>> {
    let mut l = vec![0.0; n * n];

    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0;
            for k in 0..j {
                s += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let diag = a[i * n + i] - s;
                if diag <= 0.0 {
                    return Err(InterpolateError::ComputationError(format!(
                        "Cholesky failed: matrix not positive definite at index {i} (diag={diag:.3e})"
                    )));
                }
                l[i * n + j] = diag.sqrt();
            } else {
                let denom = l[j * n + j];
                if denom.abs() < 1e-30 {
                    return Err(InterpolateError::ComputationError(
                        "Cholesky failed: near-zero diagonal element".into(),
                    ));
                }
                l[i * n + j] = (a[i * n + j] - s) / denom;
            }
        }
    }

    Ok(l)
}

/// Forward solve L * x = b where L is lower triangular (row-major, n x n).
fn forward_solve(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * x[j];
        }
        let diag = l[i * n + i];
        x[i] = if diag.abs() > 1e-30 { s / diag } else { 0.0 };
    }
    x
}

/// Backward solve L^T * x = b where L is lower triangular (row-major, n x n).
fn backward_solve_transpose(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= l[j * n + i] * x[j];
        }
        let diag = l[i * n + i];
        x[i] = if diag.abs() > 1e-30 { s / diag } else { 0.0 };
    }
    x
}

// ---------------------------------------------------------------------------
// Standard normal helpers
// ---------------------------------------------------------------------------

/// Standard normal PDF: phi(z) = (1/sqrt(2*pi)) * exp(-z^2/2).
fn standard_normal_pdf(z: f64) -> f64 {
    let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    inv_sqrt_2pi * (-0.5 * z * z).exp()
}

/// Standard normal CDF approximation (Abramowitz & Stegun).
fn standard_normal_cdf(z: f64) -> f64 {
    // Use the error function: Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2))
}

/// Approximation of the error function via Horner form (max error ~1.5e-7).
fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let p = 0.3275911;
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;

    let t = 1.0 / (1.0 + p * x);
    let poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
    let result = 1.0 - poly * (-x * x).exp();
    sign * result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_1d_data() -> (Vec<Vec<f64>>, Vec<f64>) {
        let xs: Vec<f64> = (0..6).map(|i| i as f64 * 0.5).collect();
        let pts: Vec<Vec<f64>> = xs.iter().map(|&x| vec![x]).collect();
        let vals: Vec<f64> = xs.iter().map(|&x| x.sin()).collect();
        (pts, vals)
    }

    #[test]
    fn test_gp_interpolates_training_points_zero_noise() {
        let (pts, vals) = make_1d_data();
        let config = GPSurrogateConfig {
            kernel: KernelType::SquaredExponential {
                lengthscale: 1.0,
                variance: 1.0,
            },
            noise: 1e-10,
            optimize_hyperparams: false,
            ..GPSurrogateConfig::default()
        };

        let gp =
            GaussianProcessSurrogate::fit(pts.clone(), vals.clone(), config).expect("test: fit");

        for (p, &v) in pts.iter().zip(vals.iter()) {
            let (mean, _var) = gp.predict(p).expect("test: predict");
            assert!(
                (mean - v).abs() < 1e-4,
                "at {:?}: expected {v}, got {mean}",
                p
            );
        }
    }

    #[test]
    fn test_predictive_variance_near_zero_at_training() {
        let (pts, vals) = make_1d_data();
        let config = GPSurrogateConfig {
            noise: 1e-10,
            ..GPSurrogateConfig::default()
        };

        let gp = GaussianProcessSurrogate::fit(pts.clone(), vals, config).expect("test: fit");

        for p in &pts {
            let (_mean, var) = gp.predict(p).expect("test: predict");
            assert!(
                var < 1e-4,
                "variance at training point {:?} should be near zero, got {var}",
                p
            );
        }
    }

    #[test]
    fn test_predictive_variance_increases_away() {
        let (pts, vals) = make_1d_data();
        let config = GPSurrogateConfig {
            noise: 1e-10,
            ..GPSurrogateConfig::default()
        };

        let gp = GaussianProcessSurrogate::fit(pts, vals, config).expect("test: fit");

        let (_, var_near) = gp.predict(&[1.0]).expect("test: near");
        let (_, var_far) = gp.predict(&[10.0]).expect("test: far");

        assert!(
            var_far > var_near,
            "variance far from data ({var_far}) should exceed near ({var_near})"
        );
    }

    #[test]
    fn test_marginal_likelihood_is_finite() {
        let (pts, vals) = make_1d_data();
        let config = GPSurrogateConfig::default();
        let gp = GaussianProcessSurrogate::fit(pts, vals, config).expect("test: fit");

        let lml = gp.log_marginal_likelihood();
        assert!(
            lml.is_finite(),
            "log-marginal-likelihood should be finite: {lml}"
        );
    }

    #[test]
    fn test_ei_is_non_negative() {
        let (pts, vals) = make_1d_data();
        let config = GPSurrogateConfig {
            noise: 1e-6,
            ..GPSurrogateConfig::default()
        };
        let gp = GaussianProcessSurrogate::fit(pts, vals.clone(), config).expect("test: fit");

        let f_best = vals.iter().cloned().fold(f64::INFINITY, f64::min);

        for x in [0.25, 0.75, 1.25, 3.0, 5.0] {
            let ei = gp
                .acquisition(&[x], AcquisitionFunction::EI, f_best)
                .expect("test: acquisition");
            assert!(ei >= 0.0, "EI at {x} should be non-negative, got {ei}");
        }
    }

    #[test]
    fn test_ei_zero_at_best_point() {
        // At a training point with near-zero variance, EI should be ~0
        let (pts, vals) = make_1d_data();
        let config = GPSurrogateConfig {
            noise: 1e-10,
            ..GPSurrogateConfig::default()
        };
        let gp =
            GaussianProcessSurrogate::fit(pts.clone(), vals.clone(), config).expect("test: fit");

        let f_best = vals.iter().cloned().fold(f64::INFINITY, f64::min);

        // Find the training point that achieves f_best
        let best_idx = vals
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let ei = gp
            .acquisition(&pts[best_idx], AcquisitionFunction::EI, f_best)
            .expect("test: acquisition at best");
        assert!(
            ei < 1e-4,
            "EI at best training point should be ~0, got {ei}"
        );
    }

    #[test]
    fn test_ucb_exceeds_mean() {
        let (pts, vals) = make_1d_data();
        let config = GPSurrogateConfig {
            noise: 1e-6,
            ..GPSurrogateConfig::default()
        };
        let gp = GaussianProcessSurrogate::fit(pts, vals, config).expect("test: fit");

        let x = &[1.5];
        let (mean, _) = gp.predict(x).expect("test: predict");
        let ucb = gp
            .acquisition(x, AcquisitionFunction::UCB(2.0), 0.0)
            .expect("test: ucb");

        assert!(ucb >= mean, "UCB ({ucb}) should >= mean ({mean})");
    }

    #[test]
    fn test_hyperparameter_optimization_improves_likelihood() {
        let (pts, vals) = make_1d_data();

        // Without optimisation (intentionally bad hyperparams)
        let config_bad = GPSurrogateConfig {
            kernel: KernelType::SquaredExponential {
                lengthscale: 100.0,
                variance: 0.01,
            },
            noise: 1e-6,
            optimize_hyperparams: false,
            ..GPSurrogateConfig::default()
        };
        let gp_bad = GaussianProcessSurrogate::fit(pts.clone(), vals.clone(), config_bad)
            .expect("test: fit bad");

        // With optimisation
        let config_opt = GPSurrogateConfig {
            kernel: KernelType::SquaredExponential {
                lengthscale: 100.0,
                variance: 0.01,
            },
            noise: 1e-6,
            optimize_hyperparams: true,
            ..GPSurrogateConfig::default()
        };
        let gp_opt = GaussianProcessSurrogate::fit(pts, vals, config_opt).expect("test: fit opt");

        assert!(
            gp_opt.log_marginal_likelihood() >= gp_bad.log_marginal_likelihood(),
            "optimised LML ({}) should >= bad LML ({})",
            gp_opt.log_marginal_likelihood(),
            gp_bad.log_marginal_likelihood()
        );
    }

    #[test]
    fn test_suggest_next_point_in_bounds() {
        let pts = vec![vec![0.0], vec![1.0], vec![2.0]];
        let vals = vec![1.0, 0.5, 0.8];
        let config = GPSurrogateConfig {
            noise: 1e-6,
            ..GPSurrogateConfig::default()
        };
        let gp = GaussianProcessSurrogate::fit(pts, vals.clone(), config).expect("test: fit");

        let f_best = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let bounds = vec![(0.0, 3.0)];
        let next = gp
            .suggest_next_point(&bounds, AcquisitionFunction::EI, f_best, 50)
            .expect("test: suggest");

        assert_eq!(next.len(), 1);
        assert!(
            next[0] >= 0.0 && next[0] <= 3.0,
            "suggested point {:.3} should be within bounds",
            next[0]
        );
    }

    #[test]
    fn test_predict_batch() {
        let (pts, vals) = make_1d_data();
        let config = GPSurrogateConfig::default();
        let gp = GaussianProcessSurrogate::fit(pts, vals, config).expect("test: fit");

        let query = vec![vec![0.25], vec![1.5], vec![3.0]];
        let result = gp.predict_batch(&query).expect("test: batch");

        assert_eq!(result.predictions.len(), 3);
        assert_eq!(result.variances.len(), 3);
        for v in &result.variances {
            assert!(*v >= 0.0);
        }
    }

    #[test]
    fn test_matern_kernel() {
        let pts = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let vals = vec![0.0, 1.0, 0.0, -1.0];
        let config = GPSurrogateConfig {
            kernel: KernelType::Matern {
                nu: 2.5,
                lengthscale: 1.0,
                variance: 1.0,
            },
            noise: 1e-10,
            ..GPSurrogateConfig::default()
        };

        let gp = GaussianProcessSurrogate::fit(pts.clone(), vals.clone(), config)
            .expect("test: fit matern");

        for (p, &v) in pts.iter().zip(vals.iter()) {
            let (mean, _) = gp.predict(p).expect("test: predict matern");
            assert!(
                (mean - v).abs() < 1e-3,
                "matern at {:?}: expected {v}, got {mean}",
                p
            );
        }
    }

    #[test]
    fn test_rational_quadratic_kernel() {
        let pts = vec![vec![0.0], vec![1.0], vec![2.0]];
        let vals = vec![0.0, 1.0, 0.5];
        let config = GPSurrogateConfig {
            kernel: KernelType::RationalQuadratic {
                alpha: 1.0,
                lengthscale: 1.0,
                variance: 1.0,
            },
            noise: 1e-10,
            ..GPSurrogateConfig::default()
        };

        let gp =
            GaussianProcessSurrogate::fit(pts.clone(), vals.clone(), config).expect("test: fit rq");

        for (p, &v) in pts.iter().zip(vals.iter()) {
            let (mean, _) = gp.predict(p).expect("test: predict rq");
            assert!(
                (mean - v).abs() < 1e-3,
                "rq at {:?}: expected {v}, got {mean}",
                p
            );
        }
    }

    #[test]
    fn test_empty_training_error() {
        let config = GPSurrogateConfig::default();
        let result = GaussianProcessSurrogate::fit(vec![], vec![], config);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch_predict() {
        let pts = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let vals = vec![0.0, 1.0];
        let config = GPSurrogateConfig {
            noise: 1e-6,
            ..GPSurrogateConfig::default()
        };
        let gp = GaussianProcessSurrogate::fit(pts, vals, config).expect("test: fit");
        let result = gp.predict(&[0.5]); // wrong dim
        assert!(result.is_err());
    }

    #[test]
    fn test_2d_gp() {
        let pts = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let vals = vec![0.0, 1.0, 1.0, 2.0]; // f(x,y) = x + y
        let config = GPSurrogateConfig {
            noise: 1e-10,
            ..GPSurrogateConfig::default()
        };

        let gp = GaussianProcessSurrogate::fit(pts, vals, config).expect("test: fit 2d");

        let (mean, _) = gp.predict(&[0.5, 0.5]).expect("test: predict 2d");
        assert!((mean - 1.0).abs() < 0.3, "2d GP: expected ~1.0, got {mean}");
    }

    #[test]
    fn test_pi_acquisition() {
        let (pts, vals) = make_1d_data();
        let config = GPSurrogateConfig {
            noise: 1e-6,
            ..GPSurrogateConfig::default()
        };
        let gp = GaussianProcessSurrogate::fit(pts, vals.clone(), config).expect("test: fit");

        let f_best = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let pi = gp
            .acquisition(&[1.5], AcquisitionFunction::PI, f_best)
            .expect("test: pi");
        assert!(pi >= 0.0 && pi <= 1.0, "PI should be in [0,1], got {pi}");
    }
}
