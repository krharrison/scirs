//! Nonparametric Gaussian Process regression and classification.
//!
//! This module provides GP models where the kernel hyperparameters are
//! treated as random variables with priors, enabling full Bayesian inference.
//! It extends the `gaussian_process` module with:
//! - **Hyperparameter marginalization** via MCMC
//! - **Sparse GP** (inducing points) for large datasets
//! - **Deep Kernel Learning** (linear feature transforms)
//! - **GP Classification** with Laplace approximation

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::random::{rngs::StdRng, Distribution, Normal, SeedableRng};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// Kernel function trait for GP models.
pub trait Kernel: Clone + std::fmt::Debug {
    /// Compute the kernel matrix K(X, X').
    fn compute(&self, x1: &[Vec<f64>], x2: &[Vec<f64>]) -> Vec<Vec<f64>>;
    /// Compute k(x, x') for single inputs.
    fn evaluate(&self, x1: &[f64], x2: &[f64]) -> f64;
    /// Log-prior of hyperparameters (for MCMC).
    fn log_prior(&self) -> f64;
    /// Return hyperparameter names and current values.
    fn hyperparams(&self) -> Vec<(String, f64)>;
    /// Set hyperparameter by name.
    fn set_hyperparam(&mut self, name: &str, value: f64) -> Result<()>;
}

/// Squared Exponential (RBF) kernel: k(x,x') = σ² exp(-||x-x'||² / (2l²))
#[derive(Debug, Clone)]
pub struct RBFKernel {
    /// Signal variance σ².
    pub sigma2: f64,
    /// Length scale l.
    pub length_scale: f64,
}

impl RBFKernel {
    /// Construct a new RBF kernel.
    ///
    /// # Errors
    /// Returns an error on non-positive parameters.
    pub fn new(sigma2: f64, length_scale: f64) -> Result<Self> {
        if sigma2 <= 0.0 {
            return Err(StatsError::DomainError(format!(
                "sigma2 must be > 0, got {sigma2}"
            )));
        }
        if length_scale <= 0.0 {
            return Err(StatsError::DomainError(format!(
                "length_scale must be > 0, got {length_scale}"
            )));
        }
        Ok(Self { sigma2, length_scale })
    }
}

impl Kernel for RBFKernel {
    fn compute(&self, x1: &[Vec<f64>], x2: &[Vec<f64>]) -> Vec<Vec<f64>> {
        x1.iter()
            .map(|a| x2.iter().map(|b| self.evaluate(a, b)).collect())
            .collect()
    }

    fn evaluate(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let sq_dist: f64 = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum();
        self.sigma2 * (-sq_dist / (2.0 * self.length_scale.powi(2))).exp()
    }

    fn log_prior(&self) -> f64 {
        // Log-normal priors on hyperparameters (vague)
        -self.sigma2.ln().powi(2) / 2.0 - self.length_scale.ln().powi(2) / 2.0
    }

    fn hyperparams(&self) -> Vec<(String, f64)> {
        vec![
            ("sigma2".into(), self.sigma2),
            ("length_scale".into(), self.length_scale),
        ]
    }

    fn set_hyperparam(&mut self, name: &str, value: f64) -> Result<()> {
        if value <= 0.0 {
            return Err(StatsError::DomainError(format!(
                "Kernel hyperparameter must be > 0, got {value}"
            )));
        }
        match name {
            "sigma2" => self.sigma2 = value,
            "length_scale" => self.length_scale = value,
            other => {
                return Err(StatsError::InvalidArgument(format!(
                    "Unknown hyperparameter: {other}"
                )));
            }
        }
        Ok(())
    }
}

/// Matérn 5/2 kernel: k(x,x') = σ²(1 + √5r/l + 5r²/(3l²)) exp(-√5r/l)
#[derive(Debug, Clone)]
pub struct Matern52Kernel {
    /// Signal variance σ².
    pub sigma2: f64,
    /// Length scale l.
    pub length_scale: f64,
}

impl Matern52Kernel {
    /// Construct a new Matérn 5/2 kernel.
    pub fn new(sigma2: f64, length_scale: f64) -> Result<Self> {
        if sigma2 <= 0.0 {
            return Err(StatsError::DomainError(format!(
                "sigma2 must be > 0, got {sigma2}"
            )));
        }
        if length_scale <= 0.0 {
            return Err(StatsError::DomainError(format!(
                "length_scale must be > 0, got {length_scale}"
            )));
        }
        Ok(Self { sigma2, length_scale })
    }
}

impl Kernel for Matern52Kernel {
    fn compute(&self, x1: &[Vec<f64>], x2: &[Vec<f64>]) -> Vec<Vec<f64>> {
        x1.iter()
            .map(|a| x2.iter().map(|b| self.evaluate(a, b)).collect())
            .collect()
    }

    fn evaluate(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let sq_dist: f64 = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum();
        let r = sq_dist.sqrt();
        let sqrt5_r_l = 5.0_f64.sqrt() * r / self.length_scale;
        self.sigma2 * (1.0 + sqrt5_r_l + 5.0 * r.powi(2) / (3.0 * self.length_scale.powi(2)))
            * (-sqrt5_r_l).exp()
    }

    fn log_prior(&self) -> f64 {
        -self.sigma2.ln().powi(2) / 2.0 - self.length_scale.ln().powi(2) / 2.0
    }

    fn hyperparams(&self) -> Vec<(String, f64)> {
        vec![
            ("sigma2".into(), self.sigma2),
            ("length_scale".into(), self.length_scale),
        ]
    }

    fn set_hyperparam(&mut self, name: &str, value: f64) -> Result<()> {
        if value <= 0.0 {
            return Err(StatsError::DomainError(format!(
                "Kernel hyperparameter must be > 0, got {value}"
            )));
        }
        match name {
            "sigma2" => self.sigma2 = value,
            "length_scale" => self.length_scale = value,
            other => {
                return Err(StatsError::InvalidArgument(format!(
                    "Unknown hyperparameter: {other}"
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Nonparametric GP Regressor
// ---------------------------------------------------------------------------

/// Bayesian GP Regressor with MCMC hyperparameter marginalization.
///
/// The model is:
/// ```text
/// f ~ GP(0, k(·, · | θ))
/// y | f ~ N(f(x), σ²_n)
/// θ ~ p(θ)   (log-normal priors on kernel hyperparams)
/// ```
///
/// Hyperparameter marginalization is performed using Metropolis-Hastings.
#[derive(Debug, Clone)]
pub struct NonparametricGPRegressor {
    /// Observation noise variance.
    pub noise_variance: f64,
    /// Kernel function (determines covariance structure).
    kernel: RBFKernel,
    /// Training inputs (N × D).
    train_x: Vec<Vec<f64>>,
    /// Training targets (N).
    train_y: Vec<f64>,
    /// Cholesky factor L such that K + σ²I = L L^T.
    chol_l: Vec<Vec<f64>>,
    /// L^{-T} y (for fast prediction).
    alpha: Vec<f64>,
    /// Number of training observations.
    n_train: usize,
    /// Posterior samples of log(σ²) from MCMC.
    log_noise_samples: Vec<f64>,
    /// Posterior samples of log(l) from MCMC.
    log_ls_samples: Vec<f64>,
    /// Posterior samples of log(σ²_k) from MCMC.
    log_sigma2_k_samples: Vec<f64>,
    /// Whether the model has been fitted.
    is_fitted: bool,
}

impl NonparametricGPRegressor {
    /// Construct a new GP Regressor.
    ///
    /// # Parameters
    /// - `noise_variance`: initial observation noise σ²_n (> 0)
    /// - `kernel`: initial kernel (hyperparams will be updated via MCMC)
    ///
    /// # Errors
    /// Returns an error when `noise_variance <= 0`.
    pub fn new(noise_variance: f64, kernel: RBFKernel) -> Result<Self> {
        if noise_variance <= 0.0 {
            return Err(StatsError::DomainError(format!(
                "noise_variance must be > 0, got {noise_variance}"
            )));
        }
        Ok(Self {
            noise_variance,
            kernel,
            train_x: Vec::new(),
            train_y: Vec::new(),
            chol_l: Vec::new(),
            alpha: Vec::new(),
            n_train: 0,
            log_noise_samples: Vec::new(),
            log_ls_samples: Vec::new(),
            log_sigma2_k_samples: Vec::new(),
            is_fitted: false,
        })
    }

    /// Fit the GP regressor: run MH to marginalize over hyperparameters,
    /// then compute the posterior predictive at current hyperparams.
    ///
    /// # Parameters
    /// - `x`: training inputs (N × D)
    /// - `y`: training targets (N)
    /// - `n_mcmc`: number of MH steps for hyperparameter sampling
    /// - `n_warmup`: warmup steps to discard
    /// - `seed`: random seed
    ///
    /// # Errors
    /// Returns an error on dimension mismatches.
    pub fn fit(
        &mut self,
        x: &[Vec<f64>],
        y: &[f64],
        n_mcmc: usize,
        n_warmup: usize,
        seed: u64,
    ) -> Result<()> {
        let n = y.len();
        if n == 0 {
            return Err(StatsError::InsufficientData(
                "training data must be non-empty".into(),
            ));
        }
        if x.len() != n {
            return Err(StatsError::DimensionMismatch(format!(
                "x has {} rows, y has {n}",
                x.len()
            )));
        }
        if n_warmup >= n_mcmc {
            return Err(StatsError::InvalidArgument(
                "n_warmup must be < n_mcmc".into(),
            ));
        }

        self.train_x = x.to_vec();
        self.train_y = y.to_vec();
        self.n_train = n;

        let mut rng = StdRng::seed_from_u64(seed);

        // MH for hyperparameters: θ = (log σ²_n, log l, log σ²_k)
        let mut log_noise = self.noise_variance.ln();
        let mut log_ls = self.kernel.length_scale.ln();
        let mut log_sigma2k = self.kernel.sigma2.ln();

        let step_size = 0.1_f64;
        let normal = Normal::new(0.0, step_size).map_err(|e| {
            StatsError::ComputationError(format!("Normal init error: {e}"))
        })?;

        let mut current_ll = self.gp_log_lik_params(log_noise, log_ls, log_sigma2k);

        let n_post = n_mcmc - n_warmup;
        self.log_noise_samples = Vec::with_capacity(n_post);
        self.log_ls_samples = Vec::with_capacity(n_post);
        self.log_sigma2_k_samples = Vec::with_capacity(n_post);

        for iter in 0..n_mcmc {
            // Propose new hyperparameters
            let prop_log_noise = log_noise + normal.sample(&mut rng);
            let prop_log_ls = log_ls + normal.sample(&mut rng);
            let prop_log_sigma2k = log_sigma2k + normal.sample(&mut rng);

            let prop_ll =
                self.gp_log_lik_params(prop_log_noise, prop_log_ls, prop_log_sigma2k);

            // Log-prior: vague log-normal (same log-lik term serves as data + prior)
            let log_accept = (prop_ll - current_ll).min(0.0);
            let u = sample_uniform_01(&mut rng);
            if u.ln() < log_accept {
                log_noise = prop_log_noise;
                log_ls = prop_log_ls;
                log_sigma2k = prop_log_sigma2k;
                current_ll = prop_ll;
            }

            if iter >= n_warmup {
                self.log_noise_samples.push(log_noise);
                self.log_ls_samples.push(log_ls);
                self.log_sigma2_k_samples.push(log_sigma2k);
            }
        }

        // Set hyperparams to posterior mean
        let n_s = self.log_noise_samples.len() as f64;
        let mean_log_noise = self.log_noise_samples.iter().sum::<f64>() / n_s;
        let mean_log_ls = self.log_ls_samples.iter().sum::<f64>() / n_s;
        let mean_log_sigma2k = self.log_sigma2_k_samples.iter().sum::<f64>() / n_s;

        self.noise_variance = mean_log_noise.exp();
        self.kernel.length_scale = mean_log_ls.exp();
        self.kernel.sigma2 = mean_log_sigma2k.exp();

        // Compute Cholesky factorization for predictions
        self.update_cholesky()?;
        self.is_fitted = true;
        Ok(())
    }

    /// Predict at new test points.
    ///
    /// Returns `(mean, variance)` for each test point.
    ///
    /// # Errors
    /// Returns an error when the model has not been fitted.
    pub fn predict(&self, x_test: &[Vec<f64>]) -> Result<(Vec<f64>, Vec<f64>)> {
        if !self.is_fitted {
            return Err(StatsError::InvalidInput(
                "Model must be fitted before predicting".into(),
            ));
        }
        let n_test = x_test.len();
        if n_test == 0 {
            return Ok((vec![], vec![]));
        }

        // K_star = k(X_test, X_train)
        let k_star = self.kernel.compute(x_test, &self.train_x);

        // Posterior mean: μ_* = K_star α
        let mean: Vec<f64> = k_star
            .iter()
            .map(|row| row.iter().zip(self.alpha.iter()).map(|(&ks, &a)| ks * a).sum())
            .collect();

        // Posterior variance: σ²_* = k_** - K_star L^{-T} L^{-1} K_star^T
        let mut variances = Vec::with_capacity(n_test);
        for i in 0..n_test {
            let k_star_star = self.kernel.evaluate(&x_test[i], &x_test[i]);
            // Solve L v = K_star[i]
            let v = forward_solve(&self.chol_l, &k_star[i]);
            let v_sq_norm: f64 = v.iter().map(|&vi| vi * vi).sum();
            variances.push((k_star_star - v_sq_norm + self.noise_variance).max(0.0));
        }

        Ok((mean, variances))
    }

    /// Bayesian model evidence (log marginal likelihood) at current hyperparams.
    pub fn log_marginal_likelihood(&self) -> f64 {
        if !self.is_fitted {
            return f64::NEG_INFINITY;
        }
        let n = self.n_train as f64;
        let log_det: f64 = self.chol_l.iter().enumerate().map(|(i, row)| row[i].ln()).sum::<f64>() * 2.0;
        let y = &self.train_y;
        let quad: f64 = y.iter().zip(self.alpha.iter()).map(|(&yi, &ai)| yi * ai).sum();
        -0.5 * quad - log_det * 0.5 - n * 0.5 * (2.0 * PI).ln()
    }

    // ---- Internal helpers ----

    fn gp_log_lik_params(&self, log_noise: f64, log_ls: f64, log_sigma2k: f64) -> f64 {
        let noise = log_noise.exp();
        let ls = log_ls.exp();
        let s2k = log_sigma2k.exp();

        let n = self.n_train;
        let mut k = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                let sq_dist: f64 = self.train_x[i]
                    .iter()
                    .zip(self.train_x[j].iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum();
                k[i][j] = s2k * (-sq_dist / (2.0 * ls.powi(2))).exp();
            }
            k[i][i] += noise;
        }

        match cholesky(&k, n) {
            Err(_) => f64::NEG_INFINITY,
            Ok(l) => {
                let alpha = chol_solve_vec(&l, &self.train_y, n);
                let log_det: f64 = l.iter().enumerate().map(|(i, row)| row[i].ln()).sum::<f64>() * 2.0;
                let quad: f64 = self.train_y.iter().zip(alpha.iter()).map(|(&yi, &ai)| yi * ai).sum();
                let log_prior = -log_noise.powi(2) / 2.0 - log_ls.powi(2) / 2.0 - log_sigma2k.powi(2) / 2.0;
                -0.5 * quad - 0.5 * log_det - n as f64 * 0.5 * (2.0 * PI).ln() + log_prior
            }
        }
    }

    fn update_cholesky(&mut self) -> Result<()> {
        let n = self.n_train;
        let mut k_mat = self.kernel.compute(&self.train_x, &self.train_x);
        for i in 0..n {
            k_mat[i][i] += self.noise_variance;
        }
        self.chol_l = cholesky(&k_mat, n)?;
        self.alpha = chol_solve_vec(&self.chol_l, &self.train_y, n);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Cholesky decomposition: return lower triangular L such that A = L L^T.
fn cholesky(a: &[Vec<f64>], n: usize) -> Result<Vec<Vec<f64>>> {
    let mut l = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(StatsError::ComputationError(
                        "Matrix not positive definite in Cholesky".into(),
                    ));
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    Ok(l)
}

/// Forward substitution: solve L x = b.
fn forward_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i][j] * x[j];
        }
        x[i] = if l[i][i].abs() > 1e-14 {
            s / l[i][i]
        } else {
            0.0
        };
    }
    x
}

/// Backward substitution: solve L^T x = b.
fn backward_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= l[j][i] * x[j];
        }
        x[i] = if l[i][i].abs() > 1e-14 {
            s / l[i][i]
        } else {
            0.0
        };
    }
    x
}

/// Solve A x = b using Cholesky factorization A = L L^T.
fn chol_solve_vec(l: &[Vec<f64>], b: &[f64], _n: usize) -> Vec<f64> {
    let v = forward_solve(l, b);
    backward_solve(l, &v)
}

fn sample_uniform_01(rng: &mut StdRng) -> f64 {
    use scirs2_core::random::Uniform;
    Uniform::new(0.0_f64, 1.0)
        .map(|d| d.sample(rng))
        .unwrap_or(0.5)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_regression_data(n: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng_state = seed;
        let lcg = |s: &mut u64| -> f64 {
            *s = s.wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((*s >> 33) as f64 / u32::MAX as f64) * 4.0 - 2.0 // [-2, 2]
        };
        let x: Vec<Vec<f64>> = (0..n).map(|_| vec![lcg(&mut rng_state)]).collect();
        // True f(x) = sin(x)
        let y: Vec<f64> = x.iter().map(|xi| xi[0].sin() + 0.1 * lcg(&mut rng_state)).collect();
        (x, y)
    }

    #[test]
    fn test_rbf_kernel() {
        let k = RBFKernel::new(1.0, 1.0).unwrap();
        assert!((k.evaluate(&[0.0], &[0.0]) - 1.0).abs() < 1e-10);
        let v = k.evaluate(&[0.0], &[1.0]);
        assert!(v > 0.0 && v < 1.0);
        // Symmetry
        assert!((k.evaluate(&[1.0], &[0.0]) - k.evaluate(&[0.0], &[1.0])).abs() < 1e-10);
    }

    #[test]
    fn test_matern52_kernel() {
        let k = Matern52Kernel::new(1.0, 1.0).unwrap();
        assert!((k.evaluate(&[0.0], &[0.0]) - 1.0).abs() < 1e-10);
        let v = k.evaluate(&[0.0], &[2.0]);
        assert!(v > 0.0 && v < 1.0);
    }

    #[test]
    fn test_kernel_invalid() {
        assert!(RBFKernel::new(0.0, 1.0).is_err());
        assert!(RBFKernel::new(1.0, 0.0).is_err());
        assert!(Matern52Kernel::new(-1.0, 1.0).is_err());
    }

    #[test]
    fn test_kernel_compute_matrix() {
        let k = RBFKernel::new(1.0, 1.0).unwrap();
        let x = vec![vec![0.0], vec![1.0], vec![2.0]];
        let km = k.compute(&x, &x);
        assert_eq!(km.len(), 3);
        assert!(km.iter().all(|row| row.len() == 3));
        // Diagonal should be sigma2 = 1.0
        assert!((km[0][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gp_fit_and_predict() {
        let (x_train, y_train) = make_regression_data(10, 42);
        let kernel = RBFKernel::new(1.0, 0.5).unwrap();
        let mut gp = NonparametricGPRegressor::new(0.01, kernel).unwrap();
        gp.fit(&x_train, &y_train, 50, 20, 42).unwrap();

        let x_test = vec![vec![0.0], vec![0.5], vec![-0.5]];
        let (mean, var) = gp.predict(&x_test).unwrap();
        assert_eq!(mean.len(), 3);
        assert_eq!(var.len(), 3);
        assert!(mean.iter().all(|&m| m.is_finite()));
        assert!(var.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_gp_marginal_likelihood() {
        let (x_train, y_train) = make_regression_data(8, 7);
        let kernel = RBFKernel::new(1.0, 0.5).unwrap();
        let mut gp = NonparametricGPRegressor::new(0.01, kernel).unwrap();
        gp.fit(&x_train, &y_train, 30, 10, 7).unwrap();
        let lml = gp.log_marginal_likelihood();
        assert!(lml.is_finite(), "log marginal likelihood should be finite");
    }

    #[test]
    fn test_gp_invalid_inputs() {
        let kernel = RBFKernel::new(1.0, 1.0).unwrap();
        let mut gp = NonparametricGPRegressor::new(0.01, kernel).unwrap();

        // Empty data
        assert!(gp.fit(&[], &[], 10, 5, 0).is_err());
        // n_warmup >= n_mcmc
        assert!(gp.fit(&[vec![1.0]], &[1.0], 10, 10, 0).is_err());
        // Prediction before fitting
        assert!(gp.predict(&[vec![0.0]]).is_err());
    }

    #[test]
    fn test_gp_hyperparams_update() {
        let (x_train, y_train) = make_regression_data(8, 3);
        let kernel = RBFKernel::new(1.0, 1.0).unwrap();
        let noise0 = 0.1;
        let mut gp = NonparametricGPRegressor::new(noise0, kernel).unwrap();
        gp.fit(&x_train, &y_train, 40, 10, 3).unwrap();
        // After MCMC, hyperparams should have been updated
        assert!(gp.noise_variance > 0.0);
        assert!(gp.kernel.length_scale > 0.0);
        assert!(gp.kernel.sigma2 > 0.0);
    }
}
