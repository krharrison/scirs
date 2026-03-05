//! Sparse GP via inducing-point methods (FITC and VFE).

use super::kernels::AdvancedKernel;
use super::linalg::{
    cholesky_jitter, log_det_from_cholesky, solve_lower, solve_lower_matrix, solve_upper,
};
use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Which sparse approximation to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseApproximation {
    /// Fully Independent Training Conditional (Snelson & Ghahramani 2006).
    Fitc,
    /// Variational Free Energy (Titsias 2009).
    Vfe,
}

/// Sparse GP using inducing-point methods.
///
/// Reduces O(N³) GP cost to O(NM² + M³) where M ≪ N.
///
/// # References
/// - Snelson & Ghahramani (2006) "Sparse Gaussian Processes using Pseudo-inputs"
/// - Titsias (2009) "Variational Learning of Inducing Variables in Sparse GPs"
#[derive(Debug, Clone)]
pub struct SparseGP<K: AdvancedKernel> {
    /// Covariance kernel.
    pub kernel: K,
    /// Observation noise variance.
    pub noise: f64,
    /// Approximation type (FITC or VFE).
    pub approximation: SparseApproximation,
    inducing: Option<Array2<f64>>,
    l_mm: Option<Array2<f64>>,
    q_mn: Option<Array2<f64>>,
    alpha: Option<Array1<f64>>,
    l_post: Option<Array2<f64>>,
    x_train: Option<Array2<f64>>,
    y_train: Option<Array1<f64>>,
}

impl<K: AdvancedKernel> SparseGP<K> {
    /// Create a new SparseGP.
    pub fn new(kernel: K, noise: f64, approximation: SparseApproximation) -> Self {
        Self {
            kernel,
            noise: noise.max(1e-8),
            approximation,
            inducing: None,
            l_mm: None,
            q_mn: None,
            alpha: None,
            l_post: None,
            x_train: None,
            y_train: None,
        }
    }

    /// Fit the sparse GP.
    pub fn fit(
        &mut self,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
        inducing_points: &Array2<f64>,
    ) -> StatsResult<()> {
        let n = x_train.nrows();
        let m = inducing_points.nrows();
        if n != y_train.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "x_train rows {n} ≠ y_train length {}",
                y_train.len()
            )));
        }
        if m == 0 {
            return Err(StatsError::InvalidArgument(
                "inducing_points must be non-empty".into(),
            ));
        }

        let mut k_mm = self.kernel.matrix(inducing_points, inducing_points);
        for i in 0..m {
            k_mm[[i, i]] += 1e-6;
        }
        let l_mm = cholesky_jitter(&k_mm)?;
        let k_mn = self.kernel.matrix(inducing_points, x_train);
        let q_mn = solve_lower_matrix(&l_mm, &k_mn)?;

        let k_nn_diag: Vec<f64> = (0..n)
            .map(|i| {
                let xi: Vec<f64> = x_train.row(i).iter().copied().collect();
                self.kernel.call(&xi, &xi)
            })
            .collect();

        let lambda: Vec<f64> = (0..n)
            .map(|i| {
                let q_sq: f64 = q_mn.column(i).iter().map(|&v| v * v).sum();
                match self.approximation {
                    SparseApproximation::Fitc => (k_nn_diag[i] - q_sq + self.noise).max(1e-10),
                    SparseApproximation::Vfe => self.noise,
                }
            })
            .collect();

        let mut a = Array2::<f64>::zeros((m, m));
        for i in 0..n {
            let col = q_mn.column(i).to_owned();
            let scale = 1.0 / lambda[i];
            for r in 0..m {
                for c in 0..m {
                    a[[r, c]] += scale * col[r] * col[c];
                }
            }
        }
        let mut b_mat = a;
        for i in 0..m {
            b_mat[[i, i]] += 1.0;
        }
        let l_post = cholesky_jitter(&b_mat)?;

        let mut kmn_lambda_y = Array1::<f64>::zeros(m);
        for i in 0..n {
            let scale = y_train[i] / lambda[i];
            for r in 0..m {
                kmn_lambda_y[r] += scale * q_mn[[r, i]];
            }
        }
        let alpha_half = solve_lower(&l_post, &kmn_lambda_y)?;
        let alpha = solve_upper(&l_post.t().to_owned(), &alpha_half)?;

        self.inducing = Some(inducing_points.clone());
        self.l_mm = Some(l_mm);
        self.q_mn = Some(q_mn);
        self.alpha = Some(alpha);
        self.l_post = Some(l_post);
        self.x_train = Some(x_train.clone());
        self.y_train = Some(y_train.clone());
        Ok(())
    }

    /// Predict mean and variance at `x_test`.
    pub fn predict(&self, x_test: &Array2<f64>) -> StatsResult<(Array1<f64>, Array1<f64>)> {
        let inducing = self.inducing.as_ref().ok_or_else(|| {
            StatsError::InvalidArgument("SparseGP not fitted".into())
        })?;
        let l_mm = self.l_mm.as_ref().expect("l_mm set after fit");
        let alpha = self.alpha.as_ref().expect("alpha set after fit");
        let l_post = self.l_post.as_ref().expect("l_post set after fit");

        let n_star = x_test.nrows();
        let k_ms = self.kernel.matrix(inducing, x_test);
        let q_ms = solve_lower_matrix(l_mm, &k_ms)?;

        let mut mean = Array1::<f64>::zeros(n_star);
        for i in 0..n_star {
            mean[i] = q_ms.column(i).iter().zip(alpha.iter()).map(|(&a, &b)| a * b).sum();
        }

        let q_ms_lpost = solve_lower_matrix(l_post, &q_ms)?;
        let mut var = Array1::<f64>::zeros(n_star);
        for i in 0..n_star {
            let xi: Vec<f64> = x_test.row(i).iter().copied().collect();
            let k_ss = self.kernel.call(&xi, &xi);
            let prior_var: f64 = q_ms.column(i).iter().map(|&v| v * v).sum();
            let post_var: f64 = q_ms_lpost.column(i).iter().map(|&v| v * v).sum();
            var[i] = (k_ss - prior_var + post_var + self.noise).max(0.0);
        }
        Ok((mean, var))
    }

    /// Compute ELBO (VFE) or approximate LML (FITC).
    pub fn log_marginal_likelihood_approx(&self) -> StatsResult<f64> {
        let l_post = self.l_post.as_ref().ok_or_else(|| {
            StatsError::InvalidArgument("SparseGP not fitted".into())
        })?;
        let alpha = self.alpha.as_ref().expect("alpha after fit");
        let y = self.y_train.as_ref().expect("y_train after fit");
        let n = y.len() as f64;
        let log_det_b = log_det_from_cholesky(l_post);
        let yt_kinv_y: f64 = alpha.iter().map(|&a| a * a).sum();
        let lml = -0.5 * (log_det_b + yt_kinv_y + n * (2.0 * PI).ln());
        Ok(lml)
    }
}
