//! GP Classification via Laplace approximation and Expectation Propagation.

use super::kernels::AdvancedKernel;
use super::linalg::{cholesky_jitter, solve_lower, solve_lower_matrix, solve_spd, solve_upper};
use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use std::f64::consts::PI;

// ========================
// Math helpers
// ========================

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + libm_erf(x / std::f64::consts::SQRT_2))
}

fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

fn libm_erf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let ex = x.exp(); ex / (1.0 + ex) }
}

// ========================
// Public types
// ========================

/// Likelihood type for GP classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassificationLikelihood {
    /// Probit link: p(y=1|f) = Φ(f).
    Probit,
    /// Logistic/logit link: p(y=1|f) = σ(f).
    Logistic,
}

/// Inference method for GP classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassificationInference {
    /// Laplace approximation (Newton's method at the MAP).
    Laplace,
    /// Expectation Propagation (Minka 2001).
    EP,
}

/// Gaussian Process Binary Classifier.
///
/// # References
/// - Williams & Barber (1998) "Bayesian Classification with Gaussian Processes"
/// - Minka (2001) "Expectation Propagation for Approximate Bayesian Inference"
/// - Rasmussen & Williams (2006) Chapter 3
#[derive(Debug, Clone)]
pub struct GPClassification<K: AdvancedKernel> {
    pub kernel: K,
    pub likelihood: ClassificationLikelihood,
    pub inference: ClassificationInference,
    pub max_iter: usize,
    pub tol: f64,
    x_train: Option<Array2<f64>>,
    f_mode: Option<Array1<f64>>,
    l_b: Option<Array2<f64>>,
    d_log_lik: Option<Array1<f64>>,
    ep_tau_tilde: Option<Array1<f64>>,
    ep_nu_tilde: Option<Array1<f64>>,
}

impl<K: AdvancedKernel> GPClassification<K> {
    pub fn new(
        kernel: K,
        likelihood: ClassificationLikelihood,
        inference: ClassificationInference,
    ) -> Self {
        Self {
            kernel,
            likelihood,
            inference,
            max_iter: 100,
            tol: 1e-6,
            x_train: None,
            f_mode: None,
            l_b: None,
            d_log_lik: None,
            ep_tau_tilde: None,
            ep_nu_tilde: None,
        }
    }

    fn log_lik(&self, f: f64, y: f64) -> f64 {
        match self.likelihood {
            ClassificationLikelihood::Probit => normal_cdf(y * f).max(1e-14).ln(),
            ClassificationLikelihood::Logistic => -(1.0 + (-(y * f)).exp()).max(1e-14).ln(),
        }
    }

    fn d_log_lik(&self, f: f64, y: f64) -> f64 {
        match self.likelihood {
            ClassificationLikelihood::Probit => {
                let z = y * f;
                let phi_z = normal_cdf(z).max(1e-14);
                y * normal_pdf(z) / phi_z
            }
            ClassificationLikelihood::Logistic => y * (1.0 - sigmoid(y * f)),
        }
    }

    fn d2_neg_log_lik(&self, f: f64, y: f64) -> f64 {
        match self.likelihood {
            ClassificationLikelihood::Probit => {
                let z = y * f;
                let phi_z = normal_cdf(z).max(1e-14);
                let pdf_z = normal_pdf(z);
                let ratio = pdf_z / phi_z;
                ratio * (ratio + z)
            }
            ClassificationLikelihood::Logistic => {
                let pi = sigmoid(y * f);
                pi * (1.0 - pi)
            }
        }
    }

    fn fit_laplace(
        &mut self,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
    ) -> StatsResult<()> {
        let n = x_train.nrows();
        let k = self.kernel.matrix(x_train, x_train);
        let mut f = Array1::<f64>::zeros(n);
        let mut l_b_last = Array2::<f64>::eye(n);

        for _iter in 0..self.max_iter {
            let grad: Array1<f64> = (0..n).map(|i| self.d_log_lik(f[i], y_train[i])).collect();
            let w: Vec<f64> = (0..n).map(|i| self.d2_neg_log_lik(f[i], y_train[i]).max(1e-10)).collect();
            let w_sqrt: Vec<f64> = w.iter().map(|&wi| wi.sqrt()).collect();

            let mut b_mat = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    b_mat[[i, j]] = w_sqrt[i] * k[[i, j]] * w_sqrt[j];
                }
                b_mat[[i, i]] += 1.0;
            }
            let l_b = cholesky_jitter(&b_mat)?;

            let b_vec: Array1<f64> = (0..n).map(|i| w[i] * f[i] + grad[i]).collect();
            let wk_b: Array1<f64> = (0..n).map(|i| {
                let krow: f64 = (0..n).map(|j| k[[i, j]] * b_vec[j]).sum();
                w_sqrt[i] * krow
            }).collect();

            let l_inv_wkb = solve_lower(&l_b, &wk_b)?;
            let lt_inv = solve_upper(&l_b.t().to_owned(), &l_inv_wkb)?;
            let correction: Array1<f64> = (0..n).map(|i| w_sqrt[i] * lt_inv[i]).collect();
            let a: Array1<f64> = (0..n).map(|i| b_vec[i] - correction[i]).collect();
            let f_new: Array1<f64> = (0..n).map(|i| (0..n).map(|j| k[[i, j]] * a[j]).sum()).collect();

            let delta: f64 = (0..n).map(|i| (f_new[i] - f[i]).powi(2)).sum::<f64>().sqrt();
            l_b_last = l_b;
            f = f_new;
            if delta < self.tol { break; }
        }

        let grad_mode: Array1<f64> = (0..n).map(|i| self.d_log_lik(f[i], y_train[i])).collect();
        self.f_mode = Some(f);
        self.d_log_lik = Some(grad_mode);
        self.l_b = Some(l_b_last);
        Ok(())
    }

    fn ep_moment_match(&self, y: f64, mu_cav: f64, sig_cav: f64) -> (f64, f64, f64) {
        match self.likelihood {
            ClassificationLikelihood::Probit => {
                let beta = y * mu_cav / (1.0 + sig_cav).sqrt();
                let z = normal_cdf(beta);
                if z < 1e-100 { return (z, mu_cav, sig_cav); }
                let denom = (1.0 + sig_cav).sqrt();
                let nu = y * normal_pdf(beta) / (z * denom);
                let mu_hat = mu_cav + sig_cav * nu;
                let sig_hat = (sig_cav - sig_cav * sig_cav * (nu * nu + beta * nu / denom)).max(1e-10);
                (z, mu_hat, sig_hat)
            }
            ClassificationLikelihood::Logistic => {
                let gh_weights = [0.112194_f64, 0.360762, 0.467914, 0.360762, 0.112194];
                let gh_nodes = [-2.020201_f64, -1.0, 0.0, 1.0, 2.020201];
                let sq2 = (2.0_f64 * sig_cav).sqrt();
                let z: f64 = gh_weights.iter().zip(gh_nodes.iter())
                    .map(|(&w, &t)| w * sigmoid(y * (sq2 * t + mu_cav)))
                    .sum::<f64>()
                    .max(1e-100);
                let mu_hat: f64 = gh_weights.iter().zip(gh_nodes.iter())
                    .map(|(&w, &t)| { let f = sq2 * t + mu_cav; w * f * sigmoid(y * f) })
                    .sum::<f64>() / z;
                let var_hat: f64 = (gh_weights.iter().zip(gh_nodes.iter())
                    .map(|(&w, &t)| { let f = sq2 * t + mu_cav; w * f * f * sigmoid(y * f) })
                    .sum::<f64>() / z - mu_hat * mu_hat).max(1e-10);
                (z, mu_hat, var_hat)
            }
        }
    }

    fn fit_ep(
        &mut self,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
    ) -> StatsResult<()> {
        let n = x_train.nrows();
        let k = self.kernel.matrix(x_train, x_train);
        let mut nu_tilde = Array1::<f64>::zeros(n);
        let mut tau_tilde = Array1::<f64>::zeros(n);
        let mut mu = Array1::<f64>::zeros(n);
        let mut sigma_diag: Array1<f64> = (0..n).map(|i| k[[i, i]]).collect();

        for _sweep in 0..self.max_iter {
            let mu_old = mu.clone();
            for i in 0..n {
                let tau_cav = (1.0 / sigma_diag[i] - tau_tilde[i]).max(1e-10);
                let mu_cav = mu[i] / sigma_diag[i] * (1.0 / tau_cav) - nu_tilde[i] / tau_cav;
                let sig_cav = 1.0 / tau_cav;
                let (z_hat, mu_hat, sig_hat) = self.ep_moment_match(y_train[i], mu_cav, sig_cav);
                if z_hat < 1e-100 { continue; }
                tau_tilde[i] = (1.0 / sig_hat - tau_cav).max(0.0);
                nu_tilde[i] = mu_hat / sig_hat - mu_cav * tau_cav;
                sigma_diag[i] = sig_hat;
                mu[i] = mu_hat;
            }
            // Recompute marginals
            let mut kt_k = k.clone();
            for i in 0..n {
                kt_k[[i, i]] += 1.0 / tau_tilde[i].max(1e-10);
            }
            if let Ok(x_mat) = {
                let b_mat = k.clone();
                super::linalg::solve_spd_matrix(&kt_k, &b_mat)
            } {
                for i in 0..n {
                    let row_i: Vec<f64> = k.row(i).iter().copied().collect();
                    let x_col_i: Vec<f64> = x_mat.column(i).iter().copied().collect();
                    let reduction: f64 = row_i.iter().zip(x_col_i.iter()).map(|(&a, &b)| a * b).sum();
                    sigma_diag[i] = (k[[i, i]] - reduction).max(1e-10);
                }
                let k_nu: Array1<f64> = (0..n).map(|i| (0..n).map(|j| k[[i, j]] * nu_tilde[j]).sum()).collect();
                if let Ok(k_inv_k_nu) = solve_spd(&kt_k, &k_nu) {
                    mu = (0..n).map(|i| {
                        let krow_knu: f64 = (0..n).map(|j| k[[i, j]] * k_inv_k_nu[j]).sum();
                        k_nu[i] - krow_knu
                    }).collect();
                }
            }
            let delta: f64 = (0..n).map(|i| (mu[i] - mu_old[i]).powi(2)).sum::<f64>().sqrt();
            if delta < self.tol { break; }
        }
        self.f_mode = Some(mu);
        self.ep_tau_tilde = Some(tau_tilde);
        self.ep_nu_tilde = Some(nu_tilde);
        Ok(())
    }

    /// Fit the GP classifier. `y_train` ∈ {±1}.
    pub fn fit(
        &mut self,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
    ) -> StatsResult<()> {
        for &yi in y_train.iter() {
            if yi != 1.0 && yi != -1.0 {
                return Err(StatsError::InvalidArgument(format!(
                    "y_train must be ±1, found {yi}"
                )));
            }
        }
        match self.inference {
            ClassificationInference::Laplace => self.fit_laplace(x_train, y_train)?,
            ClassificationInference::EP => self.fit_ep(x_train, y_train)?,
        }
        self.x_train = Some(x_train.clone());
        Ok(())
    }

    /// Predict P(y=1 | x) at test points.
    pub fn predict_proba(&self, x_test: &Array2<f64>) -> StatsResult<Array1<f64>> {
        let x_train = self.x_train.as_ref().ok_or_else(|| {
            StatsError::InvalidArgument("GPClassification not fitted".into())
        })?;
        let f_mode = self.f_mode.as_ref().expect("f_mode after fit");
        let n = x_train.nrows();
        let n_star = x_test.nrows();
        let k_train = self.kernel.matrix(x_train, x_train);
        let k_st = self.kernel.matrix(x_test, x_train);
        let mut proba = Array1::<f64>::zeros(n_star);

        match self.inference {
            ClassificationInference::Laplace => {
                let l_b = self.l_b.as_ref().ok_or_else(|| {
                    StatsError::InvalidArgument("L_B not computed; re-fit".into())
                })?;
                let d_ll = self.d_log_lik.as_ref().expect("d_log_lik after fit");
                let w_diag: Vec<f64> = (0..n)
                    .map(|i| self.d2_neg_log_lik(f_mode[i], 0.0).max(1e-10))
                    .collect();
                let w_sqrt: Vec<f64> = w_diag.iter().map(|&w| w.sqrt()).collect();
                let _ = k_train; // used above for W computation

                for i_star in 0..n_star {
                    let k_s = k_st.row(i_star).to_owned();
                    let wk_s: Array1<f64> = (0..n).map(|j| w_sqrt[j] * k_s[j]).collect();
                    let v = solve_lower(l_b, &wk_s).unwrap_or_else(|_| Array1::zeros(n));
                    let xi: Vec<f64> = x_test.row(i_star).iter().copied().collect();
                    let k_ss = self.kernel.call(&xi, &xi);
                    let v_sq: f64 = v.iter().map(|&x| x * x).sum();
                    let pred_var = (k_ss - v_sq).max(0.0);
                    let pred_mean: f64 = (0..n).map(|j| k_s[j] * d_ll[j]).sum();
                    proba[i_star] = match self.likelihood {
                        ClassificationLikelihood::Probit => {
                            let kappa = (1.0 + PI / 8.0 * pred_var).sqrt();
                            normal_cdf(pred_mean / kappa)
                        }
                        ClassificationLikelihood::Logistic => {
                            sigmoid(pred_mean / (1.0 + PI / 8.0 * pred_var).sqrt())
                        }
                    };
                }
            }
            ClassificationInference::EP => {
                let tau_tilde = self.ep_tau_tilde.as_ref().expect("ep_tau_tilde after EP fit");
                let nu_tilde = self.ep_nu_tilde.as_ref().expect("ep_nu_tilde after EP fit");
                let mut k_tilde = k_train.clone();
                for i in 0..n {
                    k_tilde[[i, i]] += 1.0 / tau_tilde[i].max(1e-10);
                }
                let nu: Array1<f64> = nu_tilde.clone();
                let post_mean = solve_spd(&k_tilde, &nu).unwrap_or_else(|_| Array1::zeros(n));
                for i_star in 0..n_star {
                    let k_s = k_st.row(i_star).to_owned();
                    let pred_mean: f64 = k_s.iter().zip(post_mean.iter()).map(|(&a, &b)| a * b).sum();
                    let v = solve_lower_matrix(
                        &cholesky_jitter(&k_tilde).unwrap_or_else(|_| Array2::eye(n)),
                        &k_s.clone().insert_axis(Axis(1)),
                    ).unwrap_or_else(|_| Array2::zeros((n, 1)));
                    let xi: Vec<f64> = x_test.row(i_star).iter().copied().collect();
                    let k_ss = self.kernel.call(&xi, &xi);
                    let v_sq: f64 = v.iter().map(|&x| x * x).sum();
                    let pred_var = (k_ss - v_sq).max(0.0);
                    proba[i_star] = match self.likelihood {
                        ClassificationLikelihood::Probit => {
                            normal_cdf(pred_mean / (1.0 + PI / 8.0 * pred_var).sqrt())
                        }
                        ClassificationLikelihood::Logistic => {
                            sigmoid(pred_mean / (1.0 + PI / 8.0 * pred_var).sqrt())
                        }
                    };
                }
            }
        }
        Ok(proba)
    }

    /// Predict binary labels {-1, +1}.
    pub fn predict(&self, x_test: &Array2<f64>) -> StatsResult<Array1<f64>> {
        let proba = self.predict_proba(x_test)?;
        Ok(proba.mapv(|p| if p >= 0.5 { 1.0 } else { -1.0 }))
    }
}
