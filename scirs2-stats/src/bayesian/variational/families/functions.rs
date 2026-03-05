//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::validation::*;
use std::f64::consts::PI;

use super::super::lgamma;
use super::types::{
    BetaVI, DirichletVI, GammaVI, KlDivergence, LogNormalVI, MeanFieldElbo, ReparamGradConfig,
    ReparamGradEstimator,
};

/// Trait for variational distribution families
///
/// A variational family defines a parametric distribution q(z; phi) that
/// approximates the posterior p(z | x). All families must support:
/// - Sampling via the reparameterization trick (z = f(epsilon, phi) where epsilon ~ p_base)
/// - Entropy computation (for the ELBO lower bound)
/// - Log probability evaluation (for diagnostics)
/// - Parameter get/set (for optimization)
pub trait VariationalFamily: std::fmt::Debug + Send + Sync {
    /// Dimensionality of the latent variable z
    fn dim(&self) -> usize;
    /// Number of variational parameters
    fn n_params(&self) -> usize;
    /// Get current variational parameters as a flat vector
    fn get_params(&self) -> Array1<f64>;
    /// Set variational parameters from a flat vector
    fn set_params(&mut self, params: &Array1<f64>) -> Result<()>;
    /// Sample using the reparameterization trick:
    /// z = f(epsilon; phi), where epsilon ~ base_noise_distribution
    ///
    /// Returns (z, log_q_z) where log_q_z is the log probability of the sample
    fn sample_reparam(&self, epsilon: &Array1<f64>) -> Result<(Array1<f64>, f64)>;
    /// Compute the differential entropy H\[q\] = -E_q\[log q(z)\]
    fn entropy(&self) -> f64;
    /// Compute log q(z; phi) for a given sample z
    fn log_prob(&self, z: &Array1<f64>) -> Result<f64>;
    /// Compute the gradient of the ELBO w.r.t. parameters using reparameterization.
    ///
    /// Given the gradient of log p(z, x) w.r.t. z, this returns the gradient
    /// of the ELBO w.r.t. phi using the chain rule through the reparameterization.
    ///
    /// `dlog_joint_dz`: gradient of log p(z, x) w.r.t. z
    /// `epsilon`: the base noise sample used to generate z
    ///
    /// Returns gradient of ELBO w.r.t. phi
    fn reparam_gradient(
        &self,
        dlog_joint_dz: &Array1<f64>,
        epsilon: &Array1<f64>,
    ) -> Result<Array1<f64>>;
    /// Compute KL divergence KL(q || p) where p is a standard prior.
    ///
    /// When p is the standard prior for this family (e.g., standard normal for Gaussian,
    /// uniform for Beta, symmetric Dirichlet for Dirichlet), this returns the KL.
    /// Returns None if analytical KL is not available (use MC estimation instead).
    fn kl_from_prior(&self) -> Option<f64>;
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_lognormal_vi_creation() {
        let ln = LogNormalVI::new(3).expect("should create LogNormal VI");
        assert_eq!(ln.dim, 3);
        assert_eq!(ln.n_params(), 6);
        let means = ln.means();
        let expected_mean = (0.5_f64).exp();
        for &m in means.iter() {
            assert!(
                (m - expected_mean).abs() < 1e-10,
                "mean={} expected={}",
                m,
                expected_mean
            );
        }
    }
    #[test]
    fn test_lognormal_vi_entropy() {
        let ln = LogNormalVI::new(2).expect("should create");
        let entropy = ln.entropy();
        let expected_per_dim = 0.5 * (1.0 + (2.0 * PI).ln());
        assert!((entropy - 2.0 * expected_per_dim).abs() < 1e-10);
    }
    #[test]
    fn test_lognormal_vi_log_prob() {
        let ln = LogNormalVI::new(1).expect("should create");
        let z = Array1::from_vec(vec![1.0]);
        let lp = ln.log_prob(&z).expect("should compute log prob");
        let expected = -0.5 * (2.0 * PI).ln();
        assert!(
            (lp - expected).abs() < 1e-10,
            "log_prob={} expected={}",
            lp,
            expected
        );
    }
    #[test]
    fn test_lognormal_vi_kl_from_prior() {
        let ln = LogNormalVI::new(3).expect("should create");
        let kl = ln.kl_from_prior().expect("should have analytical KL");
        assert!(
            kl.abs() < 1e-10,
            "KL from N(0,1) prior should be 0, got {}",
            kl
        );
    }
    #[test]
    fn test_lognormal_vi_sample_reparam() {
        let ln = LogNormalVI::new(3).expect("should create");
        let epsilon = Array1::from_vec(vec![0.5, -1.0, 1.5]);
        let (z, log_q) = ln.sample_reparam(&epsilon).expect("should sample");
        assert_eq!(z.len(), 3);
        for &zi in z.iter() {
            assert!(zi > 0.0, "LogNormal sample must be positive, got {}", zi);
        }
        assert!(log_q.is_finite(), "log_q should be finite");
    }
    #[test]
    fn test_lognormal_vi_params_roundtrip() {
        let mut ln = LogNormalVI::new(2).expect("should create");
        let params = Array1::from_vec(vec![1.0, -0.5, 0.3, -0.2]);
        ln.set_params(&params).expect("should set params");
        let retrieved = ln.get_params();
        for i in 0..4 {
            assert!(
                (retrieved[i] - params[i]).abs() < 1e-10,
                "param {} mismatch",
                i
            );
        }
    }
    #[test]
    fn test_lognormal_vi_reparam_gradient() {
        let ln = LogNormalVI::new(2).expect("should create");
        let dlog_joint_dz = Array1::from_vec(vec![1.0, -0.5]);
        let epsilon = Array1::from_vec(vec![0.3, -0.7]);
        let grad = ln
            .reparam_gradient(&dlog_joint_dz, &epsilon)
            .expect("should compute gradient");
        assert_eq!(grad.len(), 4);
        for &g in grad.iter() {
            assert!(g.is_finite(), "gradient should be finite");
        }
    }
    #[test]
    fn test_beta_vi_creation() {
        let bv = BetaVI::new(2).expect("should create Beta VI");
        assert_eq!(bv.dim, 2);
        assert_eq!(bv.n_params(), 4);
        let means = bv.means();
        for &m in means.iter() {
            assert!((m - 0.5).abs() < 1e-10, "uniform Beta mean should be 0.5");
        }
    }
    #[test]
    fn test_beta_vi_from_alpha_beta() {
        let alpha = Array1::from_vec(vec![2.0, 3.0]);
        let beta = Array1::from_vec(vec![5.0, 1.0]);
        let bv = BetaVI::from_alpha_beta(alpha, beta).expect("should create");
        let means = bv.means();
        assert!((means[0] - 2.0 / 7.0).abs() < 1e-10);
        assert!((means[1] - 3.0 / 4.0).abs() < 1e-10);
    }
    #[test]
    fn test_beta_vi_entropy() {
        let alpha = Array1::from_vec(vec![1.0]);
        let beta_arr = Array1::from_vec(vec![1.0]);
        let bv = BetaVI::from_alpha_beta(alpha, beta_arr).expect("should create");
        let entropy = bv.entropy();
        assert!(
            entropy.abs() < 1e-10,
            "entropy of Beta(1,1) should be 0, got {}",
            entropy
        );
    }
    #[test]
    fn test_beta_vi_log_prob_valid() {
        let alpha = Array1::from_vec(vec![2.0, 3.0]);
        let beta_arr = Array1::from_vec(vec![2.0, 3.0]);
        let bv = BetaVI::from_alpha_beta(alpha, beta_arr).expect("should create");
        let z = Array1::from_vec(vec![0.5, 0.3]);
        let lp = bv.log_prob(&z).expect("should compute log prob");
        assert!(lp.is_finite());
    }
    #[test]
    fn test_beta_vi_log_prob_out_of_range() {
        let bv = BetaVI::new(1).expect("should create");
        let z_bad = Array1::from_vec(vec![1.5]);
        let lp = bv
            .log_prob(&z_bad)
            .expect("should return NEG_INFINITY for out-of-range");
        assert_eq!(lp, f64::NEG_INFINITY);
    }
    #[test]
    fn test_beta_vi_kl_to_uniform() {
        let bv = BetaVI::new(1).expect("should create");
        let kl = bv.kl_to_uniform();
        assert!(
            kl.abs() < 1e-10,
            "KL should be 0 for same distribution, got {}",
            kl
        );
    }
    #[test]
    fn test_beta_vi_kl_to_beta_prior() {
        let alpha = Array1::from_vec(vec![2.0]);
        let beta_arr = Array1::from_vec(vec![3.0]);
        let bv = BetaVI::from_alpha_beta(alpha.clone(), beta_arr.clone()).expect("should create");
        let kl = bv
            .kl_to_beta_prior(&alpha, &beta_arr)
            .expect("should compute KL");
        assert!(kl.abs() < 1e-10, "KL to same prior should be 0, got {}", kl);
    }
    #[test]
    fn test_beta_vi_sample_reparam() {
        let alpha = Array1::from_vec(vec![2.0, 3.0]);
        let beta_arr = Array1::from_vec(vec![2.0, 3.0]);
        let bv = BetaVI::from_alpha_beta(alpha, beta_arr).expect("should create");
        let epsilon = Array1::from_vec(vec![0.5, -0.5]);
        let (z, _log_q) = bv.sample_reparam(&epsilon).expect("should sample");
        assert_eq!(z.len(), 2);
        for &zi in z.iter() {
            assert!(
                zi > 0.0 && zi < 1.0,
                "Beta sample must be in (0,1), got {}",
                zi
            );
        }
    }
    #[test]
    fn test_dirichlet_vi_creation() {
        let dv = DirichletVI::new(3).expect("should create Dirichlet VI");
        assert_eq!(dv.dim, 3);
        assert_eq!(dv.n_params(), 3);
        let means = dv.means();
        for &m in means.iter() {
            assert!(
                (m - 1.0 / 3.0).abs() < 1e-10,
                "uniform Dirichlet mean should be 1/3"
            );
        }
    }
    #[test]
    fn test_dirichlet_vi_from_alpha() {
        let alpha = Array1::from_vec(vec![2.0, 4.0, 6.0]);
        let dv = DirichletVI::from_alpha(alpha).expect("should create");
        let means = dv.means();
        assert!((means[0] - 2.0 / 12.0).abs() < 1e-10);
        assert!((means[1] - 4.0 / 12.0).abs() < 1e-10);
        assert!((means[2] - 6.0 / 12.0).abs() < 1e-10);
    }
    #[test]
    fn test_dirichlet_vi_entropy_uniform() {
        let dv = DirichletVI::new(3).expect("should create");
        let entropy = dv.entropy();
        assert!(
            entropy.is_finite(),
            "entropy should be finite, got {}",
            entropy
        );
    }
    #[test]
    fn test_dirichlet_vi_kl_to_self() {
        let alpha = Array1::from_vec(vec![2.0, 3.0, 5.0]);
        let dv = DirichletVI::from_alpha(alpha.clone()).expect("should create");
        let kl = dv.kl_to_dirichlet_prior(&alpha).expect("should compute KL");
        assert!(kl.abs() < 1e-10, "KL to itself should be 0, got {}", kl);
    }
    #[test]
    fn test_dirichlet_vi_kl_nonnegative() {
        let alpha1 = Array1::from_vec(vec![2.0, 3.0, 4.0]);
        let alpha2 = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let dv = DirichletVI::from_alpha(alpha1).expect("should create");
        let kl = dv
            .kl_to_dirichlet_prior(&alpha2)
            .expect("should compute KL");
        assert!(kl >= -1e-10, "KL should be non-negative, got {}", kl);
    }
    #[test]
    fn test_dirichlet_vi_log_prob() {
        let alpha = Array1::from_vec(vec![2.0, 3.0, 5.0]);
        let dv = DirichletVI::from_alpha(alpha).expect("should create");
        let z = Array1::from_vec(vec![0.2, 0.3, 0.5]);
        let lp = dv.log_prob(&z).expect("should compute log prob");
        assert!(lp.is_finite(), "log_prob should be finite, got {}", lp);
    }
    #[test]
    fn test_dirichlet_vi_sample_reparam_simplex() {
        let dv = DirichletVI::new(4).expect("should create");
        let epsilon = Array1::from_vec(vec![0.5, -0.5, 1.0, -1.0]);
        let (z, _log_q) = dv.sample_reparam(&epsilon).expect("should sample");
        assert_eq!(z.len(), 4);
        for &zi in z.iter() {
            assert!(
                zi > 0.0,
                "Dirichlet sample element must be positive, got {}",
                zi
            );
        }
        let sum: f64 = z.sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Dirichlet sample must sum to 1, got {}",
            sum
        );
    }
    #[test]
    fn test_dirichlet_vi_params_roundtrip() {
        let mut dv = DirichletVI::new(3).expect("should create");
        let params = Array1::from_vec(vec![0.5, -0.3, 1.2]);
        dv.set_params(&params).expect("should set params");
        let retrieved = dv.get_params();
        for i in 0..3 {
            assert!(
                (retrieved[i] - params[i]).abs() < 1e-10,
                "param {} mismatch",
                i
            );
        }
    }
    #[test]
    fn test_gamma_vi_creation() {
        let gv = GammaVI::new(2).expect("should create Gamma VI");
        assert_eq!(gv.dim, 2);
        assert_eq!(gv.n_params(), 4);
        let means = gv.means();
        for &m in means.iter() {
            assert!((m - 1.0).abs() < 1e-10, "mean should be 1 for Gamma(1,1)");
        }
    }
    #[test]
    fn test_gamma_vi_entropy() {
        let shape = Array1::from_vec(vec![2.0]);
        let rate = Array1::from_vec(vec![1.0]);
        let gv = GammaVI::from_shape_rate(shape, rate).expect("should create");
        let entropy = gv.entropy();
        assert!(
            entropy.is_finite(),
            "entropy should be finite, got {}",
            entropy
        );
        assert!(entropy > 0.0, "entropy of Gamma(2,1) should be positive");
    }
    #[test]
    fn test_gamma_vi_log_prob() {
        let gv = GammaVI::new(1).expect("should create");
        let z = Array1::from_vec(vec![1.0]);
        let lp = gv.log_prob(&z).expect("should compute log prob");
        assert!(
            (lp - (-1.0)).abs() < 1e-10,
            "Gamma(1,1) log_prob(1)=-1, got {}",
            lp
        );
    }
    #[test]
    fn test_gamma_vi_kl_from_prior() {
        let gv = GammaVI::new(2).expect("should create");
        let kl = gv.kl_from_prior().expect("should have analytical KL");
        assert!(kl.abs() < 1e-10, "KL to self should be 0, got {}", kl);
    }
    #[test]
    fn test_gamma_vi_sample_positive() {
        let gv = GammaVI::new(3).expect("should create");
        let epsilon = Array1::from_vec(vec![0.5, 1.0, -0.5]);
        let (z, _log_q) = gv.sample_reparam(&epsilon).expect("should sample");
        for &zi in z.iter() {
            assert!(zi > 0.0, "Gamma sample must be positive, got {}", zi);
        }
    }
    #[test]
    fn test_mean_field_elbo_creation() {
        let elbo = MeanFieldElbo::new(5, true).expect("should create");
        assert_eq!(elbo.n_mc_samples, 5);
        assert!(elbo.use_analytical_kl);
    }
    #[test]
    fn test_mean_field_elbo_creation_zero_samples() {
        let result = MeanFieldElbo::new(0, true);
        assert!(result.is_err(), "should fail with zero samples");
    }
    #[test]
    fn test_mean_field_elbo_lognormal() {
        let family = LogNormalVI::new(2).expect("should create");
        let elbo_calc = MeanFieldElbo::new(5, true).expect("should create");
        let (elbo, grad) = elbo_calc
            .compute_elbo_and_grad(
                &family,
                |z: &Array1<f64>| {
                    let target_mu = (2.0_f64).ln();
                    let target_sigma = 0.5_f64;
                    let mut log_p = 0.0;
                    let mut grad = Array1::zeros(z.len());
                    for i in 0..z.len() {
                        if z[i] <= 0.0 {
                            return Ok((f64::NEG_INFINITY, Array1::zeros(z.len())));
                        }
                        let lz = z[i].ln();
                        let normalized = (lz - target_mu) / target_sigma;
                        log_p += -0.5 * normalized * normalized
                            - (2.0 * PI).ln().sqrt()
                            - target_sigma.ln()
                            - lz;
                        grad[i] =
                            -(lz - target_mu) / (target_sigma * target_sigma * z[i]) - 1.0 / z[i];
                    }
                    Ok((log_p, grad))
                },
                42,
            )
            .expect("should compute ELBO");
        assert!(elbo.is_finite(), "ELBO should be finite, got {}", elbo);
        assert_eq!(grad.len(), 4);
        for &g in grad.iter() {
            assert!(g.is_finite(), "gradient should be finite");
        }
    }
    #[test]
    fn test_mean_field_elbo_beta() {
        let family = BetaVI::new(2).expect("should create");
        let elbo_calc = MeanFieldElbo::new(3, true).expect("should create");
        let (elbo, grad) = elbo_calc
            .compute_elbo_and_grad(
                &family,
                |z: &Array1<f64>| {
                    let a = 2.0_f64;
                    let b = 5.0_f64;
                    let log_b = lgamma(a) + lgamma(b) - lgamma(a + b);
                    let mut log_p = -log_b * z.len() as f64;
                    let mut grad = Array1::zeros(z.len());
                    for i in 0..z.len() {
                        if z[i] <= 0.0 || z[i] >= 1.0 {
                            return Ok((f64::NEG_INFINITY, Array1::zeros(z.len())));
                        }
                        log_p += (a - 1.0) * z[i].ln() + (b - 1.0) * (1.0 - z[i]).ln();
                        grad[i] = (a - 1.0) / z[i] - (b - 1.0) / (1.0 - z[i]);
                    }
                    Ok((log_p, grad))
                },
                42,
            )
            .expect("should compute ELBO");
        assert!(elbo.is_finite(), "ELBO should be finite, got {}", elbo);
        assert_eq!(grad.len(), 4);
    }
    #[test]
    fn test_kl_gaussian_same_distribution() {
        let kl = KlDivergence::gaussian_1d(0.0, 1.0, 0.0, 1.0).expect("should compute");
        assert!(kl.abs() < 1e-10, "KL to itself should be 0, got {}", kl);
    }
    #[test]
    fn test_kl_gaussian_known_value() {
        let kl = KlDivergence::gaussian_1d(1.0, 1.0, 0.0, 1.0).expect("should compute");
        assert!((kl - 0.5).abs() < 1e-10, "KL should be 0.5, got {}", kl);
    }
    #[test]
    fn test_kl_gaussian_nonneg() {
        let kl = KlDivergence::gaussian_1d(3.0, 2.0, 0.0, 5.0).expect("should compute");
        assert!(kl >= 0.0, "KL should be non-negative, got {}", kl);
    }
    #[test]
    fn test_kl_beta_same() {
        let kl = KlDivergence::beta(2.0, 3.0, 2.0, 3.0).expect("should compute");
        assert!(kl.abs() < 1e-10, "KL to itself should be 0, got {}", kl);
    }
    #[test]
    fn test_kl_beta_nonneg() {
        let kl = KlDivergence::beta(2.0, 5.0, 1.0, 1.0).expect("should compute");
        assert!(kl >= -1e-10, "KL should be non-negative, got {}", kl);
    }
    #[test]
    fn test_kl_dirichlet_same() {
        let alpha = Array1::from_vec(vec![2.0, 3.0, 5.0]);
        let kl = KlDivergence::dirichlet(&alpha, &alpha).expect("should compute");
        assert!(kl.abs() < 1e-10, "KL to itself should be 0, got {}", kl);
    }
    #[test]
    fn test_kl_dirichlet_nonneg() {
        let alpha1 = Array1::from_vec(vec![2.0, 3.0, 4.0]);
        let alpha2 = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let kl = KlDivergence::dirichlet(&alpha1, &alpha2).expect("should compute");
        assert!(kl >= -1e-10, "KL should be non-negative, got {}", kl);
    }
    #[test]
    fn test_kl_gamma_same() {
        let kl = KlDivergence::gamma(2.0, 1.0, 2.0, 1.0).expect("should compute");
        assert!(kl.abs() < 1e-10, "KL to itself should be 0, got {}", kl);
    }
    #[test]
    fn test_kl_gamma_nonneg() {
        let kl = KlDivergence::gamma(3.0, 2.0, 1.0, 1.0).expect("should compute");
        assert!(kl >= -1e-10, "KL should be non-negative, got {}", kl);
    }
    #[test]
    fn test_kl_lognormal_same() {
        let kl = KlDivergence::lognormal_1d(1.0, 0.5, 1.0, 0.5).expect("should compute");
        assert!(kl.abs() < 1e-10, "KL to itself should be 0, got {}", kl);
    }
    #[test]
    fn test_reparam_estimator_creation() {
        let config = ReparamGradConfig::default();
        let estimator = ReparamGradEstimator::new(config);
        assert_eq!(estimator.n_estimates, 0);
        assert_eq!(estimator.baseline, 0.0);
    }
    #[test]
    fn test_reparam_estimator_single_step() {
        let config = ReparamGradConfig {
            n_samples: 3,
            rao_blackwell: true,
            control_variates: false,
            baseline_decay: 0.99,
        };
        let mut estimator = ReparamGradEstimator::new(config);
        let family = LogNormalVI::new(2).expect("should create");
        let (elbo, grad) = estimator
            .estimate(
                &family,
                |z: &Array1<f64>| {
                    let log_p = -0.5 * z.iter().map(|&zi| zi.powi(2)).sum::<f64>();
                    let grad_z = z.mapv(|zi| -zi);
                    Ok((log_p, grad_z))
                },
                0,
            )
            .expect("should estimate");
        assert!(elbo.is_finite(), "ELBO should be finite");
        assert_eq!(grad.len(), 4);
        assert_eq!(estimator.n_estimates, 1);
    }
    #[test]
    fn test_reparam_estimator_gradient_snr() {
        let config = ReparamGradConfig {
            n_samples: 5,
            control_variates: true,
            ..ReparamGradConfig::default()
        };
        let mut estimator = ReparamGradEstimator::new(config);
        let family = GammaVI::new(2).expect("should create");
        for step in 0..3 {
            let _ = estimator
                .estimate(
                    &family,
                    |z: &Array1<f64>| {
                        let log_p = -z.iter().sum::<f64>();
                        let grad_z = Array1::ones(z.len()) * (-1.0);
                        Ok((log_p, grad_z))
                    },
                    step,
                )
                .expect("should estimate");
        }
        if !estimator.grad_variance.is_empty() {
            let snr = estimator.gradient_snr();
            for &s in snr.iter() {
                assert!(
                    s >= 0.0 || s.is_infinite(),
                    "SNR should be non-negative or infinite"
                );
            }
        }
    }
}
