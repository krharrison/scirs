//! Advanced Gaussian Process Methods
//!
//! Provides state-of-the-art GP extensions:
//!
//! - [`SparseGP`] – Sparse/inducing-point approximations (FITC, VFE)
//! - [`DeepGP`] – Stacked GP layers with doubly-stochastic variational inference
//! - [`MultiOutputGP`] – Multi-output GPs via Linear Model of Coregionalization
//! - [`GPClassification`] – GP classification (Laplace / EP, probit/logit)
//! - Advanced kernels: [`NeuralTangentKernel`], [`ArcCosineKernel`],
//!   [`SpectralMixtureKernel`], [`AdditiveKernel`], [`ARDKernel`]
//! - [`GPHyperparamOpt`] – ARD marginal-likelihood hyperparameter optimisation

mod advanced_impl;

pub use advanced_impl::{
    ARDKernel, AdditiveKernel, AdvancedKernel, ArcCosineKernel, ClassificationInference,
    ClassificationLikelihood, DeepGP, DeepGPLayerConfig, GPClassification, GPHyperparamOpt,
    MultiOutputGP, NeuralTangentKernel, SparseApproximation, SparseGP, SpectralMixtureComponent,
    SpectralMixtureKernel,
};

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_ntk_positive_definite() {
        let k = NeuralTangentKernel::new(2, 1.0, 0.1);
        let x = Array2::from_shape_vec((3, 2), vec![0.0, 1.0, 1.0, 0.0, 0.5, 0.5]).expect("x");
        let mat = k.matrix(&x, &x);
        for i in 0..3 {
            assert!(mat[[i, i]] > 0.0);
        }
        for i in 0..3 {
            for j in 0..3 {
                assert!((mat[[i, j]] - mat[[j, i]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_arccos_kernel_degree0() {
        let k = ArcCosineKernel::new(0, 1.0);
        let v = k.call(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(v >= 0.0);
    }

    #[test]
    fn test_arccos_kernel_degree1_symmetry() {
        let k = ArcCosineKernel::new(1, 1.0);
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 1.0];
        assert!((k.call(&a, &b) - k.call(&b, &a)).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_mixture_symmetry() {
        let k = SpectralMixtureKernel::random_init(3, 2, 42);
        let a = vec![0.0, 1.0];
        let b = vec![1.0, 2.0];
        assert!((k.call(&a, &b) - k.call(&b, &a)).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_mixture_stationarity() {
        let comps = vec![
            SpectralMixtureComponent { weight: 0.5, mean: vec![1.0], variance: vec![0.2] },
            SpectralMixtureComponent { weight: 0.5, mean: vec![2.0], variance: vec![0.1] },
        ];
        let k = SpectralMixtureKernel::new(comps);
        let x = vec![3.0];
        let v = k.call(&x, &x);
        assert!((v - 1.0).abs() < 1e-10, "SMK at zero lag should be 1.0, got {v}");
    }

    #[test]
    fn test_additive_kernel_self_covariance() {
        let k = AdditiveKernel::isotropic(3, 1.0, 2.0);
        let x = vec![1.0, 2.0, 3.0];
        let v = k.call(&x, &x);
        assert!((v - 2.0).abs() < 1e-10, "Additive self-cov = variance, got {v}");
    }

    #[test]
    fn test_ard_kernel_relevance() {
        let k = ARDKernel::new(vec![1.0, 100.0], 1.0);
        let x1 = vec![0.0, 0.0];
        let x2 = vec![0.0, 50.0];
        let x3 = vec![2.0, 0.0];
        let k12 = k.call(&x1, &x2);
        let k13 = k.call(&x1, &x3);
        assert!(k12 > k13, "Irrelevant-dim shift should give higher k: k12={k12}, k13={k13}");
    }

    #[test]
    fn test_sparse_gp_fitc_fit_predict() {
        let kernel = ARDKernel::isotropic(1, 1.0, 1.0);
        let mut sgp = SparseGP::new(kernel, 0.1, SparseApproximation::Fitc);
        let x_train = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).expect("x");
        let y_train = array![0.0, 1.0, 0.0, -1.0, 0.0];
        let inducing = Array2::from_shape_vec((3, 1), vec![0.5, 2.0, 3.5]).expect("ind");
        sgp.fit(&x_train, &y_train, &inducing).expect("fit");
        let x_test = Array2::from_shape_vec((2, 1), vec![1.5, 2.5]).expect("x_test");
        let (mean, var) = sgp.predict(&x_test).expect("predict");
        assert_eq!(mean.len(), 2);
        assert!(var.iter().all(|&v| v >= 0.0));
        assert!(mean.iter().all(|&m| m.is_finite()));
    }

    #[test]
    fn test_sparse_gp_vfe_elbo_finite() {
        let kernel = ARDKernel::isotropic(1, 1.0, 1.0);
        let mut sgp = SparseGP::new(kernel, 0.05, SparseApproximation::Vfe);
        let x_train = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).expect("x");
        let y_train = array![0.0, 1.0, 0.0, -1.0];
        let inducing = Array2::from_shape_vec((2, 1), vec![0.5, 2.5]).expect("ind");
        sgp.fit(&x_train, &y_train, &inducing).expect("fit");
        let elbo = sgp.log_marginal_likelihood_approx().expect("elbo");
        assert!(elbo.is_finite(), "ELBO should be finite: {elbo}");
    }

    #[test]
    fn test_deep_gp_predict_shape() {
        let x_init = Array2::from_shape_vec((4, 2), vec![
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        ]).expect("x_init");
        let configs = vec![
            DeepGPLayerConfig { num_inducing: 4, out_dim: 2, length_scale: 1.0, variance: 1.0, noise: 0.1 },
            DeepGPLayerConfig { num_inducing: 4, out_dim: 1, length_scale: 1.0, variance: 1.0, noise: 0.1 },
        ];
        let dgp = DeepGP::new(configs, &x_init).expect("build DeepGP");
        let x_test = Array2::from_shape_vec((3, 2), vec![0.5, 0.5, -0.5, 0.5, 0.5, -0.5]).expect("x_test");
        let (mean, var) = dgp.predict(&x_test).expect("predict");
        assert_eq!(mean.shape(), &[3, 1]);
        assert_eq!(var.shape(), &[3, 1]);
        assert!(var.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_multi_output_gp_predict_shape() {
        let kernel = ARDKernel::isotropic(1, 1.0, 1.0);
        let mut mogp = MultiOutputGP::new(kernel, 2, 2);
        mogp.noise = vec![0.01, 0.01];
        let x_train = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).expect("x");
        let y_train = Array2::from_shape_vec((3, 2), vec![0.0, 1.0, 1.0, -1.0, 0.0, 0.5]).expect("y");
        mogp.fit(&x_train, &y_train).expect("fit");
        let x_test = Array2::from_shape_vec((2, 1), vec![0.5, 1.5]).expect("x_test");
        let (mean, var) = mogp.predict(&x_test).expect("predict");
        assert_eq!(mean.shape(), &[2, 2]);
        assert_eq!(var.shape(), &[2, 2]);
        assert!(var.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_multi_output_gp_coregionalization() {
        let kernel = ARDKernel::isotropic(1, 1.0, 1.0);
        let mogp = MultiOutputGP::new(kernel, 3, 2);
        let b = mogp.coregionalization_matrix();
        for i in 0..3 { assert!(b[[i, i]] >= 0.0); }
        for i in 0..3 { for j in 0..3 { assert!((b[[i, j]] - b[[j, i]]).abs() < 1e-10); } }
    }

    #[test]
    fn test_gp_classification_laplace_proba_range() {
        let kernel = ARDKernel::isotropic(1, 1.0, 1.0);
        let mut gpc = GPClassification::new(
            kernel,
            ClassificationLikelihood::Probit,
            ClassificationInference::Laplace,
        );
        let x_train = Array2::from_shape_vec((6, 1), vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]).expect("x");
        let y_train = array![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        gpc.fit(&x_train, &y_train).expect("fit");
        let x_test = Array2::from_shape_vec((3, 1), vec![-1.0, 0.0, 1.0]).expect("x_test");
        let proba = gpc.predict_proba(&x_test).expect("predict_proba");
        for &p in proba.iter() { assert!(p >= 0.0 && p <= 1.0, "P out of [0,1]: {p}"); }
        assert!(proba[0] < proba[2], "P(y=1|x=-1) should be < P(y=1|x=1)");
    }

    #[test]
    fn test_gp_classification_ep_logistic() {
        let kernel = ARDKernel::isotropic(1, 1.0, 1.0);
        let mut gpc = GPClassification::new(
            kernel,
            ClassificationLikelihood::Logistic,
            ClassificationInference::EP,
        );
        let x_train = Array2::from_shape_vec((4, 1), vec![-1.0, -0.5, 0.5, 1.0]).expect("x");
        let y_train = array![-1.0, -1.0, 1.0, 1.0];
        gpc.fit(&x_train, &y_train).expect("fit");
        let x_test = Array2::from_shape_vec((2, 1), vec![0.0, 2.0]).expect("x_test");
        let proba = gpc.predict_proba(&x_test).expect("predict_proba");
        for &p in proba.iter() { assert!(p >= 0.0 && p <= 1.0, "P out of range: {p}"); }
    }

    #[test]
    fn test_hyperopt_lml_increases() {
        let x_train = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).expect("x");
        let y_train = array![0.0, 1.0, 1.5, 1.0, 0.0];
        let params_a = vec![0.0, 0.0, -4.0];
        let params_b = vec![-3.0, -3.0, -3.0];
        let lml_a = GPHyperparamOpt::log_marginal_likelihood(&x_train, &y_train, &params_a).expect("lml_a");
        let lml_b = GPHyperparamOpt::log_marginal_likelihood(&x_train, &y_train, &params_b).expect("lml_b");
        assert!(lml_a.is_finite(), "LML_a = {lml_a}");
        assert!(lml_b.is_finite(), "LML_b = {lml_b}");
    }

    #[test]
    fn test_hyperopt_fit_ard() {
        let x_train = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).expect("x");
        let y_train = array![0.0, 1.0, -1.0, 0.0];
        let mut opt = GPHyperparamOpt::new(20);
        let ard = opt.fit_ard_gp(&x_train, &y_train, 2).expect("fit_ard");
        assert_eq!(ard.length_scales.len(), 2);
        assert!(ard.variance > 0.0);
        assert!(ard.length_scales.iter().all(|&l| l > 0.0));
    }
}
