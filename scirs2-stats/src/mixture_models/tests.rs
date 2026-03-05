use super::*;

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    // -----------------------------------------------------------------------
    // Helper: generate well-separated 2D clusters for predictable GMM results
    // -----------------------------------------------------------------------
    fn two_cluster_data() -> Array2<f64> {
        array![
            [1.0, 2.0],
            [1.1, 2.1],
            [0.9, 1.9],
            [1.2, 2.2],
            [0.8, 1.8],
            [10.0, 12.0],
            [10.1, 12.1],
            [9.9, 11.9],
            [10.2, 12.2],
            [9.8, 11.8]
        ]
    }

    fn three_cluster_data() -> Array2<f64> {
        array![
            [1.0, 2.0],
            [1.1, 2.1],
            [0.9, 1.9],
            [5.0, 6.0],
            [5.1, 6.1],
            [4.9, 5.9],
            [10.0, 12.0],
            [10.1, 12.1],
            [9.9, 11.9]
        ]
    }

    // -----------------------------------------------------------------------
    // 1. Basic GMM creation and fitting
    // -----------------------------------------------------------------------
    #[test]
    fn test_gmm_basic_fit() {
        let data = two_cluster_data();
        let config = GMMConfig {
            max_iter: 100,
            tolerance: 1e-4,
            ..Default::default()
        };
        let mut gmm = GaussianMixtureModel::<f64>::new(2, config).expect("Test: new failed");
        let params = gmm.fit(&data.view()).expect("Test: fit failed");

        assert_eq!(params.weights.len(), 2);
        assert!(params.log_likelihood.is_finite());
        assert_eq!(params.means.nrows(), 2);
        assert_eq!(params.covariances.len(), 2);
    }

    // -----------------------------------------------------------------------
    // 2. GMM convergence
    // -----------------------------------------------------------------------
    #[test]
    fn test_gmm_convergence() {
        let data = two_cluster_data();
        let config = GMMConfig {
            max_iter: 200,
            tolerance: 1e-6,
            ..Default::default()
        };
        let mut gmm = GaussianMixtureModel::<f64>::new(2, config).expect("Test: new failed");
        let params = gmm.fit(&data.view()).expect("Test: fit failed");

        // With well-separated clusters, EM should converge
        assert!(params.converged);
        assert_eq!(
            params.convergence_reason,
            ConvergenceReason::LogLikelihoodTolerance
        );
    }

    // -----------------------------------------------------------------------
    // 3. Predict (hard assignment)
    // -----------------------------------------------------------------------
    #[test]
    fn test_gmm_predict() {
        let data = two_cluster_data();
        let config = GMMConfig {
            max_iter: 100,
            ..Default::default()
        };
        let mut gmm = GaussianMixtureModel::<f64>::new(2, config).expect("Test: new failed");
        gmm.fit(&data.view()).expect("Test: fit failed");

        let labels = gmm.predict(&data.view()).expect("Test: predict failed");
        assert_eq!(labels.len(), data.nrows());

        // Points in the same cluster should have the same label
        let label_a = labels[0]; // cluster around (1, 2)
        let label_b = labels[5]; // cluster around (10, 12)
        assert_ne!(label_a, label_b);

        // All first-cluster points should match
        for i in 0..5 {
            assert_eq!(labels[i], label_a);
        }
        for i in 5..10 {
            assert_eq!(labels[i], label_b);
        }
    }

    // -----------------------------------------------------------------------
    // 4. Predict probabilities (soft assignment)
    // -----------------------------------------------------------------------
    #[test]
    fn test_gmm_predict_proba() {
        let data = two_cluster_data();
        let mut gmm =
            GaussianMixtureModel::<f64>::new(2, GMMConfig::default()).expect("Test: new failed");
        gmm.fit(&data.view()).expect("Test: fit failed");

        let proba = gmm
            .predict_proba(&data.view())
            .expect("Test: predict_proba failed");
        assert_eq!(proba.dim(), (data.nrows(), 2));

        // Each row should sum to approximately 1
        for i in 0..proba.nrows() {
            let row_sum: f64 = proba.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-6, "Row {i} sums to {row_sum}");
        }
    }

    // -----------------------------------------------------------------------
    // 5. Score (average log-likelihood)
    // -----------------------------------------------------------------------
    #[test]
    fn test_gmm_score() {
        let data = two_cluster_data();
        let mut gmm =
            GaussianMixtureModel::<f64>::new(2, GMMConfig::default()).expect("Test: new failed");
        gmm.fit(&data.view()).expect("Test: fit failed");

        let avg_ll = gmm.score(&data.view()).expect("Test: score failed");
        assert!(avg_ll.is_finite());
        // Average log-likelihood can be positive when density > 1
        // (e.g. tight clusters with small variance).
        // Just verify it is finite and consistent with total LL / n.
        let total_ll = gmm
            .score_samples(&data.view())
            .expect("Test: score_samples failed")
            .sum();
        let expected = total_ll / data.nrows() as f64;
        assert!(
            (avg_ll - expected).abs() < 1e-10,
            "score ({avg_ll}) should equal mean of score_samples ({expected})"
        );
    }

    // -----------------------------------------------------------------------
    // 6. Score samples (per-sample log-likelihood)
    // -----------------------------------------------------------------------
    #[test]
    fn test_gmm_score_samples() {
        let data = two_cluster_data();
        let mut gmm =
            GaussianMixtureModel::<f64>::new(2, GMMConfig::default()).expect("Test: new failed");
        gmm.fit(&data.view()).expect("Test: fit failed");

        let scores = gmm
            .score_samples(&data.view())
            .expect("Test: score_samples failed");
        assert_eq!(scores.len(), data.nrows());
        for &s in scores.iter() {
            assert!(s.is_finite());
        }
    }

    // -----------------------------------------------------------------------
    // 7. Sample from fitted model
    // -----------------------------------------------------------------------
    #[test]
    fn test_gmm_sample() {
        let data = two_cluster_data();
        let mut gmm =
            GaussianMixtureModel::<f64>::new(2, GMMConfig::default()).expect("Test: new failed");
        gmm.fit(&data.view()).expect("Test: fit failed");

        let samples = gmm.sample(50, Some(42)).expect("Test: sample failed");
        assert_eq!(samples.dim(), (50, 2));

        // All samples should be finite
        for &v in samples.iter() {
            assert!(v.is_finite());
        }
    }

    // -----------------------------------------------------------------------
    // 8. BIC / AIC
    // -----------------------------------------------------------------------
    #[test]
    fn test_gmm_bic_aic() {
        let data = two_cluster_data();
        let mut gmm =
            GaussianMixtureModel::<f64>::new(2, GMMConfig::default()).expect("Test: new failed");
        gmm.fit(&data.view()).expect("Test: fit failed");

        let bic = gmm.bic(&data.view()).expect("Test: bic failed");
        let aic = gmm.aic(&data.view()).expect("Test: aic failed");
        assert!(bic.is_finite());
        assert!(aic.is_finite());

        // BIC typically penalizes more than AIC for moderate sample sizes
        // (BIC = -2LL + p*ln(n), AIC = -2LL + 2p; BIC > AIC when ln(n) > 2, i.e. n > 7)
        assert!(bic > aic, "BIC ({bic}) should be > AIC ({aic}) for n=10");
    }

    // -----------------------------------------------------------------------
    // 9. n_parameters
    // -----------------------------------------------------------------------
    #[test]
    fn test_gmm_n_parameters() {
        let data = two_cluster_data();
        // 2 components, 2 features, full covariance
        // weight params: 2-1 = 1
        // mean params: 2*2 = 4
        // cov params (full): 2 * 2*(2+1)/2 = 6
        // total = 11
        let mut gmm =
            GaussianMixtureModel::<f64>::new(2, GMMConfig::default()).expect("Test: new failed");
        gmm.fit(&data.view()).expect("Test: fit failed");
        let n_params = gmm.n_parameters().expect("Test: n_parameters failed");
        assert_eq!(n_params, 11);
    }

    // -----------------------------------------------------------------------
    // 10. n_parameters with diagonal covariance
    // -----------------------------------------------------------------------
    #[test]
    fn test_gmm_n_parameters_diagonal() {
        let data = two_cluster_data();
        let config = GMMConfig {
            covariance_type: CovarianceType::Diagonal,
            ..Default::default()
        };
        let mut gmm = GaussianMixtureModel::<f64>::new(2, config).expect("Test: new failed");
        gmm.fit(&data.view()).expect("Test: fit failed");
        // weight: 1, means: 4, diag cov: 2*2=4 => total=9
        let n_params = gmm.n_parameters().expect("Test: n_parameters failed");
        assert_eq!(n_params, 9);
    }

    // -----------------------------------------------------------------------
    // 11. n_parameters with spherical covariance
    // -----------------------------------------------------------------------
    #[test]
    fn test_gmm_n_parameters_spherical() {
        let data = two_cluster_data();
        let config = GMMConfig {
            covariance_type: CovarianceType::Spherical,
            ..Default::default()
        };
        let mut gmm = GaussianMixtureModel::<f64>::new(2, config).expect("Test: new failed");
        gmm.fit(&data.view()).expect("Test: fit failed");
        // weight: 1, means: 4, spherical cov: 2 => total=7
        let n_params = gmm.n_parameters().expect("Test: n_parameters failed");
        assert_eq!(n_params, 7);
    }

    // -----------------------------------------------------------------------
    // 12. Model selection (BIC-based)
    // -----------------------------------------------------------------------
    #[test]
    fn test_gmm_model_selection() {
        let data = three_cluster_data();
        let config = GMMConfig {
            max_iter: 100,
            ..Default::default()
        };
        let (best_n, params) = gmm_model_selection(&data.view(), 1, 4, Some(config))
            .expect("Test: model_selection failed");

        assert!(best_n >= 1 && best_n <= 4);
        assert!(params.model_selection.bic.is_finite());
    }

    // -----------------------------------------------------------------------
    // 13. select_n_components
    // -----------------------------------------------------------------------
    #[test]
    fn test_select_n_components_bic() {
        let data = two_cluster_data();
        let (best_k, scores) = select_n_components::<f64>(&data.view(), 4, "bic")
            .expect("Test: select_n_components failed");

        assert!(best_k >= 1 && best_k <= 4);
        assert_eq!(scores.len(), 4);
        for s in &scores {
            assert!(s.is_finite());
        }
    }

    // -----------------------------------------------------------------------
    // 14. select_n_components with AIC
    // -----------------------------------------------------------------------
    #[test]
    fn test_select_n_components_aic() {
        let data = two_cluster_data();
        let (best_k, scores) = select_n_components::<f64>(&data.view(), 3, "aic")
            .expect("Test: select_n_components AIC failed");

        assert!(best_k >= 1 && best_k <= 3);
        assert_eq!(scores.len(), 3);
    }

    // -----------------------------------------------------------------------
    // 15. Single component GMM
    // -----------------------------------------------------------------------
    #[test]
    fn test_gmm_single_component() {
        let data = array![[1.0, 2.0], [1.1, 2.1], [0.9, 1.9], [1.2, 1.8], [0.8, 2.2]];
        let mut gmm =
            GaussianMixtureModel::<f64>::new(1, GMMConfig::default()).expect("Test: new failed");
        let params = gmm.fit(&data.view()).expect("Test: fit failed");

        assert_eq!(params.weights.len(), 1);
        assert!((params.weights[0] - 1.0).abs() < 1e-6);
        assert!(params.log_likelihood.is_finite());

        // Mean should be near the data center
        let mean_x = params.means[[0, 0]];
        let mean_y = params.means[[0, 1]];
        assert!((mean_x - 1.0).abs() < 0.3);
        assert!((mean_y - 2.0).abs() < 0.3);
    }

    // -----------------------------------------------------------------------
    // 16. Weights sum to 1
    // -----------------------------------------------------------------------
    #[test]
    fn test_gmm_weights_sum_to_one() {
        let data = three_cluster_data();
        let mut gmm =
            GaussianMixtureModel::<f64>::new(3, GMMConfig::default()).expect("Test: new failed");
        let params = gmm.fit(&data.view()).expect("Test: fit failed");

        let weight_sum: f64 = params.weights.sum();
        assert!(
            (weight_sum - 1.0).abs() < 1e-6,
            "Weights sum to {weight_sum}"
        );
    }

    // -----------------------------------------------------------------------
    // 17. Robust GMM
    // -----------------------------------------------------------------------
    #[test]
    fn test_robust_gmm() {
        let data = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [0.9, 1.9],
            [100.0, 100.0], // Outlier
            [5.0, 6.0],
            [5.1, 6.1]
        ];

        let mut robust_gmm = RobustGMM::new(2, 0.1f64, 0.2f64, GMMConfig::default())
            .expect("Test: RobustGMM new failed");

        let params = robust_gmm
            .fit(&data.view())
            .expect("Test: RobustGMM fit failed");
        assert!(params.outlier_scores.is_some());

        let outliers = robust_gmm
            .detect_outliers(&data.view())
            .expect("Test: detect_outliers failed");
        assert_eq!(outliers.len(), data.nrows());
    }

    // -----------------------------------------------------------------------
    // 18. Streaming GMM
    // -----------------------------------------------------------------------
    #[test]
    fn test_streaming_gmm() {
        let batch1 = array![[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]];
        let batch2 = array![[5.0, 6.0], [5.1, 6.1], [4.9, 5.9]];

        let mut sgmm = StreamingGMM::new(2, 0.1f64, 0.9f64, GMMConfig::default())
            .expect("Test: StreamingGMM new failed");

        sgmm.partial_fit(&batch1.view())
            .expect("Test: partial_fit batch1 failed");
        sgmm.partial_fit(&batch2.view())
            .expect("Test: partial_fit batch2 failed");

        let params = sgmm.get_parameters().expect("Test: get_parameters failed");
        assert_eq!(params.weights.len(), 2);
    }

    // -----------------------------------------------------------------------
    // 19. Variational GMM
    // -----------------------------------------------------------------------
    #[test]
    fn test_variational_gmm() {
        let data = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [0.9, 1.9],
            [5.0, 6.0],
            [5.1, 6.1],
            [4.9, 5.9]
        ];

        let mut vgmm = VariationalGMM::new(2, VariationalGMMConfig::default());
        let result = vgmm.fit(&data.view()).expect("Test: VGMM fit failed");

        assert!(result.lower_bound > f64::NEG_INFINITY);
        assert!(result.effective_components > 0);
    }

    // -----------------------------------------------------------------------
    // 20. KDE basic evaluation
    // -----------------------------------------------------------------------
    #[test]
    fn test_kde_basic() {
        let data = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let points = array![[2.5], [3.0], [10.0]];

        let densities = kernel_density_estimation(
            &data.view(),
            &points.view(),
            Some(KernelType::Gaussian),
            Some(1.0),
        )
        .expect("Test: KDE failed");

        assert_eq!(densities.len(), 3);
        // Density near the data should be higher than far away
        assert!(densities[0] > densities[2], "Density near data > far away");
        assert!(densities[1] > densities[2]);
    }

    // -----------------------------------------------------------------------
    // 21. KDE with different kernels
    // -----------------------------------------------------------------------
    #[test]
    fn test_kde_kernels() {
        let data = array![[0.0], [1.0], [2.0]];
        let points = array![[1.0]];

        for kernel in &[
            KernelType::Gaussian,
            KernelType::Epanechnikov,
            KernelType::Uniform,
            KernelType::Triangular,
            KernelType::Cosine,
        ] {
            let d = kernel_density_estimation(
                &data.view(),
                &points.view(),
                Some(kernel.clone()),
                Some(1.0),
            )
            .expect("Test: KDE kernel variant failed");
            assert!(
                d[0] > 0.0,
                "Kernel {:?} should give positive density",
                kernel
            );
        }
    }

    // -----------------------------------------------------------------------
    // 22. GMM error: not fitted before predict
    // -----------------------------------------------------------------------
    #[test]
    fn test_gmm_not_fitted_error() {
        let gmm =
            GaussianMixtureModel::<f64>::new(2, GMMConfig::default()).expect("Test: new failed");
        let data = array![[1.0, 2.0]];
        let err = gmm.predict(&data.view());
        assert!(err.is_err());
    }

    // -----------------------------------------------------------------------
    // 23. GMM error: too few samples
    // -----------------------------------------------------------------------
    #[test]
    fn test_gmm_too_few_samples_error() {
        let data = array![[1.0, 2.0]]; // 1 sample, 3 components
        let mut gmm =
            GaussianMixtureModel::<f64>::new(3, GMMConfig::default()).expect("Test: new failed");
        let err = gmm.fit(&data.view());
        assert!(err.is_err());
    }

    // -----------------------------------------------------------------------
    // 24. GMM cross-validation
    // -----------------------------------------------------------------------
    #[test]
    fn test_gmm_cross_validation() {
        let data = two_cluster_data();
        let config = GMMConfig {
            max_iter: 50,
            ..Default::default()
        };
        let cv_score = gmm_cross_validation(&data.view(), 2, 2, config).expect("Test: CV failed");
        assert!(cv_score.is_finite());
    }

    // -----------------------------------------------------------------------
    // 25. Convenience function gaussian_mixture_model
    // -----------------------------------------------------------------------
    #[test]
    fn test_convenience_gaussian_mixture_model() {
        let data = two_cluster_data();
        let params =
            gaussian_mixture_model(&data.view(), 2, None).expect("Test: convenience fn failed");
        assert_eq!(params.weights.len(), 2);
        assert!(params.log_likelihood.is_finite());
    }

    // -----------------------------------------------------------------------
    // 26. hierarchical_gmm_init
    // -----------------------------------------------------------------------
    #[test]
    fn test_hierarchical_gmm_init() {
        let data = two_cluster_data();
        let params = hierarchical_gmm_init(&data.view(), 2, GMMConfig::default())
            .expect("Test: hierarchical init failed");
        assert_eq!(params.weights.len(), 2);
    }

    // -----------------------------------------------------------------------
    // 27. Model selection criteria are all finite
    // -----------------------------------------------------------------------
    #[test]
    fn test_model_selection_criteria_finite() {
        let data = two_cluster_data();
        let mut gmm =
            GaussianMixtureModel::<f64>::new(2, GMMConfig::default()).expect("Test: new failed");
        let params = gmm.fit(&data.view()).expect("Test: fit failed");

        assert!(params.model_selection.aic.is_finite());
        assert!(params.model_selection.bic.is_finite());
        assert!(params.model_selection.icl.is_finite());
        assert!(params.model_selection.hqic.is_finite());
        assert!(params.model_selection.n_parameters > 0);
    }

    // -----------------------------------------------------------------------
    // 28. Component diagnostics populated
    // -----------------------------------------------------------------------
    #[test]
    fn test_component_diagnostics() {
        let data = two_cluster_data();
        let mut gmm =
            GaussianMixtureModel::<f64>::new(2, GMMConfig::default()).expect("Test: new failed");
        let params = gmm.fit(&data.view()).expect("Test: fit failed");

        assert_eq!(params.component_diagnostics.len(), 2);
        for diag in &params.component_diagnostics {
            assert!(diag.effective_samplesize.is_finite());
            assert!(diag.condition_number.is_finite());
            assert!(diag.covariance_determinant.is_finite());
            assert!(diag.component_separation.is_finite());
        }
    }

    // -----------------------------------------------------------------------
    // 29. Responsibilities stored after fit
    // -----------------------------------------------------------------------
    #[test]
    fn test_responsibilities_stored() {
        let data = two_cluster_data();
        let mut gmm =
            GaussianMixtureModel::<f64>::new(2, GMMConfig::default()).expect("Test: new failed");
        let params = gmm.fit(&data.view()).expect("Test: fit failed");

        assert!(params.responsibilities.is_some());
        let resp = params
            .responsibilities
            .as_ref()
            .expect("Test: no responsibilities");
        assert_eq!(resp.dim(), (10, 2));
    }

    // -----------------------------------------------------------------------
    // 30. Random initialization method
    // -----------------------------------------------------------------------
    #[test]
    fn test_gmm_random_init() {
        let data = two_cluster_data();
        let config = GMMConfig {
            init_method: InitializationMethod::Random,
            seed: Some(42),
            ..Default::default()
        };
        let mut gmm = GaussianMixtureModel::<f64>::new(2, config).expect("Test: new failed");
        let params = gmm.fit(&data.view()).expect("Test: fit failed");
        assert!(params.log_likelihood.is_finite());
    }

    // -----------------------------------------------------------------------
    // 31. BIC for 1 component vs 2 components (2 is better for 2-cluster data)
    // -----------------------------------------------------------------------
    #[test]
    fn test_bic_prefers_correct_k() {
        let data = two_cluster_data();

        let mut gmm1 = GaussianMixtureModel::<f64>::new(1, GMMConfig::default())
            .expect("Test: new k=1 failed");
        gmm1.fit(&data.view()).expect("Test: fit k=1 failed");
        let bic1 = gmm1.bic(&data.view()).expect("Test: bic k=1 failed");

        let mut gmm2 = GaussianMixtureModel::<f64>::new(2, GMMConfig::default())
            .expect("Test: new k=2 failed");
        gmm2.fit(&data.view()).expect("Test: fit k=2 failed");
        let bic2 = gmm2.bic(&data.view()).expect("Test: bic k=2 failed");

        // BIC for k=2 should be lower (better) for 2-cluster data
        assert!(bic2 < bic1, "BIC k=2 ({bic2}) should be < BIC k=1 ({bic1})");
    }

    // -----------------------------------------------------------------------
    // 32. select_n_components error on max_k=0
    // -----------------------------------------------------------------------
    #[test]
    fn test_select_n_components_zero_error() {
        let data = two_cluster_data();
        let result = select_n_components::<f64>(&data.view(), 0, "bic");
        assert!(result.is_err());
    }
}
