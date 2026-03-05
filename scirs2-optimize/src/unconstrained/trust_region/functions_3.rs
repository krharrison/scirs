//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::unconstrained::Options;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

use super::types::TrustRegionConfig;

#[cfg(test)]
mod tests {
    use super::super::functions::{
        cauchy_point, dogleg_step, minimize_trust_krylov, minimize_trust_ncg,
        solve_trust_subproblem, trust_region_minimize,
    };
    use super::super::functions_2::minimize_trust_exact;
    use super::*;
    use approx::assert_abs_diff_eq;
    #[test]
    fn test_trust_ncg_quadratic() {
        let quadratic = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + 4.0 * x[1] * x[1] };
        let x0 = Array1::from_vec(vec![2.0, 1.0]);
        let options = Options::default();
        let result = minimize_trust_ncg(quadratic, x0, &options).expect("Operation failed");
        assert!(result.success);
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 1e-4);
    }
    #[test]
    fn test_trust_krylov_rosenbrock() {
        let rosenbrock = |x: &ArrayView1<f64>| -> f64 {
            let a = 1.0;
            let b = 100.0;
            (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
        };
        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let mut options = Options::default();
        options.max_iter = 2000;
        let result = minimize_trust_krylov(rosenbrock, x0, &options).expect("Operation failed");
        assert!(result.nit > 0, "Should make at least some progress");
        assert!(
            result.x[0] >= -0.1 && result.x[0] <= 1.5,
            "x[0] = {} should be near 1.0",
            result.x[0]
        );
        assert!(
            result.x[1] >= -0.1 && result.x[1] <= 1.5,
            "x[1] = {} should be near 1.0",
            result.x[1]
        );
    }
    #[test]
    fn test_trust_exact_simple() {
        let quadratic =
            |x: &ArrayView1<f64>| -> f64 { (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2) };
        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let options = Options::default();
        let result = minimize_trust_exact(quadratic, x0, &options).expect("Operation failed");
        assert!(result.success);
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(result.x[1], 2.0, epsilon = 1e-4);
    }
    #[test]
    fn test_config_default_is_valid() {
        let config = TrustRegionConfig::default();
        assert!(config.validate().is_ok());
    }
    #[test]
    fn test_config_invalid_initial_radius() {
        let mut config = TrustRegionConfig::default();
        config.initial_radius = -1.0;
        assert!(config.validate().is_err());
        config.initial_radius = 0.0;
        assert!(config.validate().is_err());
    }
    #[test]
    fn test_config_invalid_max_radius() {
        let mut config = TrustRegionConfig::default();
        config.max_radius = 0.5;
        assert!(config.validate().is_err());
    }
    #[test]
    fn test_config_invalid_eta() {
        let mut config = TrustRegionConfig::default();
        config.eta1 = 0.0;
        assert!(config.validate().is_err());
        config.eta1 = 1.0;
        assert!(config.validate().is_err());
        config.eta1 = 0.25;
        config.eta2 = 0.1;
        assert!(config.validate().is_err());
    }
    #[test]
    fn test_config_invalid_gamma() {
        let mut config = TrustRegionConfig::default();
        config.gamma1 = 0.0;
        assert!(config.validate().is_err());
        config.gamma1 = 1.0;
        assert!(config.validate().is_err());
        config.gamma1 = 0.25;
        config.gamma2 = 0.5;
        assert!(config.validate().is_err());
    }
    #[test]
    fn test_cauchy_point_zero_gradient() {
        let gradient = Array1::from_vec(vec![0.0, 0.0]);
        let hessian =
            Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 2.0]).expect("valid shape");
        let (step, hits_boundary) = cauchy_point(&gradient, &hessian, 1.0);
        assert_abs_diff_eq!(step[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(step[1], 0.0, epsilon = 1e-10);
        assert!(!hits_boundary);
    }
    #[test]
    fn test_cauchy_point_positive_definite_within_trust_region() {
        let gradient = Array1::from_vec(vec![1.0, 0.0]);
        let hessian =
            Array2::from_shape_vec((2, 2), vec![10.0, 0.0, 0.0, 10.0]).expect("valid shape");
        let (step, hits_boundary) = cauchy_point(&gradient, &hessian, 10.0);
        assert!(
            step[0] < 0.0,
            "Step should be in negative gradient direction"
        );
        assert_abs_diff_eq!(step[1], 0.0, epsilon = 1e-10);
        assert!(!hits_boundary);
    }
    #[test]
    fn test_cauchy_point_hits_boundary_with_small_radius() {
        let gradient = Array1::from_vec(vec![1.0, 1.0]);
        let hessian =
            Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 2.0]).expect("valid shape");
        let (step, hits_boundary) = cauchy_point(&gradient, &hessian, 0.01);
        let step_norm: f64 = step.dot(&step).sqrt();
        assert_abs_diff_eq!(step_norm, 0.01, epsilon = 1e-10);
        assert!(hits_boundary);
    }
    #[test]
    fn test_cauchy_point_negative_curvature() {
        let gradient = Array1::from_vec(vec![1.0, 0.0]);
        let hessian =
            Array2::from_shape_vec((2, 2), vec![-2.0, 0.0, 0.0, -2.0]).expect("valid shape");
        let (step, hits_boundary) = cauchy_point(&gradient, &hessian, 1.0);
        let step_norm: f64 = step.dot(&step).sqrt();
        assert_abs_diff_eq!(step_norm, 1.0, epsilon = 1e-10);
        assert!(hits_boundary);
    }
    #[test]
    fn test_dogleg_step_newton_within_trust_region() {
        let gradient = Array1::from_vec(vec![2.0, 2.0]);
        let hessian =
            Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 2.0]).expect("valid shape");
        let (step, hits_boundary) = dogleg_step(&gradient, &hessian, 10.0);
        assert_abs_diff_eq!(step[0], -1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(step[1], -1.0, epsilon = 1e-6);
        assert!(!hits_boundary);
    }
    #[test]
    fn test_dogleg_step_small_trust_region() {
        let gradient = Array1::from_vec(vec![2.0, 2.0]);
        let hessian =
            Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 2.0]).expect("valid shape");
        let (step, hits_boundary) = dogleg_step(&gradient, &hessian, 0.1);
        let step_norm: f64 = step.dot(&step).sqrt();
        assert_abs_diff_eq!(step_norm, 0.1, epsilon = 1e-4);
        assert!(hits_boundary);
    }
    #[test]
    fn test_dogleg_step_zero_gradient() {
        let gradient = Array1::from_vec(vec![0.0, 0.0]);
        let hessian =
            Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 2.0]).expect("valid shape");
        let (step, hits_boundary) = dogleg_step(&gradient, &hessian, 1.0);
        let step_norm: f64 = step.dot(&step).sqrt();
        assert_abs_diff_eq!(step_norm, 0.0, epsilon = 1e-10);
        assert!(!hits_boundary);
    }
    #[test]
    fn test_solve_trust_subproblem_positive_predicted_reduction() {
        let gradient = Array1::from_vec(vec![2.0, 4.0]);
        let hessian =
            Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 2.0]).expect("valid shape");
        let (step, predicted_reduction, _hits_boundary) =
            solve_trust_subproblem(&gradient, &hessian, 10.0);
        assert!(
            predicted_reduction > 0.0,
            "Predicted reduction should be positive, got {}",
            predicted_reduction
        );
        assert_abs_diff_eq!(step[0], -1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(step[1], -2.0, epsilon = 1e-6);
    }
    #[test]
    fn test_trust_region_minimize_simple_quadratic() {
        let objective = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + x[1] * x[1] };
        let gradient_fn =
            |x: &ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![2.0 * x[0], 2.0 * x[1]]) };
        let hessian_fn = |_x: &ArrayView1<f64>| -> Array2<f64> {
            Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 2.0]).expect("valid shape")
        };
        let x0 = Array1::from_vec(vec![5.0, 3.0]);
        let result =
            trust_region_minimize(objective, Some(gradient_fn), Some(hessian_fn), x0, None)
                .expect("Optimization failed");
        assert!(result.converged, "Should converge: {}", result.message);
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.f_val, 0.0, epsilon = 1e-10);
    }
    #[test]
    fn test_trust_region_minimize_shifted_quadratic() {
        let objective =
            |x: &ArrayView1<f64>| -> f64 { (x[0] - 3.0).powi(2) + (x[1] + 2.0).powi(2) };
        let gradient_fn = |x: &ArrayView1<f64>| -> Array1<f64> {
            Array1::from_vec(vec![2.0 * (x[0] - 3.0), 2.0 * (x[1] + 2.0)])
        };
        let hessian_fn = |_x: &ArrayView1<f64>| -> Array2<f64> {
            Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 2.0]).expect("valid shape")
        };
        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let result =
            trust_region_minimize(objective, Some(gradient_fn), Some(hessian_fn), x0, None)
                .expect("Optimization failed");
        assert!(result.converged, "Should converge: {}", result.message);
        assert_abs_diff_eq!(result.x[0], 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.x[1], -2.0, epsilon = 1e-6);
    }
    #[test]
    fn test_trust_region_minimize_anisotropic_quadratic() {
        let objective = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + 100.0 * x[1] * x[1] };
        let gradient_fn = |x: &ArrayView1<f64>| -> Array1<f64> {
            Array1::from_vec(vec![2.0 * x[0], 200.0 * x[1]])
        };
        let hessian_fn = |_x: &ArrayView1<f64>| -> Array2<f64> {
            Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 200.0]).expect("valid shape")
        };
        let x0 = Array1::from_vec(vec![10.0, 10.0]);
        let result =
            trust_region_minimize(objective, Some(gradient_fn), Some(hessian_fn), x0, None)
                .expect("Optimization failed");
        assert!(result.converged, "Should converge: {}", result.message);
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-5);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 1e-5);
    }
    #[test]
    fn test_trust_region_minimize_rosenbrock_with_analytic() {
        let objective = |x: &ArrayView1<f64>| -> f64 {
            (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
        };
        let gradient_fn = |x: &ArrayView1<f64>| -> Array1<f64> {
            let g0 = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2));
            let g1 = 200.0 * (x[1] - x[0].powi(2));
            Array1::from_vec(vec![g0, g1])
        };
        let hessian_fn = |x: &ArrayView1<f64>| -> Array2<f64> {
            let h00 = 2.0 - 400.0 * x[1] + 1200.0 * x[0].powi(2);
            let h01 = -400.0 * x[0];
            let h11 = 200.0;
            Array2::from_shape_vec((2, 2), vec![h00, h01, h01, h11]).expect("valid shape")
        };
        let x0 = Array1::from_vec(vec![-1.0, 1.0]);
        let mut config = TrustRegionConfig::default();
        config.max_iter = 5000;
        config.tolerance = 1e-8;
        let result = trust_region_minimize(
            objective,
            Some(gradient_fn),
            Some(hessian_fn),
            x0,
            Some(config),
        )
        .expect("Optimization failed");
        assert!(result.converged, "Should converge: {}", result.message);
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 1e-4);
        assert!(result.f_val < 1e-6, "Function value should be near zero");
    }
    #[test]
    fn test_trust_region_minimize_quadratic_finite_diff() {
        let objective =
            |x: &ArrayView1<f64>| -> f64 { (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2) };
        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let result = trust_region_minimize(
            objective,
            None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
            None::<fn(&ArrayView1<f64>) -> Array2<f64>>,
            x0,
            None,
        )
        .expect("Optimization failed");
        assert!(result.converged, "Should converge: {}", result.message);
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(result.x[1], 2.0, epsilon = 1e-4);
    }
    #[test]
    fn test_trust_region_minimize_rosenbrock_finite_diff() {
        let objective = |x: &ArrayView1<f64>| -> f64 {
            (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
        };
        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let mut config = TrustRegionConfig::default();
        config.max_iter = 5000;
        let result = trust_region_minimize(
            objective,
            None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
            None::<fn(&ArrayView1<f64>) -> Array2<f64>>,
            x0,
            Some(config),
        )
        .expect("Optimization failed");
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-2);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 1e-2);
    }
    #[test]
    fn test_trust_region_ill_conditioned_quadratic() {
        let objective = |x: &ArrayView1<f64>| -> f64 { 0.5 * (x[0] * x[0] + 1000.0 * x[1] * x[1]) };
        let gradient_fn =
            |x: &ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![x[0], 1000.0 * x[1]]) };
        let hessian_fn = |_x: &ArrayView1<f64>| -> Array2<f64> {
            Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1000.0]).expect("valid shape")
        };
        let x0 = Array1::from_vec(vec![10.0, 10.0]);
        let result =
            trust_region_minimize(objective, Some(gradient_fn), Some(hessian_fn), x0, None)
                .expect("Optimization failed");
        assert!(result.converged, "Should converge: {}", result.message);
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-5);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 1e-5);
    }
    #[test]
    fn test_trust_region_ill_conditioned_rotated() {
        let cos45 = std::f64::consts::FRAC_1_SQRT_2;
        let sin45 = std::f64::consts::FRAC_1_SQRT_2;
        let a00 = cos45 * cos45 + 1000.0 * sin45 * sin45;
        let a01 = cos45 * sin45 * (1.0 - 1000.0);
        let a11 = sin45 * sin45 + 1000.0 * cos45 * cos45;
        let objective = move |x: &ArrayView1<f64>| -> f64 {
            0.5 * (a00 * x[0] * x[0] + 2.0 * a01 * x[0] * x[1] + a11 * x[1] * x[1])
        };
        let gradient_fn = move |x: &ArrayView1<f64>| -> Array1<f64> {
            Array1::from_vec(vec![a00 * x[0] + a01 * x[1], a01 * x[0] + a11 * x[1]])
        };
        let hessian_fn = move |_x: &ArrayView1<f64>| -> Array2<f64> {
            Array2::from_shape_vec((2, 2), vec![a00, a01, a01, a11]).expect("valid shape")
        };
        let x0 = Array1::from_vec(vec![5.0, -3.0]);
        let result =
            trust_region_minimize(objective, Some(gradient_fn), Some(hessian_fn), x0, None)
                .expect("Optimization failed");
        assert!(result.converged, "Should converge: {}", result.message);
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 1e-4);
    }
    #[test]
    fn test_trust_region_minimize_3d_quadratic() {
        let objective = |x: &ArrayView1<f64>| -> f64 {
            (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2) + (x[2] - 3.0).powi(2)
        };
        let gradient_fn = |x: &ArrayView1<f64>| -> Array1<f64> {
            Array1::from_vec(vec![
                2.0 * (x[0] - 1.0),
                2.0 * (x[1] - 2.0),
                2.0 * (x[2] - 3.0),
            ])
        };
        let hessian_fn = |_x: &ArrayView1<f64>| -> Array2<f64> {
            let mut h = Array2::zeros((3, 3));
            h[[0, 0]] = 2.0;
            h[[1, 1]] = 2.0;
            h[[2, 2]] = 2.0;
            h
        };
        let x0 = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let result =
            trust_region_minimize(objective, Some(gradient_fn), Some(hessian_fn), x0, None)
                .expect("Optimization failed");
        assert!(result.converged, "Should converge: {}", result.message);
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.x[1], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.x[2], 3.0, epsilon = 1e-6);
    }
    #[test]
    fn test_trust_region_custom_config() {
        let objective = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + x[1] * x[1] };
        let gradient_fn =
            |x: &ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![2.0 * x[0], 2.0 * x[1]]) };
        let hessian_fn = |_x: &ArrayView1<f64>| -> Array2<f64> {
            Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 2.0]).expect("valid shape")
        };
        let config = TrustRegionConfig {
            initial_radius: 0.5,
            max_radius: 50.0,
            eta1: 0.1,
            eta2: 0.9,
            gamma1: 0.5,
            gamma2: 1.5,
            max_iter: 500,
            tolerance: 1e-10,
            ftol: 1e-14,
            eps: 1e-8,
            min_radius: 1e-16,
        };
        let x0 = Array1::from_vec(vec![10.0, 10.0]);
        let result = trust_region_minimize(
            objective,
            Some(gradient_fn),
            Some(hessian_fn),
            x0,
            Some(config),
        )
        .expect("Optimization failed");
        assert!(result.converged);
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-8);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 1e-8);
    }
    #[test]
    fn test_trust_region_result_fields() {
        let objective = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] };
        let gradient_fn =
            |x: &ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![2.0 * x[0]]) };
        let hessian_fn = |_x: &ArrayView1<f64>| -> Array2<f64> {
            Array2::from_shape_vec((1, 1), vec![2.0]).expect("valid shape")
        };
        let x0 = Array1::from_vec(vec![5.0]);
        let result =
            trust_region_minimize(objective, Some(gradient_fn), Some(hessian_fn), x0, None)
                .expect("Optimization failed");
        assert!(result.converged);
        assert!(result.n_iter > 0, "Should take at least one iteration");
        assert!(result.n_fev > 0, "Should evaluate function");
        assert!(result.n_gev > 0, "Should evaluate gradient");
        assert!(result.n_hev > 0, "Should evaluate Hessian");
        assert!(
            result.trust_radius_final > 0.0,
            "Trust radius should be positive"
        );
        assert!(
            result.grad_norm < 1e-6,
            "Gradient norm should be small at convergence"
        );
        assert!(!result.message.is_empty(), "Should have a message");
    }
    #[test]
    fn test_trust_region_already_at_minimum() {
        let objective = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + x[1] * x[1] };
        let gradient_fn =
            |x: &ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![2.0 * x[0], 2.0 * x[1]]) };
        let hessian_fn = |_x: &ArrayView1<f64>| -> Array2<f64> {
            Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 2.0]).expect("valid shape")
        };
        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let result =
            trust_region_minimize(objective, Some(gradient_fn), Some(hessian_fn), x0, None)
                .expect("Optimization failed");
        assert!(result.converged);
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 1e-10);
        assert!(
            result.n_iter <= 1,
            "Should converge immediately at the minimum"
        );
    }
    #[test]
    fn test_trust_region_empty_initial_guess() {
        let objective = |_x: &ArrayView1<f64>| -> f64 { 0.0 };
        let x0 = Array1::from_vec(vec![]);
        let result = trust_region_minimize(
            objective,
            None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
            None::<fn(&ArrayView1<f64>) -> Array2<f64>>,
            x0,
            None,
        );
        assert!(result.is_err(), "Should fail with empty initial guess");
    }
    #[test]
    fn test_trust_region_1d_quadratic() {
        let objective = |x: &ArrayView1<f64>| -> f64 { (x[0] - 7.0).powi(2) };
        let gradient_fn =
            |x: &ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![2.0 * (x[0] - 7.0)]) };
        let hessian_fn = |_x: &ArrayView1<f64>| -> Array2<f64> {
            Array2::from_shape_vec((1, 1), vec![2.0]).expect("valid shape")
        };
        let x0 = Array1::from_vec(vec![0.0]);
        let result =
            trust_region_minimize(objective, Some(gradient_fn), Some(hessian_fn), x0, None)
                .expect("Optimization failed");
        assert!(result.converged);
        assert_abs_diff_eq!(result.x[0], 7.0, epsilon = 1e-6);
    }
    #[test]
    fn test_trust_region_booth_function() {
        let objective = |x: &ArrayView1<f64>| -> f64 {
            (x[0] + 2.0 * x[1] - 7.0).powi(2) + (2.0 * x[0] + x[1] - 5.0).powi(2)
        };
        let gradient_fn = |x: &ArrayView1<f64>| -> Array1<f64> {
            let g0 = 2.0 * (x[0] + 2.0 * x[1] - 7.0) + 4.0 * (2.0 * x[0] + x[1] - 5.0);
            let g1 = 4.0 * (x[0] + 2.0 * x[1] - 7.0) + 2.0 * (2.0 * x[0] + x[1] - 5.0);
            Array1::from_vec(vec![g0, g1])
        };
        let hessian_fn = |_x: &ArrayView1<f64>| -> Array2<f64> {
            Array2::from_shape_vec((2, 2), vec![10.0, 8.0, 8.0, 10.0]).expect("valid shape")
        };
        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let result =
            trust_region_minimize(objective, Some(gradient_fn), Some(hessian_fn), x0, None)
                .expect("Optimization failed");
        assert!(result.converged, "Should converge: {}", result.message);
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(result.x[1], 3.0, epsilon = 1e-5);
        assert!(result.f_val < 1e-8, "Function value should be near zero");
    }
    #[test]
    fn test_trust_region_max_iter_limit() {
        let objective = |x: &ArrayView1<f64>| -> f64 {
            (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
        };
        let x0 = Array1::from_vec(vec![-5.0, -5.0]);
        let mut config = TrustRegionConfig::default();
        config.max_iter = 3;
        let result = trust_region_minimize(
            objective,
            None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
            None::<fn(&ArrayView1<f64>) -> Array2<f64>>,
            x0,
            Some(config),
        )
        .expect("Optimization should not error");
        assert!(!result.converged || result.n_iter <= 3);
        assert!(result.n_iter <= 3, "Should not exceed max_iter");
    }
}
