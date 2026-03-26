//! Compile-time API stability checks.
//!
//! Each test verifies that a public API exists and is accessible.
//! Failures indicate potential breaking changes.

use scirs2_core::error::CoreError;

/// Verify CoreError is accessible and has expected variants.
#[test]
fn test_core_error_api() {
    use scirs2_core::error::ErrorContext;

    let ctx = ErrorContext::new("test".to_string());
    let _: CoreError = CoreError::InvalidArgument(ctx.clone());
    let _: CoreError = CoreError::ShapeError(ctx.clone());
    let _: CoreError = CoreError::NotImplementedError(ctx);
}

/// Verify CoreError implements std::error::Error and Display.
#[test]
fn test_core_error_trait_impls() {
    use scirs2_core::error::ErrorContext;
    use std::error::Error;

    let ctx = ErrorContext::new("sample".to_string());
    let e: CoreError = CoreError::ComputationError(ctx);
    let _: &dyn Error = &e;
    let _msg = format!("{e}");
}

/// Verify Array creation API via ndarray.
#[test]
fn test_ndarray_api() {
    use ndarray::{Array1, Array2};
    let _a: Array1<f64> = Array1::zeros(5);
    let _b: Array2<f64> = Array2::zeros((3, 4));
    let _c: Array2<f64> = Array2::eye(3);
}

/// Verify scirs2-linalg SVD is accessible with the three-arg signature.
#[test]
fn test_linalg_svd_api() {
    use ndarray::Array2;
    use scirs2_linalg::svd;

    let a = Array2::<f64>::eye(3);
    // svd(matrix_view, full_matrices, workers)
    let result = svd(&a.view(), true, None);
    assert!(result.is_ok(), "svd of identity matrix must succeed");
    let (u, s, vt) = result.expect("svd result");
    assert_eq!(u.shape(), &[3, 3]);
    assert_eq!(s.len(), 3);
    assert_eq!(vt.shape(), &[3, 3]);
}

/// Verify scirs2-linalg QR decomposition API.
#[test]
fn test_linalg_qr_api() {
    use ndarray::Array2;
    use scirs2_linalg::qr;

    let a = Array2::<f64>::eye(4);
    let result = qr(&a.view(), None);
    assert!(result.is_ok(), "qr of identity matrix must succeed");
}

/// Verify scirs2-linalg LU decomposition API.
#[test]
fn test_linalg_lu_api() {
    use ndarray::Array2;
    use scirs2_linalg::lu;

    let a = Array2::<f64>::eye(3);
    let result = lu(&a.view(), None);
    assert!(result.is_ok(), "lu of identity matrix must succeed");
}

/// Verify stats Normal distribution API — construction and core methods.
#[test]
fn test_stats_normal_construction() {
    use scirs2_stats::distributions::Normal;

    let d = Normal::new(0.0_f64, 1.0_f64).expect("valid params");
    // Direct methods on the struct
    let _ = d.pdf(0.0);
    let _ = d.cdf(0.0);
    let ppf_result = d.ppf(0.5);
    assert!(ppf_result.is_ok(), "ppf(0.5) must succeed");
}

/// Verify stats Normal distribution trait implementation.
#[test]
fn test_stats_normal_trait_api() {
    use scirs2_stats::distributions::Normal;
    use scirs2_stats::traits::{ContinuousDistribution, Distribution};

    let d = Normal::new(0.0_f64, 1.0_f64).expect("valid params");
    // Distribution trait
    let m: f64 = d.mean();
    let v: f64 = d.var();
    let _s: f64 = d.std();
    assert!((m - 0.0).abs() < 1e-10, "mean of N(0,1) must be 0");
    assert!((v - 1.0).abs() < 1e-10, "variance of N(0,1) must be 1");
    // ContinuousDistribution trait
    let _pdf_val: f64 = ContinuousDistribution::pdf(&d, 0.0);
    let _cdf_val: f64 = ContinuousDistribution::cdf(&d, 0.0);
}

/// Verify CoreError variants introduced in v0.3+ are still present.
#[test]
fn test_core_error_variant_coverage() {
    use scirs2_core::error::ErrorContext;

    let ctx = || ErrorContext::new("test".to_string());

    // All variants that external crates may match on must remain stable
    let variants: Vec<CoreError> = vec![
        CoreError::ComputationError(ctx()),
        CoreError::DomainError(ctx()),
        CoreError::ConvergenceError(ctx()),
        CoreError::DimensionError(ctx()),
        CoreError::ShapeError(ctx()),
        CoreError::IndexError(ctx()),
        CoreError::ValueError(ctx()),
        CoreError::TypeError(ctx()),
        CoreError::NotImplementedError(ctx()),
        CoreError::MemoryError(ctx()),
        CoreError::ConfigError(ctx()),
        CoreError::InvalidArgument(ctx()),
        CoreError::ValidationError(ctx()),
        CoreError::IoError(ctx()),
    ];

    for e in &variants {
        // Each must be displayable
        let _ = format!("{e}");
    }
    assert_eq!(variants.len(), 14, "all expected variants present");
}
