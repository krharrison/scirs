//! Tests for SIMD-accelerated batch operations.
//!
//! Each test compares the SIMD batch result against the scalar implementation
//! to verify correctness within acceptable epsilon bounds.

use super::batch::*;
use approx::assert_relative_eq;
use scirs2_core::ndarray::Array1;

// ============================================================================
// Gamma function tests
// ============================================================================

#[test]
fn test_batch_gamma_f64_integer_values() {
    let input = Array1::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = batch_gamma_f64(&input.view()).expect("batch_gamma_f64 failed");

    // Gamma(n) = (n-1)!
    let expected = [1.0, 1.0, 2.0, 6.0, 24.0, 120.0];
    for (i, &exp) in expected.iter().enumerate() {
        assert_relative_eq!(result[i], exp, epsilon = 1e-8, max_relative = 1e-8);
    }
}

#[test]
fn test_batch_gamma_f64_half_integer() {
    let input = Array1::from_vec(vec![0.5f64, 1.5, 2.5, 3.5]);
    let result = batch_gamma_f64(&input.view()).expect("batch_gamma_f64 failed");

    let sqrt_pi = std::f64::consts::PI.sqrt();
    let expected = [
        sqrt_pi,              // Gamma(0.5) = sqrt(pi)
        sqrt_pi / 2.0,        // Gamma(1.5) = sqrt(pi)/2
        3.0 * sqrt_pi / 4.0,  // Gamma(2.5) = 3*sqrt(pi)/4
        15.0 * sqrt_pi / 8.0, // Gamma(3.5) = 15*sqrt(pi)/8
    ];

    for (i, &exp) in expected.iter().enumerate() {
        assert_relative_eq!(result[i], exp, epsilon = 1e-8, max_relative = 1e-8);
    }
}

#[test]
fn test_batch_gamma_f64_vs_scalar() {
    let input: Array1<f64> = Array1::from_vec((1..=20).map(|i| 0.5 + i as f64 * 0.3).collect());
    let batch_result = batch_gamma_f64(&input.view()).expect("batch_gamma_f64 failed");

    for i in 0..input.len() {
        let scalar_result = crate::gamma::gamma(input[i]);
        assert_relative_eq!(
            batch_result[i],
            scalar_result,
            epsilon = 1e-6,
            max_relative = 1e-6
        );
    }
}

#[test]
fn test_batch_gamma_f32_basic() {
    let input = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]);
    let result = batch_gamma_f32(&input.view()).expect("batch_gamma_f32 failed");

    let expected = [1.0f32, 1.0, 2.0, 6.0, 24.0];
    for (i, &exp) in expected.iter().enumerate() {
        assert_relative_eq!(result[i], exp, epsilon = 1e-4);
    }
}

#[test]
fn test_batch_gamma_f64_empty() {
    let input = Array1::<f64>::zeros(0);
    let result = batch_gamma_f64(&input.view()).expect("batch_gamma_f64 failed on empty");
    assert_eq!(result.len(), 0);
}

// ============================================================================
// Log-Gamma function tests
// ============================================================================

#[test]
fn test_batch_lgamma_f64_basic() {
    let input = Array1::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 10.0]);
    let result = batch_lgamma_f64(&input.view()).expect("batch_lgamma_f64 failed");

    // lgamma(n) = ln((n-1)!)
    let expected = [
        0.0_f64.ln_1p(),
        0.0,
        2.0_f64.ln(),
        6.0_f64.ln(),
        24.0_f64.ln(),
        362880.0_f64.ln(),
    ];
    // lgamma(1) = 0
    assert_relative_eq!(result[0], 0.0, epsilon = 1e-8);
    // lgamma(2) = 0
    assert_relative_eq!(result[1], 0.0, epsilon = 1e-8);
    // lgamma(3) = ln(2)
    assert_relative_eq!(result[2], expected[2], epsilon = 1e-8);
    // lgamma(4) = ln(6)
    assert_relative_eq!(result[3], expected[3], epsilon = 1e-8);
    // lgamma(5) = ln(24)
    assert_relative_eq!(result[4], expected[4], epsilon = 1e-8);
    // lgamma(10) = ln(362880)
    assert_relative_eq!(result[5], expected[5], epsilon = 1e-6);
}

#[test]
fn test_batch_lgamma_f64_vs_scalar() {
    let input: Array1<f64> = Array1::from_vec((1..=30).map(|i| 0.5 + i as f64 * 0.5).collect());
    let batch_result = batch_lgamma_f64(&input.view()).expect("batch_lgamma_f64 failed");

    for i in 0..input.len() {
        let scalar_result = crate::gamma::gammaln(input[i]);
        assert_relative_eq!(
            batch_result[i],
            scalar_result,
            epsilon = 1e-6,
            max_relative = 1e-6
        );
    }
}

#[test]
fn test_batch_lgamma_f32_basic() {
    let input = Array1::from_vec(vec![1.0f32, 2.0, 5.0, 10.0]);
    let result = batch_lgamma_f32(&input.view()).expect("batch_lgamma_f32 failed");

    assert_relative_eq!(result[0], 0.0, epsilon = 1e-4);
    assert_relative_eq!(result[1], 0.0, epsilon = 1e-4);
    assert_relative_eq!(result[2], 24.0f32.ln(), epsilon = 1e-3);
}

// ============================================================================
// Error function tests
// ============================================================================

#[test]
fn test_batch_erf_f64_basic() {
    let input = Array1::from_vec(vec![0.0f64, 0.5, 1.0, 2.0, 3.0, -1.0, -2.0]);
    let result = batch_erf_f64(&input.view()).expect("batch_erf_f64 failed");

    // erf(0) = 0
    assert_relative_eq!(result[0], 0.0, epsilon = 1e-12);
    // erf is odd: erf(-x) = -erf(x)
    assert_relative_eq!(result[5], -result[2], epsilon = 1e-10);
    assert_relative_eq!(result[6], -result[3], epsilon = 1e-10);
    // |erf(x)| <= 1 for all x
    for &val in result.iter() {
        assert!(val.abs() <= 1.0 + 1e-10);
    }
    // erf is monotonically increasing
    assert!(result[1] > result[0]);
    assert!(result[2] > result[1]);
    assert!(result[3] > result[2]);
}

#[test]
fn test_batch_erf_f64_vs_scalar() {
    let input: Array1<f64> = Array1::from_vec((-20..=20).map(|i| i as f64 * 0.25).collect());
    let batch_result = batch_erf_f64(&input.view()).expect("batch_erf_f64 failed");

    for i in 0..input.len() {
        let scalar_result = crate::erf::erf(input[i]);
        assert_relative_eq!(
            batch_result[i],
            scalar_result,
            epsilon = 1e-6,
            max_relative = 1e-6
        );
    }
}

#[test]
fn test_batch_erf_f32_basic() {
    let input = Array1::from_vec(vec![0.0f32, 1.0, -1.0]);
    let result = batch_erf_f32(&input.view()).expect("batch_erf_f32 failed");

    assert_relative_eq!(result[0], 0.0, epsilon = 1e-5);
    assert_relative_eq!(result[1], -result[2], epsilon = 1e-5);
}

// ============================================================================
// Complementary error function tests
// ============================================================================

#[test]
fn test_batch_erfc_f64_basic() {
    let input = Array1::from_vec(vec![0.0f64, 1.0, 2.0, 3.0, -1.0]);
    let result = batch_erfc_f64(&input.view()).expect("batch_erfc_f64 failed");

    // erfc(0) = 1
    assert_relative_eq!(result[0], 1.0, epsilon = 1e-12);
    // erfc(x) + erf(x) = 1 (check consistency)
    let erf_result = batch_erf_f64(&input.view()).expect("batch_erf_f64 failed");
    for i in 0..input.len() {
        assert_relative_eq!(result[i] + erf_result[i], 1.0, epsilon = 1e-10);
    }
    // erfc is monotonically decreasing for positive x
    assert!(result[1] < result[0]);
    assert!(result[2] < result[1]);
}

#[test]
fn test_batch_erfc_f64_vs_scalar() {
    let input: Array1<f64> = Array1::from_vec((-10..=10).map(|i| i as f64 * 0.5).collect());
    let batch_result = batch_erfc_f64(&input.view()).expect("batch_erfc_f64 failed");

    for i in 0..input.len() {
        let scalar_result = crate::erf::erfc(input[i]);
        assert_relative_eq!(
            batch_result[i],
            scalar_result,
            epsilon = 1e-6,
            max_relative = 1e-6
        );
    }
}

#[test]
fn test_batch_erfc_f32_basic() {
    let input = Array1::from_vec(vec![0.0f32, 1.0, 2.0]);
    let result = batch_erfc_f32(&input.view()).expect("batch_erfc_f32 failed");

    assert_relative_eq!(result[0], 1.0, epsilon = 1e-4);
    assert!(result[1] < 1.0 && result[1] > 0.0);
    assert!(result[2] < result[1]);
}

// ============================================================================
// Bessel J0 tests
// ============================================================================

#[test]
fn test_batch_bessel_j0_f64_at_zero() {
    let input = Array1::from_vec(vec![0.0f64]);
    let result = batch_bessel_j0_f64(&input.view()).expect("batch_bessel_j0_f64 failed");
    // J0(0) = 1
    assert_relative_eq!(result[0], 1.0, epsilon = 1e-8);
}

#[test]
fn test_batch_bessel_j0_f64_small_args() {
    let input = Array1::from_vec(vec![0.0f64, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0]);
    let result = batch_bessel_j0_f64(&input.view()).expect("batch_bessel_j0_f64 failed");

    for i in 0..input.len() {
        let scalar_result = crate::bessel::j0(input[i]);
        assert_relative_eq!(
            result[i],
            scalar_result,
            epsilon = 1e-6,
            max_relative = 1e-6
        );
    }
}

#[test]
fn test_batch_bessel_j0_f64_large_args() {
    let input = Array1::from_vec(vec![10.0f64, 20.0, 50.0, 100.0]);
    let result = batch_bessel_j0_f64(&input.view()).expect("batch_bessel_j0_f64 failed");

    for i in 0..input.len() {
        let scalar_result = crate::bessel::j0(input[i]);
        // Asymptotic expansion has slightly lower accuracy
        assert_relative_eq!(
            result[i],
            scalar_result,
            epsilon = 1e-3,
            max_relative = 1e-3
        );
    }
}

#[test]
fn test_batch_bessel_j0_f64_mixed_range() {
    // Mix small and large arguments to test branch handling
    let input = Array1::from_vec(vec![1.0f64, 15.0, 3.0, 50.0, 7.5, 0.1, 100.0]);
    let result = batch_bessel_j0_f64(&input.view()).expect("batch_bessel_j0_f64 failed");

    for i in 0..input.len() {
        let scalar_result = crate::bessel::j0(input[i]);
        assert_relative_eq!(
            result[i],
            scalar_result,
            epsilon = 1e-3,
            max_relative = 1e-3
        );
    }
}

#[test]
fn test_batch_bessel_j0_f32_basic() {
    let input = Array1::from_vec(vec![0.0f32, 1.0, 2.0, 5.0]);
    let result = batch_bessel_j0_f32(&input.view()).expect("batch_bessel_j0_f32 failed");

    assert_relative_eq!(result[0], 1.0, epsilon = 1e-5);
    for i in 1..input.len() {
        let scalar = crate::bessel::j0(input[i] as f64) as f32;
        assert_relative_eq!(result[i], scalar, epsilon = 1e-4);
    }
}

#[test]
fn test_batch_bessel_j0_f64_empty() {
    let input = Array1::<f64>::zeros(0);
    let result = batch_bessel_j0_f64(&input.view()).expect("batch_bessel_j0_f64 failed on empty");
    assert_eq!(result.len(), 0);
}

// ============================================================================
// Bessel J1 tests
// ============================================================================

#[test]
fn test_batch_bessel_j1_f64_at_zero() {
    let input = Array1::from_vec(vec![0.0f64]);
    let result = batch_bessel_j1_f64(&input.view()).expect("batch_bessel_j1_f64 failed");
    // J1(0) = 0
    assert_relative_eq!(result[0], 0.0, epsilon = 1e-10);
}

#[test]
fn test_batch_bessel_j1_f64_small_args() {
    let input = Array1::from_vec(vec![0.0f64, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0]);
    let result = batch_bessel_j1_f64(&input.view()).expect("batch_bessel_j1_f64 failed");

    for i in 0..input.len() {
        let scalar_result = crate::bessel::j1(input[i]);
        assert_relative_eq!(
            result[i],
            scalar_result,
            epsilon = 1e-6,
            max_relative = 1e-6
        );
    }
}

#[test]
fn test_batch_bessel_j1_f64_large_args() {
    let input = Array1::from_vec(vec![10.0f64, 20.0, 50.0, 100.0]);
    let result = batch_bessel_j1_f64(&input.view()).expect("batch_bessel_j1_f64 failed");

    for i in 0..input.len() {
        let scalar_result = crate::bessel::j1(input[i]);
        assert_relative_eq!(
            result[i],
            scalar_result,
            epsilon = 1e-3,
            max_relative = 1e-3
        );
    }
}

#[test]
fn test_batch_bessel_j1_f32_basic() {
    let input = Array1::from_vec(vec![0.0f32, 1.0, 2.0, 5.0]);
    let result = batch_bessel_j1_f32(&input.view()).expect("batch_bessel_j1_f32 failed");

    assert_relative_eq!(result[0], 0.0, epsilon = 1e-5);
    for i in 1..input.len() {
        let scalar = crate::bessel::j1(input[i] as f64) as f32;
        assert_relative_eq!(result[i], scalar, epsilon = 1e-4);
    }
}

// ============================================================================
// Bessel Y0 tests
// ============================================================================

#[test]
fn test_batch_bessel_y0_f64_positive_args() {
    let input = Array1::from_vec(vec![0.5f64, 1.0, 2.0, 5.0, 7.0]);
    let result = batch_bessel_y0_f64(&input.view()).expect("batch_bessel_y0_f64 failed");

    for i in 0..input.len() {
        let scalar_result = crate::bessel::y0(input[i]);
        assert_relative_eq!(
            result[i],
            scalar_result,
            epsilon = 1e-6,
            max_relative = 1e-6
        );
    }
}

#[test]
fn test_batch_bessel_y0_f64_invalid() {
    // Y0(x) is undefined for x <= 0
    let input = Array1::from_vec(vec![-1.0f64, 0.0, 1.0]);
    let result = batch_bessel_y0_f64(&input.view()).expect("batch_bessel_y0_f64 failed");

    assert!(result[0].is_nan());
    assert!(result[1].is_nan());
    assert!(result[2].is_finite());
}

#[test]
fn test_batch_bessel_y0_f64_large_args() {
    let input = Array1::from_vec(vec![10.0f64, 20.0, 50.0]);
    let result = batch_bessel_y0_f64(&input.view()).expect("batch_bessel_y0_f64 failed");

    for i in 0..input.len() {
        let scalar_result = crate::bessel::y0(input[i]);
        assert_relative_eq!(
            result[i],
            scalar_result,
            epsilon = 1e-3,
            max_relative = 1e-3
        );
    }
}

#[test]
fn test_batch_bessel_y0_f32_basic() {
    let input = Array1::from_vec(vec![1.0f32, 2.0, 5.0]);
    let result = batch_bessel_y0_f32(&input.view()).expect("batch_bessel_y0_f32 failed");

    for i in 0..input.len() {
        let scalar = crate::bessel::y0(input[i] as f64) as f32;
        assert_relative_eq!(result[i], scalar, epsilon = 1e-4);
    }
}

// ============================================================================
// Bessel Y1 tests
// ============================================================================

#[test]
fn test_batch_bessel_y1_f64_positive_args() {
    let input = Array1::from_vec(vec![0.5f64, 1.0, 2.0, 5.0, 7.0]);
    let result = batch_bessel_y1_f64(&input.view()).expect("batch_bessel_y1_f64 failed");

    for i in 0..input.len() {
        let scalar_result = crate::bessel::y1(input[i]);
        assert_relative_eq!(
            result[i],
            scalar_result,
            epsilon = 1e-6,
            max_relative = 1e-6
        );
    }
}

#[test]
fn test_batch_bessel_y1_f64_invalid() {
    let input = Array1::from_vec(vec![-1.0f64, 0.0, 1.0]);
    let result = batch_bessel_y1_f64(&input.view()).expect("batch_bessel_y1_f64 failed");

    assert!(result[0].is_nan());
    assert!(result[1].is_nan());
    assert!(result[2].is_finite());
}

#[test]
fn test_batch_bessel_y1_f64_large_args() {
    let input = Array1::from_vec(vec![10.0f64, 20.0, 50.0]);
    let result = batch_bessel_y1_f64(&input.view()).expect("batch_bessel_y1_f64 failed");

    for i in 0..input.len() {
        let scalar_result = crate::bessel::y1(input[i]);
        assert_relative_eq!(
            result[i],
            scalar_result,
            epsilon = 1e-3,
            max_relative = 1e-3
        );
    }
}

#[test]
fn test_batch_bessel_y1_f32_basic() {
    let input = Array1::from_vec(vec![1.0f32, 2.0, 5.0]);
    let result = batch_bessel_y1_f32(&input.view()).expect("batch_bessel_y1_f32 failed");

    for i in 0..input.len() {
        let scalar = crate::bessel::y1(input[i] as f64) as f32;
        assert_relative_eq!(result[i], scalar, epsilon = 1e-4);
    }
}

// ============================================================================
// Beta function tests
// ============================================================================

#[test]
fn test_batch_beta_f64_basic() {
    let a = Array1::from_vec(vec![1.0f64, 2.0, 3.0, 0.5, 1.0]);
    let b = Array1::from_vec(vec![1.0f64, 3.0, 2.0, 0.5, 5.0]);
    let result = batch_beta_f64(&a.view(), &b.view()).expect("batch_beta_f64 failed");

    // B(1,1) = 1
    assert_relative_eq!(result[0], 1.0, epsilon = 1e-8);
    // B(2,3) = 1/12
    assert_relative_eq!(result[1], 1.0 / 12.0, epsilon = 1e-8);
    // B(3,2) = 1/12 (symmetry)
    assert_relative_eq!(result[2], 1.0 / 12.0, epsilon = 1e-8);
    // B(0.5, 0.5) = pi
    assert_relative_eq!(result[3], std::f64::consts::PI, epsilon = 1e-6);
    // B(1, 5) = 1/5
    assert_relative_eq!(result[4], 0.2, epsilon = 1e-8);
}

#[test]
fn test_batch_beta_f64_vs_scalar() {
    let a: Array1<f64> = Array1::from_vec((1..=10).map(|i| 0.5 + i as f64 * 0.3).collect());
    let b: Array1<f64> = Array1::from_vec((1..=10).map(|i| 1.0 + i as f64 * 0.2).collect());
    let batch_result = batch_beta_f64(&a.view(), &b.view()).expect("batch_beta_f64 failed");

    for i in 0..a.len() {
        let scalar_result = crate::gamma::beta(a[i], b[i]);
        assert_relative_eq!(
            batch_result[i],
            scalar_result,
            epsilon = 1e-6,
            max_relative = 1e-6
        );
    }
}

#[test]
fn test_batch_beta_f64_length_mismatch() {
    let a = Array1::from_vec(vec![1.0f64, 2.0]);
    let b = Array1::from_vec(vec![1.0f64, 2.0, 3.0]);
    let result = batch_beta_f64(&a.view(), &b.view());
    assert!(result.is_err());
}

#[test]
fn test_batch_beta_f32_basic() {
    let a = Array1::from_vec(vec![1.0f32, 2.0, 0.5]);
    let b = Array1::from_vec(vec![1.0f32, 3.0, 0.5]);
    let result = batch_beta_f32(&a.view(), &b.view()).expect("batch_beta_f32 failed");

    assert_relative_eq!(result[0], 1.0, epsilon = 1e-4);
    assert_relative_eq!(result[1], 1.0 / 12.0, epsilon = 1e-4);
    assert_relative_eq!(result[2], std::f32::consts::PI, epsilon = 1e-3);
}

// ============================================================================
// Digamma function tests
// ============================================================================

#[test]
fn test_batch_digamma_f64_special_values() {
    let input = Array1::from_vec(vec![1.0f64, 2.0, 3.0]);
    let result = batch_digamma_f64(&input.view()).expect("batch_digamma_f64 failed");

    // psi(1) = -gamma (Euler-Mascheroni constant)
    let euler_mascheroni = 0.5772156649015329;
    assert_relative_eq!(result[0], -euler_mascheroni, epsilon = 1e-6);
    // psi(2) = 1 - gamma
    assert_relative_eq!(result[1], 1.0 - euler_mascheroni, epsilon = 1e-6);
    // psi is increasing for x > 0
    assert!(result[1] > result[0]);
    assert!(result[2] > result[1]);
}

#[test]
fn test_batch_digamma_f64_vs_scalar() {
    let input: Array1<f64> = Array1::from_vec((1..=15).map(|i| 0.5 + i as f64 * 0.5).collect());
    let batch_result = batch_digamma_f64(&input.view()).expect("batch_digamma_f64 failed");

    for i in 0..input.len() {
        let scalar_result = crate::gamma::digamma(input[i]);
        assert_relative_eq!(
            batch_result[i],
            scalar_result,
            epsilon = 1e-4,
            max_relative = 1e-4
        );
    }
}

#[test]
fn test_batch_digamma_f32_basic() {
    let input = Array1::from_vec(vec![1.0f32, 2.0, 5.0, 10.0]);
    let result = batch_digamma_f32(&input.view()).expect("batch_digamma_f32 failed");

    // All values should be finite for positive x
    for &val in result.iter() {
        assert!(val.is_finite());
    }
    // psi is increasing for positive x
    assert!(result[1] > result[0]);
    assert!(result[2] > result[1]);
    assert!(result[3] > result[2]);
}

// ============================================================================
// Large array tests (verify SIMD chunking works correctly)
// ============================================================================

#[test]
fn test_batch_gamma_f64_large_array() {
    let input: Array1<f64> = Array1::from_vec((0..1000).map(|i| 1.0 + i as f64 * 0.01).collect());
    let result = batch_gamma_f64(&input.view()).expect("batch_gamma_f64 failed on large array");

    assert_eq!(result.len(), 1000);
    // Check a few values
    assert_relative_eq!(result[0], 1.0, epsilon = 1e-8); // Gamma(1.0) = 1
                                                         // All positive input => positive output
    for &val in result.iter() {
        assert!(val > 0.0);
    }
}

#[test]
fn test_batch_erf_f64_large_array() {
    let input: Array1<f64> = Array1::from_vec((-500..=500).map(|i| i as f64 * 0.01).collect());
    let result = batch_erf_f64(&input.view()).expect("batch_erf_f64 failed on large array");

    assert_eq!(result.len(), 1001);
    // erf(0) should be at index 500
    assert_relative_eq!(result[500], 0.0, epsilon = 1e-12);
    // Symmetry: erf(-x) = -erf(x)
    for i in 0..500 {
        assert_relative_eq!(result[i], -result[1000 - i], epsilon = 1e-10);
    }
}

#[test]
fn test_batch_bessel_j0_f64_large_array() {
    let input: Array1<f64> = Array1::from_vec((0..500).map(|i| i as f64 * 0.1).collect());
    let result =
        batch_bessel_j0_f64(&input.view()).expect("batch_bessel_j0_f64 failed on large array");

    assert_eq!(result.len(), 500);
    // J0(0) = 1
    assert_relative_eq!(result[0], 1.0, epsilon = 1e-8);
    // All values bounded: |J0(x)| <= 1
    for &val in result.iter() {
        assert!(val.abs() <= 1.0 + 1e-6);
    }
}

// ============================================================================
// Cross-function consistency tests
// ============================================================================

#[test]
fn test_erfc_equals_one_minus_erf() {
    let input: Array1<f64> = Array1::from_vec((-20..=20).map(|i| i as f64 * 0.3).collect());
    let erf_result = batch_erf_f64(&input.view()).expect("batch_erf_f64 failed");
    let erfc_result = batch_erfc_f64(&input.view()).expect("batch_erfc_f64 failed");

    for i in 0..input.len() {
        assert_relative_eq!(erf_result[i] + erfc_result[i], 1.0, epsilon = 1e-10);
    }
}

#[test]
fn test_lgamma_vs_ln_gamma() {
    let input: Array1<f64> = Array1::from_vec((1..=20).map(|i| 0.5 + i as f64 * 0.5).collect());
    let lgamma_result = batch_lgamma_f64(&input.view()).expect("batch_lgamma_f64 failed");
    let gamma_result = batch_gamma_f64(&input.view()).expect("batch_gamma_f64 failed");

    for i in 0..input.len() {
        if gamma_result[i] > 0.0 && gamma_result[i].is_finite() {
            let expected_lgamma = gamma_result[i].ln();
            assert_relative_eq!(
                lgamma_result[i],
                expected_lgamma,
                epsilon = 1e-5,
                max_relative = 1e-5
            );
        }
    }
}
