//! Tests for fractional matrix functions

use super::super::fractional::*;
use approx::assert_relative_eq;
use scirs2_core::ndarray::{array, Array2};

#[test]
fn test_fractional_power_half() {
    let a = array![[4.0_f64, 0.0], [0.0, 9.0]];
    let result = fractionalmatrix_power(&a.view(), 0.5, "eigen").expect("Test: operation failed");

    assert!((result[[0, 0]] - 2.0).abs() < 1e-10);
    assert!((result[[1, 1]] - 3.0).abs() < 1e-10);
}

#[test]
fn test_spd_matrix_function_sqrt() {
    let a = array![[4.0_f64, 0.0], [0.0, 9.0]];
    let result = spdmatrix_function(&a.view(), |x| x.sqrt(), true).expect("Test: operation failed");

    assert!((result[[0, 0]] - 2.0).abs() < 1e-10);
    assert!((result[[1, 1]] - 3.0).abs() < 1e-10);
}

#[test]
fn test_fractional_power_zero() {
    let a = array![[2.0_f64, 1.0], [0.0, 3.0]];
    let result = fractionalmatrix_power(&a.view(), 0.0, "eigen").expect("Test: p=0");
    // A^0 = I
    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(result[[i, j]], expected, epsilon = 1e-12);
        }
    }
}

#[test]
fn test_fractional_power_one() {
    let a = array![[2.0_f64, 1.0], [0.0, 3.0]];
    let result = fractionalmatrix_power(&a.view(), 1.0, "eigen").expect("Test: p=1");
    // A^1 = A
    for i in 0..2 {
        for j in 0..2 {
            assert_relative_eq!(result[[i, j]], a[[i, j]], epsilon = 1e-12);
        }
    }
}

#[test]
fn test_fractional_power_integer_two() {
    // A^2 should equal A*A
    let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
    let result = fractionalmatrix_power(&a.view(), 2.0, "eigen").expect("Test: p=2");
    let a_sq = a.dot(&a);
    for i in 0..2 {
        for j in 0..2 {
            assert_relative_eq!(result[[i, j]], a_sq[[i, j]], epsilon = 1e-8);
        }
    }
}

#[test]
fn test_fractional_power_symmetric_half() {
    // For SPD matrix, A^0.5 * A^0.5 = A
    let a = array![[4.0_f64, 2.0], [2.0, 5.0]]; // SPD
    let sqrt_a = fractionalmatrix_power(&a.view(), 0.5, "eigen").expect("sqrt");
    let reconstructed = sqrt_a.dot(&sqrt_a);
    for i in 0..2 {
        for j in 0..2 {
            assert_relative_eq!(reconstructed[[i, j]], a[[i, j]], epsilon = 1e-8);
        }
    }
}

#[test]
fn test_fractional_power_symmetric_third() {
    // A^(1/3) * A^(1/3) * A^(1/3) should ~ A
    let a = array![[4.0_f64, 1.0], [1.0, 3.0]]; // SPD
    let third = fractionalmatrix_power(&a.view(), 1.0 / 3.0, "eigen").expect("1/3");
    let cube = third.dot(&third).dot(&third);
    for i in 0..2 {
        for j in 0..2 {
            assert_relative_eq!(cube[[i, j]], a[[i, j]], epsilon = 1e-6);
        }
    }
}

#[test]
fn test_fractional_power_negative_integer() {
    // A^(-1) should be the inverse
    let a = array![[2.0_f64, 0.0], [0.0, 4.0]];
    let result = fractionalmatrix_power(&a.view(), -1.0, "pade").expect("Test: p=-1");
    assert_relative_eq!(result[[0, 0]], 0.5, epsilon = 1e-10);
    assert_relative_eq!(result[[1, 1]], 0.25, epsilon = 1e-10);
}

#[test]
fn test_fractional_power_pade_half() {
    let a = array![[4.0_f64, 0.0], [0.0, 9.0]];
    let result = fractionalmatrix_power(&a.view(), 0.5, "pade").expect("Test: pade half");
    assert_relative_eq!(result[[0, 0]], 2.0, epsilon = 1e-6);
    assert_relative_eq!(result[[1, 1]], 3.0, epsilon = 1e-6);
}

#[test]
fn test_fractional_power_invalid_method() {
    let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
    assert!(fractionalmatrix_power(&a.view(), 0.5, "invalid").is_err());
}

#[test]
fn test_spd_matrix_function_exp() {
    // Apply exp to eigenvalues of a diagonal matrix
    let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
    let result = spdmatrix_function(&a.view(), |x| x.exp(), true).expect("exp");
    assert_relative_eq!(result[[0, 0]], 1.0_f64.exp(), epsilon = 1e-10);
    assert_relative_eq!(result[[1, 1]], 2.0_f64.exp(), epsilon = 1e-10);
}

#[test]
fn test_spd_matrix_function_log() {
    let a = array![[2.0_f64, 0.0], [0.0, 5.0]];
    let result = spdmatrix_function(&a.view(), |x| x.ln(), true).expect("log");
    assert_relative_eq!(result[[0, 0]], 2.0_f64.ln(), epsilon = 1e-10);
    assert_relative_eq!(result[[1, 1]], 5.0_f64.ln(), epsilon = 1e-10);
}

#[test]
fn test_spd_matrix_function_nonsymmetric_error() {
    let a = array![[1.0_f64, 2.0], [3.0, 4.0]]; // not symmetric
    assert!(spdmatrix_function(&a.view(), |x| x.sqrt(), true).is_err());
}

#[test]
fn test_matrix_power_schur_upper_triangular() {
    // Upper triangular matrix: Schur form is itself
    let a = array![[2.0_f64, 1.0], [0.0, 3.0]];
    let result = fractionalmatrix_power(&a.view(), 0.5, "schur").expect("schur half");
    // result * result should ~ a
    let sq = result.dot(&result);
    for i in 0..2 {
        for j in 0..2 {
            assert_relative_eq!(sq[[i, j]], a[[i, j]], epsilon = 1e-4);
        }
    }
}

#[test]
fn test_matrix_power_3x3_spd() {
    // 3x3 SPD matrix
    let a = array![[5.0_f64, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 3.0]];
    let sqrt_a = fractionalmatrix_power(&a.view(), 0.5, "eigen").expect("3x3 sqrt");
    let reconstructed = sqrt_a.dot(&sqrt_a);
    for i in 0..3 {
        for j in 0..3 {
            assert_relative_eq!(reconstructed[[i, j]], a[[i, j]], epsilon = 1e-6);
        }
    }
}
