//! Tutorial: Linear Algebra Basics with SciRS2
//!
//! This tutorial covers the fundamental linear algebra operations
//! available in scirs2-linalg, including matrix creation, basic operations,
//! solving linear systems, and matrix decompositions.
//!
//! Run with: cargo run -p scirs2-linalg --example tutorial_basics

use scirs2_core::ndarray::{array, Array2};
use scirs2_linalg::{
    cholesky, det, inv, lu, matrix_norm, qr, solve, svd, LinalgError, LinalgResult,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SciRS2 Linear Algebra Tutorial ===\n");

    section_creating_matrices()?;
    section_basic_operations()?;
    section_solving_linear_systems()?;
    section_decompositions()?;
    section_norms()?;

    println!("\n=== Tutorial Complete ===");
    Ok(())
}

/// Section 1: Creating matrices using ndarray (re-exported from scirs2-core)
fn section_creating_matrices() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- 1. Creating Matrices ---\n");

    // The `array!` macro creates matrices inline
    let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
    println!("2x2 matrix from macro:\n{}\n", a);

    // Create from a flat vector with shape
    let b = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .map_err(|e| format!("Shape error: {e}"))?;
    println!("2x3 matrix from vec:\n{}\n", b);

    // Identity matrix
    let eye: Array2<f64> = Array2::eye(3);
    println!("3x3 identity:\n{}\n", eye);

    // Zero matrix
    let zeros: Array2<f64> = Array2::zeros((2, 4));
    println!("2x4 zeros:\n{}\n", zeros);

    // Ones matrix
    let ones: Array2<f64> = Array2::ones((3, 2));
    println!("3x2 ones:\n{}\n", ones);

    // Diagonal matrix
    let diag = Array2::from_diag(&array![1.0, 2.0, 3.0]);
    println!("Diagonal matrix:\n{}\n", diag);

    Ok(())
}

/// Section 2: Basic matrix operations (determinant, inverse, trace)
fn section_basic_operations() -> LinalgResult<()> {
    println!("--- 2. Basic Matrix Operations ---\n");

    let a = array![[1.0_f64, 2.0], [3.0, 4.0]];

    // Determinant
    let d = det(&a.view(), None)?;
    println!("Matrix A:\n{}", a);
    println!("det(A) = {:.6}", d);
    // Output: det(A) = -2.000000

    // Inverse
    let a_inv = inv(&a.view(), None)?;
    println!("inv(A):\n{}\n", a_inv);

    // Verify A * A^-1 = I
    let product = a.dot(&a_inv);
    println!("A * inv(A) (should be ~identity):\n{}\n", product);

    // Matrix power: matrix raised to an integer power
    let a_sq = a.dot(&a);
    println!("A^2 = A * A:\n{}\n", a_sq);

    Ok(())
}

/// Section 3: Solving linear systems Ax = b
fn section_solving_linear_systems() -> LinalgResult<()> {
    println!("--- 3. Solving Linear Systems ---\n");

    // Solve Ax = b where:
    //   2x + y = 5
    //   x + 3y = 7
    let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
    let b = array![5.0_f64, 7.0];

    let x = solve(&a.view(), &b.view(), None)?;
    println!("System: 2x + y = 5, x + 3y = 7");
    println!("Solution: x = {:.6}, y = {:.6}", x[0], x[1]);
    // Output: x = 1.600000, y = 1.800000

    // Verify: Ax should equal b
    let ax = a.dot(&x);
    println!("Verification A*x = {:?}", ax.to_vec());
    println!("Original b   = {:?}\n", b.to_vec());

    // Larger system: 3x3
    let a3 = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]];
    let b3 = array![1.0_f64, 2.0, 3.0];
    let x3 = solve(&a3.view(), &b3.view(), None)?;
    println!("3x3 system solution: {:?}", x3.to_vec());

    // Least squares: overdetermined system (more equations than unknowns)
    let a_over = array![[1.0_f64, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]];
    let b_over = array![2.1_f64, 3.9, 6.1, 7.9];
    let result = scirs2_linalg::lstsq(&a_over.view(), &b_over.view(), None)?;
    println!("Least squares solution: {:?}", result.x.to_vec());
    println!("Residual sum of squares: {:.6}\n", result.residuals);

    Ok(())
}

/// Section 4: Matrix decompositions (LU, QR, SVD, Cholesky)
fn section_decompositions() -> LinalgResult<()> {
    println!("--- 4. Matrix Decompositions ---\n");

    // -- LU Decomposition --
    // PA = LU where P is permutation, L is lower triangular, U is upper triangular
    let a = array![[2.0_f64, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]];
    let (p, l, u) = lu(&a.view(), None)?;
    println!("LU Decomposition of:\n{}", a);
    println!("P (permutation):\n{}", p);
    println!("L (lower triangular):\n{}", l);
    println!("U (upper triangular):\n{}\n", u);

    // -- QR Decomposition --
    // A = QR where Q is orthogonal, R is upper triangular
    let a = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let (q, r) = qr(&a.view(), None)?;
    println!("QR Decomposition of:\n{}", a);
    println!("Q (orthogonal):\n{}", q);
    println!("R (upper triangular):\n{}\n", r);

    // -- SVD (Singular Value Decomposition) --
    // A = U * diag(S) * V^T
    let a = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let (u_mat, s, vt) = svd(&a.view(), true, None)?;
    println!("SVD of:\n{}", a);
    println!("Singular values: {:?}", s.to_vec());
    println!("U shape: {:?}", u_mat.dim());
    println!("Vt shape: {:?}\n", vt.dim());

    // -- Cholesky Decomposition (for symmetric positive-definite matrices) --
    // A = L * L^T where L is lower triangular
    let spd = array![[4.0_f64, 2.0], [2.0, 5.0]];
    let l = cholesky(&spd.view(), None)?;
    println!("Cholesky of SPD matrix:\n{}", spd);
    println!("L (lower triangular):\n{}", l);

    // Verify: L * L^T = A
    let reconstructed = l.dot(&l.t());
    println!("L * L^T (should match original):\n{}\n", reconstructed);

    // Demonstrate error handling: non-square matrix for Cholesky
    let non_square = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
    match cholesky(&non_square.view(), None) {
        Err(LinalgError::ShapeError(msg)) => {
            println!("Expected error for non-square Cholesky: {}", msg);
        }
        other => {
            println!("Unexpected result: {:?}", other.is_ok());
        }
    }

    Ok(())
}

/// Section 5: Matrix norms
fn section_norms() -> LinalgResult<()> {
    println!("\n--- 5. Matrix Norms ---\n");

    let a = array![[1.0_f64, 2.0], [3.0, 4.0]];

    // Frobenius norm: sqrt(sum of squared elements)
    let fro = matrix_norm(&a.view(), "fro", None)?;
    println!("Frobenius norm of A: {:.6}", fro);
    // Output: sqrt(1 + 4 + 9 + 16) = sqrt(30) ~ 5.477226

    // 1-norm (max column sum of absolute values)
    let one_norm = matrix_norm(&a.view(), "1", None)?;
    println!("1-norm of A: {:.6}", one_norm);
    // Output: max(|1|+|3|, |2|+|4|) = max(4, 6) = 6.0

    // Infinity norm (max row sum of absolute values)
    let inf_norm = matrix_norm(&a.view(), "inf", None)?;
    println!("Inf-norm of A: {:.6}", inf_norm);
    // Output: max(|1|+|2|, |3|+|4|) = max(3, 7) = 7.0

    println!();
    Ok(())
}
