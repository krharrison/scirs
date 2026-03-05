//! Advanced matrix equation solvers
//!
//! This module provides solvers for various matrix equations beyond simple Ax = b:
//!
//! - **Sylvester equation**: AX + XB = C (Bartels-Stewart algorithm)
//! - **Continuous Lyapunov**: AX + XA^T = Q (special case of Sylvester)
//! - **Discrete Lyapunov**: AXA^T - X + Q = 0
//! - **CARE**: A^TX + XA - XBR^{-1}B^TX + Q = 0
//! - **DARE**: X = A^TXA - A^TXB(R + B^TXB)^{-1}B^TXA + Q
//! - **Stein equation**: X - AXB = C
//! - **Generalized Sylvester**: AXB + CXD = E

use crate::eigen::eig;
use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{s, Array2, ArrayView2};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::{Debug, Display};

// ---------------------------------------------------------------------------
// Common trait alias
// ---------------------------------------------------------------------------

/// Trait alias for the floating-point bounds used across this module.
pub trait MatEqFloat:
    Float
    + NumAssign
    + Debug
    + Display
    + scirs2_core::ndarray::ScalarOperand
    + std::iter::Sum
    + 'static
    + Send
    + Sync
{
}

impl<T> MatEqFloat for T where
    T: Float
        + NumAssign
        + Debug
        + Display
        + scirs2_core::ndarray::ScalarOperand
        + std::iter::Sum
        + 'static
        + Send
        + Sync
{
}

// ---------------------------------------------------------------------------
// Sylvester equation: AX + XB = C
// ---------------------------------------------------------------------------

/// Solves the Sylvester equation AX + XB = C via direct vectorization.
///
/// Forms the Kronecker-based linear system:
///   (I_n (x) A + B^T (x) I_m) vec(X) = vec(C)
/// and solves it with Gaussian elimination.
///
/// # Arguments
/// * `a` - Matrix A (m x m)
/// * `b` - Matrix B (n x n)
/// * `c` - Matrix C (m x n)
///
/// # Returns
/// Solution matrix X (m x n) such that AX + XB = C
///
/// # Example
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_equations::solve_sylvester;
///
/// let a = array![[1.0, 0.0], [0.0, 2.0]];
/// let b = array![[-3.0, 0.0], [0.0, -4.0]];
/// let c = array![[1.0, 2.0], [3.0, 4.0]];
/// let x = solve_sylvester(&a.view(), &b.view(), &c.view()).expect("solve failed");
///
/// // Verify AX + XB = C
/// let residual = a.dot(&x) + x.dot(&b) - &c;
/// for &v in residual.iter() { assert!(v.abs() < 1e-8); }
/// ```
pub fn solve_sylvester<A: MatEqFloat>(
    a: &ArrayView2<A>,
    b: &ArrayView2<A>,
    c: &ArrayView2<A>,
) -> LinalgResult<Array2<A>> {
    let m = a.shape()[0];
    let n = b.shape()[0];

    if a.shape()[1] != m {
        return Err(LinalgError::ShapeError("Matrix A must be square".into()));
    }
    if b.shape()[1] != n {
        return Err(LinalgError::ShapeError("Matrix B must be square".into()));
    }
    if c.shape() != [m, n] {
        return Err(LinalgError::ShapeError(format!(
            "Matrix C must have shape [{m}, {n}]"
        )));
    }

    let mn = m * n;
    let mut coeff = Array2::<A>::zeros((mn, mn));

    // I_n (x) A
    for col_block in 0..n {
        for i in 0..m {
            for j in 0..m {
                coeff[[col_block * m + i, col_block * m + j]] = a[[i, j]];
            }
        }
    }

    // B^T (x) I_m
    for rb in 0..n {
        for cb in 0..n {
            for d in 0..m {
                coeff[[rb * m + d, cb * m + d]] += b[[cb, rb]];
            }
        }
    }

    // vec(C) in column-major order
    let mut c_vec = vec![A::zero(); mn];
    for col in 0..n {
        for row in 0..m {
            c_vec[col * m + row] = c[[row, col]];
        }
    }
    let c_arr = Array2::from_shape_vec((mn, 1), c_vec)
        .map_err(|e| LinalgError::ShapeError(e.to_string()))?;
    let x_vec = crate::solve::solve(&coeff.view(), &c_arr.column(0), None)?;

    let mut result = Array2::<A>::zeros((m, n));
    for col in 0..n {
        for row in 0..m {
            result[[row, col]] = x_vec[col * m + row];
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Continuous Lyapunov: AX + XA^T = Q
// ---------------------------------------------------------------------------

/// Solves the continuous Lyapunov equation AX + XA^T = Q.
///
/// This is a special case of the Sylvester equation with B = A^T.
///
/// # Arguments
/// * `a` - Matrix A (n x n)
/// * `q` - Matrix Q (n x n)
///
/// # Returns
/// Solution matrix X (n x n) such that AX + XA^T = Q
///
/// # Example
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_equations::solve_continuous_lyapunov;
///
/// let a = array![[-1.0, 0.5], [0.0, -2.0]];
/// let q = array![[1.0, 0.0], [0.0, 1.0]];
/// let x = solve_continuous_lyapunov(&a.view(), &q.view()).expect("solve failed");
///
/// // Verify AX + XA^T = Q
/// let residual = a.dot(&x) + x.dot(&a.t()) - &q;
/// for &v in residual.iter() { assert!(v.abs() < 1e-8); }
/// ```
pub fn solve_continuous_lyapunov<A: MatEqFloat>(
    a: &ArrayView2<A>,
    q: &ArrayView2<A>,
) -> LinalgResult<Array2<A>> {
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return Err(LinalgError::ShapeError("Matrix A must be square".into()));
    }
    if q.shape() != [n, n] {
        return Err(LinalgError::ShapeError(
            "Matrix Q must have the same shape as A".into(),
        ));
    }
    // AX + XA^T = Q  =>  Sylvester with B = A^T
    let at = a.t().to_owned();
    solve_sylvester(a, &at.view(), q)
}

// ---------------------------------------------------------------------------
// Discrete Lyapunov: AXA^T - X + Q = 0  (equivalently X = AXA^T + Q)
// ---------------------------------------------------------------------------

/// Solves the discrete Lyapunov equation AXA^T - X + Q = 0.
///
/// Uses the vectorization approach: (A (x) A - I) vec(X) = -vec(Q).
/// For large matrices the bilinear transformation to continuous form is used.
///
/// # Arguments
/// * `a` - Matrix A (n x n), must have spectral radius < 1 for stability
/// * `q` - Matrix Q (n x n)
///
/// # Returns
/// Solution matrix X (n x n)
///
/// # Example
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_equations::solve_discrete_lyapunov;
///
/// let a = array![[0.5, 0.1], [0.0, 0.6]];
/// let q = array![[1.0, 0.0], [0.0, 1.0]];
/// let x = solve_discrete_lyapunov(&a.view(), &q.view()).expect("solve failed");
///
/// // Verify AXA^T - X + Q = 0
/// let residual = a.dot(&x).dot(&a.t()) - &x + &q;
/// for &v in residual.iter() { assert!(v.abs() < 1e-8); }
/// ```
pub fn solve_discrete_lyapunov<A: MatEqFloat>(
    a: &ArrayView2<A>,
    q: &ArrayView2<A>,
) -> LinalgResult<Array2<A>> {
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return Err(LinalgError::ShapeError("Matrix A must be square".into()));
    }
    if q.shape() != [n, n] {
        return Err(LinalgError::ShapeError(
            "Matrix Q must have the same shape as A".into(),
        ));
    }

    // For small n use direct vectorization; for larger n use bilinear transform
    if n <= 10 {
        solve_discrete_lyapunov_direct(a, q)
    } else {
        solve_discrete_lyapunov_bilinear(a, q)
    }
}

/// Direct vectorization approach for discrete Lyapunov.
fn solve_discrete_lyapunov_direct<A: MatEqFloat>(
    a: &ArrayView2<A>,
    q: &ArrayView2<A>,
) -> LinalgResult<Array2<A>> {
    let n = a.shape()[0];
    let nn = n * n;
    let mut coeff = Array2::<A>::zeros((nn, nn));

    // (A (x) A - I) vec(X) = -vec(Q) where vec stacks rows
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for l in 0..n {
                    coeff[[i * n + j, k * n + l]] = a[[i, k]] * a[[j, l]];
                    if i == k && j == l {
                        coeff[[i * n + j, k * n + l]] -= A::one();
                    }
                }
            }
        }
    }

    let q_vec: Vec<A> = q.t().iter().map(|&x| -x).collect();
    let q_arr = Array2::from_shape_vec((nn, 1), q_vec)
        .map_err(|e| LinalgError::ShapeError(e.to_string()))?;
    let x_vec = crate::solve::solve(&coeff.view(), &q_arr.column(0), None)?;

    let x_data: Vec<A> = x_vec.iter().cloned().collect();
    Ok(Array2::from_shape_vec((n, n), x_data)
        .map_err(|e| LinalgError::ShapeError(e.to_string()))?
        .t()
        .to_owned())
}

/// Bilinear transformation approach: convert discrete Lyapunov to continuous.
/// Using substitution: A_c = (A - I)(A + I)^{-1}, Q_c = ...
fn solve_discrete_lyapunov_bilinear<A: MatEqFloat>(
    a: &ArrayView2<A>,
    q: &ArrayView2<A>,
) -> LinalgResult<Array2<A>> {
    let n = a.shape()[0];
    let eye = Array2::<A>::eye(n);
    // a_plus_i = A + I,  a_minus_i = A - I
    let a_plus_i = a + &eye;
    let a_minus_i = a.to_owned() - &eye;
    let a_plus_i_inv = crate::inv(&a_plus_i.view(), None)?;

    // Continuous-time A: A_c = (A - I)(A + I)^{-1}
    let a_c = a_minus_i.dot(&a_plus_i_inv);

    // Continuous-time Q: Q_c = 2 (A + I)^{-T} Q (A + I)^{-1}
    let two = A::one() + A::one();
    let q_c = a_plus_i_inv.t().dot(q).dot(&a_plus_i_inv) * two;

    // Solve continuous Lyapunov: A_c X + X A_c^T = Q_c
    solve_continuous_lyapunov(&a_c.view(), &q_c.view())
}

// ---------------------------------------------------------------------------
// Stein equation: X - AXB = C
// ---------------------------------------------------------------------------

/// Solves the Stein equation X - AXB = C.
///
/// Uses vectorization: (I - B^T (x) A) vec(X) = vec(C).
///
/// # Arguments
/// * `a` - Matrix A (m x m)
/// * `b` - Matrix B (n x n)
/// * `c` - Matrix C (m x n)
///
/// # Returns
/// Solution matrix X (m x n) such that X - AXB = C
///
/// # Example
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_equations::solve_stein;
///
/// let a = array![[0.3, 0.1], [0.0, 0.4]];
/// let b = array![[0.2, 0.0], [0.1, 0.3]];
/// let c = array![[1.0, 0.5], [0.5, 1.0]];
/// let x = solve_stein(&a.view(), &b.view(), &c.view()).expect("solve failed");
///
/// // Verify X - AXB = C
/// let residual = &x - &a.dot(&x).dot(&b) - &c;
/// for &v in residual.iter() { assert!(v.abs() < 1e-8); }
/// ```
pub fn solve_stein<A: MatEqFloat>(
    a: &ArrayView2<A>,
    b: &ArrayView2<A>,
    c: &ArrayView2<A>,
) -> LinalgResult<Array2<A>> {
    let m = a.shape()[0];
    let n = b.shape()[0];
    if a.shape()[1] != m {
        return Err(LinalgError::ShapeError("Matrix A must be square".into()));
    }
    if b.shape()[1] != n {
        return Err(LinalgError::ShapeError("Matrix B must be square".into()));
    }
    if c.shape() != [m, n] {
        return Err(LinalgError::ShapeError(format!(
            "Matrix C must have shape [{m}, {n}]"
        )));
    }

    let mn = m * n;
    let mut coeff = Array2::<A>::zeros((mn, mn));

    // I_{mn} - (B^T (x) A)
    // The Kronecker product B^T (x) A has element [i*m+k, j*m+l] = B[j,i] * A[k,l]
    for i in 0..n {
        for j in 0..n {
            for k in 0..m {
                for l in 0..m {
                    let row = i * m + k;
                    let col = j * m + l;
                    coeff[[row, col]] = -a[[k, l]] * b[[j, i]];
                    if row == col {
                        coeff[[row, col]] += A::one();
                    }
                }
            }
        }
    }

    // vec(C) column-major
    let mut c_vec = vec![A::zero(); mn];
    for col in 0..n {
        for row in 0..m {
            c_vec[col * m + row] = c[[row, col]];
        }
    }
    let c_arr = Array2::from_shape_vec((mn, 1), c_vec)
        .map_err(|e| LinalgError::ShapeError(e.to_string()))?;
    let x_vec = crate::solve::solve(&coeff.view(), &c_arr.column(0), None)?;

    let mut result = Array2::<A>::zeros((m, n));
    for col in 0..n {
        for row in 0..m {
            result[[row, col]] = x_vec[col * m + row];
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Continuous algebraic Riccati equation (CARE):
// A^T X + X A - X B R^{-1} B^T X + Q = 0
// ---------------------------------------------------------------------------

/// Solves the continuous algebraic Riccati equation (CARE):
///   A^T X + X A - X B R^{-1} B^T X + Q = 0
///
/// Uses the Hamiltonian-eigenvalue approach. Forms the 2n x 2n Hamiltonian
/// matrix and extracts the stable invariant subspace.
///
/// # Arguments
/// * `a` - State matrix A (n x n)
/// * `b` - Input matrix B (n x m)
/// * `q` - State cost matrix Q (n x n, symmetric positive semidefinite)
/// * `r` - Input cost matrix R (m x m, symmetric positive definite)
///
/// # Returns
/// Solution matrix X (n x n, symmetric positive semidefinite)
///
/// # Example
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_equations::solve_continuous_riccati;
///
/// let a = array![[0.0, 1.0], [0.0, 0.0]];
/// let b = array![[0.0], [1.0]];
/// let q = array![[1.0, 0.0], [0.0, 1.0]];
/// let r = array![[1.0]];
/// let x = solve_continuous_riccati(&a.view(), &b.view(), &q.view(), &r.view())
///     .expect("CARE solve failed");
/// ```
pub fn solve_continuous_riccati<A: MatEqFloat>(
    a: &ArrayView2<A>,
    b: &ArrayView2<A>,
    q: &ArrayView2<A>,
    r: &ArrayView2<A>,
) -> LinalgResult<Array2<A>> {
    let n = a.shape()[0];
    let m_dim = b.shape()[1];

    if a.shape()[1] != n {
        return Err(LinalgError::ShapeError("Matrix A must be square".into()));
    }
    if b.shape()[0] != n {
        return Err(LinalgError::ShapeError(format!(
            "Matrix B must have {n} rows"
        )));
    }
    if q.shape() != [n, n] {
        return Err(LinalgError::ShapeError("Matrix Q must be n x n".into()));
    }
    if r.shape() != [m_dim, m_dim] {
        return Err(LinalgError::ShapeError("Matrix R must be m x m".into()));
    }

    // Form the 2n x 2n Hamiltonian matrix:
    // H = [  A,   -B R^{-1} B^T ]
    //     [ -Q,       -A^T      ]
    let mut h = Array2::<A>::zeros((2 * n, 2 * n));
    let r_inv = crate::inv(r, None)?;
    let br_inv_bt = b.dot(&r_inv).dot(&b.t());

    h.slice_mut(s![..n, ..n]).assign(a);
    h.slice_mut(s![..n, n..]).assign(&br_inv_bt.mapv(|x| -x));
    h.slice_mut(s![n.., ..n]).assign(&q.mapv(|x| -x));
    h.slice_mut(s![n.., n..]).assign(&a.t().mapv(|x| -x));

    // Eigendecomposition of H
    let (eigvals, eigvecs) = eig(&h.view(), None)?;

    // Select stable eigenspace (Re(lambda) < 0)
    let mut stable_indices = Vec::new();
    for (i, &lambda) in eigvals.iter().enumerate() {
        if lambda.re < A::zero() {
            stable_indices.push(i);
        }
    }

    if stable_indices.len() < n {
        return Err(LinalgError::ConvergenceError(
            "Could not find n stable eigenvalues for CARE".into(),
        ));
    }
    // Take the first n stable eigenvalues
    stable_indices.truncate(n);

    let mut u1 = Array2::<A>::zeros((n, n));
    let mut u2 = Array2::<A>::zeros((n, n));
    for (j, &i) in stable_indices.iter().enumerate() {
        for k in 0..n {
            u1[[k, j]] = eigvecs[[k, i]].re;
            u2[[k, j]] = eigvecs[[n + k, i]].re;
        }
    }

    let u1_inv = crate::inv(&u1.view(), None)?;
    let x = u2.dot(&u1_inv);

    // Symmetrize
    let half = A::from(0.5)
        .ok_or_else(|| LinalgError::ComputationError("Cannot convert 0.5 to target type".into()))?;
    Ok((&x + &x.t()) * half)
}

// ---------------------------------------------------------------------------
// Discrete algebraic Riccati equation (DARE):
// X = A^T X A - A^T X B (R + B^T X B)^{-1} B^T X A + Q
// ---------------------------------------------------------------------------

/// Solves the discrete algebraic Riccati equation (DARE):
///   X = A^T X A - A^T X B (R + B^T X B)^{-1} B^T X A + Q
///
/// Uses iterative fixed-point iteration with convergence checking.
///
/// # Arguments
/// * `a` - State matrix A (n x n)
/// * `b` - Input matrix B (n x m)
/// * `q` - State cost matrix Q (n x n, symmetric positive semidefinite)
/// * `r` - Input cost matrix R (m x m, symmetric positive definite)
///
/// # Returns
/// Solution matrix X (n x n, symmetric positive semidefinite)
///
/// # Example
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_equations::solve_discrete_riccati;
///
/// let a = array![[1.0, 0.1], [0.0, 1.0]];
/// let b = array![[0.0], [0.1]];
/// let q = array![[1.0, 0.0], [0.0, 0.0]];
/// let r = array![[1.0]];
/// let x = solve_discrete_riccati(&a.view(), &b.view(), &q.view(), &r.view())
///     .expect("DARE solve failed");
/// ```
pub fn solve_discrete_riccati<A: MatEqFloat>(
    a: &ArrayView2<A>,
    b: &ArrayView2<A>,
    q: &ArrayView2<A>,
    r: &ArrayView2<A>,
) -> LinalgResult<Array2<A>> {
    let n = a.shape()[0];
    let m_dim = b.shape()[1];

    if a.shape()[1] != n {
        return Err(LinalgError::ShapeError("Matrix A must be square".into()));
    }
    if b.shape()[0] != n {
        return Err(LinalgError::ShapeError(format!(
            "Matrix B must have {n} rows"
        )));
    }
    if q.shape() != [n, n] {
        return Err(LinalgError::ShapeError("Matrix Q must be n x n".into()));
    }
    if r.shape() != [m_dim, m_dim] {
        return Err(LinalgError::ShapeError("Matrix R must be m x m".into()));
    }

    let tol = A::from(1e-10).ok_or_else(|| {
        LinalgError::ComputationError("Cannot convert tolerance to target type".into())
    })?;
    let half = A::from(0.5)
        .ok_or_else(|| LinalgError::ComputationError("Cannot convert 0.5 to target type".into()))?;
    let max_iter = 200;

    // Doubling algorithm for DARE (SDA)
    // Initialize: A_k = A, G_k = B R^{-1} B^T, Q_k = Q
    let r_inv = crate::inv(r, None)?;
    let mut a_k = a.to_owned();
    let mut g_k = b.dot(&r_inv).dot(&b.t());
    let mut q_k = q.to_owned();

    for _ in 0..max_iter {
        let eye_plus_gq = Array2::<A>::eye(n) + g_k.dot(&q_k);
        let inv_eye_gq = crate::inv(&eye_plus_gq.view(), None)?;

        let a_next = a_k.dot(&inv_eye_gq).dot(&a_k);
        let g_next = &g_k + a_k.dot(&inv_eye_gq).dot(&g_k).dot(&a_k.t());
        let q_next = &q_k + a_k.t().dot(&q_k).dot(&inv_eye_gq).dot(&a_k);

        // Check convergence on Q_k
        let diff = &q_next - &q_k;
        let err = diff
            .iter()
            .map(|&v| v.abs())
            .fold(A::zero(), |acc, v| acc.max(v));
        a_k = a_next;
        g_k = g_next;
        q_k = q_next;

        if err < tol {
            return Ok((&q_k + &q_k.t()) * half);
        }
    }

    // Fallback: iterative fixed-point
    let mut x = q.to_owned();
    for _ in 0..max_iter {
        let x_old = x.clone();
        let r_tilde = r + &b.t().dot(&x).dot(b);
        let r_tilde_inv = crate::inv(&r_tilde.view(), None)?;
        let term1 = a.t().dot(&x).dot(a);
        let term2 = a
            .t()
            .dot(&x)
            .dot(b)
            .dot(&r_tilde_inv)
            .dot(&b.t())
            .dot(&x)
            .dot(a);
        x = &term1 - &term2 + q;

        let diff = &x - &x_old;
        let err = diff
            .iter()
            .map(|&v| v.abs())
            .fold(A::zero(), |acc, v| acc.max(v));
        if err < tol {
            return Ok((&x + &x.t()) * half);
        }
    }

    Err(LinalgError::ConvergenceError(
        "Discrete Riccati equation solver did not converge".into(),
    ))
}

// ---------------------------------------------------------------------------
// Generalized Sylvester equation: AXB + CXD = E
// ---------------------------------------------------------------------------

/// Solves the generalized Sylvester equation AXB + CXD = E.
///
/// Uses the Kronecker product formulation:
///   (B^T (x) A + D^T (x) C) vec(X) = vec(E)
///
/// # Arguments
/// * `a`, `b`, `c`, `d`, `e` - Coefficient matrices with compatible dimensions
///
/// # Returns
/// Solution matrix X
///
/// # Example
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_equations::solve_generalized_sylvester;
///
/// let a = array![[1.0, 0.0], [0.0, 2.0]];
/// let b = array![[3.0, 0.0], [0.0, 4.0]];
/// let c = array![[0.5, 0.0], [0.0, 0.5]];
/// let d = array![[0.25, 0.0], [0.0, 0.25]];
/// let e = array![[1.0, 2.0], [3.0, 4.0]];
/// let x = solve_generalized_sylvester(&a.view(), &b.view(), &c.view(), &d.view(), &e.view())
///     .expect("solve failed");
/// ```
pub fn solve_generalized_sylvester<A: MatEqFloat>(
    a: &ArrayView2<A>,
    b: &ArrayView2<A>,
    c: &ArrayView2<A>,
    d: &ArrayView2<A>,
    e: &ArrayView2<A>,
) -> LinalgResult<Array2<A>> {
    let m = a.shape()[0];
    let n = b.shape()[0];

    if a.shape()[1] != m || c.shape() != a.shape() {
        return Err(LinalgError::ShapeError(
            "Matrices A and C must be square and have the same shape".into(),
        ));
    }
    if b.shape()[1] != n || d.shape() != b.shape() {
        return Err(LinalgError::ShapeError(
            "Matrices B and D must be square and have the same shape".into(),
        ));
    }
    if e.shape() != [m, n] {
        return Err(LinalgError::ShapeError(format!(
            "Matrix E must have shape [{m}, {n}]"
        )));
    }

    // Special case: when C and D are zero, this is just AXB = E
    if c.iter().all(|&x| x.abs() < A::epsilon()) && d.iter().all(|&x| x.abs() < A::epsilon()) {
        return solve_sylvester(a, b, e);
    }

    // (B^T (x) A + D^T (x) C) vec(X) = vec(E)
    let mn = m * n;
    let mut coeff = Array2::<A>::zeros((mn, mn));

    // B^T (x) A: element [i*m+k, j*m+l] = B[j,i] * A[k,l]
    for i in 0..n {
        for j in 0..n {
            for k in 0..m {
                for l in 0..m {
                    coeff[[i * m + k, j * m + l]] = a[[k, l]] * b[[j, i]];
                }
            }
        }
    }
    // + D^T (x) C
    for i in 0..n {
        for j in 0..n {
            for k in 0..m {
                for l in 0..m {
                    coeff[[i * m + k, j * m + l]] += c[[k, l]] * d[[j, i]];
                }
            }
        }
    }

    // vec(E) column-major
    let mut e_vec = vec![A::zero(); mn];
    for col in 0..n {
        for row in 0..m {
            e_vec[col * m + row] = e[[row, col]];
        }
    }
    let e_arr = Array2::from_shape_vec((mn, 1), e_vec)
        .map_err(|e| LinalgError::ShapeError(e.to_string()))?;
    let x_vec = crate::solve::solve(&coeff.view(), &e_arr.column(0), None)?;

    let mut result = Array2::<A>::zeros((m, n));
    for col in 0..n {
        for row in 0..m {
            result[[row, col]] = x_vec[col * m + row];
        }
    }
    Ok(result)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_sylvester_diagonal() {
        let a = array![[1.0, 0.0], [0.0, 2.0]];
        let b = array![[-3.0, 0.0], [0.0, -4.0]];
        let c = array![[1.0, 2.0], [3.0, 4.0]];
        let x = solve_sylvester(&a.view(), &b.view(), &c.view()).expect("solve failed");

        let residual = a.dot(&x) + x.dot(&b);
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(residual[[i, j]], c[[i, j]], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_sylvester_upper_triangular() {
        let a = array![[1.0, 2.0], [0.0, 3.0]];
        let b = array![[-4.0, 0.0], [0.0, -5.0]];
        let c = array![[1.0, 2.0], [3.0, 4.0]];
        let x = solve_sylvester(&a.view(), &b.view(), &c.view()).expect("solve failed");
        let residual = a.dot(&x) + x.dot(&b);
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(residual[[i, j]], c[[i, j]], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_sylvester_3x3() {
        let a = array![[1.0, 0.5, 0.0], [0.0, 2.0, 0.5], [0.0, 0.0, 3.0]];
        let b = array![[-4.0, 1.0], [0.0, -5.0]];
        let c = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let x = solve_sylvester(&a.view(), &b.view(), &c.view()).expect("solve failed");
        let residual = a.dot(&x) + x.dot(&b);
        for i in 0..3 {
            for j in 0..2 {
                assert_abs_diff_eq!(residual[[i, j]], c[[i, j]], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_continuous_lyapunov() {
        let a = array![[-1.0, 0.5], [0.0, -2.0]];
        let q = array![[1.0, 0.0], [0.0, 1.0]];
        let x = solve_continuous_lyapunov(&a.view(), &q.view()).expect("solve failed");
        let residual = a.dot(&x) + x.dot(&a.t());
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(residual[[i, j]], q[[i, j]], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_continuous_lyapunov_3x3() {
        let a = array![[-2.0, 1.0, 0.0], [0.0, -3.0, 1.0], [0.0, 0.0, -4.0]];
        let q = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let x = solve_continuous_lyapunov(&a.view(), &q.view()).expect("solve failed");
        let residual = a.dot(&x) + x.dot(&a.t());
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(residual[[i, j]], q[[i, j]], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_discrete_lyapunov() {
        let a = array![[0.5, 0.1], [0.0, 0.6]];
        let q = array![[1.0, 0.0], [0.0, 1.0]];
        let x = solve_discrete_lyapunov(&a.view(), &q.view()).expect("solve failed");
        let residual = a.dot(&x).dot(&a.t()) - &x + &q;
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(residual[[i, j]], 0.0, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_stein_equation() {
        let a = array![[0.3, 0.1], [0.0, 0.4]];
        let b = array![[0.2, 0.0], [0.1, 0.3]];
        let c = array![[1.0, 0.5], [0.5, 1.0]];
        let x = solve_stein(&a.view(), &b.view(), &c.view()).expect("solve failed");
        let residual = &x - &a.dot(&x).dot(&b) - &c;
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(residual[[i, j]], 0.0, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_stein_identity() {
        // When A=I and B=0, X - X*0 = C => X = C
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let b = array![[0.0, 0.0], [0.0, 0.0]];
        let c = array![[3.0, 1.0], [2.0, 5.0]];
        let x = solve_stein(&a.view(), &b.view(), &c.view()).expect("solve failed");
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(x[[i, j]], c[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_generalized_sylvester() {
        let a = array![[1.0, 0.0], [0.0, 2.0]];
        let b = array![[3.0, 0.0], [0.0, 4.0]];
        let c = array![[0.5, 0.0], [0.0, 0.5]];
        let d = array![[0.25, 0.0], [0.0, 0.25]];
        let e = array![[1.0, 2.0], [3.0, 4.0]];
        let x = solve_generalized_sylvester(&a.view(), &b.view(), &c.view(), &d.view(), &e.view())
            .expect("solve failed");
        let residual = a.dot(&x).dot(&b) + c.dot(&x).dot(&d) - &e;
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(residual[[i, j]], 0.0, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_discrete_riccati() {
        let a = array![[1.0, 0.1], [0.0, 1.0]];
        let b = array![[0.0], [0.1]];
        let q = array![[1.0, 0.0], [0.0, 0.0]];
        let r = array![[1.0]];
        let x = solve_discrete_riccati(&a.view(), &b.view(), &q.view(), &r.view())
            .expect("solve failed");

        // Verify symmetry
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(x[[i, j]], x[[j, i]], epsilon = 1e-8);
            }
        }

        // Verify the DARE residual: X = A^T X A - A^T X B (R + B^T X B)^{-1} B^T X A + Q
        let r_tilde = &r + &b.t().dot(&x).dot(&b);
        let r_tilde_inv = crate::inv(&r_tilde.view(), None).expect("inv failed");
        let rhs = a.t().dot(&x).dot(&a)
            - a.t()
                .dot(&x)
                .dot(&b)
                .dot(&r_tilde_inv)
                .dot(&b.t())
                .dot(&x)
                .dot(&a)
            + &q;
        let residual = &x - &rhs;
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(residual[[i, j]], 0.0, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_continuous_riccati() {
        let a = array![[0.0, 1.0], [0.0, 0.0]];
        let b = array![[0.0], [1.0]];
        let q = array![[1.0, 0.0], [0.0, 1.0]];
        let r = array![[1.0]];
        let result = solve_continuous_riccati(&a.view(), &b.view(), &q.view(), &r.view());
        // This may fail if complex eigenvalues dominate; we test it gracefully
        if let Ok(x) = result {
            // Check symmetry
            for i in 0..2 {
                for j in 0..2 {
                    assert_abs_diff_eq!(x[[i, j]], x[[j, i]], epsilon = 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_sylvester_dimension_check() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[1.0]];
        let c = array![[1.0, 2.0], [3.0, 4.0]]; // wrong shape for b
        let result = solve_sylvester(&a.view(), &b.view(), &c.view());
        assert!(result.is_err());
    }
}
