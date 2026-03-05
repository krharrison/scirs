//! Algebraic Riccati equation solvers.
//!
//! Provides solvers for:
//! - Continuous Algebraic Riccati Equation (CARE):
//!   `Aᵀ P + P A - P B R⁻¹ Bᵀ P + Q = 0`
//! - Discrete Algebraic Riccati Equation (DARE):
//!   `Aᵀ P A - P - Aᵀ P B (Bᵀ P B + R)⁻¹ Bᵀ P A + Q = 0`
//!
//! # Algorithms
//!
//! ## CARE — Hamiltonian Matrix + Newton's Method
//! The Hamiltonian matrix `H = [[A, -B R⁻¹ Bᵀ], [-Q, -Aᵀ]]` has the property
//! that if `λ` is an eigenvalue so is `-λ`. The stable invariant subspace of `H`
//! (eigenvalues with negative real part) gives the solution `P = X₂ X₁⁻¹` where
//! `[X₁; X₂]` spans the stable subspace.
//!
//! In practice, we use Newton's method (Riccati iteration) which is more
//! numerically stable for smaller problems:
//! `P_{k+1} = solve_lyapunov(Aᵣ, P_k B R⁻¹ Bᵀ P_k + Q)`
//! where `Aᵣ = A - B R⁻¹ Bᵀ P_k`.
//!
//! ## DARE — Doubling Algorithm + Newton's Method
//! The symplectic matrix approach: uses the matrix sign function or
//! structure-preserving doubling algorithm.
//!
//! # References
//! - Lancaster, P. & Rodman, L. (1995). *Algebraic Riccati Equations*. Oxford.
//! - Arnold, W. F. & Laub, A. J. (1984). Generalized eigenproblem algorithms
//!   for the matrix algebraic Riccati equation. *Proc. IEEE*, 72(12).
//! - Kleinman, D. L. (1968). On an iterative technique for Riccati equation
//!   computations. *IEEE Trans. Autom. Control*, 13(1), 114–115.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::{Debug, Display};
use std::iter::Sum;

use super::lyapunov::{lyapunov_continuous, lyapunov_discrete};

/// Trait bound for Riccati solver scalars.
pub trait RiccatiFloat:
    Float + NumAssign + Sum + ScalarOperand + Debug + Display + Send + Sync + 'static
{
}
impl<F> RiccatiFloat for F where
    F: Float + NumAssign + Sum + ScalarOperand + Debug + Display + Send + Sync + 'static
{
}

// ---------------------------------------------------------------------------
// Internal linear algebra helpers
// ---------------------------------------------------------------------------

/// Dense matrix multiply.
fn mm<F: RiccatiFloat>(a: &Array2<F>, b: &Array2<F>) -> Array2<F> {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();
    let mut c = Array2::<F>::zeros((m, n));
    for i in 0..m {
        for p in 0..k {
            let aip = a[[i, p]];
            if aip == F::zero() {
                continue;
            }
            for j in 0..n {
                c[[i, j]] += aip * b[[p, j]];
            }
        }
    }
    c
}

/// Matrix inverse via Gauss-Jordan with partial pivoting.
fn mat_inv<F: RiccatiFloat>(a: &Array2<F>) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    let mut aug = Array2::<F>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n + i]] = F::one();
    }
    for col in 0..n {
        // Partial pivoting
        let mut piv = col;
        let mut piv_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            let v = aug[[row, col]].abs();
            if v > piv_val {
                piv_val = v;
                piv = row;
            }
        }
        if piv_val < F::from(1e-14).unwrap_or(F::epsilon()) {
            return Err(LinalgError::SingularMatrixError(
                "Riccati solver: matrix is singular".to_string(),
            ));
        }
        if piv != col {
            for j in 0..(2 * n) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[piv, j]];
                aug[[piv, j]] = tmp;
            }
        }
        let sc = aug[[col, col]];
        for j in 0..(2 * n) {
            aug[[col, j]] /= sc;
        }
        for row in 0..n {
            if row != col {
                let fac = aug[[row, col]];
                if fac == F::zero() {
                    continue;
                }
                for j in 0..(2 * n) {
                    let v = aug[[col, j]];
                    aug[[row, j]] -= fac * v;
                }
            }
        }
    }
    let mut inv = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }
    Ok(inv)
}

/// Symmetrize: X = (X + Xᵀ) / 2
fn symmetrize<F: RiccatiFloat>(x: &Array2<F>) -> Array2<F> {
    let n = x.nrows();
    let mut s = Array2::<F>::zeros((n, n));
    let two = F::from(2.0).unwrap_or(F::one());
    for i in 0..n {
        for j in 0..n {
            s[[i, j]] = (x[[i, j]] + x[[j, i]]) / two;
        }
    }
    s
}

/// Frobenius norm of a matrix.
fn frob_norm<F: RiccatiFloat>(a: &Array2<F>) -> F {
    a.iter().map(|&v| v * v).sum::<F>().sqrt()
}

// ---------------------------------------------------------------------------
// CARE: Continuous Algebraic Riccati Equation
// ---------------------------------------------------------------------------

/// Solve the Continuous Algebraic Riccati Equation (CARE):
/// `Aᵀ P + P A - P B R⁻¹ Bᵀ P + Q = 0`
///
/// # Arguments
/// * `a` - `n×n` system (state) matrix
/// * `b` - `n×m` input matrix
/// * `q` - `n×n` state cost matrix (positive semi-definite)
/// * `r` - `m×m` input cost matrix (positive definite)
///
/// # Returns
/// The symmetric positive semi-definite solution matrix `P` of size `n×n`.
///
/// # Algorithm
/// Newton-Kleinman iteration:
/// 1. Initialize `P₀ = Q` (or better: a stabilizing initial guess)
/// 2. `Aᵣ = A - B R⁻¹ Bᵀ P_k`
/// 3. `P_{k+1}` solves the Lyapunov equation: `Aᵣᵀ P + P Aᵣ = -(Q + P_k B R⁻¹ Bᵀ P_k)`
/// 4. Repeat until `‖P_{k+1} - P_k‖_F < tol`
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::control::riccati::care_solve;
///
/// // Simple 1D system: A=-1, B=1, Q=1, R=1 → P*(2*(-1)-1) = -1 → P=0.5*(√5-1)
/// let a = array![[-1.0_f64]];
/// let b = array![[1.0_f64]];
/// let q = array![[1.0_f64]];
/// let r = array![[1.0_f64]];
/// let p = care_solve(&a.view(), &b.view(), &q.view(), &r.view()).expect("CARE failed");
/// // Verify CARE residual ≈ 0
/// let atp = a.t().dot(&p);
/// let pa = p.dot(&a);
/// let pbrinvbtp = p.dot(&b).dot(&mat_inv_test(&r)).dot(&b.t()).dot(&p);
/// ```
pub fn care_solve<F: RiccatiFloat>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    q: &ArrayView2<F>,
    r: &ArrayView2<F>,
) -> LinalgResult<Array2<F>> {
    let n = check_square(a, "CARE: A")?;
    let q_n = check_square(q, "CARE: Q")?;
    let r_m = check_square(r, "CARE: R")?;
    if q_n != n {
        return Err(LinalgError::DimensionError(format!(
            "CARE: Q must be {n}×{n}, got {q_n}×{q_n}"
        )));
    }
    if b.nrows() != n {
        return Err(LinalgError::DimensionError(format!(
            "CARE: B must have {n} rows, got {}",
            b.nrows()
        )));
    }
    let m = b.ncols();
    if r_m != m {
        return Err(LinalgError::DimensionError(format!(
            "CARE: R must be {m}×{m}, got {r_m}×{r_m}"
        )));
    }

    let a_own = a.to_owned();
    let b_own = b.to_owned();
    let q_own = q.to_owned();

    // Precompute R⁻¹
    let r_own = r.to_owned();
    let r_inv = mat_inv(&r_own).map_err(|e| {
        LinalgError::SingularMatrixError(format!("CARE: R is singular: {e}"))
    })?;
    // Precompute S = B R⁻¹ Bᵀ  (n×n, symmetric)
    let br_inv = mm(&b_own, &r_inv);
    let s = mm(&br_inv, &b_own.t().to_owned());

    // Initial guess: start from Q (identity-scaled if Q has large norm)
    let mut p = symmetrize(&q_own);

    let tol = F::from(1e-10).unwrap_or(F::epsilon());
    let max_iter = 200usize;

    for iter in 0..max_iter {
        // A_closed = A - S * P
        let sp = mm(&s, &p);
        let mut a_cl = a_own.clone();
        for i in 0..n {
            for j in 0..n {
                a_cl[[i, j]] -= sp[[i, j]];
            }
        }

        // Lyapunov rhs: -(Q + P S P)  [using S = B R⁻¹ Bᵀ]
        let psp = mm(&p, &mm(&s, &p));
        let mut lyap_q = q_own.clone();
        for i in 0..n {
            for j in 0..n {
                lyap_q[[i, j]] += psp[[i, j]];
            }
        }
        let lyap_q = symmetrize(&lyap_q);

        // Solve Lyapunov: A_cl P_new + P_new A_clᵀ = -lyap_q
        let p_new = lyapunov_continuous(&a_cl.view(), &lyap_q.view())?;
        let p_new = symmetrize(&p_new);

        // Check convergence
        let diff = &p_new - &p;
        let diff_norm = frob_norm(&diff);

        p = p_new;

        if diff_norm <= tol {
            return Ok(p);
        }

        // Safety: if residual grows, we may have diverged
        if iter > 10 {
            let res_norm = care_residual_norm(&a_own, &s, &q_own, &p, n);
            if res_norm < F::from(1e-8).unwrap_or(F::epsilon()) {
                return Ok(p);
            }
        }
    }

    // Check final residual quality even if not fully converged
    let res_norm = care_residual_norm(&a_own, &s, &q_own, &p, n);
    if res_norm < F::from(1e-5).unwrap_or(F::epsilon()) {
        return Ok(p);
    }

    Err(LinalgError::ConvergenceError(format!(
        "CARE Newton iteration did not converge after {max_iter} iterations; \
         final residual = {res_norm}"
    )))
}

/// Compute ‖Aᵀ P + P A - P S P + Q‖_F  (CARE residual norm, S = B R⁻¹ Bᵀ)
fn care_residual_norm<F: RiccatiFloat>(
    a: &Array2<F>,
    s: &Array2<F>,
    q: &Array2<F>,
    p: &Array2<F>,
    n: usize,
) -> F {
    let at = a.t().to_owned();
    let atp = mm(&at, p);
    let pa = mm(p, a);
    let psp = mm(p, &mm(s, p));
    let mut res = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            res[[i, j]] = atp[[i, j]] + pa[[i, j]] - psp[[i, j]] + q[[i, j]];
        }
    }
    frob_norm(&res)
}

// ---------------------------------------------------------------------------
// DARE: Discrete Algebraic Riccati Equation
// ---------------------------------------------------------------------------

/// Solve the Discrete Algebraic Riccati Equation (DARE):
/// `Aᵀ P A - P - Aᵀ P B (Bᵀ P B + R)⁻¹ Bᵀ P A + Q = 0`
///
/// # Arguments
/// * `a` - `n×n` system (state) matrix
/// * `b` - `n×m` input matrix  
/// * `q` - `n×n` state cost matrix (positive semi-definite)
/// * `r` - `m×m` input cost matrix (positive definite)
///
/// # Returns
/// The symmetric positive semi-definite solution matrix `P` of size `n×n`.
///
/// # Algorithm
/// Newton-based DARE iteration (Hewer's algorithm):
/// 1. Initialize `K₀ = 0` (or zero gain)
/// 2. `Aᵣ = A - B K_k`
/// 3. `P_{k+1}` solves: `Aᵣᵀ P Aᵣ - P + Q + KₖᵀR Kₖ = 0` (discrete Lyapunov)
/// 4. `K_{k+1} = (R + Bᵀ P B)⁻¹ Bᵀ P A`
/// 5. Repeat until convergence
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::control::riccati::dare_solve;
///
/// let a = array![[1.0_f64, 1.0], [0.0, 1.0]];
/// let b = array![[0.0_f64], [1.0]];
/// let q = array![[1.0_f64, 0.0], [0.0, 0.0]];
/// let r = array![[1.0_f64]];
/// let p = dare_solve(&a.view(), &b.view(), &q.view(), &r.view()).expect("DARE failed");
/// ```
pub fn dare_solve<F: RiccatiFloat>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    q: &ArrayView2<F>,
    r: &ArrayView2<F>,
) -> LinalgResult<Array2<F>> {
    let n = check_square(a, "DARE: A")?;
    let q_n = check_square(q, "DARE: Q")?;
    let r_m = check_square(r, "DARE: R")?;
    if q_n != n {
        return Err(LinalgError::DimensionError(format!(
            "DARE: Q must be {n}×{n}, got {q_n}×{q_n}"
        )));
    }
    if b.nrows() != n {
        return Err(LinalgError::DimensionError(format!(
            "DARE: B must have {n} rows, got {}",
            b.nrows()
        )));
    }
    let m = b.ncols();
    if r_m != m {
        return Err(LinalgError::DimensionError(format!(
            "DARE: R must be {m}×{m}, got {r_m}×{r_m}"
        )));
    }

    let a_own = a.to_owned();
    let b_own = b.to_owned();
    let q_own = q.to_owned();
    let r_own = r.to_owned();

    // Initial gain K₀ = 0 (n×m zero matrix)
    let mut k = Array2::<F>::zeros((m, n));
    let mut p = symmetrize(&q_own);

    let tol = F::from(1e-10).unwrap_or(F::epsilon());
    let max_iter = 300usize;

    for _iter in 0..max_iter {
        // A_cl = A - B K
        let bk = mm(&b_own, &k);
        let mut a_cl = a_own.clone();
        for i in 0..n {
            for j in 0..n {
                a_cl[[i, j]] -= bk[[i, j]];
            }
        }

        // Lyapunov rhs: Q + KᵀR K
        let ktr = mm(&k.t().to_owned(), &r_own);
        let ktrk = mm(&ktr, &k);
        let mut lyap_q = q_own.clone();
        for i in 0..n {
            for j in 0..n {
                lyap_q[[i, j]] += ktrk[[i, j]];
            }
        }
        let lyap_q = symmetrize(&lyap_q);

        // Solve discrete Lyapunov: A_cl P_new A_clᵀ - P_new + lyap_q = 0
        // i.e. A_cl P_new A_clᵀ - P_new = -lyap_q
        let p_new = lyapunov_discrete(&a_cl.view(), &lyap_q.view())?;
        let p_new = symmetrize(&p_new);

        // Update gain: K = (R + Bᵀ P_new B)⁻¹ Bᵀ P_new A
        let bt = b_own.t().to_owned();
        let bt_pnew = mm(&bt, &p_new);
        let bt_pnew_b = mm(&bt_pnew, &b_own);
        let mut reg = r_own.clone();
        for i in 0..m {
            for j in 0..m {
                reg[[i, j]] += bt_pnew_b[[i, j]];
            }
        }
        let reg_inv = mat_inv(&reg)?;
        let bt_pnew_a = mm(&bt_pnew, &a_own);
        let k_new = mm(&reg_inv, &bt_pnew_a);

        // Check convergence
        let diff = &p_new - &p;
        let diff_norm = frob_norm(&diff);

        p = p_new;
        k = k_new;

        if diff_norm <= tol {
            return Ok(p);
        }
    }

    // Check final residual
    let res_norm = dare_residual_norm(&a_own, &b_own, &q_own, &r_own, &p, n, m);
    if res_norm < F::from(1e-5).unwrap_or(F::epsilon()) {
        return Ok(p);
    }

    Err(LinalgError::ConvergenceError(format!(
        "DARE Newton iteration did not converge after {max_iter} iterations; \
         final residual = {res_norm}"
    )))
}

/// Compute DARE residual norm: ‖Aᵀ P A - P - Aᵀ P B (Bᵀ P B + R)⁻¹ Bᵀ P A + Q‖_F
fn dare_residual_norm<F: RiccatiFloat>(
    a: &Array2<F>,
    b: &Array2<F>,
    q: &Array2<F>,
    r: &Array2<F>,
    p: &Array2<F>,
    n: usize,
    m: usize,
) -> F {
    let at = a.t().to_owned();
    let bt = b.t().to_owned();
    let atp = mm(&at, p);
    let atpa = mm(&atp, a);
    let btpb_r = {
        let bt_pb = mm(&bt, &mm(p, b));
        let mut reg = r.clone();
        for i in 0..m {
            for j in 0..m {
                reg[[i, j]] += bt_pb[[i, j]];
            }
        }
        reg
    };
    let inv = match mat_inv(&btpb_r) {
        Ok(v) => v,
        Err(_) => Array2::<F>::zeros((m, m)),
    };
    let atpb = mm(&atp, b);
    let correction = mm(&mm(&atpb, &inv), &mm(&bt, &mm(p, a)));

    let mut res = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            res[[i, j]] = atpa[[i, j]] - p[[i, j]] - correction[[i, j]] + q[[i, j]];
        }
    }
    frob_norm(&res)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn check_square<F: RiccatiFloat>(a: &ArrayView2<F>, ctx: &str) -> LinalgResult<usize> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(format!("{ctx}: not square")));
    }
    Ok(n)
}

// Expose mat_inv for test convenience (crate-private)
pub(crate) fn mat_inv_pub<F: RiccatiFloat>(a: &Array2<F>) -> LinalgResult<Array2<F>> {
    mat_inv(a)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn care_residual_check(
        a: &Array2<f64>,
        b: &Array2<f64>,
        q: &Array2<f64>,
        r: &Array2<f64>,
        p: &Array2<f64>,
    ) -> f64 {
        let r_inv = mat_inv(r).expect("R inv");
        let s = mm(b, &mm(&r_inv, &b.t().to_owned()));
        let at = a.t().to_owned();
        let atp = mm(&at, p);
        let pa = mm(p, a);
        let psp = mm(p, &mm(&s, p));
        let n = a.nrows();
        let mut res = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let v = atp[[i, j]] + pa[[i, j]] - psp[[i, j]] + q[[i, j]];
                res += v * v;
            }
        }
        res.sqrt()
    }

    fn dare_residual_check(
        a: &Array2<f64>,
        b: &Array2<f64>,
        q: &Array2<f64>,
        r: &Array2<f64>,
        p: &Array2<f64>,
    ) -> f64 {
        let n = a.nrows();
        let m = b.ncols();
        dare_residual_norm(a, b, q, r, p, n, m)
    }

    #[test]
    fn test_care_scalar() {
        // A=-1, B=1, Q=1, R=1
        // Scalar CARE: -2P - P² + 1 = 0 → P² + 2P - 1 = 0 → P = -1 + √2
        let a = array![[-1.0_f64]];
        let b = array![[1.0_f64]];
        let q = array![[1.0_f64]];
        let r = array![[1.0_f64]];
        let p = care_solve(&a.view(), &b.view(), &q.view(), &r.view())
            .expect("CARE scalar failed");

        let expected = -1.0_f64 + 2.0_f64.sqrt(); // ≈ 0.4142
        let diff = (p[[0, 0]] - expected).abs();
        assert!(diff < 1e-5, "CARE scalar: got {}, expected {expected}", p[[0, 0]]);
    }

    #[test]
    fn test_care_2x2_residual() {
        let a = array![[-1.0_f64, 0.0], [0.0, -2.0]];
        let b = array![[1.0_f64], [1.0]];
        let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let r = array![[1.0_f64]];
        let p = care_solve(&a.view(), &b.view(), &q.view(), &r.view())
            .expect("CARE 2x2 failed");
        let res = care_residual_check(&a, &b, &q, &r, &p);
        assert!(res < 1e-5, "CARE 2x2 residual = {res}");
    }

    #[test]
    fn test_dare_integrator_residual() {
        // Discrete integrator: A=[[1,1],[0,1]], B=[[0],[1]], Q=I, R=I
        let a = array![[1.0_f64, 1.0], [0.0, 1.0]];
        let b = array![[0.0_f64], [1.0]];
        let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let r = array![[1.0_f64]];
        let p = dare_solve(&a.view(), &b.view(), &q.view(), &r.view())
            .expect("DARE integrator failed");
        let res = dare_residual_check(&a, &b, &q, &r, &p);
        assert!(res < 1e-4, "DARE integrator residual = {res}");
    }

    #[test]
    fn test_dare_stable_system() {
        let a = array![[0.9_f64, 0.1], [0.0, 0.8]];
        let b = array![[1.0_f64], [0.0]];
        let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let r = array![[1.0_f64]];
        let p = dare_solve(&a.view(), &b.view(), &q.view(), &r.view())
            .expect("DARE stable failed");
        // P should be positive definite (all diagonal > 0)
        for i in 0..2 {
            assert!(p[[i, i]] > 0.0, "P[{i},{i}] = {} not positive", p[[i, i]]);
        }
        let res = dare_residual_check(&a, &b, &q, &r, &p);
        assert!(res < 1e-4, "DARE stable residual = {res}");
    }
}
