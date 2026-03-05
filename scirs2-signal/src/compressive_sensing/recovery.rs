//! Signal recovery algorithms for compressive sensing.
//!
//! Provides Basis Pursuit (ADMM), BPDN, and Dantzig Selector.

use crate::error::{SignalError, SignalResult};

const EPS: f64 = 1e-12;

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Dot product.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

/// L2 norm.
fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Matrix-vector product: A (m×n) times x (n) → y (m).
fn matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| dot(row, x))
        .collect()
}

/// Transposed matrix-vector product: A^T (m×n) times v (m) → u (n).
fn matvec_t(a: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    if a.is_empty() || v.is_empty() {
        return vec![];
    }
    let n = a[0].len();
    let mut u = vec![0.0_f64; n];
    for (i, row) in a.iter().enumerate() {
        let vi = v[i];
        for (j, &aij) in row.iter().enumerate() {
            u[j] += aij * vi;
        }
    }
    u
}

/// Validate that A has consistent dimensions and matches b.
fn validate_dims(a: &[Vec<f64>], b: &[f64]) -> SignalResult<(usize, usize)> {
    let m = a.len();
    if m == 0 {
        return Err(SignalError::InvalidArgument("A has no rows".into()));
    }
    if b.len() != m {
        return Err(SignalError::DimensionMismatch(format!(
            "A has {m} rows but b has {} entries",
            b.len()
        )));
    }
    let n = a[0].len();
    if n == 0 {
        return Err(SignalError::InvalidArgument("A has no columns".into()));
    }
    for (i, row) in a.iter().enumerate() {
        if row.len() != n {
            return Err(SignalError::DimensionMismatch(format!(
                "Row {i} has {} columns, expected {n}",
                row.len()
            )));
        }
    }
    Ok((m, n))
}

/// Soft-threshold: S_λ(x) = sign(x) * max(|x| - λ, 0).
fn soft_threshold(v: &[f64], lambda: f64) -> Vec<f64> {
    v.iter()
        .map(|&x| {
            let abs = x.abs();
            if abs <= lambda {
                0.0
            } else {
                x.signum() * (abs - lambda)
            }
        })
        .collect()
}

/// Solve the normal equations `(A^T A + rho I) x = A^T b + rho * z_aug` via CG.
/// Used inside ADMM for the x-update.
fn cg_solve(
    a: &[Vec<f64>],
    at_b: &[f64],     // A^T b
    z_aug: &[f64],    // z - u (the augmented term)
    rho: f64,
    n: usize,
    max_cg_iter: usize,
    cg_tol: f64,
) -> Vec<f64> {
    // RHS = A^T b + rho * z_aug
    let rhs: Vec<f64> = at_b
        .iter()
        .zip(z_aug.iter())
        .map(|(&atb_i, &zi)| atb_i + rho * zi)
        .collect();

    // Start from x = 0, so r = rhs
    let mut x = vec![0.0_f64; n];
    let mut r = rhs;
    let mut p = r.clone();
    let mut r_norm_sq = dot(&r, &r);

    for _ in 0..max_cg_iter {
        if r_norm_sq < cg_tol * cg_tol {
            break;
        }
        // q = (A^T A + rho I) p
        let ap = matvec(a, &p);
        let atp = matvec_t(a, &ap);
        let q: Vec<f64> = (0..n).map(|j| atp[j] + rho * p[j]).collect();
        let pq = dot(&p, &q);
        if pq.abs() < EPS {
            break;
        }
        let alpha = r_norm_sq / pq;
        for j in 0..n {
            x[j] += alpha * p[j];
            r[j] -= alpha * q[j];
        }
        let r_norm_sq_new = dot(&r, &r);
        let beta = if r_norm_sq > EPS { r_norm_sq_new / r_norm_sq } else { 0.0 };
        for j in 0..n {
            p[j] = r[j] + beta * p[j];
        }
        r_norm_sq = r_norm_sq_new;
    }
    x
}

// ──────────────────────────────────────────────────────────────────────────────
// Basis Pursuit (ADMM)
// ──────────────────────────────────────────────────────────────────────────────

/// Basis Pursuit: min ||x||_1  s.t. Ax = b
///
/// Solved via ADMM:
/// - x-update: (A^T A + rho I) x = A^T b + rho (z - u)
/// - z-update: S_{1/rho}(x + u)
/// - u-update: u += x - z
pub fn basis_pursuit(
    a: &[Vec<f64>],
    b: &[f64],
    max_iter: usize,
    tol: f64,
) -> SignalResult<Vec<f64>> {
    let (m, n) = validate_dims(a, b)?;

    // Estimate a good initial rho based on the signal magnitude
    let b_norm = norm2(b);
    let mut rho = (10.0 * b_norm / (n as f64).sqrt()).max(1.0);
    let max_cg = 200_usize;
    let cg_tol = 1e-10_f64;

    // Precompute A^T b
    let at_b = matvec_t(a, b);

    let mut x = vec![0.0_f64; n];
    let mut z = vec![0.0_f64; n];
    let mut u = vec![0.0_f64; n];

    let mu = 10.0_f64; // rho adjustment factor
    let tau_incr = 2.0_f64;
    let tau_decr = 2.0_f64;

    for _iter in 0..max_iter {
        // x-update: (A^T A + rho I) x = A^T b + rho (z - u)
        let z_minus_u: Vec<f64> = z.iter().zip(u.iter()).map(|(&zi, &ui)| zi - ui).collect();
        x = cg_solve(a, &at_b, &z_minus_u, rho, n, max_cg, cg_tol);

        // z-update: z = S_{1/rho}(x + u)
        let x_plus_u: Vec<f64> = x.iter().zip(u.iter()).map(|(&xi, &ui)| xi + ui).collect();
        let z_new = soft_threshold(&x_plus_u, 1.0 / rho);

        // u-update: u += x - z
        let u_new: Vec<f64> = u
            .iter()
            .zip(x.iter())
            .zip(z_new.iter())
            .map(|((&ui, &xi), &zi)| ui + xi - zi)
            .collect();

        // Primal residual ||x - z|| and dual residual ||rho (z_new - z)||
        let primal_res = norm2(&x.iter().zip(z_new.iter()).map(|(&xi, &zi)| xi - zi).collect::<Vec<f64>>());
        let dual_res = rho * norm2(&z_new.iter().zip(z.iter()).map(|(&zn, &z_old)| zn - z_old).collect::<Vec<f64>>());

        z = z_new;
        u = u_new;

        if primal_res < tol && dual_res < tol {
            break;
        }

        // Adaptive rho: balance primal and dual residuals
        if primal_res > mu * dual_res {
            rho *= tau_incr;
            // Scale u by 1/tau_incr to keep rho*u consistent
            for ui in u.iter_mut() {
                *ui /= tau_incr;
            }
        } else if dual_res > mu * primal_res {
            rho /= tau_decr;
            for ui in u.iter_mut() {
                *ui *= tau_decr;
            }
        }
    }

    // Ensure constraint Ax = b is satisfied via least-squares correction
    let ax = matvec(a, &x);
    let residual: Vec<f64> = ax.iter().zip(b.iter()).map(|(&axi, &bi)| axi - bi).collect();
    let residual_norm = norm2(&residual);
    if residual_norm > tol * 100.0 && m <= n {
        // Apply one step of least-squares correction: x -= A^T (A A^T)^{-1} residual
        // Simplified: x -= A^T * residual / ||A residual||^2 (crude gradient step)
        let at_res = matvec_t(a, &residual);
        let a_at_res = matvec(a, &at_res);
        let denom = dot(&a_at_res, &residual);
        if denom.abs() > EPS {
            let step = dot(&residual, &residual) / denom;
            for j in 0..n {
                x[j] -= step * at_res[j];
            }
        }
    }

    Ok(x)
}

// ──────────────────────────────────────────────────────────────────────────────
// Basis Pursuit Denoising (BPDN)
// ──────────────────────────────────────────────────────────────────────────────

/// Basis Pursuit Denoising: min ||x||_1 + (lambda/2) ||Ax - b||^2
///
/// ADMM formulation:
/// - x-update: (lambda A^T A + rho I) x = lambda A^T b + rho (z - u)
/// - z-update: S_{1/rho}(x + u)
/// - u-update: u += x - z
pub fn basis_pursuit_denoising(
    a: &[Vec<f64>],
    b: &[f64],
    lambda: f64,
    max_iter: usize,
    tol: f64,
) -> SignalResult<Vec<f64>> {
    let (_, n) = validate_dims(a, b)?;
    if lambda <= 0.0 {
        return Err(SignalError::InvalidArgument("lambda must be > 0".into()));
    }

    let rho = lambda.sqrt();
    let max_cg = 50_usize;
    let cg_tol = 1e-8_f64;

    // Precompute lambda * A^T b
    let at_b_scaled: Vec<f64> = matvec_t(a, b).iter().map(|&v| lambda * v).collect();

    // Scale A by sqrt(lambda) so that (sqrt(lambda)*A)^T (sqrt(lambda)*A) = lambda A^T A
    let a_scaled: Vec<Vec<f64>> = a
        .iter()
        .map(|row| row.iter().map(|&v| v * lambda.sqrt()).collect())
        .collect();

    let mut x = vec![0.0_f64; n];
    let mut z = vec![0.0_f64; n];
    let mut u = vec![0.0_f64; n];

    for _iter in 0..max_iter {
        // x-update: (lambda A^T A + rho I) x = lambda A^T b + rho (z - u)
        let z_minus_u: Vec<f64> = z.iter().zip(u.iter()).map(|(&zi, &ui)| zi - ui).collect();
        x = cg_solve(&a_scaled, &at_b_scaled, &z_minus_u, rho, n, max_cg, cg_tol);

        // z-update: z = S_{1/rho}(x + u)
        let x_plus_u: Vec<f64> = x.iter().zip(u.iter()).map(|(&xi, &ui)| xi + ui).collect();
        let z_new = soft_threshold(&x_plus_u, 1.0 / rho);

        // u-update
        let u_new: Vec<f64> = u
            .iter()
            .zip(x.iter())
            .zip(z_new.iter())
            .map(|((&ui, &xi), &zi)| ui + xi - zi)
            .collect();

        let primal_res = norm2(&x.iter().zip(z_new.iter()).map(|(&xi, &zi)| xi - zi).collect::<Vec<f64>>());
        let dual_res = rho * norm2(&z_new.iter().zip(z.iter()).map(|(&zn, &zo)| zn - zo).collect::<Vec<f64>>());

        z = z_new;
        u = u_new;

        if primal_res < tol && dual_res < tol {
            break;
        }
    }

    Ok(x)
}

// ──────────────────────────────────────────────────────────────────────────────
// Dantzig Selector
// ──────────────────────────────────────────────────────────────────────────────

/// Dantzig Selector (approximate): min ||x||_1  s.t. ||A^T(Ax - b)||_inf ≤ delta.
///
/// Solved via iterative reweighted L1 (ADMM-like).
pub fn dantzig_selector(
    a: &[Vec<f64>],
    b: &[f64],
    delta: f64,
    max_iter: usize,
) -> SignalResult<Vec<f64>> {
    let (m, n) = validate_dims(a, b)?;
    if delta <= 0.0 {
        return Err(SignalError::InvalidArgument("delta must be > 0".into()));
    }
    let _ = m;

    // Iterative reweighted L1 approach:
    // Minimise sum_i w_i |x_i|  subject to ||A^T(Ax - b)||_inf ≤ delta
    // where weights are updated as w_i = 1 / (|x_i| + eps_irls).
    //
    // Each step: ISTA with soft threshold guided by the constraint.

    let rho = 1.0 / delta;
    let at_b = matvec_t(a, b);

    let mut x = vec![0.0_f64; n];
    let eps_irls = 1e-3;

    // Spectral norm (power method) for step size
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    for _ in 0..30 {
        let av = matvec(a, &v);
        let atav = matvec_t(a, &av);
        let nrm = norm2(&atav).max(EPS);
        v = atav.iter().map(|&vi| vi / nrm).collect();
    }
    let av = matvec(a, &v);
    let atav = matvec_t(a, &av);
    let lip = dot(&v, &atav).max(1.0);
    let step = 1.0 / lip;

    for _iter in 0..max_iter {
        // ISTA step: grad = A^T(Ax - b) = A^T A x - A^T b
        let ax = matvec(a, &x);
        let atax = matvec_t(a, &ax);
        let grad: Vec<f64> = atax.iter().zip(at_b.iter()).map(|(&atax_j, &atb_j)| atax_j - atb_j).collect();

        // Reweighted soft threshold
        let mut weights = vec![1.0; n];
        for j in 0..n {
            weights[j] = 1.0 / (x[j].abs() + eps_irls);
        }

        let x_half: Vec<f64> = x.iter().zip(grad.iter()).map(|(&xj, &gj)| xj - step * gj).collect();
        let x_new: Vec<f64> = x_half
            .iter()
            .zip(weights.iter())
            .map(|(&xh, &w)| {
                let thresh = step * rho * w;
                let abs = xh.abs();
                if abs <= thresh { 0.0 } else { xh.signum() * (abs - thresh) }
            })
            .collect();

        let diff = norm2(&x_new.iter().zip(x.iter()).map(|(&xn, &xo)| xn - xo).collect::<Vec<f64>>());
        x = x_new;

        if diff < 1e-6 {
            break;
        }
    }

    Ok(x)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build A = identity (n×n), b = e_k (unit vector at position k).
    fn identity_problem(n: usize, k: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let a: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let b: Vec<f64> = (0..n).map(|i| if i == k { 1.0 } else { 0.0 }).collect();
        (a, b)
    }

    #[test]
    fn test_bp_identity_system() {
        let (a, b) = identity_problem(8, 3);
        let x = basis_pursuit(&a, &b, 200, 1e-4).expect("bp");
        assert_eq!(x.len(), 8);
        // x[3] should be close to 1 and others small
        assert!(x[3] > 0.5, "x[3]={}", x[3]);
    }

    #[test]
    fn test_bpdn_identity_system() {
        let (a, b) = identity_problem(8, 2);
        let x = basis_pursuit_denoising(&a, &b, 1.0, 200, 1e-4).expect("bpdn");
        assert_eq!(x.len(), 8);
        assert!(x[2] > 0.3, "x[2]={}", x[2]);
    }

    #[test]
    fn test_bpdn_lambda_zero_error() {
        let (a, b) = identity_problem(4, 0);
        assert!(basis_pursuit_denoising(&a, &b, 0.0, 10, 1e-4).is_err());
    }

    #[test]
    fn test_dantzig_selector_identity() {
        let (a, b) = identity_problem(8, 5);
        let x = dantzig_selector(&a, &b, 0.1, 200).expect("dantzig");
        assert_eq!(x.len(), 8);
        assert!(x[5] > 0.3, "x[5]={}", x[5]);
    }

    #[test]
    fn test_dimension_mismatch() {
        let a: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![1.0, 0.0, 0.0]; // wrong length
        assert!(basis_pursuit(&a, &b, 10, 1e-4).is_err());
        assert!(basis_pursuit_denoising(&a, &b, 1.0, 10, 1e-4).is_err());
        assert!(dantzig_selector(&a, &b, 0.1, 10).is_err());
    }

    #[test]
    fn test_bp_underdetermined_sparse() {
        // A is 6×12, x_true is 2-sparse, measure b = Ax_true
        let n = 12_usize;
        let m = 6_usize;
        // Simple structured A
        let a: Vec<Vec<f64>> = (0..m)
            .map(|i| (0..n).map(|j| (((i + 1) * (j + 1)) as f64 * 0.1).sin()).collect())
            .collect();
        // Normalise A columns
        let a: Vec<Vec<f64>> = {
            let mut a2 = a.clone();
            for j in 0..n {
                let col_norm: f64 = a2.iter().map(|row| row[j] * row[j]).sum::<f64>().sqrt().max(EPS);
                for row in &mut a2 { row[j] /= col_norm; }
            }
            a2
        };
        let mut x_true = vec![0.0_f64; n];
        x_true[2] = 3.0;
        x_true[7] = -2.0;
        let b = matvec(&a, &x_true);
        let x_rec = basis_pursuit(&a, &b, 300, 1e-5).expect("bp underdetermined");
        // Check Ax ≈ b
        let ax = matvec(&a, &x_rec);
        let res_norm: f64 = ax.iter().zip(b.iter()).map(|(&ai, &bi)| (ai - bi).powi(2)).sum::<f64>().sqrt();
        assert!(res_norm < 0.5, "BP residual {res_norm:.4} too large");
    }
}
