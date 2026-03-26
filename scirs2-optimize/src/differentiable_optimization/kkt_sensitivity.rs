//! KKT sensitivity analysis for differentiable optimization layers.
//!
//! Given an equality-constrained QP:
//!
//!   min  ½ xᵀQx + cᵀx   s.t.  Ax = b
//!
//! the KKT conditions are:
//!
//!   Qx + c + Aᵀν = 0   (stationarity)
//!   Ax = b              (primal feasibility)
//!
//! The bordered KKT system is:
//!
//!   K = [Q  Aᵀ]
//!       [A  0 ]
//!
//! Differentiating through the KKT conditions gives the adjoint system:
//!
//!   Kᵀ [dx_adj; dν_adj] = [dL/dx; 0]
//!
//! from which parameter gradients can be extracted.
//!
//! For the general nonlinear case, we use the parametric NLP adjoint method
//! to differentiate through solutions of:
//!
//!   min  f(x, θ)   s.t.  g(x, θ) ≤ 0,  h(x, θ) = 0

use crate::error::{OptimizeError, OptimizeResult};

use super::implicit_diff::solve_implicit_system;

// ─────────────────────────────────────────────────────────────────────────────
// KKT matrix assembly
// ─────────────────────────────────────────────────────────────────────────────

/// Assemble the bordered KKT matrix for the equality-constrained QP:
///
///   min  ½ xᵀQx + cᵀx   s.t.  Ax = b
///
/// The bordered system is:
///
///   K = [Q  Aᵀ]   size: (n+p) × (n+p)
///       [A  0 ]
///
/// # Arguments
/// * `q` – n×n cost matrix (symmetric PSD).
/// * `a` – p×n equality constraint matrix.
///
/// # Returns
/// The bordered KKT matrix as a row-major `Vec<Vec<f64>>`.
pub fn kkt_matrix(q: &[Vec<f64>], a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = q.len();
    let p = a.len();
    let dim = n + p;

    let mut k = vec![vec![0.0_f64; dim]; dim];

    // Block (0,0): Q  (n×n)
    for i in 0..n {
        for j in 0..n {
            k[i][j] = if i < q.len() && j < q[i].len() {
                q[i][j]
            } else {
                0.0
            };
        }
    }

    // Block (0,1): Aᵀ  (n×p)
    for i in 0..p {
        for j in 0..n {
            let a_val = if i < a.len() && j < a[i].len() {
                a[i][j]
            } else {
                0.0
            };
            k[j][n + i] = a_val;
        }
    }

    // Block (1,0): A  (p×n)
    for i in 0..p {
        for j in 0..n {
            let a_val = if i < a.len() && j < a[i].len() {
                a[i][j]
            } else {
                0.0
            };
            k[n + i][j] = a_val;
        }
    }

    // Block (1,1): 0  (p×p) — already zero from initialization
    k
}

// ─────────────────────────────────────────────────────────────────────────────
// KKT sensitivity gradients
// ─────────────────────────────────────────────────────────────────────────────

/// Gradients of the loss w.r.t. QP parameters, computed via KKT sensitivity.
#[derive(Debug, Clone)]
pub struct KktGrad {
    /// Gradient dL/dQ  (n×n symmetric).
    pub dl_dq: Vec<Vec<f64>>,
    /// Gradient dL/dc  (n).
    pub dl_dc: Vec<f64>,
    /// Gradient dL/dA  (p×n).
    pub dl_da: Vec<Vec<f64>>,
    /// Gradient dL/db  (p).
    pub dl_db: Vec<f64>,
    /// Adjoint of x  (dx_adj, n).
    pub dx_adj: Vec<f64>,
    /// Adjoint of λ  (dnu_adj, p).
    pub dnu_adj: Vec<f64>,
}

/// Compute KKT sensitivity gradients for the equality-constrained QP.
///
/// Given the optimal primal x* and dual ν*, and the upstream gradient dL/dx,
/// solve the adjoint KKT system:
///
///   [Q  Aᵀ]ᵀ [dx_adj; dν_adj] = [dL/dx; 0]
///
/// Then extract:
///
///   dL/dQ = ½ (x dx_adjᵀ + dx_adj xᵀ)   (symmetric outer product)
///   dL/dc = dx_adj
///   dL/dA = dν_adj xᵀ + ν dx_adjᵀ        (via chain rule)
///   dL/db = -dν_adj
///
/// # Arguments
/// * `q` – n×n cost matrix.
/// * `a` – p×n equality constraint matrix.
/// * `x` – optimal primal solution (length n).
/// * `nu` – optimal dual variables (length p).
/// * `dl_dx` – upstream gradient dL/dx (length n).
///
/// # Errors
/// Returns `OptimizeError::ComputationError` if the KKT system is singular.
pub fn kkt_sensitivity(
    q: &[Vec<f64>],
    a: &[Vec<f64>],
    x: &[f64],
    nu: &[f64],
    dl_dx: &[f64],
) -> OptimizeResult<KktGrad> {
    let n = x.len();
    let p = nu.len();

    if dl_dx.len() != n {
        return Err(OptimizeError::InvalidInput(format!(
            "dl_dx length {} != n {}",
            dl_dx.len(),
            n
        )));
    }

    // Build the bordered KKT matrix K = [Q Aᵀ; A 0]
    let k = kkt_matrix(q, a);

    // Transpose K (K is symmetric for equality-constrained QP when Q is symmetric,
    // but we solve the adjoint system explicitly for correctness in general case)
    let dim = n + p;
    let mut k_t = vec![vec![0.0_f64; dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            k_t[i][j] = k[j][i];
        }
    }

    // RHS for adjoint system: [dL/dx; 0_p]
    let mut rhs = vec![0.0_f64; dim];
    for i in 0..n {
        rhs[i] = dl_dx[i];
    }
    // rhs[n..n+p] = 0 (already zero)

    // Solve: Kᵀ [dx_adj; dν_adj] = [dL/dx; 0]
    let adj = solve_implicit_system(&k_t, &rhs)?;

    let dx_adj = adj[..n].to_vec();
    let dnu_adj = adj[n..n + p].to_vec();

    // ── Compute parameter gradients via adjoint method ────────────────────
    // The adjoint rule: dL/dθ = -(∂F/∂θ)ᵀ adj
    // where adj = [dx_adj; dν_adj] satisfies Kᵀ adj = [dL/dx; 0].

    // ∂F₁/∂c = I → dL/dc = -(I·dx_adj) = -dx_adj
    let dl_dc: Vec<f64> = dx_adj.iter().map(|&v| -v).collect();

    // ∂F₂/∂b = -I → dL/db = -(-I·dν_adj) = dν_adj
    let dl_db: Vec<f64> = dnu_adj.to_vec();

    // ∂F₁/∂Q_{ij} = x_j (symmetric: we take ½ for the symmetrized form)
    // dL/dQ_{ij} = -(dx_adj_i * x_j) → symmetric: -½(x·dx_adjᵀ + dx_adj·xᵀ)
    let mut dl_dq = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            dl_dq[i][j] = -0.5 * (dx_adj[i] * x[j] + x[i] * dx_adj[j]);
        }
    }

    // ∂F₁/∂A_{ij} = ν_i δ_j·  (∂(Aᵀν)_j / ∂A_{ij} = ν_i)
    // ∂F₂/∂A_{ij} = x_j δ_i·  (∂(Ax)_i / ∂A_{ij} = x_j)
    // dL/dA_{ij} = -(ν_i * dx_adj_j + x_j * dν_adj_i)
    let mut dl_da = vec![vec![0.0_f64; n]; p];
    for i in 0..p {
        for j in 0..n {
            dl_da[i][j] = -(nu[i] * dx_adj[j] + x[j] * dnu_adj[i]);
        }
    }

    Ok(KktGrad {
        dl_dq,
        dl_dc,
        dl_da,
        dl_db,
        dx_adj,
        dnu_adj,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Parametric NLP adjoint
// ─────────────────────────────────────────────────────────────────────────────

/// Gradients for a general nonlinear program via the adjoint method.
#[derive(Debug, Clone)]
pub struct NlpGrad {
    /// Gradient of loss w.r.t. objective gradient parameters (n).
    pub dl_df_params: Vec<f64>,
    /// Gradient of loss w.r.t. inequality constraint parameters (m × n adjoint).
    pub dl_dg_params: Vec<Vec<f64>>,
    /// Gradient of loss w.r.t. equality constraint parameters (p × n adjoint).
    pub dl_dh_params: Vec<Vec<f64>>,
    /// Adjoint of primal variables (dx_adj).
    pub dx_adj: Vec<f64>,
    /// Adjoint of inequality duals (dlambda_adj).
    pub dlambda_adj: Vec<f64>,
    /// Adjoint of equality duals (dnu_adj).
    pub dnu_adj: Vec<f64>,
}

/// Compute the parametric NLP adjoint gradients.
///
/// Given the optimal solution (x*, λ*, ν*) of:
///
///   min  f(x)   s.t.  g(x) ≤ 0,  h(x) = 0
///
/// and Jacobians:
/// - `f_grad` : ∇f(x*) evaluated at x* (length n)
/// - `g_jac`  : ∂g/∂x at x* (m×n row-major)
/// - `h_jac`  : ∂h/∂x at x* (p×n row-major)
///
/// the KKT stationarity condition is:
///
///   ∇f + Gᵀλ + Hᵀν = 0
///
/// We solve the bordered KKT adjoint system using only the active inequality
/// constraints (those with λ_i > 0).
///
/// # Arguments
/// * `f_grad` – ∇f(x*) (length n).
/// * `g_jac`  – Jacobian of inequality constraints at x* (m×n).
/// * `h_jac`  – Jacobian of equality constraints at x* (p×n).
/// * `x_star` – optimal primal (length n).
/// * `lambda_star` – optimal inequality duals (length m, ≥ 0).
/// * `nu_star` – optimal equality duals (length p).
/// * `dl_dx`  – upstream gradient dL/dx* (length n).
///
/// # Errors
/// Returns `OptimizeError::ComputationError` if the adjoint system is singular.
pub fn parametric_nlp_adjoint(
    f_grad: &[f64],
    g_jac: &[Vec<f64>],
    h_jac: &[Vec<f64>],
    x_star: &[f64],
    lambda_star: &[f64],
    nu_star: &[f64],
    dl_dx: &[f64],
) -> OptimizeResult<NlpGrad> {
    let n = x_star.len();
    let m = lambda_star.len();
    let p = nu_star.len();

    if dl_dx.len() != n {
        return Err(OptimizeError::InvalidInput(format!(
            "dl_dx length {} != n {}",
            dl_dx.len(),
            n
        )));
    }

    // Identify active inequality constraints: λ_i > 0 (complementary slackness)
    let active_tol = 1e-8_f64;
    let active_ineq: Vec<usize> = (0..m).filter(|&i| lambda_star[i] > active_tol).collect();
    let m_act = active_ineq.len();

    // Build the bordered KKT Jacobian for the active constraints:
    //
    //   [∇²L_xx    G_act^T    H^T ]   size: (n + m_act + p) × (n + m_act + p)
    //   [diag(λ_act) G_act    0   ]
    //   [H          0         0   ]
    //
    // For the adjoint method we approximate ∇²L_xx ≈ Q (from f_grad structure).
    // Since we only have f_grad (not the Hessian), we use a rank-0 approximation
    // with a small regularization: ∇²L_xx ≈ reg * I.
    let reg = 1e-8_f64;
    let dim = n + m_act + p;
    let mut jac = vec![vec![0.0_f64; dim]; dim];

    // Block (0,0): reg * I  (n×n)
    for i in 0..n {
        jac[i][i] = reg;
    }

    // Block (0,1): G_act^T  (n×m_act) — columns from active rows of g_jac
    for (col, &ai) in active_ineq.iter().enumerate() {
        for row in 0..n {
            let g_val = if ai < g_jac.len() && row < g_jac[ai].len() {
                g_jac[ai][row]
            } else {
                0.0
            };
            jac[row][n + col] = g_val;
        }
    }

    // Block (0,2): H^T  (n×p)
    for i in 0..p {
        for row in 0..n {
            let h_val = if i < h_jac.len() && row < h_jac[i].len() {
                h_jac[i][row]
            } else {
                0.0
            };
            jac[row][n + m_act + i] = h_val;
        }
    }

    // Block (1,0): diag(λ_act) G_act  (m_act×n)
    for (row, &ai) in active_ineq.iter().enumerate() {
        let lam_i = lambda_star[ai];
        for col in 0..n {
            let g_val = if ai < g_jac.len() && col < g_jac[ai].len() {
                g_jac[ai][col]
            } else {
                0.0
            };
            jac[n + row][col] = lam_i * g_val;
        }
    }

    // Block (1,1): 0 (diag(g(x*)) ≈ 0 at complementarity)
    // Block (2,0): H  (p×n)
    for i in 0..p {
        for col in 0..n {
            let h_val = if i < h_jac.len() && col < h_jac[i].len() {
                h_jac[i][col]
            } else {
                0.0
            };
            jac[n + m_act + i][col] = h_val;
        }
    }

    // Transpose for adjoint system
    let mut jac_t = vec![vec![0.0_f64; dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            jac_t[i][j] = jac[j][i];
        }
    }

    // RHS: [dL/dx; 0_m_act; 0_p]
    let mut rhs = vec![0.0_f64; dim];
    for i in 0..n {
        rhs[i] = dl_dx[i];
    }

    // Solve adjoint system
    let adj = solve_implicit_system(&jac_t, &rhs)?;

    let dx_adj = adj[..n].to_vec();
    let dlambda_adj_active = adj[n..n + m_act].to_vec();
    let dnu_adj = adj[n + m_act..].to_vec();

    // Expand dlambda_adj back to full m dimension
    let mut dlambda_adj = vec![0.0_f64; m];
    for (idx, &ai) in active_ineq.iter().enumerate() {
        if idx < dlambda_adj_active.len() {
            dlambda_adj[ai] = dlambda_adj_active[idx];
        }
    }

    // Compute parameter sensitivity:
    // dL/df_params = dx_adj  (∂f/∂θ is the gradient of f w.r.t. θ, but
    //   since we don't have parametric dependence here, we return the adjoint)
    let dl_df_params = dx_adj.clone();

    // dL/dG_{ij} = dlambda_adj_i * x_j + lambda_i * dx_adj_j
    let mut dl_dg_params = vec![vec![0.0_f64; n]; m];
    for i in 0..m {
        for j in 0..n {
            dl_dg_params[i][j] = dlambda_adj[i] * x_star[j] + lambda_star[i] * dx_adj[j];
        }
    }

    // dL/dH_{ij} = dnu_adj_i * x_j + nu_i * dx_adj_j
    let mut dl_dh_params = vec![vec![0.0_f64; n]; p];
    for i in 0..p {
        for j in 0..n {
            dl_dh_params[i][j] = dnu_adj[i] * x_star[j] + nu_star[i] * dx_adj[j];
        }
    }

    Ok(NlpGrad {
        dl_df_params,
        dl_dg_params,
        dl_dh_params,
        dx_adj,
        dlambda_adj,
        dnu_adj,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: Cholesky-based KKT solve
// ─────────────────────────────────────────────────────────────────────────────

/// Augment Q with a Tikhonov regularization term δI for numerical stability.
pub fn regularize_q(q: &[Vec<f64>], delta: f64) -> Vec<Vec<f64>> {
    let n = q.len();
    let mut q_reg = q.to_vec();
    for i in 0..n {
        if i < q_reg.len() && i < q_reg[i].len() {
            q_reg[i][i] += delta;
        }
    }
    q_reg
}

/// Compute the matrix-vector product y = Ax for a row-major matrix.
pub fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| {
            row.iter()
                .zip(x.iter())
                .map(|(&a_ij, &x_j)| a_ij * x_j)
                .sum()
        })
        .collect()
}

/// Compute the outer product of two vectors: C_{ij} = a_i * b_j.
pub fn outer_product(a: &[f64], b: &[f64]) -> Vec<Vec<f64>> {
    a.iter()
        .map(|&ai| b.iter().map(|&bj| ai * bj).collect())
        .collect()
}

/// Compute the symmetric outer product: C = ½(a bᵀ + b aᵀ).
pub fn sym_outer_product(a: &[f64], b: &[f64]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut c = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            c[i][j] = 0.5 * (a[i] * b[j] + b[i] * a[j]);
        }
    }
    c
}

// ─────────────────────────────────────────────────────────────────────────────
// KKT system struct for monitoring/debugging
// ─────────────────────────────────────────────────────────────────────────────

/// Represents the assembled KKT system for an equality-constrained QP.
///
/// Stores the bordered matrix and provides utilities for solving and
/// computing sensitivity.
#[derive(Debug, Clone)]
pub struct KktSystem {
    /// The bordered KKT matrix K = [Q Aᵀ; A 0].
    pub matrix: Vec<Vec<f64>>,
    /// Dimension n (number of primal variables).
    pub n: usize,
    /// Number of equality constraints p.
    pub p: usize,
}

impl KktSystem {
    /// Build a new KKT system from Q and A.
    pub fn new(q: &[Vec<f64>], a: &[Vec<f64>]) -> Self {
        let n = q.len();
        let p = a.len();
        let matrix = kkt_matrix(q, a);
        Self { matrix, n, p }
    }

    /// Solve the KKT system: K [x; ν] = [b1; b2].
    pub fn solve(&self, b1: &[f64], b2: &[f64]) -> OptimizeResult<(Vec<f64>, Vec<f64>)> {
        let dim = self.n + self.p;
        let mut rhs = Vec::with_capacity(dim);
        rhs.extend_from_slice(b1);
        rhs.extend_from_slice(b2);

        let sol = solve_implicit_system(&self.matrix, &rhs)?;
        let x = sol[..self.n].to_vec();
        let nu = sol[self.n..].to_vec();
        Ok((x, nu))
    }

    /// Compute the sensitivity gradients for the given upstream gradient.
    pub fn sensitivity(
        &self,
        q: &[Vec<f64>],
        a: &[Vec<f64>],
        x: &[f64],
        nu: &[f64],
        dl_dx: &[f64],
    ) -> OptimizeResult<KktGrad> {
        kkt_sensitivity(q, a, x, nu, dl_dx)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: numerical gradient via central differences
    fn numerical_gradient_kkt<F>(f: F, x: &[f64], eps: f64) -> Vec<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        let n = x.len();
        let mut grad = vec![0.0_f64; n];
        for i in 0..n {
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[i] += eps;
            xm[i] -= eps;
            grad[i] = (f(&xp) - f(&xm)) / (2.0 * eps);
        }
        grad
    }

    #[test]
    fn test_kkt_matrix_shape() {
        // Q: 3×3, A: 2×3 → bordered: 5×5
        let q = vec![
            vec![2.0, 0.0, 0.0],
            vec![0.0, 3.0, 0.0],
            vec![0.0, 0.0, 4.0],
        ];
        let a = vec![vec![1.0, 1.0, 0.0], vec![0.0, 1.0, 1.0]];

        let k = kkt_matrix(&q, &a);
        assert_eq!(k.len(), 5, "KKT matrix should be 5×5");
        assert_eq!(k[0].len(), 5);

        // Top-left block: Q
        assert!((k[0][0] - 2.0).abs() < 1e-12);
        assert!((k[1][1] - 3.0).abs() < 1e-12);
        assert!((k[2][2] - 4.0).abs() < 1e-12);

        // Top-right block: Aᵀ
        // A row 0: [1,1,0] → column 0 of Aᵀ = [1,1,0]
        assert!((k[0][3] - 1.0).abs() < 1e-12);
        assert!((k[1][3] - 1.0).abs() < 1e-12);
        assert!((k[2][3] - 0.0).abs() < 1e-12);

        // Bottom-left block: A
        assert!((k[3][0] - 1.0).abs() < 1e-12);
        assert!((k[3][1] - 1.0).abs() < 1e-12);
        assert!((k[4][1] - 1.0).abs() < 1e-12);
        assert!((k[4][2] - 1.0).abs() < 1e-12);

        // Bottom-right block: 0
        assert!((k[3][3]).abs() < 1e-12);
        assert!((k[4][4]).abs() < 1e-12);
    }

    #[test]
    fn test_kkt_matrix_symmetry() {
        let q = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let a = vec![vec![1.0, 2.0]];

        let k = kkt_matrix(&q, &a);
        let dim = k.len();

        for i in 0..dim {
            for j in 0..dim {
                assert!(
                    (k[i][j] - k[j][i]).abs() < 1e-12,
                    "KKT matrix not symmetric at ({},{}): {} vs {}",
                    i,
                    j,
                    k[i][j],
                    k[j][i]
                );
            }
        }
    }

    #[test]
    fn test_kkt_sensitivity_unconstrained() {
        // For unconstrained QP: min ½ xᵀQx + cᵀx
        // x*(c) = -Q⁻¹c, so dx*/dc = -Q⁻¹
        // dL/dc = (dx*/dc)ᵀ dL/dx = -Q⁻¹ dL/dx
        // With Q = 2I, dL/dx = [1, 0]: dL/dc = -(1/2)[1, 0] = [-0.5, 0]

        let q = vec![vec![2.0, 0.0], vec![0.0, 2.0]];
        let a: Vec<Vec<f64>> = vec![];
        let x = vec![-0.5, -1.0]; // x* = -Q^{-1} c for c = [1, 2]
        let nu: Vec<f64> = vec![];
        let dl_dx = vec![1.0, 0.0];

        let grad = kkt_sensitivity(&q, &a, &x, &nu, &dl_dx).expect("KKT sensitivity failed");

        // dL/dc = -Q⁻¹ dL/dx = -0.5 * [1, 0] = [-0.5, 0]
        assert!(
            (grad.dl_dc[0] - (-0.5)).abs() < 1e-10,
            "dl/dc[0] = {} (expected -0.5)",
            grad.dl_dc[0]
        );
        assert!(
            grad.dl_dc[1].abs() < 1e-10,
            "dl/dc[1] = {} (expected 0.0)",
            grad.dl_dc[1]
        );
    }

    #[test]
    fn test_kkt_sensitivity_with_equality() {
        // min ½||x||² s.t. x[0] + x[1] = 1
        // Optimal: x = [0.5, 0.5], ν = -0.5 (or +0.5 depending on sign convention)
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]]; // note: 2I → Q = I for this form
        let a = vec![vec![1.0, 1.0]];
        let x = vec![0.5, 0.5];
        let nu = vec![-0.5];
        let dl_dx = vec![1.0, 0.0];

        let grad = kkt_sensitivity(&q, &a, &x, &nu, &dl_dx).expect("KKT sensitivity failed");

        // Verify dl/dc is finite
        assert!(grad.dl_dc[0].is_finite(), "dl/dc[0] not finite");
        assert!(grad.dl_dc[1].is_finite(), "dl/dc[1] not finite");
        // Verify dl/db is finite
        assert!(grad.dl_db[0].is_finite(), "dl/db[0] not finite");
    }

    #[test]
    fn test_kkt_sensitivity_gradient_check_c() {
        // Verify dL/dc via finite differences
        // Problem: min ½ xᵀQx + cᵀx (unconstrained), x*(c) = -Q⁻¹c
        // Loss: L = 0.5 * ||x*||²
        // dL/dc_i = x*_j * (dx*_j/dc_i) = x*_i * (-Q⁻¹)_{ii}

        let q = vec![vec![4.0, 0.0], vec![0.0, 4.0]];
        let a: Vec<Vec<f64>> = vec![];
        // c = [2, 0] → x* = [-0.5, 0]
        let c = vec![2.0_f64, 0.0];
        let x = vec![-0.5_f64, 0.0]; // x* = -Q^{-1}c = [-0.5, 0]
        let nu: Vec<f64> = vec![];

        // Loss: L = 0.5 * ||x*||² = 0.5 * 0.25 = 0.125
        // dL/dx = x* = [-0.5, 0]
        let dl_dx = vec![-0.5_f64, 0.0];

        let grad = kkt_sensitivity(&q, &a, &x, &nu, &dl_dx).expect("KKT sensitivity failed");

        // dL/dc_i = dL/dx_j * dx*_j/dc_i = dL/dx_i * (-1/q_ii) = (-0.5) * (-1/4) = 0.125
        // So dL/dc[0] = (-0.5) * (-1/4) = 0.125  → but wait:
        // dx*/dc = -Q⁻¹, so dL/dc = (dx*/dc)ᵀ dL/dx = -Q⁻¹ dL/dx
        // = -(1/4)*[-0.5, 0] = [0.125, 0]

        // Check via FD
        let eps = 1e-5_f64;
        let solve_unconstrained = |c_vec: &[f64]| -> Vec<f64> {
            // x* = -Q^{-1} c
            c_vec
                .iter()
                .enumerate()
                .map(|(i, &ci)| -ci / q[i][i])
                .collect()
        };

        let mut c_plus = c.clone();
        c_plus[0] += eps;
        let x_plus = solve_unconstrained(&c_plus);

        let mut c_minus = c.clone();
        c_minus[0] -= eps;
        let x_minus = solve_unconstrained(&c_minus);

        let loss = |xv: &[f64]| -> f64 { xv.iter().map(|&xi| 0.5 * xi * xi).sum() };
        let fd_dc0 = (loss(&x_plus) - loss(&x_minus)) / (2.0 * eps);

        assert!(
            (grad.dl_dc[0] - fd_dc0).abs() < 1e-5,
            "KKT sensitivity dL/dc[0] = {} vs FD = {}",
            grad.dl_dc[0],
            fd_dc0
        );
    }

    #[test]
    fn test_kkt_system_struct() {
        let q = vec![vec![4.0, 0.0], vec![0.0, 4.0]];
        let a = vec![vec![1.0, 1.0]];

        let sys = KktSystem::new(&q, &a);
        assert_eq!(sys.n, 2);
        assert_eq!(sys.p, 1);
        assert_eq!(sys.matrix.len(), 3);

        // KKT system: [Q Aᵀ; A 0][x; ν] = [b1; b2]
        // For min ½ xᵀQx + cᵀx s.t. Ax=b:
        //   stationarity: Qx + c + Aᵀν = 0  →  Qx + Aᵀν = -c
        //   feasibility:  Ax = b
        // With Q=4I, c=0, A=[[1,1]], b=[1]:
        //   4x₀ + ν = 0, 4x₁ + ν = 0, x₀ + x₁ = 1
        //   From stationarity: x₀ = -ν/4, x₁ = -ν/4
        //   From feasibility: -ν/4 - ν/4 = 1 → -ν/2 = 1 → ν = -2
        //   So x₀ = x₁ = 0.5, ν = -2
        let b1 = vec![0.0, 0.0]; // -c
        let b2 = vec![1.0]; // b
        let (x, nu) = sys.solve(&b1, &b2).expect("KKT solve failed");
        assert!((x[0] - 0.5).abs() < 1e-10, "x[0] = {} (expected 0.5)", x[0]);
        assert!((x[1] - 0.5).abs() < 1e-10, "x[1] = {} (expected 0.5)", x[1]);
        assert!(
            (nu[0] - (-2.0)).abs() < 1e-10,
            "ν[0] = {} (expected -2.0)",
            nu[0]
        );
    }

    #[test]
    fn test_outer_product() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let c = outer_product(&a, &b);
        assert!((c[0][0] - 3.0).abs() < 1e-12);
        assert!((c[0][1] - 4.0).abs() < 1e-12);
        assert!((c[1][0] - 6.0).abs() < 1e-12);
        assert!((c[1][1] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_sym_outer_product() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let c = sym_outer_product(&a, &b);
        // C[i][j] = 0.5*(a[i]*b[j] + b[i]*a[j])
        assert!((c[0][0] - 3.0).abs() < 1e-12); // 0.5*(1*3 + 3*1)
        assert!((c[1][1] - 8.0).abs() < 1e-12); // 0.5*(2*4 + 4*2)
                                                // Symmetry
        assert!((c[0][1] - c[1][0]).abs() < 1e-12);
    }

    #[test]
    fn test_nlp_adjoint_unconstrained() {
        // For no constraints: adjoint solves reg*I dx_adj = dL/dx → dx_adj = dL/dx / reg
        let n = 3_usize;
        let f_grad = vec![1.0, 2.0, 3.0_f64];
        let g_jac: Vec<Vec<f64>> = vec![];
        let h_jac: Vec<Vec<f64>> = vec![];
        let x_star = vec![0.5, -0.5, 0.0_f64];
        let lambda_star: Vec<f64> = vec![];
        let nu_star: Vec<f64> = vec![];
        let dl_dx = vec![1.0, 0.0, 0.0_f64];

        let grad = parametric_nlp_adjoint(
            &f_grad,
            &g_jac,
            &h_jac,
            &x_star,
            &lambda_star,
            &nu_star,
            &dl_dx,
        )
        .expect("NLP adjoint failed");

        // With only reg*I, dx_adj = dL/dx / reg
        let reg = 1e-8_f64;
        assert!(
            (grad.dx_adj[0] - dl_dx[0] / reg).abs() < 1e-3 * (dl_dx[0] / reg).abs() + 1e-8,
            "dx_adj[0] = {} (expected ~{})",
            grad.dx_adj[0],
            dl_dx[0] / reg
        );

        // Sizes match
        assert_eq!(grad.dl_df_params.len(), n);
        assert_eq!(grad.dlambda_adj.len(), 0);
        assert_eq!(grad.dnu_adj.len(), 0);
    }

    #[test]
    fn test_regularize_q() {
        let q = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let q_reg = regularize_q(&q, 0.5);
        assert!((q_reg[0][0] - 2.5).abs() < 1e-12);
        assert!((q_reg[1][1] - 3.5).abs() < 1e-12);
        assert!((q_reg[0][1] - 1.0).abs() < 1e-12); // off-diagonal unchanged
    }

    #[test]
    fn test_mat_vec() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let x = vec![1.0, 2.0];
        let y = mat_vec(&a, &x);
        assert!((y[0] - 5.0).abs() < 1e-12);
        assert!((y[1] - 11.0).abs() < 1e-12);
    }

    // The helper is used only in dead-code test path
    #[allow(dead_code)]
    fn _use_numerical_gradient(grad_fn: impl Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
        numerical_gradient_kkt(grad_fn, x, 1e-6)
    }
}
