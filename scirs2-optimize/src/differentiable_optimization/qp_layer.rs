//! ADMM-based differentiable QP layer with warm-start and active-set backward.
//!
//! Solves the QP:
//!
//!   min  ½ xᵀQx + cᵀx
//!   s.t. A_eq x = b_eq    (equality)
//!        G_ineq x ≤ h_ineq (inequality)
//!
//! The forward pass uses an OSQP-style ADMM iteration:
//!
//!   x-update: (Q + ρ Cᵀ C)⁻¹ (ρ Cᵀ (z - u) - c)   where C = [A_eq; G_ineq]
//!   z-update: projection onto {Ax=b} × {Gx ≤ h}
//!   u-update: u += C x - z
//!
//! The backward pass uses KKT sensitivity on the active constraints.

use super::implicit_diff::identify_active_constraints;
use super::kkt_sensitivity::{kkt_sensitivity, regularize_q};
use super::types::{DiffOptGrad, DiffOptParams, DiffOptResult, DiffOptStatus};
use crate::error::{OptimizeError, OptimizeResult};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the ADMM-based QP layer.
#[derive(Debug, Clone)]
pub struct QpLayerConfig {
    /// Maximum number of ADMM iterations.
    pub max_iter: usize,
    /// Primal and dual residual tolerance for convergence.
    pub tol: f64,
    /// ADMM penalty parameter ρ.
    pub rho: f64,
    /// Tikhonov regularization on Q for numerical stability.
    pub regularization: f64,
    /// Tolerance for identifying active inequality constraints in backward pass.
    pub active_tol: f64,
    /// Whether to print convergence information.
    pub verbose: bool,
}

impl Default for QpLayerConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-8,
            rho: 1.0,
            regularization: 1e-7,
            active_tol: 1e-6,
            verbose: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cholesky factorization (simplified LDLᵀ for symmetric PD matrices)
// ─────────────────────────────────────────────────────────────────────────────

/// Cholesky decomposition: returns lower triangular L such that A = L Lᵀ.
/// Uses the standard Cholesky-Banachiewicz algorithm.
fn cholesky(a: &[Vec<f64>]) -> OptimizeResult<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0_f64; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0_f64;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }
            if i == j {
                let diag = a[i][i] - sum;
                if diag <= 0.0 {
                    return Err(OptimizeError::ComputationError(format!(
                        "Cholesky failed: non-positive diagonal at index {}. diag = {diag}",
                        i
                    )));
                }
                l[i][j] = diag.sqrt();
            } else {
                let l_jj = l[j][j];
                if l_jj.abs() < 1e-30 {
                    return Err(OptimizeError::ComputationError(
                        "Cholesky failed: zero diagonal element".to_string(),
                    ));
                }
                l[i][j] = (a[i][j] - sum) / l_jj;
            }
        }
    }
    Ok(l)
}

/// Forward substitution: solve L y = b where L is lower triangular.
fn forward_sub(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i][j] * y[j];
        }
        let diag = l[i][i];
        y[i] = if diag.abs() < 1e-30 { 0.0 } else { sum / diag };
    }
    y
}

/// Backward substitution: solve Lᵀ x = y where L is lower triangular.
fn backward_sub(l: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[j][i] * x[j];
        }
        let diag = l[i][i];
        x[i] = if diag.abs() < 1e-30 { 0.0 } else { sum / diag };
    }
    x
}

/// Solve the symmetric positive definite system Ax = b via Cholesky factorization.
/// Falls back to Gaussian elimination if Cholesky fails.
fn cholesky_solve(a: &[Vec<f64>], b: &[f64]) -> OptimizeResult<Vec<f64>> {
    match cholesky(a) {
        Ok(l) => {
            let y = forward_sub(&l, b);
            Ok(backward_sub(&l, &y))
        }
        Err(_) => {
            // Fall back to implicit_diff solver
            super::implicit_diff::solve_implicit_system(a, b)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QP layer
// ─────────────────────────────────────────────────────────────────────────────

/// An ADMM-based differentiable QP layer.
///
/// Stores problem data and the last forward-pass solution for use in
/// the backward pass.
#[derive(Debug, Clone)]
pub struct QpLayer {
    config: QpLayerConfig,
    /// Cached warm-start primal.
    warm_x: Option<Vec<f64>>,
    /// Cached warm-start z.
    warm_z: Option<Vec<f64>>,
    /// Cached warm-start u.
    warm_u: Option<Vec<f64>>,
    /// Last forward result (needed for backward).
    last_result: Option<QpForwardCache>,
}

/// Cached data from the forward pass needed for gradient computation.
#[derive(Debug, Clone)]
struct QpForwardCache {
    x: Vec<f64>,
    lambda: Vec<f64>, // inequality duals
    nu: Vec<f64>,     // equality duals
    q: Vec<Vec<f64>>,
    c: Vec<f64>,
    a_eq: Vec<Vec<f64>>,
    b_eq: Vec<f64>,
    g_ineq: Vec<Vec<f64>>,
    h_ineq: Vec<f64>,
}

impl QpLayer {
    /// Create a new QP layer with default configuration.
    pub fn new() -> Self {
        Self {
            config: QpLayerConfig::default(),
            warm_x: None,
            warm_z: None,
            warm_u: None,
            last_result: None,
        }
    }

    /// Create a new QP layer with custom configuration.
    pub fn with_config(config: QpLayerConfig) -> Self {
        Self {
            config,
            warm_x: None,
            warm_z: None,
            warm_u: None,
            last_result: None,
        }
    }

    /// Solve the QP (forward pass).
    ///
    /// Uses ADMM with warm-start. The constraint matrix C = [A_eq; G_ineq] is
    /// stacked, and z is projected onto the feasible set:
    ///
    ///   z_eq   = b_eq                     (equality: exact satisfaction)
    ///   z_ineq = min(z_ineq_raw, h_ineq)  (inequality: clamp to ≤ h)
    ///
    /// # Arguments
    /// * `q`      – n×n cost matrix (symmetric PSD).
    /// * `c`      – n linear cost vector.
    /// * `a_eq`   – p×n equality constraint matrix.
    /// * `b_eq`   – p equality rhs.
    /// * `g_ineq` – m×n inequality constraint matrix.
    /// * `h_ineq` – m inequality rhs.
    pub fn forward(
        &mut self,
        q: Vec<Vec<f64>>,
        c: Vec<f64>,
        a_eq: Vec<Vec<f64>>,
        b_eq: Vec<f64>,
        g_ineq: Vec<Vec<f64>>,
        h_ineq: Vec<f64>,
    ) -> OptimizeResult<DiffOptResult> {
        let n = c.len();
        let p = b_eq.len();
        let m = h_ineq.len();
        let nc = p + m; // total constraints

        // ── Validate dimensions ────────────────────────────────────────────
        if q.len() != n {
            return Err(OptimizeError::InvalidInput(format!(
                "Q rows ({}) != n ({})",
                q.len(),
                n
            )));
        }
        if a_eq.len() != p {
            return Err(OptimizeError::InvalidInput(format!(
                "A_eq rows ({}) != p ({})",
                a_eq.len(),
                p
            )));
        }
        if g_ineq.len() != m {
            return Err(OptimizeError::InvalidInput(format!(
                "G_ineq rows ({}) != m ({})",
                g_ineq.len(),
                m
            )));
        }

        // ── Regularize Q ───────────────────────────────────────────────────
        let q_reg = regularize_q(&q, self.config.regularization);
        let rho = self.config.rho;

        // ── Build C = [A_eq; G_ineq] (nc × n) ────────────────────────────
        let c_mat: Vec<Vec<f64>> = a_eq.iter().cloned().chain(g_ineq.iter().cloned()).collect();

        // ── Build M = Q_reg + ρ CᵀC (n×n) ────────────────────────────────
        let mut m_mat = q_reg.clone();
        for row in &c_mat {
            for i in 0..n {
                for j in 0..n {
                    let ci = if i < row.len() { row[i] } else { 0.0 };
                    let cj = if j < row.len() { row[j] } else { 0.0 };
                    m_mat[i][j] += rho * ci * cj;
                }
            }
        }

        // ── Initialise from warm-start or zero ────────────────────────────
        let mut x = self
            .warm_x
            .as_ref()
            .filter(|wx| wx.len() == n)
            .cloned()
            .unwrap_or_else(|| vec![0.0_f64; n]);

        let mut z = self
            .warm_z
            .as_ref()
            .filter(|wz| wz.len() == nc)
            .cloned()
            .unwrap_or_else(|| {
                // z_eq = b_eq, z_ineq = h_ineq / 2
                let mut z0 = Vec::with_capacity(nc);
                z0.extend_from_slice(&b_eq);
                z0.extend(h_ineq.iter().map(|&hi| hi / 2.0));
                z0
            });

        let mut u = self
            .warm_u
            .as_ref()
            .filter(|wu| wu.len() == nc)
            .cloned()
            .unwrap_or_else(|| vec![0.0_f64; nc]);

        let mut converged = false;
        let mut iterations = 0_usize;

        for iter in 0..self.config.max_iter {
            iterations = iter + 1;

            // ── x-update: solve M x_new = ρ Cᵀ(z - u) - c ──────────────
            let mut rhs_x = c.iter().map(|&ci| -ci).collect::<Vec<_>>();
            for (k, row) in c_mat.iter().enumerate() {
                let zu_k =
                    if k < z.len() { z[k] } else { 0.0 } - if k < u.len() { u[k] } else { 0.0 };
                for j in 0..n {
                    let ckj = if j < row.len() { row[j] } else { 0.0 };
                    rhs_x[j] += rho * ckj * zu_k;
                }
            }

            let x_new = cholesky_solve(&m_mat, &rhs_x)?;

            // ── z-update: project (C x_new + u) onto feasible set ────────
            let mut cx = vec![0.0_f64; nc];
            for (k, row) in c_mat.iter().enumerate() {
                for j in 0..n {
                    let ckj = if j < row.len() { row[j] } else { 0.0 };
                    cx[k] += ckj * x_new[j];
                }
            }

            let mut z_new = vec![0.0_f64; nc];
            // Equality block: project onto Ax = b → z_k = b_k
            for k in 0..p {
                z_new[k] = if k < b_eq.len() { b_eq[k] } else { 0.0 };
            }
            // Inequality block: project onto Gx ≤ h → z_k = min(cx[p+k] + u[p+k], h_k)
            for k in 0..m {
                let raw = cx[p + k] + u[p + k];
                let h_k = if k < h_ineq.len() { h_ineq[k] } else { 0.0 };
                z_new[p + k] = raw.min(h_k);
            }

            // ── u-update: u += Cx - z ─────────────────────────────────────
            let mut u_new = vec![0.0_f64; nc];
            for k in 0..nc {
                u_new[k] = u[k] + cx[k] - z_new[k];
            }

            // ── Compute residuals ─────────────────────────────────────────
            let primal_res: f64 = cx
                .iter()
                .zip(z_new.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            let dual_res: f64 = {
                // rho * Cᵀ (z - z_old)
                let mut dr = 0.0_f64;
                for k in 0..nc {
                    let dz = z_new[k] - z[k];
                    for j in 0..n {
                        let ckj = if j < c_mat[k].len() { c_mat[k][j] } else { 0.0 };
                        dr += (rho * ckj * dz).powi(2);
                    }
                }
                dr.sqrt()
            };

            if self.config.verbose {
                eprintln!(
                    "iter {}: primal_res={:.2e}, dual_res={:.2e}",
                    iter, primal_res, dual_res
                );
            }

            x = x_new;
            z = z_new;
            u = u_new;

            if primal_res < self.config.tol && dual_res < self.config.tol {
                converged = true;
                break;
            }
        }

        // ── Extract dual variables ─────────────────────────────────────────
        // In ADMM, the dual variable for the k-th constraint is ρ u[k].
        let nu: Vec<f64> = u[..p].iter().map(|&ui| rho * ui).collect();
        let lambda: Vec<f64> = u[p..].iter().map(|&ui| rho * ui.max(0.0)).collect();

        // ── Compute objective ──────────────────────────────────────────────
        let mut obj = 0.0_f64;
        for i in 0..n {
            obj += c[i] * x[i];
            for j in 0..n {
                let q_ij = if i < q.len() && j < q[i].len() {
                    q[i][j]
                } else {
                    0.0
                };
                obj += 0.5 * q_ij * x[i] * x[j];
            }
        }

        let status = if converged {
            DiffOptStatus::Optimal
        } else {
            DiffOptStatus::MaxIterations
        };

        // ── Update warm-start cache ────────────────────────────────────────
        self.warm_x = Some(x.clone());
        self.warm_z = Some(z);
        self.warm_u = Some(u);

        // ── Cache for backward ────────────────────────────────────────────
        self.last_result = Some(QpForwardCache {
            x: x.clone(),
            lambda: lambda.clone(),
            nu: nu.clone(),
            q: q.clone(),
            c: c.clone(),
            a_eq: a_eq.clone(),
            b_eq: b_eq.clone(),
            g_ineq: g_ineq.clone(),
            h_ineq: h_ineq.clone(),
        });

        Ok(DiffOptResult {
            x,
            lambda,
            nu,
            objective: obj,
            status,
            iterations,
        })
    }

    /// Backward pass: compute parameter gradients via KKT sensitivity.
    ///
    /// Uses the active-set at the solution to identify binding inequality
    /// constraints, stacks them with equality constraints, and calls
    /// `kkt_sensitivity` on the resulting system.
    ///
    /// # Arguments
    /// * `dl_dx` – upstream gradient dL/dx (length n).
    ///
    /// # Errors
    /// Returns `OptimizeError::ComputationError` if no forward pass has been
    /// run, or if the KKT system is singular.
    pub fn backward(&self, dl_dx: &[f64]) -> OptimizeResult<DiffOptGrad> {
        let cache = self.last_result.as_ref().ok_or_else(|| {
            OptimizeError::ComputationError("QpLayer::backward called before forward".to_string())
        })?;

        let n = cache.x.len();
        if dl_dx.len() != n {
            return Err(OptimizeError::InvalidInput(format!(
                "dl_dx length {} != n {}",
                dl_dx.len(),
                n
            )));
        }

        // ── Identify active inequality constraints ─────────────────────────
        let active_idx = identify_active_constraints(
            &cache.g_ineq,
            &cache.h_ineq,
            &cache.x,
            self.config.active_tol,
        );

        // Stack equality constraints and active inequality rows
        let mut a_aug: Vec<Vec<f64>> = cache.a_eq.clone();
        let mut b_aug: Vec<f64> = cache.b_eq.clone();
        let mut nu_aug: Vec<f64> = cache.nu.clone();

        for &ai in &active_idx {
            if ai < cache.g_ineq.len() {
                a_aug.push(cache.g_ineq[ai].clone());
                b_aug.push(cache.h_ineq.get(ai).copied().unwrap_or(0.0));
                nu_aug.push(cache.lambda.get(ai).copied().unwrap_or(0.0));
            }
        }

        // ── Regularize Q ──────────────────────────────────────────────────
        let q_reg = regularize_q(&cache.q, self.config.regularization);

        // ── Call KKT sensitivity on augmented equality system ─────────────
        let kkt_grad = kkt_sensitivity(&q_reg, &a_aug, &cache.x, &nu_aug, dl_dx)?;

        // Split dl_da back into dl_da_eq and dl_dg (active rows only)
        let p = cache.a_eq.len();
        let m_full = cache.g_ineq.len();

        let dl_da_eq: Option<Vec<Vec<f64>>> = if p > 0 {
            Some(kkt_grad.dl_da[..p].to_vec())
        } else {
            None
        };

        let dl_db_eq = kkt_grad.dl_db[..p].to_vec();

        // Expand active gradients to full G dimension
        let mut dl_dg = vec![vec![0.0_f64; n]; m_full];
        let mut dl_dh = vec![0.0_f64; m_full];
        for (idx, &ai) in active_idx.iter().enumerate() {
            let aug_idx = p + idx;
            if ai < m_full && aug_idx < kkt_grad.dl_da.len() {
                dl_dg[ai] = kkt_grad.dl_da[aug_idx].clone();
                dl_dh[ai] = kkt_grad.dl_db.get(aug_idx).copied().unwrap_or(0.0);
            }
        }

        Ok(DiffOptGrad {
            dl_dq: Some(kkt_grad.dl_dq),
            dl_dc: kkt_grad.dl_dc,
            dl_da: dl_da_eq,
            dl_db: dl_db_eq,
            dl_dg: Some(dl_dg),
            dl_dh,
        })
    }

    /// Access the cached solution from the last forward pass.
    pub fn last_solution(&self) -> Option<&[f64]> {
        self.last_result.as_ref().map(|r| r.x.as_slice())
    }

    /// Reset warm-start cache.
    pub fn reset_warm_start(&mut self) {
        self.warm_x = None;
        self.warm_z = None;
        self.warm_u = None;
    }
}

impl Default for QpLayer {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_identity_qp(n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let q = (0..n)
            .map(|i| {
                let mut row = vec![0.0_f64; n];
                row[i] = 2.0; // 2I so x* = -Q^{-1}c = -0.5 c
                row
            })
            .collect();
        let c = vec![0.0_f64; n];
        (q, c)
    }

    #[test]
    fn test_qp_layer_config_default() {
        let cfg = QpLayerConfig::default();
        assert_eq!(cfg.max_iter, 100);
        assert!((cfg.tol - 1e-8).abs() < 1e-15);
        assert!(!cfg.verbose);
        assert!((cfg.rho - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_qp_layer_identity_q_zero_c() {
        // min ½||x||² s.t. x[0] + x[1] = 0
        // x* = [0, 0] with equality b=0
        let mut layer = QpLayer::new();
        let (q, c) = make_identity_qp(2);
        let a_eq = vec![vec![1.0, 1.0]];
        let b_eq = vec![0.0];

        let result = layer
            .forward(q, c, a_eq, b_eq, vec![], vec![])
            .expect("Forward failed");

        assert!(
            result.x[0].abs() < 1e-4,
            "x[0] = {} (expected 0)",
            result.x[0]
        );
        assert!(
            result.x[1].abs() < 1e-4,
            "x[1] = {} (expected 0)",
            result.x[1]
        );
    }

    #[test]
    fn test_qp_layer_forward_unconstrained() {
        // min x^2 + y^2 + x + 2y → x* = [-0.5, -1.0]
        let mut layer = QpLayer::new();
        let q = vec![vec![2.0, 0.0], vec![0.0, 2.0]];
        let c = vec![1.0, 2.0];

        let result = layer
            .forward(q, c, vec![], vec![], vec![], vec![])
            .expect("Forward failed");

        assert!(
            (result.x[0] - (-0.5)).abs() < 1e-3,
            "x[0] = {} (expected -0.5)",
            result.x[0]
        );
        assert!(
            (result.x[1] - (-1.0)).abs() < 1e-3,
            "x[1] = {} (expected -1.0)",
            result.x[1]
        );
    }

    #[test]
    fn test_qp_layer_forward_with_equality() {
        // min x^2 + y^2 s.t. x + y = 1 → x* = [0.5, 0.5]
        let mut layer = QpLayer::new();
        let q = vec![vec![2.0, 0.0], vec![0.0, 2.0]];
        let c = vec![0.0, 0.0];
        let a_eq = vec![vec![1.0, 1.0]];
        let b_eq = vec![1.0];

        let result = layer
            .forward(q, c, a_eq, b_eq, vec![], vec![])
            .expect("Forward failed");

        assert!(
            (result.x[0] - 0.5).abs() < 1e-3,
            "x[0] = {} (expected 0.5)",
            result.x[0]
        );
        assert!(
            (result.x[1] - 0.5).abs() < 1e-3,
            "x[1] = {} (expected 0.5)",
            result.x[1]
        );
    }

    #[test]
    fn test_qp_layer_forward_with_inequality() {
        // min x^2 + y^2 s.t. -x - y <= -1 (i.e. x + y >= 1) → x* = [0.5, 0.5]
        let mut layer = QpLayer::new();
        let q = vec![vec![2.0, 0.0], vec![0.0, 2.0]];
        let c = vec![0.0, 0.0];
        let g = vec![vec![-1.0, -1.0]];
        let h = vec![-1.0];

        let result = layer
            .forward(q, c, vec![], vec![], g, h)
            .expect("Forward failed");

        // x + y should be >= 1
        let sum = result.x[0] + result.x[1];
        assert!(sum >= 1.0 - 1e-3, "x + y = {} (should be >= 1)", sum);
    }

    #[test]
    fn test_qp_layer_backward_no_forward_error() {
        let layer = QpLayer::new();
        let result = layer.backward(&[1.0, 0.0]);
        assert!(result.is_err(), "Should error without forward pass");
    }

    #[test]
    fn test_qp_layer_backward_dl_dc_finite() {
        let mut layer = QpLayer::new();
        let q = vec![vec![2.0, 0.0], vec![0.0, 2.0]];
        let c = vec![1.0, 2.0];

        let result = layer
            .forward(q, c, vec![], vec![], vec![], vec![])
            .expect("Forward failed");
        let _ = result;

        let grad = layer.backward(&[1.0, 0.0]).expect("Backward failed");
        assert_eq!(grad.dl_dc.len(), 2);
        assert!(grad.dl_dc[0].is_finite(), "dl/dc[0] not finite");
        assert!(grad.dl_dc[1].is_finite(), "dl/dc[1] not finite");
    }

    #[test]
    fn test_qp_layer_backward_gradient_check() {
        // Verify dl/dc via finite differences
        // min x^2 + y^2 + c[0]*x + c[1]*y (unconstrained)
        // x* = [-c[0]/2, -c[1]/2]
        // Loss L = 0.5 * ||x*||^2 = 0.5*(c[0]^2/4 + c[1]^2/4)
        // dL/dc[0] = c[0]/4 = 0.25 for c=[1,0]

        let eps = 1e-5_f64;
        let c_base = vec![1.0_f64, 0.0];
        let q = vec![vec![2.0, 0.0], vec![0.0, 2.0]];

        let solve_and_loss = |c_vec: Vec<f64>| -> f64 {
            let mut layer = QpLayer::new();
            let res = layer
                .forward(q.clone(), c_vec, vec![], vec![], vec![], vec![])
                .expect("Forward failed");
            res.x.iter().map(|&xi| 0.5 * xi * xi).sum::<f64>()
        };

        // Forward + backward for analytical gradient
        let mut layer = QpLayer::new();
        let res = layer
            .forward(q.clone(), c_base.clone(), vec![], vec![], vec![], vec![])
            .expect("Forward failed");
        let dl_dx = res.x.clone(); // dL/dx = x* for L = 0.5 ||x*||^2
        let grad = layer.backward(&dl_dx).expect("Backward failed");

        // Finite difference for dc[0]
        let mut c_plus = c_base.clone();
        c_plus[0] += eps;
        let mut c_minus = c_base.clone();
        c_minus[0] -= eps;
        let fd_dc0 = (solve_and_loss(c_plus) - solve_and_loss(c_minus)) / (2.0 * eps);

        assert!(
            (grad.dl_dc[0] - fd_dc0).abs() < 1e-3,
            "dl/dc[0] analytical={} vs FD={}",
            grad.dl_dc[0],
            fd_dc0
        );
    }

    #[test]
    fn test_qp_layer_active_set_identification() {
        // min x^2 + y^2 s.t. x >= 0, y >= 0, x+y >= 0.5
        // At x* = [0.25, 0.25], x+y=0.5 is active, x>=0 and y>=0 are inactive
        let mut layer = QpLayer::new();
        let q = vec![vec![2.0, 0.0], vec![0.0, 2.0]];
        let c = vec![0.0, 0.0];
        let g = vec![
            vec![-1.0, 0.0],  // -x <= 0
            vec![0.0, -1.0],  // -y <= 0
            vec![-1.0, -1.0], // -x - y <= -0.5
        ];
        let h = vec![0.0, 0.0, -0.5];

        let result = layer
            .forward(q, c, vec![], vec![], g, h)
            .expect("Forward failed");

        // x + y should be >= 0.5
        let sum = result.x[0] + result.x[1];
        assert!(sum >= 0.5 - 1e-3, "x + y = {} (should be >= 0.5)", sum);
    }

    #[test]
    fn test_qp_layer_warm_start() {
        // Two consecutive solves with same problem — should warm start
        let mut layer = QpLayer::new();
        let q = vec![vec![2.0, 0.0], vec![0.0, 2.0]];
        let c = vec![1.0, 1.0];

        let res1 = layer
            .forward(q.clone(), c.clone(), vec![], vec![], vec![], vec![])
            .expect("Forward 1 failed");

        let res2 = layer
            .forward(q, c, vec![], vec![], vec![], vec![])
            .expect("Forward 2 failed");

        // Both should give same result
        assert!(
            (res1.x[0] - res2.x[0]).abs() < 1e-6,
            "Warm-start inconsistency"
        );
    }

    #[test]
    fn test_qp_layer_last_solution() {
        let mut layer = QpLayer::new();
        let q = vec![vec![2.0, 0.0], vec![0.0, 2.0]];
        let c = vec![1.0, 0.0];

        assert!(layer.last_solution().is_none());
        layer
            .forward(q, c, vec![], vec![], vec![], vec![])
            .expect("Forward failed");
        assert!(layer.last_solution().is_some());
    }

    #[test]
    fn test_cholesky_solve_identity() {
        let a = vec![vec![4.0, 0.0], vec![0.0, 9.0]];
        let b = vec![8.0, 18.0];
        let x = cholesky_solve(&a, &b).expect("Cholesky solve failed");
        assert!((x[0] - 2.0).abs() < 1e-10, "x[0] = {}", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-10, "x[1] = {}", x[1]);
    }
}
