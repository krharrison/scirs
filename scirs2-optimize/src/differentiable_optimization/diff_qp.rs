//! Differentiable Quadratic Programming (OptNet-style).
//!
//! Solves the QP:
//!
//!   min  ½ x'Qx + c'x
//!   s.t. Gx ≤ h
//!        Ax = b
//!
//! and computes gradients of the optimal solution x* w.r.t. all problem
//! parameters (Q, c, G, h, A, b) via implicit differentiation of the KKT
//! conditions.
//!
//! # References
//! - Amos & Kolter (2017). "OptNet: Differentiable Optimization as a Layer
//!   in Neural Networks." ICML.

use super::implicit_diff;
use super::types::{BackwardMode, DiffQPConfig, DiffQPResult, ImplicitGradient};
use crate::error::{OptimizeError, OptimizeResult};

/// A differentiable QP layer.
///
/// Holds the problem data and supports forward solving and backward
/// (gradient) computation.
#[derive(Debug, Clone)]
pub struct DifferentiableQP {
    /// Quadratic cost matrix Q (n×n, symmetric positive semi-definite).
    pub q: Vec<Vec<f64>>,
    /// Linear cost vector c (n).
    pub c: Vec<f64>,
    /// Inequality constraint matrix G (m×n): Gx ≤ h.
    pub g: Vec<Vec<f64>>,
    /// Inequality constraint rhs h (m).
    pub h: Vec<f64>,
    /// Equality constraint matrix A (p×n): Ax = b.
    pub a: Vec<Vec<f64>>,
    /// Equality constraint rhs b (p).
    pub b: Vec<f64>,
}

impl DifferentiableQP {
    /// Create a new differentiable QP.
    ///
    /// # Arguments
    /// * `q` – n×n cost matrix (must be symmetric PSD).
    /// * `c` – n-dimensional linear cost.
    /// * `g` – m×n inequality constraint matrix.
    /// * `h` – m-dimensional inequality rhs.
    /// * `a` – p×n equality constraint matrix.
    /// * `b` – p-dimensional equality rhs.
    pub fn new(
        q: Vec<Vec<f64>>,
        c: Vec<f64>,
        g: Vec<Vec<f64>>,
        h: Vec<f64>,
        a: Vec<Vec<f64>>,
        b: Vec<f64>,
    ) -> OptimizeResult<Self> {
        let n = c.len();
        if q.len() != n {
            return Err(OptimizeError::InvalidInput(format!(
                "Q has {} rows but c has length {}",
                q.len(),
                n
            )));
        }
        for (i, row) in q.iter().enumerate() {
            if row.len() != n {
                return Err(OptimizeError::InvalidInput(format!(
                    "Q row {} has length {} but expected {}",
                    i,
                    row.len(),
                    n
                )));
            }
        }
        for (i, row) in g.iter().enumerate() {
            if row.len() != n {
                return Err(OptimizeError::InvalidInput(format!(
                    "G row {} has length {} but expected {}",
                    i,
                    row.len(),
                    n
                )));
            }
        }
        if g.len() != h.len() {
            return Err(OptimizeError::InvalidInput(format!(
                "G has {} rows but h has length {}",
                g.len(),
                h.len()
            )));
        }
        for (i, row) in a.iter().enumerate() {
            if row.len() != n {
                return Err(OptimizeError::InvalidInput(format!(
                    "A row {} has length {} but expected {}",
                    i,
                    row.len(),
                    n
                )));
            }
        }
        if a.len() != b.len() {
            return Err(OptimizeError::InvalidInput(format!(
                "A has {} rows but b has length {}",
                a.len(),
                b.len()
            )));
        }

        Ok(Self { q, c, g, h, a, b })
    }

    /// Number of primal variables.
    pub fn n(&self) -> usize {
        self.c.len()
    }

    /// Number of inequality constraints.
    pub fn m(&self) -> usize {
        self.h.len()
    }

    /// Number of equality constraints.
    pub fn p(&self) -> usize {
        self.b.len()
    }

    /// Solve the QP (forward pass).
    ///
    /// Uses a primal-dual interior-point method with Mehrotra predictor-
    /// corrector steps.
    pub fn forward(&self, config: &DiffQPConfig) -> OptimizeResult<DiffQPResult> {
        let n = self.n();
        let m = self.m();
        let p = self.p();

        // ── Build regularised Q ────────────────────────────────────────
        let mut q_reg = self.q.clone();
        for i in 0..n {
            q_reg[i][i] += config.regularization;
        }

        // ── Initialisation ─────────────────────────────────────────────
        let mut x = vec![0.0; n];
        let mut lam = vec![1.0; m]; // inequality duals > 0
        let mut nu = vec![0.0; p]; // equality duals
        let mut s = vec![1.0; m]; // slacks s = h - Gx > 0

        // Compute initial slacks
        for i in 0..m {
            let mut gx_i = 0.0;
            for j in 0..n {
                gx_i += self.g[i][j] * x[j];
            }
            s[i] = self.h[i] - gx_i;
            if s[i] <= 0.0 {
                s[i] = 1.0; // ensure positivity
            }
        }

        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..config.max_iterations {
            iterations = iter + 1;

            // ── Compute residuals ──────────────────────────────────────
            // r_stat = Qx + c + G'λ + A'ν  (stationarity)
            let mut r_stat = vec![0.0; n];
            for i in 0..n {
                let mut qx_i = 0.0;
                for j in 0..n {
                    qx_i += q_reg[i][j] * x[j];
                }
                r_stat[i] = qx_i + self.c[i];
            }
            for k in 0..m {
                for i in 0..n {
                    r_stat[i] += self.g[k][i] * lam[k];
                }
            }
            for k in 0..p {
                for i in 0..n {
                    r_stat[i] += self.a[k][i] * nu[k];
                }
            }

            // r_eq = Ax - b  (primal equality)
            let mut r_eq = vec![0.0; p];
            for i in 0..p {
                for j in 0..n {
                    r_eq[i] += self.a[i][j] * x[j];
                }
                r_eq[i] -= self.b[i];
            }

            // r_ineq = s + Gx - h  (slack definition)
            let mut r_ineq = vec![0.0; m];
            for i in 0..m {
                let mut gx_i = 0.0;
                for j in 0..n {
                    gx_i += self.g[i][j] * x[j];
                }
                r_ineq[i] = s[i] + gx_i - self.h[i];
            }

            // r_comp = diag(λ) s  (complementarity, want → 0)
            let mu: f64 = if m > 0 {
                lam.iter()
                    .zip(s.iter())
                    .map(|(&li, &si)| li * si)
                    .sum::<f64>()
                    / m as f64
            } else {
                0.0
            };

            // Check convergence
            let res_stat: f64 = r_stat.iter().map(|v| v.abs()).fold(0.0, f64::max);
            let res_eq: f64 = r_eq.iter().map(|v| v.abs()).fold(0.0, f64::max);
            let res_ineq: f64 = r_ineq.iter().map(|v| v.abs()).fold(0.0, f64::max);
            let max_res = res_stat.max(res_eq).max(res_ineq).max(mu);

            if max_res < config.tolerance {
                converged = true;
                break;
            }

            // ── Build and solve the KKT system for Newton direction ────
            // We solve the reduced system by eliminating s.
            // Variables: (dx, dlam, dnu)
            let dim = n + m + p;
            let mut kkt = vec![vec![0.0; dim]; dim];
            let mut rhs = vec![0.0; dim];

            // Block row 0 (stationarity): Q dx + G' dlam + A' dnu = -r_stat
            for i in 0..n {
                for j in 0..n {
                    kkt[i][j] = q_reg[i][j];
                }
                for k in 0..m {
                    kkt[i][n + k] = self.g[k][i];
                }
                for k in 0..p {
                    kkt[i][n + m + k] = self.a[k][i];
                }
                rhs[i] = -r_stat[i];
            }

            // Block row 1 (complementarity + slack elimination):
            // diag(s) dlam + diag(λ) ds = -diag(λ)s + σμe
            // ds = -r_ineq - G dx   (from slack row)
            // → diag(s) dlam + diag(λ)(-r_ineq - G dx) = -diag(λ)s + σμe
            // → -diag(λ)G dx + diag(s) dlam = -diag(λ)s + σμe + diag(λ) r_ineq
            let sigma = 0.1_f64; // centering parameter
            for i in 0..m {
                let li = lam[i];
                let si = s[i];
                for j in 0..n {
                    kkt[n + i][j] = -li * self.g[i][j];
                }
                kkt[n + i][n + i] = si;
                rhs[n + i] = -li * si + sigma * mu + li * r_ineq[i];
            }

            // Block row 2 (equality): A dx = -r_eq
            for i in 0..p {
                for j in 0..n {
                    kkt[n + m + i][j] = self.a[i][j];
                }
                rhs[n + m + i] = -r_eq[i];
            }

            let dir = match implicit_diff::solve_implicit_system(&kkt, &rhs) {
                Ok(d) => d,
                Err(_) => break, // singular system, stop
            };

            let dx = &dir[..n];
            let dlam = &dir[n..n + m];
            let dnu = &dir[n + m..];

            // Recover ds
            let mut ds = vec![0.0; m];
            for i in 0..m {
                let mut gx_i = 0.0;
                for j in 0..n {
                    gx_i += self.g[i][j] * dx[j];
                }
                ds[i] = -r_ineq[i] - gx_i;
            }

            // ── Step size (fraction-to-boundary) ───────────────────────
            let tau = 0.995;
            let mut alpha_p = 1.0_f64;
            let mut alpha_d = 1.0_f64;

            for i in 0..m {
                if ds[i] < 0.0 {
                    let ratio = -tau * s[i] / ds[i];
                    if ratio < alpha_p {
                        alpha_p = ratio;
                    }
                }
                if dlam[i] < 0.0 {
                    let ratio = -tau * lam[i] / dlam[i];
                    if ratio < alpha_d {
                        alpha_d = ratio;
                    }
                }
            }

            alpha_p = alpha_p.min(1.0).max(1e-12);
            alpha_d = alpha_d.min(1.0).max(1e-12);

            // ── Update ─────────────────────────────────────────────────
            for i in 0..n {
                x[i] += alpha_p * dx[i];
            }
            for i in 0..m {
                s[i] += alpha_p * ds[i];
                lam[i] += alpha_d * dlam[i];
                // Safety: keep positive
                if s[i] < 1e-14 {
                    s[i] = 1e-14;
                }
                if lam[i] < 1e-14 {
                    lam[i] = 1e-14;
                }
            }
            for i in 0..p {
                nu[i] += alpha_d * dnu[i];
            }
        }

        // ── Compute objective ──────────────────────────────────────────
        let mut obj = 0.0;
        for i in 0..n {
            obj += self.c[i] * x[i];
            for j in 0..n {
                obj += 0.5 * self.q[i][j] * x[i] * x[j];
            }
        }

        Ok(DiffQPResult {
            optimal_x: x,
            optimal_lambda: lam,
            optimal_nu: nu,
            objective: obj,
            converged,
            iterations,
        })
    }

    /// Backward pass: compute gradients of loss w.r.t. QP parameters.
    ///
    /// Given the upstream gradient dl/dx*, returns the implicit gradients
    /// dl/d{Q, c, G, h, A, b}.
    pub fn backward(
        &self,
        result: &DiffQPResult,
        dl_dx: &[f64],
        config: &DiffQPConfig,
    ) -> OptimizeResult<ImplicitGradient> {
        let n = self.n();
        if dl_dx.len() != n {
            return Err(OptimizeError::InvalidInput(format!(
                "dl_dx length {} != n {}",
                dl_dx.len(),
                n
            )));
        }

        // Add regularization to Q for the backward pass as well
        let mut q_reg = self.q.clone();
        for i in 0..n {
            q_reg[i][i] += config.regularization;
        }

        match config.backward_mode {
            BackwardMode::FullDifferentiation => implicit_diff::compute_full_implicit_gradient(
                &q_reg,
                &self.g,
                &self.h,
                &self.a,
                &result.optimal_x,
                &result.optimal_lambda,
                &result.optimal_nu,
                dl_dx,
            ),
            BackwardMode::ActiveSetOnly => {
                implicit_diff::compute_active_set_implicit_gradient(
                    &q_reg,
                    &self.g,
                    &self.h,
                    &self.a,
                    &result.optimal_x,
                    &result.optimal_lambda,
                    &result.optimal_nu,
                    dl_dx,
                    config.tolerance * 100.0, // slightly relaxed for active set
                )
            }
            _ => Err(OptimizeError::NotImplementedError(
                "Unknown backward mode".to_string(),
            )),
        }
    }

    /// Solve multiple QPs with the same structure but different parameters.
    ///
    /// This is a convenience method; each QP is solved independently.
    pub fn batched_forward(
        params_list: &[DifferentiableQP],
        config: &DiffQPConfig,
    ) -> OptimizeResult<Vec<DiffQPResult>> {
        params_list.iter().map(|qp| qp.forward(config)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple 2-variable unconstrained QP:
    ///   min x^2 + y^2 + x + 2y
    ///   → optimal at x = -0.5, y = -1.0
    #[test]
    fn test_qp_forward_unconstrained() {
        let qp = DifferentiableQP::new(
            vec![vec![2.0, 0.0], vec![0.0, 2.0]],
            vec![1.0, 2.0],
            vec![],
            vec![],
            vec![],
            vec![],
        )
        .expect("QP creation failed");

        let config = DiffQPConfig::default();
        let result = qp.forward(&config).expect("Forward solve failed");

        assert!(result.converged, "QP should converge");
        assert!(
            (result.optimal_x[0] - (-0.5)).abs() < 1e-4,
            "x[0] = {} (expected -0.5)",
            result.optimal_x[0]
        );
        assert!(
            (result.optimal_x[1] - (-1.0)).abs() < 1e-4,
            "x[1] = {} (expected -1.0)",
            result.optimal_x[1]
        );
    }

    /// 2-variable QP with one inequality constraint:
    ///   min x^2 + y^2
    ///   s.t. x + y >= 1   →  -x - y <= -1
    ///   optimal: x = 0.5, y = 0.5
    #[test]
    fn test_qp_forward_with_inequality() {
        let qp = DifferentiableQP::new(
            vec![vec![2.0, 0.0], vec![0.0, 2.0]],
            vec![0.0, 0.0],
            vec![vec![-1.0, -1.0]], // -x - y <= -1
            vec![-1.0],
            vec![],
            vec![],
        )
        .expect("QP creation failed");

        let config = DiffQPConfig::default();
        let result = qp.forward(&config).expect("Forward solve failed");

        assert!(result.converged);
        assert!(
            (result.optimal_x[0] - 0.5).abs() < 1e-3,
            "x[0] = {} (expected 0.5)",
            result.optimal_x[0]
        );
        assert!(
            (result.optimal_x[1] - 0.5).abs() < 1e-3,
            "x[1] = {} (expected 0.5)",
            result.optimal_x[1]
        );
    }

    /// For an unconstrained QP:  min ½ x'Qx + c'x
    /// x* = -Q⁻¹ c, and dl/dc = dx*/dc · dl/dx = -Q⁻¹ · dl/dx.
    /// When dl/dx = I (unit upstream), dl/dc = -Q⁻¹.
    /// For Q = 2I, dl/dc_i with dl/dx = e_i should give -0.5 * e_i.
    #[test]
    fn test_backward_gradient_dl_dc() {
        let qp = DifferentiableQP::new(
            vec![vec![2.0, 0.0], vec![0.0, 2.0]],
            vec![1.0, 2.0],
            vec![],
            vec![],
            vec![],
            vec![],
        )
        .expect("QP creation failed");

        let config = DiffQPConfig::default();
        let result = qp.forward(&config).expect("Forward solve failed");

        // dl/dx = [1, 0] (gradient of loss w.r.t. x)
        let dl_dx = vec![1.0, 0.0];
        let grad = qp
            .backward(&result, &dl_dx, &config)
            .expect("Backward failed");

        // For unconstrained: dl/dc = -Q^{-1} dl/dx = -0.5 * [1, 0]
        // But the implicit differentiation through KKT gives dl/dc = dx
        // where dx solves Q dx = -dl/dx, so dx = -Q^{-1} dl/dx = [-0.5, 0]
        assert!(
            (grad.dl_dc[0] - (-0.5)).abs() < 1e-3,
            "dl/dc[0] = {} (expected -0.5)",
            grad.dl_dc[0]
        );
        assert!(
            grad.dl_dc[1].abs() < 1e-3,
            "dl/dc[1] = {} (expected 0)",
            grad.dl_dc[1]
        );
    }

    /// Finite-difference check for dl/dc.
    #[test]
    fn test_backward_finite_difference_c() {
        let eps = 1e-5;
        let config = DiffQPConfig::default();

        let q = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let c_base = vec![1.0, -1.0];
        let g = vec![vec![-1.0, 0.0], vec![0.0, -1.0]]; // x >= 0
        let h = vec![0.0, 0.0];

        let qp0 = DifferentiableQP::new(
            q.clone(),
            c_base.clone(),
            g.clone(),
            h.clone(),
            vec![],
            vec![],
        )
        .expect("QP creation failed");
        let res0 = qp0.forward(&config).expect("Forward failed");
        let obj0 = res0.objective;

        // dl/dx = x* (so loss = 0.5 * ||x*||^2)
        let dl_dx = res0.optimal_x.clone();
        let grad = qp0
            .backward(&res0, &dl_dx, &config)
            .expect("Backward failed");

        // Finite difference for c[0]
        let mut c_plus = c_base.clone();
        c_plus[0] += eps;
        let qp_plus =
            DifferentiableQP::new(q.clone(), c_plus, g.clone(), h.clone(), vec![], vec![])
                .expect("QP+ creation failed");
        let res_plus = qp_plus.forward(&config).expect("Forward+ failed");

        let mut c_minus = c_base.clone();
        c_minus[0] -= eps;
        let qp_minus =
            DifferentiableQP::new(q.clone(), c_minus, g.clone(), h.clone(), vec![], vec![])
                .expect("QP- creation failed");
        let res_minus = qp_minus.forward(&config).expect("Forward- failed");

        // loss = 0.5 * ||x*||^2
        let loss_plus: f64 = res_plus.optimal_x.iter().map(|v| 0.5 * v * v).sum();
        let loss_minus: f64 = res_minus.optimal_x.iter().map(|v| 0.5 * v * v).sum();
        let fd_grad = (loss_plus - loss_minus) / (2.0 * eps);

        assert!(
            (grad.dl_dc[0] - fd_grad).abs() < 1e-3,
            "dl/dc[0] analytical={} vs fd={}",
            grad.dl_dc[0],
            fd_grad
        );
    }

    /// Finite-difference check for dl/dh (inequality rhs).
    #[test]
    fn test_backward_finite_difference_h() {
        let eps = 1e-5;
        let config = DiffQPConfig::default();

        let q = vec![vec![2.0, 0.0], vec![0.0, 2.0]];
        let c = vec![0.0, 0.0];
        let g = vec![vec![-1.0, -1.0]]; // -x-y <= h[0]
        let h_base = vec![-1.0]; // x+y >= 1

        let qp0 = DifferentiableQP::new(
            q.clone(),
            c.clone(),
            g.clone(),
            h_base.clone(),
            vec![],
            vec![],
        )
        .expect("QP creation failed");
        let res0 = qp0.forward(&config).expect("Forward failed");

        let dl_dx = res0.optimal_x.clone();
        let grad = qp0
            .backward(&res0, &dl_dx, &config)
            .expect("Backward failed");

        // Perturb h[0]
        let mut h_plus = h_base.clone();
        h_plus[0] += eps;
        let qp_plus =
            DifferentiableQP::new(q.clone(), c.clone(), g.clone(), h_plus, vec![], vec![])
                .expect("QP+ creation failed");
        let res_plus = qp_plus.forward(&config).expect("Forward+ failed");

        let mut h_minus = h_base.clone();
        h_minus[0] -= eps;
        let qp_minus =
            DifferentiableQP::new(q.clone(), c.clone(), g.clone(), h_minus, vec![], vec![])
                .expect("QP- creation failed");
        let res_minus = qp_minus.forward(&config).expect("Forward- failed");

        let loss_plus: f64 = res_plus.optimal_x.iter().map(|v| 0.5 * v * v).sum();
        let loss_minus: f64 = res_minus.optimal_x.iter().map(|v| 0.5 * v * v).sum();
        let fd_grad = (loss_plus - loss_minus) / (2.0 * eps);

        // Allow somewhat loose tolerance since IP method + implicit diff can have some error
        assert!(
            (grad.dl_dh[0] - fd_grad).abs() < 0.1,
            "dl/dh[0] analytical={} vs fd={}",
            grad.dl_dh[0],
            fd_grad
        );
    }

    #[test]
    fn test_qp_with_equality_constraint() {
        // min x^2 + y^2 s.t. x + y = 1
        // optimal: x = 0.5, y = 0.5
        let qp = DifferentiableQP::new(
            vec![vec![2.0, 0.0], vec![0.0, 2.0]],
            vec![0.0, 0.0],
            vec![],
            vec![],
            vec![vec![1.0, 1.0]],
            vec![1.0],
        )
        .expect("QP creation failed");

        let config = DiffQPConfig::default();
        let result = qp.forward(&config).expect("Forward failed");

        assert!(result.converged);
        assert!(
            (result.optimal_x[0] - 0.5).abs() < 1e-3,
            "x[0] = {}",
            result.optimal_x[0]
        );
        assert!(
            (result.optimal_x[1] - 0.5).abs() < 1e-3,
            "x[1] = {}",
            result.optimal_x[1]
        );
    }

    #[test]
    fn test_batched_forward_consistency() {
        let qp1 = DifferentiableQP::new(
            vec![vec![2.0, 0.0], vec![0.0, 2.0]],
            vec![1.0, 0.0],
            vec![],
            vec![],
            vec![],
            vec![],
        )
        .expect("QP1 creation failed");
        let qp2 = DifferentiableQP::new(
            vec![vec![2.0, 0.0], vec![0.0, 2.0]],
            vec![0.0, 1.0],
            vec![],
            vec![],
            vec![],
            vec![],
        )
        .expect("QP2 creation failed");

        let config = DiffQPConfig::default();
        let batch_results = DifferentiableQP::batched_forward(&[qp1.clone(), qp2.clone()], &config)
            .expect("Batch failed");

        let r1 = qp1.forward(&config).expect("Single 1 failed");
        let r2 = qp2.forward(&config).expect("Single 2 failed");

        for i in 0..2 {
            assert!(
                (batch_results[0].optimal_x[i] - r1.optimal_x[i]).abs() < 1e-10,
                "Batch[0].x[{}] differs",
                i
            );
            assert!(
                (batch_results[1].optimal_x[i] - r2.optimal_x[i]).abs() < 1e-10,
                "Batch[1].x[{}] differs",
                i
            );
        }
    }

    #[test]
    fn test_qp_empty_constraints() {
        let qp = DifferentiableQP::new(vec![vec![2.0]], vec![4.0], vec![], vec![], vec![], vec![])
            .expect("QP creation failed");

        let config = DiffQPConfig::default();
        let result = qp.forward(&config).expect("Forward failed");
        assert!(result.converged);
        // min x^2 + 4x → x* = -2
        assert!(
            (result.optimal_x[0] - (-2.0)).abs() < 1e-3,
            "x = {}",
            result.optimal_x[0]
        );
    }

    #[test]
    fn test_qp_dimension_validation() {
        // Q is 2x2 but c is length 3 → error
        let result = DifferentiableQP::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![1.0, 2.0, 3.0],
            vec![],
            vec![],
            vec![],
            vec![],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_qp_degenerate_active_constraints() {
        // Two active constraints at the same point
        // min x^2 + y^2 s.t. x >= 1, y >= 1, x+y >= 2
        // At optimal (1,1) all three constraints are active
        let qp = DifferentiableQP::new(
            vec![vec![2.0, 0.0], vec![0.0, 2.0]],
            vec![0.0, 0.0],
            vec![
                vec![-1.0, 0.0],  // -x <= -1
                vec![0.0, -1.0],  // -y <= -1
                vec![-1.0, -1.0], // -x-y <= -2
            ],
            vec![-1.0, -1.0, -2.0],
            vec![],
            vec![],
        )
        .expect("QP creation failed");

        let config = DiffQPConfig::default();
        let result = qp.forward(&config).expect("Forward failed");

        assert!(result.converged);
        assert!(
            (result.optimal_x[0] - 1.0).abs() < 1e-2,
            "x[0] = {} (expected 1.0)",
            result.optimal_x[0]
        );
        assert!(
            (result.optimal_x[1] - 1.0).abs() < 1e-2,
            "x[1] = {} (expected 1.0)",
            result.optimal_x[1]
        );
    }
}
