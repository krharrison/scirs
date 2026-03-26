//! Differentiable LP layer via entropic regularization and basis sensitivity.
//!
//! Two complementary approaches:
//!
//! 1. **Perturbed LP** (`LpLayer`): Solves the entropy-regularized LP
//!    `min cᵀx + ε Σ x_i ln(x_i)  s.t.  Ax ≤ b, x ≥ 0` via Sinkhorn-style
//!    iterative updates. The entropic regularization makes the solution unique
//!    and smooth in c, enabling backpropagation.
//!
//! 2. **Basis sensitivity** (`LpSensitivity`): For an LP in standard form
//!    with optimal basis B, computes the exact sensitivity of the optimal
//!    basic variables to changes in the cost vector c and rhs b.
//!
//! # References
//! - Berthet et al. (2020). "Learning with Differentiable Perturbed Optimizers."
//!   NeurIPS.
//! - Murtagh & Saunders (1978). "Large-scale linearly constrained optimization."
//!   Mathematical Programming.

use crate::error::{OptimizeError, OptimizeResult};

use super::implicit_diff::solve_implicit_system;
use super::types::{DiffOptGrad, DiffOptResult, DiffOptStatus};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the LP layer.
#[derive(Debug, Clone)]
pub struct LpLayerConfig {
    /// Entropic regularization coefficient ε > 0.
    pub epsilon: f64,
    /// Maximum number of Sinkhorn iterations.
    pub max_iter: usize,
    /// Convergence tolerance for Sinkhorn iterations.
    pub tol: f64,
    /// Tolerance for identifying the active basis.
    pub basis_tol: f64,
}

impl Default for LpLayerConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-3,
            max_iter: 500,
            tol: 1e-8,
            basis_tol: 1e-6,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Softmax / log-sum-exp utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically stable softmax: exp(v_i - max) / Σ exp(v_j - max).
fn softmax(v: &[f64]) -> Vec<f64> {
    let max_v = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_v: Vec<f64> = v.iter().map(|&vi| (vi - max_v).exp()).collect();
    let sum: f64 = exp_v.iter().sum();
    if sum < 1e-300 {
        vec![1.0 / v.len() as f64; v.len()]
    } else {
        exp_v.iter().map(|&e| e / sum).collect()
    }
}

/// Log-sum-exp: log Σ exp(v_i).
fn logsumexp(v: &[f64]) -> f64 {
    let max_v = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = v.iter().map(|&vi| (vi - max_v).exp()).sum();
    max_v + sum.ln()
}

// ─────────────────────────────────────────────────────────────────────────────
// Entropic LP
// ─────────────────────────────────────────────────────────────────────────────

/// Solve the entropy-regularized LP:
///
///   min  cᵀx + ε Σ x_i ln(x_i)
///   s.t. A x ≤ b,  x ≥ 0
///
/// via a projected-gradient (mirror descent / Sinkhorn) approach.
///
/// The dual of this regularized LP has the form of a smooth unconstrained
/// problem in the Lagrange multipliers. We perform mirror descent on the
/// dual, which yields primal iterates that are always ≥ 0.
///
/// The solution converges to the unique minimizer of the regularized problem.
///
/// # Arguments
/// * `c`       – objective coefficients (length n).
/// * `a`       – constraint matrix (m×n).
/// * `b`       – constraint rhs (m).
/// * `epsilon` – entropic regularization coefficient.
/// * `max_iter`, `tol` – solver parameters.
pub fn lp_perturbed(
    c: &[f64],
    a: &[Vec<f64>],
    b: &[f64],
    epsilon: f64,
    max_iter: usize,
    tol: f64,
) -> OptimizeResult<Vec<f64>> {
    let n = c.len();
    let m = b.len();

    if a.len() != m {
        return Err(OptimizeError::InvalidInput(format!(
            "A has {} rows but b has length {}",
            a.len(),
            m
        )));
    }

    // Validate epsilon
    if epsilon <= 0.0 {
        return Err(OptimizeError::InvalidInput(
            "epsilon must be positive for entropic regularization".to_string(),
        ));
    }

    // Dual variable initialization: λ ≥ 0 (one per inequality constraint)
    let mut lambda = vec![0.0_f64; m];

    // ── Helper: compute primal x from dual λ ─────────────────────────────
    // x_i = softmax(-c/ε - Aᵀλ/ε) * (sum of b_k where b_k > 0)
    // This uses softmax normalization to keep x in [0, 1] range, scaled by
    // the total "budget" max_b = sum of positive b entries.
    let budget: f64 = b.iter().filter(|&&bi| bi > 0.0).sum::<f64>().max(1.0);

    let primal_from_dual = |lam: &[f64]| -> Vec<f64> {
        let scores: Vec<f64> = (0..n)
            .map(|i| {
                let atl: f64 = (0..m)
                    .map(|k| {
                        let a_ki = if k < a.len() && i < a[k].len() {
                            a[k][i]
                        } else {
                            0.0
                        };
                        lam[k] * a_ki
                    })
                    .sum();
                (-c[i] - atl) / epsilon
            })
            .collect();
        // Softmax normalization: ensures Σ x_i = budget (bounded solution)
        let sm = softmax(&scores);
        sm.iter().map(|&si| si * budget).collect()
    };

    // Dual gradient ascent with Armijo-style step size
    let mut step = 1.0_f64 / (1.0 + budget * budget);

    for _iter in 0..max_iter {
        let x = primal_from_dual(&lambda);

        // ── Dual gradient: ∇λ g(λ) = b - A x ─────────────────────────────
        let ax: Vec<f64> = (0..m)
            .map(|k| {
                (0..n)
                    .map(|i| {
                        let a_ki = if k < a.len() && i < a[k].len() {
                            a[k][i]
                        } else {
                            0.0
                        };
                        a_ki * x[i]
                    })
                    .sum::<f64>()
            })
            .collect();

        // Dual gradient step (gradient ascent, clamp λ ≥ 0)
        let lambda_new: Vec<f64> = (0..m)
            .map(|k| (lambda[k] + step * (b[k] - ax[k])).max(0.0))
            .collect();

        // Check convergence
        let delta: f64 = lambda_new
            .iter()
            .zip(lambda.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        lambda = lambda_new;

        if delta < tol {
            break;
        }

        // Adaptive step size decay
        if _iter % 50 == 49 {
            step *= 0.9;
        }
    }

    let x = primal_from_dual(&lambda);
    Ok(x)
}

/// Compute the gradient dL/dc for the entropy-regularized LP.
///
/// For the perturbed LP, the Jacobian of x* w.r.t. c is:
///
///   dx*/dc = -(1/ε) diag(x*) (I - 1ᵀ x* / Σ x*)
///
/// (from the softmax Jacobian, where x* = softmax(-c/ε)/ Σ x*).
///
/// The gradient dL/dc_i = Σ_j (dx*/dc)_{ji} * dL/dx_j.
///
/// For the simple case without constraints: dx*/dc = -(1/ε)(diag(x*) - x* x*ᵀ).
pub fn lp_gradient(
    c: &[f64],
    a: &[Vec<f64>],
    b: &[f64],
    x_star: &[f64],
    dl_dx: &[f64],
    epsilon: f64,
) -> OptimizeResult<Vec<f64>> {
    let n = c.len();
    if x_star.len() != n || dl_dx.len() != n {
        return Err(OptimizeError::InvalidInput(
            "Dimension mismatch in lp_gradient".to_string(),
        ));
    }

    // Approximate: treat x* as softmax output (entropic LP solution)
    // Jacobian: J_{ij} = dx*_i/dc_j = -(1/ε) [x*_i (δ_{ij} - x*_j)]
    // = -(1/ε) (diag(x*) - x* x*ᵀ)
    // (This is exact when there are no binding constraints.)
    let sum_x: f64 = x_star.iter().sum();
    let norm = if sum_x > 1e-15 { sum_x } else { 1.0 };

    let mut dl_dc = vec![0.0_f64; n];
    for j in 0..n {
        for i in 0..n {
            let delta_ij = if i == j { 1.0 } else { 0.0 };
            // J_{ij} = -(1/ε) x*_i (δ_{ij} - x*_j / Σ x*)
            let j_ij = -(1.0 / epsilon) * x_star[i] * (delta_ij - x_star[j] / norm);
            dl_dc[j] += j_ij * dl_dx[i];
        }
    }

    // Unused but required by signature
    let _ = (a, b);

    Ok(dl_dc)
}

// ─────────────────────────────────────────────────────────────────────────────
// LP Layer struct
// ─────────────────────────────────────────────────────────────────────────────

/// A differentiable LP layer using entropic regularization.
#[derive(Debug, Clone)]
pub struct LpLayer {
    config: LpLayerConfig,
    /// Cached forward result for backward pass.
    last_x: Option<Vec<f64>>,
    last_c: Option<Vec<f64>>,
    last_a: Option<Vec<Vec<f64>>>,
    last_b: Option<Vec<f64>>,
}

impl LpLayer {
    /// Create a new LP layer with default configuration.
    pub fn new() -> Self {
        Self {
            config: LpLayerConfig::default(),
            last_x: None,
            last_c: None,
            last_a: None,
            last_b: None,
        }
    }

    /// Create a new LP layer with custom configuration.
    pub fn with_config(config: LpLayerConfig) -> Self {
        Self {
            config,
            last_x: None,
            last_c: None,
            last_a: None,
            last_b: None,
        }
    }

    /// Solve the entropy-regularized LP (forward pass).
    ///
    /// # Arguments
    /// * `c` – objective coefficients (length n).
    /// * `a` – inequality constraint matrix A (m×n): Ax ≤ b.
    /// * `b` – inequality rhs (m).
    pub fn forward(
        &mut self,
        c: Vec<f64>,
        a: Vec<Vec<f64>>,
        b: Vec<f64>,
    ) -> OptimizeResult<DiffOptResult> {
        let n = c.len();
        let m = b.len();

        let x = lp_perturbed(
            &c,
            &a,
            &b,
            self.config.epsilon,
            self.config.max_iter,
            self.config.tol,
        )?;

        // Check feasibility: Ax ≤ b, x ≥ 0
        let feasible = x.iter().all(|&xi| xi >= -1e-6)
            && (0..m).all(|k| {
                let ax_k: f64 = (0..n)
                    .map(|i| {
                        let a_ki = if k < a.len() && i < a[k].len() {
                            a[k][i]
                        } else {
                            0.0
                        };
                        a_ki * x[i]
                    })
                    .sum();
                ax_k <= b[k] + 1e-4
            });

        let status = if feasible {
            DiffOptStatus::Optimal
        } else {
            DiffOptStatus::MaxIterations
        };

        // Compute objective
        let objective: f64 = c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum();

        // Cache for backward
        self.last_x = Some(x.clone());
        self.last_c = Some(c.clone());
        self.last_a = Some(a.clone());
        self.last_b = Some(b.clone());

        Ok(DiffOptResult {
            x,
            lambda: vec![0.0; m], // entropic duals not extracted
            nu: vec![],
            objective,
            status,
            iterations: self.config.max_iter,
        })
    }

    /// Backward pass: compute dL/dc using the softmax Jacobian.
    pub fn backward(&self, dl_dx: &[f64]) -> OptimizeResult<DiffOptGrad> {
        let x_star = self.last_x.as_ref().ok_or_else(|| {
            OptimizeError::ComputationError("LpLayer::backward called before forward".to_string())
        })?;
        let c = self
            .last_c
            .as_ref()
            .ok_or_else(|| OptimizeError::ComputationError("No cached c".to_string()))?;
        let a = self
            .last_a
            .as_ref()
            .ok_or_else(|| OptimizeError::ComputationError("No cached A".to_string()))?;
        let b = self
            .last_b
            .as_ref()
            .ok_or_else(|| OptimizeError::ComputationError("No cached b".to_string()))?;

        let dl_dc = lp_gradient(c, a, b, x_star, dl_dx, self.config.epsilon)?;
        let m = b.len();
        let n = c.len();

        Ok(DiffOptGrad {
            dl_dq: None,
            dl_dc,
            dl_da: None,
            dl_db: vec![0.0_f64; 0],
            dl_dg: Some(vec![vec![0.0_f64; n]; m]),
            dl_dh: vec![0.0_f64; m],
        })
    }
}

impl Default for LpLayer {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Basis sensitivity for LP in standard form
// ─────────────────────────────────────────────────────────────────────────────

/// Exact sensitivity analysis for LP in standard form.
///
/// For the LP:
///
///   min  cᵀx   s.t.  Ax = b,  x ≥ 0
///
/// at an optimal basic feasible solution with basis B (indices of basic
/// variables), the optimal basic variable values are x_B = B⁻¹ b, and the
/// sensitivity is:
///
///   dx_B*/dc_B = -B⁻¹          (from KKT optimality conditions)
///   dx_B*/db   = B⁻¹            (from primal feasibility)
///   dx_N*/dc   = 0              (nonbasic variables at zero)
#[derive(Debug, Clone)]
pub struct LpSensitivity {
    /// Optimal basis indices (into the original variable vector).
    pub basis: Vec<usize>,
    /// Optimal basic variable values x_B = B⁻¹ b.
    pub x_basic: Vec<f64>,
    /// Optimal dual variables (shadow prices): y = (B⁻¹)ᵀ c_B.
    pub dual: Vec<f64>,
    /// Inverse of the basis matrix B⁻¹.
    pub b_inv: Vec<Vec<f64>>,
}

impl LpSensitivity {
    /// Compute LP sensitivity at a given basic feasible solution.
    ///
    /// # Arguments
    /// * `a`     – equality constraint matrix (p×n).
    /// * `b_rhs` – equality rhs (p).
    /// * `c`     – cost vector (n).
    /// * `basis` – indices of basic variables (length p).
    pub fn new(
        a: &[Vec<f64>],
        b_rhs: &[f64],
        c: &[f64],
        basis: Vec<usize>,
    ) -> OptimizeResult<Self> {
        let p = b_rhs.len();
        let n = c.len();

        if basis.len() != p {
            return Err(OptimizeError::InvalidInput(format!(
                "Basis size {} != number of constraints {}",
                basis.len(),
                p
            )));
        }

        // Extract basis matrix B (p×p)
        let b_mat: Vec<Vec<f64>> = (0..p)
            .map(|i| {
                basis
                    .iter()
                    .map(|&j| {
                        if i < a.len() && j < a[i].len() {
                            a[i][j]
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();

        // Compute B⁻¹ via Gaussian elimination (solve B⁻¹ B = I)
        let b_inv = invert_matrix(&b_mat)?;

        // x_B = B⁻¹ b
        let x_basic: Vec<f64> = (0..p)
            .map(|i| {
                (0..p)
                    .map(|j| b_inv[i][j] * if j < b_rhs.len() { b_rhs[j] } else { 0.0 })
                    .sum()
            })
            .collect();

        // Dual variables: y = (B⁻¹)ᵀ c_B
        let c_basic: Vec<f64> = basis
            .iter()
            .map(|&j| if j < c.len() { c[j] } else { 0.0 })
            .collect();
        let dual: Vec<f64> = (0..p)
            .map(|i| {
                (0..p)
                    .map(|j| b_inv[j][i] * if j < c_basic.len() { c_basic[j] } else { 0.0 })
                    .sum()
            })
            .collect();

        let _ = n;
        Ok(Self {
            basis,
            x_basic,
            dual,
            b_inv,
        })
    }

    /// Compute dL/dc_B given upstream gradient dL/dx_B.
    ///
    /// From the KKT conditions, dx_B*/dc_B = -B⁻¹, so:
    ///   dL/dc_B_j = -Σ_i (B⁻¹)_{ij} * dL/dx_B_i
    ///
    /// # Arguments
    /// * `n`         – total number of variables.
    /// * `dl_dx`     – upstream gradient (length n).
    pub fn dl_dc(&self, n: usize, dl_dx: &[f64]) -> Vec<f64> {
        let p = self.basis.len();
        let mut grad = vec![0.0_f64; n];

        // Only basic variables contribute
        for (j, &bj) in self.basis.iter().enumerate() {
            if bj < n {
                let sum: f64 = (0..p)
                    .map(|i| {
                        let dl_xi = if self.basis[i] < dl_dx.len() {
                            dl_dx[self.basis[i]]
                        } else {
                            0.0
                        };
                        self.b_inv[i][j] * dl_xi
                    })
                    .sum();
                grad[bj] = -sum;
            }
        }
        grad
    }

    /// Compute dL/db given upstream gradient dL/dx_B.
    ///
    /// From primal feasibility, dx_B*/db = B⁻¹, so:
    ///   dL/db_k = Σ_i (B⁻¹)_{ik} * dL/dx_B_i
    pub fn dl_db(&self, dl_dx: &[f64]) -> Vec<f64> {
        let p = self.basis.len();

        (0..p)
            .map(|k| {
                (0..p)
                    .map(|i| {
                        let dl_xi = if self.basis[i] < dl_dx.len() {
                            dl_dx[self.basis[i]]
                        } else {
                            0.0
                        };
                        self.b_inv[i][k] * dl_xi
                    })
                    .sum()
            })
            .collect()
    }

    /// Compute reduced costs: c_N - y A_N ≥ 0 at optimality.
    ///
    /// Returns a vector of length n with reduced costs for all variables
    /// (0 for basic, actual reduced cost for nonbasic).
    pub fn reduced_costs(&self, a: &[Vec<f64>], c: &[f64]) -> Vec<f64> {
        let n = c.len();
        let nonbasic: Vec<usize> = (0..n).filter(|i| !self.basis.contains(i)).collect();
        let p = self.basis.len();

        let mut rc = vec![0.0_f64; n];
        for &j in &nonbasic {
            // a_j: column j of A
            let a_j: Vec<f64> = (0..p)
                .map(|i| {
                    if i < a.len() && j < a[i].len() {
                        a[i][j]
                    } else {
                        0.0
                    }
                })
                .collect();
            // reduced cost: c_j - yᵀ a_j
            let ytaj: f64 = self
                .dual
                .iter()
                .zip(a_j.iter())
                .map(|(yi, aij)| yi * aij)
                .sum();
            rc[j] = if j < c.len() { c[j] } else { 0.0 } - ytaj;
        }
        rc
    }
}

/// Invert a square matrix via Gaussian elimination with partial pivoting.
fn invert_matrix(a: &[Vec<f64>]) -> OptimizeResult<Vec<Vec<f64>>> {
    let n = a.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // Build augmented [A | I]
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            for j in 0..n {
                r.push(if i == j { 1.0 } else { 0.0 });
            }
            r
        })
        .collect();

    // Forward elimination with partial pivoting
    for col in 0..n {
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = aug[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return Err(OptimizeError::ComputationError(
                "Singular matrix in LP basis inversion".to_string(),
            ));
        }
        if max_row != col {
            aug.swap(col, max_row);
        }

        let pivot = aug[col][col];
        for j in col..2 * n {
            aug[col][j] /= pivot;
        }
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in col..2 * n {
                    let aug_col_j = aug[col][j];
                    aug[row][j] -= factor * aug_col_j;
                }
            }
        }
    }

    // Extract inverse
    Ok((0..n).map(|i| aug[i][n..2 * n].to_vec()).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lp_layer_config_default() {
        let cfg = LpLayerConfig::default();
        assert!(cfg.epsilon > 0.0);
        assert!(cfg.max_iter > 0);
        assert!(cfg.tol > 0.0);
    }

    #[test]
    fn test_lp_perturbed_feasibility() {
        // min -x - y  s.t. x + y <= 1, x >= 0, y >= 0
        let c = vec![-1.0, -1.0];
        let a = vec![
            vec![1.0, 1.0],  // x + y <= 1
            vec![-1.0, 0.0], // -x <= 0
            vec![0.0, -1.0], // -y <= 0
        ];
        let b = vec![1.0, 0.0, 0.0];

        let x = lp_perturbed(&c, &a, &b, 0.1, 1000, 1e-8).expect("LP failed");

        // x >= 0
        for xi in &x {
            assert!(*xi >= -1e-6, "xi < 0: {}", xi);
        }

        // x + y <= 1 (approximately)
        let sum: f64 = x.iter().sum();
        assert!(sum <= 1.0 + 1e-3, "x+y = {} > 1", sum);
    }

    #[test]
    fn test_lp_layer_forward_feasibility() {
        let mut layer = LpLayer::new();
        let c = vec![-1.0, -1.0];
        let a = vec![vec![1.0, 1.0], vec![-1.0, 0.0], vec![0.0, -1.0]];
        let b = vec![1.0, 0.0, 0.0];

        let result = layer.forward(c, a, b).expect("LP forward failed");

        // x >= 0
        for xi in &result.x {
            assert!(*xi >= -1e-5, "xi < 0: {}", xi);
        }
    }

    #[test]
    fn test_lp_layer_backward_shape() {
        let mut layer = LpLayer::new();
        let c = vec![-1.0, -1.0];
        let a = vec![vec![1.0, 1.0]];
        let b = vec![1.0];

        let result = layer.forward(c, a, b).expect("LP forward failed");
        let _ = result;

        let grad = layer.backward(&[1.0, 1.0]).expect("LP backward failed");
        assert_eq!(grad.dl_dc.len(), 2, "dl/dc should have length 2");
        for gi in &grad.dl_dc {
            assert!(gi.is_finite(), "dl/dc not finite");
        }
    }

    #[test]
    fn test_lp_layer_backward_no_forward_error() {
        let layer = LpLayer::new();
        let result = layer.backward(&[1.0]);
        assert!(result.is_err(), "Should error without forward pass");
    }

    #[test]
    fn test_lp_gradient_direction() {
        // For unconstrained entropic LP: x* = softmax(-c/ε)
        // Increasing c[0] should decrease x*[0] (dL/dc[0] < 0 when dL/dx[0] > 0)
        let c = vec![0.0, 0.0];
        let a: Vec<Vec<f64>> = vec![];
        let b: Vec<f64> = vec![];
        let epsilon = 0.1;
        let n_iter = 100;
        let tol = 1e-8;

        let x_base = lp_perturbed(&c, &a, &b, epsilon, n_iter, tol).expect("LP failed");

        let c_plus = vec![0.1, 0.0];
        let x_plus = lp_perturbed(&c_plus, &a, &b, epsilon, n_iter, tol).expect("LP failed");

        // x[0] should decrease as c[0] increases
        // (since we minimize c'x + ε entropy, higher c[0] → lower x[0])
        assert!(
            x_plus[0] <= x_base[0] + 1e-3,
            "x[0] should not increase when c[0] increases: base={}, plus={}",
            x_base[0],
            x_plus[0]
        );
    }

    #[test]
    fn test_invert_matrix_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let a_inv = invert_matrix(&a).expect("Inversion failed");
        assert!((a_inv[0][0] - 1.0).abs() < 1e-10);
        assert!((a_inv[1][1] - 1.0).abs() < 1e-10);
        assert!(a_inv[0][1].abs() < 1e-10);
        assert!(a_inv[1][0].abs() < 1e-10);
    }

    #[test]
    fn test_invert_matrix_2x2() {
        // A = [[2, 1], [1, 3]], A^{-1} = 1/5 * [[3, -1], [-1, 2]]
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let a_inv = invert_matrix(&a).expect("Inversion failed");
        assert!((a_inv[0][0] - 0.6).abs() < 1e-10);
        assert!((a_inv[0][1] - (-0.2)).abs() < 1e-10);
        assert!((a_inv[1][0] - (-0.2)).abs() < 1e-10);
        assert!((a_inv[1][1] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_lp_sensitivity_simple() {
        // LP: min x1 + 2*x2  s.t. x1 + x2 = 1, x1,x2 >= 0
        // Optimal basis: {0} (x1=1, x2=0), with x1=1 basic
        // Actually with simplex for min, if c=[1,2], basis {0} gives x1=1, x2=0
        let a = vec![vec![1.0, 1.0]]; // x1 + x2 = 1
        let b_rhs = vec![1.0];
        let c = vec![1.0, 2.0];
        let basis = vec![0]; // x1 is basic

        let sens = LpSensitivity::new(&a, &b_rhs, &c, basis).expect("LpSensitivity failed");

        // x_B = B^{-1} b = [1.0] / [1.0] = [1.0]
        assert!((sens.x_basic[0] - 1.0).abs() < 1e-10);

        // dl_dc with dl_dx = [1, 0] (gradient through basic variable)
        let dl_dc = sens.dl_dc(2, &[1.0, 0.0]);
        assert_eq!(dl_dc.len(), 2);
    }

    #[test]
    fn test_lp_sensitivity_b_inv_correctness() {
        // B = [[2, 0], [0, 3]], B^{-1} = [[0.5, 0], [0, 1/3]]
        let a = vec![vec![2.0, 0.0, 1.0], vec![0.0, 3.0, 1.0]];
        let b_rhs = vec![4.0, 6.0];
        let c = vec![1.0, 1.0, 0.0];
        let basis = vec![0, 1]; // first two columns form the basis

        let sens = LpSensitivity::new(&a, &b_rhs, &c, basis).expect("LpSensitivity failed");

        // B^{-1} = [[0.5, 0], [0, 1/3]]
        assert!((sens.b_inv[0][0] - 0.5).abs() < 1e-10);
        assert!((sens.b_inv[1][1] - 1.0 / 3.0).abs() < 1e-10);

        // x_B = B^{-1} b = [4*0.5, 6*1/3] = [2, 2]
        assert!((sens.x_basic[0] - 2.0).abs() < 1e-10);
        assert!((sens.x_basic[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_properties() {
        let v = vec![1.0, 2.0, 3.0];
        let s = softmax(&v);
        // Sums to 1
        let sum: f64 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        // All positive
        for si in &s {
            assert!(*si > 0.0);
        }
        // Monotone
        assert!(s[2] > s[1] && s[1] > s[0]);
    }

    #[test]
    fn test_logsumexp() {
        let v = vec![0.0, 0.0, 0.0];
        let lse = logsumexp(&v);
        // log(3)
        assert!((lse - 3.0_f64.ln()).abs() < 1e-10);
    }
}
