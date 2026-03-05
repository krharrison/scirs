//! 1D Spectral Element Method (SEM)
//!
//! Implements a high-order spectral element discretization on a 1-D domain
//! using Gauss-Lobatto-Legendre (GLL) quadrature nodes within each element.
//!
//! The method solves boundary value problems of the form:
//! ```text
//!     -u''(x) + c·u(x) = f(x),   x ∈ [a, b]
//!     u(a) = bc_left
//!     u(b) = bc_right
//! ```
//! Each element uses a polynomial of order `p` (so `p+1` GLL nodes).  The
//! global system is assembled by element contributions via direct stiffness
//! summation (DSS) and solved with a dense Gaussian elimination.
//!
//! # References
//! Karniadakis & Sherwin, *Spectral/hp Element Methods*, Oxford (2005).

use scirs2_core::ndarray::{Array1, Array2};
use std::f64::consts::PI;

use crate::error::IntegrateError;

// ─────────────────────────────────────────────────────────────────────────────
// GLL nodes and weights
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the `n` Gauss-Lobatto-Legendre (GLL) nodes on [-1, 1] and their
/// associated quadrature weights.
///
/// The GLL points include the endpoints ±1 and `n-2` interior points that are
/// zeros of `P'_{n-1}(x)` (the derivative of the Legendre polynomial of
/// degree `n-1`).
///
/// Weights:  `w_k = 2 / ((n-1)·n · [P_{n-1}(x_k)]²)`
pub fn gll_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    match n {
        0 | 1 => (vec![0.0], vec![2.0]),
        2 => (vec![-1.0, 1.0], vec![1.0, 1.0]),
        3 => {
            // Nodes: -1, 0, 1;  weights: 1/3, 4/3, 1/3
            let nodes = vec![-1.0, 0.0, 1.0];
            let weights = vec![1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0];
            (nodes, weights)
        }
        _ => compute_gll(n),
    }
}

/// General GLL computation for n ≥ 4.
fn compute_gll(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0f64; n];
    nodes[0] = -1.0;
    nodes[n - 1] = 1.0;

    // Interior nodes are zeros of P'_{n-1}(x).
    // Use initial Chebyshev-like estimates then Newton's method.
    // We iterate on P'_{n-1}(x) = 0 using the secant method / Newton.
    for k in 1..n - 1 {
        // Chebyshev initial guess (shifted to [-1,1])
        let theta = PI * (n as f64 - 1.0 - k as f64) / (n as f64 - 1.0);
        let mut x = theta.cos();

        // Newton iteration: find root of dP = P'_{n-1}(x)
        for _ in 0..50 {
            let (p, dp, d2p) = legendre_pdd(n - 1, x);
            let _ = p;
            // Newton step: x <- x - P'(x)/P''(x)
            if d2p.abs() < 1e-30 {
                break;
            }
            let delta = dp / d2p;
            x -= delta;
            if delta.abs() < 1e-15 {
                break;
            }
        }
        nodes[k] = x;
    }

    // Sort nodes (Newton can occasionally disorder them for small n)
    nodes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Enforce exact endpoints
    nodes[0] = -1.0;
    nodes[n - 1] = 1.0;

    // Quadrature weights: w_k = 2 / ((n-1)*n * [P_{n-1}(x_k)]^2)
    let weights: Vec<f64> = nodes
        .iter()
        .map(|&x| {
            let (p, _, _) = legendre_pdd(n - 1, x);
            2.0 / ((n - 1) as f64 * n as f64 * p * p)
        })
        .collect();

    (nodes, weights)
}

/// Compute P_n(x), P'_n(x), P''_n(x) via stable three-term recurrence.
fn legendre_pdd(n: usize, x: f64) -> (f64, f64, f64) {
    if n == 0 {
        return (1.0, 0.0, 0.0);
    }
    if n == 1 {
        return (x, 1.0, 0.0);
    }

    let mut p_prev = 1.0f64; // P_{k-1}
    let mut p_curr = x;      // P_k
    let mut dp_prev = 0.0f64;
    let mut dp_curr = 1.0f64;

    for k in 1..n {
        let kf = k as f64;
        // Bonnet's recurrence for P
        let p_next = ((2.0 * kf + 1.0) * x * p_curr - kf * p_prev) / (kf + 1.0);
        // Derivative: P'_{k+1} = ((2k+1) * P_k + (2k+1)*x*P'_k - k*P'_{k-1}) / (k+1)
        // Simpler: P'_{k+1}(x) = (k+1)*P_k(x) / (1-x²) + x/(1-x²) * ... 
        // Use the direct formula: P'_n(x) = n/(x²-1) * [x*P_n(x) - P_{n-1}(x)]   (for x ≠ ±1)
        // Actually use the recurrence for derivative:
        // (k+1) P_{k+1}' = (2k+1)(P_k + x P_k') - k P_{k-1}'
        let dp_next =
            ((2.0 * kf + 1.0) * (p_curr + x * dp_curr) - kf * dp_prev) / (kf + 1.0);
        p_prev = p_curr;
        p_curr = p_next;
        dp_prev = dp_curr;
        dp_curr = dp_next;
    }

    // Second derivative via Legendre ODE: (1-x²)P'' - 2x P' + n(n+1)P = 0
    // => P'' = [2x P' - n(n+1) P] / (1-x²)
    let d2p = if (1.0 - x * x).abs() > 1e-12 {
        (2.0 * x * dp_curr - n as f64 * (n as f64 + 1.0) * p_curr) / (1.0 - x * x)
    } else {
        // At x = ±1 use the closed form:
        // P''_n(1) = n(n+1)(n+2)(n-1)/8   ... derived from successive L'Hopital
        let s = if x > 0.0 { 1.0 } else { (-1.0f64).powi(n as i32) };
        s * (n as f64) * (n as f64 + 1.0) * (n as f64 + 2.0) * (n as f64 - 1.0) / 8.0
    };

    (p_curr, dp_curr, d2p)
}

// ─────────────────────────────────────────────────────────────────────────────
// Lagrange basis functions
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate all `n` Lagrange basis functions ℓ_j(ξ) at point ξ ∈ [-1,1].
fn lagrange_basis(nodes: &[f64], xi: f64) -> Vec<f64> {
    let n = nodes.len();
    let mut basis = vec![1.0f64; n];
    for i in 0..n {
        for k in 0..n {
            if k == i {
                continue;
            }
            let denom = nodes[i] - nodes[k];
            if denom.abs() < 1e-15 {
                basis[i] = 0.0;
                break;
            }
            basis[i] *= (xi - nodes[k]) / denom;
        }
    }
    basis
}

/// Evaluate dℓ_i/dξ for all i at point ξ.
fn lagrange_deriv(nodes: &[f64], xi: f64) -> Vec<f64> {
    let n = nodes.len();
    let mut derivs = vec![0.0f64; n];
    for i in 0..n {
        for m in 0..n {
            if m == i {
                continue;
            }
            let denom_mi = nodes[i] - nodes[m];
            if denom_mi.abs() < 1e-15 {
                continue;
            }
            let mut prod = 1.0 / denom_mi;
            for k in 0..n {
                if k == i || k == m {
                    continue;
                }
                let denom_ik = nodes[i] - nodes[k];
                if denom_ik.abs() < 1e-15 {
                    prod = 0.0;
                    break;
                }
                prod *= (xi - nodes[k]) / denom_ik;
            }
            derivs[i] += prod;
        }
    }
    derivs
}

// ─────────────────────────────────────────────────────────────────────────────
// Public struct
// ─────────────────────────────────────────────────────────────────────────────

/// 1-D Spectral Element discretization for solving BVPs.
///
/// The domain `[a, b]` is divided into `n_elements` equal sub-elements, each
/// using a polynomial of degree `poly_order` with GLL nodes.
#[derive(Debug, Clone)]
pub struct SpectralElement1D {
    n_elements: usize,
    poly_order: usize,
    domain: (f64, f64),
    /// GLL nodes on [-1, 1]
    pub nodes: Vec<f64>,
    /// GLL quadrature weights on [-1, 1]
    pub weights: Vec<f64>,
}

impl SpectralElement1D {
    /// Create a new spectral element discretization.
    ///
    /// # Arguments
    /// * `n_elements` – number of elements (≥ 1)
    /// * `poly_order` – polynomial degree per element (≥ 1, typically 2–8)
    /// * `domain` – (a, b) endpoints of the physical domain
    pub fn new(n_elements: usize, poly_order: usize, domain: (f64, f64)) -> Self {
        let n_nodes = poly_order + 1;
        let (nodes, weights) = gll_nodes_weights(n_nodes);
        SpectralElement1D {
            n_elements,
            poly_order,
            domain,
            nodes,
            weights,
        }
    }

    /// Number of GLL nodes per element (= poly_order + 1)
    pub fn nodes_per_element(&self) -> usize {
        self.poly_order + 1
    }

    /// Total number of unique global DOFs (with continuity between elements)
    pub fn n_global_dofs(&self) -> usize {
        self.n_elements * self.poly_order + 1
    }

    /// Solve `-u''(x) + c·u(x) = f(x)` with Dirichlet BCs u(a)=bc_left,
    /// u(b)=bc_right.
    ///
    /// Returns `(x_nodes, u_values)` where `x_nodes` contains the global
    /// collocation points and `u_values` the corresponding solution values.
    pub fn solve(
        &self,
        f: impl Fn(f64) -> f64,
        c: f64,
        bc_left: f64,
        bc_right: f64,
    ) -> Result<(Vec<f64>, Vec<f64>), IntegrateError> {
        if self.n_elements == 0 {
            return Err(IntegrateError::InvalidInput(
                "n_elements must be ≥ 1".to_string(),
            ));
        }
        if self.poly_order < 1 {
            return Err(IntegrateError::InvalidInput(
                "poly_order must be ≥ 1".to_string(),
            ));
        }

        let (a, b) = self.domain;
        if b <= a {
            return Err(IntegrateError::InvalidInput(
                "domain must satisfy b > a".to_string(),
            ));
        }

        let n_dofs = self.n_global_dofs();
        let p = self.poly_order;
        let n_local = p + 1; // nodes per element

        // Element length in physical space
        let h_elem = (b - a) / self.n_elements as f64;

        // Global x-coordinates
        let x_global = self.global_x_nodes(a, h_elem);

        // Assemble global stiffness K and load f_vec
        let mut k_mat = Array2::<f64>::zeros((n_dofs, n_dofs));
        let mut f_vec = Array1::<f64>::zeros(n_dofs);

        for elem in 0..self.n_elements {
            let x_a = a + elem as f64 * h_elem;
            // Jacobian: dx/dξ = h_elem/2  (maps ξ∈[-1,1] to [x_a, x_a+h_elem])
            let jac = h_elem / 2.0;

            let (k_loc, m_loc, f_loc) = self.element_matrices(x_a, jac, c, &f);

            // Scatter (DSS)
            let g_start = elem * p;
            for i in 0..n_local {
                let gi = g_start + i;
                f_vec[gi] += f_loc[i];
                for j in 0..n_local {
                    let gj = g_start + j;
                    k_mat[[gi, gj]] += k_loc[[i, j]] + m_loc[[i, j]];
                }
            }
        }

        // Apply Dirichlet BCs by modification
        // First: subtract BC contribution from interior equations
        // (column elimination method for symmetric BCs)
        //
        // Equation 0: u[0] = bc_left
        for row in 1..n_dofs {
            f_vec[row] -= k_mat[[row, 0]] * bc_left;
            k_mat[[row, 0]] = 0.0;
            k_mat[[0, row]] = 0.0;
        }
        k_mat[[0, 0]] = 1.0;
        f_vec[0] = bc_left;

        // Equation n-1: u[n-1] = bc_right
        for row in 0..n_dofs - 1 {
            f_vec[row] -= k_mat[[row, n_dofs - 1]] * bc_right;
            k_mat[[row, n_dofs - 1]] = 0.0;
            k_mat[[n_dofs - 1, row]] = 0.0;
        }
        k_mat[[n_dofs - 1, n_dofs - 1]] = 1.0;
        f_vec[n_dofs - 1] = bc_right;

        let _ = x_global;
        // Solve K u = f
        let u_vec = gaussian_elimination(&k_mat, &f_vec)?;

        // Rebuild x_global for output
        let x_out = self.global_x_nodes(a, h_elem);
        Ok((x_out, u_vec))
    }

    /// Interpolate solution `u` (at global GLL nodes) to a physical point `x`.
    pub fn interpolate(&self, u: &[f64], x: f64) -> f64 {
        let (a, b) = self.domain;
        let h_elem = (b - a) / self.n_elements as f64;
        let p = self.poly_order;
        let n_local = p + 1;

        let x = x.clamp(a, b);

        let elem = {
            let idx = ((x - a) / h_elem).floor() as usize;
            idx.min(self.n_elements - 1)
        };

        let x_a = a + elem as f64 * h_elem;
        let x_b = x_a + h_elem;
        let xi = 2.0 * (x - x_a) / (x_b - x_a) - 1.0;

        let basis = lagrange_basis(&self.nodes, xi);
        let g_start = elem * p;
        let mut val = 0.0;
        for i in 0..n_local {
            val += basis[i] * u[g_start + i];
        }
        val
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    fn global_x_nodes(&self, a: f64, h_elem: f64) -> Vec<f64> {
        let p = self.poly_order;
        let n_dofs = self.n_global_dofs();
        let mut x = vec![0.0f64; n_dofs];

        for elem in 0..self.n_elements {
            let x_a = a + elem as f64 * h_elem;
            let g_start = elem * p;
            for i in 0..=p {
                let xi = self.nodes[i];
                // Map ξ ∈ [-1,1] → [x_a, x_a+h_elem]
                x[g_start + i] = x_a + h_elem * (xi + 1.0) / 2.0;
            }
        }
        x
    }

    /// Compute local stiffness K_e, mass M_e, and load f_e for element at x_a.
    fn element_matrices(
        &self,
        x_a: f64,
        jac: f64, // dx/dξ = h_elem/2
        c: f64,
        f: &dyn Fn(f64) -> f64,
    ) -> (Array2<f64>, Array2<f64>, Vec<f64>) {
        let n_local = self.poly_order + 1;
        let mut k_loc = Array2::<f64>::zeros((n_local, n_local));
        let mut m_loc = Array2::<f64>::zeros((n_local, n_local));
        let mut f_loc = vec![0.0f64; n_local];

        for q in 0..self.weights.len() {
            let xi_q = self.nodes[q];
            let w_q = self.weights[q];
            // Map quadrature point to physical coordinates
            let x_phys = x_a + jac * (xi_q + 1.0);

            let basis = lagrange_basis(&self.nodes, xi_q);
            let dbasis_dxi = lagrange_deriv(&self.nodes, xi_q);

            let f_val = f(x_phys);

            for i in 0..n_local {
                f_loc[i] += w_q * basis[i] * f_val * jac;
                for j in 0..n_local {
                    // Stiffness: ∫ dℓ_i/dx dℓ_j/dx dx
                    //   = ∫ (dℓ_i/dξ / J)(dℓ_j/dξ / J) J dξ
                    //   = ∫ dℓ_i/dξ dℓ_j/dξ / J dξ
                    k_loc[[i, j]] += w_q * dbasis_dxi[i] * dbasis_dxi[j] / jac;
                    // Mass: c * ∫ ℓ_i ℓ_j J dξ
                    m_loc[[i, j]] += w_q * c * basis[i] * basis[j] * jac;
                }
            }
        }

        (k_loc, m_loc, f_loc)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian elimination (dense, partial pivoting)
// ─────────────────────────────────────────────────────────────────────────────

fn gaussian_elimination(
    a: &Array2<f64>,
    b: &Array1<f64>,
) -> Result<Vec<f64>, IntegrateError> {
    let n = b.len();
    let mut mat: Vec<Vec<f64>> = (0..n).map(|i| a.row(i).to_vec()).collect();
    let mut rhs: Vec<f64> = b.to_vec();

    for col in 0..n {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = mat[col][col].abs();
        for row in col + 1..n {
            if mat[row][col].abs() > max_val {
                max_val = mat[row][col].abs();
                max_row = row;
            }
        }
        mat.swap(col, max_row);
        rhs.swap(col, max_row);

        let pivot = mat[col][col];
        if pivot.abs() < 1e-14 {
            return Err(IntegrateError::LinearSolveError(
                "Singular matrix in Gaussian elimination".to_string(),
            ));
        }

        for row in col + 1..n {
            let factor = mat[row][col] / pivot;
            for k in col..n {
                let tmp = mat[col][k];
                mat[row][k] -= factor * tmp;
            }
            let tmp = rhs[col];
            rhs[row] -= factor * tmp;
        }
    }

    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for j in i + 1..n {
            sum -= mat[i][j] * x[j];
        }
        if mat[i][i].abs() < 1e-14 {
            return Err(IntegrateError::LinearSolveError(
                "Near-zero diagonal in back-substitution".to_string(),
            ));
        }
        x[i] = sum / mat[i][i];
    }

    Ok(x)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_gll_nodes_n2() {
        let (nodes, weights) = gll_nodes_weights(2);
        assert_eq!(nodes.len(), 2);
        assert!((nodes[0] - (-1.0)).abs() < 1e-12);
        assert!((nodes[1] - 1.0).abs() < 1e-12);
        assert!((weights[0] - 1.0).abs() < 1e-12);
        assert!((weights[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_gll_nodes_n3() {
        let (nodes, weights) = gll_nodes_weights(3);
        assert_eq!(nodes.len(), 3);
        assert!((nodes[0] - (-1.0)).abs() < 1e-12);
        assert!(
            nodes[1].abs() < 1e-10,
            "Middle node should be ~0, got {}",
            nodes[1]
        );
        assert!((nodes[2] - 1.0).abs() < 1e-12);
        assert!(
            (weights[0] - 1.0 / 3.0).abs() < 1e-10,
            "w0 = {}",
            weights[0]
        );
        assert!(
            (weights[1] - 4.0 / 3.0).abs() < 1e-10,
            "w1 = {}",
            weights[1]
        );
        assert!(
            (weights[2] - 1.0 / 3.0).abs() < 1e-10,
            "w2 = {}",
            weights[2]
        );
    }

    #[test]
    fn test_gll_nodes_quadrature_exactness() {
        // GLL with n nodes integrates polynomials of degree ≤ 2n-3 exactly.
        // For n=5 that means degree ≤ 7.
        let (nodes, weights) = gll_nodes_weights(5);
        // ∫_{-1}^{1} x^6 dx = 2/7
        let integral: f64 = nodes
            .iter()
            .zip(weights.iter())
            .map(|(&x, &w)| w * x.powi(6))
            .sum();
        assert!(
            (integral - 2.0 / 7.0).abs() < 1e-12,
            "GLL quadrature exactness failed: {} (expected {})",
            integral,
            2.0 / 7.0
        );
    }

    #[test]
    fn test_gll_symmetry() {
        for n in 2..=8 {
            let (nodes, weights) = gll_nodes_weights(n);
            for i in 0..n {
                let j = n - 1 - i;
                assert!(
                    (nodes[i] + nodes[j]).abs() < 1e-12,
                    "GLL nodes not symmetric for n={n}, i={i}: xi={}, xj={}",
                    nodes[i],
                    nodes[j]
                );
                assert!(
                    (weights[i] - weights[j]).abs() < 1e-10,
                    "GLL weights not symmetric for n={n}, i={i}"
                );
            }
        }
    }

    /// Test weight sum: ∫_{-1}^{1} 1 dx = 2
    #[test]
    fn test_gll_weight_sum() {
        for n in 2..=8 {
            let (_, weights) = gll_nodes_weights(n);
            let sum: f64 = weights.iter().sum();
            assert!(
                (sum - 2.0).abs() < 1e-12,
                "Weight sum for n={n}: {sum} ≠ 2"
            );
        }
    }

    /// Test: -u'' = π²sin(πx) on [0,1], u(0)=u(1)=0 → u = sin(πx)
    #[test]
    fn test_solve_poisson_1d() {
        let sem = SpectralElement1D::new(4, 4, (0.0, 1.0));
        let f = |x: f64| PI * PI * (PI * x).sin();
        let (x_nodes, u) = sem.solve(f, 0.0, 0.0, 0.0).expect("SEM solve failed");

        let mut max_err = 0.0f64;
        for (xi, ui) in x_nodes.iter().zip(u.iter()) {
            let exact = (PI * xi).sin();
            let err = (ui - exact).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(
            max_err < 1e-6,
            "Max error {max_err:.2e} is too large for order-4 SEM"
        );
    }

    /// Test: -u'' + u = (π²+1)sin(πx) on [0,1] → u = sin(πx)
    #[test]
    fn test_solve_reaction_diffusion_1d() {
        let sem = SpectralElement1D::new(4, 4, (0.0, 1.0));
        let f = |x: f64| (PI * PI + 1.0) * (PI * x).sin();
        let (x_nodes, u) = sem.solve(f, 1.0, 0.0, 0.0).expect("SEM solve failed");

        let mut max_err = 0.0f64;
        for (xi, ui) in x_nodes.iter().zip(u.iter()) {
            let exact = (PI * xi).sin();
            let err = (ui - exact).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(
            max_err < 1e-6,
            "Max error {max_err:.2e} for reaction-diffusion"
        );
    }

    /// Test: non-homogeneous BCs: -u''=0, u(0)=0, u(1)=1 → u=x
    #[test]
    fn test_solve_nonzero_bcs() {
        let sem = SpectralElement1D::new(3, 3, (0.0, 1.0));
        let (x_nodes, u) = sem.solve(|_| 0.0, 0.0, 0.0, 1.0).expect("SEM solve failed");

        for (xi, ui) in x_nodes.iter().zip(u.iter()) {
            assert!(
                (ui - xi).abs() < 1e-12,
                "Expected u={xi}, got u={ui} at x={xi}"
            );
        }
    }

    /// Test interpolation
    #[test]
    fn test_interpolation() {
        let sem = SpectralElement1D::new(4, 4, (0.0, 1.0));
        let f = |x: f64| PI * PI * (PI * x).sin();
        let (_, u) = sem.solve(f, 0.0, 0.0, 0.0).expect("SEM solve failed");

        let u_mid = sem.interpolate(&u, 0.5);
        let exact_mid = (PI * 0.5).sin();
        assert!(
            (u_mid - exact_mid).abs() < 1e-5,
            "Interpolation error at x=0.5: {:.2e}",
            (u_mid - exact_mid).abs()
        );
    }
}
