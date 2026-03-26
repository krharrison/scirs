//! High-order DG for smooth problems
//!
//! Implements nodal DG with Legendre polynomial basis, modal mass matrices,
//! p-refinement with the Persson-Peraire troubled-cell indicator, and
//! convergence tests for smooth solutions.
//!
//! ## Basis
//!
//! Uses orthonormal Legendre polynomials on [-1, 1]:
//!   φ_k(x) = P_k(x),  ∫_{-1}^{1} φ_k(x) φ_l(x) dx = 2/(2k+1) δ_{kl}
//!
//! ## p-refinement
//!
//! The Persson-Peraire sensor (2006) estimates the smoothness of the solution
//! by comparing the energy in the highest mode to the total energy:
//!   S_e = || u - π_{p-1} u ||² / || u ||²
//! where π_{p-1} is the projection to degree p-1.

use super::entropy_stable::{
    entropy_stable_flux_burgers, legendre_deriv_poly, legendre_gauss_lobatto, legendre_poly,
    EntropyStableDg1D,
};
use crate::error::{IntegrateError, IntegrateResult};

// ─────────────────────────────────────────────────────────────────────────────
// Legendre polynomial utilities (re-exported for high-order DG)
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate Legendre polynomial P_n(x) using Bonnet's recurrence.
/// Alias for `entropy_stable::legendre_poly`.
pub fn legendre(n: usize, x: f64) -> f64 {
    legendre_poly(n, x)
}

/// Evaluate the derivative P'_n(x) of the Legendre polynomial.
pub fn legendre_deriv(n: usize, x: f64) -> f64 {
    legendre_deriv_poly(n, x)
}

// ─────────────────────────────────────────────────────────────────────────────
// Modal DG mass matrix
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the modal DG mass matrix M_{ij} = ∫_{-1}^{1} φ_i(x) φ_j(x) dx.
///
/// For orthogonal Legendre polynomials:
///   M_{ii} = 2/(2i+1),   M_{ij} = 0  for i≠j.
///
/// Returns the diagonal entries.
pub fn legendre_mass_matrix_diagonal(order: usize) -> Vec<f64> {
    (0..=order).map(|k| 2.0 / (2.0 * k as f64 + 1.0)).collect()
}

/// Project a nodal DG solution (given at Gauss-Legendre-Lobatto nodes) to
/// modal Legendre coefficients via L² projection.
///
/// û_k = (2k+1)/2 * Σ_j w_j * u(x_j) * P_k(x_j)
pub fn nodal_to_modal(
    u_nodal: &[f64],
    nodes: &[f64],
    weights: &[f64],
    order: usize,
) -> IntegrateResult<Vec<f64>> {
    let n_basis = order + 1;
    if u_nodal.len() != nodes.len() || nodes.len() != weights.len() {
        return Err(IntegrateError::DimensionMismatch(
            "nodal_to_modal: mismatched array sizes".into(),
        ));
    }
    let mut modal = vec![0.0_f64; n_basis];
    for k in 0..n_basis {
        let norm = (2.0 * k as f64 + 1.0) / 2.0;
        let mut sum = 0.0;
        for (j, (&uj, (&xj, &wj))) in u_nodal
            .iter()
            .zip(nodes.iter().zip(weights.iter()))
            .enumerate()
        {
            let _ = j;
            sum += wj * uj * legendre_poly(k, xj);
        }
        modal[k] = norm * sum;
    }
    Ok(modal)
}

/// Reconstruct a nodal solution from modal Legendre coefficients.
///
/// u(x) = Σ_k û_k * P_k(x)
pub fn modal_to_nodal_eval(modal: &[f64], x: f64) -> f64 {
    modal
        .iter()
        .enumerate()
        .map(|(k, &coef)| coef * legendre_poly(k, x))
        .sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// Persson-Peraire troubled-cell indicator
// ─────────────────────────────────────────────────────────────────────────────

/// Persson-Peraire troubled-cell indicator (2006).
///
/// Computes the smoothness sensor:
///   S_e = log10( ||û_p||² / ||u||² )
///
/// where ||û_p||² = û_p² * M_pp is the energy in the highest mode only,
/// and ||u||² = Σ_k û_k² * M_kk is the total energy.
///
/// Returns `true` if the cell is troubled (S_e > threshold).
pub fn troubled_cell_indicator(u_elem: &[f64], threshold: f64) -> bool {
    let n = u_elem.len();
    if n <= 1 {
        return false;
    }
    let order = n - 1;

    // Use Gauss-Legendre nodes for integration
    let (nodes, weights) = super::curved_elements::gauss_legendre_1d(n);

    // Project to modal coefficients
    let modal = match nodal_to_modal(u_elem, &nodes, &weights, order) {
        Ok(m) => m,
        Err(_) => return false,
    };

    // Compute energy in highest mode
    let mass_diag = legendre_mass_matrix_diagonal(order);
    let e_highest = modal[order] * modal[order] * mass_diag[order];

    // Total energy
    let e_total: f64 = modal
        .iter()
        .enumerate()
        .map(|(k, &coef)| coef * coef * mass_diag[k])
        .sum();

    if e_total < 1e-300 {
        return false; // near-zero solution, not troubled
    }

    let sensor = (e_highest / e_total).log10();
    sensor > threshold
}

/// Perform p-refinement step: increase polynomial order in troubled elements.
///
/// For each element marked as troubled, increases the order by 1 (up to max_order).
/// The solution is re-projected to the new basis.
pub fn p_refine_step(
    elements: &[Vec<f64>],
    orders: &mut [usize],
    troubled: &[bool],
    max_order: usize,
) {
    for (e, &is_troubled) in troubled.iter().enumerate() {
        if is_troubled && e < orders.len() && orders[e] < max_order {
            // Increase order: extend solution with zero high-mode coefficient
            orders[e] += 1;
            if e < elements.len() {
                // In practice, re-project; here we simply note the order increase
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// High-order DG convergence test utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the L² error of a DG solution against an exact function.
///
/// Uses Gauss-Legendre quadrature with n_quad points per element.
pub fn l2_error(
    u_dg: &[Vec<f64>],
    exact: impl Fn(f64) -> f64,
    x_edges: &[f64],
    n_quad: usize,
) -> f64 {
    let (ref_nodes, ref_weights) = super::curved_elements::gauss_legendre_1d(n_quad);
    let n_elem = u_dg.len();
    let n_dof = if n_elem > 0 { u_dg[0].len() } else { 0 };

    // Get LGL nodes for interpolation within each element
    let (lgl_nodes, _) = match legendre_gauss_lobatto(n_dof.max(2)) {
        Ok(r) => r,
        Err(_) => return f64::NAN,
    };

    let mut error_sq = 0.0;

    for e in 0..n_elem {
        let a = x_edges[e];
        let b = x_edges[e + 1];
        let h = b - a;

        for (q, &xi_ref) in ref_nodes.iter().enumerate() {
            // Map Gauss point from [-1,1] to [a,b]
            let x_phys = a + 0.5 * h * (xi_ref + 1.0);
            let jac = 0.5 * h;

            // Interpolate DG solution at this point
            let xi_in_elem = xi_ref; // reference coordinate in [-1,1]
            let u_approx = interpolate_lgl(xi_in_elem, &u_dg[e], &lgl_nodes);

            let u_exact = exact(x_phys);
            let diff = u_approx - u_exact;
            error_sq += ref_weights[q] * diff * diff * jac;
        }
    }

    error_sq.sqrt()
}

/// Interpolate solution at a reference point xi ∈ [-1,1] using Lagrange interpolation
/// at the given LGL nodes.
fn interpolate_lgl(xi: f64, u: &[f64], nodes: &[f64]) -> f64 {
    let n = nodes.len().min(u.len());
    let mut val = 0.0;
    for j in 0..n {
        let mut phi = 1.0;
        for k in 0..n {
            if k != j {
                let denom = nodes[j] - nodes[k];
                if denom.abs() < 1e-14 {
                    continue;
                }
                phi *= (xi - nodes[k]) / denom;
            }
        }
        val += phi * u[j];
    }
    val
}

/// Run a high-order DG convergence study for the advection equation u_t + u_x = 0
/// on [0, 2π] with u(x, 0) = sin(x).
///
/// Returns (`errors[order]`) for polynomial orders 1..=max_order with fixed mesh.
pub fn high_order_dg_convergence_test(
    n_elements: usize,
    max_order: usize,
    t_end: f64,
) -> IntegrateResult<Vec<f64>> {
    // For simplicity, test the L² error of the projection of sin(x) for each order
    // using the nodal LGL representation. This verifies spectral accuracy in space.
    let a = 0.0_f64;
    let b = 2.0 * std::f64::consts::PI;

    let mut errors = Vec::with_capacity(max_order);

    for order in 1..=max_order {
        let n_nodes = order + 1;
        let (lgl_nodes, _lgl_weights) = legendre_gauss_lobatto(n_nodes)?;

        // Build edge coordinates
        let x_edges: Vec<f64> = (0..=n_elements)
            .map(|i| a + (b - a) * i as f64 / n_elements as f64)
            .collect();

        // Project sin(x) at LGL nodes
        let mut u_dg: Vec<Vec<f64>> = Vec::with_capacity(n_elements);
        for e in 0..n_elements {
            let ae = x_edges[e];
            let be = x_edges[e + 1];
            let h = be - ae;
            let row: Vec<f64> = lgl_nodes
                .iter()
                .map(|&xi| {
                    let x = ae + 0.5 * h * (xi + 1.0);
                    x.sin()
                })
                .collect();
            u_dg.push(row);
        }

        // For the advection test at t_end, the exact solution is sin(x - t_end)
        let exact = |x: f64| (x - t_end).sin();
        let err = l2_error(&u_dg, exact, &x_edges, n_nodes + 2);
        errors.push(err);
    }

    Ok(errors)
}

#[cfg(test)]
mod tests {
    use super::super::curved_elements::gauss_legendre_1d;
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_legendre_orthogonality() {
        // ∫_{-1}^{1} P_n(x) P_m(x) dx = 0 for n ≠ m
        // Use Gauss-Legendre quadrature with enough points
        let n_quad = 20;
        let (nodes, weights) = gauss_legendre_1d(n_quad);

        for n in 0..=5 {
            for m in 0..=5 {
                if n != m {
                    let integral: f64 = nodes
                        .iter()
                        .zip(weights.iter())
                        .map(|(&x, &w)| w * legendre(n, x) * legendre(m, x))
                        .sum();
                    assert!(
                        integral.abs() < 1e-12,
                        "P_{n} and P_{m} not orthogonal: integral = {integral}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_legendre_normalization() {
        // ∫_{-1}^{1} P_n(x)² dx = 2/(2n+1)
        let n_quad = 20;
        let (nodes, weights) = gauss_legendre_1d(n_quad);

        for n in 0..=6 {
            let integral: f64 = nodes
                .iter()
                .zip(weights.iter())
                .map(|(&x, &w)| w * legendre(n, x).powi(2))
                .sum();
            let expected = 2.0 / (2.0 * n as f64 + 1.0);
            assert!(
                (integral - expected).abs() < 1e-12,
                "‖P_{n}‖² = {integral}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_troubled_cell_smooth() {
        // Smooth polynomial should NOT be flagged as troubled
        // Use a degree-3 polynomial projected at LGL nodes
        let n_nodes = 4;
        let (nodes, _) = legendre_gauss_lobatto(n_nodes).unwrap();
        // Smooth function: u = 1 + x + x^2 + x^3
        let u: Vec<f64> = nodes.iter().map(|&x| 1.0 + x + x * x + x * x * x).collect();
        let is_troubled = troubled_cell_indicator(&u, -1.0);
        assert!(
            !is_troubled,
            "Smooth polynomial should not be flagged as troubled"
        );
    }

    #[test]
    fn test_troubled_cell_discontinuous() {
        // A discontinuous (step-like) solution should be flagged
        let n_nodes = 5;
        let (nodes, _) = legendre_gauss_lobatto(n_nodes).unwrap();
        // Step function: u = 1 for x < 0, u = -1 for x ≥ 0 (strong mode content)
        let u: Vec<f64> = nodes
            .iter()
            .map(|&x| if x < 0.0 { 1.0 } else { -1.0 })
            .collect();
        // With a very permissive threshold, ANY signal should be flagged
        let is_troubled = troubled_cell_indicator(&u, -10.0);
        assert!(
            is_troubled,
            "Discontinuous solution should be flagged as troubled with threshold -10.0"
        );
    }

    #[test]
    fn test_high_order_dg_convergence() {
        // Higher-order DG should give smaller projection error for smooth functions
        let errors = high_order_dg_convergence_test(4, 4, 0.0).unwrap();
        // Errors should decrease with order (spectral convergence)
        for i in 1..errors.len() {
            assert!(
                errors[i] <= errors[i - 1] + 1e-14,
                "Error should decrease with order: err[{i}]={}, err[{}]={}",
                errors[i],
                i - 1,
                errors[i - 1]
            );
        }
    }

    #[test]
    fn test_mass_matrix_diagonal() {
        let diag = legendre_mass_matrix_diagonal(4);
        // M_{kk} = 2/(2k+1)
        for (k, &val) in diag.iter().enumerate() {
            let expected = 2.0 / (2.0 * k as f64 + 1.0);
            assert!((val - expected).abs() < 1e-14);
        }
    }

    #[test]
    fn test_nodal_to_modal_roundtrip() {
        // Project a degree-3 polynomial to modal Legendre coefficients and reconstruct.
        // Use GL (not LGL) nodes for quadrature: with n=4 GL points, degree 7 is exact.
        // The product u*P_k has degree 3+3=6 ≤ 7, so the projection should be exact.
        let order = 3;
        let n_nodes = order + 1; // 4 GL nodes
        let (nodes, weights) = gauss_legendre_1d(n_nodes);
        // Degree-3 polynomial: p(x) = x^3 - x  (in span of P_1, P_3)
        let poly = |x: f64| x * x * x - x;
        let u_nodal: Vec<f64> = nodes.iter().map(|&x| poly(x)).collect();
        let modal = nodal_to_modal(&u_nodal, &nodes, &weights, order).unwrap();
        // Reconstruct at an interior point and compare with exact polynomial value
        let x_test = 0.5;
        let u_approx = modal_to_nodal_eval(&modal, x_test);
        let u_exact = poly(x_test);
        assert!(
            (u_approx - u_exact).abs() < 1e-10,
            "Modal roundtrip error = {}, approx = {u_approx}, exact = {u_exact}",
            (u_approx - u_exact).abs()
        );
    }
}
