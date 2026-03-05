//! Fluid dynamics dataset generators.
//!
//! Provides:
//! - Analytical Poiseuille channel flow profiles.
//! - Lid-driven cavity 2-D Navier–Stokes solution via the Chorin projection method.
//! - Taylor–Green vortex exact solution.
//!
//! All functions return grids indexed as `[iy][ix]` (row-major, y is the
//! vertical axis) unless stated otherwise.

use crate::error::{DatasetsError, Result};
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Poiseuille flow
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the analytical Poiseuille velocity profile on an `n_x × n_y` grid.
///
/// The flow is in the x-direction through a channel of half-height `H/2`
/// centred at `y = 0`.  The analytical solution is:
/// ```text
/// u(y) = -1/(2μ) · (dp/dx) · (H²/4 - y²)
/// ```
/// Since the pressure gradient drives the flow, convention is `dp_dx < 0` for
/// positive flow.
///
/// # Parameters
/// - `n_x`, `n_y` — number of grid cells in x and y (must be ≥ 2)
/// - `dp_dx`      — streamwise pressure gradient (Pa/m)
/// - `mu`         — dynamic viscosity (Pa·s, must be > 0)
///
/// # Returns
/// `Vec<Vec<f64>>` with shape `[n_y][n_x]`.  Every column is identical
/// (fully-developed flow), and rows vary with `y`.
pub fn poiseuille_flow(n_x: usize, n_y: usize, dp_dx: f64, mu: f64) -> Result<Vec<Vec<f64>>> {
    if n_x < 2 || n_y < 2 {
        return Err(DatasetsError::InvalidFormat(
            "poiseuille_flow: n_x and n_y must be >= 2".into(),
        ));
    }
    if mu <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "poiseuille_flow: mu must be > 0".into(),
        ));
    }

    // y ∈ [-H/2, H/2] with H = 1 (non-dimensional)
    let h_half = 0.5;
    let mut grid = vec![vec![0.0f64; n_x]; n_y];

    for iy in 0..n_y {
        let y = -h_half + (2.0 * h_half) * (iy as f64) / (n_y as f64 - 1.0);
        let u = -dp_dx / (2.0 * mu) * (h_half * h_half - y * y);
        for ix in 0..n_x {
            grid[iy][ix] = u;
        }
    }
    Ok(grid)
}

// ─────────────────────────────────────────────────────────────────────────────
// Lid-driven cavity (simplified Chorin projection)
// ─────────────────────────────────────────────────────────────────────────────

/// Solve the 2-D lid-driven cavity problem using a simplified Chorin projection.
///
/// The domain is `[0,1] × [0,1]` with:
/// - Top lid moving at `u = 1, v = 0`.
/// - No-slip on other walls.
///
/// The method alternates between:
/// 1. Convection–diffusion step for the intermediate velocity `u*`.
/// 2. Pressure Poisson solve (Jacobi iterations).
/// 3. Projection to obtain divergence-free velocity.
///
/// This is a simplified implementation intended for dataset generation rather
/// than high-fidelity CFD.  Use moderate `re` (≤ 400) and sufficient `n_steps`
/// for stable results.
///
/// # Parameters
/// - `n`       — number of interior cells per side (grid is `(n+2) × (n+2)` including ghost cells)
/// - `re`      — Reynolds number
/// - `n_steps` — number of time steps to advance
///
/// # Returns
/// `Vec<Vec<(f64, f64)>>` with shape `[n+2][n+2]`; each entry is `(u, v)`.
pub fn lid_driven_cavity_2d(
    n: usize,
    re: f64,
    n_steps: usize,
) -> Result<Vec<Vec<(f64, f64)>>> {
    if n < 2 {
        return Err(DatasetsError::InvalidFormat(
            "lid_driven_cavity_2d: n must be >= 2".into(),
        ));
    }
    if re <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "lid_driven_cavity_2d: re must be > 0".into(),
        ));
    }

    let size = n + 2; // includes ghost/boundary cells
    let dx = 1.0 / (n + 1) as f64;
    let dy = dx;
    // CFL-limited time step
    let dt = 0.25 * dx * dx * re; // diffusive time step limit
    let dt = dt.min(0.5 * dx); // also limit by convective scale

    // Velocity fields u[iy][ix], v[iy][ix]
    let mut u = vec![vec![0.0f64; size]; size];
    let mut v = vec![vec![0.0f64; size]; size];
    let mut p = vec![vec![0.0f64; size]; size];

    // Lid boundary: top row (iy = n+1) has u = 1.
    for ix in 0..size {
        u[size - 1][ix] = 1.0;
    }

    let nu = 1.0 / re; // kinematic viscosity (with unit velocity and domain)

    for _ in 0..n_steps {
        // ── Step 1: Compute intermediate velocity (u*) ──────────────────────
        let mut u_star = u.clone();
        let mut v_star = v.clone();

        for iy in 1..=n {
            for ix in 1..=n {
                // Convective terms (upwind)
                let dudx = if u[iy][ix] > 0.0 {
                    (u[iy][ix] - u[iy][ix - 1]) / dx
                } else {
                    (u[iy][ix + 1] - u[iy][ix]) / dx
                };
                let dudy = if v[iy][ix] > 0.0 {
                    (u[iy][ix] - u[iy - 1][ix]) / dy
                } else {
                    (u[iy + 1][ix] - u[iy][ix]) / dy
                };
                let dvdx = if u[iy][ix] > 0.0 {
                    (v[iy][ix] - v[iy][ix - 1]) / dx
                } else {
                    (v[iy][ix + 1] - v[iy][ix]) / dx
                };
                let dvdy = if v[iy][ix] > 0.0 {
                    (v[iy][ix] - v[iy - 1][ix]) / dy
                } else {
                    (v[iy + 1][ix] - v[iy][ix]) / dy
                };
                // Diffusive terms (central)
                let d2udx2 = (u[iy][ix + 1] - 2.0 * u[iy][ix] + u[iy][ix - 1]) / (dx * dx);
                let d2udy2 = (u[iy + 1][ix] - 2.0 * u[iy][ix] + u[iy - 1][ix]) / (dy * dy);
                let d2vdx2 = (v[iy][ix + 1] - 2.0 * v[iy][ix] + v[iy][ix - 1]) / (dx * dx);
                let d2vdy2 = (v[iy + 1][ix] - 2.0 * v[iy][ix] + v[iy - 1][ix]) / (dy * dy);

                u_star[iy][ix] = u[iy][ix]
                    + dt * (-(u[iy][ix] * dudx + v[iy][ix] * dudy)
                        + nu * (d2udx2 + d2udy2));
                v_star[iy][ix] = v[iy][ix]
                    + dt * (-(u[iy][ix] * dvdx + v[iy][ix] * dvdy)
                        + nu * (d2vdx2 + d2vdy2));
            }
        }

        // ── Step 2: Pressure Poisson (Jacobi, 20 inner iterations) ──────────
        let p_old = p.clone();
        for _ in 0..20 {
            let p_prev = p.clone();
            for iy in 1..=n {
                for ix in 1..=n {
                    let div = (u_star[iy][ix + 1] - u_star[iy][ix - 1]) / (2.0 * dx)
                        + (v_star[iy + 1][ix] - v_star[iy - 1][ix]) / (2.0 * dy);
                    p[iy][ix] = 0.25
                        * (p_prev[iy][ix + 1]
                            + p_prev[iy][ix - 1]
                            + p_prev[iy + 1][ix]
                            + p_prev[iy - 1][ix]
                            - dx * dx * div / dt);
                }
            }
            // Neumann BC on pressure walls: dp/dn = 0
            for ix in 0..size {
                p[0][ix] = p[1][ix];
                p[size - 1][ix] = p[size - 2][ix];
            }
            for iy in 0..size {
                p[iy][0] = p[iy][1];
                p[iy][size - 1] = p[iy][size - 2];
            }
        }
        // Restore old pressure at walls (no-flux)
        let _ = p_old; // suppress unused warning

        // ── Step 3: Projection ───────────────────────────────────────────────
        for iy in 1..=n {
            for ix in 1..=n {
                u[iy][ix] = u_star[iy][ix]
                    - dt * (p[iy][ix + 1] - p[iy][ix - 1]) / (2.0 * dx);
                v[iy][ix] = v_star[iy][ix]
                    - dt * (p[iy + 1][ix] - p[iy - 1][ix]) / (2.0 * dy);
            }
        }

        // ── Boundary conditions ──────────────────────────────────────────────
        // Top lid: u = 1, v = 0
        for ix in 0..size {
            u[size - 1][ix] = 1.0;
            v[size - 1][ix] = 0.0;
        }
        // Bottom wall
        for ix in 0..size {
            u[0][ix] = 0.0;
            v[0][ix] = 0.0;
        }
        // Left and right walls
        for iy in 0..size {
            u[iy][0] = 0.0;
            v[iy][0] = 0.0;
            u[iy][size - 1] = 0.0;
            v[iy][size - 1] = 0.0;
        }
    }

    // Pack into (u, v) tuples
    let result = u
        .iter()
        .zip(v.iter())
        .map(|(u_row, v_row)| {
            u_row
                .iter()
                .zip(v_row.iter())
                .map(|(&ui, &vi)| (ui, vi))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Taylor–Green vortex
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the exact Taylor–Green vortex solution at time `t`.
///
/// The exact 2-D solution on `[0, 2π] × [0, 2π]` is:
/// ```text
/// u(x, y, t) =  sin(x) · cos(y) · exp(-2·t/Re)
/// v(x, y, t) = -cos(x) · sin(y) · exp(-2·t/Re)
/// ```
///
/// # Parameters
/// - `n`  — number of grid points per side (uniform grid on `[0, 2π)`)
/// - `t`  — evaluation time (≥ 0)
/// - `re` — Reynolds number (must be > 0)
///
/// # Returns
/// `Vec<Vec<(f64, f64)>>` with shape `[n][n]`; each entry is `(u, v)`.
pub fn taylor_green_vortex(n: usize, t: f64, re: f64) -> Result<Vec<Vec<(f64, f64)>>> {
    if n < 2 {
        return Err(DatasetsError::InvalidFormat(
            "taylor_green_vortex: n must be >= 2".into(),
        ));
    }
    if t < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "taylor_green_vortex: t must be >= 0".into(),
        ));
    }
    if re <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "taylor_green_vortex: re must be > 0".into(),
        ));
    }

    let decay = (-2.0 * t / re).exp();
    let mut grid = vec![vec![(0.0f64, 0.0f64); n]; n];

    for iy in 0..n {
        let y = 2.0 * PI * (iy as f64) / (n as f64);
        for ix in 0..n {
            let x = 2.0 * PI * (ix as f64) / (n as f64);
            let u = x.sin() * y.cos() * decay;
            let v = -x.cos() * y.sin() * decay;
            grid[iy][ix] = (u, v);
        }
    }
    Ok(grid)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poiseuille_shape() {
        let g = poiseuille_flow(10, 8, -1.0, 1.0).expect("valid params");
        assert_eq!(g.len(), 8);
        assert_eq!(g[0].len(), 10);
    }

    #[test]
    fn test_poiseuille_parabolic_profile() {
        // Velocity should be zero at walls and maximum at centre.
        let g = poiseuille_flow(5, 50, -1.0, 1.0).expect("valid params");
        let bottom = g[0][0];
        let top = g[49][0];
        let center = g[25][0];
        // Wall velocities ≈ 0 (small because grid isn't exactly at y=±0.5)
        assert!(bottom.abs() < 0.1, "bottom wall u = {bottom}");
        assert!(top.abs() < 0.1, "top wall u = {top}");
        // Centre velocity is positive (dp_dx < 0 ⟹ positive flow)
        assert!(center > 0.1, "centre u should be positive: {center}");
    }

    #[test]
    fn test_poiseuille_invalid_mu() {
        assert!(poiseuille_flow(10, 10, -1.0, 0.0).is_err());
    }

    #[test]
    fn test_lid_driven_cavity_shape() {
        let vel = lid_driven_cavity_2d(8, 100.0, 5).expect("valid params");
        assert_eq!(vel.len(), 10); // n+2
        assert_eq!(vel[0].len(), 10);
    }

    #[test]
    fn test_lid_driven_cavity_lid_velocity() {
        let vel = lid_driven_cavity_2d(8, 100.0, 1).expect("valid params");
        // Top row should have u ≈ 1.
        let n = vel.len();
        for (u, _v) in &vel[n - 1] {
            assert!(
                (*u - 1.0).abs() < 1e-10,
                "lid u should be 1, got {u}"
            );
        }
    }

    #[test]
    fn test_lid_driven_cavity_invalid_n() {
        assert!(lid_driven_cavity_2d(1, 100.0, 5).is_err());
    }

    #[test]
    fn test_taylor_green_shape() {
        let g = taylor_green_vortex(16, 0.0, 1000.0).expect("valid params");
        assert_eq!(g.len(), 16);
        assert_eq!(g[0].len(), 16);
    }

    #[test]
    fn test_taylor_green_t0_values() {
        // At t=0, decay factor = 1 and the exact solution is sin/cos.
        let g = taylor_green_vortex(4, 0.0, 1000.0).expect("valid params");
        // At (ix=0, iy=0): x=0, y=0 → u = sin(0)*cos(0) = 0, v = -cos(0)*sin(0) = 0
        let (u00, v00) = g[0][0];
        assert!(u00.abs() < 1e-14, "u00 should be 0, got {u00}");
        assert!(v00.abs() < 1e-14, "v00 should be 0, got {v00}");
    }

    #[test]
    fn test_taylor_green_decay() {
        // Velocity magnitude should decrease with time.
        let g0 = taylor_green_vortex(16, 0.0, 100.0).expect("valid params");
        let g1 = taylor_green_vortex(16, 10.0, 100.0).expect("valid params");
        let mag0: f64 = g0.iter().flatten().map(|(u, v)| u * u + v * v).sum();
        let mag1: f64 = g1.iter().flatten().map(|(u, v)| u * u + v * v).sum();
        assert!(mag1 < mag0, "energy should decay over time: {mag0} -> {mag1}");
    }

    #[test]
    fn test_taylor_green_invalid_params() {
        assert!(taylor_green_vortex(1, 0.0, 100.0).is_err());
        assert!(taylor_green_vortex(8, -1.0, 100.0).is_err());
        assert!(taylor_green_vortex(8, 0.0, -1.0).is_err());
    }
}
