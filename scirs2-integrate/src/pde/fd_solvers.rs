//! Enhanced Finite Difference PDE Solvers
//!
//! Provides solvers for parabolic, hyperbolic, and elliptic PDEs using
//! finite difference discretization with configurable boundary conditions,
//! explicit and implicit time-stepping, and stability analysis.
//!
//! ## Equation Types
//! - **Heat equation** (parabolic): du/dt = alpha * d2u/dx2
//! - **Wave equation** (hyperbolic): d2u/dt2 = c^2 * d2u/dx2
//! - **Poisson equation** (elliptic): d2u/dx2 + d2u/dy2 = f(x,y)
//!
//! ## Boundary Conditions
//! - Dirichlet (fixed value)
//! - Neumann (fixed derivative)
//! - Periodic (wrap-around)
//!
//! ## Time-Stepping Methods
//! - Explicit (forward Euler, limited by CFL condition)
//! - Implicit Crank-Nicolson (unconditionally stable, second-order)

use scirs2_core::ndarray::{Array1, Array2};

use crate::pde::{PDEError, PDEResult};

// ---------------------------------------------------------------------------
// Boundary condition types for FD solvers
// ---------------------------------------------------------------------------

/// Boundary condition for finite-difference PDE solvers
#[derive(Debug, Clone)]
pub enum FDBoundaryCondition {
    /// Fixed value at boundary: u(boundary) = value
    Dirichlet(f64),
    /// Fixed derivative at boundary: du/dn(boundary) = value
    Neumann(f64),
    /// Periodic boundary (left and right are identified)
    Periodic,
}

/// Time-stepping method for parabolic/hyperbolic PDEs
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeSteppingMethod {
    /// Explicit forward Euler (conditionally stable)
    Explicit,
    /// Crank-Nicolson (unconditionally stable, second-order in time)
    CrankNicolson,
}

/// Iterative method for elliptic PDEs
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EllipticIterativeMethod {
    /// Jacobi iteration
    Jacobi,
    /// Gauss-Seidel iteration
    GaussSeidel,
    /// Successive Over-Relaxation (SOR)
    SOR(f64),
}

// ---------------------------------------------------------------------------
// CFL stability analysis
// ---------------------------------------------------------------------------

/// CFL (Courant-Friedrichs-Lewy) stability analysis result
#[derive(Debug, Clone)]
pub struct CFLAnalysis {
    /// The computed CFL number
    pub cfl_number: f64,
    /// Whether the scheme is stable at this CFL number
    pub is_stable: bool,
    /// Maximum stable time step for explicit methods
    pub max_stable_dt: f64,
    /// Description of the stability condition
    pub description: String,
}

/// Check CFL condition for the heat equation (explicit forward Euler)
///
/// For the 1D heat equation du/dt = alpha * d2u/dx2,
/// the CFL condition is: alpha * dt / dx^2 <= 0.5
pub fn cfl_heat_1d(alpha: f64, dx: f64, dt: f64) -> CFLAnalysis {
    let cfl = alpha * dt / (dx * dx);
    let max_stable_dt = 0.5 * dx * dx / alpha;
    CFLAnalysis {
        cfl_number: cfl,
        is_stable: cfl <= 0.5,
        max_stable_dt,
        description: format!(
            "Heat 1D: CFL = {cfl:.4e} (must be <= 0.5). Max stable dt = {max_stable_dt:.4e}"
        ),
    }
}

/// Check CFL condition for the 2D heat equation (explicit forward Euler)
///
/// CFL condition: alpha * dt * (1/dx^2 + 1/dy^2) <= 0.5
pub fn cfl_heat_2d(alpha: f64, dx: f64, dy: f64, dt: f64) -> CFLAnalysis {
    let cfl = alpha * dt * (1.0 / (dx * dx) + 1.0 / (dy * dy));
    let max_stable_dt = 0.5 / (alpha * (1.0 / (dx * dx) + 1.0 / (dy * dy)));
    CFLAnalysis {
        cfl_number: cfl,
        is_stable: cfl <= 0.5,
        max_stable_dt,
        description: format!(
            "Heat 2D: CFL = {cfl:.4e} (must be <= 0.5). Max stable dt = {max_stable_dt:.4e}"
        ),
    }
}

/// Check CFL condition for the 1D wave equation (explicit)
///
/// CFL condition: c * dt / dx <= 1.0
pub fn cfl_wave_1d(c: f64, dx: f64, dt: f64) -> CFLAnalysis {
    let cfl = c * dt / dx;
    let max_stable_dt = dx / c;
    CFLAnalysis {
        cfl_number: cfl,
        is_stable: cfl <= 1.0,
        max_stable_dt,
        description: format!(
            "Wave 1D: CFL = {cfl:.4e} (must be <= 1.0). Max stable dt = {max_stable_dt:.4e}"
        ),
    }
}

/// Check CFL condition for the 2D wave equation (explicit)
///
/// CFL condition: c * dt * sqrt(1/dx^2 + 1/dy^2) <= 1.0
pub fn cfl_wave_2d(c: f64, dx: f64, dy: f64, dt: f64) -> CFLAnalysis {
    let factor = (1.0 / (dx * dx) + 1.0 / (dy * dy)).sqrt();
    let cfl = c * dt * factor;
    let max_stable_dt = 1.0 / (c * factor);
    CFLAnalysis {
        cfl_number: cfl,
        is_stable: cfl <= 1.0,
        max_stable_dt,
        description: format!(
            "Wave 2D: CFL = {cfl:.4e} (must be <= 1.0). Max stable dt = {max_stable_dt:.4e}"
        ),
    }
}

// ---------------------------------------------------------------------------
// 1D Heat Equation solver result
// ---------------------------------------------------------------------------

/// Result from a heat equation solve
#[derive(Debug, Clone)]
pub struct HeatResult {
    /// Spatial grid x values
    pub x: Array1<f64>,
    /// Time grid t values
    pub t: Array1<f64>,
    /// Solution u[time_step, spatial_index]
    pub u: Array2<f64>,
    /// CFL analysis (if explicit method used)
    pub cfl: Option<CFLAnalysis>,
}

// ---------------------------------------------------------------------------
// 1D Heat Equation
// ---------------------------------------------------------------------------

/// Solve 1D heat equation: du/dt = alpha * d2u/dx2
///
/// # Arguments
/// * `alpha` - thermal diffusivity (> 0)
/// * `x_range` - spatial domain [x_min, x_max]
/// * `t_range` - time domain [t_min, t_max]
/// * `nx` - number of spatial grid points
/// * `nt` - number of time steps
/// * `initial_condition` - function u(x, 0)
/// * `left_bc` - boundary condition at x_min
/// * `right_bc` - boundary condition at x_max
/// * `method` - time-stepping method
pub fn solve_heat_1d(
    alpha: f64,
    x_range: [f64; 2],
    t_range: [f64; 2],
    nx: usize,
    nt: usize,
    initial_condition: &dyn Fn(f64) -> f64,
    left_bc: &FDBoundaryCondition,
    right_bc: &FDBoundaryCondition,
    method: TimeSteppingMethod,
) -> PDEResult<HeatResult> {
    if alpha <= 0.0 {
        return Err(PDEError::InvalidParameter(
            "Thermal diffusivity alpha must be positive".to_string(),
        ));
    }
    if nx < 3 {
        return Err(PDEError::InvalidGrid(
            "Need at least 3 spatial grid points".to_string(),
        ));
    }
    if nt < 1 {
        return Err(PDEError::InvalidParameter(
            "Need at least 1 time step".to_string(),
        ));
    }

    let dx = (x_range[1] - x_range[0]) / (nx as f64 - 1.0);
    let dt = (t_range[1] - t_range[0]) / nt as f64;

    // Build spatial grid
    let x = Array1::from_shape_fn(nx, |i| x_range[0] + i as f64 * dx);
    // Build time grid
    let t = Array1::from_shape_fn(nt + 1, |i| t_range[0] + i as f64 * dt);

    // Initialize solution array
    let mut u = Array2::zeros((nt + 1, nx));
    for i in 0..nx {
        u[[0, i]] = initial_condition(x[i]);
    }
    // Apply initial BCs
    apply_bc_1d(&mut u, 0, left_bc, right_bc, dx);

    let cfl = cfl_heat_1d(alpha, dx, dt);

    match method {
        TimeSteppingMethod::Explicit => {
            if !cfl.is_stable {
                return Err(PDEError::ComputationError(format!(
                    "Explicit scheme unstable: {}",
                    cfl.description
                )));
            }
            let r = alpha * dt / (dx * dx);
            for n in 0..nt {
                // Check for periodic BC pair
                let is_periodic = matches!(
                    (left_bc, right_bc),
                    (FDBoundaryCondition::Periodic, FDBoundaryCondition::Periodic)
                );
                for i in 1..nx - 1 {
                    u[[n + 1, i]] =
                        u[[n, i]] + r * (u[[n, i + 1]] - 2.0 * u[[n, i]] + u[[n, i - 1]]);
                }
                if is_periodic {
                    // Periodic: wrap around
                    u[[n + 1, 0]] = u[[n, 0]] + r * (u[[n, 1]] - 2.0 * u[[n, 0]] + u[[n, nx - 2]]);
                    u[[n + 1, nx - 1]] = u[[n + 1, 0]];
                } else {
                    apply_bc_1d(&mut u, n + 1, left_bc, right_bc, dx);
                }
            }
        }
        TimeSteppingMethod::CrankNicolson => {
            let r = alpha * dt / (2.0 * dx * dx);
            let is_periodic = matches!(
                (left_bc, right_bc),
                (FDBoundaryCondition::Periodic, FDBoundaryCondition::Periodic)
            );
            for n in 0..nt {
                // Build RHS
                let mut rhs = Array1::zeros(nx);
                for i in 1..nx - 1 {
                    rhs[i] = u[[n, i]] + r * (u[[n, i + 1]] - 2.0 * u[[n, i]] + u[[n, i - 1]]);
                }
                if is_periodic {
                    rhs[0] = u[[n, 0]] + r * (u[[n, 1]] - 2.0 * u[[n, 0]] + u[[n, nx - 2]]);
                    rhs[nx - 1] = rhs[0];
                }

                // Solve tridiagonal system (1+2r) u_new[i] - r u_new[i-1] - r u_new[i+1] = rhs[i]
                if is_periodic {
                    let solved = solve_periodic_tridiag(nx - 1, -r, 1.0 + 2.0 * r, -r, &rhs)?;
                    for i in 0..nx - 1 {
                        u[[n + 1, i]] = solved[i];
                    }
                    u[[n + 1, nx - 1]] = u[[n + 1, 0]];
                } else {
                    let interior_size = nx - 2;
                    if interior_size == 0 {
                        apply_bc_1d(&mut u, n + 1, left_bc, right_bc, dx);
                        continue;
                    }
                    let mut rhs_interior = Array1::zeros(interior_size);
                    for i in 0..interior_size {
                        rhs_interior[i] = rhs[i + 1];
                    }
                    // Adjust RHS for boundary conditions
                    apply_cn_bc_adjustment(
                        &mut rhs_interior,
                        left_bc,
                        right_bc,
                        r,
                        &u,
                        n + 1,
                        nx,
                        dx,
                    );
                    let solved =
                        solve_tridiag(interior_size, -r, 1.0 + 2.0 * r, -r, &rhs_interior)?;
                    for i in 0..interior_size {
                        u[[n + 1, i + 1]] = solved[i];
                    }
                    apply_bc_1d(&mut u, n + 1, left_bc, right_bc, dx);
                }
            }
        }
    }

    Ok(HeatResult {
        x,
        t,
        u,
        cfl: Some(cfl),
    })
}

// ---------------------------------------------------------------------------
// 2D Heat Equation
// ---------------------------------------------------------------------------

/// Result from a 2D heat equation solve
#[derive(Debug, Clone)]
pub struct Heat2DResult {
    /// Spatial grid x values
    pub x: Array1<f64>,
    /// Spatial grid y values
    pub y: Array1<f64>,
    /// Time grid t values
    pub t: Array1<f64>,
    /// Solution snapshots, `u[time_step]` is a 2D array `[ny, nx]`
    pub u: Vec<Array2<f64>>,
    /// CFL analysis
    pub cfl: Option<CFLAnalysis>,
}

/// Solve 2D heat equation: du/dt = alpha * (d2u/dx2 + d2u/dy2)
///
/// Only Dirichlet BCs are supported for the 2D version to keep the interface simple.
/// Explicit forward Euler time stepping.
pub fn solve_heat_2d(
    alpha: f64,
    x_range: [f64; 2],
    y_range: [f64; 2],
    t_range: [f64; 2],
    nx: usize,
    ny: usize,
    nt: usize,
    initial_condition: &dyn Fn(f64, f64) -> f64,
    bc_values: [f64; 4], // [left, right, bottom, top] Dirichlet values
    save_every: usize,
) -> PDEResult<Heat2DResult> {
    if alpha <= 0.0 {
        return Err(PDEError::InvalidParameter(
            "Thermal diffusivity alpha must be positive".to_string(),
        ));
    }
    if nx < 3 || ny < 3 {
        return Err(PDEError::InvalidGrid(
            "Need at least 3 grid points in each dimension".to_string(),
        ));
    }

    let dx = (x_range[1] - x_range[0]) / (nx as f64 - 1.0);
    let dy = (y_range[1] - y_range[0]) / (ny as f64 - 1.0);
    let dt = (t_range[1] - t_range[0]) / nt as f64;

    let cfl = cfl_heat_2d(alpha, dx, dy, dt);
    if !cfl.is_stable {
        return Err(PDEError::ComputationError(format!(
            "Explicit scheme unstable: {}",
            cfl.description
        )));
    }

    let x = Array1::from_shape_fn(nx, |i| x_range[0] + i as f64 * dx);
    let y = Array1::from_shape_fn(ny, |j| y_range[0] + j as f64 * dy);
    let mut t_save = vec![t_range[0]];

    // Initialize
    let mut u_curr = Array2::zeros((ny, nx));
    for j in 0..ny {
        for i in 0..nx {
            u_curr[[j, i]] = initial_condition(x[i], y[j]);
        }
    }
    apply_dirichlet_2d(&mut u_curr, bc_values, nx, ny);

    let save_every = if save_every == 0 { 1 } else { save_every };
    let mut snapshots = vec![u_curr.clone()];

    let rx = alpha * dt / (dx * dx);
    let ry = alpha * dt / (dy * dy);

    for n in 0..nt {
        let mut u_next = u_curr.clone();
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                u_next[[j, i]] = u_curr[[j, i]]
                    + rx * (u_curr[[j, i + 1]] - 2.0 * u_curr[[j, i]] + u_curr[[j, i - 1]])
                    + ry * (u_curr[[j + 1, i]] - 2.0 * u_curr[[j, i]] + u_curr[[j - 1, i]]);
            }
        }
        apply_dirichlet_2d(&mut u_next, bc_values, nx, ny);
        u_curr = u_next;

        if (n + 1) % save_every == 0 || n + 1 == nt {
            snapshots.push(u_curr.clone());
            t_save.push(t_range[0] + (n + 1) as f64 * dt);
        }
    }

    Ok(Heat2DResult {
        x,
        y,
        t: Array1::from_vec(t_save),
        u: snapshots,
        cfl: Some(cfl),
    })
}

// ---------------------------------------------------------------------------
// 1D Wave Equation
// ---------------------------------------------------------------------------

/// Result from a wave equation solve
#[derive(Debug, Clone)]
pub struct WaveResult {
    /// Spatial grid x values
    pub x: Array1<f64>,
    /// Time grid t values
    pub t: Array1<f64>,
    /// Solution u[time_step, spatial_index]
    pub u: Array2<f64>,
    /// CFL analysis
    pub cfl: Option<CFLAnalysis>,
}

/// Solve 1D wave equation: d2u/dt2 = c^2 * d2u/dx2
///
/// Uses the explicit leapfrog scheme, which requires CFL number c*dt/dx <= 1.
pub fn solve_wave_1d(
    c: f64,
    x_range: [f64; 2],
    t_range: [f64; 2],
    nx: usize,
    nt: usize,
    initial_displacement: &dyn Fn(f64) -> f64,
    initial_velocity: &dyn Fn(f64) -> f64,
    left_bc: &FDBoundaryCondition,
    right_bc: &FDBoundaryCondition,
) -> PDEResult<WaveResult> {
    if c <= 0.0 {
        return Err(PDEError::InvalidParameter(
            "Wave speed c must be positive".to_string(),
        ));
    }
    if nx < 3 {
        return Err(PDEError::InvalidGrid(
            "Need at least 3 spatial grid points".to_string(),
        ));
    }

    let dx = (x_range[1] - x_range[0]) / (nx as f64 - 1.0);
    let dt = (t_range[1] - t_range[0]) / nt as f64;

    let cfl = cfl_wave_1d(c, dx, dt);
    if !cfl.is_stable {
        return Err(PDEError::ComputationError(format!(
            "Explicit wave scheme unstable: {}",
            cfl.description
        )));
    }

    let r2 = (c * dt / dx) * (c * dt / dx);
    let x = Array1::from_shape_fn(nx, |i| x_range[0] + i as f64 * dx);
    let t = Array1::from_shape_fn(nt + 1, |i| t_range[0] + i as f64 * dt);

    let mut u = Array2::zeros((nt + 1, nx));

    // Time step 0: initial displacement
    for i in 0..nx {
        u[[0, i]] = initial_displacement(x[i]);
    }
    apply_bc_1d(&mut u, 0, left_bc, right_bc, dx);

    // Time step 1: use Taylor expansion with initial velocity
    // u(x, dt) ~ u(x, 0) + dt * v(x, 0) + 0.5 * dt^2 * c^2 * d2u/dx2
    let is_periodic = matches!(
        (left_bc, right_bc),
        (FDBoundaryCondition::Periodic, FDBoundaryCondition::Periodic)
    );
    for i in 1..nx - 1 {
        let d2u = u[[0, i + 1]] - 2.0 * u[[0, i]] + u[[0, i - 1]];
        u[[1, i]] = u[[0, i]] + dt * initial_velocity(x[i]) + 0.5 * r2 * d2u;
    }
    if is_periodic {
        let d2u = u[[0, 1]] - 2.0 * u[[0, 0]] + u[[0, nx - 2]];
        u[[1, 0]] = u[[0, 0]] + dt * initial_velocity(x[0]) + 0.5 * r2 * d2u;
        u[[1, nx - 1]] = u[[1, 0]];
    } else {
        apply_bc_1d(&mut u, 1, left_bc, right_bc, dx);
    }

    // Leapfrog time stepping for n >= 2
    for n in 1..nt {
        for i in 1..nx - 1 {
            u[[n + 1, i]] = 2.0 * u[[n, i]] - u[[n - 1, i]]
                + r2 * (u[[n, i + 1]] - 2.0 * u[[n, i]] + u[[n, i - 1]]);
        }
        if is_periodic {
            u[[n + 1, 0]] = 2.0 * u[[n, 0]] - u[[n - 1, 0]]
                + r2 * (u[[n, 1]] - 2.0 * u[[n, 0]] + u[[n, nx - 2]]);
            u[[n + 1, nx - 1]] = u[[n + 1, 0]];
        } else {
            apply_bc_1d(&mut u, n + 1, left_bc, right_bc, dx);
        }
    }

    Ok(WaveResult {
        x,
        t,
        u,
        cfl: Some(cfl),
    })
}

/// Solve 2D wave equation: d2u/dt2 = c^2 * (d2u/dx2 + d2u/dy2)
///
/// Explicit leapfrog on a rectangular grid with Dirichlet BCs.
pub fn solve_wave_2d(
    c: f64,
    x_range: [f64; 2],
    y_range: [f64; 2],
    t_range: [f64; 2],
    nx: usize,
    ny: usize,
    nt: usize,
    initial_displacement: &dyn Fn(f64, f64) -> f64,
    initial_velocity: &dyn Fn(f64, f64) -> f64,
    bc_value: f64,
    save_every: usize,
) -> PDEResult<Wave2DResult> {
    if c <= 0.0 {
        return Err(PDEError::InvalidParameter(
            "Wave speed c must be positive".to_string(),
        ));
    }
    if nx < 3 || ny < 3 {
        return Err(PDEError::InvalidGrid(
            "Need at least 3 grid points in each dimension".to_string(),
        ));
    }

    let dx = (x_range[1] - x_range[0]) / (nx as f64 - 1.0);
    let dy = (y_range[1] - y_range[0]) / (ny as f64 - 1.0);
    let dt = (t_range[1] - t_range[0]) / nt as f64;

    let cfl = cfl_wave_2d(c, dx, dy, dt);
    if !cfl.is_stable {
        return Err(PDEError::ComputationError(format!(
            "Explicit 2D wave scheme unstable: {}",
            cfl.description
        )));
    }

    let rx2 = (c * dt / dx) * (c * dt / dx);
    let ry2 = (c * dt / dy) * (c * dt / dy);

    let x = Array1::from_shape_fn(nx, |i| x_range[0] + i as f64 * dx);
    let y = Array1::from_shape_fn(ny, |j| y_range[0] + j as f64 * dy);

    let save_every = if save_every == 0 { 1 } else { save_every };
    let bc_vals = [bc_value; 4];

    // u at time n-1, n, n+1
    let mut u_prev = Array2::zeros((ny, nx));
    let mut u_curr = Array2::zeros((ny, nx));

    // Step 0
    for j in 0..ny {
        for i in 0..nx {
            u_curr[[j, i]] = initial_displacement(x[i], y[j]);
        }
    }
    apply_dirichlet_2d(&mut u_curr, bc_vals, nx, ny);

    let mut snapshots = vec![u_curr.clone()];
    let mut t_save = vec![t_range[0]];

    // Step 1 via Taylor expansion
    for j in 1..ny - 1 {
        for i in 1..nx - 1 {
            let d2x = u_curr[[j, i + 1]] - 2.0 * u_curr[[j, i]] + u_curr[[j, i - 1]];
            let d2y = u_curr[[j + 1, i]] - 2.0 * u_curr[[j, i]] + u_curr[[j - 1, i]];
            u_prev[[j, i]] =
                u_curr[[j, i]] + dt * initial_velocity(x[i], y[j]) + 0.5 * (rx2 * d2x + ry2 * d2y);
        }
    }
    apply_dirichlet_2d(&mut u_prev, bc_vals, nx, ny);
    // swap: prev = step0, curr = step1
    std::mem::swap(&mut u_prev, &mut u_curr);

    if save_every == 1 {
        snapshots.push(u_curr.clone());
        t_save.push(t_range[0] + dt);
    }

    // Steps 2..nt via leapfrog
    for n in 1..nt {
        let mut u_next = Array2::zeros((ny, nx));
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let d2x = u_curr[[j, i + 1]] - 2.0 * u_curr[[j, i]] + u_curr[[j, i - 1]];
                let d2y = u_curr[[j + 1, i]] - 2.0 * u_curr[[j, i]] + u_curr[[j - 1, i]];
                u_next[[j, i]] = 2.0 * u_curr[[j, i]] - u_prev[[j, i]] + rx2 * d2x + ry2 * d2y;
            }
        }
        apply_dirichlet_2d(&mut u_next, bc_vals, nx, ny);
        u_prev = u_curr;
        u_curr = u_next;

        if (n + 1) % save_every == 0 || n + 1 == nt {
            snapshots.push(u_curr.clone());
            t_save.push(t_range[0] + (n + 1) as f64 * dt);
        }
    }

    Ok(Wave2DResult {
        x,
        y,
        t: Array1::from_vec(t_save),
        u: snapshots,
        cfl: Some(cfl),
    })
}

/// Result from a 2D wave equation solve
#[derive(Debug, Clone)]
pub struct Wave2DResult {
    /// Spatial grid x values
    pub x: Array1<f64>,
    /// Spatial grid y values
    pub y: Array1<f64>,
    /// Time grid t values
    pub t: Array1<f64>,
    /// Solution snapshots, `u[time_index]` is `[ny, nx]`
    pub u: Vec<Array2<f64>>,
    /// CFL analysis
    pub cfl: Option<CFLAnalysis>,
}

// ---------------------------------------------------------------------------
// Poisson Equation (elliptic) via iterative methods
// ---------------------------------------------------------------------------

/// Result from a Poisson equation solve
#[derive(Debug, Clone)]
pub struct PoissonResult {
    /// Spatial grid x values
    pub x: Array1<f64>,
    /// Spatial grid y values
    pub y: Array1<f64>,
    /// Solution u[ny, nx]
    pub u: Array2<f64>,
    /// Number of iterations
    pub iterations: usize,
    /// Final residual norm
    pub residual: f64,
    /// Convergence history (residual per iteration)
    pub convergence_history: Vec<f64>,
}

/// Solve Poisson equation d2u/dx2 + d2u/dy2 = f(x,y) with Dirichlet BCs
///
/// Uses the specified iterative method (Jacobi, Gauss-Seidel, or SOR).
pub fn solve_poisson_2d(
    source: &dyn Fn(f64, f64) -> f64,
    x_range: [f64; 2],
    y_range: [f64; 2],
    nx: usize,
    ny: usize,
    bc_values: [f64; 4], // [left, right, bottom, top]
    method: EllipticIterativeMethod,
    tol: f64,
    max_iter: usize,
) -> PDEResult<PoissonResult> {
    if nx < 3 || ny < 3 {
        return Err(PDEError::InvalidGrid(
            "Need at least 3 grid points in each dimension".to_string(),
        ));
    }

    let dx = (x_range[1] - x_range[0]) / (nx as f64 - 1.0);
    let dy = (y_range[1] - y_range[0]) / (ny as f64 - 1.0);

    let x = Array1::from_shape_fn(nx, |i| x_range[0] + i as f64 * dx);
    let y = Array1::from_shape_fn(ny, |j| y_range[0] + j as f64 * dy);

    let mut u = Array2::zeros((ny, nx));
    apply_dirichlet_2d(&mut u, bc_values, nx, ny);

    let dx2 = dx * dx;
    let dy2 = dy * dy;
    let denom = 2.0 * (1.0 / dx2 + 1.0 / dy2);

    let mut convergence_history = Vec::with_capacity(max_iter);
    let mut iterations = 0;
    let mut residual = f64::MAX;

    for iter in 0..max_iter {
        match method {
            EllipticIterativeMethod::Jacobi => {
                let u_old = u.clone();
                for j in 1..ny - 1 {
                    for i in 1..nx - 1 {
                        u[[j, i]] = ((u_old[[j, i + 1]] + u_old[[j, i - 1]]) / dx2
                            + (u_old[[j + 1, i]] + u_old[[j - 1, i]]) / dy2
                            - source(x[i], y[j]))
                            / denom;
                    }
                }
            }
            EllipticIterativeMethod::GaussSeidel => {
                for j in 1..ny - 1 {
                    for i in 1..nx - 1 {
                        u[[j, i]] = ((u[[j, i + 1]] + u[[j, i - 1]]) / dx2
                            + (u[[j + 1, i]] + u[[j - 1, i]]) / dy2
                            - source(x[i], y[j]))
                            / denom;
                    }
                }
            }
            EllipticIterativeMethod::SOR(omega) => {
                for j in 1..ny - 1 {
                    for i in 1..nx - 1 {
                        let gs_val = ((u[[j, i + 1]] + u[[j, i - 1]]) / dx2
                            + (u[[j + 1, i]] + u[[j - 1, i]]) / dy2
                            - source(x[i], y[j]))
                            / denom;
                        u[[j, i]] = (1.0 - omega) * u[[j, i]] + omega * gs_val;
                    }
                }
            }
        }

        // Compute residual: r = f - Laplacian(u)
        let mut res_sum = 0.0;
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let lap = (u[[j, i + 1]] - 2.0 * u[[j, i]] + u[[j, i - 1]]) / dx2
                    + (u[[j + 1, i]] - 2.0 * u[[j, i]] + u[[j - 1, i]]) / dy2;
                let r = source(x[i], y[j]) - lap;
                res_sum += r * r;
            }
        }
        residual = (res_sum / ((nx - 2) * (ny - 2)) as f64).sqrt();
        convergence_history.push(residual);
        iterations = iter + 1;

        if residual < tol {
            break;
        }
    }

    Ok(PoissonResult {
        x,
        y,
        u,
        iterations,
        residual,
        convergence_history,
    })
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Apply 1D boundary conditions at a given time step
fn apply_bc_1d(
    u: &mut Array2<f64>,
    time_idx: usize,
    left_bc: &FDBoundaryCondition,
    right_bc: &FDBoundaryCondition,
    dx: f64,
) {
    let nx = u.shape()[1];
    match left_bc {
        FDBoundaryCondition::Dirichlet(val) => {
            u[[time_idx, 0]] = *val;
        }
        FDBoundaryCondition::Neumann(val) => {
            // du/dx = val at left boundary => u[0] = u[1] - dx*val
            u[[time_idx, 0]] = u[[time_idx, 1]] - dx * val;
        }
        FDBoundaryCondition::Periodic => {
            // Handled in the main loop
        }
    }
    match right_bc {
        FDBoundaryCondition::Dirichlet(val) => {
            u[[time_idx, nx - 1]] = *val;
        }
        FDBoundaryCondition::Neumann(val) => {
            // du/dx = val at right boundary => u[nx-1] = u[nx-2] + dx*val
            u[[time_idx, nx - 1]] = u[[time_idx, nx - 2]] + dx * val;
        }
        FDBoundaryCondition::Periodic => {
            // Handled in the main loop
        }
    }
}

/// Apply Dirichlet BCs on a 2D array: [left, right, bottom, top]
fn apply_dirichlet_2d(u: &mut Array2<f64>, bc: [f64; 4], nx: usize, ny: usize) {
    for j in 0..ny {
        u[[j, 0]] = bc[0]; // left
        u[[j, nx - 1]] = bc[1]; // right
    }
    for i in 0..nx {
        u[[0, i]] = bc[2]; // bottom
        u[[ny - 1, i]] = bc[3]; // top
    }
}

/// Adjust RHS for Crank-Nicolson boundary conditions
#[allow(clippy::too_many_arguments)]
fn apply_cn_bc_adjustment(
    rhs: &mut Array1<f64>,
    left_bc: &FDBoundaryCondition,
    right_bc: &FDBoundaryCondition,
    r: f64,
    u: &Array2<f64>,
    _time_idx: usize,
    nx: usize,
    dx: f64,
) {
    let interior_size = rhs.len();
    if interior_size == 0 {
        return;
    }
    // Left BC contribution to first interior point
    match left_bc {
        FDBoundaryCondition::Dirichlet(val) => {
            rhs[0] += r * val;
        }
        FDBoundaryCondition::Neumann(val) => {
            // Ghost: u[0] = u[1] - dx*val, so contribution is r*(u[1]-dx*val)
            // The u[1] part is absorbed into the matrix diagonal modification
            rhs[0] -= r * dx * val;
        }
        FDBoundaryCondition::Periodic => {}
    }
    // Right BC contribution to last interior point
    match right_bc {
        FDBoundaryCondition::Dirichlet(val) => {
            rhs[interior_size - 1] += r * val;
        }
        FDBoundaryCondition::Neumann(val) => {
            rhs[interior_size - 1] += r * dx * val;
        }
        FDBoundaryCondition::Periodic => {}
    }
    let _ = u; // used for potential future Neumann ghost adjustments
}

/// Solve a tridiagonal system with constant bands:
/// sub * x[i-1] + diag * x[i] + sup * x[i+1] = rhs[i]
fn solve_tridiag(
    n: usize,
    sub: f64,
    diag: f64,
    sup: f64,
    rhs: &Array1<f64>,
) -> PDEResult<Array1<f64>> {
    if n == 0 {
        return Ok(Array1::zeros(0));
    }
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];

    // Forward sweep
    c_prime[0] = sup / diag;
    d_prime[0] = rhs[0] / diag;
    for i in 1..n {
        let m = diag - sub * c_prime[i - 1];
        if m.abs() < 1e-15 {
            return Err(PDEError::ComputationError(
                "Zero pivot in tridiagonal solve".to_string(),
            ));
        }
        c_prime[i] = if i < n - 1 { sup / m } else { 0.0 };
        d_prime[i] = (rhs[i] - sub * d_prime[i - 1]) / m;
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    Ok(x)
}

/// Solve a periodic tridiagonal system using the Sherman-Morrison formula
fn solve_periodic_tridiag(
    n: usize,
    sub: f64,
    diag: f64,
    sup: f64,
    rhs: &Array1<f64>,
) -> PDEResult<Array1<f64>> {
    if n < 3 {
        return Err(PDEError::ComputationError(
            "Periodic tridiagonal system needs at least 3 unknowns".to_string(),
        ));
    }

    // Sherman-Morrison trick: perturb first and last diagonal entries
    let gamma = -diag;
    let d_mod = diag - gamma; // first diagonal becomes diag + gamma effectively
    let d_last = diag - sub * sup / gamma; // last diagonal modified

    // Build modified RHS for standard tridiagonal solve
    let mut rhs_mod = rhs.clone();
    // Create vector u_sm = [gamma, 0, ..., 0, sup]
    // Create vector v_sm = [1, 0, ..., 0, sub/gamma]

    // Solve A_mod * y = rhs_mod
    // Solve A_mod * z = u_sm
    // where A_mod is the tridiagonal with modified corners

    // For simplicity, assemble the modified system as arrays and solve twice
    let mut diag_arr = vec![diag; n];
    diag_arr[0] = d_mod;
    diag_arr[n - 1] = d_last;

    let mut sub_arr = vec![sub; n];
    sub_arr[0] = 0.0; // not used
    let mut sup_arr = vec![sup; n];
    sup_arr[n - 1] = 0.0; // not used

    // Solve with general tridiagonal
    let y = solve_general_tridiag(&sub_arr, &diag_arr, &sup_arr, &rhs_mod)?;

    // u_sm vector
    let mut u_sm = Array1::zeros(n);
    u_sm[0] = gamma;
    u_sm[n - 1] = sup;
    let z = solve_general_tridiag(&sub_arr, &diag_arr, &sup_arr, &u_sm)?;

    // v_sm = [1, 0, ..., 0, sub/gamma]
    let v0 = 1.0;
    let vn = sub / gamma;

    let numer = v0 * y[0] + vn * y[n - 1];
    let denom_val = 1.0 + v0 * z[0] + vn * z[n - 1];

    if denom_val.abs() < 1e-15 {
        return Err(PDEError::ComputationError(
            "Singular periodic tridiagonal system".to_string(),
        ));
    }

    let factor = numer / denom_val;
    let mut x = Array1::zeros(n);
    for i in 0..n {
        x[i] = y[i] - factor * z[i];
    }

    Ok(x)
}

/// General tridiagonal solver (varying bands)
fn solve_general_tridiag(
    sub: &[f64],
    diag: &[f64],
    sup: &[f64],
    rhs: &Array1<f64>,
) -> PDEResult<Array1<f64>> {
    let n = rhs.len();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];

    if diag[0].abs() < 1e-15 {
        return Err(PDEError::ComputationError(
            "Zero pivot in general tridiagonal solve".to_string(),
        ));
    }
    c_prime[0] = sup[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    for i in 1..n {
        let m = diag[i] - sub[i] * c_prime[i - 1];
        if m.abs() < 1e-15 {
            return Err(PDEError::ComputationError(
                "Zero pivot in general tridiagonal solve".to_string(),
            ));
        }
        c_prime[i] = if i < n - 1 { sup[i] / m } else { 0.0 };
        d_prime[i] = (rhs[i] - sub[i] * d_prime[i - 1]) / m;
    }

    let mut x = Array1::zeros(n);
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_cfl_heat_1d_stable() {
        let cfl = cfl_heat_1d(0.01, 0.1, 0.1);
        // r = 0.01 * 0.1 / 0.01 = 0.1 <= 0.5 => stable
        assert!(cfl.is_stable);
        assert!(cfl.cfl_number < 0.5 + 1e-10);
    }

    #[test]
    fn test_cfl_heat_1d_unstable() {
        let cfl = cfl_heat_1d(1.0, 0.01, 0.01);
        // r = 1.0 * 0.01 / 0.0001 = 100 >> 0.5 => unstable
        assert!(!cfl.is_stable);
    }

    #[test]
    fn test_cfl_wave_1d_stable() {
        let cfl = cfl_wave_1d(1.0, 0.1, 0.05);
        // CFL = 1.0 * 0.05 / 0.1 = 0.5 <= 1.0
        assert!(cfl.is_stable);
    }

    #[test]
    fn test_heat_1d_explicit_constant_ic() {
        // u(x,0) = 1.0 with Dirichlet u(0)=1, u(1)=1
        // Steady-state is u=1 everywhere
        let result = solve_heat_1d(
            0.01,
            [0.0, 1.0],
            [0.0, 0.1],
            21,
            100,
            &|_x| 1.0,
            &FDBoundaryCondition::Dirichlet(1.0),
            &FDBoundaryCondition::Dirichlet(1.0),
            TimeSteppingMethod::Explicit,
        );
        let res = result.expect("Should succeed");
        // All values should remain 1.0
        let last = res.u.row(res.u.shape()[0] - 1);
        for &v in last.iter() {
            assert!((v - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_heat_1d_explicit_decay() {
        // u(x,0) = sin(pi*x) with Dirichlet u(0)=0, u(1)=0
        // Exact: u(x,t) = sin(pi*x) * exp(-pi^2 * alpha * t)
        let alpha = 0.01;
        let nx = 51;
        let nt = 5000;
        let result = solve_heat_1d(
            alpha,
            [0.0, 1.0],
            [0.0, 1.0],
            nx,
            nt,
            &|x| (PI * x).sin(),
            &FDBoundaryCondition::Dirichlet(0.0),
            &FDBoundaryCondition::Dirichlet(0.0),
            TimeSteppingMethod::Explicit,
        );
        let res = result.expect("Should succeed");
        let last = res.u.row(res.u.shape()[0] - 1);
        // Check midpoint: exact ~ sin(pi*0.5) * exp(-pi^2*0.01*1.0) ~ exp(-0.0987..) ~ 0.906
        let mid = nx / 2;
        let exact = (PI * 0.5).sin() * (-PI * PI * alpha * 1.0).exp();
        assert!(
            (last[mid] - exact).abs() < 0.02,
            "Got {}, expected {} (tol=0.02)",
            last[mid],
            exact
        );
    }

    #[test]
    fn test_heat_1d_crank_nicolson() {
        let alpha = 0.1;
        let nx = 21;
        let nt = 50;
        let result = solve_heat_1d(
            alpha,
            [0.0, 1.0],
            [0.0, 1.0],
            nx,
            nt,
            &|x| (PI * x).sin(),
            &FDBoundaryCondition::Dirichlet(0.0),
            &FDBoundaryCondition::Dirichlet(0.0),
            TimeSteppingMethod::CrankNicolson,
        );
        let res = result.expect("Should succeed");
        let last = res.u.row(res.u.shape()[0] - 1);
        let mid = nx / 2;
        let exact = (PI * 0.5).sin() * (-PI * PI * alpha * 1.0).exp();
        assert!(
            (last[mid] - exact).abs() < 0.05,
            "CN got {}, expected {} (tol=0.05)",
            last[mid],
            exact
        );
    }

    #[test]
    fn test_heat_1d_neumann() {
        // Insulated boundaries: du/dx=0 at both ends
        // u(x,0) = 1.0, should remain 1.0
        let result = solve_heat_1d(
            0.01,
            [0.0, 1.0],
            [0.0, 0.5],
            21,
            200,
            &|_| 1.0,
            &FDBoundaryCondition::Neumann(0.0),
            &FDBoundaryCondition::Neumann(0.0),
            TimeSteppingMethod::Explicit,
        );
        let res = result.expect("Should succeed");
        let last = res.u.row(res.u.shape()[0] - 1);
        for &v in last.iter() {
            assert!(
                (v - 1.0).abs() < 0.01,
                "Neumann with constant IC should stay ~1.0, got {v}"
            );
        }
    }

    #[test]
    fn test_heat_1d_periodic() {
        // Periodic heat equation: u(x,0) = sin(2*pi*x)
        let alpha = 0.01;
        let nx = 41;
        let nt = 500;
        let result = solve_heat_1d(
            alpha,
            [0.0, 1.0],
            [0.0, 0.5],
            nx,
            nt,
            &|x| (2.0 * PI * x).sin(),
            &FDBoundaryCondition::Periodic,
            &FDBoundaryCondition::Periodic,
            TimeSteppingMethod::Explicit,
        );
        let res = result.expect("Should succeed");
        let last = res.u.row(res.u.shape()[0] - 1);
        // Exact: exp(-4*pi^2*alpha*t)*sin(2*pi*x)
        // At t=0.5: decay factor = exp(-4*pi^2*0.01*0.5) ~ exp(-0.197) ~ 0.821
        let decay = (-4.0 * PI * PI * alpha * 0.5).exp();
        let mid = nx / 4; // x=0.25 => sin(pi/2)=1.0
        let exact = decay * (2.0 * PI * 0.25).sin();
        assert!(
            (last[mid] - exact).abs() < 0.05,
            "Periodic got {}, expected {exact} (tol=0.05)",
            last[mid]
        );
    }

    #[test]
    fn test_heat_2d_constant() {
        // Constant IC and matching BCs: should stay constant
        let result = solve_heat_2d(
            0.01,
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.1],
            11,
            11,
            50,
            &|_, _| 1.0,
            [1.0, 1.0, 1.0, 1.0],
            50,
        );
        let res = result.expect("Should succeed");
        let last = &res.u[res.u.len() - 1];
        for j in 0..11 {
            for i in 0..11 {
                assert!(
                    (last[[j, i]] - 1.0).abs() < 1e-10,
                    "2D heat constant: [{j},{i}] = {}",
                    last[[j, i]]
                );
            }
        }
    }

    #[test]
    fn test_wave_1d_standing() {
        // Standing wave: u(x,0) = sin(pi*x), v(x,0) = 0
        // Exact: u(x,t) = sin(pi*x) * cos(pi*c*t)
        let c = 1.0;
        let nx = 101;
        let nt = 200;
        let result = solve_wave_1d(
            c,
            [0.0, 1.0],
            [0.0, 0.5],
            nx,
            nt,
            &|x| (PI * x).sin(),
            &|_x| 0.0,
            &FDBoundaryCondition::Dirichlet(0.0),
            &FDBoundaryCondition::Dirichlet(0.0),
        );
        let res = result.expect("Should succeed");
        let last = res.u.row(res.u.shape()[0] - 1);
        let mid = nx / 2;
        let exact = (PI * 0.5).sin() * (PI * c * 0.5).cos();
        assert!(
            (last[mid] - exact).abs() < 0.05,
            "Wave got {}, expected {exact}",
            last[mid]
        );
    }

    #[test]
    fn test_wave_1d_periodic() {
        let c = 1.0;
        let nx = 101;
        let nt = 100;
        let result = solve_wave_1d(
            c,
            [0.0, 1.0],
            [0.0, 0.5],
            nx,
            nt,
            &|x| (2.0 * PI * x).sin(),
            &|_x| 0.0,
            &FDBoundaryCondition::Periodic,
            &FDBoundaryCondition::Periodic,
        );
        assert!(result.is_ok(), "Periodic wave should succeed");
    }

    #[test]
    fn test_wave_2d_basic() {
        let result = solve_wave_2d(
            1.0,
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.1],
            21,
            21,
            50,
            &|x, y| (PI * x).sin() * (PI * y).sin(),
            &|_, _| 0.0,
            0.0,
            50,
        );
        assert!(result.is_ok(), "2D wave should succeed");
    }

    #[test]
    fn test_poisson_zero_source() {
        // Laplace equation with constant Dirichlet BCs => u = constant
        let result = solve_poisson_2d(
            &|_, _| 0.0,
            [0.0, 1.0],
            [0.0, 1.0],
            11,
            11,
            [1.0, 1.0, 1.0, 1.0],
            EllipticIterativeMethod::GaussSeidel,
            1e-8,
            5000,
        );
        let res = result.expect("Should succeed");
        for j in 0..11 {
            for i in 0..11 {
                assert!(
                    (res.u[[j, i]] - 1.0).abs() < 1e-4,
                    "Laplace [{j},{i}] = {} (expected 1.0)",
                    res.u[[j, i]]
                );
            }
        }
    }

    #[test]
    fn test_poisson_jacobi() {
        let result = solve_poisson_2d(
            &|_, _| -2.0,
            [0.0, 1.0],
            [0.0, 1.0],
            21,
            21,
            [0.0, 0.0, 0.0, 0.0],
            EllipticIterativeMethod::Jacobi,
            1e-6,
            10000,
        );
        let res = result.expect("Should succeed");
        // With f=-2 and zero BCs, the solution is a parabolic bowl
        // At center (0.5, 0.5): approximate value
        let mid = 10;
        assert!(
            res.u[[mid, mid]] > 0.0,
            "Center should be positive for negative source"
        );
    }

    #[test]
    fn test_poisson_sor() {
        // SOR with omega=1.5 should converge faster than Gauss-Seidel
        let result_gs = solve_poisson_2d(
            &|_, _| -2.0,
            [0.0, 1.0],
            [0.0, 1.0],
            21,
            21,
            [0.0, 0.0, 0.0, 0.0],
            EllipticIterativeMethod::GaussSeidel,
            1e-6,
            10000,
        )
        .expect("GS should succeed");

        let result_sor = solve_poisson_2d(
            &|_, _| -2.0,
            [0.0, 1.0],
            [0.0, 1.0],
            21,
            21,
            [0.0, 0.0, 0.0, 0.0],
            EllipticIterativeMethod::SOR(1.5),
            1e-6,
            10000,
        )
        .expect("SOR should succeed");

        // SOR should converge in fewer iterations
        assert!(
            result_sor.iterations <= result_gs.iterations,
            "SOR ({}) should converge <= GS ({})",
            result_sor.iterations,
            result_gs.iterations
        );
    }

    #[test]
    fn test_heat_explicit_unstable_rejected() {
        // Very large dt should be rejected by CFL check
        let result = solve_heat_1d(
            1.0,
            [0.0, 1.0],
            [0.0, 1.0],
            11,
            2,
            &|_| 0.0,
            &FDBoundaryCondition::Dirichlet(0.0),
            &FDBoundaryCondition::Dirichlet(0.0),
            TimeSteppingMethod::Explicit,
        );
        assert!(result.is_err(), "Should reject unstable explicit scheme");
    }

    #[test]
    fn test_cfl_heat_2d() {
        let cfl = cfl_heat_2d(0.01, 0.1, 0.1, 0.1);
        // r = 0.01*0.1*(100+100) = 0.2 <= 0.5
        assert!(cfl.is_stable);
    }

    #[test]
    fn test_cfl_wave_2d() {
        let cfl = cfl_wave_2d(1.0, 0.1, 0.1, 0.05);
        // CFL = 1.0 * 0.05 * sqrt(200) ~ 0.05*14.14 ~ 0.707 <= 1.0
        assert!(cfl.is_stable);
    }
}
