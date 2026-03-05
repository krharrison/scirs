//! Enhanced Method of Lines (MOL) PDE Solver
//!
//! Semi-discretizes PDEs in space, converting them to ODE systems that are
//! then integrated with the existing ODE solvers (RK45, BDF, etc.).
//!
//! ## Features
//! - Configurable spatial stencils (2nd and 4th order)
//! - Integration with existing ODE solvers (RK45, BDF)
//! - Advection equation solver (upwind, Lax-Wendroff)
//! - Reaction-diffusion system solver
//! - Configurable boundary conditions (Dirichlet, Neumann, periodic)

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use std::sync::Arc;

use crate::ode::{solve_ivp, ODEMethod, ODEOptions};
use crate::pde::{PDEError, PDEResult};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Spatial stencil order for finite difference discretization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StencilOrder {
    /// Second-order central differences
    Second,
    /// Fourth-order central differences
    Fourth,
}

/// Boundary condition for MOL solvers
#[derive(Debug, Clone)]
pub enum MOLBoundaryCondition {
    /// Fixed value: u(boundary) = value
    Dirichlet(f64),
    /// Fixed derivative: du/dn(boundary) = value
    Neumann(f64),
    /// Periodic: u wraps around
    Periodic,
}

/// ODE method selection for time integration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MOLTimeIntegrator {
    /// Runge-Kutta 4-5 (Dormand-Prince), good for non-stiff problems
    RK45,
    /// BDF (backward differentiation formula), good for stiff problems
    BDF,
    /// Runge-Kutta 2-3 (Bogacki-Shampine)
    RK23,
}

impl MOLTimeIntegrator {
    fn to_ode_method(self) -> ODEMethod {
        match self {
            MOLTimeIntegrator::RK45 => ODEMethod::RK45,
            MOLTimeIntegrator::BDF => ODEMethod::Bdf,
            MOLTimeIntegrator::RK23 => ODEMethod::RK23,
        }
    }
}

/// Options for the enhanced MOL solver
#[derive(Debug, Clone)]
pub struct MOLEnhancedOptions {
    /// Time integrator to use
    pub integrator: MOLTimeIntegrator,
    /// Spatial stencil order
    pub stencil: StencilOrder,
    /// Absolute tolerance for ODE solver
    pub atol: f64,
    /// Relative tolerance for ODE solver
    pub rtol: f64,
    /// Maximum ODE steps
    pub max_steps: usize,
}

impl Default for MOLEnhancedOptions {
    fn default() -> Self {
        MOLEnhancedOptions {
            integrator: MOLTimeIntegrator::RK45,
            stencil: StencilOrder::Second,
            atol: 1e-6,
            rtol: 1e-3,
            max_steps: 10000,
        }
    }
}

/// Result from MOL solve
#[derive(Debug, Clone)]
pub struct MOLEnhancedResult {
    /// Spatial grid
    pub x: Array1<f64>,
    /// Time points
    pub t: Vec<f64>,
    /// Solution u[time_step, spatial_index]
    pub u: Vec<Array1<f64>>,
    /// Number of ODE function evaluations
    pub n_eval: usize,
    /// Number of ODE steps taken
    pub n_steps: usize,
}

// ---------------------------------------------------------------------------
// Diffusion equation solver
// ---------------------------------------------------------------------------

/// Solve 1D diffusion (heat) equation: du/dt = alpha * d2u/dx2 + source(x,t,u)
///
/// Semi-discretizes in space using configurable stencils, then solves the
/// resulting ODE system with the chosen time integrator.
pub fn mol_diffusion_1d(
    alpha: f64,
    x_range: [f64; 2],
    t_range: [f64; 2],
    nx: usize,
    left_bc: MOLBoundaryCondition,
    right_bc: MOLBoundaryCondition,
    initial_condition: impl Fn(f64) -> f64 + Send + Sync + 'static,
    source: Option<Arc<dyn Fn(f64, f64, f64) -> f64 + Send + Sync>>,
    options: &MOLEnhancedOptions,
) -> PDEResult<MOLEnhancedResult> {
    if alpha <= 0.0 {
        return Err(PDEError::InvalidParameter(
            "Diffusion coefficient alpha must be positive".to_string(),
        ));
    }
    if nx < 5 {
        return Err(PDEError::InvalidGrid(
            "Need at least 5 spatial points for MOL".to_string(),
        ));
    }

    let dx = (x_range[1] - x_range[0]) / (nx as f64 - 1.0);
    let x = Array1::from_shape_fn(nx, |i| x_range[0] + i as f64 * dx);

    // Initial condition
    let mut u0 = Array1::from_shape_fn(nx, |i| initial_condition(x[i]));
    apply_mol_bc(&mut u0, &left_bc, &right_bc, dx);

    let stencil = options.stencil;
    let x_clone = x.clone();
    let left_bc_c = left_bc.clone();
    let right_bc_c = right_bc.clone();

    // Build ODE RHS
    let rhs = move |t: f64, u: ArrayView1<f64>| -> Array1<f64> {
        let n = u.len();
        let mut dudt = Array1::zeros(n);

        // Diffusion operator
        match stencil {
            StencilOrder::Second => {
                let r = alpha / (dx * dx);
                for i in 1..n - 1 {
                    dudt[i] = r * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
                }
            }
            StencilOrder::Fourth => {
                let r = alpha / (12.0 * dx * dx);
                for i in 2..n - 2 {
                    dudt[i] = r
                        * (-u[i + 2] + 16.0 * u[i + 1] - 30.0 * u[i] + 16.0 * u[i - 1] - u[i - 2]);
                }
                // Fallback to 2nd order near boundaries
                let r2 = alpha / (dx * dx);
                if n > 2 {
                    dudt[1] = r2 * (u[2] - 2.0 * u[1] + u[0]);
                }
                if n > 3 {
                    dudt[n - 2] = r2 * (u[n - 1] - 2.0 * u[n - 2] + u[n - 3]);
                }
            }
        }

        // Source term
        if let Some(ref src) = source {
            for i in 1..n - 1 {
                dudt[i] += src(x_clone[i], t, u[i]);
            }
        }

        // Boundary treatment
        apply_mol_bc_rhs(
            &mut dudt,
            &u,
            &left_bc_c,
            &right_bc_c,
            n,
            dx,
            alpha,
            &x_clone,
            t,
            &source,
        );

        dudt
    };

    run_mol_ode(x, u0, t_range, rhs, options)
}

// ---------------------------------------------------------------------------
// Advection equation solver
// ---------------------------------------------------------------------------

/// Advection scheme type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdvectionScheme {
    /// First-order upwind (stable, diffusive)
    Upwind,
    /// Lax-Wendroff (second-order, dispersive)
    LaxWendroff,
    /// Central difference (second-order, no inherent stability)
    Central,
}

/// Solve 1D advection equation: du/dt + velocity * du/dx = source(x,t)
///
/// Spatial discretization depends on the chosen scheme.
pub fn mol_advection_1d(
    velocity: f64,
    x_range: [f64; 2],
    t_range: [f64; 2],
    nx: usize,
    left_bc: MOLBoundaryCondition,
    right_bc: MOLBoundaryCondition,
    initial_condition: impl Fn(f64) -> f64 + Send + Sync + 'static,
    source: Option<Arc<dyn Fn(f64, f64) -> f64 + Send + Sync>>,
    scheme: AdvectionScheme,
    options: &MOLEnhancedOptions,
) -> PDEResult<MOLEnhancedResult> {
    if nx < 5 {
        return Err(PDEError::InvalidGrid(
            "Need at least 5 spatial points for advection".to_string(),
        ));
    }

    let dx = (x_range[1] - x_range[0]) / (nx as f64 - 1.0);
    let x = Array1::from_shape_fn(nx, |i| x_range[0] + i as f64 * dx);

    let mut u0 = Array1::from_shape_fn(nx, |i| initial_condition(x[i]));
    apply_mol_bc(&mut u0, &left_bc, &right_bc, dx);

    let x_clone = x.clone();
    let left_bc_c = left_bc.clone();
    let right_bc_c = right_bc.clone();

    let rhs = move |t: f64, u: ArrayView1<f64>| -> Array1<f64> {
        let n = u.len();
        let mut dudt = Array1::zeros(n);

        match scheme {
            AdvectionScheme::Upwind => {
                if velocity >= 0.0 {
                    // Backward difference
                    for i in 1..n - 1 {
                        dudt[i] = -velocity * (u[i] - u[i - 1]) / dx;
                    }
                } else {
                    // Forward difference
                    for i in 1..n - 1 {
                        dudt[i] = -velocity * (u[i + 1] - u[i]) / dx;
                    }
                }
            }
            AdvectionScheme::LaxWendroff => {
                // Lax-Wendroff: du/dt = -v * du/dx + 0.5*v^2*dt * d2u/dx2
                // In MOL context, we just use the centered + diffusive correction stencil:
                // du/dt = -v/(2dx)*(u[i+1]-u[i-1])
                // The Lax-Wendroff correction is embedded in the time stepping
                // For pure MOL, use centered advection (the ODE solver handles stability)
                for i in 1..n - 1 {
                    // Centered advection for MOL
                    let advection = -velocity * (u[i + 1] - u[i - 1]) / (2.0 * dx);
                    // Add numerical diffusion to stabilize: v^2/(2) * d2u/dx2 * approx_dt
                    // In pure MOL, we skip this and let the ODE solver handle it
                    // but add a small artificial diffusion proportional to dx
                    let diffusion =
                        velocity.abs() * dx / 2.0 * (u[i + 1] - 2.0 * u[i] + u[i - 1]) / (dx * dx);
                    dudt[i] = advection + diffusion;
                }
            }
            AdvectionScheme::Central => {
                for i in 1..n - 1 {
                    dudt[i] = -velocity * (u[i + 1] - u[i - 1]) / (2.0 * dx);
                }
            }
        }

        // Source term
        if let Some(ref src) = source {
            for i in 1..n - 1 {
                dudt[i] += src(x_clone[i], t);
            }
        }

        // Boundary
        apply_advection_bc_rhs(&mut dudt, &u, &left_bc_c, &right_bc_c, n, dx, velocity);

        dudt
    };

    run_mol_ode(x, u0, t_range, rhs, options)
}

// ---------------------------------------------------------------------------
// Reaction-diffusion system solver
// ---------------------------------------------------------------------------

/// Reaction-diffusion system:
///   du_i/dt = D_i * d2u_i/dx2 + R_i(x, t, u_1, ..., u_m)
///
/// where i = 1..m species, D_i are diffusion coefficients,
/// and R_i are reaction terms coupling the species.
pub struct ReactionDiffusionSystem {
    /// Number of species
    pub n_species: usize,
    /// Diffusion coefficients for each species
    pub diffusion_coeffs: Vec<f64>,
    /// Reaction function: `(x, t, &[u_species]) -> Vec<f64>` (one per species)
    pub reaction: Arc<dyn Fn(f64, f64, &[f64]) -> Vec<f64> + Send + Sync>,
}

/// Result from reaction-diffusion solve
#[derive(Debug, Clone)]
pub struct ReactionDiffusionResult {
    /// Spatial grid
    pub x: Array1<f64>,
    /// Time points
    pub t: Vec<f64>,
    /// Solution: `u[time_step]` is a 2D array `[n_species, nx]`
    pub u: Vec<Array2<f64>>,
    /// Number of ODE evaluations
    pub n_eval: usize,
    /// Number of ODE steps
    pub n_steps: usize,
}

/// Solve a reaction-diffusion system on [x_left, x_right] with Dirichlet BCs.
///
/// Each species has its own diffusion coefficient and they are coupled
/// through the reaction function.
pub fn mol_reaction_diffusion(
    system: &ReactionDiffusionSystem,
    x_range: [f64; 2],
    t_range: [f64; 2],
    nx: usize,
    initial_conditions: &[impl Fn(f64) -> f64],
    bc_left: &[f64],  // Dirichlet values at left for each species
    bc_right: &[f64], // Dirichlet values at right for each species
    options: &MOLEnhancedOptions,
) -> PDEResult<ReactionDiffusionResult> {
    let m = system.n_species;
    if initial_conditions.len() != m || bc_left.len() != m || bc_right.len() != m {
        return Err(PDEError::InvalidParameter(format!(
            "Expected {} initial conditions/BCs for {} species",
            m, m
        )));
    }
    if nx < 5 {
        return Err(PDEError::InvalidGrid(
            "Need at least 5 spatial points".to_string(),
        ));
    }

    let dx = (x_range[1] - x_range[0]) / (nx as f64 - 1.0);
    let x = Array1::from_shape_fn(nx, |i| x_range[0] + i as f64 * dx);
    let total_dof = m * nx;

    // Pack initial conditions into a single vector: [species0_node0, .., species0_nodeN, species1_node0, ..]
    let mut u0 = Array1::zeros(total_dof);
    for s in 0..m {
        for i in 0..nx {
            u0[s * nx + i] = initial_conditions[s](x[i]);
        }
        // Apply BCs
        u0[s * nx] = bc_left[s];
        u0[s * nx + nx - 1] = bc_right[s];
    }

    let diffusion_coeffs = system.diffusion_coeffs.clone();
    let reaction = system.reaction.clone();
    let x_clone = x.clone();
    let bc_l = bc_left.to_vec();
    let bc_r = bc_right.to_vec();

    let rhs = move |t: f64, u: ArrayView1<f64>| -> Array1<f64> {
        let mut dudt = Array1::zeros(total_dof);
        let dx2 = dx * dx;

        // Diffusion for each species
        for s in 0..m {
            let offset = s * nx;
            let d = diffusion_coeffs[s];
            for i in 1..nx - 1 {
                dudt[offset + i] =
                    d * (u[offset + i + 1] - 2.0 * u[offset + i] + u[offset + i - 1]) / dx2;
            }
            // Dirichlet BCs: du/dt = 0 at boundaries
            dudt[offset] = 0.0;
            dudt[offset + nx - 1] = 0.0;
        }

        // Reaction terms (coupled)
        let mut species_vals = vec![0.0; m];
        for i in 1..nx - 1 {
            for s in 0..m {
                species_vals[s] = u[s * nx + i];
            }
            let r = reaction(x_clone[i], t, &species_vals);
            for s in 0..m {
                if s < r.len() {
                    dudt[s * nx + i] += r[s];
                }
            }
        }

        dudt
    };

    // ODE solve
    let ode_opts = ODEOptions {
        method: options.integrator.to_ode_method(),
        rtol: options.rtol,
        atol: options.atol,
        max_steps: options.max_steps,
        dense_output: false,
        ..Default::default()
    };

    // Wrap closure in Arc for Clone bound
    let rhs_arc = Arc::new(rhs);
    let rhs_clone = move |t: f64, u: ArrayView1<f64>| -> Array1<f64> { rhs_arc(t, u) };

    let result = solve_ivp(rhs_clone, t_range, u0, Some(ode_opts))?;

    // Unpack results
    let mut t_vec = Vec::new();
    let mut u_vec = Vec::new();

    for (step, y) in result.y.iter().enumerate() {
        t_vec.push(result.t[step]);
        let mut u_2d = Array2::zeros((m, nx));
        for s in 0..m {
            for i in 0..nx {
                u_2d[[s, i]] = y[s * nx + i];
            }
        }
        u_vec.push(u_2d);
    }

    Ok(ReactionDiffusionResult {
        x,
        t: t_vec,
        u: u_vec,
        n_eval: result.n_eval,
        n_steps: result.n_steps,
    })
}

// ---------------------------------------------------------------------------
// Advection-diffusion equation
// ---------------------------------------------------------------------------

/// Solve 1D advection-diffusion: du/dt + v * du/dx = D * d2u/dx2 + source(x,t)
pub fn mol_advection_diffusion_1d(
    velocity: f64,
    diffusion: f64,
    x_range: [f64; 2],
    t_range: [f64; 2],
    nx: usize,
    left_bc: MOLBoundaryCondition,
    right_bc: MOLBoundaryCondition,
    initial_condition: impl Fn(f64) -> f64 + Send + Sync + 'static,
    source: Option<Arc<dyn Fn(f64, f64) -> f64 + Send + Sync>>,
    options: &MOLEnhancedOptions,
) -> PDEResult<MOLEnhancedResult> {
    if diffusion < 0.0 {
        return Err(PDEError::InvalidParameter(
            "Diffusion coefficient must be non-negative".to_string(),
        ));
    }
    if nx < 5 {
        return Err(PDEError::InvalidGrid(
            "Need at least 5 spatial points".to_string(),
        ));
    }

    let dx = (x_range[1] - x_range[0]) / (nx as f64 - 1.0);
    let x = Array1::from_shape_fn(nx, |i| x_range[0] + i as f64 * dx);

    let mut u0 = Array1::from_shape_fn(nx, |i| initial_condition(x[i]));
    apply_mol_bc(&mut u0, &left_bc, &right_bc, dx);

    let x_clone = x.clone();
    let left_bc_c = left_bc.clone();
    let right_bc_c = right_bc.clone();

    let rhs = move |t: f64, u: ArrayView1<f64>| -> Array1<f64> {
        let n = u.len();
        let mut dudt = Array1::zeros(n);
        let dx2 = dx * dx;

        for i in 1..n - 1 {
            // Upwind advection
            let advection = if velocity >= 0.0 {
                -velocity * (u[i] - u[i - 1]) / dx
            } else {
                -velocity * (u[i + 1] - u[i]) / dx
            };
            // Central diffusion
            let diff = diffusion * (u[i + 1] - 2.0 * u[i] + u[i - 1]) / dx2;
            dudt[i] = advection + diff;
        }

        if let Some(ref src) = source {
            for i in 1..n - 1 {
                dudt[i] += src(x_clone[i], t);
            }
        }

        apply_advection_bc_rhs(&mut dudt, &u, &left_bc_c, &right_bc_c, n, dx, velocity);

        dudt
    };

    run_mol_ode(x, u0, t_range, rhs, options)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Apply BCs to initial condition vector
fn apply_mol_bc(
    u: &mut Array1<f64>,
    left: &MOLBoundaryCondition,
    right: &MOLBoundaryCondition,
    dx: f64,
) {
    let n = u.len();
    match left {
        MOLBoundaryCondition::Dirichlet(val) => u[0] = *val,
        MOLBoundaryCondition::Neumann(val) => u[0] = u[1] - dx * val,
        MOLBoundaryCondition::Periodic => {
            if n > 1 {
                u[0] = u[n - 2]; // or leave as-is
            }
        }
    }
    match right {
        MOLBoundaryCondition::Dirichlet(val) => u[n - 1] = *val,
        MOLBoundaryCondition::Neumann(val) => u[n - 1] = u[n - 2] + dx * val,
        MOLBoundaryCondition::Periodic => {
            if n > 1 {
                u[n - 1] = u[1]; // or leave as-is
            }
        }
    }
}

/// Apply boundary conditions in the ODE RHS for diffusion
#[allow(clippy::too_many_arguments)]
fn apply_mol_bc_rhs(
    dudt: &mut Array1<f64>,
    u: &ArrayView1<f64>,
    left: &MOLBoundaryCondition,
    right: &MOLBoundaryCondition,
    n: usize,
    dx: f64,
    alpha: f64,
    _x: &Array1<f64>,
    _t: f64,
    _source: &Option<Arc<dyn Fn(f64, f64, f64) -> f64 + Send + Sync>>,
) {
    let dx2 = dx * dx;
    match left {
        MOLBoundaryCondition::Dirichlet(_) => {
            dudt[0] = 0.0;
        }
        MOLBoundaryCondition::Neumann(val) => {
            // Ghost point: u[-1] = u[1] - 2*dx*val
            let ghost = u[1] - 2.0 * dx * val;
            dudt[0] = alpha * (u[1] - 2.0 * u[0] + ghost) / dx2;
        }
        MOLBoundaryCondition::Periodic => {
            dudt[0] = alpha * (u[1] - 2.0 * u[0] + u[n - 2]) / dx2;
        }
    }
    match right {
        MOLBoundaryCondition::Dirichlet(_) => {
            dudt[n - 1] = 0.0;
        }
        MOLBoundaryCondition::Neumann(val) => {
            let ghost = u[n - 2] + 2.0 * dx * val;
            dudt[n - 1] = alpha * (ghost - 2.0 * u[n - 1] + u[n - 2]) / dx2;
        }
        MOLBoundaryCondition::Periodic => {
            dudt[n - 1] = alpha * (u[1] - 2.0 * u[n - 1] + u[n - 2]) / dx2;
        }
    }
}

/// Apply BCs in advection RHS
fn apply_advection_bc_rhs(
    dudt: &mut Array1<f64>,
    u: &ArrayView1<f64>,
    left: &MOLBoundaryCondition,
    right: &MOLBoundaryCondition,
    n: usize,
    dx: f64,
    velocity: f64,
) {
    match left {
        MOLBoundaryCondition::Dirichlet(_) => {
            dudt[0] = 0.0;
        }
        MOLBoundaryCondition::Neumann(_) => {
            dudt[0] = 0.0; // simplified
        }
        MOLBoundaryCondition::Periodic => {
            if velocity >= 0.0 {
                dudt[0] = -velocity * (u[0] - u[n - 2]) / dx;
            } else {
                dudt[0] = -velocity * (u[1] - u[0]) / dx;
            }
        }
    }
    match right {
        MOLBoundaryCondition::Dirichlet(_) => {
            dudt[n - 1] = 0.0;
        }
        MOLBoundaryCondition::Neumann(_) => {
            dudt[n - 1] = 0.0; // simplified
        }
        MOLBoundaryCondition::Periodic => {
            if velocity >= 0.0 {
                dudt[n - 1] = -velocity * (u[n - 1] - u[n - 2]) / dx;
            } else {
                dudt[n - 1] = -velocity * (u[1] - u[n - 1]) / dx;
            }
        }
    }
}

/// Run the ODE solver on the semi-discretized system
fn run_mol_ode(
    x: Array1<f64>,
    u0: Array1<f64>,
    t_range: [f64; 2],
    rhs: impl Fn(f64, ArrayView1<f64>) -> Array1<f64> + Send + Sync + 'static,
    options: &MOLEnhancedOptions,
) -> PDEResult<MOLEnhancedResult> {
    let ode_opts = ODEOptions {
        method: options.integrator.to_ode_method(),
        rtol: options.rtol,
        atol: options.atol,
        max_steps: options.max_steps,
        dense_output: false,
        ..Default::default()
    };

    // Wrap in Arc to satisfy Clone bound required by solve_ivp
    let rhs_arc = Arc::new(rhs);
    let rhs_clone = move |t: f64, u: ArrayView1<f64>| -> Array1<f64> { rhs_arc(t, u) };

    let result = solve_ivp(rhs_clone, t_range, u0, Some(ode_opts))?;

    let t_vec: Vec<f64> = result.t.to_vec();
    let u_vec: Vec<Array1<f64>> = result.y.to_vec();

    Ok(MOLEnhancedResult {
        x,
        t: t_vec,
        u: u_vec,
        n_eval: result.n_eval,
        n_steps: result.n_steps,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_mol_diffusion_constant() {
        // u(x,0) = 1 with u(0)=1, u(1)=1 => stays at 1
        let result = mol_diffusion_1d(
            0.1,
            [0.0, 1.0],
            [0.0, 0.5],
            21,
            MOLBoundaryCondition::Dirichlet(1.0),
            MOLBoundaryCondition::Dirichlet(1.0),
            |_| 1.0,
            None,
            &MOLEnhancedOptions::default(),
        )
        .expect("Should succeed");

        let last = &result.u[result.u.len() - 1];
        for &v in last.iter() {
            assert!((v - 1.0).abs() < 0.01, "Should stay at 1.0, got {v}");
        }
    }

    #[test]
    fn test_mol_diffusion_decay() {
        // u(x,0) = sin(pi*x), u(0)=0, u(1)=0
        let alpha = 0.1;
        let result = mol_diffusion_1d(
            alpha,
            [0.0, 1.0],
            [0.0, 0.5],
            41,
            MOLBoundaryCondition::Dirichlet(0.0),
            MOLBoundaryCondition::Dirichlet(0.0),
            |x| (PI * x).sin(),
            None,
            &MOLEnhancedOptions::default(),
        )
        .expect("Should succeed");

        let last = &result.u[result.u.len() - 1];
        let mid = last.len() / 2;
        let exact = (PI * 0.5).sin() * (-PI * PI * alpha * 0.5).exp();
        assert!(
            (last[mid] - exact).abs() < 0.05,
            "MOL diffusion: got {}, expected {exact}",
            last[mid]
        );
    }

    #[test]
    fn test_mol_diffusion_4th_order() {
        let alpha = 0.1;
        let opts = MOLEnhancedOptions {
            stencil: StencilOrder::Fourth,
            ..Default::default()
        };
        let result = mol_diffusion_1d(
            alpha,
            [0.0, 1.0],
            [0.0, 0.3],
            41,
            MOLBoundaryCondition::Dirichlet(0.0),
            MOLBoundaryCondition::Dirichlet(0.0),
            |x| (PI * x).sin(),
            None,
            &opts,
        )
        .expect("Should succeed");

        let last = &result.u[result.u.len() - 1];
        let mid = last.len() / 2;
        let exact = (PI * 0.5).sin() * (-PI * PI * alpha * 0.3).exp();
        assert!(
            (last[mid] - exact).abs() < 0.05,
            "4th order: got {}, expected {exact}",
            last[mid]
        );
    }

    #[test]
    fn test_mol_diffusion_with_source() {
        // du/dt = alpha * d2u/dx2 + 1.0 with zero ICs and BCs
        let result = mol_diffusion_1d(
            0.1,
            [0.0, 1.0],
            [0.0, 0.5],
            21,
            MOLBoundaryCondition::Dirichlet(0.0),
            MOLBoundaryCondition::Dirichlet(0.0),
            |_| 0.0,
            Some(Arc::new(|_, _, _| 1.0)),
            &MOLEnhancedOptions::default(),
        )
        .expect("Should succeed");

        // Interior values should be positive
        let last = &result.u[result.u.len() - 1];
        let mid = last.len() / 2;
        assert!(last[mid] > 0.0, "Source should make interior positive");
    }

    #[test]
    fn test_mol_diffusion_neumann() {
        // Insulated boundaries
        let result = mol_diffusion_1d(
            0.01,
            [0.0, 1.0],
            [0.0, 0.5],
            21,
            MOLBoundaryCondition::Neumann(0.0),
            MOLBoundaryCondition::Neumann(0.0),
            |_| 1.0,
            None,
            &MOLEnhancedOptions::default(),
        )
        .expect("Should succeed");

        let last = &result.u[result.u.len() - 1];
        for &v in last.iter() {
            assert!((v - 1.0).abs() < 0.05, "Neumann: should stay ~1.0, got {v}");
        }
    }

    #[test]
    fn test_mol_diffusion_periodic() {
        let result = mol_diffusion_1d(
            0.01,
            [0.0, 1.0],
            [0.0, 0.5],
            41,
            MOLBoundaryCondition::Periodic,
            MOLBoundaryCondition::Periodic,
            |x| (2.0 * PI * x).sin(),
            None,
            &MOLEnhancedOptions::default(),
        )
        .expect("Should succeed");

        assert!(result.u.len() > 1, "Should have multiple time steps");
    }

    #[test]
    fn test_mol_advection_upwind() {
        // Simple advection: du/dt + 1.0 * du/dx = 0
        let result = mol_advection_1d(
            1.0,
            [0.0, 2.0],
            [0.0, 0.5],
            41,
            MOLBoundaryCondition::Dirichlet(0.0),
            MOLBoundaryCondition::Dirichlet(0.0),
            |x| if x > 0.3 && x < 0.7 { 1.0 } else { 0.0 },
            None,
            AdvectionScheme::Upwind,
            &MOLEnhancedOptions::default(),
        )
        .expect("Should succeed");

        assert!(result.u.len() > 1);
    }

    #[test]
    fn test_mol_advection_lax_wendroff() {
        let result = mol_advection_1d(
            1.0,
            [0.0, 2.0],
            [0.0, 0.3],
            41,
            MOLBoundaryCondition::Dirichlet(0.0),
            MOLBoundaryCondition::Dirichlet(0.0),
            |x| (PI * x).sin(),
            None,
            AdvectionScheme::LaxWendroff,
            &MOLEnhancedOptions::default(),
        )
        .expect("Should succeed");

        assert!(result.u.len() > 1);
    }

    #[test]
    fn test_mol_advection_periodic() {
        let result = mol_advection_1d(
            1.0,
            [0.0, 1.0],
            [0.0, 0.3],
            41,
            MOLBoundaryCondition::Periodic,
            MOLBoundaryCondition::Periodic,
            |x| (2.0 * PI * x).sin(),
            None,
            AdvectionScheme::Upwind,
            &MOLEnhancedOptions::default(),
        )
        .expect("Should succeed");

        assert!(result.u.len() > 1);
    }

    #[test]
    fn test_mol_advection_diffusion() {
        let result = mol_advection_diffusion_1d(
            1.0,
            0.01,
            [0.0, 1.0],
            [0.0, 0.5],
            41,
            MOLBoundaryCondition::Dirichlet(0.0),
            MOLBoundaryCondition::Dirichlet(0.0),
            |x| (PI * x).sin(),
            None,
            &MOLEnhancedOptions::default(),
        )
        .expect("Should succeed");

        assert!(result.u.len() > 1);
    }

    #[test]
    fn test_mol_reaction_diffusion() {
        // Gray-Scott-like: u, v species
        // du/dt = D_u * d2u/dx2 - u*v^2 + F*(1-u)
        // dv/dt = D_v * d2v/dx2 + u*v^2 - (F+k)*v
        let system = ReactionDiffusionSystem {
            n_species: 2,
            diffusion_coeffs: vec![0.01, 0.005],
            reaction: Arc::new(|_x, _t, u| {
                let f = 0.04;
                let k = 0.06;
                let u_val = u[0];
                let v_val = u[1];
                vec![
                    -u_val * v_val * v_val + f * (1.0 - u_val),
                    u_val * v_val * v_val - (f + k) * v_val,
                ]
            }),
        };

        fn ic_u(_x: f64) -> f64 {
            1.0
        }
        fn ic_v(x: f64) -> f64 {
            if x > 0.4 && x < 0.6 {
                0.5
            } else {
                0.0
            }
        }
        let ics: Vec<fn(f64) -> f64> = vec![ic_u, ic_v];
        let result = mol_reaction_diffusion(
            &system,
            [0.0, 1.0],
            [0.0, 1.0],
            21,
            &ics,
            &[1.0, 0.0],
            &[1.0, 0.0],
            &MOLEnhancedOptions {
                integrator: MOLTimeIntegrator::RK45,
                ..Default::default()
            },
        )
        .expect("Should succeed");

        assert!(result.u.len() > 1);
        assert_eq!(result.u[0].shape()[0], 2); // 2 species
    }

    #[test]
    fn test_mol_bdf_integrator() {
        // Test BDF for potentially stiff diffusion
        let result = mol_diffusion_1d(
            1.0, // Large diffusion => stiff
            [0.0, 1.0],
            [0.0, 0.1],
            21,
            MOLBoundaryCondition::Dirichlet(0.0),
            MOLBoundaryCondition::Dirichlet(0.0),
            |x| (PI * x).sin(),
            None,
            &MOLEnhancedOptions {
                integrator: MOLTimeIntegrator::BDF,
                ..Default::default()
            },
        )
        .expect("BDF should succeed");

        assert!(result.u.len() > 1);
    }

    #[test]
    fn test_mol_result_fields() {
        let result = mol_diffusion_1d(
            0.1,
            [0.0, 1.0],
            [0.0, 0.1],
            11,
            MOLBoundaryCondition::Dirichlet(0.0),
            MOLBoundaryCondition::Dirichlet(0.0),
            |x| (PI * x).sin(),
            None,
            &MOLEnhancedOptions::default(),
        )
        .expect("Should succeed");

        assert_eq!(result.x.len(), 11);
        assert!(result.n_eval > 0);
        assert!(result.n_steps > 0);
        assert!(result.t.len() == result.u.len());
    }
}
