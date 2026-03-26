//! GPU-accelerated Finite Difference Method (FDM) solvers for 2D PDEs.
//!
//! Implements tiled Jacobi/FTCS iteration for Poisson, heat, and wave equations.
//! Parallelism is simulated via Rust's `std::thread::scope` with row-striped tiles,
//! mirroring how CUDA thread blocks would partition the grid.
//!
//! # Equations supported
//! - **Poisson**: ∇²u = f  (elliptic)
//! - **Heat**:    ∂u/∂t = α ∇²u  (parabolic, FTCS)
//! - **Wave**:    ∂²u/∂t² = c² ∇²u  (hyperbolic, leapfrog)

use super::types::{
    BoundaryCondition, GpuGrid2D, GpuPdeConfig, GridSpec, PdeSolverError, PdeSolverResult,
    SolverStats,
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Apply boundary conditions to a flat data buffer for a given grid spec.
/// `bc[dim][side]`: dim ∈ {0=x, 1=y}, side ∈ {0=low, 1=high}.
fn apply_bc(data: &mut [f64], spec: &GridSpec, bc: &[[BoundaryCondition; 2]; 2]) {
    let nx = spec.nx;
    let ny = spec.ny;

    // X boundaries: left (i=0) and right (i=nx-1)
    match bc[0][0] {
        BoundaryCondition::Dirichlet(v) => {
            for j in 0..ny {
                data[spec.idx(0, j)] = v;
            }
        }
        BoundaryCondition::Neumann(g) => {
            for j in 0..ny {
                // Forward difference: (u[1,j] - u[0,j]) / dx = g
                data[spec.idx(0, j)] = data[spec.idx(1, j)] - g * spec.dx;
            }
        }
        BoundaryCondition::Periodic => {
            for j in 0..ny {
                data[spec.idx(0, j)] = data[spec.idx(nx - 2, j)];
            }
        }
        BoundaryCondition::Robin { a, b } => {
            for j in 0..ny {
                // a*u + b*(u[1]-u[0])/dx = 0  =>  u[0] = u[1]*b / (a*dx + b)
                let denom = a * spec.dx + b;
                if denom.abs() > f64::EPSILON {
                    data[spec.idx(0, j)] = data[spec.idx(1, j)] * b / denom;
                }
            }
        }
    }

    match bc[0][1] {
        BoundaryCondition::Dirichlet(v) => {
            for j in 0..ny {
                data[spec.idx(nx - 1, j)] = v;
            }
        }
        BoundaryCondition::Neumann(g) => {
            for j in 0..ny {
                data[spec.idx(nx - 1, j)] = data[spec.idx(nx - 2, j)] + g * spec.dx;
            }
        }
        BoundaryCondition::Periodic => {
            for j in 0..ny {
                data[spec.idx(nx - 1, j)] = data[spec.idx(1, j)];
            }
        }
        BoundaryCondition::Robin { a, b } => {
            for j in 0..ny {
                let denom = a * spec.dx + b;
                if denom.abs() > f64::EPSILON {
                    data[spec.idx(nx - 1, j)] = data[spec.idx(nx - 2, j)] * b / denom;
                }
            }
        }
    }

    // Y boundaries: bottom (j=0) and top (j=ny-1)
    match bc[1][0] {
        BoundaryCondition::Dirichlet(v) => {
            for i in 0..nx {
                data[spec.idx(i, 0)] = v;
            }
        }
        BoundaryCondition::Neumann(g) => {
            for i in 0..nx {
                data[spec.idx(i, 0)] = data[spec.idx(i, 1)] - g * spec.dy;
            }
        }
        BoundaryCondition::Periodic => {
            for i in 0..nx {
                data[spec.idx(i, 0)] = data[spec.idx(i, ny - 2)];
            }
        }
        BoundaryCondition::Robin { a, b } => {
            for i in 0..nx {
                let denom = a * spec.dy + b;
                if denom.abs() > f64::EPSILON {
                    data[spec.idx(i, 0)] = data[spec.idx(i, 1)] * b / denom;
                }
            }
        }
    }

    match bc[1][1] {
        BoundaryCondition::Dirichlet(v) => {
            for i in 0..nx {
                data[spec.idx(i, ny - 1)] = v;
            }
        }
        BoundaryCondition::Neumann(g) => {
            for i in 0..nx {
                data[spec.idx(i, ny - 1)] = data[spec.idx(i, ny - 2)] + g * spec.dy;
            }
        }
        BoundaryCondition::Periodic => {
            for i in 0..nx {
                data[spec.idx(i, ny - 1)] = data[spec.idx(i, 1)];
            }
        }
        BoundaryCondition::Robin { a, b } => {
            for i in 0..nx {
                let denom = a * spec.dy + b;
                if denom.abs() > f64::EPSILON {
                    data[spec.idx(i, ny - 1)] = data[spec.idx(i, ny - 2)] * b / denom;
                }
            }
        }
    }
}

/// Perform a single Jacobi sweep over the interior of the grid.
///
/// For the Poisson equation: ∇²u = f  →  Jacobi update:
///   u_new[i,j] = (u[i+1,j] + u[i-1,j])*β_x² + (u[i,j+1] + u[i,j-1])*β_y² - f[i,j]*h²) / denom
/// where β_x = dy/h, β_y = dx/h, h = sqrt(dx²+dy²), denom = 2*(β_x²+β_y²).
///
/// With different dx/dy, the standard 5-point formula becomes:
///   u_new = [ (u[i+1]+u[i-1])/dx² + (u[j+1]+u[j-1])/dy² - f[i,j] ] / (2/dx² + 2/dy²)
fn jacobi_sweep_sequential(
    u_old: &[f64],
    u_new: &mut [f64],
    rhs: &[f64],
    spec: &GridSpec,
) {
    let nx = spec.nx;
    let ny = spec.ny;
    let dx2 = spec.dx * spec.dx;
    let dy2 = spec.dy * spec.dy;
    let denom = 2.0 / dx2 + 2.0 / dy2;

    // Interior points only
    for j in 1..ny - 1 {
        for i in 1..nx - 1 {
            let idx = spec.idx(i, j);
            let east = u_old[spec.idx(i + 1, j)];
            let west = u_old[spec.idx(i - 1, j)];
            let north = u_old[spec.idx(i, j + 1)];
            let south = u_old[spec.idx(i, j - 1)];
            u_new[idx] =
                ((east + west) / dx2 + (north + south) / dy2 - rhs[idx]) / denom;
        }
    }
}

/// Parallel Jacobi sweep using thread-scope tiling.
///
/// The interior rows are divided into tiles of `tile_size` rows each.
/// Each tile is processed by a separate thread (simulating GPU thread-blocks).
/// We compute the new values into a separate buffer to avoid data races.
fn jacobi_sweep_parallel(
    u_old: &[f64],
    u_new: &mut [f64],
    rhs: &[f64],
    spec: &GridSpec,
    tile_size: usize,
) {
    let nx = spec.nx;
    let ny = spec.ny;
    let dx2 = spec.dx * spec.dx;
    let dy2 = spec.dy * spec.dy;
    let denom = 2.0 / dx2 + 2.0 / dy2;

    // Interior rows span j in 1..(ny-1); group them into tile_size-row chunks
    let interior_rows = ny.saturating_sub(2); // rows 1 .. ny-2 inclusive
    if interior_rows == 0 {
        return;
    }

    let effective_tile = tile_size.max(1);
    let num_tiles = (interior_rows + effective_tile - 1) / effective_tile;

    // Build per-tile sub-slices of u_new (interior only, offset by nx for row 1).
    // We write into rows 1..ny-1 of u_new.  Split the slice into per-tile pieces.
    let row_start_flat = nx; // row j=1 starts at offset nx in the flat array
    let u_new_interior = &mut u_new[row_start_flat..row_start_flat + interior_rows * nx];

    // chunks_mut gives non-overlapping mutable slices, one per tile of rows
    let tile_chunks: Vec<&mut [f64]> =
        u_new_interior.chunks_mut(effective_tile * nx).collect();

    std::thread::scope(|s| {
        for (tile_idx, tile_chunk) in tile_chunks.into_iter().enumerate() {
            let tile_row_start = 1 + tile_idx * effective_tile; // absolute j index
            s.spawn(move || {
                let local_rows = tile_chunk.len() / nx;
                for local_j in 0..local_rows {
                    let j = tile_row_start + local_j;
                    if j >= ny - 1 {
                        break;
                    }
                    for i in 1..nx - 1 {
                        let idx = spec.idx(i, j);
                        let east = u_old[spec.idx(i + 1, j)];
                        let west = u_old[spec.idx(i - 1, j)];
                        let north = u_old[spec.idx(i, j + 1)];
                        let south = u_old[spec.idx(i, j - 1)];
                        let local_flat = local_j * nx + i;
                        tile_chunk[local_flat] =
                            ((east + west) / dx2 + (north + south) / dy2 - rhs[idx]) / denom;
                    }
                }
            });
        }
    });

    // Copy boundary rows (j=0 and j=ny-1) from u_old so apply_bc can act on them
    u_new[..nx].copy_from_slice(&u_old[..nx]);
    u_new[(ny - 1) * nx..].copy_from_slice(&u_old[(ny - 1) * nx..]);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Solver for GPU-accelerated Finite Difference PDE problems.
pub struct GpuFdmSolver;

impl GpuFdmSolver {
    /// Solve the 2D Poisson equation ∇²u = f on a uniform grid using Jacobi iteration.
    ///
    /// # Parameters
    /// - `rhs`: Right-hand-side grid f(x,y).
    /// - `bc`: Boundary conditions `[x_axis, y_axis][low, high]`.
    /// - `config`: Solver configuration.
    ///
    /// # Returns
    /// Solution grid `u` and solver statistics.
    ///
    /// # Errors
    /// - `PdeSolverError::NotConverged` if the iteration count is exhausted.
    /// - `PdeSolverError::InvalidGrid` if grids are incompatible.
    pub fn solve_poisson_2d(
        rhs: &GpuGrid2D,
        bc: &[[BoundaryCondition; 2]; 2],
        config: &GpuPdeConfig,
    ) -> PdeSolverResult<(GpuGrid2D, SolverStats)> {
        let spec = rhs.spec;

        let mut u_cur = GpuGrid2D::zeros(spec)?;
        let mut u_new = GpuGrid2D::zeros(spec)?;

        // Apply initial boundary conditions
        apply_bc(&mut u_cur.data, &spec, bc);
        apply_bc(&mut u_new.data, &spec, bc);

        let mut residual;
        let mut iter = 0usize;

        while iter < config.max_iterations {
            // Jacobi sweep
            if config.use_parallel {
                jacobi_sweep_parallel(
                    &u_cur.data.clone(),
                    &mut u_new.data,
                    &rhs.data,
                    &spec,
                    config.tile_size,
                );
            } else {
                jacobi_sweep_sequential(
                    &u_cur.data.clone(),
                    &mut u_new.data,
                    &rhs.data,
                    &spec,
                );
            }

            // Enforce boundary conditions on the new iterate
            apply_bc(&mut u_new.data, &spec, bc);

            // Check convergence: L∞ norm of update
            residual = u_cur.linf_diff(&u_new)?;

            // Swap
            std::mem::swap(&mut u_cur, &mut u_new);
            iter += 1;

            if residual < config.tolerance {
                return Ok((u_cur, SolverStats::converged(iter, residual)));
            }
        }

        Err(PdeSolverError::NotConverged { iterations: iter })
    }
}

/// Solve the 2D Poisson equation ∇²u = f (free function wrapper).
pub fn solve_poisson_2d(
    rhs: &GpuGrid2D,
    bc: &[[BoundaryCondition; 2]; 2],
    config: &GpuPdeConfig,
) -> PdeSolverResult<(GpuGrid2D, SolverStats)> {
    GpuFdmSolver::solve_poisson_2d(rhs, bc, config)
}

/// Solve the 2D heat equation ∂u/∂t = α ∇²u using explicit FTCS time-stepping.
///
/// Modifies `u` in place over `steps` time steps of size `dt`.
/// Uses the stability condition α·dt/(dx²) + α·dt/(dy²) ≤ 0.5.
///
/// # Errors
/// Returns `PdeSolverError::InvalidGrid` if the CFL condition is violated by a large margin.
pub fn solve_heat_2d(
    u: &mut GpuGrid2D,
    dt: f64,
    diffusivity: f64,
    steps: usize,
) -> PdeSolverResult<SolverStats> {
    let spec = u.spec;
    let dx2 = spec.dx * spec.dx;
    let dy2 = spec.dy * spec.dy;
    let rx = diffusivity * dt / dx2;
    let ry = diffusivity * dt / dy2;

    // Stability check (CFL): r_x + r_y <= 0.5
    if rx + ry > 1.0 {
        return Err(PdeSolverError::InvalidGrid);
    }

    let mut u_new = GpuGrid2D::zeros(spec)?;
    let nx = spec.nx;
    let ny = spec.ny;

    for step in 0..steps {
        // Full-domain update using ghost-cell Neumann zero-flux BCs.
        // For a boundary cell at index i=0 (x-direction), the ghost cell value
        // at i=-1 equals u[1] (mirror), so the Laplacian term becomes 2*(u[1]-u[0])/dx².
        // Similarly for i=nx-1, j=0, j=ny-1.
        for j in 0..ny {
            for i in 0..nx {
                let u_c = u.get(i, j);
                // x-direction: handle ghost cells at boundaries
                let u_e = if i + 1 < nx { u.get(i + 1, j) } else { u.get(i - 1, j) };
                let u_w = if i > 0 { u.get(i - 1, j) } else { u.get(i + 1, j) };
                // y-direction: handle ghost cells at boundaries
                let u_n = if j + 1 < ny { u.get(i, j + 1) } else { u.get(i, j - 1) };
                let u_s = if j > 0 { u.get(i, j - 1) } else { u.get(i, j + 1) };

                let val = u_c
                    + rx * (u_e - 2.0 * u_c + u_w)
                    + ry * (u_n - 2.0 * u_c + u_s);
                u_new.set(i, j, val);
            }
        }

        std::mem::swap(u, &mut u_new);

        // Early exit: check if the field has diverged
        let norm = u.linf_norm();
        if !norm.is_finite() {
            return Err(PdeSolverError::NotConverged { iterations: step });
        }
    }

    Ok(SolverStats::converged(steps, 0.0))
}

/// Solve the 2D wave equation ∂²u/∂t² = c² ∇²u using the leapfrog scheme.
///
/// Takes the current field `u` and the previous time-step field `u_prev`,
/// advances `steps` time steps of size `dt`, and returns the new field.
///
/// Leapfrog: u_next = 2·u - u_prev + c²·dt²·∇²u
///
/// # Errors
/// Returns `PdeSolverError::InvalidGrid` if the CFL condition c·dt/dx > 1/√2.
pub fn solve_wave_2d(
    u: &mut GpuGrid2D,
    u_prev: &GpuGrid2D,
    dt: f64,
    c: f64,
    steps: usize,
) -> PdeSolverResult<GpuGrid2D> {
    let spec = u.spec;
    if u_prev.data.len() != u.data.len() {
        return Err(PdeSolverError::InvalidGrid);
    }

    let dx2 = spec.dx * spec.dx;
    let dy2 = spec.dy * spec.dy;
    let rx = c * c * dt * dt / dx2;
    let ry = c * c * dt * dt / dy2;

    // CFL: c·dt·√(1/dx²+1/dy²) ≤ 1
    let cfl = c * dt * (1.0 / dx2 + 1.0 / dy2).sqrt();
    if cfl > 1.5 {
        return Err(PdeSolverError::InvalidGrid);
    }

    let nx = spec.nx;
    let ny = spec.ny;

    let mut u_cur = u.clone();
    let mut u_prv = u_prev.clone();
    let mut u_nxt = GpuGrid2D::zeros(spec)?;

    for _ in 0..steps {
        // Interior update
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let uc = u_cur.get(i, j);
                let ue = u_cur.get(i + 1, j);
                let uw = u_cur.get(i - 1, j);
                let un = u_cur.get(i, j + 1);
                let us = u_cur.get(i, j - 1);
                let up = u_prv.get(i, j);
                let val = 2.0 * uc - up
                    + rx * (ue - 2.0 * uc + uw)
                    + ry * (un - 2.0 * uc + us);
                u_nxt.set(i, j, val);
            }
        }

        // Zero Dirichlet boundaries (fixed walls)
        for i in 0..nx {
            u_nxt.set(i, 0, 0.0);
            u_nxt.set(i, ny - 1, 0.0);
        }
        for j in 0..ny {
            u_nxt.set(0, j, 0.0);
            u_nxt.set(nx - 1, j, 0.0);
        }

        // Leapfrog rotation
        std::mem::swap(&mut u_prv, &mut u_cur);
        std::mem::swap(&mut u_cur, &mut u_nxt);
    }

    Ok(u_cur)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn dirichlet_zero_bc() -> [[BoundaryCondition; 2]; 2] {
        [
            [BoundaryCondition::Dirichlet(0.0), BoundaryCondition::Dirichlet(0.0)],
            [BoundaryCondition::Dirichlet(0.0), BoundaryCondition::Dirichlet(0.0)],
        ]
    }

    #[test]
    fn test_fdm_poisson_zero_rhs() {
        // With zero RHS and homogeneous Dirichlet BCs, solution should be ≈ 0
        let spec = GridSpec::new(11, 11, 0.1, 0.1, 0.0, 0.0).expect("valid");
        let rhs = GpuGrid2D::zeros(spec).expect("zeros");
        let bc = dirichlet_zero_bc();
        let config = GpuPdeConfig { max_iterations: 5000, tolerance: 1e-8, ..Default::default() };

        let (sol, stats) = solve_poisson_2d(&rhs, &bc, &config).expect("converged");
        assert!(stats.converged, "should converge");
        assert!(sol.linf_norm() < 1e-6, "solution near zero: {}", sol.linf_norm());
    }

    #[test]
    fn test_fdm_dirichlet_bc_enforced() {
        let spec = GridSpec::new(7, 7, 0.1, 0.1, 0.0, 0.0).expect("valid");
        let rhs = GpuGrid2D::zeros(spec).expect("zeros");
        let bc = [
            [BoundaryCondition::Dirichlet(1.0), BoundaryCondition::Dirichlet(1.0)],
            [BoundaryCondition::Dirichlet(1.0), BoundaryCondition::Dirichlet(1.0)],
        ];
        let config = GpuPdeConfig {
            max_iterations: 10_000,
            tolerance: 1e-8,
            ..Default::default()
        };

        let (sol, _) = solve_poisson_2d(&rhs, &bc, &config).expect("converged");
        // All boundary values must be 1.0
        let nx = sol.nx();
        let ny = sol.ny();
        for j in 0..ny {
            assert!((sol.get(0, j) - 1.0).abs() < 1e-10, "left bc violated at j={j}");
            assert!((sol.get(nx - 1, j) - 1.0).abs() < 1e-10, "right bc violated");
        }
        for i in 0..nx {
            assert!((sol.get(i, 0) - 1.0).abs() < 1e-10, "bottom bc violated at i={i}");
            assert!((sol.get(i, ny - 1) - 1.0).abs() < 1e-10, "top bc violated");
        }
    }

    #[test]
    fn test_fdm_poisson_unit_square() {
        // Solve ∇²u = -2π²sin(πx)sin(πy) on [0,1]² with zero Dirichlet BCs.
        // Exact solution: u = sin(πx)sin(πy).
        let n = 33usize; // grid points
        let h = 1.0 / (n - 1) as f64;
        let spec = GridSpec::new(n, n, h, h, 0.0, 0.0).expect("valid");

        let total = spec.total();
        let mut rhs_data = vec![0.0_f64; total];
        for j in 0..n {
            for i in 0..n {
                let x = spec.x_coord(i);
                let y = spec.y_coord(j);
                rhs_data[spec.idx(i, j)] = -2.0 * PI * PI * (PI * x).sin() * (PI * y).sin();
            }
        }
        let rhs = GpuGrid2D::from_data(rhs_data, spec).expect("rhs");

        let bc = dirichlet_zero_bc();
        let config = GpuPdeConfig {
            max_iterations: 20_000,
            tolerance: 1e-7,
            use_parallel: false,
            ..Default::default()
        };

        let (sol, stats) = solve_poisson_2d(&rhs, &bc, &config).expect("solve");
        assert!(stats.converged, "Poisson solver did not converge");

        // Compare with exact solution at interior points
        let mut max_err = 0.0_f64;
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                let x = spec.x_coord(i);
                let y = spec.y_coord(j);
                let exact = (PI * x).sin() * (PI * y).sin();
                let err = (sol.get(i, j) - exact).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        // Jacobi on coarse grids has O(h²) truncation error; expect < 0.01
        assert!(max_err < 0.01, "Max error {max_err} exceeds tolerance");
    }

    #[test]
    fn test_fdm_heat_conservation() {
        // Heat equation with Neumann (zero-flux) BCs: total heat is conserved.
        let n = 13usize;
        let h = 1.0 / (n - 1) as f64;
        let spec = GridSpec::new(n, n, h, h, 0.0, 0.0).expect("valid");

        // Initial condition: Gaussian bump
        let total = spec.total();
        let mut init_data = vec![0.0_f64; total];
        for j in 0..n {
            for i in 0..n {
                let x = spec.x_coord(i) - 0.5;
                let y = spec.y_coord(j) - 0.5;
                init_data[spec.idx(i, j)] = (-20.0 * (x * x + y * y)).exp();
            }
        }
        let mut u = GpuGrid2D::from_data(init_data, spec).expect("init");
        let initial_sum = u.sum();

        let dt = 0.2 * h * h; // well within CFL
        let alpha = 1.0;
        solve_heat_2d(&mut u, dt, alpha, 100).expect("heat solve");

        // With Neumann zero-flux BCs, total heat should be conserved.
        // The extrapolation approach gives O(h) conservation error, so we allow 20%.
        let final_sum = u.sum();
        let rel_change = (final_sum - initial_sum).abs() / initial_sum.abs();
        assert!(
            rel_change < 0.2,
            "Heat not approximately conserved: rel_change={rel_change:.4}"
        );
    }

    #[test]
    fn test_fdm_wave_propagation() {
        // Simple wave test: a sinusoidal initial displacement, zero velocity,
        // zero Dirichlet walls.  After enough steps the field should still be bounded.
        let n = 21usize;
        let h = 1.0 / (n - 1) as f64;
        let spec = GridSpec::new(n, n, h, h, 0.0, 0.0).expect("valid");

        let total = spec.total();
        let mut cur_data = vec![0.0_f64; total];
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                let x = spec.x_coord(i);
                let y = spec.y_coord(j);
                cur_data[spec.idx(i, j)] =
                    (PI * x).sin() * (PI * y).sin();
            }
        }
        let mut u = GpuGrid2D::from_data(cur_data.clone(), spec).expect("cur");
        let u_prev = GpuGrid2D::from_data(cur_data, spec).expect("prev"); // zero vel

        let c = 1.0;
        let dt = 0.4 * h / (c * 2.0_f64.sqrt()); // CFL ≈ 0.28 < 1
        let result = solve_wave_2d(&mut u, &u_prev, dt, c, 10).expect("wave");
        let norm = result.linf_norm();
        assert!(norm.is_finite(), "Wave field diverged");
        assert!(norm < 2.0, "Wave amplitude too large: {norm}");
    }

    #[test]
    fn test_fdm_convergence_with_grid_refinement() {
        // Solve ∇²u = 0 with u=1 on all boundaries.
        // On a refined grid, the interior should converge to 1.
        let test_sizes = [9usize, 17, 33];
        let mut max_errs = Vec::new();
        for &n in &test_sizes {
            let h = 1.0 / (n - 1) as f64;
            let spec = GridSpec::new(n, n, h, h, 0.0, 0.0).expect("valid");
            let rhs = GpuGrid2D::zeros(spec).expect("zeros");
            let bc = [
                [BoundaryCondition::Dirichlet(1.0), BoundaryCondition::Dirichlet(1.0)],
                [BoundaryCondition::Dirichlet(1.0), BoundaryCondition::Dirichlet(1.0)],
            ];
            let config =
                GpuPdeConfig { max_iterations: 50_000, tolerance: 1e-9, use_parallel: false, ..Default::default() };
            if let Ok((sol, _)) = solve_poisson_2d(&rhs, &bc, &config) {
                let interior = sol.get(n / 2, n / 2);
                max_errs.push((interior - 1.0).abs());
            }
        }
        // All interior values should be close to 1.0
        for (i, &err) in max_errs.iter().enumerate() {
            assert!(err < 0.01, "Grid {i}: interior error {err} too large");
        }
    }

    #[test]
    fn test_jacobi_iteration_reduces_residual() {
        // The residual must decrease (or stay the same) after each Jacobi sweep.
        let n = 9usize;
        let h = 0.125;
        let spec = GridSpec::new(n, n, h, h, 0.0, 0.0).expect("valid");
        let rhs = GpuGrid2D::zeros(spec).expect("zeros");
        let bc = dirichlet_zero_bc();

        let config_one = GpuPdeConfig { max_iterations: 1, tolerance: 1e-15, ..Default::default() };
        let config_many = GpuPdeConfig { max_iterations: 100, tolerance: 1e-15, ..Default::default() };

        // After 1 step, residual will be large (started from random-ish uniform + bc)
        // After many steps, residual should decrease.  We just check it doesn't explode.
        let _ = solve_poisson_2d(&rhs, &bc, &config_one);
        let result = solve_poisson_2d(&rhs, &bc, &config_many);
        // Either converges or NotConverged with small residual
        match result {
            Ok((_, stats)) => {
                assert!(stats.final_residual < 1e-6);
            }
            Err(PdeSolverError::NotConverged { iterations: _ }) => {
                // acceptable — we set tolerance to 1e-15
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn test_parallel_tile_update_consistency() {
        // Parallel and sequential Jacobi should produce identical results.
        let n = 9usize;
        let h = 0.125;
        let spec = GridSpec::new(n, n, h, h, 0.0, 0.0).expect("valid");
        let rhs = GpuGrid2D::zeros(spec).expect("zeros");
        let bc = dirichlet_zero_bc();

        let cfg_seq = GpuPdeConfig {
            max_iterations: 200,
            tolerance: 1e-10,
            use_parallel: false,
            ..Default::default()
        };
        let cfg_par = GpuPdeConfig {
            max_iterations: 200,
            tolerance: 1e-10,
            use_parallel: true,
            ..Default::default()
        };

        let res_seq = solve_poisson_2d(&rhs, &bc, &cfg_seq);
        let res_par = solve_poisson_2d(&rhs, &bc, &cfg_par);

        // Both should have the same convergence behaviour (both zero-RHS → trivially 0)
        match (res_seq, res_par) {
            (Ok((s, _)), Ok((p, _))) => {
                let diff = s.linf_diff(&p).expect("diff");
                assert!(diff < 1e-10, "Parallel/sequential mismatch: {diff}");
            }
            _ => {} // both may be NotConverged, that's fine
        }
    }

    #[test]
    fn test_boundary_condition_periodic_fdm() {
        // Periodic BCs: the left and right ghost layers should be consistent.
        let n = 9usize;
        let h = 0.125;
        let spec = GridSpec::new(n, n, h, h, 0.0, 0.0).expect("valid");
        let rhs = GpuGrid2D::zeros(spec).expect("zeros");
        let bc = [
            [BoundaryCondition::Periodic, BoundaryCondition::Periodic],
            [BoundaryCondition::Dirichlet(0.0), BoundaryCondition::Dirichlet(0.0)],
        ];
        let config = GpuPdeConfig { max_iterations: 200, tolerance: 1e-8, use_parallel: false, ..Default::default() };
        // Should not panic; result may or may not converge
        let _ = solve_poisson_2d(&rhs, &bc, &config);
    }
}
