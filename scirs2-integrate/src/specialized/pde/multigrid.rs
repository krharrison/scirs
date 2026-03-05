//! Geometric Multigrid Solver for Elliptic PDEs
//!
//! Implements geometric multigrid for solving elliptic PDEs such as the
//! Poisson equation `−∇²u = f` on a 2D domain [0,1]² with Dirichlet boundary
//! conditions.  Also supports the general variable-coefficient elliptic PDE:
//!
//! ```text
//!   −a(x,y)·∇²u + bx(x,y)·∂u/∂x + by(x,y)·∂u/∂y + c(x,y)·u = f
//! ```
//! where `a > 0` is the diffusion coefficient.
//!
//! # Cycle types
//! * **V-cycle** – standard single-sweep multigrid cycle
//! * **W-cycle** – two recursive calls per level
//! * **F-cycle** – FMG-style nested cycle between V and W
//!
//! # Smoothers
//! Red-black Gauss-Seidel relaxation on every grid level.
//!
//! # Restriction / Prolongation
//! * Restriction: full-weighting 9-point stencil
//! * Prolongation: bilinear interpolation

use scirs2_core::ndarray::Array2;

use crate::error::IntegrateError;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Which multigrid cycle to use
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CycleType {
    /// Standard V-cycle
    V,
    /// W-cycle (two recursive calls per level)
    W,
    /// F-cycle
    F,
}

/// Configuration for the multigrid solver
#[derive(Debug, Clone)]
pub struct MultigridConfig {
    /// Number of grid levels (at least 2)
    pub n_levels: usize,
    /// Pre-smoothing steps per level
    pub pre_smooth: usize,
    /// Post-smoothing steps per level
    pub post_smooth: usize,
    /// Grid size (total points per direction) at or below which to use direct
    /// solve.  Must be ≥ 3.
    pub coarse_size: usize,
    /// Cycle type
    pub cycle: CycleType,
    /// Maximum outer iterations
    pub max_iter: usize,
    /// Convergence tolerance (on residual ‖r‖₂)
    pub tol: f64,
}

impl Default for MultigridConfig {
    fn default() -> Self {
        Self {
            n_levels: 4,
            pre_smooth: 2,
            post_smooth: 2,
            coarse_size: 4,
            cycle: CycleType::V,
            max_iter: 50,
            tol: 1e-10,
        }
    }
}

/// Statistics returned by the multigrid solver
#[derive(Debug, Clone)]
pub struct MultigridStats {
    /// Total outer iterations performed
    pub iterations: usize,
    /// Final residual norm ‖r‖₂
    pub final_residual: f64,
    /// Residual norm at each outer iteration
    pub convergence_history: Vec<f64>,
}

/// Dirichlet boundary values for a 2-D domain [0,1]².
///
/// Each edge stores a function returning the prescribed value at `(x, y)`.
/// `None` means zero on that edge.
pub struct BoundaryCondition2D {
    /// u(0, y) – left edge
    pub left: Option<Box<dyn Fn(f64, f64) -> f64>>,
    /// u(1, y) – right edge
    pub right: Option<Box<dyn Fn(f64, f64) -> f64>>,
    /// u(x, 0) – bottom edge
    pub bottom: Option<Box<dyn Fn(f64, f64) -> f64>>,
    /// u(x, 1) – top edge
    pub top: Option<Box<dyn Fn(f64, f64) -> f64>>,
}

impl Default for BoundaryCondition2D {
    fn default() -> Self {
        Self {
            left: None,
            right: None,
            bottom: None,
            top: None,
        }
    }
}

impl std::fmt::Debug for BoundaryCondition2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundaryCondition2D")
            .field("left", &self.left.is_some())
            .field("right", &self.right.is_some())
            .field("bottom", &self.bottom.is_some())
            .field("top", &self.top.is_some())
            .finish()
    }
}

/// Variable-coefficient elliptic PDE coefficients for:
/// `−a(x,y)·∇²u + bx·∂u/∂x + by·∂u/∂y + c·u = f`
pub struct EllipticCoeffs2D {
    /// Diffusion coefficient a(x, y) > 0
    pub a: Box<dyn Fn(f64, f64) -> f64>,
    /// Advection in x-direction
    pub bx: Box<dyn Fn(f64, f64) -> f64>,
    /// Advection in y-direction
    pub by: Box<dyn Fn(f64, f64) -> f64>,
    /// Reaction coefficient
    pub c: Box<dyn Fn(f64, f64) -> f64>,
}

impl std::fmt::Debug for EllipticCoeffs2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("EllipticCoeffs2D { a, bx, by, c }")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Solve `−∇²u = f` on [0,1]² with Dirichlet boundary conditions.
///
/// `f` is an N×N array (including boundary rows/columns) where N = number of
/// total grid points per direction.  Interior values `f[1..N-1, 1..N-1]` drive
/// the solver; boundary rows/columns are not used.  Grid spacing `h = 1/(N-1)`.
pub fn solve_poisson_2d(
    f: &Array2<f64>,
    bc: &BoundaryCondition2D,
    config: &MultigridConfig,
) -> Result<(Array2<f64>, MultigridStats), IntegrateError> {
    let n = f.nrows();
    if n < 4 {
        return Err(IntegrateError::InvalidInput(
            "Grid must be at least 4×4 (total points)".to_string(),
        ));
    }
    if n != f.ncols() {
        return Err(IntegrateError::InvalidInput(
            "f must be square".to_string(),
        ));
    }
    validate_config(config)?;

    let h = 1.0 / (n as f64 - 1.0);
    let coeffs = EllipticCoeffs2D {
        a: Box::new(|_, _| 1.0),
        bx: Box::new(|_, _| 0.0),
        by: Box::new(|_, _| 0.0),
        c: Box::new(|_, _| 0.0),
    };

    let mut u = build_initial_guess(n, bc, h);
    apply_bc(&mut u, bc, h);

    let mut rhs = f.clone();
    zero_boundary(&mut rhs);

    multigrid_outer_loop(&mut u, &rhs, bc, &coeffs, config, h, n)
}

/// Solve `−a·∇²u + bx·∂u/∂x + by·∂u/∂y + c·u = f` on [0,1]² with Dirichlet BCs.
pub fn solve_elliptic_2d(
    coeffs: &EllipticCoeffs2D,
    f: &Array2<f64>,
    bc: &BoundaryCondition2D,
    config: &MultigridConfig,
) -> Result<(Array2<f64>, MultigridStats), IntegrateError> {
    let n = f.nrows();
    if n < 4 {
        return Err(IntegrateError::InvalidInput(
            "Grid must be at least 4×4".to_string(),
        ));
    }
    if n != f.ncols() {
        return Err(IntegrateError::InvalidInput(
            "f must be square".to_string(),
        ));
    }
    validate_config(config)?;

    let h = 1.0 / (n as f64 - 1.0);
    let mut u = build_initial_guess(n, bc, h);
    apply_bc(&mut u, bc, h);

    let mut rhs = f.clone();
    zero_boundary(&mut rhs);

    multigrid_outer_loop(&mut u, &rhs, bc, coeffs, config, h, n)
}

// ─────────────────────────────────────────────────────────────────────────────
// Outer iteration loop
// ─────────────────────────────────────────────────────────────────────────────

fn multigrid_outer_loop(
    u: &mut Array2<f64>,
    rhs: &Array2<f64>,
    bc: &BoundaryCondition2D,
    coeffs: &EllipticCoeffs2D,
    config: &MultigridConfig,
    h: f64,
    n: usize,
) -> Result<(Array2<f64>, MultigridStats), IntegrateError> {
    let rhs_norm = l2_norm_interior(rhs);
    let ref_norm = if rhs_norm > 1e-14 { rhs_norm } else { 1.0 };

    let mut history = Vec::with_capacity(config.max_iter);
    let mut iters = 0;
    let mut final_residual = f64::INFINITY;

    let levels = compute_level_sizes(n, config);

    for _iter in 0..config.max_iter {
        mg_cycle(u, rhs, bc, coeffs, h, config, &levels, 0, config.cycle);
        apply_bc(u, bc, h);

        let res = compute_residual(u, rhs, coeffs, h);
        let res_norm = l2_norm_interior(&res);
        history.push(res_norm);

        iters += 1;
        final_residual = res_norm;

        if res_norm / ref_norm < config.tol {
            break;
        }
    }

    Ok((
        u.clone(),
        MultigridStats {
            iterations: iters,
            final_residual,
            convergence_history: history,
        },
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// Level sizes
// ─────────────────────────────────────────────────────────────────────────────

fn compute_level_sizes(n_fine: usize, config: &MultigridConfig) -> Vec<usize> {
    let mut sizes = vec![n_fine];
    let mut cur = n_fine;
    for _ in 1..config.n_levels {
        // Standard coarsening: (N-1)/2 + 1 (requires N-1 even)
        if cur <= config.coarse_size + 2 {
            break;
        }
        let coarse = (cur - 1) / 2 + 1;
        sizes.push(coarse);
        cur = coarse;
        if cur <= config.coarse_size + 1 {
            break;
        }
    }
    sizes
}

// ─────────────────────────────────────────────────────────────────────────────
// Recursive multigrid cycle
// ─────────────────────────────────────────────────────────────────────────────

fn mg_cycle(
    u: &mut Array2<f64>,
    rhs: &Array2<f64>,
    bc: &BoundaryCondition2D,
    coeffs: &EllipticCoeffs2D,
    h: f64,
    config: &MultigridConfig,
    levels: &[usize],
    level_idx: usize,
    cycle: CycleType,
) {
    let n = levels[level_idx];

    // Base case: direct solve on coarsest grid
    if level_idx + 1 >= levels.len() || n <= config.coarse_size + 1 {
        gauss_seidel_redblack(u, rhs, coeffs, h, 20);
        return;
    }

    // Pre-smoothing
    gauss_seidel_redblack(u, rhs, coeffs, h, config.pre_smooth);

    // Compute fine-grid residual r = f - A*u
    let res_fine = compute_residual(u, rhs, coeffs, h);

    // Restrict residual to coarse grid
    let n_coarse = levels[level_idx + 1];
    let h_coarse = h * ((n as f64 - 1.0) / (n_coarse as f64 - 1.0));
    let rhs_coarse = restrict_full_weighting(&res_fine, n_coarse);

    // Error correction on coarse grid (zero initial error)
    let mut e_coarse = Array2::<f64>::zeros((n_coarse, n_coarse));

    let num_calls = match cycle {
        CycleType::V => 1,
        CycleType::W | CycleType::F => 2,
    };

    for call_idx in 0..num_calls {
        let this_cycle = match cycle {
            CycleType::V => CycleType::V,
            CycleType::W => CycleType::W,
            CycleType::F => {
                if call_idx == 0 {
                    CycleType::V
                } else {
                    CycleType::W
                }
            }
        };
        // Zero BC for error equation (error is zero on boundary)
        let zero_bc = BoundaryCondition2D::default();
        mg_cycle(
            &mut e_coarse,
            &rhs_coarse,
            &zero_bc,
            coeffs,
            h_coarse,
            config,
            levels,
            level_idx + 1,
            this_cycle,
        );
    }

    // Prolongate correction to fine grid and add
    let e_fine = prolongate_bilinear(&e_coarse, n);
    let (rows, cols) = u.dim();
    for i in 0..rows {
        for j in 0..cols {
            u[[i, j]] += e_fine[[i, j]];
        }
    }

    // Re-enforce boundary conditions
    apply_bc(u, bc, h);

    // Post-smoothing
    gauss_seidel_redblack(u, rhs, coeffs, h, config.post_smooth);
}

// ─────────────────────────────────────────────────────────────────────────────
// Red-black Gauss-Seidel smoother
//
// Solves: -a*(u_E+u_W+u_N+u_S-4*u)/h² + bx*(u_E-u_W)/(2h) + by*(u_N-u_S)/(2h) + c*u = rhs
// for u at each interior node.
// ─────────────────────────────────────────────────────────────────────────────

fn gauss_seidel_redblack(
    u: &mut Array2<f64>,
    rhs: &Array2<f64>,
    coeffs: &EllipticCoeffs2D,
    h: f64,
    steps: usize,
) {
    let n = u.nrows();
    if n < 3 {
        return;
    }
    let h2 = h * h;
    let h2_inv = 1.0 / h2;

    for _ in 0..steps {
        for color in 0..2usize {
            for i in 1..n - 1 {
                for j in 1..n - 1 {
                    if (i + j) % 2 != color {
                        continue;
                    }
                    let x = j as f64 * h;
                    let y = i as f64 * h;
                    let a = (coeffs.a)(x, y);
                    let bx = (coeffs.bx)(x, y);
                    let by = (coeffs.by)(x, y);
                    let c_coeff = (coeffs.c)(x, y);

                    let u_e = u[[i, j + 1]];
                    let u_w = u[[i, j - 1]];
                    // Note: row index i corresponds to y coordinate; i=0 is top (y=1), i=n-1 is bottom (y=0)
                    // We use: u_n = u[[i-1,j]] (one row up = higher y)
                    //          u_s = u[[i+1,j]] (one row down = lower y)
                    let u_n = u[[i - 1, j]];
                    let u_s = u[[i + 1, j]];

                    // Operator: -a*(u_E+u_W+u_N+u_S-4*u)/h² + bx*(u_E-u_W)/(2h) + by*(u_N-u_S)/(2h) + c*u = rhs
                    // Solve for u:
                    // Diagonal coeff of u: 4*a/h² + c
                    // Off-diagonal terms contribution:
                    //   from Laplacian: -a*(u_E+u_W+u_N+u_S)/h²
                    //   from advection: bx*(u_E-u_W)/(2h) + by*(u_N-u_S)/(2h)
                    let diag = 4.0 * a * h2_inv + c_coeff;
                    if diag.abs() < 1e-14 {
                        continue;
                    }
                    let rhs_u = a * h2_inv * (u_e + u_w + u_n + u_s)
                        - bx * (u_e - u_w) / (2.0 * h)
                        - by * (u_n - u_s) / (2.0 * h);
                    u[[i, j]] = (rhs[[i, j]] + rhs_u) / diag;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Residual: r = f - A*u
// A*u = -a*(∇²u) + bx*∂u/∂x + by*∂u/∂y + c*u  (discretized)
// ─────────────────────────────────────────────────────────────────────────────

fn compute_residual(
    u: &Array2<f64>,
    rhs: &Array2<f64>,
    coeffs: &EllipticCoeffs2D,
    h: f64,
) -> Array2<f64> {
    let n = u.nrows();
    let mut res = Array2::<f64>::zeros((n, n));
    let h2_inv = 1.0 / (h * h);

    for i in 1..n - 1 {
        for j in 1..n - 1 {
            let x = j as f64 * h;
            let y = i as f64 * h;
            let a = (coeffs.a)(x, y);
            let bx = (coeffs.bx)(x, y);
            let by = (coeffs.by)(x, y);
            let c_coeff = (coeffs.c)(x, y);

            let u_c = u[[i, j]];
            let u_e = u[[i, j + 1]];
            let u_w = u[[i, j - 1]];
            let u_n = u[[i - 1, j]];
            let u_s = u[[i + 1, j]];

            // Discrete operator A*u:
            let neg_lap = -a * h2_inv * (u_e + u_w + u_n + u_s - 4.0 * u_c);
            let adv = bx * (u_e - u_w) / (2.0 * h) + by * (u_n - u_s) / (2.0 * h);
            let react = c_coeff * u_c;

            let au = neg_lap + adv + react;
            res[[i, j]] = rhs[[i, j]] - au;
        }
    }
    res
}

// ─────────────────────────────────────────────────────────────────────────────
// Restriction: full-weighting 9-point stencil
// ─────────────────────────────────────────────────────────────────────────────

fn restrict_full_weighting(fine: &Array2<f64>, n_coarse: usize) -> Array2<f64> {
    let n_fine = fine.nrows();
    let mut coarse = Array2::<f64>::zeros((n_coarse, n_coarse));

    if n_fine <= 1 || n_coarse <= 1 {
        return coarse;
    }

    // Ratio must be an integer (standard coarsening requires (n_fine-1) = 2*(n_coarse-1))
    let ratio = if n_coarse > 1 {
        (n_fine - 1) / (n_coarse - 1)
    } else {
        return coarse;
    };

    for i in 1..n_coarse - 1 {
        for j in 1..n_coarse - 1 {
            let fi = i * ratio;
            let fj = j * ratio;

            // Clamp to valid fine-grid indices
            let fi_p1 = (fi + 1).min(n_fine - 1);
            let fi_m1 = fi.saturating_sub(1);
            let fj_p1 = (fj + 1).min(n_fine - 1);
            let fj_m1 = fj.saturating_sub(1);

            let c = fine[[fi, fj]];
            let e = fine[[fi, fj_p1]];
            let w = fine[[fi, fj_m1]];
            let n_ = fine[[fi_m1, fj]];
            let s = fine[[fi_p1, fj]];
            let ne = fine[[fi_m1, fj_p1]];
            let nw = fine[[fi_m1, fj_m1]];
            let se = fine[[fi_p1, fj_p1]];
            let sw = fine[[fi_p1, fj_m1]];
            coarse[[i, j]] = (4.0 * c + 2.0 * (e + w + n_ + s) + (ne + nw + se + sw)) / 16.0;
        }
    }
    coarse
}

// ─────────────────────────────────────────────────────────────────────────────
// Prolongation: bilinear interpolation
// ─────────────────────────────────────────────────────────────────────────────

fn prolongate_bilinear(coarse: &Array2<f64>, n_fine: usize) -> Array2<f64> {
    let n_coarse = coarse.nrows();
    let mut fine = Array2::<f64>::zeros((n_fine, n_fine));

    if n_coarse <= 1 {
        return fine;
    }

    let ratio = (n_fine - 1) as f64 / (n_coarse - 1) as f64;

    for i in 0..n_fine {
        for j in 0..n_fine {
            let ci = i as f64 / ratio;
            let cj = j as f64 / ratio;

            let ci0 = ci.floor() as usize;
            let cj0 = cj.floor() as usize;
            let ci1 = (ci0 + 1).min(n_coarse - 1);
            let cj1 = (cj0 + 1).min(n_coarse - 1);

            let ti = ci - ci0 as f64;
            let tj = cj - cj0 as f64;

            fine[[i, j]] = (1.0 - ti) * (1.0 - tj) * coarse[[ci0, cj0]]
                + ti * (1.0 - tj) * coarse[[ci1, cj0]]
                + (1.0 - ti) * tj * coarse[[ci0, cj1]]
                + ti * tj * coarse[[ci1, cj1]];
        }
    }
    fine
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn validate_config(config: &MultigridConfig) -> Result<(), IntegrateError> {
    if config.n_levels < 1 {
        return Err(IntegrateError::InvalidInput(
            "n_levels must be ≥ 1".to_string(),
        ));
    }
    if config.coarse_size < 2 {
        return Err(IntegrateError::InvalidInput(
            "coarse_size must be ≥ 2".to_string(),
        ));
    }
    if config.tol <= 0.0 {
        return Err(IntegrateError::InvalidInput(
            "tol must be positive".to_string(),
        ));
    }
    Ok(())
}

fn build_initial_guess(n: usize, bc: &BoundaryCondition2D, h: f64) -> Array2<f64> {
    let mut u = Array2::<f64>::zeros((n, n));
    apply_bc(&mut u, bc, h);
    u
}

fn apply_bc(u: &mut Array2<f64>, bc: &BoundaryCondition2D, h: f64) {
    let n = u.nrows();
    for k in 0..n {
        let x_k = k as f64 * h;
        // Left edge: j=0, y varies with row index i
        for i in 0..n {
            let y_i = (n as f64 - 1.0 - i as f64) * h; // y=0 at bottom row, y=1 at top
            let left_val = bc.left.as_ref().map(|f| f(0.0, y_i)).unwrap_or(0.0);
            u[[i, 0]] = left_val;
            let right_val = bc.right.as_ref().map(|f| f(1.0, y_i)).unwrap_or(0.0);
            u[[i, n - 1]] = right_val;
        }
        // Bottom edge: i=n-1 (y=0), x varies with column index j
        let bottom_val = bc.bottom.as_ref().map(|f| f(x_k, 0.0)).unwrap_or(0.0);
        u[[n - 1, k]] = bottom_val;
        // Top edge: i=0 (y=1)
        let top_val = bc.top.as_ref().map(|f| f(x_k, 1.0)).unwrap_or(0.0);
        u[[0, k]] = top_val;
    }
}

fn zero_boundary(a: &mut Array2<f64>) {
    let n = a.nrows();
    let m = a.ncols();
    for j in 0..m {
        a[[0, j]] = 0.0;
        a[[n - 1, j]] = 0.0;
    }
    for i in 0..n {
        a[[i, 0]] = 0.0;
        a[[i, m - 1]] = 0.0;
    }
}

fn l2_norm_interior(a: &Array2<f64>) -> f64 {
    let n = a.nrows();
    let m = a.ncols();
    let mut sum = 0.0;
    for i in 1..n - 1 {
        for j in 1..m - 1 {
            sum += a[[i, j]] * a[[i, j]];
        }
    }
    sum.sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;
    use std::f64::consts::PI;

    /// Known solution: u(x,y) = sin(πx)sin(πy)
    /// -∇²u = 2π²·sin(πx)·sin(πy) = 2π²·u
    /// So we solve -∇²u = f with f = 2π²·sin(πx)sin(πy).
    #[test]
    fn test_poisson_known_solution() {
        let n = 33; // 2^5 + 1 for clean coarsening
        let h = 1.0 / (n as f64 - 1.0);

        let mut f = Array2::<f64>::zeros((n, n));
        for i in 1..n - 1 {
            for j in 1..n - 1 {
                let x = j as f64 * h;
                let y = (n as f64 - 1.0 - i as f64) * h; // y=0 at bottom row i=n-1
                f[[i, j]] = 2.0 * PI * PI * (PI * x).sin() * (PI * y).sin();
            }
        }

        let bc = BoundaryCondition2D::default(); // zero BCs
        let config = MultigridConfig {
            n_levels: 5,
            pre_smooth: 2,
            post_smooth: 2,
            coarse_size: 4,
            cycle: CycleType::V,
            max_iter: 100,
            tol: 1e-8,
        };

        let (u, stats) = solve_poisson_2d(&f, &bc, &config).expect("multigrid should converge");

        let mut max_err = 0.0f64;
        for i in 1..n - 1 {
            for j in 1..n - 1 {
                let x = j as f64 * h;
                let y = (n as f64 - 1.0 - i as f64) * h;
                let exact = (PI * x).sin() * (PI * y).sin();
                let err = (u[[i, j]] - exact).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }

        assert!(
            max_err < 5e-3,
            "Max error too large: {max_err:.2e}, iters={}",
            stats.iterations
        );
        assert!(
            stats.final_residual < 1e-3,
            "Residual too large: {:.2e}",
            stats.final_residual
        );
    }

    #[test]
    fn test_multigrid_v_w_f_cycles() {
        let n = 17;
        let h = 1.0 / (n as f64 - 1.0);
        let mut f = Array2::<f64>::zeros((n, n));
        for i in 1..n - 1 {
            for j in 1..n - 1 {
                let x = j as f64 * h;
                let y = (n as f64 - 1.0 - i as f64) * h;
                f[[i, j]] = 2.0 * PI * PI * (PI * x).sin() * (PI * y).sin();
            }
        }
        let bc = BoundaryCondition2D::default();
        for cycle in [CycleType::V, CycleType::W, CycleType::F] {
            let config = MultigridConfig {
                n_levels: 4,
                cycle,
                max_iter: 50,
                tol: 1e-6,
                ..Default::default()
            };
            let result = solve_poisson_2d(&f, &bc, &config);
            assert!(result.is_ok(), "Cycle {:?} failed", cycle);
            let (_, stats) = result.expect("solve_poisson_2d should succeed");
            assert!(
                stats.final_residual < 1e-3,
                "Cycle {:?} residual too large: {:.2e}",
                cycle,
                stats.final_residual
            );
        }
    }

    #[test]
    fn test_elliptic_with_coefficients() {
        // -2*∇²u = 4π²sin(πx)sin(πy), u=0 on boundary
        // Exact: u = sin(πx)sin(πy)
        let n = 17;
        let h = 1.0 / (n as f64 - 1.0);
        let mut rhs = Array2::<f64>::zeros((n, n));
        for i in 1..n - 1 {
            for j in 1..n - 1 {
                let x = j as f64 * h;
                let y = (n as f64 - 1.0 - i as f64) * h;
                rhs[[i, j]] = 4.0 * PI * PI * (PI * x).sin() * (PI * y).sin();
            }
        }
        let coeffs = EllipticCoeffs2D {
            a: Box::new(|_, _| 2.0),
            bx: Box::new(|_, _| 0.0),
            by: Box::new(|_, _| 0.0),
            c: Box::new(|_, _| 0.0),
        };
        let bc = BoundaryCondition2D::default();
        let config = MultigridConfig {
            n_levels: 4,
            max_iter: 100,
            tol: 1e-6,
            ..Default::default()
        };
        let (u, stats) = solve_elliptic_2d(&coeffs, &rhs, &bc, &config)
            .expect("elliptic solve failed");
        let mut max_err = 0.0f64;
        for i in 1..n - 1 {
            for j in 1..n - 1 {
                let x = j as f64 * h;
                let y = (n as f64 - 1.0 - i as f64) * h;
                let exact = (PI * x).sin() * (PI * y).sin();
                let err = (u[[i, j]] - exact).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        assert!(
            max_err < 0.1,
            "Max error too large: {max_err:.2e}, iters={}",
            stats.iterations
        );
    }

    #[test]
    fn test_invalid_inputs() {
        let f = Array2::<f64>::zeros((3, 3));
        let bc = BoundaryCondition2D::default();
        let config = MultigridConfig::default();
        assert!(solve_poisson_2d(&f, &bc, &config).is_err());
    }

    /// Convergence rate test: multigrid should converge geometrically
    #[test]
    fn test_convergence_history_decreasing() {
        let n = 17;
        let h = 1.0 / (n as f64 - 1.0);
        let mut f = Array2::<f64>::zeros((n, n));
        for i in 1..n - 1 {
            for j in 1..n - 1 {
                let x = j as f64 * h;
                let y = (n as f64 - 1.0 - i as f64) * h;
                f[[i, j]] = 2.0 * PI * PI * (PI * x).sin() * (PI * y).sin();
            }
        }
        let bc = BoundaryCondition2D::default();
        let config = MultigridConfig {
            n_levels: 4,
            max_iter: 20,
            tol: 1e-12, // Don't converge early
            ..Default::default()
        };
        let (_, stats) = solve_poisson_2d(&f, &bc, &config).expect("should not error");

        // Check that residuals are generally decreasing
        assert!(
            stats.convergence_history.len() >= 5,
            "Too few iterations: {}",
            stats.convergence_history.len()
        );
        let first = stats.convergence_history[0];
        let last = *stats.convergence_history.last().unwrap_or(&first);
        assert!(
            last < first,
            "Residual did not decrease: {} → {}",
            first,
            last
        );
    }
}
