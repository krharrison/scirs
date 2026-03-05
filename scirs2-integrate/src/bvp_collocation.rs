//! Boundary-value problem solver using Lobatto IIIA collocation
//!
//! This module implements a collocation method for two-point boundary-value
//! problems of the form:
//!
//! ```text
//!   y'(x) = f(x, y),   a <= x <= b
//!   bc(y(a), y(b)) = 0
//! ```
//!
//! The solver places collocation points at the Lobatto nodes (including the
//! mesh endpoints) on each sub-interval, constructs a global nonlinear system
//! from the collocation residuals plus boundary conditions, and solves it
//! with damped Newton iteration. Mesh refinement is driven by a defect-based
//! error estimate.
//!
//! ## Algorithm outline
//!
//! 1. Start with an initial mesh and guess.
//! 2. On each sub-interval `[x_i, x_{i+1}]` place the 3-point Lobatto IIIA
//!    nodes (endpoints + midpoint). The collocation polynomial is degree 3
//!    (4th-order accurate).
//! 3. Assemble the global Newton system and solve.
//! 4. Estimate the defect on each sub-interval; refine where needed.
//! 5. Repeat until the tolerance is met or the budget is exhausted.
//!
//! ## References
//!
//! - U. Ascher, R. Mattheij, R. Russell (1995), "Numerical Solution of
//!   Boundary Value Problems for Ordinary Differential Equations"
//! - J. Kierzenka, L. Shampine (2001), "A BVP Solver Based on Residual
//!   Control and the MATLAB PSE"

use crate::error::{IntegrateError, IntegrateResult};
use crate::IntegrateFloat;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};

/// Helper to convert f64 to generic float
#[inline(always)]
fn to_f<F: IntegrateFloat>(v: f64) -> F {
    F::from_f64(v).unwrap_or_else(|| F::zero())
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Options for the collocation BVP solver
#[derive(Debug, Clone)]
pub struct CollocationBVPOptions<F: IntegrateFloat> {
    /// Tolerance on the defect (default 1e-6)
    pub tol: F,
    /// Maximum Newton iterations per mesh (default 40)
    pub max_newton_iter: usize,
    /// Maximum number of mesh refinement cycles (default 10)
    pub max_mesh_refinements: usize,
    /// Maximum allowed mesh size (default 500 nodes)
    pub max_mesh_size: usize,
    /// Damping factor for Newton step (0 < factor <= 1, default 1.0)
    pub damping: F,
}

impl<F: IntegrateFloat> Default for CollocationBVPOptions<F> {
    fn default() -> Self {
        Self {
            tol: to_f(1e-6),
            max_newton_iter: 40,
            max_mesh_refinements: 10,
            max_mesh_size: 500,
            damping: F::one(),
        }
    }
}

/// Result from the collocation BVP solver
#[derive(Debug, Clone)]
pub struct CollocationBVPResult<F: IntegrateFloat> {
    /// Mesh points
    pub x: Vec<F>,
    /// Solution values at mesh points, each of length `n_dim`
    pub y: Vec<Array1<F>>,
    /// Number of Newton iterations (total across all refinements)
    pub n_newton_iter: usize,
    /// Number of mesh refinement cycles
    pub n_refinements: usize,
    /// Final maximum defect
    pub max_defect: F,
    /// Whether the solver converged
    pub converged: bool,
}

// ---------------------------------------------------------------------------
// Core solve function
// ---------------------------------------------------------------------------

/// Solve a two-point BVP using Lobatto IIIA collocation.
///
/// # Arguments
///
/// * `ode`      - Right-hand side `y'(x) = ode(x, y)`
/// * `bc`       - Boundary conditions: `bc(y(a), y(b))` returns residual vector
///   (length must equal the system dimension)
/// * `x_mesh`   - Initial mesh (strictly increasing, at least 2 points)
/// * `y_guess`  - Initial guess at each mesh point
/// * `options`  - Solver options (optional)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array1, ArrayView1};
/// use scirs2_integrate::bvp_collocation::{solve_bvp_collocation, CollocationBVPOptions};
///
/// // Solve y'' = -y on [0, pi], y(0)=0, y(pi)=0
/// // Rewrite as system: u1' = u2, u2' = -u1
/// let ode = |_x: f64, y: ArrayView1<f64>| array![y[1], -y[0]];
/// let bc = |ya: ArrayView1<f64>, yb: ArrayView1<f64>| array![ya[0], yb[0]];
///
/// let n = 11;
/// let pi = std::f64::consts::PI;
/// let x_mesh: Vec<f64> = (0..n).map(|i| i as f64 * pi / (n as f64 - 1.0)).collect();
/// let y_guess: Vec<Array1<f64>> = x_mesh.iter()
///     .map(|&x| array![x.sin(), x.cos()])
///     .collect();
///
/// let result = solve_bvp_collocation(ode, bc, &x_mesh, &y_guess, None)
///     .expect("collocation solve");
/// assert!(result.converged);
/// ```
pub fn solve_bvp_collocation<F, OdeFn, BcFn>(
    ode: OdeFn,
    bc: BcFn,
    x_mesh: &[F],
    y_guess: &[Array1<F>],
    options: Option<CollocationBVPOptions<F>>,
) -> IntegrateResult<CollocationBVPResult<F>>
where
    F: IntegrateFloat,
    OdeFn: Fn(F, ArrayView1<F>) -> Array1<F> + Copy,
    BcFn: Fn(ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
{
    let opts = options.unwrap_or_default();

    // Validate inputs
    if x_mesh.len() < 2 {
        return Err(IntegrateError::ValueError(
            "Mesh must have at least 2 points".into(),
        ));
    }
    if x_mesh.len() != y_guess.len() {
        return Err(IntegrateError::ValueError(
            "Mesh and guess must have the same length".into(),
        ));
    }

    let n_dim = y_guess[0].len();
    for (i, yg) in y_guess.iter().enumerate() {
        if yg.len() != n_dim {
            return Err(IntegrateError::ValueError(format!(
                "Guess at index {i} has wrong dimension: {} vs expected {n_dim}",
                yg.len()
            )));
        }
    }

    // Check mesh is strictly increasing
    for i in 1..x_mesh.len() {
        if x_mesh[i] <= x_mesh[i - 1] {
            return Err(IntegrateError::ValueError(
                "Mesh must be strictly increasing".into(),
            ));
        }
    }

    let mut mesh = x_mesh.to_vec();
    let mut y_sol: Vec<Array1<F>> = y_guess.to_vec();
    let mut total_newton = 0_usize;
    let mut n_refinements = 0_usize;

    loop {
        // Newton iteration on current mesh
        let (new_y, newton_iter, converged) =
            newton_collocation(&ode, &bc, &mesh, &y_sol, &opts, n_dim)?;
        total_newton += newton_iter;
        y_sol = new_y;

        if !converged {
            return Ok(CollocationBVPResult {
                x: mesh,
                y: y_sol,
                n_newton_iter: total_newton,
                n_refinements,
                max_defect: F::infinity(),
                converged: false,
            });
        }

        // Compute defect on each sub-interval
        let defects = compute_defects(&ode, &mesh, &y_sol, n_dim)?;
        let max_defect = defects
            .iter()
            .copied()
            .fold(F::zero(), |a, b| if b > a { b } else { a });

        if max_defect <= opts.tol {
            return Ok(CollocationBVPResult {
                x: mesh,
                y: y_sol,
                n_newton_iter: total_newton,
                n_refinements,
                max_defect,
                converged: true,
            });
        }

        n_refinements += 1;
        if n_refinements >= opts.max_mesh_refinements {
            return Ok(CollocationBVPResult {
                x: mesh,
                y: y_sol,
                n_newton_iter: total_newton,
                n_refinements,
                max_defect,
                converged: false,
            });
        }

        // Refine mesh: bisect intervals where defect exceeds tolerance
        let (new_mesh, new_y_sol) = refine_mesh(
            &ode,
            &mesh,
            &y_sol,
            &defects,
            opts.tol,
            opts.max_mesh_size,
            n_dim,
        )?;

        if new_mesh.len() >= opts.max_mesh_size {
            return Ok(CollocationBVPResult {
                x: new_mesh,
                y: new_y_sol,
                n_newton_iter: total_newton,
                n_refinements,
                max_defect,
                converged: false,
            });
        }

        mesh = new_mesh;
        y_sol = new_y_sol;
    }
}

// ---------------------------------------------------------------------------
// Newton collocation solve on a fixed mesh
// ---------------------------------------------------------------------------

/// Perform Newton iteration on the collocation system for the given mesh.
/// Returns `(solution, n_iterations, converged)`.
fn newton_collocation<F, OdeFn, BcFn>(
    ode: &OdeFn,
    bc: &BcFn,
    mesh: &[F],
    y_init: &[Array1<F>],
    opts: &CollocationBVPOptions<F>,
    n_dim: usize,
) -> IntegrateResult<(Vec<Array1<F>>, usize, bool)>
where
    F: IntegrateFloat,
    OdeFn: Fn(F, ArrayView1<F>) -> Array1<F>,
    BcFn: Fn(ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
{
    let n_pts = mesh.len();
    let n_intervals = n_pts - 1;
    // Total unknowns: n_pts * n_dim
    let n_vars = n_pts * n_dim;
    // Equations: n_dim boundary conditions + n_intervals * n_dim collocation equations
    let n_eqs = n_dim + n_intervals * n_dim;

    if n_eqs != n_vars {
        return Err(IntegrateError::DimensionMismatch(format!(
            "Collocation system: {n_eqs} equations vs {n_vars} unknowns"
        )));
    }

    let mut y_flat = flatten_solution(y_init, n_dim);
    let eps: F = to_f(1e-8);

    let mut converged = false;
    let mut iter = 0_usize;

    while iter < opts.max_newton_iter {
        iter += 1;

        // Evaluate residual
        let residual = assemble_residual(ode, bc, mesh, &y_flat, n_dim, n_pts)?;

        // Check convergence
        let res_norm = residual
            .iter()
            .fold(F::zero(), |acc, &r| acc + r * r)
            .sqrt()
            / to_f::<F>(n_eqs as f64).max(F::one());

        if res_norm < opts.tol {
            converged = true;
            break;
        }

        // Assemble Jacobian by finite differences
        let jac = assemble_jacobian(ode, bc, mesh, &y_flat, &residual, n_dim, n_pts, eps)?;

        // Solve J * delta = -residual
        let neg_res = residual.mapv(|r| -r);
        let delta = solve_dense_system(&jac, &neg_res)?;

        // Update with damping
        for i in 0..n_vars {
            y_flat[i] += opts.damping * delta[i];
        }
    }

    let y_sol = unflatten_solution(&y_flat, n_dim, n_pts);
    Ok((y_sol, iter, converged))
}

// ---------------------------------------------------------------------------
// Residual assembly
// ---------------------------------------------------------------------------

/// Assemble the global residual vector:
///   R = [ bc(y(a), y(b)); collocation residuals ]
fn assemble_residual<F, OdeFn, BcFn>(
    ode: &OdeFn,
    bc: &BcFn,
    mesh: &[F],
    y_flat: &Array1<F>,
    n_dim: usize,
    n_pts: usize,
) -> IntegrateResult<Array1<F>>
where
    F: IntegrateFloat,
    OdeFn: Fn(F, ArrayView1<F>) -> Array1<F>,
    BcFn: Fn(ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
{
    let n_intervals = n_pts - 1;
    let n_eqs = n_dim + n_intervals * n_dim;
    let mut res = Array1::zeros(n_eqs);

    // Extract y_a and y_b
    let y_a = y_flat.slice(s![0..n_dim]);
    let y_b = y_flat.slice(s![(n_pts - 1) * n_dim..n_pts * n_dim]);

    // Boundary conditions
    let bc_res = bc(y_a, y_b);
    for j in 0..n_dim {
        res[j] = bc_res[j];
    }

    // Collocation equations: on each interval [x_i, x_{i+1}],
    // we enforce the midpoint collocation condition:
    //   y_{i+1} - y_i - h * f(x_mid, y_mid) = 0
    // where y_mid = (y_i + y_{i+1}) / 2 + h/8 * (f_i - f_{i+1})
    // (Lobatto IIIA 3-point formula, cubic accurate)
    let half: F = to_f(0.5);
    let eighth: F = to_f(0.125);

    for i in 0..n_intervals {
        let x_i = mesh[i];
        let x_ip1 = mesh[i + 1];
        let h = x_ip1 - x_i;
        let x_mid = (x_i + x_ip1) * half;

        let y_i = y_flat.slice(s![i * n_dim..(i + 1) * n_dim]);
        let y_ip1 = y_flat.slice(s![(i + 1) * n_dim..(i + 2) * n_dim]);

        let f_i = ode(x_i, y_i);
        let f_ip1 = ode(x_ip1, y_ip1);

        // Lobatto IIIA midpoint predictor
        let mut y_mid = Array1::zeros(n_dim);
        for j in 0..n_dim {
            y_mid[j] = (y_i[j] + y_ip1[j]) * half + h * eighth * (f_i[j] - f_ip1[j]);
        }

        let f_mid = ode(x_mid, y_mid.view());

        // Lobatto IIIA collocation residual:
        //   y_{i+1} - y_i = h/6 * (f_i + 4*f_mid + f_ip1)  (Simpson-like)
        let sixth: F = to_f(1.0 / 6.0);
        let four: F = to_f(4.0);
        let eq_offset = n_dim + i * n_dim;
        for j in 0..n_dim {
            res[eq_offset + j] =
                y_ip1[j] - y_i[j] - h * sixth * (f_i[j] + four * f_mid[j] + f_ip1[j]);
        }
    }

    Ok(res)
}

// ---------------------------------------------------------------------------
// Jacobian assembly (finite differences)
// ---------------------------------------------------------------------------

fn assemble_jacobian<F, OdeFn, BcFn>(
    ode: &OdeFn,
    bc: &BcFn,
    mesh: &[F],
    y_flat: &Array1<F>,
    res0: &Array1<F>,
    n_dim: usize,
    n_pts: usize,
    eps: F,
) -> IntegrateResult<Array2<F>>
where
    F: IntegrateFloat,
    OdeFn: Fn(F, ArrayView1<F>) -> Array1<F>,
    BcFn: Fn(ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
{
    let n_vars = n_pts * n_dim;
    let n_eqs = res0.len();
    let mut jac = Array2::zeros((n_eqs, n_vars));

    for col in 0..n_vars {
        let mut y_pert = y_flat.clone();
        let delta = eps * (F::one() + y_pert[col].abs());
        y_pert[col] += delta;

        let res_pert = assemble_residual(ode, bc, mesh, &y_pert, n_dim, n_pts)?;

        for row in 0..n_eqs {
            jac[[row, col]] = (res_pert[row] - res0[row]) / delta;
        }
    }

    Ok(jac)
}

// ---------------------------------------------------------------------------
// Dense linear solver (LU with partial pivoting)
// ---------------------------------------------------------------------------

fn solve_dense_system<F: IntegrateFloat>(
    a: &Array2<F>,
    b: &Array1<F>,
) -> IntegrateResult<Array1<F>> {
    let n = a.nrows();
    if n != a.ncols() || n != b.len() {
        return Err(IntegrateError::DimensionMismatch(
            "solve_dense_system: dimension mismatch".into(),
        ));
    }

    let mut lu = a.clone();
    let mut piv: Vec<usize> = (0..n).collect();
    let tiny = F::from_f64(1e-30).unwrap_or_else(|| F::epsilon());

    for k in 0..n {
        let mut max_val = lu[[piv[k], k]].abs();
        let mut max_idx = k;
        for i in (k + 1)..n {
            let v = lu[[piv[i], k]].abs();
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        if max_val < tiny {
            return Err(IntegrateError::LinearSolveError(
                "Singular matrix in collocation solver".into(),
            ));
        }
        piv.swap(k, max_idx);

        for i in (k + 1)..n {
            let factor = lu[[piv[i], k]] / lu[[piv[k], k]];
            lu[[piv[i], k]] = factor;
            for j in (k + 1)..n {
                let val = lu[[piv[k], j]];
                lu[[piv[i], j]] -= factor * val;
            }
        }
    }

    let mut z = Array1::zeros(n);
    for i in 0..n {
        let mut s = b[piv[i]];
        for j in 0..i {
            s -= lu[[piv[i], j]] * z[j];
        }
        z[i] = s;
    }

    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut s = z[i];
        for j in (i + 1)..n {
            s -= lu[[piv[i], j]] * x[j];
        }
        if lu[[piv[i], i]].abs() < tiny {
            return Err(IntegrateError::LinearSolveError(
                "Zero diagonal in collocation LU".into(),
            ));
        }
        x[i] = s / lu[[piv[i], i]];
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Defect estimation
// ---------------------------------------------------------------------------

/// Compute the defect (residual of the continuous ODE) at the midpoint of
/// each sub-interval using the cubic Hermite interpolant.
fn compute_defects<F, OdeFn>(
    ode: &OdeFn,
    mesh: &[F],
    y_sol: &[Array1<F>],
    n_dim: usize,
) -> IntegrateResult<Vec<F>>
where
    F: IntegrateFloat,
    OdeFn: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    let n_intervals = mesh.len() - 1;
    let mut defects = Vec::with_capacity(n_intervals);
    let half: F = to_f(0.5);

    for i in 0..n_intervals {
        let h = mesh[i + 1] - mesh[i];
        let x_mid = (mesh[i] + mesh[i + 1]) * half;

        let f_i = ode(mesh[i], y_sol[i].view());
        let f_ip1 = ode(mesh[i + 1], y_sol[i + 1].view());

        // Cubic Hermite interpolation at midpoint
        let mut y_mid = Array1::zeros(n_dim);
        for j in 0..n_dim {
            y_mid[j] =
                (y_sol[i][j] + y_sol[i + 1][j]) * half + h * to_f::<F>(0.125) * (f_i[j] - f_ip1[j]);
        }

        // Derivative of Hermite interpolant at midpoint
        let mut yp_mid = Array1::zeros(n_dim);
        for j in 0..n_dim {
            yp_mid[j] = (y_sol[i + 1][j] - y_sol[i][j]) / h - to_f::<F>(0.25) * (f_i[j] + f_ip1[j])
                + half * to_f::<F>(1.0) * ((y_sol[i + 1][j] - y_sol[i][j]) / h);
            // Simplified: the cubic Hermite slope at midpoint
            yp_mid[j] = to_f::<F>(1.5) * (y_sol[i + 1][j] - y_sol[i][j]) / h
                - to_f::<F>(0.25) * (f_i[j] + f_ip1[j]);
        }

        let f_mid = ode(x_mid, y_mid.view());

        // Defect = ||yp_mid - f(x_mid, y_mid)||
        let mut defect_sq = F::zero();
        for j in 0..n_dim {
            let d = yp_mid[j] - f_mid[j];
            defect_sq += d * d;
        }
        defects.push(defect_sq.sqrt());
    }

    Ok(defects)
}

// ---------------------------------------------------------------------------
// Mesh refinement
// ---------------------------------------------------------------------------

/// Refine the mesh by bisecting intervals with large defects.
fn refine_mesh<F, OdeFn>(
    ode: &OdeFn,
    mesh: &[F],
    y_sol: &[Array1<F>],
    defects: &[F],
    tol: F,
    max_size: usize,
    n_dim: usize,
) -> IntegrateResult<(Vec<F>, Vec<Array1<F>>)>
where
    F: IntegrateFloat,
    OdeFn: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    let mut new_mesh = Vec::new();
    let mut new_y = Vec::new();

    new_mesh.push(mesh[0]);
    new_y.push(y_sol[0].clone());

    for i in 0..(mesh.len() - 1) {
        if defects[i] > tol && new_mesh.len() + 2 <= max_size {
            // Insert midpoint
            let half: F = to_f(0.5);
            let x_mid = (mesh[i] + mesh[i + 1]) * half;
            let h = mesh[i + 1] - mesh[i];

            let f_i = ode(mesh[i], y_sol[i].view());
            let f_ip1 = ode(mesh[i + 1], y_sol[i + 1].view());

            let mut y_mid = Array1::zeros(n_dim);
            for j in 0..n_dim {
                y_mid[j] = (y_sol[i][j] + y_sol[i + 1][j]) * half
                    + h * to_f::<F>(0.125) * (f_i[j] - f_ip1[j]);
            }

            new_mesh.push(x_mid);
            new_y.push(y_mid);
        }

        new_mesh.push(mesh[i + 1]);
        new_y.push(y_sol[i + 1].clone());
    }

    Ok((new_mesh, new_y))
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

fn flatten_solution<F: IntegrateFloat>(y: &[Array1<F>], n_dim: usize) -> Array1<F> {
    let n_pts = y.len();
    let mut flat = Array1::zeros(n_pts * n_dim);
    for (i, yi) in y.iter().enumerate() {
        for j in 0..n_dim {
            flat[i * n_dim + j] = yi[j];
        }
    }
    flat
}

fn unflatten_solution<F: IntegrateFloat>(
    flat: &Array1<F>,
    n_dim: usize,
    n_pts: usize,
) -> Vec<Array1<F>> {
    let mut y = Vec::with_capacity(n_pts);
    for i in 0..n_pts {
        let start = i * n_dim;
        let yi = Array1::from_vec(
            flat.slice(s![start..start + n_dim])
                .iter()
                .copied()
                .collect(),
        );
        y.push(yi);
    }
    y
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_linear_bvp() {
        // y'' = 0, y(0)=0, y(1)=1 => y(x) = x
        // System: u1' = u2, u2' = 0
        let ode = |_x: f64, y: ArrayView1<f64>| array![y[1], 0.0];
        let bc = |ya: ArrayView1<f64>, yb: ArrayView1<f64>| array![ya[0] - 0.0, yb[0] - 1.0];

        let n = 5;
        let mesh: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
        let guess: Vec<Array1<f64>> = mesh.iter().map(|&x| array![x, 1.0]).collect();

        let result = solve_bvp_collocation(ode, bc, &mesh, &guess, None).expect("linear BVP solve");

        assert!(result.converged, "linear BVP should converge");

        // Check that y(x) ≈ x at all mesh points
        for (i, xi) in result.x.iter().enumerate() {
            assert!(
                (result.y[i][0] - *xi).abs() < 1e-4,
                "y({xi}) = {}, expected {xi}",
                result.y[i][0]
            );
        }
    }

    #[test]
    fn test_exponential_bvp() {
        // y'' = y', y(0)=1, y(1)=e^(-1) => y(x) = e^(-x)
        // System: u1' = u2, u2' = -u2  (so that u1 = e^(-x))
        // Actually simpler: y' = -y, y(0)=1 is an IVP not BVP.
        // Let's use a proper 2-point BVP: y'' + y' = 0, y(0)=1, y(1)=e^{-1}
        // System: u1' = u2, u2' = -u2; BCs: u1(0)=1, u1(1)=e^{-1}
        let ode = |_x: f64, y: ArrayView1<f64>| array![y[1], -y[1]];
        let bc = |ya: ArrayView1<f64>, yb: ArrayView1<f64>| {
            let exact_end = (-1.0_f64).exp();
            array![ya[0] - 1.0, yb[0] - exact_end]
        };

        let n = 11;
        let mesh: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
        let guess: Vec<Array1<f64>> = mesh
            .iter()
            .map(|&x| array![(-x).exp(), -(-x).exp()])
            .collect();

        let result = solve_bvp_collocation(
            ode,
            bc,
            &mesh,
            &guess,
            Some(CollocationBVPOptions {
                max_newton_iter: 100,
                ..Default::default()
            }),
        )
        .expect("exp BVP solve");

        assert!(result.converged, "exp BVP should converge");

        let y_final = result.y.last().expect("has solution")[0];
        let exact = (-1.0_f64).exp();
        assert!(
            (y_final - exact).abs() < 1e-2,
            "y(1) = {y_final}, expected {exact}"
        );
    }

    #[test]
    fn test_nonlinear_bvp() {
        // Nonlinear BVP: y'' = -exp(y), y(0)=0, y(1)=0
        // (Bratu problem, has solution for small lambda)
        // System: u1' = u2, u2' = -exp(u1)
        let ode = |_x: f64, y: ArrayView1<f64>| array![y[1], -y[0].exp()];
        let bc = |ya: ArrayView1<f64>, yb: ArrayView1<f64>| array![ya[0], yb[0]];

        let n = 21;
        let mesh: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
        // Guess: parabola satisfying BCs, y ≈ 4*x*(1-x) * 0.1
        let guess: Vec<Array1<f64>> = mesh
            .iter()
            .map(|&x| {
                let y_val = 0.4 * x * (1.0 - x);
                let yp_val = 0.4 * (1.0 - 2.0 * x);
                array![y_val, yp_val]
            })
            .collect();

        let result = solve_bvp_collocation(
            ode,
            bc,
            &mesh,
            &guess,
            Some(CollocationBVPOptions {
                max_newton_iter: 100,
                ..Default::default()
            }),
        )
        .expect("Bratu BVP solve");

        assert!(result.converged, "Bratu BVP should converge");

        // Check BCs
        assert!(
            result.y[0][0].abs() < 1e-4,
            "y(0) should be 0, got {}",
            result.y[0][0]
        );
        let y_end = result.y.last().expect("has solution")[0];
        assert!(y_end.abs() < 1e-4, "y(1) should be 0, got {y_end}");

        // Solution should be positive in interior
        let mid_idx = n / 2;
        assert!(
            result.y[mid_idx][0] > 0.0,
            "Interior should be positive, got {}",
            result.y[mid_idx][0]
        );
    }

    #[test]
    fn test_stiff_bvp() {
        // Stiff BVP: epsilon*y'' - y = 0, y(0)=1, y(1)=0
        // For small epsilon, solution has boundary layer
        // With epsilon = 0.1, the exact solution involves exp(x/sqrt(eps))
        let epsilon = 0.1;
        let ode = move |_x: f64, y: ArrayView1<f64>| array![y[1], y[0] / epsilon];
        let bc = |ya: ArrayView1<f64>, yb: ArrayView1<f64>| array![ya[0] - 1.0, yb[0]];

        let n = 31;
        let mesh: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
        // Guess: linear decay from 1 to 0
        let guess: Vec<Array1<f64>> = mesh.iter().map(|&x| array![1.0 - x, -1.0]).collect();

        let result = solve_bvp_collocation(
            ode,
            bc,
            &mesh,
            &guess,
            Some(CollocationBVPOptions {
                tol: to_f(1e-4),
                max_newton_iter: 60,
                ..Default::default()
            }),
        )
        .expect("stiff BVP solve");

        assert!(result.converged, "stiff BVP should converge");
        // Check BC
        assert!(
            (result.y[0][0] - 1.0).abs() < 1e-3,
            "y(0) should be 1.0, got {}",
            result.y[0][0]
        );
        let y_end = result.y.last().expect("has solution")[0];
        assert!(y_end.abs() < 0.1, "y(1) should be ~0, got {y_end}");
    }

    #[test]
    fn test_invalid_mesh() {
        let ode = |_x: f64, _y: ArrayView1<f64>| array![0.0];
        let bc = |ya: ArrayView1<f64>, _yb: ArrayView1<f64>| array![ya[0]];

        // Mesh not increasing
        let res = solve_bvp_collocation(ode, bc, &[1.0, 0.0], &[array![0.0], array![0.0]], None);
        assert!(res.is_err());
    }

    #[test]
    fn test_mesh_guess_mismatch() {
        let ode = |_x: f64, _y: ArrayView1<f64>| array![0.0];
        let bc = |ya: ArrayView1<f64>, _yb: ArrayView1<f64>| array![ya[0]];

        let res = solve_bvp_collocation(
            ode,
            bc,
            &[0.0, 0.5, 1.0],
            &[array![0.0], array![0.0]], // wrong length
            None,
        );
        assert!(res.is_err());
    }
}
