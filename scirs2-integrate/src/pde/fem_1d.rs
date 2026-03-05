//! 1D Finite Element Method (FEM) Solver
//!
//! Provides Galerkin FEM for 1D elliptic and parabolic problems using
//! linear and quadratic Lagrange basis functions.
//!
//! ## Features
//! - Linear (P1) and quadratic (P2) Lagrange elements
//! - Assembly of stiffness and mass matrices
//! - Dirichlet and Neumann boundary condition application
//! - Steady-state solver: -d/dx(a(x) du/dx) + c(x) u = f(x)
//! - Transient solver: M du/dt + K u = F (implicit theta-method)
//! - A-posteriori error estimation via residual-based indicators
//! - Mesh refinement indicators (mark elements for refinement)

use scirs2_core::ndarray::{Array1, Array2};
use std::f64;

use crate::pde::{PDEError, PDEResult};

// ---------------------------------------------------------------------------
// Basis function type
// ---------------------------------------------------------------------------

/// Element type for 1D FEM
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FEM1DElementType {
    /// Linear (P1) Lagrange element with 2 nodes per element
    Linear,
    /// Quadratic (P2) Lagrange element with 3 nodes per element
    Quadratic,
}

/// Boundary condition for 1D FEM
#[derive(Debug, Clone)]
pub enum FEM1DBoundaryCondition {
    /// Dirichlet: u(boundary) = value
    Dirichlet(f64),
    /// Neumann: a(x) du/dx(boundary) = value (flux)
    Neumann(f64),
}

// ---------------------------------------------------------------------------
// FEM1D mesh
// ---------------------------------------------------------------------------

/// 1D finite element mesh
#[derive(Debug, Clone)]
pub struct Mesh1D {
    /// Node coordinates (sorted ascending)
    pub nodes: Array1<f64>,
    /// Element connectivity: `elements[e]` = `[start_node_global, ..., end_node_global]`
    pub elements: Vec<Vec<usize>>,
    /// Element type
    pub element_type: FEM1DElementType,
}

impl Mesh1D {
    /// Create a uniform mesh on [a, b] with `num_elements` elements.
    pub fn uniform(
        a: f64,
        b: f64,
        num_elements: usize,
        element_type: FEM1DElementType,
    ) -> PDEResult<Self> {
        if num_elements < 1 {
            return Err(PDEError::InvalidParameter(
                "Need at least 1 element".to_string(),
            ));
        }
        if a >= b {
            return Err(PDEError::DomainError("a must be < b".to_string()));
        }

        match element_type {
            FEM1DElementType::Linear => {
                let n_nodes = num_elements + 1;
                let h = (b - a) / num_elements as f64;
                let nodes = Array1::from_shape_fn(n_nodes, |i| a + i as f64 * h);
                let elements: Vec<Vec<usize>> = (0..num_elements).map(|e| vec![e, e + 1]).collect();
                Ok(Mesh1D {
                    nodes,
                    elements,
                    element_type,
                })
            }
            FEM1DElementType::Quadratic => {
                // Each quadratic element has 3 nodes: left, mid, right
                let n_nodes = 2 * num_elements + 1;
                let h = (b - a) / num_elements as f64;
                let nodes = Array1::from_shape_fn(n_nodes, |i| a + (i as f64) * h / 2.0);
                let elements: Vec<Vec<usize>> = (0..num_elements)
                    .map(|e| vec![2 * e, 2 * e + 1, 2 * e + 2])
                    .collect();
                Ok(Mesh1D {
                    nodes,
                    elements,
                    element_type,
                })
            }
        }
    }

    /// Number of nodes
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of elements
    pub fn num_elements(&self) -> usize {
        self.elements.len()
    }

    /// Element size for element `e`
    pub fn element_size(&self, e: usize) -> f64 {
        let el = &self.elements[e];
        let first = *el.first().unwrap_or(&0);
        let last = *el.last().unwrap_or(&0);
        (self.nodes[last] - self.nodes[first]).abs()
    }
}

// ---------------------------------------------------------------------------
// Steady-state solver
// ---------------------------------------------------------------------------

/// Options for steady-state FEM solver
#[derive(Debug, Clone)]
pub struct FEM1DSteadyOptions {
    /// Element type
    pub element_type: FEM1DElementType,
    /// Number of Gauss quadrature points per element
    pub num_quad_points: usize,
}

impl Default for FEM1DSteadyOptions {
    fn default() -> Self {
        FEM1DSteadyOptions {
            element_type: FEM1DElementType::Linear,
            num_quad_points: 3,
        }
    }
}

/// Result from steady-state FEM solve
#[derive(Debug, Clone)]
pub struct FEM1DSteadyResult {
    /// Node coordinates
    pub x: Array1<f64>,
    /// Solution at nodes
    pub u: Array1<f64>,
    /// Stiffness matrix
    pub stiffness: Array2<f64>,
    /// Load vector
    pub load: Array1<f64>,
    /// Element-wise error indicators
    pub error_indicators: Array1<f64>,
}

/// Solve steady-state 1D BVP:
///   -d/dx[ a(x) du/dx ] + c(x) u = f(x),  x in [x_left, x_right]
///
/// with boundary conditions at left and right endpoints.
///
/// # Arguments
/// * `a_coeff` - diffusion coefficient a(x) > 0
/// * `c_coeff` - reaction coefficient c(x) >= 0
/// * `f_source` - source/forcing function f(x)
/// * `x_left`, `x_right` - domain endpoints
/// * `num_elements` - number of finite elements
/// * `left_bc` - boundary condition at x_left
/// * `right_bc` - boundary condition at x_right
/// * `options` - solver options
pub fn solve_steady_1d(
    a_coeff: &dyn Fn(f64) -> f64,
    c_coeff: &dyn Fn(f64) -> f64,
    f_source: &dyn Fn(f64) -> f64,
    x_left: f64,
    x_right: f64,
    num_elements: usize,
    left_bc: &FEM1DBoundaryCondition,
    right_bc: &FEM1DBoundaryCondition,
    options: &FEM1DSteadyOptions,
) -> PDEResult<FEM1DSteadyResult> {
    let mesh = Mesh1D::uniform(x_left, x_right, num_elements, options.element_type)?;
    let n = mesh.num_nodes();

    // Assemble global stiffness matrix K and load vector F
    let (mut k_global, mut f_global) =
        assemble_system(&mesh, a_coeff, c_coeff, f_source, options.num_quad_points)?;

    // Apply Neumann BCs (natural BCs add flux to load vector)
    if let FEM1DBoundaryCondition::Neumann(flux) = right_bc {
        f_global[n - 1] += *flux;
    }
    if let FEM1DBoundaryCondition::Neumann(flux) = left_bc {
        f_global[0] -= *flux; // negative because outward normal is -x
    }

    // Store pre-BC matrices for error estimation
    let k_orig = k_global.clone();
    let f_orig = f_global.clone();

    // Apply Dirichlet BCs by penalty method
    apply_dirichlet_bc(&mut k_global, &mut f_global, left_bc, right_bc, n);

    // Solve K u = F
    let u = solve_linear_system_1d(&k_global, &f_global)?;

    // Compute error indicators
    let error_indicators = compute_error_indicators(&mesh, &u, &k_orig, &f_orig)?;

    Ok(FEM1DSteadyResult {
        x: mesh.nodes,
        u,
        stiffness: k_global,
        load: f_global,
        error_indicators,
    })
}

// ---------------------------------------------------------------------------
// Transient solver
// ---------------------------------------------------------------------------

/// Options for transient FEM solver
#[derive(Debug, Clone)]
pub struct FEM1DTransientOptions {
    /// Element type
    pub element_type: FEM1DElementType,
    /// Number of Gauss quadrature points
    pub num_quad_points: usize,
    /// Theta parameter: 0=explicit, 0.5=Crank-Nicolson, 1=implicit Euler
    pub theta: f64,
    /// Time step
    pub dt: f64,
    /// Number of time steps
    pub num_steps: usize,
    /// Save solution every N steps (0 or 1 = save all)
    pub save_every: usize,
}

impl Default for FEM1DTransientOptions {
    fn default() -> Self {
        FEM1DTransientOptions {
            element_type: FEM1DElementType::Linear,
            num_quad_points: 3,
            theta: 0.5, // Crank-Nicolson
            dt: 0.01,
            num_steps: 100,
            save_every: 1,
        }
    }
}

/// Result from transient FEM solve
#[derive(Debug, Clone)]
pub struct FEM1DTransientResult {
    /// Node coordinates
    pub x: Array1<f64>,
    /// Time values at saved steps
    pub t: Array1<f64>,
    /// Solution snapshots: `u[step][node]`
    pub u: Vec<Array1<f64>>,
    /// Error indicators at final time
    pub error_indicators: Array1<f64>,
}

/// Solve transient 1D problem:
///   du/dt = d/dx[ a(x) du/dx ] - c(x) u + f(x, t)
///
/// using theta-method time integration (theta=0.5 gives Crank-Nicolson).
///
/// The semidiscrete form is: M du/dt + K u = F
/// Theta-method: (M + theta*dt*K) u^{n+1} = (M - (1-theta)*dt*K) u^n + dt*F
pub fn solve_transient_1d(
    a_coeff: &dyn Fn(f64) -> f64,
    c_coeff: &dyn Fn(f64) -> f64,
    f_source: &dyn Fn(f64, f64) -> f64,
    x_left: f64,
    x_right: f64,
    num_elements: usize,
    initial_condition: &dyn Fn(f64) -> f64,
    left_bc: &FEM1DBoundaryCondition,
    right_bc: &FEM1DBoundaryCondition,
    options: &FEM1DTransientOptions,
) -> PDEResult<FEM1DTransientResult> {
    if options.theta < 0.0 || options.theta > 1.0 {
        return Err(PDEError::InvalidParameter(
            "Theta must be in [0, 1]".to_string(),
        ));
    }

    let mesh = Mesh1D::uniform(x_left, x_right, num_elements, options.element_type)?;
    let n = mesh.num_nodes();
    let dt = options.dt;
    let theta = options.theta;

    // Assemble stiffness and mass matrices
    // For transient: we assemble K (stiffness) and M (mass) separately
    let (k_global, _f_placeholder) =
        assemble_system(&mesh, a_coeff, c_coeff, &|_| 0.0, options.num_quad_points)?;
    let m_global = assemble_mass_matrix(&mesh, options.num_quad_points)?;

    // LHS matrix: A = M + theta*dt*K
    let mut a_mat = m_global.clone();
    for i in 0..n {
        for j in 0..n {
            a_mat[[i, j]] += theta * dt * k_global[[i, j]];
        }
    }

    // RHS matrix factor: B = M - (1-theta)*dt*K
    let mut b_mat = m_global.clone();
    for i in 0..n {
        for j in 0..n {
            b_mat[[i, j]] -= (1.0 - theta) * dt * k_global[[i, j]];
        }
    }

    // Initial condition
    let mut u_curr = Array1::from_shape_fn(n, |i| initial_condition(mesh.nodes[i]));

    // Apply initial Dirichlet BC
    apply_dirichlet_to_vec(&mut u_curr, left_bc, right_bc, n);

    let save_every = if options.save_every == 0 {
        1
    } else {
        options.save_every
    };
    let mut solutions = vec![u_curr.clone()];
    let mut times = vec![0.0];

    for step in 0..options.num_steps {
        let t_curr = step as f64 * dt;
        let t_next = (step + 1) as f64 * dt;

        // Compute load vector at current and next time
        let f_curr =
            assemble_load_vector(&mesh, &|x| f_source(x, t_curr), options.num_quad_points)?;
        let f_next =
            assemble_load_vector(&mesh, &|x| f_source(x, t_next), options.num_quad_points)?;

        // RHS = B * u_curr + dt * [(1-theta)*f_curr + theta*f_next]
        let mut rhs = Array1::zeros(n);
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += b_mat[[i, j]] * u_curr[j];
            }
            rhs[i] = sum + dt * ((1.0 - theta) * f_curr[i] + theta * f_next[i]);
        }

        // Apply Neumann BC contributions
        if let FEM1DBoundaryCondition::Neumann(flux) = right_bc {
            rhs[n - 1] += dt * theta * flux;
        }
        if let FEM1DBoundaryCondition::Neumann(flux) = left_bc {
            rhs[0] -= dt * theta * flux;
        }

        // Apply Dirichlet BCs to the system
        let mut a_mod = a_mat.clone();
        apply_dirichlet_bc_system(&mut a_mod, &mut rhs, left_bc, right_bc, n);

        // Solve
        u_curr = solve_linear_system_1d(&a_mod, &rhs)?;

        if (step + 1) % save_every == 0 || step + 1 == options.num_steps {
            solutions.push(u_curr.clone());
            times.push(t_next);
        }
    }

    // Error indicators at final time
    let (k_for_err, f_for_err) = assemble_system(
        &mesh,
        a_coeff,
        c_coeff,
        &|x| f_source(x, options.num_steps as f64 * dt),
        options.num_quad_points,
    )?;
    let error_indicators = compute_error_indicators(&mesh, &u_curr, &k_for_err, &f_for_err)?;

    Ok(FEM1DTransientResult {
        x: mesh.nodes,
        t: Array1::from_vec(times),
        u: solutions,
        error_indicators,
    })
}

// ---------------------------------------------------------------------------
// Error estimation and mesh refinement
// ---------------------------------------------------------------------------

/// Mesh refinement indicator based on error estimation
#[derive(Debug, Clone)]
pub struct RefinementIndicator {
    /// Element index
    pub element: usize,
    /// Error indicator value
    pub indicator: f64,
    /// Whether element should be refined
    pub refine: bool,
}

/// Mark elements for refinement using Doerfler marking strategy.
///
/// Marks the smallest set of elements whose error indicators sum to
/// at least `theta_fraction` of the total error.
pub fn mark_for_refinement(
    error_indicators: &Array1<f64>,
    theta_fraction: f64,
) -> Vec<RefinementIndicator> {
    let n_elem = error_indicators.len();
    let total_error: f64 = error_indicators.iter().sum();
    let threshold = theta_fraction * total_error;

    // Sort indices by decreasing error indicator
    let mut indices: Vec<usize> = (0..n_elem).collect();
    indices.sort_by(|&a, &b| {
        error_indicators[b]
            .partial_cmp(&error_indicators[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut cumulative = 0.0;
    let mut marked = vec![false; n_elem];
    for &idx in &indices {
        if cumulative >= threshold {
            break;
        }
        marked[idx] = true;
        cumulative += error_indicators[idx];
    }

    (0..n_elem)
        .map(|e| RefinementIndicator {
            element: e,
            indicator: error_indicators[e],
            refine: marked[e],
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Internal: assembly routines
// ---------------------------------------------------------------------------

/// Assemble global stiffness matrix and load vector
fn assemble_system(
    mesh: &Mesh1D,
    a_coeff: &dyn Fn(f64) -> f64,
    c_coeff: &dyn Fn(f64) -> f64,
    f_source: &dyn Fn(f64) -> f64,
    num_quad: usize,
) -> PDEResult<(Array2<f64>, Array1<f64>)> {
    let n = mesh.num_nodes();
    let mut k_global = Array2::zeros((n, n));
    let mut f_global = Array1::zeros(n);

    let (quad_pts, quad_wts) = gauss_legendre_1d(num_quad);

    for e in 0..mesh.num_elements() {
        let el = &mesh.elements[e];
        let n_local = el.len();
        let x_left = mesh.nodes[el[0]];
        let x_right = mesh.nodes[el[n_local - 1]];
        let h = x_right - x_left;

        if h.abs() < 1e-15 {
            continue;
        }

        // Local stiffness and load
        let mut k_local: Array2<f64> = Array2::zeros((n_local, n_local));
        let mut f_local: Array1<f64> = Array1::zeros(n_local);

        for q in 0..num_quad {
            let xi = quad_pts[q]; // reference coord in [-1, 1]
            let w = quad_wts[q];

            // Map to physical coordinate
            let x_phys = x_left + 0.5 * (1.0 + xi) * h;
            let jac = h / 2.0;

            // Evaluate basis functions and derivatives at xi
            let (phi, dphi_dxi) = eval_basis(xi, mesh.element_type);

            // Transform derivatives: dphi/dx = dphi/dxi * dxi/dx = dphi/dxi / jac
            let dphi_dx: Vec<f64> = dphi_dxi.iter().map(|&d| d / jac).collect();

            let a_val = a_coeff(x_phys);
            let c_val = c_coeff(x_phys);
            let f_val = f_source(x_phys);

            for i in 0..n_local {
                for j in 0..n_local {
                    k_local[[i, j]] +=
                        w * jac * (a_val * dphi_dx[i] * dphi_dx[j] + c_val * phi[i] * phi[j]);
                }
                f_local[i] += w * jac * f_val * phi[i];
            }
        }

        // Scatter local to global
        for i in 0..n_local {
            let gi = el[i];
            f_global[gi] += f_local[i];
            for j in 0..n_local {
                let gj = el[j];
                k_global[[gi, gj]] += k_local[[i, j]];
            }
        }
    }

    Ok((k_global, f_global))
}

/// Assemble global mass matrix
fn assemble_mass_matrix(mesh: &Mesh1D, num_quad: usize) -> PDEResult<Array2<f64>> {
    let n = mesh.num_nodes();
    let mut m_global = Array2::zeros((n, n));
    let (quad_pts, quad_wts) = gauss_legendre_1d(num_quad);

    for e in 0..mesh.num_elements() {
        let el = &mesh.elements[e];
        let n_local = el.len();
        let x_left = mesh.nodes[el[0]];
        let x_right = mesh.nodes[el[n_local - 1]];
        let h = x_right - x_left;

        if h.abs() < 1e-15 {
            continue;
        }

        let mut m_local: Array2<f64> = Array2::zeros((n_local, n_local));

        for q in 0..num_quad {
            let xi = quad_pts[q];
            let w = quad_wts[q];
            let jac = h / 2.0;
            let (phi, _) = eval_basis(xi, mesh.element_type);

            for i in 0..n_local {
                for j in 0..n_local {
                    m_local[[i, j]] += w * jac * phi[i] * phi[j];
                }
            }
        }

        for i in 0..n_local {
            let gi = el[i];
            for j in 0..n_local {
                let gj = el[j];
                m_global[[gi, gj]] += m_local[[i, j]];
            }
        }
    }

    Ok(m_global)
}

/// Assemble load vector only (for time-dependent forcing)
fn assemble_load_vector(
    mesh: &Mesh1D,
    f_source: &dyn Fn(f64) -> f64,
    num_quad: usize,
) -> PDEResult<Array1<f64>> {
    let n = mesh.num_nodes();
    let mut f_global = Array1::zeros(n);
    let (quad_pts, quad_wts) = gauss_legendre_1d(num_quad);

    for e in 0..mesh.num_elements() {
        let el = &mesh.elements[e];
        let n_local = el.len();
        let x_left = mesh.nodes[el[0]];
        let x_right = mesh.nodes[el[n_local - 1]];
        let h = x_right - x_left;

        if h.abs() < 1e-15 {
            continue;
        }

        let mut f_local: Array1<f64> = Array1::zeros(n_local);

        for q in 0..num_quad {
            let xi = quad_pts[q];
            let w = quad_wts[q];
            let x_phys = x_left + 0.5 * (1.0 + xi) * h;
            let jac = h / 2.0;
            let (phi, _) = eval_basis(xi, mesh.element_type);

            let f_val = f_source(x_phys);
            for i in 0..n_local {
                f_local[i] += w * jac * f_val * phi[i];
            }
        }

        for i in 0..n_local {
            f_global[el[i]] += f_local[i];
        }
    }

    Ok(f_global)
}

/// Evaluate basis functions and their derivatives on reference element [-1, 1]
fn eval_basis(xi: f64, element_type: FEM1DElementType) -> (Vec<f64>, Vec<f64>) {
    match element_type {
        FEM1DElementType::Linear => {
            let phi = vec![0.5 * (1.0 - xi), 0.5 * (1.0 + xi)];
            let dphi = vec![-0.5, 0.5];
            (phi, dphi)
        }
        FEM1DElementType::Quadratic => {
            // Quadratic Lagrange on [-1,1] with nodes at -1, 0, 1
            let phi = vec![0.5 * xi * (xi - 1.0), 1.0 - xi * xi, 0.5 * xi * (xi + 1.0)];
            let dphi = vec![xi - 0.5, -2.0 * xi, xi + 0.5];
            (phi, dphi)
        }
    }
}

/// Gauss-Legendre quadrature points and weights on [-1, 1]
fn gauss_legendre_1d(n: usize) -> (Vec<f64>, Vec<f64>) {
    match n {
        1 => (vec![0.0], vec![2.0]),
        2 => {
            let p = 1.0 / 3.0_f64.sqrt();
            (vec![-p, p], vec![1.0, 1.0])
        }
        3 => {
            let p = (3.0 / 5.0_f64).sqrt();
            (vec![-p, 0.0, p], vec![5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
        }
        4 => {
            let a = (3.0 / 7.0 - 2.0 / 7.0 * (6.0 / 5.0_f64).sqrt()).sqrt();
            let b = (3.0 / 7.0 + 2.0 / 7.0 * (6.0 / 5.0_f64).sqrt()).sqrt();
            let wa = (18.0 + 30.0_f64.sqrt()) / 36.0;
            let wb = (18.0 - 30.0_f64.sqrt()) / 36.0;
            (vec![-b, -a, a, b], vec![wb, wa, wa, wb])
        }
        _ => {
            // Default to 3-point
            let p = (3.0 / 5.0_f64).sqrt();
            (vec![-p, 0.0, p], vec![5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
        }
    }
}

/// Apply Dirichlet boundary conditions via penalty method
fn apply_dirichlet_bc(
    k: &mut Array2<f64>,
    f: &mut Array1<f64>,
    left_bc: &FEM1DBoundaryCondition,
    right_bc: &FEM1DBoundaryCondition,
    n: usize,
) {
    let penalty = 1e30;

    if let FEM1DBoundaryCondition::Dirichlet(val) = left_bc {
        k[[0, 0]] += penalty;
        f[0] += penalty * val;
    }
    if let FEM1DBoundaryCondition::Dirichlet(val) = right_bc {
        k[[n - 1, n - 1]] += penalty;
        f[n - 1] += penalty * val;
    }
}

/// Apply Dirichlet BC to system matrix and RHS for transient problems
fn apply_dirichlet_bc_system(
    a: &mut Array2<f64>,
    rhs: &mut Array1<f64>,
    left_bc: &FEM1DBoundaryCondition,
    right_bc: &FEM1DBoundaryCondition,
    n: usize,
) {
    if let FEM1DBoundaryCondition::Dirichlet(val) = left_bc {
        for j in 0..n {
            a[[0, j]] = 0.0;
        }
        a[[0, 0]] = 1.0;
        rhs[0] = *val;
    }
    if let FEM1DBoundaryCondition::Dirichlet(val) = right_bc {
        for j in 0..n {
            a[[n - 1, j]] = 0.0;
        }
        a[[n - 1, n - 1]] = 1.0;
        rhs[n - 1] = *val;
    }
}

/// Apply Dirichlet values to a solution vector
fn apply_dirichlet_to_vec(
    u: &mut Array1<f64>,
    left_bc: &FEM1DBoundaryCondition,
    right_bc: &FEM1DBoundaryCondition,
    n: usize,
) {
    if let FEM1DBoundaryCondition::Dirichlet(val) = left_bc {
        u[0] = *val;
    }
    if let FEM1DBoundaryCondition::Dirichlet(val) = right_bc {
        u[n - 1] = *val;
    }
}

/// Solve dense linear system Ax = b using Gaussian elimination with partial pivoting
fn solve_linear_system_1d(a: &Array2<f64>, b: &Array1<f64>) -> PDEResult<Array1<f64>> {
    let n = b.len();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    let mut a_copy = a.clone();
    let mut b_copy = b.clone();

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_val = a_copy[[k, k]].abs();
        let mut max_row = k;
        for i in k + 1..n {
            let val = a_copy[[i, k]].abs();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }
        if max_val < 1e-14 {
            return Err(PDEError::ComputationError(
                "Singular or near-singular FEM system matrix".to_string(),
            ));
        }
        // Swap rows
        if max_row != k {
            for j in k..n {
                let tmp = a_copy[[k, j]];
                a_copy[[k, j]] = a_copy[[max_row, j]];
                a_copy[[max_row, j]] = tmp;
            }
            let tmp = b_copy[k];
            b_copy[k] = b_copy[max_row];
            b_copy[max_row] = tmp;
        }
        // Eliminate
        for i in k + 1..n {
            let factor = a_copy[[i, k]] / a_copy[[k, k]];
            for j in k + 1..n {
                a_copy[[i, j]] -= factor * a_copy[[k, j]];
            }
            b_copy[i] -= factor * b_copy[k];
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in i + 1..n {
            sum += a_copy[[i, j]] * x[j];
        }
        x[i] = (b_copy[i] - sum) / a_copy[[i, i]];
    }

    Ok(x)
}

/// Compute element-wise residual-based error indicators
///
/// Uses the residual eta_E = h_E * ||R||_E where R is the residual on element E
fn compute_error_indicators(
    mesh: &Mesh1D,
    u: &Array1<f64>,
    k_global: &Array2<f64>,
    f_global: &Array1<f64>,
) -> PDEResult<Array1<f64>> {
    let n = mesh.num_nodes();
    let ne = mesh.num_elements();

    // Global residual r = f - K*u
    let mut residual = f_global.clone();
    for i in 0..n {
        for j in 0..n {
            residual[i] -= k_global[[i, j]] * u[j];
        }
    }

    // Element-wise indicators
    let mut indicators = Array1::zeros(ne);
    for e in 0..ne {
        let el = &mesh.elements[e];
        let h = mesh.element_size(e);

        // Sum squared residuals on this element's nodes
        let mut r_sq = 0.0;
        for &node in el {
            r_sq += residual[node] * residual[node];
        }

        // Error indicator: h^2 * ||r||^2 (simplification of proper element residual)
        indicators[e] = h * h * r_sq;
    }

    Ok(indicators)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_mesh_uniform_linear() {
        let mesh =
            Mesh1D::uniform(0.0, 1.0, 10, FEM1DElementType::Linear).expect("Should create mesh");
        assert_eq!(mesh.num_nodes(), 11);
        assert_eq!(mesh.num_elements(), 10);
        assert!((mesh.nodes[0] - 0.0).abs() < 1e-15);
        assert!((mesh.nodes[10] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_mesh_uniform_quadratic() {
        let mesh =
            Mesh1D::uniform(0.0, 1.0, 5, FEM1DElementType::Quadratic).expect("Should create mesh");
        assert_eq!(mesh.num_nodes(), 11);
        assert_eq!(mesh.num_elements(), 5);
    }

    #[test]
    fn test_steady_constant_solution() {
        // -u'' = 0 with u(0)=1, u(1)=1 => u=1 everywhere
        let result = solve_steady_1d(
            &|_| 1.0,
            &|_| 0.0,
            &|_| 0.0,
            0.0,
            1.0,
            10,
            &FEM1DBoundaryCondition::Dirichlet(1.0),
            &FEM1DBoundaryCondition::Dirichlet(1.0),
            &FEM1DSteadyOptions::default(),
        )
        .expect("Should succeed");

        for i in 0..result.x.len() {
            assert!(
                (result.u[i] - 1.0).abs() < 1e-8,
                "Node {i}: u={}, expected 1.0",
                result.u[i]
            );
        }
    }

    #[test]
    fn test_steady_linear_solution() {
        // -u'' = 0 with u(0)=0, u(1)=1 => u = x
        let result = solve_steady_1d(
            &|_| 1.0,
            &|_| 0.0,
            &|_| 0.0,
            0.0,
            1.0,
            10,
            &FEM1DBoundaryCondition::Dirichlet(0.0),
            &FEM1DBoundaryCondition::Dirichlet(1.0),
            &FEM1DSteadyOptions::default(),
        )
        .expect("Should succeed");

        for i in 0..result.x.len() {
            assert!(
                (result.u[i] - result.x[i]).abs() < 1e-6,
                "Node {i}: u={}, x={}, expected linear",
                result.u[i],
                result.x[i]
            );
        }
    }

    #[test]
    fn test_steady_poisson_1d() {
        // -u'' = 2 on [0,1] with u(0)=0, u(1)=0 => u = x(1-x)
        let result = solve_steady_1d(
            &|_| 1.0,
            &|_| 0.0,
            &|_| 2.0,
            0.0,
            1.0,
            20,
            &FEM1DBoundaryCondition::Dirichlet(0.0),
            &FEM1DBoundaryCondition::Dirichlet(0.0),
            &FEM1DSteadyOptions::default(),
        )
        .expect("Should succeed");

        for i in 0..result.x.len() {
            let x = result.x[i];
            let exact = x * (1.0 - x);
            assert!(
                (result.u[i] - exact).abs() < 1e-4,
                "Node {i}: u={}, exact={exact}",
                result.u[i]
            );
        }
    }

    #[test]
    fn test_steady_quadratic_elements() {
        // Quadratic elements should give exact solution for -u''=2
        let opts = FEM1DSteadyOptions {
            element_type: FEM1DElementType::Quadratic,
            num_quad_points: 3,
        };
        let result = solve_steady_1d(
            &|_| 1.0,
            &|_| 0.0,
            &|_| 2.0,
            0.0,
            1.0,
            10,
            &FEM1DBoundaryCondition::Dirichlet(0.0),
            &FEM1DBoundaryCondition::Dirichlet(0.0),
            &opts,
        )
        .expect("Should succeed");

        for i in 0..result.x.len() {
            let x = result.x[i];
            let exact = x * (1.0 - x);
            assert!(
                (result.u[i] - exact).abs() < 1e-6,
                "Quadratic node {i}: u={}, exact={exact}",
                result.u[i]
            );
        }
    }

    #[test]
    fn test_steady_neumann_bc() {
        // -u'' = 1 on [0,1] with u(0)=0, u'(1)=0
        // Exact: u = x - x^2/2
        let result = solve_steady_1d(
            &|_| 1.0,
            &|_| 0.0,
            &|_| 1.0,
            0.0,
            1.0,
            20,
            &FEM1DBoundaryCondition::Dirichlet(0.0),
            &FEM1DBoundaryCondition::Neumann(0.0),
            &FEM1DSteadyOptions::default(),
        )
        .expect("Should succeed");

        for i in 0..result.x.len() {
            let x = result.x[i];
            let exact = x - x * x / 2.0;
            assert!(
                (result.u[i] - exact).abs() < 0.02,
                "Neumann node {i}: u={}, exact={exact}",
                result.u[i]
            );
        }
    }

    #[test]
    fn test_steady_variable_coefficient() {
        // -d/dx[(1+x) du/dx] = 1, u(0)=0, u(1)=0
        // This tests variable diffusion coefficient
        let result = solve_steady_1d(
            &|x| 1.0 + x,
            &|_| 0.0,
            &|_| 1.0,
            0.0,
            1.0,
            40,
            &FEM1DBoundaryCondition::Dirichlet(0.0),
            &FEM1DBoundaryCondition::Dirichlet(0.0),
            &FEM1DSteadyOptions::default(),
        )
        .expect("Should succeed");

        // Solution should be positive inside [0,1]
        for i in 1..result.x.len() - 1 {
            assert!(
                result.u[i] > 0.0,
                "Interior node {i} should be positive: u={}",
                result.u[i]
            );
        }
    }

    #[test]
    fn test_transient_decay() {
        // du/dt = d2u/dx2, u(x,0) = sin(pi*x), u(0,t)=0, u(1,t)=0
        // Exact: u(x,t) = sin(pi*x) * exp(-pi^2*t)
        let opts = FEM1DTransientOptions {
            element_type: FEM1DElementType::Linear,
            num_quad_points: 3,
            theta: 0.5,
            dt: 0.001,
            num_steps: 100,
            save_every: 100,
        };
        let result = solve_transient_1d(
            &|_| 1.0,
            &|_| 0.0,
            &|_, _| 0.0,
            0.0,
            1.0,
            20,
            &|x| (PI * x).sin(),
            &FEM1DBoundaryCondition::Dirichlet(0.0),
            &FEM1DBoundaryCondition::Dirichlet(0.0),
            &opts,
        )
        .expect("Should succeed");

        let t_final = opts.dt * opts.num_steps as f64;
        let u_final = &result.u[result.u.len() - 1];
        let mid = result.x.len() / 2;
        let exact = (PI * 0.5).sin() * (-PI * PI * t_final).exp();
        assert!(
            (u_final[mid] - exact).abs() < 0.02,
            "Transient: u={}, exact={exact} (t={t_final})",
            u_final[mid]
        );
    }

    #[test]
    fn test_transient_steady_state() {
        // du/dt = u'' with u(x,0)=1, u(0)=1, u(1)=1
        // Should remain u=1 at all times
        let opts = FEM1DTransientOptions {
            dt: 0.01,
            num_steps: 50,
            save_every: 50,
            ..Default::default()
        };
        let result = solve_transient_1d(
            &|_| 1.0,
            &|_| 0.0,
            &|_, _| 0.0,
            0.0,
            1.0,
            10,
            &|_| 1.0,
            &FEM1DBoundaryCondition::Dirichlet(1.0),
            &FEM1DBoundaryCondition::Dirichlet(1.0),
            &opts,
        )
        .expect("Should succeed");

        let u_final = &result.u[result.u.len() - 1];
        for i in 0..u_final.len() {
            assert!(
                (u_final[i] - 1.0).abs() < 1e-6,
                "Steady: node {i}, u={}",
                u_final[i]
            );
        }
    }

    #[test]
    fn test_error_indicators_nonneg() {
        let result = solve_steady_1d(
            &|_| 1.0,
            &|_| 0.0,
            &|_| 2.0,
            0.0,
            1.0,
            10,
            &FEM1DBoundaryCondition::Dirichlet(0.0),
            &FEM1DBoundaryCondition::Dirichlet(0.0),
            &FEM1DSteadyOptions::default(),
        )
        .expect("Should succeed");

        for &val in result.error_indicators.iter() {
            assert!(val >= 0.0, "Error indicator must be non-negative");
        }
    }

    #[test]
    fn test_refinement_marking() {
        let indicators = Array1::from_vec(vec![1.0, 0.1, 0.5, 0.2, 0.05]);
        let marks = mark_for_refinement(&indicators, 0.5);
        // Total = 1.85, threshold = 0.925
        // Element 0 (1.0) alone exceeds 0.925
        assert!(marks[0].refine, "Largest element should be marked");
        assert_eq!(marks.len(), 5);
    }

    #[test]
    fn test_eval_basis_linear_partition() {
        // Basis functions should sum to 1 (partition of unity)
        for &xi in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            let (phi, _) = eval_basis(xi, FEM1DElementType::Linear);
            let sum: f64 = phi.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-14,
                "Linear partition of unity at xi={xi}: sum={sum}"
            );
        }
    }

    #[test]
    fn test_eval_basis_quadratic_partition() {
        for &xi in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            let (phi, _) = eval_basis(xi, FEM1DElementType::Quadratic);
            let sum: f64 = phi.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-14,
                "Quadratic partition of unity at xi={xi}: sum={sum}"
            );
        }
    }

    #[test]
    fn test_mass_matrix_symmetry() {
        let mesh = Mesh1D::uniform(0.0, 1.0, 5, FEM1DElementType::Linear).expect("mesh");
        let m = assemble_mass_matrix(&mesh, 3).expect("mass matrix");
        let n = m.shape()[0];
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (m[[i, j]] - m[[j, i]]).abs() < 1e-14,
                    "Mass matrix not symmetric at [{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn test_stiffness_matrix_symmetry() {
        let mesh = Mesh1D::uniform(0.0, 1.0, 5, FEM1DElementType::Linear).expect("mesh");
        let (k, _) = assemble_system(&mesh, &|_| 1.0, &|_| 0.0, &|_| 0.0, 3).expect("assembly");
        let n = k.shape()[0];
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (k[[i, j]] - k[[j, i]]).abs() < 1e-14,
                    "Stiffness matrix not symmetric at [{i},{j}]"
                );
            }
        }
    }
}
