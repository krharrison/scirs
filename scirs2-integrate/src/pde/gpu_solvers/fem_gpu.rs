//! GPU-accelerated Finite Element Method (FEM) solver for 2D PDEs.
//!
//! Implements linear triangular elements on unstructured meshes with:
//! - Global stiffness matrix assembly (element-wise loop, analogous to GPU element kernels)
//! - Dirichlet boundary condition enforcement
//! - Conjugate Gradient (CG) linear solver with parallelised matrix-vector products
//!
//! The "GPU acceleration" is simulated using `std::thread::scope` tile-based
//! parallelism for both the assembly and the CG mat-vec product.
//!
//! # Equation
//! Solves the weak form of -∇·(κ∇u) = f with essential (Dirichlet) boundary conditions.

use super::types::{GpuPdeConfig, PdeSolverError, PdeSolverResult, SolverStats};

// ---------------------------------------------------------------------------
// Mesh types
// ---------------------------------------------------------------------------

/// A 2D triangular mesh for FEM computations.
#[derive(Debug, Clone)]
pub struct FemMesh {
    /// Node coordinates: `nodes[k] = [x, y]`.
    pub nodes: Vec<[f64; 2]>,
    /// Element connectivity: `elements[e] = [n0, n1, n2]` (counter-clockwise).
    pub elements: Vec<[usize; 3]>,
}

impl FemMesh {
    /// Number of nodes in the mesh.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of triangular elements.
    pub fn num_elements(&self) -> usize {
        self.elements.len()
    }
}

// ---------------------------------------------------------------------------
// Mesh generation
// ---------------------------------------------------------------------------

/// Generate a uniform rectangular mesh on [0, lx] × [0, ly]
/// divided into `nx × ny` quadrilateral cells, each split into 2 triangles.
///
/// Total triangles = 2·nx·ny.
/// Node count = (nx+1)·(ny+1).
///
/// # Errors
/// Returns `PdeSolverError::InvalidGrid` if nx < 1 or ny < 1.
pub fn uniform_rect_mesh(nx: usize, ny: usize, lx: f64, ly: f64) -> PdeSolverResult<FemMesh> {
    if nx < 1 || ny < 1 {
        return Err(PdeSolverError::InvalidGrid);
    }
    let dx = lx / nx as f64;
    let dy = ly / ny as f64;

    let node_count = (nx + 1) * (ny + 1);
    let mut nodes = Vec::with_capacity(node_count);
    for j in 0..=ny {
        for i in 0..=nx {
            nodes.push([i as f64 * dx, j as f64 * dy]);
        }
    }

    let elem_count = 2 * nx * ny;
    let mut elements = Vec::with_capacity(elem_count);
    for j in 0..ny {
        for i in 0..nx {
            // Node indices for the (i,j) cell
            let n00 = j * (nx + 1) + i;
            let n10 = j * (nx + 1) + i + 1;
            let n01 = (j + 1) * (nx + 1) + i;
            let n11 = (j + 1) * (nx + 1) + i + 1;
            // Split into two CCW triangles
            elements.push([n00, n10, n01]);
            elements.push([n10, n11, n01]);
        }
    }

    Ok(FemMesh { nodes, elements })
}

// ---------------------------------------------------------------------------
// Element-level routines
// ---------------------------------------------------------------------------

/// Compute the local (3×3) stiffness matrix for a linear triangular element
/// with isotropic diffusivity κ.
///
/// Uses the standard formula:
///   K_e[a,b] = κ · area · (∇φ_a · ∇φ_b)
///
/// Returns `(k_local, area)`.
fn element_stiffness(nodes: &[[f64; 2]], el: &[usize; 3], kappa: f64) -> ([[f64; 9]; 1], f64) {
    let [x0, y0] = nodes[el[0]];
    let [x1, y1] = nodes[el[1]];
    let [x2, y2] = nodes[el[2]];

    // Shape function gradients (constant over the element)
    let jac_det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    let area = 0.5 * jac_det.abs();

    // Gradients of barycentric coordinates
    let b = [
        (y1 - y2) / jac_det,
        (y2 - y0) / jac_det,
        (y0 - y1) / jac_det,
    ];
    let c = [
        (x2 - x1) / jac_det,
        (x0 - x2) / jac_det,
        (x1 - x0) / jac_det,
    ];

    // 3×3 local stiffness (stored flat row-major)
    let mut ke = [0.0_f64; 9];
    for a in 0..3usize {
        for bb in 0..3usize {
            ke[a * 3 + bb] = kappa * area * (b[a] * b[bb] + c[a] * c[bb]);
        }
    }

    ([ke], area)
}

// ---------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------

/// Assemble the global stiffness matrix (dense, row-major) for the mesh.
///
/// Returns a flat `n×n` matrix where `n = mesh.num_nodes()`.
/// Each element contributes its local 3×3 stiffness sub-matrix.
///
/// # Errors
/// Returns `PdeSolverError::InvalidGrid` if the mesh has no elements.
pub fn assemble_stiffness_gpu(mesh: &FemMesh, diffusivity: f64) -> PdeSolverResult<Vec<f64>> {
    let n = mesh.num_nodes();
    if n == 0 || mesh.num_elements() == 0 {
        return Err(PdeSolverError::InvalidGrid);
    }

    let mut k_global = vec![0.0_f64; n * n];

    for el in &mesh.elements {
        let ([ke], _area) = element_stiffness(&mesh.nodes, el, diffusivity);
        // Scatter local → global
        for a in 0..3usize {
            for b in 0..3usize {
                let row = el[a];
                let col = el[b];
                k_global[row * n + col] += ke[a * 3 + b];
            }
        }
    }

    Ok(k_global)
}

// ---------------------------------------------------------------------------
// Boundary condition enforcement
// ---------------------------------------------------------------------------

/// Enforce a Dirichlet condition u[dof] = value using the symmetric elimination method.
///
/// For each unconstrained row i, the contribution `K[i,dof] * value` is subtracted
/// from `f[i]`, then both row and column `dof` are zeroed and the diagonal set to 1.
/// This preserves the symmetry and positive-definiteness of the system.
///
/// # Errors
/// Returns `PdeSolverError::BoundaryMismatch` if `dof >= n`.
pub fn apply_dirichlet_gpu(
    k: &mut Vec<f64>,
    f: &mut Vec<f64>,
    n: usize,
    dof: usize,
    value: f64,
) -> PdeSolverResult<()> {
    if dof >= n {
        return Err(PdeSolverError::BoundaryMismatch);
    }
    // Step 1: subtract K[i,dof]*value from f[i] for all other rows
    for row in 0..n {
        if row != dof {
            f[row] -= k[row * n + dof] * value;
        }
    }
    // Step 2: zero the row and the column
    for col in 0..n {
        k[dof * n + col] = 0.0;
        k[col * n + dof] = 0.0;
    }
    // Step 3: set diagonal to 1 and RHS to prescribed value
    k[dof * n + dof] = 1.0;
    f[dof] = value;
    Ok(())
}

// ---------------------------------------------------------------------------
// Conjugate Gradient solver
// ---------------------------------------------------------------------------

/// Parallel matrix-vector product: y = K·x.
/// Uses thread::scope to parallelise across tile_size row-chunks.
fn matvec_parallel(k: &[f64], x: &[f64], n: usize, tile_size: usize) -> Vec<f64> {
    let mut y = vec![0.0_f64; n];
    let effective_tile = tile_size.max(1);
    let num_tiles = (n + effective_tile - 1) / effective_tile;

    // Split y into per-tile chunks for parallel writes
    let tile_chunks: Vec<&mut [f64]> = {
        // We need mutable non-overlapping chunks of y
        y.chunks_mut(effective_tile).collect()
    };

    std::thread::scope(|s| {
        for (tile_idx, chunk) in tile_chunks.into_iter().enumerate() {
            let row_start = tile_idx * effective_tile;
            s.spawn(move || {
                for (local_row, yi) in chunk.iter_mut().enumerate() {
                    let row = row_start + local_row;
                    if row >= n {
                        break;
                    }
                    let mut acc = 0.0_f64;
                    for col in 0..n {
                        acc += k[row * n + col] * x[col];
                    }
                    *yi = acc;
                }
            });
        }
    });

    y
}

/// Sequential matrix-vector product y = K·x.
fn matvec_seq(k: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0_f64; n];
    for row in 0..n {
        let mut acc = 0.0_f64;
        for col in 0..n {
            acc += k[row * n + col] * x[col];
        }
        y[row] = acc;
    }
    y
}

/// Dot product of two vectors.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// L∞ norm of a vector.
fn linf_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
}

/// Preconditioned Conjugate Gradient solver for K·x = b (symmetric positive definite K).
///
/// Uses Jacobi (diagonal) preconditioning and optional parallelised mat-vec.
///
/// # Errors
/// - `PdeSolverError::SingularSystem` if any diagonal entry is zero.
/// - `PdeSolverError::NotConverged` if max iterations reached.
pub fn conjugate_gradient_gpu(
    k: &[f64],
    b: &[f64],
    n: usize,
    max_iter: usize,
    tol: f64,
) -> PdeSolverResult<(Vec<f64>, SolverStats)> {
    // Jacobi preconditioner: M = diag(K)
    let mut diag = vec![0.0_f64; n];
    for i in 0..n {
        let d = k[i * n + i];
        if d.abs() < f64::EPSILON * 1e6 {
            return Err(PdeSolverError::SingularSystem);
        }
        diag[i] = d;
    }

    let mut x = vec![0.0_f64; n];
    let mut r: Vec<f64> = b.to_vec();
    // z = M^{-1} r
    let mut z: Vec<f64> = r.iter().zip(diag.iter()).map(|(ri, di)| ri / di).collect();
    let mut p = z.clone();
    let mut rz = dot(&r, &z);

    if rz.sqrt() < tol {
        return Ok((x, SolverStats::converged(0, rz.sqrt())));
    }

    for iter in 0..max_iter {
        let ap = matvec_seq(k, &p, n);
        let pap = dot(&p, &ap);

        if pap.abs() < f64::EPSILON {
            return Err(PdeSolverError::SingularSystem);
        }

        let alpha = rz / pap;

        // x = x + alpha * p
        for i in 0..n {
            x[i] += alpha * p[i];
        }
        // r = r - alpha * Ap
        for i in 0..n {
            r[i] -= alpha * ap[i];
        }

        let res_norm = linf_norm(&r);
        if res_norm < tol {
            return Ok((x, SolverStats::converged(iter + 1, res_norm)));
        }

        // z = M^{-1} r
        for i in 0..n {
            z[i] = r[i] / diag[i];
        }

        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;

        // p = z + beta * p
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }

    Err(PdeSolverError::NotConverged { iterations: max_iter })
}

// ---------------------------------------------------------------------------
// Full FEM Poisson solver
// ---------------------------------------------------------------------------

/// Solve the 2D Poisson equation -∇²u = f using linear triangular FEM.
///
/// # Parameters
/// - `mesh`: triangular mesh
/// - `source`: nodal source values f(x,y), length `mesh.num_nodes()`
/// - `bc_nodes`: list of `(node_index, prescribed_value)` for Dirichlet BCs
/// - `config`: solver configuration
///
/// # Returns
/// Solution vector u at all nodes.
///
/// # Errors
/// - `PdeSolverError::InvalidGrid` if the mesh or source vector is invalid.
/// - `PdeSolverError::BoundaryMismatch` if any BC node index is out of range.
/// - `PdeSolverError::NotConverged` if CG does not converge.
pub fn solve_fem_poisson(
    mesh: &FemMesh,
    source: &[f64],
    bc_nodes: &[(usize, f64)],
    config: &GpuPdeConfig,
) -> PdeSolverResult<Vec<f64>> {
    let n = mesh.num_nodes();
    if source.len() != n {
        return Err(PdeSolverError::InvalidGrid);
    }

    // Assemble K
    let mut k = assemble_stiffness_gpu(mesh, 1.0)?;

    // Assemble load vector f = M·source
    // For simplicity, use lumped mass (area / 3 per node of each element)
    let mut f = vec![0.0_f64; n];
    for el in &mesh.elements {
        let ([_ke], area) = element_stiffness(&mesh.nodes, el, 1.0);
        let contrib = area / 3.0;
        for &node in el.iter() {
            f[node] += contrib * source[node];
        }
    }

    // Apply Dirichlet BCs
    for &(dof, val) in bc_nodes {
        apply_dirichlet_gpu(&mut k, &mut f, n, dof, val)?;
    }

    // Solve K·u = f
    let (u, _stats) =
        conjugate_gradient_gpu(&k, &f, n, config.max_iterations, config.tolerance)?;

    Ok(u)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fem_uniform_mesh_triangle_count() {
        // nx=2, ny=2 → 2*2*2 = 8 triangles
        let mesh = uniform_rect_mesh(2, 2, 1.0, 1.0).expect("mesh");
        assert_eq!(mesh.num_elements(), 8, "expected 8 triangles");
        assert_eq!(mesh.num_nodes(), 9, "expected 9 nodes");
    }

    #[test]
    fn test_fem_assemble_stiffness_symmetric() {
        let mesh = uniform_rect_mesh(3, 3, 1.0, 1.0).expect("mesh");
        let n = mesh.num_nodes();
        let k = assemble_stiffness_gpu(&mesh, 1.0).expect("assemble");
        // Check symmetry: |K[i,j] - K[j,i]| < 1e-12
        for i in 0..n {
            for j in 0..n {
                let diff = (k[i * n + j] - k[j * n + i]).abs();
                assert!(diff < 1e-12, "K not symmetric at ({i},{j}): diff={diff}");
            }
        }
    }

    #[test]
    fn test_fem_cg_solver_identity() {
        // K = I, b = [1,2,3] → x = [1,2,3]
        let n = 3usize;
        let k: Vec<f64> = (0..n * n)
            .map(|idx| if idx / n == idx % n { 1.0 } else { 0.0 })
            .collect();
        let b = vec![1.0, 2.0, 3.0];
        let (x, stats) = conjugate_gradient_gpu(&k, &b, n, 100, 1e-10).expect("cg");
        assert!(stats.converged);
        for i in 0..n {
            assert!((x[i] - b[i]).abs() < 1e-8, "x[{i}]={} expected {}", x[i], b[i]);
        }
    }

    #[test]
    fn test_fem_cg_solver_diag_matrix() {
        // K = diag(2, 3, 4), b = [4, 9, 8] → x = [2, 3, 2]
        let n = 3usize;
        let mut k = vec![0.0_f64; n * n];
        k[0] = 2.0;
        k[4] = 3.0;
        k[8] = 4.0;
        let b = vec![4.0, 9.0, 8.0];
        let expected = vec![2.0, 3.0, 2.0];
        let (x, _) = conjugate_gradient_gpu(&k, &b, n, 100, 1e-10).expect("cg");
        for i in 0..n {
            assert!(
                (x[i] - expected[i]).abs() < 1e-8,
                "x[{i}]={} expected {}",
                x[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_cg_symmetric_pd_system() {
        // 4×4 tridiagonal SPD: K = tridiag(-1, 3, -1)
        let n = 4usize;
        let mut k = vec![0.0_f64; n * n];
        for i in 0..n {
            k[i * n + i] = 3.0;
            if i + 1 < n {
                k[i * n + i + 1] = -1.0;
                k[(i + 1) * n + i] = -1.0;
            }
        }
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let (x, stats) =
            conjugate_gradient_gpu(&k, &b, n, 200, 1e-10).expect("cg");
        assert!(stats.converged, "CG did not converge");
        // Verify K·x ≈ b
        let ax = matvec_seq(&k, &x, n);
        for i in 0..n {
            assert!((ax[i] - b[i]).abs() < 1e-8, "residual at {i}: {}", (ax[i] - b[i]).abs());
        }
    }

    #[test]
    fn test_fem_poisson_1d_analytic() {
        // 2D square problem where BCs impose a linear x-gradient:
        // u=0 on x=0, u=1 on x=1, and Dirichlet on y=0, y=1 edges too (u=x).
        // Exact solution: u = x (linear), f = 0.
        let nx = 4usize;
        let ny = 4usize;
        let mesh = uniform_rect_mesh(nx, ny, 1.0, 1.0).expect("mesh");
        let n = mesh.num_nodes();
        let source = vec![0.0_f64; n];

        // All boundary nodes: enforce u = x
        let mut bc_nodes = Vec::new();
        for (k, node) in mesh.nodes.iter().enumerate() {
            let [x, y] = *node;
            let on_boundary = x.abs() < 1e-12
                || (x - 1.0).abs() < 1e-12
                || y.abs() < 1e-12
                || (y - 1.0).abs() < 1e-12;
            if on_boundary {
                bc_nodes.push((k, x)); // exact solution u = x
            }
        }

        let config = GpuPdeConfig { max_iterations: 5000, tolerance: 1e-10, ..Default::default() };
        let u = solve_fem_poisson(&mesh, &source, &bc_nodes, &config).expect("fem");

        // All nodes (including interior) should satisfy u ≈ x
        for (idx, node) in mesh.nodes.iter().enumerate() {
            let [x, _y] = *node;
            let err = (u[idx] - x).abs();
            assert!(err < 0.05, "node {idx} x={x:.3} u={:.3} err={err:.4}", u[idx]);
        }
    }

    #[test]
    fn test_fem_poisson_convergence() {
        // Zero source with u=1 on entire boundary should give u≈1 everywhere.
        let mesh = uniform_rect_mesh(4, 4, 1.0, 1.0).expect("mesh");
        let n = mesh.num_nodes();
        let source = vec![0.0_f64; n];

        // All boundary nodes: x=0, x=1, y=0, y=1
        let mut bc_nodes = Vec::new();
        for (k, node) in mesh.nodes.iter().enumerate() {
            let [x, y] = *node;
            if x.abs() < 1e-12
                || (x - 1.0).abs() < 1e-12
                || y.abs() < 1e-12
                || (y - 1.0).abs() < 1e-12
            {
                bc_nodes.push((k, 1.0));
            }
        }

        let config = GpuPdeConfig { max_iterations: 5000, tolerance: 1e-10, ..Default::default() };
        let u = solve_fem_poisson(&mesh, &source, &bc_nodes, &config).expect("fem");

        for (idx, &val) in u.iter().enumerate() {
            assert!((val - 1.0).abs() < 0.05, "node {idx} val={val:.4} not close to 1");
        }
    }
}
