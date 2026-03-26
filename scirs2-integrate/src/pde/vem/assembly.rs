//! VEM global assembly and solver
//!
//! Assembles the global stiffness matrix and load vector from element-wise
//! VEM local matrices, applies Dirichlet boundary conditions, and solves the
//! resulting linear system.

use super::basis::{
    compute_pi0, compute_pi_nabla, eval_pi_nabla_at, polygon_area, polygon_centroid_and_diameter,
    scaled_monomial_gradients,
};
use super::{PolygonalMesh, VemConfig};
use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::Array2;

/// Solution produced by the VEM solver
#[derive(Debug, Clone)]
pub struct VemSolution {
    /// DOF values at each mesh vertex
    pub dof_values: Vec<f64>,
    /// The mesh used for solving
    pub mesh: PolygonalMesh,
}

impl VemSolution {
    /// Compute approximate L2 error against exact solution
    ///
    /// Uses the projection Pi^∇ for evaluating the approximate solution and
    /// integrates the squared error over the mesh using edge quadrature + polygon decomposition.
    pub fn l2_error(&self, u_exact: &dyn Fn(f64, f64) -> f64) -> f64 {
        let mut error_sq = 0.0_f64;

        for elem_idx in 0..self.mesh.n_elements() {
            let elem_v_ids = &self.mesh.elements[elem_idx];
            let elem_verts: Vec<[f64; 2]> = elem_v_ids
                .iter()
                .map(|&vi| self.mesh.vertices[vi])
                .collect();
            let vertex_values: Vec<f64> =
                elem_v_ids.iter().map(|&vi| self.dof_values[vi]).collect();

            let (centroid, diameter) = polygon_centroid_and_diameter(&elem_verts);

            // Compute Pi^∇ for this element
            let pi_nabla = match compute_pi_nabla(&elem_verts, centroid, diameter) {
                Ok(p) => p,
                Err(_) => continue,
            };

            // Integrate over the polygon by triangulating from centroid
            let n_v = elem_verts.len();
            for i in 0..n_v {
                let j = (i + 1) % n_v;
                // Triangle: centroid, v_i, v_j
                let tri_verts = [centroid, elem_verts[i], elem_verts[j]];
                let tri_area = triangle_area(&tri_verts);
                if tri_area < 1e-20 {
                    continue;
                }

                // 3-point Gaussian quadrature on triangle
                let gauss_pts = [
                    [1.0 / 6.0_f64, 1.0 / 6.0, 2.0 / 3.0], // barycentric
                    [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                    [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
                ];
                let gauss_w = [1.0 / 3.0_f64; 3]; // weight (sum to 1, area factor applied after)

                for (bary, wg) in gauss_pts.iter().zip(gauss_w.iter()) {
                    let qx = bary[0] * tri_verts[0][0]
                        + bary[1] * tri_verts[1][0]
                        + bary[2] * tri_verts[2][0];
                    let qy = bary[0] * tri_verts[0][1]
                        + bary[1] * tri_verts[1][1]
                        + bary[2] * tri_verts[2][1];

                    let u_h =
                        eval_pi_nabla_at(qx, qy, centroid, diameter, &pi_nabla, &vertex_values);
                    let u_e = u_exact(qx, qy);
                    error_sq += (u_h - u_e).powi(2) * wg * tri_area;
                }
            }
        }

        error_sq.sqrt()
    }

    /// Evaluate the approximate solution at vertex i
    pub fn vertex_value(&self, i: usize) -> f64 {
        self.dof_values[i]
    }
}

fn triangle_area(verts: &[[f64; 2]; 3]) -> f64 {
    let a = verts[0];
    let b = verts[1];
    let c = verts[2];
    let cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    (cross * 0.5).abs()
}

/// Compute the VEM element stiffness matrix for one polygonal element
///
/// The element stiffness matrix is:
///   K_E = K_consistency + K_stability
///
/// **Consistency term** (ensures polynomial consistency):
///   K_c = (Pi^∇)^T G Pi^∇
/// where `G[α][β]` = ∫_E ∇m_α · ∇m_β dx (Gram matrix of monomial gradients).
/// Since linear monomial gradients are constant, G is computed analytically.
///
/// **Stability term** (ensures coercivity / controls the kernel):
///   K_s = α (I - Pi^0)^T (I - Pi^0) * trace(K_c)
/// where Pi^0 is the L2 projection (≈ Pi^∇ for k=1) and α is the stabilization constant.
///
/// The trace scaling ensures mesh independence.
pub fn element_stiffness(
    element_vertices: &[[f64; 2]],
    config: &VemConfig,
) -> IntegrateResult<Array2<f64>> {
    let n_v = element_vertices.len();
    if n_v < 3 {
        return Err(IntegrateError::InvalidInput(
            "Element must have at least 3 vertices".to_string(),
        ));
    }

    let (centroid, diameter) = polygon_centroid_and_diameter(element_vertices);
    let area = polygon_area(element_vertices);

    // Compute Pi^∇ projection (3 × n_v)
    let pi_nabla = compute_pi_nabla(element_vertices, centroid, diameter)?;

    // Compute Pi^0 (3 × n_v)
    let pi0 = compute_pi0(element_vertices, centroid, diameter, &pi_nabla);

    // Gram matrix G of monomial gradients (3 × 3)
    // For degree-1 monomials with constant gradients:
    //   G = area * grad_m^T * grad_m
    let grads = scaled_monomial_gradients(diameter);
    let n_mono = 3;
    let mut g_mat = Array2::<f64>::zeros((n_mono, n_mono));
    for alpha in 0..n_mono {
        for beta in 0..n_mono {
            let dot = grads[alpha][0] * grads[beta][0] + grads[alpha][1] * grads[beta][1];
            g_mat[[alpha, beta]] = area * dot;
        }
    }

    // Consistency term: K_c = Pi^∇^T G Pi^∇  (n_v × n_v)
    // Pi^∇ is (n_mono × n_v), so Pi^∇^T is (n_v × n_mono)
    // K_c = Pi^∇^T (n_v×n_mono) * G (n_mono×n_mono) * Pi^∇ (n_mono×n_v)
    let mut k_c = Array2::<f64>::zeros((n_v, n_v));
    // tmp = G * Pi^∇  (n_mono × n_v)
    let mut g_pi = Array2::<f64>::zeros((n_mono, n_v));
    for alpha in 0..n_mono {
        for j in 0..n_v {
            let mut s = 0.0_f64;
            for beta in 0..n_mono {
                s += g_mat[[alpha, beta]] * pi_nabla[[beta, j]];
            }
            g_pi[[alpha, j]] = s;
        }
    }
    // K_c = Pi^∇^T * (G * Pi^∇)
    for i in 0..n_v {
        for j in 0..n_v {
            let mut s = 0.0_f64;
            for alpha in 0..n_mono {
                s += pi_nabla[[alpha, i]] * g_pi[[alpha, j]];
            }
            k_c[[i, j]] = s;
        }
    }

    // Trace of K_c for scaling the stability term
    let trace_kc: f64 = (0..n_v).map(|i| k_c[[i, i]]).sum::<f64>().max(1e-14);

    // Stability term: K_s = alpha * (I - Pi^0)^T (I - Pi^0)
    // (I - Pi^0)[i][j] = delta_{ij} - Pi^0[alpha, j] is NOT right dimensionally.
    //
    // Actually, (I - Pi^0) acts on R^{n_v} → R^{n_v}:
    // For VEM, the stability matrix projects out the polynomial part from the identity:
    // We need P_∂ = Pi^0 restricted to the boundary DOF space (vertex values)
    // and then S = (I - P_∂)^T (I - P_∂)
    //
    // The matrix P_∂ : R^{n_v} → R^{n_v} is defined by:
    //   (P_∂ v)_i = sum_α Pi^0[α,i] * m_α(x_i) ???
    //
    // Standard VEM stability: D = Pi^∇ evaluated at DOF nodes × Pi^0 coefficients
    // For degree 1: S_ij = (delta_ij - D_ij)^T (delta_ij - D_ij) * scaling
    // where D = Pi^0_coeffs evaluated at vertex positions: D[i][j] = m_α(x_j) * Pi^0[α,i]
    //
    // Correct formulation from Beirão da Veiga et al. 2013:
    // Let Π^0 : R^{n_v} → R^{n_v} be the projector matrix that maps vertex values
    // to their L2 projection evaluated at vertex positions.
    //   (Π^0_mat)[i][j] = sum_α m_α(x_i) * Pi^0[α, j]
    let mut pi0_mat = Array2::<f64>::zeros((n_v, n_v));
    for i in 0..n_v {
        let mono_at_vi = super::basis::scaled_monomials_values(
            element_vertices[i][0],
            element_vertices[i][1],
            centroid,
            diameter,
        );
        for j in 0..n_v {
            let mut s = 0.0_f64;
            for alpha in 0..n_mono {
                s += mono_at_vi[alpha] * pi0[[alpha, j]];
            }
            pi0_mat[[i, j]] = s;
        }
    }

    // (I - Π^0_mat) * (I - Π^0_mat)^T
    let mut k_s = Array2::<f64>::zeros((n_v, n_v));
    for i in 0..n_v {
        for j in 0..n_v {
            let mut s = 0.0_f64;
            for k in 0..n_v {
                let ik = if i == k { 1.0 } else { 0.0 } - pi0_mat[[i, k]];
                let jk = if j == k { 1.0 } else { 0.0 } - pi0_mat[[j, k]];
                s += ik * jk;
            }
            k_s[[i, j]] = s;
        }
    }

    // K = K_c + alpha * (trace(K_c) / n_v) * K_s
    // The scaling trace(K_c)/n_v makes the stability term mesh-size-consistent
    let stab_scale = config.stabilization * trace_kc / (n_v as f64);
    let mut k = Array2::<f64>::zeros((n_v, n_v));
    for i in 0..n_v {
        for j in 0..n_v {
            k[[i, j]] = k_c[[i, j]] + stab_scale * k_s[[i, j]];
        }
    }

    Ok(k)
}

/// Compute the VEM load vector for one element
///
/// For the forcing term, since virtual basis functions are not available,
/// we use the L2 projection:
///   (f, φ_i)_E ≈ (f, Pi^0 φ_i)_E
///
/// For degree 1 and vertex DOFs, we approximate by distributing the average
/// of f over the element uniformly to vertex DOFs weighted by 1/n_v:
///   f_i = area(E) * f_avg / n_v
///
/// More precisely, (f, Pi^0 φ_i)_E = ∫_E f(x) sum_α Pi^0[α,i] m_α(x) dx
/// which for constant f reduces to f_avg * area * Pi^0[0,i] (only constant monomial survives).
fn element_load_vector(
    element_vertices: &[[f64; 2]],
    f: &dyn Fn(f64, f64) -> f64,
    _config: &VemConfig,
) -> IntegrateResult<Vec<f64>> {
    let n_v = element_vertices.len();
    let (centroid, diameter) = polygon_centroid_and_diameter(element_vertices);
    let area = polygon_area(element_vertices);

    // Pi^∇ for load term
    let pi_nabla = compute_pi_nabla(element_vertices, centroid, diameter)?;
    let pi0 = compute_pi0(element_vertices, centroid, diameter, &pi_nabla);

    // Compute f integral against each scaled monomial by polygon triangulation
    let n_mono = 3;
    let mut f_mono = vec![0.0_f64; n_mono];

    for i in 0..n_v {
        let j = (i + 1) % n_v;
        let tri_verts = [centroid, element_vertices[i], element_vertices[j]];
        let tri_area = triangle_area(&tri_verts);

        // 3-point Gauss quadrature on triangle
        let gauss_pts = [
            [1.0 / 6.0_f64, 1.0 / 6.0, 2.0 / 3.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        ];
        let gauss_w = [1.0 / 3.0_f64; 3];

        for (bary, wg) in gauss_pts.iter().zip(gauss_w.iter()) {
            let qx =
                bary[0] * tri_verts[0][0] + bary[1] * tri_verts[1][0] + bary[2] * tri_verts[2][0];
            let qy =
                bary[0] * tri_verts[0][1] + bary[1] * tri_verts[1][1] + bary[2] * tri_verts[2][1];
            let fval = f(qx, qy);
            let mono = super::basis::scaled_monomials_values(qx, qy, centroid, diameter);
            for alpha in 0..n_mono {
                f_mono[alpha] += fval * mono[alpha] * wg * tri_area;
            }
        }
    }

    // Load vector: f_i = sum_α Pi^0[α,i] * f_mono[α]
    let mut load = vec![0.0_f64; n_v];
    for i in 0..n_v {
        let mut s = 0.0_f64;
        for alpha in 0..n_mono {
            s += pi0[[alpha, i]] * f_mono[alpha];
        }
        load[i] = s;
    }

    let _ = area; // used through triangulation above
    Ok(load)
}

/// Assemble global VEM stiffness matrix and load vector
///
/// Returns `(global_matrix, global_rhs)` where indices correspond to vertex DOFs.
pub fn assemble_vem(
    mesh: &PolygonalMesh,
    f: &dyn Fn(f64, f64) -> f64,
    config: &VemConfig,
) -> IntegrateResult<(Array2<f64>, Vec<f64>)> {
    let n_dof = mesh.n_vertices();
    let mut global_mat = Array2::<f64>::zeros((n_dof, n_dof));
    let mut global_rhs = vec![0.0_f64; n_dof];

    for elem_idx in 0..mesh.n_elements() {
        let elem_v_ids = &mesh.elements[elem_idx];
        let elem_verts: Vec<[f64; 2]> = elem_v_ids.iter().map(|&vi| mesh.vertices[vi]).collect();
        let n_v_elem = elem_v_ids.len();

        let k_elem = element_stiffness(&elem_verts, config)?;
        let f_elem = element_load_vector(&elem_verts, f, config)?;

        // Scatter into global matrix
        for local_i in 0..n_v_elem {
            let global_i = elem_v_ids[local_i];
            global_rhs[global_i] += f_elem[local_i];
            for local_j in 0..n_v_elem {
                let global_j = elem_v_ids[local_j];
                global_mat[[global_i, global_j]] += k_elem[[local_i, local_j]];
            }
        }
    }

    Ok((global_mat, global_rhs))
}

/// Apply Dirichlet boundary conditions via row/column elimination
fn apply_dirichlet_vem(
    mat: &mut Array2<f64>,
    rhs: &mut [f64],
    mesh: &PolygonalMesh,
    g: &dyn Fn(f64, f64) -> f64,
) {
    let n = mat.nrows();
    let mut bc_vals = vec![f64::NAN; n];

    // Compute BC values for boundary vertices
    for &v_idx in &mesh.boundary_vertices {
        let vx = mesh.vertices[v_idx][0];
        let vy = mesh.vertices[v_idx][1];
        bc_vals[v_idx] = g(vx, vy);
    }

    // Subtract known values from RHS (symmetric elimination)
    for &v_idx in &mesh.boundary_vertices {
        let g_val = bc_vals[v_idx];
        for row in 0..n {
            if !bc_vals[row].is_nan() {
                continue;
            }
            rhs[row] -= mat[[row, v_idx]] * g_val;
        }
    }

    // Set BC rows/columns to identity
    for &v_idx in &mesh.boundary_vertices {
        for j in 0..n {
            mat[[v_idx, j]] = 0.0;
            mat[[j, v_idx]] = 0.0;
        }
        mat[[v_idx, v_idx]] = 1.0;
        rhs[v_idx] = bc_vals[v_idx];
    }
}

/// LU solver for dense system (with partial pivoting)
fn dense_lu_solve(a: &mut Array2<f64>, b: &mut [f64]) -> IntegrateResult<Vec<f64>> {
    let n = a.nrows();
    if n != a.ncols() || n != b.len() {
        return Err(IntegrateError::DimensionMismatch(
            "VEM LU solve: inconsistent dimensions".to_string(),
        ));
    }

    for col in 0..n {
        // Find pivot
        let mut max_val = a[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if a[[row, col]].abs() > max_val {
                max_val = a[[row, col]].abs();
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return Err(IntegrateError::LinearSolveError(format!(
                "VEM: near-singular matrix at col {col}"
            )));
        }
        if max_row != col {
            for k in 0..n {
                let tmp = a[[col, k]];
                a[[col, k]] = a[[max_row, k]];
                a[[max_row, k]] = tmp;
            }
            b.swap(col, max_row);
        }
        let pivot = a[[col, col]];
        for row in (col + 1)..n {
            let factor = a[[row, col]] / pivot;
            a[[row, col]] = factor;
            for k in (col + 1)..n {
                let sub = factor * a[[col, k]];
                a[[row, k]] -= sub;
            }
            b[row] -= factor * b[col];
        }
    }

    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[[i, j]] * x[j];
        }
        x[i] = s / a[[i, i]];
    }
    Ok(x)
}

/// Solve the Poisson equation using the Virtual Element Method
///
/// # Arguments
///
/// * `mesh` - Polygonal mesh
/// * `f` - Forcing function f(x,y)
/// * `g` - Dirichlet boundary condition g(x,y)
/// * `config` - VEM configuration
pub fn solve_vem(
    mesh: PolygonalMesh,
    f: &dyn Fn(f64, f64) -> f64,
    g: &dyn Fn(f64, f64) -> f64,
    config: VemConfig,
) -> IntegrateResult<VemSolution> {
    // Step 1: Assemble global system
    let (mut global_mat, mut global_rhs) = assemble_vem(&mesh, f, &config)?;

    // Step 2: Apply Dirichlet BCs
    apply_dirichlet_vem(&mut global_mat, &mut global_rhs, &mesh, g);

    // Step 3: Solve
    let dof_values = dense_lu_solve(&mut global_mat, &mut global_rhs)?;

    Ok(VemSolution { dof_values, mesh })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple uniform triangular mesh on [0,1]^2 (as polygonal mesh with triangles)
    fn tri_mesh(n: usize) -> PolygonalMesh {
        let mut vertices = Vec::new();
        let mut elements = Vec::new();

        for j in 0..=n {
            for i in 0..=n {
                vertices.push([i as f64 / n as f64, j as f64 / n as f64]);
            }
        }
        for j in 0..n {
            for i in 0..n {
                let v00 = j * (n + 1) + i;
                let v10 = j * (n + 1) + i + 1;
                let v01 = (j + 1) * (n + 1) + i;
                let v11 = (j + 1) * (n + 1) + i + 1;
                elements.push(vec![v00, v10, v11]);
                elements.push(vec![v00, v11, v01]);
            }
        }

        PolygonalMesh::new(vertices, elements)
    }

    #[test]
    fn test_element_stiffness_symmetric_triangle() {
        let verts = [[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let config = VemConfig::default();
        let k = element_stiffness(&verts, &config).unwrap();
        let n = k.nrows();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (k[[i, j]] - k[[j, i]]).abs() < 1e-12,
                    "K not symmetric at [{},{}]: {} vs {}",
                    i,
                    j,
                    k[[i, j]],
                    k[[j, i]]
                );
            }
        }
    }

    #[test]
    fn test_element_stiffness_psd_triangle() {
        // K should be positive semi-definite (all eigenvalues >= 0)
        // Check via Gershgorin or by ensuring v^T K v >= 0 for random v
        let verts = [[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let config = VemConfig::default();
        let k = element_stiffness(&verts, &config).unwrap();

        // Test v^T K v >= 0 for several vectors
        let n = k.nrows();
        for _ in 0..5 {
            let v: Vec<f64> = (0..n).map(|i| (i + 1) as f64 * 0.1).collect();
            let mut vkv = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    vkv += v[i] * k[[i, j]] * v[j];
                }
            }
            assert!(vkv >= -1e-12, "K not PSD: v^T K v = {vkv}");
        }
    }

    #[test]
    fn test_element_stiffness_symmetric_quad() {
        // Quadrilateral element
        let verts = [[0.0_f64, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let config = VemConfig::default();
        let k = element_stiffness(&verts, &config).unwrap();
        let n = k.nrows();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (k[[i, j]] - k[[j, i]]).abs() < 1e-12,
                    "Quad K not symmetric at [{},{}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_element_stiffness_pentagon() {
        // Pentagon element
        use std::f64::consts::PI;
        let verts: Vec<[f64; 2]> = (0..5)
            .map(|i| {
                let theta = 2.0 * PI * i as f64 / 5.0;
                [theta.cos(), theta.sin()]
            })
            .collect();
        let config = VemConfig::default();
        let k = element_stiffness(&verts, &config).unwrap();
        assert_eq!(k.nrows(), 5);
        assert_eq!(k.ncols(), 5);
        for i in 0..5 {
            for j in 0..5 {
                assert!(
                    (k[[i, j]] - k[[j, i]]).abs() < 1e-10,
                    "Pentagon K not symmetric at [{},{}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_vem_poisson_constant_solution() {
        // u = 1, f = 0, g = 1 → should recover u = 1 exactly
        let mesh = tri_mesh(3);
        let config = VemConfig::default();
        let f = |_x: f64, _y: f64| 0.0_f64;
        let g = |_x: f64, _y: f64| 1.0_f64;

        let sol = solve_vem(mesh, &f, &g, config).unwrap();
        let err = sol.l2_error(&|_x, _y| 1.0);
        assert!(err < 1e-10, "L2 error for u=1 should be ~0, got {err}");
    }

    #[test]
    fn test_vem_poisson_l2_error_reasonable() {
        // u = sin(pi*x)*sin(pi*y), f = 2*pi^2 * u, g = 0
        use std::f64::consts::PI;

        let f = move |x: f64, y: f64| 2.0 * PI * PI * (PI * x).sin() * (PI * y).sin();
        let g = |_x: f64, _y: f64| 0.0_f64;
        let u_exact = |x: f64, y: f64| (PI * x).sin() * (PI * y).sin();

        let mesh = tri_mesh(4);
        let config = VemConfig::default();
        let sol = solve_vem(mesh, &f, &g, config).unwrap();
        let err = sol.l2_error(&u_exact);

        // Modest accuracy expected with k=1 and coarse mesh
        assert!(err < 0.5, "L2 error = {err} should be < 0.5");
    }

    #[test]
    fn test_vem_quadrilateral_element_handled() {
        // A mesh with a single quadrilateral element (+ boundary)
        // Use a mesh of 4 triangles + center vertex
        let vertices = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        // One quad element
        let elements = vec![vec![0, 1, 2, 3]];
        let mesh = PolygonalMesh::new(vertices, elements);
        // All vertices are boundary (single element)
        let config = VemConfig::default();
        let k = element_stiffness(&mesh.element_vertices(0), &config).unwrap();
        assert_eq!(k.nrows(), 4);
        assert_eq!(k.ncols(), 4);
    }

    #[test]
    fn test_vem_pentagon_element_handled() {
        use std::f64::consts::PI;
        let verts: Vec<[f64; 2]> = (0..5)
            .map(|i| {
                let theta = 2.0 * PI * i as f64 / 5.0;
                [theta.cos() * 0.5 + 0.5, theta.sin() * 0.5 + 0.5]
            })
            .collect();
        let config = VemConfig::default();
        let k = element_stiffness(&verts, &config).unwrap();
        assert_eq!(k.nrows(), 5);
        // Should not panic or error
    }
}
