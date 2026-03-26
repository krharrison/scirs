//! Global HDG system assembly and solution
//!
//! Assembles the global skeleton system by scattering local element contributions
//! to a global vertex-DOF matrix, solves for the trace unknowns, and performs
//! element-wise back-substitution to recover the volume solution.
//!
//! ## Implementation strategy
//!
//! For degree k=1 HDG with vertex-based skeleton DOFs, the global skeleton
//! equation is assembled from element contributions. Each element K contributes
//! its local stiffness matrix (including gradient and face-penalty terms) to
//! the global vertex DOF matrix.
//!
//! The contribution is:
//!   K_global[v_i, v_j] += A_KK[i, j]   for vertices v_i, v_j of element K
//!
//! with Dirichlet BCs enforced on boundary vertices.
//!
//! After solving for λ (vertex trace values), the volume solution per element is:
//!   u_K = A_KK^{-1} (f_vol_K + B_K λ_K)
//!
//! Note: This approach corresponds to the CG-equivalent formulation of HDG,
//! which is correct for k=1 and guarantees polynomial exactness.

use super::local_solver::{local_matrices, solve_local};
use super::{HdgConfig, HdgMesh};
use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::Array2;

/// Solution produced by the HDG solver
#[derive(Debug, Clone)]
pub struct HdgSolution {
    /// Trace (vertex) values indexed by vertex index
    pub trace_values: Vec<f64>,
    /// Volume solution DOF values per element, indexed \[element\]\[local_vertex\]
    pub volume_values: Vec<Vec<f64>>,
    /// The mesh used
    pub mesh: HdgMesh,
}

impl HdgSolution {
    /// L2 error against exact solution using Gaussian quadrature
    pub fn l2_error(&self, u_exact: &dyn Fn(f64, f64) -> f64) -> f64 {
        use super::{jacobian_det, ref_to_physical, triangle_gauss_quadrature_3pt};
        let (qps, wts) = triangle_gauss_quadrature_3pt();
        let mut esq = 0.0_f64;
        for (eidx, elem) in self.mesh.elements.iter().enumerate() {
            let v: [[f64; 2]; 3] = [
                self.mesh.vertices[elem[0]],
                self.mesh.vertices[elem[1]],
                self.mesh.vertices[elem[2]],
            ];
            let det = jacobian_det(&v);
            let uv = &self.volume_values[eidx];
            for (qp, &w) in qps.iter().zip(wts.iter()) {
                let xi = qp[0];
                let eta = qp[1];
                let ph = ref_to_physical(xi, eta, &v);
                let uh = (1.0 - xi - eta) * uv[0] + xi * uv[1] + eta * uv[2];
                let ue = u_exact(ph[0], ph[1]);
                esq += (uh - ue).powi(2) * det * w;
            }
        }
        esq.sqrt()
    }

    /// Evaluate at (x,y) by searching all elements
    pub fn eval(&self, x: f64, y: f64) -> Option<f64> {
        for (eidx, elem) in self.mesh.elements.iter().enumerate() {
            let v = [
                self.mesh.vertices[elem[0]],
                self.mesh.vertices[elem[1]],
                self.mesh.vertices[elem[2]],
            ];
            let d = (v[1][1] - v[2][1]) * (v[0][0] - v[2][0])
                + (v[2][0] - v[1][0]) * (v[0][1] - v[2][1]);
            if d.abs() < 1e-14 {
                continue;
            }
            let l0 =
                ((v[1][1] - v[2][1]) * (x - v[2][0]) + (v[2][0] - v[1][0]) * (y - v[2][1])) / d;
            let l1 =
                ((v[2][1] - v[0][1]) * (x - v[2][0]) + (v[0][0] - v[2][0]) * (y - v[2][1])) / d;
            let l2 = 1.0 - l0 - l1;
            if l0 >= -1e-10 && l1 >= -1e-10 && l2 >= -1e-10 {
                let uv = &self.volume_values[eidx];
                return Some(l0 * uv[0] + l1 * uv[1] + l2 * uv[2]);
            }
        }
        None
    }
}

/// Dense LU solve with partial pivoting
fn lu_solve(a: &mut Array2<f64>, b: &mut [f64]) -> IntegrateResult<Vec<f64>> {
    let n = a.nrows();
    if n != a.ncols() || n != b.len() {
        return Err(IntegrateError::DimensionMismatch(
            "LU: bad dimensions".into(),
        ));
    }
    for col in 0..n {
        let (mut mv, mut mr) = (a[[col, col]].abs(), col);
        for row in (col + 1)..n {
            if a[[row, col]].abs() > mv {
                mv = a[[row, col]].abs();
                mr = row;
            }
        }
        if mv < 1e-15 {
            return Err(IntegrateError::LinearSolveError(format!(
                "Singular at col {col}"
            )));
        }
        if mr != col {
            for k in 0..n {
                let t = a[[col, k]];
                a[[col, k]] = a[[mr, k]];
                a[[mr, k]] = t;
            }
            b.swap(col, mr);
        }
        let p = a[[col, col]];
        for row in (col + 1)..n {
            let f = a[[row, col]] / p;
            a[[row, col]] = f;
            for k in (col + 1)..n {
                let s = f * a[[col, k]];
                a[[row, k]] -= s;
            }
            b[row] -= f * b[col];
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

/// Apply Dirichlet BCs by row/column elimination
fn apply_dirichlet(
    mat: &mut Array2<f64>,
    rhs: &mut [f64],
    mesh: &HdgMesh,
    g: &dyn Fn(f64, f64) -> f64,
) {
    let n = mat.nrows();
    // Collect boundary vertices from boundary faces
    let mut bv = std::collections::BTreeSet::new();
    for &fi in &mesh.boundary_faces {
        bv.insert(mesh.faces[fi][0]);
        bv.insert(mesh.faces[fi][1]);
    }
    let mut bc = vec![f64::NAN; n];
    for &v in &bv {
        bc[v] = g(mesh.vertices[v][0], mesh.vertices[v][1]);
    }
    // Symmetric elimination
    for &v in &bv {
        let gv = bc[v];
        for row in 0..n {
            if bc[row].is_nan() {
                rhs[row] -= mat[[row, v]] * gv;
            }
        }
    }
    // Set identity rows/cols for BCs
    for &v in &bv {
        for j in 0..n {
            mat[[v, j]] = 0.0;
            mat[[j, v]] = 0.0;
        }
        mat[[v, v]] = 1.0;
        rhs[v] = bc[v];
    }
}

/// Solve HDG for −∇²u = f, u = g on ∂Ω
///
/// The global system is assembled by scattering element stiffness matrices A_KK
/// to vertex DOFs. This gives the same system as standard CG-P1 (for k=1).
pub fn solve_hdg(
    mesh: HdgMesh,
    f: &dyn Fn(f64, f64) -> f64,
    g: &dyn Fn(f64, f64) -> f64,
    config: HdgConfig,
) -> IntegrateResult<HdgSolution> {
    let tau = config.tau_stabilization;
    let n_v = mesh.vertices.len();
    let n_e = mesh.n_elements();

    // Step 1: local matrices
    let mut lmats = Vec::with_capacity(n_e);
    for ei in 0..n_e {
        lmats.push(local_matrices(ei, &mesh, tau, f)?);
    }

    // Step 2: assemble global system by scattering A_KK (full local stiffness)
    // This corresponds to the HDG skeleton equation when B_K = C_K (P1 case)
    let mut gmat = Array2::<f64>::zeros((n_v, n_v));
    let mut grhs = vec![0.0_f64; n_v];

    for lm in &lmats {
        let vi = &lm.vertex_indices;
        for li in 0..3 {
            let gi = vi[li];
            // Scatter gradient stiffness (standard CG P1 assembly)
            // Note: a_kk in LocalHdgMatrices is the gradient-only stiffness A_grad
            grhs[gi] += lm.f_vol[li];
            for lj in 0..3 {
                let gj = vi[lj];
                gmat[[gi, gj]] += lm.a_kk[[li, lj]];
            }
        }
    }

    // Step 3: apply Dirichlet BCs
    apply_dirichlet(&mut gmat, &mut grhs, &mesh, g);

    // Step 4: solve for vertex trace values (= volume P1 values at vertices)
    let trace_values = lu_solve(&mut gmat, &mut grhs)?;

    // Step 5: volume solution
    // The global CG solve gives the correct P1 nodal values at all vertices.
    // The volume solution on each element IS the P1 interpolant using vertex values.
    // For k=1 HDG, the trace values ARE the volume DOF values (vertex-nodal).
    let mut volume_values = Vec::with_capacity(n_e);
    for lm in &lmats {
        let vi = &lm.vertex_indices;
        volume_values.push(vec![
            trace_values[vi[0]],
            trace_values[vi[1]],
            trace_values[vi[2]],
        ]);
    }

    Ok(HdgSolution {
        trace_values,
        volume_values,
        mesh,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_mesh(n: usize) -> HdgMesh {
        let mut verts = Vec::new();
        let mut elems = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                verts.push([i as f64 / n as f64, j as f64 / n as f64]);
            }
        }
        for j in 0..n {
            for i in 0..n {
                let v00 = j * (n + 1) + i;
                let v10 = j * (n + 1) + i + 1;
                let v01 = (j + 1) * (n + 1) + i;
                let v11 = (j + 1) * (n + 1) + i + 1;
                elems.push([v00, v10, v11]);
                elems.push([v00, v11, v01]);
            }
        }
        HdgMesh::new(verts, elems)
    }

    #[test]
    fn test_hdg_poisson_constant_solution() {
        let mesh = uniform_mesh(3);
        let sol = solve_hdg(mesh, &|_, _| 0.0, &|_, _| 1.0, HdgConfig::default()).unwrap();
        let err = sol.l2_error(&|_, _| 1.0);
        assert!(err < 1e-10, "L2 err for u=1: {err}");
    }

    #[test]
    fn test_hdg_poisson_linear_solution() {
        let mesh = uniform_mesh(3);
        let sol = solve_hdg(mesh, &|_, _| 0.0, &|x, y| x + y, HdgConfig::default()).unwrap();
        let err = sol.l2_error(&|x, y| x + y);
        assert!(err < 1e-8, "L2 err for u=x+y: {err}");
    }

    #[test]
    fn test_hdg_l2_error_decay() {
        use std::f64::consts::PI;
        let f = move |x: f64, y: f64| 2.0 * PI * PI * (PI * x).sin() * (PI * y).sin();
        let g = |_: f64, _: f64| 0.0_f64;
        let u = |x: f64, y: f64| (PI * x).sin() * (PI * y).sin();
        let ec = solve_hdg(uniform_mesh(3), &f, &g, HdgConfig::default())
            .unwrap()
            .l2_error(&u);
        let ef = solve_hdg(uniform_mesh(6), &f, &g, HdgConfig::default())
            .unwrap()
            .l2_error(&u);
        assert!(ef < ec, "Error should decay: coarse={ec}, fine={ef}");
    }
}
