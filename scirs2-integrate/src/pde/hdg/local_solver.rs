//! Local HDG solver: element matrices and static condensation
//!
//! Implements the Hybridizable Discontinuous Galerkin (HDG) method.
//!
//! ## Formulation
//!
//! For the Poisson equation −∇²u = f in Ω, u = g on ∂Ω, using the HDG
//! primal formulation with degree k=1:
//!
//! **Global system** (skeleton): vertex DOFs λ, assembled from gradient stiffness
//!   K_global = ∑_K A_grad_K  (standard Galerkin P1 stiffness)
//!   f_global = ∑_K f_vol_K   (standard Galerkin P1 load)
//!
//! **HDG augmentation**: the local recovery uses the HDG stabilization τ:
//!   Given λ (= global CG solution), recover element-wise u_K with HDG penalty
//!
//! For k=1 with τ→∞, this is exactly standard CG P1.
//! For finite τ, it provides a discontinuous local post-processing step.
//!
//! The key matrices:
//!   `A_grad_K[i,j]` = ∫_K ∇φ_i·∇φ_j dx  (gradient stiffness)
//!   `f_vol_K[i]`    = ∫_K f φ_i dx         (load vector)
//!   `C_K[i,j]`      = τ ∑_f ∫_f ψ_i ψ_j ds (trace Gram matrix)
//!   `B_K[i,j]`      = τ ∑_f ∫_f φ_i ψ_j ds (volume-trace coupling)
//!
//! The Schur complement for the HDG stabilized local recovery:
//!   A_HDG_K = A_grad_K + C_K
//!   u_K = A_HDG_K^{-1} (f_vol_K + B_K λ_K)

use super::{jacobian_det, jacobian_inv, ref_to_physical, triangle_gauss_quadrature_3pt, HdgMesh};
use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2};

/// Local HDG matrices for one triangular element
#[derive(Debug, Clone)]
pub struct LocalHdgMatrices {
    /// Gradient stiffness `A_grad[i,j]` = ∫_K ∇φ_i·∇φ_j dx
    pub a_kk: Array2<f64>,
    /// Inverse of (A_grad + C_K) for back-substitution
    pub a_kk_inv: Array2<f64>,
    /// Volume-to-trace coupling B_K (used in recovery)
    pub b_k: Array2<f64>,
    /// Trace Gram matrix C_K = τ ∑_f ∫_f ψ_i ψ_j ds
    pub c_k: Array2<f64>,
    /// Element load `f_vol[i]` = ∫_K f φ_i dx
    pub f_vol: Array1<f64>,
    /// Schur complement (structural placeholder, = C_K for this formulation)
    pub schur: Array2<f64>,
    /// Condensed load (structural placeholder)
    pub rhs_face: Array1<f64>,
    /// Global vertex indices for this element
    pub vertex_indices: [usize; 3],
}

fn p1_ref(xi: f64, eta: f64) -> ([f64; 3], [f64; 3], [f64; 3]) {
    (
        [1.0 - xi - eta, xi, eta],
        [-1.0, 1.0, 0.0],
        [-1.0, 0.0, 1.0],
    )
}

fn phys_grad(dxi: &[f64; 3], deta: &[f64; 3], ji: &[[f64; 2]; 2]) -> [[f64; 2]; 3] {
    let mut g = [[0.0_f64; 2]; 3];
    for i in 0..3 {
        g[i][0] = ji[0][0] * dxi[i] + ji[1][0] * deta[i];
        g[i][1] = ji[0][1] * dxi[i] + ji[1][1] * deta[i];
    }
    g
}

fn gauss4() -> ([f64; 4], [f64; 4]) {
    // 4-point Gauss on [0,1]
    let a = 0.5 - 0.5 * (3.0 / 7.0 - 2.0 / 7.0 * (6.0 / 5.0_f64).sqrt()).sqrt();
    let b = 0.5 + 0.5 * (3.0 / 7.0 - 2.0 / 7.0 * (6.0 / 5.0_f64).sqrt()).sqrt();
    let c = 0.5 - 0.5 * (3.0 / 7.0 + 2.0 / 7.0 * (6.0 / 5.0_f64).sqrt()).sqrt();
    let d = 0.5 + 0.5 * (3.0 / 7.0 + 2.0 / 7.0 * (6.0 / 5.0_f64).sqrt()).sqrt();
    let w1 = 0.5 * (18.0 + (30.0_f64).sqrt()) / 36.0;
    let w2 = 0.5 * (18.0 - (30.0_f64).sqrt()) / 36.0;
    ([a, b, c, d], [w1, w1, w2, w2])
}

fn ref_edge(lf: usize, t: f64) -> (f64, f64) {
    match lf {
        0 => (1.0 - t, t),
        1 => (0.0, t),
        2 => (t, 0.0),
        _ => (0.0, 0.0),
    }
}

fn elen(lf: usize, v: &[[f64; 2]; 3]) -> f64 {
    let (a, b) = match lf {
        0 => (v[1], v[2]),
        1 => (v[0], v[2]),
        2 => (v[0], v[1]),
        _ => (v[0], v[1]),
    };
    let d = [b[0] - a[0], b[1] - a[1]];
    (d[0] * d[0] + d[1] * d[1]).sqrt()
}

fn everts(lf: usize) -> [usize; 2] {
    match lf {
        0 => [1, 2],
        1 => [0, 2],
        2 => [0, 1],
        _ => [0, 1],
    }
}

fn inv3(m: &Array2<f64>) -> IntegrateResult<Array2<f64>> {
    let (a, b, c) = (m[[0, 0]], m[[0, 1]], m[[0, 2]]);
    let (d, e, f) = (m[[1, 0]], m[[1, 1]], m[[1, 2]]);
    let (g, h, k) = (m[[2, 0]], m[[2, 1]], m[[2, 2]]);
    let det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
    if det.abs() < 1e-14 {
        return Err(IntegrateError::LinearSolveError(format!(
            "Singular 3x3 det={det:.2e}"
        )));
    }
    let mut inv = Array2::<f64>::zeros((3, 3));
    inv[[0, 0]] = (e * k - f * h) / det;
    inv[[0, 1]] = (c * h - b * k) / det;
    inv[[0, 2]] = (b * f - c * e) / det;
    inv[[1, 0]] = (f * g - d * k) / det;
    inv[[1, 1]] = (a * k - c * g) / det;
    inv[[1, 2]] = (c * d - a * f) / det;
    inv[[2, 0]] = (d * h - e * g) / det;
    inv[[2, 1]] = (b * g - a * h) / det;
    inv[[2, 2]] = (a * e - b * d) / det;
    Ok(inv)
}

/// Compute local HDG matrices for element `elem_idx`
pub fn local_matrices(
    elem_idx: usize,
    mesh: &HdgMesh,
    tau: f64,
    f_func: &dyn Fn(f64, f64) -> f64,
) -> IntegrateResult<LocalHdgMatrices> {
    let elem = &mesh.elements[elem_idx];
    let v: [[f64; 2]; 3] = [
        mesh.vertices[elem[0]],
        mesh.vertices[elem[1]],
        mesh.vertices[elem[2]],
    ];
    let det = jacobian_det(&v);
    let ji = jacobian_inv(&v);

    // Gradient stiffness: A_grad[i,j] = ∫_K ∇φ_i·∇φ_j dx
    let mut a_grad = Array2::<f64>::zeros((3, 3));
    {
        let (qps, wts) = triangle_gauss_quadrature_3pt();
        for (qp, &w) in qps.iter().zip(wts.iter()) {
            let (_, dxi, deta) = p1_ref(qp[0], qp[1]);
            let gp = phys_grad(&dxi, &deta, &ji);
            let jw = det * w;
            for i in 0..3 {
                for j in 0..3 {
                    a_grad[[i, j]] += (gp[i][0] * gp[j][0] + gp[i][1] * gp[j][1]) * jw;
                }
            }
        }
    }

    // Load vector: f_vol[i] = ∫_K f φ_i dx
    let mut f_vol = Array1::<f64>::zeros(3);
    {
        let (qps, wts) = triangle_gauss_quadrature_3pt();
        for (qp, &w) in qps.iter().zip(wts.iter()) {
            let phys = ref_to_physical(qp[0], qp[1], &v);
            let fval = f_func(phys[0], phys[1]);
            let (phi, _, _) = p1_ref(qp[0], qp[1]);
            let jw = det * w;
            for i in 0..3 {
                f_vol[i] += fval * phi[i] * jw;
            }
        }
    }

    // Trace matrices C_K and B_K
    let (ep, ew) = gauss4();
    let mut c_k = Array2::<f64>::zeros((3, 3));
    let mut b_k = Array2::<f64>::zeros((3, 3));
    for lf in 0..3usize {
        let hf = elen(lf, &v);
        let [va, vb] = everts(lf);
        for (&t, &w) in ep.iter().zip(ew.iter()) {
            let pa = 1.0 - t;
            let pb = t;
            let ds = hf * w;
            c_k[[va, va]] += tau * pa * pa * ds;
            c_k[[va, vb]] += tau * pa * pb * ds;
            c_k[[vb, va]] += tau * pb * pa * ds;
            c_k[[vb, vb]] += tau * pb * pb * ds;
            let (xi, eta) = ref_edge(lf, t);
            let (phi, _, _) = p1_ref(xi, eta);
            for i in 0..3 {
                b_k[[i, va]] += tau * phi[i] * pa * ds;
                b_k[[i, vb]] += tau * phi[i] * pb * ds;
            }
        }
    }

    // For back-substitution, we need (A_grad + C_K)^{-1}
    let mut a_hdg = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            a_hdg[[i, j]] = a_grad[[i, j]] + c_k[[i, j]];
        }
    }
    let a_kk_inv = inv3(&a_hdg)?;

    // Schur complement (symmetric structure, used for tests)
    let mut schur = c_k.clone();
    let mut tmp = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                tmp[[i, j]] += a_kk_inv[[i, k]] * b_k[[k, j]];
            }
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0_f64;
            for k in 0..3 {
                s += b_k[[k, i]] * tmp[[k, j]];
            }
            schur[[i, j]] -= s;
        }
    }

    let mut tmpf = [0.0_f64; 3];
    for i in 0..3 {
        for k in 0..3 {
            tmpf[i] += a_kk_inv[[i, k]] * f_vol[k];
        }
    }
    let mut rhs_face = Array1::<f64>::zeros(3);
    for i in 0..3 {
        for k in 0..3 {
            rhs_face[i] += b_k[[k, i]] * tmpf[k];
        }
    }

    Ok(LocalHdgMatrices {
        a_kk: a_grad, // expose gradient stiffness for global assembly
        a_kk_inv,     // inverse of (a_grad + c_k) for local recovery
        b_k,
        c_k,
        f_vol,
        schur,
        rhs_face,
        vertex_indices: [elem[0], elem[1], elem[2]],
    })
}

/// Recover volume solution given trace (vertex) values λ
///
/// Uses the HDG-stabilized local problem: (A_grad + C_K) u = f_vol + B_K λ
pub fn solve_local(lambda_k: &[f64; 3], local: &LocalHdgMatrices) -> Vec<f64> {
    let mut rhs = local.f_vol.to_vec();
    for i in 0..3 {
        for j in 0..3 {
            rhs[i] += local.b_k[[i, j]] * lambda_k[j];
        }
    }
    let mut u = vec![0.0_f64; 3];
    for i in 0..3 {
        for k in 0..3 {
            u[i] += local.a_kk_inv[[i, k]] * rhs[k];
        }
    }
    u
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_mesh() -> HdgMesh {
        let vertices = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let elements = vec![[0, 1, 2]];
        HdgMesh::new(vertices, elements)
    }

    #[test]
    fn test_local_matrices_akk_positive_diagonal() {
        let m = local_matrices(0, &ref_mesh(), 1.0, &|_, _| 0.0).unwrap();
        for i in 0..3 {
            assert!(m.a_kk[[i, i]] >= 0.0, "a_kk diag[{}]={}", i, m.a_kk[[i, i]]);
        }
    }

    #[test]
    fn test_akk_symmetric() {
        let m = local_matrices(0, &ref_mesh(), 1.0, &|_, _| 0.0).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (m.a_kk[[i, j]] - m.a_kk[[j, i]]).abs() < 1e-12,
                    "A[{},{}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_local_matrices_schur_symmetric() {
        let m = local_matrices(0, &ref_mesh(), 1.0, &|_, _| 1.0).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (m.schur[[i, j]] - m.schur[[j, i]]).abs() < 1e-12,
                    "S[{},{}]={} S[{},{}]={}",
                    i,
                    j,
                    m.schur[[i, j]],
                    j,
                    i,
                    m.schur[[j, i]]
                );
            }
        }
    }

    #[test]
    fn test_local_matrices_ck_diagonal() {
        let m = local_matrices(0, &ref_mesh(), 1.0, &|_, _| 0.0).unwrap();
        for i in 0..3 {
            assert!(m.c_k[[i, i]] > 0.0, "C_K diag[{}]={}", i, m.c_k[[i, i]]);
        }
    }

    #[test]
    fn test_solve_local_zero_trace() {
        let m = local_matrices(0, &ref_mesh(), 1.0, &|_, _| 2.0).unwrap();
        let u = solve_local(&[0.0, 0.0, 0.0], &m);
        assert_eq!(u.len(), 3);
    }
}
