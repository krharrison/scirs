//! Hybridizable Discontinuous Galerkin (HDG) method for diffusion-dominated problems
//!
//! The HDG method is a class of discontinuous Galerkin methods that introduces
//! a single-valued trace variable on the skeleton of the mesh (the inter-element
//! boundaries). This hybridization reduces the globally coupled unknowns to only
//! the trace (face/edge) degrees of freedom, with element-wise volume unknowns
//! recovered via local static condensation.
//!
//! ## Problem formulation
//!
//! Solve the Poisson equation:
//!   −∇²u = f   in Ω
//!       u = g   on ∂Ω
//!
//! ## Key properties
//!
//! * Compact stencil: global system only involves face/skeleton unknowns
//! * Static condensation: volume unknowns computed element-locally after global solve
//! * Superconvergence: for polynomial degree k, the flux error converges as O(h^(k+1))
//!   and a post-processed solution converges as O(h^(k+2))
//! * Stabilization parameter τ > 0 ensures well-posedness
//!
//! ## References
//!
//! * Cockburn, B., Gopalakrishnan, J., & Lazarov, R. (2009). Unified hybridization of
//!   discontinuous Galerkin, mixed, and continuous Galerkin methods for second order
//!   elliptic problems. *SIAM J. Numer. Anal.*, 47(2), 1319-1365.

pub mod global_system;
pub mod local_solver;

pub use global_system::{solve_hdg, HdgSolution};
pub use local_solver::LocalHdgMatrices;

use crate::error::IntegrateError;

/// Configuration for the HDG solver
#[derive(Debug, Clone)]
pub struct HdgConfig {
    /// Polynomial degree of the approximation space (default: 1)
    pub polynomial_degree: usize,
    /// Stabilization parameter τ > 0 (default: 1.0)
    /// Larger τ provides more numerical dissipation; smaller τ allows more oscillations
    pub tau_stabilization: f64,
    /// Tolerance for linear solvers and convergence checks (default: 1e-10)
    pub tol: f64,
}

impl Default for HdgConfig {
    fn default() -> Self {
        HdgConfig {
            polynomial_degree: 1,
            tau_stabilization: 1.0,
            tol: 1e-10,
        }
    }
}

/// A triangular mesh for HDG computations
///
/// The mesh consists of triangular elements, faces (edges), and boundary faces.
/// Face orientation is consistent: face `[a, b]` with `a < b`.
#[derive(Debug, Clone)]
pub struct HdgMesh {
    /// Vertices of the mesh, each as `[x, y]`
    pub vertices: Vec<[f64; 2]>,
    /// Triangular elements, each as three vertex indices `[v0, v1, v2]`
    pub elements: Vec<[usize; 3]>,
    /// All faces (edges), each as two vertex indices `[v0, v1]` (v0 < v1)
    pub faces: Vec<[usize; 2]>,
    /// Indices into `faces` that lie on the domain boundary
    pub boundary_faces: Vec<usize>,
}

impl HdgMesh {
    /// Construct mesh from vertices and elements, automatically computing faces.
    ///
    /// Faces shared by two elements are interior; faces belonging to only one element
    /// are boundary faces.
    pub fn new(vertices: Vec<[f64; 2]>, elements: Vec<[usize; 3]>) -> Self {
        use std::collections::HashMap;

        // Build face list: map from sorted vertex pair to face index
        let mut face_map: HashMap<[usize; 2], usize> = HashMap::new();
        let mut faces: Vec<[usize; 2]> = Vec::new();
        let mut face_count: HashMap<[usize; 2], usize> = HashMap::new();

        for elem in &elements {
            let local_faces = [
                [elem[0].min(elem[1]), elem[0].max(elem[1])],
                [elem[1].min(elem[2]), elem[1].max(elem[2])],
                [elem[0].min(elem[2]), elem[0].max(elem[2])],
            ];
            for face in &local_faces {
                let count = face_count.entry(*face).or_insert(0);
                *count += 1;
                if !face_map.contains_key(face) {
                    let idx = faces.len();
                    face_map.insert(*face, idx);
                    faces.push(*face);
                }
            }
        }

        let boundary_faces = faces
            .iter()
            .enumerate()
            .filter(|(_, f)| face_count.get(*f).copied().unwrap_or(0) == 1)
            .map(|(i, _)| i)
            .collect();

        HdgMesh {
            vertices,
            elements,
            faces,
            boundary_faces,
        }
    }

    /// Return the face indices (into `self.faces`) for a given element
    pub fn element_faces(&self, elem_idx: usize) -> [usize; 3] {
        use std::collections::HashMap;
        // Build face map on the fly (cached externally in practice)
        let mut face_map: HashMap<[usize; 2], usize> = HashMap::new();
        for (i, f) in self.faces.iter().enumerate() {
            face_map.insert(*f, i);
        }
        let elem = &self.elements[elem_idx];
        let local_faces = [
            [elem[0].min(elem[1]), elem[0].max(elem[1])],
            [elem[1].min(elem[2]), elem[1].max(elem[2])],
            [elem[0].min(elem[2]), elem[0].max(elem[2])],
        ];
        [
            face_map[&local_faces[0]],
            face_map[&local_faces[1]],
            face_map[&local_faces[2]],
        ]
    }

    /// Number of faces
    pub fn n_faces(&self) -> usize {
        self.faces.len()
    }

    /// Number of elements
    pub fn n_elements(&self) -> usize {
        self.elements.len()
    }

    /// Midpoint of a face
    pub fn face_midpoint(&self, face_idx: usize) -> [f64; 2] {
        let f = &self.faces[face_idx];
        let v0 = &self.vertices[f[0]];
        let v1 = &self.vertices[f[1]];
        [(v0[0] + v1[0]) * 0.5, (v0[1] + v1[1]) * 0.5]
    }

    /// Check if a face index is a boundary face
    pub fn is_boundary_face(&self, face_idx: usize) -> bool {
        self.boundary_faces.contains(&face_idx)
    }
}

/// HDG degree-of-freedom layout
///
/// For degree k = 1: one DOF per face (the trace value at the face midpoint).
/// For higher degree: multiple DOFs per face, but we implement k = 1 here.
#[derive(Debug, Clone)]
pub struct HdgDof {
    /// Total number of skeleton (trace) DOFs
    pub n_trace_dofs: usize,
    /// Total number of volume DOFs per element (degree+1 per dimension, for degree k: (k+1)(k+2)/2)
    pub n_volume_dofs_per_element: usize,
    /// Mapping from face index to trace DOF index
    pub face_to_dof: Vec<usize>,
}

impl HdgDof {
    /// Create DOF layout for given mesh and polynomial degree
    pub fn new(mesh: &HdgMesh, _degree: usize) -> Result<Self, IntegrateError> {
        // For degree 1: one trace DOF per face (midpoint value)
        // For degree k: (k+1) DOFs per face, but we simplify to 1 per face for k=1
        let n_faces = mesh.n_faces();
        let face_to_dof: Vec<usize> = (0..n_faces).collect();
        // Volume DOFs: for degree k triangles, (k+1)(k+2)/2
        // For k=1: 3 volume DOFs per element
        let n_volume_dofs_per_element = 3;

        Ok(HdgDof {
            n_trace_dofs: n_faces,
            n_volume_dofs_per_element,
            face_to_dof,
        })
    }
}

/// Gaussian quadrature points and weights on the reference triangle \[0,0\],\[1,0\],\[0,1\]
///
/// 3-point rule, exact for degree ≤ 2.
pub fn triangle_gauss_quadrature_3pt() -> (Vec<[f64; 2]>, Vec<f64>) {
    let pts = vec![
        [1.0 / 6.0, 1.0 / 6.0],
        [2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0],
    ];
    let weights = vec![1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0];
    (pts, weights)
}

/// Map from reference triangle to physical triangle
///
/// Reference: (xi, eta) in {(xi,eta) | xi>=0, eta>=0, xi+eta<=1}
/// Physical: x = v0 + J * [xi; eta] where J = [v1-v0 | v2-v0]
pub fn ref_to_physical(xi: f64, eta: f64, v: &[[f64; 2]; 3]) -> [f64; 2] {
    let x = v[0][0] + (v[1][0] - v[0][0]) * xi + (v[2][0] - v[0][0]) * eta;
    let y = v[0][1] + (v[1][1] - v[0][1]) * xi + (v[2][1] - v[0][1]) * eta;
    [x, y]
}

/// Jacobian determinant (twice area of triangle)
pub fn jacobian_det(v: &[[f64; 2]; 3]) -> f64 {
    let j11 = v[1][0] - v[0][0];
    let j12 = v[2][0] - v[0][0];
    let j21 = v[1][1] - v[0][1];
    let j22 = v[2][1] - v[0][1];
    (j11 * j22 - j12 * j21).abs()
}

/// Inverse Jacobian (2x2 matrix) for gradient transformations
/// Returns [[J^{-1}_{11}, J^{-1}_{12}], [J^{-1}_{21}, J^{-1}_{22}]]
pub fn jacobian_inv(v: &[[f64; 2]; 3]) -> [[f64; 2]; 2] {
    let j11 = v[1][0] - v[0][0];
    let j12 = v[2][0] - v[0][0];
    let j21 = v[1][1] - v[0][1];
    let j22 = v[2][1] - v[0][1];
    let det = j11 * j22 - j12 * j21;
    [[j22 / det, -j12 / det], [-j21 / det, j11 / det]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_construction_two_triangles() {
        // Unit square split into two triangles
        let vertices = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let elements = vec![[0, 1, 2], [0, 2, 3]];
        let mesh = HdgMesh::new(vertices, elements);

        assert_eq!(mesh.n_elements(), 2);
        // 5 edges: 4 outer + 1 diagonal
        assert_eq!(mesh.n_faces(), 5);
        // 4 boundary edges
        assert_eq!(mesh.boundary_faces.len(), 4);
    }

    #[test]
    fn test_mesh_element_faces() {
        let vertices = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let elements = vec![[0, 1, 2], [0, 2, 3]];
        let mesh = HdgMesh::new(vertices, elements);

        let faces0 = mesh.element_faces(0);
        let faces1 = mesh.element_faces(1);
        // Each element has 3 faces
        assert_eq!(faces0.len(), 3);
        assert_eq!(faces1.len(), 3);
        // The two elements share one face (the diagonal 0-2)
        let shared: Vec<usize> = faces0
            .iter()
            .filter(|f| faces1.contains(f))
            .copied()
            .collect();
        assert_eq!(shared.len(), 1);
    }

    #[test]
    fn test_default_config() {
        let cfg = HdgConfig::default();
        assert_eq!(cfg.polynomial_degree, 1);
        assert!((cfg.tau_stabilization - 1.0).abs() < 1e-15);
        assert!((cfg.tol - 1e-10).abs() < 1e-20);
    }

    #[test]
    fn test_gauss_quadrature_3pt_weight_sum() {
        let (_, weights) = triangle_gauss_quadrature_3pt();
        let sum: f64 = weights.iter().sum();
        // Weights sum to 1/2 (area of reference triangle)
        assert!((sum - 0.5).abs() < 1e-14);
    }

    #[test]
    fn test_jacobian_det_unit_triangle() {
        let v = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let det = jacobian_det(&v);
        // det should be 1.0 (2 * area = 2 * 0.5 = 1)
        assert!((det - 1.0).abs() < 1e-14);
    }
}
