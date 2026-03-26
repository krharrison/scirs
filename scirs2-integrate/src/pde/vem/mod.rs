//! Virtual Element Method (VEM) for polygonal meshes
//!
//! The Virtual Element Method (VEM) generalizes the finite element method to
//! arbitrary polygonal (and polyhedral in 3D) meshes. Unlike FEM, VEM basis
//! functions are **not** explicitly constructed — only their polynomial projections
//! are needed, keeping computations purely based on degrees of freedom on the
//! boundary of each element.
//!
//! ## Problem formulation
//!
//! Solve the Poisson equation:
//!   −∇²u = f   in Ω
//!       u = g   on ∂Ω
//!
//! ## Degree 1 VEM
//!
//! For polynomial degree k = 1:
//! * DOFs: one value per mesh vertex
//! * Virtual basis functions exist on each element but are **not** computed
//! * Energy projection Pi^∇_1: L² best-approximation of gradient onto P₁
//! * Stability term: compensates for kernel of Pi^∇ (rigid motions)
//!
//! ## Key properties
//!
//! * Works on meshes with degenerate or non-convex elements
//! * Maintains optimal convergence rates
//! * Extends naturally to higher degrees and 3D
//!
//! ## References
//!
//! * Beirão da Veiga, L., Brezzi, F., Cangiani, A., Manzini, G., Marini, L. D., & Russo, A. (2013).
//!   Basic principles of virtual element methods. *Math. Models Methods Appl. Sci.*, 23(1), 199-214.
//! * Beirão da Veiga, L., Brezzi, F., Marini, L. D., & Russo, A. (2014). The hitchhiker's guide
//!   to the virtual element method. *Math. Models Methods Appl. Sci.*, 24(8), 1541-1573.

pub mod assembly;
pub mod basis;

pub use assembly::{solve_vem, VemSolution};
pub use basis::{polygon_centroid_and_diameter, scaled_monomials_values};

/// Configuration for the VEM solver
#[derive(Debug, Clone)]
pub struct VemConfig {
    /// Polynomial degree of the approximation (default: 1)
    ///
    /// Currently only degree 1 is fully implemented, providing O(h) convergence
    /// in the H1-seminorm and O(h²) in the L2-norm.
    pub polynomial_degree: usize,
    /// Stabilization scaling factor α (default: 1.0)
    ///
    /// The stability term is scaled by `stabilization * (element diameter)^2`.
    /// Must be positive; see Beirão da Veiga et al. (2014) for guidance.
    pub stabilization: f64,
}

impl Default for VemConfig {
    fn default() -> Self {
        VemConfig {
            polynomial_degree: 1,
            stabilization: 1.0,
        }
    }
}

/// A polygonal mesh supporting arbitrary polygon types
///
/// Each element is defined by an ordered list of vertex indices forming a simple polygon.
/// Vertices should be ordered counterclockwise for correct outward normal computation.
#[derive(Debug, Clone)]
pub struct PolygonalMesh {
    /// Mesh vertices as `[x, y]` coordinates
    pub vertices: Vec<[f64; 2]>,
    /// Elements: each element is an ordered list of vertex indices (counterclockwise)
    pub elements: Vec<Vec<usize>>,
    /// Boundary vertex indices (vertices on ∂Ω)
    pub boundary_vertices: Vec<usize>,
}

impl PolygonalMesh {
    /// Create a new polygonal mesh
    ///
    /// # Arguments
    ///
    /// * `vertices` - Vertex coordinates
    /// * `elements` - Element connectivity (CCW vertex ordering)
    ///
    /// Boundary vertices are automatically detected as vertices belonging to
    /// edges that appear in exactly one element.
    pub fn new(vertices: Vec<[f64; 2]>, elements: Vec<Vec<usize>>) -> Self {
        use std::collections::HashMap;

        // Count edge occurrences to find boundary edges
        let mut edge_count: HashMap<[usize; 2], usize> = HashMap::new();
        for elem in &elements {
            let n = elem.len();
            for i in 0..n {
                let a = elem[i];
                let b = elem[(i + 1) % n];
                let edge = [a.min(b), a.max(b)];
                *edge_count.entry(edge).or_insert(0) += 1;
            }
        }

        // Boundary vertices: appear on an edge that occurs exactly once
        let mut boundary_set = std::collections::HashSet::new();
        for (edge, &count) in &edge_count {
            if count == 1 {
                boundary_set.insert(edge[0]);
                boundary_set.insert(edge[1]);
            }
        }
        let mut boundary_vertices: Vec<usize> = boundary_set.into_iter().collect();
        boundary_vertices.sort();

        PolygonalMesh {
            vertices,
            elements,
            boundary_vertices,
        }
    }

    /// Number of vertices
    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Number of elements
    pub fn n_elements(&self) -> usize {
        self.elements.len()
    }

    /// Check if a vertex is a boundary vertex
    pub fn is_boundary_vertex(&self, v_idx: usize) -> bool {
        self.boundary_vertices.contains(&v_idx)
    }

    /// Get vertices of a specific element as coordinate slices
    pub fn element_vertices(&self, elem_idx: usize) -> Vec<[f64; 2]> {
        self.elements[elem_idx]
            .iter()
            .map(|&v| self.vertices[v])
            .collect()
    }
}

/// VEM degree-of-freedom layout for degree 1
///
/// For degree k=1: one DOF per vertex (the nodal value).
#[derive(Debug, Clone)]
pub struct VemDof {
    /// Total number of DOFs (= number of vertices for degree 1)
    pub n_dofs: usize,
    /// DOF index for each vertex
    pub vertex_to_dof: Vec<usize>,
}

impl VemDof {
    /// Create DOF layout for given mesh
    pub fn new(mesh: &PolygonalMesh) -> Self {
        let n = mesh.n_vertices();
        VemDof {
            n_dofs: n,
            vertex_to_dof: (0..n).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polygonal_mesh_square() {
        // Unit square as 4 triangles with center vertex
        let vertices = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5], // center
        ];
        let elements = vec![vec![0, 1, 4], vec![1, 2, 4], vec![2, 3, 4], vec![3, 0, 4]];
        let mesh = PolygonalMesh::new(vertices, elements);
        assert_eq!(mesh.n_vertices(), 5);
        assert_eq!(mesh.n_elements(), 4);
        // Corner vertices should be boundary
        assert!(mesh.is_boundary_vertex(0));
        assert!(mesh.is_boundary_vertex(1));
        // Center vertex should be interior
        assert!(!mesh.is_boundary_vertex(4));
    }

    #[test]
    fn test_vem_config_default() {
        let cfg = VemConfig::default();
        assert_eq!(cfg.polynomial_degree, 1);
        assert!((cfg.stabilization - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_polygon_centroid_square() {
        use crate::pde::vem::basis::polygon_centroid_and_diameter;
        let verts = [[0.0_f64, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let (centroid, diameter) = polygon_centroid_and_diameter(&verts);
        assert!((centroid[0] - 0.5).abs() < 1e-12, "cx={}", centroid[0]);
        assert!((centroid[1] - 0.5).abs() < 1e-12, "cy={}", centroid[1]);
        // Diameter of unit square = sqrt(2)
        let expected_diam = 2.0_f64.sqrt();
        assert!(
            (diameter - expected_diam).abs() < 1e-12,
            "diam={}",
            diameter
        );
    }
}
