//! Types for advanced Discontinuous Galerkin methods
//!
//! Provides configuration, element descriptions, and solution containers
//! for curved-element, high-order, and entropy-stable DG schemes.

/// Configuration for advanced DG solver
#[derive(Debug, Clone)]
pub struct DgAdvancedConfig {
    /// Polynomial order for DG basis
    pub poly_order: usize,
    /// Number of quadrature points per element/dimension
    pub n_quad_points: usize,
    /// Whether to use curved (isoparametric) elements
    pub curved: bool,
    /// Whether to use entropy-stable numerical flux
    pub entropy_stable: bool,
}

impl Default for DgAdvancedConfig {
    fn default() -> Self {
        Self {
            poly_order: 3,
            n_quad_points: 5,
            curved: false,
            entropy_stable: false,
        }
    }
}

/// Type of geometric mapping from reference to physical element
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
pub enum GeometricMap {
    /// Affine (linear) map — constant Jacobian
    #[default]
    Linear,
    /// Isoparametric map — polynomial mapping using element nodes
    Isoparametric,
    /// Gordon-Hall blended transfinite interpolation
    BlendedTransfinite,
}

/// A curved element defined by its physical vertices and geometric map type
#[derive(Debug, Clone)]
pub struct CurvedElement {
    /// Physical coordinates of the element nodes, shape: \[n_nodes\]\[2\]
    pub vertices: Vec<[f64; 2]>,
    /// Geometric map type
    pub mapping: GeometricMap,
}

impl CurvedElement {
    /// Create a new curved element with given vertices and mapping
    pub fn new(vertices: Vec<[f64; 2]>, mapping: GeometricMap) -> Self {
        Self { vertices, mapping }
    }

    /// Create a linear triangular element from three corner vertices
    pub fn linear_triangle(v0: [f64; 2], v1: [f64; 2], v2: [f64; 2]) -> Self {
        Self {
            vertices: vec![v0, v1, v2],
            mapping: GeometricMap::Linear,
        }
    }
}

/// DG solution storage: polynomial coefficients on each element
#[derive(Debug, Clone)]
pub struct DgSolution {
    /// Coefficients: `coefficients[e][k]` is DOF k on element e
    pub coefficients: Vec<Vec<f64>>,
    /// Number of elements
    pub n_elements: usize,
    /// Polynomial order
    pub order: usize,
}

impl DgSolution {
    /// Create a zero solution
    pub fn zeros(n_elements: usize, order: usize) -> Self {
        let n_dofs = order + 1;
        Self {
            coefficients: vec![vec![0.0; n_dofs]; n_elements],
            n_elements,
            order,
        }
    }

    /// Access coefficients for element `e`
    pub fn element(&self, e: usize) -> &[f64] {
        &self.coefficients[e]
    }

    /// Mutable access to coefficients for element `e`
    pub fn element_mut(&mut self, e: usize) -> &mut Vec<f64> {
        &mut self.coefficients[e]
    }
}

/// Numerical flux type used at element interfaces
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
pub enum FluxType {
    /// Rusanov (local Lax-Friedrichs) flux
    #[default]
    Rusanov,
    /// Roe approximate Riemann solver flux
    Roe,
    /// HLLC (Harten-Lax-van Leer Contact) flux
    Hllc,
    /// Entropy-stable (Tadmor symmetrized) flux
    EntropyStable,
}

/// Configuration for entropy-stable DG schemes
#[derive(Debug, Clone)]
pub struct EntropyStableConfig {
    /// Flux type at element interfaces
    pub flux: FluxType,
    /// Interior penalty parameter τ ≥ 0
    pub tau: f64,
}

impl Default for EntropyStableConfig {
    fn default() -> Self {
        Self {
            flux: FluxType::EntropyStable,
            tau: 1.0,
        }
    }
}
