//! E(n)-Equivariant Graph Neural Networks and SE(3)-Transformer
//!
//! This module implements equivariant GNN architectures that respect geometric
//! symmetries (rotations, reflections, translations) in their design:
//!
//! - **EGNN** (Satorras et al. 2021): E(n)-equivariant message passing
//! - **SE(3)-Transformer** (Fuchs et al. 2020): Attention with SE(3) equivariance
//!   using spherical harmonics and tensor products
//!
//! ## Features
//!
//! - Clebsch-Gordan coefficients for type-l tensor products
//! - Real spherical harmonics up to l=2
//! - Equivariance tests (rotation + translation invariance / equivariance)
//! - Energy prediction for molecular property tasks

pub mod cg_coefficients;
pub mod egnn;
pub mod se3_transformer;

// Re-export main types
pub use cg_coefficients::{clebsch_gordan, tensor_product, CgTable};
pub use egnn::{Activation, Egnn, EgnnConfig, EgnnLayer};
pub use se3_transformer::{
    EquivariantFeatures, Se3Config, Se3Layer, Se3Transformer, SphericalHarmonics,
};

/// Container holding type-0 (scalar) and type-1 (vector) equivariant features.
///
/// In the irreducible representation decomposition of E(3):
/// - Type-0 (l=0): scalars invariant under rotation
/// - Type-1 (l=1): 3-vectors that rotate with the group element
#[derive(Debug, Clone)]
pub struct EquivariantNodeFeatures {
    /// Scalar (type-0) features: shape [n_nodes × n_scalars]
    pub scalars: Vec<f64>,
    /// Vector (type-1) features: 3D vectors per node
    pub vectors: Vec<[f64; 3]>,
    /// Number of nodes
    pub n_nodes: usize,
}

impl EquivariantNodeFeatures {
    /// Create a new container of equivariant features.
    pub fn new(n_nodes: usize, n_scalars_per_node: usize) -> Self {
        Self {
            scalars: vec![0.0; n_nodes * n_scalars_per_node],
            vectors: vec![[0.0; 3]; n_nodes],
            n_nodes,
        }
    }
}
