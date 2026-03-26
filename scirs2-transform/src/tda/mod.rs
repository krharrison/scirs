//! Topological Data Analysis (TDA) module
//!
//! This module provides advanced TDA algorithms:
//!
//! - [`alpha_complex`]: Alpha complex filtration via Bowyer-Watson Delaunay triangulation
//! - [`cubical_complex`]: Cubical complex persistence for image/grid data
//! - [`zigzag`]: Zigzag persistence for sequences of simplicial complexes
//! - `gromov_wasserstein`: Gromov-Wasserstein distance and multi-marginal OT
//!
//! ## References
//!
//! - Edelsbrunner & Harer (2010). Computational Topology.
//! - Carlsson (2009). Topology and Data.
//! - Mémoli (2011). Gromov-Wasserstein Distances.

pub mod alpha_complex;
pub mod cubical_complex;
pub mod gromov_wasserstein;
pub mod zigzag;

// Re-export key public types
pub use alpha_complex::{AlphaComplex, AlphaConfig, Simplex};
pub use cubical_complex::{CubicalCell, CubicalComplex, CubicalConfig};
pub use gromov_wasserstein::{
    gromov_wasserstein, multi_marginal_ot, sinkhorn_log_stabilized, GwConfig, GwResult,
};
pub use zigzag::{compute_zigzag, ZigzagDirection, ZigzagPersistence, ZigzagStep};

// Re-export VietorisRips from the tda_vr module so doc tests can use
// `scirs2_transform::tda::VietorisRips`
pub use crate::tda_vr::{PersistenceDiagram, PersistencePoint, VietorisRips};
