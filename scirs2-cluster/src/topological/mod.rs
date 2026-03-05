//! Topological Data Analysis (TDA) based clustering.
//!
//! This module provides advanced clustering algorithms rooted in algebraic
//! topology and persistent homology.  Unlike conventional clustering methods
//! that operate purely on metric structure, TDA-based approaches expose the
//! *shape* of data — capturing loops, voids, and connected components that
//! survive across multiple scales.
//!
//! # Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`cover_tree`] | Cover Tree for O(c^6 log n) k-NN search |
//! | [`filtrations`] | Lens/filter functions (eccentricity, density, PCA, Laplacian, geodesic) |
//! | [`mapper`] | Mapper algorithm (Singh–Mémoli–Carlsson 2007) |
//! | [`pers_homology_cluster`] | ToMATo and persistence-based flat clustering |
//!
//! # Quick Start
//!
//! ## Mapper
//!
//! ```rust
//! use scirs2_core::ndarray::Array2;
//! use scirs2_cluster::topological::mapper::{Mapper, MapperConfig};
//! use scirs2_cluster::topological::filtrations::EccentricityFiltration;
//!
//! let data = Array2::from_shape_vec((8, 2), vec![
//!     0.0, 0.0,  0.2, 0.1,  0.1, 0.2,  0.15, 0.05,
//!     5.0, 5.0,  5.2, 4.9,  4.9, 5.1,  5.1, 5.0,
//! ]).expect("operation should succeed");
//!
//! let config = MapperConfig {
//!     n_intervals: 5,
//!     overlap: 0.4,
//!     min_cluster_size: 1,
//! };
//! let filt = EccentricityFiltration::default();
//! let graph = Mapper::fit(data.view(), &filt, &config).expect("operation should succeed");
//! println!("{} nodes, {} edges", graph.n_nodes(), graph.n_edges());
//! ```
//!
//! ## ToMATo
//!
//! ```rust
//! use scirs2_core::ndarray::Array2;
//! use scirs2_cluster::topological::pers_homology_cluster::{tomato, TomaToConfig};
//!
//! let data = Array2::from_shape_vec((6, 2), vec![
//!     0.0, 0.0,  0.1, 0.1,  -0.1, 0.0,
//!     5.0, 5.0,  5.1, 4.9,   4.9, 5.0,
//! ]).expect("operation should succeed");
//!
//! let config = TomaToConfig { k_neighbors: 2, auto_threshold: true, ..Default::default() };
//! let result = tomato(data.view(), &config).expect("operation should succeed");
//! println!("Found {} clusters", result.n_clusters);
//! ```
//!
//! ## Cover Tree
//!
//! ```rust
//! use scirs2_core::ndarray::Array2;
//! use scirs2_cluster::topological::cover_tree::{CoverTree, CoverTreeConfig, L2Distance};
//!
//! let data = Array2::from_shape_vec((5, 2), vec![
//!     0.0, 0.0,  1.0, 0.0,  0.5, 0.5,  5.0, 5.0,  5.1, 5.0,
//! ]).expect("operation should succeed");
//!
//! let tree = CoverTree::build(data.view(), CoverTreeConfig::default(), &L2Distance).expect("operation should succeed");
//! let neighbours = tree.knn_with_metric(data.row(0), 2, &L2Distance).expect("operation should succeed");
//! assert_eq!(neighbours.len(), 2);
//! ```

pub mod cover_tree;
pub mod filtrations;
pub mod mapper;
pub mod pers_homology_cluster;

// ── Convenience re-exports ────────────────────────────────────────────────────

pub use cover_tree::{CoverTree, CoverTreeConfig, CoverTreeMetric, L2Distance};

pub use filtrations::{
    DensityFiltration, EccentricityFiltration, Filtration, GeodesicDistanceFiltration,
    LaplacianEigenvectorFiltration, PcaFiltration,
};

pub use mapper::{
    uniform_cover, Mapper, MapperConfig, MapperGraph, MapperNode,
};

pub use pers_homology_cluster::{
    cluster_stats, density_persistence, flat_clustering_from_persistence, gap_threshold, tomato,
    tomato_n_clusters, ClusterStats, PersistenceBar, TomaToConfig, TomaToResult,
};
