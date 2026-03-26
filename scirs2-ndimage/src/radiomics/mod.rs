//! Radiomics feature extraction module
//!
//! Provides 3D shape-based radiomics features and first-order intensity statistics
//! used in quantitative image analysis and medical imaging.

pub mod intensity_statistics;
pub mod region_adjacency;
pub mod shape_features;

pub use intensity_statistics::{compute_intensity_features, IntensityFeatures};
pub use region_adjacency::{
    build_rag_2d, build_rag_3d, merge_small_regions as rag_merge_small_regions,
    rag_to_adjacency_matrix, RagConfig, RagEdge, RegionAdjacencyGraph,
};
pub use shape_features::{compute_shape_features, ShapeFeatures};
