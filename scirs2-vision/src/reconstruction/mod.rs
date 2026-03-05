//! 3D Reconstruction algorithms
//!
//! This module provides:
//! - Structure from Motion (SfM): fundamental/essential matrix estimation,
//!   triangulation, bundle adjustment, incremental SfM pipeline.
//! - Dense reconstruction: Semi-Global Matching, Multi-View Stereo, TSDF
//!   depth fusion, Poisson surface reconstruction.
//! - Reconstruction utilities: DLT/midpoint triangulation, disparity-to-depth,
//!   point cloud back-projection.

pub mod dense;
pub mod sfm;
pub mod utils;

// Re-export the most commonly used types
pub use dense::{
    DepthFusion, MVSConfig, MVSView, OrientedPoint, PoissonSurface, SGMConfig, MVS, SGM,
};
pub use sfm::{
    triangulate_dlt_single, BundleAdjustment, Camera, EssentialMatrix, FundamentalMatrix,
    IntrinsicMatrix, Observation, PointCloud, SFMPipeline, SFMResult, Track, Triangulation,
};
pub use utils::{
    depth_from_disparity, point_cloud_from_depth, triangulate_dlt, triangulate_midpoint,
};
