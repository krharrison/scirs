//! Neural 3D reconstruction and depth estimation
//!
//! This module provides:
//! - **NeRF** (Neural Radiance Fields): volume rendering, positional encoding,
//!   hierarchical sampling, and ray generation from camera parameters.
//! - **Depth estimation**: simplified MiDaS-style relative depth, depth completion,
//!   depth-from-disparity, depth-to-point-cloud, depth colorization, and depth edge detection.

pub mod depth;
pub mod nerf;

pub use depth::{
    depth_colorize, depth_completion, depth_edge_detection, depth_from_disparity,
    depth_to_point_cloud, DepthCompletionConfig, DepthEstimator, DepthEstimatorConfig,
    RelativeDepthMap,
};
pub use nerf::{
    generate_rays, volume_render, HierarchicalSampler, NeRFConfig, NeRFModel, PositionalEncoding,
    Ray, RayBundle, VolumeRenderResult,
};
