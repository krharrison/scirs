//! PointNet++ module for hierarchical 3D point cloud learning.
//!
//! Provides Farthest Point Sampling, Ball Query, K-Nearest Neighbors,
//! Set Abstraction layers, Feature Propagation layers, and the full
//! PointNet++ backbone for 3D object detection.
//!
//! # References
//! - Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point Sets
//!   in a Metric Space", NeurIPS 2017.

pub mod backbone;
pub mod feature_propagation;
pub mod sampling;
pub mod set_abstraction;

pub use backbone::{PointNetPPBackbone, PointNetPPConfig};
pub use feature_propagation::{FPConfig, FeaturePropagation};
pub use sampling::{ball_query, farthest_point_sampling, knn_query};
pub use set_abstraction::{SAConfig, SetAbstraction};
