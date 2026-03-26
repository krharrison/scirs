//! 3D Object Detection
//!
//! Provides PointPillar-style 3D object detection from point cloud data.
//! The pipeline consists of:
//! 1. **Voxelization** — discretize the point cloud into pillars/voxels
//! 2. **Backbone** — 2D CNN over bird's-eye-view features
//! 3. **Detection head** — anchor-based classification and bounding-box regression

pub mod backbone;
pub mod detection_head;
pub mod frustum;
pub mod types;
pub mod voxelization;

pub use backbone::{FeaturePyramidNeck, SimpleBEVBackbone};
pub use detection_head::{
    decode_predictions, focal_loss, iou_3d_bev, nms_3d, smooth_l1_loss, AnchorGenerator,
    DetectionHead,
};
pub use types::{BoundingBox3D, Detection3D, DetectionResult, PointPillarConfig, VoxelConfig};
pub use voxelization::{scatter_to_bev, PillarFeatureNet, Voxel, VoxelGrid};
