//! 3D bounding box IoU and detection metrics.
//!
//! Provides:
//! - Axis-aligned 3D IoU (fast path)
//! - Bird's-Eye View (BEV) rotated rectangle IoU via Sutherland-Hodgman clipping
//! - Full 3D IoU combining BEV polygon overlap with Z-axis extent
//! - Average Precision computation for 3D object detection
//! - Rotated 2D bounding box IoU and NMS

pub mod iou_3d;
pub mod rotated_iou;

pub use iou_3d::{compute_ap_3d, BBox3D, Detection3dMetrics};
pub use rotated_iou::{
    polygon_area, rotated_bbox_corners, rotated_iou, rotated_iou_matrix, rotated_nms, RotatedBBox,
};
