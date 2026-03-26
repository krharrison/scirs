//! Types for 3D object detection.

use std::f64::consts::PI;

/// Axis-aligned or yaw-rotated 3D bounding box.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundingBox3D {
    /// Centre of the box (x, y, z).
    pub center: [f64; 3],
    /// Half-extent-free size: \[length, width, height\].
    pub size: [f64; 3],
    /// Rotation around the Z-axis in radians.
    pub yaw: f64,
}

impl BoundingBox3D {
    /// Create a new bounding box.
    pub fn new(center: [f64; 3], size: [f64; 3], yaw: f64) -> Self {
        Self { center, size, yaw }
    }

    /// Volume of the box.
    #[inline]
    pub fn volume(&self) -> f64 {
        self.size[0] * self.size[1] * self.size[2]
    }

    /// The eight corners of the box in world coordinates.
    ///
    /// Order: bottom-face corners first (counter-clockwise), then top-face.
    pub fn corners(&self) -> [[f64; 3]; 8] {
        let (cx, cy, cz) = (self.center[0], self.center[1], self.center[2]);
        let (hl, hw, hh) = (self.size[0] / 2.0, self.size[1] / 2.0, self.size[2] / 2.0);
        let cos_y = self.yaw.cos();
        let sin_y = self.yaw.sin();

        // Local corners before rotation (length along X, width along Y).
        let local = [[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]];

        let mut out = [[0.0f64; 3]; 8];
        for (i, &[lx, ly]) in local.iter().enumerate() {
            let rx = lx * cos_y - ly * sin_y + cx;
            let ry = lx * sin_y + ly * cos_y + cy;
            out[i] = [rx, ry, cz - hh]; // bottom
            out[i + 4] = [rx, ry, cz + hh]; // top
        }
        out
    }

    /// Check whether a point (x, y, z) lies inside the box.
    pub fn contains_point(&self, point: &[f64; 3]) -> bool {
        let cos_y = self.yaw.cos();
        let sin_y = self.yaw.sin();

        // Translate to box-local frame.
        let dx = point[0] - self.center[0];
        let dy = point[1] - self.center[1];
        let dz = point[2] - self.center[2];

        // Inverse rotation.
        let lx = dx * cos_y + dy * sin_y;
        let ly = -dx * sin_y + dy * cos_y;

        lx.abs() <= self.size[0] / 2.0
            && ly.abs() <= self.size[1] / 2.0
            && dz.abs() <= self.size[2] / 2.0
    }
}

/// A single 3D detection result.
#[derive(Debug, Clone)]
pub struct Detection3D {
    /// 3D bounding box.
    pub bbox: BoundingBox3D,
    /// Numeric class identifier.
    pub class_id: usize,
    /// Detection confidence in \[0, 1\].
    pub confidence: f64,
    /// Optional human-readable class name.
    pub class_name: Option<String>,
}

/// Aggregated result of a detection pass.
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Individual detections after NMS.
    pub detections: Vec<Detection3D>,
    /// Number of input points that were processed.
    pub num_points_processed: usize,
}

// ---------------------------------------------------------------------------
// Configs
// ---------------------------------------------------------------------------

/// Configuration for the PointPillar pipeline.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PointPillarConfig {
    /// X range \[min, max\] in metres.
    pub x_range: [f64; 2],
    /// Y range \[min, max\] in metres.
    pub y_range: [f64; 2],
    /// Z range \[min, max\] in metres.
    pub z_range: [f64; 2],
    /// Voxel size \[dx, dy, dz\] in metres.
    pub voxel_size: [f64; 3],
    /// Maximum number of points kept per pillar.
    pub max_points_per_voxel: usize,
    /// Maximum number of non-empty pillars.
    pub max_pillars: usize,
    /// Feature dimension after pillar encoding.
    pub n_features: usize,
}

impl Default for PointPillarConfig {
    fn default() -> Self {
        Self {
            x_range: [-50.0, 50.0],
            y_range: [-50.0, 50.0],
            z_range: [-5.0, 3.0],
            voxel_size: [0.2, 0.2, 8.0],
            max_points_per_voxel: 32,
            max_pillars: 12000,
            n_features: 64,
        }
    }
}

/// Configuration for generic voxelization.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct VoxelConfig {
    /// X range \[min, max\] in metres.
    pub x_range: [f64; 2],
    /// Y range \[min, max\] in metres.
    pub y_range: [f64; 2],
    /// Z range \[min, max\] in metres.
    pub z_range: [f64; 2],
    /// Voxel size \[dx, dy, dz\] in metres.
    pub voxel_size: [f64; 3],
    /// Maximum number of points kept per voxel.
    pub max_points_per_voxel: usize,
    /// Maximum number of non-empty voxels.
    pub max_voxels: usize,
}

impl Default for VoxelConfig {
    fn default() -> Self {
        Self {
            x_range: [-50.0, 50.0],
            y_range: [-50.0, 50.0],
            z_range: [-5.0, 3.0],
            voxel_size: [0.2, 0.2, 0.2],
            max_points_per_voxel: 32,
            max_voxels: 20000,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bbox_volume() {
        let b = BoundingBox3D::new([0.0; 3], [4.0, 2.0, 1.0], 0.0);
        assert!((b.volume() - 8.0).abs() < 1e-12);
    }

    #[test]
    fn bbox_corners_no_yaw() {
        let b = BoundingBox3D::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], 0.0);
        let c = b.corners();
        // Bottom corners should be at z = -1, top at z = 1.
        for i in 0..4 {
            assert!((c[i][2] - (-1.0)).abs() < 1e-12);
            assert!((c[i + 4][2] - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn bbox_contains_center() {
        let b = BoundingBox3D::new([1.0, 2.0, 3.0], [4.0, 4.0, 4.0], 0.5);
        assert!(b.contains_point(&[1.0, 2.0, 3.0]));
    }

    #[test]
    fn bbox_does_not_contain_far_point() {
        let b = BoundingBox3D::new([0.0; 3], [1.0, 1.0, 1.0], 0.0);
        assert!(!b.contains_point(&[10.0, 0.0, 0.0]));
    }

    #[test]
    fn bbox_corners_with_yaw() {
        let b = BoundingBox3D::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], PI / 4.0);
        let c = b.corners();
        // After 45-degree rotation the first bottom corner should be along negative y-axis.
        let expected_x = -(PI / 4.0).cos() + (PI / 4.0).sin();
        let expected_y = -(PI / 4.0).sin() - (PI / 4.0).cos();
        assert!((c[0][0] - expected_x).abs() < 1e-10);
        assert!((c[0][1] - expected_y).abs() < 1e-10);
    }

    #[test]
    fn default_configs() {
        let _pp = PointPillarConfig::default();
        let _vc = VoxelConfig::default();
    }
}
