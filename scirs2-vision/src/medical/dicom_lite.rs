//! Lightweight DICOM-compatible medical image metadata and volumetric image management.
//!
//! This module provides a simplified DICOM-like data model suitable for:
//! - Storing per-slice metadata (patient/study/series IDs, pixel spacing, etc.)
//! - Managing 3D volumetric images as stacks of 2D slices
//! - Computing intensity projections (MIP, MinIP, AvgIP) along any axis

use crate::error::VisionError;
use scirs2_core::ndarray::{Array2, Array3, ArrayView2};
use scirs2_core::Axis;

/// Simplified medical image metadata compatible with DICOM tag semantics.
#[derive(Debug, Clone)]
pub struct MedicalImageMetadata {
    /// Patient identifier (DICOM tag 0010,0020)
    pub patient_id: Option<String>,
    /// Study instance UID (DICOM tag 0020,000D)
    pub study_id: Option<String>,
    /// Series instance UID (DICOM tag 0020,000E)
    pub series_id: Option<String>,
    /// Instance (slice) number within series (DICOM tag 0020,0013)
    pub instance_number: Option<usize>,
    /// Number of pixel rows in the image
    pub rows: usize,
    /// Number of pixel columns in the image
    pub cols: usize,
    /// Physical size of a pixel: (row_spacing, col_spacing) in mm (DICOM tag 0028,0030)
    pub pixel_spacing: Option<(f64, f64)>,
    /// Physical thickness of a slice in mm (DICOM tag 0050,0018)
    pub slice_thickness: Option<f64>,
    /// Display window centre (DICOM tag 0028,1050)
    pub window_center: Option<f64>,
    /// Display window width (DICOM tag 0028,1051)
    pub window_width: Option<f64>,
    /// Imaging modality string, e.g. "CT", "MR", "US" (DICOM tag 0008,0060)
    pub modality: Option<String>,
    /// Number of bits allocated per pixel (DICOM tag 0028,0100)
    pub bits_allocated: u8,
    /// Slope for converting stored pixel values to real-world units (DICOM tag 0028,1053)
    pub rescale_slope: f64,
    /// Intercept for converting stored pixel values to real-world units (DICOM tag 0028,1052)
    pub rescale_intercept: f64,
}

impl Default for MedicalImageMetadata {
    fn default() -> Self {
        Self {
            patient_id: None,
            study_id: None,
            series_id: None,
            instance_number: None,
            rows: 0,
            cols: 0,
            pixel_spacing: None,
            slice_thickness: None,
            window_center: None,
            window_width: None,
            modality: None,
            bits_allocated: 16,
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
        }
    }
}

impl MedicalImageMetadata {
    /// Construct minimal metadata for an image of the given dimensions.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            ..Default::default()
        }
    }

    /// Convert a stored pixel value to a real-world value (e.g. Hounsfield Units for CT).
    ///
    /// Applies the linear transformation:
    /// `real = pixel * rescale_slope + rescale_intercept`
    #[inline]
    pub fn to_real_value(&self, pixel_value: i32) -> f64 {
        pixel_value as f64 * self.rescale_slope + self.rescale_intercept
    }

    /// Describe the modality as a human-readable string, defaulting to `"Unknown"`.
    pub fn modality_str(&self) -> &str {
        self.modality.as_deref().unwrap_or("Unknown")
    }
}

/// A 3-D stack of medical image slices with associated per-slice metadata.
///
/// `data` has shape `(n_slices, rows, cols)` and pixel values in real-world units
/// (e.g. Hounsfield Units for CT, signal amplitude for MR).
#[derive(Debug, Clone)]
pub struct MedicalVolume {
    /// Voxel data with shape `(n_slices, rows, cols)`.
    pub data: Array3<f64>,
    /// Per-slice metadata; may have fewer entries than slices if metadata is partial.
    pub metadata: Vec<MedicalImageMetadata>,
    /// Physical voxel spacing `(z_mm, y_mm, x_mm)`.
    pub voxel_spacing: (f64, f64, f64),
}

impl MedicalVolume {
    /// Create a new `MedicalVolume` from a 3-D array and isotropic-or-anisotropic voxel spacing.
    pub fn new(data: Array3<f64>, voxel_spacing: (f64, f64, f64)) -> Self {
        Self {
            data,
            metadata: Vec::new(),
            voxel_spacing,
        }
    }

    /// Number of slices along the first (z) axis.
    #[inline]
    pub fn n_slices(&self) -> usize {
        self.data.shape()[0]
    }

    /// Volume shape as `(n_slices, rows, cols)`.
    #[inline]
    pub fn shape(&self) -> (usize, usize, usize) {
        let s = self.data.shape();
        (s[0], s[1], s[2])
    }

    /// Return a 2-D view of slice `idx`.
    ///
    /// # Errors
    ///
    /// Returns [`VisionError::InvalidParameter`] when `idx >= n_slices`.
    pub fn slice(&self, idx: usize) -> Result<ArrayView2<f64>, VisionError> {
        let n = self.n_slices();
        if idx >= n {
            return Err(VisionError::InvalidParameter(format!(
                "slice index {idx} out of bounds for volume with {n} slices"
            )));
        }
        Ok(self.data.index_axis(Axis(0), idx))
    }

    /// Maximum-Intensity Projection along `axis` (0=z, 1=y, 2=x).
    ///
    /// # Errors
    ///
    /// Returns [`VisionError::InvalidParameter`] for an axis index ≥ 3 or an empty volume.
    pub fn mip(&self, axis: usize) -> Result<Array2<f64>, VisionError> {
        intensity_projection(&self.data, axis, ProjectionKind::Max)
    }

    /// Minimum-Intensity Projection along `axis`.
    pub fn min_ip(&self, axis: usize) -> Result<Array2<f64>, VisionError> {
        intensity_projection(&self.data, axis, ProjectionKind::Min)
    }

    /// Average-Intensity Projection along `axis`.
    pub fn avg_ip(&self, axis: usize) -> Result<Array2<f64>, VisionError> {
        intensity_projection(&self.data, axis, ProjectionKind::Avg)
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

enum ProjectionKind {
    Max,
    Min,
    Avg,
}

fn intensity_projection(
    data: &Array3<f64>,
    axis: usize,
    kind: ProjectionKind,
) -> Result<Array2<f64>, VisionError> {
    if axis >= 3 {
        return Err(VisionError::InvalidParameter(format!(
            "axis must be 0, 1, or 2; got {axis}"
        )));
    }
    let shape = data.shape();
    if shape[axis] == 0 {
        return Err(VisionError::InvalidParameter(
            "Volume has zero extent along the requested axis".to_string(),
        ));
    }

    // Determine output dimensions
    let (out_rows, out_cols, depth) = match axis {
        0 => (shape[1], shape[2], shape[0]),
        1 => (shape[0], shape[2], shape[1]),
        _ => (shape[0], shape[1], shape[2]),
    };

    let mut out = Array2::<f64>::from_elem(
        (out_rows, out_cols),
        match kind {
            ProjectionKind::Max => f64::NEG_INFINITY,
            ProjectionKind::Min => f64::INFINITY,
            ProjectionKind::Avg => 0.0,
        },
    );

    for d in 0..depth {
        for i in 0..out_rows {
            for j in 0..out_cols {
                let v = match axis {
                    0 => data[[d, i, j]],
                    1 => data[[i, d, j]],
                    _ => data[[i, j, d]],
                };
                let cur = out[[i, j]];
                out[[i, j]] = match kind {
                    ProjectionKind::Max => cur.max(v),
                    ProjectionKind::Min => cur.min(v),
                    ProjectionKind::Avg => cur + v,
                };
            }
        }
    }

    if let ProjectionKind::Avg = kind {
        let n = depth as f64;
        out.mapv_inplace(|v| v / n);
    }

    Ok(out)
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // ── MedicalImageMetadata ─────────────────────────────────────────────────

    #[test]
    fn test_metadata_default() {
        let meta = MedicalImageMetadata::default();
        assert_eq!(meta.rows, 0);
        assert_eq!(meta.cols, 0);
        assert_eq!(meta.rescale_slope, 1.0);
        assert_eq!(meta.rescale_intercept, 0.0);
        assert_eq!(meta.bits_allocated, 16);
        assert!(meta.patient_id.is_none());
    }

    #[test]
    fn test_metadata_new() {
        let meta = MedicalImageMetadata::new(512, 512);
        assert_eq!(meta.rows, 512);
        assert_eq!(meta.cols, 512);
    }

    #[test]
    fn test_to_real_value_identity() {
        let meta = MedicalImageMetadata::default(); // slope=1, intercept=0
        assert!((meta.to_real_value(0) - 0.0).abs() < 1e-10);
        assert!((meta.to_real_value(1000) - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_to_real_value_ct_rescale() {
        // Typical CT: slope=1, intercept=-1024  → pixel 0 ≡ -1024 HU
        let mut meta = MedicalImageMetadata::new(512, 512);
        meta.rescale_slope = 1.0;
        meta.rescale_intercept = -1024.0;
        assert!((meta.to_real_value(0) - (-1024.0)).abs() < 1e-10);
        assert!((meta.to_real_value(1024) - 0.0).abs() < 1e-10);
        assert!((meta.to_real_value(2024) - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_modality_str_default() {
        let meta = MedicalImageMetadata::default();
        assert_eq!(meta.modality_str(), "Unknown");
    }

    #[test]
    fn test_modality_str_ct() {
        let mut meta = MedicalImageMetadata::new(512, 512);
        meta.modality = Some("CT".to_string());
        assert_eq!(meta.modality_str(), "CT");
    }

    // ── MedicalVolume ─────────────────────────────────────────────────────────

    fn make_volume() -> MedicalVolume {
        // 4 slices, 3 rows, 3 cols, values 0..36
        let data = Array3::from_shape_fn((4, 3, 3), |(z, y, x)| (z * 9 + y * 3 + x) as f64);
        MedicalVolume::new(data, (2.0, 0.5, 0.5))
    }

    #[test]
    fn test_volume_shape() {
        let vol = make_volume();
        assert_eq!(vol.shape(), (4, 3, 3));
        assert_eq!(vol.n_slices(), 4);
    }

    #[test]
    fn test_volume_slice_valid() {
        let vol = make_volume();
        let s = vol.slice(0).expect("Should succeed for index 0");
        assert_eq!(s.shape(), &[3, 3]);
        assert!((s[[0, 0]] - 0.0).abs() < 1e-10);
        assert!((s[[1, 1]] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_volume_slice_out_of_bounds() {
        let vol = make_volume();
        assert!(vol.slice(4).is_err());
        assert!(vol.slice(100).is_err());
    }

    #[test]
    fn test_volume_mip_axis0() {
        let vol = make_volume();
        let mip = vol.mip(0).expect("MIP axis 0 should succeed");
        assert_eq!(mip.shape(), &[3, 3]);
        // Maximum over z axis: last slice (z=3) has highest values
        assert!((mip[[0, 0]] - 27.0).abs() < 1e-10);
        assert!((mip[[2, 2]] - 35.0).abs() < 1e-10);
    }

    #[test]
    fn test_volume_min_ip_axis0() {
        let vol = make_volume();
        let min_ip = vol.min_ip(0).expect("MinIP axis 0 should succeed");
        // Minimum over z axis: first slice (z=0) has lowest values
        assert!((min_ip[[0, 0]] - 0.0).abs() < 1e-10);
        assert!((min_ip[[2, 2]] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_volume_avg_ip_axis0() {
        let vol = make_volume();
        let avg = vol.avg_ip(0).expect("AvgIP axis 0 should succeed");
        // pixel (0,0) across z=0,1,2,3 → values 0,9,18,27 → avg=13.5
        assert!((avg[[0, 0]] - 13.5).abs() < 1e-10);
    }

    #[test]
    fn test_volume_mip_invalid_axis() {
        let vol = make_volume();
        assert!(vol.mip(3).is_err());
    }
}
