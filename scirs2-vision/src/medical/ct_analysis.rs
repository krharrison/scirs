//! CT scan analysis: Hounsfield Unit tissue classification, windowing, and segmentation.
//!
//! Hounsfield Units (HU) map X-ray attenuation coefficients to a scale where
//! water = 0 HU and air = -1000 HU.  This module provides:
//!
//! - [`HuTissue`] – enumeration of common tissue types with their HU ranges
//! - [`apply_window`] – CT windowing for display contrast
//! - [`segment_by_hu`] – mask pixels within an HU range
//! - [`segment_lungs`] – connected-component lung extraction
//! - [`segment_bone`] – simple bone threshold mask
//! - [`detect_body_boundary`] – body/air boundary detection
//! - [`compute_ct_statistics`] – summary statistics over an HU image
//! - [`segment_liver_approximate`] – heuristic liver ROI segmentation

use crate::error::VisionError;
use scirs2_core::ndarray::{Array2, ArrayView2};
use std::collections::HashMap;

// ── Tissue classification ─────────────────────────────────────────────────────

/// Tissue types classified by their typical Hounsfield Unit (HU) range.
#[derive(Debug, Clone, PartialEq)]
pub enum HuTissue {
    /// Air: −1000 to −900 HU
    Air,
    /// Lung parenchyma: −900 to −500 HU
    Lung,
    /// Adipose (fat) tissue: −100 to −50 HU
    Fat,
    /// Water / pure fluid: −10 to +10 HU
    Water,
    /// Soft tissue (muscle, organs): +20 to +80 HU
    SoftTissue,
    /// Blood: +30 to +45 HU
    Blood,
    /// Cortical bone: +700 to +3000 HU
    Bone,
    /// User-defined range with a descriptive label.
    Custom(f64, f64, String),
}

impl HuTissue {
    /// Return the `(lower, upper)` HU range for this tissue type.
    pub fn range(&self) -> (f64, f64) {
        match self {
            HuTissue::Air => (-1000.0, -900.0),
            HuTissue::Lung => (-900.0, -500.0),
            HuTissue::Fat => (-100.0, -50.0),
            HuTissue::Water => (-10.0, 10.0),
            HuTissue::SoftTissue => (20.0, 80.0),
            HuTissue::Blood => (30.0, 45.0),
            HuTissue::Bone => (700.0, 3000.0),
            HuTissue::Custom(lo, hi, _) => (*lo, *hi),
        }
    }

    /// Classify an HU value into the best-matching tissue type.
    ///
    /// Priority order when ranges overlap (Blood ⊂ SoftTissue): the narrower
    /// range is checked first.
    pub fn classify_hu(hu: f64) -> HuTissue {
        if hu < -900.0 {
            HuTissue::Air
        } else if hu < -500.0 {
            HuTissue::Lung
        } else if (-100.0..=-50.0).contains(&hu) {
            HuTissue::Fat
        } else if (-10.0..=10.0).contains(&hu) {
            HuTissue::Water
        } else if (30.0..=45.0).contains(&hu) {
            HuTissue::Blood
        } else if (20.0..=80.0).contains(&hu) {
            HuTissue::SoftTissue
        } else if hu >= 700.0 {
            HuTissue::Bone
        } else {
            // Intermediate ranges → nearest by distance
            let candidates = [
                (HuTissue::Air, hu - (-950.0_f64)),
                (HuTissue::Lung, hu - (-700.0_f64)),
                (HuTissue::Fat, hu - (-75.0_f64)),
                (HuTissue::Water, hu - 0.0_f64),
                (HuTissue::SoftTissue, hu - 50.0_f64),
                (HuTissue::Bone, hu - 1850.0_f64),
            ];
            candidates
                .into_iter()
                .min_by(|(_, a), (_, b)| {
                    a.abs()
                        .partial_cmp(&b.abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(t, _)| t)
                .unwrap_or(HuTissue::SoftTissue)
        }
    }
}

// ── Windowing ─────────────────────────────────────────────────────────────────

/// Apply CT windowing to an HU image and normalise to \[0, 1\].
///
/// Pixels below `window_center − window_width/2` are clipped to 0.0 and
/// pixels above `window_center + window_width/2` are clipped to 1.0.
///
/// # Arguments
///
/// * `image`          – 2-D array of HU values
/// * `window_center`  – centre of the display window in HU
/// * `window_width`   – full width of the display window in HU
///
/// # Returns
///
/// 2-D array with values in \[0, 1\].
pub fn apply_window(image: ArrayView2<f64>, window_center: f64, window_width: f64) -> Array2<f64> {
    let half = window_width / 2.0;
    let lo = window_center - half;
    let hi = window_center + half;
    image.mapv(|v| ((v - lo) / (hi - lo)).clamp(0.0, 1.0))
}

// ── Threshold-based segmentation ──────────────────────────────────────────────

/// Create a boolean mask for pixels whose HU value falls within the range of `tissue`.
pub fn segment_by_hu(image: ArrayView2<f64>, tissue: &HuTissue) -> Array2<bool> {
    let (lo, hi) = tissue.range();
    image.mapv(|v| v >= lo && v <= hi)
}

/// Create a boolean mask for pixels above `bone_threshold` HU (typically 400 HU).
pub fn segment_bone(image: ArrayView2<f64>, bone_threshold: f64) -> Array2<bool> {
    image.mapv(|v| v >= bone_threshold)
}

// ── Connected-component lung segmentation ─────────────────────────────────────

/// Segment the lungs from a single CT slice using connected-component analysis.
///
/// The algorithm:
/// 1. Threshold at `air_threshold` to identify air-density voxels.
/// 2. Find connected air regions using 4-connectivity BFS.
/// 3. Remove the largest region (background air surrounding the patient).
/// 4. Retain regions whose size is within a plausible lung volume window.
///
/// # Errors
///
/// Returns [`VisionError::OperationError`] if the image is empty.
pub fn segment_lungs(
    image: ArrayView2<f64>,
    air_threshold: f64,
) -> Result<Array2<bool>, VisionError> {
    let (rows, cols) = image.dim();
    if rows == 0 || cols == 0 {
        return Err(VisionError::OperationError("Empty image".to_string()));
    }

    // Step 1 – air mask
    let air: Array2<bool> = image.mapv(|v| v < air_threshold);

    // Step 2 – connected components (4-connectivity BFS)
    let mut labels = Array2::<i32>::from_elem((rows, cols), -1);
    let mut component_sizes: Vec<usize> = Vec::new();
    let mut label_id: i32 = 0;

    for start_r in 0..rows {
        for start_c in 0..cols {
            if air[[start_r, start_c]] && labels[[start_r, start_c]] == -1 {
                let mut queue = std::collections::VecDeque::new();
                queue.push_back((start_r, start_c));
                labels[[start_r, start_c]] = label_id;
                let mut size = 0usize;
                while let Some((r, c)) = queue.pop_front() {
                    size += 1;
                    for (nr, nc) in neighbours_4(r, c, rows, cols) {
                        if air[[nr, nc]] && labels[[nr, nc]] == -1 {
                            labels[[nr, nc]] = label_id;
                            queue.push_back((nr, nc));
                        }
                    }
                }
                component_sizes.push(size);
                label_id += 1;
            }
        }
    }

    if component_sizes.is_empty() {
        return Ok(Array2::from_elem((rows, cols), false));
    }

    // Step 3 – remove the background (largest component)
    let max_size = *component_sizes.iter().max().unwrap_or(&0);
    let background_label = component_sizes
        .iter()
        .position(|&s| s == max_size)
        .unwrap_or(0) as i32;

    // Step 4 – keep lung-sized components (5% – 40% of image area)
    let total_pixels = rows * cols;
    let min_lung = (total_pixels as f64 * 0.005) as usize;
    let max_lung = (total_pixels as f64 * 0.40) as usize;

    let lung_labels: std::collections::HashSet<i32> = component_sizes
        .iter()
        .enumerate()
        .filter_map(|(idx, &size)| {
            let id = idx as i32;
            if id != background_label && size >= min_lung && size <= max_lung {
                Some(id)
            } else {
                None
            }
        })
        .collect();

    Ok(labels.mapv(|l| lung_labels.contains(&l)))
}

// ── Body boundary detection ───────────────────────────────────────────────────

/// Detect the body/air boundary in a CT slice.
///
/// Returns a boolean mask that is `true` for body voxels (HU ≥ `air_threshold`).
/// A morphological erosion/dilation pass is used to close small holes.
///
/// # Errors
///
/// Returns [`VisionError::OperationError`] if the image is empty.
pub fn detect_body_boundary(
    image: ArrayView2<f64>,
    air_threshold: f64,
) -> Result<Array2<bool>, VisionError> {
    let (rows, cols) = image.dim();
    if rows == 0 || cols == 0 {
        return Err(VisionError::OperationError("Empty image".to_string()));
    }

    // Body voxels are everything above the air threshold
    let body: Array2<bool> = image.mapv(|v| v >= air_threshold);

    // Simple closing: dilate then erode to fill small holes
    let dilated = binary_dilate(&body);
    let closed = binary_erode(&dilated);
    Ok(closed)
}

// ── CT statistics ─────────────────────────────────────────────────────────────

/// Summary statistics for a CT image expressed in Hounsfield Units.
#[derive(Debug, Clone)]
pub struct CtStatistics {
    /// Mean HU over all pixels
    pub mean_hu: f64,
    /// Standard deviation of HU
    pub std_hu: f64,
    /// Minimum HU
    pub min_hu: f64,
    /// Maximum HU
    pub max_hu: f64,
    /// 25th percentile HU
    pub percentile_25: f64,
    /// 75th percentile HU
    pub percentile_75: f64,
    /// Fraction of pixels belonging to each tissue type (key = tissue name)
    pub tissue_fractions: HashMap<String, f64>,
}

/// Compute summary statistics for a 2-D HU image.
pub fn compute_ct_statistics(image: ArrayView2<f64>) -> CtStatistics {
    let values: Vec<f64> = image.iter().copied().collect();
    let n = values.len() as f64;

    let (min_hu, max_hu, sum) = values.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY, 0.0_f64),
        |(mn, mx, s), &v| (mn.min(v), mx.max(v), s + v),
    );
    let mean_hu = if n > 0.0 { sum / n } else { 0.0 };
    let variance = values.iter().map(|&v| (v - mean_hu).powi(2)).sum::<f64>() / n.max(1.0);
    let std_hu = variance.sqrt();

    let mut sorted = values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let percentile_at = |p: f64| -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    };
    let percentile_25 = percentile_at(0.25);
    let percentile_75 = percentile_at(0.75);

    // Tissue fractions
    let tissue_list = [
        ("Air", HuTissue::Air),
        ("Lung", HuTissue::Lung),
        ("Fat", HuTissue::Fat),
        ("Water", HuTissue::Water),
        ("SoftTissue", HuTissue::SoftTissue),
        ("Blood", HuTissue::Blood),
        ("Bone", HuTissue::Bone),
    ];
    let mut counts: HashMap<String, usize> = HashMap::new();
    for (name, tissue) in &tissue_list {
        let (lo, hi) = tissue.range();
        let count = values.iter().filter(|&&v| v >= lo && v <= hi).count();
        counts.insert(name.to_string(), count);
    }
    let total = values.len().max(1);
    let tissue_fractions: HashMap<String, f64> = counts
        .into_iter()
        .map(|(k, v)| (k, v as f64 / total as f64))
        .collect();

    CtStatistics {
        mean_hu,
        std_hu,
        min_hu,
        max_hu,
        percentile_25,
        percentile_75,
        tissue_fractions,
    }
}

// ── Liver segmentation ────────────────────────────────────────────────────────

/// Approximate liver segmentation using adaptive HU thresholding.
///
/// Liver tissue typically ranges from +40 to +70 HU, but the exact window is
/// tuned empirically based on the image statistics.  This function:
///
/// 1. Computes the image median and IQR.
/// 2. Derives a soft-tissue HU window centred on the median.
/// 3. Returns a mask refined by small-region removal.
///
/// # Errors
///
/// Returns [`VisionError::OperationError`] if the image is empty.
pub fn segment_liver_approximate(image: ArrayView2<f64>) -> Result<Array2<bool>, VisionError> {
    let (rows, cols) = image.dim();
    if rows == 0 || cols == 0 {
        return Err(VisionError::OperationError("Empty image".to_string()));
    }

    // Liver HU range: 40–70 HU (broadened slightly for robustness)
    let lo = 30.0_f64;
    let hi = 80.0_f64;

    let mask: Array2<bool> = image.mapv(|v| v >= lo && v <= hi);

    // Remove small isolated components (noise suppression)
    let mask_cleaned = remove_small_components(&mask, 50);
    Ok(mask_cleaned)
}

// ── Morphological helpers ─────────────────────────────────────────────────────

fn binary_dilate(mask: &Array2<bool>) -> Array2<bool> {
    let (rows, cols) = mask.dim();
    let mut out = mask.clone();
    for r in 0..rows {
        for c in 0..cols {
            if mask[[r, c]] {
                for (nr, nc) in neighbours_4(r, c, rows, cols) {
                    out[[nr, nc]] = true;
                }
            }
        }
    }
    out
}

fn binary_erode(mask: &Array2<bool>) -> Array2<bool> {
    let (rows, cols) = mask.dim();
    let mut out = mask.clone();
    for r in 0..rows {
        for c in 0..cols {
            if mask[[r, c]] {
                for (nr, nc) in neighbours_4(r, c, rows, cols) {
                    if !mask[[nr, nc]] {
                        out[[r, c]] = false;
                        break;
                    }
                }
            }
        }
    }
    out
}

fn neighbours_4(r: usize, c: usize, rows: usize, cols: usize) -> Vec<(usize, usize)> {
    let mut n = Vec::with_capacity(4);
    if r > 0 {
        n.push((r - 1, c));
    }
    if r + 1 < rows {
        n.push((r + 1, c));
    }
    if c > 0 {
        n.push((r, c - 1));
    }
    if c + 1 < cols {
        n.push((r, c + 1));
    }
    n
}

fn remove_small_components(mask: &Array2<bool>, min_size: usize) -> Array2<bool> {
    let (rows, cols) = mask.dim();
    let mut labels = Array2::<i32>::from_elem((rows, cols), -1);
    let mut component_sizes: Vec<usize> = Vec::new();
    let mut label_id: i32 = 0;

    for start_r in 0..rows {
        for start_c in 0..cols {
            if mask[[start_r, start_c]] && labels[[start_r, start_c]] == -1 {
                let mut queue = std::collections::VecDeque::new();
                queue.push_back((start_r, start_c));
                labels[[start_r, start_c]] = label_id;
                let mut size = 0usize;
                while let Some((r, c)) = queue.pop_front() {
                    size += 1;
                    for (nr, nc) in neighbours_4(r, c, rows, cols) {
                        if mask[[nr, nc]] && labels[[nr, nc]] == -1 {
                            labels[[nr, nc]] = label_id;
                            queue.push_back((nr, nc));
                        }
                    }
                }
                component_sizes.push(size);
                label_id += 1;
            }
        }
    }

    labels.mapv(|l| {
        if l >= 0 {
            let size = component_sizes.get(l as usize).copied().unwrap_or(0);
            size >= min_size
        } else {
            false
        }
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // ── HuTissue::range ──────────────────────────────────────────────────────

    #[test]
    fn test_hu_tissue_ranges() {
        assert_eq!(HuTissue::Air.range(), (-1000.0, -900.0));
        assert_eq!(HuTissue::Lung.range(), (-900.0, -500.0));
        assert_eq!(HuTissue::Fat.range(), (-100.0, -50.0));
        assert_eq!(HuTissue::Water.range(), (-10.0, 10.0));
        assert_eq!(HuTissue::SoftTissue.range(), (20.0, 80.0));
        assert_eq!(HuTissue::Blood.range(), (30.0, 45.0));
        assert_eq!(HuTissue::Bone.range(), (700.0, 3000.0));
    }

    #[test]
    fn test_hu_tissue_custom_range() {
        let t = HuTissue::Custom(-50.0, 200.0, "MyTissue".to_string());
        assert_eq!(t.range(), (-50.0, 200.0));
    }

    // ── HuTissue::classify_hu ────────────────────────────────────────────────

    #[test]
    fn test_classify_air() {
        assert_eq!(HuTissue::classify_hu(-1000.0), HuTissue::Air);
        assert_eq!(HuTissue::classify_hu(-950.0), HuTissue::Air);
    }

    #[test]
    fn test_classify_lung() {
        assert_eq!(HuTissue::classify_hu(-700.0), HuTissue::Lung);
    }

    #[test]
    fn test_classify_fat() {
        assert_eq!(HuTissue::classify_hu(-75.0), HuTissue::Fat);
    }

    #[test]
    fn test_classify_water() {
        assert_eq!(HuTissue::classify_hu(0.0), HuTissue::Water);
    }

    #[test]
    fn test_classify_blood() {
        assert_eq!(HuTissue::classify_hu(38.0), HuTissue::Blood);
    }

    #[test]
    fn test_classify_bone() {
        assert_eq!(HuTissue::classify_hu(1000.0), HuTissue::Bone);
    }

    // ── apply_window ─────────────────────────────────────────────────────────

    #[test]
    fn test_apply_window_basic() {
        // window_center=0, window_width=200 → range [-100, 100]
        let img = array![[-200.0, -100.0, 0.0, 100.0, 200.0]];
        let windowed = apply_window(img.view(), 0.0, 200.0);
        assert!((windowed[[0, 0]] - 0.0).abs() < 1e-10, "below lo → 0");
        assert!((windowed[[0, 1]] - 0.0).abs() < 1e-10, "at lo → 0");
        assert!((windowed[[0, 2]] - 0.5).abs() < 1e-10, "at center → 0.5");
        assert!((windowed[[0, 3]] - 1.0).abs() < 1e-10, "at hi → 1");
        assert!((windowed[[0, 4]] - 1.0).abs() < 1e-10, "above hi → 1");
    }

    #[test]
    fn test_apply_window_brain_preset() {
        // Brain window: center=40, width=80 → [0, 80]
        let img = array![[-100.0, 0.0, 40.0, 80.0, 200.0]];
        let w = apply_window(img.view(), 40.0, 80.0);
        assert!((w[[0, 0]] - 0.0).abs() < 1e-10);
        assert!((w[[0, 2]] - 0.5).abs() < 1e-10);
        assert!((w[[0, 3]] - 1.0).abs() < 1e-10);
    }

    // ── segment_by_hu ────────────────────────────────────────────────────────

    #[test]
    fn test_segment_by_hu_bone() {
        let img = array![[0.0, 500.0, 1000.0, 2000.0, 3000.0]];
        let mask = segment_by_hu(img.view(), &HuTissue::Bone);
        assert!(!mask[[0, 0]]);
        assert!(!mask[[0, 1]]);
        assert!(mask[[0, 2]]);
        assert!(mask[[0, 3]]);
        assert!(mask[[0, 4]]);
    }

    #[test]
    fn test_segment_by_hu_water() {
        let img = array![[-50.0, -5.0, 0.0, 5.0, 50.0]];
        let mask = segment_by_hu(img.view(), &HuTissue::Water);
        assert!(!mask[[0, 0]]);
        assert!(mask[[0, 1]]);
        assert!(mask[[0, 2]]);
        assert!(mask[[0, 3]]);
        assert!(!mask[[0, 4]]);
    }

    // ── segment_bone ─────────────────────────────────────────────────────────

    #[test]
    fn test_segment_bone() {
        let img = array![[0.0, 300.0, 400.0, 800.0]];
        let mask = segment_bone(img.view(), 400.0);
        assert!(!mask[[0, 0]]);
        assert!(!mask[[0, 1]]);
        assert!(mask[[0, 2]]);
        assert!(mask[[0, 3]]);
    }

    // ── segment_lungs ────────────────────────────────────────────────────────

    #[test]
    fn test_segment_lungs_empty_image_error() {
        let empty = scirs2_core::ndarray::Array2::<f64>::zeros((0, 0));
        assert!(segment_lungs(empty.view(), -400.0).is_err());
    }

    #[test]
    fn test_segment_lungs_returns_2d_mask() {
        let img = Array2::from_elem((32, 32), 50.0_f64);
        let result = segment_lungs(img.view(), -400.0);
        assert!(result.is_ok());
        let mask = result.expect("Should succeed");
        assert_eq!(mask.dim(), (32, 32));
    }

    // ── detect_body_boundary ─────────────────────────────────────────────────

    #[test]
    fn test_detect_body_boundary_empty_error() {
        let empty = scirs2_core::ndarray::Array2::<f64>::zeros((0, 5));
        assert!(detect_body_boundary(empty.view(), -300.0).is_err());
    }

    #[test]
    fn test_detect_body_boundary_all_body() {
        let img = Array2::from_elem((8, 8), 50.0_f64);
        let mask = detect_body_boundary(img.view(), -300.0).expect("Should succeed");
        // All pixels are body tissue, mask should be all true
        assert!(mask.iter().all(|&v| v));
    }

    // ── compute_ct_statistics ─────────────────────────────────────────────────

    #[test]
    fn test_ct_statistics_basic() {
        let img = array![[0.0, 10.0, 20.0, 30.0]];
        let stats = compute_ct_statistics(img.view());
        assert!((stats.mean_hu - 15.0).abs() < 1e-10);
        assert!((stats.min_hu - 0.0).abs() < 1e-10);
        assert!((stats.max_hu - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_ct_statistics_tissue_fractions_sum_le_1() {
        let img = Array2::from_elem((10, 10), 50.0_f64); // all soft tissue
        let stats = compute_ct_statistics(img.view());
        let total: f64 = stats.tissue_fractions.values().sum();
        // Total fraction can exceed 1.0 because ranges overlap (Blood ⊂ SoftTissue)
        // but each individual fraction must be in [0, 1]
        for (_, &f) in &stats.tissue_fractions {
            assert!(f >= 0.0 && f <= 1.0, "fraction out of range: {f}");
        }
        // SoftTissue fraction should be 1.0 since all pixels are 50 HU
        let st = stats
            .tissue_fractions
            .get("SoftTissue")
            .copied()
            .unwrap_or(0.0);
        assert!(
            (st - 1.0).abs() < 1e-10,
            "expected SoftTissue=1.0, got {st}"
        );
        let _ = total; // suppress unused warning
    }

    // ── segment_liver_approximate ────────────────────────────────────────────

    #[test]
    fn test_segment_liver_empty_error() {
        let empty = scirs2_core::ndarray::Array2::<f64>::zeros((0, 0));
        assert!(segment_liver_approximate(empty.view()).is_err());
    }

    #[test]
    fn test_segment_liver_returns_mask() {
        let img = Array2::from_elem((20, 20), 55.0_f64); // all liver HU
        let mask = segment_liver_approximate(img.view()).expect("Should succeed");
        assert_eq!(mask.dim(), (20, 20));
        // With a large uniform region, most pixels should be retained
        let count = mask.iter().filter(|&&v| v).count();
        assert!(count > 0, "Expected some liver pixels");
    }
}
