//! Enhanced region properties computation
//!
//! Provides comprehensive geometric and statistical properties for labeled regions
//! in 2D images, similar to scikit-image's `regionprops`.

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign};
use std::collections::HashMap;
use std::fmt::Debug;

/// Comprehensive properties of a labeled region in a 2D image
#[derive(Debug, Clone)]
pub struct RegionProperties2D<T: Float> {
    /// Label value of this region
    pub label: usize,
    /// Number of pixels in the region
    pub area: usize,
    /// Centroid (row, col) -- geometric center of mass
    pub centroid: (T, T),
    /// Weighted centroid using intensity values (row, col)
    pub weighted_centroid: (T, T),
    /// Bounding box (min_row, min_col, max_row, max_col) -- max is exclusive
    pub bounding_box: (usize, usize, usize, usize),
    /// Perimeter of the region (boundary pixel count, inner boundary)
    pub perimeter: T,
    /// Eccentricity of the equivalent ellipse (0 = circle, 1 = line)
    pub eccentricity: T,
    /// Orientation angle in radians (angle of major axis w.r.t. horizontal)
    pub orientation: T,
    /// Equivalent diameter: diameter of a circle with the same area
    pub equivalent_diameter: T,
    /// Length of the major axis of the equivalent ellipse
    pub major_axis_length: T,
    /// Length of the minor axis of the equivalent ellipse
    pub minor_axis_length: T,
    /// Solidity: ratio of area to convex hull area
    pub solidity: T,
    /// Extent: ratio of area to bounding box area
    pub extent: T,
    /// Mean intensity of the region
    pub mean_intensity: T,
    /// Minimum intensity in the region
    pub min_intensity: T,
    /// Maximum intensity in the region
    pub max_intensity: T,
    /// Second-order central moments (mu_20, mu_11, mu_02)
    pub moments_central: (T, T, T),
}

/// Internal structure for accumulating pixel data during a single pass
struct RegionAccumulator {
    area: usize,
    sum_r: f64,
    sum_c: f64,
    sum_r_weighted: f64,
    sum_c_weighted: f64,
    sum_intensity: f64,
    min_intensity: f64,
    max_intensity: f64,
    min_row: usize,
    max_row: usize,
    min_col: usize,
    max_col: usize,
    // For second-order moments (accumulated after centroid is known)
    coords: Vec<(usize, usize)>,
    intensities: Vec<f64>,
}

impl RegionAccumulator {
    fn new() -> Self {
        RegionAccumulator {
            area: 0,
            sum_r: 0.0,
            sum_c: 0.0,
            sum_r_weighted: 0.0,
            sum_c_weighted: 0.0,
            sum_intensity: 0.0,
            min_intensity: f64::INFINITY,
            max_intensity: f64::NEG_INFINITY,
            min_row: usize::MAX,
            max_row: 0,
            min_col: usize::MAX,
            max_col: 0,
            coords: Vec::new(),
            intensities: Vec::new(),
        }
    }

    fn add_pixel(&mut self, r: usize, c: usize, intensity: f64) {
        self.area += 1;
        self.sum_r += r as f64;
        self.sum_c += c as f64;
        self.sum_r_weighted += r as f64 * intensity;
        self.sum_c_weighted += c as f64 * intensity;
        self.sum_intensity += intensity;
        if intensity < self.min_intensity {
            self.min_intensity = intensity;
        }
        if intensity > self.max_intensity {
            self.max_intensity = intensity;
        }
        if r < self.min_row {
            self.min_row = r;
        }
        if r > self.max_row {
            self.max_row = r;
        }
        if c < self.min_col {
            self.min_col = c;
        }
        if c > self.max_col {
            self.max_col = c;
        }
        self.coords.push((r, c));
        self.intensities.push(intensity);
    }
}

/// Compute comprehensive properties for each labeled region in a 2D image
///
/// Analyzes a labeled image and computes geometric, statistical, and shape
/// properties for each region. This is the 2D equivalent of scikit-image's
/// `regionprops` function.
///
/// # Arguments
///
/// * `image` - Intensity image (grayscale)
/// * `labels` - Labeled image (0 = background, positive integers = region labels)
///
/// # Returns
///
/// * `Result<Vec<RegionProperties2D<T>>>` - Properties for each region, sorted by label
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::measurements::regionprops_2d;
///
/// let image: scirs2_core::ndarray::Array2<f64> = array![
///     [100.0, 100.0, 0.0, 200.0, 200.0],
///     [100.0, 100.0, 0.0, 200.0, 200.0],
///     [0.0,   0.0,   0.0, 0.0,   0.0],
///     [150.0, 150.0, 150.0, 150.0, 150.0],
///     [150.0, 150.0, 150.0, 150.0, 150.0],
/// ];
///
/// let labels = array![
///     [1, 1, 0, 2, 2],
///     [1, 1, 0, 2, 2],
///     [0, 0, 0, 0, 0],
///     [3, 3, 3, 3, 3],
///     [3, 3, 3, 3, 3],
/// ];
///
/// let props = regionprops_2d(&image, &labels).expect("regionprops_2d should succeed");
/// assert_eq!(props.len(), 3);
/// assert_eq!(props[0].area, 4); // Region 1
/// ```
pub fn regionprops_2d<T>(
    image: &Array2<T>,
    labels: &Array2<usize>,
) -> NdimageResult<Vec<RegionProperties2D<T>>>
where
    T: Float + FromPrimitive + NumAssign + Debug + Copy + 'static,
{
    if image.shape() != labels.shape() {
        return Err(NdimageError::DimensionError(
            "Image and labels must have the same shape".to_string(),
        ));
    }

    let rows = image.nrows();
    let cols = image.ncols();

    if rows == 0 || cols == 0 {
        return Ok(Vec::new());
    }

    // Single pass to accumulate data per label
    let mut accumulators: HashMap<usize, RegionAccumulator> = HashMap::new();

    for r in 0..rows {
        for c in 0..cols {
            let lbl = labels[[r, c]];
            if lbl == 0 {
                continue;
            }
            let intensity = image[[r, c]].to_f64().unwrap_or(0.0);
            let acc = accumulators
                .entry(lbl)
                .or_insert_with(RegionAccumulator::new);
            acc.add_pixel(r, c, intensity);
        }
    }

    // For perimeter computation, we need to check boundaries
    // We'll pre-compute a boundary map
    let mut perimeter_counts: HashMap<usize, usize> = HashMap::new();
    for r in 0..rows {
        for c in 0..cols {
            let lbl = labels[[r, c]];
            if lbl == 0 {
                continue;
            }
            // A pixel is on the perimeter if any 4-connected neighbor has a different label
            let is_boundary = (r == 0 || labels[[r - 1, c]] != lbl)
                || (r + 1 >= rows || labels[[r + 1, c]] != lbl)
                || (c == 0 || labels[[r, c - 1]] != lbl)
                || (c + 1 >= cols || labels[[r, c + 1]] != lbl);
            if is_boundary {
                *perimeter_counts.entry(lbl).or_insert(0) += 1;
            }
        }
    }

    // Build properties for each region
    let mut result = Vec::with_capacity(accumulators.len());

    for (lbl, acc) in &accumulators {
        let area = acc.area;
        let area_f = area as f64;

        // Geometric centroid
        let centroid_r = acc.sum_r / area_f;
        let centroid_c = acc.sum_c / area_f;

        // Weighted centroid
        let (wcr, wcc) = if acc.sum_intensity > 0.0 {
            (
                acc.sum_r_weighted / acc.sum_intensity,
                acc.sum_c_weighted / acc.sum_intensity,
            )
        } else {
            (centroid_r, centroid_c)
        };

        // Second-order central moments
        let mut mu_20 = 0.0;
        let mut mu_11 = 0.0;
        let mut mu_02 = 0.0;
        for &(r, c) in &acc.coords {
            let dr = r as f64 - centroid_r;
            let dc = c as f64 - centroid_c;
            mu_20 += dr * dr;
            mu_11 += dr * dc;
            mu_02 += dc * dc;
        }
        // Normalize by area
        mu_20 /= area_f;
        mu_11 /= area_f;
        mu_02 /= area_f;

        // Orientation: angle of major axis
        let orientation = 0.5 * (2.0 * mu_11).atan2(mu_20 - mu_02);

        // Eigenvalues of the inertia tensor for ellipse fitting
        let trace = mu_20 + mu_02;
        let det = mu_20 * mu_02 - mu_11 * mu_11;
        let discriminant = (trace * trace - 4.0 * det).max(0.0);
        let sqrt_disc = discriminant.sqrt();
        let lambda1 = (trace + sqrt_disc) / 2.0; // larger eigenvalue
        let lambda2 = (trace - sqrt_disc) / 2.0; // smaller eigenvalue

        // Axis lengths (4 * sqrt(eigenvalue) for the major/minor axes)
        let major_axis = 4.0 * lambda1.max(0.0).sqrt();
        let minor_axis = 4.0 * lambda2.max(0.0).sqrt();

        // Eccentricity
        let eccentricity = if major_axis > 1e-15 {
            let ratio = minor_axis / major_axis;
            (1.0 - ratio * ratio).max(0.0).sqrt()
        } else {
            0.0
        };

        // Equivalent diameter
        let equivalent_diameter = (4.0 * area_f / std::f64::consts::PI).sqrt();

        // Perimeter
        let perim = *perimeter_counts.get(lbl).unwrap_or(&0) as f64;

        // Bounding box area
        let bbox_height = acc.max_row - acc.min_row + 1;
        let bbox_width = acc.max_col - acc.min_col + 1;
        let bbox_area = bbox_height * bbox_width;

        // Extent: area / bbox_area
        let extent = if bbox_area > 0 {
            area_f / bbox_area as f64
        } else {
            0.0
        };

        // Solidity: area / convex_hull_area
        // Computing true convex hull is expensive; approximate with
        // Andrew's monotone chain for the collected coordinates
        let convex_hull_area = compute_convex_hull_area(&acc.coords);
        let solidity = if convex_hull_area > 1e-15 {
            area_f / convex_hull_area
        } else {
            1.0 // single pixel or collinear
        };

        // Mean intensity
        let mean_intensity = acc.sum_intensity / area_f;

        // Convert f64 values to T
        let to_t = |v: f64| -> T { T::from_f64(v).unwrap_or(T::zero()) };

        result.push(RegionProperties2D {
            label: *lbl,
            area,
            centroid: (to_t(centroid_r), to_t(centroid_c)),
            weighted_centroid: (to_t(wcr), to_t(wcc)),
            bounding_box: (acc.min_row, acc.min_col, acc.max_row + 1, acc.max_col + 1),
            perimeter: to_t(perim),
            eccentricity: to_t(eccentricity),
            orientation: to_t(orientation),
            equivalent_diameter: to_t(equivalent_diameter),
            major_axis_length: to_t(major_axis),
            minor_axis_length: to_t(minor_axis),
            solidity: to_t(solidity.min(1.0)),
            extent: to_t(extent),
            mean_intensity: to_t(mean_intensity),
            min_intensity: to_t(acc.min_intensity),
            max_intensity: to_t(acc.max_intensity),
            moments_central: (to_t(mu_20), to_t(mu_11), to_t(mu_02)),
        });
    }

    result.sort_by_key(|p| p.label);
    Ok(result)
}

/// Compute the area of the convex hull of a set of 2D points
/// using Andrew's monotone chain algorithm
fn compute_convex_hull_area(coords: &[(usize, usize)]) -> f64 {
    if coords.len() < 3 {
        return coords.len() as f64; // 0, 1, or 2 points
    }

    // Sort points by (col, row) for the monotone chain algorithm
    let mut points: Vec<(f64, f64)> = coords.iter().map(|&(r, c)| (c as f64, r as f64)).collect();
    points.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    points.dedup();

    if points.len() < 3 {
        return points.len() as f64;
    }

    // Build lower hull
    let mut lower: Vec<(f64, f64)> = Vec::new();
    for &p in &points {
        while lower.len() >= 2 {
            let len = lower.len();
            if cross(lower[len - 2], lower[len - 1], p) <= 0.0 {
                lower.pop();
            } else {
                break;
            }
        }
        lower.push(p);
    }

    // Build upper hull
    let mut upper: Vec<(f64, f64)> = Vec::new();
    for &p in points.iter().rev() {
        while upper.len() >= 2 {
            let len = upper.len();
            if cross(upper[len - 2], upper[len - 1], p) <= 0.0 {
                upper.pop();
            } else {
                break;
            }
        }
        upper.push(p);
    }

    // Remove last point of each half because it's repeated
    lower.pop();
    upper.pop();

    let hull: Vec<(f64, f64)> = lower.into_iter().chain(upper).collect();

    if hull.len() < 3 {
        return hull.len() as f64;
    }

    // Shoelace formula for polygon area
    let n = hull.len();
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += hull[i].0 * hull[j].1;
        area -= hull[j].0 * hull[i].1;
    }

    area.abs() / 2.0
}

/// Cross product of vectors OA and OB where O, A, B are 2D points
#[inline]
fn cross(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;
    // All test values are f64 so  is identity

    #[test]
    fn test_regionprops_2d_basic() {
        let image: Array2<f64> = array![
            [100.0, 100.0, 0.0, 200.0, 200.0],
            [100.0, 100.0, 0.0, 200.0, 200.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [150.0, 150.0, 150.0, 150.0, 150.0],
            [150.0, 150.0, 150.0, 150.0, 150.0],
        ];
        let labels = array![
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0],
            [3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3],
        ];

        let props = regionprops_2d(&image, &labels).expect("regionprops should succeed");
        assert_eq!(props.len(), 3);

        // Region 1: 4 pixels in rows 0-1, cols 0-1
        let r1 = &props[0];
        assert_eq!(r1.label, 1);
        assert_eq!(r1.area, 4);
        assert_abs_diff_eq!(r1.centroid.0, 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(r1.centroid.1, 0.5, epsilon = 1e-10);
        assert_eq!(r1.bounding_box, (0, 0, 2, 2));
        assert_abs_diff_eq!(r1.mean_intensity, 100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_regionprops_2d_shape_mismatch() {
        let image: Array2<f64> = Array2::zeros((3, 3));
        let labels: Array2<usize> = Array2::zeros((4, 4));
        let result = regionprops_2d(&image, &labels);
        assert!(result.is_err());
    }

    #[test]
    fn test_regionprops_2d_empty_labels() {
        let image: Array2<f64> = Array2::from_elem((3, 3), 1.0);
        let labels: Array2<usize> = Array2::zeros((3, 3));
        let props = regionprops_2d(&image, &labels).expect("empty labels should succeed");
        assert!(props.is_empty());
    }

    #[test]
    fn test_regionprops_2d_single_pixel() {
        let image: Array2<f64> = Array2::from_elem((5, 5), 0.0);
        let mut labels: Array2<usize> = Array2::zeros((5, 5));
        labels[[2, 3]] = 1;

        let props = regionprops_2d(&image, &labels).expect("single pixel should succeed");
        assert_eq!(props.len(), 1);
        assert_eq!(props[0].area, 1);
        assert_abs_diff_eq!(props[0].centroid.0, 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(props[0].centroid.1, 3.0, epsilon = 1e-10);
        assert_eq!(props[0].bounding_box, (2, 3, 3, 4));
    }

    #[test]
    fn test_regionprops_2d_circle_eccentricity() {
        // Approximate circle: eccentricity should be near 0
        let mut labels: Array2<usize> = Array2::zeros((21, 21));
        let mut image: Array2<f64> = Array2::zeros((21, 21));

        for r in 0..21 {
            for c in 0..21 {
                let dr = r as f64 - 10.0;
                let dc = c as f64 - 10.0;
                if dr * dr + dc * dc <= 64.0 {
                    // radius 8
                    labels[[r, c]] = 1;
                    image[[r, c]] = 1.0;
                }
            }
        }

        let props = regionprops_2d(&image, &labels).expect("circle should succeed");
        assert_eq!(props.len(), 1);

        let ecc: f64 = props[0].eccentricity;
        // A circle should have eccentricity very close to 0
        assert!(ecc < 0.2, "Circle eccentricity {} should be < 0.2", ecc);
    }

    #[test]
    fn test_regionprops_2d_elongated_eccentricity() {
        // Elongated region: eccentricity should be near 1
        let mut labels: Array2<usize> = Array2::zeros((3, 21));
        let mut image: Array2<f64> = Array2::zeros((3, 21));

        // A thin horizontal line
        for c in 0..21 {
            labels[[1, c]] = 1;
            image[[1, c]] = 1.0;
        }

        let props = regionprops_2d(&image, &labels).expect("elongated should succeed");
        assert_eq!(props.len(), 1);

        let ecc = props[0].eccentricity;
        assert!(ecc > 0.9, "Elongated eccentricity {} should be > 0.9", ecc);
    }

    #[test]
    fn test_regionprops_2d_solidity() {
        // A filled square should have solidity = 1.0
        let mut labels: Array2<usize> = Array2::zeros((10, 10));
        let mut image: Array2<f64> = Array2::zeros((10, 10));

        for r in 2..8 {
            for c in 2..8 {
                labels[[r, c]] = 1;
                image[[r, c]] = 1.0;
            }
        }

        let props = regionprops_2d(&image, &labels).expect("square should succeed");
        assert_eq!(props.len(), 1);

        let sol = props[0].solidity;
        assert_abs_diff_eq!(sol, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_regionprops_2d_extent() {
        // A filled rectangle: extent should be 1.0
        let mut labels: Array2<usize> = Array2::zeros((10, 10));
        let mut image: Array2<f64> = Array2::zeros((10, 10));

        for r in 2..5 {
            for c in 3..8 {
                labels[[r, c]] = 1;
                image[[r, c]] = 1.0;
            }
        }

        let props = regionprops_2d(&image, &labels).expect("rectangle should succeed");
        let ext = props[0].extent;
        assert_abs_diff_eq!(ext, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_regionprops_2d_intensity_stats() {
        let image: Array2<f64> =
            array![[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0],];
        let labels: Array2<usize> = Array2::from_elem((3, 3), 1);

        let props = regionprops_2d(&image, &labels).expect("intensity should succeed");
        assert_eq!(props.len(), 1);
        assert_abs_diff_eq!(props[0].mean_intensity, 50.0, epsilon = 1e-10);
        assert_abs_diff_eq!(props[0].min_intensity, 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(props[0].max_intensity, 90.0, epsilon = 1e-10);
    }

    #[test]
    fn test_regionprops_2d_equivalent_diameter() {
        // Region of area 100: equivalent diameter = sqrt(4*100/pi)
        let mut labels: Array2<usize> = Array2::zeros((12, 12));
        let image: Array2<f64> = Array2::from_elem((12, 12), 1.0);

        // Create a 10x10 block = area 100
        for r in 1..11 {
            for c in 1..11 {
                labels[[r, c]] = 1;
            }
        }

        let props = regionprops_2d(&image, &labels).expect("eq diam should succeed");
        let expected_diam = (4.0 * 100.0 / std::f64::consts::PI).sqrt();
        assert_abs_diff_eq!(props[0].equivalent_diameter, expected_diam, epsilon = 0.01);
    }

    #[test]
    fn test_regionprops_2d_multiple_regions() {
        let image: Array2<f64> = Array2::from_elem((6, 6), 1.0);
        let labels = array![
            [1, 1, 0, 2, 2, 2],
            [1, 1, 0, 2, 2, 2],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 3, 3, 0, 0],
            [0, 0, 3, 3, 0, 0],
            [0, 0, 3, 3, 0, 0],
        ];

        let props = regionprops_2d(&image, &labels).expect("multiple regions should succeed");
        assert_eq!(props.len(), 3);

        // Check sorted by label
        assert_eq!(props[0].label, 1);
        assert_eq!(props[1].label, 2);
        assert_eq!(props[2].label, 3);

        assert_eq!(props[0].area, 4);
        assert_eq!(props[1].area, 6);
        assert_eq!(props[2].area, 6);
    }

    #[test]
    fn test_convex_hull_area_triangle() {
        // Triangle: (0,0), (0,4), (3,0) -> area = 0.5 * 4 * 3 = 6
        let coords = vec![(0, 0), (0, 4), (3, 0)];
        let area = compute_convex_hull_area(&coords);
        assert_abs_diff_eq!(area, 6.0, epsilon = 0.01);
    }

    #[test]
    fn test_convex_hull_area_square() {
        // 3x3 block of pixels
        let mut coords = Vec::new();
        for r in 0..3 {
            for c in 0..3 {
                coords.push((r, c));
            }
        }
        let area = compute_convex_hull_area(&coords);
        // Convex hull is a 2x2 square -> area = 4
        assert_abs_diff_eq!(area, 4.0, epsilon = 0.01);
    }
}
