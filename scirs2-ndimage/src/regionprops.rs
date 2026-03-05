//! Region properties analysis for labeled images
//!
//! This module provides comprehensive geometric, statistical, and shape
//! measurements for labeled regions in 2D images. It is modeled after
//! scikit-image's `regionprops` and scipy.ndimage's `measurements`.
//!
//! # Features
//!
//! - **Connected component labeling**: 4-connectivity and 8-connectivity
//! - **Geometric properties**: area, perimeter, centroid, bounding box
//! - **Shape descriptors**: eccentricity, orientation, solidity, extent,
//!   major/minor axis lengths, Euler number
//! - **Hu moments**: 7 rotation/translation/scale-invariant moment descriptors
//! - **Region filtering**: filter regions by any computed property

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign};
use std::collections::HashMap;
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Connected component labeling
// ---------------------------------------------------------------------------

/// Union-Find data structure for efficient connected component labeling
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }
}

/// Connectivity mode for connected component labeling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Connectivity {
    /// 4-connectivity: only face-adjacent (up, down, left, right)
    Conn4,
    /// 8-connectivity: face-adjacent plus diagonals
    Conn8,
}

impl Default for Connectivity {
    fn default() -> Self {
        Connectivity::Conn4
    }
}

/// Label connected components in a binary image
///
/// Uses a two-pass union-find algorithm optimized for 2D images.
///
/// # Arguments
///
/// * `binary` - Input binary image (`true` = foreground)
/// * `connectivity` - `Conn4` (4-connected) or `Conn8` (8-connected)
///
/// # Returns
///
/// * Labeled image (0 = background, 1..n = component labels)
/// * Number of components
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::regionprops::{label_components, Connectivity};
///
/// let binary = array![
///     [true,  true,  false, false],
///     [true,  true,  false, false],
///     [false, false, true,  true],
///     [false, false, true,  true],
/// ];
///
/// let (labeled, n) = label_components(&binary, Connectivity::Conn4)
///     .expect("should succeed");
/// assert_eq!(n, 2);
/// ```
pub fn label_components(
    binary: &Array2<bool>,
    connectivity: Connectivity,
) -> NdimageResult<(Array2<usize>, usize)> {
    let rows = binary.nrows();
    let cols = binary.ncols();

    if rows == 0 || cols == 0 {
        return Ok((Array2::zeros((rows, cols)), 0));
    }

    let total = rows * cols;
    let mut uf = UnionFind::new(total);
    let use_diag = connectivity == Connectivity::Conn8;

    // First pass
    for r in 0..rows {
        for c in 0..cols {
            if !binary[[r, c]] {
                continue;
            }
            let idx = r * cols + c;

            if r > 0 && binary[[r - 1, c]] {
                uf.union(idx, (r - 1) * cols + c);
            }
            if c > 0 && binary[[r, c - 1]] {
                uf.union(idx, r * cols + (c - 1));
            }
            if use_diag {
                if r > 0 && c > 0 && binary[[r - 1, c - 1]] {
                    uf.union(idx, (r - 1) * cols + (c - 1));
                }
                if r > 0 && c + 1 < cols && binary[[r - 1, c + 1]] {
                    uf.union(idx, (r - 1) * cols + (c + 1));
                }
            }
        }
    }

    // Second pass: assign sequential labels
    let mut root_to_label: HashMap<usize, usize> = HashMap::new();
    let mut next_label = 1usize;
    let mut output = Array2::zeros((rows, cols));

    for r in 0..rows {
        for c in 0..cols {
            if !binary[[r, c]] {
                continue;
            }
            let root = uf.find(r * cols + c);
            let lbl = match root_to_label.get(&root) {
                Some(&l) => l,
                None => {
                    let l = next_label;
                    root_to_label.insert(root, l);
                    next_label += 1;
                    l
                }
            };
            output[[r, c]] = lbl;
        }
    }

    Ok((output, next_label - 1))
}

// ---------------------------------------------------------------------------
// Region properties
// ---------------------------------------------------------------------------

/// Comprehensive properties of a single labeled region
#[derive(Debug, Clone)]
pub struct RegionProps<T: Float> {
    /// Label value
    pub label: usize,
    /// Number of pixels
    pub area: usize,
    /// Geometric centroid (row, col)
    pub centroid: (T, T),
    /// Intensity-weighted centroid (row, col)
    pub weighted_centroid: (T, T),
    /// Bounding box (min_row, min_col, max_row_exclusive, max_col_exclusive)
    pub bounding_box: (usize, usize, usize, usize),
    /// Perimeter length (boundary pixel count, 4-connected)
    pub perimeter: T,
    /// Eccentricity of equivalent ellipse (0 = circle, approaching 1 = line)
    pub eccentricity: T,
    /// Orientation angle in radians of major axis
    pub orientation: T,
    /// Diameter of a circle with the same area
    pub equivalent_diameter: T,
    /// Length of the major axis of the equivalent ellipse
    pub major_axis_length: T,
    /// Length of the minor axis of the equivalent ellipse
    pub minor_axis_length: T,
    /// Euler number: #objects - #holes (4-connected topology)
    pub euler_number: i32,
    /// Solidity: area / convex_hull_area
    pub solidity: T,
    /// Extent: area / bounding_box_area
    pub extent: T,
    /// Mean intensity
    pub mean_intensity: T,
    /// Minimum intensity
    pub min_intensity: T,
    /// Maximum intensity
    pub max_intensity: T,
    /// 7 Hu invariant moments (translation, scale, rotation invariant)
    pub hu_moments: [T; 7],
    /// Raw central moments (mu_00, mu_20, mu_11, mu_02, mu_30, mu_21, mu_12, mu_03)
    pub central_moments: [T; 8],
    /// Normalized central moments (nu_20, nu_11, nu_02, nu_30, nu_21, nu_12, nu_03)
    pub normalized_moments: [T; 7],
}

/// Internal accumulator for single-pass pixel gathering
struct PixelAccumulator {
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
    coords: Vec<(usize, usize)>,
    intensities: Vec<f64>,
}

impl PixelAccumulator {
    fn new() -> Self {
        Self {
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

    fn add(&mut self, r: usize, c: usize, intensity: f64) {
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

/// Compute the Euler number of a binary region using the bit-quad method
///
/// For a 4-connected foreground, the Euler number is calculated by counting
/// specific 2x2 bit patterns (quads) in the binary image.
fn compute_euler_number(coords: &[(usize, usize)], rows: usize, cols: usize) -> i32 {
    // Build a binary image for this region
    let mut img = Array2::<bool>::from_elem((rows, cols), false);
    for &(r, c) in coords {
        img[[r, c]] = true;
    }

    // Count quad patterns for 4-connected Euler number
    // Q1: exactly one pixel set in 2x2 block
    // Q3: exactly three pixels set in 2x2 block
    // Qd: diagonal pair set (two diagonally opposite pixels)
    // Euler = (Q1 - Q3 + 2*Qd) / 4
    let mut q1 = 0i32;
    let mut q3 = 0i32;
    let mut qd = 0i32;

    for r in 0..rows.saturating_sub(1) + 1 {
        for c in 0..cols.saturating_sub(1) + 1 {
            // Count set pixels in the 2x2 block at (r, c)
            let mut count = 0u32;
            let mut pattern = 0u8;

            let p00 = if r < rows && c < cols {
                img[[r, c]]
            } else {
                false
            };
            let p01 = if r < rows && c + 1 < cols {
                img[[r, c + 1]]
            } else {
                false
            };
            let p10 = if r + 1 < rows && c < cols {
                img[[r + 1, c]]
            } else {
                false
            };
            let p11 = if r + 1 < rows && c + 1 < cols {
                img[[r + 1, c + 1]]
            } else {
                false
            };

            if p00 {
                count += 1;
                pattern |= 1;
            }
            if p01 {
                count += 1;
                pattern |= 2;
            }
            if p10 {
                count += 1;
                pattern |= 4;
            }
            if p11 {
                count += 1;
                pattern |= 8;
            }

            if count == 1 {
                q1 += 1;
            } else if count == 3 {
                q3 += 1;
            } else if count == 2 {
                // Check diagonal patterns: 0101 (bits 0,3) or 1010 (bits 1,2)
                if pattern == 0b1001 || pattern == 0b0110 {
                    qd += 1;
                }
            }
        }
    }

    (q1 - q3 + 2 * qd) / 4
}

/// Compute Hu's 7 invariant moments from normalized central moments
///
/// These moments are invariant to translation, scale, and rotation.
/// Reference: Hu, M.K. (1962). "Visual pattern recognition by moment invariants"
fn compute_hu_moments(
    nu_20: f64,
    nu_11: f64,
    nu_02: f64,
    nu_30: f64,
    nu_21: f64,
    nu_12: f64,
    nu_03: f64,
) -> [f64; 7] {
    // H1
    let h1 = nu_20 + nu_02;

    // H2
    let h2 = (nu_20 - nu_02).powi(2) + 4.0 * nu_11.powi(2);

    // H3
    let h3 = (nu_30 - 3.0 * nu_12).powi(2) + (3.0 * nu_21 - nu_03).powi(2);

    // H4
    let h4 = (nu_30 + nu_12).powi(2) + (nu_21 + nu_03).powi(2);

    // H5
    let h5 = (nu_30 - 3.0 * nu_12)
        * (nu_30 + nu_12)
        * ((nu_30 + nu_12).powi(2) - 3.0 * (nu_21 + nu_03).powi(2))
        + (3.0 * nu_21 - nu_03)
            * (nu_21 + nu_03)
            * (3.0 * (nu_30 + nu_12).powi(2) - (nu_21 + nu_03).powi(2));

    // H6
    let h6 = (nu_20 - nu_02) * ((nu_30 + nu_12).powi(2) - (nu_21 + nu_03).powi(2))
        + 4.0 * nu_11 * (nu_30 + nu_12) * (nu_21 + nu_03);

    // H7 (skew invariant, sign changes for mirrored images)
    let h7 = (3.0 * nu_21 - nu_03)
        * (nu_30 + nu_12)
        * ((nu_30 + nu_12).powi(2) - 3.0 * (nu_21 + nu_03).powi(2))
        - (nu_30 - 3.0 * nu_12)
            * (nu_21 + nu_03)
            * (3.0 * (nu_30 + nu_12).powi(2) - (nu_21 + nu_03).powi(2));

    [h1, h2, h3, h4, h5, h6, h7]
}

/// Convex hull area via Andrew's monotone chain algorithm + shoelace formula
fn convex_hull_area(coords: &[(usize, usize)]) -> f64 {
    if coords.len() < 3 {
        return coords.len() as f64;
    }

    let mut pts: Vec<(f64, f64)> = coords.iter().map(|&(r, c)| (c as f64, r as f64)).collect();
    pts.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    pts.dedup();

    if pts.len() < 3 {
        return pts.len() as f64;
    }

    // Lower hull
    let mut lower: Vec<(f64, f64)> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 {
            let n = lower.len();
            if cross_2d(lower[n - 2], lower[n - 1], p) <= 0.0 {
                lower.pop();
            } else {
                break;
            }
        }
        lower.push(p);
    }

    // Upper hull
    let mut upper: Vec<(f64, f64)> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 {
            let n = upper.len();
            if cross_2d(upper[n - 2], upper[n - 1], p) <= 0.0 {
                upper.pop();
            } else {
                break;
            }
        }
        upper.push(p);
    }

    lower.pop();
    upper.pop();

    let hull: Vec<(f64, f64)> = lower.into_iter().chain(upper).collect();

    if hull.len() < 3 {
        return hull.len() as f64;
    }

    // Shoelace formula
    let n = hull.len();
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += hull[i].0 * hull[j].1;
        area -= hull[j].0 * hull[i].1;
    }
    area.abs() / 2.0
}

#[inline]
fn cross_2d(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

/// Compute region properties for all labeled regions in a 2D image
///
/// Analyzes each non-zero label and computes a comprehensive set of geometric,
/// statistical, and moment-based properties.
///
/// # Arguments
///
/// * `image`  - Intensity image (grayscale)
/// * `labels` - Label image (0 = background, positive integers = regions)
///
/// # Returns
///
/// Vector of `RegionProps` sorted by label value.
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::regionprops::region_properties;
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
/// let props = region_properties(&image, &labels).expect("should succeed");
/// assert_eq!(props.len(), 3);
/// assert_eq!(props[0].area, 4);
/// ```
pub fn region_properties<T>(
    image: &Array2<T>,
    labels: &Array2<usize>,
) -> NdimageResult<Vec<RegionProps<T>>>
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

    // Accumulate pixel data
    let mut accumulators: HashMap<usize, PixelAccumulator> = HashMap::new();

    for r in 0..rows {
        for c in 0..cols {
            let lbl = labels[[r, c]];
            if lbl == 0 {
                continue;
            }
            let intensity = image[[r, c]].to_f64().unwrap_or(0.0);
            accumulators
                .entry(lbl)
                .or_insert_with(PixelAccumulator::new)
                .add(r, c, intensity);
        }
    }

    // Perimeter computation (4-connected boundary count)
    let mut perimeter_counts: HashMap<usize, usize> = HashMap::new();
    for r in 0..rows {
        for c in 0..cols {
            let lbl = labels[[r, c]];
            if lbl == 0 {
                continue;
            }
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
    let to_t = |v: f64| -> T { T::from_f64(v).unwrap_or(T::zero()) };

    let mut result = Vec::with_capacity(accumulators.len());

    for (&lbl, acc) in &accumulators {
        let area = acc.area;
        let area_f = area as f64;

        // Centroid
        let cr = acc.sum_r / area_f;
        let cc = acc.sum_c / area_f;

        // Weighted centroid
        let (wcr, wcc) = if acc.sum_intensity.abs() > 1e-15 {
            (
                acc.sum_r_weighted / acc.sum_intensity,
                acc.sum_c_weighted / acc.sum_intensity,
            )
        } else {
            (cr, cc)
        };

        // Central moments up to order 3
        let mut mu_00 = 0.0;
        let mut mu_20 = 0.0;
        let mut mu_11 = 0.0;
        let mut mu_02 = 0.0;
        let mut mu_30 = 0.0;
        let mut mu_21 = 0.0;
        let mut mu_12 = 0.0;
        let mut mu_03 = 0.0;

        for &(r, c) in &acc.coords {
            let dr = r as f64 - cr;
            let dc = c as f64 - cc;
            mu_00 += 1.0;
            mu_20 += dr * dr;
            mu_11 += dr * dc;
            mu_02 += dc * dc;
            mu_30 += dr * dr * dr;
            mu_21 += dr * dr * dc;
            mu_12 += dr * dc * dc;
            mu_03 += dc * dc * dc;
        }

        // Normalized central moments: nu_pq = mu_pq / mu_00^((p+q)/2 + 1)
        let gamma = |p: i32, q: i32| -> f64 {
            let exp = (p + q) as f64 / 2.0 + 1.0;
            if mu_00.abs() > 1e-15 {
                mu_00.powf(exp)
            } else {
                1.0
            }
        };

        let nu_20 = mu_20 / gamma(2, 0);
        let nu_11 = mu_11 / gamma(1, 1);
        let nu_02 = mu_02 / gamma(0, 2);
        let nu_30 = mu_30 / gamma(3, 0);
        let nu_21 = mu_21 / gamma(2, 1);
        let nu_12 = mu_12 / gamma(1, 2);
        let nu_03 = mu_03 / gamma(0, 3);

        // Hu moments
        let hu = compute_hu_moments(nu_20, nu_11, nu_02, nu_30, nu_21, nu_12, nu_03);

        // Orientation (angle of major axis)
        let orientation = 0.5 * (2.0 * mu_11).atan2(mu_20 - mu_02);

        // Eigenvalues of the 2x2 inertia tensor for ellipse fitting
        let mu_20_n = mu_20 / area_f;
        let mu_11_n = mu_11 / area_f;
        let mu_02_n = mu_02 / area_f;

        let trace = mu_20_n + mu_02_n;
        let det = mu_20_n * mu_02_n - mu_11_n * mu_11_n;
        let discriminant = (trace * trace - 4.0 * det).max(0.0);
        let sqrt_disc = discriminant.sqrt();
        let lambda1 = (trace + sqrt_disc) / 2.0;
        let lambda2 = (trace - sqrt_disc) / 2.0;

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
        let eq_diam = (4.0 * area_f / std::f64::consts::PI).sqrt();

        // Perimeter
        let perim = *perimeter_counts.get(&lbl).unwrap_or(&0) as f64;

        // Bounding box area
        let bbox_h = acc.max_row - acc.min_row + 1;
        let bbox_w = acc.max_col - acc.min_col + 1;
        let bbox_area = bbox_h * bbox_w;

        // Extent
        let extent = if bbox_area > 0 {
            area_f / bbox_area as f64
        } else {
            0.0
        };

        // Solidity
        let ch_area = convex_hull_area(&acc.coords);
        let solidity = if ch_area > 1e-15 {
            (area_f / ch_area).min(1.0)
        } else {
            1.0
        };

        // Euler number
        let euler = compute_euler_number(&acc.coords, rows, cols);

        // Mean intensity
        let mean_intensity = acc.sum_intensity / area_f;

        result.push(RegionProps {
            label: lbl,
            area,
            centroid: (to_t(cr), to_t(cc)),
            weighted_centroid: (to_t(wcr), to_t(wcc)),
            bounding_box: (acc.min_row, acc.min_col, acc.max_row + 1, acc.max_col + 1),
            perimeter: to_t(perim),
            eccentricity: to_t(eccentricity),
            orientation: to_t(orientation),
            equivalent_diameter: to_t(eq_diam),
            major_axis_length: to_t(major_axis),
            minor_axis_length: to_t(minor_axis),
            euler_number: euler,
            solidity: to_t(solidity),
            extent: to_t(extent),
            mean_intensity: to_t(mean_intensity),
            min_intensity: to_t(acc.min_intensity),
            max_intensity: to_t(acc.max_intensity),
            hu_moments: [
                to_t(hu[0]),
                to_t(hu[1]),
                to_t(hu[2]),
                to_t(hu[3]),
                to_t(hu[4]),
                to_t(hu[5]),
                to_t(hu[6]),
            ],
            central_moments: [
                to_t(mu_00),
                to_t(mu_20),
                to_t(mu_11),
                to_t(mu_02),
                to_t(mu_30),
                to_t(mu_21),
                to_t(mu_12),
                to_t(mu_03),
            ],
            normalized_moments: [
                to_t(nu_20),
                to_t(nu_11),
                to_t(nu_02),
                to_t(nu_30),
                to_t(nu_21),
                to_t(nu_12),
                to_t(nu_03),
            ],
        });
    }

    result.sort_by_key(|p| p.label);
    Ok(result)
}

// ---------------------------------------------------------------------------
// Region filtering
// ---------------------------------------------------------------------------

/// Predicate for filtering regions by property
pub enum RegionFilter<T: Float> {
    /// Keep regions whose area is in [min, max]
    AreaRange { min: usize, max: usize },
    /// Keep regions whose perimeter is in [min, max]
    PerimeterRange { min: T, max: T },
    /// Keep regions whose eccentricity is in [min, max]
    EccentricityRange { min: T, max: T },
    /// Keep regions whose solidity is in [min, max]
    SolidityRange { min: T, max: T },
    /// Keep regions whose mean intensity is in [min, max]
    IntensityRange { min: T, max: T },
    /// Keep regions whose extent is in [min, max]
    ExtentRange { min: T, max: T },
    /// Custom predicate
    Custom(Box<dyn Fn(&RegionProps<T>) -> bool>),
}

/// Filter regions by one or more property criteria
///
/// Returns a new labeled image with only the regions that satisfy all filters.
///
/// # Arguments
///
/// * `labels`  - Labeled image
/// * `props`   - Pre-computed region properties
/// * `filters` - List of filter predicates (all must be satisfied)
///
/// # Returns
///
/// New labeled image with only accepted regions (relabeled sequentially).
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::regionprops::{region_properties, filter_regions, RegionFilter};
///
/// let image: scirs2_core::ndarray::Array2<f64> = scirs2_core::ndarray::Array2::from_elem((6, 6), 1.0);
/// let labels = array![
///     [1, 1, 0, 2, 2, 2],
///     [1, 1, 0, 2, 2, 2],
///     [0, 0, 0, 0, 0, 0],
///     [0, 0, 3, 0, 0, 0],
///     [0, 0, 0, 0, 0, 0],
///     [0, 0, 0, 0, 0, 0],
/// ];
///
/// let props = region_properties(&image, &labels).expect("should succeed");
/// let filters = vec![RegionFilter::AreaRange { min: 3, max: 100 }];
/// let filtered = filter_regions(&labels, &props, &filters).expect("should succeed");
///
/// // Region 3 (area 1) is filtered out
/// let mut unique = std::collections::HashSet::new();
/// for &v in filtered.iter() { if v > 0 { unique.insert(v); } }
/// assert_eq!(unique.len(), 2);
/// ```
pub fn filter_regions<T>(
    labels: &Array2<usize>,
    props: &[RegionProps<T>],
    filters: &[RegionFilter<T>],
) -> NdimageResult<Array2<usize>>
where
    T: Float + FromPrimitive + NumAssign + Debug + Copy + 'static,
{
    let rows = labels.nrows();
    let cols = labels.ncols();

    // Determine which labels pass all filters
    let mut accepted: HashMap<usize, usize> = HashMap::new();
    let mut next_label = 1usize;

    for rp in props {
        let pass = filters.iter().all(|f| match f {
            RegionFilter::AreaRange { min, max } => rp.area >= *min && rp.area <= *max,
            RegionFilter::PerimeterRange { min, max } => {
                rp.perimeter >= *min && rp.perimeter <= *max
            }
            RegionFilter::EccentricityRange { min, max } => {
                rp.eccentricity >= *min && rp.eccentricity <= *max
            }
            RegionFilter::SolidityRange { min, max } => rp.solidity >= *min && rp.solidity <= *max,
            RegionFilter::IntensityRange { min, max } => {
                rp.mean_intensity >= *min && rp.mean_intensity <= *max
            }
            RegionFilter::ExtentRange { min, max } => rp.extent >= *min && rp.extent <= *max,
            RegionFilter::Custom(pred) => pred(rp),
        });

        if pass {
            accepted.insert(rp.label, next_label);
            next_label += 1;
        }
    }

    // Relabel
    let mut output = Array2::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let lbl = labels[[r, c]];
            if let Some(&new_lbl) = accepted.get(&lbl) {
                output[[r, c]] = new_lbl;
            }
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_label_components_4conn() {
        let binary = array![
            [true, true, false, false],
            [true, true, false, false],
            [false, false, true, true],
            [false, false, true, true],
        ];

        let (labeled, n) = label_components(&binary, Connectivity::Conn4).expect("should succeed");
        assert_eq!(n, 2);
        let l1 = labeled[[0, 0]];
        let l2 = labeled[[2, 2]];
        assert_ne!(l1, 0);
        assert_ne!(l2, 0);
        assert_ne!(l1, l2);
    }

    #[test]
    fn test_label_components_8conn() {
        let binary = array![
            [true, false, false],
            [false, true, false],
            [false, false, true],
        ];

        let (labeled, n) = label_components(&binary, Connectivity::Conn8).expect("should succeed");
        assert_eq!(n, 1);
        assert_eq!(labeled[[0, 0]], labeled[[1, 1]]);
        assert_eq!(labeled[[1, 1]], labeled[[2, 2]]);
    }

    #[test]
    fn test_label_components_4conn_diagonal() {
        let binary = array![
            [true, false, false],
            [false, true, false],
            [false, false, true],
        ];

        let (_, n) = label_components(&binary, Connectivity::Conn4).expect("should succeed");
        assert_eq!(n, 3);
    }

    #[test]
    fn test_label_components_empty() {
        let binary = Array2::from_elem((3, 3), false);
        let (labeled, n) = label_components(&binary, Connectivity::Conn4).expect("should succeed");
        assert_eq!(n, 0);
        for &v in labeled.iter() {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn test_region_properties_basic() {
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

        let props = region_properties(&image, &labels).expect("should succeed");
        assert_eq!(props.len(), 3);

        // Region 1
        assert_eq!(props[0].label, 1);
        assert_eq!(props[0].area, 4);
        assert_abs_diff_eq!(props[0].centroid.0, 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(props[0].centroid.1, 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(props[0].mean_intensity, 100.0, epsilon = 1e-10);
        assert_eq!(props[0].bounding_box, (0, 0, 2, 2));
    }

    #[test]
    fn test_region_properties_hu_moments() {
        // A symmetric region should have specific moment properties
        let mut labels = Array2::<usize>::zeros((11, 11));
        let image = Array2::<f64>::from_elem((11, 11), 1.0);

        // Create a symmetric cross pattern
        for i in 4..7 {
            for j in 0..11 {
                labels[[i, j]] = 1;
                labels[[j, i]] = 1;
            }
        }

        let props = region_properties(&image, &labels).expect("should succeed");
        assert_eq!(props.len(), 1);

        // For a symmetric shape, odd-order Hu moments should be near zero
        // H3 and H5 involve odd-order normalized moments
        assert!(
            props[0].hu_moments[2].abs() < 0.1,
            "H3 should be near zero for symmetric shape"
        );
    }

    #[test]
    fn test_region_properties_eccentricity_circle() {
        let mut labels = Array2::<usize>::zeros((21, 21));
        let image = Array2::<f64>::from_elem((21, 21), 1.0);

        for r in 0..21 {
            for c in 0..21 {
                let dr = r as f64 - 10.0;
                let dc = c as f64 - 10.0;
                if dr * dr + dc * dc <= 64.0 {
                    labels[[r, c]] = 1;
                }
            }
        }

        let props = region_properties(&image, &labels).expect("should succeed");
        assert_eq!(props.len(), 1);
        let ecc: f64 = props[0].eccentricity;
        assert!(ecc < 0.2, "Circle eccentricity {} should be < 0.2", ecc);
    }

    #[test]
    fn test_region_properties_eccentricity_line() {
        let mut labels = Array2::<usize>::zeros((3, 21));
        let image = Array2::<f64>::from_elem((3, 21), 1.0);

        for c in 0..21 {
            labels[[1, c]] = 1;
        }

        let props = region_properties(&image, &labels).expect("should succeed");
        let ecc = props[0].eccentricity;
        assert!(ecc > 0.9, "Line eccentricity {} should be > 0.9", ecc);
    }

    #[test]
    fn test_region_properties_euler_number() {
        // Simple filled rectangle: Euler number = 1
        let mut labels = Array2::<usize>::zeros((10, 10));
        let image = Array2::<f64>::from_elem((10, 10), 1.0);

        for r in 2..8 {
            for c in 2..8 {
                labels[[r, c]] = 1;
            }
        }

        let props = region_properties(&image, &labels).expect("should succeed");
        assert_eq!(
            props[0].euler_number, 1,
            "Filled rectangle Euler should be 1"
        );
    }

    #[test]
    fn test_region_properties_euler_number_with_hole() {
        // Rectangle with a hole: Euler number = 0 (1 object - 1 hole)
        let mut labels = Array2::<usize>::zeros((12, 12));
        let image = Array2::<f64>::from_elem((12, 12), 1.0);

        for r in 1..11 {
            for c in 1..11 {
                labels[[r, c]] = 1;
            }
        }
        // Remove interior to create hole
        for r in 4..8 {
            for c in 4..8 {
                labels[[r, c]] = 0;
            }
        }

        let props = region_properties(&image, &labels).expect("should succeed");
        assert_eq!(
            props[0].euler_number, 0,
            "Rectangle with hole Euler should be 0"
        );
    }

    #[test]
    fn test_region_properties_solidity() {
        // A filled square should have solidity near 1.0
        let mut labels = Array2::<usize>::zeros((10, 10));
        let image = Array2::<f64>::from_elem((10, 10), 1.0);

        for r in 2..8 {
            for c in 2..8 {
                labels[[r, c]] = 1;
            }
        }

        let props = region_properties(&image, &labels).expect("should succeed");
        let sol = props[0].solidity;
        assert_abs_diff_eq!(sol, 1.0, epsilon = 0.02);
    }

    #[test]
    fn test_region_properties_extent() {
        // A filled rectangle: extent = 1.0
        let mut labels = Array2::<usize>::zeros((10, 10));
        let image = Array2::<f64>::from_elem((10, 10), 1.0);

        for r in 2..5 {
            for c in 3..8 {
                labels[[r, c]] = 1;
            }
        }

        let props = region_properties(&image, &labels).expect("should succeed");
        let ext = props[0].extent;
        assert_abs_diff_eq!(ext, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_region_properties_shape_mismatch() {
        let image: Array2<f64> = Array2::zeros((3, 3));
        let labels: Array2<usize> = Array2::zeros((4, 4));
        let result = region_properties(&image, &labels);
        assert!(result.is_err());
    }

    #[test]
    fn test_region_properties_equivalent_diameter() {
        let mut labels = Array2::<usize>::zeros((12, 12));
        let image = Array2::<f64>::from_elem((12, 12), 1.0);

        for r in 1..11 {
            for c in 1..11 {
                labels[[r, c]] = 1;
            }
        }

        let props = region_properties(&image, &labels).expect("should succeed");
        let expected_diam = (4.0 * 100.0 / std::f64::consts::PI).sqrt();
        let actual_diam = props[0].equivalent_diameter;
        assert_abs_diff_eq!(actual_diam, expected_diam, epsilon = 0.01);
    }

    #[test]
    fn test_filter_regions_by_area() {
        let image = Array2::<f64>::from_elem((6, 6), 1.0);
        let labels = array![
            [1, 1, 0, 2, 2, 2],
            [1, 1, 0, 2, 2, 2],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ];

        let props = region_properties(&image, &labels).expect("should succeed");
        let filters = vec![RegionFilter::AreaRange { min: 3, max: 100 }];
        let filtered = filter_regions(&labels, &props, &filters).expect("should succeed");

        let mut unique = std::collections::HashSet::new();
        for &v in filtered.iter() {
            if v > 0 {
                unique.insert(v);
            }
        }
        // Region 3 (area 1) should be filtered out
        assert_eq!(unique.len(), 2);
    }

    #[test]
    fn test_filter_regions_by_eccentricity() {
        let image = Array2::<f64>::from_elem((20, 20), 1.0);
        let mut labels = Array2::<usize>::zeros((20, 20));

        // Region 1: roughly circular
        for r in 2..8 {
            for c in 2..8 {
                labels[[r, c]] = 1;
            }
        }

        // Region 2: elongated horizontal line
        for c in 0..20 {
            labels[[15, c]] = 2;
        }

        let props = region_properties(&image, &labels).expect("should succeed");
        let filters = vec![RegionFilter::EccentricityRange {
            min: 0.0f64,
            max: 0.5,
        }];
        let filtered = filter_regions(&labels, &props, &filters).expect("should succeed");

        // Only the compact region should remain
        let mut unique = std::collections::HashSet::new();
        for &v in filtered.iter() {
            if v > 0 {
                unique.insert(v);
            }
        }
        assert_eq!(unique.len(), 1, "Only the compact region should remain");
    }

    #[test]
    fn test_region_properties_intensity_stats() {
        let image: Array2<f64> = array![[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]];
        let labels = Array2::from_elem((3, 3), 1usize);

        let props = region_properties(&image, &labels).expect("should succeed");
        assert_abs_diff_eq!(props[0].mean_intensity, 50.0, epsilon = 1e-10);
        assert_abs_diff_eq!(props[0].min_intensity, 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(props[0].max_intensity, 90.0, epsilon = 1e-10);
    }

    #[test]
    fn test_label_components_zero_size() {
        let binary: Array2<bool> = Array2::from_elem((0, 0), false);
        let (_, n) = label_components(&binary, Connectivity::Conn4).expect("should succeed");
        assert_eq!(n, 0);
    }
}
