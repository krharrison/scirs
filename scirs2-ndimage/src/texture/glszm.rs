//! Gray-Level Size Zone Matrix (GLSZM) Computation and Feature Extraction
//!
//! The GLSZM characterizes texture by counting connected regions (zones) of
//! pixels sharing the same gray level and recording their sizes. The matrix
//! `S[i, j]` gives the number of zones with gray level `i` and zone size
//! `j + 1` (number of pixels).
//!
//! Connected component labeling is used to identify zones, with configurable
//! 4-connectivity or 8-connectivity.
//!
//! # Features
//!
//! - Small Zone Emphasis (SZE)
//! - Large Zone Emphasis (LZE)
//! - Gray-Level Non-Uniformity (GLN)
//! - Zone Size Non-Uniformity (ZSN)
//! - Zone Percentage (ZP)
//! - Low Gray-Level Zone Emphasis (LGZE)
//! - High Gray-Level Zone Emphasis (HGZE)
//!
//! # References
//!
//! - Thibault, G. et al. (2009). "Texture Indexes and Gray Level Size Zone
//!   Matrix. Application to Cell Nuclei Classification." Pattern Recognition
//!   and Information Processing.

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::Array2;

/// Connectivity type for connected component labeling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlszmConnectivity {
    /// 4-connectivity: only horizontal and vertical neighbors
    Four,
    /// 8-connectivity: horizontal, vertical, and diagonal neighbors
    Eight,
}

/// Features extracted from a GLSZM.
#[derive(Debug, Clone)]
pub struct GlszmFeatures {
    /// Small Zone Emphasis: emphasizes small zones.
    /// `SZE = (1/n_zones) * sum_j sum_i S(i,j) / (j+1)^2`
    pub small_zone_emphasis: f64,
    /// Large Zone Emphasis: emphasizes large zones.
    /// `LZE = (1/n_zones) * sum_j sum_i S(i,j) * (j+1)^2`
    pub large_zone_emphasis: f64,
    /// Gray-Level Non-Uniformity: variation of gray-level distribution.
    /// `GLN = (1/n_zones) * sum_i (sum_j S(i,j))^2`
    pub gray_level_non_uniformity: f64,
    /// Zone Size Non-Uniformity: variation of zone size distribution.
    /// `ZSN = (1/n_zones) * sum_j (sum_i S(i,j))^2`
    pub zone_size_non_uniformity: f64,
    /// Zone Percentage: ratio of total zones to total pixels.
    /// `ZP = n_zones / n_pixels`
    pub zone_percentage: f64,
    /// Low Gray-Level Zone Emphasis: emphasizes zones at low gray levels.
    /// `LGZE = (1/n_zones) * sum_j sum_i S(i,j) / (i+1)^2`
    pub low_gray_level_zone_emphasis: f64,
    /// High Gray-Level Zone Emphasis: emphasizes zones at high gray levels.
    /// `HGZE = (1/n_zones) * sum_j sum_i S(i,j) * (i+1)^2`
    pub high_gray_level_zone_emphasis: f64,
}

/// Combined result from GLSZM computation.
#[derive(Debug, Clone)]
pub struct GlszmResult {
    /// The GLSZM of shape `(n_levels, max_zone_size)`.
    pub matrix: Array2<f64>,
    /// Extracted features.
    pub features: GlszmFeatures,
}

/// Compute the Gray-Level Size Zone Matrix from an 8-bit image.
///
/// Uses connected component labeling to find zones of same gray level
/// and records their sizes.
///
/// # Parameters
/// - `image` - 8-bit grayscale image with values in `[0, n_levels-1]`
/// - `n_levels` - number of gray levels (>= 2)
/// - `connectivity` - 4 or 8 connectivity for zone identification
///
/// # Returns
/// `Array2<f64>` of shape `(n_levels, max_zone_size)`.
///
/// # Errors
/// Returns error if `n_levels < 2` or image is empty.
pub fn compute_glszm(
    image: &Array2<u8>,
    n_levels: usize,
    connectivity: GlszmConnectivity,
) -> NdimageResult<Array2<f64>> {
    if n_levels < 2 {
        return Err(NdimageError::InvalidInput(
            "n_levels must be at least 2".into(),
        ));
    }
    let (rows, cols) = image.dim();
    if rows == 0 || cols == 0 {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }

    // Find all connected zones and their sizes using union-find
    let zones = find_zones(image, rows, cols, n_levels, connectivity);

    if zones.is_empty() {
        return Ok(Array2::zeros((n_levels, 1)));
    }

    let max_size = zones.iter().map(|&(_, s)| s).max().unwrap_or(1);

    let mut matrix = Array2::<f64>::zeros((n_levels, max_size));
    for &(g, s) in &zones {
        matrix[[g, s - 1]] += 1.0;
    }

    Ok(matrix)
}

/// Extract GLSZM features from a pre-computed matrix.
///
/// # Parameters
/// - `glszm` - Size zone matrix of shape `(n_levels, max_zone_size)`
/// - `n_pixels` - total number of pixels in the image
///
/// # Errors
/// Returns error if the matrix is empty or `n_pixels == 0`.
pub fn glszm_features(glszm: &Array2<f64>, n_pixels: usize) -> NdimageResult<GlszmFeatures> {
    let (n_levels, max_size) = glszm.dim();
    if n_levels == 0 || max_size == 0 {
        return Err(NdimageError::InvalidInput("GLSZM must be non-empty".into()));
    }
    if n_pixels == 0 {
        return Err(NdimageError::InvalidInput("n_pixels must be > 0".into()));
    }

    let n_zones: f64 = glszm.iter().sum();
    if n_zones < 1e-15 {
        return Ok(GlszmFeatures {
            small_zone_emphasis: 0.0,
            large_zone_emphasis: 0.0,
            gray_level_non_uniformity: 0.0,
            zone_size_non_uniformity: 0.0,
            zone_percentage: 0.0,
            low_gray_level_zone_emphasis: 0.0,
            high_gray_level_zone_emphasis: 0.0,
        });
    }

    let mut sze = 0.0f64;
    let mut lze = 0.0f64;
    let mut lgze = 0.0f64;
    let mut hgze = 0.0f64;

    for i in 0..n_levels {
        for j in 0..max_size {
            let s = glszm[[i, j]];
            if s < 1e-15 {
                continue;
            }
            let zone_size = (j + 1) as f64;
            let gray = (i + 1) as f64;

            sze += s / (zone_size * zone_size);
            lze += s * zone_size * zone_size;
            lgze += s / (gray * gray);
            hgze += s * gray * gray;
        }
    }

    // Gray-Level Non-Uniformity
    let mut gln = 0.0f64;
    for i in 0..n_levels {
        let row_sum: f64 = (0..max_size).map(|j| glszm[[i, j]]).sum();
        gln += row_sum * row_sum;
    }

    // Zone Size Non-Uniformity
    let mut zsn = 0.0f64;
    for j in 0..max_size {
        let col_sum: f64 = (0..n_levels).map(|i| glszm[[i, j]]).sum();
        zsn += col_sum * col_sum;
    }

    Ok(GlszmFeatures {
        small_zone_emphasis: sze / n_zones,
        large_zone_emphasis: lze / n_zones,
        gray_level_non_uniformity: gln / n_zones,
        zone_size_non_uniformity: zsn / n_zones,
        zone_percentage: n_zones / n_pixels as f64,
        low_gray_level_zone_emphasis: lgze / n_zones,
        high_gray_level_zone_emphasis: hgze / n_zones,
    })
}

// ---------------------------------------------------------------------------
// Internal: connected component labeling via union-find
// ---------------------------------------------------------------------------

/// Union-Find data structure for connected component labeling.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
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

/// Find connected zones and return `(gray_level, zone_size)` pairs.
fn find_zones(
    image: &Array2<u8>,
    rows: usize,
    cols: usize,
    n_levels: usize,
    connectivity: GlszmConnectivity,
) -> Vec<(usize, usize)> {
    let n = rows * cols;
    let mut uf = UnionFind::new(n);

    let neighbors_4: [(i64, i64); 2] = [(-1, 0), (0, -1)];
    let neighbors_8: [(i64, i64); 4] = [(-1, -1), (-1, 0), (-1, 1), (0, -1)];

    for r in 0..rows {
        for c in 0..cols {
            let g = (image[[r, c]] as usize).min(n_levels - 1);
            let idx = r * cols + c;

            let neighbor_offsets: &[(i64, i64)] = match connectivity {
                GlszmConnectivity::Four => &neighbors_4,
                GlszmConnectivity::Eight => &neighbors_8,
            };

            for &(dr, dc) in neighbor_offsets {
                let nr = r as i64 + dr;
                let nc = c as i64 + dc;
                if nr < 0 || nr >= rows as i64 || nc < 0 || nc >= cols as i64 {
                    continue;
                }
                let nr = nr as usize;
                let nc = nc as usize;
                let ng = (image[[nr, nc]] as usize).min(n_levels - 1);
                if ng == g {
                    let nidx = nr * cols + nc;
                    uf.union(idx, nidx);
                }
            }
        }
    }

    // Count zone sizes grouped by (root, gray_level)
    let mut zone_map: std::collections::HashMap<usize, (usize, usize)> =
        std::collections::HashMap::new();

    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            let root = uf.find(idx);
            let g = (image[[r, c]] as usize).min(n_levels - 1);
            let entry = zone_map.entry(root).or_insert((g, 0));
            entry.1 += 1;
        }
    }

    zone_map.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_glszm_single_zone() {
        // Uniform image: one zone spanning the entire image
        let img = Array2::from_elem((4, 4), 1u8);
        let glszm = compute_glszm(&img, 4, GlszmConnectivity::Four).expect("glszm");
        // One zone at gray=1, size=16
        assert_eq!(glszm.dim(), (4, 16));
        assert!((glszm[[1, 15]] - 1.0).abs() < 1e-10);
        // No other zones
        let total: f64 = glszm.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_glszm_checkerboard_4conn() {
        // Checkerboard: with 4-connectivity, each pixel is its own zone
        let img = Array2::from_shape_fn((4, 4), |(i, j)| ((i + j) % 2) as u8);
        let glszm = compute_glszm(&img, 2, GlszmConnectivity::Four).expect("glszm");
        // All zones are size 1
        assert_eq!(glszm.dim(), (2, 1));
        // 8 pixels of each color
        assert!((glszm[[0, 0]] - 8.0).abs() < 1e-10);
        assert!((glszm[[1, 0]] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_glszm_known_image() {
        // 3x3 image:
        // 0 0 1
        // 0 0 1
        // 1 1 1
        let img = Array2::from_shape_vec((3, 3), vec![0, 0, 1, 0, 0, 1, 1, 1, 1]).expect("ok");
        let glszm = compute_glszm(&img, 2, GlszmConnectivity::Four).expect("glszm");

        // With 4-connectivity:
        // Zone at gray=0: {(0,0),(0,1),(1,0),(1,1)} -> size 4
        // Zone at gray=1: {(0,2),(1,2),(2,0),(2,1),(2,2)} -> size 5
        let total: f64 = glszm.iter().sum();
        assert!(
            (total - 2.0).abs() < 1e-10,
            "Should have 2 zones, got {}",
            total
        );
    }

    #[test]
    fn test_glszm_8conn_vs_4conn() {
        // Diagonal connection matters for 8-conn
        // 1 0
        // 0 1
        let img = Array2::from_shape_vec((2, 2), vec![1, 0, 0, 1]).expect("ok");

        let glszm_4 = compute_glszm(&img, 2, GlszmConnectivity::Four).expect("4conn");
        let glszm_8 = compute_glszm(&img, 2, GlszmConnectivity::Eight).expect("8conn");

        let total_4: f64 = glszm_4.iter().sum();
        let total_8: f64 = glszm_8.iter().sum();

        // 4-conn: 4 separate zones (each pixel isolated)
        assert!((total_4 - 4.0).abs() < 1e-10, "4-conn should have 4 zones");
        // 8-conn: 2 zones (diagonal pixels connected)
        assert!((total_8 - 2.0).abs() < 1e-10, "8-conn should have 2 zones");
    }

    #[test]
    fn test_glszm_features_single_zone() {
        // Single zone of size N
        let img = Array2::from_elem((4, 4), 0u8);
        let glszm = compute_glszm(&img, 2, GlszmConnectivity::Four).expect("glszm");
        let feats = glszm_features(&glszm, 16).expect("features");

        // One zone => ZP = 1/16
        assert!(
            (feats.zone_percentage - 1.0 / 16.0).abs() < 1e-10,
            "ZP should be 1/16, got {}",
            feats.zone_percentage
        );
        // SZE = 1/16^2 (small when zone is large)
        assert!(feats.small_zone_emphasis < 0.01);
        // LZE = 16^2 = 256
        assert!(
            (feats.large_zone_emphasis - 256.0).abs() < 1e-10,
            "LZE should be 256, got {}",
            feats.large_zone_emphasis
        );
    }

    #[test]
    fn test_glszm_features_bounds() {
        let img = Array2::from_shape_fn((8, 8), |(i, j)| ((i * 8 + j) % 4) as u8);
        let glszm = compute_glszm(&img, 4, GlszmConnectivity::Four).expect("glszm");
        let feats = glszm_features(&glszm, 64).expect("features");

        assert!(feats.small_zone_emphasis >= 0.0);
        assert!(feats.large_zone_emphasis >= 0.0);
        assert!(feats.gray_level_non_uniformity >= 0.0);
        assert!(feats.zone_size_non_uniformity >= 0.0);
        assert!(feats.zone_percentage > 0.0 && feats.zone_percentage <= 1.0);
        assert!(feats.low_gray_level_zone_emphasis >= 0.0);
        assert!(feats.high_gray_level_zone_emphasis >= 0.0);
    }

    #[test]
    fn test_glszm_errors() {
        let img = Array2::from_elem((4, 4), 0u8);
        assert!(compute_glszm(&img, 1, GlszmConnectivity::Four).is_err());

        let empty = Array2::<u8>::zeros((0, 0));
        assert!(compute_glszm(&empty, 2, GlszmConnectivity::Four).is_err());

        let glszm = Array2::<f64>::zeros((0, 0));
        assert!(glszm_features(&glszm, 16).is_err());
    }
}
