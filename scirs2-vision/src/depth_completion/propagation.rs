//! Propagation-based depth completion algorithms.
//!
//! - Nearest-neighbour fill
//! - Bilateral guided upsampling
//! - Inverse-distance-weighted interpolation

use scirs2_core::ndarray::Array2;

use crate::error::{Result, VisionError};

use super::types::{
    CompletionMethod, CompletionResult, DepthCompletionConfig, SparseDepthMap, SparseMeasurement,
};

// ─────────────────────────────────────────────────────────────────────────────
// Spatial index helper
// ─────────────────────────────────────────────────────────────────────────────

/// Simple 2-D spatial index backed by a grid of cells.
///
/// Each cell stores indices into the original measurement slice so that
/// neighbour lookups only scan nearby cells.
struct GridIndex {
    cell_size: usize,
    cols_cells: usize,
    rows_cells: usize,
    cells: Vec<Vec<usize>>,
}

impl GridIndex {
    /// Build a grid index over the given measurements.
    fn build(measurements: &[SparseMeasurement], height: usize, width: usize) -> Self {
        // Choose cell size so that each cell is roughly 16x16 pixels.
        let cell_size = 16usize;
        let cols_cells = width.div_ceil(cell_size);
        let rows_cells = height.div_ceil(cell_size);
        let num_cells = cols_cells * rows_cells;
        let mut cells: Vec<Vec<usize>> = vec![Vec::new(); num_cells];

        for (idx, m) in measurements.iter().enumerate() {
            let cr = m.row / cell_size;
            let cc = m.col / cell_size;
            let cell_idx = cr * cols_cells + cc;
            if cell_idx < num_cells {
                cells[cell_idx].push(idx);
            }
        }

        Self {
            cell_size,
            cols_cells,
            rows_cells,
            cells,
        }
    }

    /// Return indices of measurements in cells within `radius` cells of (row, col).
    fn neighbours_within(
        &self,
        row: usize,
        col: usize,
        radius_cells: usize,
    ) -> impl Iterator<Item = usize> + '_ {
        let cr = row / self.cell_size;
        let cc = col / self.cell_size;
        let r_lo = cr.saturating_sub(radius_cells);
        let r_hi = (cr + radius_cells + 1).min(self.rows_cells);
        let c_lo = cc.saturating_sub(radius_cells);
        let c_hi = (cc + radius_cells + 1).min(self.cols_cells);

        (r_lo..r_hi).flat_map(move |r| {
            (c_lo..c_hi).flat_map(move |c| {
                let cell_idx = r * self.cols_cells + c;
                self.cells[cell_idx].iter().copied()
            })
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Nearest-neighbour fill
// ─────────────────────────────────────────────────────────────────────────────

/// Fill every pixel in the image with the depth of the nearest sparse
/// measurement (by Euclidean distance).
///
/// Confidence is set to `1 / (1 + distance)`.
///
/// # Errors
/// Returns an error if `sparse` has no measurements.
pub fn nearest_neighbor_fill(sparse: &SparseDepthMap) -> Result<CompletionResult> {
    sparse.validate_non_empty()?;

    let h = sparse.height;
    let w = sparse.width;
    let mut dense = Array2::zeros((h, w));
    let mut conf = Array2::zeros((h, w));

    let index = GridIndex::build(&sparse.measurements, h, w);

    // For each pixel find the nearest measurement.
    for row in 0..h {
        for col in 0..w {
            let mut best_dist_sq = f64::MAX;
            let mut best_depth = 0.0;

            // Start with a small search radius and widen if needed.
            let max_radius = sparse.height.max(sparse.width).div_ceil(index.cell_size);
            let mut radius = 1usize;
            while radius <= max_radius {
                for idx in index.neighbours_within(row, col, radius) {
                    let m = &sparse.measurements[idx];
                    let dr = row as f64 - m.row as f64;
                    let dc = col as f64 - m.col as f64;
                    let d2 = dr * dr + dc * dc;
                    if d2 < best_dist_sq {
                        best_dist_sq = d2;
                        best_depth = m.depth;
                    }
                }
                // If we found something within current radius boundary, we are done.
                let boundary = (radius * index.cell_size) as f64;
                if best_dist_sq <= boundary * boundary {
                    break;
                }
                radius += 1;
            }

            dense[[row, col]] = best_depth;
            let dist = best_dist_sq.sqrt();
            conf[[row, col]] = 1.0 / (1.0 + dist);
        }
    }

    Ok(CompletionResult {
        dense_depth: dense,
        confidence_map: conf,
        method_used: CompletionMethod::NearestNeighbor,
        iterations: 0,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Bilateral guided upsampling
// ─────────────────────────────────────────────────────────────────────────────

/// Edge-aware depth upsampling guided by an RGB intensity image.
///
/// For each empty pixel, a weighted average of nearby sparse measurements is
/// computed.  Weights combine spatial proximity and intensity similarity to the
/// guide image, preserving edges in the guide.
///
/// # Arguments
/// * `sparse`    - sparse depth measurements
/// * `rgb_guide` - H x W intensity guide (e.g. grayscale, or luminance channel)
/// * `config`    - algorithm parameters (`sigma_spatial`, `sigma_intensity`)
///
/// # Errors
/// Returns an error if the guide dimensions do not match the sparse map or if
/// no measurements exist.
pub fn bilateral_upsample(
    sparse: &SparseDepthMap,
    rgb_guide: &Array2<f64>,
    config: &DepthCompletionConfig,
) -> Result<CompletionResult> {
    sparse.validate_non_empty()?;

    let h = sparse.height;
    let w = sparse.width;
    let (gh, gw) = rgb_guide.dim();
    if gh != h || gw != w {
        return Err(VisionError::DimensionMismatch(format!(
            "guide image is {gh}x{gw} but sparse map is {h}x{w}"
        )));
    }

    let sigma_s = config.sigma_spatial;
    let sigma_i = config.sigma_intensity;
    let two_sigma_s_sq = 2.0 * sigma_s * sigma_s;
    let two_sigma_i_sq = 2.0 * sigma_i * sigma_i;

    // Search radius in pixels (3 sigma).
    let search_radius = (3.0 * sigma_s).ceil() as usize;
    let cell_radius = search_radius.div_ceil(16); // match cell_size=16

    let index = GridIndex::build(&sparse.measurements, h, w);
    let mut dense = Array2::zeros((h, w));
    let mut conf = Array2::zeros((h, w));

    for row in 0..h {
        for col in 0..w {
            let intensity = rgb_guide[[row, col]];
            let mut weight_sum = 0.0;
            let mut depth_sum = 0.0;

            for idx in index.neighbours_within(row, col, cell_radius) {
                let m = &sparse.measurements[idx];
                let dr = row as f64 - m.row as f64;
                let dc = col as f64 - m.col as f64;
                let spatial_dist_sq = dr * dr + dc * dc;

                if spatial_dist_sq > (search_radius as f64 * search_radius as f64) {
                    continue;
                }

                let m_intensity = rgb_guide[[m.row, m.col]];
                let intensity_diff = intensity - m_intensity;
                let intensity_dist_sq = intensity_diff * intensity_diff;

                let w_s = (-spatial_dist_sq / two_sigma_s_sq).exp();
                let w_i = (-intensity_dist_sq / two_sigma_i_sq).exp();
                let weight = w_s * w_i * m.confidence;

                weight_sum += weight;
                depth_sum += weight * m.depth;
            }

            if weight_sum > 1e-12 {
                dense[[row, col]] = depth_sum / weight_sum;
                conf[[row, col]] = weight_sum.min(1.0);
            }
        }
    }

    Ok(CompletionResult {
        dense_depth: dense,
        confidence_map: conf,
        method_used: CompletionMethod::BilateralGuided,
        iterations: 0,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Inverse-distance-weighted interpolation
// ─────────────────────────────────────────────────────────────────────────────

/// Inverse-distance-weighted (IDW) depth interpolation.
///
/// For each pixel:  `d(x) = sum(w_i * d_i) / sum(w_i)`  where `w_i = 1 / dist^power`.
///
/// # Arguments
/// * `sparse` - sparse depth measurements
/// * `power`  - distance exponent (commonly 2.0)
///
/// # Errors
/// Returns an error if `sparse` has no measurements.
pub fn inverse_distance_weighted(sparse: &SparseDepthMap, power: f64) -> Result<CompletionResult> {
    sparse.validate_non_empty()?;

    let h = sparse.height;
    let w = sparse.width;
    let mut dense = Array2::zeros((h, w));
    let mut conf = Array2::zeros((h, w));

    for row in 0..h {
        for col in 0..w {
            let mut weight_sum = 0.0;
            let mut depth_sum = 0.0;
            let mut exact_match = None;

            for m in &sparse.measurements {
                let dr = row as f64 - m.row as f64;
                let dc = col as f64 - m.col as f64;
                let dist = (dr * dr + dc * dc).sqrt();

                if dist < 1e-12 {
                    exact_match = Some(m.depth);
                    break;
                }

                let w = 1.0 / dist.powf(power);
                weight_sum += w;
                depth_sum += w * m.depth;
            }

            if let Some(d) = exact_match {
                dense[[row, col]] = d;
                conf[[row, col]] = 1.0;
            } else if weight_sum > 1e-12 {
                dense[[row, col]] = depth_sum / weight_sum;
                // Confidence based on how concentrated the weights are.
                let max_w = sparse
                    .measurements
                    .iter()
                    .map(|m| {
                        let dr = row as f64 - m.row as f64;
                        let dc = col as f64 - m.col as f64;
                        let dist = (dr * dr + dc * dc).sqrt();
                        if dist < 1e-12 {
                            f64::MAX
                        } else {
                            1.0 / dist.powf(power)
                        }
                    })
                    .fold(0.0_f64, f64::max);
                conf[[row, col]] = (max_w / weight_sum).min(1.0);
            }
        }
    }

    Ok(CompletionResult {
        dense_depth: dense,
        confidence_map: conf,
        method_used: CompletionMethod::InverseDistanceWeighted,
        iterations: 0,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn single_measurement_map(h: usize, w: usize, depth: f64) -> SparseDepthMap {
        SparseDepthMap::new(
            h,
            w,
            vec![SparseMeasurement {
                row: h / 2,
                col: w / 2,
                depth,
                confidence: 1.0,
            }],
        )
        .expect("valid map")
    }

    #[test]
    fn nn_fills_constant_from_single_measurement() {
        let sparse = single_measurement_map(8, 8, 5.0);
        let result = nearest_neighbor_fill(&sparse).expect("should succeed");
        // All pixels should have depth 5.0
        for row in 0..8 {
            for col in 0..8 {
                assert!(
                    (result.dense_depth[[row, col]] - 5.0).abs() < 1e-9,
                    "pixel ({row},{col}) should be 5.0"
                );
            }
        }
    }

    #[test]
    fn nn_closer_measurement_wins() {
        let sparse = SparseDepthMap::new(
            10,
            10,
            vec![
                SparseMeasurement {
                    row: 0,
                    col: 0,
                    depth: 1.0,
                    confidence: 1.0,
                },
                SparseMeasurement {
                    row: 9,
                    col: 9,
                    depth: 9.0,
                    confidence: 1.0,
                },
            ],
        )
        .expect("valid");

        let result = nearest_neighbor_fill(&sparse).expect("ok");
        // Top-left corner should be 1.0 (closer to (0,0))
        assert!((result.dense_depth[[0, 1]] - 1.0).abs() < 1e-9);
        // Bottom-right corner should be 9.0 (closer to (9,9))
        assert!((result.dense_depth[[9, 8]] - 9.0).abs() < 1e-9);
    }

    #[test]
    fn bilateral_preserves_edge() {
        // Guide has a step: left half=0, right half=1
        let h = 10;
        let w = 10;
        let mut guide = Array2::zeros((h, w));
        for row in 0..h {
            for col in 5..w {
                guide[[row, col]] = 1.0;
            }
        }

        // Sparse measurements: depth 2 on left, depth 8 on right
        let sparse = SparseDepthMap::new(
            h,
            w,
            vec![
                SparseMeasurement {
                    row: 5,
                    col: 2,
                    depth: 2.0,
                    confidence: 1.0,
                },
                SparseMeasurement {
                    row: 5,
                    col: 7,
                    depth: 8.0,
                    confidence: 1.0,
                },
            ],
        )
        .expect("valid");

        let config = DepthCompletionConfig {
            sigma_spatial: 3.0,
            sigma_intensity: 0.05,
            ..Default::default()
        };

        let result = bilateral_upsample(&sparse, &guide, &config).expect("ok");
        // Pixel on left side should be closer to 2.0
        let left_val = result.dense_depth[[5, 1]];
        // Pixel on right side should be closer to 8.0
        let right_val = result.dense_depth[[5, 8]];
        assert!(
            left_val < 5.0,
            "left side should lean towards 2.0, got {left_val}"
        );
        assert!(
            right_val > 5.0,
            "right side should lean towards 8.0, got {right_val}"
        );
    }

    #[test]
    fn idw_close_points_dominate() {
        let sparse = SparseDepthMap::new(
            10,
            10,
            vec![
                SparseMeasurement {
                    row: 1,
                    col: 1,
                    depth: 10.0,
                    confidence: 1.0,
                },
                SparseMeasurement {
                    row: 9,
                    col: 9,
                    depth: 100.0,
                    confidence: 1.0,
                },
            ],
        )
        .expect("valid");

        let result = inverse_distance_weighted(&sparse, 2.0).expect("ok");
        // Pixel (2, 2) is close to (1,1), depth should be much closer to 10 than 100
        let val = result.dense_depth[[2, 2]];
        assert!(
            val < 30.0,
            "pixel near (1,1) should be close to 10.0, got {val}"
        );
    }

    #[test]
    fn empty_sparse_map_errors() {
        let sparse = SparseDepthMap::new(5, 5, vec![]).expect("valid empty");
        assert!(nearest_neighbor_fill(&sparse).is_err());
        assert!(inverse_distance_weighted(&sparse, 2.0).is_err());
    }

    #[test]
    fn out_of_bounds_measurement_errors() {
        let result = SparseDepthMap::new(
            5,
            5,
            vec![SparseMeasurement {
                row: 10,
                col: 0,
                depth: 1.0,
                confidence: 1.0,
            }],
        );
        assert!(result.is_err());
    }

    #[test]
    fn bilateral_dimension_mismatch_errors() {
        let sparse = single_measurement_map(5, 5, 1.0);
        let guide = Array2::zeros((3, 3)); // wrong size
        let config = DepthCompletionConfig::default();
        assert!(bilateral_upsample(&sparse, &guide, &config).is_err());
    }
}
