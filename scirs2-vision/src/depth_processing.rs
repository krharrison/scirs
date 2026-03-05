//! Depth Image Processing
//!
//! Utilities for converting depth images to point clouds, filling depth
//! holes via a simple inward-propagation (fast-marching-style) scheme,
//! bilateral filtering for edge-preserving depth smoothing, surface normal
//! computation, and disparity-to-depth conversion.

use crate::point_cloud::PointCloud;
use scirs2_core::ndarray::{Array2, Array3};
use std::collections::VecDeque;

// ─────────────────────────────────────────────────────────────────────────────
// depth_to_pointcloud
// ─────────────────────────────────────────────────────────────────────────────

/// Unproject a depth image into a 3D point cloud using pinhole camera
/// intrinsics.
///
/// # Arguments
/// * `depth`       – H×W depth image in **metres** (0.0 means invalid).
/// * `fx`, `fy`    – Focal lengths in pixels.
/// * `cx`, `cy`    – Principal point in pixels.
/// * `depth_scale` – Multiply raw pixel values by this factor to get metres
///   (use 1.0 when values are already in metres).
///
/// # Returns
/// A [`PointCloud`] with one point per valid (non-zero) depth pixel.
pub fn depth_to_pointcloud(
    depth: &Array2<f32>,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    depth_scale: f64,
) -> PointCloud {
    let (height, width) = depth.dim();
    let mut pts: Vec<[f64; 3]> = Vec::with_capacity(height * width);

    for row in 0..height {
        for col in 0..width {
            let raw = depth[[row, col]] as f64 * depth_scale;
            if raw <= 0.0 || !raw.is_finite() {
                continue;
            }
            let z = raw;
            let x = (col as f64 - cx) * z / fx;
            let y = (row as f64 - cy) * z / fy;
            pts.push([x, y, z]);
        }
    }

    PointCloud::from_vec(pts)
}

// ─────────────────────────────────────────────────────────────────────────────
// fill_depth_holes
// ─────────────────────────────────────────────────────────────────────────────

/// Fill holes (zero-valued pixels) in a depth image using a
/// fast-marching-inspired inward propagation.
///
/// Starting from all valid pixels adjacent to holes, values propagate
/// inward using a distance-weighted average.  Only connected regions of
/// zeros with an area ≤ `max_hole_size` pixels are filled.
pub fn fill_depth_holes(depth: &Array2<f32>, max_hole_size: usize) -> Array2<f32> {
    let (h, w) = depth.dim();
    let mut output = depth.clone();

    // Identify hole pixels (depth == 0) and label connected regions.
    let mut visited = Array2::from_elem((h, w), false);
    let neighbors4: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    for start_r in 0..h {
        for start_c in 0..w {
            if depth[[start_r, start_c]] > 0.0 || visited[[start_r, start_c]] {
                continue;
            }
            // BFS to find the extent of this hole.
            let mut region: Vec<(usize, usize)> = Vec::new();
            let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
            queue.push_back((start_r, start_c));
            visited[[start_r, start_c]] = true;

            while let Some((r, c)) = queue.pop_front() {
                region.push((r, c));
                for &(dr, dc) in &neighbors4 {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr < 0 || nr >= h as i32 || nc < 0 || nc >= w as i32 {
                        continue;
                    }
                    let nr = nr as usize;
                    let nc = nc as usize;
                    if !visited[[nr, nc]] && depth[[nr, nc]] == 0.0 {
                        visited[[nr, nc]] = true;
                        queue.push_back((nr, nc));
                    }
                }
            }

            // Skip large holes.
            if region.len() > max_hole_size {
                continue;
            }

            // Fill each pixel with the weighted average of its valid neighbours
            // within the original depth image.
            for (r, c) in region {
                let mut sum = 0.0f64;
                let mut weight = 0.0f64;
                for &(dr, dc) in &neighbors4 {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr < 0 || nr >= h as i32 || nc < 0 || nc >= w as i32 {
                        continue;
                    }
                    let v = depth[[nr as usize, nc as usize]];
                    if v > 0.0 {
                        sum += v as f64;
                        weight += 1.0;
                    }
                }
                if weight > 0.0 {
                    output[[r, c]] = (sum / weight) as f32;
                }
            }
        }
    }

    output
}

// ─────────────────────────────────────────────────────────────────────────────
// bilateral_filter_depth
// ─────────────────────────────────────────────────────────────────────────────

/// Bilateral filter for edge-preserving depth-image smoothing.
///
/// Each output pixel is a Gaussian-weighted average of its spatial
/// neighbourhood where the weight is further attenuated by the depth
/// difference (colour term).  Zero-valued (invalid) pixels are excluded from
/// the computation.
///
/// # Arguments
/// * `depth`        – H×W depth image (0 means invalid).
/// * `d`            – Filter diameter (the kernel extends ±(d/2) pixels).
/// * `sigma_color`  – Standard deviation of the intensity (depth) Gaussian.
/// * `sigma_space`  – Standard deviation of the spatial Gaussian.
pub fn bilateral_filter_depth(
    depth: &Array2<f32>,
    d: usize,
    sigma_color: f64,
    sigma_space: f64,
) -> Array2<f32> {
    let (h, w) = depth.dim();
    let mut output = depth.clone();

    if d == 0 || sigma_color <= 0.0 || sigma_space <= 0.0 {
        return output;
    }

    let radius = (d / 2) as i32;
    let inv_2_sc2 = 1.0 / (2.0 * sigma_color * sigma_color);
    let inv_2_ss2 = 1.0 / (2.0 * sigma_space * sigma_space);

    for row in 0..h {
        for col in 0..w {
            let center = depth[[row, col]];
            if center == 0.0 {
                continue;
            }
            let mut sum = 0.0f64;
            let mut weight_total = 0.0f64;

            for dr in -radius..=radius {
                for dc in -radius..=radius {
                    let nr = row as i32 + dr;
                    let nc = col as i32 + dc;
                    if nr < 0 || nr >= h as i32 || nc < 0 || nc >= w as i32 {
                        continue;
                    }
                    let v = depth[[nr as usize, nc as usize]];
                    if v == 0.0 {
                        continue;
                    }
                    let color_diff = (v as f64 - center as f64).powi(2);
                    let space_dist = (dr * dr + dc * dc) as f64;
                    let w_c = (-color_diff * inv_2_sc2).exp();
                    let w_s = (-space_dist * inv_2_ss2).exp();
                    let w = w_c * w_s;
                    sum += v as f64 * w;
                    weight_total += w;
                }
            }

            if weight_total > 1e-12 {
                output[[row, col]] = (sum / weight_total) as f32;
            }
        }
    }

    output
}

// ─────────────────────────────────────────────────────────────────────────────
// depth_normals
// ─────────────────────────────────────────────────────────────────────────────

/// Compute surface normals from a depth image using finite differences.
///
/// For each pixel the normal is computed as the cross product of the
/// forward horizontal and vertical depth gradients (in camera space),
/// then normalised.  Invalid pixels (depth == 0) yield a zero normal.
///
/// # Returns
/// An `(H, W, 3)` array where the last axis contains the (nx, ny, nz)
/// components in camera space.
pub fn depth_normals(depth: &Array2<f32>, fx: f64, fy: f64) -> Array3<f64> {
    let (h, w) = depth.dim();
    let mut normals = Array3::zeros((h, w, 3));

    for row in 0..h {
        for col in 0..w {
            let z = depth[[row, col]];
            if z == 0.0 {
                continue;
            }
            // Horizontal gradient (finite difference along x-axis).
            let (z_r, valid_r) = if col + 1 < w && depth[[row, col + 1]] > 0.0 {
                (depth[[row, col + 1]], true)
            } else if col > 0 && depth[[row, col - 1]] > 0.0 {
                (depth[[row, col - 1]], true)
            } else {
                (z, false)
            };
            // Vertical gradient (finite difference along y-axis).
            let (z_d, valid_d) = if row + 1 < h && depth[[row + 1, col]] > 0.0 {
                (depth[[row + 1, col]], true)
            } else if row > 0 && depth[[row - 1, col]] > 0.0 {
                (depth[[row - 1, col]], true)
            } else {
                (z, false)
            };

            if !valid_r || !valid_d {
                continue;
            }

            // 3-D positions of the three vertices.
            let to_xyz = |r: usize, c: usize, dv: f32| -> [f64; 3] {
                let dv64 = dv as f64;
                [
                    (c as f64 - (w as f64 / 2.0)) * dv64 / fx,
                    (r as f64 - (h as f64 / 2.0)) * dv64 / fy,
                    dv64,
                ]
            };

            let p0 = to_xyz(row, col, z);
            let p_r = to_xyz(row, col + 1, z_r);
            let p_d = to_xyz(row + 1, col, z_d);

            // Tangent vectors.
            let tx = [p_r[0] - p0[0], p_r[1] - p0[1], p_r[2] - p0[2]];
            let ty = [p_d[0] - p0[0], p_d[1] - p0[1], p_d[2] - p0[2]];

            // Cross product.
            let nx = ty[1] * tx[2] - ty[2] * tx[1];
            let ny = ty[2] * tx[0] - ty[0] * tx[2];
            let nz = ty[0] * tx[1] - ty[1] * tx[0];
            let len = (nx * nx + ny * ny + nz * nz).sqrt();

            if len > 1e-12 {
                normals[[row, col, 0]] = nx / len;
                normals[[row, col, 1]] = ny / len;
                normals[[row, col, 2]] = nz / len;
            }
        }
    }

    normals
}

// ─────────────────────────────────────────────────────────────────────────────
// disparity_to_depth
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a stereo disparity map to a depth map.
///
/// Uses the stereo camera equation:
/// `depth = (focal_length × baseline) / disparity`
///
/// Zero-disparity pixels (invalid) produce zero depth.
///
/// # Arguments
/// * `disparity`    – H×W disparity image in pixels.
/// * `baseline`     – Stereo baseline in metres.
/// * `focal_length` – Camera focal length in pixels.
pub fn disparity_to_depth(
    disparity: &Array2<f32>,
    baseline: f64,
    focal_length: f64,
) -> Array2<f32> {
    let (h, w) = disparity.dim();
    let mut depth = Array2::zeros((h, w));

    let fb = (focal_length * baseline) as f32;

    for row in 0..h {
        for col in 0..w {
            let d = disparity[[row, col]];
            if d > 0.0 {
                depth[[row, col]] = fb / d;
            }
        }
    }

    depth
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array2;

    // ─── depth_to_pointcloud ───────────────────────────────────────────────

    #[test]
    fn test_depth_to_pointcloud_shape() {
        // 4×4 depth image, all pixels at 1.0 m.
        let depth = Array2::from_elem((4, 4), 1.0f32);
        let pc = depth_to_pointcloud(&depth, 100.0, 100.0, 2.0, 2.0, 1.0);
        // All 16 pixels should produce a point.
        assert_eq!(pc.n_points(), 16);
    }

    #[test]
    fn test_depth_to_pointcloud_invalid_zeros_excluded() {
        let mut depth = Array2::zeros((4, 4));
        depth[[1, 1]] = 2.0f32;
        depth[[2, 2]] = 3.0f32;
        let pc = depth_to_pointcloud(&depth, 100.0, 100.0, 2.0, 2.0, 1.0);
        assert_eq!(pc.n_points(), 2);
    }

    #[test]
    fn test_depth_to_pointcloud_center_pixel() {
        // A single pixel at the principal point should project to (0, 0, z).
        let mut depth = Array2::zeros((5, 5));
        depth[[2, 2]] = 1.0f32;
        let pc = depth_to_pointcloud(&depth, 100.0, 100.0, 2.0, 2.0, 1.0);
        assert_eq!(pc.n_points(), 1);
        // x = (2 - 2) * 1.0 / 100 = 0, y = (2 - 2) * 1.0 / 100 = 0, z = 1.0
        assert_abs_diff_eq!(pc.points[[0, 0]], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(pc.points[[0, 1]], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(pc.points[[0, 2]], 1.0, epsilon = 1e-9);
    }

    // ─── fill_depth_holes ────────────────────────────────────────────────

    #[test]
    fn test_fill_depth_holes_small_hole() {
        // 5×5 grid with one zero pixel surrounded by 1.0.
        let mut depth = Array2::from_elem((5, 5), 1.0f32);
        depth[[2, 2]] = 0.0;
        let filled = fill_depth_holes(&depth, 10);
        // The hole should now be filled.
        assert!(filled[[2, 2]] > 0.0, "hole should be filled");
        assert_abs_diff_eq!(filled[[2, 2]], 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_fill_depth_holes_large_hole_not_filled() {
        // 10×10 grid of zeros – too large to fill.
        let depth = Array2::zeros((10, 10));
        let filled = fill_depth_holes(&depth, 5);
        // No valid neighbours exist, so nothing can be filled.
        assert!(filled.iter().all(|&v| v == 0.0));
    }

    // ─── bilateral_filter_depth ──────────────────────────────────────────

    #[test]
    fn test_bilateral_filter_preserves_valid_shape() {
        let depth = Array2::from_elem((8, 8), 2.0f32);
        let filtered = bilateral_filter_depth(&depth, 5, 0.5, 2.0);
        assert_eq!(filtered.dim(), (8, 8));
    }

    #[test]
    fn test_bilateral_filter_edge_preservation() {
        // Image with sharp step: left half at 1.0, right half at 5.0.
        let (h, w) = (10, 10);
        let mut depth = Array2::zeros((h, w));
        for r in 0..h {
            for c in 0..w {
                depth[[r, c]] = if c < w / 2 { 1.0 } else { 5.0 };
            }
        }
        let filtered = bilateral_filter_depth(&depth, 5, 0.5, 2.0);
        // Pixels far from the edge should be nearly unchanged.
        assert_abs_diff_eq!(filtered[[5, 0]], 1.0, epsilon = 0.1);
        assert_abs_diff_eq!(filtered[[5, 9]], 5.0, epsilon = 0.1);
    }

    // ─── depth_normals ────────────────────────────────────────────────────

    #[test]
    fn test_depth_normals_shape() {
        let depth = Array2::from_elem((6, 6), 1.0f32);
        let normals = depth_normals(&depth, 100.0, 100.0);
        assert_eq!(normals.dim(), (6, 6, 3));
    }

    #[test]
    fn test_depth_normals_flat_surface_points_forward() {
        // Flat surface – normals should point roughly in the -z direction
        // (camera looks along +z).  For a flat depth image the cross product
        // of the tangent vectors from the projection model points in the
        // +z direction (into the scene).
        let depth = Array2::from_elem((5, 5), 1.0f32);
        let normals = depth_normals(&depth, 100.0, 100.0);
        // The z-component of an interior pixel should be dominant.
        let nz = normals[[2, 2, 2]].abs();
        assert!(
            nz > 0.5,
            "z-component should dominate for a flat surface, got {nz}"
        );
    }

    // ─── disparity_to_depth ───────────────────────────────────────────────

    #[test]
    fn test_disparity_to_depth_values() {
        let disp = Array2::from_elem((3, 3), 2.0f32);
        let depth = disparity_to_depth(&disp, 0.1, 500.0);
        // depth = 500 * 0.1 / 2 = 25.0
        for r in 0..3 {
            for c in 0..3 {
                assert_abs_diff_eq!(depth[[r, c]], 25.0, epsilon = 1e-3);
            }
        }
    }

    #[test]
    fn test_disparity_to_depth_zero_disparity() {
        let mut disp = Array2::from_elem((3, 3), 2.0f32);
        disp[[1, 1]] = 0.0;
        let depth = disparity_to_depth(&disp, 0.1, 500.0);
        assert_abs_diff_eq!(depth[[1, 1]], 0.0, epsilon = 1e-6);
    }
}
