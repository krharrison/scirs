//! Distance transform algorithms for 2D binary images
//!
//! This module provides exact and approximate distance transforms including:
//!
//! - **Euclidean distance transform (EDT)**: Exact via the Meijster et al. (2000)
//!   separable algorithm, running in O(n) per pixel.
//! - **City block (Manhattan) distance transform**: L1 distance via two-pass scanning.
//! - **Chessboard distance transform**: L-infinity distance via two-pass scanning.
//! - **Geodesic distance transform**: Distance constrained to a mask region.
//! - **Signed distance function (SDF)**: Positive outside, negative inside,
//!   useful for level-set methods.
//!
//! # References
//!
//! - Meijster, A., Roerdink, J.B.T.M., Hesselink, W.H. (2000).
//!   "A General Algorithm for Computing Distance Transforms in Linear Time"
//! - Felzenszwalb, P.F. & Huttenlocher, D.P. (2012).
//!   "Distance Transforms of Sampled Functions"

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::Array2;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// ---------------------------------------------------------------------------
// Euclidean Distance Transform (Meijster algorithm)
// ---------------------------------------------------------------------------

/// Exact Euclidean Distance Transform using the Meijster separable algorithm
///
/// For each foreground pixel (`true`), computes the exact Euclidean distance to
/// the nearest background pixel (`false`). Background pixels have distance 0.
///
/// The algorithm runs in O(n) time per pixel (linear in the total number of
/// pixels) using a separable two-phase approach.
///
/// # Arguments
///
/// * `binary` - Input binary image (`true` = foreground, `false` = background)
///
/// # Returns
///
/// 2D array of distances (f64).
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::distance_transforms::euclidean_dt;
///
/// let binary = array![
///     [false, false, false, false, false],
///     [false, true,  true,  true,  false],
///     [false, true,  true,  true,  false],
///     [false, true,  true,  true,  false],
///     [false, false, false, false, false],
/// ];
///
/// let dist = euclidean_dt(&binary).expect("should succeed");
/// assert!((dist[[0, 0]] - 0.0).abs() < 1e-10);
/// assert!((dist[[2, 2]] - 2.0).abs() < 1e-10); // center pixel, distance = 2
/// ```
pub fn euclidean_dt(binary: &Array2<bool>) -> NdimageResult<Array2<f64>> {
    let rows = binary.nrows();
    let cols = binary.ncols();

    if rows == 0 || cols == 0 {
        return Ok(Array2::zeros((rows, cols)));
    }

    let inf = (rows + cols) as f64; // safe "infinity" sentinel

    // Phase 1: compute G(x, y) = distance to nearest background in the same column
    // G[r][c] = distance from (r,c) to nearest background pixel in column c
    let mut g = Array2::<f64>::from_elem((rows, cols), 0.0);

    // Forward pass per column
    for c in 0..cols {
        // Top-down
        if binary[[0, c]] {
            g[[0, c]] = inf;
        }
        for r in 1..rows {
            if binary[[r, c]] {
                g[[r, c]] = g[[r - 1, c]] + 1.0;
            } else {
                g[[r, c]] = 0.0;
            }
        }

        // Bottom-up
        for r in (0..rows.saturating_sub(1)).rev() {
            let candidate = g[[r + 1, c]] + 1.0;
            if candidate < g[[r, c]] {
                g[[r, c]] = candidate;
            }
        }
    }

    // Phase 2: use the Meijster separable approach along rows
    // For each row r, compute DT[r][c] = min over q { (c-q)^2 + G[r][q]^2 }
    // This is done with the lower-envelope parabola technique.
    let mut dt = Array2::<f64>::zeros((rows, cols));

    // Temporary buffers for the lower envelope
    let mut s = vec![0i64; cols]; // positions of parabola vertices
    let mut t = vec![0.0f64; cols]; // intersection points
                                    // sep function: intersection of parabolas at q1 and q2 with offsets f_q1, f_q2
                                    // sep(q1, f1, q2, f2) = floor( (q2^2 - q1^2 + f2^2 - f1^2) / (2*(q2-q1)) )
                                    // We compute in f64 for precision.

    for r in 0..rows {
        // Build lower envelope of parabolas { f_q(x) = (x-q)^2 + g[r][q]^2 } for q in 0..cols

        let mut k = 0usize; // index into s/t
        s[0] = 0;
        t[0] = f64::NEG_INFINITY;

        for q in 1..cols {
            let fq = g[[r, q]];
            let fq2 = fq * fq;

            loop {
                let sq = s[k] as f64;
                let fsq = g[[r, s[k] as usize]];
                let fsq2 = fsq * fsq;

                // Intersection of parabolas at s[k] and q
                let sep =
                    ((q as f64 * q as f64 - sq * sq) + (fq2 - fsq2)) / (2.0 * (q as f64 - sq));

                if sep > t[k] {
                    // This parabola is above the current envelope at intersection
                    k += 1;
                    s[k] = q as i64;
                    t[k] = sep;
                    break;
                }

                if k == 0 {
                    s[0] = q as i64;
                    // t[0] stays at NEG_INFINITY
                    break;
                }
                k -= 1;
            }
        }

        // Scan columns and evaluate the lower envelope
        let mut k_scan = k;
        for c in (0..cols).rev() {
            while k_scan > 0 && t[k_scan] > c as f64 {
                k_scan -= 1;
            }
            let sq = s[k_scan] as usize;
            let dx = c as f64 - sq as f64;
            let gy = g[[r, sq]];
            dt[[r, c]] = (dx * dx + gy * gy).sqrt();
        }
    }

    Ok(dt)
}

// ---------------------------------------------------------------------------
// City Block (Manhattan) Distance Transform
// ---------------------------------------------------------------------------

/// City block (Manhattan / L1) distance transform
///
/// For each foreground pixel, computes the L1 distance to the nearest
/// background pixel. Uses a fast two-pass scanning algorithm.
///
/// # Arguments
///
/// * `binary` - Input binary image (`true` = foreground)
///
/// # Returns
///
/// 2D array of L1 distances.
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::distance_transforms::cityblock_dt;
///
/// let binary = array![
///     [false, false, false, false, false],
///     [false, true,  true,  true,  false],
///     [false, true,  true,  true,  false],
///     [false, true,  true,  true,  false],
///     [false, false, false, false, false],
/// ];
///
/// let dist = cityblock_dt(&binary).expect("should succeed");
/// assert_eq!(dist[[2, 2]], 2.0); // center, 2 steps to border
/// assert_eq!(dist[[1, 1]], 1.0); // one step from border
/// ```
pub fn cityblock_dt(binary: &Array2<bool>) -> NdimageResult<Array2<f64>> {
    let rows = binary.nrows();
    let cols = binary.ncols();

    if rows == 0 || cols == 0 {
        return Ok(Array2::zeros((rows, cols)));
    }

    let inf = (rows + cols) as f64;
    let mut dist = Array2::<f64>::zeros((rows, cols));

    // Initialize
    for r in 0..rows {
        for c in 0..cols {
            if binary[[r, c]] {
                dist[[r, c]] = inf;
            }
        }
    }

    // Forward pass (top-left to bottom-right)
    for r in 0..rows {
        for c in 0..cols {
            if !binary[[r, c]] {
                continue;
            }
            if r > 0 {
                let candidate = dist[[r - 1, c]] + 1.0;
                if candidate < dist[[r, c]] {
                    dist[[r, c]] = candidate;
                }
            }
            if c > 0 {
                let candidate = dist[[r, c - 1]] + 1.0;
                if candidate < dist[[r, c]] {
                    dist[[r, c]] = candidate;
                }
            }
        }
    }

    // Backward pass (bottom-right to top-left)
    for r in (0..rows).rev() {
        for c in (0..cols).rev() {
            if !binary[[r, c]] {
                continue;
            }
            if r + 1 < rows {
                let candidate = dist[[r + 1, c]] + 1.0;
                if candidate < dist[[r, c]] {
                    dist[[r, c]] = candidate;
                }
            }
            if c + 1 < cols {
                let candidate = dist[[r, c + 1]] + 1.0;
                if candidate < dist[[r, c]] {
                    dist[[r, c]] = candidate;
                }
            }
        }
    }

    Ok(dist)
}

// ---------------------------------------------------------------------------
// Chessboard (Chebyshev / L-inf) Distance Transform
// ---------------------------------------------------------------------------

/// Chessboard (Chebyshev / L-infinity) distance transform
///
/// For each foreground pixel, computes the L-infinity distance to the nearest
/// background pixel. L-infinity considers 8-connected neighbors, so moving
/// diagonally costs 1 step.
///
/// # Arguments
///
/// * `binary` - Input binary image (`true` = foreground)
///
/// # Returns
///
/// 2D array of L-infinity distances.
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::distance_transforms::chessboard_dt;
///
/// let binary = array![
///     [false, false, false, false, false],
///     [false, true,  true,  true,  false],
///     [false, true,  true,  true,  false],
///     [false, true,  true,  true,  false],
///     [false, false, false, false, false],
/// ];
///
/// let dist = chessboard_dt(&binary).expect("should succeed");
/// assert_eq!(dist[[2, 2]], 2.0); // center
/// assert_eq!(dist[[1, 1]], 1.0); // corner, 1 diagonal step
/// ```
pub fn chessboard_dt(binary: &Array2<bool>) -> NdimageResult<Array2<f64>> {
    let rows = binary.nrows();
    let cols = binary.ncols();

    if rows == 0 || cols == 0 {
        return Ok(Array2::zeros((rows, cols)));
    }

    let inf = (rows + cols) as f64;
    let mut dist = Array2::<f64>::zeros((rows, cols));

    // Initialize
    for r in 0..rows {
        for c in 0..cols {
            if binary[[r, c]] {
                dist[[r, c]] = inf;
            }
        }
    }

    // Forward pass (top-left to bottom-right)
    for r in 0..rows {
        for c in 0..cols {
            if !binary[[r, c]] {
                continue;
            }
            // Check already-visited neighbors (top, left, top-left, top-right)
            let candidates = [
                if r > 0 { dist[[r - 1, c]] + 1.0 } else { inf },
                if c > 0 { dist[[r, c - 1]] + 1.0 } else { inf },
                if r > 0 && c > 0 {
                    dist[[r - 1, c - 1]] + 1.0
                } else {
                    inf
                },
                if r > 0 && c + 1 < cols {
                    dist[[r - 1, c + 1]] + 1.0
                } else {
                    inf
                },
            ];

            for &cand in &candidates {
                if cand < dist[[r, c]] {
                    dist[[r, c]] = cand;
                }
            }
        }
    }

    // Backward pass (bottom-right to top-left)
    for r in (0..rows).rev() {
        for c in (0..cols).rev() {
            if !binary[[r, c]] {
                continue;
            }
            let candidates = [
                if r + 1 < rows {
                    dist[[r + 1, c]] + 1.0
                } else {
                    inf
                },
                if c + 1 < cols {
                    dist[[r, c + 1]] + 1.0
                } else {
                    inf
                },
                if r + 1 < rows && c + 1 < cols {
                    dist[[r + 1, c + 1]] + 1.0
                } else {
                    inf
                },
                if r + 1 < rows && c > 0 {
                    dist[[r + 1, c - 1]] + 1.0
                } else {
                    inf
                },
            ];

            for &cand in &candidates {
                if cand < dist[[r, c]] {
                    dist[[r, c]] = cand;
                }
            }
        }
    }

    Ok(dist)
}

// ---------------------------------------------------------------------------
// Geodesic Distance Transform
// ---------------------------------------------------------------------------

/// Entry for the geodesic distance priority queue
#[derive(Clone, Debug)]
struct GeoEntry {
    row: usize,
    col: usize,
    dist: f64,
}

impl PartialEq for GeoEntry {
    fn eq(&self, other: &Self) -> bool {
        self.row == other.row && self.col == other.col
    }
}
impl Eq for GeoEntry {}

impl PartialOrd for GeoEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GeoEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: reverse for BinaryHeap
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
    }
}

/// Geodesic distance transform
///
/// Computes the geodesic distance from each pixel in `mask` to the nearest
/// seed pixel in `seeds`. The geodesic distance is the shortest path distance
/// that stays within the `mask` region.
///
/// Uses Dijkstra-like propagation with 8-connectivity. The cost of moving to
/// an adjacent pixel equals the Euclidean step distance (1.0 for face-adjacent,
/// sqrt(2) for diagonal).
///
/// # Arguments
///
/// * `mask`  - Binary mask defining the allowed region
/// * `seeds` - Binary mask of seed (source) pixels; must be a subset of `mask`
///
/// # Returns
///
/// 2D distance array. Pixels outside `mask` have distance `f64::INFINITY`.
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::distance_transforms::geodesic_dt;
///
/// let mask = array![
///     [true,  true,  true,  true, true],
///     [true,  false, false, false, true],
///     [true,  false, false, false, true],
///     [true,  false, false, false, true],
///     [true,  true,  true,  true, true],
/// ];
///
/// let mut seeds = scirs2_core::ndarray::Array2::from_elem((5, 5), false);
/// seeds[[0, 0]] = true;
///
/// let dist = geodesic_dt(&mask, &seeds).expect("should succeed");
/// assert!((dist[[0, 0]] - 0.0).abs() < 1e-10); // seed
/// assert!(dist[[2, 2]].is_infinite()); // outside mask
/// assert!(dist[[0, 4]].is_finite()); // reachable through mask border
/// ```
pub fn geodesic_dt(mask: &Array2<bool>, seeds: &Array2<bool>) -> NdimageResult<Array2<f64>> {
    if mask.shape() != seeds.shape() {
        return Err(NdimageError::DimensionError(
            "Mask and seeds must have the same shape".to_string(),
        ));
    }

    let rows = mask.nrows();
    let cols = mask.ncols();

    if rows == 0 || cols == 0 {
        return Ok(Array2::zeros((rows, cols)));
    }

    let sqrt2 = std::f64::consts::SQRT_2;

    // 8-connected neighbors with their step costs
    let neighbors: [(isize, isize, f64); 8] = [
        (-1, -1, sqrt2),
        (-1, 0, 1.0),
        (-1, 1, sqrt2),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (1, -1, sqrt2),
        (1, 0, 1.0),
        (1, 1, sqrt2),
    ];

    let mut dist = Array2::<f64>::from_elem((rows, cols), f64::INFINITY);
    let mut visited = Array2::<bool>::from_elem((rows, cols), false);
    let mut queue = BinaryHeap::new();

    // Initialize seeds
    for r in 0..rows {
        for c in 0..cols {
            if seeds[[r, c]] && mask[[r, c]] {
                dist[[r, c]] = 0.0;
                queue.push(GeoEntry {
                    row: r,
                    col: c,
                    dist: 0.0,
                });
            }
        }
    }

    // Dijkstra propagation
    while let Some(entry) = queue.pop() {
        let r = entry.row;
        let c = entry.col;

        if visited[[r, c]] {
            continue;
        }
        visited[[r, c]] = true;

        for &(dr, dc, step_cost) in &neighbors {
            let nr = r as isize + dr;
            let nc = c as isize + dc;

            if nr < 0 || nr >= rows as isize || nc < 0 || nc >= cols as isize {
                continue;
            }

            let nr = nr as usize;
            let nc = nc as usize;

            if !mask[[nr, nc]] || visited[[nr, nc]] {
                continue;
            }

            let new_dist = entry.dist + step_cost;
            if new_dist < dist[[nr, nc]] {
                dist[[nr, nc]] = new_dist;
                queue.push(GeoEntry {
                    row: nr,
                    col: nc,
                    dist: new_dist,
                });
            }
        }
    }

    Ok(dist)
}

// ---------------------------------------------------------------------------
// Signed Distance Function (SDF)
// ---------------------------------------------------------------------------

/// Signed distance function (SDF) for a binary image
///
/// Computes a signed distance where:
/// - Positive values are outside the foreground region (distance to nearest foreground pixel)
/// - Negative values are inside the foreground region (negative distance to nearest background pixel)
/// - Zero is on the boundary
///
/// This is useful for level-set methods where the zero level set represents the contour.
///
/// Uses the Meijster EDT algorithm applied to both the foreground and its complement.
///
/// # Arguments
///
/// * `binary` - Input binary image (`true` = foreground / inside)
///
/// # Returns
///
/// Signed distance array. Negative inside, positive outside, zero on boundary.
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::distance_transforms::signed_distance_function;
///
/// let binary = array![
///     [false, false, false, false, false],
///     [false, true,  true,  true,  false],
///     [false, true,  true,  true,  false],
///     [false, true,  true,  true,  false],
///     [false, false, false, false, false],
/// ];
///
/// let sdf = signed_distance_function(&binary).expect("should succeed");
/// assert!(sdf[[0, 0]] > 0.0);  // outside
/// assert!(sdf[[2, 2]] < 0.0);  // inside
/// // Boundary pixels should be close to zero
/// ```
pub fn signed_distance_function(binary: &Array2<bool>) -> NdimageResult<Array2<f64>> {
    let rows = binary.nrows();
    let cols = binary.ncols();

    if rows == 0 || cols == 0 {
        return Ok(Array2::zeros((rows, cols)));
    }

    // Check edge cases: all foreground or all background
    let has_foreground = binary.iter().any(|&v| v);
    let has_background = binary.iter().any(|&v| !v);

    if !has_foreground {
        // All background: distance to nearest foreground is undefined, use 0
        return Ok(Array2::zeros((rows, cols)));
    }
    if !has_background {
        // All foreground: distance to nearest background is undefined, use 0 (on boundary)
        return Ok(Array2::zeros((rows, cols)));
    }

    // Distance to nearest background pixel (for inside pixels)
    let dist_inside = euclidean_dt(binary)?;

    // Create complement
    let complement: Array2<bool> = binary.mapv(|v| !v);

    // Distance to nearest foreground pixel (for outside pixels)
    let dist_outside = euclidean_dt(&complement)?;

    // Combine: negative inside, positive outside
    let mut sdf = Array2::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            if binary[[r, c]] {
                // Inside: negative distance
                sdf[[r, c]] = -dist_inside[[r, c]];
            } else {
                // Outside: positive distance
                sdf[[r, c]] = dist_outside[[r, c]];
            }
        }
    }

    Ok(sdf)
}

/// Compute the nearest background pixel indices for each foreground pixel
///
/// This is the feature transform (also known as the Voronoi diagram of background pixels).
///
/// # Arguments
///
/// * `binary` - Input binary image (`true` = foreground)
///
/// # Returns
///
/// Two 2D arrays `(nearest_row, nearest_col)` containing the coordinates of
/// the nearest background pixel for each pixel. Background pixels map to themselves.
pub fn nearest_background(binary: &Array2<bool>) -> NdimageResult<(Array2<usize>, Array2<usize>)> {
    let rows = binary.nrows();
    let cols = binary.ncols();

    if rows == 0 || cols == 0 {
        return Ok((Array2::zeros((rows, cols)), Array2::zeros((rows, cols))));
    }

    let inf_dist = ((rows * rows + cols * cols) as f64).sqrt() + 1.0;

    // Initialize
    let mut dist = Array2::<f64>::from_elem((rows, cols), inf_dist);
    let mut nearest_r = Array2::<usize>::zeros((rows, cols));
    let mut nearest_c = Array2::<usize>::zeros((rows, cols));

    // Background pixels have distance 0 to themselves
    for r in 0..rows {
        for c in 0..cols {
            if !binary[[r, c]] {
                dist[[r, c]] = 0.0;
                nearest_r[[r, c]] = r;
                nearest_c[[r, c]] = c;
            }
        }
    }

    // Multi-pass propagation (approximate but practical)
    // Forward pass
    for r in 0..rows {
        for c in 0..cols {
            if !binary[[r, c]] {
                continue;
            }

            // Check top
            if r > 0 {
                let nr = nearest_r[[r - 1, c]];
                let nc = nearest_c[[r - 1, c]];
                let d =
                    (((r as f64 - nr as f64).powi(2)) + ((c as f64 - nc as f64).powi(2))).sqrt();
                if d < dist[[r, c]] {
                    dist[[r, c]] = d;
                    nearest_r[[r, c]] = nr;
                    nearest_c[[r, c]] = nc;
                }
            }
            // Check left
            if c > 0 {
                let nr = nearest_r[[r, c - 1]];
                let nc = nearest_c[[r, c - 1]];
                let d =
                    (((r as f64 - nr as f64).powi(2)) + ((c as f64 - nc as f64).powi(2))).sqrt();
                if d < dist[[r, c]] {
                    dist[[r, c]] = d;
                    nearest_r[[r, c]] = nr;
                    nearest_c[[r, c]] = nc;
                }
            }
            // Check top-left
            if r > 0 && c > 0 {
                let nr = nearest_r[[r - 1, c - 1]];
                let nc = nearest_c[[r - 1, c - 1]];
                let d =
                    (((r as f64 - nr as f64).powi(2)) + ((c as f64 - nc as f64).powi(2))).sqrt();
                if d < dist[[r, c]] {
                    dist[[r, c]] = d;
                    nearest_r[[r, c]] = nr;
                    nearest_c[[r, c]] = nc;
                }
            }
            // Check top-right
            if r > 0 && c + 1 < cols {
                let nr = nearest_r[[r - 1, c + 1]];
                let nc = nearest_c[[r - 1, c + 1]];
                let d =
                    (((r as f64 - nr as f64).powi(2)) + ((c as f64 - nc as f64).powi(2))).sqrt();
                if d < dist[[r, c]] {
                    dist[[r, c]] = d;
                    nearest_r[[r, c]] = nr;
                    nearest_c[[r, c]] = nc;
                }
            }
        }
    }

    // Backward pass
    for r in (0..rows).rev() {
        for c in (0..cols).rev() {
            if !binary[[r, c]] {
                continue;
            }

            // Check bottom
            if r + 1 < rows {
                let nr = nearest_r[[r + 1, c]];
                let nc = nearest_c[[r + 1, c]];
                let d =
                    (((r as f64 - nr as f64).powi(2)) + ((c as f64 - nc as f64).powi(2))).sqrt();
                if d < dist[[r, c]] {
                    dist[[r, c]] = d;
                    nearest_r[[r, c]] = nr;
                    nearest_c[[r, c]] = nc;
                }
            }
            // Check right
            if c + 1 < cols {
                let nr = nearest_r[[r, c + 1]];
                let nc = nearest_c[[r, c + 1]];
                let d =
                    (((r as f64 - nr as f64).powi(2)) + ((c as f64 - nc as f64).powi(2))).sqrt();
                if d < dist[[r, c]] {
                    dist[[r, c]] = d;
                    nearest_r[[r, c]] = nr;
                    nearest_c[[r, c]] = nc;
                }
            }
            // Check bottom-right
            if r + 1 < rows && c + 1 < cols {
                let nr = nearest_r[[r + 1, c + 1]];
                let nc = nearest_c[[r + 1, c + 1]];
                let d =
                    (((r as f64 - nr as f64).powi(2)) + ((c as f64 - nc as f64).powi(2))).sqrt();
                if d < dist[[r, c]] {
                    dist[[r, c]] = d;
                    nearest_r[[r, c]] = nr;
                    nearest_c[[r, c]] = nc;
                }
            }
            // Check bottom-left
            if r + 1 < rows && c > 0 {
                let nr = nearest_r[[r + 1, c - 1]];
                let nc = nearest_c[[r + 1, c - 1]];
                let d =
                    (((r as f64 - nr as f64).powi(2)) + ((c as f64 - nc as f64).powi(2))).sqrt();
                if d < dist[[r, c]] {
                    dist[[r, c]] = d;
                    nearest_r[[r, c]] = nr;
                    nearest_c[[r, c]] = nc;
                }
            }
        }
    }

    Ok((nearest_r, nearest_c))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    // ---- Euclidean DT ----

    #[test]
    fn test_euclidean_dt_empty() {
        let binary: Array2<bool> = Array2::from_elem((0, 0), false);
        let dist = euclidean_dt(&binary).expect("should succeed");
        assert_eq!(dist.len(), 0);
    }

    #[test]
    fn test_euclidean_dt_all_background() {
        let binary = Array2::from_elem((5, 5), false);
        let dist = euclidean_dt(&binary).expect("should succeed");
        for &v in dist.iter() {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_euclidean_dt_all_foreground() {
        // With no background pixels, all distances should be very large
        let binary = Array2::from_elem((5, 5), true);
        let dist = euclidean_dt(&binary).expect("should succeed");
        // The minimum distance should be > 0 for interior pixels
        // (Actually with no background, distances grow from edge of image
        //  but the algorithm uses large sentinel values)
        for &v in dist.iter() {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn test_euclidean_dt_single_background() {
        // Single background pixel at center
        let mut binary = Array2::from_elem((5, 5), true);
        binary[[2, 2]] = false;

        let dist = euclidean_dt(&binary).expect("should succeed");
        assert_abs_diff_eq!(dist[[2, 2]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(dist[[2, 3]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(dist[[1, 1]], std::f64::consts::SQRT_2, epsilon = 1e-10);
        assert_abs_diff_eq!(dist[[0, 2]], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_euclidean_dt_border() {
        let binary = array![
            [false, false, false, false, false],
            [false, true, true, true, false],
            [false, true, true, true, false],
            [false, true, true, true, false],
            [false, false, false, false, false],
        ];

        let dist = euclidean_dt(&binary).expect("should succeed");

        // Corners of foreground region: distance = 1
        assert_abs_diff_eq!(dist[[1, 1]], 1.0, epsilon = 1e-10);
        // Center: distance = 2
        assert_abs_diff_eq!(dist[[2, 2]], 2.0, epsilon = 1e-10);
        // Background: distance = 0
        assert_abs_diff_eq!(dist[[0, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_euclidean_dt_asymmetric() {
        // Test non-square image
        let binary = array![
            [false, false, false, false, false, false, false],
            [false, true, true, true, true, true, false],
            [false, true, true, true, true, true, false],
            [false, false, false, false, false, false, false],
        ];

        let dist = euclidean_dt(&binary).expect("should succeed");
        assert_abs_diff_eq!(dist[[1, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(dist[[1, 3]], 1.0, epsilon = 1e-10);
    }

    // ---- City Block DT ----

    #[test]
    fn test_cityblock_dt_basic() {
        let binary = array![
            [false, false, false, false, false],
            [false, true, true, true, false],
            [false, true, true, true, false],
            [false, true, true, true, false],
            [false, false, false, false, false],
        ];

        let dist = cityblock_dt(&binary).expect("should succeed");
        assert_abs_diff_eq!(dist[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(dist[[1, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(dist[[2, 2]], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cityblock_dt_single_pixel() {
        let mut binary = Array2::from_elem((5, 5), false);
        binary[[2, 2]] = true;

        let dist = cityblock_dt(&binary).expect("should succeed");
        assert_abs_diff_eq!(dist[[2, 2]], 1.0, epsilon = 1e-10);
        // It's surrounded by background so distance = 1
    }

    #[test]
    fn test_cityblock_dt_empty() {
        let binary: Array2<bool> = Array2::from_elem((0, 0), false);
        let dist = cityblock_dt(&binary).expect("should succeed");
        assert_eq!(dist.len(), 0);
    }

    // ---- Chessboard DT ----

    #[test]
    fn test_chessboard_dt_basic() {
        let binary = array![
            [false, false, false, false, false],
            [false, true, true, true, false],
            [false, true, true, true, false],
            [false, true, true, true, false],
            [false, false, false, false, false],
        ];

        let dist = chessboard_dt(&binary).expect("should succeed");
        assert_abs_diff_eq!(dist[[0, 0]], 0.0, epsilon = 1e-10);
        // Corner of foreground: 1 diagonal step
        assert_abs_diff_eq!(dist[[1, 1]], 1.0, epsilon = 1e-10);
        // Center: 2 steps
        assert_abs_diff_eq!(dist[[2, 2]], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_chessboard_dt_vs_cityblock() {
        // Chessboard distance should always be <= cityblock distance
        let binary = array![
            [false, false, false, false, false],
            [false, true, true, true, false],
            [false, true, true, true, false],
            [false, true, true, true, false],
            [false, false, false, false, false],
        ];

        let chess = chessboard_dt(&binary).expect("chessboard");
        let city = cityblock_dt(&binary).expect("cityblock");

        for r in 0..5 {
            for c in 0..5 {
                assert!(
                    chess[[r, c]] <= city[[r, c]] + 1e-10,
                    "Chessboard should be <= cityblock at ({}, {})",
                    r,
                    c
                );
            }
        }
    }

    // ---- Geodesic DT ----

    #[test]
    fn test_geodesic_dt_basic() {
        // A mask with a hole; geodesic distance must go around it
        let mask = array![
            [true, true, true, true, true],
            [true, false, false, false, true],
            [true, false, false, false, true],
            [true, false, false, false, true],
            [true, true, true, true, true],
        ];

        let mut seeds = Array2::from_elem((5, 5), false);
        seeds[[0, 0]] = true;

        let dist = geodesic_dt(&mask, &seeds).expect("should succeed");

        // Seed should have distance 0
        assert_abs_diff_eq!(dist[[0, 0]], 0.0, epsilon = 1e-10);

        // Inside the hole: infinite (not reachable)
        assert!(dist[[2, 2]].is_infinite());

        // Corner pixel [0, 4]: reachable along the top border
        assert!(dist[[0, 4]].is_finite());
        assert_abs_diff_eq!(dist[[0, 4]], 4.0, epsilon = 1e-10);

        // [4, 0]: reachable along the left border
        assert!(dist[[4, 0]].is_finite());
        assert_abs_diff_eq!(dist[[4, 0]], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_geodesic_dt_shape_mismatch() {
        let mask = Array2::from_elem((3, 3), true);
        let seeds = Array2::from_elem((4, 4), false);
        assert!(geodesic_dt(&mask, &seeds).is_err());
    }

    #[test]
    fn test_geodesic_dt_empty() {
        let mask: Array2<bool> = Array2::from_elem((0, 0), false);
        let seeds: Array2<bool> = Array2::from_elem((0, 0), false);
        let dist = geodesic_dt(&mask, &seeds).expect("should succeed");
        assert_eq!(dist.len(), 0);
    }

    #[test]
    fn test_geodesic_dt_full_mask() {
        // Full mask, seed at center: should give regular Dijkstra distances
        let mask = Array2::from_elem((5, 5), true);
        let mut seeds = Array2::from_elem((5, 5), false);
        seeds[[2, 2]] = true;

        let dist = geodesic_dt(&mask, &seeds).expect("should succeed");
        assert_abs_diff_eq!(dist[[2, 2]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(dist[[2, 3]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(dist[[1, 1]], std::f64::consts::SQRT_2, epsilon = 1e-10);
    }

    // ---- Signed Distance Function ----

    #[test]
    fn test_sdf_basic() {
        let binary = array![
            [false, false, false, false, false],
            [false, true, true, true, false],
            [false, true, true, true, false],
            [false, true, true, true, false],
            [false, false, false, false, false],
        ];

        let sdf = signed_distance_function(&binary).expect("should succeed");

        // Outside: positive
        assert!(sdf[[0, 0]] > 0.0);

        // Inside: negative
        assert!(sdf[[2, 2]] < 0.0);

        // Inside, one pixel from border: should be -1
        assert_abs_diff_eq!(sdf[[1, 2]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sdf[[2, 2]], -2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sdf_empty() {
        let binary: Array2<bool> = Array2::from_elem((0, 0), false);
        let sdf = signed_distance_function(&binary).expect("should succeed");
        assert_eq!(sdf.len(), 0);
    }

    #[test]
    fn test_sdf_all_foreground() {
        let binary = Array2::from_elem((5, 5), true);
        let sdf = signed_distance_function(&binary).expect("should succeed");
        // All values should be negative (inside)
        for &v in sdf.iter() {
            assert!(v <= 0.0);
        }
    }

    #[test]
    fn test_sdf_all_background() {
        let binary = Array2::from_elem((5, 5), false);
        let sdf = signed_distance_function(&binary).expect("should succeed");
        // All should be zero (background is distance 0 from background)
        for &v in sdf.iter() {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-10);
        }
    }

    // ---- Nearest Background ----

    #[test]
    fn test_nearest_background_basic() {
        let binary = array![
            [false, false, false],
            [false, true, false],
            [false, false, false],
        ];

        let (nr, nc) = nearest_background(&binary).expect("should succeed");

        // The center foreground pixel should map to any adjacent background pixel
        // Due to the forward pass, it should map to one of the closest
        let dist_r = (1.0 - nr[[1, 1]] as f64).abs();
        let dist_c = (1.0 - nc[[1, 1]] as f64).abs();
        let total_dist = (dist_r * dist_r + dist_c * dist_c).sqrt();
        assert!(total_dist <= 1.5, "Should map to adjacent pixel");
    }

    #[test]
    fn test_nearest_background_empty() {
        let binary: Array2<bool> = Array2::from_elem((0, 0), false);
        let (nr, nc) = nearest_background(&binary).expect("should succeed");
        assert_eq!(nr.len(), 0);
        assert_eq!(nc.len(), 0);
    }

    // ---- Cross-validation ----

    #[test]
    fn test_edt_vs_cityblock_ordering() {
        // Euclidean distance should always be <= cityblock distance
        let binary = array![
            [false, false, false, false, false],
            [false, true, true, true, false],
            [false, true, true, true, false],
            [false, true, true, true, false],
            [false, false, false, false, false],
        ];

        let edt = euclidean_dt(&binary).expect("edt");
        let city = cityblock_dt(&binary).expect("cityblock");

        for r in 0..5 {
            for c in 0..5 {
                assert!(
                    edt[[r, c]] <= city[[r, c]] + 1e-10,
                    "EDT should be <= cityblock at ({}, {}): edt={}, city={}",
                    r,
                    c,
                    edt[[r, c]],
                    city[[r, c]]
                );
            }
        }
    }

    #[test]
    fn test_edt_symmetric() {
        // EDT should be symmetric for a symmetric input
        let binary = array![
            [false, false, false, false, false],
            [false, true, true, true, false],
            [false, true, true, true, false],
            [false, true, true, true, false],
            [false, false, false, false, false],
        ];

        let dist = euclidean_dt(&binary).expect("should succeed");

        // Check symmetry
        let rows = dist.nrows();
        let cols = dist.ncols();
        for r in 0..rows {
            for c in 0..cols {
                // Vertical symmetry
                assert_abs_diff_eq!(dist[[r, c]], dist[[rows - 1 - r, c]], epsilon = 1e-10);
                // Horizontal symmetry
                assert_abs_diff_eq!(dist[[r, c]], dist[[r, cols - 1 - c]], epsilon = 1e-10);
            }
        }
    }
}
