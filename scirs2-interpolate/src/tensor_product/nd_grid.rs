//! N-dimensional regular grid interpolation via tensor-product linear interpolation.
//!
//! `NdGridInterp` performs multilinear (N-linear) interpolation over an
//! arbitrary-dimensional rectilinear grid.  The grid is defined by one sorted
//! coordinate vector per dimension; values are stored in a flat, row-major
//! (`C`-order) array.
//!
//! The algorithm iterates over all 2^N voxel corners at evaluation time and
//! accumulates the weighted sum.  This is equivalent to `TrilinearInterp` for
//! N=3 and `BilinearInterp` for N=2.
//!
//! # Memory layout
//!
//! Values are stored in a flat `Vec<f64>` in row-major (C) order:
//! the last axis varies fastest.  For a grid with shape `[n0, n1, ..., n_{d-1}]`,
//! the element at index `[i0, i1, ..., i_{d-1}]` is stored at flat position
//! `i0 * (n1 * n2 * … * n_{d-1}) + i1 * (n2 * … * n_{d-1}) + … + i_{d-1}`.

use crate::error::{InterpolateError, InterpolateResult};

/// N-dimensional multilinear grid interpolation.
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::tensor_product::NdGridInterp;
///
/// // 2D example: f(x,y) = x + y
/// let grids = vec![
///     vec![0.0_f64, 1.0, 2.0],
///     vec![0.0_f64, 1.0, 2.0],
/// ];
/// // Row-major: values[i*3 + j] = grids[0][i] + grids[1][j]
/// let values: Vec<f64> = (0..3)
///     .flat_map(|i| (0..3).map(move |j| (i as f64) + (j as f64)))
///     .collect();
/// let interp = NdGridInterp::new(grids, values).expect("valid");
/// let v = interp.interpolate(&[0.5, 1.5]).expect("valid");
/// assert!((v - 2.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct NdGridInterp {
    /// Coordinate grids for each dimension (each strictly increasing).
    pub grids: Vec<Vec<f64>>,
    /// Flat, row-major value array with total length = product of grid lengths.
    pub values: Vec<f64>,
    /// Shape of the value array (one entry per dimension).
    pub shape: Vec<usize>,
    /// Number of dimensions.
    pub ndim: usize,
    /// Precomputed row-major strides (stride[d] = product of shape[d+1..ndim]).
    strides: Vec<usize>,
}

impl NdGridInterp {
    /// Construct a new `NdGridInterp`.
    ///
    /// # Arguments
    ///
    /// * `grids` – one strictly increasing coordinate vector per dimension;
    ///   each must have length ≥ 2.
    /// * `values` – flat, row-major value array.  Its length must equal the
    ///   product of all grid lengths.
    ///
    /// # Errors
    ///
    /// Returns an error when:
    /// - `grids` is empty.
    /// - Any grid has fewer than 2 points.
    /// - Any grid is not strictly increasing.
    /// - `values.len()` ≠ product of grid lengths.
    pub fn new(grids: Vec<Vec<f64>>, values: Vec<f64>) -> InterpolateResult<Self> {
        let ndim = grids.len();
        if ndim == 0 {
            return Err(InterpolateError::empty_data("NdGridInterp"));
        }

        for (d, grid) in grids.iter().enumerate() {
            if grid.len() < 2 {
                return Err(InterpolateError::insufficient_points(
                    2,
                    grid.len(),
                    &format!("NdGridInterp grid[{}]", d),
                ));
            }
            for i in 1..grid.len() {
                if grid[i] <= grid[i - 1] {
                    return Err(InterpolateError::invalid_input(format!(
                        "NdGridInterp: grid[{}] not strictly increasing at index {}: {} <= {}",
                        d,
                        i,
                        grid[i],
                        grid[i - 1]
                    )));
                }
            }
        }

        let shape: Vec<usize> = grids.iter().map(|g| g.len()).collect();
        let total: usize = shape.iter().product();

        if total != values.len() {
            return Err(InterpolateError::dimension_mismatch(
                total,
                values.len(),
                "NdGridInterp: values length vs grid product",
            ));
        }

        // Precompute row-major strides
        let mut strides = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            strides[d] = strides[d + 1] * shape[d + 1];
        }

        Ok(Self {
            grids,
            values,
            shape,
            ndim,
            strides,
        })
    }

    /// Multilinear interpolation at `coords` (one coordinate per dimension).
    ///
    /// Points outside the grid are clamped to the boundary in each dimension.
    ///
    /// # Arguments
    ///
    /// * `coords` – slice of length `ndim` with the query point coordinates.
    ///
    /// # Errors
    ///
    /// Returns an error when `coords.len() != ndim`.
    pub fn interpolate(&self, coords: &[f64]) -> InterpolateResult<f64> {
        if coords.len() != self.ndim {
            return Err(InterpolateError::dimension_mismatch(
                self.ndim,
                coords.len(),
                "NdGridInterp::interpolate coords length",
            ));
        }

        // Find the left-cell index and the fractional position (weight) in
        // each dimension.
        let mut cell_indices = vec![0usize; self.ndim];
        let mut weights = vec![0.0f64; self.ndim];

        for d in 0..self.ndim {
            let ix = find_idx(&self.grids[d], coords[d]);
            let g = &self.grids[d];
            let t = (coords[d].max(g[0]).min(g[g.len() - 1]) - g[ix]) / (g[ix + 1] - g[ix]);
            cell_indices[d] = ix;
            weights[d] = t;
        }

        // Sum over all 2^ndim corners
        let n_corners = 1usize << self.ndim;
        let mut result = 0.0;

        for corner in 0..n_corners {
            let mut flat_idx = 0usize;
            let mut weight = 1.0f64;

            for d in 0..self.ndim {
                // Bit d of `corner` selects the upper (1) or lower (0) face
                let bit = (corner >> (self.ndim - 1 - d)) & 1;
                let coord_idx = cell_indices[d] + bit;
                let w = if bit == 0 {
                    1.0 - weights[d]
                } else {
                    weights[d]
                };
                weight *= w;
                flat_idx += coord_idx * self.strides[d];
            }

            // Guard against edge-case index overflow (should not happen with
            // correct clamping, but ensures safety).
            if flat_idx < self.values.len() {
                result += weight * self.values[flat_idx];
            }
        }

        Ok(result)
    }

    /// Batch interpolation at multiple query points.
    ///
    /// # Arguments
    ///
    /// * `points` – slice of coordinate slices, each of length `ndim`.
    pub fn interpolate_batch(&self, points: &[Vec<f64>]) -> InterpolateResult<Vec<f64>> {
        points.iter().map(|p| self.interpolate(p)).collect()
    }

    /// Return the total number of grid points (product of all grid lengths).
    pub fn total_points(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Binary-search helper: returns the largest index `lo` such that
/// `grid[lo] <= x`.  Input is clamped to `[grid[0], grid[n-1]]`.
fn find_idx(grid: &[f64], x: f64) -> usize {
    let n = grid.len();
    let x = x.max(grid[0]).min(grid[n - 1]);
    let mut lo = 0usize;
    let mut hi = n - 2;
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if grid[mid] <= x {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_product::trilinear::TrilinearInterp;

    // -----------------------------------------------------------------------
    // Helper builders
    // -----------------------------------------------------------------------

    fn make_2d_interp(f: impl Fn(f64, f64) -> f64, n: usize) -> NdGridInterp {
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut values: Vec<f64> = Vec::with_capacity(n * n);
        for &xi in &x {
            for &yj in &y {
                values.push(f(xi, yj));
            }
        }
        NdGridInterp::new(vec![x, y], values).expect("valid 2D")
    }

    fn make_3d_interp(
        f: impl Fn(f64, f64, f64) -> f64,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> NdGridInterp {
        let x: Vec<f64> = (0..nx).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..ny).map(|j| j as f64).collect();
        let z: Vec<f64> = (0..nz).map(|k| k as f64).collect();
        let mut values: Vec<f64> = Vec::with_capacity(nx * ny * nz);
        for &xi in &x {
            for &yj in &y {
                for &zk in &z {
                    values.push(f(xi, yj, zk));
                }
            }
        }
        NdGridInterp::new(vec![x, y, z], values).expect("valid 3D")
    }

    // -----------------------------------------------------------------------
    // 2D tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_nd_grid_2d_linear_exact() {
        // f(x,y) = 3x - 2y + 1 (linear → multilinear is exact)
        let interp = make_2d_interp(|x, y| 3.0 * x - 2.0 * y + 1.0, 5);
        let test_pts = [(0.5, 0.5), (1.3, 2.7), (0.0, 4.0), (3.9, 1.1)];
        for (x, y) in test_pts {
            let v = interp.interpolate(&[x, y]).expect("valid");
            let expected = 3.0 * x - 2.0 * y + 1.0;
            assert!(
                (v - expected).abs() < 1e-10,
                "2D linear at ({},{}): expected {}, got {}",
                x,
                y,
                expected,
                v
            );
        }
    }

    #[test]
    fn test_nd_grid_2d_exact_at_nodes() {
        let interp = make_2d_interp(|x, y| x * y, 4);
        for i in 0..4 {
            for j in 0..4 {
                let v = interp.interpolate(&[i as f64, j as f64]).expect("valid");
                let expected = (i * j) as f64;
                assert!(
                    (v - expected).abs() < 1e-12,
                    "node ({},{}): expected {}, got {}",
                    i,
                    j,
                    expected,
                    v
                );
            }
        }
    }

    #[test]
    fn test_nd_grid_2d_clamping() {
        let interp = make_2d_interp(|x, y| x + y, 3);
        // Points outside [0,2]^2 are clamped
        let v_lo = interp.interpolate(&[-1.0, -1.0]).expect("valid");
        let v_hi = interp.interpolate(&[5.0, 5.0]).expect("valid");
        assert!((v_lo - 0.0).abs() < 1e-12, "lo clamped: got {}", v_lo);
        assert!((v_hi - 4.0).abs() < 1e-12, "hi clamped: got {}", v_hi);
    }

    // -----------------------------------------------------------------------
    // 3D tests: NdGridInterp vs TrilinearInterp agreement
    // -----------------------------------------------------------------------

    #[test]
    fn test_nd_grid_3d_vs_trilinear_linear() {
        // f(x,y,z) = x + 2y + 3z
        let f = |x: f64, y: f64, z: f64| x + 2.0 * y + 3.0 * z;
        let nx = 4;
        let ny = 5;
        let nz = 3;

        // Build NdGridInterp
        let nd_interp = make_3d_interp(f, nx, ny, nz);

        // Build TrilinearInterp
        let x: Vec<f64> = (0..nx).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..ny).map(|j| j as f64).collect();
        let z: Vec<f64> = (0..nz).map(|k| k as f64).collect();
        let values_3d: Vec<Vec<Vec<f64>>> = x
            .iter()
            .map(|&xi| {
                y.iter()
                    .map(|&yj| z.iter().map(|&zk| f(xi, yj, zk)).collect())
                    .collect()
            })
            .collect();
        let tri_interp = TrilinearInterp::new(x, y, z, values_3d).expect("valid trilinear");

        let test_pts = [
            (0.5, 0.5, 0.5),
            (1.7, 2.3, 1.1),
            (0.1, 3.9, 0.8),
            (2.5, 1.0, 2.0),
        ];

        for (xi, yj, zk) in test_pts {
            let nd_val = nd_interp.interpolate(&[xi, yj, zk]).expect("nd_grid");
            let tri_val = tri_interp.interpolate(xi, yj, zk).expect("trilinear");
            assert!(
                (nd_val - tri_val).abs() < 1e-10,
                "NdGrid vs Trilinear disagree at ({},{},{}): {} vs {}",
                xi,
                yj,
                zk,
                nd_val,
                tri_val
            );
        }
    }

    #[test]
    fn test_nd_grid_3d_vs_trilinear_product() {
        // f(x,y,z) = x * y * z
        let f = |x: f64, y: f64, z: f64| x * y * z;
        let n = 4;

        let nd_interp = make_3d_interp(f, n, n, n);

        let g: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let values_3d: Vec<Vec<Vec<f64>>> = g
            .iter()
            .map(|&xi| {
                g.iter()
                    .map(|&yj| g.iter().map(|&zk| f(xi, yj, zk)).collect())
                    .collect()
            })
            .collect();
        let tri_interp = TrilinearInterp::new(g.clone(), g.clone(), g.clone(), values_3d)
            .expect("valid trilinear");

        let test_pts = [(0.5, 0.5, 0.5), (1.5, 2.5, 0.5), (0.3, 1.7, 2.2)];
        for (xi, yj, zk) in test_pts {
            let nd_val = nd_interp.interpolate(&[xi, yj, zk]).expect("nd_grid");
            let tri_val = tri_interp.interpolate(xi, yj, zk).expect("trilinear");
            assert!(
                (nd_val - tri_val).abs() < 1e-10,
                "NdGrid vs Trilinear disagree at ({},{},{}): {} vs {}",
                xi,
                yj,
                zk,
                nd_val,
                tri_val
            );
        }
    }

    // -----------------------------------------------------------------------
    // Higher-dimensional tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_nd_grid_4d_linear() {
        // 4D grid: f(x0,x1,x2,x3) = x0 + x1 + x2 + x3 (linear → exact)
        let n = 4;
        let grids: Vec<Vec<f64>> = (0..4).map(|_| (0..n).map(|i| i as f64).collect()).collect();
        let shape = [n, n, n, n];
        let mut values = vec![0.0f64; n * n * n * n];
        for i0 in 0..n {
            for i1 in 0..n {
                for i2 in 0..n {
                    for i3 in 0..n {
                        let idx = i0 * n * n * n + i1 * n * n + i2 * n + i3;
                        values[idx] = (i0 + i1 + i2 + i3) as f64;
                    }
                }
            }
        }
        let _ = shape; // used for construction clarity
        let interp = NdGridInterp::new(grids, values).expect("valid 4D");

        let test_pts: &[&[f64]] = &[&[0.5, 0.5, 0.5, 0.5], &[1.3, 0.7, 2.1, 1.9]];
        for pt in test_pts {
            let v = interp.interpolate(pt).expect("valid");
            let expected: f64 = pt.iter().sum();
            assert!(
                (v - expected).abs() < 1e-10,
                "4D linear at {:?}: expected {}, got {}",
                pt,
                expected,
                v
            );
        }
    }

    // -----------------------------------------------------------------------
    // Error handling
    // -----------------------------------------------------------------------

    #[test]
    fn test_nd_grid_empty_grids() {
        assert!(NdGridInterp::new(vec![], vec![]).is_err());
    }

    #[test]
    fn test_nd_grid_insufficient_points() {
        let grids = vec![vec![0.0], vec![0.0, 1.0]]; // first grid has only 1 point
        let values = vec![0.0, 0.0];
        assert!(NdGridInterp::new(grids, values).is_err());
    }

    #[test]
    fn test_nd_grid_wrong_values_length() {
        let grids = vec![vec![0.0, 1.0], vec![0.0, 1.0]];
        let values = vec![0.0, 1.0, 2.0]; // should be 4, not 3
        assert!(NdGridInterp::new(grids, values).is_err());
    }

    #[test]
    fn test_nd_grid_wrong_coords_length() {
        let interp = make_2d_interp(|x, y| x + y, 3);
        assert!(interp.interpolate(&[0.5]).is_err()); // 1 coord for 2D grid
    }

    #[test]
    fn test_nd_grid_batch() {
        let interp = make_2d_interp(|x, y| x + y, 4);
        let points = vec![vec![0.5, 0.5], vec![1.0, 2.0], vec![2.5, 1.5]];
        let results = interp.interpolate_batch(&points).expect("valid");
        assert_eq!(results.len(), 3);
        for (res, pt) in results.iter().zip(points.iter()) {
            let expected = pt[0] + pt[1];
            assert!((res - expected).abs() < 1e-10, "batch: got {}", res);
        }
    }
}
