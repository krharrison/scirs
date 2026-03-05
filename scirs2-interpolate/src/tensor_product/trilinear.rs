//! Trilinear interpolation on regular 3D grids.
//!
//! Trilinear interpolation is the 3D analogue of bilinear interpolation.
//! Given a rectilinear grid defined by three sorted coordinate vectors
//! (`x_grid`, `y_grid`, `z_grid`) and a value tensor
//! `values[i][j][k] = f(x_grid[i], y_grid[j], z_grid[k])`,
//! the interpolated value at an arbitrary point `(x, y, z)` is a
//! trilinear combination of the eight surrounding voxel corners.

use crate::error::{InterpolateError, InterpolateResult};

/// Trilinear interpolation on a rectilinear 3D grid.
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::tensor_product::TrilinearInterp;
///
/// let x = vec![0.0_f64, 1.0];
/// let y = vec![0.0_f64, 1.0];
/// let z = vec![0.0_f64, 1.0];
/// // f(x,y,z) = x + y + z
/// let values = vec![
///     vec![vec![0.0, 1.0], vec![1.0, 2.0]],
///     vec![vec![1.0, 2.0], vec![2.0, 3.0]],
/// ];
/// let interp = TrilinearInterp::new(x, y, z, values).expect("valid");
/// let v = interp.interpolate(0.5, 0.5, 0.5).expect("valid");
/// assert!((v - 1.5).abs() < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct TrilinearInterp {
    /// X grid coordinates (strictly increasing).
    pub x_grid: Vec<f64>,
    /// Y grid coordinates (strictly increasing).
    pub y_grid: Vec<f64>,
    /// Z grid coordinates (strictly increasing).
    pub z_grid: Vec<f64>,
    /// Value tensor: `values[i][j][k] = f(x_grid[i], y_grid[j], z_grid[k])`.
    pub values: Vec<Vec<Vec<f64>>>,
    /// Number of x grid points.
    nx: usize,
    /// Number of y grid points.
    ny: usize,
    /// Number of z grid points.
    nz: usize,
}

impl TrilinearInterp {
    /// Construct a new `TrilinearInterp`.
    ///
    /// # Arguments
    ///
    /// * `x_grid` – strictly increasing x coordinates, length ≥ 2.
    /// * `y_grid` – strictly increasing y coordinates, length ≥ 2.
    /// * `z_grid` – strictly increasing z coordinates, length ≥ 2.
    /// * `values` – value tensor with shape `[nx][ny][nz]`.
    ///
    /// # Errors
    ///
    /// Returns an error when fewer than 2 points are given in any dimension,
    /// when coordinates are not strictly increasing, or when the value tensor
    /// shape does not match the grid dimensions.
    pub fn new(
        x_grid: Vec<f64>,
        y_grid: Vec<f64>,
        z_grid: Vec<f64>,
        values: Vec<Vec<Vec<f64>>>,
    ) -> InterpolateResult<Self> {
        let nx = x_grid.len();
        let ny = y_grid.len();
        let nz = z_grid.len();

        if nx < 2 {
            return Err(InterpolateError::insufficient_points(
                2,
                nx,
                "TrilinearInterp x_grid",
            ));
        }
        if ny < 2 {
            return Err(InterpolateError::insufficient_points(
                2,
                ny,
                "TrilinearInterp y_grid",
            ));
        }
        if nz < 2 {
            return Err(InterpolateError::insufficient_points(
                2,
                nz,
                "TrilinearInterp z_grid",
            ));
        }

        // Validate strictly increasing
        for i in 1..nx {
            if x_grid[i] <= x_grid[i - 1] {
                return Err(InterpolateError::invalid_input(format!(
                    "TrilinearInterp: x_grid not strictly increasing at {}",
                    i
                )));
            }
        }
        for j in 1..ny {
            if y_grid[j] <= y_grid[j - 1] {
                return Err(InterpolateError::invalid_input(format!(
                    "TrilinearInterp: y_grid not strictly increasing at {}",
                    j
                )));
            }
        }
        for k in 1..nz {
            if z_grid[k] <= z_grid[k - 1] {
                return Err(InterpolateError::invalid_input(format!(
                    "TrilinearInterp: z_grid not strictly increasing at {}",
                    k
                )));
            }
        }

        // Validate shape of values
        if values.len() != nx {
            return Err(InterpolateError::dimension_mismatch(
                nx,
                values.len(),
                "TrilinearInterp: values x-dimension",
            ));
        }
        for (i, yz_plane) in values.iter().enumerate() {
            if yz_plane.len() != ny {
                return Err(InterpolateError::dimension_mismatch(
                    ny,
                    yz_plane.len(),
                    &format!("TrilinearInterp: values y-dimension at x-index {}", i),
                ));
            }
            for (j, z_row) in yz_plane.iter().enumerate() {
                if z_row.len() != nz {
                    return Err(InterpolateError::dimension_mismatch(
                        nz,
                        z_row.len(),
                        &format!("TrilinearInterp: values z-dimension at [{},{}]", i, j),
                    ));
                }
            }
        }

        Ok(Self {
            x_grid,
            y_grid,
            z_grid,
            values,
            nx,
            ny,
            nz,
        })
    }

    /// Interpolate at a single point `(x, y, z)`.
    ///
    /// Points outside the grid are clamped to the boundary.
    pub fn interpolate(&self, x: f64, y: f64, z: f64) -> InterpolateResult<f64> {
        // Clamp inputs to grid bounds so that tx/ty/tz stay in [0, 1].
        let x = x.max(self.x_grid[0]).min(self.x_grid[self.nx - 1]);
        let y = y.max(self.y_grid[0]).min(self.y_grid[self.ny - 1]);
        let z = z.max(self.z_grid[0]).min(self.z_grid[self.nz - 1]);

        let ix = find_index(&self.x_grid, x);
        let iy = find_index(&self.y_grid, y);
        let iz = find_index(&self.z_grid, z);

        let tx = (x - self.x_grid[ix]) / (self.x_grid[ix + 1] - self.x_grid[ix]);
        let ty = (y - self.y_grid[iy]) / (self.y_grid[iy + 1] - self.y_grid[iy]);
        let tz = (z - self.z_grid[iz]) / (self.z_grid[iz + 1] - self.z_grid[iz]);

        let v = &self.values;
        // Trilinear combination of 8 voxel corners
        let val = (1.0 - tx) * (1.0 - ty) * (1.0 - tz) * v[ix][iy][iz]
            + (1.0 - tx) * (1.0 - ty) * tz * v[ix][iy][iz + 1]
            + (1.0 - tx) * ty * (1.0 - tz) * v[ix][iy + 1][iz]
            + (1.0 - tx) * ty * tz * v[ix][iy + 1][iz + 1]
            + tx * (1.0 - ty) * (1.0 - tz) * v[ix + 1][iy][iz]
            + tx * (1.0 - ty) * tz * v[ix + 1][iy][iz + 1]
            + tx * ty * (1.0 - tz) * v[ix + 1][iy + 1][iz]
            + tx * ty * tz * v[ix + 1][iy + 1][iz + 1];

        Ok(val)
    }

    /// Return dimensions `(nx, ny, nz)` of the grid.
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }
}

/// Binary-search helper: returns the largest index `lo` such that
/// `grid[lo] <= x`.  Input is clamped to `[grid[0], grid[n-1]]`.
pub(crate) fn find_index(grid: &[f64], x: f64) -> usize {
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

    fn build_unit_cube_interp(f: impl Fn(f64, f64, f64) -> f64) -> TrilinearInterp {
        let g = vec![0.0, 1.0];
        let values: Vec<Vec<Vec<f64>>> = g
            .iter()
            .map(|&x| {
                g.iter()
                    .map(|&y| g.iter().map(|&z| f(x, y, z)).collect())
                    .collect()
            })
            .collect();
        TrilinearInterp::new(g.clone(), g.clone(), g.clone(), values).expect("valid")
    }

    #[test]
    fn test_trilinear_linear_exact() {
        // f(x,y,z) = x + y + z (linear → trilinear is exact)
        let interp = build_unit_cube_interp(|x, y, z| x + y + z);
        let test_pts = [
            (0.5, 0.5, 0.5, 1.5),
            (0.25, 0.75, 0.1, 1.1),
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0, 3.0),
        ];
        for (x, y, z, expected) in test_pts {
            let v = interp.interpolate(x, y, z).expect("valid");
            assert!(
                (v - expected).abs() < 1e-12,
                "at ({},{},{}): got {}",
                x,
                y,
                z,
                v
            );
        }
    }

    #[test]
    fn test_trilinear_center() {
        // At centre of unit cube, trilinear weight = 1/8 from each corner
        let interp = build_unit_cube_interp(|x, y, z| x * y * z);
        // f values at corners: only (1,1,1) gives 1, rest 0
        // trilinear at (0.5,0.5,0.5) = 0.5^3 = 0.125
        let v = interp.interpolate(0.5, 0.5, 0.5).expect("valid");
        assert!((v - 0.125).abs() < 1e-12, "got {}", v);
    }

    #[test]
    fn test_trilinear_multi_cell_grid() {
        // 3D grid: f(x,y,z) = 2x - y + 3z (linear → exact)
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 2.0];
        let z = vec![0.0, 1.0, 2.0];
        let values: Vec<Vec<Vec<f64>>> = x
            .iter()
            .map(|&xi| {
                y.iter()
                    .map(|&yj| z.iter().map(|&zk| 2.0 * xi - yj + 3.0 * zk).collect())
                    .collect()
            })
            .collect();
        let interp = TrilinearInterp::new(x, y, z, values).expect("valid");

        let test_pts = [(0.5, 0.5, 0.5), (1.3, 0.7, 1.9), (0.1, 1.5, 0.8)];
        for (xi, yj, zk) in test_pts {
            let v = interp.interpolate(xi, yj, zk).expect("valid");
            let expected = 2.0 * xi - yj + 3.0 * zk;
            assert!(
                (v - expected).abs() < 1e-12,
                "at ({},{},{}): expected {}, got {}",
                xi,
                yj,
                zk,
                expected,
                v
            );
        }
    }

    #[test]
    fn test_trilinear_clamping() {
        let interp = build_unit_cube_interp(|x, y, z| x + y + z);
        let v = interp.interpolate(-5.0, -5.0, -5.0).expect("valid");
        assert!((v - 0.0).abs() < 1e-12);
        let v2 = interp.interpolate(5.0, 5.0, 5.0).expect("valid");
        assert!((v2 - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_trilinear_insufficient_points() {
        let x = vec![0.0]; // only 1 point
        let y = vec![0.0, 1.0];
        let z = vec![0.0, 1.0];
        let values = vec![vec![vec![0.0, 1.0], vec![1.0, 2.0]]];
        assert!(TrilinearInterp::new(x, y, z, values).is_err());
    }

    #[test]
    fn test_trilinear_dimension_mismatch() {
        let x = vec![0.0, 1.0];
        let y = vec![0.0, 1.0];
        let z = vec![0.0, 1.0];
        // Wrong shape: only one x-slice instead of two
        let values = vec![vec![vec![0.0, 1.0], vec![1.0, 2.0]]];
        assert!(TrilinearInterp::new(x, y, z, values).is_err());
    }

    #[test]
    fn test_trilinear_non_uniform_grid() {
        let x = vec![0.0, 0.3, 1.0];
        let y = vec![0.0, 0.6, 1.0];
        let z = vec![0.0, 0.4, 1.0];
        // f(x,y,z) = x + y + z (linear → exact regardless of spacing)
        let values: Vec<Vec<Vec<f64>>> = x
            .iter()
            .map(|&xi| {
                y.iter()
                    .map(|&yj| z.iter().map(|&zk| xi + yj + zk).collect())
                    .collect()
            })
            .collect();
        let interp = TrilinearInterp::new(x, y, z, values).expect("valid");
        let v = interp.interpolate(0.15, 0.45, 0.2).expect("valid");
        let expected = 0.15 + 0.45 + 0.2;
        assert!((v - expected).abs() < 1e-12, "got {}", v);
    }
}
