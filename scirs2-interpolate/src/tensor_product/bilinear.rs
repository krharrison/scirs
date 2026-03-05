//! Bilinear interpolation on regular 2D grids.
//!
//! Provides efficient bilinear interpolation over rectilinear (non-uniform) 2D grids.
//! The grid is defined by sorted x and y coordinate vectors; values are stored as a
//! row-major matrix `values[i][j] = f(x_grid[i], y_grid[j])`.

use crate::error::{InterpolateError, InterpolateResult};

/// Bilinear interpolation on a regular (rectilinear) 2D grid.
///
/// The interpolated value at point (x, y) is computed as the weighted sum of
/// the four surrounding grid values using bilinear weights.
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::tensor_product::BilinearInterp;
///
/// let x = vec![0.0_f64, 1.0, 2.0];
/// let y = vec![0.0_f64, 1.0, 2.0];
/// let values = vec![
///     vec![0.0, 0.0, 0.0],
///     vec![0.0, 1.0, 2.0],
///     vec![0.0, 2.0, 4.0],
/// ];
/// let interp = BilinearInterp::new(x, y, values).expect("valid");
/// let v = interp.interpolate(0.5, 0.5).expect("valid");
/// assert!((v - 0.25).abs() < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct BilinearInterp {
    /// X grid coordinates (strictly increasing).
    pub x_grid: Vec<f64>,
    /// Y grid coordinates (strictly increasing).
    pub y_grid: Vec<f64>,
    /// Grid values: `values[i][j] = f(x_grid[i], y_grid[j])`.
    pub values: Vec<Vec<f64>>,
    /// Number of x grid points.
    pub nx: usize,
    /// Number of y grid points.
    pub ny: usize,
}

impl BilinearInterp {
    /// Construct a new `BilinearInterp`.
    ///
    /// # Arguments
    ///
    /// * `x_grid` – strictly increasing x coordinates, length ≥ 2.
    /// * `y_grid` – strictly increasing y coordinates, length ≥ 2.
    /// * `values` – `nx × ny` value matrix where `values[i][j] = f(x_grid[i], y_grid[j])`.
    ///
    /// # Errors
    ///
    /// Returns an error when fewer than 2 grid points are given in either dimension,
    /// when coordinates are not strictly increasing, or when the value matrix shape
    /// does not match the grid dimensions.
    pub fn new(
        x_grid: Vec<f64>,
        y_grid: Vec<f64>,
        values: Vec<Vec<f64>>,
    ) -> InterpolateResult<Self> {
        let nx = x_grid.len();
        let ny = y_grid.len();

        if nx < 2 {
            return Err(InterpolateError::insufficient_points(
                2,
                nx,
                "BilinearInterp x_grid",
            ));
        }
        if ny < 2 {
            return Err(InterpolateError::insufficient_points(
                2,
                ny,
                "BilinearInterp y_grid",
            ));
        }

        // Validate strictly increasing x
        for i in 1..nx {
            if x_grid[i] <= x_grid[i - 1] {
                return Err(InterpolateError::invalid_input(format!(
                    "BilinearInterp: x_grid not strictly increasing at index {}: {} <= {}",
                    i,
                    x_grid[i],
                    x_grid[i - 1]
                )));
            }
        }
        // Validate strictly increasing y
        for j in 1..ny {
            if y_grid[j] <= y_grid[j - 1] {
                return Err(InterpolateError::invalid_input(format!(
                    "BilinearInterp: y_grid not strictly increasing at index {}: {} <= {}",
                    j,
                    y_grid[j],
                    y_grid[j - 1]
                )));
            }
        }

        if values.len() != nx {
            return Err(InterpolateError::dimension_mismatch(
                nx,
                values.len(),
                "BilinearInterp: values row count vs nx",
            ));
        }
        for (i, row) in values.iter().enumerate() {
            if row.len() != ny {
                return Err(InterpolateError::dimension_mismatch(
                    ny,
                    row.len(),
                    &format!("BilinearInterp: values row {} length vs ny", i),
                ));
            }
        }

        Ok(Self {
            x_grid,
            y_grid,
            values,
            nx,
            ny,
        })
    }

    /// Interpolate at a single point `(x, y)`.
    ///
    /// Points outside the grid are clamped to the boundary.
    ///
    /// # Errors
    ///
    /// Never returns an error in practice (clamping is applied), but returns
    /// `InterpolateResult` for API consistency.
    pub fn interpolate(&self, x: f64, y: f64) -> InterpolateResult<f64> {
        // Clamp to grid bounds for boundary handling
        let xc = x.max(self.x_grid[0]).min(*self.x_grid.last().unwrap_or(&x));
        let yc = y.max(self.y_grid[0]).min(*self.y_grid.last().unwrap_or(&y));

        let ix = self.find_index(&self.x_grid, xc);
        let iy = self.find_index(&self.y_grid, yc);

        let x0 = self.x_grid[ix];
        let x1 = self.x_grid[ix + 1];
        let y0 = self.y_grid[iy];
        let y1 = self.y_grid[iy + 1];

        let tx = (xc - x0) / (x1 - x0);
        let ty = (yc - y0) / (y1 - y0);

        let f00 = self.values[ix][iy];
        let f01 = self.values[ix][iy + 1];
        let f10 = self.values[ix + 1][iy];
        let f11 = self.values[ix + 1][iy + 1];

        Ok((1.0 - tx) * (1.0 - ty) * f00
            + (1.0 - tx) * ty * f01
            + tx * (1.0 - ty) * f10
            + tx * ty * f11)
    }

    /// Interpolate on a grid of `(x, y)` evaluation points.
    ///
    /// Returns a `len(x_pts) × len(y_pts)` matrix of interpolated values.
    pub fn interpolate_grid(
        &self,
        x_pts: &[f64],
        y_pts: &[f64],
    ) -> InterpolateResult<Vec<Vec<f64>>> {
        x_pts
            .iter()
            .map(|&x| y_pts.iter().map(|&y| self.interpolate(x, y)).collect())
            .collect()
    }

    /// Binary search: find the largest index `lo` such that `grid[lo] <= x`.
    ///
    /// The input `x` is first clamped to `[grid[0], grid[n-1]]`.
    fn find_index(&self, grid: &[f64], x: f64) -> usize {
        let n = grid.len();
        // Clamp to valid range
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_xy_grid() -> BilinearInterp {
        // f(x, y) = x * y on [0,2] x [0,2]
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 2.0];
        let values = vec![
            vec![0.0 * 0.0, 0.0 * 1.0, 0.0 * 2.0],
            vec![1.0 * 0.0, 1.0 * 1.0, 1.0 * 2.0],
            vec![2.0 * 0.0, 2.0 * 1.0, 2.0 * 2.0],
        ];
        BilinearInterp::new(x, y, values).expect("valid")
    }

    #[test]
    fn test_bilinear_exact_at_grid_nodes() {
        let interp = make_xy_grid();
        for (i, &x) in interp.x_grid.clone().iter().enumerate() {
            for (j, &y) in interp.y_grid.clone().iter().enumerate() {
                let v = interp.interpolate(x, y).expect("valid");
                let expected = x * y;
                assert!(
                    (v - expected).abs() < 1e-12,
                    "At ({}, {}): expected {}, got {}",
                    x,
                    y,
                    expected,
                    v
                );
            }
        }
    }

    #[test]
    fn test_bilinear_midpoint() {
        let interp = make_xy_grid();
        // At (0.5, 0.5): bilinear interpolation of x*y is exact for linear
        // combination basis, but x*y is not bilinear (it's the product).
        // Bilinear interp at (0.5, 0.5): weighted corners of [0,1]x[0,1]
        // = 0.25*(f00+f01+f10+f11) = 0.25*(0+0+0+1) = 0.25
        let v = interp.interpolate(0.5, 0.5).expect("valid");
        assert!((v - 0.25).abs() < 1e-12, "midpoint: got {}", v);
    }

    #[test]
    fn test_bilinear_center() {
        let interp = make_xy_grid();
        // At (1.0, 1.0): exact grid node → 1.0
        let v = interp.interpolate(1.0, 1.0).expect("valid");
        assert!((v - 1.0).abs() < 1e-12, "center: got {}", v);
    }

    #[test]
    fn test_bilinear_clamp_boundary() {
        let interp = make_xy_grid();
        // Outside-boundary points are clamped
        let v_lo = interp.interpolate(-1.0, -1.0).expect("valid");
        let v_hi = interp.interpolate(3.0, 3.0).expect("valid");
        assert!((v_lo - 0.0).abs() < 1e-12);
        assert!((v_hi - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_bilinear_grid() {
        let interp = make_xy_grid();
        let xs = vec![0.5, 1.5];
        let ys = vec![0.5, 1.5];
        let grid = interp.interpolate_grid(&xs, &ys).expect("valid");
        assert_eq!(grid.len(), 2);
        assert_eq!(grid[0].len(), 2);
    }

    #[test]
    fn test_bilinear_insufficient_points() {
        let x = vec![0.0]; // only 1 point
        let y = vec![0.0, 1.0];
        let values = vec![vec![0.0, 1.0]];
        assert!(BilinearInterp::new(x, y, values).is_err());
    }

    #[test]
    fn test_bilinear_dimension_mismatch() {
        let x = vec![0.0, 1.0];
        let y = vec![0.0, 1.0];
        let values = vec![vec![0.0, 1.0]]; // wrong: only 1 row
        assert!(BilinearInterp::new(x, y, values).is_err());
    }

    #[test]
    fn test_bilinear_non_uniform_grid() {
        // Non-uniform grid: x = [0, 0.3, 1.0], y = [0, 0.7, 1.0]
        let x = vec![0.0, 0.3, 1.0];
        let y = vec![0.0, 0.7, 1.0];
        // f(x,y) = x + y (linear → exact bilinear interpolation)
        let values: Vec<Vec<f64>> = x
            .iter()
            .map(|&xi| y.iter().map(|&yj| xi + yj).collect())
            .collect();
        let interp = BilinearInterp::new(x, y, values).expect("valid");
        let v = interp.interpolate(0.15, 0.35).expect("valid");
        let expected = 0.15 + 0.35;
        assert!((v - expected).abs() < 1e-12, "linear fn: got {}", v);
    }
}
