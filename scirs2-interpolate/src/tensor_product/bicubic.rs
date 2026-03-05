//! Bicubic interpolation using tensor product cubic Hermite splines.
//!
//! The standard 16-coefficient bicubic patch is constructed from the function values,
//! x-partial derivatives, y-partial derivatives and the mixed partial (cross derivative)
//! at the four corners of each grid cell.  Partial derivatives are estimated from
//! the surrounding grid data using central differences (with forward/backward
//! differences at boundaries).
//!
//! The resulting interpolant is C1 continuous across cell boundaries.

use crate::error::{InterpolateError, InterpolateResult};

/// Bicubic interpolation on a rectilinear 2D grid.
///
/// Constructs a bicubic Hermite patch per grid cell using precomputed
/// 4×4 coefficient tensors.  Evaluation is O(1) per point after
/// O(nx·ny) preprocessing.
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::tensor_product::BicubicInterp;
///
/// let x: Vec<f64> = (0..=4).map(|i| i as f64).collect();
/// let y: Vec<f64> = (0..=4).map(|j| j as f64).collect();
/// let values: Vec<Vec<f64>> = x.iter()
///     .map(|&xi| y.iter().map(|&yj| xi * yj).collect())
///     .collect();
/// let interp = BicubicInterp::new(x, y, values).expect("valid");
/// let v = interp.interpolate(1.5, 2.5).expect("valid");
/// // x*y at (1.5, 2.5) = 3.75 — bicubic is exact for bilinear functions
/// assert!((v - 3.75).abs() < 1e-10, "got {}", v);
/// ```
#[derive(Debug, Clone)]
pub struct BicubicInterp {
    /// X grid coordinates (strictly increasing, length ≥ 4).
    pub x_grid: Vec<f64>,
    /// Y grid coordinates (strictly increasing, length ≥ 4).
    pub y_grid: Vec<f64>,
    /// Precomputed bicubic patch coefficients, indexed `[ix][iy][k][l]`.
    ///
    /// Evaluating the patch at normalised coordinates `(tx, ty) ∈ [0,1]²`:
    /// ```text
    /// p(tx, ty) = Σ_{k=0}^{3} Σ_{l=0}^{3}  coefficients[ix][iy][k][l] · tx^k · ty^l
    /// ```
    coefficients: Vec<Vec<[[f64; 4]; 4]>>,
    /// Number of x grid points.
    nx: usize,
    /// Number of y grid points.
    ny: usize,
}

impl BicubicInterp {
    /// Construct a new `BicubicInterp`.
    ///
    /// Requires at least 4 grid points in each dimension so that
    /// central-difference derivative estimates can be computed everywhere.
    ///
    /// # Arguments
    ///
    /// * `x_grid` – strictly increasing x coordinates, length ≥ 4.
    /// * `y_grid` – strictly increasing y coordinates, length ≥ 4.
    /// * `values` – `nx × ny` value matrix.
    ///
    /// # Errors
    ///
    /// Returns an error when < 4 grid points are supplied in either dimension,
    /// when coordinates are not strictly increasing, or when the value matrix
    /// shape does not match the grid dimensions.
    pub fn new(
        x_grid: Vec<f64>,
        y_grid: Vec<f64>,
        values: Vec<Vec<f64>>,
    ) -> InterpolateResult<Self> {
        let nx = x_grid.len();
        let ny = y_grid.len();

        if nx < 4 {
            return Err(InterpolateError::insufficient_points(
                4,
                nx,
                "BicubicInterp x_grid",
            ));
        }
        if ny < 4 {
            return Err(InterpolateError::insufficient_points(
                4,
                ny,
                "BicubicInterp y_grid",
            ));
        }

        for i in 1..nx {
            if x_grid[i] <= x_grid[i - 1] {
                return Err(InterpolateError::invalid_input(format!(
                    "BicubicInterp: x_grid not strictly increasing at index {}: {} <= {}",
                    i,
                    x_grid[i],
                    x_grid[i - 1]
                )));
            }
        }
        for j in 1..ny {
            if y_grid[j] <= y_grid[j - 1] {
                return Err(InterpolateError::invalid_input(format!(
                    "BicubicInterp: y_grid not strictly increasing at index {}: {} <= {}",
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
                "BicubicInterp: values row count vs nx",
            ));
        }
        for (i, row) in values.iter().enumerate() {
            if row.len() != ny {
                return Err(InterpolateError::dimension_mismatch(
                    ny,
                    row.len(),
                    &format!("BicubicInterp: values row {} length vs ny", i),
                ));
            }
        }

        let coefficients = Self::build_coefficients(&x_grid, &y_grid, &values, nx, ny);

        Ok(Self {
            x_grid,
            y_grid,
            coefficients,
            nx,
            ny,
        })
    }

    // -----------------------------------------------------------------------
    // Coefficient construction
    // -----------------------------------------------------------------------

    /// Pre-compute the 4×4 bicubic Hermite coefficients for every patch.
    ///
    /// Uses the standard Ferguson / Coons construction:
    /// given `f`, `fx`, `fy`, `fxy` at the four cell corners, the 16
    /// coefficients are computed by inverting the 4×4 Hermite matrix `M`.
    ///
    /// Reference: Numerical Recipes § 3.6 ("Bicubic Interpolation").
    fn build_coefficients(
        x: &[f64],
        y: &[f64],
        f: &[Vec<f64>],
        nx: usize,
        ny: usize,
    ) -> Vec<Vec<[[f64; 4]; 4]>> {
        // Estimate all first- and mixed-partial derivatives on the grid.
        let fx = Self::x_derivatives(x, f, nx, ny);
        let fy = Self::y_derivatives(y, f, nx, ny);
        let fxy = Self::xy_derivatives(x, y, f, nx, ny);

        let mut coeffs = vec![vec![[[0.0f64; 4]; 4]; ny - 1]; nx - 1];

        for i in 0..nx - 1 {
            let hx = x[i + 1] - x[i]; // cell width  in x
            for j in 0..ny - 1 {
                let hy = y[j + 1] - y[j]; // cell height in y

                // Collect the 16 input values (f, fx*hx, fy*hy, fxy*hx*hy at
                // the 4 corners) — stored row-major as required by the
                // Hermite inversion formula.
                //
                // Corner ordering: (i,j), (i+1,j), (i,j+1), (i+1,j+1)
                let fv = [f[i][j], f[i + 1][j], f[i][j + 1], f[i + 1][j + 1]];
                // Derivatives scaled to the unit square [0,1]^2
                let fxv = [
                    fx[i][j] * hx,
                    fx[i + 1][j] * hx,
                    fx[i][j + 1] * hx,
                    fx[i + 1][j + 1] * hx,
                ];
                let fyv = [
                    fy[i][j] * hy,
                    fy[i + 1][j] * hy,
                    fy[i][j + 1] * hy,
                    fy[i + 1][j + 1] * hy,
                ];
                let fxyv = [
                    fxy[i][j] * hx * hy,
                    fxy[i + 1][j] * hx * hy,
                    fxy[i][j + 1] * hx * hy,
                    fxy[i + 1][j + 1] * hx * hy,
                ];

                // Apply the 4×4 Hermite inversion (analytic inverse of
                // M ⊗ M where M is the standard cubic Hermite matrix).
                coeffs[i][j] = Self::hermite_coefficients(&fv, &fxv, &fyv, &fxyv);
            }
        }

        coeffs
    }

    /// Compute the 4×4 bicubic coefficient matrix from Hermite boundary data.
    ///
    /// The 16 data values are transformed by `A = M · X · M^T` where
    ///
    /// ```text
    /// M = [[ 1,  0,  0,  0],
    ///      [ 0,  0,  1,  0],
    ///      [-3,  3, -2, -1],
    ///      [ 2, -2,  1,  1]]
    /// ```
    ///
    /// is the standard cubic Hermite basis-change matrix.
    ///
    /// Input order for each array: corners (0,0), (1,0), (0,1), (1,1) on the
    /// unit square.
    fn hermite_coefficients(
        fv: &[f64; 4],
        fxv: &[f64; 4],
        fyv: &[f64; 4],
        fxyv: &[f64; 4],
    ) -> [[f64; 4]; 4] {
        // Assemble the 4×4 input matrix X (rows: f, fx, fy, fxy;
        // columns: corners ordered as (0,0),(1,0),(0,1),(1,1)).
        //
        // X = [[f00,   f10,   f01,   f11  ],
        //      [fx00,  fx10,  fx01,  fx11 ],
        //      [fy00,  fy10,  fy01,  fy11 ],
        //      [fxy00, fxy10, fxy01, fxy11]]
        //
        // Then  A = M · X · M^T.

        // The Hermite matrix M (acts on column vector [p(0), p(1), p'(0), p'(1)]):
        // M = [[ 1,  0,  0,  0],
        //      [ 0,  0,  1,  0],
        //      [-3,  3, -2, -1],
        //      [ 2, -2,  1,  1]]
        //
        // We compute directly by matrix multiplication.

        // Step 1: form the 4×4 input matrix following Numerical Recipes 3.6.
        //
        // The wt matrix transforms 1D Hermite data [p(0), p(1), p'(0), p'(1)]
        // into polynomial coefficients [a0, a1, a2, a3].
        //
        // For the 2D case  c[k][l] = Σ_{r,s}  wt[k][r] * d[r][s] * wt[l][s]
        // the left multiplication acts on the x-direction (rows), so r indexes
        // [f(x=0,·), f(x=1,·), fx(x=0,·), fx(x=1,·)].
        // The right multiplication acts on the y-direction (columns), so s indexes
        // [f(·,y=0), f(·,y=1), fy(·,y=0), fy(·,y=1)].
        //
        //   d = [[ f(0,0),   f(0,1),   fy(0,0),  fy(0,1) ],
        //        [ f(1,0),   f(1,1),   fy(1,0),  fy(1,1) ],
        //        [ fx(0,0),  fx(0,1),  fxy(0,0), fxy(0,1)],
        //        [ fx(1,0),  fx(1,1),  fxy(1,0), fxy(1,1)]]
        let d: [[f64; 4]; 4] = [
            [fv[0], fv[2], fyv[0], fyv[2]],
            [fv[1], fv[3], fyv[1], fyv[3]],
            [fxv[0], fxv[2], fxyv[0], fxyv[2]],
            [fxv[1], fxv[3], fxyv[1], fxyv[3]],
        ];

        // The inverse cubic Hermite matrix M^{-1} maps boundary data
        // [p(0), p(1), p'(0), p'(1)] to polynomial coefficients [a0, a1, a2, a3].
        //
        // M^{-1} = [[ 1,  0,  0,  0],
        //           [ 0,  0,  1,  0],
        //           [-3,  3, -2, -1],
        //           [ 2, -2,  1,  1]]
        //
        // The 2D formula is:  c = M^{-1} · d · (M^{-1})^T
        // which we compute as  tmp = wt · d, then  c = tmp · wt^T.
        let wt: [[f64; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-3.0, 3.0, -2.0, -1.0],
            [2.0, -2.0, 1.0, 1.0],
        ];

        // tmp = wt · d   (4×4)
        let mut tmp = [[0.0f64; 4]; 4];
        for k in 0..4 {
            for s in 0..4 {
                for r in 0..4 {
                    tmp[k][s] += wt[k][r] * d[r][s];
                }
            }
        }

        // c = tmp · wt^T   (4×4)
        let mut c = [[0.0f64; 4]; 4];
        for k in 0..4 {
            for l in 0..4 {
                for s in 0..4 {
                    c[k][l] += tmp[k][s] * wt[l][s];
                }
            }
        }

        c
    }

    // -----------------------------------------------------------------------
    // Finite-difference derivative estimators
    // -----------------------------------------------------------------------

    /// Compute the derivative of a cubic polynomial interpolating 4 points
    /// at one of those points.
    ///
    /// Given points `(xs[0], fs[0]), ..., (xs[3], fs[3])`, returns the
    /// derivative of the unique cubic polynomial through them, evaluated
    /// at `xs[at]`.  This is exact for polynomials of degree <= 3.
    fn lagrange4_deriv(xs: &[f64; 4], fs: &[f64; 4], at: usize) -> f64 {
        let xp = xs[at];
        // Derivative of Lagrange interpolant at xp:
        //   L'(xp) = Σ_j f_j · Σ_{m != j} [ Π_{k != j, k != m} (xp - x_k) / Π_{k != j} (x_j - x_k) ]
        //
        // Since xp = xs[at], most (xp - x_k) terms vanish when k = at,
        // simplifying the product significantly.
        let mut result = 0.0;
        for j in 0..4 {
            // Denominator: product of (x_j - x_k) for k != j
            let mut denom = 1.0;
            for k in 0..4 {
                if k != j {
                    denom *= xs[j] - xs[k];
                }
            }

            // Numerator: d/dx [Π_{k != j} (x - x_k)] evaluated at x = xp
            // = Σ_{m != j} Π_{k != j, k != m} (xp - x_k)
            let mut numer_sum = 0.0;
            for m in 0..4 {
                if m == j {
                    continue;
                }
                let mut prod = 1.0;
                for k in 0..4 {
                    if k != j && k != m {
                        prod *= xp - xs[k];
                    }
                }
                numer_sum += prod;
            }

            result += fs[j] * numer_sum / denom;
        }
        result
    }

    /// Select the 4-point stencil centered on index `i` in a grid of size `n`.
    ///
    /// Returns the starting index `s` such that the stencil covers `s..s+4`.
    fn stencil_start(i: usize, n: usize) -> usize {
        // Try to center: i should be at position 1 or 2 in the stencil.
        // For i=0 => s=0, for i=n-1 => s=n-4, otherwise s=max(0, i-1).
        if i <= 1 {
            0
        } else if i >= n - 2 {
            n - 4
        } else {
            i - 1
        }
    }

    /// Estimate ∂f/∂x at every grid node using 4-point Lagrange
    /// differentiation (exact for polynomials of degree <= 3).
    fn x_derivatives(x: &[f64], f: &[Vec<f64>], nx: usize, ny: usize) -> Vec<Vec<f64>> {
        let mut dx = vec![vec![0.0f64; ny]; nx];
        for i in 0..nx {
            let s = Self::stencil_start(i, nx);
            let xs = [x[s], x[s + 1], x[s + 2], x[s + 3]];
            let at = i - s;
            for j in 0..ny {
                let fs = [f[s][j], f[s + 1][j], f[s + 2][j], f[s + 3][j]];
                dx[i][j] = Self::lagrange4_deriv(&xs, &fs, at);
            }
        }
        dx
    }

    /// Estimate ∂f/∂y at every grid node using 4-point Lagrange
    /// differentiation (exact for polynomials of degree <= 3).
    fn y_derivatives(y: &[f64], f: &[Vec<f64>], nx: usize, ny: usize) -> Vec<Vec<f64>> {
        let mut dy = vec![vec![0.0f64; ny]; nx];
        for i in 0..nx {
            for j in 0..ny {
                let s = Self::stencil_start(j, ny);
                let ys = [y[s], y[s + 1], y[s + 2], y[s + 3]];
                let fs = [f[i][s], f[i][s + 1], f[i][s + 2], f[i][s + 3]];
                dy[i][j] = Self::lagrange4_deriv(&ys, &fs, j - s);
            }
        }
        dy
    }

    /// Estimate ∂²f/∂x∂y at every grid node.
    ///
    /// Computed as the x-derivative of the y-derivative array (or equivalently
    /// y-derivative of the x-derivative array — the two give the same result
    /// for smooth functions).
    fn xy_derivatives(x: &[f64], y: &[f64], f: &[Vec<f64>], nx: usize, ny: usize) -> Vec<Vec<f64>> {
        // First compute fy, then differentiate that in x.
        let fy = Self::y_derivatives(y, f, nx, ny);
        Self::x_derivatives(x, &fy, nx, ny)
    }

    // -----------------------------------------------------------------------
    // Evaluation
    // -----------------------------------------------------------------------

    /// Interpolate at a single point `(x, y)`.
    ///
    /// Points outside the grid are clamped to the boundary cell.
    pub fn interpolate(&self, x: f64, y: f64) -> InterpolateResult<f64> {
        // Clamp inputs to grid bounds so that tx/ty stay in [0, 1].
        let x = x.max(self.x_grid[0]).min(self.x_grid[self.nx - 1]);
        let y = y.max(self.y_grid[0]).min(self.y_grid[self.ny - 1]);

        let ix = Self::find_index(&self.x_grid, x);
        let iy = Self::find_index(&self.y_grid, y);

        let tx = (x - self.x_grid[ix]) / (self.x_grid[ix + 1] - self.x_grid[ix]);
        let ty = (y - self.y_grid[iy]) / (self.y_grid[iy + 1] - self.y_grid[iy]);

        let c = &self.coefficients[ix][iy];
        let mut val = 0.0;
        for k in 0..4 {
            for l in 0..4 {
                val += c[k][l] * tx.powi(k as i32) * ty.powi(l as i32);
            }
        }
        Ok(val)
    }

    /// Interpolate on a grid of evaluation points, returning an `nx × ny` matrix.
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

    /// Binary-search for the left cell index in `grid` such that
    /// `grid[lo] <= x <= grid[lo+1]`.  Input is clamped to `[grid[0], grid[n-1]]`.
    fn find_index(grid: &[f64], x: f64) -> usize {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| a + (b - a) * (i as f64) / ((n - 1) as f64))
            .collect()
    }

    fn make_grid(nx: usize, ny: usize) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
        let x = linspace(0.0, 3.0, nx);
        let y = linspace(0.0, 3.0, ny);
        let values: Vec<Vec<f64>> = x
            .iter()
            .map(|&xi| y.iter().map(|&yj| xi * yj).collect())
            .collect();
        (x, y, values)
    }

    #[test]
    fn test_bicubic_exact_nodes() {
        let (x, y, values) = make_grid(5, 5);
        let interp = BicubicInterp::new(x.clone(), y.clone(), values).expect("valid");
        for &xi in &x {
            for &yj in &y {
                let v = interp.interpolate(xi, yj).expect("valid");
                let expected = xi * yj;
                assert!(
                    (v - expected).abs() < 1e-10,
                    "At ({},{}) expected {} got {}",
                    xi,
                    yj,
                    expected,
                    v
                );
            }
        }
    }

    #[test]
    fn test_bicubic_linear_exact() {
        // f(x,y) = 2x + 3y + 1 (linear in x and y — bicubic is exact)
        let x = linspace(0.0, 4.0, 6);
        let y = linspace(0.0, 4.0, 6);
        let values: Vec<Vec<f64>> = x
            .iter()
            .map(|&xi| y.iter().map(|&yj| 2.0 * xi + 3.0 * yj + 1.0).collect())
            .collect();
        let interp = BicubicInterp::new(x, y, values).expect("valid");

        let test_pts = [(0.7, 1.3), (1.5, 2.5), (2.1, 0.9), (3.3, 3.7)];
        for (xi, yj) in test_pts {
            let v = interp.interpolate(xi, yj).expect("valid");
            let expected = 2.0 * xi + 3.0 * yj + 1.0;
            assert!(
                (v - expected).abs() < 1e-8,
                "linear at ({},{}): expected {}, got {}",
                xi,
                yj,
                expected,
                v
            );
        }
    }

    #[test]
    fn test_bicubic_insufficient_points() {
        let x = vec![0.0, 1.0, 2.0]; // only 3 < 4
        let y = linspace(0.0, 2.0, 5);
        let values: Vec<Vec<f64>> = x
            .iter()
            .map(|&xi| y.iter().map(|&yj| xi + yj).collect())
            .collect();
        assert!(BicubicInterp::new(x, y, values).is_err());
    }

    #[test]
    fn test_bicubic_cubic_polynomial() {
        // f(x,y) = x^3 — a cubic function, bicubic should reproduce exactly
        // on a sufficiently fine grid.
        let x = linspace(0.0, 2.0, 8);
        let y = linspace(0.0, 2.0, 8);
        let values: Vec<Vec<f64>> = x
            .iter()
            .map(|&xi| y.iter().map(|_yj| xi * xi * xi).collect())
            .collect();
        let interp = BicubicInterp::new(x, y, values).expect("valid");
        // Test at a few interior points
        for &xi in &[0.25, 0.75, 1.25, 1.75f64] {
            let v = interp.interpolate(xi, 1.0).expect("valid");
            let expected = xi * xi * xi;
            assert!(
                (v - expected).abs() < 1e-4,
                "x^3 at ({},1): expected {}, got {}",
                xi,
                expected,
                v
            );
        }
    }

    #[test]
    fn test_bicubic_clamping() {
        let (x, y, values) = make_grid(5, 5);
        let interp = BicubicInterp::new(x, y, values).expect("valid");
        // Should not panic — clamped to boundary
        let _v = interp.interpolate(-1.0, -1.0).expect("valid");
        let _v2 = interp.interpolate(10.0, 10.0).expect("valid");
    }
}
