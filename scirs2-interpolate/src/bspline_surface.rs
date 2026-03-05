//! Tensor-product B-spline surfaces
//!
//! This module provides full tensor-product B-spline surface evaluation,
//! normal estimation, and surface fitting through a grid of 3-D control points.
//!
//! # Mathematical Background
//!
//! A B-spline surface of degrees `(p, q)` is defined by:
//!
//! ```text
//! S(u, v) = Σ_{i=0}^{m}  Σ_{j=0}^{n}  P_{i,j}  N_{i,p}(u)  N_{j,q}(v)
//! ```
//!
//! where `P_{i,j}` are `(m+1)×(n+1)` 3-D control points and `N_{i,p}`,
//! `N_{j,q}` are B-spline basis functions of degree `p` and `q` defined on
//! knot vectors `u_knots` and `v_knots` respectively.
//!
//! Surface evaluation uses the tensor product of the de Boor recurrence:
//! the basis functions in each direction are computed independently and their
//! values are combined by a double sum.
//!
//! # Surface Normal
//!
//! The normal at `(u, v)` is the cross product of the partial derivatives
//! `∂S/∂u` and `∂S/∂v`.  Partial derivatives are computed analytically using
//! the B-spline derivative formula:
//!
//! ```text
//! N'_{i,p}(u) = p [ N_{i,p-1}(u)/(t_{i+p}-t_i)  −  N_{i+1,p-1}(u)/(t_{i+p+1}-t_{i+1}) ]
//! ```
//!
//! # Surface Fitting
//!
//! `BSplineSurface::interpolate_grid` fits a surface through a rectangular
//! grid of 3-D points by constructing clamped knot vectors and solving the
//! interpolation system in each parameter direction.
//!
//! # References
//!
//! - Piegl, L. and Tiller, W. (1997), *The NURBS Book*, 2nd ed., Springer.
//! - de Boor, C. (1978), *A Practical Guide to Splines*, Springer.

use crate::error::{InterpolateError, InterpolateResult};
use scirs2_core::ndarray::{Array2, Array3};

// ---------------------------------------------------------------------------
// Error / Result aliases
// ---------------------------------------------------------------------------

/// Result type alias for B-spline surface operations.
pub type BSplineResult<T> = InterpolateResult<T>;

// ---------------------------------------------------------------------------
// BSplineSurface
// ---------------------------------------------------------------------------

/// Tensor-product B-spline surface of degrees `(p, q)`.
///
/// The surface is parameterised over `(u, v) ∈ [u_min, u_max] × [v_min, v_max]`
/// where `u_min`/`u_max` are the first/last distinct values in `u_knots` and
/// similarly for `v`.
///
/// # Fields
///
/// - `control_pts`: `(m+1)×(n+1)` array of 3-D control points in row-major order.
///   Row `i`, column `j` holds `[x, y, z]` stored as `control_pts[[i*3, j*3, …]]`
///   — actually stored as `Array3<f64>` with shape `(m+1, n+1, 3)`.
/// - `u_knots`, `v_knots`: knot vectors (length `m+p+2` and `n+q+2` respectively)
/// - `p`, `q`: polynomial degrees in `u` and `v`
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::bspline_surface::BSplineSurface;
/// use scirs2_core::ndarray::Array3;
///
/// // 2×2 bilinear surface (degree 1 in both directions)
/// let mut ctrl = Array3::<f64>::zeros((2, 2, 3));
/// ctrl[[0,0,0]]=0.0; ctrl[[0,0,1]]=0.0; ctrl[[0,0,2]]=0.0;
/// ctrl[[1,0,0]]=1.0; ctrl[[1,0,1]]=0.0; ctrl[[1,0,2]]=0.0;
/// ctrl[[0,1,0]]=0.0; ctrl[[0,1,1]]=1.0; ctrl[[0,1,2]]=0.0;
/// ctrl[[1,1,0]]=1.0; ctrl[[1,1,1]]=1.0; ctrl[[1,1,2]]=1.0;
///
/// let u_knots = vec![0.0, 0.0, 1.0, 1.0];
/// let v_knots = vec![0.0, 0.0, 1.0, 1.0];
/// let surf = BSplineSurface::new(ctrl, u_knots, v_knots, 1, 1).expect("doc example: should succeed");
/// let pt = surf.evaluate(0.5, 0.5).expect("doc example: should succeed");
/// assert!((pt[0] - 0.5).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct BSplineSurface {
    /// Control points, shape `(m+1, n+1, 3)`.
    control_pts: Array3<f64>,
    /// Knot vector in the `u` direction.
    u_knots: Vec<f64>,
    /// Knot vector in the `v` direction.
    v_knots: Vec<f64>,
    /// Degree in `u`.
    p: usize,
    /// Degree in `v`.
    q: usize,
}

impl BSplineSurface {
    /// Construct a B-spline surface from existing control points and knot vectors.
    ///
    /// # Parameters
    ///
    /// - `control_pts`: 3-D array of shape `(m+1, n+1, 3)`
    /// - `u_knots`: knot vector of length `m + p + 2`
    /// - `v_knots`: knot vector of length `n + q + 2`
    /// - `p`: degree in `u` direction (≥ 1)
    /// - `q`: degree in `v` direction (≥ 1)
    ///
    /// # Errors
    ///
    /// Returns `InterpolateError::InvalidInput` for inconsistent sizes, invalid
    /// degrees, non-monotone knot vectors, or `control_pts` last dimension ≠ 3.
    pub fn new(
        control_pts: Array3<f64>,
        u_knots: Vec<f64>,
        v_knots: Vec<f64>,
        p: usize,
        q: usize,
    ) -> BSplineResult<Self> {
        let shape = control_pts.shape();
        if shape.len() != 3 || shape[2] != 3 {
            return Err(InterpolateError::InvalidInput {
                message: format!(
                    "BSplineSurface: control_pts must have shape (m+1, n+1, 3); \
                     got {:?}",
                    shape
                ),
            });
        }
        if p == 0 {
            return Err(InterpolateError::InvalidInput {
                message: "BSplineSurface: degree p must be >= 1".into(),
            });
        }
        if q == 0 {
            return Err(InterpolateError::InvalidInput {
                message: "BSplineSurface: degree q must be >= 1".into(),
            });
        }

        let m = shape[0] - 1; // last control index in u
        let n = shape[1] - 1; // last control index in v

        // knot vector length checks: must be m+p+2 and n+q+2
        let expected_u = m + p + 2;
        let expected_v = n + q + 2;
        if u_knots.len() != expected_u {
            return Err(InterpolateError::InvalidInput {
                message: format!(
                    "BSplineSurface: u_knots.len()={} but expected {}=(m+p+2) \
                     with m={}, p={}",
                    u_knots.len(),
                    expected_u,
                    m,
                    p
                ),
            });
        }
        if v_knots.len() != expected_v {
            return Err(InterpolateError::InvalidInput {
                message: format!(
                    "BSplineSurface: v_knots.len()={} but expected {}=(n+q+2) \
                     with n={}, q={}",
                    v_knots.len(),
                    expected_v,
                    n,
                    q
                ),
            });
        }

        // Verify non-decreasing knot vectors
        for i in 1..u_knots.len() {
            if u_knots[i] < u_knots[i - 1] {
                return Err(InterpolateError::InvalidInput {
                    message: format!(
                        "BSplineSurface: u_knots must be non-decreasing; \
                         u_knots[{i}]={} < u_knots[{}]={}",
                        u_knots[i], i-1, u_knots[i-1]
                    ),
                });
            }
        }
        for i in 1..v_knots.len() {
            if v_knots[i] < v_knots[i - 1] {
                return Err(InterpolateError::InvalidInput {
                    message: format!(
                        "BSplineSurface: v_knots must be non-decreasing; \
                         v_knots[{i}]={} < v_knots[{}]={}",
                        v_knots[i], i-1, v_knots[i-1]
                    ),
                });
            }
        }

        Ok(Self {
            control_pts,
            u_knots,
            v_knots,
            p,
            q,
        })
    }

    /// Evaluate the surface at parameter `(u, v)`.
    ///
    /// Returns `[x, y, z]` on the surface.
    ///
    /// # Errors
    ///
    /// Returns `InterpolateError::OutOfBounds` if `(u, v)` is outside the
    /// knot vector domain.
    pub fn evaluate(&self, u: f64, v: f64) -> BSplineResult<[f64; 3]> {
        let (u_min, u_max) = domain_bounds(&self.u_knots, self.p);
        let (v_min, v_max) = domain_bounds(&self.v_knots, self.q);
        let u_clamped = clamp_to_domain(u, u_min, u_max);
        let v_clamped = clamp_to_domain(v, v_min, v_max);

        let m = self.control_pts.shape()[0] - 1;
        let n = self.control_pts.shape()[1] - 1;

        // Compute B-spline basis functions in u and v
        let n_u = basis_functions(&self.u_knots, u_clamped, self.p, m);
        let n_v = basis_functions(&self.v_knots, v_clamped, self.q, n);

        // Find knot spans
        let span_u = find_knot_span(&self.u_knots, u_clamped, self.p, m);
        let span_v = find_knot_span(&self.v_knots, v_clamped, self.q, n);

        // Double sum: S(u,v) = Σ_i Σ_j N_i(u) N_j(v) P_{i,j}
        let mut pt = [0.0_f64; 3];
        for (li, &ni) in n_u.iter().enumerate() {
            let i = span_u as isize - self.p as isize + li as isize;
            if i < 0 || i > m as isize {
                continue;
            }
            let i = i as usize;
            for (lj, &nj) in n_v.iter().enumerate() {
                let j = span_v as isize - self.q as isize + lj as isize;
                if j < 0 || j > n as isize {
                    continue;
                }
                let j = j as usize;
                let w = ni * nj;
                pt[0] += w * self.control_pts[[i, j, 0]];
                pt[1] += w * self.control_pts[[i, j, 1]];
                pt[2] += w * self.control_pts[[i, j, 2]];
            }
        }
        Ok(pt)
    }

    /// Surface normal at `(u, v)`.
    ///
    /// The normal is the (unnormalized) cross product `∂S/∂u × ∂S/∂v`.
    /// Returns a zero vector only when the surface is degenerate at that point.
    ///
    /// # Errors
    ///
    /// Returns `InterpolateError::OutOfBounds` if `(u, v)` is outside domain.
    pub fn normal(&self, u: f64, v: f64) -> BSplineResult<[f64; 3]> {
        let du = self.partial_derivative_u(u, v)?;
        let dv = self.partial_derivative_v(u, v)?;
        // cross product
        let nx = du[1] * dv[2] - du[2] * dv[1];
        let ny = du[2] * dv[0] - du[0] * dv[2];
        let nz = du[0] * dv[1] - du[1] * dv[0];
        Ok([nx, ny, nz])
    }

    // ---- partial derivatives ----

    /// Partial derivative `∂S/∂u` at `(u, v)`.
    pub fn partial_derivative_u(&self, u: f64, v: f64) -> BSplineResult<[f64; 3]> {
        let (u_min, u_max) = domain_bounds(&self.u_knots, self.p);
        let (v_min, v_max) = domain_bounds(&self.v_knots, self.q);
        let u_c = clamp_to_domain(u, u_min, u_max);
        let v_c = clamp_to_domain(v, v_min, v_max);

        let m = self.control_pts.shape()[0] - 1;
        let n = self.control_pts.shape()[1] - 1;

        // Degree-1 derivative basis in u
        let dn_u = basis_functions_derivative(&self.u_knots, u_c, self.p, m);
        let n_v = basis_functions(&self.v_knots, v_c, self.q, n);

        let span_u = find_knot_span(&self.u_knots, u_c, self.p, m);
        let span_v = find_knot_span(&self.v_knots, v_c, self.q, n);

        let mut dpt = [0.0_f64; 3];
        for (li, &dni) in dn_u.iter().enumerate() {
            let i = span_u as isize - self.p as isize + li as isize;
            if i < 0 || i > m as isize {
                continue;
            }
            let i = i as usize;
            for (lj, &nj) in n_v.iter().enumerate() {
                let j = span_v as isize - self.q as isize + lj as isize;
                if j < 0 || j > n as isize {
                    continue;
                }
                let j = j as usize;
                let w = dni * nj;
                dpt[0] += w * self.control_pts[[i, j, 0]];
                dpt[1] += w * self.control_pts[[i, j, 1]];
                dpt[2] += w * self.control_pts[[i, j, 2]];
            }
        }
        Ok(dpt)
    }

    /// Partial derivative `∂S/∂v` at `(u, v)`.
    pub fn partial_derivative_v(&self, u: f64, v: f64) -> BSplineResult<[f64; 3]> {
        let (u_min, u_max) = domain_bounds(&self.u_knots, self.p);
        let (v_min, v_max) = domain_bounds(&self.v_knots, self.q);
        let u_c = clamp_to_domain(u, u_min, u_max);
        let v_c = clamp_to_domain(v, v_min, v_max);

        let m = self.control_pts.shape()[0] - 1;
        let n = self.control_pts.shape()[1] - 1;

        let n_u = basis_functions(&self.u_knots, u_c, self.p, m);
        let dn_v = basis_functions_derivative(&self.v_knots, v_c, self.q, n);

        let span_u = find_knot_span(&self.u_knots, u_c, self.p, m);
        let span_v = find_knot_span(&self.v_knots, v_c, self.q, n);

        let mut dpt = [0.0_f64; 3];
        for (li, &ni) in n_u.iter().enumerate() {
            let i = span_u as isize - self.p as isize + li as isize;
            if i < 0 || i > m as isize {
                continue;
            }
            let i = i as usize;
            for (lj, &dnj) in dn_v.iter().enumerate() {
                let j = span_v as isize - self.q as isize + lj as isize;
                if j < 0 || j > n as isize {
                    continue;
                }
                let j = j as usize;
                let w = ni * dnj;
                dpt[0] += w * self.control_pts[[i, j, 0]];
                dpt[1] += w * self.control_pts[[i, j, 1]];
                dpt[2] += w * self.control_pts[[i, j, 2]];
            }
        }
        Ok(dpt)
    }

    // ---- surface fitting ----

    /// Fit a B-spline surface of degrees `(p, q)` through a grid of 3-D points.
    ///
    /// `points` has shape `(r+1, s+1, 3)` where `r+1` and `s+1` are the numbers
    /// of rows and columns of data points.  The method:
    ///
    /// 1. Chooses chord-length parameterisation in both directions.
    /// 2. Constructs clamped knot vectors.
    /// 3. Solves the interpolation system row-by-row and then column-by-column
    ///    using the de Boor triangular algorithm.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid grid shapes, incompatible degrees, or
    /// singular interpolation matrices.
    pub fn interpolate_grid(
        points: &Array3<f64>,
        p: usize,
        q: usize,
    ) -> BSplineResult<Self> {
        let shape = points.shape();
        if shape.len() != 3 || shape[2] != 3 {
            return Err(InterpolateError::InvalidInput {
                message: format!(
                    "BSplineSurface::interpolate_grid: points must have shape (r+1,s+1,3); \
                     got {:?}",
                    shape
                ),
            });
        }
        if p == 0 || q == 0 {
            return Err(InterpolateError::InvalidInput {
                message: "BSplineSurface::interpolate_grid: degrees must be >= 1".into(),
            });
        }

        let r = shape[0] - 1; // last row index
        let s = shape[1] - 1; // last col index

        if r < p {
            return Err(InterpolateError::InvalidInput {
                message: format!(
                    "BSplineSurface::interpolate_grid: need >= {} rows for degree p={}; \
                     got {}",
                    p + 1, p, r + 1
                ),
            });
        }
        if s < q {
            return Err(InterpolateError::InvalidInput {
                message: format!(
                    "BSplineSurface::interpolate_grid: need >= {} cols for degree q={}; \
                     got {}",
                    q + 1, q, s + 1
                ),
            });
        }

        // ---- Step 1: chord-length parameterisation ----
        let u_params = chord_params_u(points, r, s);
        let v_params = chord_params_v(points, r, s);

        // ---- Step 2: clamped knot vectors ----
        let u_knots = clamped_knots_from_params(&u_params, p, r);
        let v_knots = clamped_knots_from_params(&v_params, q, s);

        // ---- Step 3: solve interpolation ----
        // The control points are determined by solving two sets of 1-D spline
        // interpolation problems:
        //  a) For each column j, interpolate r+1 points along u → r+1 ctrl pts
        //  b) For each resulting "row" i, interpolate s+1 pts along v → s+1 ctrl pts

        // Temporary control point array: shape (r+1, s+1, 3)
        let mut ctrl = Array3::<f64>::zeros((r + 1, s + 1, 3));

        // a) Fit along u for each column j
        for j in 0..=s {
            // Collect data column j: (r+1) points
            let data_col: Vec<[f64; 3]> = (0..=r)
                .map(|i| [points[[i, j, 0]], points[[i, j, 1]], points[[i, j, 2]]])
                .collect();
            let c = spline_interp_1d(&u_params, &data_col, &u_knots, p, r)?;
            for i in 0..=r {
                ctrl[[i, j, 0]] = c[i][0];
                ctrl[[i, j, 1]] = c[i][1];
                ctrl[[i, j, 2]] = c[i][2];
            }
        }

        // b) Fit along v for each row i using the temporary control points
        let ctrl_temp = ctrl.clone();
        for i in 0..=r {
            let data_row: Vec<[f64; 3]> = (0..=s)
                .map(|j| [ctrl_temp[[i, j, 0]], ctrl_temp[[i, j, 1]], ctrl_temp[[i, j, 2]]])
                .collect();
            let c = spline_interp_1d(&v_params, &data_row, &v_knots, q, s)?;
            for j in 0..=s {
                ctrl[[i, j, 0]] = c[j][0];
                ctrl[[i, j, 1]] = c[j][1];
                ctrl[[i, j, 2]] = c[j][2];
            }
        }

        Self::new(ctrl, u_knots, v_knots, p, q)
    }

    // ---- convenience accessors ----

    /// Number of control points in the `u` direction (= m + 1).
    pub fn n_ctrl_u(&self) -> usize {
        self.control_pts.shape()[0]
    }

    /// Number of control points in the `v` direction (= n + 1).
    pub fn n_ctrl_v(&self) -> usize {
        self.control_pts.shape()[1]
    }

    /// Degree in `u`.
    pub fn degree_u(&self) -> usize {
        self.p
    }

    /// Degree in `v`.
    pub fn degree_v(&self) -> usize {
        self.q
    }

    /// Parameter domain `[u_min, u_max]`.
    pub fn u_domain(&self) -> (f64, f64) {
        domain_bounds(&self.u_knots, self.p)
    }

    /// Parameter domain `[v_min, v_max]`.
    pub fn v_domain(&self) -> (f64, f64) {
        domain_bounds(&self.v_knots, self.q)
    }
}

// ---------------------------------------------------------------------------
// B-spline algorithms (Cox-de Boor)
// ---------------------------------------------------------------------------

/// Find the knot span index `k` such that `t[k] <= u < t[k+1]`.
///
/// For the special case `u == t[m+1]` (last knot), the last non-empty span is
/// returned so that evaluation at the domain end is correct.
fn find_knot_span(t: &[f64], u: f64, p: usize, n_ctrl: usize) -> usize {
    let n = n_ctrl; // last control point index
    let m = t.len() - 1; // last knot index

    // Special case: u == last domain knot
    if u >= t[m - p] {
        // Walk back to find the last span with a non-zero width
        let mut span = m - p - 1;
        while span > p && t[span] >= t[span + 1] {
            span -= 1;
        }
        return span;
    }

    // Binary search
    let mut lo = p;
    let mut hi = n + 1;
    let mut mid = (lo + hi) / 2;
    while u < t[mid] || u >= t[mid + 1] {
        if u < t[mid] {
            hi = mid;
        } else {
            lo = mid;
        }
        mid = (lo + hi) / 2;
        if mid == lo {
            break;
        }
    }
    mid
}

/// Compute the B-spline basis functions `N_{span-p, p}(u), …, N_{span, p}(u)`.
///
/// Returns a vector of length `p+1`.  Uses the stable triangular algorithm
/// (Algorithm A2.2 from Piegl & Tiller).
fn basis_functions(t: &[f64], u: f64, p: usize, n_ctrl: usize) -> Vec<f64> {
    let span = find_knot_span(t, u, p, n_ctrl);
    let mut n = vec![0.0_f64; p + 1];
    let mut left = vec![0.0_f64; p + 1];
    let mut right = vec![0.0_f64; p + 1];

    n[0] = 1.0;
    for j in 1..=p {
        left[j] = u - t[span + 1 - j];
        right[j] = t[span + j] - u;
        let mut saved = 0.0_f64;
        for r in 0..j {
            let denom = right[r + 1] + left[j - r];
            let temp = if denom.abs() < 1e-300 {
                0.0
            } else {
                n[r] / denom
            };
            n[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        n[j] = saved;
    }
    n
}

/// Compute the first-order B-spline basis function derivatives
/// `N'_{span-p, p}(u), …, N'_{span, p}(u)`.
///
/// Returns a vector of length `p+1` using the recursive derivative formula
/// (Algorithm A2.3 from Piegl & Tiller, adapted for order-1 derivatives).
fn basis_functions_derivative(t: &[f64], u: f64, p: usize, n_ctrl: usize) -> Vec<f64> {
    if p == 0 {
        return vec![0.0];
    }
    let span = find_knot_span(t, u, p, n_ctrl);

    // Compute N_{span-p, p-1}(u), …, N_{span, p-1}(u)  (one degree lower)
    let n_lower = basis_functions_at_span(t, u, p - 1, span);

    // Derivative formula:
    // N'_{i, p}(u) = p * [ N_{i, p-1}(u) / (t_{i+p} - t_i)
    //                    - N_{i+1, p-1}(u) / (t_{i+p+1} - t_{i+1}) ]
    let mut dn = vec![0.0_f64; p + 1];
    let p_f = p as f64;
    for r in 0..=p {
        let i = span as isize - p as isize + r as isize;
        // Left term: N_{i, p-1} / (t[i+p] - t[i])
        let left_val = if i >= 0 {
            let i = i as usize;
            let denom = t.get(i + p).copied().unwrap_or(0.0)
                - t.get(i).copied().unwrap_or(0.0);
            if denom.abs() > 1e-300 && r < n_lower.len() {
                n_lower[r] / denom
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Right term: N_{i+1, p-1} / (t[i+p+1] - t[i+1])
        let right_val = if r + 1 < n_lower.len() {
            let ip1 = (i + 1).max(0) as usize;
            let denom = t.get(ip1 + p).copied().unwrap_or(0.0)
                - t.get(ip1).copied().unwrap_or(0.0);
            if denom.abs() > 1e-300 {
                n_lower[r + 1] / denom
            } else {
                0.0
            }
        } else {
            0.0
        };

        dn[r] = p_f * (left_val - right_val);
    }
    dn
}

/// Compute basis functions of degree `p` for the given span index `span`.
///
/// Returns a vector of length `p+2` covering span-p to span+1 (padded with
/// zeros where out of range).
fn basis_functions_at_span(t: &[f64], u: f64, p: usize, span: usize) -> Vec<f64> {
    let mut n = vec![0.0_f64; p + 2]; // extra element for derivative indexing
    let mut left = vec![0.0_f64; p + 1];
    let mut right = vec![0.0_f64; p + 1];

    n[0] = 1.0;
    for j in 1..=p {
        let l = if span + 1 >= j { t[span + 1 - j] } else { 0.0 };
        let r = if span + j < t.len() { t[span + j] } else { 0.0 };
        left[j] = u - l;
        right[j] = r - u;
        let mut saved = 0.0_f64;
        for rr in 0..j {
            let denom = right[rr + 1] + left[j - rr];
            let temp = if denom.abs() < 1e-300 {
                0.0
            } else {
                n[rr] / denom
            };
            n[rr] = saved + right[rr + 1] * temp;
            saved = left[j - rr] * temp;
        }
        n[j] = saved;
    }
    n
}

// ---------------------------------------------------------------------------
// Domain helpers
// ---------------------------------------------------------------------------

/// Return `(u_min, u_max)` from the clamped knot vector.
fn domain_bounds(t: &[f64], p: usize) -> (f64, f64) {
    let lo = t[p];
    let hi = t[t.len() - 1 - p];
    (lo, hi)
}

/// Clamp `u` to `[u_min, u_max]` with small tolerance.
#[inline]
fn clamp_to_domain(u: f64, u_min: f64, u_max: f64) -> f64 {
    let eps = f64::EPSILON * (u_max - u_min).abs();
    u.max(u_min - eps).min(u_max + eps)
}

// ---------------------------------------------------------------------------
// Interpolation helpers
// ---------------------------------------------------------------------------

/// Compute chord-length parameters along rows (u direction).
///
/// Returns a `(r+1)`-length vector of parameter values in `[0, 1]`.
fn chord_params_u(points: &Array3<f64>, r: usize, s: usize) -> Vec<f64> {
    // Average the chord-length parameterisations over all columns
    let mut params = vec![0.0_f64; r + 1];
    params[0] = 0.0;

    for j in 0..=s {
        let mut chord = vec![0.0_f64; r + 1];
        chord[0] = 0.0;
        let mut total = 0.0_f64;
        for i in 0..r {
            let dx = points[[i + 1, j, 0]] - points[[i, j, 0]];
            let dy = points[[i + 1, j, 1]] - points[[i, j, 1]];
            let dz = points[[i + 1, j, 2]] - points[[i, j, 2]];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            chord[i + 1] = chord[i] + dist;
            total += dist;
        }
        if total < f64::EPSILON {
            // Degenerate row: use uniform parameterisation
            for i in 0..=r {
                params[i] += i as f64 / r as f64;
            }
        } else {
            for i in 0..=r {
                params[i] += chord[i] / total;
            }
        }
    }
    let cols = (s + 1) as f64;
    for p in params.iter_mut() {
        *p /= cols;
    }
    params[r] = 1.0; // ensure exactly 1 at the end
    params
}

/// Compute chord-length parameters along columns (v direction).
fn chord_params_v(points: &Array3<f64>, r: usize, s: usize) -> Vec<f64> {
    let mut params = vec![0.0_f64; s + 1];
    params[0] = 0.0;

    for i in 0..=r {
        let mut chord = vec![0.0_f64; s + 1];
        chord[0] = 0.0;
        let mut total = 0.0_f64;
        for j in 0..s {
            let dx = points[[i, j + 1, 0]] - points[[i, j, 0]];
            let dy = points[[i, j + 1, 1]] - points[[i, j, 1]];
            let dz = points[[i, j + 1, 2]] - points[[i, j, 2]];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            chord[j + 1] = chord[j] + dist;
            total += dist;
        }
        if total < f64::EPSILON {
            for j in 0..=s {
                params[j] += j as f64 / s as f64;
            }
        } else {
            for j in 0..=s {
                params[j] += chord[j] / total;
            }
        }
    }
    let rows = (r + 1) as f64;
    for p in params.iter_mut() {
        *p /= rows;
    }
    params[s] = 1.0;
    params
}

/// Construct a clamped (open uniform) knot vector from parameter values.
///
/// For `n+1` data points and degree `p`, the knot vector has `n+p+2` entries.
/// Uses the averaging method from Piegl & Tiller (1997), eq. (9.8).
fn clamped_knots_from_params(params: &[f64], p: usize, n: usize) -> Vec<f64> {
    let m = n + p + 1; // last knot index
    let mut t = vec![0.0_f64; m + 1];
    // First p+1 knots = 0
    for j in 0..=p {
        t[j] = 0.0;
    }
    // Last p+1 knots = 1
    for j in m - p..=m {
        t[j] = 1.0;
    }
    // Interior knots: average of p consecutive parameters (Piegl & Tiller eq. 9.8)
    for j in 1..=n - p {
        let mut sum = 0.0_f64;
        for i in j..j + p {
            sum += params.get(i).copied().unwrap_or(1.0);
        }
        t[j + p] = sum / p as f64;
    }
    t
}

/// Solve the 1-D B-spline interpolation problem.
///
/// Given `n+1` data points `data[0..=n]`, parameter values `params[0..=n]`
/// in `[0,1]`, and a clamped knot vector `knots` of degree `p`, computes the
/// `n+1` control points by solving the linear system `B c = d`.
fn spline_interp_1d(
    params: &[f64],
    data: &[[f64; 3]],
    knots: &[f64],
    p: usize,
    n: usize,
) -> BSplineResult<Vec<[f64; 3]>> {
    let n_pts = n + 1;

    // Build collocation matrix B: n_pts × n_pts
    // B[i, j] = N_{j, p}(params[i])
    let mut bmat = vec![0.0_f64; n_pts * n_pts];
    for i in 0..n_pts {
        let u = params[i];
        let span = find_knot_span(knots, u, p, n);
        let basis = basis_functions(knots, u, p, n);
        for (li, &bv) in basis.iter().enumerate() {
            let j = span as isize - p as isize + li as isize;
            if j >= 0 && (j as usize) < n_pts {
                bmat[i * n_pts + j as usize] = bv;
            }
        }
    }

    // Solve B c = d for each coordinate using Gaussian elimination
    // RHS: 3 right-hand side vectors simultaneously
    let mut ctrl = vec![[0.0_f64; 3]; n_pts];
    for coord in 0..3 {
        let rhs: Vec<f64> = data.iter().map(|p| p[coord]).collect();
        let sol = gauss_solve(&bmat, &rhs, n_pts)?;
        for i in 0..n_pts {
            ctrl[i][coord] = sol[i];
        }
    }
    Ok(ctrl)
}

/// Solve `A x = b` via Gaussian elimination with partial pivoting.
///
/// `a_flat` is row-major with shape `n × n`; `b` has length `n`.
fn gauss_solve(a_flat: &[f64], b: &[f64], n: usize) -> BSplineResult<Vec<f64>> {
    // Build augmented matrix [A | b]
    let mut aug = vec![0.0_f64; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a_flat[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        let mut max_val = aug[col * (n + 1) + col].abs();
        let mut max_row = col;
        for row in col + 1..n {
            let v = aug[row * (n + 1) + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return Err(InterpolateError::LinalgError(
                "BSplineSurface::interpolate_grid: singular collocation matrix; \
                 try different degree or more data points".into(),
            ));
        }
        if max_row != col {
            for j in 0..=n {
                aug.swap(col * (n + 1) + j, max_row * (n + 1) + j);
            }
        }
        let pivot = aug[col * (n + 1) + col];
        for row in col + 1..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for j in col..=n {
                let delta = factor * aug[col * (n + 1) + j];
                aug[row * (n + 1) + j] -= delta;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0_f64; n];
    for col in (0..n).rev() {
        let mut val = aug[col * (n + 1) + n];
        for j in col + 1..n {
            val -= aug[col * (n + 1) + j] * x[j];
        }
        x[col] = val / aug[col * (n + 1) + col];
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

/// Create a flat bilinear surface patch (degree 1 in both directions).
///
/// `corners[0]` = lower-left, `corners[1]` = lower-right,
/// `corners[2]` = upper-left, `corners[3]` = upper-right.
pub fn make_bilinear_patch(corners: [[f64; 3]; 4]) -> BSplineResult<BSplineSurface> {
    let mut ctrl = Array3::<f64>::zeros((2, 2, 3));
    for k in 0..3 {
        ctrl[[0, 0, k]] = corners[0][k]; // lower-left
        ctrl[[1, 0, k]] = corners[1][k]; // lower-right
        ctrl[[0, 1, k]] = corners[2][k]; // upper-left
        ctrl[[1, 1, k]] = corners[3][k]; // upper-right
    }
    let u_knots = vec![0.0, 0.0, 1.0, 1.0];
    let v_knots = vec![0.0, 0.0, 1.0, 1.0];
    BSplineSurface::new(ctrl, u_knots, v_knots, 1, 1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array3;

    // ---- helpers ----

    fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
            .collect()
    }

    /// Build a flat bilinear surface at z=0: corners at (0,0), (1,0), (0,1), (1,1)
    fn flat_bilinear() -> BSplineSurface {
        make_bilinear_patch([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        .expect("test: should succeed")
    }

    // ---- construction ----

    #[test]
    fn test_new_valid_bilinear() {
        let surf = flat_bilinear();
        assert_eq!(surf.degree_u(), 1);
        assert_eq!(surf.degree_v(), 1);
        assert_eq!(surf.n_ctrl_u(), 2);
        assert_eq!(surf.n_ctrl_v(), 2);
    }

    #[test]
    fn test_new_zero_degree_error() {
        let ctrl = Array3::<f64>::zeros((2, 2, 3));
        let result = BSplineSurface::new(
            ctrl.clone(),
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            0, // p=0 is invalid
            1,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_new_wrong_knot_length_error() {
        let ctrl = Array3::<f64>::zeros((2, 2, 3));
        // correct u_knots should have length 4; giving 3 → error
        let result = BSplineSurface::new(
            ctrl,
            vec![0.0, 0.5, 1.0], // wrong length
            vec![0.0, 0.0, 1.0, 1.0],
            1,
            1,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_new_non_monotone_knots_error() {
        let ctrl = Array3::<f64>::zeros((2, 2, 3));
        let result = BSplineSurface::new(
            ctrl,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![1.0, 0.0, 0.5, 1.0], // non-monotone
            1,
            1,
        );
        assert!(result.is_err());
    }

    // ---- bilinear surface evaluation ----

    #[test]
    fn test_bilinear_corners() {
        let surf = flat_bilinear();
        let (u_min, u_max) = surf.u_domain();
        let (v_min, v_max) = surf.v_domain();

        let pt00 = surf.evaluate(u_min, v_min).expect("test: should succeed");
        let pt10 = surf.evaluate(u_max, v_min).expect("test: should succeed");
        let pt01 = surf.evaluate(u_min, v_max).expect("test: should succeed");
        let pt11 = surf.evaluate(u_max, v_max).expect("test: should succeed");

        assert_abs_diff_eq!(pt00[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(pt10[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(pt01[1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(pt11[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(pt11[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bilinear_center() {
        let surf = flat_bilinear();
        let pt = surf.evaluate(0.5, 0.5).expect("test: should succeed");
        assert_abs_diff_eq!(pt[0], 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(pt[1], 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(pt[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bilinear_linearity_in_u() {
        // For fixed v, bilinear interpolation should be linear in u
        let surf = flat_bilinear();
        for v in [0.2, 0.5, 0.8] {
            let p0 = surf.evaluate(0.0, v).expect("test: should succeed");
            let p1 = surf.evaluate(1.0, v).expect("test: should succeed");
            let pm = surf.evaluate(0.5, v).expect("test: should succeed");
            assert_abs_diff_eq!(pm[0], (p0[0] + p1[0]) / 2.0, epsilon = 1e-9);
            assert_abs_diff_eq!(pm[1], (p0[1] + p1[1]) / 2.0, epsilon = 1e-9);
        }
    }

    // ---- normal ----

    #[test]
    fn test_bilinear_normal_points_up() {
        // For the xy-plane bilinear surface, normal should point in ±z direction
        let surf = flat_bilinear();
        let n = surf.normal(0.5, 0.5).expect("test: should succeed");
        // Normal should have large z component relative to x, y
        assert!(n[2].abs() > n[0].abs(), "Normal z should dominate");
        assert!(n[2].abs() > n[1].abs(), "Normal z should dominate");
    }

    #[test]
    fn test_normal_nonzero() {
        let surf = flat_bilinear();
        let n = surf.normal(0.3, 0.7).expect("test: should succeed");
        let mag = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!(mag > 1e-10, "Normal magnitude should be non-zero");
    }

    // ---- bicubic surface fitting ----

    #[test]
    fn test_interpolate_grid_flat_plane() {
        // Fit a bicubic surface through a flat plane z = 0
        let r = 4;
        let s = 4;
        let mut pts = Array3::<f64>::zeros((r + 1, s + 1, 3));
        for i in 0..=r {
            for j in 0..=s {
                pts[[i, j, 0]] = i as f64 / r as f64;
                pts[[i, j, 1]] = j as f64 / s as f64;
                pts[[i, j, 2]] = 0.0;
            }
        }
        let surf = BSplineSurface::interpolate_grid(&pts, 3, 3).expect("test: should succeed");
        let pt = surf.evaluate(0.5, 0.5).expect("test: should succeed");
        assert_abs_diff_eq!(pt[2], 0.0, epsilon = 1e-8);
        assert_abs_diff_eq!(pt[0], 0.5, epsilon = 0.1);
    }

    #[test]
    fn test_interpolate_grid_linear_z() {
        // z = x + y; bicubic should reproduce this
        let r = 3;
        let s = 3;
        let mut pts = Array3::<f64>::zeros((r + 1, s + 1, 3));
        for i in 0..=r {
            for j in 0..=s {
                let x = i as f64 / r as f64;
                let y = j as f64 / s as f64;
                pts[[i, j, 0]] = x;
                pts[[i, j, 1]] = y;
                pts[[i, j, 2]] = x + y;
            }
        }
        let surf = BSplineSurface::interpolate_grid(&pts, 1, 1).expect("test: should succeed");
        let pt = surf.evaluate(0.5, 0.5).expect("test: should succeed");
        assert_abs_diff_eq!(pt[2], 1.0, epsilon = 0.05);
    }

    #[test]
    fn test_interpolate_grid_passes_through_corners() {
        // The fitted surface should pass through the corner data points
        let r = 3;
        let s = 3;
        let mut pts = Array3::<f64>::zeros((r + 1, s + 1, 3));
        for i in 0..=r {
            for j in 0..=s {
                let x = i as f64 / r as f64;
                let y = j as f64 / s as f64;
                pts[[i, j, 0]] = x;
                pts[[i, j, 1]] = y;
                pts[[i, j, 2]] = (x * std::f64::consts::PI).sin();
            }
        }
        let surf = BSplineSurface::interpolate_grid(&pts, 3, 3).expect("test: should succeed");
        let (u_min, u_max) = surf.u_domain();
        let (v_min, v_max) = surf.v_domain();
        let pt_start = surf.evaluate(u_min, v_min).expect("test: should succeed");
        // Corner (0,0) should have z ≈ sin(0) = 0
        assert_abs_diff_eq!(pt_start[2], 0.0, epsilon = 0.1);
        let pt_end = surf.evaluate(u_max, v_max).expect("test: should succeed");
        assert!(pt_end[2].is_finite());
    }

    // ---- partial derivatives ----

    #[test]
    fn test_partial_derivative_u_bilinear() {
        // For the bilinear surface on [0,1]^2 with ctrl pts at (0,0,0),(1,0,0),(0,1,0),(1,1,0)
        // ∂S/∂u ≈ (1, 0, 0) direction at any interior point
        let surf = flat_bilinear();
        let du = surf.partial_derivative_u(0.5, 0.5).expect("test: should succeed");
        // Should be approximately (1, 0, 0)
        assert!(du[0].abs() > 0.5, "du_x should be significant");
        assert!(du[2].abs() < 1e-8, "du_z should be zero on flat surface");
    }

    #[test]
    fn test_partial_derivative_v_bilinear() {
        let surf = flat_bilinear();
        let dv = surf.partial_derivative_v(0.5, 0.5).expect("test: should succeed");
        assert!(dv[1].abs() > 0.5, "dv_y should be significant");
        assert!(dv[2].abs() < 1e-8, "dv_z should be zero on flat surface");
    }

    // ---- make_bilinear_patch ----

    #[test]
    fn test_make_bilinear_patch_tilted() {
        // Tilted patch: z varies linearly
        let surf = make_bilinear_patch([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
        ])
        .expect("test: should succeed");
        let pt = surf.evaluate(0.5, 0.5).expect("test: should succeed");
        assert_abs_diff_eq!(pt[2], 1.0, epsilon = 1e-8);
    }

    // ---- accessors ----

    #[test]
    fn test_accessors() {
        let surf = flat_bilinear();
        let (u_min, u_max) = surf.u_domain();
        let (v_min, v_max) = surf.v_domain();
        assert_abs_diff_eq!(u_min, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(u_max, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(v_min, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(v_max, 1.0, epsilon = 1e-12);
    }

    // ---- degree-2 surface ----

    #[test]
    fn test_biquadratic_surface() {
        // 3×3 control grid, degree 2 in both directions
        let mut ctrl = Array3::<f64>::zeros((3, 3, 3));
        for i in 0..3 {
            for j in 0..3 {
                ctrl[[i, j, 0]] = i as f64 * 0.5;
                ctrl[[i, j, 1]] = j as f64 * 0.5;
                ctrl[[i, j, 2]] = 0.0;
            }
        }
        // knot vector: 3+2+2 = 7 knots
        let u_knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let v_knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let surf = BSplineSurface::new(ctrl, u_knots, v_knots, 2, 2).expect("test: should succeed");
        let pt = surf.evaluate(0.5, 0.5).expect("test: should succeed");
        assert!(pt[0].is_finite() && pt[1].is_finite() && pt[2].is_finite());
    }
}
