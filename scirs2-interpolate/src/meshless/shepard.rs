//! Shepard's Method and Variants for Scattered Data Interpolation
//!
//! This module implements three flavours of Shepard interpolation:
//!
//! 1. **Basic (global) Shepard**: classic inverse-distance weighting
//!    `w_i(x) = 1 / ‖x − xᵢ‖^p`.
//! 2. **Modified (local) Shepard**: fits a local quadratic in each neighbourhood
//!    (Renka 1988) and blends with Franke-Little-style compact-support weights.
//! 3. **Franke-Little weights**: replaces the singular `1/rᵖ` weights with the
//!    smooth compact-support form `((R − r)₊ / (R r))²` (Franke & Nielson 1980).
//!
//! A unified high-level struct [`ShepardInterpolant`] provides all three modes
//! via the [`ShepardMode`] enum, and supports batch evaluation.
//!
//! ## References
//!
//! - Shepard, D. (1968). A two-dimensional interpolation function for
//!   irregularly-spaced data. *Proc. 23rd ACM Nat. Conf.*, 517-524.
//! - Franke, R. & Nielson, G. (1980). Smooth interpolation of large sets of
//!   scattered data. *Int. J. Numer. Meth. Eng.* 15, 1691-1704.
//! - Renka, R. J. (1988). Multivariate interpolation of large sets of scattered
//!   data. *ACM TOMS* 14(2), 139-148.
//! - Renka, R. J. & Cline, A. K. (1984). A triangle-based C¹ interpolation
//!   method. *Rocky Mountain J. Math.* 14, 223-237.

use crate::error::{InterpolateError, InterpolateResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_linalg::solve;

// ---------------------------------------------------------------------------
// Distance utilities
// ---------------------------------------------------------------------------

#[inline]
fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Return sorted (distance, original_index) pairs for a query point.
fn sorted_distances(query: &[f64], points: &Array2<f64>) -> Vec<(f64, usize)> {
    let n = points.nrows();
    let d = points.ncols();
    let mut dists: Vec<(f64, usize)> = (0..n)
        .map(|i| {
            let row: Vec<f64> = (0..d).map(|k| points[[i, k]]).collect();
            (euclidean_dist(query, &row), i)
        })
        .collect();
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    dists
}

// ---------------------------------------------------------------------------
// ShepardMode
// ---------------------------------------------------------------------------

/// Selects the variant of Shepard interpolation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShepardMode {
    /// Global inverse-distance weighting: `w_i = 1 / r^power`.
    Global {
        /// IDW power, typically `p = 2`.
        power: f64,
    },
    /// Modified Shepard (Renka): local quadratic fit + Franke-Little blending
    /// within radius `R` (chosen automatically or manually).
    Modified {
        /// Support radius.  Set to `0.0` to use automatic selection from the
        /// `k`-th nearest neighbour.
        radius: f64,
        /// Number of nearest neighbours used for automatic radius selection
        /// (only used when `radius == 0`).
        k_auto: usize,
    },
    /// Franke-Little weights with explicit radius.
    FrankeLittle {
        /// Support radius.
        radius: f64,
        /// IDW power for the base weights inside the support.
        power: f64,
    },
}

// ---------------------------------------------------------------------------
// ShepardInterpolant
// ---------------------------------------------------------------------------

/// Unified Shepard interpolant.
///
/// # Examples
///
/// ## Basic (global) Shepard
///
/// ```rust
/// use scirs2_interpolate::meshless::shepard::{ShepardInterpolant, ShepardMode};
/// use scirs2_core::ndarray::{array, Array2};
///
/// let pts = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0,  1.0, 0.0,  0.0, 1.0,  1.0, 1.0,
/// ]).expect("doc example: should succeed");
/// let vals = array![0.0, 1.0, 1.0, 2.0];
/// let s = ShepardInterpolant::new(&pts.view(), &vals.view(), ShepardMode::Global { power: 2.0 }).expect("doc example: should succeed");
/// let v = s.evaluate(&[0.5, 0.5]).expect("doc example: should succeed");
/// assert!((v - 1.0).abs() < 1e-10);
/// ```
///
/// ## Modified Shepard
///
/// ```rust
/// use scirs2_interpolate::meshless::shepard::{ShepardInterpolant, ShepardMode};
/// use scirs2_core::ndarray::{array, Array2};
///
/// let pts = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0,  1.0, 0.0,  0.0, 1.0,  1.0, 1.0,
/// ]).expect("doc example: should succeed");
/// let vals = array![0.0, 1.0, 1.0, 2.0];
/// let s = ShepardInterpolant::new(
///     &pts.view(), &vals.view(),
///     ShepardMode::Modified { radius: 1.5, k_auto: 4 },
/// ).expect("doc example: should succeed");
/// let v = s.evaluate(&[0.5, 0.5]).expect("doc example: should succeed");
/// assert!((v - 1.0).abs() < 1e-3);
/// ```
pub struct ShepardInterpolant {
    points: Array2<f64>,
    values: Array1<f64>,
    mode: ShepardMode,
    /// Pre-computed pairwise support radii for modified Shepard
    /// (one per data site for adaptive variant).
    radii: Option<Array1<f64>>,
}

impl ShepardInterpolant {
    /// Create a new Shepard interpolant.
    ///
    /// # Arguments
    ///
    /// * `points` – `(n, d)` data-site coordinates.
    /// * `values` – `n` function values.
    /// * `mode`   – Which Shepard variant to use.
    pub fn new(
        points: &ArrayView2<f64>,
        values: &ArrayView1<f64>,
        mode: ShepardMode,
    ) -> InterpolateResult<Self> {
        let n = points.nrows();
        if values.len() != n {
            return Err(InterpolateError::DimensionMismatch(format!(
                "points has {n} rows, values has {} entries",
                values.len()
            )));
        }
        if n == 0 {
            return Err(InterpolateError::InsufficientData(
                "Shepard interpolation requires at least one data point".to_string(),
            ));
        }

        // Validate mode parameters
        match mode {
            ShepardMode::Global { power } if power <= 0.0 => {
                return Err(InterpolateError::InvalidInput {
                    message: format!("Shepard power must be > 0, got {power}"),
                });
            }
            ShepardMode::FrankeLittle { radius, power } => {
                if radius <= 0.0 {
                    return Err(InterpolateError::InvalidInput {
                        message: format!("Franke-Little radius must be > 0, got {radius}"),
                    });
                }
                if power <= 0.0 {
                    return Err(InterpolateError::InvalidInput {
                        message: format!("Franke-Little power must be > 0, got {power}"),
                    });
                }
            }
            _ => {}
        }

        let pts_owned = points.to_owned();

        // For Modified Shepard with automatic radius, precompute per-node radii.
        let radii = match mode {
            ShepardMode::Modified { radius, k_auto } if radius == 0.0 => {
                let k = k_auto.max(1).min(n - 1);
                let mut r_vec = Array1::<f64>::zeros(n);
                for i in 0..n {
                    let qi: Vec<f64> = (0..pts_owned.ncols()).map(|k2| pts_owned[[i, k2]]).collect();
                    let dists = sorted_distances(&qi, &pts_owned);
                    // k-th nearest (skip self at index 0)
                    let kth = dists.get(k).map(|(d, _)| *d).unwrap_or(1.0);
                    r_vec[i] = kth * 2.0; // double the k-NN distance
                }
                Some(r_vec)
            }
            _ => None,
        };

        Ok(Self {
            points: pts_owned,
            values: values.to_owned(),
            mode,
            radii,
        })
    }

    /// Evaluate the interpolant at `query`.
    pub fn evaluate(&self, query: &[f64]) -> InterpolateResult<f64> {
        let d = self.points.ncols();
        if query.len() != d {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query has {} dims, points have {d}",
                query.len()
            )));
        }

        match self.mode {
            ShepardMode::Global { power } => self.eval_global(query, power),
            ShepardMode::Modified { radius, .. } => self.eval_modified(query, radius),
            ShepardMode::FrankeLittle { radius, power } => {
                self.eval_franke_little(query, radius, power)
            }
        }
    }

    /// Evaluate at multiple query points.
    pub fn evaluate_batch(&self, queries: &ArrayView2<f64>) -> InterpolateResult<Array1<f64>> {
        let nq = queries.nrows();
        let mut out = Array1::<f64>::zeros(nq);
        for i in 0..nq {
            let q: Vec<f64> = (0..queries.ncols()).map(|j| queries[[i, j]]).collect();
            out[i] = self.evaluate(&q)?;
        }
        Ok(out)
    }

    // -----------------------------------------------------------------------
    // Global Shepard (classic IDW)
    // -----------------------------------------------------------------------

    fn eval_global(&self, query: &[f64], power: f64) -> InterpolateResult<f64> {
        let n = self.points.nrows();
        let d = self.points.ncols();
        let mut wsum = 0.0_f64;
        let mut fsum = 0.0_f64;

        for i in 0..n {
            let row: Vec<f64> = (0..d).map(|k| self.points[[i, k]]).collect();
            let r = euclidean_dist(query, &row);
            if r <= 0.0 {
                // Query coincides with a data site → return its value exactly
                return Ok(self.values[i]);
            }
            let w = r.powf(-power);
            wsum += w;
            fsum += w * self.values[i];
        }

        if wsum == 0.0 {
            return Err(InterpolateError::NumericalError(
                "All weights are zero in global Shepard".to_string(),
            ));
        }
        Ok(fsum / wsum)
    }

    // -----------------------------------------------------------------------
    // Modified (local) Shepard – Renka's algorithm
    // -----------------------------------------------------------------------
    //
    // For each query point x₀ we:
    //  1. Find all neighbours within radius R (or use pre-computed adaptive R).
    //  2. Fit a local quadratic Q_i(x) to each neighbour i using a weighted
    //     least squares with Franke-Little weights; in the simplest variant we
    //     use Q_i(x) = f_i + gradient_i · (x − x_i) (linear, not full quadratic)
    //     augmented with a Hessian term if enough points are available.
    //  3. Blend: s(x) = Σ W_i(x) Q_i(x) / Σ W_i(x),
    //     where W_i(x) = ((R-r)/(R·r))²  (Franke-Little weight).

    fn eval_modified(&self, query: &[f64], explicit_radius: f64) -> InterpolateResult<f64> {
        let n = self.points.nrows();
        let d = self.points.ncols();

        let mut wsum = 0.0_f64;
        let mut fsum = 0.0_f64;

        for i in 0..n {
            let xi: Vec<f64> = (0..d).map(|k| self.points[[i, k]]).collect();
            let r = euclidean_dist(query, &xi);

            // Determine effective radius for this node
            let ri = if explicit_radius > 0.0 {
                explicit_radius
            } else {
                self.radii.as_ref().map(|rv| rv[i]).unwrap_or(explicit_radius)
            };

            if r >= ri {
                continue; // outside support
            }

            // Franke-Little weight W_i(r) = ((R-r)/(R·r))²
            let w = if r <= 0.0 {
                // Query equals this data site
                return Ok(self.values[i]);
            } else {
                let ratio = (ri - r) / (ri * r);
                ratio * ratio
            };

            // Local quadratic fit Q_i(x)
            let q_val = self.local_polynomial_at(i, query, ri)?;

            wsum += w;
            fsum += w * q_val;
        }

        if wsum <= 0.0 {
            // Fall back to global Shepard when no neighbours within any R
            return self.eval_global(query, 2.0);
        }
        Ok(fsum / wsum)
    }

    /// Fit a local polynomial `Q_i(x)` centred at data site `i` using all
    /// neighbours within radius `radius`, then evaluate at `query`.
    fn local_polynomial_at(
        &self,
        center_idx: usize,
        query: &[f64],
        radius: f64,
    ) -> InterpolateResult<f64> {
        let n = self.points.nrows();
        let d = self.points.ncols();

        let xi: Vec<f64> = (0..d).map(|k| self.points[[center_idx, k]]).collect();

        // Gather neighbours within radius, excluding center itself
        let mut nbr_pts: Vec<Vec<f64>> = Vec::new();
        let mut nbr_vals: Vec<f64> = Vec::new();
        let mut nbr_dists: Vec<f64> = Vec::new();

        for j in 0..n {
            let xj: Vec<f64> = (0..d).map(|k| self.points[[j, k]]).collect();
            let r_ij = euclidean_dist(&xi, &xj);
            if r_ij < radius {
                nbr_pts.push(xj);
                nbr_vals.push(self.values[j]);
                nbr_dists.push(r_ij.max(1e-14));
            }
        }

        if nbr_pts.is_empty() {
            // Isolated node: constant extrapolation
            return Ok(self.values[center_idx]);
        }

        // Determine polynomial degree based on available points
        // Linear needs d+1 parameters; quadratic needs (d+1)(d+2)/2
        let lin_params = 1 + d;
        let quad_params = (d + 1) * (d + 2) / 2;

        let use_quad = nbr_pts.len() >= quad_params + 1;
        let use_lin = nbr_pts.len() >= lin_params;

        if !use_lin {
            // Constant: weighted mean
            let mut wsum = 0.0_f64;
            let mut fsum = 0.0_f64;
            for (j, &fj) in nbr_vals.iter().enumerate() {
                let w = 1.0 / nbr_dists[j].powi(2);
                wsum += w;
                fsum += w * fj;
            }
            return Ok(if wsum > 0.0 {
                fsum / wsum
            } else {
                self.values[center_idx]
            });
        }

        // Build weighted least squares:
        // minimise Σ_j w_j (Q(xⱼ) − fⱼ)²
        // where Q(x) = c₀ + c₁(x₁−xi₁) + … + cd(xd−xid) [+ quadratic terms]

        let num_params = if use_quad { quad_params } else { lin_params };
        let k = nbr_pts.len();

        // Design matrix B (k × num_params) in local coords
        let mut b = Array2::<f64>::zeros((k, num_params));
        let mut rhs_vec = Array1::<f64>::zeros(k);
        let mut weights = Array1::<f64>::zeros(k);

        for (j, (xj, &fj)) in nbr_pts.iter().zip(nbr_vals.iter()).enumerate() {
            let wj = 1.0 / nbr_dists[j].powi(2);
            weights[j] = wj;
            rhs_vec[j] = fj;

            let mut col = 0usize;
            b[[j, col]] = 1.0;
            col += 1;
            for k2 in 0..d {
                b[[j, col]] = xj[k2] - xi[k2];
                col += 1;
            }
            if use_quad {
                for k2 in 0..d {
                    for l in k2..d {
                        b[[j, col]] = (xj[k2] - xi[k2]) * (xj[l] - xi[l]);
                        col += 1;
                    }
                }
            }
        }

        // Form normal equations: (BᵀWB) c = BᵀW f
        let mut btb = Array2::<f64>::zeros((num_params, num_params));
        let mut btf = Array1::<f64>::zeros(num_params);
        for j in 0..k {
            let wj = weights[j];
            let fj = rhs_vec[j];
            for p in 0..num_params {
                btf[p] += wj * b[[j, p]] * fj;
                for q in 0..num_params {
                    btb[[p, q]] += wj * b[[j, p]] * b[[j, q]];
                }
            }
        }

        // Regularise diagonal
        let reg = 1e-12 * (0..num_params).map(|p| btb[[p, p]]).sum::<f64>() / num_params as f64;
        for p in 0..num_params {
            btb[[p, p]] += reg.max(1e-14);
        }

        let btb_view = btb.view();
        let btf_view = btf.view();
        let coeffs = solve(&btb_view, &btf_view, None).map_err(|e| {
            InterpolateError::LinalgError(format!("Modified Shepard local fit failed: {e}"))
        })?;

        // Evaluate polynomial at query
        let mut val = coeffs[0]; // constant
        let mut col = 1usize;
        for k2 in 0..d {
            val += coeffs[col] * (query[k2] - xi[k2]);
            col += 1;
        }
        if use_quad {
            for k2 in 0..d {
                for l in k2..d {
                    val += coeffs[col] * (query[k2] - xi[k2]) * (query[l] - xi[l]);
                    col += 1;
                }
            }
        }
        Ok(val)
    }

    // -----------------------------------------------------------------------
    // Franke-Little weights (stand-alone variant)
    // -----------------------------------------------------------------------

    fn eval_franke_little(&self, query: &[f64], radius: f64, power: f64) -> InterpolateResult<f64> {
        let n = self.points.nrows();
        let d = self.points.ncols();
        let mut wsum = 0.0_f64;
        let mut fsum = 0.0_f64;

        for i in 0..n {
            let row: Vec<f64> = (0..d).map(|k| self.points[[i, k]]).collect();
            let r = euclidean_dist(query, &row);
            if r <= 0.0 {
                return Ok(self.values[i]);
            }
            if r >= radius {
                continue;
            }
            // Franke-Little weight: ((R-r)/(R·r))^p  or  (R-r)^2 / (R^2 r^p)
            let w = ((radius - r) / (radius * r)).powf(power);
            wsum += w;
            fsum += w * self.values[i];
        }

        if wsum <= 0.0 {
            // Fall back to global IDW when no points in radius
            return self.eval_global(query, power);
        }
        Ok(fsum / wsum)
    }
}

// ---------------------------------------------------------------------------
// Free-function API (convenience wrappers)
// ---------------------------------------------------------------------------

/// Evaluate basic (global) Shepard at `query`.
///
/// # Arguments
///
/// * `query`  – Query point.
/// * `points` – Data sites `(n, d)`.
/// * `values` – Function values `n`.
/// * `power`  – IDW power `p > 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::meshless::shepard::basic_shepard;
/// use scirs2_core::ndarray::{array, Array2};
///
/// let pts = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).expect("doc example: should succeed");
/// let vals = array![0.0, 1.0, 4.0];
/// let v = basic_shepard(&[0.5], &pts.view(), &vals.view(), 2.0).expect("doc example: should succeed");
/// assert!(v.is_finite());
/// ```
pub fn basic_shepard(
    query: &[f64],
    points: &ArrayView2<f64>,
    values: &ArrayView1<f64>,
    power: f64,
) -> InterpolateResult<f64> {
    let s = ShepardInterpolant::new(points, values, ShepardMode::Global { power })?;
    s.evaluate(query)
}

/// Evaluate modified (local) Shepard at `query`.
///
/// # Arguments
///
/// * `query`  – Query point.
/// * `points` – Data sites `(n, d)`.
/// * `values` – Function values `n`.
/// * `radius` – Support radius (0 for automatic selection from `k_auto` neighbours).
/// * `k_auto` – Number of neighbours for auto radius (used when `radius == 0`).
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::meshless::shepard::modified_shepard;
/// use scirs2_core::ndarray::{array, Array2};
///
/// let pts = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).expect("doc example: should succeed");
/// let vals = array![0.0, 1.0, 4.0, 9.0];
/// let v = modified_shepard(&[1.5], &pts.view(), &vals.view(), 2.5, 3).expect("doc example: should succeed");
/// assert!(v.is_finite());
/// ```
pub fn modified_shepard(
    query: &[f64],
    points: &ArrayView2<f64>,
    values: &ArrayView1<f64>,
    radius: f64,
    k_auto: usize,
) -> InterpolateResult<f64> {
    let s = ShepardInterpolant::new(points, values, ShepardMode::Modified { radius, k_auto })?;
    s.evaluate(query)
}

/// Evaluate Franke-Little weighted Shepard at `query`.
///
/// # Arguments
///
/// * `query`  – Query point.
/// * `points` – Data sites `(n, d)`.
/// * `values` – Function values `n`.
/// * `radius` – Support radius `R > 0`.
/// * `power`  – Weight power `p > 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::meshless::shepard::franke_little_shepard;
/// use scirs2_core::ndarray::{array, Array2};
///
/// let pts = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).expect("doc example: should succeed");
/// let vals = array![0.0, 1.0, 4.0, 9.0];
/// let v = franke_little_shepard(&[1.5], &pts.view(), &vals.view(), 2.5, 2.0).expect("doc example: should succeed");
/// assert!(v.is_finite());
/// ```
pub fn franke_little_shepard(
    query: &[f64],
    points: &ArrayView2<f64>,
    values: &ArrayView1<f64>,
    radius: f64,
    power: f64,
) -> InterpolateResult<f64> {
    let s = ShepardInterpolant::new(
        points,
        values,
        ShepardMode::FrankeLittle { radius, power },
    )?;
    s.evaluate(query)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    fn pts_1d(xs: &[f64]) -> Array2<f64> {
        Array2::from_shape_vec((xs.len(), 1), xs.to_vec()).expect("test: should succeed")
    }

    #[test]
    fn test_global_shepard_exact_at_nodes() {
        let pts = pts_1d(&[0.0, 1.0, 2.0, 3.0]);
        let vals = array![0.0_f64, 1.0, 4.0, 9.0];
        let s = ShepardInterpolant::new(
            &pts.view(),
            &vals.view(),
            ShepardMode::Global { power: 2.0 },
        )
        .expect("test: should succeed");
        for i in 0..4 {
            let v = s.evaluate(&[i as f64]).expect("test: should succeed");
            assert_abs_diff_eq!(v, vals[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_global_shepard_symmetry() {
        // f(x,y) = x + y → at the centre of a symmetric 4-point grid,
        // the IDW prediction should equal 1 for p=2
        let pts = Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        .expect("test: should succeed");
        let vals = array![0.0_f64, 1.0, 1.0, 2.0];
        let s = ShepardInterpolant::new(
            &pts.view(),
            &vals.view(),
            ShepardMode::Global { power: 2.0 },
        )
        .expect("test: should succeed");
        let v = s.evaluate(&[0.5, 0.5]).expect("test: should succeed");
        assert_abs_diff_eq!(v, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_modified_shepard_linear() {
        // f(x) = 2x → modified Shepard with quadratic local fit should reproduce
        let pts = pts_1d(&[0.0, 0.5, 1.0, 1.5, 2.0]);
        let vals: Array1<f64> = (0..5).map(|i| 2.0 * i as f64 * 0.5).collect();
        let s = ShepardInterpolant::new(
            &pts.view(),
            &vals.view(),
            ShepardMode::Modified {
                radius: 1.2,
                k_auto: 4,
            },
        )
        .expect("test: should succeed");
        let v = s.evaluate(&[0.75]).expect("test: should succeed");
        assert_abs_diff_eq!(v, 1.5, epsilon = 1e-6);
    }

    #[test]
    fn test_franke_little_finite() {
        let pts = pts_1d(&[0.0, 1.0, 2.0, 3.0]);
        let vals = array![0.0_f64, 1.0, 4.0, 9.0];
        let v = franke_little_shepard(&[1.5], &pts.view(), &vals.view(), 2.5, 2.0).expect("test: should succeed");
        assert!(v.is_finite());
        assert!(v > 0.0);
    }

    #[test]
    fn test_basic_shepard_free_fn() {
        let pts = pts_1d(&[0.0, 1.0, 2.0]);
        let vals = array![0.0_f64, 1.0, 2.0];
        let v = basic_shepard(&[0.5], &pts.view(), &vals.view(), 2.0).expect("test: should succeed");
        assert!(v.is_finite());
        assert!(v > 0.0 && v < 1.0);
    }

    #[test]
    fn test_modified_shepard_auto_radius() {
        let pts = pts_1d(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let vals: Array1<f64> = (0..5).map(|i| i as f64).collect();
        let s = ShepardInterpolant::new(
            &pts.view(),
            &vals.view(),
            ShepardMode::Modified {
                radius: 0.0,
                k_auto: 3,
            },
        )
        .expect("test: should succeed");
        let v = s.evaluate(&[2.5]).expect("test: should succeed");
        assert!(v.is_finite());
        assert!((v - 2.5).abs() < 0.5);
    }

    #[test]
    fn test_batch_equals_individual() {
        let pts = Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        .expect("test: should succeed");
        let vals = array![0.0_f64, 1.0, 1.0, 2.0];
        let s = ShepardInterpolant::new(
            &pts.view(),
            &vals.view(),
            ShepardMode::Global { power: 2.0 },
        )
        .expect("test: should succeed");

        let queries = Array2::from_shape_vec(
            (3, 2),
            vec![0.2, 0.3, 0.7, 0.8, 0.5, 0.5],
        )
        .expect("test: should succeed");
        let batch = s.evaluate_batch(&queries.view()).expect("test: should succeed");
        for i in 0..3 {
            let q = vec![queries[[i, 0]], queries[[i, 1]]];
            let single = s.evaluate(&q).expect("test: should succeed");
            assert_abs_diff_eq!(batch[i], single, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_invalid_power_rejected() {
        let pts = pts_1d(&[0.0, 1.0]);
        let vals = array![0.0_f64, 1.0];
        let result = ShepardInterpolant::new(
            &pts.view(),
            &vals.view(),
            ShepardMode::Global { power: -1.0 },
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_points_rejected() {
        let pts = Array2::<f64>::zeros((0, 2));
        let vals = Array1::<f64>::zeros(0);
        let result = ShepardInterpolant::new(
            &pts.view(),
            &vals.view(),
            ShepardMode::Global { power: 2.0 },
        );
        assert!(result.is_err());
    }
}
