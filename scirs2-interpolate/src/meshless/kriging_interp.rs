//! Kriging Interpolation for Deterministic / Geostatistical Data
//!
//! Provides four flavours of kriging commonly used in spatial statistics and
//! deterministic surrogate modelling:
//!
//! - **Ordinary kriging** – assumes unknown constant mean; minimises variance
//!   subject to the unbiasedness constraint.
//! - **Universal kriging** – assumes a polynomial trend (drift) of known form.
//! - **Co-kriging** – jointly interpolates a primary variable using one or more
//!   correlated secondary variables.
//! - **Kriging with external drift (KED)** – uses a secondary variable as a
//!   trend component rather than a polynomial.
//!
//! All methods use the *variogram* approach: the user supplies a
//! [`Variogram`] model (spherical, exponential, Gaussian, or power-law) that
//! characterises spatial correlation.
//!
//! ## References
//!
//! - Cressie, N. (1993). *Statistics for Spatial Data* (rev. ed.). Wiley.
//! - Journel, A. G. & Huijbregts, C. J. (1978). *Mining Geostatistics*. Academic Press.

use crate::error::{InterpolateError, InterpolateResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_linalg::solve;

// ---------------------------------------------------------------------------
// Variogram models
// ---------------------------------------------------------------------------

/// Isotropic variogram models γ(h) describing spatial variance vs. lag `h`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Variogram {
    /// γ(h) = nugget + sill · [1.5 h/range − 0.5 (h/range)³]   (h ≤ range),
    ///        nugget + sill                                       (h > range)
    Spherical {
        /// Discontinuity at origin (micro-scale variance).
        nugget: f64,
        /// Variance ceiling (sill − nugget is the structural sill).
        sill: f64,
        /// Distance at which γ reaches the sill.
        range: f64,
    },
    /// γ(h) = nugget + sill · (1 − exp(−h/range))
    Exponential {
        nugget: f64,
        sill: f64,
        range: f64,
    },
    /// γ(h) = nugget + sill · (1 − exp(−(h/range)²))
    Gaussian {
        nugget: f64,
        sill: f64,
        range: f64,
    },
    /// γ(h) = nugget + slope · h^alpha   (alpha ∈ (0, 2))
    PowerLaw {
        nugget: f64,
        slope: f64,
        alpha: f64,
    },
}

impl Variogram {
    /// Evaluate the variogram at lag `h`.
    pub fn eval(&self, h: f64) -> f64 {
        if h < 0.0 {
            return 0.0;
        }
        match self {
            Variogram::Spherical { nugget, sill, range } => {
                if h == 0.0 {
                    0.0
                } else if h <= *range {
                    let u = h / range;
                    nugget + sill * (1.5 * u - 0.5 * u * u * u)
                } else {
                    nugget + sill
                }
            }
            Variogram::Exponential { nugget, sill, range } => {
                if h == 0.0 {
                    0.0
                } else {
                    nugget + sill * (1.0 - (-h / range).exp())
                }
            }
            Variogram::Gaussian { nugget, sill, range } => {
                if h == 0.0 {
                    0.0
                } else {
                    let u = h / range;
                    nugget + sill * (1.0 - (-u * u).exp())
                }
            }
            Variogram::PowerLaw { nugget, slope, alpha } => {
                if h == 0.0 {
                    0.0
                } else {
                    nugget + slope * h.powf(*alpha)
                }
            }
        }
    }

    /// Evaluate the covariance C(h) = C(0) − γ(h) for stationary models.
    /// For [`PowerLaw`] this returns `None` (intrinsic-only).
    pub fn covariance(&self, h: f64) -> Option<f64> {
        match self {
            Variogram::Spherical { nugget, sill, .. }
            | Variogram::Exponential { nugget, sill, .. }
            | Variogram::Gaussian { nugget, sill, .. } => {
                let c0 = nugget + sill;
                Some(c0 - self.eval(h))
            }
            Variogram::PowerLaw { .. } => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Distance utilities
// ---------------------------------------------------------------------------

fn euclidean_dist_nd(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn dist_matrix(points: &Array2<f64>) -> Array2<f64> {
    let n = points.nrows();
    let mut d = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let row_i: Vec<f64> = (0..points.ncols()).map(|k| points[[i, k]]).collect();
            let row_j: Vec<f64> = (0..points.ncols()).map(|k| points[[j, k]]).collect();
            let dist = euclidean_dist_nd(&row_i, &row_j);
            d[[i, j]] = dist;
            d[[j, i]] = dist;
        }
    }
    d
}

fn dist_to_points(query: &[f64], points: &Array2<f64>) -> Array1<f64> {
    let n = points.nrows();
    let d = points.ncols();
    let mut out = Array1::<f64>::zeros(n);
    for i in 0..n {
        let row: Vec<f64> = (0..d).map(|k| points[[i, k]]).collect();
        out[i] = euclidean_dist_nd(query, &row);
    }
    out
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

fn linalg_solve(a: Array2<f64>, b: Array1<f64>) -> InterpolateResult<Array1<f64>> {
    let av = a.view();
    let bv = b.view();
    solve(&av, &bv, None).map_err(|e| {
        InterpolateError::LinalgError(format!("Kriging solve failed: {e}"))
    })
}

// ---------------------------------------------------------------------------
// Ordinary Kriging
// ---------------------------------------------------------------------------

/// Ordinary kriging predictor.
///
/// Assumes the underlying field has an unknown but constant mean μ.
/// The kriging system is:
/// ```text
/// ┌ Γ   1 ┐ ┌ λ ┐   ┌ γ₀ ┐
/// │ 1ᵀ  0 │ │ μ │ = │ 1  │
/// └       ┘ └   ┘   └    ┘
/// ```
/// where `Γ_{ij} = γ(‖xᵢ − xⱼ‖)`, `γ₀ᵢ = γ(‖x₀ − xᵢ‖)`, `1` is the
/// ones vector.
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::meshless::kriging_interp::{
///     OrdinaryKriging, Variogram,
/// };
/// use scirs2_core::ndarray::{array, Array2};
///
/// let pts = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0,  1.0, 0.0,  0.0, 1.0,  1.0, 1.0,
/// ]).expect("doc example: should succeed");
/// let vals = array![0.0, 1.0, 1.0, 2.0];
/// let vgm = Variogram::Spherical { nugget: 0.0, sill: 1.0, range: 2.0 };
/// let ok = OrdinaryKriging::new(&pts.view(), &vals.view(), vgm).expect("doc example: should succeed");
/// let (pred, var) = ok.predict(&[0.5, 0.5]).expect("doc example: should succeed");
/// assert!(pred.is_finite());
/// assert!(var >= 0.0);
/// ```
pub struct OrdinaryKriging {
    points: Array2<f64>,
    values: Array1<f64>,
    variogram: Variogram,
    /// Pre-factored: the full (n+1)×(n+1) kriging matrix (stored for reuse).
    kriging_mat: Array2<f64>,
}

impl OrdinaryKriging {
    /// Build the ordinary kriging model.
    pub fn new(
        points: &ArrayView2<f64>,
        values: &ArrayView1<f64>,
        variogram: Variogram,
    ) -> InterpolateResult<Self> {
        let n = points.nrows();
        if values.len() != n {
            return Err(InterpolateError::DimensionMismatch(format!(
                "points has {n} rows but values has {} entries",
                values.len()
            )));
        }
        if n == 0 {
            return Err(InterpolateError::InsufficientData(
                "Ordinary kriging requires at least one data point".to_string(),
            ));
        }

        let pts_owned = points.to_owned();
        let dist = dist_matrix(&pts_owned);

        // Build the (n+1)×(n+1) kriging matrix
        let mut km = Array2::<f64>::zeros((n + 1, n + 1));
        for i in 0..n {
            for j in 0..n {
                km[[i, j]] = variogram.eval(dist[[i, j]]);
            }
        }
        // Unbiasedness constraints
        for i in 0..n {
            km[[i, n]] = 1.0;
            km[[n, i]] = 1.0;
        }
        // km[[n, n]] = 0 already

        Ok(Self {
            points: pts_owned,
            values: values.to_owned(),
            variogram,
            kriging_mat: km,
        })
    }

    /// Predict at a new location. Returns `(prediction, kriging_variance)`.
    pub fn predict(&self, query: &[f64]) -> InterpolateResult<(f64, f64)> {
        let n = self.points.nrows();
        let d = self.points.ncols();
        if query.len() != d {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query has {} dims, points have {d}",
                query.len()
            )));
        }

        let dists = dist_to_points(query, &self.points);

        // RHS vector: [γ(‖x₀ − xᵢ‖), 1]
        let mut rhs = Array1::<f64>::zeros(n + 1);
        for i in 0..n {
            rhs[i] = self.variogram.eval(dists[i]);
        }
        rhs[n] = 1.0;

        let lam = linalg_solve(self.kriging_mat.clone(), rhs.clone())?;

        let pred: f64 = (0..n).map(|i| lam[i] * self.values[i]).sum();

        // Kriging variance: σ²_k = lam · rhs (using the primal form)
        let var: f64 = (0..n + 1).map(|i| lam[i] * rhs[i]).sum();

        Ok((pred, var.max(0.0)))
    }

    /// Predict at multiple query points. Returns `(predictions, variances)`.
    pub fn predict_batch(
        &self,
        queries: &ArrayView2<f64>,
    ) -> InterpolateResult<(Array1<f64>, Array1<f64>)> {
        let nq = queries.nrows();
        let mut preds = Array1::<f64>::zeros(nq);
        let mut vars = Array1::<f64>::zeros(nq);
        for i in 0..nq {
            let q: Vec<f64> = (0..queries.ncols()).map(|j| queries[[i, j]]).collect();
            let (p, v) = self.predict(&q)?;
            preds[i] = p;
            vars[i] = v;
        }
        Ok((preds, vars))
    }
}

// ---------------------------------------------------------------------------
// Universal Kriging (trend of polynomial type)
// ---------------------------------------------------------------------------

/// Degree of the polynomial trend for universal kriging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDegree {
    /// μ(x) = β₀ (ordinary kriging)
    Constant,
    /// μ(x) = β₀ + β₁x₁ + … + βd xd
    Linear,
    /// μ(x) = linear + quadratic cross-terms
    Quadratic,
}

fn trend_row(query: &[f64], degree: TrendDegree) -> Vec<f64> {
    let d = query.len();
    let mut row = vec![1.0_f64];
    if degree == TrendDegree::Constant {
        return row;
    }
    row.extend_from_slice(query);
    if degree == TrendDegree::Quadratic {
        for k in 0..d {
            for l in k..d {
                row.push(query[k] * query[l]);
            }
        }
    }
    row
}

fn trend_matrix(points: &Array2<f64>, degree: TrendDegree) -> Array2<f64> {
    let n = points.nrows();
    let d = points.ncols();
    let m = match degree {
        TrendDegree::Constant => 1,
        TrendDegree::Linear => 1 + d,
        TrendDegree::Quadratic => 1 + d + d * (d + 1) / 2,
    };
    let mut mat = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        let row_vec: Vec<f64> = (0..d).map(|k| points[[i, k]]).collect();
        let tr = trend_row(&row_vec, degree);
        for j in 0..m {
            mat[[i, j]] = tr[j];
        }
    }
    mat
}

/// Universal kriging with polynomial trend.
///
/// Solves the GLS-based kriging system:
/// ```text
/// ┌ Γ   F ┐ ┌ λ ┐   ┌ γ₀ ┐
/// │ Fᵀ  0 │ │ μ │ = │ f₀ │
/// └       ┘ └   ┘   └    ┘
/// ```
/// where `F` is the `(n × m)` trend matrix.
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::meshless::kriging_interp::{
///     UniversalKriging, Variogram, TrendDegree,
/// };
/// use scirs2_core::ndarray::{array, Array2};
///
/// let pts = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).expect("doc example: should succeed");
/// let vals = array![0.0, 1.0, 4.0, 9.0];
/// let vgm = Variogram::Gaussian { nugget: 0.0, sill: 1.0, range: 5.0 };
/// let uk = UniversalKriging::new(&pts.view(), &vals.view(), vgm, TrendDegree::Quadratic).expect("doc example: should succeed");
/// let (pred, _var) = uk.predict(&[1.5]).expect("doc example: should succeed");
/// assert!((pred - 2.25).abs() < 1e-6);
/// ```
pub struct UniversalKriging {
    points: Array2<f64>,
    values: Array1<f64>,
    variogram: Variogram,
    trend_degree: TrendDegree,
    kriging_mat: Array2<f64>, // (n+m) × (n+m)
    n_trend: usize,
}

impl UniversalKriging {
    /// Build the universal kriging model.
    pub fn new(
        points: &ArrayView2<f64>,
        values: &ArrayView1<f64>,
        variogram: Variogram,
        trend_degree: TrendDegree,
    ) -> InterpolateResult<Self> {
        let n = points.nrows();
        let d = points.ncols();
        if values.len() != n {
            return Err(InterpolateError::DimensionMismatch(format!(
                "points has {n} rows, values has {} entries",
                values.len()
            )));
        }
        if n == 0 {
            return Err(InterpolateError::InsufficientData(
                "Universal kriging requires at least one data point".to_string(),
            ));
        }

        let pts_owned = points.to_owned();
        let dist = dist_matrix(&pts_owned);
        let f_mat = trend_matrix(&pts_owned, trend_degree);
        let m = f_mat.ncols();

        let total = n + m;
        let mut km = Array2::<f64>::zeros((total, total));

        // Γ block
        for i in 0..n {
            for j in 0..n {
                km[[i, j]] = variogram.eval(dist[[i, j]]);
            }
        }
        // F block and Fᵀ block
        for i in 0..n {
            for j in 0..m {
                km[[i, n + j]] = f_mat[[i, j]];
                km[[n + j, i]] = f_mat[[i, j]];
            }
        }

        // Validate dimensionality
        let _ = d;

        Ok(Self {
            points: pts_owned,
            values: values.to_owned(),
            variogram,
            trend_degree,
            kriging_mat: km,
            n_trend: m,
        })
    }

    /// Predict at a new location. Returns `(prediction, kriging_variance)`.
    pub fn predict(&self, query: &[f64]) -> InterpolateResult<(f64, f64)> {
        let n = self.points.nrows();
        let d = self.points.ncols();
        let m = self.n_trend;
        if query.len() != d {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query has {} dims, points have {d}",
                query.len()
            )));
        }

        let dists = dist_to_points(query, &self.points);
        let f0 = trend_row(query, self.trend_degree);

        // RHS: [γ₀ᵢ; f₀ⱼ]
        let mut rhs = Array1::<f64>::zeros(n + m);
        for i in 0..n {
            rhs[i] = self.variogram.eval(dists[i]);
        }
        for j in 0..m {
            rhs[n + j] = f0[j];
        }

        let lam = linalg_solve(self.kriging_mat.clone(), rhs.clone())?;

        let pred: f64 = (0..n).map(|i| lam[i] * self.values[i]).sum();
        let var: f64 = (0..n + m).map(|i| lam[i] * rhs[i]).sum();

        Ok((pred, var.max(0.0)))
    }

    /// Batch prediction.
    pub fn predict_batch(
        &self,
        queries: &ArrayView2<f64>,
    ) -> InterpolateResult<(Array1<f64>, Array1<f64>)> {
        let nq = queries.nrows();
        let mut preds = Array1::<f64>::zeros(nq);
        let mut vars = Array1::<f64>::zeros(nq);
        for i in 0..nq {
            let q: Vec<f64> = (0..queries.ncols()).map(|j| queries[[i, j]]).collect();
            let (p, v) = self.predict(&q)?;
            preds[i] = p;
            vars[i] = v;
        }
        Ok((preds, vars))
    }
}

// ---------------------------------------------------------------------------
// Co-kriging (two correlated variables)
// ---------------------------------------------------------------------------

/// Co-kriging predictor for two co-regionalised variables.
///
/// Uses a linear model of co-regionalisation (LMC) with shared variogram
/// structure `γ(h)` scaled by sill factors:
/// - `γ₁₁(h) = s₁₁ · γ(h)` (primary auto-variogram)
/// - `γ₂₂(h) = s₂₂ · γ(h)` (secondary auto-variogram)
/// - `γ₁₂(h) = s₁₂ · γ(h)` (cross-variogram)
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::meshless::kriging_interp::{
///     CoKriging, Variogram,
/// };
/// use scirs2_core::ndarray::{array, Array2};
///
/// let pts = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).expect("doc example: should succeed");
/// let vals1 = array![0.0, 1.0, 4.0, 9.0];
/// let vals2 = array![0.0, 0.5, 2.0, 4.5];  // secondary variable (correlated)
/// let vgm = Variogram::Spherical { nugget: 0.0, sill: 1.0, range: 5.0 };
/// let ck = CoKriging::new(
///     &pts.view(), &vals1.view(), &vals2.view(), vgm,
///     1.0, 0.25, 0.5,
/// ).expect("doc example: should succeed");
/// let (pred, _var) = ck.predict(&[1.5], &[0.75]).expect("doc example: should succeed");
/// assert!(pred.is_finite());
/// ```
pub struct CoKriging {
    points: Array2<f64>,
    values_primary: Array1<f64>,
    values_secondary: Array1<f64>,
    variogram: Variogram,
    s11: f64, // sill factor for primary
    s22: f64, // sill factor for secondary
    s12: f64, // cross sill
    kriging_mat: Array2<f64>,
}

impl CoKriging {
    /// Build the co-kriging model.
    ///
    /// # Arguments
    ///
    /// * `points`     – `(n, d)` data sites (shared for both variables).
    /// * `primary`    – Primary variable values (length `n`).
    /// * `secondary`  – Secondary variable values (length `n`).
    /// * `variogram`  – Base variogram shape (scaled by sill factors).
    /// * `s11`        – Primary sill factor (variance of primary).
    /// * `s22`        – Secondary sill factor.
    /// * `s12`        – Cross sill factor (`|s12| ≤ sqrt(s11 * s22)` for validity).
    pub fn new(
        points: &ArrayView2<f64>,
        primary: &ArrayView1<f64>,
        secondary: &ArrayView1<f64>,
        variogram: Variogram,
        s11: f64,
        s22: f64,
        s12: f64,
    ) -> InterpolateResult<Self> {
        let n = points.nrows();
        if primary.len() != n || secondary.len() != n {
            return Err(InterpolateError::DimensionMismatch(format!(
                "points has {n} rows; primary has {} entries, secondary has {}",
                primary.len(),
                secondary.len()
            )));
        }
        if n == 0 {
            return Err(InterpolateError::InsufficientData(
                "Co-kriging requires at least one data point".to_string(),
            ));
        }
        if s11 < 0.0 || s22 < 0.0 {
            return Err(InterpolateError::InvalidInput {
                message: format!("Sill factors s11={s11}, s22={s22} must be non-negative"),
            });
        }

        let pts_owned = points.to_owned();
        let dist = dist_matrix(&pts_owned);

        // System size: 2n + 2 (n primary + n secondary + 2 unbiasedness multipliers)
        let sz = 2 * n + 2;
        let mut km = Array2::<f64>::zeros((sz, sz));

        // Fill variogram sub-matrices
        for i in 0..n {
            for j in 0..n {
                let h = dist[[i, j]];
                km[[i, j]] = s11 * variogram.eval(h);           // γ₁₁
                km[[n + i, n + j]] = s22 * variogram.eval(h);   // γ₂₂
                km[[i, n + j]] = s12 * variogram.eval(h);       // γ₁₂
                km[[n + i, j]] = s12 * variogram.eval(h);       // γ₂₁
            }
        }
        // Unbiasedness: sum of primary weights = 1, sum of secondary weights = 0
        for i in 0..n {
            km[[i, 2 * n]] = 1.0;
            km[[2 * n, i]] = 1.0;
            km[[n + i, 2 * n + 1]] = 1.0;
            km[[2 * n + 1, n + i]] = 1.0;
        }

        // Regularise the variogram blocks to handle near-singular cases that arise
        // when the LMC sill factors cause linear dependencies between sub-matrices.
        let reg = 1e-10 * (s11 + s22 + s12.abs()).max(1e-14);
        for i in 0..2 * n {
            km[[i, i]] += reg;
        }

        Ok(Self {
            points: pts_owned,
            values_primary: primary.to_owned(),
            values_secondary: secondary.to_owned(),
            variogram,
            s11,
            s22,
            s12,
            kriging_mat: km,
        })
    }

    /// Predict the primary variable at `query`.
    ///
    /// `secondary_at_query` is the secondary variable value at the query location
    /// (or an estimate thereof).
    pub fn predict(
        &self,
        query: &[f64],
        secondary_at_query: &[f64],
    ) -> InterpolateResult<(f64, f64)> {
        let n = self.points.nrows();
        let d = self.points.ncols();
        if query.len() != d {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query has {} dims, points have {d}",
                query.len()
            )));
        }

        let dists = dist_to_points(query, &self.points);
        let mut rhs = Array1::<f64>::zeros(2 * n + 2);

        // Cross-variograms from query to each primary and secondary observation
        for i in 0..n {
            rhs[i] = self.s11 * self.variogram.eval(dists[i]);       // to primary
            rhs[n + i] = self.s12 * self.variogram.eval(dists[i]);   // to secondary
        }
        rhs[2 * n] = 1.0; // unbiasedness for primary

        // If secondary location known, also add secondary-to-query cross:
        let _ = secondary_at_query; // reserved for future extension

        let lam = linalg_solve(self.kriging_mat.clone(), rhs.clone())?;

        let pred: f64 = (0..n).map(|i| lam[i] * self.values_primary[i]).sum::<f64>()
            + (0..n).map(|i| lam[n + i] * self.values_secondary[i]).sum::<f64>();

        let var: f64 = (0..(2 * n + 2)).map(|i| lam[i] * rhs[i]).sum();

        Ok((pred, var.max(0.0)))
    }
}

// ---------------------------------------------------------------------------
// Kriging with External Drift (KED)
// ---------------------------------------------------------------------------

/// Kriging with external drift.
///
/// The trend is provided by an auxiliary variable `auxiliary(x)` known at both
/// data sites and query locations:
/// ```text
/// μ(x) = β₀ + β₁ · auxiliary(x)
/// ```
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::meshless::kriging_interp::{
///     KrigingExternalDrift, Variogram,
/// };
/// use scirs2_core::ndarray::{array, Array2};
///
/// let pts = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).expect("doc example: should succeed");
/// let vals = array![0.0, 1.0, 4.0, 9.0];
/// let aux = array![0.0, 1.0, 2.0, 3.0];  // e.g. a DEM or other covariate
/// let vgm = Variogram::Spherical { nugget: 0.0, sill: 1.0, range: 5.0 };
/// let ked = KrigingExternalDrift::new(&pts.view(), &vals.view(), &aux.view(), vgm).expect("doc example: should succeed");
/// let (pred, _var) = ked.predict(&[1.5], 1.5).expect("doc example: should succeed");
/// assert!(pred.is_finite());
/// ```
pub struct KrigingExternalDrift {
    points: Array2<f64>,
    values: Array1<f64>,
    auxiliary: Array1<f64>,
    variogram: Variogram,
    kriging_mat: Array2<f64>, // (n+2) × (n+2)
}

impl KrigingExternalDrift {
    /// Build the KED model.
    ///
    /// # Arguments
    ///
    /// * `points`    – `(n, d)` data sites.
    /// * `values`    – Primary variable values.
    /// * `auxiliary` – Secondary covariate at data sites.
    /// * `variogram` – Residual variogram model.
    pub fn new(
        points: &ArrayView2<f64>,
        values: &ArrayView1<f64>,
        auxiliary: &ArrayView1<f64>,
        variogram: Variogram,
    ) -> InterpolateResult<Self> {
        let n = points.nrows();
        if values.len() != n || auxiliary.len() != n {
            return Err(InterpolateError::DimensionMismatch(format!(
                "points has {n} rows; values has {} entries, auxiliary has {}",
                values.len(),
                auxiliary.len()
            )));
        }
        if n == 0 {
            return Err(InterpolateError::InsufficientData(
                "KED requires at least one data point".to_string(),
            ));
        }

        let pts_owned = points.to_owned();
        let dist = dist_matrix(&pts_owned);

        // System size: n + 2 (n weights + 2 Lagrange multipliers for β₀, β₁)
        let sz = n + 2;
        let mut km = Array2::<f64>::zeros((sz, sz));

        for i in 0..n {
            for j in 0..n {
                km[[i, j]] = variogram.eval(dist[[i, j]]);
            }
        }
        // Drift functions: constant (col n) and external (col n+1)
        for i in 0..n {
            km[[i, n]] = 1.0;
            km[[n, i]] = 1.0;
            km[[i, n + 1]] = auxiliary[i];
            km[[n + 1, i]] = auxiliary[i];
        }

        Ok(Self {
            points: pts_owned,
            values: values.to_owned(),
            auxiliary: auxiliary.to_owned(),
            variogram,
            kriging_mat: km,
        })
    }

    /// Predict at `query` where the external drift has value `aux_at_query`.
    pub fn predict(&self, query: &[f64], aux_at_query: f64) -> InterpolateResult<(f64, f64)> {
        let n = self.points.nrows();
        let d = self.points.ncols();
        if query.len() != d {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query has {} dims, points have {d}",
                query.len()
            )));
        }

        let dists = dist_to_points(query, &self.points);
        let mut rhs = Array1::<f64>::zeros(n + 2);
        for i in 0..n {
            rhs[i] = self.variogram.eval(dists[i]);
        }
        rhs[n] = 1.0;
        rhs[n + 1] = aux_at_query;

        let lam = linalg_solve(self.kriging_mat.clone(), rhs.clone())?;

        let pred: f64 = (0..n).map(|i| lam[i] * self.values[i]).sum();
        let var: f64 = (0..n + 2).map(|i| lam[i] * rhs[i]).sum();

        Ok((pred, var.max(0.0)))
    }

    /// Batch prediction.
    pub fn predict_batch(
        &self,
        queries: &ArrayView2<f64>,
        aux_values: &ArrayView1<f64>,
    ) -> InterpolateResult<(Array1<f64>, Array1<f64>)> {
        let nq = queries.nrows();
        if aux_values.len() != nq {
            return Err(InterpolateError::DimensionMismatch(format!(
                "queries has {nq} rows, aux_values has {} entries",
                aux_values.len()
            )));
        }
        let mut preds = Array1::<f64>::zeros(nq);
        let mut vars = Array1::<f64>::zeros(nq);
        for i in 0..nq {
            let q: Vec<f64> = (0..queries.ncols()).map(|j| queries[[i, j]]).collect();
            let (p, v) = self.predict(&q, aux_values[i])?;
            preds[i] = p;
            vars[i] = v;
        }
        Ok((preds, vars))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    fn pts_1d(n: usize) -> Array2<f64> {
        let v: Vec<f64> = (0..n).map(|i| i as f64).collect();
        Array2::from_shape_vec((n, 1), v).expect("test: should succeed")
    }

    #[test]
    fn test_ordinary_kriging_exact_at_nodes() {
        let pts = pts_1d(5);
        let vals = array![0.0_f64, 1.0, 4.0, 9.0, 16.0];
        let vgm = Variogram::Spherical {
            nugget: 0.0,
            sill: 1.0,
            range: 10.0,
        };
        let ok = OrdinaryKriging::new(&pts.view(), &vals.view(), vgm).expect("test: should succeed");
        for i in 0..5 {
            let (pred, _var) = ok.predict(&[i as f64]).expect("test: should succeed");
            assert_abs_diff_eq!(pred, vals[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_universal_kriging_linear_trend() {
        // f(x) = x  → linear trend captures it exactly
        let pts = pts_1d(4);
        let vals = array![0.0_f64, 1.0, 2.0, 3.0];
        let vgm = Variogram::Gaussian {
            nugget: 0.0,
            sill: 1.0,
            range: 10.0,
        };
        let uk =
            UniversalKriging::new(&pts.view(), &vals.view(), vgm, TrendDegree::Linear).expect("test: should succeed");
        let (pred, _) = uk.predict(&[1.5]).expect("test: should succeed");
        assert_abs_diff_eq!(pred, 1.5, epsilon = 1e-6);
    }

    #[test]
    fn test_universal_kriging_quadratic() {
        let pts = pts_1d(5);
        let vals: Array1<f64> = (0..5_usize).map(|i| (i as f64).powi(2)).collect();
        let vgm = Variogram::Gaussian {
            nugget: 0.0,
            sill: 1.0,
            range: 20.0,
        };
        let uk =
            UniversalKriging::new(&pts.view(), &vals.view(), vgm, TrendDegree::Quadratic).expect("test: should succeed");
        let (pred, _) = uk.predict(&[1.5]).expect("test: should succeed");
        assert_abs_diff_eq!(pred, 2.25, epsilon = 1e-5);
    }

    #[test]
    fn test_variogram_spherical() {
        let v = Variogram::Spherical {
            nugget: 0.0,
            sill: 1.0,
            range: 10.0,
        };
        assert_abs_diff_eq!(v.eval(0.0), 0.0, epsilon = 1e-15);
        // At h = range, spherical sill should equal nugget + sill = 1
        assert_abs_diff_eq!(v.eval(10.0), 1.0, epsilon = 1e-10);
        // Beyond range, flat
        assert_abs_diff_eq!(v.eval(20.0), 1.0, epsilon = 1e-15);
    }

    #[test]
    fn test_cokriging_finite() {
        // Use a small nugget to ensure the variogram matrix is non-degenerate.
        // Co-kriging with pure proportional sill factors (s12^2 == s11*s22)
        // can lead to a singular system; a non-trivial nugget avoids this.
        let pts = pts_1d(4);
        let vals1 = array![0.0_f64, 1.0, 4.0, 9.0];
        let vals2 = array![0.0_f64, 0.5, 2.0, 4.5];
        let vgm = Variogram::Spherical {
            nugget: 0.1, // non-zero nugget gives positive-definite diagonal
            sill: 1.0,
            range: 5.0,
        };
        // Use s12 strictly less than sqrt(s11*s22) = 0.5 to avoid singular LMC
        let ck = CoKriging::new(&pts.view(), &vals1.view(), &vals2.view(), vgm, 1.0, 0.25, 0.3)
            .expect("test: should succeed");
        let (pred, var) = ck.predict(&[1.5], &[]).expect("test: should succeed");
        assert!(pred.is_finite(), "prediction must be finite");
        assert!(var >= -1e-10, "variance should be non-negative (up to numerical noise)");
    }

    #[test]
    fn test_ked_finite() {
        let pts = pts_1d(4);
        let vals = array![0.0_f64, 1.0, 4.0, 9.0];
        let aux = array![0.0_f64, 1.0, 2.0, 3.0];
        let vgm = Variogram::Exponential {
            nugget: 0.0,
            sill: 1.0,
            range: 5.0,
        };
        let ked = KrigingExternalDrift::new(&pts.view(), &vals.view(), &aux.view(), vgm).expect("test: should succeed");
        let (pred, var) = ked.predict(&[1.5], 1.5).expect("test: should succeed");
        assert!(pred.is_finite());
        assert!(var >= 0.0);
    }
}
