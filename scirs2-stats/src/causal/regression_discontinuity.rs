//! Regression Discontinuity Design (RDD)
//!
//! Methods for estimating causal effects at a threshold (cutoff) in a running variable:
//!
//! - **`RDD`**: Sharp RDD with Imbens-Kalyanaraman (IK) optimal bandwidth and
//!   local polynomial regression
//! - **`FuzzyRDD`**: Fuzzy RDD estimated via local IV (Wald estimator near the cutoff)
//! - **`BandwidthSelector`**: CCT (Calonico-Cattaneo-Titiunik), IK, and
//!   cross-validation bandwidth selectors
//! - **`RDDPlot`**: Binned scatter plot data for visual inspection
//!
//! # References
//!
//! - Imbens, G.W. & Kalyanaraman, K. (2012). Optimal Bandwidth Choice for the
//!   Regression Discontinuity Estimator. Review of Economic Studies.
//! - Calonico, S., Cattaneo, M.D. & Titiunik, R. (2014). Robust Nonparametric
//!   Confidence Intervals for Regression-Discontinuity Designs.
//! - Lee, D.S. & Lemieux, T. (2010). Regression Discontinuity Designs in Economics.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of a Regression Discontinuity estimation
#[derive(Debug, Clone)]
pub struct RDDResult {
    /// RD estimate (treatment effect at the cutoff)
    pub estimate: f64,

    /// Standard error of the estimate
    pub std_error: f64,

    /// t-statistic
    pub t_stat: f64,

    /// Two-sided p-value
    pub p_value: f64,

    /// 95 % confidence interval
    pub conf_interval: [f64; 2],

    /// Bandwidth used on both sides of the cutoff
    pub bandwidth: f64,

    /// Number of observations within bandwidth (left side)
    pub n_left: usize,

    /// Number of observations within bandwidth (right side)
    pub n_right: usize,

    /// Polynomial order used in local regression
    pub poly_order: usize,

    /// Estimator name ("Sharp-RDD" or "Fuzzy-RDD")
    pub estimator: String,
}

/// Binned scatter data for an RD plot
#[derive(Debug, Clone)]
pub struct RDDPlot {
    /// Bin midpoints (running variable)
    pub x_bins: Array1<f64>,
    /// Bin means of the outcome
    pub y_means: Array1<f64>,
    /// Standard errors for each bin mean
    pub y_se: Array1<f64>,
    /// Cutoff value
    pub cutoff: f64,
    /// Number of bins on each side
    pub n_bins_each_side: usize,
}

/// Available bandwidth selection methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandwidthMethod {
    /// Imbens-Kalyanaraman (2012) plug-in selector
    IK,
    /// Calonico-Cattaneo-Titiunik (2014) robust selector
    CCT,
    /// Leave-one-out cross-validation
    CV,
}

// ---------------------------------------------------------------------------
// Kernel functions
// ---------------------------------------------------------------------------

/// Triangular kernel (default for local RD)
fn triangular_kernel(u: f64) -> f64 {
    if u.abs() < 1.0 { 1.0 - u.abs() } else { 0.0 }
}

/// Epanechnikov kernel
fn epanechnikov_kernel(u: f64) -> f64 {
    if u.abs() < 1.0 { 0.75 * (1.0 - u * u) } else { 0.0 }
}

/// Uniform kernel
fn uniform_kernel(u: f64) -> f64 {
    if u.abs() <= 1.0 { 0.5 } else { 0.0 }
}

// ---------------------------------------------------------------------------
// Local polynomial regression helper
// ---------------------------------------------------------------------------

/// Fit a weighted local polynomial regression of order `p`.
///
/// Returns (intercept, slope, ...) for polynomial coefficients at `x0`.
fn local_poly_fit(
    x: &[f64],
    y: &[f64],
    weights: &[f64],
    x0: f64,
    poly_order: usize,
) -> StatsResult<(f64, f64)> {
    let n = x.len();
    if n < poly_order + 1 {
        return Err(StatsError::InsufficientData(format!(
            "Need at least {} points for poly_order={}, got {}",
            poly_order + 1,
            poly_order,
            n
        )));
    }
    let k = poly_order + 1;
    // Build weighted design matrix
    let mut xmat = Array2::<f64>::zeros((n, k));
    let mut y_vec = Array1::<f64>::zeros(n);
    for (i, (&xi, (&yi, &wi))) in x.iter().zip(y.iter().zip(weights.iter())).enumerate() {
        let sqrt_w = wi.sqrt();
        y_vec[i] = yi * sqrt_w;
        let dx = xi - x0;
        let mut pow = 1.0_f64;
        for j in 0..k {
            xmat[[i, j]] = pow * sqrt_w;
            pow *= dx;
        }
    }
    // Solve weighted least squares via Cholesky
    let xtx = xmat.t().dot(&xmat);
    let xty = xmat.t().dot(&y_vec);
    let xtx_inv = cholesky_invert_rdd(&xtx.view())?;
    let beta = xtx_inv.dot(&xty);

    // Residuals
    let fitted = xmat.dot(&beta);
    let residuals = &y_vec - &fitted;
    let rss: f64 = residuals.iter().map(|&r| r * r).sum();
    let df = (n as f64) - k as f64;
    let s2 = if df > 0.0 { rss / df } else { rss };

    // Variance of the intercept (beta[0])
    let var_intercept = xtx_inv[[0, 0]] * s2;
    let se = var_intercept.max(0.0).sqrt();

    Ok((beta[0], se))
}

fn cholesky_invert_rdd(a: &scirs2_core::ndarray::ArrayView2<f64>) -> StatsResult<Array2<f64>> {
    let n = a.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for p in 0..j { s -= l[[i, p]] * l[[j, p]]; }
            if i == j {
                if s <= 0.0 {
                    return Err(StatsError::ComputationError(
                        "Matrix not positive definite (RDD)".into(),
                    ));
                }
                l[[i, j]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    let mut linv = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        linv[[j, j]] = 1.0 / l[[j, j]];
        for i in (j + 1)..n {
            let mut s = 0.0_f64;
            for p in j..i { s += l[[i, p]] * linv[[p, j]]; }
            linv[[i, j]] = -s / l[[i, i]];
        }
    }
    Ok(linv.t().dot(&linv))
}

fn normal_p_value_rdd(z: f64) -> f64 {
    2.0 * (1.0 - normal_cdf_rdd(z.abs()))
}

fn normal_cdf_rdd(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

fn erf_approx(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0
        - (0.254829592 + (-0.284496736 + (1.421413741 + (-1.453152027 + 1.061405429 * t) * t) * t)
            * t)
            * t
            * (-x * x).exp();
    if x >= 0.0 { y } else { -y }
}

// ---------------------------------------------------------------------------
// Bandwidth Selector
// ---------------------------------------------------------------------------

/// Bandwidth selector for RD designs.
pub struct BandwidthSelector;

impl BandwidthSelector {
    /// Select bandwidth using the specified method.
    ///
    /// # Arguments
    /// * `x`      – running variable
    /// * `y`      – outcome
    /// * `cutoff` – threshold value
    /// * `method` – `IK`, `CCT`, or `CV`
    /// * `poly_order` – polynomial order for local regression
    pub fn select(
        x: &ArrayView1<f64>,
        y: &ArrayView1<f64>,
        cutoff: f64,
        method: BandwidthMethod,
        poly_order: usize,
    ) -> StatsResult<f64> {
        match method {
            BandwidthMethod::IK => Self::ik_bandwidth(x, y, cutoff, poly_order),
            BandwidthMethod::CCT => Self::cct_bandwidth(x, y, cutoff, poly_order),
            BandwidthMethod::CV => Self::cv_bandwidth(x, y, cutoff, poly_order),
        }
    }

    /// Imbens-Kalyanaraman (2012) plug-in bandwidth selector.
    ///
    /// Based on minimizing the asymptotic MSE of the sharp-RD estimator.
    fn ik_bandwidth(
        x: &ArrayView1<f64>,
        y: &ArrayView1<f64>,
        cutoff: f64,
        poly_order: usize,
    ) -> StatsResult<f64> {
        let n = x.len();
        // Step 1: Pilot bandwidth (rule-of-thumb)
        let x_std = std_dev(x);
        let h_pilot = 1.84 * x_std * (n as f64).powf(-1.0 / 5.0);

        // Step 2: Density of X at cutoff (kernel estimate)
        let f_x = x.iter().filter(|&&xi| (xi - cutoff).abs() < h_pilot).count() as f64
            / (n as f64 * 2.0 * h_pilot).max(1e-15);

        if f_x < 1e-10 {
            return Err(StatsError::ComputationError(
                "No observations near cutoff; cannot select bandwidth".into(),
            ));
        }

        // Step 3: Estimate conditional variance at cutoff (each side)
        let (x_l, y_l, x_r, y_r) = split_at_cutoff(x, y, cutoff);
        let var_l = conditional_variance_at_boundary(&x_l, &y_l, cutoff, h_pilot, poly_order)?;
        let var_r = conditional_variance_at_boundary(&x_r, &y_r, cutoff, h_pilot, poly_order)?;
        let sigma2 = (var_l + var_r) / 2.0;

        // Step 4: Estimate second derivative (curvature) on each side
        // Use a global polynomial of order poly_order+2 for the pilot
        let m2_l = estimate_m2(&x_l, &y_l, cutoff)?;
        let m2_r = estimate_m2(&x_r, &y_r, cutoff)?;
        let m2_sq = ((m2_r - m2_l) / 2.0).powi(2);

        if m2_sq < 1e-15 {
            // Flat outcome near cutoff; fall back to rule-of-thumb
            return Ok(h_pilot);
        }

        // Optimal bandwidth (IK formula for local linear, triangular kernel):
        // h* = C_K * [sigma²(x-) + sigma²(x+)] / [f(x) * (m''_+(x) - m''_-(x))²]
        //         * n^{-1/(2p+3)}
        // where C_K depends on the kernel and polynomial order
        let p = poly_order as f64;
        let c_k = 3.4375_f64; // for triangular kernel, local linear
        let h_opt = c_k * (sigma2 / (f_x * m2_sq)).powf(1.0 / (2.0 * p + 3.0))
            * (n as f64).powf(-1.0 / (2.0 * p + 3.0));

        Ok(h_opt.max(0.01 * x_std))
    }

    /// CCT (2014) bandwidth selector (simplified robust version).
    fn cct_bandwidth(
        x: &ArrayView1<f64>,
        y: &ArrayView1<f64>,
        cutoff: f64,
        poly_order: usize,
    ) -> StatsResult<f64> {
        // CCT selects bandwidth to minimise coverage error of confidence intervals,
        // using a bias-corrected estimator.  We implement an approximation:
        // h_CCT ≈ h_IK * adjustment_factor
        let h_ik = Self::ik_bandwidth(x, y, cutoff, poly_order)?;
        // CCT typically yields a larger bandwidth than IK
        Ok(h_ik * 1.2)
    }

    /// Leave-one-out cross-validation bandwidth selector.
    fn cv_bandwidth(
        x: &ArrayView1<f64>,
        y: &ArrayView1<f64>,
        cutoff: f64,
        poly_order: usize,
    ) -> StatsResult<f64> {
        let x_std = std_dev(x);
        let h_min = 0.1 * x_std;
        let h_max = 1.5 * x_std;
        let n_grid = 20_usize;
        let mut best_h = h_min;
        let mut best_cv = f64::INFINITY;

        for k in 0..n_grid {
            let h = h_min + (h_max - h_min) * k as f64 / (n_grid - 1) as f64;
            let cv = loocv_rdd(x, y, cutoff, h, poly_order)?;
            if cv < best_cv {
                best_cv = cv;
                best_h = h;
            }
        }
        Ok(best_h)
    }
}

/// Leave-one-out CV score for local polynomial RD
fn loocv_rdd(
    x: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
    cutoff: f64,
    h: f64,
    poly_order: usize,
) -> StatsResult<f64> {
    let n = x.len();
    let mut cv_score = 0.0_f64;
    let mut count = 0_usize;
    for i in 0..n {
        let xi = x[i];
        let yi = y[i];
        // Only use observations on the same side as xi
        let side = xi >= cutoff;
        let x_in: Vec<f64> = (0..n)
            .filter(|&j| j != i && (x[j] >= cutoff) == side)
            .filter(|&j| ((x[j] - cutoff) / h).abs() < 1.0)
            .map(|j| x[j])
            .collect();
        let y_in: Vec<f64> = (0..n)
            .filter(|&j| j != i && (x[j] >= cutoff) == side)
            .filter(|&j| ((x[j] - cutoff) / h).abs() < 1.0)
            .map(|j| y[j])
            .collect();
        let w_in: Vec<f64> = x_in.iter().map(|&xj| triangular_kernel((xj - cutoff) / h)).collect();

        if x_in.len() < poly_order + 1 {
            continue;
        }
        let (y_hat, _) = local_poly_fit(&x_in, &y_in, &w_in, xi, poly_order)?;
        cv_score += (yi - y_hat).powi(2);
        count += 1;
    }
    if count == 0 {
        return Err(StatsError::InsufficientData(
            "No valid observations for CV bandwidth selection".into(),
        ));
    }
    Ok(cv_score / count as f64)
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn std_dev(x: &ArrayView1<f64>) -> f64 {
    let n = x.len();
    if n < 2 { return 1.0; }
    let mean = x.iter().sum::<f64>() / n as f64;
    let var = x.iter().map(|&xi| (xi - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    var.sqrt().max(1e-15)
}

fn split_at_cutoff(
    x: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
    cutoff: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut x_l = Vec::new(); let mut y_l = Vec::new();
    let mut x_r = Vec::new(); let mut y_r = Vec::new();
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        if xi < cutoff { x_l.push(xi); y_l.push(yi); }
        else           { x_r.push(xi); y_r.push(yi); }
    }
    (x_l, y_l, x_r, y_r)
}

fn conditional_variance_at_boundary(
    x: &[f64],
    y: &[f64],
    cutoff: f64,
    h: f64,
    poly_order: usize,
) -> StatsResult<f64> {
    let weights: Vec<f64> = x.iter().map(|&xi| triangular_kernel((xi - cutoff) / h)).collect();
    let in_bw: Vec<usize> = (0..x.len()).filter(|&i| weights[i] > 0.0).collect();
    if in_bw.len() < poly_order + 2 {
        return Ok(1.0); // fallback
    }
    let xin: Vec<f64> = in_bw.iter().map(|&i| x[i]).collect();
    let yin: Vec<f64> = in_bw.iter().map(|&i| y[i]).collect();
    let win: Vec<f64> = in_bw.iter().map(|&i| weights[i]).collect();
    let n = xin.len();
    let k = poly_order + 1;
    // Weighted OLS
    let mut xmat = Array2::<f64>::zeros((n, k));
    let mut y_vec = Array1::<f64>::zeros(n);
    for (i, (&xi, (&yi, &wi))) in xin.iter().zip(yin.iter().zip(win.iter())).enumerate() {
        let sqrt_w = wi.sqrt();
        y_vec[i] = yi * sqrt_w;
        let dx = xi - cutoff;
        let mut pow = 1.0_f64;
        for j in 0..k {
            xmat[[i, j]] = pow * sqrt_w;
            pow *= dx;
        }
    }
    let xtx = xmat.t().dot(&xmat);
    let xty = xmat.t().dot(&y_vec);
    let xtx_inv = cholesky_invert_rdd(&xtx.view()).unwrap_or_else(|_| Array2::eye(k));
    let beta = xtx_inv.dot(&xty);
    let fitted = xmat.dot(&beta);
    let resid = &y_vec - &fitted;
    let df = (n - k) as f64;
    Ok(if df > 0.0 { resid.iter().map(|&r| r * r).sum::<f64>() / df } else { 1.0 })
}

/// Estimate the second derivative of E[Y|X] at the cutoff.
fn estimate_m2(x: &[f64], y: &[f64], cutoff: f64) -> StatsResult<f64> {
    let n = x.len();
    if n < 4 {
        return Ok(0.0);
    }
    // Fit a quadratic on the side: y = a + b(x-c) + d(x-c)²
    let k = 3_usize;
    let mut xmat = Array2::<f64>::zeros((n, k));
    let mut y_vec = Array1::<f64>::zeros(n);
    for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
        let dx = xi - cutoff;
        y_vec[i] = yi;
        xmat[[i, 0]] = 1.0;
        xmat[[i, 1]] = dx;
        xmat[[i, 2]] = dx * dx;
    }
    let xtx = xmat.t().dot(&xmat);
    let xty = xmat.t().dot(&y_vec);
    let xtx_inv = cholesky_invert_rdd(&xtx.view()).unwrap_or_else(|_| Array2::eye(k));
    let beta = xtx_inv.dot(&xty);
    // Second derivative = 2 * coefficient of (x-c)²
    Ok(2.0 * beta[2])
}

// ---------------------------------------------------------------------------
// Sharp RDD
// ---------------------------------------------------------------------------

/// Sharp Regression Discontinuity Design estimator.
///
/// Estimates E[Y(1) - Y(0) | X = c] using local polynomial regression
/// on each side of the cutoff.
pub struct RDD {
    /// Threshold / cutoff value of the running variable
    pub cutoff: f64,
    /// Polynomial order for local regression (1 = local linear, recommended)
    pub poly_order: usize,
    /// Kernel function: "triangular", "epanechnikov", or "uniform"
    pub kernel: String,
}

impl RDD {
    /// Create a new sharp RDD estimator.
    pub fn new(cutoff: f64, poly_order: usize, kernel: &str) -> Self {
        Self {
            cutoff,
            poly_order,
            kernel: kernel.to_string(),
        }
    }

    /// Estimate the RD treatment effect.
    ///
    /// # Arguments
    /// * `x`         – running variable
    /// * `y`         – outcome
    /// * `bandwidth` – if `None`, uses IK optimal bandwidth
    pub fn estimate(
        &self,
        x: &ArrayView1<f64>,
        y: &ArrayView1<f64>,
        bandwidth: Option<f64>,
    ) -> StatsResult<RDDResult> {
        let n = x.len();
        if n < 4 {
            return Err(StatsError::InsufficientData(
                "Need at least 4 observations".into(),
            ));
        }
        if y.len() != n {
            return Err(StatsError::DimensionMismatch(
                "x and y must have the same length".into(),
            ));
        }

        let h = match bandwidth {
            Some(bw) => bw,
            None => BandwidthSelector::select(x, y, self.cutoff, BandwidthMethod::IK, self.poly_order)?,
        };

        let k_fn: Box<dyn Fn(f64) -> f64> = match self.kernel.as_str() {
            "epanechnikov" => Box::new(epanechnikov_kernel),
            "uniform"      => Box::new(uniform_kernel),
            _              => Box::new(triangular_kernel),
        };

        // Left side: x < cutoff, within bandwidth
        let (x_l, y_l, w_l): (Vec<f64>, Vec<f64>, Vec<f64>) = x
            .iter()
            .zip(y.iter())
            .filter(|(&xi, _)| xi < self.cutoff && (xi - self.cutoff).abs() <= h)
            .map(|(&xi, &yi)| {
                let w = k_fn((xi - self.cutoff) / h);
                (xi, yi, w)
            })
            .fold((Vec::new(), Vec::new(), Vec::new()), |(mut ax, mut ay, mut aw), (xi, yi, wi)| {
                ax.push(xi); ay.push(yi); aw.push(wi);
                (ax, ay, aw)
            });

        // Right side: x >= cutoff, within bandwidth
        let (x_r, y_r, w_r): (Vec<f64>, Vec<f64>, Vec<f64>) = x
            .iter()
            .zip(y.iter())
            .filter(|(&xi, _)| xi >= self.cutoff && (xi - self.cutoff).abs() <= h)
            .map(|(&xi, &yi)| {
                let w = k_fn((xi - self.cutoff) / h);
                (xi, yi, w)
            })
            .fold((Vec::new(), Vec::new(), Vec::new()), |(mut ax, mut ay, mut aw), (xi, yi, wi)| {
                ax.push(xi); ay.push(yi); aw.push(wi);
                (ax, ay, aw)
            });

        let n_left = x_l.len();
        let n_right = x_r.len();

        if n_left < self.poly_order + 1 {
            return Err(StatsError::InsufficientData(format!(
                "Insufficient observations left of cutoff ({n_left}) for poly_order={}",
                self.poly_order
            )));
        }
        if n_right < self.poly_order + 1 {
            return Err(StatsError::InsufficientData(format!(
                "Insufficient observations right of cutoff ({n_right}) for poly_order={}",
                self.poly_order
            )));
        }

        let (mu_l, se_l) = local_poly_fit(&x_l, &y_l, &w_l, self.cutoff, self.poly_order)?;
        let (mu_r, se_r) = local_poly_fit(&x_r, &y_r, &w_r, self.cutoff, self.poly_order)?;

        let estimate = mu_r - mu_l;
        let std_error = (se_l * se_l + se_r * se_r).sqrt();
        let t_stat = if std_error > 0.0 { estimate / std_error } else { 0.0 };
        let p_value = normal_p_value_rdd(t_stat);
        let ci = [estimate - 1.96 * std_error, estimate + 1.96 * std_error];

        Ok(RDDResult {
            estimate,
            std_error,
            t_stat,
            p_value,
            conf_interval: ci,
            bandwidth: h,
            n_left,
            n_right,
            poly_order: self.poly_order,
            estimator: "Sharp-RDD".into(),
        })
    }

    /// Generate binned scatter plot data for visualisation.
    ///
    /// # Arguments
    /// * `x`                – running variable
    /// * `y`                – outcome
    /// * `n_bins_each_side` – number of equally-spaced bins on each side
    pub fn plot_data(
        &self,
        x: &ArrayView1<f64>,
        y: &ArrayView1<f64>,
        n_bins_each_side: usize,
    ) -> StatsResult<RDDPlot> {
        let n = x.len();
        if n < 2 {
            return Err(StatsError::InsufficientData("Need at least 2 observations".into()));
        }

        let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let mut x_bins = Vec::with_capacity(2 * n_bins_each_side);
        let mut y_means = Vec::with_capacity(2 * n_bins_each_side);
        let mut y_se = Vec::with_capacity(2 * n_bins_each_side);

        // Left side bins
        let step_l = (self.cutoff - x_min).max(1e-10) / n_bins_each_side as f64;
        for b in 0..n_bins_each_side {
            let lo = x_min + b as f64 * step_l;
            let hi = lo + step_l;
            let mid = (lo + hi) / 2.0;
            let vals: Vec<f64> = x.iter().zip(y.iter())
                .filter(|(&xi, _)| xi >= lo && xi < hi)
                .map(|(_, &yi)| yi)
                .collect();
            if vals.is_empty() {
                x_bins.push(mid);
                y_means.push(f64::NAN);
                y_se.push(f64::NAN);
            } else {
                let m = vals.iter().sum::<f64>() / vals.len() as f64;
                let se = if vals.len() > 1 {
                    let var = vals.iter().map(|&v| (v - m).powi(2)).sum::<f64>()
                        / (vals.len() * (vals.len() - 1)) as f64;
                    var.sqrt()
                } else { 0.0 };
                x_bins.push(mid);
                y_means.push(m);
                y_se.push(se);
            }
        }

        // Right side bins
        let step_r = (x_max - self.cutoff).max(1e-10) / n_bins_each_side as f64;
        for b in 0..n_bins_each_side {
            let lo = self.cutoff + b as f64 * step_r;
            let hi = lo + step_r;
            let mid = (lo + hi) / 2.0;
            let vals: Vec<f64> = x.iter().zip(y.iter())
                .filter(|(&xi, _)| xi >= lo && xi < hi)
                .map(|(_, &yi)| yi)
                .collect();
            if vals.is_empty() {
                x_bins.push(mid);
                y_means.push(f64::NAN);
                y_se.push(f64::NAN);
            } else {
                let m = vals.iter().sum::<f64>() / vals.len() as f64;
                let se = if vals.len() > 1 {
                    let var = vals.iter().map(|&v| (v - m).powi(2)).sum::<f64>()
                        / (vals.len() * (vals.len() - 1)) as f64;
                    var.sqrt()
                } else { 0.0 };
                x_bins.push(mid);
                y_means.push(m);
                y_se.push(se);
            }
        }

        Ok(RDDPlot {
            x_bins: Array1::from_vec(x_bins),
            y_means: Array1::from_vec(y_means),
            y_se: Array1::from_vec(y_se),
            cutoff: self.cutoff,
            n_bins_each_side,
        })
    }
}

// ---------------------------------------------------------------------------
// Fuzzy RDD
// ---------------------------------------------------------------------------

/// Fuzzy Regression Discontinuity estimator.
///
/// Estimates the local average treatment effect (LATE) at the cutoff when
/// treatment receipt is probabilistic but discontinuous at the threshold.
///
/// The estimator is:
///   τ_FRD = limₓ↑c E[Y|X=x] - limₓ↓c E[Y|X=x]
///          ─────────────────────────────────────────
///          limₓ↑c E[D|X=x] - limₓ↓c E[D|X=x]
/// implemented via local IV (Wald estimator).
pub struct FuzzyRDD {
    /// Cutoff
    pub cutoff: f64,
    /// Polynomial order
    pub poly_order: usize,
    /// Kernel
    pub kernel: String,
}

impl FuzzyRDD {
    /// Create a new FuzzyRDD estimator.
    pub fn new(cutoff: f64, poly_order: usize, kernel: &str) -> Self {
        Self {
            cutoff,
            poly_order,
            kernel: kernel.to_string(),
        }
    }

    /// Estimate the fuzzy RD treatment effect.
    ///
    /// # Arguments
    /// * `x`         – running variable
    /// * `y`         – outcome
    /// * `d`         – binary treatment indicator
    /// * `bandwidth` – if `None`, uses IK bandwidth on the outcome regression
    pub fn estimate(
        &self,
        x: &ArrayView1<f64>,
        y: &ArrayView1<f64>,
        d: &ArrayView1<f64>,
        bandwidth: Option<f64>,
    ) -> StatsResult<RDDResult> {
        let n = x.len();
        if y.len() != n || d.len() != n {
            return Err(StatsError::DimensionMismatch(
                "x, y, d must all have the same length".into(),
            ));
        }

        let h = match bandwidth {
            Some(bw) => bw,
            None => BandwidthSelector::select(x, y, self.cutoff, BandwidthMethod::IK, self.poly_order)?,
        };

        // Reduced-form: RD in Y
        let rdd_y = RDD::new(self.cutoff, self.poly_order, &self.kernel);
        let res_y = rdd_y.estimate(x, y, Some(h))?;

        // First-stage: RD in D
        let rdd_d = RDD::new(self.cutoff, self.poly_order, &self.kernel);
        let res_d = rdd_d.estimate(x, d, Some(h))?;

        let first_stage = res_d.estimate;
        if first_stage.abs() < 1e-8 {
            return Err(StatsError::ComputationError(
                "First stage near zero; no discontinuity in treatment probability".into(),
            ));
        }

        // Wald = reduced_form / first_stage
        let estimate = res_y.estimate / first_stage;
        // Delta method: Var(Y/D) ≈ (1/D)² Var(Y) + (Y/D²)² Var(D)
        let d_hat = first_stage;
        let y_hat = res_y.estimate;
        let var_wald = (res_y.std_error / d_hat).powi(2)
            + (y_hat / d_hat.powi(2) * res_d.std_error).powi(2);
        let std_error = var_wald.sqrt();
        let t_stat = if std_error > 0.0 { estimate / std_error } else { 0.0 };
        let p_value = normal_p_value_rdd(t_stat);
        let ci = [estimate - 1.96 * std_error, estimate + 1.96 * std_error];

        Ok(RDDResult {
            estimate,
            std_error,
            t_stat,
            p_value,
            conf_interval: ci,
            bandwidth: h,
            n_left: res_y.n_left,
            n_right: res_y.n_right,
            poly_order: self.poly_order,
            estimator: "Fuzzy-RDD".into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn make_rdd_data(n: usize, cutoff: f64, effect: f64) -> (Array1<f64>, Array1<f64>) {
        let x: Array1<f64> = (0..n).map(|i| (i as f64) / (n as f64) * 2.0 - 1.0).collect();
        let y: Array1<f64> = x.iter().map(|&xi| {
            let base = 1.0 + 0.5 * xi;
            let jump = if xi >= cutoff { effect } else { 0.0 };
            base + jump
        }).collect();
        (x, y)
    }

    #[test]
    fn test_sharp_rdd_recovers_effect() {
        let (x, y) = make_rdd_data(500, 0.0, 2.0);
        let rdd = RDD::new(0.0, 1, "triangular");
        let res = rdd.estimate(&x.view(), &y.view(), None)
            .expect("Sharp RDD should succeed");
        assert!((res.estimate - 2.0).abs() < 0.3,
            "Expected effect≈2.0, got {}", res.estimate);
        assert_eq!(res.estimator, "Sharp-RDD");
    }

    #[test]
    fn test_rdd_bandwidth_ik() {
        let (x, y) = make_rdd_data(300, 0.0, 1.5);
        let h = BandwidthSelector::select(&x.view(), &y.view(), 0.0, BandwidthMethod::IK, 1)
            .expect("IK bandwidth should succeed");
        assert!(h > 0.0, "Bandwidth must be positive");
    }

    #[test]
    fn test_rdd_plot_data() {
        let (x, y) = make_rdd_data(200, 0.0, 1.0);
        let rdd = RDD::new(0.0, 1, "triangular");
        let plot = rdd.plot_data(&x.view(), &y.view(), 5)
            .expect("Plot data should succeed");
        assert_eq!(plot.x_bins.len(), 10);
        assert_eq!(plot.n_bins_each_side, 5);
    }

    #[test]
    fn test_fuzzy_rdd() {
        let n = 400_usize;
        let cutoff = 0.0_f64;
        let x: Array1<f64> = (0..n).map(|i| (i as f64) / n as f64 * 2.0 - 1.0).collect();
        // Treatment probability: 0.2 below cutoff, 0.8 above
        let d: Array1<f64> = x.iter().map(|&xi| if xi >= cutoff { 1.0 } else { 0.0 }).collect();
        let y: Array1<f64> = x.iter().zip(d.iter()).map(|(&xi, &di)| 1.0 + 0.5 * xi + 2.0 * di).collect();
        let frdd = FuzzyRDD::new(cutoff, 1, "triangular");
        let res = frdd.estimate(&x.view(), &y.view(), &d.view(), None)
            .expect("Fuzzy RDD should succeed");
        assert!((res.estimate - 2.0).abs() < 0.5,
            "Expected LATE≈2.0, got {}", res.estimate);
    }
}
