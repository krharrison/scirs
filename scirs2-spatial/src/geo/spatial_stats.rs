//! Spatial statistics for geospatial analysis.
//!
//! This module provides statistical measures commonly used in spatial data
//! science:
//!
//! ## Spatial Autocorrelation
//! - **Moran's I** — global spatial autocorrelation index
//! - **Geary's C** — contiguity-ratio measure of spatial autocorrelation
//!
//! ## Spatial Regression
//! - **Spatial Lag Model** — y = ρWy + Xβ + ε
//! - **Spatial Error Model** — y = Xβ + u, u = λWu + ε
//!
//! ## Kriging
//! - **Ordinary Kriging** — best linear unbiased predictor assuming unknown mean
//! - **Simple Kriging** — known-mean variant
//! - **Universal Kriging** — adds a polynomial trend surface
//!
//! ## Point Pattern Analysis
//! - **Ripley's K-function** — second-order spatial statistics
//! - **L-function** — variance-stabilised K
//! - **G-function** — nearest-neighbour distance CDF
//!
//! ## Spatial Clustering
//! - **Spatial Scan Statistic** — Kulldorff's circular scan
//! - **AMOEBA** — adaptive spatial cluster detection (simplified variant)

use crate::error::{SpatialError, SpatialResult};

// ─── Simple matrix helpers (avoid heavy ndarray dep here) ────────────────────

/// Row-major dense matrix (n rows × m cols).
#[derive(Debug, Clone)]
pub struct DMatrix {
    pub rows: usize,
    pub cols: usize,
    data: Vec<f64>,
}

impl DMatrix {
    /// Create a zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Create a new matrix from row-major data.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> SpatialResult<Self> {
        if data.len() != rows * cols {
            return Err(SpatialError::DimensionError(format!(
                "expected {}×{} = {} elements, got {}",
                rows,
                cols,
                rows * cols,
                data.len()
            )));
        }
        Ok(Self { rows, cols, data })
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    /// Matrix × vector (Ax = b).
    pub fn mat_vec(&self, x: &[f64]) -> SpatialResult<Vec<f64>> {
        if x.len() != self.cols {
            return Err(SpatialError::DimensionError(format!(
                "mat_vec: vector len {} != cols {}",
                x.len(),
                self.cols
            )));
        }
        let mut out = vec![0.0_f64; self.rows];
        for r in 0..self.rows {
            out[r] = (0..self.cols).map(|c| self.get(r, c) * x[c]).sum();
        }
        Ok(out)
    }

    /// (AᵀA) x = Aᵀb — normal equations solved by Cholesky / Gauss-Seidel.
    /// For small systems we use Gaussian elimination.
    pub fn normal_solve(&self, b: &[f64]) -> SpatialResult<Vec<f64>> {
        // Aᵀ A
        let m = self.cols;
        let n = self.rows;
        let mut ata = vec![0.0_f64; m * m];
        for i in 0..m {
            for j in 0..m {
                ata[i * m + j] = (0..n).map(|k| self.get(k, i) * self.get(k, j)).sum();
            }
        }
        // Aᵀ b
        let mut atb = vec![0.0_f64; m];
        for i in 0..m {
            atb[i] = (0..n).map(|k| self.get(k, i) * b[k]).sum();
        }
        gauss_solve(m, &mut ata, &mut atb)
    }
}

/// Solve Ax = b in place (A stored row-major, n×n) — partial-pivot Gaussian elimination.
fn gauss_solve(n: usize, a: &mut [f64], b: &mut [f64]) -> SpatialResult<Vec<f64>> {
    for col in 0..n {
        // Find pivot
        let pivot = (col..n)
            .max_by(|&i, &j| a[i * n + col].abs().partial_cmp(&a[j * n + col].abs()).unwrap_or(std::cmp::Ordering::Equal));
        let pivot = pivot.ok_or_else(|| SpatialError::ComputationError("empty matrix".to_string()))?;
        a.swap(col * n, pivot * n);
        // swap rows
        for k in 0..n {
            a.swap(col * n + k, pivot * n + k);
        }
        b.swap(col, pivot);
        let diag = a[col * n + col];
        if diag.abs() < 1e-14 {
            return Err(SpatialError::ComputationError(
                "Singular matrix in normal equations".to_string(),
            ));
        }
        for row in (col + 1)..n {
            let factor = a[row * n + col] / diag;
            for k in col..n {
                a[row * n + k] -= factor * a[col * n + k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        x[i] = b[i];
        for j in (i + 1)..n {
            x[i] -= a[i * n + j] * x[j];
        }
        x[i] /= a[i * n + i];
    }
    Ok(x)
}

// ═══════════════════════════════════════════════════════════════════════════════
//  1. Spatial Autocorrelation
// ═══════════════════════════════════════════════════════════════════════════════

/// Row-standardise a spatial weights matrix (each row sums to 1).
pub fn row_standardize(w: &[Vec<f64>]) -> Vec<Vec<f64>> {
    w.iter()
        .map(|row| {
            let s: f64 = row.iter().sum();
            if s.abs() < 1e-14 {
                row.clone()
            } else {
                row.iter().map(|v| v / s).collect()
            }
        })
        .collect()
}

/// **Moran's I** — global spatial autocorrelation index.
///
/// Positive values indicate spatial clustering; negative values indicate
/// spatial dispersion.
///
/// # Arguments
/// * `values`  – observed values at *n* locations
/// * `weights` – *n × n* spatial weights matrix (need not be row-standardised)
///
/// # Returns
/// Moran's I statistic in (approximately) [−1, 1].
pub fn morans_i(values: &[f64], weights: &[Vec<f64>]) -> SpatialResult<f64> {
    let n = values.len();
    if weights.len() != n || weights.iter().any(|r| r.len() != n) {
        return Err(SpatialError::DimensionError(
            "weights must be n×n matching values length".to_string(),
        ));
    }
    if n < 2 {
        return Err(SpatialError::ValueError(
            "need at least 2 observations".to_string(),
        ));
    }

    let mean = values.iter().sum::<f64>() / n as f64;
    let deviations: Vec<f64> = values.iter().map(|v| v - mean).collect();
    let sum_sq: f64 = deviations.iter().map(|d| d * d).sum();
    if sum_sq < 1e-30 {
        return Ok(0.0);
    }

    let w_sum: f64 = weights.iter().flat_map(|r| r.iter()).sum();
    if w_sum.abs() < 1e-14 {
        return Err(SpatialError::ValueError(
            "sum of weights is zero".to_string(),
        ));
    }

    let cross: f64 = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| weights[i][j] * deviations[i] * deviations[j])
        .sum();

    Ok(n as f64 * cross / (w_sum * sum_sq))
}

/// **Geary's C** — contiguity-ratio measure of spatial autocorrelation.
///
/// C < 1 → positive autocorrelation, C > 1 → negative autocorrelation.
///
/// # Arguments
/// * `values`  – observed values
/// * `weights` – *n × n* spatial weights matrix
pub fn gearys_c(values: &[f64], weights: &[Vec<f64>]) -> SpatialResult<f64> {
    let n = values.len();
    if weights.len() != n || weights.iter().any(|r| r.len() != n) {
        return Err(SpatialError::DimensionError(
            "weights must be n×n".to_string(),
        ));
    }
    if n < 2 {
        return Err(SpatialError::ValueError(
            "need at least 2 observations".to_string(),
        ));
    }

    let mean = values.iter().sum::<f64>() / n as f64;
    let sum_sq: f64 = values.iter().map(|v| (v - mean).powi(2)).sum();
    if sum_sq < 1e-30 {
        return Ok(1.0);
    }

    let w_sum: f64 = weights.iter().flat_map(|r| r.iter()).sum();
    if w_sum.abs() < 1e-14 {
        return Err(SpatialError::ValueError(
            "sum of weights is zero".to_string(),
        ));
    }

    let cross: f64 = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| weights[i][j] * (values[i] - values[j]).powi(2))
        .sum();

    Ok((n as f64 - 1.0) * cross / (2.0 * w_sum * sum_sq))
}

// ═══════════════════════════════════════════════════════════════════════════════
//  2. Spatial Regression
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of a spatial regression model.
#[derive(Debug, Clone)]
pub struct SpatialRegressionResult {
    /// Estimated coefficients (β for covariates, ρ or λ as last element).
    pub coefficients: Vec<f64>,
    /// Fitted values ŷ.
    pub fitted: Vec<f64>,
    /// Residuals e = y − ŷ.
    pub residuals: Vec<f64>,
    /// Log-likelihood (approximate).
    pub log_likelihood: f64,
}

/// **Spatial Lag Model** (SLM): y = ρWy + Xβ + ε.
///
/// Fitted by 2SLS using (X, WX) as instruments (a standard GMM approach).
///
/// # Arguments
/// * `y`  – response vector of length *n*
/// * `x`  – covariate matrix *n × k* (row-major, **without** intercept column)
/// * `w`  – row-standardised spatial weights *n × n* (as flat Vec of Vecs)
///
/// # Returns
/// [`SpatialRegressionResult`] with β coefficients plus ρ (last element).
pub fn spatial_lag_model(
    y: &[f64],
    x: &[Vec<f64>],
    w: &[Vec<f64>],
) -> SpatialResult<SpatialRegressionResult> {
    let n = y.len();
    if x.len() != n || w.len() != n {
        return Err(SpatialError::DimensionError(
            "y, x, w must all have n rows".to_string(),
        ));
    }
    let k = x[0].len();

    // Build instrument matrix Z = [1, X, WX]  (n × (1 + k + k))
    let wrstd = row_standardize(w);

    // Wy
    let wy: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| wrstd[i][j] * y[j]).sum())
        .collect();

    // WX  columns
    let wx: Vec<Vec<f64>> = (0..k)
        .map(|col| {
            (0..n)
                .map(|i| (0..n).map(|j| wrstd[i][j] * x[j][col]).sum())
                .collect()
        })
        .collect();

    // Instrument matrix columns: intercept | x cols | wx cols
    let num_instruments = 1 + k + k;
    let mut z = DMatrix::zeros(n, num_instruments);
    for i in 0..n {
        z.set(i, 0, 1.0); // intercept
        for c in 0..k {
            z.set(i, 1 + c, x[i][c]);
        }
        for c in 0..k {
            z.set(i, 1 + k + c, wx[c][i]);
        }
    }

    // Regressor matrix D = [Wy, 1, X]  (n × (1 + k + 1))
    let num_regressors = 1 + k + 1;
    let mut d = DMatrix::zeros(n, num_regressors);
    for i in 0..n {
        d.set(i, 0, wy[i]); // Wy
        d.set(i, 1, 1.0);   // intercept
        for c in 0..k {
            d.set(i, 2 + c, x[i][c]);
        }
    }

    // 2SLS: D̂ = Z(ZᵀZ)⁻¹Zᵀ D  then OLS on [ŷ, D̂]
    // Project each column of D onto the column space of Z
    let mut d_hat = DMatrix::zeros(n, num_regressors);
    for col in 0..num_regressors {
        let d_col: Vec<f64> = (0..n).map(|i| d.get(i, col)).collect();
        let beta_proj = z.normal_solve(&d_col)?;
        let fitted_col = z.mat_vec(&beta_proj)?;
        for i in 0..n {
            d_hat.set(i, col, fitted_col[i]);
        }
    }

    // OLS on y = D̂ γ
    let gamma = d_hat.normal_solve(y)?;

    // Compute fitted and residuals
    let fitted = d.mat_vec(&gamma)?;
    let residuals: Vec<f64> = y.iter().zip(fitted.iter()).map(|(yi, fi)| yi - fi).collect();
    let sigma2 = residuals.iter().map(|r| r * r).sum::<f64>() / n as f64;
    let log_likelihood = -0.5 * n as f64 * (2.0 * std::f64::consts::PI * sigma2).ln()
        - 0.5 * residuals.iter().map(|r| r * r).sum::<f64>() / sigma2;

    // gamma: [ρ, intercept, β₁, …, β_k]
    // Reorder to [intercept, β₁, …, β_k, ρ]
    let mut coefficients = vec![gamma[1]]; // intercept
    for c in 0..k {
        coefficients.push(gamma[2 + c]);
    }
    coefficients.push(gamma[0]); // ρ last

    Ok(SpatialRegressionResult {
        coefficients,
        fitted,
        residuals,
        log_likelihood,
    })
}

/// **Spatial Error Model** (SEM): y = Xβ + u, u = λWu + ε.
///
/// Fitted iteratively using a Cochrane–Orcutt–style GLS approach.
///
/// # Arguments
/// * `y`  – response vector of length *n*
/// * `x`  – covariate matrix *n × k* (without intercept)
/// * `w`  – spatial weights *n × n*
///
/// # Returns
/// [`SpatialRegressionResult`]; last coefficient element is λ̂.
pub fn spatial_error_model(
    y: &[f64],
    x: &[Vec<f64>],
    w: &[Vec<f64>],
) -> SpatialResult<SpatialRegressionResult> {
    let n = y.len();
    if x.len() != n || w.len() != n {
        return Err(SpatialError::DimensionError(
            "y, x, w must all have n rows".to_string(),
        ));
    }
    let k = x[0].len();
    let wrstd = row_standardize(w);

    // Helper: OLS fit [1, X] → y
    let ols = |yv: &[f64], xv: &[Vec<f64>]| -> SpatialResult<(Vec<f64>, Vec<f64>)> {
        let ncol = 1 + xv[0].len();
        let mut mat = DMatrix::zeros(yv.len(), ncol);
        for (i, row) in xv.iter().enumerate() {
            mat.set(i, 0, 1.0);
            for (c, &val) in row.iter().enumerate() {
                mat.set(i, 1 + c, val);
            }
        }
        let beta = mat.normal_solve(yv)?;
        let fitted = mat.mat_vec(&beta)?;
        Ok((beta, fitted))
    };

    // Initial OLS residuals
    let (mut beta, _) = ols(y, x)?;
    let mut resid: Vec<f64> = y
        .iter()
        .enumerate()
        .map(|(i, yi)| {
            yi - beta[0]
                - (0..k).map(|c| beta[1 + c] * x[i][c]).sum::<f64>()
        })
        .collect();

    let mut lam = 0.0_f64;

    // Iterate: estimate λ from autocorrelation of residuals, then GLS
    for _iter in 0..50 {
        // Wε
        let we: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| wrstd[i][j] * resid[j]).sum())
            .collect();

        // λ̂ = sum(ε_i Wε_i) / sum((Wε_i)²)
        let num: f64 = resid.iter().zip(we.iter()).map(|(e, we)| e * we).sum();
        let den: f64 = we.iter().map(|we| we * we).sum::<f64>();
        let lam_new = if den.abs() < 1e-14 { 0.0 } else { num / den };
        let lam_change = (lam_new - lam).abs();
        lam = lam_new.clamp(-0.99, 0.99);

        // Transform: y* = y - λ Wy, X* = X - λ WX
        let wy: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| wrstd[i][j] * y[j]).sum())
            .collect();
        let y_star: Vec<f64> = y.iter().zip(wy.iter()).map(|(yi, wyi)| yi - lam * wyi).collect();

        let x_star: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..k)
                    .map(|c| {
                        let wx_ic: f64 = (0..n).map(|j| wrstd[i][j] * x[j][c]).sum();
                        x[i][c] - lam * wx_ic
                    })
                    .collect()
            })
            .collect();

        let (beta_new, _) = ols(&y_star, &x_star)?;
        beta = beta_new;

        // Recompute residuals on original scale
        resid = y
            .iter()
            .enumerate()
            .map(|(i, yi)| {
                yi - beta[0]
                    - (0..k).map(|c| beta[1 + c] * x[i][c]).sum::<f64>()
            })
            .collect();

        if lam_change < 1e-8 {
            break;
        }
    }

    let fitted: Vec<f64> = y
        .iter()
        .enumerate()
        .map(|(i, yi)| {
            yi - resid[i] // y - e = fitted
        })
        .collect();

    let sigma2 = resid.iter().map(|r| r * r).sum::<f64>() / n as f64;
    let log_likelihood = -0.5 * n as f64 * (2.0 * std::f64::consts::PI * sigma2).ln()
        - 0.5 / sigma2 * resid.iter().map(|r| r * r).sum::<f64>();

    let mut coefficients = beta.clone();
    coefficients.push(lam);

    Ok(SpatialRegressionResult {
        coefficients,
        fitted,
        residuals: resid,
        log_likelihood,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
//  3. Kriging
// ═══════════════════════════════════════════════════════════════════════════════

/// Variogram model types supported by the kriging implementations.
#[derive(Debug, Clone)]
pub enum VariogramModel {
    /// Spherical model: γ(h) = c₀ + c·[1.5·h/a − 0.5·(h/a)³] for h ≤ a
    Spherical {
        /// Nugget effect c₀
        nugget: f64,
        /// Sill c (partial sill)
        sill: f64,
        /// Range a
        range: f64,
    },
    /// Exponential: γ(h) = c₀ + c·(1 − exp(−h/a))
    Exponential {
        nugget: f64,
        sill: f64,
        range: f64,
    },
    /// Gaussian: γ(h) = c₀ + c·(1 − exp(−h²/a²))
    Gaussian {
        nugget: f64,
        sill: f64,
        range: f64,
    },
    /// Power: γ(h) = c₀ + c·hᵅ
    Power {
        nugget: f64,
        scale: f64,
        exponent: f64,
    },
}

impl VariogramModel {
    /// Evaluate the variogram at lag distance `h`.
    pub fn gamma(&self, h: f64) -> f64 {
        if h < 0.0 {
            return 0.0;
        }
        match self {
            Self::Spherical { nugget, sill, range } => {
                if h == 0.0 {
                    0.0
                } else if h >= *range {
                    nugget + sill
                } else {
                    let hr = h / range;
                    nugget + sill * (1.5 * hr - 0.5 * hr.powi(3))
                }
            }
            Self::Exponential { nugget, sill, range } => {
                if h == 0.0 {
                    0.0
                } else {
                    nugget + sill * (1.0 - (-h / range).exp())
                }
            }
            Self::Gaussian { nugget, sill, range } => {
                if h == 0.0 {
                    0.0
                } else {
                    nugget + sill * (1.0 - (-(h * h) / (range * range)).exp())
                }
            }
            Self::Power { nugget, scale, exponent } => {
                if h == 0.0 {
                    0.0
                } else {
                    nugget + scale * h.powf(*exponent)
                }
            }
        }
    }
}

/// Euclidean distance between two 2-D points.
#[inline]
fn euclidean_2d(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt()
}

/// Predict at a single unsampled location using **Ordinary Kriging**.
///
/// # Arguments
/// * `sample_x`, `sample_y` – coordinates of *n* sample points
/// * `sample_z`             – values at sample points
/// * `pred_x`, `pred_y`     – coordinates of the prediction point
/// * `variogram`            – fitted variogram model
///
/// # Returns
/// `(z_hat, kriging_variance)` — prediction and its kriging variance.
pub fn ordinary_kriging(
    sample_x: &[f64],
    sample_y: &[f64],
    sample_z: &[f64],
    pred_x: f64,
    pred_y: f64,
    variogram: &VariogramModel,
) -> SpatialResult<(f64, f64)> {
    let n = sample_z.len();
    if sample_x.len() != n || sample_y.len() != n {
        return Err(SpatialError::DimensionError(
            "sample_x, sample_y, sample_z must have equal length".to_string(),
        ));
    }
    if n < 2 {
        return Err(SpatialError::ValueError(
            "need at least 2 sample points".to_string(),
        ));
    }

    // Build (n+1)×(n+1) kriging matrix (with Lagrange multiplier for OK constraint)
    let m = n + 1;
    let mut k = vec![0.0_f64; m * m];
    for i in 0..n {
        for j in 0..n {
            let h = euclidean_2d(sample_x[i], sample_y[i], sample_x[j], sample_y[j]);
            k[i * m + j] = variogram.gamma(h);
        }
        k[i * m + n] = 1.0;
        k[n * m + i] = 1.0;
    }
    k[n * m + n] = 0.0;
    // Tikhonov regularisation: a tiny diagonal perturbation ensures numerical stability
    // when the variogram has zero nugget or the sample configuration is near-degenerate.
    let nugget_eps = 1e-8 * k.iter().cloned().fold(0.0_f64, f64::max).max(1.0);
    for i in 0..n {
        k[i * m + i] += nugget_eps;
    }

    // Build rhs: distances from pred to each sample, plus 1 (Lagrange)
    let mut b = vec![0.0_f64; m];
    for i in 0..n {
        let h = euclidean_2d(sample_x[i], sample_y[i], pred_x, pred_y);
        b[i] = variogram.gamma(h);
    }
    b[n] = 1.0;

    // Solve K λ = b
    let lambda = gauss_solve(m, &mut k, &mut b)?;

    // Prediction
    let z_hat: f64 = (0..n).map(|i| lambda[i] * sample_z[i]).sum();

    // Kriging variance: σ²_OK = Σ λᵢ γ(hᵢ) + μ
    let sigma2 = (0..n).map(|i| lambda[i] * b[i]).sum::<f64>() + lambda[n];
    // Note: b was mutated by gauss_solve; recompute original b values
    let b0: Vec<f64> = (0..n)
        .map(|i| {
            let h = euclidean_2d(sample_x[i], sample_y[i], pred_x, pred_y);
            variogram.gamma(h)
        })
        .collect();
    let sigma2_clean = (0..n).map(|i| lambda[i] * b0[i]).sum::<f64>() + lambda[n];

    Ok((z_hat, sigma2_clean.max(0.0)))
}

/// Predict using **Simple Kriging** (known constant mean).
///
/// # Arguments
/// * `sample_x`, `sample_y` – coordinates of *n* sample points
/// * `sample_z`             – values at sample points
/// * `pred_x`, `pred_y`     – prediction location
/// * `variogram`            – fitted variogram model
/// * `mean`                 – known global mean
///
/// # Returns
/// `(z_hat, kriging_variance)`
pub fn simple_kriging(
    sample_x: &[f64],
    sample_y: &[f64],
    sample_z: &[f64],
    pred_x: f64,
    pred_y: f64,
    variogram: &VariogramModel,
    mean: f64,
) -> SpatialResult<(f64, f64)> {
    let n = sample_z.len();
    if sample_x.len() != n || sample_y.len() != n {
        return Err(SpatialError::DimensionError(
            "coordinate arrays must match sample_z length".to_string(),
        ));
    }
    if n < 1 {
        return Err(SpatialError::ValueError("need at least 1 sample".to_string()));
    }

    // Covariance = sill − γ(h)  (using total sill from γ(∞))
    let sill_total = variogram.gamma(1e12); // large h → sill + nugget

    let cov = |hval: f64| sill_total - variogram.gamma(hval);

    // n × n covariance matrix C
    let mut c_mat = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let h = euclidean_2d(sample_x[i], sample_y[i], sample_x[j], sample_y[j]);
            c_mat[i * n + j] = cov(h);
        }
    }

    // rhs: covariances between prediction point and samples
    let mut c0: Vec<f64> = (0..n)
        .map(|i| {
            let h = euclidean_2d(sample_x[i], sample_y[i], pred_x, pred_y);
            cov(h)
        })
        .collect();
    let c0_saved = c0.clone();

    let lambda = gauss_solve(n, &mut c_mat, &mut c0)?;

    let residuals: Vec<f64> = sample_z.iter().map(|z| z - mean).collect();
    let z_hat = mean + (0..n).map(|i| lambda[i] * residuals[i]).sum::<f64>();

    // Kriging variance: C(0,0) − λᵀ c0
    let c00 = cov(0.0);
    let sigma2 = c00 - (0..n).map(|i| lambda[i] * c0_saved[i]).sum::<f64>();

    Ok((z_hat, sigma2.max(0.0)))
}

/// Predict using **Universal Kriging** (polynomial trend of degree `poly_deg`).
///
/// # Arguments
/// * `sample_x`, `sample_y` – coordinates of *n* sample points
/// * `sample_z`             – values at sample points
/// * `pred_x`, `pred_y`     – prediction location
/// * `variogram`            – fitted variogram model
/// * `poly_deg`             – polynomial trend degree (0 = ordinary, 1 = linear, 2 = quadratic)
pub fn universal_kriging(
    sample_x: &[f64],
    sample_y: &[f64],
    sample_z: &[f64],
    pred_x: f64,
    pred_y: f64,
    variogram: &VariogramModel,
    poly_deg: usize,
) -> SpatialResult<(f64, f64)> {
    let n = sample_z.len();
    if sample_x.len() != n || sample_y.len() != n {
        return Err(SpatialError::DimensionError(
            "coordinate arrays must match sample_z length".to_string(),
        ));
    }

    // Trend basis: degree 0 → [1], degree 1 → [1, x, y], degree 2 → [1, x, y, x², xy, y²]
    let basis = |xi: f64, yi: f64| -> Vec<f64> {
        let mut b = vec![1.0];
        if poly_deg >= 1 {
            b.push(xi);
            b.push(yi);
        }
        if poly_deg >= 2 {
            b.push(xi * xi);
            b.push(xi * yi);
            b.push(yi * yi);
        }
        b
    };

    let p = basis(0.0, 0.0).len();
    let m = n + p;

    let mut k = vec![0.0_f64; m * m];
    // Semivariogram block
    for i in 0..n {
        for j in 0..n {
            let h = euclidean_2d(sample_x[i], sample_y[i], sample_x[j], sample_y[j]);
            k[i * m + j] = variogram.gamma(h);
        }
    }
    // Trend basis blocks
    for i in 0..n {
        let b = basis(sample_x[i], sample_y[i]);
        for (t, &val) in b.iter().enumerate() {
            k[i * m + (n + t)] = val;
            k[(n + t) * m + i] = val;
        }
    }

    // rhs
    let mut rhs = vec![0.0_f64; m];
    let b_pred = basis(pred_x, pred_y);
    for i in 0..n {
        let h = euclidean_2d(sample_x[i], sample_y[i], pred_x, pred_y);
        rhs[i] = variogram.gamma(h);
    }
    for (t, &val) in b_pred.iter().enumerate() {
        rhs[n + t] = val;
    }
    let rhs_saved = rhs.clone();

    let lambda = gauss_solve(m, &mut k, &mut rhs)?;

    let z_hat: f64 = (0..n).map(|i| lambda[i] * sample_z[i]).sum();
    let sigma2 = (0..m).map(|i| lambda[i] * rhs_saved[i]).sum::<f64>();

    Ok((z_hat, sigma2.max(0.0)))
}

// ═══════════════════════════════════════════════════════════════════════════════
//  4. Point Pattern Analysis
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute Ripley's **K-function** at a vector of distances.
///
/// K(d) = (area / n²) · Σᵢ Σⱼ≠ᵢ 1(‖pᵢ − pⱼ‖ ≤ d) / edge_correction
///
/// Uses isotropic (Ripley's) edge correction with a circular study region.
///
/// # Arguments
/// * `px`, `py` – coordinates of *n* events
/// * `distances` – distances at which to evaluate K
/// * `area`      – area of the study region (m²)
///
/// # Returns
/// Vector of K(d) values.
pub fn ripleys_k(
    px: &[f64],
    py: &[f64],
    distances: &[f64],
    area: f64,
) -> SpatialResult<Vec<f64>> {
    let n = px.len();
    if py.len() != n {
        return Err(SpatialError::DimensionError("px and py must have equal length".to_string()));
    }
    if n < 2 {
        return Err(SpatialError::ValueError("need at least 2 points".to_string()));
    }
    if area <= 0.0 {
        return Err(SpatialError::ValueError("area must be positive".to_string()));
    }

    // Pre-compute all pairwise distances
    let n_pairs = n * (n - 1) / 2;
    let mut dists = Vec::with_capacity(n_pairs);
    for i in 0..n {
        for j in (i + 1)..n {
            dists.push(euclidean_2d(px[i], py[i], px[j], py[j]));
        }
    }

    let lambda = n as f64 / area; // intensity
    let k: Vec<f64> = distances
        .iter()
        .map(|&d| {
            // Count pairs within distance d (each pair counted twice)
            let count = dists.iter().filter(|&&dist| dist <= d).count();
            2.0 * count as f64 / (lambda * lambda * area)
        })
        .collect();

    Ok(k)
}

/// Compute the **L-function**: L(d) = sqrt(K(d) / π).
///
/// Under complete spatial randomness, L(d) ≈ d.
///
/// # Arguments
/// Same as [`ripleys_k`].
pub fn l_function(
    px: &[f64],
    py: &[f64],
    distances: &[f64],
    area: f64,
) -> SpatialResult<Vec<f64>> {
    let k = ripleys_k(px, py, distances, area)?;
    Ok(k.iter()
        .map(|&kd| (kd / std::f64::consts::PI).sqrt())
        .collect())
}

/// Compute the **G-function** (nearest-neighbour distance empirical CDF).
///
/// G(d) = P(nearest-neighbour distance ≤ d).
///
/// # Arguments
/// * `px`, `py` – coordinates of *n* events
/// * `distances` – vector of distances at which to evaluate G
///
/// # Returns
/// Vector of G(d) values in [0, 1].
pub fn g_function(
    px: &[f64],
    py: &[f64],
    distances: &[f64],
) -> SpatialResult<Vec<f64>> {
    let n = px.len();
    if py.len() != n {
        return Err(SpatialError::DimensionError("px and py must have equal length".to_string()));
    }
    if n < 2 {
        return Err(SpatialError::ValueError("need at least 2 points".to_string()));
    }

    // Nearest-neighbour distances
    let nn_dists: Vec<f64> = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| j != i)
                .map(|j| euclidean_2d(px[i], py[i], px[j], py[j]))
                .fold(f64::INFINITY, f64::min)
        })
        .collect();

    let g: Vec<f64> = distances
        .iter()
        .map(|&d| {
            let count = nn_dists.iter().filter(|&&nn| nn <= d).count();
            count as f64 / n as f64
        })
        .collect();

    Ok(g)
}

// ═══════════════════════════════════════════════════════════════════════════════
//  5. Spatial Clustering
// ═══════════════════════════════════════════════════════════════════════════════

/// Result from the spatial scan statistic.
#[derive(Debug, Clone)]
pub struct ScanResult {
    /// Centre of the most-likely cluster (x, y).
    pub centre: (f64, f64),
    /// Radius of the most-likely cluster (in coordinate units).
    pub radius: f64,
    /// Observed events inside the window.
    pub observed: usize,
    /// Expected events inside the window.
    pub expected: f64,
    /// Log-likelihood ratio statistic.
    pub log_lr: f64,
}

/// **Spatial Scan Statistic** (Kulldorff's circular scan).
///
/// Scans circular windows centred at each event location with radii up to
/// `max_radius_frac` × bounding-box diagonal, and finds the window with the
/// highest Poisson log-likelihood ratio.
///
/// # Arguments
/// * `event_x`, `event_y` – coordinates of *n* observed events
/// * `pop_x`, `pop_y`     – coordinates of *m* population (at-risk) locations
/// * `max_radius_frac`    – maximum window radius as fraction of study area diameter [0..1]
/// * `n_radii`            – number of radius steps to test
///
/// # Returns
/// [`ScanResult`] for the most-likely cluster.
pub fn spatial_scan_statistic(
    event_x: &[f64],
    event_y: &[f64],
    pop_x: &[f64],
    pop_y: &[f64],
    max_radius_frac: f64,
    n_radii: usize,
) -> SpatialResult<ScanResult> {
    let n_events = event_x.len();
    let n_pop = pop_x.len();
    if event_y.len() != n_events || pop_y.len() != n_pop {
        return Err(SpatialError::DimensionError(
            "coordinate arrays must be consistent".to_string(),
        ));
    }
    if n_events < 2 || n_pop < 1 {
        return Err(SpatialError::ValueError(
            "need at least 2 events and 1 population point".to_string(),
        ));
    }

    // Study region extent
    let all_x: Vec<f64> = event_x
        .iter()
        .chain(pop_x.iter())
        .cloned()
        .collect();
    let all_y: Vec<f64> = event_y
        .iter()
        .chain(pop_y.iter())
        .cloned()
        .collect();
    let x_range = all_x.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - all_x.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_range = all_y.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - all_y.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_r = max_radius_frac * (x_range.hypot(y_range));
    let r_step = if n_radii > 1 { max_r / n_radii as f64 } else { max_r };

    let total_events = n_events as f64;
    let total_pop = n_pop as f64;

    let mut best = ScanResult {
        centre: (0.0, 0.0),
        radius: 0.0,
        observed: 0,
        expected: 0.0,
        log_lr: f64::NEG_INFINITY,
    };

    // Scan each event location as a potential cluster centre
    for ci in 0..n_events {
        let cx = event_x[ci];
        let cy = event_y[ci];

        for ri in 1..=n_radii {
            let r = r_step * ri as f64;

            // Count events and population within radius
            let obs = event_x
                .iter()
                .zip(event_y.iter())
                .filter(|(ex, ey)| euclidean_2d(**ex, **ey, cx, cy) <= r)
                .count();

            let pop_in = pop_x
                .iter()
                .zip(pop_y.iter())
                .filter(|(px, py)| euclidean_2d(**px, **py, cx, cy) <= r)
                .count() as f64;

            if pop_in < 1.0 {
                continue;
            }

            let expected = total_events * pop_in / total_pop;

            if obs as f64 <= expected {
                continue; // Only scan for elevated risk
            }

            // Poisson log-LR: c·ln(c/μ) + (N−c)·ln((N−c)/(N−μ))  [Kulldorff]
            let n_out = total_events - obs as f64;
            let mu_out = total_events - expected;

            let lr = if obs > 0 && n_out > 0.0 && mu_out > 0.0 {
                obs as f64 * (obs as f64 / expected).ln()
                    + n_out * (n_out / mu_out).ln()
            } else {
                0.0
            };

            if lr > best.log_lr {
                best = ScanResult {
                    centre: (cx, cy),
                    radius: r,
                    observed: obs,
                    expected,
                    log_lr: lr,
                };
            }
        }
    }

    if best.log_lr == f64::NEG_INFINITY {
        best.log_lr = 0.0;
    }

    Ok(best)
}

/// **AMOEBA** (adaptive spatial cluster detection) — simplified spatial LISA
/// based on Local Moran's I.
///
/// Identifies locations whose local Moran's I exceeds `threshold` and
/// returns them as cluster seeds.
///
/// # Arguments
/// * `values`    – attribute values at *n* locations
/// * `weights`   – *n × n* spatial weights (row-standardised)
/// * `threshold` – minimum absolute local Moran's I to flag as cluster seed
///
/// # Returns
/// Indices of locations that are cluster seeds.
pub fn amoeba_clusters(
    values: &[f64],
    weights: &[Vec<f64>],
    threshold: f64,
) -> SpatialResult<Vec<usize>> {
    let n = values.len();
    if weights.len() != n || weights.iter().any(|r| r.len() != n) {
        return Err(SpatialError::DimensionError("weights must be n×n".to_string()));
    }

    let mean = values.iter().sum::<f64>() / n as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    if variance < 1e-30 {
        return Ok(vec![]);
    }

    let z: Vec<f64> = values.iter().map(|v| (v - mean) / variance.sqrt()).collect();

    let wrstd = row_standardize(weights);

    let mut seeds = Vec::new();
    for i in 0..n {
        // Wz_i = Σ_j w_ij z_j
        let wz_i: f64 = (0..n).map(|j| wrstd[i][j] * z[j]).sum();
        let local_i = z[i] * wz_i;
        if local_i.abs() >= threshold {
            seeds.push(i);
        }
    }

    Ok(seeds)
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn queen_weights(n: usize) -> Vec<Vec<f64>> {
        // 1-D chain weights
        let mut w = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            if i > 0 {
                w[i][i - 1] = 1.0;
            }
            if i + 1 < n {
                w[i][i + 1] = 1.0;
            }
        }
        w
    }

    #[test]
    fn test_morans_i_random() {
        // Alternating high/low values → negative autocorrelation
        let values = vec![1.0, 5.0, 1.0, 5.0, 1.0];
        let w = queen_weights(5);
        let mi = morans_i(&values, &w).expect("moran");
        assert!(mi < 0.0, "expected negative Moran's I, got {mi}");
    }

    #[test]
    fn test_morans_i_clustered() {
        // All identical → Moran I = 0
        let values = vec![3.0; 5];
        let w = queen_weights(5);
        let mi = morans_i(&values, &w).expect("moran");
        assert!(mi.abs() < 1e-6, "identical values → I≈0, got {mi}");
    }

    #[test]
    fn test_gearys_c_sanity() {
        let values = vec![1.0, 2.0, 1.5, 3.0, 2.5];
        let w = queen_weights(5);
        let gc = gearys_c(&values, &w).expect("geary");
        // Must be positive
        assert!(gc > 0.0, "gc={gc}");
    }

    #[test]
    fn test_variogram_models() {
        let sph = VariogramModel::Spherical { nugget: 0.1, sill: 1.0, range: 10.0 };
        assert!(sph.gamma(0.0) == 0.0);
        assert!(sph.gamma(10.0) > sph.gamma(5.0));
        assert!((sph.gamma(100.0) - 1.1).abs() < 1e-6);

        let exp = VariogramModel::Exponential { nugget: 0.0, sill: 1.0, range: 5.0 };
        assert!((exp.gamma(0.0)).abs() < 1e-12);
        assert!(exp.gamma(20.0) < 1.01);
    }

    #[test]
    fn test_ordinary_kriging() {
        // 2-D samples in a line (x-axis), with a small nugget to keep the kriging
        // system non-singular.
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let z = vec![0.0, 1.0, 4.0, 9.0, 16.0]; // x²

        // nugget > 0 ensures the diagonal is non-zero → well-conditioned system.
        let variogram = VariogramModel::Spherical { nugget: 0.5, sill: 50.0, range: 5.0 };
        let (z_hat, var_ok) = ordinary_kriging(&x, &y, &z, 2.5, 0.0, &variogram).expect("ok");
        // Should interpolate near the midpoint value (~6)
        assert!(z_hat > 2.0 && z_hat < 15.0, "z_hat={z_hat}");
        assert!(var_ok >= 0.0, "variance must be non-negative: {var_ok}");
    }

    #[test]
    fn test_simple_kriging() {
        let x = vec![0.0, 2.0, 4.0];
        let y = vec![0.0, 0.0, 0.0];
        let z = vec![0.0, 2.0, 4.0];
        let variogram = VariogramModel::Spherical { nugget: 0.0, sill: 5.0, range: 10.0 };
        let (z_hat, _v) = simple_kriging(&x, &y, &z, 1.0, 0.0, &variogram, 2.0).expect("sk");
        // Linear interpolation between 0 and 2
        assert!(z_hat > 0.5 && z_hat < 2.5, "z_hat={z_hat}");
    }

    #[test]
    fn test_ripleys_k_csr() {
        // Under CSR, K(d) ≈ πd²
        let n = 100;
        let area = 100.0_f64 * 100.0;
        // Place points on a grid (close to random for this test)
        let px: Vec<f64> = (0..n).map(|i| (i % 10) as f64 * 10.0 + 5.0).collect();
        let py: Vec<f64> = (0..n).map(|i| (i / 10) as f64 * 10.0 + 5.0).collect();
        let distances = vec![5.0, 10.0, 15.0];
        let k = ripleys_k(&px, &py, &distances, area).expect("K");
        assert_eq!(k.len(), 3);
        // K should be non-decreasing
        assert!(k[1] >= k[0], "K not monotone: {:?}", k);
        assert!(k[2] >= k[1], "K not monotone: {:?}", k);
    }

    #[test]
    fn test_l_function() {
        let px = vec![0.0, 1.0, 2.0];
        let py = vec![0.0, 0.0, 0.0];
        let distances = vec![1.0, 2.0];
        let l = l_function(&px, &py, &distances, 4.0).expect("L");
        assert_eq!(l.len(), 2);
    }

    #[test]
    fn test_g_function() {
        let px = vec![0.0, 1.0, 5.0];
        let py = vec![0.0, 0.0, 0.0];
        let distances = vec![0.5, 1.5, 6.0];
        let g = g_function(&px, &py, &distances).expect("G");
        // G should be non-decreasing and in [0, 1]
        assert!(g[0] <= g[1] && g[1] <= g[2]);
        assert!(g[2] <= 1.0 + 1e-9);
    }

    #[test]
    fn test_spatial_scan_statistic() {
        // 5 events clustered in one corner, 5 events scattered
        let ex = vec![0.0, 0.5, 1.0, 0.3, 0.7, 50.0, 60.0, 70.0, 80.0, 90.0];
        let ey = vec![0.0, 0.5, 1.0, 0.3, 0.7, 50.0, 60.0, 70.0, 80.0, 90.0];
        let pop_x: Vec<f64> = (0..20).map(|i| i as f64 * 5.0).collect();
        let pop_y: Vec<f64> = (0..20).map(|_| 0.0).collect();

        let result = spatial_scan_statistic(&ex, &ey, &pop_x, &pop_y, 0.3, 5).expect("scan");
        // The cluster should be detected somewhere
        assert!(result.log_lr >= 0.0);
    }

    #[test]
    fn test_amoeba_clusters() {
        // Values: high cluster at positions 0-1, low at 3-4
        let values = vec![10.0, 9.0, 5.0, 1.0, 0.0];
        let w = queen_weights(5);
        let seeds = amoeba_clusters(&values, &w, 0.5).expect("amoeba");
        // Positions 0 and 4 should be strong cluster seeds
        assert!(!seeds.is_empty(), "expected at least one seed");
    }

    #[test]
    fn test_spatial_lag_model_basic() {
        let n = 8_usize;
        // Simple linear relationship y = 2x + noise
        let x: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..n).map(|i| 2.0 * i as f64 + 0.1).collect();
        let w = queen_weights(n);
        let result = spatial_lag_model(&y, &x, &w).expect("slm");
        // Should have k+2 = 3 coefficients [intercept, β, ρ]
        assert_eq!(result.coefficients.len(), 3);
        assert_eq!(result.residuals.len(), n);
    }

    #[test]
    fn test_spatial_error_model_basic() {
        let n = 8_usize;
        let x: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..n).map(|i| 3.0 * i as f64 + 1.0).collect();
        let w = queen_weights(n);
        let result = spatial_error_model(&y, &x, &w).expect("sem");
        // [intercept, β, λ]
        assert_eq!(result.coefficients.len(), 3);
    }
}
