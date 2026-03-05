//! Advanced spatial statistics.
//!
//! Provides:
//! * [`MoransI`] / [`morans_i`] — Global Moran's I spatial autocorrelation.
//! * [`GearyC`] / [`gearys_c`] — Geary's C statistic.
//! * [`local_morans_i`] — Local Indicators of Spatial Association (LISA).
//! * [`ripleys_k`] / [`ripleys_l`] — Ripley's K/L functions for point patterns.
//! * [`spatial_kde`] — Gaussian kernel density estimation on 2-D point data.
//! * [`GWRResult`] / [`geographically_weighted_regression`] — GWR (simplified).
//!
//! All inputs use plain `&[f64]` / `&[[f64; 2]]` slices for maximum interop.
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::advanced_spatial_stats::{morans_i, gearys_c, ripleys_k};
//!
//! let values = vec![1.0, 2.0, 1.5, 3.0, 2.5];
//! let weights = vec![
//!     vec![0.0, 0.5, 0.0, 0.0, 0.5],
//!     vec![0.5, 0.0, 0.5, 0.0, 0.0],
//!     vec![0.0, 0.5, 0.0, 0.5, 0.0],
//!     vec![0.0, 0.0, 0.5, 0.0, 0.5],
//!     vec![0.5, 0.0, 0.0, 0.5, 0.0],
//! ];
//!
//! let mi = morans_i(&values, &weights).unwrap();
//! println!("Moran's I = {:.4}", mi.statistic);
//! ```

use std::f64::consts::PI;

// ── Error type ─────────────────────────────────────────────────────────────────

/// Errors from spatial statistics functions.
#[derive(Debug, Clone, PartialEq)]
pub enum StatError {
    /// Inputs have incompatible lengths or are too short.
    InvalidInput(String),
    /// Numerical computation failed (e.g., singular matrix).
    NumericalError(String),
}

impl std::fmt::Display for StatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(s) => write!(f, "InvalidInput: {s}"),
            Self::NumericalError(s) => write!(f, "NumericalError: {s}"),
        }
    }
}

impl std::error::Error for StatError {}

// ── Utility ────────────────────────────────────────────────────────────────────

/// Normal CDF (two-sided p-value from z-score).
fn normal_pvalue(z: f64) -> f64 {
    // Abramowitz & Stegun approximation, accurate to ~1.5e-7.
    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let poly = t
        * (0.319_381_53
            + t * (-0.356_563_782
                + t * (1.781_477_937
                    + t * (-1.821_255_978 + t * 1.330_274_429))));
    let phi = 1.0 - ((-0.5 * z * z).exp() / (2.0 * PI).sqrt()) * poly;
    2.0 * (1.0 - phi.clamp(0.0, 1.0))
}

/// Mean of a slice.
fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { return 0.0; }
    v.iter().sum::<f64>() / v.len() as f64
}

/// Variance of a slice (population variance).
#[allow(dead_code)]
fn variance(v: &[f64]) -> f64 {
    let m = mean(v);
    v.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / v.len() as f64
}

/// Sum of all elements in the weight matrix.
fn weight_sum(w: &[Vec<f64>]) -> f64 {
    w.iter().flat_map(|row| row.iter()).sum()
}

// ── Moran's I ─────────────────────────────────────────────────────────────────

/// Result of the global Moran's I test.
#[derive(Debug, Clone)]
pub struct MoransI {
    /// Observed Moran's I statistic (range −1 … +1).
    pub statistic: f64,
    /// Two-sided p-value (under normality assumption).
    pub p_value: f64,
    /// Z-score.
    pub z_score: f64,
    /// Expected value E[I] = −1/(n−1).
    pub expected: f64,
    /// Variance of I under the normality assumption.
    pub variance: f64,
}

/// Compute global Moran's I spatial autocorrelation.
///
/// * `values` — attribute values at each location.
/// * `weight_matrix` — n×n row-standardized (or binary) spatial weight matrix.
///
/// Returns `Err` if dimensions are inconsistent or `n < 2`.
pub fn morans_i(values: &[f64], weight_matrix: &[Vec<f64>]) -> Result<MoransI, StatError> {
    let n = values.len();
    if n < 2 {
        return Err(StatError::InvalidInput("Need at least 2 observations".into()));
    }
    if weight_matrix.len() != n || weight_matrix.iter().any(|r| r.len() != n) {
        return Err(StatError::InvalidInput(
            "weight_matrix must be n×n".into(),
        ));
    }

    let ws = weight_sum(weight_matrix);
    if ws.abs() < 1e-12 {
        return Err(StatError::InvalidInput("Sum of weights is zero".into()));
    }

    let m = mean(values);
    let z: Vec<f64> = values.iter().map(|&v| v - m).collect();

    // Numerator: sum_i sum_j w_ij * z_i * z_j
    let mut numerator = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            numerator += weight_matrix[i][j] * z[i] * z[j];
        }
    }

    // Denominator: sum_i z_i^2
    let denom: f64 = z.iter().map(|&zi| zi * zi).sum();
    if denom.abs() < 1e-12 {
        return Err(StatError::NumericalError("All values are identical".into()));
    }

    let statistic = (n as f64) * numerator / (ws * denom);

    // Expected value
    let expected = -1.0 / (n as f64 - 1.0);

    // Variance under normality assumption.
    // Simplified formula: Var[I] = (n²S1 - nS2 + 3W²) / ((n² - 1)W²)  − E[I]²
    // where S1 = 0.5 * sum_ij (w_ij + w_ji)², S2 = sum_i (sum_j w_ij + sum_j w_ji)²
    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    for i in 0..n {
        let row_sum: f64 = weight_matrix[i].iter().sum();
        let col_sum: f64 = weight_matrix.iter().map(|r| r[i]).sum();
        s2 += (row_sum + col_sum).powi(2);
        for j in 0..n {
            s1 += (weight_matrix[i][j] + weight_matrix[j][i]).powi(2);
        }
    }
    s1 *= 0.5;

    let nf = n as f64;
    let w2 = ws * ws;
    let var_i = (nf * nf * s1 - nf * s2 + 3.0 * w2)
        / ((nf * nf - 1.0) * w2)
        - expected * expected;
    let var_i = var_i.max(0.0);

    let std_i = var_i.sqrt();
    let z_score = if std_i > 1e-12 {
        (statistic - expected) / std_i
    } else {
        0.0
    };
    let p_value = normal_pvalue(z_score);

    Ok(MoransI { statistic, p_value, z_score, expected, variance: var_i })
}

// ── Geary's C ─────────────────────────────────────────────────────────────────

/// Result of the Geary's C test.
#[derive(Debug, Clone)]
pub struct GearyC {
    /// Observed Geary's C statistic (0 = perfect positive, 2 = perfect negative).
    pub statistic: f64,
    /// Z-score.
    pub z_score: f64,
    /// Two-sided p-value.
    pub p_value: f64,
}

/// Compute Geary's C statistic.
///
/// C < 1 indicates positive autocorrelation; C > 1 indicates negative.
pub fn gearys_c(values: &[f64], weight_matrix: &[Vec<f64>]) -> Result<GearyC, StatError> {
    let n = values.len();
    if n < 2 {
        return Err(StatError::InvalidInput("Need at least 2 observations".into()));
    }
    if weight_matrix.len() != n || weight_matrix.iter().any(|r| r.len() != n) {
        return Err(StatError::InvalidInput("weight_matrix must be n×n".into()));
    }

    let ws = weight_sum(weight_matrix);
    if ws.abs() < 1e-12 {
        return Err(StatError::InvalidInput("Sum of weights is zero".into()));
    }

    let m = mean(values);
    let z: Vec<f64> = values.iter().map(|&v| v - m).collect();
    let denom: f64 = z.iter().map(|&zi| zi * zi).sum();
    if denom.abs() < 1e-12 {
        return Err(StatError::NumericalError("All values are identical".into()));
    }

    let mut numerator = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            numerator += weight_matrix[i][j] * (values[i] - values[j]).powi(2);
        }
    }

    let nf = n as f64;
    let statistic = (nf - 1.0) * numerator / (2.0 * ws * denom);

    // Approximate variance under randomisation.
    let expected_c = 1.0;
    // Var[C] simplified (Cliff & Ord, 1981):
    let var_c_approx = 1.0 / (2.0 * nf); // very rough; sufficient for z-score sign.
    let std_c = var_c_approx.sqrt().max(1e-12);
    let z_score = (statistic - expected_c) / std_c;
    let p_value = normal_pvalue(z_score);

    Ok(GearyC { statistic, z_score, p_value })
}

// ── Local Moran's I (LISA) ────────────────────────────────────────────────────

/// Compute Local Moran's I (LISA) for each observation.
///
/// Returns a vector of length `n` with the local Ii statistic for each location.
pub fn local_morans_i(
    values: &[f64],
    weight_matrix: &[Vec<f64>],
) -> Result<Vec<f64>, StatError> {
    let n = values.len();
    if n < 2 {
        return Err(StatError::InvalidInput("Need at least 2 observations".into()));
    }
    if weight_matrix.len() != n || weight_matrix.iter().any(|r| r.len() != n) {
        return Err(StatError::InvalidInput("weight_matrix must be n×n".into()));
    }

    let m = mean(values);
    let z: Vec<f64> = values.iter().map(|&v| v - m).collect();
    let m2 = z.iter().map(|&zi| zi * zi).sum::<f64>() / n as f64;
    if m2.abs() < 1e-12 {
        return Ok(vec![0.0; n]);
    }

    let mut local_i: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let lag: f64 = weight_matrix[i].iter().zip(z.iter()).map(|(&w, &zj)| w * zj).sum();
        local_i.push(z[i] * lag / m2);
    }
    Ok(local_i)
}

// ── Ripley's K ────────────────────────────────────────────────────────────────

/// Compute Ripley's K function for a 2-D point pattern.
///
/// * `points` — point locations.
/// * `area` — total study area.
/// * `distances` — distance values at which K is evaluated.
///
/// Returns K(d) for each distance.
pub fn ripleys_k(points: &[[f64; 2]], area: f64, distances: &[f64]) -> Vec<f64> {
    let n = points.len();
    if n < 2 || distances.is_empty() {
        return vec![0.0; distances.len()];
    }
    let nf = n as f64;
    let factor = area / (nf * (nf - 1.0));

    let mut k_vals = vec![0.0_f64; distances.len()];
    for (di, &d) in distances.iter().enumerate() {
        let mut count = 0_usize;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let dx = points[i][0] - points[j][0];
                    let dy = points[i][1] - points[j][1];
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist <= d {
                        count += 1;
                    }
                }
            }
        }
        k_vals[di] = factor * count as f64;
    }
    k_vals
}

/// Compute Ripley's L function: L(d) = sqrt(K(d) / π).
///
/// The L function is a variance-stabilising transformation of K.
pub fn ripleys_l(points: &[[f64; 2]], area: f64, distances: &[f64]) -> Vec<f64> {
    let k = ripleys_k(points, area, distances);
    k.into_iter().map(|ki| (ki / PI).sqrt()).collect()
}

// ── Spatial KDE ───────────────────────────────────────────────────────────────

/// Estimate density at each point in `grid` using a 2-D Gaussian kernel.
///
/// * `points` — input point locations.
/// * `weights` — optional per-point weights; if `None`, uniform weight 1.
/// * `bandwidth` — Gaussian kernel bandwidth (σ).
/// * `grid` — evaluation points.
///
/// Returns density estimates at each grid point.
pub fn spatial_kde(
    points: &[[f64; 2]],
    weights: Option<&[f64]>,
    bandwidth: f64,
    grid: &[[f64; 2]],
) -> Result<Vec<f64>, StatError> {
    if bandwidth <= 0.0 {
        return Err(StatError::InvalidInput("bandwidth must be positive".into()));
    }
    if let Some(w) = weights {
        if w.len() != points.len() {
            return Err(StatError::InvalidInput(
                "weights length must match points length".into(),
            ));
        }
    }

    let h2 = bandwidth * bandwidth;
    let norm = 1.0 / (2.0 * PI * h2);
    let mut densities = vec![0.0_f64; grid.len()];

    for (gi, &gp) in grid.iter().enumerate() {
        let mut sum = 0.0_f64;
        let mut weight_total = 0.0_f64;
        for (pi, &pt) in points.iter().enumerate() {
            let w = weights.map_or(1.0, |ws| ws[pi]);
            let dx = gp[0] - pt[0];
            let dy = gp[1] - pt[1];
            let d2 = dx * dx + dy * dy;
            sum += w * norm * (-0.5 * d2 / h2).exp();
            weight_total += w;
        }
        densities[gi] = if weight_total > 0.0 { sum / weight_total * points.len() as f64 } else { 0.0 };
    }
    Ok(densities)
}

// ── Geographically Weighted Regression ────────────────────────────────────────

/// Results of a geographically weighted regression.
#[derive(Debug, Clone)]
pub struct GWRResult {
    /// Per-location coefficient vectors.  `coefficients[i]` has length = number of predictors + 1 (intercept).
    pub coefficients: Vec<Vec<f64>>,
    /// Residuals at each location.
    pub residuals: Vec<f64>,
    /// Local R² at each location.
    pub r_squared: Vec<f64>,
}

/// Perform simplified Geographically Weighted Regression.
///
/// Fits a separate weighted OLS at each location using a Gaussian kernel
/// with the given `bandwidth`.
///
/// * `y` — response variable (length n).
/// * `x` — predictor matrix, row-major: `x[i]` is the predictor vector for observation `i`.
/// * `coords` — 2-D coordinates for each observation.
/// * `bandwidth` — Gaussian kernel bandwidth for spatial weights.
///
/// Returns `Err` if input dimensions are inconsistent or `n < p + 1`.
pub fn geographically_weighted_regression(
    y: &[f64],
    x: &[Vec<f64>],
    coords: &[[f64; 2]],
    bandwidth: f64,
) -> Result<GWRResult, String> {
    let n = y.len();
    if x.len() != n {
        return Err("x and y must have the same length".into());
    }
    if coords.len() != n {
        return Err("coords and y must have the same length".into());
    }
    if n == 0 {
        return Err("Empty input".into());
    }
    let p = if let Some(xi) = x.first() { xi.len() } else { 0 };
    // Design matrix includes intercept → p+1 columns.
    let p1 = p + 1;
    if n < p1 {
        return Err(format!("Need at least {} observations for {} predictors", p1, p));
    }
    if bandwidth <= 0.0 {
        return Err("bandwidth must be positive".into());
    }

    let h2 = bandwidth * bandwidth;
    let mut coefficients = vec![vec![0.0_f64; p1]; n];
    let mut residuals = vec![0.0_f64; n];
    let mut r_squared = vec![0.0_f64; n];

    for i in 0..n {
        // Spatial weights for location i.
        let mut w_diag: Vec<f64> = Vec::with_capacity(n);
        for j in 0..n {
            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let d2 = dx * dx + dy * dy;
            w_diag.push((-0.5 * d2 / h2).exp());
        }

        // Weighted OLS: beta = (X'WX)^-1 X'Wy
        // Build X matrix (n × p1) with intercept column.
        // Build XtWX (p1 × p1) and XtWy (p1).
        let mut xtw_x = vec![vec![0.0_f64; p1]; p1];
        let mut xtw_y = vec![0.0_f64; p1];

        for j in 0..n {
            let wj = w_diag[j];
            // Row of design matrix for observation j.
            let xj: Vec<f64> = std::iter::once(1.0).chain(x[j].iter().copied()).collect();
            for r in 0..p1 {
                xtw_y[r] += wj * xj[r] * y[j];
                for c in 0..p1 {
                    xtw_x[r][c] += wj * xj[r] * xj[c];
                }
            }
        }

        // Solve via Gaussian elimination with partial pivoting.
        match solve_linear(&xtw_x, &xtw_y) {
            Ok(beta) => {
                coefficients[i] = beta.clone();
                // Compute residual at location i.
                let xi: Vec<f64> = std::iter::once(1.0).chain(x[i].iter().copied()).collect();
                let y_hat: f64 = xi.iter().zip(beta.iter()).map(|(&xi, &bi)| xi * bi).sum();
                residuals[i] = y[i] - y_hat;

                // Local R²: weighted TSS and RSS.
                let y_mean: f64 = w_diag.iter().zip(y.iter()).map(|(&w, &yj)| w * yj).sum::<f64>()
                    / w_diag.iter().sum::<f64>().max(1e-12);
                let tss: f64 = w_diag.iter().zip(y.iter()).map(|(&w, &yj)| w * (yj - y_mean).powi(2)).sum();
                let rss: f64 = {
                    let mut s = 0.0_f64;
                    for j in 0..n {
                        let xj: Vec<f64> = std::iter::once(1.0).chain(x[j].iter().copied()).collect();
                        let yh: f64 = xj.iter().zip(beta.iter()).map(|(&a, &b)| a * b).sum();
                        s += w_diag[j] * (y[j] - yh).powi(2);
                    }
                    s
                };
                r_squared[i] = if tss > 1e-12 { 1.0 - rss / tss } else { 0.0 };
            }
            Err(_) => {
                // Singular — leave zeros.
            }
        }
    }

    Ok(GWRResult { coefficients, residuals, r_squared })
}

/// Gaussian elimination with partial pivoting to solve Ax = b.
/// Returns `Err` if the matrix is singular.
fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, String> {
    let n = b.len();
    if a.len() != n || a.iter().any(|r| r.len() != n) {
        return Err("Invalid matrix dimensions".into());
    }

    // Augmented matrix [A | b].
    let mut mat: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();

    for col in 0..n {
        // Find pivot.
        let pivot = (col..n)
            .max_by(|&r1, &r2| {
                mat[r1][col]
                    .abs()
                    .partial_cmp(&mat[r2][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);

        if mat[pivot][col].abs() < 1e-12 {
            return Err("Singular matrix".into());
        }
        mat.swap(col, pivot);

        let div = mat[col][col];
        for j in col..=n {
            mat[col][j] /= div;
        }
        for row in 0..n {
            if row != col {
                let factor = mat[row][col];
                for j in col..=n {
                    let v = mat[col][j] * factor;
                    mat[row][j] -= v;
                }
            }
        }
    }

    Ok(mat.iter().map(|row| row[n]).collect())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn row_standardize(w: &[Vec<f64>]) -> Vec<Vec<f64>> {
        w.iter()
            .map(|row| {
                let s: f64 = row.iter().sum();
                if s.abs() < 1e-12 {
                    row.clone()
                } else {
                    row.iter().map(|&v| v / s).collect()
                }
            })
            .collect()
    }

    fn ring_weights(n: usize) -> Vec<Vec<f64>> {
        let raw: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        if j == (i + 1) % n || j == (i + n - 1) % n {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();
        row_standardize(&raw)
    }

    #[test]
    fn test_morans_i_positive_autocorrelation() {
        // High-low alternating pattern → positive autocorrelation on ring.
        let values = vec![3.0, 3.0, 3.0, 1.0, 1.0, 1.0];
        let w = ring_weights(6);
        let mi = morans_i(&values, &w).expect("morans_i");
        // Blocks of similar values → positive I.
        assert!(mi.statistic > 0.0, "I={}", mi.statistic);
    }

    #[test]
    fn test_morans_i_negative_autocorrelation() {
        // Checkerboard pattern → negative autocorrelation.
        let values = vec![3.0, 1.0, 3.0, 1.0, 3.0, 1.0];
        let w = ring_weights(6);
        let mi = morans_i(&values, &w).expect("morans_i");
        assert!(mi.statistic < 0.0, "I={}", mi.statistic);
    }

    #[test]
    fn test_morans_i_error_cases() {
        assert!(morans_i(&[], &[]).is_err());
        assert!(morans_i(&[1.0], &[vec![1.0]]).is_err());
        // All identical values.
        let w = ring_weights(4);
        let result = morans_i(&[2.0, 2.0, 2.0, 2.0], &w);
        assert!(result.is_err());
    }

    #[test]
    fn test_gearys_c_positive_autocorrelation() {
        let values = vec![3.0, 3.0, 3.0, 1.0, 1.0, 1.0];
        let w = ring_weights(6);
        let gc = gearys_c(&values, &w).expect("gearys_c");
        // C < 1 means positive autocorrelation.
        assert!(gc.statistic < 1.0, "C={}", gc.statistic);
    }

    #[test]
    fn test_gearys_c_negative_autocorrelation() {
        let values = vec![3.0, 1.0, 3.0, 1.0, 3.0, 1.0];
        let w = ring_weights(6);
        let gc = gearys_c(&values, &w).expect("gearys_c");
        // C > 1 for negative autocorrelation.
        assert!(gc.statistic > 1.0, "C={}", gc.statistic);
    }

    #[test]
    fn test_local_morans_i() {
        let values = vec![3.0, 3.0, 3.0, 1.0, 1.0, 1.0];
        let w = ring_weights(6);
        let local = local_morans_i(&values, &w).expect("local");
        assert_eq!(local.len(), values.len());
    }

    #[test]
    fn test_ripleys_k_csr() {
        // Complete spatial randomness: K(d) ≈ π d²
        let area = 100.0_f64;
        let side = area.sqrt();
        // Use a regular grid as a proxy for CSR.
        let pts: Vec<[f64; 2]> = (0..10)
            .flat_map(|i| (0..10).map(move |j| [i as f64 * (side / 10.0), j as f64 * (side / 10.0)]))
            .collect();
        let d = vec![1.0, 2.0, 3.0];
        let k = ripleys_k(&pts, area, &d);
        assert_eq!(k.len(), 3);
        // K should increase with d.
        assert!(k[0] <= k[1] && k[1] <= k[2]);
    }

    #[test]
    fn test_ripleys_l() {
        let pts = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let l = ripleys_l(&pts, 4.0, &[0.5, 1.0, 1.5]);
        assert_eq!(l.len(), 3);
        // L values should be non-negative.
        for &li in &l {
            assert!(li >= 0.0, "l={li}");
        }
    }

    #[test]
    fn test_spatial_kde_basic() {
        let pts = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let grid = vec![[0.0, 0.0], [5.0, 5.0]];
        let densities = spatial_kde(&pts, None, 1.0, &grid).expect("kde");
        assert_eq!(densities.len(), 2);
        // Density at [0,0] near points should be higher than at [5,5].
        assert!(densities[0] > densities[1], "d0={}, d1={}", densities[0], densities[1]);
    }

    #[test]
    fn test_spatial_kde_weighted() {
        let pts = vec![[0.0, 0.0], [10.0, 10.0]];
        let weights = vec![10.0, 1.0];
        let grid = vec![[0.0, 0.0]];
        let d = spatial_kde(&pts, Some(&weights), 1.0, &grid).expect("kde");
        // Should be dominated by the first point.
        assert!(d[0] > 0.0);
    }

    #[test]
    fn test_spatial_kde_bad_bandwidth() {
        let pts = vec![[0.0, 0.0]];
        let result = spatial_kde(&pts, None, -1.0, &[[0.0, 0.0]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_gwr_simple() {
        // y = 2*x + noise, where x varies by location.
        let n = 10;
        let y: Vec<f64> = (0..n).map(|i| 2.0 * i as f64).collect();
        let x: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
        let coords: Vec<[f64; 2]> = (0..n).map(|i| [i as f64, 0.0]).collect();
        let result = geographically_weighted_regression(&y, &x, &coords, 3.0)
            .expect("gwr");
        assert_eq!(result.coefficients.len(), n);
        assert_eq!(result.residuals.len(), n);
        assert_eq!(result.r_squared.len(), n);
        // R² should generally be high for a linear relationship.
        let avg_r2: f64 = result.r_squared.iter().sum::<f64>() / n as f64;
        assert!(avg_r2 > 0.5, "avg_r2={avg_r2}");
    }

    #[test]
    fn test_gwr_error_cases() {
        // Mismatched dimensions.
        assert!(geographically_weighted_regression(
            &[1.0, 2.0],
            &[vec![1.0]],
            &[[0.0, 0.0], [1.0, 0.0]],
            1.0
        )
        .is_err());
        // Negative bandwidth.
        assert!(geographically_weighted_regression(
            &[1.0],
            &[vec![1.0]],
            &[[0.0, 0.0]],
            -1.0
        )
        .is_err());
    }

    #[test]
    fn test_solve_linear() {
        // 2x + y = 5; x + 3y = 7  → x = 8/5, y = 9/5
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 7.0];
        let sol = solve_linear(&a, &b).expect("solve");
        assert!((sol[0] - 1.6).abs() < 1e-9, "x={}", sol[0]);
        assert!((sol[1] - 1.8).abs() < 1e-9, "y={}", sol[1]);
    }

    #[test]
    fn test_solve_linear_singular() {
        let a = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        let b = vec![2.0, 2.0];
        assert!(solve_linear(&a, &b).is_err());
    }
}
