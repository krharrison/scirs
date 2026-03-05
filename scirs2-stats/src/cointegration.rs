//! Cointegration Analysis
//!
//! This module provides methods for testing and estimating cointegrating
//! relationships in multivariate time series:
//!
//! - **Engle-Granger two-step method**: OLS-based residual unit root test
//! - **Johansen cointegration test**: maximum likelihood approach via
//!   trace and max-eigenvalue statistics
//! - **VECM** (Vector Error Correction Model): estimation and representation
//! - **Critical value tables**: for trace and max-eigenvalue statistics
//! - **Cointegrating rank determination**: sequential testing procedure
//!
//! # References
//!
//! - Engle, R.F. & Granger, C.W.J. (1987). Co-Integration and Error Correction:
//!   Representation, Estimation, and Testing. Econometrica.
//! - Johansen, S. (1991). Estimation and Hypothesis Testing of Cointegration Vectors
//!   in Gaussian Vector Autoregressive Models. Econometrica.
//! - Johansen, S. (1995). Likelihood-Based Inference in Cointegrated Vector
//!   Autoregressive Models. Oxford University Press.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of the Engle-Granger cointegration test
#[derive(Debug, Clone)]
pub struct EngleGrangerResult {
    /// ADF test statistic on the cointegrating residuals
    pub statistic: f64,
    /// Approximate p-value
    pub p_value: f64,
    /// Number of lags used in the ADF test
    pub used_lags: usize,
    /// Critical values at 1%, 5%, 10%
    pub critical_values: CointegrationCriticalValues,
    /// Estimated cointegrating vector (OLS coefficients)
    pub cointegrating_vector: Array1<f64>,
    /// Residuals from the cointegrating regression
    pub residuals: Array1<f64>,
}

/// Result of the Johansen cointegration test
#[derive(Debug, Clone)]
pub struct JohansenResult {
    /// Trace test statistics for r=0, r<=1, ..., r<=n-1
    pub trace_stats: Vec<f64>,
    /// Trace test p-values
    pub trace_p_values: Vec<f64>,
    /// Max-eigenvalue test statistics
    pub max_eig_stats: Vec<f64>,
    /// Max-eigenvalue test p-values
    pub max_eig_p_values: Vec<f64>,
    /// Estimated eigenvalues (sorted descending)
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors (cointegrating vectors, columns of beta)
    pub eigenvectors: Array2<f64>,
    /// Estimated cointegrating rank (trace test, 5% level)
    pub cointegrating_rank: usize,
    /// Number of variables
    pub n_vars: usize,
    /// Deterministic specification
    pub det_order: JohansenDeterministic,
}

/// VECM estimation result
#[derive(Debug, Clone)]
pub struct VecmResult {
    /// Adjustment (loading) matrix alpha (n_vars x rank)
    pub alpha: Array2<f64>,
    /// Cointegrating matrix beta (n_vars x rank)
    pub beta: Array2<f64>,
    /// Short-run coefficient matrices (one per lag)
    pub gamma: Vec<Array2<f64>>,
    /// Deterministic term coefficients
    pub deterministic: Option<Array2<f64>>,
    /// Residual matrix (T x n_vars)
    pub residuals: Array2<f64>,
    /// Cointegrating rank used
    pub rank: usize,
    /// Number of lags in the VECM
    pub lags: usize,
    /// Log-likelihood
    pub log_likelihood: f64,
}

/// Critical values for cointegration tests
#[derive(Debug, Clone, Copy)]
pub struct CointegrationCriticalValues {
    /// 1% critical value
    pub one_pct: f64,
    /// 5% critical value
    pub five_pct: f64,
    /// 10% critical value
    pub ten_pct: f64,
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Deterministic term specification for Johansen test
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JohansenDeterministic {
    /// No deterministic terms (Case 1)
    None,
    /// Restricted constant (constant in the cointegrating relation only, Case 2)
    RestrictedConstant,
    /// Unrestricted constant (Case 3, default)
    UnrestrictedConstant,
    /// Restricted trend (linear trend in cointegrating relation, Case 4)
    RestrictedTrend,
    /// Unrestricted constant and restricted trend (Case 5)
    UnrestrictedConstantRestrictedTrend,
}

// ---------------------------------------------------------------------------
// Helper: matrix operations
// ---------------------------------------------------------------------------

/// Solve A*x = b for symmetric positive-definite A via Cholesky.
fn solve_spd(a: &Array2<f64>, b: &Array2<f64>) -> StatsResult<Array2<f64>> {
    let n = a.nrows();
    let m = b.ncols();
    if n != a.ncols() || n != b.nrows() {
        return Err(StatsError::DimensionMismatch(
            "solve_spd: dimension mismatch".into(),
        ));
    }
    // Cholesky with regularization
    let mut l = Array2::<f64>::zeros((n, n));
    let ridge = {
        let max_diag = (0..n).map(|i| a[[i, i]].abs()).fold(0.0_f64, f64::max);
        1e-12 * max_diag.max(1e-12)
    };
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for kk in 0..j {
                sum += l[[i, kk]] * l[[j, kk]];
            }
            if i == j {
                let diag = a[[i, i]] + ridge - sum;
                l[[i, j]] = if diag > 0.0 { diag.sqrt() } else { 1e-15 };
            } else {
                let denom = l[[j, j]];
                l[[i, j]] = if denom.abs() > 1e-15 {
                    (a[[i, j]] - sum) / denom
                } else {
                    0.0
                };
            }
        }
    }
    // Solve for each column of b
    let mut result = Array2::<f64>::zeros((n, m));
    for col in 0..m {
        // Forward: L*z = b_col
        let mut z = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..i {
                s += l[[i, j]] * z[j];
            }
            z[i] = if l[[i, i]].abs() > 1e-15 {
                (b[[i, col]] - s) / l[[i, i]]
            } else {
                0.0
            };
        }
        // Backward: L^T * x = z
        for i in (0..n).rev() {
            let mut s = 0.0;
            for j in (i + 1)..n {
                s += l[[j, i]] * result[[j, col]];
            }
            result[[i, col]] = if l[[i, i]].abs() > 1e-15 {
                (z[i] - s) / l[[i, i]]
            } else {
                0.0
            };
        }
    }
    Ok(result)
}

/// Compute eigenvalues and eigenvectors of a symmetric matrix using Jacobi iteration.
fn symmetric_eigen(a: &Array2<f64>, max_iter: usize) -> StatsResult<(Vec<f64>, Array2<f64>)> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(StatsError::DimensionMismatch(
            "symmetric_eigen: matrix must be square".into(),
        ));
    }
    let mut mat = a.clone();
    let mut v = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        v[[i, i]] = 1.0;
    }
    for _iter in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if mat[[i, j]].abs() > max_val {
                    max_val = mat[[i, j]].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-14 {
            break;
        }
        // Compute rotation
        let app = mat[[p, p]];
        let aqq = mat[[q, q]];
        let apq = mat[[p, q]];
        let theta = if (app - aqq).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };
        let c = theta.cos();
        let s = theta.sin();
        // Apply Givens rotation
        let mut new_mat = mat.clone();
        for i in 0..n {
            if i != p && i != q {
                new_mat[[i, p]] = c * mat[[i, p]] + s * mat[[i, q]];
                new_mat[[p, i]] = new_mat[[i, p]];
                new_mat[[i, q]] = -s * mat[[i, p]] + c * mat[[i, q]];
                new_mat[[q, i]] = new_mat[[i, q]];
            }
        }
        new_mat[[p, p]] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
        new_mat[[q, q]] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
        new_mat[[p, q]] = 0.0;
        new_mat[[q, p]] = 0.0;
        mat = new_mat;
        // Update eigenvectors
        let mut new_v = v.clone();
        for i in 0..n {
            new_v[[i, p]] = c * v[[i, p]] + s * v[[i, q]];
            new_v[[i, q]] = -s * v[[i, p]] + c * v[[i, q]];
        }
        v = new_v;
    }
    let eigenvalues: Vec<f64> = (0..n).map(|i| mat[[i, i]]).collect();
    Ok((eigenvalues, v))
}

/// Solve the generalized eigenvalue problem A*x = lambda*B*x
/// Returns (eigenvalues, eigenvectors) sorted by descending eigenvalue.
fn generalized_eigen(a: &Array2<f64>, b: &Array2<f64>) -> StatsResult<(Vec<f64>, Array2<f64>)> {
    let n = a.nrows();
    // Compute B^{-1} * A via solving B * X = A
    let b_inv_a = solve_spd(b, a)?;
    // Symmetrize for numerical stability: C = B^{-1/2} A B^{-1/2}
    // For simplicity, use Jacobi on B^{-1}*A (may not be exactly symmetric)
    // Symmetrize
    let mut sym = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            sym[[i, j]] = 0.5 * (b_inv_a[[i, j]] + b_inv_a[[j, i]]);
        }
    }
    let (mut evals, evecs) = symmetric_eigen(&sym, 200)?;
    // Sort by descending eigenvalue
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        evals[b]
            .partial_cmp(&evals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted_evals: Vec<f64> = indices.iter().map(|&i| evals[i]).collect();
    let mut sorted_evecs = Array2::<f64>::zeros((n, n));
    for (new_col, &old_col) in indices.iter().enumerate() {
        for row in 0..n {
            sorted_evecs[[row, new_col]] = evecs[[row, old_col]];
        }
    }
    Ok((sorted_evals, sorted_evecs))
}

/// Mean-center columns of a matrix
fn demean_columns(x: &Array2<f64>) -> Array2<f64> {
    let n = x.nrows() as f64;
    let k = x.ncols();
    let mut result = x.clone();
    for j in 0..k {
        let mean: f64 = (0..x.nrows()).map(|i| x[[i, j]]).sum::<f64>() / n;
        for i in 0..x.nrows() {
            result[[i, j]] -= mean;
        }
    }
    result
}

/// First-difference each column of a matrix
fn diff_matrix(x: &Array2<f64>) -> Array2<f64> {
    let n = x.nrows();
    let k = x.ncols();
    if n < 2 {
        return Array2::zeros((0, k));
    }
    let mut d = Array2::<f64>::zeros((n - 1, k));
    for j in 0..k {
        for i in 0..(n - 1) {
            d[[i, j]] = x[[i + 1, j]] - x[[i, j]];
        }
    }
    d
}

// ---------------------------------------------------------------------------
// Engle-Granger two-step method
// ---------------------------------------------------------------------------

/// Engle-Granger critical values (MacKinnon 1990/2010, larger than standard ADF
/// because of the generated regressor problem).
fn eg_critical_values(n_vars: usize) -> CointegrationCriticalValues {
    match n_vars {
        2 => CointegrationCriticalValues {
            one_pct: -3.90,
            five_pct: -3.34,
            ten_pct: -3.04,
        },
        3 => CointegrationCriticalValues {
            one_pct: -4.29,
            five_pct: -3.74,
            ten_pct: -3.45,
        },
        4 => CointegrationCriticalValues {
            one_pct: -4.64,
            five_pct: -4.10,
            ten_pct: -3.81,
        },
        5 => CointegrationCriticalValues {
            one_pct: -4.96,
            five_pct: -4.42,
            ten_pct: -4.13,
        },
        _ => CointegrationCriticalValues {
            // Approximate for larger n_vars
            one_pct: -4.96 - 0.3 * ((n_vars as f64) - 5.0),
            five_pct: -4.42 - 0.3 * ((n_vars as f64) - 5.0),
            ten_pct: -4.13 - 0.3 * ((n_vars as f64) - 5.0),
        },
    }
}

/// Perform the Engle-Granger two-step cointegration test.
///
/// Step 1: Regress the first variable on the remaining variables using OLS.
/// Step 2: Test the residuals for a unit root using ADF.
///
/// The null hypothesis is no cointegration (residuals have a unit root).
///
/// # Arguments
/// * `data` - Matrix of shape (T, n_vars) where each column is a time series
/// * `max_lags` - Maximum lags for the ADF test on residuals
///
/// # Example
/// ```
/// use scirs2_stats::cointegration::engle_granger_test;
/// use scirs2_core::ndarray::Array2;
///
/// // Two cointegrated series: y2 = 2*y1 + noise
/// let n = 100;
/// let mut data = Array2::<f64>::zeros((n, 2));
/// let mut cumsum = 0.0_f64;
/// for i in 0..n {
///     cumsum += ((i as f64) * 1.3).sin() * 0.3;
///     data[[i, 0]] = cumsum;
///     data[[i, 1]] = 2.0 * cumsum + ((i as f64) * 0.7).sin() * 0.1;
/// }
/// let result = engle_granger_test(&data.view(), None).expect("EG test failed");
/// assert!(result.statistic.is_finite());
/// ```
pub fn engle_granger_test(
    data: &ArrayView2<f64>,
    max_lags: Option<usize>,
) -> StatsResult<EngleGrangerResult> {
    let n = data.nrows();
    let k = data.ncols();
    if k < 2 {
        return Err(StatsError::InvalidArgument(
            "Engle-Granger requires at least 2 variables".into(),
        ));
    }
    if n < 20 {
        return Err(StatsError::InsufficientData(
            "Engle-Granger requires at least 20 observations".into(),
        ));
    }
    // Step 1: OLS regression y1 = a + b2*y2 + ... + bk*yk + e
    let mut design = Array2::<f64>::zeros((n, k)); // constant + (k-1) regressors
    for i in 0..n {
        design[[i, 0]] = 1.0; // constant
        for j in 1..k {
            design[[i, j]] = data[[i, j]];
        }
    }
    let dep: Array1<f64> = Array1::from_vec((0..n).map(|i| data[[i, 0]]).collect());
    let ols = crate::stationarity::ols_regression(&dep.view(), &design)?;

    // Step 2: ADF test on residuals
    let resid = &ols.residuals;
    let adf = crate::stationarity::adf_test(
        &resid.view(),
        max_lags,
        crate::stationarity::AdfRegression::None,
        crate::stationarity::LagCriterion::Aic,
    )?;

    // Use EG-specific critical values
    let cv = eg_critical_values(k);
    let p_value = eg_p_value(adf.statistic, k);

    Ok(EngleGrangerResult {
        statistic: adf.statistic,
        p_value,
        used_lags: adf.used_lags,
        critical_values: cv,
        cointegrating_vector: ols.coefficients,
        residuals: ols.residuals,
    })
}

/// Approximate p-value for Engle-Granger using the EG critical values
fn eg_p_value(stat: f64, n_vars: usize) -> f64 {
    let cv = eg_critical_values(n_vars);
    if stat <= cv.one_pct {
        0.005
    } else if stat <= cv.five_pct {
        let frac = (stat - cv.one_pct) / (cv.five_pct - cv.one_pct);
        0.01 + frac * 0.04
    } else if stat <= cv.ten_pct {
        let frac = (stat - cv.five_pct) / (cv.ten_pct - cv.five_pct);
        0.05 + frac * 0.05
    } else {
        let frac = (stat - cv.ten_pct) / cv.ten_pct.abs().max(1.0);
        (0.10 + frac * 0.4).min(0.999)
    }
}

// ---------------------------------------------------------------------------
// Johansen critical value tables
// ---------------------------------------------------------------------------

/// Johansen trace test critical values (Case 3: unrestricted constant)
/// Source: Osterwald-Lenum (1992)
fn johansen_trace_cv(n_vars: usize, rank: usize) -> CointegrationCriticalValues {
    let p_minus_r = n_vars - rank;
    match p_minus_r {
        1 => CointegrationCriticalValues {
            one_pct: 11.65,
            five_pct: 8.18,
            ten_pct: 6.50,
        },
        2 => CointegrationCriticalValues {
            one_pct: 23.52,
            five_pct: 17.95,
            ten_pct: 15.66,
        },
        3 => CointegrationCriticalValues {
            one_pct: 37.22,
            five_pct: 31.52,
            ten_pct: 28.71,
        },
        4 => CointegrationCriticalValues {
            one_pct: 54.46,
            five_pct: 47.21,
            ten_pct: 43.95,
        },
        5 => CointegrationCriticalValues {
            one_pct: 75.98,
            five_pct: 68.52,
            ten_pct: 64.84,
        },
        _ => {
            // Extrapolation for larger systems
            let base = 75.98 + 25.0 * ((p_minus_r as f64) - 5.0);
            CointegrationCriticalValues {
                one_pct: base,
                five_pct: base - 7.5,
                ten_pct: base - 11.0,
            }
        }
    }
}

/// Johansen max-eigenvalue critical values (Case 3)
fn johansen_max_eig_cv(n_vars: usize, rank: usize) -> CointegrationCriticalValues {
    let p_minus_r = n_vars - rank;
    match p_minus_r {
        1 => CointegrationCriticalValues {
            one_pct: 11.65,
            five_pct: 8.18,
            ten_pct: 6.50,
        },
        2 => CointegrationCriticalValues {
            one_pct: 19.19,
            five_pct: 14.90,
            ten_pct: 12.91,
        },
        3 => CointegrationCriticalValues {
            one_pct: 25.75,
            five_pct: 21.07,
            ten_pct: 18.90,
        },
        4 => CointegrationCriticalValues {
            one_pct: 32.14,
            five_pct: 27.14,
            ten_pct: 25.12,
        },
        5 => CointegrationCriticalValues {
            one_pct: 38.78,
            five_pct: 33.32,
            ten_pct: 31.22,
        },
        _ => {
            let base = 38.78 + 7.0 * ((p_minus_r as f64) - 5.0);
            CointegrationCriticalValues {
                one_pct: base,
                five_pct: base - 5.5,
                ten_pct: base - 7.5,
            }
        }
    }
}

/// Approximate p-value for Johansen statistics via interpolation
fn johansen_p_value(stat: f64, cv: &CointegrationCriticalValues) -> f64 {
    if stat >= cv.one_pct {
        let overshoot = (stat - cv.one_pct) / cv.one_pct.max(0.001);
        (0.01 - overshoot * 0.005).max(0.001)
    } else if stat >= cv.five_pct {
        let frac = (stat - cv.five_pct) / (cv.one_pct - cv.five_pct);
        0.05 - frac * 0.04
    } else if stat >= cv.ten_pct {
        let frac = (stat - cv.ten_pct) / (cv.five_pct - cv.ten_pct);
        0.10 - frac * 0.05
    } else {
        let frac = stat / cv.ten_pct.max(0.001);
        (0.10 + (1.0 - frac) * 0.4).min(0.999)
    }
}

// ---------------------------------------------------------------------------
// Johansen cointegration test
// ---------------------------------------------------------------------------

/// Perform the Johansen cointegration test.
///
/// Tests for the number of cointegrating relationships among a set of variables
/// using the maximum likelihood approach.
///
/// # Arguments
/// * `data` - Matrix of shape (T, n_vars)
/// * `lags` - Number of lags in the underlying VAR model (default: 1)
/// * `det_order` - Deterministic specification
///
/// # Example
/// ```
/// use scirs2_stats::cointegration::{johansen_test, JohansenDeterministic};
/// use scirs2_core::ndarray::Array2;
///
/// let n = 100;
/// let mut data = Array2::<f64>::zeros((n, 2));
/// let mut cumsum = 0.0_f64;
/// for i in 0..n {
///     cumsum += ((i as f64) * 1.3).sin() * 0.3;
///     data[[i, 0]] = cumsum;
///     data[[i, 1]] = 2.0 * cumsum + ((i as f64) * 0.7).sin() * 0.1;
/// }
/// let result = johansen_test(&data.view(), 1, JohansenDeterministic::UnrestrictedConstant)
///     .expect("Johansen test failed");
/// assert!(result.trace_stats.len() > 0);
/// ```
pub fn johansen_test(
    data: &ArrayView2<f64>,
    lags: usize,
    det_order: JohansenDeterministic,
) -> StatsResult<JohansenResult> {
    let n = data.nrows();
    let k = data.ncols();
    if k < 2 {
        return Err(StatsError::InvalidArgument(
            "Johansen test requires at least 2 variables".into(),
        ));
    }
    let p = lags.max(1);
    if n < 2 * p + k + 10 {
        return Err(StatsError::InsufficientData(format!(
            "Johansen test needs more observations (n={}, lags={}, vars={})",
            n, p, k
        )));
    }

    // Convert to owned Array2
    let data_owned = data.to_owned();

    // First differences
    let dy = diff_matrix(&data_owned);
    let t_eff = dy.nrows() - p;
    if t_eff < k + 2 {
        return Err(StatsError::InsufficientData(
            "effective sample too small for Johansen".into(),
        ));
    }

    // Build the Y0 (dy_t) and Y1 (y_{t-1}) matrices, plus lagged diffs
    // Y0: dy from period p to end -> (t_eff x k)
    let mut y0 = Array2::<f64>::zeros((t_eff, k));
    let mut y1 = Array2::<f64>::zeros((t_eff, k));
    for i in 0..t_eff {
        for j in 0..k {
            y0[[i, j]] = dy[[i + p, j]];
            y1[[i, j]] = data_owned[[i + p, j]]; // y_{t-1}
        }
    }

    // Lagged differences for short-run dynamics
    let n_det = match det_order {
        JohansenDeterministic::None => 0,
        JohansenDeterministic::RestrictedConstant => 0, // goes into CI space
        JohansenDeterministic::UnrestrictedConstant => 1,
        JohansenDeterministic::RestrictedTrend => 1,
        JohansenDeterministic::UnrestrictedConstantRestrictedTrend => 1,
    };
    let n_x_cols = (p - 1) * k + n_det;
    let has_x = n_x_cols > 0;

    // Build auxiliary regressors (lagged diffs + deterministic)
    let mut x_mat = if has_x {
        let mut x = Array2::<f64>::zeros((t_eff, n_x_cols));
        let mut col = 0;
        // Lagged differences
        for lag in 1..p {
            for j in 0..k {
                for i in 0..t_eff {
                    x[[i, col]] = dy[[i + p - lag, j]];
                }
                col += 1;
            }
        }
        // Deterministic
        if n_det > 0 {
            for i in 0..t_eff {
                x[[i, col]] = 1.0; // constant
            }
        }
        Some(x)
    } else {
        None
    };

    // Concentrate out the auxiliary regressors from Y0 and Y1
    let (r0, r1) = if let Some(ref x) = x_mat {
        // Regress Y0 on X, get residuals R0
        // Regress Y1 on X, get residuals R1
        let xtx = x.t().dot(x);
        let r0 = residualize(&y0, x, &xtx)?;
        let r1 = residualize(&y1, x, &xtx)?;
        (r0, r1)
    } else {
        let r0 = demean_columns(&y0);
        let r1 = demean_columns(&y1);
        (r0, r1)
    };

    let tf = t_eff as f64;

    // Product moment matrices
    let s00 = r0.t().dot(&r0) / tf;
    let s11 = r1.t().dot(&r1) / tf;
    let s01 = r0.t().dot(&r1) / tf;
    let s10 = r1.t().dot(&r0) / tf;

    // Solve the generalized eigenvalue problem:
    //   |lambda * S11 - S10 * S00^{-1} * S01| = 0
    let s00_inv_s01 = solve_spd(&s00.to_owned(), &s01.to_owned())?;
    let candidate = s10.dot(&s00_inv_s01);

    let (eigenvalues, eigenvectors) = generalized_eigen(&candidate.to_owned(), &s11.to_owned())?;

    // Clamp eigenvalues to [0, 1)
    let eigenvalues: Vec<f64> = eigenvalues
        .iter()
        .map(|&ev| ev.max(0.0).min(1.0 - 1e-15))
        .collect();

    // Compute trace and max-eigenvalue statistics
    let mut trace_stats = Vec::with_capacity(k);
    let mut trace_p_values = Vec::with_capacity(k);
    let mut max_eig_stats = Vec::with_capacity(k);
    let mut max_eig_p_values = Vec::with_capacity(k);

    for r in 0..k {
        // Trace statistic: -T * sum_{i=r+1}^{k} ln(1 - lambda_i)
        let trace: f64 = eigenvalues[r..]
            .iter()
            .map(|&lam| -(tf) * (1.0 - lam).max(1e-15).ln())
            .sum();
        let max_eig = -(tf) * (1.0 - eigenvalues[r]).max(1e-15).ln();

        let trace_cv = johansen_trace_cv(k, r);
        let max_cv = johansen_max_eig_cv(k, r);

        trace_stats.push(trace);
        trace_p_values.push(johansen_p_value(trace, &trace_cv));
        max_eig_stats.push(max_eig);
        max_eig_p_values.push(johansen_p_value(max_eig, &max_cv));
    }

    // Determine cointegrating rank using trace test at 5%
    let mut coint_rank = 0;
    for r in 0..k {
        let cv = johansen_trace_cv(k, r);
        if trace_stats[r] > cv.five_pct {
            coint_rank = r + 1;
        } else {
            break;
        }
    }

    Ok(JohansenResult {
        trace_stats,
        trace_p_values,
        max_eig_stats,
        max_eig_p_values,
        eigenvalues,
        eigenvectors: eigenvectors.to_owned(),
        cointegrating_rank: coint_rank,
        n_vars: k,
        det_order,
    })
}

/// Residualize Y on X: return Y - X * (X'X)^{-1} * X'Y
fn residualize(y: &Array2<f64>, x: &Array2<f64>, xtx: &Array2<f64>) -> StatsResult<Array2<f64>> {
    let xty = x.t().dot(y);
    let beta = solve_spd(xtx, &xty)?;
    let fitted = x.dot(&beta);
    let mut resid = y.clone();
    for i in 0..y.nrows() {
        for j in 0..y.ncols() {
            resid[[i, j]] -= fitted[[i, j]];
        }
    }
    Ok(resid)
}

/// Determine the cointegrating rank using sequential testing.
///
/// Tests r=0, r=1, ... until the null is not rejected. Returns the
/// estimated rank at the given significance level.
///
/// # Arguments
/// * `johansen_result` - Result from `johansen_test`
/// * `significance` - Significance level (default 0.05)
/// * `method` - "trace" or "max_eig"
pub fn determine_rank(
    result: &JohansenResult,
    significance: f64,
    method: &str,
) -> StatsResult<usize> {
    let k = result.n_vars;
    let use_trace = method != "max_eig";

    for r in 0..k {
        let cv = if use_trace {
            johansen_trace_cv(k, r)
        } else {
            johansen_max_eig_cv(k, r)
        };
        let stat = if use_trace {
            result.trace_stats[r]
        } else {
            result.max_eig_stats[r]
        };
        let critical = if significance <= 0.01 {
            cv.one_pct
        } else if significance <= 0.05 {
            cv.five_pct
        } else {
            cv.ten_pct
        };
        if stat <= critical {
            return Ok(r);
        }
    }
    Ok(k)
}

// ---------------------------------------------------------------------------
// VECM estimation
// ---------------------------------------------------------------------------

/// Estimate a Vector Error Correction Model (VECM).
///
/// The VECM representation: Delta y_t = alpha * beta' * y_{t-1} + sum Gamma_i * Delta y_{t-i} + mu + e_t
///
/// # Arguments
/// * `data` - Matrix of shape (T, n_vars)
/// * `rank` - Cointegrating rank (from Johansen test or `determine_rank`)
/// * `lags` - Number of lags in the VECM (number of lagged differences)
///
/// # Example
/// ```
/// use scirs2_stats::cointegration::estimate_vecm;
/// use scirs2_core::ndarray::Array2;
///
/// let n = 100;
/// let mut data = Array2::<f64>::zeros((n, 2));
/// let mut cumsum = 0.0_f64;
/// for i in 0..n {
///     cumsum += ((i as f64) * 1.3).sin() * 0.3;
///     data[[i, 0]] = cumsum;
///     data[[i, 1]] = 2.0 * cumsum + ((i as f64) * 0.7).sin() * 0.1;
/// }
/// let result = estimate_vecm(&data.view(), 1, 1).expect("VECM estimation failed");
/// assert_eq!(result.rank, 1);
/// ```
pub fn estimate_vecm(data: &ArrayView2<f64>, rank: usize, lags: usize) -> StatsResult<VecmResult> {
    let n = data.nrows();
    let k = data.ncols();
    if rank == 0 || rank > k {
        return Err(StatsError::InvalidArgument(format!(
            "rank must be in [1, {}], got {}",
            k, rank
        )));
    }
    let p = lags.max(1);
    if n < 2 * p + k + 10 {
        return Err(StatsError::InsufficientData(
            "insufficient observations for VECM".into(),
        ));
    }

    // Get beta from Johansen
    let joh = johansen_test(data, p, JohansenDeterministic::UnrestrictedConstant)?;
    // beta = first `rank` columns of eigenvectors
    let mut beta = Array2::<f64>::zeros((k, rank));
    for i in 0..k {
        for r in 0..rank {
            beta[[i, r]] = joh.eigenvectors[[i, r]];
        }
    }

    let data_owned = data.to_owned();
    let dy = diff_matrix(&data_owned);
    let t_eff = dy.nrows() - p;
    if t_eff < 2 {
        return Err(StatsError::InsufficientData(
            "effective sample too small for VECM".into(),
        ));
    }

    // Build dependent: dy_t for t = p..end
    let mut y_dep = Array2::<f64>::zeros((t_eff, k));
    for i in 0..t_eff {
        for j in 0..k {
            y_dep[[i, j]] = dy[[i + p, j]];
        }
    }

    // Build regressors: error correction term + lagged diffs + constant
    // EC term: beta' * y_{t-1} -> (t_eff x rank)
    let mut ec_term = Array2::<f64>::zeros((t_eff, rank));
    for i in 0..t_eff {
        for r in 0..rank {
            let mut sum = 0.0;
            for j in 0..k {
                sum += beta[[j, r]] * data_owned[[i + p, j]]; // y_{t-1}
            }
            ec_term[[i, r]] = sum;
        }
    }

    let n_reg = rank + (p - 1) * k + 1; // EC + lagged diffs + constant
    let mut x_reg = Array2::<f64>::zeros((t_eff, n_reg));
    for i in 0..t_eff {
        let mut col = 0;
        // EC terms
        for r in 0..rank {
            x_reg[[i, col]] = ec_term[[i, r]];
            col += 1;
        }
        // Lagged diffs
        for lag in 1..p {
            for j in 0..k {
                x_reg[[i, col]] = dy[[i + p - lag, j]];
                col += 1;
            }
        }
        // Constant
        x_reg[[i, col]] = 1.0;
    }

    // Solve for each equation
    let xtx = x_reg.t().dot(&x_reg);
    let xty = x_reg.t().dot(&y_dep);
    let coef = solve_spd(&xtx, &xty)?;

    // Extract alpha (loading matrix)
    let mut alpha = Array2::<f64>::zeros((k, rank));
    for j in 0..k {
        for r in 0..rank {
            alpha[[j, r]] = coef[[r, j]];
        }
    }

    // Extract Gamma matrices
    let mut gamma_mats = Vec::new();
    if p > 1 {
        for lag in 0..(p - 1) {
            let mut g = Array2::<f64>::zeros((k, k));
            for j in 0..k {
                for jj in 0..k {
                    g[[j, jj]] = coef[[rank + lag * k + jj, j]];
                }
            }
            gamma_mats.push(g);
        }
    }

    // Deterministic
    let mut det_coef = Array2::<f64>::zeros((k, 1));
    let det_col = n_reg - 1;
    for j in 0..k {
        det_coef[[j, 0]] = coef[[det_col, j]];
    }

    // Residuals
    let fitted = x_reg.dot(&coef);
    let mut residuals = Array2::<f64>::zeros((t_eff, k));
    for i in 0..t_eff {
        for j in 0..k {
            residuals[[i, j]] = y_dep[[i, j]] - fitted[[i, j]];
        }
    }

    // Log-likelihood (multivariate normal)
    let tf = t_eff as f64;
    let sigma = residuals.t().dot(&residuals) / tf;
    let log_det = (0..k).map(|i| sigma[[i, i]].max(1e-15).ln()).sum::<f64>();
    let log_lik = -0.5 * tf * (k as f64 * (2.0 * std::f64::consts::PI).ln() + log_det + k as f64);

    Ok(VecmResult {
        alpha,
        beta,
        gamma: gamma_mats,
        deterministic: Some(det_coef),
        residuals,
        rank,
        lags: p,
        log_likelihood: log_lik,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_cointegrated_pair(n: usize) -> Array2<f64> {
        let mut data = Array2::<f64>::zeros((n, 2));
        let mut cumsum = 0.0_f64;
        for i in 0..n {
            cumsum += ((i as f64) * 1.3 + 0.5).sin() * 0.3;
            data[[i, 0]] = cumsum;
            data[[i, 1]] = 2.0 * cumsum + ((i as f64) * 0.7 + 0.2).sin() * 0.1;
        }
        data
    }

    fn make_independent_walks(n: usize) -> Array2<f64> {
        let mut data = Array2::<f64>::zeros((n, 2));
        let mut s1 = 0.0_f64;
        let mut s2 = 0.0_f64;
        for i in 0..n {
            s1 += ((i as f64) * 1.7 + 0.3).sin() * 0.3;
            s2 += ((i as f64) * 2.3 + 0.8).sin() * 0.3;
            data[[i, 0]] = s1;
            data[[i, 1]] = s2;
        }
        data
    }

    #[test]
    fn test_engle_granger_cointegrated() {
        let data = make_cointegrated_pair(200);
        let result = engle_granger_test(&data.view(), None);
        assert!(result.is_ok());
        let r = result.expect("EG should succeed");
        assert!(r.statistic.is_finite());
        assert!(r.p_value >= 0.0 && r.p_value <= 1.0);
        assert_eq!(r.cointegrating_vector.len(), 2); // constant + 1 regressor
    }

    #[test]
    fn test_engle_granger_independent() {
        let data = make_independent_walks(200);
        let result = engle_granger_test(&data.view(), None);
        assert!(result.is_ok());
        let r = result.expect("EG should succeed");
        assert!(r.statistic.is_finite());
    }

    #[test]
    fn test_johansen_basic() {
        let data = make_cointegrated_pair(200);
        let result = johansen_test(&data.view(), 1, JohansenDeterministic::UnrestrictedConstant);
        assert!(result.is_ok());
        let r = result.expect("Johansen should succeed");
        assert_eq!(r.trace_stats.len(), 2);
        assert_eq!(r.max_eig_stats.len(), 2);
        assert_eq!(r.eigenvalues.len(), 2);
    }

    #[test]
    fn test_johansen_three_vars() {
        let n = 200;
        let mut data = Array2::<f64>::zeros((n, 3));
        let mut s = 0.0_f64;
        for i in 0..n {
            s += ((i as f64) * 1.1).sin() * 0.3;
            data[[i, 0]] = s;
            data[[i, 1]] = 2.0 * s + ((i as f64) * 0.5).sin() * 0.1;
            data[[i, 2]] = -1.5 * s + ((i as f64) * 0.9).sin() * 0.15;
        }
        let result = johansen_test(&data.view(), 1, JohansenDeterministic::UnrestrictedConstant);
        assert!(result.is_ok());
        let r = result.expect("Johansen 3-var should succeed");
        assert_eq!(r.trace_stats.len(), 3);
    }

    #[test]
    fn test_determine_rank_trace() {
        let data = make_cointegrated_pair(200);
        let joh = johansen_test(&data.view(), 1, JohansenDeterministic::UnrestrictedConstant)
            .expect("Johansen should succeed");
        let rank = determine_rank(&joh, 0.05, "trace");
        assert!(rank.is_ok());
        let r = rank.expect("rank determination should succeed");
        assert!(r <= 2);
    }

    #[test]
    fn test_determine_rank_max_eig() {
        let data = make_cointegrated_pair(200);
        let joh = johansen_test(&data.view(), 1, JohansenDeterministic::UnrestrictedConstant)
            .expect("Johansen should succeed");
        let rank = determine_rank(&joh, 0.05, "max_eig");
        assert!(rank.is_ok());
    }

    #[test]
    fn test_vecm_estimation() {
        let data = make_cointegrated_pair(200);
        let result = estimate_vecm(&data.view(), 1, 1);
        assert!(result.is_ok());
        let r = result.expect("VECM should succeed");
        assert_eq!(r.alpha.nrows(), 2);
        assert_eq!(r.alpha.ncols(), 1);
        assert_eq!(r.beta.nrows(), 2);
        assert_eq!(r.beta.ncols(), 1);
        assert_eq!(r.rank, 1);
    }

    #[test]
    fn test_vecm_with_lags() {
        let data = make_cointegrated_pair(200);
        let result = estimate_vecm(&data.view(), 1, 3);
        assert!(result.is_ok());
        let r = result.expect("VECM with lags should succeed");
        assert_eq!(r.lags, 3);
        assert_eq!(r.gamma.len(), 2); // p-1 = 2 gamma matrices
    }

    #[test]
    fn test_engle_granger_insufficient() {
        let data = Array2::<f64>::zeros((5, 2));
        let result = engle_granger_test(&data.view(), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_johansen_insufficient() {
        let data = Array2::<f64>::zeros((5, 2));
        let result = johansen_test(&data.view(), 1, JohansenDeterministic::UnrestrictedConstant);
        assert!(result.is_err());
    }

    #[test]
    fn test_vecm_invalid_rank() {
        let data = make_cointegrated_pair(200);
        let result = estimate_vecm(&data.view(), 0, 1);
        assert!(result.is_err());
        let result = estimate_vecm(&data.view(), 5, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_eg_critical_values_known() {
        let cv = eg_critical_values(2);
        assert!(cv.one_pct < cv.five_pct);
        assert!(cv.five_pct < cv.ten_pct);
    }

    #[test]
    fn test_johansen_critical_values() {
        let cv = johansen_trace_cv(2, 0);
        assert!(cv.one_pct > cv.five_pct);
        assert!(cv.five_pct > cv.ten_pct);
    }

    #[test]
    fn test_vecm_log_likelihood_finite() {
        let data = make_cointegrated_pair(200);
        let result = estimate_vecm(&data.view(), 1, 1).expect("VECM should succeed");
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_johansen_no_det() {
        let data = make_cointegrated_pair(200);
        let result = johansen_test(&data.view(), 1, JohansenDeterministic::None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_symmetric_eigen_identity() {
        let id = Array2::<f64>::from_diag(&Array1::from_vec(vec![3.0, 2.0, 1.0]));
        let (evals, _evecs) = symmetric_eigen(&id, 100).expect("eigen should succeed");
        let mut sorted = evals.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        assert!((sorted[0] - 3.0).abs() < 1e-10);
        assert!((sorted[1] - 2.0).abs() < 1e-10);
        assert!((sorted[2] - 1.0).abs() < 1e-10);
    }
}
