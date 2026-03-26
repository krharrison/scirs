//! Econometrics Methods
//!
//! This module provides econometric estimators for causal inference:
//!
//! - [`instrumental_variables`]: Two-Stage Least Squares (2SLS) with diagnostics
//! - [`did`]: Difference-in-Differences (classic, regression-based, staggered)
//! - [`synthetic_control`]: Synthetic Control Method (Abadie et al., 2010)
//!
//! ## Common Regression Utilities
//!
//! The module provides shared OLS and robust standard error helpers used
//! across all estimators.

pub mod did;
pub mod instrumental_variables;
pub mod synthetic_control;

// Re-exports
pub use did::{
    DidEstimator, DidResult, EventStudyCoefficient, EventStudyEstimator, EventStudyResult,
    StaggeredAttGt, StaggeredDidEstimator, StaggeredDidResult,
};
pub use instrumental_variables::{IvDiagnostics, IvResult, TwoStageLeastSquares};
pub use synthetic_control::{PlaceboResult, SyntheticControlEstimator, SyntheticControlResult};

// ---------------------------------------------------------------------------
// Common regression utilities
// ---------------------------------------------------------------------------

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// OLS via normal equations: beta = (X'X)^{-1} X'y.
///
/// Returns (beta, residuals, (X'X)^{-1}).
pub(crate) fn ols_fit(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
) -> StatsResult<(Array1<f64>, Array1<f64>, Array2<f64>)> {
    let n = x.nrows();
    let k = x.ncols();
    if n < k {
        return Err(StatsError::InsufficientData(format!(
            "OLS requires at least {k} observations, got {n}"
        )));
    }
    if y.len() != n {
        return Err(StatsError::DimensionMismatch(format!(
            "y length {} != X rows {n}",
            y.len()
        )));
    }
    let xtx = x.t().dot(x);
    let xty = x.t().dot(y);
    let xtx_inv = cholesky_invert(&xtx.view())?;
    let beta = xtx_inv.dot(&xty);
    let fitted = x.dot(&beta);
    let residuals = y.to_owned() - fitted;
    Ok((beta, residuals, xtx_inv))
}

/// HC1 heteroscedasticity-robust variance-covariance matrix.
///
/// V_HC1 = n/(n-k) * (X'X)^{-1} X' diag(e^2) X (X'X)^{-1}
pub(crate) fn robust_vcov_hc1(
    x: &ArrayView2<f64>,
    residuals: &Array1<f64>,
    xtx_inv: &Array2<f64>,
) -> Array2<f64> {
    let n = x.nrows();
    let k = x.ncols();
    let scale = n as f64 / (n as f64 - k as f64).max(1.0);
    let mut meat = Array2::<f64>::zeros((k, k));
    for i in 0..n {
        let e2 = residuals[i] * residuals[i];
        for r in 0..k {
            for c in 0..k {
                meat[[r, c]] += e2 * x[[i, r]] * x[[i, c]];
            }
        }
    }
    let sandwich = xtx_inv.dot(&meat).dot(xtx_inv);
    sandwich.mapv(|v| v * scale)
}

/// Clustered variance-covariance matrix (Liang-Zeger sandwich).
///
/// V_cluster = (G/(G-1)) * (n-1)/(n-k) * (X'X)^{-1} B (X'X)^{-1}
/// where B = sum_g (X_g' e_g)(X_g' e_g)'.
pub(crate) fn clustered_vcov(
    x: &ArrayView2<f64>,
    residuals: &Array1<f64>,
    cluster_ids: &[usize],
    xtx_inv: &Array2<f64>,
) -> Array2<f64> {
    let n = x.nrows();
    let k = x.ncols();

    // Group rows by cluster
    let mut cluster_map: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, &cid) in cluster_ids.iter().enumerate() {
        cluster_map.entry(cid).or_default().push(i);
    }
    let g = cluster_map.len();

    let mut meat = Array2::<f64>::zeros((k, k));
    for (_cid, rows) in &cluster_map {
        // x_g' e_g
        let mut xe = Array1::<f64>::zeros(k);
        for &i in rows {
            for j in 0..k {
                xe[j] += x[[i, j]] * residuals[i];
            }
        }
        // outer product
        for r in 0..k {
            for c in 0..k {
                meat[[r, c]] += xe[r] * xe[c];
            }
        }
    }

    let scale = (g as f64 / (g as f64 - 1.0).max(1.0)) * ((n - 1) as f64 / (n - k) as f64).max(1.0);
    let sandwich = xtx_inv.dot(&meat).dot(xtx_inv);
    sandwich.mapv(|v| v * scale)
}

/// Cholesky inversion for symmetric positive-definite matrix.
pub(crate) fn cholesky_invert(a: &ArrayView2<f64>) -> StatsResult<Array2<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(StatsError::DimensionMismatch(
            "Matrix must be square for inversion".into(),
        ));
    }
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for p in 0..j {
                s -= l[[i, p]] * l[[j, p]];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(StatsError::ComputationError(
                        "Matrix is not positive definite; check for multicollinearity".into(),
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
            for p in j..i {
                s += l[[i, p]] * linv[[p, j]];
            }
            linv[[i, j]] = -s / l[[i, i]];
        }
    }
    Ok(linv.t().dot(&linv))
}

/// Two-sided p-value from t-distribution.
pub(crate) fn t_dist_p_value(t: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return 1.0;
    }
    if df > 300.0 {
        return normal_p_value(t);
    }
    let x = df / (df + t * t);
    regularized_incomplete_beta(x, df / 2.0, 0.5).clamp(0.0, 1.0)
}

/// Two-sided p-value from standard normal.
pub(crate) fn normal_p_value(z: f64) -> f64 {
    2.0 * (1.0 - normal_cdf(z.abs()))
}

/// Standard normal CDF.
pub(crate) fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function (Abramowitz & Stegun 7.1.26).
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0
        - (0.254829592
            + (-0.284496736 + (1.421413741 + (-1.453152027 + 1.061405429 * t) * t) * t) * t)
            * t
            * (-x * x).exp();
    if x >= 0.0 {
        y
    } else {
        -y
    }
}

/// Regularized incomplete beta function I_x(a,b).
pub(crate) fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(1.0 - x, b, a);
    }
    let log_cf =
        (a * x.ln() + b * (1.0 - x).ln() - ln_gamma(a) - ln_gamma(b) + ln_gamma(a + b)).exp() / a;
    log_cf * beta_cf(x, a, b)
}

fn beta_cf(x: f64, a: f64, b: f64) -> f64 {
    let fpmin = 1e-300_f64;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0_f64;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < fpmin {
        d = fpmin;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=200_i32 {
        let mf = m as f64;
        let aa = mf * (b - mf) * x / ((qam + 2.0 * mf) * (a + 2.0 * mf));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa2 = -(a + mf) * (qab + mf) * x / ((a + 2.0 * mf) * (qap + 2.0 * mf));
        d = 1.0 + aa2 * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa2 / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 3e-15 {
            break;
        }
    }
    h
}

/// ln(Gamma(x)) via Lanczos approximation.
pub(crate) fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.13857_109_526_572_012,
        9.984_369_578_019_572e-6,
        1.5056_327_351_493_116e-7,
    ];
    if x < 0.5 {
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let z = x - 1.0;
        let mut s = C[0];
        for (i, &ci) in C[1..].iter().enumerate() {
            s += ci / (z + (i as f64) + 1.0);
        }
        let t = z + G + 0.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + s.ln()
    }
}

/// Chi-squared upper-tail p-value: P(X^2 >= x).
pub(crate) fn chi2_p_value(x: f64, df: usize) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let a = df as f64 / 2.0;
    1.0 - regularized_gamma_lower(a, x / 2.0)
}

/// F-distribution upper-tail p-value: P(F >= x).
pub(crate) fn f_dist_p_value(x: f64, df1: usize, df2: usize) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let d1 = df1 as f64;
    let d2 = df2 as f64;
    let z = d2 / (d2 + d1 * x);
    regularized_incomplete_beta(z, d2 / 2.0, d1 / 2.0).clamp(0.0, 1.0)
}

/// Regularized lower incomplete gamma P(a, x).
fn regularized_gamma_lower(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        let mut ap = a;
        let mut del = 1.0 / a;
        let mut sum = del;
        for _ in 0..200 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 3e-15 {
                break;
            }
        }
        sum * (-x + a * x.ln() - ln_gamma(a)).exp()
    } else {
        1.0 - regularized_gamma_upper_cf(a, x)
    }
}

fn regularized_gamma_upper_cf(a: f64, x: f64) -> f64 {
    let fpmin = 1e-300_f64;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / fpmin;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=200_i64 {
        let an = -(i as f64) * ((i as f64) - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = b + an / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 3e-15 {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// Approximate t-distribution critical value for one-tailed alpha via Newton-Raphson.
pub(crate) fn t_critical(alpha: f64, df: usize) -> f64 {
    let df_f = df as f64;
    let mut t = 2.0_f64;
    for _ in 0..50 {
        let p = t_dist_p_value(t, df_f);
        let target = 2.0 * alpha;
        let err = p - target;
        let delta = 1e-6;
        let dp = (t_dist_p_value(t + delta, df_f) - p) / delta;
        if dp.abs() < 1e-15 {
            break;
        }
        t -= err / dp;
        if err.abs() < 1e-10 {
            break;
        }
    }
    t.max(0.0)
}

/// Normal quantile (inverse CDF) via rational approximation (Abramowitz & Stegun 26.2.23).
pub(crate) fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }
    let sign = if p < 0.5 { -1.0 } else { 1.0 };
    let pp = if p < 0.5 { p } else { 1.0 - p };
    let t = (-2.0 * pp.ln()).sqrt();
    const C0: f64 = 2.515517;
    const C1: f64 = 0.802853;
    const C2: f64 = 0.010328;
    const D1: f64 = 1.432788;
    const D2: f64 = 0.189269;
    const D3: f64 = 0.001308;
    let z = t - (C0 + C1 * t + C2 * t * t) / (1.0 + D1 * t + D2 * t * t + D3 * t * t * t);
    sign * z
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_ols_fit_exact() {
        let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
        let (beta, resid, _) = ols_fit(&x.view(), &y.view()).expect("OLS should succeed");
        assert!((beta[1] - 2.0).abs() < 1e-6);
        assert!(resid.iter().all(|&r| r.abs() < 1e-6));
    }

    #[test]
    fn test_robust_vcov_hc1() {
        let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]];
        let y = array![1.1, 2.0, 2.9, 4.0];
        let (_, resid, xtx_inv) = ols_fit(&x.view(), &y.view()).expect("OLS");
        let vcov = robust_vcov_hc1(&x.view(), &resid, &xtx_inv);
        assert!(vcov[[0, 0]] >= 0.0);
        assert!(vcov[[1, 1]] >= 0.0);
    }

    #[test]
    fn test_normal_p_value_symmetry() {
        let p1 = normal_p_value(1.96);
        assert!((p1 - 0.05).abs() < 0.01);
        let p2 = normal_p_value(-1.96);
        assert!((p1 - p2).abs() < 1e-10);
    }

    #[test]
    fn test_f_dist_p_value() {
        // F(1, 100) = 3.94 should give p ~ 0.05
        let p = f_dist_p_value(3.94, 1, 100);
        assert!(
            p > 0.03 && p < 0.07,
            "F-test p-value should be ~0.05, got {}",
            p
        );
    }
}
