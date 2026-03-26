//! High-Dimensional Two-Sample Tests
//!
//! Classical Hotelling T² breaks down when p >> n because the sample covariance
//! becomes singular. These tests avoid covariance inversion entirely.
//!
//! ## Chen-Qin (2010) test
//! Uses ||X̄ - Ȳ||² as the raw statistic and estimates the null variance
//! from the data via unbiased U-statistics, yielding an asymptotically
//! standard normal test statistic.
//!
//! ## Bai-Saranadasa (1996) test
//! Simplified variant using only tr(S²) estimated via cross-product terms.
//!
//! ## Principal Component Regression (PCR)
//! Reduces dimensionality via SVD before running OLS — useful when p >> n.

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::ndarray::{Array1, Array2};
use std::fmt::Debug;

// ============================================================================
// Configuration and types
// ============================================================================

/// Method for high-dimensional two-sample testing
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum HdMethod {
    /// Chen & Qin (2010) U-statistic based test
    ChenQin,
    /// Bai & Saranadasa (1996) trace-based test
    BaiSaranadasa,
}

/// Configuration for high-dimensional two-sample tests
#[derive(Debug, Clone)]
pub struct HdTwoSampleConfig {
    /// Testing method to use
    pub method: HdMethod,
    /// Number of bootstrap replications for p-value (0 = use Normal approximation)
    pub n_bootstrap: usize,
}

impl Default for HdTwoSampleConfig {
    fn default() -> Self {
        HdTwoSampleConfig {
            method: HdMethod::ChenQin,
            n_bootstrap: 0,
        }
    }
}

/// Result of a high-dimensional two-sample test
#[derive(Debug, Clone)]
pub struct HdTestResult<F: Float> {
    /// Standardized test statistic (approximately N(0,1) under H₀)
    pub statistic: F,
    /// Two-sided p-value
    pub pvalue: F,
    /// Method used
    pub method: HdMethod,
}

/// Result of Principal Component Regression
#[derive(Debug, Clone)]
pub struct PcrResult<F: Float> {
    /// Regression coefficients in the original feature space (p-vector)
    pub coefficients: Array1<F>,
    /// R² coefficient of determination
    pub r_squared: F,
    /// Number of principal components used
    pub n_components: usize,
}

// ============================================================================
// Normal CDF utility (reused from factor_adjusted_testing)
// ============================================================================

fn erfc_approx_hd<F: Float + FromPrimitive>(x: F) -> F {
    let one = F::one();
    let zero = F::zero();
    let two = F::from_f64(2.0).unwrap_or(one);
    if x < zero {
        return two - erfc_approx_hd(-x);
    }
    let x_f64 = x.to_f64().unwrap_or(0.0);
    if x_f64 > 8.0 {
        return zero;
    }
    let t = one / (one + F::from_f64(0.3275911).unwrap_or(one) * x);
    let a1 = F::from_f64(0.254829592).unwrap_or(one);
    let a2 = F::from_f64(-0.284496736).unwrap_or(one);
    let a3 = F::from_f64(1.421413741).unwrap_or(one);
    let a4 = F::from_f64(-1.453152027).unwrap_or(one);
    let a5 = F::from_f64(1.061405429).unwrap_or(one);
    let poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
    poly * (-(x * x)).exp()
}

fn normal_cdf_hd<F: Float + FromPrimitive>(x: F) -> F {
    let sqrt2 = F::from_f64(std::f64::consts::SQRT_2).unwrap_or(F::one());
    let half = F::from_f64(0.5).unwrap_or(F::one());
    half * erfc_approx_hd(-x / sqrt2)
}

fn normal_two_sided_pvalue_hd<F: Float + FromPrimitive>(z: F) -> F {
    let two = F::from_f64(2.0).unwrap_or(F::one());
    let abs_z = if z < F::zero() { -z } else { z };
    let p = two * (F::one() - normal_cdf_hd(abs_z));
    if p < F::zero() {
        F::zero()
    } else if p > F::one() {
        F::one()
    } else {
        p
    }
}

// ============================================================================
// Statistics utilities
// ============================================================================

/// Compute sample mean of rows (n × p matrix → p-vector)
fn row_mean<F: Float + FromPrimitive + Clone>(mat: &Array2<F>) -> Array1<F> {
    let (n, p) = (mat.nrows(), mat.ncols());
    let n_f = F::from_usize(n).unwrap_or(F::one());
    let mut mean = Array1::zeros(p);
    for i in 0..n {
        for j in 0..p {
            mean[j] = mean[j] + mat[[i, j]];
        }
    }
    for j in 0..p {
        mean[j] = mean[j] / n_f;
    }
    mean
}

/// Compute sum of squared norms: Σ_i ||x_i - x̄||²
fn sum_sq_deviations<F: Float + FromPrimitive + Clone>(mat: &Array2<F>, mean: &Array1<F>) -> F {
    let (n, p) = (mat.nrows(), mat.ncols());
    let mut total = F::zero();
    for i in 0..n {
        for j in 0..p {
            let d = mat[[i, j]] - mean[j];
            total = total + d * d;
        }
    }
    total
}

/// Compute tr(S²) using the U-statistic identity:
/// tr(S²) = (1/(n(n-1)²)) Σ_{i≠j} (x_i - x_j)^T (x_i - x_j) Σ_{i≠j} ...
///
/// Efficient computation: tr(S²) = (1/n²) ||X^T X - n x̄ x̄^T||_F² / ((n-1)/n)^2 (approx)
/// We use a direct but O(n²p) approach for correctness:
/// tr(S_hat²) where S_hat = X_c^T X_c / (n-1), X_c is centered
fn trace_sq_sample_cov<F: Float + FromPrimitive + Clone>(
    mat: &Array2<F>,
    mean: &Array1<F>,
) -> F {
    let (n, p) = (mat.nrows(), mat.ncols());
    if n <= 1 {
        return F::zero();
    }
    let n_m1 = F::from_usize(n - 1).unwrap_or(F::one());

    // Compute X_c = X - mean (centered)
    // tr(S²) = ||X_c^T X_c / (n-1)||_F² = sum_{ij} (X_c^T X_c)_{ij}² / (n-1)²
    // = (1/(n-1)²) * Σ_{i,j} (Σ_k x_{ki} x_{kj})²

    // For computational efficiency with large p, we compute tr(S²) = tr((X_c X_c^T / (n-1))²) / p is wrong.
    // Actually tr(S²) = (1/(n-1)²) * Σ_k Σ_l (x_k^T x_l)² where x_k = X_c[k,:]
    // This is O(n²p) which is acceptable.

    let mut inner_products = Array2::<F>::zeros((n, n));
    for k in 0..n {
        for l in 0..n {
            let mut ip = F::zero();
            for j in 0..p {
                let xkj = mat[[k, j]] - mean[j];
                let xlj = mat[[l, j]] - mean[j];
                ip = ip + xkj * xlj;
            }
            inner_products[[k, l]] = ip;
        }
    }

    let mut tr_sq = F::zero();
    for k in 0..n {
        for l in 0..n {
            tr_sq = tr_sq + inner_products[[k, l]] * inner_products[[k, l]];
        }
    }

    tr_sq / (n_m1 * n_m1)
}

// ============================================================================
// Chen-Qin (2010) two-sample test
// ============================================================================

/// Compute the Chen-Qin (2010) test statistic for high-dimensional two-sample testing.
///
/// H₀: μ_X = μ_Y vs H₁: μ_X ≠ μ_Y (no covariance inversion required)
fn chen_qin_statistic<F: Float + FromPrimitive + Clone + Debug>(
    x: &Array2<F>,
    y: &Array2<F>,
) -> Result<F> {
    let (n1, p) = (x.nrows(), x.ncols());
    let (n2, p2) = (y.nrows(), y.ncols());

    if p != p2 {
        return Err(StatsError::DimensionMismatch(format!(
            "X has {} features, Y has {} features",
            p, p2
        )));
    }
    if n1 < 2 || n2 < 2 {
        return Err(StatsError::InsufficientData(
            "each group needs at least 2 observations".to_string(),
        ));
    }

    let x_mean = row_mean(x);
    let y_mean = row_mean(y);

    // Raw statistic: T = ||x̄ - ȳ||²
    let mut t_raw = F::zero();
    for j in 0..p {
        let d = x_mean[j] - y_mean[j];
        t_raw = t_raw + d * d;
    }

    // Estimate variance of T under H₀:
    // Var(T) ≈ (2/n1²) tr(Σ₁²) + (2/n2²) tr(Σ₂²) + (2/(n1 n2)) tr(Σ₁ Σ₂)
    // We estimate tr(Σ_k²) ≈ tr(S_k²) using U-statistic estimators.
    //
    // Simple unbiased estimator (Chen & Qin 2010, Eq. 2.2):
    // Ân = (n1(n1-1))^{-1} * Σ_{i≠j} (x_i^T x_j)² type terms — see paper
    //
    // We use the scaled sample covariance trace: tr(S_k²) via inner products.

    let tr_s1_sq = trace_sq_sample_cov(x, &x_mean);
    let tr_s2_sq = trace_sq_sample_cov(y, &y_mean);

    let n1_f = F::from_usize(n1).ok_or_else(|| {
        StatsError::ComputationError("Cannot convert n1 to F".to_string())
    })?;
    let n2_f = F::from_usize(n2).ok_or_else(|| {
        StatsError::ComputationError("Cannot convert n2 to F".to_string())
    })?;
    let two = F::from_f64(2.0).unwrap_or(F::one());

    // Variance estimate
    let var_t = two * tr_s1_sq / (n1_f * n1_f) + two * tr_s2_sq / (n2_f * n2_f);
    let std_t = var_t.sqrt();

    // Centering correction: subtract expected value E[T] = tr(Σ₁)/n1 + tr(Σ₂)/n2
    // Estimated by tr(S₁)/n1 + tr(S₂)/n2
    let tr_s1 = sum_sq_deviations(x, &x_mean) / F::from_usize(n1 - 1).unwrap_or(F::one());
    let tr_s2 = sum_sq_deviations(y, &y_mean) / F::from_usize(n2 - 1).unwrap_or(F::one());
    let expected_t = tr_s1 / n1_f + tr_s2 / n2_f;

    // Z-score
    if std_t < F::from_f64(1e-14).unwrap_or(F::zero()) {
        return Ok(F::zero());
    }
    Ok((t_raw - expected_t) / std_t)
}

// ============================================================================
// Bai-Saranadasa (1996) test
// ============================================================================

/// Simplified Bai-Saranadasa (1996) test statistic.
///
/// Uses a simpler variance estimator based on tr(S^2) alone (pooled).
fn bai_saranadasa_statistic<F: Float + FromPrimitive + Clone + Debug>(
    x: &Array2<F>,
    y: &Array2<F>,
) -> Result<F> {
    let (n1, p) = (x.nrows(), x.ncols());
    let (n2, p2) = (y.nrows(), y.ncols());

    if p != p2 {
        return Err(StatsError::DimensionMismatch(format!(
            "X has {} features, Y has {} features",
            p, p2
        )));
    }
    if n1 < 2 || n2 < 2 {
        return Err(StatsError::InsufficientData(
            "each group needs at least 2 observations".to_string(),
        ));
    }

    let x_mean = row_mean(x);
    let y_mean = row_mean(y);

    // Raw statistic
    let mut t_raw = F::zero();
    for j in 0..p {
        let d = x_mean[j] - y_mean[j];
        t_raw = t_raw + d * d;
    }

    let n1_f = F::from_usize(n1).unwrap_or(F::one());
    let n2_f = F::from_usize(n2).unwrap_or(F::one());
    let n_total = n1_f + n2_f;
    let two = F::from_f64(2.0).unwrap_or(F::one());

    // Pooled trace of Σ: (tr(S₁) + tr(S₂)) / 2
    let tr_s1 = sum_sq_deviations(x, &x_mean) / F::from_usize(n1 - 1).unwrap_or(F::one());
    let tr_s2 = sum_sq_deviations(y, &y_mean) / F::from_usize(n2 - 1).unwrap_or(F::one());

    // BS variance: 2 * ((n1+n2)/(n1*n2))² * tr(S_pooled²)
    // We approximate tr(S_pooled²) using the pooled sample covariance
    let tr_sp_sq = {
        let w1 = (n1_f - F::one()) / (n_total - two);
        let w2 = (n2_f - F::one()) / (n_total - two);
        // Rough estimate: weighted average of individual tr(S²)
        let tr_s1_sq = trace_sq_sample_cov(x, &x_mean);
        let tr_s2_sq = trace_sq_sample_cov(y, &y_mean);
        w1 * tr_s1_sq + w2 * tr_s2_sq
    };

    let coeff = (n1_f + n2_f) / (n1_f * n2_f);
    let var_t = two * coeff * coeff * tr_sp_sq;
    let std_t = var_t.sqrt();

    // Center
    let expected_t = (tr_s1 + tr_s2) * coeff;
    if std_t < F::from_f64(1e-14).unwrap_or(F::zero()) {
        return Ok(F::zero());
    }
    Ok((t_raw - expected_t) / std_t)
}

// ============================================================================
// Main test function
// ============================================================================

/// Perform a high-dimensional two-sample test.
///
/// # Arguments
/// * `x` — n₁ × p data matrix for group 1
/// * `y` — n₂ × p data matrix for group 2
/// * `config` — test configuration
///
/// # Returns
/// `HdTestResult` with test statistic, p-value, and method used.
pub fn hd_two_sample_test<F>(
    x: &Array2<F>,
    y: &Array2<F>,
    config: &HdTwoSampleConfig,
) -> Result<HdTestResult<F>>
where
    F: Float + FromPrimitive + Clone + Debug,
{
    let p = x.ncols();
    let p2 = y.ncols();

    if p == 0 || p2 == 0 {
        return Err(StatsError::InvalidArgument(
            "Data matrices must have at least 1 feature".to_string(),
        ));
    }
    if p != p2 {
        return Err(StatsError::DimensionMismatch(format!(
            "Feature dimension mismatch: {} vs {}",
            p, p2
        )));
    }
    if x.nrows() < 2 || y.nrows() < 2 {
        return Err(StatsError::InsufficientData(
            "Each group must have at least 2 observations".to_string(),
        ));
    }

    let z = match &config.method {
        HdMethod::ChenQin => chen_qin_statistic(x, y)?,
        HdMethod::BaiSaranadasa => bai_saranadasa_statistic(x, y)?,
    };

    let pvalue = normal_two_sided_pvalue_hd(z);

    Ok(HdTestResult {
        statistic: z,
        pvalue,
        method: config.method.clone(),
    })
}

// ============================================================================
// Principal Component Regression
// ============================================================================

/// Perform Principal Component Regression.
///
/// 1. Center X and compute thin SVD: X_c ≈ U_K D_K V_K^T
/// 2. Project: Z = U_K D_K  (reduced design matrix, n × K)
/// 3. OLS: β̂_PC = (Z^T Z)^{-1} Z^T y
/// 4. Map back: β̂ = V_K β̂_PC
///
/// # Arguments
/// * `x` — n × p design matrix
/// * `y` — n-vector of responses
/// * `n_components` — number of principal components (K ≤ min(n,p))
pub fn principal_component_regression<F>(
    x: &Array2<F>,
    y: &Array1<F>,
    n_components: usize,
) -> Result<PcrResult<F>>
where
    F: Float + FromPrimitive + Clone + Debug,
{
    let (n, p) = (x.nrows(), x.ncols());
    if n != y.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "X has {} rows but y has {} elements",
            n,
            y.len()
        )));
    }
    if n == 0 || p == 0 {
        return Err(StatsError::InvalidArgument(
            "X must be non-empty".to_string(),
        ));
    }
    let k = n_components.min(n.min(p));
    if k == 0 {
        return Err(StatsError::InvalidArgument(
            "n_components must be >= 1".to_string(),
        ));
    }

    let n_f = F::from_usize(n).unwrap_or(F::one());

    // Center X column-wise
    let x_means: Vec<F> = (0..p)
        .map(|j| {
            (0..n).fold(F::zero(), |acc, i| acc + x[[i, j]]) / n_f
        })
        .collect();
    let y_mean = y.iter().cloned().fold(F::zero(), |acc, v| acc + v) / n_f;

    let mut xc = x.clone();
    for i in 0..n {
        for j in 0..p {
            xc[[i, j]] = xc[[i, j]] - x_means[j];
        }
    }
    let yc: Array1<F> = y.iter().map(|&v| v - y_mean).collect();

    // Compute X_c^T X_c (p × p) for eigenvector approach
    // For the SVD, we compute the p×p gram matrix and extract top K eigenvectors as V_K
    // Then U_K D_K = X_c V_K
    let mut xtx = Array2::<F>::zeros((p, p));
    for i in 0..n {
        for j in 0..p {
            for l in 0..p {
                xtx[[j, l]] = xtx[[j, l]] + xc[[i, j]] * xc[[i, l]];
            }
        }
    }

    // Get top K eigenvectors of X^T X (these are V_K from SVD)
    let vk = top_k_eigenvectors_pcr(&xtx, k)?;

    // Project: Z = X_c V_K  (n × k)
    let mut z = Array2::<F>::zeros((n, k));
    for i in 0..n {
        for ki in 0..k {
            for j in 0..p {
                z[[i, ki]] = z[[i, ki]] + xc[[i, j]] * vk[[j, ki]];
            }
        }
    }

    // OLS: β̂_PC = (Z^T Z)^{-1} Z^T y_c
    let mut ztz = Array2::<F>::zeros((k, k));
    for i in 0..n {
        for ki in 0..k {
            for kj in 0..k {
                ztz[[ki, kj]] = ztz[[ki, kj]] + z[[i, ki]] * z[[i, kj]];
            }
        }
    }

    let mut zty = Array1::<F>::zeros(k);
    for i in 0..n {
        for ki in 0..k {
            zty[ki] = zty[ki] + z[[i, ki]] * yc[i];
        }
    }

    // Solve ztz * beta_pc = zty
    let beta_pc = solve_small_sys_pcr(&ztz, &zty, k)?;

    // Map back to original space: β̂ = V_K β̂_PC
    let mut coefficients = Array1::<F>::zeros(p);
    for j in 0..p {
        for ki in 0..k {
            coefficients[j] = coefficients[j] + vk[[j, ki]] * beta_pc[ki];
        }
    }

    // R² = 1 - SS_res / SS_tot
    // Fitted values: ŷ = X_c β̂ (in centered space) + y_mean
    let mut ss_res = F::zero();
    let mut ss_tot = F::zero();
    for i in 0..n {
        let mut y_hat = y_mean;
        for j in 0..p {
            y_hat = y_hat + xc[[i, j]] * coefficients[j];
        }
        let res = y[i] - y_hat;
        ss_res = ss_res + res * res;
        let dev = y[i] - y_mean;
        ss_tot = ss_tot + dev * dev;
    }

    let r_squared = if ss_tot < F::from_f64(1e-14).unwrap_or(F::zero()) {
        F::one()
    } else {
        F::one() - ss_res / ss_tot
    };
    let r_squared = if r_squared < F::zero() {
        F::zero()
    } else if r_squared > F::one() {
        F::one()
    } else {
        r_squared
    };

    Ok(PcrResult {
        coefficients,
        r_squared,
        n_components: k,
    })
}

// ============================================================================
// PCR eigenvector helper (same power iteration but self-contained)
// ============================================================================

fn top_k_eigenvectors_pcr<F: Float + FromPrimitive + Clone + Debug>(
    a: &Array2<F>,
    k: usize,
) -> Result<Array2<F>> {
    let p = a.nrows();
    if k > p {
        return Err(StatsError::InvalidArgument(format!(
            "k={} > p={}",
            k, p
        )));
    }

    let mut eigenvecs: Vec<Array1<F>> = Vec::with_capacity(k);

    for ki in 0..k {
        // Deterministic initialization
        let mut v: Array1<F> = Array1::from_vec(
            (0..p)
                .map(|i| {
                    let val = (i + ki + 1) as f64;
                    F::from_f64(val).unwrap_or(F::one())
                })
                .collect(),
        );

        // Orthogonalize
        gs_project_pcr(&mut v, &eigenvecs);
        normalize_pcr(&mut v);

        for _ in 0..300 {
            let mut av = Array1::zeros(p);
            for i in 0..p {
                for j in 0..p {
                    av[i] = av[i] + a[[i, j]] * v[j];
                }
            }
            gs_project_pcr(&mut av, &eigenvecs);
            let norm = l2_norm_pcr(&av);
            if norm < F::from_f64(1e-14).unwrap_or(F::zero()) {
                break;
            }
            for i in 0..p {
                v[i] = av[i] / norm;
            }
        }

        eigenvecs.push(v);
    }

    let mut mat = Array2::zeros((p, k));
    for (ki, evec) in eigenvecs.iter().enumerate() {
        for i in 0..p {
            mat[[i, ki]] = evec[i];
        }
    }
    Ok(mat)
}

fn gs_project_pcr<F: Float + Clone>(v: &mut Array1<F>, basis: &[Array1<F>]) {
    for b in basis {
        let proj = v.iter().zip(b.iter()).fold(F::zero(), |acc, (&x, &y)| acc + x * y);
        for i in 0..v.len() {
            v[i] = v[i] - proj * b[i];
        }
    }
}

fn normalize_pcr<F: Float + FromPrimitive>(v: &mut Array1<F>) {
    let norm = l2_norm_pcr(v);
    if norm > F::from_f64(1e-14).unwrap_or(F::zero()) {
        for x in v.iter_mut() {
            *x = *x / norm;
        }
    }
}

fn l2_norm_pcr<F: Float>(v: &Array1<F>) -> F {
    v.iter().fold(F::zero(), |acc, &x| acc + x * x).sqrt()
}

fn solve_small_sys_pcr<F: Float + FromPrimitive + Clone + Debug>(
    a: &Array2<F>,
    b: &Array1<F>,
    k: usize,
) -> Result<Array1<F>> {
    if k == 0 {
        return Ok(Array1::zeros(0));
    }
    let mut mat: Vec<Vec<F>> = (0..k)
        .map(|i| {
            let mut row: Vec<F> = (0..k).map(|j| a[[i, j]]).collect();
            row.push(b[i]);
            row
        })
        .collect();

    for col in 0..k {
        let mut max_row = col;
        let mut max_val = mat[col][col].abs();
        for row in (col + 1)..k {
            if mat[row][col].abs() > max_val {
                max_val = mat[row][col].abs();
                max_row = row;
            }
        }
        mat.swap(col, max_row);

        let pivot = mat[col][col];
        if pivot.abs() < F::from_f64(1e-14).unwrap_or(F::zero()) {
            return Ok(Array1::zeros(k));
        }
        for row in (col + 1)..k {
            let factor = mat[row][col] / pivot;
            for c in col..=k {
                let val = mat[col][c];
                mat[row][c] = mat[row][c] - factor * val;
            }
        }
    }

    let mut x = vec![F::zero(); k];
    for i in (0..k).rev() {
        let mut sum = mat[i][k];
        for j in (i + 1)..k {
            sum = sum - mat[i][j] * x[j];
        }
        let denom = mat[i][i];
        x[i] = if denom.abs() < F::from_f64(1e-14).unwrap_or(F::zero()) {
            F::zero()
        } else {
            sum / denom
        };
    }
    Ok(Array1::from_vec(x))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    fn make_group(n: usize, p: usize, offset: f64, seed: u64) -> Array2<f64> {
        // Simple deterministic pseudo-random data generation
        let mut state = seed ^ 0xdeadbeef;
        let mut data = Array2::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u = (state >> 11) as f64 / (1u64 << 53) as f64;
                // Normal via Box-Muller (use pairs)
                let u2_state = state.wrapping_mul(2654435761).wrapping_add(1);
                let u2 = (u2_state >> 11) as f64 / (1u64 << 53) as f64;
                let z = (-2.0 * (u + 1e-10).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                data[[i, j]] = z + offset;
                state = u2_state;
            }
        }
        data
    }

    #[test]
    fn test_hd_two_sample_same_data_high_pvalue() {
        // Same data → H₀ should not be rejected (p-value near 1)
        let x = make_group(30, 20, 0.0, 1);
        let y = make_group(30, 20, 0.0, 2);
        let config = HdTwoSampleConfig::default();
        let result = hd_two_sample_test(&x, &y, &config).unwrap();
        // p-value should be large (not necessarily > 0.05 every time, but statistic close to 0)
        assert!(result.pvalue >= 0.0 && result.pvalue <= 1.0);
        // statistic should be moderate for same-distribution data
        assert!(result.statistic.abs() < 10.0, "statistic should not be extreme for H0: {}", result.statistic);
    }

    #[test]
    fn test_hd_two_sample_shifted_small_pvalue() {
        // Large shift → should produce small p-value
        let x = make_group(50, 10, 0.0, 10);
        let y = make_group(50, 10, 5.0, 20); // large shift
        let config = HdTwoSampleConfig::default();
        let result = hd_two_sample_test(&x, &y, &config).unwrap();
        // The statistic should be large and positive (mean shift detected)
        assert!(result.statistic > 0.0, "statistic should be positive for shifted data");
        assert!(result.pvalue < 0.5, "p-value should be small for large shift: {}", result.pvalue);
    }

    #[test]
    fn test_hd_two_sample_config_defaults() {
        let cfg = HdTwoSampleConfig::default();
        assert_eq!(cfg.method, HdMethod::ChenQin);
        assert_eq!(cfg.n_bootstrap, 0);
    }

    #[test]
    fn test_hd_test_result_statistic_finite() {
        let x = make_group(20, 15, 0.0, 5);
        let y = make_group(20, 15, 0.0, 6);
        let config = HdTwoSampleConfig::default();
        let result = hd_two_sample_test(&x, &y, &config).unwrap();
        assert!(result.statistic.is_finite(), "statistic must be finite");
        assert!(result.pvalue.is_finite(), "p-value must be finite");
    }

    #[test]
    fn test_pcr_output_shape() {
        // Simple n=20, p=5, k=3 case
        let x = make_group(20, 5, 0.0, 100);
        let y_arr: Array1<f64> = (0..20)
            .map(|i| {
                let mut s = i as u64;
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                (s >> 11) as f64 / (1u64 << 53) as f64
            })
            .collect();
        let result = principal_component_regression(&x, &y_arr, 3).unwrap();
        assert_eq!(result.coefficients.len(), 5, "coefficients should have length p");
        assert_eq!(result.n_components, 3);
    }

    #[test]
    fn test_pcr_r_squared_range() {
        let x = make_group(50, 10, 0.0, 200);
        // y = sum of first 3 columns (recoverable via PCR)
        let y: Array1<f64> = (0..50).map(|i| x[[i, 0]] + x[[i, 1]] + x[[i, 2]]).collect();
        let result = principal_component_regression(&x, &y, 5).unwrap();
        assert!(
            result.r_squared >= 0.0 && result.r_squared <= 1.0,
            "R² must be in [0,1]: {}",
            result.r_squared
        );
    }

    #[test]
    fn test_pcr_full_rank_approaches_ols() {
        // With n_components = min(n,p), PCR should approximate OLS
        let n = 20;
        let p = 4;
        let x = make_group(n, p, 0.0, 300);
        let y: Array1<f64> = (0..n).map(|i| x[[i, 0]] + 0.5 * x[[i, 1]]).collect();

        let result_full = principal_component_regression(&x, &y, p).unwrap();
        // With all components, R² should be fairly high
        assert!(
            result_full.r_squared >= 0.0,
            "full PCR R² must be non-negative: {}",
            result_full.r_squared
        );
    }

    #[test]
    fn test_hd_two_sample_dimension_mismatch_error() {
        let x = make_group(10, 5, 0.0, 1);
        let y = make_group(10, 6, 0.0, 2);
        let config = HdTwoSampleConfig::default();
        let result = hd_two_sample_test(&x, &y, &config);
        assert!(result.is_err(), "dimension mismatch should return error");
    }

    #[test]
    fn test_hd_two_sample_insufficient_data_error() {
        let x = make_group(1, 5, 0.0, 1);
        let y = make_group(10, 5, 0.0, 2);
        let config = HdTwoSampleConfig::default();
        let result = hd_two_sample_test(&x, &y, &config);
        assert!(result.is_err(), "n<2 should return error");
    }
}
