//! Factor-Adjusted Multiple Testing (Fan, Hall, Yao 2007)
//!
//! When testing H₀: μ_j = 0 for j = 1..p, marginal t-statistics are often
//! strongly correlated due to latent factors (e.g., market factors in finance,
//! batch effects in genomics). Naive multiple testing correction is too
//! conservative or anti-conservative in such settings.
//!
//! The FHY procedure:
//! 1. Compute marginal test statistics T_j for j = 1..p
//! 2. Estimate latent factors via PCA of the statistic matrix (or data matrix)
//! 3. Regress each T_j on the estimated factors
//! 4. Apply BH/BY/Bonferroni to the p-values from the (approximately independent) residuals

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::ndarray::{Array1, Array2};
use std::fmt::Debug;

// ============================================================================
// Configuration and types
// ============================================================================

/// Multiple testing correction method
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TestingMethod {
    /// Benjamini-Hochberg (FDR control under independence / PRDS)
    BenjaminiHochberg,
    /// Benjamini-Yekutieli (FDR control under arbitrary dependence)
    BenjaminiYekutieli,
    /// Bonferroni (family-wise error rate control)
    Bonferroni,
}

/// Configuration for factor-adjusted multiple testing
#[derive(Debug, Clone)]
pub struct FhyConfig {
    /// Number of latent factors to estimate (ignored if detect_n_factors=true)
    pub n_factors: usize,
    /// Multiple testing correction method
    pub method: TestingMethod,
    /// Significance level α
    pub alpha: f64,
    /// Automatically detect number of factors via parallel analysis
    pub detect_n_factors: bool,
}

impl Default for FhyConfig {
    fn default() -> Self {
        FhyConfig {
            n_factors: 3,
            method: TestingMethod::BenjaminiHochberg,
            alpha: 0.05,
            detect_n_factors: true,
        }
    }
}

/// Result of factor-adjusted multiple testing
#[derive(Debug, Clone)]
pub struct FactorAdjustedResult<F: Float> {
    /// Adjusted p-values after factor removal and multiple testing correction
    pub adjusted_pvalues: Vec<F>,
    /// Whether each hypothesis was rejected at level α
    pub rejected: Vec<bool>,
    /// Estimated factor loading matrix A (p × K)
    pub factors: Array2<F>,
    /// Number of factors actually used
    pub n_factors_used: usize,
}

// ============================================================================
// Normal / t-distribution utilities
// ============================================================================

/// Standard normal CDF via rational approximation (Abramowitz & Stegun 26.2.17)
fn normal_cdf<F: Float + FromPrimitive>(x: F) -> F {
    // Use erfc approximation: Φ(x) = 0.5 * erfc(-x / sqrt(2))
    let sqrt2 = F::from_f64(std::f64::consts::SQRT_2).unwrap_or(F::one());
    let half = F::from_f64(0.5).unwrap_or(F::one());
    half * erfc_approx(-x / sqrt2)
}

/// Complementary error function via continued fraction / Horner approximation
fn erfc_approx<F: Float + FromPrimitive>(x: F) -> F {
    // Approximation good to ~6 significant digits for all x
    // erfc(x) ≈ 1 - erf(x), erf(x) via Horner polynomial (Abramowitz & Stegun 7.1.26)
    let one = F::one();
    let zero = F::zero();
    let two = F::from_f64(2.0).unwrap_or(one);

    if x < zero {
        return two - erfc_approx(-x);
    }

    // For large x, use asymptotic: erfc(x) ≈ exp(-x²) / (x * sqrt(π)) * ...
    let x_f64 = x.to_f64().unwrap_or(0.0);
    if x_f64 > 5.0 {
        // Extremely small, treat as 0
        return zero;
    }

    // Horner polynomial approximation (AS 7.1.26)
    let t = one / (one + F::from_f64(0.3275911).unwrap_or(one) * x);
    let a1 = F::from_f64(0.254829592).unwrap_or(one);
    let a2 = F::from_f64(-0.284496736).unwrap_or(one);
    let a3 = F::from_f64(1.421413741).unwrap_or(one);
    let a4 = F::from_f64(-1.453152027).unwrap_or(one);
    let a5 = F::from_f64(1.061405429).unwrap_or(one);
    let poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
    poly * (-(x * x)).exp()
}

/// Two-tailed p-value from standard normal: 2 * Φ(-|z|)
fn normal_two_sided_pvalue<F: Float + FromPrimitive>(z: F) -> F {
    let two = F::from_f64(2.0).unwrap_or(F::one());
    let abs_z = if z < F::zero() { -z } else { z };
    let p = two * (F::one() - normal_cdf(abs_z));
    // Clamp to [0, 1]
    if p < F::zero() {
        F::zero()
    } else if p > F::one() {
        F::one()
    } else {
        p
    }
}

// ============================================================================
// PCA utilities (simple power-iteration / Gram-Schmidt for top K eigenvectors)
// ============================================================================

/// Compute the top K eigenvectors of a symmetric p×p matrix using the
/// Lanczos-like power iteration with deflation (Gram-Schmidt orthogonalization).
///
/// Returns (eigenvalues, eigenvectors as columns of p×K matrix).
fn top_k_eigenvectors<F: Float + FromPrimitive + Clone + Debug>(
    a: &Array2<F>,
    k: usize,
) -> Result<(Array1<F>, Array2<F>)> {
    let p = a.nrows();
    if k == 0 || k > p {
        return Err(StatsError::InvalidArgument(format!(
            "k={} must be in [1, {}]",
            k, p
        )));
    }

    let mut eigenvecs: Vec<Array1<F>> = Vec::with_capacity(k);
    let mut eigenvals: Vec<F> = Vec::with_capacity(k);

    for _ki in 0..k {
        // Initialize random-ish vector (use deterministic initialization)
        let mut v: Array1<F> = Array1::from_vec(
            (0..p)
                .map(|i| F::from_usize(i + 1).unwrap_or(F::one()))
                .collect(),
        );

        // Orthogonalize against already-found eigenvectors
        gram_schmidt_project(&mut v, &eigenvecs);
        normalize_inplace(&mut v);

        // Power iteration
        for _iter in 0..200 {
            let mut av = Array1::zeros(p);
            for i in 0..p {
                for j in 0..p {
                    av[i] = av[i] + a[[i, j]] * v[j];
                }
            }
            gram_schmidt_project(&mut av, &eigenvecs);
            let norm = l2_norm(&av);
            if norm < F::from_f64(1e-14).unwrap_or(F::zero()) {
                break;
            }
            for i in 0..p {
                v[i] = av[i] / norm;
            }
        }

        // Rayleigh quotient = eigenvalue
        let mut av = Array1::zeros(p);
        for i in 0..p {
            for j in 0..p {
                av[i] = av[i] + a[[i, j]] * v[j];
            }
        }
        let eigenval = dot_product(&v, &av);

        eigenvecs.push(v);
        eigenvals.push(eigenval);
    }

    // Build output matrix (p × k)
    let mut evec_mat = Array2::zeros((p, k));
    for (ki, evec) in eigenvecs.iter().enumerate() {
        for i in 0..p {
            evec_mat[[i, ki]] = evec[i];
        }
    }

    Ok((Array1::from_vec(eigenvals), evec_mat))
}

fn gram_schmidt_project<F: Float + Clone>(v: &mut Array1<F>, basis: &[Array1<F>]) {
    for b in basis {
        let proj = dot_product(v, b);
        for i in 0..v.len() {
            v[i] = v[i] - proj * b[i];
        }
    }
}

fn normalize_inplace<F: Float + FromPrimitive>(v: &mut Array1<F>) {
    let norm = l2_norm(v);
    if norm > F::from_f64(1e-14).unwrap_or(F::zero()) {
        for x in v.iter_mut() {
            *x = *x / norm;
        }
    }
}

fn l2_norm<F: Float>(v: &Array1<F>) -> F {
    v.iter().fold(F::zero(), |acc, &x| acc + x * x).sqrt()
}

fn dot_product<F: Float>(a: &Array1<F>, b: &Array1<F>) -> F {
    a.iter()
        .zip(b.iter())
        .fold(F::zero(), |acc, (&x, &y)| acc + x * y)
}

// ============================================================================
// Multiple testing procedures
// ============================================================================

/// Benjamini-Hochberg procedure for FDR control.
///
/// Returns a boolean vector where `true` means rejected at level `alpha`.
pub fn bh_procedure<F: Float + FromPrimitive + Clone + Debug>(
    pvalues: &[F],
    alpha: F,
) -> Vec<bool> {
    let m = pvalues.len();
    if m == 0 {
        return vec![];
    }

    // Create sorted indices by p-value (ascending)
    let mut indexed: Vec<(usize, F)> = pvalues
        .iter()
        .cloned()
        .enumerate()
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let m_f = F::from_usize(m).unwrap_or(F::one());
    let mut rejected = vec![false; m];

    // BH step-up: find largest k such that p_(k) <= k*alpha/m
    let mut max_reject = 0usize;
    for (rank, &(orig_idx, pval)) in indexed.iter().enumerate() {
        let rank_f = F::from_usize(rank + 1).unwrap_or(F::one());
        let threshold = rank_f * alpha / m_f;
        if pval <= threshold {
            max_reject = rank + 1;
        }
        let _ = orig_idx; // suppress unused warning
    }

    // Reject all hypotheses with rank <= max_reject
    for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
        if rank < max_reject {
            rejected[orig_idx] = true;
        }
    }

    rejected
}

/// Benjamini-Yekutieli procedure (BH with harmonic number correction)
fn by_procedure<F: Float + FromPrimitive + Clone + Debug>(
    pvalues: &[F],
    alpha: F,
) -> Vec<bool> {
    let m = pvalues.len();
    // c(m) = Σ_{k=1}^{m} 1/k  (harmonic number)
    let c_m: F = (1..=m)
        .map(|k| F::one() / F::from_usize(k).unwrap_or(F::one()))
        .fold(F::zero(), |acc, v| acc + v);
    bh_procedure(pvalues, alpha / c_m)
}

/// Bonferroni correction
fn bonferroni_procedure<F: Float + FromPrimitive + Clone + Debug>(
    pvalues: &[F],
    alpha: F,
) -> Vec<bool> {
    let m = pvalues.len();
    let m_f = F::from_usize(m).unwrap_or(F::one());
    let threshold = alpha / m_f;
    pvalues.iter().map(|&p| p <= threshold).collect()
}

// ============================================================================
// Factor detection via parallel analysis
// ============================================================================

/// Estimate number of significant factors via a simplified scree plot criterion:
/// keep factors whose eigenvalue exceeds the mean of random matrix eigenvalues.
///
/// For a p×n random matrix, the Marchenko-Pastur distribution gives an upper
/// edge at (1 + sqrt(p/n))². We use this as the threshold.
fn detect_n_factors<F: Float + FromPrimitive + Clone + Debug>(
    eigenvalues: &[F],
    p: usize,
    n: usize,
    max_k: usize,
) -> usize {
    let gamma = if n > 0 {
        (p as f64 / n as f64).sqrt()
    } else {
        1.0
    };
    let mp_upper = F::from_f64((1.0 + gamma) * (1.0 + gamma)).unwrap_or(F::one());

    let mut k = 0;
    for (i, &eval) in eigenvalues.iter().enumerate() {
        if i >= max_k {
            break;
        }
        if eval > mp_upper {
            k = i + 1;
        } else {
            break;
        }
    }
    k.max(1) // always use at least 1 factor
}

// ============================================================================
// Main factor-adjusted test
// ============================================================================

/// Factor-adjusted multiple testing using the FHY (Fan-Hall-Yao) procedure.
///
/// # Arguments
/// * `t_stats` — vector of marginal test statistics T_j for j = 1..p
/// * `n_obs`   — number of observations (used for p-value computation)
/// * `config`  — algorithm configuration
///
/// # Returns
/// `FactorAdjustedResult` with adjusted p-values and rejection decisions
pub fn factor_adjusted_test<F>(
    t_stats: &[F],
    n_obs: usize,
    config: &FhyConfig,
) -> Result<FactorAdjustedResult<F>>
where
    F: Float + FromPrimitive + Clone + Debug,
{
    let p = t_stats.len();
    if p == 0 {
        return Err(StatsError::InvalidArgument(
            "t_stats must be non-empty".to_string(),
        ));
    }
    if n_obs < 2 {
        return Err(StatsError::InvalidArgument(
            "n_obs must be >= 2".to_string(),
        ));
    }

    // Step 1: Compute raw p-values using Normal approximation (valid for large n_obs)
    // p_j = 2 * Φ(-|T_j|) for two-sided test
    let raw_pvalues: Vec<F> = t_stats
        .iter()
        .map(|&t| normal_two_sided_pvalue(t))
        .collect();

    // If n_factors = 0, skip factor adjustment and apply directly
    let max_k = config.n_factors.min(p);
    if max_k == 0 {
        let alpha = F::from_f64(config.alpha).unwrap_or_else(|| F::from_f64(0.05).unwrap_or(F::one()));
        let rejected = match &config.method {
            TestingMethod::BenjaminiHochberg => bh_procedure(&raw_pvalues, alpha),
            TestingMethod::BenjaminiYekutieli => by_procedure(&raw_pvalues, alpha),
            TestingMethod::Bonferroni => bonferroni_procedure(&raw_pvalues, alpha),
        };
        return Ok(FactorAdjustedResult {
            adjusted_pvalues: raw_pvalues,
            rejected,
            factors: Array2::zeros((p, 0)),
            n_factors_used: 0,
        });
    }

    // Step 2: Build correlation matrix of T-statistics
    // (T is a p-vector, so correlation is just the outer product normalized)
    // We treat T as a p × 1 matrix and compute its "correlation" via centering
    let t_arr: Array1<F> = Array1::from_vec(t_stats.to_vec());
    let t_mean = t_arr.iter().cloned().fold(F::zero(), |a, v| a + v)
        / F::from_usize(p).unwrap_or(F::one());
    let t_std = {
        let var = t_arr
            .iter()
            .fold(F::zero(), |acc, &v| acc + (v - t_mean) * (v - t_mean))
            / F::from_usize(p).unwrap_or(F::one());
        var.sqrt()
    };

    // Standardized statistics
    let t_std_safe = if t_std < F::from_f64(1e-14).unwrap_or(F::zero()) {
        F::one()
    } else {
        t_std
    };
    let t_standardized: Vec<F> = t_arr
        .iter()
        .map(|&v| (v - t_mean) / t_std_safe)
        .collect();

    // Build outer product matrix C = t_s * t_s^T / p  (rank-1 correlation proxy)
    // For a single vector, all eigenvectors beyond the 1st are zero.
    // We build a simple p×p correlation matrix using the fact that with one sample,
    // the sample correlation of the T-vector with itself gives the leading factor structure.
    //
    // For a more realistic scenario: C_ij = t_i * t_j / ||t||²
    let t_sq_norm = t_standardized
        .iter()
        .fold(F::zero(), |acc, &v| acc + v * v);
    let t_sq_norm_safe = if t_sq_norm < F::from_f64(1e-14).unwrap_or(F::zero()) {
        F::one()
    } else {
        t_sq_norm
    };

    let mut corr_mat = Array2::<F>::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            corr_mat[[i, j]] = t_standardized[i] * t_standardized[j] / t_sq_norm_safe;
        }
        corr_mat[[i, i]] = corr_mat[[i, i]] + F::one(); // add identity for numerical stability
    }

    // Step 3: Compute top K eigenvectors
    let (eigenvalues, loading_mat) = top_k_eigenvectors(&corr_mat, max_k)?;

    // Step 4: Detect number of factors
    let k_used = if config.detect_n_factors {
        let evals_slice: Vec<F> = eigenvalues.iter().cloned().collect();
        detect_n_factors(&evals_slice, p, n_obs, max_k)
    } else {
        max_k
    };

    // Trim loading matrix to k_used columns
    let a_mat = loading_mat.slice(scirs2_core::ndarray::s![.., ..k_used]).to_owned();

    // Step 5: OLS projection — project T onto factor space and compute residuals
    // β = (A^T A)^{-1} A^T t  (k_used × 1)
    // e = t - A β  (residuals, p-vector)
    //
    // A^T A is k_used × k_used; since A has orthonormal columns, A^T A ≈ I_k
    // So β = A^T t (simplified)
    let t_vec: Array1<F> = Array1::from_vec(t_stats.to_vec());

    let mut beta = Array1::<F>::zeros(k_used);
    for ki in 0..k_used {
        for i in 0..p {
            beta[ki] = beta[ki] + a_mat[[i, ki]] * t_vec[i];
        }
    }

    // Compute A^T A
    let mut ata = Array2::<F>::zeros((k_used, k_used));
    for ki in 0..k_used {
        for kj in 0..k_used {
            for i in 0..p {
                ata[[ki, kj]] = ata[[ki, kj]] + a_mat[[i, ki]] * a_mat[[i, kj]];
            }
        }
    }

    // Solve (A^T A) γ = β for γ using simple Gauss elimination (small k_used)
    let gamma = solve_small_system(&ata, &beta, k_used)?;

    // Residuals: e = t - A γ
    let mut residuals: Vec<F> = t_stats.to_vec();
    for i in 0..p {
        for ki in 0..k_used {
            residuals[i] = residuals[i] - a_mat[[i, ki]] * gamma[ki];
        }
    }

    // Step 6: Compute residual p-values
    let res_std = {
        let mean = residuals.iter().cloned().fold(F::zero(), |a, v| a + v)
            / F::from_usize(p).unwrap_or(F::one());
        let var = residuals
            .iter()
            .fold(F::zero(), |acc, &v| acc + (v - mean) * (v - mean))
            / F::from_usize(p).unwrap_or(F::one());
        var.sqrt()
    };
    let res_std_safe = if res_std < F::from_f64(1e-14).unwrap_or(F::zero()) {
        F::one()
    } else {
        res_std
    };

    let residual_pvalues: Vec<F> = residuals
        .iter()
        .map(|&e| normal_two_sided_pvalue(e / res_std_safe))
        .collect();

    // Step 7: Apply multiple testing correction
    let alpha = F::from_f64(config.alpha).unwrap_or_else(|| F::from_f64(0.05).unwrap_or(F::one()));
    let rejected = match &config.method {
        TestingMethod::BenjaminiHochberg => bh_procedure(&residual_pvalues, alpha),
        TestingMethod::BenjaminiYekutieli => by_procedure(&residual_pvalues, alpha),
        TestingMethod::Bonferroni => bonferroni_procedure(&residual_pvalues, alpha),
    };

    Ok(FactorAdjustedResult {
        adjusted_pvalues: residual_pvalues,
        rejected,
        factors: a_mat,
        n_factors_used: k_used,
    })
}

/// Solve a small k×k linear system A x = b using Gaussian elimination with partial pivoting
fn solve_small_system<F: Float + FromPrimitive + Clone + Debug>(
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

    // Gaussian elimination with partial pivoting
    for col in 0..k {
        // Find pivot
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
            // Near-singular: return zeros
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

    // Back substitution
    let mut x = vec![F::zero(); k];
    for i in (0..k).rev() {
        let mut sum = mat[i][k]; // RHS
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

    #[test]
    fn test_fhy_config_defaults() {
        let cfg = FhyConfig::default();
        assert_eq!(cfg.n_factors, 3);
        assert_eq!(cfg.method, TestingMethod::BenjaminiHochberg);
        assert!((cfg.alpha - 0.05).abs() < 1e-12);
        assert!(cfg.detect_n_factors);
    }

    #[test]
    fn test_bh_all_null() {
        // All p-values ~ 1 → none rejected
        let pvals = vec![0.8f64, 0.9, 0.7, 0.85, 0.95];
        let rejected = bh_procedure(&pvals, 0.05);
        assert!(rejected.iter().all(|&r| !r), "no null should be rejected");
    }

    #[test]
    fn test_bh_some_small() {
        // Very small p-values → some rejected
        let pvals = vec![1e-10f64, 1e-8, 0.8, 0.9, 0.7];
        let rejected = bh_procedure(&pvals, 0.05);
        assert!(rejected[0], "p=1e-10 should be rejected");
        assert!(rejected[1], "p=1e-8 should be rejected");
    }

    #[test]
    fn test_factor_adjusted_output_length() {
        let t_stats = vec![0.5f64, -1.0, 2.0, -0.3, 1.5, -2.5, 0.1, -0.8];
        let config = FhyConfig {
            n_factors: 2,
            detect_n_factors: false,
            ..Default::default()
        };
        let result = factor_adjusted_test(&t_stats, 50, &config).unwrap();
        assert_eq!(result.adjusted_pvalues.len(), t_stats.len());
        assert_eq!(result.rejected.len(), t_stats.len());
    }

    #[test]
    fn test_rejected_consistent_with_pvalues() {
        let t_stats = vec![0.1f64, -0.2, 3.0, -3.5, 0.5, -4.0];
        let config = FhyConfig {
            n_factors: 1,
            detect_n_factors: false,
            alpha: 0.05,
            method: TestingMethod::Bonferroni,
        };
        let result = factor_adjusted_test(&t_stats, 100, &config).unwrap();
        // Check consistency: if rejected[i], then pvalue[i] <= alpha/n
        let n = t_stats.len();
        let threshold = 0.05 / n as f64;
        for i in 0..n {
            if result.rejected[i] {
                assert!(
                    result.adjusted_pvalues[i] <= threshold + 1e-10,
                    "rejected[{}] but pvalue {} > threshold {}",
                    i,
                    result.adjusted_pvalues[i],
                    threshold
                );
            }
        }
    }

    #[test]
    fn test_n_factors_used_bounded() {
        let t_stats: Vec<f64> = (0..10).map(|i| (i as f64 - 5.0) / 2.0).collect();
        let config = FhyConfig {
            n_factors: 3,
            detect_n_factors: true,
            ..Default::default()
        };
        let result = factor_adjusted_test(&t_stats, 30, &config).unwrap();
        assert!(
            result.n_factors_used <= config.n_factors,
            "n_factors_used={} > n_factors={}",
            result.n_factors_used,
            config.n_factors
        );
    }

    #[test]
    fn test_zero_factors_equals_unadjusted_bh() {
        let t_stats = vec![0.5f64, -1.0, 2.5, -0.3, 1.5];
        let config_0 = FhyConfig {
            n_factors: 0,
            detect_n_factors: false,
            alpha: 0.05,
            method: TestingMethod::BenjaminiHochberg,
        };
        let result = factor_adjusted_test(&t_stats, 50, &config_0).unwrap();
        // Raw p-values should match normal_two_sided_pvalue(t_j)
        for (i, &t) in t_stats.iter().enumerate() {
            let expected_p = normal_two_sided_pvalue(t);
            assert!(
                (result.adjusted_pvalues[i] - expected_p).abs() < 1e-10,
                "p-value mismatch at {}: {} vs {}",
                i,
                result.adjusted_pvalues[i],
                expected_p
            );
        }
    }

    #[test]
    fn test_all_pvalues_in_unit_interval() {
        let t_stats: Vec<f64> = vec![-4.0, -2.0, 0.0, 2.0, 4.0, 1.5, -1.5, 0.5];
        let config = FhyConfig::default();
        let result = factor_adjusted_test(&t_stats, 100, &config).unwrap();
        for &p in &result.adjusted_pvalues {
            assert!(p >= 0.0 && p <= 1.0, "p-value out of [0,1]: {}", p);
        }
    }

    #[test]
    fn test_larger_alpha_more_rejections() {
        let t_stats: Vec<f64> = vec![-3.0, -2.5, -2.0, -1.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let config_small = FhyConfig {
            n_factors: 1,
            detect_n_factors: false,
            alpha: 0.01,
            method: TestingMethod::BenjaminiHochberg,
        };
        let config_large = FhyConfig {
            n_factors: 1,
            detect_n_factors: false,
            alpha: 0.2,
            method: TestingMethod::BenjaminiHochberg,
        };
        let r_small = factor_adjusted_test(&t_stats, 100, &config_small).unwrap();
        let r_large = factor_adjusted_test(&t_stats, 100, &config_large).unwrap();
        let n_small: usize = r_small.rejected.iter().filter(|&&r| r).count();
        let n_large: usize = r_large.rejected.iter().filter(|&&r| r).count();
        assert!(
            n_large >= n_small,
            "larger alpha should reject at least as many: {} vs {}",
            n_large,
            n_small
        );
    }
}
