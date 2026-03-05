//! Multiple comparison correction methods
//!
//! When performing many simultaneous hypothesis tests, the probability of at least one
//! false positive (Type I error) increases. Multiple testing corrections adjust p-values
//! or significance thresholds to control for this.
//!
//! ## Methods Provided
//!
//! - **Bonferroni** - Controls the family-wise error rate (FWER) by multiplying each p-value by m
//! - **Holm-Bonferroni** (step-down) - A more powerful step-down variant of Bonferroni
//! - **Hochberg** (step-up) - A step-up variant that is more powerful than Holm under independence
//! - **Benjamini-Hochberg** - Controls the false discovery rate (FDR)
//! - **Benjamini-Yekutieli** - Controls FDR under arbitrary dependence
//! - **Sidak** - Uses the Sidak correction: 1 - (1 - alpha)^(1/m)

use crate::error::{StatsError, StatsResult};

/// Result of a multiple testing correction
#[derive(Debug, Clone)]
pub struct MultipleCorrectionResult {
    /// Adjusted p-values (in the same order as the input)
    pub pvalues_corrected: Vec<f64>,
    /// Boolean array indicating which hypotheses are rejected at the given alpha
    pub reject: Vec<bool>,
    /// The method used for correction
    pub method: String,
    /// The alpha level used
    pub alpha: f64,
}

// ========================================================================
// Bonferroni correction
// ========================================================================

/// Applies Bonferroni correction to a set of p-values.
///
/// The Bonferroni correction is the simplest and most conservative method for
/// controlling the family-wise error rate (FWER). Each p-value is multiplied
/// by the number of tests.
///
/// # Arguments
///
/// * `pvalues` - Slice of raw p-values
/// * `alpha` - Significance level (e.g., 0.05)
///
/// # Returns
///
/// A `MultipleCorrectionResult` with adjusted p-values and rejection decisions.
///
/// # Examples
///
/// ```
/// use scirs2_stats::multiple_testing::bonferroni;
///
/// let pvals = vec![0.01, 0.04, 0.03, 0.005];
/// let result = bonferroni(&pvals, 0.05).expect("Bonferroni correction failed");
/// // p-values are multiplied by 4
/// assert!((result.pvalues_corrected[0] - 0.04).abs() < 1e-10);
/// assert!((result.pvalues_corrected[3] - 0.02).abs() < 1e-10);
/// ```
pub fn bonferroni(pvalues: &[f64], alpha: f64) -> StatsResult<MultipleCorrectionResult> {
    validate_inputs(pvalues, alpha)?;

    let m = pvalues.len() as f64;
    let corrected: Vec<f64> = pvalues.iter().map(|&p| (p * m).min(1.0)).collect();
    let reject: Vec<bool> = corrected.iter().map(|&p| p <= alpha).collect();

    Ok(MultipleCorrectionResult {
        pvalues_corrected: corrected,
        reject,
        method: "Bonferroni".to_string(),
        alpha,
    })
}

// ========================================================================
// Holm-Bonferroni (step-down)
// ========================================================================

/// Applies the Holm-Bonferroni step-down correction.
///
/// The Holm method is uniformly more powerful than Bonferroni while still
/// controlling the FWER. It works by sorting p-values, then multiplying
/// the i-th smallest p-value by (m - i + 1).
///
/// # Arguments
///
/// * `pvalues` - Slice of raw p-values
/// * `alpha` - Significance level
///
/// # Returns
///
/// A `MultipleCorrectionResult` with adjusted p-values and rejection decisions.
///
/// # Examples
///
/// ```
/// use scirs2_stats::multiple_testing::holm_bonferroni;
///
/// let pvals = vec![0.01, 0.04, 0.03, 0.005];
/// let result = holm_bonferroni(&pvals, 0.05).expect("Holm correction failed");
/// // The smallest p-value (0.005) is multiplied by 4, second (0.01) by 3, etc.
/// assert!(result.reject[3]); // 0.005 * 4 = 0.02 <= 0.05
/// ```
pub fn holm_bonferroni(pvalues: &[f64], alpha: f64) -> StatsResult<MultipleCorrectionResult> {
    validate_inputs(pvalues, alpha)?;

    let m = pvalues.len();

    // Sort indices by p-value
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| {
        pvalues[a]
            .partial_cmp(&pvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut corrected = vec![0.0_f64; m];

    // Step-down: enforce monotonicity (adjusted values must be non-decreasing)
    let mut max_so_far = 0.0_f64;
    for (rank, &idx) in order.iter().enumerate() {
        let multiplier = (m - rank) as f64;
        let adjusted = (pvalues[idx] * multiplier).min(1.0);
        max_so_far = max_so_far.max(adjusted);
        corrected[idx] = max_so_far;
    }

    let reject: Vec<bool> = corrected.iter().map(|&p| p <= alpha).collect();

    Ok(MultipleCorrectionResult {
        pvalues_corrected: corrected,
        reject,
        method: "Holm-Bonferroni".to_string(),
        alpha,
    })
}

// ========================================================================
// Hochberg (step-up)
// ========================================================================

/// Applies the Hochberg step-up correction.
///
/// The Hochberg procedure is a step-up version of the Holm method. It is valid
/// under certain positive dependence conditions (PRDS). The i-th largest p-value
/// is multiplied by (m - i + 1), enforcing monotonicity from the top.
///
/// # Arguments
///
/// * `pvalues` - Slice of raw p-values
/// * `alpha` - Significance level
///
/// # Returns
///
/// A `MultipleCorrectionResult` with adjusted p-values and rejection decisions.
///
/// # Examples
///
/// ```
/// use scirs2_stats::multiple_testing::hochberg;
///
/// let pvals = vec![0.01, 0.04, 0.03, 0.005];
/// let result = hochberg(&pvals, 0.05).expect("Hochberg correction failed");
/// assert!(result.reject[3]); // smallest p-value should be rejected
/// ```
pub fn hochberg(pvalues: &[f64], alpha: f64) -> StatsResult<MultipleCorrectionResult> {
    validate_inputs(pvalues, alpha)?;

    let m = pvalues.len();

    // Sort indices by p-value (ascending)
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| {
        pvalues[a]
            .partial_cmp(&pvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut corrected = vec![0.0_f64; m];

    // Step-up: go from largest to smallest, enforce monotonicity downward
    let mut min_so_far = 1.0_f64;
    for rank in (0..m).rev() {
        let idx = order[rank];
        let multiplier = (m - rank) as f64;
        let adjusted = (pvalues[idx] * multiplier).min(1.0);
        min_so_far = min_so_far.min(adjusted);
        corrected[idx] = min_so_far;
    }

    let reject: Vec<bool> = corrected.iter().map(|&p| p <= alpha).collect();

    Ok(MultipleCorrectionResult {
        pvalues_corrected: corrected,
        reject,
        method: "Hochberg".to_string(),
        alpha,
    })
}

// ========================================================================
// Benjamini-Hochberg (FDR control)
// ========================================================================

/// Applies the Benjamini-Hochberg procedure for FDR control.
///
/// The Benjamini-Hochberg (BH) procedure controls the false discovery rate (FDR)
/// rather than the family-wise error rate. FDR is the expected proportion of false
/// discoveries among all discoveries. The BH procedure is less conservative than
/// FWER-controlling methods and is widely used in genomics, neuroimaging, etc.
///
/// # Arguments
///
/// * `pvalues` - Slice of raw p-values
/// * `alpha` - Target FDR level (e.g., 0.05)
///
/// # Returns
///
/// A `MultipleCorrectionResult` with adjusted p-values (q-values) and rejection decisions.
///
/// # Examples
///
/// ```
/// use scirs2_stats::multiple_testing::benjamini_hochberg;
///
/// let pvals = vec![0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216];
/// let result = benjamini_hochberg(&pvals, 0.05).expect("BH correction failed");
/// // The first few p-values should still be significant after BH correction
/// assert!(result.reject[0]);
/// ```
pub fn benjamini_hochberg(pvalues: &[f64], alpha: f64) -> StatsResult<MultipleCorrectionResult> {
    validate_inputs(pvalues, alpha)?;

    let m = pvalues.len();
    let mf = m as f64;

    // Sort indices by p-value (ascending)
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| {
        pvalues[a]
            .partial_cmp(&pvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut corrected = vec![0.0_f64; m];

    // Step-up: start from the largest p-value
    let mut min_so_far = 1.0_f64;
    for rank in (0..m).rev() {
        let idx = order[rank];
        let rank_1based = (rank + 1) as f64;
        let adjusted = (pvalues[idx] * mf / rank_1based).min(1.0);
        min_so_far = min_so_far.min(adjusted);
        corrected[idx] = min_so_far;
    }

    let reject: Vec<bool> = corrected.iter().map(|&p| p <= alpha).collect();

    Ok(MultipleCorrectionResult {
        pvalues_corrected: corrected,
        reject,
        method: "Benjamini-Hochberg".to_string(),
        alpha,
    })
}

// ========================================================================
// Benjamini-Yekutieli
// ========================================================================

/// Applies the Benjamini-Yekutieli procedure for FDR control under arbitrary dependence.
///
/// The BY procedure is a modification of BH that controls the FDR under arbitrary
/// dependence between the test statistics. It is more conservative than BH because
/// it divides by an additional factor c(m) = sum(1/i, i=1..m).
///
/// # Arguments
///
/// * `pvalues` - Slice of raw p-values
/// * `alpha` - Target FDR level
///
/// # Returns
///
/// A `MultipleCorrectionResult` with adjusted p-values and rejection decisions.
///
/// # Examples
///
/// ```
/// use scirs2_stats::multiple_testing::benjamini_yekutieli;
///
/// let pvals = vec![0.001, 0.008, 0.039, 0.041, 0.06];
/// let result = benjamini_yekutieli(&pvals, 0.05).expect("BY correction failed");
/// assert!(result.reject[0]); // Smallest p-value should still be significant
/// ```
pub fn benjamini_yekutieli(pvalues: &[f64], alpha: f64) -> StatsResult<MultipleCorrectionResult> {
    validate_inputs(pvalues, alpha)?;

    let m = pvalues.len();
    let mf = m as f64;

    // c(m) = sum(1/i for i=1..m)
    let cm: f64 = (1..=m).map(|i| 1.0 / i as f64).sum();

    // Sort indices by p-value (ascending)
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| {
        pvalues[a]
            .partial_cmp(&pvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut corrected = vec![0.0_f64; m];

    // Step-up with c(m) correction
    let mut min_so_far = 1.0_f64;
    for rank in (0..m).rev() {
        let idx = order[rank];
        let rank_1based = (rank + 1) as f64;
        let adjusted = (pvalues[idx] * mf * cm / rank_1based).min(1.0);
        min_so_far = min_so_far.min(adjusted);
        corrected[idx] = min_so_far;
    }

    let reject: Vec<bool> = corrected.iter().map(|&p| p <= alpha).collect();

    Ok(MultipleCorrectionResult {
        pvalues_corrected: corrected,
        reject,
        method: "Benjamini-Yekutieli".to_string(),
        alpha,
    })
}

// ========================================================================
// Sidak correction
// ========================================================================

/// Applies the Sidak correction.
///
/// The Sidak correction is based on the formula: adjusted_p = 1 - (1 - p)^m.
/// For independent tests, it is slightly less conservative than Bonferroni.
///
/// # Arguments
///
/// * `pvalues` - Slice of raw p-values
/// * `alpha` - Significance level
///
/// # Returns
///
/// A `MultipleCorrectionResult` with adjusted p-values and rejection decisions.
///
/// # Examples
///
/// ```
/// use scirs2_stats::multiple_testing::sidak;
///
/// let pvals = vec![0.01, 0.04, 0.03, 0.005];
/// let result = sidak(&pvals, 0.05).expect("Sidak correction failed");
/// // Sidak is slightly less conservative than Bonferroni
/// assert!(result.pvalues_corrected[3] < 0.02); // 1 - (1 - 0.005)^4
/// ```
pub fn sidak(pvalues: &[f64], alpha: f64) -> StatsResult<MultipleCorrectionResult> {
    validate_inputs(pvalues, alpha)?;

    let m = pvalues.len() as f64;
    let corrected: Vec<f64> = pvalues
        .iter()
        .map(|&p| {
            // 1 - (1 - p)^m, but handle edge cases
            if p >= 1.0 {
                1.0
            } else if p <= 0.0 {
                0.0
            } else {
                (1.0 - (1.0 - p).powf(m)).min(1.0)
            }
        })
        .collect();

    let reject: Vec<bool> = corrected.iter().map(|&p| p <= alpha).collect();

    Ok(MultipleCorrectionResult {
        pvalues_corrected: corrected,
        reject,
        method: "Sidak".to_string(),
        alpha,
    })
}

// ========================================================================
// Convenience: apply correction by name
// ========================================================================

/// Applies a multiple testing correction by method name.
///
/// This is a convenience function that dispatches to the appropriate correction
/// method based on the given name string.
///
/// # Arguments
///
/// * `pvalues` - Slice of raw p-values
/// * `alpha` - Significance level
/// * `method` - Method name: "bonferroni", "holm", "hochberg", "fdr_bh", "fdr_by", "sidak"
///
/// # Returns
///
/// A `MultipleCorrectionResult` with adjusted p-values and rejection decisions.
pub fn multipletests(
    pvalues: &[f64],
    alpha: f64,
    method: &str,
) -> StatsResult<MultipleCorrectionResult> {
    match method {
        "bonferroni" => bonferroni(pvalues, alpha),
        "holm" | "holm-bonferroni" => holm_bonferroni(pvalues, alpha),
        "hochberg" => hochberg(pvalues, alpha),
        "fdr_bh" | "benjamini-hochberg" | "bh" => benjamini_hochberg(pvalues, alpha),
        "fdr_by" | "benjamini-yekutieli" | "by" => benjamini_yekutieli(pvalues, alpha),
        "sidak" => sidak(pvalues, alpha),
        _ => Err(StatsError::InvalidArgument(format!(
            "Unknown correction method '{}'. Valid methods: bonferroni, holm, hochberg, fdr_bh, fdr_by, sidak",
            method
        ))),
    }
}

// ========================================================================
// Input validation helper
// ========================================================================

fn validate_inputs(pvalues: &[f64], alpha: f64) -> StatsResult<()> {
    if pvalues.is_empty() {
        return Err(StatsError::InvalidArgument(
            "p-values array cannot be empty".to_string(),
        ));
    }
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(StatsError::InvalidArgument(format!(
            "alpha must be in (0, 1), got {}",
            alpha
        )));
    }
    for (i, &p) in pvalues.iter().enumerate() {
        if p < 0.0 || p > 1.0 {
            return Err(StatsError::InvalidArgument(format!(
                "p-value at index {} is {}, must be in [0, 1]",
                i, p
            )));
        }
    }
    Ok(())
}

// ========================================================================
// Tests
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bonferroni_basic() {
        let pvals = vec![0.01, 0.04, 0.03, 0.005];
        let result = bonferroni(&pvals, 0.05).expect("Bonferroni failed");

        assert_eq!(result.pvalues_corrected.len(), 4);
        assert!((result.pvalues_corrected[0] - 0.04).abs() < 1e-10);
        assert!((result.pvalues_corrected[1] - 0.16).abs() < 1e-10);
        assert!((result.pvalues_corrected[2] - 0.12).abs() < 1e-10);
        assert!((result.pvalues_corrected[3] - 0.02).abs() < 1e-10);

        assert!(result.reject[0]); // 0.04 <= 0.05
        assert!(!result.reject[1]); // 0.16 > 0.05
        assert!(!result.reject[2]); // 0.12 > 0.05
        assert!(result.reject[3]); // 0.02 <= 0.05
    }

    #[test]
    fn test_bonferroni_clamped() {
        let pvals = vec![0.5, 0.6, 0.7];
        let result = bonferroni(&pvals, 0.05).expect("Bonferroni failed");
        // 0.5 * 3 = 1.5 => clamped to 1.0
        assert!((result.pvalues_corrected[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_holm_bonferroni_basic() {
        let pvals = vec![0.01, 0.04, 0.03, 0.005];
        let result = holm_bonferroni(&pvals, 0.05).expect("Holm failed");

        assert_eq!(result.pvalues_corrected.len(), 4);
        // Sorted order: [0.005(3), 0.01(0), 0.03(2), 0.04(1)]
        // Rank 0: 0.005 * 4 = 0.02
        // Rank 1: 0.01 * 3 = 0.03, max(0.02, 0.03) = 0.03
        // Rank 2: 0.03 * 2 = 0.06, max(0.03, 0.06) = 0.06
        // Rank 3: 0.04 * 1 = 0.04, max(0.06, 0.04) = 0.06
        assert!((result.pvalues_corrected[3] - 0.02).abs() < 1e-10);
        assert!((result.pvalues_corrected[0] - 0.03).abs() < 1e-10);
        assert!(result.reject[3]); // 0.02 <= 0.05
        assert!(result.reject[0]); // 0.03 <= 0.05
    }

    #[test]
    fn test_hochberg_basic() {
        let pvals = vec![0.01, 0.04, 0.03, 0.005];
        let result = hochberg(&pvals, 0.05).expect("Hochberg failed");

        assert_eq!(result.pvalues_corrected.len(), 4);
        // Should be at least as powerful as Bonferroni
        let bonf_result = bonferroni(&pvals, 0.05).expect("Bonferroni failed");
        for i in 0..pvals.len() {
            assert!(result.pvalues_corrected[i] <= bonf_result.pvalues_corrected[i] + 1e-10);
        }
    }

    #[test]
    fn test_benjamini_hochberg_basic() {
        let pvals = vec![
            0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216,
        ];
        let result = benjamini_hochberg(&pvals, 0.05).expect("BH failed");

        assert_eq!(result.pvalues_corrected.len(), 10);
        // All corrected p-values should be in [0, 1]
        for &p in &result.pvalues_corrected {
            assert!(p >= 0.0 && p <= 1.0);
        }
        // The first p-value should still be significant
        assert!(result.reject[0]);
    }

    #[test]
    fn test_benjamini_hochberg_monotonicity() {
        let pvals = vec![0.01, 0.02, 0.03, 0.04, 0.05];
        let result = benjamini_hochberg(&pvals, 0.05).expect("BH failed");

        // Adjusted p-values should be non-decreasing when sorted by original p-value
        let mut order: Vec<usize> = (0..pvals.len()).collect();
        order.sort_by(|&a, &b| {
            pvals[a]
                .partial_cmp(&pvals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for i in 1..order.len() {
            assert!(
                result.pvalues_corrected[order[i]]
                    >= result.pvalues_corrected[order[i - 1]] - 1e-10,
                "Monotonicity violation at rank {}",
                i
            );
        }
    }

    #[test]
    fn test_benjamini_yekutieli_basic() {
        let pvals = vec![0.001, 0.008, 0.039, 0.041, 0.06];
        let result = benjamini_yekutieli(&pvals, 0.05).expect("BY failed");

        // BY should be more conservative than BH
        let bh_result = benjamini_hochberg(&pvals, 0.05).expect("BH failed");
        for i in 0..pvals.len() {
            assert!(result.pvalues_corrected[i] >= bh_result.pvalues_corrected[i] - 1e-10);
        }
    }

    #[test]
    fn test_sidak_basic() {
        let pvals = vec![0.01, 0.04, 0.03, 0.005];
        let result = sidak(&pvals, 0.05).expect("Sidak failed");

        // Sidak should be slightly less conservative than Bonferroni
        let bonf_result = bonferroni(&pvals, 0.05).expect("Bonferroni failed");
        for i in 0..pvals.len() {
            assert!(result.pvalues_corrected[i] <= bonf_result.pvalues_corrected[i] + 1e-10);
        }
    }

    #[test]
    fn test_sidak_values() {
        let pvals = vec![0.005];
        let result = sidak(&pvals, 0.05).expect("Sidak failed");
        // For m=1, Sidak(p) = 1 - (1-p)^1 = p
        assert!((result.pvalues_corrected[0] - 0.005).abs() < 1e-10);

        let pvals2 = vec![0.01, 0.01];
        let result2 = sidak(&pvals2, 0.05).expect("Sidak failed");
        // 1 - (1 - 0.01)^2 = 1 - 0.99^2 = 1 - 0.9801 = 0.0199
        assert!((result2.pvalues_corrected[0] - 0.0199).abs() < 1e-4);
    }

    #[test]
    fn test_multipletests_dispatch() {
        let pvals = vec![0.01, 0.04, 0.03, 0.005];

        let r1 = multipletests(&pvals, 0.05, "bonferroni").expect("dispatch failed");
        assert_eq!(r1.method, "Bonferroni");

        let r2 = multipletests(&pvals, 0.05, "holm").expect("dispatch failed");
        assert_eq!(r2.method, "Holm-Bonferroni");

        let r3 = multipletests(&pvals, 0.05, "hochberg").expect("dispatch failed");
        assert_eq!(r3.method, "Hochberg");

        let r4 = multipletests(&pvals, 0.05, "fdr_bh").expect("dispatch failed");
        assert_eq!(r4.method, "Benjamini-Hochberg");

        let r5 = multipletests(&pvals, 0.05, "fdr_by").expect("dispatch failed");
        assert_eq!(r5.method, "Benjamini-Yekutieli");

        let r6 = multipletests(&pvals, 0.05, "sidak").expect("dispatch failed");
        assert_eq!(r6.method, "Sidak");

        let r7 = multipletests(&pvals, 0.05, "unknown_method");
        assert!(r7.is_err());
    }

    #[test]
    fn test_empty_input() {
        let empty: Vec<f64> = vec![];
        assert!(bonferroni(&empty, 0.05).is_err());
        assert!(holm_bonferroni(&empty, 0.05).is_err());
        assert!(hochberg(&empty, 0.05).is_err());
        assert!(benjamini_hochberg(&empty, 0.05).is_err());
        assert!(benjamini_yekutieli(&empty, 0.05).is_err());
        assert!(sidak(&empty, 0.05).is_err());
    }

    #[test]
    fn test_invalid_alpha() {
        let pvals = vec![0.01, 0.05];
        assert!(bonferroni(&pvals, 0.0).is_err());
        assert!(bonferroni(&pvals, 1.0).is_err());
        assert!(bonferroni(&pvals, -0.1).is_err());
    }

    #[test]
    fn test_invalid_pvalues() {
        let pvals = vec![0.01, 1.5];
        assert!(bonferroni(&pvals, 0.05).is_err());

        let pvals2 = vec![-0.01, 0.05];
        assert!(bonferroni(&pvals2, 0.05).is_err());
    }

    #[test]
    fn test_single_pvalue() {
        let pvals = vec![0.03];
        let result = bonferroni(&pvals, 0.05).expect("single pval failed");
        assert!((result.pvalues_corrected[0] - 0.03).abs() < 1e-10);
        assert!(result.reject[0]);
    }

    #[test]
    fn test_all_significant() {
        let pvals = vec![0.001, 0.002, 0.003];
        let result = bonferroni(&pvals, 0.05).expect("all sig failed");
        // 0.001 * 3 = 0.003, 0.002 * 3 = 0.006, 0.003 * 3 = 0.009
        assert!(result.reject.iter().all(|&r| r));
    }

    #[test]
    fn test_none_significant() {
        let pvals = vec![0.5, 0.6, 0.7, 0.8, 0.9];
        let result = bonferroni(&pvals, 0.05).expect("none sig failed");
        assert!(result.reject.iter().all(|&r| !r));
    }

    #[test]
    fn test_holm_vs_bonferroni_power() {
        // Holm should reject at least as many as Bonferroni
        let pvals = vec![0.001, 0.01, 0.04, 0.06, 0.10];
        let bonf = bonferroni(&pvals, 0.05).expect("Bonferroni failed");
        let holm = holm_bonferroni(&pvals, 0.05).expect("Holm failed");

        let bonf_count: usize = bonf.reject.iter().filter(|&&r| r).count();
        let holm_count: usize = holm.reject.iter().filter(|&&r| r).count();
        assert!(holm_count >= bonf_count);
    }

    #[test]
    fn test_bh_less_conservative_than_bonferroni() {
        let pvals = vec![0.001, 0.01, 0.04, 0.06, 0.10];
        let bonf = bonferroni(&pvals, 0.05).expect("Bonferroni failed");
        let bh = benjamini_hochberg(&pvals, 0.05).expect("BH failed");

        let bonf_count: usize = bonf.reject.iter().filter(|&&r| r).count();
        let bh_count: usize = bh.reject.iter().filter(|&&r| r).count();
        // BH (FDR) should reject at least as many as Bonferroni (FWER)
        assert!(bh_count >= bonf_count);
    }
}
