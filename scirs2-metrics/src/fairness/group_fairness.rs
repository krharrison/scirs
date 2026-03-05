//! Enhanced Group Fairness Metrics
//!
//! Standalone functions for evaluating fairness across demographic groups.
//! These supplement the existing fairness module with multi-group support,
//! threshold-based analysis, and comprehensive fairness auditing.
//!
//! # Metrics
//!
//! - **Demographic Parity Ratio**: Ratio version (vs. existing difference version)
//! - **Equal Opportunity Ratio**: Ratio of true positive rates
//! - **Equalized Odds**: Combined FPR and TPR comparison
//! - **Disparate Impact Ratio**: Four-fifths rule compliance check
//! - **Predictive Parity**: Positive predictive value comparison
//! - **Treatment Equality**: Ratio of FN to FP across groups
//! - **Multi-group Fairness**: Extensions to more than two groups
//!
//! # Examples
//!
//! ```
//! use scirs2_metrics::fairness::group_fairness::{
//!     demographic_parity_ratio, equal_opportunity_ratio,
//!     equalized_odds_ratio, disparate_impact_check,
//! };
//!
//! let y_true = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
//! let y_pred = vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0];
//! let groups = vec![0, 0, 0, 0, 1, 1, 1, 1]; // Two groups
//!
//! let dp = demographic_parity_ratio(&y_pred, &groups).expect("Failed");
//! let eo = equal_opportunity_ratio(&y_true, &y_pred, &groups).expect("Failed");
//! let eod = equalized_odds_ratio(&y_true, &y_pred, &groups).expect("Failed");
//! let di = disparate_impact_check(&y_pred, &groups, 0.8).expect("Failed");
//! ```

use crate::error::{MetricsError, Result};
use std::collections::HashMap;

/// Result of a fairness audit on a model's predictions.
#[derive(Debug, Clone)]
pub struct FairnessAuditResult {
    /// Demographic parity ratio (min group rate / max group rate)
    pub demographic_parity_ratio: f64,
    /// Equal opportunity ratio (min TPR / max TPR)
    pub equal_opportunity_ratio: f64,
    /// Equalized odds ratio (min of FPR ratio and TPR ratio)
    pub equalized_odds_ratio: f64,
    /// Disparate impact ratio (min selection rate / max selection rate)
    pub disparate_impact_ratio: f64,
    /// Whether the four-fifths rule is satisfied (DI >= 0.8)
    pub four_fifths_rule_satisfied: bool,
    /// Per-group positive prediction rates
    pub group_selection_rates: HashMap<usize, f64>,
    /// Per-group true positive rates
    pub group_tpr: HashMap<usize, f64>,
    /// Per-group false positive rates
    pub group_fpr: HashMap<usize, f64>,
    /// Per-group positive predictive values
    pub group_ppv: HashMap<usize, f64>,
}

/// Computes the demographic parity ratio across groups.
///
/// Demographic parity ratio = min(P(Y=1|G=g)) / max(P(Y=1|G=g))
///
/// A ratio of 1.0 indicates perfect demographic parity.
///
/// # Arguments
///
/// * `y_pred` - Predicted labels (values > 0 are positive).
/// * `groups` - Group membership labels for each sample.
///
/// # Returns
///
/// The demographic parity ratio in [0, 1].
pub fn demographic_parity_ratio(y_pred: &[f64], groups: &[usize]) -> Result<f64> {
    if y_pred.len() != groups.len() {
        return Err(MetricsError::InvalidInput(
            "y_pred and groups must have the same length".to_string(),
        ));
    }
    if y_pred.is_empty() {
        return Err(MetricsError::InvalidInput(
            "inputs must not be empty".to_string(),
        ));
    }

    let rates = group_positive_rates(y_pred, groups)?;

    let min_rate = rates.values().copied().fold(f64::INFINITY, f64::min);
    let max_rate = rates.values().copied().fold(f64::NEG_INFINITY, f64::max);

    if max_rate <= 0.0 {
        // No group has positive predictions
        return Ok(1.0);
    }

    Ok(min_rate / max_rate)
}

/// Computes the equal opportunity ratio across groups.
///
/// Equal opportunity focuses on the true positive rate (TPR = recall):
///   ratio = min(TPR_g) / max(TPR_g)
///
/// A ratio of 1.0 means all groups have equal TPR.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels (values > 0 are positive).
/// * `y_pred` - Predicted labels.
/// * `groups` - Group membership labels.
///
/// # Returns
///
/// The equal opportunity ratio in [0, 1].
pub fn equal_opportunity_ratio(y_true: &[f64], y_pred: &[f64], groups: &[usize]) -> Result<f64> {
    validate_ternary_inputs(y_true, y_pred, groups)?;

    let tpr_map = group_true_positive_rates(y_true, y_pred, groups)?;

    if tpr_map.is_empty() {
        return Err(MetricsError::InvalidInput(
            "No groups with positive samples found".to_string(),
        ));
    }

    let min_tpr = tpr_map.values().copied().fold(f64::INFINITY, f64::min);
    let max_tpr = tpr_map.values().copied().fold(f64::NEG_INFINITY, f64::max);

    if max_tpr <= 0.0 {
        return Ok(1.0);
    }

    Ok(min_tpr / max_tpr)
}

/// Computes the equalized odds ratio across groups.
///
/// Equalized odds considers both TPR and FPR:
///   ratio = min(min(TPR_g/max_TPR, FPR_g_ratio))
///
/// A ratio of 1.0 means both TPR and FPR are equal across groups.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels.
/// * `y_pred` - Predicted labels.
/// * `groups` - Group membership labels.
///
/// # Returns
///
/// The equalized odds ratio in [0, 1].
pub fn equalized_odds_ratio(y_true: &[f64], y_pred: &[f64], groups: &[usize]) -> Result<f64> {
    validate_ternary_inputs(y_true, y_pred, groups)?;

    let tpr_map = group_true_positive_rates(y_true, y_pred, groups)?;
    let fpr_map = group_false_positive_rates(y_true, y_pred, groups)?;

    let tpr_ratio = if !tpr_map.is_empty() {
        let min_tpr = tpr_map.values().copied().fold(f64::INFINITY, f64::min);
        let max_tpr = tpr_map.values().copied().fold(f64::NEG_INFINITY, f64::max);
        if max_tpr <= 0.0 {
            1.0
        } else {
            min_tpr / max_tpr
        }
    } else {
        1.0
    };

    let fpr_ratio = if !fpr_map.is_empty() {
        let min_fpr = fpr_map.values().copied().fold(f64::INFINITY, f64::min);
        let max_fpr = fpr_map.values().copied().fold(f64::NEG_INFINITY, f64::max);
        if max_fpr <= 0.0 {
            1.0
        } else {
            min_fpr / max_fpr
        }
    } else {
        1.0
    };

    // Return the minimum of the two ratios
    Ok(tpr_ratio.min(fpr_ratio))
}

/// Checks whether the disparate impact four-fifths rule is satisfied.
///
/// The four-fifths rule states that the selection rate for any group should
/// be at least 80% of the highest selection rate.
///
/// # Arguments
///
/// * `y_pred` - Predicted labels (values > 0 are positive).
/// * `groups` - Group membership labels.
/// * `threshold` - The minimum acceptable ratio (default 0.8 for four-fifths rule).
///
/// # Returns
///
/// A [`DisparateImpactResult`] with the ratio and compliance status.
pub fn disparate_impact_check(
    y_pred: &[f64],
    groups: &[usize],
    threshold: f64,
) -> Result<DisparateImpactResult> {
    let ratio = demographic_parity_ratio(y_pred, groups)?;
    let rates = group_positive_rates(y_pred, groups)?;

    Ok(DisparateImpactResult {
        ratio,
        passes_threshold: ratio >= threshold,
        threshold,
        group_rates: rates,
    })
}

/// Result of a disparate impact check.
#[derive(Debug, Clone)]
pub struct DisparateImpactResult {
    /// The disparate impact ratio (min rate / max rate)
    pub ratio: f64,
    /// Whether the ratio meets or exceeds the threshold
    pub passes_threshold: bool,
    /// The threshold used
    pub threshold: f64,
    /// Per-group positive prediction rates
    pub group_rates: HashMap<usize, f64>,
}

/// Computes predictive parity difference across groups.
///
/// Predictive parity is satisfied when the Positive Predictive Value (PPV)
/// is equal across groups.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels.
/// * `y_pred` - Predicted labels.
/// * `groups` - Group membership labels.
///
/// # Returns
///
/// The max PPV difference across any pair of groups.
pub fn predictive_parity_difference(
    y_true: &[f64],
    y_pred: &[f64],
    groups: &[usize],
) -> Result<f64> {
    validate_ternary_inputs(y_true, y_pred, groups)?;

    let ppv_map = group_positive_predictive_values(y_true, y_pred, groups)?;

    if ppv_map.len() < 2 {
        return Ok(0.0);
    }

    let values: Vec<f64> = ppv_map.values().copied().collect();
    let mut max_diff = 0.0_f64;
    for i in 0..values.len() {
        for j in i + 1..values.len() {
            let diff = (values[i] - values[j]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    Ok(max_diff)
}

/// Computes treatment equality ratio across groups.
///
/// Treatment equality measures the ratio of false negatives to false positives
/// for each group. Equal treatment means this ratio is the same across groups.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels.
/// * `y_pred` - Predicted labels.
/// * `groups` - Group membership labels.
///
/// # Returns
///
/// The maximum difference in FN/FP ratios across groups.
pub fn treatment_equality_difference(
    y_true: &[f64],
    y_pred: &[f64],
    groups: &[usize],
) -> Result<f64> {
    validate_ternary_inputs(y_true, y_pred, groups)?;

    let mut group_fn_fp: HashMap<usize, (usize, usize)> = HashMap::new();

    for i in 0..y_true.len() {
        let g = groups[i];
        let entry = group_fn_fp.entry(g).or_insert((0, 0));

        let is_positive_true = y_true[i] > 0.0;
        let is_positive_pred = y_pred[i] > 0.0;

        if is_positive_true && !is_positive_pred {
            entry.0 += 1; // false negative
        } else if !is_positive_true && is_positive_pred {
            entry.1 += 1; // false positive
        }
    }

    // Compute FN/FP ratio for each group
    let mut ratios: Vec<f64> = Vec::new();
    for (fn_count, fp_count) in group_fn_fp.values() {
        if *fp_count > 0 {
            ratios.push(*fn_count as f64 / *fp_count as f64);
        } else if *fn_count > 0 {
            ratios.push(f64::INFINITY);
        } else {
            ratios.push(0.0); // No errors
        }
    }

    if ratios.len() < 2 {
        return Ok(0.0);
    }

    // Filter out infinite ratios and compute max difference
    let finite_ratios: Vec<f64> = ratios.iter().copied().filter(|r| r.is_finite()).collect();

    if finite_ratios.len() < 2 {
        // If only one finite ratio, some groups have no FP, which is extreme inequality
        if ratios.iter().any(|r| r.is_infinite()) && finite_ratios.len() == 1 {
            return Ok(f64::INFINITY);
        }
        return Ok(0.0);
    }

    let mut max_diff = 0.0_f64;
    for i in 0..finite_ratios.len() {
        for j in i + 1..finite_ratios.len() {
            let diff = (finite_ratios[i] - finite_ratios[j]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    Ok(max_diff)
}

/// Performs a comprehensive fairness audit across all metrics.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels.
/// * `y_pred` - Predicted labels.
/// * `groups` - Group membership labels.
///
/// # Returns
///
/// A [`FairnessAuditResult`] with all fairness metrics.
pub fn fairness_audit(
    y_true: &[f64],
    y_pred: &[f64],
    groups: &[usize],
) -> Result<FairnessAuditResult> {
    validate_ternary_inputs(y_true, y_pred, groups)?;

    let dp_ratio = demographic_parity_ratio(y_pred, groups)?;
    let eo_ratio = equal_opportunity_ratio(y_true, y_pred, groups).unwrap_or(1.0);
    let eod_ratio = equalized_odds_ratio(y_true, y_pred, groups).unwrap_or(1.0);

    let selection_rates = group_positive_rates(y_pred, groups)?;
    let tpr = group_true_positive_rates(y_true, y_pred, groups).unwrap_or_default();
    let fpr = group_false_positive_rates(y_true, y_pred, groups).unwrap_or_default();
    let ppv = group_positive_predictive_values(y_true, y_pred, groups).unwrap_or_default();

    Ok(FairnessAuditResult {
        demographic_parity_ratio: dp_ratio,
        equal_opportunity_ratio: eo_ratio,
        equalized_odds_ratio: eod_ratio,
        disparate_impact_ratio: dp_ratio,
        four_fifths_rule_satisfied: dp_ratio >= 0.8,
        group_selection_rates: selection_rates,
        group_tpr: tpr,
        group_fpr: fpr,
        group_ppv: ppv,
    })
}

// ---- Helper functions ----

/// Computes the positive prediction rate for each group.
fn group_positive_rates(y_pred: &[f64], groups: &[usize]) -> Result<HashMap<usize, f64>> {
    let mut group_counts: HashMap<usize, (usize, usize)> = HashMap::new(); // (positive, total)

    for i in 0..y_pred.len() {
        let entry = group_counts.entry(groups[i]).or_insert((0, 0));
        entry.1 += 1;
        if y_pred[i] > 0.0 {
            entry.0 += 1;
        }
    }

    let mut rates = HashMap::new();
    for (g, (pos, total)) in &group_counts {
        if *total > 0 {
            rates.insert(*g, *pos as f64 / *total as f64);
        }
    }

    Ok(rates)
}

/// Computes TPR (recall) for each group.
fn group_true_positive_rates(
    y_true: &[f64],
    y_pred: &[f64],
    groups: &[usize],
) -> Result<HashMap<usize, f64>> {
    let mut group_tp: HashMap<usize, usize> = HashMap::new();
    let mut group_pos: HashMap<usize, usize> = HashMap::new();

    for i in 0..y_true.len() {
        let g = groups[i];
        if y_true[i] > 0.0 {
            *group_pos.entry(g).or_insert(0) += 1;
            if y_pred[i] > 0.0 {
                *group_tp.entry(g).or_insert(0) += 1;
            }
        }
    }

    let mut rates = HashMap::new();
    for (g, &pos) in &group_pos {
        if pos > 0 {
            let tp = group_tp.get(g).copied().unwrap_or(0);
            rates.insert(*g, tp as f64 / pos as f64);
        }
    }

    Ok(rates)
}

/// Computes FPR for each group.
fn group_false_positive_rates(
    y_true: &[f64],
    y_pred: &[f64],
    groups: &[usize],
) -> Result<HashMap<usize, f64>> {
    let mut group_fp: HashMap<usize, usize> = HashMap::new();
    let mut group_neg: HashMap<usize, usize> = HashMap::new();

    for i in 0..y_true.len() {
        let g = groups[i];
        if y_true[i] <= 0.0 {
            *group_neg.entry(g).or_insert(0) += 1;
            if y_pred[i] > 0.0 {
                *group_fp.entry(g).or_insert(0) += 1;
            }
        }
    }

    let mut rates = HashMap::new();
    for (g, &neg) in &group_neg {
        if neg > 0 {
            let fp = group_fp.get(g).copied().unwrap_or(0);
            rates.insert(*g, fp as f64 / neg as f64);
        }
    }

    Ok(rates)
}

/// Computes PPV for each group.
fn group_positive_predictive_values(
    y_true: &[f64],
    y_pred: &[f64],
    groups: &[usize],
) -> Result<HashMap<usize, f64>> {
    let mut group_tp: HashMap<usize, usize> = HashMap::new();
    let mut group_pred_pos: HashMap<usize, usize> = HashMap::new();

    for i in 0..y_true.len() {
        let g = groups[i];
        if y_pred[i] > 0.0 {
            *group_pred_pos.entry(g).or_insert(0) += 1;
            if y_true[i] > 0.0 {
                *group_tp.entry(g).or_insert(0) += 1;
            }
        }
    }

    let mut ppv = HashMap::new();
    for (g, &pred_pos) in &group_pred_pos {
        if pred_pos > 0 {
            let tp = group_tp.get(g).copied().unwrap_or(0);
            ppv.insert(*g, tp as f64 / pred_pos as f64);
        }
    }

    Ok(ppv)
}

/// Validates ternary inputs (y_true, y_pred, groups).
fn validate_ternary_inputs(y_true: &[f64], y_pred: &[f64], groups: &[usize]) -> Result<()> {
    if y_true.len() != y_pred.len() || y_true.len() != groups.len() {
        return Err(MetricsError::InvalidInput(format!(
            "All inputs must have the same length: {}, {}, {}",
            y_true.len(),
            y_pred.len(),
            groups.len()
        )));
    }
    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "inputs must not be empty".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- demographic_parity_ratio tests ----

    #[test]
    fn test_dp_ratio_perfect() {
        // Both groups have same positive rate (50%)
        let y_pred = vec![1.0, 0.0, 1.0, 0.0]; // group 0: 50%, group 1: 50%
        let groups = vec![0, 0, 1, 1];
        let val = demographic_parity_ratio(&y_pred, &groups).expect("should succeed");
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dp_ratio_imbalanced() {
        // Group 0: 100% positive, Group 1: 0% positive
        let y_pred = vec![1.0, 1.0, 0.0, 0.0];
        let groups = vec![0, 0, 1, 1];
        let val = demographic_parity_ratio(&y_pred, &groups).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_dp_ratio_partial() {
        // Group 0: 2/3 positive, Group 1: 1/3 positive
        let y_pred = vec![1.0, 1.0, 0.0, 1.0, 0.0, 0.0];
        let groups = vec![0, 0, 0, 1, 1, 1];
        let val = demographic_parity_ratio(&y_pred, &groups).expect("should succeed");
        // ratio = (1/3) / (2/3) = 0.5
        assert!((val - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_dp_ratio_multi_group() {
        // 3 groups
        let y_pred = vec![1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let groups = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let val = demographic_parity_ratio(&y_pred, &groups).expect("should succeed");
        // Group 0: 2/3, Group 1: 1/3, Group 2: 3/3
        // ratio = (1/3) / (3/3) = 1/3
        assert!((val - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_dp_ratio_empty() {
        assert!(demographic_parity_ratio(&[], &[]).is_err());
    }

    // ---- equal_opportunity_ratio tests ----

    #[test]
    fn test_eo_ratio_perfect() {
        // Both groups have TPR = 1.0
        let y_true = vec![1.0, 0.0, 1.0, 0.0];
        let y_pred = vec![1.0, 0.0, 1.0, 0.0];
        let groups = vec![0, 0, 1, 1];
        let val = equal_opportunity_ratio(&y_true, &y_pred, &groups).expect("should succeed");
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_eo_ratio_unequal() {
        // Group 0: TPR = 1/1 = 1.0, Group 1: TPR = 0/1 = 0.0
        let y_true = vec![1.0, 0.0, 1.0, 0.0];
        let y_pred = vec![1.0, 0.0, 0.0, 0.0];
        let groups = vec![0, 0, 1, 1];
        let val = equal_opportunity_ratio(&y_true, &y_pred, &groups).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_eo_ratio_partial() {
        // Group 0: TPR = 1/2 = 0.5, Group 1: TPR = 2/2 = 1.0
        let y_true = vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let y_pred = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        let groups = vec![0, 0, 0, 1, 1, 1];
        let val = equal_opportunity_ratio(&y_true, &y_pred, &groups).expect("should succeed");
        assert!((val - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_eo_ratio_mismatched() {
        assert!(equal_opportunity_ratio(&[1.0], &[0.0], &[0, 1]).is_err());
    }

    // ---- equalized_odds_ratio tests ----

    #[test]
    fn test_eod_ratio_perfect() {
        let y_true = vec![1.0, 0.0, 1.0, 0.0];
        let y_pred = vec![1.0, 0.0, 1.0, 0.0];
        let groups = vec![0, 0, 1, 1];
        let val = equalized_odds_ratio(&y_true, &y_pred, &groups).expect("should succeed");
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_eod_ratio_unequal() {
        let y_true = vec![1.0, 0.0, 1.0, 0.0];
        let y_pred = vec![1.0, 0.0, 0.0, 1.0]; // Group 1: TPR=0, FPR=1
        let groups = vec![0, 0, 1, 1];
        let val = equalized_odds_ratio(&y_true, &y_pred, &groups).expect("should succeed");
        assert!(val < 0.5);
    }

    #[test]
    fn test_eod_ratio_empty() {
        assert!(equalized_odds_ratio(&[], &[], &[]).is_err());
    }

    #[test]
    fn test_eod_ratio_multi_group() {
        let y_true = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let y_pred = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let groups = vec![0, 0, 1, 1, 2, 2];
        let val = equalized_odds_ratio(&y_true, &y_pred, &groups).expect("should succeed");
        assert!((val - 1.0).abs() < 1e-10);
    }

    // ---- disparate_impact_check tests ----

    #[test]
    fn test_di_check_passes() {
        let y_pred = vec![1.0, 1.0, 1.0, 1.0]; // 100% for both groups
        let groups = vec![0, 0, 1, 1];
        let result = disparate_impact_check(&y_pred, &groups, 0.8).expect("should succeed");
        assert!(result.passes_threshold);
        assert!((result.ratio - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_di_check_fails() {
        let y_pred = vec![1.0, 1.0, 0.0, 0.0]; // G0=100%, G1=0%
        let groups = vec![0, 0, 1, 1];
        let result = disparate_impact_check(&y_pred, &groups, 0.8).expect("should succeed");
        assert!(!result.passes_threshold);
        assert!((result.ratio - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_di_check_borderline() {
        // G0: 100%, G1: 80%
        let y_pred = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
        let groups = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
        let result = disparate_impact_check(&y_pred, &groups, 0.8).expect("should succeed");
        assert!(result.passes_threshold);
    }

    #[test]
    fn test_di_check_empty() {
        assert!(disparate_impact_check(&[], &[], 0.8).is_err());
    }

    // ---- predictive_parity_difference tests ----

    #[test]
    fn test_pp_difference_equal() {
        // Both groups have PPV = 1.0
        let y_true = vec![1.0, 0.0, 1.0, 0.0];
        let y_pred = vec![1.0, 0.0, 1.0, 0.0];
        let groups = vec![0, 0, 1, 1];
        let val = predictive_parity_difference(&y_true, &y_pred, &groups).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_pp_difference_unequal() {
        // Group 0: PPV = 1/1 = 1.0, Group 1: PPV = 0/1 = 0.0
        let y_true = vec![1.0, 0.0, 0.0, 0.0];
        let y_pred = vec![1.0, 0.0, 1.0, 0.0];
        let groups = vec![0, 0, 1, 1];
        let val = predictive_parity_difference(&y_true, &y_pred, &groups).expect("should succeed");
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pp_difference_partial() {
        // Group 0: PPV = 1/2 = 0.5, Group 1: PPV = 2/2 = 1.0
        let y_true = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        let y_pred = vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let groups = vec![0, 0, 0, 1, 1, 1];
        let val = predictive_parity_difference(&y_true, &y_pred, &groups).expect("should succeed");
        assert!((val - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_pp_difference_empty() {
        assert!(predictive_parity_difference(&[], &[], &[]).is_err());
    }

    // ---- treatment_equality_difference tests ----

    #[test]
    fn test_treatment_equality_equal() {
        // Both groups have same FN/FP ratio
        let y_true = vec![1.0, 0.0, 1.0, 0.0];
        let y_pred = vec![0.0, 1.0, 0.0, 1.0]; // 1FN+1FP per group
        let groups = vec![0, 0, 1, 1];
        let val = treatment_equality_difference(&y_true, &y_pred, &groups).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_treatment_equality_unequal() {
        // Group 0: 1FN, 0FP (ratio=inf), Group 1: 0FN, 1FP (ratio=0)
        let y_true = vec![1.0, 1.0, 0.0, 0.0];
        let y_pred = vec![0.0, 1.0, 1.0, 0.0];
        let groups = vec![0, 0, 1, 1];
        let val = treatment_equality_difference(&y_true, &y_pred, &groups).expect("should succeed");
        // One group has infinite ratio, one has 0
        assert!(val.is_infinite() || val > 0.5);
    }

    #[test]
    fn test_treatment_equality_no_errors() {
        let y_true = vec![1.0, 0.0, 1.0, 0.0];
        let y_pred = vec![1.0, 0.0, 1.0, 0.0];
        let groups = vec![0, 0, 1, 1];
        let val = treatment_equality_difference(&y_true, &y_pred, &groups).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_treatment_equality_empty() {
        assert!(treatment_equality_difference(&[], &[], &[]).is_err());
    }

    // ---- fairness_audit tests ----

    #[test]
    fn test_fairness_audit_basic() {
        let y_true = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let y_pred = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let groups = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let result = fairness_audit(&y_true, &y_pred, &groups).expect("should succeed");

        assert!((result.demographic_parity_ratio - 1.0).abs() < 1e-10);
        assert!((result.equal_opportunity_ratio - 1.0).abs() < 1e-10);
        assert!(result.four_fifths_rule_satisfied);
    }

    #[test]
    fn test_fairness_audit_unfair() {
        let y_true = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let y_pred = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]; // only G0 gets positive
        let groups = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let result = fairness_audit(&y_true, &y_pred, &groups).expect("should succeed");

        assert!((result.demographic_parity_ratio - 0.0).abs() < 1e-10);
        assert!(!result.four_fifths_rule_satisfied);
    }

    #[test]
    fn test_fairness_audit_has_per_group_data() {
        let y_true = vec![0.0, 1.0, 0.0, 1.0];
        let y_pred = vec![0.0, 1.0, 0.0, 1.0];
        let groups = vec![0, 0, 1, 1];
        let result = fairness_audit(&y_true, &y_pred, &groups).expect("should succeed");

        assert!(result.group_selection_rates.contains_key(&0));
        assert!(result.group_selection_rates.contains_key(&1));
    }

    #[test]
    fn test_fairness_audit_empty() {
        assert!(fairness_audit(&[], &[], &[]).is_err());
    }
}
