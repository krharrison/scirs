//! Advanced fairness metrics with simple slice-based API.
//!
//! Provides functions that operate directly on `&[bool]` and `&[f64]` inputs,
//! complementing the ndarray-based API in the parent module.

use crate::error::{MetricsError, Result};

/// Demographic parity difference.
///
/// Measures the difference in positive prediction rates between the unprotected
/// group (sensitive=false) and the protected group (sensitive=true).
///
/// DPD = P(Ŷ=1|A=0) - P(Ŷ=1|A=1)
///
/// A value of 0 indicates perfect demographic parity.
pub fn demographic_parity_difference(y_pred: &[bool], sensitive: &[bool]) -> Result<f64> {
    validate_lengths(y_pred, sensitive, "y_pred", "sensitive")?;
    if y_pred.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty input arrays".to_string(),
        ));
    }

    let (pos_unprot, total_unprot) = group_positive_rate(y_pred, sensitive, false);
    let (pos_prot, total_prot) = group_positive_rate(y_pred, sensitive, true);

    if total_unprot == 0 || total_prot == 0 {
        return Err(MetricsError::InvalidInput(
            "One of the sensitive groups is empty".to_string(),
        ));
    }

    Ok(pos_unprot as f64 / total_unprot as f64 - pos_prot as f64 / total_prot as f64)
}

/// Equalized odds difference.
///
/// Returns `(tpr_difference, fpr_difference)` where each is measured as
/// group_0 rate − group_1 rate.
pub fn equalized_odds_difference(
    y_true: &[bool],
    y_pred: &[bool],
    sensitive: &[bool],
) -> Result<(f64, f64)> {
    validate_lengths3(y_true, y_pred, sensitive)?;
    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty input arrays".to_string(),
        ));
    }

    let tpr0 = compute_tpr(y_true, y_pred, sensitive, false)?;
    let tpr1 = compute_tpr(y_true, y_pred, sensitive, true)?;
    let fpr0 = compute_fpr(y_true, y_pred, sensitive, false)?;
    let fpr1 = compute_fpr(y_true, y_pred, sensitive, true)?;

    Ok((tpr0 - tpr1, fpr0 - fpr1))
}

/// Equal opportunity difference.
///
/// EOD = TPR(A=0) - TPR(A=1)
///
/// A value of 0 means both groups have the same true positive rate.
pub fn equal_opportunity_difference(
    y_true: &[bool],
    y_pred: &[bool],
    sensitive: &[bool],
) -> Result<f64> {
    validate_lengths3(y_true, y_pred, sensitive)?;
    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty input arrays".to_string(),
        ));
    }

    let tpr0 = compute_tpr(y_true, y_pred, sensitive, false)?;
    let tpr1 = compute_tpr(y_true, y_pred, sensitive, true)?;
    Ok(tpr0 - tpr1)
}

/// Disparate impact ratio.
///
/// DIR = P(Ŷ=1|A=1) / P(Ŷ=1|A=0)
///
/// Values in [0.8, 1.25] are generally considered acceptable.
/// Returns 0.0 when the unprotected group has zero positive predictions.
pub fn disparate_impact_ratio(y_pred: &[bool], sensitive: &[bool]) -> Result<f64> {
    validate_lengths(y_pred, sensitive, "y_pred", "sensitive")?;
    if y_pred.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty input arrays".to_string(),
        ));
    }

    let (pos_prot, total_prot) = group_positive_rate(y_pred, sensitive, true);
    let (pos_unprot, total_unprot) = group_positive_rate(y_pred, sensitive, false);

    if total_prot == 0 || total_unprot == 0 {
        return Err(MetricsError::InvalidInput(
            "One of the sensitive groups is empty".to_string(),
        ));
    }

    let rate_prot = pos_prot as f64 / total_prot as f64;
    let rate_unprot = pos_unprot as f64 / total_unprot as f64;

    if rate_unprot == 0.0 {
        return Ok(0.0);
    }
    Ok(rate_prot / rate_unprot)
}

/// Predictive parity (precision difference).
///
/// PP = PPV(A=0) - PPV(A=1)
///
/// PPV = Positive Predictive Value = TP / (TP + FP)
pub fn predictive_parity(
    y_true: &[bool],
    y_pred: &[bool],
    sensitive: &[bool],
) -> Result<f64> {
    validate_lengths3(y_true, y_pred, sensitive)?;
    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty input arrays".to_string(),
        ));
    }

    let ppv0 = compute_ppv(y_true, y_pred, sensitive, false)?;
    let ppv1 = compute_ppv(y_true, y_pred, sensitive, true)?;
    Ok(ppv0 - ppv1)
}

/// Individual fairness score.
///
/// Computes the fraction of pairs of individuals with similar features
/// that also receive similar predictions.
///
/// # Arguments
/// * `y_pred` — continuous predictions for each individual
/// * `features` — feature vectors for each individual
/// * `similarity_threshold` — Euclidean distance threshold for "similar" individuals
///
/// # Returns
/// Fraction of similar pairs with similar predictions (higher = more fair).
pub fn individual_fairness_score(
    y_pred: &[f64],
    features: &[Vec<f64>],
    similarity_threshold: f64,
) -> Result<f64> {
    if y_pred.len() != features.len() {
        return Err(MetricsError::InvalidInput(format!(
            "y_pred ({}) and features ({}) have different lengths",
            y_pred.len(),
            features.len()
        )));
    }
    if y_pred.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty input arrays".to_string(),
        ));
    }
    if similarity_threshold <= 0.0 {
        return Err(MetricsError::InvalidInput(
            "similarity_threshold must be positive".to_string(),
        ));
    }

    let n = y_pred.len();
    if n < 2 {
        // No pairs to compare
        return Ok(1.0);
    }

    let mut similar_pairs = 0_u64;
    let mut fair_pairs = 0_u64;

    for i in 0..n {
        for j in (i + 1)..n {
            let feat_dist = euclidean_distance(&features[i], &features[j]);
            if feat_dist <= similarity_threshold {
                similar_pairs += 1;
                let pred_diff = (y_pred[i] - y_pred[j]).abs();
                // "Similar predictions" use the same threshold for simplicity
                if pred_diff <= similarity_threshold {
                    fair_pairs += 1;
                }
            }
        }
    }

    if similar_pairs == 0 {
        // No similar pairs found → vacuously fair
        return Ok(1.0);
    }

    Ok(fair_pairs as f64 / similar_pairs as f64)
}

/// Counterfactual fairness check.
///
/// Measures the mean absolute difference between original predictions and
/// counterfactual predictions (i.e., predictions when sensitive attributes
/// are flipped).
///
/// # Arguments
/// * `predictions` — original predictions
/// * `counterfactual_predictions` — predictions with flipped sensitive attributes
/// * `threshold` — maximum acceptable mean difference for a "fair" model
///
/// # Returns
/// Mean absolute difference between original and counterfactual predictions.
/// Lower values indicate better counterfactual fairness.
pub fn counterfactual_fairness_check(
    predictions: &[f64],
    counterfactual_predictions: &[f64],
    threshold: f64,
) -> Result<f64> {
    if predictions.len() != counterfactual_predictions.len() {
        return Err(MetricsError::InvalidInput(format!(
            "predictions ({}) and counterfactual_predictions ({}) have different lengths",
            predictions.len(),
            counterfactual_predictions.len()
        )));
    }
    if predictions.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty input arrays".to_string(),
        ));
    }
    if threshold < 0.0 {
        return Err(MetricsError::InvalidInput(
            "threshold must be non-negative".to_string(),
        ));
    }

    let mean_diff: f64 = predictions
        .iter()
        .zip(counterfactual_predictions.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum::<f64>()
        / predictions.len() as f64;

    Ok(mean_diff)
}

// ─── helper functions ───────────────────────────────────────────────────────

fn validate_lengths<T, U>(a: &[T], b: &[U], name_a: &str, name_b: &str) -> Result<()> {
    if a.len() != b.len() {
        return Err(MetricsError::InvalidInput(format!(
            "{} ({}) and {} ({}) have different lengths",
            name_a,
            a.len(),
            name_b,
            b.len()
        )));
    }
    Ok(())
}

fn validate_lengths3(y_true: &[bool], y_pred: &[bool], sensitive: &[bool]) -> Result<()> {
    validate_lengths(y_true, y_pred, "y_true", "y_pred")?;
    validate_lengths(y_true, sensitive, "y_true", "sensitive")?;
    Ok(())
}

/// Count (positives, total) for a given group.
fn group_positive_rate(y_pred: &[bool], sensitive: &[bool], group: bool) -> (usize, usize) {
    let members: Vec<bool> = y_pred
        .iter()
        .zip(sensitive.iter())
        .filter(|(_, &s)| s == group)
        .map(|(&p, _)| p)
        .collect();
    let total = members.len();
    let pos = members.iter().filter(|&&p| p).count();
    (pos, total)
}

fn compute_tpr(
    y_true: &[bool],
    y_pred: &[bool],
    sensitive: &[bool],
    group: bool,
) -> Result<f64> {
    let (tp, total_pos) = y_true
        .iter()
        .zip(y_pred.iter())
        .zip(sensitive.iter())
        .filter(|((_, _), &s)| s == group)
        .fold((0_usize, 0_usize), |(tp, tot_pos), ((&yt, &yp), _)| {
            (tp + (yt && yp) as usize, tot_pos + yt as usize)
        });

    if total_pos == 0 {
        return Err(MetricsError::InvalidInput(format!(
            "Group {} has no positive ground-truth samples",
            if group { 1 } else { 0 }
        )));
    }
    Ok(tp as f64 / total_pos as f64)
}

fn compute_fpr(
    y_true: &[bool],
    y_pred: &[bool],
    sensitive: &[bool],
    group: bool,
) -> Result<f64> {
    let (fp, total_neg) = y_true
        .iter()
        .zip(y_pred.iter())
        .zip(sensitive.iter())
        .filter(|((_, _), &s)| s == group)
        .fold((0_usize, 0_usize), |(fp, tot_neg), ((&yt, &yp), _)| {
            (fp + (!yt && yp) as usize, tot_neg + (!yt) as usize)
        });

    if total_neg == 0 {
        return Err(MetricsError::InvalidInput(format!(
            "Group {} has no negative ground-truth samples",
            if group { 1 } else { 0 }
        )));
    }
    Ok(fp as f64 / total_neg as f64)
}

fn compute_ppv(
    y_true: &[bool],
    y_pred: &[bool],
    sensitive: &[bool],
    group: bool,
) -> Result<f64> {
    let (tp, total_pred_pos) = y_true
        .iter()
        .zip(y_pred.iter())
        .zip(sensitive.iter())
        .filter(|((_, _), &s)| s == group)
        .fold((0_usize, 0_usize), |(tp, tot_pp), ((&yt, &yp), _)| {
            (tp + (yt && yp) as usize, tot_pp + yp as usize)
        });

    if total_pred_pos == 0 {
        // No positive predictions in this group → PPV undefined; return 0
        return Ok(0.0);
    }
    Ok(tp as f64 / total_pred_pos as f64)
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demographic_parity_difference_parity() {
        // Both groups predict 50% positive → difference = 0
        let y_pred = vec![true, false, true, false];
        let sensitive = vec![false, false, true, true];
        let dpd = demographic_parity_difference(&y_pred, &sensitive).expect("should succeed");
        assert!((dpd - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_demographic_parity_difference_disparity() {
        // Group 0: 100% positive, Group 1: 0% positive → DPD = 1.0
        let y_pred = vec![true, true, false, false];
        let sensitive = vec![false, false, true, true];
        let dpd = demographic_parity_difference(&y_pred, &sensitive).expect("should succeed");
        assert!((dpd - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_equalized_odds_difference() {
        // Perfect classifier for both groups
        let y_true = vec![true, false, true, false];
        let y_pred = vec![true, false, true, false];
        let sensitive = vec![false, false, true, true];
        let (tpr_diff, fpr_diff) =
            equalized_odds_difference(&y_true, &y_pred, &sensitive).expect("should succeed");
        assert!((tpr_diff - 0.0).abs() < 1e-10);
        assert!((fpr_diff - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_equal_opportunity_difference_zero() {
        let y_true = vec![true, false, true, false];
        let y_pred = vec![true, false, true, false];
        let sensitive = vec![false, false, true, true];
        let eod = equal_opportunity_difference(&y_true, &y_pred, &sensitive).expect("should succeed");
        assert!((eod - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_disparate_impact_ratio_fair() {
        // Both groups: 50% positive → DIR = 1.0
        let y_pred = vec![true, false, true, false];
        let sensitive = vec![false, false, true, true];
        let dir = disparate_impact_ratio(&y_pred, &sensitive).expect("should succeed");
        assert!((dir - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_predictive_parity_equal() {
        // Both groups have same PPV
        let y_true = vec![true, false, true, false];
        let y_pred = vec![true, false, true, false];
        let sensitive = vec![false, false, true, true];
        let pp = predictive_parity(&y_true, &y_pred, &sensitive).expect("should succeed");
        assert!((pp - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_individual_fairness_score_identical_predictions() {
        let y_pred = vec![0.5, 0.5, 0.5];
        let features = vec![vec![1.0, 0.0], vec![1.1, 0.0], vec![5.0, 5.0]];
        // All predictions identical → all similar pairs are "fair"
        let score = individual_fairness_score(&y_pred, &features, 0.5).expect("should succeed");
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_counterfactual_fairness_identical() {
        let pred = vec![0.7, 0.3, 0.8];
        let cf_pred = vec![0.7, 0.3, 0.8];
        let diff = counterfactual_fairness_check(&pred, &cf_pred, 0.1).expect("should succeed");
        assert!((diff - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_counterfactual_fairness_difference() {
        let pred = vec![0.7, 0.3];
        let cf_pred = vec![0.3, 0.7];
        let diff = counterfactual_fairness_check(&pred, &cf_pred, 0.1).expect("should succeed");
        assert!((diff - 0.4).abs() < 1e-10);
    }
}
