//! Evaluation metrics for probabilistic energy forecasts
//!
//! Implements pinball loss, CRPS, coverage, Winkler score, reliability
//! diagrams, skill scores, and sharpness.

use super::types::QuantileForecast;

/// Compute the pinball (check) loss for quantile τ.
///
/// L_τ(y, q) = τ(y - q) if y ≥ q, else (1 - τ)(q - y).
/// Returns the mean pinball loss over all observations.
pub fn pinball_loss(y: &[f64], q_hat: &[f64], tau: f64) -> f64 {
    if y.is_empty() {
        return 0.0;
    }
    let n = y.len().min(q_hat.len());
    let mut total = 0.0;
    for i in 0..n {
        let diff = y[i] - q_hat[i];
        if diff >= 0.0 {
            total += tau * diff;
        } else {
            total += (1.0 - tau) * (-diff);
        }
    }
    total / n as f64
}

/// Compute CRPS by integrating pinball loss over quantile levels (trapezoidal).
///
/// CRPS ≈ ∫₀¹ 2·pinball_τ(y, q̂_τ) dτ, approximated using the
/// provided quantile forecasts.
pub fn crps(y: &[f64], quantile_forecasts: &[QuantileForecast]) -> f64 {
    if quantile_forecasts.is_empty() || y.is_empty() {
        return 0.0;
    }

    // Sort by quantile level
    let mut sorted_qf: Vec<&QuantileForecast> = quantile_forecasts.iter().collect();
    sorted_qf.sort_by(|a, b| {
        a.quantile
            .partial_cmp(&b.quantile)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Trapezoidal integration of 2 * pinball_loss over tau
    let mut integral = 0.0;
    for i in 0..sorted_qf.len() {
        let tau = sorted_qf[i].quantile;
        let pb = pinball_loss(y, &sorted_qf[i].values, tau);

        if i == 0 {
            // Left edge: [0, tau]
            integral += 2.0 * pb * tau;
        } else {
            let prev_tau = sorted_qf[i - 1].quantile;
            let prev_pb = pinball_loss(y, &sorted_qf[i - 1].values, prev_tau);
            // Trapezoidal rule
            integral += (tau - prev_tau) * (2.0 * pb + 2.0 * prev_pb) / 2.0;
        }

        if i == sorted_qf.len() - 1 && tau < 1.0 {
            // Right edge: [tau, 1]
            integral += 2.0 * pb * (1.0 - tau);
        }
    }

    integral
}

/// Compute empirical coverage: fraction of y in \[lower, upper\].
pub fn coverage(y: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
    if y.is_empty() {
        return 0.0;
    }
    let n = y.len().min(lower.len()).min(upper.len());
    let count = (0..n)
        .filter(|&i| y[i] >= lower[i] && y[i] <= upper[i])
        .count();
    count as f64 / n as f64
}

/// Compute the Winkler interval score.
///
/// score = width + (2/α) * (lower - y) if y < lower
///       = width + (2/α) * (y - upper) if y > upper
///       = width otherwise
///
/// Returns the mean Winkler score.
pub fn winkler_score(y: &[f64], lower: &[f64], upper: &[f64], alpha: f64) -> f64 {
    if y.is_empty() || alpha <= 0.0 {
        return 0.0;
    }
    let n = y.len().min(lower.len()).min(upper.len());
    let mut total = 0.0;
    let penalty_factor = 2.0 / alpha;

    for i in 0..n {
        let width = upper[i] - lower[i];
        let penalty = if y[i] < lower[i] {
            penalty_factor * (lower[i] - y[i])
        } else if y[i] > upper[i] {
            penalty_factor * (y[i] - upper[i])
        } else {
            0.0
        };
        total += width + penalty;
    }
    total / n as f64
}

/// Compute a reliability diagram: nominal quantile vs empirical coverage.
///
/// For each quantile forecast, computes the fraction of observations
/// below the forecast (empirical quantile). Returns (nominal, empirical) pairs.
pub fn reliability_diagram(
    y: &[f64],
    quantile_forecasts: &[QuantileForecast],
    _n_bins: usize,
) -> Vec<(f64, f64)> {
    if y.is_empty() || quantile_forecasts.is_empty() {
        return Vec::new();
    }

    let n = y.len();
    let mut result = Vec::with_capacity(quantile_forecasts.len());

    for qf in quantile_forecasts {
        let effective_n = n.min(qf.values.len());
        let count_below = (0..effective_n).filter(|&i| y[i] <= qf.values[i]).count();
        let empirical = count_below as f64 / effective_n as f64;
        result.push((qf.quantile, empirical));
    }

    result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    result
}

/// Compute skill score: 1 - crps_model / crps_baseline.
///
/// A skill score of 0 means no improvement over baseline.
/// Positive values indicate improvement, negative means worse.
pub fn skill_score(crps_model: f64, crps_baseline: f64) -> f64 {
    if crps_baseline.abs() < 1e-15 {
        return 0.0;
    }
    1.0 - crps_model / crps_baseline
}

/// Compute sharpness: mean interval width.
pub fn sharpness(lower: &[f64], upper: &[f64]) -> f64 {
    if lower.is_empty() {
        return 0.0;
    }
    let n = lower.len().min(upper.len());
    let total: f64 = (0..n).map(|i| upper[i] - lower[i]).sum();
    total / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pinball_loss_median() {
        // At τ=0.5, pinball loss = 0.5 * |y - q|
        let y = vec![10.0, 20.0, 30.0];
        let q = vec![12.0, 18.0, 30.0];
        let loss = pinball_loss(&y, &q, 0.5);
        // (0.5*2 + 0.5*2 + 0) / 3 = 2/3
        assert!(
            (loss - 2.0 / 3.0).abs() < 1e-10,
            "pinball at 0.5 = half abs error, got {}",
            loss
        );
    }

    #[test]
    fn test_crps_perfect_forecast() {
        // If all quantile forecasts equal y, each pinball loss = 0 => CRPS ≈ 0
        let y = vec![5.0, 10.0, 15.0];
        let qfs: Vec<QuantileForecast> = [0.1, 0.5, 0.9]
            .iter()
            .map(|&tau| QuantileForecast {
                quantile: tau,
                values: y.clone(),
            })
            .collect();
        let c = crps(&y, &qfs);
        assert!(
            c.abs() < 1e-10,
            "CRPS for perfect forecast should be 0, got {}",
            c
        );
    }

    #[test]
    fn test_coverage() {
        let y = vec![1.0, 5.0, 10.0, 15.0, 20.0];
        let lower = vec![0.0, 4.0, 8.0, 12.0, 25.0]; // last one misses
        let upper = vec![2.0, 6.0, 12.0, 16.0, 26.0];
        let cov = coverage(&y, &lower, &upper);
        assert!(
            (cov - 0.8).abs() < 1e-10,
            "4/5 should be covered, got {}",
            cov
        );
    }

    #[test]
    fn test_winkler_score_penalty() {
        // y outside the interval should incur penalty
        let y = vec![5.0];
        let lower = vec![10.0];
        let upper = vec![20.0];
        let alpha = 0.1; // 90% interval
        let score = winkler_score(&y, &lower, &upper, alpha);
        // width = 10, penalty = 2/0.1 * (10-5) = 100
        assert!((score - 110.0).abs() < 1e-10, "expected 110, got {}", score);
    }

    #[test]
    fn test_winkler_no_penalty() {
        let y = vec![15.0];
        let lower = vec![10.0];
        let upper = vec![20.0];
        let score = winkler_score(&y, &lower, &upper, 0.1);
        // width = 10, no penalty
        assert!((score - 10.0).abs() < 1e-10, "expected 10, got {}", score);
    }

    #[test]
    fn test_skill_score_baseline() {
        // Model equals baseline => skill = 0
        let ss = skill_score(5.0, 5.0);
        assert!(ss.abs() < 1e-10, "same as baseline => 0");
    }

    #[test]
    fn test_skill_score_better() {
        let ss = skill_score(2.5, 5.0);
        assert!((ss - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sharpness() {
        let lower = vec![10.0, 20.0, 30.0];
        let upper = vec![15.0, 30.0, 35.0];
        let s = sharpness(&lower, &upper);
        // widths: 5, 10, 5 => mean = 20/3
        assert!((s - 20.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_reliability_diagram() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let qfs = vec![QuantileForecast {
            quantile: 0.5,
            values: vec![3.0, 3.0, 3.0, 3.0, 3.0],
        }];
        let diag = reliability_diagram(&y, &qfs, 10);
        assert_eq!(diag.len(), 1);
        // y <= 3: values 1, 2, 3 => 3/5 = 0.6
        assert!((diag[0].0 - 0.5).abs() < 1e-10);
        assert!((diag[0].1 - 0.6).abs() < 1e-10);
    }
}
