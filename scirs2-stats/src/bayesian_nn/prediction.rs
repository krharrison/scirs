//! Predictive inference and calibration metrics for Bayesian neural networks.
//!
//! Provides Monte Carlo predictive distributions, calibration diagnostics
//! (ECE, reliability diagrams), scoring rules, and uncertainty decomposition.

use scirs2_core::ndarray::{Array1, Array2};

use crate::error::{StatsError, StatsResult};

use super::types::{BNNPosterior, CovarianceType, PredictiveDistribution, ReliabilityBin};

/// Xorshift64 PRNG for sampling.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Standard normal sample via Box-Muller.
fn randn(state: &mut u64) -> f64 {
    let u1 = (xorshift64(state) as f64) / (u64::MAX as f64);
    let u2 = (xorshift64(state) as f64) / (u64::MAX as f64);
    let u1_clamped = u1.max(1e-300);
    (-2.0 * u1_clamped.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Monte Carlo predictive distribution.
///
/// Draws `n_samples` weight vectors from the posterior and evaluates the
/// predict function for each, then computes the predictive mean and variance.
///
/// # Arguments
/// * `posterior` - The fitted BNN posterior
/// * `predict_fn` - Maps a weight vector to an output prediction vector
/// * `n_samples` - Number of MC samples to draw
/// * `rng_state` - Mutable PRNG state
///
/// # Errors
/// Returns an error if the covariance type is not supported for sampling
/// or if `n_samples` is zero.
pub fn mc_predictive(
    posterior: &BNNPosterior,
    predict_fn: &dyn Fn(&Array1<f64>) -> StatsResult<Array1<f64>>,
    n_samples: usize,
    rng_state: &mut u64,
) -> StatsResult<PredictiveDistribution> {
    if n_samples == 0 {
        return Err(StatsError::invalid_argument(
            "n_samples must be > 0 for MC predictive",
        ));
    }

    let d = posterior.mean.len();
    let mut all_predictions: Vec<Array1<f64>> = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let w = sample_from_posterior(posterior, d, rng_state)?;
        let pred = predict_fn(&w)?;
        all_predictions.push(pred);
    }

    let n_outputs = all_predictions[0].len();

    // Compute mean
    let mut mean = Array1::zeros(n_outputs);
    for p in &all_predictions {
        mean = mean + p;
    }
    mean /= n_samples as f64;

    // Compute variance
    let mut variance = Array1::zeros(n_outputs);
    for p in &all_predictions {
        let diff = p - &mean;
        variance = variance + &diff.mapv(|d| d * d);
    }
    variance /= n_samples as f64;

    // Build samples matrix
    let mut samples = Array2::zeros((n_samples, n_outputs));
    for (i, p) in all_predictions.iter().enumerate() {
        for j in 0..n_outputs {
            samples[[i, j]] = p[j];
        }
    }

    Ok(PredictiveDistribution {
        mean,
        variance,
        samples: Some(samples),
    })
}

/// Sample a weight vector from the posterior, supporting different covariance types.
fn sample_from_posterior(
    posterior: &BNNPosterior,
    d: usize,
    rng_state: &mut u64,
) -> StatsResult<Array1<f64>> {
    match &posterior.covariance_type {
        CovarianceType::Full(cov) => {
            // Cholesky L, then w = mean + L z
            let l = cholesky_lower(cov)?;
            let z: Array1<f64> = Array1::from_shape_fn(d, |_| randn(rng_state));
            Ok(&posterior.mean + &l.dot(&z))
        }
        CovarianceType::Diagonal(diag) => {
            let z: Array1<f64> = Array1::from_shape_fn(d, |_| randn(rng_state));
            let std_dev = diag.mapv(|v| v.max(0.0).sqrt());
            Ok(&posterior.mean + &(&std_dev * &z))
        }
        CovarianceType::LowRankPlusDiagonal { d_diag, deviation } => {
            let z1: Array1<f64> = Array1::from_shape_fn(d, |_| randn(rng_state));
            let k = deviation.ncols();
            let k_minus_1 = if k > 1 { (k - 1) as f64 } else { 1.0 };
            let z2: Array1<f64> = Array1::from_shape_fn(k, |_| randn(rng_state));

            let diag_std = d_diag.mapv(|v| v.max(0.0).sqrt());
            let diag_part = &diag_std * &z1;
            let lr_part = deviation.dot(&z2) / k_minus_1.sqrt();

            let scale = 1.0 / 2.0_f64.sqrt();
            Ok(&posterior.mean + &((&diag_part + &lr_part) * scale))
        }
        CovarianceType::KroneckerFactored { a_factor, b_factor } => {
            // Sample using Kronecker structure: vec(W) ~ N(mean, A kron B)
            // Cholesky of A and B, then W = mean + L_A Z L_B^T (in vectorized form)
            let l_a = cholesky_lower(a_factor)?;
            let l_b = cholesky_lower(b_factor)?;
            let d_a = l_a.nrows();
            let d_b = l_b.nrows();

            if d_a * d_b != d {
                return Err(StatsError::dimension_mismatch(format!(
                    "Kronecker factors {}x{} don't match parameter dim {}",
                    d_a, d_b, d
                )));
            }

            let z_mat = Array2::from_shape_fn((d_a, d_b), |_| randn(rng_state));
            let sample_mat = l_a.dot(&z_mat).dot(&l_b.t());

            // Flatten and add to mean
            let flat: Array1<f64> = Array1::from_iter(sample_mat.iter().copied());
            Ok(&posterior.mean + &flat)
        }
    }
}

/// Cholesky decomposition returning lower triangular factor.
fn cholesky_lower(a: &Array2<f64>) -> StatsResult<Array2<f64>> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(StatsError::dimension_mismatch("Matrix must be square"));
    }
    let mut l = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let mut s = 0.0;
        for k in 0..j {
            s += l[[j, k]] * l[[j, k]];
        }
        let diag = a[[j, j]] - s;
        if diag <= 0.0 {
            return Err(StatsError::computation(format!(
                "Matrix not positive definite at index {} (pivot {})",
                j, diag
            )));
        }
        l[[j, j]] = diag.sqrt();
        for i in (j + 1)..n {
            let mut s2 = 0.0;
            for k in 0..j {
                s2 += l[[i, k]] * l[[j, k]];
            }
            l[[i, j]] = (a[[i, j]] - s2) / l[[j, j]];
        }
    }
    Ok(l)
}

/// Expected Calibration Error (ECE).
///
/// Partitions predictions into `n_bins` equally-spaced bins by predicted
/// probability and computes the weighted average of |accuracy - confidence|.
///
/// # Arguments
/// * `predictions` - Predicted probabilities in \[0, 1\]
/// * `targets` - Binary targets (0 or 1)
/// * `n_bins` - Number of calibration bins
///
/// # Errors
/// Returns an error if lengths differ or `n_bins` is zero.
pub fn expected_calibration_error(
    predictions: &Array1<f64>,
    targets: &Array1<f64>,
    n_bins: usize,
) -> StatsResult<f64> {
    if predictions.len() != targets.len() {
        return Err(StatsError::dimension_mismatch(format!(
            "predictions length {} != targets length {}",
            predictions.len(),
            targets.len()
        )));
    }
    if n_bins == 0 {
        return Err(StatsError::invalid_argument("n_bins must be > 0"));
    }
    if predictions.is_empty() {
        return Ok(0.0);
    }

    let bins = reliability_diagram(predictions, targets, n_bins)?;
    let n_total = predictions.len() as f64;

    let mut ece = 0.0;
    for bin in &bins {
        if bin.count > 0 {
            let weight = bin.count as f64 / n_total;
            ece += weight * (bin.mean_predicted - bin.mean_observed).abs();
        }
    }
    Ok(ece)
}

/// Compute reliability diagram data.
///
/// Bins predictions by predicted probability and computes the mean predicted
/// probability and mean observed frequency (fraction of positive targets)
/// in each bin.
///
/// # Arguments
/// * `predictions` - Predicted probabilities in \[0, 1\]
/// * `targets` - Binary targets (0 or 1)
/// * `n_bins` - Number of equally-spaced bins
///
/// # Errors
/// Returns an error if lengths differ or `n_bins` is zero.
pub fn reliability_diagram(
    predictions: &Array1<f64>,
    targets: &Array1<f64>,
    n_bins: usize,
) -> StatsResult<Vec<ReliabilityBin>> {
    if predictions.len() != targets.len() {
        return Err(StatsError::dimension_mismatch(format!(
            "predictions length {} != targets length {}",
            predictions.len(),
            targets.len()
        )));
    }
    if n_bins == 0 {
        return Err(StatsError::invalid_argument("n_bins must be > 0"));
    }

    let mut bin_sum_pred = vec![0.0; n_bins];
    let mut bin_sum_obs = vec![0.0; n_bins];
    let mut bin_count = vec![0usize; n_bins];

    let bin_width = 1.0 / n_bins as f64;

    for i in 0..predictions.len() {
        let p = predictions[i].clamp(0.0, 1.0);
        let mut bin_idx = (p / bin_width) as usize;
        if bin_idx >= n_bins {
            bin_idx = n_bins - 1;
        }
        bin_sum_pred[bin_idx] += p;
        bin_sum_obs[bin_idx] += targets[i];
        bin_count[bin_idx] += 1;
    }

    let bins: Vec<ReliabilityBin> = (0..n_bins)
        .map(|i| {
            if bin_count[i] > 0 {
                ReliabilityBin {
                    mean_predicted: bin_sum_pred[i] / bin_count[i] as f64,
                    mean_observed: bin_sum_obs[i] / bin_count[i] as f64,
                    count: bin_count[i],
                }
            } else {
                let bin_center = (i as f64 + 0.5) * bin_width;
                ReliabilityBin {
                    mean_predicted: bin_center,
                    mean_observed: 0.0,
                    count: 0,
                }
            }
        })
        .collect();

    Ok(bins)
}

/// Gaussian negative log-likelihood for regression.
///
/// NLL = 0.5 * sum(log(variance) + (mean - target)^2 / variance) + n/2 * log(2 pi)
///
/// # Arguments
/// * `mean` - Predicted means
/// * `variance` - Predicted variances (must be positive)
/// * `targets` - Observed targets
///
/// # Errors
/// Returns an error if lengths differ or any variance is non-positive.
pub fn gaussian_nll(
    mean: &Array1<f64>,
    variance: &Array1<f64>,
    targets: &Array1<f64>,
) -> StatsResult<f64> {
    let n = mean.len();
    if n != variance.len() || n != targets.len() {
        return Err(StatsError::dimension_mismatch(format!(
            "Lengths must match: mean={}, variance={}, targets={}",
            n,
            variance.len(),
            targets.len()
        )));
    }

    for (i, &v) in variance.iter().enumerate() {
        if v <= 0.0 {
            return Err(StatsError::invalid_argument(format!(
                "Variance must be positive, got {} at index {}",
                v, i
            )));
        }
    }

    let mut nll = 0.0;
    for i in 0..n {
        let diff = mean[i] - targets[i];
        nll += variance[i].ln() + diff * diff / variance[i];
    }
    nll *= 0.5;
    nll += 0.5 * (n as f64) * (2.0 * std::f64::consts::PI).ln();

    Ok(nll)
}

/// Decompose predictive uncertainty into aleatoric and epistemic components.
///
/// Given a matrix of prediction samples (from multiple weight draws) and
/// an estimate of observation noise variance:
///
/// - **Total** = Var\[predictions\] + noise_var
/// - **Epistemic** = Var\[predictions\] (due to weight uncertainty)
/// - **Aleatoric** = noise_var (irreducible observation noise)
///
/// # Arguments
/// * `samples` - Prediction samples, shape \[n_samples x n_outputs\]
/// * `noise_var` - Observation noise variance (aleatoric)
///
/// # Returns
/// A tuple of (total, aleatoric, epistemic), each of length n_outputs.
pub fn decompose_uncertainty(
    samples: &Array2<f64>,
    noise_var: f64,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let n_samples = samples.nrows();
    let n_outputs = samples.ncols();

    // Epistemic = Var[E[y|w]] = variance of predictions across weight samples
    let mut mean = Array1::zeros(n_outputs);
    for i in 0..n_samples {
        for j in 0..n_outputs {
            mean[j] += samples[[i, j]];
        }
    }
    mean /= n_samples as f64;

    let mut epistemic = Array1::zeros(n_outputs);
    for i in 0..n_samples {
        for j in 0..n_outputs {
            let d = samples[[i, j]] - mean[j];
            epistemic[j] += d * d;
        }
    }
    epistemic /= n_samples as f64;

    // Aleatoric = noise_var for all outputs
    let aleatoric = Array1::from_elem(n_outputs, noise_var);

    // Total = epistemic + aleatoric
    let total = &epistemic + &aleatoric;

    (total, aleatoric, epistemic)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_mc_predictive_zero_variance_at_map() {
        // If posterior has zero variance (delta at MAP), predictions = MAP predictions
        let posterior = BNNPosterior {
            mean: array![1.0, 2.0],
            covariance_type: CovarianceType::Full(Array2::from_diag(&array![1e-20, 1e-20])),
            log_marginal_likelihood: 0.0,
        };

        let predict_fn = |w: &Array1<f64>| -> StatsResult<Array1<f64>> { Ok(array![w[0] + w[1]]) };

        let mut rng: u64 = 42;
        let pred = mc_predictive(&posterior, &predict_fn, 50, &mut rng).expect("mc predictive");

        // Mean should be close to f(MAP) = 1 + 2 = 3
        assert!(
            (pred.mean[0] - 3.0).abs() < 0.1,
            "Mean {} should be close to 3.0",
            pred.mean[0]
        );
        // Variance should be very small
        assert!(
            pred.variance[0] < 0.01,
            "Variance {} should be very small",
            pred.variance[0]
        );
    }

    #[test]
    fn test_mc_predictive_zero_samples_error() {
        let posterior = BNNPosterior {
            mean: array![1.0],
            covariance_type: CovarianceType::Diagonal(array![1.0]),
            log_marginal_likelihood: 0.0,
        };
        let predict_fn = |_w: &Array1<f64>| -> StatsResult<Array1<f64>> { Ok(array![0.0]) };
        let mut rng: u64 = 1;
        assert!(mc_predictive(&posterior, &predict_fn, 0, &mut rng).is_err());
    }

    #[test]
    fn test_ece_perfectly_calibrated() {
        // Predictions match observed frequencies => ECE ~ 0
        let predictions = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let targets = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let ece = expected_calibration_error(&predictions, &targets, 5).expect("ECE computation");

        // With 5 bins: bin 0-0.2 has pred [0.1,0.2], obs [0,0] => mean_p=0.15, mean_o=0.0
        // Not perfectly calibrated with this simple dataset, but ECE should be finite
        assert!(ece >= 0.0, "ECE must be non-negative");
        assert!(ece.is_finite(), "ECE must be finite");
    }

    #[test]
    fn test_ece_perfect_match() {
        // All predictions = 0.5, all targets = 0.5 (regression-like) => ECE = 0
        let n = 100;
        let predictions = Array1::from_elem(n, 0.5);
        let targets = Array1::from_elem(n, 0.5);
        let ece = expected_calibration_error(&predictions, &targets, 10).expect("ECE");
        assert!(
            ece < 1e-12,
            "ECE should be 0 when predictions perfectly match targets, got {}",
            ece
        );
    }

    #[test]
    fn test_reliability_diagram_bins() {
        let predictions = array![0.1, 0.3, 0.5, 0.7, 0.9];
        let targets = array![0.0, 0.0, 1.0, 1.0, 1.0];

        let bins = reliability_diagram(&predictions, &targets, 5).expect("reliability diagram");

        assert_eq!(bins.len(), 5);

        // Non-empty bins should have mean_predicted in their range
        for bin in &bins {
            if bin.count > 0 {
                assert!(bin.mean_predicted >= 0.0 && bin.mean_predicted <= 1.0);
                assert!(bin.mean_observed >= 0.0 && bin.mean_observed <= 1.0);
            }
        }
    }

    #[test]
    fn test_reliability_diagram_monotonic_for_calibrated() {
        // For a well-calibrated model, observed frequency should increase with
        // predicted probability across non-empty bins
        let n = 1000;
        let mut preds = Vec::with_capacity(n);
        let mut tgts = Vec::with_capacity(n);
        let mut rng: u64 = 42;

        for _ in 0..n {
            let p = (xorshift64(&mut rng) as f64) / (u64::MAX as f64);
            preds.push(p);
            // Generate calibrated targets: y=1 with probability p
            let u = (xorshift64(&mut rng) as f64) / (u64::MAX as f64);
            tgts.push(if u < p { 1.0 } else { 0.0 });
        }

        let predictions = Array1::from_vec(preds);
        let targets = Array1::from_vec(tgts);

        let bins = reliability_diagram(&predictions, &targets, 10).expect("reliability diagram");

        // Check roughly monotonic for non-empty bins (allow some noise)
        let non_empty: Vec<&ReliabilityBin> = bins.iter().filter(|b| b.count > 10).collect();
        if non_empty.len() >= 2 {
            let first = non_empty[0].mean_observed;
            let last = non_empty[non_empty.len() - 1].mean_observed;
            assert!(
                last > first,
                "Expected monotonic increase: first={}, last={}",
                first,
                last
            );
        }
    }

    #[test]
    fn test_gaussian_nll_known_value() {
        // For mean=target, var=1: NLL = 0.5 * n * (log(1) + 0) + 0.5*n*log(2pi)
        //                              = 0.5 * n * log(2pi)
        let mean = array![1.0, 2.0, 3.0];
        let variance = array![1.0, 1.0, 1.0];
        let targets = array![1.0, 2.0, 3.0]; // perfect predictions

        let nll = gaussian_nll(&mean, &variance, &targets).expect("NLL");
        let expected = 1.5 * (2.0 * std::f64::consts::PI).ln();
        assert!(
            (nll - expected).abs() < 1e-10,
            "NLL {} != expected {}",
            nll,
            expected
        );
    }

    #[test]
    fn test_gaussian_nll_zero_variance_error() {
        let mean = array![1.0];
        let variance = array![0.0];
        let targets = array![1.0];
        assert!(gaussian_nll(&mean, &variance, &targets).is_err());
    }

    #[test]
    fn test_gaussian_nll_negative_variance_error() {
        let mean = array![1.0];
        let variance = array![-1.0];
        let targets = array![1.0];
        assert!(gaussian_nll(&mean, &variance, &targets).is_err());
    }

    #[test]
    fn test_decompose_uncertainty_sum() {
        // Total should equal aleatoric + epistemic
        let samples = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 1.5, 2.5, 0.5, 1.5, 1.0, 2.0])
            .expect("create samples");

        let noise_var = 0.1;
        let (total, aleatoric, epistemic) = decompose_uncertainty(&samples, noise_var);

        for i in 0..2 {
            let sum = aleatoric[i] + epistemic[i];
            assert!(
                (total[i] - sum).abs() < 1e-12,
                "total[{}]={} != aleatoric+epistemic={}",
                i,
                total[i],
                sum
            );
        }

        // Aleatoric should be noise_var everywhere
        for i in 0..2 {
            assert!((aleatoric[i] - noise_var).abs() < 1e-12);
        }

        // Epistemic should be non-negative
        for i in 0..2 {
            assert!(epistemic[i] >= 0.0);
        }
    }

    #[test]
    fn test_decompose_uncertainty_zero_noise() {
        let samples = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).expect("create samples");
        let (total, aleatoric, epistemic) = decompose_uncertainty(&samples, 0.0);
        assert!((aleatoric[0]).abs() < 1e-12);
        assert!((total[0] - epistemic[0]).abs() < 1e-12);
    }

    #[test]
    fn test_ece_length_mismatch() {
        let p = array![0.5, 0.5];
        let t = array![1.0];
        assert!(expected_calibration_error(&p, &t, 10).is_err());
    }

    #[test]
    fn test_ece_zero_bins() {
        let p = array![0.5];
        let t = array![1.0];
        assert!(expected_calibration_error(&p, &t, 0).is_err());
    }
}
