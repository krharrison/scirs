//! SWAG: Stochastic Weight Averaging Gaussian.
//!
//! SWAG approximates the posterior over neural network weights by collecting
//! weight snapshots during SGD training and fitting a Gaussian with a
//! diagonal + low-rank covariance structure:
//!
//!   Sigma_SWAG = (1/2)(Sigma_diag + Sigma_low_rank)
//!
//! where Sigma_diag = diag(mean(theta^2) - mean(theta)^2) and
//! Sigma_low_rank = (1/(K-1)) D D^T with D = [theta_1 - theta_bar, ...].

use scirs2_core::ndarray::{Array1, Array2};

use crate::error::{StatsError, StatsResult};

use super::types::{BNNConfig, BNNPosterior, CovarianceType, PredictiveDistribution};

/// Xorshift64 PRNG for sampling without external dependencies.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Generate a standard normal sample using Box-Muller transform.
fn randn(state: &mut u64) -> f64 {
    let u1 = (xorshift64(state) as f64) / (u64::MAX as f64);
    let u2 = (xorshift64(state) as f64) / (u64::MAX as f64);
    let u1_clamped = u1.max(1e-300); // avoid log(0)
    (-2.0 * u1_clamped.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Collector that accumulates weight snapshots during SGD training
/// and builds a SWAG posterior.
#[derive(Debug, Clone)]
pub struct SWAGCollector {
    /// Running mean of weights
    mean: Array1<f64>,
    /// Running mean of squared weights
    sq_mean: Array1<f64>,
    /// Low-rank deviation columns (theta_i - theta_bar), up to swag_rank
    deviations: Vec<Array1<f64>>,
    /// Number of snapshots collected so far
    n_collected: usize,
    /// Configuration
    config: BNNConfig,
    /// Number of parameters
    n_params: usize,
}

impl SWAGCollector {
    /// Create a new SWAG collector.
    ///
    /// # Arguments
    /// * `n_params` - Number of neural network parameters
    /// * `config` - BNN configuration (swag_rank controls max deviation columns)
    pub fn new(n_params: usize, config: &BNNConfig) -> Self {
        Self {
            mean: Array1::zeros(n_params),
            sq_mean: Array1::zeros(n_params),
            deviations: Vec::new(),
            n_collected: 0,
            config: config.clone(),
            n_params,
        }
    }

    /// Collect a weight snapshot during training.
    ///
    /// Updates the running mean and squared mean, and stores a low-rank
    /// deviation vector. If more than `swag_rank` deviations have been
    /// collected, the oldest deviation is dropped (FIFO).
    ///
    /// # Arguments
    /// * `weights` - Current weight vector, length must equal n_params
    pub fn collect(&mut self, weights: &Array1<f64>) {
        let n = self.n_collected as f64;
        let n1 = n + 1.0;

        // Online update: mean = (n * mean + w) / (n + 1)
        self.mean = &self.mean * (n / n1) + &(weights * (1.0 / n1));
        // sq_mean = (n * sq_mean + w^2) / (n + 1)
        let w_sq = weights.mapv(|w| w * w);
        self.sq_mean = &self.sq_mean * (n / n1) + &(&w_sq * (1.0 / n1));

        self.n_collected += 1;

        // Store deviation: we use a temporary placeholder; the actual
        // deviation (w - mean) will be computed at build time using final mean.
        // However, SWAG paper stores raw snapshots and computes deviations
        // from the running mean at collection time. We store the raw weights
        // and compute deviations in build_posterior.
        // For memory efficiency with the FIFO behavior, store w directly.
        if self.deviations.len() < self.config.swag_rank {
            self.deviations.push(weights.clone());
        } else {
            // FIFO: remove oldest, push newest
            self.deviations.remove(0);
            self.deviations.push(weights.clone());
        }
    }

    /// Build the posterior from collected snapshots.
    ///
    /// The SWAG posterior has covariance:
    ///   Sigma_SWAG = (1/2)(Sigma_diag + Sigma_low_rank)
    ///
    /// where:
    /// - Sigma_diag = diag(mean(theta^2) - mean(theta)^2)
    /// - Sigma_low_rank = (1/(K-1)) D D^T
    /// - D = \[theta_1 - theta_bar, ..., theta_K - theta_bar\]
    ///
    /// # Errors
    /// Returns an error if fewer than 2 snapshots have been collected.
    pub fn build_posterior(&self) -> StatsResult<BNNPosterior> {
        if self.n_collected < 2 {
            return Err(StatsError::invalid_argument(
                "SWAG requires at least 2 weight snapshots",
            ));
        }

        // Diagonal variance: var = E[w^2] - E[w]^2, clamped to non-negative
        let diag_var = &self.sq_mean - &self.mean.mapv(|m| m * m);
        let diag_var = diag_var.mapv(|v| v.max(0.0));

        // Build deviation matrix from stored snapshots
        let k = self.deviations.len();
        let mut deviation = Array2::<f64>::zeros((self.n_params, k));
        for (col_idx, snapshot) in self.deviations.iter().enumerate() {
            for row_idx in 0..self.n_params {
                deviation[[row_idx, col_idx]] = snapshot[row_idx] - self.mean[row_idx];
            }
        }

        let covariance_type = CovarianceType::LowRankPlusDiagonal {
            d_diag: diag_var,
            deviation,
        };

        // Approximate log marginal likelihood (not well-defined for SWAG,
        // use negative training loss as proxy)
        let log_marginal = 0.0;

        Ok(BNNPosterior {
            mean: self.mean.clone(),
            covariance_type,
            log_marginal_likelihood: log_marginal,
        })
    }

    /// Sample weights from the SWAG posterior.
    ///
    /// theta ~ N(theta_bar, (1/2)(Sigma_diag + Sigma_low_rank))
    ///       = theta_bar + (1/sqrt(2)) * (Sigma_diag^{1/2} z_1 + (1/sqrt(K-1)) D z_2)
    ///
    /// where z_1 ~ N(0, I_d), z_2 ~ N(0, I_K).
    ///
    /// # Errors
    /// Returns an error if fewer than 2 snapshots have been collected.
    pub fn sample_weights(&self, rng_state: &mut u64) -> StatsResult<Array1<f64>> {
        if self.n_collected < 2 {
            return Err(StatsError::invalid_argument(
                "SWAG requires at least 2 snapshots to sample",
            ));
        }

        // Diagonal variance
        let diag_var = &self.sq_mean - &self.mean.mapv(|m| m * m);
        let diag_std = diag_var.mapv(|v| v.max(0.0).sqrt());

        // z_1 ~ N(0, I_d)
        let z1: Array1<f64> = Array1::from_shape_fn(self.n_params, |_| randn(rng_state));

        // Diagonal contribution: Sigma_diag^{1/2} z_1
        let diag_part = &diag_std * &z1;

        // Low-rank contribution
        let k = self.deviations.len();
        let k_minus_1 = if k > 1 { (k - 1) as f64 } else { 1.0 };

        // Build deviation vectors
        let mut lr_part = Array1::zeros(self.n_params);
        if k > 0 {
            let z2: Array1<f64> = Array1::from_shape_fn(k, |_| randn(rng_state));
            for (col_idx, snapshot) in self.deviations.iter().enumerate() {
                let dev = snapshot - &self.mean;
                lr_part = lr_part + &(&dev * z2[col_idx]);
            }
            lr_part /= k_minus_1.sqrt();
        }

        // Combine: theta = theta_bar + (1/sqrt(2)) * (diag_part + lr_part)
        let scale = 1.0 / 2.0_f64.sqrt();
        let sample = &self.mean + &((&diag_part + &lr_part) * scale);

        Ok(sample)
    }

    /// Return the number of snapshots collected so far.
    pub fn n_collected(&self) -> usize {
        self.n_collected
    }

    /// Return the current running mean.
    pub fn mean(&self) -> &Array1<f64> {
        &self.mean
    }
}

/// Multi-SWAG: ensemble of independent SWAG models.
///
/// Combines predictions from multiple SWAG posteriors for improved
/// uncertainty estimation.
///
/// # Arguments
/// * `models` - Slice of trained SWAG collectors
/// * `predict_fn` - Function that maps weights to predictions
/// * `n_samples_per_model` - Number of MC samples per SWAG model
/// * `rng_state` - Mutable PRNG state
///
/// # Errors
/// Returns an error if models is empty or if any model has insufficient snapshots.
pub fn multi_swag_predict(
    models: &[SWAGCollector],
    predict_fn: &dyn Fn(&Array1<f64>) -> StatsResult<Array1<f64>>,
    n_samples_per_model: usize,
    rng_state: &mut u64,
) -> StatsResult<PredictiveDistribution> {
    if models.is_empty() {
        return Err(StatsError::invalid_argument("Need at least one SWAG model"));
    }
    if n_samples_per_model == 0 {
        return Err(StatsError::invalid_argument(
            "Need at least 1 sample per model",
        ));
    }

    let total_samples = models.len() * n_samples_per_model;
    let mut all_predictions: Vec<Array1<f64>> = Vec::with_capacity(total_samples);

    for model in models {
        for _ in 0..n_samples_per_model {
            let w = model.sample_weights(rng_state)?;
            let pred = predict_fn(&w)?;
            all_predictions.push(pred);
        }
    }

    // Stack predictions and compute mean/variance
    let n_outputs = all_predictions[0].len();
    let n_total = all_predictions.len();

    let mut mean = Array1::zeros(n_outputs);
    for p in &all_predictions {
        mean = mean + p;
    }
    mean /= n_total as f64;

    let mut variance = Array1::zeros(n_outputs);
    for p in &all_predictions {
        let diff = p - &mean;
        variance = variance + &diff.mapv(|d| d * d);
    }
    variance /= n_total as f64;

    // Build samples matrix [n_total x n_outputs]
    let mut samples = Array2::zeros((n_total, n_outputs));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> BNNConfig {
        BNNConfig {
            swag_rank: 5,
            ..BNNConfig::default()
        }
    }

    #[test]
    fn test_swag_collector_mean_converges() {
        let config = make_config();
        let mut collector = SWAGCollector::new(3, &config);

        // Collect snapshots centered around [1, 2, 3]
        let target = array![1.0, 2.0, 3.0];
        let mut rng: u64 = 42;
        for _ in 0..100 {
            let noise = Array1::from_shape_fn(3, |_| randn(&mut rng) * 0.01);
            collector.collect(&(&target + &noise));
        }

        let mean = collector.mean();
        for i in 0..3 {
            assert!(
                (mean[i] - target[i]).abs() < 0.1,
                "Mean[{}] = {}, expected ~{}",
                i,
                mean[i],
                target[i]
            );
        }
    }

    use scirs2_core::ndarray::array;

    #[test]
    fn test_swag_deviations_stored() {
        let config = BNNConfig {
            swag_rank: 3,
            ..BNNConfig::default()
        };
        let mut collector = SWAGCollector::new(2, &config);

        collector.collect(&array![1.0, 2.0]);
        collector.collect(&array![3.0, 4.0]);
        collector.collect(&array![5.0, 6.0]);

        assert_eq!(collector.deviations.len(), 3);

        // Collect one more, should drop oldest (FIFO)
        collector.collect(&array![7.0, 8.0]);
        assert_eq!(collector.deviations.len(), 3);
    }

    #[test]
    fn test_swag_build_posterior() {
        let config = make_config();
        let mut collector = SWAGCollector::new(2, &config);

        collector.collect(&array![1.0, 2.0]);
        collector.collect(&array![3.0, 4.0]);
        collector.collect(&array![5.0, 6.0]);

        let posterior = collector.build_posterior().expect("build posterior");
        assert_eq!(posterior.mean.len(), 2);

        match &posterior.covariance_type {
            CovarianceType::LowRankPlusDiagonal { d_diag, deviation } => {
                assert_eq!(d_diag.len(), 2);
                assert_eq!(deviation.nrows(), 2);
                assert_eq!(deviation.ncols(), 3);
                // Diagonal variance should be non-negative
                for &v in d_diag.iter() {
                    assert!(v >= 0.0, "Diagonal variance should be >= 0, got {}", v);
                }
            }
            _ => panic!("Expected LowRankPlusDiagonal covariance"),
        }
    }

    #[test]
    fn test_swag_sample_correct_dimension() {
        let config = make_config();
        let mut collector = SWAGCollector::new(4, &config);

        for i in 0..5 {
            let w = Array1::from_shape_fn(4, |j| (i * 4 + j) as f64);
            collector.collect(&w);
        }

        let mut rng: u64 = 123;
        let sample = collector.sample_weights(&mut rng).expect("sample");
        assert_eq!(sample.len(), 4);
    }

    #[test]
    fn test_swag_insufficient_snapshots() {
        let config = make_config();
        let mut collector = SWAGCollector::new(2, &config);
        collector.collect(&array![1.0, 2.0]);

        assert!(collector.build_posterior().is_err());
        let mut rng: u64 = 1;
        assert!(collector.sample_weights(&mut rng).is_err());
    }

    #[test]
    fn test_multi_swag_predict() {
        let config = make_config();
        let mut c1 = SWAGCollector::new(2, &config);
        let mut c2 = SWAGCollector::new(2, &config);

        for i in 0..5 {
            let w = array![i as f64, (i as f64) * 0.5];
            c1.collect(&w);
            c2.collect(&(&w + &array![0.1, 0.1]));
        }

        let predict_fn = |w: &Array1<f64>| -> StatsResult<Array1<f64>> { Ok(array![w[0] + w[1]]) };

        let mut rng: u64 = 42;
        let result = multi_swag_predict(&[c1, c2], &predict_fn, 5, &mut rng).expect("multi swag");

        assert_eq!(result.mean.len(), 1);
        assert_eq!(result.variance.len(), 1);
        assert!(result.samples.is_some());
        let samples = result.samples.as_ref().expect("samples should exist");
        assert_eq!(samples.nrows(), 10); // 2 models * 5 samples
    }

    #[test]
    fn test_multi_swag_empty_models() {
        let models: Vec<SWAGCollector> = vec![];
        let predict_fn = |_w: &Array1<f64>| -> StatsResult<Array1<f64>> { Ok(array![0.0]) };
        let mut rng: u64 = 1;
        assert!(multi_swag_predict(&models, &predict_fn, 5, &mut rng).is_err());
    }
}
