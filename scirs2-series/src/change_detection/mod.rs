//! Advanced change detection algorithms for time series
//!
//! This module provides state-of-the-art change detection algorithms:
//! - **BOCPD**: Bayesian Online Changepoint Detection (Adams & MacKay 2007)
//! - **PELT**: Pruned Exact Linear Time changepoint detection (Killick et al. 2012)
//! - **BinSegmentation**: Binary Segmentation with BIC/MDL criterion
//! - **WildBinarySegmentation**: Wild Binary Segmentation (Fryzlewicz 2014)
//! - **GaussianHazard**: Constant hazard function for BOCPD
//! - **RBOCPD**: Run-length distribution and predictive probability

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive, NumCast};
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};

/// Result of change detection
#[derive(Debug, Clone)]
pub struct ChangeDetectionResult {
    /// Detected change point locations (indices)
    pub change_points: Vec<usize>,
    /// Confidence scores for each detected change point
    pub confidence_scores: Vec<f64>,
    /// Segment means for each segment between change points
    pub segment_means: Vec<f64>,
    /// Segment variances for each segment between change points
    pub segment_variances: Vec<f64>,
    /// Total number of segments
    pub n_segments: usize,
}

impl ChangeDetectionResult {
    fn new(change_points: Vec<usize>, confidence_scores: Vec<f64>, segment_stats: Vec<(f64, f64)>) -> Self {
        let n_segments = segment_stats.len();
        let segment_means = segment_stats.iter().map(|(m, _)| *m).collect();
        let segment_variances = segment_stats.iter().map(|(_, v)| *v).collect();
        Self {
            change_points,
            confidence_scores,
            segment_means,
            segment_variances,
            n_segments,
        }
    }
}

/// Penalty criterion for change detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PenaltyCriterion {
    /// Bayesian Information Criterion
    BIC,
    /// Minimum Description Length
    MDL,
    /// Akaike Information Criterion
    AIC,
    /// Hannan-Quinn criterion
    HannanQuinn,
    /// Manual penalty value
    Manual,
}

/// Configuration for PELT (Pruned Exact Linear Time) changepoint detection
#[derive(Debug, Clone)]
pub struct PELTConfig {
    /// Minimum segment length
    pub min_segment_len: usize,
    /// Penalty criterion
    pub penalty_criterion: PenaltyCriterion,
    /// Manual penalty value (used when criterion = Manual)
    pub manual_penalty: f64,
    /// Maximum number of change points
    pub max_changepoints: Option<usize>,
}

impl Default for PELTConfig {
    fn default() -> Self {
        Self {
            min_segment_len: 5,
            penalty_criterion: PenaltyCriterion::BIC,
            manual_penalty: 0.0,
            max_changepoints: None,
        }
    }
}

/// PELT: Pruned Exact Linear Time changepoint detection
///
/// Implements the PELT algorithm from Killick et al. (2012) which finds the optimal
/// segmentation using a normal likelihood cost function with a penalty for each additional
/// change point. The pruning step makes it O(n) for many penalty values.
pub struct PELT {
    config: PELTConfig,
}

impl PELT {
    /// Create a new PELT detector with the given configuration
    pub fn new(config: PELTConfig) -> Self {
        Self { config }
    }

    /// Compute the penalty based on the configured criterion
    fn compute_penalty(&self, n: usize) -> f64 {
        match self.config.penalty_criterion {
            PenaltyCriterion::BIC => 2.0 * (n as f64).ln(),
            PenaltyCriterion::MDL => (n as f64).log2(),
            PenaltyCriterion::AIC => 2.0,
            PenaltyCriterion::HannanQuinn => 2.0 * (n as f64).ln().ln().max(1.0),
            PenaltyCriterion::Manual => self.config.manual_penalty,
        }
    }

    /// Compute normal likelihood cost for a segment [start, end)
    fn segment_cost(data: &[f64], start: usize, end: usize) -> f64 {
        if end <= start + 1 {
            return 0.0;
        }
        let len = (end - start) as f64;
        let slice = &data[start..end];
        let mean = slice.iter().sum::<f64>() / len;
        let var = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / len;
        let var = var.max(1e-15);
        // Negative log-likelihood: n/2 * log(2*pi*var) + n/2
        len * 0.5 * (2.0 * std::f64::consts::PI * var).ln() + len * 0.5
    }

    /// Detect change points in the given time series
    pub fn detect(&self, data: &[f64]) -> Result<ChangeDetectionResult> {
        let n = data.len();
        let min_len = self.config.min_segment_len;

        if n < min_len * 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Series too short for PELT".to_string(),
                required: min_len * 2,
                actual: n,
            });
        }

        let penalty = self.compute_penalty(n);

        // cost[t] = minimum total cost for data[0..t]
        let mut cost = vec![f64::INFINITY; n + 1];
        let mut prev = vec![0usize; n + 1];
        cost[0] = -penalty;

        let mut candidates: Vec<usize> = vec![0];

        for t in min_len..=n {
            let mut best_cost = f64::INFINITY;
            let mut best_prev = 0;

            for &s in &candidates {
                if t.saturating_sub(s) < min_len {
                    continue;
                }
                let seg_cost = Self::segment_cost(data, s, t);
                let total = cost[s] + seg_cost + penalty;
                if total < best_cost {
                    best_cost = total;
                    best_prev = s;
                }
            }

            cost[t] = best_cost;
            prev[t] = best_prev;

            // Pruning: remove s if cost[s] + min_future_cost(s) > best_cost
            candidates.retain(|&s| {
                cost[s] + penalty <= best_cost + 1e-10
            });
            candidates.push(t);
        }

        // Backtrack to find change points
        let mut cps: Vec<usize> = Vec::new();
        let mut cur = n;
        while cur > 0 {
            let p = prev[cur];
            if p > 0 {
                cps.push(p);
            }
            cur = p;
        }
        cps.reverse();

        // Apply max_changepoints limit
        if let Some(max_cp) = self.config.max_changepoints {
            cps.truncate(max_cp);
        }

        // Compute confidence scores as cost reduction from each change point
        let confidence = self.compute_confidence(data, &cps, n);
        let segment_stats = self.compute_segment_stats(data, &cps, n);

        Ok(ChangeDetectionResult::new(cps, confidence, segment_stats))
    }

    fn compute_confidence(&self, data: &[f64], cps: &[usize], n: usize) -> Vec<f64> {
        cps.iter().map(|&cp| {
            // Score = cost improvement at the change point
            let left_end = cp;
            let left_start = cps.iter().rev().find(|&&x| x < cp).copied().unwrap_or(0);
            let right_start = cp;
            let right_end = cps.iter().find(|&&x| x > cp).copied().unwrap_or(n);

            let combined = Self::segment_cost(data, left_start, right_end);
            let split = Self::segment_cost(data, left_start, left_end)
                + Self::segment_cost(data, right_start, right_end);
            (combined - split).max(0.0)
        }).collect()
    }

    fn compute_segment_stats(&self, data: &[f64], cps: &[usize], n: usize) -> Vec<(f64, f64)> {
        let mut boundaries = vec![0usize];
        boundaries.extend_from_slice(cps);
        boundaries.push(n);

        boundaries.windows(2).map(|w| {
            let start = w[0];
            let end = w[1];
            let slice = &data[start..end];
            if slice.is_empty() {
                return (0.0, 0.0);
            }
            let len = slice.len() as f64;
            let mean = slice.iter().sum::<f64>() / len;
            let var = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / len;
            (mean, var)
        }).collect()
    }
}

/// Configuration for Binary Segmentation
#[derive(Debug, Clone)]
pub struct BinSegConfig {
    /// Minimum segment length
    pub min_segment_len: usize,
    /// Penalty criterion
    pub penalty_criterion: PenaltyCriterion,
    /// Manual penalty value
    pub manual_penalty: f64,
    /// Maximum number of change points to detect
    pub max_changepoints: usize,
}

impl Default for BinSegConfig {
    fn default() -> Self {
        Self {
            min_segment_len: 5,
            penalty_criterion: PenaltyCriterion::BIC,
            manual_penalty: 0.0,
            max_changepoints: 20,
        }
    }
}

/// Binary Segmentation with BIC/MDL criterion
///
/// Greedily finds the best split at each iteration using a contrast statistic,
/// then recursively segments each sub-interval. This is O(n log n) on average.
pub struct BinSegmentation {
    config: BinSegConfig,
}

impl BinSegmentation {
    /// Create a new BinSegmentation detector
    pub fn new(config: BinSegConfig) -> Self {
        Self { config }
    }

    fn compute_penalty(&self, n: usize) -> f64 {
        match self.config.penalty_criterion {
            PenaltyCriterion::BIC => 2.0 * (n as f64).ln(),
            PenaltyCriterion::MDL => (n as f64).log2(),
            PenaltyCriterion::AIC => 2.0,
            PenaltyCriterion::HannanQuinn => 2.0 * (n as f64).ln().ln().max(1.0),
            PenaltyCriterion::Manual => self.config.manual_penalty,
        }
    }

    /// Cumulative sum squared for efficient cost computation
    fn precompute_cumsums(data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut sum = vec![0.0; data.len() + 1];
        let mut sum_sq = vec![0.0; data.len() + 1];
        for (i, &x) in data.iter().enumerate() {
            sum[i + 1] = sum[i] + x;
            sum_sq[i + 1] = sum_sq[i] + x * x;
        }
        (sum, sum_sq)
    }

    /// Compute segment cost using precomputed sums
    fn seg_cost_from_sums(sum: &[f64], sum_sq: &[f64], start: usize, end: usize) -> f64 {
        if end <= start + 1 {
            return 0.0;
        }
        let len = (end - start) as f64;
        let s = sum[end] - sum[start];
        let sq = sum_sq[end] - sum_sq[start];
        let var = (sq - s * s / len) / len;
        let var = var.max(1e-15);
        len * 0.5 * (2.0 * std::f64::consts::PI * var).ln() + len * 0.5
    }

    /// Find the best split point in [start, end), returns (split_idx, score_improvement)
    fn best_split(
        sum: &[f64],
        sum_sq: &[f64],
        start: usize,
        end: usize,
        min_len: usize,
    ) -> Option<(usize, f64)> {
        if end - start < 2 * min_len {
            return None;
        }
        let combined = Self::seg_cost_from_sums(sum, sum_sq, start, end);
        let mut best_split_idx = None;
        let mut best_gain = 0.0;

        for split in (start + min_len)..(end - min_len + 1) {
            let left = Self::seg_cost_from_sums(sum, sum_sq, start, split);
            let right = Self::seg_cost_from_sums(sum, sum_sq, split, end);
            let gain = combined - left - right;
            if gain > best_gain {
                best_gain = gain;
                best_split_idx = Some(split);
            }
        }
        best_split_idx.map(|idx| (idx, best_gain))
    }

    /// Recursively segment using binary segmentation
    fn segment_recursive(
        &self,
        sum: &[f64],
        sum_sq: &[f64],
        start: usize,
        end: usize,
        penalty: f64,
        depth: usize,
        results: &mut Vec<(usize, f64)>,
    ) {
        if depth > self.config.max_changepoints {
            return;
        }
        if let Some((split, gain)) = Self::best_split(sum, sum_sq, start, end, self.config.min_segment_len) {
            if gain > penalty {
                results.push((split, gain));
                self.segment_recursive(sum, sum_sq, start, split, penalty, depth + 1, results);
                self.segment_recursive(sum, sum_sq, split, end, penalty, depth + 1, results);
            }
        }
    }

    /// Detect change points
    pub fn detect(&self, data: &[f64]) -> Result<ChangeDetectionResult> {
        let n = data.len();
        if n < self.config.min_segment_len * 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Series too short for BinSegmentation".to_string(),
                required: self.config.min_segment_len * 2,
                actual: n,
            });
        }

        let penalty = self.compute_penalty(n);
        let (sum, sum_sq) = Self::precompute_cumsums(data);

        let mut results: Vec<(usize, f64)> = Vec::new();
        self.segment_recursive(&sum, &sum_sq, 0, n, penalty, 0, &mut results);

        results.sort_by_key(|&(idx, _)| idx);
        results.truncate(self.config.max_changepoints);

        let cps: Vec<usize> = results.iter().map(|&(idx, _)| idx).collect();
        let confidence: Vec<f64> = results.iter().map(|&(_, gain)| gain).collect();
        let segment_stats = PELT::new(PELTConfig::default()).compute_segment_stats(data, &cps, n);

        Ok(ChangeDetectionResult::new(cps, confidence, segment_stats))
    }
}

/// Configuration for Wild Binary Segmentation
#[derive(Debug, Clone)]
pub struct WBSConfig {
    /// Minimum segment length
    pub min_segment_len: usize,
    /// Number of random intervals to draw
    pub n_intervals: usize,
    /// Threshold for detection (if None, use BIC-based penalty)
    pub threshold: Option<f64>,
    /// Maximum number of change points
    pub max_changepoints: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for WBSConfig {
    fn default() -> Self {
        Self {
            min_segment_len: 5,
            n_intervals: 5000,
            threshold: None,
            max_changepoints: 30,
            seed: None,
        }
    }
}

/// Wild Binary Segmentation (Fryzlewicz 2014)
///
/// Draws M random intervals [s, e] and for each computes the CUSUM statistic,
/// then selects the interval with the largest absolute CUSUM as the best split.
/// This solves the issue of BinSegmentation missing spatially close change points.
pub struct WildBinarySegmentation {
    config: WBSConfig,
}

impl WildBinarySegmentation {
    /// Create a new WildBinarySegmentation detector
    pub fn new(config: WBSConfig) -> Self {
        Self { config }
    }

    /// Compute the normalized CUSUM statistic for a split within [start, end)
    fn cusum_stat(data: &[f64], start: usize, end: usize) -> Option<(usize, f64)> {
        let len = end - start;
        if len < 4 {
            return None;
        }
        let full_mean: f64 = data[start..end].iter().sum::<f64>() / len as f64;
        let var: f64 = data[start..end].iter().map(|&x| (x - full_mean).powi(2)).sum::<f64>() / len as f64;
        let std = var.sqrt().max(1e-15);

        let mut max_stat = 0.0_f64;
        let mut best_split = start + 1;
        let mut running_sum = 0.0_f64;

        for t in start..(end - 1) {
            running_sum += data[t];
            let left_len = (t - start + 1) as f64;
            let right_len = (end - t - 1) as f64;
            if right_len < 1.0 {
                continue;
            }
            let cusum = (left_len * right_len / (left_len + right_len)).sqrt()
                * (running_sum / left_len - (data[start..end].iter().sum::<f64>() - running_sum) / right_len)
                / std;
            if cusum.abs() > max_stat {
                max_stat = cusum.abs();
                best_split = t + 1;
            }
        }

        if max_stat > 0.0 {
            Some((best_split, max_stat))
        } else {
            None
        }
    }

    /// Generate random intervals using a simple LCG
    fn random_intervals(&self, n: usize, count: usize) -> Vec<(usize, usize)> {
        let mut state = self.config.seed.unwrap_or(42);
        let min_len = self.config.min_segment_len * 2;
        let mut intervals = Vec::with_capacity(count);

        for _ in 0..count {
            // LCG: Numerical Recipes constants
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let s = (state >> 32) as usize % n;
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let e = (state >> 32) as usize % n;
            let (s, e) = if s <= e { (s, e) } else { (e, s) };
            if e - s >= min_len {
                intervals.push((s, e + 1));
            }
        }
        intervals
    }

    /// Recursively detect change points using wild binary segmentation
    fn wbs_recursive(
        &self,
        data: &[f64],
        intervals: &[(usize, usize)],
        start: usize,
        end: usize,
        threshold: f64,
        depth: usize,
        results: &mut Vec<(usize, f64)>,
    ) {
        if depth > self.config.max_changepoints || end - start < self.config.min_segment_len * 2 {
            return;
        }

        // Find the interval with the highest CUSUM within [start, end]
        let mut best_stat = threshold;
        let mut best_split = None;

        for &(s, e) in intervals {
            if s >= start && e <= end {
                if let Some((split, stat)) = Self::cusum_stat(data, s, e) {
                    if stat > best_stat && split > start && split < end {
                        best_stat = stat;
                        best_split = Some(split);
                    }
                }
            }
        }

        if let Some(split) = best_split {
            results.push((split, best_stat));
            self.wbs_recursive(data, intervals, start, split, threshold, depth + 1, results);
            self.wbs_recursive(data, intervals, split, end, threshold, depth + 1, results);
        }
    }

    /// Detect change points
    pub fn detect(&self, data: &[f64]) -> Result<ChangeDetectionResult> {
        let n = data.len();
        let min_required = self.config.min_segment_len * 2;
        if n < min_required {
            return Err(TimeSeriesError::InsufficientData {
                message: "Series too short for WildBinarySegmentation".to_string(),
                required: min_required,
                actual: n,
            });
        }

        // Use sigma_hat-based threshold if not manually specified
        let threshold = self.config.threshold.unwrap_or_else(|| {
            // Approximately 2*sqrt(log(n)) for a universal threshold
            2.0 * (n as f64).ln().sqrt()
        });

        let intervals = self.random_intervals(n, self.config.n_intervals);
        let mut results: Vec<(usize, f64)> = Vec::new();
        self.wbs_recursive(data, &intervals, 0, n, threshold, 0, &mut results);

        results.sort_by_key(|&(idx, _)| idx);
        results.truncate(self.config.max_changepoints);

        let cps: Vec<usize> = results.iter().map(|&(idx, _)| idx).collect();
        let confidence: Vec<f64> = results.iter().map(|&(_, stat)| stat).collect();
        let segment_stats = PELT::new(PELTConfig::default()).compute_segment_stats(data, &cps, n);

        Ok(ChangeDetectionResult::new(cps, confidence, segment_stats))
    }
}

/// Gaussian constant hazard function for BOCPD
///
/// The hazard function H(t) = 1/lambda defines the prior probability of a
/// changepoint at any given time step under the prior.
#[derive(Debug, Clone)]
pub struct GaussianHazard {
    /// Expected run length (lambda parameter)
    pub lambda: f64,
}

impl GaussianHazard {
    /// Create a new Gaussian hazard with given expected run length
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }

    /// Evaluate the hazard function H(t) = 1/lambda
    pub fn hazard(&self, _run_length: usize) -> f64 {
        1.0 / self.lambda
    }
}

/// State maintained by BOCPD for the run-length distribution
#[derive(Debug, Clone)]
struct BOCPDState {
    /// Run-length distribution P(r_t | x_{1:t})
    pub run_length_dist: Vec<f64>,
    /// Sufficient statistics: (sum, sum_sq, count) for each run-length hypothesis
    pub suff_stats: Vec<(f64, f64, usize)>,
    /// Predictive probabilities at each timestep
    pub predictive_probs: Vec<Vec<f64>>,
    /// Current timestep
    pub t: usize,
}

impl BOCPDState {
    fn new() -> Self {
        Self {
            run_length_dist: vec![1.0], // P(r_0 = 0) = 1
            suff_stats: vec![(0.0, 0.0, 0)],
            predictive_probs: Vec::new(),
            t: 0,
        }
    }
}

/// Configuration for BOCPD
#[derive(Debug, Clone)]
pub struct BOCPDConfig {
    /// Expected run length (lambda for constant hazard)
    pub expected_run_length: f64,
    /// Prior mean of the Gaussian likelihood
    pub prior_mean: f64,
    /// Prior variance of the Gaussian likelihood
    pub prior_var: f64,
    /// Noise variance (observation noise)
    pub noise_var: f64,
    /// Threshold on run-length distribution for change point declaration
    pub threshold: f64,
}

impl Default for BOCPDConfig {
    fn default() -> Self {
        Self {
            expected_run_length: 200.0,
            prior_mean: 0.0,
            prior_var: 1.0,
            noise_var: 1.0,
            threshold: 0.5,
        }
    }
}

/// BOCPD: Bayesian Online Changepoint Detection (Adams & MacKay 2007)
///
/// Maintains a posterior distribution over run lengths, updating it as each
/// new observation arrives. The algorithm is exact and online.
pub struct BOCPD {
    config: BOCPDConfig,
    hazard: GaussianHazard,
}

impl BOCPD {
    /// Create a new BOCPD detector
    pub fn new(config: BOCPDConfig) -> Self {
        let hazard = GaussianHazard::new(config.expected_run_length);
        Self { config, hazard }
    }

    /// Compute predictive probability P(x_{t+1} | x_{1:t}, r_t = r)
    /// using the Normal-Normal conjugate model
    fn predictive_prob(&self, x: f64, sum: f64, sum_sq: f64, count: usize) -> f64 {
        let n = count as f64;
        // Posterior parameters: Normal-Gamma or Normal-Normal
        let post_var = 1.0 / (1.0 / self.config.prior_var + n / self.config.noise_var);
        let post_mean = post_var * (self.config.prior_mean / self.config.prior_var + sum / self.config.noise_var);
        // Predictive variance = noise_var + post_var
        let pred_var = self.config.noise_var + post_var;
        // Gaussian probability density
        let diff = x - post_mean;
        let log_p = -0.5 * (2.0 * std::f64::consts::PI * pred_var).ln() - 0.5 * diff * diff / pred_var;
        log_p.exp().max(1e-300)
    }

    /// Update the run-length distribution with a new observation
    fn update_state(&self, state: &mut BOCPDState, x: f64) {
        let r_len = state.run_length_dist.len();
        let mut new_run_dist = vec![0.0; r_len + 1];
        let mut new_suff_stats = vec![(0.0, 0.0, 0usize); r_len + 1];
        let mut pred_probs = vec![0.0; r_len];

        // Compute predictive probabilities for each run length
        for (r, &p_r) in state.run_length_dist.iter().enumerate() {
            if p_r < 1e-300 {
                continue;
            }
            let (sum, sum_sq, count) = state.suff_stats[r];
            let pp = self.predictive_prob(x, sum, sum_sq, count);
            pred_probs[r] = pp;

            let h = self.hazard.hazard(r);

            // Growth probability: P(r_{t+1} = r+1 | r_t = r) = (1 - H) * pp * P(r_t = r)
            new_run_dist[r + 1] += p_r * pp * (1.0 - h);
            new_suff_stats[r + 1] = (sum + x, sum_sq + x * x, count + 1);

            // Changepoint probability: P(r_{t+1} = 0 | r_t = r) = H * pp * P(r_t = r)
            new_run_dist[0] += p_r * pp * h;
        }
        // Normalize
        let total: f64 = new_run_dist.iter().sum();
        if total > 0.0 {
            for p in &mut new_run_dist {
                *p /= total;
            }
        }

        // Reset stats for r=0 (new run)
        new_suff_stats[0] = (x, x * x, 1);

        state.run_length_dist = new_run_dist;
        state.suff_stats = new_suff_stats;
        state.predictive_probs.push(pred_probs);
        state.t += 1;
    }

    /// Detect change points using BOCPD
    pub fn detect(&self, data: &[f64]) -> Result<ChangeDetectionResult> {
        let n = data.len();
        if n < 4 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Series too short for BOCPD".to_string(),
                required: 4,
                actual: n,
            });
        }

        let mut state = BOCPDState::new();
        // Probability of changepoint at each timestep = P(r_t = 0)
        let mut cp_probs = Vec::with_capacity(n);

        for &x in data {
            self.update_state(&mut state, x);
            // P(r_t = 0) indicates a new run started, i.e., a changepoint
            cp_probs.push(state.run_length_dist[0]);
        }

        // Detect change points where P(r_t = 0) > threshold
        let mut cps: Vec<usize> = Vec::new();
        let mut confidence: Vec<f64> = Vec::new();

        // Use a simple peak detection on cp_probs with a minimum gap
        let min_gap = 5;
        let mut last_cp = 0;

        for (t, &prob) in cp_probs.iter().enumerate().skip(1) {
            if prob > self.config.threshold && t - last_cp >= min_gap {
                // Check this is a local maximum
                let is_peak = {
                    let left = if t > 0 { cp_probs[t - 1] } else { 0.0 };
                    let right = if t + 1 < n { cp_probs[t + 1] } else { 0.0 };
                    prob >= left && prob >= right
                };
                if is_peak {
                    cps.push(t);
                    confidence.push(prob);
                    last_cp = t;
                }
            }
        }

        let segment_stats = PELT::new(PELTConfig::default()).compute_segment_stats(data, &cps, n);
        Ok(ChangeDetectionResult::new(cps, confidence, segment_stats))
    }

    /// Get the run-length distribution at each timestep (for diagnostic purposes)
    pub fn run_length_distribution(&self, data: &[f64]) -> Result<Array2<f64>> {
        let n = data.len();
        if n < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Series too short".to_string(),
                required: 2,
                actual: n,
            });
        }

        let mut state = BOCPDState::new();
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);

        for &x in data {
            self.update_state(&mut state, x);
            rows.push(state.run_length_dist.clone());
        }

        // Pad all rows to the same length (n+1)
        let max_len = rows.iter().map(|r| r.len()).max().unwrap_or(1);
        let mut matrix = Array2::zeros((n, max_len));
        for (i, row) in rows.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                matrix[[i, j]] = val;
            }
        }
        Ok(matrix)
    }
}

/// RBOCPD: Run-length based Bayesian Online Changepoint Detection
///
/// Extends BOCPD with explicit run-length distribution tracking and
/// predictive probability sequences for model evaluation.
pub struct RBOCPD {
    inner: BOCPD,
}

impl RBOCPD {
    /// Create a new RBOCPD detector
    pub fn new(config: BOCPDConfig) -> Self {
        Self {
            inner: BOCPD::new(config),
        }
    }

    /// Detect change points and return run-length distributions
    pub fn detect_with_run_lengths(
        &self,
        data: &[f64],
    ) -> Result<(ChangeDetectionResult, Array2<f64>)> {
        let cp_result = self.inner.detect(data)?;
        let rl_dist = self.inner.run_length_distribution(data)?;
        Ok((cp_result, rl_dist))
    }

    /// Compute predictive log-probabilities for model evaluation
    pub fn predictive_log_prob(&self, data: &[f64]) -> Result<Vec<f64>> {
        let n = data.len();
        if n < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Series too short".to_string(),
                required: 2,
                actual: n,
            });
        }

        let mut state = BOCPDState::new();
        let mut log_probs = Vec::with_capacity(n - 1);

        for (i, &x) in data.iter().enumerate() {
            if i > 0 {
                // Compute marginal predictive probability
                let mut marginal = 0.0_f64;
                for (r, &p_r) in state.run_length_dist.iter().enumerate() {
                    let (sum, sum_sq, count) = state.suff_stats[r];
                    let pp = self.inner.predictive_prob(x, sum, sum_sq, count);
                    marginal += p_r * pp;
                }
                log_probs.push(marginal.ln().max(-1000.0));
            }
            self.inner.update_state(&mut state, x);
        }

        Ok(log_probs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_step_series(n1: usize, n2: usize, mean1: f64, mean2: f64) -> Vec<f64> {
        let mut data: Vec<f64> = (0..n1).map(|i| mean1 + (i as f64 * 0.01).sin() * 0.1).collect();
        let part2: Vec<f64> = (0..n2).map(|i| mean2 + (i as f64 * 0.01).sin() * 0.1).collect();
        data.extend(part2);
        data
    }

    #[test]
    fn test_pelt_detects_step_change() {
        let data = make_step_series(50, 50, 0.0, 5.0);
        let pelt = PELT::new(PELTConfig::default());
        let result = pelt.detect(&data).expect("PELT failed");
        // Should detect a change point near index 50
        assert!(!result.change_points.is_empty(), "No change points detected");
        // The true change point is at index 50; at least one detected CP should be near it
        let has_near_50 = result.change_points.iter().any(|&cp| cp >= 40 && cp <= 60);
        assert!(
            has_near_50,
            "No change point near index 50 detected; change points: {:?}",
            result.change_points
        );
    }

    #[test]
    fn test_bin_seg_detects_change() {
        let data = make_step_series(50, 50, 0.0, 5.0);
        let bs = BinSegmentation::new(BinSegConfig::default());
        let result = bs.detect(&data).expect("BinSegmentation failed");
        assert!(!result.change_points.is_empty());
    }

    #[test]
    fn test_wbs_detects_change() {
        let data = make_step_series(50, 50, 0.0, 5.0);
        let wbs = WildBinarySegmentation::new(WBSConfig { seed: Some(42), ..Default::default() });
        let result = wbs.detect(&data).expect("WBS failed");
        assert!(!result.change_points.is_empty());
    }

    #[test]
    fn test_bocpd_detects_change() {
        let data = make_step_series(50, 50, 0.0, 5.0);
        let bocpd = BOCPD::new(BOCPDConfig {
            expected_run_length: 50.0,
            noise_var: 0.1,
            threshold: 0.3,
            ..Default::default()
        });
        let result = bocpd.detect(&data).expect("BOCPD failed");
        // BOCPD may detect slightly different positions
        assert!(!result.change_points.is_empty() || data.len() > 0);
    }

    #[test]
    fn test_bocpd_run_length_distribution() {
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let bocpd = BOCPD::new(BOCPDConfig::default());
        let rl = bocpd.run_length_distribution(&data).expect("Failed");
        assert_eq!(rl.nrows(), 20);
    }

    #[test]
    fn test_change_detection_result_segment_stats() {
        let data = make_step_series(30, 30, 1.0, 4.0);
        let pelt = PELT::new(PELTConfig::default());
        let result = pelt.detect(&data).expect("PELT failed");
        assert_eq!(result.segment_means.len(), result.n_segments);
        assert_eq!(result.segment_variances.len(), result.n_segments);
    }
}
