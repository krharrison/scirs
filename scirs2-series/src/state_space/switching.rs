//! Markov Switching Models (Hamilton 1989).
//!
//! Regime-switching models where the observed data comes from one of K
//! latent regimes, with transitions governed by a first-order Markov chain.
//!
//! Observation model:  y_t | S_t = k  ~  N(μ_k, σ²_k)
//! Transition:  P(S_t = j | S_{t-1} = i) = p_{ij}
//!
//! Estimation via the Baum-Welch (EM) algorithm:
//!   E-step: forward-backward algorithm for smoothed regime probabilities
//!   M-step: update μ_k, σ²_k, p_{ij}
//!
//! Decoding via the Viterbi algorithm (most likely regime sequence).

use crate::error::{Result, TimeSeriesError};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// RegimeParameters
// ---------------------------------------------------------------------------

/// Parameters for a single regime (Gaussian emission).
#[derive(Debug, Clone)]
pub struct RegimeParameters {
    /// Regime mean μ_k
    pub mean: f64,
    /// Regime variance σ²_k
    pub variance: f64,
}

impl RegimeParameters {
    /// Create regime parameters with given mean and variance.
    pub fn new(mean: f64, variance: f64) -> Self {
        Self {
            mean,
            variance: variance.max(1e-10),
        }
    }

    /// Gaussian log-density: log N(y; μ, σ²)
    #[inline]
    pub fn log_density(&self, y: f64) -> f64 {
        let var = self.variance.max(1e-10);
        -0.5 * ((2.0 * PI).ln() + var.ln() + (y - self.mean).powi(2) / var)
    }

    /// Gaussian density: N(y; μ, σ²)
    #[inline]
    pub fn density(&self, y: f64) -> f64 {
        self.log_density(y).exp()
    }
}

// ---------------------------------------------------------------------------
// MarkovSwitchingModel
// ---------------------------------------------------------------------------

/// Markov Switching Model with K regimes and Gaussian emissions.
#[derive(Debug, Clone)]
pub struct MarkovSwitchingModel {
    /// Number of regimes K
    pub n_regimes: usize,
    /// Emission parameters for each regime \[K\]
    pub params: Vec<RegimeParameters>,
    /// Row-stochastic transition matrix P: \[K\]\[K\] where P\[i\]\[j\] = P(S_t=j|S_{t-1}=i)
    pub transition: Vec<Vec<f64>>,
    /// Stationary (or initial) regime probabilities π \[K\]
    pub initial_probs: Vec<f64>,
}

impl MarkovSwitchingModel {
    /// Create a new Markov switching model with K regimes.
    ///
    /// Initialises with evenly-spaced means, unit variances, and uniform
    /// transition and initial probabilities.
    pub fn new(n_regimes: usize) -> Self {
        assert!(n_regimes >= 2, "Need at least 2 regimes");
        let k = n_regimes;

        // Evenly spaced means; will be updated during fitting
        let params: Vec<RegimeParameters> = (0..k)
            .map(|i| RegimeParameters::new(i as f64, 1.0))
            .collect();

        // Uniform transition matrix
        let p_val = 1.0 / k as f64;
        let transition = vec![vec![p_val; k]; k];

        // Uniform initial distribution
        let initial_probs = vec![p_val; k];

        Self {
            n_regimes: k,
            params,
            transition,
            initial_probs,
        }
    }

    /// Create with specified parameters (for use in tests / manual construction).
    pub fn with_params(
        params: Vec<RegimeParameters>,
        transition: Vec<Vec<f64>>,
        initial_probs: Vec<f64>,
    ) -> Result<Self> {
        let k = params.len();
        if transition.len() != k || initial_probs.len() != k {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: k,
                actual: transition.len(),
            });
        }
        // Validate transition rows sum to 1
        for (i, row) in transition.iter().enumerate() {
            if row.len() != k {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: k,
                    actual: row.len(),
                });
            }
            let s: f64 = row.iter().sum();
            if (s - 1.0).abs() > 1e-6 {
                return Err(TimeSeriesError::InvalidInput(format!(
                    "Transition row {i} sums to {s} (expected 1.0)"
                )));
            }
        }
        Ok(Self {
            n_regimes: k,
            params,
            transition,
            initial_probs,
        })
    }

    // -----------------------------------------------------------------------
    // Hamilton filter
    // -----------------------------------------------------------------------

    /// Hamilton filter: compute filtered regime probabilities P(S_t=k | y_{1:t}).
    ///
    /// Returns `(filtered_probs [T][K], log_likelihood)`.
    pub fn filter(&self, data: &[f64]) -> (Vec<Vec<f64>>, f64) {
        let t_len = data.len();
        let k = self.n_regimes;

        if t_len == 0 {
            return (vec![], 0.0);
        }

        let mut alpha = self.initial_probs.clone(); // P(S_t=j | y_{1:t})
        let mut log_lik = 0.0_f64;
        let mut filtered = Vec::with_capacity(t_len);

        for t in 0..t_len {
            let y = data[t];

            // Predict: π_{t|t-1,k} = Σ_j α_{t-1,j} P_{jk}
            let predicted: Vec<f64> = if t == 0 {
                self.initial_probs.clone()
            } else {
                let prev: &Vec<f64> = &filtered[t - 1];
                (0..k)
                    .map(|j| {
                        let s: f64 = (0..k).map(|i| prev[i] * self.transition[i][j]).sum();
                        s
                    })
                    .collect()
            };

            // Update: weight by density
            let dens: Vec<f64> = (0..k)
                .map(|j| self.params[j].density(y).max(1e-300))
                .collect();

            // Numerator: η_{t,j} = f(y_t | S_t=j) * π_{t|t-1,j}
            let num: Vec<f64> = (0..k).map(|j| dens[j] * predicted[j]).collect();
            let scale: f64 = num.iter().sum::<f64>().max(1e-300);

            log_lik += scale.ln();

            alpha = num.iter().map(|&v| v / scale).collect();
            filtered.push(alpha.clone());
        }

        (filtered, log_lik)
    }

    // -----------------------------------------------------------------------
    // Forward-backward (Baum-Welch)
    // -----------------------------------------------------------------------

    /// Run the forward-backward algorithm for smoothed regime probabilities.
    ///
    /// Returns `(smoothed_probs [T][K], xi [T-1][K][K], log_likelihood)`.
    ///
    /// `xi[t][i][j]` = P(S_t=i, S_{t+1}=j | y_{1:T})
    fn forward_backward(&self, data: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>, f64) {
        let t_len = data.len();
        let k = self.n_regimes;

        if t_len == 0 {
            return (vec![], vec![], 0.0);
        }

        // Forward pass (scaled)
        let mut alpha_scaled: Vec<Vec<f64>> = Vec::with_capacity(t_len);
        let mut scales: Vec<f64> = Vec::with_capacity(t_len);

        // t=0
        {
            let y = data[0];
            let mut a: Vec<f64> = (0..k)
                .map(|j| self.initial_probs[j] * self.params[j].density(y).max(1e-300))
                .collect();
            let c = a.iter().sum::<f64>().max(1e-300);
            a.iter_mut().for_each(|v| *v /= c);
            scales.push(c);
            alpha_scaled.push(a);
        }

        for t in 1..t_len {
            let y = data[t];
            let prev = &alpha_scaled[t - 1];
            let mut a: Vec<f64> = (0..k)
                .map(|j| {
                    let pred: f64 = (0..k).map(|i| prev[i] * self.transition[i][j]).sum();
                    pred * self.params[j].density(y).max(1e-300)
                })
                .collect();
            let c = a.iter().sum::<f64>().max(1e-300);
            a.iter_mut().for_each(|v| *v /= c);
            scales.push(c);
            alpha_scaled.push(a);
        }

        // Log-likelihood from scales
        let log_lik: f64 = scales.iter().map(|c| c.ln()).sum();

        // Backward pass (scaled)
        let mut beta_scaled: Vec<Vec<f64>> = vec![vec![1.0f64; k]; t_len];

        for t in (0..t_len - 1).rev() {
            let y_next = data[t + 1];
            // Use split_at_mut to avoid borrow conflict between beta_scaled[t] and beta_scaled[t+1]
            let (left, right) = beta_scaled.split_at_mut(t + 1);
            let b_current = &mut left[t];
            let b_next = &right[0];
            for i in 0..k {
                let val: f64 = (0..k)
                    .map(|j| {
                        self.transition[i][j]
                            * self.params[j].density(y_next).max(1e-300)
                            * b_next[j]
                    })
                    .sum();
                b_current[i] = val;
            }
            // Scale to avoid underflow
            let c = b_current
                .iter()
                .cloned()
                .fold(0.0_f64, f64::max)
                .max(1e-300);
            b_current.iter_mut().for_each(|v| *v /= c);
        }

        // Smoothed probabilities: γ_t(j) ∝ α_t(j) β_t(j)
        let mut gamma: Vec<Vec<f64>> = Vec::with_capacity(t_len);
        for t in 0..t_len {
            let mut g: Vec<f64> = (0..k)
                .map(|j| alpha_scaled[t][j] * beta_scaled[t][j])
                .collect();
            let s: f64 = g.iter().sum::<f64>().max(1e-300);
            g.iter_mut().for_each(|v| *v /= s);
            gamma.push(g);
        }

        // Joint smoothed: ξ_t(i,j) = P(S_t=i, S_{t+1}=j | Y)
        let mut xi: Vec<Vec<Vec<f64>>> = Vec::with_capacity(t_len - 1);
        for t in 0..t_len - 1 {
            let y_next = data[t + 1];
            let mut xi_t = vec![vec![0.0f64; k]; k];
            let mut total = 0.0_f64;
            for i in 0..k {
                for j in 0..k {
                    let val = alpha_scaled[t][i]
                        * self.transition[i][j]
                        * self.params[j].density(y_next).max(1e-300)
                        * beta_scaled[t + 1][j];
                    xi_t[i][j] = val;
                    total += val;
                }
            }
            let total = total.max(1e-300);
            for row in &mut xi_t {
                for v in row.iter_mut() {
                    *v /= total;
                }
            }
            xi.push(xi_t);
        }

        (gamma, xi, log_lik)
    }

    // -----------------------------------------------------------------------
    // EM (Baum-Welch)
    // -----------------------------------------------------------------------

    /// Fit the model via the Baum-Welch EM algorithm.
    ///
    /// Before fitting, the model parameters are initialised using k-means-like
    /// partitioning of the data.
    ///
    /// Returns log-likelihood history per iteration.
    pub fn fit(&mut self, data: &[f64], n_iter: usize) -> Result<Vec<f64>> {
        let t_len = data.len();
        if t_len < self.n_regimes * 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Markov switching model needs more observations than 2*K".to_string(),
                required: self.n_regimes * 2,
                actual: t_len,
            });
        }

        // Initialise parameters from sorted data quantiles
        self.init_params_from_data(data);

        let k = self.n_regimes;
        let tol = 1e-6;
        let mut ll_history = Vec::with_capacity(n_iter);

        for iter in 0..n_iter {
            let (gamma, xi, ll) = self.forward_backward(data);
            ll_history.push(ll);

            if iter > 0 {
                let prev = ll_history[iter - 1];
                if (ll - prev).abs() < tol {
                    break;
                }
            }

            // M-step
            // Update emission parameters
            for j in 0..k {
                let n_j: f64 = gamma.iter().map(|g| g[j]).sum::<f64>().max(1e-10);
                let mu_j: f64 = gamma
                    .iter()
                    .zip(data.iter())
                    .map(|(g, &y)| g[j] * y)
                    .sum::<f64>()
                    / n_j;
                let var_j: f64 = (gamma
                    .iter()
                    .zip(data.iter())
                    .map(|(g, &y)| g[j] * (y - mu_j).powi(2))
                    .sum::<f64>()
                    / n_j)
                    .max(1e-6);
                self.params[j].mean = mu_j;
                self.params[j].variance = var_j;
            }

            // Update transition matrix
            if !xi.is_empty() {
                for i in 0..k {
                    let row_sum: f64 = xi
                        .iter()
                        .map(|xi_t| xi_t[i].iter().sum::<f64>())
                        .sum::<f64>()
                        .max(1e-10);
                    for j in 0..k {
                        let xi_ij: f64 = xi.iter().map(|xi_t| xi_t[i][j]).sum();
                        self.transition[i][j] = xi_ij / row_sum;
                    }
                }
            }

            // Update initial probabilities from first smoothed step
            if !gamma.is_empty() {
                let sum: f64 = gamma[0].iter().sum::<f64>().max(1e-10);
                for j in 0..k {
                    self.initial_probs[j] = gamma[0][j] / sum;
                }
            }
        }

        Ok(ll_history)
    }

    /// Initialise parameters from sorted quantiles of the data.
    fn init_params_from_data(&mut self, data: &[f64]) {
        let k = self.n_regimes;
        let mut sorted: Vec<f64> = data.iter().cloned().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();

        // Assign data quantile segments to each regime
        let global_var: f64 = {
            let mean = sorted.iter().sum::<f64>() / n as f64;
            sorted.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64
        };

        for j in 0..k {
            let lo = j * n / k;
            let hi = (j + 1) * n / k;
            let seg = &sorted[lo..hi];
            if seg.is_empty() {
                continue;
            }
            let mean = seg.iter().sum::<f64>() / seg.len() as f64;
            let var = (seg.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / seg.len() as f64)
                .max(global_var * 0.01)
                .max(1e-6);
            self.params[j].mean = mean;
            self.params[j].variance = var;
        }

        // Keep transition and initial uniform at this stage
    }

    // -----------------------------------------------------------------------
    // Viterbi decoding
    // -----------------------------------------------------------------------

    /// Viterbi algorithm: find the most likely regime sequence.
    ///
    /// Returns a vector of regime indices of length T.
    pub fn viterbi(&self, data: &[f64]) -> Vec<usize> {
        let t_len = data.len();
        let k = self.n_regimes;

        if t_len == 0 {
            return vec![];
        }

        // Log-domain Viterbi
        let log_trans: Vec<Vec<f64>> = self
            .transition
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&p| if p > 0.0 { p.ln() } else { f64::NEG_INFINITY })
                    .collect()
            })
            .collect();

        // viterbi[t][j] = max log-prob of sequence ending in state j at time t
        let mut vit: Vec<Vec<f64>> = vec![vec![f64::NEG_INFINITY; k]; t_len];
        let mut backtrack: Vec<Vec<usize>> = vec![vec![0usize; k]; t_len];

        // Initialisation
        for j in 0..k {
            let log_pi = if self.initial_probs[j] > 0.0 {
                self.initial_probs[j].ln()
            } else {
                f64::NEG_INFINITY
            };
            vit[0][j] = log_pi + self.params[j].log_density(data[0]);
        }

        // Recursion
        for t in 1..t_len {
            for j in 0..k {
                let log_em = self.params[j].log_density(data[t]);
                let (best_i, best_score) = (0..k)
                    .map(|i| (i, vit[t - 1][i] + log_trans[i][j]))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((0, f64::NEG_INFINITY));
                vit[t][j] = best_score + log_em;
                backtrack[t][j] = best_i;
            }
        }

        // Termination: find best final state
        let best_final = (0..k)
            .max_by(|&a, &b| {
                vit[t_len - 1][a]
                    .partial_cmp(&vit[t_len - 1][b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);

        // Backtrack
        let mut path = vec![0usize; t_len];
        path[t_len - 1] = best_final;
        for t in (0..t_len - 1).rev() {
            path[t] = backtrack[t + 1][path[t + 1]];
        }

        path
    }

    // -----------------------------------------------------------------------
    // Forecasting
    // -----------------------------------------------------------------------

    /// Forecast h steps ahead: compute expected value E[y_{T+h}].
    ///
    /// Uses regime probability propagation via the transition matrix.
    pub fn forecast(&self, h: usize, last_regime_probs: &[f64]) -> Vec<f64> {
        let k = self.n_regimes;
        if h == 0 || last_regime_probs.len() != k {
            return vec![];
        }

        let mut probs = last_regime_probs.to_vec();
        let mut forecasts = Vec::with_capacity(h);

        for _ in 0..h {
            // Propagate: probs_next[j] = Σ_i probs[i] * P[i][j]
            let next: Vec<f64> = (0..k)
                .map(|j| (0..k).map(|i| probs[i] * self.transition[i][j]).sum())
                .collect();

            // E[y | probs] = Σ_j probs[j] * μ_j
            let forecast: f64 = (0..k).map(|j| next[j] * self.params[j].mean).sum();
            forecasts.push(forecast);
            probs = next;
        }

        forecasts
    }

    /// Forecast with regime-conditional confidence intervals.
    ///
    /// Returns `(means, lower_95, upper_95)`, each of length h.
    pub fn forecast_with_intervals(
        &self,
        h: usize,
        last_regime_probs: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let k = self.n_regimes;
        let z95 = 1.959_963_985_f64;

        if h == 0 || last_regime_probs.len() != k {
            return (vec![], vec![], vec![]);
        }

        let mut probs = last_regime_probs.to_vec();
        let mut means = Vec::with_capacity(h);
        let mut lowers = Vec::with_capacity(h);
        let mut uppers = Vec::with_capacity(h);

        for _ in 0..h {
            let next: Vec<f64> = (0..k)
                .map(|j| (0..k).map(|i| probs[i] * self.transition[i][j]).sum())
                .collect();

            let mu: f64 = (0..k).map(|j| next[j] * self.params[j].mean).sum();

            // Mixture variance: Var[y] = Σ_j π_j (σ²_j + μ²_j) - (Σ_j π_j μ_j)²
            let second_moment: f64 = (0..k)
                .map(|j| next[j] * (self.params[j].variance + self.params[j].mean.powi(2)))
                .sum();
            let mix_var = (second_moment - mu * mu).max(0.0);
            let mix_std = mix_var.sqrt();

            means.push(mu);
            lowers.push(mu - z95 * mix_std);
            uppers.push(mu + z95 * mix_std);

            probs = next;
        }

        (means, lowers, uppers)
    }

    /// Return the stationary distribution π of the transition matrix.
    ///
    /// Computes by iterating P^n until convergence.
    pub fn stationary_distribution(&self) -> Vec<f64> {
        let k = self.n_regimes;
        let mut pi = vec![1.0 / k as f64; k];
        for _ in 0..1000 {
            let pi_next: Vec<f64> = (0..k)
                .map(|j| (0..k).map(|i| pi[i] * self.transition[i][j]).sum())
                .collect();
            let diff: f64 = (0..k).map(|j| (pi_next[j] - pi[j]).abs()).sum();
            pi = pi_next;
            if diff < 1e-12 {
                break;
            }
        }
        pi
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a two-regime time series with known regime sequence.
    fn two_regime_data() -> (Vec<f64>, Vec<usize>) {
        let regime_seq = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1,
            1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        ];
        // Means: regime 0 = -2.0, regime 1 = +2.0; small noise
        let data: Vec<f64> = regime_seq
            .iter()
            .enumerate()
            .map(|(i, &r)| {
                let noise = 0.2 * ((i as f64 * 1.7 + 0.3).sin());
                if r == 0 {
                    -2.0 + noise
                } else {
                    2.0 + noise
                }
            })
            .collect();
        (data, regime_seq.to_vec())
    }

    #[test]
    fn test_construction() {
        let m = MarkovSwitchingModel::new(3);
        assert_eq!(m.n_regimes, 3);
        assert_eq!(m.params.len(), 3);
        assert_eq!(m.transition.len(), 3);
        // Rows sum to 1
        for row in &m.transition {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_regime_log_density() {
        let rp = RegimeParameters::new(0.0, 1.0);
        // log N(0; 0, 1) = -0.5 * log(2π)
        let expected = -0.5 * (2.0 * PI).ln();
        assert!((rp.log_density(0.0) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_hamilton_filter_length() {
        let m = MarkovSwitchingModel::new(2);
        let data: Vec<f64> = (0..20).map(|i| (i as f64 * 0.3).sin()).collect();
        let (filt, ll) = m.filter(&data);
        assert_eq!(filt.len(), 20);
        assert!(ll.is_finite());
        // Each row is a probability distribution
        for row in &filt {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "row sum = {s}");
        }
    }

    #[test]
    fn test_fit_two_regimes() {
        let (data, _true_regimes) = two_regime_data();
        let mut m = MarkovSwitchingModel::new(2);
        let ll_hist = m.fit(&data, 50).expect("fit ok");

        assert!(!ll_hist.is_empty());
        // LL should be approximately non-decreasing
        for i in 1..ll_hist.len() {
            assert!(
                ll_hist[i] >= ll_hist[i - 1] - 0.5,
                "LL dropped at iter {i}: {} -> {}",
                ll_hist[i - 1],
                ll_hist[i]
            );
        }

        // After fitting, one regime should have mean ~ -2 and the other ~ +2
        let (mu0, mu1) = (m.params[0].mean, m.params[1].mean);
        let low_mean = mu0.min(mu1);
        let high_mean = mu0.max(mu1);
        assert!(
            low_mean < 0.0,
            "Low regime mean should be < 0, got {low_mean}"
        );
        assert!(
            high_mean > 0.0,
            "High regime mean should be > 0, got {high_mean}"
        );
    }

    #[test]
    fn test_viterbi_two_regimes() {
        let (data, _true_regimes) = two_regime_data();
        let mut m = MarkovSwitchingModel::new(2);
        m.fit(&data, 50).expect("fit ok");

        let path = m.viterbi(&data);
        assert_eq!(path.len(), data.len());
        // All indices must be in {0, 1}
        for &s in &path {
            assert!(s < 2, "Invalid regime index {s}");
        }
    }

    #[test]
    fn test_forecast_length() {
        let m = MarkovSwitchingModel::new(2);
        let probs = vec![0.8, 0.2];
        let fc = m.forecast(5, &probs);
        assert_eq!(fc.len(), 5);
    }

    #[test]
    fn test_forecast_with_intervals() {
        let m = MarkovSwitchingModel::new(2);
        let probs = vec![0.6, 0.4];
        let (means, lows, ups) = m.forecast_with_intervals(4, &probs);
        assert_eq!(means.len(), 4);
        for i in 0..4 {
            assert!(lows[i] <= means[i] + 1e-10);
            assert!(ups[i] >= means[i] - 1e-10);
        }
    }

    #[test]
    fn test_stationary_distribution() {
        let m = MarkovSwitchingModel::new(2);
        let pi = m.stationary_distribution();
        assert_eq!(pi.len(), 2);
        let s: f64 = pi.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_with_params_validation() {
        let params = vec![
            RegimeParameters::new(-1.0, 0.5),
            RegimeParameters::new(1.0, 0.5),
        ];
        let trans = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
        let init = vec![0.5, 0.5];
        let m = MarkovSwitchingModel::with_params(params, trans, init);
        assert!(m.is_ok());
    }

    #[test]
    fn test_viterbi_empty() {
        let m = MarkovSwitchingModel::new(2);
        let path = m.viterbi(&[]);
        assert!(path.is_empty());
    }
}
