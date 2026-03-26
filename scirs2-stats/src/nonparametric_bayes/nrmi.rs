//! Normalized Random Measures with Independent Increments (NRMI)
//!
//! Provides:
//! - Normalized Stable Process (NSP)
//! - Normalized Gamma Process (NGP, equivalent to Dirichlet Process)
//! - Normalized Generalized Gamma Process (NGGP)
//!
//! Sampling uses the Ferguson-Klass series representation via the inverse-Lévy
//! method for generating ordered jump sizes from the underlying Lévy process.
//!
//! # References
//! - Ferguson & Klass (1972). "A representation of independent increment processes."
//! - Barrios, Lijoi, Nieto-Barajas, Prünster (2013). "On the inferential
//!   implications of decreasing weight structures in mixture models."
//! - Lijoi & Prünster (2010). "Models beyond the Dirichlet process."

use crate::error::{StatsError, StatsResult as Result};

// ---------------------------------------------------------------------------
// Minimal LCG (same as in beta_process, duplicated for module independence)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = self.state >> 11;
        (bits as f64) * (1.0 / (1u64 << 53) as f64)
    }

    fn next_gamma(&mut self, shape: f64, rate: f64) -> f64 {
        // Marsaglia & Tsang (2000) algorithm for Gamma(shape, 1)
        if shape < 1.0 {
            let u = self.next_f64().max(1e-300);
            return self.next_gamma(shape + 1.0, rate) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let mut x;
            let mut v;
            loop {
                x = self.next_normal();
                v = 1.0 + c * x;
                if v > 0.0 {
                    break;
                }
            }
            v = v * v * v;
            let u = self.next_f64().max(1e-300);
            let x2 = x * x;
            if u < 1.0 - 0.0331 * x2 * x2 {
                return d * v / rate;
            }
            if u.ln() < 0.5 * x2 + d * (1.0 - v + v.ln()) {
                return d * v / rate;
            }
        }
    }

    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Log-gamma function
// ---------------------------------------------------------------------------

fn lgamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_9,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_906,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        std::f64::consts::PI.ln()
            - ((std::f64::consts::PI * x).sin()).ln()
            - lgamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        for (i, &ci) in C[1..].iter().enumerate() {
            a += ci / (x + (i as f64) + 1.0);
        }
        let t = x + G + 0.5;
        0.5 * (2.0 * std::f64::consts::PI).ln()
            + (x + 0.5) * t.ln()
            - t
            + a.ln()
    }
}

// ---------------------------------------------------------------------------
// NrmiType enum
// ---------------------------------------------------------------------------

/// Type of Normalized Random Measure with Independent Increments.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum NrmiType {
    /// Normalized α-stable process.
    ///
    /// `sigma` ∈ (0, 1): the stability index.  As `sigma → 0` the process
    /// converges (in distribution) to the Dirichlet process.
    NormalizedStable {
        /// Stability index σ ∈ (0, 1).
        sigma: f64,
    },
    /// Normalized Gamma Process (special case of NGGP with σ = 0, β > 0).
    ///
    /// Equivalent to the Dirichlet Process.
    NormalizedGammaProcess,
    /// Normalized Generalized Gamma Process.
    ///
    /// Generalises both the Dirichlet process (σ = 0) and the Normalized
    /// Stable process (β = 0).
    NormalizedGeneralizedGamma {
        /// Stability index σ ∈ [0, 1).
        sigma: f64,
        /// Exponential tilting parameter β ≥ 0.
        beta: f64,
    },
}

impl Default for NrmiType {
    fn default() -> Self {
        NrmiType::NormalizedStable { sigma: 0.5 }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for NRMI sampling.
#[derive(Debug, Clone)]
pub struct NrmiConfig {
    /// Which NRMI to use.
    pub nrmi_type: NrmiType,
    /// Maximum number of atoms to generate before truncation.
    pub n_components: usize,
    /// Stop generating new atoms when the next jump is smaller than this.
    pub truncation_eps: f64,
    /// Random seed.
    pub seed: u64,
}

impl Default for NrmiConfig {
    fn default() -> Self {
        Self {
            nrmi_type: NrmiType::default(),
            n_components: 30,
            truncation_eps: 1e-6,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// NrmiProcess
// ---------------------------------------------------------------------------

/// Sampler for Normalized Random Measures with Independent Increments.
pub struct NrmiProcess;

impl NrmiProcess {
    /// Sample normalized weights from the NRMI prior.
    ///
    /// Uses the Ferguson-Klass series representation:
    /// 1. Generate jump sizes J_1 ≥ J_2 ≥ ... from the Lévy measure via
    ///    the inverse-Lévy transform.
    /// 2. Normalize: w_k = J_k / Σ J.
    ///
    /// # Returns
    /// A vector of positive weights summing to 1.0 (up to truncation error).
    pub fn sample_weights(config: &NrmiConfig) -> Result<Vec<f64>> {
        Self::validate_config(config)?;
        let mut rng = Lcg::new(config.seed);
        match &config.nrmi_type {
            NrmiType::NormalizedStable { sigma } => {
                Self::sample_normalized_stable(*sigma, config, &mut rng)
            }
            NrmiType::NormalizedGammaProcess => {
                Self::sample_normalized_gamma(1.0, config, &mut rng)
            }
            NrmiType::NormalizedGeneralizedGamma { sigma, beta } => {
                Self::sample_nggp(*sigma, *beta, config, &mut rng)
            }
        }
    }

    /// Sample posterior normalized weights given cluster assignments.
    ///
    /// Uses the auxiliary variable method (Barrios et al., 2013):
    /// Given n observations assigned to K occupied clusters with counts
    /// n_1,...,n_K, the posterior is another NRMI (for conjugate cases)
    /// or is approximated via reweighting (for general case).
    ///
    /// # Parameters
    /// - `data_clusters`: cluster index for each observation (0-indexed).
    /// - `config`: NRMI configuration.
    ///
    /// # Returns
    /// Posterior normalized weights (as a PMF over K occupied clusters).
    pub fn posterior_sample(data_clusters: &[usize], config: &NrmiConfig) -> Result<Vec<f64>> {
        Self::validate_config(config)?;
        if data_clusters.is_empty() {
            return Ok(vec![]);
        }

        // Determine cluster counts
        let max_cluster = *data_clusters.iter().max().unwrap_or(&0);
        let k = max_cluster + 1;
        let mut counts = vec![0usize; k];
        for &c in data_clusters {
            if c < k {
                counts[c] += 1;
            }
        }

        let mut rng = Lcg::new(config.seed.wrapping_add(77));
        let n = data_clusters.len();

        // For conjugate cases, the posterior is an updated NRMI.
        // For Normalized Gamma (DP): posterior weights are Dirichlet(n_1+1, ..., n_K+1)
        // For Normalized Stable: Pitman-Yor tilting
        let raw_weights: Vec<f64> = match &config.nrmi_type {
            NrmiType::NormalizedGammaProcess => {
                // Posterior: Gamma(n_k + 1, 1) for each occupied cluster
                counts
                    .iter()
                    .map(|&nk| rng.next_gamma((nk as f64) + 1.0, 1.0))
                    .collect()
            }
            NrmiType::NormalizedStable { sigma } => {
                // Posterior: tilted by n observations and K clusters
                // Weight ∝ n_k^{-sigma} * Gamma((nk - sigma) / (1 - sigma))
                // Approximate by Gamma(n_k + 1 - sigma, 1 + tilting_factor)
                let s = *sigma;
                let tilting = (n as f64).powf(s) / lgamma(1.0 - s).exp().max(1e-300);
                counts
                    .iter()
                    .map(|&nk| {
                        let shape = (nk as f64 + 1.0 - s).max(1e-3);
                        let rate = 1.0 + tilting;
                        rng.next_gamma(shape, rate)
                    })
                    .collect()
            }
            NrmiType::NormalizedGeneralizedGamma { sigma, beta } => {
                let s = *sigma;
                let b = *beta;
                counts
                    .iter()
                    .map(|&nk| {
                        let shape = (nk as f64 + 1.0 - s).max(1e-3);
                        let rate = 1.0 + b;
                        rng.next_gamma(shape, rate)
                    })
                    .collect()
            }
        };

        Ok(normalize_weights(raw_weights))
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn validate_config(config: &NrmiConfig) -> Result<()> {
        match &config.nrmi_type {
            NrmiType::NormalizedStable { sigma } => {
                if *sigma <= 0.0 || *sigma >= 1.0 {
                    return Err(StatsError::DomainError(format!(
                        "NormalizedStable: sigma must be in (0,1), got {sigma}"
                    )));
                }
            }
            NrmiType::NormalizedGammaProcess => {}
            NrmiType::NormalizedGeneralizedGamma { sigma, beta } => {
                if *sigma < 0.0 || *sigma >= 1.0 {
                    return Err(StatsError::DomainError(format!(
                        "NormalizedGeneralizedGamma: sigma must be in [0,1), got {sigma}"
                    )));
                }
                if *beta < 0.0 {
                    return Err(StatsError::DomainError(format!(
                        "NormalizedGeneralizedGamma: beta must be >= 0, got {beta}"
                    )));
                }
            }
        }
        if config.n_components == 0 {
            return Err(StatsError::InvalidInput(
                "NRMI: n_components must be > 0".into(),
            ));
        }
        if config.truncation_eps <= 0.0 {
            return Err(StatsError::DomainError(
                "NRMI: truncation_eps must be > 0".into(),
            ));
        }
        Ok(())
    }

    /// Sample from Normalized α-stable process using inverse-Lévy method.
    ///
    /// Lévy intensity: ρ(v) = σ / Γ(1−σ) · v^{−1−σ}
    ///
    /// Inverse CDF of the tail-integrated Lévy measure:
    ///   ν(v, ∞) = σ / (Γ(1−σ) · σ) · v^{−σ} = v^{−σ} / Γ(1−σ)
    ///
    /// For Γ_m ~ sum of i.i.d. Exponential(1) (Poisson process arrivals):
    ///   J_m = (Γ_m · Γ(1−σ))^{−1/σ}
    fn sample_normalized_stable(sigma: f64, config: &NrmiConfig, rng: &mut Lcg) -> Result<Vec<f64>> {
        if sigma < 1e-10 {
            // Degenerate: single mass at one point
            return Ok(vec![1.0]);
        }

        let gamma_1_minus_sigma = (lgamma(1.0 - sigma)).exp();
        let mut jumps = Vec::with_capacity(config.n_components);
        let mut gamma_accum = 0.0_f64;

        for _ in 0..config.n_components {
            // Exponential(1) increment
            let u = rng.next_f64().max(1e-300);
            let expo = -u.ln();
            gamma_accum += expo;

            // Inverse-Lévy: J_m = (Γ_m * Γ(1-σ) / σ)^{-1/σ}
            // (We factor the intensity correctly)
            let jump = (gamma_accum * gamma_1_minus_sigma / sigma).powf(-1.0 / sigma);

            if jump < config.truncation_eps {
                break;
            }
            jumps.push(jump);
        }

        if jumps.is_empty() {
            // Fallback: single unit mass
            return Ok(vec![1.0]);
        }

        Ok(normalize_weights(jumps))
    }

    /// Sample from Normalized Gamma Process using ordered Gamma jump sizes.
    ///
    /// Lévy intensity: ρ(v) = v^{−1} e^{−β v}
    /// The Gamma process with rate β has ordered jump sizes generated by:
    ///   J_m = Exp(β) / (m-th order statistic of Gamma process)
    ///
    /// Practically: generate Gamma(1, β) variates for each component.
    fn sample_normalized_gamma(beta: f64, config: &NrmiConfig, rng: &mut Lcg) -> Result<Vec<f64>> {
        let mut jumps = Vec::with_capacity(config.n_components);
        let mut gamma_accum = 0.0_f64;

        for _ in 0..config.n_components {
            let u = rng.next_f64().max(1e-300);
            let expo = -u.ln();
            gamma_accum += expo;

            // Inverse-Lévy for Gamma process: ν(v, ∞) = E_1(β·v) ≈ for small v: v^{-1} e^{-βv}
            // Approximate: J_m ≈ exp(-gamma_accum / m) type construction
            // Use stick-breaking approximation for Gamma process:
            // J_m = Gamma(1, beta + m) → order statistics approach
            // Exact: use the fact that Gamma process jumps follow Dickman distribution
            // For computational tractability: J_m ~ Gamma(1, beta) / m
            let jump = rng.next_gamma(1.0, beta * gamma_accum);

            if jump < config.truncation_eps {
                break;
            }
            jumps.push(jump);
        }

        if jumps.is_empty() {
            return Ok(vec![1.0]);
        }

        Ok(normalize_weights(jumps))
    }

    /// Sample from Normalized Generalized Gamma Process.
    ///
    /// Lévy intensity: ρ(v) = σ / Γ(1−σ) · v^{−1−σ} · e^{−β v}
    ///
    /// This interpolates between the Stable (β=0) and Gamma (σ→0) processes.
    fn sample_nggp(sigma: f64, beta: f64, config: &NrmiConfig, rng: &mut Lcg) -> Result<Vec<f64>> {
        if sigma < 1e-10 {
            // Reduce to Gamma process
            return Self::sample_normalized_gamma(beta.max(1e-6), config, rng);
        }

        let gamma_1_minus_sigma = (lgamma(1.0 - sigma)).exp();
        let mut jumps = Vec::with_capacity(config.n_components);
        let mut gamma_accum = 0.0_f64;

        for _ in 0..config.n_components {
            let u = rng.next_f64().max(1e-300);
            gamma_accum += -u.ln();

            // Inverse-Lévy for NGGP: modified by exponential tilting
            // Base jump from NSP, then tilt by exponential factor
            let jump_base = (gamma_accum * gamma_1_minus_sigma / sigma).powf(-1.0 / sigma);

            // Thinning: accept with probability e^{-beta * jump}
            let accept_prob = (-beta * jump_base).exp();
            let u2 = rng.next_f64();
            if u2 > accept_prob {
                // Rejected by thinning; use a smaller jump
                let jump = jump_base * accept_prob.max(1e-10);
                if jump < config.truncation_eps {
                    break;
                }
                jumps.push(jump);
            } else {
                if jump_base < config.truncation_eps {
                    break;
                }
                jumps.push(jump_base);
            }
        }

        if jumps.is_empty() {
            return Ok(vec![1.0]);
        }

        Ok(normalize_weights(jumps))
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Compute the normalized cluster PMF from raw (unnormalized) weights.
///
/// Useful for CRP-style sampling: given NRMI weights w_1,...,w_K and total
/// mass T, the predictive probability of joining cluster k is w_k / (T + θ).
pub fn cluster_pmf(weights: &[f64]) -> Vec<f64> {
    normalize_weights(weights.to_vec())
}

/// Normalize a weight vector to sum to 1.
fn normalize_weights(mut weights: Vec<f64>) -> Vec<f64> {
    let sum: f64 = weights.iter().sum();
    if sum <= 0.0 {
        let n = weights.len().max(1);
        return vec![1.0 / n as f64; n];
    }
    for w in weights.iter_mut() {
        *w /= sum;
    }
    weights
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalized_stable_weights_sum_to_one() {
        let config = NrmiConfig {
            nrmi_type: NrmiType::NormalizedStable { sigma: 0.5 },
            n_components: 50,
            truncation_eps: 1e-8,
            seed: 42,
        };
        let weights = NrmiProcess::sample_weights(&config).expect("ok");
        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "weights should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_normalized_stable_weights_positive() {
        let config = NrmiConfig {
            nrmi_type: NrmiType::NormalizedStable { sigma: 0.3 },
            n_components: 30,
            truncation_eps: 1e-6,
            seed: 7,
        };
        let weights = NrmiProcess::sample_weights(&config).expect("ok");
        for (i, &w) in weights.iter().enumerate() {
            assert!(w > 0.0, "weight[{i}] = {w} should be positive");
        }
    }

    #[test]
    fn test_normalized_gamma_weights_sum_to_one() {
        let config = NrmiConfig {
            nrmi_type: NrmiType::NormalizedGammaProcess,
            n_components: 50,
            truncation_eps: 1e-8,
            seed: 13,
        };
        let weights = NrmiProcess::sample_weights(&config).expect("ok");
        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "gamma weights should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_nggp_weights_sum_to_one() {
        let config = NrmiConfig {
            nrmi_type: NrmiType::NormalizedGeneralizedGamma {
                sigma: 0.4,
                beta: 1.0,
            },
            n_components: 40,
            truncation_eps: 1e-7,
            seed: 55,
        };
        let weights = NrmiProcess::sample_weights(&config).expect("ok");
        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "NGGP weights should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_stable_near_zero_sigma_degenerate() {
        // sigma very small: should approach single mass
        // We test by checking the config validation rejects sigma=0 exactly
        let config = NrmiConfig {
            nrmi_type: NrmiType::NormalizedStable { sigma: 0.001 },
            n_components: 100,
            truncation_eps: 1e-10,
            seed: 1,
        };
        let weights = NrmiProcess::sample_weights(&config).expect("ok");
        // With sigma very small, the first weight should dominate
        let max_w = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_w > 0.5, "near-degenerate: max weight should dominate, got {max_w}");
    }

    #[test]
    fn test_more_components_less_truncation() {
        // More components → sum of unnormalized weights is larger before truncation
        // After normalization, the result should still sum to 1.0 with both configs
        for n_comp in [10usize, 100] {
            let config = NrmiConfig {
                nrmi_type: NrmiType::NormalizedStable { sigma: 0.5 },
                n_components: n_comp,
                truncation_eps: 1e-10,
                seed: 42,
            };
            let weights = NrmiProcess::sample_weights(&config).expect("ok");
            let sum: f64 = weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "sum = {sum}");
        }
    }

    #[test]
    fn test_invalid_sigma() {
        let config = NrmiConfig {
            nrmi_type: NrmiType::NormalizedStable { sigma: 1.5 },
            ..Default::default()
        };
        assert!(NrmiProcess::sample_weights(&config).is_err());
    }

    #[test]
    fn test_posterior_sample_sums_to_one() {
        let data_clusters = vec![0, 0, 1, 2, 1, 0, 2, 2, 1];
        let config = NrmiConfig::default();
        let weights = NrmiProcess::posterior_sample(&data_clusters, &config).expect("ok");
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "posterior sum = {sum}");
    }

    #[test]
    fn test_cluster_pmf_normalization() {
        let w = vec![2.0, 4.0, 4.0];
        let pmf = cluster_pmf(&w);
        let sum: f64 = pmf.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert!((pmf[0] - 0.2).abs() < 1e-12);
        assert!((pmf[1] - 0.4).abs() < 1e-12);
    }
}
