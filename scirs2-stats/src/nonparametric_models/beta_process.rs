//! Beta Process: a completely random measure as prior for hazard processes
//! and sparse feature models.
//!
//! The Beta Process BP(c, H) (Hjort 1990) is a distribution over random
//! measures where increments are Beta-distributed.  It is the de Finetti
//! prior for the Indian Buffet Process.
//!
//! Three parameterizations:
//! - **Lévy process**: via Lévy measure ν(dπ) = c * π^{-1}(1-π)^{c-1} H(dω)
//! - **Stick-breaking**: B ~ Σ_k π_k δ_{ω_k}, where π_k ~ Beta(c*H({ω_k}), c)
//! - **CRM (Completely Random Measure)**: finite truncation for computation

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::random::{rngs::StdRng, Beta as RandBeta, CoreRandom, Distribution, Rng, SeedableRng};

// ---------------------------------------------------------------------------
// Beta Process
// ---------------------------------------------------------------------------

/// Beta Process with concentration `c` and base measure `H`.
///
/// The base measure `H` is modeled as a finite discrete measure over `K` atoms
/// at locations `0, 1, …, K-1` with equal mass `H_total / K`.
#[derive(Debug, Clone)]
pub struct BetaProcess {
    /// Concentration parameter c > 0.
    pub c: f64,
    /// Total mass of the base measure.
    pub base_measure_mass: f64,
    /// Number of atoms in the truncated representation.
    pub n_atoms: usize,
    /// Atom probabilities π_k ~ Beta(c*H_k, c*(1-H_k)).
    pub atom_probs: Vec<f64>,
    /// Atom locations (indices 0..K).
    pub atom_locations: Vec<usize>,
    /// Whether the process has been sampled.
    pub is_sampled: bool,
}

impl BetaProcess {
    /// Construct a new Beta Process.
    ///
    /// # Parameters
    /// - `c`: concentration parameter (> 0)
    /// - `base_measure_mass`: total mass of H (> 0)
    /// - `n_atoms`: number of atoms for the truncated representation
    ///
    /// # Errors
    /// Returns an error on invalid parameters.
    pub fn new(c: f64, base_measure_mass: f64, n_atoms: usize) -> Result<Self> {
        if c <= 0.0 {
            return Err(StatsError::DomainError(format!(
                "Beta Process c must be > 0, got {c}"
            )));
        }
        if base_measure_mass <= 0.0 {
            return Err(StatsError::DomainError(format!(
                "base_measure_mass must be > 0, got {base_measure_mass}"
            )));
        }
        if n_atoms == 0 {
            return Err(StatsError::InvalidArgument(
                "n_atoms must be >= 1".into(),
            ));
        }
        Ok(Self {
            c,
            base_measure_mass,
            n_atoms,
            atom_probs: Vec::new(),
            atom_locations: (0..n_atoms).collect(),
            is_sampled: false,
        })
    }

    /// Sample the beta process by drawing atom probabilities.
    ///
    /// Each atom k has π_k ~ Beta(h_k, c - h_k) where
    /// h_k = base_measure_mass / n_atoms is the per-atom mass.
    ///
    /// For the standard BP(c, H) parameterization:
    /// π_k ~ Beta(c * H({ω_k}), c * (1 - H({ω_k})))
    pub fn sample<R: Rng>(&mut self, rng: &mut CoreRandom<R>) -> Result<()> {
        let h_per_atom = self.base_measure_mass / self.n_atoms as f64;
        let alpha_beta = h_per_atom * self.c;
        let beta_beta = self.c * (1.0 - h_per_atom).max(1e-10);

        self.atom_probs = Vec::with_capacity(self.n_atoms);
        for _ in 0..self.n_atoms {
            let b = RandBeta::new(alpha_beta.max(1e-10), beta_beta.max(1e-10)).map_err(|e| {
                StatsError::ComputationError(format!("Beta sampling error: {e}"))
            })?;
            self.atom_probs.push(b.sample(rng).clamp(0.0, 1.0));
        }
        self.is_sampled = true;
        Ok(())
    }

    /// Draw a binary Bernoulli realization from the beta process.
    ///
    /// For each atom k, sample z_k ~ Bernoulli(π_k).
    /// Returns a binary vector of length `n_atoms`.
    ///
    /// # Errors
    /// Returns an error when the process has not been sampled yet.
    pub fn draw_bernoulli<R: Rng>(&self, rng: &mut CoreRandom<R>) -> Result<Vec<bool>> {
        if !self.is_sampled {
            return Err(StatsError::InvalidInput(
                "Beta process must be sampled first (call .sample())".into(),
            ));
        }
        Ok(self.atom_probs.iter().map(|&pi| {
            let u = sample_uniform_01(rng);
            u < pi
        }).collect())
    }

    /// Expected number of active features per observation.
    ///
    /// E[#{active features}] = Σ_k π_k ≈ base_measure_mass.
    pub fn expected_active_features(&self) -> f64 {
        if self.is_sampled {
            self.atom_probs.iter().sum()
        } else {
            self.base_measure_mass
        }
    }

    /// Log-probability of a binary feature vector under the Beta Process.
    ///
    /// `log p(z | B) = Σ_k [z_k log π_k + (1-z_k) log(1-π_k)]`
    ///
    /// # Errors
    /// Returns an error when the process has not been sampled or when
    /// `z` has wrong length.
    pub fn log_prob_features(&self, z: &[bool]) -> Result<f64> {
        if !self.is_sampled {
            return Err(StatsError::InvalidInput(
                "Beta process must be sampled first".into(),
            ));
        }
        if z.len() != self.n_atoms {
            return Err(StatsError::DimensionMismatch(format!(
                "z has {} entries, expected {}",
                z.len(),
                self.n_atoms
            )));
        }
        let log_p: f64 = z
            .iter()
            .zip(self.atom_probs.iter())
            .map(|(&zk, &pi)| {
                let pi_clipped = pi.clamp(1e-300, 1.0 - 1e-300);
                if zk {
                    pi_clipped.ln()
                } else {
                    (1.0 - pi_clipped).ln()
                }
            })
            .sum();
        Ok(log_p)
    }

    /// Lévy-Khintchine representation: expected number of atoms with probability > threshold.
    ///
    /// For a BP(c, H) with total mass M = base_measure_mass:
    /// E[#{k: π_k > ε}] ≈ M * B(ε; 1, c) / B(1/K; 1, c)
    pub fn expected_atoms_above(&self, threshold: f64) -> f64 {
        if threshold <= 0.0 || threshold >= 1.0 {
            return self.base_measure_mass;
        }
        // Approximate using the incomplete beta function via the Beta CDF
        // For small h_k: π_k ~ Beta(c*h_k, c) → most mass concentrated near 0
        // Expected atoms with π_k > ε ≈ n_atoms * P(Beta(c*h_k, c) > ε)
        let h_per_atom = self.base_measure_mass / self.n_atoms as f64;
        let alpha_b = h_per_atom * self.c;
        let beta_b = self.c;
        // Approximate P(Beta(a,b) > ε) using normal approximation of Beta
        let mean_b = alpha_b / (alpha_b + beta_b);
        let var_b = alpha_b * beta_b / ((alpha_b + beta_b).powi(2) * (alpha_b + beta_b + 1.0));
        if var_b < 1e-15 {
            return if mean_b > threshold {
                self.n_atoms as f64
            } else {
                0.0
            };
        }
        let std_b = var_b.sqrt();
        let z = (threshold - mean_b) / std_b;
        let p_above = normal_cdf_complement(z);
        self.n_atoms as f64 * p_above
    }

    /// Posterior Beta Process after observing `n_obs` Bernoulli drawings.
    ///
    /// If m_k atoms out of n_obs observations activated feature k, the
    /// posterior for π_k is Beta(c*h_k + m_k, c*(1-h_k) + n_obs - m_k).
    ///
    /// # Parameters
    /// - `feature_counts`: m_k for each atom k
    /// - `n_obs`: total number of observations
    ///
    /// # Returns
    /// A new `BetaProcess` with posterior parameters.
    pub fn posterior<R: Rng>(
        &self,
        feature_counts: &[usize],
        n_obs: usize,
        rng: &mut CoreRandom<R>,
    ) -> Result<Self> {
        if feature_counts.len() != self.n_atoms {
            return Err(StatsError::DimensionMismatch(format!(
                "feature_counts has {} entries, expected {}",
                feature_counts.len(),
                self.n_atoms
            )));
        }
        let h_per_atom = self.base_measure_mass / self.n_atoms as f64;
        let mut post = Self::new(self.c, self.base_measure_mass, self.n_atoms)?;

        post.atom_probs = Vec::with_capacity(self.n_atoms);
        for k in 0..self.n_atoms {
            let m_k = feature_counts[k] as f64;
            let alpha_post = (self.c * h_per_atom + m_k).max(1e-10);
            let beta_post = (self.c * (1.0 - h_per_atom) + n_obs as f64 - m_k).max(1e-10);
            let b = RandBeta::new(alpha_post, beta_post).map_err(|e| {
                StatsError::ComputationError(format!("Beta sampling error: {e}"))
            })?;
            post.atom_probs.push(b.sample(rng).clamp(0.0, 1.0));
        }
        post.is_sampled = true;
        Ok(post)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn sample_uniform_01<R: Rng>(rng: &mut CoreRandom<R>) -> f64 {
    use scirs2_core::random::Uniform;
    Uniform::new(0.0_f64, 1.0)
        .map(|d| d.sample(rng))
        .unwrap_or(0.5)
}

/// Complementary CDF of standard normal: P(Z > z).
fn normal_cdf_complement(z: f64) -> f64 {
    // Abramowitz & Stegun approximation
    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let poly = t
        * (0.319_381_53
            + t * (-0.356_563_782
                + t * (1.781_477_937
                    + t * (-1.821_255_978 + t * 1.330_274_429))));
    let pdf = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let result = pdf * poly;
    if z >= 0.0 {
        result
    } else {
        1.0 - result
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta_process_construction() {
        assert!(BetaProcess::new(1.0, 2.0, 10).is_ok());
        assert!(BetaProcess::new(0.0, 2.0, 10).is_err());
        assert!(BetaProcess::new(1.0, 0.0, 10).is_err());
        assert!(BetaProcess::new(1.0, 2.0, 0).is_err());
    }

    #[test]
    fn test_beta_process_sample() {
        let mut bp = BetaProcess::new(1.0, 2.0, 10).expect("construction failed");
        let mut rng = CoreRandom::seed(42);
        bp.sample(&mut rng).expect("sampling failed");
        assert!(bp.is_sampled);
        assert_eq!(bp.atom_probs.len(), 10);
        assert!(bp.atom_probs.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_beta_process_draw_bernoulli() {
        let mut bp = BetaProcess::new(2.0, 1.0, 5).expect("construction failed");
        let mut rng = CoreRandom::seed(7);
        bp.sample(&mut rng).expect("sampling failed");
        let z = bp.draw_bernoulli(&mut rng).expect("draw failed");
        assert_eq!(z.len(), 5);
    }

    #[test]
    fn test_beta_process_unsampled_error() {
        let bp = BetaProcess::new(1.0, 2.0, 5).expect("construction failed");
        let mut rng = CoreRandom::seed(0);
        assert!(bp.draw_bernoulli(&mut rng).is_err());
        assert!(bp.log_prob_features(&[true, false, true, false, true]).is_err());
    }

    #[test]
    fn test_beta_process_log_prob() {
        let mut bp = BetaProcess::new(1.0, 2.0, 3).expect("construction failed");
        let mut rng = CoreRandom::seed(5);
        bp.sample(&mut rng).expect("sampling failed");
        let lp = bp.log_prob_features(&[true, false, true]).expect("log_prob failed");
        assert!(lp.is_finite());
        assert!(lp <= 0.0);
        // Wrong length
        assert!(bp.log_prob_features(&[true, false]).is_err());
    }

    #[test]
    fn test_beta_process_posterior() {
        let mut bp = BetaProcess::new(1.0, 3.0, 4).expect("construction failed");
        let mut rng = CoreRandom::seed(42);
        bp.sample(&mut rng).expect("sampling failed");
        let counts = vec![3usize, 1, 0, 2];
        let post = bp.posterior(&counts, 5, &mut rng).expect("posterior failed");
        assert_eq!(post.atom_probs.len(), 4);
        assert!(post.is_sampled);
        // Wrong counts length
        assert!(bp.posterior(&[1, 2], 5, &mut rng).is_err());
    }

    #[test]
    fn test_expected_active_features() {
        let mut bp = BetaProcess::new(1.0, 5.0, 50).expect("construction failed");
        // Before sampling: returns base_measure_mass
        assert!((bp.expected_active_features() - 5.0).abs() < 1e-10);
        let mut rng = CoreRandom::seed(42);
        bp.sample(&mut rng).expect("sampling failed");
        // After sampling: sum of atom probabilities (should be near 5.0)
        let expected = bp.expected_active_features();
        assert!(expected > 0.0, "expected active features = {expected}");
    }
}
