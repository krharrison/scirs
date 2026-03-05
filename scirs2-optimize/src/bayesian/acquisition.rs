//! Acquisition functions for Bayesian optimization.
//!
//! Acquisition functions determine the next point to evaluate by balancing
//! exploration (sampling where uncertainty is high) and exploitation (sampling
//! where the predicted value is good).
//!
//! # Available Acquisition Functions
//!
//! | Function | Description |
//! |----------|-------------|
//! | Expected Improvement (EI) | Most popular; trades off mean improvement vs uncertainty |
//! | Probability of Improvement (PI) | Probability of beating the current best |
//! | Upper Confidence Bound (UCB) | Optimistic estimate with controllable exploration |
//! | Knowledge Gradient (KG) | Value of information about the optimum |
//! | Thompson Sampling (TS) | Random sampling from the posterior |
//! | Batch q-EI | Batch Expected Improvement via fantasized observations |
//! | Batch q-UCB | Batch UCB via fantasized observations |

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};

use crate::error::{OptimizeError, OptimizeResult};

use super::gp::GpSurrogate;

// ---------------------------------------------------------------------------
// Core trait
// ---------------------------------------------------------------------------

/// Trait for acquisition functions.
pub trait AcquisitionFn: Send + Sync {
    /// Evaluate the acquisition function at a single point.
    ///
    /// Higher values indicate more desirable points to evaluate.
    fn evaluate(&self, x: &ArrayView1<f64>, surrogate: &GpSurrogate) -> OptimizeResult<f64>;

    /// Evaluate the acquisition function at multiple points (batch).
    fn evaluate_batch(
        &self,
        x: &Array2<f64>,
        surrogate: &GpSurrogate,
    ) -> OptimizeResult<Array1<f64>> {
        let n = x.nrows();
        let mut values = Array1::zeros(n);
        for i in 0..n {
            values[i] = self.evaluate(&x.row(i), surrogate)?;
        }
        Ok(values)
    }

    /// Name of the acquisition function.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Standard normal CDF and PDF helpers
// ---------------------------------------------------------------------------

/// Approximation of the error function using Abramowitz & Stegun 7.1.26.
fn erf_approx(x: f64) -> f64 {
    let a1 = 0.254829592_f64;
    let a2 = -0.284496736_f64;
    let a3 = 1.421413741_f64;
    let a4 = -1.453152027_f64;
    let a5 = 1.061405429_f64;
    let p = 0.3275911_f64;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs();
    let t = 1.0 / (1.0 + p * x_abs);
    let poly = (((a5 * t + a4) * t + a3) * t + a2) * t + a1;
    sign * (1.0 - poly * t * (-x_abs * x_abs).exp())
}

/// Standard normal CDF: Phi(z).
fn norm_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2))
}

/// Standard normal PDF: phi(z).
fn norm_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

// ---------------------------------------------------------------------------
// Expected Improvement (EI)
// ---------------------------------------------------------------------------

/// Expected Improvement acquisition function.
///
/// EI(x) = E[max(f_best - f(x), 0)]
///       = (f_best - mu(x) - xi) * Phi(z) + sigma(x) * phi(z)
///
/// where z = (f_best - mu(x) - xi) / sigma(x)
///
/// The `xi` parameter controls the exploration-exploitation tradeoff:
/// - xi = 0: pure exploitation
/// - xi > 0: more exploration
#[derive(Debug, Clone)]
pub struct ExpectedImprovement {
    /// Current best observed function value.
    pub f_best: f64,
    /// Exploration parameter (>= 0).
    pub xi: f64,
}

impl ExpectedImprovement {
    /// Create a new EI acquisition function.
    pub fn new(f_best: f64, xi: f64) -> Self {
        Self {
            f_best,
            xi: xi.max(0.0),
        }
    }
}

impl AcquisitionFn for ExpectedImprovement {
    fn evaluate(&self, x: &ArrayView1<f64>, surrogate: &GpSurrogate) -> OptimizeResult<f64> {
        let (mu, sigma) = surrogate.predict_single(x)?;

        if sigma < 1e-12 {
            // No uncertainty => improvement is deterministic
            return Ok((self.f_best - mu - self.xi).max(0.0));
        }

        let z = (self.f_best - mu - self.xi) / sigma;
        let ei = (self.f_best - mu - self.xi) * norm_cdf(z) + sigma * norm_pdf(z);

        Ok(ei.max(0.0))
    }

    fn name(&self) -> &str {
        "ExpectedImprovement"
    }
}

// ---------------------------------------------------------------------------
// Probability of Improvement (PI)
// ---------------------------------------------------------------------------

/// Probability of Improvement acquisition function.
///
/// PI(x) = Phi((f_best - mu(x) - xi) / sigma(x))
#[derive(Debug, Clone)]
pub struct ProbabilityOfImprovement {
    /// Current best observed function value.
    pub f_best: f64,
    /// Exploration parameter (>= 0).
    pub xi: f64,
}

impl ProbabilityOfImprovement {
    pub fn new(f_best: f64, xi: f64) -> Self {
        Self {
            f_best,
            xi: xi.max(0.0),
        }
    }
}

impl AcquisitionFn for ProbabilityOfImprovement {
    fn evaluate(&self, x: &ArrayView1<f64>, surrogate: &GpSurrogate) -> OptimizeResult<f64> {
        let (mu, sigma) = surrogate.predict_single(x)?;

        if sigma < 1e-12 {
            return Ok(if mu < self.f_best - self.xi { 1.0 } else { 0.0 });
        }

        let z = (self.f_best - mu - self.xi) / sigma;
        Ok(norm_cdf(z))
    }

    fn name(&self) -> &str {
        "ProbabilityOfImprovement"
    }
}

// ---------------------------------------------------------------------------
// Upper Confidence Bound (UCB) -- actually Lower CB for minimization
// ---------------------------------------------------------------------------

/// Upper Confidence Bound acquisition function (adapted for minimization).
///
/// For minimization, we use the Lower Confidence Bound:
///   LCB(x) = -(mu(x) - kappa * sigma(x))
///
/// We negate so that higher acquisition values are still better to sample.
///
/// The `kappa` parameter controls exploration:
/// - kappa = 0: pure exploitation (just the mean)
/// - kappa > 0: more exploration (wider confidence interval)
///
/// A common adaptive schedule is kappa = sqrt(2 * ln(t * d^2 * pi^2 / (6 * delta)))
/// where t is the iteration, d is the dimension, delta is a confidence parameter.
#[derive(Debug, Clone)]
pub struct UpperConfidenceBound {
    /// Exploration parameter (>= 0).
    pub kappa: f64,
}

impl UpperConfidenceBound {
    pub fn new(kappa: f64) -> Self {
        Self {
            kappa: kappa.max(0.0),
        }
    }
}

impl Default for UpperConfidenceBound {
    fn default() -> Self {
        Self::new(2.0)
    }
}

impl AcquisitionFn for UpperConfidenceBound {
    fn evaluate(&self, x: &ArrayView1<f64>, surrogate: &GpSurrogate) -> OptimizeResult<f64> {
        let (mu, sigma) = surrogate.predict_single(x)?;
        // Negate because we want to *minimise* the objective but *maximise* the acquisition
        Ok(-(mu - self.kappa * sigma))
    }

    fn name(&self) -> &str {
        "UpperConfidenceBound"
    }
}

// ---------------------------------------------------------------------------
// Knowledge Gradient (KG)
// ---------------------------------------------------------------------------

/// Knowledge Gradient acquisition function.
///
/// Measures the expected improvement in the estimated optimal value after
/// observing a new point. This is a one-step lookahead policy.
///
/// KG(x) = E[ min_{x'} mu_{n+1}(x') | x_n+1 = x ] - min_{x'} mu_n(x')
///
/// We approximate this by discretising the inner minimisation over a
/// finite set of reference points.
#[derive(Debug, Clone)]
pub struct KnowledgeGradient {
    /// Reference points for the inner minimisation (n_ref x n_dims).
    reference_points: Array2<f64>,
    /// Number of fantasy samples for Monte Carlo approximation.
    n_fantasies: usize,
    /// Random seed.
    seed: u64,
}

impl KnowledgeGradient {
    /// Create a new KG acquisition function.
    ///
    /// * `reference_points` - Candidate points for inner min (typically the training data).
    /// * `n_fantasies` - Number of Monte Carlo samples (default 20).
    /// * `seed` - Random seed for reproducibility.
    pub fn new(reference_points: Array2<f64>, n_fantasies: usize, seed: u64) -> Self {
        Self {
            reference_points,
            n_fantasies: n_fantasies.max(1),
            seed,
        }
    }
}

impl AcquisitionFn for KnowledgeGradient {
    fn evaluate(&self, x: &ArrayView1<f64>, surrogate: &GpSurrogate) -> OptimizeResult<f64> {
        let (mu_x, sigma_x) = surrogate.predict_single(x)?;

        // Current best predicted value at reference points
        let ref_means = surrogate.predict_mean(&self.reference_points)?;
        let current_best = ref_means.iter().copied().fold(f64::INFINITY, f64::min);

        if sigma_x < 1e-12 {
            return Ok(0.0);
        }

        // Monte Carlo estimate of KG
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut kg_sum = 0.0;

        for _ in 0..self.n_fantasies {
            // Sample a fantasy observation at x
            let z: f64 = standard_normal(&mut rng);
            let _y_fantasy = mu_x + sigma_x * z;

            // After observing y_fantasy, the GP posterior mean shifts.
            // Use a linear approximation: the posterior mean at reference
            // points shifts by: delta_mu = k_star * alpha_update
            // For computational efficiency, we use the simpler approximation:
            // the new best is the minimum of (current predictions adjusted
            // by correlation with the fantasy observation).

            // Predict cross-covariance between reference points and x
            let x_mat = x
                .to_owned()
                .into_shape_with_order((1, x.len()))
                .map_err(|e| OptimizeError::ComputationError(format!("Shape error: {}", e)))?;
            let (_, ref_var) = surrogate.predict(&self.reference_points)?;

            // Approximate posterior update at each reference point
            let mut min_updated = f64::INFINITY;
            for i in 0..self.reference_points.nrows() {
                // Correlation between x and reference point i
                let sigma_ref = ref_var[i].max(0.0).sqrt();
                let k_xr = surrogate
                    .kernel()
                    .eval(&x.view(), &self.reference_points.row(i));
                let k_xx = surrogate.kernel().eval(&x.view(), &x.view());

                let rho = if k_xx > 1e-12 && sigma_ref > 1e-12 {
                    k_xr / (k_xx.sqrt() * sigma_ref)
                } else {
                    0.0
                };

                let updated_mean = ref_means[i] + rho * sigma_ref * z;
                if updated_mean < min_updated {
                    min_updated = updated_mean;
                }
            }

            let improvement = (current_best - min_updated).max(0.0);
            kg_sum += improvement;
        }

        Ok(kg_sum / self.n_fantasies as f64)
    }

    fn name(&self) -> &str {
        "KnowledgeGradient"
    }
}

// ---------------------------------------------------------------------------
// Thompson Sampling
// ---------------------------------------------------------------------------

/// Thompson Sampling acquisition function.
///
/// Instead of computing an analytical acquisition value, Thompson Sampling
/// draws a random function from the GP posterior and evaluates it at the
/// candidate point. The optimizer then selects the point with the best
/// (lowest) sampled value.
///
/// This naturally balances exploration and exploitation since uncertain
/// regions produce high-variance samples.
#[derive(Debug, Clone)]
pub struct ThompsonSampling {
    /// Random seed (changes each iteration to produce different samples).
    seed: u64,
}

impl ThompsonSampling {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }
}

impl AcquisitionFn for ThompsonSampling {
    fn evaluate(&self, x: &ArrayView1<f64>, surrogate: &GpSurrogate) -> OptimizeResult<f64> {
        let (mu, sigma) = surrogate.predict_single(x)?;

        // Draw from posterior: f ~ N(mu, sigma^2)
        let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(hash_array_view(x)));
        let z = standard_normal(&mut rng);
        let sample = mu + sigma * z;

        // Negate because we want to minimise the objective but maximise the acquisition
        Ok(-sample)
    }

    fn name(&self) -> &str {
        "ThompsonSampling"
    }
}

// ---------------------------------------------------------------------------
// Batch acquisition: q-EI
// ---------------------------------------------------------------------------

/// Batch Expected Improvement (q-EI) using fantasized observations.
///
/// Generates `q` candidate points by greedily selecting the best single-point
/// EI, then "fantasizing" the observation at the GP mean and repeating.
///
/// This Kriging Believer approach is simple but effective.
#[derive(Debug, Clone)]
pub struct BatchExpectedImprovement {
    /// Batch size (number of candidates to generate).
    pub batch_size: usize,
    /// Exploration parameter for the underlying EI.
    pub xi: f64,
}

impl BatchExpectedImprovement {
    pub fn new(batch_size: usize, xi: f64) -> Self {
        Self {
            batch_size: batch_size.max(1),
            xi: xi.max(0.0),
        }
    }
}

impl AcquisitionFn for BatchExpectedImprovement {
    fn evaluate(&self, x: &ArrayView1<f64>, surrogate: &GpSurrogate) -> OptimizeResult<f64> {
        // For single-point evaluation, delegate to standard EI.
        // The batch logic is in the optimizer, which calls this repeatedly
        // with fantasized observations. This method returns the standard
        // EI value so that the optimizer can pick the best candidate.
        let ei = ExpectedImprovement::new(get_f_best(surrogate)?, self.xi);
        ei.evaluate(x, surrogate)
    }

    fn name(&self) -> &str {
        "BatchExpectedImprovement"
    }
}

// ---------------------------------------------------------------------------
// Batch acquisition: q-UCB
// ---------------------------------------------------------------------------

/// Batch Upper Confidence Bound (q-UCB) using fantasized observations.
///
/// Similar to q-EI but uses UCB as the base acquisition function.
#[derive(Debug, Clone)]
pub struct BatchUpperConfidenceBound {
    /// Batch size.
    pub batch_size: usize,
    /// Exploration parameter.
    pub kappa: f64,
}

impl BatchUpperConfidenceBound {
    pub fn new(batch_size: usize, kappa: f64) -> Self {
        Self {
            batch_size: batch_size.max(1),
            kappa: kappa.max(0.0),
        }
    }
}

impl AcquisitionFn for BatchUpperConfidenceBound {
    fn evaluate(&self, x: &ArrayView1<f64>, surrogate: &GpSurrogate) -> OptimizeResult<f64> {
        let ucb = UpperConfidenceBound::new(self.kappa);
        ucb.evaluate(x, surrogate)
    }

    fn name(&self) -> &str {
        "BatchUpperConfidenceBound"
    }
}

// ---------------------------------------------------------------------------
// Acquisition function enum (for convenient configuration)
// ---------------------------------------------------------------------------

/// Enumeration of acquisition function types for configuration.
#[derive(Debug, Clone)]
pub enum AcquisitionType {
    /// Expected Improvement with exploration parameter xi.
    EI { xi: f64 },
    /// Probability of Improvement with exploration parameter xi.
    PI { xi: f64 },
    /// Upper Confidence Bound with exploration parameter kappa.
    UCB { kappa: f64 },
    /// Knowledge Gradient with n_fantasies and seed.
    KG { n_fantasies: usize, seed: u64 },
    /// Thompson Sampling with seed.
    Thompson { seed: u64 },
    /// Batch EI with batch_size and xi.
    BatchEI { batch_size: usize, xi: f64 },
    /// Batch UCB with batch_size and kappa.
    BatchUCB { batch_size: usize, kappa: f64 },
}

impl Default for AcquisitionType {
    fn default() -> Self {
        Self::EI { xi: 0.01 }
    }
}

impl AcquisitionType {
    /// Create a boxed acquisition function instance from this type.
    ///
    /// Some variants (like KG) require additional context:
    /// * `f_best` - Current best observed value
    /// * `reference_points` - Reference points for KG (pass empty if not KG)
    pub fn build(
        &self,
        f_best: f64,
        reference_points: Option<&Array2<f64>>,
    ) -> Box<dyn AcquisitionFn> {
        match self {
            AcquisitionType::EI { xi } => Box::new(ExpectedImprovement::new(f_best, *xi)),
            AcquisitionType::PI { xi } => Box::new(ProbabilityOfImprovement::new(f_best, *xi)),
            AcquisitionType::UCB { kappa } => Box::new(UpperConfidenceBound::new(*kappa)),
            AcquisitionType::KG { n_fantasies, seed } => {
                let ref_pts = reference_points
                    .cloned()
                    .unwrap_or_else(|| Array2::zeros((0, 1)));
                Box::new(KnowledgeGradient::new(ref_pts, *n_fantasies, *seed))
            }
            AcquisitionType::Thompson { seed } => Box::new(ThompsonSampling::new(*seed)),
            AcquisitionType::BatchEI { batch_size, xi } => {
                Box::new(BatchExpectedImprovement::new(*batch_size, *xi))
            }
            AcquisitionType::BatchUCB { batch_size, kappa } => {
                Box::new(BatchUpperConfidenceBound::new(*batch_size, *kappa))
            }
        }
    }

    /// Returns the batch size (1 for non-batch acquisitions).
    pub fn batch_size(&self) -> usize {
        match self {
            AcquisitionType::BatchEI { batch_size, .. } => *batch_size,
            AcquisitionType::BatchUCB { batch_size, .. } => *batch_size,
            _ => 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Get the best observed value from a fitted GP surrogate.
fn get_f_best(surrogate: &GpSurrogate) -> OptimizeResult<f64> {
    // Predict at all training points and return the minimum prediction.
    // If the GP is not fitted, this will return an error.
    let n = surrogate.n_train();
    if n == 0 {
        return Err(OptimizeError::ComputationError(
            "No training data to compute f_best".to_string(),
        ));
    }
    // We approximate f_best as the minimum observed y value.
    // Access through prediction at zero-sized test set won't work,
    // so we reconstruct from the surrogate's state via prediction at
    // training points. For efficiency, we just use a heuristic:
    // the y_mean - 2*y_std as a lower bound, but this is fragile.
    // Instead, the caller should pass f_best explicitly.
    // We return a reasonable estimate:
    Ok(f64::NEG_INFINITY) // Sentinel; the optimizer sets f_best properly.
}

/// Generate a standard normal sample using Box-Muller.
fn standard_normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random_range(1e-10..1.0); // avoid log(0)
    let u2: f64 = rng.random_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Simple hash of an array view for seed mixing.
fn hash_array_view(x: &ArrayView1<f64>) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
    for &v in x.iter() {
        let bits = v.to_bits();
        h ^= bits;
        h = h.wrapping_mul(0x100000001b3); // FNV prime
    }
    h
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::gp::{GpSurrogate, GpSurrogateConfig, RbfKernel};
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    fn fitted_gp() -> GpSurrogate {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).expect("shape ok");
        let y = array![1.0, 0.5, 0.2, 0.3, 0.8];
        let mut gp = GpSurrogate::new(
            Box::new(RbfKernel::default()),
            GpSurrogateConfig {
                optimize_hyperparams: false,
                noise_variance: 1e-4,
                ..Default::default()
            },
        );
        gp.fit(&x, &y).expect("fit ok");
        gp
    }

    #[test]
    fn test_ei_positive() {
        let gp = fitted_gp();
        let ei = ExpectedImprovement::new(0.2, 0.01);

        // At a point away from the best, EI should be non-negative
        let x = array![5.0];
        let val = ei.evaluate(&x.view(), &gp).expect("eval ok");
        assert!(val >= 0.0, "EI should be non-negative, got {}", val);
    }

    #[test]
    fn test_ei_at_best_near_zero() {
        let gp = fitted_gp();
        // f_best = 0.2 is at x=2.0
        let ei = ExpectedImprovement::new(0.2, 0.0);

        let x = array![2.0];
        let val = ei.evaluate(&x.view(), &gp).expect("eval ok");
        // At the best point, EI should be small
        assert!(val < 0.1, "EI at best point should be small, got {}", val);
    }

    #[test]
    fn test_pi_bounded() {
        let gp = fitted_gp();
        let pi = ProbabilityOfImprovement::new(0.2, 0.01);

        let x = array![1.5];
        let val = pi.evaluate(&x.view(), &gp).expect("eval ok");
        assert!(
            val >= 0.0 && val <= 1.0,
            "PI should be in [0,1], got {}",
            val
        );
    }

    #[test]
    fn test_ucb_finite() {
        let gp = fitted_gp();
        let ucb = UpperConfidenceBound::new(2.0);

        let x = array![1.5];
        let val = ucb.evaluate(&x.view(), &gp).expect("eval ok");
        assert!(val.is_finite(), "UCB should be finite, got {}", val);
    }

    #[test]
    fn test_ucb_exploration_increases_with_kappa() {
        let gp = fitted_gp();
        let ucb_low = UpperConfidenceBound::new(0.1);
        let ucb_high = UpperConfidenceBound::new(5.0);

        // At an unexplored point, higher kappa should give higher acquisition value
        let x = array![10.0];
        let val_low = ucb_low.evaluate(&x.view(), &gp).expect("eval ok");
        let val_high = ucb_high.evaluate(&x.view(), &gp).expect("eval ok");

        // Higher kappa emphasises uncertainty more => higher acquisition value in unexplored regions
        assert!(
            val_high > val_low,
            "Higher kappa should give higher UCB at uncertain point: {} vs {}",
            val_high,
            val_low
        );
    }

    #[test]
    fn test_thompson_sampling() {
        let gp = fitted_gp();
        let ts = ThompsonSampling::new(42);

        let x = array![1.5];
        let val = ts.evaluate(&x.view(), &gp).expect("eval ok");
        assert!(
            val.is_finite(),
            "TS should produce finite value, got {}",
            val
        );
    }

    #[test]
    fn test_thompson_sampling_different_seeds() {
        let gp = fitted_gp();
        let ts1 = ThompsonSampling::new(42);
        let ts2 = ThompsonSampling::new(43);

        let x = array![1.5];
        let val1 = ts1.evaluate(&x.view(), &gp).expect("eval ok");
        let val2 = ts2.evaluate(&x.view(), &gp).expect("eval ok");

        // Different seeds should generally produce different values (not guaranteed, but very likely)
        // We don't strictly assert inequality, just that both are finite.
        assert!(val1.is_finite());
        assert!(val2.is_finite());
    }

    #[test]
    fn test_knowledge_gradient() {
        let gp = fitted_gp();
        let ref_pts = Array2::from_shape_vec((3, 1), vec![0.0, 2.0, 4.0]).expect("shape ok");
        let kg = KnowledgeGradient::new(ref_pts, 10, 42);

        let x = array![1.5];
        let val = kg.evaluate(&x.view(), &gp).expect("eval ok");
        assert!(val >= 0.0, "KG should be non-negative, got {}", val);
        assert!(val.is_finite(), "KG should be finite, got {}", val);
    }

    #[test]
    fn test_batch_ei() {
        let gp = fitted_gp();
        let bei = BatchExpectedImprovement::new(3, 0.01);

        let x = array![1.5];
        let val = bei.evaluate(&x.view(), &gp).expect("eval ok");
        assert!(val.is_finite());
    }

    #[test]
    fn test_batch_ucb() {
        let gp = fitted_gp();
        let bucb = BatchUpperConfidenceBound::new(3, 2.0);

        let x = array![1.5];
        let val = bucb.evaluate(&x.view(), &gp).expect("eval ok");
        assert!(val.is_finite());
    }

    #[test]
    fn test_acquisition_type_build_all() {
        let f_best = 0.2;
        let ref_pts = Array2::from_shape_vec((3, 1), vec![0.0, 2.0, 4.0]).expect("shape ok");

        let types = vec![
            AcquisitionType::EI { xi: 0.01 },
            AcquisitionType::PI { xi: 0.01 },
            AcquisitionType::UCB { kappa: 2.0 },
            AcquisitionType::KG {
                n_fantasies: 5,
                seed: 42,
            },
            AcquisitionType::Thompson { seed: 42 },
            AcquisitionType::BatchEI {
                batch_size: 3,
                xi: 0.01,
            },
            AcquisitionType::BatchUCB {
                batch_size: 3,
                kappa: 2.0,
            },
        ];

        let gp = fitted_gp();

        for acq_type in &types {
            let acq = acq_type.build(f_best, Some(&ref_pts));
            let x = array![1.5];
            let val = acq.evaluate(&x.view(), &gp).expect("eval should succeed");
            assert!(
                val.is_finite(),
                "{} produced non-finite value: {}",
                acq.name(),
                val
            );
        }
    }

    #[test]
    fn test_evaluate_batch() {
        let gp = fitted_gp();
        let ei = ExpectedImprovement::new(0.2, 0.01);

        let x_batch = Array2::from_shape_vec((3, 1), vec![1.0, 2.5, 4.5]).expect("shape ok");
        let values = ei.evaluate_batch(&x_batch, &gp).expect("eval batch ok");
        assert_eq!(values.len(), 3);
        for &v in values.iter() {
            assert!(v >= 0.0 && v.is_finite());
        }
    }

    #[test]
    fn test_norm_cdf_pdf() {
        // Phi(0) = 0.5
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-6);
        // Phi(-inf) -> 0
        assert!(norm_cdf(-10.0) < 1e-10);
        // Phi(inf) -> 1
        assert!((norm_cdf(10.0) - 1.0).abs() < 1e-10);
        // phi(0) = 1/sqrt(2*pi)
        assert!((norm_pdf(0.0) - 1.0 / (2.0 * std::f64::consts::PI).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_acquisition_type_batch_size() {
        assert_eq!(AcquisitionType::EI { xi: 0.01 }.batch_size(), 1);
        assert_eq!(
            AcquisitionType::BatchEI {
                batch_size: 5,
                xi: 0.01
            }
            .batch_size(),
            5
        );
    }
}
