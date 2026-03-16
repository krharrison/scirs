//! Unified distribution interface for the SciRS2 ecosystem
//!
//! This module provides a consistent interface for all statistical distributions,
//! ensuring compatibility across the entire SciRS2 ecosystem including ToRSh, SkleaRS, etc.
//!
//! ## Design Philosophy
//!
//! 1. **Zero Breaking Changes**: All existing code continues to work
//! 2. **Full Compatibility**: Direct access to all rand_distr distributions
//! 3. **Enhanced Functionality**: Additional scientific computing features
//! 4. **Type Safety**: Unified trait system for distribution operations
//!
//! ## Usage Examples
//!
//! ```rust
//! use scirs2_core::random::distributions_unified::*;
//! use scirs2_core::random::thread_rng;
//!
//! // Create distributions with unified interface
//! let normal = UnifiedNormal::new(0.0, 1.0).expect("Operation failed");
//! let beta = UnifiedBeta::new(2.0, 5.0).expect("Operation failed");
//! let student_t = UnifiedStudentT::new(10.0).expect("Operation failed");
//!
//! let mut rng = thread_rng();
//! let sample = normal.sample_unified(&mut rng);
//! ```

use crate::random::core::Random;
use ::ndarray::{Array1, ArrayD, Dimension, IxDyn};
use rand::{Rng, RngExt};
use rand_distr::Distribution;
use std::fmt;

/// Error type for unified distribution operations
#[derive(Debug, Clone)]
pub enum UnifiedDistributionError {
    /// Invalid parameter value
    InvalidParameter(String),
    /// Construction failed
    ConstructionFailed(String),
    /// Generic error
    Other(String),
}

impl fmt::Display for UnifiedDistributionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            Self::ConstructionFailed(msg) => write!(f, "Construction failed: {}", msg),
            Self::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for UnifiedDistributionError {}

// Implement From for all rand_distr error types we use
impl From<rand_distr::NormalError> for UnifiedDistributionError {
    fn from(e: rand_distr::NormalError) -> Self {
        Self::ConstructionFailed(format!("Normal distribution error: {:?}", e))
    }
}

impl From<rand_distr::BetaError> for UnifiedDistributionError {
    fn from(e: rand_distr::BetaError) -> Self {
        Self::ConstructionFailed(format!("Beta distribution error: {:?}", e))
    }
}

impl From<rand_distr::CauchyError> for UnifiedDistributionError {
    fn from(e: rand_distr::CauchyError) -> Self {
        Self::ConstructionFailed(format!("Cauchy distribution error: {:?}", e))
    }
}

impl From<rand_distr::ChiSquaredError> for UnifiedDistributionError {
    fn from(e: rand_distr::ChiSquaredError) -> Self {
        Self::ConstructionFailed(format!("ChiSquared distribution error: {:?}", e))
    }
}

impl From<rand_distr::FisherFError> for UnifiedDistributionError {
    fn from(e: rand_distr::FisherFError) -> Self {
        Self::ConstructionFailed(format!("FisherF distribution error: {:?}", e))
    }
}

impl From<rand_distr::ExpError> for UnifiedDistributionError {
    fn from(e: rand_distr::ExpError) -> Self {
        Self::ConstructionFailed(format!("Exponential distribution error: {:?}", e))
    }
}

impl From<rand_distr::GammaError> for UnifiedDistributionError {
    fn from(e: rand_distr::GammaError) -> Self {
        Self::ConstructionFailed(format!("Gamma distribution error: {:?}", e))
    }
}

impl From<rand_distr::WeibullError> for UnifiedDistributionError {
    fn from(e: rand_distr::WeibullError) -> Self {
        Self::ConstructionFailed(format!("Weibull distribution error: {:?}", e))
    }
}

impl From<rand_distr::BinomialError> for UnifiedDistributionError {
    fn from(e: rand_distr::BinomialError) -> Self {
        Self::ConstructionFailed(format!("Binomial distribution error: {:?}", e))
    }
}

impl From<rand_distr::PoissonError> for UnifiedDistributionError {
    fn from(e: rand_distr::PoissonError) -> Self {
        Self::ConstructionFailed(format!("Poisson distribution error: {:?}", e))
    }
}

impl From<std::io::Error> for UnifiedDistributionError {
    fn from(e: std::io::Error) -> Self {
        Self::Other(e.to_string())
    }
}

/// Unified distribution trait for consistent interface across all distributions
pub trait UnifiedDistribution<T> {
    /// Sample a single value from the distribution
    fn sample_unified<R: Rng>(&self, rng: &mut Random<R>) -> T;

    /// Sample multiple values into a vector
    fn sample_vec<R: Rng>(&self, rng: &mut Random<R>, n: usize) -> Vec<T>;

    /// Sample into an ndarray
    fn sample_array<R: Rng>(&self, rng: &mut Random<R>, shape: IxDyn) -> ArrayD<T>
    where
        T: Clone;

    /// Get distribution parameters as a string (for debugging/logging)
    fn parameters_string(&self) -> String;

    /// Validate distribution parameters
    fn validate(&self) -> Result<(), UnifiedDistributionError>;
}

/// Macro to implement unified wrapper for rand_distr distributions
macro_rules! impl_unified_distribution {
    ($name:ident, $inner:ty, $output:ty, $params:expr) => {
        #[derive(Debug, Clone)]
        pub struct $name {
            inner: $inner,
        }

        impl $name {
            /// Get reference to inner distribution
            pub fn inner(&self) -> &$inner {
                &self.inner
            }

            /// Get mutable reference to inner distribution
            pub fn inner_mut(&mut self) -> &mut $inner {
                &mut self.inner
            }
        }

        impl UnifiedDistribution<$output> for $name {
            fn sample_unified<R: Rng>(&self, rng: &mut Random<R>) -> $output {
                rng.sample(&self.inner)
            }

            fn sample_vec<R: Rng>(&self, rng: &mut Random<R>, n: usize) -> Vec<$output> {
                (0..n).map(|_| self.sample_unified(rng)).collect()
            }

            fn sample_array<R: Rng>(&self, rng: &mut Random<R>, shape: IxDyn) -> ArrayD<$output>
            where
                $output: Clone,
            {
                let size = shape.size();
                let values = self.sample_vec(rng, size);
                ArrayD::from_shape_vec(shape, values).expect("Operation failed")
            }

            fn parameters_string(&self) -> String {
                $params(&self.inner)
            }

            fn validate(&self) -> Result<(), UnifiedDistributionError> {
                // Validation is done during construction
                Ok(())
            }
        }

        impl Distribution<$output> for $name {
            fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> $output {
                self.inner.sample(rng)
            }
        }
    };
}

// Continuous distributions

impl_unified_distribution!(
    UnifiedNormal,
    rand_distr::Normal<f64>,
    f64,
    |d: &rand_distr::Normal<f64>| format!("Normal(mean={}, std={})", d.mean(), d.std_dev())
);

impl UnifiedNormal {
    pub fn new(mean: f64, std_dev: f64) -> Result<Self, UnifiedDistributionError> {
        Ok(Self {
            inner: rand_distr::Normal::new(mean, std_dev)?,
        })
    }

    pub fn mean(&self) -> f64 {
        self.inner.mean()
    }

    pub fn std_dev(&self) -> f64 {
        self.inner.std_dev()
    }
}

impl_unified_distribution!(
    UnifiedBeta,
    rand_distr::Beta<f64>,
    f64,
    |_: &rand_distr::Beta<f64>| "Beta(alpha, beta)".to_string()
);

impl UnifiedBeta {
    pub fn new(alpha: f64, beta: f64) -> Result<Self, UnifiedDistributionError> {
        Ok(Self {
            inner: rand_distr::Beta::new(alpha, beta)?,
        })
    }
}

impl_unified_distribution!(
    UnifiedCauchy,
    rand_distr::Cauchy<f64>,
    f64,
    |_: &rand_distr::Cauchy<f64>| "Cauchy(median, scale)".to_string()
);

impl UnifiedCauchy {
    pub fn new(median: f64, scale: f64) -> Result<Self, UnifiedDistributionError> {
        Ok(Self {
            inner: rand_distr::Cauchy::new(median, scale)?,
        })
    }
}

impl_unified_distribution!(
    UnifiedChiSquared,
    rand_distr::ChiSquared<f64>,
    f64,
    |_: &rand_distr::ChiSquared<f64>| "ChiSquared(k)".to_string()
);

impl UnifiedChiSquared {
    pub fn new(k: f64) -> Result<Self, UnifiedDistributionError> {
        Ok(Self {
            inner: rand_distr::ChiSquared::new(k)?,
        })
    }
}

impl_unified_distribution!(
    UnifiedFisherF,
    rand_distr::FisherF<f64>,
    f64,
    |_: &rand_distr::FisherF<f64>| "FisherF(m, n)".to_string()
);

impl UnifiedFisherF {
    pub fn new(m: f64, n: f64) -> Result<Self, UnifiedDistributionError> {
        Ok(Self {
            inner: rand_distr::FisherF::new(m, n)?,
        })
    }
}

impl_unified_distribution!(
    UnifiedStudentT,
    rand_distr::StudentT<f64>,
    f64,
    |_: &rand_distr::StudentT<f64>| "StudentT(n)".to_string()
);

impl UnifiedStudentT {
    pub fn new(n: f64) -> Result<Self, UnifiedDistributionError> {
        Ok(Self {
            inner: rand_distr::StudentT::new(n)?,
        })
    }
}

impl_unified_distribution!(
    UnifiedLogNormal,
    rand_distr::LogNormal<f64>,
    f64,
    |_: &rand_distr::LogNormal<f64>| "LogNormal(mean, std)".to_string()
);

impl UnifiedLogNormal {
    pub fn new(mean: f64, std_dev: f64) -> Result<Self, UnifiedDistributionError> {
        Ok(Self {
            inner: rand_distr::LogNormal::new(mean, std_dev)?,
        })
    }
}

impl_unified_distribution!(
    UnifiedWeibull,
    rand_distr::Weibull<f64>,
    f64,
    |_: &rand_distr::Weibull<f64>| "Weibull(scale, shape)".to_string()
);

impl UnifiedWeibull {
    pub fn new(scale: f64, shape: f64) -> Result<Self, UnifiedDistributionError> {
        Ok(Self {
            inner: rand_distr::Weibull::new(scale, shape)?,
        })
    }
}

impl_unified_distribution!(
    UnifiedGamma,
    rand_distr::Gamma<f64>,
    f64,
    |_: &rand_distr::Gamma<f64>| "Gamma(shape, scale)".to_string()
);

impl UnifiedGamma {
    pub fn new(shape: f64, scale: f64) -> Result<Self, UnifiedDistributionError> {
        Ok(Self {
            inner: rand_distr::Gamma::new(shape, scale)?,
        })
    }
}

impl_unified_distribution!(
    UnifiedExp,
    rand_distr::Exp<f64>,
    f64,
    |_: &rand_distr::Exp<f64>| "Exp(lambda)".to_string()
);

impl UnifiedExp {
    pub fn new(lambda: f64) -> Result<Self, UnifiedDistributionError> {
        Ok(Self {
            inner: rand_distr::Exp::new(lambda)?,
        })
    }
}

// Discrete distributions

impl_unified_distribution!(
    UnifiedBinomial,
    rand_distr::Binomial,
    u64,
    |_: &rand_distr::Binomial| "Binomial(n, p)".to_string()
);

impl UnifiedBinomial {
    pub fn new(n: u64, p: f64) -> Result<Self, UnifiedDistributionError> {
        Ok(Self {
            inner: rand_distr::Binomial::new(n, p)?,
        })
    }
}

// Poisson<f64> in rand_distr 0.5 samples f64, not u64
impl_unified_distribution!(
    UnifiedPoisson,
    rand_distr::Poisson<f64>,
    f64,
    |_: &rand_distr::Poisson<f64>| "Poisson(lambda)".to_string()
);

impl UnifiedPoisson {
    pub fn new(lambda: f64) -> Result<Self, UnifiedDistributionError> {
        Ok(Self {
            inner: rand_distr::Poisson::new(lambda)?,
        })
    }
}

// Multivariate distributions

/// Unified Dirichlet distribution
///
/// Uses `scirs2_core::random::distributions::Dirichlet` which supports `Vec<f64>`
/// instead of `rand_distr::Dirichlet` which requires fixed-size arrays `[f64; N]`
#[derive(Debug, Clone)]
pub struct UnifiedDirichlet {
    inner: crate::random::distributions::Dirichlet,
}

impl UnifiedDirichlet {
    pub fn new(alpha: Vec<f64>) -> Result<Self, UnifiedDistributionError> {
        Ok(Self {
            inner: crate::random::distributions::Dirichlet::new(alpha).map_err(|e| {
                UnifiedDistributionError::ConstructionFailed(format!("Dirichlet error: {}", e))
            })?,
        })
    }

    /// Sample from Dirichlet distribution
    pub fn sample<R: Rng>(&self, rng: &mut Random<R>) -> Vec<f64> {
        self.inner.sample(rng)
    }

    /// Sample into an Array1
    pub fn sample_array<R: Rng>(&self, rng: &mut Random<R>) -> Array1<f64> {
        Array1::from_vec(self.sample(rng))
    }

    /// Sample multiple times
    pub fn sample_multiple<R: Rng>(&self, rng: &mut Random<R>, n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|_| self.sample(rng)).collect()
    }

    /// Get alpha parameters
    pub fn alphas(&self) -> &[f64] {
        self.inner.alphas()
    }
}

impl Distribution<Vec<f64>> for UnifiedDirichlet {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        // Sample from constituent Gamma distributions manually
        // This avoids the Random<R> vs &mut R type mismatch
        use rand_distr::Gamma;
        let gamma_samples: Vec<f64> = self
            .inner
            .alphas()
            .iter()
            .map(|&alpha| {
                let gamma = Gamma::new(alpha, 1.0).expect("Operation failed");
                rng.sample(gamma)
            })
            .collect();

        // Normalize to get Dirichlet sample
        let sum: f64 = gamma_samples.iter().sum();
        gamma_samples.into_iter().map(|x| x / sum).collect()
    }
}

impl UnifiedDistribution<Vec<f64>> for UnifiedDirichlet {
    fn sample_unified<R: Rng>(&self, rng: &mut Random<R>) -> Vec<f64> {
        self.sample(rng)
    }

    fn sample_vec<R: Rng>(&self, rng: &mut Random<R>, n: usize) -> Vec<Vec<f64>> {
        self.sample_multiple(rng, n)
    }

    fn sample_array<R: Rng>(&self, rng: &mut Random<R>, shape: IxDyn) -> ArrayD<Vec<f64>> {
        let size = shape.size();
        let values = self.sample_vec(rng, size);
        ArrayD::from_shape_vec(shape, values).expect("Operation failed")
    }

    fn parameters_string(&self) -> String {
        format!("Dirichlet(alpha=[{} values])", self.alphas().len())
    }

    fn validate(&self) -> Result<(), UnifiedDistributionError> {
        Ok(())
    }
}

/// Convenience functions for creating distributions with default parameters
pub mod defaults {
    use super::*;

    /// Create a standard normal distribution (mean=0, std=1)
    pub fn standard_normal() -> UnifiedNormal {
        UnifiedNormal::new(0.0, 1.0).expect("Operation failed")
    }

    /// Create a uniform distribution on [0, 1)
    pub fn uniform_01() -> rand_distr::Uniform<f64> {
        rand_distr::Uniform::new(0.0, 1.0).expect("Operation failed")
    }

    /// Create a standard exponential distribution (lambda=1)
    pub fn standard_exponential() -> UnifiedExp {
        UnifiedExp::new(1.0).expect("Operation failed")
    }

    /// Create a standard gamma distribution (shape=1, scale=1)
    pub fn standard_gamma() -> UnifiedGamma {
        UnifiedGamma::new(1.0, 1.0).expect("Operation failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::thread_rng;

    #[test]
    fn test_unified_normal() {
        let dist = UnifiedNormal::new(0.0, 1.0).expect("Operation failed");
        let mut rng = thread_rng();

        let sample = dist.sample_unified(&mut rng);
        assert!(sample.is_finite());

        let samples = dist.sample_vec(&mut rng, 100);
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn test_unified_beta() {
        let dist = UnifiedBeta::new(2.0, 5.0).expect("Operation failed");
        let mut rng = thread_rng();

        let sample = dist.sample_unified(&mut rng);
        assert!(sample >= 0.0 && sample <= 1.0);
    }

    #[test]
    fn test_unified_poisson() {
        let dist = UnifiedPoisson::new(5.0).expect("Operation failed");
        let mut rng = thread_rng();

        let sample = dist.sample_unified(&mut rng);
        assert!(sample >= 0.0);
    }

    #[test]
    fn test_unified_dirichlet() {
        let dist = UnifiedDirichlet::new(vec![1.0, 2.0, 3.0]).expect("Operation failed");
        let mut rng = thread_rng();

        let sample = dist.sample(&mut rng);
        assert_eq!(sample.len(), 3);

        let sum: f64 = sample.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_distribution_trait() {
        let dist = UnifiedNormal::new(0.0, 1.0).expect("Operation failed");
        let mut rng = rand::rng();

        // Test that Distribution trait works
        let sample: f64 = rng.sample(&dist);
        assert!(sample.is_finite());
    }
}
