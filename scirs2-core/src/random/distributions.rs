//! Advanced statistical distributions for SCIRS2 ecosystem
//!
//! This module provides comprehensive statistical distributions for scientific computing,
//! machine learning, and statistical analysis. All distributions are optimized for
//! performance and numerical stability.

use crate::random::core::Random;
use ::ndarray::{Array, Array1, Array2};
use rand::{Rng, RngExt};
use rand_distr::{Distribution, Gamma, Normal, Uniform};
use std::f64::consts::PI;

/// Beta distribution for modeling proportions and confidence intervals
#[derive(Debug, Clone)]
pub struct Beta {
    alpha: f64,
    beta: f64,
    gamma_alpha: Gamma<f64>,
    gamma_beta: Gamma<f64>,
}

impl Beta {
    /// Create a new Beta distribution
    pub fn new(alpha: f64, beta: f64) -> Result<Self, String> {
        if alpha <= 0.0 || beta <= 0.0 {
            return Err("Alpha and beta parameters must be positive".to_string());
        }

        let gamma_alpha = Gamma::new(alpha, 1.0).expect("Operation failed");
        let gamma_beta = Gamma::new(beta, 1.0).expect("Operation failed");

        Ok(Self {
            alpha,
            beta,
            gamma_alpha,
            gamma_beta,
        })
    }

    /// Sample from the Beta distribution using the gamma method
    pub fn sample<R: Rng>(&self, rng: &mut Random<R>) -> f64 {
        let x = rng.sample(self.gamma_alpha);
        let y = rng.sample(self.gamma_beta);
        x / (x + y)
    }

    /// Generate multiple samples
    pub fn sample_vec<R: Rng>(&self, rng: &mut Random<R>, count: usize) -> Vec<f64> {
        (0..count).map(|_| self.sample(rng)).collect()
    }

    /// Generate an array of samples
    pub fn sample_array<R: Rng>(&self, rng: &mut Random<R>, count: usize) -> Array1<f64> {
        let samples = self.sample_vec(rng, count);
        Array1::from_vec(samples)
    }

    /// Get distribution parameters
    pub fn parameters(&self) -> (f64, f64) {
        (self.alpha, self.beta)
    }

    /// Calculate mean
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Calculate variance
    pub fn variance(&self) -> f64 {
        let ab_sum = self.alpha + self.beta;
        (self.alpha * self.beta) / (ab_sum * ab_sum * (ab_sum + 1.0))
    }
}

/// Categorical distribution for discrete outcomes
#[derive(Debug, Clone)]
pub struct Categorical {
    weights: Vec<f64>,
    cumulative: Vec<f64>,
}

impl Categorical {
    /// Create a new Categorical distribution
    pub fn new(weights: Vec<f64>) -> Result<Self, String> {
        if weights.is_empty() {
            return Err("Weights vector cannot be empty".to_string());
        }

        if weights.iter().any(|&w| w < 0.0) {
            return Err("All weights must be non-negative".to_string());
        }

        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Err("Sum of weights must be positive".to_string());
        }

        // Create cumulative distribution
        let mut cumulative = Vec::with_capacity(weights.len());
        let mut sum = 0.0;
        for &weight in &weights {
            sum += weight / total;
            cumulative.push(sum);
        }

        Ok(Self {
            weights,
            cumulative,
        })
    }

    /// Sample from the Categorical distribution, returning the index
    pub fn sample<R: Rng>(&self, rng: &mut Random<R>) -> usize {
        let u = rng.sample(Uniform::new(0.0, 1.0).expect("Operation failed"));

        // Binary search for the cumulative probability
        match self
            .cumulative
            .binary_search_by(|&x| x.partial_cmp(&u).expect("Operation failed"))
        {
            Ok(idx) => idx,
            Err(idx) => idx.min(self.cumulative.len() - 1),
        }
    }

    /// Get the number of categories
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Generate multiple samples
    pub fn sample_vec<R: Rng>(&self, rng: &mut Random<R>, count: usize) -> Vec<usize> {
        (0..count).map(|_| self.sample(rng)).collect()
    }

    /// Get probability of category i
    pub fn probability(&self, i: usize) -> Option<f64> {
        if i < self.weights.len() {
            let total: f64 = self.weights.iter().sum();
            Some(self.weights[i] / total)
        } else {
            None
        }
    }
}

/// Weighted choice distribution for selecting from items with weights
#[derive(Debug, Clone)]
pub struct WeightedChoice<T> {
    items: Vec<T>,
    categorical: Categorical,
}

impl<T: Clone> WeightedChoice<T> {
    /// Create a new WeightedChoice distribution
    pub fn new(items: Vec<T>, weights: Vec<f64>) -> Result<Self, String> {
        if items.len() != weights.len() {
            return Err("Items and weights must have the same length".to_string());
        }

        let categorical = Categorical::new(weights)?;

        Ok(Self { items, categorical })
    }

    /// Sample from the WeightedChoice distribution
    pub fn sample<R: Rng>(&self, rng: &mut Random<R>) -> &T {
        let index = self.categorical.sample(rng);
        &self.items[index]
    }

    /// Sample and clone the result
    pub fn sample_cloned<R: Rng>(&self, rng: &mut Random<R>) -> T {
        self.sample(rng).clone()
    }

    /// Generate multiple samples
    pub fn sample_vec<R: Rng>(&self, rng: &mut Random<R>, count: usize) -> Vec<T> {
        (0..count).map(|_| self.sample_cloned(rng)).collect()
    }

    /// Get the number of items
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Get items and their probabilities
    pub fn items_and_probabilities(&self) -> Vec<(&T, f64)> {
        self.items
            .iter()
            .enumerate()
            .map(|(i, item)| (item, self.categorical.probability(i).unwrap_or(0.0)))
            .collect()
    }
}

/// Enhanced Exponential distribution
#[derive(Debug, Clone)]
pub struct ExponentialDist {
    lambda: f64,
    exponential: rand_distr::Exp<f64>,
}

impl ExponentialDist {
    /// Create a new Exponential distribution
    pub fn new(lambda: f64) -> Result<Self, String> {
        if lambda <= 0.0 {
            return Err("Lambda parameter must be positive".to_string());
        }

        let exponential = rand_distr::Exp::new(lambda).expect("Operation failed");

        Ok(Self {
            lambda,
            exponential,
        })
    }

    /// Sample from the Exponential distribution
    pub fn sample<R: Rng>(&self, rng: &mut Random<R>) -> f64 {
        rng.sample(self.exponential)
    }

    /// Generate multiple samples
    pub fn sample_vec<R: Rng>(&self, rng: &mut Random<R>, count: usize) -> Vec<f64> {
        (0..count).map(|_| self.sample(rng)).collect()
    }

    /// Generate an array of samples
    pub fn sample_array<R: Rng>(&self, rng: &mut Random<R>, count: usize) -> Array1<f64> {
        let samples = self.sample_vec(rng, count);
        Array1::from_vec(samples)
    }

    /// Get rate parameter
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Calculate mean
    pub fn mean(&self) -> f64 {
        1.0 / self.lambda
    }

    /// Calculate variance
    pub fn variance(&self) -> f64 {
        1.0 / (self.lambda * self.lambda)
    }
}

/// Enhanced Gamma distribution
#[derive(Debug, Clone)]
pub struct GammaDist {
    alpha: f64,
    beta: f64,
    gamma: Gamma<f64>,
}

impl GammaDist {
    /// Create a new Gamma distribution with shape (alpha) and scale (beta) parameters
    pub fn new(alpha: f64, beta: f64) -> Result<Self, String> {
        if alpha <= 0.0 || beta <= 0.0 {
            return Err("Alpha and beta parameters must be positive".to_string());
        }

        let gamma = Gamma::new(alpha, beta).expect("Operation failed");

        Ok(Self { alpha, beta, gamma })
    }

    /// Sample from the Gamma distribution
    pub fn sample<R: Rng>(&self, rng: &mut Random<R>) -> f64 {
        rng.sample(self.gamma)
    }

    /// Generate multiple samples
    pub fn sample_vec<R: Rng>(&self, rng: &mut Random<R>, count: usize) -> Vec<f64> {
        (0..count).map(|_| self.sample(rng)).collect()
    }

    /// Generate an array of samples
    pub fn sample_array<R: Rng>(&self, rng: &mut Random<R>, count: usize) -> Array1<f64> {
        let samples = self.sample_vec(rng, count);
        Array1::from_vec(samples)
    }

    /// Get distribution parameters
    pub fn parameters(&self) -> (f64, f64) {
        (self.alpha, self.beta)
    }

    /// Calculate mean
    pub fn mean(&self) -> f64 {
        self.alpha * self.beta
    }

    /// Calculate variance
    pub fn variance(&self) -> f64 {
        self.alpha * self.beta * self.beta
    }
}

/// Von Mises distribution for circular data
#[derive(Debug, Clone)]
pub struct VonMises {
    mu: f64,
    kappa: f64,
}

impl VonMises {
    /// Create a new Von Mises distribution
    pub fn mu(mu: f64, kappa: f64) -> Result<Self, String> {
        if kappa < 0.0 {
            return Err("Kappa parameter must be non-negative".to_string());
        }

        Ok(Self { mu, kappa })
    }

    /// Sample from the Von Mises distribution using rejection sampling
    pub fn sample<R: Rng>(&self, rng: &mut Random<R>) -> f64 {
        if self.kappa < 1e-6 {
            // Uniform distribution for very small kappa
            return rng.sample(Uniform::new(0.0, 2.0 * PI).expect("Operation failed"));
        }

        // Use Best and Fisher algorithm for rejection sampling
        let s = 0.5 / self.kappa;
        let r = s + (1.0 + s * s).sqrt();

        loop {
            let u1 = rng.sample(Uniform::new(0.0, 1.0).expect("Operation failed"));
            let z = (r * u1).cos();
            let d = z / (r + z);
            let u2 = rng.sample(Uniform::new(0.0, 1.0).expect("Operation failed"));

            if u2 < 1.0 - d * d || u2 <= (1.0 - d) * (-self.kappa * d).exp() {
                let u3 = rng.sample(Uniform::new(0.0, 1.0).expect("Operation failed"));
                let theta = if u3 > 0.5 {
                    self.mu + d.acos()
                } else {
                    self.mu - d.acos()
                };
                return ((theta % (2.0 * PI)) + 2.0 * PI) % (2.0 * PI);
            }
        }
    }

    /// Generate multiple samples
    pub fn sample_vec<R: Rng>(&self, rng: &mut Random<R>, count: usize) -> Vec<f64> {
        (0..count).map(|_| self.sample(rng)).collect()
    }

    /// Get distribution parameters
    pub fn parameters(&self) -> (f64, f64) {
        (self.mu, self.kappa)
    }
}

/// Multivariate Normal distribution
#[derive(Debug, Clone)]
pub struct MultivariateNormal {
    mean: Vec<f64>,
    cholesky: Array2<f64>,
    dimension: usize,
}

impl MultivariateNormal {
    /// Create a new Multivariate Normal distribution
    pub fn new(mean: Vec<f64>, covariance: Vec<Vec<f64>>) -> Result<Self, String> {
        let dimension = mean.len();

        if covariance.len() != dimension {
            return Err("Covariance matrix must be square and match mean dimension".to_string());
        }

        for row in &covariance {
            if row.len() != dimension {
                return Err("Covariance matrix must be square".to_string());
            }
        }

        // Convert to ndarray and compute Cholesky decomposition
        let mut cov_array = Array2::zeros((dimension, dimension));
        for (i, row) in covariance.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                cov_array[[i, j]] = val;
            }
        }

        // Simple Cholesky decomposition (for positive definite matrices)
        let cholesky = Self::cholesky_decomposition(cov_array)?;

        Ok(Self {
            mean,
            cholesky,
            dimension,
        })
    }

    /// Compute Cholesky decomposition
    fn cholesky_decomposition(mut a: Array2<f64>) -> Result<Array2<f64>, String> {
        let n = a.nrows();
        let mut l = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += l[[j, k]] * l[[j, k]];
                    }
                    let val = a[[j, j]] - sum;
                    if val <= 0.0 {
                        return Err("Matrix is not positive definite".to_string());
                    }
                    l[[j, j]] = val.sqrt();
                } else {
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += l[[i, k]] * l[[j, k]];
                    }
                    l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
                }
            }
        }

        Ok(l)
    }

    /// Sample from the Multivariate Normal distribution
    pub fn sample<R: Rng>(&self, rng: &mut Random<R>) -> Vec<f64> {
        // Generate standard normal samples
        let standard_normal = Normal::new(0.0, 1.0).expect("Operation failed");
        let z: Vec<f64> = (0..self.dimension)
            .map(|_| rng.sample(standard_normal))
            .collect();

        // Transform using Cholesky decomposition: X = μ + L * Z
        let mut result = self.mean.clone();
        for i in 0..self.dimension {
            for j in 0..=i {
                result[i] += self.cholesky[[i, j]] * z[j];
            }
        }

        result
    }

    /// Generate multiple samples
    pub fn sample_vec<R: Rng>(&self, rng: &mut Random<R>, count: usize) -> Vec<Vec<f64>> {
        (0..count).map(|_| self.sample(rng)).collect()
    }

    /// Generate samples as an array
    pub fn sample_array<R: Rng>(&self, rng: &mut Random<R>, count: usize) -> Array2<f64> {
        let samples = self.sample_vec(rng, count);
        let mut array = Array2::zeros((count, self.dimension));

        for (i, sample) in samples.iter().enumerate() {
            for (j, &val) in sample.iter().enumerate() {
                array[[i, j]] = val;
            }
        }

        array
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get mean vector
    pub fn mean(&self) -> &Vec<f64> {
        &self.mean
    }
}

/// Dirichlet distribution for probability simplexes
#[derive(Debug, Clone)]
pub struct Dirichlet {
    alphas: Vec<f64>,
    gamma_distributions: Vec<Gamma<f64>>,
}

impl Dirichlet {
    /// Create a new Dirichlet distribution
    pub fn new(alphas: Vec<f64>) -> Result<Self, String> {
        if alphas.is_empty() {
            return Err("Alpha parameters cannot be empty".to_string());
        }

        if alphas.iter().any(|&alpha| alpha <= 0.0) {
            return Err("All alpha parameters must be positive".to_string());
        }

        let gamma_distributions: Result<Vec<_>, _> =
            alphas.iter().map(|&alpha| Gamma::new(alpha, 1.0)).collect();

        let gamma_distributions = gamma_distributions.expect("Operation failed");

        Ok(Self {
            alphas,
            gamma_distributions,
        })
    }

    /// Sample from the Dirichlet distribution
    pub fn sample<R: Rng>(&self, rng: &mut Random<R>) -> Vec<f64> {
        // Sample from constituent Gamma distributions
        let gamma_samples: Vec<f64> = self
            .gamma_distributions
            .iter()
            .map(|gamma| rng.sample(*gamma))
            .collect();

        // Normalize to get Dirichlet sample
        let sum: f64 = gamma_samples.iter().sum();
        gamma_samples.into_iter().map(|x| x / sum).collect()
    }

    /// Generate multiple samples
    pub fn sample_vec<R: Rng>(&self, rng: &mut Random<R>, count: usize) -> Vec<Vec<f64>> {
        (0..count).map(|_| self.sample(rng)).collect()
    }

    /// Generate samples as an array
    pub fn sample_array<R: Rng>(&self, rng: &mut Random<R>, count: usize) -> Array2<f64> {
        let samples = self.sample_vec(rng, count);
        let mut array = Array2::zeros((count, self.alphas.len()));

        for (i, sample) in samples.iter().enumerate() {
            for (j, &val) in sample.iter().enumerate() {
                array[[i, j]] = val;
            }
        }

        array
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.alphas.len()
    }

    /// Get alpha parameters
    pub fn alphas(&self) -> &Vec<f64> {
        &self.alphas
    }

    /// Calculate mean
    pub fn mean(&self) -> Vec<f64> {
        let sum: f64 = self.alphas.iter().sum();
        self.alphas.iter().map(|&alpha| alpha / sum).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::core::seeded_rng;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_beta_distribution() {
        let beta = Beta::new(2.0, 3.0).expect("Operation failed");
        let mut rng = seeded_rng(42);

        let sample = beta.sample(&mut rng);
        assert!((0.0..1.0).contains(&sample));

        let samples = beta.sample_vec(&mut rng, 100);
        assert_eq!(samples.len(), 100);
        assert!(samples.iter().all(|&x| (0.0..1.0).contains(&x)));

        // Test statistics
        assert_abs_diff_eq!(beta.mean(), 0.4, epsilon = 1e-10);
        assert!(beta.variance() > 0.0);

        // Test error cases
        assert!(Beta::new(-1.0, 2.0).is_err());
        assert!(Beta::new(2.0, -1.0).is_err());
    }

    #[test]
    fn test_categorical_distribution() {
        let weights = vec![0.2, 0.3, 0.5];
        let categorical = Categorical::new(weights).expect("Operation failed");
        let mut rng = seeded_rng(123);

        let samples = categorical.sample_vec(&mut rng, 1000);
        assert_eq!(samples.len(), 1000);
        assert!(samples.iter().all(|&x| x < 3));

        // Check basic statistics
        let count_0 = samples.iter().filter(|&&x| x == 0).count();
        let count_1 = samples.iter().filter(|&&x| x == 1).count();
        let count_2 = samples.iter().filter(|&&x| x == 2).count();

        // Should have some samples in each category
        assert!(count_0 > 0);
        assert!(count_1 > 0);
        assert!(count_2 > 0);

        // Test probabilities
        assert_abs_diff_eq!(
            categorical.probability(0).expect("Operation failed"),
            0.2,
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            categorical.probability(1).expect("Operation failed"),
            0.3,
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            categorical.probability(2).expect("Operation failed"),
            0.5,
            epsilon = 1e-10
        );

        // Test error cases
        assert!(Categorical::new(vec![]).is_err());
        assert!(Categorical::new(vec![-1.0, 0.5]).is_err());
    }

    #[test]
    fn test_multivariate_normal() {
        let mean = vec![0.0, 0.0];
        let cov = vec![vec![1.0, 0.5], vec![0.5, 1.0]];

        let mvn = MultivariateNormal::new(mean, cov).expect("Operation failed");
        let mut rng = seeded_rng(456);
        let sample = mvn.sample(&mut rng);

        assert_eq!(sample.len(), 2);
        assert_eq!(mvn.dimension(), 2);

        let samples = mvn.sample_vec(&mut rng, 10);
        assert_eq!(samples.len(), 10);
        assert!(samples.iter().all(|s| s.len() == 2));
    }

    #[test]
    fn test_dirichlet_distribution() {
        let alphas = vec![1.0, 2.0, 3.0];
        let dirichlet = Dirichlet::new(alphas).expect("Operation failed");

        let mut rng = seeded_rng(789);
        let sample = dirichlet.sample(&mut rng);

        assert_eq!(sample.len(), 3);
        assert_abs_diff_eq!(sample.iter().sum::<f64>(), 1.0, epsilon = 1e-10);
        assert!(sample.iter().all(|&x| x >= 0.0));

        let mean = dirichlet.mean();
        assert_eq!(mean.len(), 3);
        assert_abs_diff_eq!(mean.iter().sum::<f64>(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_von_mises_distribution() {
        let von_mises = VonMises::mu(0.0, 1.0).expect("Operation failed");
        let mut rng = seeded_rng(101112);

        let samples = von_mises.sample_vec(&mut rng, 100);
        assert_eq!(samples.len(), 100);
        assert!(samples.iter().all(|&x| (0.0..2.0 * PI).contains(&x)));

        let (mu, kappa) = von_mises.parameters();
        assert_eq!(mu, 0.0);
        assert_eq!(kappa, 1.0);
    }

    #[test]
    fn test_weighted_choice() {
        let items = vec!["A", "B", "C"];
        let weights = vec![0.2, 0.3, 0.5];
        let weighted_choice = WeightedChoice::new(items, weights).expect("Operation failed");
        let mut rng = seeded_rng(131415);

        let samples = weighted_choice.sample_vec(&mut rng, 100);
        assert_eq!(samples.len(), 100);
        assert!(samples.iter().all(|&x| ["A", "B", "C"].contains(&x)));

        let items_probs = weighted_choice.items_and_probabilities();
        assert_eq!(items_probs.len(), 3);

        // Test error case
        let items_wrong = vec!["A", "B"];
        let weights_wrong = vec![0.2, 0.3, 0.5];
        assert!(WeightedChoice::new(items_wrong, weights_wrong).is_err());
    }
}
