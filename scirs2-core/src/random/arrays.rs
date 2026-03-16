//! Optimized array operations for random generation
//!
//! This module provides optimized array generation functions that address performance
//! concerns identified in OxiRS production deployment. The key optimization is using
//! a single RNG instance for bulk operations instead of creating RNG-per-element.

use crate::random::core::Random;
use ::ndarray::{Array, Dimension, Ix1, Ix2, IxDyn};
use rand::{Rng, RngExt};
use rand_distr::{Distribution, Exp, Gamma, Normal, Uniform};

/// Trait for optimized random array generation
///
/// This trait provides bulk array generation methods that use a single RNG instance
/// for all elements, dramatically improving performance compared to RNG-per-element approaches.
pub trait OptimizedArrayRandom<T, D: Dimension> {
    /// Generate a random array using a single RNG instance (bulk operation)
    fn random_bulk<R, Dist>(shape: D, distribution: Dist, rng: &mut Random<R>) -> Self
    where
        R: Rng,
        Dist: Distribution<T> + Copy;

    /// Generate a random array using a closure (single RNG instance)
    fn random_using_bulk<R, F>(shape: D, rng: &mut Random<R>, f: F) -> Self
    where
        R: Rng,
        F: FnMut(&mut Random<R>) -> T;
}

impl<T, D> OptimizedArrayRandom<T, D> for Array<T, D>
where
    D: Dimension,
{
    fn random_bulk<R, Dist>(shape: D, distribution: Dist, rng: &mut Random<R>) -> Self
    where
        R: Rng,
        Dist: Distribution<T> + Copy,
    {
        let size = shape.size();
        let mut data = Vec::with_capacity(size);

        // Use single RNG instance for all elements (performance optimization)
        for _ in 0..size {
            data.push(distribution.sample(&mut rng.rng));
        }

        Array::from_shape_vec(shape, data).expect("Operation failed")
    }

    fn random_using_bulk<R, F>(shape: D, rng: &mut Random<R>, mut f: F) -> Self
    where
        R: Rng,
        F: FnMut(&mut Random<R>) -> T,
    {
        let size = shape.size();
        let mut data = Vec::with_capacity(size);

        // Use single RNG instance for all elements
        for _ in 0..size {
            data.push(f(rng));
        }

        Array::from_shape_vec(shape, data).expect("Operation failed")
    }
}

/// Convenience functions for common array operations
/// Generate a random array with uniform distribution [0, 1)
pub fn random_uniform_array<D: Dimension>(shape: D, rng: &mut Random<impl Rng>) -> Array<f64, D> {
    Array::random_bulk(
        shape,
        Uniform::new(0.0, 1.0).expect("Operation failed"),
        rng,
    )
}

/// Generate a random array with normal distribution
pub fn random_normal_array<D: Dimension>(
    shape: D,
    mean: f64,
    std_dev: f64,
    rng: &mut Random<impl Rng>,
) -> Array<f64, D> {
    Array::random_bulk(
        shape,
        Normal::new(mean, std_dev).expect("Operation failed"),
        rng,
    )
}

/// Generate a random array with exponential distribution
pub fn random_exponential_array<D: Dimension>(
    shape: D,
    lambda: f64,
    rng: &mut Random<impl Rng>,
) -> Array<f64, D> {
    Array::random_bulk(shape, Exp::new(lambda).expect("Operation failed"), rng)
}

/// Generate a random array with gamma distribution
pub fn random_gamma_array<D: Dimension>(
    shape: D,
    alpha: f64,
    beta: f64,
    rng: &mut Random<impl Rng>,
) -> Array<f64, D> {
    Array::random_bulk(
        shape,
        Gamma::new(alpha, beta).expect("Operation failed"),
        rng,
    )
}

/// Generate a random sparse array (with specified sparsity ratio)
pub fn random_sparse_array<D: Dimension>(
    shape: D,
    sparsity: f64,
    rng: &mut Random<impl Rng>,
) -> Array<f64, D> {
    Array::random_using_bulk(shape, rng, |rng| {
        if rng.random_range(0.0..1.0) < sparsity {
            0.0
        } else {
            rng.random_range(-1.0..1.0)
        }
    })
}

/// Generate random weights for neural networks (Xavier/Glorot initialization)
pub fn random_xavier_weights(
    fan_in: usize,
    fan_out: usize,
    rng: &mut Random<impl Rng>,
) -> Array<f64, Ix2> {
    let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
    Array::random_bulk(
        crate::ndarray::Ix2(fan_out, fan_in),
        Uniform::new(-limit, limit).expect("Operation failed"),
        rng,
    )
}

/// Generate random weights using He initialization
pub fn random_he_weights(
    fan_in: usize,
    fan_out: usize,
    rng: &mut Random<impl Rng>,
) -> Array<f64, Ix2> {
    let std_dev = (2.0 / fan_in as f64).sqrt();
    Array::random_bulk(
        crate::ndarray::Ix2(fan_out, fan_in),
        Normal::new(0.0, std_dev).expect("Operation failed"),
        rng,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::core::seeded_rng;
    use ::ndarray::Ix2;

    #[test]
    fn test_optimized_array_random_bulk() {
        let mut rng = seeded_rng(42);
        let shape = Ix2(5, 5);

        // Test random_bulk method
        let array = Array::<f64, _>::random_bulk(
            shape,
            Uniform::new(0.0, 1.0).expect("Operation failed"),
            &mut rng,
        );
        assert_eq!(array.shape(), &[5, 5]);
        assert!(array.iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn test_optimized_array_random_using_bulk() {
        let mut rng = seeded_rng(123);
        let shape = Ix2(3, 4);

        // Test random_using_bulk method
        let array =
            Array::<i32, _>::random_using_bulk(shape, &mut rng, |rng| rng.random_range(1..100));
        assert_eq!(array.shape(), &[3, 4]);
        assert!(array.iter().all(|&x| (1..100).contains(&x)));
    }

    #[test]
    fn test_random_uniform_array() {
        let mut rng = seeded_rng(456);
        let array = random_uniform_array(Ix2(10, 10), &mut rng);

        assert_eq!(array.shape(), &[10, 10]);
        assert!(array.iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn test_random_normal_array() {
        let mut rng = seeded_rng(789);
        let array = random_normal_array(Ix2(5, 5), 0.0, 1.0, &mut rng);

        assert_eq!(array.shape(), &[5, 5]);
        // Normal distribution can produce any real value, so just check shape
    }

    #[test]
    fn test_random_exponential_array() {
        let mut rng = seeded_rng(101112);
        let array = random_exponential_array(Ix2(3, 3), 1.0, &mut rng);

        assert_eq!(array.shape(), &[3, 3]);
        assert!(array.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_random_gamma_array() {
        let mut rng = seeded_rng(131415);
        let array = random_gamma_array(Ix2(4, 4), 2.0, 1.0, &mut rng);

        assert_eq!(array.shape(), &[4, 4]);
        assert!(array.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_random_sparse_array() {
        let mut rng = seeded_rng(161718);
        let array = random_sparse_array(Ix2(6, 6), 0.7, &mut rng);

        assert_eq!(array.shape(), &[6, 6]);
        let zero_count = array.iter().filter(|&&x| x == 0.0).count();
        assert!(zero_count > 0); // Should have some zeros due to sparsity
    }

    #[test]
    fn test_neural_network_weight_initialization() {
        let mut rng = seeded_rng(192021);

        // Test Xavier weights
        let xavier_weights = random_xavier_weights(10, 5, &mut rng);
        assert_eq!(xavier_weights.shape(), &[5, 10]);

        // Test He weights
        let he_weights = random_he_weights(10, 5, &mut rng);
        assert_eq!(he_weights.shape(), &[5, 10]);
    }
}
