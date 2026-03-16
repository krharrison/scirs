//! Core random number generation functionality for SCIRS2 ecosystem
//!
//! This module provides the foundational Random struct and traits that serve as the
//! basis for all random number generation across the SCIRS2 scientific computing ecosystem.

use ::ndarray::{Array, Dimension, Ix2, IxDyn};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Uniform};
use std::cell::RefCell;
use std::convert::Infallible;

/// Enhanced random number generator for scientific computing
///
/// This is the core random number generator used throughout the SCIRS2 ecosystem.
/// It provides deterministic, high-quality random number generation with support
/// for seeding, thread-local storage, and scientific reproducibility.
#[derive(Debug)]
pub struct Random<R = rand::rngs::ThreadRng> {
    pub(crate) rng: R,
}

impl Default for Random<rand::rngs::ThreadRng> {
    fn default() -> Self {
        Random { rng: rand::rng() }
    }
}

impl Random<StdRng> {
    /// Create a new random number generator with a specific seed
    ///
    /// This ensures deterministic behavior across runs, which is critical
    /// for scientific reproducibility and testing.
    pub fn seed(seed: u64) -> Random<StdRng> {
        Random {
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

// Implement SeedableRng for Random<StdRng> to support ecosystem requirements
impl SeedableRng for Random<StdRng> {
    type Seed = <StdRng as SeedableRng>::Seed;

    fn from_seed(seed: Self::Seed) -> Self {
        Random {
            rng: StdRng::from_seed(seed),
        }
    }

    fn seed_from_u64(state: u64) -> Self {
        Random {
            rng: StdRng::seed_from_u64(state),
        }
    }
}

/// Create a seeded random number generator (convenience function)
///
/// This is the primary way to create deterministic RNGs across the SCIRS2 ecosystem.
pub fn seeded_rng(seed: u64) -> Random<StdRng> {
    Random::seed_from_u64(seed)
}

/// Get a thread-local random number generator (convenience function)
///
/// This provides fast access to a thread-local RNG for performance-critical code.
pub fn thread_rng() -> Random<rand::rngs::ThreadRng> {
    Random::default()
}

// Implement TryRng for Random to forward to inner RNG
// In rand_core 0.10, TryRng is the base trait; Rng and RngCore are auto-implemented
// for TryRng<Error = Infallible>.
impl<R: Rng> rand::TryRng for Random<R> {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(self.rng.next_u32())
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        Ok(self.rng.next_u64())
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Self::Error> {
        self.rng.fill_bytes(dest);
        Ok(())
    }
}

// Implement Distribution sampling methods
impl<R: Rng> Random<R> {
    /// Sample from a distribution
    pub fn sample<T, D: Distribution<T>>(&mut self, distribution: D) -> T {
        distribution.sample(&mut self.rng)
    }

    /// Generate a random value in a range
    pub fn random_range<T, B>(&mut self, range: B) -> T
    where
        T: rand_distr::uniform::SampleUniform,
        B: rand_distr::uniform::SampleRange<T>,
    {
        rand::RngExt::random_range(&mut self.rng, range)
    }

    /// Generate a random boolean
    pub fn random_bool(&mut self, p: f64) -> bool {
        rand::RngExt::random_bool(&mut self.rng, p)
    }

    /// Generate a random value of the inferred type
    pub fn random<T>(&mut self) -> T
    where
        rand_distr::StandardUniform: rand_distr::Distribution<T>,
    {
        rand::RngExt::random(&mut self.rng)
    }

    /// Backward-compat alias for `random_range`
    pub fn gen_range<T, B>(&mut self, range: B) -> T
    where
        T: rand_distr::uniform::SampleUniform,
        B: rand_distr::uniform::SampleRange<T>,
    {
        self.random_range(range)
    }

    /// Backward-compat alias for `random_bool`
    pub fn gen_bool(&mut self, p: f64) -> bool {
        self.random_bool(p)
    }

    /// Fill a slice with random values
    pub fn fill<T>(&mut self, slice: &mut [T])
    where
        rand_distr::StandardUniform: rand_distr::Distribution<T>,
    {
        for item in slice.iter_mut() {
            *item = rand::RngExt::random(&mut self.rng);
        }
    }

    /// Generate a vector of random values
    pub fn sample_vec<T, D>(&mut self, distribution: D, size: usize) -> Vec<T>
    where
        D: Distribution<T> + Copy,
    {
        (0..size).map(|_| self.sample(distribution)).collect()
    }

    /// Generate a random array with specified shape and distribution
    pub fn sample_array<T, Dim, D>(&mut self, shape: Dim, distribution: D) -> Array<T, Dim>
    where
        Dim: Dimension,
        D: Distribution<T> + Copy,
    {
        let size = shape.size();
        let values: Vec<T> = (0..size).map(|_| self.sample(distribution)).collect();
        Array::from_shape_vec(shape, values).expect("Operation failed")
    }

    /// Access the underlying RNG (for advanced use cases)
    pub fn rng_mut(&mut self) -> &mut R {
        &mut self.rng
    }

    /// Access the underlying RNG (read-only)
    pub fn rng(&self) -> &R {
        &self.rng
    }
}

/// Extension trait for distributions to create arrays directly
///
/// This provides a consistent interface for generating random arrays
/// across the SCIRS2 ecosystem.
pub trait DistributionExt<T>: Distribution<T> + Sized {
    /// Create a random array with values from this distribution
    fn random_array<R: Rng, Dim: Dimension>(&self, rng: &mut Random<R>, shape: Dim) -> Array<T, Dim>
    where
        Self: Copy,
    {
        rng.sample_array(shape, *self)
    }

    /// Create a random vector with values from this distribution
    fn sample_vec<R: Rng>(&self, rng: &mut Random<R>, size: usize) -> Vec<T>
    where
        Self: Copy,
    {
        rng.sample_vec(*self, size)
    }
}

// Implement the extension trait for all distributions
impl<D, T> DistributionExt<T> for D where D: Distribution<T> {}

thread_local! {
    static THREAD_RNG: RefCell<Random> = RefCell::new(Random::default());
}

/// Get a reference to the thread-local random number generator
#[allow(dead_code)]
pub fn get_rng<F, R>(f: F) -> R
where
    F: FnOnce(&mut Random) -> R,
{
    THREAD_RNG.with(|rng| f(&mut rng.borrow_mut()))
}

/// Scientific random number generation utilities
pub mod scientific {
    use super::*;

    /// Generate reproducible random sequences for scientific experiments
    pub struct ReproducibleSequence {
        seed: u64,
        sequence_id: u64,
    }

    impl ReproducibleSequence {
        /// Create a new reproducible sequence
        pub fn new(seed: u64) -> Self {
            Self {
                seed,
                sequence_id: 0,
            }
        }

        /// Get the next RNG in the sequence
        pub fn next_rng(&mut self) -> Random<StdRng> {
            let combined_seed = self.seed.wrapping_mul(31).wrapping_add(self.sequence_id);
            self.sequence_id += 1;
            Random::seed(combined_seed)
        }

        /// Reset the sequence
        pub fn reset(&mut self) {
            self.sequence_id = 0;
        }

        /// Get current sequence position
        pub fn position(&self) -> u64 {
            self.sequence_id
        }
    }

    /// Deterministic random state for reproducible experiments
    #[derive(Debug, Clone)]
    pub struct DeterministicState {
        pub seed: u64,
        pub call_count: u64,
    }

    impl DeterministicState {
        /// Create a new deterministic state
        pub fn new(seed: u64) -> Self {
            Self {
                seed,
                call_count: 0,
            }
        }

        /// Create an RNG from this state and advance the counter
        pub fn next_rng(&mut self) -> Random<StdRng> {
            let rng_seed = self.seed.wrapping_mul(31).wrapping_add(self.call_count);
            self.call_count += 1;
            Random::seed(rng_seed)
        }

        /// Get current state without advancing
        pub fn current_state(&self) -> (u64, u64) {
            (self.seed, self.call_count)
        }

        /// Get current position in the sequence
        pub fn position(&self) -> u64 {
            self.call_count
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_random_creation() {
        let mut rng = Random::default();
        let _val = rng.sample(Uniform::new(0.0, 1.0).expect("Operation failed"));
    }

    #[test]
    fn test_seeded_rng() {
        let mut rng1 = seeded_rng(42);
        let mut rng2 = seeded_rng(42);

        let val1 = rng1.sample(Uniform::new(0.0, 1.0).expect("Operation failed"));
        let val2 = rng2.sample(Uniform::new(0.0, 1.0).expect("Operation failed"));

        assert_eq!(val1, val2);
    }

    #[test]
    fn test_thread_rng() {
        let mut rng = thread_rng();
        let val = rng.sample(Uniform::new(0.0, 1.0).expect("Operation failed"));
        assert!((0.0..1.0).contains(&val));
    }

    #[test]
    fn test_reproducible_sequence() {
        let mut seq1 = scientific::ReproducibleSequence::new(123);
        let mut seq2 = scientific::ReproducibleSequence::new(123);

        let mut rng1_1 = seq1.next_rng();
        let mut rng1_2 = seq1.next_rng();

        let mut rng2_1 = seq2.next_rng();
        let mut rng2_2 = seq2.next_rng();

        let val1_1 = rng1_1.sample(Uniform::new(0.0, 1.0).expect("Operation failed"));
        let val1_2 = rng1_2.sample(Uniform::new(0.0, 1.0).expect("Operation failed"));

        let val2_1 = rng2_1.sample(Uniform::new(0.0, 1.0).expect("Operation failed"));
        let val2_2 = rng2_2.sample(Uniform::new(0.0, 1.0).expect("Operation failed"));

        assert_eq!(val1_1, val2_1);
        assert_eq!(val1_2, val2_2);
        assert_ne!(val1_1, val1_2);
    }

    #[test]
    fn test_deterministic_state() {
        let mut state1 = scientific::DeterministicState::new(456);
        let mut state2 = scientific::DeterministicState::new(456);

        let mut rng1 = state1.next_rng();
        let mut rng2 = state2.next_rng();

        let val1 = rng1.sample(Uniform::new(0.0, 1.0).expect("Operation failed"));
        let val2 = rng2.sample(Uniform::new(0.0, 1.0).expect("Operation failed"));

        assert_eq!(val1, val2);
        assert_eq!(state1.position(), state2.position());
    }

    #[test]
    fn test_sample_array() {
        let mut rng = seeded_rng(789);
        let array = rng.sample_array(Ix2(3, 3), Uniform::new(0.0, 1.0).expect("Operation failed"));

        assert_eq!(array.shape(), &[3, 3]);
        assert!(array.iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn test_distribution_ext() {
        let mut rng = seeded_rng(101112);
        let distribution = Uniform::new(-1.0, 1.0).expect("Operation failed");

        let vec = distribution.sample_vec(&mut rng, 10);
        assert_eq!(vec.len(), 10);
        assert!(vec.iter().all(|&x| (-1.0..1.0).contains(&x)));

        let array = distribution.random_array(&mut rng, Ix2(2, 5));
        assert_eq!(array.shape(), &[2, 5]);
    }
}
