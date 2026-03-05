//! Monte Carlo integration methods
//!
//! This module provides numerical integration methods based on Monte Carlo
//! sampling, which are particularly useful for high-dimensional integrals.
//!
//! ## Submodules
//!
//! | Submodule | Description |
//! |-----------|-------------|
//! | [`mod@importance_sampling`] | IS, self-normalized IS, sequential IS, ESS, stratified sampling |
//! | [`quasi_monte_carlo`] | Halton, Sobol, lattice rules, Owen scrambling |
//! | [`multilevel_mc`] | MLMC estimator, adaptive levels, GBM hierarchy |
//! | [`basic_mc`] | MonteCarloIntegrator: 1-D, N-D, antithetic, control variate, stratified |
//! | [`path_sampling`] | Feynman-Kac, diffusion bridge, Metropolis walk, AIS |

pub mod basic_mc;
pub mod importance_sampling;
pub mod multilevel_mc;
pub mod path_sampling;
pub mod quasi_monte_carlo;

// Re-export key items from submodules for convenience
// basic_mc types are accessible as monte_carlo::basic_mc::{MCResult, MonteCarloIntegrator}
// to avoid conflict with monte_carlo_advanced::MonteCarloIntegrator
pub use importance_sampling::{
    effective_sample_size, importance_sampling_integral, self_normalized_is, sequential_is,
    stratified_sampling, ImportanceSamplingResult, StratifiedResult,
};
pub use multilevel_mc::{
    geometric_brownian_motion_levels, mlmc_adaptive, mlmc_complexity, mlmc_integrate, MLMCLevel,
    MLMCOptions, MLMCResult,
};
pub use path_sampling::{
    annealed_importance_sampling, diffusion_bridge, metropolis_walk, AISOptions, AISResult,
    DiffusionBridgeResult, FeynmanKacEstimator, FeynmanKacOptions, MetropolisWalkOptions,
    MetropolisWalkResult,
};
pub use quasi_monte_carlo::{
    qmc_integrate, scrambled_net, HaltonSequence, LatticeRule, QmcIntegrateOptions,
    QmcIntegrateResult, SobolSequence,
};

use crate::error::{IntegrateError, IntegrateResult};
use crate::IntegrateFloat;
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::random::prelude::*;
use scirs2_core::random::uniform::SampleUniform;
use scirs2_core::random::{Distribution, Uniform};
use std::f64::consts::PI;
use std::fmt::Debug;
use std::marker::PhantomData;

/// Options for controlling the behavior of Monte Carlo integration
#[derive(Debug, Clone)]
pub struct MonteCarloOptions<F: IntegrateFloat> {
    /// Number of sample points to use
    pub n_samples: usize,
    /// Random number generator seed (for reproducibility)
    pub seed: Option<u64>,
    /// Error estimation method
    pub error_method: ErrorEstimationMethod,
    /// Use antithetic variates for variance reduction
    pub use_antithetic: bool,
    /// Phantom data for generic type
    pub phantom: PhantomData<F>,
}

/// Method for estimating the error in Monte Carlo integration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorEstimationMethod {
    /// Standard error of the mean
    StandardError,
    /// Batch means method
    BatchMeans,
}

impl<F: IntegrateFloat> Default for MonteCarloOptions<F> {
    fn default() -> Self {
        Self {
            n_samples: 10000,
            seed: None,
            error_method: ErrorEstimationMethod::StandardError,
            use_antithetic: false,
            phantom: PhantomData,
        }
    }
}

/// Result of a Monte Carlo integration
#[derive(Debug, Clone)]
pub struct MonteCarloResult<F: IntegrateFloat> {
    /// Estimated value of the integral
    pub value: F,
    /// Estimated standard error
    pub std_error: F,
    /// Number of function evaluations
    pub n_evals: usize,
}

/// Perform Monte Carlo integration of a function over a hypercube
///
/// # Arguments
///
/// * `f` - The function to integrate
/// * `ranges` - Integration ranges (a, b) for each dimension
/// * `options` - Optional Monte Carlo parameters
///
/// # Returns
///
/// * `IntegrateResult<MonteCarloResult<F>>` - The result of the integration
///
/// # Examples
///
/// ```
/// use scirs2_integrate::monte_carlo::{monte_carlo, MonteCarloOptions};
/// use scirs2_core::ndarray::ArrayView1;
/// use std::marker::PhantomData;
///
/// // Integrate f(x,y) = x²+y² over [0,1]×[0,1] (exact result: 2/3)
/// let options = MonteCarloOptions {
///     n_samples: 100000,  // Use more samples for better accuracy
///     phantom: PhantomData,
///     ..Default::default()
/// };
///
/// let result = monte_carlo(
///     |x: ArrayView1<f64>| x[0] * x[0] + x[1] * x[1],
///     &[(0.0, 1.0), (0.0, 1.0)],
///     Some(options)
/// ).expect("Operation failed");
///
/// // Should be close to 2/3, but Monte Carlo has statistical error
/// assert!((result.value - 2.0/3.0).abs() < 0.01);
/// ```
#[allow(dead_code)]
pub fn monte_carlo<F, Func>(
    f: Func,
    ranges: &[(F, F)],
    options: Option<MonteCarloOptions<F>>,
) -> IntegrateResult<MonteCarloResult<F>>
where
    F: IntegrateFloat + Send + Sync + SampleUniform,
    Func: Fn(ArrayView1<F>) -> F + Sync,
    scirs2_core::random::StandardNormal: Distribution<F>,
{
    let opts = options.unwrap_or_default();
    let n_dims = ranges.len();

    if n_dims == 0 {
        return Err(IntegrateError::ValueError(
            "Integration ranges cannot be empty".to_string(),
        ));
    }

    if opts.n_samples == 0 {
        return Err(IntegrateError::ValueError(
            "Number of samples must be positive".to_string(),
        ));
    }

    // Calculate the volume of the integration domain
    let mut volume = F::one();
    for &(a, b) in ranges {
        volume *= b - a;
    }

    // Initialize random number generator
    let mut rng = if let Some(seed) = opts.seed {
        StdRng::seed_from_u64(seed)
    } else {
        let mut os_rng = scirs2_core::random::rng();
        StdRng::from_rng(&mut os_rng)
    };

    // Create uniform distributions for each dimension
    let distributions: Vec<_> = ranges
        .iter()
        .map(|&(a, b)| Uniform::new_inclusive(a, b).expect("valid range for Uniform distribution"))
        .collect();

    // Prepare to store samples
    let n_actual_samples = if opts.use_antithetic {
        opts.n_samples / 2 * 2 // Ensure even number for antithetic pairs
    } else {
        opts.n_samples
    };

    // Buffer for a single sample point
    let mut point = Array1::zeros(n_dims);

    // Sample and evaluate the function
    let mut sum = F::zero();
    let mut sum_sq = F::zero();
    let mut n_evals = 0;

    if opts.use_antithetic {
        // Use antithetic sampling for variance reduction
        for _ in 0..(n_actual_samples / 2) {
            for (i, dist) in distributions.iter().enumerate() {
                point[i] = dist.sample(&mut rng);
            }
            let value = f(point.view());
            sum += value;
            sum_sq += value * value;
            n_evals += 1;

            // Antithetic point: reflect around the center of the hypercube
            for (i, &(a, b)) in ranges.iter().enumerate() {
                point[i] = a + b - point[i];
            }
            let antithetic_value = f(point.view());
            sum += antithetic_value;
            sum_sq += antithetic_value * antithetic_value;
            n_evals += 1;
        }
    } else {
        // Standard Monte Carlo sampling
        for _ in 0..n_actual_samples {
            for (i, dist) in distributions.iter().enumerate() {
                point[i] = dist.sample(&mut rng);
            }
            let value = f(point.view());
            sum += value;
            sum_sq += value * value;
            n_evals += 1;
        }
    }

    let mean = sum / F::from_usize(n_actual_samples).expect("n_actual_samples fits in F");
    let integral_value = mean * volume;

    let std_error = match opts.error_method {
        ErrorEstimationMethod::StandardError | ErrorEstimationMethod::BatchMeans => {
            let variance = (sum_sq
                - sum * sum / F::from_usize(n_actual_samples).expect("n_actual_samples fits in F"))
                / F::from_usize(n_actual_samples - 1).expect("n_actual_samples-1 fits in F");
            (variance / F::from_usize(n_actual_samples).expect("n_actual_samples fits in F")).sqrt()
                * volume
        }
    };

    Ok(MonteCarloResult {
        value: integral_value,
        std_error,
        n_evals,
    })
}

/// Perform Monte Carlo integration with importance sampling (legacy API)
///
/// For the more complete importance sampling API see
/// [`importance_sampling::importance_sampling_integral`].
///
/// # Examples
///
/// ```
/// use scirs2_integrate::monte_carlo::{importance_sampling, MonteCarloOptions};
/// use scirs2_core::ndarray::{Array1, ArrayView1};
/// use scirs2_core::random::prelude::*;
///
/// // Uniform sampler for [0,1]
/// let uniform_sampler = |rng: &mut StdRng, dims: usize| {
///     let mut point = Array1::zeros(dims);
///     for i in 0..dims {
///         point[i] = rng.random_range(0.0..1.0);
///     }
///     point
/// };
///
/// // Uniform PDF on [0,1]
/// let uniform_pdf = |x: ArrayView1<f64>| -> f64 {
///     for &xi in x.iter() {
///         if xi < 0.0 || xi > 1.0 { return 0.0; }
///     }
///     1.0
/// };
///
/// let options = MonteCarloOptions {
///     n_samples: 10000,
///     seed: Some(42),
///     ..Default::default()
/// };
///
/// let result = importance_sampling(
///     |x: ArrayView1<f64>| x[0] * x[0],
///     uniform_pdf,
///     uniform_sampler,
///     &[(0.0, 1.0)],
///     Some(options)
/// ).expect("Operation failed");
///
/// assert!((result.value - 1.0/3.0).abs() < 0.01);
/// ```
#[allow(dead_code)]
pub fn importance_sampling<F, Func, Pdf, Sampler>(
    f: Func,
    g: Pdf,
    sampler: Sampler,
    ranges: &[(F, F)],
    options: Option<MonteCarloOptions<F>>,
) -> IntegrateResult<MonteCarloResult<F>>
where
    F: IntegrateFloat + Send + Sync + SampleUniform,
    Func: Fn(ArrayView1<F>) -> F + Sync,
    Pdf: Fn(ArrayView1<F>) -> F + Sync,
    Sampler: Fn(&mut StdRng, usize) -> Array1<F> + Sync,
    scirs2_core::random::StandardNormal: Distribution<F>,
{
    let opts = options.unwrap_or_default();
    let n_dims = ranges.len();

    if n_dims == 0 {
        return Err(IntegrateError::ValueError(
            "Integration ranges cannot be empty".to_string(),
        ));
    }

    if opts.n_samples == 0 {
        return Err(IntegrateError::ValueError(
            "Number of samples must be positive".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = opts.seed {
        StdRng::seed_from_u64(seed)
    } else {
        let mut os_rng = scirs2_core::random::rng();
        StdRng::from_rng(&mut os_rng)
    };

    let mut sum = F::zero();
    let mut sum_sq = F::zero();
    let n_samples = opts.n_samples;
    let mut valid_samples = 0;

    for _ in 0..n_samples {
        let point = sampler(&mut rng, n_dims);

        let fx = f(point.view());
        let gx = g(point.view());

        if gx <= F::from_f64(1e-10).expect("1e-10 fits in F") {
            continue;
        }

        if fx.is_nan() || fx.is_infinite() || gx.is_nan() || gx.is_infinite() {
            continue;
        }

        let ratio = fx / gx;

        if ratio.is_nan() || ratio.is_infinite() {
            continue;
        }

        sum += ratio;
        sum_sq += ratio * ratio;
        valid_samples += 1;
    }

    if valid_samples < 10 {
        return Err(IntegrateError::ConvergenceError(
            "Too few valid samples in importance sampling".to_string(),
        ));
    }

    let valid_samples_f = F::from_usize(valid_samples).expect("valid_samples fits in F");
    let mean = sum / valid_samples_f;

    let variance = if valid_samples > 1 {
        (sum_sq - sum * sum / valid_samples_f)
            / F::from_usize(valid_samples - 1).expect("valid_samples-1 fits in F")
    } else {
        F::zero()
    };

    let std_error = (variance / valid_samples_f).sqrt();

    Ok(MonteCarloResult {
        value: mean,
        std_error,
        n_evals: valid_samples,
    })
}

/// Parallel Monte Carlo integration using multiple threads
///
/// # Examples
///
/// ```rust
/// use scirs2_integrate::monte_carlo::{monte_carlo_parallel, MonteCarloOptions};
/// use scirs2_core::ndarray::ArrayView1;
/// use std::marker::PhantomData;
///
/// let options = MonteCarloOptions {
///     n_samples: 100000,
///     ..Default::default()
/// };
///
/// let result = monte_carlo_parallel(
///     |x: ArrayView1<f64>| (x[0] * x[1]).sin() * (-x[0]*x[0] - x[1]*x[1]).exp(),
///     &[(-2.0, 2.0), (-2.0, 2.0)],
///     Some(options),
///     Some(4)
/// ).expect("Operation failed");
/// ```
#[allow(dead_code)]
pub fn monte_carlo_parallel<F, Func>(
    f: Func,
    ranges: &[(F, F)],
    options: Option<MonteCarloOptions<F>>,
    workers: Option<usize>,
) -> IntegrateResult<MonteCarloResult<F>>
where
    F: IntegrateFloat + Send + Sync + SampleUniform,
    Func: Fn(ArrayView1<F>) -> F + Sync + Send,
    scirs2_core::random::StandardNormal: Distribution<F>,
{
    #[cfg(feature = "parallel")]
    {
        if workers.is_some() {
            use crate::monte_carlo_parallel::{parallel_monte_carlo, ParallelMonteCarloOptions};

            let opts = options.unwrap_or_default();
            let parallel_opts = ParallelMonteCarloOptions {
                n_samples: opts.n_samples,
                seed: opts.seed,
                error_method: opts.error_method,
                use_antithetic: opts.use_antithetic,
                n_threads: workers,
                batch_size: 1000,
                use_chunking: true,
                phantom: PhantomData,
            };

            return parallel_monte_carlo(f, ranges, Some(parallel_opts));
        }
    }

    let _ = workers;
    monte_carlo(f, ranges, options)
}

#[cfg(test)]
mod tests {
    use crate::{importance_sampling, monte_carlo, monte_carlo_parallel, MonteCarloOptions};
    use scirs2_core::ndarray::{Array1, ArrayView1};
    use scirs2_core::random::prelude::StdRng;
    use scirs2_core::random::Distribution;
    use std::f64::consts::PI;
    use std::marker::PhantomData;

    fn is_close_enough(result: f64, expected: f64, epsilon: f64) -> bool {
        (result - expected).abs() < epsilon
    }

    #[test]
    fn test_monte_carlo_1d() {
        let options = MonteCarloOptions {
            n_samples: 100000,
            seed: Some(12345),
            phantom: PhantomData,
            ..Default::default()
        };

        let result =
            monte_carlo(|x| x[0] * x[0], &[(0.0, 1.0)], Some(options)).expect("Operation failed");

        assert!(is_close_enough(result.value, 1.0 / 3.0, 0.01));
    }

    #[test]
    fn test_monte_carlo_2d() {
        let options = MonteCarloOptions {
            n_samples: 100000,
            seed: Some(12345),
            phantom: PhantomData,
            ..Default::default()
        };

        let result = monte_carlo(
            |x| x[0] * x[0] + x[1] * x[1],
            &[(0.0, 1.0), (0.0, 1.0)],
            Some(options),
        )
        .expect("Failed to integrate");

        assert!(is_close_enough(result.value, 2.0 / 3.0, 0.01));
    }

    #[test]
    fn test_monte_carlo_with_antithetic() {
        let options = MonteCarloOptions {
            n_samples: 100000,
            seed: Some(12345),
            use_antithetic: true,
            phantom: PhantomData,
            ..Default::default()
        };

        let result =
            monte_carlo(|x| x[0] * x[0], &[(0.0, 1.0)], Some(options)).expect("Operation failed");

        assert!(is_close_enough(result.value, 1.0 / 3.0, 0.01));
    }

    #[test]
    fn test_importance_sampling() {
        let sampler = |rng: &mut StdRng, dims: usize| {
            let mut point = Array1::zeros(dims);
            let normal = scirs2_core::random::Normal::new(0.0, 1.0).expect("valid normal params");

            for i in 0..dims {
                let mut x: f64 = normal.sample(rng);
                x = x.abs();
                if x > 3.0 {
                    x = 6.0 - x;
                    if x < 0.0 {
                        x = 0.0;
                    }
                }
                point[i] = x;
            }
            point
        };

        let normal_pdf = |x: ArrayView1<f64>| {
            let mut pdf_val = 1.0;
            for &xi in x.iter() {
                let z = xi;
                let density = (-0.5 * z * z).exp() / (2.0f64 * PI).sqrt();
                let folded_density = if xi < 3.0 { 2.0 * density } else { 0.0 };
                pdf_val *= folded_density.max(1e-10);
            }
            pdf_val
        };

        let options = MonteCarloOptions {
            n_samples: 100000,
            seed: Some(12345),
            phantom: PhantomData,
            ..Default::default()
        };

        let exact_value = (PI).sqrt() / 2.0 * libm::erf(3.0);

        let result = importance_sampling(
            |x| (-x[0] * x[0]).exp(),
            normal_pdf,
            sampler,
            &[(0.0, 3.0)],
            Some(options),
        )
        .expect("Failed to integrate");

        assert!(is_close_enough(result.value, exact_value, 0.1));
        println!(
            "Importance sampling test: got {}, expected {}",
            result.value, exact_value
        );
    }

    #[test]
    fn test_monte_carlo_parallel_workers() {
        let f = |x: ArrayView1<f64>| x[0].powi(2);
        let ranges = vec![(0.0, 1.0)];
        let options = MonteCarloOptions {
            n_samples: 10000,
            ..Default::default()
        };

        let result =
            monte_carlo_parallel(f, &ranges, Some(options), Some(2)).expect("Operation failed");

        assert!(is_close_enough(result.value, 1.0 / 3.0, 0.1));
        assert!(result.std_error >= 0.0);
    }
}
