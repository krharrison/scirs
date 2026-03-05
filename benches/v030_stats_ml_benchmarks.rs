//! SciRS2 v0.3.0 Statistics & Machine Learning Benchmark Suite
//!
//! Benchmarks for v0.3.0 statistics/ML additions:
//! - Gaussian Process regression: fit and predict throughput
//! - NUTS MCMC: samples/second, warm-up overhead
//! - Hamiltonian Monte Carlo: comparison with NUTS
//!
//! Performance Targets (v0.3.0):
//! - GP fit  (n=100, d=2): < 50 ms
//! - GP predict (n=1000): < 10 ms
//! - NUTS (100 samples, 5-dim): > 20 samples/sec
//! - HMC (100 samples, 5-dim):  > 50 samples/sec

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{ChaCha8Rng, SeedableRng, Uniform, Distribution};
use scirs2_stats::gaussian_process::{
    GaussianProcessRegressor, SquaredExponential, Matern52,
};
use scirs2_stats::mcmc::nuts::{NutsConfig, NutsSampler};
use scirs2_stats::mcmc::hamiltonian::{
    HamiltonianMonteCarlo, CustomDifferentiableTarget,
};
use std::hint::black_box;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Data generators
// ---------------------------------------------------------------------------

fn make_gp_training_data(n: usize, d: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let dist = Uniform::new(-3.0f64, 3.0f64).expect("uniform range");
    let noise = Uniform::new(-0.05f64, 0.05f64).expect("uniform range");

    let x_flat: Vec<f64> = (0..n * d).map(|_| dist.sample(&mut rng)).collect();
    let x_train = Array2::from_shape_vec((n, d), x_flat).expect("shape");

    let y: Vec<f64> = (0..n)
        .map(|i| {
            let row_sum: f64 = (0..d).map(|j| x_train[[i, j]].powi(2)).sum();
            row_sum.sqrt().sin() + noise.sample(&mut rng)
        })
        .collect();

    (x_train, Array1::from(y))
}

fn make_gp_test_data(n: usize, d: usize) -> Array2<f64> {
    let step = 6.0 / n as f64;
    let x: Vec<f64> = (0..n * d)
        .map(|k| -3.0 + (k / d) as f64 * step)
        .collect();
    Array2::from_shape_vec((n, d), x).expect("shape")
}

// ---------------------------------------------------------------------------
// Gaussian Process benchmarks
// ---------------------------------------------------------------------------

fn bench_gp_rbf_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("gp_rbf_fit");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let train_sizes: &[usize] = &[20, 50, 100, 200];
    let d = 2usize;

    for &n in train_sizes {
        let (x_train, y_train) = make_gp_training_data(n, d, 42);

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(
            BenchmarkId::new("gp_rbf_fit", n),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    // SquaredExponential default: length_scale=1.0, signal_variance=1.0
                    let kernel = SquaredExponential::new(1.0, 1.0);
                    let mut gpr = GaussianProcessRegressor::new(kernel);
                    let result = gpr.fit(black_box(&x_train), black_box(&y_train));
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn bench_gp_matern_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("gp_matern52_fit");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let train_sizes: &[usize] = &[20, 50, 100, 200];
    let d = 2usize;

    for &n in train_sizes {
        let (x_train, y_train) = make_gp_training_data(n, d, 77);

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(
            BenchmarkId::new("gp_matern52_fit", n),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let kernel = Matern52::new(1.0, 1.0);
                    let mut gpr = GaussianProcessRegressor::new(kernel);
                    let result = gpr.fit(black_box(&x_train), black_box(&y_train));
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn bench_gp_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("gp_predict");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let n_train = 80usize;
    let d = 2usize;
    let (x_train, y_train) = make_gp_training_data(n_train, d, 42);

    let kernel = SquaredExponential::new(1.0, 1.0);
    let mut gpr = GaussianProcessRegressor::new(kernel);
    gpr.fit(&x_train, &y_train).expect("GP fit in setup");

    let test_sizes: &[usize] = &[10, 50, 100, 500, 1000];

    for &n_test in test_sizes {
        let x_test = make_gp_test_data(n_test, d);

        group.throughput(Throughput::Elements(n_test as u64));

        group.bench_with_input(
            BenchmarkId::new("predict_mean", n_test),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let result = gpr.predict(black_box(&x_test));
                    black_box(result)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("predict_with_std", n_test),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let result = gpr.predict_with_std(black_box(&x_test));
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn bench_gp_high_dim(c: &mut Criterion) {
    let mut group = c.benchmark_group("gp_high_dimensional");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let n_train = 60usize;
    let dimensions: &[usize] = &[2, 5, 10, 20];

    for &d in dimensions {
        let (x_train, y_train) = make_gp_training_data(n_train, d, 99);
        let x_test = make_gp_test_data(50, d);

        group.throughput(Throughput::Elements((n_train * d) as u64));

        group.bench_with_input(
            BenchmarkId::new("gp_fit_predict_dim", d),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let kernel = SquaredExponential::new(1.0, 1.0);
                    let mut gpr = GaussianProcessRegressor::new(kernel);
                    let fit_ok = gpr.fit(black_box(&x_train), black_box(&y_train));
                    if fit_ok.is_ok() {
                        let predict_result = gpr.predict(black_box(&x_test));
                        black_box(predict_result)
                    } else {
                        black_box(fit_ok.map(|_| Array1::zeros(1)))
                    }
                })
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// MCMC benchmarks
// ---------------------------------------------------------------------------

/// Standard multivariate normal N(0, I) log-density + gradient combined
fn log_prob_and_grad(x: &[f64]) -> (f64, Vec<f64>) {
    let log_p = -0.5 * x.iter().map(|v| v * v).sum::<f64>();
    let grad: Vec<f64> = x.iter().map(|v| -v).collect();
    (log_p, grad)
}

fn bench_nuts_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("nuts_sampling");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let dims: &[usize] = &[2, 5, 10, 20];
    let n_samples = 50usize;

    for &dim in dims {
        let initial = vec![0.0f64; dim];
        let config = NutsConfig {
            step_size: 0.1,
            max_tree_depth: 5,
            target_accept: 0.8,
            adapt_step_size: false,
            ..NutsConfig::default()
        };

        group.throughput(Throughput::Elements(n_samples as u64));

        group.bench_with_input(
            BenchmarkId::new("nuts_samples", dim),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let mut sampler = NutsSampler::new(config.clone());
                    let result = sampler.sample(
                        log_prob_and_grad,
                        black_box(&initial),
                        black_box(n_samples),
                    );
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn bench_hmc_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("hmc_sampling");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let dims: &[usize] = &[2, 5, 10, 20];
    let n_samples = 50usize;

    for &dim in dims {
        group.throughput(Throughput::Elements(n_samples as u64));

        group.bench_with_input(
            BenchmarkId::new("hmc_samples", dim),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let mut rng = ChaCha8Rng::seed_from_u64(42);
                    let initial = Array1::zeros(dim);

                    let target_result = CustomDifferentiableTarget::new(
                        dim,
                        |x: &Array1<f64>| -0.5 * x.iter().map(|v| v * v).sum::<f64>(),
                        |x: &Array1<f64>| -x.clone(),
                    );

                    match target_result {
                        Ok(target) => {
                            let hmc_result = HamiltonianMonteCarlo::new(
                                target,
                                initial,
                                0.1,  // step size
                                10,   // leapfrog steps
                            );
                            match hmc_result {
                                Ok(mut sampler) => {
                                    let result = sampler.sample(
                                        black_box(n_samples),
                                        &mut rng,
                                    );
                                    black_box(result)
                                }
                                Err(e) => black_box(Err(e)),
                            }
                        }
                        Err(e) => black_box(Err(e)),
                    }
                })
            },
        );
    }
    group.finish();
}

fn bench_nuts_vs_hmc_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("nuts_vs_hmc");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let dim = 10usize;
    let n_samples = 100usize;

    group.throughput(Throughput::Elements(n_samples as u64));

    // NUTS
    group.bench_with_input(
        BenchmarkId::new("nuts", dim),
        &(),
        |bencher, _| {
            bencher.iter(|| {
                let initial = vec![0.0f64; dim];
                let config = NutsConfig {
                    step_size: 0.1,
                    max_tree_depth: 6,
                    target_accept: 0.8,
                    adapt_step_size: false,
                    ..NutsConfig::default()
                };
                let mut sampler = NutsSampler::new(config);
                let result = sampler.sample(
                    log_prob_and_grad,
                    black_box(&initial),
                    black_box(n_samples),
                );
                black_box(result)
            })
        },
    );

    // HMC
    group.bench_with_input(
        BenchmarkId::new("hmc", dim),
        &(),
        |bencher, _| {
            bencher.iter(|| {
                let mut rng = ChaCha8Rng::seed_from_u64(42);
                let initial = Array1::zeros(dim);

                let target_result = CustomDifferentiableTarget::new(
                    dim,
                    |x: &Array1<f64>| -0.5 * x.iter().map(|v| v * v).sum::<f64>(),
                    |x: &Array1<f64>| -x.clone(),
                );
                match target_result {
                    Ok(target) => {
                        let hmc_result = HamiltonianMonteCarlo::new(target, initial, 0.1, 10);
                        match hmc_result {
                            Ok(mut sampler) => {
                                let result = sampler.sample(black_box(n_samples), &mut rng);
                                black_box(result)
                            }
                            Err(e) => black_box(Err(e)),
                        }
                    }
                    Err(e) => black_box(Err(e)),
                }
            })
        },
    );

    group.finish();
}

fn bench_mcmc_chain_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcmc_chain_scaling");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let dim = 5usize;
    let sample_counts: &[usize] = &[20, 50, 100, 200];

    for &n in sample_counts {
        let initial = vec![0.0f64; dim];
        let config = NutsConfig {
            step_size: 0.1,
            max_tree_depth: 5,
            target_accept: 0.8,
            adapt_step_size: false,
            ..NutsConfig::default()
        };

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(
            BenchmarkId::new("nuts_chain", n),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let mut sampler = NutsSampler::new(config.clone());
                    let result = sampler.sample(
                        log_prob_and_grad,
                        black_box(&initial),
                        black_box(n),
                    );
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion groups and main
// ---------------------------------------------------------------------------

criterion_group!(
    gp_benchmarks,
    bench_gp_rbf_fit,
    bench_gp_matern_fit,
    bench_gp_predict,
    bench_gp_high_dim,
);

criterion_group!(
    mcmc_benchmarks,
    bench_nuts_sampling,
    bench_hmc_sampling,
    bench_nuts_vs_hmc_comparison,
    bench_mcmc_chain_scaling,
);

criterion_main!(gp_benchmarks, mcmc_benchmarks);
