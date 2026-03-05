//! SciRS2 v0.3.0 Optimization & Clustering Benchmark Suite
//!
//! Benchmarks for v0.3.0 additions in:
//! - Trust-region methods: Dogleg, TRCG, trust-NCG
//! - Bayesian optimization: GP-based BO with different acquisition functions
//! - GMM clustering: EM algorithm throughput (full / diagonal covariance)
//! - SOM: training throughput across grid sizes and topologies
//!
//! Performance Targets (v0.3.0):
//! - Trust-region (Rosenbrock, 10-dim): < 100 ms to convergence
//! - Bayesian opt (3-dim, 15 evaluations): < 5 s
//! - GMM fit (n=1000, k=5, d=10): < 1 s
//! - SOM train (n=1000, grid 10×10, d=8): < 2 s

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::{ChaCha8Rng, SeedableRng, Uniform, Distribution};
use scirs2_optimize::{
    trust_region_minimize, TrustRegionConfig,
    AdvancedBayesianOptimizer, BayesianOptimizerConfig,
};
use scirs2_cluster::{
    gaussian_mixture, CovarianceType, GMMInit, GMMOptions, GaussianMixture,
    Som, SomConfig, SomTopology,
};
use std::hint::black_box;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Test functions for optimization
// ---------------------------------------------------------------------------

/// n-dimensional Rosenbrock function
fn rosenbrock(x: &ArrayView1<f64>) -> f64 {
    x.windows(2)
        .into_iter()
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
}

fn rosenbrock_gradient(x: &ArrayView1<f64>) -> Array1<f64> {
    let n = x.len();
    let mut grad = Array1::zeros(n);
    for i in 0..n - 1 {
        grad[i] += -400.0 * x[i] * (x[i + 1] - x[i].powi(2)) - 2.0 * (1.0 - x[i]);
        grad[i + 1] += 200.0 * (x[i + 1] - x[i].powi(2));
    }
    grad
}

/// n-dimensional sphere function
fn sphere(x: &ArrayView1<f64>) -> f64 {
    x.iter().map(|v| v * v).sum()
}

fn sphere_gradient(x: &ArrayView1<f64>) -> Array1<f64> {
    x * 2.0
}

fn sphere_hessian(x: &ArrayView1<f64>) -> Array2<f64> {
    Array2::eye(x.len()) * 2.0
}

// ---------------------------------------------------------------------------
// Trust-region benchmarks
// ---------------------------------------------------------------------------

fn bench_trust_region_rosenbrock(c: &mut Criterion) {
    let mut group = c.benchmark_group("trust_region_rosenbrock");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let dimensions: &[usize] = &[2, 5, 10, 20];

    for &dim in dimensions {
        let x0: Array1<f64> = Array1::from_vec(vec![-1.2; dim]);
        let config = TrustRegionConfig {
            initial_radius: 1.0,
            max_radius: 100.0,
            eta: 0.15,
            gtol: 1e-5,
            max_iter: 500,
            ..TrustRegionConfig::default()
        };

        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(
            BenchmarkId::new("dogleg_gradient_only", dim),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    // Explicit type params: no hessian supplied
                    let result = trust_region_minimize::<
                        _,
                        fn(&ArrayView1<f64>) -> Array1<f64>,
                        fn(&ArrayView1<f64>) -> Array2<f64>,
                    >(
                        |x| rosenbrock(black_box(x)),
                        Some(|x: &ArrayView1<f64>| rosenbrock_gradient(x)),
                        None::<fn(&ArrayView1<f64>) -> Array2<f64>>,
                        black_box(x0.clone()),
                        Some(config.clone()),
                    );
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn bench_trust_region_sphere(c: &mut Criterion) {
    let mut group = c.benchmark_group("trust_region_sphere");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let dimensions: &[usize] = &[5, 10, 20, 50, 100];

    for &dim in dimensions {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let dist = Uniform::new(-2.0f64, 2.0f64).expect("uniform range");
        let x0 = Array1::from_iter((0..dim).map(|_| dist.sample(&mut rng)));

        let config = TrustRegionConfig {
            initial_radius: 1.0,
            max_radius: 10.0,
            eta: 0.15,
            gtol: 1e-8,
            max_iter: 200,
            ..TrustRegionConfig::default()
        };

        group.throughput(Throughput::Elements(dim as u64));

        // Gradient only
        group.bench_with_input(
            BenchmarkId::new("gradient_only", dim),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let result = trust_region_minimize::<
                        _,
                        fn(&ArrayView1<f64>) -> Array1<f64>,
                        fn(&ArrayView1<f64>) -> Array2<f64>,
                    >(
                        |x| sphere(black_box(x)),
                        Some(|x: &ArrayView1<f64>| sphere_gradient(x)),
                        None::<fn(&ArrayView1<f64>) -> Array2<f64>>,
                        black_box(x0.clone()),
                        Some(config.clone()),
                    );
                    black_box(result)
                })
            },
        );

        // Gradient + Hessian
        group.bench_with_input(
            BenchmarkId::new("gradient_hessian", dim),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let result = trust_region_minimize(
                        |x| sphere(black_box(x)),
                        Some(|x: &ArrayView1<f64>| sphere_gradient(x)),
                        Some(|x: &ArrayView1<f64>| sphere_hessian(x)),
                        black_box(x0.clone()),
                        Some(config.clone()),
                    );
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Bayesian optimization benchmarks
// ---------------------------------------------------------------------------

fn bench_bayesian_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("bayesian_optimization");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(15));

    let dimensions: &[usize] = &[1, 2, 5, 10];
    let n_iter = 15usize;

    for &dim in dimensions {
        let bounds: Vec<(f64, f64)> = (0..dim).map(|_| (-5.0, 5.0)).collect();
        let config = BayesianOptimizerConfig::default();

        group.throughput(Throughput::Elements(n_iter as u64));

        group.bench_with_input(
            BenchmarkId::new("bo_sphere_dim", dim),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let bo_result = AdvancedBayesianOptimizer::new(
                        black_box(bounds.clone()),
                        config.clone(),
                    );
                    match bo_result {
                        Ok(mut opt) => {
                            let result = opt.optimize(
                                |x: &Array1<f64>| x.iter().map(|v| v * v).sum::<f64>(),
                                black_box(n_iter),
                            );
                            black_box(result)
                        }
                        Err(e) => black_box(Err(e)),
                    }
                })
            },
        );
    }
    group.finish();
}

fn bench_bayesian_opt_iterations(c: &mut Criterion) {
    let mut group = c.benchmark_group("bayesian_opt_iteration_scaling");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(15));

    let dim = 3usize;
    let bounds: Vec<(f64, f64)> = (0..dim).map(|_| (-5.0, 5.0)).collect();
    let n_iterations: &[usize] = &[5, 10, 20, 30];

    for &n_iter in n_iterations {
        let config = BayesianOptimizerConfig::default();

        group.throughput(Throughput::Elements(n_iter as u64));

        group.bench_with_input(
            BenchmarkId::new("bo_iters", n_iter),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let bo_result = AdvancedBayesianOptimizer::new(
                        black_box(bounds.clone()),
                        config.clone(),
                    );
                    match bo_result {
                        Ok(mut opt) => {
                            let result = opt.optimize(
                                |x: &Array1<f64>| {
                                    // Cheap Ackley-style objective
                                    let df = dim as f64;
                                    let sum_sq: f64 = x.iter().map(|v| v * v).sum();
                                    let cos_sum: f64 = x
                                        .iter()
                                        .map(|v| (2.0 * std::f64::consts::PI * v).cos())
                                        .sum();
                                    -(-20.0 * (-0.2 * (sum_sq / df).sqrt()).exp()
                                        - (cos_sum / df).exp()
                                        + 20.0
                                        + std::f64::consts::E)
                                },
                                black_box(n_iter),
                            );
                            black_box(result)
                        }
                        Err(e) => black_box(Err(e)),
                    }
                })
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// GMM benchmarks
// ---------------------------------------------------------------------------

/// Generate clustered data: k clusters, n points total, d dimensions
fn make_cluster_data(n: usize, k: usize, d: usize, seed: u64) -> Array2<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let noise = Uniform::new(-0.5f64, 0.5f64).expect("uniform range");
    let center_range = Uniform::new(-5.0f64, 5.0f64).expect("uniform range");

    let centers: Vec<Vec<f64>> = (0..k)
        .map(|_| (0..d).map(|_| center_range.sample(&mut rng)).collect())
        .collect();

    let data: Vec<f64> = (0..n)
        .flat_map(|i| {
            let c = i % k;
            let centers_ref = &centers;
            (0..d).map(move |j| centers_ref[c][j] + noise.sample(&mut rng))
        })
        .collect();

    Array2::from_shape_vec((n, d), data).expect("valid shape")
}

fn bench_gmm_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("gmm_fit");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let configs: &[(usize, usize, usize)] = &[
        (500,  3, 5),
        (1000, 5, 5),
        (1000, 5, 10),
        (2000, 8, 10),
    ];

    for &(n, k, d) in configs {
        let data = make_cluster_data(n, k, d, 42);

        group.throughput(Throughput::Elements(n as u64));

        let label = format!("n={n}_k={k}_d={d}");

        // Full covariance
        group.bench_with_input(
            BenchmarkId::new("gmm_full_cov", &label),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let options: GMMOptions<f64> = GMMOptions {
                        n_components: k,
                        covariance_type: CovarianceType::Full,
                        tol: 1e-4,
                        max_iter: 200,
                        n_init: 1,
                        init_method: GMMInit::KMeans,
                        random_seed: Some(42),
                        reg_covar: 1e-6,
                    };
                    let mut gmm = GaussianMixture::new(options);
                    let result = gmm.fit(black_box(data.view()));
                    black_box(result)
                })
            },
        );

        // Diagonal covariance (faster)
        group.bench_with_input(
            BenchmarkId::new("gmm_diag_cov", &label),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let options: GMMOptions<f64> = GMMOptions {
                        n_components: k,
                        covariance_type: CovarianceType::Diagonal,
                        tol: 1e-4,
                        max_iter: 200,
                        n_init: 1,
                        init_method: GMMInit::KMeans,
                        random_seed: Some(42),
                        reg_covar: 1e-6,
                    };
                    let mut gmm = GaussianMixture::new(options);
                    let result = gmm.fit(black_box(data.view()));
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn bench_gmm_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("gmm_predict");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let n_train = 500usize;
    let k = 5usize;
    let d = 8usize;
    let train_data = make_cluster_data(n_train, k, d, 42);

    // Fit model in setup
    let options: GMMOptions<f64> = GMMOptions {
        n_components: k,
        covariance_type: CovarianceType::Full,
        tol: 1e-4,
        max_iter: 200,
        n_init: 1,
        init_method: GMMInit::KMeans,
        random_seed: Some(42),
        reg_covar: 1e-6,
    };
    let mut model = GaussianMixture::new(options);
    model.fit(train_data.view()).expect("GMM fit in setup");

    let test_sizes: &[usize] = &[100, 500, 1000, 5000];
    for &n_test in test_sizes {
        let x_test = make_cluster_data(n_test, k, d, 99);

        group.throughput(Throughput::Elements(n_test as u64));

        group.bench_with_input(
            BenchmarkId::new("gmm_predict", n_test),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let result = model.predict(black_box(x_test.view()));
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn bench_gmm_functional(c: &mut Criterion) {
    let mut group = c.benchmark_group("gmm_functional_api");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    // Using the top-level gaussian_mixture convenience function
    let n = 1000usize;
    let component_counts: &[usize] = &[3, 5, 8, 10];

    for &k in component_counts {
        let data = make_cluster_data(n, k, 6, 42);

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(
            BenchmarkId::new("gaussian_mixture_fn", k),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let options: GMMOptions<f64> = GMMOptions {
                        n_components: k,
                        covariance_type: CovarianceType::Diagonal,
                        tol: 1e-4,
                        max_iter: 100,
                        n_init: 1,
                        init_method: GMMInit::KMeans,
                        random_seed: Some(42),
                        reg_covar: 1e-6,
                    };
                    let result = gaussian_mixture(black_box(data.view()), options);
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// SOM benchmarks
// ---------------------------------------------------------------------------

fn bench_som_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("som_training");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(12));

    let data_configs: &[(usize, usize)] = &[
        (200, 4),
        (500, 8),
        (1000, 8),
        (2000, 16),
    ];
    let grid_sizes: &[(usize, usize)] = &[(8, 8), (10, 10)];

    for &(n, d) in data_configs {
        let data = make_cluster_data(n, 5, d, 42);

        for &(rows, cols) in grid_sizes {
            let config = SomConfig {
                grid_rows: rows,
                grid_cols: cols,
                n_iter: 500,
                learning_rate: 0.5,
                lr_decay: 1000.0,
                sigma: (rows.max(cols) / 2) as f64,
                sigma_decay: 1000.0,
                topology: SomTopology::Rectangular,
                random_seed: Some(42),
            };

            group.throughput(Throughput::Elements((n * rows * cols) as u64));

            let label = format!("n={n}_d={d}_grid={rows}x{cols}");
            group.bench_with_input(
                BenchmarkId::new("som_fit", &label),
                &(),
                |bencher, _| {
                    bencher.iter(|| {
                        let mut som = Som::new(d, config.clone());
                        let result = som.fit(black_box(&data));
                        black_box(result)
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_som_topologies(c: &mut Criterion) {
    let mut group = c.benchmark_group("som_topologies");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let n = 1000usize;
    let d = 8usize;
    let data = make_cluster_data(n, 5, d, 77);

    group.throughput(Throughput::Elements(n as u64));

    let topologies = [
        ("rectangular", SomTopology::Rectangular),
        ("hexagonal",   SomTopology::Hexagonal),
    ];

    for (name, topology) in &topologies {
        let config = SomConfig {
            grid_rows: 10,
            grid_cols: 10,
            n_iter: 500,
            learning_rate: 0.5,
            lr_decay: 1000.0,
            sigma: 5.0,
            sigma_decay: 1000.0,
            topology: *topology,
            random_seed: Some(42),
        };

        group.bench_with_input(
            BenchmarkId::new("som_topology", name),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let mut som = Som::new(d, config.clone());
                    let result = som.fit(black_box(&data));
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn bench_som_grid_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("som_grid_sizes");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let n = 1000usize;
    let d = 6usize;
    let data = make_cluster_data(n, 5, d, 55);

    let grid_sizes: &[(usize, usize)] = &[(5, 5), (8, 8), (10, 10), (12, 12), (15, 15)];

    for &(rows, cols) in grid_sizes {
        let config = SomConfig {
            grid_rows: rows,
            grid_cols: cols,
            n_iter: 300,
            learning_rate: 0.5,
            lr_decay: 500.0,
            sigma: rows.max(cols) as f64 / 2.0,
            sigma_decay: 500.0,
            topology: SomTopology::Rectangular,
            random_seed: Some(42),
        };

        group.throughput(Throughput::Elements((rows * cols) as u64));

        group.bench_with_input(
            BenchmarkId::new("som_grid", format!("{rows}x{cols}")),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let mut som = Som::new(d, config.clone());
                    let result = som.fit(black_box(&data));
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
    trust_region_benchmarks,
    bench_trust_region_rosenbrock,
    bench_trust_region_sphere,
);

criterion_group!(
    bayesian_opt_benchmarks,
    bench_bayesian_optimization,
    bench_bayesian_opt_iterations,
);

criterion_group!(
    clustering_benchmarks,
    bench_gmm_fit,
    bench_gmm_predict,
    bench_gmm_functional,
    bench_som_training,
    bench_som_topologies,
    bench_som_grid_sizes,
);

criterion_main!(trust_region_benchmarks, bayesian_opt_benchmarks, clustering_benchmarks);
