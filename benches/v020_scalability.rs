//! SciRS2 v0.2.0 Scalability Benchmarks
//!
//! This benchmark suite measures performance scaling characteristics:
//! - Single-threaded vs multi-threaded performance
//! - Scaling from 1K to 1M+ elements
//! - Thread count scaling (1, 2, 4, 8, 16 threads)
//! - Data size scaling (algorithmic complexity validation)
//! - Parallel efficiency measurement
//!
//! Performance Targets:
//! - Parallel efficiency: >80% for 2 threads, >60% for 4 threads, >40% for 8 threads
//! - Linear scaling for embarrassingly parallel operations
//! - Sub-linear but predictable scaling for data-dependent operations
//! - Algorithmic complexity matches theoretical predictions

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::{Distribution, Uniform};
use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, RandomExt};
use std::hint::black_box;
use std::time::Duration;

/// Helper to create a Uniform distribution without unwrap
fn uniform_f64(low: f64, high: f64) -> Uniform<f64> {
    Uniform::new(low, high).expect("Failed to create Uniform distribution")
}

// =============================================================================
// Thread Count Scaling
// =============================================================================

/// Benchmark parallel reduction with varying thread counts
fn bench_parallel_reduction_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability/parallel_reduction");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let size = 10_000_000;
    let data: Vec<f64> = (0..size).map(|i| (i as f64).sin()).collect();

    group.throughput(Throughput::Elements(size as u64));

    // Sequential baseline
    group.bench_function("sequential", |b| {
        b.iter(|| {
            let sum: f64 = data.iter().sum();
            black_box(sum)
        })
    });

    // Parallel with different thread pool sizes
    for &n_threads in &[2, 4, 8, 16] {
        let label = format!("{}threads", n_threads);
        group.bench_with_input(
            BenchmarkId::new("parallel", &label),
            &label,
            |bencher, _| {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(n_threads)
                    .build()
                    .expect("Failed to build thread pool");

                bencher.iter(|| {
                    pool.install(|| {
                        let sum: f64 = data.par_iter().sum();
                        black_box(sum)
                    })
                })
            },
        );
    }

    group.finish();
}

/// Benchmark parallel map with varying thread counts
fn bench_parallel_map_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability/parallel_map");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let size = 5_000_000;
    let data: Vec<f64> = (0..size).map(|i| i as f64).collect();

    group.throughput(Throughput::Elements(size as u64));

    // Sequential baseline
    group.bench_function("sequential", |b| {
        b.iter(|| {
            let result: Vec<f64> = data.iter().map(|x| x.sin()).collect();
            black_box(result)
        })
    });

    // Parallel with different thread counts
    for &n_threads in &[2, 4, 8, 16] {
        let label = format!("{}threads", n_threads);
        group.bench_with_input(
            BenchmarkId::new("parallel", &label),
            &label,
            |bencher, _| {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(n_threads)
                    .build()
                    .expect("Failed to build thread pool");

                bencher.iter(|| {
                    pool.install(|| {
                        let result: Vec<f64> = data.par_iter().map(|x| x.sin()).collect();
                        black_box(result)
                    })
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Data Size Scaling
// =============================================================================

/// Benchmark O(n) operations scaling
fn bench_linear_complexity_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability/linear_complexity");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    // Powers of 10 for clear scaling analysis
    let sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000];

    for &size in &sizes {
        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("sum", size), &size, |b, _| {
            b.iter(|| {
                let sum: f64 = data.iter().sum();
                black_box(sum)
            })
        });

        group.bench_with_input(BenchmarkId::new("map", size), &size, |b, _| {
            b.iter(|| {
                let result: Vec<f64> = data.iter().map(|x| x * 2.0).collect();
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark O(n log n) operations scaling
fn bench_n_log_n_complexity_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability/n_log_n_complexity");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let sizes = [1_000, 10_000, 100_000, 1_000_000];

    for &size in &sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = uniform_f64(-1000.0, 1000.0);
        let data: Vec<f64> = (0..size).map(|_| dist.sample(&mut rng)).collect();

        let complexity = (size as f64 * (size as f64).log2()) as u64;
        group.throughput(Throughput::Elements(complexity));

        group.bench_with_input(BenchmarkId::new("sort", size), &size, |b, _| {
            b.iter(|| {
                let mut sorted = data.clone();
                sorted.sort_by(|a_val: &f64, b_val: &f64| {
                    a_val.partial_cmp(b_val).expect("Failed comparison")
                });
                black_box(sorted)
            })
        });
    }

    group.finish();
}

/// Benchmark O(n²) operations scaling
fn bench_quadratic_complexity_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability/quadratic_complexity");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let sizes = [32, 64, 128, 256, 512];

    for &size in &sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = uniform_f64(-1.0, 1.0);

        let matrix = Array2::random_using((size, size), dist, &mut rng);

        let complexity = (size * size) as u64;
        group.throughput(Throughput::Elements(complexity));

        // Matrix-vector multiplication (O(n^2))
        let vector = Array1::random_using(size, dist, &mut rng);

        group.bench_with_input(BenchmarkId::new("matvec", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = matrix.dot(&vector);
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark O(n³) operations scaling
fn bench_cubic_complexity_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability/cubic_complexity");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let sizes = [16, 32, 64, 128, 256, 512];

    for &size in &sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = uniform_f64(-1.0, 1.0);

        let a = Array2::random_using((size, size), dist, &mut rng);
        let b_mat = Array2::random_using((size, size), dist, &mut rng);

        let complexity = (2 * size * size * size) as u64; // 2n^3 for GEMM
        group.throughput(Throughput::Elements(complexity));

        group.bench_with_input(BenchmarkId::new("matmul", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = a.dot(&b_mat);
                black_box(result)
            })
        });
    }

    group.finish();
}

// =============================================================================
// Parallel Efficiency Measurement
// =============================================================================

/// Benchmark parallel efficiency for matrix operations
fn bench_parallel_efficiency_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability/parallel_efficiency_matmul");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let size = 512;
    let mut rng = StdRng::seed_from_u64(42);
    let dist = uniform_f64(-1.0, 1.0);

    let a = Array2::random_using((size, size), dist, &mut rng);
    let b_mat = Array2::random_using((size, size), dist, &mut rng);

    let flops = (2 * size * size * size) as u64;
    group.throughput(Throughput::Elements(flops));

    // Sequential baseline
    group.bench_function("1thread", |bencher| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("Failed to build thread pool");

        bencher.iter(|| {
            pool.install(|| {
                let result = a.dot(&b_mat);
                black_box(result)
            })
        })
    });

    // Parallel with various thread counts
    for &n_threads in &[2, 4, 8] {
        let label = format!("{}threads", n_threads);
        group.bench_with_input(
            BenchmarkId::new("parallel", &label),
            &label,
            |bencher, _| {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(n_threads)
                    .build()
                    .expect("Failed to build thread pool");

                bencher.iter(|| {
                    pool.install(|| {
                        let result = a.dot(&b_mat);
                        black_box(result)
                    })
                })
            },
        );
    }

    group.finish();
}

/// Benchmark parallel efficiency for FFT
fn bench_parallel_efficiency_fft(c: &mut Criterion) {
    use scirs2_fft::fft;

    let mut group = c.benchmark_group("scalability/parallel_efficiency_fft");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let size = 1_048_576; // 1M points
    let mut rng = StdRng::seed_from_u64(42);
    let dist = uniform_f64(-1.0, 1.0);

    let signal = Array1::random_using(size, dist, &mut rng);
    let signal_slice: Vec<f64> = signal.iter().copied().collect();

    let complexity = (size as f64 * (size as f64).log2()) as u64;
    group.throughput(Throughput::Elements(complexity));

    // Sequential
    group.bench_function("1thread", |bencher| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("Failed to build thread pool");

        bencher.iter(|| {
            pool.install(|| {
                let result = fft(&signal_slice, None);
                black_box(result)
            })
        })
    });

    // Parallel
    for &n_threads in &[2, 4, 8] {
        let label = format!("{}threads", n_threads);
        group.bench_with_input(
            BenchmarkId::new("parallel", &label),
            &label,
            |bencher, _| {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(n_threads)
                    .build()
                    .expect("Failed to build thread pool");

                bencher.iter(|| {
                    pool.install(|| {
                        let result = fft(&signal_slice, None);
                        black_box(result)
                    })
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Batch Processing Scaling
// =============================================================================

/// Benchmark batch processing scaling
fn bench_batch_processing_scaling(c: &mut Criterion) {
    use scirs2_linalg::solve;

    let mut group = c.benchmark_group("scalability/batch_processing");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let matrix_size = 100;
    let batch_sizes = [1, 10, 100, 1000];

    for &batch_size in &batch_sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = uniform_f64(-1.0, 1.0);

        // Create batch of systems to solve
        let matrices: Vec<Array2<f64>> = (0..batch_size)
            .map(|_| {
                let mut a = Array2::random_using((matrix_size, matrix_size), dist, &mut rng);
                // Make diagonally dominant for stability
                for i in 0..matrix_size {
                    a[[i, i]] += (matrix_size as f64) * 2.0;
                }
                a
            })
            .collect();

        let vectors: Vec<Array1<f64>> = (0..batch_size)
            .map(|_| Array1::random_using(matrix_size, dist, &mut rng))
            .collect();

        let total_ops = (batch_size * matrix_size * matrix_size) as u64;
        group.throughput(Throughput::Elements(total_ops));

        // Sequential batch processing
        group.bench_with_input(
            BenchmarkId::new("sequential", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let results: Vec<_> = matrices
                        .iter()
                        .zip(vectors.iter())
                        .map(|(a, b)| solve(&a.view(), &b.view(), None))
                        .collect();
                    black_box(results)
                })
            },
        );

        // Parallel batch processing
        if batch_size >= 10 {
            group.bench_with_input(
                BenchmarkId::new("parallel", batch_size),
                &batch_size,
                |b, _| {
                    b.iter(|| {
                        let results: Vec<_> = matrices
                            .par_iter()
                            .zip(vectors.par_iter())
                            .map(|(a, b)| solve(&a.view(), &b.view(), None))
                            .collect();
                        black_box(results)
                    })
                },
            );
        }
    }

    group.finish();
}

// =============================================================================
// Dimension Scaling
// =============================================================================

/// Benchmark scaling with different dimensionalities
fn bench_dimension_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability/dimension_scaling");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    // 1D operations
    let sizes_1d = [1_000, 10_000, 100_000, 1_000_000];
    for &size in &sizes_1d {
        let data = Array1::<f64>::zeros(size);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("1d_sum", size), &size, |b, _| {
            b.iter(|| {
                let sum = data.sum();
                black_box(sum)
            })
        });
    }

    // 2D operations
    let sizes_2d = [32, 64, 128, 256, 512];
    for &size in &sizes_2d {
        let data = Array2::<f64>::zeros((size, size));

        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(BenchmarkId::new("2d_sum", size), &size, |b, _| {
            b.iter(|| {
                let sum = data.sum();
                black_box(sum)
            })
        });
    }

    group.finish();
}

// =============================================================================
// Chunk Size Optimization
// =============================================================================

/// Benchmark optimal chunk size for parallel operations
fn bench_chunk_size_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability/chunk_size");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    let data_size = 10_000_000;
    let data: Vec<f64> = (0..data_size).map(|i| i as f64).collect();

    group.throughput(Throughput::Elements(data_size as u64));

    // Different chunk sizes
    let chunk_sizes = [100, 1000, 10_000, 100_000, 1_000_000];

    for &chunk_size in &chunk_sizes {
        let label = format!("chunk_{}", chunk_size);
        group.bench_with_input(
            BenchmarkId::new("parallel_chunks", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let sum: f64 = data
                        .par_chunks(chunk_size)
                        .map(|chunk| chunk.iter().sum::<f64>())
                        .sum();
                    black_box(sum)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    thread_scaling_benchmarks,
    bench_parallel_reduction_scaling,
    bench_parallel_map_scaling,
);

criterion_group!(
    data_size_benchmarks,
    bench_linear_complexity_scaling,
    bench_n_log_n_complexity_scaling,
    bench_quadratic_complexity_scaling,
    bench_cubic_complexity_scaling,
);

criterion_group!(
    efficiency_benchmarks,
    bench_parallel_efficiency_matmul,
    bench_parallel_efficiency_fft,
);

criterion_group!(batch_benchmarks, bench_batch_processing_scaling,);

criterion_group!(
    dimension_benchmarks,
    bench_dimension_scaling,
    bench_chunk_size_optimization,
);

criterion_main!(
    thread_scaling_benchmarks,
    data_size_benchmarks,
    efficiency_benchmarks,
    batch_benchmarks,
    dimension_benchmarks,
);
