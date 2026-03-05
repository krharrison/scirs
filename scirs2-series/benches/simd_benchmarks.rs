//! SIMD Performance Benchmarks for Time Series Operations
//!
//! Benchmarks comparing SIMD vs scalar implementations to demonstrate speedup.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array1;

#[cfg(feature = "simd")]
use scirs2_series::simd_ops::*;

/// Helper function to generate test data
fn generate_test_data(n: usize) -> Array1<f64> {
    Array1::linspace(0.0, n as f64, n)
}

/// Helper function to generate seasonal data
fn generate_seasonal_data(n: usize, period: usize) -> Array1<f64> {
    let mut data = Array1::zeros(n);
    for i in 0..n {
        let t = i as f64;
        data[i] = 10.0 * (t / period as f64).sin() + 0.5 * t + (t * 0.1).cos();
    }
    data
}

/// Scalar reference implementation of differencing
fn scalar_difference(data: &Array1<f64>, order: usize) -> Array1<f64> {
    let mut result = data.clone();
    for _ in 0..order {
        let mut diff = Array1::zeros(result.len() - 1);
        for i in 1..result.len() {
            diff[i - 1] = result[i] - result[i - 1];
        }
        result = diff;
    }
    result
}

/// Scalar reference implementation of autocorrelation
fn scalar_autocorrelation(data: &Array1<f64>, max_lag: usize) -> Array1<f64> {
    let n = data.len();
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let c0: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum();

    let mut acf = Array1::zeros(max_lag + 1);
    for lag in 0..=max_lag {
        let mut sum = 0.0;
        for i in 0..(n - lag) {
            sum += (data[i] - mean) * (data[i + lag] - mean);
        }
        acf[lag] = sum / c0;
    }
    acf
}

/// Scalar reference implementation of seasonal differencing
fn scalar_seasonal_difference(data: &Array1<f64>, period: usize, order: usize) -> Array1<f64> {
    let mut result = data.clone();
    for _ in 0..order {
        let n = result.len();
        if n <= period {
            return Array1::zeros(0);
        }
        let mut diff = Array1::zeros(n - period);
        for i in period..n {
            diff[i - period] = result[i] - result[i - period];
        }
        result = diff;
    }
    result
}

fn bench_differencing(c: &mut Criterion) {
    let mut group = c.benchmark_group("differencing");

    for size in [100, 500, 1000, 5000, 10000].iter() {
        let data = generate_test_data(*size);

        group.throughput(Throughput::Elements(*size as u64));

        // Scalar benchmark
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |b, _| {
            b.iter(|| {
                black_box(scalar_difference(black_box(&data), 1));
            });
        });

        // SIMD benchmark
        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, _| {
            b.iter(|| {
                black_box(simd_difference_f64(black_box(&data.view()), 1).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_seasonal_differencing(c: &mut Criterion) {
    let mut group = c.benchmark_group("seasonal_differencing");

    let period = 12; // Monthly seasonality
    for size in [120, 240, 600, 1200, 2400].iter() {
        let data = generate_seasonal_data(*size, period);

        group.throughput(Throughput::Elements(*size as u64));

        // Scalar benchmark
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |b, _| {
            b.iter(|| {
                black_box(scalar_seasonal_difference(black_box(&data), period, 1));
            });
        });

        // SIMD benchmark
        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    simd_seasonal_difference_f64(black_box(&data.view()), period, 1).unwrap(),
                );
            });
        });
    }

    group.finish();
}

fn bench_autocorrelation(c: &mut Criterion) {
    let mut group = c.benchmark_group("autocorrelation");

    for size in [100, 500, 1000, 5000].iter() {
        let data = generate_seasonal_data(*size, 12);
        let max_lag = (*size / 10).min(50);

        group.throughput(Throughput::Elements((*size * max_lag) as u64));

        // Scalar benchmark
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |b, _| {
            b.iter(|| {
                black_box(scalar_autocorrelation(black_box(&data), max_lag));
            });
        });

        // SIMD benchmark
        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    simd_autocorrelation_f64(black_box(&data.view()), Some(max_lag)).unwrap(),
                );
            });
        });
    }

    group.finish();
}

fn bench_moving_mean(c: &mut Criterion) {
    let mut group = c.benchmark_group("moving_mean");

    let window = 7;
    for size in [100, 500, 1000, 5000, 10000].iter() {
        let data = generate_seasonal_data(*size, 12);

        group.throughput(Throughput::Elements(*size as u64));

        // SIMD benchmark
        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, _| {
            b.iter(|| {
                black_box(simd_moving_mean_f64(black_box(&data.view()), window).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_convolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("convolution");

    let kernel = Array1::from(vec![1.0, 0.5, 0.25, 0.125]);

    for size in [100, 500, 1000, 5000].iter() {
        let signal = generate_test_data(*size);

        group.throughput(Throughput::Elements(*size as u64));

        // SIMD benchmark
        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    simd_convolve_f64(black_box(&signal.view()), black_box(&kernel.view()))
                        .unwrap(),
                );
            });
        });
    }

    group.finish();
}

fn bench_seasonal_means(c: &mut Criterion) {
    let mut group = c.benchmark_group("seasonal_means");

    let period = 12;
    for size in [120, 240, 600, 1200, 2400].iter() {
        let data = generate_seasonal_data(*size, period);

        group.throughput(Throughput::Elements(*size as u64));

        // SIMD benchmark
        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, _| {
            b.iter(|| {
                black_box(simd_seasonal_means_f64(black_box(&data.view()), period).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_ema(c: &mut Criterion) {
    let mut group = c.benchmark_group("exponential_moving_average");

    let alpha = 0.3;
    for size in [100, 500, 1000, 5000, 10000].iter() {
        let data = generate_seasonal_data(*size, 12);

        group.throughput(Throughput::Elements(*size as u64));

        // SIMD benchmark
        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    simd_exponential_moving_average_f64(black_box(&data.view()), alpha).unwrap(),
                );
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_differencing,
    bench_seasonal_differencing,
    bench_autocorrelation,
    bench_moving_mean,
    bench_convolution,
    bench_seasonal_means,
    bench_ema
);

criterion_main!(benches);
