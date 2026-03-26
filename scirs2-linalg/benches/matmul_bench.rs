//! Benchmarks for matrix multiply (matmul) operations.
//!
//! Covers multiple multiplication strategies across representative matrix sizes:
//! 64×64, 128×128 and 256×256. Each variant is placed in its own Criterion
//! benchmark group so results can be compared and tracked independently.

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array2;
use scirs2_linalg::{
    blas_accelerated::matmul as blas_matmul,
    optim::{block_matmul, strassen_matmul, tiled_matmul},
    simd_ops::{simd_matmul_f32, simd_matmul_f64},
};

/// Build a deterministic, non-trivial f64 matrix of size n×n.
fn make_f64(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, n), |(i, j)| {
        if i == j {
            (i + 1) as f64
        } else {
            0.01 * ((i * n + j) as f64 * 0.01).sin()
        }
    })
}

/// Same as `make_f64` but in f32.
fn make_f32(n: usize) -> Array2<f32> {
    make_f64(n).mapv(|x| x as f32)
}

/// Benchmark ndarray's native `.dot()` as the baseline reference.
fn bench_ndarray_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul/ndarray_dot");
    group.measurement_time(Duration::from_secs(5));

    for &size in &[64usize, 128, 256] {
        let a = make_f64(size);
        let b = make_f64(size);

        group.throughput(Throughput::Elements((size * size * size) as u64));
        group.bench_with_input(BenchmarkId::new("f64", size), &size, |bench, _| {
            bench.iter(|| black_box(&a).dot(black_box(&b)))
        });
    }

    group.finish();
}

/// Benchmark the blas_accelerated matmul wrapper.
fn bench_blas_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul/blas_accelerated");
    group.measurement_time(Duration::from_secs(5));

    for &size in &[64usize, 128, 256] {
        let a = make_f64(size);
        let b = make_f64(size);

        group.throughput(Throughput::Elements((size * size * size) as u64));
        group.bench_with_input(BenchmarkId::new("f64", size), &size, |bench, _| {
            bench.iter(|| blas_matmul(black_box(&a.view()), black_box(&b.view())))
        });
    }

    group.finish();
}

/// Benchmark SIMD-accelerated matmul for both f32 and f64.
fn bench_simd_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul/simd");
    group.measurement_time(Duration::from_secs(5));

    for &size in &[64usize, 128, 256] {
        let af64 = make_f64(size);
        let bf64 = make_f64(size);
        let af32 = make_f32(size);
        let bf32 = make_f32(size);

        group.throughput(Throughput::Elements((size * size * size) as u64));

        group.bench_with_input(BenchmarkId::new("f64", size), &size, |bench, _| {
            bench.iter(|| simd_matmul_f64(black_box(&af64.view()), black_box(&bf64.view())))
        });

        group.bench_with_input(BenchmarkId::new("f32", size), &size, |bench, _| {
            bench.iter(|| simd_matmul_f32(black_box(&af32.view()), black_box(&bf32.view())))
        });
    }

    group.finish();
}

/// Benchmark tiled (cache-oblivious) matmul.
fn bench_tiled_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul/tiled");
    group.measurement_time(Duration::from_secs(5));

    for &size in &[64usize, 128, 256] {
        let a = make_f64(size);
        let b = make_f64(size);

        group.throughput(Throughput::Elements((size * size * size) as u64));
        group.bench_with_input(BenchmarkId::new("f64", size), &size, |bench, _| {
            bench.iter(|| tiled_matmul(black_box(&a.view()), black_box(&b.view()), None))
        });
    }

    group.finish();
}

/// Benchmark block matmul (explicit block size).
fn bench_block_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul/block");
    group.measurement_time(Duration::from_secs(5));

    for &size in &[64usize, 128, 256] {
        let a = make_f64(size);
        let b = make_f64(size);

        group.throughput(Throughput::Elements((size * size * size) as u64));
        group.bench_with_input(BenchmarkId::new("f64", size), &size, |bench, _| {
            bench.iter(|| block_matmul(black_box(&a.view()), black_box(&b.view()), None))
        });
    }

    group.finish();
}

/// Benchmark Strassen's algorithm (only meaningful for >=128 because the
/// default cutoff is 128; smaller sizes fall back to standard multiply).
fn bench_strassen_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul/strassen");
    group.measurement_time(Duration::from_secs(5));

    for &size in &[128usize, 256] {
        let a = make_f64(size);
        let b = make_f64(size);

        group.throughput(Throughput::Elements((size * size * size) as u64));
        group.bench_with_input(BenchmarkId::new("f64", size), &size, |bench, _| {
            bench.iter(|| strassen_matmul(black_box(&a.view()), black_box(&b.view()), None))
        });
    }

    group.finish();
}

criterion_group!(
    matmul_benches,
    bench_ndarray_dot,
    bench_blas_matmul,
    bench_simd_matmul,
    bench_tiled_matmul,
    bench_block_matmul,
    bench_strassen_matmul,
);
criterion_main!(matmul_benches);
