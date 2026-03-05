//! Cache Optimization Benchmark
//!
//! Measures the performance impact of cache-efficient batch sizes across
//! different data sizes (L1-fit, L2-fit, L3-fit).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array1;
use scirs2_core::simd_ops::{
    SimdUnifiedOps, SIMD_BATCH_L1_F64, SIMD_BATCH_L2_F64, SIMD_BATCH_L3_F64,
};
use std::hint::black_box;

fn benchmark_simd_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_add_cache_sizes");

    let sizes = vec![
        ("L1-fit", SIMD_BATCH_L1_F64),
        ("L2-fit", SIMD_BATCH_L2_F64),
        ("L3-fit", SIMD_BATCH_L3_F64),
        ("Beyond-L3", 2_000_000),
    ];

    for (label, size) in sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(label), &size, |b, &size| {
            let a = Array1::linspace(0.0, 1.0, size);
            let b_data = Array1::from_elem(size, 1.5);
            b.iter(|| black_box(f64::simd_add(&a.view(), &b_data.view())));
        });
    }

    group.finish();
}

fn benchmark_simd_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_mul_cache_sizes");

    let sizes = vec![
        ("L1-fit", SIMD_BATCH_L1_F64),
        ("L2-fit", SIMD_BATCH_L2_F64),
        ("L3-fit", SIMD_BATCH_L3_F64),
    ];

    for (label, size) in sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(label), &size, |b, &size| {
            let a = Array1::linspace(0.0, 1.0, size);
            let b_data = Array1::from_elem(size, 2.0);
            b.iter(|| black_box(f64::simd_mul(&a.view(), &b_data.view())));
        });
    }

    group.finish();
}

fn benchmark_simd_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_sum_cache_sizes");

    let sizes = vec![
        ("L1-fit", SIMD_BATCH_L1_F64),
        ("L2-fit", SIMD_BATCH_L2_F64),
        ("L3-fit", SIMD_BATCH_L3_F64),
    ];

    for (label, size) in sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(label), &size, |b, &size| {
            let a = Array1::linspace(0.0, 1.0, size);
            b.iter(|| black_box(f64::simd_sum(&a.view())));
        });
    }

    group.finish();
}

fn benchmark_simd_exp(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_exp_cache_sizes");

    let sizes = vec![("L1-fit", SIMD_BATCH_L1_F64), ("L2-fit", SIMD_BATCH_L2_F64)];

    for (label, size) in sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(label), &size, |b, &size| {
            let a = Array1::linspace(-2.0, 2.0, size);
            b.iter(|| black_box(f64::simd_exp(&a.view())));
        });
    }

    group.finish();
}

fn benchmark_simd_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_dot_cache_sizes");

    let sizes = vec![
        ("L1-fit", SIMD_BATCH_L1_F64),
        ("L2-fit", SIMD_BATCH_L2_F64),
        ("L3-fit", SIMD_BATCH_L3_F64),
    ];

    for (label, size) in sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(label), &size, |b, &size| {
            let a = Array1::linspace(0.0, 1.0, size);
            let b_data = Array1::linspace(1.0, 0.0, size);
            b.iter(|| black_box(f64::simd_dot(&a.view(), &b_data.view())));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_simd_add,
    benchmark_simd_mul,
    benchmark_simd_sum,
    benchmark_simd_exp,
    benchmark_simd_dot
);
criterion_main!(benches);
