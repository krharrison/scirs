//! Benchmarks for SIMD-accelerated tensor operations
//!
//! This benchmark suite compares SIMD vs scalar performance for:
//! - Element-wise operations (add, sub, mul, div)
//! - Gradient accumulation
//! - Broadcast operations
//! - Activation functions (relu, sigmoid, tanh)
//! - Reduction operations (sum, mean, max, min)
//! - Dot products

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_autograd as ag;
use scirs2_core::ndarray::{array, Array1};

// ============================================================================
// Element-wise Operations
// ============================================================================

fn bench_elementwise_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_add");

    for size in [64, 256, 1024, 4096, 16384].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // SIMD version
        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, &size| {
            b.iter(|| {
                ag::run::<f32, _, _>(|ctx| {
                    let a_vec: Vec<f32> = (0..size).map(|i| i as f32).collect();
                    let b_vec: Vec<f32> = (0..size).map(|i| (size - i) as f32).collect();

                    let a = ag::tensor_ops::convert_to_tensor(Array1::from_vec(a_vec), ctx);
                    let b = ag::tensor_ops::convert_to_tensor(Array1::from_vec(b_vec), ctx);

                    #[cfg(feature = "simd")]
                    let result = ag::tensor_ops::simd_ops::simd_elementwise_add(&a, &b);
                    #[cfg(not(feature = "simd"))]
                    let result = ag::tensor_ops::add(a, b);

                    black_box(result.eval(ctx).expect("Operation failed"));
                });
            });
        });

        // Scalar version (baseline)
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |b, &size| {
            b.iter(|| {
                ag::run::<f32, _, _>(|ctx| {
                    let a_vec: Vec<f32> = (0..size).map(|i| i as f32).collect();
                    let b_vec: Vec<f32> = (0..size).map(|i| (size - i) as f32).collect();

                    let a = ag::tensor_ops::convert_to_tensor(Array1::from_vec(a_vec), ctx);
                    let b = ag::tensor_ops::convert_to_tensor(Array1::from_vec(b_vec), ctx);

                    let result = ag::tensor_ops::add(a, b);
                    black_box(result.eval(ctx).expect("Operation failed"));
                });
            });
        });
    }

    group.finish();
}

fn bench_elementwise_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_mul");

    for size in [64, 256, 1024, 4096].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, &size| {
            b.iter(|| {
                ag::run::<f32, _, _>(|ctx| {
                    let a_vec: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
                    let b_vec: Vec<f32> = (0..size).map(|i| (i as f32) * 0.2).collect();

                    let a = ag::tensor_ops::convert_to_tensor(Array1::from_vec(a_vec), ctx);
                    let b = ag::tensor_ops::convert_to_tensor(Array1::from_vec(b_vec), ctx);

                    #[cfg(feature = "simd")]
                    let result = ag::tensor_ops::simd_ops::simd_elementwise_mul(&a, &b);
                    #[cfg(not(feature = "simd"))]
                    let result = ag::tensor_ops::mul(a, b);

                    black_box(result.eval(ctx).expect("Operation failed"));
                });
            });
        });

        group.bench_with_input(BenchmarkId::new("scalar", size), size, |b, &size| {
            b.iter(|| {
                ag::run::<f32, _, _>(|ctx| {
                    let a_vec: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
                    let b_vec: Vec<f32> = (0..size).map(|i| (i as f32) * 0.2).collect();

                    let a = ag::tensor_ops::convert_to_tensor(Array1::from_vec(a_vec), ctx);
                    let b = ag::tensor_ops::convert_to_tensor(Array1::from_vec(b_vec), ctx);

                    let result = ag::tensor_ops::mul(a, b);
                    black_box(result.eval(ctx).expect("Operation failed"));
                });
            });
        });
    }

    group.finish();
}

// ============================================================================
// Gradient Accumulation (Critical for backprop)
// ============================================================================

fn bench_gradient_accumulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_accumulation");

    for size in [256, 1024, 4096, 16384].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, &size| {
            b.iter(|| {
                ag::run::<f32, _, _>(|ctx| {
                    let acc_vec: Vec<f32> = (0..size).map(|i| i as f32).collect();
                    let grad_vec: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();

                    let acc = ag::tensor_ops::convert_to_tensor(Array1::from_vec(acc_vec), ctx);
                    let grad = ag::tensor_ops::convert_to_tensor(Array1::from_vec(grad_vec), ctx);

                    #[cfg(feature = "simd")]
                    let result = ag::tensor_ops::simd_ops::simd_gradient_accumulate(&acc, &grad);
                    #[cfg(not(feature = "simd"))]
                    let result = ag::tensor_ops::add(&acc, &grad);

                    black_box(result.eval(ctx).expect("Operation failed"));
                });
            });
        });
    }

    group.finish();
}

// ============================================================================
// Activation Functions
// ============================================================================

fn bench_relu(c: &mut Criterion) {
    let mut group = c.benchmark_group("relu");

    for size in [256, 1024, 4096, 16384].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, &size| {
            b.iter(|| {
                ag::run::<f32, _, _>(|ctx| {
                    let x_vec: Vec<f32> = (0..size)
                        .map(|i| (i as f32) - (size as f32) / 2.0)
                        .collect();
                    let x = ag::tensor_ops::convert_to_tensor(Array1::from_vec(x_vec), ctx);

                    #[cfg(feature = "simd")]
                    let result = ag::tensor_ops::simd_ops::simd_activation_relu(&x);
                    #[cfg(not(feature = "simd"))]
                    let result = ag::tensor_ops::relu(x);

                    black_box(result.eval(ctx).expect("Operation failed"));
                });
            });
        });

        group.bench_with_input(BenchmarkId::new("scalar", size), size, |b, &size| {
            b.iter(|| {
                ag::run::<f32, _, _>(|ctx| {
                    let x_vec: Vec<f32> = (0..size)
                        .map(|i| (i as f32) - (size as f32) / 2.0)
                        .collect();
                    let x = ag::tensor_ops::convert_to_tensor(Array1::from_vec(x_vec), ctx);
                    let result = ag::tensor_ops::relu(x);
                    black_box(result.eval(ctx).expect("Operation failed"));
                });
            });
        });
    }

    group.finish();
}

fn bench_sigmoid(c: &mut Criterion) {
    let mut group = c.benchmark_group("sigmoid");

    for size in [256, 1024, 4096].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, &size| {
            b.iter(|| {
                ag::run::<f32, _, _>(|ctx| {
                    let x_vec: Vec<f32> = (0..size)
                        .map(|i| ((i as f32) / (size as f32)) * 10.0 - 5.0)
                        .collect();
                    let x = ag::tensor_ops::convert_to_tensor(Array1::from_vec(x_vec), ctx);

                    #[cfg(feature = "simd")]
                    let result = ag::tensor_ops::simd_ops::simd_activation_sigmoid(&x);
                    #[cfg(not(feature = "simd"))]
                    let result = ag::tensor_ops::sigmoid(x);

                    black_box(result.eval(ctx).expect("Operation failed"));
                });
            });
        });
    }

    group.finish();
}

// ============================================================================
// Reduction Operations
// ============================================================================

fn bench_reduction_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_sum");

    for size in [256, 1024, 4096, 16384].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, &size| {
            b.iter(|| {
                ag::run::<f32, _, _>(|ctx| {
                    let x_vec: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
                    let x = ag::tensor_ops::convert_to_tensor(Array1::from_vec(x_vec), ctx);

                    #[cfg(feature = "simd")]
                    let result = ag::tensor_ops::simd_ops::simd_reduction_sum(&x);
                    #[cfg(not(feature = "simd"))]
                    let result = ag::tensor_ops::sum_all(x);

                    black_box(result.eval(ctx).expect("Operation failed"));
                });
            });
        });

        group.bench_with_input(BenchmarkId::new("scalar", size), size, |b, &size| {
            b.iter(|| {
                ag::run::<f32, _, _>(|ctx| {
                    let x_vec: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
                    let x = ag::tensor_ops::convert_to_tensor(Array1::from_vec(x_vec), ctx);
                    let result = ag::tensor_ops::sum_all(x);
                    black_box(result.eval(ctx).expect("Operation failed"));
                });
            });
        });
    }

    group.finish();
}

fn bench_reduction_mean(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_mean");

    for size in [256, 1024, 4096, 16384].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, &size| {
            b.iter(|| {
                ag::run::<f32, _, _>(|ctx| {
                    let x_vec: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
                    let x = ag::tensor_ops::convert_to_tensor(Array1::from_vec(x_vec), ctx);

                    #[cfg(feature = "simd")]
                    let result = ag::tensor_ops::simd_ops::simd_reduction_mean(&x);
                    #[cfg(not(feature = "simd"))]
                    let result = ag::tensor_ops::mean_all(x);

                    black_box(result.eval(ctx).expect("Operation failed"));
                });
            });
        });
    }

    group.finish();
}

fn bench_reduction_max(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_max");

    for size in [256, 1024, 4096, 16384].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, &size| {
            b.iter(|| {
                ag::run::<f32, _, _>(|ctx| {
                    let x_vec: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
                    let x = ag::tensor_ops::convert_to_tensor(Array1::from_vec(x_vec), ctx);

                    #[cfg(feature = "simd")]
                    let result = ag::tensor_ops::simd_ops::simd_reduction_max(&x);
                    #[cfg(not(feature = "simd"))]
                    let result = ag::tensor_ops::reduce_max(&x, &[], false);

                    black_box(result.eval(ctx).expect("Operation failed"));
                });
            });
        });
    }

    group.finish();
}

fn bench_reduction_min(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_min");

    for size in [256, 1024, 4096, 16384].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, &size| {
            b.iter(|| {
                ag::run::<f32, _, _>(|ctx| {
                    let x_vec: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
                    let x = ag::tensor_ops::convert_to_tensor(Array1::from_vec(x_vec), ctx);

                    #[cfg(feature = "simd")]
                    let result = ag::tensor_ops::simd_ops::simd_reduction_min(&x);
                    #[cfg(not(feature = "simd"))]
                    let result = ag::tensor_ops::reduce_min(&x, &[], false);

                    black_box(result.eval(ctx).expect("Operation failed"));
                });
            });
        });
    }

    group.finish();
}

// ============================================================================
// Dot Product
// ============================================================================

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

    for size in [256, 1024, 4096, 16384].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, &size| {
            b.iter(|| {
                ag::run::<f32, _, _>(|ctx| {
                    let a_vec: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
                    let b_vec: Vec<f32> = (0..size).map(|i| (i as f32) * 0.2).collect();

                    let a = ag::tensor_ops::convert_to_tensor(Array1::from_vec(a_vec), ctx);
                    let b = ag::tensor_ops::convert_to_tensor(Array1::from_vec(b_vec), ctx);

                    #[cfg(feature = "simd")]
                    let result = ag::tensor_ops::simd_ops::simd_dot_product(&a, &b);
                    #[cfg(not(feature = "simd"))]
                    let result = ag::tensor_ops::sum_all(ag::tensor_ops::mul(&a, &b));

                    black_box(result.eval(ctx).expect("Operation failed"));
                });
            });
        });
    }

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    benches,
    bench_elementwise_add,
    bench_elementwise_mul,
    bench_gradient_accumulation,
    bench_relu,
    bench_sigmoid,
    bench_reduction_sum,
    bench_reduction_mean,
    bench_reduction_max,
    bench_reduction_min,
    bench_dot_product,
);

criterion_main!(benches);
