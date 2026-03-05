//! Benchmarks for operation fusion
//!
//! Demonstrates the performance benefits of fusing multiple operations
//! into single kernels versus executing them separately.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_autograd::optimization::fusion::ops::{
    fused_affine, fused_elementwise_chain, fused_linear, fused_linear_gelu, fused_linear_relu,
    fused_linear_sigmoid, fused_linear_tanh, fused_mean, fused_softmax, fused_variance,
};
use scirs2_core::ndarray::{Array, IxDyn};

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn arr2d(rows: usize, cols: usize) -> Array<f64, IxDyn> {
    Array::from_shape_fn((rows, cols), |(i, j)| (i * cols + j) as f64 * 0.1).into_dyn()
}

fn arr1d(len: usize) -> Array<f64, IxDyn> {
    Array::from_shape_fn(len, |i| i as f64 * 0.1).into_dyn()
}

// Unfused baseline: matmul + bias + activation separately
fn unfused_linear_relu(
    x: &Array<f64, IxDyn>,
    w: &Array<f64, IxDyn>,
    bias: &Array<f64, IxDyn>,
) -> Array<f64, IxDyn> {
    // Step 1: matmul
    let batch = x.shape()[0];
    let in_features = x.shape()[1];
    let out_features = w.shape()[1];

    let mut matmul_result = Array::<f64, _>::zeros(vec![batch, out_features]);
    for i in 0..batch {
        for j in 0..out_features {
            let mut acc = 0.0;
            for k in 0..in_features {
                acc += x[[i, k]] * w[[k, j]];
            }
            matmul_result[[i, j]] = acc;
        }
    }

    // Step 2: add bias
    let bias_slice: Vec<f64> = bias.iter().copied().collect();
    for i in 0..batch {
        for j in 0..out_features {
            matmul_result[[i, j]] += bias_slice[j];
        }
    }

    // Step 3: relu
    matmul_result.mapv(|v| if v > 0.0 { v } else { 0.0 })
}

fn unfused_affine(
    x: &Array<f64, IxDyn>,
    scale: &Array<f64, IxDyn>,
    shift: &Array<f64, IxDyn>,
) -> Array<f64, IxDyn> {
    // Step 1: multiply
    let scaled = x * scale;
    // Step 2: add
    &scaled + shift
}

// ---------------------------------------------------------------------------
// Benchmark: Fused vs unfused linear + ReLU
// ---------------------------------------------------------------------------

fn bench_linear_relu(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_relu");

    for size in [32, 128, 512].iter() {
        let batch = *size;
        let in_features = *size;
        let out_features = *size;

        let x = arr2d(batch, in_features);
        let w = arr2d(in_features, out_features);
        let bias = arr1d(out_features);

        let bytes = (batch * in_features + in_features * out_features + out_features) * 8;
        group.throughput(Throughput::Bytes(bytes as u64));

        group.bench_with_input(
            BenchmarkId::new("fused", size),
            &(x.clone(), w.clone(), bias.clone()),
            |b, (x, w, bias)| {
                b.iter(|| {
                    let _ = black_box(fused_linear_relu(
                        black_box(x),
                        black_box(w),
                        black_box(bias),
                    ));
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("unfused", size),
            &(x.clone(), w.clone(), bias.clone()),
            |b, (x, w, bias)| {
                b.iter(|| {
                    let _ = black_box(unfused_linear_relu(
                        black_box(x),
                        black_box(w),
                        black_box(bias),
                    ));
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Different activation fusions
// ---------------------------------------------------------------------------

fn bench_activations(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_activations");

    let batch = 128;
    let in_features = 256;
    let out_features = 128;

    let x = arr2d(batch, in_features);
    let w = arr2d(in_features, out_features);
    let bias = arr1d(out_features);

    group.bench_function("relu", |b| {
        b.iter(|| {
            let _ = black_box(fused_linear_relu(
                black_box(&x),
                black_box(&w),
                black_box(&bias),
            ));
        })
    });

    group.bench_function("sigmoid", |b| {
        b.iter(|| {
            let _ = black_box(fused_linear_sigmoid(
                black_box(&x),
                black_box(&w),
                black_box(&bias),
            ));
        })
    });

    group.bench_function("tanh", |b| {
        b.iter(|| {
            let _ = black_box(fused_linear_tanh(
                black_box(&x),
                black_box(&w),
                black_box(&bias),
            ));
        })
    });

    group.bench_function("gelu", |b| {
        b.iter(|| {
            let _ = black_box(fused_linear_gelu(
                black_box(&x),
                black_box(&w),
                black_box(&bias),
            ));
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Fused vs unfused affine
// ---------------------------------------------------------------------------

fn bench_affine(c: &mut Criterion) {
    let mut group = c.benchmark_group("affine");

    for size in [1024, 4096, 16384].iter() {
        let len = *size;

        let x = arr1d(len);
        let scale = arr1d(len);
        let shift = arr1d(len);

        group.throughput(Throughput::Elements(len as u64));

        group.bench_with_input(
            BenchmarkId::new("fused", size),
            &(x.clone(), scale.clone(), shift.clone()),
            |b, (x, scale, shift)| {
                b.iter(|| {
                    let _ = black_box(fused_affine(
                        black_box(x),
                        black_box(scale),
                        black_box(shift),
                    ));
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("unfused", size),
            &(x.clone(), scale.clone(), shift.clone()),
            |b, (x, scale, shift)| {
                b.iter(|| {
                    let _ = black_box(unfused_affine(
                        black_box(x),
                        black_box(scale),
                        black_box(shift),
                    ));
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Element-wise operation chains
// ---------------------------------------------------------------------------

fn bench_elementwise_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_chain");

    let len = 10000;
    let x = arr1d(len);

    group.bench_function("single_relu", |b| {
        b.iter(|| {
            let _ = black_box(fused_elementwise_chain(black_box(&x), &["relu"]));
        })
    });

    group.bench_function("relu_neg", |b| {
        b.iter(|| {
            let _ = black_box(fused_elementwise_chain(black_box(&x), &["relu", "neg"]));
        })
    });

    group.bench_function("square_sqrt", |b| {
        b.iter(|| {
            let _ = black_box(fused_elementwise_chain(black_box(&x), &["square", "sqrt"]));
        })
    });

    group.bench_function("sigmoid_tanh", |b| {
        b.iter(|| {
            let _ = black_box(fused_elementwise_chain(black_box(&x), &["sigmoid", "tanh"]));
        })
    });

    group.bench_function("long_chain", |b| {
        b.iter(|| {
            let _ = black_box(fused_elementwise_chain(
                black_box(&x),
                &["abs", "sqrt", "square", "relu"],
            ));
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Reduction fusions
// ---------------------------------------------------------------------------

fn bench_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("reductions");

    let rows = 1000;
    let cols = 500;
    let x = arr2d(rows, cols);

    group.bench_function("fused_mean_axis0", |b| {
        b.iter(|| {
            let _ = black_box(fused_mean(black_box(&x), 0));
        })
    });

    group.bench_function("fused_mean_axis1", |b| {
        b.iter(|| {
            let _ = black_box(fused_mean(black_box(&x), 1));
        })
    });

    group.bench_function("fused_variance_axis0", |b| {
        b.iter(|| {
            let _ = black_box(fused_variance(black_box(&x), 0));
        })
    });

    group.bench_function("fused_variance_axis1", |b| {
        b.iter(|| {
            let _ = black_box(fused_variance(black_box(&x), 1));
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Softmax (most complex fusion)
// ---------------------------------------------------------------------------

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");

    for size in [32, 128, 512, 2048].iter() {
        let rows = 100;
        let cols = *size;

        let x = arr2d(rows, cols);

        group.throughput(Throughput::Elements((rows * cols) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &x, |b, x| {
            b.iter(|| {
                let _ = black_box(fused_softmax(black_box(x), 1));
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Memory traffic comparison
// ---------------------------------------------------------------------------

fn bench_memory_traffic(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_traffic");
    group.sample_size(50);

    // Large matrices to emphasize memory bandwidth effects
    let batch = 256;
    let in_features = 512;
    let out_features = 512;

    let x = arr2d(batch, in_features);
    let w = arr2d(in_features, out_features);
    let bias = arr1d(out_features);

    let total_elements = batch * in_features + in_features * out_features + out_features;
    group.throughput(Throughput::Elements(total_elements as u64));

    group.bench_function("fused_linear", |b| {
        b.iter(|| {
            let _ = black_box(fused_linear(black_box(&x), black_box(&w), black_box(&bias)));
        })
    });

    group.bench_function("fused_linear_relu", |b| {
        b.iter(|| {
            let _ = black_box(fused_linear_relu(
                black_box(&x),
                black_box(&w),
                black_box(&bias),
            ));
        })
    });

    group.bench_function("fused_linear_gelu", |b| {
        b.iter(|| {
            let _ = black_box(fused_linear_gelu(
                black_box(&x),
                black_box(&w),
                black_box(&bias),
            ));
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_linear_relu,
    bench_activations,
    bench_affine,
    bench_elementwise_chain,
    bench_reductions,
    bench_softmax,
    bench_memory_traffic,
);
criterion_main!(benches);
