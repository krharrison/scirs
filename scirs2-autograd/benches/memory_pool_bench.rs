use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_autograd::memory_pool::{
    global_buffer_pool_f64, global_gradient_pool_f64, global_pool, ArenaAllocator, BufferPool,
    GradientPool, PooledArray, TensorPool,
};
use scirs2_autograd::ndarray_ext::NdArray;
use scirs2_core::ndarray;

/// Benchmark direct allocation vs. pooled allocation.
fn bench_allocation_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_strategies");

    let shapes = vec![vec![64, 64], vec![128, 128], vec![256, 256]];

    for shape in &shapes {
        let shape_str = format!("{}x{}", shape[0], shape[1]);

        // Direct allocation (baseline).
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("direct_alloc", &shape_str),
            shape,
            |b, shape| {
                b.iter(|| {
                    let _arr: NdArray<f64> = ndarray::Array::zeros(ndarray::IxDyn(shape));
                    black_box(_arr);
                });
            },
        );

        // TensorPool allocation.
        group.bench_with_input(
            BenchmarkId::new("tensor_pool", &shape_str),
            shape,
            |b, shape| {
                let pool = global_pool();
                b.iter(|| {
                    let _buf: PooledArray<f64> = pool.acquire(shape);
                    black_box(_buf);
                });
            },
        );

        // BufferPool allocation.
        group.bench_with_input(
            BenchmarkId::new("buffer_pool", &shape_str),
            shape,
            |b, shape| {
                let pool = global_buffer_pool_f64();
                let num_elements = shape.iter().product();
                b.iter(|| {
                    let _buf = pool.acquire(num_elements);
                    black_box(_buf);
                });
            },
        );

        // GradientPool allocation.
        group.bench_with_input(
            BenchmarkId::new("gradient_pool", &shape_str),
            shape,
            |b, shape| {
                let pool = global_gradient_pool_f64();
                b.iter(|| {
                    let _grad = pool.acquire_gradient(shape);
                    black_box(_grad);
                });
            },
        );

        // ArenaAllocator.
        group.bench_with_input(
            BenchmarkId::new("arena_alloc", &shape_str),
            shape,
            |b, shape| {
                b.iter_batched(
                    || ArenaAllocator::<f64>::new(),
                    |arena| {
                        let _t = arena.alloc(shape);
                        black_box(_t);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

/// Benchmark reuse performance (warm cache).
fn bench_reuse_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("reuse_performance");

    let shape = vec![128, 128];
    let num_iterations = 100;

    group.throughput(Throughput::Elements(num_iterations));

    // TensorPool with warm cache.
    group.bench_function("tensor_pool_warm", |b| {
        let pool = TensorPool::new();
        // Pre-warm the pool.
        for _ in 0..10 {
            let buf: PooledArray<f64> = pool.acquire(&shape);
            drop(buf);
        }

        b.iter(|| {
            for _ in 0..num_iterations {
                let buf: PooledArray<f64> = pool.acquire(&shape);
                black_box(&buf);
                drop(buf);
            }
        });
    });

    // GradientPool with warm cache.
    group.bench_function("gradient_pool_warm", |b| {
        let pool = GradientPool::<f64>::new();
        // Pre-warm.
        for _ in 0..10 {
            let grad = pool.acquire_gradient(&shape);
            pool.release_gradient(grad);
        }

        b.iter(|| {
            for _ in 0..num_iterations {
                let grad = pool.acquire_gradient(&shape);
                black_box(&grad);
                pool.release_gradient(grad);
            }
        });
    });

    // Direct allocation (cold).
    group.bench_function("direct_alloc_cold", |b| {
        b.iter(|| {
            for _ in 0..num_iterations {
                let arr: NdArray<f64> = ndarray::Array::zeros(ndarray::IxDyn(&shape));
                black_box(&arr);
            }
        });
    });

    group.finish();
}

/// Benchmark gradient accumulation pattern (typical backward pass).
fn bench_gradient_accumulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_accumulation");

    let shapes = vec![vec![32, 64], vec![64, 128], vec![128, 256]];

    for shape in &shapes {
        let shape_str = format!("{}x{}", shape[0], shape[1]);

        // Without pool.
        group.bench_with_input(
            BenchmarkId::new("no_pool", &shape_str),
            shape,
            |b, shape| {
                b.iter(|| {
                    let mut grads = Vec::with_capacity(10);
                    for _ in 0..10 {
                        let grad: NdArray<f64> = ndarray::Array::zeros(ndarray::IxDyn(shape));
                        grads.push(grad);
                    }
                    black_box(grads);
                });
            },
        );

        // With GradientPool.
        group.bench_with_input(
            BenchmarkId::new("gradient_pool", &shape_str),
            shape,
            |b, shape| {
                let pool = GradientPool::<f64>::new();
                b.iter(|| {
                    let mut grads = Vec::with_capacity(10);
                    for _ in 0..10 {
                        let grad = pool.acquire_gradient(shape);
                        grads.push(grad);
                    }
                    // Release all at once.
                    for grad in grads {
                        pool.release_gradient(grad);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory pressure under high contention.
fn bench_memory_pressure(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pressure");
    group.sample_size(10); // Reduce sample size for this heavy test.

    let shape = vec![256, 256];
    let n_allocs = 1000;

    // Direct allocation.
    group.bench_function("direct_alloc_pressure", |b| {
        b.iter(|| {
            let mut arrays = Vec::with_capacity(n_allocs);
            for _ in 0..n_allocs {
                let arr: NdArray<f64> = ndarray::Array::zeros(ndarray::IxDyn(&shape));
                arrays.push(arr);
            }
            black_box(arrays);
        });
    });

    // With TensorPool (limited capacity).
    group.bench_function("tensor_pool_pressure", |b| {
        let pool = TensorPool::with_max_per_bucket(50);
        b.iter(|| {
            let mut bufs = Vec::with_capacity(n_allocs);
            for _ in 0..n_allocs {
                let buf: PooledArray<f64> = pool.acquire(&shape);
                bufs.push(buf);
            }
            black_box(bufs);
        });
    });

    // Arena allocator (batch deallocation).
    group.bench_function("arena_pressure", |b| {
        b.iter(|| {
            let arena = ArenaAllocator::<f64>::new();
            for _ in 0..n_allocs {
                let _t = arena.alloc(&shape);
            }
            black_box(arena);
        });
    });

    group.finish();
}

/// Benchmark pool statistics overhead.
fn bench_stats_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("stats_overhead");

    let pool = TensorPool::new();
    for _ in 0..100 {
        let buf: PooledArray<f64> = pool.acquire(&[128, 128]);
        drop(buf);
    }

    group.bench_function("tensor_pool_stats", |b| {
        b.iter(|| {
            let stats = pool.stats();
            black_box(stats);
        });
    });

    let grad_pool = GradientPool::<f64>::new();
    for _ in 0..100 {
        let grad = grad_pool.acquire_gradient(&[128, 128]);
        grad_pool.release_gradient(grad);
    }

    group.bench_function("gradient_pool_stats", |b| {
        b.iter(|| {
            let stats = grad_pool.stats();
            black_box(stats);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_allocation_strategies,
    bench_reuse_performance,
    bench_gradient_accumulation,
    bench_memory_pressure,
    bench_stats_overhead,
);
criterion_main!(benches);
