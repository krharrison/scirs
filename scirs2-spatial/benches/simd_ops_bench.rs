//! Benchmarks for SIMD-accelerated spatial operations
//!
//! Compares SIMD-accelerated implementations against scalar implementations
//! for various spatial operations.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_spatial::distance;
use scirs2_spatial::simd_ops::*;

fn generate_random_points(n_points: usize, n_dims: usize) -> Array2<f64> {
    use scirs2_core::random::{rng, Rng};
    let mut rng = rng();
    Array2::from_shape_fn((n_points, n_dims), |_| rng.random::<f64>())
}

fn generate_random_point(n_dims: usize) -> Array1<f64> {
    use scirs2_core::random::{rng, Rng};
    let mut rng = rng();
    Array1::from_shape_fn(n_dims, |_| rng.random::<f64>())
}

// ============================================================================
// Distance Computation Benchmarks
// ============================================================================

fn bench_euclidean_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("euclidean_distance");

    for dims in [2, 3, 10, 50, 100, 500, 1000].iter() {
        let a = generate_random_point(*dims);
        let b = generate_random_point(*dims);

        group.throughput(Throughput::Elements(*dims as u64));

        // Scalar implementation
        group.bench_with_input(BenchmarkId::new("scalar", dims), dims, |bench, _| {
            bench.iter(|| {
                let a_slice = a.as_slice().expect("Failed to get slice");
                let b_slice = b.as_slice().expect("Failed to get slice");
                std::hint::black_box(distance::euclidean(a_slice, b_slice))
            });
        });

        // SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", dims), dims, |bench, _| {
            bench.iter(|| {
                std::hint::black_box(simd_euclidean_distance(&a.view(), &b.view()).expect("Failed"))
            });
        });
    }

    group.finish();
}

fn bench_manhattan_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("manhattan_distance");

    for dims in [2, 3, 10, 50, 100, 500].iter() {
        let a = generate_random_point(*dims);
        let b = generate_random_point(*dims);

        group.throughput(Throughput::Elements(*dims as u64));

        // Scalar implementation
        group.bench_with_input(BenchmarkId::new("scalar", dims), dims, |bench, _| {
            bench.iter(|| {
                let a_slice = a.as_slice().expect("Failed to get slice");
                let b_slice = b.as_slice().expect("Failed to get slice");
                std::hint::black_box(distance::manhattan(a_slice, b_slice))
            });
        });

        // SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", dims), dims, |bench, _| {
            bench.iter(|| {
                std::hint::black_box(simd_manhattan_distance(&a.view(), &b.view()).expect("Failed"))
            });
        });
    }

    group.finish();
}

fn bench_chebyshev_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("chebyshev_distance");

    for dims in [2, 3, 10, 50, 100, 500].iter() {
        let a = generate_random_point(*dims);
        let b = generate_random_point(*dims);

        group.throughput(Throughput::Elements(*dims as u64));

        // Scalar implementation
        group.bench_with_input(BenchmarkId::new("scalar", dims), dims, |bench, _| {
            bench.iter(|| {
                let a_slice = a.as_slice().expect("Failed to get slice");
                let b_slice = b.as_slice().expect("Failed to get slice");
                std::hint::black_box(distance::chebyshev(a_slice, b_slice))
            });
        });

        // SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", dims), dims, |bench, _| {
            bench.iter(|| {
                std::hint::black_box(simd_chebyshev_distance(&a.view(), &b.view()).expect("Failed"))
            });
        });
    }

    group.finish();
}

fn bench_minkowski_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("minkowski_distance_p3");

    for dims in [2, 3, 10, 50, 100, 500].iter() {
        let a = generate_random_point(*dims);
        let b = generate_random_point(*dims);
        let p = 3.0;

        group.throughput(Throughput::Elements(*dims as u64));

        // Scalar implementation
        group.bench_with_input(BenchmarkId::new("scalar", dims), dims, |bench, _| {
            bench.iter(|| {
                let a_slice = a.as_slice().expect("Failed to get slice");
                let b_slice = b.as_slice().expect("Failed to get slice");
                std::hint::black_box(distance::minkowski(a_slice, b_slice, p))
            });
        });

        // SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", dims), dims, |bench, _| {
            bench.iter(|| {
                std::hint::black_box(
                    simd_minkowski_distance(&a.view(), &b.view(), p).expect("Failed"),
                )
            });
        });
    }

    group.finish();
}

fn bench_cosine_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_distance");

    for dims in [2, 3, 10, 50, 100, 500].iter() {
        let a = generate_random_point(*dims);
        let b = generate_random_point(*dims);

        group.throughput(Throughput::Elements(*dims as u64));

        // Scalar implementation
        group.bench_with_input(BenchmarkId::new("scalar", dims), dims, |bench, _| {
            bench.iter(|| {
                let a_slice = a.as_slice().expect("Failed to get slice");
                let b_slice = b.as_slice().expect("Failed to get slice");
                std::hint::black_box(distance::cosine(a_slice, b_slice))
            });
        });

        // SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", dims), dims, |bench, _| {
            bench.iter(|| {
                std::hint::black_box(simd_cosine_distance(&a.view(), &b.view()).expect("Failed"))
            });
        });
    }

    group.finish();
}

// ============================================================================
// Batch Operations Benchmarks
// ============================================================================

fn bench_batch_distances(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_distances");

    for n_points in [10, 50, 100, 500, 1000].iter() {
        let dims = 3;
        let points1 = generate_random_points(*n_points, dims);
        let points2 = generate_random_points(*n_points, dims);

        group.throughput(Throughput::Elements(*n_points as u64));

        // Scalar implementation (iterate)
        group.bench_with_input(
            BenchmarkId::new("scalar", n_points),
            n_points,
            |bench, _| {
                bench.iter(|| {
                    let mut distances = Vec::with_capacity(*n_points);
                    for i in 0..*n_points {
                        let row1 = points1.row(i);
                        let row2 = points2.row(i);
                        let p1 = row1.as_slice().expect("Failed");
                        let p2 = row2.as_slice().expect("Failed");
                        distances.push(distance::euclidean(p1, p2));
                    }
                    std::hint::black_box(distances)
                });
            },
        );

        // SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", n_points), n_points, |bench, _| {
            bench.iter(|| {
                std::hint::black_box(
                    simd_batch_distances(&points1.view(), &points2.view()).expect("Failed"),
                )
            });
        });
    }

    group.finish();
}

// ============================================================================
// k-NN Search Benchmarks
// ============================================================================

fn bench_knn_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_search");

    for n_points in [100, 500, 1000, 5000].iter() {
        let dims = 3;
        let k = 10;
        let data_points = generate_random_points(*n_points, dims);
        let query = generate_random_point(dims);

        group.throughput(Throughput::Elements(*n_points as u64));

        // SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", n_points), n_points, |bench, _| {
            bench.iter(|| {
                std::hint::black_box(
                    simd_knn_search(&query.view(), &data_points.view(), k).expect("Failed"),
                )
            });
        });
    }

    group.finish();
}

// ============================================================================
// Radius Search Benchmarks
// ============================================================================

fn bench_radius_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("radius_search");

    for n_points in [100, 500, 1000, 5000].iter() {
        let dims = 3;
        let radius = 0.5;
        let data_points = generate_random_points(*n_points, dims);
        let query = generate_random_point(dims);

        group.throughput(Throughput::Elements(*n_points as u64));

        // SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", n_points), n_points, |bench, _| {
            bench.iter(|| {
                std::hint::black_box(
                    simd_radius_search(&query.view(), &data_points.view(), radius).expect("Failed"),
                )
            });
        });
    }

    group.finish();
}

// ============================================================================
// KD-Tree Helper Benchmarks
// ============================================================================

fn bench_point_to_box_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("point_to_box_distance");

    for dims in [2, 3, 10, 50, 100].iter() {
        let point = generate_random_point(*dims);
        let box_min = generate_random_point(*dims);
        let box_max = &box_min + 1.0;

        group.throughput(Throughput::Elements(*dims as u64));

        group.bench_with_input(BenchmarkId::new("simd", dims), dims, |bench, _| {
            bench.iter(|| {
                std::hint::black_box(
                    simd_point_to_box_min_distance_squared(
                        &point.view(),
                        &box_min.view(),
                        &box_max.view(),
                    )
                    .expect("Failed"),
                )
            });
        });
    }

    group.finish();
}

fn bench_box_intersection(c: &mut Criterion) {
    let mut group = c.benchmark_group("box_intersection");

    for dims in [2, 3, 10, 50, 100].iter() {
        let box1_min = generate_random_point(*dims);
        let box1_max = &box1_min + 1.0;
        let box2_min = &box1_min + 0.5;
        let box2_max = &box2_min + 1.0;

        group.throughput(Throughput::Elements(*dims as u64));

        group.bench_with_input(BenchmarkId::new("simd", dims), dims, |bench, _| {
            bench.iter(|| {
                std::hint::black_box(
                    simd_box_box_intersection(
                        &box1_min.view(),
                        &box1_max.view(),
                        &box2_min.view(),
                        &box2_max.view(),
                    )
                    .expect("Failed"),
                )
            });
        });
    }

    group.finish();
}

// ============================================================================
// Distance Matrix Benchmarks
// ============================================================================

fn bench_pairwise_distance_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("pairwise_distance_matrix");
    group.sample_size(10); // Reduce samples for expensive operations

    for n_points in [10, 50, 100, 200].iter() {
        let dims = 3;
        let points = generate_random_points(*n_points, dims);

        group.throughput(Throughput::Elements((n_points * n_points) as u64));

        // SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", n_points), n_points, |bench, _| {
            bench.iter(|| {
                std::hint::black_box(simd_pairwise_distance_matrix(&points.view()).expect("Failed"))
            });
        });
    }

    group.finish();
}

criterion_group!(
    distance_benches,
    bench_euclidean_distance,
    bench_manhattan_distance,
    bench_chebyshev_distance,
    bench_minkowski_distance,
    bench_cosine_distance,
);

criterion_group!(batch_benches, bench_batch_distances,);

criterion_group!(search_benches, bench_knn_search, bench_radius_search,);

criterion_group!(
    kdtree_benches,
    bench_point_to_box_distance,
    bench_box_intersection,
);

criterion_group!(matrix_benches, bench_pairwise_distance_matrix,);

criterion_main!(
    distance_benches,
    batch_benches,
    search_benches,
    kdtree_benches,
    matrix_benches,
);
