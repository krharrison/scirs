//! SciRS2 v0.3.0 Linear Algebra Advanced Benchmark Suite
//!
//! Benchmarks for v0.3.0 linalg additions:
//! - Tensor decompositions: CP/PARAFAC, Tucker, HOSVD, Tensor Train
//! - Matrix sign function and related matrix functions
//! - Structured matrix operations: Toeplitz, Circulant, Cauchy
//! - Randomized algorithms: randomized SVD, CUR decomposition
//!
//! Performance Targets (v0.3.0):
//! - CP decomposition (10×10×10, rank-5): < 200 ms
//! - Tucker (20×20×20, ranks 5/5/5): < 500 ms
//! - Tensor Train SVD (10×10×10): < 100 ms
//! - Randomized SVD (1000×500, rank-50): < 500 ms

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use scirs2_linalg::tensor_decomp::{
    tensor_utils::Tensor3D,
    parafac::fit_als,
    tucker::tucker_als,
    tensor_train::tt_svd,
    hosvd::hosvd,
};
use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::random::{ChaCha8Rng, SeedableRng, Uniform, Distribution};
use std::hint::black_box;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Tensor data generators
// ---------------------------------------------------------------------------

fn make_tensor3d(dims: [usize; 3], seed: u64) -> Tensor3D {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let dist = Uniform::new(-1.0f64, 1.0f64).expect("uniform range");
    let total = dims[0] * dims[1] * dims[2];
    let data: Vec<f64> = (0..total).map(|_| dist.sample(&mut rng)).collect();
    Tensor3D::new(data, dims).expect("valid tensor")
}

fn make_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let dist = Uniform::new(-1.0f64, 1.0f64).expect("uniform range");
    let data: Vec<f64> = (0..rows * cols).map(|_| dist.sample(&mut rng)).collect();
    Array2::from_shape_vec((rows, cols), data).expect("valid matrix")
}

// ---------------------------------------------------------------------------
// CP / PARAFAC benchmarks
// ---------------------------------------------------------------------------

fn bench_cp_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_cp_parafac");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    // Tensor shapes
    let shapes: &[[usize; 3]] = &[
        [5,  5,  5],
        [8,  8,  8],
        [10, 10, 10],
        [15, 15, 10],
        [20, 15, 10],
    ];
    let ranks: &[usize] = &[3, 5];

    for shape in shapes {
        let tensor = make_tensor3d(*shape, 42);
        let total_elements = shape[0] * shape[1] * shape[2];

        for &rank in ranks {
            group.throughput(Throughput::Elements(total_elements as u64));

            let label = format!("{}x{}x{}_r{}", shape[0], shape[1], shape[2], rank);
            group.bench_with_input(
                BenchmarkId::new("cp_als", &label),
                &(),
                |bencher, _| {
                    bencher.iter(|| {
                        let result = fit_als(
                            black_box(&tensor),
                            black_box(rank),
                            200,  // max_iter
                            1e-6, // tol
                        );
                        black_box(result)
                    })
                },
            );
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Tucker decomposition benchmarks
// ---------------------------------------------------------------------------

fn bench_tucker_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_tucker");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let shapes_and_ranks: &[([usize; 3], [usize; 3])] = &[
        ([6,  6,  6],  [3, 3, 3]),
        ([10, 10, 10], [4, 4, 4]),
        ([15, 12, 10], [5, 4, 4]),
        ([20, 15, 12], [6, 5, 4]),
    ];

    for (shape, tucker_ranks) in shapes_and_ranks {
        let tensor = make_tensor3d(*shape, 77);
        let total_elements = shape[0] * shape[1] * shape[2];

        group.throughput(Throughput::Elements(total_elements as u64));

        let label = format!(
            "{}x{}x{}_r{}x{}x{}",
            shape[0], shape[1], shape[2],
            tucker_ranks[0], tucker_ranks[1], tucker_ranks[2]
        );
        group.bench_with_input(
            BenchmarkId::new("tucker_als", &label),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let result = tucker_als(
                        black_box(&tensor),
                        black_box(*tucker_ranks),
                        50,   // max_iter
                        1e-8, // tol
                    );
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// HOSVD benchmarks
// ---------------------------------------------------------------------------

fn bench_hosvd(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_hosvd");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let shapes: &[[usize; 3]] = &[
        [5,  5,  5],
        [10, 10, 10],
        [15, 12, 10],
        [20, 20, 15],
    ];

    for shape in shapes {
        let tensor = make_tensor3d(*shape, 55);
        let total = shape[0] * shape[1] * shape[2];

        group.throughput(Throughput::Elements(total as u64));

        let label = format!("{}x{}x{}", shape[0], shape[1], shape[2]);
        group.bench_with_input(
            BenchmarkId::new("hosvd", &label),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let result = hosvd(black_box(&tensor));
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Tensor Train (TT-SVD) benchmarks
// ---------------------------------------------------------------------------

fn bench_tensor_train_svd(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_train_svd");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let configs: &[(&str, &[usize], usize)] = &[
        ("3d_5x5x5",     &[5, 5, 5],     4),
        ("3d_10x10x10",  &[10, 10, 10],  8),
        ("3d_20x15x10",  &[20, 15, 10],  10),
        ("4d_5x5x5x5",   &[5, 5, 5, 5],  6),
    ];

    for &(label, shape, max_rank) in configs {
        let total: usize = shape.iter().product();
        let mut rng = ChaCha8Rng::seed_from_u64(33);
        let dist = Uniform::new(-1.0f64, 1.0f64).expect("uniform range");
        let data: Vec<f64> = (0..total).map(|_| dist.sample(&mut rng)).collect();

        group.throughput(Throughput::Elements(total as u64));

        group.bench_with_input(
            BenchmarkId::new("tt_svd", label),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let result = tt_svd(
                        black_box(&data),
                        black_box(shape),
                        black_box(max_rank),
                        black_box(1e-8),
                    );
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Randomized SVD benchmarks
// ---------------------------------------------------------------------------

fn bench_randomized_svd(c: &mut Criterion) {
    use scirs2_linalg::lowrank::randomized_svd;

    let mut group = c.benchmark_group("randomized_svd");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let matrix_configs: &[(usize, usize, usize)] = &[
        (200,  100,  10),
        (500,  200,  20),
        (1000, 500,  50),
        (2000, 1000, 100),
    ];

    for &(rows, cols, rank) in matrix_configs {
        let matrix = make_matrix(rows, cols, 42);

        group.throughput(Throughput::Elements((rows * cols) as u64));

        let label = format!("{rows}x{cols}_r{rank}");
        group.bench_with_input(
            BenchmarkId::new("randomized_svd", &label),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let result = randomized_svd(
                        black_box(&matrix.view()),
                        black_box(rank),
                        Some(5),  // n_oversamples
                        Some(2),  // n_power_iter
                        None,     // workers (use default)
                    );
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Structured matrix benchmarks (Toeplitz, Circulant)
// ---------------------------------------------------------------------------

fn bench_structured_matrices(c: &mut Criterion) {
    use scirs2_linalg::structured::{ToeplitzMatrix, CirculantMatrix, StructuredMatrix};

    let mut group = c.benchmark_group("structured_matrices");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let sizes: &[usize] = &[64, 256, 1024, 4096];

    for &n in sizes {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let dist = Uniform::new(-1.0f64, 1.0f64).expect("uniform range");
        let first_row: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let first_col: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let x: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();

        group.throughput(Throughput::Elements((n * n) as u64));

        // Toeplitz matrix-vector multiply
        group.bench_with_input(
            BenchmarkId::new("toeplitz_matvec", n),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let first_col_arr = scirs2_core::ndarray::Array1::from(first_col.clone());
                    let first_row_arr = scirs2_core::ndarray::Array1::from(first_row.clone());
                    let x_arr = scirs2_core::ndarray::Array1::from(x.clone());
                    let toeplitz = ToeplitzMatrix::new(
                        black_box(first_row_arr.view()),
                        black_box(first_col_arr.view()),
                    );
                    let result = toeplitz.map(|t| t.matvec(black_box(&x_arr.view())));
                    black_box(result)
                })
            },
        );

        // Circulant matrix-vector multiply (FFT-based O(n log n))
        group.bench_with_input(
            BenchmarkId::new("circulant_matvec", n),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let first_row_arr = scirs2_core::ndarray::Array1::from(first_row.clone());
                    let x_arr = scirs2_core::ndarray::Array1::from(x.clone());
                    let circ = CirculantMatrix::new(black_box(first_row_arr.view()));
                    let result = circ.map(|c| c.matvec(black_box(&x_arr.view())));
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
    tensor_decomp_benchmarks,
    bench_cp_decomposition,
    bench_tucker_decomposition,
    bench_hosvd,
    bench_tensor_train_svd,
);

criterion_group!(
    matrix_algorithm_benchmarks,
    bench_randomized_svd,
    bench_structured_matrices,
);

criterion_main!(tensor_decomp_benchmarks, matrix_algorithm_benchmarks);
