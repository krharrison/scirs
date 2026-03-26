//! Benchmarks for matrix decompositions: LU, QR, and SVD.
//!
//! Three Criterion benchmark groups cover the canonical factorizations at
//! matrix sizes 64×64, 128×128 and 256×256.  Each group also includes a
//! "solve" pass (LU group) and a "least-squares" pass (QR group) to capture
//! the full end-to-end decomposition+application cost.

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_linalg::{lu, qr, solve, svd};

/// Build a diagonally-dominant, well-conditioned n×n f64 matrix.
fn make_matrix(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, n), |(i, j)| {
        if i == j {
            (n + i + 1) as f64
        } else {
            0.1 * ((i * n + j) as f64 * 0.01).sin()
        }
    })
}

/// Build a right-hand-side vector of length n.
fn make_rhs(n: usize) -> Array1<f64> {
    Array1::from_shape_fn(n, |i| ((i + 1) as f64 * 0.1).sin())
}

/// Build a non-square m×n matrix for SVD testing.
fn make_rect(m: usize, n: usize) -> Array2<f64> {
    Array2::from_shape_fn((m, n), |(i, j)| {
        ((i + j + 1) as f64 * 0.07).sin() + 0.001 * (i as f64)
    })
}

// ---------------------------------------------------------------------------
// LU decomposition
// ---------------------------------------------------------------------------

fn bench_lu(c: &mut Criterion) {
    let mut group = c.benchmark_group("decomp/lu");
    group.measurement_time(Duration::from_secs(5));

    for &size in &[64usize, 128, 256] {
        let a = make_matrix(size);
        let b = make_rhs(size);

        group.throughput(Throughput::Elements((size * size) as u64));

        // Pure factorization
        group.bench_with_input(BenchmarkId::new("factorize", size), &size, |bench, _| {
            bench.iter(|| lu(black_box(&a.view()), None).expect("lu failed"))
        });

        // Factorize + solve in one go (using scirs2-linalg `solve` which uses LU internally)
        group.bench_with_input(BenchmarkId::new("solve", size), &size, |bench, _| {
            bench.iter(|| {
                solve(black_box(&a.view()), black_box(&b.view()), None).expect("solve failed")
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// QR decomposition
// ---------------------------------------------------------------------------

fn bench_qr(c: &mut Criterion) {
    let mut group = c.benchmark_group("decomp/qr");
    group.measurement_time(Duration::from_secs(5));

    for &size in &[64usize, 128, 256] {
        let a = make_matrix(size);

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(BenchmarkId::new("factorize", size), &size, |bench, _| {
            bench.iter(|| qr(black_box(&a.view()), None).expect("qr failed"))
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// SVD
// ---------------------------------------------------------------------------

fn bench_svd(c: &mut Criterion) {
    let mut group = c.benchmark_group("decomp/svd");
    // SVD is O(min(m,n)²·max(m,n)); keep sample_size modest.
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(8));

    for &size in &[64usize, 128, 256] {
        let square = make_matrix(size);
        // Tall rectangular matrix (2:1 aspect ratio)
        let rect = make_rect(size * 2, size);

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(BenchmarkId::new("square_full", size), &size, |bench, _| {
            bench.iter(|| svd(black_box(&square.view()), true, None).expect("svd failed"))
        });

        group.bench_with_input(
            BenchmarkId::new("square_values_only", size),
            &size,
            |bench, _| {
                bench.iter(|| svd(black_box(&square.view()), false, None).expect("svd failed"))
            },
        );

        group.bench_with_input(BenchmarkId::new("rect_full", size), &size, |bench, _| {
            bench.iter(|| svd(black_box(&rect.view()), true, None).expect("svd failed"))
        });
    }

    group.finish();
}

criterion_group!(decomp_benches, bench_lu, bench_qr, bench_svd);
criterion_main!(decomp_benches);
