//! Focused criterion benchmarks for core FFT operations.
//!
//! Covers 1-D, 2-D and real FFT transforms at sizes 64, 128 and 256.
//! Each variant lives in a dedicated Criterion benchmark group so that
//! regression tracking remains granular.

use std::f64::consts::PI;
use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array2;
use scirs2_core::Complex64;
use scirs2_fft::{fft, fft2, ifft, irfft, rfft};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a real-valued sine signal of length `n`.
fn make_real_signal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (2.0 * PI * 10.0 * i as f64 / n as f64).sin())
        .collect()
}

/// Wrap a real signal as a complex signal (imaginary part = 0).
fn make_complex_signal(n: usize) -> Vec<Complex64> {
    make_real_signal(n)
        .into_iter()
        .map(|x| Complex64::new(x, 0.0))
        .collect()
}

/// Generate a 2-D real array of shape `n×n` as a flat ndarray.
fn make_2d_array(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, n), |(i, j)| {
        let x = i as f64 / n as f64;
        let y = j as f64 / n as f64;
        (2.0 * PI * (5.0 * x + 3.0 * y)).sin()
    })
}

// ---------------------------------------------------------------------------
// 1-D complex FFT / IFFT
// ---------------------------------------------------------------------------

fn bench_fft_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft/1d_complex_forward");
    group.measurement_time(Duration::from_secs(5));

    for &size in &[64usize, 128, 256] {
        let signal = make_complex_signal(size);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("fft", size), &signal, |b, s| {
            b.iter(|| fft(black_box(s), None))
        });
    }

    group.finish();
}

fn bench_ifft_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft/1d_complex_inverse");
    group.measurement_time(Duration::from_secs(5));

    for &size in &[64usize, 128, 256] {
        let signal = make_complex_signal(size);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("ifft", size), &signal, |b, s| {
            b.iter(|| ifft(black_box(s), None))
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 1-D real FFT / IRFFT
// ---------------------------------------------------------------------------

fn bench_rfft_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft/1d_real_forward");
    group.measurement_time(Duration::from_secs(5));

    for &size in &[64usize, 128, 256] {
        let signal = make_real_signal(size);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("rfft", size), &signal, |b, s| {
            b.iter(|| rfft(black_box(s), None))
        });
    }

    group.finish();
}

fn bench_irfft_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft/1d_real_inverse");
    group.measurement_time(Duration::from_secs(5));

    for &size in &[64usize, 128, 256] {
        let signal = make_real_signal(size);
        let spectrum = rfft(&signal, None).expect("rfft failed");

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("irfft", size), &spectrum, |b, s| {
            b.iter(|| irfft(black_box(s), Some(size)))
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 2-D FFT
// ---------------------------------------------------------------------------

fn bench_fft_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft/2d_forward");
    // 2-D transforms are heavier — reduce sample size.
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(8));

    for &size in &[64usize, 128, 256] {
        let data = make_2d_array(size);

        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(
            BenchmarkId::new("fft2", size),
            &data,
            |b, d: &Array2<f64>| b.iter(|| fft2(black_box(d), None, None, None)),
        );
    }

    group.finish();
}

criterion_group!(
    fft_benches,
    bench_fft_1d,
    bench_ifft_1d,
    bench_rfft_1d,
    bench_irfft_1d,
    bench_fft_2d,
);
criterion_main!(fft_benches);
