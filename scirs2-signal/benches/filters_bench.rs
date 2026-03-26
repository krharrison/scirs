//! Criterion benchmark suite for digital filter operations.
//!
//! Covers:
//! - FIR filter design and application (64-tap, 1024-sample signal)
//! - IIR Butterworth filter design and application
//! - STFT on 4096-sample signal
//!
//! Data is generated arithmetically — no file I/O.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_signal::filter::{butter, filtfilt, firwin, lfilter, FilterType};
use scirs2_signal::stft;
use std::hint::black_box;

// ---------------------------------------------------------------------------
// Signal helpers — no rand/ndarray, pure arithmetic
// ---------------------------------------------------------------------------

/// Generate a sinusoidal test signal with a known frequency.
fn make_signal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (2.0 * std::f64::consts::PI * 5.0 * t).sin()
                + 0.5 * (2.0 * std::f64::consts::PI * 50.0 * t).sin()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// FIR filter benchmarks
// ---------------------------------------------------------------------------

fn fir_design_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("fir_design");

    for &ntaps in &[32usize, 64, 128] {
        group.bench_with_input(
            BenchmarkId::new("firwin_lowpass", ntaps),
            &ntaps,
            |b, &n| {
                b.iter(|| {
                    black_box(
                        firwin(n, 0.3_f64, "hamming", true).expect("FIR filter design failed"),
                    )
                });
            },
        );
    }

    group.finish();
}

fn fir_apply_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("fir_apply");

    // Pre-design filter so design cost is not included in apply measurement.
    let taps =
        firwin(64_usize, 0.3_f64, "hamming", true).expect("FIR filter design (64-tap) failed");
    let a = vec![1.0_f64]; // FIR denominator is all-ones

    for &n in &[512usize, 1024, 4096] {
        let signal = make_signal(n);

        group.bench_with_input(BenchmarkId::new("lfilter_64tap", n), &signal, |b, sig| {
            b.iter(|| black_box(lfilter(&taps, &a, sig).expect("lfilter failed")));
        });

        group.bench_with_input(BenchmarkId::new("filtfilt_64tap", n), &signal, |b, sig| {
            b.iter(|| black_box(filtfilt(&taps, &a, sig).expect("filtfilt failed")));
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// IIR Butterworth filter benchmarks
// ---------------------------------------------------------------------------

fn iir_design_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("iir_design");

    for &order in &[2usize, 4, 6, 8] {
        group.bench_with_input(
            BenchmarkId::new("butter_lowpass", order),
            &order,
            |b, &ord| {
                b.iter(|| {
                    black_box(
                        butter(ord, 0.3_f64, FilterType::Lowpass)
                            .expect("Butterworth design failed"),
                    )
                });
            },
        );
    }

    group.finish();
}

fn iir_apply_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("iir_apply");

    // Pre-design filters to isolate apply cost.
    let (b4, a4) =
        butter(4usize, 0.3_f64, FilterType::Lowpass).expect("Butterworth order-4 design failed");
    let (b8, a8) =
        butter(8usize, 0.3_f64, FilterType::Lowpass).expect("Butterworth order-8 design failed");

    for &n in &[512usize, 1024, 4096] {
        let signal = make_signal(n);

        group.bench_with_input(BenchmarkId::new("lfilter_butter4", n), &signal, |b, sig| {
            b.iter(|| black_box(lfilter(&b4, &a4, sig).expect("lfilter (butter4) failed")));
        });

        group.bench_with_input(BenchmarkId::new("lfilter_butter8", n), &signal, |b, sig| {
            b.iter(|| black_box(lfilter(&b8, &a8, sig).expect("lfilter (butter8) failed")));
        });

        group.bench_with_input(
            BenchmarkId::new("filtfilt_butter4", n),
            &signal,
            |b, sig| {
                b.iter(|| black_box(filtfilt(&b4, &a4, sig).expect("filtfilt (butter4) failed")));
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// STFT benchmark
// ---------------------------------------------------------------------------

fn stft_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("stft");

    for &n in &[1024usize, 4096, 8192] {
        let signal = make_signal(n);

        group.bench_with_input(BenchmarkId::new("stft_hann256", n), &signal, |b, sig| {
            b.iter(|| {
                black_box(
                    stft(
                        sig,
                        Some(1000.0), // fs
                        Some("hann"),
                        Some(256), // nperseg
                        Some(128), // noverlap
                        None,
                        None,
                        None,
                        None,
                    )
                    .expect("STFT failed"),
                )
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    fir_design_bench,
    fir_apply_bench,
    iir_design_bench,
    iir_apply_bench,
    stft_bench
);
criterion_main!(benches);
