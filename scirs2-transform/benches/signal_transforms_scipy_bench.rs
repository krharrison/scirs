#![allow(clippy::unwrap_used)]
//! Benchmarks for Signal Transforms vs SciPy
//!
//! Compares performance of scirs2-transform signal transforms against SciPy equivalents.
//!
//! Run with: cargo bench --bench signal_transforms_scipy_bench

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::ndarray::Array1;
use scirs2_transform::signal_transforms::stft::SpectrogramScaling;
use scirs2_transform::signal_transforms::*;
use std::hint::black_box;

// Helper to generate test signals
fn generate_sine_wave(n: usize, freq: f64, sample_rate: f64) -> Array1<f64> {
    let dt = 1.0 / sample_rate;
    Array1::from_vec(
        (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 * dt).sin())
            .collect(),
    )
}

fn generate_noisy_signal(n: usize) -> Array1<f64> {
    use scirs2_core::random::{Rng, RngExt};
    let mut rng = scirs2_core::random::thread_rng();

    // Base signal: sum of sine waves
    let mut signal = Array1::zeros(n);
    for i in 0..n {
        let t = i as f64 / 1000.0;
        signal[i] = (2.0 * std::f64::consts::PI * 50.0 * t).sin()
            + 0.5 * (2.0 * std::f64::consts::PI * 120.0 * t).sin();
        // Add noise
        signal[i] += rng.gen_range(-0.1..0.1);
    }

    signal
}

// DWT Benchmarks
fn bench_dwt(c: &mut Criterion) {
    let mut group = c.benchmark_group("DWT");

    for size in &[256, 512, 1024, 2048, 4096] {
        let signal = generate_noisy_signal(*size);

        group.bench_with_input(BenchmarkId::new("Haar", size), &signal, |b, sig| {
            b.iter(|| {
                let dwt = DWT::new(WaveletType::Haar).unwrap();
                dwt.wavedec(&sig.view()).unwrap()
            })
        });

        group.bench_with_input(BenchmarkId::new("DB4", size), &signal, |b, sig| {
            b.iter(|| {
                let dwt = DWT::new(WaveletType::Daubechies(4)).unwrap();
                dwt.wavedec(&sig.view()).unwrap()
            })
        });
    }

    group.finish();
}

// DWT2D Benchmarks
fn bench_dwt2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("DWT2D");

    for size in &[64, 128, 256] {
        let image = scirs2_core::ndarray::Array2::from_shape_fn((*size, *size), |(i, j)| {
            ((i + j) as f64 * 0.1).sin()
        });

        group.bench_with_input(BenchmarkId::new("Haar", size), &image, |b, img| {
            b.iter(|| {
                let dwt2d = DWT2D::new(WaveletType::Haar).unwrap();
                dwt2d.decompose2(&img.view()).unwrap()
            })
        });
    }

    group.finish();
}

// CWT Benchmarks
fn bench_cwt(c: &mut Criterion) {
    let mut group = c.benchmark_group("CWT");

    for size in &[256, 512, 1024] {
        let signal = generate_sine_wave(*size, 50.0, 1000.0);

        group.bench_with_input(
            BenchmarkId::new("Morlet_Direct", size),
            &signal,
            |b, sig| {
                b.iter(|| {
                    let wavelet = MorletWavelet::default();
                    let cwt = CWT::new(wavelet, vec![1.0, 2.0, 4.0, 8.0, 16.0]);
                    cwt.transform(&sig.view()).unwrap()
                })
            },
        );

        group.bench_with_input(BenchmarkId::new("Morlet_FFT", size), &signal, |b, sig| {
            b.iter(|| {
                let wavelet = MorletWavelet::default();
                let cwt = CWT::new(wavelet, vec![1.0, 2.0, 4.0, 8.0, 16.0]);
                cwt.transform_fft(&sig.view()).unwrap()
            })
        });

        group.bench_with_input(BenchmarkId::new("MexicanHat", size), &signal, |b, sig| {
            b.iter(|| {
                let wavelet = MexicanHatWavelet::default();
                let cwt = CWT::new(wavelet, vec![1.0, 2.0, 4.0, 8.0, 16.0]);
                cwt.transform_fft(&sig.view()).unwrap()
            })
        });
    }

    group.finish();
}

// WPT Benchmarks
fn bench_wpt(c: &mut Criterion) {
    let mut group = c.benchmark_group("WPT");

    for size in &[256, 512, 1024] {
        let signal = generate_noisy_signal(*size);

        group.bench_with_input(BenchmarkId::new("Decompose", size), &signal, |b, sig| {
            b.iter(|| {
                let mut wpt = WPT::new(WaveletType::Haar, 3);
                wpt.decompose(&sig.view()).unwrap()
            })
        });

        group.bench_with_input(BenchmarkId::new("BestBasis", size), &signal, |b, sig| {
            b.iter(|| {
                let mut wpt = WPT::new(WaveletType::Haar, 3);
                wpt.decompose(&sig.view()).unwrap();
                wpt.best_basis().unwrap()
            })
        });
    }

    group.finish();
}

// STFT Benchmarks
fn bench_stft(c: &mut Criterion) {
    let mut group = c.benchmark_group("STFT");

    for size in &[1024, 2048, 4096, 8192] {
        let signal = generate_sine_wave(*size, 50.0, 1000.0);

        group.bench_with_input(BenchmarkId::new("Forward", size), &signal, |b, sig| {
            b.iter(|| {
                let stft = STFT::with_params(256, 128);
                stft.transform(&sig.view()).unwrap()
            })
        });

        let stft = STFT::with_params(256, 128);
        let transformed = stft.transform(&signal.view()).unwrap();

        group.bench_with_input(
            BenchmarkId::new("Inverse", size),
            &transformed,
            |b, spec| b.iter(|| stft.inverse(spec).unwrap()),
        );
    }

    group.finish();
}

// Spectrogram Benchmarks
fn bench_spectrogram(c: &mut Criterion) {
    let mut group = c.benchmark_group("Spectrogram");

    for size in &[1024, 2048, 4096] {
        let signal = generate_noisy_signal(*size);

        group.bench_with_input(BenchmarkId::new("Power", size), &signal, |b, sig| {
            b.iter(|| {
                let config = STFTConfig {
                    window_size: 256,
                    hop_size: 128,
                    ..Default::default()
                };
                let spec = Spectrogram::new(config).with_scaling(SpectrogramScaling::Power);
                spec.compute(&sig.view()).unwrap()
            })
        });

        group.bench_with_input(BenchmarkId::new("Decibel", size), &signal, |b, sig| {
            b.iter(|| {
                let config = STFTConfig {
                    window_size: 256,
                    hop_size: 128,
                    ..Default::default()
                };
                let spec = Spectrogram::new(config).with_scaling(SpectrogramScaling::Decibel);
                spec.compute(&sig.view()).unwrap()
            })
        });
    }

    group.finish();
}

// MFCC Benchmarks
fn bench_mfcc(c: &mut Criterion) {
    let mut group = c.benchmark_group("MFCC");

    for size in &[8000, 16000, 32000] {
        let signal = generate_noisy_signal(*size);

        group.bench_with_input(BenchmarkId::new("Extract", size), &signal, |b, sig| {
            b.iter(|| {
                let mfcc = MFCC::default().unwrap();
                mfcc.extract(&sig.view()).unwrap()
            })
        });

        group.bench_with_input(BenchmarkId::new("WithDeltas", size), &signal, |b, sig| {
            b.iter(|| {
                let mfcc = MFCC::default().unwrap();
                mfcc.extract_with_deltas(&sig.view()).unwrap()
            })
        });
    }

    group.finish();
}

// CQT Benchmarks
fn bench_cqt(c: &mut Criterion) {
    let mut group = c.benchmark_group("CQT");

    for size in &[11025, 22050] {
        let signal = generate_sine_wave(*size, 440.0, 22050.0);

        group.bench_with_input(BenchmarkId::new("Transform", size), &signal, |b, sig| {
            b.iter(|| {
                let cqt = CQT::default().unwrap();
                cqt.transform(&sig.view()).unwrap()
            })
        });

        group.bench_with_input(BenchmarkId::new("Magnitude", size), &signal, |b, sig| {
            b.iter(|| {
                let cqt = CQT::default().unwrap();
                cqt.magnitude(&sig.view()).unwrap()
            })
        });
    }

    group.finish();
}

// Chromagram Benchmarks
fn bench_chromagram(c: &mut Criterion) {
    let mut group = c.benchmark_group("Chromagram");

    for size in &[11025, 22050] {
        let signal = generate_sine_wave(*size, 440.0, 22050.0);

        group.bench_with_input(BenchmarkId::new("Compute", size), &signal, |b, sig| {
            b.iter(|| {
                let chroma = Chromagram::default().unwrap();
                chroma.compute(&sig.view()).unwrap()
            })
        });

        group.bench_with_input(BenchmarkId::new("Normalized", size), &signal, |b, sig| {
            b.iter(|| {
                let chroma = Chromagram::default().unwrap();
                chroma.compute_normalized(&sig.view()).unwrap()
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_dwt,
    bench_dwt2d,
    bench_cwt,
    bench_wpt,
    bench_stft,
    bench_spectrogram,
    bench_mfcc,
    bench_cqt,
    bench_chromagram
);

criterion_main!(benches);
