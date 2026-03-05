//! SciRS2 v0.3.0 Signal Processing Advanced Benchmark Suite
//!
//! Benchmarks for v0.3.0 signal processing additions:
//! - MFCC: Mel-Frequency Cepstral Coefficients (speech features)
//! - EMD: Empirical Mode Decomposition
//! - VMD: Variational Mode Decomposition
//! - Zoom FFT / Sliding DFT / Goertzel algorithm
//!
//! Performance Targets (v0.3.0):
//! - MFCC (1-sec audio @16kHz, 13 coeffs): < 5 ms per call
//! - EMD (1000-sample signal, 5 IMFs): < 50 ms per call
//! - VMD (1000-sample signal, 4 modes): < 100 ms per call

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use scirs2_core::random::{ChaCha8Rng, SeedableRng, Uniform, Distribution};
use scirs2_signal::cepstral::{mfcc, MfccConfig};
use scirs2_signal::emd::{emd, EmdConfig};
use scirs2_signal::multiscale::vmd;
use scirs2_signal::zoom_fft::{goertzel, zoom_fft as zoom_fft_fn};
use std::hint::black_box;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Signal generators
// ---------------------------------------------------------------------------

/// Generate a multi-component sinusoidal signal (simulates speech-like content)
fn make_multicomponent_signal(n: usize, sample_rate: f64, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let amp_dist   = Uniform::new(0.1f64, 1.0f64).expect("uniform range");
    let freq_dist  = Uniform::new(100.0f64, 4000.0f64).expect("uniform range");
    let phase_dist = Uniform::new(0.0f64, std::f64::consts::TAU).expect("uniform range");

    let n_components = 5usize;
    let mut components: Vec<(f64, f64, f64)> = (0..n_components)
        .map(|_| (
            amp_dist.sample(&mut rng),
            freq_dist.sample(&mut rng),
            phase_dist.sample(&mut rng),
        ))
        .collect();
    // Add a dominant fundamental for realism
    components.push((2.0, 200.0, 0.0));

    (0..n).map(|i| {
        let t = i as f64 / sample_rate;
        components.iter().map(|(amp, freq, phase)| {
            amp * (std::f64::consts::TAU * freq * t + phase).sin()
        }).sum::<f64>()
    }).collect()
}

/// Generate a noisy signal for EMD/VMD (sum of AM-FM components + noise)
fn make_nonlinear_signal(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let noise_dist = Uniform::new(-0.1f64, 0.1f64).expect("uniform range");

    (0..n).map(|i| {
        let t = i as f64 / n as f64;
        // Three IMF-like components at different scales
        let c1 = (2.0 * std::f64::consts::TAU * 50.0 * t).sin();
        let c2 = 0.5 * (2.0 * std::f64::consts::TAU * 10.0 * t
                        + 0.3 * (2.0 * std::f64::consts::TAU * 2.0 * t).sin()).sin();
        let c3 = 0.3 * (2.0 * std::f64::consts::TAU * 2.0 * t);
        let noise = noise_dist.sample(&mut rng);
        c1 + c2 + c3 + noise
    }).collect()
}

// ---------------------------------------------------------------------------
// MFCC benchmarks
// ---------------------------------------------------------------------------

fn bench_mfcc(c: &mut Criterion) {
    let mut group = c.benchmark_group("mfcc");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let sample_rate = 16000.0f64;
    let signal_lengths_ms: &[usize] = &[100, 250, 500, 1000, 2000];
    let n_mfcc_coeffs:     &[usize] = &[13, 20, 40];

    for &ms in signal_lengths_ms {
        let n = (sample_rate * ms as f64 / 1000.0) as usize;
        let signal = make_multicomponent_signal(n, sample_rate, 42);

        group.throughput(Throughput::Elements(n as u64));

        for &n_mfcc in n_mfcc_coeffs {
            let label = format!("{ms}ms_{n_mfcc}coeff");
            group.bench_with_input(
                BenchmarkId::new("mfcc", &label),
                &(),
                |bencher, _| {
                    bencher.iter(|| {
                        let result = mfcc(
                            black_box(&signal),
                            black_box(sample_rate),
                            black_box(n_mfcc),
                        );
                        black_box(result)
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_mfcc_config_variants(c: &mut Criterion) {
    let mut group = c.benchmark_group("mfcc_config_variants");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(6));

    let sample_rate = 22050.0f64;
    let n = 22050usize; // 1 second
    let signal = make_multicomponent_signal(n, sample_rate, 77);

    group.throughput(Throughput::Elements(n as u64));

    // Different window sizes
    let fft_sizes: &[usize] = &[256, 512, 1024, 2048];
    for &fft_size in fft_sizes {
        let mut cfg = MfccConfig::new(sample_rate);
        cfg.n_fft = fft_size;
        cfg.hop_length = fft_size / 2;
        cfg.n_mels = 40;
        cfg.n_mfcc = 13;

        group.bench_with_input(
            BenchmarkId::new("fft_size", fft_size),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    use scirs2_signal::cepstral::mfcc_extract;
                    let result = mfcc_extract(black_box(&signal), black_box(&cfg));
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// EMD benchmarks
// ---------------------------------------------------------------------------

fn bench_emd(c: &mut Criterion) {
    let mut group = c.benchmark_group("emd_empirical_mode_decomposition");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let signal_sizes: &[usize] = &[256, 512, 1000, 2048];
    let max_imfs_variants: &[usize] = &[3, 5];

    for &n in signal_sizes {
        let signal = make_nonlinear_signal(n, 42);

        group.throughput(Throughput::Elements(n as u64));

        for &max_imfs in max_imfs_variants {
            let config = EmdConfig {
                max_imfs,
                max_sifting_iterations: 20,
                sd_threshold: 0.2,
                ..EmdConfig::default()
            };
            let label = format!("n={n}_imfs={max_imfs}");
            group.bench_with_input(
                BenchmarkId::new("emd", &label),
                &(),
                |bencher, _| {
                    bencher.iter(|| {
                        let result = emd(black_box(&signal), black_box(&config));
                        black_box(result)
                    })
                },
            );
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// VMD benchmarks
// ---------------------------------------------------------------------------

fn bench_vmd(c: &mut Criterion) {
    let mut group = c.benchmark_group("vmd_variational_mode_decomposition");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let signal_sizes: &[usize] = &[256, 512, 1024];
    let n_modes_variants: &[usize] = &[2, 3, 4];

    for &n in signal_sizes {
        let signal = make_nonlinear_signal(n, 55);

        group.throughput(Throughput::Elements(n as u64));

        for &n_modes in n_modes_variants {
            let label = format!("n={n}_modes={n_modes}");
            group.bench_with_input(
                BenchmarkId::new("vmd", &label),
                &(),
                |bencher, _| {
                    bencher.iter(|| {
                        let result = vmd(
                            black_box(&signal),
                            black_box(n_modes),
                            2000.0, // alpha: bandwidth constraint
                            0.0,    // tau: Lagrangian multiplier update rate
                            200,    // max_iter
                            1e-6,   // tol
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
// Zoom FFT / Goertzel benchmarks
// ---------------------------------------------------------------------------

fn bench_goertzel_vs_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("goertzel_vs_fft");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(6));

    let signal_sizes: &[usize] = &[256, 1024, 4096, 16384];
    let n_target_freqs: &[usize] = &[1, 4, 16];
    let sample_rate = 8000.0f64;

    for &n in signal_sizes {
        let signal = make_multicomponent_signal(n, sample_rate, 33);

        group.throughput(Throughput::Elements(n as u64));

        // Full FFT baseline
        group.bench_with_input(
            BenchmarkId::new("rfft_full", n),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let result = scirs2_fft::rfft(black_box(&signal));
                    black_box(result)
                })
            },
        );

        // Goertzel for specific frequencies
        for &n_freqs in n_target_freqs {
            let target_freqs: Vec<f64> = (0..n_freqs)
                .map(|i| 440.0 * (i + 1) as f64)
                .collect();

            let label = format!("n={n}_freqs={n_freqs}");
            group.bench_with_input(
                BenchmarkId::new("goertzel", &label),
                &(),
                |bencher, _| {
                    bencher.iter(|| {
                        let results: Vec<_> = target_freqs
                            .iter()
                            .map(|&freq| {
                                goertzel(
                                    black_box(&signal),
                                    black_box(freq),
                                    black_box(sample_rate),
                                )
                            })
                            .collect();
                        black_box(results)
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_zoom_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("zoom_fft");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(6));

    let sample_rate = 16000.0f64;
    let n = 8192usize;
    let signal = make_multicomponent_signal(n, sample_rate, 88);

    group.throughput(Throughput::Elements(n as u64));

    // Different zoom windows
    let zoom_configs: &[(&str, f64, f64, usize)] = &[
        ("narrow_100",   200.0,  300.0,  256),
        ("medium_1k",    500.0,  1500.0, 512),
        ("wide_4k",      0.0,    4000.0, 1024),
    ];

    for &(label, f_low, f_high, n_points) in zoom_configs {
        group.bench_with_input(
            BenchmarkId::new("zoom_fft", label),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let result = zoom_fft_fn(
                        black_box(&signal),
                        black_box(f_low),
                        black_box(f_high),
                        black_box(n_points),
                        black_box(sample_rate),
                    );
                    black_box(result)
                })
            },
        );
    }

    // Full FFT for comparison
    group.bench_with_input(
        BenchmarkId::new("rfft_baseline", n),
        &(),
        |bencher, _| {
            bencher.iter(|| {
                let result = scirs2_fft::rfft(black_box(&signal));
                black_box(result)
            })
        },
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion groups and main
// ---------------------------------------------------------------------------

criterion_group!(
    mfcc_benchmarks,
    bench_mfcc,
    bench_mfcc_config_variants,
);

criterion_group!(
    decomposition_benchmarks,
    bench_emd,
    bench_vmd,
);

criterion_group!(
    frequency_benchmarks,
    bench_goertzel_vs_fft,
    bench_zoom_fft,
);

criterion_main!(mfcc_benchmarks, decomposition_benchmarks, frequency_benchmarks);
