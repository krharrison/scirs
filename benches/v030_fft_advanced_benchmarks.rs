//! SciRS2 v0.3.0 FFT Advanced Benchmark Suite
//!
//! Benchmarks for new v0.3.0 FFT capabilities:
//! - Sparse FFT: sub-linear runtime for k-sparse signals
//! - Number Theoretic Transform (NTT): exact integer convolution
//! - Comparison of FFT vs sparse FFT at various sparsity levels
//!
//! Performance Targets (v0.3.0):
//! - Sparse FFT: sub-linear scaling when k << N
//! - NTT forward/inverse round-trip: <2× dense FFT for N <= 2^20
//! - Polynomial multiply via NTT: competitive with schoolbook for N >= 512

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use scirs2_core::random::{ChaCha8Rng, SeedableRng, Uniform, Distribution};
use scirs2_fft::sparse_fft::{sparse_fft, SparseFFTConfig, SparsityEstimationMethod};
use scirs2_fft::ntt::{
    ntt_forward, ntt_inverse, poly_multiply_ntt, convolve_integer, MOD_998244353,
};
use std::hint::black_box;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Helper: build a k-sparse signal of length N
// ---------------------------------------------------------------------------

fn make_sparse_signal(n: usize, k: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let freq_dist = Uniform::new(0usize, n).expect("uniform range");
    let amp_dist  = Uniform::new(0.1f64, 2.0f64).expect("uniform range");
    let phase_dist= Uniform::new(0.0f64, std::f64::consts::TAU).expect("uniform range");

    let mut signal = vec![0.0f64; n];
    let mut freqs_chosen = Vec::with_capacity(k);

    // Pick k distinct frequencies
    let mut attempts = 0usize;
    while freqs_chosen.len() < k && attempts < n * 4 {
        let f = freq_dist.sample(&mut rng);
        if !freqs_chosen.contains(&f) {
            freqs_chosen.push(f);
        }
        attempts += 1;
    }

    for freq in &freqs_chosen {
        let amp   = amp_dist.sample(&mut rng);
        let phase = phase_dist.sample(&mut rng);
        for (i, s) in signal.iter_mut().enumerate() {
            let t = i as f64 / n as f64;
            *s += amp * (std::f64::consts::TAU * *freq as f64 * t + phase).cos();
        }
    }
    signal
}

// ---------------------------------------------------------------------------
// Helper: next power of two
// ---------------------------------------------------------------------------

fn next_pow2(n: usize) -> usize {
    if n == 0 { return 1; }
    let mut p = 1usize;
    while p < n { p <<= 1; }
    p
}

// ---------------------------------------------------------------------------
// Sparse FFT benchmarks
// ---------------------------------------------------------------------------

fn bench_sparse_fft_vs_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_fft_vs_dense");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    // Signal sizes (power-of-two)
    let signal_sizes: &[usize] = &[1024, 4096, 16384, 65536];
    // Sparsity: k components
    let sparsities: &[usize] = &[8, 32, 128];

    for &n in signal_sizes {
        for &k in sparsities {
            if k >= n / 4 { continue; } // skip trivially dense cases

            let signal = make_sparse_signal(n, k, 42);

            let cfg = SparseFFTConfig {
                sparsity: k * 2,
                estimation_method: SparsityEstimationMethod::Adaptive,
                ..SparseFFTConfig::default()
            };

            group.throughput(Throughput::Elements(n as u64));

            // Benchmark sparse FFT
            let label = format!("N={n}_k={k}");
            group.bench_with_input(
                BenchmarkId::new("sparse_fft", &label),
                &(),
                |bencher, _| {
                    bencher.iter(|| {
                        let result = sparse_fft(
                            black_box(&signal),
                            black_box(k * 2),
                            Some(cfg.clone()),
                            None,
                        );
                        black_box(result)
                    })
                },
            );

            // Benchmark full FFT for comparison (using scirs2-fft's standard path)
            group.bench_with_input(
                BenchmarkId::new("dense_rfft", &label),
                &(),
                |bencher, _| {
                    bencher.iter(|| {
                        let result = scirs2_fft::rfft(black_box(&signal));
                        black_box(result)
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_sparse_fft_algorithms(c: &mut Criterion) {
    use scirs2_fft::sparse_fft::SparseFFTAlgorithm;

    let mut group = c.benchmark_group("sparse_fft_algorithms");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(6));

    let n = 8192usize;
    let k = 16usize;
    let signal = make_sparse_signal(n, k, 99);

    group.throughput(Throughput::Elements(n as u64));

    let algorithms = [
        ("adaptive",          SparseFFTAlgorithm::Adaptive),
        ("frequency_pruning", SparseFFTAlgorithm::FrequencyPruning),
        ("spectral_flatness", SparseFFTAlgorithm::SpectralFlatness),
    ];

    for (name, alg) in &algorithms {
        let cfg = SparseFFTConfig {
            sparsity: k * 2,
            algorithm: alg.clone(),
            ..SparseFFTConfig::default()
        };
        group.bench_with_input(
            BenchmarkId::new("algorithm", name),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let result = sparse_fft(
                        black_box(&signal),
                        black_box(k * 2),
                        Some(cfg.clone()),
                        None,
                    );
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn bench_sparse_fft_sparsity_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_fft_sparsity_scaling");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    let n = 16384usize;
    let sparsities: &[usize] = &[4, 8, 16, 32, 64, 128, 256];

    for &k in sparsities {
        let signal = make_sparse_signal(n, k, 77);
        let cfg = SparseFFTConfig {
            sparsity: k * 2,
            estimation_method: SparsityEstimationMethod::Adaptive,
            ..SparseFFTConfig::default()
        };

        group.throughput(Throughput::Elements(k as u64));
        group.bench_with_input(
            BenchmarkId::new("sparse_fft_k", k),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let result = sparse_fft(
                        black_box(&signal),
                        black_box(k * 2),
                        Some(cfg.clone()),
                        None,
                    );
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// NTT benchmarks
// ---------------------------------------------------------------------------

fn make_ntt_data(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let dist = Uniform::new(0u64, MOD_998244353).expect("uniform range");
    (0..n).map(|_| dist.sample(&mut rng)).collect()
}

fn bench_ntt_forward_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt_forward_inverse");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let sizes: &[usize] = &[256, 1024, 4096, 16384, 65536, 262144];

    for &n in sizes {
        let data = make_ntt_data(n, 42);
        let modulus = MOD_998244353;

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(
            BenchmarkId::new("ntt_forward", n),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let result = ntt_forward(black_box(&data), black_box(modulus));
                    black_box(result)
                })
            },
        );

        let fwd = ntt_forward(&data, modulus).expect("ntt_forward in setup");
        group.bench_with_input(
            BenchmarkId::new("ntt_inverse", n),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let result = ntt_inverse(black_box(&fwd), black_box(modulus));
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn bench_ntt_polynomial_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt_polynomial_multiply");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    // Polynomial sizes (actual degree; NTT pads to next power of two)
    let poly_sizes: &[usize] = &[128, 512, 2048, 8192, 32768];
    let modulus = MOD_998244353;

    for &n in poly_sizes {
        let a = make_ntt_data(n, 11);
        let b = make_ntt_data(n, 22);

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(
            BenchmarkId::new("poly_multiply_ntt", n),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let result = poly_multiply_ntt(
                        black_box(&a),
                        black_box(&b),
                        black_box(modulus),
                    );
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn bench_ntt_integer_convolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt_integer_convolution");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(6));

    let sizes: &[usize] = &[256, 1024, 4096, 16384];

    for &n in sizes {
        let mut rng = ChaCha8Rng::seed_from_u64(55);
        let dist = Uniform::new(-1000i64, 1000i64).expect("uniform range");
        let a: Vec<i64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let b: Vec<i64> = (0..n).map(|_| dist.sample(&mut rng)).collect();

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(
            BenchmarkId::new("convolve_integer", n),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let result = convolve_integer(black_box(&a), black_box(&b));
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn bench_ntt_vs_float_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt_vs_float_fft_throughput");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let sizes: &[usize] = &[1024, 4096, 16384, 65536];
    let modulus = MOD_998244353;

    for &n in sizes {
        let ntt_data = make_ntt_data(n, 42);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let dist = Uniform::new(-1.0f64, 1.0f64).expect("uniform range");
        let fft_data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(
            BenchmarkId::new("ntt_u64", n),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let result = ntt_forward(black_box(&ntt_data), black_box(modulus));
                    black_box(result)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("rfft_f64", n),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    let result = scirs2_fft::rfft(black_box(&fft_data));
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
    sparse_fft_benchmarks,
    bench_sparse_fft_vs_dense,
    bench_sparse_fft_algorithms,
    bench_sparse_fft_sparsity_scaling,
);

criterion_group!(
    ntt_benchmarks,
    bench_ntt_forward_inverse,
    bench_ntt_polynomial_multiply,
    bench_ntt_integer_convolution,
    bench_ntt_vs_float_fft,
);

criterion_main!(sparse_fft_benchmarks, ntt_benchmarks);
