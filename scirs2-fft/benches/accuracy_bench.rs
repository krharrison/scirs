//! FFT numerical accuracy benchmarks for scirs2-fft.
//!
//! These benchmarks compare SciRS2 FFT outputs against analytic reference
//! values.  No external library is required — all references are derived
//! from closed-form Fourier analysis:
//!
//! * DFT of a DC signal has energy only at bin 0 (magnitude = N).
//! * DFT of a pure tone sin(2π k₀ n / N) has energy at bins ±k₀ (magnitude = N/2 each).
//! * IDFT(DFT(x)) ≈ x with error bounded by ε · N · log₂N.
//! * Parseval: ∑|x[n]|² = (1/N) ∑|X[k]|².
//! * Linearity: FFT(a·x + b·y) = a·FFT(x) + b·FFT(y).
//!
//! Every benchmark panics if the measured error exceeds its tolerance,
//! making accuracy regressions visible in Criterion output.

use std::f64::consts::PI;
use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::numeric::Complex64;
use scirs2_fft::{fft, ifft};

// ---------------------------------------------------------------------------
// Signal generators
// ---------------------------------------------------------------------------

/// DC signal: x[n] = 1.0 for all n.
fn dc_signal(n: usize) -> Vec<f64> {
    vec![1.0_f64; n]
}

/// Pure-tone signal: x[n] = sin(2π k₀ n / N).
fn pure_tone(n: usize, k0: usize) -> Vec<f64> {
    (0..n)
        .map(|m| (2.0 * PI * k0 as f64 * m as f64 / n as f64).sin())
        .collect()
}

/// Gaussian signal centred at N/2 with standard-deviation sigma.
fn gaussian_signal(n: usize, sigma: f64) -> Vec<f64> {
    let center = n as f64 / 2.0;
    (0..n)
        .map(|m| {
            let t = m as f64 - center;
            (-0.5 * (t / sigma).powi(2)).exp()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Benchmark: DC signal DFT
// ---------------------------------------------------------------------------

fn bench_dc_fft_accuracy(c: &mut Criterion) {
    let sizes: &[usize] = &[8, 16, 32, 64, 128, 256, 512, 1024];
    let mut group = c.benchmark_group("accuracy/fft_dc_signal");
    group.measurement_time(Duration::from_secs(3));

    for &n in sizes {
        let x = dc_signal(n);

        group.bench_with_input(BenchmarkId::new("dc_fft", n), &n, |bench, _| {
            bench.iter(|| {
                let spectrum: Vec<Complex64> =
                    fft(black_box(x.as_slice()), None).expect("fft of DC signal must succeed");

                let n_f64 = n as f64;
                // Bin 0 magnitude = N
                let dc_mag = spectrum[0].norm();
                assert!(
                    (dc_mag - n_f64).abs() / n_f64 < 1e-13,
                    "DC FFT bin 0: mag={dc_mag:.6e}, expected {n_f64}"
                );

                // All non-DC bins ≈ 0
                let tol = n_f64 * f64::EPSILON * 10.0;
                for (k, c) in spectrum.iter().enumerate().skip(1) {
                    let mag = c.norm();
                    assert!(mag < tol, "DC FFT bin {k}: mag={mag:.2e} > tol={tol:.2e}");
                }
                spectrum
            })
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: pure-tone DFT energy concentration
// ---------------------------------------------------------------------------

fn bench_tone_fft_accuracy(c: &mut Criterion) {
    // (N, k0)
    let cases: &[(usize, usize)] = &[(8, 1), (16, 3), (32, 5), (64, 10), (128, 20), (256, 40)];

    let mut group = c.benchmark_group("accuracy/fft_pure_tone");
    group.measurement_time(Duration::from_secs(3));

    for &(n, k0) in cases {
        let x = pure_tone(n, k0);
        let label = format!("N{n}_k{k0}");

        group.bench_with_input(
            BenchmarkId::new("tone_fft", &label),
            &label,
            |bench, _| {
                bench.iter(|| {
                    let spectrum: Vec<Complex64> = fft(black_box(x.as_slice()), None)
                        .expect("fft of pure tone must succeed");

                    let half_n = (n as f64) / 2.0;
                    let tol_signal = half_n * 1e-10;
                    let tol_noise  = half_n * 1e-10;

                    for (k, c) in spectrum.iter().enumerate() {
                        let mag = c.norm();
                        if k == k0 || k == n - k0 {
                            assert!(
                                (mag - half_n).abs() < tol_signal,
                                "tone FFT N={n} k0={k0}: signal bin {k} mag={mag:.4e}, expected {half_n:.4e}"
                            );
                        } else {
                            assert!(
                                mag < tol_noise,
                                "tone FFT N={n} k0={k0}: leakage bin {k} mag={mag:.2e} > tol {tol_noise:.2e}"
                            );
                        }
                    }
                    spectrum
                })
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: IDFT round-trip accuracy  IDFT(FFT(x)) ≈ x
// ---------------------------------------------------------------------------

fn bench_roundtrip_accuracy(c: &mut Criterion) {
    let sizes: &[usize] = &[8, 32, 128, 512, 2048];
    let mut group = c.benchmark_group("accuracy/fft_roundtrip");
    group.measurement_time(Duration::from_secs(3));

    for &n in sizes {
        // Multi-tone signal for a realistic test
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let fi = i as f64;
                (2.0 * PI * fi / n as f64 * 3.0).sin()
                    + 0.5 * (2.0 * PI * fi / n as f64 * 7.0).cos()
            })
            .collect();

        group.bench_with_input(BenchmarkId::new("roundtrip", n), &n, |bench, _| {
            bench.iter(|| {
                let spectrum: Vec<Complex64> =
                    fft(black_box(x.as_slice()), None).expect("fft must succeed for round-trip");
                let recovered: Vec<Complex64> = ifft(black_box(spectrum.as_slice()), None)
                    .expect("ifft must succeed for round-trip");

                // Standard FFT error bound: ε × N × log₂(N)
                let tol = f64::EPSILON * (n as f64) * (n as f64).log2().max(1.0) * 4.0;
                for (k, (orig, rec)) in x.iter().zip(recovered.iter()).enumerate() {
                    let err = (orig - rec.re).abs();
                    assert!(
                        err < tol,
                        "IDFT(FFT(x)) round-trip N={n} idx={k}: error={err:.2e} > tol={tol:.2e}"
                    );
                    // Imaginary part should be negligible for real input
                    assert!(
                        rec.im.abs() < tol,
                        "IDFT(FFT(x)) imag part N={n} idx={k}: {:.2e} > tol={tol:.2e}",
                        rec.im
                    );
                }
                recovered
            })
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Parseval's theorem  ∑|x[n]|² = (1/N) ∑|X[k]|²
// ---------------------------------------------------------------------------

fn bench_parseval_theorem(c: &mut Criterion) {
    let sizes: &[usize] = &[16, 64, 256, 1024];
    let mut group = c.benchmark_group("accuracy/fft_parseval");
    group.measurement_time(Duration::from_secs(3));

    for &n in sizes {
        let x: Vec<f64> = (0..n)
            .map(|i| ((i as f64 * 2.0 * PI / n as f64) * 5.0).sin())
            .collect();
        let time_energy: f64 = x.iter().map(|v| v * v).sum();

        group.bench_with_input(BenchmarkId::new("parseval", n), &n, |bench, _| {
            bench.iter(|| {
                let spectrum: Vec<Complex64> = fft(black_box(x.as_slice()), None)
                    .expect("fft must succeed for Parseval");

                // Parseval: ‖x‖² = (1/N) ‖X‖²
                let freq_energy: f64 = spectrum.iter().map(|c| c.norm_sqr()).sum::<f64>()
                    / n as f64;

                let rel_err = (time_energy - freq_energy).abs()
                    / time_energy.abs().max(f64::EPSILON);
                assert!(
                    rel_err < 1e-11,
                    "Parseval N={n}: time={time_energy:.6e} freq={freq_energy:.6e} rel_err={rel_err:.2e}"
                );
                freq_energy
            })
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Linearity  FFT(a·x + b·y) = a·FFT(x) + b·FFT(y)
// ---------------------------------------------------------------------------

fn bench_fft_linearity(c: &mut Criterion) {
    let n = 64_usize;
    let a_coeff = 2.5_f64;
    let b_coeff = -1.3_f64;

    let x: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 2.0 * PI / n as f64 * 3.0).sin())
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 2.0 * PI / n as f64 * 7.0).cos())
        .collect();
    let xy: Vec<f64> = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| a_coeff * xi + b_coeff * yi)
        .collect();

    let mut group = c.benchmark_group("accuracy/fft_linearity");
    group.measurement_time(Duration::from_secs(3));

    group.bench_function(format!("N{n}"), |bench| {
        bench.iter(|| {
            let fx: Vec<Complex64> =
                fft(black_box(x.as_slice()), None).expect("fft x must succeed");
            let fy: Vec<Complex64> =
                fft(black_box(y.as_slice()), None).expect("fft y must succeed");
            let fxy: Vec<Complex64> =
                fft(black_box(xy.as_slice()), None).expect("fft xy must succeed");

            // fxy[k] == a*fx[k] + b*fy[k] for all k
            let tol = f64::EPSILON * (n as f64) * 100.0;
            for k in 0..n {
                let expected = Complex64::new(
                    a_coeff * fx[k].re + b_coeff * fy[k].re,
                    a_coeff * fx[k].im + b_coeff * fy[k].im,
                );
                let diff = (fxy[k] - expected).norm();
                assert!(
                    diff < tol,
                    "linearity bin {k}: diff={diff:.2e} > tol={tol:.2e}"
                );
            }
            (fx, fy, fxy)
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: FFT of a Gaussian has monotone-decreasing magnitude near DC
// ---------------------------------------------------------------------------

fn bench_gaussian_fft_shape(c: &mut Criterion) {
    let n = 128_usize;
    let sigma = 8.0_f64;
    let x = gaussian_signal(n, sigma);

    let mut group = c.benchmark_group("accuracy/fft_gaussian_shape");
    group.measurement_time(Duration::from_secs(3));

    group.bench_function(format!("N{n}_sigma{sigma}"), |bench| {
        bench.iter(|| {
            let spectrum: Vec<Complex64> =
                fft(black_box(x.as_slice()), None).expect("fft of Gaussian must succeed");

            // The magnitude of the (unshifted) spectrum should be dominated
            // by bin 0 and decrease through the first 10 bins.
            let mut prev_mag = f64::INFINITY;
            for (k, s) in spectrum.iter().enumerate().take(10) {
                let mag = s.norm();
                assert!(
                    mag <= prev_mag * 1.01, // 1% tolerance for numerical noise
                    "Gaussian FFT: non-monotone at bin {k}: {mag:.4e} > prev {prev_mag:.4e}"
                );
                prev_mag = mag;
            }
            spectrum
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion harness wiring
// ---------------------------------------------------------------------------

criterion_group!(
    benches_accuracy,
    bench_dc_fft_accuracy,
    bench_tone_fft_accuracy,
    bench_roundtrip_accuracy,
    bench_parseval_theorem,
    bench_fft_linearity,
    bench_gaussian_fft_shape,
);

criterion_main!(benches_accuracy);
