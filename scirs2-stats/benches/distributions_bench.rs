//! Focused criterion benchmark suite for statistical distribution PDF/CDF/PPF.
//!
//! Benchmarks normal, beta, gamma, and chi-square distributions across
//! 1000 samples to detect performance regressions in the critical evaluation paths.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_stats::distributions::{beta, chi2, gamma, norm};
use scirs2_stats::{ContinuousDistribution, Distribution};
use std::hint::black_box;

const N: usize = 1000;

fn make_prob_values() -> Vec<f64> {
    (1..=N).map(|i| i as f64 / (N + 1) as f64).collect()
}

fn make_positive_values() -> Vec<f64> {
    (0..N).map(|i| 0.05 + i as f64 * 0.02).collect()
}

fn make_symmetric_values() -> Vec<f64> {
    (0..N)
        .map(|i| -3.0 + i as f64 * (6.0 / (N - 1) as f64))
        .collect()
}

// ---------------------------------------------------------------------------
// PDF benchmarks
// ---------------------------------------------------------------------------

fn pdf_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("pdf_bench");

    let sym = make_symmetric_values();
    let pos = make_positive_values();

    group.bench_with_input(BenchmarkId::new("normal", N), &sym, |b, xs| {
        let dist = norm(0.0, 1.0).expect("normal distribution construction failed");
        b.iter(|| {
            for &x in xs {
                black_box(dist.pdf(x));
            }
        });
    });

    group.bench_with_input(BenchmarkId::new("beta", N), &pos, |b, xs| {
        let dist = beta(2.0, 5.0, 0.0, 1.0).expect("beta distribution construction failed");
        // Beta PDF is defined on (0, 1); clamp xs to that range
        let unit: Vec<f64> = xs.iter().map(|&x| x.clamp(0.001, 0.999)).collect();
        b.iter(|| {
            for &x in &unit {
                black_box(dist.pdf(x));
            }
        });
    });

    group.bench_with_input(BenchmarkId::new("gamma", N), &pos, |b, xs| {
        let dist = gamma(2.0, 1.0, 0.0).expect("gamma distribution construction failed");
        b.iter(|| {
            for &x in xs {
                black_box(dist.pdf(x));
            }
        });
    });

    group.bench_with_input(BenchmarkId::new("chisquare", N), &pos, |b, xs| {
        let dist = chi2(5.0, 0.0, 1.0).expect("chi-square distribution construction failed");
        b.iter(|| {
            for &x in xs {
                black_box(dist.pdf(x));
            }
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// CDF benchmarks
// ---------------------------------------------------------------------------

fn cdf_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("cdf_bench");

    let sym = make_symmetric_values();
    let pos = make_positive_values();

    group.bench_with_input(BenchmarkId::new("normal", N), &sym, |b, xs| {
        let dist = norm(0.0, 1.0).expect("normal distribution construction failed");
        b.iter(|| {
            for &x in xs {
                black_box(dist.cdf(x));
            }
        });
    });

    group.bench_with_input(BenchmarkId::new("beta", N), &pos, |b, xs| {
        let dist = beta(2.0, 5.0, 0.0, 1.0).expect("beta distribution construction failed");
        let unit: Vec<f64> = xs.iter().map(|&x| x.clamp(0.001, 0.999)).collect();
        b.iter(|| {
            for &x in &unit {
                black_box(dist.cdf(x));
            }
        });
    });

    group.bench_with_input(BenchmarkId::new("gamma", N), &pos, |b, xs| {
        let dist = gamma(2.0, 1.0, 0.0).expect("gamma distribution construction failed");
        b.iter(|| {
            for &x in xs {
                black_box(dist.cdf(x));
            }
        });
    });

    group.bench_with_input(BenchmarkId::new("chisquare", N), &pos, |b, xs| {
        let dist = chi2(5.0, 0.0, 1.0).expect("chi-square distribution construction failed");
        b.iter(|| {
            for &x in xs {
                black_box(dist.cdf(x));
            }
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// PPF benchmarks
// ---------------------------------------------------------------------------

fn ppf_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("ppf_bench");

    let probs = make_prob_values();

    group.bench_with_input(BenchmarkId::new("normal", N), &probs, |b, ps| {
        let dist = norm(0.0, 1.0).expect("normal distribution construction failed");
        b.iter(|| {
            for &p in ps {
                let _ = black_box(dist.ppf(p));
            }
        });
    });

    group.bench_with_input(BenchmarkId::new("beta", N), &probs, |b, ps| {
        let dist = beta(2.0, 5.0, 0.0, 1.0).expect("beta distribution construction failed");
        b.iter(|| {
            for &p in ps {
                let _ = black_box(dist.ppf(p));
            }
        });
    });

    group.bench_with_input(BenchmarkId::new("gamma", N), &probs, |b, ps| {
        let dist = gamma(2.0, 1.0, 0.0).expect("gamma distribution construction failed");
        b.iter(|| {
            for &p in ps {
                let _ = black_box(dist.ppf(p));
            }
        });
    });

    group.bench_with_input(BenchmarkId::new("chisquare", N), &probs, |b, ps| {
        let dist = chi2(5.0, 0.0, 1.0).expect("chi-square distribution construction failed");
        b.iter(|| {
            for &p in ps {
                let _ = black_box(dist.ppf(p));
            }
        });
    });

    group.finish();
}

criterion_group!(benches, pdf_bench, cdf_bench, ppf_bench);
criterion_main!(benches);
