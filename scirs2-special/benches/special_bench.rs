//! Criterion benchmark suite for special mathematical functions.
//!
//! Benchmarks gamma, Bessel, Legendre, hypergeometric, and elliptic integral
//! functions over arrays of 1000 values to detect performance regressions.
//!
//! Input data is generated arithmetically — no file I/O, no rand/ndarray.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_special::{
    bessel, chebyshev, ellipe, ellipk, erf, erfc, gamma, gammaln, hermite, hyp1f1, hyp2f1,
    laguerre, legendre,
};
use std::hint::black_box;

const N: usize = 1000;

// ---------------------------------------------------------------------------
// Input generators — pure arithmetic, no external deps
// ---------------------------------------------------------------------------

fn make_positive(n: usize) -> Vec<f64> {
    (1..=n).map(|i| i as f64 * 0.005 + 0.5).collect()
}

fn make_unit_interval(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
}

fn make_symmetric(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| -1.0 + 2.0 * i as f64 / (n - 1) as f64)
        .collect()
}

fn make_elliptic_m(n: usize) -> Vec<f64> {
    // m in (0, 1) exclusive — avoid singularity at m=1
    (1..=n).map(|i| i as f64 / (n + 2) as f64).collect()
}

// ---------------------------------------------------------------------------
// Gamma function benchmarks
// ---------------------------------------------------------------------------

fn gamma_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("gamma");

    let xs = make_positive(N);

    group.bench_with_input(BenchmarkId::new("gamma", N), &xs, |b, xs| {
        b.iter(|| {
            for &x in xs {
                black_box(gamma(black_box(x)));
            }
        });
    });

    group.bench_with_input(BenchmarkId::new("gammaln", N), &xs, |b, xs| {
        b.iter(|| {
            for &x in xs {
                black_box(gammaln(black_box(x)));
            }
        });
    });

    group.bench_with_input(BenchmarkId::new("erf", N), &xs, |b, xs| {
        b.iter(|| {
            for &x in xs {
                black_box(erf(black_box(x)));
            }
        });
    });

    group.bench_with_input(BenchmarkId::new("erfc", N), &xs, |b, xs| {
        b.iter(|| {
            for &x in xs {
                black_box(erfc(black_box(x)));
            }
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Bessel function benchmarks
// ---------------------------------------------------------------------------

fn bessel_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("bessel");

    let xs = make_positive(N);

    group.bench_with_input(BenchmarkId::new("j0", N), &xs, |b, xs| {
        b.iter(|| {
            for &x in xs {
                black_box(bessel::j0(black_box(x)));
            }
        });
    });

    group.bench_with_input(BenchmarkId::new("j1", N), &xs, |b, xs| {
        b.iter(|| {
            for &x in xs {
                black_box(bessel::j1(black_box(x)));
            }
        });
    });

    group.bench_with_input(BenchmarkId::new("i0", N), &xs, |b, xs| {
        b.iter(|| {
            for &x in xs {
                black_box(bessel::i0(black_box(x)));
            }
        });
    });

    // j_n for higher orders
    for order in [2i32, 5, 10] {
        group.bench_with_input(
            BenchmarkId::new(format!("jn_order{}", order), N),
            &xs,
            |b, xs| {
                b.iter(|| {
                    for &x in xs {
                        black_box(bessel::jn(black_box(order), black_box(x)));
                    }
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Legendre & orthogonal polynomial benchmarks
// ---------------------------------------------------------------------------

fn legendre_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("orthogonal_polynomials");

    let xs = make_symmetric(N);
    let unit = make_unit_interval(N);

    for degree in [2usize, 5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::new(format!("legendre_n{}", degree), N),
            &xs,
            |b, xs| {
                b.iter(|| {
                    for &x in xs {
                        black_box(legendre(black_box(degree), black_box(x)));
                    }
                });
            },
        );
    }

    group.bench_with_input(BenchmarkId::new("hermite_n5", N), &xs, |b, xs| {
        b.iter(|| {
            for &x in xs {
                black_box(hermite(black_box(5usize), black_box(x)));
            }
        });
    });

    group.bench_with_input(BenchmarkId::new("laguerre_n5", N), &unit, |b, xs| {
        b.iter(|| {
            for &x in xs {
                black_box(laguerre(black_box(5usize), black_box(x)));
            }
        });
    });

    group.bench_with_input(BenchmarkId::new("chebyshev_n10", N), &xs, |b, xs| {
        b.iter(|| {
            for &x in xs {
                black_box(chebyshev(black_box(10usize), black_box(x), true));
            }
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Hypergeometric function benchmarks
// ---------------------------------------------------------------------------

fn hypergeometric_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("hypergeometric");

    // Smaller N for hypergeometric — these are more expensive
    let n_hyp = 200usize;
    let xs_pos: Vec<f64> = (1..=n_hyp).map(|i| i as f64 * 0.01).collect();
    let xs_unit: Vec<f64> = (1..n_hyp).map(|i| i as f64 / n_hyp as f64 * 0.9).collect();

    group.bench_with_input(BenchmarkId::new("hyp1f1_a1_b2", n_hyp), &xs_pos, |b, xs| {
        b.iter(|| {
            for &x in xs {
                // hyp1f1 returns SpecialResult; ignore error path in bench
                let _ =
                    black_box(hyp1f1(black_box(1.0_f64), black_box(2.0_f64), black_box(x)).ok());
            }
        });
    });

    group.bench_with_input(
        BenchmarkId::new("hyp2f1_a1_b1_c2", n_hyp),
        &xs_unit,
        |b, xs| {
            b.iter(|| {
                for &x in xs {
                    let _ = black_box(
                        hyp2f1(
                            black_box(1.0_f64),
                            black_box(1.0_f64),
                            black_box(2.0_f64),
                            black_box(x),
                        )
                        .ok(),
                    );
                }
            });
        },
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// Elliptic integral benchmarks
// ---------------------------------------------------------------------------

fn elliptic_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("elliptic_integrals");

    let ms = make_elliptic_m(N);

    group.bench_with_input(BenchmarkId::new("ellipk", N), &ms, |b, ms| {
        b.iter(|| {
            for &m in ms {
                black_box(ellipk(black_box(m)));
            }
        });
    });

    group.bench_with_input(BenchmarkId::new("ellipe", N), &ms, |b, ms| {
        b.iter(|| {
            for &m in ms {
                black_box(ellipe(black_box(m)));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    gamma_bench,
    bessel_bench,
    legendre_bench,
    hypergeometric_bench,
    elliptic_bench,
);
criterion_main!(benches);
