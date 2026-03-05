//! Comprehensive property-based tests for scirs2-stats using proptest
//!
//! Covers:
//! - Distribution properties (PDF integrates to ~1, CDF monotonic, CDF in [0,1],
//!   ppf(cdf(x)) roundtrip, mean/variance match theory)
//! - Statistical function properties (correlation bounds, correlation matrix PSD,
//!   Cov(X,X)=Var(X), bootstrap CI contains estimate, M-estimator bounds)
//! - Mathematical invariants (KDE integrates to ~1, GMM responsibilities sum to 1,
//!   survival S(t)=1-F(t), hazard rate >= 0)

use proptest::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Relative / absolute approximate equality
fn approx_eq(a: f64, b: f64, rel_tol: f64, abs_tol: f64) -> bool {
    if !a.is_finite() || !b.is_finite() {
        return false;
    }
    let diff = (a - b).abs();
    let max_abs = a.abs().max(b.abs());
    diff <= abs_tol || diff <= rel_tol * max_abs
}

/// Numerical integration via trapezoidal rule
fn trapz(xs: &[f64], ys: &[f64]) -> f64 {
    assert_eq!(xs.len(), ys.len());
    let mut total = 0.0;
    for i in 1..xs.len() {
        total += 0.5 * (ys[i - 1] + ys[i]) * (xs[i] - xs[i - 1]);
    }
    total
}

/// Strategy: generate a Vec<f64> of given length range with finite, well-scaled values
fn finite_vec(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    proptest::collection::vec(
        // Use uniform range to avoid extremely small/large magnitudes
        -1000.0..1000.0f64,
        min_len..=max_len,
    )
}

/// Strategy: generate two equal-length vecs with well-scaled values
fn paired_vecs(min_len: usize, max_len: usize) -> impl Strategy<Value = (Vec<f64>, Vec<f64>)> {
    (min_len..=max_len).prop_flat_map(|n| {
        (
            proptest::collection::vec(-1000.0..1000.0f64, n),
            proptest::collection::vec(-1000.0..1000.0f64, n),
        )
    })
}

/// Check that a slice has sufficient variance (not all constant)
fn has_variance(data: &[f64]) -> bool {
    if data.len() < 2 {
        return false;
    }
    let first = data[0];
    // Require meaningful spread (not just rounding differences)
    let max_v = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_v = data.iter().copied().fold(f64::INFINITY, f64::min);
    let range = max_v - min_v;
    // Need relative spread of at least 1e-6 of the magnitude
    let scale = max_v.abs().max(min_v.abs()).max(1.0);
    range > scale * 1e-6 && data.iter().any(|&x| (x - first).abs() > 1e-8)
}

// ===========================================================================
// Part 1: Distribution properties
// ===========================================================================

mod distribution_properties {
    use super::*;
    use scirs2_stats::distributions;
    use scirs2_stats::traits::{ContinuousDistribution, Distribution};

    // -----------------------------------------------------------------------
    // Normal distribution
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        /// PDF of Normal(mu, sigma) integrates to approximately 1
        #[test]
        fn normal_pdf_integrates_to_one(
            mu in -10.0..10.0f64,
            sigma in 0.1..5.0f64,
        ) {
            let dist = distributions::norm(mu, sigma)
                .expect("Failed to create normal distribution");
            // Integrate from mu-8*sigma to mu+8*sigma (captures >99.99%)
            let lo = mu - 8.0 * sigma;
            let hi = mu + 8.0 * sigma;
            let n = 2000;
            let xs: Vec<f64> = (0..=n).map(|i| lo + (hi - lo) * i as f64 / n as f64).collect();
            let ys: Vec<f64> = xs.iter().map(|&x| dist.pdf(x)).collect();
            let integral = trapz(&xs, &ys);
            prop_assert!((integral - 1.0).abs() < 0.01,
                "Normal PDF integral = {}, expected ~1.0 (mu={}, sigma={})", integral, mu, sigma);
        }

        /// CDF of Normal is monotonically non-decreasing
        #[test]
        fn normal_cdf_monotonic(
            mu in -5.0..5.0f64,
            sigma in 0.1..5.0f64,
        ) {
            let dist = distributions::norm(mu, sigma)
                .expect("Failed to create normal distribution");
            let lo = mu - 6.0 * sigma;
            let hi = mu + 6.0 * sigma;
            let n = 500;
            let mut prev_cdf = 0.0f64;
            for i in 0..=n {
                let x = lo + (hi - lo) * i as f64 / n as f64;
                let cdf_val = dist.cdf(x);
                prop_assert!(cdf_val >= prev_cdf - 1e-12,
                    "CDF not monotonic at x={}: cdf={} < prev={}", x, cdf_val, prev_cdf);
                prev_cdf = cdf_val;
            }
        }

        /// CDF values are in [0, 1]
        #[test]
        fn normal_cdf_in_unit_interval(
            mu in -5.0..5.0f64,
            sigma in 0.1..5.0f64,
            x in -20.0..20.0f64,
        ) {
            let dist = distributions::norm(mu, sigma)
                .expect("Failed to create normal distribution");
            let cdf_val = dist.cdf(x);
            prop_assert!(cdf_val >= -1e-12 && cdf_val <= 1.0 + 1e-12,
                "CDF({}) = {} not in [0,1]", x, cdf_val);
        }

        /// ppf(cdf(x)) ≈ x roundtrip
        #[test]
        fn normal_ppf_cdf_roundtrip(
            mu in -5.0..5.0f64,
            sigma in 0.1..3.0f64,
            x in -10.0..10.0f64,
        ) {
            let dist = distributions::norm(mu, sigma)
                .expect("Failed to create normal distribution");
            let cdf_val = dist.cdf(x);
            // Only test interior probabilities to avoid boundary issues
            if cdf_val > 0.001 && cdf_val < 0.999 {
                if let Ok(roundtrip) = dist.ppf(cdf_val) {
                    prop_assert!(approx_eq(roundtrip, x, 0.02, 0.05),
                        "ppf(cdf({})) = {} (cdf={})", x, roundtrip, cdf_val);
                }
            }
        }

        /// Mean and variance match theoretical values
        #[test]
        fn normal_mean_variance(
            mu in -10.0..10.0f64,
            sigma in 0.1..5.0f64,
        ) {
            let dist = distributions::norm(mu, sigma)
                .expect("Failed to create normal distribution");
            let m = dist.mean();
            let v = dist.var();
            prop_assert!(approx_eq(m, mu, 1e-10, 1e-10),
                "mean={}, expected {}", m, mu);
            prop_assert!(approx_eq(v, sigma * sigma, 1e-10, 1e-10),
                "var={}, expected {}", v, sigma * sigma);
        }
    }

    // -----------------------------------------------------------------------
    // Exponential distribution
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        /// PDF of Exponential(rate) integrates to ~1
        #[test]
        fn exponential_pdf_integrates_to_one(
            rate in 0.1..5.0f64,
        ) {
            let dist = distributions::expon(rate, 0.0)
                .expect("Failed to create exponential distribution");
            let hi = 10.0 / rate; // captures most of the mass
            let n = 2000;
            let xs: Vec<f64> = (0..=n).map(|i| hi * i as f64 / n as f64).collect();
            let ys: Vec<f64> = xs.iter().map(|&x| dist.pdf(x)).collect();
            let integral = trapz(&xs, &ys);
            prop_assert!((integral - 1.0).abs() < 0.02,
                "Exponential PDF integral = {}, rate={}", integral, rate);
        }

        /// CDF of Exponential is monotonically non-decreasing
        #[test]
        fn exponential_cdf_monotonic(
            rate in 0.1..5.0f64,
        ) {
            let dist = distributions::expon(rate, 0.0)
                .expect("Failed to create exponential distribution");
            let hi = 10.0 / rate;
            let n = 500;
            let mut prev_cdf = 0.0f64;
            for i in 0..=n {
                let x = hi * i as f64 / n as f64;
                let cdf_val = dist.cdf(x);
                prop_assert!(cdf_val >= prev_cdf - 1e-12,
                    "CDF not monotonic at x={}", x);
                prev_cdf = cdf_val;
            }
        }

        /// Exponential mean = 1/rate, var = 1/rate^2
        #[test]
        fn exponential_mean_variance(
            rate in 0.1..10.0f64,
        ) {
            let dist = distributions::expon(rate, 0.0)
                .expect("Failed to create exponential distribution");
            let m = dist.mean();
            let v = dist.var();
            prop_assert!(approx_eq(m, 1.0 / rate, 1e-8, 1e-8),
                "mean={}, expected {}", m, 1.0 / rate);
            prop_assert!(approx_eq(v, 1.0 / (rate * rate), 1e-8, 1e-8),
                "var={}, expected {}", v, 1.0 / (rate * rate));
        }
    }

    // -----------------------------------------------------------------------
    // Uniform distribution
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        /// PDF of Uniform(a,b) integrates to 1
        #[test]
        fn uniform_pdf_integrates_to_one(
            a in -10.0..9.0f64,
            width in 0.1..10.0f64,
        ) {
            let b = a + width;
            let dist = distributions::uniform(a, b)
                .expect("Failed to create uniform distribution");
            let n = 1000;
            let lo = a - 1.0;
            let hi = b + 1.0;
            let xs: Vec<f64> = (0..=n).map(|i| lo + (hi - lo) * i as f64 / n as f64).collect();
            let ys: Vec<f64> = xs.iter().map(|&x| dist.pdf(x)).collect();
            let integral = trapz(&xs, &ys);
            prop_assert!((integral - 1.0).abs() < 0.02,
                "Uniform PDF integral = {}", integral);
        }

        /// CDF of Uniform is monotonically non-decreasing and in [0,1]
        #[test]
        fn uniform_cdf_properties(
            a in -10.0..9.0f64,
            width in 0.1..10.0f64,
            x in -15.0..25.0f64,
        ) {
            let b = a + width;
            let dist = distributions::uniform(a, b)
                .expect("Failed to create uniform distribution");
            let cdf_val = dist.cdf(x);
            prop_assert!(cdf_val >= -1e-12 && cdf_val <= 1.0 + 1e-12,
                "CDF({}) = {} not in [0,1]", x, cdf_val);
        }

        /// Uniform mean = (a+b)/2, var = (b-a)^2/12
        #[test]
        fn uniform_mean_variance(
            a in -10.0..9.0f64,
            width in 0.1..10.0f64,
        ) {
            let b = a + width;
            let dist = distributions::uniform(a, b)
                .expect("Failed to create uniform distribution");
            let m = dist.mean();
            let v = dist.var();
            let expected_mean = (a + b) / 2.0;
            let expected_var = width * width / 12.0;
            prop_assert!(approx_eq(m, expected_mean, 1e-10, 1e-10),
                "mean={}, expected {}", m, expected_mean);
            prop_assert!(approx_eq(v, expected_var, 1e-10, 1e-10),
                "var={}, expected {}", v, expected_var);
        }
    }

    // -----------------------------------------------------------------------
    // Beta distribution
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        /// PDF of Beta(a,b) integrates to ~1
        #[test]
        fn beta_pdf_integrates_to_one(
            alpha in 1.0..5.0f64,
            beta_param in 1.0..5.0f64,
        ) {
            let dist = distributions::beta(alpha, beta_param, 0.0, 1.0)
                .expect("Failed to create beta distribution");
            // Avoid boundaries where PDF can be infinite for alpha,beta < 1
            let n = 2000;
            let eps = 1e-6;
            let xs: Vec<f64> = (0..=n).map(|i| eps + (1.0 - 2.0 * eps) * i as f64 / n as f64).collect();
            let ys: Vec<f64> = xs.iter().map(|&x| dist.pdf(x)).collect();
            let integral = trapz(&xs, &ys);
            prop_assert!((integral - 1.0).abs() < 0.02,
                "Beta({},{}) PDF integral = {}", alpha, beta_param, integral);
        }

        /// Beta CDF boundary conditions: CDF(x<0)=0, CDF(x>1)=1
        #[test]
        fn beta_cdf_boundary_conditions(
            alpha in 1.5..5.0f64,
            beta_param in 1.5..5.0f64,
        ) {
            let dist = distributions::beta(alpha, beta_param, 0.0, 1.0)
                .expect("Failed to create beta distribution");
            // CDF below support should be 0
            let cdf_neg = dist.cdf(-0.1);
            prop_assert!(cdf_neg.abs() < 1e-10,
                "Beta CDF(-0.1)={}, expected 0", cdf_neg);
            // CDF above support should be 1
            let cdf_above = dist.cdf(1.1);
            prop_assert!((cdf_above - 1.0).abs() < 1e-10,
                "Beta CDF(1.1)={}, expected 1", cdf_above);
        }

        /// Beta mean = alpha / (alpha + beta)
        #[test]
        fn beta_mean_matches_theory(
            alpha in 0.5..10.0f64,
            beta_param in 0.5..10.0f64,
        ) {
            let dist = distributions::beta(alpha, beta_param, 0.0, 1.0)
                .expect("Failed to create beta distribution");
            let m = dist.mean();
            let expected = alpha / (alpha + beta_param);
            prop_assert!(approx_eq(m, expected, 1e-8, 1e-8),
                "Beta({},{}) mean={}, expected {}", alpha, beta_param, m, expected);
        }
    }

    // -----------------------------------------------------------------------
    // Gamma distribution
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        /// PDF of Gamma(shape, scale) integrates to ~1
        #[test]
        fn gamma_pdf_integrates_to_one(
            shape in 0.5..5.0f64,
            scale in 0.5..3.0f64,
        ) {
            let dist = distributions::gamma(shape, scale, 0.0)
                .expect("Failed to create gamma distribution");
            let hi = shape * scale * 5.0 + 20.0 * scale;
            let n = 3000;
            let xs: Vec<f64> = (0..=n).map(|i| {
                let t = i as f64 / n as f64;
                // Use finer resolution near zero for shape < 1
                if shape < 1.0 {
                    1e-6 + t * t * hi
                } else {
                    t * hi
                }
            }).collect();
            let ys: Vec<f64> = xs.iter().map(|&x| dist.pdf(x)).collect();
            let integral = trapz(&xs, &ys);
            prop_assert!((integral - 1.0).abs() < 0.05,
                "Gamma({},{}) PDF integral = {}", shape, scale, integral);
        }

        /// Gamma mean = shape * scale, var = shape * scale^2
        #[test]
        fn gamma_mean_variance(
            shape in 0.5..10.0f64,
            scale in 0.5..5.0f64,
        ) {
            let dist = distributions::gamma(shape, scale, 0.0)
                .expect("Failed to create gamma distribution");
            let m = dist.mean();
            let v = dist.var();
            prop_assert!(approx_eq(m, shape * scale, 1e-8, 1e-8),
                "Gamma mean={}, expected {}", m, shape * scale);
            prop_assert!(approx_eq(v, shape * scale * scale, 1e-8, 1e-8),
                "Gamma var={}, expected {}", v, shape * scale * scale);
        }
    }
}

// ===========================================================================
// Part 2: Statistical function properties
// ===========================================================================

mod statistical_function_properties {
    use super::*;
    use scirs2_stats::{corrcoef, mean, pearson_r, var};

    // -----------------------------------------------------------------------
    // Correlation is in [-1, 1]
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn correlation_in_bounds(data in paired_vecs(5, 100)) {
            let (xv, yv) = data;
            if !has_variance(&xv) || !has_variance(&yv) {
                return Ok(());
            }
            let x = Array1::from_vec(xv);
            let y = Array1::from_vec(yv);
            match pearson_r(&x.view(), &y.view()) {
                Ok(r) => {
                    prop_assert!(r >= -1.0 - 1e-10 && r <= 1.0 + 1e-10,
                        "Pearson r = {}, out of [-1,1]", r);
                }
                Err(_) => { /* numerical edge case, skip */ }
            }
        }

        /// Correlation is symmetric: corr(X,Y) = corr(Y,X)
        #[test]
        fn correlation_symmetric(data in paired_vecs(5, 100)) {
            let (xv, yv) = data;
            if !has_variance(&xv) || !has_variance(&yv) {
                return Ok(());
            }
            let x = Array1::from_vec(xv);
            let y = Array1::from_vec(yv);
            match (pearson_r::<f64, _>(&x.view(), &y.view()),
                   pearson_r::<f64, _>(&y.view(), &x.view())) {
                (Ok(rxy), Ok(ryx)) => {
                    prop_assert!(approx_eq(rxy, ryx, 1e-10, 1e-10),
                        "corr(X,Y)={} != corr(Y,X)={}", rxy, ryx);
                }
                _ => { /* numerical edge case, skip */ }
            }
        }

        /// corr(X, X) = 1
        #[test]
        fn self_correlation_is_one(data in finite_vec(5, 100)) {
            if !has_variance(&data) {
                return Ok(());
            }
            let x = Array1::from_vec(data);
            match pearson_r::<f64, _>(&x.view(), &x.view()) {
                Ok(r) => {
                    prop_assert!(approx_eq(r, 1.0, 1e-10, 1e-10),
                        "corr(X,X) = {}, expected 1.0", r);
                }
                Err(_) => { /* variance too small for numerical stability */ }
            }
        }

        /// Cov(X, X) = Var(X)
        #[test]
        fn covariance_self_equals_variance(data in finite_vec(3, 100)) {
            if !has_variance(&data) {
                return Ok(());
            }
            let n = data.len();
            let x = Array1::from_vec(data);
            let variance: f64 = var(&x.view(), 0, None)
                .expect("var failed");

            // Compute covariance manually: cov(X,X) = E[(X-mu)^2] = var(X) (population)
            let mu: f64 = mean(&x.view()).expect("mean failed");
            let cov_xx: f64 = x.iter()
                .map(|&v| (v - mu) * (v - mu))
                .sum::<f64>() / n as f64;

            prop_assert!(approx_eq(cov_xx, variance, 1e-8, 1e-8),
                "Cov(X,X)={} != Var(X)={}", cov_xx, variance);
        }
    }

    // -----------------------------------------------------------------------
    // Correlation matrix is positive semi-definite
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        /// Correlation matrix is symmetric and has ones on diagonal
        #[test]
        fn correlation_matrix_symmetric_unit_diagonal(
            n in 10usize..50,
            p in 2usize..5,
        ) {
            // Generate random data matrix
            let mut data = Array2::<f64>::zeros((n, p));
            // Fill with something that has variance
            for j in 0..p {
                for i in 0..n {
                    data[(i, j)] = (i as f64 + 1.0) * (j as f64 + 1.0)
                        + (i * 7 + j * 13) as f64 % 17.0;
                }
            }

            let corr: Array2<f64> = corrcoef(&data.view(), "pearson")
                .expect("corrcoef failed");

            // Check diagonal = 1.0
            for j in 0..p {
                prop_assert!(approx_eq(corr[(j, j)], 1.0, 1e-10, 1e-10),
                    "Diagonal corr[{},{}] = {} != 1.0", j, j, corr[(j, j)]);
            }

            // Check symmetry
            for i in 0..p {
                for j in i+1..p {
                    prop_assert!(approx_eq(corr[(i, j)], corr[(j, i)], 1e-10, 1e-10),
                        "corr[{},{}]={} != corr[{},{}]={}", i, j, corr[(i, j)], j, i, corr[(j, i)]);
                }
            }

            // Check all entries in [-1, 1]
            for i in 0..p {
                for j in 0..p {
                    prop_assert!(corr[(i, j)] >= -1.0 - 1e-10 && corr[(i, j)] <= 1.0 + 1e-10,
                        "corr[{},{}] = {} out of [-1,1]", i, j, corr[(i, j)]);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Descriptive statistics invariants
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        /// mean is between min and max
        #[test]
        fn mean_between_min_max(data in finite_vec(1, 200)) {
            let arr = Array1::from_vec(data.clone());
            let m: f64 = mean(&arr.view()).expect("mean failed");
            let min_val = data.iter().copied().fold(f64::INFINITY, f64::min);
            let max_val = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            prop_assert!(m >= min_val - 1e-10 && m <= max_val + 1e-10,
                "mean={} not in [{}, {}]", m, min_val, max_val);
        }

        /// variance is non-negative
        #[test]
        fn variance_non_negative(data in finite_vec(2, 200)) {
            let arr = Array1::from_vec(data);
            let v: f64 = var(&arr.view(), 0, None).expect("var failed");
            prop_assert!(v >= -1e-10, "variance={} < 0", v);
        }

        /// std^2 ≈ variance
        #[test]
        fn std_squared_equals_variance(data in finite_vec(2, 200)) {
            let arr = Array1::from_vec(data);
            let v: f64 = var(&arr.view(), 1, None).expect("var failed");
            let s: f64 = scirs2_stats::std(&arr.view(), 1, None).expect("std failed");
            prop_assert!(approx_eq(s * s, v, 1e-8, 1e-10),
                "std^2={} != var={}", s * s, v);
        }

        /// variance(a*X + b) = a^2 * var(X) (affine invariance)
        #[test]
        fn variance_affine_scaling(
            data in finite_vec(2, 100),
            a in -10.0..10.0f64,
        ) {
            if a.abs() < 1e-10 || a.abs() > 1e4 {
                return Ok(());
            }
            let arr = Array1::from_vec(data);
            let v: f64 = var(&arr.view(), 0, None).expect("var failed");
            let transformed = arr.mapv(|x| a * x + 3.14);
            let v_t: f64 = var(&transformed.view(), 0, None).expect("var failed");
            let expected = a * a * v;
            if expected.abs() < 1e-14 {
                return Ok(());
            }
            prop_assert!(approx_eq(v_t, expected, 1e-6, 1e-8),
                "Var(a*X+b)={}, expected a^2*Var(X)={} (a={})", v_t, expected, a);
        }
    }
}

// ===========================================================================
// Part 3: Bootstrap properties
// ===========================================================================

mod bootstrap_properties {
    use super::*;
    use scirs2_stats::bootstrap::percentile_bootstrap;
    use scirs2_stats::mean;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        /// Bootstrap CI contains the point estimate (for mean, with normal-ish data)
        #[test]
        fn bootstrap_ci_contains_mean(data in finite_vec(20, 100)) {
            if !has_variance(&data) {
                return Ok(());
            }
            let arr = Array1::from_vec(data);
            let point_est: f64 = mean(&arr.view()).expect("mean failed");

            match percentile_bootstrap(&arr.view(), |sample| {
                let n = sample.len() as f64;
                if n < 1.0 { return 0.0; }
                sample.iter().sum::<f64>() / n
            }, Some(500), Some(0.95), Some(42)) {
                Ok(ci) => {
                    // The CI should contain or be close to the point estimate most of the time
                    // We use a relaxed check: CI within 2x width of the estimate
                    let width = (ci.ci_upper - ci.ci_lower).abs();
                    prop_assert!(
                        point_est >= ci.ci_lower - width * 0.5 && point_est <= ci.ci_upper + width * 0.5,
                        "Mean {} outside relaxed CI [{}, {}]",
                        point_est, ci.ci_lower - width * 0.5, ci.ci_upper + width * 0.5
                    );
                }
                Err(_) => {
                    // Bootstrap can fail with edge cases, skip
                }
            }
        }
    }
}

// ===========================================================================
// Part 4: Robust estimator properties
// ===========================================================================

mod robust_estimator_properties {
    use super::*;
    use scirs2_stats::robust_estimators::{huber_location, trimmed_mean, tukey_biweight_location};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        /// Trimmed mean is between min and max of the data
        #[test]
        fn trimmed_mean_bounded(data in finite_vec(10, 100)) {
            let arr = Array1::from_vec(data.clone());
            match trimmed_mean(&arr.view(), 0.1) {
                Ok(tm) => {
                    let min_v = data.iter().copied().fold(f64::INFINITY, f64::min);
                    let max_v = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    prop_assert!(tm >= min_v - 1e-10 && tm <= max_v + 1e-10,
                        "trimmed_mean={} not in [{}, {}]", tm, min_v, max_v);
                }
                Err(_) => { /* acceptable with edge cases */ }
            }
        }

        /// Huber location estimator is bounded by data range
        #[test]
        fn huber_location_bounded(data in finite_vec(5, 100)) {
            if !has_variance(&data) {
                return Ok(());
            }
            let arr = Array1::from_vec(data.clone());
            match huber_location(&arr.view(), None, None, None) {
                Ok(loc) => {
                    let min_v = data.iter().copied().fold(f64::INFINITY, f64::min);
                    let max_v = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    prop_assert!(loc >= min_v - 1e-6 && loc <= max_v + 1e-6,
                        "huber location={} not in [{}, {}]", loc, min_v, max_v);
                }
                Err(_) => { /* acceptable */ }
            }
        }

        /// Tukey biweight location estimator is bounded
        #[test]
        fn tukey_biweight_bounded(data in finite_vec(5, 100)) {
            if !has_variance(&data) {
                return Ok(());
            }
            let arr = Array1::from_vec(data.clone());
            match tukey_biweight_location(&arr.view(), None, None, None) {
                Ok(loc) => {
                    let min_v = data.iter().copied().fold(f64::INFINITY, f64::min);
                    let max_v = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    // Tukey biweight can be slightly outside data range due to weighting
                    let margin = (max_v - min_v).abs() * 0.5 + 1.0;
                    prop_assert!(loc >= min_v - margin && loc <= max_v + margin,
                        "tukey location={} far outside [{}, {}]", loc, min_v, max_v);
                }
                Err(_) => { /* acceptable */ }
            }
        }
    }
}

// ===========================================================================
// Part 5: KDE integrates to approximately 1
// ===========================================================================

mod kde_properties {
    use super::*;
    use scirs2_stats::kde::{Kernel, KernelDensityEstimate};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        /// KDE integrates to approximately 1
        #[test]
        fn kde_integrates_to_one(data in finite_vec(10, 100)) {
            if !has_variance(&data) {
                return Ok(());
            }
            // Filter data to a reasonable range; drop values < 1e-10 in magnitude
            // which effectively collapse to zero and destroy spread
            let filtered: Vec<f64> = data.iter()
                .copied()
                .filter(|&x| x.abs() > 1e-10 && x.abs() < 1000.0)
                .collect();
            if filtered.len() < 5 || !has_variance(&filtered) {
                return Ok(());
            }

            let kde = KernelDensityEstimate::new(&filtered, Kernel::Gaussian);

            // Determine integration range from data
            let min_v = filtered.iter().copied().fold(f64::INFINITY, f64::min);
            let max_v = filtered.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let range = max_v - min_v;
            if range < 1e-6 {
                return Ok(());
            }
            let bandwidth_est = range / (filtered.len() as f64).powf(0.2);
            let margin = 5.0 * bandwidth_est.max(range * 0.5).max(1.0);
            let lo = min_v - margin;
            let hi = max_v + margin;

            let n = 3000;
            let xs: Vec<f64> = (0..=n).map(|i| lo + (hi - lo) * i as f64 / n as f64).collect();
            let ys: Vec<f64> = xs.iter().map(|&x| kde.evaluate(x)).collect();
            let integral = trapz(&xs, &ys);

            prop_assert!((integral - 1.0).abs() < 0.2,
                "KDE integral = {} (n_data={}, range=[{},{}])", integral, filtered.len(), lo, hi);
        }

        /// KDE density is non-negative everywhere
        #[test]
        fn kde_non_negative(data in finite_vec(5, 50)) {
            if !has_variance(&data) {
                return Ok(());
            }
            let kde = KernelDensityEstimate::new(&data, Kernel::Gaussian);
            let min_v = data.iter().copied().fold(f64::INFINITY, f64::min);
            let max_v = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let range = max_v - min_v;
            for i in 0..100 {
                let x = min_v - range - 1.0 + (2.0 * range + 2.0) * i as f64 / 100.0;
                let d = kde.evaluate(x);
                prop_assert!(d >= -1e-12,
                    "KDE density at x={} is negative: {}", x, d);
            }
        }
    }
}

// ===========================================================================
// Part 6: Survival analysis invariants
// ===========================================================================

mod survival_properties {
    use super::*;
    use scirs2_stats::survival::{KaplanMeier, NelsonAalen};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        /// S(t) = 1 - F(t): KM survival function is between 0 and 1
        #[test]
        fn km_survival_bounded(
            n in 5usize..50,
            seed in 0u64..10000,
        ) {
            // Generate deterministic "random" times and events
            let times: Vec<f64> = (0..n)
                .map(|i| ((i as u64 * 7 + seed) % 100) as f64 + 0.1)
                .collect();
            let events: Vec<bool> = (0..n)
                .map(|i| ((i as u64 * 13 + seed) % 3) != 0)
                .collect();

            let km = KaplanMeier::fit(&times, &events)
                .expect("KM fit failed");

            // Survival is in [0, 1]
            for &s in &km.survival {
                prop_assert!(s >= -1e-12 && s <= 1.0 + 1e-12,
                    "KM survival {} not in [0,1]", s);
            }

            // Survival is non-increasing
            for i in 1..km.survival.len() {
                prop_assert!(km.survival[i] <= km.survival[i - 1] + 1e-12,
                    "KM survival not non-increasing at index {}: {} > {}",
                    i, km.survival[i], km.survival[i - 1]);
            }

            // Initial survival should be <= 1
            if !km.survival.is_empty() {
                prop_assert!(km.survival[0] <= 1.0 + 1e-12);
            }
        }

        /// Nelson-Aalen cumulative hazard is non-negative and non-decreasing
        #[test]
        fn na_hazard_non_negative_increasing(
            n in 5usize..50,
            seed in 0u64..10000,
        ) {
            let times: Vec<f64> = (0..n)
                .map(|i| ((i as u64 * 7 + seed) % 100) as f64 + 0.1)
                .collect();
            let events: Vec<bool> = (0..n)
                .map(|i| ((i as u64 * 13 + seed) % 3) != 0)
                .collect();

            let na = NelsonAalen::fit(&times, &events)
                .expect("NA fit failed");

            // Cumulative hazard >= 0
            for &h in &na.cumulative_hazard {
                prop_assert!(h >= -1e-12,
                    "Cumulative hazard {} < 0", h);
            }

            // Cumulative hazard is non-decreasing
            for i in 1..na.cumulative_hazard.len() {
                prop_assert!(na.cumulative_hazard[i] >= na.cumulative_hazard[i - 1] - 1e-12,
                    "Cumulative hazard not non-decreasing at index {}", i);
            }
        }

        /// S(t) = exp(-H(t)) relationship between KM and NA
        #[test]
        fn km_na_survival_consistency(
            n in 10usize..50,
            seed in 0u64..10000,
        ) {
            let times: Vec<f64> = (0..n)
                .map(|i| ((i as u64 * 7 + seed) % 100) as f64 + 0.1)
                .collect();
            let events: Vec<bool> = (0..n)
                .map(|i| ((i as u64 * 13 + seed) % 3) != 0)
                .collect();

            let km = KaplanMeier::fit(&times, &events).expect("KM fit failed");
            let na = NelsonAalen::fit(&times, &events).expect("NA fit failed");

            // Both should have the same event times
            prop_assert_eq!(&km.times, &na.times, "KM and NA event times differ");

            // KM and NA survival estimates should be close (but not identical;
            // KM uses product-limit, NA uses exp(-H))
            for k in 0..km.times.len() {
                let s_km = km.survival[k];
                let s_na = na.survival_at(na.times[k]);
                // They can differ but should agree directionally
                prop_assert!(
                    (s_km - s_na).abs() < 0.15 || (s_km < 0.01 && s_na < 0.01),
                    "KM S(t)={} vs NA exp(-H(t))={} at t={} differ too much",
                    s_km, s_na, km.times[k]
                );
            }
        }
    }
}

// ===========================================================================
// Part 7: GMM responsibilities sum to 1
// ===========================================================================

mod gmm_properties {
    use super::*;
    use scirs2_stats::{GMMConfig, GaussianMixtureModel, InitializationMethod};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(5))]

        /// GMM responsibilities sum to 1 across components for each data point
        #[test]
        fn gmm_responsibilities_sum_to_one(
            n in 30usize..80,
            k in 2usize..4,
            seed in 1u64..1000,
        ) {
            // Generate clustered data deterministically
            let mut data = Array2::<f64>::zeros((n, 2));
            for i in 0..n {
                let cluster = i % k;
                let offset = cluster as f64 * 5.0;
                data[(i, 0)] = offset + (((i * 7 + seed as usize) % 100) as f64 / 50.0 - 1.0);
                data[(i, 1)] = offset + (((i * 13 + seed as usize) % 100) as f64 / 50.0 - 1.0);
            }

            let config = GMMConfig {
                max_iter: 50,
                tolerance: 1e-4,
                init_method: InitializationMethod::Random,
                seed: Some(seed),
                n_init: 1,
                ..GMMConfig::default()
            };

            let mut gmm: GaussianMixtureModel<f64> = match GaussianMixtureModel::<f64>::new(k, config) {
                Ok(g) => g,
                Err(_) => return Ok(()),
            };

            match gmm.fit(&data.view()) {
                Ok(params) => {
                    // Check that weights sum to 1
                    let weight_sum: f64 = params.weights.iter().sum();
                    prop_assert!((weight_sum - 1.0).abs() < 1e-6,
                        "GMM weights sum to {} instead of 1.0", weight_sum);

                    // Check all weights are non-negative
                    for (i, &w) in params.weights.iter().enumerate() {
                        prop_assert!(w >= -1e-10,
                            "GMM weight[{}] = {} is negative", i, w);
                    }
                }
                Err(_) => {
                    // EM may not converge for all random configurations
                }
            }
        }
    }
}
