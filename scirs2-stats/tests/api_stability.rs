// API Stability Tests for scirs2-stats
//
// These tests verify that the core public API surface has not been accidentally
// broken. If this file fails to compile, a previously-stable public item has
// been removed or its signature changed in a backward-incompatible way.
//
// Guidelines:
// - Each test function covers one logical group of public items.
// - No `unwrap()` — use `expect("…")` for any fallible calls.
// - Keep each test focused on one logical group of exports.

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Verify that the error module types are accessible.
#[test]
fn test_error_types_accessible() {
    use scirs2_stats::error::{StatsError, StatsResult};

    fn _returns_stats_result() -> StatsResult<f64> {
        Ok(0.0)
    }
    let v = _returns_stats_result().expect("should succeed");
    assert!((v - 0.0).abs() < 1e-14);

    // StatsError can be constructed
    let _err = StatsError::InvalidInput("test".to_string());
}

// ---------------------------------------------------------------------------
// Descriptive statistics
// ---------------------------------------------------------------------------

/// Verify that core descriptive functions are accessible.
#[test]
fn test_descriptive_stats_accessible() {
    use scirs2_core::ndarray::array;
    use scirs2_stats::{kurtosis, mean, median, skew, std, var};

    let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0];

    let m = mean(&data.view()).expect("mean should succeed");
    assert!((m - 3.0).abs() < 1e-10);

    let med = median(&data.view()).expect("median should succeed");
    assert!((med - 3.0).abs() < 1e-10);

    let v = var(&data.view(), 1, None).expect("var should succeed (ddof=1)");
    assert!(v > 0.0);

    let s = std(&data.view(), 1, None).expect("std should succeed (ddof=1)");
    assert!(s > 0.0);

    let sk = skew(&data.view(), false, None).expect("skew should succeed");
    // Uniform 1..5 is symmetric, so skew ≈ 0
    assert!(sk.abs() < 1e-10);

    let kurt = kurtosis(&data.view(), true, false, None).expect("kurtosis should succeed");
    let _ = kurt; // value varies; we just verify the call compiles and executes
}

// ---------------------------------------------------------------------------
// Distributions
// ---------------------------------------------------------------------------

/// Verify that canonical distribution factory functions are accessible.
#[test]
fn test_distributions_accessible() {
    use scirs2_stats::distributions;
    use scirs2_stats::Distribution;

    let norm = distributions::norm(0.0f64, 1.0).expect("norm distribution should construct");
    let pdf = norm.pdf(0.0);
    assert!(pdf > 0.0);
    let cdf = norm.cdf(0.0);
    assert!((cdf - 0.5).abs() < 1e-6);

    let pois = distributions::poisson(3.0f64, 0.0).expect("poisson distribution should construct");
    let pmf = pois.pmf(3.0);
    assert!(pmf > 0.0);

    let gam = distributions::gamma(2.0f64, 1.0, 0.0).expect("gamma distribution should construct");
    let _ = gam.pdf(1.0);

    let bet =
        distributions::beta(2.0f64, 3.0, 0.0, 1.0).expect("beta distribution should construct");
    let _ = bet.pdf(0.5);
}

// ---------------------------------------------------------------------------
// Hypothesis tests
// ---------------------------------------------------------------------------

/// Verify that t-test functions and Alternative enum are accessible.
#[test]
fn test_ttest_accessible() {
    use scirs2_core::ndarray::array;
    use scirs2_stats::tests::ttest::Alternative;
    use scirs2_stats::{ttest_1samp, ttest_ind};

    let data = array![5.1f64, 4.9, 6.2, 5.7, 5.5, 5.1, 5.2, 5.0, 5.3, 5.4];

    let result = ttest_1samp(&data.view(), 5.0, Alternative::TwoSided, "propagate")
        .expect("ttest_1samp should succeed");
    let _ = result.statistic;
    let _ = result.pvalue;

    let g1 = array![5.1f64, 4.9, 6.2, 5.7, 5.5];
    let g2 = array![4.8f64, 5.2, 5.1, 4.7, 4.9];
    let result2 = ttest_ind(
        &g1.view(),
        &g2.view(),
        true,
        Alternative::TwoSided,
        "propagate",
    )
    .expect("ttest_ind should succeed");
    let _ = result2.pvalue;

    // Verify all Alternative variants exist
    let _two_sided = Alternative::TwoSided;
    let _greater = Alternative::Greater;
    let _less = Alternative::Less;
}

/// Verify non-parametric tests are accessible.
#[test]
fn test_nonparametric_tests_accessible() {
    use scirs2_core::ndarray::array;
    use scirs2_stats::{friedman, kruskal_wallis, mann_whitney, wilcoxon};

    let g1 = array![5.1f64, 4.9, 6.2, 5.7, 5.5];
    let g2 = array![4.8f64, 5.2, 5.1, 4.7, 4.9];

    let (u, p) = mann_whitney(&g1.view(), &g2.view(), "two-sided", true)
        .expect("mann_whitney should succeed");
    assert!(u >= 0.0);
    assert!((0.0..=1.0).contains(&p));

    let before = array![125.0f64, 115.0, 130.0, 140.0, 140.0];
    let after = array![110.0f64, 122.0, 125.0, 120.0, 140.0];
    let (w, pw) =
        wilcoxon(&before.view(), &after.view(), "wilcox", true).expect("wilcoxon should succeed");
    let _ = (w, pw);

    let g3 = array![5.0f64, 6.0, 7.0];
    let (h, pk) =
        kruskal_wallis(&[g1.view(), g2.view(), g3.view()]).expect("kruskal_wallis should succeed");
    let _ = (h, pk);

    // friedman requires 3+ groups as a 2-D array
    use scirs2_core::ndarray::array as ndarray_array;
    let blocks = ndarray_array![[5.0f64, 6.0], [4.0, 7.0], [6.0, 5.0]];
    let (fstat, pf) = friedman(&blocks.view()).expect("friedman should succeed");
    let _ = (fstat, pf);
}

/// Verify normality tests are accessible.
#[test]
fn test_normality_tests_accessible() {
    use scirs2_core::ndarray::array;
    use scirs2_stats::{anderson_darling, dagostino_k2, shapiro_wilk};

    let data = array![
        5.1f64, 4.9, 6.2, 5.7, 5.5, 5.1, 5.2, 5.0, 5.3, 5.4, 5.6, 5.8, 5.9, 6.0, 5.2, 5.4, 5.3,
        5.1, 5.2, 5.0
    ];

    let (w, p) = shapiro_wilk(&data.view()).expect("shapiro_wilk should succeed");
    assert!(w > 0.0 && w <= 1.0);
    assert!((0.0..=1.0).contains(&p));

    let (a2, pa) = anderson_darling(&data.view()).expect("anderson_darling should succeed");
    assert!(a2 >= 0.0);
    let _ = pa;

    let (k2, pk) = dagostino_k2(&data.view()).expect("dagostino_k2 should succeed");
    assert!(k2 >= 0.0);
    let _ = pk;
}

// ---------------------------------------------------------------------------
// Correlation functions
// ---------------------------------------------------------------------------

/// Verify that pearsonr, spearmanr, corrcoef are accessible.
#[test]
fn test_correlation_accessible() {
    use scirs2_core::ndarray::array;
    use scirs2_stats::{corrcoef, pearsonr, spearmanr};

    let x = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let y = array![5.0f64, 4.0, 3.0, 2.0, 1.0];

    let (r, p) = pearsonr(&x.view(), &y.view(), "two-sided").expect("pearsonr should succeed");
    assert!((r - (-1.0)).abs() < 1e-6);
    assert!(p >= 0.0);

    let (rho, prho) =
        spearmanr(&x.view(), &y.view(), "two-sided").expect("spearmanr should succeed");
    let _ = (rho, prho);

    let data = array![[1.0f64, 5.0], [2.0, 4.0], [3.0, 3.0]];
    let cm = corrcoef(&data.view(), "pearson").expect("corrcoef should succeed");
    assert_eq!(cm.shape(), &[2, 2]);
}

// ---------------------------------------------------------------------------
// Regression
// ---------------------------------------------------------------------------

/// Verify that linear regression is accessible.
#[test]
fn test_regression_accessible() {
    use scirs2_core::ndarray::array;
    use scirs2_stats::regression::linear_regression;

    let x = array![[1.0f64], [2.0], [3.0], [4.0], [5.0]];
    let y = array![2.1f64, 4.0, 5.9, 8.1, 10.0];

    let result =
        linear_regression(&x.view(), &y.view(), None).expect("linear_regression should succeed");
    // Slope should be approximately 2.0
    assert!((result.coefficients[0] - 2.0).abs() < 0.1);
}

// ---------------------------------------------------------------------------
// Distribution trait
// ---------------------------------------------------------------------------

/// Verify that the Distribution trait and key trait methods exist.
#[test]
fn test_distribution_trait_accessible() {
    use scirs2_stats::distributions;
    use scirs2_stats::Distribution;

    let norm = distributions::norm(0.0f64, 1.0).expect("norm distribution should construct");

    // Core trait methods
    let mean = norm.mean();
    assert!((mean - 0.0).abs() < 1e-10);

    let variance = norm.var();
    assert!((variance - 1.0).abs() < 1e-10);

    let samples = norm.rvs(10).expect("rvs should generate 10 samples");
    assert_eq!(samples.len(), 10);
}

// ---------------------------------------------------------------------------
// StatsError variants
// ---------------------------------------------------------------------------

/// Verify key error variants remain accessible.
#[test]
fn test_stats_error_variants_accessible() {
    use scirs2_stats::StatsError;

    let _invalid = StatsError::InvalidInput("test".to_string());
    let _insufficient = StatsError::InsufficientData("need more data".to_string());
    let _computation = StatsError::ComputationError("failed".to_string());
}

// ---------------------------------------------------------------------------
// Sampling module
// ---------------------------------------------------------------------------

/// Verify that bootstrap and permutation sampling are accessible.
#[test]
fn test_sampling_accessible() {
    use scirs2_core::ndarray::array;
    use scirs2_stats::sampling;

    let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0];

    let bootstrap =
        sampling::bootstrap(&data.view(), 5, Some(42)).expect("bootstrap should succeed");
    assert_eq!(bootstrap.nrows(), 5);

    let perm = sampling::permutation(&data.view(), Some(123)).expect("permutation should succeed");
    assert_eq!(perm.len(), 5);
}
