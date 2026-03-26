//! Tutorial: Statistical Distributions with SciRS2
//!
//! This tutorial covers working with probability distributions,
//! descriptive statistics, hypothesis testing, and regression
//! in scirs2-stats.
//!
//! Run with: cargo run -p scirs2-stats --example tutorial_distributions

use scirs2_core::ndarray::{array, ArrayView1};
use scirs2_stats::distributions::{Beta, Exponential, Gamma, Normal};
use scirs2_stats::error::StatsResult;
use scirs2_stats::tests::ttest::Alternative;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SciRS2 Statistics Tutorial ===\n");

    section_descriptive_stats()?;
    section_distributions()?;
    section_hypothesis_testing()?;
    section_regression()?;

    println!("\n=== Tutorial Complete ===");
    Ok(())
}

/// Section 1: Descriptive statistics (mean, variance, skewness, etc.)
fn section_descriptive_stats() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- 1. Descriptive Statistics ---\n");

    let data = array![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    println!("Data: {:?}\n", data.to_vec());

    // Mean
    let m = scirs2_stats::mean(&data.view())?;
    println!("Mean:     {:.4}", m);
    // Output: Mean: 5.0

    // Median
    let med = scirs2_stats::median(&data.view())?;
    println!("Median:   {:.4}", med);
    // Output: Median: 4.5

    // Variance (ddof=1 for sample variance)
    let v = scirs2_stats::var(&data.view(), 1, None)?;
    println!("Variance (sample): {:.4}", v);

    // Standard deviation
    let s = scirs2_stats::std(&data.view(), 1, None)?;
    println!("Std dev (sample):  {:.4}", s);

    // Skewness
    let sk = scirs2_stats::skew(&data.view(), true, None)?;
    println!("Skewness: {:.4}", sk);

    // Kurtosis (excess kurtosis, Fisher=true by default)
    let k = scirs2_stats::kurtosis(&data.view(), true, true, None)?;
    println!("Kurtosis: {:.4}\n", k);

    Ok(())
}

/// Section 2: Working with probability distributions
fn section_distributions() -> StatsResult<()> {
    println!("--- 2. Probability Distributions ---\n");

    // --- Normal Distribution ---
    println!("  Normal Distribution (mu=0, sigma=1):");
    let norm = Normal::new(0.0_f64, 1.0)?;

    // PDF: probability density function
    let pdf_0 = norm.pdf(0.0);
    println!("    PDF(0)   = {:.6}", pdf_0);
    // Output: ~0.398942 (1/sqrt(2*pi))

    let pdf_1 = norm.pdf(1.0);
    println!("    PDF(1)   = {:.6}", pdf_1);

    // CDF: cumulative distribution function
    let cdf_0 = norm.cdf(0.0);
    println!("    CDF(0)   = {:.6}", cdf_0);
    // Output: 0.500000

    let cdf_196 = norm.cdf(1.96);
    println!("    CDF(1.96)= {:.6}", cdf_196);
    // Output: ~0.975 (the 97.5th percentile)

    // PPF: percent point function (inverse CDF / quantile function)
    let ppf_975 = norm.ppf(0.975)?;
    println!("    PPF(0.975) = {:.4}", ppf_975);
    // Output: ~1.96
    println!();

    // --- Exponential Distribution ---
    println!("  Exponential Distribution (lambda=2.0):");
    let exp_dist = Exponential::new(2.0_f64, 0.0)?;
    println!("    PDF(0.5) = {:.6}", exp_dist.pdf(0.5));
    println!("    CDF(1.0) = {:.6}", exp_dist.cdf(1.0));
    println!("    Mean     = {:.6}", exp_dist.mean());
    println!();

    // --- Beta Distribution ---
    println!("  Beta Distribution (alpha=2, beta=5):");
    let beta = Beta::new(2.0_f64, 5.0, 0.0, 1.0)?;
    println!("    PDF(0.3) = {:.6}", beta.pdf(0.3));
    println!("    CDF(0.5) = {:.6}", beta.cdf(0.5));
    let q50 = beta.ppf(0.5)?;
    println!("    Median (PPF(0.5)) = {:.6}", q50);
    println!();

    // --- Gamma Distribution ---
    println!("  Gamma Distribution (shape=2, scale=1):");
    let gam = Gamma::new(2.0_f64, 1.0, 0.0)?;
    println!("    PDF(1.0) = {:.6}", gam.pdf(1.0));
    println!("    CDF(2.0) = {:.6}", gam.cdf(2.0));
    println!();

    Ok(())
}

/// Section 3: Hypothesis testing
fn section_hypothesis_testing() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- 3. Hypothesis Testing ---\n");

    // One-sample t-test: test if mean differs from a known value
    let sample = array![5.1_f64, 4.9, 5.2, 5.0, 4.8, 5.1, 5.3, 4.9, 5.0, 5.2];
    let result =
        scirs2_stats::ttest_1samp(&sample.view(), 5.0, Alternative::TwoSided, "propagate")?;
    println!("One-sample t-test (H0: mu = 5.0):");
    println!("  t-statistic = {:.4}", result.statistic);
    println!("  p-value     = {:.4}", result.pvalue);
    println!(
        "  Conclusion: {}",
        if result.pvalue < 0.05 {
            "Reject H0 at 5% level"
        } else {
            "Fail to reject H0 at 5% level"
        }
    );
    println!();

    // Two-sample t-test: compare means of two groups
    let group_a = array![12.0_f64, 14.0, 15.0, 13.0, 11.0, 14.0, 13.5, 12.5];
    let group_b = array![10.0_f64, 11.0, 12.0, 10.5, 11.5, 10.0, 11.0, 10.5];

    let result2 = scirs2_stats::ttest_ind(
        &group_a.view(),
        &group_b.view(),
        true,
        Alternative::TwoSided,
        "propagate",
    )?;
    println!("Two-sample t-test (equal variances assumed):");
    println!("  t-statistic = {:.4}", result2.statistic);
    println!("  p-value     = {:.6}", result2.pvalue);
    println!(
        "  Conclusion: {}",
        if result2.pvalue < 0.05 {
            "Groups differ significantly"
        } else {
            "No significant difference"
        }
    );
    println!();

    // Normality test: Shapiro-Wilk
    // Returns (W-statistic, p-value)
    let data = array![2.3_f64, 3.1, 2.8, 3.5, 2.9, 3.0, 2.7, 3.2, 2.6, 3.4];
    let (w_stat, sw_pvalue) = scirs2_stats::shapiro_wilk(&data.view())?;
    println!("Shapiro-Wilk normality test:");
    println!("  W-statistic = {:.4}", w_stat);
    println!("  p-value     = {:.4}", sw_pvalue);
    println!(
        "  Conclusion: {}",
        if sw_pvalue > 0.05 {
            "Data appears normally distributed"
        } else {
            "Data may not be normally distributed"
        }
    );
    println!();

    Ok(())
}

/// Section 4: Regression analysis
fn section_regression() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- 4. Regression ---\n");

    // Simple linear regression: y = slope * x + intercept
    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = array![2.1_f64, 3.9, 6.2, 7.8, 10.1, 12.0, 13.9, 16.1];

    // linregress returns (slope, intercept, r_value, p_value, std_err)
    let (slope, intercept, r_value, p_value, std_err) =
        scirs2_stats::linregress(&x.view(), &y.view())?;
    println!("Simple linear regression: y = slope * x + intercept");
    println!("  Slope:     {:.4}", slope);
    println!("  Intercept: {:.4}", intercept);
    println!("  R-value:   {:.4}", r_value);
    println!("  R-squared: {:.4}", r_value * r_value);
    println!("  P-value:   {:.6}", p_value);
    println!("  Std error: {:.4}", std_err);
    println!();

    // Multiple linear regression
    let x_multi = array![
        [1.0_f64, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
        [5.0, 6.0],
        [6.0, 7.0],
        [7.0, 8.0],
        [8.0, 9.0]
    ];
    let y_multi = array![5.1_f64, 8.0, 11.1, 13.9, 17.0, 19.9, 23.1, 26.0];

    let result = scirs2_stats::linear_regression(&x_multi.view(), &y_multi.view(), None)?;
    println!("Multiple linear regression:");
    println!("  Coefficients: {:?}", result.coefficients.to_vec());
    println!("  R-squared:    {:.4}", result.r_squared);
    println!("  Adj R-squared:{:.4}", result.adj_r_squared);
    println!("  F-statistic:  {:.4}", result.f_statistic);
    println!();

    Ok(())
}
