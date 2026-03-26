# Statistics (scirs2-stats)

`scirs2-stats` provides production-ready statistical computing modeled after `scipy.stats`.
It covers descriptive statistics, probability distributions (100+), hypothesis testing,
regression, Bayesian methods, MCMC, survival analysis, and more.

## Descriptive Statistics

```rust
use scirs2_core::ndarray::array;
use scirs2_stats::{mean, median, std, var, skew, kurtosis};

fn descriptive() -> Result<(), Box<dyn std::error::Error>> {
    let data = array![1.0, 2.0, 3.0, 4.0, 5.0];

    let m = mean(&data.view())?;           // 3.0
    let med = median(&data.view())?;       // 3.0
    let s = std(&data.view(), 1, None)?;   // Sample std deviation
    let v = var(&data.view(), 1, None)?;   // Sample variance
    let sk = skew(&data.view(), false, None)?;
    let ku = kurtosis(&data.view(), true, false, None)?;

    Ok(())
}
```

## Probability Distributions

### Continuous Distributions

All distributions implement the `Distribution` trait with `pdf`, `cdf`, `ppf` (inverse CDF),
`mean`, `var`, and `rvs` (random sampling).

```rust
use scirs2_stats::distributions;
use scirs2_stats::Distribution;

fn distributions_demo() -> Result<(), Box<dyn std::error::Error>> {
    // Normal: N(mu=0, sigma=1)
    let normal = distributions::norm(0.0f64, 1.0)?;
    let pdf = normal.pdf(0.0);       // 0.3989...
    let cdf = normal.cdf(1.96);     // 0.975...
    let ppf = normal.ppf(0.975)?;   // 1.96...

    // Student's t
    let t = distributions::t(10.0f64, 0.0, 1.0)?;  // df=10

    // Beta distribution
    let beta = distributions::beta(2.0f64, 5.0)?;

    // Gamma distribution
    let gamma = distributions::gamma(2.0f64, 0.0, 1.0)?;

    // Exponential
    let exp = distributions::expon(0.0f64, 1.0)?;

    // Chi-squared
    let chi2 = distributions::chi2(5.0f64, 0.0, 1.0)?;

    // F-distribution
    let f = distributions::f(5.0f64, 10.0, 0.0, 1.0)?;

    // Generate samples from any distribution
    let samples = normal.rvs(1000)?;

    Ok(())
}
```

### Discrete Distributions

```rust
use scirs2_stats::distributions;
use scirs2_stats::Distribution;

fn discrete_demo() -> Result<(), Box<dyn std::error::Error>> {
    // Poisson: Poisson(lambda=3)
    let poisson = distributions::poisson(3.0f64, 0.0)?;
    let pmf = poisson.pmf(2.0);     // P(X = 2)
    let cdf = poisson.cdf(4.0);     // P(X <= 4)

    // Binomial: Binomial(n=10, p=0.5)
    let binom = distributions::binom(10.0f64, 0.5)?;

    Ok(())
}
```

### Multivariate Distributions

```rust,ignore
use scirs2_core::ndarray::array;
use scirs2_stats::distributions::multivariate;

let mean = array![0.0, 0.0];
let cov = array![[1.0, 0.5], [0.5, 2.0]];
let mvn = multivariate::multivariate_normal(mean, cov)?;
let samples = mvn.rvs(100)?;  // 100 x 2 array
```

## Hypothesis Testing

### t-Tests

```rust
use scirs2_core::ndarray::array;
use scirs2_stats::{ttest_1samp, ttest_ind};
use scirs2_stats::tests::ttest::Alternative;

fn t_test_demo() -> Result<(), Box<dyn std::error::Error>> {
    let data = array![5.1, 4.9, 6.2, 5.7, 5.5];

    // One-sample t-test: H0: mu = 5.0
    let result = ttest_1samp(
        &data.view(), 5.0, Alternative::TwoSided, "propagate"
    )?;
    println!("t = {:.4}, p = {:.4}", result.statistic, result.pvalue);

    // Two-sample t-test
    let group_a = array![5.1, 4.9, 6.2, 5.7, 5.5];
    let group_b = array![4.2, 3.8, 4.5, 4.1, 3.9];
    let result = ttest_ind(
        &group_a.view(), &group_b.view(),
        true,  // equal_var
        Alternative::TwoSided,
        "propagate"
    )?;

    Ok(())
}
```

### Non-Parametric Tests

```rust,ignore
use scirs2_stats::{mann_whitney, shapiro, ks_test};

// Mann-Whitney U test (non-parametric alternative to two-sample t-test)
let result = mann_whitney(&x.view(), &y.view(), Alternative::TwoSided)?;

// Shapiro-Wilk normality test
let result = shapiro(&data.view())?;

// Kolmogorov-Smirnov test
let result = ks_test(&data.view(), "norm")?;
```

## Regression

### Linear Regression

```rust,ignore
use scirs2_stats::regression::{linear_regression, RegressionResult};

let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
let y = array![2.1, 3.9, 6.1, 8.0, 9.9];

let result = linear_regression(&x.view(), &y.view())?;
println!("Coefficients: {:?}", result.coefficients);
println!("R-squared: {:.4}", result.r_squared);
```

### Regularized Regression

```rust,ignore
use scirs2_stats::regression::{ridge, lasso, elastic_net};

// Ridge regression (L2 penalty)
let result = ridge(&x.view(), &y.view(), 1.0)?;  // alpha = 1.0

// Lasso regression (L1 penalty)
let result = lasso(&x.view(), &y.view(), 0.1)?;

// Elastic net (L1 + L2)
let result = elastic_net(&x.view(), &y.view(), 0.1, 0.5)?;
```

## Advanced Methods

### MCMC Sampling

```rust,ignore
use scirs2_stats::mcmc::{MetropolisHastings, MCMCOptions};

let options = MCMCOptions::default()
    .with_num_samples(10000)
    .with_burn_in(1000);

let sampler = MetropolisHastings::new(log_posterior, proposal, options);
let chain = sampler.run(&initial_state)?;
```

### Bayesian Inference

```rust,ignore
use scirs2_stats::variational::{ADVI, BBVIOptions};

// Automatic Differentiation Variational Inference
let advi = ADVI::new(model, BBVIOptions::default())?;
let posterior = advi.fit(num_iterations)?;
```

### Survival Analysis

```rust,ignore
use scirs2_stats::survival::{kaplan_meier, cox_proportional_hazards};

let km = kaplan_meier(&times, &events)?;
println!("Median survival: {:.2}", km.median_survival_time()?);

let cox = cox_proportional_hazards(&x.view(), &times, &events)?;
println!("Hazard ratios: {:?}", cox.hazard_ratios());
```

### Conformal Prediction

```rust,ignore
use scirs2_stats::conformal::{SplitConformal, CQR};

// Split conformal prediction for distribution-free coverage
let cp = SplitConformal::new(alpha)?;  // 1-alpha coverage
let intervals = cp.predict(&calibration_scores, &test_points)?;

// Conformalized Quantile Regression (Romano 2019)
let cqr = CQR::new(alpha)?;
let intervals = cqr.predict(&lower_quantiles, &upper_quantiles, &residuals)?;
```

### Causal Inference

```rust,ignore
use scirs2_stats::causal::{IDAlgorithm, SemiMarkovGraph};

// ID algorithm for causal effect identification
let graph = SemiMarkovGraph::new(nodes, edges, bidirected_edges);
let id = IDAlgorithm::new(&graph);
let expression = id.identify(&target, &intervention)?;
```

## SciPy Equivalence Table

| SciPy | SciRS2 |
|-------|--------|
| `scipy.stats.norm` | `distributions::norm` |
| `scipy.stats.gamma` | `distributions::gamma` |
| `scipy.stats.beta` | `distributions::beta` |
| `scipy.stats.ttest_1samp` | `ttest_1samp` |
| `scipy.stats.ttest_ind` | `ttest_ind` |
| `scipy.stats.mannwhitneyu` | `mann_whitney` |
| `scipy.stats.kstest` | `ks_test` |
| `scipy.stats.linregress` | `regression::linear_regression` |
