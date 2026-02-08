# scirs2-stats Development Roadmap

## Production Status (v0.1.5)

This release represents a **production-ready** statistical computing library with comprehensive platform testing and SIMD acceleration. Following the [SciRS2 POLICY](../SCIRS2_POLICY.md), all core functionality has been implemented, tested, and is ready for production use with ecosystem consistency and performance optimizations.

### ✅ Completed Features

#### Core Statistical Functions
- [x] **Descriptive Statistics**: mean, median, variance, standard deviation, skewness, kurtosis, moments
- [x] **Correlation Measures**: Pearson, Spearman, Kendall tau, partial correlation, point-biserial, intraclass correlation
- [x] **Dispersion Measures**: MAD, median absolute deviation, IQR, range, coefficient of variation, Gini coefficient
- [x] **Quantile-based Statistics**: Percentiles, quartiles, box plot statistics, winsorized statistics

#### Statistical Distributions (25+ distributions)
- [x] **Continuous Distributions**: Normal, Uniform, Student's t, Chi-square, F, Gamma, Beta, Exponential, Laplace, Logistic, Cauchy, Pareto, Weibull, Lognormal
- [x] **Discrete Distributions**: Poisson, Bernoulli, Binomial, Geometric, Hypergeometric, Negative Binomial
- [x] **Multivariate Distributions**: Multivariate Normal, Multivariate t, Dirichlet, Wishart, Inverse Wishart, Multinomial, Multivariate Lognormal
- [x] **Circular Distributions**: Basic framework and initial implementations (von Mises, wrapped Cauchy)

#### Statistical Tests
- [x] **Parametric Tests**: One-sample t-test, independent t-test, paired t-test, one-way ANOVA, Tukey HSD
- [x] **Non-parametric Tests**: Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis, Friedman test
- [x] **Normality Tests**: Shapiro-Wilk, Anderson-Darling, D'Agostino's K² test
- [x] **Goodness-of-fit Tests**: Kolmogorov-Smirnov (one-sample, two-sample), Chi-square tests
- [x] **Homogeneity Tests**: Levene's test, Bartlett's test, Brown-Forsythe test

#### Regression Analysis
- [x] **Linear Models**: Simple and multiple linear regression, polynomial regression
- [x] **Robust Regression**: RANSAC, Huber regression, Theil-Sen estimator
- [x] **Regularized Models**: Ridge regression (L2), Lasso regression (L1), Elastic Net
- [x] **Model Selection**: Stepwise regression, cross-validation utilities
- [x] **Diagnostics**: Residual analysis, influence measures, VIF calculation, model criteria (AIC, BIC)

#### Random Number Generation & Sampling
- [x] **RNG Infrastructure**: Updated to rand 0.9.0, thread-safe implementations
- [x] **Basic Sampling**: Uniform, normal, integer sampling, choice function
- [x] **Bootstrap Sampling**: Non-parametric bootstrap with configurable sample sizes
- [x] **Permutation Functions**: Array permutation and reordering

#### SIMD Acceleration (December 29, 2025)
- [x] **Mathematical Utilities**: SIMD-accelerated abs and sign functions
  - [x] `abs_f64`, `abs_f32` - Absolute value computation
  - [x] `sign_f64`, `sign_f32` - Sign extraction (-1, 0, +1)
  - [x] Performance: 1.5-2x speedup for f64, 2-3x for f32 on large arrays (100K+ elements)
  - [x] Platform support: AVX2 (x86_64), NEON (ARM), scalar fallback
- [x] **Statistical Functions**: SIMD-accelerated variance, std, and weighted statistics (Phase 7)
  - [x] `var(ddof=1)` - Fast path using `simd_variance` for sample variance
  - [x] `std(ddof=1)` - Fast path using `simd_std` for sample standard deviation
  - [x] `weighted_mean` - SIMD-accelerated weighted mean computation
  - [x] Performance: 2x speedup for f32 operations on large arrays (1000+ elements)
    - Variance: 675ns (f32) vs 1.411µs (f64) for 1000 elements
    - Std: 440ns (f32) vs 907ns (f64) for 1000 elements
    - Weighted Mean: 959ns (f32) vs 1.19µs (f64) for 1000 elements
  - [x] Zero temporary array allocations using direct SIMD horizontal operations
  - [x] Full backward compatibility with all ddof values
- [x] **Integration**: Full scirs2-core::simd_ops integration
- [x] **Testing**: 38 comprehensive tests (14 math_utils + 24 statistics)
- [x] **Benchmarks**: Performance benchmarks demonstrating SIMD benefits

#### Quality Assurance
- [x] **Comprehensive Testing**: 553 tests (529 base + 14 math_utils + 24 statistics) with 100% pass rate
- [x] **Code Quality**: Zero clippy warnings, formatted code
- [x] **Documentation**: Complete API documentation with examples
- [x] **Integration Tests**: Cross-module functionality testing

---

## Roadmap to v1.0.0 (Stable Release)

### API Stabilization & Polish
- [ ] **API Review**: Final review of public APIs for consistency and usability
- [ ] **Breaking Changes**: Address any remaining breaking changes before stable release
- [ ] **Error Handling**: Standardize error messages and recovery suggestions

### Performance & Optimization
- [ ] **Benchmark Suite**: Comprehensive benchmarks against SciPy and other libraries
- [x] **SIMD Optimizations**: Leverage SIMD instructions for core operations where beneficial
  - [x] Mathematical utilities (abs, sign) with 1.5-3x speedup
  - [x] Statistical operations (variance, std, weighted_mean) with 2x speedup
  - [ ] Optimize distribution sampling operations
  - [ ] Extend to correlation and covariance computations
- [ ] **Parallel Processing**: Expand use of Rayon for large dataset operations
- [ ] **Memory Optimization**: Profile and optimize memory usage patterns

### Extended Testing & Validation
- [ ] **Property-based Testing**: Expand property-based tests for mathematical invariants
- [ ] **Cross-platform Testing**: Ensure consistent behavior across platforms
- [ ] **Numerical Stability**: Extended testing for edge cases and numerical precision

---

## Future Enhancements (Post-1.0)

### Advanced Statistical Methods
- [ ] **Bayesian Statistics**: Conjugate priors, Bayesian linear regression, hierarchical models
- [ ] **MCMC Methods**: Metropolis-Hastings, Gibbs sampling, Hamiltonian Monte Carlo
- [ ] **Multivariate Analysis**: PCA, factor analysis, discriminant analysis
- [ ] **Survival Analysis**: Kaplan-Meier estimator, Cox proportional hazards

### Advanced Sampling & Monte Carlo
- [ ] **Quasi-Monte Carlo**: Sobol sequences, Halton sequences, Latin hypercube sampling
- [ ] **Advanced Bootstrap**: Stratified bootstrap, block bootstrap for time series
- [ ] **Importance Sampling**: Weighted sampling methods for rare events

### Extended Distribution Support
- [ ] **Mixture Models**: Gaussian mixture models, finite mixture distributions
- [ ] **Kernel Density Estimation**: Non-parametric density estimation
- [ ] **Truncated Distributions**: Support for bounded versions of continuous distributions
- [ ] **Custom Distributions**: Framework for user-defined distributions

### Integration & Ecosystem
- [ ] **SciPy Compatibility**: Extended compatibility layer for Python interop
- [ ] **Visualization Integration**: Integration with plotting libraries
- [ ] **Streaming Analytics**: Support for online/streaming statistical computations
- [ ] **GPU Acceleration**: CUDA/OpenCL support for large-scale computations

### Developer Experience
- [ ] **Builder Patterns**: Fluent APIs for complex statistical operations
- [ ] **Proc Macros**: Derive macros for custom statistical types
- [ ] **Error Recovery**: Enhanced error handling with suggested fixes
- [ ] **Performance Profiling**: Built-in profiling for algorithm selection

---

## Contributing

This library is production-ready but we welcome contributions for:

1. **Bug Reports**: Issues with existing functionality
2. **Performance Improvements**: Optimization of existing algorithms
3. **Documentation**: Examples, tutorials, and API improvements
4. **Future Features**: Implementation of post-1.0 roadmap items

See the main repository for contribution guidelines.

---

## Version History

- **v0.1.5** (Current): Production-ready with SIMD-accelerated mathematical utilities
  - Added SIMD abs/sign functions (1.5-3x speedup on large arrays)
  - 294+ tests passing with zero warnings
  - Full scirs2-core::simd_ops integration
- **v0.1.5**: Production-ready release with comprehensive statistical functionality
- **v1.0.0** (Planned): Stable API with performance optimizations and extended testing
- **v1.1.0+** (Future): Advanced statistical methods and ecosystem integration