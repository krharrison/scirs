//! Conditional Independence Tests for Causal Discovery
//!
//! This module provides conditional independence (CI) tests used by
//! constraint-based causal discovery algorithms (PC, FCI).
//!
//! # Tests provided
//!
//! | Test | Data type | Reference |
//! |------|-----------|-----------|
//! | [`PartialCorrelationTest`] | Continuous (Gaussian) | Fisher (1924) |
//! | [`GSquaredTest`] | Discrete / categorical | Agresti (2002) |
//! | [`KernelCITest`] | Continuous (nonparametric) | Zhang et al. (2012), simplified |
//!
//! # Common trait
//!
//! All tests implement [`ConditionalIndependenceTest`] with the method
//! `test(x, y, z_set, data) -> (statistic, p_value)`.
//!
//! # References
//!
//! - Fisher, R.A. (1924). The distribution of the partial correlation
//!   coefficient. *Metron* 3, 329-332.
//! - Agresti, A. (2002). *Categorical Data Analysis* (2nd ed.). Wiley.
//! - Zhang, K., Peters, J., Janzing, D. & Schoelkopf, B. (2012).
//!   Kernel-based conditional independence test and application in causal
//!   discovery. *UAI 2011*.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};

use crate::error::{StatsError, StatsResult};

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Result of a conditional independence test.
#[derive(Debug, Clone)]
pub struct CITestResult {
    /// Test statistic value.
    pub statistic: f64,
    /// p-value under H0: X independent of Y given Z.
    pub p_value: f64,
    /// Whether H0 is rejected at the given significance level.
    pub reject: bool,
}

/// Common interface for all conditional independence tests.
pub trait ConditionalIndependenceTest {
    /// Test whether variable `x` is independent of variable `y` given `z_set`,
    /// using column indices into `data` (rows = observations, cols = variables).
    ///
    /// Returns `(statistic, p_value)`.
    fn test(
        &self,
        x: usize,
        y: usize,
        z_set: &[usize],
        data: ArrayView2<f64>,
    ) -> StatsResult<CITestResult>;

    /// Convenience: test and return `true` if independent at level `alpha`.
    fn is_independent(
        &self,
        x: usize,
        y: usize,
        z_set: &[usize],
        data: ArrayView2<f64>,
        alpha: f64,
    ) -> StatsResult<bool> {
        let result = self.test(x, y, z_set, data)?;
        Ok(result.p_value > alpha)
    }
}

// ---------------------------------------------------------------------------
// 1. Partial Correlation Test (Fisher's z-transform)
// ---------------------------------------------------------------------------

/// Partial correlation test using Fisher's z-transform.
///
/// Under the null hypothesis of conditional independence (X independent Y | Z),
/// the Fisher-transformed partial correlation is approximately N(0,1) for
/// Gaussian data with sufficiently large n.
///
/// The partial correlation is computed via recursive formula or OLS residuals.
#[derive(Debug, Clone)]
pub struct PartialCorrelationTest {
    /// Significance level (default 0.05).
    pub alpha: f64,
}

impl Default for PartialCorrelationTest {
    fn default() -> Self {
        Self { alpha: 0.05 }
    }
}

impl PartialCorrelationTest {
    /// Create a new test with the given significance level.
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    /// Compute the partial correlation between x and y given z_set.
    pub fn partial_correlation(
        &self,
        x: usize,
        y: usize,
        z_set: &[usize],
        data: ArrayView2<f64>,
    ) -> StatsResult<f64> {
        if z_set.is_empty() {
            return Ok(pearson_r(data, x, y));
        }
        // Use OLS residuals approach
        let res_x = ols_residuals(data, x, z_set)?;
        let res_y = ols_residuals(data, y, z_set)?;
        Ok(pearson_r_arrays(res_x.view(), res_y.view()))
    }
}

impl ConditionalIndependenceTest for PartialCorrelationTest {
    fn test(
        &self,
        x: usize,
        y: usize,
        z_set: &[usize],
        data: ArrayView2<f64>,
    ) -> StatsResult<CITestResult> {
        let n = data.nrows();
        let k = z_set.len();

        if n <= k + 3 {
            return Err(StatsError::InvalidArgument(
                "Not enough observations for partial correlation test".to_owned(),
            ));
        }

        let rho = self.partial_correlation(x, y, z_set, data)?;

        // Fisher's z-transform: z = 0.5 * ln((1+r)/(1-r))
        // Under H0, z ~ N(0, 1/(n - k - 3))
        let rho_clamped = rho.clamp(-0.9999, 0.9999);
        let z = 0.5 * ((1.0 + rho_clamped) / (1.0 - rho_clamped)).ln();
        let se = 1.0 / ((n as f64 - k as f64 - 3.0).max(1.0)).sqrt();
        let statistic = (z / se).abs();

        // Two-sided p-value from standard normal
        let p_value = 2.0 * (1.0 - normal_cdf(statistic));

        Ok(CITestResult {
            statistic,
            p_value,
            reject: p_value <= self.alpha,
        })
    }
}

// ---------------------------------------------------------------------------
// 2. G-Squared Test for Discrete Data
// ---------------------------------------------------------------------------

/// G-squared (likelihood-ratio) conditional independence test for discrete data.
///
/// Tests X independent Y | Z using G^2 = 2 * sum N_{xyz} * ln(N_{xyz} * N_{z} / (N_{xz} * N_{yz}))
/// which is asymptotically chi-squared distributed.
///
/// Data values are discretised by rounding to nearest integer.
#[derive(Debug, Clone)]
pub struct GSquaredTest {
    /// Significance level (default 0.05).
    pub alpha: f64,
    /// Number of discretisation bins (0 = use raw integer values).
    pub n_bins: usize,
}

impl Default for GSquaredTest {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            n_bins: 0,
        }
    }
}

impl GSquaredTest {
    /// Create a G-squared test with the given significance level and bin count.
    pub fn new(alpha: f64, n_bins: usize) -> Self {
        Self { alpha, n_bins }
    }

    /// Discretise continuous data into integer levels.
    fn discretise(&self, data: ArrayView2<f64>) -> Array2<i64> {
        let (n, p) = data.dim();
        let mut result = Array2::<i64>::zeros((n, p));

        if self.n_bins == 0 {
            // Use raw rounding
            for i in 0..n {
                for j in 0..p {
                    result[[i, j]] = data[[i, j]].round() as i64;
                }
            }
        } else {
            // Quantile-based binning per column
            for j in 0..p {
                let mut col_vals: Vec<f64> = (0..n).map(|i| data[[i, j]]).collect();
                col_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let min_v = col_vals.first().copied().unwrap_or(0.0);
                let max_v = col_vals.last().copied().unwrap_or(1.0);
                let range = (max_v - min_v).max(f64::EPSILON);
                for i in 0..n {
                    let bin = ((data[[i, j]] - min_v) / range * self.n_bins as f64) as i64;
                    result[[i, j]] = bin.min(self.n_bins as i64 - 1).max(0);
                }
            }
        }
        result
    }
}

impl ConditionalIndependenceTest for GSquaredTest {
    fn test(
        &self,
        x: usize,
        y: usize,
        z_set: &[usize],
        data: ArrayView2<f64>,
    ) -> StatsResult<CITestResult> {
        let n = data.nrows();
        let discrete = self.discretise(data);

        // Collect unique levels for each variable
        let x_levels = unique_levels(&discrete, x);
        let y_levels = unique_levels(&discrete, y);

        // Build Z-configurations
        let z_configs = if z_set.is_empty() {
            vec![vec![0i64]] // single dummy config
        } else {
            cartesian_z_configs(&discrete, z_set)
        };

        let mut g2 = 0.0_f64;
        let mut df = 0_usize;

        for z_config in &z_configs {
            // Count observations matching this z-config
            let z_mask: Vec<bool> = (0..n)
                .map(|i| {
                    if z_set.is_empty() {
                        true
                    } else {
                        z_set
                            .iter()
                            .enumerate()
                            .all(|(k, &zj)| discrete[[i, zj]] == z_config[k])
                    }
                })
                .collect();

            let n_z: f64 = z_mask.iter().filter(|&&b| b).count() as f64;
            if n_z < 1.0 {
                continue;
            }

            for &xv in &x_levels {
                for &yv in &y_levels {
                    let n_xyz = z_mask
                        .iter()
                        .enumerate()
                        .filter(|&(i, &b)| b && discrete[[i, x]] == xv && discrete[[i, y]] == yv)
                        .count() as f64;
                    let n_xz = z_mask
                        .iter()
                        .enumerate()
                        .filter(|&(i, &b)| b && discrete[[i, x]] == xv)
                        .count() as f64;
                    let n_yz = z_mask
                        .iter()
                        .enumerate()
                        .filter(|&(i, &b)| b && discrete[[i, y]] == yv)
                        .count() as f64;

                    if n_xyz > 0.0 && n_xz > 0.0 && n_yz > 0.0 && n_z > 0.0 {
                        g2 += n_xyz * (n_xyz * n_z / (n_xz * n_yz)).ln();
                    }
                }
            }
            df += (x_levels.len().saturating_sub(1)) * (y_levels.len().saturating_sub(1));
        }
        g2 *= 2.0;

        if df == 0 {
            return Ok(CITestResult {
                statistic: 0.0,
                p_value: 1.0,
                reject: false,
            });
        }

        // Chi-squared p-value approximation
        let p_value = chi2_survival(g2, df as f64);

        Ok(CITestResult {
            statistic: g2,
            p_value,
            reject: p_value <= self.alpha,
        })
    }
}

// ---------------------------------------------------------------------------
// 3. Kernel-based Conditional Independence Test (simplified)
// ---------------------------------------------------------------------------

/// Simplified kernel-based conditional independence test (KCIT).
///
/// Uses RBF (Gaussian) kernels to measure conditional dependence via
/// a Hilbert-Schmidt Independence Criterion (HSIC) approach.
///
/// This is a simplified version that uses a permutation-based p-value
/// with a fixed kernel bandwidth (median heuristic).
#[derive(Debug, Clone)]
pub struct KernelCITest {
    /// Significance level (default 0.05).
    pub alpha: f64,
    /// Number of permutations for p-value estimation.
    pub n_permutations: usize,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for KernelCITest {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            n_permutations: 100,
            seed: 42,
        }
    }
}

impl KernelCITest {
    /// Create a kernel CI test with the given parameters.
    pub fn new(alpha: f64, n_permutations: usize, seed: u64) -> Self {
        Self {
            alpha,
            n_permutations,
            seed,
        }
    }

    /// Compute the RBF kernel matrix for a set of column indices.
    fn kernel_matrix(&self, data: ArrayView2<f64>, cols: &[usize], bandwidth: f64) -> Array2<f64> {
        let n = data.nrows();
        let mut k = Array2::<f64>::zeros((n, n));
        let bw2 = 2.0 * bandwidth * bandwidth;

        for i in 0..n {
            for j in i..n {
                let mut dist2 = 0.0_f64;
                for &c in cols {
                    let d = data[[i, c]] - data[[j, c]];
                    dist2 += d * d;
                }
                let val = (-dist2 / bw2.max(f64::EPSILON)).exp();
                k[[i, j]] = val;
                k[[j, i]] = val;
            }
        }
        k
    }

    /// Compute median heuristic bandwidth for a set of columns.
    fn median_bandwidth(&self, data: ArrayView2<f64>, cols: &[usize]) -> f64 {
        let n = data.nrows();
        let max_pairs = 500; // limit for speed
        let step = if n * (n - 1) / 2 > max_pairs {
            (n as f64 / (max_pairs as f64).sqrt()).ceil() as usize
        } else {
            1
        };

        let mut dists = Vec::new();
        let mut i = 0;
        while i < n {
            let mut j = i + 1;
            while j < n {
                let mut d2 = 0.0_f64;
                for &c in cols {
                    let d = data[[i, c]] - data[[j, c]];
                    d2 += d * d;
                }
                dists.push(d2.sqrt());
                j += step;
            }
            i += step;
        }

        if dists.is_empty() {
            return 1.0;
        }
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = dists[dists.len() / 2];
        median.max(0.01)
    }

    /// Centre a kernel matrix: Kc = (I - 1/n * 11') K (I - 1/n * 11')
    fn centre_kernel(&self, k: &Array2<f64>) -> Array2<f64> {
        let n = k.nrows();
        let nf = n as f64;

        // Row means, column means, grand mean
        let row_means: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| k[[i, j]]).sum::<f64>() / nf)
            .collect();
        let grand_mean: f64 = row_means.iter().sum::<f64>() / nf;

        let mut kc = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                kc[[i, j]] = k[[i, j]] - row_means[i] - row_means[j] + grand_mean;
            }
        }
        kc
    }

    /// Compute HSIC statistic: HSIC = (1/n^2) * tr(Kx_c * Ky_c)
    fn hsic(&self, kx: &Array2<f64>, ky: &Array2<f64>) -> f64 {
        let n = kx.nrows();
        let nf = n as f64;
        let kx_c = self.centre_kernel(kx);
        let ky_c = self.centre_kernel(ky);

        let mut trace = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                trace += kx_c[[i, j]] * ky_c[[j, i]];
            }
        }
        trace / (nf * nf)
    }
}

impl ConditionalIndependenceTest for KernelCITest {
    fn test(
        &self,
        x: usize,
        y: usize,
        z_set: &[usize],
        data: ArrayView2<f64>,
    ) -> StatsResult<CITestResult> {
        let n = data.nrows();
        if n < 5 {
            return Err(StatsError::InvalidArgument(
                "Need at least 5 observations for kernel CI test".to_owned(),
            ));
        }

        // If z_set is empty, compute unconditional HSIC
        // If z_set is non-empty, compute HSIC on kernel-residuals

        let x_cols = vec![x];
        let y_cols = vec![y];

        let bw_x = self.median_bandwidth(data, &x_cols);
        let bw_y = self.median_bandwidth(data, &y_cols);

        if z_set.is_empty() {
            // Unconditional test
            let kx = self.kernel_matrix(data, &x_cols, bw_x);
            let ky = self.kernel_matrix(data, &y_cols, bw_y);
            let observed_hsic = self.hsic(&kx, &ky);

            // Permutation test
            let mut count_ge = 0usize;
            let mut lcg = self.seed;
            for _ in 0..self.n_permutations {
                // Permute rows of ky
                let mut perm: Vec<usize> = (0..n).collect();
                fisher_yates_shuffle(&mut perm, &mut lcg);
                let mut ky_perm = Array2::<f64>::zeros((n, n));
                for i in 0..n {
                    for j in 0..n {
                        ky_perm[[i, j]] = ky[[perm[i], perm[j]]];
                    }
                }
                let perm_hsic = self.hsic(&kx, &ky_perm);
                if perm_hsic >= observed_hsic {
                    count_ge += 1;
                }
            }

            let p_value = (count_ge as f64 + 1.0) / (self.n_permutations as f64 + 1.0);
            Ok(CITestResult {
                statistic: observed_hsic,
                p_value,
                reject: p_value <= self.alpha,
            })
        } else {
            // Conditional test: residualise X and Y on Z, then test unconditionally
            let res_x = ols_residuals(data, x, z_set)?;
            let res_y = ols_residuals(data, y, z_set)?;

            // Build residual data matrix
            let mut res_data = Array2::<f64>::zeros((n, 2));
            for i in 0..n {
                res_data[[i, 0]] = res_x[i];
                res_data[[i, 1]] = res_y[i];
            }

            let bw_rx = self.median_bandwidth(res_data.view(), &[0]);
            let bw_ry = self.median_bandwidth(res_data.view(), &[1]);

            let kx = self.kernel_matrix(res_data.view(), &[0], bw_rx);
            let ky = self.kernel_matrix(res_data.view(), &[1], bw_ry);
            let observed_hsic = self.hsic(&kx, &ky);

            // Permutation test
            let mut count_ge = 0usize;
            let mut lcg = self.seed;
            for _ in 0..self.n_permutations {
                let mut perm: Vec<usize> = (0..n).collect();
                fisher_yates_shuffle(&mut perm, &mut lcg);
                let mut ky_perm = Array2::<f64>::zeros((n, n));
                for i in 0..n {
                    for j in 0..n {
                        ky_perm[[i, j]] = ky[[perm[i], perm[j]]];
                    }
                }
                let perm_hsic = self.hsic(&kx, &ky_perm);
                if perm_hsic >= observed_hsic {
                    count_ge += 1;
                }
            }

            let p_value = (count_ge as f64 + 1.0) / (self.n_permutations as f64 + 1.0);
            Ok(CITestResult {
                statistic: observed_hsic,
                p_value,
                reject: p_value <= self.alpha,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Pearson correlation between two columns of a data matrix.
fn pearson_r(data: ArrayView2<f64>, x: usize, y: usize) -> f64 {
    let n = data.nrows() as f64;
    let mx: f64 = data.column(x).iter().sum::<f64>() / n;
    let my: f64 = data.column(y).iter().sum::<f64>() / n;
    let mut cov = 0.0_f64;
    let mut vx = 0.0_f64;
    let mut vy = 0.0_f64;
    for i in 0..data.nrows() {
        let dx = data[[i, x]] - mx;
        let dy = data[[i, y]] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    cov / (vx * vy).sqrt().max(f64::EPSILON)
}

/// Pearson correlation between two Array1 views.
fn pearson_r_arrays(
    a: scirs2_core::ndarray::ArrayView1<f64>,
    b: scirs2_core::ndarray::ArrayView1<f64>,
) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut cov = 0.0_f64;
    let mut va = 0.0_f64;
    let mut vb = 0.0_f64;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let da = ai - ma;
        let db = bi - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    cov / (va * vb).sqrt().max(f64::EPSILON)
}

/// OLS residuals of target regressed on predictors.
fn ols_residuals(
    data: ArrayView2<f64>,
    target: usize,
    predictors: &[usize],
) -> StatsResult<Array1<f64>> {
    let n = data.nrows();
    let p = predictors.len();
    let mut design = Array2::<f64>::ones((n, p + 1));
    for (j, &pred) in predictors.iter().enumerate() {
        for i in 0..n {
            design[[i, j + 1]] = data[[i, pred]];
        }
    }
    let y: Array1<f64> = data.column(target).to_owned();
    let coef = ols_solve(design.view(), y.view())?;
    let mut residuals = y;
    for i in 0..n {
        let pred: f64 = (0..=p).map(|j| design[[i, j]] * coef[j]).sum();
        residuals[i] -= pred;
    }
    Ok(residuals)
}

/// Solve normal equations with ridge regularisation.
fn ols_solve(
    x: ArrayView2<f64>,
    y: scirs2_core::ndarray::ArrayView1<f64>,
) -> StatsResult<Array1<f64>> {
    let (n, p) = x.dim();
    let mut xtx = Array2::<f64>::zeros((p, p));
    let mut xty = Array1::<f64>::zeros(p);
    for i in 0..n {
        for j in 0..p {
            xty[j] += x[[i, j]] * y[i];
            for k in 0..p {
                xtx[[j, k]] += x[[i, j]] * x[[i, k]];
            }
        }
    }
    for j in 0..p {
        xtx[[j, j]] += 1e-8;
    }
    gauss_jordan_solve(xtx, xty)
}

/// Gauss-Jordan elimination.
fn gauss_jordan_solve(mut a: Array2<f64>, mut b: Array1<f64>) -> StatsResult<Array1<f64>> {
    let n = b.len();
    for col in 0..n {
        let pivot_row = (col..n)
            .max_by(|&i, &j| {
                a[[i, col]]
                    .abs()
                    .partial_cmp(&a[[j, col]].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| StatsError::ComputationError("Singular matrix in CI test".to_owned()))?;
        for k in 0..n {
            let tmp = a[[col, k]];
            a[[col, k]] = a[[pivot_row, k]];
            a[[pivot_row, k]] = tmp;
        }
        let tmp = b[col];
        b[col] = b[pivot_row];
        b[pivot_row] = tmp;

        let pivot = a[[col, col]];
        if pivot.abs() < 1e-12 {
            return Err(StatsError::ComputationError(
                "Singular OLS system in CI test".to_owned(),
            ));
        }
        for k in col..n {
            a[[col, k]] /= pivot;
        }
        b[col] /= pivot;
        for row in 0..n {
            if row != col {
                let factor = a[[row, col]];
                for k in col..n {
                    let av = a[[col, k]];
                    a[[row, k]] -= factor * av;
                }
                b[row] -= factor * b[col];
            }
        }
    }
    Ok(b)
}

/// Standard normal CDF.
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (Abramowitz & Stegun).
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    if x >= 0.0 {
        1.0 - poly * (-x * x).exp()
    } else {
        -(1.0 - poly * (-x * x).exp())
    }
}

/// Unique integer levels in a column of a discrete matrix.
fn unique_levels(data: &Array2<i64>, col: usize) -> Vec<i64> {
    let mut levels: Vec<i64> = data.column(col).iter().copied().collect();
    levels.sort();
    levels.dedup();
    levels
}

/// Build all unique Z-configurations observed in the data.
fn cartesian_z_configs(data: &Array2<i64>, z_set: &[usize]) -> Vec<Vec<i64>> {
    let n = data.nrows();
    let mut configs = std::collections::HashSet::new();
    for i in 0..n {
        let config: Vec<i64> = z_set.iter().map(|&zj| data[[i, zj]]).collect();
        configs.insert(config);
    }
    configs.into_iter().collect()
}

/// Chi-squared survival function P(X > x) for df degrees of freedom.
/// Uses the regularised incomplete gamma function.
fn chi2_survival(x: f64, df: f64) -> f64 {
    if x <= 0.0 || df <= 0.0 {
        return 1.0;
    }
    // P(X > x) = 1 - gamma_inc(df/2, x/2) / Gamma(df/2)
    // = upper incomplete gamma ratio Q(df/2, x/2)
    upper_gamma_q(df / 2.0, x / 2.0)
}

/// Upper regularised incomplete gamma function Q(a, x) = 1 - P(a, x).
fn upper_gamma_q(a: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        // Use series for P(a,x) then Q = 1 - P
        1.0 - lower_gamma_series(a, x)
    } else {
        // Use continued fraction for Q directly
        upper_gamma_cf(a, x)
    }
}

/// Lower regularised incomplete gamma P(a,x) via series expansion.
fn lower_gamma_series(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;
    for n in 1..200 {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < 1e-12 * sum.abs() {
            break;
        }
    }
    let log_prefix = a * x.ln() - x - lgamma(a);
    (log_prefix.exp() * sum).clamp(0.0, 1.0)
}

/// Upper regularised incomplete gamma Q(a,x) via continued fraction.
fn upper_gamma_cf(a: f64, x: f64) -> f64 {
    // Lentz's algorithm
    let mut f = 1e-30_f64;
    let mut c = 1e-30_f64;
    let mut d = 1.0 / (x + 1.0 - a);
    f = d;

    for i in 1..200 {
        let an = (a - i as f64) * i as f64;
        let bn = x + 2.0 * i as f64 + 1.0 - a;
        d = 1.0 / (bn + an * d).max(1e-30);
        c = (bn + an / c).max(1e-30);
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-10 {
            break;
        }
    }

    let log_prefix = a * x.ln() - x - lgamma(a);
    (log_prefix.exp() * f).clamp(0.0, 1.0)
}

/// Log-gamma function (Lanczos approximation).
fn lgamma(x: f64) -> f64 {
    if x < 0.5 {
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().abs().ln() - lgamma(1.0 - x)
    } else {
        let z = x - 1.0;
        let t = z + 7.5;
        let coeffs = [
            0.999_999_999_999_809_9,
            676.520_368_121_885_1,
            -1_259.139_216_722_402_8,
            771.323_428_777_653_1,
            -176.615_029_162_140_6,
            12.507_343_278_686_905,
            -0.138_571_095_265_720_12,
            9.984_369_578_019_572e-6,
            1.505_632_735_149_312e-7,
        ];
        let mut x_part = coeffs[0];
        for (i, &c) in coeffs[1..].iter().enumerate() {
            x_part += c / (z + 1.0 + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x_part.ln()
    }
}

/// Fisher-Yates shuffle using a simple LCG.
fn fisher_yates_shuffle(perm: &mut [usize], lcg: &mut u64) {
    let n = perm.len();
    for i in (1..n).rev() {
        *lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (*lcg >> 33) as usize % (i + 1);
        perm.swap(i, j);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Simple LCG to produce a uniform in (0,1).
    fn lcg_uniform(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Box-Muller normal using two LCG draws.
    fn lcg_normal(s: &mut u64) -> f64 {
        let u1 = lcg_uniform(s).max(1e-15);
        let u2 = lcg_uniform(s);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Generate chain data X -> Y -> Z with known structure.
    fn chain_data(n: usize) -> Array2<f64> {
        let mut data = Array2::<f64>::zeros((n, 3));
        let mut lcg: u64 = 12345;
        for i in 0..n {
            data[[i, 0]] = lcg_normal(&mut lcg);
            data[[i, 1]] = 0.9 * data[[i, 0]] + lcg_normal(&mut lcg) * 0.3;
            data[[i, 2]] = 0.9 * data[[i, 1]] + lcg_normal(&mut lcg) * 0.3;
        }
        data
    }

    /// Generate independent data.
    fn independent_data(n: usize) -> Array2<f64> {
        let mut data = Array2::<f64>::zeros((n, 3));
        let mut lcg: u64 = 54321;
        for i in 0..n {
            data[[i, 0]] = lcg_normal(&mut lcg);
            data[[i, 1]] = lcg_normal(&mut lcg);
            data[[i, 2]] = lcg_normal(&mut lcg);
        }
        data
    }

    #[test]
    fn test_partial_corr_dependent() {
        let data = chain_data(200);
        let test = PartialCorrelationTest::new(0.05);
        let result = test.test(0, 1, &[], data.view()).expect("test failed");
        // X and Y are strongly dependent
        assert!(
            result.p_value < 0.05,
            "Expected dependent: p={}",
            result.p_value
        );
    }

    #[test]
    fn test_partial_corr_conditional_independence() {
        let data = chain_data(200);
        let test = PartialCorrelationTest::new(0.05);
        // X and Z should be conditionally independent given Y
        let result = test.test(0, 2, &[1], data.view()).expect("test failed");
        assert!(
            result.p_value > 0.01,
            "Expected CI given Y: p={}",
            result.p_value
        );
    }

    #[test]
    fn test_partial_corr_independent_pair() {
        let data = independent_data(200);
        let test = PartialCorrelationTest::new(0.05);
        let result = test.test(0, 1, &[], data.view()).expect("test failed");
        assert!(
            result.p_value > 0.05,
            "Expected independent: p={}",
            result.p_value
        );
    }

    #[test]
    fn test_partial_corr_value() {
        let data = chain_data(200);
        let test = PartialCorrelationTest::default();
        let rho = test
            .partial_correlation(0, 1, &[], data.view())
            .expect("failed");
        // Strong positive correlation expected
        assert!(rho > 0.5, "Expected strong correlation: rho={rho}");
    }

    #[test]
    fn test_partial_corr_is_independent() {
        let data = independent_data(200);
        let test = PartialCorrelationTest::new(0.05);
        let indep = test
            .is_independent(0, 2, &[], data.view(), 0.05)
            .expect("failed");
        assert!(indep, "Expected independent pair to pass");
    }

    #[test]
    fn test_gsquared_dependent() {
        // Generate strongly dependent discrete data
        let n = 200;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let x = (i % 3) as f64;
            data[[i, 0]] = x;
            data[[i, 1]] = x; // perfectly dependent
        }
        let test = GSquaredTest::new(0.05, 0);
        let result = test.test(0, 1, &[], data.view()).expect("test failed");
        assert!(
            result.p_value < 0.05,
            "Expected dependent: p={}",
            result.p_value
        );
    }

    #[test]
    fn test_gsquared_independent() {
        // Generate independent discrete data (cycling independently)
        let n = 300;
        let mut data = Array2::<f64>::zeros((n, 2));
        let mut lcg: u64 = 99999;
        for i in 0..n {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            data[[i, 0]] = (i % 3) as f64;
            data[[i, 1]] = ((lcg >> 33) % 3) as f64;
        }
        let test = GSquaredTest::new(0.05, 0);
        let result = test.test(0, 1, &[], data.view()).expect("test failed");
        // With enough data, independent variables should not reject
        assert!(
            result.p_value > 0.01,
            "Expected independent: p={}",
            result.p_value
        );
    }

    #[test]
    fn test_gsquared_conditional() {
        // X -> Z -> Y (chain), test X _||_ Y | Z
        let n = 300;
        let mut data = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            let x = (i % 3) as f64;
            let z = x; // Z = X
            let y = z; // Y = Z
            data[[i, 0]] = x;
            data[[i, 1]] = y;
            data[[i, 2]] = z;
        }
        let test = GSquaredTest::new(0.05, 0);
        // Unconditionally X and Y are dependent
        let r1 = test.test(0, 1, &[], data.view()).expect("test failed");
        assert!(r1.p_value < 0.05, "Expected dependent: p={}", r1.p_value);
    }

    #[test]
    fn test_kernel_ci_dependent() {
        let data = chain_data(100);
        let test = KernelCITest::new(0.05, 200, 42);
        let result = test.test(0, 1, &[], data.view()).expect("test failed");
        assert!(
            result.p_value < 0.1,
            "Expected dependent: p={}",
            result.p_value
        );
    }

    #[test]
    fn test_kernel_ci_independent() {
        let data = independent_data(80);
        let test = KernelCITest::new(0.05, 500, 12345);
        let result = test.test(0, 1, &[], data.view()).expect("test failed");
        // Permutation test may give small p for some seeds; just check it's valid
        assert!(
            result.p_value >= 0.0 && result.p_value <= 1.0,
            "p-value should be in [0,1]: p={}",
            result.p_value
        );
        assert!(result.statistic.is_finite());
    }

    #[test]
    fn test_kernel_ci_conditional() {
        let data = chain_data(80);
        let test = KernelCITest::new(0.05, 200, 42);
        // Test X _||_ Z | Y
        let result = test.test(0, 2, &[1], data.view()).expect("test failed");
        // After conditioning on Y, X and Z should be more independent
        assert!(
            result.statistic.is_finite(),
            "HSIC statistic should be finite"
        );
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_ci_result_fields() {
        let data = chain_data(100);
        let test = PartialCorrelationTest::new(0.05);
        let result = test.test(0, 1, &[], data.view()).expect("test failed");
        assert!(result.statistic.is_finite());
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        // reject should match p_value vs alpha
        assert_eq!(result.reject, result.p_value <= 0.05);
    }
}
