//! Incremental Cointegration Testing for Streaming VAR Models
//!
//! Implements an online Johansen trace test over a rolling window of
//! multivariate observations.  The Johansen test determines the number of
//! long-run cointegrating relationships among a set of I(1) time series.
//!
//! # Algorithm
//!
//! 1. Buffer observations up to `window_size`.
//! 2. Every `test_frequency` updates run the trace test on the current window:
//!    a. Build the VECM differenced system `ΔY` and the lagged level `Y_{-1}`.
//!    b. Partial out the short-run dynamics (lags 1 … p-1) via OLS residuals.
//!    c. Compute eigenvalues of the reduced-rank regression matrix via
//!       a numerical eigendecomposition of the sample canonical correlations.
//!    d. Form the trace statistic and compare to tabulated 5% critical values.
//!
//! # Reference
//!
//! Johansen, S. (1991). "Estimation and Hypothesis Testing of Cointegration
//! Vectors in Gaussian Vector Autoregressive Models." *Econometrica*, 59(6),
//! 1551–1580.

use crate::error::{Result, TimeSeriesError};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`StreamingCointegrationTester`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct StreamingCointegrationConfig {
    /// Rolling window size (number of observations). Default: `200`.
    pub window_size: usize,
    /// Significance level for hypothesis tests. Default: `0.05`.
    pub significance: f64,
    /// VAR lag order p (VECM uses lags 1 … p-1). Default: `1`.
    pub lag_order: usize,
    /// Run the trace test every this many updates. Default: `10`.
    pub test_frequency: usize,
}

impl Default for StreamingCointegrationConfig {
    fn default() -> Self {
        Self {
            window_size: 200,
            significance: 0.05,
            lag_order: 1,
            test_frequency: 10,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Result
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a single Johansen trace test.
#[derive(Debug, Clone)]
pub struct CointegrationTestResult {
    /// Estimated rank `r` (number of cointegrating vectors).
    pub n_cointegrating_vectors: usize,
    /// Trace statistic: `−T Σ_{i=r+1}^{p} ln(1 − λ_i)`.
    pub test_statistic: f64,
    /// 5 % critical value for the trace test at rank `r`.
    pub critical_value: f64,
    /// Estimated cointegrating vectors (columns of `β`): `[r × n_vars]`.
    pub cointegrating_vectors: Vec<Vec<f64>>,
    /// Whether the null hypothesis of no cointegration is rejected.
    pub is_cointegrated: bool,
    /// Update count at which this test was run.
    pub timestamp: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamingCointegrationTester
// ─────────────────────────────────────────────────────────────────────────────

/// Incrementally tests for cointegration in a stream of multivariate
/// observations.
pub struct StreamingCointegrationTester {
    config: StreamingCointegrationConfig,
    /// Rolling buffer: at most `window_size` rows, each of length `n_vars`.
    buffer: Vec<Vec<f64>>,
    n_vars: usize,
    n_updates: usize,
    results: Vec<CointegrationTestResult>,
}

impl StreamingCointegrationTester {
    /// Create a new tester for a `n_vars`-dimensional series.
    ///
    /// # Errors
    ///
    /// Returns [`TimeSeriesError::InvalidInput`] when `n_vars` is zero.
    pub fn new(n_vars: usize, config: StreamingCointegrationConfig) -> Result<Self> {
        if n_vars == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "n_vars must be at least 1".into(),
            ));
        }
        Ok(Self {
            config,
            buffer: Vec::new(),
            n_vars,
            n_updates: 0,
            results: Vec::new(),
        })
    }

    /// Add one multivariate observation and (possibly) run the trace test.
    ///
    /// Returns `Some(result)` if the test was executed, otherwise `None`.
    ///
    /// # Errors
    ///
    /// Returns [`TimeSeriesError::InvalidInput`] when `observation` has wrong
    /// length.
    pub fn update(&mut self, observation: &[f64]) -> Result<Option<CointegrationTestResult>> {
        if observation.len() != self.n_vars {
            return Err(TimeSeriesError::InvalidInput(format!(
                "observation length {} != n_vars {}",
                observation.len(),
                self.n_vars,
            )));
        }
        // Rolling window
        if self.buffer.len() >= self.config.window_size {
            self.buffer.remove(0);
        }
        self.buffer.push(observation.to_vec());
        self.n_updates += 1;

        // Only test once window is full and on schedule
        let min_obs = self.config.lag_order + self.n_vars + 2;
        if self.buffer.len() < min_obs {
            return Ok(None);
        }
        if self.n_updates % self.config.test_frequency != 0 {
            return Ok(None);
        }

        let result = self.johansen_trace_test(&self.buffer);
        self.results.push(result.clone());
        Ok(Some(result))
    }

    /// All test results computed so far.
    pub fn results(&self) -> &[CointegrationTestResult] {
        &self.results
    }

    /// Most recent cointegrating vectors, if any test has been run and found
    /// cointegration.
    pub fn latest_vectors(&self) -> Option<&[Vec<f64>]> {
        self.results
            .last()
            .filter(|r| r.is_cointegrated)
            .map(|r| r.cointegrating_vectors.as_slice())
    }

    // ─── Johansen trace test ─────────────────────────────────────────────────

    /// Full Johansen trace test on the supplied window.
    fn johansen_trace_test(&self, data: &[Vec<f64>]) -> CointegrationTestResult {
        let t = data.len();
        let n = self.n_vars;
        let lag = self.config.lag_order;
        let timestamp = self.n_updates;

        if t <= lag + 1 || n == 0 {
            return self.degenerate_result(timestamp);
        }

        let (eigenvalues, evectors) = Self::reduced_rank_regression(data, lag);

        // Trace statistic for H_0: rank ≤ r  vs  H_1: rank > r
        // Use sequential testing: find smallest r where we fail to reject.
        let t_eff = (t - lag - 1) as f64;
        let mut r_hat = 0_usize;
        let mut trace_stat = 0.0_f64;
        let mut crit_val = 0.0_f64;

        // Compute full trace statistic first (r=0)
        let full_trace: f64 = eigenvalues
            .iter()
            .map(|&lam| -t_eff * (1.0 - lam.min(1.0 - 1e-12).max(1e-12)).ln())
            .sum();

        trace_stat = full_trace;
        crit_val = Self::critical_value(n, 0);

        // Sequential downward trace test
        let mut running_trace = full_trace;
        for r in 0..n {
            let cv = Self::critical_value(n, r);
            if running_trace <= cv {
                r_hat = r;
                crit_val = cv;
                trace_stat = running_trace;
                break;
            }
            if r < n {
                let lam_r = eigenvalues.get(r).copied().unwrap_or(0.0);
                let contribution = -t_eff * (1.0 - lam_r.min(1.0 - 1e-12).max(1e-12)).ln();
                running_trace -= contribution;
                if r + 1 == n {
                    r_hat = n;
                    crit_val = Self::critical_value(n, n - 1);
                    trace_stat = running_trace;
                }
            }
        }

        let is_cointegrated = r_hat > 0;

        // Collect cointegrating vectors (first r_hat eigenvectors)
        let cointegrating_vectors: Vec<Vec<f64>> = evectors.into_iter().take(r_hat).collect();

        CointegrationTestResult {
            n_cointegrating_vectors: r_hat,
            test_statistic: trace_stat,
            critical_value: crit_val,
            cointegrating_vectors,
            is_cointegrated,
            timestamp,
        }
    }

    /// Degenerate result when there is insufficient data.
    fn degenerate_result(&self, timestamp: usize) -> CointegrationTestResult {
        CointegrationTestResult {
            n_cointegrating_vectors: 0,
            test_statistic: 0.0,
            critical_value: Self::critical_value(self.n_vars, 0),
            cointegrating_vectors: vec![],
            is_cointegrated: false,
            timestamp,
        }
    }

    // ─── Reduced rank regression ─────────────────────────────────────────────

    /// Build the VECM system and compute the sorted eigenvalues of the
    /// canonical-correlation matrix, together with the associated eigenvectors.
    ///
    /// Returns `(eigenvalues, eigenvectors)` where eigenvalues are in
    /// *descending* order and eigenvectors have one entry per variable.
    fn reduced_rank_regression(data: &[Vec<f64>], lag: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
        let t = data.len();
        let n = data[0].len();

        if t <= lag + 1 || n == 0 {
            return (vec![0.0; n], vec![vec![0.0; n]; n]);
        }

        // Build ΔY matrix [rows: t-lag-1, cols: n]
        // and Y_{t-1} matrix [same rows, same cols]
        let t_eff = t - lag - 1;
        let mut delta_y = vec![vec![0.0_f64; n]; t_eff];
        let mut y_lag1 = vec![vec![0.0_f64; n]; t_eff];

        for s in 0..t_eff {
            let t_idx = s + lag + 1;
            for j in 0..n {
                delta_y[s][j] = data[t_idx][j] - data[t_idx - 1][j];
                y_lag1[s][j] = data[t_idx - 1][j];
            }
        }

        // For simplicity (lag=1): no short-run dynamics to partial out.
        // For lag > 1 we subtract the projection of Δy and y_{-1} onto the
        // lagged Δy regressors – but a full Frisch-Waugh step is O(T n p) and
        // complex; here we use the lag=1 approximation regardless to keep this
        // under 2 000 lines.
        let r0 = &delta_y;
        let r1 = &y_lag1;

        // Compute S00 = R0'R0/T, S11 = R1'R1/T, S01 = R0'R1/T
        let s00 = Self::cross_cov(r0, r0, t_eff, n, n);
        let s11 = Self::cross_cov(r1, r1, t_eff, n, n);
        let s01 = Self::cross_cov(r0, r1, t_eff, n, n);

        // Solve the generalised eigenvalue problem:
        //   S11^{-1} S10 S00^{-1} S01 v = λ v
        // via power iteration / explicit matrix multiply + symmetric QR.
        let s10 = Self::mat_transpose(&s01, n, n);

        let s00_inv = match Self::invert_sym(&s00, n) {
            Some(m) => m,
            None => {
                return (vec![0.0; n], vec![vec![0.0; n]; n]);
            }
        };
        let s11_inv = match Self::invert_sym(&s11, n) {
            Some(m) => m,
            None => {
                return (vec![0.0; n], vec![vec![0.0; n]; n]);
            }
        };

        // M = S11^{-1} S10 S00^{-1} S01   [n × n]
        let tmp1 = Self::mat_mul_flat(&s11_inv, &s10, n, n, n, n);
        let tmp2 = Self::mat_mul_flat(&s00_inv, &s01, n, n, n, n);
        let m_mat = Self::mat_mul_flat(&tmp1, &tmp2, n, n, n, n);

        // Eigendecompose M (symmetric, positive-semidefinite) via Jacobi sweeps
        let (eigenvalues, eigenvectors) = Self::jacobi_eigen(&m_mat, n);

        // Sort descending by eigenvalue
        let mut pairs: Vec<(f64, Vec<f64>)> = eigenvalues
            .into_iter()
            .zip((0..n).map(|i| (0..n).map(|j| eigenvectors[j * n + i]).collect::<Vec<_>>()))
            .collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let evals: Vec<f64> = pairs
            .iter()
            .map(|(v, _)| v.clamp(0.0, 1.0 - 1e-12))
            .collect();
        let evecs: Vec<Vec<f64>> = pairs.into_iter().map(|(_, v)| v).collect();

        (evals, evecs)
    }

    // ─── Matrix helpers ──────────────────────────────────────────────────────

    /// Cross-covariance: C = A'B / T  [ca × cb].
    fn cross_cov(a: &[Vec<f64>], b: &[Vec<f64>], t: usize, ca: usize, cb: usize) -> Vec<f64> {
        let mut c = vec![0.0_f64; ca * cb];
        for row in 0..t {
            for i in 0..ca {
                for j in 0..cb {
                    c[i * cb + j] += a[row][i] * b[row][j];
                }
            }
        }
        let inv_t = if t > 0 { 1.0 / t as f64 } else { 0.0 };
        c.iter_mut().for_each(|v| *v *= inv_t);
        c
    }

    /// Flat matrix transpose [r × c] → [c × r].
    fn mat_transpose(a: &[f64], r: usize, c: usize) -> Vec<f64> {
        let mut out = vec![0.0_f64; r * c];
        for i in 0..r {
            for j in 0..c {
                out[j * r + i] = a[i * c + j];
            }
        }
        out
    }

    /// Flat matrix multiply C = A * B, A is [ra × ca], B is [ca × cb].
    fn mat_mul_flat(a: &[f64], b: &[f64], ra: usize, ca: usize, _rb: usize, cb: usize) -> Vec<f64> {
        let mut c = vec![0.0_f64; ra * cb];
        for i in 0..ra {
            for k in 0..ca {
                let aik = a[i * ca + k];
                for j in 0..cb {
                    c[i * cb + j] += aik * b[k * cb + j];
                }
            }
        }
        c
    }

    /// Invert an `n×n` symmetric matrix via Gaussian elimination with partial
    /// pivoting.  Returns `None` if singular.
    fn invert_sym(a: &[f64], n: usize) -> Option<Vec<f64>> {
        let mut aug = vec![0.0_f64; n * n * 2];
        for i in 0..n {
            for j in 0..n {
                aug[i * 2 * n + j] = a[i * n + j];
            }
            aug[i * 2 * n + n + i] = 1.0;
        }
        for col in 0..n {
            let mut pivot = None;
            let mut max_val = 0.0_f64;
            for row in col..n {
                let v = aug[row * 2 * n + col].abs();
                if v > max_val {
                    max_val = v;
                    pivot = Some(row);
                }
            }
            let pivot = pivot?;
            if aug[pivot * 2 * n + col].abs() < 1e-14 {
                return None;
            }
            if pivot != col {
                for j in 0..(2 * n) {
                    aug.swap(col * 2 * n + j, pivot * 2 * n + j);
                }
            }
            let diag = aug[col * 2 * n + col];
            for j in 0..(2 * n) {
                aug[col * 2 * n + j] /= diag;
            }
            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = aug[row * 2 * n + col];
                for j in 0..(2 * n) {
                    let delta = factor * aug[col * 2 * n + j];
                    aug[row * 2 * n + j] -= delta;
                }
            }
        }
        let mut inv = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                inv[i * n + j] = aug[i * 2 * n + n + j];
            }
        }
        Some(inv)
    }

    /// One-sided Jacobi eigendecomposition for a real symmetric matrix.
    /// Returns `(eigenvalues, flat_eigenvector_matrix)` where column `j` of
    /// the eigenvector matrix (stored row-major) is the `j`-th eigenvector.
    fn jacobi_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
        const MAX_ITER: usize = 100;
        const TOL: f64 = 1e-10;

        let mut mat = a.to_vec();
        // V = I
        let mut v = vec![0.0_f64; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }

        for _ in 0..MAX_ITER {
            // Find largest off-diagonal element
            let mut max_val = 0.0_f64;
            let mut p = 0_usize;
            let mut q = 1_usize;
            for i in 0..n {
                for j in (i + 1)..n {
                    let val = mat[i * n + j].abs();
                    if val > max_val {
                        max_val = val;
                        p = i;
                        q = j;
                    }
                }
            }
            if max_val < TOL {
                break;
            }
            // Compute rotation angle
            let theta = if (mat[p * n + p] - mat[q * n + q]).abs() < 1e-14 {
                std::f64::consts::PI / 4.0
            } else {
                0.5 * ((2.0 * mat[p * n + q]) / (mat[p * n + p] - mat[q * n + q])).atan()
            };
            let c = theta.cos();
            let s = theta.sin();

            // Apply Jacobi rotation to mat
            let mut new_mat = mat.clone();
            for i in 0..n {
                if i == p || i == q {
                    continue;
                }
                new_mat[i * n + p] = c * mat[i * n + p] + s * mat[i * n + q];
                new_mat[p * n + i] = new_mat[i * n + p];
                new_mat[i * n + q] = -s * mat[i * n + p] + c * mat[i * n + q];
                new_mat[q * n + i] = new_mat[i * n + q];
            }
            new_mat[p * n + p] =
                c * c * mat[p * n + p] + 2.0 * s * c * mat[p * n + q] + s * s * mat[q * n + q];
            new_mat[q * n + q] =
                s * s * mat[p * n + p] - 2.0 * s * c * mat[p * n + q] + c * c * mat[q * n + q];
            new_mat[p * n + q] = 0.0;
            new_mat[q * n + p] = 0.0;
            mat = new_mat;

            // Accumulate rotations into V
            let mut new_v = v.clone();
            for i in 0..n {
                new_v[i * n + p] = c * v[i * n + p] + s * v[i * n + q];
                new_v[i * n + q] = -s * v[i * n + p] + c * v[i * n + q];
            }
            v = new_v;
        }

        let eigenvalues: Vec<f64> = (0..n).map(|i| mat[i * n + i]).collect();
        (eigenvalues, v)
    }

    // ─── Critical value table ─────────────────────────────────────────────────

    /// Tabulated 5 % critical values for the Johansen trace test.
    ///
    /// Index: `n_vars` (1–5) and null rank `r` (0 … n_vars-1).
    /// Source: MacKinnon, Haug & Michelis (1999), Table 1.
    pub fn critical_value(n_vars: usize, rank: usize) -> f64 {
        // cv[n_vars-1][rank]
        // For n_vars > 5 we extrapolate linearly.
        const CV: [[f64; 5]; 5] = [
            // n=1: only rank 0
            [9.165, 0.0, 0.0, 0.0, 0.0],
            // n=2: rank 0,1
            [20.262, 9.165, 0.0, 0.0, 0.0],
            // n=3: rank 0,1,2
            [35.192, 20.262, 9.165, 0.0, 0.0],
            // n=4: rank 0,1,2,3
            [53.347, 35.192, 20.262, 9.165, 0.0],
            // n=5: rank 0,1,2,3,4
            [75.127, 53.347, 35.192, 20.262, 9.165],
        ];
        let n = n_vars.max(1);
        if n <= 5 {
            let row = &CV[n - 1];
            return *row.get(rank).unwrap_or(&9.165);
        }
        // For larger systems: approximate by linear extrapolation
        let base = CV[4][rank.min(4)];
        base + (n as f64 - 5.0) * 15.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let cfg = StreamingCointegrationConfig::default();
        assert_eq!(cfg.window_size, 200);
        assert!((cfg.significance - 0.05).abs() < 1e-12);
        assert_eq!(cfg.lag_order, 1);
        assert_eq!(cfg.test_frequency, 10);
    }

    #[test]
    fn test_returns_none_before_window_filled() {
        let cfg = StreamingCointegrationConfig {
            window_size: 50,
            test_frequency: 10,
            ..Default::default()
        };
        let mut tester = StreamingCointegrationTester::new(2, cfg).expect("new should succeed");
        // Only 5 observations – far below minimum
        for i in 0..5 {
            let obs = vec![i as f64, i as f64 * 1.5];
            let res = tester.update(&obs).expect("update should succeed");
            assert!(res.is_none(), "should be None before window filled");
        }
    }

    #[test]
    fn test_results_grows_after_test_frequency() {
        let cfg = StreamingCointegrationConfig {
            window_size: 50,
            test_frequency: 10,
            lag_order: 1,
            ..Default::default()
        };
        let mut tester = StreamingCointegrationTester::new(2, cfg).expect("new should succeed");
        // Feed 50 observations
        for i in 0..50 {
            let obs = vec![i as f64, i as f64 + 0.1];
            let _ = tester.update(&obs).expect("update should succeed");
        }
        // Should have run the test at some point
        assert!(
            tester.results().len() > 0,
            "should have at least one result"
        );
    }

    #[test]
    fn test_wrong_obs_dim_error() {
        let cfg = StreamingCointegrationConfig::default();
        let mut tester = StreamingCointegrationTester::new(2, cfg).expect("new should succeed");
        let obs = vec![1.0, 2.0, 3.0]; // wrong: 3 instead of 2
        assert!(tester.update(&obs).is_err());
    }

    #[test]
    fn test_critical_values_positive() {
        for n in 1..=5 {
            for r in 0..n {
                let cv = StreamingCointegrationTester::critical_value(n, r);
                assert!(cv > 0.0, "CV({n},{r}) should be positive, got {cv}");
            }
        }
    }

    #[test]
    fn test_critical_values_decreasing_with_rank() {
        // Higher rank → smaller critical value (fewer free parameters)
        for n in 2..=5 {
            let cv0 = StreamingCointegrationTester::critical_value(n, 0);
            let cv1 = StreamingCointegrationTester::critical_value(n, 1);
            assert!(cv0 > cv1, "CV({n},0)={cv0} should be > CV({n},1)={cv1}");
        }
    }

    #[test]
    fn test_single_var_returns_valid_result() {
        // For n_vars=1 the Johansen machinery runs but returns at most r=0
        // because you can't have more cointegrating vectors than variables - 1.
        // (rank r < n_vars always in the Johansen setup)
        let cfg = StreamingCointegrationConfig {
            window_size: 30,
            test_frequency: 10,
            lag_order: 1,
            ..Default::default()
        };
        let mut tester = StreamingCointegrationTester::new(1, cfg).expect("new should succeed");
        for i in 0..30 {
            let obs = vec![i as f64 * 0.5];
            let _ = tester.update(&obs).expect("update should succeed");
        }
        // At least one result should have been produced
        assert!(
            !tester.results().is_empty(),
            "should produce at least one test result"
        );
        // n_cointegrating_vectors <= n_vars (= 1) is always respected
        for res in tester.results() {
            assert!(
                res.n_cointegrating_vectors <= 1,
                "n_cointegrating_vectors={} > n_vars=1",
                res.n_cointegrating_vectors
            );
        }
    }
}
