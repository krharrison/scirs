//! Online Hierarchical Forecasting Reconciliation
//!
//! Provides incremental / online reconciliation of hierarchical time series
//! forecasts so that the forecasted values are *coherent* – i.e. they satisfy
//! the aggregation constraints encoded in the summing matrix **S**.
//!
//! # Supported methods
//!
//! | Method      | Description |
//! |-------------|-------------|
//! | `MinT`      | Minimum-trace reconciliation (Wickramasuriya et al. 2019) |
//! | `OLS`       | Ordinary-least-squares reconciliation |
//! | `BottomUp`  | Aggregate bottom-level forecasts upward |
//! | `TopDown`   | Disaggregate top-level forecast using historical proportions |
//!
//! # References
//!
//! - Wickramasuriya, S.L., Athanasopoulos, G. & Hyndman, R.J. (2019).
//!   "Optimal forecast reconciliation using unbiased estimating equations."
//!   *J. American Statistical Association*, 114(526):804–819.

use crate::error::{Result, TimeSeriesError};

// ─────────────────────────────────────────────────────────────────────────────
// Public enum: reconciliation method
// ─────────────────────────────────────────────────────────────────────────────

/// Method to use for online hierarchical reconciliation.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnlineReconcileMethod {
    /// Minimum Trace reconciliation (default).
    MinT,
    /// Bottom-up aggregation: aggregate bottom-level forecasts upward.
    BottomUp,
    /// Top-down proportional disaggregation.
    TopDown,
    /// Ordinary Least Squares reconciliation.
    OLS,
}

impl Default for OnlineReconcileMethod {
    fn default() -> Self {
        Self::MinT
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`OnlineReconciler`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct OnlineReconciliationConfig {
    /// Reconciliation method. Default: [`OnlineReconcileMethod::MinT`].
    pub method: OnlineReconcileMethod,
    /// Forgetting factor λ for exponential-weighted covariance estimation.
    /// Must be in `(0, 1]`. Default: `0.99`.
    pub forgetting_factor: f64,
    /// Tikhonov regularisation added to the diagonal of W before inversion.
    /// Default: `1e-6`.
    pub regularization: f64,
    /// Number of bottom-level series (leaves in the hierarchy). Default: `4`.
    pub n_bottom: usize,
    /// Total number of series (bottom + aggregate). Default: `7`.
    pub n_total: usize,
}

impl Default for OnlineReconciliationConfig {
    fn default() -> Self {
        Self {
            method: OnlineReconcileMethod::default(),
            forgetting_factor: 0.99,
            regularization: 1e-6,
            n_bottom: 4,
            n_total: 7,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Result type
// ─────────────────────────────────────────────────────────────────────────────

/// Outcome of a single reconciliation pass.
#[derive(Debug, Clone)]
pub struct ReconciliationResult {
    /// Reconciled forecasts (length = `n_total`).
    pub reconciled: Vec<f64>,
    /// Difference between the reconciled and the original base forecasts.
    pub residuals: Vec<f64>,
    /// Whether the reconciled forecasts satisfy the summing constraints
    /// (within a relative tolerance of 1e-6).
    pub coherent: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// OnlineReconciler
// ─────────────────────────────────────────────────────────────────────────────

/// Incrementally reconciles hierarchical forecasts.
///
/// The reconciler maintains a running estimate of the forecast-error
/// covariance matrix **W** using exponential weighting and applies one of
/// the supported reconciliation methods.
///
/// # Layout conventions
///
/// * The summing matrix **S** has shape `[n_total × n_bottom]`.  Row `i`
///   of **S** contains ones in the columns corresponding to the bottom-level
///   series that contribute to series `i`, and zeros elsewhere.  For a
///   bottom-level series `j`, the row `j` of **S** is the `j`-th standard
///   basis vector.
/// * Vectors are stored **row-major** as plain `Vec<f64>`.  A matrix element
///   at row `r`, column `c` in an `[rows × cols]` matrix is at index
///   `r * cols + c`.
pub struct OnlineReconciler {
    config: OnlineReconciliationConfig,
    /// S: [n_total × n_bottom]
    s_matrix: Vec<f64>,
    /// W: [n_total × n_total] running covariance estimate
    covariance: Vec<f64>,
    n_observations: usize,
}

impl OnlineReconciler {
    /// Create a new [`OnlineReconciler`].
    ///
    /// # Arguments
    ///
    /// * `config`   – reconciliation configuration.
    /// * `s_matrix` – flat row-major summing matrix of shape
    ///                `[n_total × n_bottom]`.  Must have exactly
    ///                `config.n_total * config.n_bottom` elements.
    ///
    /// # Errors
    ///
    /// Returns [`TimeSeriesError::InvalidInput`] when `s_matrix` has the
    /// wrong length.
    pub fn new(config: OnlineReconciliationConfig, s_matrix: Vec<f64>) -> Result<Self> {
        let expected = config.n_total * config.n_bottom;
        if s_matrix.len() != expected {
            return Err(TimeSeriesError::InvalidInput(format!(
                "s_matrix length {} != n_total({}) * n_bottom({})",
                s_matrix.len(),
                config.n_total,
                config.n_bottom,
            )));
        }
        let n = config.n_total;
        // Initialise W as a scaled identity matrix.
        let mut covariance = vec![0.0_f64; n * n];
        for i in 0..n {
            covariance[i * n + i] = 1.0;
        }
        Ok(Self {
            config,
            s_matrix,
            covariance,
            n_observations: 0,
        })
    }

    /// Incorporate a new forecast-error vector into the running covariance.
    ///
    /// Uses the exponential-forgetting update:
    /// `W ← λ W + (1-λ) e eᵀ`
    ///
    /// # Errors
    ///
    /// Returns [`TimeSeriesError::InvalidInput`] if `forecast_errors` has the
    /// wrong length.
    pub fn update(&mut self, forecast_errors: &[f64]) -> Result<()> {
        let n = self.config.n_total;
        if forecast_errors.len() != n {
            return Err(TimeSeriesError::InvalidInput(format!(
                "forecast_errors length {} != n_total {}",
                forecast_errors.len(),
                n,
            )));
        }
        let lam = self.config.forgetting_factor;
        let one_minus_lam = 1.0 - lam;
        for i in 0..n {
            for j in 0..n {
                let outer = forecast_errors[i] * forecast_errors[j];
                self.covariance[i * n + j] =
                    lam * self.covariance[i * n + j] + one_minus_lam * outer;
            }
        }
        self.n_observations += 1;
        Ok(())
    }

    /// Reconcile the base forecasts `y_hat` (length `n_total`).
    ///
    /// # Errors
    ///
    /// Returns [`TimeSeriesError::InvalidInput`] if `y_hat` has wrong length.
    pub fn reconcile(&self, y_hat: &[f64]) -> Result<ReconciliationResult> {
        let n = self.config.n_total;
        if y_hat.len() != n {
            return Err(TimeSeriesError::InvalidInput(format!(
                "y_hat length {} != n_total {}",
                y_hat.len(),
                n,
            )));
        }
        let reconciled = match self.config.method {
            OnlineReconcileMethod::MinT => self.mint_reconcile(y_hat),
            OnlineReconcileMethod::OLS => self.ols_reconcile(y_hat),
            OnlineReconcileMethod::BottomUp => self.bottom_up_reconcile(y_hat),
            OnlineReconcileMethod::TopDown => self.top_down_reconcile(y_hat),
            _ => self.mint_reconcile(y_hat),
        };
        let residuals = reconciled
            .iter()
            .zip(y_hat.iter())
            .map(|(r, h)| r - h)
            .collect::<Vec<_>>();
        let coherent = self.check_coherence(&reconciled);
        Ok(ReconciliationResult {
            reconciled,
            residuals,
            coherent,
        })
    }

    // ─── private helpers ────────────────────────────────────────────────────

    /// MinT reconciliation:
    /// `ỹ = S (SᵀW⁻¹S)⁻¹ Sᵀ W⁻¹ ŷ`
    fn mint_reconcile(&self, y_hat: &[f64]) -> Vec<f64> {
        let nt = self.config.n_total;
        let nb = self.config.n_bottom;

        // Regularised W
        let mut w_reg = self.covariance.clone();
        for i in 0..nt {
            w_reg[i * nt + i] += self.config.regularization;
        }

        // W⁻¹ via Cholesky / Gaussian elimination
        let w_inv = match Self::invert_symmetric(&w_reg, nt) {
            Some(m) => m,
            None => {
                // Fall back to identity if singular
                let mut id = vec![0.0_f64; nt * nt];
                for i in 0..nt {
                    id[i * nt + i] = 1.0;
                }
                id
            }
        };

        // A = Sᵀ W⁻¹ S  [nb × nb]
        let st_winv = Self::mat_mul_t_left(&self.s_matrix, &w_inv, nb, nt, nt, nt);
        let a = Self::mat_mul(&st_winv, &self.s_matrix, nb, nt, nt, nb);

        // A⁻¹
        let mut a_reg = a.clone();
        for i in 0..nb {
            a_reg[i * nb + i] += self.config.regularization;
        }
        let a_inv = match Self::invert_symmetric(&a_reg, nb) {
            Some(m) => m,
            None => {
                // degenerate case: use OLS fallback
                return self.ols_reconcile(y_hat);
            }
        };

        // b = W⁻¹ ŷ  [nt]
        let b = Self::mat_vec_mul(&w_inv, y_hat, nt, nt);

        // c = Sᵀ b  [nb]
        let c = Self::mat_t_vec_mul(&self.s_matrix, &b, nt, nb);

        // d = A⁻¹ c  [nb]  (bottom-level reconciled forecasts)
        let d = Self::mat_vec_mul(&a_inv, &c, nb, nb);

        // ỹ = S d  [nt]
        Self::mat_vec_mul(&self.s_matrix, &d, nt, nb)
    }

    /// OLS reconciliation:
    /// `ỹ = S (SᵀS)⁻¹ Sᵀ ŷ`
    fn ols_reconcile(&self, y_hat: &[f64]) -> Vec<f64> {
        let nt = self.config.n_total;
        let nb = self.config.n_bottom;

        // A = SᵀS  [nb × nb]
        let a = Self::mat_mul_t_left_identity(&self.s_matrix, nb, nt);

        let mut a_reg = a.clone();
        for i in 0..nb {
            a_reg[i * nb + i] += self.config.regularization;
        }
        let a_inv = match Self::invert_symmetric(&a_reg, nb) {
            Some(m) => m,
            None => {
                return self.bottom_up_reconcile(y_hat);
            }
        };

        // c = Sᵀ ŷ  [nb]
        let c = Self::mat_t_vec_mul(&self.s_matrix, y_hat, nt, nb);

        // d = A⁻¹ c  [nb]
        let d = Self::mat_vec_mul(&a_inv, &c, nb, nb);

        // ỹ = S d  [nt]
        Self::mat_vec_mul(&self.s_matrix, &d, nt, nb)
    }

    /// Bottom-up: take bottom-level forecasts and aggregate upward.
    fn bottom_up_reconcile(&self, y_hat: &[f64]) -> Vec<f64> {
        let nt = self.config.n_total;
        let nb = self.config.n_bottom;

        // Bottom-level forecasts are the last `nb` entries by convention,
        // but use S to identify them: row i of S has exactly one 1 in column i
        // for i < nb (diagonal block identity for bottom level).
        // Here we directly extract the bottom-level segment from y_hat.
        // If nt == nb, all rows are bottom level.
        let bottom: Vec<f64> = if nt > nb {
            // The bottom nb entries of y_hat are the bottom-level forecasts
            y_hat[nt - nb..].to_vec()
        } else {
            y_hat.to_vec()
        };

        // Aggregate: ỹ = S * bottom
        Self::mat_vec_mul(&self.s_matrix, &bottom, nt, nb)
    }

    /// Top-down: disaggregate total forecast using equal proportions.
    fn top_down_reconcile(&self, y_hat: &[f64]) -> Vec<f64> {
        let nt = self.config.n_total;
        let nb = self.config.n_bottom;

        // Total is y_hat[0]
        let total = y_hat[0];

        // Equal proportions (1/nb each) for bottom level
        let prop = 1.0 / nb as f64;
        let bottom: Vec<f64> = (0..nb).map(|_| total * prop).collect();

        // Aggregate upward
        Self::mat_vec_mul(&self.s_matrix, &bottom, nt, nb)
    }

    /// Check if the reconciled forecasts satisfy S * bottom == full.
    fn check_coherence(&self, reconciled: &[f64]) -> bool {
        let nt = self.config.n_total;
        let nb = self.config.n_bottom;
        if nt == 0 || nb == 0 || reconciled.len() != nt {
            return false;
        }
        // Extract bottom level
        let bottom: Vec<f64> = if nt > nb {
            reconciled[nt - nb..].to_vec()
        } else {
            reconciled.to_vec()
        };
        // S * bottom
        let recon_check = Self::mat_vec_mul(&self.s_matrix, &bottom, nt, nb);
        // Compare with reconciled
        let tol = 1e-6;
        for (a, b) in reconciled.iter().zip(recon_check.iter()) {
            let denom = a.abs().max(1.0);
            if (a - b).abs() / denom > tol {
                return false;
            }
        }
        true
    }

    // ─── matrix utilities ───────────────────────────────────────────────────

    /// Gaussian elimination to invert an `n×n` matrix.
    /// Returns `None` if singular.
    fn invert_symmetric(a: &[f64], n: usize) -> Option<Vec<f64>> {
        // Augmented matrix [A | I]
        let mut aug = vec![0.0_f64; n * n * 2];
        for i in 0..n {
            for j in 0..n {
                aug[i * 2 * n + j] = a[i * n + j];
            }
            aug[i * 2 * n + n + i] = 1.0;
        }
        for col in 0..n {
            // Find pivot
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
            // Swap rows
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

    /// Matrix multiply: C = A * B where A is [ra × ca], B is [rb × cb].
    /// ra must equal inner dimension used; ca == rb for compatibility.
    fn mat_mul(a: &[f64], b: &[f64], ra: usize, ca: usize, rb: usize, cb: usize) -> Vec<f64> {
        debug_assert_eq!(ca, rb, "inner dimensions must match");
        let _ = rb; // suppress unused warning
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

    /// Compute `Aᵀ * B` where A is [ra × ca], B is [ra × cb].
    /// Result is [ca × cb].
    fn mat_mul_t_left(
        a: &[f64],
        b: &[f64],
        ca: usize,
        ra: usize,
        _rb: usize,
        cb: usize,
    ) -> Vec<f64> {
        let mut c = vec![0.0_f64; ca * cb];
        for k in 0..ra {
            for i in 0..ca {
                let aki = a[k * ca + i];
                for j in 0..cb {
                    c[i * cb + j] += aki * b[k * cb + j];
                }
            }
        }
        c
    }

    /// Compute `Sᵀ S` where S is [nt × nb]. Result is [nb × nb].
    fn mat_mul_t_left_identity(s: &[f64], nb: usize, nt: usize) -> Vec<f64> {
        let mut c = vec![0.0_f64; nb * nb];
        for k in 0..nt {
            for i in 0..nb {
                let ski = s[k * nb + i];
                for j in 0..nb {
                    c[i * nb + j] += ski * s[k * nb + j];
                }
            }
        }
        c
    }

    /// Matrix-vector multiply: y = A x, A is [ra × ca].
    fn mat_vec_mul(a: &[f64], x: &[f64], ra: usize, ca: usize) -> Vec<f64> {
        let mut y = vec![0.0_f64; ra];
        for i in 0..ra {
            for j in 0..ca {
                y[i] += a[i * ca + j] * x[j];
            }
        }
        y
    }

    /// Matrix-transpose-vector multiply: y = Aᵀ x, A is [ra × ca].
    fn mat_t_vec_mul(a: &[f64], x: &[f64], ra: usize, ca: usize) -> Vec<f64> {
        let mut y = vec![0.0_f64; ca];
        for i in 0..ra {
            for j in 0..ca {
                y[j] += a[i * ca + j] * x[i];
            }
        }
        y
    }

    // ─── accessors ──────────────────────────────────────────────────────────

    /// Number of covariance updates applied so far.
    pub fn n_observations(&self) -> usize {
        self.n_observations
    }

    /// Current covariance matrix estimate (row-major, `[n_total × n_total]`).
    pub fn covariance(&self) -> &[f64] {
        &self.covariance
    }

    /// Configuration.
    pub fn config(&self) -> &OnlineReconciliationConfig {
        &self.config
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a simple 3-level hierarchy
    //   Total  (id 0)
    //   ├── A  (id 1)  = A1 + A2
    //   └── B  (id 2)  = B1 + B2
    //   A1, A2, B1, B2 are bottom level (ids 3,4,5,6 → columns 0,1,2,3)
    //
    // S: [7 × 4]
    //   row 0 (Total): [1,1,1,1]
    //   row 1 (A):     [1,1,0,0]
    //   row 2 (B):     [0,0,1,1]
    //   row 3 (A1):    [1,0,0,0]
    //   row 4 (A2):    [0,1,0,0]
    //   row 5 (B1):    [0,0,1,0]
    //   row 6 (B2):    [0,0,0,1]
    fn default_s_matrix() -> Vec<f64> {
        #[rustfmt::skip]
        let s = vec![
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 1.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        s
    }

    fn default_config() -> OnlineReconciliationConfig {
        OnlineReconciliationConfig {
            method: OnlineReconcileMethod::BottomUp,
            n_total: 7,
            n_bottom: 4,
            ..Default::default()
        }
    }

    #[test]
    fn test_config_default() {
        let cfg = OnlineReconciliationConfig::default();
        assert_eq!(cfg.n_bottom, 4);
        assert_eq!(cfg.n_total, 7);
        assert!((cfg.forgetting_factor - 0.99).abs() < 1e-10);
        assert_eq!(cfg.method, OnlineReconcileMethod::MinT);
    }

    #[test]
    fn test_bottom_up_aggregate() {
        let cfg = default_config();
        let s = default_s_matrix();
        let rec = OnlineReconciler::new(cfg, s).expect("new should succeed");

        // Bottom-level forecasts: A1=2, A2=3, B1=5, B2=7  → last 4 of y_hat
        let y_hat = vec![0.0_f64, 0.0, 0.0, 2.0, 3.0, 5.0, 7.0];
        let res = rec.reconcile(&y_hat).expect("reconcile should succeed");

        // Total = 2+3+5+7 = 17
        assert!((res.reconciled[0] - 17.0).abs() < 1e-9, "total mismatch");
        // A = 2+3 = 5
        assert!((res.reconciled[1] - 5.0).abs() < 1e-9, "A mismatch");
        // B = 5+7 = 12
        assert!((res.reconciled[2] - 12.0).abs() < 1e-9, "B mismatch");
    }

    #[test]
    fn test_mint_constraints_satisfied() {
        let cfg = OnlineReconciliationConfig {
            method: OnlineReconcileMethod::MinT,
            n_total: 7,
            n_bottom: 4,
            ..Default::default()
        };
        let s = default_s_matrix();
        let rec = OnlineReconciler::new(cfg, s).expect("new should succeed");

        let y_hat = vec![20.0_f64, 8.0, 12.0, 2.0, 3.0, 5.0, 7.0];
        let res = rec.reconcile(&y_hat).expect("reconcile should succeed");

        // Verify S-matrix constraint
        let n_total = 7;
        let n_bottom = 4;
        let s_mat = default_s_matrix();
        let bottom = &res.reconciled[n_total - n_bottom..];
        for (i, r_i) in res.reconciled.iter().enumerate() {
            let expected: f64 = (0..n_bottom)
                .map(|j| s_mat[i * n_bottom + j] * bottom[j])
                .sum();
            assert!(
                (r_i - expected).abs() < 1e-6,
                "constraint not satisfied at row {i}: {r_i} != {expected}"
            );
        }
        assert!(res.coherent);
    }

    #[test]
    fn test_ols_reconcile() {
        let cfg = OnlineReconciliationConfig {
            method: OnlineReconcileMethod::OLS,
            n_total: 7,
            n_bottom: 4,
            ..Default::default()
        };
        let s = default_s_matrix();
        let rec = OnlineReconciler::new(cfg, s).expect("new should succeed");

        let y_hat = vec![20.0_f64, 8.0, 12.0, 2.0, 3.0, 5.0, 7.0];
        let res = rec.reconcile(&y_hat).expect("reconcile should succeed");
        assert_eq!(res.reconciled.len(), 7);
        assert!(res.coherent);
    }

    #[test]
    fn test_update_changes_covariance() {
        let cfg = OnlineReconciliationConfig {
            method: OnlineReconcileMethod::MinT,
            n_total: 7,
            n_bottom: 4,
            ..Default::default()
        };
        let s = default_s_matrix();
        let mut rec = OnlineReconciler::new(cfg, s).expect("new should succeed");

        let cov_before: Vec<f64> = rec.covariance().to_vec();
        let errors = vec![0.5_f64, -0.3, 0.1, 0.2, -0.1, 0.4, -0.2];
        rec.update(&errors).expect("update should succeed");

        let cov_after = rec.covariance();
        // The off-diagonals should now be non-zero
        assert!(
            cov_before[0 * 7 + 1] != cov_after[0 * 7 + 1],
            "off-diagonal should change"
        );
        assert_eq!(rec.n_observations(), 1);
    }

    #[test]
    fn test_coherent_flag() {
        let cfg = OnlineReconciliationConfig {
            method: OnlineReconcileMethod::BottomUp,
            n_total: 7,
            n_bottom: 4,
            ..Default::default()
        };
        let s = default_s_matrix();
        let rec = OnlineReconciler::new(cfg, s).expect("new should succeed");

        let y_hat = vec![0.0_f64, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let res = rec.reconcile(&y_hat).expect("reconcile should succeed");
        assert!(res.coherent, "bottom-up should always be coherent");
    }
}
