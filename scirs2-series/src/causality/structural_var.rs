//! Structural Vector Autoregression (SVAR) Identification
//!
//! This module implements the Structural VAR model with three identification
//! schemes:
//!
//! - **Cholesky** (recursive): Lower-triangular impact matrix via Cholesky
//!   decomposition of the residual covariance.
//! - **LongRun**: Blanchard-Quah long-run restrictions; structural shocks have
//!   no long-run effect on certain variables.
//! - **SignRestriction**: Sign restrictions placed on IRF impact effects.
//!
//! ## Pipeline
//!
//! 1. Estimate the reduced-form VAR using OLS (`fit_var`).
//! 2. Apply the chosen identification scheme to recover the structural impact
//!    matrix **B** such that the reduced-form residuals u = B ε, where ε are
//!    orthogonal structural shocks.
//! 3. Compute Impulse Response Functions (IRF) and Forecast Error Variance
//!    Decomposition (FEVD).
//!
//! ## References
//!
//! - Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
//!   Springer, Berlin.
//! - Blanchard, O. & Quah, D. (1989). "The Dynamic Effects of Aggregate Demand
//!   and Aggregate Supply Disturbances." *AER*, 79(4):655–673.

use crate::error::TimeSeriesError;

use super::CausalityResult;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Identification scheme for Structural VAR
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvarIdent {
    /// Recursive (Cholesky) identification — lower-triangular impact matrix
    Cholesky,
    /// Long-run restrictions (Blanchard-Quah style)
    LongRun,
    /// Sign restrictions on impact effects
    SignRestriction,
}

impl Default for SvarIdent {
    fn default() -> Self {
        Self::Cholesky
    }
}

/// Configuration for the Structural VAR model
#[derive(Debug, Clone)]
pub struct SvarConfig {
    /// VAR lag order p
    pub max_lag: usize,
    /// Identification scheme
    pub identification: SvarIdent,
    /// IRF horizon (number of steps ahead)
    pub irf_horizon: usize,
    /// Sign restrictions matrix (n_vars × n_vars), entry is +1/−1/0 (0 = no restriction).
    /// Only used when `identification == SignRestriction`.
    pub sign_restrictions: Option<Vec<Vec<f64>>>,
}

impl Default for SvarConfig {
    fn default() -> Self {
        Self {
            max_lag: 1,
            identification: SvarIdent::Cholesky,
            irf_horizon: 12,
            sign_restrictions: None,
        }
    }
}

/// Result of Structural VAR estimation
#[derive(Debug, Clone)]
pub struct SvarResult {
    /// Number of variables
    pub n_vars: usize,
    /// VAR lag order used
    pub lag_order: usize,
    /// Reduced-form VAR coefficient matrices A_1 … A_p (p × n_vars × n_vars)
    pub a_matrices: Vec<Vec<Vec<f64>>>,
    /// Residual covariance matrix Σ (n_vars × n_vars)
    pub sigma: Vec<Vec<f64>>,
    /// Structural impact matrix B (n_vars × n_vars), u = B ε
    pub b_matrix: Vec<Vec<f64>>,
    /// Impulse Response Functions — `irf[h][i][j]` = response of variable i to
    /// shock j at horizon h (shape: (horizon+1) × n × n)
    pub irf: Vec<Vec<Vec<f64>>>,
    /// Forecast Error Variance Decomposition — `fevd[i][j]` = fraction of
    /// variable i's forecast error variance attributable to shock j at horizon
    /// `irf_horizon`.
    pub fevd: Vec<Vec<f64>>,
}

/// SVAR model that wraps estimation and identification
pub struct SvarModel {
    config: SvarConfig,
}

impl SvarModel {
    /// Create a new SvarModel with the given configuration
    pub fn new(config: SvarConfig) -> Self {
        Self { config }
    }

    /// Fit the Structural VAR to data
    ///
    /// # Arguments
    /// * `data` — T × n_vars matrix stored as `Vec<Vec<f64>>` (outer: time steps, inner: vars)
    ///
    /// # Returns
    /// `SvarResult` containing A matrices, B matrix, IRF, and FEVD
    pub fn fit(&self, data: &[Vec<f64>]) -> CausalityResult<SvarResult> {
        let t = data.len();
        if t == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "Data must be non-empty".to_string(),
            ));
        }
        let n = data[0].len();
        if n == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "Data must have at least one variable".to_string(),
            ));
        }
        let p = self.config.max_lag;
        if t <= p + n {
            return Err(TimeSeriesError::InsufficientData {
                message: "Too few observations for VAR estimation".to_string(),
                required: p + n + 1,
                actual: t,
            });
        }

        // ── Step 1: Estimate reduced-form VAR via OLS ──────────────────────
        let (a_matrices, sigma) = fit_var(data, p)?;

        // ── Step 2: Identify structural impact matrix B ────────────────────
        let b_matrix = match self.config.identification {
            SvarIdent::Cholesky => cholesky_identification(&sigma)?,
            SvarIdent::LongRun => long_run_identification(&a_matrices, &sigma, p)?,
            SvarIdent::SignRestriction => {
                let restrictions = self.config.sign_restrictions.as_deref().ok_or_else(|| {
                    TimeSeriesError::InvalidInput(
                        "sign_restrictions must be provided for SignRestriction identification"
                            .to_string(),
                    )
                })?;
                sign_restriction_identification(&sigma, restrictions)?
            }
        };

        // ── Step 3: Compute IRF ────────────────────────────────────────────
        let horizon = self.config.irf_horizon;
        let irf = compute_irf(&a_matrices, &b_matrix, n, p, horizon)?;

        // ── Step 4: Compute FEVD ───────────────────────────────────────────
        let fevd = compute_fevd(&irf, n, horizon)?;

        Ok(SvarResult {
            n_vars: n,
            lag_order: p,
            a_matrices,
            sigma,
            b_matrix,
            irf,
            fevd,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Reduced-form VAR estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate a reduced-form VAR(p) model by OLS.
///
/// Returns `(A_matrices, Sigma)` where `A_matrices[k]` is the k-th lag coefficient
/// matrix (n × n) and Σ is the residual covariance matrix (n × n).
pub fn fit_var(
    data: &[Vec<f64>],
    p: usize,
) -> CausalityResult<(Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>)> {
    let t = data.len();
    let n = data[0].len();
    let n_eff = t - p; // usable observations (p+1 … T)
    let k = n * p; // total regressors per equation (no constant for simplicity)

    if n_eff < k + 1 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Too few effective observations for OLS".to_string(),
            required: k + 2,
            actual: n_eff,
        });
    }

    // Build design matrix X (n_eff × k): rows are [y_{t-1}', …, y_{t-p}']
    let mut x_mat = vec![vec![0.0_f64; k]; n_eff];
    let mut y_mat = vec![vec![0.0_f64; n]; n_eff];

    for row in 0..n_eff {
        let t_now = row + p;
        // Response
        for j in 0..n {
            y_mat[row][j] = data[t_now][j];
        }
        // Regressors: lag 1 … p
        for lag in 1..=p {
            let t_lag = t_now - lag;
            for j in 0..n {
                x_mat[row][(lag - 1) * n + j] = data[t_lag][j];
            }
        }
    }

    // For each equation i, solve: Y_i = X β_i using normal equations X'X β = X'Y_i
    let xtx = mat_mul_t(&x_mat, &x_mat, n_eff, k, k); // k × k
    let mut a_flat = vec![vec![0.0_f64; k]; n]; // n × k

    for i in 0..n {
        let xty: Vec<f64> = (0..k)
            .map(|jj| (0..n_eff).map(|row| x_mat[row][jj] * y_mat[row][i]).sum())
            .collect();
        let beta = solve_linear_system_vec(&xtx, &xty)?;
        a_flat[i] = beta;
    }

    // Reshape a_flat into p lag matrices (n × n each)
    let mut a_matrices: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; n]; n]; p];
    for lag_idx in 0..p {
        for i in 0..n {
            for j in 0..n {
                a_matrices[lag_idx][i][j] = a_flat[i][lag_idx * n + j];
            }
        }
    }

    // Compute residuals and covariance Σ
    let mut residuals = vec![vec![0.0_f64; n]; n_eff];
    for row in 0..n_eff {
        for i in 0..n {
            let mut pred = 0.0_f64;
            for jj in 0..k {
                pred += x_mat[row][jj] * a_flat[i][jj];
            }
            residuals[row][i] = y_mat[row][i] - pred;
        }
    }

    let denom = (n_eff - k) as f64;
    let denom = if denom > 0.0 { denom } else { 1.0 };
    let mut sigma = vec![vec![0.0_f64; n]; n];
    for row in 0..n_eff {
        for i in 0..n {
            for j in 0..n {
                sigma[i][j] += residuals[row][i] * residuals[row][j];
            }
        }
    }
    for i in 0..n {
        for j in 0..n {
            sigma[i][j] /= denom;
        }
    }

    Ok((a_matrices, sigma))
}

// ─────────────────────────────────────────────────────────────────────────────
// Identification schemes
// ─────────────────────────────────────────────────────────────────────────────

/// Cholesky (recursive) identification.
///
/// Returns B = chol(Σ) (lower triangular).
pub fn cholesky_identification(sigma: &[Vec<f64>]) -> CausalityResult<Vec<Vec<f64>>> {
    cholesky_decomp(sigma)
}

/// Long-run identification (Blanchard-Quah).
///
/// The long-run impact matrix C = (I - A_1 - … - A_p)^{-1} B has a
/// lower-triangular structure. We solve for B from chol(C Σ C') ≈ C B B' C'.
pub fn long_run_identification(
    a_matrices: &[Vec<Vec<f64>>],
    sigma: &[Vec<f64>],
    _p: usize,
) -> CausalityResult<Vec<Vec<f64>>> {
    let n = sigma.len();
    // Compute (I - A_1 - … - A_p)
    let mut sum_a = vec![vec![0.0_f64; n]; n];
    for a in a_matrices {
        for i in 0..n {
            for j in 0..n {
                sum_a[i][j] += a[i][j];
            }
        }
    }
    let mut c_inv = vec![vec![0.0_f64; n]; n]; // I - sum_a
    for i in 0..n {
        for j in 0..n {
            c_inv[i][j] = if i == j { 1.0 } else { 0.0 } - sum_a[i][j];
        }
    }
    // C = c_inv^{-1}
    let c = mat_inverse(&c_inv)?;
    // Long-run covariance: C Σ C'
    let csigct = mat_mul_matmul(&c, sigma, &transpose(&c), n);
    // Cholesky of long-run covariance
    let chol_lr = cholesky_decomp(&csigct)?;
    // B = C^{-1} chol_lr  (so that C B = chol_lr → lower triangular)
    let b = mat_mul_mat(&c_inv, &chol_lr, n);
    Ok(b)
}

/// Sign restriction identification: iterative rotation search.
///
/// Draws random orthogonal matrices Q via QR decomposition of random normal
/// matrices, checks sign restrictions B0 = chol(Σ) Q, and returns the first
/// Q that satisfies all specified restrictions.
pub fn sign_restriction_identification(
    sigma: &[Vec<f64>],
    restrictions: &[Vec<f64>],
) -> CausalityResult<Vec<Vec<f64>>> {
    let n = sigma.len();
    let b0 = cholesky_decomp(sigma)?;

    let mut lcg_state: u64 = 0xdeadbeef_cafe1234;
    let max_draws = 10_000usize;

    for _ in 0..max_draws {
        // Generate random normal n×n matrix using LCG
        let mut rand_mat = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                rand_mat[i][j] = lcg_normal(&mut lcg_state);
            }
        }
        // QR decomposition via Gram-Schmidt
        let q = gram_schmidt(&rand_mat, n);
        // Candidate B = B0 Q
        let b_cand = mat_mul_mat(&b0, &q, n);
        // Check sign restrictions
        if check_sign_restrictions(&b_cand, restrictions) {
            return Ok(b_cand);
        }
    }
    // Fallback to Cholesky if no valid rotation found
    Ok(b0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Impulse Response Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Impulse Response Functions (IRF).
///
/// Φ_0 = B (structural impact)
/// Φ_h = Σ_{k=1}^{min(h,p)} A_k Φ_{h-k}  for h ≥ 1
///
/// Returns `irf[h][i][j]` = response of variable i to shock j at horizon h.
/// Shape: (horizon+1) × n × n.
pub fn compute_irf(
    a_matrices: &[Vec<Vec<f64>>],
    b_matrix: &[Vec<f64>],
    n: usize,
    p: usize,
    horizon: usize,
) -> CausalityResult<Vec<Vec<Vec<f64>>>> {
    let mut phi: Vec<Vec<Vec<f64>>> = Vec::with_capacity(horizon + 1);
    // Φ_0 = B
    phi.push(b_matrix.to_vec());

    for h in 1..=horizon {
        let mut phi_h = vec![vec![0.0_f64; n]; n];
        let max_k = h.min(p);
        for k in 1..=max_k {
            // phi_h += A_k @ phi[h-k]
            let phi_prev = &phi[h - k];
            for i in 0..n {
                for j in 0..n {
                    let mut acc = 0.0_f64;
                    for m in 0..n {
                        acc += a_matrices[k - 1][i][m] * phi_prev[m][j];
                    }
                    phi_h[i][j] += acc;
                }
            }
        }
        phi.push(phi_h);
    }

    Ok(phi)
}

// ─────────────────────────────────────────────────────────────────────────────
// Forecast Error Variance Decomposition
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Forecast Error Variance Decomposition (FEVD).
///
/// `MSE_h[i] = Sum_{k=0}^h Sum_j Phi_k[i,j]^2`
/// `FEVD[i,j] = (Sum_{k=0}^h Phi_k[i,j]^2) / MSE_h[i]`
///
/// Returns `fevd[i][j]` at the specified horizon.
pub fn compute_fevd(
    irf: &[Vec<Vec<f64>>],
    n: usize,
    horizon: usize,
) -> CausalityResult<Vec<Vec<f64>>> {
    let h_max = horizon.min(irf.len().saturating_sub(1));

    // Accumulate squared IRF contributions
    let mut contrib = vec![vec![0.0_f64; n]; n]; // contrib[i][j]
    for h in 0..=h_max {
        for i in 0..n {
            for j in 0..n {
                let phi_hij = irf[h][i][j];
                contrib[i][j] += phi_hij * phi_hij;
            }
        }
    }

    // FEVD[i][j] = contrib[i][j] / Σ_j contrib[i][j]
    let mut fevd = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        let mse_i: f64 = contrib[i].iter().sum();
        if mse_i.abs() > f64::EPSILON {
            for j in 0..n {
                fevd[i][j] = contrib[i][j] / mse_i;
            }
        } else {
            // Degenerate: uniform attribution
            let uniform = 1.0 / n as f64;
            for j in 0..n {
                fevd[i][j] = uniform;
            }
        }
    }

    Ok(fevd)
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear algebra helpers (pure Rust, no ndarray/rand)
// ─────────────────────────────────────────────────────────────────────────────

/// Cholesky decomposition — returns lower triangular L such that L L' = A.
pub fn cholesky_decomp(a: &[Vec<f64>]) -> CausalityResult<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0_f64; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0_f64;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }
            if i == j {
                let diag_sq = a[i][i] - sum;
                if diag_sq < 0.0 {
                    // Add small regularization for numerical stability
                    let reg = diag_sq.abs() + 1e-10;
                    l[i][j] = reg.sqrt();
                } else {
                    l[i][j] = diag_sq.sqrt();
                }
            } else {
                let ljj = l[j][j];
                l[i][j] = if ljj.abs() > f64::EPSILON {
                    (a[i][j] - sum) / ljj
                } else {
                    0.0
                };
            }
        }
    }

    Ok(l)
}

/// Solve a lower-triangular system Lx = b (forward substitution).
fn forward_substitution(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i][j] * x[j];
        }
        x[i] = if l[i][i].abs() > f64::EPSILON {
            s / l[i][i]
        } else {
            0.0
        };
    }
    x
}

/// Solve upper-triangular system U x = b (back substitution).
fn back_substitution(u: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= u[i][j] * x[j];
        }
        x[i] = if u[i][i].abs() > f64::EPSILON {
            s / u[i][i]
        } else {
            0.0
        };
    }
    x
}

/// Matrix inverse via Cholesky (for symmetric positive-definite).
/// Falls back to Gauss-Jordan for general matrices.
fn mat_inverse(a: &[Vec<f64>]) -> CausalityResult<Vec<Vec<f64>>> {
    let n = a.len();
    // Gauss-Jordan elimination with partial pivoting
    let mut aug = vec![vec![0.0_f64; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-14 {
            return Err(TimeSeriesError::NumericalInstability(
                "Singular matrix in mat_inverse".to_string(),
            ));
        }

        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..(2 * n) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    let mut inv = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }
    Ok(inv)
}

/// Transpose a square matrix
fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = if n > 0 { a[0].len() } else { 0 };
    let mut t = vec![vec![0.0_f64; n]; m];
    for i in 0..n {
        for j in 0..m {
            t[j][i] = a[i][j];
        }
    }
    t
}

/// Matrix multiply A (n×m) @ B (m×k) → (n×k)
fn mat_mul_mat(a: &[Vec<f64>], b: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let m = b.len();
    let k = if m > 0 { b[0].len() } else { 0 };
    let a_rows = a.len();
    let mut c = vec![vec![0.0_f64; k]; a_rows];
    for i in 0..a_rows {
        for j in 0..k {
            for l in 0..n.min(m) {
                c[i][j] += a[i][l] * b[l][j];
            }
        }
    }
    c
}

/// Compute X' X from row-major X (n_rows × n_cols), result is (n_cols × n_cols)
fn mat_mul_t(
    x: &[Vec<f64>],
    _x2: &[Vec<f64>],
    n_rows: usize,
    n_cols: usize,
    _k: usize,
) -> Vec<Vec<f64>> {
    let mut xtx = vec![vec![0.0_f64; n_cols]; n_cols];
    for row in 0..n_rows {
        for i in 0..n_cols {
            for j in 0..n_cols {
                xtx[i][j] += x[row][i] * x[row][j];
            }
        }
    }
    xtx
}

/// Matrix triple product A B C (all n×n)
fn mat_mul_matmul(a: &[Vec<f64>], b: &[Vec<f64>], c: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let ab = mat_mul_mat(a, b, n);
    mat_mul_mat(&ab, c, n)
}

/// Solve n×n linear system via Cholesky-based forward/back substitution.
/// Falls back to Gauss elimination for non-SPD matrices.
pub fn solve_linear_system_vec(a: &[Vec<f64>], b: &[f64]) -> CausalityResult<Vec<f64>> {
    let n = a.len();
    if n == 0 || b.len() != n {
        return Err(TimeSeriesError::InvalidInput(
            "Incompatible dimensions in linear system".to_string(),
        ));
    }

    // Gauss elimination with partial pivoting
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            row.push(b[i]);
            row
        })
        .collect();

    for col in 0..n {
        // Find max pivot
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-14 {
            // Regularize and continue
            aug[col][col] = 1e-10;
        }
        let pivot = aug[col][col];

        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for j in col..(n + 1) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = aug[i][n];
        for j in (i + 1)..n {
            s -= aug[i][j] * x[j];
        }
        x[i] = if aug[i][i].abs() > 1e-14 {
            s / aug[i][i]
        } else {
            0.0
        };
    }

    Ok(x)
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers for sign restriction identification
// ─────────────────────────────────────────────────────────────────────────────

/// LCG pseudo-random normal variate via Box-Muller
fn lcg_normal(state: &mut u64) -> f64 {
    fn lcg_uniform(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }
    let u1 = lcg_uniform(state).max(1e-15);
    let u2 = lcg_uniform(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Gram-Schmidt QR decomposition, returns orthogonal Q (n×n).
fn gram_schmidt(a: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let mut q: Vec<Vec<f64>> = Vec::with_capacity(n);

    for j in 0..n {
        let mut v: Vec<f64> = (0..n).map(|i| a[i][j]).collect();
        for qk in &q {
            let proj: f64 = v.iter().zip(qk.iter()).map(|(vi, qi)| vi * qi).sum();
            for i in 0..n {
                v[i] -= proj * qk[i];
            }
        }
        let norm: f64 = v.iter().map(|vi| vi * vi).sum::<f64>().sqrt();
        if norm > 1e-14 {
            q.push(v.iter().map(|vi| vi / norm).collect());
        } else {
            // Degenerate column: use standard basis vector
            let mut e = vec![0.0_f64; n];
            e[j] = 1.0;
            q.push(e);
        }
    }

    // Convert from column-wise to row-wise storage (transpose for mat_mul_mat)
    let mut q_mat = vec![vec![0.0_f64; n]; n];
    for j in 0..n {
        for i in 0..n {
            q_mat[i][j] = q[j][i];
        }
    }
    q_mat
}

/// Check that sign restrictions are satisfied.
/// `restrictions[i][j]` is +1 (positive), −1 (negative), or 0 (no constraint).
fn check_sign_restrictions(b: &[Vec<f64>], restrictions: &[Vec<f64>]) -> bool {
    let n = b.len().min(restrictions.len());
    for i in 0..n {
        let m = b[i].len().min(restrictions[i].len());
        for j in 0..m {
            let r = restrictions[i][j];
            if r > 0.5 && b[i][j] <= 0.0 {
                return false;
            }
            if r < -0.5 && b[i][j] >= 0.0 {
                return false;
            }
        }
    }
    true
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_rand(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    }

    /// Generate AR(1) bivariate data: x_t = 0.7 x_{t-1} + ε, y_t = 0.5 x_{t-1} + 0.3 y_{t-1} + ε
    fn gen_bivar_ar1(n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut data = vec![vec![0.0_f64; 2]; n];
        let mut state = seed;
        for t in 1..n {
            let e1 = lcg_rand(&mut state) * 0.2;
            let e2 = lcg_rand(&mut state) * 0.2;
            data[t][0] = 0.7 * data[t - 1][0] + e1;
            data[t][1] = 0.5 * data[t - 1][0] + 0.3 * data[t - 1][1] + e2;
        }
        data
    }

    #[test]
    fn test_cholesky_decomp_correct() {
        // 2x2 positive definite matrix
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let l = cholesky_decomp(&a).expect("cholesky failed");
        // Verify L L' = A
        let n = 2;
        for i in 0..n {
            for j in 0..n {
                let val: f64 = (0..n).map(|k| l[i][k] * l[j][k]).sum();
                assert!(
                    (val - a[i][j]).abs() < 1e-10,
                    "L L'[{i},{j}] = {val} != a[{i},{j}] = {}",
                    a[i][j]
                );
            }
        }
    }

    #[test]
    fn test_cholesky_identity() {
        let id = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let l = cholesky_decomp(&id).expect("cholesky failed");
        assert!((l[0][0] - 1.0).abs() < 1e-10);
        assert!((l[1][1] - 1.0).abs() < 1e-10);
        assert!(l[1][0].abs() < 1e-10);
        assert!(l[0][1].abs() < 1e-10);
    }

    #[test]
    fn test_var_fit_ar1() {
        let data = gen_bivar_ar1(500, 42);
        let (a_mats, _sigma) = fit_var(&data, 1).expect("fit_var failed");
        assert_eq!(a_mats.len(), 1);
        // A[0][0][0] should be near 0.7 (self-AR coefficient of X)
        let a11 = a_mats[0][0][0];
        assert!((a11 - 0.7).abs() < 0.1, "A[1][0,0] = {a11}, expected ≈ 0.7");
        // A[0][1][0] should be near 0.5 (cross-lag Y on X)
        let a21 = a_mats[0][1][0];
        assert!(
            (a21 - 0.5).abs() < 0.15,
            "A[1][1,0] = {a21}, expected ≈ 0.5"
        );
    }

    #[test]
    fn test_irf_impact_effect() {
        // IRF[0] should equal B
        let b = vec![vec![1.0, 0.0], vec![0.5, 1.0]];
        let a = vec![vec![vec![0.3, 0.0], vec![0.1, 0.2]]];
        let irf = compute_irf(&a, &b, 2, 1, 5).expect("irf failed");
        // impact (h=0)
        assert!((irf[0][0][0] - b[0][0]).abs() < 1e-10);
        assert!((irf[0][1][0] - b[1][0]).abs() < 1e-10);
    }

    #[test]
    fn test_irf_zero_at_horizon() {
        // Very stable VAR — IRF should decay toward zero
        let b = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let a = vec![vec![vec![0.1, 0.0], vec![0.0, 0.1]]];
        let irf = compute_irf(&a, &b, 2, 1, 30).expect("irf failed");
        // At h=30 the responses should be tiny
        let max_abs = irf[30]
            .iter()
            .flat_map(|row| row.iter())
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        assert!(max_abs < 0.5, "IRF should decay: max = {max_abs}");
    }

    #[test]
    fn test_fevd_sums_to_one() {
        let b = vec![vec![1.0, 0.0], vec![0.5, 1.0]];
        let a = vec![vec![vec![0.5, 0.1], vec![0.0, 0.4]]];
        let irf = compute_irf(&a, &b, 2, 1, 12).expect("irf failed");
        let fevd = compute_fevd(&irf, 2, 12).expect("fevd failed");
        for i in 0..2 {
            let sum: f64 = fevd[i].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "FEVD row {i} sums to {sum}, expected 1"
            );
        }
    }

    #[test]
    fn test_svar_cholesky_ident() {
        // B should be lower triangular
        let data = gen_bivar_ar1(300, 7);
        let config = SvarConfig {
            max_lag: 1,
            identification: SvarIdent::Cholesky,
            irf_horizon: 8,
            sign_restrictions: None,
        };
        let model = SvarModel::new(config);
        let result = model.fit(&data).expect("svar fit failed");
        // Upper-right entry should be zero (lower triangular)
        assert!(
            result.b_matrix[0][1].abs() < 1e-10,
            "B[0,1] = {} should be zero for Cholesky",
            result.b_matrix[0][1]
        );
    }

    #[test]
    fn test_svar_full_pipeline() {
        let data = gen_bivar_ar1(400, 99);
        let config = SvarConfig::default();
        let model = SvarModel::new(config);
        let result = model.fit(&data).expect("svar full pipeline failed");
        assert_eq!(result.n_vars, 2);
        assert!(!result.irf.is_empty());
        // FEVD rows sum to 1
        for i in 0..2 {
            let sum: f64 = result.fevd[i].iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }
}
