//! Scalar-on-Function Regression (Functional Linear Model)
//!
//! Models the relationship between a scalar response Y and a functional
//! predictor X(t):
//!
//!   Y_i = α + ∫ β(t) X_i(t) dt + ε_i
//!
//! The unknown coefficient function β(t) is expanded in a basis:
//!
//!   β(t) = Σ_k c_k φ_k(t)
//!
//! so the integral becomes Z_ij = ∫ X_i(t) φ_j(t) dt, giving
//!
//!   Y = Z c + ε
//!
//! with a roughness penalty λ ∫ [β''(t)]² dt = λ c' Ω c.
//!
//! # Basis choices
//!
//! | Variant | Details |
//! |---------|---------|
//! | `BSpline{n_basis, degree}` | B-splines via de Boor recursion, equidistant knots |
//! | `Fourier{n_basis}` | Fourier (sin/cos) basis on \[0,1\] |
//! | `Wavelet{n_basis}` | Haar wavelet basis (power-of-two levels) |
//!
//! # References
//!
//! - Ramsay, J.O. & Silverman, B.W. (2005). *Functional Data Analysis* (2nd ed.). Springer.
//! - Cardot, H., Ferraty, F. & Sarda, P. (1999). Functional linear model.
//!   *Statistics & Probability Letters* 45, 11-22.

use crate::error::{StatsError, StatsResult};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Basis used to represent the coefficient function β(t).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionalBasis {
    /// B-spline basis with `n_basis` functions of degree `degree`.
    BSpline {
        /// Number of B-spline basis functions.
        n_basis: usize,
        /// Polynomial degree (e.g., 3 for cubic splines).
        degree: usize,
    },
    /// Fourier (trigonometric) basis with `n_basis` functions on [0, 1].
    Fourier {
        /// Number of Fourier basis functions (should be odd for symmetry).
        n_basis: usize,
    },
    /// Haar wavelet basis with `n_basis` functions (must be power of two).
    Wavelet {
        /// Number of wavelet basis functions.
        n_basis: usize,
    },
}

/// Configuration for functional regression.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct FofConfig {
    /// Basis for β(t) (default: cubic B-spline with 10 functions).
    pub basis: FunctionalBasis,
    /// Roughness penalty weight λ ≥ 0 (default: 0.01).
    pub lambda: f64,
    /// Number of quadrature grid points for ∫ X_i(t) φ_j(t) dt (default: 100).
    pub n_grid: usize,
}

impl Default for FofConfig {
    fn default() -> Self {
        Self {
            basis: FunctionalBasis::BSpline {
                n_basis: 10,
                degree: 3,
            },
            lambda: 0.01,
            n_grid: 100,
        }
    }
}

/// Result of fitting a scalar-on-function regression.
#[derive(Debug, Clone)]
pub struct FofResult {
    /// Basis expansion coefficients for β(t).
    pub beta_coefs: Vec<f64>,
    /// β(t) evaluated at the internal grid.
    pub beta_values: Vec<f64>,
    /// Time grid at which β is evaluated.
    pub grid: Vec<f64>,
    /// Intercept α.
    pub intercept: f64,
    /// Coefficient of determination R².
    pub r_squared: f64,
    /// Generalised cross-validation score GCV(λ).
    pub gcv_score: f64,
}

/// Scalar-on-function regression estimator.
#[derive(Debug, Clone)]
pub struct FunctionalRegression {
    config: FofConfig,
    /// Stored basis coefficients after fitting (None before fit).
    beta_coefs: Option<Vec<f64>>,
    /// Fitted intercept.
    intercept: Option<f64>,
    /// Grid used during fit (for prediction).
    fit_grid: Option<Vec<f64>>,
}

impl FunctionalRegression {
    /// Create a new estimator with the given configuration.
    pub fn new(config: FofConfig) -> Self {
        Self {
            config,
            beta_coefs: None,
            intercept: None,
            fit_grid: None,
        }
    }

    /// Fit the model.
    ///
    /// # Arguments
    ///
    /// * `data`     – observed functional predictors, shape `(n_obs, n_time)`.
    ///                Each row is a discretised curve X_i(t).
    /// * `response` – scalar response values, length `n_obs`.
    /// * `grid`     – time points at which the curves are observed, length `n_time`.
    ///
    /// # Errors
    ///
    /// Returns `StatsError` when dimensions are inconsistent, `n_time < 2`,
    /// `n_obs < n_basis + 1`, or the penalised system is singular.
    pub fn fit(
        &mut self,
        data: &[Vec<f64>],
        response: &[f64],
        grid: &[f64],
    ) -> StatsResult<FofResult> {
        let n_obs = data.len();
        if n_obs == 0 {
            return Err(StatsError::InsufficientData(
                "need at least one observation".to_owned(),
            ));
        }
        let n_time = grid.len();
        if n_time < 2 {
            return Err(StatsError::InvalidArgument(
                "grid must have at least 2 points".to_owned(),
            ));
        }
        if response.len() != n_obs {
            return Err(StatsError::DimensionMismatch(format!(
                "response length {} != n_obs {}",
                response.len(),
                n_obs
            )));
        }
        for (i, row) in data.iter().enumerate() {
            if row.len() != n_time {
                return Err(StatsError::DimensionMismatch(format!(
                    "data[{}] has {} time points, expected {}",
                    i,
                    row.len(),
                    n_time
                )));
            }
        }

        let n_basis = self.n_basis_fns();
        if n_obs < n_basis + 1 {
            return Err(StatsError::InsufficientData(format!(
                "need n_obs >= n_basis+1 = {} but got {}",
                n_basis + 1,
                n_obs
            )));
        }

        // --- 1. Evaluate basis at observation grid -------------------------
        let phi = self.evaluate_basis(grid); // shape: n_time × n_basis

        // --- 2. Build Z matrix (n_obs × n_basis): Z_ij = ∫ X_i(t) φ_j(t) dt
        //        Using trapezoidal rule over the observation grid.
        let z = build_z_matrix(data, &phi, grid); // n_obs × n_basis

        // --- 3. Roughness penalty matrix Ω (n_basis × n_basis) ------------
        let omega = self.roughness_penalty(n_basis);

        // --- 4. Centre the response and add intercept ---------------------
        let y_mean = response.iter().sum::<f64>() / n_obs as f64;
        let y_centred: Vec<f64> = response.iter().map(|&y| y - y_mean).collect();

        // Centre Z columns too (to decorrelate intercept from slopes)
        let z_col_means: Vec<f64> = (0..n_basis)
            .map(|j| z.iter().map(|row| row[j]).sum::<f64>() / n_obs as f64)
            .collect();
        let z_centred: Vec<Vec<f64>> = z
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(j, &v)| v - z_col_means[j])
                    .collect()
            })
            .collect();

        // --- 5. Penalised least-squares: (Z'Z + λΩ) c = Z' y_c ----------
        let coefs = penalized_ls(&z_centred, &y_centred, &omega, self.config.lambda)?;

        // Recover intercept: α = ȳ - z̄' c
        let intercept = y_mean
            - z_col_means
                .iter()
                .zip(coefs.iter())
                .map(|(&zm, &c)| zm * c)
                .sum::<f64>();

        // --- 6. Fitted values and R² --------------------------------------
        let y_hat: Vec<f64> = z
            .iter()
            .map(|row| {
                intercept
                    + row
                        .iter()
                        .zip(coefs.iter())
                        .map(|(&z_ij, &c)| z_ij * c)
                        .sum::<f64>()
            })
            .collect();

        let ss_res: f64 = response
            .iter()
            .zip(y_hat.iter())
            .map(|(&y, &yh)| (y - yh).powi(2))
            .sum();
        let ss_tot: f64 = response.iter().map(|&y| (y - y_mean).powi(2)).sum();
        let r_squared = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            1.0
        };

        // --- 7. GCV score --------------------------------------------------
        // GCV(λ) = (1/n) ||y - ŷ||² / (1 - h̄)²
        // where h̄ = trace(H) / n and H = Z (Z'Z + λΩ)⁻¹ Z'
        let gcv_score = compute_gcv(&z, &y_hat, response, &omega, self.config.lambda, n_obs);

        // --- 8. Evaluate β on evaluation grid -----------------------------
        let eval_grid = linspace(grid[0], *grid.last().unwrap_or(&1.0), self.config.n_grid);
        let phi_eval = self.evaluate_basis(&eval_grid);
        let beta_values: Vec<f64> = eval_grid
            .iter()
            .enumerate()
            .map(|(t, _)| {
                phi_eval[t]
                    .iter()
                    .zip(coefs.iter())
                    .map(|(&p, &c)| p * c)
                    .sum()
            })
            .collect();

        // Store state for prediction
        self.beta_coefs = Some(coefs.clone());
        self.intercept = Some(intercept);
        self.fit_grid = Some(grid.to_vec());

        Ok(FofResult {
            beta_coefs: coefs,
            beta_values,
            grid: eval_grid,
            intercept,
            r_squared,
            gcv_score,
        })
    }

    /// Predict scalar responses for new functional observations.
    ///
    /// # Arguments
    ///
    /// * `new_data` – new functional observations, shape `(n_new, n_time)`.
    /// * `grid`     – the same time grid used during `fit`.
    ///
    /// # Errors
    ///
    /// Returns `StatsError` if the model has not been fitted yet, or if
    /// dimension mismatches exist.
    pub fn predict(&self, new_data: &[Vec<f64>], grid: &[f64]) -> StatsResult<Vec<f64>> {
        let coefs = self
            .beta_coefs
            .as_ref()
            .ok_or_else(|| StatsError::ComputationError("model not fitted yet".to_owned()))?;
        let intercept = self.intercept.unwrap_or(0.0);

        let n_basis = self.n_basis_fns();
        let phi = self.evaluate_basis(grid);
        let z = build_z_matrix(new_data, &phi, grid);

        let preds = z
            .iter()
            .map(|row| {
                intercept
                    + row
                        .iter()
                        .zip(coefs.iter())
                        .take(n_basis)
                        .map(|(&z_ij, &c)| z_ij * c)
                        .sum::<f64>()
            })
            .collect();

        Ok(preds)
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Return the number of basis functions.
    fn n_basis_fns(&self) -> usize {
        match self.config.basis {
            FunctionalBasis::BSpline { n_basis, .. } => n_basis,
            FunctionalBasis::Fourier { n_basis } => n_basis,
            FunctionalBasis::Wavelet { n_basis } => n_basis,
        }
    }

    /// Evaluate basis functions at each grid point.
    ///
    /// Returns a matrix of shape `(n_grid, n_basis)` where `phi[t][k]` = φ_k(grid[t]).
    fn evaluate_basis(&self, grid: &[f64]) -> Vec<Vec<f64>> {
        match self.config.basis {
            FunctionalBasis::BSpline { n_basis, degree } => bspline_basis(grid, n_basis, degree),
            FunctionalBasis::Fourier { n_basis } => fourier_basis(grid, n_basis),
            FunctionalBasis::Wavelet { n_basis } => wavelet_basis(grid, n_basis),
        }
    }

    /// Second-difference roughness penalty matrix Ω.
    ///
    /// Ω = D' D where D is the (n_basis-2) × n_basis second-difference matrix.
    fn roughness_penalty(&self, n_basis: usize) -> Vec<Vec<f64>> {
        roughness_penalty(n_basis)
    }
}

// ---------------------------------------------------------------------------
// B-spline basis (de Boor recursion)
// ---------------------------------------------------------------------------

/// Compute B-spline basis functions at the given grid points.
///
/// Uses the de Boor recursion on equidistant knots. Returns a matrix of
/// shape `(n_grid, n_basis)`.
pub fn bspline_basis(grid: &[f64], n_basis: usize, degree: usize) -> Vec<Vec<f64>> {
    let n_grid = grid.len();
    if n_basis == 0 || n_grid == 0 {
        return vec![vec![]; n_grid];
    }

    let t_min = grid[0];
    let t_max = *grid.last().unwrap_or(&1.0);

    // Number of internal knots: n_basis - degree + 1 (for clamped B-splines)
    // Total knot vector length: n_basis + degree + 1
    let n_knots = n_basis + degree + 1;
    let knots = build_clamped_knots(t_min, t_max, n_knots, degree);

    let mut phi = vec![vec![0.0f64; n_basis]; n_grid];
    for (t_idx, &t) in grid.iter().enumerate() {
        for k in 0..n_basis {
            phi[t_idx][k] = de_boor_basis(t, k, degree, &knots);
        }
    }
    phi
}

/// Build a clamped B-spline knot vector with `n_knots` total knots.
fn build_clamped_knots(t_min: f64, t_max: f64, n_knots: usize, degree: usize) -> Vec<f64> {
    let mut knots = Vec::with_capacity(n_knots);
    // Clamp `degree` copies at each end, uniform interior knots
    let n_interior = n_knots.saturating_sub(2 * (degree + 1));
    for _ in 0..=degree {
        knots.push(t_min);
    }
    for i in 1..=(n_interior) {
        let t = t_min + (t_max - t_min) * (i as f64) / (n_interior + 1) as f64;
        knots.push(t);
    }
    while knots.len() < n_knots - (degree) {
        let t = t_max;
        knots.push(t);
    }
    for _ in 0..=degree {
        knots.push(t_max);
    }
    knots.truncate(n_knots);
    // Ensure last knots are t_max
    while knots.len() < n_knots {
        knots.push(t_max);
    }
    knots
}

/// Evaluate B-spline basis function B_{k,p}(t) using the Cox-de Boor recursion.
///
/// * `k`     – index of the basis function (0-based)
/// * `p`     – degree
/// * `knots` – knot vector
fn de_boor_basis(t: f64, k: usize, p: usize, knots: &[f64]) -> f64 {
    let n_knots = knots.len();
    if k + p + 1 >= n_knots {
        return 0.0;
    }

    if p == 0 {
        // Indicator: 1 if knots[k] ≤ t < knots[k+1] (with special case at right end)
        let at_right_end = (t - knots[k + 1]).abs() < 1e-14
            && knots[k + 1] >= knots.last().copied().unwrap_or(f64::INFINITY);
        return if (t >= knots[k] && t < knots[k + 1]) || at_right_end {
            1.0
        } else {
            0.0
        };
    }

    let denom1 = knots[k + p] - knots[k];
    let left = if denom1.abs() > 1e-14 {
        (t - knots[k]) / denom1 * de_boor_basis(t, k, p - 1, knots)
    } else {
        0.0
    };

    let denom2 = knots[k + p + 1] - knots[k + 1];
    let right = if denom2.abs() > 1e-14 {
        (knots[k + p + 1] - t) / denom2 * de_boor_basis(t, k + 1, p - 1, knots)
    } else {
        0.0
    };

    left + right
}

// ---------------------------------------------------------------------------
// Fourier basis
// ---------------------------------------------------------------------------

/// Fourier basis on the normalised interval [0, 1].
///
/// For `n_basis = 2m+1`: {1, cos(2πt), sin(2πt), cos(4πt), sin(4πt), ...}
fn fourier_basis(grid: &[f64], n_basis: usize) -> Vec<Vec<f64>> {
    let n_grid = grid.len();
    if n_basis == 0 || n_grid == 0 {
        return vec![vec![]; n_grid];
    }

    let t_min = grid[0];
    let t_max = *grid.last().unwrap_or(&1.0);
    let span = (t_max - t_min).max(1e-12);

    let mut phi = vec![vec![0.0f64; n_basis]; n_grid];
    for (t_idx, &t) in grid.iter().enumerate() {
        let s = (t - t_min) / span; // normalise to [0,1]
        phi[t_idx][0] = 1.0; // constant
        let mut k = 1usize;
        let mut freq = 1usize;
        while k < n_basis {
            let omega = 2.0 * std::f64::consts::PI * freq as f64 * s;
            if k < n_basis {
                phi[t_idx][k] = omega.cos();
                k += 1;
            }
            if k < n_basis {
                phi[t_idx][k] = omega.sin();
                k += 1;
            }
            freq += 1;
        }
    }
    phi
}

// ---------------------------------------------------------------------------
// Haar wavelet basis
// ---------------------------------------------------------------------------

/// Haar wavelet basis on the normalised interval [0, 1].
///
/// The first basis function is the scaling function 1_{[0,1)}.
/// Subsequent functions are Haar wavelets at dyadic levels.
fn wavelet_basis(grid: &[f64], n_basis: usize) -> Vec<Vec<f64>> {
    let n_grid = grid.len();
    if n_basis == 0 || n_grid == 0 {
        return vec![vec![]; n_grid];
    }

    let t_min = grid[0];
    let t_max = *grid.last().unwrap_or(&1.0);
    let span = (t_max - t_min).max(1e-12);

    let mut phi = vec![vec![0.0f64; n_basis]; n_grid];
    for (t_idx, &t) in grid.iter().enumerate() {
        let s = (t - t_min) / span; // normalise to [0,1]
                                    // k=0: constant scaling function
        phi[t_idx][0] = 1.0;

        // k>=1: Haar wavelets; encode (level, translate) in index
        let mut k = 1usize;
        let mut level = 0usize;
        while k < n_basis {
            let n_at_level = 1usize << level; // 2^level wavelets at this level
            let scale = (n_at_level as f64).sqrt(); // L2 normalisation
            for translate in 0..n_at_level {
                if k >= n_basis {
                    break;
                }
                let t0 = translate as f64 / n_at_level as f64;
                let tmid = (translate as f64 + 0.5) / n_at_level as f64;
                let t1 = (translate + 1) as f64 / n_at_level as f64;
                phi[t_idx][k] = if s >= t0 && s < tmid {
                    scale
                } else if s >= tmid && s < t1 {
                    -scale
                } else {
                    0.0
                };
                k += 1;
            }
            level += 1;
        }
    }
    phi
}

// ---------------------------------------------------------------------------
// Z matrix: numerical integration ∫ X_i(t) φ_j(t) dt
// ---------------------------------------------------------------------------

/// Compute the design matrix Z where Z_{ij} = ∫ X_i(t) φ_j(t) dt.
///
/// Uses the composite trapezoidal rule over `grid`.
fn build_z_matrix(data: &[Vec<f64>], phi: &[Vec<f64>], grid: &[f64]) -> Vec<Vec<f64>> {
    let n_obs = data.len();
    let n_time = grid.len();
    let n_basis = phi.first().map(|r| r.len()).unwrap_or(0);

    let mut z = vec![vec![0.0f64; n_basis]; n_obs];

    for (i, xi) in data.iter().enumerate() {
        for j in 0..n_basis {
            // Trapezoidal rule: Σ_{t=0}^{n_time-2} (h/2)(f_t + f_{t+1})
            let mut integral = 0.0f64;
            for t in 0..(n_time.saturating_sub(1)) {
                let h = grid[t + 1] - grid[t];
                let f_t = xi[t] * phi[t][j];
                let f_t1 = xi[t + 1] * phi[t + 1][j];
                integral += 0.5 * h * (f_t + f_t1);
            }
            z[i][j] = integral;
        }
    }
    z
}

// ---------------------------------------------------------------------------
// Roughness penalty matrix
// ---------------------------------------------------------------------------

/// Second-difference roughness penalty matrix Ω = D'D.
///
/// D is the `(n_basis - 2) × n_basis` second-difference operator.
pub fn roughness_penalty(n_basis: usize) -> Vec<Vec<f64>> {
    if n_basis < 3 {
        return vec![vec![0.0; n_basis]; n_basis];
    }
    let m = n_basis - 2; // number of rows in D
                         // D_{k, k} = 1, D_{k, k+1} = -2, D_{k, k+2} = 1
    let mut omega = vec![vec![0.0f64; n_basis]; n_basis];
    for row in 0..m {
        // D'D: omega[j][k] = Σ_r D[r,j] * D[r,k]
        // D[r, r] = 1, D[r, r+1] = -2, D[r, r+2] = 1
        // Only three non-zero entries per row of D
        let cols = [(row, 1.0f64), (row + 1, -2.0f64), (row + 2, 1.0f64)];
        for &(c1, v1) in &cols {
            for &(c2, v2) in &cols {
                omega[c1][c2] += v1 * v2;
            }
        }
    }
    omega
}

// ---------------------------------------------------------------------------
// Penalised least squares solver
// ---------------------------------------------------------------------------

/// Solve the penalised LS system: (Z'Z + λΩ) c = Z'y.
///
/// Uses Cholesky decomposition via an LDL' variant that falls back to
/// positive-definite Gaussian elimination.
pub fn penalized_ls(
    z: &[Vec<f64>],
    y: &[f64],
    omega: &[Vec<f64>],
    lambda: f64,
) -> StatsResult<Vec<f64>> {
    let n = z.first().map(|r| r.len()).unwrap_or(0);
    if n == 0 {
        return Ok(Vec::new());
    }

    // A = Z'Z + λΩ
    let mut a = vec![vec![0.0f64; n]; n];
    for obs in z {
        for j in 0..n {
            for k in 0..n {
                a[j][k] += obs[j] * obs[k];
            }
        }
    }
    for j in 0..n {
        for k in 0..n {
            a[j][k] += lambda * omega[j][k];
        }
    }

    // b = Z'y
    let mut b = vec![0.0f64; n];
    for (obs, &yi) in z.iter().zip(y) {
        for j in 0..n {
            b[j] += obs[j] * yi;
        }
    }

    // Solve A c = b via Gaussian elimination with partial pivoting
    gauss_solve(&a, &b)
}

/// Gaussian elimination with partial pivoting.
fn gauss_solve(a: &[Vec<f64>], b: &[f64]) -> StatsResult<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Augmented matrix [A | b]
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();

    for col in 0..n {
        // Find pivot
        let pivot_row = (col..n).max_by(|&r1, &r2| {
            m[r1][col]
                .abs()
                .partial_cmp(&m[r2][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let pivot_row = pivot_row
            .ok_or_else(|| StatsError::ComputationError("singular penalised system".to_owned()))?;

        m.swap(col, pivot_row);

        let pivot = m[col][col];
        if pivot.abs() < 1e-300 {
            return Err(StatsError::ComputationError(
                "near-singular penalised normal equations; increase lambda".to_owned(),
            ));
        }

        for row in (col + 1)..n {
            let factor = m[row][col] / pivot;
            for k in col..=n {
                let val = m[col][k];
                m[row][k] -= factor * val;
            }
        }
    }

    // Back-substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = m[i][n];
        for j in (i + 1)..n {
            sum -= m[i][j] * x[j];
        }
        x[i] = sum / m[i][i];
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// GCV score
// ---------------------------------------------------------------------------

/// Compute the GCV score.
///
/// GCV(λ) = (1/n) ||y - ŷ||² / (1 - trace(H)/n)²
///
/// where H = Z (Z'Z + λΩ)⁻¹ Z' (the hat matrix).
/// We approximate `trace(H)` via the diagonal sum of H = Z A⁻¹ Z'.
fn compute_gcv(
    z: &[Vec<f64>],
    y_hat: &[f64],
    response: &[f64],
    omega: &[Vec<f64>],
    lambda: f64,
    n_obs: usize,
) -> f64 {
    let n = z.first().map(|r| r.len()).unwrap_or(0);
    if n == 0 || n_obs == 0 {
        return f64::INFINITY;
    }

    // Build A = Z'Z + λΩ
    let mut a = vec![vec![0.0f64; n]; n];
    for obs in z {
        for j in 0..n {
            for k in 0..n {
                a[j][k] += obs[j] * obs[k];
            }
        }
    }
    for j in 0..n {
        for k in 0..n {
            a[j][k] += lambda * omega[j][k];
        }
    }

    // Invert A via Gauss-Jordan (small n_basis, typically ≤ 20)
    let a_inv = match invert_matrix(&a) {
        Ok(inv) => inv,
        Err(_) => return f64::INFINITY,
    };

    // trace(H) = trace(Z A⁻¹ Z') = Σ_i (z_i' A⁻¹ z_i)
    let tr_h: f64 = z
        .iter()
        .map(|zi| {
            // A⁻¹ z_i
            let az: Vec<f64> = (0..n)
                .map(|j| (0..n).map(|k| a_inv[j][k] * zi[k]).sum::<f64>())
                .collect();
            // z_i' (A⁻¹ z_i)
            zi.iter().zip(az.iter()).map(|(&v, &w)| v * w).sum::<f64>()
        })
        .sum();

    let df_hat = tr_h / n_obs as f64;
    if (1.0 - df_hat).abs() < 1e-10 {
        return f64::INFINITY;
    }

    let ss_res: f64 = response
        .iter()
        .zip(y_hat.iter())
        .map(|(&y, &yh)| (y - yh).powi(2))
        .sum();

    (ss_res / n_obs as f64) / (1.0 - df_hat).powi(2)
}

/// Invert an n×n matrix using Gauss-Jordan elimination.
fn invert_matrix(a: &[Vec<f64>]) -> StatsResult<Vec<Vec<f64>>> {
    let n = a.len();
    // Augment with identity
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.resize(2 * n, 0.0);
            r[n + i] = 1.0;
            r
        })
        .collect();

    for col in 0..n {
        // Pivot
        let pivot_row = (col..n).max_by(|&r1, &r2| {
            m[r1][col]
                .abs()
                .partial_cmp(&m[r2][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let pivot_row =
            pivot_row.ok_or_else(|| StatsError::ComputationError("singular matrix".to_owned()))?;
        m.swap(col, pivot_row);

        let pivot = m[col][col];
        if pivot.abs() < 1e-300 {
            return Err(StatsError::ComputationError("singular matrix".to_owned()));
        }
        let scale = 1.0 / pivot;
        for k in 0..(2 * n) {
            m[col][k] *= scale;
        }
        for row in 0..n {
            if row != col {
                let factor = m[row][col];
                for k in 0..(2 * n) {
                    let val = m[col][k];
                    m[row][k] -= factor * val;
                }
            }
        }
    }

    let inv: Vec<Vec<f64>> = m.iter().map(|row| row[n..].to_vec()).collect();
    Ok(inv)
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Generate `n` equidistant points in `[start, end]`.
fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }
    (0..n)
        .map(|i| start + (end - start) * i as f64 / (n - 1) as f64)
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Simple LCG random number generator to avoid external dependencies.
    fn lcg(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*s >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Generate `n` sample paths of a smooth function X_i(t) = a_i * f(t) + noise.
    fn smooth_data(n_obs: usize, n_time: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        let grid: Vec<f64> = (0..n_time)
            .map(|i| i as f64 / (n_time - 1) as f64)
            .collect();
        let mut rng = seed;
        let mut data = Vec::with_capacity(n_obs);
        let mut response = Vec::with_capacity(n_obs);
        // β(t) = sin(πt), so ∫ β(t) X_i(t) dt = a_i ∫ sin(πt) sin(2πt) dt + noise
        // We use X_i(t) = a_i * sin(2πt) so we can compute expected ∫
        for _ in 0..n_obs {
            let a = lcg(&mut rng) * 2.0 - 1.0; // a in [-1,1]
            let curve: Vec<f64> = grid
                .iter()
                .map(|&t| a * (2.0 * std::f64::consts::PI * t).sin())
                .collect();
            // ∫₀¹ sin(πt) a sin(2πt) dt = a ∫₀¹ sin(πt)sin(2πt)dt ≈ 0 (orthogonal on [0,1])
            // Use a simpler β(t)=t: Y_i = ∫₀¹ t * a sin(2πt) dt = a ∫₀¹ t sin(2πt) dt
            let integral: f64 = grid
                .windows(2)
                .map(|w| {
                    let t0 = w[0];
                    let t1 = w[1];
                    let dt = t1 - t0;
                    let f0 = t0 * a * (2.0 * std::f64::consts::PI * t0).sin();
                    let f1 = t1 * a * (2.0 * std::f64::consts::PI * t1).sin();
                    0.5 * dt * (f0 + f1)
                })
                .sum();
            response.push(integral + (lcg(&mut rng) - 0.5) * 0.01); // tiny noise
            data.push(curve);
        }
        (data, response, grid)
    }

    // -----------------------------------------------------------------------
    // Config defaults
    // -----------------------------------------------------------------------

    #[test]
    fn test_fof_config_default() {
        let cfg = FofConfig::default();
        assert_eq!(
            cfg.basis,
            FunctionalBasis::BSpline {
                n_basis: 10,
                degree: 3
            }
        );
        assert!((cfg.lambda - 0.01).abs() < 1e-15);
        assert_eq!(cfg.n_grid, 100);
    }

    // -----------------------------------------------------------------------
    // B-spline basis
    // -----------------------------------------------------------------------

    #[test]
    fn test_bspline_basis_partition_of_unity() {
        // B-splines of any degree form a partition of unity: Σ_k φ_k(t) = 1
        let grid: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
        let phi = bspline_basis(&grid, 8, 3);
        for (t_idx, _) in grid.iter().enumerate() {
            let s: f64 = phi[t_idx].iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-8,
                "partition of unity at t={t_idx}: sum={s}"
            );
        }
    }

    #[test]
    fn test_bspline_basis_non_negative() {
        let grid: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
        let phi = bspline_basis(&grid, 10, 3);
        for row in &phi {
            for &v in row {
                assert!(v >= -1e-10, "negative B-spline value: {v}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Roughness penalty
    // -----------------------------------------------------------------------

    #[test]
    fn test_roughness_penalty_symmetry() {
        let omega = roughness_penalty(8);
        let n = omega.len();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (omega[i][j] - omega[j][i]).abs() < 1e-14,
                    "Omega not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_roughness_penalty_psd() {
        // Ω = D'D is positive semi-definite; verify all diagonal entries ≥ 0
        let omega = roughness_penalty(6);
        for (i, row) in omega.iter().enumerate() {
            assert!(row[i] >= 0.0, "negative diagonal in Omega");
        }
    }

    // -----------------------------------------------------------------------
    // Penalised LS
    // -----------------------------------------------------------------------

    #[test]
    fn test_penalized_ls_identity() {
        // With Z = I, Ω = 0, λ=0: solution should be y
        let n = 4;
        let z: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let omega = vec![vec![0.0; n]; n];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let c = penalized_ls(&z, &y, &omega, 0.0).expect("penalized_ls failed");
        for (ci, &yi) in c.iter().zip(y.iter()) {
            assert!((ci - yi).abs() < 1e-10, "expected {yi}, got {ci}");
        }
    }

    // -----------------------------------------------------------------------
    // Fit and predict
    // -----------------------------------------------------------------------

    #[test]
    fn test_fit_r_squared_high_on_clean_data() {
        let (data, response, grid) = smooth_data(50, 40, 42);
        let config = FofConfig {
            basis: FunctionalBasis::BSpline {
                n_basis: 8,
                degree: 3,
            },
            lambda: 1e-4,
            n_grid: 50,
        };
        let mut model = FunctionalRegression::new(config);
        let result = model.fit(&data, &response, &grid).expect("fit failed");
        // R² should be high (> 0.9) on near-noise-free data
        assert!(result.r_squared > 0.9, "R² too low: {}", result.r_squared);
    }

    #[test]
    fn test_predict_length() {
        let (data, response, grid) = smooth_data(30, 30, 7);
        let config = FofConfig {
            basis: FunctionalBasis::BSpline {
                n_basis: 6,
                degree: 3,
            },
            lambda: 0.01,
            n_grid: 50,
        };
        let mut model = FunctionalRegression::new(config);
        model.fit(&data, &response, &grid).expect("fit failed");

        let (new_data, _, _) = smooth_data(10, 30, 99);
        let preds = model.predict(&new_data, &grid).expect("predict failed");
        assert_eq!(preds.len(), 10, "predict length mismatch");
    }

    #[test]
    fn test_predict_before_fit_returns_error() {
        let config = FofConfig::default();
        let model = FunctionalRegression::new(config);
        let grid: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
        let data = vec![vec![0.0; 10]];
        let res = model.predict(&data, &grid);
        assert!(res.is_err(), "predict before fit should return error");
    }

    #[test]
    fn test_fit_with_fourier_basis() {
        let (data, response, grid) = smooth_data(40, 40, 123);
        let config = FofConfig {
            basis: FunctionalBasis::Fourier { n_basis: 9 },
            lambda: 0.01,
            n_grid: 50,
        };
        let mut model = FunctionalRegression::new(config);
        let result = model.fit(&data, &response, &grid).expect("fit failed");
        assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0 + 1e-10);
    }

    #[test]
    fn test_fit_with_wavelet_basis() {
        let (data, response, grid) = smooth_data(40, 40, 456);
        let config = FofConfig {
            basis: FunctionalBasis::Wavelet { n_basis: 8 },
            lambda: 0.01,
            n_grid: 50,
        };
        let mut model = FunctionalRegression::new(config);
        let result = model.fit(&data, &response, &grid).expect("fit failed");
        assert!(result.r_squared >= 0.0, "r_squared should be non-negative");
    }

    #[test]
    fn test_gcv_score_finite() {
        let (data, response, grid) = smooth_data(30, 30, 11);
        let config = FofConfig {
            basis: FunctionalBasis::BSpline {
                n_basis: 6,
                degree: 3,
            },
            lambda: 0.1,
            n_grid: 40,
        };
        let mut model = FunctionalRegression::new(config);
        let result = model.fit(&data, &response, &grid).expect("fit failed");
        assert!(result.gcv_score.is_finite(), "GCV should be finite");
        assert!(result.gcv_score >= 0.0, "GCV should be non-negative");
    }

    #[test]
    fn test_beta_values_length() {
        let (data, response, grid) = smooth_data(30, 30, 13);
        let n_grid = 60;
        let config = FofConfig {
            basis: FunctionalBasis::BSpline {
                n_basis: 6,
                degree: 3,
            },
            lambda: 0.01,
            n_grid,
        };
        let mut model = FunctionalRegression::new(config);
        let result = model.fit(&data, &response, &grid).expect("fit failed");
        assert_eq!(result.beta_values.len(), n_grid);
        assert_eq!(result.grid.len(), n_grid);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let response = vec![1.0]; // wrong length
        let grid = vec![0.0, 0.5, 1.0];
        let config = FofConfig::default();
        let mut model = FunctionalRegression::new(config);
        let res = model.fit(&data, &response, &grid);
        assert!(res.is_err(), "should return dimension mismatch error");
    }

    #[test]
    fn test_gcv_varies_with_lambda() {
        // GCV should not be constant as lambda changes
        let (data, response, grid) = smooth_data(40, 40, 77);
        let lambdas = [1e-4, 1e-2, 1.0];
        let mut gcv_scores = Vec::new();
        for &lam in &lambdas {
            let config = FofConfig {
                basis: FunctionalBasis::BSpline {
                    n_basis: 6,
                    degree: 3,
                },
                lambda: lam,
                n_grid: 40,
            };
            let mut model = FunctionalRegression::new(config);
            let result = model.fit(&data, &response, &grid).expect("fit failed");
            gcv_scores.push(result.gcv_score);
        }
        // At least two GCV scores should differ
        let all_same = gcv_scores.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-14);
        assert!(!all_same, "GCV should vary with lambda");
    }
}
