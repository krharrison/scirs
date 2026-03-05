//! Functional data smoothing methods
//!
//! Provides penalized least squares, kernel smoothers, and cubic smoothing splines
//! for converting discrete noisy observations into smooth functional representations.
//!
//! # Methods
//!
//! - [`PenalizedLeastSquares`]: Roughness-penalty smoothing with GCV selection
//! - [`KernelSmoother`]: Nadaraya-Watson, local linear, and local polynomial regression
//! - [`SplineSmoother`]: Cubic smoothing spline with cross-validation for λ
//! - [`FunctionalData`]: Smooth curve object supporting evaluation and differentiation

use crate::error::{Result, TimeSeriesError};
use crate::functional::basis::{evaluate_basis_matrix, BasisSystem, BSplineBasis};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, Axis};
use scirs2_linalg::{lstsq, solve};

// ============================================================
// FunctionalData: Smooth curve representation
// ============================================================

/// A smooth functional data object represented as a linear combination of basis functions.
///
/// Stores the basis coefficients and provides methods for evaluation,
/// derivative computation, and integration.
#[derive(Debug, Clone)]
pub struct FunctionalData<B: BasisSystem + Clone> {
    /// Basis system
    pub basis: B,
    /// Coefficient vector (length = basis.n_basis())
    pub coefficients: Array1<f64>,
    /// Optional label
    pub label: Option<String>,
}

impl<B: BasisSystem + Clone> FunctionalData<B> {
    /// Create a functional data object from basis and coefficients
    pub fn new(basis: B, coefficients: Array1<f64>) -> Result<Self> {
        if coefficients.len() != basis.n_basis() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: basis.n_basis(),
                actual: coefficients.len(),
            });
        }
        Ok(Self {
            basis,
            coefficients,
            label: None,
        })
    }

    /// Evaluate the functional data object at a single point
    pub fn eval(&self, t: f64) -> Result<f64> {
        let phi = self.basis.evaluate(t)?;
        Ok(phi.dot(&self.coefficients))
    }

    /// Evaluate at multiple points
    pub fn eval_vec(&self, points: &Array1<f64>) -> Result<Array1<f64>> {
        let mut vals = Array1::zeros(points.len());
        for (i, &t) in points.iter().enumerate() {
            vals[i] = self.eval(t)?;
        }
        Ok(vals)
    }

    /// Evaluate the `order`-th derivative at a single point
    pub fn eval_deriv(&self, t: f64, order: usize) -> Result<f64> {
        let dphi = self.basis.evaluate_deriv(t, order)?;
        Ok(dphi.dot(&self.coefficients))
    }

    /// Evaluate derivative at multiple points
    pub fn eval_deriv_vec(&self, points: &Array1<f64>, order: usize) -> Result<Array1<f64>> {
        let mut vals = Array1::zeros(points.len());
        for (i, &t) in points.iter().enumerate() {
            vals[i] = self.eval_deriv(t, order)?;
        }
        Ok(vals)
    }

    /// Compute the inner product with another FunctionalData object having the same basis
    /// ⟨f, g⟩ = ∫ f(t) g(t) dt ≈ c_f^T G c_g where G is the Gram matrix
    pub fn inner_product(&self, other: &FunctionalData<B>) -> Result<f64> {
        let g = self.basis.gram_matrix()?;
        Ok(self.coefficients.dot(&g.dot(&other.coefficients)))
    }

    /// Compute the L2 norm: ||f|| = sqrt(⟨f, f⟩)
    pub fn l2_norm(&self) -> Result<f64> {
        let ip = self.inner_product(self)?;
        Ok(ip.max(0.0).sqrt())
    }
}

// ============================================================
// GCV: Generalized Cross-Validation
// ============================================================

/// Generalized cross-validation (GCV) criterion for smoothing parameter selection.
///
/// GCV(λ) = (1/n) * ||y - ŷ(λ)||² / (1 - tr(H(λ))/n)²
/// where H(λ) is the hat matrix.
#[derive(Debug, Clone)]
pub struct GCV {
    /// Evaluated GCV values at each candidate λ
    pub lambda_grid: Vec<f64>,
    /// GCV criterion values
    pub gcv_values: Vec<f64>,
    /// Optimal λ (minimizing GCV)
    pub optimal_lambda: f64,
    /// GCV value at optimal λ
    pub optimal_gcv: f64,
}

impl GCV {
    /// Compute GCV over a log-spaced grid and find the optimal λ
    ///
    /// - `lambda_min`, `lambda_max`: search range
    /// - `n_lambda`: number of candidate values
    /// - `gcv_fn`: closure that returns (GCV value, trace of hat matrix) for a given λ
    pub fn select<F>(
        lambda_min: f64,
        lambda_max: f64,
        n_lambda: usize,
        gcv_fn: F,
    ) -> Result<Self>
    where
        F: Fn(f64) -> Result<f64>,
    {
        if n_lambda < 2 {
            return Err(TimeSeriesError::InvalidInput(
                "n_lambda must be >= 2".to_string(),
            ));
        }
        let log_min = lambda_min.ln();
        let log_max = lambda_max.ln();
        let step = (log_max - log_min) / (n_lambda - 1) as f64;
        let mut lambda_grid = Vec::with_capacity(n_lambda);
        let mut gcv_values = Vec::with_capacity(n_lambda);
        for i in 0..n_lambda {
            let lam = (log_min + i as f64 * step).exp();
            let gcv_val = gcv_fn(lam)?;
            lambda_grid.push(lam);
            gcv_values.push(gcv_val);
        }
        // Find minimum
        let (opt_idx, &optimal_gcv) = gcv_values
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| TimeSeriesError::ComputationError("GCV: empty grid".to_string()))?;
        let optimal_lambda = lambda_grid[opt_idx];
        Ok(Self {
            lambda_grid,
            gcv_values,
            optimal_lambda,
            optimal_gcv,
        })
    }
}

// ============================================================
// PenalizedLeastSquares
// ============================================================

/// Roughness-penalized least squares smoother.
///
/// Fits a function f in a basis space by minimizing:
/// `||y - Φ c||² + λ c^T P c`
/// where P is the penalty matrix (integral of squared `penalty_order`-th derivative).
///
/// The smoothing parameter λ is selected automatically by GCV if not provided.
#[derive(Debug, Clone)]
pub struct PenalizedLeastSquares {
    /// Smoothing parameter λ (None = auto-select via GCV)
    pub lambda: Option<f64>,
    /// Order of derivative for roughness penalty
    pub penalty_order: usize,
    /// Number of candidate λ values for GCV grid search
    pub n_lambda_grid: usize,
    /// Minimum candidate λ
    pub lambda_min: f64,
    /// Maximum candidate λ
    pub lambda_max: f64,
}

impl Default for PenalizedLeastSquares {
    fn default() -> Self {
        Self {
            lambda: None,
            penalty_order: 2,
            n_lambda_grid: 50,
            lambda_min: 1e-10,
            lambda_max: 1e4,
        }
    }
}

impl PenalizedLeastSquares {
    /// Fit a smooth function to (t_i, y_i) observations using the given basis
    pub fn fit<B: BasisSystem + Clone>(
        &self,
        basis: B,
        t: &Array1<f64>,
        y: &Array1<f64>,
    ) -> Result<FunctionalData<B>> {
        let n = t.len();
        if n != y.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: y.len(),
            });
        }
        if n < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "PenalizedLeastSquares requires at least 2 observations".to_string(),
                required: 2,
                actual: n,
            });
        }

        // Build basis matrix Φ (n × k)
        let phi = evaluate_basis_matrix(&basis, t)?;
        let penalty = basis.penalty_matrix(self.penalty_order)?;

        // Determine λ
        let lambda = match self.lambda {
            Some(lam) => lam,
            None => {
                // GCV-based selection
                let gcv = GCV::select(
                    self.lambda_min,
                    self.lambda_max,
                    self.n_lambda_grid,
                    |lam| {
                        let gcv_val = Self::gcv_criterion(&phi, y, &penalty, lam, n)?;
                        Ok(gcv_val)
                    },
                )?;
                gcv.optimal_lambda
            }
        };

        // Solve (Φ^T Φ + λ P) c = Φ^T y
        let coefficients = Self::solve_penalized(&phi, y, &penalty, lambda)?;
        FunctionalData::new(basis, coefficients)
    }

    /// Compute GCV criterion for given λ
    fn gcv_criterion(
        phi: &Array2<f64>,
        y: &Array1<f64>,
        penalty: &Array2<f64>,
        lambda: f64,
        n: usize,
    ) -> Result<f64> {
        let k = phi.ncols();
        // A = Φ^T Φ + λ P
        let phi_t_phi = phi.t().dot(phi);
        let mut a = phi_t_phi.clone();
        for i in 0..k {
            for j in 0..k {
                a[[i, j]] += lambda * penalty[[i, j]];
            }
        }
        // Solve A c = Φ^T y
        let phi_t_y = phi.t().dot(y);
        let c = solve_linear_system(&a, &phi_t_y)?;

        // Fitted values ŷ = Φ c
        let y_hat = phi.dot(&c);

        // Residual sum of squares
        let rss: f64 = y
            .iter()
            .zip(y_hat.iter())
            .map(|(yi, yhi)| (yi - yhi).powi(2))
            .sum();

        // Trace of hat matrix H = Φ (Φ^T Φ + λ P)^{-1} Φ^T
        // tr(H) = sum_i [Φ A^{-1} Φ^T]_{ii}
        // Efficient: tr(H) = sum of diagonal of Φ A^{-1} Φ^T
        // = tr(Φ^T Φ A^{-1}) = tr(A^{-1} Φ^T Φ)
        let a_inv_phi_t = solve_matrix_system(&a, &phi.t().to_owned())?;
        let h_diag_sum: f64 = (0..n)
            .map(|i| {
                let row = phi.row(i);
                row.dot(&a_inv_phi_t.t().row(i).to_owned())
            })
            .sum();
        let trace_h = h_diag_sum;

        let n_f = n as f64;
        let denom = 1.0 - trace_h / n_f;
        if denom.abs() < 1e-10 {
            return Ok(f64::INFINITY);
        }
        Ok((rss / n_f) / (denom * denom))
    }

    /// Solve the penalized system (Φ^T Φ + λ P) c = Φ^T y
    fn solve_penalized(
        phi: &Array2<f64>,
        y: &Array1<f64>,
        penalty: &Array2<f64>,
        lambda: f64,
    ) -> Result<Array1<f64>> {
        let k = phi.ncols();
        let phi_t_phi = phi.t().dot(phi);
        let mut a = phi_t_phi;
        for i in 0..k {
            for j in 0..k {
                a[[i, j]] += lambda * penalty[[i, j]];
            }
        }
        let phi_t_y = phi.t().dot(y);
        solve_linear_system(&a, &phi_t_y)
    }
}

// ============================================================
// KernelSmoother
// ============================================================

/// Kernel function type for local smoothing
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KernelType {
    /// Epanechnikov kernel: K(u) = (3/4)(1-u²) for |u| ≤ 1
    Epanechnikov,
    /// Gaussian kernel: K(u) = exp(-u²/2)/√(2π)
    Gaussian,
    /// Uniform (box) kernel: K(u) = 1/2 for |u| ≤ 1
    Uniform,
    /// Biweight (quartic) kernel: K(u) = (15/16)(1-u²)² for |u| ≤ 1
    Biweight,
    /// Tricubic kernel: K(u) = (70/81)(1-|u|³)³ for |u| ≤ 1
    Tricubic,
}

impl KernelType {
    fn eval(self, u: f64) -> f64 {
        match self {
            KernelType::Epanechnikov => {
                if u.abs() <= 1.0 {
                    0.75 * (1.0 - u * u)
                } else {
                    0.0
                }
            }
            KernelType::Gaussian => (-0.5 * u * u).exp() / (2.0 * std::f64::consts::PI).sqrt(),
            KernelType::Uniform => {
                if u.abs() <= 1.0 {
                    0.5
                } else {
                    0.0
                }
            }
            KernelType::Biweight => {
                if u.abs() <= 1.0 {
                    let v = 1.0 - u * u;
                    (15.0 / 16.0) * v * v
                } else {
                    0.0
                }
            }
            KernelType::Tricubic => {
                let abs_u = u.abs();
                if abs_u <= 1.0 {
                    let v = 1.0 - abs_u.powi(3);
                    (70.0 / 81.0) * v * v * v
                } else {
                    0.0
                }
            }
        }
    }
}

/// Local polynomial regression order
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LocalPolynomialOrder {
    /// Nadaraya-Watson (degree 0)
    NadarayaWatson,
    /// Local linear regression (degree 1)
    LocalLinear,
    /// Local quadratic regression (degree 2)
    LocalQuadratic,
}

/// Kernel smoother supporting Nadaraya-Watson, local linear, and local polynomial regression.
///
/// Bandwidth `h` can be provided or selected via leave-one-out cross-validation.
#[derive(Debug, Clone)]
pub struct KernelSmoother {
    /// Bandwidth parameter (None = auto via LOO-CV)
    pub bandwidth: Option<f64>,
    /// Kernel function
    pub kernel: KernelType,
    /// Local polynomial order
    pub poly_order: LocalPolynomialOrder,
    /// Number of bandwidth candidates for CV
    pub n_bw_candidates: usize,
}

impl Default for KernelSmoother {
    fn default() -> Self {
        Self {
            bandwidth: None,
            kernel: KernelType::Epanechnikov,
            poly_order: LocalPolynomialOrder::LocalLinear,
            n_bw_candidates: 40,
        }
    }
}

impl KernelSmoother {
    /// Fit kernel smoother to data, returning fitted values at the same observation points
    pub fn fit(
        &self,
        t: &Array1<f64>,
        y: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let n = t.len();
        if n != y.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: y.len(),
            });
        }
        let h = match self.bandwidth {
            Some(bw) => bw,
            None => self.select_bandwidth_cv(t, y)?,
        };
        self.predict_at(t, y, t, h)
    }

    /// Predict at new query points `t_new`
    pub fn predict_at(
        &self,
        t_obs: &Array1<f64>,
        y_obs: &Array1<f64>,
        t_new: &Array1<f64>,
        bandwidth: f64,
    ) -> Result<Array1<f64>> {
        let m = t_new.len();
        let mut result = Array1::zeros(m);
        for (i, &t0) in t_new.iter().enumerate() {
            result[i] = self.local_fit(t_obs, y_obs, t0, bandwidth)?;
        }
        Ok(result)
    }

    /// Evaluate the local polynomial estimate at a single point t0
    fn local_fit(
        &self,
        t: &Array1<f64>,
        y: &Array1<f64>,
        t0: f64,
        h: f64,
    ) -> Result<f64> {
        let n = t.len();
        // Compute kernel weights
        let weights: Vec<f64> = t.iter().map(|&ti| self.kernel.eval((ti - t0) / h)).collect();
        let sum_w: f64 = weights.iter().sum();
        if sum_w < 1e-15 {
            // No observations in window: return nearest neighbor
            let nearest = t
                .iter()
                .enumerate()
                .min_by(|(_, &a), (_, &b)| {
                    (a - t0)
                        .abs()
                        .partial_cmp(&(b - t0).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            return Ok(y[nearest]);
        }

        match self.poly_order {
            LocalPolynomialOrder::NadarayaWatson => {
                let num: f64 = weights.iter().zip(y.iter()).map(|(&w, &yi)| w * yi).sum();
                Ok(num / sum_w)
            }
            LocalPolynomialOrder::LocalLinear => {
                // Solve weighted least squares: β = (X^T W X)^{-1} X^T W y
                // X = [1, t-t0], predict at 0
                let mut xtwx = Array2::zeros((2, 2));
                let mut xtwy = Array1::zeros(2);
                for i in 0..n {
                    let wi = weights[i];
                    let xi = t[i] - t0;
                    xtwx[[0, 0]] += wi;
                    xtwx[[0, 1]] += wi * xi;
                    xtwx[[1, 0]] += wi * xi;
                    xtwx[[1, 1]] += wi * xi * xi;
                    xtwy[0] += wi * y[i];
                    xtwy[1] += wi * xi * y[i];
                }
                let beta = solve_linear_system(&xtwx, &xtwy)?;
                Ok(beta[0])
            }
            LocalPolynomialOrder::LocalQuadratic => {
                // Degree 2: X = [1, t-t0, (t-t0)^2]
                let mut xtwx = Array2::zeros((3, 3));
                let mut xtwy = Array1::zeros(3);
                for i in 0..n {
                    let wi = weights[i];
                    let xi = t[i] - t0;
                    let xi2 = xi * xi;
                    let row = [1.0, xi, xi2];
                    for p in 0..3 {
                        for q in 0..3 {
                            xtwx[[p, q]] += wi * row[p] * row[q];
                        }
                        xtwy[p] += wi * row[p] * y[i];
                    }
                }
                let beta = solve_linear_system(&xtwx, &xtwy)?;
                Ok(beta[0])
            }
        }
    }

    /// Select bandwidth via leave-one-out cross-validation
    fn select_bandwidth_cv(&self, t: &Array1<f64>, y: &Array1<f64>) -> Result<f64> {
        let n = t.len();
        // Bandwidth range: from 0.1 * range to range
        let t_min = t.iter().cloned().fold(f64::INFINITY, f64::min);
        let t_max = t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = t_max - t_min;
        if range < 1e-15 {
            return Err(TimeSeriesError::InvalidInput(
                "All time points are identical".to_string(),
            ));
        }
        let h_min = 0.05 * range;
        let h_max = range;
        let log_min = h_min.ln();
        let log_max = h_max.ln();
        let step = (log_max - log_min) / (self.n_bw_candidates - 1).max(1) as f64;

        let mut best_h = h_min;
        let mut best_cv = f64::INFINITY;

        for k in 0..self.n_bw_candidates {
            let h = (log_min + k as f64 * step).exp();
            // LOO-CV: predict each point leaving it out
            let mut cv_error = 0.0;
            for i in 0..n {
                // Build leave-one-out arrays
                let t_loo: Vec<f64> = t
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, &v)| v)
                    .collect();
                let y_loo: Vec<f64> = y
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, &v)| v)
                    .collect();
                let t_arr = Array1::from(t_loo);
                let y_arr = Array1::from(y_loo);
                let y_pred = self.local_fit(&t_arr, &y_arr, t[i], h)?;
                cv_error += (y[i] - y_pred).powi(2);
            }
            cv_error /= n as f64;
            if cv_error < best_cv {
                best_cv = cv_error;
                best_h = h;
            }
        }
        Ok(best_h)
    }
}

// ============================================================
// SplineSmoother
// ============================================================

/// Cubic smoothing spline smoother.
///
/// Minimizes the penalized criterion:
/// `||y - f||² + λ ∫ [f''(t)]² dt`
/// over all twice-differentiable functions, using a B-spline basis representation.
#[derive(Debug, Clone)]
pub struct SplineSmoother {
    /// Smoothing parameter λ (None = auto via GCV)
    pub lambda: Option<f64>,
    /// Number of interior knots for B-spline representation
    pub n_interior_knots: usize,
    /// Number of λ candidates for GCV
    pub n_lambda_grid: usize,
    /// Minimum λ for grid search
    pub lambda_min: f64,
    /// Maximum λ for grid search
    pub lambda_max: f64,
}

impl Default for SplineSmoother {
    fn default() -> Self {
        Self {
            lambda: None,
            n_interior_knots: 20,
            n_lambda_grid: 60,
            lambda_min: 1e-12,
            lambda_max: 1e6,
        }
    }
}

impl SplineSmoother {
    /// Fit a cubic smoothing spline to (t_i, y_i) data.
    ///
    /// Returns the fitted [`FunctionalData`] object.
    pub fn fit(
        &self,
        t: &Array1<f64>,
        y: &Array1<f64>,
    ) -> Result<FunctionalData<BSplineBasis>> {
        let n = t.len();
        if n != y.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: y.len(),
            });
        }
        if n < 4 {
            return Err(TimeSeriesError::InsufficientData {
                message: "SplineSmoother requires at least 4 observations".to_string(),
                required: 4,
                actual: n,
            });
        }

        // Construct B-spline basis with cubic (order 4) splines
        let basis = BSplineBasis::uniform(self.n_interior_knots, 4)?;
        let pls = PenalizedLeastSquares {
            lambda: self.lambda,
            penalty_order: 2,
            n_lambda_grid: self.n_lambda_grid,
            lambda_min: self.lambda_min,
            lambda_max: self.lambda_max,
        };
        pls.fit(basis, t, y)
    }

    /// Fit and return both the smoothed function and the GCV-selected λ
    pub fn fit_with_lambda(
        &self,
        t: &Array1<f64>,
        y: &Array1<f64>,
    ) -> Result<(FunctionalData<BSplineBasis>, f64)> {
        let n = t.len();
        if n < 4 {
            return Err(TimeSeriesError::InsufficientData {
                message: "SplineSmoother requires at least 4 observations".to_string(),
                required: 4,
                actual: n,
            });
        }
        let basis = BSplineBasis::uniform(self.n_interior_knots, 4)?;
        let phi = evaluate_basis_matrix(&basis, t)?;
        let penalty = basis.penalty_matrix(2)?;

        let lambda = match self.lambda {
            Some(lam) => lam,
            None => {
                let gcv = GCV::select(self.lambda_min, self.lambda_max, self.n_lambda_grid, |lam| {
                    PenalizedLeastSquares::gcv_criterion(&phi, y, &penalty, lam, n)
                })?;
                gcv.optimal_lambda
            }
        };

        let coefficients = PenalizedLeastSquares::solve_penalized(&phi, y, &penalty, lambda)?;
        let fd = FunctionalData::new(basis, coefficients)?;
        Ok((fd, lambda))
    }

    /// Cross-validate over a range of λ values and return all CV scores
    pub fn cross_validate(
        &self,
        t: &Array1<f64>,
        y: &Array1<f64>,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let n = t.len();
        if n < 4 {
            return Err(TimeSeriesError::InsufficientData {
                message: "SplineSmoother requires at least 4 observations".to_string(),
                required: 4,
                actual: n,
            });
        }
        let basis = BSplineBasis::uniform(self.n_interior_knots, 4)?;
        let phi = evaluate_basis_matrix(&basis, t)?;
        let penalty = basis.penalty_matrix(2)?;

        let log_min = self.lambda_min.ln();
        let log_max = self.lambda_max.ln();
        let step = (log_max - log_min) / (self.n_lambda_grid - 1).max(1) as f64;

        let mut lambdas = Vec::with_capacity(self.n_lambda_grid);
        let mut gcv_vals = Vec::with_capacity(self.n_lambda_grid);
        for k in 0..self.n_lambda_grid {
            let lam = (log_min + k as f64 * step).exp();
            let gcv = PenalizedLeastSquares::gcv_criterion(&phi, y, &penalty, lam, n)?;
            lambdas.push(lam);
            gcv_vals.push(gcv);
        }
        Ok((lambdas, gcv_vals))
    }
}

// ============================================================
// Utility: Linear system solvers
// ============================================================

/// Solve a symmetric positive definite linear system Ax = b
pub(crate) fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let result = lstsq(&a.view(), &b.view(), None).map_err(|e| {
        TimeSeriesError::ComputationError(format!("lstsq failed: {}", e))
    })?;
    let sol = result.x.clone();
    Ok(sol)
}

/// Solve a matrix system AX = B, returning X
pub(crate) fn solve_matrix_system(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
    let k = b.ncols();
    let n = b.nrows();
    let mut x = Array2::zeros((n, k));
    for j in 0..k {
        let bj = b.column(j).to_owned();
        let xj = solve_linear_system(a, &bj)?;
        for i in 0..n {
            x[[i, j]] = xj[i];
        }
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array1;

    fn make_noisy_sine(n: usize, noise: f64) -> (Array1<f64>, Array1<f64>) {
        let t: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * 2.0 * std::f64::consts::PI).collect();
        let y: Vec<f64> = t.iter().map(|&ti| ti.sin() + noise * ((ti * 7.3).cos())).collect();
        (Array1::from(t), Array1::from(y))
    }

    #[test]
    fn test_spline_smoother_basic() {
        let (t, y) = make_noisy_sine(50, 0.1);
        let smoother = SplineSmoother::default();
        let fd = smoother.fit(&t, &y).expect("smoother fit");
        // Should be able to evaluate at any point
        let val = fd.eval(std::f64::consts::PI).expect("eval");
        // sin(π) ≈ 0, smooth estimate should be near 0
        assert!(val.abs() < 0.5);
    }

    #[test]
    fn test_kernel_smoother_nadaraya_watson() {
        let (t, y) = make_noisy_sine(60, 0.05);
        let smoother = KernelSmoother {
            bandwidth: Some(0.3),
            kernel: KernelType::Gaussian,
            poly_order: LocalPolynomialOrder::NadarayaWatson,
            n_bw_candidates: 10,
        };
        let fitted = smoother.fit(&t, &y).expect("kernel smoother fit");
        assert_eq!(fitted.len(), t.len());
    }

    #[test]
    fn test_penalized_least_squares_fit() {
        let (t, y) = make_noisy_sine(40, 0.1);
        let basis = crate::functional::basis::BSplineBasis::uniform(8, 4).expect("basis");
        let pls = PenalizedLeastSquares {
            lambda: Some(1e-3),
            ..Default::default()
        };
        let fd = pls.fit(basis, &t, &y).expect("PLS fit");
        assert_eq!(fd.coefficients.len(), fd.basis.n_basis());
    }

    #[test]
    fn test_functional_data_inner_product() {
        let basis = crate::functional::basis::BSplineBasis::uniform(5, 4).expect("basis");
        let n_basis = basis.n_basis();
        let c1 = Array1::from_elem(n_basis, 1.0);
        let c2 = Array1::from_elem(n_basis, 2.0);
        let fd1 = FunctionalData::new(basis.clone(), c1).expect("fd1");
        let fd2 = FunctionalData::new(basis, c2).expect("fd2");
        let ip = fd1.inner_product(&fd2).expect("inner product");
        // Should be positive
        assert!(ip > 0.0);
    }
}
