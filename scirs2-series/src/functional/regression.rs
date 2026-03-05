//! Functional regression models
//!
//! Provides regression methods where predictor and/or response are functional objects.
//!
//! # Methods
//!
//! - [`FunctionalLinearModel`]: Scalar-on-function regression β(t) via penalized basis estimation
//! - [`FunctionOnScalarRegression`]: Function-on-scalar regression (functional response, scalar predictors)
//! - [`ConcurrentModel`]: Concurrent (varying-coefficient) functional regression
//! - [`FunctionalANOVA`]: One-way ANOVA for functional data with permutation test

use crate::error::{Result, TimeSeriesError};
use crate::functional::basis::{evaluate_basis_matrix, BasisSystem, BSplineBasis};
use crate::functional::smoothing::{
    solve_linear_system, solve_matrix_system, FunctionalData, PenalizedLeastSquares,
};
use scirs2_core::ndarray::{s, Array1, Array2, Axis};

// ============================================================
// FunctionalLinearModel: Scalar-on-Function Regression
// ============================================================

/// Result of scalar-on-function regression
#[derive(Debug, Clone)]
pub struct FLMResult<B: BasisSystem + Clone> {
    /// Intercept term α
    pub intercept: f64,
    /// Coefficient function β(t) as a FunctionalData object
    pub beta_function: FunctionalData<B>,
    /// Fitted scalar values ŷ_i = α + ⟨x_i, β⟩
    pub fitted_values: Array1<f64>,
    /// Residuals y_i - ŷ_i
    pub residuals: Array1<f64>,
    /// Coefficient of determination R²
    pub r_squared: f64,
    /// Adjusted R²
    pub adjusted_r_squared: f64,
    /// Residual sum of squares
    pub rss: f64,
    /// Mean squared error
    pub mse: f64,
    /// Smoothing parameter λ used
    pub lambda: f64,
}

/// Scalar-on-function regression via penalized basis expansion.
///
/// Models the scalar response y_i as:
/// `y_i = α + ⟨x_i, β⟩ + ε_i = α + ∫ x_i(t) β(t) dt + ε_i`
///
/// Estimates α and β(·) jointly by expanding β in a B-spline basis:
/// `β(t) = Σ_k b_k φ_k(t)`
/// and solving a penalized regression problem.
#[derive(Debug, Clone)]
pub struct FunctionalLinearModel {
    /// Smoothing parameter for β(t) (None = GCV)
    pub lambda: Option<f64>,
    /// Derivative order for roughness penalty
    pub penalty_order: usize,
    /// Number of interior knots for β basis
    pub n_interior_knots: usize,
    /// Number of λ candidates for GCV
    pub n_lambda_grid: usize,
    /// Smoothing parameter for input curves (None = GCV each)
    pub curve_lambda: Option<f64>,
    /// Lambda search range
    pub lambda_min: f64,
    /// Lambda search range
    pub lambda_max: f64,
}

impl Default for FunctionalLinearModel {
    fn default() -> Self {
        Self {
            lambda: None,
            penalty_order: 2,
            n_interior_knots: 15,
            n_lambda_grid: 40,
            curve_lambda: Some(1e-4),
            lambda_min: 1e-10,
            lambda_max: 1e4,
        }
    }
}

impl FunctionalLinearModel {
    /// Fit the functional linear model.
    ///
    /// # Arguments
    /// - `t_list`: observation time grids for each functional predictor x_i(t)
    /// - `y_list`: functional predictor observations (one Array1 per subject)
    /// - `scalar_y`: scalar response for each subject
    pub fn fit(
        &self,
        t_list: &[Array1<f64>],
        y_list: &[Array1<f64>],
        scalar_y: &Array1<f64>,
    ) -> Result<FLMResult<BSplineBasis>> {
        let n = t_list.len();
        if n < 4 {
            return Err(TimeSeriesError::InsufficientData {
                message: "FunctionalLinearModel requires at least 4 observations".to_string(),
                required: 4,
                actual: n,
            });
        }
        if scalar_y.len() != n {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: scalar_y.len(),
            });
        }

        // Build β basis
        let basis = BSplineBasis::uniform(self.n_interior_knots, 4)?;
        let k = basis.n_basis();

        // Smooth each predictor curve and get coefficient matrix
        let pls_curve = PenalizedLeastSquares {
            lambda: self.curve_lambda,
            penalty_order: 2,
            n_lambda_grid: 20,
            lambda_min: 1e-10,
            lambda_max: 1e3,
        };

        let gram = basis.gram_matrix()?;

        // Build design matrix Z (n × k): Z[i, :] = G * c_i (Gram-weighted coefficients)
        // This represents ∫ x_i(t) φ_j(t) dt via the basis expansion
        let mut z = Array2::zeros((n, k));
        for (i, (t, y)) in t_list.iter().zip(y_list.iter()).enumerate() {
            let fd = pls_curve.fit(basis.clone(), t, y)?;
            let gc = gram.dot(&fd.coefficients);
            for j in 0..k {
                z[[i, j]] = gc[j];
            }
        }

        // Augment with intercept column: X = [1, Z] (n × (k+1))
        let mut x_design = Array2::zeros((n, k + 1));
        for i in 0..n {
            x_design[[i, 0]] = 1.0;
            for j in 0..k {
                x_design[[i, j + 1]] = z[[i, j]];
            }
        }

        // Build penalty matrix for β: [0, 0; 0, P] (first entry = intercept, no penalty)
        let penalty_beta = basis.penalty_matrix(self.penalty_order)?;
        let mut penalty_aug = Array2::zeros((k + 1, k + 1));
        for i in 0..k {
            for j in 0..k {
                penalty_aug[[i + 1, j + 1]] = penalty_beta[[i, j]];
            }
        }

        // Solve (X^T X + λ P) θ = X^T y with GCV or fixed λ
        let lambda = match self.lambda {
            Some(lam) => lam,
            None => {
                let gcv = crate::functional::smoothing::GCV::select(
                    self.lambda_min,
                    self.lambda_max,
                    self.n_lambda_grid,
                    |lam| {
                        let gcv_val = self.flm_gcv(&x_design, scalar_y, &penalty_aug, lam, n)?;
                        Ok(gcv_val)
                    },
                )?;
                gcv.optimal_lambda
            }
        };

        let theta = self.solve_flm(&x_design, scalar_y, &penalty_aug, lambda)?;
        let intercept = theta[0];
        let beta_coeff: Array1<f64> = theta.slice(s![1..]).to_owned();

        let fitted_values = x_design.dot(&theta);
        let residuals: Array1<f64> = scalar_y - &fitted_values;
        let rss: f64 = residuals.iter().map(|&r| r * r).sum();
        let mse = rss / n as f64;

        let y_mean = scalar_y.sum() / n as f64;
        let tss: f64 = scalar_y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
        let r_squared = if tss > 0.0 { 1.0 - rss / tss } else { 0.0 };
        let adjusted_r_squared = if tss > 0.0 && n > k + 1 {
            1.0 - (rss / (n - k - 1) as f64) / (tss / (n - 1) as f64)
        } else {
            r_squared
        };

        let beta_function = FunctionalData::new(basis, beta_coeff)?;

        Ok(FLMResult {
            intercept,
            beta_function,
            fitted_values,
            residuals,
            r_squared,
            adjusted_r_squared,
            rss,
            mse,
            lambda,
        })
    }

    fn flm_gcv(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        penalty: &Array2<f64>,
        lambda: f64,
        n: usize,
    ) -> Result<f64> {
        let k = x.ncols();
        let xtx = x.t().dot(x);
        let mut a = xtx;
        for i in 0..k {
            for j in 0..k {
                a[[i, j]] += lambda * penalty[[i, j]];
            }
        }
        let xty = x.t().dot(y);
        let theta = solve_linear_system(&a, &xty)?;
        let y_hat = x.dot(&theta);
        let rss: f64 = y.iter().zip(y_hat.iter()).map(|(yi, yhi)| (yi - yhi).powi(2)).sum();

        let a_inv_xt = solve_matrix_system(&a, &x.t().to_owned())?;
        let trace_h: f64 = (0..n)
            .map(|i| {
                let row = x.row(i);
                row.dot(&a_inv_xt.t().row(i).to_owned())
            })
            .sum();

        let denom = 1.0 - trace_h / n as f64;
        if denom.abs() < 1e-10 {
            return Ok(f64::INFINITY);
        }
        Ok((rss / n as f64) / (denom * denom))
    }

    fn solve_flm(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        penalty: &Array2<f64>,
        lambda: f64,
    ) -> Result<Array1<f64>> {
        let k = x.ncols();
        let xtx = x.t().dot(x);
        let mut a = xtx;
        for i in 0..k {
            for j in 0..k {
                a[[i, j]] += lambda * penalty[[i, j]];
            }
        }
        let xty = x.t().dot(y);
        solve_linear_system(&a, &xty)
    }
}

// ============================================================
// FunctionOnScalarRegression
// ============================================================

/// Result of function-on-scalar regression
#[derive(Debug, Clone)]
pub struct FoSResult<B: BasisSystem + Clone> {
    /// Intercept function α(t)
    pub intercept_function: FunctionalData<B>,
    /// Coefficient functions β_j(t) for each scalar predictor
    pub coefficient_functions: Vec<FunctionalData<B>>,
    /// Fitted functional values (n_obs × n_time)
    pub fitted_coefficients: Array2<f64>,
    /// Residual sum of squares (integrated)
    pub rss: f64,
    /// R² (integrated)
    pub r_squared: f64,
    /// Smoothing parameter used
    pub lambda: f64,
}

/// Function-on-scalar regression.
///
/// Models a functional response Y_i(t) given scalar predictors x_{i1}, ..., x_{ip}:
/// `Y_i(t) = α(t) + Σ_j x_{ij} β_j(t) + ε_i(t)`
///
/// Each coefficient function β_j(·) is estimated in a B-spline basis with
/// a roughness penalty.
#[derive(Debug, Clone)]
pub struct FunctionOnScalarRegression {
    /// Smoothing parameter for coefficient functions (None = GCV per function)
    pub lambda: Option<f64>,
    /// Penalty order
    pub penalty_order: usize,
    /// Interior knots for basis
    pub n_interior_knots: usize,
    /// Number of lambda candidates
    pub n_lambda_grid: usize,
    /// Lambda range
    pub lambda_min: f64,
    /// Lambda range
    pub lambda_max: f64,
}

impl Default for FunctionOnScalarRegression {
    fn default() -> Self {
        Self {
            lambda: None,
            penalty_order: 2,
            n_interior_knots: 15,
            n_lambda_grid: 40,
            lambda_min: 1e-10,
            lambda_max: 1e4,
        }
    }
}

impl FunctionOnScalarRegression {
    /// Fit function-on-scalar regression.
    ///
    /// # Arguments
    /// - `x_scalar`: (n_obs × p) matrix of scalar predictors (not including intercept)
    /// - `t_obs`: common time grid for all functional responses
    /// - `y_curves`: (n_obs × n_time) matrix of functional responses evaluated at `t_obs`
    pub fn fit(
        &self,
        x_scalar: &Array2<f64>,
        t_obs: &Array1<f64>,
        y_curves: &Array2<f64>,
    ) -> Result<FoSResult<BSplineBasis>> {
        let n = x_scalar.nrows();
        let p = x_scalar.ncols();
        if y_curves.nrows() != n {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: y_curves.nrows(),
            });
        }
        if y_curves.ncols() != t_obs.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: t_obs.len(),
                actual: y_curves.ncols(),
            });
        }
        if n < p + 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "FunctionOnScalarRegression needs more observations than predictors"
                    .to_string(),
                required: p + 2,
                actual: n,
            });
        }

        let basis = BSplineBasis::uniform(self.n_interior_knots, 4)?;
        let k = basis.n_basis();
        let phi = evaluate_basis_matrix(&basis, t_obs)?;
        let penalty = basis.penalty_matrix(self.penalty_order)?;

        // Design matrix with intercept: X_aug = [1, x_scalar] (n × (p+1))
        let mut x_aug = Array2::zeros((n, p + 1));
        for i in 0..n {
            x_aug[[i, 0]] = 1.0;
            for j in 0..p {
                x_aug[[i, j + 1]] = x_scalar[[i, j]];
            }
        }

        // For each time point (or better: each basis coefficient), fit the regression
        // Vectorize: for each basis function j, y_j = [y_curves * phi_j] (n-vector)
        // Then: y_j = X_aug * b_j  with b_j (p+1)-vector

        // First transform y_curves to basis coefficients for each observation
        // via least squares: c_i = argmin ||y_i - phi c||^2 + lambda ||P c||^2
        let pls = PenalizedLeastSquares {
            lambda: self.lambda,
            penalty_order: self.penalty_order,
            n_lambda_grid: self.n_lambda_grid,
            lambda_min: self.lambda_min,
            lambda_max: self.lambda_max,
        };

        let mut y_coeffs = Array2::zeros((n, k));
        for i in 0..n {
            let yi = y_curves.row(i).to_owned();
            let fd = pls.fit(basis.clone(), t_obs, &yi)?;
            for j in 0..k {
                y_coeffs[[i, j]] = fd.coefficients[j];
            }
        }

        // For each basis dimension j: regress y_coeffs[:, j] on X_aug
        // with penalty on β (not intercept)
        let mut coeff_functions_coeff = Array2::zeros((p + 1, k));
        let xtx = x_aug.t().dot(&x_aug);
        let mut pen_aug = Array2::zeros((p + 1, p + 1));
        // No penalty on intercept (index 0), add identity ridge for stability
        for i in 1..(p + 1) {
            pen_aug[[i, i]] = 1.0; // simplified: use identity penalty on x coefficients
        }

        let lambda_used = match self.lambda {
            Some(lam) => lam,
            None => {
                // Use simple GCV on first time point as representative
                let y_first = y_coeffs.column(0).to_owned();
                let gcv = crate::functional::smoothing::GCV::select(
                    self.lambda_min,
                    self.lambda_max,
                    self.n_lambda_grid,
                    |lam| {
                        let mut a = xtx.clone();
                        for i in 0..(p + 1) {
                            for j in 0..(p + 1) {
                                a[[i, j]] += lam * pen_aug[[i, j]];
                            }
                        }
                        let xty = x_aug.t().dot(&y_first);
                        let b = solve_linear_system(&a, &xty)?;
                        let yhat = x_aug.dot(&b);
                        let rss: f64 =
                            y_first.iter().zip(yhat.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                        let a_inv_xt = solve_matrix_system(&a, &x_aug.t().to_owned())?;
                        let trace_h: f64 = (0..n)
                            .map(|i| x_aug.row(i).dot(&a_inv_xt.t().row(i).to_owned()))
                            .sum();
                        let denom = 1.0 - trace_h / n as f64;
                        if denom.abs() < 1e-10 {
                            return Ok(f64::INFINITY);
                        }
                        Ok((rss / n as f64) / (denom * denom))
                    },
                )?;
                gcv.optimal_lambda
            }
        };

        let mut a = xtx.clone();
        for i in 0..(p + 1) {
            for j in 0..(p + 1) {
                a[[i, j]] += lambda_used * pen_aug[[i, j]];
            }
        }

        for j in 0..k {
            let yj = y_coeffs.column(j).to_owned();
            let xtyi = x_aug.t().dot(&yj);
            let bj = solve_linear_system(&a, &xtyi)?;
            for i in 0..(p + 1) {
                coeff_functions_coeff[[i, j]] = bj[i];
            }
        }

        // Extract intercept and predictor coefficient functions
        let intercept_coeff: Array1<f64> = coeff_functions_coeff.row(0).to_owned();
        let intercept_function = FunctionalData::new(basis.clone(), intercept_coeff)?;

        let mut coefficient_functions = Vec::with_capacity(p);
        for i in 0..p {
            let beta_coeff: Array1<f64> = coeff_functions_coeff.row(i + 1).to_owned();
            coefficient_functions.push(FunctionalData::new(basis.clone(), beta_coeff)?);
        }

        // Compute fitted coefficient matrix
        let fitted_coefficients = x_aug.dot(&coeff_functions_coeff);

        // Compute RSS (integrated over t)
        let gram = basis.gram_matrix()?;
        let mut rss = 0.0;
        let mut tss = 0.0;
        let mean_coeff: Array1<f64> = y_coeffs.mean_axis(Axis(0)).ok_or_else(|| {
            TimeSeriesError::ComputationError("mean computation failed".to_string())
        })?;
        for i in 0..n {
            let res: Array1<f64> = y_coeffs.row(i).to_owned() - &fitted_coefficients.row(i).to_owned();
            let dev: Array1<f64> = y_coeffs.row(i).to_owned() - &mean_coeff;
            rss += res.dot(&gram.dot(&res));
            tss += dev.dot(&gram.dot(&dev));
        }
        let r_squared = if tss > 0.0 { 1.0 - rss / tss } else { 0.0 };

        Ok(FoSResult {
            intercept_function,
            coefficient_functions,
            fitted_coefficients,
            rss,
            r_squared,
            lambda: lambda_used,
        })
    }
}

// ============================================================
// ConcurrentModel: Varying-Coefficient Regression
// ============================================================

/// Result of concurrent functional regression
#[derive(Debug, Clone)]
pub struct ConcurrentResult<B: BasisSystem + Clone> {
    /// Intercept function α(t)
    pub intercept_function: FunctionalData<B>,
    /// Slope function β(t) (the varying coefficient)
    pub slope_function: FunctionalData<B>,
    /// Fitted functional values at observation points (n_obs × n_time)
    pub fitted_values: Array2<f64>,
    /// Residuals (n_obs × n_time)
    pub residuals: Array2<f64>,
    /// R² (integrated)
    pub r_squared: f64,
}

/// Concurrent (varying-coefficient) functional regression.
///
/// Models the relationship between two functional variables at the SAME time point:
/// `Y_i(t) = α(t) + β(t) X_i(t) + ε_i(t)`
///
/// The coefficient function β(t) is estimated by pointwise weighted regression
/// then smoothed using a B-spline penalty.
#[derive(Debug, Clone)]
pub struct ConcurrentModel {
    /// Smoothing parameter for coefficient functions
    pub lambda: Option<f64>,
    /// Penalty order
    pub penalty_order: usize,
    /// Interior knots for basis expansion
    pub n_interior_knots: usize,
    /// Lambda search range
    pub lambda_min: f64,
    /// Lambda range
    pub lambda_max: f64,
    /// Number of lambda candidates for GCV
    pub n_lambda_grid: usize,
}

impl Default for ConcurrentModel {
    fn default() -> Self {
        Self {
            lambda: None,
            penalty_order: 2,
            n_interior_knots: 15,
            lambda_min: 1e-10,
            lambda_max: 1e4,
            n_lambda_grid: 40,
        }
    }
}

impl ConcurrentModel {
    /// Fit a concurrent model.
    ///
    /// # Arguments
    /// - `t_obs`: common time grid (length T)
    /// - `x_curves`: (n_obs × T) matrix of predictor functional values
    /// - `y_curves`: (n_obs × T) matrix of response functional values
    pub fn fit(
        &self,
        t_obs: &Array1<f64>,
        x_curves: &Array2<f64>,
        y_curves: &Array2<f64>,
    ) -> Result<ConcurrentResult<BSplineBasis>> {
        let n_t = t_obs.len();
        let n_obs = x_curves.nrows();
        if x_curves.ncols() != n_t || y_curves.nrows() != n_obs || y_curves.ncols() != n_t {
            return Err(TimeSeriesError::InvalidInput(
                "x_curves and y_curves must have same shape (n_obs × n_t)".to_string(),
            ));
        }
        if n_obs < 3 {
            return Err(TimeSeriesError::InsufficientData {
                message: "ConcurrentModel requires at least 3 observations".to_string(),
                required: 3,
                actual: n_obs,
            });
        }

        // Step 1: Pointwise OLS at each time point to get raw α(t) and β(t) estimates
        let mut alpha_raw = Array1::zeros(n_t);
        let mut beta_raw = Array1::zeros(n_t);
        for s in 0..n_t {
            let xs: Array1<f64> = x_curves.column(s).to_owned();
            let ys: Array1<f64> = y_curves.column(s).to_owned();
            let x_mean = xs.sum() / n_obs as f64;
            let y_mean = ys.sum() / n_obs as f64;
            let sxx: f64 = xs.iter().map(|&xi| (xi - x_mean).powi(2)).sum();
            let sxy: f64 = xs.iter().zip(ys.iter()).map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean)).sum();
            let beta_s = if sxx > 1e-15 { sxy / sxx } else { 0.0 };
            let alpha_s = y_mean - beta_s * x_mean;
            alpha_raw[s] = alpha_s;
            beta_raw[s] = beta_s;
        }

        // Step 2: Smooth the raw estimates using penalized B-splines
        let pls = PenalizedLeastSquares {
            lambda: self.lambda,
            penalty_order: self.penalty_order,
            n_lambda_grid: self.n_lambda_grid,
            lambda_min: self.lambda_min,
            lambda_max: self.lambda_max,
        };

        let basis = BSplineBasis::uniform(self.n_interior_knots, 4)?;
        let alpha_fd = pls.fit(basis.clone(), t_obs, &alpha_raw)?;
        let beta_fd = pls.fit(basis.clone(), t_obs, &beta_raw)?;

        // Step 3: Compute fitted values and R²
        let alpha_fitted = alpha_fd.eval_vec(t_obs)?;
        let beta_fitted = beta_fd.eval_vec(t_obs)?;

        let mut fitted_values = Array2::zeros((n_obs, n_t));
        let mut residuals = Array2::zeros((n_obs, n_t));
        for i in 0..n_obs {
            for s in 0..n_t {
                let yhat = alpha_fitted[s] + beta_fitted[s] * x_curves[[i, s]];
                fitted_values[[i, s]] = yhat;
                residuals[[i, s]] = y_curves[[i, s]] - yhat;
            }
        }

        // R² (averaged over time)
        let mut rss = 0.0;
        let mut tss = 0.0;
        for s in 0..n_t {
            let y_mean = y_curves.column(s).sum() / n_obs as f64;
            for i in 0..n_obs {
                rss += residuals[[i, s]].powi(2);
                tss += (y_curves[[i, s]] - y_mean).powi(2);
            }
        }
        let r_squared = if tss > 0.0 { 1.0 - rss / tss } else { 0.0 };

        Ok(ConcurrentResult {
            intercept_function: alpha_fd,
            slope_function: beta_fd,
            fitted_values,
            residuals,
            r_squared,
        })
    }
}

// ============================================================
// FunctionalANOVA
// ============================================================

/// Result of functional ANOVA
#[derive(Debug, Clone)]
pub struct FunctionalANOVAResult<B: BasisSystem + Clone> {
    /// Group mean functions
    pub group_means: Vec<FunctionalData<B>>,
    /// Grand mean function
    pub grand_mean: FunctionalData<B>,
    /// Observed F-statistic function F(t)
    pub f_statistic: Array1<f64>,
    /// Integrated F-statistic (scalar test statistic)
    pub integrated_f: f64,
    /// P-value from permutation test
    pub p_value: f64,
    /// Number of permutations used
    pub n_permutations: usize,
    /// Time grid used for evaluation
    pub t_grid: Array1<f64>,
}

/// One-way Functional ANOVA with permutation test.
///
/// Tests H₀: μ₁(t) = μ₂(t) = ... = μ_G(t) ∀t
/// using the integrated F-statistic:
/// `F_int = ∫ F(t) dt`
///
/// Significance is assessed via permutation of group labels.
#[derive(Debug, Clone)]
pub struct FunctionalANOVA {
    /// Number of permutation replications
    pub n_permutations: usize,
    /// Smoothing parameter for group means
    pub smooth_lambda: Option<f64>,
    /// Number of interior knots
    pub n_interior_knots: usize,
    /// Number of evaluation points for the F-statistic function
    pub n_eval_points: usize,
    /// Random seed for reproducibility (None = entropy)
    pub seed: Option<u64>,
}

impl Default for FunctionalANOVA {
    fn default() -> Self {
        Self {
            n_permutations: 999,
            smooth_lambda: Some(1e-4),
            n_interior_knots: 15,
            n_eval_points: 100,
            seed: Some(42),
        }
    }
}

impl FunctionalANOVA {
    /// Fit functional ANOVA.
    ///
    /// # Arguments
    /// - `groups`: A slice of groups, each containing a slice of (t_i, y_i) tuples.
    ///   All functional observations share a common domain.
    pub fn fit(
        &self,
        groups: &[Vec<(Array1<f64>, Array1<f64>)>],
    ) -> Result<FunctionalANOVAResult<BSplineBasis>> {
        let g = groups.len();
        if g < 2 {
            return Err(TimeSeriesError::InvalidInput(
                "FunctionalANOVA requires at least 2 groups".to_string(),
            ));
        }

        let total_n: usize = groups.iter().map(|grp| grp.len()).sum();
        if total_n < g + 1 {
            return Err(TimeSeriesError::InsufficientData {
                message: "FunctionalANOVA needs more observations than groups".to_string(),
                required: g + 1,
                actual: total_n,
            });
        }

        let basis = BSplineBasis::uniform(self.n_interior_knots, 4)?;
        let pls = PenalizedLeastSquares {
            lambda: self.smooth_lambda,
            penalty_order: 2,
            n_lambda_grid: 20,
            lambda_min: 1e-10,
            lambda_max: 1e3,
        };

        // Evaluation grid
        let t_grid: Array1<f64> = Array1::from_vec(
            (0..self.n_eval_points)
                .map(|i| i as f64 / (self.n_eval_points - 1) as f64)
                .collect(),
        );

        // Smooth all observations, collect group membership
        let mut all_curves: Vec<Array1<f64>> = Vec::new(); // evaluated at t_grid
        let mut group_labels: Vec<usize> = Vec::new();

        for (gi, group) in groups.iter().enumerate() {
            for (t, y) in group.iter() {
                let fd = pls.fit(basis.clone(), t, y)?;
                let curve = fd.eval_vec(&t_grid)?;
                all_curves.push(curve);
                group_labels.push(gi);
            }
        }

        let n_total = all_curves.len();
        let n_t = t_grid.len();

        // Build (n_total × n_t) matrix
        let mut curve_mat = Array2::zeros((n_total, n_t));
        for (i, curve) in all_curves.iter().enumerate() {
            for s in 0..n_t {
                curve_mat[[i, s]] = curve[s];
            }
        }

        // Compute group sizes
        let group_sizes: Vec<usize> = groups.iter().map(|grp| grp.len()).collect();

        // Observed F-statistic
        let f_obs = compute_f_statistic(&curve_mat, &group_labels, &group_sizes, n_t, g);
        let integrated_f_obs: f64 = f_obs.iter().sum::<f64>() / n_t as f64;

        // Permutation test
        let mut perm_labels = group_labels.clone();
        let mut count_exceed = 0usize;

        // Simple LCG for reproducible permutation
        let mut rng_state = self.seed.unwrap_or(12345u64);
        let lcg_mult = 6364136223846793005u64;
        let lcg_inc = 1442695040888963407u64;
        let next_rand = |state: &mut u64| -> f64 {
            *state = state.wrapping_mul(lcg_mult).wrapping_add(lcg_inc);
            (*state >> 33) as f64 / (1u64 << 31) as f64
        };

        for _ in 0..self.n_permutations {
            // Fisher-Yates shuffle
            for i in (1..n_total).rev() {
                let j = (next_rand(&mut rng_state) * (i + 1) as f64) as usize;
                let j = j.min(i);
                perm_labels.swap(i, j);
            }
            let f_perm =
                compute_f_statistic(&curve_mat, &perm_labels, &group_sizes, n_t, g);
            let integrated_f_perm: f64 = f_perm.iter().sum::<f64>() / n_t as f64;
            if integrated_f_perm >= integrated_f_obs {
                count_exceed += 1;
            }
        }

        let p_value = (count_exceed + 1) as f64 / (self.n_permutations + 1) as f64;

        // Compute group mean functions
        let grand_mean_curve: Array1<f64> = {
            let sum: Array1<f64> = curve_mat.mean_axis(Axis(0)).ok_or_else(|| {
                TimeSeriesError::ComputationError("mean computation failed".to_string())
            })?;
            sum
        };

        // Smooth group means for the result
        let mut group_means = Vec::with_capacity(g);
        for (gi, group) in groups.iter().enumerate() {
            let indices: Vec<usize> = group_labels
                .iter()
                .enumerate()
                .filter(|(_, &l)| l == gi)
                .map(|(i, _)| i)
                .collect();
            let mut mean_curve = Array1::zeros(n_t);
            for &idx in &indices {
                for s in 0..n_t {
                    mean_curve[s] += curve_mat[[idx, s]];
                }
            }
            let ni = indices.len().max(1);
            mean_curve.mapv_inplace(|x| x / ni as f64);

            // Fit the mean curve as a FunctionalData
            let fd = pls.fit(basis.clone(), &t_grid, &mean_curve)?;
            group_means.push(fd);
        }

        let grand_fd = pls.fit(basis.clone(), &t_grid, &grand_mean_curve)?;

        Ok(FunctionalANOVAResult {
            group_means,
            grand_mean: grand_fd,
            f_statistic: f_obs,
            integrated_f: integrated_f_obs,
            p_value,
            n_permutations: self.n_permutations,
            t_grid,
        })
    }
}

/// Compute pointwise F-statistic at each time point
fn compute_f_statistic(
    curve_mat: &Array2<f64>,
    labels: &[usize],
    group_sizes: &[usize],
    n_t: usize,
    n_groups: usize,
) -> Array1<f64> {
    let n_total = curve_mat.nrows();
    let mut f_stat = Array1::zeros(n_t);

    for s in 0..n_t {
        let overall_mean = curve_mat.column(s).sum() / n_total as f64;

        // Between-group sum of squares
        let mut ss_between = 0.0;
        for g in 0..n_groups {
            let g_indices: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter(|(_, &l)| l == g)
                .map(|(i, _)| i)
                .collect();
            let ng = g_indices.len() as f64;
            if ng < 1.0 {
                continue;
            }
            let g_mean: f64 = g_indices.iter().map(|&i| curve_mat[[i, s]]).sum::<f64>() / ng;
            ss_between += ng * (g_mean - overall_mean).powi(2);
        }

        // Within-group sum of squares
        let mut ss_within = 0.0;
        for g in 0..n_groups {
            let g_indices: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter(|(_, &l)| l == g)
                .map(|(i, _)| i)
                .collect();
            let ng = g_indices.len() as f64;
            if ng < 1.0 {
                continue;
            }
            let g_mean: f64 = g_indices.iter().map(|&i| curve_mat[[i, s]]).sum::<f64>() / ng;
            for &i in &g_indices {
                ss_within += (curve_mat[[i, s]] - g_mean).powi(2);
            }
        }

        let df_between = (n_groups - 1) as f64;
        let df_within = (n_total - n_groups) as f64;
        if ss_within > 1e-15 && df_within > 0.0 {
            f_stat[s] = (ss_between / df_between) / (ss_within / df_within);
        } else {
            f_stat[s] = 0.0;
        }
    }
    f_stat
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn make_functional_predictors(
        n: usize,
    ) -> (Vec<Array1<f64>>, Vec<Array1<f64>>, Array1<f64>) {
        let n_t = 40;
        let t: Array1<f64> = Array1::from_vec(
            (0..n_t).map(|i| i as f64 / (n_t - 1) as f64 * PI).collect(),
        );
        let mut t_list = Vec::new();
        let mut y_list = Vec::new();
        let mut scalars = Vec::new();
        for i in 0..n {
            let phase = i as f64 * 0.2;
            let y: Array1<f64> = t.mapv(|ti| (ti + phase).sin());
            // Scalar response: integral of x(t) * cos(t) (known analytical form)
            let scalar: f64 = y.iter().zip(t.iter()).map(|(&yi, &ti)| yi * ti.cos()).sum::<f64>() / n_t as f64;
            t_list.push(t.clone());
            y_list.push(y);
            scalars.push(scalar);
        }
        (t_list, y_list, Array1::from(scalars))
    }

    #[test]
    fn test_functional_linear_model() {
        let (t_list, y_list, scalar_y) = make_functional_predictors(20);
        let flm = FunctionalLinearModel {
            lambda: Some(1e-3),
            n_interior_knots: 8,
            ..Default::default()
        };
        let result = flm.fit(&t_list, &y_list, &scalar_y).expect("FLM fit failed");
        assert_eq!(result.fitted_values.len(), 20);
        assert!(result.r_squared.is_finite());
    }

    #[test]
    fn test_function_on_scalar_regression() {
        let n = 20;
        let p = 2;
        let n_t = 30;
        let t_obs: Array1<f64> = Array1::from_vec(
            (0..n_t).map(|i| i as f64 / (n_t - 1) as f64 * PI).collect(),
        );
        let mut x_scalar = Array2::zeros((n, p));
        let mut y_curves = Array2::zeros((n, n_t));
        for i in 0..n {
            let x1 = i as f64 / n as f64;
            let x2 = (i as f64 * 0.3).sin();
            x_scalar[[i, 0]] = x1;
            x_scalar[[i, 1]] = x2;
            for s in 0..n_t {
                let t = t_obs[s];
                y_curves[[i, s]] = 1.0 + x1 * t.sin() + x2 * t.cos();
            }
        }
        let fos = FunctionOnScalarRegression {
            lambda: Some(1e-3),
            n_interior_knots: 8,
            ..Default::default()
        };
        let result = fos.fit(&x_scalar, &t_obs, &y_curves).expect("FoS fit failed");
        assert!(result.r_squared > 0.0);
    }

    #[test]
    fn test_concurrent_model() {
        let n_obs = 25;
        let n_t = 40;
        let t_obs: Array1<f64> = Array1::from_vec(
            (0..n_t).map(|i| i as f64 / (n_t - 1) as f64 * PI).collect(),
        );
        let mut x_curves = Array2::zeros((n_obs, n_t));
        let mut y_curves = Array2::zeros((n_obs, n_t));
        for i in 0..n_obs {
            for s in 0..n_t {
                let t = t_obs[s];
                let x = (t + i as f64 * 0.1).sin();
                x_curves[[i, s]] = x;
                y_curves[[i, s]] = 0.5 + 2.0 * t.cos() * x; // β(t) = 2*cos(t)
            }
        }
        let cm = ConcurrentModel {
            lambda: Some(1e-3),
            n_interior_knots: 8,
            ..Default::default()
        };
        let result = cm.fit(&t_obs, &x_curves, &y_curves).expect("concurrent fit failed");
        assert!(result.r_squared > 0.0);
    }

    #[test]
    fn test_functional_anova() {
        let n_t = 30;
        let t_obs: Array1<f64> = Array1::from_vec(
            (0..n_t).map(|i| i as f64 / (n_t - 1) as f64 * PI).collect(),
        );
        // Create 3 groups with distinct mean functions
        let mut groups: Vec<Vec<(Array1<f64>, Array1<f64>)>> = Vec::new();
        for g in 0..3 {
            let mut obs = Vec::new();
            for i in 0..5 {
                let y: Array1<f64> = t_obs.mapv(|t| (t + g as f64 * 0.5).sin() + i as f64 * 0.01);
                obs.push((t_obs.clone(), y));
            }
            groups.push(obs);
        }
        let anova = FunctionalANOVA {
            n_permutations: 99,
            smooth_lambda: Some(1e-3),
            n_interior_knots: 6,
            n_eval_points: 20,
            seed: Some(42),
        };
        let result = anova.fit(&groups).expect("ANOVA fit failed");
        assert_eq!(result.group_means.len(), 3);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }
}
