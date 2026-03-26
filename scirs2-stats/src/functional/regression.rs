//! Functional regression models.
//!
//! Provides scalar-on-function and function-on-function regression:
//!
//! - **Scalar-on-function**: Predict a scalar response from functional predictors.
//!   Model: y_i = alpha + integral x_i(t) * beta(t) dt + epsilon_i
//!
//! - **Function-on-function**: Predict a functional response from functional predictors.
//!   Model: y_i(t) = integral x_i(s) * beta(s,t) ds + epsilon_i(t)
//!
//! Both use penalized basis expansion for the coefficient function(s).

use scirs2_core::ndarray::{Array1, Array2};

use super::basis::{compute_basis_coefficients, evaluate_basis};
use super::types::{BasisType, FoFResult, FunctionalConfig, FunctionalData, SoFResult};
use crate::error::{StatsError, StatsResult};

/// Compute R-squared (coefficient of determination).
///
/// R^2 = 1 - SS_res / SS_tot
///
/// Returns a value in (-inf, 1]. A value of 1.0 indicates perfect fit.
pub fn r_squared(y: &[f64], y_hat: &[f64]) -> f64 {
    let n = y.len();
    if n == 0 {
        return 0.0;
    }
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = y
        .iter()
        .zip(y_hat.iter())
        .map(|(&yi, &yh)| (yi - yh).powi(2))
        .sum();

    if ss_tot < 1e-14 {
        return if ss_res < 1e-14 { 1.0 } else { 0.0 };
    }
    1.0 - ss_res / ss_tot
}

/// Scalar-on-function regression.
///
/// Model: y_i = alpha + integral x_i(t) * beta(t) dt + epsilon_i
///
/// The coefficient function beta(t) is expanded in a basis:
///   beta(t) = sum_k b_k * phi_k(t)
///
/// The integral becomes: integral x_i(t) * beta(t) dt approx= sum_j x_i(t_j) * beta(t_j) * dt_j
///
/// After discretization: y = alpha + X_int * b + epsilon
/// where `X_int[i,k] = sum_j x_i(t_j) * phi_k(t_j) * dt_j`
///
/// Penalized least squares: b = (X_int' X_int + lambda * P)^{-1} X_int' (y - alpha)
pub struct ScalarOnFunctionRegression;

impl ScalarOnFunctionRegression {
    /// Fit a scalar-on-function regression model.
    ///
    /// # Arguments
    /// * `data` - Functional predictor data (n curves on a grid)
    /// * `y` - Scalar responses (length n)
    /// * `config` - Configuration (basis type, smoothing parameter)
    ///
    /// # Returns
    /// A `SoFResult` containing the estimated coefficient function, intercept, etc.
    ///
    /// # Errors
    /// Returns an error if dimensions mismatch or the linear system cannot be solved.
    pub fn fit(
        data: &FunctionalData,
        y: &[f64],
        config: &FunctionalConfig,
    ) -> StatsResult<SoFResult> {
        let n_curves = data.n_curves();
        let n_grid = data.n_grid();

        if y.len() != n_curves {
            return Err(StatsError::DimensionMismatch(format!(
                "y has length {}, but data has {} curves",
                y.len(),
                n_curves
            )));
        }

        // Evaluate basis
        let phi = evaluate_basis(&data.grid, &config.basis)?;
        let n_basis = phi.ncols();

        // Compute grid spacing for numerical integration (trapezoidal)
        let dt = compute_grid_spacing(&data.grid);

        // Build the design matrix X_int: X_int[i,k] = integral x_i(t) * phi_k(t) dt
        // Using raw predictor curves (the basis expansion of beta provides smoothing)
        let mut x_int = Array2::<f64>::zeros((n_curves, n_basis));
        for i in 0..n_curves {
            for k in 0..n_basis {
                let mut integral = 0.0;
                for j in 0..n_grid {
                    integral += data.observations[i][j] * phi[[j, k]] * dt[j];
                }
                x_int[[i, k]] = integral;
            }
        }

        // Compute intercept: y_mean
        let y_arr = Array1::from_vec(y.to_vec());
        let y_mean = y_arr.mean().unwrap_or(0.0);

        // Center y
        let y_centered: Array1<f64> = y_arr
            .iter()
            .map(|&yi| yi - y_mean)
            .collect::<Vec<_>>()
            .into();

        // Determine lambda for the coefficient function penalty
        let lambda_beta = self_select_lambda_sof(&x_int, &y_centered, n_basis);

        // Solve: (X_int' X_int + lambda * P) b = X_int' y_centered
        let xtx = x_int.t().dot(&x_int);
        let xty = x_int.t().dot(&y_centered);
        let penalty = second_derivative_penalty(n_basis);
        let mut lhs = xtx + &(&penalty * lambda_beta);

        let b = solve_symmetric_positive(&mut lhs, &xty)?;

        // Evaluate beta(t) = phi * b
        let beta_func = phi.dot(&b);

        // Compute fitted values
        let fitted_centered = x_int.dot(&b);
        let fitted: Array1<f64> = fitted_centered
            .iter()
            .map(|&fc| fc + y_mean)
            .collect::<Vec<_>>()
            .into();

        let r2 = r_squared(y, fitted.as_slice().unwrap_or(&[]));

        Ok(SoFResult {
            beta: beta_func,
            intercept: y_mean,
            beta_coefficients: b,
            basis: config.basis.clone(),
            grid: data.grid.clone(),
            lambda: lambda_beta,
            fitted_values: fitted,
            r_squared: r2,
        })
    }

    /// Predict scalar responses for new functional data.
    ///
    /// # Arguments
    /// * `result` - Fitted model result
    /// * `new_data` - New functional predictor data
    ///
    /// # Returns
    /// Predicted scalar responses.
    ///
    /// # Errors
    /// Returns an error if grids don't match or basis evaluation fails.
    pub fn predict(result: &SoFResult, new_data: &FunctionalData) -> StatsResult<Vec<f64>> {
        if new_data.grid.len() != result.grid.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "New data grid has {} points, model was fitted on {} points",
                new_data.grid.len(),
                result.grid.len()
            )));
        }

        let phi = evaluate_basis(&new_data.grid, &result.basis)?;
        let dt = compute_grid_spacing(&new_data.grid);
        let n_grid = new_data.n_grid();
        let n_basis = phi.ncols();

        let mut predictions = Vec::with_capacity(new_data.n_curves());
        for obs in &new_data.observations {
            // Compute integral: sum_j x(t_j) * phi_k(t_j) * dt_j
            let mut x_int_row = Array1::<f64>::zeros(n_basis);
            for k in 0..n_basis {
                let mut integral = 0.0;
                for j in 0..n_grid {
                    integral += obs[j] * phi[[j, k]] * dt[j];
                }
                x_int_row[k] = integral;
            }

            let pred_centered: f64 = x_int_row
                .iter()
                .zip(result.beta_coefficients.iter())
                .map(|(&x, &b)| x * b)
                .sum();
            predictions.push(pred_centered + result.intercept);
        }

        Ok(predictions)
    }
}

/// Function-on-function regression.
///
/// Model: y_i(t) = integral x_i(s) * beta(s,t) ds + epsilon_i(t)
///
/// The bivariate coefficient beta(s,t) is expanded in a tensor product basis:
///   beta(s,t) = sum_{j,k} B_{jk} * phi_j(s) * psi_k(t)
///
/// This is solved via penalized least squares on the vectorized coefficients.
pub struct FunctionOnFunctionRegression;

impl FunctionOnFunctionRegression {
    /// Fit a function-on-function regression model.
    ///
    /// # Arguments
    /// * `predictor_data` - Functional predictor data (n curves)
    /// * `response_data` - Functional response data (n curves, possibly different grid)
    /// * `predictor_config` - Configuration for predictor basis
    /// * `response_config` - Configuration for response basis
    ///
    /// # Returns
    /// A `FoFResult` containing the bivariate coefficient surface, fitted curves, etc.
    ///
    /// # Errors
    /// Returns an error if dimensions mismatch or the linear system fails.
    pub fn fit(
        predictor_data: &FunctionalData,
        response_data: &FunctionalData,
        predictor_config: &FunctionalConfig,
        response_config: &FunctionalConfig,
    ) -> StatsResult<FoFResult> {
        let n_curves = predictor_data.n_curves();
        if n_curves != response_data.n_curves() {
            return Err(StatsError::DimensionMismatch(format!(
                "Predictor has {} curves, response has {} curves",
                n_curves,
                response_data.n_curves()
            )));
        }

        let n_grid_s = predictor_data.n_grid();
        let n_grid_t = response_data.n_grid();

        // Evaluate predictor and response bases
        let phi_s = evaluate_basis(&predictor_data.grid, &predictor_config.basis)?;
        let phi_t = evaluate_basis(&response_data.grid, &response_config.basis)?;
        let n_basis_s = phi_s.ncols();
        let n_basis_t = phi_t.ncols();
        let n_total = n_basis_s * n_basis_t;

        // Grid spacings
        let ds = compute_grid_spacing(&predictor_data.grid);
        let dt = compute_grid_spacing(&response_data.grid);

        // Smoothing parameters
        let lambda_s = predictor_config.smoothing_param.unwrap_or(1e-4);
        let lambda_t = response_config.smoothing_param.unwrap_or(1e-4);

        // Smooth predictor curves and compute integral design matrices
        // For each curve i and predictor basis j:
        //   X_s[i,j] = integral x_i(s) * phi_j(s) ds
        let mut x_s = Array2::<f64>::zeros((n_curves, n_basis_s));
        for i in 0..n_curves {
            let coeffs =
                compute_basis_coefficients(&predictor_data.observations[i], &phi_s, lambda_s)?;
            let smoothed = phi_s.dot(&coeffs);
            for j in 0..n_basis_s {
                let mut integral = 0.0;
                for l in 0..n_grid_s {
                    integral += smoothed[l] * phi_s[[l, j]] * ds[l];
                }
                x_s[[i, j]] = integral;
            }
        }

        // Smooth response curves and get coefficients
        let mut y_coeffs = Array2::<f64>::zeros((n_curves, n_basis_t));
        for i in 0..n_curves {
            let coeffs =
                compute_basis_coefficients(&response_data.observations[i], &phi_t, lambda_t)?;
            for k in 0..n_basis_t {
                y_coeffs[[i, k]] = coeffs[k];
            }
        }

        // Build the regression: for each response basis k:
        //   y_coeffs[:, k] = X_s * B[:, k]
        // This is n_basis_t separate regressions, or one big system.
        //
        // We solve column-by-column with a penalty on B.
        let penalty_s = second_derivative_penalty(n_basis_s);
        let lambda_pen = (lambda_s + lambda_t) / 2.0; // Combined penalty

        let xtx = x_s.t().dot(&x_s);
        let mut lhs = &xtx + &(&penalty_s * lambda_pen);

        let mut b_matrix = Array2::<f64>::zeros((n_basis_s, n_basis_t));
        for k in 0..n_basis_t {
            let rhs = x_s.t().dot(&y_coeffs.column(k));
            let b_col = solve_symmetric_positive(&mut lhs, &rhs)?;
            for j in 0..n_basis_s {
                b_matrix[[j, k]] = b_col[j];
            }
        }

        // Evaluate beta(s,t) on the grids
        // beta(s,t) = sum_{j,k} B_{jk} * phi_j(s) * psi_k(t)
        let mut beta_surface = Array2::<f64>::zeros((n_grid_s, n_grid_t));
        for si in 0..n_grid_s {
            for ti in 0..n_grid_t {
                let mut val = 0.0;
                for j in 0..n_basis_s {
                    for k in 0..n_basis_t {
                        val += b_matrix[[j, k]] * phi_s[[si, j]] * phi_t[[ti, k]];
                    }
                }
                beta_surface[[si, ti]] = val;
            }
        }

        // Compute fitted response curves
        // y_hat_i(t) = sum_k (sum_j X_s[i,j] * B[j,k]) * psi_k(t)
        let y_hat_coeffs = x_s.dot(&b_matrix); // (n_curves, n_basis_t)
        let fitted_curves = y_hat_coeffs.dot(&phi_t.t()); // (n_curves, n_grid_t)

        // Vectorize B for storage
        let mut beta_vec = Array1::<f64>::zeros(n_total);
        for j in 0..n_basis_s {
            for k in 0..n_basis_t {
                beta_vec[j * n_basis_t + k] = b_matrix[[j, k]];
            }
        }

        Ok(FoFResult {
            beta_surface,
            beta_coefficients: beta_vec,
            predictor_basis: predictor_config.basis.clone(),
            response_basis: response_config.basis.clone(),
            predictor_grid: predictor_data.grid.clone(),
            response_grid: response_data.grid.clone(),
            lambda: lambda_pen,
            fitted_curves,
        })
    }

    /// Predict response curves for new predictor data.
    ///
    /// # Arguments
    /// * `result` - Fitted model result
    /// * `new_data` - New functional predictor data
    ///
    /// # Returns
    /// Predicted response curves, one per input curve.
    ///
    /// # Errors
    /// Returns an error if grids don't match or computation fails.
    pub fn predict(result: &FoFResult, new_data: &FunctionalData) -> StatsResult<Vec<Vec<f64>>> {
        let n_grid_s = result.predictor_grid.len();
        if new_data.n_grid() != n_grid_s {
            return Err(StatsError::DimensionMismatch(format!(
                "New data grid has {} points, predictor was fitted on {} points",
                new_data.n_grid(),
                n_grid_s
            )));
        }

        let phi_s = evaluate_basis(&new_data.grid, &result.predictor_basis)?;
        let phi_t = evaluate_basis(&result.response_grid, &result.response_basis)?;
        let n_basis_s = phi_s.ncols();
        let n_basis_t = phi_t.ncols();
        let ds = compute_grid_spacing(&new_data.grid);

        // Reconstruct B matrix
        let mut b_matrix = Array2::<f64>::zeros((n_basis_s, n_basis_t));
        for j in 0..n_basis_s {
            for k in 0..n_basis_t {
                b_matrix[[j, k]] = result.beta_coefficients[j * n_basis_t + k];
            }
        }

        let lambda_smooth = result.lambda.min(1e-4);
        let mut predictions = Vec::with_capacity(new_data.n_curves());

        for obs in &new_data.observations {
            let coeffs = compute_basis_coefficients(obs, &phi_s, lambda_smooth)?;
            let smoothed = phi_s.dot(&coeffs);

            // Compute X_s row
            let mut x_row = Array1::<f64>::zeros(n_basis_s);
            for j in 0..n_basis_s {
                let mut integral = 0.0;
                for l in 0..n_grid_s {
                    integral += smoothed[l] * phi_s[[l, j]] * ds[l];
                }
                x_row[j] = integral;
            }

            // y_hat(t) = sum_k (sum_j x_row[j] * B[j,k]) * psi_k(t)
            let coeff_t = x_row.dot(&b_matrix); // (n_basis_t,)
            let pred_curve = phi_t.dot(&coeff_t); // (n_grid_t,)
            predictions.push(pred_curve.to_vec());
        }

        Ok(predictions)
    }
}

/// Simple GCV-like lambda selection for scalar-on-function regression.
fn self_select_lambda_sof(x_int: &Array2<f64>, y: &Array1<f64>, n_basis: usize) -> f64 {
    let n = y.len() as f64;
    let xtx = x_int.t().dot(x_int);
    let xty = x_int.t().dot(y);
    let penalty = second_derivative_penalty(n_basis);

    let mut best_lambda = 1e-2;
    let mut best_gcv = f64::INFINITY;

    let log_lambdas: Vec<f64> = (-6..=4).map(|k| 10.0f64.powi(k)).collect();

    for &lam in &log_lambdas {
        let mut lhs = &xtx + &(&penalty * lam);
        let b = match solve_symmetric_positive(&mut lhs, &xty) {
            Ok(b) => b,
            Err(_) => continue,
        };

        let y_hat = x_int.dot(&b);
        let residuals = y - &y_hat;
        let rss: f64 = residuals.iter().map(|r| r * r).sum();

        // Approximate effective df
        let trace = compute_trace_approx(&xtx, &penalty, lam, n_basis);
        let denom = 1.0 - trace / n;
        if denom.abs() < 1e-10 {
            continue;
        }
        let gcv = (rss / n) / (denom * denom);

        if gcv < best_gcv {
            best_gcv = gcv;
            best_lambda = lam;
        }
    }

    best_lambda
}

/// Approximate trace of (X'X + lambda*P)^{-1} X'X.
fn compute_trace_approx(xtx: &Array2<f64>, penalty: &Array2<f64>, lambda: f64, n: usize) -> f64 {
    let mut lhs = xtx + &(penalty * lambda);
    let mut trace = 0.0;
    for j in 0..n {
        let rhs = xtx.column(j).to_owned();
        if let Ok(x) = solve_symmetric_positive(&mut lhs, &rhs) {
            trace += x[j];
        }
    }
    trace
}

/// Compute trapezoidal-rule grid spacing weights.
fn compute_grid_spacing(grid: &[f64]) -> Vec<f64> {
    let n = grid.len();
    if n <= 1 {
        return vec![1.0; n];
    }
    let mut dt = vec![0.0; n];
    dt[0] = (grid[1] - grid[0]) / 2.0;
    for i in 1..n - 1 {
        dt[i] = (grid[i + 1] - grid[i - 1]) / 2.0;
    }
    dt[n - 1] = (grid[n - 1] - grid[n - 2]) / 2.0;
    dt
}

/// Construct a second-derivative penalty matrix.
fn second_derivative_penalty(n_basis: usize) -> Array2<f64> {
    if n_basis < 3 {
        return Array2::<f64>::zeros((n_basis, n_basis));
    }
    let m = n_basis - 2;
    let mut d2 = Array2::<f64>::zeros((m, n_basis));
    for i in 0..m {
        d2[[i, i]] = 1.0;
        d2[[i, i + 1]] = -2.0;
        d2[[i, i + 2]] = 1.0;
    }
    d2.t().dot(&d2)
}

/// Solve symmetric positive-definite system via Cholesky.
fn solve_symmetric_positive(a: &mut Array2<f64>, b: &Array1<f64>) -> StatsResult<Array1<f64>> {
    let n = a.nrows();
    if n != a.ncols() || n != b.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "Matrix is {}x{}, vector length {}",
            a.nrows(),
            a.ncols(),
            b.len()
        )));
    }

    // Regularize
    for i in 0..n {
        a[[i, i]] += 1e-10;
    }

    let mut l = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let mut sum = 0.0;
        for k in 0..j {
            sum += l[[j, k]] * l[[j, k]];
        }
        let diag = a[[j, j]] - sum;
        if diag <= 0.0 {
            return Err(StatsError::ComputationError(
                "Matrix is not positive definite".to_string(),
            ));
        }
        l[[j, j]] = diag.sqrt();

        for i in (j + 1)..n {
            let mut sum2 = 0.0;
            for k in 0..j {
                sum2 += l[[i, k]] * l[[j, k]];
            }
            l[[i, j]] = (a[[i, j]] - sum2) / l[[j, j]];
        }
    }

    let mut z = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[[i, k]] * z[k];
        }
        z[i] = s / l[[i, i]];
    }

    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut s = z[i];
        for k in (i + 1)..n {
            s -= l[[k, i]] * x[k];
        }
        x[i] = s / l[[i, i]];
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functional::types::FunctionalData;

    #[test]
    fn test_r_squared_perfect_fit() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_hat = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let r2 = r_squared(&y, &y_hat);
        assert!((r2 - 1.0).abs() < 1e-10, "R^2 should be 1.0, got {}", r2);
    }

    #[test]
    fn test_r_squared_mean_model() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_hat = vec![3.0, 3.0, 3.0, 3.0, 3.0]; // mean prediction
        let r2 = r_squared(&y, &y_hat);
        assert!(r2.abs() < 1e-10, "R^2 should be 0.0, got {}", r2);
    }

    #[test]
    fn test_scalar_on_function_known_beta() {
        // True model: y_i = integral x_i(t) * beta(t) dt
        // where beta(t) = sin(pi * t).
        // Predictor curves must span a rich enough subspace to identify beta.
        // Use Fourier-like curves: x_i(t) = sum_k c_{ik} * sin(k*pi*t)
        let n_grid = 80;
        let grid: Vec<f64> = (0..n_grid)
            .map(|i| i as f64 / (n_grid - 1) as f64)
            .collect();
        let n_curves = 60;
        let n_harmonics = 8;

        let dt: Vec<f64> = {
            let mut d = vec![0.0; n_grid];
            d[0] = (grid[1] - grid[0]) / 2.0;
            for i in 1..n_grid - 1 {
                d[i] = (grid[i + 1] - grid[i - 1]) / 2.0;
            }
            d[n_grid - 1] = (grid[n_grid - 1] - grid[n_grid - 2]) / 2.0;
            d
        };

        let beta_true: Vec<f64> = grid
            .iter()
            .map(|&t| (std::f64::consts::PI * t).sin())
            .collect();

        let mut observations = Vec::with_capacity(n_curves);
        let mut y = Vec::with_capacity(n_curves);

        for i in 0..n_curves {
            // Rich predictor curve using multiple harmonics
            let curve: Vec<f64> = grid
                .iter()
                .map(|&t| {
                    let mut val = 0.0;
                    for k in 1..=n_harmonics {
                        let c = ((i * k) as f64 * 0.37 + k as f64 * 0.13).sin();
                        val += c * (k as f64 * std::f64::consts::PI * t).sin();
                    }
                    val
                })
                .collect();

            // True response: integral x(t) * beta(t) dt
            let yi: f64 = curve
                .iter()
                .zip(beta_true.iter())
                .zip(dt.iter())
                .map(|((&x, &b), &d)| x * b * d)
                .sum();
            y.push(yi);
            observations.push(curve);
        }

        let data =
            FunctionalData::new(grid.clone(), observations).expect("Data creation should succeed");
        let config = FunctionalConfig {
            basis: BasisType::BSpline {
                n_basis: 20,
                degree: 3,
            },
            smoothing_param: Some(1e-6),
            n_components: 3,
        };

        let result =
            ScalarOnFunctionRegression::fit(&data, &y, &config).expect("Fit should succeed");

        // R-squared should be close to 1.0 (clean data)
        assert!(
            result.r_squared > 0.95,
            "R^2 should be > 0.95, got {}",
            result.r_squared
        );

        // Beta should resemble sin(pi*t)
        // Check correlation between estimated and true beta
        let beta_est = result.beta.as_slice().unwrap_or(&[]);
        let mean_est = beta_est.iter().sum::<f64>() / beta_est.len() as f64;
        let mean_true = beta_true.iter().sum::<f64>() / beta_true.len() as f64;
        let cov: f64 = beta_est
            .iter()
            .zip(beta_true.iter())
            .map(|(&e, &t)| (e - mean_est) * (t - mean_true))
            .sum();
        let var_est: f64 = beta_est.iter().map(|&e| (e - mean_est).powi(2)).sum();
        let var_true: f64 = beta_true.iter().map(|&t| (t - mean_true).powi(2)).sum();
        let denom = var_est.sqrt() * var_true.sqrt();
        let corr = if denom > 1e-14 { cov / denom } else { 0.0 };
        assert!(
            corr > 0.9,
            "Correlation between true and estimated beta should be > 0.9, got {}",
            corr
        );
    }

    #[test]
    fn test_scalar_on_function_prediction() {
        let n_grid = 60;
        let grid: Vec<f64> = (0..n_grid)
            .map(|i| i as f64 / (n_grid - 1) as f64)
            .collect();
        let n_train = 30;
        let n_test = 10;

        let dt: Vec<f64> = {
            let mut d = vec![0.0; n_grid];
            d[0] = (grid[1] - grid[0]) / 2.0;
            for i in 1..n_grid - 1 {
                d[i] = (grid[i + 1] - grid[i - 1]) / 2.0;
            }
            d[n_grid - 1] = (grid[n_grid - 1] - grid[n_grid - 2]) / 2.0;
            d
        };

        let beta_true: Vec<f64> = grid.iter().map(|&t| t * (1.0 - t)).collect();

        // Generate training data
        let mut train_obs = Vec::with_capacity(n_train);
        let mut y_train = Vec::with_capacity(n_train);
        for i in 0..n_train {
            let a = (i as f64 * 0.5).sin();
            let curve: Vec<f64> = grid.iter().map(|&t| a * (2.0 * t - 1.0)).collect();
            let yi: f64 = curve
                .iter()
                .zip(beta_true.iter())
                .zip(dt.iter())
                .map(|((&x, &b), &d)| x * b * d)
                .sum();
            y_train.push(yi);
            train_obs.push(curve);
        }

        // Generate test data
        let mut test_obs = Vec::with_capacity(n_test);
        let mut y_test = Vec::with_capacity(n_test);
        for i in 0..n_test {
            let a = ((n_train + i) as f64 * 0.5).sin();
            let curve: Vec<f64> = grid.iter().map(|&t| a * (2.0 * t - 1.0)).collect();
            let yi: f64 = curve
                .iter()
                .zip(beta_true.iter())
                .zip(dt.iter())
                .map(|((&x, &b), &d)| x * b * d)
                .sum();
            y_test.push(yi);
            test_obs.push(curve);
        }

        let train_data = FunctionalData::new(grid.clone(), train_obs)
            .expect("Train data creation should succeed");
        let test_data =
            FunctionalData::new(grid.clone(), test_obs).expect("Test data creation should succeed");

        let config = FunctionalConfig {
            basis: BasisType::BSpline {
                n_basis: 15,
                degree: 3,
            },
            smoothing_param: Some(1e-5),
            n_components: 3,
        };

        let result = ScalarOnFunctionRegression::fit(&train_data, &y_train, &config)
            .expect("Fit should succeed");

        let predictions = ScalarOnFunctionRegression::predict(&result, &test_data)
            .expect("Predict should succeed");

        assert_eq!(predictions.len(), n_test);

        let r2_test = r_squared(&y_test, &predictions);
        assert!(r2_test > 0.9, "Test R^2 should be > 0.9, got {}", r2_test);
    }

    #[test]
    fn test_function_on_function_basic() {
        let n_grid = 40;
        let grid: Vec<f64> = (0..n_grid)
            .map(|i| i as f64 / (n_grid - 1) as f64)
            .collect();
        let n_curves = 25;

        // Simple model: y_i(t) = c_i * t where c_i depends on x_i
        let mut pred_obs = Vec::with_capacity(n_curves);
        let mut resp_obs = Vec::with_capacity(n_curves);

        for i in 0..n_curves {
            let a = (i as f64 * 0.4).sin();
            let pred_curve: Vec<f64> = grid.iter().map(|&s| a * s).collect();
            // Response is proportional to the predictor's "energy"
            let energy: f64 = pred_curve.iter().map(|x| x * x).sum::<f64>() / n_grid as f64;
            let resp_curve: Vec<f64> = grid.iter().map(|&t| energy * t).collect();
            pred_obs.push(pred_curve);
            resp_obs.push(resp_curve);
        }

        let pred_data = FunctionalData::new(grid.clone(), pred_obs)
            .expect("Predictor data creation should succeed");
        let resp_data = FunctionalData::new(grid.clone(), resp_obs)
            .expect("Response data creation should succeed");

        let config = FunctionalConfig {
            basis: BasisType::BSpline {
                n_basis: 10,
                degree: 3,
            },
            smoothing_param: Some(1e-4),
            n_components: 3,
        };

        let result = FunctionOnFunctionRegression::fit(&pred_data, &resp_data, &config, &config)
            .expect("FoF fit should succeed");

        // Check that beta_surface has correct dimensions
        assert_eq!(result.beta_surface.nrows(), n_grid);
        assert_eq!(result.beta_surface.ncols(), n_grid);

        // Fitted curves should have correct shape
        assert_eq!(result.fitted_curves.nrows(), n_curves);
        assert_eq!(result.fitted_curves.ncols(), n_grid);
    }

    #[test]
    fn test_function_on_function_prediction() {
        let n_grid = 30;
        let grid: Vec<f64> = (0..n_grid)
            .map(|i| i as f64 / (n_grid - 1) as f64)
            .collect();
        let n_train = 20;
        let n_test = 5;

        let mut train_pred = Vec::with_capacity(n_train);
        let mut train_resp = Vec::with_capacity(n_train);
        for i in 0..n_train {
            let a = (i as f64 * 0.3).sin();
            let pred: Vec<f64> = grid.iter().map(|&s| a * s).collect();
            let resp: Vec<f64> = grid.iter().map(|&t| a * t * 0.5).collect();
            train_pred.push(pred);
            train_resp.push(resp);
        }

        let mut test_pred = Vec::with_capacity(n_test);
        for i in 0..n_test {
            let a = ((n_train + i) as f64 * 0.3).sin();
            let pred: Vec<f64> = grid.iter().map(|&s| a * s).collect();
            test_pred.push(pred);
        }

        let train_pred_data = FunctionalData::new(grid.clone(), train_pred)
            .expect("Train pred creation should succeed");
        let train_resp_data = FunctionalData::new(grid.clone(), train_resp)
            .expect("Train resp creation should succeed");
        let test_pred_data = FunctionalData::new(grid.clone(), test_pred)
            .expect("Test pred creation should succeed");

        let config = FunctionalConfig {
            basis: BasisType::BSpline {
                n_basis: 8,
                degree: 3,
            },
            smoothing_param: Some(1e-3),
            n_components: 3,
        };

        let result =
            FunctionOnFunctionRegression::fit(&train_pred_data, &train_resp_data, &config, &config)
                .expect("FoF fit should succeed");

        let predictions = FunctionOnFunctionRegression::predict(&result, &test_pred_data)
            .expect("FoF predict should succeed");

        assert_eq!(predictions.len(), n_test);
        assert_eq!(predictions[0].len(), n_grid);
    }

    #[test]
    fn test_r_squared_close_to_one_clean_data() {
        // With clean, noiseless data, R^2 should be very close to 1
        let n_grid = 50;
        let grid: Vec<f64> = (0..n_grid)
            .map(|i| i as f64 / (n_grid - 1) as f64)
            .collect();
        let n_curves = 30;

        let dt: Vec<f64> = {
            let mut d = vec![0.0; n_grid];
            d[0] = (grid[1] - grid[0]) / 2.0;
            for i in 1..n_grid - 1 {
                d[i] = (grid[i + 1] - grid[i - 1]) / 2.0;
            }
            d[n_grid - 1] = (grid[n_grid - 1] - grid[n_grid - 2]) / 2.0;
            d
        };

        // Simple linear beta
        let beta_true: Vec<f64> = grid.iter().map(|&t| t).collect();

        let mut observations = Vec::with_capacity(n_curves);
        let mut y = Vec::with_capacity(n_curves);
        for i in 0..n_curves {
            let c = i as f64 / n_curves as f64 * 4.0 - 2.0;
            let curve: Vec<f64> = grid.iter().map(|&t| c * (1.0 + t)).collect();
            let yi: f64 = curve
                .iter()
                .zip(beta_true.iter())
                .zip(dt.iter())
                .map(|((&x, &b), &d)| x * b * d)
                .sum();
            y.push(yi);
            observations.push(curve);
        }

        let data = FunctionalData::new(grid, observations).expect("Data creation should succeed");
        let config = FunctionalConfig {
            basis: BasisType::BSpline {
                n_basis: 12,
                degree: 3,
            },
            smoothing_param: Some(1e-6),
            n_components: 3,
        };

        let result =
            ScalarOnFunctionRegression::fit(&data, &y, &config).expect("Fit should succeed");

        assert!(
            result.r_squared > 0.99,
            "R^2 should be > 0.99 on clean data, got {}",
            result.r_squared
        );
    }
}
