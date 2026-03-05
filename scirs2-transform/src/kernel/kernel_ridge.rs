//! Kernel Ridge Regression
//!
//! Kernel Ridge Regression (KRR) combines Ridge Regression with the kernel trick.
//! It learns a linear function in the kernel-induced feature space that corresponds
//! to a nonlinear function in the original space.
//!
//! ## Algorithm
//!
//! The KRR solution is: alpha = (K + lambda * I)^{-1} y
//! Prediction: y_pred = K_test * alpha
//!
//! ## Features
//!
//! - Tikhonov regularized kernel regression
//! - Leave-one-out cross-validation in closed form (O(n^3) once)
//! - Multiple output support (each output trained independently)
//! - Support for all kernel types

use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_linalg::solve;

use super::kernels::{cross_gram_matrix, gram_matrix, KernelType};
use crate::error::{Result, TransformError};

/// Kernel Ridge Regression
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_transform::kernel::{KernelRidgeRegression, KernelType};
/// use scirs2_core::ndarray::{Array1, Array2};
///
/// let x = Array2::<f64>::zeros((50, 3));
/// let y = Array1::<f64>::zeros(50);
/// let mut krr = KernelRidgeRegression::new(1.0, KernelType::RBF { gamma: 0.1 });
/// krr.fit(&x, &y).expect("should succeed");
/// let predictions = krr.predict(&x).expect("should succeed");
/// ```
#[derive(Debug, Clone)]
pub struct KernelRidgeRegression {
    /// Regularization parameter (lambda)
    alpha: f64,
    /// Kernel function type
    kernel: KernelType,
    /// Dual coefficients (solution in kernel space)
    dual_coef: Option<Array2<f64>>,
    /// Training data
    training_data: Option<Array2<f64>>,
    /// Training kernel matrix (for LOO-CV)
    k_train: Option<Array2<f64>>,
    /// Number of outputs
    n_outputs: usize,
}

impl KernelRidgeRegression {
    /// Create a new KernelRidgeRegression
    ///
    /// # Arguments
    /// * `alpha` - Regularization parameter (lambda). Larger values = more regularization.
    /// * `kernel` - The kernel function to use
    pub fn new(alpha: f64, kernel: KernelType) -> Self {
        KernelRidgeRegression {
            alpha,
            kernel,
            dual_coef: None,
            training_data: None,
            k_train: None,
            n_outputs: 0,
        }
    }

    /// Set the regularization parameter
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Get the dual coefficients
    pub fn dual_coef(&self) -> Option<&Array2<f64>> {
        self.dual_coef.as_ref()
    }

    /// Get the kernel type
    pub fn kernel(&self) -> &KernelType {
        &self.kernel
    }

    /// Get the regularization parameter
    pub fn regularization(&self) -> f64 {
        self.alpha
    }

    /// Fit the model with a single output target
    ///
    /// # Arguments
    /// * `x` - Training data, shape (n_samples, n_features)
    /// * `y` - Target values, shape (n_samples,)
    pub fn fit<S1, S2>(&mut self, x: &ArrayBase<S1, Ix2>, y: &ArrayBase<S2, Ix1>) -> Result<()>
    where
        S1: Data,
        S2: Data,
        S1::Elem: Float + NumCast,
        S2::Elem: Float + NumCast,
    {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }
        if n_samples != y.len() {
            return Err(TransformError::InvalidInput(format!(
                "x has {} samples but y has {} elements",
                n_samples,
                y.len()
            )));
        }

        // Convert y to a column matrix for uniform handling
        let y_f64: Array1<f64> = y.mapv(|v| NumCast::from(v).unwrap_or(0.0));
        let mut y_mat = Array2::zeros((n_samples, 1));
        for i in 0..n_samples {
            y_mat[[i, 0]] = y_f64[i];
        }

        self.fit_multi(x, &y_mat.view())
    }

    /// Fit the model with multiple output targets
    ///
    /// # Arguments
    /// * `x` - Training data, shape (n_samples, n_features)
    /// * `y` - Target values, shape (n_samples, n_outputs)
    pub fn fit_multi<S1, S2>(
        &mut self,
        x: &ArrayBase<S1, Ix2>,
        y: &ArrayBase<S2, Ix2>,
    ) -> Result<()>
    where
        S1: Data,
        S2: Data,
        S1::Elem: Float + NumCast,
        S2::Elem: Float + NumCast,
    {
        let n_samples = x.nrows();
        let n_outputs = y.ncols();

        if n_samples == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }
        if n_samples != y.nrows() {
            return Err(TransformError::InvalidInput(format!(
                "x has {} samples but y has {} rows",
                n_samples,
                y.nrows()
            )));
        }
        if self.alpha < 0.0 {
            return Err(TransformError::InvalidInput(
                "Regularization parameter alpha must be non-negative".to_string(),
            ));
        }

        let x_f64 = x.mapv(|v| NumCast::from(v).unwrap_or(0.0));
        let y_f64: Array2<f64> = y.mapv(|v| NumCast::from(v).unwrap_or(0.0));

        // Compute kernel matrix
        let k = gram_matrix(&x_f64.view(), &self.kernel)?;

        // Add regularization: K + alpha * I
        let mut k_reg = k.clone();
        for i in 0..n_samples {
            k_reg[[i, i]] += self.alpha;
        }

        // Solve (K + alpha*I) * alpha_coef = Y for each output
        let mut dual_coef = Array2::zeros((n_samples, n_outputs));
        for out in 0..n_outputs {
            let y_col = y_f64.column(out).to_owned();
            let coef = solve(&k_reg.view(), &y_col.view(), None).map_err(|e| {
                TransformError::ComputationError(format!(
                    "Failed to solve kernel system for output {}: {}",
                    out, e
                ))
            })?;

            for i in 0..n_samples {
                dual_coef[[i, out]] = coef[i];
            }
        }

        self.dual_coef = Some(dual_coef);
        self.training_data = Some(x_f64);
        self.k_train = Some(k);
        self.n_outputs = n_outputs;

        Ok(())
    }

    /// Predict for new data (single output)
    ///
    /// # Arguments
    /// * `x` - Test data, shape (n_test, n_features)
    ///
    /// # Returns
    /// * Predictions, shape (n_test,) for single output
    pub fn predict<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array1<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let predictions = self.predict_multi(x)?;
        if self.n_outputs == 1 {
            Ok(predictions.column(0).to_owned())
        } else {
            Err(TransformError::InvalidInput(
                "Model was fitted with multiple outputs. Use predict_multi instead.".to_string(),
            ))
        }
    }

    /// Predict for new data (multiple outputs)
    ///
    /// # Arguments
    /// * `x` - Test data, shape (n_test, n_features)
    ///
    /// # Returns
    /// * Predictions, shape (n_test, n_outputs)
    pub fn predict_multi<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let dual_coef = self
            .dual_coef
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("KRR not fitted".to_string()))?;
        let training_data = self
            .training_data
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Training data not available".to_string()))?;

        let x_f64 = x.mapv(|v| NumCast::from(v).unwrap_or(0.0));

        // Compute kernel between test and training data
        let k_test = cross_gram_matrix(&x_f64.view(), &training_data.view(), &self.kernel)?;

        // Predict: y = K_test * dual_coef
        let n_test = x_f64.nrows();
        let n_train = training_data.nrows();
        let mut predictions = Array2::zeros((n_test, self.n_outputs));

        for i in 0..n_test {
            for out in 0..self.n_outputs {
                let mut pred = 0.0;
                for j in 0..n_train {
                    pred += k_test[[i, j]] * dual_coef[[j, out]];
                }
                predictions[[i, out]] = pred;
            }
        }

        Ok(predictions)
    }

    /// Leave-one-out cross-validation in closed form
    ///
    /// Computes the LOO-CV predictions and error without explicitly
    /// re-fitting the model n times. Uses the formula:
    ///
    /// LOO_residual_i = alpha_i / (K + lambda*I)^{-1}_{ii}
    ///
    /// which requires only one matrix inversion.
    ///
    /// # Returns
    /// * `(loo_predictions, loo_mse)` - LOO predictions for each sample and mean squared error
    pub fn loo_cv(&self) -> Result<(Array2<f64>, f64)> {
        let dual_coef = self
            .dual_coef
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("KRR not fitted".to_string()))?;
        let k_train = self.k_train.as_ref().ok_or_else(|| {
            TransformError::NotFitted("Training kernel not available".to_string())
        })?;

        let n = k_train.nrows();

        // Compute (K + alpha*I)^{-1}
        // We need the diagonal of the inverse
        // Solve (K + alpha*I) * X = I to get the inverse
        let mut k_reg = k_train.clone();
        for i in 0..n {
            k_reg[[i, i]] += self.alpha;
        }

        // Compute inverse column by column
        let mut k_inv_diag = Array1::zeros(n);
        for col in 0..n {
            let mut e = Array1::zeros(n);
            e[col] = 1.0;
            let inv_col = solve(&k_reg.view(), &e.view(), None).map_err(|e| {
                TransformError::ComputationError(format!(
                    "Failed to compute inverse for LOO-CV: {}",
                    e
                ))
            })?;
            k_inv_diag[col] = inv_col[col];
        }

        // LOO residual: r_i = alpha_i / (K_inv)_{ii}
        // LOO prediction: y_loo_i = y_i - r_i
        // But we need y_i = K * alpha (training predictions)
        let mut y_train = Array2::zeros((n, self.n_outputs));
        for i in 0..n {
            for out in 0..self.n_outputs {
                let mut pred = 0.0;
                for j in 0..n {
                    pred += k_train[[i, j]] * dual_coef[[j, out]];
                }
                y_train[[i, out]] = pred;
            }
        }

        let mut loo_predictions = Array2::zeros((n, self.n_outputs));
        let mut total_sq_error = 0.0;

        for i in 0..n {
            let h_ii = k_inv_diag[i];
            if h_ii.abs() < 1e-15 {
                // Degenerate case, skip
                for out in 0..self.n_outputs {
                    loo_predictions[[i, out]] = y_train[[i, out]];
                }
                continue;
            }

            for out in 0..self.n_outputs {
                let residual = dual_coef[[i, out]] / h_ii;
                loo_predictions[[i, out]] = y_train[[i, out]] - residual;
                total_sq_error += residual * residual;
            }
        }

        let loo_mse = total_sq_error / (n as f64 * self.n_outputs as f64);

        Ok((loo_predictions, loo_mse))
    }

    /// Automatic selection of the regularization parameter via LOO-CV
    ///
    /// Tries multiple alpha values and selects the one with lowest LOO-CV error.
    ///
    /// # Arguments
    /// * `x` - Training data
    /// * `y` - Target values (single output)
    /// * `alpha_values` - Candidate regularization parameters
    ///
    /// # Returns
    /// * `(best_alpha, best_mse)` - Best alpha and corresponding LOO-CV MSE
    pub fn auto_select_alpha<S1, S2>(
        x: &ArrayBase<S1, Ix2>,
        y: &ArrayBase<S2, Ix1>,
        kernel: &KernelType,
        alpha_values: &[f64],
    ) -> Result<(f64, f64)>
    where
        S1: Data,
        S2: Data,
        S1::Elem: Float + NumCast,
        S2::Elem: Float + NumCast,
    {
        if alpha_values.is_empty() {
            return Err(TransformError::InvalidInput(
                "alpha_values must not be empty".to_string(),
            ));
        }

        let mut best_alpha = alpha_values[0];
        let mut best_mse = f64::INFINITY;

        for &alpha in alpha_values {
            let mut krr = KernelRidgeRegression::new(alpha, kernel.clone());
            match krr.fit(x, y) {
                Ok(()) => {}
                Err(_) => continue,
            }

            match krr.loo_cv() {
                Ok((_, mse)) => {
                    if mse < best_mse {
                        best_mse = mse;
                        best_alpha = alpha;
                    }
                }
                Err(_) => continue,
            }
        }

        if best_mse.is_infinite() {
            return Err(TransformError::ComputationError(
                "All alpha values failed in LOO-CV".to_string(),
            ));
        }

        Ok((best_alpha, best_mse))
    }

    /// Compute the R-squared score for the training data
    ///
    /// # Arguments
    /// * `y_true` - True target values
    ///
    /// # Returns
    /// * R-squared score
    pub fn score<S>(&self, x: &ArrayBase<S, Ix2>, y_true: &Array1<f64>) -> Result<f64>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let y_pred = self.predict(x)?;

        let n = y_true.len();
        if n != y_pred.len() {
            return Err(TransformError::InvalidInput(
                "Predictions and true values have different lengths".to_string(),
            ));
        }

        let y_mean = y_true.sum() / n as f64;

        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        for i in 0..n {
            let residual = y_true[i] - y_pred[i];
            ss_res += residual * residual;
            let deviation = y_true[i] - y_mean;
            ss_tot += deviation * deviation;
        }

        if ss_tot < 1e-15 {
            // All targets are the same
            Ok(if ss_res < 1e-15 { 1.0 } else { 0.0 })
        } else {
            Ok(1.0 - ss_res / ss_tot)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    fn make_regression_data(n: usize) -> (Array2<f64>, Array1<f64>) {
        let mut x_data = Vec::with_capacity(n * 2);
        let mut y_data = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / n as f64 * 4.0;
            x_data.push(t);
            x_data.push(t * t);
            y_data.push((t * std::f64::consts::PI).sin() + 0.1 * (i as f64 * 0.1));
        }
        let x = Array::from_shape_vec((n, 2), x_data).expect("Failed");
        let y = Array::from_vec(y_data);
        (x, y)
    }

    #[test]
    fn test_krr_basic_fit_predict() {
        let (x, y) = make_regression_data(30);
        let mut krr = KernelRidgeRegression::new(1.0, KernelType::RBF { gamma: 0.5 });
        krr.fit(&x, &y).expect("KRR fit failed");

        let predictions = krr.predict(&x).expect("KRR predict failed");
        assert_eq!(predictions.len(), 30);
        for val in predictions.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_krr_linear_kernel() {
        let (x, y) = make_regression_data(20);
        let mut krr = KernelRidgeRegression::new(0.1, KernelType::Linear);
        krr.fit(&x, &y).expect("KRR fit failed");

        let predictions = krr.predict(&x).expect("KRR predict failed");
        assert_eq!(predictions.len(), 20);
        for val in predictions.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_krr_polynomial_kernel() {
        let (x, y) = make_regression_data(20);
        let kernel = KernelType::Polynomial {
            gamma: 1.0,
            coef0: 1.0,
            degree: 2,
        };
        let mut krr = KernelRidgeRegression::new(0.5, kernel);
        krr.fit(&x, &y).expect("KRR fit failed");

        let predictions = krr.predict(&x).expect("KRR predict failed");
        assert_eq!(predictions.len(), 20);
    }

    #[test]
    fn test_krr_multi_output() {
        let n = 20;
        let mut x_data = Vec::with_capacity(n * 2);
        let mut y_data = Vec::with_capacity(n * 2);
        for i in 0..n {
            let t = i as f64 / n as f64;
            x_data.push(t);
            x_data.push(t * t);
            y_data.push(t.sin());
            y_data.push(t.cos());
        }
        let x = Array::from_shape_vec((n, 2), x_data).expect("Failed");
        let y = Array::from_shape_vec((n, 2), y_data).expect("Failed");

        let mut krr = KernelRidgeRegression::new(0.1, KernelType::RBF { gamma: 1.0 });
        krr.fit_multi(&x, &y).expect("KRR multi-fit failed");

        let predictions = krr.predict_multi(&x).expect("KRR predict_multi failed");
        assert_eq!(predictions.shape(), &[n, 2]);
        for val in predictions.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_krr_loo_cv() {
        let (x, y) = make_regression_data(20);
        let mut krr = KernelRidgeRegression::new(1.0, KernelType::RBF { gamma: 0.5 });
        krr.fit(&x, &y).expect("KRR fit failed");

        let (loo_preds, loo_mse) = krr.loo_cv().expect("LOO-CV failed");
        assert_eq!(loo_preds.shape(), &[20, 1]);
        assert!(loo_mse >= 0.0);
        assert!(loo_mse.is_finite());
    }

    #[test]
    fn test_krr_auto_alpha() {
        let (x, y) = make_regression_data(20);
        let kernel = KernelType::RBF { gamma: 0.5 };
        let alphas = vec![0.001, 0.01, 0.1, 1.0, 10.0];

        let (best_alpha, best_mse) =
            KernelRidgeRegression::auto_select_alpha(&x.view(), &y.view(), &kernel, &alphas)
                .expect("Auto alpha failed");

        assert!(best_alpha > 0.0);
        assert!(best_mse >= 0.0);
        assert!(best_mse.is_finite());
    }

    #[test]
    fn test_krr_r_squared() {
        let (x, y) = make_regression_data(30);
        let mut krr = KernelRidgeRegression::new(0.1, KernelType::RBF { gamma: 1.0 });
        krr.fit(&x, &y).expect("KRR fit failed");

        let r2 = krr.score(&x, &y).expect("R2 score failed");
        // On training data with RBF kernel, R2 should be high
        assert!(r2 > 0.5, "R2 should be > 0.5 on training data, got {}", r2);
        assert!(r2 <= 1.0 + 1e-10);
    }

    #[test]
    fn test_krr_out_of_sample() {
        let (x_train, y_train) = make_regression_data(30);
        let mut krr = KernelRidgeRegression::new(0.5, KernelType::RBF { gamma: 0.5 });
        krr.fit(&x_train, &y_train).expect("KRR fit failed");

        let x_test =
            Array::from_shape_vec((3, 2), vec![0.5, 0.25, 1.0, 1.0, 2.0, 4.0]).expect("Failed");

        let predictions = krr.predict(&x_test).expect("KRR predict failed");
        assert_eq!(predictions.len(), 3);
        for val in predictions.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_krr_empty_data() {
        let x: Array2<f64> = Array2::zeros((0, 3));
        let y: Array1<f64> = Array1::zeros(0);
        let mut krr = KernelRidgeRegression::new(1.0, KernelType::Linear);
        assert!(krr.fit(&x, &y).is_err());
    }

    #[test]
    fn test_krr_mismatched_samples() {
        let x = Array::from_shape_vec((5, 2), vec![1.0; 10]).expect("Failed");
        let y = Array::from_vec(vec![1.0; 3]);
        let mut krr = KernelRidgeRegression::new(1.0, KernelType::Linear);
        assert!(krr.fit(&x, &y).is_err());
    }

    #[test]
    fn test_krr_not_fitted() {
        let krr = KernelRidgeRegression::new(1.0, KernelType::Linear);
        let x = Array::from_shape_vec((3, 2), vec![1.0; 6]).expect("Failed");
        assert!(krr.predict(&x).is_err());
    }

    #[test]
    fn test_krr_laplacian_kernel() {
        let (x, y) = make_regression_data(20);
        let mut krr = KernelRidgeRegression::new(0.5, KernelType::Laplacian { gamma: 0.5 });
        krr.fit(&x, &y).expect("KRR fit failed");

        let predictions = krr.predict(&x).expect("KRR predict failed");
        assert_eq!(predictions.len(), 20);
        for val in predictions.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_krr_high_regularization() {
        let (x, y) = make_regression_data(20);
        let mut krr = KernelRidgeRegression::new(1000.0, KernelType::RBF { gamma: 1.0 });
        krr.fit(&x, &y).expect("KRR fit failed");

        // High regularization should make predictions closer to the mean
        let predictions = krr.predict(&x).expect("KRR predict failed");
        let pred_var: f64 = {
            let mean = predictions.sum() / predictions.len() as f64;
            predictions
                .iter()
                .map(|&p| (p - mean) * (p - mean))
                .sum::<f64>()
                / predictions.len() as f64
        };
        // Variance should be small with high regularization
        assert!(
            pred_var < 1.0,
            "High regularization should reduce prediction variance, got {}",
            pred_var
        );
    }
}
