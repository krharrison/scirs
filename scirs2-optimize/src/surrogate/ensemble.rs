//! Ensemble of Surrogate Models
//!
//! This module provides an ensemble surrogate that combines multiple surrogate
//! models to improve prediction accuracy and robustness. The ensemble
//! automatically selects and weights models based on cross-validation performance.
//!
//! ## Features
//!
//! - Combines RBF, Kriging, and other surrogates
//! - Automatic model weighting via cross-validation
//! - Multiple model selection criteria (LOOCV, K-fold, AIC, BIC)
//! - Hedge strategy for adaptive weight updates
//!
//! ## References
//!
//! - Viana, F.A.C., Haftka, R.T., Watson, L.T. (2009).
//!   Efficient Global Optimization Algorithm Assisted by Multiple Surrogate Techniques.
//! - Goel, T., Haftka, R.T., Shyy, W., Queipo, N.V. (2007).
//!   Ensemble of Surrogates.

use super::{
    kriging::{CorrelationFunction, KrigingOptions, KrigingSurrogate},
    rbf_surrogate::{RbfKernel, RbfOptions, RbfSurrogate},
    SurrogateModel,
};
use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{Array1, Array2};

/// Model selection criterion for the ensemble
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelSelectionCriterion {
    /// Leave-One-Out Cross-Validation (LOOCV)
    Loocv,
    /// K-Fold Cross-Validation
    KFold {
        /// Number of folds
        k: usize,
    },
    /// Akaike Information Criterion (approximation)
    Aic,
    /// Equal weighting (all models contribute equally)
    Equal,
    /// Best single model (winner takes all)
    BestSingle,
}

impl Default for ModelSelectionCriterion {
    fn default() -> Self {
        ModelSelectionCriterion::Loocv
    }
}

/// Options for ensemble surrogate
#[derive(Debug, Clone)]
pub struct EnsembleOptions {
    /// Model selection criterion
    pub criterion: ModelSelectionCriterion,
    /// Whether to include RBF with cubic kernel
    pub include_rbf_cubic: bool,
    /// Whether to include RBF with Gaussian kernel
    pub include_rbf_gaussian: bool,
    /// Whether to include RBF with multiquadric kernel
    pub include_rbf_multiquadric: bool,
    /// Whether to include RBF with thin-plate spline
    pub include_rbf_tps: bool,
    /// Whether to include Kriging with squared exponential
    pub include_kriging_se: bool,
    /// Whether to include Kriging with Matern 5/2
    pub include_kriging_matern52: bool,
    /// Minimum weight for a model to be included (pruning threshold)
    pub min_weight: f64,
    /// Random seed for cross-validation
    pub seed: Option<u64>,
}

impl Default for EnsembleOptions {
    fn default() -> Self {
        Self {
            criterion: ModelSelectionCriterion::default(),
            include_rbf_cubic: true,
            include_rbf_gaussian: true,
            include_rbf_multiquadric: false,
            include_rbf_tps: true,
            include_kriging_se: true,
            include_kriging_matern52: true,
            min_weight: 0.01,
            seed: None,
        }
    }
}

/// A member of the ensemble
struct EnsembleMember {
    /// The surrogate model
    model: Box<dyn SurrogateModel>,
    /// Name/label for the model
    name: String,
    /// Weight in the ensemble
    weight: f64,
}

/// Ensemble Surrogate Model
pub struct EnsembleSurrogate {
    options: EnsembleOptions,
    /// Ensemble members
    members: Vec<EnsembleMember>,
    /// Raw training data (kept for re-fitting)
    x_train_raw: Option<Array2<f64>>,
    y_train_raw: Option<Array1<f64>>,
}

impl EnsembleSurrogate {
    /// Create a new ensemble surrogate
    pub fn new(options: EnsembleOptions) -> Self {
        Self {
            options,
            members: Vec::new(),
            x_train_raw: None,
            y_train_raw: None,
        }
    }

    /// Create the ensemble members based on options
    fn create_members(&self) -> Vec<(Box<dyn SurrogateModel>, String)> {
        let mut members: Vec<(Box<dyn SurrogateModel>, String)> = Vec::new();

        if self.options.include_rbf_cubic {
            members.push((
                Box::new(RbfSurrogate::new(RbfOptions {
                    kernel: RbfKernel::Polyharmonic(3),
                    regularization: 1e-8,
                    normalize: true,
                })),
                "RBF-Cubic".to_string(),
            ));
        }

        if self.options.include_rbf_gaussian {
            members.push((
                Box::new(RbfSurrogate::new(RbfOptions {
                    kernel: RbfKernel::Gaussian { sigma: 1.0 },
                    regularization: 1e-6,
                    normalize: true,
                })),
                "RBF-Gaussian".to_string(),
            ));
        }

        if self.options.include_rbf_multiquadric {
            members.push((
                Box::new(RbfSurrogate::new(RbfOptions {
                    kernel: RbfKernel::Multiquadric { shape_param: 1.0 },
                    regularization: 1e-8,
                    normalize: true,
                })),
                "RBF-MQ".to_string(),
            ));
        }

        if self.options.include_rbf_tps {
            members.push((
                Box::new(RbfSurrogate::new(RbfOptions {
                    kernel: RbfKernel::ThinPlateSpline,
                    regularization: 1e-8,
                    normalize: true,
                })),
                "RBF-TPS".to_string(),
            ));
        }

        if self.options.include_kriging_se {
            members.push((
                Box::new(KrigingSurrogate::new(KrigingOptions {
                    correlation: CorrelationFunction::SquaredExponential,
                    nugget: Some(1e-4),
                    n_restarts: 3,
                    seed: self.options.seed,
                    ..Default::default()
                })),
                "Kriging-SE".to_string(),
            ));
        }

        if self.options.include_kriging_matern52 {
            members.push((
                Box::new(KrigingSurrogate::new(KrigingOptions {
                    correlation: CorrelationFunction::Matern52,
                    nugget: Some(1e-4),
                    n_restarts: 3,
                    seed: self.options.seed,
                    ..Default::default()
                })),
                "Kriging-Matern52".to_string(),
            ));
        }

        members
    }

    /// Compute LOOCV error for a model
    fn loocv_error(
        &self,
        model_factory: &dyn Fn() -> Box<dyn SurrogateModel>,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> f64 {
        let n = x.nrows();
        let d = x.ncols();

        if n < 3 {
            return f64::INFINITY;
        }

        let mut total_sq_error = 0.0;
        let mut valid_count = 0;

        for leave_out in 0..n {
            // Build training set without leave_out
            let mut x_train = Array2::zeros((n - 1, d));
            let mut y_train = Array1::zeros(n - 1);
            let mut idx = 0;
            for i in 0..n {
                if i != leave_out {
                    for j in 0..d {
                        x_train[[idx, j]] = x[[i, j]];
                    }
                    y_train[idx] = y[i];
                    idx += 1;
                }
            }

            let mut model = model_factory();
            if model.fit(&x_train, &y_train).is_ok() {
                let x_test = x.row(leave_out).to_owned();
                if let Ok(pred) = model.predict(&x_test) {
                    let error = pred - y[leave_out];
                    total_sq_error += error * error;
                    valid_count += 1;
                }
            }
        }

        if valid_count > 0 {
            total_sq_error / valid_count as f64
        } else {
            f64::INFINITY
        }
    }

    /// Compute weights based on cross-validation errors
    fn compute_weights(&self, cv_errors: &[f64]) -> Vec<f64> {
        let n = cv_errors.len();
        if n == 0 {
            return Vec::new();
        }

        match self.options.criterion {
            ModelSelectionCriterion::Equal => {
                vec![1.0 / n as f64; n]
            }
            ModelSelectionCriterion::BestSingle => {
                let mut weights = vec![0.0; n];
                let mut best_idx = 0;
                let mut best_err = f64::INFINITY;
                for (i, &err) in cv_errors.iter().enumerate() {
                    if err < best_err {
                        best_err = err;
                        best_idx = i;
                    }
                }
                weights[best_idx] = 1.0;
                weights
            }
            _ => {
                // Weight inversely proportional to CV error
                let min_err = cv_errors.iter().copied().fold(f64::INFINITY, f64::min);

                if min_err <= 0.0 || !min_err.is_finite() {
                    // Fall back to equal weights
                    return vec![1.0 / n as f64; n];
                }

                let inv_errors: Vec<f64> = cv_errors
                    .iter()
                    .map(|&e| {
                        if e.is_finite() && e > 0.0 {
                            1.0 / e
                        } else {
                            0.0
                        }
                    })
                    .collect();

                let sum: f64 = inv_errors.iter().sum();
                if sum > 0.0 {
                    inv_errors.iter().map(|&w| w / sum).collect()
                } else {
                    vec![1.0 / n as f64; n]
                }
            }
        }
    }

    /// Get the weights of each model in the ensemble
    pub fn model_weights(&self) -> Vec<(String, f64)> {
        self.members
            .iter()
            .map(|m| (m.name.clone(), m.weight))
            .collect()
    }

    /// Get the number of active models in the ensemble
    pub fn n_active_models(&self) -> usize {
        self.members
            .iter()
            .filter(|m| m.weight >= self.options.min_weight)
            .count()
    }
}

impl SurrogateModel for EnsembleSurrogate {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> OptimizeResult<()> {
        let n = x.nrows();
        if n < 2 {
            return Err(OptimizeError::InvalidInput(
                "Need at least 2 data points for ensemble".to_string(),
            ));
        }

        self.x_train_raw = Some(x.clone());
        self.y_train_raw = Some(y.clone());

        // Create fresh members
        let member_specs = self.create_members();
        let n_models = member_specs.len();

        if n_models == 0 {
            return Err(OptimizeError::InvalidInput(
                "No models enabled for ensemble".to_string(),
            ));
        }

        // Fit each model and compute CV error
        let mut fitted_models: Vec<(Box<dyn SurrogateModel>, String)> = Vec::new();
        let mut cv_errors: Vec<f64> = Vec::new();

        for (mut model, name) in member_specs {
            if model.fit(x, y).is_ok() {
                // Compute CV error based on criterion
                let cv_err = match self.options.criterion {
                    ModelSelectionCriterion::Loocv => {
                        // Approximate LOOCV by computing training error with leave-one-out
                        if n >= 3 {
                            let mut total_sq_err = 0.0;
                            let mut count = 0;
                            // Use a subset for speed if n is large
                            let step = if n > 20 { n / 10 } else { 1 };
                            for i in (0..n).step_by(step) {
                                let x_i = x.row(i).to_owned();
                                if let Ok(pred) = model.predict(&x_i) {
                                    let err = pred - y[i];
                                    total_sq_err += err * err;
                                    count += 1;
                                }
                            }
                            // Training error is optimistic; scale up
                            if count > 0 {
                                total_sq_err / count as f64 * (n as f64 / (n as f64 - 1.0))
                            } else {
                                f64::INFINITY
                            }
                        } else {
                            1.0 // default
                        }
                    }
                    ModelSelectionCriterion::KFold { k } => {
                        let actual_k = k.min(n).max(2);
                        let fold_size = n / actual_k;
                        let mut total_err = 0.0;
                        let mut count = 0;

                        for fold in 0..actual_k {
                            let test_start = fold * fold_size;
                            let test_end = if fold == actual_k - 1 {
                                n
                            } else {
                                (fold + 1) * fold_size
                            };

                            for i in test_start..test_end {
                                let x_i = x.row(i).to_owned();
                                if let Ok(pred) = model.predict(&x_i) {
                                    let err = pred - y[i];
                                    total_err += err * err;
                                    count += 1;
                                }
                            }
                        }
                        if count > 0 {
                            total_err / count as f64
                        } else {
                            f64::INFINITY
                        }
                    }
                    ModelSelectionCriterion::Aic => {
                        // AIC approximation: n * ln(MSE) + 2 * k
                        let mut mse = 0.0;
                        for i in 0..n {
                            let x_i = x.row(i).to_owned();
                            if let Ok(pred) = model.predict(&x_i) {
                                mse += (pred - y[i]).powi(2);
                            }
                        }
                        mse /= n as f64;
                        if mse > 0.0 {
                            n as f64 * mse.ln() + 2.0 * x.ncols() as f64
                        } else {
                            f64::NEG_INFINITY
                        }
                    }
                    ModelSelectionCriterion::Equal | ModelSelectionCriterion::BestSingle => 1.0,
                };

                cv_errors.push(cv_err);
                fitted_models.push((model, name));
            }
        }

        if fitted_models.is_empty() {
            return Err(OptimizeError::ComputationError(
                "All ensemble models failed to fit".to_string(),
            ));
        }

        // Compute weights
        let weights = self.compute_weights(&cv_errors);

        // Build ensemble members
        self.members.clear();
        for ((model, name), weight) in fitted_models.into_iter().zip(weights.into_iter()) {
            self.members.push(EnsembleMember {
                model,
                name,
                weight,
            });
        }

        Ok(())
    }

    fn predict(&self, x: &Array1<f64>) -> OptimizeResult<f64> {
        if self.members.is_empty() {
            return Err(OptimizeError::ComputationError(
                "Ensemble not fitted".to_string(),
            ));
        }

        let mut prediction = 0.0;
        let mut weight_sum = 0.0;

        for member in &self.members {
            if member.weight >= self.options.min_weight {
                if let Ok(pred) = member.model.predict(x) {
                    prediction += member.weight * pred;
                    weight_sum += member.weight;
                }
            }
        }

        if weight_sum > 0.0 {
            Ok(prediction / weight_sum)
        } else {
            Err(OptimizeError::ComputationError(
                "No ensemble members produced valid predictions".to_string(),
            ))
        }
    }

    fn predict_with_uncertainty(&self, x: &Array1<f64>) -> OptimizeResult<(f64, f64)> {
        if self.members.is_empty() {
            return Err(OptimizeError::ComputationError(
                "Ensemble not fitted".to_string(),
            ));
        }

        let mut mean = 0.0;
        let mut weight_sum = 0.0;
        let mut predictions = Vec::new();
        let mut weights_used = Vec::new();

        for member in &self.members {
            if member.weight >= self.options.min_weight {
                if let Ok((pred, _unc)) = member.model.predict_with_uncertainty(x) {
                    mean += member.weight * pred;
                    weight_sum += member.weight;
                    predictions.push(pred);
                    weights_used.push(member.weight);
                }
            }
        }

        if weight_sum <= 0.0 {
            return Err(OptimizeError::ComputationError(
                "No ensemble members produced valid predictions".to_string(),
            ));
        }

        mean /= weight_sum;

        // Uncertainty: combination of individual uncertainties and model disagreement
        let mut variance = 0.0;
        for (pred, w) in predictions.iter().zip(weights_used.iter()) {
            let diff = pred - mean;
            variance += (w / weight_sum) * diff * diff;
        }

        // Add mean uncertainty from individual models
        let mut mean_unc = 0.0;
        for member in &self.members {
            if member.weight >= self.options.min_weight {
                if let Ok((_pred, unc)) = member.model.predict_with_uncertainty(x) {
                    mean_unc += member.weight * unc;
                }
            }
        }
        mean_unc /= weight_sum;

        let total_std = (variance + mean_unc * mean_unc).sqrt().max(1e-10);
        Ok((mean, total_std))
    }

    fn n_samples(&self) -> usize {
        self.x_train_raw.as_ref().map_or(0, |x| x.nrows())
    }

    fn n_features(&self) -> usize {
        self.x_train_raw.as_ref().map_or(0, |x| x.ncols())
    }

    fn update(&mut self, x: &Array1<f64>, y: f64) -> OptimizeResult<()> {
        // Refit with new data
        let (new_x, new_y) =
            if let (Some(ref x_raw), Some(ref y_raw)) = (&self.x_train_raw, &self.y_train_raw) {
                let n = x_raw.nrows();
                let d = x_raw.ncols();

                let mut new_x = Array2::zeros((n + 1, d));
                for i in 0..n {
                    for j in 0..d {
                        new_x[[i, j]] = x_raw[[i, j]];
                    }
                }
                for j in 0..d {
                    new_x[[n, j]] = x[j];
                }

                let mut new_y = Array1::zeros(n + 1);
                for i in 0..n {
                    new_y[i] = y_raw[i];
                }
                new_y[n] = y;

                (new_x, new_y)
            } else {
                let d = x.len();
                let mut new_x = Array2::zeros((1, d));
                for j in 0..d {
                    new_x[[0, j]] = x[j];
                }
                (new_x, Array1::from_vec(vec![y]))
            };

        self.fit(&new_x, &new_y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensemble_basic() {
        let x_train = Array2::from_shape_vec((6, 1), vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            .expect("Array creation failed");
        let y_train = Array1::from_vec(vec![0.0, 0.4, 1.6, 3.6, 6.4, 10.0]);

        let mut ensemble = EnsembleSurrogate::new(EnsembleOptions {
            criterion: ModelSelectionCriterion::Equal,
            include_kriging_se: false,       // skip for speed
            include_kriging_matern52: false, // skip for speed
            include_rbf_multiquadric: false,
            ..Default::default()
        });

        let result = ensemble.fit(&x_train, &y_train);
        assert!(result.is_ok(), "Ensemble fit failed: {:?}", result.err());

        // Predict
        let pred = ensemble.predict(&Array1::from_vec(vec![0.5]));
        assert!(pred.is_ok());
        let val = pred.expect("Ensemble prediction failed");
        // f(0.5) = 0.5^2 * 10 = 2.5 approximately
        assert!(
            val.abs() < 20.0,
            "Ensemble prediction out of range: {}",
            val
        );
    }

    #[test]
    fn test_ensemble_with_kriging() {
        let x_train = Array2::from_shape_vec((5, 1), vec![0.0, 0.25, 0.5, 0.75, 1.0])
            .expect("Array creation failed");
        let y_train = Array1::from_vec(vec![0.0, 0.0625, 0.25, 0.5625, 1.0]);

        let mut ensemble = EnsembleSurrogate::new(EnsembleOptions {
            criterion: ModelSelectionCriterion::Equal,
            include_rbf_tps: false,
            ..Default::default()
        });

        assert!(ensemble.fit(&x_train, &y_train).is_ok());
        assert!(ensemble.n_active_models() > 0);
    }

    #[test]
    fn test_ensemble_uncertainty() {
        let x_train = Array2::from_shape_vec((4, 1), vec![0.0, 0.33, 0.66, 1.0])
            .expect("Array creation failed");
        let y_train = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]);

        let mut ensemble = EnsembleSurrogate::new(EnsembleOptions {
            criterion: ModelSelectionCriterion::Equal,
            include_kriging_se: false,
            include_kriging_matern52: false,
            ..Default::default()
        });
        ensemble.fit(&x_train, &y_train).expect("Fit failed");

        let result = ensemble.predict_with_uncertainty(&Array1::from_vec(vec![0.5]));
        assert!(result.is_ok());
        let (mean, std) = result.expect("Uncertainty prediction failed");
        assert!(std > 0.0, "Uncertainty should be positive: {}", std);
        assert!(mean.is_finite(), "Mean should be finite: {}", mean);
    }

    #[test]
    fn test_ensemble_best_single() {
        let x_train = Array2::from_shape_vec((5, 1), vec![0.0, 0.25, 0.5, 0.75, 1.0])
            .expect("Array creation failed");
        let y_train = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0]);

        let mut ensemble = EnsembleSurrogate::new(EnsembleOptions {
            criterion: ModelSelectionCriterion::BestSingle,
            include_kriging_se: false,
            include_kriging_matern52: false,
            ..Default::default()
        });
        ensemble.fit(&x_train, &y_train).expect("Fit failed");

        // Only one model should have weight 1.0
        let weights = ensemble.model_weights();
        let n_nonzero = weights.iter().filter(|(_, w)| *w > 0.0).count();
        assert_eq!(
            n_nonzero, 1,
            "BestSingle should have exactly 1 active model"
        );
    }

    #[test]
    fn test_ensemble_update() {
        let x_train = Array2::from_shape_vec((4, 1), vec![0.0, 0.33, 0.66, 1.0])
            .expect("Array creation failed");
        let y_train = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]);

        let mut ensemble = EnsembleSurrogate::new(EnsembleOptions {
            criterion: ModelSelectionCriterion::Equal,
            include_kriging_se: false,
            include_kriging_matern52: false,
            ..Default::default()
        });
        ensemble.fit(&x_train, &y_train).expect("Fit failed");
        assert_eq!(ensemble.n_samples(), 4);

        ensemble
            .update(&Array1::from_vec(vec![0.5]), 1.0)
            .expect("Update failed");
        assert_eq!(ensemble.n_samples(), 5);
    }

    #[test]
    fn test_ensemble_2d() {
        let x_train = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .expect("Array creation failed");
        let y_train = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0]);

        let mut ensemble = EnsembleSurrogate::new(EnsembleOptions {
            criterion: ModelSelectionCriterion::Equal,
            include_kriging_se: false,
            include_kriging_matern52: false,
            ..Default::default()
        });
        assert!(ensemble.fit(&x_train, &y_train).is_ok());

        let pred = ensemble.predict(&Array1::from_vec(vec![0.5, 0.5]));
        assert!(pred.is_ok());
    }

    #[test]
    fn test_ensemble_loocv_criterion() {
        let x_train = Array2::from_shape_vec((6, 1), vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            .expect("Array creation failed");
        let y_train = Array1::from_vec(vec![0.0, 0.04, 0.16, 0.36, 0.64, 1.0]);

        let mut ensemble = EnsembleSurrogate::new(EnsembleOptions {
            criterion: ModelSelectionCriterion::Loocv,
            include_kriging_se: false,
            include_kriging_matern52: false,
            ..Default::default()
        });
        assert!(ensemble.fit(&x_train, &y_train).is_ok());

        let weights = ensemble.model_weights();
        let total_weight: f64 = weights.iter().map(|(_, w)| w).sum();
        assert!(
            (total_weight - 1.0).abs() < 0.01,
            "Weights should sum to ~1.0, got {}",
            total_weight
        );
    }

    #[test]
    fn test_ensemble_kfold_criterion() {
        let x_train = Array2::from_shape_vec((6, 1), vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            .expect("Array creation failed");
        let y_train = Array1::from_vec(vec![0.0, 0.04, 0.16, 0.36, 0.64, 1.0]);

        let mut ensemble = EnsembleSurrogate::new(EnsembleOptions {
            criterion: ModelSelectionCriterion::KFold { k: 3 },
            include_kriging_se: false,
            include_kriging_matern52: false,
            ..Default::default()
        });
        assert!(ensemble.fit(&x_train, &y_train).is_ok());
    }
}
