//! Concrete ML model predictors for the ML Pipeline API.
//!
//! This module provides pure-Rust implementations of classic ML algorithms:
//!
//! - [`KnnRegressor`] — k-Nearest Neighbours regression
//! - [`KnnClassifier`] — k-Nearest Neighbours classification
//! - [`LinearRegressor`] — Ordinary Least Squares via normal equations
//! - [`LogisticRegressor`] — Binary logistic regression via gradient descent
//!
//! All implementations follow the ML Pipeline trait contract: no panics,
//! no external ML libraries, full error propagation via [`PipelineError`].

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};

use super::builder::PipelineError;

// ─────────────────────────────────────────────────────────────────────────────
// DistanceMetric
// ─────────────────────────────────────────────────────────────────────────────

/// Distance metric for k-nearest neighbours algorithms.
#[derive(Debug, Clone, PartialEq)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance.
    Euclidean,
    /// Manhattan (L1) distance.
    Manhattan,
    /// Chebyshev (L∞) distance.
    Chebyshev,
    /// Cosine distance: `1 - cos(a, b)`.
    Cosine,
}

impl DistanceMetric {
    /// Compute the distance between two row vectors.
    pub fn compute(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        match self {
            DistanceMetric::Euclidean => {
                a.iter()
                    .zip(b.iter())
                    .map(|(&ai, &bi)| (ai - bi).powi(2))
                    .sum::<f64>()
                    .sqrt()
            }
            DistanceMetric::Manhattan => {
                a.iter()
                    .zip(b.iter())
                    .map(|(&ai, &bi)| (ai - bi).abs())
                    .sum::<f64>()
            }
            DistanceMetric::Chebyshev => a
                .iter()
                .zip(b.iter())
                .map(|(&ai, &bi)| (ai - bi).abs())
                .fold(0.0_f64, f64::max),
            DistanceMetric::Cosine => {
                let dot: f64 = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum();
                let norm_a: f64 = a.iter().map(|&ai| ai * ai).sum::<f64>().sqrt();
                let norm_b: f64 = b.iter().map(|&bi| bi * bi).sum::<f64>().sqrt();
                if norm_a < f64::EPSILON || norm_b < f64::EPSILON {
                    1.0
                } else {
                    1.0 - dot / (norm_a * norm_b)
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KnnRegressor
// ─────────────────────────────────────────────────────────────────────────────

/// k-Nearest Neighbours regressor.
///
/// Predicts by taking the **mean** of the `k` nearest training samples.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "ml_pipeline")]
/// # {
/// use scirs2_core::ml_pipeline::predictor::{KnnRegressor, DistanceMetric};
/// use ndarray::{Array1, Array2};
///
/// let x_train = Array2::from_shape_vec((4, 1), vec![1.0f64, 2.0, 3.0, 4.0]).expect("ok");
/// let y_train = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
///
/// let mut knn = KnnRegressor::new(1);
/// knn.fit(x_train.view(), y_train.view()).expect("fit ok");
///
/// let x_test = Array2::from_shape_vec((1, 1), vec![2.1f64]).expect("ok");
/// let preds = knn.predict(x_test.view()).expect("predict ok");
/// assert!((preds[0] - 2.0).abs() < 1.0, "nearest to 2.1 is 2.0");
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct KnnRegressor {
    k: usize,
    metric: DistanceMetric,
    x_train: Option<Array2<f64>>,
    y_train: Option<Array1<f64>>,
}

impl KnnRegressor {
    /// Create a `KnnRegressor` with Euclidean distance.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            metric: DistanceMetric::Euclidean,
            x_train: None,
            y_train: None,
        }
    }

    /// Create a `KnnRegressor` with a specified distance metric.
    pub fn with_metric(k: usize, metric: DistanceMetric) -> Self {
        Self {
            k,
            metric,
            x_train: None,
            y_train: None,
        }
    }

    /// Whether the model has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.x_train.is_some()
    }

    /// Fit the model: store training data.
    pub fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<(), PipelineError> {
        if x.nrows() == 0 {
            return Err(PipelineError::EmptyInput("KnnRegressor.fit".to_string()));
        }
        if x.nrows() != y.len() {
            return Err(PipelineError::StepError {
                step: "KnnRegressor".to_string(),
                message: format!(
                    "x has {} rows but y has {} elements",
                    x.nrows(),
                    y.len()
                ),
            });
        }
        if self.k == 0 || self.k > x.nrows() {
            return Err(PipelineError::StepError {
                step: "KnnRegressor".to_string(),
                message: format!(
                    "k={} is invalid for {} training samples",
                    self.k,
                    x.nrows()
                ),
            });
        }
        self.x_train = Some(x.to_owned());
        self.y_train = Some(y.to_owned());
        Ok(())
    }

    /// Predict for test matrix.
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, PipelineError> {
        let x_train = self.x_train.as_ref().ok_or_else(|| {
            PipelineError::NotFitted("KnnRegressor".to_string())
        })?;
        let y_train = self.y_train.as_ref().ok_or_else(|| {
            PipelineError::NotFitted("KnnRegressor".to_string())
        })?;

        if x.nrows() == 0 {
            return Err(PipelineError::EmptyInput("KnnRegressor.predict".to_string()));
        }
        if x.ncols() != x_train.ncols() {
            return Err(PipelineError::FeatureCountMismatch {
                step: "KnnRegressor".to_string(),
                expected: x_train.ncols(),
                actual: x.ncols(),
            });
        }

        let n_test = x.nrows();
        let n_train = x_train.nrows();
        let mut predictions = Array1::<f64>::zeros(n_test);

        for i in 0..n_test {
            let query = x.row(i);
            // Compute distances to all training points
            let mut dists: Vec<(f64, usize)> = (0..n_train)
                .map(|j| {
                    let d = self.metric.compute(query, x_train.row(j));
                    (d, j)
                })
                .collect();

            // Partial sort: find the k smallest distances
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let mean = dists[..self.k]
                .iter()
                .map(|&(_, idx)| y_train[idx])
                .sum::<f64>()
                / self.k as f64;

            predictions[i] = mean;
        }
        Ok(predictions)
    }

    /// Model name.
    pub fn model_name(&self) -> &str {
        "KnnRegressor"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KnnClassifier
// ─────────────────────────────────────────────────────────────────────────────

/// k-Nearest Neighbours classifier.
///
/// Predicts by **majority vote** among the `k` nearest training samples.
/// Class labels are `usize` values starting from 0.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "ml_pipeline")]
/// # {
/// use scirs2_core::ml_pipeline::predictor::KnnClassifier;
/// use ndarray::{Array1, Array2};
///
/// let x_train = Array2::from_shape_vec((4, 2),
///     vec![0.0f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).expect("ok");
/// let y_train = Array1::from_vec(vec![0usize, 0, 1, 1]);
///
/// let mut knn = KnnClassifier::new(1);
/// knn.fit(x_train.view(), y_train.view()).expect("fit ok");
///
/// let x_test = Array2::from_shape_vec((1, 2), vec![0.1f64, 0.1]).expect("ok");
/// let preds = knn.predict(x_test.view()).expect("predict ok");
/// assert_eq!(preds[0], 0usize);
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct KnnClassifier {
    k: usize,
    metric: DistanceMetric,
    x_train: Option<Array2<f64>>,
    y_train: Option<Array1<usize>>,
}

impl KnnClassifier {
    /// Create a `KnnClassifier` with Euclidean distance.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            metric: DistanceMetric::Euclidean,
            x_train: None,
            y_train: None,
        }
    }

    /// Create a `KnnClassifier` with a specified distance metric.
    pub fn with_metric(k: usize, metric: DistanceMetric) -> Self {
        Self {
            k,
            metric,
            x_train: None,
            y_train: None,
        }
    }

    /// Whether the model has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.x_train.is_some()
    }

    /// Fit the model: store training data.
    pub fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<usize>) -> Result<(), PipelineError> {
        if x.nrows() == 0 {
            return Err(PipelineError::EmptyInput("KnnClassifier.fit".to_string()));
        }
        if x.nrows() != y.len() {
            return Err(PipelineError::StepError {
                step: "KnnClassifier".to_string(),
                message: format!(
                    "x has {} rows but y has {} elements",
                    x.nrows(),
                    y.len()
                ),
            });
        }
        if self.k == 0 || self.k > x.nrows() {
            return Err(PipelineError::StepError {
                step: "KnnClassifier".to_string(),
                message: format!(
                    "k={} is invalid for {} training samples",
                    self.k,
                    x.nrows()
                ),
            });
        }
        self.x_train = Some(x.to_owned());
        self.y_train = Some(y.to_owned());
        Ok(())
    }

    /// Predict class labels for test matrix.
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<usize>, PipelineError> {
        let x_train = self.x_train.as_ref().ok_or_else(|| {
            PipelineError::NotFitted("KnnClassifier".to_string())
        })?;
        let y_train = self.y_train.as_ref().ok_or_else(|| {
            PipelineError::NotFitted("KnnClassifier".to_string())
        })?;

        if x.nrows() == 0 {
            return Err(PipelineError::EmptyInput("KnnClassifier.predict".to_string()));
        }
        if x.ncols() != x_train.ncols() {
            return Err(PipelineError::FeatureCountMismatch {
                step: "KnnClassifier".to_string(),
                expected: x_train.ncols(),
                actual: x.ncols(),
            });
        }

        let n_test = x.nrows();
        let n_train = x_train.nrows();
        let mut predictions = Array1::<usize>::zeros(n_test);

        for i in 0..n_test {
            let query = x.row(i);
            let mut dists: Vec<(f64, usize)> = (0..n_train)
                .map(|j| (self.metric.compute(query, x_train.row(j)), j))
                .collect();

            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Majority vote among k nearest
            let mut votes: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::new();
            for &(_, idx) in dists[..self.k].iter() {
                *votes.entry(y_train[idx]).or_insert(0) += 1;
            }
            let winner = votes
                .into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(label, _)| label)
                .unwrap_or(0);

            predictions[i] = winner;
        }
        Ok(predictions)
    }

    /// Model name.
    pub fn model_name(&self) -> &str {
        "KnnClassifier"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LinearRegressor
// ─────────────────────────────────────────────────────────────────────────────

/// Ordinary Least Squares linear regression.
///
/// Solves the normal equations `(XᵀX)β = Xᵀy` via Cholesky decomposition
/// with a small ridge regularisation (λ = 1e-10) for numerical stability.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "ml_pipeline")]
/// # {
/// use scirs2_core::ml_pipeline::predictor::LinearRegressor;
/// use ndarray::{Array1, Array2};
///
/// // y = 2*x + 1
/// let x = Array2::from_shape_vec((5, 1), vec![1.0f64, 2.0, 3.0, 4.0, 5.0]).expect("ok");
/// let y = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0]);
///
/// let mut lr = LinearRegressor::new();
/// lr.fit(x.view(), y.view()).expect("fit ok");
///
/// let x_test = Array2::from_shape_vec((1, 1), vec![6.0f64]).expect("ok");
/// let preds = lr.predict(x_test.view()).expect("predict ok");
/// assert!((preds[0] - 13.0).abs() < 1e-6);
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct LinearRegressor {
    coefficients: Option<Array1<f64>>,
    intercept: f64,
    fit_intercept: bool,
}

impl Default for LinearRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearRegressor {
    /// Create a `LinearRegressor` that fits an intercept term.
    pub fn new() -> Self {
        Self {
            coefficients: None,
            intercept: 0.0,
            fit_intercept: true,
        }
    }

    /// Create a `LinearRegressor` without an intercept term.
    pub fn no_intercept() -> Self {
        Self {
            coefficients: None,
            intercept: 0.0,
            fit_intercept: false,
        }
    }

    /// Return the fitted coefficients (one per feature), or `None` if not fitted.
    pub fn coefficients(&self) -> Option<&Array1<f64>> {
        self.coefficients.as_ref()
    }

    /// Return the fitted intercept term (0 if `fit_intercept=false`).
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Whether the model has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.coefficients.is_some()
    }

    /// Fit via normal equations: β = (XᵀX + λI)⁻¹ Xᵀy
    ///
    /// Uses direct Cholesky-free Gaussian elimination (LU decomposition) on the
    /// (d x d) system for robustness.
    pub fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<(), PipelineError> {
        let n = x.nrows();
        let d = x.ncols();
        if n == 0 || d == 0 {
            return Err(PipelineError::EmptyInput("LinearRegressor".to_string()));
        }
        if n != y.len() {
            return Err(PipelineError::StepError {
                step: "LinearRegressor".to_string(),
                message: format!("x has {} rows but y has {} elements", n, y.len()),
            });
        }

        // Build augmented design matrix (with bias column if requested)
        let (a_mat, b_vec) = if self.fit_intercept {
            // Augment X with a column of ones: shape (n, d+1)
            let mut a = Array2::<f64>::ones((n, d + 1));
            for i in 0..n {
                for j in 0..d {
                    a[[i, j]] = x[[i, j]];
                }
            }
            (a, y.to_owned())
        } else {
            (x.to_owned(), y.to_owned())
        };

        let aug_d = a_mat.ncols();

        // Normal equations: (AᵀA + λI) β = Aᵀb
        let lambda = 1e-10_f64;
        let mut xxt = Array2::<f64>::zeros((aug_d, aug_d));
        let mut xty = Array1::<f64>::zeros(aug_d);

        for i in 0..n {
            for j in 0..aug_d {
                for k in 0..aug_d {
                    xxt[[j, k]] += a_mat[[i, j]] * a_mat[[i, k]];
                }
                xty[j] += a_mat[[i, j]] * b_vec[i];
            }
        }
        // Ridge regularisation
        for j in 0..aug_d {
            xxt[[j, j]] += lambda;
        }

        // Solve (aug_d x aug_d) linear system via Gaussian elimination
        let beta = solve_linear_system(&xxt, &xty).map_err(|e| PipelineError::StepError {
            step: "LinearRegressor".to_string(),
            message: e,
        })?;

        if self.fit_intercept {
            self.intercept = beta[d];
            self.coefficients = Some(beta.slice(ndarray::s![..d]).to_owned());
        } else {
            self.intercept = 0.0;
            self.coefficients = Some(beta);
        }
        Ok(())
    }

    /// Predict target values for test matrix.
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, PipelineError> {
        let coef = self.coefficients.as_ref().ok_or_else(|| {
            PipelineError::NotFitted("LinearRegressor".to_string())
        })?;
        if x.nrows() == 0 {
            return Err(PipelineError::EmptyInput("LinearRegressor.predict".to_string()));
        }
        if x.ncols() != coef.len() {
            return Err(PipelineError::FeatureCountMismatch {
                step: "LinearRegressor".to_string(),
                expected: coef.len(),
                actual: x.ncols(),
            });
        }
        let n = x.nrows();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut pred = self.intercept;
            for j in 0..coef.len() {
                pred += coef[j] * x[[i, j]];
            }
            out[i] = pred;
        }
        Ok(out)
    }

    /// Model name.
    pub fn model_name(&self) -> &str {
        "LinearRegressor"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LogisticRegressor
// ─────────────────────────────────────────────────────────────────────────────

/// Binary logistic regression via mini-batch gradient descent.
///
/// Minimises the binary cross-entropy loss using full-batch gradient descent
/// with configurable learning rate, max iterations, and convergence tolerance.
/// Class labels must be 0 or 1.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "ml_pipeline")]
/// # {
/// use scirs2_core::ml_pipeline::predictor::LogisticRegressor;
/// use ndarray::{Array1, Array2};
///
/// // Simple linearly separable data
/// let x = Array2::from_shape_vec((4, 1), vec![-2.0f64, -1.0, 1.0, 2.0]).expect("ok");
/// let y = Array1::from_vec(vec![0.0f64, 0.0, 1.0, 1.0]);
///
/// let mut lr = LogisticRegressor::new();
/// lr.fit(x.view(), y.view()).expect("fit ok");
///
/// let x_test = Array2::from_shape_vec((2, 1), vec![-1.5f64, 1.5]).expect("ok");
/// let preds = lr.predict(x_test.view()).expect("predict ok");
/// assert_eq!(preds[0], 0usize);
/// assert_eq!(preds[1], 1usize);
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct LogisticRegressor {
    coefficients: Option<Array1<f64>>,
    intercept: f64,
    learning_rate: f64,
    max_iter: usize,
    tol: f64,
}

impl Default for LogisticRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl LogisticRegressor {
    /// Create a `LogisticRegressor` with default hyperparameters.
    pub fn new() -> Self {
        Self {
            coefficients: None,
            intercept: 0.0,
            learning_rate: 0.1,
            max_iter: 1000,
            tol: 1e-6,
        }
    }

    /// Create a `LogisticRegressor` with custom learning rate and max iterations.
    pub fn with_config(learning_rate: f64, max_iter: usize) -> Self {
        Self {
            coefficients: None,
            intercept: 0.0,
            learning_rate,
            max_iter,
            tol: 1e-6,
        }
    }

    /// Set convergence tolerance.
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Whether the model has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.coefficients.is_some()
    }

    /// Compute the sigmoid function: σ(z) = 1 / (1 + e^{-z})
    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    /// Predict probabilities P(y=1|x) for test matrix.
    pub fn predict_proba(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, PipelineError> {
        let coef = self.coefficients.as_ref().ok_or_else(|| {
            PipelineError::NotFitted("LogisticRegressor".to_string())
        })?;
        if x.nrows() == 0 {
            return Err(PipelineError::EmptyInput(
                "LogisticRegressor.predict_proba".to_string(),
            ));
        }
        if x.ncols() != coef.len() {
            return Err(PipelineError::FeatureCountMismatch {
                step: "LogisticRegressor".to_string(),
                expected: coef.len(),
                actual: x.ncols(),
            });
        }
        let n = x.nrows();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let z = self.intercept
                + coef
                    .iter()
                    .zip(x.row(i).iter())
                    .map(|(&c, &xi)| c * xi)
                    .sum::<f64>();
            out[i] = Self::sigmoid(z);
        }
        Ok(out)
    }

    /// Fit via full-batch gradient descent on binary cross-entropy.
    pub fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<(), PipelineError> {
        let n = x.nrows();
        let d = x.ncols();
        if n == 0 || d == 0 {
            return Err(PipelineError::EmptyInput("LogisticRegressor".to_string()));
        }
        if n != y.len() {
            return Err(PipelineError::StepError {
                step: "LogisticRegressor".to_string(),
                message: format!("x has {} rows but y has {} elements", n, y.len()),
            });
        }
        // Validate labels: must be 0.0 or 1.0
        for &yi in y.iter() {
            if (yi - 0.0).abs() > 1e-10 && (yi - 1.0).abs() > 1e-10 {
                return Err(PipelineError::StepError {
                    step: "LogisticRegressor".to_string(),
                    message: format!("labels must be 0 or 1, got {yi}"),
                });
            }
        }

        let mut coef = Array1::<f64>::zeros(d);
        let mut intercept = 0.0_f64;
        let inv_n = 1.0 / n as f64;

        for _iter in 0..self.max_iter {
            // Compute predictions
            let mut errors = Array1::<f64>::zeros(n);
            for i in 0..n {
                let z = intercept
                    + coef
                        .iter()
                        .zip(x.row(i).iter())
                        .map(|(&c, &xi)| c * xi)
                        .sum::<f64>();
                errors[i] = Self::sigmoid(z) - y[i];
            }

            // Compute gradients
            let mut grad = Array1::<f64>::zeros(d);
            let mut grad_intercept = 0.0_f64;
            for i in 0..n {
                let err = errors[i];
                for j in 0..d {
                    grad[j] += err * x[[i, j]];
                }
                grad_intercept += err;
            }
            grad.mapv_inplace(|v| v * inv_n);
            grad_intercept *= inv_n;

            // Update parameters
            let grad_norm: f64 = grad.iter().map(|&v| v * v).sum::<f64>().sqrt()
                + grad_intercept.abs();

            for j in 0..d {
                coef[j] -= self.learning_rate * grad[j];
            }
            intercept -= self.learning_rate * grad_intercept;

            if grad_norm < self.tol {
                break;
            }
        }

        self.coefficients = Some(coef);
        self.intercept = intercept;
        Ok(())
    }

    /// Predict class labels (0 or 1) for test matrix.
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<usize>, PipelineError> {
        let proba = self.predict_proba(x)?;
        let labels: Vec<usize> = proba.iter().map(|&p| if p >= 0.5 { 1 } else { 0 }).collect();
        Ok(Array1::from_vec(labels))
    }

    /// Model name.
    pub fn model_name(&self) -> &str {
        "LogisticRegressor"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear algebra helper: Gaussian elimination
// ─────────────────────────────────────────────────────────────────────────────

/// Solve `Ax = b` for `x` using Gaussian elimination with partial pivoting.
///
/// Returns an error string if the system is singular.
fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, String> {
    let n = a.nrows();
    debug_assert_eq!(n, a.ncols());
    debug_assert_eq!(n, b.len());

    // Build augmented matrix [A | b]
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row: Vec<f64> = (0..n).map(|j| a[[i, j]]).collect();
            row.push(b[i]);
            row
        })
        .collect();

    for col in 0..n {
        // Find pivot
        let pivot_row = (col..n)
            .max_by(|&r1, &r2| {
                aug[r1][col]
                    .abs()
                    .partial_cmp(&aug[r2][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| "empty matrix".to_string())?;

        aug.swap(col, pivot_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-15 {
            return Err(format!("singular matrix at column {col}"));
        }

        // Eliminate below
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for k in col..=n {
                let val = aug[col][k];
                aug[row][k] -= factor * val;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }
    Ok(Array1::from_vec(x))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    // ── KnnRegressor ──────────────────────────────────────────────────────────

    #[test]
    fn test_knn_regressor_k1_exact() {
        // k=1: predict = nearest training point
        let x_train = Array2::from_shape_vec((3, 1), vec![1.0f64, 2.0, 3.0]).expect("ok");
        let y_train = Array1::from_vec(vec![10.0, 20.0, 30.0]);
        let mut knn = KnnRegressor::new(1);
        knn.fit(x_train.view(), y_train.view()).expect("fit ok");

        let x_test = Array2::from_shape_vec((1, 1), vec![1.9f64]).expect("ok");
        let preds = knn.predict(x_test.view()).expect("predict ok");
        assert!((preds[0] - 20.0).abs() < 1e-10, "nearest to 1.9 is 2.0 (y=20)");
    }

    #[test]
    fn test_knn_regressor_k3_mean() {
        let x_train = Array2::from_shape_vec((3, 1), vec![0.0f64, 1.0, 2.0]).expect("ok");
        let y_train = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let mut knn = KnnRegressor::new(3);
        knn.fit(x_train.view(), y_train.view()).expect("fit ok");

        let x_test = Array2::from_shape_vec((1, 1), vec![1.0f64]).expect("ok");
        let preds = knn.predict(x_test.view()).expect("predict ok");
        // Mean of all 3 = 1.0
        assert!((preds[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_knn_regressor_manhattan_distance() {
        let x_train = Array2::from_shape_vec((2, 2), vec![0.0f64, 0.0, 3.0, 4.0]).expect("ok");
        let y_train = Array1::from_vec(vec![1.0, 2.0]);
        let mut knn = KnnRegressor::with_metric(1, DistanceMetric::Manhattan);
        knn.fit(x_train.view(), y_train.view()).expect("fit ok");

        // Query [1,1]: manhattan dist to [0,0]=2, to [3,4]=6 → nearest is [0,0] (y=1)
        let x_test = Array2::from_shape_vec((1, 2), vec![1.0f64, 1.0]).expect("ok");
        let preds = knn.predict(x_test.view()).expect("predict ok");
        assert!((preds[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_knn_regressor_not_fitted() {
        let knn = KnnRegressor::new(1);
        let x = Array2::from_shape_vec((1, 1), vec![1.0f64]).expect("ok");
        assert!(matches!(knn.predict(x.view()), Err(PipelineError::NotFitted(_))));
    }

    #[test]
    fn test_knn_regressor_k_too_large() {
        let x_train = Array2::from_shape_vec((2, 1), vec![1.0f64, 2.0]).expect("ok");
        let y_train = Array1::from_vec(vec![1.0, 2.0]);
        let mut knn = KnnRegressor::new(5); // k > n_train
        let result = knn.fit(x_train.view(), y_train.view());
        assert!(matches!(result, Err(PipelineError::StepError { .. })));
    }

    // ── KnnClassifier ─────────────────────────────────────────────────────────

    #[test]
    fn test_knn_classifier_basic() {
        let x_train = Array2::from_shape_vec(
            (4, 2),
            vec![0.0f64, 0.0, 0.1, 0.1, 1.0, 1.0, 0.9, 0.9],
        )
        .expect("ok");
        let y_train = Array1::from_vec(vec![0usize, 0, 1, 1]);
        let mut knn = KnnClassifier::new(1);
        knn.fit(x_train.view(), y_train.view()).expect("fit ok");

        let x_test = Array2::from_shape_vec((2, 2), vec![0.05f64, 0.05, 0.95, 0.95]).expect("ok");
        let preds = knn.predict(x_test.view()).expect("predict ok");
        assert_eq!(preds[0], 0usize);
        assert_eq!(preds[1], 1usize);
    }

    #[test]
    fn test_knn_classifier_majority_vote() {
        // 3 points: two class 0, one class 1 close to query
        let x_train = Array2::from_shape_vec(
            (4, 1),
            vec![0.0f64, 0.2, 0.4, 10.0],
        )
        .expect("ok");
        let y_train = Array1::from_vec(vec![0usize, 0, 1, 1]);
        let mut knn = KnnClassifier::new(3); // k=3, 3 nearest
        knn.fit(x_train.view(), y_train.view()).expect("fit ok");

        // Query at 0.1: 3 nearest are [0,0.2,0.4] → labels [0,0,1] → vote: 0 wins
        let x_test = Array2::from_shape_vec((1, 1), vec![0.1f64]).expect("ok");
        let preds = knn.predict(x_test.view()).expect("predict ok");
        assert_eq!(preds[0], 0usize);
    }

    // ── LinearRegressor ───────────────────────────────────────────────────────

    #[test]
    fn test_linear_regressor_perfect_fit() {
        // y = 2x + 1
        let x = Array2::from_shape_vec((4, 1), vec![1.0f64, 2.0, 3.0, 4.0]).expect("ok");
        let y = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0]);
        let mut lr = LinearRegressor::new();
        lr.fit(x.view(), y.view()).expect("fit ok");

        let x_test = Array2::from_shape_vec((2, 1), vec![5.0f64, 0.0]).expect("ok");
        let preds = lr.predict(x_test.view()).expect("predict ok");
        assert!((preds[0] - 11.0).abs() < 1e-4, "y(5) ≈ 11, got {}", preds[0]);
        assert!((preds[1] - 1.0).abs() < 1e-4, "y(0) ≈ 1, got {}", preds[1]);
    }

    #[test]
    fn test_linear_regressor_no_intercept() {
        // y = 3x (no intercept)
        let x = Array2::from_shape_vec((3, 1), vec![1.0f64, 2.0, 3.0]).expect("ok");
        let y = Array1::from_vec(vec![3.0, 6.0, 9.0]);
        let mut lr = LinearRegressor::no_intercept();
        lr.fit(x.view(), y.view()).expect("fit ok");

        let x_test = Array2::from_shape_vec((1, 1), vec![4.0f64]).expect("ok");
        let preds = lr.predict(x_test.view()).expect("predict ok");
        assert!((preds[0] - 12.0).abs() < 1e-4, "y(4) ≈ 12, got {}", preds[0]);
    }

    #[test]
    fn test_linear_regressor_multivariate() {
        // y = x0 + 2*x1 + 1
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![1.0f64, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0],
        )
        .expect("ok");
        let y = Array1::from_vec(vec![4.0, 7.0, 6.0, 5.0]);
        let mut lr = LinearRegressor::new();
        lr.fit(x.view(), y.view()).expect("fit ok");

        // Predict for [1,1]: expected 4.0
        let x_test = Array2::from_shape_vec((1, 2), vec![1.0f64, 1.0]).expect("ok");
        let preds = lr.predict(x_test.view()).expect("predict ok");
        assert!((preds[0] - 4.0).abs() < 0.1, "y ≈ 4, got {}", preds[0]);
    }

    #[test]
    fn test_linear_regressor_not_fitted() {
        let lr = LinearRegressor::new();
        let x = Array2::from_shape_vec((1, 1), vec![1.0f64]).expect("ok");
        assert!(matches!(lr.predict(x.view()), Err(PipelineError::NotFitted(_))));
    }

    // ── LogisticRegressor ─────────────────────────────────────────────────────

    #[test]
    fn test_logistic_regressor_linearly_separable() {
        let x = Array2::from_shape_vec((4, 1), vec![-2.0f64, -1.0, 1.0, 2.0]).expect("ok");
        let y = Array1::from_vec(vec![0.0f64, 0.0, 1.0, 1.0]);
        let mut lr = LogisticRegressor::with_config(0.5, 2000);
        lr.fit(x.view(), y.view()).expect("fit ok");

        let x_test = Array2::from_shape_vec((2, 1), vec![-1.5f64, 1.5]).expect("ok");
        let preds = lr.predict(x_test.view()).expect("predict ok");
        assert_eq!(preds[0], 0usize);
        assert_eq!(preds[1], 1usize);
    }

    #[test]
    fn test_logistic_regressor_predict_proba_range() {
        let x = Array2::from_shape_vec((4, 1), vec![-5.0f64, -1.0, 1.0, 5.0]).expect("ok");
        let y = Array1::from_vec(vec![0.0f64, 0.0, 1.0, 1.0]);
        let mut lr = LogisticRegressor::new();
        lr.fit(x.view(), y.view()).expect("fit ok");

        let proba = lr.predict_proba(x.view()).expect("predict_proba ok");
        for &p in proba.iter() {
            assert!(p >= 0.0 && p <= 1.0, "probability out of range: {p}");
        }
    }

    #[test]
    fn test_logistic_regressor_invalid_labels() {
        let x = Array2::from_shape_vec((2, 1), vec![1.0f64, 2.0]).expect("ok");
        let y = Array1::from_vec(vec![0.0f64, 2.0]); // 2 is invalid
        let mut lr = LogisticRegressor::new();
        let result = lr.fit(x.view(), y.view());
        assert!(matches!(result, Err(PipelineError::StepError { .. })));
    }

    // ── DistanceMetric ────────────────────────────────────────────────────────

    #[test]
    fn test_distance_metric_euclidean() {
        let a = Array1::from_vec(vec![0.0f64, 0.0]);
        let b = Array1::from_vec(vec![3.0f64, 4.0]);
        let d = DistanceMetric::Euclidean.compute(a.view(), b.view());
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_metric_cosine_orthogonal() {
        let a = Array1::from_vec(vec![1.0f64, 0.0]);
        let b = Array1::from_vec(vec![0.0f64, 1.0]);
        let d = DistanceMetric::Cosine.compute(a.view(), b.view());
        // cos(90°)=0 → distance=1
        assert!((d - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_metric_chebyshev() {
        let a = Array1::from_vec(vec![1.0f64, 5.0, 3.0]);
        let b = Array1::from_vec(vec![4.0f64, 2.0, 3.0]);
        let d = DistanceMetric::Chebyshev.compute(a.view(), b.view());
        // max(|1-4|, |5-2|, |3-3|) = max(3, 3, 0) = 3
        assert!((d - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_linear_system_identity() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0f64, 0.0, 0.0, 1.0]).expect("ok");
        let b = Array1::from_vec(vec![3.0f64, 7.0]);
        let x = solve_linear_system(&a, &b).expect("solve ok");
        assert!((x[0] - 3.0).abs() < 1e-10);
        assert!((x[1] - 7.0).abs() < 1e-10);
    }
}
