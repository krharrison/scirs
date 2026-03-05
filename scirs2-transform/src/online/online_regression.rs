//! Online regression algorithms for streaming data.
//!
//! Provides online regression algorithms that update model parameters
//! incrementally without storing the full dataset.

use crate::error::{Result, TransformError};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Online Ridge Regression via recursive least squares.
#[derive(Debug, Clone)]
pub struct OnlineRidgeRegression {
    /// Regularization parameter.
    pub lambda: f64,
    /// Current weight vector.
    weights: Option<Array1<f64>>,
    /// Precision matrix (inverse of covariance + lambda*I).
    precision: Option<Array2<f64>>,
    n_features: usize,
}

impl OnlineRidgeRegression {
    /// Create a new online ridge regression model.
    pub fn new(lambda: f64) -> Self {
        Self {
            lambda,
            weights: None,
            precision: None,
            n_features: 0,
        }
    }

    /// Update the model with a new batch of data.
    pub fn partial_fit(
        &mut self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<()> {
        let n_features = x.ncols();
        if self.weights.is_none() {
            self.n_features = n_features;
            self.weights = Some(Array1::zeros(n_features));
            let mut p = Array2::<f64>::eye(n_features);
            p.mapv_inplace(|v| v / self.lambda);
            self.precision = Some(p);
        }
        if x.ncols() != self.n_features {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        // Sherman-Morrison-Woodbury rank-1 updates
        let p = self.precision.as_mut().expect("precision initialized above");
        let w = self.weights.as_mut().expect("weights initialized above");
        for (xi, yi) in x.rows().into_iter().zip(y.iter()) {
            let p_xi = p.dot(&xi);
            let denom = 1.0 + xi.dot(&p_xi);
            let k = p_xi.mapv(|v| v / denom);
            // P <- P - k * (P x_i)'
            for i in 0..self.n_features {
                for j in 0..self.n_features {
                    p[[i, j]] -= k[i] * p_xi[j];
                }
            }
            let err = yi - w.dot(&xi);
            for i in 0..self.n_features {
                w[i] += k[i] * err;
            }
        }
        Ok(())
    }

    /// Predict targets for new data.
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let w = self.weights.as_ref().ok_or_else(|| {
            TransformError::NotFitted("OnlineRidgeRegression".to_string())
        })?;
        Ok(x.dot(w))
    }

    /// Return current weight vector.
    pub fn weights(&self) -> Option<&Array1<f64>> {
        self.weights.as_ref()
    }
}

/// Online LASSO via truncated gradient descent (Langford et al. 2009).
#[derive(Debug, Clone)]
pub struct OnlineLasso {
    /// L1 regularization strength.
    pub lambda: f64,
    /// Learning rate.
    pub eta: f64,
    weights: Option<Array1<f64>>,
    n_features: usize,
    t: usize,
}

impl OnlineLasso {
    /// Create a new online LASSO model.
    pub fn new(lambda: f64, eta: f64) -> Self {
        Self {
            lambda,
            eta,
            weights: None,
            n_features: 0,
            t: 0,
        }
    }

    /// Update with a mini-batch.
    pub fn partial_fit(
        &mut self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<()> {
        let n_features = x.ncols();
        if self.weights.is_none() {
            self.n_features = n_features;
            self.weights = Some(Array1::zeros(n_features));
        }
        if x.ncols() != self.n_features {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let w = self.weights.as_mut().expect("weights initialized");
        for (xi, yi) in x.rows().into_iter().zip(y.iter()) {
            self.t += 1;
            let pred = w.dot(&xi);
            let err = yi - pred;
            let eta_t = self.eta / (self.t as f64).sqrt();
            // Gradient step
            for (wj, xj) in w.iter_mut().zip(xi.iter()) {
                *wj += eta_t * err * xj;
            }
            // Truncated gradient (soft threshold)
            let threshold = eta_t * self.lambda;
            for wj in w.iter_mut() {
                if *wj > threshold {
                    *wj -= threshold;
                } else if *wj < -threshold {
                    *wj += threshold;
                } else {
                    *wj = 0.0;
                }
            }
        }
        Ok(())
    }

    /// Predict targets for new data.
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let w = self.weights.as_ref().ok_or_else(|| {
            TransformError::NotFitted("OnlineLasso".to_string())
        })?;
        Ok(x.dot(w))
    }
}

/// Passive-Aggressive Regressor (PA-I/PA-II).
///
/// Crammer et al. (2006): "Online Passive-Aggressive Algorithms".
#[derive(Debug, Clone)]
pub struct PassiveAggressiveRegressor {
    /// Aggressiveness parameter C.
    pub c: f64,
    /// Sensitivity margin epsilon.
    pub epsilon: f64,
    weights: Option<Array1<f64>>,
    n_features: usize,
}

impl PassiveAggressiveRegressor {
    /// Create a new PA-I regressor.
    pub fn new(c: f64, epsilon: f64) -> Self {
        Self {
            c,
            epsilon,
            weights: None,
            n_features: 0,
        }
    }

    /// Update with a mini-batch.
    pub fn partial_fit(
        &mut self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<()> {
        let n_features = x.ncols();
        if self.weights.is_none() {
            self.n_features = n_features;
            self.weights = Some(Array1::zeros(n_features));
        }
        if x.ncols() != self.n_features {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let w = self.weights.as_mut().expect("weights initialized");
        for (xi, yi) in x.rows().into_iter().zip(y.iter()) {
            let pred = w.dot(&xi);
            let loss = (yi - pred).abs() - self.epsilon;
            if loss <= 0.0 {
                continue; // passive
            }
            let xi_sq = xi.dot(&xi);
            // PA-I: tau = min(C, loss / ||x||^2)
            let tau = if xi_sq < 1e-12 {
                0.0
            } else {
                (loss / xi_sq).min(self.c)
            };
            let sign = if yi > &pred { 1.0 } else { -1.0 };
            for (wj, xj) in w.iter_mut().zip(xi.iter()) {
                *wj += tau * sign * xj;
            }
        }
        Ok(())
    }

    /// Predict targets for new data.
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let w = self.weights.as_ref().ok_or_else(|| {
            TransformError::NotFitted("PassiveAggressiveRegressor".to_string())
        })?;
        Ok(x.dot(w))
    }
}

/// Follow-The-Regularized-Leader (FTRL) Regressor with L1/L2 proximal step.
///
/// McMahan et al. (2013): "Ad Click Prediction: a View from the Trenches".
#[derive(Debug, Clone)]
pub struct FtrlRegressor {
    /// L1 coefficient.
    pub alpha: f64,
    /// L2 coefficient.
    pub beta: f64,
    /// Learning rate parameter (alpha_ftrl).
    pub eta: f64,
    weights: Option<Array1<f64>>,
    z: Option<Array1<f64>>,
    n_acc: Option<Array1<f64>>,
    n_features: usize,
    t: usize,
}

impl FtrlRegressor {
    /// Create a new FTRL-proximal regressor.
    pub fn new(alpha: f64, beta: f64, eta: f64) -> Self {
        Self {
            alpha,
            beta,
            eta,
            weights: None,
            z: None,
            n_acc: None,
            n_features: 0,
            t: 0,
        }
    }

    /// Update with a mini-batch.
    pub fn partial_fit(
        &mut self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<()> {
        let n_features = x.ncols();
        if self.weights.is_none() {
            self.n_features = n_features;
            self.weights = Some(Array1::zeros(n_features));
            self.z = Some(Array1::zeros(n_features));
            self.n_acc = Some(Array1::zeros(n_features));
        }
        if x.ncols() != self.n_features {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let w = self.weights.as_mut().expect("weights initialized");
        let z = self.z.as_mut().expect("z initialized");
        let n = self.n_acc.as_mut().expect("n_acc initialized");
        for (xi, yi) in x.rows().into_iter().zip(y.iter()) {
            self.t += 1;
            // Recompute weights from z, n
            for j in 0..self.n_features {
                let sign_z = if z[j] >= 0.0 { 1.0 } else { -1.0 };
                if (z[j] * sign_z) <= self.alpha {
                    w[j] = 0.0;
                } else {
                    w[j] = -(z[j] - sign_z * self.alpha)
                        / ((self.beta + n[j].sqrt()) / self.eta + self.beta);
                }
            }
            let pred = w.dot(&xi);
            let grad = pred - yi;
            // Update z and n
            for (j, xj) in xi.iter().enumerate() {
                let gj = grad * xj;
                let sigma = (((n[j] + gj * gj).sqrt() - n[j].sqrt()) / self.eta).max(0.0);
                z[j] += gj - sigma * w[j];
                n[j] += gj * gj;
            }
        }
        Ok(())
    }

    /// Predict targets for new data.
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let w = self.weights.as_ref().ok_or_else(|| {
            TransformError::NotFitted("FtrlRegressor".to_string())
        })?;
        Ok(x.dot(w))
    }
}
