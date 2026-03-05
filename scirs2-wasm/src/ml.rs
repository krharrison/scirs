//! Machine Learning Inference for WASM
//!
//! Provides lightweight ML models suitable for client-side inference in the
//! browser or Node.js.  The focus is on models that are:
//!
//! - **Small** – weights fit comfortably in memory
//! - **Fast** – prediction latency is dominated by WASM overhead, not flops
//! - **Portable** – no native dependencies beyond `ndarray`
//!
//! All model types support:
//! - `predict()` / `predict_batch()` – single and batched prediction
//! - JSON serialization of weights so they can be loaded from a URL
//! - `#[wasm_bindgen]` exports for direct JavaScript use
//!
//! ## Model summary
//!
//! | Type | Task | Input | Output |
//! |------|------|-------|--------|
//! | `WasmLinearModel` | Regression | Float32Array | f64 |
//! | `WasmKMeans` | Clustering | Float64Array | cluster label (u32) |
//! | `WasmNaiveBayes` | Classification | Float64Array | class label (u32) |

use crate::error::WasmError;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Helper utilities
// ---------------------------------------------------------------------------

/// Compute the squared Euclidean distance between two equal-length slices.
fn squared_euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Compute the dot product of two equal-length slices.
fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// WasmLinearModel – linear regression / ridge regression predictor
// ---------------------------------------------------------------------------

/// A linear regression predictor: `y = weights · x + bias`.
///
/// ## JavaScript usage
///
/// ```javascript
/// const model = WasmLinearModel.from_json(json_string);
/// const prediction = model.predict(new Float64Array([1.0, 2.0, 3.0]));
/// const batch = model.predict_batch(new Float64Array([1,2,3, 4,5,6]), 3);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct WasmLinearModel {
    /// Model coefficients (one per feature).
    weights: Vec<f64>,
    /// Intercept / bias term.
    bias: f64,
    /// Number of input features.
    n_features: usize,
    /// Optional feature mean (for mean-centering at inference).
    feature_mean: Option<Vec<f64>>,
    /// Optional feature scale (for standardization at inference).
    feature_scale: Option<Vec<f64>>,
}

#[wasm_bindgen]
impl WasmLinearModel {
    /// Construct from explicit weights and bias.
    ///
    /// # Errors
    ///
    /// Returns a JS error if `weights` is empty.
    #[wasm_bindgen(constructor)]
    pub fn new(weights: Vec<f64>, bias: f64) -> Result<WasmLinearModel, JsValue> {
        if weights.is_empty() {
            return Err(
                WasmError::InvalidParameter("WasmLinearModel: weights must not be empty".to_string())
                    .into(),
            );
        }
        let n_features = weights.len();
        Ok(WasmLinearModel {
            weights,
            bias,
            n_features,
            feature_mean: None,
            feature_scale: None,
        })
    }

    /// Load a model from a JSON string.
    ///
    /// The JSON schema mirrors the serialised form produced by
    /// `WasmLinearModel::to_json()`.
    ///
    /// # Errors
    ///
    /// Returns a JS error if the JSON is malformed.
    pub fn from_json(json: &str) -> Result<WasmLinearModel, JsValue> {
        serde_json::from_str(json)
            .map_err(|e| WasmError::SerializationError(e.to_string()).into())
    }

    /// Serialise the model to JSON.
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(self)
            .map_err(|e| WasmError::SerializationError(e.to_string()).into())
    }

    /// Predict for a single sample.
    ///
    /// # Errors
    ///
    /// Returns a JS error if the input length does not match `n_features`.
    pub fn predict(&self, x: &[f64]) -> Result<f64, JsValue> {
        if x.len() != self.n_features {
            return Err(WasmError::InvalidParameter(format!(
                "WasmLinearModel::predict: expected {} features, got {}",
                self.n_features,
                x.len()
            ))
            .into());
        }

        let standardized: Vec<f64> = self.standardize_sample(x);
        Ok(dot_f64(&standardized, &self.weights) + self.bias)
    }

    /// Batch prediction from a flat row-major array.
    ///
    /// # Arguments
    ///
    /// * `x`          – flat buffer of `n_samples × n_features` f64 values
    /// * `n_features` – number of features per sample
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` of length `n_samples`.
    ///
    /// # Errors
    ///
    /// Returns a JS error if `x.len()` is not a multiple of `n_features`, or
    /// `n_features` does not match the model's expected feature count.
    pub fn predict_batch(&self, x: &[f64], n_features: usize) -> Result<Vec<f64>, JsValue> {
        if n_features != self.n_features {
            return Err(WasmError::InvalidParameter(format!(
                "WasmLinearModel::predict_batch: model expects {} features, got {}",
                self.n_features, n_features
            ))
            .into());
        }
        if x.len() % n_features != 0 {
            return Err(WasmError::InvalidParameter(format!(
                "WasmLinearModel::predict_batch: x length {} is not a multiple of n_features {}",
                x.len(),
                n_features
            ))
            .into());
        }

        let n_samples = x.len() / n_features;
        let mut predictions = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let sample = &x[i * n_features..(i + 1) * n_features];
            let standardized = self.standardize_sample(sample);
            predictions.push(dot_f64(&standardized, &self.weights) + self.bias);
        }

        Ok(predictions)
    }

    /// Return the number of expected input features.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Return the weights as a `Vec<f64>`.
    pub fn weights(&self) -> Vec<f64> {
        self.weights.clone()
    }

    /// Return the bias term.
    pub fn bias(&self) -> f64 {
        self.bias
    }
}

impl WasmLinearModel {
    /// Apply stored feature normalization (if any) to a sample.
    fn standardize_sample(&self, x: &[f64]) -> Vec<f64> {
        match (&self.feature_mean, &self.feature_scale) {
            (Some(mean), Some(scale)) => x
                .iter()
                .zip(mean.iter().zip(scale.iter()))
                .map(|(&xi, (&m, &s))| if s == 0.0 { 0.0 } else { (xi - m) / s })
                .collect(),
            _ => x.to_vec(),
        }
    }
}

// ---------------------------------------------------------------------------
// WasmKMeans – k-means clustering inference
// ---------------------------------------------------------------------------

/// K-Means clustering for browser-side inference.
///
/// Stores pre-trained centroids and assigns new samples to the nearest
/// centroid using Euclidean distance.
///
/// ## JavaScript usage
///
/// ```javascript
/// const km = WasmKMeans.from_json(json_string);
/// const label = km.predict(new Float64Array([1.2, 3.4]));
/// const labels = km.predict_batch(new Float64Array([1,2, 3,4, 5,6]), 2);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct WasmKMeans {
    /// Centroids: `k × n_features` row-major flat vector.
    centroids: Vec<f64>,
    /// Number of clusters.
    k: usize,
    /// Number of features per sample.
    n_features: usize,
    /// Optional cluster labels (strings for readability).
    cluster_labels: Option<Vec<String>>,
}

#[wasm_bindgen]
impl WasmKMeans {
    /// Construct from pre-trained centroids.
    ///
    /// # Arguments
    ///
    /// * `centroids`  – row-major flat array of shape `k × n_features`
    /// * `k`          – number of clusters
    /// * `n_features` – features per sample
    ///
    /// # Errors
    ///
    /// Returns a JS error if `centroids.len() != k * n_features`.
    #[wasm_bindgen(constructor)]
    pub fn new(centroids: Vec<f64>, k: usize, n_features: usize) -> Result<WasmKMeans, JsValue> {
        if k == 0 || n_features == 0 {
            return Err(WasmError::InvalidParameter(
                "WasmKMeans: k and n_features must be > 0".to_string(),
            )
            .into());
        }
        let expected = k.checked_mul(n_features).ok_or_else(|| {
            WasmError::InvalidParameter("WasmKMeans: k × n_features overflow".to_string())
        })?;
        if centroids.len() != expected {
            return Err(WasmError::InvalidParameter(format!(
                "WasmKMeans: expected {} centroid values ({}×{}), got {}",
                expected,
                k,
                n_features,
                centroids.len()
            ))
            .into());
        }
        Ok(WasmKMeans {
            centroids,
            k,
            n_features,
            cluster_labels: None,
        })
    }

    /// Deserialise from JSON produced by `to_json`.
    pub fn from_json(json: &str) -> Result<WasmKMeans, JsValue> {
        serde_json::from_str(json)
            .map_err(|e| WasmError::SerializationError(e.to_string()).into())
    }

    /// Serialise to JSON.
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(self)
            .map_err(|e| WasmError::SerializationError(e.to_string()).into())
    }

    /// Assign a single sample to its nearest centroid.
    ///
    /// Returns the 0-based cluster index (u32 for JS compatibility).
    ///
    /// # Errors
    ///
    /// Returns a JS error if `x.len() != n_features`.
    pub fn predict(&self, x: &[f64]) -> Result<u32, JsValue> {
        if x.len() != self.n_features {
            return Err(WasmError::InvalidParameter(format!(
                "WasmKMeans::predict: expected {} features, got {}",
                self.n_features,
                x.len()
            ))
            .into());
        }

        let best = self.nearest_centroid(x).ok_or_else(|| {
            WasmError::ComputationError("WasmKMeans: no centroids available".to_string())
        })?;

        Ok(best as u32)
    }

    /// Assign a batch of samples to clusters.
    ///
    /// # Arguments
    ///
    /// * `x`          – flat row-major `Float64Array`, length `n_samples × n_features`
    /// * `n_features` – features per row (must match model)
    ///
    /// # Returns
    ///
    /// `Vec<u32>` of cluster assignments.
    pub fn predict_batch(&self, x: &[f64], n_features: usize) -> Result<Vec<u32>, JsValue> {
        if n_features != self.n_features {
            return Err(WasmError::InvalidParameter(format!(
                "WasmKMeans::predict_batch: model has {} features, got {}",
                self.n_features, n_features
            ))
            .into());
        }
        if x.len() % n_features != 0 {
            return Err(WasmError::InvalidParameter(
                "WasmKMeans::predict_batch: x.len() not divisible by n_features".to_string(),
            )
            .into());
        }

        let n_samples = x.len() / n_features;
        let mut labels = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let sample = &x[i * n_features..(i + 1) * n_features];
            let label = self.nearest_centroid(sample).unwrap_or(0);
            labels.push(label as u32);
        }

        Ok(labels)
    }

    /// Return the centroid coordinates for cluster `k_idx`.
    ///
    /// # Errors
    ///
    /// Returns a JS error if `k_idx >= k`.
    pub fn get_centroid(&self, k_idx: usize) -> Result<Vec<f64>, JsValue> {
        if k_idx >= self.k {
            return Err(WasmError::IndexOutOfBounds(format!(
                "WasmKMeans::get_centroid: index {} out of range (k={})",
                k_idx, self.k
            ))
            .into());
        }
        let start = k_idx * self.n_features;
        Ok(self.centroids[start..start + self.n_features].to_vec())
    }

    /// Return the number of clusters.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Return the feature count per sample.
    pub fn n_features(&self) -> usize {
        self.n_features
    }
}

impl WasmKMeans {
    /// Find the index of the nearest centroid to `x` (Euclidean distance).
    fn nearest_centroid(&self, x: &[f64]) -> Option<usize> {
        let mut best_idx = 0usize;
        let mut best_dist = f64::INFINITY;

        for ci in 0..self.k {
            let start = ci * self.n_features;
            let centroid = &self.centroids[start..start + self.n_features];
            let d = squared_euclidean(x, centroid);
            if d < best_dist {
                best_dist = d;
                best_idx = ci;
            }
        }

        if best_dist.is_finite() {
            Some(best_idx)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// WasmNaiveBayes – Gaussian Naïve Bayes classifier
// ---------------------------------------------------------------------------

/// Per-class statistics stored for Gaussian Naïve Bayes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianClass {
    /// Class label index.
    pub label: u32,
    /// Log prior: log(P(class)).
    pub log_prior: f64,
    /// Feature means: length = n_features.
    pub means: Vec<f64>,
    /// Feature variances: length = n_features.
    pub variances: Vec<f64>,
}

/// Gaussian Naïve Bayes classifier.
///
/// Predicts using the log-sum of Gaussian log-likelihoods per feature.
///
/// ## JavaScript usage
///
/// ```javascript
/// const nb = WasmNaiveBayes.from_json(json_string);
/// const label = nb.predict(new Float64Array([5.1, 3.5, 1.4, 0.2]));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct WasmNaiveBayes {
    /// One entry per class, in ascending label order.
    classes: Vec<GaussianClass>,
    /// Number of features.
    n_features: usize,
    /// Variance smoothing (additive Laplace-like regularization).
    var_smoothing: f64,
}

#[wasm_bindgen]
impl WasmNaiveBayes {
    /// Construct from class statistics.
    ///
    /// The `classes_json` parameter should be a JSON array of objects with the
    /// shape defined by `GaussianClass`:
    ///
    /// ```json
    /// [
    ///   { "label": 0, "log_prior": -0.693, "means": [5.0, 3.4], "variances": [0.1, 0.2] },
    ///   ...
    /// ]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a JS error if the JSON is malformed or classes have inconsistent
    /// feature counts.
    #[wasm_bindgen(constructor)]
    pub fn new(classes_json: &str, var_smoothing: f64) -> Result<WasmNaiveBayes, JsValue> {
        let classes: Vec<GaussianClass> = serde_json::from_str(classes_json)
            .map_err(|e| WasmError::SerializationError(e.to_string()))?;

        if classes.is_empty() {
            return Err(WasmError::InvalidParameter(
                "WasmNaiveBayes: at least one class required".to_string(),
            )
            .into());
        }

        let n_features = classes[0].means.len();
        for cls in &classes {
            if cls.means.len() != n_features || cls.variances.len() != n_features {
                return Err(WasmError::InvalidParameter(format!(
                    "WasmNaiveBayes: class {} has inconsistent feature count",
                    cls.label
                ))
                .into());
            }
        }

        Ok(WasmNaiveBayes {
            classes,
            n_features,
            var_smoothing: var_smoothing.max(0.0),
        })
    }

    /// Deserialise from JSON produced by `to_json`.
    pub fn from_json(json: &str) -> Result<WasmNaiveBayes, JsValue> {
        serde_json::from_str(json)
            .map_err(|e| WasmError::SerializationError(e.to_string()).into())
    }

    /// Serialise to JSON.
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(self)
            .map_err(|e| WasmError::SerializationError(e.to_string()).into())
    }

    /// Predict the class label for a single sample.
    ///
    /// # Errors
    ///
    /// Returns a JS error if `x.len() != n_features`.
    pub fn predict(&self, x: &[f64]) -> Result<u32, JsValue> {
        self.predict_internal(x)
    }

    /// Predict class labels for a batch of samples.
    ///
    /// # Arguments
    ///
    /// * `x`          – flat row-major buffer, `n_samples × n_features`
    /// * `n_features` – features per row (must match model)
    pub fn predict_batch(&self, x: &[f64], n_features: usize) -> Result<Vec<u32>, JsValue> {
        if n_features != self.n_features {
            return Err(WasmError::InvalidParameter(format!(
                "WasmNaiveBayes::predict_batch: expected {} features, got {}",
                self.n_features, n_features
            ))
            .into());
        }
        if x.len() % n_features != 0 {
            return Err(WasmError::InvalidParameter(
                "WasmNaiveBayes::predict_batch: x.len() not divisible by n_features".to_string(),
            )
            .into());
        }

        let n_samples = x.len() / n_features;
        let mut out = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let sample = &x[i * n_features..(i + 1) * n_features];
            let label = self.predict_internal(sample)?;
            out.push(label);
        }

        Ok(out)
    }

    /// Return log-posterior scores for all classes given a sample.
    ///
    /// # Errors
    ///
    /// Returns a JS error if `x.len() != n_features`.
    pub fn log_posteriors(&self, x: &[f64]) -> Result<Vec<f64>, JsValue> {
        if x.len() != self.n_features {
            return Err(WasmError::InvalidParameter(format!(
                "WasmNaiveBayes::log_posteriors: expected {} features, got {}",
                self.n_features,
                x.len()
            ))
            .into());
        }
        Ok(self.classes.iter().map(|c| self.log_posterior(c, x)).collect())
    }

    /// Return the number of classes.
    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }

    /// Return the number of expected features.
    pub fn n_features(&self) -> usize {
        self.n_features
    }
}

impl WasmNaiveBayes {
    fn log_posterior(&self, cls: &GaussianClass, x: &[f64]) -> f64 {
        let mut log_prob = cls.log_prior;
        for (i, &xi) in x.iter().enumerate() {
            let mean = cls.means[i];
            let var = cls.variances[i] + self.var_smoothing;
            // Gaussian log-likelihood: -0.5 * (log(2π var) + (x-μ)²/σ²)
            let diff = xi - mean;
            log_prob += -0.5 * ((2.0 * std::f64::consts::PI * var).ln() + diff * diff / var);
        }
        log_prob
    }

    fn predict_internal(&self, x: &[f64]) -> Result<u32, JsValue> {
        if x.len() != self.n_features {
            return Err(WasmError::InvalidParameter(format!(
                "WasmNaiveBayes::predict: expected {} features, got {}",
                self.n_features,
                x.len()
            ))
            .into());
        }

        let best = self
            .classes
            .iter()
            .max_by(|a, b| {
                let la = self.log_posterior(a, x);
                let lb = self.log_posterior(b, x);
                la.partial_cmp(&lb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| {
                WasmError::ComputationError("WasmNaiveBayes: no classes defined".to_string())
            })?;

        Ok(best.label)
    }
}

// ---------------------------------------------------------------------------
// Convenience free functions
// ---------------------------------------------------------------------------

/// Create and immediately run batch prediction with a linear model loaded
/// from a JSON string.
///
/// Convenience function for one-shot inference without allocating a model
/// object on the Rust side.
///
/// # Errors
///
/// Returns a JS error if the model JSON is invalid or the data is malformed.
#[wasm_bindgen]
pub fn linear_model_predict_batch(
    model_json: &str,
    x: &[f64],
    n_features: usize,
) -> Result<Vec<f64>, JsValue> {
    let model = WasmLinearModel::from_json(model_json)?;
    model.predict_batch(x, n_features)
}

/// Convenience: run k-means batch prediction without holding a model object.
#[wasm_bindgen]
pub fn kmeans_predict_batch(
    model_json: &str,
    x: &[f64],
    n_features: usize,
) -> Result<Vec<u32>, JsValue> {
    let model = WasmKMeans::from_json(model_json)?;
    model.predict_batch(x, n_features)
}

/// Convenience: run Naïve Bayes batch prediction.
#[wasm_bindgen]
pub fn naive_bayes_predict_batch(
    model_json: &str,
    x: &[f64],
    n_features: usize,
) -> Result<Vec<u32>, JsValue> {
    let model = WasmNaiveBayes::from_json(model_json)?;
    model.predict_batch(x, n_features)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // WasmLinearModel
    // -----------------------------------------------------------------------

    #[test]
    fn test_linear_model_predict() {
        // y = 2x₀ + 3x₁ + 1
        let model = WasmLinearModel::new(vec![2.0, 3.0], 1.0).expect("model ok");
        let pred = model.predict(&[1.0, 1.0]).expect("predict ok");
        assert!((pred - 6.0).abs() < 1e-10, "expected 6, got {pred}");
    }

    #[test]
    fn test_linear_model_batch() {
        let model = WasmLinearModel::new(vec![1.0, 1.0], 0.0).expect("model ok");
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let preds = model.predict_batch(&x, 2).expect("batch ok");
        assert_eq!(preds.len(), 2);
        assert!((preds[0] - 3.0).abs() < 1e-10);
        assert!((preds[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_model_feature_mismatch() {
        let model = WasmLinearModel::new(vec![1.0, 2.0], 0.0).expect("ok");
        assert!(model.predict(&[1.0]).is_err());
    }

    #[test]
    fn test_linear_model_json_roundtrip() {
        let model = WasmLinearModel::new(vec![1.5, -2.3, 0.7], 0.42).expect("ok");
        let json = model.to_json().expect("to_json ok");
        let recovered = WasmLinearModel::from_json(&json).expect("from_json ok");
        assert!((recovered.bias - model.bias).abs() < 1e-12);
        for (a, b) in recovered.weights.iter().zip(model.weights.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    // -----------------------------------------------------------------------
    // WasmKMeans
    // -----------------------------------------------------------------------

    #[test]
    fn test_kmeans_predict() {
        // 2 clusters in 2D: cluster 0 near origin, cluster 1 near (10,10)
        let centroids = vec![0.0_f64, 0.0, 10.0, 10.0];
        let km = WasmKMeans::new(centroids, 2, 2).expect("kmeans ok");

        let label_near = km.predict(&[0.5, 0.5]).expect("predict ok");
        let label_far = km.predict(&[9.8, 10.2]).expect("predict ok");

        assert_eq!(label_near, 0, "should assign to cluster 0");
        assert_eq!(label_far, 1, "should assign to cluster 1");
    }

    #[test]
    fn test_kmeans_batch() {
        let centroids = vec![0.0_f64, 0.0, 10.0, 10.0];
        let km = WasmKMeans::new(centroids, 2, 2).expect("ok");
        let x = vec![0.1, 0.1, 9.9, 9.9, 0.2, -0.1];
        let labels = km.predict_batch(&x, 2).expect("batch ok");
        assert_eq!(labels, vec![0, 1, 0]);
    }

    #[test]
    fn test_kmeans_size_mismatch() {
        let result = WasmKMeans::new(vec![0.0; 5], 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_kmeans_json_roundtrip() {
        let km = WasmKMeans::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).expect("ok");
        let json = km.to_json().expect("to_json ok");
        let recovered = WasmKMeans::from_json(&json).expect("from_json ok");
        assert_eq!(recovered.k, 2);
        assert_eq!(recovered.n_features, 2);
    }

    // -----------------------------------------------------------------------
    // WasmNaiveBayes
    // -----------------------------------------------------------------------

    fn make_nb_json() -> String {
        // Equal priors for 2 classes: log(1/2) = log(0.5) ≈ -0.693
        let log_prior = 0.5_f64.ln();
        serde_json::json!([
            {
                "label": 0,
                "log_prior": log_prior,
                "means": [1.0, 2.0],
                "variances": [0.1, 0.1]
            },
            {
                "label": 1,
                "log_prior": log_prior,
                "means": [5.0, 6.0],
                "variances": [0.1, 0.1]
            }
        ])
        .to_string()
    }

    #[test]
    fn test_naive_bayes_predict() {
        let nb = WasmNaiveBayes::new(&make_nb_json(), 1e-9).expect("nb ok");

        // Close to class 0 centroid
        let label = nb.predict(&[1.05, 1.95]).expect("predict ok");
        assert_eq!(label, 0);

        // Close to class 1 centroid
        let label = nb.predict(&[4.9, 6.1]).expect("predict ok");
        assert_eq!(label, 1);
    }

    #[test]
    fn test_naive_bayes_batch() {
        let nb = WasmNaiveBayes::new(&make_nb_json(), 1e-9).expect("nb ok");
        let x = vec![1.0_f64, 2.0, 5.0, 6.0, 1.1, 2.1];
        let labels = nb.predict_batch(&x, 2).expect("batch ok");
        assert_eq!(labels, vec![0, 1, 0]);
    }

    #[test]
    fn test_naive_bayes_json_roundtrip() {
        let nb = WasmNaiveBayes::new(&make_nb_json(), 1e-9).expect("ok");
        let json = nb.to_json().expect("to_json ok");
        let recovered = WasmNaiveBayes::from_json(&json).expect("from_json ok");
        assert_eq!(recovered.n_classes(), 2);
        assert_eq!(recovered.n_features(), 2);
    }

    #[test]
    fn test_naive_bayes_wrong_features() {
        let nb = WasmNaiveBayes::new(&make_nb_json(), 1e-9).expect("ok");
        assert!(nb.predict(&[1.0]).is_err());
    }
}
