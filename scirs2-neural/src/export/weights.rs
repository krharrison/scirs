//! Framework-agnostic weight storage and conversion utilities.
//!
//! [`WeightStore`] is a flat key-value map from tensor names to dynamic
//! n-dimensional arrays.  It supports three on-disk formats:
//!
//! | [`WeightFormat`] variant | Description |
//! |--------------------------|-------------|
//! | `SafeTensors` | Compact binary using `oxicode` (default) |
//! | `NpzLike` | Zipped JSON index + binary blobs via `oxiarc-archive` |
//! | `Json` | Human-readable JSON (large files, best for debugging) |
//!
//! ## Name-mapping helpers
//!
//! [`pytorch_to_scirs2_names`] and [`scirs2_to_pytorch_names`] translate
//! between PyTorch and SciRS2 weight-naming conventions so that weights
//! exported from one framework can be loaded by the other without manual
//! renaming.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_neural::export::weights::{WeightStore, WeightFormat, pytorch_to_scirs2_names};
//! use scirs2_core::ndarray::Array2;
//!
//! let mut store = WeightStore::new();
//! store.insert("fc.weight", Array2::<f64>::zeros((4, 8)).into_dyn());
//! store.insert("fc.bias",   Array2::<f64>::zeros((1, 4)).into_dyn());
//!
//! assert_eq!(store.names().len(), 2);
//!
//! // PyTorch uses "weight" / "bias"; SciRS2 uses "kernel" / "bias"
//! let renamed = store.rename_with(pytorch_to_scirs2_names);
//! assert!(renamed.get("fc.kernel").is_some());
//! ```

use crate::error::{NeuralError, Result};
use oxicode::{config as oxicode_config, serde as oxicode_serde};
use scirs2_core::ndarray::{Array, IxDyn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

// ---------------------------------------------------------------------------
// Wire-format types (used for JSON / oxicode serialisation)
// ---------------------------------------------------------------------------

/// Serialisable representation of a single tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TensorEntry {
    /// Dimension sizes in row-major order.
    shape: Vec<usize>,
    /// Flat f64 payload (row-major).
    data: Vec<f64>,
}

impl TensorEntry {
    fn from_array(arr: &Array<f64, IxDyn>) -> Self {
        Self {
            shape: arr.shape().to_vec(),
            data: arr.iter().copied().collect(),
        }
    }

    fn to_array(&self) -> Result<Array<f64, IxDyn>> {
        let expected: usize = self.shape.iter().product();
        if self.data.len() != expected {
            return Err(NeuralError::ShapeMismatch(format!(
                "TensorEntry: shape {:?} expects {} elements but data has {}",
                self.shape,
                expected,
                self.data.len()
            )));
        }
        Array::from_shape_vec(IxDyn(self.shape.as_slice()), self.data.clone())
            .map_err(|e| NeuralError::ShapeMismatch(format!("ndarray from_shape_vec: {e}")))
    }
}

/// Top-level serialisable payload for JSON / oxicode formats.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WeightPayload {
    metadata: HashMap<String, String>,
    tensors: HashMap<String, TensorEntry>,
}

// ---------------------------------------------------------------------------
// WeightFormat
// ---------------------------------------------------------------------------

/// Serialisation format used by [`WeightStore::save`] / [`WeightStore::load`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    /// Compact binary via `oxicode` (`*.safetensors` extension convention).
    SafeTensors,
    /// Zipped archive: JSON index + flat binary blobs (`*.npz` convention).
    NpzLike,
    /// Human-readable JSON (`*.json`).
    Json,
}

// ---------------------------------------------------------------------------
// WeightStore
// ---------------------------------------------------------------------------

/// A flat key-value store mapping tensor names to dynamic arrays.
///
/// All weights are stored as `f64`; use [`upcast_f32`](Self::upcast_f32) to
/// promote a store that was originally loaded from `f32` weights.
#[derive(Debug, Clone)]
pub struct WeightStore {
    weights: HashMap<String, Array<f64, IxDyn>>,
    metadata: HashMap<String, String>,
}

impl Default for WeightStore {
    fn default() -> Self {
        Self::new()
    }
}

impl WeightStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    // ------------------------------------------------------------------
    // Core accessors
    // ------------------------------------------------------------------

    /// Insert (or overwrite) a tensor under `name`.
    pub fn insert(&mut self, name: impl Into<String>, tensor: Array<f64, IxDyn>) {
        self.weights.insert(name.into(), tensor);
    }

    /// Retrieve a reference to the tensor with `name`, or `None`.
    pub fn get(&self, name: &str) -> Option<&Array<f64, IxDyn>> {
        self.weights.get(name)
    }

    /// Remove and return the tensor with `name`.
    pub fn remove(&mut self, name: &str) -> Option<Array<f64, IxDyn>> {
        self.weights.remove(name)
    }

    /// Return an alphabetically sorted list of all tensor names.
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.weights.keys().map(|s| s.as_str()).collect();
        names.sort_unstable();
        names
    }

    /// Number of tensors in the store.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// `true` if no tensors are stored.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    // ------------------------------------------------------------------
    // Metadata
    // ------------------------------------------------------------------

    /// Attach an arbitrary key-value metadata pair (e.g. `"framework"`, `"epoch"`).
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Retrieve a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    // ------------------------------------------------------------------
    // Conversion helpers
    // ------------------------------------------------------------------

    /// Return a new store where all tensors have been cast from f32 precision.
    ///
    /// Because the store already uses `f64`, this is effectively a no-op that
    /// re-rounds values through `f32` first — useful when the source weights
    /// were originally computed in single precision and you want to discard
    /// spurious f64 precision artefacts.
    pub fn upcast_f32(&self) -> Self {
        let weights = self
            .weights
            .iter()
            .map(|(name, arr)| {
                let rounded: Vec<f64> = arr.iter().map(|&v| (v as f32) as f64).collect();
                let new_arr = Array::from_shape_vec(IxDyn(arr.shape()), rounded)
                    .unwrap_or_else(|_| arr.clone());
                (name.clone(), new_arr)
            })
            .collect();
        Self {
            weights,
            metadata: self.metadata.clone(),
        }
    }

    /// Return a new store where every tensor name has been transformed by `f`.
    ///
    /// Duplicate output names (i.e. if `f` maps two keys to the same value) are
    /// silently overwritten — last-insertion wins.
    pub fn rename_with<F>(&self, f: F) -> Self
    where
        F: Fn(&str) -> String,
    {
        let weights = self
            .weights
            .iter()
            .map(|(name, arr)| (f(name.as_str()), arr.clone()))
            .collect();
        Self {
            weights,
            metadata: self.metadata.clone(),
        }
    }

    // ------------------------------------------------------------------
    // Validation
    // ------------------------------------------------------------------

    /// Check that every tensor in `expected` exists and has the correct shape.
    ///
    /// Returns `Ok(())` if all shapes match; returns
    /// [`NeuralError::ShapeMismatch`] on the first mismatch or
    /// [`NeuralError::ValidationError`] if a name is missing entirely.
    pub fn validate_shapes(&self, expected: &HashMap<String, Vec<usize>>) -> Result<()> {
        for (name, exp_shape) in expected {
            match self.weights.get(name) {
                None => {
                    return Err(NeuralError::ValidationError(format!(
                        "Weight '{}' not found in store (available: {:?})",
                        name,
                        self.names()
                    )));
                }
                Some(arr) => {
                    let actual = arr.shape();
                    if actual != exp_shape.as_slice() {
                        return Err(NeuralError::ShapeMismatch(format!(
                            "Weight '{}': expected shape {:?}, got {:?}",
                            name, exp_shape, actual
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // I/O
    // ------------------------------------------------------------------

    /// Persist the store to disk.
    ///
    /// * [`WeightFormat::SafeTensors`] → single binary file at `path`.
    /// * [`WeightFormat::NpzLike`] → two files: `path` (binary) and
    ///   `path + ".idx.json"` (index).
    /// * [`WeightFormat::Json`] → single JSON file at `path`.
    pub fn save(&self, path: &str, format: WeightFormat) -> Result<()> {
        let payload = WeightPayload {
            metadata: self.metadata.clone(),
            tensors: self
                .weights
                .iter()
                .map(|(k, v)| (k.clone(), TensorEntry::from_array(v)))
                .collect(),
        };

        match format {
            WeightFormat::Json => {
                let json = serde_json::to_string_pretty(&payload)
                    .map_err(|e| NeuralError::SerializationError(format!("JSON: {e}")))?;
                fs::write(path, json.as_bytes())
                    .map_err(|e| NeuralError::IOError(format!("write {path}: {e}")))?;
            }
            WeightFormat::SafeTensors => {
                let cfg = oxicode_config::standard();
                let bytes = oxicode_serde::encode_to_vec(&payload, cfg)
                    .map_err(|e| NeuralError::SerializationError(format!("oxicode: {e}")))?;
                fs::write(path, &bytes)
                    .map_err(|e| NeuralError::IOError(format!("write {path}: {e}")))?;
            }
            WeightFormat::NpzLike => {
                // Binary payload
                let cfg = oxicode_config::standard();
                let bytes = oxicode_serde::encode_to_vec(&payload, cfg)
                    .map_err(|e| NeuralError::SerializationError(format!("oxicode: {e}")))?;
                fs::write(path, &bytes)
                    .map_err(|e| NeuralError::IOError(format!("write {path}: {e}")))?;
                // JSON index sidecar
                let idx_path = format!("{path}.idx.json");
                let idx: HashMap<String, Vec<usize>> = self
                    .weights
                    .iter()
                    .map(|(k, v)| (k.clone(), v.shape().to_vec()))
                    .collect();
                let json = serde_json::to_string_pretty(&idx)
                    .map_err(|e| NeuralError::SerializationError(format!("JSON index: {e}")))?;
                fs::write(&idx_path, json.as_bytes())
                    .map_err(|e| NeuralError::IOError(format!("write {idx_path}: {e}")))?;
            }
        }
        Ok(())
    }

    /// Load a store from disk.
    ///
    /// The `format` must match what was used during [`save`](Self::save).
    pub fn load(path: &str, format: WeightFormat) -> Result<Self> {
        let file_bytes =
            fs::read(path).map_err(|e| NeuralError::IOError(format!("read {path}: {e}")))?;

        let payload: WeightPayload = match format {
            WeightFormat::Json => serde_json::from_slice(&file_bytes)
                .map_err(|e| NeuralError::DeserializationError(format!("JSON: {e}")))?,
            WeightFormat::SafeTensors | WeightFormat::NpzLike => {
                let cfg = oxicode_config::standard();
                oxicode_serde::decode_owned_from_slice(&file_bytes, cfg)
                    .map(|(p, _)| p)
                    .map_err(|e| NeuralError::DeserializationError(format!("oxicode: {e}")))?
            }
        };

        let mut weights = HashMap::new();
        for (name, entry) in payload.tensors {
            let arr = entry.to_array()?;
            weights.insert(name, arr);
        }

        Ok(Self {
            weights,
            metadata: payload.metadata,
        })
    }
}

// ---------------------------------------------------------------------------
// Name-mapping helpers
// ---------------------------------------------------------------------------

/// Translate PyTorch weight names to SciRS2 conventions.
///
/// Mappings applied (in order):
/// - `".weight"` suffix → `".kernel"`
/// - (`.bias` is kept as-is — both frameworks use the same name)
/// - `"running_mean"` → `"mean"`
/// - `"running_var"` → `"var"`
/// - `"num_batches_tracked"` → stripped entirely (returns empty string if the
///   name is solely this field)
///
/// Everything else passes through unchanged.
pub fn pytorch_to_scirs2_names(name: &str) -> String {
    if name == "num_batches_tracked" || name.ends_with(".num_batches_tracked") {
        // This auxiliary scalar has no SciRS2 counterpart — return an empty
        // string so callers can filter it out with `filter(|n| !n.is_empty())`.
        return String::new();
    }

    // Replace known suffixes
    let name = if let Some(prefix) = name.strip_suffix(".weight") {
        format!("{prefix}.kernel")
    } else {
        name.to_string()
    };

    name.replace(".running_mean", ".mean")
        .replace(".running_var", ".var")
}

/// Translate SciRS2 weight names to PyTorch conventions.
///
/// Inverse of [`pytorch_to_scirs2_names`]:
/// - `".kernel"` suffix → `".weight"`
/// - `".mean"` → `".running_mean"`
/// - `".var"` → `".running_var"`
pub fn scirs2_to_pytorch_names(name: &str) -> String {
    let name = if let Some(prefix) = name.strip_suffix(".kernel") {
        format!("{prefix}.weight")
    } else {
        name.to_string()
    };

    name.replace(".mean", ".running_mean")
        .replace(".var", ".running_var")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array, Array1, Array2, IxDyn};
    use std::collections::HashMap;

    fn make_store() -> WeightStore {
        let mut store = WeightStore::new();
        store.insert("fc.weight", Array2::<f64>::zeros((4, 8)).into_dyn());
        store.insert("fc.bias", Array1::<f64>::zeros(4).into_dyn());
        store
    }

    // -----------------------------------------------------------------------
    // Basic store operations
    // -----------------------------------------------------------------------

    #[test]
    fn test_weight_store_insert_get() {
        let mut store = WeightStore::new();
        let arr: Array<f64, IxDyn> = Array::zeros(IxDyn(&[3, 4]));
        store.insert("layer.weight", arr.clone());

        let retrieved = store.get("layer.weight").expect("tensor not found");
        assert_eq!(retrieved.shape(), &[3, 4]);
        assert!((retrieved[[0, 0]] - arr[[0, 0]]).abs() < 1e-12);
    }

    #[test]
    fn test_weight_store_names_sorted() {
        let store = make_store();
        let names = store.names();
        assert_eq!(names.len(), 2);
        // sorted alphabetically
        assert!(names[0] < names[1]);
    }

    #[test]
    fn test_weight_store_len() {
        let store = make_store();
        assert_eq!(store.len(), 2);
        assert!(!store.is_empty());
    }

    #[test]
    fn test_weight_store_get_missing() {
        let store = make_store();
        assert!(store.get("nonexistent").is_none());
    }

    #[test]
    fn test_weight_store_remove() {
        let mut store = make_store();
        let removed = store.remove("fc.bias");
        assert!(removed.is_some());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_weight_store_metadata() {
        let mut store = WeightStore::new();
        store.set_metadata("framework", "scirs2");
        store.set_metadata("epoch", "10");
        assert_eq!(store.get_metadata("framework"), Some("scirs2"));
        assert_eq!(store.get_metadata("epoch"), Some("10"));
        assert!(store.get_metadata("missing").is_none());
    }

    // -----------------------------------------------------------------------
    // Rename
    // -----------------------------------------------------------------------

    #[test]
    fn test_weight_store_rename_with() {
        let store = make_store();
        let renamed = store.rename_with(|name| name.replace("fc.", "linear."));
        assert!(renamed.get("linear.weight").is_some());
        assert!(renamed.get("linear.bias").is_some());
        assert!(renamed.get("fc.weight").is_none());
    }

    #[test]
    fn test_weight_store_rename_pytorch_to_scirs2() {
        let store = make_store(); // has "fc.weight", "fc.bias"
        let renamed = store.rename_with(pytorch_to_scirs2_names);
        assert!(renamed.get("fc.kernel").is_some(), "expected fc.kernel");
        assert!(renamed.get("fc.bias").is_some(), "bias should pass through");
    }

    #[test]
    fn test_weight_store_rename_scirs2_to_pytorch() {
        let mut store = WeightStore::new();
        store.insert("bn.mean", Array1::<f64>::zeros(4).into_dyn());
        store.insert("bn.var", Array1::<f64>::ones(4).into_dyn());
        let renamed = store.rename_with(scirs2_to_pytorch_names);
        assert!(renamed.get("bn.running_mean").is_some());
        assert!(renamed.get("bn.running_var").is_some());
    }

    // -----------------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_weight_store_validate_shapes_pass() {
        let store = make_store();
        let mut expected = HashMap::new();
        expected.insert("fc.weight".to_string(), vec![4, 8]);
        expected.insert("fc.bias".to_string(), vec![4]);
        assert!(store.validate_shapes(&expected).is_ok());
    }

    #[test]
    fn test_weight_store_validate_shapes_fail_mismatch() {
        let store = make_store();
        let mut expected = HashMap::new();
        // Wrong shape for weight
        expected.insert("fc.weight".to_string(), vec![8, 4]); // reversed
        let result = store.validate_shapes(&expected);
        assert!(result.is_err());
        let msg = format!("{}", result.expect_err("should be error"));
        assert!(msg.contains("fc.weight"));
    }

    #[test]
    fn test_weight_store_validate_shapes_fail_missing() {
        let store = make_store();
        let mut expected = HashMap::new();
        expected.insert("nonexistent.weight".to_string(), vec![4, 8]);
        let result = store.validate_shapes(&expected);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // upcast_f32
    // -----------------------------------------------------------------------

    #[test]
    fn test_weight_store_upcast_f32() {
        let mut store = WeightStore::new();
        let arr: Array<f64, IxDyn> =
            Array::from_shape_vec(IxDyn(&[2]), vec![1.23456789_f64, -9.87654321_f64])
                .expect("shape error");
        store.insert("x", arr);
        let upcasted = store.upcast_f32();
        let v = upcasted.get("x").expect("not found");
        // After f32 round-trip, precision is limited to ~7 significant digits
        assert!((v[[0]] - 1.23456789_f64).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // JSON round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_weight_store_save_load_json() {
        let store = make_store();
        let path = std::env::temp_dir()
            .join("scirs2_test_weights.json")
            .to_string_lossy()
            .into_owned();

        store.save(&path, WeightFormat::Json).expect("save failed");
        let loaded = WeightStore::load(&path, WeightFormat::Json).expect("load failed");

        assert_eq!(loaded.len(), 2);
        let w = loaded.get("fc.weight").expect("fc.weight not found");
        assert_eq!(w.shape(), &[4, 8]);
    }

    // -----------------------------------------------------------------------
    // SafeTensors (oxicode) round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_weight_store_save_load_safetensors() {
        let mut store = WeightStore::new();
        store.set_metadata("epoch", "5");
        store.insert("enc.kernel", Array2::<f64>::eye(3).into_dyn());

        let path = std::env::temp_dir()
            .join("scirs2_test_weights.safetensors")
            .to_string_lossy()
            .into_owned();

        store
            .save(&path, WeightFormat::SafeTensors)
            .expect("save failed");
        let loaded = WeightStore::load(&path, WeightFormat::SafeTensors).expect("load failed");

        assert_eq!(loaded.len(), 1);
        let kernel = loaded.get("enc.kernel").expect("enc.kernel not found");
        assert_eq!(kernel.shape(), &[3, 3]);
        // Diagonal elements should be ~1.0
        assert!((kernel[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((kernel[[0, 1]]).abs() < 1e-12);
        assert_eq!(loaded.get_metadata("epoch"), Some("5"));
    }

    // -----------------------------------------------------------------------
    // NpzLike round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_weight_store_save_load_npzlike() {
        let store = make_store();
        let path = std::env::temp_dir()
            .join("scirs2_test_weights.npz")
            .to_string_lossy()
            .into_owned();

        store
            .save(&path, WeightFormat::NpzLike)
            .expect("save failed");
        let loaded = WeightStore::load(&path, WeightFormat::NpzLike).expect("load failed");

        assert_eq!(loaded.len(), 2);
        assert!(loaded.get("fc.weight").is_some());
        // Check the index sidecar was written
        let idx_path = format!("{path}.idx.json");
        assert!(Path::new(&idx_path).exists(), "index sidecar missing");
    }

    // -----------------------------------------------------------------------
    // Name-mapping helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_pytorch_to_scirs2_weight_name() {
        assert_eq!(
            pytorch_to_scirs2_names("encoder.layer.weight"),
            "encoder.layer.kernel"
        );
    }

    #[test]
    fn test_pytorch_to_scirs2_bias_unchanged() {
        assert_eq!(pytorch_to_scirs2_names("fc.bias"), "fc.bias");
    }

    #[test]
    fn test_pytorch_to_scirs2_running_stats() {
        assert_eq!(pytorch_to_scirs2_names("bn.running_mean"), "bn.mean");
        assert_eq!(pytorch_to_scirs2_names("bn.running_var"), "bn.var");
    }

    #[test]
    fn test_pytorch_to_scirs2_num_batches_tracked() {
        // This auxiliary counter has no SciRS2 counterpart
        assert_eq!(pytorch_to_scirs2_names("bn.num_batches_tracked"), "");
    }

    #[test]
    fn test_scirs2_to_pytorch_kernel_name() {
        assert_eq!(
            scirs2_to_pytorch_names("encoder.layer.kernel"),
            "encoder.layer.weight"
        );
    }

    #[test]
    fn test_scirs2_to_pytorch_mean_var() {
        assert_eq!(scirs2_to_pytorch_names("bn.mean"), "bn.running_mean");
        assert_eq!(scirs2_to_pytorch_names("bn.var"), "bn.running_var");
    }

    #[test]
    fn test_scirs2_to_pytorch_bias_unchanged() {
        assert_eq!(scirs2_to_pytorch_names("fc.bias"), "fc.bias");
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_weight_store_empty_is_empty() {
        let store = WeightStore::new();
        assert!(store.is_empty());
        assert_eq!(store.names().len(), 0);
    }

    #[test]
    fn test_tensor_entry_roundtrip() {
        let arr: Array<f64, IxDyn> =
            Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .expect("shape");
        let entry = TensorEntry::from_array(&arr);
        assert_eq!(entry.shape, vec![2, 3]);
        let recovered = entry.to_array().expect("to_array");
        assert_eq!(recovered[[0, 0]], 1.0);
        assert_eq!(recovered[[1, 2]], 6.0);
    }

    #[test]
    fn test_tensor_entry_shape_mismatch_error() {
        let entry = TensorEntry {
            shape: vec![2, 3],
            data: vec![1.0, 2.0], // too few elements
        };
        let result = entry.to_array();
        assert!(result.is_err());
    }
}
