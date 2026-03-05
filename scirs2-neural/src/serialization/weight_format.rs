//! Weight-only storage format for neural network layers.
//!
//! This module provides [`WeightStore`], a flat key-value store that maps
//! layer names (e.g., `"encoder.layer.0.query.weight"`) to dense multi-dimensional
//! arrays (stored as flat `Vec<f64>` plus a shape descriptor). The store can be
//! serialized to/from disk in a compact binary format and supports partial loading
//! for transfer learning scenarios.
//!
//! ## Wire format
//!
//! The binary layout is:
//!
//! ```text
//! [8 bytes]  magic number: b"SCRSWT01"
//! [4 bytes]  u32 LE: number of tensors
//! for each tensor:
//!   [4 bytes]  u32 LE: byte length of name (UTF-8)
//!   [N bytes]  UTF-8 name
//!   [4 bytes]  u32 LE: rank (number of dimensions)
//!   [rank * 8 bytes]  u64 LE values: shape dimensions
//!   [4 bytes]  u32 LE: dtype tag  (0 = F32, 1 = F64)
//!   [elems * dtype_size bytes]  raw element data (little-endian)
//! ```
//!
//! The format is intentionally simple and self-describing so that it can be
//! parsed without a schema or external library.
//!
//! ## JSON sidecar
//!
//! [`WeightStore::save`] writes two files:
//! - `<stem>.weights` — binary payload described above
//! - `<stem>.weights.json` — human-readable index with name→shape mappings
//!
//! [`WeightStore::load`] reads only the `.weights` file; the JSON sidecar is
//! optional metadata for introspection.

use crate::error::{NeuralError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{Read as IoRead, Write as IoWrite};
use std::path::Path;

// ============================================================================
// Magic + version
// ============================================================================

/// Binary magic for the `.weights` file format.
const MAGIC: &[u8; 8] = b"SCRSWT01";

/// Dtype tag used in the binary wire format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
enum WireDtype {
    F32 = 0,
    F64 = 1,
}

impl WireDtype {
    fn element_size(self) -> usize {
        match self {
            WireDtype::F32 => 4,
            WireDtype::F64 => 8,
        }
    }

    fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(WireDtype::F32),
            1 => Ok(WireDtype::F64),
            other => Err(NeuralError::DeserializationError(format!(
                "Unknown WireDtype tag: {other}"
            ))),
        }
    }
}

// ============================================================================
// WeightEntry
// ============================================================================

/// A single named weight tensor in the store.
///
/// Values are stored as `f64` regardless of the on-disk dtype; they are
/// up-cast / down-cast during I/O.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightEntry {
    /// Layer-qualified name (e.g., `"conv1.weight"`).
    pub name: String,
    /// Shape of the tensor (e.g., `[64, 3, 3, 3]` for a Conv2D weight).
    pub shape: Vec<usize>,
    /// Flat array of values in row-major (C) order.
    pub values: Vec<f64>,
}

impl WeightEntry {
    /// Create a new weight entry.
    pub fn new(name: impl Into<String>, shape: Vec<usize>, values: Vec<f64>) -> Result<Self> {
        let name = name.into();
        let expected: usize = shape.iter().product();
        if values.len() != expected {
            return Err(NeuralError::ShapeMismatch(format!(
                "WeightEntry '{name}': expected {expected} values for shape {shape:?}, got {}",
                values.len()
            )));
        }
        Ok(Self {
            name,
            shape,
            values,
        })
    }

    /// Total number of scalar elements.
    pub fn num_elements(&self) -> usize {
        self.values.len()
    }

    /// Return the rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Return the memory footprint (bytes) assuming f64 storage.
    pub fn byte_size_f64(&self) -> usize {
        self.values.len() * 8
    }
}

// ============================================================================
// WeightStore
// ============================================================================

/// A flat, ordered collection of named weight tensors.
///
/// ## Example
///
/// ```rust
/// use scirs2_neural::serialization::weight_format::WeightStore;
///
/// let mut store = WeightStore::new();
/// store.insert("fc1.weight", vec![256, 784], vec![0.0f64; 256 * 784]).expect("insert");
/// store.insert("fc1.bias",   vec![256],      vec![0.0f64; 256]).expect("insert");
///
/// assert_eq!(store.len(), 2);
/// assert_eq!(store.total_parameters(), 256 * 784 + 256);
///
/// let json = store.to_json().expect("json");
/// let restored = WeightStore::from_json(&json).expect("parse");
/// assert_eq!(restored.len(), store.len());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightStore {
    /// Ordered list of weight entries.
    ///
    /// We keep an ordered `Vec` rather than a raw `HashMap` so that the
    /// serialization order is deterministic and matches the model's layer order.
    entries: Vec<WeightEntry>,
    /// Fast lookup: name → index in `entries`.
    #[serde(skip)]
    index: HashMap<String, usize>,
}

impl WeightStore {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create an empty [`WeightStore`].
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// Rebuild the name→index map from the entries list.
    ///
    /// Called after JSON deserialization (since `index` is `#[serde(skip)]`).
    fn rebuild_index(&mut self) {
        self.index.clear();
        for (i, entry) in self.entries.iter().enumerate() {
            self.index.insert(entry.name.clone(), i);
        }
    }

    // -----------------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------------

    /// Insert a weight tensor.
    ///
    /// Returns an error if `values.len()` does not match the product of `shape`.
    /// If an entry with the same `name` already exists it is **replaced**.
    pub fn insert(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<()> {
        let entry = WeightEntry::new(name, shape, values)?;
        let name = entry.name.clone();
        if let Some(&idx) = self.index.get(&name) {
            self.entries[idx] = entry;
        } else {
            let idx = self.entries.len();
            self.index.insert(name, idx);
            self.entries.push(entry);
        }
        Ok(())
    }

    /// Remove a weight tensor by name.
    ///
    /// Returns `true` if an entry was removed, `false` if the name was not found.
    pub fn remove(&mut self, name: &str) -> bool {
        if let Some(idx) = self.index.remove(name) {
            self.entries.remove(idx);
            // Rebuild index because all subsequent indices shifted
            self.rebuild_index();
            true
        } else {
            false
        }
    }

    // -----------------------------------------------------------------------
    // Read-only accessors
    // -----------------------------------------------------------------------

    /// Return the number of weight entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if the store has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up a weight entry by name.
    pub fn get(&self, name: &str) -> Option<&WeightEntry> {
        self.index
            .get(name)
            .and_then(|&idx| self.entries.get(idx))
    }

    /// Return an iterator over all entries in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = &WeightEntry> {
        self.entries.iter()
    }

    /// Return the names of all weight tensors in insertion order.
    pub fn names(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.name.as_str()).collect()
    }

    /// Total scalar parameter count across all entries.
    pub fn total_parameters(&self) -> usize {
        self.entries.iter().map(|e| e.num_elements()).sum()
    }

    /// Total in-memory byte footprint assuming f64 storage.
    pub fn total_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.byte_size_f64()).sum()
    }

    // -----------------------------------------------------------------------
    // JSON serialization
    // -----------------------------------------------------------------------

    /// Serialize the weight store to a JSON string.
    ///
    /// This includes all weight values as JSON arrays of `f64`. Useful for
    /// human-readable inspection but produces large files; prefer binary
    /// [`save`](WeightStore::save) for production use.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| NeuralError::SerializationError(e.to_string()))
    }

    /// Deserialize a [`WeightStore`] from a JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        let mut store: Self = serde_json::from_str(json)
            .map_err(|e| NeuralError::DeserializationError(e.to_string()))?;
        store.rebuild_index();
        Ok(store)
    }

    // -----------------------------------------------------------------------
    // Binary serialization (save / load)
    // -----------------------------------------------------------------------

    /// Save the weight store to disk in binary format.
    ///
    /// Writes two files under `path`:
    /// - `{path}` — binary payload
    /// - `{path}.json` — JSON index (human-readable sidecar)
    ///
    /// The parent directory is created automatically if it does not exist.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| NeuralError::IOError(e.to_string()))?;
            }
        }
        // Write binary weights file
        let bytes = self.to_bytes()?;
        fs::write(path, &bytes).map_err(|e| NeuralError::IOError(e.to_string()))?;

        // Write JSON sidecar
        let sidecar_path = {
            let mut p = path.to_path_buf();
            let ext = p
                .extension()
                .map(|s| format!("{}.json", s.to_string_lossy()))
                .unwrap_or_else(|| "json".to_string());
            p.set_extension(ext);
            p
        };
        let index = self.build_json_index();
        let index_json = serde_json::to_string_pretty(&index)
            .map_err(|e| NeuralError::SerializationError(e.to_string()))?;
        fs::write(&sidecar_path, index_json.as_bytes())
            .map_err(|e| NeuralError::IOError(e.to_string()))?;

        Ok(())
    }

    /// Load weights from a binary file written by [`WeightStore::save`].
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).map_err(|e| NeuralError::IOError(e.to_string()))?;
        Self::from_bytes(&bytes)
    }

    // -----------------------------------------------------------------------
    // Partial (transfer-learning) load
    // -----------------------------------------------------------------------

    /// Load weights from `path`, but only for layer names present in `allowed_names`.
    ///
    /// All tensors whose names are NOT in `allowed_names` are silently skipped.
    /// This is the primary entry-point for transfer-learning workflows where a
    /// pre-trained store may contain more layers than the target model.
    ///
    /// # Example
    ///
    /// ```rust
    /// use scirs2_neural::serialization::weight_format::WeightStore;
    /// use std::collections::HashSet;
    ///
    /// let mut store = WeightStore::new();
    /// store.insert("fc1.weight", vec![4, 2], vec![1.0f64; 8]).expect("insert");
    /// store.insert("fc2.weight", vec![2, 4], vec![2.0f64; 8]).expect("insert");
    ///
    /// let dir = std::env::temp_dir().join("scirs2_partial_load_test");
    /// let path = dir.join("model.weights");
    /// store.save(&path).expect("save");
    ///
    /// let allowed: HashSet<String> = ["fc1.weight".to_string()].into();
    /// let partial = WeightStore::partial_load(&path, &allowed).expect("partial load");
    /// assert_eq!(partial.len(), 1);
    /// assert!(partial.get("fc1.weight").is_some());
    /// assert!(partial.get("fc2.weight").is_none());
    ///
    /// let _ = std::fs::remove_dir_all(&dir);
    /// ```
    pub fn partial_load(
        path: &Path,
        allowed_names: &std::collections::HashSet<String>,
    ) -> Result<Self> {
        let full = Self::load(path)?;
        let mut filtered = WeightStore::new();
        for entry in full.iter() {
            if allowed_names.contains(&entry.name) {
                filtered.insert(entry.name.clone(), entry.shape.clone(), entry.values.clone())?;
            }
        }
        Ok(filtered)
    }

    /// Merge weights from `other` into `self`.
    ///
    /// For each entry in `other`:
    /// - If `self` already has an entry with the same name, it is **replaced**
    ///   (shape and values from `other`).
    /// - If `self` does not have that name, the entry is **inserted**.
    ///
    /// This is useful for combining pretrained weights with freshly-initialized
    /// classification heads.
    pub fn merge_from(&mut self, other: &WeightStore) -> Result<()> {
        for entry in other.iter() {
            self.insert(entry.name.clone(), entry.shape.clone(), entry.values.clone())?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Binary encoding / decoding
    // -----------------------------------------------------------------------

    /// Serialize the weight store to a `Vec<u8>` in the binary wire format.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut buf: Vec<u8> = Vec::new();
        // Magic
        buf.extend_from_slice(MAGIC);
        // Number of tensors (u32 LE)
        write_u32(&mut buf, self.entries.len() as u32);
        for entry in &self.entries {
            // Name length + name bytes
            let name_bytes = entry.name.as_bytes();
            write_u32(&mut buf, name_bytes.len() as u32);
            buf.extend_from_slice(name_bytes);
            // Rank + shape
            write_u32(&mut buf, entry.shape.len() as u32);
            for &dim in &entry.shape {
                write_u64(&mut buf, dim as u64);
            }
            // We always store as F64 for precision
            write_u32(&mut buf, WireDtype::F64 as u32);
            // Values
            for &v in &entry.values {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        Ok(buf)
    }

    /// Deserialize a [`WeightStore`] from a byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut cursor = 0usize;

        // Magic
        if bytes.len() < 8 {
            return Err(NeuralError::DeserializationError(
                "WeightStore: buffer too short for magic".to_string(),
            ));
        }
        let magic = &bytes[cursor..cursor + 8];
        if magic != MAGIC {
            return Err(NeuralError::DeserializationError(format!(
                "WeightStore: invalid magic {:?}",
                magic
            )));
        }
        cursor += 8;

        // Number of tensors
        let num_tensors = read_u32(bytes, &mut cursor)? as usize;

        let mut store = WeightStore::new();

        for _ in 0..num_tensors {
            // Name
            let name_len = read_u32(bytes, &mut cursor)? as usize;
            let name_bytes = read_bytes(bytes, &mut cursor, name_len)?;
            let name = std::str::from_utf8(name_bytes)
                .map_err(|e| NeuralError::DeserializationError(e.to_string()))?
                .to_string();

            // Shape
            let rank = read_u32(bytes, &mut cursor)? as usize;
            let mut shape = Vec::with_capacity(rank);
            for _ in 0..rank {
                let dim = read_u64(bytes, &mut cursor)? as usize;
                shape.push(dim);
            }

            // Dtype
            let dtype_tag = read_u32(bytes, &mut cursor)?;
            let dtype = WireDtype::from_u32(dtype_tag)?;
            let num_elements: usize = shape.iter().product();
            let elem_size = dtype.element_size();
            let data_bytes = read_bytes(bytes, &mut cursor, num_elements * elem_size)?;

            let values: Vec<f64> = match dtype {
                WireDtype::F32 => {
                    let mut out = Vec::with_capacity(num_elements);
                    for chunk in data_bytes.chunks_exact(4) {
                        let arr: [u8; 4] = chunk
                            .try_into()
                            .map_err(|_| NeuralError::DeserializationError(
                                "F32 chunk size error".to_string(),
                            ))?;
                        out.push(f32::from_le_bytes(arr) as f64);
                    }
                    out
                }
                WireDtype::F64 => {
                    let mut out = Vec::with_capacity(num_elements);
                    for chunk in data_bytes.chunks_exact(8) {
                        let arr: [u8; 8] = chunk
                            .try_into()
                            .map_err(|_| NeuralError::DeserializationError(
                                "F64 chunk size error".to_string(),
                            ))?;
                        out.push(f64::from_le_bytes(arr));
                    }
                    out
                }
            };

            store.insert(name, shape, values)?;
        }

        Ok(store)
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build a JSON index object used for the `.json` sidecar file.
    fn build_json_index(&self) -> serde_json::Value {
        let entries: Vec<serde_json::Value> = self
            .entries
            .iter()
            .map(|e| {
                serde_json::json!({
                    "name": e.name,
                    "shape": e.shape,
                    "num_elements": e.num_elements(),
                    "dtype": "f64",
                })
            })
            .collect();
        serde_json::json!({
            "format": "scirs2-weights-v1",
            "num_tensors": self.entries.len(),
            "total_parameters": self.total_parameters(),
            "tensors": entries,
        })
    }
}

impl Default for WeightStore {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Binary I/O helpers
// ============================================================================

#[inline]
fn write_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn write_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32> {
    if *cursor + 4 > bytes.len() {
        return Err(NeuralError::DeserializationError(
            "WeightStore: unexpected EOF reading u32".to_string(),
        ));
    }
    let arr: [u8; 4] = bytes[*cursor..*cursor + 4]
        .try_into()
        .map_err(|_| NeuralError::DeserializationError("u32 slice error".to_string()))?;
    *cursor += 4;
    Ok(u32::from_le_bytes(arr))
}

#[inline]
fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
    if *cursor + 8 > bytes.len() {
        return Err(NeuralError::DeserializationError(
            "WeightStore: unexpected EOF reading u64".to_string(),
        ));
    }
    let arr: [u8; 8] = bytes[*cursor..*cursor + 8]
        .try_into()
        .map_err(|_| NeuralError::DeserializationError("u64 slice error".to_string()))?;
    *cursor += 8;
    Ok(u64::from_le_bytes(arr))
}

#[inline]
fn read_bytes<'a>(bytes: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8]> {
    if *cursor + len > bytes.len() {
        return Err(NeuralError::DeserializationError(format!(
            "WeightStore: unexpected EOF reading {len} bytes at offset {cursor}"
        )));
    }
    let slice = &bytes[*cursor..*cursor + len];
    *cursor += len;
    Ok(slice)
}

// ============================================================================
// Convenience functions
// ============================================================================

/// Save a collection of named tensors (as `(name, flat_values, shape)` triples)
/// to a binary weight file.
///
/// This is a convenience wrapper around [`WeightStore`] for callers that already
/// have data in a flat format.
///
/// # Example
///
/// ```rust
/// use scirs2_neural::serialization::weight_format::save_weights;
///
/// let tensors: Vec<(String, Vec<f64>, Vec<usize>)> = vec![
///     ("layer1.w".to_string(), vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]),
///     ("layer1.b".to_string(), vec![0.0, 0.0],             vec![2]),
/// ];
///
/// let dir = std::env::temp_dir().join("scirs2_save_weights_test");
/// let path = dir.join("test.weights");
/// save_weights(&path, &tensors).expect("save");
/// let _ = std::fs::remove_dir_all(&dir);
/// ```
pub fn save_weights(
    path: &Path,
    tensors: &[(String, Vec<f64>, Vec<usize>)],
) -> Result<()> {
    let mut store = WeightStore::new();
    for (name, values, shape) in tensors {
        store.insert(name.clone(), shape.clone(), values.clone())?;
    }
    store.save(path)
}

/// Load weights from a binary weight file into a vector of
/// `(name, flat_values, shape)` triples.
///
/// # Example
///
/// ```rust
/// use scirs2_neural::serialization::weight_format::{save_weights, load_weights};
///
/// let tensors: Vec<(String, Vec<f64>, Vec<usize>)> = vec![
///     ("a".to_string(), vec![1.0, 2.0], vec![2]),
///     ("b".to_string(), vec![3.0],      vec![1]),
/// ];
///
/// let dir = std::env::temp_dir().join("scirs2_load_weights_test");
/// let path = dir.join("test.weights");
/// save_weights(&path, &tensors).expect("save");
///
/// let loaded = load_weights(&path).expect("load");
/// assert_eq!(loaded.len(), 2);
/// assert_eq!(loaded[0].0, "a");
///
/// let _ = std::fs::remove_dir_all(&dir);
/// ```
pub fn load_weights(path: &Path) -> Result<Vec<(String, Vec<f64>, Vec<usize>)>> {
    let store = WeightStore::load(path)?;
    Ok(store
        .iter()
        .map(|e| (e.name.clone(), e.values.clone(), e.shape.clone()))
        .collect())
}

/// Load only the subset of weights whose names are in `layer_names`.
///
/// See [`WeightStore::partial_load`] for detailed semantics.
pub fn partial_load_weights(
    path: &Path,
    layer_names: &std::collections::HashSet<String>,
) -> Result<Vec<(String, Vec<f64>, Vec<usize>)>> {
    let store = WeightStore::partial_load(path, layer_names)?;
    Ok(store
        .iter()
        .map(|e| (e.name.clone(), e.values.clone(), e.shape.clone()))
        .collect())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn make_store() -> WeightStore {
        let mut s = WeightStore::new();
        s.insert("fc1.weight", vec![4, 3], vec![1.0; 12]).expect("insert");
        s.insert("fc1.bias", vec![4], vec![0.5; 4]).expect("insert");
        s.insert("fc2.weight", vec![2, 4], vec![2.0; 8]).expect("insert");
        s.insert("fc2.bias", vec![2], vec![0.1; 2]).expect("insert");
        s
    }

    #[test]
    fn test_weight_store_basic_ops() {
        let s = make_store();
        assert_eq!(s.len(), 4);
        assert_eq!(s.total_parameters(), 12 + 4 + 8 + 2);
        assert!(!s.is_empty());

        let entry = s.get("fc1.weight").expect("should exist");
        assert_eq!(entry.shape, vec![4, 3]);
        assert_eq!(entry.values.len(), 12);
        assert!((entry.values[0] - 1.0).abs() < 1e-12);

        assert!(s.get("nonexistent").is_none());
    }

    #[test]
    fn test_weight_store_insert_replace() {
        let mut s = WeightStore::new();
        s.insert("layer.w", vec![2, 2], vec![1.0; 4]).expect("first");
        s.insert("layer.w", vec![2, 2], vec![9.0; 4]).expect("replace");
        assert_eq!(s.len(), 1);
        let e = s.get("layer.w").expect("exists");
        assert!((e.values[0] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_weight_store_remove() {
        let mut s = make_store();
        assert!(s.remove("fc1.bias"));
        assert_eq!(s.len(), 3);
        assert!(s.get("fc1.bias").is_none());
        assert!(!s.remove("nonexistent"));
    }

    #[test]
    fn test_binary_roundtrip() {
        let original = make_store();
        let bytes = original.to_bytes().expect("to_bytes");
        let restored = WeightStore::from_bytes(&bytes).expect("from_bytes");

        assert_eq!(restored.len(), original.len());
        assert_eq!(restored.total_parameters(), original.total_parameters());

        for entry in original.iter() {
            let r = restored.get(&entry.name).expect("entry should exist");
            assert_eq!(r.shape, entry.shape);
            for (a, b) in entry.values.iter().zip(r.values.iter()) {
                assert!((a - b).abs() < 1e-12, "value mismatch for {}", entry.name);
            }
        }
    }

    #[test]
    fn test_json_roundtrip() {
        let original = make_store();
        let json = original.to_json().expect("to_json");
        let restored = WeightStore::from_json(&json).expect("from_json");
        assert_eq!(restored.len(), original.len());
        for entry in original.iter() {
            assert!(restored.get(&entry.name).is_some());
        }
    }

    #[test]
    fn test_save_load_file() {
        let original = make_store();
        let dir = std::env::temp_dir().join("scirs2_weight_store_save_test");
        let path = dir.join("weights.weights");
        original.save(&path).expect("save");
        assert!(path.exists());
        // Check sidecar
        let sidecar = dir.join("weights.weights.json");
        assert!(sidecar.exists());

        let loaded = WeightStore::load(&path).expect("load");
        assert_eq!(loaded.len(), original.len());
        assert_eq!(loaded.total_parameters(), original.total_parameters());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_partial_load() {
        let original = make_store();
        let dir = std::env::temp_dir().join("scirs2_partial_load_direct_test");
        let path = dir.join("weights.weights");
        original.save(&path).expect("save");

        let allowed: HashSet<String> =
            ["fc1.weight".to_string(), "fc2.bias".to_string()].into();
        let partial = WeightStore::partial_load(&path, &allowed).expect("partial_load");
        assert_eq!(partial.len(), 2);
        assert!(partial.get("fc1.weight").is_some());
        assert!(partial.get("fc2.bias").is_some());
        assert!(partial.get("fc1.bias").is_none());
        assert!(partial.get("fc2.weight").is_none());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_weights_convenience() {
        let tensors = vec![
            ("w1".to_string(), vec![1.0f64, 2.0], vec![2usize]),
            ("w2".to_string(), vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]),
        ];
        let dir = std::env::temp_dir().join("scirs2_save_weights_conv_test");
        let path = dir.join("model.weights");
        save_weights(&path, &tensors).expect("save_weights");

        let loaded = load_weights(&path).expect("load_weights");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].0, "w1");
        assert!((loaded[0].1[0] - 1.0).abs() < 1e-12);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_partial_load_weights_convenience() {
        let tensors = vec![
            ("encoder.w".to_string(), vec![0.5f64; 4], vec![2, 2]),
            ("decoder.w".to_string(), vec![1.5f64; 6], vec![2, 3]),
            ("head.b".to_string(), vec![0.0f64; 2], vec![2]),
        ];
        let dir = std::env::temp_dir().join("scirs2_partial_weights_conv_test");
        let path = dir.join("full.weights");
        save_weights(&path, &tensors).expect("save");

        let allowed: HashSet<String> = ["encoder.w".to_string(), "head.b".to_string()].into();
        let partial = partial_load_weights(&path, &allowed).expect("partial_load");
        assert_eq!(partial.len(), 2);
        let names: Vec<&str> = partial.iter().map(|(n, _, _)| n.as_str()).collect();
        assert!(names.contains(&"encoder.w"));
        assert!(names.contains(&"head.b"));
        assert!(!names.contains(&"decoder.w"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_merge_from() {
        let mut base = WeightStore::new();
        base.insert("shared.w", vec![2], vec![1.0; 2]).expect("insert");
        base.insert("head.w", vec![3], vec![1.0; 3]).expect("insert");

        let mut new_head = WeightStore::new();
        new_head.insert("head.w", vec![3], vec![9.0; 3]).expect("insert");
        new_head.insert("extra.w", vec![1], vec![5.0]).expect("insert");

        base.merge_from(&new_head).expect("merge");
        assert_eq!(base.len(), 3);
        // head.w should be replaced
        let hw = base.get("head.w").expect("head.w");
        assert!((hw.values[0] - 9.0).abs() < 1e-12);
        // extra.w should be added
        assert!(base.get("extra.w").is_some());
        // shared.w unchanged
        let sw = base.get("shared.w").expect("shared.w");
        assert!((sw.values[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_bad_magic_rejected() {
        let mut bad = vec![0u8; 8];
        bad[0] = b'B';
        bad[1] = b'A';
        bad[2] = b'D';
        let result = WeightStore::from_bytes(&bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_truncated_bytes_rejected() {
        let result = WeightStore::from_bytes(&[0u8; 4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_mismatch_rejected() {
        let result = WeightStore::new().insert(
            "bad",
            vec![3, 3], // expects 9 elements
            vec![1.0; 5], // only 5
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_names_ordering() {
        let s = make_store();
        let names = s.names();
        assert_eq!(names, vec!["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]);
    }

    #[test]
    fn test_weight_entry_rank() {
        let entry = WeightEntry::new("x", vec![3, 4, 5], vec![0.0; 60]).expect("new");
        assert_eq!(entry.rank(), 3);
        assert_eq!(entry.num_elements(), 60);
        assert_eq!(entry.byte_size_f64(), 480);
    }

    #[test]
    fn test_empty_store_roundtrip() {
        let empty = WeightStore::new();
        let bytes = empty.to_bytes().expect("to_bytes");
        let restored = WeightStore::from_bytes(&bytes).expect("from_bytes");
        assert!(restored.is_empty());
    }

    #[test]
    fn test_large_tensor_roundtrip() {
        // 1000 × 1000 = 1M parameters
        let n = 1_000_000usize;
        let values: Vec<f64> = (0..n).map(|i| i as f64 * 0.001).collect();
        let mut store = WeightStore::new();
        store.insert("big.weight", vec![1000, 1000], values.clone()).expect("insert");

        let bytes = store.to_bytes().expect("bytes");
        let restored = WeightStore::from_bytes(&bytes).expect("restore");
        let e = restored.get("big.weight").expect("entry");
        assert_eq!(e.values.len(), n);
        assert!((e.values[999_999] - values[999_999]).abs() < 1e-9);
    }
}
