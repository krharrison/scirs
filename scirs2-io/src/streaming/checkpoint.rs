//! Checkpointing and restart support for long-running streaming jobs.
//!
//! Provides facilities for periodically snapshotting operator state to durable
//! storage so that a streaming job can be resumed after a failure without
//! reprocessing the entire input from the beginning.
//!
//! ## Design
//!
//! - [`CheckpointManager`] handles saving/loading raw byte blobs tagged with a
//!   `state_id` string.  Files are named `{state_id}_{timestamp_ms}.ckpt` and
//!   stored under `CheckpointConfig::storage_path`.
//! - [`CheckpointableState`] is the serialisation contract; any type that can
//!   write itself to `Vec<u8>` and reconstruct itself from `&[u8]` qualifies.
//! - [`CheckpointBarrier`] is a lightweight token injected into the stream to
//!   coordinate when all operators should snapshot their state.
//!
//! ## Simple byte serialisation
//!
//! Rather than pulling in an external dependency, primitive helpers convert
//! `Vec<f64>` and `HashMap<String, f64>` to/from little-endian bytes.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{IoError, Result};

// ---------------------------------------------------------------------------
// CheckpointableState trait
// ---------------------------------------------------------------------------

/// Any streaming operator state that can be serialised and restored.
pub trait CheckpointableState: Sized {
    /// Serialise the state to a byte vector.
    fn serialize(&self) -> Vec<u8>;

    /// Deserialise from a byte slice.
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes are malformed.
    fn deserialize(data: &[u8]) -> Result<Self>;
}

// ---------------------------------------------------------------------------
// Built-in CheckpointableState implementations
// ---------------------------------------------------------------------------

impl CheckpointableState for Vec<f64> {
    fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + self.len() * 8);
        out.extend_from_slice(&(self.len() as u64).to_le_bytes());
        for &v in self {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn deserialize(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(IoError::DeserializationError(
                "Vec<f64> checkpoint too short".to_string(),
            ));
        }
        let len = u64::from_le_bytes(
            data[..8]
                .try_into()
                .map_err(|_| IoError::DeserializationError("bad length bytes".to_string()))?,
        ) as usize;
        let expected = 8 + len * 8;
        if data.len() < expected {
            return Err(IoError::DeserializationError(format!(
                "Vec<f64> checkpoint: expected {expected} bytes, got {}",
                data.len()
            )));
        }
        let mut vec = Vec::with_capacity(len);
        for i in 0..len {
            let start = 8 + i * 8;
            let bytes: [u8; 8] = data[start..start + 8]
                .try_into()
                .map_err(|_| IoError::DeserializationError("bad f64 bytes".to_string()))?;
            vec.push(f64::from_le_bytes(bytes));
        }
        Ok(vec)
    }
}

impl CheckpointableState for HashMap<String, f64> {
    fn serialize(&self) -> Vec<u8> {
        // Format: n_entries (8 bytes) | for each: key_len (8) | key (key_len) | value (8)
        let mut out = Vec::new();
        out.extend_from_slice(&(self.len() as u64).to_le_bytes());
        // Sorted for determinism.
        let mut entries: Vec<(&String, &f64)> = self.iter().collect();
        entries.sort_by_key(|(k, _)| k.as_str());
        for (k, &v) in entries {
            let kb = k.as_bytes();
            out.extend_from_slice(&(kb.len() as u64).to_le_bytes());
            out.extend_from_slice(kb);
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn deserialize(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(IoError::DeserializationError(
                "HashMap checkpoint too short".to_string(),
            ));
        }
        let n = u64::from_le_bytes(
            data[..8]
                .try_into()
                .map_err(|_| IoError::DeserializationError("bad n bytes".to_string()))?,
        ) as usize;
        let mut cursor = 8usize;
        let mut map = HashMap::with_capacity(n);
        for _ in 0..n {
            if cursor + 8 > data.len() {
                return Err(IoError::DeserializationError(
                    "HashMap checkpoint truncated (key len)".to_string(),
                ));
            }
            let key_len = u64::from_le_bytes(
                data[cursor..cursor + 8]
                    .try_into()
                    .map_err(|_| IoError::DeserializationError("bad key len".to_string()))?,
            ) as usize;
            cursor += 8;
            if cursor + key_len + 8 > data.len() {
                return Err(IoError::DeserializationError(
                    "HashMap checkpoint truncated (key/value)".to_string(),
                ));
            }
            let key = String::from_utf8(data[cursor..cursor + key_len].to_vec()).map_err(|e| {
                IoError::DeserializationError(format!("non-UTF8 key in HashMap checkpoint: {e}"))
            })?;
            cursor += key_len;
            let value = f64::from_le_bytes(
                data[cursor..cursor + 8]
                    .try_into()
                    .map_err(|_| IoError::DeserializationError("bad f64 value".to_string()))?,
            );
            cursor += 8;
            map.insert(key, value);
        }
        Ok(map)
    }
}

// ---------------------------------------------------------------------------
// CheckpointConfig
// ---------------------------------------------------------------------------

/// Configuration for the checkpoint manager.
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Minimum interval between automatic checkpoints in milliseconds.
    pub interval_ms: u64,
    /// Directory where checkpoint files are stored.
    pub storage_path: String,
    /// Number of most-recent checkpoints to retain (older ones are deleted).
    pub max_checkpoints: usize,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            interval_ms: 60_000,
            storage_path: {
                let mut p = std::env::temp_dir();
                p.push("scirs2_checkpoint");
                p.to_string_lossy().into_owned()
            },
            max_checkpoints: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// CheckpointManager
// ---------------------------------------------------------------------------

/// Manages serialised state blobs on the local filesystem.
#[derive(Debug, Clone)]
pub struct CheckpointManager {
    config: CheckpointConfig,
}

impl CheckpointManager {
    /// Create a new manager with the given configuration.
    pub fn new(config: CheckpointConfig) -> Self {
        Self { config }
    }

    /// Create a new manager with the default configuration.
    pub fn with_defaults() -> Self {
        Self::new(CheckpointConfig::default())
    }

    /// Save `data` as a checkpoint for `state_id`.
    ///
    /// The file is named `{state_id}_{timestamp_ms}.ckpt` and written to the
    /// configured `storage_path`.  Returns the absolute path of the file
    /// written.
    ///
    /// After saving, old checkpoints beyond `max_checkpoints` are removed.
    pub fn save(&self, state_id: &str, data: &[u8]) -> Result<String> {
        let dir = Path::new(&self.config.storage_path);
        fs::create_dir_all(dir).map_err(|e| {
            IoError::FileError(format!(
                "Cannot create checkpoint directory {}: {e}",
                dir.display()
            ))
        })?;

        let ts_ms = current_time_ms();
        let filename = format!("{state_id}_{ts_ms}.ckpt");
        let path = dir.join(&filename);

        fs::write(&path, data).map_err(|e| {
            IoError::FileError(format!(
                "Failed to write checkpoint {}: {e}",
                path.display()
            ))
        })?;

        let abs = path
            .canonicalize()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| path.to_string_lossy().into_owned());

        self.cleanup_old(state_id);
        Ok(abs)
    }

    /// Load the most recently saved checkpoint for `state_id`.
    ///
    /// # Errors
    ///
    /// Returns an error if no checkpoint is found or if reading fails.
    pub fn load_latest(&self, state_id: &str) -> Result<Vec<u8>> {
        let files = self.list_checkpoints(state_id);
        let latest = files.last().ok_or_else(|| {
            IoError::NotFound(format!("No checkpoint found for state_id '{state_id}'"))
        })?;
        fs::read(latest)
            .map_err(|e| IoError::FileError(format!("Failed to read checkpoint {latest}: {e}")))
    }

    /// Delete all but the newest `max_checkpoints` checkpoint files for `state_id`.
    pub fn cleanup_old(&self, state_id: &str) {
        let files = self.list_checkpoints(state_id);
        if files.len() <= self.config.max_checkpoints {
            return;
        }
        let to_delete = files.len() - self.config.max_checkpoints;
        for path in files.iter().take(to_delete) {
            let _ = fs::remove_file(path);
        }
    }

    /// Return a sorted list of checkpoint file paths for `state_id` (oldest first).
    pub fn list_checkpoints(&self, state_id: &str) -> Vec<String> {
        let dir = Path::new(&self.config.storage_path);
        let prefix = format!("{state_id}_");
        let suffix = ".ckpt";

        let mut paths: Vec<PathBuf> = match fs::read_dir(dir) {
            Ok(entries) => entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with(&prefix) && n.ends_with(suffix))
                        .unwrap_or(false)
                })
                .collect(),
            Err(_) => Vec::new(),
        };

        // Sort by the numeric timestamp embedded in the filename.
        paths.sort_by_key(|p| extract_timestamp_ms(p));
        paths
            .into_iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// CheckpointBarrier
// ---------------------------------------------------------------------------

/// A barrier token that can be injected into a stream to signal all downstream
/// operators to checkpoint their state.
#[derive(Debug, Clone)]
pub struct CheckpointBarrier {
    /// Unique barrier identifier, monotonically increasing.
    pub checkpoint_id: u64,
    /// Wall-clock timestamp when the barrier was created (ms since epoch).
    pub timestamp_ms: u64,
}

impl CheckpointBarrier {
    /// Create a new barrier with the given id, stamping the current time.
    pub fn new(checkpoint_id: u64) -> Self {
        Self {
            checkpoint_id,
            timestamp_ms: current_time_ms(),
        }
    }

    /// Create a barrier with an explicit timestamp (useful in tests).
    pub fn with_timestamp(checkpoint_id: u64, timestamp_ms: u64) -> Self {
        Self {
            checkpoint_id,
            timestamp_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn current_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Extract the numeric timestamp from a path like `{prefix}_{ts}.ckpt`.
fn extract_timestamp_ms(path: &Path) -> u64 {
    path.file_name()
        .and_then(|n| n.to_str())
        .and_then(|name| {
            // Find the last '_' and parse everything between it and '.ckpt'.
            let without_ext = name.strip_suffix(".ckpt")?;
            let pos = without_ext.rfind('_')?;
            without_ext[pos + 1..].parse::<u64>().ok()
        })
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::env;

    fn temp_checkpoint_manager(suffix: &str) -> CheckpointManager {
        let dir = env::temp_dir().join(format!("scirs2_ckpt_test_{suffix}"));
        CheckpointManager::new(CheckpointConfig {
            storage_path: dir.to_string_lossy().into_owned(),
            max_checkpoints: 3,
            ..Default::default()
        })
    }

    #[test]
    fn test_save_and_load_latest() {
        let mgr = temp_checkpoint_manager("save_load");
        let data = b"hello checkpoint world".to_vec();
        mgr.save("job1", &data).expect("save should succeed");
        let loaded = mgr.load_latest("job1").expect("load should succeed");
        assert_eq!(loaded, data);
    }

    #[test]
    fn test_cleanup_keeps_max_checkpoints() {
        let mgr = temp_checkpoint_manager("cleanup");
        // Save 5 checkpoints.
        for i in 0..5u64 {
            // Small sleep to ensure distinct ms timestamps on fast machines.
            std::thread::sleep(std::time::Duration::from_millis(2));
            mgr.save("job2", &i.to_le_bytes()).expect("save");
        }
        let files = mgr.list_checkpoints("job2");
        assert!(
            files.len() <= mgr.config.max_checkpoints,
            "Expected ≤ {} checkpoints, found {}",
            mgr.config.max_checkpoints,
            files.len()
        );
    }

    #[test]
    fn test_load_latest_no_checkpoint_returns_error() {
        let mgr = temp_checkpoint_manager("missing");
        assert!(mgr.load_latest("nonexistent_id").is_err());
    }

    #[test]
    fn test_vec_f64_round_trip() {
        let v = vec![1.0_f64, 2.5, std::f64::consts::PI, -42.0];
        let bytes = v.serialize();
        let restored = Vec::<f64>::deserialize(&bytes).expect("deserialize");
        assert_eq!(v, restored);
    }

    #[test]
    fn test_hashmap_string_f64_round_trip() {
        let mut m = HashMap::new();
        m.insert("alpha".to_string(), 1.5_f64);
        m.insert("beta".to_string(), -0.5_f64);
        m.insert("gamma".to_string(), std::f64::consts::E);
        let bytes = m.serialize();
        let restored = HashMap::<String, f64>::deserialize(&bytes).expect("deserialize");
        assert_eq!(m, restored);
    }

    #[test]
    fn test_checkpoint_barrier_fields() {
        let barrier = CheckpointBarrier::with_timestamp(42, 999_000);
        assert_eq!(barrier.checkpoint_id, 42);
        assert_eq!(barrier.timestamp_ms, 999_000);
    }
}
