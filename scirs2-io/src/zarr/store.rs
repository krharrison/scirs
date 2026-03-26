//! Directory-based store for Zarr chunk and metadata storage.

use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{IoError, Result};

/// A filesystem directory store for Zarr data.
///
/// Keys are forward-slash-separated paths relative to the store root.
/// The store creates subdirectories as needed when writing.
#[derive(Debug, Clone)]
pub struct DirectoryStore {
    root: PathBuf,
}

impl DirectoryStore {
    /// Open (or create) a directory store at the given path.
    pub fn open<P: AsRef<Path>>(root: P) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        if !root.exists() {
            fs::create_dir_all(&root).map_err(|e| {
                IoError::FileError(format!(
                    "Failed to create store directory {}: {e}",
                    root.display()
                ))
            })?;
        }
        Ok(Self { root })
    }

    /// Root path of the store.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Read a value by key.
    pub fn get(&self, key: &str) -> Result<Vec<u8>> {
        let path = self.key_to_path(key);
        fs::read(&path).map_err(|e| {
            IoError::FileNotFound(format!("Store key '{}' ({}): {e}", key, path.display()))
        })
    }

    /// Write a value by key (creates parent directories as needed).
    pub fn set(&self, key: &str, value: &[u8]) -> Result<()> {
        let path = self.key_to_path(key);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                IoError::FileError(format!(
                    "Failed to create directory {}: {e}",
                    parent.display()
                ))
            })?;
        }
        fs::write(&path, value)
            .map_err(|e| IoError::FileError(format!("Failed to write {}: {e}", path.display())))
    }

    /// Delete a key.
    pub fn delete(&self, key: &str) -> Result<()> {
        let path = self.key_to_path(key);
        if path.exists() {
            fs::remove_file(&path).map_err(|e| {
                IoError::FileError(format!("Failed to delete {}: {e}", path.display()))
            })?;
        }
        Ok(())
    }

    /// Check whether a key exists.
    pub fn exists(&self, key: &str) -> bool {
        self.key_to_path(key).exists()
    }

    /// List all keys under a given prefix.
    pub fn list_prefix(&self, prefix: &str) -> Result<Vec<String>> {
        let base = self.key_to_path(prefix);
        if !base.exists() {
            return Ok(Vec::new());
        }
        let mut result = Vec::new();
        self.collect_keys(&base, &self.root, &mut result)?;
        Ok(result)
    }

    /// List all keys with a given directory prefix (returns keys relative to store root).
    pub fn list_dir(&self, prefix: &str) -> Result<Vec<String>> {
        let base = self.key_to_path(prefix);
        if !base.is_dir() {
            return Ok(Vec::new());
        }
        let mut result = Vec::new();
        let entries = fs::read_dir(&base).map_err(|e| {
            IoError::FileError(format!("Failed to read dir {}: {e}", base.display()))
        })?;
        for entry in entries {
            let entry =
                entry.map_err(|e| IoError::FileError(format!("Failed to read dir entry: {e}")))?;
            let rel = entry
                .path()
                .strip_prefix(&self.root)
                .map_err(|e| IoError::FileError(format!("Strip prefix error: {e}")))?
                .to_string_lossy()
                .replace('\\', "/");
            result.push(rel);
        }
        result.sort();
        Ok(result)
    }

    // ── internal helpers ─────────────────────────────────────────────────

    fn key_to_path(&self, key: &str) -> PathBuf {
        let mut path = self.root.clone();
        for component in key.split('/') {
            if !component.is_empty() {
                path.push(component);
            }
        }
        path
    }

    fn collect_keys(&self, dir: &Path, root: &Path, out: &mut Vec<String>) -> Result<()> {
        if dir.is_file() {
            let rel = dir
                .strip_prefix(root)
                .map_err(|e| IoError::FileError(format!("Strip prefix error: {e}")))?
                .to_string_lossy()
                .replace('\\', "/");
            out.push(rel);
            return Ok(());
        }
        if !dir.is_dir() {
            return Ok(());
        }
        let entries = fs::read_dir(dir).map_err(|e| {
            IoError::FileError(format!("Failed to read dir {}: {e}", dir.display()))
        })?;
        for entry in entries {
            let entry =
                entry.map_err(|e| IoError::FileError(format!("Failed to read dir entry: {e}")))?;
            self.collect_keys(&entry.path(), root, out)?;
        }
        Ok(())
    }
}

/// Encode chunk coordinates to a v2-style key (e.g. `"0.1.2"`).
pub fn chunk_key_v2(prefix: &str, coords: &[u64], sep: &str) -> String {
    let coord_str: Vec<String> = coords.iter().map(|c| c.to_string()).collect();
    let joined = coord_str.join(sep);
    if prefix.is_empty() {
        joined
    } else {
        format!("{prefix}/{joined}")
    }
}

/// Encode chunk coordinates to a v3-style key (e.g. `"c/0/1/2"`).
pub fn chunk_key_v3(prefix: &str, coords: &[u64], separator: &str) -> String {
    let coord_str: Vec<String> = coords.iter().map(|c| c.to_string()).collect();
    let joined = coord_str.join(separator);
    if prefix.is_empty() {
        format!("c/{joined}")
    } else {
        format!("{prefix}/c/{joined}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_directory_store_write_read() {
        let dir = std::env::temp_dir().join("zarr_store_test_wr");
        let _ = fs::remove_dir_all(&dir);
        let store = DirectoryStore::open(&dir).expect("open store");

        store
            .set("group/.zgroup", b"{\"zarr_format\":2}")
            .expect("write");
        let data = store.get("group/.zgroup").expect("read");
        assert_eq!(data, b"{\"zarr_format\":2}");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_directory_store_exists_delete() {
        let dir = std::env::temp_dir().join("zarr_store_test_ed");
        let _ = fs::remove_dir_all(&dir);
        let store = DirectoryStore::open(&dir).expect("open store");

        assert!(!store.exists("foo"));
        store.set("foo", b"bar").expect("write");
        assert!(store.exists("foo"));
        store.delete("foo").expect("delete");
        assert!(!store.exists("foo"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_directory_store_list_prefix() {
        let dir = std::env::temp_dir().join("zarr_store_test_lp");
        let _ = fs::remove_dir_all(&dir);
        let store = DirectoryStore::open(&dir).expect("open store");

        store.set("arr/.zarray", b"{}").expect("w1");
        store.set("arr/0.0", b"chunk1").expect("w2");
        store.set("arr/0.1", b"chunk2").expect("w3");

        let mut keys = store.list_prefix("arr").expect("list");
        keys.sort();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&"arr/.zarray".to_string()));
        assert!(keys.contains(&"arr/0.0".to_string()));
        assert!(keys.contains(&"arr/0.1".to_string()));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_directory_store_nested_keys() {
        let dir = std::env::temp_dir().join("zarr_store_test_nk");
        let _ = fs::remove_dir_all(&dir);
        let store = DirectoryStore::open(&dir).expect("open store");

        store.set("a/b/c/d", b"deep").expect("write deep");
        assert!(store.exists("a/b/c/d"));
        let data = store.get("a/b/c/d").expect("read");
        assert_eq!(data, b"deep");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_chunk_key_v2() {
        assert_eq!(chunk_key_v2("data", &[0, 1, 2], "."), "data/0.1.2");
        assert_eq!(chunk_key_v2("", &[3, 4], "."), "3.4");
        assert_eq!(chunk_key_v2("arr", &[0], "."), "arr/0");
    }

    #[test]
    fn test_chunk_key_v3() {
        assert_eq!(chunk_key_v3("data", &[0, 1, 2], "/"), "data/c/0/1/2");
        assert_eq!(chunk_key_v3("", &[3, 4], "/"), "c/3/4");
    }

    #[test]
    fn test_get_missing_key_returns_error() {
        let dir = std::env::temp_dir().join("zarr_store_test_miss");
        let _ = fs::remove_dir_all(&dir);
        let store = DirectoryStore::open(&dir).expect("open store");

        let result = store.get("nonexistent");
        assert!(result.is_err());

        let _ = fs::remove_dir_all(&dir);
    }
}
