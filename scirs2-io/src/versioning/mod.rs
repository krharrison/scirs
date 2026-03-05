//! Data versioning for scientific datasets
//!
//! This module provides Git-like versioning capabilities for scientific data,
//! enabling tracking of dataset lineage, diff computation between versions,
//! branch management, and named checkpoints.
//!
//! ## Features
//!
//! - `DataVersion`: Semantic version + content hash + timestamp
//! - `VersionedDataStore`: Persist multiple versions with metadata
//! - `DiffEngine`: Compute row-level diffs between versions
//! - `VersionGraph`: DAG of dataset versions with parent-child relationships
//! - `BranchManager`: Named branches pointing to version heads
//! - `Checkpoint`: Named snapshots of data state with restore support
//!
//! ## Example
//!
//! ```rust,no_run
//! use scirs2_io::versioning::{DataVersion, VersionedDataStore};
//! use std::collections::HashMap;
//!
//! let mut store = VersionedDataStore::new(std::env::temp_dir().join("my_ds"));
//! let v1 = DataVersion::new(1, 0, 0);
//! let rows: Vec<HashMap<String, String>> = vec![];
//! store.commit(v1, rows, Some("initial commit".to_string())).unwrap();
//! ```

#![allow(dead_code)]
#![allow(missing_docs)]

use crate::error::{IoError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// DataVersion
// ---------------------------------------------------------------------------

/// Semantic version identifier with content hash and timestamp
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DataVersion {
    /// Major version number
    pub major: u32,
    /// Minor version number
    pub minor: u32,
    /// Patch version number
    pub patch: u32,
    /// SHA-256 hex digest of the content (set after commit)
    pub hash: Option<String>,
    /// UTC creation timestamp
    pub timestamp: DateTime<Utc>,
    /// Unique commit identifier
    pub id: String,
}

impl DataVersion {
    /// Create a new version with current timestamp.
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            hash: None,
            timestamp: Utc::now(),
            id: Uuid::new_v4().to_string(),
        }
    }

    /// Return SemVer string, e.g. `"1.2.3"`.
    pub fn semver(&self) -> String {
        format!("{}.{}.{}", self.major, self.minor, self.patch)
    }

    /// Return a short hash prefix (first 8 chars) if available.
    pub fn short_hash(&self) -> Option<&str> {
        self.hash.as_deref().map(|h| &h[..h.len().min(8)])
    }

    /// Bump the patch component and return a new version.
    pub fn bump_patch(&self) -> Self {
        let mut v = Self::new(self.major, self.minor, self.patch + 1);
        v.hash = None;
        v
    }

    /// Bump the minor component (reset patch to 0) and return a new version.
    pub fn bump_minor(&self) -> Self {
        Self::new(self.major, self.minor + 1, 0)
    }

    /// Bump the major component (reset minor+patch to 0) and return a new version.
    pub fn bump_major(&self) -> Self {
        Self::new(self.major + 1, 0, 0)
    }
}

impl std::fmt::Display for DataVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "v{}.{}.{} ({})",
            self.major,
            self.minor,
            self.patch,
            &self.id[..8]
        )
    }
}

impl PartialOrd for DataVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DataVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.major, self.minor, self.patch).cmp(&(other.major, other.minor, other.patch))
    }
}

// ---------------------------------------------------------------------------
// Row type alias
// ---------------------------------------------------------------------------

/// A data row represented as a map of column-name → cell-value strings.
pub type DataRow = HashMap<String, String>;

// ---------------------------------------------------------------------------
// VersionMetadata
// ---------------------------------------------------------------------------

/// Metadata attached to each committed version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionMetadata {
    /// The version descriptor
    pub version: DataVersion,
    /// Optional human-readable commit message
    pub message: Option<String>,
    /// Parent version IDs (empty for the root commit)
    pub parents: Vec<String>,
    /// Number of rows in this snapshot
    pub row_count: usize,
    /// Column names present in this snapshot
    pub columns: Vec<String>,
    /// Arbitrary user-defined key-value tags
    pub tags: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// VersionedDataStore
// ---------------------------------------------------------------------------

/// Store that persists multiple named versions of a dataset on disk.
///
/// Layout:
/// ```text
/// <root>/
///   meta.json          ← store-level metadata (version list, current head)
///   commits/
///     <id>.json        ← per-commit metadata
///     <id>.data.json   ← row data for this commit
///   branches.json      ← branch name → version id
///   checkpoints.json   ← checkpoint name → version id
/// ```
pub struct VersionedDataStore {
    root: PathBuf,
    /// Ordered list of committed version metadata
    history: Vec<VersionMetadata>,
    /// HEAD version id
    head: Option<String>,
}

impl VersionedDataStore {
    /// Open (or create) a versioned data store at `root`.
    pub fn new<P: AsRef<Path>>(root: P) -> Self {
        let root = root.as_ref().to_path_buf();
        let _ = fs::create_dir_all(&root);
        let _ = fs::create_dir_all(root.join("commits"));

        let mut store = Self {
            root,
            history: Vec::new(),
            head: None,
        };
        // Try to load existing state
        let _ = store.load_meta();
        store
    }

    // ---- persistence helpers ----

    fn meta_path(&self) -> PathBuf {
        self.root.join("meta.json")
    }

    fn commit_meta_path(&self, id: &str) -> PathBuf {
        self.root.join("commits").join(format!("{id}.json"))
    }

    fn commit_data_path(&self, id: &str) -> PathBuf {
        self.root.join("commits").join(format!("{id}.data.json"))
    }

    fn load_meta(&mut self) -> Result<()> {
        let path = self.meta_path();
        if !path.exists() {
            return Ok(());
        }
        let mut f = fs::File::open(&path)
            .map_err(|e| IoError::FileError(format!("Cannot open meta.json: {e}")))?;
        let mut buf = String::new();
        f.read_to_string(&mut buf)
            .map_err(|e| IoError::FileError(format!("Cannot read meta.json: {e}")))?;

        #[derive(Deserialize)]
        struct StoreMeta {
            head: Option<String>,
            commit_ids: Vec<String>,
        }
        let sm: StoreMeta = serde_json::from_str(&buf)
            .map_err(|e| IoError::ParseError(format!("Bad meta.json: {e}")))?;

        self.head = sm.head;
        self.history = Vec::new();
        for id in &sm.commit_ids {
            let meta_path = self.commit_meta_path(id);
            if meta_path.exists() {
                let mut mf = fs::File::open(&meta_path)
                    .map_err(|e| IoError::FileError(format!("Cannot open commit meta: {e}")))?;
                let mut mbuf = String::new();
                mf.read_to_string(&mut mbuf)
                    .map_err(|e| IoError::FileError(format!("Cannot read commit meta: {e}")))?;
                let vm: VersionMetadata = serde_json::from_str(&mbuf)
                    .map_err(|e| IoError::ParseError(format!("Bad commit meta: {e}")))?;
                self.history.push(vm);
            }
        }
        Ok(())
    }

    fn save_meta(&self) -> Result<()> {
        #[derive(Serialize)]
        struct StoreMeta<'a> {
            head: &'a Option<String>,
            commit_ids: Vec<&'a str>,
        }
        let sm = StoreMeta {
            head: &self.head,
            commit_ids: self.history.iter().map(|vm| vm.version.id.as_str()).collect(),
        };
        let json = serde_json::to_string_pretty(&sm)
            .map_err(|e| IoError::SerializationError(format!("Cannot serialize meta: {e}")))?;
        let mut f = fs::File::create(self.meta_path())
            .map_err(|e| IoError::FileError(format!("Cannot create meta.json: {e}")))?;
        f.write_all(json.as_bytes())
            .map_err(|e| IoError::FileError(format!("Cannot write meta.json: {e}")))?;
        Ok(())
    }

    // ---- public API ----

    /// Commit a snapshot of `rows` under `version`.
    ///
    /// The content hash is computed from the JSON serialisation of `rows`.
    pub fn commit(
        &mut self,
        mut version: DataVersion,
        rows: Vec<DataRow>,
        message: Option<String>,
    ) -> Result<String> {
        // Collect columns from all rows
        let mut col_set = std::collections::BTreeSet::new();
        for row in &rows {
            for k in row.keys() {
                col_set.insert(k.clone());
            }
        }
        let columns: Vec<String> = col_set.into_iter().collect();

        // Compute content hash
        let data_json = serde_json::to_string(&rows)
            .map_err(|e| IoError::SerializationError(format!("Cannot serialize rows: {e}")))?;
        let hash = {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(data_json.as_bytes());
            format!("{:x}", hasher.finalize())
        };
        version.hash = Some(hash);

        let parent = self.head.clone().into_iter().collect::<Vec<_>>();
        let id = version.id.clone();

        let meta = VersionMetadata {
            version: version.clone(),
            message,
            parents: parent,
            row_count: rows.len(),
            columns,
            tags: HashMap::new(),
        };

        // Write commit metadata
        let meta_json = serde_json::to_string_pretty(&meta)
            .map_err(|e| IoError::SerializationError(format!("Cannot serialize commit meta: {e}")))?;
        let mut mf = fs::File::create(self.commit_meta_path(&id))
            .map_err(|e| IoError::FileError(format!("Cannot create commit meta file: {e}")))?;
        mf.write_all(meta_json.as_bytes())
            .map_err(|e| IoError::FileError(format!("Cannot write commit meta: {e}")))?;

        // Write row data
        let mut df = fs::File::create(self.commit_data_path(&id))
            .map_err(|e| IoError::FileError(format!("Cannot create data file: {e}")))?;
        df.write_all(data_json.as_bytes())
            .map_err(|e| IoError::FileError(format!("Cannot write data: {e}")))?;

        self.history.push(meta);
        self.head = Some(id.clone());
        self.save_meta()?;

        Ok(id)
    }

    /// Read the rows for a given version id.
    pub fn read_version(&self, id: &str) -> Result<Vec<DataRow>> {
        let path = self.commit_data_path(id);
        let mut f = fs::File::open(&path)
            .map_err(|_| IoError::NotFound(format!("Version {id} not found")))?;
        let mut buf = String::new();
        f.read_to_string(&mut buf)
            .map_err(|e| IoError::FileError(format!("Cannot read data: {e}")))?;
        let rows: Vec<DataRow> = serde_json::from_str(&buf)
            .map_err(|e| IoError::ParseError(format!("Bad data: {e}")))?;
        Ok(rows)
    }

    /// Return the metadata for a given version id.
    pub fn get_metadata(&self, id: &str) -> Option<&VersionMetadata> {
        self.history.iter().find(|vm| vm.version.id == id)
    }

    /// Return the full commit history (oldest first).
    pub fn history(&self) -> &[VersionMetadata] {
        &self.history
    }

    /// Current HEAD version id.
    pub fn head(&self) -> Option<&str> {
        self.head.as_deref()
    }

    /// Add a tag to an existing version.
    pub fn tag_version(&mut self, id: &str, key: String, value: String) -> Result<()> {
        let vm = self
            .history
            .iter_mut()
            .find(|vm| vm.version.id == id)
            .ok_or_else(|| IoError::NotFound(format!("Version {id} not found")))?;
        vm.tags.insert(key, value);

        // Persist updated metadata
        let meta_json = serde_json::to_string_pretty(vm)
            .map_err(|e| IoError::SerializationError(format!("{e}")))?;
        let mut f = fs::File::create(self.commit_meta_path(id))
            .map_err(|e| IoError::FileError(format!("{e}")))?;
        f.write_all(meta_json.as_bytes())
            .map_err(|e| IoError::FileError(format!("{e}")))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DiffEngine
// ---------------------------------------------------------------------------

/// Change type for a single row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RowChange {
    /// Row was added in the newer version (index in new)
    Added(usize),
    /// Row was removed from the older version (index in old)
    Removed(usize),
    /// Row was modified: (old_index, new_index, changed columns)
    Modified(usize, usize, Vec<String>),
}

/// Result of comparing two dataset versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffResult {
    /// Rows added in `new` relative to `old`
    pub added: Vec<DataRow>,
    /// Rows removed from `old`
    pub removed: Vec<DataRow>,
    /// Rows that exist in both but differ
    pub modified: Vec<(DataRow, DataRow, Vec<String>)>,
    /// Row-level change records
    pub changes: Vec<RowChange>,
    /// Summary statistics
    pub summary: DiffSummary,
}

/// High-level statistics about a diff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffSummary {
    /// Total rows in old version
    pub old_rows: usize,
    /// Total rows in new version
    pub new_rows: usize,
    /// Number of added rows
    pub added: usize,
    /// Number of removed rows
    pub removed: usize,
    /// Number of modified rows
    pub modified: usize,
    /// Number of unchanged rows
    pub unchanged: usize,
}

/// Engine for computing row-level diffs between dataset versions.
///
/// The diff algorithm:
/// 1. Build a fingerprint for each row using its JSON hash.
/// 2. Find rows only in `old` (removed), only in `new` (added).
/// 3. For rows present in both, compare column-by-column to detect modifications.
pub struct DiffEngine {
    /// Column to use as a primary key for matching rows (None = position-based)
    pub key_column: Option<String>,
}

impl DiffEngine {
    /// Create a new diff engine.  If `key_column` is `Some`, rows are matched
    /// by that column value; otherwise positional matching is used.
    pub fn new(key_column: Option<String>) -> Self {
        Self { key_column }
    }

    /// Compute the diff between `old` and `new` row sets.
    pub fn diff(&self, old: &[DataRow], new: &[DataRow]) -> DiffResult {
        match &self.key_column {
            Some(key) => self.diff_by_key(old, new, key),
            None => self.diff_positional(old, new),
        }
    }

    fn row_hash(row: &DataRow) -> String {
        let mut pairs: Vec<(&String, &String)> = row.iter().collect();
        pairs.sort_by_key(|(k, _)| *k);
        let serialised = serde_json::to_string(&pairs).unwrap_or_default();
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(serialised.as_bytes());
        format!("{:x}", h.finalize())
    }

    fn diff_by_key(&self, old: &[DataRow], new: &[DataRow], key: &str) -> DiffResult {
        // Index old rows by key value
        let mut old_by_key: HashMap<String, (usize, &DataRow)> = HashMap::new();
        for (i, row) in old.iter().enumerate() {
            if let Some(k) = row.get(key) {
                old_by_key.insert(k.clone(), (i, row));
            }
        }
        let mut new_by_key: HashMap<String, (usize, &DataRow)> = HashMap::new();
        for (i, row) in new.iter().enumerate() {
            if let Some(k) = row.get(key) {
                new_by_key.insert(k.clone(), (i, row));
            }
        }

        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut modified = Vec::new();
        let mut changes = Vec::new();
        let mut unchanged = 0usize;

        // Removed rows
        for (kv, (oi, old_row)) in &old_by_key {
            if !new_by_key.contains_key(kv) {
                removed.push((*old_row).clone());
                changes.push(RowChange::Removed(*oi));
            }
        }

        // Added & modified rows
        for (kv, (ni, new_row)) in &new_by_key {
            if let Some((oi, old_row)) = old_by_key.get(kv) {
                if Self::row_hash(old_row) == Self::row_hash(new_row) {
                    unchanged += 1;
                } else {
                    let changed_cols = Self::changed_columns(old_row, new_row);
                    modified.push(((*old_row).clone(), (*new_row).clone(), changed_cols.clone()));
                    changes.push(RowChange::Modified(*oi, *ni, changed_cols));
                }
            } else {
                added.push((*new_row).clone());
                changes.push(RowChange::Added(*ni));
            }
        }

        let summary = DiffSummary {
            old_rows: old.len(),
            new_rows: new.len(),
            added: added.len(),
            removed: removed.len(),
            modified: modified.len(),
            unchanged,
        };

        DiffResult { added, removed, modified, changes, summary }
    }

    fn diff_positional(&self, old: &[DataRow], new: &[DataRow]) -> DiffResult {
        let min_len = old.len().min(new.len());
        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut modified = Vec::new();
        let mut changes = Vec::new();
        let mut unchanged = 0usize;

        for i in 0..min_len {
            if Self::row_hash(&old[i]) == Self::row_hash(&new[i]) {
                unchanged += 1;
            } else {
                let changed_cols = Self::changed_columns(&old[i], &new[i]);
                modified.push((old[i].clone(), new[i].clone(), changed_cols.clone()));
                changes.push(RowChange::Modified(i, i, changed_cols));
            }
        }

        for i in min_len..old.len() {
            removed.push(old[i].clone());
            changes.push(RowChange::Removed(i));
        }

        for i in min_len..new.len() {
            added.push(new[i].clone());
            changes.push(RowChange::Added(i));
        }

        let summary = DiffSummary {
            old_rows: old.len(),
            new_rows: new.len(),
            added: added.len(),
            removed: removed.len(),
            modified: modified.len(),
            unchanged,
        };

        DiffResult { added, removed, modified, changes, summary }
    }

    fn changed_columns(old: &DataRow, new: &DataRow) -> Vec<String> {
        let mut cols = std::collections::BTreeSet::new();
        for (k, v) in old {
            match new.get(k) {
                Some(nv) if nv != v => { cols.insert(k.clone()); }
                None => { cols.insert(k.clone()); }
                _ => {}
            }
        }
        for k in new.keys() {
            if !old.contains_key(k) {
                cols.insert(k.clone());
            }
        }
        cols.into_iter().collect()
    }
}

// ---------------------------------------------------------------------------
// VersionGraph
// ---------------------------------------------------------------------------

/// A node in the version DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionNode {
    /// Version id
    pub id: String,
    /// SemVer string
    pub semver: String,
    /// Parent version ids
    pub parents: Vec<String>,
    /// Child version ids
    pub children: Vec<String>,
    /// Optional commit message
    pub message: Option<String>,
}

/// A directed acyclic graph of dataset versions.
///
/// - Each node corresponds to one committed version.
/// - Edges represent parent → child (chronological) relationships.
/// - Supports merge commits (multiple parents).
pub struct VersionGraph {
    nodes: HashMap<String, VersionNode>,
    roots: Vec<String>,
}

impl VersionGraph {
    /// Create an empty version graph.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            roots: Vec::new(),
        }
    }

    /// Build a `VersionGraph` from a `VersionedDataStore`.
    pub fn from_store(store: &VersionedDataStore) -> Self {
        let mut g = Self::new();
        for vm in store.history() {
            g.add_version(
                vm.version.id.clone(),
                vm.version.semver(),
                vm.parents.clone(),
                vm.message.clone(),
            );
        }
        g
    }

    /// Add a version node to the graph.
    pub fn add_version(
        &mut self,
        id: String,
        semver: String,
        parents: Vec<String>,
        message: Option<String>,
    ) {
        // Update parent nodes' children lists
        for parent_id in &parents {
            if let Some(parent) = self.nodes.get_mut(parent_id) {
                if !parent.children.contains(&id) {
                    parent.children.push(id.clone());
                }
            }
        }

        if parents.is_empty() {
            self.roots.push(id.clone());
        }

        self.nodes.insert(
            id.clone(),
            VersionNode {
                id,
                semver,
                parents,
                children: Vec::new(),
                message,
            },
        );
    }

    /// Get a node by id.
    pub fn get(&self, id: &str) -> Option<&VersionNode> {
        self.nodes.get(id)
    }

    /// Return root node ids (nodes with no parents).
    pub fn roots(&self) -> &[String] {
        &self.roots
    }

    /// Topological ordering (BFS from roots).
    pub fn topological_order(&self) -> Vec<&VersionNode> {
        let mut order = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();

        for root in &self.roots {
            queue.push_back(root.as_str());
        }

        while let Some(id) = queue.pop_front() {
            if visited.contains(id) {
                continue;
            }
            visited.insert(id);
            if let Some(node) = self.nodes.get(id) {
                order.push(node);
                for child in &node.children {
                    queue.push_back(child.as_str());
                }
            }
        }
        order
    }

    /// Compute the shortest path (as node ids) between two versions.
    /// Returns `None` if no path exists.
    pub fn shortest_path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        if from == to {
            return Some(vec![from.to_string()]);
        }
        let mut visited = std::collections::HashSet::new();
        let mut queue: std::collections::VecDeque<Vec<String>> = std::collections::VecDeque::new();
        queue.push_back(vec![from.to_string()]);
        visited.insert(from.to_string());

        while let Some(path) = queue.pop_front() {
            let current = path.last().expect("path is never empty");
            if let Some(node) = self.nodes.get(current) {
                // Traverse both children and parents
                let mut neighbors: Vec<String> = node.children.clone();
                neighbors.extend(node.parents.iter().cloned());
                for neighbor in neighbors {
                    if neighbor == to {
                        let mut result = path.clone();
                        result.push(neighbor);
                        return Some(result);
                    }
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor.clone());
                        let mut new_path = path.clone();
                        new_path.push(neighbor);
                        queue.push_back(new_path);
                    }
                }
            }
        }
        None
    }

    /// Count total nodes in the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Return true if the graph has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

// ---------------------------------------------------------------------------
// BranchManager
// ---------------------------------------------------------------------------

/// Named references pointing to version head ids.
///
/// Branches are persisted as a JSON file alongside the `VersionedDataStore`.
pub struct BranchManager {
    path: PathBuf,
    branches: HashMap<String, String>,
}

impl BranchManager {
    /// Load (or create) the branch manager associated with `store_root`.
    pub fn for_store<P: AsRef<Path>>(store_root: P) -> Result<Self> {
        let path = store_root.as_ref().join("branches.json");
        let branches = if path.exists() {
            let mut f = fs::File::open(&path)
                .map_err(|e| IoError::FileError(format!("Cannot open branches.json: {e}")))?;
            let mut buf = String::new();
            f.read_to_string(&mut buf)
                .map_err(|e| IoError::FileError(format!("Cannot read branches.json: {e}")))?;
            serde_json::from_str(&buf)
                .map_err(|e| IoError::ParseError(format!("Bad branches.json: {e}")))?
        } else {
            HashMap::new()
        };
        Ok(Self { path, branches })
    }

    /// Create or update a branch to point to `version_id`.
    pub fn set_branch(&mut self, name: impl Into<String>, version_id: impl Into<String>) -> Result<()> {
        self.branches.insert(name.into(), version_id.into());
        self.persist()
    }

    /// Get the version id that `name` points to.
    pub fn get_branch(&self, name: &str) -> Option<&str> {
        self.branches.get(name).map(|s| s.as_str())
    }

    /// Delete a branch (does NOT delete the version data).
    pub fn delete_branch(&mut self, name: &str) -> Result<()> {
        self.branches.remove(name);
        self.persist()
    }

    /// List all branches with their target version ids.
    pub fn list_branches(&self) -> Vec<(&str, &str)> {
        self.branches.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect()
    }

    fn persist(&self) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.branches)
            .map_err(|e| IoError::SerializationError(format!("{e}")))?;
        let mut f = fs::File::create(&self.path)
            .map_err(|e| IoError::FileError(format!("Cannot create branches.json: {e}")))?;
        f.write_all(json.as_bytes())
            .map_err(|e| IoError::FileError(format!("Cannot write branches.json: {e}")))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Checkpoint
// ---------------------------------------------------------------------------

/// A named, restorable snapshot of a dataset's state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointEntry {
    /// Unique checkpoint name
    pub name: String,
    /// Version id this checkpoint captures
    pub version_id: String,
    /// When this checkpoint was created
    pub created_at: DateTime<Utc>,
    /// Optional description
    pub description: Option<String>,
}

/// Manages named checkpoints for a `VersionedDataStore`.
pub struct Checkpoint {
    path: PathBuf,
    entries: Vec<CheckpointEntry>,
}

impl Checkpoint {
    /// Load (or create) checkpoint storage for `store_root`.
    pub fn for_store<P: AsRef<Path>>(store_root: P) -> Result<Self> {
        let path = store_root.as_ref().join("checkpoints.json");
        let entries = if path.exists() {
            let mut f = fs::File::open(&path)
                .map_err(|e| IoError::FileError(format!("Cannot open checkpoints.json: {e}")))?;
            let mut buf = String::new();
            f.read_to_string(&mut buf)
                .map_err(|e| IoError::FileError(format!("Cannot read checkpoints.json: {e}")))?;
            serde_json::from_str(&buf)
                .map_err(|e| IoError::ParseError(format!("Bad checkpoints.json: {e}")))?
        } else {
            Vec::new()
        };
        Ok(Self { path, entries })
    }

    /// Save a new checkpoint pointing to `version_id`.
    pub fn save(
        &mut self,
        name: impl Into<String>,
        version_id: impl Into<String>,
        description: Option<String>,
    ) -> Result<()> {
        let name = name.into();
        // Remove any existing checkpoint with this name
        self.entries.retain(|e| e.name != name);
        self.entries.push(CheckpointEntry {
            name,
            version_id: version_id.into(),
            created_at: Utc::now(),
            description,
        });
        self.persist()
    }

    /// Return the version id for a named checkpoint.
    pub fn restore(&self, name: &str) -> Result<&str> {
        self.entries
            .iter()
            .find(|e| e.name == name)
            .map(|e| e.version_id.as_str())
            .ok_or_else(|| IoError::NotFound(format!("Checkpoint '{name}' not found")))
    }

    /// Delete a named checkpoint.
    pub fn delete(&mut self, name: &str) -> Result<()> {
        let before = self.entries.len();
        self.entries.retain(|e| e.name != name);
        if self.entries.len() == before {
            return Err(IoError::NotFound(format!("Checkpoint '{name}' not found")));
        }
        self.persist()
    }

    /// List all checkpoints.
    pub fn list(&self) -> &[CheckpointEntry] {
        &self.entries
    }

    fn persist(&self) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.entries)
            .map_err(|e| IoError::SerializationError(format!("{e}")))?;
        let mut f = fs::File::create(&self.path)
            .map_err(|e| IoError::FileError(format!("Cannot create checkpoints.json: {e}")))?;
        f.write_all(json.as_bytes())
            .map_err(|e| IoError::FileError(format!("Cannot write checkpoints.json: {e}")))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    fn make_row(pairs: &[(&str, &str)]) -> DataRow {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn test_data_version_ordering() {
        let v1 = DataVersion::new(1, 0, 0);
        let v2 = DataVersion::new(1, 0, 1);
        let v3 = DataVersion::new(2, 0, 0);
        assert!(v1 < v2);
        assert!(v2 < v3);
    }

    #[test]
    fn test_version_bump() {
        let v = DataVersion::new(1, 2, 3);
        assert_eq!(v.bump_patch().semver(), "1.2.4");
        assert_eq!(v.bump_minor().semver(), "1.3.0");
        assert_eq!(v.bump_major().semver(), "2.0.0");
    }

    #[test]
    fn test_versioned_store_commit_and_read() {
        let dir = temp_dir().join(format!("scirs2_vs_{}", Uuid::new_v4()));
        let mut store = VersionedDataStore::new(&dir);

        let rows = vec![
            make_row(&[("id", "1"), ("name", "Alice")]),
            make_row(&[("id", "2"), ("name", "Bob")]),
        ];
        let v1 = DataVersion::new(1, 0, 0);
        let id = store.commit(v1, rows.clone(), Some("initial".to_string())).unwrap();

        let loaded = store.read_version(&id).unwrap();
        assert_eq!(loaded.len(), 2);

        // Re-open store and verify persistence
        let store2 = VersionedDataStore::new(&dir);
        assert_eq!(store2.history().len(), 1);
        assert_eq!(store2.head(), Some(id.as_str()));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_diff_engine_key_based() {
        let old = vec![
            make_row(&[("id", "1"), ("v", "100")]),
            make_row(&[("id", "2"), ("v", "200")]),
        ];
        let new = vec![
            make_row(&[("id", "1"), ("v", "999")]),
            make_row(&[("id", "3"), ("v", "300")]),
        ];
        let engine = DiffEngine::new(Some("id".to_string()));
        let diff = engine.diff(&old, &new);

        assert_eq!(diff.summary.added, 1);
        assert_eq!(diff.summary.removed, 1);
        assert_eq!(diff.summary.modified, 1);
    }

    #[test]
    fn test_diff_engine_positional() {
        let old = vec![make_row(&[("x", "1")]), make_row(&[("x", "2")])];
        let new = vec![make_row(&[("x", "1")]), make_row(&[("x", "99")])];
        let engine = DiffEngine::new(None);
        let diff = engine.diff(&old, &new);
        assert_eq!(diff.summary.unchanged, 1);
        assert_eq!(diff.summary.modified, 1);
    }

    #[test]
    fn test_version_graph() {
        let dir = temp_dir().join(format!("scirs2_vg_{}", Uuid::new_v4()));
        let mut store = VersionedDataStore::new(&dir);
        let id1 = store.commit(DataVersion::new(1, 0, 0), vec![], None).unwrap();
        let id2 = store.commit(DataVersion::new(1, 1, 0), vec![], None).unwrap();
        let id3 = store.commit(DataVersion::new(1, 2, 0), vec![], None).unwrap();

        let graph = VersionGraph::from_store(&store);
        assert_eq!(graph.len(), 3);
        assert_eq!(graph.roots().len(), 1);

        let path = graph.shortest_path(&id1, &id3).unwrap();
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], id1);
        assert_eq!(path[2], id3);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_branch_manager() {
        let dir = temp_dir().join(format!("scirs2_bm_{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let mut bm = BranchManager::for_store(&dir).unwrap();
        bm.set_branch("main", "v1-id").unwrap();
        bm.set_branch("dev", "v2-id").unwrap();
        assert_eq!(bm.get_branch("main"), Some("v1-id"));
        assert_eq!(bm.list_branches().len(), 2);
        bm.delete_branch("dev").unwrap();
        assert_eq!(bm.list_branches().len(), 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_checkpoint() {
        let dir = temp_dir().join(format!("scirs2_cp_{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let mut cp = Checkpoint::for_store(&dir).unwrap();
        cp.save("stable", "v1-id", Some("stable release".to_string())).unwrap();
        cp.save("beta", "v2-id", None).unwrap();
        assert_eq!(cp.list().len(), 2);
        assert_eq!(cp.restore("stable").unwrap(), "v1-id");
        cp.delete("beta").unwrap();
        assert_eq!(cp.list().len(), 1);
        let _ = fs::remove_dir_all(&dir);
    }
}
