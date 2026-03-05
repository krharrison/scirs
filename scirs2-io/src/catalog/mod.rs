//! Data catalog — registry of datasets with metadata, search, and lineage.
//!
//! Provides a persistent, searchable catalog of dataset entries, each described
//! by a schema, location, format, tags, and description.
//!
//! # Example
//!
//! ```rust
//! use scirs2_io::catalog::{DataCatalog, DatasetEntry, DataFormat, CatalogSearcher, SearchFilter};
//! use scirs2_io::schema::{Schema, SchemaField, FieldType};
//!
//! let mut catalog = DataCatalog::new();
//!
//! let entry = DatasetEntry::builder()
//!     .name("iris")
//!     .description("Classic Iris flower dataset")
//!     .location("/data/iris.csv")
//!     .format(DataFormat::Csv)
//!     .tag("biology")
//!     .tag("classification")
//!     .schema(
//!         Schema::builder()
//!             .field(SchemaField::new("sepal_length", FieldType::Float64))
//!             .field(SchemaField::new("species", FieldType::Utf8))
//!             .build()
//!     )
//!     .build();
//!
//! catalog.register(entry).expect("register failed");
//!
//! let results = CatalogSearcher::new(&catalog)
//!     .filter(SearchFilter::Tag("biology".to_string()))
//!     .search();
//! assert_eq!(results.len(), 1);
//! ```

#![allow(missing_docs)]

use crate::error::{IoError, Result};
use crate::lineage::DataLineage;
use crate::schema::Schema;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ─── DataFormat ──────────────────────────────────────────────────────────────

/// The storage format of a catalogued dataset.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DataFormat {
    /// Comma-separated values
    Csv,
    /// Tab-separated values
    Tsv,
    /// JSON Lines (NDJSON)
    JsonLines,
    /// JSON document
    Json,
    /// Apache Parquet columnar
    Parquet,
    /// HDF5 hierarchical data
    Hdf5,
    /// NetCDF scientific array data
    NetCdf,
    /// Apache Arrow IPC
    ArrowIpc,
    /// MATLAB .mat file
    Matlab,
    /// Matrix Market sparse format
    MatrixMarket,
    /// NumPy npy/npz
    Numpy,
    /// WAV audio
    Wav,
    /// Custom format with a name
    Custom(String),
    /// Unknown format
    Unknown,
}

impl DataFormat {
    /// Return a lowercase string identifier.
    pub fn as_str(&self) -> &str {
        match self {
            DataFormat::Csv => "csv",
            DataFormat::Tsv => "tsv",
            DataFormat::JsonLines => "jsonlines",
            DataFormat::Json => "json",
            DataFormat::Parquet => "parquet",
            DataFormat::Hdf5 => "hdf5",
            DataFormat::NetCdf => "netcdf",
            DataFormat::ArrowIpc => "arrow_ipc",
            DataFormat::Matlab => "matlab",
            DataFormat::MatrixMarket => "matrix_market",
            DataFormat::Numpy => "numpy",
            DataFormat::Wav => "wav",
            DataFormat::Custom(name) => name.as_str(),
            DataFormat::Unknown => "unknown",
        }
    }

    /// Guess format from file extension.
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "csv" => DataFormat::Csv,
            "tsv" | "tab" => DataFormat::Tsv,
            "jsonl" | "ndjson" => DataFormat::JsonLines,
            "json" => DataFormat::Json,
            "parquet" => DataFormat::Parquet,
            "h5" | "hdf5" | "hdf" => DataFormat::Hdf5,
            "nc" | "netcdf" | "nc4" => DataFormat::NetCdf,
            "arrow" | "ipc" | "feather" => DataFormat::ArrowIpc,
            "mat" => DataFormat::Matlab,
            "mtx" | "mm" => DataFormat::MatrixMarket,
            "npy" | "npz" => DataFormat::Numpy,
            "wav" => DataFormat::Wav,
            other => DataFormat::Custom(other.to_string()),
        }
    }
}

impl std::fmt::Display for DataFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ─── DatasetEntry ────────────────────────────────────────────────────────────

/// A single entry in the data catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetEntry {
    /// Unique identifier (UUID v4)
    pub id: Uuid,
    /// Dataset name (unique within the catalog)
    pub name: String,
    /// Human-readable description
    pub description: Option<String>,
    /// Physical or logical location (file path, URL, table name, etc.)
    pub location: String,
    /// Storage format
    pub format: DataFormat,
    /// Associated schema (optional)
    pub schema: Option<Schema>,
    /// User-defined tags for search and filtering
    pub tags: Vec<String>,
    /// Arbitrary key-value metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// When this entry was registered
    pub created_at: DateTime<Utc>,
    /// When this entry was last updated
    pub updated_at: DateTime<Utc>,
    /// Number of rows (if known)
    pub row_count: Option<u64>,
    /// Number of columns (if known)
    pub column_count: Option<u64>,
    /// Size in bytes (if known)
    pub size_bytes: Option<u64>,
    /// Owner / author
    pub owner: Option<String>,
    /// Version string
    pub version: Option<String>,
    /// Whether the dataset is currently accessible
    pub is_active: bool,
}

impl DatasetEntry {
    /// Start building a new entry.
    pub fn builder() -> DatasetEntryBuilder {
        DatasetEntryBuilder::default()
    }

    /// Update the `updated_at` timestamp to now.
    pub fn touch(&mut self) {
        self.updated_at = Utc::now();
    }

    /// Add a tag if not already present.
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        let tag = tag.into();
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    /// Remove a tag.
    pub fn remove_tag(&mut self, tag: &str) {
        self.tags.retain(|t| t != tag);
    }

    /// Check if this entry has a specific tag.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }

    /// Set a metadata value.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.metadata.insert(key.into(), value);
        self.updated_at = Utc::now();
    }

    /// Get a metadata value.
    pub fn get_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.metadata.get(key)
    }
}

// ─── DatasetEntryBuilder ────────────────────────────────────────────────────

/// Fluent builder for [`DatasetEntry`].
#[derive(Debug, Default)]
pub struct DatasetEntryBuilder {
    name: Option<String>,
    description: Option<String>,
    location: Option<String>,
    format: Option<DataFormat>,
    schema: Option<Schema>,
    tags: Vec<String>,
    metadata: HashMap<String, serde_json::Value>,
    row_count: Option<u64>,
    column_count: Option<u64>,
    size_bytes: Option<u64>,
    owner: Option<String>,
    version: Option<String>,
}

impl DatasetEntryBuilder {
    /// Set the dataset name (required).
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the location (required).
    pub fn location(mut self, loc: impl Into<String>) -> Self {
        self.location = Some(loc.into());
        self
    }

    /// Set the storage format (required).
    pub fn format(mut self, fmt: DataFormat) -> Self {
        self.format = Some(fmt);
        self
    }

    /// Set the schema.
    pub fn schema(mut self, schema: Schema) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Add a tag.
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add a metadata key-value.
    pub fn metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Set row count.
    pub fn row_count(mut self, n: u64) -> Self {
        self.row_count = Some(n);
        self
    }

    /// Set column count.
    pub fn column_count(mut self, n: u64) -> Self {
        self.column_count = Some(n);
        self
    }

    /// Set size in bytes.
    pub fn size_bytes(mut self, bytes: u64) -> Self {
        self.size_bytes = Some(bytes);
        self
    }

    /// Set owner.
    pub fn owner(mut self, owner: impl Into<String>) -> Self {
        self.owner = Some(owner.into());
        self
    }

    /// Set version.
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Finalize the entry. Panics if `name` or `location` is missing.
    pub fn build(self) -> DatasetEntry {
        let now = Utc::now();
        DatasetEntry {
            id: Uuid::new_v4(),
            name: self.name.expect("DatasetEntry requires a name"),
            description: self.description,
            location: self.location.unwrap_or_default(),
            format: self.format.unwrap_or(DataFormat::Unknown),
            schema: self.schema,
            tags: self.tags,
            metadata: self.metadata,
            created_at: now,
            updated_at: now,
            row_count: self.row_count,
            column_count: self.column_count,
            size_bytes: self.size_bytes,
            owner: self.owner,
            version: self.version,
            is_active: true,
        }
    }
}

// ─── DataCatalog ─────────────────────────────────────────────────────────────

/// A registry of dataset entries, indexed by name and UUID.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct DataCatalog {
    /// All entries, keyed by UUID
    entries: HashMap<Uuid, DatasetEntry>,
    /// Name → UUID index for fast lookup
    #[serde(skip)]
    name_index: HashMap<String, Uuid>,
    /// Optional catalog name
    pub name: Option<String>,
    /// Optional catalog description
    pub description: Option<String>,
    /// Lineage tracker
    pub lineage: DataLineage,
}

impl DataCatalog {
    /// Create an empty catalog.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a catalog with a name.
    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            ..Default::default()
        }
    }

    /// Register a new entry. Returns an error if the name already exists.
    pub fn register(&mut self, entry: DatasetEntry) -> Result<Uuid> {
        if self.name_index.contains_key(&entry.name) {
            return Err(IoError::ValidationError(format!(
                "Dataset '{}' already registered in catalog",
                entry.name
            )));
        }
        let id = entry.id;
        self.name_index.insert(entry.name.clone(), id);
        self.entries.insert(id, entry);
        Ok(id)
    }

    /// Register or replace an existing entry with the same name.
    pub fn upsert(&mut self, entry: DatasetEntry) -> Uuid {
        let id = entry.id;
        // Remove old entry for same name (if any)
        if let Some(old_id) = self.name_index.get(&entry.name).copied() {
            self.entries.remove(&old_id);
        }
        self.name_index.insert(entry.name.clone(), id);
        self.entries.insert(id, entry);
        id
    }

    /// Look up an entry by name.
    pub fn get_by_name(&self, name: &str) -> Option<&DatasetEntry> {
        let id = self.name_index.get(name)?;
        self.entries.get(id)
    }

    /// Look up an entry by UUID.
    pub fn get_by_id(&self, id: Uuid) -> Option<&DatasetEntry> {
        self.entries.get(&id)
    }

    /// Get a mutable reference to an entry by name.
    pub fn get_by_name_mut(&mut self, name: &str) -> Option<&mut DatasetEntry> {
        let id = *self.name_index.get(name)?;
        self.entries.get_mut(&id)
    }

    /// Remove an entry by name. Returns the removed entry, if present.
    pub fn deregister(&mut self, name: &str) -> Option<DatasetEntry> {
        let id = self.name_index.remove(name)?;
        self.entries.remove(&id)
    }

    /// Return all entries as a slice-like iterator.
    pub fn all_entries(&self) -> impl Iterator<Item = &DatasetEntry> {
        self.entries.values()
    }

    /// Return the number of registered entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the catalog has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Rebuild the name index from stored entries (needed after deserialization).
    pub fn rebuild_index(&mut self) {
        self.name_index.clear();
        for (id, entry) in &self.entries {
            self.name_index.insert(entry.name.clone(), *id);
        }
    }

    /// Serialize the catalog to a JSON string.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| IoError::SerializationError(e.to_string()))
    }

    /// Deserialize a catalog from a JSON string.
    /// Note: rebuilds the name index automatically.
    pub fn from_json(json: &str) -> Result<Self> {
        let mut catalog: Self =
            serde_json::from_str(json).map_err(|e| IoError::DeserializationError(e.to_string()))?;
        catalog.rebuild_index();
        Ok(catalog)
    }

    /// Save catalog to a JSON file.
    pub fn save_to_file(&self, path: &str) -> Result<()> {
        let json = self.to_json()?;
        std::fs::write(path, json).map_err(|e| IoError::Io(e))
    }

    /// Load catalog from a JSON file.
    pub fn load_from_file(path: &str) -> Result<Self> {
        let json = std::fs::read_to_string(path).map_err(|e| IoError::Io(e))?;
        Self::from_json(&json)
    }

    /// Return all unique tags across all entries.
    pub fn all_tags(&self) -> Vec<String> {
        let mut tags: std::collections::HashSet<String> = std::collections::HashSet::new();
        for entry in self.entries.values() {
            for tag in &entry.tags {
                tags.insert(tag.clone());
            }
        }
        let mut sorted: Vec<String> = tags.into_iter().collect();
        sorted.sort();
        sorted
    }

    /// Return all unique formats in the catalog.
    pub fn all_formats(&self) -> Vec<DataFormat> {
        let mut seen: Vec<DataFormat> = Vec::new();
        for entry in self.entries.values() {
            if !seen.contains(&entry.format) {
                seen.push(entry.format.clone());
            }
        }
        seen
    }
}

// ─── CatalogSearcher ────────────────────────────────────────────────────────

/// A filter criterion for catalog searches.
#[derive(Clone)]
pub enum SearchFilter {
    /// Match entries with a specific tag
    Tag(String),
    /// Match entries whose name contains the given substring (case-insensitive)
    NameContains(String),
    /// Match entries with a specific format
    Format(DataFormat),
    /// Match entries owned by a specific owner
    Owner(String),
    /// Match entries that have a schema
    HasSchema,
    /// Match entries that are active
    IsActive,
    /// Combine multiple filters with AND logic
    And(Vec<SearchFilter>),
    /// Combine multiple filters with OR logic
    Or(Vec<SearchFilter>),
    /// Negate a filter
    Not(Box<SearchFilter>),
    /// Custom predicate (non-serializable; for runtime use only)
    #[allow(clippy::type_complexity)]
    Custom(String, std::sync::Arc<dyn Fn(&DatasetEntry) -> bool + Send + Sync>),
}

impl std::fmt::Debug for SearchFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchFilter::Tag(s) => write!(f, "Tag({s:?})"),
            SearchFilter::NameContains(s) => write!(f, "NameContains({s:?})"),
            SearchFilter::Format(fmt) => write!(f, "Format({fmt:?})"),
            SearchFilter::Owner(s) => write!(f, "Owner({s:?})"),
            SearchFilter::HasSchema => write!(f, "HasSchema"),
            SearchFilter::IsActive => write!(f, "IsActive"),
            SearchFilter::And(v) => write!(f, "And({v:?})"),
            SearchFilter::Or(v) => write!(f, "Or({v:?})"),
            SearchFilter::Not(b) => write!(f, "Not({b:?})"),
            SearchFilter::Custom(name, _) => write!(f, "Custom({name:?}, <fn>)"),
        }
    }
}

impl SearchFilter {
    /// Evaluate whether an entry matches this filter.
    pub fn matches(&self, entry: &DatasetEntry) -> bool {
        match self {
            SearchFilter::Tag(tag) => entry.has_tag(tag),
            SearchFilter::NameContains(substr) => entry
                .name
                .to_lowercase()
                .contains(&substr.to_lowercase()),
            SearchFilter::Format(fmt) => &entry.format == fmt,
            SearchFilter::Owner(owner) => {
                entry.owner.as_deref() == Some(owner.as_str())
            }
            SearchFilter::HasSchema => entry.schema.is_some(),
            SearchFilter::IsActive => entry.is_active,
            SearchFilter::And(filters) => filters.iter().all(|f| f.matches(entry)),
            SearchFilter::Or(filters) => filters.iter().any(|f| f.matches(entry)),
            SearchFilter::Not(filter) => !filter.matches(entry),
            SearchFilter::Custom(_, pred) => pred(entry),
        }
    }
}

/// Searches a [`DataCatalog`] with composable filters.
pub struct CatalogSearcher<'a> {
    catalog: &'a DataCatalog,
    filters: Vec<SearchFilter>,
    max_results: Option<usize>,
    sort_by: SortOrder,
}

/// Sort order for search results.
#[derive(Debug, Clone, Default)]
pub enum SortOrder {
    /// Sort by name ascending (default)
    #[default]
    NameAsc,
    /// Sort by name descending
    NameDesc,
    /// Sort by creation time, newest first
    NewestFirst,
    /// Sort by creation time, oldest first
    OldestFirst,
    /// Sort by row count descending
    LargestFirst,
}

impl<'a> CatalogSearcher<'a> {
    /// Create a new searcher over `catalog`.
    pub fn new(catalog: &'a DataCatalog) -> Self {
        Self {
            catalog,
            filters: Vec::new(),
            max_results: None,
            sort_by: SortOrder::NameAsc,
        }
    }

    /// Add a filter criterion.
    pub fn filter(mut self, f: SearchFilter) -> Self {
        self.filters.push(f);
        self
    }

    /// Limit the number of results.
    pub fn limit(mut self, n: usize) -> Self {
        self.max_results = Some(n);
        self
    }

    /// Set sort order.
    pub fn sort_by(mut self, order: SortOrder) -> Self {
        self.sort_by = order;
        self
    }

    /// Execute the search and return matching entries.
    pub fn search(&self) -> Vec<&DatasetEntry> {
        let mut results: Vec<&DatasetEntry> = self
            .catalog
            .all_entries()
            .filter(|entry| self.filters.iter().all(|f| f.matches(entry)))
            .collect();

        match &self.sort_by {
            SortOrder::NameAsc => results.sort_by(|a, b| a.name.cmp(&b.name)),
            SortOrder::NameDesc => results.sort_by(|a, b| b.name.cmp(&a.name)),
            SortOrder::NewestFirst => results.sort_by(|a, b| b.created_at.cmp(&a.created_at)),
            SortOrder::OldestFirst => results.sort_by(|a, b| a.created_at.cmp(&b.created_at)),
            SortOrder::LargestFirst => results.sort_by(|a, b| {
                b.row_count
                    .unwrap_or(0)
                    .cmp(&a.row_count.unwrap_or(0))
            }),
        }

        if let Some(n) = self.max_results {
            results.truncate(n);
        }

        results
    }
}

// ─── CatalogSerializer ───────────────────────────────────────────────────────

/// Serialize and deserialize a [`DataCatalog`] to/from various formats.
pub struct CatalogSerializer;

impl CatalogSerializer {
    /// Serialize to JSON bytes.
    pub fn to_json_bytes(catalog: &DataCatalog) -> Result<Vec<u8>> {
        serde_json::to_vec_pretty(catalog)
            .map_err(|e| IoError::SerializationError(e.to_string()))
    }

    /// Deserialize from JSON bytes.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<DataCatalog> {
        let mut catalog: DataCatalog =
            serde_json::from_slice(bytes).map_err(|e| IoError::DeserializationError(e.to_string()))?;
        catalog.rebuild_index();
        Ok(catalog)
    }

    /// Save catalog to a JSON file atomically (write to temp, rename).
    pub fn save(catalog: &DataCatalog, path: &str) -> Result<()> {
        let bytes = Self::to_json_bytes(catalog)?;
        let tmp = format!("{path}.tmp");
        std::fs::write(&tmp, &bytes).map_err(IoError::Io)?;
        std::fs::rename(&tmp, path).map_err(IoError::Io)?;
        Ok(())
    }

    /// Load catalog from a JSON file.
    pub fn load(path: &str) -> Result<DataCatalog> {
        let bytes = std::fs::read(path).map_err(IoError::Io)?;
        Self::from_json_bytes(&bytes)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{FieldType, Schema, SchemaField};

    fn sample_entry(name: &str, format: DataFormat, tags: &[&str]) -> DatasetEntry {
        let mut b = DatasetEntry::builder()
            .name(name)
            .location(format!("/data/{name}.csv"))
            .format(format);
        for t in tags {
            b = b.tag(*t);
        }
        b.build()
    }

    #[test]
    fn test_catalog_register_and_retrieve() {
        let mut cat = DataCatalog::new();
        let entry = sample_entry("iris", DataFormat::Csv, &["biology", "ml"]);
        let id = cat.register(entry).unwrap();
        let found = cat.get_by_id(id).unwrap();
        assert_eq!(found.name, "iris");
        assert_eq!(found.format, DataFormat::Csv);
    }

    #[test]
    fn test_catalog_duplicate_name_error() {
        let mut cat = DataCatalog::new();
        cat.register(sample_entry("iris", DataFormat::Csv, &[])).unwrap();
        let result = cat.register(sample_entry("iris", DataFormat::Csv, &[]));
        assert!(result.is_err());
    }

    #[test]
    fn test_catalog_search_by_tag() {
        let mut cat = DataCatalog::new();
        cat.register(sample_entry("iris", DataFormat::Csv, &["biology"])).unwrap();
        cat.register(sample_entry("mnist", DataFormat::Numpy, &["image", "ml"])).unwrap();
        cat.register(sample_entry("titanic", DataFormat::Csv, &["tabular", "ml"])).unwrap();

        let searcher = CatalogSearcher::new(&cat)
            .filter(SearchFilter::Tag("ml".to_string()));
        let results = searcher.search();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_catalog_search_by_format() {
        let mut cat = DataCatalog::new();
        cat.register(sample_entry("a", DataFormat::Csv, &[])).unwrap();
        cat.register(sample_entry("b", DataFormat::Csv, &[])).unwrap();
        cat.register(sample_entry("c", DataFormat::Parquet, &[])).unwrap();

        let searcher = CatalogSearcher::new(&cat)
            .filter(SearchFilter::Format(DataFormat::Csv));
        let results = searcher.search();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_catalog_search_name_contains() {
        let mut cat = DataCatalog::new();
        cat.register(sample_entry("train_data", DataFormat::Csv, &[])).unwrap();
        cat.register(sample_entry("test_data", DataFormat::Csv, &[])).unwrap();
        cat.register(sample_entry("validation_set", DataFormat::Csv, &[])).unwrap();

        let searcher = CatalogSearcher::new(&cat)
            .filter(SearchFilter::NameContains("data".to_string()));
        let results = searcher.search();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_catalog_json_roundtrip() {
        let mut cat = DataCatalog::with_name("my_catalog");
        let entry = DatasetEntry::builder()
            .name("ds1")
            .location("/data/ds1.parquet")
            .format(DataFormat::Parquet)
            .tag("finance")
            .row_count(10000)
            .schema(
                Schema::builder()
                    .field(SchemaField::new("price", FieldType::Float64))
                    .build(),
            )
            .build();
        cat.register(entry).unwrap();

        let json = cat.to_json().unwrap();
        let restored = DataCatalog::from_json(&json).unwrap();
        assert_eq!(restored.len(), 1);
        let ds = restored.get_by_name("ds1").unwrap();
        assert_eq!(ds.row_count, Some(10000));
        assert!(ds.schema.is_some());
    }

    #[test]
    fn test_catalog_save_load_file() {
        let tmp = std::env::temp_dir().join("test_catalog_save.json");
        let path = tmp.to_str().expect("temp path");

        let mut cat = DataCatalog::new();
        cat.register(sample_entry("test", DataFormat::Csv, &["temp"])).unwrap();
        CatalogSerializer::save(&cat, path).unwrap();

        let loaded = CatalogSerializer::load(path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert!(loaded.get_by_name("test").is_some());

        let _ = std::fs::remove_file(tmp);
    }

    #[test]
    fn test_catalog_deregister() {
        let mut cat = DataCatalog::new();
        cat.register(sample_entry("del_me", DataFormat::Csv, &[])).unwrap();
        assert!(cat.get_by_name("del_me").is_some());

        let removed = cat.deregister("del_me");
        assert!(removed.is_some());
        assert!(cat.get_by_name("del_me").is_none());
        assert_eq!(cat.len(), 0);
    }

    #[test]
    fn test_search_filter_and() {
        let mut cat = DataCatalog::new();
        cat.register(sample_entry("train_csv", DataFormat::Csv, &["ml"])).unwrap();
        cat.register(sample_entry("train_parquet", DataFormat::Parquet, &["ml"])).unwrap();
        cat.register(sample_entry("raw_csv", DataFormat::Csv, &["raw"])).unwrap();

        let searcher = CatalogSearcher::new(&cat)
            .filter(SearchFilter::And(vec![
                SearchFilter::Tag("ml".to_string()),
                SearchFilter::Format(DataFormat::Csv),
            ]));
        let results = searcher.search();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "train_csv");
    }

    #[test]
    fn test_all_tags() {
        let mut cat = DataCatalog::new();
        cat.register(sample_entry("a", DataFormat::Csv, &["x", "y"])).unwrap();
        cat.register(sample_entry("b", DataFormat::Csv, &["y", "z"])).unwrap();

        let tags = cat.all_tags();
        assert_eq!(tags.len(), 3);
        assert!(tags.contains(&"x".to_string()));
        assert!(tags.contains(&"y".to_string()));
        assert!(tags.contains(&"z".to_string()));
    }

    #[test]
    fn test_data_format_from_extension() {
        assert_eq!(DataFormat::from_extension("csv"), DataFormat::Csv);
        assert_eq!(DataFormat::from_extension("parquet"), DataFormat::Parquet);
        assert_eq!(DataFormat::from_extension("jsonl"), DataFormat::JsonLines);
        assert_eq!(DataFormat::from_extension("h5"), DataFormat::Hdf5);
    }
}
