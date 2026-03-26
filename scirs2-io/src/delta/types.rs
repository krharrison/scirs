//! Delta Lake type definitions.
//!
//! Core types for the Delta Lake transaction log format including actions,
//! versions, tables, transactions, and errors.

use std::collections::HashMap;
use std::path::PathBuf;

/// Configuration for a Delta Lake table.
#[derive(Debug, Clone)]
pub struct DeltaConfig {
    /// Base path of the Delta table on disk.
    pub base_path: PathBuf,
    /// How often (in commits) to write a checkpoint file.
    pub checkpoint_interval: usize,
    /// Maximum number of data files to scan (soft cap).
    pub max_files_to_scan: usize,
}

impl Default for DeltaConfig {
    fn default() -> Self {
        Self {
            base_path: PathBuf::from("."),
            checkpoint_interval: 10,
            max_files_to_scan: 10_000,
        }
    }
}

/// A single action stored in a Delta log commit file.
///
/// Delta actions are the fundamental unit of change in the transaction log.
/// Each commit consists of one or more actions that describe file additions,
/// removals, metadata changes, or protocol upgrades.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum DeltaAction {
    /// A data file was added to the table.
    Add {
        /// Relative path of the data file.
        path: String,
        /// Size in bytes.
        size: u64,
        /// Unix epoch milliseconds of last modification.
        modification_time: u64,
        /// Whether this add represents a data change (true) or just a compaction/optimization.
        data_change: bool,
        /// Optional partition values for this file.
        partition_values: HashMap<String, String>,
        /// Optional serialised statistics JSON.
        stats_json: Option<String>,
    },
    /// A data file was logically removed (tombstoned) from the table.
    Remove {
        /// Relative path of the data file.
        path: String,
        /// Unix epoch milliseconds when the deletion was recorded.
        deletion_timestamp: u64,
        /// Whether this remove is a data change (true) or reorganization.
        data_change: bool,
    },
    /// Table metadata (schema, partition columns, etc.).
    Metadata {
        /// Schema definition as a JSON string.
        schema: String,
        /// Columns used for partitioning.
        partition_columns: Vec<String>,
        /// Optional table description.
        description: Option<String>,
        /// Additional configuration properties.
        configuration: HashMap<String, String>,
    },
    /// Protocol version requirements.
    Protocol {
        /// Minimum reader version required to read this table.
        min_reader_version: u32,
        /// Minimum writer version required to write to this table.
        min_writer_version: u32,
    },
    /// Commit information (operation metadata).
    CommitInfo {
        /// Unix epoch milliseconds at commit time.
        timestamp: i64,
        /// Human-readable operation name (e.g., "WRITE", "DELETE", "MERGE").
        operation: String,
        /// Optional operation parameters.
        operation_parameters: HashMap<String, String>,
    },
}

/// A versioned set of actions forming a single commit.
#[derive(Debug, Clone)]
pub struct DeltaVersion {
    /// Log version number (0-based, monotonically increasing).
    pub version: u64,
    /// Unix epoch milliseconds when this version was committed.
    pub timestamp: u64,
    /// All actions in this version.
    pub actions: Vec<DeltaAction>,
}

/// Column schema definition for Delta tables.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnSchema {
    /// Column name.
    pub name: String,
    /// Column data type as a string (e.g., "double", "string", "long").
    pub data_type: String,
    /// Whether the column is nullable.
    pub nullable: bool,
}

/// Schema definition for a Delta table.
#[derive(Debug, Clone, PartialEq)]
pub struct Schema {
    /// Ordered list of columns.
    pub columns: Vec<ColumnSchema>,
}

impl Schema {
    /// Create a new schema from a list of columns.
    pub fn new(columns: Vec<ColumnSchema>) -> Self {
        Self { columns }
    }

    /// Serialize the schema to a JSON string.
    pub fn to_json(&self) -> Result<String, DeltaError> {
        let fields: Vec<serde_json::Value> = self
            .columns
            .iter()
            .map(|c| {
                serde_json::json!({
                    "name": c.name,
                    "type": c.data_type,
                    "nullable": c.nullable,
                })
            })
            .collect();
        let schema_obj = serde_json::json!({
            "type": "struct",
            "fields": fields,
        });
        serde_json::to_string(&schema_obj)
            .map_err(|e| DeltaError::Serialization(format!("schema to JSON: {e}")))
    }

    /// Deserialize a schema from a JSON string.
    pub fn from_json(json_str: &str) -> Result<Self, DeltaError> {
        let v: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| DeltaError::Parse(format!("schema JSON parse: {e}")))?;
        let fields = v
            .get("fields")
            .and_then(|f| f.as_array())
            .ok_or_else(|| DeltaError::Parse("missing 'fields' array in schema".to_string()))?;
        let mut columns = Vec::with_capacity(fields.len());
        for field in fields {
            let name = field
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or_default()
                .to_string();
            let data_type = field
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("string")
                .to_string();
            let nullable = field
                .get("nullable")
                .and_then(|n| n.as_bool())
                .unwrap_or(true);
            columns.push(ColumnSchema {
                name,
                data_type,
                nullable,
            });
        }
        Ok(Self { columns })
    }

    /// Get column names.
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|c| c.name.as_str()).collect()
    }
}

/// Represents the reconstructed logical state of a Delta table.
#[derive(Debug, Clone)]
pub struct DeltaTable {
    /// Configuration for this table.
    pub config: DeltaConfig,
    /// Current table version.
    pub version: u64,
    /// Active data files (path -> Add action details).
    pub active_files: HashMap<String, FileInfo>,
    /// Current schema, if known.
    pub schema: Option<Schema>,
    /// Partition columns.
    pub partition_columns: Vec<String>,
    /// Protocol version.
    pub protocol: Option<(u32, u32)>,
}

/// Information about an active data file.
#[derive(Debug, Clone)]
pub struct FileInfo {
    /// File path relative to table root.
    pub path: String,
    /// Size in bytes.
    pub size: u64,
    /// Last modification time (Unix epoch ms).
    pub modification_time: u64,
    /// Partition values for this file.
    pub partition_values: HashMap<String, String>,
}

/// An in-progress transaction that can be committed atomically.
#[derive(Debug, Clone)]
pub struct DeltaTransaction {
    /// Actions accumulated in this transaction.
    pub actions: Vec<DeltaAction>,
    /// The version this transaction will be committed as.
    pub target_version: u64,
}

impl DeltaTransaction {
    /// Create a new transaction targeting the given version.
    pub fn new(target_version: u64) -> Self {
        Self {
            actions: Vec::new(),
            target_version,
        }
    }

    /// Add an action to this transaction.
    pub fn add_action(&mut self, action: DeltaAction) {
        self.actions.push(action);
    }
}

/// Errors specific to Delta Lake operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum DeltaError {
    /// I/O error during file operations.
    Io(std::io::Error),
    /// JSON parsing or serialization error.
    Parse(String),
    /// Serialization error.
    Serialization(String),
    /// Version conflict during optimistic concurrency.
    VersionConflict {
        /// The version the transaction expected to write.
        expected: u64,
        /// The actual latest version found on disk.
        actual: u64,
    },
    /// Table not found or not initialized.
    TableNotFound(String),
    /// Schema evolution error.
    SchemaError(String),
    /// General error.
    Other(String),
}

impl std::fmt::Display for DeltaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeltaError::Io(e) => write!(f, "Delta I/O error: {e}"),
            DeltaError::Parse(msg) => write!(f, "Delta parse error: {msg}"),
            DeltaError::Serialization(msg) => write!(f, "Delta serialization error: {msg}"),
            DeltaError::VersionConflict { expected, actual } => {
                write!(
                    f,
                    "Delta version conflict: expected {expected}, found {actual}"
                )
            }
            DeltaError::TableNotFound(path) => write!(f, "Delta table not found: {path}"),
            DeltaError::SchemaError(msg) => write!(f, "Delta schema error: {msg}"),
            DeltaError::Other(msg) => write!(f, "Delta error: {msg}"),
        }
    }
}

impl std::error::Error for DeltaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DeltaError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for DeltaError {
    fn from(e: std::io::Error) -> Self {
        DeltaError::Io(e)
    }
}

impl From<DeltaError> for crate::error::IoError {
    fn from(e: DeltaError) -> Self {
        crate::error::IoError::Other(format!("{e}"))
    }
}
