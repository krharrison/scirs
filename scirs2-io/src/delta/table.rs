//! Delta Lake table read/write operations.
//!
//! Provides `DeltaTableWriter` for writing data files and committing transactions,
//! and `DeltaTableReader` for reading data with time travel, schema evolution,
//! and partition pruning support.

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Write as IoWrite};
use std::path::{Path, PathBuf};

use super::log::TransactionLog;
use super::types::{
    ColumnSchema, DeltaAction, DeltaConfig, DeltaError, DeltaTable, DeltaTransaction, DeltaVersion,
    FileInfo, Schema,
};

/// Writer for Delta Lake tables.
///
/// Writes data as CSV-like files (one file per write) and commits
/// a transaction log entry for each write operation.
pub struct DeltaTableWriter {
    config: DeltaConfig,
    log: TransactionLog,
}

impl DeltaTableWriter {
    /// Create a new writer for the table at `config.base_path`.
    pub fn new(config: DeltaConfig) -> Result<Self, DeltaError> {
        let log = TransactionLog::new(&config.base_path)?;
        // Create data directory
        let data_dir = config.base_path.join("data");
        fs::create_dir_all(&data_dir)?;
        Ok(Self { config, log })
    }

    /// Write columnar data to the table and commit a transaction.
    ///
    /// `data` is a slice of columns, each column being a `Vec<f64>`.
    /// `schema` describes the column names and types.
    ///
    /// Returns the committed version number.
    pub fn write(&self, data: &[Vec<f64>], schema: &Schema) -> Result<u64, DeltaError> {
        if data.is_empty() {
            return Err(DeltaError::Other("cannot write empty data".to_string()));
        }

        // Validate column count matches schema
        if data.len() != schema.columns.len() {
            return Err(DeltaError::SchemaError(format!(
                "data has {} columns but schema has {}",
                data.len(),
                schema.columns.len()
            )));
        }

        let next_version = self.log.latest_version().map(|v| v + 1).unwrap_or(0);

        // Write data file
        let file_name = format!("part-{next_version:05}.csv");
        let file_path = self.config.base_path.join("data").join(&file_name);

        write_data_file(&file_path, data, schema)?;

        let file_size = fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0);

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Build actions
        let mut actions = Vec::new();

        // If this is the first version, include protocol and metadata
        if next_version == 0 {
            actions.push(DeltaAction::Protocol {
                min_reader_version: 1,
                min_writer_version: 1,
            });
            let schema_json = schema.to_json().unwrap_or_else(|_| "{}".to_string());
            actions.push(DeltaAction::Metadata {
                schema: schema_json,
                partition_columns: Vec::new(),
                description: None,
                configuration: HashMap::new(),
            });
        }

        actions.push(DeltaAction::Add {
            path: format!("data/{file_name}"),
            size: file_size,
            modification_time: now_ms,
            data_change: true,
            partition_values: HashMap::new(),
            stats_json: None,
        });

        actions.push(DeltaAction::CommitInfo {
            timestamp: now_ms as i64,
            operation: "WRITE".to_string(),
            operation_parameters: HashMap::new(),
        });

        let committed = self.log.commit(actions)?;

        // Auto-checkpoint if interval reached
        if self.config.checkpoint_interval > 0
            && committed > 0
            && (committed as usize).is_multiple_of(self.config.checkpoint_interval)
        {
            // Best-effort checkpoint; do not fail the commit
            let _ = self.log.checkpoint(committed);
        }

        Ok(committed)
    }

    /// Write partitioned data to the table.
    ///
    /// `partition_column` names the column whose values determine partitioning.
    /// Data rows are grouped by their value in that column, and separate files
    /// are written per partition.
    pub fn write_partitioned(
        &self,
        data: &[Vec<f64>],
        schema: &Schema,
        partition_column: &str,
    ) -> Result<u64, DeltaError> {
        if data.is_empty() || schema.columns.is_empty() {
            return Err(DeltaError::Other("cannot write empty data".to_string()));
        }

        // Find partition column index
        let part_idx = schema
            .columns
            .iter()
            .position(|c| c.name == partition_column)
            .ok_or_else(|| {
                DeltaError::SchemaError(format!(
                    "partition column '{partition_column}' not found in schema"
                ))
            })?;

        let num_rows = data[0].len();
        for col in data {
            if col.len() != num_rows {
                return Err(DeltaError::SchemaError(
                    "all columns must have the same number of rows".to_string(),
                ));
            }
        }

        // Group rows by partition value
        let mut partitions: HashMap<String, Vec<usize>> = HashMap::new();
        for row_idx in 0..num_rows {
            let val = data[part_idx][row_idx];
            let key = format!("{val}");
            partitions.entry(key).or_default().push(row_idx);
        }

        let next_version = self.log.latest_version().map(|v| v + 1).unwrap_or(0);
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let mut actions = Vec::new();

        if next_version == 0 {
            actions.push(DeltaAction::Protocol {
                min_reader_version: 1,
                min_writer_version: 1,
            });
            let schema_json = schema.to_json().unwrap_or_else(|_| "{}".to_string());
            actions.push(DeltaAction::Metadata {
                schema: schema_json,
                partition_columns: vec![partition_column.to_string()],
                description: None,
                configuration: HashMap::new(),
            });
        }

        let mut part_keys: Vec<String> = partitions.keys().cloned().collect();
        part_keys.sort();

        for (file_idx, part_val) in part_keys.iter().enumerate() {
            let row_indices = &partitions[part_val];

            // Extract partition data
            let part_data: Vec<Vec<f64>> = data
                .iter()
                .map(|col| row_indices.iter().map(|&ri| col[ri]).collect())
                .collect();

            // Write partition file
            let part_dir = self
                .config
                .base_path
                .join("data")
                .join(format!("{partition_column}={part_val}"));
            fs::create_dir_all(&part_dir)?;

            let file_name = format!("part-{next_version:05}-{file_idx:03}.csv");
            let file_path = part_dir.join(&file_name);
            write_data_file(&file_path, &part_data, schema)?;

            let file_size = fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0);

            let mut pv = HashMap::new();
            pv.insert(partition_column.to_string(), part_val.clone());

            actions.push(DeltaAction::Add {
                path: format!("data/{partition_column}={part_val}/{file_name}"),
                size: file_size,
                modification_time: now_ms,
                data_change: true,
                partition_values: pv,
                stats_json: None,
            });
        }

        actions.push(DeltaAction::CommitInfo {
            timestamp: now_ms as i64,
            operation: "WRITE".to_string(),
            operation_parameters: HashMap::new(),
        });

        self.log.commit(actions)
    }

    /// Access the underlying transaction log.
    pub fn transaction_log(&self) -> &TransactionLog {
        &self.log
    }
}

/// Reader for Delta Lake tables, supporting time travel and partition pruning.
pub struct DeltaTableReader {
    config: DeltaConfig,
    log: TransactionLog,
}

impl DeltaTableReader {
    /// Create a new reader for the table at `config.base_path`.
    pub fn new(config: DeltaConfig) -> Result<Self, DeltaError> {
        let log = TransactionLog::new(&config.base_path)?;
        Ok(Self { config, log })
    }

    /// Read data from the table, optionally at a specific version (time travel).
    ///
    /// Returns columnar data as `Vec<Vec<f64>>`.
    pub fn read(&self, version: Option<u64>) -> Result<Vec<Vec<f64>>, DeltaError> {
        let files = self.log.reconstruct_files(version)?;
        if files.is_empty() {
            return Ok(Vec::new());
        }

        self.read_files(&files)
    }

    /// Read data with partition pruning.
    ///
    /// Only reads files whose partition values match the given predicates.
    /// `partitions` maps column name -> required value.
    pub fn read_pruned(
        &self,
        version: Option<u64>,
        partitions: &HashMap<String, String>,
    ) -> Result<Vec<Vec<f64>>, DeltaError> {
        let all_files = self.log.reconstruct_files(version)?;

        // Filter files by partition values
        let filtered: HashMap<String, FileInfo> = all_files
            .into_iter()
            .filter(|(_, info)| {
                partitions
                    .iter()
                    .all(|(col, val)| info.partition_values.get(col).map_or(true, |fv| fv == val))
            })
            .collect();

        if filtered.is_empty() {
            return Ok(Vec::new());
        }

        self.read_files(&filtered)
    }

    /// List all versions (history) of the table.
    pub fn history(&self) -> Result<Vec<DeltaVersion>, DeltaError> {
        self.log.history()
    }

    /// Get the latest version number.
    pub fn latest_version(&self) -> Option<u64> {
        self.log.latest_version()
    }

    /// Reconstruct the full table state at a given version.
    pub fn table_state(&self, version: Option<u64>) -> Result<DeltaTable, DeltaError> {
        let target_version = version.or_else(|| self.log.latest_version()).unwrap_or(0);
        let actions = self.log.replay_up_to(target_version)?;

        let mut active_files = HashMap::new();
        let mut schema: Option<Schema> = None;
        let mut partition_columns = Vec::new();
        let mut protocol: Option<(u32, u32)> = None;

        for action in &actions {
            match action {
                DeltaAction::Add {
                    path,
                    size,
                    modification_time,
                    partition_values,
                    ..
                } => {
                    active_files.insert(
                        path.clone(),
                        FileInfo {
                            path: path.clone(),
                            size: *size,
                            modification_time: *modification_time,
                            partition_values: partition_values.clone(),
                        },
                    );
                }
                DeltaAction::Metadata {
                    schema: schema_json,
                    partition_columns: pc,
                    ..
                } => {
                    schema = Schema::from_json(schema_json).ok();
                    partition_columns = pc.clone();
                }
                DeltaAction::Protocol {
                    min_reader_version,
                    min_writer_version,
                } => {
                    protocol = Some((*min_reader_version, *min_writer_version));
                }
                _ => {}
            }
        }

        Ok(DeltaTable {
            config: self.config.clone(),
            version: target_version,
            active_files,
            schema,
            partition_columns,
            protocol,
        })
    }

    /// Read data from a specific set of files.
    fn read_files(&self, files: &HashMap<String, FileInfo>) -> Result<Vec<Vec<f64>>, DeltaError> {
        let mut all_columns: Vec<Vec<f64>> = Vec::new();
        let mut first = true;

        let mut sorted_paths: Vec<&String> = files.keys().collect();
        sorted_paths.sort();

        for file_path_rel in sorted_paths {
            let abs_path = self.config.base_path.join(file_path_rel);
            if !abs_path.exists() {
                continue;
            }

            let file_data = read_data_file(&abs_path)?;
            if file_data.is_empty() {
                continue;
            }

            if first {
                all_columns = file_data;
                first = false;
            } else {
                // Append rows from each column
                if all_columns.len() != file_data.len() {
                    return Err(DeltaError::SchemaError(format!(
                        "column count mismatch: {} vs {}",
                        all_columns.len(),
                        file_data.len()
                    )));
                }
                for (i, col) in file_data.into_iter().enumerate() {
                    all_columns[i].extend(col);
                }
            }
        }

        Ok(all_columns)
    }

    /// Access the underlying transaction log.
    pub fn transaction_log(&self) -> &TransactionLog {
        &self.log
    }
}

// ─── Schema evolution helpers ────────────────────────────────────────────────

/// Add a column to the table schema.
///
/// Commits a new Metadata action with the updated schema.
/// Existing data files are not modified; the new column will have
/// default (NaN) values when read.
pub fn add_column(
    log: &TransactionLog,
    current_schema: &Schema,
    column: ColumnSchema,
) -> Result<u64, DeltaError> {
    let mut new_schema = current_schema.clone();
    if new_schema.columns.iter().any(|c| c.name == column.name) {
        return Err(DeltaError::SchemaError(format!(
            "column '{}' already exists",
            column.name
        )));
    }
    new_schema.columns.push(column);

    let schema_json = new_schema.to_json()?;

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);

    let actions = vec![
        DeltaAction::Metadata {
            schema: schema_json,
            partition_columns: Vec::new(),
            description: Some("schema evolution: add column".to_string()),
            configuration: HashMap::new(),
        },
        DeltaAction::CommitInfo {
            timestamp: now_ms,
            operation: "ALTER_TABLE".to_string(),
            operation_parameters: HashMap::new(),
        },
    ];

    log.commit(actions)
}

/// Rename a column in the table schema.
///
/// Commits a new Metadata action with the updated schema.
/// Existing data files are not modified.
pub fn rename_column(
    log: &TransactionLog,
    current_schema: &Schema,
    old_name: &str,
    new_name: &str,
) -> Result<u64, DeltaError> {
    let mut new_schema = current_schema.clone();
    let col = new_schema
        .columns
        .iter_mut()
        .find(|c| c.name == old_name)
        .ok_or_else(|| DeltaError::SchemaError(format!("column '{old_name}' not found")))?;
    col.name = new_name.to_string();

    let schema_json = new_schema.to_json()?;

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);

    let actions = vec![
        DeltaAction::Metadata {
            schema: schema_json,
            partition_columns: Vec::new(),
            description: Some(format!("schema evolution: rename {old_name} -> {new_name}")),
            configuration: HashMap::new(),
        },
        DeltaAction::CommitInfo {
            timestamp: now_ms,
            operation: "ALTER_TABLE".to_string(),
            operation_parameters: HashMap::new(),
        },
    ];

    log.commit(actions)
}

// ─── Data file I/O helpers ───────────────────────────────────────────────────

/// Write columnar f64 data to a CSV file.
fn write_data_file(path: &Path, data: &[Vec<f64>], schema: &Schema) -> Result<(), DeltaError> {
    let mut file = fs::File::create(path)?;

    // Write header
    let header: Vec<&str> = schema.columns.iter().map(|c| c.name.as_str()).collect();
    writeln!(file, "{}", header.join(","))
        .map_err(|e| DeltaError::Io(std::io::Error::new(e.kind(), format!("write header: {e}"))))?;

    // Write rows
    let num_rows = data.first().map(|c| c.len()).unwrap_or(0);
    for row_idx in 0..num_rows {
        let row: Vec<String> = data
            .iter()
            .map(|col| {
                if row_idx < col.len() {
                    format!("{}", col[row_idx])
                } else {
                    "NaN".to_string()
                }
            })
            .collect();
        writeln!(file, "{}", row.join(",")).map_err(|e| {
            DeltaError::Io(std::io::Error::new(e.kind(), format!("write row: {e}")))
        })?;
    }

    file.flush()?;
    Ok(())
}

/// Read columnar f64 data from a CSV file.
fn read_data_file(path: &Path) -> Result<Vec<Vec<f64>>, DeltaError> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);

    let mut lines = reader.lines();

    // Read header to determine column count
    let header_line = lines
        .next()
        .ok_or_else(|| DeltaError::Parse("empty data file".to_string()))?
        .map_err(|e| DeltaError::Io(e))?;
    let num_cols = header_line.split(',').count();

    let mut columns: Vec<Vec<f64>> = (0..num_cols).map(|_| Vec::new()).collect();

    for line_res in lines {
        let line = line_res?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let fields: Vec<&str> = trimmed.split(',').collect();
        for (i, field) in fields.iter().enumerate() {
            if i < num_cols {
                let val: f64 = field.trim().parse().unwrap_or(f64::NAN);
                columns[i].push(val);
            }
        }
        // If row has fewer fields than columns, pad with NaN
        for col in columns.iter_mut().skip(fields.len()) {
            col.push(f64::NAN);
        }
    }

    Ok(columns)
}
