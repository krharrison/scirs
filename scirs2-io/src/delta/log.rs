//! Delta Lake transaction log.
//!
//! Implements an append-only transaction log with JSON commit files,
//! checkpoint compaction, version conflict detection, and log replay.

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::types::{DeltaAction, DeltaError, DeltaVersion, FileInfo};

/// Append-only transaction log for a Delta table.
///
/// The log is stored as numbered JSON files under `<base_path>/_delta_log/`.
/// Each file contains one JSON object per line, representing a single action.
/// Checkpoints compact the log into a single snapshot file.
#[derive(Debug)]
pub struct TransactionLog {
    /// Root path of the Delta table.
    base_path: PathBuf,
    /// Path to the `_delta_log` directory.
    log_dir: PathBuf,
}

impl TransactionLog {
    /// Create a new transaction log rooted at `base_path`.
    ///
    /// Creates the `_delta_log/` directory if it does not exist.
    pub fn new(base_path: &Path) -> Result<Self, DeltaError> {
        let log_dir = base_path.join("_delta_log");
        fs::create_dir_all(&log_dir)?;
        Ok(Self {
            base_path: base_path.to_path_buf(),
            log_dir,
        })
    }

    /// Return the base path of this table.
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Return the log directory path.
    pub fn log_dir(&self) -> &Path {
        &self.log_dir
    }

    /// Commit a set of actions as a new version.
    ///
    /// Writes a JSON log entry and returns the committed version number.
    /// Uses optimistic concurrency: if the target version file already exists,
    /// returns a `VersionConflict` error.
    pub fn commit(&self, actions: Vec<DeltaAction>) -> Result<u64, DeltaError> {
        let next_version = self.latest_version().map(|v| v + 1).unwrap_or(0);

        let file_path = self.version_file_path(next_version);

        // Optimistic concurrency: check the file does not already exist.
        if file_path.exists() {
            return Err(DeltaError::VersionConflict {
                expected: next_version,
                actual: self.latest_version().unwrap_or(0),
            });
        }

        let mut file = fs::File::create(&file_path)?;
        for action in &actions {
            let json_line = serialize_action(action)?;
            writeln!(file, "{json_line}").map_err(|e| {
                DeltaError::Io(std::io::Error::new(
                    e.kind(),
                    format!("write to {}: {e}", file_path.display()),
                ))
            })?;
        }
        file.flush()?;

        Ok(next_version)
    }

    /// Read a specific version from the log.
    pub fn read_version(&self, version: u64) -> Result<DeltaVersion, DeltaError> {
        let file_path = self.version_file_path(version);
        if !file_path.exists() {
            return Err(DeltaError::TableNotFound(format!(
                "version {version} not found at {}",
                file_path.display()
            )));
        }

        let file = fs::File::open(&file_path)?;
        let reader = BufReader::new(file);

        let mut actions = Vec::new();
        let mut timestamp: u64 = 0;

        for (lineno, line_res) in reader.lines().enumerate() {
            let line = line_res.map_err(|e| {
                DeltaError::Io(std::io::Error::new(
                    e.kind(),
                    format!("read error at line {lineno}: {e}"),
                ))
            })?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let action = parse_action_line(trimmed)?;
            if let DeltaAction::CommitInfo { timestamp: ts, .. } = &action {
                timestamp = *ts as u64;
            }
            actions.push(action);
        }

        Ok(DeltaVersion {
            version,
            timestamp,
            actions,
        })
    }

    /// Scan the log directory and return the latest (highest) version number.
    ///
    /// Returns `None` if the log is empty.
    pub fn latest_version(&self) -> Option<u64> {
        self.list_version_files()
            .ok()
            .and_then(|files| files.last().map(|(v, _)| *v))
    }

    /// Write a checkpoint file that compacts all actions up to `version`
    /// into a single snapshot file.
    ///
    /// The checkpoint is stored as `<version:020>.checkpoint.json` in the
    /// log directory.
    pub fn checkpoint(&self, version: u64) -> Result<(), DeltaError> {
        let all_actions = self.replay_up_to(version)?;

        let checkpoint_path = self.log_dir.join(format!("{version:020}.checkpoint.json"));

        let mut file = fs::File::create(&checkpoint_path)?;
        for action in &all_actions {
            let json_line = serialize_action(action)?;
            writeln!(file, "{json_line}").map_err(|e| {
                DeltaError::Io(std::io::Error::new(
                    e.kind(),
                    format!("write checkpoint: {e}"),
                ))
            })?;
        }
        file.flush()?;

        Ok(())
    }

    /// Replay all actions from version 0 to the latest version,
    /// returning the final consolidated set of actions that represent
    /// the current table state.
    ///
    /// Add/Remove pairs are resolved: a removed file is eliminated from
    /// the result. Only the latest Metadata and Protocol actions are kept.
    pub fn replay(&self) -> Result<Vec<DeltaAction>, DeltaError> {
        let latest = self.latest_version().unwrap_or(0);
        self.replay_up_to(latest)
    }

    /// Replay all actions from version 0 up to (and including) `up_to_version`.
    pub fn replay_up_to(&self, up_to_version: u64) -> Result<Vec<DeltaAction>, DeltaError> {
        let version_files = self.list_version_files()?;

        let mut active_files: HashMap<String, DeltaAction> = HashMap::new();
        let mut latest_metadata: Option<DeltaAction> = None;
        let mut latest_protocol: Option<DeltaAction> = None;

        for (ver, path) in &version_files {
            if *ver > up_to_version {
                break;
            }
            let delta_ver = self.read_version_from_path(path)?;
            for action in delta_ver.actions {
                match &action {
                    DeltaAction::Add { path: fpath, .. } => {
                        active_files.insert(fpath.clone(), action);
                    }
                    DeltaAction::Remove { path: fpath, .. } => {
                        active_files.remove(fpath);
                    }
                    DeltaAction::Metadata { .. } => {
                        latest_metadata = Some(action);
                    }
                    DeltaAction::Protocol { .. } => {
                        latest_protocol = Some(action);
                    }
                    DeltaAction::CommitInfo { .. } => {
                        // CommitInfo is transient, not replayed into state
                    }
                }
            }
        }

        let mut result: Vec<DeltaAction> = Vec::new();
        if let Some(proto) = latest_protocol {
            result.push(proto);
        }
        if let Some(meta) = latest_metadata {
            result.push(meta);
        }
        let mut file_actions: Vec<DeltaAction> = active_files.into_values().collect();
        file_actions.sort_by(|a, b| {
            let path_a = match a {
                DeltaAction::Add { path, .. } => path.as_str(),
                _ => "",
            };
            let path_b = match b {
                DeltaAction::Add { path, .. } => path.as_str(),
                _ => "",
            };
            path_a.cmp(path_b)
        });
        result.extend(file_actions);

        Ok(result)
    }

    /// List all version commit files, sorted by version number.
    fn list_version_files(&self) -> Result<Vec<(u64, PathBuf)>, DeltaError> {
        if !self.log_dir.exists() {
            return Ok(Vec::new());
        }

        let mut files: Vec<(u64, PathBuf)> = Vec::new();
        for entry_res in fs::read_dir(&self.log_dir)? {
            let entry = entry_res?;
            let fname = entry.file_name();
            let fname_str = fname.to_string_lossy();
            if fname_str.ends_with(".json") && !fname_str.contains("checkpoint") {
                let stem = fname_str.trim_end_matches(".json").trim_start_matches('0');
                let ver: u64 = if stem.is_empty() {
                    0
                } else {
                    stem.parse().unwrap_or(0)
                };
                files.push((ver, entry.path()));
            }
        }
        files.sort_by_key(|(v, _)| *v);
        Ok(files)
    }

    /// Read a version from a specific file path.
    fn read_version_from_path(&self, path: &Path) -> Result<DeltaVersion, DeltaError> {
        let fname = path.file_stem().and_then(|s| s.to_str()).unwrap_or("0");
        let stem = fname.trim_start_matches('0');
        let version: u64 = if stem.is_empty() {
            0
        } else {
            stem.parse().unwrap_or(0)
        };

        let file = fs::File::open(path)?;
        let reader = BufReader::new(file);

        let mut actions = Vec::new();
        let mut timestamp: u64 = 0;

        for line_res in reader.lines() {
            let line = line_res?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let action = parse_action_line(trimmed)?;
            if let DeltaAction::CommitInfo { timestamp: ts, .. } = &action {
                timestamp = *ts as u64;
            }
            actions.push(action);
        }

        Ok(DeltaVersion {
            version,
            timestamp,
            actions,
        })
    }

    /// Return the path for a version's commit file.
    fn version_file_path(&self, version: u64) -> PathBuf {
        self.log_dir.join(format!("{version:020}.json"))
    }

    /// Reconstruct the table's active file set by replaying up to `version`.
    pub fn reconstruct_files(
        &self,
        version: Option<u64>,
    ) -> Result<HashMap<String, FileInfo>, DeltaError> {
        let up_to = version.or_else(|| self.latest_version()).unwrap_or(0);
        let actions = self.replay_up_to(up_to)?;

        let mut files = HashMap::new();
        for action in actions {
            if let DeltaAction::Add {
                path,
                size,
                modification_time,
                partition_values,
                ..
            } = action
            {
                files.insert(
                    path.clone(),
                    FileInfo {
                        path,
                        size,
                        modification_time,
                        partition_values,
                    },
                );
            }
        }
        Ok(files)
    }

    /// List all versions as `DeltaVersion` structs.
    pub fn history(&self) -> Result<Vec<DeltaVersion>, DeltaError> {
        let version_files = self.list_version_files()?;
        let mut versions = Vec::with_capacity(version_files.len());
        for (_, path) in &version_files {
            versions.push(self.read_version_from_path(path)?);
        }
        Ok(versions)
    }
}

// ─── Action serialization / parsing ─────────────────────────────────────────

/// Serialize a `DeltaAction` to a single-line JSON string.
pub fn serialize_action(action: &DeltaAction) -> Result<String, DeltaError> {
    let value = match action {
        DeltaAction::Add {
            path,
            size,
            modification_time,
            data_change,
            partition_values,
            stats_json,
        } => {
            let mut add = serde_json::Map::new();
            add.insert("path".to_string(), Value::String(path.clone()));
            add.insert("size".to_string(), Value::Number((*size).into()));
            add.insert(
                "modificationTime".to_string(),
                Value::Number((*modification_time).into()),
            );
            add.insert("dataChange".to_string(), Value::Bool(*data_change));
            if !partition_values.is_empty() {
                let pv: serde_json::Map<String, Value> = partition_values
                    .iter()
                    .map(|(k, v)| (k.clone(), Value::String(v.clone())))
                    .collect();
                add.insert("partitionValues".to_string(), Value::Object(pv));
            }
            if let Some(sj) = stats_json {
                add.insert("stats".to_string(), Value::String(sj.clone()));
            }
            let mut obj = serde_json::Map::new();
            obj.insert("add".to_string(), Value::Object(add));
            Value::Object(obj)
        }
        DeltaAction::Remove {
            path,
            deletion_timestamp,
            data_change,
        } => {
            let mut rm = serde_json::Map::new();
            rm.insert("path".to_string(), Value::String(path.clone()));
            rm.insert(
                "deletionTimestamp".to_string(),
                Value::Number((*deletion_timestamp).into()),
            );
            rm.insert("dataChange".to_string(), Value::Bool(*data_change));
            let mut obj = serde_json::Map::new();
            obj.insert("remove".to_string(), Value::Object(rm));
            Value::Object(obj)
        }
        DeltaAction::Metadata {
            schema,
            partition_columns,
            description,
            configuration,
        } => {
            let mut md = serde_json::Map::new();
            md.insert("schema".to_string(), Value::String(schema.clone()));
            let pc: Vec<Value> = partition_columns
                .iter()
                .map(|c| Value::String(c.clone()))
                .collect();
            md.insert("partitionColumns".to_string(), Value::Array(pc));
            if let Some(desc) = description {
                md.insert("description".to_string(), Value::String(desc.clone()));
            }
            if !configuration.is_empty() {
                let cfg: serde_json::Map<String, Value> = configuration
                    .iter()
                    .map(|(k, v)| (k.clone(), Value::String(v.clone())))
                    .collect();
                md.insert("configuration".to_string(), Value::Object(cfg));
            }
            let mut obj = serde_json::Map::new();
            obj.insert("metaData".to_string(), Value::Object(md));
            Value::Object(obj)
        }
        DeltaAction::Protocol {
            min_reader_version,
            min_writer_version,
        } => {
            let mut proto = serde_json::Map::new();
            proto.insert(
                "minReaderVersion".to_string(),
                Value::Number((*min_reader_version).into()),
            );
            proto.insert(
                "minWriterVersion".to_string(),
                Value::Number((*min_writer_version).into()),
            );
            let mut obj = serde_json::Map::new();
            obj.insert("protocol".to_string(), Value::Object(proto));
            Value::Object(obj)
        }
        DeltaAction::CommitInfo {
            timestamp,
            operation,
            operation_parameters,
        } => {
            let mut ci = serde_json::Map::new();
            ci.insert("timestamp".to_string(), Value::Number((*timestamp).into()));
            ci.insert("operation".to_string(), Value::String(operation.clone()));
            if !operation_parameters.is_empty() {
                let params: serde_json::Map<String, Value> = operation_parameters
                    .iter()
                    .map(|(k, v)| (k.clone(), Value::String(v.clone())))
                    .collect();
                ci.insert("operationParameters".to_string(), Value::Object(params));
            }
            let mut obj = serde_json::Map::new();
            obj.insert("commitInfo".to_string(), Value::Object(ci));
            Value::Object(obj)
        }
    };

    serde_json::to_string(&value)
        .map_err(|e| DeltaError::Serialization(format!("serialize action: {e}")))
}

/// Parse a single JSON line from a Delta log into a `DeltaAction`.
pub fn parse_action_line(line: &str) -> Result<DeltaAction, DeltaError> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Err(DeltaError::Parse("empty line".to_string()));
    }

    let obj: Value =
        serde_json::from_str(trimmed).map_err(|e| DeltaError::Parse(format!("JSON parse: {e}")))?;
    let map = obj
        .as_object()
        .ok_or_else(|| DeltaError::Parse("not a JSON object".to_string()))?;

    if let Some(add) = map.get("add") {
        let path = add
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let size = add.get("size").and_then(|v| v.as_u64()).unwrap_or(0);
        let modification_time = add
            .get("modificationTime")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let data_change = add
            .get("dataChange")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let partition_values = add
            .get("partitionValues")
            .and_then(|v| v.as_object())
            .map(|pv| {
                pv.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();
        let stats_json = add.get("stats").map(|v| v.to_string());
        return Ok(DeltaAction::Add {
            path,
            size,
            modification_time,
            data_change,
            partition_values,
            stats_json,
        });
    }

    if let Some(remove) = map.get("remove") {
        let path = remove
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let deletion_timestamp = remove
            .get("deletionTimestamp")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let data_change = remove
            .get("dataChange")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        return Ok(DeltaAction::Remove {
            path,
            deletion_timestamp,
            data_change,
        });
    }

    if let Some(md) = map.get("metaData") {
        let schema = md
            .get("schema")
            .and_then(|v| v.as_str())
            .unwrap_or("{}")
            .to_string();
        let partition_columns = md
            .get("partitionColumns")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let description = md
            .get("description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let configuration = md
            .get("configuration")
            .and_then(|v| v.as_object())
            .map(|cfg| {
                cfg.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();
        return Ok(DeltaAction::Metadata {
            schema,
            partition_columns,
            description,
            configuration,
        });
    }

    if let Some(proto) = map.get("protocol") {
        let min_reader_version = proto
            .get("minReaderVersion")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as u32;
        let min_writer_version = proto
            .get("minWriterVersion")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as u32;
        return Ok(DeltaAction::Protocol {
            min_reader_version,
            min_writer_version,
        });
    }

    if let Some(ci) = map.get("commitInfo") {
        let timestamp = ci.get("timestamp").and_then(|v| v.as_i64()).unwrap_or(0);
        let operation = ci
            .get("operation")
            .and_then(|v| v.as_str())
            .unwrap_or("UNKNOWN")
            .to_string();
        let operation_parameters = ci
            .get("operationParameters")
            .and_then(|v| v.as_object())
            .map(|params| {
                params
                    .iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();
        return Ok(DeltaAction::CommitInfo {
            timestamp,
            operation,
            operation_parameters,
        });
    }

    // Unknown action type - return as commit info with UNKNOWN operation
    Ok(DeltaAction::CommitInfo {
        timestamp: 0,
        operation: format!("UNKNOWN(keys={:?})", map.keys().collect::<Vec<_>>()),
        operation_parameters: HashMap::new(),
    })
}
