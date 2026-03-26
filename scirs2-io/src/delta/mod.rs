//! Delta Lake log-based table format reader and writer.
//!
//! Delta Lake is an open-source storage layer that brings ACID transactions to
//! data workloads. It stores data files plus a JSON-based transaction log in
//! `_delta_log/`.
//!
//! This module implements:
//! - Core types: `DeltaConfig`, `DeltaAction`, `DeltaVersion`, `DeltaTable`,
//!   `DeltaTransaction`, `DeltaError`
//! - Transaction log: append-only commit, version conflict detection, checkpoint,
//!   replay
//! - Table operations: write data, read with time travel, schema evolution,
//!   partition pruning

pub mod log;
pub mod table;
pub mod types;

// Re-export key types at the delta module level
pub use log::{parse_action_line, serialize_action, TransactionLog};
pub use table::{add_column, rename_column, DeltaTableReader, DeltaTableWriter};
pub use types::{
    ColumnSchema, DeltaAction, DeltaConfig, DeltaError, DeltaTable, DeltaTransaction, DeltaVersion,
    FileInfo, Schema,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn temp_table_path() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let path = dir.path().to_path_buf();
        (dir, path)
    }

    fn sample_schema() -> Schema {
        Schema::new(vec![
            ColumnSchema {
                name: "x".to_string(),
                data_type: "double".to_string(),
                nullable: false,
            },
            ColumnSchema {
                name: "y".to_string(),
                data_type: "double".to_string(),
                nullable: false,
            },
        ])
    }

    // ─── Transaction log tests ───────────────────────────────────────────

    #[test]
    fn test_commit_and_read_version() {
        let (_dir, path) = temp_table_path();
        let log = TransactionLog::new(&path).expect("create log");

        let actions = vec![
            DeltaAction::Protocol {
                min_reader_version: 1,
                min_writer_version: 1,
            },
            DeltaAction::Add {
                path: "data/part-0.csv".to_string(),
                size: 100,
                modification_time: 1000,
                data_change: true,
                partition_values: HashMap::new(),
                stats_json: None,
            },
        ];

        let ver = log.commit(actions).expect("commit");
        assert_eq!(ver, 0);

        let dv = log.read_version(0).expect("read v0");
        assert_eq!(dv.version, 0);
        assert_eq!(dv.actions.len(), 2);
    }

    #[test]
    fn test_latest_version() {
        let (_dir, path) = temp_table_path();
        let log = TransactionLog::new(&path).expect("create log");

        assert_eq!(log.latest_version(), None);

        log.commit(vec![DeltaAction::Add {
            path: "f1".to_string(),
            size: 1,
            modification_time: 0,
            data_change: true,
            partition_values: HashMap::new(),
            stats_json: None,
        }])
        .expect("c0");

        assert_eq!(log.latest_version(), Some(0));

        log.commit(vec![DeltaAction::Add {
            path: "f2".to_string(),
            size: 2,
            modification_time: 0,
            data_change: true,
            partition_values: HashMap::new(),
            stats_json: None,
        }])
        .expect("c1");

        assert_eq!(log.latest_version(), Some(1));
    }

    #[test]
    fn test_replay_add_remove() {
        let (_dir, path) = temp_table_path();
        let log = TransactionLog::new(&path).expect("create log");

        log.commit(vec![
            DeltaAction::Add {
                path: "f1".to_string(),
                size: 10,
                modification_time: 0,
                data_change: true,
                partition_values: HashMap::new(),
                stats_json: None,
            },
            DeltaAction::Add {
                path: "f2".to_string(),
                size: 20,
                modification_time: 0,
                data_change: true,
                partition_values: HashMap::new(),
                stats_json: None,
            },
        ])
        .expect("c0");

        log.commit(vec![DeltaAction::Remove {
            path: "f1".to_string(),
            deletion_timestamp: 100,
            data_change: true,
        }])
        .expect("c1");

        let replayed = log.replay().expect("replay");
        let add_paths: Vec<&str> = replayed
            .iter()
            .filter_map(|a| match a {
                DeltaAction::Add { path, .. } => Some(path.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(add_paths, vec!["f2"]);
    }

    #[test]
    fn test_checkpoint_and_replay_equivalence() {
        let (_dir, path) = temp_table_path();
        let log = TransactionLog::new(&path).expect("create log");

        // Write multiple versions
        for i in 0..5u64 {
            log.commit(vec![DeltaAction::Add {
                path: format!("file-{i}.csv"),
                size: i * 10,
                modification_time: i * 1000,
                data_change: true,
                partition_values: HashMap::new(),
                stats_json: None,
            }])
            .expect("commit");
        }

        // Remove file-2
        log.commit(vec![DeltaAction::Remove {
            path: "file-2.csv".to_string(),
            deletion_timestamp: 9000,
            data_change: true,
        }])
        .expect("remove");

        // Get replay result before checkpoint
        let before = log.replay().expect("replay before");

        // Write checkpoint
        log.checkpoint(log.latest_version().expect("latest"))
            .expect("checkpoint");

        // Replay again — result should be identical
        let after = log.replay().expect("replay after");
        assert_eq!(before.len(), after.len());

        let before_paths: Vec<String> = before
            .iter()
            .filter_map(|a| match a {
                DeltaAction::Add { path, .. } => Some(path.clone()),
                _ => None,
            })
            .collect();
        let after_paths: Vec<String> = after
            .iter()
            .filter_map(|a| match a {
                DeltaAction::Add { path, .. } => Some(path.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(before_paths, after_paths);
    }

    #[test]
    fn test_version_conflict_detection() {
        let (_dir, path) = temp_table_path();
        let log = TransactionLog::new(&path).expect("create log");

        log.commit(vec![DeltaAction::Add {
            path: "f1".to_string(),
            size: 1,
            modification_time: 0,
            data_change: true,
            partition_values: HashMap::new(),
            stats_json: None,
        }])
        .expect("c0");

        // Manually create version 1 to simulate concurrent writer
        let v1_path = path.join("_delta_log").join("00000000000000000001.json");
        std::fs::write(
            &v1_path,
            r#"{"add":{"path":"conflict.csv","size":1,"modificationTime":0,"dataChange":true}}"#,
        )
        .expect("write conflict file");

        // Now try to commit — should get version 2, not conflict
        let result = log.commit(vec![DeltaAction::Add {
            path: "f2".to_string(),
            size: 2,
            modification_time: 0,
            data_change: true,
            partition_values: HashMap::new(),
            stats_json: None,
        }]);

        assert!(result.is_ok());
        assert_eq!(result.expect("commit"), 2);
    }

    #[test]
    fn test_history() {
        let (_dir, path) = temp_table_path();
        let log = TransactionLog::new(&path).expect("create log");

        for i in 0..3u64 {
            log.commit(vec![
                DeltaAction::Add {
                    path: format!("f-{i}.csv"),
                    size: i,
                    modification_time: i * 100,
                    data_change: true,
                    partition_values: HashMap::new(),
                    stats_json: None,
                },
                DeltaAction::CommitInfo {
                    timestamp: (i * 100) as i64,
                    operation: "WRITE".to_string(),
                    operation_parameters: HashMap::new(),
                },
            ])
            .expect("commit");
        }

        let history = log.history().expect("history");
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].version, 0);
        assert_eq!(history[1].version, 1);
        assert_eq!(history[2].version, 2);
    }

    // ─── Table writer/reader tests ───────────────────────────────────────

    #[test]
    fn test_write_read_roundtrip() {
        let (_dir, path) = temp_table_path();
        let config = DeltaConfig {
            base_path: path.clone(),
            ..Default::default()
        };

        let writer = DeltaTableWriter::new(config.clone()).expect("writer");
        let schema = sample_schema();
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let ver = writer.write(&data, &schema).expect("write");
        assert_eq!(ver, 0);

        let reader = DeltaTableReader::new(config).expect("reader");
        let read_data = reader.read(None).expect("read");

        assert_eq!(read_data.len(), 2);
        assert_eq!(read_data[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(read_data[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_version_history_grows() {
        let (_dir, path) = temp_table_path();
        let config = DeltaConfig {
            base_path: path.clone(),
            ..Default::default()
        };

        let writer = DeltaTableWriter::new(config.clone()).expect("writer");
        let schema = sample_schema();

        writer.write(&[vec![1.0], vec![2.0]], &schema).expect("w0");
        writer.write(&[vec![3.0], vec![4.0]], &schema).expect("w1");
        writer.write(&[vec![5.0], vec![6.0]], &schema).expect("w2");

        let reader = DeltaTableReader::new(config).expect("reader");
        let history = reader.history().expect("history");
        assert_eq!(history.len(), 3);
    }

    #[test]
    fn test_time_travel_to_v0() {
        let (_dir, path) = temp_table_path();
        let config = DeltaConfig {
            base_path: path.clone(),
            ..Default::default()
        };

        let writer = DeltaTableWriter::new(config.clone()).expect("writer");
        let schema = sample_schema();

        writer
            .write(&[vec![1.0, 2.0], vec![10.0, 20.0]], &schema)
            .expect("w0");
        writer
            .write(&[vec![3.0, 4.0], vec![30.0, 40.0]], &schema)
            .expect("w1");

        let reader = DeltaTableReader::new(config).expect("reader");

        // Read at version 0 — should only have the first write's data
        let v0_data = reader.read(Some(0)).expect("read v0");
        assert_eq!(v0_data.len(), 2);
        assert_eq!(v0_data[0], vec![1.0, 2.0]);
        assert_eq!(v0_data[1], vec![10.0, 20.0]);

        // Read latest — should have both writes' data
        let latest_data = reader.read(None).expect("read latest");
        assert_eq!(latest_data[0].len(), 4);
    }

    #[test]
    fn test_schema_evolution_add_column() {
        let (_dir, path) = temp_table_path();
        let config = DeltaConfig {
            base_path: path.clone(),
            ..Default::default()
        };

        let writer = DeltaTableWriter::new(config.clone()).expect("writer");
        let schema = sample_schema();

        writer.write(&[vec![1.0], vec![2.0]], &schema).expect("w0");

        // Add a new column
        let new_col = ColumnSchema {
            name: "z".to_string(),
            data_type: "double".to_string(),
            nullable: true,
        };
        let ver = add_column(writer.transaction_log(), &schema, new_col).expect("add col");
        assert!(ver > 0);

        // Verify schema was updated
        let reader = DeltaTableReader::new(config).expect("reader");
        let state = reader.table_state(None).expect("state");
        let new_schema = state.schema.expect("schema should exist");
        assert_eq!(new_schema.columns.len(), 3);
        assert_eq!(new_schema.columns[2].name, "z");
    }

    #[test]
    fn test_schema_evolution_rename_column() {
        let (_dir, path) = temp_table_path();
        let config = DeltaConfig {
            base_path: path.clone(),
            ..Default::default()
        };

        let writer = DeltaTableWriter::new(config.clone()).expect("writer");
        let schema = sample_schema();

        writer.write(&[vec![1.0], vec![2.0]], &schema).expect("w0");

        let ver =
            rename_column(writer.transaction_log(), &schema, "x", "x_renamed").expect("rename col");
        assert!(ver > 0);

        let reader = DeltaTableReader::new(config).expect("reader");
        let state = reader.table_state(None).expect("state");
        let updated_schema = state.schema.expect("schema");
        assert!(updated_schema.columns.iter().any(|c| c.name == "x_renamed"));
        assert!(!updated_schema.columns.iter().any(|c| c.name == "x"));
    }

    #[test]
    fn test_action_serialize_deserialize() {
        let action = DeltaAction::Add {
            path: "test.csv".to_string(),
            size: 42,
            modification_time: 1000,
            data_change: true,
            partition_values: HashMap::new(),
            stats_json: None,
        };

        let json = serialize_action(&action).expect("serialize");
        let parsed = parse_action_line(&json).expect("parse");

        match parsed {
            DeltaAction::Add { path, size, .. } => {
                assert_eq!(path, "test.csv");
                assert_eq!(size, 42);
            }
            other => panic!("expected Add, got {other:?}"),
        }
    }

    #[test]
    fn test_metadata_action_roundtrip() {
        let action = DeltaAction::Metadata {
            schema: r#"{"type":"struct","fields":[]}"#.to_string(),
            partition_columns: vec!["date".to_string()],
            description: Some("test table".to_string()),
            configuration: HashMap::new(),
        };

        let json = serialize_action(&action).expect("serialize");
        let parsed = parse_action_line(&json).expect("parse");

        match parsed {
            DeltaAction::Metadata {
                partition_columns,
                description,
                ..
            } => {
                assert_eq!(partition_columns, vec!["date".to_string()]);
                assert_eq!(description, Some("test table".to_string()));
            }
            other => panic!("expected Metadata, got {other:?}"),
        }
    }

    #[test]
    fn test_protocol_action_roundtrip() {
        let action = DeltaAction::Protocol {
            min_reader_version: 2,
            min_writer_version: 3,
        };

        let json = serialize_action(&action).expect("serialize");
        let parsed = parse_action_line(&json).expect("parse");

        match parsed {
            DeltaAction::Protocol {
                min_reader_version,
                min_writer_version,
            } => {
                assert_eq!(min_reader_version, 2);
                assert_eq!(min_writer_version, 3);
            }
            other => panic!("expected Protocol, got {other:?}"),
        }
    }

    #[test]
    fn test_delta_config_default() {
        let config = DeltaConfig::default();
        assert_eq!(config.checkpoint_interval, 10);
        assert_eq!(config.max_files_to_scan, 10_000);
    }

    #[test]
    fn test_delta_transaction() {
        let mut txn = DeltaTransaction::new(5);
        assert_eq!(txn.target_version, 5);
        assert!(txn.actions.is_empty());

        txn.add_action(DeltaAction::Add {
            path: "f.csv".to_string(),
            size: 1,
            modification_time: 0,
            data_change: true,
            partition_values: HashMap::new(),
            stats_json: None,
        });
        assert_eq!(txn.actions.len(), 1);
    }

    #[test]
    fn test_schema_to_json_and_back() {
        let schema = sample_schema();
        let json = schema.to_json().expect("to json");
        let parsed = Schema::from_json(&json).expect("from json");
        assert_eq!(parsed.columns.len(), 2);
        assert_eq!(parsed.columns[0].name, "x");
        assert_eq!(parsed.columns[1].name, "y");
    }
}
