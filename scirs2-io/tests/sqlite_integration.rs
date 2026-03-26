//! Integration tests for the SQLite database backend.
//!
//! These tests exercise the full database I/O pipeline against a real, on-disk
//! SQLite database created in a temporary directory.  They are intentionally
//! separate from the unit tests in `src/database/mod.rs` so that they can
//! require the `sqlite` feature without contaminating the main test suite.

#![allow(clippy::unwrap_used)]

#[cfg(feature = "sqlite")]
mod sqlite_tests {
    use scirs2_core::ndarray::Array2;
    use scirs2_io::database::{
        ColumnDef, DataType, DatabaseConfig, DatabaseConnector, DatabaseType, Index, QueryBuilder,
        TableSchema,
    };
    use std::env;

    /// Return a unique temporary SQLite database path for a single test.
    fn temp_db_path(tag: &str) -> std::path::PathBuf {
        let mut p = env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        p.push(format!("scirs2_sqlite_test_{}_{}.db", tag, nanos));
        p
    }

    // -------------------------------------------------------------------------
    // 1. Connection creation and teardown
    // -------------------------------------------------------------------------

    #[test]
    fn test_sqlite_connection_open_close() {
        let path = temp_db_path("open_close");
        let config = DatabaseConfig::new(DatabaseType::SQLite, path.to_str().expect("valid path"));
        let conn = DatabaseConnector::connect(&config);
        assert!(conn.is_ok(), "Expected successful connection, got an error");
        // Connection is dropped here — no explicit close required.
        let _ = std::fs::remove_file(&path);
    }

    // -------------------------------------------------------------------------
    // 2. Table creation
    // -------------------------------------------------------------------------

    #[test]
    fn test_sqlite_create_table() {
        let path = temp_db_path("create_table");
        let config = DatabaseConfig::new(DatabaseType::SQLite, path.to_str().expect("valid path"));
        let conn = DatabaseConnector::connect(&config).expect("connection");

        let schema = TableSchema {
            name: "measurements".to_string(),
            columns: vec![
                ColumnDef {
                    name: "id".to_string(),
                    data_type: DataType::Integer,
                    nullable: false,
                    default: None,
                },
                ColumnDef {
                    name: "value".to_string(),
                    data_type: DataType::Double,
                    nullable: true,
                    default: None,
                },
            ],
            primary_key: Some(vec!["id".to_string()]),
            indexes: Vec::<Index>::new(),
        };

        let result = conn.create_table("measurements", &schema);
        assert!(result.is_ok(), "create_table failed: {:?}", result);
        let _ = std::fs::remove_file(&path);
    }

    // -------------------------------------------------------------------------
    // 3. Table existence check — positive case
    // -------------------------------------------------------------------------

    #[test]
    fn test_sqlite_table_exists_after_creation() {
        let path = temp_db_path("table_exists");
        let config = DatabaseConfig::new(DatabaseType::SQLite, path.to_str().expect("valid path"));
        let conn = DatabaseConnector::connect(&config).expect("connection");

        let schema = TableSchema {
            name: "sensor_data".to_string(),
            columns: vec![ColumnDef {
                name: "reading".to_string(),
                data_type: DataType::Float,
                nullable: true,
                default: None,
            }],
            primary_key: None,
            indexes: Vec::<Index>::new(),
        };

        conn.create_table("sensor_data", &schema)
            .expect("create_table");

        let exists = conn.table_exists("sensor_data").expect("table_exists");
        assert!(exists, "Table should exist after creation");
        let _ = std::fs::remove_file(&path);
    }

    // -------------------------------------------------------------------------
    // 4. Table existence check — negative case
    // -------------------------------------------------------------------------

    #[test]
    fn test_sqlite_table_not_exists_for_unknown_table() {
        let path = temp_db_path("table_not_exists");
        let config = DatabaseConfig::new(DatabaseType::SQLite, path.to_str().expect("valid path"));
        let conn = DatabaseConnector::connect(&config).expect("connection");

        let exists = conn
            .table_exists("definitely_does_not_exist_xyz")
            .expect("table_exists");
        assert!(!exists, "Table should not exist in an empty database");
        let _ = std::fs::remove_file(&path);
    }

    // -------------------------------------------------------------------------
    // 5. Array insert round-trip: insert_array + raw SELECT
    // -------------------------------------------------------------------------

    #[test]
    fn test_sqlite_insert_array_and_select_back() {
        let path = temp_db_path("insert_array");
        let config = DatabaseConfig::new(DatabaseType::SQLite, path.to_str().expect("valid path"));
        let conn = DatabaseConnector::connect(&config).expect("connection");

        // Create table
        let schema = TableSchema {
            name: "matrix_data".to_string(),
            columns: vec![
                ColumnDef {
                    name: "x".to_string(),
                    data_type: DataType::Double,
                    nullable: false,
                    default: None,
                },
                ColumnDef {
                    name: "y".to_string(),
                    data_type: DataType::Double,
                    nullable: false,
                    default: None,
                },
            ],
            primary_key: None,
            indexes: Vec::<Index>::new(),
        };
        conn.create_table("matrix_data", &schema)
            .expect("create_table");

        // Insert a 3×2 matrix
        let data = Array2::from_shape_vec((3, 2), vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("array from shape");

        let inserted = conn
            .insert_array("matrix_data", data.view(), &["x", "y"])
            .expect("insert_array");
        assert_eq!(inserted, 3, "Expected 3 rows inserted");

        // Read back via raw SQL
        let rs = conn
            .execute_sql("SELECT x, y FROM matrix_data ORDER BY x", &[])
            .expect("execute_sql");

        assert_eq!(rs.row_count(), 3);
        assert_eq!(rs.column_count(), 2);

        // Spot-check first and last rows
        let first_x = rs.rows[0][0].as_f64().expect("f64");
        assert!(
            (first_x - 1.0).abs() < 1e-9,
            "Expected first x = 1.0, got {}",
            first_x
        );
        let last_y = rs.rows[2][1].as_f64().expect("f64");
        assert!(
            (last_y - 6.0).abs() < 1e-9,
            "Expected last y = 6.0, got {}",
            last_y
        );

        let _ = std::fs::remove_file(&path);
    }

    // -------------------------------------------------------------------------
    // 6. QueryBuilder SELECT round-trip
    // -------------------------------------------------------------------------

    #[test]
    fn test_sqlite_query_builder_select() {
        let path = temp_db_path("query_builder");
        let config = DatabaseConfig::new(DatabaseType::SQLite, path.to_str().expect("valid path"));
        let conn = DatabaseConnector::connect(&config).expect("connection");

        // Setup table and data
        conn.execute_sql("CREATE TABLE scores (id INTEGER, score REAL NOT NULL)", &[])
            .expect("create");
        conn.execute_sql("INSERT INTO scores VALUES (1, 95.5)", &[])
            .expect("insert 1");
        conn.execute_sql("INSERT INTO scores VALUES (2, 87.2)", &[])
            .expect("insert 2");
        conn.execute_sql("INSERT INTO scores VALUES (3, 42.0)", &[])
            .expect("insert 3");

        // Query with LIMIT
        let query = QueryBuilder::select("scores")
            .columns(vec!["id", "score"])
            .order_by("score", true)
            .limit(2);

        let rs = conn.query(&query).expect("query");
        assert_eq!(rs.row_count(), 2, "LIMIT 2 should return 2 rows");

        // The highest score should be first (ORDER BY score DESC)
        let top_score = rs.rows[0][1].as_f64().expect("f64");
        assert!(
            (top_score - 95.5).abs() < 1e-9,
            "Expected top score 95.5, got {}",
            top_score
        );

        let _ = std::fs::remove_file(&path);
    }

    // -------------------------------------------------------------------------
    // 7. Schema introspection via get_schema
    // -------------------------------------------------------------------------

    #[test]
    fn test_sqlite_get_schema_reflects_created_table() {
        let path = temp_db_path("get_schema");
        let config = DatabaseConfig::new(DatabaseType::SQLite, path.to_str().expect("valid path"));
        let conn = DatabaseConnector::connect(&config).expect("connection");

        let schema = TableSchema {
            name: "experiment".to_string(),
            columns: vec![
                ColumnDef {
                    name: "trial_id".to_string(),
                    data_type: DataType::Integer,
                    nullable: false,
                    default: None,
                },
                ColumnDef {
                    name: "result".to_string(),
                    data_type: DataType::Double,
                    nullable: true,
                    default: None,
                },
                ColumnDef {
                    name: "label".to_string(),
                    data_type: DataType::Text,
                    nullable: true,
                    default: None,
                },
            ],
            primary_key: Some(vec!["trial_id".to_string()]),
            indexes: Vec::<Index>::new(),
        };

        conn.create_table("experiment", &schema)
            .expect("create_table");

        let retrieved = conn.get_schema("experiment").expect("get_schema");
        assert_eq!(retrieved.name, "experiment");
        assert_eq!(retrieved.columns.len(), 3);

        let col_names: Vec<&str> = retrieved.columns.iter().map(|c| c.name.as_str()).collect();
        assert!(col_names.contains(&"trial_id"));
        assert!(col_names.contains(&"result"));
        assert!(col_names.contains(&"label"));

        let _ = std::fs::remove_file(&path);
    }

    // -------------------------------------------------------------------------
    // 8. Result set to_array conversion
    // -------------------------------------------------------------------------

    #[test]
    fn test_sqlite_result_set_to_array() {
        let path = temp_db_path("to_array");
        let config = DatabaseConfig::new(DatabaseType::SQLite, path.to_str().expect("valid path"));
        let conn = DatabaseConnector::connect(&config).expect("connection");

        conn.execute_sql("CREATE TABLE nums (a REAL, b REAL, c REAL)", &[])
            .expect("create");
        conn.execute_sql("INSERT INTO nums VALUES (1.0, 2.0, 3.0)", &[])
            .expect("insert");
        conn.execute_sql("INSERT INTO nums VALUES (4.0, 5.0, 6.0)", &[])
            .expect("insert");

        let rs = conn
            .execute_sql("SELECT a, b, c FROM nums ORDER BY a", &[])
            .expect("execute_sql");

        let arr = rs.to_array().expect("to_array");
        assert_eq!(arr.shape(), &[2, 3]);
        assert!((arr[[0, 0]] - 1.0).abs() < 1e-9);
        assert!((arr[[1, 2]] - 6.0).abs() < 1e-9);

        let _ = std::fs::remove_file(&path);
    }

    // -------------------------------------------------------------------------
    // 9. Result set get_column
    // -------------------------------------------------------------------------

    #[test]
    fn test_sqlite_result_set_get_column() {
        let path = temp_db_path("get_column");
        let config = DatabaseConfig::new(DatabaseType::SQLite, path.to_str().expect("valid path"));
        let conn = DatabaseConnector::connect(&config).expect("connection");

        conn.execute_sql("CREATE TABLE vectors (x REAL, y REAL)", &[])
            .expect("create");
        for i in 1i64..=5i64 {
            conn.execute_sql(
                &format!("INSERT INTO vectors VALUES ({}, {})", i, i * 2),
                &[],
            )
            .expect("insert");
        }

        let rs = conn
            .execute_sql("SELECT x, y FROM vectors ORDER BY x", &[])
            .expect("execute_sql");

        let xs = rs.get_column("x").expect("get_column x");
        assert_eq!(xs.len(), 5);
        assert!((xs[0] - 1.0).abs() < 1e-9);
        assert!((xs[4] - 5.0).abs() < 1e-9);

        let ys = rs.get_column("y").expect("get_column y");
        assert!((ys[0] - 2.0).abs() < 1e-9);
        assert!((ys[4] - 10.0).abs() < 1e-9);

        let _ = std::fs::remove_file(&path);
    }

    // -------------------------------------------------------------------------
    // 10. Error handling — invalid SQL
    // -------------------------------------------------------------------------

    #[test]
    fn test_sqlite_invalid_sql_returns_error() {
        let path = temp_db_path("invalid_sql");
        let config = DatabaseConfig::new(DatabaseType::SQLite, path.to_str().expect("valid path"));
        let conn = DatabaseConnector::connect(&config).expect("connection");

        let result = conn.execute_sql("THIS IS NOT VALID SQL AT ALL ;;; ???", &[]);
        assert!(result.is_err(), "Expected an error for invalid SQL, got Ok");

        let _ = std::fs::remove_file(&path);
    }

    // -------------------------------------------------------------------------
    // 11. Error handling — query against non-existent table
    // -------------------------------------------------------------------------

    #[test]
    fn test_sqlite_select_from_missing_table_returns_error() {
        let path = temp_db_path("missing_table");
        let config = DatabaseConfig::new(DatabaseType::SQLite, path.to_str().expect("valid path"));
        let conn = DatabaseConnector::connect(&config).expect("connection");

        let result = conn.execute_sql("SELECT * FROM ghost_table", &[]);
        assert!(
            result.is_err(),
            "Expected an error when selecting from non-existent table"
        );

        let _ = std::fs::remove_file(&path);
    }

    // -------------------------------------------------------------------------
    // 12. Unimplemented DatabaseType variants return errors
    // -------------------------------------------------------------------------

    #[test]
    #[allow(deprecated)]
    fn test_unsupported_database_types_return_errors() {
        // MongoDB
        let mongo_config = DatabaseConfig::new(DatabaseType::MongoDB, "mydb");
        let result = DatabaseConnector::connect(&mongo_config);
        assert!(
            result.is_err(),
            "MongoDB connect should return an error (not implemented)"
        );

        // Redis
        let redis_config = DatabaseConfig::new(DatabaseType::Redis, "mydb");
        let result = DatabaseConnector::connect(&redis_config);
        assert!(
            result.is_err(),
            "Redis connect should return an error (not implemented)"
        );

        // Cassandra
        let cass_config = DatabaseConfig::new(DatabaseType::Cassandra, "mydb");
        let result = DatabaseConnector::connect(&cass_config);
        assert!(
            result.is_err(),
            "Cassandra connect should return an error (not implemented)"
        );

        // InfluxDB
        let influx_config = DatabaseConfig::new(DatabaseType::InfluxDB, "mydb");
        let result = DatabaseConnector::connect(&influx_config);
        assert!(
            result.is_err(),
            "InfluxDB connect should return an error (not implemented)"
        );
    }
}
