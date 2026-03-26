//! DataFusion-compatible table provider interface.
//!
//! This module defines a standalone, zero-dependency table abstraction that mirrors
//! the Apache DataFusion `TableProvider` trait without pulling in the `datafusion`
//! or `arrow` crates.
//!
//! # Overview
//!
//! - `Schema` / `Field` / `DataType` — schema definitions
//! - `ColumnValues` / `ColumnBatch` / `RecordBatch` — in-memory columnar batches
//! - `TableProvider` trait — uniform scan interface
//! - `InMemoryTable` — simple in-memory implementation
//! - `CsvTableProvider` — reads a CSV file and exposes it as a table
//! - `filter_batch` / `project_batch` — predicate pushdown and column projection helpers

use std::fs;

use crate::error::IoError;

// ─── Schema types ─────────────────────────────────────────────────────────────

/// Scalar or nested data type for a table column.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    /// 32-bit signed integer.
    Int32,
    /// 64-bit signed integer.
    Int64,
    /// 32-bit IEEE 754 floating-point number.
    Float32,
    /// 64-bit IEEE 754 floating-point number.
    Float64,
    /// UTF-8 encoded string.
    Utf8,
    /// Boolean (true / false).
    Boolean,
    /// Opaque binary data.
    Binary,
    /// Variable-length list of elements of the given type.
    List(Box<DataType>),
    /// Struct composed of named fields.
    Struct(Vec<Field>),
}

/// A named, typed column descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    /// Column name.
    pub name: String,
    /// Column data type.
    pub data_type: DataType,
    /// Whether the column may contain `NULL` values.
    pub nullable: bool,
}

impl Field {
    /// Create a new field with the given name, type, and nullability.
    pub fn new(name: impl Into<String>, data_type: DataType, nullable: bool) -> Self {
        Self {
            name: name.into(),
            data_type,
            nullable,
        }
    }
}

/// An ordered collection of `Field` descriptors forming a table schema.
#[derive(Debug, Clone, PartialEq)]
pub struct Schema {
    /// Column descriptors, in order.
    pub fields: Vec<Field>,
}

impl Schema {
    /// Create a new schema from a list of fields.
    pub fn new(fields: Vec<Field>) -> Self {
        Self { fields }
    }

    /// Find a field by name (case-sensitive linear scan).
    pub fn find_field(&self, name: &str) -> Option<&Field> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Return the number of columns in this schema.
    pub fn n_fields(&self) -> usize {
        self.fields.len()
    }
}

// ─── In-memory column data ────────────────────────────────────────────────────

/// Typed, nullable column values for one column in a `RecordBatch`.
#[derive(Debug, Clone)]
pub enum ColumnValues {
    /// 32-bit signed integers.
    Int32s(Vec<Option<i32>>),
    /// 64-bit signed integers.
    Int64s(Vec<Option<i64>>),
    /// 32-bit floats.
    Float32s(Vec<Option<f32>>),
    /// 64-bit floats.
    Float64s(Vec<Option<f64>>),
    /// UTF-8 strings.
    Utf8s(Vec<Option<String>>),
    /// Booleans.
    Booleans(Vec<Option<bool>>),
}

impl ColumnValues {
    /// Return the number of rows represented by this column.
    pub fn len(&self) -> usize {
        match self {
            ColumnValues::Int32s(v) => v.len(),
            ColumnValues::Int64s(v) => v.len(),
            ColumnValues::Float32s(v) => v.len(),
            ColumnValues::Float64s(v) => v.len(),
            ColumnValues::Utf8s(v) => v.len(),
            ColumnValues::Booleans(v) => v.len(),
        }
    }

    /// Return true if this column has no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Slice row range `[offset, offset + length)`.
    pub fn slice(&self, offset: usize, length: usize) -> ColumnValues {
        let end = (offset + length).min(self.len());
        match self {
            ColumnValues::Int32s(v) => ColumnValues::Int32s(v[offset..end].to_vec()),
            ColumnValues::Int64s(v) => ColumnValues::Int64s(v[offset..end].to_vec()),
            ColumnValues::Float32s(v) => ColumnValues::Float32s(v[offset..end].to_vec()),
            ColumnValues::Float64s(v) => ColumnValues::Float64s(v[offset..end].to_vec()),
            ColumnValues::Utf8s(v) => ColumnValues::Utf8s(v[offset..end].to_vec()),
            ColumnValues::Booleans(v) => ColumnValues::Booleans(v[offset..end].to_vec()),
        }
    }

    /// Filter rows by a boolean mask.
    pub fn filter_by_mask(&self, mask: &[bool]) -> ColumnValues {
        match self {
            ColumnValues::Int32s(v) => ColumnValues::Int32s(
                v.iter()
                    .zip(mask)
                    .filter_map(|(val, &m)| if m { Some(val.clone()) } else { None })
                    .collect(),
            ),
            ColumnValues::Int64s(v) => ColumnValues::Int64s(
                v.iter()
                    .zip(mask)
                    .filter_map(|(val, &m)| if m { Some(val.clone()) } else { None })
                    .collect(),
            ),
            ColumnValues::Float32s(v) => ColumnValues::Float32s(
                v.iter()
                    .zip(mask)
                    .filter_map(|(val, &m)| if m { Some(*val) } else { None })
                    .collect(),
            ),
            ColumnValues::Float64s(v) => ColumnValues::Float64s(
                v.iter()
                    .zip(mask)
                    .filter_map(|(val, &m)| if m { Some(*val) } else { None })
                    .collect(),
            ),
            ColumnValues::Utf8s(v) => ColumnValues::Utf8s(
                v.iter()
                    .zip(mask)
                    .filter_map(|(val, &m)| if m { Some(val.clone()) } else { None })
                    .collect(),
            ),
            ColumnValues::Booleans(v) => ColumnValues::Booleans(
                v.iter()
                    .zip(mask)
                    .filter_map(|(val, &m)| if m { Some(*val) } else { None })
                    .collect(),
            ),
        }
    }
}

/// A named column batch: column name + typed values.
#[derive(Debug, Clone)]
pub struct ColumnBatch {
    /// Column name matching the schema field.
    pub name: String,
    /// Declared data type.
    pub data_type: DataType,
    /// The actual values.
    pub values: ColumnValues,
}

/// An in-memory columnar batch: a schema plus one `ColumnBatch` per field.
#[derive(Debug, Clone)]
pub struct RecordBatch {
    /// Schema of this batch.
    pub schema: Schema,
    /// Column data, one per schema field, in schema order.
    pub columns: Vec<ColumnBatch>,
    /// Number of rows.
    pub n_rows: usize,
}

impl RecordBatch {
    /// Create a new `RecordBatch`, validating that all columns have consistent row counts.
    pub fn new(schema: Schema, columns: Vec<ColumnBatch>) -> Result<Self, IoError> {
        // Validate column count.
        if columns.len() != schema.n_fields() {
            return Err(IoError::ValidationError(format!(
                "schema has {} fields but {} columns were provided",
                schema.n_fields(),
                columns.len()
            )));
        }

        // Validate row count consistency.
        let n_rows = if columns.is_empty() {
            0
        } else {
            columns[0].values.len()
        };
        for col in &columns {
            if col.values.len() != n_rows {
                return Err(IoError::ValidationError(format!(
                    "column '{}' has {} rows but expected {n_rows}",
                    col.name,
                    col.values.len()
                )));
            }
        }

        Ok(Self {
            schema,
            columns,
            n_rows,
        })
    }

    /// Look up a column by name.
    pub fn column(&self, name: &str) -> Option<&ColumnBatch> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// Return a sub-batch with rows `[offset, offset + length)`.
    pub fn slice(&self, offset: usize, length: usize) -> Result<RecordBatch, IoError> {
        if offset > self.n_rows {
            return Err(IoError::ValidationError(format!(
                "slice offset {offset} exceeds n_rows {}",
                self.n_rows
            )));
        }
        let actual_len = length.min(self.n_rows.saturating_sub(offset));
        let new_columns: Vec<ColumnBatch> = self
            .columns
            .iter()
            .map(|c| ColumnBatch {
                name: c.name.clone(),
                data_type: c.data_type.clone(),
                values: c.values.slice(offset, actual_len),
            })
            .collect();

        Ok(RecordBatch {
            schema: self.schema.clone(),
            columns: new_columns,
            n_rows: actual_len,
        })
    }
}

// ─── TableProvider trait ──────────────────────────────────────────────────────

/// A uniform table scan interface compatible with Apache DataFusion's `TableProvider`.
///
/// Implementors may expose any data source (in-memory, CSV, Parquet, Delta Lake…)
/// through this interface.
pub trait TableProvider: Send + Sync {
    /// Return the table's schema.
    fn schema(&self) -> &Schema;

    /// Produce an iterator of `RecordBatch`es.
    ///
    /// `projection` is an optional slice of **column indices** to include.
    /// `limit` is an optional upper bound on the total number of rows to return.
    fn scan(
        &self,
        projection: Option<&[usize]>,
        limit: Option<usize>,
    ) -> Result<Box<dyn Iterator<Item = Result<RecordBatch, IoError>>>, IoError>;

    /// Optional row-count estimate for query planning.
    fn n_rows_estimate(&self) -> Option<usize> {
        None
    }
}

// ─── InMemoryTable ────────────────────────────────────────────────────────────

/// A simple `TableProvider` backed by an in-memory list of `RecordBatch`es.
pub struct InMemoryTable {
    schema: Schema,
    batches: Vec<RecordBatch>,
}

impl InMemoryTable {
    /// Create a new in-memory table with the given schema and batches.
    pub fn new(schema: Schema, batches: Vec<RecordBatch>) -> Self {
        Self { schema, batches }
    }

    /// Return the total number of rows across all batches.
    pub fn total_rows(&self) -> usize {
        self.batches.iter().map(|b| b.n_rows).sum()
    }
}

impl TableProvider for InMemoryTable {
    fn schema(&self) -> &Schema {
        &self.schema
    }

    fn scan(
        &self,
        projection: Option<&[usize]>,
        limit: Option<usize>,
    ) -> Result<Box<dyn Iterator<Item = Result<RecordBatch, IoError>>>, IoError> {
        let proj_indices: Option<Vec<usize>> = projection.map(|p| p.to_vec());
        let mut batches: Vec<Result<RecordBatch, IoError>> = Vec::new();
        let mut rows_remaining = limit;

        for batch in &self.batches {
            // Apply limit.
            let batch = if let Some(rem) = rows_remaining {
                if rem == 0 {
                    break;
                }
                let sliced = batch.slice(0, rem)?;
                rows_remaining = Some(rem.saturating_sub(sliced.n_rows));
                sliced
            } else {
                batch.clone()
            };

            // Apply projection.
            let batch = if let Some(ref indices) = proj_indices {
                project_batch_by_indices(&batch, indices)?
            } else {
                batch
            };

            batches.push(Ok(batch));
        }

        Ok(Box::new(batches.into_iter()))
    }

    fn n_rows_estimate(&self) -> Option<usize> {
        Some(self.total_rows())
    }
}

// ─── CsvTableProvider ─────────────────────────────────────────────────────────

/// A `TableProvider` that reads a CSV file and serves it as a single `RecordBatch`.
///
/// Schema inference rule: all columns are inferred as `Utf8`; column names come
/// from the first row when `has_header = true`, otherwise `col_0`, `col_1`, …
pub struct CsvTableProvider {
    schema: Schema,
    batch: RecordBatch,
}

impl CsvTableProvider {
    /// Open a CSV file, parse it, and infer the schema.
    pub fn open(path: &str, has_header: bool) -> Result<Self, IoError> {
        let content = fs::read_to_string(path)
            .map_err(|e| IoError::FileNotFound(format!("cannot read CSV file {path}: {e}")))?;

        let mut lines = content.lines();
        let first_line = match lines.next() {
            Some(l) => l,
            None => {
                // Empty file — return empty schema/batch.
                let schema = Schema::new(vec![]);
                let batch = RecordBatch {
                    schema: schema.clone(),
                    columns: vec![],
                    n_rows: 0,
                };
                return Ok(Self { schema, batch });
            }
        };

        let header_fields: Vec<String> = split_csv_line(first_line);

        // Build schema: all columns are Utf8 (inference can be extended later).
        let column_names: Vec<String> = if has_header {
            header_fields.clone()
        } else {
            (0..header_fields.len())
                .map(|i| format!("col_{i}"))
                .collect()
        };

        let fields: Vec<Field> = column_names
            .iter()
            .map(|name| Field::new(name.clone(), DataType::Utf8, true))
            .collect();
        let schema = Schema::new(fields);

        // Parse rows.
        let mut row_data: Vec<Vec<Option<String>>> = Vec::new();

        // If no header, the first line is data.
        if !has_header {
            let row: Vec<Option<String>> = header_fields
                .into_iter()
                .map(|v| if v.is_empty() { None } else { Some(v) })
                .collect();
            row_data.push(row);
        }

        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let cols = split_csv_line(line);
            let row: Vec<Option<String>> = cols
                .into_iter()
                .map(|v| if v.is_empty() { None } else { Some(v) })
                .collect();
            row_data.push(row);
        }

        let n_cols = schema.n_fields();
        let n_rows = row_data.len();

        // Build per-column vectors.
        let mut col_vecs: Vec<Vec<Option<String>>> = vec![Vec::with_capacity(n_rows); n_cols];
        for row in &row_data {
            for (ci, cell) in row.iter().enumerate() {
                if ci < n_cols {
                    col_vecs[ci].push(cell.clone());
                }
            }
            // Pad missing columns.
            for ci in row.len()..n_cols {
                col_vecs[ci].push(None);
            }
        }

        let columns: Vec<ColumnBatch> = schema
            .fields
            .iter()
            .zip(col_vecs)
            .map(|(field, vals)| ColumnBatch {
                name: field.name.clone(),
                data_type: DataType::Utf8,
                values: ColumnValues::Utf8s(vals),
            })
            .collect();

        let batch = RecordBatch::new(schema.clone(), columns)?;
        Ok(Self { schema, batch })
    }
}

impl TableProvider for CsvTableProvider {
    fn schema(&self) -> &Schema {
        &self.schema
    }

    fn scan(
        &self,
        projection: Option<&[usize]>,
        limit: Option<usize>,
    ) -> Result<Box<dyn Iterator<Item = Result<RecordBatch, IoError>>>, IoError> {
        let batch = if let Some(lim) = limit {
            self.batch.slice(0, lim)?
        } else {
            self.batch.clone()
        };

        let batch = if let Some(proj) = projection {
            project_batch_by_indices(&batch, proj)?
        } else {
            batch
        };

        Ok(Box::new(std::iter::once(Ok(batch))))
    }

    fn n_rows_estimate(&self) -> Option<usize> {
        Some(self.batch.n_rows)
    }
}

// ─── Batch helpers ────────────────────────────────────────────────────────────

/// Apply a row-level predicate to a `RecordBatch`, returning only matching rows.
///
/// `predicate(values, row_index)` should return `true` to keep the row.
pub fn filter_batch(
    batch: &RecordBatch,
    column: &str,
    predicate: impl Fn(&ColumnValues, usize) -> bool,
) -> Result<RecordBatch, IoError> {
    let col = batch
        .column(column)
        .ok_or_else(|| IoError::NotFound(format!("column '{column}' not found in batch")))?;

    let mask: Vec<bool> = (0..batch.n_rows)
        .map(|i| predicate(&col.values, i))
        .collect();

    let n_rows = mask.iter().filter(|&&m| m).count();

    let new_columns: Vec<ColumnBatch> = batch
        .columns
        .iter()
        .map(|c| ColumnBatch {
            name: c.name.clone(),
            data_type: c.data_type.clone(),
            values: c.values.filter_by_mask(&mask),
        })
        .collect();

    Ok(RecordBatch {
        schema: batch.schema.clone(),
        columns: new_columns,
        n_rows,
    })
}

/// Project a `RecordBatch` to the named columns, in the order specified.
pub fn project_batch(batch: &RecordBatch, columns: &[&str]) -> Result<RecordBatch, IoError> {
    let mut new_fields = Vec::new();
    let mut new_columns = Vec::new();

    for &name in columns {
        let field = batch
            .schema
            .find_field(name)
            .ok_or_else(|| IoError::NotFound(format!("projection: column '{name}' not found")))?;
        let col = batch.column(name).ok_or_else(|| {
            IoError::NotFound(format!("projection: column data '{name}' not found"))
        })?;
        new_fields.push(field.clone());
        new_columns.push(col.clone());
    }

    let new_schema = Schema::new(new_fields);
    Ok(RecordBatch {
        schema: new_schema,
        columns: new_columns,
        n_rows: batch.n_rows,
    })
}

/// Project a `RecordBatch` to the columns at the given **indices** (0-based).
pub fn project_batch_by_indices(
    batch: &RecordBatch,
    indices: &[usize],
) -> Result<RecordBatch, IoError> {
    let mut new_fields = Vec::new();
    let mut new_columns = Vec::new();

    for &idx in indices {
        let field = batch.schema.fields.get(idx).ok_or_else(|| {
            IoError::ValidationError(format!(
                "projection index {idx} out of range (schema has {} fields)",
                batch.schema.n_fields()
            ))
        })?;
        let col = batch
            .columns
            .get(idx)
            .ok_or_else(|| IoError::ValidationError(format!("column index {idx} out of range")))?;
        new_fields.push(field.clone());
        new_columns.push(col.clone());
    }

    let new_schema = Schema::new(new_fields);
    Ok(RecordBatch {
        schema: new_schema,
        columns: new_columns,
        n_rows: batch.n_rows,
    })
}

// ─── CSV parsing helper ───────────────────────────────────────────────────────

/// Split a CSV line on commas, stripping optional double-quote wrapping.
fn split_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '"' if !in_quotes => {
                in_quotes = true;
            }
            '"' if in_quotes => {
                // Escaped double quote?
                if chars.peek() == Some(&'"') {
                    current.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            }
            ',' if !in_quotes => {
                fields.push(current.trim().to_string());
                current = String::new();
            }
            other => current.push(other),
        }
    }
    fields.push(current.trim().to_string());
    fields
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_schema() -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("score", DataType::Float64, true),
        ])
    }

    fn make_batch(schema: &Schema, n: usize) -> RecordBatch {
        let ids: Vec<Option<i32>> = (0..n as i32).map(Some).collect();
        let names: Vec<Option<String>> = (0..n).map(|i| Some(format!("user_{i}"))).collect();
        let scores: Vec<Option<f64>> = (0..n).map(|i| Some(i as f64 * 1.5)).collect();

        RecordBatch::new(
            schema.clone(),
            vec![
                ColumnBatch {
                    name: "id".to_string(),
                    data_type: DataType::Int32,
                    values: ColumnValues::Int32s(ids),
                },
                ColumnBatch {
                    name: "name".to_string(),
                    data_type: DataType::Utf8,
                    values: ColumnValues::Utf8s(names),
                },
                ColumnBatch {
                    name: "score".to_string(),
                    data_type: DataType::Float64,
                    values: ColumnValues::Float64s(scores),
                },
            ],
        )
        .expect("make_batch")
    }

    // ── Schema ────────────────────────────────────────────────────────────────

    #[test]
    fn test_schema_find_field() {
        let schema = make_schema();
        assert!(schema.find_field("id").is_some());
        assert!(schema.find_field("name").is_some());
        assert!(schema.find_field("missing").is_none());
        assert_eq!(schema.n_fields(), 3);
    }

    // ── RecordBatch ───────────────────────────────────────────────────────────

    #[test]
    fn test_record_batch_column_count() {
        let schema = make_schema();
        let batch = make_batch(&schema, 10);
        assert_eq!(batch.n_rows, 10);
        assert!(batch.column("id").is_some());
        assert!(batch.column("score").is_some());
        assert!(batch.column("missing").is_none());
    }

    #[test]
    fn test_record_batch_slice() {
        let schema = make_schema();
        let batch = make_batch(&schema, 10);
        let sliced = batch.slice(2, 3).expect("slice");
        assert_eq!(sliced.n_rows, 3);
        if let ColumnValues::Int32s(ref ids) = sliced.column("id").unwrap().values {
            assert_eq!(ids[0], Some(2));
            assert_eq!(ids[2], Some(4));
        } else {
            panic!("expected Int32s");
        }
    }

    #[test]
    fn test_record_batch_slice_beyond_end() {
        let schema = make_schema();
        let batch = make_batch(&schema, 5);
        let sliced = batch.slice(3, 100).expect("slice beyond end");
        assert_eq!(sliced.n_rows, 2);
    }

    #[test]
    fn test_record_batch_new_column_count_mismatch() {
        let schema = make_schema();
        // Provide only 2 columns for a 3-field schema.
        let result = RecordBatch::new(
            schema,
            vec![
                ColumnBatch {
                    name: "id".to_string(),
                    data_type: DataType::Int32,
                    values: ColumnValues::Int32s(vec![Some(1)]),
                },
                ColumnBatch {
                    name: "name".to_string(),
                    data_type: DataType::Utf8,
                    values: ColumnValues::Utf8s(vec![Some("a".to_string())]),
                },
            ],
        );
        assert!(result.is_err());
    }

    // ── InMemoryTable ─────────────────────────────────────────────────────────

    #[test]
    fn test_inmemory_table_scan_all() {
        let schema = make_schema();
        let batch1 = make_batch(&schema, 5);
        let batch2 = make_batch(&schema, 3);
        let table = InMemoryTable::new(schema, vec![batch1, batch2]);

        let batches: Vec<RecordBatch> = table
            .scan(None, None)
            .expect("scan")
            .map(|r| r.expect("batch"))
            .collect();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].n_rows, 5);
        assert_eq!(batches[1].n_rows, 3);
        assert_eq!(table.n_rows_estimate(), Some(8));
    }

    #[test]
    fn test_inmemory_table_scan_limit() {
        let schema = make_schema();
        let batch = make_batch(&schema, 10);
        let table = InMemoryTable::new(schema, vec![batch]);

        let batches: Vec<RecordBatch> = table
            .scan(None, Some(4))
            .expect("scan")
            .map(|r| r.expect("batch"))
            .collect();
        let total: usize = batches.iter().map(|b| b.n_rows).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_inmemory_table_projection() {
        let schema = make_schema();
        let batch = make_batch(&schema, 5);
        let table = InMemoryTable::new(schema, vec![batch]);

        // Project only columns 0 and 2 (id and score).
        let batches: Vec<RecordBatch> = table
            .scan(Some(&[0, 2]), None)
            .expect("scan")
            .map(|r| r.expect("batch"))
            .collect();
        assert_eq!(batches[0].schema.n_fields(), 2);
        assert!(batches[0].column("id").is_some());
        assert!(batches[0].column("score").is_some());
        assert!(batches[0].column("name").is_none());
    }

    // ── project_batch ─────────────────────────────────────────────────────────

    #[test]
    fn test_project_batch_by_name() {
        let schema = make_schema();
        let batch = make_batch(&schema, 4);
        let projected = project_batch(&batch, &["id", "score"]).expect("project");
        assert_eq!(projected.schema.n_fields(), 2);
        assert_eq!(projected.n_rows, 4);
    }

    #[test]
    fn test_project_batch_missing_column() {
        let schema = make_schema();
        let batch = make_batch(&schema, 4);
        let result = project_batch(&batch, &["nonexistent"]);
        assert!(result.is_err());
    }

    // ── filter_batch ──────────────────────────────────────────────────────────

    #[test]
    fn test_filter_batch_int32() {
        let schema = make_schema();
        let batch = make_batch(&schema, 10);

        // Keep rows where id >= 5.
        let filtered = filter_batch(&batch, "id", |vals, row| {
            if let ColumnValues::Int32s(ref v) = vals {
                v[row].map_or(false, |id| id >= 5)
            } else {
                false
            }
        })
        .expect("filter");

        assert_eq!(filtered.n_rows, 5);
    }

    #[test]
    fn test_filter_batch_missing_column() {
        let schema = make_schema();
        let batch = make_batch(&schema, 5);
        let result = filter_batch(&batch, "nonexistent", |_, _| true);
        assert!(result.is_err());
    }

    // ── CsvTableProvider ──────────────────────────────────────────────────────

    fn write_temp_csv(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        f.write_all(content.as_bytes()).expect("write csv");
        f
    }

    #[test]
    fn test_csv_provider_with_header() {
        let csv = "id,name,value\n1,alice,10.5\n2,bob,20.0\n3,carol,30.3\n";
        let tmp = write_temp_csv(csv);
        let provider =
            CsvTableProvider::open(tmp.path().to_str().unwrap(), true).expect("open csv");

        assert_eq!(provider.schema().n_fields(), 3);
        assert!(provider.schema().find_field("id").is_some());
        assert!(provider.schema().find_field("name").is_some());
        assert!(provider.schema().find_field("value").is_some());

        let batches: Vec<RecordBatch> = provider
            .scan(None, None)
            .expect("scan")
            .map(|r| r.expect("batch"))
            .collect();
        assert_eq!(batches[0].n_rows, 3);
    }

    #[test]
    fn test_csv_provider_without_header() {
        let csv = "1,alice\n2,bob\n";
        let tmp = write_temp_csv(csv);
        let provider = CsvTableProvider::open(tmp.path().to_str().unwrap(), false)
            .expect("open csv no-header");

        assert_eq!(provider.schema().n_fields(), 2);
        assert!(provider.schema().find_field("col_0").is_some());
        assert!(provider.schema().find_field("col_1").is_some());

        let batches: Vec<RecordBatch> = provider
            .scan(None, None)
            .expect("scan")
            .map(|r| r.expect("batch"))
            .collect();
        assert_eq!(batches[0].n_rows, 2);
    }

    #[test]
    fn test_csv_provider_limit() {
        let csv = "a,b\n1,2\n3,4\n5,6\n7,8\n";
        let tmp = write_temp_csv(csv);
        let provider =
            CsvTableProvider::open(tmp.path().to_str().unwrap(), true).expect("open csv");

        let batches: Vec<RecordBatch> = provider
            .scan(None, Some(2))
            .expect("scan")
            .map(|r| r.expect("batch"))
            .collect();
        let total: usize = batches.iter().map(|b| b.n_rows).sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn test_csv_provider_projection() {
        let csv = "x,y,z\n1,2,3\n4,5,6\n";
        let tmp = write_temp_csv(csv);
        let provider =
            CsvTableProvider::open(tmp.path().to_str().unwrap(), true).expect("open csv");

        // Project only column 0 (x) and column 2 (z).
        let batches: Vec<RecordBatch> = provider
            .scan(Some(&[0, 2]), None)
            .expect("scan")
            .map(|r| r.expect("batch"))
            .collect();
        assert_eq!(batches[0].schema.n_fields(), 2);
        assert!(batches[0].column("x").is_some());
        assert!(batches[0].column("z").is_some());
        assert!(batches[0].column("y").is_none());
    }

    // ── split_csv_line ────────────────────────────────────────────────────────

    #[test]
    fn test_split_csv_quoted() {
        let line = r#"hello,"world, with comma",42"#;
        let fields = split_csv_line(line);
        assert_eq!(fields, vec!["hello", "world, with comma", "42"]);
    }

    #[test]
    fn test_split_csv_simple() {
        let fields = split_csv_line("a,b,c");
        assert_eq!(fields, vec!["a", "b", "c"]);
    }
}
