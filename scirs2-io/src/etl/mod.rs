//! ETL (Extract-Transform-Load) pipeline for composable data processing.
//!
//! Provides a trait-based pipeline architecture where [`Extractor`], [`Transformer`],
//! and [`Loader`] implementations can be composed into an [`ETLPipeline`].
//!
//! # Example
//!
//! ```rust,no_run
//! use scirs2_io::etl::{
//!     ETLPipeline, CSVExtractor, DeduplicateTransform, FillNullTransform, NormalizeTransform,
//! };
//! use scirs2_io::database::table::ColumnValue;
//!
//! // Build a pipeline: CSV → deduplicate → fill nulls → normalize
//! let pipeline = ETLPipeline::new()
//!     .extract(CSVExtractor::from_path("/data/raw.csv").has_header(true))
//!     .transform(DeduplicateTransform::on_column("id"))
//!     .transform(FillNullTransform::with_constant("score", ColumnValue::Float(0.0)))
//!     .transform(NormalizeTransform::min_max("score"));
//!
//! let table = pipeline.run().unwrap();
//! println!("Processed {} rows", table.row_count());
//! ```

#![allow(missing_docs)]

use crate::database::table::{
    AggFunc, ColumnSchema, ColumnType, ColumnValue, GroupBy, InMemoryTable, SortKey, TableFilter,
    TableSort, Predicate,
};
use crate::error::{IoError, Result};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::io::{BufRead, BufReader};

// ─── Extractor trait ─────────────────────────────────────────────────────────

/// Extracts data from a source and returns an [`InMemoryTable`].
pub trait Extractor: Send + Sync {
    /// Extract data and return an [`InMemoryTable`].
    fn extract(&self) -> Result<InMemoryTable>;

    /// Optional human-readable description.
    fn description(&self) -> &str {
        "extractor"
    }
}

// ─── Transformer trait ───────────────────────────────────────────────────────

/// Transforms an [`InMemoryTable`] and returns a new table.
pub trait Transformer: Send + Sync {
    /// Apply the transformation.
    fn transform(&self, table: InMemoryTable) -> Result<InMemoryTable>;

    /// Optional human-readable description.
    fn description(&self) -> &str {
        "transformer"
    }
}

// ─── Loader trait ────────────────────────────────────────────────────────────

/// Loads an [`InMemoryTable`] into a sink.
pub trait Loader: Send + Sync {
    /// Load the table into the target sink.
    fn load(&self, table: &InMemoryTable) -> Result<()>;

    /// Optional human-readable description.
    fn description(&self) -> &str {
        "loader"
    }
}

// ─── ETLPipeline ─────────────────────────────────────────────────────────────

/// A composable ETL pipeline.
///
/// Assembles an optional extractor, a chain of transformers, and an optional loader.
pub struct ETLPipeline {
    extractor: Option<Box<dyn Extractor>>,
    transformers: Vec<Box<dyn Transformer>>,
    loader: Option<Box<dyn Loader>>,
}

impl Default for ETLPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl ETLPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self {
            extractor: None,
            transformers: Vec::new(),
            loader: None,
        }
    }

    /// Set the extractor.
    pub fn extract(mut self, e: impl Extractor + 'static) -> Self {
        self.extractor = Some(Box::new(e));
        self
    }

    /// Add a transformer to the chain.
    pub fn transform(mut self, t: impl Transformer + 'static) -> Self {
        self.transformers.push(Box::new(t));
        self
    }

    /// Set the loader.
    pub fn load_to(mut self, l: impl Loader + 'static) -> Self {
        self.loader = Some(Box::new(l));
        self
    }

    /// Run the pipeline and return the final table.
    ///
    /// 1. Calls the extractor (if set) to get the initial table.
    /// 2. Applies each transformer in order.
    /// 3. Calls the loader (if set) with the final table.
    pub fn run(&self) -> Result<InMemoryTable> {
        let mut table = match &self.extractor {
            Some(e) => e.extract()?,
            None => {
                return Err(IoError::ConfigError(
                    "ETLPipeline has no extractor set".to_string(),
                ))
            }
        };

        for transformer in &self.transformers {
            table = transformer.transform(table)?;
        }

        if let Some(loader) = &self.loader {
            loader.load(&table)?;
        }

        Ok(table)
    }

    /// Run the pipeline starting from a pre-extracted table (skips extractor).
    pub fn run_from(&self, input: InMemoryTable) -> Result<InMemoryTable> {
        let mut table = input;
        for transformer in &self.transformers {
            table = transformer.transform(table)?;
        }
        if let Some(loader) = &self.loader {
            loader.load(&table)?;
        }
        Ok(table)
    }
}

// ─── CSVExtractor ────────────────────────────────────────────────────────────

/// Extracts data from a CSV file.
pub struct CSVExtractor {
    path: String,
    has_header: bool,
    delimiter: char,
    column_types: Option<Vec<(String, ColumnType)>>,
    max_rows: Option<usize>,
    skip_rows: usize,
}

impl CSVExtractor {
    /// Create from a file path.
    pub fn from_path(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            has_header: true,
            delimiter: ',',
            column_types: None,
            max_rows: None,
            skip_rows: 0,
        }
    }

    /// Set whether the file has a header row.
    pub fn has_header(mut self, v: bool) -> Self {
        self.has_header = v;
        self
    }

    /// Set the delimiter character.
    pub fn delimiter(mut self, d: char) -> Self {
        self.delimiter = d;
        self
    }

    /// Override column types.
    pub fn column_types(mut self, types: Vec<(String, ColumnType)>) -> Self {
        self.column_types = Some(types);
        self
    }

    /// Limit the number of data rows read.
    pub fn max_rows(mut self, n: usize) -> Self {
        self.max_rows = Some(n);
        self
    }

    /// Skip `n` rows before the header.
    pub fn skip_rows(mut self, n: usize) -> Self {
        self.skip_rows = n;
        self
    }

    fn split_row(&self, line: &str) -> Vec<String> {
        // Minimal CSV splitter supporting quoted fields
        let mut fields = Vec::new();
        let mut current = String::new();
        let mut in_quotes = false;
        let mut chars = line.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '"' {
                if in_quotes {
                    if chars.peek() == Some(&'"') {
                        // Escaped quote
                        chars.next();
                        current.push('"');
                    } else {
                        in_quotes = false;
                    }
                } else {
                    in_quotes = true;
                }
            } else if c == self.delimiter && !in_quotes {
                fields.push(current.trim().to_string());
                current = String::new();
            } else {
                current.push(c);
            }
        }
        fields.push(current.trim().to_string());
        fields
    }
}

impl Extractor for CSVExtractor {
    fn extract(&self) -> Result<InMemoryTable> {
        let file = std::fs::File::open(&self.path).map_err(IoError::Io)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Skip initial rows
        for _ in 0..self.skip_rows {
            lines.next();
        }

        // Parse header
        let headers: Vec<String> = if self.has_header {
            match lines.next() {
                Some(line) => self.split_row(&line.map_err(IoError::Io)?),
                None => {
                    return Err(IoError::ParseError("CSV file is empty".to_string()));
                }
            }
        } else {
            Vec::new() // Will be filled after first data row
        };

        let col_types: Vec<ColumnType> = if let Some(ref types) = self.column_types {
            types.iter().map(|(_, t)| t.clone()).collect()
        } else {
            vec![] // Will infer as Utf8
        };

        let mut table: Option<InMemoryTable> = None;
        let mut row_count = 0usize;
        let mut header_resolved = false;

        for line_result in lines {
            if let Some(max) = self.max_rows {
                if row_count >= max {
                    break;
                }
            }

            let line = line_result.map_err(IoError::Io)?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let fields = self.split_row(line);

            // First data row: resolve schema
            if !header_resolved {
                let final_headers: Vec<String> = if self.has_header {
                    headers.clone()
                } else {
                    (0..fields.len())
                        .map(|i| format!("col_{i}"))
                        .collect()
                };

                let schema: Vec<(String, ColumnType)> = final_headers
                    .iter()
                    .enumerate()
                    .map(|(i, name)| {
                        let ct = col_types.get(i).cloned().unwrap_or(ColumnType::Utf8);
                        (name.clone(), ct)
                    })
                    .collect();

                table = Some(InMemoryTable::new(schema));
                header_resolved = true;
            }

            if let Some(ref mut t) = table {
                // Pad or truncate fields to column count
                let mut row: Vec<ColumnValue> = Vec::with_capacity(t.column_count());
                for (i, col) in t.columns.iter().enumerate() {
                    let val = fields.get(i).map(|s| s.as_str()).unwrap_or("");
                    row.push(parse_csv_value(val, &col.col_type));
                }
                t.push_row(&row)?;
                row_count += 1;
            }
        }

        match table {
            Some(t) => Ok(t),
            None => {
                // Empty file with header — return empty table
                let schema: Vec<(String, ColumnType)> = headers
                    .iter()
                    .map(|h| (h.clone(), ColumnType::Utf8))
                    .collect();
                Ok(InMemoryTable::new(schema))
            }
        }
    }

    fn description(&self) -> &str {
        "csv_extractor"
    }
}

fn parse_csv_value(s: &str, col_type: &ColumnType) -> ColumnValue {
    if s.is_empty() {
        return ColumnValue::Null;
    }
    match col_type {
        ColumnType::Int64 => s
            .parse::<i64>()
            .map(ColumnValue::Int)
            .unwrap_or(ColumnValue::Null),
        ColumnType::Float64 => s
            .parse::<f64>()
            .map(ColumnValue::Float)
            .unwrap_or(ColumnValue::Null),
        ColumnType::Boolean => match s.to_lowercase().as_str() {
            "true" | "1" | "yes" => ColumnValue::Boolean(true),
            "false" | "0" | "no" => ColumnValue::Boolean(false),
            _ => ColumnValue::Null,
        },
        ColumnType::Utf8 => ColumnValue::Utf8(s.to_string()),
        ColumnType::Nullable(inner) => parse_csv_value(s, inner),
    }
}

// ─── JSONExtractor ───────────────────────────────────────────────────────────

/// Extracts data from a JSON Lines (NDJSON) or JSON array file.
pub struct JSONExtractor {
    path: String,
    is_jsonl: bool,
    column_types: HashMap<String, ColumnType>,
    max_rows: Option<usize>,
}

impl JSONExtractor {
    /// Read a JSON Lines file (one JSON object per line).
    pub fn jsonl(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            is_jsonl: true,
            column_types: HashMap::new(),
            max_rows: None,
        }
    }

    /// Read a JSON array file (`[{...}, {...}, ...]`).
    pub fn json_array(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            is_jsonl: false,
            column_types: HashMap::new(),
            max_rows: None,
        }
    }

    /// Override the column type for a specific column name.
    pub fn column_type(mut self, name: impl Into<String>, ct: ColumnType) -> Self {
        self.column_types.insert(name.into(), ct);
        self
    }

    /// Limit rows.
    pub fn max_rows(mut self, n: usize) -> Self {
        self.max_rows = Some(n);
        self
    }

    fn json_objects_from_path(&self) -> Result<Vec<serde_json::Map<String, JsonValue>>> {
        let content = std::fs::read_to_string(&self.path).map_err(IoError::Io)?;
        if self.is_jsonl {
            let mut objects = Vec::new();
            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let v: JsonValue = serde_json::from_str(line)
                    .map_err(|e| IoError::ParseError(e.to_string()))?;
                if let JsonValue::Object(obj) = v {
                    objects.push(obj);
                }
                if let Some(max) = self.max_rows {
                    if objects.len() >= max {
                        break;
                    }
                }
            }
            Ok(objects)
        } else {
            let v: JsonValue =
                serde_json::from_str(&content).map_err(|e| IoError::ParseError(e.to_string()))?;
            match v {
                JsonValue::Array(arr) => {
                    let limit = self.max_rows.unwrap_or(arr.len()).min(arr.len());
                    Ok(arr[..limit]
                        .iter()
                        .filter_map(|item| {
                            if let JsonValue::Object(obj) = item {
                                Some(obj.clone())
                            } else {
                                None
                            }
                        })
                        .collect())
                }
                _ => Err(IoError::ParseError(
                    "Expected JSON array at top level".to_string(),
                )),
            }
        }
    }
}

impl Extractor for JSONExtractor {
    fn extract(&self) -> Result<InMemoryTable> {
        let objects = self.json_objects_from_path()?;
        if objects.is_empty() {
            return Ok(InMemoryTable::new(vec![]));
        }

        // Collect all column names in order
        let mut col_names: Vec<String> = Vec::new();
        for obj in &objects {
            for key in obj.keys() {
                if !col_names.contains(key) {
                    col_names.push(key.clone());
                }
            }
        }

        // Determine column types
        let schema: Vec<(String, ColumnType)> = col_names
            .iter()
            .map(|name| {
                let ct = self
                    .column_types
                    .get(name)
                    .cloned()
                    .unwrap_or(ColumnType::Utf8);
                (name.clone(), ct)
            })
            .collect();

        let mut table = InMemoryTable::new(schema);

        for obj in &objects {
            let row: Vec<ColumnValue> = table
                .columns
                .iter()
                .map(|col| {
                    obj.get(&col.name)
                        .map(|v| json_to_column_value(v, &col.col_type))
                        .unwrap_or(ColumnValue::Null)
                })
                .collect();
            table.push_row(&row)?;
        }

        Ok(table)
    }

    fn description(&self) -> &str {
        "json_extractor"
    }
}

fn json_to_column_value(v: &JsonValue, ct: &ColumnType) -> ColumnValue {
    match (ct, v) {
        (_, JsonValue::Null) => ColumnValue::Null,
        (ColumnType::Int64, JsonValue::Number(n)) => {
            ColumnValue::Int(n.as_i64().unwrap_or_default())
        }
        (ColumnType::Float64, JsonValue::Number(n)) => {
            ColumnValue::Float(n.as_f64().unwrap_or_default())
        }
        (ColumnType::Boolean, JsonValue::Bool(b)) => ColumnValue::Boolean(*b),
        (ColumnType::Utf8, JsonValue::String(s)) => ColumnValue::Utf8(s.clone()),
        (ColumnType::Utf8, other) => ColumnValue::Utf8(other.to_string()),
        (ColumnType::Nullable(inner), val) => json_to_column_value(val, inner),
        _ => ColumnValue::Utf8(v.to_string()),
    }
}

// ─── ParquetExtractor ────────────────────────────────────────────────────────

/// Extracts data from a Parquet-lite file (scirs2-io native format).
pub struct ParquetExtractor {
    path: String,
    max_rows: Option<usize>,
}

impl ParquetExtractor {
    /// Create from path.
    pub fn from_path(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            max_rows: None,
        }
    }

    /// Limit rows.
    pub fn max_rows(mut self, n: usize) -> Self {
        self.max_rows = Some(n);
        self
    }
}

impl Extractor for ParquetExtractor {
    fn extract(&self) -> Result<InMemoryTable> {
        use crate::parquet_lite::{ColumnData, ColumnType as PqType, ParquetReader};

        let bytes = std::fs::read(&self.path).map_err(IoError::Io)?;
        let (schema, all_col_data) = ParquetReader::read_typed(&bytes)
            .map_err(|e| IoError::ParseError(format!("Parquet read error: {e}")))?;

        let table_schema: Vec<(String, ColumnType)> = schema
            .columns
            .iter()
            .map(|(name, pq_type)| {
                let ct = match pq_type {
                    PqType::Float64 | PqType::Float32 => ColumnType::Float64,
                    PqType::Int64 | PqType::Int32 => ColumnType::Int64,
                    PqType::Boolean => ColumnType::Boolean,
                    _ => ColumnType::Utf8,
                };
                (name.clone(), ct)
            })
            .collect();

        let mut table = InMemoryTable::new(table_schema);

        let row_count = all_col_data
            .first()
            .map(|c| c.len())
            .unwrap_or(0);

        let limit = self.max_rows.unwrap_or(row_count).min(row_count);

        for row_idx in 0..limit {
            let row: Vec<ColumnValue> = all_col_data
                .iter()
                .map(|col| match col {
                    ColumnData::Float64(v) => v
                        .get(row_idx)
                        .copied()
                        .map(ColumnValue::Float)
                        .unwrap_or(ColumnValue::Null),
                    ColumnData::Float32(v) => v
                        .get(row_idx)
                        .copied()
                        .map(|x| ColumnValue::Float(x as f64))
                        .unwrap_or(ColumnValue::Null),
                    ColumnData::Int64(v) => v
                        .get(row_idx)
                        .copied()
                        .map(ColumnValue::Int)
                        .unwrap_or(ColumnValue::Null),
                    ColumnData::Int32(v) => v
                        .get(row_idx)
                        .copied()
                        .map(|x| ColumnValue::Int(x as i64))
                        .unwrap_or(ColumnValue::Null),
                    ColumnData::Boolean(v) => v
                        .get(row_idx)
                        .copied()
                        .map(ColumnValue::Boolean)
                        .unwrap_or(ColumnValue::Null),
                    ColumnData::Utf8(v) => v
                        .get(row_idx)
                        .cloned()
                        .map(ColumnValue::Utf8)
                        .unwrap_or(ColumnValue::Null),
                })
                .collect();
            table.push_row(&row)?;
        }

        Ok(table)
    }

    fn description(&self) -> &str {
        "parquet_extractor"
    }
}
// ─── InMemoryExtractor ───────────────────────────────────────────────────────

/// Wraps an existing [`InMemoryTable`] as an extractor.
pub struct InMemoryExtractor {
    table: InMemoryTable,
}

impl InMemoryExtractor {
    /// Create from an existing table.
    pub fn new(table: InMemoryTable) -> Self {
        Self { table }
    }
}

impl Extractor for InMemoryExtractor {
    fn extract(&self) -> Result<InMemoryTable> {
        Ok(self.table.clone())
    }

    fn description(&self) -> &str {
        "in_memory_extractor"
    }
}

// ─── DeduplicateTransform ────────────────────────────────────────────────────

/// Removes duplicate rows based on a key column or all columns.
pub struct DeduplicateTransform {
    key_columns: Option<Vec<String>>,
}

impl DeduplicateTransform {
    /// Deduplicate on all columns.
    pub fn all_columns() -> Self {
        Self { key_columns: None }
    }

    /// Deduplicate based on a single column.
    pub fn on_column(col: impl Into<String>) -> Self {
        Self {
            key_columns: Some(vec![col.into()]),
        }
    }

    /// Deduplicate based on multiple columns.
    pub fn on_columns(cols: Vec<String>) -> Self {
        Self {
            key_columns: Some(cols),
        }
    }
}

impl Transformer for DeduplicateTransform {
    fn transform(&self, table: InMemoryTable) -> Result<InMemoryTable> {
        let key_indices: Vec<usize> = match &self.key_columns {
            Some(cols) => cols
                .iter()
                .map(|c| {
                    table.column_index(c).ok_or_else(|| {
                        IoError::ValidationError(format!(
                            "Deduplicate column '{}' not found",
                            c
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            None => (0..table.column_count()).collect(),
        };

        let mut seen: std::collections::HashSet<Vec<String>> =
            std::collections::HashSet::new();
        let mut new_rows: Vec<Vec<ColumnValue>> = Vec::new();

        for row in &table.rows {
            let key: Vec<String> = key_indices.iter().map(|&i| row[i].to_string()).collect();
            if seen.insert(key) {
                new_rows.push(row.clone());
            }
        }

        Ok(InMemoryTable {
            columns: table.columns,
            rows: new_rows,
            name: table.name,
        })
    }

    fn description(&self) -> &str {
        "deduplicate_transform"
    }
}

// ─── FillNullTransform ───────────────────────────────────────────────────────

/// Fills NULL values with a constant or aggregated value.
pub struct FillNullTransform {
    column: String,
    strategy: FillNullStrategy,
}

/// Strategy for filling null values.
#[derive(Debug, Clone)]
pub enum FillNullStrategy {
    /// Fill with a constant value
    Constant(ColumnValue),
    /// Fill with the column mean
    Mean,
    /// Fill with the column median
    Median,
    /// Fill forward (use previous non-null value)
    ForwardFill,
    /// Fill backward (use next non-null value)
    BackwardFill,
}

impl FillNullTransform {
    /// Fill nulls in `column` with a constant.
    pub fn with_constant(column: impl Into<String>, value: ColumnValue) -> Self {
        Self {
            column: column.into(),
            strategy: FillNullStrategy::Constant(value),
        }
    }

    /// Fill nulls in `column` with the column mean.
    pub fn with_mean(column: impl Into<String>) -> Self {
        Self {
            column: column.into(),
            strategy: FillNullStrategy::Mean,
        }
    }

    /// Fill nulls in `column` with the column median.
    pub fn with_median(column: impl Into<String>) -> Self {
        Self {
            column: column.into(),
            strategy: FillNullStrategy::Median,
        }
    }

    /// Forward fill.
    pub fn forward_fill(column: impl Into<String>) -> Self {
        Self {
            column: column.into(),
            strategy: FillNullStrategy::ForwardFill,
        }
    }

    /// Backward fill.
    pub fn backward_fill(column: impl Into<String>) -> Self {
        Self {
            column: column.into(),
            strategy: FillNullStrategy::BackwardFill,
        }
    }
}

impl Transformer for FillNullTransform {
    fn transform(&self, mut table: InMemoryTable) -> Result<InMemoryTable> {
        let col_idx = table.column_index(&self.column).ok_or_else(|| {
            IoError::ValidationError(format!(
                "FillNull column '{}' not found",
                self.column
            ))
        })?;

        let fill_value = match &self.strategy {
            FillNullStrategy::Constant(v) => v.clone(),
            FillNullStrategy::Mean => {
                let vals: Vec<f64> = table
                    .rows
                    .iter()
                    .filter_map(|r| r[col_idx].as_f64())
                    .collect();
                if vals.is_empty() {
                    ColumnValue::Null
                } else {
                    ColumnValue::Float(vals.iter().sum::<f64>() / vals.len() as f64)
                }
            }
            FillNullStrategy::Median => {
                let mut vals: Vec<f64> = table
                    .rows
                    .iter()
                    .filter_map(|r| r[col_idx].as_f64())
                    .collect();
                if vals.is_empty() {
                    ColumnValue::Null
                } else {
                    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let mid = vals.len() / 2;
                    let median = if vals.len() % 2 == 0 {
                        (vals[mid - 1] + vals[mid]) / 2.0
                    } else {
                        vals[mid]
                    };
                    ColumnValue::Float(median)
                }
            }
            FillNullStrategy::ForwardFill => {
                let mut last_non_null: Option<ColumnValue> = None;
                for row in table.rows.iter_mut() {
                    if matches!(row[col_idx], ColumnValue::Null) {
                        if let Some(ref fill) = last_non_null {
                            row[col_idx] = fill.clone();
                        }
                    } else {
                        last_non_null = Some(row[col_idx].clone());
                    }
                }
                return Ok(table);
            }
            FillNullStrategy::BackwardFill => {
                let mut next_non_null: Option<ColumnValue> = None;
                for row in table.rows.iter_mut().rev() {
                    if matches!(row[col_idx], ColumnValue::Null) {
                        if let Some(ref fill) = next_non_null {
                            row[col_idx] = fill.clone();
                        }
                    } else {
                        next_non_null = Some(row[col_idx].clone());
                    }
                }
                return Ok(table);
            }
        };

        for row in table.rows.iter_mut() {
            if matches!(row[col_idx], ColumnValue::Null) {
                row[col_idx] = fill_value.clone();
            }
        }

        Ok(table)
    }

    fn description(&self) -> &str {
        "fill_null_transform"
    }
}

// ─── NormalizeTransform ──────────────────────────────────────────────────────

/// Normalizes a numeric column to [0, 1] (min-max) or z-score.
pub struct NormalizeTransform {
    column: String,
    method: NormMethod,
}

/// Normalization method.
#[derive(Debug, Clone)]
pub enum NormMethod {
    /// Min-max scaling to [0, 1]
    MinMax,
    /// Z-score normalization (subtract mean, divide by std)
    ZScore,
    /// Scale to [-1, 1]
    MaxAbs,
}

impl NormalizeTransform {
    /// Min-max normalization.
    pub fn min_max(column: impl Into<String>) -> Self {
        Self {
            column: column.into(),
            method: NormMethod::MinMax,
        }
    }

    /// Z-score normalization.
    pub fn z_score(column: impl Into<String>) -> Self {
        Self {
            column: column.into(),
            method: NormMethod::ZScore,
        }
    }

    /// Max-abs normalization to [-1, 1].
    pub fn max_abs(column: impl Into<String>) -> Self {
        Self {
            column: column.into(),
            method: NormMethod::MaxAbs,
        }
    }
}

impl Transformer for NormalizeTransform {
    fn transform(&self, mut table: InMemoryTable) -> Result<InMemoryTable> {
        let col_idx = table.column_index(&self.column).ok_or_else(|| {
            IoError::ValidationError(format!(
                "Normalize column '{}' not found",
                self.column
            ))
        })?;

        let vals: Vec<f64> = table
            .rows
            .iter()
            .filter_map(|r| r[col_idx].as_f64())
            .collect();

        if vals.is_empty() {
            return Ok(table);
        }

        match &self.method {
            NormMethod::MinMax => {
                let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let range = max - min;
                if range.abs() < f64::EPSILON {
                    return Ok(table); // All same value; no-op
                }
                for row in table.rows.iter_mut() {
                    if let Some(f) = row[col_idx].as_f64() {
                        row[col_idx] = ColumnValue::Float((f - min) / range);
                    }
                }
            }
            NormMethod::ZScore => {
                let n = vals.len() as f64;
                let mean = vals.iter().sum::<f64>() / n;
                let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
                let std_dev = variance.sqrt();
                if std_dev < f64::EPSILON {
                    return Ok(table);
                }
                for row in table.rows.iter_mut() {
                    if let Some(f) = row[col_idx].as_f64() {
                        row[col_idx] = ColumnValue::Float((f - mean) / std_dev);
                    }
                }
            }
            NormMethod::MaxAbs => {
                let max_abs = vals.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
                if max_abs < f64::EPSILON {
                    return Ok(table);
                }
                for row in table.rows.iter_mut() {
                    if let Some(f) = row[col_idx].as_f64() {
                        row[col_idx] = ColumnValue::Float(f / max_abs);
                    }
                }
            }
        }

        Ok(table)
    }

    fn description(&self) -> &str {
        "normalize_transform"
    }
}

// ─── FilterTransform ────────────────────────────────────────────────────────

/// Apply a predicate-based row filter as a transformer step.
pub struct FilterTransform {
    predicate: Predicate,
}

impl FilterTransform {
    /// Create a filter with the given predicate.
    pub fn new(predicate: Predicate) -> Self {
        Self { predicate }
    }
}

impl Transformer for FilterTransform {
    fn transform(&self, table: InMemoryTable) -> Result<InMemoryTable> {
        TableFilter::new(&table)
            .predicate(self.predicate.clone())
            .apply()
    }

    fn description(&self) -> &str {
        "filter_transform"
    }
}

// ─── SortTransform ──────────────────────────────────────────────────────────

/// Sort the table as a transformer step.
pub struct SortTransform {
    keys: Vec<SortKey>,
}

impl SortTransform {
    /// Create a sort transform with the given keys.
    pub fn new(keys: Vec<SortKey>) -> Self {
        Self { keys }
    }
}

impl Transformer for SortTransform {
    fn transform(&self, table: InMemoryTable) -> Result<InMemoryTable> {
        TableSort::sort(&table, &self.keys)
    }

    fn description(&self) -> &str {
        "sort_transform"
    }
}

// ─── RenameTransform ────────────────────────────────────────────────────────

/// Rename columns in the table.
pub struct RenameTransform {
    renames: HashMap<String, String>,
}

impl RenameTransform {
    /// Create a rename transformer.
    pub fn new(renames: HashMap<String, String>) -> Self {
        Self { renames }
    }

    /// Single column rename.
    pub fn one(from: impl Into<String>, to: impl Into<String>) -> Self {
        let mut m = HashMap::new();
        m.insert(from.into(), to.into());
        Self { renames: m }
    }
}

impl Transformer for RenameTransform {
    fn transform(&self, mut table: InMemoryTable) -> Result<InMemoryTable> {
        for col in table.columns.iter_mut() {
            if let Some(new_name) = self.renames.get(&col.name) {
                col.name = new_name.clone();
            }
        }
        Ok(table)
    }

    fn description(&self) -> &str {
        "rename_transform"
    }
}

// ─── CastTransform ──────────────────────────────────────────────────────────

/// Cast a column to a different type.
pub struct CastTransform {
    column: String,
    target_type: ColumnType,
}

impl CastTransform {
    /// Create a cast transformer.
    pub fn new(column: impl Into<String>, target_type: ColumnType) -> Self {
        Self {
            column: column.into(),
            target_type,
        }
    }
}

impl Transformer for CastTransform {
    fn transform(&self, mut table: InMemoryTable) -> Result<InMemoryTable> {
        let col_idx = table.column_index(&self.column).ok_or_else(|| {
            IoError::ValidationError(format!("Cast column '{}' not found", self.column))
        })?;

        for row in table.rows.iter_mut() {
            let new_val = cast_value(&row[col_idx], &self.target_type);
            row[col_idx] = new_val;
        }

        // Update schema type
        table.columns[col_idx].col_type = self.target_type.clone();
        Ok(table)
    }

    fn description(&self) -> &str {
        "cast_transform"
    }
}

fn cast_value(val: &ColumnValue, target: &ColumnType) -> ColumnValue {
    match (target, val) {
        (_, ColumnValue::Null) => ColumnValue::Null,
        (ColumnType::Int64, ColumnValue::Float(f)) => ColumnValue::Int(*f as i64),
        (ColumnType::Int64, ColumnValue::Boolean(b)) => ColumnValue::Int(if *b { 1 } else { 0 }),
        (ColumnType::Int64, ColumnValue::Utf8(s)) => s
            .parse::<i64>()
            .map(ColumnValue::Int)
            .unwrap_or(ColumnValue::Null),
        (ColumnType::Float64, ColumnValue::Int(i)) => ColumnValue::Float(*i as f64),
        (ColumnType::Float64, ColumnValue::Boolean(b)) => {
            ColumnValue::Float(if *b { 1.0 } else { 0.0 })
        }
        (ColumnType::Float64, ColumnValue::Utf8(s)) => s
            .parse::<f64>()
            .map(ColumnValue::Float)
            .unwrap_or(ColumnValue::Null),
        (ColumnType::Boolean, ColumnValue::Int(i)) => ColumnValue::Boolean(*i != 0),
        (ColumnType::Boolean, ColumnValue::Float(f)) => ColumnValue::Boolean(*f != 0.0),
        (ColumnType::Boolean, ColumnValue::Utf8(s)) => match s.to_lowercase().as_str() {
            "true" | "1" | "yes" => ColumnValue::Boolean(true),
            _ => ColumnValue::Boolean(false),
        },
        (ColumnType::Utf8, v) => ColumnValue::Utf8(v.to_string()),
        _ => val.clone(),
    }
}

// ─── AggregateTransform ──────────────────────────────────────────────────────

/// Group-by aggregation as a transformer step.
pub struct AggregateTransform {
    group_cols: Vec<String>,
    agg_funcs: Vec<AggFunc>,
}

impl AggregateTransform {
    /// Create an aggregate transformer.
    pub fn new(group_cols: Vec<String>, agg_funcs: Vec<AggFunc>) -> Self {
        Self {
            group_cols,
            agg_funcs,
        }
    }
}

impl Transformer for AggregateTransform {
    fn transform(&self, table: InMemoryTable) -> Result<InMemoryTable> {
        let mut gb = GroupBy::new(&table, self.group_cols.clone());
        for f in &self.agg_funcs {
            gb = gb.agg(f.clone());
        }
        gb.apply()
    }

    fn description(&self) -> &str {
        "aggregate_transform"
    }
}

// ─── InMemoryLoader ──────────────────────────────────────────────────────────

/// Loads the result into an [`InMemoryTable`] stored in a `Mutex`.
pub struct InMemoryLoader {
    target: std::sync::Arc<std::sync::Mutex<Option<InMemoryTable>>>,
}

impl InMemoryLoader {
    /// Create a new in-memory loader.
    pub fn new() -> (Self, std::sync::Arc<std::sync::Mutex<Option<InMemoryTable>>>) {
        let target = std::sync::Arc::new(std::sync::Mutex::new(None));
        let loader = Self {
            target: target.clone(),
        };
        (loader, target)
    }
}

impl Default for InMemoryLoader {
    fn default() -> Self {
        Self {
            target: std::sync::Arc::new(std::sync::Mutex::new(None)),
        }
    }
}

impl Loader for InMemoryLoader {
    fn load(&self, table: &InMemoryTable) -> Result<()> {
        let mut guard = self
            .target
            .lock()
            .map_err(|e| IoError::Other(format!("InMemoryLoader lock error: {e}")))?;
        *guard = Some(table.clone());
        Ok(())
    }

    fn description(&self) -> &str {
        "in_memory_loader"
    }
}

// ─── CSVLoader ───────────────────────────────────────────────────────────────

/// Saves the result to a CSV file.
pub struct CSVLoader {
    path: String,
    write_header: bool,
    delimiter: char,
}

impl CSVLoader {
    /// Create a CSV loader.
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            write_header: true,
            delimiter: ',',
        }
    }

    /// Set whether to write a header row.
    pub fn write_header(mut self, v: bool) -> Self {
        self.write_header = v;
        self
    }

    /// Set the delimiter.
    pub fn delimiter(mut self, d: char) -> Self {
        self.delimiter = d;
        self
    }
}

impl Loader for CSVLoader {
    fn load(&self, table: &InMemoryTable) -> Result<()> {
        use std::io::Write;
        let file = std::fs::File::create(&self.path).map_err(IoError::Io)?;
        let mut writer = std::io::BufWriter::new(file);

        if self.write_header {
            let header: Vec<String> = table.columns.iter().map(|c| c.name.clone()).collect();
            writeln!(writer, "{}", header.join(&self.delimiter.to_string()))
                .map_err(IoError::Io)?;
        }

        for row in &table.rows {
            let fields: Vec<String> = row
                .iter()
                .map(|v| {
                    let s = v.to_string();
                    if s.contains(self.delimiter) || s.contains('"') || s.contains('\n') {
                        format!("\"{}\"", s.replace('"', "\"\""))
                    } else {
                        s
                    }
                })
                .collect();
            writeln!(writer, "{}", fields.join(&self.delimiter.to_string()))
                .map_err(IoError::Io)?;
        }

        writer.flush().map_err(IoError::Io)?;
        Ok(())
    }

    fn description(&self) -> &str {
        "csv_loader"
    }
}

// ─── JSONLinesLoader ─────────────────────────────────────────────────────────

/// Saves the result to a JSON Lines file.
pub struct JSONLinesLoader {
    path: String,
}

impl JSONLinesLoader {
    /// Create a JSON Lines loader.
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }
}

impl Loader for JSONLinesLoader {
    fn load(&self, table: &InMemoryTable) -> Result<()> {
        use std::io::Write;
        let file = std::fs::File::create(&self.path).map_err(IoError::Io)?;
        let mut writer = std::io::BufWriter::new(file);

        for row in &table.rows {
            let obj: serde_json::Map<String, JsonValue> = table
                .columns
                .iter()
                .zip(row.iter())
                .map(|(col, val)| (col.name.clone(), val.to_json()))
                .collect();
            let line = serde_json::to_string(&obj)
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
            writeln!(writer, "{line}").map_err(IoError::Io)?;
        }
        writer.flush().map_err(IoError::Io)?;
        Ok(())
    }

    fn description(&self) -> &str {
        "jsonlines_loader"
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_csv(content: &str) -> String {
        let path = std::env::temp_dir()
            .join(format!("etl_test_{}.csv", uuid::Uuid::new_v4()))
            .to_str()
            .expect("temp path")
            .to_string();
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn test_csv_extractor_basic() {
        let content = "id,name,score\n1,Alice,95.0\n2,Bob,82.5\n3,Carol,91.0\n";
        let path = write_temp_csv(content);
        let extractor = CSVExtractor::from_path(&path).has_header(true);
        let table = extractor.extract().unwrap();
        assert_eq!(table.row_count(), 3);
        assert_eq!(table.column_count(), 3);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_deduplicate_transform() {
        let mut t = InMemoryTable::new(vec![
            ("id".to_string(), ColumnType::Int64),
            ("val".to_string(), ColumnType::Float64),
        ]);
        t.push_row(&[ColumnValue::Int(1), ColumnValue::Float(1.0)]).unwrap();
        t.push_row(&[ColumnValue::Int(1), ColumnValue::Float(1.0)]).unwrap(); // dup
        t.push_row(&[ColumnValue::Int(2), ColumnValue::Float(2.0)]).unwrap();

        let dedup = DeduplicateTransform::on_column("id");
        let result = dedup.transform(t).unwrap();
        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn test_fill_null_constant() {
        let mut t = InMemoryTable::new(vec![
            ("x".to_string(), ColumnType::Float64),
        ]);
        t.push_row(&[ColumnValue::Float(1.0)]).unwrap();
        t.push_row(&[ColumnValue::Null]).unwrap();
        t.push_row(&[ColumnValue::Float(3.0)]).unwrap();

        let transform = FillNullTransform::with_constant("x", ColumnValue::Float(0.0));
        let result = transform.transform(t).unwrap();
        assert_eq!(result.rows[1][0], ColumnValue::Float(0.0));
    }

    #[test]
    fn test_fill_null_mean() {
        let mut t = InMemoryTable::new(vec![("x".to_string(), ColumnType::Float64)]);
        t.push_row(&[ColumnValue::Float(2.0)]).unwrap();
        t.push_row(&[ColumnValue::Null]).unwrap();
        t.push_row(&[ColumnValue::Float(4.0)]).unwrap();

        let transform = FillNullTransform::with_mean("x");
        let result = transform.transform(t).unwrap();
        if let ColumnValue::Float(v) = result.rows[1][0] {
            assert!((v - 3.0).abs() < 1e-9, "expected mean 3.0, got {v}");
        } else {
            panic!("Expected float");
        }
    }

    #[test]
    fn test_fill_null_forward_fill() {
        let mut t = InMemoryTable::new(vec![("x".to_string(), ColumnType::Float64)]);
        t.push_row(&[ColumnValue::Float(5.0)]).unwrap();
        t.push_row(&[ColumnValue::Null]).unwrap();
        t.push_row(&[ColumnValue::Null]).unwrap();
        t.push_row(&[ColumnValue::Float(9.0)]).unwrap();

        let transform = FillNullTransform::forward_fill("x");
        let result = transform.transform(t).unwrap();
        assert_eq!(result.rows[1][0], ColumnValue::Float(5.0));
        assert_eq!(result.rows[2][0], ColumnValue::Float(5.0));
        assert_eq!(result.rows[3][0], ColumnValue::Float(9.0));
    }

    #[test]
    fn test_normalize_min_max() {
        let mut t = InMemoryTable::new(vec![("v".to_string(), ColumnType::Float64)]);
        t.push_row(&[ColumnValue::Float(0.0)]).unwrap();
        t.push_row(&[ColumnValue::Float(5.0)]).unwrap();
        t.push_row(&[ColumnValue::Float(10.0)]).unwrap();

        let norm = NormalizeTransform::min_max("v");
        let result = norm.transform(t).unwrap();
        assert_eq!(result.rows[0][0], ColumnValue::Float(0.0));
        assert_eq!(result.rows[1][0], ColumnValue::Float(0.5));
        assert_eq!(result.rows[2][0], ColumnValue::Float(1.0));
    }

    #[test]
    fn test_normalize_z_score() {
        let mut t = InMemoryTable::new(vec![("v".to_string(), ColumnType::Float64)]);
        // Mean=2, Std=1
        t.push_row(&[ColumnValue::Float(1.0)]).unwrap();
        t.push_row(&[ColumnValue::Float(2.0)]).unwrap();
        t.push_row(&[ColumnValue::Float(3.0)]).unwrap();

        let norm = NormalizeTransform::z_score("v");
        let result = norm.transform(t).unwrap();
        if let ColumnValue::Float(v) = result.rows[1][0] {
            assert!(v.abs() < 1e-9, "z-score of mean should be ~0, got {v}");
        }
    }

    #[test]
    fn test_cast_transform_float_to_int() {
        let mut t = InMemoryTable::new(vec![("x".to_string(), ColumnType::Float64)]);
        t.push_row(&[ColumnValue::Float(3.7)]).unwrap();

        let cast = CastTransform::new("x", ColumnType::Int64);
        let result = cast.transform(t).unwrap();
        assert_eq!(result.rows[0][0], ColumnValue::Int(3));
    }

    #[test]
    fn test_pipeline_csv_to_normalize() {
        let content = "x\n1.0\n2.0\n3.0\n4.0\n5.0\n";
        let path = write_temp_csv(content);

        let pipeline = ETLPipeline::new()
            .extract(
                CSVExtractor::from_path(&path)
                    .has_header(true)
                    .column_types(vec![("x".to_string(), ColumnType::Float64)]),
            )
            .transform(NormalizeTransform::min_max("x"));

        let table = pipeline.run().unwrap();
        assert_eq!(table.row_count(), 5);
        assert_eq!(table.rows[0][0], ColumnValue::Float(0.0));
        assert_eq!(table.rows[4][0], ColumnValue::Float(1.0));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_pipeline_csv_loader_roundtrip() {
        let content = "id,name\n1,Alice\n2,Bob\n";
        let src_path = write_temp_csv(content);
        let dst_path = std::env::temp_dir()
            .join(format!("etl_out_{}.csv", uuid::Uuid::new_v4()))
            .to_str()
            .expect("temp path")
            .to_string();

        let (loader, _) = InMemoryLoader::new();
        let _ = ETLPipeline::new()
            .extract(CSVExtractor::from_path(&src_path).has_header(true))
            .load_to(CSVLoader::new(&dst_path))
            .run()
            .unwrap();

        // Verify output file
        let content_out = std::fs::read_to_string(&dst_path).unwrap();
        assert!(content_out.contains("id,name"));
        assert!(content_out.contains("Alice"));

        let _ = std::fs::remove_file(&src_path);
        let _ = std::fs::remove_file(&dst_path);
        drop(loader);
    }

    #[test]
    fn test_rename_transform() {
        let mut t = InMemoryTable::new(vec![
            ("old_name".to_string(), ColumnType::Utf8),
        ]);
        t.push_row(&[ColumnValue::Utf8("Alice".to_string())]).unwrap();

        let rename = RenameTransform::one("old_name", "new_name");
        let result = rename.transform(t).unwrap();
        assert_eq!(result.columns[0].name, "new_name");
    }

    #[test]
    fn test_aggregate_transform() {
        let mut t = InMemoryTable::new(vec![
            ("dept".to_string(), ColumnType::Utf8),
            ("salary".to_string(), ColumnType::Float64),
        ]);
        t.push_row(&[ColumnValue::Utf8("eng".to_string()), ColumnValue::Float(100.0)]).unwrap();
        t.push_row(&[ColumnValue::Utf8("eng".to_string()), ColumnValue::Float(200.0)]).unwrap();
        t.push_row(&[ColumnValue::Utf8("hr".to_string()), ColumnValue::Float(80.0)]).unwrap();

        let agg = AggregateTransform::new(
            vec!["dept".to_string()],
            vec![AggFunc::Sum("salary".to_string()), AggFunc::Count],
        );
        let result = agg.transform(t).unwrap();
        assert_eq!(result.row_count(), 2);
    }
}
