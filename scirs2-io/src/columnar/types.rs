//! Column types and data structures for the columnar format.
//!
//! Defines the core types for column-oriented storage including
//! column data variants, encoding strategies, and table structures.

use std::collections::HashMap;
use std::fmt;

use crate::error::{IoError, Result};

/// Magic bytes identifying the columnar format file
pub const COLUMNAR_MAGIC: &[u8; 8] = b"SCIRCOL\x01";

/// Current format version
pub const FORMAT_VERSION: u32 = 1;

/// Column data type tag stored in the file header
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ColumnTypeTag {
    /// 64-bit floating point
    Float64 = 0,
    /// 64-bit signed integer
    Int64 = 1,
    /// UTF-8 string
    Str = 2,
    /// Boolean
    Bool = 3,
}

impl TryFrom<u8> for ColumnTypeTag {
    type Error = IoError;

    fn try_from(value: u8) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(ColumnTypeTag::Float64),
            1 => Ok(ColumnTypeTag::Int64),
            2 => Ok(ColumnTypeTag::Str),
            3 => Ok(ColumnTypeTag::Bool),
            _ => Err(IoError::FormatError(format!(
                "Unknown column type tag: {}",
                value
            ))),
        }
    }
}

/// Encoding strategy for a column
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EncodingType {
    /// Plain (no encoding)
    Plain = 0,
    /// Run-length encoding
    Rle = 1,
    /// Dictionary encoding
    Dictionary = 2,
    /// Delta encoding (for sorted numeric columns)
    Delta = 3,
}

impl TryFrom<u8> for EncodingType {
    type Error = IoError;

    fn try_from(value: u8) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(EncodingType::Plain),
            1 => Ok(EncodingType::Rle),
            2 => Ok(EncodingType::Dictionary),
            3 => Ok(EncodingType::Delta),
            _ => Err(IoError::FormatError(format!(
                "Unknown encoding type: {}",
                value
            ))),
        }
    }
}

/// A single column's data
#[derive(Debug, Clone)]
pub enum ColumnData {
    /// 64-bit floating point values
    Float64(Vec<f64>),
    /// 64-bit signed integer values
    Int64(Vec<i64>),
    /// UTF-8 string values
    Str(Vec<String>),
    /// Boolean values
    Bool(Vec<bool>),
}

impl ColumnData {
    /// Returns the number of values in this column
    pub fn len(&self) -> usize {
        match self {
            ColumnData::Float64(v) => v.len(),
            ColumnData::Int64(v) => v.len(),
            ColumnData::Str(v) => v.len(),
            ColumnData::Bool(v) => v.len(),
        }
    }

    /// Returns true if the column is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the column type tag
    pub fn type_tag(&self) -> ColumnTypeTag {
        match self {
            ColumnData::Float64(_) => ColumnTypeTag::Float64,
            ColumnData::Int64(_) => ColumnTypeTag::Int64,
            ColumnData::Str(_) => ColumnTypeTag::Str,
            ColumnData::Bool(_) => ColumnTypeTag::Bool,
        }
    }

    /// Try to get f64 data
    pub fn as_f64(&self) -> Result<&[f64]> {
        match self {
            ColumnData::Float64(v) => Ok(v),
            _ => Err(IoError::ConversionError(format!(
                "Column is {:?}, not Float64",
                self.type_tag()
            ))),
        }
    }

    /// Try to get i64 data
    pub fn as_i64(&self) -> Result<&[i64]> {
        match self {
            ColumnData::Int64(v) => Ok(v),
            _ => Err(IoError::ConversionError(format!(
                "Column is {:?}, not Int64",
                self.type_tag()
            ))),
        }
    }

    /// Try to get string data
    pub fn as_str(&self) -> Result<&[String]> {
        match self {
            ColumnData::Str(v) => Ok(v),
            _ => Err(IoError::ConversionError(format!(
                "Column is {:?}, not Str",
                self.type_tag()
            ))),
        }
    }

    /// Try to get bool data
    pub fn as_bool(&self) -> Result<&[bool]> {
        match self {
            ColumnData::Bool(v) => Ok(v),
            _ => Err(IoError::ConversionError(format!(
                "Column is {:?}, not Bool",
                self.type_tag()
            ))),
        }
    }

    /// Determine best encoding for this column's data
    pub fn best_encoding(&self) -> EncodingType {
        match self {
            ColumnData::Float64(v) => {
                if is_sorted_f64(v) {
                    EncodingType::Delta
                } else if has_runs_f64(v) {
                    EncodingType::Rle
                } else {
                    EncodingType::Plain
                }
            }
            ColumnData::Int64(v) => {
                if is_sorted_i64(v) {
                    EncodingType::Delta
                } else if has_runs_i64(v) {
                    EncodingType::Rle
                } else {
                    EncodingType::Plain
                }
            }
            ColumnData::Str(v) => {
                let unique_count = count_unique_strings(v);
                if unique_count < v.len() / 2 {
                    EncodingType::Dictionary
                } else if has_runs_str(v) {
                    EncodingType::Rle
                } else {
                    EncodingType::Plain
                }
            }
            ColumnData::Bool(v) => {
                if has_runs_bool(v) {
                    EncodingType::Rle
                } else {
                    EncodingType::Plain
                }
            }
        }
    }
}

impl fmt::Display for ColumnData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ColumnData::Float64(v) => write!(f, "Float64[{}]", v.len()),
            ColumnData::Int64(v) => write!(f, "Int64[{}]", v.len()),
            ColumnData::Str(v) => write!(f, "Str[{}]", v.len()),
            ColumnData::Bool(v) => write!(f, "Bool[{}]", v.len()),
        }
    }
}

/// A named column in a table
#[derive(Debug, Clone)]
pub struct Column {
    /// Column name
    pub name: String,
    /// Column data
    pub data: ColumnData,
}

impl Column {
    /// Create a new column with f64 data
    pub fn float64(name: impl Into<String>, data: Vec<f64>) -> Self {
        Column {
            name: name.into(),
            data: ColumnData::Float64(data),
        }
    }

    /// Create a new column with i64 data
    pub fn int64(name: impl Into<String>, data: Vec<i64>) -> Self {
        Column {
            name: name.into(),
            data: ColumnData::Int64(data),
        }
    }

    /// Create a new column with string data
    pub fn string(name: impl Into<String>, data: Vec<String>) -> Self {
        Column {
            name: name.into(),
            data: ColumnData::Str(data),
        }
    }

    /// Create a new column with bool data
    pub fn boolean(name: impl Into<String>, data: Vec<bool>) -> Self {
        Column {
            name: name.into(),
            data: ColumnData::Bool(data),
        }
    }

    /// Returns the length (number of rows) of this column
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if column is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// A table containing multiple named columns
#[derive(Debug, Clone)]
pub struct ColumnarTable {
    /// Columns in order
    columns: Vec<Column>,
    /// Name-to-index lookup
    index: HashMap<String, usize>,
}

impl ColumnarTable {
    /// Create a new empty table
    pub fn new() -> Self {
        ColumnarTable {
            columns: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// Create a table from a list of columns
    pub fn from_columns(columns: Vec<Column>) -> Result<Self> {
        // Validate all columns have same length
        if !columns.is_empty() {
            let expected_len = columns[0].len();
            for col in &columns[1..] {
                if col.len() != expected_len {
                    return Err(IoError::FormatError(format!(
                        "Column '{}' has {} rows, expected {}",
                        col.name,
                        col.len(),
                        expected_len
                    )));
                }
            }
        }

        let mut index = HashMap::new();
        for (i, col) in columns.iter().enumerate() {
            if index.contains_key(&col.name) {
                return Err(IoError::FormatError(format!(
                    "Duplicate column name: '{}'",
                    col.name
                )));
            }
            index.insert(col.name.clone(), i);
        }

        Ok(ColumnarTable { columns, index })
    }

    /// Add a column to the table
    pub fn add_column(&mut self, column: Column) -> Result<()> {
        if !self.columns.is_empty() && column.len() != self.num_rows() {
            return Err(IoError::FormatError(format!(
                "Column '{}' has {} rows, expected {}",
                column.name,
                column.len(),
                self.num_rows()
            )));
        }
        if self.index.contains_key(&column.name) {
            return Err(IoError::FormatError(format!(
                "Duplicate column name: '{}'",
                column.name
            )));
        }
        let idx = self.columns.len();
        self.index.insert(column.name.clone(), idx);
        self.columns.push(column);
        Ok(())
    }

    /// Number of rows in the table
    pub fn num_rows(&self) -> usize {
        self.columns.first().map(|c| c.len()).unwrap_or(0)
    }

    /// Number of columns
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Get column names in order
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|c| c.name.as_str()).collect()
    }

    /// Get a column by name
    pub fn column(&self, name: &str) -> Result<&Column> {
        self.index
            .get(name)
            .map(|&idx| &self.columns[idx])
            .ok_or_else(|| IoError::NotFound(format!("Column '{}' not found", name)))
    }

    /// Get a column by index
    pub fn column_by_index(&self, idx: usize) -> Result<&Column> {
        self.columns
            .get(idx)
            .ok_or_else(|| IoError::NotFound(format!("Column index {} out of range", idx)))
    }

    /// Get all columns as a slice
    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    /// Get f64 column data by name
    pub fn get_f64(&self, name: &str) -> Result<&[f64]> {
        self.column(name)?.data.as_f64()
    }

    /// Get i64 column data by name
    pub fn get_i64(&self, name: &str) -> Result<&[i64]> {
        self.column(name)?.data.as_i64()
    }

    /// Get string column data by name
    pub fn get_str(&self, name: &str) -> Result<&[String]> {
        self.column(name)?.data.as_str()
    }

    /// Get bool column data by name
    pub fn get_bool(&self, name: &str) -> Result<&[bool]> {
        self.column(name)?.data.as_bool()
    }
}

impl Default for ColumnarTable {
    fn default() -> Self {
        Self::new()
    }
}

// Helper functions for encoding detection

fn is_sorted_f64(data: &[f64]) -> bool {
    if data.len() < 2 {
        return true;
    }
    data.windows(2).all(|w| w[0] <= w[1])
}

fn is_sorted_i64(data: &[i64]) -> bool {
    if data.len() < 2 {
        return true;
    }
    data.windows(2).all(|w| w[0] <= w[1])
}

fn has_runs_f64(data: &[f64]) -> bool {
    if data.len() < 4 {
        return false;
    }
    let mut run_count = 0;
    let mut i = 0;
    while i < data.len() {
        let val = data[i];
        let mut run_len = 1;
        while i + run_len < data.len() && data[i + run_len] == val {
            run_len += 1;
        }
        if run_len > 1 {
            run_count += 1;
        }
        i += run_len;
    }
    // Beneficial if at least 20% of groups are runs
    run_count * 5 >= data.len()
}

fn has_runs_i64(data: &[i64]) -> bool {
    if data.len() < 4 {
        return false;
    }
    let mut run_count = 0;
    let mut i = 0;
    while i < data.len() {
        let val = data[i];
        let mut run_len = 1;
        while i + run_len < data.len() && data[i + run_len] == val {
            run_len += 1;
        }
        if run_len > 1 {
            run_count += 1;
        }
        i += run_len;
    }
    run_count * 5 >= data.len()
}

fn has_runs_str(data: &[String]) -> bool {
    if data.len() < 4 {
        return false;
    }
    let mut run_count = 0;
    let mut i = 0;
    while i < data.len() {
        let val = &data[i];
        let mut run_len = 1;
        while i + run_len < data.len() && &data[i + run_len] == val {
            run_len += 1;
        }
        if run_len > 1 {
            run_count += 1;
        }
        i += run_len;
    }
    run_count * 5 >= data.len()
}

fn has_runs_bool(data: &[bool]) -> bool {
    if data.len() < 4 {
        return false;
    }
    let mut run_count = 0;
    let mut i = 0;
    while i < data.len() {
        let val = data[i];
        let mut run_len = 1;
        while i + run_len < data.len() && data[i + run_len] == val {
            run_len += 1;
        }
        if run_len > 1 {
            run_count += 1;
        }
        i += run_len;
    }
    run_count * 5 >= data.len()
}

fn count_unique_strings(data: &[String]) -> usize {
    let mut seen = std::collections::HashSet::new();
    for s in data {
        seen.insert(s.as_str());
    }
    seen.len()
}
