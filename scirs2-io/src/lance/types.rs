//! Lance columnar format types: schema, fields, columns, and batches.

use std::collections::HashMap;

/// Data types supported by the Lance columnar format.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LanceDataType {
    /// 32-bit float
    Float32,
    /// 64-bit float
    Float64,
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
    /// 32-bit unsigned integer
    UInt32,
    /// 64-bit unsigned integer
    UInt64,
    /// UTF-8 string
    Utf8,
    /// Boolean
    Boolean,
}

impl LanceDataType {
    /// Wire type tag byte used in the binary format.
    pub fn type_tag(&self) -> u8 {
        match self {
            LanceDataType::Float32 => 0,
            LanceDataType::Float64 => 1,
            LanceDataType::Int32 => 2,
            LanceDataType::Int64 => 3,
            LanceDataType::UInt32 => 4,
            LanceDataType::UInt64 => 5,
            LanceDataType::Utf8 => 6,
            LanceDataType::Boolean => 7,
        }
    }

    /// Parse a type tag byte back to a `LanceDataType`.
    pub fn from_type_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(LanceDataType::Float32),
            1 => Some(LanceDataType::Float64),
            2 => Some(LanceDataType::Int32),
            3 => Some(LanceDataType::Int64),
            4 => Some(LanceDataType::UInt32),
            5 => Some(LanceDataType::UInt64),
            6 => Some(LanceDataType::Utf8),
            7 => Some(LanceDataType::Boolean),
            _ => None,
        }
    }
}

/// Describes a single column in a Lance schema.
#[non_exhaustive]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LanceField {
    /// Column name.
    pub name: String,
    /// Column data type.
    pub dtype: LanceDataType,
    /// Whether the column may contain null values.
    pub nullable: bool,
}

impl LanceField {
    /// Create a non-nullable field.
    pub fn new(name: impl Into<String>, dtype: LanceDataType) -> Self {
        Self {
            name: name.into(),
            dtype,
            nullable: false,
        }
    }

    /// Create a nullable field.
    pub fn nullable(name: impl Into<String>, dtype: LanceDataType) -> Self {
        Self {
            name: name.into(),
            dtype,
            nullable: true,
        }
    }
}

/// Schema for a Lance dataset.
#[non_exhaustive]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LanceSchema {
    /// Ordered list of fields.
    pub fields: Vec<LanceField>,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl LanceSchema {
    /// Create a schema with the given fields and no metadata.
    pub fn new(fields: Vec<LanceField>) -> Self {
        Self {
            fields,
            metadata: HashMap::new(),
        }
    }

    /// Create a schema with fields and metadata.
    pub fn with_metadata(fields: Vec<LanceField>, metadata: HashMap<String, String>) -> Self {
        Self { fields, metadata }
    }
}

impl Default for LanceSchema {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

/// A typed column of values.
///
/// Each variant holds a `Vec` of the corresponding primitive type.
/// Use `Nullable(inner, validity)` to wrap any variant with per-row validity bits.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum LanceColumn {
    /// 32-bit floats
    Float32(Vec<f32>),
    /// 64-bit floats
    Float64(Vec<f64>),
    /// 32-bit signed integers
    Int32(Vec<i32>),
    /// 64-bit signed integers
    Int64(Vec<i64>),
    /// 32-bit unsigned integers
    UInt32(Vec<u32>),
    /// 64-bit unsigned integers
    UInt64(Vec<u64>),
    /// UTF-8 strings
    Utf8(Vec<String>),
    /// Booleans
    Boolean(Vec<bool>),
    /// Nullable wrapper: (inner column, validity bitmap; true = valid).
    Nullable(Box<LanceColumn>, Vec<bool>),
}

impl LanceColumn {
    /// Number of rows in the column.
    pub fn len(&self) -> usize {
        match self {
            LanceColumn::Float32(v) => v.len(),
            LanceColumn::Float64(v) => v.len(),
            LanceColumn::Int32(v) => v.len(),
            LanceColumn::Int64(v) => v.len(),
            LanceColumn::UInt32(v) => v.len(),
            LanceColumn::UInt64(v) => v.len(),
            LanceColumn::Utf8(v) => v.len(),
            LanceColumn::Boolean(v) => v.len(),
            LanceColumn::Nullable(inner, _) => inner.len(),
        }
    }

    /// Returns `true` if the column has no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the `LanceDataType` tag for this column.
    pub fn data_type(&self) -> LanceDataType {
        match self {
            LanceColumn::Float32(_) => LanceDataType::Float32,
            LanceColumn::Float64(_) => LanceDataType::Float64,
            LanceColumn::Int32(_) => LanceDataType::Int32,
            LanceColumn::Int64(_) => LanceDataType::Int64,
            LanceColumn::UInt32(_) => LanceDataType::UInt32,
            LanceColumn::UInt64(_) => LanceDataType::UInt64,
            LanceColumn::Utf8(_) => LanceDataType::Utf8,
            LanceColumn::Boolean(_) => LanceDataType::Boolean,
            LanceColumn::Nullable(inner, _) => inner.data_type(),
        }
    }
}

/// A record batch: a schema plus aligned columns.
#[derive(Debug, Clone)]
pub struct LanceBatch {
    /// Schema describing the columns.
    pub schema: LanceSchema,
    /// Column data in the same order as `schema.fields`.
    pub columns: Vec<LanceColumn>,
    /// Number of rows.
    pub num_rows: usize,
}

impl LanceBatch {
    /// Create a batch.  `num_rows` must equal the length of each column.
    pub fn new(schema: LanceSchema, columns: Vec<LanceColumn>, num_rows: usize) -> Self {
        Self {
            schema,
            columns,
            num_rows,
        }
    }

    /// Create an empty batch with the given schema.
    pub fn empty(schema: LanceSchema) -> Self {
        let columns = schema
            .fields
            .iter()
            .map(|f| empty_column_for(&f.dtype))
            .collect();
        Self {
            schema,
            columns,
            num_rows: 0,
        }
    }
}

/// Build an empty column of the appropriate variant.
fn empty_column_for(dtype: &LanceDataType) -> LanceColumn {
    match dtype {
        LanceDataType::Float32 => LanceColumn::Float32(Vec::new()),
        LanceDataType::Float64 => LanceColumn::Float64(Vec::new()),
        LanceDataType::Int32 => LanceColumn::Int32(Vec::new()),
        LanceDataType::Int64 => LanceColumn::Int64(Vec::new()),
        LanceDataType::UInt32 => LanceColumn::UInt32(Vec::new()),
        LanceDataType::UInt64 => LanceColumn::UInt64(Vec::new()),
        LanceDataType::Utf8 => LanceColumn::Utf8(Vec::new()),
        LanceDataType::Boolean => LanceColumn::Boolean(Vec::new()),
    }
}
