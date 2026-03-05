//! Schema validation for scientific data I/O
//!
//! Provides structured schema definitions, validation against data, and automatic schema inference.
//!
//! # Example
//!
//! ```rust
//! use scirs2_io::schema::{Schema, SchemaField, FieldType, SchemaValidator, CoercionMode};
//! use std::collections::HashMap;
//!
//! let schema = Schema::builder()
//!     .field(SchemaField::new("id", FieldType::Int64).not_nullable())
//!     .field(SchemaField::new("name", FieldType::Utf8).nullable())
//!     .field(SchemaField::new("score", FieldType::Float64).nullable().with_default(serde_json::json!(0.0)))
//!     .build();
//!
//! let mut row: HashMap<String, serde_json::Value> = HashMap::new();
//! row.insert("id".to_string(), serde_json::json!(42));
//! row.insert("name".to_string(), serde_json::json!("Alice"));
//! row.insert("score".to_string(), serde_json::json!(99.5));
//!
//! let validator = SchemaValidator::new(schema, CoercionMode::Lenient);
//! let report = validator.validate_row(&row, 0);
//! assert!(report.violations.is_empty());
//! ```

#![allow(missing_docs)]

pub mod validation;

pub use validation::{
    CoercionMode, SchemaInference, SchemaValidator, TypeCoercion, ValidationReport,
    ValidationViolation,
};

use crate::error::{IoError, Result};
use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ─── Field type ─────────────────────────────────────────────────────────────

/// The data type of a schema field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FieldType {
    /// 8-bit signed integer
    Int8,
    /// 16-bit signed integer
    Int16,
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
    /// 32-bit unsigned integer
    UInt32,
    /// 64-bit unsigned integer
    UInt64,
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
    /// UTF-8 string (optionally bounded)
    Utf8,
    /// Boolean
    Boolean,
    /// Calendar date (no time zone)
    Date,
    /// Date + time with UTC timezone
    Timestamp,
    /// Raw byte array
    Binary,
    /// JSON value (any structure)
    Json,
    /// List of values with an element type
    List(Box<FieldType>),
    /// Struct with named sub-fields
    Struct(Vec<SchemaField>),
    /// Decimal with precision and scale
    Decimal {
        /// Total number of digits
        precision: u8,
        /// Number of digits after the decimal point
        scale: u8,
    },
}

impl FieldType {
    /// Human-readable name of this type.
    pub fn type_name(&self) -> String {
        match self {
            FieldType::Int8 => "int8".to_string(),
            FieldType::Int16 => "int16".to_string(),
            FieldType::Int32 => "int32".to_string(),
            FieldType::Int64 => "int64".to_string(),
            FieldType::UInt32 => "uint32".to_string(),
            FieldType::UInt64 => "uint64".to_string(),
            FieldType::Float32 => "float32".to_string(),
            FieldType::Float64 => "float64".to_string(),
            FieldType::Utf8 => "string".to_string(),
            FieldType::Boolean => "boolean".to_string(),
            FieldType::Date => "date".to_string(),
            FieldType::Timestamp => "timestamp".to_string(),
            FieldType::Binary => "binary".to_string(),
            FieldType::Json => "json".to_string(),
            FieldType::List(elem) => format!("list<{}>", elem.type_name()),
            FieldType::Struct(_) => "struct".to_string(),
            FieldType::Decimal { precision, scale } => {
                format!("decimal({precision},{scale})")
            }
        }
    }

    /// Returns true if the type is numeric (integer or float).
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            FieldType::Int8
                | FieldType::Int16
                | FieldType::Int32
                | FieldType::Int64
                | FieldType::UInt32
                | FieldType::UInt64
                | FieldType::Float32
                | FieldType::Float64
                | FieldType::Decimal { .. }
        )
    }

    /// Returns true if the type is an integer variant.
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            FieldType::Int8
                | FieldType::Int16
                | FieldType::Int32
                | FieldType::Int64
                | FieldType::UInt32
                | FieldType::UInt64
        )
    }

    /// Returns true if the type is a float variant.
    pub fn is_float(&self) -> bool {
        matches!(self, FieldType::Float32 | FieldType::Float64)
    }
}

impl fmt::Display for FieldType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.type_name())
    }
}

// ─── Constraint ─────────────────────────────────────────────────────────────

/// An additional constraint on a schema field value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Constraint {
    /// Value must be unique across all rows
    Unique,
    /// Value must be within [min, max] (inclusive)
    Range {
        /// Minimum allowed value (inclusive)
        min: serde_json::Value,
        /// Maximum allowed value (inclusive)
        max: serde_json::Value,
    },
    /// String value must match the regex pattern
    Regex(String),
    /// String length must not exceed max
    MaxLength(usize),
    /// String length must be at least min
    MinLength(usize),
    /// Value must be one of the listed options
    OneOf(Vec<serde_json::Value>),
    /// Custom constraint with name and description
    Custom {
        /// Name of the custom constraint
        name: String,
        /// Human-readable description
        description: String,
    },
}

impl Constraint {
    /// Returns the constraint name for error messages.
    pub fn name(&self) -> &str {
        match self {
            Constraint::Unique => "unique",
            Constraint::Range { .. } => "range",
            Constraint::Regex(_) => "regex",
            Constraint::MaxLength(_) => "max_length",
            Constraint::MinLength(_) => "min_length",
            Constraint::OneOf(_) => "one_of",
            Constraint::Custom { name, .. } => name,
        }
    }
}

// ─── SchemaField ────────────────────────────────────────────────────────────

/// A single field definition within a [`Schema`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchemaField {
    /// Column name
    pub name: String,
    /// Data type
    pub field_type: FieldType,
    /// Whether NULL/missing values are allowed
    pub nullable: bool,
    /// Default value used when the field is missing
    pub default: Option<serde_json::Value>,
    /// Additional constraints
    pub constraints: Vec<Constraint>,
    /// Optional human-readable description
    pub description: Option<String>,
    /// Custom metadata key-value pairs
    pub metadata: HashMap<String, String>,
}

impl SchemaField {
    /// Create a new nullable field without constraints.
    pub fn new(name: impl Into<String>, field_type: FieldType) -> Self {
        Self {
            name: name.into(),
            field_type,
            nullable: true,
            default: None,
            constraints: Vec::new(),
            description: None,
            metadata: HashMap::new(),
        }
    }

    /// Mark the field as NOT NULL.
    pub fn not_nullable(mut self) -> Self {
        self.nullable = false;
        self
    }

    /// Mark the field as nullable (default).
    pub fn nullable(mut self) -> Self {
        self.nullable = true;
        self
    }

    /// Set a default value.
    pub fn with_default(mut self, default: serde_json::Value) -> Self {
        self.default = Some(default);
        self
    }

    /// Add a constraint.
    pub fn with_constraint(mut self, c: Constraint) -> Self {
        self.constraints.push(c);
        self
    }

    /// Set a human-readable description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Insert a metadata key-value pair.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

// ─── Schema ──────────────────────────────────────────────────────────────────

/// A full schema: ordered list of fields plus optional table-level metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Schema {
    /// Ordered list of field definitions
    pub fields: Vec<SchemaField>,
    /// Optional schema name/table name
    pub name: Option<String>,
    /// Optional human-readable description
    pub description: Option<String>,
    /// Schema version string
    pub version: Option<String>,
    /// Schema-level metadata
    pub metadata: HashMap<String, String>,
}

impl Schema {
    /// Create an empty schema.
    pub fn new() -> Self {
        Self::default()
    }

    /// Start building a schema fluently.
    pub fn builder() -> SchemaBuilder {
        SchemaBuilder::default()
    }

    /// Look up a field by name (case-sensitive).
    pub fn field(&self, name: &str) -> Option<&SchemaField> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Return the number of fields.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Returns true if there are no fields.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Return column names in order.
    pub fn column_names(&self) -> Vec<&str> {
        self.fields.iter().map(|f| f.name.as_str()).collect()
    }

    /// Merge another schema into this one, adding fields not already present.
    /// Returns an error if a field with the same name exists but has a different type.
    pub fn merge(&mut self, other: &Schema) -> Result<()> {
        for other_field in &other.fields {
            match self.fields.iter().find(|f| f.name == other_field.name) {
                Some(existing) if existing.field_type != other_field.field_type => {
                    return Err(IoError::ValidationError(format!(
                        "Incompatible types for field '{}': {:?} vs {:?}",
                        other_field.name, existing.field_type, other_field.field_type
                    )));
                }
                Some(_) => {
                    // Field already present with compatible type — skip
                }
                None => {
                    self.fields.push(other_field.clone());
                }
            }
        }
        Ok(())
    }

    /// Serialize schema to JSON string.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| IoError::SerializationError(e.to_string()))
    }

    /// Deserialize schema from JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| IoError::DeserializationError(e.to_string()))
    }
}

// ─── SchemaBuilder ───────────────────────────────────────────────────────────

/// Fluent builder for [`Schema`].
#[derive(Debug, Default)]
pub struct SchemaBuilder {
    fields: Vec<SchemaField>,
    name: Option<String>,
    description: Option<String>,
    version: Option<String>,
    metadata: HashMap<String, String>,
}

impl SchemaBuilder {
    /// Add a field.
    pub fn field(mut self, f: SchemaField) -> Self {
        self.fields.push(f);
        self
    }

    /// Set the schema name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the schema description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the schema version.
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Add schema-level metadata.
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Build the final [`Schema`].
    pub fn build(self) -> Schema {
        Schema {
            fields: self.fields,
            name: self.name,
            description: self.description,
            version: self.version,
            metadata: self.metadata,
        }
    }
}

// ─── Typed cell value ────────────────────────────────────────────────────────

/// A fully typed value that conforms to a [`FieldType`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TypedValue {
    /// Null / missing
    Null,
    /// Boolean
    Boolean(bool),
    /// Integer (covers all int variants)
    Int(i64),
    /// Unsigned integer (for UInt32/UInt64)
    UInt(u64),
    /// Floating-point (covers Float32/Float64)
    Float(f64),
    /// UTF-8 string
    Utf8(String),
    /// Calendar date
    Date(NaiveDate),
    /// Timestamp
    Timestamp(DateTime<Utc>),
    /// Binary bytes
    Binary(Vec<u8>),
    /// JSON value (arbitrary structure)
    Json(serde_json::Value),
}

impl TypedValue {
    /// Try to convert a raw `serde_json::Value` into a `TypedValue` given a target `FieldType`.
    /// This performs strict conversion — use [`TypeCoercion`] for lenient coercion.
    pub fn from_json_strict(value: &serde_json::Value, field_type: &FieldType) -> Result<Self> {
        match (field_type, value) {
            (_, serde_json::Value::Null) => Ok(TypedValue::Null),
            (FieldType::Boolean, serde_json::Value::Bool(b)) => Ok(TypedValue::Boolean(*b)),
            (ft, serde_json::Value::Number(n)) if ft.is_integer() => {
                let i = n.as_i64().ok_or_else(|| {
                    IoError::ConversionError(format!("Cannot convert {n} to integer"))
                })?;
                Ok(TypedValue::Int(i))
            }
            (ft, serde_json::Value::Number(n)) if ft.is_float() => {
                let f = n.as_f64().ok_or_else(|| {
                    IoError::ConversionError(format!("Cannot convert {n} to float"))
                })?;
                Ok(TypedValue::Float(f))
            }
            (FieldType::Utf8, serde_json::Value::String(s)) => Ok(TypedValue::Utf8(s.clone())),
            (FieldType::Date, serde_json::Value::String(s)) => {
                let date = s.parse::<NaiveDate>().map_err(|e| {
                    IoError::ParseError(format!("Cannot parse date '{}': {}", s, e))
                })?;
                Ok(TypedValue::Date(date))
            }
            (FieldType::Timestamp, serde_json::Value::String(s)) => {
                let ts = s.parse::<DateTime<Utc>>().map_err(|e| {
                    IoError::ParseError(format!("Cannot parse timestamp '{}': {}", s, e))
                })?;
                Ok(TypedValue::Timestamp(ts))
            }
            (FieldType::Json, v) => Ok(TypedValue::Json(v.clone())),
            (ft, v) => Err(IoError::ConversionError(format!(
                "Cannot convert JSON {} to type {}",
                v,
                ft.type_name()
            ))),
        }
    }

    /// Return the field type this value corresponds to.
    pub fn inferred_type(&self) -> FieldType {
        match self {
            TypedValue::Null => FieldType::Json,
            TypedValue::Boolean(_) => FieldType::Boolean,
            TypedValue::Int(_) => FieldType::Int64,
            TypedValue::UInt(_) => FieldType::UInt64,
            TypedValue::Float(_) => FieldType::Float64,
            TypedValue::Utf8(_) => FieldType::Utf8,
            TypedValue::Date(_) => FieldType::Date,
            TypedValue::Timestamp(_) => FieldType::Timestamp,
            TypedValue::Binary(_) => FieldType::Binary,
            TypedValue::Json(_) => FieldType::Json,
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_builder_roundtrip() {
        let schema = Schema::builder()
            .name("test_table")
            .field(SchemaField::new("id", FieldType::Int64).not_nullable())
            .field(SchemaField::new("label", FieldType::Utf8).nullable())
            .build();

        assert_eq!(schema.len(), 2);
        assert_eq!(schema.name, Some("test_table".to_string()));
        let json = schema.to_json().expect("to_json failed");
        let restored = Schema::from_json(&json).expect("from_json failed");
        assert_eq!(schema, restored);
    }

    #[test]
    fn test_field_type_display() {
        assert_eq!(FieldType::Float64.type_name(), "float64");
        assert_eq!(
            FieldType::List(Box::new(FieldType::Int32)).type_name(),
            "list<int32>"
        );
        assert_eq!(
            FieldType::Decimal {
                precision: 10,
                scale: 4
            }
            .type_name(),
            "decimal(10,4)"
        );
    }

    #[test]
    fn test_schema_merge() {
        let mut a = Schema::builder()
            .field(SchemaField::new("id", FieldType::Int64))
            .build();
        let b = Schema::builder()
            .field(SchemaField::new("value", FieldType::Float64))
            .build();
        a.merge(&b).expect("merge failed");
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn test_schema_merge_type_conflict() {
        let mut a = Schema::builder()
            .field(SchemaField::new("id", FieldType::Int64))
            .build();
        let b = Schema::builder()
            .field(SchemaField::new("id", FieldType::Utf8))
            .build();
        assert!(a.merge(&b).is_err());
    }

    #[test]
    fn test_typed_value_from_json_strict() {
        let v = serde_json::json!(42i64);
        let tv = TypedValue::from_json_strict(&v, &FieldType::Int64).unwrap();
        assert_eq!(tv, TypedValue::Int(42));

        let v2 = serde_json::json!("hello");
        let tv2 = TypedValue::from_json_strict(&v2, &FieldType::Utf8).unwrap();
        assert_eq!(tv2, TypedValue::Utf8("hello".to_string()));
    }
}
