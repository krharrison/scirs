//! Core types for the Protocol Buffer schema registry.
//!
//! This module defines the fundamental data structures used across the schema
//! registry: schema identifiers, field and message descriptors, the schema
//! container itself, and the error hierarchy.

use serde::{Deserialize, Serialize};

// ─── Identifier newtypes ─────────────────────────────────────────────────────

/// Opaque identifier for a registered schema, assigned by the registry on first
/// registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct SchemaId(pub u32);

impl SchemaId {
    /// Return the inner `u32` value.
    pub fn value(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for SchemaId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SchemaId({})", self.0)
    }
}

/// Monotonically increasing version counter for a given schema.  Version 1 is
/// the initial registration; each successful `register_version` call increments
/// this by one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct SchemaVersion(pub u32);

impl SchemaVersion {
    /// Return the inner `u32` value.
    pub fn value(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for SchemaVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SchemaVersion({})", self.0)
    }
}

// ─── FieldType ───────────────────────────────────────────────────────────────

/// The scalar or composite type of a Protocol Buffer field.
///
/// This mirrors the proto3 type system with the addition of `Repeated` for
/// repeated (list) fields and `Message` for embedded sub-messages referenced by
/// name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FieldType {
    /// `int32` — varint-encoded signed 32-bit integer (two's complement on the wire).
    Int32,
    /// `int64` — varint-encoded signed 64-bit integer.
    Int64,
    /// `uint32` — varint-encoded unsigned 32-bit integer.
    UInt32,
    /// `uint64` — varint-encoded unsigned 64-bit integer.
    UInt64,
    /// `float` — 32-bit IEEE 754 floating-point (wire type fixed32).
    Float,
    /// `double` — 64-bit IEEE 754 floating-point (wire type fixed64).
    Double,
    /// `bool` — varint-encoded boolean (0 or 1).
    Bool,
    /// `string` — length-delimited UTF-8 string.
    String,
    /// `bytes` — length-delimited arbitrary byte string.
    Bytes,
    /// Embedded sub-message, referenced by its fully-qualified message name.
    Message(std::string::String),
    /// Repeated (list) field; the inner box holds the element type.
    Repeated(Box<FieldType>),
}

impl FieldType {
    /// Human-readable proto-style name.
    pub fn proto_name(&self) -> std::string::String {
        match self {
            FieldType::Int32 => "int32".to_string(),
            FieldType::Int64 => "int64".to_string(),
            FieldType::UInt32 => "uint32".to_string(),
            FieldType::UInt64 => "uint64".to_string(),
            FieldType::Float => "float".to_string(),
            FieldType::Double => "double".to_string(),
            FieldType::Bool => "bool".to_string(),
            FieldType::String => "string".to_string(),
            FieldType::Bytes => "bytes".to_string(),
            FieldType::Message(name) => format!("message({name})"),
            FieldType::Repeated(inner) => format!("repeated {}", inner.proto_name()),
        }
    }

    /// Returns `true` if this type maps to proto wire type 0 (varint).
    pub fn is_varint(&self) -> bool {
        matches!(
            self,
            FieldType::Int32
                | FieldType::Int64
                | FieldType::UInt32
                | FieldType::UInt64
                | FieldType::Bool
        )
    }

    /// Returns `true` if this type is length-delimited on the wire.
    pub fn is_length_delimited(&self) -> bool {
        matches!(
            self,
            FieldType::String | FieldType::Bytes | FieldType::Message(_) | FieldType::Repeated(_)
        )
    }
}

impl std::fmt::Display for FieldType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.proto_name())
    }
}

// ─── FieldDescriptor ─────────────────────────────────────────────────────────

/// A single field within a [`MessageDescriptor`].
///
/// Mirrors the information a `.proto` file carries per field declaration:
/// field number, name, type, and whether the field must appear on the wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldDescriptor {
    /// Proto field number in [1, 536_870_911]; numbers [19000, 19999] are
    /// reserved by the Protocol Buffer spec.
    pub field_number: u32,
    /// Camel-case or snake_case field name (must be unique within the message).
    pub name: std::string::String,
    /// Scalar or composite type of the field.
    pub field_type: FieldType,
    /// If `true`, the encoder MUST emit this field; the decoder MUST reject
    /// a message that omits it.  Mirrors `proto2` `required` semantics.
    pub required: bool,
}

impl FieldDescriptor {
    /// Convenience constructor: create an optional field (required = false).
    pub fn optional(
        field_number: u32,
        name: impl Into<std::string::String>,
        field_type: FieldType,
    ) -> Self {
        Self {
            field_number,
            name: name.into(),
            field_type,
            required: false,
        }
    }

    /// Convenience constructor: create a required field.
    pub fn required(
        field_number: u32,
        name: impl Into<std::string::String>,
        field_type: FieldType,
    ) -> Self {
        Self {
            field_number,
            name: name.into(),
            field_type,
            required: true,
        }
    }
}

// ─── MessageDescriptor ───────────────────────────────────────────────────────

/// A complete message type definition, analogous to a `message` block in a
/// `.proto` file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageDescriptor {
    /// Unqualified message name, e.g. `"Person"`.
    pub name: std::string::String,
    /// Optional dot-separated package, e.g. `"myorg.myapp.v1"`.
    pub package: std::string::String,
    /// Ordered list of field descriptors.
    pub fields: Vec<FieldDescriptor>,
}

impl MessageDescriptor {
    /// Create a new descriptor with no fields.
    pub fn new(
        name: impl Into<std::string::String>,
        package: impl Into<std::string::String>,
    ) -> Self {
        Self {
            name: name.into(),
            package: package.into(),
            fields: Vec::new(),
        }
    }

    /// Add a field and return `self` for chaining.
    pub fn with_field(mut self, field: FieldDescriptor) -> Self {
        self.fields.push(field);
        self
    }

    /// Fully-qualified name: `"<package>.<name>"` or just `"<name>"` when the
    /// package is empty.
    pub fn fully_qualified_name(&self) -> std::string::String {
        if self.package.is_empty() {
            self.name.clone()
        } else {
            format!("{}.{}", self.package, self.name)
        }
    }

    /// Look up a field by its field number.
    pub fn field_by_number(&self, number: u32) -> Option<&FieldDescriptor> {
        self.fields.iter().find(|f| f.field_number == number)
    }

    /// Look up a field by name.
    pub fn field_by_name(&self, name: &str) -> Option<&FieldDescriptor> {
        self.fields.iter().find(|f| f.name == name)
    }
}

// ─── Schema ──────────────────────────────────────────────────────────────────

/// A versioned schema entry stored in the registry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Schema {
    /// Registry-assigned stable identifier.
    pub id: SchemaId,
    /// Monotonically increasing version of this entry.
    pub version: SchemaVersion,
    /// The message descriptor for this version.
    pub descriptor: MessageDescriptor,
    /// Unix timestamp (seconds since epoch) at the time of registration.
    /// Populated with 0 in test contexts where a clock is not available.
    pub created_at: u64,
}

impl Schema {
    /// Construct a new schema entry.
    pub fn new(
        id: SchemaId,
        version: SchemaVersion,
        descriptor: MessageDescriptor,
        created_at: u64,
    ) -> Self {
        Self {
            id,
            version,
            descriptor,
            created_at,
        }
    }
}

// ─── RegistryConfig ──────────────────────────────────────────────────────────

/// Configuration knobs for a [`SchemaRegistry`](super::registry::SchemaRegistry).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegistryConfig {
    /// Maximum number of distinct schemas (by id) the registry will hold before
    /// rejecting new registrations.  Defaults to 1 000.
    pub max_schemas: usize,
    /// When `true` (the default), new versions of an existing schema are
    /// accepted provided they are backward-compatible.  When `false`, any
    /// attempt to call `register_version` on an existing id is rejected with
    /// [`SchemaRegistryError::VersionConflict`].
    pub allow_schema_evolution: bool,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            max_schemas: 1_000,
            allow_schema_evolution: true,
        }
    }
}

// ─── SchemaRegistryError ─────────────────────────────────────────────────────

/// Errors produced by the schema registry and wire format operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum SchemaRegistryError {
    /// No schema with the given id exists in the registry.
    NotFound(SchemaId),
    /// A version already exists and evolution is disabled, or the requested
    /// version is not monotonically greater than the current latest.
    VersionConflict,
    /// The proposed new descriptor is not backward-compatible with the existing
    /// one; the inner string describes which constraint was violated.
    IncompatibleEvolution(std::string::String),
    /// A serialization or deserialization error occurred (e.g. malformed JSON).
    Serialization(std::string::String),
    /// The registry has reached its configured `max_schemas` limit.
    RegistryFull,
    /// The requested schema version does not exist.
    VersionNotFound {
        /// The schema whose version was queried.
        id: SchemaId,
        /// The version that was not found.
        version: SchemaVersion,
    },
    /// A wire-format encoding or decoding error.
    WireFormat(std::string::String),
    /// A field validation error.
    Validation(std::string::String),
}

impl std::fmt::Display for SchemaRegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchemaRegistryError::NotFound(id) => write!(f, "Schema not found: {id}"),
            SchemaRegistryError::VersionConflict => {
                write!(
                    f,
                    "Schema version conflict: evolution disabled or non-monotonic version"
                )
            }
            SchemaRegistryError::IncompatibleEvolution(msg) => {
                write!(f, "Incompatible schema evolution: {msg}")
            }
            SchemaRegistryError::Serialization(msg) => write!(f, "Serialization error: {msg}"),
            SchemaRegistryError::RegistryFull => write!(f, "Schema registry is full"),
            SchemaRegistryError::VersionNotFound { id, version } => {
                write!(f, "Version {version} not found for schema {id}")
            }
            SchemaRegistryError::WireFormat(msg) => write!(f, "Wire format error: {msg}"),
            SchemaRegistryError::Validation(msg) => write!(f, "Validation error: {msg}"),
        }
    }
}

impl std::error::Error for SchemaRegistryError {}

/// Convenient alias for results throughout the schema registry.
pub type SchemaRegistryResult<T> = Result<T, SchemaRegistryError>;

// ─── FieldValue ──────────────────────────────────────────────────────────────

/// A typed value that can be stored in a Protocol Buffer field.
///
/// Used as the currency type when encoding/decoding messages against a
/// [`MessageDescriptor`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum FieldValue {
    /// Signed 32-bit integer.
    Int32(i32),
    /// Signed 64-bit integer.
    Int64(i64),
    /// Unsigned 32-bit integer.
    UInt32(u32),
    /// Unsigned 64-bit integer.
    UInt64(u64),
    /// 32-bit IEEE 754 float.
    Float(f32),
    /// 64-bit IEEE 754 float.
    Double(f64),
    /// Boolean value.
    Bool(bool),
    /// UTF-8 string.
    Str(std::string::String),
    /// Arbitrary byte sequence.
    Bytes(Vec<u8>),
    /// Pre-encoded nested message bytes.
    Message(Vec<u8>),
}

impl FieldValue {
    /// Return the [`FieldType`] that best describes this value.
    pub fn field_type(&self) -> FieldType {
        match self {
            FieldValue::Int32(_) => FieldType::Int32,
            FieldValue::Int64(_) => FieldType::Int64,
            FieldValue::UInt32(_) => FieldType::UInt32,
            FieldValue::UInt64(_) => FieldType::UInt64,
            FieldValue::Float(_) => FieldType::Float,
            FieldValue::Double(_) => FieldType::Double,
            FieldValue::Bool(_) => FieldType::Bool,
            FieldValue::Str(_) => FieldType::String,
            FieldValue::Bytes(_) => FieldType::Bytes,
            FieldValue::Message(_) => FieldType::Message(std::string::String::new()),
        }
    }
}

impl std::fmt::Display for FieldValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FieldValue::Int32(v) => write!(f, "{v}"),
            FieldValue::Int64(v) => write!(f, "{v}"),
            FieldValue::UInt32(v) => write!(f, "{v}"),
            FieldValue::UInt64(v) => write!(f, "{v}"),
            FieldValue::Float(v) => write!(f, "{v}"),
            FieldValue::Double(v) => write!(f, "{v}"),
            FieldValue::Bool(v) => write!(f, "{v}"),
            FieldValue::Str(s) => write!(f, "{s}"),
            FieldValue::Bytes(b) => write!(f, "<{} bytes>", b.len()),
            FieldValue::Message(b) => write!(f, "<message {} bytes>", b.len()),
        }
    }
}
