//! Protocol Buffer schema registry for scirs2-io.
//!
//! This module provides a self-contained, pure-Rust Protocol Buffer schema
//! registry.  It does **not** depend on `prost`, `protobuf`, or any other
//! code-generation tool, in accordance with the COOLJAPAN pure-Rust policy.
//!
//! # Overview
//!
//! The registry stores versioned [`MessageDescriptor`]s identified by
//! auto-assigned [`SchemaId`]s.  Consumers can:
//!
//! * **Register** a brand-new schema and receive its `SchemaId`.
//! * **Evolve** an existing schema by registering a new version, subject to
//!   backward-compatibility validation.
//! * **Retrieve** the latest (or a specific) version of a schema.
//! * **Encode / decode** concrete Protocol Buffer messages against a schema.
//! * **Export / import** the entire registry as JSON.
//!
//! # Quick start
//!
//! ```rust
//! use scirs2_io::schema_registry::{
//!     registry::SchemaRegistry,
//!     types::{FieldDescriptor, FieldType, FieldValue, MessageDescriptor, RegistryConfig},
//!     wire::{decode_message, encode_message},
//! };
//!
//! // 1. Create a registry and register a schema
//! let mut reg = SchemaRegistry::new(RegistryConfig::default());
//!
//! let person = MessageDescriptor::new("Person", "example")
//!     .with_field(FieldDescriptor::optional(1, "id",   FieldType::Int64))
//!     .with_field(FieldDescriptor::optional(2, "name", FieldType::String));
//!
//! let id = reg.register(person.clone()).unwrap();
//! assert_eq!(reg.get(id).unwrap().version.value(), 1);
//!
//! // 2. Encode a message
//! let values = vec![
//!     (1, FieldValue::Int64(42)),
//!     (2, FieldValue::Str("Alice".to_string())),
//! ];
//! let bytes = encode_message(&person, &values);
//!
//! // 3. Decode it back
//! let decoded = decode_message(&person, &bytes).unwrap();
//! assert_eq!(decoded[0].0, "id");
//! assert_eq!(decoded[1].1, FieldValue::Str("Alice".to_string()));
//! ```
//!
//! # Sub-modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`types`] | Core types: `SchemaId`, `FieldType`, `MessageDescriptor`, etc. |
//! | [`registry`] | [`SchemaRegistry`] — versioned schema storage and compatibility checks |
//! | [`wire`] | Pure-Rust proto wire format encoder / decoder |
//! | [`validation`] | Structural validation of message descriptors |

pub mod registry;
pub mod types;
pub mod validation;
pub mod wire;

// ─── Convenience re-exports ───────────────────────────────────────────────────

pub use registry::SchemaRegistry;
pub use types::{
    FieldDescriptor, FieldType, FieldValue, MessageDescriptor, RegistryConfig, Schema, SchemaId,
    SchemaRegistryError, SchemaRegistryResult, SchemaVersion,
};
pub use validation::validate_descriptor;
pub use wire::{
    decode_message, decode_varint, encode_field, encode_message, encode_varint, ProtoDecoder,
    ProtoEncoder, WireType, WireValue,
};
