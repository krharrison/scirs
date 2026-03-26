//! In-memory Protocol Buffer schema registry.
//!
//! The [`SchemaRegistry`] stores multiple [`MessageDescriptor`]s identified by
//! auto-assigned [`SchemaId`]s.  Each id may have one or more [`SchemaVersion`]s;
//! only backward-compatible evolutions are accepted when
//! [`RegistryConfig::allow_schema_evolution`] is `true`.
//!
//! ## Backward-compatibility rules
//!
//! A new descriptor is considered backward-compatible with an older one when:
//!
//! 1. Every field in the old descriptor is also present in the new one (no
//!    field removals — the field number and wire type must remain valid for
//!    existing serialized data).
//! 2. No existing field changes its field number.
//! 3. No existing field changes its type (scalar type changes break wire
//!    compatibility).
//! 4. New optional fields may be freely added.
//!
//! Required-to-optional promotion is allowed (it relaxes constraints); the
//! inverse (optional-to-required) is rejected.

use std::collections::HashMap;

use super::types::{
    FieldDescriptor, FieldType, MessageDescriptor, RegistryConfig, Schema, SchemaId,
    SchemaRegistryError, SchemaRegistryResult, SchemaVersion,
};
use super::validation::validate_descriptor;

// ─── SchemaRegistry ──────────────────────────────────────────────────────────

/// In-memory registry that tracks versioned Protocol Buffer message descriptors.
///
/// # Example
///
/// ```rust
/// use scirs2_io::schema_registry::{
///     registry::SchemaRegistry,
///     types::{FieldDescriptor, FieldType, MessageDescriptor, RegistryConfig},
/// };
///
/// let mut reg = SchemaRegistry::new(RegistryConfig::default());
///
/// let desc = MessageDescriptor::new("Person", "example")
///     .with_field(FieldDescriptor::optional(1, "name", FieldType::String))
///     .with_field(FieldDescriptor::optional(2, "age",  FieldType::Int32));
///
/// let id = reg.register(desc).unwrap();
/// let schema = reg.get(id).unwrap();
/// assert_eq!(schema.version.value(), 1);
/// ```
#[derive(Debug)]
pub struct SchemaRegistry {
    /// Versioned schema storage: each key maps to an ordered list of schema
    /// versions, where index 0 = version 1, index 1 = version 2, etc.
    schemas: HashMap<SchemaId, Vec<Schema>>,
    /// Next id to assign when a brand-new schema is registered.
    next_id: u32,
    /// Registry configuration.
    config: RegistryConfig,
}

impl SchemaRegistry {
    /// Create a new, empty registry with the given configuration.
    pub fn new(config: RegistryConfig) -> Self {
        Self {
            schemas: HashMap::new(),
            next_id: 1,
            config,
        }
    }

    /// Create a registry with default configuration.
    pub fn default_config() -> Self {
        Self::new(RegistryConfig::default())
    }

    // ── Registration ─────────────────────────────────────────────────────────

    /// Register a brand-new schema descriptor.
    ///
    /// Assigns a fresh [`SchemaId`], validates the descriptor, stores it as
    /// version 1, and returns the new id.
    ///
    /// # Errors
    ///
    /// Returns [`SchemaRegistryError::Validation`] if the descriptor is
    /// structurally invalid, or [`SchemaRegistryError::RegistryFull`] if the
    /// registry has reached its configured limit.
    pub fn register(&mut self, descriptor: MessageDescriptor) -> SchemaRegistryResult<SchemaId> {
        if self.schemas.len() >= self.config.max_schemas {
            return Err(SchemaRegistryError::RegistryFull);
        }

        validate_descriptor(&descriptor)?;

        let id = SchemaId(self.next_id);
        self.next_id += 1;

        let schema = Schema::new(id, SchemaVersion(1), descriptor, now_secs());
        self.schemas.insert(id, vec![schema]);
        Ok(id)
    }

    /// Register a new version of an existing schema.
    ///
    /// The new descriptor must be backward-compatible with the latest version
    /// unless [`RegistryConfig::allow_schema_evolution`] is `false`, in which
    /// case any attempt to add a version is rejected.
    ///
    /// # Errors
    ///
    /// * [`SchemaRegistryError::NotFound`] — if `id` does not exist.
    /// * [`SchemaRegistryError::VersionConflict`] — if evolution is disabled.
    /// * [`SchemaRegistryError::IncompatibleEvolution`] — if the new descriptor
    ///   breaks backward compatibility.
    /// * [`SchemaRegistryError::Validation`] — if the descriptor is invalid.
    pub fn register_version(
        &mut self,
        id: SchemaId,
        descriptor: MessageDescriptor,
    ) -> SchemaRegistryResult<SchemaVersion> {
        if !self.config.allow_schema_evolution {
            return Err(SchemaRegistryError::VersionConflict);
        }

        validate_descriptor(&descriptor)?;

        let versions = self
            .schemas
            .get_mut(&id)
            .ok_or(SchemaRegistryError::NotFound(id))?;

        let current = versions.last().expect("non-empty vec");
        if !is_compatible(&current.descriptor, &descriptor) {
            return Err(SchemaRegistryError::IncompatibleEvolution(
                incompatibility_reason(&current.descriptor, &descriptor),
            ));
        }

        let next_version = SchemaVersion(current.version.value() + 1);
        let schema = Schema::new(id, next_version, descriptor, now_secs());
        versions.push(schema);
        Ok(next_version)
    }

    // ── Retrieval ─────────────────────────────────────────────────────────────

    /// Return a reference to the **latest** version of schema `id`.
    pub fn get(&self, id: SchemaId) -> SchemaRegistryResult<&Schema> {
        self.schemas
            .get(&id)
            .and_then(|v| v.last())
            .ok_or(SchemaRegistryError::NotFound(id))
    }

    /// Return a reference to a specific version of schema `id`.
    pub fn get_version(
        &self,
        id: SchemaId,
        version: SchemaVersion,
    ) -> SchemaRegistryResult<&Schema> {
        let versions = self
            .schemas
            .get(&id)
            .ok_or(SchemaRegistryError::NotFound(id))?;

        let idx = version
            .value()
            .checked_sub(1)
            .ok_or_else(|| SchemaRegistryError::VersionNotFound { id, version })?
            as usize;

        versions
            .get(idx)
            .ok_or(SchemaRegistryError::VersionNotFound { id, version })
    }

    /// Return the list of all registered schema ids, sorted ascending.
    pub fn list(&self) -> Vec<SchemaId> {
        let mut ids: Vec<SchemaId> = self.schemas.keys().copied().collect();
        ids.sort();
        ids
    }

    /// Return the number of distinct schemas (not counting versions).
    pub fn schema_count(&self) -> usize {
        self.schemas.len()
    }

    /// Return the total number of versions across all schemas.
    pub fn version_count(&self) -> usize {
        self.schemas.values().map(|v| v.len()).sum()
    }

    /// Return all versions for schema `id`, oldest first.
    pub fn all_versions(&self, id: SchemaId) -> SchemaRegistryResult<&[Schema]> {
        self.schemas
            .get(&id)
            .map(|v| v.as_slice())
            .ok_or(SchemaRegistryError::NotFound(id))
    }

    // ── JSON export / import ──────────────────────────────────────────────────

    /// Serialize the entire registry to a JSON string.
    ///
    /// The output is a JSON array of objects, each carrying the schema id,
    /// and an array of versions with their descriptors.
    pub fn export_json(&self) -> String {
        let mut entries: Vec<(SchemaId, &Vec<Schema>)> =
            self.schemas.iter().map(|(k, v)| (*k, v)).collect();
        entries.sort_by_key(|(id, _)| *id);

        let mut out = String::from("[\n");
        for (entry_idx, (id, versions)) in entries.iter().enumerate() {
            out.push_str("  {\n");
            out.push_str(&format!("    \"schema_id\": {},\n", id.value()));
            out.push_str("    \"versions\": [\n");

            for (vi, schema) in versions.iter().enumerate() {
                out.push_str("      {\n");
                out.push_str(&format!(
                    "        \"version\": {},\n",
                    schema.version.value()
                ));
                out.push_str(&format!("        \"created_at\": {},\n", schema.created_at));
                out.push_str("        \"descriptor\": ");
                out.push_str(&serialize_descriptor(&schema.descriptor));
                out.push('\n');
                out.push_str("      }");
                if vi + 1 < versions.len() {
                    out.push(',');
                }
                out.push('\n');
            }

            out.push_str("    ]\n");
            out.push_str("  }");
            if entry_idx + 1 < entries.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push(']');
        out
    }

    /// Deserialize a registry from a JSON string previously produced by
    /// [`export_json`](SchemaRegistry::export_json).
    ///
    /// Returns a fresh `SchemaRegistry` with the same schemas and `next_id`
    /// set beyond the maximum imported id.
    pub fn import_json(json: &str) -> SchemaRegistryResult<Self> {
        let value: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| SchemaRegistryError::Serialization(e.to_string()))?;

        let array = value.as_array().ok_or_else(|| {
            SchemaRegistryError::Serialization("expected JSON array at top level".to_string())
        })?;

        let mut registry = Self::new(RegistryConfig::default());

        for entry in array {
            let schema_id_raw = entry["schema_id"].as_u64().ok_or_else(|| {
                SchemaRegistryError::Serialization("missing schema_id".to_string())
            })? as u32;

            let id = SchemaId(schema_id_raw);

            let versions_arr = entry["versions"].as_array().ok_or_else(|| {
                SchemaRegistryError::Serialization("missing versions array".to_string())
            })?;

            let mut schema_versions: Vec<Schema> = Vec::new();

            for ver_obj in versions_arr {
                let version_num = ver_obj["version"].as_u64().ok_or_else(|| {
                    SchemaRegistryError::Serialization("missing version".to_string())
                })? as u32;

                let created_at = ver_obj["created_at"].as_u64().ok_or_else(|| {
                    SchemaRegistryError::Serialization("missing created_at".to_string())
                })?;

                let descriptor = deserialize_descriptor(&ver_obj["descriptor"])?;
                let schema = Schema::new(id, SchemaVersion(version_num), descriptor, created_at);
                schema_versions.push(schema);
            }

            registry.schemas.insert(id, schema_versions);
            if schema_id_raw >= registry.next_id {
                registry.next_id = schema_id_raw + 1;
            }
        }

        Ok(registry)
    }
}

// ─── Compatibility check ──────────────────────────────────────────────────────

/// Returns `true` if `new_desc` is backward-compatible with `old_desc`.
///
/// The compatibility rules are:
///
/// 1. Every field present in `old_desc` must also be present in `new_desc`
///    with the same field number and the same type (no field removal, no type
///    change).
/// 2. New fields (present in `new_desc` but not in `old_desc`) must be
///    optional (i.e. `required == false`).
/// 3. A required field in `old_desc` may be relaxed to optional in `new_desc`
///    (this is a non-breaking promotion), but the inverse is not allowed.
pub fn is_compatible(old: &MessageDescriptor, new: &MessageDescriptor) -> bool {
    // Rule 1: all old fields must survive
    for old_field in &old.fields {
        match new.field_by_number(old_field.field_number) {
            None => return false, // field removed
            Some(new_field) => {
                if !types_wire_compatible(&old_field.field_type, &new_field.field_type) {
                    return false; // type changed
                }
                // optional → required is not allowed
                if !old_field.required && new_field.required {
                    return false;
                }
            }
        }
    }

    // Rule 2: new-only fields must be optional
    for new_field in &new.fields {
        if old.field_by_number(new_field.field_number).is_none() && new_field.required {
            return false;
        }
    }

    true
}

/// Produce a human-readable explanation of why `new_desc` is incompatible with
/// `old_desc`.  Returns a generic message if no specific reason is found (which
/// should not happen in practice).
fn incompatibility_reason(old: &MessageDescriptor, new: &MessageDescriptor) -> String {
    for old_field in &old.fields {
        match new.field_by_number(old_field.field_number) {
            None => {
                return format!(
                    "field '{}' (number {}) was removed",
                    old_field.name, old_field.field_number
                );
            }
            Some(new_field) => {
                if !types_wire_compatible(&old_field.field_type, &new_field.field_type) {
                    return format!(
                        "field '{}' (number {}) changed type from {} to {}",
                        old_field.name,
                        old_field.field_number,
                        old_field.field_type.proto_name(),
                        new_field.field_type.proto_name()
                    );
                }
                if !old_field.required && new_field.required {
                    return format!(
                        "field '{}' (number {}) was promoted from optional to required",
                        old_field.name, old_field.field_number
                    );
                }
            }
        }
    }

    for new_field in &new.fields {
        if old.field_by_number(new_field.field_number).is_none() && new_field.required {
            return format!(
                "new field '{}' (number {}) is marked required",
                new_field.name, new_field.field_number
            );
        }
    }

    "incompatible schema change (no specific reason identified)".to_string()
}

/// Check whether two [`FieldType`]s are considered wire-compatible, i.e.
/// the on-the-wire representation is the same.  For the purpose of this
/// registry, we use a strict check: the types must be identical (after
/// normalizing `Message` names, which we ignore for compatibility purposes).
fn types_wire_compatible(old_type: &FieldType, new_type: &FieldType) -> bool {
    match (old_type, new_type) {
        (FieldType::Int32, FieldType::Int32) => true,
        (FieldType::Int64, FieldType::Int64) => true,
        (FieldType::UInt32, FieldType::UInt32) => true,
        (FieldType::UInt64, FieldType::UInt64) => true,
        (FieldType::Float, FieldType::Float) => true,
        (FieldType::Double, FieldType::Double) => true,
        (FieldType::Bool, FieldType::Bool) => true,
        (FieldType::String, FieldType::String) => true,
        (FieldType::Bytes, FieldType::Bytes) => true,
        // Message type changes: any message name is compatible with any other
        // (the schema evolution check is at the nested message level)
        (FieldType::Message(_), FieldType::Message(_)) => true,
        // Repeated: element types must match
        (FieldType::Repeated(a), FieldType::Repeated(b)) => types_wire_compatible(a, b),
        _ => false,
    }
}

// ─── JSON serialisation helpers ───────────────────────────────────────────────

fn serialize_field_type(ft: &FieldType) -> serde_json::Value {
    match ft {
        FieldType::Int32 => serde_json::json!("int32"),
        FieldType::Int64 => serde_json::json!("int64"),
        FieldType::UInt32 => serde_json::json!("uint32"),
        FieldType::UInt64 => serde_json::json!("uint64"),
        FieldType::Float => serde_json::json!("float"),
        FieldType::Double => serde_json::json!("double"),
        FieldType::Bool => serde_json::json!("bool"),
        FieldType::String => serde_json::json!("string"),
        FieldType::Bytes => serde_json::json!("bytes"),
        FieldType::Message(name) => serde_json::json!({ "message": name }),
        FieldType::Repeated(inner) => {
            serde_json::json!({ "repeated": serialize_field_type(inner) })
        }
    }
}

fn deserialize_field_type(v: &serde_json::Value) -> SchemaRegistryResult<FieldType> {
    if let Some(s) = v.as_str() {
        return match s {
            "int32" => Ok(FieldType::Int32),
            "int64" => Ok(FieldType::Int64),
            "uint32" => Ok(FieldType::UInt32),
            "uint64" => Ok(FieldType::UInt64),
            "float" => Ok(FieldType::Float),
            "double" => Ok(FieldType::Double),
            "bool" => Ok(FieldType::Bool),
            "string" => Ok(FieldType::String),
            "bytes" => Ok(FieldType::Bytes),
            other => Err(SchemaRegistryError::Serialization(format!(
                "unknown field type: {other}"
            ))),
        };
    }

    if let Some(msg_name) = v.get("message").and_then(|m| m.as_str()) {
        return Ok(FieldType::Message(msg_name.to_string()));
    }

    if let Some(inner) = v.get("repeated") {
        let inner_type = deserialize_field_type(inner)?;
        return Ok(FieldType::Repeated(Box::new(inner_type)));
    }

    Err(SchemaRegistryError::Serialization(format!(
        "cannot deserialize field type from: {v}"
    )))
}

fn serialize_descriptor(desc: &MessageDescriptor) -> String {
    let fields: Vec<serde_json::Value> = desc
        .fields
        .iter()
        .map(|f| {
            serde_json::json!({
                "field_number": f.field_number,
                "name": f.name,
                "field_type": serialize_field_type(&f.field_type),
                "required": f.required
            })
        })
        .collect();

    let obj = serde_json::json!({
        "name": desc.name,
        "package": desc.package,
        "fields": fields
    });

    serde_json::to_string(&obj).unwrap_or_else(|_| "{}".to_string())
}

fn deserialize_descriptor(v: &serde_json::Value) -> SchemaRegistryResult<MessageDescriptor> {
    let name = v["name"]
        .as_str()
        .ok_or_else(|| SchemaRegistryError::Serialization("missing descriptor.name".to_string()))?
        .to_string();

    let package = v["package"].as_str().unwrap_or("").to_string();

    let fields_arr = v["fields"].as_array().ok_or_else(|| {
        SchemaRegistryError::Serialization("missing descriptor.fields".to_string())
    })?;

    let mut fields = Vec::new();
    for f in fields_arr {
        let field_number = f["field_number"]
            .as_u64()
            .ok_or_else(|| SchemaRegistryError::Serialization("missing field_number".to_string()))?
            as u32;
        let fname = f["name"]
            .as_str()
            .ok_or_else(|| SchemaRegistryError::Serialization("missing field name".to_string()))?
            .to_string();
        let field_type = deserialize_field_type(&f["field_type"])?;
        let required = f["required"].as_bool().unwrap_or(false);

        fields.push(FieldDescriptor {
            field_number,
            name: fname,
            field_type,
            required,
        });
    }

    Ok(MessageDescriptor {
        name,
        package,
        fields,
    })
}

// ─── Time helper ─────────────────────────────────────────────────────────────

/// Return seconds since the Unix epoch.  Falls back to 0 on platforms without
/// a system clock (e.g. wasm32-unknown-unknown without WASI).
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_registry::types::{FieldDescriptor, FieldType, MessageDescriptor};

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn person_v1() -> MessageDescriptor {
        MessageDescriptor::new("Person", "example")
            .with_field(FieldDescriptor::optional(1, "id", FieldType::Int64))
            .with_field(FieldDescriptor::optional(2, "name", FieldType::String))
    }

    fn person_v2_new_optional_field() -> MessageDescriptor {
        MessageDescriptor::new("Person", "example")
            .with_field(FieldDescriptor::optional(1, "id", FieldType::Int64))
            .with_field(FieldDescriptor::optional(2, "name", FieldType::String))
            .with_field(FieldDescriptor::optional(3, "email", FieldType::String))
    }

    fn person_v2_type_change() -> MessageDescriptor {
        // Changes field 1 from Int64 → String — incompatible
        MessageDescriptor::new("Person", "example")
            .with_field(FieldDescriptor::optional(1, "id", FieldType::String))
            .with_field(FieldDescriptor::optional(2, "name", FieldType::String))
    }

    fn person_v2_field_removed() -> MessageDescriptor {
        // Removes field 2 — incompatible
        MessageDescriptor::new("Person", "example").with_field(FieldDescriptor::optional(
            1,
            "id",
            FieldType::Int64,
        ))
    }

    fn make_registry() -> SchemaRegistry {
        SchemaRegistry::new(RegistryConfig::default())
    }

    // ── Test 1: register and get ──────────────────────────────────────────────

    #[test]
    fn test_register_and_get() {
        let mut reg = make_registry();
        let id = reg.register(person_v1()).expect("register ok");
        let schema = reg.get(id).expect("get ok");
        assert_eq!(schema.id, id);
        assert_eq!(schema.version, SchemaVersion(1));
        assert_eq!(schema.descriptor.name, "Person");
    }

    // ── Test 2: register version compatible ──────────────────────────────────

    #[test]
    fn test_register_version_compatible() {
        let mut reg = make_registry();
        let id = reg.register(person_v1()).expect("register ok");
        let v = reg
            .register_version(id, person_v2_new_optional_field())
            .expect("version ok");
        assert_eq!(v, SchemaVersion(2));

        let schema = reg.get(id).expect("get latest");
        assert_eq!(schema.version, SchemaVersion(2));
        assert_eq!(schema.descriptor.fields.len(), 3);
    }

    // ── Test 3: incompatible — type change ───────────────────────────────────

    #[test]
    fn test_register_version_incompatible_type_change() {
        let mut reg = make_registry();
        let id = reg.register(person_v1()).expect("register ok");
        let err = reg
            .register_version(id, person_v2_type_change())
            .expect_err("should fail");
        assert!(matches!(err, SchemaRegistryError::IncompatibleEvolution(_)));
    }

    // ── Test 4: incompatible — field removal ─────────────────────────────────

    #[test]
    fn test_register_version_incompatible_field_removal() {
        let mut reg = make_registry();
        let id = reg.register(person_v1()).expect("register ok");
        let err = reg
            .register_version(id, person_v2_field_removed())
            .expect_err("should fail");
        assert!(matches!(err, SchemaRegistryError::IncompatibleEvolution(_)));
        assert!(err.to_string().contains("removed") || err.to_string().contains("removed"));
    }

    // ── Test 5: new optional field ok ─────────────────────────────────────────

    #[test]
    fn test_register_version_new_field_ok() {
        let mut reg = make_registry();
        let id = reg.register(person_v1()).expect("register ok");
        let v2 = reg
            .register_version(id, person_v2_new_optional_field())
            .expect("v2 ok");
        assert_eq!(v2.value(), 2);

        let schema_v2 = reg.get_version(id, SchemaVersion(2)).expect("v2 get");
        assert!(schema_v2.descriptor.field_by_name("email").is_some());
    }

    // ── Test 6: list schemas ──────────────────────────────────────────────────

    #[test]
    fn test_list_schemas() {
        let mut reg = make_registry();
        assert!(reg.list().is_empty());

        let id1 = reg.register(person_v1()).expect("r1");
        let id2 = reg
            .register(
                MessageDescriptor::new("Order", "shop").with_field(FieldDescriptor::optional(
                    1,
                    "order_id",
                    FieldType::UInt64,
                )),
            )
            .expect("r2");

        let ids = reg.list();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }

    // ── Test 7: schema evolution tracking ────────────────────────────────────

    #[test]
    fn test_schema_evolution_tracking() {
        let mut reg = make_registry();
        let id = reg.register(person_v1()).expect("r");

        let v2_desc = person_v2_new_optional_field();
        reg.register_version(id, v2_desc).expect("v2");

        // Add a third version: add a phone field
        let v3_desc = MessageDescriptor::new("Person", "example")
            .with_field(FieldDescriptor::optional(1, "id", FieldType::Int64))
            .with_field(FieldDescriptor::optional(2, "name", FieldType::String))
            .with_field(FieldDescriptor::optional(3, "email", FieldType::String))
            .with_field(FieldDescriptor::optional(4, "phone", FieldType::String));

        reg.register_version(id, v3_desc).expect("v3");

        let all = reg.all_versions(id).expect("versions");
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].version, SchemaVersion(1));
        assert_eq!(all[1].version, SchemaVersion(2));
        assert_eq!(all[2].version, SchemaVersion(3));
        assert_eq!(all[2].descriptor.fields.len(), 4);
    }

    // ── Test 8: export / import JSON roundtrip ────────────────────────────────

    #[test]
    fn test_export_import_json_roundtrip() {
        let mut reg = make_registry();
        let id = reg.register(person_v1()).expect("r");
        reg.register_version(id, person_v2_new_optional_field())
            .expect("v2");

        // Also register a second schema with repeated and message fields
        let complex_desc = MessageDescriptor::new("Invoice", "billing")
            .with_field(FieldDescriptor::optional(
                1,
                "invoice_id",
                FieldType::String,
            ))
            .with_field(FieldDescriptor::optional(2, "amount", FieldType::Double))
            .with_field(FieldDescriptor::optional(
                3,
                "tags",
                FieldType::Repeated(Box::new(FieldType::String)),
            ))
            .with_field(FieldDescriptor::optional(
                4,
                "address",
                FieldType::Message("Address".to_string()),
            ));
        reg.register(complex_desc).expect("invoice");

        let json = reg.export_json();
        assert!(!json.is_empty());
        assert!(json.contains("Person"));
        assert!(json.contains("Invoice"));

        let restored = SchemaRegistry::import_json(&json).expect("import ok");
        assert_eq!(restored.schema_count(), 2);
        assert_eq!(restored.version_count(), 3); // 2 versions for Person + 1 for Invoice

        let person_schema = restored.get(id).expect("person get");
        assert_eq!(person_schema.version.value(), 2);
        assert_eq!(person_schema.descriptor.fields.len(), 3);
    }

    // ── Test 9: varint roundtrip (delegated to wire module) ──────────────────

    #[test]
    fn test_encode_decode_varint_roundtrip() {
        use crate::schema_registry::wire::{decode_varint, encode_varint};
        let values = [0u64, 1, 127, 128, 300, u32::MAX as u64, u64::MAX];
        for v in values {
            let mut buf = Vec::new();
            encode_varint(v, &mut buf);
            let mut pos = 0;
            let decoded = decode_varint(&buf, &mut pos).expect("decode ok");
            assert_eq!(decoded, v, "roundtrip failed for {v}");
        }
    }

    // ── Test 10: encode / decode all field types ──────────────────────────────

    #[test]
    fn test_encode_decode_all_field_types() {
        use crate::schema_registry::types::FieldValue;
        use crate::schema_registry::wire::{decode_message, encode_message};

        let desc = MessageDescriptor::new("AllTypes", "test")
            .with_field(FieldDescriptor::optional(1, "i32", FieldType::Int32))
            .with_field(FieldDescriptor::optional(2, "i64", FieldType::Int64))
            .with_field(FieldDescriptor::optional(3, "u32", FieldType::UInt32))
            .with_field(FieldDescriptor::optional(4, "u64", FieldType::UInt64))
            .with_field(FieldDescriptor::optional(5, "flt", FieldType::Float))
            .with_field(FieldDescriptor::optional(6, "dbl", FieldType::Double))
            .with_field(FieldDescriptor::optional(7, "b", FieldType::Bool))
            .with_field(FieldDescriptor::optional(8, "s", FieldType::String))
            .with_field(FieldDescriptor::optional(9, "raw", FieldType::Bytes));

        let values: Vec<(u32, FieldValue)> = vec![
            (1, FieldValue::Int32(-42)),
            (2, FieldValue::Int64(i64::MIN)),
            (3, FieldValue::UInt32(0xdeadbeef)),
            (4, FieldValue::UInt64(u64::MAX)),
            (5, FieldValue::Float(1.0)),
            (6, FieldValue::Double(std::f64::consts::E)),
            (7, FieldValue::Bool(false)),
            (8, FieldValue::Str("world".to_string())),
            (9, FieldValue::Bytes(vec![0x01, 0x02, 0x03])),
        ];

        let bytes = encode_message(&desc, &values);
        let decoded = decode_message(&desc, &bytes).expect("decode");
        assert_eq!(decoded.len(), 9);
    }

    // ── Test 11: ProtoEncoder builder ────────────────────────────────────────

    #[test]
    fn test_proto_encoder_builder() {
        use crate::schema_registry::wire::{ProtoDecoder, ProtoEncoder, WireValue};

        let bytes = ProtoEncoder::new()
            .int32(1, -1)
            .string(2, "hello")
            .double(3, 99.9)
            .bool(4, true)
            .build();

        let mut dec = ProtoDecoder::new(&bytes);
        let fields = dec.collect_all().expect("ok");
        assert_eq!(fields.len(), 4);
        assert_eq!(fields[0].0, 1);
        assert_eq!(fields[1].0, 2);
        if let WireValue::LengthDelimited(ref b) = fields[1].1 {
            assert_eq!(b, b"hello");
        }
    }

    // ── Test 12: ProtoDecoder fields ─────────────────────────────────────────

    #[test]
    fn test_proto_decoder_fields() {
        use crate::schema_registry::wire::{ProtoDecoder, ProtoEncoder, WireValue};

        let bytes = ProtoEncoder::new()
            .uint64(5, 12345678)
            .bytes(6, b"\xff\xfe")
            .build();

        let mut dec = ProtoDecoder::new(&bytes);
        let f1 = dec.next_field().expect("f1").expect("ok");
        assert_eq!(f1.0, 5);
        assert_eq!(f1.1, WireValue::Varint(12345678));

        let f2 = dec.next_field().expect("f2").expect("ok");
        assert_eq!(f2.0, 6);
        if let WireValue::LengthDelimited(b) = f2.1 {
            assert_eq!(b, b"\xff\xfe");
        } else {
            panic!("expected LengthDelimited");
        }
        assert!(dec.is_empty());
    }

    // ── Test 13: wire type tag encoding ──────────────────────────────────────

    #[test]
    fn test_wire_type_tag_encoding() {
        use crate::schema_registry::wire::{encode_field, WireType};

        // field 1, varint → 0x08
        let mut buf = Vec::new();
        encode_field(1, WireType::Varint, &mut buf);
        assert_eq!(buf, [0x08]);

        // field 1, fixed32 → (1<<3)|5 = 0x0d
        let mut buf2 = Vec::new();
        encode_field(1, WireType::Fixed32, &mut buf2);
        assert_eq!(buf2, [0x0d]);

        // field 1, fixed64 → (1<<3)|1 = 0x09
        let mut buf3 = Vec::new();
        encode_field(1, WireType::Fixed64, &mut buf3);
        assert_eq!(buf3, [0x09]);

        // field 2, len-delim → (2<<3)|2 = 0x12
        let mut buf4 = Vec::new();
        encode_field(2, WireType::LengthDelimited, &mut buf4);
        assert_eq!(buf4, [0x12]);
    }

    // ── Test 14: validate no duplicate field numbers ──────────────────────────

    #[test]
    fn test_validate_no_duplicate_field_numbers() {
        use crate::schema_registry::validation::validate_descriptor;

        let desc = MessageDescriptor::new("Dup", "test")
            .with_field(FieldDescriptor::optional(1, "a", FieldType::Int32))
            .with_field(FieldDescriptor::optional(1, "b", FieldType::Int64));

        assert!(validate_descriptor(&desc).is_err());
    }

    // ── Test 15: validate reserved field numbers ──────────────────────────────

    #[test]
    fn test_validate_reserved_field_numbers() {
        use crate::schema_registry::validation::validate_descriptor;

        let desc = MessageDescriptor::new("Reserved", "test")
            .with_field(FieldDescriptor::optional(19_000, "bad", FieldType::Bool));
        assert!(validate_descriptor(&desc).is_err());
    }

    // ── Test 16: validate field number range ─────────────────────────────────

    #[test]
    fn test_validate_field_number_range() {
        use crate::schema_registry::validation::{validate_descriptor, MAX_FIELD_NUMBER};

        // Valid max
        let desc_max = MessageDescriptor::new("Max", "test").with_field(FieldDescriptor::optional(
            MAX_FIELD_NUMBER,
            "f",
            FieldType::Bool,
        ));
        assert!(validate_descriptor(&desc_max).is_ok());

        // One over max
        let desc_over = MessageDescriptor::new("Over", "test").with_field(
            FieldDescriptor::optional(MAX_FIELD_NUMBER + 1, "f", FieldType::Bool),
        );
        assert!(validate_descriptor(&desc_over).is_err());
    }

    // ── Test 17: RegistryConfig default ──────────────────────────────────────

    #[test]
    fn test_registry_config_default() {
        let cfg = RegistryConfig::default();
        assert_eq!(cfg.max_schemas, 1_000);
        assert!(cfg.allow_schema_evolution);
    }

    // ── Test 18: schema not found ─────────────────────────────────────────────

    #[test]
    fn test_schema_not_found() {
        let reg = make_registry();
        let err = reg.get(SchemaId(9999)).expect_err("should not find");
        assert!(matches!(err, SchemaRegistryError::NotFound(SchemaId(9999))));
    }

    // ── Test 19: message encode/decode roundtrip ──────────────────────────────

    #[test]
    fn test_message_encode_decode_roundtrip() {
        use crate::schema_registry::types::FieldValue;
        use crate::schema_registry::wire::{decode_message, encode_message};

        let desc = MessageDescriptor::new("Coords", "geo")
            .with_field(FieldDescriptor::optional(1, "lat", FieldType::Double))
            .with_field(FieldDescriptor::optional(2, "lon", FieldType::Double))
            .with_field(FieldDescriptor::optional(3, "label", FieldType::String));

        let values = vec![
            (1, FieldValue::Double(48.8566)),
            (2, FieldValue::Double(2.3522)),
            (3, FieldValue::Str("Paris".to_string())),
        ];

        let encoded = encode_message(&desc, &values);
        let decoded = decode_message(&desc, &encoded).expect("decode ok");

        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[2].0, "label");
        assert_eq!(decoded[2].1, FieldValue::Str("Paris".to_string()));

        if let FieldValue::Double(lat) = decoded[0].1 {
            assert!((lat - 48.8566).abs() < 1e-9);
        } else {
            panic!("expected Double for lat");
        }
    }

    // ── Test 20: nested message encoding ─────────────────────────────────────

    #[test]
    fn test_nested_message_encoding() {
        use crate::schema_registry::types::FieldValue;
        use crate::schema_registry::wire::{decode_message, encode_message};

        let inner_desc = MessageDescriptor::new("Tag", "meta")
            .with_field(FieldDescriptor::optional(1, "key", FieldType::String))
            .with_field(FieldDescriptor::optional(2, "value", FieldType::String));

        let inner_values = vec![
            (1, FieldValue::Str("env".to_string())),
            (2, FieldValue::Str("prod".to_string())),
        ];
        let inner_bytes = encode_message(&inner_desc, &inner_values);

        let outer_desc = MessageDescriptor::new("Config", "meta")
            .with_field(FieldDescriptor::optional(1, "name", FieldType::String))
            .with_field(FieldDescriptor::optional(
                2,
                "tag",
                FieldType::Message("Tag".to_string()),
            ));

        let outer_values = vec![
            (1, FieldValue::Str("app_config".to_string())),
            (2, FieldValue::Message(inner_bytes.clone())),
        ];

        let outer_bytes = encode_message(&outer_desc, &outer_values);
        let outer_decoded = decode_message(&outer_desc, &outer_bytes).expect("ok");

        assert_eq!(
            outer_decoded[0].1,
            FieldValue::Str("app_config".to_string())
        );
        if let FieldValue::Message(payload) = &outer_decoded[1].1 {
            // Decode inner from payload
            let inner_decoded = decode_message(&inner_desc, payload).expect("inner ok");
            assert_eq!(inner_decoded[0].1, FieldValue::Str("env".to_string()));
            assert_eq!(inner_decoded[1].1, FieldValue::Str("prod".to_string()));
        } else {
            panic!("expected Message variant for 'tag'");
        }
    }

    // ── Test 21: repeated field encoding ─────────────────────────────────────

    #[test]
    fn test_repeated_field_encoding() {
        use crate::schema_registry::types::FieldValue;
        use crate::schema_registry::wire::{decode_message, encode_message, encode_varint};

        let desc = MessageDescriptor::new("NumList", "test").with_field(FieldDescriptor::optional(
            1,
            "numbers",
            FieldType::Repeated(Box::new(FieldType::Int32)),
        ));

        // Pack [10, 20, 30] as varints
        let mut packed = Vec::new();
        for v in [10u64, 20, 30] {
            encode_varint(v, &mut packed);
        }

        let values = vec![(1, FieldValue::Bytes(packed.clone()))];
        let bytes = encode_message(&desc, &values);
        let decoded = decode_message(&desc, &bytes).expect("ok");

        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].0, "numbers");
        if let FieldValue::Bytes(b) = &decoded[0].1 {
            assert_eq!(b, &packed);
        } else {
            panic!("expected Bytes for repeated field");
        }
    }

    // ── Test 22: backward-compatible evolution ────────────────────────────────

    #[test]
    fn test_backward_compatible_evolution() {
        let v1 = MessageDescriptor::new("Event", "analytics")
            .with_field(FieldDescriptor::optional(1, "event_id", FieldType::String))
            .with_field(FieldDescriptor::optional(2, "timestamp", FieldType::Int64));

        let v2 = MessageDescriptor::new("Event", "analytics")
            .with_field(FieldDescriptor::optional(1, "event_id", FieldType::String))
            .with_field(FieldDescriptor::optional(2, "timestamp", FieldType::Int64))
            .with_field(FieldDescriptor::optional(3, "user_id", FieldType::String))
            .with_field(FieldDescriptor::optional(4, "metadata", FieldType::Bytes));

        let v3 = MessageDescriptor::new("Event", "analytics")
            .with_field(FieldDescriptor::optional(1, "event_id", FieldType::String))
            .with_field(FieldDescriptor::optional(2, "timestamp", FieldType::Int64))
            .with_field(FieldDescriptor::optional(3, "user_id", FieldType::String))
            .with_field(FieldDescriptor::optional(4, "metadata", FieldType::Bytes))
            .with_field(FieldDescriptor::optional(
                5,
                "properties",
                FieldType::Repeated(Box::new(FieldType::String)),
            ));

        assert!(is_compatible(&v1, &v2));
        assert!(is_compatible(&v2, &v3));
        // v1 → v3 direct: also compatible
        assert!(is_compatible(&v1, &v3));

        let mut reg = make_registry();
        let id = reg.register(v1).expect("v1");
        reg.register_version(id, v2).expect("v2");
        reg.register_version(id, v3).expect("v3");

        assert_eq!(reg.get(id).expect("latest").version, SchemaVersion(3));
        assert_eq!(reg.all_versions(id).expect("all").len(), 3);
    }
}
