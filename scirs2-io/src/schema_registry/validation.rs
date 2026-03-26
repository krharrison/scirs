//! Structural validation for Protocol Buffer message descriptors and field values.
//!
//! The rules enforced here mirror the constraints defined by the Protocol Buffer
//! language specification:
//!
//! * Field numbers must be in the range `[1, 536_870_911]`
//!   (the upper bound is `2²⁹ − 1`).
//! * Field numbers `[19_000, 19_999]` are reserved by Google for internal
//!   runtime use and must not appear in user-defined schemas.
//! * Within a single message, field numbers must be unique.
//! * Within a single message, field names must be unique (case-sensitive).

use super::types::{
    FieldDescriptor, FieldType, FieldValue, MessageDescriptor, SchemaRegistryError,
    SchemaRegistryResult,
};
use std::collections::HashSet;

// ─── Constants ───────────────────────────────────────────────────────────────

/// Maximum legal field number per the Protocol Buffer specification.
pub const MAX_FIELD_NUMBER: u32 = 536_870_911; // 2²⁹ − 1

/// First reserved field number (inclusive).
pub const RESERVED_RANGE_START: u32 = 19_000;

/// Last reserved field number (inclusive).
pub const RESERVED_RANGE_END: u32 = 19_999;

// ─── Descriptor validation ───────────────────────────────────────────────────

/// Validate a [`MessageDescriptor`] for structural correctness.
///
/// Returns `Ok(())` if every field:
///
/// * has a field number in `[1, MAX_FIELD_NUMBER]`,
/// * has a field number outside the reserved range `[19_000, 19_999]`,
/// * has a field number unique within the message,
/// * has a name unique within the message.
///
/// The descriptor is also checked for a non-empty message name.
pub fn validate_descriptor(desc: &MessageDescriptor) -> SchemaRegistryResult<()> {
    if desc.name.is_empty() {
        return Err(SchemaRegistryError::Validation(
            "message name must not be empty".to_string(),
        ));
    }

    let mut seen_numbers: HashSet<u32> = HashSet::new();
    let mut seen_names: HashSet<&str> = HashSet::new();

    for field in &desc.fields {
        validate_field_number(field)?;

        if !seen_numbers.insert(field.field_number) {
            return Err(SchemaRegistryError::Validation(format!(
                "duplicate field number {} in message '{}'",
                field.field_number, desc.name
            )));
        }

        if !seen_names.insert(field.name.as_str()) {
            return Err(SchemaRegistryError::Validation(format!(
                "duplicate field name '{}' in message '{}'",
                field.name, desc.name
            )));
        }

        validate_field_name(field)?;
    }

    Ok(())
}

/// Validate the field number of a single field descriptor.
fn validate_field_number(field: &FieldDescriptor) -> SchemaRegistryResult<()> {
    if field.field_number == 0 {
        return Err(SchemaRegistryError::Validation(format!(
            "field '{}': field number 0 is not allowed (must be ≥ 1)",
            field.name
        )));
    }

    if field.field_number > MAX_FIELD_NUMBER {
        return Err(SchemaRegistryError::Validation(format!(
            "field '{}': field number {} exceeds the maximum of {}",
            field.name, field.field_number, MAX_FIELD_NUMBER
        )));
    }

    if (RESERVED_RANGE_START..=RESERVED_RANGE_END).contains(&field.field_number) {
        return Err(SchemaRegistryError::Validation(format!(
            "field '{}': field number {} is in the reserved range [{}, {}]",
            field.name, field.field_number, RESERVED_RANGE_START, RESERVED_RANGE_END
        )));
    }

    Ok(())
}

/// Validate that the field name is not empty.
fn validate_field_name(field: &FieldDescriptor) -> SchemaRegistryResult<()> {
    if field.name.is_empty() {
        return Err(SchemaRegistryError::Validation(format!(
            "field number {}: field name must not be empty",
            field.field_number
        )));
    }
    Ok(())
}

// ─── Field value type check ───────────────────────────────────────────────────

/// Check whether `value` is type-compatible with the declared `field`.
///
/// Returns `true` if the value's runtime type matches the descriptor's
/// [`FieldType`], or if the descriptor type is `Repeated` and the value is
/// `Bytes` (repeated fields are pre-encoded as a length-delimited byte
/// sequence).
///
/// This is a *permissive* check — it deliberately allows int32 values to be
/// stored in int64 fields (and vice-versa) to accommodate common widening
/// patterns seen in schema evolution.
pub fn validate_field_value(field: &FieldDescriptor, value: &FieldValue) -> bool {
    type_compatible(&field.field_type, value)
}

fn type_compatible(ft: &FieldType, value: &FieldValue) -> bool {
    match (ft, value) {
        // Exact matches
        (FieldType::Int32, FieldValue::Int32(_)) => true,
        (FieldType::Int64, FieldValue::Int64(_)) => true,
        (FieldType::UInt32, FieldValue::UInt32(_)) => true,
        (FieldType::UInt64, FieldValue::UInt64(_)) => true,
        (FieldType::Float, FieldValue::Float(_)) => true,
        (FieldType::Double, FieldValue::Double(_)) => true,
        (FieldType::Bool, FieldValue::Bool(_)) => true,
        (FieldType::String, FieldValue::Str(_)) => true,
        (FieldType::Bytes, FieldValue::Bytes(_)) => true,
        (FieldType::Message(_), FieldValue::Message(_)) => true,
        // Widening: int32 value stored in int64 field
        (FieldType::Int64, FieldValue::Int32(_)) => true,
        // Widening: uint32 value stored in uint64 field
        (FieldType::UInt64, FieldValue::UInt32(_)) => true,
        // Widening: float value stored in double field
        (FieldType::Double, FieldValue::Float(_)) => true,
        // Repeated fields carry their payload as raw bytes
        (FieldType::Repeated(_), FieldValue::Bytes(_)) => true,
        _ => false,
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_registry::types::{FieldDescriptor, FieldType, MessageDescriptor};

    fn make_descriptor(fields: Vec<FieldDescriptor>) -> MessageDescriptor {
        MessageDescriptor {
            name: "TestMsg".to_string(),
            package: "test".to_string(),
            fields,
        }
    }

    #[test]
    fn test_valid_descriptor_passes() {
        let desc = make_descriptor(vec![
            FieldDescriptor::optional(1, "id", FieldType::Int64),
            FieldDescriptor::optional(2, "name", FieldType::String),
        ]);
        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_duplicate_field_number_rejected() {
        let desc = make_descriptor(vec![
            FieldDescriptor::optional(1, "a", FieldType::Int32),
            FieldDescriptor::optional(1, "b", FieldType::Int32),
        ]);
        let err = validate_descriptor(&desc).unwrap_err();
        assert!(matches!(err, SchemaRegistryError::Validation(_)));
        assert!(err.to_string().contains("duplicate field number"));
    }

    #[test]
    fn test_duplicate_field_name_rejected() {
        let desc = make_descriptor(vec![
            FieldDescriptor::optional(1, "value", FieldType::Int32),
            FieldDescriptor::optional(2, "value", FieldType::Int64),
        ]);
        let err = validate_descriptor(&desc).unwrap_err();
        assert!(matches!(err, SchemaRegistryError::Validation(_)));
        assert!(err.to_string().contains("duplicate field name"));
    }

    #[test]
    fn test_field_number_zero_rejected() {
        let desc = make_descriptor(vec![FieldDescriptor::optional(0, "bad", FieldType::Bool)]);
        assert!(validate_descriptor(&desc).is_err());
    }

    #[test]
    fn test_field_number_exceeds_max_rejected() {
        let desc = make_descriptor(vec![FieldDescriptor::optional(
            MAX_FIELD_NUMBER + 1,
            "big",
            FieldType::Bool,
        )]);
        assert!(validate_descriptor(&desc).is_err());
    }

    #[test]
    fn test_reserved_range_rejected() {
        for n in [19_000u32, 19_500, 19_999] {
            let desc = make_descriptor(vec![FieldDescriptor::optional(
                n,
                "reserved",
                FieldType::Bool,
            )]);
            let err = validate_descriptor(&desc).unwrap_err();
            assert!(
                err.to_string().contains("reserved range"),
                "field_number={n}"
            );
        }
    }

    #[test]
    fn test_boundary_just_outside_reserved_range_ok() {
        let below = make_descriptor(vec![FieldDescriptor::optional(
            18_999,
            "below",
            FieldType::Bool,
        )]);
        assert!(validate_descriptor(&below).is_ok());

        let above = make_descriptor(vec![FieldDescriptor::optional(
            20_000,
            "above",
            FieldType::Bool,
        )]);
        assert!(validate_descriptor(&above).is_ok());
    }

    #[test]
    fn test_empty_message_name_rejected() {
        let desc = MessageDescriptor {
            name: std::string::String::new(),
            package: "pkg".to_string(),
            fields: vec![],
        };
        assert!(validate_descriptor(&desc).is_err());
    }

    #[test]
    fn test_validate_field_value_exact_types() {
        let fd = FieldDescriptor::optional(1, "x", FieldType::Int32);
        assert!(validate_field_value(&fd, &FieldValue::Int32(42)));
        assert!(!validate_field_value(&fd, &FieldValue::Int64(42)));
    }

    #[test]
    fn test_validate_field_value_widening_allowed() {
        let fd = FieldDescriptor::optional(1, "x", FieldType::Int64);
        assert!(validate_field_value(&fd, &FieldValue::Int32(42)));
    }

    #[test]
    fn test_validate_field_value_repeated_bytes() {
        let fd =
            FieldDescriptor::optional(1, "items", FieldType::Repeated(Box::new(FieldType::Int32)));
        assert!(validate_field_value(
            &fd,
            &FieldValue::Bytes(vec![0x01, 0x02])
        ));
    }
}
