//! Pure-Rust Protocol Buffer wire format encoder and decoder.
//!
//! This module implements the proto3 wire format at the level required by the
//! schema registry: varint encoding/decoding, field tagging, fixed-width
//! fields, and length-delimited payloads.  It deliberately avoids the `prost`
//! crate to remain dependency-free (COOLJAPAN pure-Rust policy).
//!
//! # Wire type table
//!
//! | ID | Meaning          | Rust type     |
//! |----|------------------|---------------|
//! | 0  | Varint           | u64           |
//! | 1  | 64-bit fixed     | [u8; 8]       |
//! | 2  | Length-delimited | `Vec<u8>`     |
//! | 5  | 32-bit fixed     | [u8; 4]       |
//!
//! Wire types 3 and 4 (start/end group) are not supported.

use super::types::{
    FieldDescriptor, FieldType, FieldValue, MessageDescriptor, SchemaRegistryError,
    SchemaRegistryResult,
};

// ─── Wire type ───────────────────────────────────────────────────────────────

/// The four wire types used in this implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WireType {
    /// Wire type 0 — variable-length integer (LEB-128).
    Varint,
    /// Wire type 1 — 64-bit little-endian fixed-width.
    Fixed64,
    /// Wire type 2 — length-delimited byte sequence.
    LengthDelimited,
    /// Wire type 5 — 32-bit little-endian fixed-width.
    Fixed32,
}

impl WireType {
    /// Return the numeric wire-type id as used in the proto tag varint.
    pub fn id(self) -> u32 {
        match self {
            WireType::Varint => 0,
            WireType::Fixed64 => 1,
            WireType::LengthDelimited => 2,
            WireType::Fixed32 => 5,
        }
    }

    /// Parse a numeric wire-type id.
    pub fn from_id(id: u32) -> SchemaRegistryResult<Self> {
        match id {
            0 => Ok(WireType::Varint),
            1 => Ok(WireType::Fixed64),
            2 => Ok(WireType::LengthDelimited),
            5 => Ok(WireType::Fixed32),
            _ => Err(SchemaRegistryError::WireFormat(format!(
                "unknown wire type id: {id}"
            ))),
        }
    }

    /// Return the wire type implied by a [`FieldType`].
    pub fn for_field_type(ft: &FieldType) -> Self {
        match ft {
            FieldType::Int32
            | FieldType::Int64
            | FieldType::UInt32
            | FieldType::UInt64
            | FieldType::Bool => WireType::Varint,
            FieldType::Float => WireType::Fixed32,
            FieldType::Double => WireType::Fixed64,
            FieldType::String
            | FieldType::Bytes
            | FieldType::Message(_)
            | FieldType::Repeated(_) => WireType::LengthDelimited,
        }
    }
}

// ─── WireValue ────────────────────────────────────────────────────────────────

/// A raw decoded wire value before schema-guided interpretation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum WireValue {
    /// Wire type 0 payload.
    Varint(u64),
    /// Wire type 1 payload (little-endian bytes).
    Fixed64([u8; 8]),
    /// Wire type 2 payload.
    LengthDelimited(Vec<u8>),
    /// Wire type 5 payload (little-endian bytes).
    Fixed32([u8; 4]),
}

// ─── Varint primitives ───────────────────────────────────────────────────────

/// Encode `value` as a Protocol Buffer varint (LEB-128 variant) and append it
/// to `buf`.
pub fn encode_varint(value: u64, buf: &mut Vec<u8>) {
    let mut v = value;
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            buf.push(byte);
            break;
        }
        buf.push(byte | 0x80);
    }
}

/// Decode a varint from `buf` starting at `*pos`.  On success, advances
/// `*pos` past the varint bytes and returns the decoded value.  Returns
/// `None` on truncated input or overflow.
pub fn decode_varint(buf: &[u8], pos: &mut usize) -> Option<u64> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    loop {
        if *pos >= buf.len() || shift >= 64 {
            return None;
        }
        let byte = buf[*pos];
        *pos += 1;
        result |= ((byte & 0x7f) as u64) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            return Some(result);
        }
    }
}

// ─── Field tag encoding ──────────────────────────────────────────────────────

/// Encode the field tag (field_number << 3 | wire_type) and append to `buf`.
pub fn encode_field(field_number: u32, wire_type: WireType, buf: &mut Vec<u8>) {
    let tag: u64 = ((field_number as u64) << 3) | (wire_type.id() as u64);
    encode_varint(tag, buf);
}

// ─── ProtoEncoder ────────────────────────────────────────────────────────────

/// Fluent builder for constructing Protocol Buffer messages byte-by-byte.
///
/// Each method encodes one field (tag + payload) and returns `&mut Self` for
/// chaining.  Call [`build`](ProtoEncoder::build) to extract the completed byte
/// vector.
///
/// # Example
///
/// ```rust
/// use scirs2_io::schema_registry::wire::ProtoEncoder;
///
/// let bytes = ProtoEncoder::new()
///     .int64(1, 12345)
///     .string(2, "hello world")
///     .build();
/// assert!(!bytes.is_empty());
/// ```
#[derive(Debug, Default)]
pub struct ProtoEncoder {
    buf: Vec<u8>,
}

impl ProtoEncoder {
    /// Create a new, empty encoder.
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Encode a signed 32-bit integer as a varint (wire type 0).
    pub fn int32(mut self, field_number: u32, value: i32) -> Self {
        encode_field(field_number, WireType::Varint, &mut self.buf);
        encode_varint(value as u64, &mut self.buf);
        self
    }

    /// Encode a signed 64-bit integer as a varint (wire type 0).
    pub fn int64(mut self, field_number: u32, value: i64) -> Self {
        encode_field(field_number, WireType::Varint, &mut self.buf);
        encode_varint(value as u64, &mut self.buf);
        self
    }

    /// Encode an unsigned 32-bit integer as a varint (wire type 0).
    pub fn uint32(mut self, field_number: u32, value: u32) -> Self {
        encode_field(field_number, WireType::Varint, &mut self.buf);
        encode_varint(value as u64, &mut self.buf);
        self
    }

    /// Encode an unsigned 64-bit integer as a varint (wire type 0).
    pub fn uint64(mut self, field_number: u32, value: u64) -> Self {
        encode_field(field_number, WireType::Varint, &mut self.buf);
        encode_varint(value, &mut self.buf);
        self
    }

    /// Encode a boolean as a varint (wire type 0).
    pub fn bool(mut self, field_number: u32, value: bool) -> Self {
        encode_field(field_number, WireType::Varint, &mut self.buf);
        encode_varint(value as u64, &mut self.buf);
        self
    }

    /// Encode a 32-bit float (wire type 5).
    pub fn float(mut self, field_number: u32, value: f32) -> Self {
        encode_field(field_number, WireType::Fixed32, &mut self.buf);
        self.buf.extend_from_slice(&value.to_le_bytes());
        self
    }

    /// Encode a 64-bit float (wire type 1).
    pub fn double(mut self, field_number: u32, value: f64) -> Self {
        encode_field(field_number, WireType::Fixed64, &mut self.buf);
        self.buf.extend_from_slice(&value.to_le_bytes());
        self
    }

    /// Encode a UTF-8 string as a length-delimited field (wire type 2).
    pub fn string(mut self, field_number: u32, value: &str) -> Self {
        self.write_length_delimited(field_number, value.as_bytes());
        self
    }

    /// Encode a raw byte sequence as a length-delimited field (wire type 2).
    pub fn bytes(mut self, field_number: u32, value: &[u8]) -> Self {
        self.write_length_delimited(field_number, value);
        self
    }

    /// Encode a pre-encoded nested message as a length-delimited field (wire type 2).
    pub fn message(mut self, field_number: u32, nested_bytes: &[u8]) -> Self {
        self.write_length_delimited(field_number, nested_bytes);
        self
    }

    /// Consume the encoder and return the completed byte vector.
    pub fn build(self) -> Vec<u8> {
        self.buf
    }

    // ── Internal helpers ────────────────────────────────────────────────────

    fn write_length_delimited(&mut self, field_number: u32, data: &[u8]) {
        encode_field(field_number, WireType::LengthDelimited, &mut self.buf);
        encode_varint(data.len() as u64, &mut self.buf);
        self.buf.extend_from_slice(data);
    }
}

// ─── ProtoDecoder ────────────────────────────────────────────────────────────

/// Iterator-style decoder over a Protocol Buffer wire-format byte slice.
///
/// Each call to [`next_field`](ProtoDecoder::next_field) consumes one complete
/// field (tag + payload) from the buffer and returns `(field_number, WireValue)`.
pub struct ProtoDecoder<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ProtoDecoder<'a> {
    /// Create a new decoder over `buf`.
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Returns `true` when the entire buffer has been consumed.
    pub fn is_empty(&self) -> bool {
        self.pos >= self.buf.len()
    }

    /// Decode the next field from the buffer.
    ///
    /// Returns `None` at end-of-buffer, or `Some(Err(_))` on a malformed
    /// message.
    pub fn next_field(&mut self) -> Option<SchemaRegistryResult<(u32, WireValue)>> {
        if self.is_empty() {
            return None;
        }

        // Decode tag varint
        let tag = decode_varint(self.buf, &mut self.pos)?;
        let wire_type_id = (tag & 0x07) as u32;
        let field_number = (tag >> 3) as u32;

        let wire_type = match WireType::from_id(wire_type_id) {
            Ok(wt) => wt,
            Err(e) => return Some(Err(e)),
        };

        let value = match self.decode_payload(wire_type) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };

        Some(Ok((field_number, value)))
    }

    /// Decode all remaining fields into a `Vec`.
    pub fn collect_all(&mut self) -> SchemaRegistryResult<Vec<(u32, WireValue)>> {
        let mut out = Vec::new();
        while let Some(result) = self.next_field() {
            out.push(result?);
        }
        Ok(out)
    }

    // ── Internal helpers ────────────────────────────────────────────────────

    fn decode_payload(&mut self, wire_type: WireType) -> SchemaRegistryResult<WireValue> {
        match wire_type {
            WireType::Varint => {
                let v = decode_varint(self.buf, &mut self.pos).ok_or_else(|| {
                    SchemaRegistryError::WireFormat("truncated varint".to_string())
                })?;
                Ok(WireValue::Varint(v))
            }
            WireType::Fixed64 => {
                if self.pos + 8 > self.buf.len() {
                    return Err(SchemaRegistryError::WireFormat(
                        "truncated fixed64 field".to_string(),
                    ));
                }
                let mut b = [0u8; 8];
                b.copy_from_slice(&self.buf[self.pos..self.pos + 8]);
                self.pos += 8;
                Ok(WireValue::Fixed64(b))
            }
            WireType::LengthDelimited => {
                let len = decode_varint(self.buf, &mut self.pos).ok_or_else(|| {
                    SchemaRegistryError::WireFormat(
                        "truncated length prefix in length-delimited field".to_string(),
                    )
                })? as usize;

                if self.pos + len > self.buf.len() {
                    return Err(SchemaRegistryError::WireFormat(format!(
                        "length-delimited field claims {len} bytes but only {} remain",
                        self.buf.len() - self.pos
                    )));
                }
                let payload = self.buf[self.pos..self.pos + len].to_vec();
                self.pos += len;
                Ok(WireValue::LengthDelimited(payload))
            }
            WireType::Fixed32 => {
                if self.pos + 4 > self.buf.len() {
                    return Err(SchemaRegistryError::WireFormat(
                        "truncated fixed32 field".to_string(),
                    ));
                }
                let mut b = [0u8; 4];
                b.copy_from_slice(&self.buf[self.pos..self.pos + 4]);
                self.pos += 4;
                Ok(WireValue::Fixed32(b))
            }
        }
    }
}

// ─── Schema-guided encode / decode ────────────────────────────────────────────

/// Encode a set of `(field_number, FieldValue)` pairs according to the message
/// descriptor and return the wire-format bytes.
///
/// Fields not declared in the descriptor are silently skipped.  The field
/// values are encoded in the order they appear in `values`.
pub fn encode_message(schema: &MessageDescriptor, values: &[(u32, FieldValue)]) -> Vec<u8> {
    let mut buf = Vec::new();

    for (field_number, value) in values {
        // Only encode fields that appear in the descriptor.
        let field_desc = match schema.field_by_number(*field_number) {
            Some(fd) => fd,
            None => continue,
        };

        encode_field_value(*field_number, &field_desc.field_type, value, &mut buf);
    }

    buf
}

/// Decode wire-format `bytes` into a list of `(field_name, FieldValue)` pairs
/// using the supplied message descriptor for type guidance.
///
/// Unknown field numbers (not in the descriptor) are silently skipped — this
/// mirrors proto3 unknown-field handling and allows forward-compatible readers.
pub fn decode_message(
    schema: &MessageDescriptor,
    bytes: &[u8],
) -> SchemaRegistryResult<Vec<(std::string::String, FieldValue)>> {
    let mut decoder = ProtoDecoder::new(bytes);
    let raw_fields = decoder.collect_all()?;
    let mut out = Vec::new();

    for (field_number, wire_value) in raw_fields {
        let field_desc = match schema.field_by_number(field_number) {
            Some(fd) => fd,
            None => continue, // Unknown field — skip
        };

        let field_value = wire_value_to_field_value(&field_desc.field_type, wire_value)?;
        out.push((field_desc.name.clone(), field_value));
    }

    Ok(out)
}

// ─── Internal helpers ────────────────────────────────────────────────────────

fn encode_field_value(
    field_number: u32,
    field_type: &FieldType,
    value: &FieldValue,
    buf: &mut Vec<u8>,
) {
    match (field_type, value) {
        (FieldType::Int32, FieldValue::Int32(v)) => {
            encode_field(field_number, WireType::Varint, buf);
            encode_varint(*v as u64, buf);
        }
        (FieldType::Int64, FieldValue::Int64(v)) => {
            encode_field(field_number, WireType::Varint, buf);
            encode_varint(*v as u64, buf);
        }
        (FieldType::Int64, FieldValue::Int32(v)) => {
            // Widening: int32 into int64 field
            encode_field(field_number, WireType::Varint, buf);
            encode_varint(*v as u64, buf);
        }
        (FieldType::UInt32, FieldValue::UInt32(v)) => {
            encode_field(field_number, WireType::Varint, buf);
            encode_varint(*v as u64, buf);
        }
        (FieldType::UInt64, FieldValue::UInt64(v)) => {
            encode_field(field_number, WireType::Varint, buf);
            encode_varint(*v, buf);
        }
        (FieldType::UInt64, FieldValue::UInt32(v)) => {
            encode_field(field_number, WireType::Varint, buf);
            encode_varint(*v as u64, buf);
        }
        (FieldType::Bool, FieldValue::Bool(v)) => {
            encode_field(field_number, WireType::Varint, buf);
            encode_varint(*v as u64, buf);
        }
        (FieldType::Float, FieldValue::Float(v)) => {
            encode_field(field_number, WireType::Fixed32, buf);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        (FieldType::Double, FieldValue::Double(v)) => {
            encode_field(field_number, WireType::Fixed64, buf);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        (FieldType::Double, FieldValue::Float(v)) => {
            encode_field(field_number, WireType::Fixed64, buf);
            buf.extend_from_slice(&(*v as f64).to_le_bytes());
        }
        (FieldType::String, FieldValue::Str(s)) => {
            let data = s.as_bytes();
            encode_field(field_number, WireType::LengthDelimited, buf);
            encode_varint(data.len() as u64, buf);
            buf.extend_from_slice(data);
        }
        (FieldType::Bytes, FieldValue::Bytes(data)) => {
            encode_field(field_number, WireType::LengthDelimited, buf);
            encode_varint(data.len() as u64, buf);
            buf.extend_from_slice(data);
        }
        (FieldType::Message(_), FieldValue::Message(data)) => {
            encode_field(field_number, WireType::LengthDelimited, buf);
            encode_varint(data.len() as u64, buf);
            buf.extend_from_slice(data);
        }
        (FieldType::Repeated(_), FieldValue::Bytes(data)) => {
            // Pre-encoded repeated/packed field
            encode_field(field_number, WireType::LengthDelimited, buf);
            encode_varint(data.len() as u64, buf);
            buf.extend_from_slice(data);
        }
        _ => {
            // Type mismatch — silently skip (caller should validate first)
        }
    }
}

fn wire_value_to_field_value(ft: &FieldType, wv: WireValue) -> SchemaRegistryResult<FieldValue> {
    match (ft, wv) {
        (FieldType::Int32, WireValue::Varint(v)) => Ok(FieldValue::Int32(v as i32)),
        (FieldType::Int64, WireValue::Varint(v)) => Ok(FieldValue::Int64(v as i64)),
        (FieldType::UInt32, WireValue::Varint(v)) => Ok(FieldValue::UInt32(v as u32)),
        (FieldType::UInt64, WireValue::Varint(v)) => Ok(FieldValue::UInt64(v)),
        (FieldType::Bool, WireValue::Varint(v)) => Ok(FieldValue::Bool(v != 0)),
        (FieldType::Float, WireValue::Fixed32(b)) => Ok(FieldValue::Float(f32::from_le_bytes(b))),
        (FieldType::Double, WireValue::Fixed64(b)) => Ok(FieldValue::Double(f64::from_le_bytes(b))),
        (FieldType::String, WireValue::LengthDelimited(data)) => {
            let s = std::string::String::from_utf8(data).map_err(|e| {
                SchemaRegistryError::WireFormat(format!("invalid UTF-8 in string field: {e}"))
            })?;
            Ok(FieldValue::Str(s))
        }
        (FieldType::Bytes, WireValue::LengthDelimited(data)) => Ok(FieldValue::Bytes(data)),
        (FieldType::Message(_), WireValue::LengthDelimited(data)) => Ok(FieldValue::Message(data)),
        (FieldType::Repeated(_), WireValue::LengthDelimited(data)) => Ok(FieldValue::Bytes(data)),
        (ft, wv) => Err(SchemaRegistryError::WireFormat(format!(
            "wire type mismatch for field type {}: got {:?}",
            ft.proto_name(),
            wv
        ))),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Varint ────────────────────────────────────────────────────────────────

    #[test]
    fn test_varint_zero() {
        let mut buf = Vec::new();
        encode_varint(0, &mut buf);
        assert_eq!(buf, [0x00]);
        let mut pos = 0;
        assert_eq!(decode_varint(&buf, &mut pos), Some(0));
        assert_eq!(pos, 1);
    }

    #[test]
    fn test_varint_one_byte_boundary() {
        let mut buf = Vec::new();
        encode_varint(127, &mut buf);
        assert_eq!(buf, [0x7f]);
    }

    #[test]
    fn test_varint_two_byte_boundary() {
        let mut buf = Vec::new();
        encode_varint(128, &mut buf);
        assert_eq!(buf, [0x80, 0x01]);
    }

    #[test]
    fn test_varint_300() {
        let mut buf = Vec::new();
        encode_varint(300, &mut buf);
        assert_eq!(buf, [0xac, 0x02]);
        let mut pos = 0;
        assert_eq!(decode_varint(&buf, &mut pos), Some(300));
    }

    #[test]
    fn test_varint_u64_max() {
        let mut buf = Vec::new();
        encode_varint(u64::MAX, &mut buf);
        assert_eq!(buf.len(), 10);
        let mut pos = 0;
        assert_eq!(decode_varint(&buf, &mut pos), Some(u64::MAX));
    }

    #[test]
    fn test_varint_roundtrip_sequence() {
        let values: &[u64] = &[0, 1, 127, 128, 255, 1024, 65535, 1 << 32, u64::MAX];
        for &v in values {
            let mut buf = Vec::new();
            encode_varint(v, &mut buf);
            let mut pos = 0;
            assert_eq!(decode_varint(&buf, &mut pos), Some(v), "value={v}");
        }
    }

    #[test]
    fn test_decode_varint_truncated_returns_none() {
        // A multi-byte varint that is cut short
        let buf = [0x80]; // continuation bit set but no following byte
        let mut pos = 0;
        assert_eq!(decode_varint(&buf, &mut pos), None);
    }

    // ── Field tag encoding ────────────────────────────────────────────────────

    #[test]
    fn test_wire_type_tag_field_1_varint() {
        // field 1, wire type 0 → tag = (1 << 3) | 0 = 8
        let mut buf = Vec::new();
        encode_field(1, WireType::Varint, &mut buf);
        assert_eq!(buf, [0x08]);
    }

    #[test]
    fn test_wire_type_tag_field_2_len_delim() {
        // field 2, wire type 2 → tag = (2 << 3) | 2 = 18
        let mut buf = Vec::new();
        encode_field(2, WireType::LengthDelimited, &mut buf);
        assert_eq!(buf, [0x12]);
    }

    // ── ProtoEncoder ─────────────────────────────────────────────────────────

    #[test]
    fn test_proto_encoder_int32() {
        let bytes = ProtoEncoder::new().int32(1, 150).build();
        // tag = 0x08, varint 150 = [0x96, 0x01]
        assert_eq!(bytes, [0x08, 0x96, 0x01]);
    }

    #[test]
    fn test_proto_encoder_string() {
        let bytes = ProtoEncoder::new().string(1, "testing").build();
        // tag = 0x0a (field 1, wire type 2), length = 7, then "testing"
        assert_eq!(bytes[0], 0x0a);
        assert_eq!(bytes[1], 7);
        assert_eq!(&bytes[2..], b"testing");
    }

    #[test]
    fn test_proto_encoder_bool_true() {
        let bytes = ProtoEncoder::new().bool(1, true).build();
        assert_eq!(bytes, [0x08, 0x01]);
    }

    #[test]
    fn test_proto_encoder_bool_false() {
        let bytes = ProtoEncoder::new().bool(1, false).build();
        assert_eq!(bytes, [0x08, 0x00]);
    }

    #[test]
    fn test_proto_encoder_float() {
        let v = 1.0_f32;
        let bytes = ProtoEncoder::new().float(1, v).build();
        // tag field 1 wire type 5 = (1 << 3) | 5 = 0x0d
        assert_eq!(bytes[0], 0x0d);
        let decoded = f32::from_le_bytes(bytes[1..5].try_into().expect("slice"));
        assert!((decoded - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_proto_encoder_double() {
        let v = std::f64::consts::PI;
        let bytes = ProtoEncoder::new().double(1, v).build();
        // tag field 1 wire type 1 = (1 << 3) | 1 = 0x09
        assert_eq!(bytes[0], 0x09);
        let decoded = f64::from_le_bytes(bytes[1..9].try_into().expect("slice"));
        assert!((decoded - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn test_proto_encoder_bytes() {
        let data = b"\xde\xad\xbe\xef";
        let bytes = ProtoEncoder::new().bytes(1, data).build();
        assert_eq!(bytes[0], 0x0a); // field 1, wire type 2
        assert_eq!(bytes[1], 4); // length
        assert_eq!(&bytes[2..], data);
    }

    #[test]
    fn test_proto_encoder_message_nested() {
        let inner = ProtoEncoder::new().int32(1, 42).build();
        let outer = ProtoEncoder::new().message(1, &inner).build();
        // The outer message wraps the inner as a length-delimited field
        let mut dec = ProtoDecoder::new(&outer);
        let (fn_, wv) = dec.next_field().expect("field").expect("ok");
        assert_eq!(fn_, 1);
        if let WireValue::LengthDelimited(payload) = wv {
            assert_eq!(payload, inner);
        } else {
            panic!("expected LengthDelimited");
        }
    }

    // ── ProtoDecoder ─────────────────────────────────────────────────────────

    #[test]
    fn test_proto_decoder_varint_field() {
        let bytes = ProtoEncoder::new().int64(3, 9999).build();
        let mut dec = ProtoDecoder::new(&bytes);
        let (fn_, wv) = dec.next_field().expect("field").expect("ok");
        assert_eq!(fn_, 3);
        assert_eq!(wv, WireValue::Varint(9999));
        assert!(dec.is_empty());
    }

    #[test]
    fn test_proto_decoder_multiple_fields() {
        let bytes = ProtoEncoder::new()
            .int32(1, 1)
            .string(2, "abc")
            .bool(3, true)
            .build();

        let mut dec = ProtoDecoder::new(&bytes);
        let fields = dec.collect_all().expect("ok");
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].0, 1);
        assert_eq!(fields[1].0, 2);
        assert_eq!(fields[2].0, 3);
    }

    // ── Schema-guided encode/decode ───────────────────────────────────────────

    #[test]
    fn test_encode_decode_all_field_types() {
        use crate::schema_registry::types::FieldDescriptor;

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
            (1, FieldValue::Int32(-7)),
            (2, FieldValue::Int64(-9_999_999_999)),
            (3, FieldValue::UInt32(42)),
            (4, FieldValue::UInt64(u64::MAX)),
            (5, FieldValue::Float(3.25)),
            (6, FieldValue::Double(2.345_678_901)),
            (7, FieldValue::Bool(true)),
            (8, FieldValue::Str("hello".to_string())),
            (9, FieldValue::Bytes(vec![0xca, 0xfe])),
        ];

        let bytes = encode_message(&desc, &values);
        let decoded = decode_message(&desc, &bytes).expect("decode ok");
        assert_eq!(decoded.len(), 9);

        assert_eq!(decoded[0], ("i32".to_string(), FieldValue::Int32(-7)));
        assert_eq!(
            decoded[1],
            ("i64".to_string(), FieldValue::Int64(-9_999_999_999))
        );
        assert_eq!(decoded[2], ("u32".to_string(), FieldValue::UInt32(42)));
        assert_eq!(
            decoded[3],
            ("u64".to_string(), FieldValue::UInt64(u64::MAX))
        );
        assert_eq!(decoded[6], ("b".to_string(), FieldValue::Bool(true)));
        assert_eq!(
            decoded[7],
            ("s".to_string(), FieldValue::Str("hello".to_string()))
        );
        assert_eq!(
            decoded[8],
            ("raw".to_string(), FieldValue::Bytes(vec![0xca, 0xfe]))
        );
    }

    #[test]
    fn test_message_encode_decode_roundtrip() {
        use crate::schema_registry::types::FieldDescriptor;

        let desc = MessageDescriptor::new("Point", "geometry")
            .with_field(FieldDescriptor::optional(1, "x", FieldType::Double))
            .with_field(FieldDescriptor::optional(2, "y", FieldType::Double))
            .with_field(FieldDescriptor::optional(3, "label", FieldType::String));

        let values = vec![
            (1, FieldValue::Double(1.5)),
            (2, FieldValue::Double(-3.75)),
            (3, FieldValue::Str("origin".to_string())),
        ];

        let encoded = encode_message(&desc, &values);
        let decoded = decode_message(&desc, &encoded).expect("decode ok");

        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[2].1, FieldValue::Str("origin".to_string()));
    }

    #[test]
    fn test_nested_message_encoding() {
        use crate::schema_registry::types::FieldDescriptor;

        // Inner: { id: int32 }
        let inner_desc = MessageDescriptor::new("Inner", "test")
            .with_field(FieldDescriptor::optional(1, "id", FieldType::Int32));

        let inner_bytes = encode_message(&inner_desc, &[(1, FieldValue::Int32(99))]);

        // Outer: { nested: message(Inner) }
        let outer_desc = MessageDescriptor::new("Outer", "test").with_field(
            FieldDescriptor::optional(1, "nested", FieldType::Message("Inner".to_string())),
        );

        let outer_values = vec![(1, FieldValue::Message(inner_bytes.clone()))];
        let outer_bytes = encode_message(&outer_desc, &outer_values);
        let outer_decoded = decode_message(&outer_desc, &outer_bytes).expect("ok");

        assert_eq!(outer_decoded.len(), 1);
        if let FieldValue::Message(payload) = &outer_decoded[0].1 {
            assert_eq!(payload, &inner_bytes);
        } else {
            panic!("expected Message variant");
        }
    }

    #[test]
    fn test_repeated_field_encoding() {
        use crate::schema_registry::types::FieldDescriptor;

        // Repeated int32: packed encoding — client pre-packs as bytes
        let desc = MessageDescriptor::new("Bag", "test").with_field(FieldDescriptor::optional(
            1,
            "items",
            FieldType::Repeated(Box::new(FieldType::Int32)),
        ));

        // Pack [1, 2, 3] as varints
        let mut packed = Vec::new();
        for v in [1u64, 2, 3] {
            encode_varint(v, &mut packed);
        }

        let values = vec![(1, FieldValue::Bytes(packed.clone()))];
        let bytes = encode_message(&desc, &values);
        let decoded = decode_message(&desc, &bytes).expect("ok");

        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].0, "items");
        if let FieldValue::Bytes(b) = &decoded[0].1 {
            assert_eq!(b, &packed);
        } else {
            panic!("expected Bytes");
        }
    }

    #[test]
    fn test_unknown_field_skipped_on_decode() {
        use crate::schema_registry::types::FieldDescriptor;

        // Encode a message with two fields
        let desc_full = MessageDescriptor::new("M", "test")
            .with_field(FieldDescriptor::optional(1, "a", FieldType::Int32))
            .with_field(FieldDescriptor::optional(2, "b", FieldType::String));

        let bytes = encode_message(
            &desc_full,
            &[
                (1, FieldValue::Int32(7)),
                (2, FieldValue::Str("x".to_string())),
            ],
        );

        // Decode with a schema that only knows about field 1
        let desc_partial = MessageDescriptor::new("M", "test")
            .with_field(FieldDescriptor::optional(1, "a", FieldType::Int32));

        let decoded = decode_message(&desc_partial, &bytes).expect("ok");
        // Field 2 should be silently skipped
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].1, FieldValue::Int32(7));
    }
}
