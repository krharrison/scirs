//! Minimal ONNX model format I/O using hand-crafted protobuf binary parsing.
//!
//! Implements just enough of the protobuf binary wire format to:
//! - Parse an ONNX ModelProto from raw bytes
//! - Build a summary of nodes, opsets, inputs, and outputs
//! - Emit a minimal ONNX binary (no external protobuf crate)
//!
//! ONNX protobuf field numbers from the official spec:
//! <https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3>

use crate::error::IoError;

// ---- Wire types ----------------------------------------------------------------

/// A protobuf field value decoded from the binary stream.
#[derive(Debug, Clone)]
pub enum WireType {
    /// Varint (wire type 0): encoded as LEB-128 unsigned integer.
    Varint(u64),
    /// Length-delimited (wire type 2): arbitrary byte slice.
    LengthDelimited(Vec<u8>),
    /// 32-bit fixed (wire type 5).
    Fixed32(u32),
    /// 64-bit fixed (wire type 1).
    Fixed64(u64),
}

// ---- Low-level proto parsing ---------------------------------------------------

/// Decode an unsigned LEB-128 varint from `data` starting at `*pos`.
///
/// Advances `*pos` past the consumed bytes and returns the decoded value.
pub fn decode_varint(data: &[u8], pos: &mut usize) -> Result<u64, IoError> {
    let mut result: u64 = 0;
    let mut shift = 0u32;
    loop {
        if *pos >= data.len() {
            return Err(IoError::ParseError(
                "protobuf: unexpected end of data in varint".to_string(),
            ));
        }
        let byte = data[*pos];
        *pos += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(result);
        }
        shift += 7;
        if shift >= 64 {
            return Err(IoError::ParseError(
                "protobuf: varint overflow (>64 bits)".to_string(),
            ));
        }
    }
}

/// Decode the next protobuf field tag and value from `data` at `*pos`.
///
/// Returns `(field_number, WireType)`.
pub fn decode_field(data: &[u8], pos: &mut usize) -> Result<(u32, WireType), IoError> {
    let tag = decode_varint(data, pos)?;
    let field_num = (tag >> 3) as u32;
    let wire_type = tag & 0x7;

    let value = match wire_type {
        0 => {
            // Varint
            let v = decode_varint(data, pos)?;
            WireType::Varint(v)
        }
        1 => {
            // 64-bit fixed
            if *pos + 8 > data.len() {
                return Err(IoError::ParseError(
                    "protobuf: unexpected end of data in fixed64".to_string(),
                ));
            }
            let bytes: [u8; 8] = data[*pos..*pos + 8]
                .try_into()
                .map_err(|_| IoError::ParseError("protobuf: fixed64 slice error".to_string()))?;
            *pos += 8;
            WireType::Fixed64(u64::from_le_bytes(bytes))
        }
        2 => {
            // Length-delimited
            let len = decode_varint(data, pos)? as usize;
            if *pos + len > data.len() {
                return Err(IoError::ParseError(format!(
                    "protobuf: length-delimited field needs {} bytes but only {} remain",
                    len,
                    data.len() - *pos
                )));
            }
            let bytes = data[*pos..*pos + len].to_vec();
            *pos += len;
            WireType::LengthDelimited(bytes)
        }
        5 => {
            // 32-bit fixed
            if *pos + 4 > data.len() {
                return Err(IoError::ParseError(
                    "protobuf: unexpected end of data in fixed32".to_string(),
                ));
            }
            let bytes: [u8; 4] = data[*pos..*pos + 4]
                .try_into()
                .map_err(|_| IoError::ParseError("protobuf: fixed32 slice error".to_string()))?;
            *pos += 4;
            WireType::Fixed32(u32::from_le_bytes(bytes))
        }
        wt => {
            return Err(IoError::ParseError(format!(
                "protobuf: unsupported wire type {wt} for field {field_num}"
            )));
        }
    };
    Ok((field_num, value))
}

/// Decode all fields from a sub-message byte slice.
fn parse_message(data: &[u8]) -> Result<Vec<(u32, WireType)>, IoError> {
    let mut fields = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let (field_num, value) = decode_field(data, &mut pos)?;
        fields.push((field_num, value));
    }
    Ok(fields)
}

/// Interpret a WireType as a UTF-8 string (length-delimited).
fn wire_to_string(wt: &WireType) -> Result<String, IoError> {
    match wt {
        WireType::LengthDelimited(bytes) => String::from_utf8(bytes.clone())
            .map_err(|e| IoError::ParseError(format!("protobuf: invalid UTF-8 string: {e}"))),
        other => Err(IoError::ParseError(format!(
            "protobuf: expected length-delimited for string, got {:?}",
            std::mem::discriminant(other)
        ))),
    }
}

/// Interpret a WireType as a u64 (varint).
fn wire_to_u64(wt: &WireType) -> Result<u64, IoError> {
    match wt {
        WireType::Varint(v) => Ok(*v),
        WireType::Fixed64(v) => Ok(*v),
        other => Err(IoError::ParseError(format!(
            "protobuf: expected varint for integer, got {:?}",
            std::mem::discriminant(other)
        ))),
    }
}

// ---- ONNX data structures -----------------------------------------------------

/// Simplified ONNX TensorProto.
#[derive(Debug, Clone, Default)]
pub struct OnnxTensorProto {
    /// Tensor dimensions (field 1, repeated int64).
    pub dims: Vec<i64>,
    /// Data type (field 2, int32 enum).
    pub data_type: i32,
    /// Float data (field 4, repeated float).
    pub float_data: Vec<f32>,
    /// Int64 data (field 7, repeated int64).
    pub int64_data: Vec<i64>,
    /// Tensor name (field 9, string).
    pub name: String,
    /// Raw data bytes (field 12, bytes).
    pub raw_data: Vec<u8>,
}

impl OnnxTensorProto {
    fn parse(data: &[u8]) -> Result<Self, IoError> {
        let fields = parse_message(data)?;
        let mut tp = OnnxTensorProto::default();
        for (field_num, value) in fields {
            match field_num {
                1 => tp.dims.push(wire_to_u64(&value)? as i64),
                2 => tp.data_type = wire_to_u64(&value)? as i32,
                4 => match &value {
                    WireType::Fixed32(bits) => tp.float_data.push(f32::from_bits(*bits)),
                    WireType::LengthDelimited(bytes) => {
                        // packed encoding: sequence of float32 little-endian
                        for chunk in bytes.chunks(4) {
                            if chunk.len() == 4 {
                                let arr: [u8; 4] = chunk.try_into().map_err(|_| {
                                    IoError::ParseError("packed float chunk error".to_string())
                                })?;
                                tp.float_data.push(f32::from_le_bytes(arr));
                            }
                        }
                    }
                    _ => {}
                },
                7 => match &value {
                    WireType::Varint(v) => tp.int64_data.push(*v as i64),
                    WireType::LengthDelimited(bytes) => {
                        // packed encoding: sequence of varints
                        let mut pos = 0;
                        while pos < bytes.len() {
                            let v = decode_varint(bytes, &mut pos)?;
                            tp.int64_data.push(v as i64);
                        }
                    }
                    _ => {}
                },
                9 => tp.name = wire_to_string(&value)?,
                12 => {
                    if let WireType::LengthDelimited(bytes) = value {
                        tp.raw_data = bytes;
                    }
                }
                _ => {}
            }
        }
        Ok(tp)
    }
}

/// Simplified ONNX AttributeProto.
#[derive(Debug, Clone, Default)]
pub struct OnnxAttributeProto {
    /// Attribute name (field 1).
    pub name: String,
    /// Attribute type (field 20, int32).
    pub attribute_type: i32,
    /// Float value (field 4).
    pub f: f32,
    /// Integer value (field 3).
    pub i: i64,
    /// String value (field 5).
    pub s: Vec<u8>,
}

impl OnnxAttributeProto {
    fn parse(data: &[u8]) -> Result<Self, IoError> {
        let fields = parse_message(data)?;
        let mut attr = OnnxAttributeProto::default();
        for (field_num, value) in fields {
            match field_num {
                1 => attr.name = wire_to_string(&value)?,
                3 => attr.i = wire_to_u64(&value)? as i64,
                4 => {
                    if let WireType::Fixed32(bits) = value {
                        attr.f = f32::from_bits(bits);
                    }
                }
                5 => {
                    if let WireType::LengthDelimited(bytes) = value {
                        attr.s = bytes;
                    }
                }
                20 => attr.attribute_type = wire_to_u64(&value)? as i32,
                _ => {}
            }
        }
        Ok(attr)
    }
}

/// Simplified ONNX NodeProto.
#[derive(Debug, Clone, Default)]
pub struct OnnxNodeProto {
    /// Input tensor names (field 1, repeated string).
    pub input: Vec<String>,
    /// Output tensor names (field 2, repeated string).
    pub output: Vec<String>,
    /// Node name (field 3, string).
    pub name: String,
    /// Operator type (field 4, string).
    pub op_type: String,
    /// Node attributes (field 5, repeated AttributeProto).
    pub attribute: Vec<OnnxAttributeProto>,
    /// Domain (field 7, string).
    pub domain: String,
}

impl OnnxNodeProto {
    fn parse(data: &[u8]) -> Result<Self, IoError> {
        let fields = parse_message(data)?;
        let mut node = OnnxNodeProto::default();
        for (field_num, value) in fields {
            match field_num {
                1 => node.input.push(wire_to_string(&value)?),
                2 => node.output.push(wire_to_string(&value)?),
                3 => node.name = wire_to_string(&value)?,
                4 => node.op_type = wire_to_string(&value)?,
                5 => {
                    if let WireType::LengthDelimited(bytes) = value {
                        node.attribute.push(OnnxAttributeProto::parse(&bytes)?);
                    }
                }
                7 => node.domain = wire_to_string(&value)?,
                _ => {}
            }
        }
        Ok(node)
    }
}

/// Simplified ONNX ValueInfoProto (input/output spec).
#[derive(Debug, Clone, Default)]
pub struct OnnxValueInfoProto {
    /// Name of the value (field 1, string).
    pub name: String,
}

impl OnnxValueInfoProto {
    fn parse(data: &[u8]) -> Result<Self, IoError> {
        let fields = parse_message(data)?;
        let mut vi = OnnxValueInfoProto::default();
        for (field_num, value) in &fields {
            if *field_num == 1 {
                vi.name = wire_to_string(value)?;
            }
        }
        Ok(vi)
    }
}

/// Simplified ONNX OperatorSetIdProto.
#[derive(Debug, Clone, Default)]
pub struct OnnxOperatorSetIdProto {
    /// Domain of the operator set (field 1, string; "" = default ONNX).
    pub domain: String,
    /// Opset version (field 2, int64).
    pub version: u64,
}

impl OnnxOperatorSetIdProto {
    fn parse(data: &[u8]) -> Result<Self, IoError> {
        let fields = parse_message(data)?;
        let mut op = OnnxOperatorSetIdProto::default();
        for (field_num, value) in fields {
            match field_num {
                1 => op.domain = wire_to_string(&value)?,
                2 => op.version = wire_to_u64(&value)?,
                _ => {}
            }
        }
        Ok(op)
    }
}

/// Simplified ONNX GraphProto.
#[derive(Debug, Clone, Default)]
pub struct OnnxGraphProto {
    /// Computation nodes (field 1, repeated NodeProto).
    pub node: Vec<OnnxNodeProto>,
    /// Graph name (field 2, string).
    pub name: String,
    /// Initializers (weights, field 5, repeated TensorProto).
    pub initializer: Vec<OnnxTensorProto>,
    /// Graph inputs (field 11, repeated ValueInfoProto).
    pub input: Vec<OnnxValueInfoProto>,
    /// Graph outputs (field 12, repeated ValueInfoProto).
    pub output: Vec<OnnxValueInfoProto>,
}

impl OnnxGraphProto {
    fn parse(data: &[u8]) -> Result<Self, IoError> {
        let fields = parse_message(data)?;
        let mut graph = OnnxGraphProto::default();
        for (field_num, value) in fields {
            match field_num {
                1 => {
                    if let WireType::LengthDelimited(bytes) = value {
                        graph.node.push(OnnxNodeProto::parse(&bytes)?);
                    }
                }
                2 => graph.name = wire_to_string(&value)?,
                5 => {
                    if let WireType::LengthDelimited(bytes) = value {
                        graph.initializer.push(OnnxTensorProto::parse(&bytes)?);
                    }
                }
                11 => {
                    if let WireType::LengthDelimited(bytes) = value {
                        graph.input.push(OnnxValueInfoProto::parse(&bytes)?);
                    }
                }
                12 => {
                    if let WireType::LengthDelimited(bytes) = value {
                        graph.output.push(OnnxValueInfoProto::parse(&bytes)?);
                    }
                }
                _ => {}
            }
        }
        Ok(graph)
    }
}

/// Parsed ONNX ModelProto.
#[derive(Debug, Clone, Default)]
pub struct OnnxModelProto {
    /// IR version (field 1, int64).
    pub ir_version: u64,
    /// Opset imports (field 8, repeated OperatorSetIdProto).
    pub opset_import: Vec<OnnxOperatorSetIdProto>,
    /// Domain (field 2, string).
    pub domain: String,
    /// Model version (field 5, int64).
    pub model_version: u64,
    /// Documentation string (field 6, string).
    pub doc_string: String,
    /// The computation graph (field 7, GraphProto).
    pub graph: OnnxGraphProto,
}

impl OnnxModelProto {
    /// Parse an ONNX model from raw bytes.
    pub fn parse(data: &[u8]) -> Result<OnnxModelProto, IoError> {
        let fields = parse_message(data)?;
        let mut model = OnnxModelProto::default();
        for (field_num, value) in fields {
            match field_num {
                1 => model.ir_version = wire_to_u64(&value)?,
                2 => model.domain = wire_to_string(&value)?,
                5 => model.model_version = wire_to_u64(&value)?,
                6 => model.doc_string = wire_to_string(&value)?,
                7 => {
                    if let WireType::LengthDelimited(bytes) = value {
                        model.graph = OnnxGraphProto::parse(&bytes)?;
                    }
                }
                8 => {
                    if let WireType::LengthDelimited(bytes) = value {
                        model
                            .opset_import
                            .push(OnnxOperatorSetIdProto::parse(&bytes)?);
                    }
                }
                _ => {}
            }
        }
        Ok(model)
    }

    /// Produce a compact summary of the model.
    pub fn to_summary(&self) -> OnnxModelSummary {
        let n_nodes = self.graph.node.len();
        let n_initializers = self.graph.initializer.len();
        let opset_version = self
            .opset_import
            .iter()
            .filter(|op| op.domain.is_empty())
            .map(|op| op.version)
            .next()
            .unwrap_or(0);
        let mut op_types: Vec<String> = self.graph.node.iter().map(|n| n.op_type.clone()).collect();
        op_types.sort();
        op_types.dedup();
        let input_names: Vec<String> = self.graph.input.iter().map(|v| v.name.clone()).collect();
        let output_names: Vec<String> = self.graph.output.iter().map(|v| v.name.clone()).collect();
        OnnxModelSummary {
            n_nodes,
            n_initializers,
            opset_version,
            op_types,
            input_names,
            output_names,
        }
    }
}

/// A compact summary of an ONNX model.
#[derive(Debug, Clone)]
pub struct OnnxModelSummary {
    /// Number of computation nodes.
    pub n_nodes: usize,
    /// Number of weight initializers.
    pub n_initializers: usize,
    /// Default opset version (domain "").
    pub opset_version: u64,
    /// Sorted, deduplicated list of operator types used.
    pub op_types: Vec<String>,
    /// Graph input names.
    pub input_names: Vec<String>,
    /// Graph output names.
    pub output_names: Vec<String>,
}

// ---- Writing -------------------------------------------------------------------

/// Encode a u64 as an unsigned LEB-128 varint.
pub fn encode_varint(mut value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            break;
        } else {
            out.push(byte | 0x80);
        }
    }
    out
}

/// Encode a protobuf field tag: `(field_num << 3) | wire_type`.
pub fn write_field_tag(field_num: u32, wire_type: u8) -> Vec<u8> {
    encode_varint(((field_num as u64) << 3) | (wire_type as u64))
}

/// Write a length-delimited field (wire type 2).
fn write_length_delimited(field_num: u32, data: &[u8]) -> Vec<u8> {
    let mut out = write_field_tag(field_num, 2);
    out.extend(encode_varint(data.len() as u64));
    out.extend_from_slice(data);
    out
}

/// Write a string field (wire type 2).
fn write_string_field(field_num: u32, s: &str) -> Vec<u8> {
    write_length_delimited(field_num, s.as_bytes())
}

/// Write a varint field (wire type 0).
fn write_varint_field(field_num: u32, value: u64) -> Vec<u8> {
    let mut out = write_field_tag(field_num, 0);
    out.extend(encode_varint(value));
    out
}

/// Encode an `OnnxNodeProto` to bytes.
fn encode_node(node: &OnnxNodeProto) -> Vec<u8> {
    let mut out = Vec::new();
    for inp in &node.input {
        out.extend(write_string_field(1, inp));
    }
    for outp in &node.output {
        out.extend(write_string_field(2, outp));
    }
    if !node.name.is_empty() {
        out.extend(write_string_field(3, &node.name));
    }
    out.extend(write_string_field(4, &node.op_type));
    out
}

/// Encode a `ValueInfoProto` (just the name) to bytes.
fn encode_value_info(name: &str) -> Vec<u8> {
    write_string_field(1, name)
}

/// Create a minimal ONNX binary suitable for testing or stub models.
///
/// Encodes:
/// - ir_version = 7 (ONNX IR version 7)
/// - opset_import: domain="" version=17
/// - graph with provided nodes, input names, and output names
pub fn create_minimal_onnx(
    nodes: &[OnnxNodeProto],
    inputs: &[String],
    outputs: &[String],
) -> Vec<u8> {
    // Build GraphProto bytes
    let mut graph_bytes = Vec::new();
    for node in nodes {
        let nb = encode_node(node);
        graph_bytes.extend(write_length_delimited(1, &nb));
    }
    graph_bytes.extend(write_string_field(2, "main_graph"));
    for inp in inputs {
        let vi = encode_value_info(inp);
        graph_bytes.extend(write_length_delimited(11, &vi));
    }
    for out in outputs {
        let vi = encode_value_info(out);
        graph_bytes.extend(write_length_delimited(12, &vi));
    }

    // Build opset_import bytes (field 8)
    let mut opset_bytes = Vec::new();
    opset_bytes.extend(write_string_field(1, "")); // domain=""
    opset_bytes.extend(write_varint_field(2, 17)); // version=17

    // Build ModelProto bytes
    let mut model_bytes = Vec::new();
    model_bytes.extend(write_varint_field(1, 7)); // ir_version = 7
    model_bytes.extend(write_length_delimited(7, &graph_bytes)); // graph
    model_bytes.extend(write_length_delimited(8, &opset_bytes)); // opset_import

    model_bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal 2-field proto and verify field extraction.
    #[test]
    fn test_decode_varint() {
        let data = [0x96, 0x01]; // 150 in LEB-128
        let mut pos = 0;
        let v = decode_varint(&data, &mut pos).expect("decode varint 150");
        assert_eq!(v, 150);
        assert_eq!(pos, 2);
    }

    #[test]
    fn test_decode_varint_single_byte() {
        let data = [0x05];
        let mut pos = 0;
        let v = decode_varint(&data, &mut pos).expect("single byte varint");
        assert_eq!(v, 5);
    }

    #[test]
    fn test_encode_decode_varint_roundtrip() {
        for val in [
            0u64,
            1,
            127,
            128,
            255,
            300,
            16383,
            16384,
            1_000_000,
            u32::MAX as u64,
        ] {
            let encoded = encode_varint(val);
            let mut pos = 0;
            let decoded = decode_varint(&encoded, &mut pos).expect("roundtrip");
            assert_eq!(decoded, val, "varint roundtrip failed for {val}");
        }
    }

    #[test]
    fn test_create_and_parse_minimal_onnx() {
        let node = OnnxNodeProto {
            input: vec!["x".to_string()],
            output: vec!["y".to_string()],
            op_type: "Relu".to_string(),
            name: "relu0".to_string(),
            ..Default::default()
        };
        let bytes = create_minimal_onnx(&[node], &["x".to_string()], &["y".to_string()]);

        let model = OnnxModelProto::parse(&bytes).expect("parse minimal onnx");
        assert_eq!(model.ir_version, 7);
        assert_eq!(model.graph.node.len(), 1);
        assert_eq!(model.graph.node[0].op_type, "Relu");
        assert_eq!(model.graph.input[0].name, "x");
        assert_eq!(model.graph.output[0].name, "y");
    }

    #[test]
    fn test_opset_import_parsing() {
        let bytes = create_minimal_onnx(&[], &[], &[]);
        let model = OnnxModelProto::parse(&bytes).expect("parse opset test");
        assert!(!model.opset_import.is_empty());
        let default_opset = model
            .opset_import
            .iter()
            .find(|op| op.domain.is_empty())
            .expect("default opset");
        assert_eq!(default_opset.version, 17);
    }

    #[test]
    fn test_model_summary() {
        let nodes = vec![
            OnnxNodeProto {
                input: vec!["x".to_string()],
                output: vec!["h".to_string()],
                op_type: "Gemm".to_string(),
                name: "gemm0".to_string(),
                ..Default::default()
            },
            OnnxNodeProto {
                input: vec!["h".to_string()],
                output: vec!["y".to_string()],
                op_type: "Relu".to_string(),
                name: "relu1".to_string(),
                ..Default::default()
            },
        ];
        let bytes = create_minimal_onnx(&nodes, &["x".to_string()], &["y".to_string()]);
        let model = OnnxModelProto::parse(&bytes).expect("parse");
        let summary = model.to_summary();
        assert_eq!(summary.n_nodes, 2);
        assert_eq!(summary.opset_version, 17);
        assert!(summary.op_types.contains(&"Gemm".to_string()));
        assert!(summary.op_types.contains(&"Relu".to_string()));
        assert_eq!(summary.input_names, vec!["x"]);
        assert_eq!(summary.output_names, vec!["y"]);
    }

    #[test]
    fn test_decode_field_length_delimited() {
        // Manually craft: field 2, wire type 2, length 3, bytes [0x61, 0x62, 0x63]
        // tag = (2 << 3) | 2 = 0x12; length = 3 = 0x03
        let data = [0x12u8, 0x03, 0x61, 0x62, 0x63];
        let mut pos = 0;
        let (field_num, wt) = decode_field(&data, &mut pos).expect("decode ld field");
        assert_eq!(field_num, 2);
        match wt {
            WireType::LengthDelimited(bytes) => assert_eq!(bytes, vec![0x61, 0x62, 0x63]),
            _ => panic!("expected LengthDelimited"),
        }
        assert_eq!(pos, 5);
    }

    #[test]
    fn test_write_field_tag() {
        let tag = write_field_tag(1, 0);
        assert_eq!(tag, vec![0x08]); // (1 << 3) | 0 = 8
        let tag2 = write_field_tag(2, 2);
        assert_eq!(tag2, vec![0x12]); // (2 << 3) | 2 = 18
    }

    #[test]
    fn test_empty_model() {
        let bytes = create_minimal_onnx(&[], &[], &[]);
        let model = OnnxModelProto::parse(&bytes).expect("parse empty model");
        let summary = model.to_summary();
        assert_eq!(summary.n_nodes, 0);
        assert_eq!(summary.n_initializers, 0);
        assert!(summary.input_names.is_empty());
        assert!(summary.output_names.is_empty());
    }
}
