//! Tensor serialization formats: SafeTensors, ONNX proto I/O, TFRecord
//!
//! Provides reading and writing support for machine learning tensor formats:
//! - **SafeTensors**: HuggingFace simple tensor serialization format
//! - **ONNX proto**: ONNX model binary parsing and writing (no protobuf dependency)
//! - **TFRecord**: TensorFlow data pipeline record format

pub mod onnx_proto;
pub mod safetensors;
pub mod tfrecord;

pub use onnx_proto::{
    create_minimal_onnx, decode_field, decode_varint, encode_varint, write_field_tag,
    OnnxGraphProto, OnnxModelProto, OnnxModelSummary, OnnxNodeProto, OnnxTensorProto,
    OnnxValueInfoProto, WireType,
};
pub use safetensors::{DType, SafeTensors, SafeTensorsHeader, TensorMeta};
pub use tfrecord::{
    crc32c, masked_crc32c, parse_example, read_all_records, Example, Feature, TfRecord,
    TfRecordReader,
};
