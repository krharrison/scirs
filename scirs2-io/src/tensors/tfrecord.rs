//! TFRecord format reader for TensorFlow data pipeline files.
//!
//! TFRecord binary format per record:
//! ```text
//! [uint64 LE data_length]
//! [uint32 LE masked_crc32c(length_bytes)]
//! [data bytes]
//! [uint32 LE masked_crc32c(data_bytes)]
//! ```
//!
//! CRC32C masking: `((crc >> 15) | (crc << 17)).wrapping_add(0xa282ead8u32)`
//!
//! Reference: <https://www.tensorflow.org/tutorials/load_data/tfrecord>

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom};

use crate::error::IoError;

// ---- CRC32C (Castagnoli polynomial 0x1EDC6F41) --------------------------------

const CRC32C_POLY: u32 = 0x1EDC6F41;

/// Build the CRC32C lookup table (Castagnoli polynomial).
fn make_crc32c_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    for i in 0..256u32 {
        let mut crc = i;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ CRC32C_POLY.reverse_bits();
            } else {
                crc >>= 1;
            }
        }
        table[i as usize] = crc;
    }
    table
}

/// Compute CRC32C (Castagnoli) checksum for the given data.
pub fn crc32c(data: &[u8]) -> u32 {
    let table = make_crc32c_table();
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        let idx = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ table[idx];
    }
    crc ^ 0xFFFF_FFFF
}

/// Compute masked CRC32C as used in TFRecord format.
///
/// `masked = ((crc >> 15) | (crc << 17)).wrapping_add(0xa282ead8)`
pub fn masked_crc32c(data: &[u8]) -> u32 {
    let crc = crc32c(data);
    let rotated = crc.rotate_right(15);
    rotated.wrapping_add(0xa282ead8u32)
}

// ---- TFRecord data structures -------------------------------------------------

/// A single TFRecord: the raw bytes from one record.
#[derive(Debug, Clone)]
pub struct TfRecord {
    /// Raw record data bytes.
    pub data: Vec<u8>,
}

/// Sequential TFRecord file reader.
pub struct TfRecordReader {
    file: fs::File,
    /// Path kept for error messages.
    path: String,
}

impl TfRecordReader {
    /// Open a TFRecord file for sequential reading.
    pub fn open(path: &str) -> Result<Self, IoError> {
        let file = fs::File::open(path).map_err(|e| {
            IoError::FileNotFound(format!("TfRecordReader: cannot open '{path}': {e}"))
        })?;
        Ok(TfRecordReader {
            file,
            path: path.to_string(),
        })
    }

    /// Read the next record from the file.
    ///
    /// Returns `Ok(None)` at end-of-file.
    pub fn next_record(&mut self) -> Result<Option<TfRecord>, IoError> {
        // Read 8-byte data length
        let mut len_buf = [0u8; 8];
        match self.file.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => {
                return Err(IoError::Io(e));
            }
        }

        // Read and verify masked CRC of length
        let mut len_crc_buf = [0u8; 4];
        self.file.read_exact(&mut len_crc_buf).map_err(|e| {
            IoError::FileError(format!(
                "TfRecord '{path}': read len_crc: {e}",
                path = self.path
            ))
        })?;
        let len_crc_expected = u32::from_le_bytes(len_crc_buf);
        let len_crc_actual = masked_crc32c(&len_buf);
        if len_crc_actual != len_crc_expected {
            return Err(IoError::ChecksumError(format!(
                "TfRecord '{}': length CRC mismatch: expected 0x{:08x}, got 0x{:08x}",
                self.path, len_crc_expected, len_crc_actual
            )));
        }

        let data_len = u64::from_le_bytes(len_buf) as usize;

        // Read data bytes
        let mut data = vec![0u8; data_len];
        self.file
            .read_exact(&mut data)
            .map_err(|e| IoError::FileError(format!("TfRecord '{}': read data: {e}", self.path)))?;

        // Read and verify masked CRC of data
        let mut data_crc_buf = [0u8; 4];
        self.file.read_exact(&mut data_crc_buf).map_err(|e| {
            IoError::FileError(format!("TfRecord '{}': read data_crc: {e}", self.path))
        })?;
        let data_crc_expected = u32::from_le_bytes(data_crc_buf);
        let data_crc_actual = masked_crc32c(&data);
        if data_crc_actual != data_crc_expected {
            return Err(IoError::ChecksumError(format!(
                "TfRecord '{}': data CRC mismatch: expected 0x{:08x}, got 0x{:08x}",
                self.path, data_crc_expected, data_crc_actual
            )));
        }

        Ok(Some(TfRecord { data }))
    }

    /// Rewind to the beginning of the file.
    pub fn rewind(&mut self) -> Result<(), IoError> {
        self.file
            .seek(SeekFrom::Start(0))
            .map(|_| ())
            .map_err(|e| IoError::FileError(format!("TfRecordReader rewind: {e}")))
    }
}

/// Read all records from a TFRecord file into memory.
pub fn read_all_records(path: &str) -> Result<Vec<TfRecord>, IoError> {
    let mut reader = TfRecordReader::open(path)?;
    let mut records = Vec::new();
    while let Some(rec) = reader.next_record()? {
        records.push(rec);
    }
    Ok(records)
}

/// Encode a single record as TFRecord bytes (for writing).
pub fn encode_tfrecord(data: &[u8]) -> Vec<u8> {
    let length = data.len() as u64;
    let len_bytes = length.to_le_bytes();
    let len_crc = masked_crc32c(&len_bytes);
    let data_crc = masked_crc32c(data);

    let mut out = Vec::with_capacity(8 + 4 + data.len() + 4);
    out.extend_from_slice(&len_bytes);
    out.extend_from_slice(&len_crc.to_le_bytes());
    out.extend_from_slice(data);
    out.extend_from_slice(&data_crc.to_le_bytes());
    out
}

// ---- tf.Example protobuf parsing ----------------------------------------------

use crate::tensors::onnx_proto::decode_varint;

/// A tf.Example feature value.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Feature {
    /// List of byte strings.
    BytesList(Vec<Vec<u8>>),
    /// List of 32-bit floats.
    FloatList(Vec<f32>),
    /// List of 64-bit signed integers.
    Int64List(Vec<i64>),
}

/// A parsed tf.Example protobuf.
#[derive(Debug, Clone)]
pub struct Example {
    /// Feature map: name → Feature value.
    pub features: HashMap<String, Feature>,
}

/// Parse a `BytesList` proto (field 1 = bytes, repeated).
fn parse_bytes_list(data: &[u8]) -> Result<Vec<Vec<u8>>, IoError> {
    let mut pos = 0;
    let mut items = Vec::new();
    while pos < data.len() {
        let tag = decode_varint(data, &mut pos)?;
        let field_num = (tag >> 3) as u32;
        let wire_type = tag & 0x7;
        if field_num == 1 && wire_type == 2 {
            let len = decode_varint(data, &mut pos)? as usize;
            if pos + len > data.len() {
                return Err(IoError::ParseError(
                    "BytesList: length-delimited overrun".to_string(),
                ));
            }
            items.push(data[pos..pos + len].to_vec());
            pos += len;
        } else {
            // Skip unknown field
            skip_field(data, &mut pos, wire_type)?;
        }
    }
    Ok(items)
}

/// Parse a `FloatList` proto (field 1 = float, repeated).
fn parse_float_list(data: &[u8]) -> Result<Vec<f32>, IoError> {
    let mut pos = 0;
    let mut items = Vec::new();
    while pos < data.len() {
        let tag = decode_varint(data, &mut pos)?;
        let field_num = (tag >> 3) as u32;
        let wire_type = tag & 0x7;
        match (field_num, wire_type) {
            (1, 5) => {
                // Fixed32
                if pos + 4 > data.len() {
                    return Err(IoError::ParseError(
                        "FloatList: fixed32 overrun".to_string(),
                    ));
                }
                let arr: [u8; 4] = data[pos..pos + 4].try_into().map_err(|_| {
                    IoError::ParseError("FloatList: fixed32 slice error".to_string())
                })?;
                items.push(f32::from_le_bytes(arr));
                pos += 4;
            }
            (1, 2) => {
                // Packed floats
                let len = decode_varint(data, &mut pos)? as usize;
                if pos + len > data.len() {
                    return Err(IoError::ParseError(
                        "FloatList: packed float overrun".to_string(),
                    ));
                }
                for chunk in data[pos..pos + len].chunks(4) {
                    if chunk.len() == 4 {
                        let arr: [u8; 4] = chunk.try_into().map_err(|_| {
                            IoError::ParseError("FloatList: packed chunk error".to_string())
                        })?;
                        items.push(f32::from_le_bytes(arr));
                    }
                }
                pos += len;
            }
            _ => skip_field(data, &mut pos, wire_type)?,
        }
    }
    Ok(items)
}

/// Parse an `Int64List` proto (field 1 = int64, repeated).
fn parse_int64_list(data: &[u8]) -> Result<Vec<i64>, IoError> {
    let mut pos = 0;
    let mut items = Vec::new();
    while pos < data.len() {
        let tag = decode_varint(data, &mut pos)?;
        let field_num = (tag >> 3) as u32;
        let wire_type = tag & 0x7;
        match (field_num, wire_type) {
            (1, 0) => {
                let v = decode_varint(data, &mut pos)?;
                items.push(v as i64);
            }
            (1, 2) => {
                // Packed varints
                let len = decode_varint(data, &mut pos)? as usize;
                if pos + len > data.len() {
                    return Err(IoError::ParseError(
                        "Int64List: packed varint overrun".to_string(),
                    ));
                }
                let mut inner_pos = 0;
                let inner = &data[pos..pos + len];
                while inner_pos < inner.len() {
                    let v = decode_varint(inner, &mut inner_pos)?;
                    items.push(v as i64);
                }
                pos += len;
            }
            _ => skip_field(data, &mut pos, wire_type)?,
        }
    }
    Ok(items)
}

/// Skip a field with the given wire type at the current position.
fn skip_field(data: &[u8], pos: &mut usize, wire_type: u64) -> Result<(), IoError> {
    match wire_type {
        0 => {
            decode_varint(data, pos)?;
        }
        1 => {
            if *pos + 8 > data.len() {
                return Err(IoError::ParseError(
                    "skip_field: fixed64 overrun".to_string(),
                ));
            }
            *pos += 8;
        }
        2 => {
            let len = decode_varint(data, pos)? as usize;
            if *pos + len > data.len() {
                return Err(IoError::ParseError(format!(
                    "skip_field: LD overrun: need {len} but {} remain",
                    data.len() - *pos
                )));
            }
            *pos += len;
        }
        5 => {
            if *pos + 4 > data.len() {
                return Err(IoError::ParseError(
                    "skip_field: fixed32 overrun".to_string(),
                ));
            }
            *pos += 4;
        }
        wt => {
            return Err(IoError::ParseError(format!(
                "skip_field: unknown wire type {wt}"
            )));
        }
    }
    Ok(())
}

/// Parse a single `Feature` proto from bytes.
fn parse_feature(data: &[u8]) -> Result<Feature, IoError> {
    // Feature has oneof: bytes_list (1), float_list (2), int64_list (3)
    let mut pos = 0;
    while pos < data.len() {
        let tag = decode_varint(data, &mut pos)?;
        let field_num = (tag >> 3) as u32;
        let wire_type = tag & 0x7;
        if wire_type == 2 {
            let len = decode_varint(data, &mut pos)? as usize;
            if pos + len > data.len() {
                return Err(IoError::ParseError(
                    "Feature: sub-message overrun".to_string(),
                ));
            }
            let sub = &data[pos..pos + len];
            pos += len;
            match field_num {
                1 => return Ok(Feature::BytesList(parse_bytes_list(sub)?)),
                2 => return Ok(Feature::FloatList(parse_float_list(sub)?)),
                3 => return Ok(Feature::Int64List(parse_int64_list(sub)?)),
                _ => {}
            }
        } else {
            skip_field(data, &mut pos, wire_type)?;
        }
    }
    // Default: empty bytes list if nothing found
    Ok(Feature::BytesList(Vec::new()))
}

/// Parse a `Features` proto: map<string, Feature> encoded as
/// repeated `Feature { key, value }` under field 1.
fn parse_features(data: &[u8]) -> Result<HashMap<String, Feature>, IoError> {
    let mut map = HashMap::new();
    let mut pos = 0;
    while pos < data.len() {
        let tag = decode_varint(data, &mut pos)?;
        let field_num = (tag >> 3) as u32;
        let wire_type = tag & 0x7;
        if field_num == 1 && wire_type == 2 {
            // MapEntry sub-message: key(1)=string, value(2)=Feature
            let len = decode_varint(data, &mut pos)? as usize;
            if pos + len > data.len() {
                return Err(IoError::ParseError(
                    "Features: map entry overrun".to_string(),
                ));
            }
            let entry_data = &data[pos..pos + len];
            pos += len;
            let mut epos = 0;
            let mut key = String::new();
            let mut feature_bytes: Option<Vec<u8>> = None;
            while epos < entry_data.len() {
                let etag = decode_varint(entry_data, &mut epos)?;
                let efn = (etag >> 3) as u32;
                let ewt = etag & 0x7;
                if ewt == 2 {
                    let elen = decode_varint(entry_data, &mut epos)? as usize;
                    if epos + elen > entry_data.len() {
                        return Err(IoError::ParseError(
                            "Features: map entry field overrun".to_string(),
                        ));
                    }
                    let field_bytes = &entry_data[epos..epos + elen];
                    epos += elen;
                    match efn {
                        1 => {
                            key = String::from_utf8(field_bytes.to_vec()).map_err(|e| {
                                IoError::ParseError(format!("Features: key utf8: {e}"))
                            })?;
                        }
                        2 => {
                            feature_bytes = Some(field_bytes.to_vec());
                        }
                        _ => {}
                    }
                } else {
                    skip_field(entry_data, &mut epos, ewt)?;
                }
            }
            if let Some(fb) = feature_bytes {
                map.insert(key, parse_feature(&fb)?);
            }
        } else {
            skip_field(data, &mut pos, wire_type)?;
        }
    }
    Ok(map)
}

/// Parse a `tf.Example` proto from raw bytes.
///
/// Example proto structure:
/// - field 1: `Features` (length-delimited)
///   - field 1: `map<string, Feature>` (repeated map entry)
pub fn parse_example(data: &[u8]) -> Result<Example, IoError> {
    let mut pos = 0;
    let mut features = HashMap::new();
    while pos < data.len() {
        let tag = decode_varint(data, &mut pos)?;
        let field_num = (tag >> 3) as u32;
        let wire_type = tag & 0x7;
        if field_num == 1 && wire_type == 2 {
            let len = decode_varint(data, &mut pos)? as usize;
            if pos + len > data.len() {
                return Err(IoError::ParseError(
                    "Example: features sub-message overrun".to_string(),
                ));
            }
            features = parse_features(&data[pos..pos + len])?;
            pos += len;
        } else {
            skip_field(data, &mut pos, wire_type)?;
        }
    }
    Ok(Example { features })
}

// ---- Helper: build a tf.Example proto from scratch (for testing) --------------

/// Encode a string as a protobuf string/bytes field (wire type 2).
fn proto_string_field(field_num: u32, s: &str) -> Vec<u8> {
    use crate::tensors::onnx_proto::{encode_varint, write_field_tag};
    let mut out = write_field_tag(field_num, 2);
    out.extend(encode_varint(s.len() as u64));
    out.extend_from_slice(s.as_bytes());
    out
}

/// Encode a bytes field (wire type 2).
fn proto_bytes_field(field_num: u32, data: &[u8]) -> Vec<u8> {
    use crate::tensors::onnx_proto::{encode_varint, write_field_tag};
    let mut out = write_field_tag(field_num, 2);
    out.extend(encode_varint(data.len() as u64));
    out.extend_from_slice(data);
    out
}

/// Build a `tf.Example` proto bytes containing a single `Int64List` feature.
///
/// Useful for writing test TFRecord files without TensorFlow.
pub fn build_example_int64(key: &str, values: &[i64]) -> Vec<u8> {
    use crate::tensors::onnx_proto::{encode_varint, write_field_tag};

    // Encode Int64List: field 1 (value, repeated varint)
    let mut int64_list = Vec::new();
    for &v in values {
        let mut tag = write_field_tag(1, 0);
        tag.extend(encode_varint(v as u64));
        int64_list.extend(tag);
    }

    // Feature: oneof field 3 = int64_list
    let feature_bytes = proto_bytes_field(3, &int64_list);

    // MapEntry: key (field 1 = string), value (field 2 = Feature)
    let mut map_entry = proto_string_field(1, key);
    map_entry.extend(proto_bytes_field(2, &feature_bytes));

    // Features: field 1 = repeated map entry
    let features_bytes = proto_bytes_field(1, &map_entry);

    // Example: field 1 = Features
    proto_bytes_field(1, &features_bytes)
}

/// Build a `tf.Example` proto bytes containing a single `FloatList` feature.
pub fn build_example_floats(key: &str, values: &[f32]) -> Vec<u8> {
    use crate::tensors::onnx_proto::write_field_tag;

    // Encode FloatList: field 1 (value, repeated fixed32)
    let mut float_list = Vec::new();
    for &v in values {
        let mut tag = write_field_tag(1, 5);
        tag.extend_from_slice(&v.to_le_bytes());
        float_list.extend(tag);
    }

    // Feature: oneof field 2 = float_list
    let feature_bytes = proto_bytes_field(2, &float_list);

    // MapEntry
    let mut map_entry = proto_string_field(1, key);
    map_entry.extend(proto_bytes_field(2, &feature_bytes));

    // Features
    let features_bytes = proto_bytes_field(1, &map_entry);

    // Example
    proto_bytes_field(1, &features_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::io::Write;

    #[test]
    fn test_crc32c_known_value() {
        // CRC32C of empty string = 0x00000000
        assert_eq!(crc32c(b""), 0x00000000);
        // Known test vector: CRC32C of "123456789" = 0xE3069283
        assert_eq!(crc32c(b"123456789"), 0xE3069283);
    }

    #[test]
    fn test_masked_crc32c() {
        // Verify masking formula
        let data = b"hello world";
        let raw = crc32c(data);
        let masked = masked_crc32c(data);
        let expected = raw.rotate_right(15).wrapping_add(0xa282ead8u32);
        assert_eq!(masked, expected);
    }

    #[test]
    fn test_encode_tfrecord_and_read_back() {
        let original = b"hello tfrecord";
        let encoded = encode_tfrecord(original);

        // Verify structure: 8 (len) + 4 (len_crc) + data_len + 4 (data_crc)
        assert_eq!(encoded.len(), 8 + 4 + original.len() + 4);

        // Verify length field
        let len_val = u64::from_le_bytes(encoded[0..8].try_into().unwrap());
        assert_eq!(len_val, original.len() as u64);

        // Verify data
        assert_eq!(&encoded[12..12 + original.len()], original);
    }

    #[test]
    fn test_write_and_read_tfrecord_file() {
        let tmp_dir = env::temp_dir();
        let path = tmp_dir
            .join("test_ws146.tfrecord")
            .to_string_lossy()
            .to_string();

        // Write two records
        let rec1 = b"record one data";
        let rec2 = b"record two data longer content";
        let mut file = fs::File::create(&path).expect("create tfrecord");
        file.write_all(&encode_tfrecord(rec1)).expect("write rec1");
        file.write_all(&encode_tfrecord(rec2)).expect("write rec2");
        drop(file);

        let records = read_all_records(&path).expect("read_all_records");
        assert_eq!(records.len(), 2, "expected 2 records");
        assert_eq!(records[0].data, rec1);
        assert_eq!(records[1].data, rec2);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_crc_mismatch_detected() {
        let original = b"crc test";
        let mut encoded = encode_tfrecord(original);
        // Corrupt data CRC
        let last = encoded.len() - 1;
        encoded[last] ^= 0xFF;

        let tmp_dir = env::temp_dir();
        let path = tmp_dir
            .join("test_crc_corrupt_ws146.tfrecord")
            .to_string_lossy()
            .to_string();
        let mut file = fs::File::create(&path).expect("create corrupt file");
        file.write_all(&encoded).expect("write corrupt");
        drop(file);

        let result = read_all_records(&path);
        assert!(result.is_err(), "expected CRC error");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_parse_example_int64() {
        let proto_bytes = build_example_int64("label", &[3, 7, 42]);
        let example = parse_example(&proto_bytes).expect("parse example int64");
        let feature = example.features.get("label").expect("label feature");
        match feature {
            Feature::Int64List(vals) => {
                assert_eq!(vals, &[3, 7, 42]);
            }
            other => panic!("expected Int64List, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_example_floats() {
        let proto_bytes = build_example_floats("scores", &[1.0, 2.5, 3.25]);
        let example = parse_example(&proto_bytes).expect("parse example floats");
        let feature = example.features.get("scores").expect("scores feature");
        match feature {
            Feature::FloatList(vals) => {
                assert_eq!(vals.len(), 3);
                assert!((vals[0] - 1.0).abs() < 1e-6);
                assert!((vals[1] - 2.5).abs() < 1e-6);
                assert!((vals[2] - 3.25).abs() < 1e-5);
            }
            other => panic!("expected FloatList, got {other:?}"),
        }
    }

    #[test]
    fn test_tfrecord_example_roundtrip_via_file() {
        let proto_bytes = build_example_int64("ids", &[10, 20, 30]);
        let record_bytes = encode_tfrecord(&proto_bytes);

        let tmp_dir = env::temp_dir();
        let path = tmp_dir
            .join("test_example_ws146.tfrecord")
            .to_string_lossy()
            .to_string();
        let mut file = fs::File::create(&path).expect("create");
        file.write_all(&record_bytes).expect("write");
        drop(file);

        let records = read_all_records(&path).expect("read");
        assert_eq!(records.len(), 1);

        let example = parse_example(&records[0].data).expect("parse");
        match example.features.get("ids").expect("ids") {
            Feature::Int64List(vals) => assert_eq!(vals, &[10, 20, 30]),
            other => panic!("wrong feature type: {other:?}"),
        }

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_eof_returns_none() {
        let tmp_dir = env::temp_dir();
        let path = tmp_dir
            .join("test_empty_ws146.tfrecord")
            .to_string_lossy()
            .to_string();
        fs::File::create(&path).expect("create empty");

        let mut reader = TfRecordReader::open(&path).expect("open empty");
        let result = reader.next_record().expect("next_record on empty");
        assert!(result.is_none(), "expected None on empty file");

        let _ = fs::remove_file(&path);
    }
}
