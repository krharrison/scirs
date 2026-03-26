//! SafeTensors format read/write implementation.
//!
//! Implements the HuggingFace SafeTensors binary format:
//! - 8-byte little-endian u64 header size
//! - JSON header with tensor metadata
//! - Flat raw data buffer
//!
//! Reference: <https://github.com/huggingface/safetensors>

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};

use serde::{Deserialize, Serialize};

use crate::error::IoError;

/// Data type of a tensor stored in SafeTensors format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "UPPERCASE")]
pub enum DType {
    /// 64-bit float
    F64,
    /// 32-bit float
    F32,
    /// 16-bit float (IEEE 754 half)
    F16,
    /// Brain float (bfloat16)
    BF16,
    /// 64-bit signed integer
    I64,
    /// 32-bit signed integer
    I32,
    /// 16-bit signed integer
    I16,
    /// 8-bit signed integer
    I8,
    /// 8-bit unsigned integer
    U8,
    /// Boolean (1 byte per element)
    BOOL,
}

impl DType {
    /// Returns the number of bytes per element for this dtype.
    pub fn item_size(&self) -> usize {
        match self {
            DType::F64 | DType::I64 => 8,
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 | DType::I16 => 2,
            DType::I8 | DType::U8 | DType::BOOL => 1,
        }
    }

    /// Returns the canonical string representation used in the JSON header.
    pub fn as_str(&self) -> &'static str {
        match self {
            DType::F64 => "F64",
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            DType::I64 => "I64",
            DType::I32 => "I32",
            DType::I16 => "I16",
            DType::I8 => "I8",
            DType::U8 => "U8",
            DType::BOOL => "BOOL",
        }
    }

    /// Parse a dtype from its string representation.
    pub fn from_str(s: &str) -> Result<DType, IoError> {
        match s {
            "F64" => Ok(DType::F64),
            "F32" => Ok(DType::F32),
            "F16" => Ok(DType::F16),
            "BF16" => Ok(DType::BF16),
            "I64" => Ok(DType::I64),
            "I32" => Ok(DType::I32),
            "I16" => Ok(DType::I16),
            "I8" => Ok(DType::I8),
            "U8" => Ok(DType::U8),
            "BOOL" => Ok(DType::BOOL),
            other => Err(IoError::ParseError(format!("Unknown dtype: {other}"))),
        }
    }
}

/// Metadata for a single tensor in the SafeTensors header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMeta {
    /// Data type of the tensor elements.
    pub dtype: DType,
    /// Shape of the tensor (list of dimension sizes).
    pub shape: Vec<usize>,
    /// Byte offsets `[start, end)` into the flat data buffer.
    pub data_offsets: (usize, usize),
}

// Custom serde for TensorMeta to match SafeTensors JSON format
// The JSON format uses "data_offsets" as an array [start, end]
mod tensor_meta_serde {
    use super::*;
    use serde::{Deserializer, Serializer};

    pub fn serialize_offsets<S: Serializer>(
        offsets: &(usize, usize),
        ser: S,
    ) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut seq = ser.serialize_seq(Some(2))?;
        seq.serialize_element(&offsets.0)?;
        seq.serialize_element(&offsets.1)?;
        seq.end()
    }

    pub fn deserialize_offsets<'de, D: Deserializer<'de>>(
        de: D,
    ) -> Result<(usize, usize), D::Error> {
        let arr: [usize; 2] = Deserialize::deserialize(de)?;
        Ok((arr[0], arr[1]))
    }
}

// Re-define TensorMeta with proper serialization
#[derive(Debug, Clone)]
struct TensorMetaRaw {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

impl<'de> Deserialize<'de> for TensorMetaRaw {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::MapAccess;

        struct TensorMetaVisitor;

        impl<'de> serde::de::Visitor<'de> for TensorMetaVisitor {
            type Value = TensorMetaRaw;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a SafeTensors tensor metadata object")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut dtype = None;
                let mut shape = None;
                let mut data_offsets = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "dtype" => dtype = Some(map.next_value::<String>()?),
                        "shape" => shape = Some(map.next_value::<Vec<usize>>()?),
                        "data_offsets" => data_offsets = Some(map.next_value::<[usize; 2]>()?),
                        _ => {
                            let _ = map.next_value::<serde_json::Value>()?;
                        }
                    }
                }

                Ok(TensorMetaRaw {
                    dtype: dtype.ok_or_else(|| serde::de::Error::missing_field("dtype"))?,
                    shape: shape.ok_or_else(|| serde::de::Error::missing_field("shape"))?,
                    data_offsets: data_offsets
                        .ok_or_else(|| serde::de::Error::missing_field("data_offsets"))?,
                })
            }
        }

        deserializer.deserialize_map(TensorMetaVisitor)
    }
}

impl Serialize for TensorMetaRaw {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(3))?;
        map.serialize_entry("dtype", &self.dtype)?;
        map.serialize_entry("shape", &self.shape)?;
        map.serialize_entry("data_offsets", &self.data_offsets)?;
        map.end()
    }
}

/// The parsed JSON header of a SafeTensors file.
#[derive(Debug, Clone)]
pub struct SafeTensorsHeader {
    /// Map from tensor name to its metadata.
    pub tensors: HashMap<String, TensorMeta>,
}

impl SafeTensorsHeader {
    /// Serialize the header to a JSON string.
    fn to_json(&self) -> Result<String, IoError> {
        let raw: HashMap<String, TensorMetaRaw> = self
            .tensors
            .iter()
            .map(|(name, meta)| {
                (
                    name.clone(),
                    TensorMetaRaw {
                        dtype: meta.dtype.as_str().to_string(),
                        shape: meta.shape.clone(),
                        data_offsets: [meta.data_offsets.0, meta.data_offsets.1],
                    },
                )
            })
            .collect();
        serde_json::to_string(&raw)
            .map_err(|e| IoError::SerializationError(format!("JSON header serialize error: {e}")))
    }

    /// Parse a JSON string into a header.
    fn from_json(json: &str) -> Result<SafeTensorsHeader, IoError> {
        let raw: HashMap<String, TensorMetaRaw> = serde_json::from_str(json)
            .map_err(|e| IoError::ParseError(format!("JSON header parse error: {e}")))?;
        let mut tensors = HashMap::new();
        for (name, raw_meta) in raw {
            // Skip the special __metadata__ key
            if name == "__metadata__" {
                continue;
            }
            let dtype = DType::from_str(&raw_meta.dtype)?;
            tensors.insert(
                name,
                TensorMeta {
                    dtype,
                    shape: raw_meta.shape,
                    data_offsets: (raw_meta.data_offsets[0], raw_meta.data_offsets[1]),
                },
            );
        }
        Ok(SafeTensorsHeader { tensors })
    }
}

/// A parsed SafeTensors file: header metadata + flat raw data buffer.
#[derive(Debug, Clone)]
pub struct SafeTensors {
    /// Header containing tensor metadata.
    pub header: SafeTensorsHeader,
    /// Flat byte buffer containing all tensor data.
    pub data: Vec<u8>,
}

impl SafeTensors {
    /// Create an empty SafeTensors container.
    pub fn new() -> Self {
        SafeTensors {
            header: SafeTensorsHeader {
                tensors: HashMap::new(),
            },
            data: Vec::new(),
        }
    }

    /// Parse a SafeTensors byte buffer.
    ///
    /// Format:
    /// - Bytes 0..8: header_size (u64 little-endian)
    /// - Bytes 8..8+header_size: JSON header
    /// - Bytes 8+header_size..: flat tensor data
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, IoError> {
        if bytes.len() < 8 {
            return Err(IoError::FormatError(
                "SafeTensors: file too short to contain header size".to_string(),
            ));
        }

        let header_size = u64::from_le_bytes(
            bytes[0..8]
                .try_into()
                .map_err(|_| IoError::FormatError("Failed to read header size".to_string()))?,
        ) as usize;

        let header_end = 8 + header_size;
        if bytes.len() < header_end {
            return Err(IoError::FormatError(format!(
                "SafeTensors: file too short: need {} bytes for header, have {}",
                header_end,
                bytes.len()
            )));
        }

        let json_str = std::str::from_utf8(&bytes[8..header_end]).map_err(|e| {
            IoError::ParseError(format!("SafeTensors header is not valid UTF-8: {e}"))
        })?;

        let header = SafeTensorsHeader::from_json(json_str)?;
        let data = bytes[header_end..].to_vec();

        Ok(SafeTensors { header, data })
    }

    /// Serialize this SafeTensors container to bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, IoError> {
        let json_str = self.header.to_json()?;
        let json_bytes = json_str.as_bytes();
        let header_size = json_bytes.len() as u64;

        let mut out = Vec::with_capacity(8 + json_bytes.len() + self.data.len());
        out.extend_from_slice(&header_size.to_le_bytes());
        out.extend_from_slice(json_bytes);
        out.extend_from_slice(&self.data);
        Ok(out)
    }

    /// Extract a named tensor and convert its elements to f32.
    ///
    /// Supports F32, F64, I32, I64, I16, I8, U8 sources.
    pub fn get_tensor_f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), IoError> {
        let meta =
            self.header.tensors.get(name).ok_or_else(|| {
                IoError::NotFound(format!("SafeTensors: tensor '{name}' not found"))
            })?;

        let (start, end) = meta.data_offsets;
        if end > self.data.len() {
            return Err(IoError::FormatError(format!(
                "Tensor '{name}' data_offsets [{start},{end}) exceed data buffer length {}",
                self.data.len()
            )));
        }
        let raw = &self.data[start..end];
        let values = bytes_to_f32(raw, meta.dtype)?;
        Ok((meta.shape.clone(), values))
    }

    /// Extract a named tensor and convert its elements to f64.
    pub fn get_tensor_f64(&self, name: &str) -> Result<(Vec<usize>, Vec<f64>), IoError> {
        let meta =
            self.header.tensors.get(name).ok_or_else(|| {
                IoError::NotFound(format!("SafeTensors: tensor '{name}' not found"))
            })?;

        let (start, end) = meta.data_offsets;
        if end > self.data.len() {
            return Err(IoError::FormatError(format!(
                "Tensor '{name}' data_offsets [{start},{end}) exceed data buffer length {}",
                self.data.len()
            )));
        }
        let raw = &self.data[start..end];
        let values = bytes_to_f64(raw, meta.dtype)?;
        Ok((meta.shape.clone(), values))
    }

    /// Insert an f32 tensor into this container.
    pub fn insert_f32(
        &mut self,
        name: String,
        shape: Vec<usize>,
        data: &[f32],
    ) -> Result<(), IoError> {
        let expected_numel: usize = shape.iter().product();
        if data.len() != expected_numel {
            return Err(IoError::ValidationError(format!(
                "insert_f32: shape product {expected_numel} != data length {}",
                data.len()
            )));
        }
        let start = self.data.len();
        for &v in data {
            self.data.extend_from_slice(&v.to_le_bytes());
        }
        let end = self.data.len();
        self.header.tensors.insert(
            name,
            TensorMeta {
                dtype: DType::F32,
                shape,
                data_offsets: (start, end),
            },
        );
        Ok(())
    }

    /// Insert an f64 tensor into this container.
    pub fn insert_f64(
        &mut self,
        name: String,
        shape: Vec<usize>,
        data: &[f64],
    ) -> Result<(), IoError> {
        let expected_numel: usize = shape.iter().product();
        if data.len() != expected_numel {
            return Err(IoError::ValidationError(format!(
                "insert_f64: shape product {expected_numel} != data length {}",
                data.len()
            )));
        }
        let start = self.data.len();
        for &v in data {
            self.data.extend_from_slice(&v.to_le_bytes());
        }
        let end = self.data.len();
        self.header.tensors.insert(
            name,
            TensorMeta {
                dtype: DType::F64,
                shape,
                data_offsets: (start, end),
            },
        );
        Ok(())
    }

    /// Returns the names of all tensors in this container.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.header.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Convenience: save a list of raw tensors to a file.
    ///
    /// Each entry is `(name, shape, dtype, raw_bytes)`.
    pub fn save_to_file(
        path: &str,
        tensors: &[(String, Vec<usize>, DType, Vec<u8>)],
    ) -> Result<(), IoError> {
        let mut st = SafeTensors::new();
        for (name, shape, dtype, raw) in tensors {
            let expected_bytes: usize = shape.iter().product::<usize>() * dtype.item_size();
            if raw.len() != expected_bytes {
                return Err(IoError::ValidationError(format!(
                    "save_to_file: tensor '{name}' expected {expected_bytes} bytes, got {}",
                    raw.len()
                )));
            }
            let start = st.data.len();
            st.data.extend_from_slice(raw);
            let end = st.data.len();
            st.header.tensors.insert(
                name.clone(),
                TensorMeta {
                    dtype: *dtype,
                    shape: shape.clone(),
                    data_offsets: (start, end),
                },
            );
        }
        let bytes = st.to_bytes()?;
        let mut file = fs::File::create(path)
            .map_err(|e| IoError::FileError(format!("Cannot create '{path}': {e}")))?;
        file.write_all(&bytes)
            .map_err(|e| IoError::FileError(format!("Cannot write '{path}': {e}")))?;
        Ok(())
    }

    /// Convenience: load a SafeTensors file from disk.
    pub fn load_from_file(path: &str) -> Result<SafeTensors, IoError> {
        let mut file = fs::File::open(path)
            .map_err(|e| IoError::FileNotFound(format!("Cannot open '{path}': {e}")))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|e| IoError::FileError(format!("Cannot read '{path}': {e}")))?;
        SafeTensors::from_bytes(&bytes)
    }
}

impl Default for SafeTensors {
    fn default() -> Self {
        Self::new()
    }
}

// ---- helper: byte slice → typed elements ----

fn bytes_to_f32(raw: &[u8], dtype: DType) -> Result<Vec<f32>, IoError> {
    let item = dtype.item_size();
    if !raw.len().is_multiple_of(item) {
        return Err(IoError::FormatError(format!(
            "bytes_to_f32: buffer length {} not divisible by item_size {}",
            raw.len(),
            item
        )));
    }
    let n = raw.len() / item;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let chunk = &raw[i * item..(i + 1) * item];
        let v: f32 = match dtype {
            DType::F32 => {
                let arr: [u8; 4] = chunk.try_into().map_err(|_| {
                    IoError::FormatError("bytes_to_f32: F32 chunk slice error".to_string())
                })?;
                f32::from_le_bytes(arr)
            }
            DType::F64 => {
                let arr: [u8; 8] = chunk.try_into().map_err(|_| {
                    IoError::FormatError("bytes_to_f32: F64 chunk slice error".to_string())
                })?;
                f64::from_le_bytes(arr) as f32
            }
            DType::F16 => {
                let arr: [u8; 2] = chunk.try_into().map_err(|_| {
                    IoError::FormatError("bytes_to_f32: F16 chunk slice error".to_string())
                })?;
                f16_to_f32(u16::from_le_bytes(arr))
            }
            DType::BF16 => {
                let arr: [u8; 2] = chunk.try_into().map_err(|_| {
                    IoError::FormatError("bytes_to_f32: BF16 chunk slice error".to_string())
                })?;
                bf16_to_f32(u16::from_le_bytes(arr))
            }
            DType::I64 => {
                let arr: [u8; 8] = chunk.try_into().map_err(|_| {
                    IoError::FormatError("bytes_to_f32: I64 chunk slice error".to_string())
                })?;
                i64::from_le_bytes(arr) as f32
            }
            DType::I32 => {
                let arr: [u8; 4] = chunk.try_into().map_err(|_| {
                    IoError::FormatError("bytes_to_f32: I32 chunk slice error".to_string())
                })?;
                i32::from_le_bytes(arr) as f32
            }
            DType::I16 => {
                let arr: [u8; 2] = chunk.try_into().map_err(|_| {
                    IoError::FormatError("bytes_to_f32: I16 chunk slice error".to_string())
                })?;
                i16::from_le_bytes(arr) as f32
            }
            DType::I8 => chunk[0] as i8 as f32,
            DType::U8 => chunk[0] as f32,
            DType::BOOL => {
                if chunk[0] != 0 {
                    1.0_f32
                } else {
                    0.0_f32
                }
            }
        };
        out.push(v);
    }
    Ok(out)
}

fn bytes_to_f64(raw: &[u8], dtype: DType) -> Result<Vec<f64>, IoError> {
    let item = dtype.item_size();
    if !raw.len().is_multiple_of(item) {
        return Err(IoError::FormatError(format!(
            "bytes_to_f64: buffer length {} not divisible by item_size {}",
            raw.len(),
            item
        )));
    }
    let n = raw.len() / item;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let chunk = &raw[i * item..(i + 1) * item];
        let v: f64 = match dtype {
            DType::F64 => {
                let arr: [u8; 8] = chunk.try_into().map_err(|_| {
                    IoError::FormatError("bytes_to_f64: F64 chunk slice error".to_string())
                })?;
                f64::from_le_bytes(arr)
            }
            DType::F32 => {
                let arr: [u8; 4] = chunk.try_into().map_err(|_| {
                    IoError::FormatError("bytes_to_f64: F32 chunk slice error".to_string())
                })?;
                f32::from_le_bytes(arr) as f64
            }
            DType::F16 => {
                let arr: [u8; 2] = chunk.try_into().map_err(|_| {
                    IoError::FormatError("bytes_to_f64: F16 chunk slice error".to_string())
                })?;
                f16_to_f32(u16::from_le_bytes(arr)) as f64
            }
            DType::BF16 => {
                let arr: [u8; 2] = chunk.try_into().map_err(|_| {
                    IoError::FormatError("bytes_to_f64: BF16 chunk slice error".to_string())
                })?;
                bf16_to_f32(u16::from_le_bytes(arr)) as f64
            }
            DType::I64 => {
                let arr: [u8; 8] = chunk.try_into().map_err(|_| {
                    IoError::FormatError("bytes_to_f64: I64 chunk slice error".to_string())
                })?;
                i64::from_le_bytes(arr) as f64
            }
            DType::I32 => {
                let arr: [u8; 4] = chunk.try_into().map_err(|_| {
                    IoError::FormatError("bytes_to_f64: I32 chunk slice error".to_string())
                })?;
                i32::from_le_bytes(arr) as f64
            }
            DType::I16 => {
                let arr: [u8; 2] = chunk.try_into().map_err(|_| {
                    IoError::FormatError("bytes_to_f64: I16 chunk slice error".to_string())
                })?;
                i16::from_le_bytes(arr) as f64
            }
            DType::I8 => chunk[0] as i8 as f64,
            DType::U8 => chunk[0] as f64,
            DType::BOOL => {
                if chunk[0] != 0 {
                    1.0_f64
                } else {
                    0.0_f64
                }
            }
        };
        out.push(v);
    }
    Ok(out)
}

/// Convert IEEE 754 half-precision (F16) bits to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits & 0x7C00) as u32) >> 10;
    let frac = (bits & 0x03FF) as u32;

    let (new_exp, new_frac) = if exp == 0 {
        if frac == 0 {
            (0u32, 0u32)
        } else {
            // Subnormal: normalise
            let mut e = 0u32;
            let mut f = frac << 1;
            while f & 0x400 == 0 {
                f <<= 1;
                e += 1;
            }
            (127 - 14 - e, (f & 0x3FF) << 13)
        }
    } else if exp == 0x1F {
        (0xFF, frac << 13) // Inf/NaN
    } else {
        (exp + 127 - 15, frac << 13)
    };

    f32::from_bits(sign | (new_exp << 23) | new_frac)
}

/// Convert bfloat16 bits to f32 (simple zero-padding of mantissa).
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_dtype_item_size() {
        assert_eq!(DType::F64.item_size(), 8);
        assert_eq!(DType::F32.item_size(), 4);
        assert_eq!(DType::F16.item_size(), 2);
        assert_eq!(DType::BF16.item_size(), 2);
        assert_eq!(DType::I64.item_size(), 8);
        assert_eq!(DType::I32.item_size(), 4);
        assert_eq!(DType::I16.item_size(), 2);
        assert_eq!(DType::I8.item_size(), 1);
        assert_eq!(DType::U8.item_size(), 1);
        assert_eq!(DType::BOOL.item_size(), 1);
    }

    #[test]
    fn test_dtype_roundtrip() {
        for dtype in &[
            DType::F64,
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::I64,
            DType::I32,
            DType::I16,
            DType::I8,
            DType::U8,
            DType::BOOL,
        ] {
            let s = dtype.as_str();
            let parsed = DType::from_str(s).expect("roundtrip");
            assert_eq!(parsed, *dtype, "dtype roundtrip failed for {s}");
        }
    }

    #[test]
    fn test_insert_and_extract_f32() {
        let mut st = SafeTensors::new();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        st.insert_f32("weight".to_string(), vec![2, 3], &data)
            .expect("insert_f32");

        let (shape, extracted) = st.get_tensor_f32("weight").expect("get_tensor_f32");
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(extracted.len(), 6);
        for (a, b) in data.iter().zip(extracted.iter()) {
            assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_insert_and_extract_f64() {
        let mut st = SafeTensors::new();
        let data: Vec<f64> = vec![10.0, 20.0, 30.0];
        st.insert_f64("bias".to_string(), vec![3], &data)
            .expect("insert_f64");

        let (shape, extracted) = st.get_tensor_f64("bias").expect("get_tensor_f64");
        assert_eq!(shape, vec![3]);
        for (a, b) in data.iter().zip(extracted.iter()) {
            assert!((a - b).abs() < 1e-12, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_bytes_roundtrip() {
        let mut st = SafeTensors::new();
        let data_f32: Vec<f32> = vec![1.5, 2.5, 3.5];
        let data_f64: Vec<f64> = vec![100.0, 200.0];
        st.insert_f32("a".to_string(), vec![3], &data_f32)
            .expect("insert a");
        st.insert_f64("b".to_string(), vec![2], &data_f64)
            .expect("insert b");

        let bytes = st.to_bytes().expect("to_bytes");
        let st2 = SafeTensors::from_bytes(&bytes).expect("from_bytes");

        let (_, a2) = st2.get_tensor_f32("a").expect("get a");
        let (_, b2) = st2.get_tensor_f64("b").expect("get b");
        for (orig, got) in data_f32.iter().zip(a2.iter()) {
            assert!((orig - got).abs() < 1e-6);
        }
        for (orig, got) in data_f64.iter().zip(b2.iter()) {
            assert!((orig - got).abs() < 1e-12);
        }
    }

    #[test]
    fn test_save_load_file() {
        let tmp_dir = env::temp_dir();
        let path = tmp_dir
            .join("test_safetensors.safetensors")
            .to_string_lossy()
            .to_string();

        let data_f32: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let raw_bytes: Vec<u8> = data_f32.iter().flat_map(|v| v.to_le_bytes()).collect();

        SafeTensors::save_to_file(
            &path,
            &[(
                "layer.weight".to_string(),
                vec![2, 2],
                DType::F32,
                raw_bytes,
            )],
        )
        .expect("save_to_file");

        let st = SafeTensors::load_from_file(&path).expect("load_from_file");
        let (shape, vals) = st.get_tensor_f32("layer.weight").expect("get tensor");
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(vals.len(), 4);
        for (orig, got) in data_f32.iter().zip(vals.iter()) {
            assert!((orig - got).abs() < 1e-6, "file roundtrip mismatch");
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_tensor_names() {
        let mut st = SafeTensors::new();
        st.insert_f32("w1".to_string(), vec![2], &[1.0_f32, 2.0])
            .expect("insert w1");
        st.insert_f32("w2".to_string(), vec![3], &[3.0_f32, 4.0, 5.0])
            .expect("insert w2");
        let mut names = st.tensor_names();
        names.sort();
        assert_eq!(names, vec!["w1", "w2"]);
    }

    #[test]
    fn test_error_not_found() {
        let st = SafeTensors::new();
        let result = st.get_tensor_f32("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_f16_conversion() {
        // 1.0 in F16 = 0x3C00
        let v = f16_to_f32(0x3C00);
        assert!((v - 1.0_f32).abs() < 1e-5, "f16 1.0 = {v}");
        // 0.0 in F16 = 0x0000
        let z = f16_to_f32(0x0000);
        assert_eq!(z, 0.0_f32);
    }

    #[test]
    fn test_bf16_conversion() {
        // 1.0 in BF16 = 0x3F80
        let v = bf16_to_f32(0x3F80);
        assert!((v - 1.0_f32).abs() < 1e-5, "bf16 1.0 = {v}");
    }
}
