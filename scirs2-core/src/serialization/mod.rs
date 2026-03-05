//! Cross-language serialization protocol for SciRS2.
//!
//! Provides a versioned binary format (`.scirs2` files) that can be
//! read by Python bindings, WASM modules, and native Rust code.
//!
//! # Format
//!
//! Header: 64 bytes, little-endian
//! - Magic: `b"SCIRS2\0\0"` (8 bytes)
//! - Version: u16 major, u16 minor (4 bytes)
//! - Payload type: u8 (1 byte): 0=array, 1=model, 2=stats, 3=custom
//! - Compression: u8 (1 byte): 0=none, 1=lz4, 2=zstd
//! - Checksum: u32 CRC32 of **uncompressed** payload (4 bytes)
//! - Payload length: u64 (8 bytes) — bytes actually stored on disk
//! - Reserved: 38 bytes of zeros
//!
//! Payload (variable length, may be compressed):
//! - For arrays: `dtype(u8)` + `ndim(u8)` + `shape(u64 each, little-endian)` + raw element bytes (little-endian)
//! - For models: JSON config + raw parameter bytes
//! - For stats: key-value pairs with typed values
//! - For custom: raw bytes (caller-defined format)
//!
//! # Example
//!
//! ```no_run
//! use scirs2_core::serialization::{save_array, load_array, CompressionType};
//! use ndarray::Array2;
//! use std::path::Path;
//!
//! let data = Array2::<f32>::ones((3, 4)).into_dyn();
//! let path = Path::new("/tmp/test.scirs2");
//! save_array(path, &data, CompressionType::None).expect("should succeed");
//! let loaded = load_array::<f32>(path).expect("should succeed");
//! assert_eq!(data, loaded);
//! ```

use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use ndarray::{Array, IxDyn};

const MAGIC: &[u8; 8] = b"SCIRS2\0\0";
const VERSION_MAJOR: u16 = 0;
const VERSION_MINOR: u16 = 3;
const HEADER_SIZE: usize = 64;

// Byte offsets within the 64-byte header
const OFFSET_MAGIC: usize = 0;
const OFFSET_VERSION_MAJOR: usize = 8;
const OFFSET_VERSION_MINOR: usize = 10;
const OFFSET_PAYLOAD_TYPE: usize = 12;
const OFFSET_COMPRESSION: usize = 13;
const OFFSET_CHECKSUM: usize = 14;
const OFFSET_PAYLOAD_LENGTH: usize = 18;
// bytes 26..64 are reserved (38 bytes, must be zero)

// ─── PayloadType ──────────────────────────────────────────────────────────────

/// Identifies the kind of data stored in the `.scirs2` payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PayloadType {
    /// N-dimensional array with dtype prefix and shape encoding.
    Array = 0,
    /// Model: JSON config followed by raw parameter bytes.
    Model = 1,
    /// Statistics: key-value pairs with typed values.
    Stats = 2,
    /// Custom: raw bytes with caller-defined semantics.
    Custom = 3,
}

impl PayloadType {
    fn from_u8(v: u8) -> Result<Self, SerializationError> {
        match v {
            0 => Ok(Self::Array),
            1 => Ok(Self::Model),
            2 => Ok(Self::Stats),
            3 => Ok(Self::Custom),
            other => Err(SerializationError::UnknownPayloadType(other)),
        }
    }
}

// ─── CompressionType ──────────────────────────────────────────────────────────

/// Compression algorithm applied to the payload bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CompressionType {
    /// No compression; payload stored verbatim.
    None = 0,
    /// LZ4 frame compression — very fast, moderate ratio.
    Lz4 = 1,
    /// Zstandard compression — moderate speed, excellent ratio.
    Zstd = 2,
}

impl CompressionType {
    fn from_u8(v: u8) -> Result<Self, SerializationError> {
        match v {
            0 => Ok(Self::None),
            1 => Ok(Self::Lz4),
            2 => Ok(Self::Zstd),
            other => Err(SerializationError::Compression(format!(
                "unknown compression type byte: {}",
                other
            ))),
        }
    }
}

// ─── Header ───────────────────────────────────────────────────────────────────

/// Parsed `.scirs2` file header (64 bytes).
#[derive(Debug, Clone)]
pub struct Scirs2Header {
    /// `(major, minor)` format version.
    pub version: (u16, u16),
    /// Kind of data in the payload.
    pub payload_type: PayloadType,
    /// How the payload is compressed on disk.
    pub compression: CompressionType,
    /// CRC32 of the **uncompressed** payload.
    pub checksum: u32,
    /// Byte count stored on disk (after optional compression).
    pub payload_length: u64,
}

impl Scirs2Header {
    /// Serialize this header into a fixed 64-byte array.
    fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[OFFSET_MAGIC..OFFSET_MAGIC + 8].copy_from_slice(MAGIC);
        buf[OFFSET_VERSION_MAJOR..OFFSET_VERSION_MAJOR + 2]
            .copy_from_slice(&self.version.0.to_le_bytes());
        buf[OFFSET_VERSION_MINOR..OFFSET_VERSION_MINOR + 2]
            .copy_from_slice(&self.version.1.to_le_bytes());
        buf[OFFSET_PAYLOAD_TYPE] = self.payload_type as u8;
        buf[OFFSET_COMPRESSION] = self.compression as u8;
        buf[OFFSET_CHECKSUM..OFFSET_CHECKSUM + 4].copy_from_slice(&self.checksum.to_le_bytes());
        buf[OFFSET_PAYLOAD_LENGTH..OFFSET_PAYLOAD_LENGTH + 8]
            .copy_from_slice(&self.payload_length.to_le_bytes());
        // bytes 26..64 remain zero (reserved)
        buf
    }

    /// Parse a 64-byte buffer into a `Scirs2Header`.
    fn from_bytes(buf: &[u8; HEADER_SIZE]) -> Result<Self, SerializationError> {
        // Validate magic
        if &buf[OFFSET_MAGIC..OFFSET_MAGIC + 8] != MAGIC.as_slice() {
            return Err(SerializationError::InvalidMagic);
        }

        let major = u16::from_le_bytes([buf[OFFSET_VERSION_MAJOR], buf[OFFSET_VERSION_MAJOR + 1]]);
        let minor = u16::from_le_bytes([buf[OFFSET_VERSION_MINOR], buf[OFFSET_VERSION_MINOR + 1]]);

        // Forward-compatibility: reject files written by a future major version
        if major > VERSION_MAJOR {
            return Err(SerializationError::UnsupportedVersion(major, minor));
        }

        let payload_type = PayloadType::from_u8(buf[OFFSET_PAYLOAD_TYPE])?;
        let compression = CompressionType::from_u8(buf[OFFSET_COMPRESSION])?;

        let checksum = u32::from_le_bytes([
            buf[OFFSET_CHECKSUM],
            buf[OFFSET_CHECKSUM + 1],
            buf[OFFSET_CHECKSUM + 2],
            buf[OFFSET_CHECKSUM + 3],
        ]);

        // SAFETY: OFFSET_PAYLOAD_LENGTH..OFFSET_PAYLOAD_LENGTH+8 is always within [0,64)
        let pl_bytes: [u8; 8] = buf[OFFSET_PAYLOAD_LENGTH..OFFSET_PAYLOAD_LENGTH + 8]
            .try_into()
            .map_err(|_| {
                SerializationError::Io(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "internal: slice length invariant violated reading payload_length",
                ))
            })?;
        let payload_length = u64::from_le_bytes(pl_bytes);

        Ok(Self {
            version: (major, minor),
            payload_type,
            compression,
            checksum,
            payload_length,
        })
    }
}

// ─── Scirs2Writer ─────────────────────────────────────────────────────────────

/// Low-level writer for `.scirs2` files.
///
/// Writes exactly one payload per instance. For multiple payloads
/// use separate writers.
///
/// # Example
///
/// ```no_run
/// use scirs2_core::serialization::{Scirs2Writer, PayloadType, CompressionType};
/// use std::fs::File;
///
/// let file = File::create("/tmp/out.scirs2").expect("should succeed");
/// let mut writer = Scirs2Writer::new(file);
/// writer
///     .write_payload(PayloadType::Custom, b"hello scirs2", CompressionType::None)
///     .expect("should succeed");
/// ```
pub struct Scirs2Writer<W: Write> {
    inner: W,
}

impl<W: Write> Scirs2Writer<W> {
    /// Wrap an existing [`Write`] implementor.
    pub fn new(writer: W) -> Self {
        Self { inner: writer }
    }

    /// Compress (if requested) and write a single payload to the underlying writer.
    ///
    /// The CRC32 checksum is computed over the **uncompressed** `payload` bytes.
    /// The stored bytes (in the file, after the header) may be compressed.
    pub fn write_payload(
        &mut self,
        payload_type: PayloadType,
        payload: &[u8],
        compression: CompressionType,
    ) -> Result<(), SerializationError> {
        let checksum = crc32fast::hash(payload);
        let stored = compress_payload(payload, compression)?;

        let header = Scirs2Header {
            version: (VERSION_MAJOR, VERSION_MINOR),
            payload_type,
            compression,
            checksum,
            payload_length: stored.len() as u64,
        };

        self.inner.write_all(&header.to_bytes())?;
        self.inner.write_all(&stored)?;
        Ok(())
    }
}

// ─── Scirs2Reader ─────────────────────────────────────────────────────────────

/// Low-level reader for `.scirs2` files.
///
/// The header is parsed eagerly on construction; the payload bytes are read
/// lazily on demand.
///
/// # Example
///
/// ```no_run
/// use scirs2_core::serialization::Scirs2Reader;
/// use std::fs::File;
/// use std::io::BufReader;
///
/// let file = BufReader::new(File::open("/tmp/out.scirs2").expect("should succeed"));
/// let mut reader = Scirs2Reader::new(file).expect("should succeed");
/// println!("payload type = {:?}", reader.header.payload_type);
/// let bytes = reader.read_payload().expect("should succeed");
/// ```
pub struct Scirs2Reader<R: Read + Seek> {
    inner: R,
    /// Header parsed from the beginning of the file.
    pub header: Scirs2Header,
}

impl<R: Read + Seek> Scirs2Reader<R> {
    /// Open a `.scirs2` reader, validating and parsing the header immediately.
    ///
    /// Returns [`SerializationError::InvalidMagic`] when the file is not a
    /// valid `.scirs2` file, or [`SerializationError::UnsupportedVersion`] when
    /// the format major version is newer than this library.
    pub fn new(mut reader: R) -> Result<Self, SerializationError> {
        let mut buf = [0u8; HEADER_SIZE];
        reader.read_exact(&mut buf)?;
        let header = Scirs2Header::from_bytes(&buf)?;
        Ok(Self {
            inner: reader,
            header,
        })
    }

    /// Read and decompress the payload, returning the raw (uncompressed) bytes.
    ///
    /// This method seeks back to the start of the payload each time it is
    /// called, so repeated calls are safe.
    pub fn read_payload(&mut self) -> Result<Vec<u8>, SerializationError> {
        self.inner.seek(SeekFrom::Start(HEADER_SIZE as u64))?;

        let len = self.header.payload_length as usize;
        let mut stored = vec![0u8; len];
        self.inner.read_exact(&mut stored)?;

        decompress_payload(&stored, self.header.compression, len)
    }

    /// Read the payload and verify its CRC32 against the header checksum.
    ///
    /// Returns `Ok(true)` if the checksum matches, `Ok(false)` otherwise.
    /// Returns `Err` on I/O or decompression failure.
    pub fn verify_checksum(&mut self) -> Result<bool, SerializationError> {
        let payload = self.read_payload()?;
        let computed = crc32fast::hash(&payload);
        Ok(computed == self.header.checksum)
    }
}

// ─── Compression helpers ──────────────────────────────────────────────────────

/// Compress `data` using the requested algorithm.
fn compress_payload(
    data: &[u8],
    compression: CompressionType,
) -> Result<Vec<u8>, SerializationError> {
    match compression {
        CompressionType::None => Ok(data.to_vec()),

        CompressionType::Lz4 => {
            #[cfg(feature = "serialization")]
            {
                oxiarc_lz4::compress(data)
                    .map_err(|e| SerializationError::Compression(format!("LZ4 compress: {}", e)))
            }
            #[cfg(not(feature = "serialization"))]
            {
                let _ = data;
                Err(SerializationError::Compression(
                    "LZ4 compression requires the `serialization` feature".to_string(),
                ))
            }
        }

        CompressionType::Zstd => {
            #[cfg(feature = "serialization")]
            {
                oxiarc_zstd::compress(data)
                    .map_err(|e| SerializationError::Compression(format!("Zstd compress: {}", e)))
            }
            #[cfg(not(feature = "serialization"))]
            {
                let _ = data;
                Err(SerializationError::Compression(
                    "Zstd compression requires the `serialization` feature".to_string(),
                ))
            }
        }
    }
}

/// Decompress `data` using the stored compression type.
///
/// `stored_len` is used as a hint for decompressors that require an output
/// size hint (LZ4 frame decompressor uses `stored_len * 4` as the upper bound).
fn decompress_payload(
    data: &[u8],
    compression: CompressionType,
    stored_len: usize,
) -> Result<Vec<u8>, SerializationError> {
    match compression {
        CompressionType::None => Ok(data.to_vec()),

        CompressionType::Lz4 => {
            #[cfg(feature = "serialization")]
            {
                // LZ4 frame decompression: use 4× the stored size as an upper bound.
                // For highly compressible data this might need to be larger; in practice
                // scientific arrays have at most ~8× decompression ratios.
                let max_output = stored_len.saturating_mul(8).max(4096);
                oxiarc_lz4::decompress(data, max_output)
                    .map_err(|e| SerializationError::Compression(format!("LZ4 decompress: {}", e)))
            }
            #[cfg(not(feature = "serialization"))]
            {
                let _ = (data, stored_len);
                Err(SerializationError::Compression(
                    "LZ4 decompression requires the `serialization` feature".to_string(),
                ))
            }
        }

        CompressionType::Zstd => {
            #[cfg(feature = "serialization")]
            {
                let _ = stored_len;
                oxiarc_zstd::decompress(data)
                    .map_err(|e| SerializationError::Compression(format!("Zstd decompress: {}", e)))
            }
            #[cfg(not(feature = "serialization"))]
            {
                let _ = (data, stored_len);
                Err(SerializationError::Compression(
                    "Zstd decompression requires the `serialization` feature".to_string(),
                ))
            }
        }
    }
}

// ─── ArrayElement trait ───────────────────────────────────────────────────────

/// Element type supported by the `.scirs2` array payload.
///
/// Each concrete type carries a stable 1-byte `dtype_id` embedded in the file,
/// enabling typed deserialization and cross-language interoperability.
///
/// # Dtype IDs
///
/// | ID | Type |
/// |----|------|
/// | 1  | f32  |
/// | 2  | f64  |
/// | 3  | i32  |
/// | 4  | i64  |
/// | 5  | u32  |
/// | 6  | u64  |
pub trait ArrayElement: Copy + 'static {
    /// Stable 1-byte dtype identifier embedded in the binary format.
    fn dtype_id() -> u8;
    /// Size in bytes of one element.
    fn element_size() -> usize;
    /// Deserialize `n` elements from a little-endian byte slice.
    fn from_le_bytes_slice(bytes: &[u8], n: usize) -> Vec<Self>;
    /// Serialize a slice of elements to little-endian bytes.
    fn to_le_bytes_vec(slice: &[Self]) -> Vec<u8>;
}

/// Implement `ArrayElement` for a primitive numeric type.
macro_rules! impl_array_element {
    ($ty:ty, $id:expr, $size:expr, $arr:expr) => {
        impl ArrayElement for $ty {
            fn dtype_id() -> u8 {
                $id
            }
            fn element_size() -> usize {
                $size
            }

            fn from_le_bytes_slice(bytes: &[u8], n: usize) -> Vec<Self> {
                (0..n)
                    .map(|i| {
                        let start = i * $size;
                        // We checked that `bytes` has enough data before calling this
                        let arr: [u8; $size] =
                            bytes[start..start + $size].try_into().unwrap_or($arr);
                        <$ty>::from_le_bytes(arr)
                    })
                    .collect()
            }

            fn to_le_bytes_vec(slice: &[Self]) -> Vec<u8> {
                slice.iter().flat_map(|v| v.to_le_bytes()).collect()
            }
        }
    };
}

impl_array_element!(f32, 1, 4, [0u8; 4]);
impl_array_element!(f64, 2, 8, [0u8; 8]);
impl_array_element!(i32, 3, 4, [0u8; 4]);
impl_array_element!(i64, 4, 8, [0u8; 8]);
impl_array_element!(u32, 5, 4, [0u8; 4]);
impl_array_element!(u64, 6, 8, [0u8; 8]);

// ─── Array encoding / decoding ────────────────────────────────────────────────

/// Encode an ndarray into the `.scirs2` array payload format.
///
/// Layout: `dtype_id(u8)` | `ndim(u8)` | `dim_0(u64le)` | … | `dim_{n-1}(u64le)` | `data(le bytes)`
fn encode_array<F: ArrayElement>(array: &Array<F, IxDyn>) -> Vec<u8> {
    let shape = array.shape();
    let ndim = shape.len();

    let header_bytes = 2 + ndim * 8;
    let data_bytes = array.len() * F::element_size();
    let mut buf = Vec::with_capacity(header_bytes + data_bytes);

    buf.push(F::dtype_id());
    buf.push(ndim as u8);

    for &dim in shape {
        buf.extend_from_slice(&(dim as u64).to_le_bytes());
    }

    // Collect in C-contiguous (row-major) iteration order
    let data: Vec<F> = array.iter().copied().collect();
    buf.extend_from_slice(&F::to_le_bytes_vec(&data));

    buf
}

/// Decode an ndarray from the `.scirs2` array payload format.
fn decode_array<F: ArrayElement>(payload: &[u8]) -> Result<Array<F, IxDyn>, SerializationError> {
    if payload.len() < 2 {
        return Err(SerializationError::Io(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "payload too short to contain array header (need at least 2 bytes)",
        )));
    }

    let actual_dtype = payload[0];
    let expected_dtype = F::dtype_id();
    if actual_dtype != expected_dtype {
        return Err(SerializationError::TypeMismatch {
            expected: expected_dtype,
            actual: actual_dtype,
        });
    }

    let ndim = payload[1] as usize;
    let shape_end = 2 + ndim * 8;

    if payload.len() < shape_end {
        return Err(SerializationError::Io(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!(
                "payload too short to read shape: need {} bytes for {} dims, have {}",
                shape_end,
                ndim,
                payload.len()
            ),
        )));
    }

    let mut shape = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let offset = 2 + i * 8;
        let dim_bytes: [u8; 8] = payload[offset..offset + 8].try_into().map_err(|_| {
            SerializationError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("internal: failed to read dim {} from payload", i),
            ))
        })?;
        shape.push(u64::from_le_bytes(dim_bytes) as usize);
    }

    let n_elements: usize = shape.iter().product();
    let data_bytes = n_elements * F::element_size();

    if payload.len() < shape_end + data_bytes {
        return Err(SerializationError::Io(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!(
                "payload too short for array data: need {} bytes, have {}",
                shape_end + data_bytes,
                payload.len()
            ),
        )));
    }

    let elements = F::from_le_bytes_slice(&payload[shape_end..shape_end + data_bytes], n_elements);

    Array::from_shape_vec(IxDyn(&shape), elements).map_err(|e| {
        SerializationError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("shape/data mismatch during array reconstruction: {}", e),
        ))
    })
}

// ─── Public convenience API ───────────────────────────────────────────────────

/// Save an n-dimensional array to a `.scirs2` file.
///
/// The file is created (or truncated) at `path`. The element type `F` is
/// embedded in the payload so [`load_array`] can verify type safety on load.
///
/// # Arguments
///
/// * `path` — Destination file path.
/// * `array` — The array to serialize.
/// * `compression` — Compression algorithm applied to the payload.
///
/// # Errors
///
/// Returns [`SerializationError`] on I/O failures or if the chosen compression
/// algorithm is unavailable in this build.
///
/// # Example
///
/// ```no_run
/// use scirs2_core::serialization::{save_array, CompressionType};
/// use ndarray::Array2;
///
/// let data = Array2::<f64>::eye(4).into_dyn();
/// save_array(std::path::Path::new("/tmp/eye4.scirs2"), &data, CompressionType::None).expect("should succeed");
/// ```
pub fn save_array<F: ArrayElement>(
    path: &Path,
    array: &Array<F, IxDyn>,
    compression: CompressionType,
) -> Result<(), SerializationError> {
    let file = std::fs::File::create(path)?;
    let writer = BufWriter::new(file);
    let mut scirs2 = Scirs2Writer::new(writer);
    let payload = encode_array(array);
    scirs2.write_payload(PayloadType::Array, &payload, compression)
}

/// Load an n-dimensional array from a `.scirs2` file.
///
/// The element type `F` is checked against the dtype stored in the file;
/// a [`SerializationError::TypeMismatch`] is returned if they differ.
///
/// # Example
///
/// ```no_run
/// use scirs2_core::serialization::load_array;
///
/// let arr = load_array::<f64>(std::path::Path::new("/tmp/eye4.scirs2")).expect("should succeed");
/// println!("shape: {:?}", arr.shape());
/// ```
pub fn load_array<F: ArrayElement>(path: &Path) -> Result<Array<F, IxDyn>, SerializationError> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut scirs2 = Scirs2Reader::new(reader)?;

    if scirs2.header.payload_type != PayloadType::Array {
        return Err(SerializationError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "expected Array payload type (0), found {:?} ({})",
                scirs2.header.payload_type, scirs2.header.payload_type as u8
            ),
        )));
    }

    let payload = scirs2.read_payload()?;
    decode_array::<F>(&payload)
}

// ─── Error type ───────────────────────────────────────────────────────────────

/// Errors that can occur during `.scirs2` serialization or deserialization.
#[derive(Debug, thiserror::Error)]
pub enum SerializationError {
    /// Underlying I/O failure (file not found, permission denied, etc.).
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// The file does not start with the expected `b"SCIRS2\0\0"` magic bytes.
    #[error("Invalid magic bytes — not a valid .scirs2 file")]
    InvalidMagic,

    /// The file was written by a newer major version of this library.
    #[error(
        "Unsupported version {0}.{1} (this library supports up to {major}.x)",
        major = VERSION_MAJOR
    )]
    UnsupportedVersion(u16, u16),

    /// CRC32 of the decompressed payload does not match the stored checksum.
    #[error("Checksum mismatch — file may be corrupted")]
    ChecksumMismatch,

    /// A compression or decompression operation failed.
    #[error("Compression error: {0}")]
    Compression(String),

    /// The payload type byte is not one of the defined [`PayloadType`] variants.
    #[error("Unknown payload type: {0}")]
    UnknownPayloadType(u8),

    /// The dtype stored in the file differs from the type `F` requested by the caller.
    #[error("Type mismatch: expected dtype {expected}, found {actual}")]
    TypeMismatch {
        /// Dtype ID the caller requested.
        expected: u8,
        /// Dtype ID found in the file.
        actual: u8,
    },
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, Array3};
    use std::io::Cursor;

    // ── header roundtrip ──────────────────────────────────────────────────────

    #[test]
    fn test_header_roundtrip_all_fields() {
        let original = Scirs2Header {
            version: (0, 3),
            payload_type: PayloadType::Array,
            compression: CompressionType::None,
            checksum: 0xDEAD_BEEF,
            payload_length: 1_234_567_890,
        };
        let bytes = original.to_bytes();
        assert_eq!(bytes.len(), HEADER_SIZE);

        let parsed = Scirs2Header::from_bytes(&bytes).expect("header parse failed");
        assert_eq!(parsed.version, original.version);
        assert_eq!(parsed.payload_type, original.payload_type);
        assert_eq!(parsed.compression, original.compression);
        assert_eq!(parsed.checksum, original.checksum);
        assert_eq!(parsed.payload_length, original.payload_length);
    }

    #[test]
    fn test_header_reserved_bytes_are_zero() {
        let header = Scirs2Header {
            version: (0, 3),
            payload_type: PayloadType::Custom,
            compression: CompressionType::None,
            checksum: 42,
            payload_length: 8,
        };
        let bytes = header.to_bytes();
        // Reserved bytes: 26..64
        for i in 26..64 {
            assert_eq!(bytes[i], 0, "reserved byte {} should be zero", i);
        }
    }

    #[test]
    fn test_invalid_magic_rejected() {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..8].copy_from_slice(b"BADMAGIC");
        assert!(
            matches!(
                Scirs2Header::from_bytes(&buf),
                Err(SerializationError::InvalidMagic)
            ),
            "should reject non-SCIRS2 magic"
        );
    }

    #[test]
    fn test_future_major_version_rejected() {
        let header = Scirs2Header {
            version: (255, 0),
            payload_type: PayloadType::Custom,
            compression: CompressionType::None,
            checksum: 0,
            payload_length: 0,
        };
        let bytes = header.to_bytes();
        assert!(
            matches!(
                Scirs2Header::from_bytes(&bytes),
                Err(SerializationError::UnsupportedVersion(255, 0))
            ),
            "should reject future major version"
        );
    }

    #[test]
    fn test_unknown_payload_type_rejected() {
        let header = Scirs2Header {
            version: (0, 3),
            payload_type: PayloadType::Custom,
            compression: CompressionType::None,
            checksum: 0,
            payload_length: 0,
        };
        let mut bytes = header.to_bytes();
        bytes[OFFSET_PAYLOAD_TYPE] = 99; // unknown type — not in PayloadType enum
        let result = Scirs2Header::from_bytes(&bytes);
        assert!(
            matches!(result, Err(SerializationError::UnknownPayloadType(99))),
            "should return UnknownPayloadType(99) for unknown payload type byte"
        );
    }

    #[test]
    fn test_payload_type_from_u8_all_variants() {
        assert!(matches!(PayloadType::from_u8(0), Ok(PayloadType::Array)));
        assert!(matches!(PayloadType::from_u8(1), Ok(PayloadType::Model)));
        assert!(matches!(PayloadType::from_u8(2), Ok(PayloadType::Stats)));
        assert!(matches!(PayloadType::from_u8(3), Ok(PayloadType::Custom)));
        assert!(matches!(
            PayloadType::from_u8(4),
            Err(SerializationError::UnknownPayloadType(4))
        ));
    }

    // ── writer / reader roundtrip ─────────────────────────────────────────────

    #[test]
    fn test_custom_payload_no_compression_roundtrip() {
        let payload = b"the quick brown fox jumps over the lazy dog";
        let mut buf = Vec::new();
        {
            let mut writer = Scirs2Writer::new(&mut buf);
            writer
                .write_payload(PayloadType::Custom, payload, CompressionType::None)
                .expect("write_payload failed");
        }

        let cursor = Cursor::new(&buf);
        let mut reader = Scirs2Reader::new(cursor).expect("Scirs2Reader::new failed");
        assert_eq!(reader.header.payload_type, PayloadType::Custom);
        assert_eq!(reader.header.compression, CompressionType::None);
        assert_eq!(reader.header.payload_length, payload.len() as u64);

        let out = reader.read_payload().expect("read_payload failed");
        assert_eq!(out.as_slice(), payload.as_slice());
    }

    #[test]
    fn test_empty_payload_roundtrip() {
        let payload: &[u8] = b"";
        let mut buf = Vec::new();
        {
            let mut writer = Scirs2Writer::new(&mut buf);
            writer
                .write_payload(PayloadType::Stats, payload, CompressionType::None)
                .expect("write empty payload failed");
        }
        let cursor = Cursor::new(&buf);
        let mut reader = Scirs2Reader::new(cursor).expect("reader init failed");
        let out = reader.read_payload().expect("read empty payload failed");
        assert!(out.is_empty());
    }

    #[test]
    fn test_verify_checksum_passes_for_intact_data() {
        let payload = b"integrity check payload 0xDEADBEEF";
        let mut buf = Vec::new();
        {
            let mut writer = Scirs2Writer::new(&mut buf);
            writer
                .write_payload(PayloadType::Stats, payload, CompressionType::None)
                .expect("write failed");
        }
        let cursor = Cursor::new(&buf);
        let mut reader = Scirs2Reader::new(cursor).expect("reader init failed");
        assert!(
            reader.verify_checksum().expect("checksum check failed"),
            "checksum should pass for intact data"
        );
    }

    #[test]
    fn test_verify_checksum_fails_on_bit_flip() {
        let payload = b"data that will be corrupted in transit";
        let mut buf = Vec::new();
        {
            let mut writer = Scirs2Writer::new(&mut buf);
            writer
                .write_payload(PayloadType::Custom, payload, CompressionType::None)
                .expect("write failed");
        }

        // Flip the last byte of the payload section
        let last = buf.len() - 1;
        buf[last] ^= 0xFF;

        let cursor = Cursor::new(&buf);
        let mut reader = Scirs2Reader::new(cursor).expect("reader init (corrupted) failed");
        assert!(
            !reader.verify_checksum().expect("checksum check errored"),
            "checksum should fail after bit flip"
        );
    }

    #[test]
    fn test_version_fields_in_file() {
        let payload = b"version test";
        let mut buf = Vec::new();
        let mut writer = Scirs2Writer::new(&mut buf);
        writer
            .write_payload(PayloadType::Custom, payload, CompressionType::None)
            .expect("write failed");

        let cursor = Cursor::new(&buf);
        let reader = Scirs2Reader::new(cursor).expect("reader failed");
        assert_eq!(reader.header.version, (VERSION_MAJOR, VERSION_MINOR));
    }

    // ── array encode / decode (in-memory) ────────────────────────────────────

    #[test]
    fn test_encode_decode_f32_1d() {
        let original =
            Array1::<f32>::from_vec(vec![1.0, 2.5, -3.0, f32::MAX, f32::MIN_POSITIVE]).into_dyn();
        let encoded = encode_array(&original);
        let decoded = decode_array::<f32>(&encoded).expect("f32 decode failed");
        assert_eq!(original, decoded, "f32 1d roundtrip mismatch");
    }

    #[test]
    fn test_encode_decode_f64_2d() {
        let original = Array2::<f64>::from_shape_vec(
            (4, 6),
            (0..24)
                .map(|i| i as f64 * std::f64::consts::PI / 12.0)
                .collect(),
        )
        .expect("shape error")
        .into_dyn();

        let encoded = encode_array(&original);
        let decoded = decode_array::<f64>(&encoded).expect("f64 2d decode failed");
        assert_eq!(original, decoded, "f64 2d roundtrip mismatch");
    }

    #[test]
    fn test_encode_decode_i32_3d() {
        let original =
            Array3::<i32>::from_shape_vec((2, 3, 4), (0..24).map(|i| i as i32 - 12).collect())
                .expect("shape error")
                .into_dyn();

        let encoded = encode_array(&original);
        let decoded = decode_array::<i32>(&encoded).expect("i32 3d decode failed");
        assert_eq!(original, decoded, "i32 3d roundtrip mismatch");
    }

    #[test]
    fn test_encode_decode_i64_1d() {
        let original = Array1::<i64>::from_vec(vec![i64::MIN, -1, 0, 1, i64::MAX]).into_dyn();
        let encoded = encode_array(&original);
        let decoded = decode_array::<i64>(&encoded).expect("i64 decode failed");
        assert_eq!(original, decoded, "i64 roundtrip mismatch");
    }

    #[test]
    fn test_encode_decode_u32() {
        let original = Array1::<u32>::from_vec(vec![0, 1, u32::MAX / 2, u32::MAX]).into_dyn();
        let encoded = encode_array(&original);
        let decoded = decode_array::<u32>(&encoded).expect("u32 decode failed");
        assert_eq!(original, decoded, "u32 roundtrip mismatch");
    }

    #[test]
    fn test_encode_decode_u64() {
        let original = Array1::<u64>::from_vec(vec![0, 1, u64::MAX / 2, u64::MAX]).into_dyn();
        let encoded = encode_array(&original);
        let decoded = decode_array::<u64>(&encoded).expect("u64 decode failed");
        assert_eq!(original, decoded, "u64 roundtrip mismatch");
    }

    #[test]
    fn test_dtype_mismatch_error() {
        let original = Array1::<f32>::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
        let encoded = encode_array(&original); // dtype_id = 1 (f32)
                                               // Try to decode as f64 (dtype_id = 2)
        let result = decode_array::<f64>(&encoded);
        assert!(
            matches!(
                result,
                Err(SerializationError::TypeMismatch {
                    expected: 2,
                    actual: 1
                })
            ),
            "expected TypeMismatch error"
        );
    }

    #[test]
    fn test_encode_zero_dimensional_array() {
        // 0-dimensional array (scalar)
        let original = Array::<f64, IxDyn>::from_elem(IxDyn(&[]), 42.0);
        let encoded = encode_array(&original);
        let decoded = decode_array::<f64>(&encoded).expect("0d decode failed");
        assert_eq!(original, decoded, "0d array roundtrip mismatch");
    }

    // ── save_array / load_array (file I/O) ────────────────────────────────────

    #[test]
    fn test_save_load_f32_no_compression() {
        let tmp = std::env::temp_dir().join("scirs2_test_f32_nocomp.scirs2");
        let original =
            Array2::<f32>::from_shape_vec((8, 8), (0..64).map(|i| i as f32 * 0.5 - 16.0).collect())
                .expect("shape error")
                .into_dyn();

        save_array(&tmp, &original, CompressionType::None).expect("save_array failed");
        let loaded = load_array::<f32>(&tmp).expect("load_array failed");

        assert_eq!(original, loaded, "f32 save/load mismatch");
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_save_load_f64_no_compression() {
        let tmp = std::env::temp_dir().join("scirs2_test_f64_nocomp.scirs2");
        let original = Array1::<f64>::linspace(0.0, 1.0, 500).into_dyn();

        save_array(&tmp, &original, CompressionType::None).expect("save_array f64 failed");
        let loaded = load_array::<f64>(&tmp).expect("load_array f64 failed");

        assert_eq!(original, loaded, "f64 save/load mismatch");
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_save_load_empty_array() {
        let tmp = std::env::temp_dir().join("scirs2_test_empty.scirs2");
        let original = Array1::<f64>::from_vec(vec![]).into_dyn();

        save_array(&tmp, &original, CompressionType::None).expect("save empty failed");
        let loaded = load_array::<f64>(&tmp).expect("load empty failed");

        assert_eq!(original, loaded, "empty array roundtrip mismatch");
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_save_load_large_f64_array() {
        let tmp = std::env::temp_dir().join("scirs2_test_large_f64.scirs2");
        let n = 100_000usize;
        let original =
            Array1::<f64>::from_iter((0..n).map(|i| (i as f64 / n as f64).sin())).into_dyn();

        save_array(&tmp, &original, CompressionType::None).expect("save large failed");
        let loaded = load_array::<f64>(&tmp).expect("load large failed");

        assert_eq!(
            original.shape(),
            loaded.shape(),
            "shape mismatch for large array"
        );
        for (a, b) in original.iter().zip(loaded.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "element mismatch in large array (bit-exact)"
            );
        }
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_save_load_3d_i32_array() {
        let tmp = std::env::temp_dir().join("scirs2_test_3d_i32.scirs2");
        let original =
            Array3::<i32>::from_shape_fn((5, 6, 7), |(i, j, k)| (i * 100 + j * 10 + k) as i32)
                .into_dyn();

        save_array(&tmp, &original, CompressionType::None).expect("save 3d i32 failed");
        let loaded = load_array::<i32>(&tmp).expect("load 3d i32 failed");

        assert_eq!(original, loaded, "3d i32 save/load mismatch");
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_wrong_payload_type_error() {
        let tmp = std::env::temp_dir().join("scirs2_test_wrong_type.scirs2");
        {
            let file = std::fs::File::create(&tmp).expect("create failed");
            let mut writer = Scirs2Writer::new(BufWriter::new(file));
            writer
                .write_payload(
                    PayloadType::Custom,
                    b"definitely not an array",
                    CompressionType::None,
                )
                .expect("write failed");
        }
        let result = load_array::<f64>(&tmp);
        assert!(
            result.is_err(),
            "load_array should fail when payload type is not Array"
        );
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_file_not_found_error() {
        let result = load_array::<f32>(Path::new("/nonexistent/path/does_not_exist.scirs2"));
        assert!(
            matches!(result, Err(SerializationError::Io(_))),
            "should return Io error for missing file"
        );
    }

    #[test]
    fn test_checksum_is_stored_in_file() {
        let tmp = std::env::temp_dir().join("scirs2_test_checksum_stored.scirs2");
        let original = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
        save_array(&tmp, &original, CompressionType::None).expect("save failed");

        // Read and verify through Scirs2Reader
        let file = std::fs::File::open(&tmp).expect("open failed");
        let mut reader = Scirs2Reader::new(BufReader::new(file)).expect("reader failed");
        let ok = reader.verify_checksum().expect("checksum check failed");
        assert!(ok, "checksum should pass for freshly saved file");

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_lz4_compression_roundtrip() {
        let tmp = std::env::temp_dir().join("scirs2_test_lz4.scirs2");
        // Highly compressible data: a constant array
        let original = Array1::<f32>::from_elem(1000, 3.14159_f32).into_dyn();

        let result = save_array(&tmp, &original, CompressionType::Lz4);
        match result {
            Ok(()) => {
                let loaded = load_array::<f32>(&tmp).expect("load lz4 failed");
                assert_eq!(original, loaded, "lz4 roundtrip mismatch");
            }
            Err(SerializationError::Compression(_)) => {
                // LZ4 not available in this configuration — skip
                eprintln!("LZ4 not available, skipping lz4 test");
            }
            Err(e) => panic!("unexpected error during lz4 test: {}", e),
        }
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_zstd_compression_roundtrip() {
        let tmp = std::env::temp_dir().join("scirs2_test_zstd.scirs2");
        // Highly compressible data
        let original = Array2::<f64>::zeros((100, 100)).into_dyn();

        let result = save_array(&tmp, &original, CompressionType::Zstd);
        match result {
            Ok(()) => {
                let loaded = load_array::<f64>(&tmp).expect("load zstd failed");
                assert_eq!(original, loaded, "zstd roundtrip mismatch");
            }
            Err(SerializationError::Compression(_)) => {
                eprintln!("Zstd not available, skipping zstd test");
            }
            Err(e) => panic!("unexpected error during zstd test: {}", e),
        }
        std::fs::remove_file(&tmp).ok();
    }
}
