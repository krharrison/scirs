//! Types and utilities for NumPy binary format support.

use crate::error::{IoError, Result};

/// NumPy magic string
pub const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";

/// NPY format major version
pub const NPY_MAJOR_VERSION: u8 = 1;

/// NPY format minor version
pub const NPY_MINOR_VERSION: u8 = 0;

/// NumPy data type descriptor
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpyDtype {
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
}

impl NpyDtype {
    /// Size in bytes of one element
    pub fn element_size(&self) -> usize {
        match self {
            NpyDtype::Float32 => 4,
            NpyDtype::Float64 => 8,
            NpyDtype::Int32 => 4,
            NpyDtype::Int64 => 8,
        }
    }

    /// NumPy dtype string for little-endian
    pub fn npy_str_le(&self) -> &'static str {
        match self {
            NpyDtype::Float32 => "<f4",
            NpyDtype::Float64 => "<f8",
            NpyDtype::Int32 => "<i4",
            NpyDtype::Int64 => "<i8",
        }
    }

    /// NumPy dtype string for big-endian
    pub fn npy_str_be(&self) -> &'static str {
        match self {
            NpyDtype::Float32 => ">f4",
            NpyDtype::Float64 => ">f8",
            NpyDtype::Int32 => ">i4",
            NpyDtype::Int64 => ">i8",
        }
    }

    /// Parse dtype from NumPy descriptor string
    pub fn from_descr(descr: &str) -> Result<(Self, ByteOrder)> {
        let descr = descr.trim().trim_matches('\'').trim_matches('"');
        if descr.len() < 3 {
            return Err(IoError::FormatError(format!(
                "Invalid dtype descriptor: '{}'",
                descr
            )));
        }

        let endian_char = descr.as_bytes()[0];
        let type_char = descr.as_bytes()[1];
        let size_str = &descr[2..];

        let byte_order = match endian_char {
            b'<' | b'=' => ByteOrder::LittleEndian,
            b'>' => ByteOrder::BigEndian,
            b'|' => ByteOrder::NotApplicable,
            _ => {
                return Err(IoError::FormatError(format!(
                    "Unknown endian prefix: '{}'",
                    endian_char as char
                )))
            }
        };

        let size: usize = size_str
            .parse()
            .map_err(|_| IoError::FormatError(format!("Invalid dtype size: '{}'", size_str)))?;

        let dtype = match (type_char, size) {
            (b'f', 4) => NpyDtype::Float32,
            (b'f', 8) => NpyDtype::Float64,
            (b'i', 4) => NpyDtype::Int32,
            (b'i', 8) => NpyDtype::Int64,
            _ => {
                return Err(IoError::FormatError(format!(
                    "Unsupported dtype: type='{}', size={}",
                    type_char as char, size
                )))
            }
        };

        Ok((dtype, byte_order))
    }
}

/// Byte ordering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    /// Little-endian
    LittleEndian,
    /// Big-endian
    BigEndian,
    /// Not applicable (single-byte types)
    NotApplicable,
}

/// Parsed header from a .npy file
#[derive(Debug, Clone)]
pub struct NpyHeader {
    /// Data type
    pub dtype: NpyDtype,
    /// Byte order
    pub byte_order: ByteOrder,
    /// Whether the data is in Fortran (column-major) order
    pub fortran_order: bool,
    /// Shape of the array
    pub shape: Vec<usize>,
}

impl NpyHeader {
    /// Total number of elements
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Serialize as NumPy header dict string
    pub fn to_header_string(&self) -> String {
        let descr = if cfg!(target_endian = "little") {
            self.dtype.npy_str_le()
        } else {
            self.dtype.npy_str_be()
        };

        let fortran_str = if self.fortran_order { "True" } else { "False" };

        let shape_str = if self.shape.len() == 1 {
            format!("({},)", self.shape[0])
        } else {
            let parts: Vec<String> = self.shape.iter().map(|s| s.to_string()).collect();
            format!("({})", parts.join(", "))
        };

        format!(
            "{{'descr': '{}', 'fortran_order': {}, 'shape': {}, }}",
            descr, fortran_str, shape_str
        )
    }
}

/// Data read from a .npy file
#[derive(Debug, Clone)]
pub enum NpyArray {
    /// f32 data with shape
    Float32 {
        /// Flat data
        data: Vec<f32>,
        /// Shape
        shape: Vec<usize>,
    },
    /// f64 data with shape
    Float64 {
        /// Flat data
        data: Vec<f64>,
        /// Shape
        shape: Vec<usize>,
    },
    /// i32 data with shape
    Int32 {
        /// Flat data
        data: Vec<i32>,
        /// Shape
        shape: Vec<usize>,
    },
    /// i64 data with shape
    Int64 {
        /// Flat data
        data: Vec<i64>,
        /// Shape
        shape: Vec<usize>,
    },
}

impl NpyArray {
    /// Get the shape
    pub fn shape(&self) -> &[usize] {
        match self {
            NpyArray::Float32 { shape, .. } => shape,
            NpyArray::Float64 { shape, .. } => shape,
            NpyArray::Int32 { shape, .. } => shape,
            NpyArray::Int64 { shape, .. } => shape,
        }
    }

    /// Get the dtype
    pub fn dtype(&self) -> NpyDtype {
        match self {
            NpyArray::Float32 { .. } => NpyDtype::Float32,
            NpyArray::Float64 { .. } => NpyDtype::Float64,
            NpyArray::Int32 { .. } => NpyDtype::Int32,
            NpyArray::Int64 { .. } => NpyDtype::Int64,
        }
    }

    /// Total number of elements
    pub fn num_elements(&self) -> usize {
        self.shape().iter().product()
    }

    /// Try to get f64 data
    pub fn as_f64(&self) -> Result<&[f64]> {
        match self {
            NpyArray::Float64 { data, .. } => Ok(data),
            _ => Err(IoError::ConversionError(format!(
                "Array is {:?}, not Float64",
                self.dtype()
            ))),
        }
    }

    /// Try to get f32 data
    pub fn as_f32(&self) -> Result<&[f32]> {
        match self {
            NpyArray::Float32 { data, .. } => Ok(data),
            _ => Err(IoError::ConversionError(format!(
                "Array is {:?}, not Float32",
                self.dtype()
            ))),
        }
    }

    /// Try to get i32 data
    pub fn as_i32(&self) -> Result<&[i32]> {
        match self {
            NpyArray::Int32 { data, .. } => Ok(data),
            _ => Err(IoError::ConversionError(format!(
                "Array is {:?}, not Int32",
                self.dtype()
            ))),
        }
    }

    /// Try to get i64 data
    pub fn as_i64(&self) -> Result<&[i64]> {
        match self {
            NpyArray::Int64 { data, .. } => Ok(data),
            _ => Err(IoError::ConversionError(format!(
                "Array is {:?}, not Int64",
                self.dtype()
            ))),
        }
    }
}

/// Parse the header dict from raw header string
pub fn parse_header_dict(header_str: &str) -> Result<NpyHeader> {
    let header_str = header_str
        .trim()
        .trim_end_matches('\n')
        .trim_end_matches('\0');

    // Extract 'descr' value
    let descr = extract_dict_value(header_str, "descr")?;
    let (dtype, byte_order) = NpyDtype::from_descr(&descr)?;

    // Extract 'fortran_order'
    let fortran_str = extract_dict_value(header_str, "fortran_order")?;
    let fortran_order = fortran_str.trim() == "True";

    // Extract 'shape'
    let shape_str = extract_dict_value(header_str, "shape")?;
    let shape = parse_shape(&shape_str)?;

    Ok(NpyHeader {
        dtype,
        byte_order,
        fortran_order,
        shape,
    })
}

/// Extract a value from a Python dict string by key
fn extract_dict_value(dict_str: &str, key: &str) -> Result<String> {
    let search = format!("'{}': ", key);
    let pos = dict_str.find(&search).or_else(|| {
        let alt_search = format!("\"{}\":", key);
        dict_str.find(&alt_search)
    });

    let start = match pos {
        Some(p) => p + search.len(),
        None => {
            // Try alternate format
            let alt = format!("'{}':", key);
            match dict_str.find(&alt) {
                Some(p) => p + alt.len(),
                None => {
                    return Err(IoError::FormatError(format!(
                        "Key '{}' not found in header: {}",
                        key, dict_str
                    )))
                }
            }
        }
    };

    let remaining = dict_str[start..].trim_start();

    // Handle different value types
    if remaining.starts_with('\'') || remaining.starts_with('"') {
        let quote = remaining.as_bytes()[0];
        let end = remaining[1..]
            .find(|c: char| c as u8 == quote)
            .ok_or_else(|| {
                IoError::FormatError(format!("Unterminated string for key '{}'", key))
            })?;
        Ok(remaining[1..end + 1].to_string())
    } else if remaining.starts_with('(') {
        let end = remaining
            .find(')')
            .ok_or_else(|| IoError::FormatError(format!("Unterminated tuple for key '{}'", key)))?;
        Ok(remaining[..end + 1].to_string())
    } else {
        // Boolean or other: read until comma or '}'
        let end = remaining.find([',', '}']).unwrap_or(remaining.len());
        Ok(remaining[..end].trim().to_string())
    }
}

/// Parse a Python tuple shape string like "(3, 4)" or "(5,)"
fn parse_shape(shape_str: &str) -> Result<Vec<usize>> {
    let inner = shape_str
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')');

    if inner.is_empty() {
        return Ok(vec![]); // scalar
    }

    let mut shape = Vec::new();
    for part in inner.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let dim: usize = part
            .parse()
            .map_err(|_| IoError::FormatError(format!("Invalid shape dimension: '{}'", part)))?;
        shape.push(dim);
    }

    Ok(shape)
}
