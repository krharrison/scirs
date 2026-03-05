//! Pure Rust NetCDF Classic format reader/writer
//!
//! Implements the NetCDF Classic (version 1) binary format without any C dependencies.
//! The NetCDF Classic format stores array-oriented scientific data with:
//! - Named dimensions (including unlimited/record dimensions)
//! - Typed variables with associated dimensions
//! - Global and per-variable attributes
//!
//! # Supported data types
//!
//! | Type   | NetCDF name | Bytes |
//! |--------|-------------|-------|
//! | `i8`   | NC_BYTE     | 1     |
//! | `i16`  | NC_SHORT    | 2     |
//! | `i32`  | NC_INT      | 4     |
//! | `f32`  | NC_FLOAT    | 4     |
//! | `f64`  | NC_DOUBLE   | 8     |
//! | `char` | NC_CHAR     | 1     |
//!
//! # File format
//!
//! NetCDF Classic files are big-endian and use a structured binary layout:
//! - Magic bytes: `CDF\x01`
//! - Number of records (unlimited dimension length)
//! - Dimension list
//! - Global attribute list
//! - Variable list (header + data offsets)
//! - Variable data (contiguous or record-based)
//!
//! # Examples
//!
//! ```rust
//! use scirs2_io::netcdf_lite::{NcFile, NcDataType, NcValue};
//!
//! // Create a new NetCDF file in memory
//! let mut nc = NcFile::new();
//! nc.add_dimension("x", Some(3)).expect("add dim failed");
//! nc.add_dimension("y", Some(4)).expect("add dim failed");
//! nc.add_variable("temperature", NcDataType::Float, &["x", "y"])
//!     .expect("add var failed");
//! nc.add_global_attribute("title", NcValue::Text("Test Data".to_string()))
//!     .expect("add attr failed");
//!
//! // Write and read back
//! let mut buf = Vec::new();
//! nc.write_to(&mut buf).expect("write failed");
//!
//! let loaded = NcFile::read_from(&mut std::io::Cursor::new(&buf)).expect("read failed");
//! assert_eq!(loaded.dimensions().len(), 2);
//! ```

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};

use crate::error::{IoError, Result};

// =============================================================================
// Constants - NetCDF Classic format magic numbers and tags
// =============================================================================

/// NetCDF Classic format magic bytes
const NC_MAGIC: &[u8; 4] = b"CDF\x01";

/// Tag for dimension list
const NC_DIMENSION: u32 = 0x0000_000A;
/// Tag for variable list
const NC_VARIABLE: u32 = 0x0000_000B;
/// Tag for attribute list
const NC_ATTRIBUTE: u32 = 0x0000_000C;

/// Tag indicating absent (empty) list
const NC_ABSENT: u32 = 0x0000_0000;

// NetCDF data type codes
const NC_BYTE: u32 = 1;
const NC_CHAR: u32 = 2;
const NC_SHORT: u32 = 3;
const NC_INT: u32 = 4;
const NC_FLOAT: u32 = 5;
const NC_DOUBLE: u32 = 6;

// =============================================================================
// Data types
// =============================================================================

/// NetCDF data type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcDataType {
    /// Signed 8-bit integer (NC_BYTE)
    Byte,
    /// Character / text (NC_CHAR)
    Char,
    /// Signed 16-bit integer (NC_SHORT)
    Short,
    /// Signed 32-bit integer (NC_INT)
    Int,
    /// 32-bit IEEE float (NC_FLOAT)
    Float,
    /// 64-bit IEEE float (NC_DOUBLE)
    Double,
}

impl NcDataType {
    /// Size in bytes for one element of this type
    pub fn element_size(self) -> usize {
        match self {
            NcDataType::Byte | NcDataType::Char => 1,
            NcDataType::Short => 2,
            NcDataType::Int | NcDataType::Float => 4,
            NcDataType::Double => 8,
        }
    }

    fn to_nc_type(self) -> u32 {
        match self {
            NcDataType::Byte => NC_BYTE,
            NcDataType::Char => NC_CHAR,
            NcDataType::Short => NC_SHORT,
            NcDataType::Int => NC_INT,
            NcDataType::Float => NC_FLOAT,
            NcDataType::Double => NC_DOUBLE,
        }
    }

    fn from_nc_type(code: u32) -> Result<Self> {
        match code {
            NC_BYTE => Ok(NcDataType::Byte),
            NC_CHAR => Ok(NcDataType::Char),
            NC_SHORT => Ok(NcDataType::Short),
            NC_INT => Ok(NcDataType::Int),
            NC_FLOAT => Ok(NcDataType::Float),
            NC_DOUBLE => Ok(NcDataType::Double),
            _ => Err(IoError::FormatError(format!(
                "Unknown NetCDF data type code: {}",
                code
            ))),
        }
    }
}

// =============================================================================
// Attribute values
// =============================================================================

/// A NetCDF attribute value
#[derive(Debug, Clone, PartialEq)]
pub enum NcValue {
    /// Array of signed bytes
    Bytes(Vec<i8>),
    /// Text string (NC_CHAR array)
    Text(String),
    /// Array of signed 16-bit integers
    Shorts(Vec<i16>),
    /// Array of signed 32-bit integers
    Ints(Vec<i32>),
    /// Array of 32-bit floats
    Floats(Vec<f32>),
    /// Array of 64-bit floats
    Doubles(Vec<f64>),
}

impl NcValue {
    fn nc_type(&self) -> u32 {
        match self {
            NcValue::Bytes(_) => NC_BYTE,
            NcValue::Text(_) => NC_CHAR,
            NcValue::Shorts(_) => NC_SHORT,
            NcValue::Ints(_) => NC_INT,
            NcValue::Floats(_) => NC_FLOAT,
            NcValue::Doubles(_) => NC_DOUBLE,
        }
    }

    fn element_count(&self) -> usize {
        match self {
            NcValue::Bytes(v) => v.len(),
            NcValue::Text(s) => s.len(),
            NcValue::Shorts(v) => v.len(),
            NcValue::Ints(v) => v.len(),
            NcValue::Floats(v) => v.len(),
            NcValue::Doubles(v) => v.len(),
        }
    }
}

// =============================================================================
// Dimension
// =============================================================================

/// A named dimension
#[derive(Debug, Clone)]
pub struct NcDimension {
    /// Dimension name
    pub name: String,
    /// Dimension length (None = unlimited/record dimension)
    pub length: Option<usize>,
    /// Whether this is the unlimited (record) dimension
    pub is_unlimited: bool,
}

// =============================================================================
// Variable metadata
// =============================================================================

/// Metadata for a variable stored in the file
#[derive(Debug, Clone)]
pub struct NcVariable {
    /// Variable name
    pub name: String,
    /// Data type
    pub data_type: NcDataType,
    /// Dimension indices (referencing the file's dimension list)
    pub dim_indices: Vec<usize>,
    /// Variable-level attributes
    pub attributes: Vec<(String, NcValue)>,
    /// Raw data bytes (stored in big-endian)
    pub(crate) data: Vec<u8>,
}

impl NcVariable {
    /// Get the shape of this variable based on dimension lengths
    pub fn shape(&self, dimensions: &[NcDimension], num_records: usize) -> Vec<usize> {
        self.dim_indices
            .iter()
            .map(|&idx| {
                if dimensions[idx].is_unlimited {
                    num_records
                } else {
                    dimensions[idx].length.unwrap_or(0)
                }
            })
            .collect()
    }

    /// Get total number of elements
    pub fn total_elements(&self, dimensions: &[NcDimension], num_records: usize) -> usize {
        let shape = self.shape(dimensions, num_records);
        if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        }
    }

    /// Read data as f64 values
    pub fn as_f64(&self, dimensions: &[NcDimension], num_records: usize) -> Result<Vec<f64>> {
        let n = self.total_elements(dimensions, num_records);
        let mut cursor = std::io::Cursor::new(&self.data);
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            let val = match self.data_type {
                NcDataType::Byte => cursor
                    .read_i8()
                    .map_err(|e| IoError::FormatError(e.to_string()))?
                    as f64,
                NcDataType::Short => cursor
                    .read_i16::<BigEndian>()
                    .map_err(|e| IoError::FormatError(e.to_string()))?
                    as f64,
                NcDataType::Int => cursor
                    .read_i32::<BigEndian>()
                    .map_err(|e| IoError::FormatError(e.to_string()))?
                    as f64,
                NcDataType::Float => cursor
                    .read_f32::<BigEndian>()
                    .map_err(|e| IoError::FormatError(e.to_string()))?
                    as f64,
                NcDataType::Double => cursor
                    .read_f64::<BigEndian>()
                    .map_err(|e| IoError::FormatError(e.to_string()))?,
                NcDataType::Char => cursor
                    .read_u8()
                    .map_err(|e| IoError::FormatError(e.to_string()))?
                    as f64,
            };
            result.push(val);
        }
        Ok(result)
    }

    /// Read data as f32 values
    pub fn as_f32(&self, dimensions: &[NcDimension], num_records: usize) -> Result<Vec<f32>> {
        self.as_f64(dimensions, num_records)
            .map(|v| v.into_iter().map(|x| x as f32).collect())
    }

    /// Read data as i32 values
    pub fn as_i32(&self, dimensions: &[NcDimension], num_records: usize) -> Result<Vec<i32>> {
        self.as_f64(dimensions, num_records)
            .map(|v| v.into_iter().map(|x| x as i32).collect())
    }

    /// Read data as text (for NC_CHAR variables)
    pub fn as_text(&self) -> Result<String> {
        if self.data_type != NcDataType::Char {
            return Err(IoError::ConversionError(
                "Variable is not NC_CHAR type".to_string(),
            ));
        }
        let s = String::from_utf8_lossy(&self.data);
        Ok(s.trim_end_matches('\0').to_string())
    }
}

// =============================================================================
// NcFile - the main file structure
// =============================================================================

/// A NetCDF Classic format file in memory
///
/// Holds all dimensions, variables (with data), and global attributes.
/// Can be read from and written to any `Read`/`Write` stream.
#[derive(Debug, Clone)]
pub struct NcFile {
    /// Dimensions
    dims: Vec<NcDimension>,
    /// Global attributes
    global_attrs: Vec<(String, NcValue)>,
    /// Variables (with embedded data)
    vars: Vec<NcVariable>,
    /// Number of records (unlimited dimension current length)
    num_records: usize,
}

impl NcFile {
    /// Create a new empty NetCDF file
    pub fn new() -> Self {
        NcFile {
            dims: Vec::new(),
            global_attrs: Vec::new(),
            vars: Vec::new(),
            num_records: 0,
        }
    }

    /// Add a dimension
    ///
    /// - `length = Some(n)` for a fixed dimension of size n
    /// - `length = None` for the unlimited (record) dimension
    ///
    /// Only one unlimited dimension is allowed per file.
    pub fn add_dimension(&mut self, name: &str, length: Option<usize>) -> Result<()> {
        let is_unlimited = length.is_none();

        // Check for duplicate name
        if self.dims.iter().any(|d| d.name == name) {
            return Err(IoError::FormatError(format!(
                "Dimension '{}' already exists",
                name
            )));
        }

        // Only one unlimited dimension allowed
        if is_unlimited && self.dims.iter().any(|d| d.is_unlimited) {
            return Err(IoError::FormatError(
                "Only one unlimited dimension is allowed in NetCDF Classic format".to_string(),
            ));
        }

        self.dims.push(NcDimension {
            name: name.to_string(),
            length,
            is_unlimited,
        });

        Ok(())
    }

    /// Add a variable
    ///
    /// Dimension names must already be defined via `add_dimension`.
    pub fn add_variable(
        &mut self,
        name: &str,
        data_type: NcDataType,
        dim_names: &[&str],
    ) -> Result<()> {
        // Check for duplicate
        if self.vars.iter().any(|v| v.name == name) {
            return Err(IoError::FormatError(format!(
                "Variable '{}' already exists",
                name
            )));
        }

        // Resolve dimension indices
        let mut dim_indices = Vec::with_capacity(dim_names.len());
        for &dname in dim_names {
            let idx = self
                .dims
                .iter()
                .position(|d| d.name == dname)
                .ok_or_else(|| IoError::FormatError(format!("Dimension '{}' not found", dname)))?;
            dim_indices.push(idx);
        }

        self.vars.push(NcVariable {
            name: name.to_string(),
            data_type,
            dim_indices,
            attributes: Vec::new(),
            data: Vec::new(),
        });

        Ok(())
    }

    /// Add a global attribute
    pub fn add_global_attribute(&mut self, name: &str, value: NcValue) -> Result<()> {
        // Replace if exists
        if let Some(pos) = self.global_attrs.iter().position(|(n, _)| n == name) {
            self.global_attrs[pos] = (name.to_string(), value);
        } else {
            self.global_attrs.push((name.to_string(), value));
        }
        Ok(())
    }

    /// Add a variable attribute
    pub fn add_variable_attribute(
        &mut self,
        var_name: &str,
        attr_name: &str,
        value: NcValue,
    ) -> Result<()> {
        let var = self
            .vars
            .iter_mut()
            .find(|v| v.name == var_name)
            .ok_or_else(|| IoError::NotFound(format!("Variable '{}' not found", var_name)))?;

        if let Some(pos) = var.attributes.iter().position(|(n, _)| n == attr_name) {
            var.attributes[pos] = (attr_name.to_string(), value);
        } else {
            var.attributes.push((attr_name.to_string(), value));
        }
        Ok(())
    }

    /// Set variable data from f64 slice
    pub fn set_variable_f64(&mut self, var_name: &str, data: &[f64]) -> Result<()> {
        let var = self
            .vars
            .iter_mut()
            .find(|v| v.name == var_name)
            .ok_or_else(|| IoError::NotFound(format!("Variable '{}' not found", var_name)))?;

        let mut buf = Vec::with_capacity(data.len() * var.data_type.element_size());
        for &val in data {
            match var.data_type {
                NcDataType::Byte => buf
                    .write_i8(val as i8)
                    .map_err(|e| IoError::FileError(e.to_string()))?,
                NcDataType::Short => buf
                    .write_i16::<BigEndian>(val as i16)
                    .map_err(|e| IoError::FileError(e.to_string()))?,
                NcDataType::Int => buf
                    .write_i32::<BigEndian>(val as i32)
                    .map_err(|e| IoError::FileError(e.to_string()))?,
                NcDataType::Float => buf
                    .write_f32::<BigEndian>(val as f32)
                    .map_err(|e| IoError::FileError(e.to_string()))?,
                NcDataType::Double => buf
                    .write_f64::<BigEndian>(val)
                    .map_err(|e| IoError::FileError(e.to_string()))?,
                NcDataType::Char => buf
                    .write_u8(val as u8)
                    .map_err(|e| IoError::FileError(e.to_string()))?,
            }
        }
        var.data = buf;

        // Update num_records if variable uses unlimited dimension
        self.update_num_records();
        Ok(())
    }

    /// Set variable data from f32 slice
    pub fn set_variable_f32(&mut self, var_name: &str, data: &[f32]) -> Result<()> {
        let f64_data: Vec<f64> = data.iter().map(|&x| x as f64).collect();
        self.set_variable_f64(var_name, &f64_data)
    }

    /// Set variable data from i32 slice
    pub fn set_variable_i32(&mut self, var_name: &str, data: &[i32]) -> Result<()> {
        let f64_data: Vec<f64> = data.iter().map(|&x| x as f64).collect();
        self.set_variable_f64(var_name, &f64_data)
    }

    /// Set variable data as text (for NC_CHAR variables)
    pub fn set_variable_text(&mut self, var_name: &str, text: &str) -> Result<()> {
        let var = self
            .vars
            .iter_mut()
            .find(|v| v.name == var_name)
            .ok_or_else(|| IoError::NotFound(format!("Variable '{}' not found", var_name)))?;

        if var.data_type != NcDataType::Char {
            return Err(IoError::ConversionError(format!(
                "Variable '{}' is not NC_CHAR type",
                var_name
            )));
        }

        var.data = text.as_bytes().to_vec();
        self.update_num_records();
        Ok(())
    }

    /// Get dimensions
    pub fn dimensions(&self) -> &[NcDimension] {
        &self.dims
    }

    /// Get global attributes
    pub fn global_attributes(&self) -> &[(String, NcValue)] {
        &self.global_attrs
    }

    /// Get variable names
    pub fn variable_names(&self) -> Vec<&str> {
        self.vars.iter().map(|v| v.name.as_str()).collect()
    }

    /// Get a variable by name
    pub fn variable(&self, name: &str) -> Result<&NcVariable> {
        self.vars
            .iter()
            .find(|v| v.name == name)
            .ok_or_else(|| IoError::NotFound(format!("Variable '{}' not found", name)))
    }

    /// Get number of records (unlimited dimension length)
    pub fn num_records(&self) -> usize {
        self.num_records
    }

    /// Update num_records from current variable data
    fn update_num_records(&mut self) {
        for var in &self.vars {
            if var.dim_indices.is_empty() {
                continue;
            }
            if self.dims[var.dim_indices[0]].is_unlimited && !var.data.is_empty() {
                let elem_size = var.data_type.element_size();
                let per_record_elements: usize = var.dim_indices[1..]
                    .iter()
                    .map(|&idx| self.dims[idx].length.unwrap_or(1))
                    .product::<usize>()
                    .max(1);
                let total_elements = var.data.len() / elem_size;
                let records = if per_record_elements > 0 {
                    total_elements / per_record_elements
                } else {
                    0
                };
                if records > self.num_records {
                    self.num_records = records;
                }
            }
        }
    }

    // =========================================================================
    // Writing
    // =========================================================================

    /// Write the NetCDF file to a writer
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Magic + numrecs
        writer
            .write_all(NC_MAGIC)
            .map_err(|e| IoError::FileError(e.to_string()))?;
        writer
            .write_u32::<BigEndian>(self.num_records as u32)
            .map_err(|e| IoError::FileError(e.to_string()))?;

        // Dimensions
        self.write_dim_list(writer)?;

        // Global attributes
        self.write_attr_list(writer, &self.global_attrs)?;

        // Variables header + data
        self.write_var_list(writer)?;

        Ok(())
    }

    /// Write to a file path
    pub fn write_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let file = std::fs::File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
        let mut writer = std::io::BufWriter::new(file);
        self.write_to(&mut writer)?;
        writer
            .flush()
            .map_err(|e| IoError::FileError(e.to_string()))?;
        Ok(())
    }

    fn write_dim_list<W: Write>(&self, w: &mut W) -> Result<()> {
        if self.dims.is_empty() {
            w.write_u32::<BigEndian>(NC_ABSENT)
                .map_err(|e| IoError::FileError(e.to_string()))?;
            w.write_u32::<BigEndian>(0)
                .map_err(|e| IoError::FileError(e.to_string()))?;
            return Ok(());
        }

        w.write_u32::<BigEndian>(NC_DIMENSION)
            .map_err(|e| IoError::FileError(e.to_string()))?;
        w.write_u32::<BigEndian>(self.dims.len() as u32)
            .map_err(|e| IoError::FileError(e.to_string()))?;

        for dim in &self.dims {
            write_name(w, &dim.name)?;
            let len = if dim.is_unlimited {
                0u32
            } else {
                dim.length.unwrap_or(0) as u32
            };
            w.write_u32::<BigEndian>(len)
                .map_err(|e| IoError::FileError(e.to_string()))?;
        }
        Ok(())
    }

    fn write_attr_list<W: Write>(&self, w: &mut W, attrs: &[(String, NcValue)]) -> Result<()> {
        if attrs.is_empty() {
            w.write_u32::<BigEndian>(NC_ABSENT)
                .map_err(|e| IoError::FileError(e.to_string()))?;
            w.write_u32::<BigEndian>(0)
                .map_err(|e| IoError::FileError(e.to_string()))?;
            return Ok(());
        }

        w.write_u32::<BigEndian>(NC_ATTRIBUTE)
            .map_err(|e| IoError::FileError(e.to_string()))?;
        w.write_u32::<BigEndian>(attrs.len() as u32)
            .map_err(|e| IoError::FileError(e.to_string()))?;

        for (name, value) in attrs {
            write_name(w, name)?;
            write_attr_value(w, value)?;
        }
        Ok(())
    }

    fn write_var_list<W: Write>(&self, w: &mut W) -> Result<()> {
        if self.vars.is_empty() {
            w.write_u32::<BigEndian>(NC_ABSENT)
                .map_err(|e| IoError::FileError(e.to_string()))?;
            w.write_u32::<BigEndian>(0)
                .map_err(|e| IoError::FileError(e.to_string()))?;
            return Ok(());
        }

        w.write_u32::<BigEndian>(NC_VARIABLE)
            .map_err(|e| IoError::FileError(e.to_string()))?;
        w.write_u32::<BigEndian>(self.vars.len() as u32)
            .map_err(|e| IoError::FileError(e.to_string()))?;

        // First pass: calculate header size to determine data offsets
        // For simplicity, we embed data right after the header section.
        // We pre-compute data sizes for the offset fields.
        let mut data_sizes: Vec<usize> = Vec::with_capacity(self.vars.len());
        for var in &self.vars {
            let raw_size = var.data.len();
            let padded = pad_to_4(raw_size);
            data_sizes.push(padded);
        }

        // We need to know the current offset after writing all variable headers.
        // But we need the offset *within the file* for the `begin` field.
        // Since we write variable headers followed immediately by data,
        // we calculate header sizes.
        let mut header_total = 0usize;
        for var in &self.vars {
            // name
            header_total += 4 + pad_to_4(var.name.len());
            // ndims + dim_ids
            header_total += 4 + var.dim_indices.len() * 4;
            // vatt_list
            header_total += self.attr_list_size(&var.attributes);
            // nc_type + vsize + begin
            header_total += 4 + 4 + 4;
        }

        // Current position = 8 (magic + numrecs) + dim_list_size + gatt_list_size + 8 (var tag + count) + header_total
        // But since we write data inline, we just need offsets relative to file start.
        // For correctness, we calculate the absolute start of the first var data.

        // Calculate sizes for preceding sections
        let dim_list_size = self.dim_list_size();
        let gatt_list_size = self.attr_list_size(&self.global_attrs);
        let file_header_size = 8 + dim_list_size + gatt_list_size + 8 + header_total;

        let mut current_data_offset = file_header_size;

        // Write variable headers
        for (i, var) in self.vars.iter().enumerate() {
            write_name(w, &var.name)?;

            // Dimension ID list
            w.write_u32::<BigEndian>(var.dim_indices.len() as u32)
                .map_err(|e| IoError::FileError(e.to_string()))?;
            for &dim_idx in &var.dim_indices {
                w.write_u32::<BigEndian>(dim_idx as u32)
                    .map_err(|e| IoError::FileError(e.to_string()))?;
            }

            // Variable attributes
            self.write_attr_list(w, &var.attributes)?;

            // nc_type
            w.write_u32::<BigEndian>(var.data_type.to_nc_type())
                .map_err(|e| IoError::FileError(e.to_string()))?;

            // vsize (padded data size)
            w.write_u32::<BigEndian>(data_sizes[i] as u32)
                .map_err(|e| IoError::FileError(e.to_string()))?;

            // begin (offset to data)
            w.write_u32::<BigEndian>(current_data_offset as u32)
                .map_err(|e| IoError::FileError(e.to_string()))?;

            current_data_offset += data_sizes[i];
        }

        // Write variable data
        for (i, var) in self.vars.iter().enumerate() {
            w.write_all(&var.data)
                .map_err(|e| IoError::FileError(e.to_string()))?;

            // Pad to 4-byte boundary
            let padding_needed = data_sizes[i] - var.data.len();
            if padding_needed > 0 {
                let pad = vec![0u8; padding_needed];
                w.write_all(&pad)
                    .map_err(|e| IoError::FileError(e.to_string()))?;
            }
        }

        Ok(())
    }

    fn dim_list_size(&self) -> usize {
        if self.dims.is_empty() {
            return 8; // tag + count
        }
        let mut size = 8; // tag + count
        for dim in &self.dims {
            size += 4 + pad_to_4(dim.name.len()); // name
            size += 4; // length
        }
        size
    }

    fn attr_list_size(&self, attrs: &[(String, NcValue)]) -> usize {
        if attrs.is_empty() {
            return 8; // tag + count
        }
        let mut size = 8; // tag + count
        for (name, value) in attrs {
            size += 4 + pad_to_4(name.len()); // name
            size += 4 + 4; // nc_type + nelems
            size += pad_to_4(value.element_count() * element_size_for_nc_type(value.nc_type()));
        }
        size
    }

    // =========================================================================
    // Reading
    // =========================================================================

    /// Read a NetCDF Classic file from a reader
    pub fn read_from<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        // Read and verify magic
        let mut magic = [0u8; 4];
        reader
            .read_exact(&mut magic)
            .map_err(|e| IoError::FormatError(format!("Failed to read magic: {}", e)))?;
        if &magic != NC_MAGIC {
            return Err(IoError::FormatError(
                "Not a NetCDF Classic format file (bad magic)".to_string(),
            ));
        }

        // Number of records
        let num_records = reader
            .read_u32::<BigEndian>()
            .map_err(|e| IoError::FormatError(e.to_string()))? as usize;

        // Dimensions
        let dims = read_dim_list(reader)?;

        // Global attributes
        let global_attrs = read_attr_list(reader)?;

        // Variables (headers only first)
        let (mut vars, offsets, vsizes) = read_var_headers(reader)?;

        // Read variable data from offsets
        for (i, var) in vars.iter_mut().enumerate() {
            let offset = offsets[i];
            let vsize = vsizes[i];

            reader
                .seek(SeekFrom::Start(offset as u64))
                .map_err(|e| IoError::FormatError(format!("Failed to seek to var data: {}", e)))?;

            // Calculate actual data size (without padding)
            let total_elements = var_total_elements(var, &dims, num_records);
            let actual_size = total_elements * var.data_type.element_size();
            let read_size = actual_size.min(vsize);

            let mut data = vec![0u8; read_size];
            reader
                .read_exact(&mut data)
                .map_err(|e| IoError::FormatError(format!("Failed to read var data: {}", e)))?;

            var.data = data;
        }

        Ok(NcFile {
            dims,
            global_attrs,
            vars,
            num_records,
        })
    }

    /// Read from a file path
    pub fn read_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
        let mut reader = std::io::BufReader::new(file);
        Self::read_from(&mut reader)
    }
}

impl Default for NcFile {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Helper functions
// =============================================================================

/// Pad a size to the next multiple of 4
fn pad_to_4(n: usize) -> usize {
    (n + 3) & !3
}

fn element_size_for_nc_type(nc_type: u32) -> usize {
    match nc_type {
        NC_BYTE | NC_CHAR => 1,
        NC_SHORT => 2,
        NC_INT | NC_FLOAT => 4,
        NC_DOUBLE => 8,
        _ => 1,
    }
}

/// Write a name (length-prefixed, padded to 4 bytes)
fn write_name<W: Write>(w: &mut W, name: &str) -> Result<()> {
    let bytes = name.as_bytes();
    w.write_u32::<BigEndian>(bytes.len() as u32)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    w.write_all(bytes)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    // Pad to 4-byte boundary
    let padding = pad_to_4(bytes.len()) - bytes.len();
    if padding > 0 {
        let pad = vec![0u8; padding];
        w.write_all(&pad)
            .map_err(|e| IoError::FileError(e.to_string()))?;
    }
    Ok(())
}

/// Write an attribute value
fn write_attr_value<W: Write>(w: &mut W, value: &NcValue) -> Result<()> {
    w.write_u32::<BigEndian>(value.nc_type())
        .map_err(|e| IoError::FileError(e.to_string()))?;
    w.write_u32::<BigEndian>(value.element_count() as u32)
        .map_err(|e| IoError::FileError(e.to_string()))?;

    let elem_size = element_size_for_nc_type(value.nc_type());
    let raw_size = value.element_count() * elem_size;

    match value {
        NcValue::Bytes(v) => {
            for &b in v {
                w.write_i8(b)
                    .map_err(|e| IoError::FileError(e.to_string()))?;
            }
        }
        NcValue::Text(s) => {
            w.write_all(s.as_bytes())
                .map_err(|e| IoError::FileError(e.to_string()))?;
        }
        NcValue::Shorts(v) => {
            for &val in v {
                w.write_i16::<BigEndian>(val)
                    .map_err(|e| IoError::FileError(e.to_string()))?;
            }
        }
        NcValue::Ints(v) => {
            for &val in v {
                w.write_i32::<BigEndian>(val)
                    .map_err(|e| IoError::FileError(e.to_string()))?;
            }
        }
        NcValue::Floats(v) => {
            for &val in v {
                w.write_f32::<BigEndian>(val)
                    .map_err(|e| IoError::FileError(e.to_string()))?;
            }
        }
        NcValue::Doubles(v) => {
            for &val in v {
                w.write_f64::<BigEndian>(val)
                    .map_err(|e| IoError::FileError(e.to_string()))?;
            }
        }
    }

    // Pad to 4-byte boundary
    let padding = pad_to_4(raw_size) - raw_size;
    if padding > 0 {
        let pad = vec![0u8; padding];
        w.write_all(&pad)
            .map_err(|e| IoError::FileError(e.to_string()))?;
    }

    Ok(())
}

/// Read a padded name from a reader
fn read_name<R: Read>(r: &mut R) -> Result<String> {
    let len = r
        .read_u32::<BigEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read name length: {}", e)))?
        as usize;
    let padded_len = pad_to_4(len);
    let mut buf = vec![0u8; padded_len];
    r.read_exact(&mut buf)
        .map_err(|e| IoError::FormatError(format!("Failed to read name: {}", e)))?;
    buf.truncate(len);
    String::from_utf8(buf)
        .map_err(|e| IoError::FormatError(format!("Invalid UTF-8 in name: {}", e)))
}

/// Read a dimension list
fn read_dim_list<R: Read>(r: &mut R) -> Result<Vec<NcDimension>> {
    let tag = r
        .read_u32::<BigEndian>()
        .map_err(|e| IoError::FormatError(e.to_string()))?;
    let count = r
        .read_u32::<BigEndian>()
        .map_err(|e| IoError::FormatError(e.to_string()))? as usize;

    if tag == NC_ABSENT || count == 0 {
        return Ok(Vec::new());
    }

    if tag != NC_DIMENSION {
        return Err(IoError::FormatError(format!(
            "Expected NC_DIMENSION tag, got 0x{:08X}",
            tag
        )));
    }

    let mut dims = Vec::with_capacity(count);
    for _ in 0..count {
        let name = read_name(r)?;
        let len = r
            .read_u32::<BigEndian>()
            .map_err(|e| IoError::FormatError(e.to_string()))? as usize;

        let is_unlimited = len == 0;
        let length = if is_unlimited { None } else { Some(len) };

        dims.push(NcDimension {
            name,
            length,
            is_unlimited,
        });
    }
    Ok(dims)
}

/// Read an attribute list
fn read_attr_list<R: Read>(r: &mut R) -> Result<Vec<(String, NcValue)>> {
    let tag = r
        .read_u32::<BigEndian>()
        .map_err(|e| IoError::FormatError(e.to_string()))?;
    let count = r
        .read_u32::<BigEndian>()
        .map_err(|e| IoError::FormatError(e.to_string()))? as usize;

    if tag == NC_ABSENT || count == 0 {
        return Ok(Vec::new());
    }

    if tag != NC_ATTRIBUTE {
        return Err(IoError::FormatError(format!(
            "Expected NC_ATTRIBUTE tag, got 0x{:08X}",
            tag
        )));
    }

    let mut attrs = Vec::with_capacity(count);
    for _ in 0..count {
        let name = read_name(r)?;
        let value = read_attr_value(r)?;
        attrs.push((name, value));
    }
    Ok(attrs)
}

/// Read an attribute value
fn read_attr_value<R: Read>(r: &mut R) -> Result<NcValue> {
    let nc_type = r
        .read_u32::<BigEndian>()
        .map_err(|e| IoError::FormatError(e.to_string()))?;
    let nelems = r
        .read_u32::<BigEndian>()
        .map_err(|e| IoError::FormatError(e.to_string()))? as usize;

    let elem_size = element_size_for_nc_type(nc_type);
    let raw_size = nelems * elem_size;
    let padded_size = pad_to_4(raw_size);

    let value = match nc_type {
        NC_BYTE => {
            let mut v = Vec::with_capacity(nelems);
            for _ in 0..nelems {
                v.push(
                    r.read_i8()
                        .map_err(|e| IoError::FormatError(e.to_string()))?,
                );
            }
            // Read padding
            let padding = padded_size - raw_size;
            if padding > 0 {
                let mut pad = vec![0u8; padding];
                r.read_exact(&mut pad)
                    .map_err(|e| IoError::FormatError(e.to_string()))?;
            }
            NcValue::Bytes(v)
        }
        NC_CHAR => {
            let mut buf = vec![0u8; nelems];
            r.read_exact(&mut buf)
                .map_err(|e| IoError::FormatError(e.to_string()))?;
            // Read padding
            let padding = padded_size - raw_size;
            if padding > 0 {
                let mut pad = vec![0u8; padding];
                r.read_exact(&mut pad)
                    .map_err(|e| IoError::FormatError(e.to_string()))?;
            }
            let s = String::from_utf8_lossy(&buf)
                .trim_end_matches('\0')
                .to_string();
            NcValue::Text(s)
        }
        NC_SHORT => {
            let mut v = Vec::with_capacity(nelems);
            for _ in 0..nelems {
                v.push(
                    r.read_i16::<BigEndian>()
                        .map_err(|e| IoError::FormatError(e.to_string()))?,
                );
            }
            let padding = padded_size - raw_size;
            if padding > 0 {
                let mut pad = vec![0u8; padding];
                r.read_exact(&mut pad)
                    .map_err(|e| IoError::FormatError(e.to_string()))?;
            }
            NcValue::Shorts(v)
        }
        NC_INT => {
            let mut v = Vec::with_capacity(nelems);
            for _ in 0..nelems {
                v.push(
                    r.read_i32::<BigEndian>()
                        .map_err(|e| IoError::FormatError(e.to_string()))?,
                );
            }
            NcValue::Ints(v)
        }
        NC_FLOAT => {
            let mut v = Vec::with_capacity(nelems);
            for _ in 0..nelems {
                v.push(
                    r.read_f32::<BigEndian>()
                        .map_err(|e| IoError::FormatError(e.to_string()))?,
                );
            }
            NcValue::Floats(v)
        }
        NC_DOUBLE => {
            let mut v = Vec::with_capacity(nelems);
            for _ in 0..nelems {
                v.push(
                    r.read_f64::<BigEndian>()
                        .map_err(|e| IoError::FormatError(e.to_string()))?,
                );
            }
            NcValue::Doubles(v)
        }
        _ => {
            // Skip unknown type
            let mut skip = vec![0u8; padded_size];
            r.read_exact(&mut skip)
                .map_err(|e| IoError::FormatError(e.to_string()))?;
            NcValue::Bytes(Vec::new())
        }
    };

    Ok(value)
}

/// Read variable headers (without data)
fn read_var_headers<R: Read>(r: &mut R) -> Result<(Vec<NcVariable>, Vec<usize>, Vec<usize>)> {
    let tag = r
        .read_u32::<BigEndian>()
        .map_err(|e| IoError::FormatError(e.to_string()))?;
    let count = r
        .read_u32::<BigEndian>()
        .map_err(|e| IoError::FormatError(e.to_string()))? as usize;

    if tag == NC_ABSENT || count == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }

    if tag != NC_VARIABLE {
        return Err(IoError::FormatError(format!(
            "Expected NC_VARIABLE tag, got 0x{:08X}",
            tag
        )));
    }

    let mut vars = Vec::with_capacity(count);
    let mut offsets = Vec::with_capacity(count);
    let mut vsizes = Vec::with_capacity(count);

    for _ in 0..count {
        let name = read_name(r)?;

        // Dimension IDs
        let ndims = r
            .read_u32::<BigEndian>()
            .map_err(|e| IoError::FormatError(e.to_string()))? as usize;
        let mut dim_indices = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            dim_indices.push(
                r.read_u32::<BigEndian>()
                    .map_err(|e| IoError::FormatError(e.to_string()))? as usize,
            );
        }

        // Variable attributes
        let attributes = read_attr_list(r)?;

        // nc_type
        let nc_type = r
            .read_u32::<BigEndian>()
            .map_err(|e| IoError::FormatError(e.to_string()))?;
        let data_type = NcDataType::from_nc_type(nc_type)?;

        // vsize
        let vsize = r
            .read_u32::<BigEndian>()
            .map_err(|e| IoError::FormatError(e.to_string()))? as usize;

        // begin (offset)
        let begin = r
            .read_u32::<BigEndian>()
            .map_err(|e| IoError::FormatError(e.to_string()))? as usize;

        vars.push(NcVariable {
            name,
            data_type,
            dim_indices,
            attributes,
            data: Vec::new(), // filled later
        });
        offsets.push(begin);
        vsizes.push(vsize);
    }

    Ok((vars, offsets, vsizes))
}

/// Calculate total elements for a variable
fn var_total_elements(var: &NcVariable, dims: &[NcDimension], num_records: usize) -> usize {
    if var.dim_indices.is_empty() {
        return 1; // scalar
    }
    var.dim_indices
        .iter()
        .map(|&idx| {
            if dims[idx].is_unlimited {
                num_records
            } else {
                dims[idx].length.unwrap_or(0)
            }
        })
        .product::<usize>()
        .max(0)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_empty_file() {
        let nc = NcFile::new();
        assert!(nc.dimensions().is_empty());
        assert!(nc.global_attributes().is_empty());
        assert!(nc.variable_names().is_empty());
    }

    #[test]
    fn test_add_dimensions() {
        let mut nc = NcFile::new();
        nc.add_dimension("x", Some(10))
            .expect("Failed to add x dim");
        nc.add_dimension("y", Some(20))
            .expect("Failed to add y dim");
        nc.add_dimension("time", None)
            .expect("Failed to add unlimited dim");

        assert_eq!(nc.dimensions().len(), 3);
        assert_eq!(nc.dimensions()[0].name, "x");
        assert_eq!(nc.dimensions()[0].length, Some(10));
        assert!(!nc.dimensions()[0].is_unlimited);
        assert_eq!(nc.dimensions()[2].name, "time");
        assert!(nc.dimensions()[2].is_unlimited);
    }

    #[test]
    fn test_duplicate_dimension_rejected() {
        let mut nc = NcFile::new();
        nc.add_dimension("x", Some(10)).expect("first add ok");
        let result = nc.add_dimension("x", Some(5));
        assert!(result.is_err());
    }

    #[test]
    fn test_only_one_unlimited_allowed() {
        let mut nc = NcFile::new();
        nc.add_dimension("time", None).expect("first unlimited ok");
        let result = nc.add_dimension("step", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_variable() {
        let mut nc = NcFile::new();
        nc.add_dimension("x", Some(3)).expect("dim failed");
        nc.add_dimension("y", Some(4)).expect("dim failed");
        nc.add_variable("temp", NcDataType::Float, &["x", "y"])
            .expect("var failed");

        let names = nc.variable_names();
        assert_eq!(names.len(), 1);
        assert_eq!(names[0], "temp");
    }

    #[test]
    fn test_variable_undefined_dimension() {
        let mut nc = NcFile::new();
        nc.add_dimension("x", Some(3)).expect("dim failed");
        let result = nc.add_variable("temp", NcDataType::Float, &["x", "z"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_roundtrip_float_data() {
        let mut nc = NcFile::new();
        nc.add_dimension("x", Some(3)).expect("dim failed");
        nc.add_variable("vals", NcDataType::Float, &["x"])
            .expect("var failed");
        nc.set_variable_f32("vals", &[1.5, 2.5, 3.5])
            .expect("set failed");

        // Write to buffer
        let mut buf = Vec::new();
        nc.write_to(&mut buf).expect("write failed");

        // Read back
        let loaded = NcFile::read_from(&mut std::io::Cursor::new(&buf)).expect("read failed");

        assert_eq!(loaded.dimensions().len(), 1);
        assert_eq!(loaded.variable_names(), vec!["vals"]);

        let var = loaded.variable("vals").expect("var not found");
        let data = var
            .as_f32(loaded.dimensions(), loaded.num_records())
            .expect("as_f32 failed");
        assert_eq!(data.len(), 3);
        assert!((data[0] - 1.5).abs() < 1e-6);
        assert!((data[1] - 2.5).abs() < 1e-6);
        assert!((data[2] - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_roundtrip_double_data() {
        let mut nc = NcFile::new();
        nc.add_dimension("n", Some(4)).expect("dim failed");
        nc.add_variable("data", NcDataType::Double, &["n"])
            .expect("var failed");
        nc.set_variable_f64("data", &[1.0, 2.0, 3.0, 4.0])
            .expect("set failed");

        let mut buf = Vec::new();
        nc.write_to(&mut buf).expect("write failed");

        let loaded = NcFile::read_from(&mut std::io::Cursor::new(&buf)).expect("read failed");
        let var = loaded.variable("data").expect("var not found");
        let data = var
            .as_f64(loaded.dimensions(), loaded.num_records())
            .expect("as_f64 failed");
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_roundtrip_int_data() {
        let mut nc = NcFile::new();
        nc.add_dimension("n", Some(5)).expect("dim failed");
        nc.add_variable("ids", NcDataType::Int, &["n"])
            .expect("var failed");
        nc.set_variable_i32("ids", &[10, 20, 30, 40, 50])
            .expect("set failed");

        let mut buf = Vec::new();
        nc.write_to(&mut buf).expect("write failed");

        let loaded = NcFile::read_from(&mut std::io::Cursor::new(&buf)).expect("read failed");
        let var = loaded.variable("ids").expect("var not found");
        let data = var
            .as_i32(loaded.dimensions(), loaded.num_records())
            .expect("as_i32 failed");
        assert_eq!(data, vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_roundtrip_text_data() {
        let mut nc = NcFile::new();
        nc.add_dimension("len", Some(12)).expect("dim failed");
        nc.add_variable("msg", NcDataType::Char, &["len"])
            .expect("var failed");
        nc.set_variable_text("msg", "Hello World!")
            .expect("set failed");

        let mut buf = Vec::new();
        nc.write_to(&mut buf).expect("write failed");

        let loaded = NcFile::read_from(&mut std::io::Cursor::new(&buf)).expect("read failed");
        let var = loaded.variable("msg").expect("var not found");
        let text = var.as_text().expect("as_text failed");
        assert_eq!(text, "Hello World!");
    }

    #[test]
    fn test_roundtrip_global_attributes() {
        let mut nc = NcFile::new();
        nc.add_global_attribute("title", NcValue::Text("My Dataset".to_string()))
            .expect("attr failed");
        nc.add_global_attribute("version", NcValue::Ints(vec![2]))
            .expect("attr failed");
        nc.add_global_attribute("scale", NcValue::Doubles(vec![0.01]))
            .expect("attr failed");

        let mut buf = Vec::new();
        nc.write_to(&mut buf).expect("write failed");

        let loaded = NcFile::read_from(&mut std::io::Cursor::new(&buf)).expect("read failed");
        let attrs = loaded.global_attributes();
        assert_eq!(attrs.len(), 3);

        assert_eq!(attrs[0].0, "title");
        if let NcValue::Text(ref s) = attrs[0].1 {
            assert_eq!(s, "My Dataset");
        } else {
            panic!("Expected text attribute");
        }

        assert_eq!(attrs[1].0, "version");
        if let NcValue::Ints(ref v) = attrs[1].1 {
            assert_eq!(v, &[2]);
        } else {
            panic!("Expected int attribute");
        }
    }

    #[test]
    fn test_roundtrip_variable_attributes() {
        let mut nc = NcFile::new();
        nc.add_dimension("x", Some(3)).expect("dim failed");
        nc.add_variable("temp", NcDataType::Float, &["x"])
            .expect("var failed");
        nc.add_variable_attribute("temp", "units", NcValue::Text("Celsius".to_string()))
            .expect("attr failed");
        nc.add_variable_attribute("temp", "scale_factor", NcValue::Floats(vec![0.01]))
            .expect("attr failed");
        nc.set_variable_f32("temp", &[20.0, 21.5, 22.0])
            .expect("set failed");

        let mut buf = Vec::new();
        nc.write_to(&mut buf).expect("write failed");

        let loaded = NcFile::read_from(&mut std::io::Cursor::new(&buf)).expect("read failed");
        let var = loaded.variable("temp").expect("var not found");
        assert_eq!(var.attributes.len(), 2);
        assert_eq!(var.attributes[0].0, "units");
        if let NcValue::Text(ref s) = var.attributes[0].1 {
            assert_eq!(s, "Celsius");
        } else {
            panic!("Expected text attr");
        }
    }

    #[test]
    fn test_roundtrip_2d_data() {
        let mut nc = NcFile::new();
        nc.add_dimension("x", Some(2)).expect("dim failed");
        nc.add_dimension("y", Some(3)).expect("dim failed");
        nc.add_variable("grid", NcDataType::Double, &["x", "y"])
            .expect("var failed");
        // Row-major: [1,2,3, 4,5,6]
        nc.set_variable_f64("grid", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("set failed");

        let mut buf = Vec::new();
        nc.write_to(&mut buf).expect("write failed");

        let loaded = NcFile::read_from(&mut std::io::Cursor::new(&buf)).expect("read failed");
        let var = loaded.variable("grid").expect("var not found");
        let data = var
            .as_f64(loaded.dimensions(), loaded.num_records())
            .expect("as_f64 failed");
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_roundtrip_unlimited_dimension() {
        let mut nc = NcFile::new();
        nc.add_dimension("time", None).expect("dim failed");
        nc.add_dimension("x", Some(3)).expect("dim failed");
        nc.add_variable("data", NcDataType::Float, &["time", "x"])
            .expect("var failed");
        // 2 records, each with 3 values
        nc.set_variable_f32("data", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("set failed");

        assert_eq!(nc.num_records(), 2);

        let mut buf = Vec::new();
        nc.write_to(&mut buf).expect("write failed");

        let loaded = NcFile::read_from(&mut std::io::Cursor::new(&buf)).expect("read failed");
        assert_eq!(loaded.num_records(), 2);

        let var = loaded.variable("data").expect("var not found");
        let shape = var.shape(loaded.dimensions(), loaded.num_records());
        assert_eq!(shape, vec![2, 3]);

        let data = var
            .as_f32(loaded.dimensions(), loaded.num_records())
            .expect("as_f32 failed");
        assert_eq!(data.len(), 6);
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[5] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_multiple_variables() {
        let mut nc = NcFile::new();
        nc.add_dimension("x", Some(3)).expect("dim failed");
        nc.add_dimension("y", Some(2)).expect("dim failed");

        nc.add_variable("temp", NcDataType::Float, &["x", "y"])
            .expect("var failed");
        nc.add_variable("pressure", NcDataType::Double, &["x"])
            .expect("var failed");

        nc.set_variable_f32("temp", &[20.0, 21.0, 22.0, 23.0, 24.0, 25.0])
            .expect("set failed");
        nc.set_variable_f64("pressure", &[1013.0, 1012.5, 1012.0])
            .expect("set failed");

        let mut buf = Vec::new();
        nc.write_to(&mut buf).expect("write failed");

        let loaded = NcFile::read_from(&mut std::io::Cursor::new(&buf)).expect("read failed");
        assert_eq!(loaded.variable_names().len(), 2);

        let temp = loaded.variable("temp").expect("var not found");
        let temp_data = temp
            .as_f32(loaded.dimensions(), loaded.num_records())
            .expect("as_f32 failed");
        assert_eq!(temp_data.len(), 6);
        assert!((temp_data[0] - 20.0).abs() < 1e-4);

        let pressure = loaded.variable("pressure").expect("var not found");
        let p_data = pressure
            .as_f64(loaded.dimensions(), loaded.num_records())
            .expect("as_f64 failed");
        assert_eq!(p_data.len(), 3);
        assert!((p_data[0] - 1013.0).abs() < 1e-10);
    }

    #[test]
    fn test_byte_data() {
        let mut nc = NcFile::new();
        nc.add_dimension("n", Some(4)).expect("dim failed");
        nc.add_variable("flags", NcDataType::Byte, &["n"])
            .expect("var failed");
        nc.set_variable_f64("flags", &[0.0, 1.0, 2.0, -1.0])
            .expect("set failed");

        let mut buf = Vec::new();
        nc.write_to(&mut buf).expect("write failed");

        let loaded = NcFile::read_from(&mut std::io::Cursor::new(&buf)).expect("read failed");
        let var = loaded.variable("flags").expect("var not found");
        let data = var
            .as_f64(loaded.dimensions(), loaded.num_records())
            .expect("as_f64 failed");
        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], 1.0);
        assert_eq!(data[2], 2.0);
        assert_eq!(data[3], -1.0);
    }

    #[test]
    fn test_short_data() {
        let mut nc = NcFile::new();
        nc.add_dimension("n", Some(3)).expect("dim failed");
        nc.add_variable("vals", NcDataType::Short, &["n"])
            .expect("var failed");
        nc.set_variable_f64("vals", &[100.0, -200.0, 300.0])
            .expect("set failed");

        let mut buf = Vec::new();
        nc.write_to(&mut buf).expect("write failed");

        let loaded = NcFile::read_from(&mut std::io::Cursor::new(&buf)).expect("read failed");
        let var = loaded.variable("vals").expect("var not found");
        let data = var
            .as_f64(loaded.dimensions(), loaded.num_records())
            .expect("as_f64 failed");
        assert_eq!(data[0], 100.0);
        assert_eq!(data[1], -200.0);
        assert_eq!(data[2], 300.0);
    }

    #[test]
    fn test_file_roundtrip() {
        let dir = std::env::temp_dir().join("scirs2_nc_lite_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.nc");

        let mut nc = NcFile::new();
        nc.add_dimension("x", Some(5)).expect("dim failed");
        nc.add_variable("data", NcDataType::Double, &["x"])
            .expect("var failed");
        nc.set_variable_f64("data", &[1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("set failed");
        nc.add_global_attribute("title", NcValue::Text("Test File".to_string()))
            .expect("attr failed");

        nc.write_to_file(&path).expect("write failed");

        let loaded = NcFile::read_from_file(&path).expect("read failed");
        assert_eq!(loaded.dimensions().len(), 1);
        assert_eq!(loaded.variable_names(), vec!["data"]);

        let var = loaded.variable("data").expect("var not found");
        let data = var
            .as_f64(loaded.dimensions(), loaded.num_records())
            .expect("as_f64 failed");
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_empty_file_roundtrip() {
        let nc = NcFile::new();
        let mut buf = Vec::new();
        nc.write_to(&mut buf).expect("write failed");

        let loaded = NcFile::read_from(&mut std::io::Cursor::new(&buf)).expect("read failed");
        assert!(loaded.dimensions().is_empty());
        assert!(loaded.variable_names().is_empty());
        assert!(loaded.global_attributes().is_empty());
    }

    #[test]
    fn test_short_attribute_values() {
        let mut nc = NcFile::new();
        nc.add_global_attribute("short_vals", NcValue::Shorts(vec![10, 20, 30]))
            .expect("attr failed");
        nc.add_global_attribute("byte_vals", NcValue::Bytes(vec![1, 2, -1]))
            .expect("attr failed");

        let mut buf = Vec::new();
        nc.write_to(&mut buf).expect("write failed");

        let loaded = NcFile::read_from(&mut std::io::Cursor::new(&buf)).expect("read failed");
        let attrs = loaded.global_attributes();
        assert_eq!(attrs.len(), 2);

        if let NcValue::Shorts(ref v) = attrs[0].1 {
            assert_eq!(v, &[10, 20, 30]);
        } else {
            panic!("Expected shorts");
        }

        if let NcValue::Bytes(ref v) = attrs[1].1 {
            assert_eq!(v, &[1, 2, -1]);
        } else {
            panic!("Expected bytes");
        }
    }

    #[test]
    fn test_float_attribute_values() {
        let mut nc = NcFile::new();
        nc.add_global_attribute("scale", NcValue::Floats(vec![0.5, 1.0]))
            .expect("attr failed");

        let mut buf = Vec::new();
        nc.write_to(&mut buf).expect("write failed");

        let loaded = NcFile::read_from(&mut std::io::Cursor::new(&buf)).expect("read failed");
        if let NcValue::Floats(ref v) = loaded.global_attributes()[0].1 {
            assert!((v[0] - 0.5).abs() < 1e-6);
            assert!((v[1] - 1.0).abs() < 1e-6);
        } else {
            panic!("Expected floats");
        }
    }

    #[test]
    fn test_bad_magic_rejected() {
        let bad_data = b"NOTCDF\x00\x00";
        let result = NcFile::read_from(&mut std::io::Cursor::new(bad_data.as_ref()));
        assert!(result.is_err());
    }

    #[test]
    fn test_replace_global_attribute() {
        let mut nc = NcFile::new();
        nc.add_global_attribute("title", NcValue::Text("Old".to_string()))
            .expect("attr failed");
        nc.add_global_attribute("title", NcValue::Text("New".to_string()))
            .expect("replace failed");

        assert_eq!(nc.global_attributes().len(), 1);
        if let NcValue::Text(ref s) = nc.global_attributes()[0].1 {
            assert_eq!(s, "New");
        }
    }

    #[test]
    fn test_replace_variable_attribute() {
        let mut nc = NcFile::new();
        nc.add_dimension("x", Some(1)).expect("dim failed");
        nc.add_variable("v", NcDataType::Float, &["x"])
            .expect("var failed");
        nc.add_variable_attribute("v", "units", NcValue::Text("m".to_string()))
            .expect("attr failed");
        nc.add_variable_attribute("v", "units", NcValue::Text("km".to_string()))
            .expect("replace failed");

        let var = nc.variable("v").expect("var not found");
        assert_eq!(var.attributes.len(), 1);
        if let NcValue::Text(ref s) = var.attributes[0].1 {
            assert_eq!(s, "km");
        }
    }
}
