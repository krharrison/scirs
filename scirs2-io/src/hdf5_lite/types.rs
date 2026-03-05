//! Type definitions for HDF5 lite reader

use std::collections::HashMap;

/// HDF5 data type descriptor
#[derive(Debug, Clone, PartialEq)]
pub enum Hdf5DataType {
    /// 8-bit signed integer
    Int8,
    /// 16-bit signed integer
    Int16,
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
    /// 8-bit unsigned integer
    UInt8,
    /// 16-bit unsigned integer
    UInt16,
    /// 32-bit unsigned integer
    UInt32,
    /// 64-bit unsigned integer
    UInt64,
    /// 32-bit float
    Float32,
    /// 64-bit float
    Float64,
    /// Fixed-length string with given byte count
    FixedString(usize),
    /// Variable-length string
    VarString,
    /// Compound type with named fields
    Compound(Vec<(String, Hdf5DataType, usize)>),
    /// Array type: element type and dimensions
    Array(Box<Hdf5DataType>, Vec<usize>),
    /// Opaque data with given byte count
    Opaque(usize),
    /// Unknown / unsupported type
    Unknown(u8, usize),
}

impl Hdf5DataType {
    /// Size in bytes for a single element of this type
    pub fn element_size(&self) -> usize {
        match self {
            Self::Int8 | Self::UInt8 => 1,
            Self::Int16 | Self::UInt16 => 2,
            Self::Int32 | Self::UInt32 | Self::Float32 => 4,
            Self::Int64 | Self::UInt64 | Self::Float64 => 8,
            Self::FixedString(n) => *n,
            Self::VarString => 16, // global heap reference size
            Self::Compound(fields) => fields
                .iter()
                .map(|(_, dt, _)| dt.element_size())
                .sum::<usize>()
                .max(1),
            Self::Array(elem, dims) => {
                let count: usize = dims.iter().product();
                elem.element_size() * count
            }
            Self::Opaque(n) => *n,
            Self::Unknown(_, size) => *size,
        }
    }
}

/// Typed value container for HDF5 data
#[derive(Debug, Clone, PartialEq)]
pub enum Hdf5Value {
    /// Signed 8-bit integers
    Int8(Vec<i8>),
    /// Signed 16-bit integers
    Int16(Vec<i16>),
    /// Signed 32-bit integers
    Int32(Vec<i32>),
    /// Signed 64-bit integers
    Int64(Vec<i64>),
    /// Unsigned 8-bit integers
    UInt8(Vec<u8>),
    /// Unsigned 16-bit integers
    UInt16(Vec<u16>),
    /// Unsigned 32-bit integers
    UInt32(Vec<u32>),
    /// Unsigned 64-bit integers
    UInt64(Vec<u64>),
    /// 32-bit floats
    Float32(Vec<f32>),
    /// 64-bit floats
    Float64(Vec<f64>),
    /// String values
    Strings(Vec<String>),
    /// Raw bytes (opaque or unknown type)
    Raw(Vec<u8>),
}

impl Hdf5Value {
    /// Number of elements in the value
    pub fn len(&self) -> usize {
        match self {
            Self::Int8(v) => v.len(),
            Self::Int16(v) => v.len(),
            Self::Int32(v) => v.len(),
            Self::Int64(v) => v.len(),
            Self::UInt8(v) => v.len(),
            Self::UInt16(v) => v.len(),
            Self::UInt32(v) => v.len(),
            Self::UInt64(v) => v.len(),
            Self::Float32(v) => v.len(),
            Self::Float64(v) => v.len(),
            Self::Strings(v) => v.len(),
            Self::Raw(v) => v.len(),
        }
    }

    /// Whether the value is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Try to convert to f64 slice (returns None for non-numeric types)
    pub fn as_f64(&self) -> Option<Vec<f64>> {
        match self {
            Self::Int8(v) => Some(v.iter().map(|x| *x as f64).collect()),
            Self::Int16(v) => Some(v.iter().map(|x| *x as f64).collect()),
            Self::Int32(v) => Some(v.iter().map(|x| *x as f64).collect()),
            Self::Int64(v) => Some(v.iter().map(|x| *x as f64).collect()),
            Self::UInt8(v) => Some(v.iter().map(|x| *x as f64).collect()),
            Self::UInt16(v) => Some(v.iter().map(|x| *x as f64).collect()),
            Self::UInt32(v) => Some(v.iter().map(|x| *x as f64).collect()),
            Self::UInt64(v) => Some(v.iter().map(|x| *x as f64).collect()),
            Self::Float32(v) => Some(v.iter().map(|x| *x as f64).collect()),
            Self::Float64(v) => Some(v.clone()),
            _ => None,
        }
    }
}

/// An HDF5 attribute (name + typed value)
#[derive(Debug, Clone)]
pub struct Hdf5Attribute {
    /// Attribute name
    pub name: String,
    /// Data type of the attribute
    pub dtype: Hdf5DataType,
    /// Attribute value
    pub value: Hdf5Value,
}

/// An HDF5 dataset with shape, type, and data
#[derive(Debug, Clone)]
pub struct Hdf5Dataset {
    /// Dataset name
    pub name: String,
    /// Full path in the file
    pub path: String,
    /// Data type
    pub dtype: Hdf5DataType,
    /// Shape (dimension sizes)
    pub shape: Vec<usize>,
    /// Dataset values
    pub data: Hdf5Value,
    /// Attributes on this dataset
    pub attributes: HashMap<String, Hdf5Attribute>,
}

/// Type of an HDF5 node
#[derive(Debug, Clone, PartialEq)]
pub enum Hdf5NodeType {
    /// Group node (container)
    Group,
    /// Dataset node (data)
    Dataset,
}

/// An HDF5 node in the hierarchy
#[derive(Debug, Clone)]
pub struct Hdf5Node {
    /// Node name
    pub name: String,
    /// Full path in the file
    pub path: String,
    /// Node type
    pub node_type: Hdf5NodeType,
}

/// An HDF5 group with child names and attributes
#[derive(Debug, Clone)]
pub struct Hdf5Group {
    /// Group name
    pub name: String,
    /// Full path
    pub path: String,
    /// Names of child nodes (groups and datasets)
    pub children: Vec<String>,
    /// Child node details
    pub nodes: Vec<Hdf5Node>,
    /// Attributes on this group
    pub attributes: HashMap<String, Hdf5Attribute>,
}
