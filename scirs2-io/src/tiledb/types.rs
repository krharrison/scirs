//! TileDB type definitions.
//!
//! Core types for the TileDB array storage engine, supporting both dense
//! and sparse multidimensional arrays with configurable tiling and compression.

/// Configuration for a TileDB array.
#[derive(Debug, Clone)]
pub struct TileDBConfig {
    /// Maximum number of elements per tile.
    pub tile_capacity: usize,
    /// Compression algorithm to use for tiles.
    pub compression: Compression,
    /// Tile memory order.
    pub tile_order: TileOrder,
    /// Cell order within tiles.
    pub cell_order: TileOrder,
}

impl Default for TileDBConfig {
    fn default() -> Self {
        Self {
            tile_capacity: 1024,
            compression: Compression::None,
            tile_order: TileOrder::RowMajor,
            cell_order: TileOrder::RowMajor,
        }
    }
}

/// Compression algorithm for tiles.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    /// No compression.
    None,
    /// Run-length encoding.
    Rle,
    /// Dictionary encoding.
    Dictionary,
    /// Delta encoding for sorted data.
    Delta,
}

/// Array schema: either dense or sparse.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum ArraySchema {
    /// Dense array with fixed-size dimensions.
    Dense {
        /// Dimensions defining the array shape and tiling.
        dimensions: Vec<Dimension>,
        /// Attributes stored at each cell.
        attributes: Vec<Attribute>,
    },
    /// Sparse array with variable-occupancy cells.
    Sparse {
        /// Dimensions defining the coordinate space.
        dimensions: Vec<Dimension>,
        /// Attributes stored at each occupied cell.
        attributes: Vec<Attribute>,
    },
}

impl ArraySchema {
    /// Return the dimensions of this schema.
    pub fn dimensions(&self) -> &[Dimension] {
        match self {
            ArraySchema::Dense { dimensions, .. } => dimensions,
            ArraySchema::Sparse { dimensions, .. } => dimensions,
        }
    }

    /// Return the attributes of this schema.
    pub fn attributes(&self) -> &[Attribute] {
        match self {
            ArraySchema::Dense { attributes, .. } => attributes,
            ArraySchema::Sparse { attributes, .. } => attributes,
        }
    }

    /// Return true if this is a dense schema.
    pub fn is_dense(&self) -> bool {
        matches!(self, ArraySchema::Dense { .. })
    }
}

/// A single dimension in a TileDB array.
#[derive(Debug, Clone)]
pub struct Dimension {
    /// Dimension name.
    pub name: String,
    /// Domain bounds (inclusive min, inclusive max).
    pub domain: (f64, f64),
    /// Tile extent along this dimension.
    pub tile_extent: f64,
}

impl Dimension {
    /// Create a new dimension.
    pub fn new(name: &str, domain: (f64, f64), tile_extent: f64) -> Self {
        Self {
            name: name.to_string(),
            domain,
            tile_extent,
        }
    }

    /// Return the number of cells in this dimension.
    pub fn num_cells(&self) -> usize {
        let span = self.domain.1 - self.domain.0 + 1.0;
        if span <= 0.0 {
            0
        } else {
            span as usize
        }
    }

    /// Return the number of tiles along this dimension.
    pub fn num_tiles(&self) -> usize {
        let cells = self.num_cells() as f64;
        if self.tile_extent <= 0.0 {
            1
        } else {
            (cells / self.tile_extent).ceil() as usize
        }
    }
}

/// An attribute stored at each cell of the array.
#[derive(Debug, Clone)]
pub struct Attribute {
    /// Attribute name.
    pub name: String,
    /// Data type of this attribute.
    pub dtype: DataType,
}

impl Attribute {
    /// Create a new attribute.
    pub fn new(name: &str, dtype: DataType) -> Self {
        Self {
            name: name.to_string(),
            dtype,
        }
    }
}

/// Supported data types for TileDB attributes.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// 64-bit floating point.
    Float64,
    /// 32-bit floating point.
    Float32,
    /// 64-bit signed integer.
    Int64,
    /// 32-bit signed integer.
    Int32,
    /// Unsigned 8-bit integer (byte).
    UInt8,
}

/// Memory layout order for tiles and cells.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileOrder {
    /// Row-major (C-style) order.
    RowMajor,
    /// Column-major (Fortran-style) order.
    ColMajor,
    /// Hilbert space-filling curve order (for sparse arrays).
    Hilbert,
}

/// Errors specific to TileDB operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum TileDBError {
    /// I/O error.
    Io(std::io::Error),
    /// Schema mismatch or invalid schema.
    SchemaError(String),
    /// Out-of-bounds access.
    OutOfBounds(String),
    /// Invalid query or configuration.
    InvalidQuery(String),
    /// General error.
    Other(String),
}

impl std::fmt::Display for TileDBError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TileDBError::Io(e) => write!(f, "TileDB I/O error: {e}"),
            TileDBError::SchemaError(msg) => write!(f, "TileDB schema error: {msg}"),
            TileDBError::OutOfBounds(msg) => write!(f, "TileDB out of bounds: {msg}"),
            TileDBError::InvalidQuery(msg) => write!(f, "TileDB invalid query: {msg}"),
            TileDBError::Other(msg) => write!(f, "TileDB error: {msg}"),
        }
    }
}

impl std::error::Error for TileDBError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TileDBError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for TileDBError {
    fn from(e: std::io::Error) -> Self {
        TileDBError::Io(e)
    }
}

impl From<TileDBError> for crate::error::IoError {
    fn from(e: TileDBError) -> Self {
        crate::error::IoError::Other(format!("{e}"))
    }
}
