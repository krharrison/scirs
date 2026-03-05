//! Pure Rust HDF5 file reader (read-only)
//!
//! Implements reading of HDF5 (Hierarchical Data Format version 5) binary files
//! without any C library dependency. Supports the most common use cases:
//!
//! - HDF5 v0 and v1 superblocks
//! - B-tree based group navigation (v1 B-tree nodes)
//! - Dataset reading: contiguous and chunked storage layouts
//! - Data types: integers (8/16/32/64-bit), floats (32/64-bit), fixed/variable strings
//! - Attribute reading on groups and datasets
//! - Recursive group traversal
//!
//! # Limitations
//!
//! This is a read-only implementation. Writing HDF5 files is not supported.
//! Some advanced features are not implemented:
//! - Compact storage layout
//! - External storage layout
//! - Virtual datasets
//! - SZIP/custom filter decompression (deflate/zlib is supported)
//!
//! # Example
//!
//! ```rust,no_run
//! use scirs2_io::hdf5_lite::{Hdf5Reader, Hdf5Value};
//!
//! let reader = Hdf5Reader::open("data.h5")?;
//!
//! // List root group contents
//! let root = reader.root_group()?;
//! for name in &root.children {
//!     println!("  {}", name);
//! }
//!
//! // Read a dataset
//! let dataset = reader.read_dataset("/measurements/temperature")?;
//! println!("Shape: {:?}", dataset.shape);
//! if let Hdf5Value::Float64(data) = &dataset.data {
//!     println!("First value: {}", data[0]);
//! }
//! # Ok::<(), scirs2_io::error::IoError>(())
//! ```

mod reader;
mod superblock;
mod types;

pub use reader::Hdf5Reader;
pub use types::{
    Hdf5Attribute, Hdf5DataType, Hdf5Dataset, Hdf5Group, Hdf5Node, Hdf5NodeType, Hdf5Value,
};

#[cfg(test)]
mod tests;
