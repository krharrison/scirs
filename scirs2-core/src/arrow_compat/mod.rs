//! Arrow Format Interoperability for SciRS2
//!
//! This module provides zero-copy (where possible) conversions between
//! ndarray types and Apache Arrow columnar format, enabling efficient
//! data exchange across processes, languages, and frameworks.
//!
//! # Overview
//!
//! Apache Arrow defines a language-independent columnar memory format for
//! flat and hierarchical data. This module bridges the gap between SciRS2's
//! ndarray-based computation and Arrow's ecosystem of tools (DataFusion,
//! Polars, DuckDB, etc.).
//!
//! # Features
//!
//! - **ndarray ↔ Arrow conversions**: Convert between `Array1<T>`/`Array2<T>`
//!   and Arrow arrays/RecordBatch with zero-copy when memory layouts align.
//!
//! - **Type support**: f32, f64, i32, i64, bool, and String conversions,
//!   including nullable (`Option<T>`) support.
//!
//! - **Arrow IPC**: Serialize and deserialize arrays using Arrow's IPC
//!   format for cross-process communication and persistent storage.
//!
//! - **Memory-mapped files**: Read Arrow IPC files via mmap for efficient
//!   access to large datasets without loading into RAM.
//!
//! - **Schema utilities**: Create and inspect Arrow schemas from ndarray
//!   shape and dtype information, with metadata for roundtrip fidelity.
//!
//! # Quick Start
//!
//! ```rust
//! use scirs2_core::arrow_compat::conversions::{array1_to_arrow, arrow_to_array1};
//! use ndarray::Array1;
//!
//! // Convert ndarray to Arrow
//! let arr = Array1::from_vec(vec![1.0_f64, 2.0, 3.0]);
//! let arrow_arr = array1_to_arrow(&arr).expect("conversion failed");
//!
//! // Convert back
//! let recovered: Array1<f64> = arrow_to_array1(&arrow_arr).expect("conversion failed");
//! assert_eq!(arr, recovered);
//! ```
//!
//! # 2D Array ↔ RecordBatch
//!
//! ```rust
//! use scirs2_core::arrow_compat::conversions::{array2_to_record_batch, record_batch_to_array2};
//! use ndarray::Array2;
//!
//! let matrix = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
//!     .expect("shape error");
//! let batch = array2_to_record_batch(&matrix, Some(&["x", "y"])).expect("conversion failed");
//! let recovered: Array2<f64> = record_batch_to_array2(&batch).expect("conversion failed");
//! assert_eq!(matrix, recovered);
//! ```
//!
//! # Arrow IPC Serialization
//!
//! ```rust
//! use scirs2_core::arrow_compat::conversions::array2_to_record_batch;
//! use scirs2_core::arrow_compat::ipc::{record_batch_to_ipc_stream, ipc_stream_to_record_batches};
//! use ndarray::Array2;
//!
//! let matrix = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
//!     .expect("shape error");
//! let batch = array2_to_record_batch(&matrix, None).expect("conversion failed");
//!
//! // Serialize to bytes (for cross-process sharing)
//! let bytes = record_batch_to_ipc_stream(&[batch]).expect("serialization failed");
//!
//! // Deserialize back
//! let batches = ipc_stream_to_record_batches(&bytes).expect("deserialization failed");
//! assert_eq!(batches[0].num_rows(), 3);
//! ```
//!
//! # Feature Gate
//!
//! This module is only available when the `arrow` feature is enabled:
//!
//! ```toml
//! [dependencies]
//! scirs2-core = { version = "0.3.0", features = ["arrow"] }
//! ```

pub mod conversions;
pub mod error;
pub mod ipc;
pub mod schema;
pub mod traits;

// Re-export commonly used items at the module level
pub use conversions::{
    array1_to_arrow, array1_to_arrow_zero_copy, array2_to_record_batch, arrow_to_array1,
    arrow_to_array1_nullable, nullable_array1_to_arrow, record_batch_column_by_name,
    record_batch_column_to_array1, record_batch_to_array2,
};
pub use error::{ArrowCompatError, ArrowResult};
pub use ipc::{
    ipc_file_schema, ipc_file_to_record_batches, ipc_stream_schema, ipc_stream_to_record_batches,
    mmap_read_ipc_file, read_ipc_file, read_ipc_stream_file, record_batch_to_ipc_file,
    record_batch_to_ipc_stream, write_ipc_file, write_ipc_stream_file, MmapIpcReader,
};
pub use schema::{
    extract_ndarray_shape, infer_schema, ndarray_schema_1d, ndarray_schema_2d,
    ndarray_shape_schema, schema_with_metadata, validate_schema,
};
pub use traits::{FromArrowArray, ToArrowArray, ZeroCopyFromArrow};
