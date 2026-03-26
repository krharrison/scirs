//! Lance-inspired columnar storage format (pure Rust, in-process).
//!
//! Provides reading and writing of a Lance-like binary columnar format:
//!
//! - `LanceSchema` / `LanceField` / `LanceDataType` — schema descriptions
//! - `LanceColumn` / `LanceBatch` — in-memory column storage
//! - `LanceWriter` — streaming batch writer (`Write` sink)
//! - `LanceReader` — batch reader (`Read + Seek` source)
//!
//! Binary layout:
//! ```text
//! File  = LANCE001(8) + SchemaLen(4 LE u32) + SchemaJSON(N) + Batch* + Footer
//! Batch = NumRows(4 LE u32) + Column*
//! Col   = TypeTag(1) + Nullable(1) + DataLen(4 LE u32) + Data + [validity]
//! Footer= BatchCount(4 LE u32) + LANCEEOF(8)
//! ```

pub mod types;
pub mod writer;

pub use types::{LanceBatch, LanceColumn, LanceDataType, LanceField, LanceSchema};
pub use writer::{LanceReader, LanceWriter, LANCE_EOF, LANCE_MAGIC};
