//! Streaming support for large datasets.
//!
//! This module provides two complementary APIs:
//!
//! 1. **Legacy streaming** (`StreamingIterator`, `StreamConfig`, `DataChunk`, …):
//!    the original chunk-based streaming from CSV and binary files.
//!
//! 2. **New streaming iterator** (`NewStreamingIterator`, `DataSource`,
//!    `StreamingIteratorConfig`, `StreamingDataChunk`): a cleaner API that
//!    supports in-memory, CSV, and directory sources with optional shuffle.
//!
//! 3. **DataLoader** (`DataLoader`, `Batch`, `SamplingStrategy`,
//!    `DataLoaderConfig`): PyTorch-style mini-batch iterator for neural-
//!    network training, with stratified, weighted, and epoch-level shuffle
//!    strategies.
//!
//! 4. **Transforms** (`Transform`, `Normalize`, `Filter`, `MapFeatures`,
//!    `TransformPipeline`): composable, lazy transformations on
//!    `StreamingDataChunk` values.

// Sub-modules
pub mod dataloader;
pub mod iterator;
pub mod legacy;
pub mod transforms;

// Re-export legacy API (preserves all existing public re-exports from lib.rs)
pub use legacy::{
    stream_classification, stream_csv, stream_regression, DataChunk, StreamConfig, StreamProcessor,
    StreamStats, StreamTransformer, StreamingIterator,
};

// Re-export new streaming iterator API
pub use iterator::{DataSource, NewStreamingIterator, StreamingDataChunk, StreamingIteratorConfig};

// Re-export DataLoader API
pub use dataloader::{Batch, DataLoader, DataLoaderConfig, SamplingStrategy};

// Re-export transform API
pub use transforms::{Filter, MapFeatures, Normalize, Transform, TransformPipeline};
