//! Out-of-core array V2 implementation with streaming serialization.
//!
//! This module provides efficient chunked serialization and deserialization
//! for large arrays that don't fit in memory. Key features:
//! - Incremental chunk writing without full materialization
//! - Direct chunk seeking and reading
//! - Optional per-chunk compression
//! - Backward compatibility with V1 format

use super::chunk_format::{
    detect_format_version, read_chunk_index, read_header, write_chunk_index, write_header,
    ChunkIndex, ChunkIndexEntry, CompressionType, FormatVersion, OutOfCoreHeaderV2,
};
use super::chunked::ChunkingStrategy;
use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
use crate::ndarray::{Array, ArrayBase, Data, Dimension};
use oxicode::{config, serde as oxicode_serde};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

/// V2 out-of-core array with streaming support
#[derive(Debug)]
pub struct OutOfCoreArrayV2<A, D>
where
    A: Clone + Serialize + for<'de> Deserialize<'de>,
    D: Dimension + Serialize + for<'de> Deserialize<'de>,
{
    /// File path
    pub file_path: PathBuf,

    /// Array header
    pub header: OutOfCoreHeaderV2,

    /// Chunk index
    pub chunk_index: ChunkIndex,

    /// Whether file is temporary
    is_temp: bool,

    /// Phantom data
    phantom: PhantomData<(A, D)>,
}

impl<A, D> OutOfCoreArrayV2<A, D>
where
    A: Clone + Serialize + for<'de> Deserialize<'de>,
    D: Dimension + Serialize + for<'de> Deserialize<'de>,
{
    /// Create a new V2 out-of-core array with streaming chunk writing
    ///
    /// This method writes chunks incrementally without materializing the entire array
    pub fn new_streaming<S>(
        data: &ArrayBase<S, D>,
        file_path: &Path,
        strategy: ChunkingStrategy,
        compression: CompressionType,
    ) -> CoreResult<Self>
    where
        S: Data<Elem = A>,
    {
        let shape = data.shape().to_vec();
        let total_elements = data.len();
        let element_size = std::mem::size_of::<A>();

        // Calculate chunk size
        let chunk_size = Self::calculate_chunk_size(total_elements, &strategy, element_size);
        let num_chunks = total_elements.div_ceil(chunk_size);

        // Create header with placeholder chunk_index_offset to ensure consistent size
        let mut header = OutOfCoreHeaderV2::new(shape, element_size, num_chunks, compression);
        header.chunk_index_offset = u64::MAX; // Placeholder to ensure header size is consistent

        // Open file for writing
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(file_path)
            .map_err(|e| {
                CoreError::IoError(
                    ErrorContext::new(format!("Failed to create file: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;

        let mut writer = BufWriter::new(file);

        // Write header (will update later with chunk_index_offset)
        let _header_size = write_header(&mut writer, &header)?;

        // Write chunks and build index
        let mut chunk_index = ChunkIndex::new();
        let data_slice = data.as_slice().ok_or_else(|| {
            CoreError::InvalidArgument(
                ErrorContext::new("Array must be contiguous for streaming".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        for chunk_id in 0..num_chunks {
            let start_idx = chunk_id * chunk_size;
            let end_idx = std::cmp::min((chunk_id + 1) * chunk_size, total_elements);
            let chunk_data = &data_slice[start_idx..end_idx];
            let num_elements = chunk_data.len();

            // Get current offset before writing chunk
            let offset = writer.stream_position().map_err(|e| {
                CoreError::IoError(
                    ErrorContext::new(format!("Failed to get stream position: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;

            // Serialize chunk (convert slice to Vec for oxicode)
            let cfg = config::standard();
            let chunk_vec = chunk_data.to_vec();
            let chunk_bytes = oxicode_serde::encode_to_vec(&chunk_vec, cfg).map_err(|e| {
                CoreError::IoError(
                    ErrorContext::new(format!("Failed to serialize chunk {chunk_id}: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;

            let uncompressed_size = chunk_bytes.len();

            // Apply compression if requested
            let (final_bytes, compressed_size) = match compression {
                CompressionType::None => (chunk_bytes, 0),
                #[cfg(feature = "memory_compression")]
                CompressionType::Lz4 => {
                    let compressed = oxiarc_lz4::compress_block(&chunk_bytes).map_err(|e| {
                        CoreError::IoError(
                            ErrorContext::new(format!("LZ4 compression failed: {e}"))
                                .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    })?;
                    let size = compressed.len();
                    (compressed, size)
                }
                #[cfg(feature = "memory_compression")]
                CompressionType::Zstd => {
                    let compressed = oxiarc_zstd::encode_all(&chunk_bytes[..], 3).map_err(|e| {
                        CoreError::IoError(
                            ErrorContext::new(format!("Zstd compression failed: {e}"))
                                .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    })?;
                    let size = compressed.len();
                    (compressed, size)
                }
                #[cfg(feature = "memory_compression")]
                CompressionType::Snappy => {
                    let mut compressed = Vec::new();
                    let mut encoder = snap::write::FrameEncoder::new(&mut compressed);
                    encoder.write_all(&chunk_bytes).map_err(|e| {
                        CoreError::IoError(
                            ErrorContext::new(format!("Snappy compression failed: {e}"))
                                .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    })?;
                    drop(encoder);
                    let size = compressed.len();
                    (compressed, size)
                }
                #[cfg(not(feature = "memory_compression"))]
                _ => {
                    return Err(CoreError::InvalidArgument(
                        ErrorContext::new(
                            "Compression requested but memory_compression feature is not enabled"
                                .to_string(),
                        )
                        .with_location(ErrorLocation::new(file!(), line!())),
                    ))
                }
            };

            // Write chunk data
            writer.write_all(&final_bytes).map_err(|e| {
                CoreError::IoError(
                    ErrorContext::new(format!("Failed to write chunk {chunk_id}: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;

            // Create index entry
            let entry = if compressed_size > 0 {
                ChunkIndexEntry::new(chunk_id, offset, uncompressed_size, num_elements)
                    .with_compression(compressed_size)
            } else {
                ChunkIndexEntry::new(chunk_id, offset, uncompressed_size, num_elements)
            };

            chunk_index.add_entry(entry);
        }

        // Get offset for chunk index
        let chunk_index_offset = writer.stream_position().map_err(|e| {
            CoreError::IoError(
                ErrorContext::new(format!("Failed to get stream position: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        // Write chunk index
        write_chunk_index(&mut writer, &chunk_index)?;

        // Flush writer
        writer.flush().map_err(|e| {
            CoreError::IoError(
                ErrorContext::new(format!("Failed to flush writer: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        // Update header with chunk_index_offset
        header.chunk_index_offset = chunk_index_offset;

        // Re-open file to update header
        drop(writer);
        let mut file = OpenOptions::new()
            .write(true)
            .open(file_path)
            .map_err(|e| {
                CoreError::IoError(
                    ErrorContext::new(format!("Failed to reopen file: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;

        let mut writer = BufWriter::new(&mut file);
        write_header(&mut writer, &header)?;
        writer.flush().map_err(|e| {
            CoreError::IoError(
                ErrorContext::new(format!("Failed to flush header update: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        Ok(Self {
            file_path: file_path.to_path_buf(),
            header,
            chunk_index,
            is_temp: false,
            phantom: PhantomData,
        })
    }

    /// Open an existing V2 out-of-core array
    pub fn open(file_path: &Path) -> CoreResult<Self> {
        let file = File::open(file_path).map_err(|e| {
            CoreError::IoError(
                ErrorContext::new(format!("Failed to open file: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        let mut reader = BufReader::new(file);

        // Detect format version
        let version = detect_format_version(&mut reader)?;

        if version != FormatVersion::V2 {
            return Err(CoreError::InvalidArgument(
                ErrorContext::new(format!("Expected V2 format, got {version:?}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Read header
        let header = read_header(&mut reader)?;

        // Read chunk index
        let chunk_index = read_chunk_index(&mut reader, header.chunk_index_offset)?;

        Ok(Self {
            file_path: file_path.to_path_buf(),
            header,
            chunk_index,
            is_temp: false,
            phantom: PhantomData,
        })
    }

    /// Load a specific chunk by ID
    pub fn load_chunk_v2(&self, chunk_id: usize) -> CoreResult<Vec<A>> {
        // Get chunk entry
        let entry = self.chunk_index.get_entry(chunk_id).ok_or_else(|| {
            CoreError::IndexError(
                ErrorContext::new(format!(
                    "Chunk {chunk_id} out of bounds (max {})",
                    self.chunk_index.len()
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        // Open file for reading
        let file = File::open(&self.file_path).map_err(|e| {
            CoreError::IoError(
                ErrorContext::new(format!("Failed to open file: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        let mut reader = BufReader::new(file);

        // Seek to chunk offset
        reader.seek(SeekFrom::Start(entry.offset)).map_err(|e| {
            CoreError::IoError(
                ErrorContext::new(format!("Failed to seek to chunk {chunk_id}: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        // Read chunk data
        let mut chunk_bytes = vec![0u8; entry.disk_size()];
        reader.read_exact(&mut chunk_bytes).map_err(|e| {
            CoreError::IoError(
                ErrorContext::new(format!("Failed to read chunk {chunk_id}: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        // Decompress if needed
        let decompressed_bytes =
            if entry.is_compressed() {
                match self.header.compression {
                    CompressionType::None => chunk_bytes,
                    #[cfg(feature = "memory_compression")]
                    CompressionType::Lz4 => oxiarc_lz4::decompress_block(&chunk_bytes, entry.size)
                        .map_err(|e| {
                            CoreError::IoError(
                                ErrorContext::new(format!("LZ4 decompression failed: {e}"))
                                    .with_location(ErrorLocation::new(file!(), line!())),
                            )
                        })?,
                    #[cfg(feature = "memory_compression")]
                    CompressionType::Zstd => {
                        oxiarc_zstd::decode_all(&chunk_bytes[..]).map_err(|e| {
                            CoreError::IoError(
                                ErrorContext::new(format!("Zstd decompression failed: {e}"))
                                    .with_location(ErrorLocation::new(file!(), line!())),
                            )
                        })?
                    }
                    #[cfg(feature = "memory_compression")]
                    CompressionType::Snappy => {
                        let mut decompressed = Vec::new();
                        let mut decoder = snap::read::FrameDecoder::new(&chunk_bytes[..]);
                        decoder.read_to_end(&mut decompressed).map_err(|e| {
                            CoreError::IoError(
                                ErrorContext::new(format!("Snappy decompression failed: {e}"))
                                    .with_location(ErrorLocation::new(file!(), line!())),
                            )
                        })?;
                        decompressed
                    }
                    #[cfg(not(feature = "memory_compression"))]
                    _ => return Err(CoreError::InvalidArgument(
                        ErrorContext::new(
                            "Compression detected but memory_compression feature is not enabled"
                                .to_string(),
                        )
                        .with_location(ErrorLocation::new(file!(), line!())),
                    )),
                }
            } else {
                chunk_bytes
            };

        // Deserialize chunk
        let cfg = config::standard();
        let (chunk_data, _len): (Vec<A>, usize) =
            oxicode_serde::decode_owned_from_slice(&decompressed_bytes, cfg).map_err(|e| {
                CoreError::IoError(
                    ErrorContext::new(format!("Failed to deserialize chunk {chunk_id}: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;

        Ok(chunk_data)
    }

    /// Load the entire array (all chunks)
    pub fn load(&self) -> CoreResult<Array<A, D>> {
        let mut all_data = Vec::with_capacity(self.header.total_elements);

        for chunk_id in 0..self.chunk_index.len() {
            let chunk_data = self.load_chunk_v2(chunk_id)?;
            all_data.extend(chunk_data);
        }

        // Create array from data using dynamic dimension first
        let dyn_array = Array::from_shape_vec(crate::ndarray::IxDyn(&self.header.shape), all_data)
            .map_err(|e| {
                CoreError::DimensionError(
                    ErrorContext::new(format!("Failed to create dynamic array: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;

        // Convert to target dimension type
        let array = dyn_array.into_dimensionality::<D>().map_err(|e| {
            CoreError::DimensionError(
                ErrorContext::new(format!("Failed to convert to target dimension: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        Ok(array)
    }

    /// Get number of chunks
    pub fn num_chunks(&self) -> usize {
        self.chunk_index.len()
    }

    /// Check if array is temporary
    pub const fn is_temp(&self) -> bool {
        self.is_temp
    }

    /// Mark array as temporary (will be deleted on drop)
    pub fn set_temp(&mut self, is_temp: bool) {
        self.is_temp = is_temp;
    }

    /// Calculate chunk size based on strategy
    fn calculate_chunk_size(
        total_elements: usize,
        strategy: &ChunkingStrategy,
        element_size: usize,
    ) -> usize {
        match strategy {
            ChunkingStrategy::Auto => {
                // Default to 64MB chunks
                let target_bytes = 64 * 1024 * 1024;
                (target_bytes / element_size).max(1)
            }
            ChunkingStrategy::Fixed(size) => *size,
            ChunkingStrategy::FixedBytes(bytes) => (bytes / element_size).max(1),
            ChunkingStrategy::NumChunks(n) => total_elements.div_ceil(*n),
            ChunkingStrategy::Advanced(_) => {
                let target_bytes = 64 * 1024 * 1024;
                (target_bytes / element_size).max(1)
            }
        }
    }
}

impl<A, D> Drop for OutOfCoreArrayV2<A, D>
where
    A: Clone + Serialize + for<'de> Deserialize<'de>,
    D: Dimension + Serialize + for<'de> Deserialize<'de>,
{
    fn drop(&mut self) {
        if self.is_temp {
            let _ = std::fs::remove_file(&self.file_path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndarray::Array1;
    use tempfile::NamedTempFile;

    #[test]
    fn test_streaming_write_and_read() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let file_path = temp_file.path();

        // Create test data
        let data: Array1<f64> = Array1::from_vec((0..10000).map(|i| i as f64).collect());

        // Write with streaming
        let oc_array = OutOfCoreArrayV2::new_streaming(
            &data,
            file_path,
            ChunkingStrategy::Fixed(1000),
            CompressionType::None,
        )
        .expect("Failed to create out-of-core array");

        assert_eq!(oc_array.num_chunks(), 10);

        // Read back
        let loaded = oc_array.load().expect("Failed to load array");

        assert_eq!(loaded.len(), data.len());
        for i in 0..data.len() {
            assert!((loaded[i] - data[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_chunk_loading() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let file_path = temp_file.path();

        // Create test data
        let data: Array1<f64> = Array1::from_vec((0..5000).map(|i| i as f64).collect());

        // Write with streaming
        let oc_array = OutOfCoreArrayV2::new_streaming(
            &data,
            file_path,
            ChunkingStrategy::Fixed(1000),
            CompressionType::None,
        )
        .expect("Failed to create out-of-core array");

        // Load specific chunk
        let chunk_data = oc_array.load_chunk_v2(2).expect("Failed to load chunk");

        assert_eq!(chunk_data.len(), 1000);
        assert!((chunk_data[0] - 2000.0).abs() < 1e-10);
        assert!((chunk_data[999] - 2999.0).abs() < 1e-10);
    }

    #[test]
    #[cfg(feature = "memory_compression")]
    fn test_compression() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let file_path = temp_file.path();

        // Create test data with repetitive pattern (compresses well)
        let data: Array1<f64> = Array1::from_vec(vec![42.0; 10000]);

        // Write with LZ4 compression
        let oc_array = OutOfCoreArrayV2::new_streaming(
            &data,
            file_path,
            ChunkingStrategy::Fixed(1000),
            CompressionType::Lz4,
        )
        .expect("Failed to create out-of-core array");

        // Verify compression worked
        let entry = oc_array.chunk_index.get_entry(0).expect("Chunk not found");
        assert!(entry.is_compressed());
        assert!(entry.compressed_size < entry.size);

        // Read back and verify
        let loaded = oc_array.load().expect("Failed to load array");
        assert_eq!(loaded.len(), data.len());
        for i in 0..data.len() {
            assert!((loaded[i] - data[i]).abs() < 1e-10);
        }
    }
}
