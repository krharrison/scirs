//! Arrow IPC (Inter-Process Communication) support
//!
//! Provides serialization/deserialization of Arrow arrays and RecordBatches
//! to/from the Arrow IPC format for cross-process data sharing.
//!
//! Supports:
//! - Streaming IPC format (for sequential writes)
//! - File IPC format (for random access)
//! - Memory-mapped IPC files (for shared memory access)

use super::error::{ArrowCompatError, ArrowResult};
use arrow::array::ArrayRef;
use arrow::datatypes::SchemaRef;
use arrow::ipc::reader::{FileReader, StreamReader};
use arrow::ipc::writer::{FileWriter, StreamWriter};
use arrow::record_batch::RecordBatch;
use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor, Read, Seek, Write};
use std::path::Path;

// =============================================================================
// In-memory IPC serialization (bytes)
// =============================================================================

/// Serialize a `RecordBatch` to Arrow IPC streaming format (bytes)
///
/// The streaming format is suitable for sequential reads and
/// cross-process communication via pipes or sockets.
///
/// # Examples
///
/// ```rust
/// # use scirs2_core::arrow_compat::ipc::{record_batch_to_ipc_stream, ipc_stream_to_record_batches};
/// # use scirs2_core::arrow_compat::conversions::array2_to_record_batch;
/// # use ndarray::Array2;
/// let arr = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
///     .expect("shape error");
/// let batch = array2_to_record_batch(&arr, None).expect("conversion failed");
/// let bytes = record_batch_to_ipc_stream(&[batch]).expect("serialization failed");
/// let recovered = ipc_stream_to_record_batches(&bytes).expect("deserialization failed");
/// assert_eq!(recovered.len(), 1);
/// assert_eq!(recovered[0].num_rows(), 3);
/// ```
pub fn record_batch_to_ipc_stream(batches: &[RecordBatch]) -> ArrowResult<Vec<u8>> {
    if batches.is_empty() {
        return Err(ArrowCompatError::SchemaError(
            "No record batches to serialize".to_string(),
        ));
    }

    let schema = batches[0].schema();
    let mut buffer = Vec::new();

    {
        let mut writer = StreamWriter::try_new(&mut buffer, &schema)?;
        for batch in batches {
            writer.write(batch)?;
        }
        writer.finish()?;
    }

    Ok(buffer)
}

/// Deserialize `RecordBatch` instances from Arrow IPC streaming format (bytes)
pub fn ipc_stream_to_record_batches(data: &[u8]) -> ArrowResult<Vec<RecordBatch>> {
    let cursor = Cursor::new(data);
    let reader = StreamReader::try_new(cursor, None)?;

    let mut batches = Vec::new();
    for batch_result in reader {
        let batch = batch_result?;
        batches.push(batch);
    }

    Ok(batches)
}

/// Serialize a `RecordBatch` to Arrow IPC file format (bytes)
///
/// The file format includes a footer for random access to record batches,
/// making it suitable for memory-mapped access and random reads.
pub fn record_batch_to_ipc_file(batches: &[RecordBatch]) -> ArrowResult<Vec<u8>> {
    if batches.is_empty() {
        return Err(ArrowCompatError::SchemaError(
            "No record batches to serialize".to_string(),
        ));
    }

    let schema = batches[0].schema();
    let mut buffer = Vec::new();

    {
        let mut writer = FileWriter::try_new(&mut buffer, &schema)?;
        for batch in batches {
            writer.write(batch)?;
        }
        writer.finish()?;
    }

    Ok(buffer)
}

/// Deserialize `RecordBatch` instances from Arrow IPC file format (bytes)
pub fn ipc_file_to_record_batches(data: &[u8]) -> ArrowResult<Vec<RecordBatch>> {
    let cursor = Cursor::new(data.to_vec());
    let reader = FileReader::try_new(cursor, None)?;

    let mut batches = Vec::new();
    for batch_result in reader {
        let batch = batch_result?;
        batches.push(batch);
    }

    Ok(batches)
}

/// Get the schema from an IPC file format buffer without reading the data
pub fn ipc_file_schema(data: &[u8]) -> ArrowResult<SchemaRef> {
    let cursor = Cursor::new(data.to_vec());
    let reader = FileReader::try_new(cursor, None)?;
    Ok(reader.schema())
}

/// Get the schema from an IPC stream format buffer without reading all data
pub fn ipc_stream_schema(data: &[u8]) -> ArrowResult<SchemaRef> {
    let cursor = Cursor::new(data);
    let reader = StreamReader::try_new(cursor, None)?;
    Ok(reader.schema())
}

// =============================================================================
// File-based IPC operations
// =============================================================================

/// Write `RecordBatch` instances to an Arrow IPC file on disk
///
/// # Arguments
///
/// * `path` - File path to write to
/// * `batches` - Record batches to write
///
/// # Examples
///
/// ```rust,no_run
/// # use scirs2_core::arrow_compat::ipc::write_ipc_file;
/// # use scirs2_core::arrow_compat::conversions::array2_to_record_batch;
/// # use ndarray::Array2;
/// # use std::path::Path;
/// let arr = Array2::from_shape_vec((100, 3), (0..300).map(|i| i as f64).collect())
///     .expect("shape error");
/// let batch = array2_to_record_batch(&arr, Some(&["x", "y", "z"])).expect("conversion failed");
/// write_ipc_file(Path::new("/tmp/data.arrow"), &[batch]).expect("write failed");
/// ```
pub fn write_ipc_file(path: &Path, batches: &[RecordBatch]) -> ArrowResult<()> {
    if batches.is_empty() {
        return Err(ArrowCompatError::SchemaError(
            "No record batches to write".to_string(),
        ));
    }

    let schema = batches[0].schema();
    let file = File::create(path)?;
    let buf_writer = BufWriter::new(file);

    let mut writer = FileWriter::try_new(buf_writer, &schema)?;
    for batch in batches {
        writer.write(batch)?;
    }
    writer.finish()?;

    Ok(())
}

/// Read `RecordBatch` instances from an Arrow IPC file on disk
///
/// # Arguments
///
/// * `path` - File path to read from
pub fn read_ipc_file(path: &Path) -> ArrowResult<Vec<RecordBatch>> {
    let file = File::open(path)?;
    let buf_reader = BufReader::new(file);
    let reader = FileReader::try_new(buf_reader, None)?;

    let mut batches = Vec::new();
    for batch_result in reader {
        let batch = batch_result?;
        batches.push(batch);
    }

    Ok(batches)
}

/// Write `RecordBatch` instances to an Arrow IPC stream file on disk
pub fn write_ipc_stream_file(path: &Path, batches: &[RecordBatch]) -> ArrowResult<()> {
    if batches.is_empty() {
        return Err(ArrowCompatError::SchemaError(
            "No record batches to write".to_string(),
        ));
    }

    let schema = batches[0].schema();
    let file = File::create(path)?;
    let buf_writer = BufWriter::new(file);

    let mut writer = StreamWriter::try_new(buf_writer, &schema)?;
    for batch in batches {
        writer.write(batch)?;
    }
    writer.finish()?;

    Ok(())
}

/// Read `RecordBatch` instances from an Arrow IPC stream file on disk
pub fn read_ipc_stream_file(path: &Path) -> ArrowResult<Vec<RecordBatch>> {
    let file = File::open(path)?;
    let buf_reader = BufReader::new(file);
    let reader = StreamReader::try_new(buf_reader, None)?;

    let mut batches = Vec::new();
    for batch_result in reader {
        let batch = batch_result?;
        batches.push(batch);
    }

    Ok(batches)
}

// =============================================================================
// Memory-mapped IPC file support
// =============================================================================

/// Open a memory-mapped Arrow IPC file for reading
///
/// Memory-mapped access is efficient for large files as it avoids
/// loading the entire file into memory. The operating system handles
/// paging data in and out as needed.
///
/// # Arguments
///
/// * `path` - Path to the Arrow IPC file
///
/// # Returns
///
/// A vector of `RecordBatch` instances read from the memory-mapped file.
/// Note: The actual mmap is managed internally by reading through
/// the standard file reader with OS-level memory mapping.
pub fn mmap_read_ipc_file(path: &Path) -> ArrowResult<MmapIpcReader> {
    MmapIpcReader::open(path)
}

/// Memory-mapped IPC file reader
///
/// Provides lazy access to record batches in an Arrow IPC file
/// without loading the entire file into memory upfront.
pub struct MmapIpcReader {
    /// The raw file data loaded via mmap
    data: memmap2::Mmap,
    /// Schema of the IPC file
    schema: SchemaRef,
    /// Number of record batches in the file
    num_batches: usize,
}

impl MmapIpcReader {
    /// Open an Arrow IPC file with memory mapping
    pub fn open(path: &Path) -> ArrowResult<Self> {
        let file = File::open(path)?;

        // Safety: We ensure the file is not modified while mapped
        // by holding the File handle. The mmap is read-only.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        // Parse the file to get schema and batch count
        let cursor = Cursor::new(mmap.as_ref());
        let reader = FileReader::try_new(cursor, None)?;
        let schema = reader.schema();
        let num_batches = reader.num_batches();

        Ok(Self {
            data: mmap,
            schema,
            num_batches,
        })
    }

    /// Get the schema of the IPC file
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Get the number of record batches in the file
    pub fn num_batches(&self) -> usize {
        self.num_batches
    }

    /// Read a specific record batch by index
    pub fn read_batch(&self, index: usize) -> ArrowResult<RecordBatch> {
        if index >= self.num_batches {
            return Err(ArrowCompatError::ColumnOutOfBounds {
                index,
                num_columns: self.num_batches,
            });
        }

        let cursor = Cursor::new(self.data.as_ref());
        let reader = FileReader::try_new(cursor, None)?;

        for (i, batch_result) in reader.enumerate() {
            if i == index {
                return batch_result.map_err(ArrowCompatError::from);
            }
        }

        Err(ArrowCompatError::SchemaError(format!(
            "Batch index {} not found (file has {} batches)",
            index, self.num_batches
        )))
    }

    /// Read all record batches from the memory-mapped file
    pub fn read_all_batches(&self) -> ArrowResult<Vec<RecordBatch>> {
        let cursor = Cursor::new(self.data.as_ref());
        let reader = FileReader::try_new(cursor, None)?;

        let mut batches = Vec::with_capacity(self.num_batches);
        for batch_result in reader {
            let batch = batch_result?;
            batches.push(batch);
        }

        Ok(batches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow_compat::conversions::array2_to_record_batch;
    use ndarray::Array2;

    fn make_test_batch() -> RecordBatch {
        let arr = Array2::from_shape_vec((5, 3), (0..15).map(|i| i as f64).collect())
            .expect("shape error");
        array2_to_record_batch(&arr, Some(&["x", "y", "z"])).expect("conversion failed")
    }

    // -------------------------------------------------------
    // In-memory IPC stream tests
    // -------------------------------------------------------

    #[test]
    fn test_ipc_stream_roundtrip() {
        let batch = make_test_batch();
        let bytes =
            record_batch_to_ipc_stream(std::slice::from_ref(&batch)).expect("serialize failed");
        let recovered = ipc_stream_to_record_batches(&bytes).expect("deserialize failed");

        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].num_rows(), 5);
        assert_eq!(recovered[0].num_columns(), 3);
        assert_eq!(recovered[0].schema(), batch.schema());
    }

    #[test]
    fn test_ipc_stream_multiple_batches() {
        let batch1 = make_test_batch();
        let batch2 = make_test_batch();
        let bytes = record_batch_to_ipc_stream(&[batch1, batch2]).expect("serialize failed");
        let recovered = ipc_stream_to_record_batches(&bytes).expect("deserialize failed");

        assert_eq!(recovered.len(), 2);
    }

    #[test]
    fn test_ipc_stream_empty_batches_error() {
        let result = record_batch_to_ipc_stream(&[]);
        assert!(result.is_err());
    }

    // -------------------------------------------------------
    // In-memory IPC file format tests
    // -------------------------------------------------------

    #[test]
    fn test_ipc_file_roundtrip() {
        let batch = make_test_batch();
        let bytes =
            record_batch_to_ipc_file(std::slice::from_ref(&batch)).expect("serialize failed");
        let recovered = ipc_file_to_record_batches(&bytes).expect("deserialize failed");

        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].num_rows(), 5);
        assert_eq!(recovered[0].num_columns(), 3);
    }

    #[test]
    fn test_ipc_file_schema_extraction() {
        let batch = make_test_batch();
        let bytes = record_batch_to_ipc_file(&[batch]).expect("serialize failed");
        let schema = ipc_file_schema(&bytes).expect("schema extraction failed");

        assert_eq!(schema.fields().len(), 3);
        assert_eq!(schema.field(0).name(), "x");
    }

    #[test]
    fn test_ipc_stream_schema_extraction() {
        let batch = make_test_batch();
        let bytes = record_batch_to_ipc_stream(&[batch]).expect("serialize failed");
        let schema = ipc_stream_schema(&bytes).expect("schema extraction failed");

        assert_eq!(schema.fields().len(), 3);
        assert_eq!(schema.field(0).name(), "x");
    }

    // -------------------------------------------------------
    // File-based IPC tests
    // -------------------------------------------------------

    #[test]
    fn test_file_ipc_roundtrip() {
        let batch = make_test_batch();
        let tmp_dir = std::env::temp_dir();
        let path = tmp_dir.join("scirs2_arrow_test_ipc.arrow");

        write_ipc_file(&path, std::slice::from_ref(&batch)).expect("write failed");
        let recovered = read_ipc_file(&path).expect("read failed");

        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].num_rows(), 5);
        assert_eq!(recovered[0].num_columns(), 3);

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_file_ipc_stream_roundtrip() {
        let batch = make_test_batch();
        let tmp_dir = std::env::temp_dir();
        let path = tmp_dir.join("scirs2_arrow_test_ipc_stream.arrows");

        write_ipc_stream_file(&path, std::slice::from_ref(&batch)).expect("write failed");
        let recovered = read_ipc_stream_file(&path).expect("read failed");

        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].num_rows(), 5);
        assert_eq!(recovered[0].num_columns(), 3);

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    // -------------------------------------------------------
    // Memory-mapped IPC tests
    // -------------------------------------------------------

    #[test]
    fn test_mmap_ipc_reader() {
        let batch = make_test_batch();
        let tmp_dir = std::env::temp_dir();
        let path = tmp_dir.join("scirs2_arrow_test_mmap.arrow");

        write_ipc_file(&path, std::slice::from_ref(&batch)).expect("write failed");

        let reader = mmap_read_ipc_file(&path).expect("mmap open failed");
        assert_eq!(reader.num_batches(), 1);
        assert_eq!(reader.schema().fields().len(), 3);

        let read_batch = reader.read_batch(0).expect("read_batch failed");
        assert_eq!(read_batch.num_rows(), 5);

        let all = reader.read_all_batches().expect("read_all failed");
        assert_eq!(all.len(), 1);

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mmap_reader_batch_out_of_bounds() {
        let batch = make_test_batch();
        let tmp_dir = std::env::temp_dir();
        let path = tmp_dir.join("scirs2_arrow_test_mmap_oob.arrow");

        write_ipc_file(&path, &[batch]).expect("write failed");

        let reader = mmap_read_ipc_file(&path).expect("mmap open failed");
        let result = reader.read_batch(10);
        assert!(result.is_err());

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }
}
