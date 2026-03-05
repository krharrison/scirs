//! Async out-of-core array I/O using tokio.
//!
//! Provides non-blocking chunk loading/saving for datasets larger than RAM.
//! This module is feature-gated behind the `async` feature and is not
//! available on WASM targets.
//!
//! # Overview
//!
//! - `AsyncChunkLoader`: Loads `Array2<F>` chunks from a binary file in the
//!   background while the current chunk is being processed.
//! - `AsyncChunkWriter`: Appends `Array2<F>` chunks to a binary file
//!   asynchronously.
//! - `async_map_chunks`: Apply a transformation function to every chunk of an
//!   input file and write results to an output file, overlapping I/O and
//!   computation via `tokio::task::spawn_blocking`.
//!
//! # File Format
//!
//! All three types share a simple binary format so that files written by
//! `AsyncChunkWriter` can be read back by `AsyncChunkLoader`:
//!
//! ```text
//! Offset    Size    Field
//! 0         8       Magic bytes: b"SCI2ASYNC"  (8 bytes, last byte '\0')
//! 8         4       Version (little-endian u32 = 1)
//! 12        4       element_size (little-endian u32)
//! 16        8       num_cols (little-endian u64)
//! 24        8       total_rows (little-endian u64)
//! 32        8       chunk_size (rows per chunk, little-endian u64)
//! 40        8       num_chunks (little-endian u64)
//! 48        16      Reserved (zeroes)
//! 64        …       Raw data: row-major f32/f64/… values
//! ```
//!
//! The data section contains rows laid out consecutively.  Each chunk occupies
//! `chunk_size * num_cols * element_size` bytes, except the final chunk which
//! may be smaller.
//!
//! # Feature Gates
//!
//! This module is compiled only when **both** conditions are met:
//! - The `async` Cargo feature is enabled.
//! - The target is not `wasm32` (tokio's multi-thread runtime is not available
//!   on WASM).

use crate::memory::out_of_core::OutOfCoreError;

use ndarray::Array2;
use std::io::{SeekFrom, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use tokio::fs::File as TokioFile;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, BufReader, BufWriter};

// ============================================================================
// File format constants and header
// ============================================================================

const MAGIC: [u8; 8] = *b"SCI2ASY\0"; // 8 bytes
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 64; // bytes

/// Parsed representation of the on-disk file header.
#[derive(Debug, Clone, Copy)]
struct FileHeader {
    element_size: u32,
    num_cols: u64,
    total_rows: u64,
    chunk_rows: u64,
    num_chunks: u64,
}

impl FileHeader {
    /// Serialise to the 64-byte on-disk representation.
    fn to_bytes(self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..8].copy_from_slice(&MAGIC);
        buf[8..12].copy_from_slice(&VERSION.to_le_bytes());
        buf[12..16].copy_from_slice(&self.element_size.to_le_bytes());
        buf[16..24].copy_from_slice(&self.num_cols.to_le_bytes());
        buf[24..32].copy_from_slice(&self.total_rows.to_le_bytes());
        buf[32..40].copy_from_slice(&self.chunk_rows.to_le_bytes());
        buf[40..48].copy_from_slice(&self.num_chunks.to_le_bytes());
        // bytes 48..64 are reserved zeroes
        buf
    }

    /// Deserialise from a 64-byte buffer, validating magic and version.
    fn from_bytes<F>(buf: &[u8; HEADER_SIZE]) -> Result<Self, OutOfCoreError> {
        if buf[0..8] != MAGIC {
            return Err(OutOfCoreError::SerializationError(
                "Invalid magic bytes in async out-of-core file".to_string(),
            ));
        }
        let version = u32::from_le_bytes(buf[8..12].try_into().map_err(|_| {
            OutOfCoreError::SerializationError("Failed to read version field".to_string())
        })?);
        if version != VERSION {
            return Err(OutOfCoreError::SerializationError(format!(
                "Unsupported file version: {} (expected {})",
                version, VERSION
            )));
        }
        let element_size = u32::from_le_bytes(buf[12..16].try_into().map_err(|_| {
            OutOfCoreError::SerializationError("Failed to read element_size".to_string())
        })?);
        if element_size != std::mem::size_of::<F>() as u32 {
            return Err(OutOfCoreError::SerializationError(format!(
                "Element size mismatch: file has {element_size} bytes, \
                 expected {} bytes for type {}",
                std::mem::size_of::<F>(),
                std::any::type_name::<F>(),
            )));
        }
        let num_cols = u64::from_le_bytes(buf[16..24].try_into().map_err(|_| {
            OutOfCoreError::SerializationError("Failed to read num_cols".to_string())
        })?);
        let total_rows = u64::from_le_bytes(buf[24..32].try_into().map_err(|_| {
            OutOfCoreError::SerializationError("Failed to read total_rows".to_string())
        })?);
        let chunk_rows = u64::from_le_bytes(buf[32..40].try_into().map_err(|_| {
            OutOfCoreError::SerializationError("Failed to read chunk_rows".to_string())
        })?);
        let num_chunks = u64::from_le_bytes(buf[40..48].try_into().map_err(|_| {
            OutOfCoreError::SerializationError("Failed to read num_chunks".to_string())
        })?);
        Ok(Self {
            element_size,
            num_cols,
            total_rows,
            chunk_rows,
            num_chunks,
        })
    }
}

// ============================================================================
// Helper: convert raw bytes to typed Vec<F>
// ============================================================================

/// Interpret `raw_bytes` as a flat sequence of `F` values.
///
/// # Safety
///
/// The caller guarantees that `raw_bytes.len()` is a multiple of
/// `size_of::<F>()` and that the bytes represent valid `F` bit-patterns
/// (true for all primitive numeric types).
unsafe fn bytes_to_vec<F: Copy + Default>(raw_bytes: &[u8]) -> Vec<F> {
    let elem_size = std::mem::size_of::<F>();
    debug_assert_eq!(raw_bytes.len() % elem_size, 0);
    let n = raw_bytes.len() / elem_size;
    let mut result = vec![F::default(); n];
    std::ptr::copy_nonoverlapping(
        raw_bytes.as_ptr(),
        result.as_mut_ptr() as *mut u8,
        raw_bytes.len(),
    );
    result
}

/// Reinterpret a slice of `F` as a byte slice.
///
/// # Safety
///
/// `F` must be a primitive numeric type whose byte representation is valid to
/// copy verbatim (all `Copy + Default` primitives satisfy this).
unsafe fn slice_to_bytes<F>(slice: &[F]) -> &[u8] {
    std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice))
}

// ============================================================================
// AsyncChunkLoader
// ============================================================================

/// Async chunk loader: loads `Array2<F>` chunks from a binary file produced by
/// [`AsyncChunkWriter`] while CPU-bound processing of the previous chunk runs
/// in parallel.
///
/// # Example
///
/// ```rust,no_run
/// # #[cfg(all(feature = "async", not(target_arch = "wasm32")))]
/// # {
/// use scirs2_core::memory::async_out_of_core::AsyncChunkLoader;
/// use std::path::Path;
///
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let mut loader = AsyncChunkLoader::<f64>::open(Path::new("/tmp/data.aoc"), 256).await?;
/// while let Some(chunk) = loader.next_chunk().await? {
///     // process chunk (Array2<f64>)
///     let _ = chunk;
/// }
/// # Ok(())
/// # }
/// # }
/// ```
pub struct AsyncChunkLoader<F> {
    file_path: PathBuf,
    total_rows: usize,
    num_cols: usize,
    chunk_size: usize, // rows per chunk
    current_chunk: usize,
    num_chunks: usize,
    file: BufReader<TokioFile>,
    _phantom: PhantomData<F>,
}

impl<F: Copy + Default + Send + 'static> AsyncChunkLoader<F> {
    /// Open an existing async out-of-core file and prepare to stream its
    /// chunks with the given `chunk_size` (rows per chunk).
    ///
    /// The `chunk_size` passed here overrides the one stored in the file and
    /// controls how many rows are returned per call to [`next_chunk`](Self::next_chunk).
    /// Pass `0` to use the chunk size from the file header.
    pub async fn open(path: &Path, chunk_size: usize) -> Result<Self, OutOfCoreError> {
        let mut raw_file = TokioFile::open(path).await.map_err(|e| {
            OutOfCoreError::IoError(std::io::Error::new(
                e.kind(),
                format!("Failed to open {}: {e}", path.display()),
            ))
        })?;

        // Read header
        let mut header_buf = [0u8; HEADER_SIZE];
        raw_file.read_exact(&mut header_buf).await.map_err(|e| {
            OutOfCoreError::IoError(std::io::Error::new(
                e.kind(),
                format!("Failed to read header: {e}"),
            ))
        })?;
        let header = FileHeader::from_bytes::<F>(&header_buf)?;

        let effective_chunk_size = if chunk_size == 0 {
            header.chunk_rows as usize
        } else {
            chunk_size
        };

        if effective_chunk_size == 0 {
            return Err(OutOfCoreError::InvalidChunkSize(
                "chunk_size must be greater than zero".to_string(),
            ));
        }

        let total_rows = header.total_rows as usize;
        let num_cols = header.num_cols as usize;
        let num_chunks = total_rows.div_ceil(effective_chunk_size);

        Ok(Self {
            file_path: path.to_path_buf(),
            total_rows,
            num_cols,
            chunk_size: effective_chunk_size,
            current_chunk: 0,
            num_chunks,
            file: BufReader::new(raw_file),
            _phantom: PhantomData,
        })
    }

    /// Return the next chunk as an `Array2<F>`, or `None` at end-of-stream.
    ///
    /// Chunks are loaded sequentially; rows are returned in order.
    pub async fn next_chunk(&mut self) -> Result<Option<Array2<F>>, OutOfCoreError> {
        if self.current_chunk >= self.num_chunks {
            return Ok(None);
        }

        let chunk_start_row = self.current_chunk * self.chunk_size;
        let chunk_end_row = (chunk_start_row + self.chunk_size).min(self.total_rows);
        let rows_in_chunk = chunk_end_row - chunk_start_row;
        let elem_size = std::mem::size_of::<F>();
        let byte_count = rows_in_chunk * self.num_cols * elem_size;

        // Seek to correct byte offset in the data section.
        let data_byte_offset =
            HEADER_SIZE as u64 + (chunk_start_row as u64 * self.num_cols as u64 * elem_size as u64);
        self.file
            .seek(SeekFrom::Start(data_byte_offset))
            .await
            .map_err(|e| {
                OutOfCoreError::IoError(std::io::Error::new(
                    e.kind(),
                    format!("Seek failed for chunk {}: {e}", self.current_chunk),
                ))
            })?;

        let mut raw_buf = vec![0u8; byte_count];
        self.file.read_exact(&mut raw_buf).await.map_err(|e| {
            OutOfCoreError::IoError(std::io::Error::new(
                e.kind(),
                format!("Read failed for chunk {}: {e}", self.current_chunk),
            ))
        })?;

        // Convert bytes → Vec<F> → Array2<F>
        // SAFETY: raw_buf contains exactly `byte_count` bytes of valid F
        // bit-patterns as written by AsyncChunkWriter.
        let flat: Vec<F> = unsafe { bytes_to_vec::<F>(&raw_buf) };

        let shape = (rows_in_chunk, self.num_cols);
        let chunk = Array2::from_shape_vec(shape, flat).map_err(|e| {
            OutOfCoreError::SerializationError(format!(
                "Failed to build Array2 from chunk data: {e}"
            ))
        })?;

        self.current_chunk += 1;
        Ok(Some(chunk))
    }

    /// Prefetch the next `n` chunks as background tasks.
    ///
    /// Each task is given its own independent file handle so that seeking
    /// does not interfere with the main sequential read path.  The caller
    /// owns the returned [`JoinHandle`](tokio::task::JoinHandle) vec and
    /// must `.await` each handle to obtain the chunk data.
    pub async fn prefetch(
        &self,
        n: usize,
    ) -> Vec<tokio::task::JoinHandle<Result<Array2<F>, OutOfCoreError>>>
    where
        F: Copy + Default + Send + 'static,
    {
        let mut handles = Vec::new();
        let elem_size = std::mem::size_of::<F>();

        for i in 0..n {
            let future_chunk = self.current_chunk + i;
            if future_chunk >= self.num_chunks {
                break;
            }

            let path = self.file_path.clone();
            let chunk_start_row = future_chunk * self.chunk_size;
            let chunk_end_row = (chunk_start_row + self.chunk_size).min(self.total_rows);
            let rows_in_chunk = chunk_end_row - chunk_start_row;
            let num_cols = self.num_cols;
            let byte_count = rows_in_chunk * num_cols * elem_size;
            let data_byte_offset =
                HEADER_SIZE as u64 + chunk_start_row as u64 * num_cols as u64 * elem_size as u64;

            let handle = tokio::spawn(async move {
                let mut file = TokioFile::open(&path).await.map_err(|e| {
                    OutOfCoreError::IoError(std::io::Error::new(
                        e.kind(),
                        format!("Prefetch open failed: {e}"),
                    ))
                })?;
                file.seek(SeekFrom::Start(data_byte_offset))
                    .await
                    .map_err(|e| {
                        OutOfCoreError::IoError(std::io::Error::new(
                            e.kind(),
                            format!("Prefetch seek failed: {e}"),
                        ))
                    })?;
                let mut raw_buf = vec![0u8; byte_count];
                file.read_exact(&mut raw_buf).await.map_err(|e| {
                    OutOfCoreError::IoError(std::io::Error::new(
                        e.kind(),
                        format!("Prefetch read failed: {e}"),
                    ))
                })?;
                // SAFETY: same guarantee as in next_chunk
                let flat: Vec<F> = unsafe { bytes_to_vec::<F>(&raw_buf) };
                Array2::from_shape_vec((rows_in_chunk, num_cols), flat).map_err(|e| {
                    OutOfCoreError::SerializationError(format!(
                        "Prefetch Array2 construction failed: {e}"
                    ))
                })
            });
            handles.push(handle);
        }
        handles
    }

    /// Total number of chunks this loader will yield given the current
    /// `chunk_size`.
    pub fn num_chunks(&self) -> usize {
        self.num_chunks
    }

    /// Reset the loader to the beginning of the file so that chunks can be
    /// streamed again from the start.
    pub fn reset(&mut self) {
        self.current_chunk = 0;
        // The BufReader seek in `next_chunk` will reposition the file cursor
        // before the next read, so there is nothing more to do here.
    }

    /// Total number of rows across all chunks.
    pub fn total_rows(&self) -> usize {
        self.total_rows
    }

    /// Number of columns per row.
    pub fn num_cols(&self) -> usize {
        self.num_cols
    }
}

// ============================================================================
// AsyncChunkWriter
// ============================================================================

/// Async chunk writer: writes `Array2<F>` chunks to a binary file that can
/// later be read back by [`AsyncChunkLoader`].
///
/// Call [`write_chunk`](Self::write_chunk) for each chunk in order, then
/// [`flush`](Self::flush) / [`finalize`](Self::finalize) to persist all data
/// and update the header with the final row count.
///
/// # Example
///
/// ```rust,no_run
/// # #[cfg(all(feature = "async", not(target_arch = "wasm32")))]
/// # {
/// use scirs2_core::memory::async_out_of_core::AsyncChunkWriter;
/// use ndarray::Array2;
/// use std::path::Path;
///
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let mut writer = AsyncChunkWriter::<f64>::create(Path::new("/tmp/out.aoc"), 4).await?;
/// let chunk = Array2::from_elem((256, 4), 1.0_f64);
/// writer.write_chunk(&chunk).await?;
/// writer.finalize().await?;
/// # Ok(())
/// # }
/// # }
/// ```
pub struct AsyncChunkWriter<F> {
    file_path: PathBuf,
    num_cols: usize,
    chunk_rows: usize,
    total_rows_written: usize,
    chunks_written: usize,
    file: BufWriter<TokioFile>,
    _phantom: PhantomData<F>,
}

impl<F: Copy + Default + Send + 'static> AsyncChunkWriter<F> {
    /// Create (or overwrite) the file at `path` and write an initial header.
    ///
    /// `num_cols` must equal the number of columns in every chunk passed to
    /// [`write_chunk`](Self::write_chunk).  The header's `total_rows` field
    /// is updated when [`finalize`](Self::finalize) is called.
    pub async fn create(path: &Path, num_cols: usize) -> Result<Self, OutOfCoreError> {
        if num_cols == 0 {
            return Err(OutOfCoreError::InvalidChunkSize(
                "num_cols must be greater than zero".to_string(),
            ));
        }

        let file = TokioFile::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .await
            .map_err(|e| {
                OutOfCoreError::IoError(std::io::Error::new(
                    e.kind(),
                    format!("Failed to create {}: {e}", path.display()),
                ))
            })?;

        // Write a placeholder header; we will overwrite it in `finalize`.
        let placeholder_header = FileHeader {
            element_size: std::mem::size_of::<F>() as u32,
            num_cols: num_cols as u64,
            total_rows: 0,
            chunk_rows: 0,
            num_chunks: 0,
        };
        let mut writer = BufWriter::new(file);
        writer
            .write_all(&placeholder_header.to_bytes())
            .await
            .map_err(|e| {
                OutOfCoreError::IoError(std::io::Error::new(
                    e.kind(),
                    format!("Failed to write placeholder header: {e}"),
                ))
            })?;

        Ok(Self {
            file_path: path.to_path_buf(),
            num_cols,
            chunk_rows: 0,
            total_rows_written: 0,
            chunks_written: 0,
            file: writer,
            _phantom: PhantomData,
        })
    }

    /// Append a chunk to the file asynchronously.
    ///
    /// # Errors
    ///
    /// Returns [`OutOfCoreError::InvalidChunkSize`] when `chunk.ncols()` does
    /// not match the column count supplied at construction time.
    pub async fn write_chunk(&mut self, chunk: &Array2<F>) -> Result<(), OutOfCoreError> {
        if chunk.ncols() != self.num_cols {
            return Err(OutOfCoreError::InvalidChunkSize(format!(
                "Chunk has {} columns but writer expects {}",
                chunk.ncols(),
                self.num_cols
            )));
        }
        if chunk.nrows() == 0 {
            return Ok(());
        }

        // Record the chunk size from the first real write.
        if self.chunks_written == 0 {
            self.chunk_rows = chunk.nrows();
        }

        // Ensure the ndarray is in standard (C / row-major) layout before
        // taking a raw byte slice.
        let flat: ndarray::CowArray<F, ndarray::Ix1>;
        let raw_slice: &[F] = if chunk.is_standard_layout() {
            // SAFETY: Array2 in row-major layout has a contiguous memory
            // region spanning all elements.
            chunk.as_slice().ok_or_else(|| {
                OutOfCoreError::SerializationError(
                    "Array2 is standard layout but as_slice() returned None".to_string(),
                )
            })?
        } else {
            // Fall back to a contiguous owned copy.
            flat = chunk
                .view()
                .into_dyn()
                .into_dimensionality::<ndarray::IxDyn>()
                .map_err(|e| {
                    OutOfCoreError::SerializationError(format!("Dimensionality error: {e}"))
                })?
                .into_shape_with_order(chunk.len())
                .map_err(|e| OutOfCoreError::SerializationError(format!("Reshape error: {e}")))?
                .into();
            flat.as_slice().ok_or_else(|| {
                OutOfCoreError::SerializationError(
                    "Contiguous copy as_slice() returned None".to_string(),
                )
            })?
        };

        // SAFETY: F is Copy + Default (i.e. a primitive numeric type) so its
        // bit representation is always valid bytes.
        let raw_bytes: &[u8] = unsafe { slice_to_bytes(raw_slice) };
        self.file.write_all(raw_bytes).await.map_err(|e| {
            OutOfCoreError::IoError(std::io::Error::new(
                e.kind(),
                format!("Failed to write chunk {}: {e}", self.chunks_written),
            ))
        })?;

        self.total_rows_written += chunk.nrows();
        self.chunks_written += 1;
        Ok(())
    }

    /// Flush internal buffers to the OS without updating the file header.
    pub async fn flush(&mut self) -> Result<(), OutOfCoreError> {
        self.file.flush().await.map_err(|e| {
            OutOfCoreError::IoError(std::io::Error::new(e.kind(), format!("Flush failed: {e}")))
        })
    }

    /// Flush all pending writes **and** rewrite the header with the final row
    /// count so that [`AsyncChunkLoader`] can read the file correctly.
    ///
    /// This method should be called exactly once when all chunks have been
    /// written.  The writer is consumed to prevent further writes.
    pub async fn finalize(mut self) -> Result<(), OutOfCoreError> {
        // Flush buffered data to the OS.
        self.file.flush().await.map_err(|e| {
            OutOfCoreError::IoError(std::io::Error::new(
                e.kind(),
                format!("Final flush failed: {e}"),
            ))
        })?;

        // Unwrap the BufWriter to get the underlying TokioFile so we can seek
        // back to the header position.
        let mut raw_file = self.file.into_inner();
        raw_file.seek(SeekFrom::Start(0)).await.map_err(|e| {
            OutOfCoreError::IoError(std::io::Error::new(
                e.kind(),
                format!("Seek to header failed: {e}"),
            ))
        })?;

        let num_chunks = if self.chunk_rows == 0 {
            0u64
        } else {
            self.total_rows_written.div_ceil(self.chunk_rows) as u64
        };

        let final_header = FileHeader {
            element_size: std::mem::size_of::<F>() as u32,
            num_cols: self.num_cols as u64,
            total_rows: self.total_rows_written as u64,
            chunk_rows: self.chunk_rows as u64,
            num_chunks,
        };
        raw_file
            .write_all(&final_header.to_bytes())
            .await
            .map_err(|e| {
                OutOfCoreError::IoError(std::io::Error::new(
                    e.kind(),
                    format!("Failed to rewrite final header: {e}"),
                ))
            })?;

        raw_file.flush().await.map_err(|e| {
            OutOfCoreError::IoError(std::io::Error::new(
                e.kind(),
                format!("Post-header flush failed: {e}"),
            ))
        })
    }

    /// Number of rows written so far.
    pub fn total_rows_written(&self) -> usize {
        self.total_rows_written
    }

    /// Number of chunks written so far.
    pub fn chunks_written(&self) -> usize {
        self.chunks_written
    }
}

// ============================================================================
// async_map_chunks
// ============================================================================

/// Apply a function to each chunk of an input file, writing results to an
/// output file.
///
/// Reading and writing are overlapped with CPU-bound work using
/// `tokio::task::spawn_blocking`.  The transformation `map_fn` receives an
/// owned `Array2<F>` and must return something that can be converted into an
/// `Array2<F>` (most naturally another `Array2<F>`).
///
/// # Parameters
///
/// - `input_path`  – Path to a file produced by [`AsyncChunkWriter`].
/// - `output_path` – Destination file (created or truncated).
/// - `chunk_size`  – Number of rows per chunk (0 = use file's chunk size).
/// - `map_fn`      – Transformation applied to each chunk.
///
/// # Errors
///
/// Propagates any I/O or deserialization errors from the underlying reader or
/// writer.
///
/// # Example
///
/// ```rust,no_run
/// # #[cfg(all(feature = "async", not(target_arch = "wasm32")))]
/// # {
/// use scirs2_core::memory::async_out_of_core::{async_map_chunks, AsyncChunkWriter};
/// use ndarray::Array2;
/// use std::path::Path;
///
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// // Double every element
/// async_map_chunks::<f64, _, _>(
///     Path::new("/tmp/input.aoc"),
///     Path::new("/tmp/output.aoc"),
///     256,
///     |chunk| chunk.mapv(|x| x * 2.0),
/// )
/// .await?;
/// # Ok(())
/// # }
/// # }
/// ```
pub async fn async_map_chunks<F, G, R>(
    input_path: &Path,
    output_path: &Path,
    chunk_size: usize,
    map_fn: G,
) -> Result<(), OutOfCoreError>
where
    F: Copy + Default + Send + Sync + 'static,
    G: Fn(Array2<F>) -> R + Send + Sync + 'static,
    R: Into<Array2<F>> + Send + 'static,
{
    let mut loader = AsyncChunkLoader::<F>::open(input_path, chunk_size).await?;

    // We need `num_cols` before creating the writer.
    let num_cols = loader.num_cols();
    let mut writer = AsyncChunkWriter::<F>::create(output_path, num_cols).await?;

    // Wrap map_fn in an Arc so it can be moved into spawn_blocking closures.
    let map_fn = std::sync::Arc::new(map_fn);

    while let Some(chunk) = loader.next_chunk().await? {
        let map_clone = std::sync::Arc::clone(&map_fn);

        // Offload the CPU-bound transformation to a blocking thread.
        let transformed: Array2<F> = tokio::task::spawn_blocking(move || {
            let result: R = map_clone(chunk);
            result.into()
        })
        .await
        .map_err(|e| OutOfCoreError::SerializationError(format!("map_fn task panicked: {e}")))?;

        writer.write_chunk(&transformed).await?;
    }

    writer.finalize().await
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn make_array(rows: usize, cols: usize, base: f64) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |(r, c)| base + r as f64 * 100.0 + c as f64)
    }

    // ------------------------------------------------------------------
    // Basic write → read round-trip
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_write_read_roundtrip_f64() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_async_ooc_roundtrip_f64.aoc");

        let chunk1 = make_array(100, 4, 0.0);
        let chunk2 = make_array(50, 4, 1000.0);

        // Write
        {
            let mut writer = AsyncChunkWriter::<f64>::create(&path, 4)
                .await
                .expect("create writer");
            writer.write_chunk(&chunk1).await.expect("write chunk1");
            writer.write_chunk(&chunk2).await.expect("write chunk2");
            writer.finalize().await.expect("finalize");
        }

        // Read back
        {
            let mut loader = AsyncChunkLoader::<f64>::open(&path, 100)
                .await
                .expect("open loader");

            assert_eq!(loader.total_rows(), 150);
            assert_eq!(loader.num_cols(), 4);
            assert_eq!(loader.num_chunks(), 2);

            let got1 = loader
                .next_chunk()
                .await
                .expect("read chunk1")
                .expect("Some");
            assert_eq!(got1.shape(), &[100, 4]);
            assert!((got1[[0, 0]] - chunk1[[0, 0]]).abs() < 1e-12);
            assert!((got1[[99, 3]] - chunk1[[99, 3]]).abs() < 1e-12);

            let got2 = loader
                .next_chunk()
                .await
                .expect("read chunk2")
                .expect("Some");
            assert_eq!(got2.shape(), &[50, 4]);
            assert!((got2[[0, 0]] - chunk2[[0, 0]]).abs() < 1e-12);
            assert!((got2[[49, 3]] - chunk2[[49, 3]]).abs() < 1e-12);

            let eof = loader.next_chunk().await.expect("eof check");
            assert!(eof.is_none());
        }

        let _ = std::fs::remove_file(&path);
    }

    // ------------------------------------------------------------------
    // Write → read with f32
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_write_read_roundtrip_f32() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_async_ooc_roundtrip_f32.aoc");

        let data: Array2<f32> = Array2::from_shape_fn((200, 8), |(r, c)| r as f32 + c as f32 * 0.1);

        {
            let mut writer = AsyncChunkWriter::<f32>::create(&path, 8)
                .await
                .expect("create");
            writer.write_chunk(&data).await.expect("write");
            writer.finalize().await.expect("finalize");
        }

        {
            let mut loader = AsyncChunkLoader::<f32>::open(&path, 0).await.expect("open");
            // chunk_size == 0 → use file's chunk size which equals the one
            // chunk we wrote (200 rows)
            let got = loader.next_chunk().await.expect("read").expect("Some");
            assert_eq!(got.shape(), data.shape());
            for r in 0..200 {
                for c in 0..8 {
                    assert!((got[[r, c]] - data[[r, c]]).abs() < 1e-5);
                }
            }
        }

        let _ = std::fs::remove_file(&path);
    }

    // ------------------------------------------------------------------
    // Partial last chunk
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_partial_last_chunk() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_async_ooc_partial.aoc");

        // 350 rows, chunk_size = 100 → 3 full + 1 partial (50 rows)
        let all_data: Array2<f64> = Array2::from_shape_fn((350, 3), |(r, c)| r as f64 + c as f64);

        {
            let mut writer = AsyncChunkWriter::<f64>::create(&path, 3)
                .await
                .expect("create");
            // Write 100 rows at a time
            for start in (0..350).step_by(100) {
                let end = (start + 100).min(350);
                let slice = all_data.slice(ndarray::s![start..end, ..]);
                writer.write_chunk(&slice.to_owned()).await.expect("write");
            }
            writer.finalize().await.expect("finalize");
        }

        {
            let mut loader = AsyncChunkLoader::<f64>::open(&path, 100)
                .await
                .expect("open");
            assert_eq!(loader.total_rows(), 350);
            assert_eq!(loader.num_chunks(), 4);

            let mut reconstructed: Vec<Array2<f64>> = Vec::new();
            while let Some(c) = loader.next_chunk().await.expect("read chunk") {
                reconstructed.push(c);
            }
            assert_eq!(reconstructed.len(), 4);
            assert_eq!(reconstructed[3].nrows(), 50);

            // Verify a sample value
            let expected = all_data[[149, 2]];
            let got = reconstructed[1][[49, 2]];
            assert!((got - expected).abs() < 1e-12);
        }

        let _ = std::fs::remove_file(&path);
    }

    // ------------------------------------------------------------------
    // reset() allows re-streaming
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_reset() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_async_ooc_reset.aoc");

        let data: Array2<f64> = Array2::from_elem((200, 5), 7.0);
        {
            let mut w = AsyncChunkWriter::<f64>::create(&path, 5)
                .await
                .expect("create");
            w.write_chunk(&data).await.expect("write");
            w.finalize().await.expect("finalize");
        }

        let mut loader = AsyncChunkLoader::<f64>::open(&path, 100)
            .await
            .expect("open");

        // First pass
        let c1 = loader
            .next_chunk()
            .await
            .expect("pass1 chunk1")
            .expect("Some");
        let c2 = loader
            .next_chunk()
            .await
            .expect("pass1 chunk2")
            .expect("Some");
        assert_eq!(c1.nrows() + c2.nrows(), 200);
        assert!(loader.next_chunk().await.expect("pass1 eof").is_none());

        // Reset and stream again
        loader.reset();
        let r1 = loader
            .next_chunk()
            .await
            .expect("pass2 chunk1")
            .expect("Some");
        assert_eq!(r1.nrows(), c1.nrows());
        assert!((r1[[0, 0]] - 7.0).abs() < 1e-12);

        let _ = std::fs::remove_file(&path);
    }

    // ------------------------------------------------------------------
    // async_map_chunks
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_async_map_chunks_doubles_values() {
        let dir = std::env::temp_dir();
        let input = dir.join("scirs2_async_ooc_map_in.aoc");
        let output = dir.join("scirs2_async_ooc_map_out.aoc");

        // Create input file with known values
        {
            let mut w = AsyncChunkWriter::<f64>::create(&input, 4)
                .await
                .expect("create input");
            for chunk_idx in 0..3u32 {
                let arr = Array2::from_elem((100, 4), chunk_idx as f64);
                w.write_chunk(&arr).await.expect("write");
            }
            w.finalize().await.expect("finalize input");
        }

        // Apply doubling transformation
        async_map_chunks::<f64, _, _>(&input, &output, 100, |chunk| chunk.mapv(|x| x * 2.0))
            .await
            .expect("map_chunks");

        // Verify output
        {
            let mut loader = AsyncChunkLoader::<f64>::open(&output, 100)
                .await
                .expect("open output");
            assert_eq!(loader.total_rows(), 300);

            let chunk0 = loader.next_chunk().await.expect("chunk0").expect("Some");
            let chunk1 = loader.next_chunk().await.expect("chunk1").expect("Some");
            let chunk2 = loader.next_chunk().await.expect("chunk2").expect("Some");

            assert!((chunk0[[0, 0]] - 0.0).abs() < 1e-12);
            assert!((chunk1[[0, 0]] - 2.0).abs() < 1e-12);
            assert!((chunk2[[0, 0]] - 4.0).abs() < 1e-12);
        }

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    // ------------------------------------------------------------------
    // prefetch returns correct data
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_prefetch() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_async_ooc_prefetch.aoc");

        {
            let mut w = AsyncChunkWriter::<f64>::create(&path, 2)
                .await
                .expect("create");
            for i in 0u32..4 {
                let arr = Array2::from_elem((50, 2), i as f64 * 10.0);
                w.write_chunk(&arr).await.expect("write");
            }
            w.finalize().await.expect("finalize");
        }

        let loader = AsyncChunkLoader::<f64>::open(&path, 50)
            .await
            .expect("open");
        assert_eq!(loader.num_chunks(), 4);

        // Prefetch next 3 chunks (indices 0, 1, 2)
        let handles = loader.prefetch(3).await;
        assert_eq!(handles.len(), 3);

        let chunks: Vec<Array2<f64>> = {
            let mut v = Vec::new();
            for h in handles {
                v.push(h.await.expect("join").expect("chunk"));
            }
            v
        };

        assert!((chunks[0][[0, 0]] - 0.0).abs() < 1e-12);
        assert!((chunks[1][[0, 0]] - 10.0).abs() < 1e-12);
        assert!((chunks[2][[0, 0]] - 20.0).abs() < 1e-12);

        let _ = std::fs::remove_file(&path);
    }

    // ------------------------------------------------------------------
    // Error: column count mismatch
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_write_wrong_col_count() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_async_ooc_col_mismatch.aoc");

        let mut w = AsyncChunkWriter::<f64>::create(&path, 4)
            .await
            .expect("create");

        let wrong = Array2::from_elem((10, 8), 0.0_f64); // 8 cols != 4
        let result = w.write_chunk(&wrong).await;
        assert!(result.is_err());

        let _ = std::fs::remove_file(&path);
    }

    // ------------------------------------------------------------------
    // Error: element type mismatch on open
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_open_wrong_element_type() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_async_ooc_type_mismatch.aoc");

        // Write as f64
        {
            let mut w = AsyncChunkWriter::<f64>::create(&path, 2)
                .await
                .expect("create");
            let arr = Array2::from_elem((10, 2), 1.0_f64);
            w.write_chunk(&arr).await.expect("write");
            w.finalize().await.expect("finalize");
        }

        // Open as f32 → should fail
        let result = AsyncChunkLoader::<f32>::open(&path, 10).await;
        assert!(result.is_err());

        let _ = std::fs::remove_file(&path);
    }

    // ------------------------------------------------------------------
    // Writer statistics
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_writer_statistics() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_async_ooc_stats.aoc");

        let mut w = AsyncChunkWriter::<f64>::create(&path, 3)
            .await
            .expect("create");

        assert_eq!(w.chunks_written(), 0);
        assert_eq!(w.total_rows_written(), 0);

        let arr = Array2::from_elem((100, 3), 0.0_f64);
        w.write_chunk(&arr).await.expect("write 1");
        assert_eq!(w.chunks_written(), 1);
        assert_eq!(w.total_rows_written(), 100);

        w.write_chunk(&arr).await.expect("write 2");
        assert_eq!(w.chunks_written(), 2);
        assert_eq!(w.total_rows_written(), 200);

        w.finalize().await.expect("finalize");

        let _ = std::fs::remove_file(&path);
    }
}
