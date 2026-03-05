//! Memory-Mapped Array Wrapper for SciRS2
//!
//! Provides a generic `MmapArray<T>` that wraps a memory-mapped file-backed array
//! with support for read-only, read-write, and copy-on-write (COW) semantics.
//!
//! # Features
//!
//! - Zero-copy views into memory-mapped regions
//! - COW semantics for safe mutation without affecting the underlying file
//! - Typed access with proper alignment checking
//! - Integration with ndarray for scientific computing
//!
//! # Example
//!
//! ```rust,no_run
//! # #[cfg(feature = "memory_efficient")]
//! # {
//! use scirs2_core::memory_efficient::mmap_array::{MmapArray, MmapMode};
//! use std::path::Path;
//!
//! // Create from existing data
//! let data = vec![1.0f64, 2.0, 3.0, 4.0];
//! let path = Path::new("/tmp/test_array.dat");
//! let arr = MmapArray::<f64>::from_slice(&data, path, MmapMode::ReadWrite)
//!     .expect("Failed to create mmap array");
//!
//! // Access as a slice (zero-copy)
//! let slice = arr.as_slice().expect("Failed to get slice");
//! assert_eq!(slice.len(), 4);
//! # }
//! ```

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
use memmap2::{Mmap, MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

/// Mode for memory-mapped array access
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmapMode {
    /// Read-only access. Writes are not permitted.
    ReadOnly,
    /// Read-write access. Changes are persisted to the underlying file.
    ReadWrite,
    /// Copy-on-write access. Reads come from the file, but writes go to
    /// a private copy in memory and are NOT persisted to the file.
    CopyOnWrite,
}

impl MmapMode {
    /// Returns a human-readable description of the mode
    pub const fn description(&self) -> &'static str {
        match self {
            MmapMode::ReadOnly => "read-only",
            MmapMode::ReadWrite => "read-write",
            MmapMode::CopyOnWrite => "copy-on-write",
        }
    }
}

/// A file header stored at the beginning of MmapArray files.
/// This allows reopening files with proper type and size information.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MmapArrayHeader {
    /// Magic bytes for file identification: "SCI2MMAP"
    magic: [u8; 8],
    /// Version of the file format
    version: u32,
    /// Size of each element in bytes
    element_size: u32,
    /// Total number of elements
    num_elements: u64,
    /// Alignment requirement for elements
    alignment: u32,
    /// Reserved for future use
    _reserved: [u8; 36],
}

impl MmapArrayHeader {
    const MAGIC: [u8; 8] = *b"SCI2MMAP";
    const VERSION: u32 = 1;
    const HEADER_SIZE: usize = std::mem::size_of::<Self>();

    fn new<T>(num_elements: usize) -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            element_size: std::mem::size_of::<T>() as u32,
            num_elements: num_elements as u64,
            alignment: std::mem::align_of::<T>() as u32,
            _reserved: [0u8; 36],
        }
    }

    fn validate<T>(&self) -> CoreResult<()> {
        if self.magic != Self::MAGIC {
            return Err(CoreError::ValidationError(
                ErrorContext::new("Invalid magic bytes in MmapArray file".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
        if self.version != Self::VERSION {
            return Err(CoreError::ValidationError(
                ErrorContext::new(format!(
                    "Unsupported MmapArray file version: {} (expected {})",
                    self.version,
                    Self::VERSION
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
        if self.element_size != std::mem::size_of::<T>() as u32 {
            return Err(CoreError::ValidationError(
                ErrorContext::new(format!(
                    "Element size mismatch: file has {} bytes, expected {} bytes for type {}",
                    self.element_size,
                    std::mem::size_of::<T>(),
                    std::any::type_name::<T>()
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
        if self.alignment != std::mem::align_of::<T>() as u32 {
            return Err(CoreError::ValidationError(
                ErrorContext::new(format!(
                    "Alignment mismatch: file requires {} bytes, type {} requires {} bytes",
                    self.alignment,
                    std::any::type_name::<T>(),
                    std::mem::align_of::<T>()
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
        Ok(())
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_bytes(&self) -> &[u8] {
        // SAFETY: MmapArrayHeader is repr(C) and contains only plain data types
        unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }

    fn from_bytes(bytes: &[u8]) -> CoreResult<&Self> {
        if bytes.len() < std::mem::size_of::<Self>() {
            return Err(CoreError::ValidationError(
                ErrorContext::new(format!(
                    "File too small for header: {} bytes (need at least {})",
                    bytes.len(),
                    std::mem::size_of::<Self>()
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
        let ptr = bytes.as_ptr() as *const Self;
        if (ptr as usize) % std::mem::align_of::<Self>() != 0 {
            // Copy to aligned buffer if unaligned
            return Err(CoreError::MemoryError(
                ErrorContext::new("Header data is not properly aligned".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
        // SAFETY: We checked size and alignment above
        Ok(unsafe { &*ptr })
    }
}

/// Internal storage for the memory map
enum MmapStorage {
    /// Read-only memory map
    ReadOnly(Mmap),
    /// Mutable memory map (read-write or COW)
    Mutable(MmapMut),
}

/// A generic memory-mapped array backed by a file.
///
/// `MmapArray<T>` provides zero-copy access to array data stored in a file,
/// using the operating system's virtual memory system to page data in and out
/// as needed.
///
/// # Type Requirements
///
/// `T` must be `Copy + Send + Sync + 'static` to ensure safe memory-mapped access.
/// The type must also have a fixed, well-defined memory layout (no padding concerns
/// for single elements).
///
/// # Thread Safety
///
/// `MmapArray` is `Send + Sync` and can be shared across threads. However,
/// concurrent mutation (with `ReadWrite` mode) requires external synchronization.
pub struct MmapArray<T: Copy + Send + Sync + 'static> {
    /// The underlying memory map
    storage: MmapStorage,
    /// Path to the backing file
    file_path: PathBuf,
    /// Access mode
    mode: MmapMode,
    /// Number of elements of type T
    num_elements: usize,
    /// Data offset (after header) in bytes
    data_offset: usize,
    /// Whether the COW copy has been materialized
    cow_materialized: AtomicBool,
    /// COW buffer (populated lazily on first write in COW mode)
    cow_buffer: std::sync::Mutex<Option<Vec<T>>>,
    /// Phantom type marker
    _phantom: PhantomData<T>,
}

// SAFETY: MmapArray is safe to send/share because:
// 1. T: Send + Sync
// 2. Mmap/MmapMut are Send + Sync
// 3. Internal mutation is protected by Mutex
unsafe impl<T: Copy + Send + Sync + 'static> Send for MmapArray<T> {}
unsafe impl<T: Copy + Send + Sync + 'static> Sync for MmapArray<T> {}

impl<T: Copy + Send + Sync + 'static> MmapArray<T> {
    /// Create a new `MmapArray` from a slice of data, writing it to the specified path.
    ///
    /// This creates a new file (or truncates an existing one) and writes
    /// the header and data to it.
    pub fn from_slice(data: &[T], path: &Path, mode: MmapMode) -> CoreResult<Self> {
        let num_elements = data.len();
        let element_size = std::mem::size_of::<T>();

        if element_size == 0 {
            return Err(CoreError::InvalidArgument(
                ErrorContext::new("Zero-sized types are not supported for MmapArray".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Write header and data to file
        let header = MmapArrayHeader::new::<T>(num_elements);
        let data_bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|e| {
                CoreError::IoError(
                    ErrorContext::new(format!("Failed to create file {}: {e}", path.display()))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;

        file.write_all(header.to_bytes()).map_err(|e| {
            CoreError::IoError(
                ErrorContext::new(format!("Failed to write header: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
        file.write_all(data_bytes).map_err(|e| {
            CoreError::IoError(
                ErrorContext::new(format!("Failed to write data: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
        file.flush().map_err(|e| {
            CoreError::IoError(
                ErrorContext::new(format!("Failed to flush file: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        // Now memory-map the file
        Self::open(path, mode)
    }

    /// Open an existing `MmapArray` file.
    ///
    /// The file must have been created by `MmapArray::from_slice` or
    /// `MmapArray::from_ndarray`. The type parameter `T` must match the
    /// type used when the file was created.
    pub fn open(path: &Path, mode: MmapMode) -> CoreResult<Self> {
        let element_size = std::mem::size_of::<T>();
        if element_size == 0 {
            return Err(CoreError::InvalidArgument(
                ErrorContext::new("Zero-sized types are not supported for MmapArray".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        let file = match mode {
            MmapMode::ReadOnly | MmapMode::CopyOnWrite => File::open(path).map_err(|e| {
                CoreError::IoError(
                    ErrorContext::new(format!(
                        "Failed to open file {} for reading: {e}",
                        path.display()
                    ))
                    .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?,
            MmapMode::ReadWrite => OpenOptions::new()
                .read(true)
                .write(true)
                .open(path)
                .map_err(|e| {
                    CoreError::IoError(
                        ErrorContext::new(format!(
                            "Failed to open file {} for read-write: {e}",
                            path.display()
                        ))
                        .with_location(ErrorLocation::new(file!(), line!())),
                    )
                })?,
        };

        let file_len = file
            .metadata()
            .map_err(|e| {
                CoreError::IoError(
                    ErrorContext::new(format!("Failed to get file metadata: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?
            .len() as usize;

        if file_len < MmapArrayHeader::HEADER_SIZE {
            return Err(CoreError::ValidationError(
                ErrorContext::new(format!(
                    "File is too small ({} bytes) to contain a valid MmapArray header ({} bytes required)",
                    file_len,
                    MmapArrayHeader::HEADER_SIZE
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Memory-map the entire file
        let storage = match mode {
            MmapMode::ReadOnly => {
                let mmap = unsafe {
                    MmapOptions::new().map(&file).map_err(|e| {
                        CoreError::IoError(
                            ErrorContext::new(format!("Failed to create read-only mmap: {e}"))
                                .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    })?
                };
                MmapStorage::ReadOnly(mmap)
            }
            MmapMode::ReadWrite => {
                let mmap = unsafe {
                    MmapOptions::new().map_mut(&file).map_err(|e| {
                        CoreError::IoError(
                            ErrorContext::new(format!("Failed to create read-write mmap: {e}"))
                                .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    })?
                };
                MmapStorage::Mutable(mmap)
            }
            MmapMode::CopyOnWrite => {
                let mmap = unsafe {
                    MmapOptions::new().map_copy(&file).map_err(|e| {
                        CoreError::IoError(
                            ErrorContext::new(format!("Failed to create COW mmap: {e}"))
                                .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    })?
                };
                MmapStorage::Mutable(mmap)
            }
        };

        // Read and validate header
        let raw_bytes = match &storage {
            MmapStorage::ReadOnly(mmap) => &mmap[..],
            MmapStorage::Mutable(mmap) => &mmap[..],
        };
        let header = MmapArrayHeader::from_bytes(raw_bytes)?;
        header.validate::<T>()?;

        let num_elements = header.num_elements as usize;
        let data_offset = MmapArrayHeader::HEADER_SIZE;
        let expected_data_size = num_elements * element_size;

        if file_len < data_offset + expected_data_size {
            return Err(CoreError::ValidationError(
                ErrorContext::new(format!(
                    "File too small: {} bytes, need {} bytes (header: {}, data: {} elements * {} bytes)",
                    file_len,
                    data_offset + expected_data_size,
                    data_offset,
                    num_elements,
                    element_size
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Validate data alignment
        let data_ptr = match &storage {
            MmapStorage::ReadOnly(mmap) => mmap[data_offset..].as_ptr(),
            MmapStorage::Mutable(mmap) => mmap[data_offset..].as_ptr(),
        };
        if (data_ptr as usize) % std::mem::align_of::<T>() != 0 {
            return Err(CoreError::MemoryError(
                ErrorContext::new(format!(
                    "Memory-mapped data is not properly aligned for type {} (alignment: {}, address: 0x{:x})",
                    std::any::type_name::<T>(),
                    std::mem::align_of::<T>(),
                    data_ptr as usize
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        Ok(Self {
            storage,
            file_path: path.to_path_buf(),
            mode,
            num_elements,
            data_offset,
            cow_materialized: AtomicBool::new(false),
            cow_buffer: std::sync::Mutex::new(None),
            _phantom: PhantomData,
        })
    }

    /// Create a new `MmapArray` from an ndarray Array1.
    pub fn from_ndarray(
        array: &::ndarray::Array1<T>,
        path: &Path,
        mode: MmapMode,
    ) -> CoreResult<Self> {
        let slice = array.as_slice().ok_or_else(|| {
            CoreError::InvalidArgument(
                ErrorContext::new("Array must be contiguous for memory mapping".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
        Self::from_slice(slice, path, mode)
    }

    /// Get the number of elements in the array
    pub fn len(&self) -> usize {
        self.num_elements
    }

    /// Check if the array is empty
    pub fn is_empty(&self) -> bool {
        self.num_elements == 0
    }

    /// Get the size in bytes of the data portion (excluding header)
    pub fn data_size_bytes(&self) -> usize {
        self.num_elements * std::mem::size_of::<T>()
    }

    /// Get the total file size in bytes (header + data)
    pub fn file_size_bytes(&self) -> usize {
        self.data_offset + self.data_size_bytes()
    }

    /// Get the access mode
    pub fn mode(&self) -> MmapMode {
        self.mode
    }

    /// Get the file path
    pub fn path(&self) -> &Path {
        &self.file_path
    }

    /// Get a zero-copy immutable view of the data as a slice.
    ///
    /// For COW mode, if the buffer has been materialized (i.e., a write
    /// occurred), this returns the COW buffer instead of the mmap data.
    pub fn as_slice(&self) -> CoreResult<&[T]> {
        if self.mode == MmapMode::CopyOnWrite && self.cow_materialized.load(Ordering::Acquire) {
            let guard = self.cow_buffer.lock().map_err(|e| {
                CoreError::MemoryError(
                    ErrorContext::new(format!("Failed to lock COW buffer: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;
            if let Some(ref buf) = *guard {
                // SAFETY: We return a reference to the Vec's contents.
                // The Vec is behind a Mutex and won't be dropped while this reference exists
                // because the caller holds a reference to self.
                // This is safe as long as the caller doesn't mutate through as_slice_mut simultaneously.
                let ptr = buf.as_ptr();
                let len = buf.len();
                return Ok(unsafe { std::slice::from_raw_parts(ptr, len) });
            }
        }
        self.raw_data_slice()
    }

    /// Get a zero-copy mutable view of the data as a mutable slice.
    ///
    /// # Errors
    ///
    /// Returns an error if the array is in ReadOnly mode.
    ///
    /// For COW mode, the first call to this method materializes the COW buffer
    /// by copying the mapped data. Subsequent calls return the COW buffer directly.
    pub fn as_slice_mut(&mut self) -> CoreResult<&mut [T]> {
        match self.mode {
            MmapMode::ReadOnly => Err(CoreError::InvalidArgument(
                ErrorContext::new("Cannot get mutable slice for read-only MmapArray".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            )),
            MmapMode::ReadWrite => {
                match &self.storage {
                    MmapStorage::Mutable(mmap) => {
                        let ptr = mmap[self.data_offset..].as_ptr() as *mut T;
                        // SAFETY: We checked mode is ReadWrite, the data is aligned,
                        // and the mmap is mutable.
                        Ok(unsafe { std::slice::from_raw_parts_mut(ptr, self.num_elements) })
                    }
                    MmapStorage::ReadOnly(_) => Err(CoreError::MemoryError(
                        ErrorContext::new(
                            "Internal error: ReadWrite mode but storage is ReadOnly".to_string(),
                        )
                        .with_location(ErrorLocation::new(file!(), line!())),
                    )),
                }
            }
            MmapMode::CopyOnWrite => {
                self.materialize_cow()?;
                let mut guard = self.cow_buffer.lock().map_err(|e| {
                    CoreError::MemoryError(
                        ErrorContext::new(format!("Failed to lock COW buffer: {e}"))
                            .with_location(ErrorLocation::new(file!(), line!())),
                    )
                })?;
                if let Some(ref mut buf) = *guard {
                    let ptr = buf.as_mut_ptr();
                    let len = buf.len();
                    // SAFETY: We own the buffer and are returning a mutable reference.
                    Ok(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
                } else {
                    Err(CoreError::MemoryError(
                        ErrorContext::new(
                            "COW buffer not materialized despite materialize_cow call".to_string(),
                        )
                        .with_location(ErrorLocation::new(file!(), line!())),
                    ))
                }
            }
        }
    }

    /// Get a single element by index.
    pub fn get(&self, index: usize) -> CoreResult<T> {
        if index >= self.num_elements {
            return Err(CoreError::InvalidArgument(
                ErrorContext::new(format!(
                    "Index {} out of bounds for MmapArray of length {}",
                    index, self.num_elements
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
        let slice = self.as_slice()?;
        Ok(slice[index])
    }

    /// Set a single element by index.
    ///
    /// # Errors
    ///
    /// Returns an error if the array is in ReadOnly mode or the index is out of bounds.
    pub fn set(&mut self, index: usize, value: T) -> CoreResult<()> {
        if index >= self.num_elements {
            return Err(CoreError::InvalidArgument(
                ErrorContext::new(format!(
                    "Index {} out of bounds for MmapArray of length {}",
                    index, self.num_elements
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
        let slice = self.as_slice_mut()?;
        slice[index] = value;
        Ok(())
    }

    /// Convert to an ndarray Array1 by copying the data.
    pub fn to_ndarray(&self) -> CoreResult<::ndarray::Array1<T>> {
        let slice = self.as_slice()?;
        Ok(::ndarray::Array1::from_vec(slice.to_vec()))
    }

    /// Convert to an ndarray ArrayView1 (zero-copy).
    ///
    /// The returned view borrows from this MmapArray.
    pub fn as_ndarray_view(&self) -> CoreResult<::ndarray::ArrayView1<T>> {
        let slice = self.as_slice()?;
        Ok(::ndarray::ArrayView1::from(slice))
    }

    /// Flush changes to disk (only meaningful for ReadWrite mode).
    pub fn flush(&self) -> CoreResult<()> {
        match (&self.storage, self.mode) {
            (MmapStorage::Mutable(mmap), MmapMode::ReadWrite) => mmap.flush().map_err(|e| {
                CoreError::IoError(
                    ErrorContext::new(format!("Failed to flush mmap: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            }),
            _ => Ok(()), // No-op for read-only and COW modes
        }
    }

    /// Flush changes asynchronously (non-blocking).
    pub fn flush_async(&self) -> CoreResult<()> {
        match (&self.storage, self.mode) {
            (MmapStorage::Mutable(mmap), MmapMode::ReadWrite) => mmap.flush_async().map_err(|e| {
                CoreError::IoError(
                    ErrorContext::new(format!("Failed to async flush mmap: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            }),
            _ => Ok(()),
        }
    }

    /// Create a read-only view of a subrange of the array (zero-copy).
    pub fn view(&self, start: usize, len: usize) -> CoreResult<&[T]> {
        if start + len > self.num_elements {
            return Err(CoreError::InvalidArgument(
                ErrorContext::new(format!(
                    "View range [{}..{}] out of bounds for MmapArray of length {}",
                    start,
                    start + len,
                    self.num_elements
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
        let full_slice = self.as_slice()?;
        Ok(&full_slice[start..start + len])
    }

    /// Iterate over the elements of the array.
    pub fn iter(&self) -> CoreResult<std::slice::Iter<'_, T>> {
        let slice = self.as_slice()?;
        Ok(slice.iter())
    }

    /// Get raw pointer to the data (for advanced use).
    ///
    /// # Safety
    ///
    /// The returned pointer is only valid while `self` is alive and the
    /// underlying memory map is not remapped.
    pub fn as_ptr(&self) -> CoreResult<*const T> {
        let slice = self.as_slice()?;
        Ok(slice.as_ptr())
    }

    /// Apply a function to each element and return a new Vec.
    pub fn map<U, F>(&self, f: F) -> CoreResult<Vec<U>>
    where
        F: Fn(T) -> U,
    {
        let slice = self.as_slice()?;
        Ok(slice.iter().map(|&x| f(x)).collect())
    }

    /// Apply a function to chunks of elements.
    /// Useful for processing large arrays without loading everything at once.
    pub fn chunked_iter(&self, chunk_size: usize) -> CoreResult<MmapChunkIter<'_, T>> {
        if chunk_size == 0 {
            return Err(CoreError::InvalidArgument(
                ErrorContext::new("Chunk size must be greater than 0".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
        let slice = self.as_slice()?;
        Ok(MmapChunkIter {
            data: slice,
            chunk_size,
            position: 0,
        })
    }

    // --- Private helpers ---

    /// Get the raw data slice from the memory map
    fn raw_data_slice(&self) -> CoreResult<&[T]> {
        let raw_bytes = match &self.storage {
            MmapStorage::ReadOnly(mmap) => &mmap[self.data_offset..],
            MmapStorage::Mutable(mmap) => &mmap[self.data_offset..],
        };
        let ptr = raw_bytes.as_ptr() as *const T;

        // Validate alignment
        if (ptr as usize) % std::mem::align_of::<T>() != 0 {
            return Err(CoreError::MemoryError(
                ErrorContext::new(format!(
                    "Data pointer is not properly aligned for type {}",
                    std::any::type_name::<T>()
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // SAFETY: pointer is aligned, data is within the mmap bounds, and
        // we validated the number of elements during open/creation
        Ok(unsafe { std::slice::from_raw_parts(ptr, self.num_elements) })
    }

    /// Materialize the COW buffer by copying data from the mmap
    fn materialize_cow(&self) -> CoreResult<()> {
        if self.cow_materialized.load(Ordering::Acquire) {
            return Ok(()); // Already materialized
        }

        let source_slice = self.raw_data_slice()?;
        let buffer = source_slice.to_vec();

        let mut guard = self.cow_buffer.lock().map_err(|e| {
            CoreError::MemoryError(
                ErrorContext::new(format!("Failed to lock COW buffer: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
        if guard.is_none() {
            *guard = Some(buffer);
        }
        self.cow_materialized.store(true, Ordering::Release);
        Ok(())
    }
}

impl<T: Copy + Send + Sync + 'static> std::fmt::Debug for MmapArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmapArray")
            .field("path", &self.file_path)
            .field("mode", &self.mode)
            .field("num_elements", &self.num_elements)
            .field("element_size", &std::mem::size_of::<T>())
            .field("data_size_bytes", &self.data_size_bytes())
            .finish()
    }
}

/// Iterator over chunks of an MmapArray.
pub struct MmapChunkIter<'a, T> {
    data: &'a [T],
    chunk_size: usize,
    position: usize,
}

impl<'a, T> Iterator for MmapChunkIter<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.data.len() {
            return None;
        }
        let end = (self.position + self.chunk_size).min(self.data.len());
        let chunk = &self.data[self.position..end];
        self.position = end;
        Some(chunk)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.data.len().saturating_sub(self.position);
        let count = remaining.div_ceil(self.chunk_size);
        (count, Some(count))
    }
}

impl<'a, T> ExactSizeIterator for MmapChunkIter<'a, T> {}

/// Builder for creating MmapArray instances with configuration options.
pub struct MmapArrayBuilder<T: Copy + Send + Sync + 'static> {
    mode: MmapMode,
    _phantom: PhantomData<T>,
}

impl<T: Copy + Send + Sync + 'static> MmapArrayBuilder<T> {
    /// Create a new builder with the specified mode.
    pub fn new(mode: MmapMode) -> Self {
        Self {
            mode,
            _phantom: PhantomData,
        }
    }

    /// Build from a slice of data, writing to the specified path.
    pub fn from_slice(self, data: &[T], path: &Path) -> CoreResult<MmapArray<T>> {
        MmapArray::from_slice(data, path, self.mode)
    }

    /// Build by opening an existing file.
    pub fn open(self, path: &Path) -> CoreResult<MmapArray<T>> {
        MmapArray::open(path, self.mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_read_f64() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_mmap_test_f64.dat");

        let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.5).collect();
        let arr =
            MmapArray::from_slice(&data, &path, MmapMode::ReadOnly).expect("Failed to create");

        assert_eq!(arr.len(), 1000);
        assert!(!arr.is_empty());

        let slice = arr.as_slice().expect("Failed to get slice");
        for (i, &val) in slice.iter().enumerate() {
            let expected = i as f64 * 0.5;
            assert!(
                (val - expected).abs() < 1e-10,
                "Mismatch at index {i}: {val} vs {expected}"
            );
        }

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_create_and_read_i32() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_mmap_test_i32.dat");

        let data: Vec<i32> = (0..500).collect();
        let arr =
            MmapArray::from_slice(&data, &path, MmapMode::ReadOnly).expect("Failed to create");

        assert_eq!(arr.len(), 500);
        let val = arr.get(42).expect("Failed to get element");
        assert_eq!(val, 42);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_read_write_mode() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_mmap_test_rw.dat");

        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut arr =
            MmapArray::from_slice(&data, &path, MmapMode::ReadWrite).expect("Failed to create");

        // Mutate through the mmap
        arr.set(2, 99.0).expect("Failed to set element");
        arr.flush().expect("Failed to flush");

        // Verify by re-opening
        let arr2 = MmapArray::<f64>::open(&path, MmapMode::ReadOnly).expect("Failed to reopen");
        let val = arr2.get(2).expect("Failed to get element");
        assert!((val - 99.0).abs() < 1e-10);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_cow_mode() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_mmap_test_cow.dat");

        let data: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0];
        let _arr_rw =
            MmapArray::from_slice(&data, &path, MmapMode::ReadWrite).expect("Failed to create");
        drop(_arr_rw);

        // Open in COW mode
        let mut arr_cow =
            MmapArray::<f64>::open(&path, MmapMode::CopyOnWrite).expect("Failed to open COW");

        // Read should work
        let val = arr_cow.get(0).expect("Failed to get");
        assert!((val - 10.0).abs() < 1e-10);

        // Write should work (goes to COW buffer, not file)
        arr_cow.set(0, 999.0).expect("Failed to set COW");
        let val_cow = arr_cow.get(0).expect("Failed to get after COW write");
        assert!((val_cow - 999.0).abs() < 1e-10);

        // Original file should be unchanged
        let arr_verify =
            MmapArray::<f64>::open(&path, MmapMode::ReadOnly).expect("Failed to verify");
        let val_orig = arr_verify.get(0).expect("Failed to get original");
        assert!(
            (val_orig - 10.0).abs() < 1e-10,
            "COW should not have modified the original file"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_readonly_write_fails() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_mmap_test_ro_fail.dat");

        let data: Vec<f64> = vec![1.0, 2.0, 3.0];
        let mut arr =
            MmapArray::from_slice(&data, &path, MmapMode::ReadOnly).expect("Failed to create");

        let result = arr.set(0, 42.0);
        assert!(result.is_err(), "Write to read-only should fail");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_out_of_bounds() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_mmap_test_oob.dat");

        let data: Vec<f64> = vec![1.0, 2.0, 3.0];
        let arr =
            MmapArray::from_slice(&data, &path, MmapMode::ReadOnly).expect("Failed to create");

        let result = arr.get(100);
        assert!(result.is_err(), "Out of bounds get should fail");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_view() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_mmap_test_view.dat");

        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let arr =
            MmapArray::from_slice(&data, &path, MmapMode::ReadOnly).expect("Failed to create");

        let view = arr.view(10, 20).expect("Failed to get view");
        assert_eq!(view.len(), 20);
        assert!((view[0] - 10.0).abs() < 1e-10);
        assert!((view[19] - 29.0).abs() < 1e-10);

        // Invalid range should fail
        let result = arr.view(90, 20);
        assert!(result.is_err(), "Out of bounds view should fail");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_chunked_iter() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_mmap_test_chunks.dat");

        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let arr =
            MmapArray::from_slice(&data, &path, MmapMode::ReadOnly).expect("Failed to create");

        let chunks: Vec<&[f64]> = arr
            .chunked_iter(30)
            .expect("Failed to get chunks")
            .collect();

        assert_eq!(chunks.len(), 4); // 100 / 30 = 3 full + 1 partial
        assert_eq!(chunks[0].len(), 30);
        assert_eq!(chunks[1].len(), 30);
        assert_eq!(chunks[2].len(), 30);
        assert_eq!(chunks[3].len(), 10);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_to_ndarray() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_mmap_test_ndarray.dat");

        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let arr =
            MmapArray::from_slice(&data, &path, MmapMode::ReadOnly).expect("Failed to create");

        let nd = arr.to_ndarray().expect("Failed to convert to ndarray");
        assert_eq!(nd.len(), 5);
        assert!((nd[0] - 1.0).abs() < 1e-10);
        assert!((nd[4] - 5.0).abs() < 1e-10);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_ndarray_view() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_mmap_test_ndview.dat");

        let data: Vec<f64> = vec![10.0, 20.0, 30.0];
        let arr =
            MmapArray::from_slice(&data, &path, MmapMode::ReadOnly).expect("Failed to create");

        let view = arr.as_ndarray_view().expect("Failed to get ndarray view");
        assert_eq!(view.len(), 3);
        assert!((view[1] - 20.0).abs() < 1e-10);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_from_ndarray() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_mmap_test_fromnd.dat");

        let nd = ::ndarray::Array1::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let arr =
            MmapArray::from_ndarray(&nd, &path, MmapMode::ReadOnly).expect("Failed to create");

        assert_eq!(arr.len(), 4);
        let val = arr.get(3).expect("Failed to get");
        assert!((val - 4.0).abs() < 1e-10);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_map_function() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_mmap_test_map.dat");

        let data: Vec<f64> = vec![1.0, 4.0, 9.0, 16.0];
        let arr =
            MmapArray::from_slice(&data, &path, MmapMode::ReadOnly).expect("Failed to create");

        let sqrt_vals = arr.map(|x| x.sqrt()).expect("Failed to map");
        assert!((sqrt_vals[0] - 1.0).abs() < 1e-10);
        assert!((sqrt_vals[1] - 2.0).abs() < 1e-10);
        assert!((sqrt_vals[2] - 3.0).abs() < 1e-10);
        assert!((sqrt_vals[3] - 4.0).abs() < 1e-10);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_builder() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_mmap_test_builder.dat");

        let data: Vec<f64> = vec![100.0, 200.0];
        let builder = MmapArrayBuilder::<f64>::new(MmapMode::ReadOnly);
        let arr = builder.from_slice(&data, &path).expect("Failed to build");

        assert_eq!(arr.len(), 2);
        assert_eq!(arr.mode(), MmapMode::ReadOnly);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_empty_array() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_mmap_test_empty.dat");

        let data: Vec<f64> = vec![];
        let arr =
            MmapArray::from_slice(&data, &path, MmapMode::ReadOnly).expect("Failed to create");

        assert_eq!(arr.len(), 0);
        assert!(arr.is_empty());

        let slice = arr.as_slice().expect("Failed to get slice");
        assert!(slice.is_empty());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_flush_async() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_mmap_test_flush_async.dat");

        let data: Vec<f64> = vec![1.0, 2.0, 3.0];
        let mut arr =
            MmapArray::from_slice(&data, &path, MmapMode::ReadWrite).expect("Failed to create");

        arr.set(1, 42.0).expect("Failed to set");
        arr.flush_async().expect("Failed to async flush");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_debug_format() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_mmap_test_debug.dat");

        let data: Vec<f64> = vec![1.0];
        let arr =
            MmapArray::from_slice(&data, &path, MmapMode::ReadOnly).expect("Failed to create");

        let debug_str = format!("{:?}", arr);
        assert!(debug_str.contains("MmapArray"));
        assert!(debug_str.contains("num_elements"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_reopen_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_mmap_test_reopen.dat");

        // Create and write
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut arr1 =
            MmapArray::from_slice(&data, &path, MmapMode::ReadWrite).expect("Failed to create");
        arr1.set(3, 42.0).expect("Failed to set");
        arr1.flush().expect("Failed to flush");
        drop(arr1);

        // Reopen and verify
        let arr2 = MmapArray::<f64>::open(&path, MmapMode::ReadOnly).expect("Failed to reopen");
        assert_eq!(arr2.len(), 5);
        let val = arr2.get(3).expect("Failed to get");
        assert!((val - 42.0).abs() < 1e-10);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_u8_array() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_mmap_test_u8.dat");

        let data: Vec<u8> = (0..=255).collect();
        let arr =
            MmapArray::from_slice(&data, &path, MmapMode::ReadOnly).expect("Failed to create");

        assert_eq!(arr.len(), 256);
        let val = arr.get(128).expect("Failed to get");
        assert_eq!(val, 128u8);

        let _ = std::fs::remove_file(&path);
    }
}
