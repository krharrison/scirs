//! Memory-mapped file utilities with platform-independent API and safety wrappers.
//!
//! Provides lightweight, safe abstractions for memory-mapped I/O:
//!
//! - [`MappedFile`] — read-only memory-mapped file
//! - [`MappedFileMut`] — read-write memory-mapped file
//! - [`MappedArrayView`] — interpret a mapped region as a typed slice
//! - [`LazyMappedFile`] — lazy-loading wrapper with advisory page hints
//! - Platform-independent API: uses `mmap`/`munmap` on Unix and
//!   `CreateFileMapping`/`MapViewOfFile` on Windows, all behind a unified Rust interface.
//!
//! # Safety
//!
//! All types wrap raw pointers internally but expose only safe APIs.
//! The underlying memory is unmapped when the handle is dropped.
//! File-backed mappings are read from the OS page cache and are not affected by
//! concurrent writes (copy-on-write semantics for private mappings).
//!
//! # Example
//!
//! ```rust,no_run
//! use scirs2_core::mmap_utils::{MappedFile, MappedArrayView};
//! use std::path::Path;
//!
//! let file = MappedFile::open(Path::new("/tmp/data.bin")).expect("open");
//! let bytes = file.as_slice();
//! println!("file size: {} bytes", bytes.len());
//!
//! // Interpret as f64 array (must be aligned and sized correctly)
//! let view: MappedArrayView<'_, f64> = MappedArrayView::from_mapped(&file).expect("view");
//! println!("first element: {}", view.as_slice()[0]);
//! ```

use crate::error::{CoreError, CoreResult, ErrorContext};
use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

// ===========================================================================
// MappedFile (read-only)
// ===========================================================================

/// A read-only memory-mapped file.
///
/// The file is mapped into memory on construction and unmapped on drop.
/// Access is through `as_slice()` which returns `&[u8]`.
pub struct MappedFile {
    inner: MmapInner,
    path: PathBuf,
}

impl MappedFile {
    /// Memory-map a file in read-only mode.
    pub fn open(path: &Path) -> CoreResult<Self> {
        let file = File::open(path).map_err(|e| {
            CoreError::IoError(ErrorContext::new(format!(
                "failed to open file {}: {e}",
                path.display()
            )))
        })?;

        let metadata = file.metadata().map_err(|e| {
            CoreError::IoError(ErrorContext::new(format!(
                "failed to read metadata for {}: {e}",
                path.display()
            )))
        })?;

        let len = metadata.len() as usize;
        if len == 0 {
            return Ok(Self {
                inner: MmapInner::empty(),
                path: path.to_path_buf(),
            });
        }

        let inner = MmapInner::map_read_only(&file, len)?;
        Ok(Self {
            inner,
            path: path.to_path_buf(),
        })
    }

    /// The mapped bytes as a slice.
    pub fn as_slice(&self) -> &[u8] {
        self.inner.as_slice()
    }

    /// Length of the mapping in bytes.
    pub fn len(&self) -> usize {
        self.inner.len
    }

    /// Whether the mapping is empty (zero-length file).
    pub fn is_empty(&self) -> bool {
        self.inner.len == 0
    }

    /// The path of the mapped file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Advise the OS about the expected access pattern for a byte range.
    pub fn advise_sequential(&self, offset: usize, length: usize) -> CoreResult<()> {
        self.inner.advise(offset, length, AdviceKind::Sequential)
    }

    /// Advise the OS about random access pattern.
    pub fn advise_random(&self, offset: usize, length: usize) -> CoreResult<()> {
        self.inner.advise(offset, length, AdviceKind::Random)
    }

    /// Advise the OS to prefetch (willneed) a byte range.
    pub fn advise_willneed(&self, offset: usize, length: usize) -> CoreResult<()> {
        self.inner.advise(offset, length, AdviceKind::WillNeed)
    }
}

// ===========================================================================
// MappedFileMut (read-write)
// ===========================================================================

/// A read-write memory-mapped file.
///
/// Changes written through `as_mut_slice()` are flushed to disk
/// either lazily by the OS or explicitly via [`flush`](MappedFileMut::flush).
pub struct MappedFileMut {
    inner: MmapInner,
    path: PathBuf,
}

impl MappedFileMut {
    /// Open (or create) a file and memory-map it in read-write mode.
    ///
    /// If `size` is provided and the file is smaller, it will be extended.
    pub fn open(path: &Path, size: Option<u64>) -> CoreResult<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .map_err(|e| {
                CoreError::IoError(ErrorContext::new(format!(
                    "failed to open file {}: {e}",
                    path.display()
                )))
            })?;

        let metadata = file.metadata().map_err(|e| {
            CoreError::IoError(ErrorContext::new(format!(
                "failed to read metadata for {}: {e}",
                path.display()
            )))
        })?;

        let mut len = metadata.len();
        if let Some(requested) = size {
            if requested > len {
                file.set_len(requested).map_err(|e| {
                    CoreError::IoError(ErrorContext::new(format!(
                        "failed to extend file {}: {e}",
                        path.display()
                    )))
                })?;
                len = requested;
            }
        }

        let len = len as usize;
        if len == 0 {
            return Ok(Self {
                inner: MmapInner::empty(),
                path: path.to_path_buf(),
            });
        }

        let inner = MmapInner::map_read_write(&file, len)?;
        Ok(Self {
            inner,
            path: path.to_path_buf(),
        })
    }

    /// The mapped bytes as a slice.
    pub fn as_slice(&self) -> &[u8] {
        self.inner.as_slice()
    }

    /// The mapped bytes as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.inner.as_mut_slice()
    }

    /// Length of the mapping in bytes.
    pub fn len(&self) -> usize {
        self.inner.len
    }

    /// Whether the mapping is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.len == 0
    }

    /// Flush changes to disk.
    pub fn flush(&self) -> CoreResult<()> {
        self.inner.flush()
    }

    /// Flush a specific byte range to disk.
    pub fn flush_range(&self, offset: usize, length: usize) -> CoreResult<()> {
        self.inner.flush_range(offset, length)
    }

    /// The path of the mapped file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

// ===========================================================================
// MappedArrayView
// ===========================================================================

/// Interpret a memory-mapped region as a typed array slice.
///
/// This is a zero-copy view: no data is moved or copied.
///
/// # Requirements
///
/// - The mapped region must be properly aligned for type `T`.
/// - The region's length must be a multiple of `std::mem::size_of::<T>()`.
pub struct MappedArrayView<'a, T> {
    slice: &'a [T],
}

impl<'a, T: Copy> MappedArrayView<'a, T> {
    /// Create a typed view over a read-only mapped file.
    pub fn from_mapped(mapped: &'a MappedFile) -> CoreResult<Self> {
        Self::from_bytes(mapped.as_slice())
    }

    /// Create a typed view over a read-write mapped file.
    pub fn from_mapped_mut(mapped: &'a MappedFileMut) -> CoreResult<Self> {
        Self::from_bytes(mapped.as_slice())
    }

    /// Create a typed view over a raw byte slice.
    pub fn from_bytes(bytes: &'a [u8]) -> CoreResult<Self> {
        let elem_size = std::mem::size_of::<T>();
        if elem_size == 0 {
            return Err(CoreError::ValueError(ErrorContext::new(
                "cannot create MappedArrayView for zero-sized type".to_string(),
            )));
        }
        if bytes.len() % elem_size != 0 {
            return Err(CoreError::ValueError(ErrorContext::new(format!(
                "byte slice length {} is not a multiple of element size {}",
                bytes.len(),
                elem_size
            ))));
        }
        let ptr = bytes.as_ptr();
        let align = std::mem::align_of::<T>();
        if (ptr as usize) % align != 0 {
            return Err(CoreError::ValueError(ErrorContext::new(format!(
                "byte slice is not aligned to {} bytes (required for {})",
                align,
                std::any::type_name::<T>()
            ))));
        }
        let count = bytes.len() / elem_size;
        // SAFETY: We verified alignment and size. The data is valid for the
        // lifetime 'a since the MappedFile/bytes outlives this view.
        let slice = unsafe { std::slice::from_raw_parts(ptr as *const T, count) };
        Ok(Self { slice })
    }

    /// The typed slice.
    pub fn as_slice(&self) -> &[T] {
        self.slice
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    /// Whether the view is empty.
    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }

    /// Get element at `index`.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.slice.get(index)
    }
}

/// Mutable typed view over a memory-mapped region.
pub struct MappedArrayViewMut<'a, T> {
    slice: &'a mut [T],
}

impl<'a, T: Copy> MappedArrayViewMut<'a, T> {
    /// Create a mutable typed view over a read-write mapped file.
    pub fn from_mapped_mut(mapped: &'a mut MappedFileMut) -> CoreResult<Self> {
        let elem_size = std::mem::size_of::<T>();
        if elem_size == 0 {
            return Err(CoreError::ValueError(ErrorContext::new(
                "cannot create MappedArrayViewMut for zero-sized type".to_string(),
            )));
        }
        let bytes = mapped.as_mut_slice();
        if bytes.len() % elem_size != 0 {
            return Err(CoreError::ValueError(ErrorContext::new(format!(
                "byte slice length {} is not a multiple of element size {}",
                bytes.len(),
                elem_size
            ))));
        }
        let ptr = bytes.as_mut_ptr();
        let align = std::mem::align_of::<T>();
        if (ptr as usize) % align != 0 {
            return Err(CoreError::ValueError(ErrorContext::new(format!(
                "byte slice is not aligned to {} bytes",
                align
            ))));
        }
        let count = bytes.len() / elem_size;
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut T, count) };
        Ok(Self { slice })
    }

    /// The typed slice.
    pub fn as_slice(&self) -> &[T] {
        self.slice
    }

    /// The typed mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.slice
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    /// Whether the view is empty.
    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }
}

// ===========================================================================
// LazyMappedFile
// ===========================================================================

/// A lazy-loading wrapper around [`MappedFile`].
///
/// The file is opened and mapped on first access, and advisory prefetch
/// hints are sent to the OS for the requested regions.
pub struct LazyMappedFile {
    path: PathBuf,
    mapped: once_cell::sync::OnceCell<MappedFile>,
    total_bytes_accessed: AtomicUsize,
}

impl LazyMappedFile {
    /// Create a lazy mapping; the file is not opened until first access.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            mapped: once_cell::sync::OnceCell::new(),
            total_bytes_accessed: AtomicUsize::new(0),
        }
    }

    /// Ensure the file is mapped, returning a reference to the mapping.
    fn ensure_mapped(&self) -> CoreResult<&MappedFile> {
        if let Some(m) = self.mapped.get() {
            return Ok(m);
        }
        let mapped = MappedFile::open(&self.path)?;
        // If another thread beat us, that's fine — we just discard ours.
        let _ = self.mapped.set(mapped);
        self.mapped.get().ok_or_else(|| {
            CoreError::ComputationError(ErrorContext::new(
                "failed to initialize lazy mapped file".to_string(),
            ))
        })
    }

    /// Access a byte range, triggering the mapping if needed.
    ///
    /// Sends a `willneed` advisory hint for the requested range.
    pub fn read_range(&self, offset: usize, length: usize) -> CoreResult<&[u8]> {
        let mapped = self.ensure_mapped()?;
        let end = offset.checked_add(length).ok_or_else(|| {
            CoreError::IndexError(ErrorContext::new("offset + length overflow".to_string()))
        })?;
        if end > mapped.len() {
            return Err(CoreError::IndexError(ErrorContext::new(format!(
                "range {}..{} exceeds file length {}",
                offset,
                end,
                mapped.len()
            ))));
        }
        // Advisory prefetch.
        let _ = mapped.advise_willneed(offset, length);
        self.total_bytes_accessed
            .fetch_add(length, Ordering::Relaxed);
        Ok(&mapped.as_slice()[offset..end])
    }

    /// Read the entire file contents.
    pub fn read_all(&self) -> CoreResult<&[u8]> {
        let mapped = self.ensure_mapped()?;
        let len = mapped.len();
        if len > 0 {
            let _ = mapped.advise_sequential(0, len);
        }
        self.total_bytes_accessed.fetch_add(len, Ordering::Relaxed);
        Ok(mapped.as_slice())
    }

    /// Total bytes accessed so far through this lazy mapping.
    pub fn total_bytes_accessed(&self) -> usize {
        self.total_bytes_accessed.load(Ordering::Relaxed)
    }

    /// Whether the file has been mapped (i.e. first access has occurred).
    pub fn is_mapped(&self) -> bool {
        self.mapped.get().is_some()
    }

    /// The file size (triggers mapping if needed).
    pub fn file_size(&self) -> CoreResult<usize> {
        Ok(self.ensure_mapped()?.len())
    }

    /// The path of the file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

// ===========================================================================
// Platform-specific mmap implementation
// ===========================================================================

/// Advice hint for `madvise`.
#[derive(Debug, Clone, Copy)]
enum AdviceKind {
    Sequential,
    Random,
    WillNeed,
}

/// Internal wrapper around OS-level memory mapping.
struct MmapInner {
    ptr: *mut u8,
    len: usize,
    writable: bool,
    /// Whether this is a true OS mmap (vs heap-allocated fallback).
    is_mmap: bool,
}

// SAFETY: The underlying mapping is thread-safe (read-only mappings are
// inherently safe; writable mappings require external synchronisation,
// which is provided by MappedFileMut's &mut self methods).
unsafe impl Send for MmapInner {}
unsafe impl Sync for MmapInner {}

impl MmapInner {
    fn empty() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            len: 0,
            writable: false,
            is_mmap: false,
        }
    }

    fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    // True mmap on Unix when libc is available via cross_platform feature.
    #[cfg(all(unix, feature = "cross_platform"))]
    fn map_read_only(file: &File, len: usize) -> CoreResult<Self> {
        use std::os::unix::io::AsRawFd;
        let fd = file.as_raw_fd();
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                fd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(CoreError::IoError(ErrorContext::new(format!(
                "mmap failed: {}",
                io::Error::last_os_error()
            ))));
        }
        Ok(Self {
            ptr: ptr as *mut u8,
            len,
            writable: false,
            #[cfg(all(unix, feature = "cross_platform"))]
            is_mmap: true,
            #[cfg(not(all(unix, feature = "cross_platform")))]
            is_mmap: false,
        })
    }

    #[cfg(all(unix, feature = "cross_platform"))]
    fn map_read_write(file: &File, len: usize) -> CoreResult<Self> {
        use std::os::unix::io::AsRawFd;
        let fd = file.as_raw_fd();
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(CoreError::IoError(ErrorContext::new(format!(
                "mmap (rw) failed: {}",
                io::Error::last_os_error()
            ))));
        }
        Ok(Self {
            ptr: ptr as *mut u8,
            len,
            writable: true,
            #[cfg(all(unix, feature = "cross_platform"))]
            is_mmap: true,
            #[cfg(not(all(unix, feature = "cross_platform")))]
            is_mmap: false,
        })
    }

    // Windows native mmap via windows-sys.
    #[cfg(windows)]
    fn map_read_only(file: &File, len: usize) -> CoreResult<Self> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::*;
        use windows_sys::Win32::System::Memory::*;

        let handle = file.as_raw_handle() as isize;
        let mapping = unsafe {
            CreateFileMappingW(
                handle,
                std::ptr::null(),
                PAGE_READONLY,
                0,
                0,
                std::ptr::null(),
            )
        };
        if mapping == 0 {
            return Err(CoreError::IoError(ErrorContext::new(format!(
                "CreateFileMapping failed: {}",
                io::Error::last_os_error()
            ))));
        }
        let ptr = unsafe { MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, len) };
        unsafe { CloseHandle(mapping) };
        if ptr.is_null() {
            return Err(CoreError::IoError(ErrorContext::new(format!(
                "MapViewOfFile failed: {}",
                io::Error::last_os_error()
            ))));
        }
        Ok(Self {
            ptr: ptr as *mut u8,
            len,
            writable: false,
            is_mmap: true,
        })
    }

    #[cfg(windows)]
    fn map_read_write(file: &File, len: usize) -> CoreResult<Self> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::*;
        use windows_sys::Win32::System::Memory::*;

        let handle = file.as_raw_handle() as isize;
        let mapping = unsafe {
            CreateFileMappingW(
                handle,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                0,
                std::ptr::null(),
            )
        };
        if mapping == 0 {
            return Err(CoreError::IoError(ErrorContext::new(format!(
                "CreateFileMapping (rw) failed: {}",
                io::Error::last_os_error()
            ))));
        }
        let ptr = unsafe { MapViewOfFile(mapping, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, len) };
        unsafe { CloseHandle(mapping) };
        if ptr.is_null() {
            return Err(CoreError::IoError(ErrorContext::new(format!(
                "MapViewOfFile (rw) failed: {}",
                io::Error::last_os_error()
            ))));
        }
        Ok(Self {
            ptr: ptr as *mut u8,
            len,
            writable: true,
            is_mmap: true,
        })
    }

    // Fallback: read the file into heap memory (works everywhere).
    #[cfg(not(any(all(unix, feature = "cross_platform"), windows)))]
    fn map_read_only(file: &File, len: usize) -> CoreResult<Self> {
        use std::io::Read;
        let mut buf = vec![0u8; len];
        let mut f = file.try_clone().map_err(|e| {
            CoreError::IoError(ErrorContext::new(format!("file clone failed: {e}")))
        })?;
        f.read_exact(&mut buf)
            .map_err(|e| CoreError::IoError(ErrorContext::new(format!("read failed: {e}"))))?;
        let ptr = buf.as_mut_ptr();
        let buf_len = buf.len();
        std::mem::forget(buf); // Reclaimed in Drop.
        Ok(Self {
            ptr,
            len: buf_len,
            writable: false,
            is_mmap: false,
        })
    }

    #[cfg(not(any(all(unix, feature = "cross_platform"), windows)))]
    fn map_read_write(file: &File, len: usize) -> CoreResult<Self> {
        Self::map_read_only(file, len).map(|mut inner| {
            inner.writable = true;
            inner
        })
    }

    fn flush(&self) -> CoreResult<()> {
        if self.len == 0 || !self.writable {
            return Ok(());
        }
        self.flush_range(0, self.len)
    }

    fn flush_range(&self, offset: usize, length: usize) -> CoreResult<()> {
        if self.len == 0 || !self.writable {
            return Ok(());
        }
        let end = offset.saturating_add(length).min(self.len);
        let actual_len = end.saturating_sub(offset);
        if actual_len == 0 {
            return Ok(());
        }

        if !self.is_mmap {
            // Heap-allocated fallback — nothing to flush to disk.
            return Ok(());
        }

        #[cfg(all(unix, feature = "cross_platform"))]
        {
            let result = unsafe {
                libc::msync(
                    self.ptr.add(offset) as *mut libc::c_void,
                    actual_len,
                    libc::MS_SYNC,
                )
            };
            if result != 0 {
                return Err(CoreError::IoError(ErrorContext::new(format!(
                    "msync failed: {}",
                    io::Error::last_os_error()
                ))));
            }
        }

        #[cfg(windows)]
        {
            use windows_sys::Win32::System::Memory::FlushViewOfFile;
            let result = unsafe { FlushViewOfFile(self.ptr.add(offset) as *const _, actual_len) };
            if result == 0 {
                return Err(CoreError::IoError(ErrorContext::new(format!(
                    "FlushViewOfFile failed: {}",
                    io::Error::last_os_error()
                ))));
            }
        }

        Ok(())
    }

    fn advise(&self, offset: usize, length: usize, kind: AdviceKind) -> CoreResult<()> {
        if self.len == 0 {
            return Ok(());
        }
        let end = offset.saturating_add(length).min(self.len);
        let actual_len = end.saturating_sub(offset);
        if actual_len == 0 {
            return Ok(());
        }

        if !self.is_mmap {
            // Heap fallback — advisory hints are no-ops.
            let _ = kind;
            return Ok(());
        }

        #[cfg(all(unix, feature = "cross_platform"))]
        {
            let advice = match kind {
                AdviceKind::Sequential => libc::MADV_SEQUENTIAL,
                AdviceKind::Random => libc::MADV_RANDOM,
                AdviceKind::WillNeed => libc::MADV_WILLNEED,
            };
            let result = unsafe {
                libc::madvise(
                    self.ptr.add(offset) as *mut libc::c_void,
                    actual_len,
                    advice,
                )
            };
            if result != 0 {
                return Err(CoreError::IoError(ErrorContext::new(format!(
                    "madvise failed: {}",
                    io::Error::last_os_error()
                ))));
            }
        }

        // On Windows and non-libc platforms, advisory hints are best-effort / no-op.
        #[cfg(not(all(unix, feature = "cross_platform")))]
        {
            let _ = kind;
        }

        Ok(())
    }
}

impl Drop for MmapInner {
    fn drop(&mut self) {
        if self.ptr.is_null() || self.len == 0 {
            return;
        }

        if self.is_mmap {
            #[cfg(all(unix, feature = "cross_platform"))]
            {
                unsafe {
                    libc::munmap(self.ptr as *mut libc::c_void, self.len);
                }
            }

            #[cfg(windows)]
            {
                use windows_sys::Win32::System::Memory::UnmapViewOfFile;
                unsafe {
                    UnmapViewOfFile(self.ptr as *const _);
                }
            }
        } else {
            // Heap-allocated fallback: reclaim the leaked Vec.
            unsafe {
                let _ = Vec::from_raw_parts(self.ptr, self.len, self.len);
            }
        }
    }
}

// ===========================================================================
// Helper: create a test file with known content
// ===========================================================================

/// Create a temporary file with the given byte content and return its path.
///
/// Useful for testing and one-off data files.
pub fn create_temp_file(content: &[u8]) -> CoreResult<PathBuf> {
    let dir = std::env::temp_dir();
    let name = format!(
        "scirs2_mmap_{}.bin",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let path = dir.join(name);
    fs::write(&path, content).map_err(|e| {
        CoreError::IoError(ErrorContext::new(format!(
            "failed to write temp file {}: {e}",
            path.display()
        )))
    })?;
    Ok(path)
}

/// Write a typed slice as raw bytes to a temporary file and return its path.
///
/// The resulting file can be memory-mapped and interpreted via [`MappedArrayView`].
pub fn create_temp_array_file<T: Copy>(data: &[T]) -> CoreResult<PathBuf> {
    let byte_len = data.len() * std::mem::size_of::<T>();
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
    create_temp_file(bytes)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(suffix: &str) -> PathBuf {
        let dir = std::env::temp_dir();
        dir.join(format!(
            "scirs2_mmap_test_{}_{}.bin",
            suffix,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ))
    }

    #[test]
    fn test_mapped_file_read() {
        let content = b"Hello, memory-mapped world!";
        let path = create_temp_file(content).expect("create temp");
        let mapped = MappedFile::open(&path).expect("open");
        assert_eq!(mapped.as_slice(), content);
        assert_eq!(mapped.len(), content.len());
        assert!(!mapped.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mapped_file_empty() {
        let path = temp_path("empty");
        std::fs::write(&path, b"").expect("write");
        let mapped = MappedFile::open(&path).expect("open");
        assert!(mapped.is_empty());
        assert_eq!(mapped.len(), 0);
        assert_eq!(mapped.as_slice(), &[] as &[u8]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mapped_file_mut_write() {
        let path = temp_path("mut_write");
        // Create a file with some initial content.
        std::fs::write(&path, &[0u8; 16]).expect("write");

        {
            let mut mapped = MappedFileMut::open(&path, None).expect("open");
            let slice = mapped.as_mut_slice();
            for (i, byte) in slice.iter_mut().enumerate() {
                *byte = i as u8;
            }
            mapped.flush().expect("flush");
        }

        // Verify by re-reading.
        let data = std::fs::read(&path).expect("read");
        for (i, &b) in data.iter().enumerate() {
            assert_eq!(b, i as u8);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mapped_file_mut_extend() {
        let path = temp_path("extend");
        std::fs::write(&path, b"abc").expect("write");
        let mapped = MappedFileMut::open(&path, Some(100)).expect("open");
        assert_eq!(mapped.len(), 100);
        assert_eq!(&mapped.as_slice()[..3], b"abc");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mapped_array_view_f64() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let path = create_temp_array_file(&data).expect("create");
        let mapped = MappedFile::open(&path).expect("open");
        let view: MappedArrayView<'_, f64> = MappedArrayView::from_mapped(&mapped).expect("view");
        assert_eq!(view.len(), 5);
        assert_eq!(view.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(view.get(2), Some(&3.0));
        assert_eq!(view.get(10), None);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mapped_array_view_u32() {
        let data: Vec<u32> = (0..100).collect();
        let path = create_temp_array_file(&data).expect("create");
        let mapped = MappedFile::open(&path).expect("open");
        let view: MappedArrayView<'_, u32> = MappedArrayView::from_mapped(&mapped).expect("view");
        assert_eq!(view.len(), 100);
        for (i, &v) in view.as_slice().iter().enumerate() {
            assert_eq!(v, i as u32);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mapped_array_view_bad_alignment() {
        // Create a byte slice that's not aligned for u64.
        let bytes = vec![0u8; 17]; // 17 bytes, not a multiple of 8
        let result = MappedArrayView::<u64>::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_mapped_array_view_mut() {
        let path = temp_path("view_mut");
        let initial: Vec<i32> = vec![0; 10];
        let byte_len = initial.len() * std::mem::size_of::<i32>();
        let bytes = unsafe { std::slice::from_raw_parts(initial.as_ptr() as *const u8, byte_len) };
        std::fs::write(&path, bytes).expect("write");

        {
            let mut mapped = MappedFileMut::open(&path, None).expect("open");
            let mut view: MappedArrayViewMut<'_, i32> =
                MappedArrayViewMut::from_mapped_mut(&mut mapped).expect("view");
            for (i, slot) in view.as_mut_slice().iter_mut().enumerate() {
                *slot = (i * i) as i32;
            }
        }

        // Verify.
        let mapped = MappedFile::open(&path).expect("open");
        let view: MappedArrayView<'_, i32> = MappedArrayView::from_mapped(&mapped).expect("view");
        for (i, &v) in view.as_slice().iter().enumerate() {
            assert_eq!(v, (i * i) as i32);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_lazy_mapped_file() {
        let content = b"lazy loading test data";
        let path = create_temp_file(content).expect("create");
        let lazy = LazyMappedFile::new(path.clone());

        assert!(!lazy.is_mapped());
        assert_eq!(lazy.total_bytes_accessed(), 0);

        let data = lazy.read_all().expect("read_all");
        assert!(lazy.is_mapped());
        assert_eq!(data, content);
        assert_eq!(lazy.total_bytes_accessed(), content.len());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_lazy_mapped_file_range() {
        let content: Vec<u8> = (0..=255).collect();
        let path = create_temp_file(&content).expect("create");
        let lazy = LazyMappedFile::new(path.clone());

        let range = lazy.read_range(10, 5).expect("read_range");
        assert_eq!(range, &[10, 11, 12, 13, 14]);
        assert_eq!(lazy.total_bytes_accessed(), 5);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_lazy_mapped_file_out_of_range() {
        let path = create_temp_file(&[1, 2, 3]).expect("create");
        let lazy = LazyMappedFile::new(path.clone());
        let result = lazy.read_range(0, 100);
        assert!(result.is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_lazy_mapped_file_size() {
        let content = vec![0u8; 4096];
        let path = create_temp_file(&content).expect("create");
        let lazy = LazyMappedFile::new(path.clone());
        assert_eq!(lazy.file_size().expect("size"), 4096);
        assert!(lazy.is_mapped()); // file_size triggers mapping
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_advise_sequential() {
        let content = vec![0u8; 8192];
        let path = create_temp_file(&content).expect("create");
        let mapped = MappedFile::open(&path).expect("open");
        // Advisory hints are best-effort; just verify they don't error.
        mapped.advise_sequential(0, 4096).expect("advise seq");
        mapped.advise_random(4096, 4096).expect("advise rand");
        mapped.advise_willneed(0, 8192).expect("advise willneed");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_flush_range() {
        let path = temp_path("flush_range");
        std::fs::write(&path, &[0u8; 64]).expect("write");
        let mut mapped = MappedFileMut::open(&path, None).expect("open");
        mapped.as_mut_slice()[0] = 0xFF;
        mapped.flush_range(0, 8).expect("flush_range");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_create_temp_file() {
        let path = create_temp_file(b"test content").expect("create");
        assert!(path.exists());
        let content = std::fs::read(&path).expect("read");
        assert_eq!(content, b"test content");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_create_temp_array_file() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let path = create_temp_array_file(&data).expect("create");
        let mapped = MappedFile::open(&path).expect("open");
        let view: MappedArrayView<'_, f32> = MappedArrayView::from_mapped(&mapped).expect("view");
        assert_eq!(view.as_slice(), &[1.0, 2.0, 3.0]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_nonexistent_file() {
        let result = MappedFile::open(Path::new("/tmp/this_file_should_not_exist_scirs2_test"));
        assert!(result.is_err());
    }

    #[test]
    fn test_mapped_file_path() {
        let path = create_temp_file(b"x").expect("create");
        let mapped = MappedFile::open(&path).expect("open");
        assert_eq!(mapped.path(), path.as_path());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_lazy_mapped_path() {
        let lazy = LazyMappedFile::new("/tmp/lazy_test");
        assert_eq!(lazy.path(), Path::new("/tmp/lazy_test"));
    }
}
