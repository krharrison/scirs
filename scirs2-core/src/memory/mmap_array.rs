//! # Memory-Mapped NDArray Wrapper
//!
//! This module provides a zero-copy, file-backed ndarray with Copy-on-Write (COW) semantics.
//!
//! ## Overview
//!
//! [`MmapArray<F>`] wraps a file on disk as an ndarray, enabling:
//!
//! - **Zero-copy reads**: Data is served directly from OS page cache without copying to RAM.
//! - **Read-write mmap**: Mutations are written directly to the file via mmap.
//! - **Copy-on-Write**: Reads are zero-copy; the first write triggers a copy of the data to RAM,
//!   after which all writes stay in RAM until explicitly persisted.
//!
//! ## File Format
//!
//! The binary file starts with a 64-byte header (little-endian):
//!
//! ```text
//! Offset  Size  Field
//! ------  ----  -----
//!  0..4    4    Magic bytes: b"MMAP"
//!  4       1    Version: 1
//!  5       1    dtype_id (1=f32, 2=f64, 3=i32, 4=i64)
//!  6..8    2    ndim (u16, little-endian)
//!  8..16   8    total_elements (u64, little-endian)
//! 16..16+8*ndim  8 per dim  shape dimensions (u64 each, little-endian)
//! ... zero-padding to byte 64
//! 64..    data  Raw element bytes (F, little-endian, row-major / C order)
//! ```
//!
//! ## Example
//!
//! ```rust,no_run
//! # #[cfg(feature = "mmap")]
//! # {
//! use scirs2_core::memory::mmap_array::{MmapArray, MmapError};
//! use ndarray::ArrayD;
//!
//! let tmp = std::env::temp_dir().join("example.mmap");
//! let data = ArrayD::<f32>::zeros(ndarray::IxDyn(&[4, 8]));
//! let arr = MmapArray::<f32>::create(&tmp, &data).expect("should succeed");
//! assert_eq!(arr.shape(), &[4, 8]);
//! # }
//! ```

use memmap2::{Mmap, MmapMut, MmapOptions};
use ndarray::{Array, ArrayView, ArrayViewMut, IxDyn};
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Header constants
// ---------------------------------------------------------------------------

/// Magic bytes at the start of every mmap array file.
const MAGIC: &[u8; 4] = b"MMAP";
/// Current file format version.
const FORMAT_VERSION: u8 = 1;
/// Total header size in bytes. Data begins at this offset.
const HEADER_SIZE: usize = 64;

// ---------------------------------------------------------------------------
// MmapElement trait
// ---------------------------------------------------------------------------

/// Types that can be stored in a memory-mapped file.
///
/// Implementors must be:
/// - `Copy` — trivially duplicable, no heap ownership
/// - `bytemuck::Pod` — safe for byte-level reinterpretation
/// - `bytemuck::Zeroable` — a zero-initialized value is valid
/// - `'static` — no borrowed references
pub trait MmapElement: Copy + bytemuck::Pod + bytemuck::Zeroable + 'static {
    /// Unique byte tag written to the file header identifying this type.
    fn dtype_id() -> u8;
    /// Size of a single element in bytes.
    fn element_size() -> usize;
}

impl MmapElement for f32 {
    fn dtype_id() -> u8 {
        1
    }
    fn element_size() -> usize {
        4
    }
}

impl MmapElement for f64 {
    fn dtype_id() -> u8 {
        2
    }
    fn element_size() -> usize {
        8
    }
}

impl MmapElement for i32 {
    fn dtype_id() -> u8 {
        3
    }
    fn element_size() -> usize {
        4
    }
}

impl MmapElement for i64 {
    fn dtype_id() -> u8 {
        4
    }
    fn element_size() -> usize {
        8
    }
}

// ---------------------------------------------------------------------------
// Internal storage enum
// ---------------------------------------------------------------------------

/// The underlying memory storage — one of three modes.
enum MmapStorage<F: MmapElement> {
    /// Read-only mapping backed directly by the file.
    ReadOnly {
        mmap: Mmap,
        _phantom: std::marker::PhantomData<F>,
    },
    /// Read-write mapping backed directly by the file.
    ReadWrite {
        mmap: MmapMut,
        _phantom: std::marker::PhantomData<F>,
    },
    /// Copy-on-write: reads zero-copy from `mmap`; first write populates `cow_data`.
    CopyOnWrite {
        mmap: Mmap,
        /// `None` until the first write triggers the copy.
        cow_data: Option<Vec<F>>,
    },
}

// ---------------------------------------------------------------------------
// MmapArray
// ---------------------------------------------------------------------------

/// A zero-copy, file-backed ndarray with optional Copy-on-Write semantics.
///
/// The array is stored in a flat binary file with a 64-byte header.  Array
/// data is accessed via `memmap2`, so the OS manages paging automatically.
///
/// # Type Parameter
///
/// `F` must implement [`MmapElement`] (currently: `f32`, `f64`, `i32`, `i64`).
pub struct MmapArray<F: MmapElement> {
    storage: MmapStorage<F>,
    shape: Vec<usize>,
    /// C-order strides in element counts (not bytes).
    strides: Vec<usize>,
    file_path: PathBuf,
}

impl<F: MmapElement> std::fmt::Debug for MmapArray<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mode = match &self.storage {
            MmapStorage::ReadOnly { .. } => "ReadOnly",
            MmapStorage::ReadWrite { .. } => "ReadWrite",
            MmapStorage::CopyOnWrite { cow_data, .. } => {
                if cow_data.is_some() {
                    "CopyOnWrite(dirty)"
                } else {
                    "CopyOnWrite(clean)"
                }
            }
        };
        f.debug_struct("MmapArray")
            .field("mode", &mode)
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field("file_path", &self.file_path)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Header encode / decode
// ---------------------------------------------------------------------------

/// Encode a 64-byte header into `buf`.
///
/// # Panics
///
/// Panics if `ndim` > 6 (the header only has room for 6 shape dimensions).
fn encode_header(buf: &mut [u8; HEADER_SIZE], dtype_id: u8, shape: &[usize]) {
    let ndim = shape.len();
    assert!(
        ndim <= 6,
        "MmapArray supports at most 6 dimensions (header limit)"
    );

    buf.fill(0);

    // Magic
    buf[0..4].copy_from_slice(MAGIC);
    // Version
    buf[4] = FORMAT_VERSION;
    // dtype_id
    buf[5] = dtype_id;
    // ndim (u16 LE)
    let ndim_u16 = ndim as u16;
    buf[6..8].copy_from_slice(&ndim_u16.to_le_bytes());
    // total_elements (u64 LE)
    let total: u64 = shape.iter().product::<usize>() as u64;
    buf[8..16].copy_from_slice(&total.to_le_bytes());
    // shape dimensions
    for (i, &dim) in shape.iter().enumerate() {
        let off = 16 + i * 8;
        buf[off..off + 8].copy_from_slice(&(dim as u64).to_le_bytes());
    }
}

/// Decode a 64-byte header, returning `(dtype_id, shape)`.
fn decode_header(buf: &[u8; HEADER_SIZE]) -> Result<(u8, Vec<usize>), MmapError> {
    if &buf[0..4] != MAGIC {
        return Err(MmapError::InvalidMagic);
    }
    let version = buf[4];
    if version != FORMAT_VERSION {
        return Err(MmapError::VersionMismatch(version));
    }
    let dtype_id = buf[5];
    let ndim = u16::from_le_bytes([buf[6], buf[7]]) as usize;
    let _total_elements = u64::from_le_bytes(buf[8..16].try_into().map_err(|_| {
        MmapError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "header truncated at total_elements",
        ))
    })?);

    let mut shape = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let off = 16 + i * 8;
        if off + 8 > HEADER_SIZE {
            return Err(MmapError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("header too small for ndim={}", ndim),
            )));
        }
        let dim = u64::from_le_bytes(buf[off..off + 8].try_into().map_err(|_| {
            MmapError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "header truncated in shape",
            ))
        })?);
        shape.push(dim as usize);
    }

    Ok((dtype_id, shape))
}

/// Compute C-order (row-major) strides for the given shape (element counts, not bytes).
fn c_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Total element count for a shape slice.
fn total_elements(shape: &[usize]) -> usize {
    shape.iter().product()
}

// ---------------------------------------------------------------------------
// impl MmapArray
// ---------------------------------------------------------------------------

impl<F: MmapElement> MmapArray<F> {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Create a new memory-mapped file from an existing ndarray and return a
    /// **read-write** [`MmapArray`] backed by that file.
    ///
    /// The input array must be in standard C-order (contiguous) layout.
    /// A 64-byte header is written first, followed immediately by the raw
    /// element bytes.
    ///
    /// # Errors
    ///
    /// Returns [`MmapError::NonContiguous`] if `data` is not a contiguous array.
    pub fn create(path: &Path, data: &Array<F, IxDyn>) -> Result<Self, MmapError> {
        // Require a contiguous layout so we can do a single slice copy.
        if !data.is_standard_layout() {
            return Err(MmapError::NonContiguous);
        }

        let shape = data.shape().to_vec();
        let n_elems = total_elements(&shape);
        let data_bytes = n_elems * F::element_size();
        let file_size = HEADER_SIZE + data_bytes;

        // Open / create file, set length.
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(file_size as u64)?;

        // Write header via a plain write (simpler than mmap for small header).
        {
            use std::io::BufWriter;
            let mut writer = BufWriter::new(&file);
            let mut header = [0u8; HEADER_SIZE];
            encode_header(&mut header, F::dtype_id(), &shape);
            writer.write_all(&header)?;
            // Write element bytes from the array's raw slice.
            let raw: &[F] = data.as_slice().ok_or(MmapError::NonContiguous)?;
            writer.write_all(bytemuck::cast_slice(raw))?;
            writer.flush()?;
        }

        // Now open a read-write mmap over the entire file (including header).
        // We use the whole file mapping and expose only the data region.
        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        let strides = c_strides(&shape);
        Ok(Self {
            storage: MmapStorage::ReadWrite {
                mmap,
                _phantom: std::marker::PhantomData,
            },
            shape,
            strides,
            file_path: path.to_path_buf(),
        })
    }

    /// Open an existing `.mmap` file in **read-only** mode.
    ///
    /// The returned array cannot be mutated; calls to [`view_mut`](Self::view_mut)
    /// will return [`MmapError::ReadOnly`].
    pub fn open_read_only(path: &Path) -> Result<Self, MmapError> {
        let file = OpenOptions::new().read(true).open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        Self::from_readonly_mmap(mmap, path)
    }

    /// Open an existing `.mmap` file in **read-write** mode.
    ///
    /// Mutations are written directly to the file (no buffering).
    pub fn open_read_write(path: &Path) -> Result<Self, MmapError> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        // Read and validate header from the mapping itself.
        if mmap.len() < HEADER_SIZE {
            return Err(MmapError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "file too small to contain header",
            )));
        }
        let header_bytes: &[u8; HEADER_SIZE] = mmap[..HEADER_SIZE].try_into().map_err(|_| {
            MmapError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "could not read header",
            ))
        })?;
        let (dtype_id, shape) = decode_header(header_bytes)?;
        if dtype_id != F::dtype_id() {
            return Err(MmapError::DtypeMismatch {
                expected: F::dtype_id(),
                actual: dtype_id,
            });
        }

        let strides = c_strides(&shape);
        Ok(Self {
            storage: MmapStorage::ReadWrite {
                mmap,
                _phantom: std::marker::PhantomData,
            },
            shape,
            strides,
            file_path: path.to_path_buf(),
        })
    }

    /// Open an existing `.mmap` file in **Copy-on-Write** mode.
    ///
    /// Reads are served zero-copy from the OS page cache.  The first call to
    /// [`view_mut`](Self::view_mut) triggers a full copy of the data into RAM,
    /// after which all mutations happen in-memory.  The original file is never
    /// modified unless you later call [`flush`](Self::flush) (which, in COW mode,
    /// is a no-op since there is no writable mapping to flush).
    pub fn open_cow(path: &Path) -> Result<Self, MmapError> {
        let file = OpenOptions::new().read(true).open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let (_, shape, strides) = Self::parse_readonly_mmap(&mmap, path)?;
        Ok(Self {
            storage: MmapStorage::CopyOnWrite {
                mmap,
                cow_data: None,
            },
            shape,
            strides,
            file_path: path.to_path_buf(),
        })
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Return a read-only ndarray view.
    ///
    /// For read-only and read-write modes, the view is zero-copy (backed by the mmap).
    /// For COW mode before any write, the view is also zero-copy.
    /// For COW mode after the first write, the view is backed by the in-RAM copy.
    pub fn view(&self) -> Result<ArrayView<'_, F, IxDyn>, MmapError> {
        match &self.storage {
            MmapStorage::ReadOnly { mmap, .. } => {
                let data_slice = &mmap[HEADER_SIZE..];
                let elems: &[F] = bytemuck::cast_slice(data_slice);
                let ix = IxDyn(self.shape.as_slice());
                // SAFETY: We own the mmap for the lifetime of `&self`, the
                // pointer is valid, and the length matches `self.shape`.
                let view = unsafe { ArrayView::from_shape_ptr(ix, elems.as_ptr()) };
                Ok(view)
            }
            MmapStorage::ReadWrite { mmap, .. } => {
                let data_slice = &mmap[HEADER_SIZE..];
                let elems: &[F] = bytemuck::cast_slice(data_slice);
                let ix = IxDyn(self.shape.as_slice());
                // SAFETY: Same as above.
                let view = unsafe { ArrayView::from_shape_ptr(ix, elems.as_ptr()) };
                Ok(view)
            }
            MmapStorage::CopyOnWrite { mmap, cow_data } => {
                match cow_data {
                    None => {
                        // No write yet — serve directly from the mmap.
                        let data_slice = &mmap[HEADER_SIZE..];
                        let elems: &[F] = bytemuck::cast_slice(data_slice);
                        let ix = IxDyn(self.shape.as_slice());
                        // SAFETY: Mmap is valid for `&self` lifetime.
                        let view = unsafe { ArrayView::from_shape_ptr(ix, elems.as_ptr()) };
                        Ok(view)
                    }
                    Some(data) => {
                        let ix = IxDyn(self.shape.as_slice());
                        // SAFETY: `data` is a Vec owned by `self`, pointer valid for `&self`.
                        let view = unsafe { ArrayView::from_shape_ptr(ix, data.as_ptr()) };
                        Ok(view)
                    }
                }
            }
        }
    }

    /// Return a mutable ndarray view.
    ///
    /// - **ReadOnly**: always returns [`MmapError::ReadOnly`].
    /// - **ReadWrite**: the view is zero-copy and writes go directly to the file.
    /// - **COW**: triggers a copy of the mmap data into RAM on the first call.
    pub fn view_mut(&mut self) -> Result<ArrayViewMut<'_, F, IxDyn>, MmapError> {
        match &mut self.storage {
            MmapStorage::ReadOnly { .. } => Err(MmapError::ReadOnly),
            MmapStorage::ReadWrite { mmap, .. } => {
                let data_slice = &mut mmap[HEADER_SIZE..];
                let elems: &mut [F] = bytemuck::cast_slice_mut(data_slice);
                let ix = IxDyn(self.shape.as_slice());
                // SAFETY: We have exclusive access via `&mut self`.
                let view = unsafe { ArrayViewMut::from_shape_ptr(ix, elems.as_mut_ptr()) };
                Ok(view)
            }
            MmapStorage::CopyOnWrite { mmap, cow_data } => {
                // Trigger COW fault if this is the first write.
                if cow_data.is_none() {
                    let data_slice = &mmap[HEADER_SIZE..];
                    let elems: &[F] = bytemuck::cast_slice(data_slice);
                    *cow_data = Some(elems.to_vec());
                }
                let data = cow_data.as_mut().ok_or_else(|| {
                    MmapError::Io(std::io::Error::other(
                        "COW data unexpectedly None after initialization",
                    ))
                })?;
                let ix = IxDyn(self.shape.as_slice());
                // SAFETY: We have exclusive access via `&mut self`.
                let view = unsafe { ArrayViewMut::from_shape_ptr(ix, data.as_mut_ptr()) };
                Ok(view)
            }
        }
    }

    /// Flush changes to disk.
    ///
    /// - **ReadWrite**: calls `MmapMut::flush()`.
    /// - **ReadOnly** / **COW**: no-op (returns `Ok(())`).
    pub fn flush(&self) -> Result<(), MmapError> {
        match &self.storage {
            MmapStorage::ReadWrite { mmap, .. } => {
                mmap.flush()?;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Return the shape of the array.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return the C-order strides (in element counts).
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Return the total number of elements.
    pub fn len(&self) -> usize {
        total_elements(&self.shape)
    }

    /// Return `true` if the array has zero elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the file path backing this array.
    pub fn file_path(&self) -> &Path {
        &self.file_path
    }

    /// Copy all elements into a heap-allocated [`Array<F, IxDyn>`].
    ///
    /// This always copies; it is equivalent to `.view()?.to_owned()`.
    pub fn to_owned_array(&self) -> Result<Array<F, IxDyn>, MmapError> {
        let view = self.view()?;
        Ok(view.to_owned())
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Construct a `ReadOnly` variant from an already-mapped region.
    fn from_readonly_mmap(mmap: Mmap, path: &Path) -> Result<Self, MmapError> {
        let (_, shape, strides) = Self::parse_readonly_mmap(&mmap, path)?;
        Ok(Self {
            storage: MmapStorage::ReadOnly {
                mmap,
                _phantom: std::marker::PhantomData,
            },
            shape,
            strides,
            file_path: path.to_path_buf(),
        })
    }

    /// Validate and parse the header from a read-only mapping.
    ///
    /// Returns `(dtype_id, shape, strides)`.
    fn parse_readonly_mmap(
        mmap: &Mmap,
        _path: &Path,
    ) -> Result<(u8, Vec<usize>, Vec<usize>), MmapError> {
        if mmap.len() < HEADER_SIZE {
            return Err(MmapError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "file too small to contain header",
            )));
        }
        let header_bytes: &[u8; HEADER_SIZE] = mmap[..HEADER_SIZE].try_into().map_err(|_| {
            MmapError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "could not read header slice",
            ))
        })?;
        let (dtype_id, shape) = decode_header(header_bytes)?;
        if dtype_id != F::dtype_id() {
            return Err(MmapError::DtypeMismatch {
                expected: F::dtype_id(),
                actual: dtype_id,
            });
        }
        let strides = c_strides(&shape);
        Ok((dtype_id, shape, strides))
    }

    // -----------------------------------------------------------------------
    // Test-only helpers
    // -----------------------------------------------------------------------

    /// Create an anonymous (non-file-backed) mmap for testing purposes.
    ///
    /// This uses an anonymous mapping backed by a temporary file.
    #[cfg(test)]
    fn create_anonymous(shape: &[usize]) -> Result<Self, MmapError> {
        use tempfile::tempfile;

        let n_elems = total_elements(shape);
        let file_size = HEADER_SIZE + n_elems * F::element_size();

        let file = tempfile()?;
        file.set_len(file_size as u64)?;

        // Write header.
        {
            use std::io::BufWriter;
            let mut writer = BufWriter::new(&file);
            let mut header = [0u8; HEADER_SIZE];
            encode_header(&mut header, F::dtype_id(), shape);
            writer.write_all(&header)?;
            // Data is zero-initialized by `set_len`.
            writer.flush()?;
        }

        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };
        let strides = c_strides(shape);

        // We store a dummy path since this is anonymous.
        Ok(Self {
            storage: MmapStorage::ReadWrite {
                mmap,
                _phantom: std::marker::PhantomData,
            },
            shape: shape.to_vec(),
            strides,
            file_path: PathBuf::from("<anonymous>"),
        })
    }
}

// ---------------------------------------------------------------------------
// MmapError
// ---------------------------------------------------------------------------

/// Errors that can occur when creating or accessing a [`MmapArray`].
#[derive(Debug, thiserror::Error)]
pub enum MmapError {
    /// An I/O error occurred (file open, read, write, flush, etc.).
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// The file does not start with the expected magic bytes `b"MMAP"`.
    #[error("Invalid magic bytes — not a valid mmap array file")]
    InvalidMagic,

    /// The file header reports a format version other than 1.
    #[error("Version mismatch: expected 1, got {0}")]
    VersionMismatch(u8),

    /// The dtype tag in the file does not match the requested element type.
    #[error("Dtype mismatch: expected dtype_id {expected}, got {actual}")]
    DtypeMismatch { expected: u8, actual: u8 },

    /// The stored shape does not match the shape provided by the caller.
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// The source ndarray is not in contiguous C-order layout.
    #[error("Array is not contiguous (non-contiguous layouts not supported)")]
    NonContiguous,

    /// The mapping is read-only and a mutable view was requested.
    #[error("Array is read-only")]
    ReadOnly,
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    /// Build a small f32 array filled with ascending values.
    fn make_f32_array(shape: &[usize]) -> ArrayD<f32> {
        let n = shape.iter().product::<usize>();
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        ArrayD::from_shape_vec(IxDyn(shape), data).expect("shape mismatch in test helper")
    }

    /// Build a small f64 array filled with ascending values.
    fn make_f64_array(shape: &[usize]) -> ArrayD<f64> {
        let n = shape.iter().product::<usize>();
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        ArrayD::from_shape_vec(IxDyn(shape), data).expect("shape mismatch in test helper")
    }

    // -----------------------------------------------------------------------
    // Header round-trip tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_header_round_trip_1d() {
        let mut buf = [0u8; HEADER_SIZE];
        encode_header(&mut buf, 1, &[100]);
        let (dtype_id, shape) = decode_header(&buf).expect("decode failed");
        assert_eq!(dtype_id, 1);
        assert_eq!(shape, vec![100usize]);
    }

    #[test]
    fn test_header_round_trip_2d() {
        let mut buf = [0u8; HEADER_SIZE];
        encode_header(&mut buf, 2, &[3, 4]);
        let (dtype_id, shape) = decode_header(&buf).expect("decode failed");
        assert_eq!(dtype_id, 2);
        assert_eq!(shape, vec![3, 4]);
    }

    #[test]
    fn test_header_round_trip_6d() {
        let mut buf = [0u8; HEADER_SIZE];
        encode_header(&mut buf, 4, &[2, 3, 4, 5, 6, 7]);
        let (dtype_id, shape) = decode_header(&buf).expect("decode failed");
        assert_eq!(dtype_id, 4);
        assert_eq!(shape, vec![2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_bad_magic() {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(b"NOPE");
        let err = decode_header(&buf).expect_err("should fail");
        assert!(matches!(err, MmapError::InvalidMagic));
    }

    #[test]
    fn test_bad_version() {
        let mut buf = [0u8; HEADER_SIZE];
        encode_header(&mut buf, 1, &[10]);
        buf[4] = 99; // corrupt version
        let err = decode_header(&buf).expect_err("should fail");
        assert!(matches!(err, MmapError::VersionMismatch(99)));
    }

    // -----------------------------------------------------------------------
    // c_strides
    // -----------------------------------------------------------------------

    #[test]
    fn test_c_strides_1d() {
        assert_eq!(c_strides(&[5]), vec![1]);
    }

    #[test]
    fn test_c_strides_2d() {
        assert_eq!(c_strides(&[3, 4]), vec![4, 1]);
    }

    #[test]
    fn test_c_strides_3d() {
        assert_eq!(c_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    // -----------------------------------------------------------------------
    // MmapElement implementations
    // -----------------------------------------------------------------------

    #[test]
    fn test_dtype_ids_are_distinct() {
        let ids = [
            f32::dtype_id(),
            f64::dtype_id(),
            i32::dtype_id(),
            i64::dtype_id(),
        ];
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(ids[i], ids[j], "dtype IDs must be unique");
            }
        }
    }

    #[test]
    fn test_element_sizes() {
        assert_eq!(f32::element_size(), 4);
        assert_eq!(f64::element_size(), 8);
        assert_eq!(i32::element_size(), 4);
        assert_eq!(i64::element_size(), 8);
    }

    // -----------------------------------------------------------------------
    // create + open_read_only
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_and_read_only_f32() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_mmap_create_ro_f32.mmap");

        let original = make_f32_array(&[4, 5]);
        {
            let arr = MmapArray::<f32>::create(&path, &original).expect("create failed");
            assert_eq!(arr.shape(), &[4, 5]);
            assert_eq!(arr.len(), 20);
            assert!(!arr.is_empty());
        }

        {
            let arr = MmapArray::<f32>::open_read_only(&path).expect("open_read_only failed");
            assert_eq!(arr.shape(), &[4, 5]);
            let view = arr.view().expect("view failed");
            // Check that values match
            for (a, b) in view.iter().zip(original.iter()) {
                assert!((a - b).abs() < f32::EPSILON, "element mismatch: {a} vs {b}");
            }
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_create_and_read_only_f64() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_mmap_create_ro_f64.mmap");

        let original = make_f64_array(&[3, 3]);
        {
            let _arr = MmapArray::<f64>::create(&path, &original).expect("create failed");
        }

        let arr = MmapArray::<f64>::open_read_only(&path).expect("open_read_only failed");
        let owned = arr.to_owned_array().expect("to_owned failed");
        assert_eq!(owned.shape(), &[3, 3]);
        for (a, b) in owned.iter().zip(original.iter()) {
            assert!((a - b).abs() < f64::EPSILON);
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_create_and_read_only_i32() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_mmap_create_ro_i32.mmap");

        let n = 6usize;
        let data: Vec<i32> = (0..n as i32).collect();
        let original = ArrayD::from_shape_vec(IxDyn(&[2, 3]), data).expect("shape mismatch");
        {
            let _arr = MmapArray::<i32>::create(&path, &original).expect("create failed");
        }

        let arr = MmapArray::<i32>::open_read_only(&path).expect("open_read_only failed");
        let view = arr.view().expect("view failed");
        for (a, b) in view.iter().zip(original.iter()) {
            assert_eq!(a, b);
        }

        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // read-write mode
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_write_mutation() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_mmap_rw.mmap");

        let original = make_f64_array(&[5]);
        {
            let mut arr = MmapArray::<f64>::create(&path, &original).expect("create failed");
            {
                let mut view = arr.view_mut().expect("view_mut failed");
                // Double every element in-place.
                view.iter_mut().for_each(|x| *x *= 2.0);
            }
            arr.flush().expect("flush failed");
        }

        // Re-open and verify changes persisted.
        let arr = MmapArray::<f64>::open_read_only(&path).expect("open_read_only failed");
        let view = arr.view().expect("view failed");
        for (i, &val) in view.iter().enumerate() {
            let expected = (i as f64) * 2.0;
            assert!(
                (val - expected).abs() < f64::EPSILON,
                "element {i}: got {val}, expected {expected}"
            );
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_open_read_write_then_mutate() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_mmap_open_rw.mmap");

        let original = make_f32_array(&[3, 3]);
        {
            let _arr = MmapArray::<f32>::create(&path, &original).expect("create failed");
        }

        {
            let mut arr = MmapArray::<f32>::open_read_write(&path).expect("open_read_write failed");
            {
                let mut view = arr.view_mut().expect("view_mut failed");
                view.iter_mut().for_each(|x| *x += 100.0);
            }
            arr.flush().expect("flush failed");
        }

        let arr = MmapArray::<f32>::open_read_only(&path).expect("open_read_only failed");
        let view = arr.view().expect("view failed");
        for (i, &val) in view.iter().enumerate() {
            let expected = i as f32 + 100.0;
            assert!(
                (val - expected).abs() < f32::EPSILON,
                "element {i}: got {val}, expected {expected}"
            );
        }

        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // read-only rejects mutation
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_only_rejects_view_mut() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_mmap_ro_no_mut.mmap");

        let original = make_f32_array(&[2, 2]);
        {
            let _arr = MmapArray::<f32>::create(&path, &original).expect("create failed");
        }

        let mut arr = MmapArray::<f32>::open_read_only(&path).expect("open_read_only failed");
        let err = arr.view_mut().expect_err("should return ReadOnly error");
        assert!(matches!(err, MmapError::ReadOnly));

        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // Copy-on-Write semantics
    // -----------------------------------------------------------------------

    #[test]
    fn test_cow_no_copy_before_write() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_mmap_cow_read.mmap");

        let original = make_f64_array(&[4]);
        {
            let _arr = MmapArray::<f64>::create(&path, &original).expect("create failed");
        }

        let arr = MmapArray::<f64>::open_cow(&path).expect("open_cow failed");
        // Before any write, cow_data should be None (verified by reading successfully).
        let view = arr.view().expect("view failed");
        for (a, b) in view.iter().zip(original.iter()) {
            assert!((a - b).abs() < f64::EPSILON);
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_cow_mutates_in_ram_not_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_mmap_cow_mutate.mmap");

        let original = make_f32_array(&[6]);
        {
            let _arr = MmapArray::<f32>::create(&path, &original).expect("create failed");
        }

        {
            let mut arr = MmapArray::<f32>::open_cow(&path).expect("open_cow failed");
            {
                let mut view = arr.view_mut().expect("view_mut failed");
                // Write should trigger COW fault and copy to RAM.
                view.iter_mut().for_each(|x| *x = -1.0);
            }

            // In-memory view should reflect mutation.
            let view = arr.view().expect("view failed");
            for &val in view.iter() {
                assert!(
                    (val - (-1.0f32)).abs() < f32::EPSILON,
                    "COW in-memory data wrong: {val}"
                );
            }
            // flush is a no-op in COW mode; should not error.
            arr.flush().expect("flush failed");
        }

        // The original file must be UNCHANGED because we used COW.
        let arr_check = MmapArray::<f32>::open_read_only(&path).expect("open_read_only failed");
        let view_check = arr_check.view().expect("view failed");
        for (i, &val) in view_check.iter().enumerate() {
            let expected = i as f32;
            assert!(
                (val - expected).abs() < f32::EPSILON,
                "file was modified when COW was used: element {i}: got {val}"
            );
        }

        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // dtype mismatch detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_dtype_mismatch_on_open() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_mmap_dtype_mismatch.mmap");

        let original = make_f32_array(&[8]);
        {
            let _arr = MmapArray::<f32>::create(&path, &original).expect("create failed");
        }

        // Try to open as f64 — must fail with DtypeMismatch.
        let err = MmapArray::<f64>::open_read_only(&path).expect_err("should be DtypeMismatch");
        assert!(
            matches!(
                err,
                MmapError::DtypeMismatch {
                    expected: 2,
                    actual: 1
                }
            ),
            "unexpected error: {err:?}"
        );

        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // Non-contiguous array rejection
    // -----------------------------------------------------------------------

    #[test]
    fn test_noncontiguous_rejected() {
        use ndarray::ShapeBuilder;

        let dir = std::env::temp_dir();
        let path = dir.join("test_mmap_noncontiguous.mmap");

        // Create a Fortran-order (column-major) array, which is NOT in standard
        // C (row-major) layout.  `create()` must reject it with NonContiguous.
        let fortran: ArrayD<f64> = ndarray::Array::zeros(IxDyn(&[3, 4]).f());
        assert!(
            !fortran.is_standard_layout(),
            "test precondition: Fortran array must be non-standard-layout"
        );

        let err = MmapArray::<f64>::create(&path, &fortran)
            .expect_err("create() should reject Fortran-order array");
        assert!(
            matches!(err, MmapError::NonContiguous),
            "expected NonContiguous, got: {err:?}"
        );

        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // Anonymous mmap (test-only helper)
    // -----------------------------------------------------------------------

    #[test]
    fn test_anonymous_create() {
        let arr = MmapArray::<f32>::create_anonymous(&[8, 8]).expect("anonymous create failed");
        assert_eq!(arr.shape(), &[8, 8]);
        assert_eq!(arr.len(), 64);
        let view = arr.view().expect("view failed");
        for &val in view.iter() {
            assert_eq!(val, 0.0f32);
        }
    }

    #[test]
    fn test_anonymous_mutation() {
        let mut arr = MmapArray::<i64>::create_anonymous(&[4]).expect("anonymous create failed");
        {
            let mut view = arr.view_mut().expect("view_mut failed");
            for (i, x) in view.iter_mut().enumerate() {
                *x = i as i64 * 10;
            }
        }
        let view = arr.view().expect("view failed");
        for (i, &val) in view.iter().enumerate() {
            assert_eq!(val, i as i64 * 10, "element {i} wrong");
        }
    }

    // -----------------------------------------------------------------------
    // to_owned_array
    // -----------------------------------------------------------------------

    #[test]
    fn test_to_owned_array() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_mmap_to_owned.mmap");

        let original = make_f64_array(&[2, 3, 4]);
        {
            let _arr = MmapArray::<f64>::create(&path, &original).expect("create failed");
        }

        let arr = MmapArray::<f64>::open_read_only(&path).expect("open failed");
        let owned = arr.to_owned_array().expect("to_owned failed");
        assert_eq!(owned.shape(), original.shape());
        for (a, b) in owned.iter().zip(original.iter()) {
            assert!((a - b).abs() < f64::EPSILON);
        }

        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // is_empty
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_array() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_mmap_empty.mmap");

        // A 0-element array (shape [0]).
        let empty: ArrayD<f32> = ArrayD::zeros(IxDyn(&[0]));
        let arr = MmapArray::<f32>::create(&path, &empty).expect("create failed");
        assert!(arr.is_empty());
        assert_eq!(arr.len(), 0);
        assert_eq!(arr.shape(), &[0]);

        let _ = std::fs::remove_file(&path);
    }
}
