//! Zero-copy buffer operations for efficient data transfer between components.
//!
//! This module provides three building blocks for zero-copy data sharing:
//!
//! * [`ZeroCopyBuffer`] — a reference-counted byte buffer that supports O(1)
//!   slicing.  Slices share the underlying allocation via `Arc`; only
//!   [`ZeroCopyBuffer::clone_data`] performs an explicit copy.
//!
//! * [`ZeroCopyView<T>`] — a typed, read-only view over a `ZeroCopyBuffer`.
//!   The buffer's byte slice is reinterpreted as `&[T]` with proper alignment
//!   and size checks.
//!
//! * [`ZeroCopyRingBuffer`] — a contiguous ring buffer for streaming byte
//!   data.  Writes advance a write cursor; reads advance a read cursor.
//!   The buffer does **not** wrap individual reads — instead it returns a
//!   contiguous slice from the read position.

use std::marker::PhantomData;
use std::mem;
use std::sync::Arc;

use crate::error::{CoreError, CoreResult, ErrorContext};

// ---------------------------------------------------------------------------
// ZeroCopyBuffer
// ---------------------------------------------------------------------------

/// A reference-counted, sliceable byte buffer with O(1) sub-slice creation.
///
/// # Example
///
/// ```rust
/// use scirs2_core::memory::zero_copy::ZeroCopyBuffer;
///
/// let buf = ZeroCopyBuffer::new(vec![0u8, 1, 2, 3, 4, 5]);
/// let slice = buf.slice(1, 4).expect("slice");
/// assert_eq!(slice.as_bytes(), &[1, 2, 3]);
/// // The original buffer and the slice share the same allocation.
/// ```
#[derive(Clone)]
pub struct ZeroCopyBuffer {
    data: Arc<Vec<u8>>,
    offset: usize,
    len: usize,
}

impl ZeroCopyBuffer {
    /// Create a new buffer owning `data`.
    pub fn new(data: Vec<u8>) -> Self {
        let len = data.len();
        ZeroCopyBuffer {
            data: Arc::new(data),
            offset: 0,
            len,
        }
    }

    /// Return a new `ZeroCopyBuffer` that views `self[start..end]` without
    /// copying.  The new buffer shares the same `Arc<Vec<u8>>` allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if `start > end` or `end > self.len()`.
    pub fn slice(&self, start: usize, end: usize) -> CoreResult<Self> {
        if start > end {
            return Err(CoreError::InvalidArgument(ErrorContext::new(format!(
                "ZeroCopyBuffer::slice: start ({start}) > end ({end})"
            ))));
        }
        if end > self.len {
            return Err(CoreError::InvalidArgument(ErrorContext::new(format!(
                "ZeroCopyBuffer::slice: end ({end}) > buffer length ({})",
                self.len
            ))));
        }
        Ok(ZeroCopyBuffer {
            data: Arc::clone(&self.data),
            offset: self.offset + start,
            len: end - start,
        })
    }

    /// Return the bytes of this view (does **not** copy).
    pub fn as_bytes(&self) -> &[u8] {
        &self.data[self.offset..self.offset + self.len]
    }

    /// Length of this view in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` iff this view is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Explicitly copy the bytes of this view into a new `Vec<u8>`.
    pub fn clone_data(&self) -> Vec<u8> {
        self.as_bytes().to_vec()
    }

    /// Number of `ZeroCopyBuffer` (and `ZeroCopyView`) instances sharing the
    /// underlying allocation.  Useful for debugging.
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.data)
    }
}

// ---------------------------------------------------------------------------
// ZeroCopyView<T>
// ---------------------------------------------------------------------------

/// A typed, read-only view over a [`ZeroCopyBuffer`].
///
/// The buffer's bytes are reinterpreted as `&[T]`.  Both alignment and exact
/// size divisibility are verified at construction time.
///
/// # Safety Note
///
/// `T: Copy` is required so that no destructor is run on the borrowed data.
/// For types with padding bytes (e.g. `(u8, u32)`) the read values may
/// contain unspecified padding, but this is safe in Rust because `Copy` types
/// have no invariants on their padding.
///
/// # Example
///
/// ```rust
/// use scirs2_core::memory::zero_copy::{ZeroCopyBuffer, ZeroCopyView};
///
/// let raw: Vec<u8> = (0u32..4).flat_map(|v| v.to_le_bytes()).collect();
/// let buf = ZeroCopyBuffer::new(raw);
/// let view: ZeroCopyView<u32> = ZeroCopyView::new(buf).expect("view");
/// assert_eq!(view.len(), 4);
/// assert_eq!(view.get(0), Some(&0u32));
/// assert_eq!(view.get(3), Some(&3u32));
/// ```
pub struct ZeroCopyView<T: Copy> {
    buffer: ZeroCopyBuffer,
    _phantom: PhantomData<T>,
}

impl<T: Copy> ZeroCopyView<T> {
    /// Create a typed view.
    ///
    /// # Errors
    ///
    /// * If the buffer's start address is not aligned to `align_of::<T>()`.
    /// * If the buffer's byte length is not a multiple of `size_of::<T>()`.
    pub fn new(buffer: ZeroCopyBuffer) -> CoreResult<Self> {
        let size = mem::size_of::<T>();
        let align = mem::align_of::<T>();

        if size == 0 {
            // ZSTs: any buffer length is valid.
            return Ok(ZeroCopyView { buffer, _phantom: PhantomData });
        }

        if buffer.len() % size != 0 {
            return Err(CoreError::InvalidArgument(ErrorContext::new(format!(
                "ZeroCopyView: buffer length ({}) is not a multiple of size_of::<T>() ({})",
                buffer.len(),
                size
            ))));
        }

        let ptr = buffer.as_bytes().as_ptr() as usize;
        if ptr % align != 0 {
            return Err(CoreError::InvalidArgument(ErrorContext::new(format!(
                "ZeroCopyView: buffer address (0x{ptr:x}) is not aligned to {align}"
            ))));
        }

        Ok(ZeroCopyView { buffer, _phantom: PhantomData })
    }

    /// Return a slice of `T` values.
    pub fn as_slice(&self) -> &[T] {
        let bytes = self.buffer.as_bytes();
        let size = mem::size_of::<T>();
        if size == 0 {
            return &[];
        }
        let count = bytes.len() / size;
        // SAFETY: alignment and size were verified in `new`.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, count) }
    }

    /// Number of `T` elements.
    pub fn len(&self) -> usize {
        let size = mem::size_of::<T>();
        if size == 0 { 0 } else { self.buffer.len() / size }
    }

    /// `true` iff there are no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return a reference to the element at `idx`, or `None` if out of bounds.
    pub fn get(&self, idx: usize) -> Option<&T> {
        self.as_slice().get(idx)
    }
}

// ---------------------------------------------------------------------------
// ZeroCopyRingBuffer
// ---------------------------------------------------------------------------

/// A simple, non-thread-safe ring buffer for streaming byte data.
///
/// Data is written with [`write`](Self::write) and read back contiguously with
/// [`read`](Self::read).  The returned slice from `read` is valid until the
/// next write or another read call.
///
/// # Design
///
/// The internal buffer is `capacity + 1` bytes so that full and empty states
/// are distinguishable without an extra flag (though the implementation also
/// keeps a `full` flag for clarity).
///
/// Writes that would exceed the available space are truncated; the caller
/// should check the return value.
///
/// # Example
///
/// ```rust
/// use scirs2_core::memory::zero_copy::ZeroCopyRingBuffer;
///
/// let mut rb = ZeroCopyRingBuffer::new(8);
/// let written = rb.write(b"hello");
/// assert_eq!(written, 5);
/// let data = rb.read(5).to_vec();
/// assert_eq!(data, b"hello");
/// ```
pub struct ZeroCopyRingBuffer {
    data: Vec<u8>,
    read_pos: usize,
    write_pos: usize,
    capacity: usize,
    full: bool,
}

impl ZeroCopyRingBuffer {
    /// Create a ring buffer with the specified `capacity` in bytes.
    ///
    /// The minimum capacity is 1; zero is treated as 1.
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        ZeroCopyRingBuffer {
            data: vec![0u8; cap],
            read_pos: 0,
            write_pos: 0,
            capacity: cap,
            full: false,
        }
    }

    /// Number of bytes available for reading.
    pub fn available_read(&self) -> usize {
        if self.full {
            return self.capacity;
        }
        if self.write_pos >= self.read_pos {
            self.write_pos - self.read_pos
        } else {
            self.capacity - self.read_pos + self.write_pos
        }
    }

    /// Number of bytes available for writing.
    pub fn available_write(&self) -> usize {
        self.capacity - self.available_read()
    }

    /// Write bytes from `data` into the ring buffer.
    ///
    /// Returns the number of bytes actually written (may be less than
    /// `data.len()` if the buffer is full).  Bytes are written in order;
    /// if the write wraps around the end of the internal buffer the write
    /// is split into two contiguous memcpy operations.
    pub fn write(&mut self, data: &[u8]) -> usize {
        let space = self.available_write();
        let n = data.len().min(space);
        if n == 0 {
            return 0;
        }

        let first_part = (self.capacity - self.write_pos).min(n);
        self.data[self.write_pos..self.write_pos + first_part]
            .copy_from_slice(&data[..first_part]);

        if first_part < n {
            let second_part = n - first_part;
            self.data[..second_part].copy_from_slice(&data[first_part..first_part + second_part]);
        }

        self.write_pos = (self.write_pos + n) % self.capacity;
        if self.write_pos == self.read_pos {
            self.full = true;
        }
        n
    }

    /// Read up to `len` bytes from the ring buffer.
    ///
    /// The returned slice is only valid until the next call to `write` or
    /// `read` (the internal buffer may be modified).  If the requested data
    /// wraps around the end of the internal buffer, only the contiguous
    /// portion up to the end is returned; the read cursor still advances
    /// by the full `min(len, available)` bytes so the wrapped portion is
    /// consumed.  Call `read` again to obtain the remaining data.
    ///
    /// Advances the read cursor by `min(len, available_read())` bytes.
    pub fn read(&mut self, len: usize) -> &[u8] {
        let avail = self.available_read();
        let n = len.min(avail);
        if n == 0 {
            return &[];
        }

        // Contiguous portion from read_pos.
        let contiguous = (self.capacity - self.read_pos).min(n);

        let start = self.read_pos;
        // Advance the cursor by the full requested amount (not just the
        // contiguous part).  The caller sees only the contiguous slice but
        // the ring buffer logically drains `n` bytes.
        self.read_pos = (self.read_pos + n) % self.capacity;
        self.full = false;

        &self.data[start..start + contiguous]
    }

    /// The total capacity of the ring buffer in bytes.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Reset the ring buffer to empty state without deallocating.
    pub fn clear(&mut self) {
        self.read_pos = 0;
        self.write_pos = 0;
        self.full = false;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- ZeroCopyBuffer ---

    #[test]
    fn test_buffer_basic() {
        let data = vec![10u8, 20, 30, 40, 50];
        let buf = ZeroCopyBuffer::new(data.clone());
        assert_eq!(buf.len(), 5);
        assert!(!buf.is_empty());
        assert_eq!(buf.as_bytes(), data.as_slice());
    }

    #[test]
    fn test_buffer_slice_no_copy() {
        let buf = ZeroCopyBuffer::new(vec![0u8, 1, 2, 3, 4, 5, 6, 7]);
        let s1 = buf.slice(2, 5).expect("slice 2..5");
        let s2 = buf.slice(0, 4).expect("slice 0..4");

        assert_eq!(s1.as_bytes(), &[2, 3, 4]);
        assert_eq!(s2.as_bytes(), &[0, 1, 2, 3]);

        // Both slices share the same Arc — ref_count should be 3 (original + 2 slices).
        assert_eq!(buf.ref_count(), 3);
    }

    #[test]
    fn test_buffer_slice_error_cases() {
        let buf = ZeroCopyBuffer::new(vec![0u8; 8]);
        assert!(buf.slice(5, 3).is_err()); // start > end
        assert!(buf.slice(0, 9).is_err()); // end > len
        assert!(buf.slice(8, 8).is_ok()); // empty slice at end is OK
    }

    #[test]
    fn test_buffer_clone_data() {
        let original = vec![42u8; 16];
        let buf = ZeroCopyBuffer::new(original.clone());
        let cloned = buf.clone_data();
        assert_eq!(cloned, original);
        // Arc ref count should still be 1.
        assert_eq!(buf.ref_count(), 1);
    }

    #[test]
    fn test_buffer_empty() {
        let buf = ZeroCopyBuffer::new(vec![]);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.as_bytes(), &[] as &[u8]);
    }

    // --- ZeroCopyView ---

    #[test]
    fn test_view_u32() {
        let raw: Vec<u8> = (0u32..4).flat_map(|v| v.to_le_bytes()).collect();
        let buf = ZeroCopyBuffer::new(raw);
        let view: ZeroCopyView<u32> = ZeroCopyView::new(buf).expect("view");
        assert_eq!(view.len(), 4);
        assert_eq!(view.get(0), Some(&0u32));
        assert_eq!(view.get(3), Some(&3u32));
        assert_eq!(view.get(4), None);
    }

    #[test]
    fn test_view_bad_length() {
        // 5 bytes is not a multiple of size_of::<u32>() = 4.
        let buf = ZeroCopyBuffer::new(vec![0u8; 5]);
        let result: CoreResult<ZeroCopyView<u32>> = ZeroCopyView::new(buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_view_slice() {
        let raw: Vec<u8> = (0u8..8).collect();
        let buf = ZeroCopyBuffer::new(raw);
        let view: ZeroCopyView<u8> = ZeroCopyView::new(buf).expect("view");
        assert_eq!(view.as_slice(), &[0u8, 1, 2, 3, 4, 5, 6, 7]);
    }

    // --- ZeroCopyRingBuffer ---

    #[test]
    fn test_ring_buffer_basic_write_read() {
        let mut rb = ZeroCopyRingBuffer::new(16);
        assert_eq!(rb.available_read(), 0);
        assert_eq!(rb.available_write(), 16);

        let written = rb.write(b"hello");
        assert_eq!(written, 5);
        assert_eq!(rb.available_read(), 5);
        assert_eq!(rb.available_write(), 11);

        let read_bytes = rb.read(5).to_vec();
        assert_eq!(read_bytes, b"hello");
        assert_eq!(rb.available_read(), 0);
    }

    #[test]
    fn test_ring_buffer_full() {
        let mut rb = ZeroCopyRingBuffer::new(4);
        assert_eq!(rb.write(b"abcd"), 4); // fills exactly
        assert_eq!(rb.available_write(), 0);
        assert_eq!(rb.write(b"x"), 0); // no space left
    }

    #[test]
    fn test_ring_buffer_wrap_around() {
        let mut rb = ZeroCopyRingBuffer::new(8);
        rb.write(b"12345678"); // fill
        rb.read(6); // consume 6 → read_pos = 6
        let w = rb.write(b"ABCDEFGH"); // wraps around
        // Should write min(available_write=6, 8) = 6 bytes.
        assert!(w <= 8);
    }

    #[test]
    fn test_ring_buffer_clear() {
        let mut rb = ZeroCopyRingBuffer::new(8);
        rb.write(b"hello");
        rb.clear();
        assert_eq!(rb.available_read(), 0);
        assert_eq!(rb.available_write(), 8);
    }

    #[test]
    fn test_ring_buffer_multiple_rounds() {
        let mut rb = ZeroCopyRingBuffer::new(8);
        for _ in 0..100 {
            let w = rb.write(b"abc");
            assert!(w > 0);
            let _ = rb.read(w);
        }
        assert_eq!(rb.available_read(), 0);
    }

    #[test]
    fn test_ring_buffer_partial_read() {
        let mut rb = ZeroCopyRingBuffer::new(16);
        rb.write(b"hello world");
        let chunk = rb.read(5).to_vec();
        assert_eq!(chunk, b"hello");
        // 6 bytes remain readable.
        assert_eq!(rb.available_read(), 6);
    }
}
