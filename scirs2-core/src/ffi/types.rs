//! C ABI compatible types for SciRS2 FFI.
//!
//! This module defines `#[repr(C)]` structs that can be safely passed across the
//! FFI boundary. All types use C-compatible layouts and primitive types.
//!
//! # Safety Contract
//!
//! - All pointer fields may be null; callers must validate before use.
//! - Memory allocated by SciRS2 must be freed by SciRS2 (via `sci_vector_free`, etc.).
//! - Memory allocated by the caller must be freed by the caller.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

// ---------------------------------------------------------------------------
// SciVector
// ---------------------------------------------------------------------------

/// A contiguous 1-D array of `f64` values exposed over the C ABI.
///
/// # Memory ownership
///
/// When created by `sci_vector_new`, the `data` pointer is owned by SciRS2
/// and **must** be freed with `sci_vector_free`. The caller must not free the
/// pointer directly (e.g. via `free()`).
///
/// When the caller fills in a `SciVector` themselves (e.g. to pass input data),
/// they retain ownership and SciRS2 will never free that pointer.
#[repr(C)]
#[derive(Debug)]
pub struct SciVector {
    /// Pointer to the first element. May be null for a zero-length vector.
    pub data: *mut f64,
    /// Number of elements.
    pub len: usize,
}

impl SciVector {
    /// Create a new `SciVector` from a Rust `Vec<f64>`.
    ///
    /// The vector's memory is leaked intentionally so the C caller can use it.
    /// It must later be reclaimed via `sci_vector_free`.
    pub fn from_vec(mut v: Vec<f64>) -> Self {
        let len = v.len();
        let data = v.as_mut_ptr();
        std::mem::forget(v);
        SciVector { data, len }
    }

    /// Attempt to create a borrowed slice from this vector.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `data` is valid for `len` elements and
    /// that the memory will not be modified or freed while the slice exists.
    pub unsafe fn as_slice(&self) -> Option<&[f64]> {
        if self.data.is_null() {
            if self.len == 0 {
                return Some(&[]);
            }
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts(self.data, self.len) })
    }

    /// Attempt to create a mutable borrowed slice from this vector.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `data` is valid for `len` elements,
    /// that the memory will not be aliased, and that no other references exist.
    pub unsafe fn as_mut_slice(&mut self) -> Option<&mut [f64]> {
        if self.data.is_null() {
            if self.len == 0 {
                return Some(&mut []);
            }
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts_mut(self.data, self.len) })
    }
}

// ---------------------------------------------------------------------------
// SciMatrix
// ---------------------------------------------------------------------------

/// A contiguous 2-D array of `f64` values in **row-major** order.
///
/// The element at row `i`, column `j` is located at `data[i * cols + j]`.
///
/// # Memory ownership
///
/// Same conventions as [`SciVector`]: memory created by SciRS2 must be freed
/// by SciRS2 via `sci_matrix_free`.
#[repr(C)]
#[derive(Debug)]
pub struct SciMatrix {
    /// Pointer to the first element. May be null for a zero-size matrix.
    pub data: *mut f64,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
}

impl SciMatrix {
    /// Create a `SciMatrix` from a flat `Vec<f64>` with the given dimensions.
    ///
    /// Returns `None` if `rows * cols != v.len()`.
    pub fn from_vec(mut v: Vec<f64>, rows: usize, cols: usize) -> Option<Self> {
        if rows.checked_mul(cols) != Some(v.len()) {
            return None;
        }
        let data = v.as_mut_ptr();
        std::mem::forget(v);
        Some(SciMatrix { data, rows, cols })
    }

    /// Attempt to view the matrix data as a flat slice.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `data` is valid for `rows * cols` elements.
    pub unsafe fn as_slice(&self) -> Option<&[f64]> {
        let total = self.rows.checked_mul(self.cols)?;
        if self.data.is_null() {
            if total == 0 {
                return Some(&[]);
            }
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts(self.data, total) })
    }
}

// ---------------------------------------------------------------------------
// SciComplexVector -- for eigenvalue results etc.
// ---------------------------------------------------------------------------

/// A pair of vectors representing complex values: real parts + imaginary parts.
///
/// Used for operations like eigenvalue decomposition that return complex results.
#[repr(C)]
#[derive(Debug)]
pub struct SciComplexVector {
    /// Pointer to real parts.
    pub real: *mut f64,
    /// Pointer to imaginary parts.
    pub imag: *mut f64,
    /// Number of complex elements.
    pub len: usize,
}

impl SciComplexVector {
    /// Create from two `Vec<f64>` (real, imag). Both must have the same length.
    pub fn from_vecs(mut real: Vec<f64>, mut imag: Vec<f64>) -> Option<Self> {
        if real.len() != imag.len() {
            return None;
        }
        let len = real.len();
        let real_ptr = real.as_mut_ptr();
        let imag_ptr = imag.as_mut_ptr();
        std::mem::forget(real);
        std::mem::forget(imag);
        Some(SciComplexVector {
            real: real_ptr,
            imag: imag_ptr,
            len,
        })
    }
}

// ---------------------------------------------------------------------------
// SciResult
// ---------------------------------------------------------------------------

/// Status code returned by every FFI function.
///
/// If `success` is `true`, the output parameters contain valid data.
/// If `success` is `false`, `error_msg` points to a NUL-terminated C string
/// describing the error. The string is allocated by SciRS2 and must be freed
/// with `sci_free_error`.
#[repr(C)]
#[derive(Debug)]
pub struct SciResult {
    /// `true` if the operation succeeded.
    pub success: bool,
    /// A NUL-terminated error message, or null on success.
    /// Owned by SciRS2 -- free with `sci_free_error`.
    pub error_msg: *const c_char,
}

impl SciResult {
    /// Construct a success result.
    pub fn ok() -> Self {
        SciResult {
            success: true,
            error_msg: ptr::null(),
        }
    }

    /// Construct a failure result from an error message string.
    pub fn err(msg: &str) -> Self {
        let c_msg = CString::new(msg).unwrap_or_else(|_| {
            CString::new("(error message contained interior NUL byte)").unwrap_or_else(|_| {
                // This should be impossible since the fallback has no NUL bytes
                // but we satisfy the no-unwrap policy with a safe default.
                unsafe { CString::from_vec_unchecked(b"unknown error".to_vec()) }
            })
        });
        SciResult {
            success: false,
            error_msg: c_msg.into_raw(),
        }
    }

    /// Construct a failure result from a `Box<dyn std::any::Any + Send>` (from `catch_unwind`).
    pub fn from_panic(payload: Box<dyn std::any::Any + Send>) -> Self {
        let msg = if let Some(s) = payload.downcast_ref::<&str>() {
            format!("panic: {}", s)
        } else if let Some(s) = payload.downcast_ref::<String>() {
            format!("panic: {}", s)
        } else {
            "panic: unknown panic payload".to_string()
        };
        Self::err(&msg)
    }
}

// ---------------------------------------------------------------------------
// SciSvdResult
// ---------------------------------------------------------------------------

/// Result of an SVD decomposition: A = U * diag(S) * Vt.
#[repr(C)]
#[derive(Debug)]
pub struct SciSvdResult {
    /// U matrix (m x m), row-major.
    pub u: SciMatrix,
    /// Singular values vector (min(m,n) elements).
    pub s: SciVector,
    /// Vt matrix (n x n), row-major.
    pub vt: SciMatrix,
}

// ---------------------------------------------------------------------------
// SciEigResult
// ---------------------------------------------------------------------------

/// Result of an eigenvalue decomposition.
#[repr(C)]
#[derive(Debug)]
pub struct SciEigResult {
    /// Eigenvalues (complex).
    pub eigenvalues: SciComplexVector,
    /// Eigenvectors as a matrix (column-major eigenvectors stored row-major).
    pub eigenvectors: SciMatrix,
}

// ---------------------------------------------------------------------------
// Helper: free error message
// ---------------------------------------------------------------------------

/// Free an error message string that was allocated by SciRS2.
///
/// # Safety
///
/// `msg` must be a pointer returned by a SciRS2 FFI function in the
/// `error_msg` field of [`SciResult`], or null (which is a no-op).
/// Calling this with any other pointer is undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn sci_free_error(msg: *const c_char) {
    if !msg.is_null() {
        // Reclaim the CString
        let _ = unsafe { CString::from_raw(msg as *mut c_char) };
    }
}

/// Internal helper: validate that a string pointer is non-null and valid UTF-8.
///
/// Returns an owned `String` on success.
///
/// # Safety
///
/// The pointer must be a valid NUL-terminated C string.
pub(crate) unsafe fn str_from_c_ptr(ptr: *const c_char) -> Result<String, String> {
    if ptr.is_null() {
        return Err("null string pointer".to_string());
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map(|s| s.to_string())
        .map_err(|e| format!("invalid UTF-8: {}", e))
}
