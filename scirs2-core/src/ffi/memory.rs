//! Memory management functions for the C ABI FFI.
//!
//! These functions create and free `SciVector` and `SciMatrix` instances.
//! All memory allocated by these functions must be freed by the corresponding
//! `*_free` function. Mixing allocators (e.g. calling C `free()` on memory
//! returned by `sci_vector_new`) is undefined behavior.

use std::panic::catch_unwind;
use std::ptr;

use super::types::{SciComplexVector, SciMatrix, SciResult, SciVector};

// ---------------------------------------------------------------------------
// SciVector allocation / deallocation
// ---------------------------------------------------------------------------

/// Allocate a new `SciVector` of `len` elements, all initialized to zero.
///
/// On success, `*out` is filled with a valid `SciVector`. The caller must
/// eventually call `sci_vector_free` to release the memory.
///
/// # Safety
///
/// - `out` must be a valid, non-null pointer to a `SciVector`.
/// - The caller must not read `*out` until this function returns successfully.
#[no_mangle]
pub unsafe extern "C" fn sci_vector_new(len: usize, out: *mut SciVector) -> SciResult {
    if out.is_null() {
        return SciResult::err("sci_vector_new: out pointer is null");
    }

    let result = catch_unwind(|| {
        let v = vec![0.0f64; len];
        SciVector::from_vec(v)
    });

    match result {
        Ok(sv) => {
            unsafe { ptr::write(out, sv) };
            SciResult::ok()
        }
        Err(e) => SciResult::from_panic(e),
    }
}

/// Create a `SciVector` from existing data by **copying** `len` elements from `src`.
///
/// # Safety
///
/// - `src` must point to at least `len` valid `f64` elements.
/// - `out` must be a valid, non-null pointer to a `SciVector`.
#[no_mangle]
pub unsafe extern "C" fn sci_vector_from_data(
    src: *const f64,
    len: usize,
    out: *mut SciVector,
) -> SciResult {
    if out.is_null() {
        return SciResult::err("sci_vector_from_data: out pointer is null");
    }
    if src.is_null() && len > 0 {
        return SciResult::err("sci_vector_from_data: src is null but len > 0");
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let slice = if len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(src, len) }
        };
        SciVector::from_vec(slice.to_vec())
    }));

    match result {
        Ok(sv) => {
            unsafe { ptr::write(out, sv) };
            SciResult::ok()
        }
        Err(e) => SciResult::from_panic(e),
    }
}

/// Free a `SciVector` that was allocated by SciRS2.
///
/// After this call, the `SciVector`'s `data` pointer is invalid.
/// Passing a `SciVector` with a null `data` pointer is a safe no-op.
///
/// # Safety
///
/// - `vec` must be a valid pointer to a `SciVector` previously returned
///   by a SciRS2 FFI function.
/// - Must not be called twice on the same `SciVector`.
#[no_mangle]
pub unsafe extern "C" fn sci_vector_free(vec: *mut SciVector) {
    if vec.is_null() {
        return;
    }
    let sv = unsafe { ptr::read(vec) };
    if !sv.data.is_null() && sv.len > 0 {
        // Reconstruct the Vec so it is properly dropped.
        let _ = unsafe { Vec::from_raw_parts(sv.data, sv.len, sv.len) };
    }
    // Zero out the struct to prevent double-free.
    unsafe {
        (*vec).data = ptr::null_mut();
        (*vec).len = 0;
    }
}

// ---------------------------------------------------------------------------
// SciMatrix allocation / deallocation
// ---------------------------------------------------------------------------

/// Allocate a new `SciMatrix` of `rows x cols` elements, all initialized to zero.
///
/// On success, `*out` is filled with a valid `SciMatrix`. The caller must
/// eventually call `sci_matrix_free` to release the memory.
///
/// # Safety
///
/// - `out` must be a valid, non-null pointer to a `SciMatrix`.
#[no_mangle]
pub unsafe extern "C" fn sci_matrix_new(
    rows: usize,
    cols: usize,
    out: *mut SciMatrix,
) -> SciResult {
    if out.is_null() {
        return SciResult::err("sci_matrix_new: out pointer is null");
    }

    let result = catch_unwind(|| {
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| format!("sci_matrix_new: rows*cols overflow ({} x {})", rows, cols))?;
        let v = vec![0.0f64; total];
        SciMatrix::from_vec(v, rows, cols)
            .ok_or_else(|| "sci_matrix_new: internal dimension mismatch".to_string())
    });

    match result {
        Ok(Ok(sm)) => {
            unsafe { ptr::write(out, sm) };
            SciResult::ok()
        }
        Ok(Err(msg)) => SciResult::err(&msg),
        Err(e) => SciResult::from_panic(e),
    }
}

/// Create a `SciMatrix` from existing data by **copying** `rows * cols` elements from `src`.
///
/// The data is expected in row-major order.
///
/// # Safety
///
/// - `src` must point to at least `rows * cols` valid `f64` elements.
/// - `out` must be a valid, non-null pointer to a `SciMatrix`.
#[no_mangle]
pub unsafe extern "C" fn sci_matrix_from_data(
    src: *const f64,
    rows: usize,
    cols: usize,
    out: *mut SciMatrix,
) -> SciResult {
    if out.is_null() {
        return SciResult::err("sci_matrix_from_data: out pointer is null");
    }

    let total = match rows.checked_mul(cols) {
        Some(n) => n,
        None => {
            return SciResult::err("sci_matrix_from_data: rows*cols overflow");
        }
    };

    if src.is_null() && total > 0 {
        return SciResult::err("sci_matrix_from_data: src is null but size > 0");
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let slice = if total == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(src, total) }
        };
        SciMatrix::from_vec(slice.to_vec(), rows, cols)
            .ok_or_else(|| "sci_matrix_from_data: internal dimension mismatch".to_string())
    }));

    match result {
        Ok(Ok(sm)) => {
            unsafe { ptr::write(out, sm) };
            SciResult::ok()
        }
        Ok(Err(msg)) => SciResult::err(&msg),
        Err(e) => SciResult::from_panic(e),
    }
}

/// Free a `SciMatrix` that was allocated by SciRS2.
///
/// After this call, the `SciMatrix`'s `data` pointer is invalid.
///
/// # Safety
///
/// - `mat` must be a valid pointer to a `SciMatrix` previously returned
///   by a SciRS2 FFI function.
/// - Must not be called twice on the same `SciMatrix`.
#[no_mangle]
pub unsafe extern "C" fn sci_matrix_free(mat: *mut SciMatrix) {
    if mat.is_null() {
        return;
    }
    let sm = unsafe { ptr::read(mat) };
    let total = sm.rows.saturating_mul(sm.cols);
    if !sm.data.is_null() && total > 0 {
        let _ = unsafe { Vec::from_raw_parts(sm.data, total, total) };
    }
    unsafe {
        (*mat).data = ptr::null_mut();
        (*mat).rows = 0;
        (*mat).cols = 0;
    }
}

// ---------------------------------------------------------------------------
// SciComplexVector deallocation
// ---------------------------------------------------------------------------

/// Free a `SciComplexVector` that was allocated by SciRS2.
///
/// # Safety
///
/// - `cv` must be a valid pointer to a `SciComplexVector` previously returned
///   by a SciRS2 FFI function.
#[no_mangle]
pub unsafe extern "C" fn sci_complex_vector_free(cv: *mut SciComplexVector) {
    if cv.is_null() {
        return;
    }
    let c = unsafe { ptr::read(cv) };
    if !c.real.is_null() && c.len > 0 {
        let _ = unsafe { Vec::from_raw_parts(c.real, c.len, c.len) };
    }
    if !c.imag.is_null() && c.len > 0 {
        let _ = unsafe { Vec::from_raw_parts(c.imag, c.len, c.len) };
    }
    unsafe {
        (*cv).real = ptr::null_mut();
        (*cv).imag = ptr::null_mut();
        (*cv).len = 0;
    }
}

// ---------------------------------------------------------------------------
// SciSvdResult deallocation
// ---------------------------------------------------------------------------

/// Free a `SciSvdResult` that was allocated by SciRS2.
///
/// This frees all three components (U, S, Vt).
///
/// # Safety
///
/// - `svd` must be a valid pointer to a `SciSvdResult` previously returned
///   by `sci_svd`.
#[no_mangle]
pub unsafe extern "C" fn sci_svd_result_free(svd: *mut super::types::SciSvdResult) {
    if svd.is_null() {
        return;
    }
    unsafe {
        sci_matrix_free(&mut (*svd).u);
        sci_vector_free(&mut (*svd).s);
        sci_matrix_free(&mut (*svd).vt);
    }
}

/// Free a `SciEigResult` that was allocated by SciRS2.
///
/// # Safety
///
/// - `eig` must be a valid pointer to a `SciEigResult` previously returned
///   by `sci_eig`.
#[no_mangle]
pub unsafe extern "C" fn sci_eig_result_free(eig: *mut super::types::SciEigResult) {
    if eig.is_null() {
        return;
    }
    unsafe {
        sci_complex_vector_free(&mut (*eig).eigenvalues);
        sci_matrix_free(&mut (*eig).eigenvectors);
    }
}
