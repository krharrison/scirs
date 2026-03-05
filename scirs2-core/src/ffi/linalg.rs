//! Linear algebra FFI functions.
//!
//! These functions expose core linear algebra operations (determinant, inverse,
//! SVD, solve, eigenvalue decomposition) through a C-compatible ABI.
//!
//! All functions follow the SciRS2 FFI conventions:
//! - Return [`SciResult`] to indicate success/failure.
//! - Never panic across the FFI boundary (all panics caught with `catch_unwind`).
//! - Validate all pointers before dereferencing.
//! - Output parameters are written only on success.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

use super::types::{SciComplexVector, SciEigResult, SciMatrix, SciResult, SciSvdResult, SciVector};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a `SciMatrix` pointer to an ndarray `Array2<f64>`.
///
/// # Safety
///
/// The `SciMatrix`'s data pointer must be valid for `rows * cols` elements.
unsafe fn matrix_to_array2(m: *const SciMatrix) -> Result<::ndarray::Array2<f64>, String> {
    if m.is_null() {
        return Err("null matrix pointer".to_string());
    }
    let mat = unsafe { &*m };
    let total = mat
        .rows
        .checked_mul(mat.cols)
        .ok_or_else(|| "matrix dimension overflow".to_string())?;
    if mat.data.is_null() && total > 0 {
        return Err("matrix data pointer is null".to_string());
    }
    let slice = if total == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(mat.data, total) }
    };
    ::ndarray::Array2::from_shape_vec((mat.rows, mat.cols), slice.to_vec())
        .map_err(|e| format!("failed to create Array2: {}", e))
}

/// Convert a `SciVector` pointer to an ndarray `Array1<f64>`.
///
/// # Safety
///
/// The `SciVector`'s data pointer must be valid for `len` elements.
unsafe fn vector_to_array1(v: *const SciVector) -> Result<::ndarray::Array1<f64>, String> {
    if v.is_null() {
        return Err("null vector pointer".to_string());
    }
    let vec = unsafe { &*v };
    if vec.data.is_null() && vec.len > 0 {
        return Err("vector data pointer is null".to_string());
    }
    let slice = if vec.len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(vec.data, vec.len) }
    };
    Ok(::ndarray::Array1::from_vec(slice.to_vec()))
}

/// Convert an `Array2<f64>` to a `SciMatrix`, transferring ownership.
fn array2_to_sci_matrix(a: ::ndarray::Array2<f64>) -> Result<SciMatrix, String> {
    let (rows, cols) = (a.nrows(), a.ncols());
    // Ensure the array is in standard (row-major, contiguous) layout.
    let a = a.as_standard_layout().into_owned();
    let v: Vec<f64> = a.into_raw_vec_and_offset().0;
    SciMatrix::from_vec(v, rows, cols)
        .ok_or_else(|| "failed to convert Array2 to SciMatrix".to_string())
}

/// Convert an `Array1<f64>` to a `SciVector`, transferring ownership.
fn array1_to_sci_vector(a: ::ndarray::Array1<f64>) -> SciVector {
    SciVector::from_vec(a.into_raw_vec_and_offset().0)
}

// ---------------------------------------------------------------------------
// sci_det  --  determinant
// ---------------------------------------------------------------------------

/// Compute the determinant of a square matrix.
///
/// # Parameters
///
/// - `mat`: pointer to a `SciMatrix` (must be square).
/// - `out`: pointer to an `f64` where the determinant will be written.
///
/// # Safety
///
/// - `mat` must point to a valid `SciMatrix` with valid data.
/// - `out` must be a valid, non-null pointer to `f64`.
#[no_mangle]
pub unsafe extern "C" fn sci_det(mat: *const SciMatrix, out: *mut f64) -> SciResult {
    if out.is_null() {
        return SciResult::err("sci_det: out pointer is null");
    }

    let result = catch_unwind(AssertUnwindSafe(|| -> Result<f64, String> {
        let a = unsafe { matrix_to_array2(mat) }?;
        if a.nrows() != a.ncols() {
            return Err(format!(
                "sci_det: matrix must be square, got {}x{}",
                a.nrows(),
                a.ncols()
            ));
        }
        crate::linalg::det_ndarray(&a).map_err(|e| format!("sci_det: {}", e))
    }));

    match result {
        Ok(Ok(det)) => {
            unsafe { ptr::write(out, det) };
            SciResult::ok()
        }
        Ok(Err(ref msg)) => SciResult::err(msg),
        Err(e) => SciResult::from_panic(e),
    }
}

// ---------------------------------------------------------------------------
// sci_inv  --  matrix inverse
// ---------------------------------------------------------------------------

/// Compute the inverse of a square matrix.
///
/// On success, `*out` is filled with the inverse matrix. The caller must
/// free it with `sci_matrix_free`.
///
/// # Safety
///
/// - `mat` must point to a valid square `SciMatrix`.
/// - `out` must be a valid, non-null pointer to `SciMatrix`.
#[no_mangle]
pub unsafe extern "C" fn sci_inv(mat: *const SciMatrix, out: *mut SciMatrix) -> SciResult {
    if out.is_null() {
        return SciResult::err("sci_inv: out pointer is null");
    }

    let result = catch_unwind(AssertUnwindSafe(|| -> Result<SciMatrix, String> {
        let a = unsafe { matrix_to_array2(mat) }?;
        if a.nrows() != a.ncols() {
            return Err(format!(
                "sci_inv: matrix must be square, got {}x{}",
                a.nrows(),
                a.ncols()
            ));
        }
        let inv = crate::linalg::inv_ndarray(&a).map_err(|e| format!("sci_inv: {}", e))?;
        array2_to_sci_matrix(inv)
    }));

    match result {
        Ok(Ok(sm)) => {
            unsafe { ptr::write(out, sm) };
            SciResult::ok()
        }
        Ok(Err(ref msg)) => SciResult::err(msg),
        Err(e) => SciResult::from_panic(e),
    }
}

// ---------------------------------------------------------------------------
// sci_svd  --  singular value decomposition
// ---------------------------------------------------------------------------

/// Compute the full SVD of a matrix: A = U * diag(S) * Vt.
///
/// On success, `*out` is filled with the SVD result. The caller must
/// free it with `sci_svd_result_free`.
///
/// # Safety
///
/// - `mat` must point to a valid `SciMatrix`.
/// - `out` must be a valid, non-null pointer to `SciSvdResult`.
#[no_mangle]
pub unsafe extern "C" fn sci_svd(mat: *const SciMatrix, out: *mut SciSvdResult) -> SciResult {
    if out.is_null() {
        return SciResult::err("sci_svd: out pointer is null");
    }

    let result = catch_unwind(AssertUnwindSafe(|| -> Result<SciSvdResult, String> {
        let a = unsafe { matrix_to_array2(mat) }?;
        let svd = crate::linalg::svd_ndarray(&a).map_err(|e| format!("sci_svd: {}", e))?;

        let u_mat = array2_to_sci_matrix(svd.u)?;
        let s_vec = array1_to_sci_vector(svd.s);
        let vt_mat = array2_to_sci_matrix(svd.vt)?;

        Ok(SciSvdResult {
            u: u_mat,
            s: s_vec,
            vt: vt_mat,
        })
    }));

    match result {
        Ok(Ok(svd_result)) => {
            unsafe { ptr::write(out, svd_result) };
            SciResult::ok()
        }
        Ok(Err(ref msg)) => SciResult::err(msg),
        Err(e) => SciResult::from_panic(e),
    }
}

// ---------------------------------------------------------------------------
// sci_solve  --  solve linear system Ax = b
// ---------------------------------------------------------------------------

/// Solve the linear system A * x = b for x.
///
/// On success, `*out` is filled with the solution vector. The caller must
/// free it with `sci_vector_free`.
///
/// # Parameters
///
/// - `a`: pointer to a square coefficient matrix A.
/// - `b`: pointer to the right-hand side vector b.
/// - `out`: pointer to a `SciVector` for the solution.
///
/// # Safety
///
/// - `a` must point to a valid square `SciMatrix`.
/// - `b` must point to a valid `SciVector` with length equal to the number of rows in A.
/// - `out` must be a valid, non-null pointer to `SciVector`.
#[no_mangle]
pub unsafe extern "C" fn sci_solve(
    a: *const SciMatrix,
    b: *const SciVector,
    out: *mut SciVector,
) -> SciResult {
    if out.is_null() {
        return SciResult::err("sci_solve: out pointer is null");
    }

    let result = catch_unwind(AssertUnwindSafe(|| -> Result<SciVector, String> {
        let a_arr = unsafe { matrix_to_array2(a) }?;
        let b_arr = unsafe { vector_to_array1(b) }?;

        if a_arr.nrows() != a_arr.ncols() {
            return Err(format!(
                "sci_solve: A must be square, got {}x{}",
                a_arr.nrows(),
                a_arr.ncols()
            ));
        }
        if a_arr.nrows() != b_arr.len() {
            return Err(format!(
                "sci_solve: dimension mismatch: A is {}x{} but b has {} elements",
                a_arr.nrows(),
                a_arr.ncols(),
                b_arr.len()
            ));
        }

        let x = crate::linalg::solve_ndarray(&a_arr, &b_arr)
            .map_err(|e| format!("sci_solve: {}", e))?;
        Ok(array1_to_sci_vector(x))
    }));

    match result {
        Ok(Ok(sv)) => {
            unsafe { ptr::write(out, sv) };
            SciResult::ok()
        }
        Ok(Err(ref msg)) => SciResult::err(msg),
        Err(e) => SciResult::from_panic(e),
    }
}

// ---------------------------------------------------------------------------
// sci_eig  --  eigenvalue decomposition
// ---------------------------------------------------------------------------

/// Compute the eigenvalue decomposition of a square matrix.
///
/// Returns complex eigenvalues and eigenvectors. For real symmetric matrices,
/// the imaginary parts will be zero.
///
/// On success, `*out` is filled with the eigenvalue result. The caller must
/// free it with `sci_eig_result_free`.
///
/// # Safety
///
/// - `mat` must point to a valid square `SciMatrix`.
/// - `out` must be a valid, non-null pointer to `SciEigResult`.
#[no_mangle]
pub unsafe extern "C" fn sci_eig(mat: *const SciMatrix, out: *mut SciEigResult) -> SciResult {
    if out.is_null() {
        return SciResult::err("sci_eig: out pointer is null");
    }

    let result = catch_unwind(AssertUnwindSafe(|| -> Result<SciEigResult, String> {
        let a = unsafe { matrix_to_array2(mat) }?;
        if a.nrows() != a.ncols() {
            return Err(format!(
                "sci_eig: matrix must be square, got {}x{}",
                a.nrows(),
                a.ncols()
            ));
        }

        let eig = crate::linalg::eig_ndarray(&a).map_err(|e| format!("sci_eig: {}", e))?;

        // Convert complex eigenvalues to separate real/imag vectors
        let n = eig.eigenvalues.len();
        let mut real_parts = Vec::with_capacity(n);
        let mut imag_parts = Vec::with_capacity(n);
        for ev in &eig.eigenvalues {
            real_parts.push(ev.real);
            imag_parts.push(ev.imag);
        }

        let eigenvalues = SciComplexVector::from_vecs(real_parts, imag_parts)
            .ok_or_else(|| "sci_eig: failed to create complex vector".to_string())?;

        // Convert right eigenvector matrix (real part); imaginary parts are discarded in FFI
        let eigvec_real = eig
            .eigenvectors_real
            .ok_or_else(|| "sci_eig: eigenvectors not computed".to_string())?;
        let eigvec_mat =
            array2_to_sci_matrix(eigvec_real).map_err(|e| format!("sci_eig: {}", e))?;

        Ok(SciEigResult {
            eigenvalues,
            eigenvectors: eigvec_mat,
        })
    }));

    match result {
        Ok(Ok(eig_result)) => {
            unsafe { ptr::write(out, eig_result) };
            SciResult::ok()
        }
        Ok(Err(ref msg)) => SciResult::err(msg),
        Err(e) => SciResult::from_panic(e),
    }
}
