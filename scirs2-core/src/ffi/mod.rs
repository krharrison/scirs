//! C ABI FFI module for SciRS2.
//!
//! This module enables calling SciRS2 from C, Julia, Python (via ctypes/cffi),
//! and any other language that supports the C calling convention.
//!
//! # Feature Gate
//!
//! This module is only available when the `ffi` feature is enabled:
//!
//! ```toml
//! [dependencies]
//! scirs2-core = { version = "0.3.0", features = ["ffi"] }
//! ```
//!
//! # Architecture
//!
//! The module is organized into sub-modules by domain:
//!
//! | Module    | Description |
//! |-----------|-------------|
//! | `types`   | `#[repr(C)]` data structures (`SciVector`, `SciMatrix`, `SciResult`, etc.) |
//! | `memory`  | Allocation and deallocation functions |
//! | `linalg`  | Linear algebra: determinant, inverse, SVD, solve, eigenvalues |
//! | `stats`   | Descriptive statistics: mean, std, percentile, correlation |
//! | `fft`     | Fast Fourier Transform (forward and inverse) |
//! | `optimize`| Scalar minimization and root-finding |
//!
//! # Safety Guarantees
//!
//! All `extern "C"` functions in this module uphold the following invariants:
//!
//! 1. **No panics cross the FFI boundary.** Every function body is wrapped in
//!    `std::panic::catch_unwind`. If a panic occurs, the function returns a
//!    `SciResult` with `success = false` and an error message.
//!
//! 2. **All pointers are validated before use.** Null output pointers result in
//!    an immediate error return. Null input pointers are checked before
//!    dereferencing.
//!
//! 3. **Memory ownership is explicit.** Memory allocated by SciRS2 must be freed
//!    by SciRS2 (via the corresponding `*_free` function). Memory allocated by
//!    the caller is never freed by SciRS2.
//!
//! # Example (C)
//!
//! ```c
//! #include <stdio.h>
//! #include <stdlib.h>
//!
//! // Declarations (would normally come from a generated header)
//! typedef struct { double* data; size_t len; } SciVector;
//! typedef struct { int success; const char* error_msg; } SciResult;
//!
//! extern SciResult sci_vector_new(size_t len, SciVector* out);
//! extern SciResult sci_mean(const SciVector* vec, double* out);
//! extern void sci_vector_free(SciVector* vec);
//! extern void sci_free_error(const char* msg);
//!
//! int main() {
//!     SciVector v;
//!     SciResult r = sci_vector_new(5, &v);
//!     if (!r.success) {
//!         fprintf(stderr, "Error: %s\n", r.error_msg);
//!         sci_free_error(r.error_msg);
//!         return 1;
//!     }
//!
//!     // Fill with data
//!     for (size_t i = 0; i < v.len; i++) v.data[i] = (double)(i + 1);
//!
//!     double mean;
//!     r = sci_mean(&v, &mean);
//!     if (r.success) {
//!         printf("Mean: %f\n", mean);  // 3.0
//!     }
//!
//!     sci_vector_free(&v);
//!     return 0;
//! }
//! ```
//!
//! # Example (Julia)
//!
//! ```julia
//! # Load the shared library
//! const libscirs = "path/to/libscirs2_core.dylib"
//!
//! struct SciVector
//!     data::Ptr{Float64}
//!     len::Csize_t
//! end
//!
//! struct SciResult
//!     success::Bool
//!     error_msg::Ptr{Cchar}
//! end
//!
//! function sci_mean(v::SciVector)
//!     out = Ref{Float64}(0.0)
//!     r = ccall((:sci_mean, libscirs), SciResult, (Ref{SciVector}, Ptr{Float64}), Ref(v), out)
//!     r.success || error(unsafe { Base.unsafe_string(r.error_msg) })
//!     return out[]
//! end
//! ```

pub mod fft;
pub mod memory;
pub mod optimize;
pub mod stats;
pub mod types;

// Conditionally include linalg FFI only when the linalg feature is also enabled,
// since it depends on OxiBLAS.
#[cfg(feature = "linalg")]
pub mod linalg;

// Re-export all public types and functions for flat access via `ffi::*`.
pub use self::fft::*;
pub use self::memory::*;
pub use self::optimize::*;
pub use self::stats::*;
pub use self::types::*;

#[cfg(feature = "linalg")]
pub use self::linalg::*;

// ---------------------------------------------------------------------------
// Version introspection
// ---------------------------------------------------------------------------

/// Return the SciRS2 core version as a NUL-terminated C string.
///
/// The returned pointer is a static string and must **not** be freed.
///
/// # Safety
///
/// The returned pointer is valid for the lifetime of the process.
#[no_mangle]
pub extern "C" fn sci_version() -> *const std::os::raw::c_char {
    // Using a static byte string avoids allocation.
    static VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();
    VERSION.as_ptr() as *const std::os::raw::c_char
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    // -----------------------------------------------------------------------
    // Memory management tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_vector_new_and_free() {
        let mut v = SciVector {
            data: ptr::null_mut(),
            len: 0,
        };
        // Safety: `v` is a valid local variable.
        let r = unsafe { sci_vector_new(10, &mut v) };
        assert!(r.success);
        assert_eq!(v.len, 10);
        assert!(!v.data.is_null());

        // Verify all elements are zero.
        // Safety: sci_vector_new guarantees the data pointer is valid for 10 elements.
        let slice = unsafe { std::slice::from_raw_parts(v.data, v.len) };
        for &val in slice {
            assert_eq!(val, 0.0);
        }

        // Safety: the vector was allocated by sci_vector_new.
        unsafe { sci_vector_free(&mut v) };
        assert!(v.data.is_null());
        assert_eq!(v.len, 0);
    }

    #[test]
    fn test_vector_from_data() {
        let src = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut v = SciVector {
            data: ptr::null_mut(),
            len: 0,
        };
        // Safety: src is a valid array, v is a valid local variable.
        let r = unsafe { sci_vector_from_data(src.as_ptr(), src.len(), &mut v) };
        assert!(r.success);
        assert_eq!(v.len, 5);

        // Verify the data was copied.
        let slice = unsafe { std::slice::from_raw_parts(v.data, v.len) };
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0, 5.0]);

        unsafe { sci_vector_free(&mut v) };
    }

    #[test]
    fn test_vector_new_null_out() {
        // Safety: passing null intentionally to test error handling.
        let r = unsafe { sci_vector_new(10, ptr::null_mut()) };
        assert!(!r.success);
        assert!(!r.error_msg.is_null());
        // Safety: error_msg was allocated by SciResult::err.
        unsafe { sci_free_error(r.error_msg) };
    }

    #[test]
    fn test_matrix_new_and_free() {
        let mut m = SciMatrix {
            data: ptr::null_mut(),
            rows: 0,
            cols: 0,
        };
        // Safety: m is a valid local variable.
        let r = unsafe { sci_matrix_new(3, 4, &mut m) };
        assert!(r.success);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 4);
        assert!(!m.data.is_null());

        let slice = unsafe { std::slice::from_raw_parts(m.data, m.rows * m.cols) };
        for &val in slice {
            assert_eq!(val, 0.0);
        }

        unsafe { sci_matrix_free(&mut m) };
        assert!(m.data.is_null());
    }

    #[test]
    fn test_matrix_from_data() {
        // 2x3 matrix: [[1,2,3],[4,5,6]]
        let src = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut m = SciMatrix {
            data: ptr::null_mut(),
            rows: 0,
            cols: 0,
        };
        let r = unsafe { sci_matrix_from_data(src.as_ptr(), 2, 3, &mut m) };
        assert!(r.success);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);

        let slice = unsafe { std::slice::from_raw_parts(m.data, m.rows * m.cols) };
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        unsafe { sci_matrix_free(&mut m) };
    }

    // -----------------------------------------------------------------------
    // Statistics tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mean() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let v = SciVector {
            data: data.as_ptr() as *mut f64,
            len: data.len(),
        };
        let mut result = 0.0;
        // Safety: v.data points to valid data, result is a valid pointer.
        let r = unsafe { sci_mean(&v, &mut result) };
        assert!(r.success);
        assert!((result - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_empty() {
        let v = SciVector {
            data: ptr::null_mut(),
            len: 0,
        };
        let mut result = 0.0;
        let r = unsafe { sci_mean(&v, &mut result) };
        assert!(!r.success);
        unsafe { sci_free_error(r.error_msg) };
    }

    #[test]
    fn test_std_sample() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let v = SciVector {
            data: data.as_ptr() as *mut f64,
            len: data.len(),
        };
        let mut result = 0.0;
        // ddof=1 for sample standard deviation
        let r = unsafe { sci_std(&v, 1, &mut result) };
        assert!(r.success);
        // Mean = 5.0; sum of squared deviations = 32.0; sample variance = 32/7.
        // Sample std = sqrt(32/7) ≈ 2.1381 (distinct from population std = 2.0).
        let expected = (32.0_f64 / 7.0_f64).sqrt();
        assert!(
            (result - expected).abs() < 1e-10,
            "result={result}, expected={expected}"
        );
    }

    #[test]
    fn test_std_population() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let v = SciVector {
            data: data.as_ptr() as *mut f64,
            len: data.len(),
        };
        let mut result = 0.0;
        // ddof=0 for population standard deviation
        let r = unsafe { sci_std(&v, 0, &mut result) };
        assert!(r.success);
        // Population std = sqrt(4) = 2.0 ... actually sqrt(32/8) = sqrt(4) = 2
        // Mean = 5.0, sum of squared deviations = 4+1+1+1+0+0+4+16 = 32 (wrong)
        // Actually: mean = 40/8 = 5.0
        // (2-5)^2 + (4-5)^2 + (4-5)^2 + (4-5)^2 + (5-5)^2 + (5-5)^2 + (7-5)^2 + (9-5)^2
        // = 9 + 1 + 1 + 1 + 0 + 0 + 4 + 16 = 32
        // pop variance = 32/8 = 4, pop std = 2.0
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentile() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let v = SciVector {
            data: data.as_ptr() as *mut f64,
            len: data.len(),
        };

        let mut result = 0.0;

        // 0th percentile = min
        let r = unsafe { sci_percentile(&v, 0.0, &mut result) };
        assert!(r.success);
        assert!((result - 1.0).abs() < 1e-10);

        // 50th percentile = median
        let r = unsafe { sci_percentile(&v, 50.0, &mut result) };
        assert!(r.success);
        assert!((result - 3.0).abs() < 1e-10);

        // 100th percentile = max
        let r = unsafe { sci_percentile(&v, 100.0, &mut result) };
        assert!(r.success);
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentile_interpolation() {
        // Test linear interpolation with 4 elements [10, 20, 30, 40]
        let data = [10.0, 20.0, 30.0, 40.0];
        let v = SciVector {
            data: data.as_ptr() as *mut f64,
            len: data.len(),
        };

        let mut result = 0.0;
        // 25th percentile: rank = 0.25 * 3 = 0.75
        // lower=0, upper=1, frac=0.75
        // result = 10 + 0.75 * (20 - 10) = 17.5
        let r = unsafe { sci_percentile(&v, 25.0, &mut result) };
        assert!(r.success);
        assert!((result - 17.5).abs() < 1e-10);
    }

    #[test]
    fn test_percentile_invalid_q() {
        let data = [1.0, 2.0, 3.0];
        let v = SciVector {
            data: data.as_ptr() as *mut f64,
            len: data.len(),
        };
        let mut result = 0.0;

        let r = unsafe { sci_percentile(&v, -1.0, &mut result) };
        assert!(!r.success);
        unsafe { sci_free_error(r.error_msg) };

        let r = unsafe { sci_percentile(&v, 101.0, &mut result) };
        assert!(!r.success);
        unsafe { sci_free_error(r.error_msg) };
    }

    #[test]
    fn test_correlation() {
        // Perfect positive correlation
        let x_data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y_data = [2.0, 4.0, 6.0, 8.0, 10.0];
        let x = SciVector {
            data: x_data.as_ptr() as *mut f64,
            len: x_data.len(),
        };
        let y = SciVector {
            data: y_data.as_ptr() as *mut f64,
            len: y_data.len(),
        };

        let mut result = 0.0;
        let r = unsafe { sci_correlation(&x, &y, &mut result) };
        assert!(r.success);
        assert!((result - 1.0).abs() < 1e-10);

        // Perfect negative correlation
        let y_neg = [10.0, 8.0, 6.0, 4.0, 2.0];
        let yn = SciVector {
            data: y_neg.as_ptr() as *mut f64,
            len: y_neg.len(),
        };
        let r = unsafe { sci_correlation(&x, &yn, &mut result) };
        assert!(r.success);
        assert!((result - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_length_mismatch() {
        let x_data = [1.0, 2.0, 3.0];
        let y_data = [1.0, 2.0];
        let x = SciVector {
            data: x_data.as_ptr() as *mut f64,
            len: x_data.len(),
        };
        let y = SciVector {
            data: y_data.as_ptr() as *mut f64,
            len: y_data.len(),
        };

        let mut result = 0.0;
        let r = unsafe { sci_correlation(&x, &y, &mut result) };
        assert!(!r.success);
        unsafe { sci_free_error(r.error_msg) };
    }

    #[test]
    fn test_variance() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let v = SciVector {
            data: data.as_ptr() as *mut f64,
            len: data.len(),
        };
        let mut result = 0.0;

        // Population variance (ddof=0)
        let r = unsafe { sci_variance(&v, 0, &mut result) };
        assert!(r.success);
        assert!((result - 4.0).abs() < 1e-10);

        // Sample variance (ddof=1)
        let r = unsafe { sci_variance(&v, 1, &mut result) };
        assert!(r.success);
        // 32/7 ~= 4.571428...
        assert!((result - 32.0 / 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_median() {
        // Odd number of elements
        let data = [3.0, 1.0, 2.0, 5.0, 4.0];
        let v = SciVector {
            data: data.as_ptr() as *mut f64,
            len: data.len(),
        };
        let mut result = 0.0;
        let r = unsafe { sci_median(&v, &mut result) };
        assert!(r.success);
        assert!((result - 3.0).abs() < 1e-10);

        // Even number of elements
        let data2 = [3.0, 1.0, 2.0, 4.0];
        let v2 = SciVector {
            data: data2.as_ptr() as *mut f64,
            len: data2.len(),
        };
        let r = unsafe { sci_median(&v2, &mut result) };
        assert!(r.success);
        assert!((result - 2.5).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // FFT tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fft_roundtrip() {
        // Forward FFT then inverse should return the original signal.
        let real = [1.0, 2.0, 3.0, 4.0];
        let imag = [0.0, 0.0, 0.0, 0.0];

        let mut fwd = SciComplexVector {
            real: ptr::null_mut(),
            imag: ptr::null_mut(),
            len: 0,
        };

        // Forward FFT.
        let r = unsafe { sci_fft_forward(real.as_ptr(), imag.as_ptr(), 4, &mut fwd) };
        assert!(r.success);
        assert_eq!(fwd.len, 4);

        // Inverse FFT.
        let mut inv = SciComplexVector {
            real: ptr::null_mut(),
            imag: ptr::null_mut(),
            len: 0,
        };
        let r = unsafe { sci_fft_inverse(fwd.real, fwd.imag, fwd.len, &mut inv) };
        assert!(r.success);
        assert_eq!(inv.len, 4);

        // Check the reconstructed signal matches the original.
        let inv_real = unsafe { std::slice::from_raw_parts(inv.real, inv.len) };
        let inv_imag = unsafe { std::slice::from_raw_parts(inv.imag, inv.len) };
        for i in 0..4 {
            assert!(
                (inv_real[i] - real[i]).abs() < 1e-10,
                "real[{}]: expected {}, got {}",
                i,
                real[i],
                inv_real[i]
            );
            assert!(
                inv_imag[i].abs() < 1e-10,
                "imag[{}]: expected 0, got {}",
                i,
                inv_imag[i]
            );
        }

        // Cleanup.
        unsafe {
            sci_complex_vector_free(&mut fwd);
            sci_complex_vector_free(&mut inv);
        }
    }

    #[test]
    fn test_fft_known_values() {
        // FFT of [1, 0, 0, 0] should give [1, 1, 1, 1].
        let real = [1.0, 0.0, 0.0, 0.0];
        let mut fwd = SciComplexVector {
            real: ptr::null_mut(),
            imag: ptr::null_mut(),
            len: 0,
        };

        let r = unsafe { sci_fft_forward(real.as_ptr(), ptr::null(), 4, &mut fwd) };
        assert!(r.success);

        let fwd_real = unsafe { std::slice::from_raw_parts(fwd.real, fwd.len) };
        let fwd_imag = unsafe { std::slice::from_raw_parts(fwd.imag, fwd.len) };
        for i in 0..4 {
            assert!(
                (fwd_real[i] - 1.0).abs() < 1e-10,
                "FFT real[{}] = {}, expected 1.0",
                i,
                fwd_real[i]
            );
            assert!(
                fwd_imag[i].abs() < 1e-10,
                "FFT imag[{}] = {}, expected 0.0",
                i,
                fwd_imag[i]
            );
        }

        unsafe { sci_complex_vector_free(&mut fwd) };
    }

    #[test]
    fn test_fft_non_power_of_2() {
        // Test with length 5 (non-power-of-2, uses Bluestein's algorithm).
        let real = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut fwd = SciComplexVector {
            real: ptr::null_mut(),
            imag: ptr::null_mut(),
            len: 0,
        };

        let r = unsafe { sci_fft_forward(real.as_ptr(), ptr::null(), 5, &mut fwd) };
        assert!(r.success);
        assert_eq!(fwd.len, 5);

        // Roundtrip test.
        let mut inv = SciComplexVector {
            real: ptr::null_mut(),
            imag: ptr::null_mut(),
            len: 0,
        };
        let r = unsafe { sci_fft_inverse(fwd.real, fwd.imag, fwd.len, &mut inv) };
        assert!(r.success);

        let inv_real = unsafe { std::slice::from_raw_parts(inv.real, inv.len) };
        for i in 0..5 {
            assert!(
                (inv_real[i] - real[i]).abs() < 1e-9,
                "roundtrip real[{}]: expected {}, got {}",
                i,
                real[i],
                inv_real[i]
            );
        }

        unsafe {
            sci_complex_vector_free(&mut fwd);
            sci_complex_vector_free(&mut inv);
        }
    }

    #[test]
    fn test_rfft() {
        let real = [1.0, 2.0, 3.0, 4.0];
        let mut out = SciComplexVector {
            real: ptr::null_mut(),
            imag: ptr::null_mut(),
            len: 0,
        };

        let r = unsafe { sci_rfft(real.as_ptr(), 4, &mut out) };
        assert!(r.success);
        assert_eq!(out.len, 4);

        // DC component should be the sum
        let out_real = unsafe { std::slice::from_raw_parts(out.real, out.len) };
        assert!((out_real[0] - 10.0).abs() < 1e-10);

        unsafe { sci_complex_vector_free(&mut out) };
    }

    // -----------------------------------------------------------------------
    // Optimization tests
    // -----------------------------------------------------------------------

    /// A simple test function: f(x) = (x - 2)^2
    unsafe extern "C" fn quadratic_fn(x: f64, _user_data: *mut std::ffi::c_void) -> f64 {
        (x - 2.0) * (x - 2.0)
    }

    #[test]
    fn test_minimize_scalar_golden_section() {
        let mut x_min = 0.0;
        let mut f_min = 0.0;

        // Safety: quadratic_fn is a valid function pointer, x_min/f_min are valid pointers.
        let r = unsafe {
            sci_minimize_scalar(
                quadratic_fn,
                ptr::null_mut(),
                -10.0,
                10.0,
                1e-8,
                0, // default max_iter
                &mut x_min,
                &mut f_min,
            )
        };
        assert!(r.success);
        assert!((x_min - 2.0).abs() < 1e-6, "x_min = {}", x_min);
        assert!(f_min.abs() < 1e-10, "f_min = {}", f_min);
    }

    #[test]
    fn test_minimize_brent() {
        let mut x_min = 0.0;
        let mut f_min = 0.0;

        let r = unsafe {
            sci_minimize_brent(
                quadratic_fn,
                ptr::null_mut(),
                -10.0,
                10.0,
                1e-8,
                0,
                &mut x_min,
                &mut f_min,
            )
        };
        assert!(r.success);
        assert!((x_min - 2.0).abs() < 1e-6, "x_min = {}", x_min);
        assert!(f_min.abs() < 1e-10, "f_min = {}", f_min);
    }

    /// Test function for root finding: f(x) = x^2 - 4 (roots at x = -2, x = 2)
    unsafe extern "C" fn root_fn(x: f64, _user_data: *mut std::ffi::c_void) -> f64 {
        x * x - 4.0
    }

    #[test]
    fn test_root_find_positive() {
        let mut root = 0.0;
        let r = unsafe { sci_root_find(root_fn, ptr::null_mut(), 0.0, 10.0, 1e-12, 0, &mut root) };
        assert!(r.success);
        assert!((root - 2.0).abs() < 1e-10, "root = {}", root);
    }

    #[test]
    fn test_root_find_negative() {
        let mut root = 0.0;
        let r = unsafe { sci_root_find(root_fn, ptr::null_mut(), -10.0, 0.0, 1e-12, 0, &mut root) };
        assert!(r.success);
        assert!((root - (-2.0)).abs() < 1e-10, "root = {}", root);
    }

    #[test]
    fn test_root_find_no_sign_change() {
        let mut root = 0.0;
        // f(1) = -3, f(0.5) = -3.75 -- same sign, should fail
        let r = unsafe { sci_root_find(root_fn, ptr::null_mut(), 0.0, 1.0, 1e-12, 0, &mut root) };
        assert!(!r.success);
        unsafe { sci_free_error(r.error_msg) };
    }

    /// Test with user_data
    unsafe extern "C" fn shifted_fn(x: f64, user_data: *mut std::ffi::c_void) -> f64 {
        let shift = if user_data.is_null() {
            0.0
        } else {
            unsafe { *(user_data as *const f64) }
        };
        (x - shift) * (x - shift)
    }

    #[test]
    fn test_minimize_with_user_data() {
        let shift: f64 = 5.0;
        let mut x_min = 0.0;
        let mut f_min = 0.0;

        let r = unsafe {
            sci_minimize_scalar(
                shifted_fn,
                &shift as *const f64 as *mut std::ffi::c_void,
                0.0,
                10.0,
                1e-8,
                0,
                &mut x_min,
                &mut f_min,
            )
        };
        assert!(r.success);
        assert!((x_min - 5.0).abs() < 1e-6, "x_min = {}", x_min);
    }

    // -----------------------------------------------------------------------
    // Version test
    // -----------------------------------------------------------------------

    #[test]
    fn test_version() {
        let v = sci_version();
        assert!(!v.is_null());
        let c_str = unsafe { std::ffi::CStr::from_ptr(v) };
        let version = c_str.to_str().expect("version is valid UTF-8");
        assert!(!version.is_empty());
    }

    // -----------------------------------------------------------------------
    // Linalg tests (only when linalg feature is enabled)
    // -----------------------------------------------------------------------

    #[cfg(feature = "linalg")]
    mod linalg_tests {
        use super::*;

        #[test]
        fn test_det_identity() {
            // 3x3 identity matrix
            let data = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
            let m = SciMatrix {
                data: data.as_ptr() as *mut f64,
                rows: 3,
                cols: 3,
            };

            let mut det = 0.0;
            let r = unsafe { sci_det(&m, &mut det) };
            assert!(r.success);
            assert!((det - 1.0).abs() < 1e-10, "det = {}", det);
        }

        #[test]
        fn test_det_2x2() {
            // [[1, 2], [3, 4]], det = 1*4 - 2*3 = -2
            let data = [1.0, 2.0, 3.0, 4.0];
            let m = SciMatrix {
                data: data.as_ptr() as *mut f64,
                rows: 2,
                cols: 2,
            };

            let mut det = 0.0;
            let r = unsafe { sci_det(&m, &mut det) };
            assert!(r.success);
            assert!((det - (-2.0)).abs() < 1e-10, "det = {}", det);
        }

        #[test]
        fn test_det_non_square() {
            let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            let m = SciMatrix {
                data: data.as_ptr() as *mut f64,
                rows: 2,
                cols: 3,
            };

            let mut det = 0.0;
            let r = unsafe { sci_det(&m, &mut det) };
            assert!(!r.success);
            unsafe { sci_free_error(r.error_msg) };
        }

        #[test]
        fn test_inv_2x2() {
            // [[4, 7], [2, 6]], inv = [[0.6, -0.7], [-0.2, 0.4]]
            let data = [4.0, 7.0, 2.0, 6.0];
            let m = SciMatrix {
                data: data.as_ptr() as *mut f64,
                rows: 2,
                cols: 2,
            };

            let mut inv_m = SciMatrix {
                data: ptr::null_mut(),
                rows: 0,
                cols: 0,
            };
            let r = unsafe { sci_inv(&m, &mut inv_m) };
            assert!(r.success);
            assert_eq!(inv_m.rows, 2);
            assert_eq!(inv_m.cols, 2);

            let inv_data = unsafe { std::slice::from_raw_parts(inv_m.data, 4) };
            assert!((inv_data[0] - 0.6).abs() < 1e-10);
            assert!((inv_data[1] - (-0.7)).abs() < 1e-10);
            assert!((inv_data[2] - (-0.2)).abs() < 1e-10);
            assert!((inv_data[3] - 0.4).abs() < 1e-10);

            unsafe { sci_matrix_free(&mut inv_m) };
        }

        #[test]
        fn test_solve_2x2() {
            // Solve: [[2, 1], [1, 3]] * x = [5, 7]
            // Solution: x = [8/5, 9/5] = [1.6, 1.8]
            let a_data = [2.0, 1.0, 1.0, 3.0];
            let b_data = [5.0, 7.0];
            let a = SciMatrix {
                data: a_data.as_ptr() as *mut f64,
                rows: 2,
                cols: 2,
            };
            let b = SciVector {
                data: b_data.as_ptr() as *mut f64,
                len: 2,
            };

            let mut x = SciVector {
                data: ptr::null_mut(),
                len: 0,
            };
            let r = unsafe { sci_solve(&a, &b, &mut x) };
            assert!(r.success);
            assert_eq!(x.len, 2);

            let x_data = unsafe { std::slice::from_raw_parts(x.data, x.len) };
            assert!((x_data[0] - 1.6).abs() < 1e-10, "x[0] = {}", x_data[0]);
            assert!((x_data[1] - 1.8).abs() < 1e-10, "x[1] = {}", x_data[1]);

            unsafe { sci_vector_free(&mut x) };
        }

        #[test]
        fn test_svd_identity() {
            // SVD of 2x2 identity
            let data = [1.0, 0.0, 0.0, 1.0];
            let m = SciMatrix {
                data: data.as_ptr() as *mut f64,
                rows: 2,
                cols: 2,
            };

            let mut svd_result = SciSvdResult {
                u: SciMatrix {
                    data: ptr::null_mut(),
                    rows: 0,
                    cols: 0,
                },
                s: SciVector {
                    data: ptr::null_mut(),
                    len: 0,
                },
                vt: SciMatrix {
                    data: ptr::null_mut(),
                    rows: 0,
                    cols: 0,
                },
            };
            let r = unsafe { sci_svd(&m, &mut svd_result) };
            assert!(r.success);

            // Singular values should both be 1.0
            assert_eq!(svd_result.s.len, 2);
            let s = unsafe { std::slice::from_raw_parts(svd_result.s.data, svd_result.s.len) };
            assert!((s[0] - 1.0).abs() < 1e-10);
            assert!((s[1] - 1.0).abs() < 1e-10);

            unsafe { sci_svd_result_free(&mut svd_result) };
        }

        #[test]
        fn test_eig_diagonal() {
            // Eigenvalues of [[3, 0], [0, 5]] should be 3 and 5
            let data = [3.0, 0.0, 0.0, 5.0];
            let m = SciMatrix {
                data: data.as_ptr() as *mut f64,
                rows: 2,
                cols: 2,
            };

            let mut eig_result = SciEigResult {
                eigenvalues: SciComplexVector {
                    real: ptr::null_mut(),
                    imag: ptr::null_mut(),
                    len: 0,
                },
                eigenvectors: SciMatrix {
                    data: ptr::null_mut(),
                    rows: 0,
                    cols: 0,
                },
            };
            let r = unsafe { sci_eig(&m, &mut eig_result) };
            assert!(r.success);
            assert_eq!(eig_result.eigenvalues.len, 2);

            let real = unsafe { std::slice::from_raw_parts(eig_result.eigenvalues.real, 2) };
            let imag = unsafe { std::slice::from_raw_parts(eig_result.eigenvalues.imag, 2) };

            // Eigenvalues should be 3 and 5 (order may vary)
            let mut evals: Vec<f64> = real.to_vec();
            evals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            assert!((evals[0] - 3.0).abs() < 1e-10, "eval[0] = {}", evals[0]);
            assert!((evals[1] - 5.0).abs() < 1e-10, "eval[1] = {}", evals[1]);

            // Imaginary parts should be zero for a real symmetric matrix
            for &im in imag {
                assert!(im.abs() < 1e-10);
            }

            unsafe { sci_eig_result_free(&mut eig_result) };
        }
    }
}
