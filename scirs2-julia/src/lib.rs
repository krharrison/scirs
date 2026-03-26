//! # scirs2-julia
//!
//! Julia JLL-compatible shared library (`libscirs2_julia`) for the SciRS2
//! scientific computing library.
//!
//! This crate produces a `cdylib` (and `staticlib`) that exposes a stable
//! C ABI designed for consumption by the Julia `SciRS2_jll` package.  The
//! ABI is a thin, versioned wrapper around the `ffi` feature of `scirs2-core`
//! and adds:
//!
//! * A **version introspection** symbol (`scirs2_julia_version`,
//!   `scirs2_julia_abi_version`) so Julia can verify compatibility at runtime.
//! * A **capability flags** symbol (`scirs2_julia_capabilities`) that encodes
//!   which sub-modules are compiled in (linalg, stats, fft, optimize).
//! * All existing `sci_*` symbols re-exported under the same names so the
//!   existing `julia/SciRS2/src/SciRS2.jl` wrapper works without modification.
//! * Convenience batch functions that reduce the number of `ccall` round-trips
//!   for hot paths (e.g. `scirs2_batch_mean`, `scirs2_batch_variance`).
//!
//! # Safety
//!
//! All `extern "C"` functions uphold the same invariants as `scirs2-core/ffi`:
//!
//! 1. No panics cross the FFI boundary (`std::panic::catch_unwind`).
//! 2. All pointers are validated before dereferencing.
//! 3. Memory allocated by this library must be freed by this library.
//!
//! # ABI stability
//!
//! The `ABI_VERSION` constant is incremented whenever breaking changes are made
//! to the exported symbols.  Julia's `SciRS2_jll` package checks this at load
//! time and refuses to proceed if it detects a mismatch.

// Re-export all existing sci_* symbols from scirs2-core/ffi so that the
// pre-existing Julia wrapper (julia/SciRS2/src/SciRS2.jl) continues to work
// unchanged when loading `libscirs2_julia` instead of `libscirs2_core`.
//
// scirs2-core is always compiled with the `ffi` feature in this crate (see
// Cargo.toml dependency declaration), so the re-export is unconditional.
pub use scirs2_core::ffi::*;

use std::ffi::CString;
use std::os::raw::{c_char, c_uint};
use std::panic::{catch_unwind, AssertUnwindSafe};

// ---------------------------------------------------------------------------
// ABI version constants
// ---------------------------------------------------------------------------

/// Semantic version string of this JLL-targeted library.
const LIBRARY_VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");

/// ABI version integer.  Increment for every breaking ABI change.
const ABI_VERSION: u32 = 1;

/// Capability bit flags (OR-combination).
const CAP_LINALG: u32   = 0b0000_0001;
const CAP_STATS: u32    = 0b0000_0010;
const CAP_FFT: u32      = 0b0000_0100;
const CAP_OPTIMIZE: u32 = 0b0000_1000;

// ---------------------------------------------------------------------------
// Version introspection
// ---------------------------------------------------------------------------

/// Return the SciRS2 Julia library version string as a static C string.
///
/// The returned pointer is valid for the lifetime of the process and must
/// **not** be freed by the caller.
///
/// # Safety
///
/// Always safe to call.  Returns a non-null pointer.
#[no_mangle]
pub extern "C" fn scirs2_julia_version() -> *const c_char {
    LIBRARY_VERSION.as_ptr() as *const c_char
}

/// Return the ABI version integer for compatibility checking.
///
/// Julia's `SciRS2_jll` package verifies that the loaded library's ABI
/// version matches what was expected at package installation time.
#[no_mangle]
pub extern "C" fn scirs2_julia_abi_version() -> c_uint {
    ABI_VERSION
}

/// Return a bitmask of compiled-in capability flags.
///
/// Bit layout (LSB first):
/// - bit 0: linalg
/// - bit 1: stats
/// - bit 2: fft
/// - bit 3: optimize
///
/// Julia code can test: `(scirs2_julia_capabilities() & 0x01) != 0` to check
/// for linalg support.
#[no_mangle]
pub extern "C" fn scirs2_julia_capabilities() -> c_uint {
    let mut caps: u32 = 0;

    #[cfg(feature = "linalg")]
    { caps |= CAP_LINALG; }

    #[cfg(feature = "stats")]
    { caps |= CAP_STATS; }

    #[cfg(feature = "fft")]
    { caps |= CAP_FFT; }

    #[cfg(feature = "optimize")]
    { caps |= CAP_OPTIMIZE; }

    caps
}

// ---------------------------------------------------------------------------
// Batch statistics helpers — reduce ccall overhead for hot paths
// ---------------------------------------------------------------------------

/// Result type for batch operations.  Mirrors `SciResult` in scirs2-core/ffi.
#[repr(C)]
pub struct JllResult {
    /// `true` if the operation succeeded.
    pub success: bool,
    /// NUL-terminated error message allocated by SciRS2, or null on success.
    /// Free with [`scirs2_julia_free_error`].
    pub error_msg: *const c_char,
}

impl JllResult {
    fn ok() -> Self {
        JllResult { success: true, error_msg: std::ptr::null() }
    }

    fn err(msg: &str) -> Self {
        let c_msg = CString::new(msg).unwrap_or_else(|_| {
            CString::new("(error message encoding failed)").unwrap_or_else(|_| {
                // Absolute fallback — no interior NULs.
                unsafe { CString::from_vec_unchecked(b"unknown error".to_vec()) }
            })
        });
        JllResult { success: false, error_msg: c_msg.into_raw() }
    }

    fn from_panic(payload: Box<dyn std::any::Any + Send>) -> Self {
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

/// Free an error message allocated by any `scirs2_julia_*` function.
///
/// # Safety
///
/// `msg` must be a pointer returned in the `error_msg` field of a
/// [`JllResult`], or null (a no-op).  Passing any other pointer is UB.
#[no_mangle]
pub unsafe extern "C" fn scirs2_julia_free_error(msg: *const c_char) {
    if !msg.is_null() {
        let _ = unsafe { CString::from_raw(msg as *mut c_char) };
    }
}

/// Compute the arithmetic mean of `len` f64 values at `data`.
///
/// This is a convenience duplicate of `sci_mean` that uses the `JllResult`
/// type so Julia can use the same result struct for all batch operations.
///
/// # Safety
///
/// `data` must be a valid pointer to at least `len` `f64` values.
/// `out` must be a valid non-null pointer to a writable `f64`.
#[no_mangle]
pub unsafe extern "C" fn scirs2_batch_mean(
    data: *const f64,
    len:  usize,
    out:  *mut f64,
) -> JllResult {
    match catch_unwind(AssertUnwindSafe(|| {
        if data.is_null() {
            return Err("scirs2_batch_mean: data pointer is null".to_string());
        }
        if out.is_null() {
            return Err("scirs2_batch_mean: out pointer is null".to_string());
        }
        if len == 0 {
            return Err("scirs2_batch_mean: empty input".to_string());
        }
        let slice = unsafe { std::slice::from_raw_parts(data, len) };
        let mean = slice.iter().copied().sum::<f64>() / len as f64;
        unsafe { *out = mean; }
        Ok(())
    })) {
        Ok(Ok(())) => JllResult::ok(),
        Ok(Err(e)) => JllResult::err(&e),
        Err(payload) => JllResult::from_panic(payload),
    }
}

/// Compute the variance of `len` f64 values at `data`.
///
/// Uses `ddof` degrees of freedom: `ddof = 1` gives sample variance,
/// `ddof = 0` gives population variance.
///
/// # Safety
///
/// `data` must be a valid pointer to at least `len` `f64` values.
/// `out` must be a valid non-null pointer to a writable `f64`.
#[no_mangle]
pub unsafe extern "C" fn scirs2_batch_variance(
    data: *const f64,
    len:  usize,
    ddof: usize,
    out:  *mut f64,
) -> JllResult {
    match catch_unwind(AssertUnwindSafe(|| {
        if data.is_null() {
            return Err("scirs2_batch_variance: data pointer is null".to_string());
        }
        if out.is_null() {
            return Err("scirs2_batch_variance: out pointer is null".to_string());
        }
        if len <= ddof {
            return Err(format!(
                "scirs2_batch_variance: len ({}) must be > ddof ({})",
                len, ddof
            ));
        }
        let slice = unsafe { std::slice::from_raw_parts(data, len) };
        let mean = slice.iter().copied().sum::<f64>() / len as f64;
        let var = slice.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>()
            / (len - ddof) as f64;
        unsafe { *out = var; }
        Ok(())
    })) {
        Ok(Ok(())) => JllResult::ok(),
        Ok(Err(e)) => JllResult::err(&e),
        Err(payload) => JllResult::from_panic(payload),
    }
}

/// Compute the standard deviation of `len` f64 values.
///
/// `ddof` controls degrees of freedom (1 = sample, 0 = population).
///
/// # Safety
///
/// Same as [`scirs2_batch_variance`].
#[no_mangle]
pub unsafe extern "C" fn scirs2_batch_std(
    data: *const f64,
    len:  usize,
    ddof: usize,
    out:  *mut f64,
) -> JllResult {
    let mut var = 0.0_f64;
    let r = unsafe { scirs2_batch_variance(data, len, ddof, &mut var as *mut f64) };
    if r.success {
        if out.is_null() {
            return JllResult::err("scirs2_batch_std: out pointer is null");
        }
        unsafe { *out = var.sqrt(); }
    }
    r
}

/// Compute the minimum and maximum of `len` f64 values simultaneously.
///
/// On success, `out_min` and `out_max` are written with the respective values.
///
/// # Safety
///
/// `data` must be a valid pointer to at least `len` `f64` values.
/// `out_min` and `out_max` must be valid non-null pointers to writable `f64`.
#[no_mangle]
pub unsafe extern "C" fn scirs2_batch_minmax(
    data:    *const f64,
    len:     usize,
    out_min: *mut f64,
    out_max: *mut f64,
) -> JllResult {
    match catch_unwind(AssertUnwindSafe(|| {
        if data.is_null() {
            return Err("scirs2_batch_minmax: data pointer is null".to_string());
        }
        if out_min.is_null() || out_max.is_null() {
            return Err("scirs2_batch_minmax: output pointer is null".to_string());
        }
        if len == 0 {
            return Err("scirs2_batch_minmax: empty input".to_string());
        }
        let slice = unsafe { std::slice::from_raw_parts(data, len) };
        let mut mn = slice[0];
        let mut mx = slice[0];
        for &x in &slice[1..] {
            if x < mn { mn = x; }
            if x > mx { mx = x; }
        }
        unsafe {
            *out_min = mn;
            *out_max = mx;
        }
        Ok(())
    })) {
        Ok(Ok(())) => JllResult::ok(),
        Ok(Err(e)) => JllResult::err(&e),
        Err(payload) => JllResult::from_panic(payload),
    }
}

/// Compute the sum of `len` f64 values.
///
/// # Safety
///
/// `data` must be a valid pointer to at least `len` `f64` values.
/// `out` must be a valid non-null pointer to a writable `f64`.
#[no_mangle]
pub unsafe extern "C" fn scirs2_batch_sum(
    data: *const f64,
    len:  usize,
    out:  *mut f64,
) -> JllResult {
    match catch_unwind(AssertUnwindSafe(|| {
        if data.is_null() {
            return Err("scirs2_batch_sum: data pointer is null".to_string());
        }
        if out.is_null() {
            return Err("scirs2_batch_sum: out pointer is null".to_string());
        }
        let slice = unsafe { std::slice::from_raw_parts(data, len) };
        unsafe { *out = slice.iter().copied().sum::<f64>(); }
        Ok(())
    })) {
        Ok(Ok(())) => JllResult::ok(),
        Ok(Err(e)) => JllResult::err(&e),
        Err(payload) => JllResult::from_panic(payload),
    }
}

/// Compute the dot product of two f64 arrays of length `len`.
///
/// # Safety
///
/// `a` and `b` must each be valid pointers to at least `len` `f64` values.
/// `out` must be a valid non-null pointer to a writable `f64`.
#[no_mangle]
pub unsafe extern "C" fn scirs2_batch_dot(
    a:   *const f64,
    b:   *const f64,
    len: usize,
    out: *mut f64,
) -> JllResult {
    match catch_unwind(AssertUnwindSafe(|| {
        if a.is_null() || b.is_null() {
            return Err("scirs2_batch_dot: input pointer is null".to_string());
        }
        if out.is_null() {
            return Err("scirs2_batch_dot: out pointer is null".to_string());
        }
        let sa = unsafe { std::slice::from_raw_parts(a, len) };
        let sb = unsafe { std::slice::from_raw_parts(b, len) };
        let dot = sa.iter().zip(sb.iter()).map(|(x, y)| x * y).sum::<f64>();
        unsafe { *out = dot; }
        Ok(())
    })) {
        Ok(Ok(())) => JllResult::ok(),
        Ok(Err(e)) => JllResult::err(&e),
        Err(payload) => JllResult::from_panic(payload),
    }
}

/// Apply an element-wise scale (`alpha * x[i]`) in-place on `len` values.
///
/// # Safety
///
/// `data` must be a valid, non-null, non-aliased pointer to `len` writable
/// `f64` values.
#[no_mangle]
pub unsafe extern "C" fn scirs2_batch_scale(
    data:  *mut f64,
    len:   usize,
    alpha: f64,
) -> JllResult {
    match catch_unwind(AssertUnwindSafe(|| {
        if data.is_null() {
            return Err("scirs2_batch_scale: data pointer is null".to_string());
        }
        let slice = unsafe { std::slice::from_raw_parts_mut(data, len) };
        for x in slice.iter_mut() {
            *x *= alpha;
        }
        Ok(())
    })) {
        Ok(Ok(())) => JllResult::ok(),
        Ok(Err(e)) => JllResult::err(&e),
        Err(payload) => JllResult::from_panic(payload),
    }
}

/// Compute the L2 (Euclidean) norm of `len` f64 values.
///
/// # Safety
///
/// `data` must be a valid pointer to at least `len` `f64` values.
/// `out` must be a valid non-null pointer to a writable `f64`.
#[no_mangle]
pub unsafe extern "C" fn scirs2_batch_norm_l2(
    data: *const f64,
    len:  usize,
    out:  *mut f64,
) -> JllResult {
    match catch_unwind(AssertUnwindSafe(|| {
        if data.is_null() {
            return Err("scirs2_batch_norm_l2: data pointer is null".to_string());
        }
        if out.is_null() {
            return Err("scirs2_batch_norm_l2: out pointer is null".to_string());
        }
        let slice = unsafe { std::slice::from_raw_parts(data, len) };
        let sq_sum: f64 = slice.iter().map(|&x| x * x).sum();
        unsafe { *out = sq_sum.sqrt(); }
        Ok(())
    })) {
        Ok(Ok(())) => JllResult::ok(),
        Ok(Err(e)) => JllResult::err(&e),
        Err(payload) => JllResult::from_panic(payload),
    }
}

/// Compute the L1 (Manhattan) norm of `len` f64 values.
///
/// # Safety
///
/// `data` must be a valid pointer to at least `len` `f64` values.
/// `out` must be a valid non-null pointer to a writable `f64`.
#[no_mangle]
pub unsafe extern "C" fn scirs2_batch_norm_l1(
    data: *const f64,
    len:  usize,
    out:  *mut f64,
) -> JllResult {
    match catch_unwind(AssertUnwindSafe(|| {
        if data.is_null() {
            return Err("scirs2_batch_norm_l1: data pointer is null".to_string());
        }
        if out.is_null() {
            return Err("scirs2_batch_norm_l1: out pointer is null".to_string());
        }
        let slice = unsafe { std::slice::from_raw_parts(data, len) };
        let norm: f64 = slice.iter().map(|&x| x.abs()).sum();
        unsafe { *out = norm; }
        Ok(())
    })) {
        Ok(Ok(())) => JllResult::ok(),
        Ok(Err(e)) => JllResult::err(&e),
        Err(payload) => JllResult::from_panic(payload),
    }
}

/// Sort `len` f64 values in-place in ascending order.
///
/// Uses a NaN-safe total ordering (NaNs are placed at the end).
///
/// # Safety
///
/// `data` must be a valid, non-null, non-aliased pointer to `len` writable
/// `f64` values.
#[no_mangle]
pub unsafe extern "C" fn scirs2_batch_sort_ascending(
    data: *mut f64,
    len:  usize,
) -> JllResult {
    match catch_unwind(AssertUnwindSafe(|| {
        if data.is_null() {
            return Err("scirs2_batch_sort_ascending: data pointer is null".to_string());
        }
        let slice = unsafe { std::slice::from_raw_parts_mut(data, len) };
        slice.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
        Ok(())
    })) {
        Ok(Ok(())) => JllResult::ok(),
        Ok(Err(e)) => JllResult::err(&e),
        Err(payload) => JllResult::from_panic(payload),
    }
}

/// Compute the cumulative sum of `len` f64 values and write results to `out`.
///
/// `out` must have at least `len` elements.  Input and output buffers must
/// not overlap.
///
/// # Safety
///
/// `data` must be a valid pointer to at least `len` `f64` values.
/// `out` must be a valid non-null pointer to at least `len` writable `f64`.
#[no_mangle]
pub unsafe extern "C" fn scirs2_batch_cumsum(
    data: *const f64,
    len:  usize,
    out:  *mut f64,
) -> JllResult {
    match catch_unwind(AssertUnwindSafe(|| {
        if data.is_null() {
            return Err("scirs2_batch_cumsum: data pointer is null".to_string());
        }
        if out.is_null() {
            return Err("scirs2_batch_cumsum: out pointer is null".to_string());
        }
        let src = unsafe { std::slice::from_raw_parts(data, len) };
        let dst = unsafe { std::slice::from_raw_parts_mut(out, len) };
        let mut acc = 0.0_f64;
        for (i, &x) in src.iter().enumerate() {
            acc += x;
            dst[i] = acc;
        }
        Ok(())
    })) {
        Ok(Ok(())) => JllResult::ok(),
        Ok(Err(e)) => JllResult::err(&e),
        Err(payload) => JllResult::from_panic(payload),
    }
}

/// Compute the Pearson correlation coefficient between two f64 arrays.
///
/// Returns `NaN` if either input is constant (zero variance).
///
/// # Safety
///
/// `x` and `y` must each be valid pointers to at least `len` `f64` values.
/// `out` must be a valid non-null pointer to a writable `f64`.
#[no_mangle]
pub unsafe extern "C" fn scirs2_batch_correlation(
    x:   *const f64,
    y:   *const f64,
    len: usize,
    out: *mut f64,
) -> JllResult {
    match catch_unwind(AssertUnwindSafe(|| {
        if x.is_null() || y.is_null() {
            return Err("scirs2_batch_correlation: input pointer is null".to_string());
        }
        if out.is_null() {
            return Err("scirs2_batch_correlation: out pointer is null".to_string());
        }
        if len < 2 {
            return Err("scirs2_batch_correlation: need at least 2 elements".to_string());
        }
        let sx = unsafe { std::slice::from_raw_parts(x, len) };
        let sy = unsafe { std::slice::from_raw_parts(y, len) };
        let n = len as f64;
        let mx = sx.iter().copied().sum::<f64>() / n;
        let my = sy.iter().copied().sum::<f64>() / n;
        let cov = sx.iter().zip(sy.iter())
            .map(|(&xi, &yi)| (xi - mx) * (yi - my))
            .sum::<f64>();
        let vx = sx.iter().map(|&xi| (xi - mx) * (xi - mx)).sum::<f64>();
        let vy = sy.iter().map(|&yi| (yi - my) * (yi - my)).sum::<f64>();
        let denom = (vx * vy).sqrt();
        let corr = if denom < f64::EPSILON { f64::NAN } else { cov / denom };
        unsafe { *out = corr; }
        Ok(())
    })) {
        Ok(Ok(())) => JllResult::ok(),
        Ok(Err(e)) => JllResult::err(&e),
        Err(payload) => JllResult::from_panic(payload),
    }
}

// ---------------------------------------------------------------------------
// JLL-specific metadata symbols
// ---------------------------------------------------------------------------

/// Write the JLL soname into the caller-supplied buffer.
///
/// `buf` must be at least `buf_len` bytes.  The output is NUL-terminated and
/// truncated if necessary.  Returns the number of bytes written (excluding the
/// NUL terminator).
///
/// # Safety
///
/// `buf` must be a valid pointer to at least `buf_len` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn scirs2_julia_soname(buf: *mut c_char, buf_len: usize) -> usize {
    const SONAME: &[u8] = b"libscirs2_julia.so.0\0";
    if buf.is_null() || buf_len == 0 {
        return 0;
    }
    let write_len = SONAME.len().min(buf_len);
    let dst = unsafe { std::slice::from_raw_parts_mut(buf as *mut u8, write_len) };
    dst.copy_from_slice(&SONAME[..write_len]);
    // Ensure NUL termination.
    dst[write_len - 1] = 0;
    write_len - 1
}

// ---------------------------------------------------------------------------
// Tests (compiled only when running `cargo test`)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_not_null() {
        let ptr = scirs2_julia_version();
        assert!(!ptr.is_null());
        let s = unsafe { std::ffi::CStr::from_ptr(ptr) };
        let sv = s.to_str().expect("version is valid UTF-8");
        assert!(sv.contains('.'), "version should look like X.Y.Z, got: {sv}");
    }

    #[test]
    fn test_abi_version() {
        assert_eq!(scirs2_julia_abi_version(), ABI_VERSION);
    }

    #[test]
    fn test_capabilities_nonzero_in_default_build() {
        // With default features all four bits should be set.
        let caps = scirs2_julia_capabilities();
        assert_ne!(caps, 0, "at least one capability should be compiled in");
    }

    #[test]
    fn test_batch_mean() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let mut out = 0.0_f64;
        let r = unsafe {
            scirs2_batch_mean(data.as_ptr(), data.len(), &mut out as *mut f64)
        };
        assert!(r.success, "batch_mean should succeed");
        assert!((out - 3.0).abs() < 1e-10, "mean of [1..5] should be 3.0, got {out}");
    }

    #[test]
    fn test_batch_mean_empty_fails() {
        let mut out = 0.0_f64;
        let r = unsafe { scirs2_batch_mean([].as_ptr(), 0, &mut out as *mut f64) };
        assert!(!r.success, "empty mean should fail");
        if !r.error_msg.is_null() {
            unsafe { scirs2_julia_free_error(r.error_msg) };
        }
    }

    #[test]
    fn test_batch_variance() {
        // [2,4,4,4,5,5,7,9]: mean=5, sum_sq=32
        // population variance (ddof=0) = 32/8 = 4.0
        // sample variance (ddof=1) = 32/7 ≈ 4.5714...
        let data = vec![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let mut out = 0.0_f64;
        let r = unsafe {
            scirs2_batch_variance(data.as_ptr(), data.len(), 0, &mut out as *mut f64)
        };
        assert!(r.success, "batch_variance should succeed");
        assert!((out - 4.0).abs() < 1e-10, "population variance should be 4.0, got {out}");
        // Also verify sample variance (ddof=1)
        let mut out2 = 0.0_f64;
        let r2 = unsafe {
            scirs2_batch_variance(data.as_ptr(), data.len(), 1, &mut out2 as *mut f64)
        };
        assert!(r2.success);
        let expected_sample = 32.0_f64 / 7.0;
        assert!(
            (out2 - expected_sample).abs() < 1e-10,
            "sample variance should be {expected_sample}, got {out2}"
        );
    }

    #[test]
    fn test_batch_std() {
        // population std (ddof=0) = sqrt(4.0) = 2.0
        let data = vec![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let mut out = 0.0_f64;
        let r = unsafe {
            scirs2_batch_std(data.as_ptr(), data.len(), 0, &mut out as *mut f64)
        };
        assert!(r.success);
        assert!((out - 2.0).abs() < 1e-10, "population std should be 2.0, got {out}");
    }

    #[test]
    fn test_batch_minmax() {
        let data = vec![3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let mut mn = 0.0_f64;
        let mut mx = 0.0_f64;
        let r = unsafe {
            scirs2_batch_minmax(
                data.as_ptr(), data.len(),
                &mut mn as *mut f64, &mut mx as *mut f64,
            )
        };
        assert!(r.success);
        assert!((mn - 1.0).abs() < 1e-10, "min should be 1.0, got {mn}");
        assert!((mx - 9.0).abs() < 1e-10, "max should be 9.0, got {mx}");
    }

    #[test]
    fn test_batch_sum() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let mut out = 0.0_f64;
        let r = unsafe {
            scirs2_batch_sum(data.as_ptr(), data.len(), &mut out as *mut f64)
        };
        assert!(r.success);
        assert!((out - 10.0).abs() < 1e-10, "sum should be 10.0, got {out}");
    }

    #[test]
    fn test_batch_dot() {
        let a = vec![1.0_f64, 2.0, 3.0];
        let b = vec![4.0_f64, 5.0, 6.0];
        let mut out = 0.0_f64;
        let r = unsafe {
            scirs2_batch_dot(a.as_ptr(), b.as_ptr(), a.len(), &mut out as *mut f64)
        };
        assert!(r.success);
        // 1*4 + 2*5 + 3*6 = 32
        assert!((out - 32.0).abs() < 1e-10, "dot product should be 32.0, got {out}");
    }

    #[test]
    fn test_batch_norm_l2() {
        let data = vec![3.0_f64, 4.0];
        let mut out = 0.0_f64;
        let r = unsafe {
            scirs2_batch_norm_l2(data.as_ptr(), data.len(), &mut out as *mut f64)
        };
        assert!(r.success);
        assert!((out - 5.0).abs() < 1e-10, "L2 norm of [3,4] should be 5.0, got {out}");
    }

    #[test]
    fn test_batch_norm_l1() {
        let data = vec![-3.0_f64, 4.0];
        let mut out = 0.0_f64;
        let r = unsafe {
            scirs2_batch_norm_l1(data.as_ptr(), data.len(), &mut out as *mut f64)
        };
        assert!(r.success);
        assert!((out - 7.0).abs() < 1e-10, "L1 norm of [-3,4] should be 7.0, got {out}");
    }

    #[test]
    fn test_batch_sort_ascending() {
        let mut data = vec![5.0_f64, 3.0, 1.0, 4.0, 2.0];
        let r = unsafe {
            scirs2_batch_sort_ascending(data.as_mut_ptr(), data.len())
        };
        assert!(r.success);
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_batch_cumsum() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let mut out = vec![0.0_f64; 4];
        let r = unsafe {
            scirs2_batch_cumsum(data.as_ptr(), data.len(), out.as_mut_ptr())
        };
        assert!(r.success);
        assert_eq!(out, vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_batch_correlation_perfect_positive() {
        let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0_f64, 4.0, 6.0, 8.0, 10.0];
        let mut out = 0.0_f64;
        let r = unsafe {
            scirs2_batch_correlation(x.as_ptr(), y.as_ptr(), x.len(), &mut out as *mut f64)
        };
        assert!(r.success);
        assert!((out - 1.0).abs() < 1e-10, "perfect positive corr should be 1.0, got {out}");
    }

    #[test]
    fn test_batch_scale() {
        let mut data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let r = unsafe {
            scirs2_batch_scale(data.as_mut_ptr(), data.len(), 2.0)
        };
        assert!(r.success);
        assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_soname_output() {
        let mut buf = vec![0u8; 64];
        let n = unsafe {
            scirs2_julia_soname(buf.as_mut_ptr() as *mut c_char, buf.len())
        };
        assert!(n > 0);
        let s = std::ffi::CStr::from_bytes_until_nul(&buf)
            .expect("soname is NUL-terminated")
            .to_str()
            .expect("soname is valid UTF-8");
        assert!(s.contains("scirs2_julia"), "soname should contain 'scirs2_julia', got: {s}");
    }
}
