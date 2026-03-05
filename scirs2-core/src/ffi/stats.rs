//! Statistical FFI functions.
//!
//! These functions expose basic descriptive statistics through the C ABI:
//! mean, standard deviation, percentile, and correlation.
//!
//! All functions follow the SciRS2 FFI conventions:
//! - Return [`SciResult`] to indicate success/failure.
//! - Never panic across the FFI boundary.
//! - Validate all pointers before dereferencing.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

use super::types::{SciResult, SciVector};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Validate a `SciVector` pointer and extract a borrowed slice.
///
/// # Safety
///
/// The `SciVector`'s data pointer must be valid for `len` elements.
unsafe fn validate_vector(v: *const SciVector, name: &str) -> Result<&'static [f64], String> {
    if v.is_null() {
        return Err(format!("{}: null vector pointer", name));
    }
    let vec = unsafe { &*v };
    if vec.data.is_null() && vec.len > 0 {
        return Err(format!("{}: data pointer is null but len > 0", name));
    }
    if vec.len == 0 {
        // Return empty slice for zero-length vectors.
        return Ok(&[]);
    }
    Ok(unsafe { std::slice::from_raw_parts(vec.data, vec.len) })
}

// ---------------------------------------------------------------------------
// sci_mean
// ---------------------------------------------------------------------------

/// Compute the arithmetic mean of a vector.
///
/// # Parameters
///
/// - `vec`: pointer to a `SciVector`. Must have at least 1 element.
/// - `out`: pointer to an `f64` where the mean will be written.
///
/// # Safety
///
/// - `vec` must point to a valid `SciVector` with valid data.
/// - `out` must be a valid, non-null pointer to `f64`.
#[no_mangle]
pub unsafe extern "C" fn sci_mean(vec: *const SciVector, out: *mut f64) -> SciResult {
    if out.is_null() {
        return SciResult::err("sci_mean: out pointer is null");
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let data = unsafe { validate_vector(vec, "sci_mean") }?;
        if data.is_empty() {
            return Err("sci_mean: empty vector".to_string());
        }
        let sum: f64 = data.iter().sum();
        Ok(sum / data.len() as f64)
    }));

    match result {
        Ok(Ok(mean)) => {
            unsafe { ptr::write(out, mean) };
            SciResult::ok()
        }
        Ok(Err(msg)) => SciResult::err(&msg),
        Err(e) => SciResult::from_panic(e),
    }
}

// ---------------------------------------------------------------------------
// sci_std
// ---------------------------------------------------------------------------

/// Compute the sample standard deviation of a vector (Bessel-corrected, ddof=1).
///
/// For a population standard deviation (ddof=0), use `sci_std_population`.
///
/// # Parameters
///
/// - `vec`: pointer to a `SciVector`. Must have at least 2 elements.
/// - `ddof`: delta degrees of freedom (0 for population, 1 for sample).
/// - `out`: pointer to an `f64` where the standard deviation will be written.
///
/// # Safety
///
/// - `vec` must point to a valid `SciVector` with valid data.
/// - `out` must be a valid, non-null pointer to `f64`.
#[no_mangle]
pub unsafe extern "C" fn sci_std(vec: *const SciVector, ddof: usize, out: *mut f64) -> SciResult {
    if out.is_null() {
        return SciResult::err("sci_std: out pointer is null");
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let data = unsafe { validate_vector(vec, "sci_std") }?;
        let n = data.len();
        if n == 0 {
            return Err("sci_std: empty vector".to_string());
        }
        if n <= ddof {
            return Err(format!(
                "sci_std: need at least {} elements for ddof={}, got {}",
                ddof + 1,
                ddof,
                n
            ));
        }

        let mean: f64 = data.iter().sum::<f64>() / n as f64;
        let variance: f64 =
            data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (n - ddof) as f64;
        Ok(variance.sqrt())
    }));

    match result {
        Ok(Ok(std_dev)) => {
            unsafe { ptr::write(out, std_dev) };
            SciResult::ok()
        }
        Ok(Err(msg)) => SciResult::err(&msg),
        Err(e) => SciResult::from_panic(e),
    }
}

// ---------------------------------------------------------------------------
// sci_percentile
// ---------------------------------------------------------------------------

/// Compute the q-th percentile of a vector using linear interpolation.
///
/// Uses the same method as NumPy's `numpy.percentile` with the default
/// `interpolation='linear'` option.
///
/// # Parameters
///
/// - `vec`: pointer to a `SciVector`. Must have at least 1 element.
/// - `q`: percentile to compute, in the range [0, 100].
/// - `out`: pointer to an `f64` where the result will be written.
///
/// # Safety
///
/// - `vec` must point to a valid `SciVector` with valid data.
/// - `out` must be a valid, non-null pointer to `f64`.
#[no_mangle]
pub unsafe extern "C" fn sci_percentile(vec: *const SciVector, q: f64, out: *mut f64) -> SciResult {
    if out.is_null() {
        return SciResult::err("sci_percentile: out pointer is null");
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let data = unsafe { validate_vector(vec, "sci_percentile") }?;
        if data.is_empty() {
            return Err("sci_percentile: empty vector".to_string());
        }
        if !(0.0..=100.0).contains(&q) {
            return Err(format!("sci_percentile: q must be in [0, 100], got {}", q));
        }
        if q.is_nan() {
            return Err("sci_percentile: q is NaN".to_string());
        }

        // Sort a copy of the data
        let mut sorted: Vec<f64> = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        if n == 1 {
            return Ok(sorted[0]);
        }

        // Linear interpolation (same as numpy default)
        let rank = (q / 100.0) * (n - 1) as f64;
        let lower = rank.floor() as usize;
        let upper = rank.ceil() as usize;
        let frac = rank - lower as f64;

        if lower == upper || upper >= n {
            Ok(sorted[lower.min(n - 1)])
        } else {
            Ok(sorted[lower] + frac * (sorted[upper] - sorted[lower]))
        }
    }));

    match result {
        Ok(Ok(pct)) => {
            unsafe { ptr::write(out, pct) };
            SciResult::ok()
        }
        Ok(Err(msg)) => SciResult::err(&msg),
        Err(e) => SciResult::from_panic(e),
    }
}

// ---------------------------------------------------------------------------
// sci_correlation
// ---------------------------------------------------------------------------

/// Compute the Pearson correlation coefficient between two vectors.
///
/// Returns a value in [-1, 1]. Both vectors must have the same length
/// and at least 2 elements.
///
/// # Parameters
///
/// - `x`: pointer to the first `SciVector`.
/// - `y`: pointer to the second `SciVector`.
/// - `out`: pointer to an `f64` where the correlation will be written.
///
/// # Safety
///
/// - `x` and `y` must point to valid `SciVector`s with valid data.
/// - `out` must be a valid, non-null pointer to `f64`.
#[no_mangle]
pub unsafe extern "C" fn sci_correlation(
    x: *const SciVector,
    y: *const SciVector,
    out: *mut f64,
) -> SciResult {
    if out.is_null() {
        return SciResult::err("sci_correlation: out pointer is null");
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let x_data = unsafe { validate_vector(x, "sci_correlation (x)") }?;
        let y_data = unsafe { validate_vector(y, "sci_correlation (y)") }?;

        if x_data.len() != y_data.len() {
            return Err(format!(
                "sci_correlation: x and y must have the same length, got {} and {}",
                x_data.len(),
                y_data.len()
            ));
        }
        let n = x_data.len();
        if n < 2 {
            return Err(format!(
                "sci_correlation: need at least 2 elements, got {}",
                n
            ));
        }

        let x_mean: f64 = x_data.iter().sum::<f64>() / n as f64;
        let y_mean: f64 = y_data.iter().sum::<f64>() / n as f64;

        let mut cov = 0.0f64;
        let mut var_x = 0.0f64;
        let mut var_y = 0.0f64;

        for i in 0..n {
            let dx = x_data[i] - x_mean;
            let dy = y_data[i] - y_mean;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let denom = (var_x * var_y).sqrt();
        if denom == 0.0 {
            return Err("sci_correlation: one or both vectors have zero variance".to_string());
        }

        Ok(cov / denom)
    }));

    match result {
        Ok(Ok(corr)) => {
            unsafe { ptr::write(out, corr) };
            SciResult::ok()
        }
        Ok(Err(msg)) => SciResult::err(&msg),
        Err(e) => SciResult::from_panic(e),
    }
}

// ---------------------------------------------------------------------------
// sci_variance
// ---------------------------------------------------------------------------

/// Compute the variance of a vector.
///
/// # Parameters
///
/// - `vec`: pointer to a `SciVector`. Must have at least 1 element (or ddof+1).
/// - `ddof`: delta degrees of freedom (0 for population, 1 for sample).
/// - `out`: pointer to an `f64` where the variance will be written.
///
/// # Safety
///
/// - `vec` must point to a valid `SciVector` with valid data.
/// - `out` must be a valid, non-null pointer to `f64`.
#[no_mangle]
pub unsafe extern "C" fn sci_variance(
    vec: *const SciVector,
    ddof: usize,
    out: *mut f64,
) -> SciResult {
    if out.is_null() {
        return SciResult::err("sci_variance: out pointer is null");
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let data = unsafe { validate_vector(vec, "sci_variance") }?;
        let n = data.len();
        if n == 0 {
            return Err("sci_variance: empty vector".to_string());
        }
        if n <= ddof {
            return Err(format!(
                "sci_variance: need at least {} elements for ddof={}, got {}",
                ddof + 1,
                ddof,
                n
            ));
        }

        let mean: f64 = data.iter().sum::<f64>() / n as f64;
        let variance: f64 =
            data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (n - ddof) as f64;
        Ok(variance)
    }));

    match result {
        Ok(Ok(var)) => {
            unsafe { ptr::write(out, var) };
            SciResult::ok()
        }
        Ok(Err(msg)) => SciResult::err(&msg),
        Err(e) => SciResult::from_panic(e),
    }
}

// ---------------------------------------------------------------------------
// sci_median
// ---------------------------------------------------------------------------

/// Compute the median of a vector.
///
/// For an even number of elements, returns the average of the two middle values.
///
/// # Safety
///
/// - `vec` must point to a valid `SciVector` with valid data.
/// - `out` must be a valid, non-null pointer to `f64`.
#[no_mangle]
pub unsafe extern "C" fn sci_median(vec: *const SciVector, out: *mut f64) -> SciResult {
    if out.is_null() {
        return SciResult::err("sci_median: out pointer is null");
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let data = unsafe { validate_vector(vec, "sci_median") }?;
        if data.is_empty() {
            return Err("sci_median: empty vector".to_string());
        }

        let mut sorted: Vec<f64> = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        if n % 2 == 1 {
            Ok(sorted[n / 2])
        } else {
            Ok((sorted[n / 2 - 1] + sorted[n / 2]) / 2.0)
        }
    }));

    match result {
        Ok(Ok(med)) => {
            unsafe { ptr::write(out, med) };
            SciResult::ok()
        }
        Ok(Err(msg)) => SciResult::err(&msg),
        Err(e) => SciResult::from_panic(e),
    }
}
