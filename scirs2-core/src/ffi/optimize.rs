//! Optimization FFI functions.
//!
//! These functions expose scalar minimization and root-finding through the C ABI.
//!
//! The implementations are self-contained pure Rust (no external optimizer dependencies),
//! using golden-section search for minimization and Brent's method for root-finding.
//!
//! All functions follow the SciRS2 FFI conventions:
//! - Return [`SciResult`] to indicate success/failure.
//! - Never panic across the FFI boundary.
//! - Validate all pointers before dereferencing.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

use super::types::SciResult;

/// Type alias for a C function pointer: `double fn(double, void* user_data)`.
///
/// The `user_data` pointer is passed through opaquely so callers can attach
/// arbitrary context to the callback.
pub type SciFnPtr = unsafe extern "C" fn(f64, *mut std::ffi::c_void) -> f64;

// ---------------------------------------------------------------------------
// sci_minimize_scalar  --  golden-section search
// ---------------------------------------------------------------------------

/// Minimize a scalar function f(x) on the interval [a, b] using golden-section search.
///
/// The function pointer `f` is called as `f(x, user_data)` where `user_data`
/// is an opaque pointer provided by the caller (may be null).
///
/// # Parameters
///
/// - `f`: function pointer `double(double, void*)`.
/// - `user_data`: opaque pointer passed through to `f` (may be null).
/// - `a`: left endpoint of the search interval.
/// - `b`: right endpoint of the search interval. Must satisfy `a < b`.
/// - `tol`: convergence tolerance (absolute). Must be positive.
/// - `max_iter`: maximum number of iterations. If 0, defaults to 500.
/// - `x_out`: pointer where the minimizer x* will be written.
/// - `f_out`: pointer where the minimum value f(x*) will be written (may be null
///   if the caller doesn't need it).
///
/// # Safety
///
/// - `f` must be a valid function pointer that doesn't panic.
/// - `user_data` must be valid for all calls to `f` during this function.
/// - `x_out` must be a valid, non-null pointer to `f64`.
#[no_mangle]
pub unsafe extern "C" fn sci_minimize_scalar(
    f: SciFnPtr,
    user_data: *mut std::ffi::c_void,
    a: f64,
    b: f64,
    tol: f64,
    max_iter: usize,
    x_out: *mut f64,
    f_out: *mut f64,
) -> SciResult {
    if x_out.is_null() {
        return SciResult::err("sci_minimize_scalar: x_out pointer is null");
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        if a >= b {
            return Err(format!(
                "sci_minimize_scalar: a must be < b, got a={}, b={}",
                a, b
            ));
        }
        if tol <= 0.0 || tol.is_nan() {
            return Err(format!(
                "sci_minimize_scalar: tol must be positive, got {}",
                tol
            ));
        }

        let max_it = if max_iter == 0 { 500 } else { max_iter };

        // Golden section search
        let golden_ratio = (5.0_f64.sqrt() - 1.0) / 2.0;
        let mut lo = a;
        let mut hi = b;

        let mut x1 = hi - golden_ratio * (hi - lo);
        let mut x2 = lo + golden_ratio * (hi - lo);
        let mut f1 = unsafe { f(x1, user_data) };
        let mut f2 = unsafe { f(x2, user_data) };

        for _ in 0..max_it {
            if (hi - lo).abs() < tol {
                break;
            }

            if f1 < f2 {
                hi = x2;
                x2 = x1;
                f2 = f1;
                x1 = hi - golden_ratio * (hi - lo);
                f1 = unsafe { f(x1, user_data) };
            } else {
                lo = x1;
                x1 = x2;
                f1 = f2;
                x2 = lo + golden_ratio * (hi - lo);
                f2 = unsafe { f(x2, user_data) };
            }
        }

        let x_min = (lo + hi) / 2.0;
        let f_min = unsafe { f(x_min, user_data) };
        Ok((x_min, f_min))
    }));

    match result {
        Ok(Ok((x_min, f_min))) => {
            unsafe { ptr::write(x_out, x_min) };
            if !f_out.is_null() {
                unsafe { ptr::write(f_out, f_min) };
            }
            SciResult::ok()
        }
        Ok(Err(msg)) => SciResult::err(&msg),
        Err(e) => SciResult::from_panic(e),
    }
}

// ---------------------------------------------------------------------------
// sci_root_find  --  Brent's method
// ---------------------------------------------------------------------------

/// Find a root of f(x) = 0 on the interval [a, b] using Brent's method.
///
/// Requires that f(a) and f(b) have opposite signs (i.e., there is a sign change).
///
/// # Parameters
///
/// - `f`: function pointer `double(double, void*)`.
/// - `user_data`: opaque pointer passed through to `f` (may be null).
/// - `a`: left endpoint of the bracket.
/// - `b`: right endpoint of the bracket.
/// - `tol`: convergence tolerance (absolute). Must be positive.
/// - `max_iter`: maximum number of iterations. If 0, defaults to 500.
/// - `x_out`: pointer where the root will be written.
///
/// # Safety
///
/// - `f` must be a valid function pointer that doesn't panic.
/// - `user_data` must be valid for all calls to `f` during this function.
/// - `x_out` must be a valid, non-null pointer to `f64`.
#[no_mangle]
pub unsafe extern "C" fn sci_root_find(
    f: SciFnPtr,
    user_data: *mut std::ffi::c_void,
    a: f64,
    b: f64,
    tol: f64,
    max_iter: usize,
    x_out: *mut f64,
) -> SciResult {
    if x_out.is_null() {
        return SciResult::err("sci_root_find: x_out pointer is null");
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        if tol <= 0.0 || tol.is_nan() {
            return Err(format!("sci_root_find: tol must be positive, got {}", tol));
        }

        let max_it = if max_iter == 0 { 500 } else { max_iter };

        let fa = unsafe { f(a, user_data) };
        let fb = unsafe { f(b, user_data) };

        if fa * fb > 0.0 {
            return Err(format!(
                "sci_root_find: f(a) and f(b) must have opposite signs, got f({})={}, f({})={}",
                a, fa, b, fb
            ));
        }

        // Handle exact roots
        if fa.abs() < f64::EPSILON {
            return Ok(a);
        }
        if fb.abs() < f64::EPSILON {
            return Ok(b);
        }

        // Brent's method
        let mut a_curr = a;
        let mut b_curr = b;
        let mut fa_curr = fa;
        let mut fb_curr = fb;
        let mut c = a_curr;
        let mut fc = fa_curr;
        let mut d = b_curr - a_curr;
        let mut e = d;

        for _ in 0..max_it {
            if fb_curr.abs() < tol {
                return Ok(b_curr);
            }

            if fc * fb_curr > 0.0 {
                c = a_curr;
                fc = fa_curr;
                d = b_curr - a_curr;
                e = d;
            }

            if fc.abs() < fb_curr.abs() {
                a_curr = b_curr;
                b_curr = c;
                c = a_curr;
                fa_curr = fb_curr;
                fb_curr = fc;
                fc = fa_curr;
            }

            let tol1 = 2.0 * f64::EPSILON * b_curr.abs() + 0.5 * tol;
            let m = 0.5 * (c - b_curr);

            if m.abs() <= tol1 || fb_curr.abs() < f64::EPSILON {
                return Ok(b_curr);
            }

            if e.abs() >= tol1 && fa_curr.abs() > fb_curr.abs() {
                // Attempt inverse quadratic interpolation
                let s = fb_curr / fa_curr;
                let (p, q_val) = if (a_curr - c).abs() < f64::EPSILON {
                    let p = 2.0 * m * s;
                    let q = 1.0 - s;
                    (p, q)
                } else {
                    let q = fa_curr / fc;
                    let r = fb_curr / fc;
                    let p = s * (2.0 * m * q * (q - r) - (b_curr - a_curr) * (r - 1.0));
                    let q = (q - 1.0) * (r - 1.0) * (s - 1.0);
                    (p, q)
                };

                let (p, q_val) = if p > 0.0 { (p, -q_val) } else { (-p, q_val) };

                if 2.0 * p < (3.0 * m * q_val - (tol1 * q_val).abs()).min(e * q_val) {
                    e = d;
                    d = p / q_val;
                } else {
                    d = m;
                    e = m;
                }
            } else {
                d = m;
                e = m;
            }

            a_curr = b_curr;
            fa_curr = fb_curr;

            if d.abs() > tol1 {
                b_curr += d;
            } else if m > 0.0 {
                b_curr += tol1;
            } else {
                b_curr -= tol1;
            }

            fb_curr = unsafe { f(b_curr, user_data) };
        }

        // Max iterations reached, return best estimate
        Ok(b_curr)
    }));

    match result {
        Ok(Ok(root)) => {
            unsafe { ptr::write(x_out, root) };
            SciResult::ok()
        }
        Ok(Err(msg)) => SciResult::err(&msg),
        Err(e) => SciResult::from_panic(e),
    }
}

// ---------------------------------------------------------------------------
// sci_minimize_brent  --  Brent's method for minimization
// ---------------------------------------------------------------------------

/// Minimize a scalar function f(x) on the interval [a, b] using Brent's method.
///
/// This is generally faster than golden-section search because it uses
/// parabolic interpolation when possible.
///
/// # Parameters
///
/// Same as `sci_minimize_scalar`.
///
/// # Safety
///
/// Same safety requirements as `sci_minimize_scalar`.
#[no_mangle]
pub unsafe extern "C" fn sci_minimize_brent(
    f: SciFnPtr,
    user_data: *mut std::ffi::c_void,
    a: f64,
    b: f64,
    tol: f64,
    max_iter: usize,
    x_out: *mut f64,
    f_out: *mut f64,
) -> SciResult {
    if x_out.is_null() {
        return SciResult::err("sci_minimize_brent: x_out pointer is null");
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        if a >= b {
            return Err(format!(
                "sci_minimize_brent: a must be < b, got a={}, b={}",
                a, b
            ));
        }
        if tol <= 0.0 || tol.is_nan() {
            return Err(format!(
                "sci_minimize_brent: tol must be positive, got {}",
                tol
            ));
        }

        let max_it = if max_iter == 0 { 500 } else { max_iter };
        let golden = 0.381_966_011_250_105_1; // (3 - sqrt(5)) / 2

        let mut lo = a;
        let mut hi = b;
        let mut x = lo + golden * (hi - lo);
        let mut w = x;
        let mut v = x;
        let mut fx = unsafe { f(x, user_data) };
        let mut fw = fx;
        let mut fv = fx;
        let mut d_step = 0.0_f64;
        let mut e_step = 0.0_f64;

        for _ in 0..max_it {
            let midpoint = 0.5 * (lo + hi);
            let tol1 = tol * x.abs() + 1.0e-10;
            let tol2 = 2.0 * tol1;

            if (x - midpoint).abs() <= tol2 - 0.5 * (hi - lo) {
                // Converged
                let f_min = unsafe { f(x, user_data) };
                return Ok((x, f_min));
            }

            // Try parabolic interpolation
            let mut use_golden = true;
            if e_step.abs() > tol1 {
                // Fit parabola
                let r = (x - w) * (fx - fv);
                let q = (x - v) * (fx - fw);
                let p = (x - v) * q - (x - w) * r;
                let q = 2.0 * (q - r);
                let (p, q) = if q > 0.0 { (-p, q) } else { (p, -q) };

                if p.abs() < (0.5 * q * e_step).abs() && p > q * (lo - x) && p < q * (hi - x) {
                    // Take the parabolic step
                    d_step = p / q;
                    let u = x + d_step;
                    if (u - lo) < tol2 || (hi - u) < tol2 {
                        d_step = if x < midpoint { tol1 } else { -tol1 };
                    }
                    use_golden = false;
                }
            }

            if use_golden {
                e_step = if x < midpoint { hi - x } else { lo - x };
                d_step = golden * e_step;
            } else {
                e_step = d_step;
            }

            let u = if d_step.abs() >= tol1 {
                x + d_step
            } else if d_step > 0.0 {
                x + tol1
            } else {
                x - tol1
            };

            let fu = unsafe { f(u, user_data) };

            if fu <= fx {
                if u < x {
                    hi = x;
                } else {
                    lo = x;
                }
                v = w;
                fv = fw;
                w = x;
                fw = fx;
                x = u;
                fx = fu;
            } else {
                if u < x {
                    lo = u;
                } else {
                    hi = u;
                }
                if fu <= fw || (w - x).abs() < f64::EPSILON {
                    v = w;
                    fv = fw;
                    w = u;
                    fw = fu;
                } else if fu <= fv || (v - x).abs() < f64::EPSILON || (v - w).abs() < f64::EPSILON {
                    v = u;
                    fv = fu;
                }
            }
        }

        Ok((x, fx))
    }));

    match result {
        Ok(Ok((x_min, f_min))) => {
            unsafe { ptr::write(x_out, x_min) };
            if !f_out.is_null() {
                unsafe { ptr::write(f_out, f_min) };
            }
            SciResult::ok()
        }
        Ok(Err(msg)) => SciResult::err(&msg),
        Err(e) => SciResult::from_panic(e),
    }
}
