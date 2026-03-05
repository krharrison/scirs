//! FFT (Fast Fourier Transform) FFI functions.
//!
//! These functions expose forward and inverse FFT operations through the C ABI.
//! The implementation uses a pure-Rust Cooley-Tukey radix-2 FFT algorithm
//! with Bluestein's algorithm fallback for non-power-of-2 lengths, ensuring
//! no C/Fortran dependencies (COOLJAPAN Pure Rust Policy).
//!
//! All functions follow the SciRS2 FFI conventions:
//! - Return [`SciResult`] to indicate success/failure.
//! - Never panic across the FFI boundary.
//! - Validate all pointers before dereferencing.

use std::f64::consts::PI;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

use super::types::{SciComplexVector, SciResult};

// ---------------------------------------------------------------------------
// Internal FFT implementation (pure Rust)
// ---------------------------------------------------------------------------

/// A complex number for internal FFT computation.
#[derive(Clone, Copy, Debug)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Complex { re, im }
    }

    fn mul(self, other: Self) -> Self {
        Complex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    fn add(self, other: Self) -> Self {
        Complex {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    fn sub(self, other: Self) -> Self {
        Complex {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

/// Check if n is a power of 2.
fn is_power_of_2(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Find the next power of 2 >= n.
fn next_power_of_2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut v = n - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v + 1
}

/// In-place iterative Cooley-Tukey radix-2 FFT.
///
/// `data` must have a power-of-2 length.
/// `inverse`: if true, compute the inverse FFT (without the 1/N scaling).
fn fft_radix2(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(is_power_of_2(n));

    // Bit-reversal permutation
    let mut j: usize = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }

    // Cooley-Tukey butterfly
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut size = 2;
    while size <= n {
        let half = size / 2;
        let angle = sign * 2.0 * PI / size as f64;
        let w_base = Complex::new(angle.cos(), angle.sin());

        let mut k = 0;
        while k < n {
            let mut w = Complex::new(1.0, 0.0);
            for jj in 0..half {
                let u = data[k + jj];
                let t = w.mul(data[k + jj + half]);
                data[k + jj] = u.add(t);
                data[k + jj + half] = u.sub(t);
                w = w.mul(w_base);
            }
            k += size;
        }
        size <<= 1;
    }
}

/// Bluestein's algorithm for arbitrary-length FFT.
///
/// This allows FFT of any length, not just powers of 2.
fn fft_bluestein(input: &[Complex], inverse: bool) -> Vec<Complex> {
    let n = input.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return input.to_vec();
    }

    // If power of 2, use radix-2 directly
    if is_power_of_2(n) {
        let mut data = input.to_vec();
        fft_radix2(&mut data, inverse);
        return data;
    }

    let sign = if inverse { 1.0 } else { -1.0 };

    // Compute chirp: w_k = exp(sign * i * pi * k^2 / n)
    let mut chirp = Vec::with_capacity(n);
    for k in 0..n {
        let angle = sign * PI * (k as f64 * k as f64) / n as f64;
        chirp.push(Complex::new(angle.cos(), angle.sin()));
    }

    // Convolution length: next power of 2 >= 2*n - 1
    let m = next_power_of_2(2 * n - 1);

    // a[k] = x[k] * conj(chirp[k])
    let mut a = vec![Complex::new(0.0, 0.0); m];
    for k in 0..n {
        let conj_chirp = Complex::new(chirp[k].re, -chirp[k].im);
        a[k] = input[k].mul(conj_chirp);
    }

    // b[k] = chirp[k] for k=0..n-1, and chirp[n-k] for k=m-n+1..m-1
    let mut b = vec![Complex::new(0.0, 0.0); m];
    b[0] = chirp[0];
    for k in 1..n {
        b[k] = chirp[k];
        b[m - k] = chirp[k];
    }

    // Forward FFT of a and b
    fft_radix2(&mut a, false);
    fft_radix2(&mut b, false);

    // Pointwise multiply
    for i in 0..m {
        a[i] = a[i].mul(b[i]);
    }

    // Inverse FFT
    fft_radix2(&mut a, true);

    // Scale by 1/m and multiply by conj(chirp)
    let scale = 1.0 / m as f64;
    let mut result = Vec::with_capacity(n);
    for k in 0..n {
        let scaled = Complex::new(a[k].re * scale, a[k].im * scale);
        let conj_chirp = Complex::new(chirp[k].re, -chirp[k].im);
        result.push(scaled.mul(conj_chirp));
    }

    result
}

// ---------------------------------------------------------------------------
// sci_fft_forward
// ---------------------------------------------------------------------------

/// Compute the forward FFT of a complex signal.
///
/// Input: two arrays of length `len` containing real and imaginary parts.
/// Output: a `SciComplexVector` with the FFT result. Must be freed by
/// `sci_complex_vector_free`.
///
/// Supports arbitrary lengths (not just powers of 2).
///
/// # Parameters
///
/// - `real_in`: pointer to `len` real-part values.
/// - `imag_in`: pointer to `len` imaginary-part values (may be null for real input,
///   in which case imaginary parts are assumed zero).
/// - `len`: number of elements.
/// - `out`: pointer to a `SciComplexVector` for the result.
///
/// # Safety
///
/// - `real_in` must point to at least `len` valid `f64` elements.
/// - `imag_in` must be null or point to at least `len` valid `f64` elements.
/// - `out` must be a valid, non-null pointer to `SciComplexVector`.
#[no_mangle]
pub unsafe extern "C" fn sci_fft_forward(
    real_in: *const f64,
    imag_in: *const f64,
    len: usize,
    out: *mut SciComplexVector,
) -> SciResult {
    if out.is_null() {
        return SciResult::err("sci_fft_forward: out pointer is null");
    }
    if real_in.is_null() && len > 0 {
        return SciResult::err("sci_fft_forward: real_in is null but len > 0");
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let real_slice = if len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(real_in, len) }
        };

        let imag_slice: Vec<f64> = if imag_in.is_null() {
            vec![0.0; len]
        } else if len == 0 {
            vec![]
        } else {
            unsafe { std::slice::from_raw_parts(imag_in, len) }.to_vec()
        };

        // Build complex input
        let input: Vec<Complex> = real_slice
            .iter()
            .zip(imag_slice.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();

        let output = fft_bluestein(&input, false);

        let real_out: Vec<f64> = output.iter().map(|c| c.re).collect();
        let imag_out: Vec<f64> = output.iter().map(|c| c.im).collect();

        SciComplexVector::from_vecs(real_out, imag_out)
            .ok_or_else(|| "sci_fft_forward: internal error creating result".to_string())
    }));

    match result {
        Ok(Ok(cv)) => {
            unsafe { ptr::write(out, cv) };
            SciResult::ok()
        }
        Ok(Err(msg)) => SciResult::err(&msg),
        Err(e) => SciResult::from_panic(e),
    }
}

// ---------------------------------------------------------------------------
// sci_fft_inverse
// ---------------------------------------------------------------------------

/// Compute the inverse FFT of a complex signal.
///
/// The result is scaled by 1/N. Input and output follow the same conventions
/// as `sci_fft_forward`.
///
/// # Safety
///
/// Same safety requirements as `sci_fft_forward`.
#[no_mangle]
pub unsafe extern "C" fn sci_fft_inverse(
    real_in: *const f64,
    imag_in: *const f64,
    len: usize,
    out: *mut SciComplexVector,
) -> SciResult {
    if out.is_null() {
        return SciResult::err("sci_fft_inverse: out pointer is null");
    }
    if real_in.is_null() && len > 0 {
        return SciResult::err("sci_fft_inverse: real_in is null but len > 0");
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let real_slice = if len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(real_in, len) }
        };

        let imag_slice: Vec<f64> = if imag_in.is_null() {
            vec![0.0; len]
        } else if len == 0 {
            vec![]
        } else {
            unsafe { std::slice::from_raw_parts(imag_in, len) }.to_vec()
        };

        let input: Vec<Complex> = real_slice
            .iter()
            .zip(imag_slice.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();

        let mut output = fft_bluestein(&input, true);

        // Scale by 1/N for inverse FFT
        let n = output.len() as f64;
        if n > 0.0 {
            for c in &mut output {
                c.re /= n;
                c.im /= n;
            }
        }

        let real_out: Vec<f64> = output.iter().map(|c| c.re).collect();
        let imag_out: Vec<f64> = output.iter().map(|c| c.im).collect();

        SciComplexVector::from_vecs(real_out, imag_out)
            .ok_or_else(|| "sci_fft_inverse: internal error creating result".to_string())
    }));

    match result {
        Ok(Ok(cv)) => {
            unsafe { ptr::write(out, cv) };
            SciResult::ok()
        }
        Ok(Err(msg)) => SciResult::err(&msg),
        Err(e) => SciResult::from_panic(e),
    }
}

/// Compute the forward FFT of a purely real signal.
///
/// This is a convenience function that avoids requiring the caller to
/// provide imaginary parts. Equivalent to calling `sci_fft_forward`
/// with `imag_in = null`.
///
/// # Safety
///
/// - `real_in` must point to at least `len` valid `f64` elements.
/// - `out` must be a valid, non-null pointer to `SciComplexVector`.
#[no_mangle]
pub unsafe extern "C" fn sci_rfft(
    real_in: *const f64,
    len: usize,
    out: *mut SciComplexVector,
) -> SciResult {
    unsafe { sci_fft_forward(real_in, ptr::null(), len, out) }
}
