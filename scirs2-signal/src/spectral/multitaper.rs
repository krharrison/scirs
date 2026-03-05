//! Multitaper Spectrum Estimation (Thomson's method)
//!
//! Implements DPSS (Discrete Prolate Spheroidal Sequences) based multitaper
//! PSD estimation with adaptive weighting and F-test for spectral lines.
//!
//! # References
//!
//! - Thomson, D.J. (1982). "Spectrum estimation and harmonic analysis."
//!   Proceedings of the IEEE, 70(9), 1055-1096.
//! - Percival, D.B. and Walden, A.T. (1993). "Spectral Analysis for Physical
//!   Applications." Cambridge University Press.
//! - Slepian, D. (1978). "Prolate spheroidal wave functions, Fourier analysis,
//!   and uncertainty V." Bell System Technical Journal, 57(5), 1371-1430.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute next power of 2 >= n
fn next_pow2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

/// Compute FFT of real-valued slice, returns complex spectrum of length `nfft`.
/// Uses the Cooley-Tukey radix-2 DIT algorithm.
fn rfft(x: &[f64], nfft: usize) -> Vec<Complex64> {
    // Zero-pad or truncate to nfft
    let mut buf: Vec<Complex64> = Vec::with_capacity(nfft);
    for i in 0..nfft {
        let re = if i < x.len() { x[i] } else { 0.0 };
        buf.push(Complex64::new(re, 0.0));
    }
    fft_inplace(&mut buf);
    // Return only non-negative frequencies (0..nfft/2+1)
    let n_pos = nfft / 2 + 1;
    buf.truncate(n_pos);
    buf
}

/// In-place Cooley-Tukey radix-2 DIT FFT.  `buf.len()` must be a power of 2.
fn fft_inplace(buf: &mut Vec<Complex64>) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }
    // FFT butterfly stages
    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * PI / len as f64;
        let wlen = Complex64::new(ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let mut w = Complex64::new(1.0, 0.0);
            for k in 0..(len / 2) {
                let u = buf[i + k];
                let v = buf[i + k + len / 2] * w;
                buf[i + k] = u + v;
                buf[i + k + len / 2] = u - v;
                w *= wlen;
            }
            i += len;
        }
        len <<= 1;
    }
}

// ---------------------------------------------------------------------------
// DPSS (Slepian sequences) via tridiagonal eigenvalue method
// ---------------------------------------------------------------------------

/// Compute Discrete Prolate Spheroidal Sequences (DPSS) via the tridiagonal
/// symmetric eigenvalue method (Percival & Walden 1993, Chapter 8).
///
/// # Arguments
///
/// * `n`  – Sequence length.
/// * `nw` – Time-bandwidth product (typically 2.0 – 4.0).
/// * `k`  – Number of tapers to compute (typically `2*nw - 1`).
///
/// # Returns
///
/// A vector of `k` DPSS sequences, each of length `n`.  The sequences are
/// ordered by concentration (eigenvalue magnitude, decreasing).
///
/// # Errors
///
/// Returns `SignalError::ValueError` for invalid parameters.
///
/// # Example
///
/// ```
/// use scirs2_signal::spectral::multitaper::dpss_windows;
///
/// let tapers = dpss_windows(256, 4.0, 7).expect("dpss_windows failed");
/// assert_eq!(tapers.len(), 7);
/// assert_eq!(tapers[0].len(), 256);
/// ```
pub fn dpss_windows(n: usize, nw: f64, k: usize) -> SignalResult<Vec<Vec<f64>>> {
    if n < 2 {
        return Err(SignalError::ValueError(
            "dpss_windows: n must be >= 2".to_string(),
        ));
    }
    if nw <= 0.0 || nw >= n as f64 / 2.0 {
        return Err(SignalError::ValueError(format!(
            "dpss_windows: nw must be in (0, n/2), got nw={nw}, n={n}"
        )));
    }
    if k == 0 || k > n {
        return Err(SignalError::ValueError(format!(
            "dpss_windows: k must be in [1, {n}], got {k}"
        )));
    }

    let w = nw / n as f64; // normalised half-bandwidth

    // Build the symmetric tridiagonal matrix T of size n×n:
    //   diagonal[i]    = ((n-1)/2 - i)^2 * cos(2πW)
    //   off-diagonal[i] = i*(n-i)/2    (for i=1..n-1)
    let mut diag = vec![0.0f64; n];
    let mut off = vec![0.0f64; n - 1]; // off[i] = T[i, i+1]

    let cos2piw = (2.0 * PI * w).cos();
    for i in 0..n {
        let half = (n as f64 - 1.0) / 2.0 - i as f64;
        diag[i] = half * half * cos2piw;
    }
    for i in 0..(n - 1) {
        let i1 = (i + 1) as f64;
        off[i] = i1 * (n as f64 - i1) / 2.0;
    }

    // Compute the k largest eigenvalues / eigenvectors via QR iteration on the
    // tridiagonal matrix.  We use implicit QR shifts (LAPACK dstebz approach
    // reduced to pure Rust).
    let eigvecs = tridiag_eig_k_largest(&diag, &off, k)?;

    // Post-process: ensure positive polarity convention (first non-zero element positive)
    let mut tapers: Vec<Vec<f64>> = Vec::with_capacity(k);
    for mut v in eigvecs {
        // Normalize
        let norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > 1e-30 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        // Convention: sum of taper should be positive (or first non-zero positive)
        let s: f64 = v.iter().sum();
        if s < 0.0 {
            for x in v.iter_mut() {
                *x = -*x;
            }
        }
        tapers.push(v);
    }

    Ok(tapers)
}

// ---------------------------------------------------------------------------
// Tridiagonal symmetric eigenvector computation
// ---------------------------------------------------------------------------

/// Compute the `k` eigenvectors corresponding to the `k` largest (by absolute
/// value) eigenvalues of the symmetric tridiagonal matrix defined by `diag`
/// and `off`.
///
/// Uses inverse iteration with deflation to compute eigenvectors one by one.
fn tridiag_eig_k_largest(diag: &[f64], off: &[f64], k: usize) -> SignalResult<Vec<Vec<f64>>> {
    let n = diag.len();

    // First estimate all eigenvalues using the bisection method (Sturm sequence).
    let eigenvalues = tridiag_eigenvalues_bisection(diag, off)?;

    // Sort by absolute value (descending) and take the first k
    let mut indexed: Vec<(usize, f64)> = eigenvalues.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| {
        b.1.abs()
            .partial_cmp(&a.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let selected: Vec<(usize, f64)> = indexed.into_iter().take(k).collect();

    // For each selected eigenvalue, compute eigenvector by inverse iteration.
    let mut eigenvectors: Vec<Vec<f64>> = Vec::with_capacity(k);
    for &(_, lambda) in &selected {
        let v = inverse_iteration(diag, off, lambda, n)?;
        eigenvectors.push(v);
    }

    // Re-order eigenvectors so that they correspond to eigenvalues sorted by
    // concentration (largest first).  For DPSS the concentration ordering
    // matches the absolute-eigenvalue ordering already achieved above.
    Ok(eigenvectors)
}

/// Estimate all n eigenvalues of a symmetric tridiagonal matrix using Sturm
/// sequence bisection.
fn tridiag_eigenvalues_bisection(diag: &[f64], off: &[f64]) -> SignalResult<Vec<f64>> {
    let n = diag.len();
    // Gershgorin bound on spectral radius
    let mut lo = f64::MAX;
    let mut hi = f64::MIN;
    for i in 0..n {
        let r = if i == 0 {
            off[0].abs()
        } else if i == n - 1 {
            off[n - 2].abs()
        } else {
            off[i - 1].abs() + off[i].abs()
        };
        let lo_i = diag[i] - r;
        let hi_i = diag[i] + r;
        if lo_i < lo {
            lo = lo_i;
        }
        if hi_i > hi {
            hi = hi_i;
        }
    }
    lo -= 1.0;
    hi += 1.0;

    // Find each eigenvalue by bisection to locate exactly one root
    let mut eigenvalues: Vec<f64> = Vec::with_capacity(n);
    // Divide spectral interval into n sub-intervals using Sturm counts
    // to isolate each eigenvalue.
    let intervals = isolate_eigenvalues(diag, off, lo, hi, n)?;
    for (a, b) in intervals {
        let lam = bisect_eigenvalue(diag, off, a, b, 1e-12)?;
        eigenvalues.push(lam);
    }
    Ok(eigenvalues)
}

/// Sturm sequence count: number of eigenvalues < x.
fn sturm_count(diag: &[f64], off: &[f64], x: f64) -> usize {
    let n = diag.len();
    let mut count = 0usize;
    let mut d = diag[0] - x;
    if d < 0.0 {
        count += 1;
    }
    for i in 1..n {
        let b = off[i - 1];
        d = (diag[i] - x)
            - if d.abs() > 1e-300 {
                b * b / d
            } else {
                b.abs() / 1e-300
            };
        if d < 0.0 {
            count += 1;
        }
    }
    count
}

/// Isolate n eigenvalues into n separate sub-intervals via Sturm count.
fn isolate_eigenvalues(
    diag: &[f64],
    off: &[f64],
    lo: f64,
    hi: f64,
    n: usize,
) -> SignalResult<Vec<(f64, f64)>> {
    // Use divide-and-conquer: recursively split interval until we have singleton intervals.
    let mut intervals: Vec<(f64, f64)> = Vec::with_capacity(n);
    split_interval(
        diag,
        off,
        lo,
        hi,
        sturm_count(diag, off, lo),
        n,
        &mut intervals,
        0,
    )?;
    intervals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(intervals)
}

fn split_interval(
    diag: &[f64],
    off: &[f64],
    lo: f64,
    hi: f64,
    count_lo: usize,
    count_hi: usize,
    out: &mut Vec<(f64, f64)>,
    depth: usize,
) -> SignalResult<()> {
    if depth > 200 {
        // Safety guard; push whatever we have
        out.push((lo, hi));
        return Ok(());
    }
    let n_eig = count_hi.saturating_sub(count_lo);
    if n_eig == 0 {
        return Ok(());
    }
    if n_eig == 1 || (hi - lo) < 1e-14 {
        out.push((lo, hi));
        return Ok(());
    }
    let mid = (lo + hi) / 2.0;
    let count_mid = sturm_count(diag, off, mid);
    split_interval(diag, off, lo, mid, count_lo, count_mid, out, depth + 1)?;
    split_interval(diag, off, mid, hi, count_mid, count_hi, out, depth + 1)?;
    Ok(())
}

/// Bisect to find eigenvalue in (lo, hi).
fn bisect_eigenvalue(
    diag: &[f64],
    off: &[f64],
    mut lo: f64,
    mut hi: f64,
    tol: f64,
) -> SignalResult<f64> {
    let c_lo = sturm_count(diag, off, lo);
    for _ in 0..200 {
        if hi - lo <= tol * (1.0 + hi.abs() + lo.abs()) {
            break;
        }
        let mid = (lo + hi) / 2.0;
        let c_mid = sturm_count(diag, off, mid);
        if c_mid == c_lo {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok((lo + hi) / 2.0)
}

/// Compute eigenvector for eigenvalue `lambda` via inverse iteration with
/// Rayleigh quotient shifts.
fn inverse_iteration(diag: &[f64], off: &[f64], lambda: f64, n: usize) -> SignalResult<Vec<f64>> {
    // Start with random-ish seed vector
    let mut v: Vec<f64> = (0..n).map(|i| ((i as f64 + 1.0) * 0.7).sin()).collect();
    let norm_v = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm_v > 1e-30 {
        for x in v.iter_mut() {
            *x /= norm_v;
        }
    }

    // Perturb lambda to avoid exact singularity
    let shift = lambda + 1e-8;

    for _iter in 0..50 {
        // Solve (T - shift*I) w = v  (tridiagonal system via LU)
        let w = tridiag_solve(diag, off, shift, &v)?;
        // Normalize
        let norm_w = w.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm_w < 1e-30 {
            break;
        }
        for (vi, &wi) in v.iter_mut().zip(w.iter()) {
            *vi = wi / norm_w;
        }
    }
    Ok(v)
}

/// Solve (T - shift*I) x = b for tridiagonal T.
fn tridiag_solve(diag: &[f64], off: &[f64], shift: f64, b: &[f64]) -> SignalResult<Vec<f64>> {
    let n = diag.len();
    // Thomas algorithm (LU decomposition for tridiagonal)
    let mut c: Vec<f64> = vec![0.0; n]; // upper diagonal of U
    let mut d: Vec<f64> = b.to_vec(); // modified RHS

    // Diagonal of U
    c[0] = off[0] / (diag[0] - shift + 1e-30);
    d[0] /= diag[0] - shift + 1e-30;

    for i in 1..n {
        let a_i = if i > 0 { off[i - 1] } else { 0.0 };
        let denom = (diag[i] - shift) - a_i * c[i - 1];
        let denom = if denom.abs() < 1e-30 {
            denom.signum() * 1e-30
        } else {
            denom
        };
        if i < n - 1 {
            c[i] = off[i] / denom;
        }
        d[i] = (d[i] - a_i * d[i - 1]) / denom;
    }

    // Back substitution
    let mut x = d;
    for i in (0..(n - 1)).rev() {
        x[i] -= c[i] * x[i + 1];
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// Multitaper PSD
// ---------------------------------------------------------------------------

/// Compute the multitaper power spectral density estimate (Thomson's method).
///
/// # Arguments
///
/// * `x`   – Input time series.
/// * `fs`  – Sampling frequency in Hz.
/// * `nw`  – Time-bandwidth product (typical: 2.0–4.0).
/// * `k`   – Number of tapers (typical: `2*nw - 1`).
///
/// # Returns
///
/// `(frequencies, psd)` – Both vectors have length `n/2 + 1`.
///
/// # Errors
///
/// Returns `SignalError::ValueError` for invalid inputs.
///
/// # Example
///
/// ```
/// use scirs2_signal::spectral::multitaper::multitaper_psd;
///
/// let n = 256usize;
/// let fs = 256.0f64;
/// let x: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * 40.0 * i as f64 / fs).sin()).collect();
/// let (freqs, psd) = multitaper_psd(&x, fs, 4.0, 7).expect("multitaper_psd failed");
/// assert_eq!(freqs.len(), psd.len());
/// assert!(psd.iter().all(|&p| p >= 0.0));
/// ```
pub fn multitaper_psd(x: &[f64], fs: f64, nw: f64, k: usize) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    if x.is_empty() {
        return Err(SignalError::ValueError("x must not be empty".to_string()));
    }
    if fs <= 0.0 {
        return Err(SignalError::ValueError("fs must be positive".to_string()));
    }
    let n = x.len();
    let nfft = next_pow2(n);
    let n_pos = nfft / 2 + 1;

    let tapers = dpss_windows(n, nw, k)?;

    let mut psd = vec![0.0f64; n_pos];

    for taper in &tapers {
        // Apply taper and compute FFT
        let tapered: Vec<f64> = x
            .iter()
            .zip(taper.iter())
            .map(|(&xi, &wi)| xi * wi)
            .collect();
        let spectrum = rfft(&tapered, nfft);
        // Accumulate |X|^2, with doubled weight for non-DC/Nyquist bins
        for (i, s) in spectrum.iter().enumerate() {
            let power = s.re * s.re + s.im * s.im;
            psd[i] += power;
        }
    }

    // Normalise: average over tapers, convert to one-sided PSD
    let scale = 1.0 / (k as f64 * fs * nfft as f64);
    psd[0] *= scale;
    if n_pos > 1 {
        psd[n_pos - 1] *= scale;
        for p in psd[1..n_pos - 1].iter_mut() {
            *p *= 2.0 * scale; // one-sided: double for positive freqs
        }
    }

    let df = fs / nfft as f64;
    let freqs: Vec<f64> = (0..n_pos).map(|i| i as f64 * df).collect();

    Ok((freqs, psd))
}

// ---------------------------------------------------------------------------
// Adaptive multitaper PSD (Thomson 1982)
// ---------------------------------------------------------------------------

/// Compute adaptive-weighted multitaper PSD.
///
/// Iteratively adjusts taper weights so that each taper contributes
/// according to its spectral concentration relative to the broadband
/// noise level, reducing spectral leakage adaptively.
///
/// # Arguments
///
/// * `x`        – Input time series.
/// * `fs`       – Sampling frequency in Hz.
/// * `nw`       – Time-bandwidth product.
/// * `k`        – Number of tapers.
/// * `max_iter` – Maximum number of adaptive weighting iterations.
///
/// # Returns
///
/// `(frequencies, psd)` – Both vectors have length `nfft/2 + 1`.
///
/// # Example
///
/// ```
/// use scirs2_signal::spectral::multitaper::adaptive_multitaper_psd;
///
/// let n = 256usize;
/// let fs = 256.0f64;
/// let x: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * 30.0 * i as f64 / fs).sin()).collect();
/// let (freqs, psd) = adaptive_multitaper_psd(&x, fs, 4.0, 7, 3).expect("adaptive failed");
/// assert_eq!(freqs.len(), psd.len());
/// ```
pub fn adaptive_multitaper_psd(
    x: &[f64],
    fs: f64,
    nw: f64,
    k: usize,
    max_iter: usize,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    if x.is_empty() {
        return Err(SignalError::ValueError("x must not be empty".to_string()));
    }
    if fs <= 0.0 {
        return Err(SignalError::ValueError("fs must be positive".to_string()));
    }
    if max_iter == 0 {
        return Err(SignalError::ValueError("max_iter must be >= 1".to_string()));
    }

    let n = x.len();
    let nfft = next_pow2(n);
    let n_pos = nfft / 2 + 1;

    let tapers = dpss_windows(n, nw, k)?;

    // Compute eigenspectra for each taper
    let mut eigenspectra: Vec<Vec<f64>> = Vec::with_capacity(k);
    for taper in &tapers {
        let tapered: Vec<f64> = x
            .iter()
            .zip(taper.iter())
            .map(|(&xi, &wi)| xi * wi)
            .collect();
        let spectrum = rfft(&tapered, nfft);
        let ps: Vec<f64> = spectrum.iter().map(|s| s.re * s.re + s.im * s.im).collect();
        eigenspectra.push(ps);
    }

    // Concentration ratios λ_k: approximate as 1 - (small leakage)
    // For DPSS, the first k tapers have concentration ≈ 1 - epsilon_k
    // Use the approximation lambda_k ≈ 1 - 2^(1-2k) for NW >= k/2.
    let lambda: Vec<f64> = (0..k)
        .map(|i| 1.0 - 2.0_f64.powi(1 - 2 * i as i32).min(0.5))
        .collect();

    // Initial estimate: simple average
    let mut psd: Vec<f64> = vec![0.0f64; n_pos];
    for ek in &eigenspectra {
        for (i, &p) in ek.iter().enumerate() {
            psd[i] += p;
        }
    }
    for p in psd.iter_mut() {
        *p /= k as f64;
    }

    // Estimate broadband noise variance σ^2 from the initial estimate
    // (low-eigenvalue-weighted mean)
    let sigma2 = estimate_noise_variance(x);

    // Adaptive iteration
    for _iter in 0..max_iter {
        // Compute per-taper weights: b_k(f)^2 = lambda_k * S(f) / (lambda_k * S(f) + (1 - lambda_k) * sigma2)
        let mut psd_new = vec![0.0f64; n_pos];
        let mut weight_sum = vec![0.0f64; n_pos];

        for (kk, ek) in eigenspectra.iter().enumerate() {
            let lam = lambda[kk];
            for i in 0..n_pos {
                let s = psd[i].max(1e-30);
                let b2 = (lam * s) / (lam * s + (1.0 - lam) * sigma2.max(1e-30));
                psd_new[i] += b2 * ek[i];
                weight_sum[i] += b2;
            }
        }

        for i in 0..n_pos {
            psd[i] = if weight_sum[i] > 1e-30 {
                psd_new[i] / weight_sum[i]
            } else {
                psd_new[i]
            };
        }
    }

    // Normalise
    let scale = 1.0 / (fs * nfft as f64);
    psd[0] *= scale;
    if n_pos > 1 {
        psd[n_pos - 1] *= scale;
        for p in psd[1..n_pos - 1].iter_mut() {
            *p *= 2.0 * scale;
        }
    }

    let df = fs / nfft as f64;
    let freqs: Vec<f64> = (0..n_pos).map(|i| i as f64 * df).collect();

    Ok((freqs, psd))
}

/// Estimate broadband noise variance from time series.
fn estimate_noise_variance(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 1.0;
    }
    // Use 2nd differences to estimate noise variance robustly
    let sum_sq: f64 = x
        .windows(3)
        .map(|w| (w[2] - 2.0 * w[1] + w[0]).powi(2))
        .sum();
    sum_sq / (6.0 * (n - 2) as f64).max(1.0)
}

// ---------------------------------------------------------------------------
// F-test for spectral lines
// ---------------------------------------------------------------------------

/// Compute the F-test statistic for detecting spectral line components.
///
/// The F-test (Thomson 1982) identifies frequencies where the signal has a
/// significant sinusoidal component by comparing the periodic and broadband
/// spectral estimates.
///
/// # Arguments
///
/// * `x`  – Input time series.
/// * `fs` – Sampling frequency in Hz.
/// * `nw` – Time-bandwidth product.
///
/// # Returns
///
/// `(frequencies, f_statistics)` where large F values indicate spectral lines.
///
/// # Example
///
/// ```
/// use scirs2_signal::spectral::multitaper::f_test_statistic;
///
/// let n = 256usize;
/// let fs = 256.0f64;
/// let x: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * 40.0 * i as f64 / fs).sin()).collect();
/// let (freqs, fstats) = f_test_statistic(&x, fs, 4.0).expect("f_test failed");
/// assert_eq!(freqs.len(), fstats.len());
/// assert!(fstats.iter().all(|&f| f >= 0.0));
/// ```
pub fn f_test_statistic(x: &[f64], fs: f64, nw: f64) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    if x.is_empty() {
        return Err(SignalError::ValueError("x must not be empty".to_string()));
    }
    if fs <= 0.0 {
        return Err(SignalError::ValueError("fs must be positive".to_string()));
    }

    let n = x.len();
    let k = ((2.0 * nw).floor() as usize).max(2);
    let nfft = next_pow2(n);
    let n_pos = nfft / 2 + 1;

    let tapers = dpss_windows(n, nw, k)?;

    // Compute complex eigenspectra
    let mut complex_specs: Vec<Vec<Complex64>> = Vec::with_capacity(k);
    for taper in &tapers {
        let tapered: Vec<f64> = x
            .iter()
            .zip(taper.iter())
            .map(|(&xi, &wi)| xi * wi)
            .collect();
        let spectrum = rfft(&tapered, nfft);
        complex_specs.push(spectrum);
    }

    // Taper DC sums: u_k = Σ_n h_k(n) (the mean of each taper)
    let u: Vec<f64> = tapers
        .iter()
        .map(|taper| taper.iter().sum::<f64>())
        .collect();
    let sum_u2: f64 = u.iter().map(|&ui| ui * ui).sum();

    let mut f_stats = vec![0.0f64; n_pos];

    for i in 0..n_pos {
        // Estimate of sinusoidal amplitude: μ̂(f) = Σ_k u_k * X_k(f) / Σ_k u_k^2
        let mut num = Complex64::new(0.0, 0.0);
        for kk in 0..k {
            num += complex_specs[kk][i] * u[kk];
        }
        let mu_hat = if sum_u2 > 1e-30 {
            num / sum_u2
        } else {
            Complex64::new(0.0, 0.0)
        };

        // Periodic power: |μ̂|^2 * Σ u_k^2
        let periodic = (mu_hat.re * mu_hat.re + mu_hat.im * mu_hat.im) * sum_u2;

        // Broadband residual: Σ_k |X_k(f) - u_k * μ̂(f)|^2
        let mut broadband = 0.0f64;
        for kk in 0..k {
            let residual = complex_specs[kk][i] - mu_hat * u[kk];
            broadband += residual.re * residual.re + residual.im * residual.im;
        }
        broadband /= (k - 1) as f64;

        // F statistic: (K-1) * periodic / broadband
        f_stats[i] = if broadband > 1e-30 {
            periodic / broadband
        } else if periodic > 1e-30 {
            f64::MAX
        } else {
            0.0
        };
    }

    let df = fs / nfft as f64;
    let freqs: Vec<f64> = (0..n_pos).map(|i| i as f64 * df).collect();

    Ok((freqs, f_stats))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dpss_windows_basic() {
        let n = 128;
        let nw = 4.0;
        let k = 7;
        let tapers = dpss_windows(n, nw, k).expect("dpss_windows failed");
        assert_eq!(tapers.len(), k);
        for t in &tapers {
            assert_eq!(t.len(), n);
            // Each taper should be unit-norm (to within floating point)
            let norm: f64 = t.iter().map(|&x| x * x).sum::<f64>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-6,
                "Taper norm {norm} not close to 1.0"
            );
        }
    }

    #[test]
    fn test_dpss_windows_orthogonality() {
        let tapers = dpss_windows(64, 3.0, 5).expect("dpss_windows failed");
        for i in 0..tapers.len() {
            for j in (i + 1)..tapers.len() {
                let dot: f64 = tapers[i]
                    .iter()
                    .zip(tapers[j].iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                assert!(
                    dot.abs() < 0.05,
                    "Tapers {i} and {j} not orthogonal: dot = {dot}"
                );
            }
        }
    }

    #[test]
    fn test_dpss_invalid_params() {
        assert!(dpss_windows(0, 4.0, 7).is_err());
        assert!(dpss_windows(1, 4.0, 7).is_err());
        assert!(dpss_windows(64, -1.0, 7).is_err());
        assert!(dpss_windows(64, 4.0, 0).is_err());
    }

    #[test]
    fn test_multitaper_psd_sinusoid() {
        let n = 256;
        let fs = 256.0f64;
        let f0 = 40.0f64;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f0 * i as f64 / fs).sin())
            .collect();
        let (freqs, psd) = multitaper_psd(&x, fs, 4.0, 7).expect("multitaper_psd failed");
        assert_eq!(freqs.len(), psd.len());
        // PSD should be non-negative
        assert!(psd.iter().all(|&p| p >= 0.0), "PSD has negative values");
        // Peak should be near 40 Hz
        let peak_idx = psd
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let peak_freq = freqs[peak_idx];
        assert!(
            (peak_freq - f0).abs() < 5.0,
            "Peak at {peak_freq} Hz, expected near {f0} Hz"
        );
    }

    #[test]
    fn test_adaptive_multitaper_psd() {
        let n = 256;
        let fs = 256.0f64;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 30.0 * i as f64 / fs).sin())
            .collect();
        let (freqs, psd) =
            adaptive_multitaper_psd(&x, fs, 4.0, 7, 5).expect("adaptive_multitaper_psd failed");
        assert_eq!(freqs.len(), psd.len());
        assert!(psd.iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn test_f_test_statistic_detects_line() {
        let n = 256;
        let fs = 256.0f64;
        let f0 = 50.0f64;
        // Pure sinusoid: should show strong F-stat at f0
        let x: Vec<f64> = (0..n)
            .map(|i| 2.0 * (2.0 * PI * f0 * i as f64 / fs).sin())
            .collect();
        let (freqs, fstats) = f_test_statistic(&x, fs, 4.0).expect("f_test failed");
        assert_eq!(freqs.len(), fstats.len());
        assert!(fstats.iter().all(|&f| f >= 0.0));
        // F-stat at f0 should be among the largest
        let peak_idx = fstats
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let peak_freq = freqs[peak_idx];
        assert!(
            (peak_freq - f0).abs() < 5.0,
            "F-stat peak at {peak_freq} Hz, expected near {f0} Hz"
        );
    }
}
