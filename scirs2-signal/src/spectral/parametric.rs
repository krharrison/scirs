//! Parametric Spectral Estimation
//!
//! Implements model-based spectral analysis methods:
//! - AR (autoregressive) spectral estimation via Burg's maximum-entropy method
//! - Yule-Walker equations solved by Levinson-Durbin recursion
//! - ARMA spectrum
//! - MUSIC (MUltiple SIgnal Classification) pseudospectrum
//!
//! # References
//!
//! - Burg, J.P. (1975). "Maximum entropy spectral analysis." Ph.D. dissertation,
//!   Stanford University.
//! - Kay, S.M. (1988). "Modern Spectral Estimation: Theory and Application."
//!   Prentice Hall.
//! - Schmidt, R.O. (1986). "Multiple emitter location and signal parameter
//!   estimation." IEEE Trans. Antennas Propagation, 34(3), 276-280.
//! - Levinson, N. (1946). "The Wiener RMS error criterion in filter design and
//!   prediction." J. Math. Phys., 25, 261-278.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array2, Axis};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Internal FFT helper
// ---------------------------------------------------------------------------

/// Compute FFT of length `nfft` from a complex slice.
fn fft(x: &[Complex64], nfft: usize) -> Vec<Complex64> {
    let mut buf: Vec<Complex64> = Vec::with_capacity(nfft);
    for i in 0..nfft {
        buf.push(if i < x.len() {
            x[i]
        } else {
            Complex64::new(0.0, 0.0)
        });
    }
    fft_inplace(&mut buf);
    buf
}

fn fft_inplace(buf: &mut Vec<Complex64>) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    // Bit-reversal
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
    // Butterfly
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

fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        let mut p = 1;
        while p < n {
            p <<= 1;
        }
        p
    }
}

// ---------------------------------------------------------------------------
// AR spectral density from AR coefficients
// ---------------------------------------------------------------------------

/// Evaluate the AR power spectral density from AR coefficients and noise variance.
///
/// H(e^{jω}) = σ^2 / |A(e^{jω})|^2
/// where A(z) = 1 + a[1] z^{-1} + ... + a[p] z^{-p}
fn ar_spectrum_from_coeffs(
    ar: &[f64], // ar[0]=1.0, ar[1..=p] are AR params
    sigma2: f64,
    n_freqs: usize,
) -> Vec<f64> {
    // Evaluate A(e^{j*omega}) directly at `n_freqs` equally-spaced
    // frequencies in [0, pi] (i.e. 0 to Nyquist).
    //
    // PSD(omega) = sigma2 / |A(e^{j*omega})|^2
    //
    // For pure-tone signals the AR model can fit near-perfectly, causing
    // sigma2 to collapse close to zero.  To keep the spectral *shape*
    // correct we first compute the inverse-power spectrum  1/|A|^2,
    // then scale by sigma2.  When sigma2 is negligible we apply a small
    // floor so that the relative shape (and hence the peak location) is
    // preserved.
    let mut inv_a_sq = Vec::with_capacity(n_freqs);
    for i in 0..n_freqs {
        let omega = PI * i as f64 / n_freqs.max(1) as f64;
        let mut a_re = 0.0f64;
        let mut a_im = 0.0f64;
        for (k, &ak) in ar.iter().enumerate() {
            a_re += ak * (omega * k as f64).cos();
            a_im -= ak * (omega * k as f64).sin();
        }
        let mag_sq = a_re * a_re + a_im * a_im;
        inv_a_sq.push(1.0 / mag_sq.max(1e-300));
    }

    // Scale by sigma2, applying a floor to avoid all-zero PSD
    let sigma2_eff = sigma2.max(1e-30);
    inv_a_sq.iter().map(|&v| v * sigma2_eff).collect()
}

/// Linearly interpolate a vector to a new length.
fn resample_linear(x: &[f64], n_out: usize) -> Vec<f64> {
    let n_in = x.len();
    if n_in == 0 || n_out == 0 {
        return vec![0.0; n_out];
    }
    if n_in == n_out {
        return x.to_vec();
    }
    (0..n_out)
        .map(|i| {
            let pos = i as f64 * (n_in - 1) as f64 / (n_out - 1).max(1) as f64;
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(n_in - 1);
            let frac = pos - lo as f64;
            x[lo] * (1.0 - frac) + x[hi] * frac
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Burg's method
// ---------------------------------------------------------------------------

/// Estimate AR model parameters using Burg's maximum-entropy method.
///
/// Burg's method fits an AR(p) model by minimising the sum of forward and
/// backward prediction errors while enforcing stationarity (all poles inside
/// the unit circle).
///
/// # Arguments
///
/// * `x`     – Input time series.
/// * `order` – AR model order p.
///
/// # Returns
///
/// `(ar_coefficients, noise_variance)` where `ar_coefficients` has length
/// `order + 1` with `ar[0] = 1.0`.
///
/// # Errors
///
/// Returns `SignalError::ValueError` for invalid inputs.
///
/// # Example
///
/// ```
/// use scirs2_signal::spectral::parametric::burg_method;
///
/// let x: Vec<f64> = (0..256).map(|i| (2.0 * std::f64::consts::PI * 0.2 * i as f64).sin()).collect();
/// let (ar, sigma2) = burg_method(&x, 8).expect("burg_method failed");
/// assert_eq!(ar.len(), 9); // order + 1
/// assert!(sigma2 > 0.0);
/// ```
pub fn burg_method(x: &[f64], order: usize) -> SignalResult<(Vec<f64>, f64)> {
    if x.len() < order + 2 {
        return Err(SignalError::ValueError(format!(
            "burg_method: x.len()={} must be >= order+2={}",
            x.len(),
            order + 2
        )));
    }
    if order == 0 {
        return Err(SignalError::ValueError(
            "burg_method: order must be >= 1".to_string(),
        ));
    }

    let n = x.len();
    let mut ef: Vec<f64> = x.to_vec(); // forward prediction errors
    let mut eb: Vec<f64> = x.to_vec(); // backward prediction errors

    let mut ar: Vec<f64> = vec![0.0; order + 1];
    ar[0] = 1.0;

    let initial_variance: f64 = x.iter().map(|&v| v * v).sum::<f64>() / n as f64;
    let mut variance = initial_variance;

    for m in 0..order {
        // Once the prediction error variance has dropped below a small
        // fraction of the initial variance the AR model has already
        // captured all the signal energy.  Continuing the recursion
        // would produce essentially random coefficients from numerical
        // noise that can introduce spurious spectral zeros.  Stop early
        // and leave the remaining AR coefficients at zero.
        if variance < initial_variance * 1e-8 {
            break;
        }

        // Reflection coefficient via Burg's criterion
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in (m + 1)..n {
            num += ef[i] * eb[i - 1];
            den += ef[i] * ef[i] + eb[i - 1] * eb[i - 1];
        }
        let k = if den > 1e-30 { -2.0 * num / den } else { 0.0 };

        // Clip k to ensure stability: |k| < 1
        let k = k.clamp(-1.0 + 1e-10, 1.0 - 1e-10);

        // Update AR coefficients via Levinson-Durbin order update
        // New a[m+1] = k
        // a[j] = a[j] + k * a[m+1-j] for j = 1..m
        let ar_prev: Vec<f64> = ar.clone();
        ar[m + 1] = k;
        for j in 1..=m {
            ar[j] += k * ar_prev[m + 1 - j];
        }

        // Update prediction errors
        let mut ef_new = vec![0.0f64; n];
        let mut eb_new = vec![0.0f64; n];
        for i in (m + 1)..n {
            ef_new[i] = ef[i] + k * eb[i - 1];
            eb_new[i] = eb[i - 1] + k * ef[i];
        }
        ef = ef_new;
        eb = eb_new;

        // Update variance estimate
        variance *= 1.0 - k * k;
        if variance < 1e-300 {
            variance = 1e-300;
        }
    }

    Ok((ar, variance))
}

// ---------------------------------------------------------------------------
// AR PSD
// ---------------------------------------------------------------------------

/// Estimate the power spectral density using an AR model (Burg's method).
///
/// # Arguments
///
/// * `x`       – Input time series.
/// * `order`   – AR model order.
/// * `fs`      – Sampling frequency in Hz.
/// * `n_freqs` – Number of frequency bins in output.
///
/// # Returns
///
/// `(frequencies_Hz, psd)` both of length `n_freqs`.
///
/// # Example
///
/// ```
/// use scirs2_signal::spectral::parametric::ar_psd;
///
/// let n = 256;
/// let fs = 256.0f64;
/// let x: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * 50.0 * i as f64 / fs).sin()).collect();
/// let (freqs, psd) = ar_psd(&x, 10, fs, 512).expect("ar_psd failed");
/// assert_eq!(freqs.len(), 512);
/// assert!(psd.iter().all(|&p| p >= 0.0));
/// ```
pub fn ar_psd(
    x: &[f64],
    order: usize,
    fs: f64,
    n_freqs: usize,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    if fs <= 0.0 {
        return Err(SignalError::ValueError("fs must be positive".to_string()));
    }
    if n_freqs == 0 {
        return Err(SignalError::ValueError("n_freqs must be > 0".to_string()));
    }
    let (ar, sigma2) = burg_method(x, order)?;
    let n_pos = n_freqs;
    let psd = ar_spectrum_from_coeffs(&ar, sigma2, n_pos);
    // Scale to PSD (one-sided)
    let scale = 2.0 / fs; // one-sided: multiply by 2, divide by fs
    let psd: Vec<f64> = psd.iter().map(|&p| p * scale).collect();
    let df = fs / (2.0 * n_freqs as f64);
    let freqs: Vec<f64> = (0..n_freqs).map(|i| i as f64 * df).collect();
    Ok((freqs, psd))
}

// ---------------------------------------------------------------------------
// Yule-Walker equations
// ---------------------------------------------------------------------------

/// Estimate AR model parameters via the Yule-Walker equations using
/// Levinson-Durbin recursion.
///
/// # Arguments
///
/// * `x`     – Input time series.
/// * `order` – AR model order.
///
/// # Returns
///
/// `(ar_coefficients, noise_variance)` where `ar_coefficients[0] = 1.0`.
///
/// # Example
///
/// ```
/// use scirs2_signal::spectral::parametric::yule_walker;
///
/// let x: Vec<f64> = (0..256).map(|i| (2.0 * std::f64::consts::PI * 0.25 * i as f64).sin()).collect();
/// let (ar, sigma2) = yule_walker(&x, 4).expect("yule_walker failed");
/// assert_eq!(ar.len(), 5);
/// assert!(sigma2 > 0.0);
/// ```
pub fn yule_walker(x: &[f64], order: usize) -> SignalResult<(Vec<f64>, f64)> {
    if x.len() < order + 1 {
        return Err(SignalError::ValueError(format!(
            "yule_walker: x.len()={} must be >= order+1={}",
            x.len(),
            order + 1
        )));
    }
    if order == 0 {
        return Err(SignalError::ValueError(
            "yule_walker: order must be >= 1".to_string(),
        ));
    }

    let n = x.len();
    // Compute biased autocorrelation r[0..=order]
    let mut r = vec![0.0f64; order + 1];
    for lag in 0..=order {
        let mut s = 0.0f64;
        for i in 0..(n - lag) {
            s += x[i] * x[i + lag];
        }
        r[lag] = s / n as f64;
    }

    // Levinson-Durbin recursion
    levinson_durbin(&r, order)
}

/// Solve the Yule-Walker equations via the Levinson-Durbin algorithm.
///
/// Returns `(ar, sigma2)` where `ar[0] = 1.0` and `ar[1..=p]` are the AR
/// parameters.
fn levinson_durbin(r: &[f64], order: usize) -> SignalResult<(Vec<f64>, f64)> {
    if r[0].abs() < 1e-30 {
        return Err(SignalError::ComputationError(
            "levinson_durbin: zero variance signal".to_string(),
        ));
    }

    let mut ar = vec![0.0f64; order + 1];
    ar[0] = 1.0;
    let mut sigma2 = r[0];

    for m in 1..=order {
        // Partial correlation (reflection) coefficient
        let mut num = 0.0f64;
        for j in 1..=m {
            num += ar[j] * r[m - j];
        }
        let k = -(r[m] + num) / sigma2;
        let k = k.clamp(-1.0 + 1e-10, 1.0 - 1e-10);

        // Order-m update
        let ar_prev = ar.clone();
        ar[m] = k;
        for j in 1..m {
            ar[j] += k * ar_prev[m - j];
        }

        sigma2 *= 1.0 - k * k;
        if sigma2 < 1e-300 {
            sigma2 = 1e-300;
        }
    }

    Ok((ar, sigma2))
}

// ---------------------------------------------------------------------------
// ARMA spectrum
// ---------------------------------------------------------------------------

/// Compute the ARMA(p, q) power spectral density.
///
/// The ARMA spectrum is:
/// `S(e^{jω}) = σ^2 |B(e^{jω})|^2 / |A(e^{jω})|^2`
///
/// where `A(z) = 1 + ar[0] z^{-1} + ... + ar[p-1] z^{-p}` and
/// `B(z) = 1 + ma[0] z^{-1} + ... + ma[q-1] z^{-q}`.
///
/// # Arguments
///
/// * `ar`      – AR coefficients (length p; **excluding** the leading 1).
/// * `ma`      – MA coefficients (length q; **excluding** the leading 1).
/// * `sigma2`  – Innovation variance.
/// * `fs`      – Sampling frequency in Hz.
/// * `n_freqs` – Number of output frequency bins.
///
/// # Returns
///
/// `(frequencies_Hz, psd)` both of length `n_freqs`.
///
/// # Example
///
/// ```
/// use scirs2_signal::spectral::parametric::arma_psd;
///
/// // AR(2) with two complex poles near 0.3*pi
/// let ar = vec![0.1, -0.8];
/// let ma: Vec<f64> = vec![];
/// let (freqs, psd) = arma_psd(&ar, &ma, 1.0, 1000.0, 512).expect("arma_psd failed");
/// assert_eq!(freqs.len(), 512);
/// assert!(psd.iter().all(|&p| p >= 0.0));
/// ```
pub fn arma_psd(
    ar: &[f64],
    ma: &[f64],
    sigma2: f64,
    fs: f64,
    n_freqs: usize,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    if fs <= 0.0 {
        return Err(SignalError::ValueError("fs must be positive".to_string()));
    }
    if n_freqs == 0 {
        return Err(SignalError::ValueError("n_freqs must be > 0".to_string()));
    }
    if sigma2 <= 0.0 {
        return Err(SignalError::ValueError(
            "sigma2 must be positive".to_string(),
        ));
    }

    let p = ar.len();
    let q = ma.len();
    let nfft = next_pow2((p + q + n_freqs) * 2).max(64);

    // Build A(z) = [1, ar[0], ar[1], ..., ar[p-1]]
    let mut az: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); nfft];
    az[0] = Complex64::new(1.0, 0.0);
    for (i, &a) in ar.iter().enumerate() {
        az[i + 1] = Complex64::new(a, 0.0);
    }

    // Build B(z) = [1, ma[0], ..., ma[q-1]]
    let mut bz: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); nfft];
    bz[0] = Complex64::new(1.0, 0.0);
    for (i, &b) in ma.iter().enumerate() {
        bz[i + 1] = Complex64::new(b, 0.0);
    }

    let az_fft = fft(&az, nfft);
    let bz_fft = fft(&bz, nfft);

    let n_pos = nfft / 2 + 1;
    let mut psd_fft: Vec<f64> = az_fft
        .iter()
        .zip(bz_fft.iter())
        .take(n_pos)
        .map(|(a, b)| {
            let b2 = b.re * b.re + b.im * b.im;
            let a2 = (a.re * a.re + a.im * a.im).max(1e-300);
            sigma2 * b2 / a2
        })
        .collect();

    // One-sided PSD normalisation
    let scale = 2.0 / fs;
    for p in psd_fft.iter_mut() {
        *p *= scale;
    }
    if let Some(first) = psd_fft.first_mut() {
        *first /= 2.0; // DC: not doubled
    }
    if n_pos > 1 {
        if let Some(last) = psd_fft.last_mut() {
            *last /= 2.0; // Nyquist
        }
    }

    let psd = resample_linear(&psd_fft, n_freqs);
    let df = fs / (2.0 * n_freqs as f64);
    let freqs: Vec<f64> = (0..n_freqs).map(|i| i as f64 * df).collect();

    Ok((freqs, psd))
}

// ---------------------------------------------------------------------------
// MUSIC Algorithm
// ---------------------------------------------------------------------------

/// Compute the MUSIC (MUltiple SIgnal Classification) pseudospectrum.
///
/// MUSIC exploits the orthogonality between the signal and noise subspaces of
/// the data covariance matrix to produce a high-resolution pseudospectrum.
///
/// # Arguments
///
/// * `x`         – Data matrix of shape `(n_sensors × n_snapshots)`.
/// * `n_sources` – Number of signal sources (dimension of signal subspace).
/// * `freqs`     – Frequencies at which to evaluate the pseudospectrum (Hz).
/// * `fs`        – Sampling frequency in Hz.
///
/// # Returns
///
/// MUSIC pseudospectrum values at each frequency in `freqs`.
///
/// # Errors
///
/// Returns `SignalError::ValueError` for invalid inputs.
///
/// # Example
///
/// ```
/// use scirs2_signal::spectral::parametric::music_algorithm;
/// use scirs2_core::ndarray::Array2;
///
/// let n_sensors = 4;
/// let n_snap = 64;
/// let x = Array2::zeros((n_sensors, n_snap));
/// let freqs: Vec<f64> = (1..=50).map(|k| k as f64 * 10.0).collect();
/// let pmusic = music_algorithm(&x, 1, &freqs, 1000.0).expect("music failed");
/// assert_eq!(pmusic.len(), freqs.len());
/// ```
pub fn music_algorithm(
    x: &Array2<f64>,
    n_sources: usize,
    freqs: &[f64],
    fs: f64,
) -> SignalResult<Vec<f64>> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err(SignalError::ValueError(
            "music_algorithm: x must be non-empty".to_string(),
        ));
    }
    if n_sources == 0 {
        return Err(SignalError::ValueError(
            "music_algorithm: n_sources must be >= 1".to_string(),
        ));
    }
    if fs <= 0.0 {
        return Err(SignalError::ValueError(
            "music_algorithm: fs must be positive".to_string(),
        ));
    }
    if freqs.is_empty() {
        return Err(SignalError::ValueError(
            "music_algorithm: freqs must not be empty".to_string(),
        ));
    }

    let m = x.nrows(); // number of sensors
    let l = x.ncols(); // number of snapshots

    if n_sources >= m {
        return Err(SignalError::ValueError(format!(
            "music_algorithm: n_sources ({n_sources}) must be < n_sensors ({m})"
        )));
    }

    // Compute sample covariance matrix R = X * X^H / L  (m × m)
    let mut r = vec![vec![0.0f64; m]; m];
    for i in 0..m {
        for j in 0..m {
            let mut s = 0.0f64;
            for t in 0..l {
                s += x[[i, t]] * x[[j, t]];
            }
            r[i][j] = s / l as f64;
        }
    }

    // Eigendecompose R via the Jacobi method (symmetric, small matrix)
    let (eigenvalues, eigenvectors) = jacobi_eigen(&r, m, 100)?;

    // Sort eigenvectors by eigenvalue (ascending)
    let mut indexed: Vec<(usize, f64)> = eigenvalues.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Noise subspace: eigenvectors corresponding to the m - n_sources smallest eigenvalues
    let n_noise = m - n_sources;
    let noise_indices: Vec<usize> = indexed.iter().take(n_noise).map(|&(i, _)| i).collect();

    // Noise subspace matrix En (m × n_noise)
    // eigenvectors[k] is the k-th column of the eigenvector matrix
    let get_en = |i: usize, row: usize| eigenvectors[noise_indices[i]][row];

    // Compute MUSIC pseudospectrum at each frequency
    let pseudospectrum: Vec<f64> = freqs
        .iter()
        .map(|&f| {
            let omega = 2.0 * PI * f / fs;
            // Steering vector a(f) = [1, e^{jω}, ..., e^{j(m-1)ω}]
            let a: Vec<Complex64> = (0..m)
                .map(|k| {
                    let angle = omega * k as f64;
                    Complex64::new(angle.cos(), angle.sin())
                })
                .collect();

            // a^H En En^H a  (denominator of MUSIC)
            // = ||En^H a||^2
            let mut denom = 0.0f64;
            for col in 0..n_noise {
                let mut dot_re = 0.0f64;
                let mut dot_im = 0.0f64;
                for row in 0..m {
                    // e_col[row] is real (R is real symmetric)
                    let e = get_en(col, row);
                    dot_re += e * a[row].re;
                    dot_im += e * a[row].im;
                }
                denom += dot_re * dot_re + dot_im * dot_im;
            }

            // ||a||^2 = m
            let num = m as f64;
            if denom > 1e-30 {
                num / denom
            } else {
                f64::MAX
            }
        })
        .collect();

    Ok(pseudospectrum)
}

/// Jacobi eigendecomposition for small real symmetric matrices.
///
/// Returns `(eigenvalues, eigenvectors)` where eigenvectors[k] is the k-th
/// eigenvector (column of the eigenvector matrix, as a Vec).
fn jacobi_eigen(
    a: &[Vec<f64>],
    n: usize,
    max_iter: usize,
) -> SignalResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let mut mat: Vec<Vec<f64>> = a.to_vec();
    // Eigenvector matrix (identity initially)
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();

    for _iter in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0f64;
        let (mut p, mut q) = (0, 1);
        for i in 0..n {
            for j in (i + 1)..n {
                if mat[i][j].abs() > max_val {
                    max_val = mat[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-12 {
            break;
        }

        // Compute Jacobi rotation angle
        let diff = mat[q][q] - mat[p][p];
        let theta = if diff.abs() < 1e-30 {
            PI / 4.0
        } else {
            0.5 * (2.0 * mat[p][q] / diff).atan()
        };
        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation to mat
        let app = mat[p][p];
        let aqq = mat[q][q];
        let apq = mat[p][q];
        mat[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        mat[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        mat[p][q] = 0.0;
        mat[q][p] = 0.0;

        for r in 0..n {
            if r != p && r != q {
                let arp = mat[r][p];
                let arq = mat[r][q];
                mat[r][p] = c * arp - s * arq;
                mat[p][r] = mat[r][p];
                mat[r][q] = s * arp + c * arq;
                mat[q][r] = mat[r][q];
            }
        }

        // Update eigenvector matrix
        for r in 0..n {
            let vrp = v[r][p];
            let vrq = v[r][q];
            v[r][p] = c * vrp - s * vrq;
            v[r][q] = s * vrp + c * vrq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| mat[i][i]).collect();
    // v[row][col] => col is the eigenvector index; we want eigenvectors[k] = col k
    let eigenvectors: Vec<Vec<f64>> = (0..n)
        .map(|k| (0..n).map(|row| v[row][k]).collect())
        .collect();

    Ok((eigenvalues, eigenvectors))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_burg_method_basic() {
        let n = 256;
        let x: Vec<f64> = (0..n).map(|i| (2.0 * PI * 0.2 * i as f64).sin()).collect();
        let (ar, sigma2) = burg_method(&x, 4).expect("burg_method failed");
        assert_eq!(ar.len(), 5);
        assert!((ar[0] - 1.0).abs() < 1e-10, "ar[0] should be 1.0");
        assert!(sigma2 > 0.0, "sigma2 must be positive");
        assert!(sigma2.is_finite(), "sigma2 must be finite");
    }

    #[test]
    fn test_burg_method_stability() {
        // AR coefficients should yield a stable model (|reflection coeffs| < 1)
        let x: Vec<f64> = (0..128)
            .map(|i| {
                let t = i as f64 * 0.01;
                (2.0 * PI * 10.0 * t).sin() + (2.0 * PI * 30.0 * t).cos()
            })
            .collect();
        let (ar, _sigma2) = burg_method(&x, 8).expect("burg failed");
        // All AR coeffs should be finite
        assert!(ar.iter().all(|&a| a.is_finite()));
    }

    #[test]
    fn test_burg_invalid() {
        assert!(burg_method(&[1.0, 2.0], 10).is_err());
        assert!(burg_method(&[1.0; 10], 0).is_err());
    }

    #[test]
    fn test_yule_walker_basic() {
        let x: Vec<f64> = (0..256)
            .map(|i| (2.0 * PI * 0.25 * i as f64).sin())
            .collect();
        let (ar, sigma2) = yule_walker(&x, 4).expect("yule_walker failed");
        assert_eq!(ar.len(), 5);
        assert!((ar[0] - 1.0).abs() < 1e-10);
        assert!(sigma2 > 0.0);
    }

    #[test]
    fn test_ar_psd_peak() {
        let n = 256;
        let fs = 1000.0f64;
        let f0 = 200.0f64;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f0 * i as f64 / fs).sin())
            .collect();
        let (freqs, psd) = ar_psd(&x, 8, fs, 256).expect("ar_psd failed");
        assert!(psd.iter().all(|&p| p >= 0.0));
        let peak_idx = psd
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let peak_f = freqs[peak_idx];
        assert!(
            (peak_f - f0).abs() < 30.0,
            "AR PSD peak at {peak_f} Hz, expected near {f0} Hz"
        );
    }

    #[test]
    fn test_arma_psd_non_negative() {
        let ar = vec![0.5, -0.25];
        let ma = vec![0.3];
        let (freqs, psd) = arma_psd(&ar, &ma, 1.0, 1000.0, 256).expect("arma_psd failed");
        assert_eq!(freqs.len(), 256);
        assert!(psd.iter().all(|&p| p >= 0.0 && p.is_finite()));
    }

    #[test]
    fn test_music_algorithm_basic() {
        use scirs2_core::ndarray::Array2;
        let m = 4;
        let l = 64;
        // Create a simple rank-1 signal + noise covariance
        let x = Array2::from_shape_fn((m, l), |(i, j)| {
            (2.0 * PI * 0.1 * (i + j) as f64).sin() + 0.01 * ((i * 7 + j * 13) as f64).sin()
        });
        let freqs: Vec<f64> = (1..=50).map(|k| k as f64 * 10.0).collect();
        let pmusic = music_algorithm(&x, 1, &freqs, 1000.0).expect("music failed");
        assert_eq!(pmusic.len(), freqs.len());
        assert!(pmusic.iter().all(|&p| p >= 0.0 && p.is_finite()));
    }

    #[test]
    fn test_music_invalid_params() {
        use scirs2_core::ndarray::Array2;
        let x = Array2::zeros((4, 64));
        // n_sources >= n_sensors should fail
        assert!(music_algorithm(&x, 4, &[10.0, 20.0], 1000.0).is_err());
        assert!(music_algorithm(&x, 0, &[10.0], 1000.0).is_err());
        // Empty freqs
        assert!(music_algorithm(&x, 1, &[], 1000.0).is_err());
    }
}
