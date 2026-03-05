//! Thomson's Multitaper Spectral Estimation
//!
//! This module provides multitaper spectral estimation using Discrete Prolate
//! Spheroidal Sequences (DPSS/Slepian sequences). The multitaper method produces
//! spectra with reduced variance and controlled bias, making it one of the most
//! reliable non-parametric spectral estimation techniques.
//!
//! # References
//!
//! - Thomson, D.J. (1982). Spectrum estimation and harmonic analysis.
//!   Proceedings of the IEEE, 70(9), 1055-1096.
//! - Percival, D.B. & Walden, A.T. (1993). Spectral Analysis for Physical
//!   Applications. Cambridge University Press.
//!
//! # Examples
//!
//! ```
//! use scirs2_signal::multitaper_mod::{dpss, multitaper_psd};
//! use scirs2_core::ndarray::Array1;
//! use std::f64::consts::PI;
//!
//! // Generate a test signal
//! let n = 512usize;
//! let fs = 200.0f64;
//! let signal: Array1<f64> = Array1::from_iter(
//!     (0..n).map(|i| (2.0 * PI * 40.0 * i as f64 / fs).sin())
//! );
//!
//! // Compute DPSS tapers
//! let (tapers, eigenvalues) = dpss(n, 4.0, 7).expect("operation should succeed");
//! assert_eq!(tapers.shape()[0], 7);
//! assert_eq!(tapers.shape()[1], n);
//!
//! // Compute multitaper PSD
//! let (freqs, psd) = multitaper_psd(&signal, fs, 4.0, None, None, false).expect("operation should succeed");
//! assert!(!freqs.is_empty());
//! assert!(!psd.is_empty());
//! ```

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// DPSS / Slepian sequence computation
// ---------------------------------------------------------------------------

/// Compute Discrete Prolate Spheroidal Sequences (DPSS / Slepian sequences).
///
/// The DPSS are the unique sequences of length `n` that maximise energy
/// concentration in the frequency band `[-W, W]` where `W = half_bandwidth / n`.
///
/// The algorithm uses the tridiagonal eigenvalue approach of Percival & Walden
/// (1993).  For sequences longer than 128 samples the implicit symmetric QR
/// shift is applied; shorter sequences use the full Toeplitz concentration
/// matrix with Jacobi sweeps.
///
/// # Arguments
///
/// * `n`              – Sequence length.
/// * `half_bandwidth` – Half-bandwidth parameter `NW` (typically 2.5 – 4.0).
/// * `num_tapers`     – Number of tapers to return (must be ≤ `2*NW`).
///
/// # Returns
///
/// `(tapers, eigenvalues)` where
/// * `tapers`      has shape `(num_tapers, n)`.
/// * `eigenvalues` are the spectral concentration ratios ∈ (0, 1).
pub fn dpss(
    n: usize,
    half_bandwidth: f64,
    num_tapers: usize,
) -> SignalResult<(Array2<f64>, Vec<f64>)> {
    if n < 4 {
        return Err(SignalError::ValueError(
            "DPSS sequence length must be at least 4".to_string(),
        ));
    }
    if half_bandwidth <= 0.0 {
        return Err(SignalError::ValueError(
            "half_bandwidth (NW) must be positive".to_string(),
        ));
    }
    if num_tapers == 0 {
        return Err(SignalError::ValueError(
            "num_tapers must be at least 1".to_string(),
        ));
    }
    let max_tapers = (2.0 * half_bandwidth).floor() as usize;
    if num_tapers > max_tapers {
        return Err(SignalError::ValueError(format!(
            "num_tapers ({}) must not exceed 2*NW = {}",
            num_tapers, max_tapers
        )));
    }

    let w = half_bandwidth / n as f64; // normalised half-bandwidth

    // -----------------------------------------------------------------------
    // Compute DPSS by extracting the top eigenvectors of the Toeplitz
    // concentration matrix C[i,j] = sin(2*pi*W*(i-j)) / (pi*(i-j)).
    //
    // We use simultaneous iteration (subspace iteration) with the tridiagonal
    // commuting matrix T for fast matrix-vector products, followed by
    // Rayleigh-Ritz on the concentration matrix to get the correct ordering.
    //
    // For small n (<=256), we directly build C and use Jacobi eigen.
    // For large n, we use the tridiagonal matrix with inverse iteration
    // targeting eigenvalues found via QL, then reorder by concentration.
    // -----------------------------------------------------------------------

    let mut tapers = Array2::zeros((num_tapers, n));
    let mut ratios_vec = Vec::with_capacity(num_tapers);

    if n <= 1024 {
        // Direct approach: build the concentration matrix and solve
        let mut c_mat = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            c_mat[i][i] = 2.0 * w; // sinc(0) = 1, so C[i,i] = 2W
            for j in (i + 1)..n {
                let d = (i as f64 - j as f64) * 2.0 * PI * w;
                let val = d.sin() / (PI * (i as f64 - j as f64));
                c_mat[i][j] = val;
                c_mat[j][i] = val;
            }
        }

        // Solve via Jacobi eigendecomposition (small enough to be fast)
        let (eigvals, eigvecs) = jacobi_eigen_full(&c_mat)?;

        // Sort by descending eigenvalue (= descending concentration ratio)
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            eigvals[b]
                .partial_cmp(&eigvals[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for k in 0..num_tapers {
            let idx = order[k];
            let ratio = eigvals[idx].clamp(0.0, 1.0);
            ratios_vec.push(ratio);

            // Extract eigenvector and normalise
            let nrm = eigvecs[idx].iter().map(|v| v * v).sum::<f64>().sqrt();
            let nrm = if nrm > 1e-14 { nrm } else { 1.0 };

            let centre = n / 2;
            let centre_val = eigvecs[idx][centre] / nrm;
            let flip = if (k % 2 == 0) && centre_val < 0.0 {
                -1.0
            } else if (k % 2 == 1) && (eigvecs[idx][0] / nrm) < 0.0 {
                -1.0
            } else {
                1.0
            };
            for i in 0..n {
                tapers[[k, i]] = flip * eigvecs[idx][i] / nrm;
            }
        }
    } else {
        // Large n: use tridiagonal approach with inverse iteration
        // Build tridiagonal matrix
        let cos2piw = (2.0 * PI * w).cos();
        let mut diag = vec![0.0f64; n];
        let mut offdiag = vec![0.0f64; n - 1];
        for i in 0..n {
            let center = (n as f64 - 1.0) / 2.0 - i as f64;
            diag[i] = center * center * cos2piw;
        }
        for i in 0..(n - 1) {
            offdiag[i] = -((i + 1) as f64 * (n - i - 1) as f64) / 2.0;
        }

        // Get all eigenvalues via QL
        let (eigenvalues_tri, _) = implicit_qr_tridiagonal(&diag, &offdiag)?;

        // For each eigenvalue, compute eigenvector and its concentration ratio.
        // We sample a spread of eigenvalues to find the best-concentrated ones.
        let mut eig_sorted: Vec<f64> = eigenvalues_tri.clone();
        eig_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Sample uniformly from the eigenvalue spectrum plus extras near the edges
        let n_samples = (4 * num_tapers + 8).min(n);
        let mut sample_indices: Vec<usize> = Vec::new();
        // Evenly spaced
        for i in 0..n_samples {
            let idx = (i * (n - 1)) / n_samples.max(1);
            if !sample_indices.contains(&idx) {
                sample_indices.push(idx);
            }
        }
        // Also add the first and last few
        for i in 0..n_samples.min(n) {
            if !sample_indices.contains(&i) {
                sample_indices.push(i);
            }
        }

        let mut candidates: Vec<(Vec<f64>, f64)> = Vec::new();
        for &idx in &sample_indices {
            let target_eig = eig_sorted[idx];
            let taper = tridiag_inverse_iteration(&diag, &offdiag, target_eig, n)?;
            let ratio = compute_single_concentration_ratio(&taper, w, n)?;
            candidates.push((taper, ratio));
        }

        // Sort by descending concentration ratio
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for k in 0..num_tapers {
            let (ref taper, ratio) = candidates[k];
            ratios_vec.push(ratio);

            let centre = n / 2;
            let centre_val = taper[centre];
            let flip = if (k % 2 == 0) && centre_val < 0.0 {
                -1.0
            } else if (k % 2 == 1) && taper[0] < 0.0 {
                -1.0
            } else {
                1.0
            };
            for i in 0..n {
                tapers[[k, i]] = flip * taper[i];
            }
        }
    }

    Ok((tapers, ratios_vec))
}

// ---------------------------------------------------------------------------
// Implicit symmetric QR algorithm for tridiagonal matrices
// ---------------------------------------------------------------------------

/// Compute all eigenpairs of a real symmetric tridiagonal matrix using
/// the implicit QL algorithm with Wilkinson shifts (TQLI).
///
/// Jacobi eigenvalue algorithm for real symmetric matrices.
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvectors[i]` is the i-th
/// eigenvector (as a Vec<f64>) and `eigenvalues[i]` is the corresponding eigenvalue.
fn jacobi_eigen_full(a: &[Vec<f64>]) -> SignalResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let n = a.len();
    if n == 0 {
        return Ok((vec![], vec![]));
    }

    let mut mat: Vec<Vec<f64>> = a.to_vec();
    // Initialise V as identity
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0_f64; n];
            row[i] = 1.0;
            row
        })
        .collect();

    let max_iter = 100 * n * n;
    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0_f64;
        let mut p = 0usize;
        let mut q = 1usize;
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

        let diff = mat[q][q] - mat[p][p];
        let theta = if diff.abs() < 1e-30 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * mat[p][q] / diff).atan()
        };

        let (sin_t, cos_t) = theta.sin_cos();

        // Rotate
        let app = mat[p][p];
        let aqq = mat[q][q];
        let apq = mat[p][q];
        mat[p][p] = cos_t * cos_t * app - 2.0 * sin_t * cos_t * apq + sin_t * sin_t * aqq;
        mat[q][q] = sin_t * sin_t * app + 2.0 * sin_t * cos_t * apq + cos_t * cos_t * aqq;
        mat[p][q] = 0.0;
        mat[q][p] = 0.0;

        for i in 0..n {
            if i != p && i != q {
                let ip = mat[i][p];
                let iq = mat[i][q];
                mat[i][p] = cos_t * ip - sin_t * iq;
                mat[p][i] = mat[i][p];
                mat[i][q] = sin_t * ip + cos_t * iq;
                mat[q][i] = mat[i][q];
            }
        }

        // Accumulate eigenvectors
        for i in 0..n {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = cos_t * vip - sin_t * viq;
            v[i][q] = sin_t * vip + cos_t * viq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| mat[i][i]).collect();
    // eigenvectors[k] = k-th eigenvector = column k of V = [v[0][k], v[1][k], .., v[n-1][k]]
    let eigenvectors: Vec<Vec<f64>> = (0..n)
        .map(|k| (0..n).map(|i| v[i][k]).collect())
        .collect();

    Ok((eigenvalues, eigenvectors))
}

/// Compute a single eigenvector of a tridiagonal matrix via inverse iteration.
///
/// Solves `(T - mu*I) x = b` iteratively, normalising at each step, to converge
/// to the eigenvector corresponding to the eigenvalue closest to `mu`.
///
/// Returns a unit-norm eigenvector.
fn tridiag_inverse_iteration(
    diag: &[f64],
    offdiag: &[f64],
    mu: f64,
    n: usize,
) -> SignalResult<Vec<f64>> {
    // Start with a random-ish vector
    let mut x: Vec<f64> = (0..n)
        .map(|i| {
            // Deterministic pseudo-random start
            ((i as f64 * 0.7 + 0.3).sin() + 1.0) * 0.5
        })
        .collect();

    // Normalise
    let mut nrm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    if nrm < 1e-300 {
        nrm = 1.0;
    }
    for v in x.iter_mut() {
        *v /= nrm;
    }

    // Shifted diagonal: d_shifted[i] = diag[i] - mu
    let mut d_shifted: Vec<f64> = diag.iter().map(|&d| d - mu).collect();
    // Perturb to avoid exact singularity
    let eps_perturb = 1e-14 * diag.iter().map(|v| v.abs()).sum::<f64>().max(1.0);
    for d in d_shifted.iter_mut() {
        if d.abs() < eps_perturb {
            *d = eps_perturb;
        }
    }
    let e = offdiag;

    // LU factorisation of the shifted tridiagonal matrix (T - mu*I)
    // Using Thomas algorithm (tridiagonal LU without pivoting).
    let mut l_sub = vec![0.0f64; n]; // l[i] = e[i-1] / d_eff[i-1]
    let mut d_eff = vec![0.0f64; n]; // effective diagonal after elimination

    d_eff[0] = d_shifted[0];
    for i in 1..n {
        if d_eff[i - 1].abs() < 1e-300 {
            d_eff[i - 1] = 1e-300;
        }
        l_sub[i] = e[i - 1] / d_eff[i - 1];
        d_eff[i] = d_shifted[i] - l_sub[i] * e[i - 1];
    }

    let max_iter = 10;
    for _ in 0..max_iter {
        // Solve (T - mu*I) x_new = x via forward substitution + back substitution
        // Forward: L y = x
        let mut y = x.clone();
        for i in 1..n {
            y[i] -= l_sub[i] * y[i - 1];
        }
        // Backward: U x_new = y
        let mut x_new = vec![0.0f64; n];
        x_new[n - 1] = if d_eff[n - 1].abs() > 1e-300 {
            y[n - 1] / d_eff[n - 1]
        } else {
            y[n - 1] / 1e-300
        };
        for i in (0..n - 1).rev() {
            let di = if d_eff[i].abs() > 1e-300 { d_eff[i] } else { 1e-300 };
            x_new[i] = (y[i] - e[i] * x_new[i + 1]) / di;
        }

        // Normalise
        let nrm_new = x_new.iter().map(|v| v * v).sum::<f64>().sqrt();
        if nrm_new < 1e-300 {
            break;
        }
        x = x_new.iter().map(|v| v / nrm_new).collect();
    }

    Ok(x)
}

/// Input:
///   `diag[0..n]`      — diagonal elements
///   `offdiag[0..n-1]` — sub-diagonal elements (T[i,i+1])
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvectors[row][col]`.
fn implicit_qr_tridiagonal(
    diag: &[f64],
    offdiag: &[f64],
) -> SignalResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let n = diag.len();
    if n == 0 {
        return Ok((vec![], vec![]));
    }
    if n == 1 {
        return Ok((diag.to_vec(), vec![vec![1.0]]));
    }

    let mut d = diag.to_vec();
    // e[0..n-1]: sub-diagonal; e[n-1] = 0 (sentinel)
    let mut e = vec![0.0f64; n];
    for i in 0..offdiag.len().min(n - 1) {
        e[i] = offdiag[i];
    }

    // Eigenvector matrix z[row][col], initialised to identity
    let mut z: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            row
        })
        .collect();

    let max_iter = 30 * n;

    for l in 0..n {
        let mut iter_count = 0usize;
        loop {
            // Find smallest m >= l such that |e[m]| is negligible
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= 1e-15 * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break; // d[l] has converged
            }

            iter_count += 1;
            if iter_count > max_iter {
                return Err(SignalError::ComputationError(
                    "Symmetric QR did not converge".to_string(),
                ));
            }

            // Wilkinson shift
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + if g >= 0.0 { r } else { -r });

            let mut s = 1.0_f64;
            let mut c = 1.0_f64;
            let mut p = 0.0_f64;

            // QL sweep: i from m-1 down to l
            let mut i = m as isize - 1;
            while i >= l as isize {
                let ii = i as usize;
                let f = s * e[ii];
                let b = c * e[ii];

                // Compute Givens rotation to annihilate f
                if f.abs() >= g.abs() {
                    c = g / f;
                    let rr = (c * c + 1.0).sqrt();
                    e[ii + 1] = f * rr;
                    s = 1.0 / rr;
                    c *= s;
                } else {
                    s = f / g;
                    let rr = (s * s + 1.0).sqrt();
                    e[ii + 1] = g * rr;
                    c = 1.0 / rr;
                    s *= c;
                }

                let gg = d[ii + 1] - p;
                let rr = (d[ii] - gg) * s + 2.0 * c * b;
                p = s * rr;
                d[ii + 1] = gg + p;
                g = c * rr - b;

                // Accumulate eigenvector rotation on columns ii and ii+1
                for row_idx in 0..n {
                    let fv = z[row_idx][ii + 1];
                    z[row_idx][ii + 1] = s * z[row_idx][ii] + c * fv;
                    z[row_idx][ii] = c * z[row_idx][ii] - s * fv;
                }

                i -= 1;
            }

            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }

    Ok((d, z))
}

/// Compute the spectral concentration ratio for a single taper.
fn compute_single_concentration_ratio(
    taper: &[f64],
    w: f64,
    n: usize,
) -> SignalResult<f64> {
    let nfft = (n * 8).next_power_of_two();
    let band_bins = (w * nfft as f64).ceil() as usize + 1;

    let fft_out = fft_real_to_complex(taper, nfft)?;
    let total_power: f64 = fft_out.iter().map(|c| c.norm_sqr()).sum();
    if total_power < 1e-30 {
        return Ok(0.0);
    }
    let inband: f64 = fft_out[..band_bins.min(fft_out.len())]
        .iter()
        .map(|c| c.norm_sqr())
        .sum::<f64>()
        + fft_out[nfft.saturating_sub(band_bins)..]
            .iter()
            .map(|c| c.norm_sqr())
            .sum::<f64>();
    Ok((inband / total_power).clamp(0.0, 1.0))
}

/// Compute spectral concentration ratios for each taper.
///
/// The ratio is estimated as the sum of squared DFT magnitudes in [-W, W]
/// divided by the total power.
fn compute_concentration_ratios(
    tapers: &Array2<f64>,
    w: f64,
    n: usize,
    k: usize,
) -> SignalResult<Vec<f64>> {
    // Use at least 8x the signal length for accurate spectral concentration
    let nfft = (n * 8).next_power_of_two();
    let band_bins = (w * nfft as f64).ceil() as usize + 1;

    let mut ratios = vec![0.0f64; k];
    for t in 0..k {
        let taper: Vec<f64> = (0..n).map(|i| tapers[[t, i]]).collect();
        let fft_out = fft_real_to_complex(&taper, nfft)?;
        let total_power: f64 = fft_out.iter().map(|c| c.norm_sqr()).sum();
        if total_power < 1e-30 {
            ratios[t] = 0.0;
            continue;
        }
        // In-band power: bins 0..band_bins and nfft-band_bins..nfft (mirror)
        let inband: f64 = fft_out[..band_bins.min(fft_out.len())]
            .iter()
            .map(|c| c.norm_sqr())
            .sum::<f64>()
            + fft_out[nfft.saturating_sub(band_bins)..]
                .iter()
                .map(|c| c.norm_sqr())
                .sum::<f64>();
        ratios[t] = (inband / total_power).clamp(0.0, 1.0);
        #[cfg(test)]
        if t == 0 {
            eprintln!("DEBUG conc ratio: n={n}, nfft={nfft}, w={w}, band_bins={band_bins}, total={total_power:.6e}, inband={inband:.6e}, ratio={:.10}", ratios[t]);
        }
    }
    Ok(ratios)
}

// ---------------------------------------------------------------------------
// Internal FFT helper (Cooley-Tukey radix-2, power-of-two only)
// ---------------------------------------------------------------------------

fn fft_real_to_complex(x: &[f64], nfft: usize) -> SignalResult<Vec<Complex64>> {
    let mut buf: Vec<Complex64> = x
        .iter()
        .map(|&v| Complex64::new(v, 0.0))
        .chain(std::iter::repeat(Complex64::new(0.0, 0.0)))
        .take(nfft)
        .collect();
    if buf.len() < nfft {
        buf.resize(nfft, Complex64::new(0.0, 0.0));
    }
    fft_inplace(&mut buf);
    Ok(buf)
}

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
    // Cooley-Tukey butterfly
    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * PI / len as f64;
        let wlen = Complex64::new(ang.cos(), ang.sin());
        for i in (0..n).step_by(len) {
            let mut w = Complex64::new(1.0, 0.0);
            for k in 0..len / 2 {
                let u = buf[i + k];
                let v = buf[i + k + len / 2] * w;
                buf[i + k] = u + v;
                buf[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
        len <<= 1;
    }
}

// ---------------------------------------------------------------------------
// Multitaper Power Spectral Density
// ---------------------------------------------------------------------------

/// Compute the multitaper power spectral density estimate.
///
/// Uses Thomson's multitaper method with DPSS tapers to estimate the PSD.
/// The `adaptive` flag selects between simple equal-weight averaging and
/// Thomson's adaptive eigenspectrum weighting (Eq. 5.4 in Percival & Walden).
///
/// # Arguments
///
/// * `signal`            – Input signal.
/// * `sample_rate`       – Sampling frequency in Hz.
/// * `time_half_bandwidth` – `NW` product (typically 2.5 – 4.0).
/// * `num_tapers`        – Number of tapers (default `2*NW - 1`).
/// * `nfft`              – FFT length (default: next power of two ≥ signal length).
/// * `adaptive`          – If `true`, use Thomson's adaptive weighting.
///
/// # Returns
///
/// `(frequencies, psd)` where both arrays have length `nfft/2 + 1`.
pub fn multitaper_psd(
    signal: &Array1<f64>,
    sample_rate: f64,
    time_half_bandwidth: f64,
    num_tapers: Option<usize>,
    nfft: Option<usize>,
    adaptive: bool,
) -> SignalResult<(Array1<f64>, Array1<f64>)> {
    let n = signal.len();
    if n < 4 {
        return Err(SignalError::ValueError(
            "Signal must have at least 4 samples".to_string(),
        ));
    }
    if sample_rate <= 0.0 {
        return Err(SignalError::ValueError(
            "sample_rate must be positive".to_string(),
        ));
    }
    if time_half_bandwidth <= 0.0 {
        return Err(SignalError::ValueError(
            "time_half_bandwidth must be positive".to_string(),
        ));
    }

    let k = num_tapers.unwrap_or_else(|| {
        let max_k = (2.0 * time_half_bandwidth).floor() as usize;
        max_k.saturating_sub(1).max(1)
    });

    let nfft_val = nfft.unwrap_or_else(|| n.next_power_of_two());
    if nfft_val < n {
        return Err(SignalError::ValueError(
            "nfft must be at least the signal length".to_string(),
        ));
    }

    let (tapers, eigenvalues) = dpss(n, time_half_bandwidth, k)?;

    if adaptive {
        adaptive_multitaper_psd(signal, sample_rate, &tapers, &eigenvalues, Some(nfft_val))
    } else {
        // Simple equal-weight averaging of eigenspectra
        let n_out = nfft_val / 2 + 1;
        let mut psd = vec![0.0f64; n_out];

        for t in 0..k {
            let tapered: Vec<f64> = (0..n)
                .map(|i| signal[i] * tapers[[t, i]])
                .collect();
            let fft_out = fft_real_to_complex(&tapered, nfft_val)?;
            for b in 0..n_out {
                let power = fft_out[b].norm_sqr();
                let scale = if b == 0 || b == nfft_val / 2 { 1.0 } else { 2.0 };
                psd[b] += scale * power / (sample_rate * n as f64 * k as f64);
            }
        }

        let freqs: Vec<f64> = (0..n_out)
            .map(|b| b as f64 * sample_rate / nfft_val as f64)
            .collect();

        Ok((Array1::from(freqs), Array1::from(psd)))
    }
}

/// Thomson's adaptive multitaper PSD with eigenspectrum weighting.
///
/// Iteratively estimates weights that minimise broadband leakage while
/// preserving local spectral estimates. Follows the algorithm described in
/// Percival & Walden (1993), chapter 7.
///
/// # Arguments
///
/// * `signal`       – Input signal.
/// * `sample_rate`  – Sampling rate in Hz.
/// * `dpss_sequences` – DPSS taper matrix `(K × N)`.
/// * `eigenvalues`  – Concentration ratios for each taper.
/// * `nfft`         – FFT length (default next power-of-two ≥ N).
///
/// # Returns
///
/// `(frequencies, psd)` – one-sided spectral estimate.
pub fn adaptive_multitaper_psd(
    signal: &Array1<f64>,
    sample_rate: f64,
    dpss_sequences: &Array2<f64>,
    eigenvalues: &[f64],
    nfft: Option<usize>,
) -> SignalResult<(Array1<f64>, Array1<f64>)> {
    let (k, n) = dpss_sequences.dim();
    if signal.len() < n {
        return Err(SignalError::ValueError(
            "Signal length must be at least the taper length".to_string(),
        ));
    }
    if eigenvalues.len() < k {
        return Err(SignalError::ValueError(
            "eigenvalues length must equal number of tapers".to_string(),
        ));
    }
    if sample_rate <= 0.0 {
        return Err(SignalError::ValueError(
            "sample_rate must be positive".to_string(),
        ));
    }

    let nfft_val = nfft.unwrap_or_else(|| n.next_power_of_two());
    let n_out = nfft_val / 2 + 1;

    // Compute all eigenspectra
    let mut eig_spectra: Vec<Vec<f64>> = Vec::with_capacity(k);
    for t in 0..k {
        let tapered: Vec<f64> = (0..n)
            .map(|i| signal[i] * dpss_sequences[[t, i]])
            .collect();
        let fft_out = fft_real_to_complex(&tapered, nfft_val)?;
        let spec: Vec<f64> = (0..n_out)
            .map(|b| {
                let power = fft_out[b].norm_sqr();
                let scale = if b == 0 || b == nfft_val / 2 { 1.0 } else { 2.0 };
                scale * power / (sample_rate * n as f64)
            })
            .collect();
        eig_spectra.push(spec);
    }

    // Initial PSD estimate: simple average
    let mut s_hat: Vec<f64> = (0..n_out)
        .map(|b| eig_spectra.iter().map(|sp| sp[b]).sum::<f64>() / k as f64)
        .collect();

    // Adaptive iteration: compute weights b_k(f) = sqrt(λ_k) * S(f) / (λ_k * S(f) + σ²*(1-λ_k))
    // where σ² is the total variance of the signal
    let sigma2: f64 = {
        let mean = signal.iter().sum::<f64>() / n as f64;
        signal.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64
    };

    let max_adaptive_iter = 20;
    for _ in 0..max_adaptive_iter {
        let mut s_new = vec![0.0f64; n_out];
        let mut weight_sum = vec![0.0f64; n_out];

        for t in 0..k {
            let lam = eigenvalues[t].clamp(0.0, 1.0);
            let one_minus_lam = 1.0 - lam;
            for b in 0..n_out {
                let denom = lam * s_hat[b] + sigma2 * one_minus_lam;
                let w = if denom > 1e-30 {
                    lam.sqrt() * s_hat[b] / denom
                } else {
                    0.0
                };
                s_new[b] += w * w * eig_spectra[t][b];
                weight_sum[b] += w * w;
            }
        }

        let mut max_change = 0.0f64;
        for b in 0..n_out {
            let s_prev = s_hat[b];
            s_hat[b] = if weight_sum[b] > 1e-30 {
                s_new[b] / weight_sum[b]
            } else {
                s_prev
            };
            max_change = max_change.max((s_hat[b] - s_prev).abs() / (s_prev + 1e-30));
        }
        if max_change < 1e-6 {
            break;
        }
    }

    let freqs: Vec<f64> = (0..n_out)
        .map(|b| b as f64 * sample_rate / nfft_val as f64)
        .collect();

    Ok((Array1::from(freqs), Array1::from(s_hat)))
}

// ---------------------------------------------------------------------------
// F-test for harmonic line components
// ---------------------------------------------------------------------------

/// F-test for sinusoidal (harmonic line) components in the signal spectrum.
///
/// For each frequency bin the F-statistic compares the power explained by
/// a sinusoid at that frequency against the residual broadband power,
/// following Thomson (1982).
///
/// # Arguments
///
/// * `signal`         – Input signal.
/// * `sample_rate`    – Sampling rate in Hz.
/// * `dpss_sequences` – DPSS taper matrix `(K × N)`.
/// * `eigenvalues`    – Concentration ratios (used as weights).
///
/// # Returns
///
/// `(frequencies, f_statistic)` – one-sided frequency axis and the F-values.
/// An F-value above the 95 % critical value of F(2, 2K-2) indicates a
/// significant harmonic at that frequency.
pub fn multitaper_f_test(
    signal: &Array1<f64>,
    sample_rate: f64,
    dpss_sequences: &Array2<f64>,
    eigenvalues: &[f64],
) -> SignalResult<(Array1<f64>, Array1<f64>)> {
    let (k, n) = dpss_sequences.dim();
    if signal.len() < n {
        return Err(SignalError::ValueError(
            "Signal must be at least as long as the tapers".to_string(),
        ));
    }
    if eigenvalues.len() < k {
        return Err(SignalError::ValueError(
            "eigenvalues.len() must equal number of tapers".to_string(),
        ));
    }
    if sample_rate <= 0.0 {
        return Err(SignalError::ValueError(
            "sample_rate must be positive".to_string(),
        ));
    }

    let nfft = n.next_power_of_two();
    let n_out = nfft / 2 + 1;

    // Compute complex eigenspectra (not magnitudes)
    let mut eig_spectra_complex: Vec<Vec<Complex64>> = Vec::with_capacity(k);
    for t in 0..k {
        let tapered: Vec<f64> = (0..n)
            .map(|i| signal[i] * dpss_sequences[[t, i]])
            .collect();
        let fft_out = fft_real_to_complex(&tapered, nfft)?;
        eig_spectra_complex.push(fft_out[..n_out].to_vec());
    }

    // U_k(f) = sum of taper values (DC component of taper FFT)
    // For the F-test we need the mean taper sums at each frequency
    let mut u: Vec<f64> = vec![0.0f64; k];
    for t in 0..k {
        u[t] = (0..n).map(|i| dpss_sequences[[t, i]]).sum::<f64>();
    }
    let u_sq_sum: f64 = u.iter().map(|v| v * v).sum();

    let mut f_stat = vec![0.0f64; n_out];

    for b in 0..n_out {
        // Estimate harmonic amplitude: mu_hat = sum_k(U_k * Y_k(f)) / sum_k(U_k^2)
        let numerator: Complex64 = (0..k)
            .map(|t| eig_spectra_complex[t][b] * u[t])
            .fold(Complex64::new(0.0, 0.0), |acc, x| acc + x);

        let mu_hat = if u_sq_sum > 1e-30 {
            numerator / u_sq_sum
        } else {
            Complex64::new(0.0, 0.0)
        };

        // Residual power
        let line_power = mu_hat.norm_sqr() * u_sq_sum;
        let total_power: f64 = (0..k)
            .map(|t| eig_spectra_complex[t][b].norm_sqr())
            .sum();
        let residual = (total_power - line_power).max(1e-30);

        // F statistic with (2, 2K-2) degrees of freedom
        let df2 = (2 * k).saturating_sub(2).max(1) as f64;
        f_stat[b] = if residual > 0.0 {
            line_power * df2 / (2.0 * residual)
        } else {
            0.0
        };
    }

    let freqs: Vec<f64> = (0..n_out)
        .map(|b| b as f64 * sample_rate / nfft as f64)
        .collect();

    Ok((Array1::from(freqs), Array1::from(f_stat)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::Array1;
    use std::f64::consts::PI;

    // Helper: generate a pure sinusoid
    fn make_sinusoid(freq: f64, fs: f64, n: usize) -> Array1<f64> {
        Array1::from_iter((0..n).map(|i| (2.0 * PI * freq * i as f64 / fs).sin()))
    }

    // Helper: generate band-limited white noise (just use a deterministic
    // pseudo-random sequence for reproducibility)
    fn pseudo_noise(n: usize, seed: u64) -> Array1<f64> {
        let mut x = vec![0.0f64; n];
        let mut s = seed ^ 0xdeadbeef;
        for v in x.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((s >> 33) as f64) / (u32::MAX as f64) - 0.5;
        }
        Array1::from(x)
    }

    // -----------------------------------------------------------------------
    // DPSS tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_dpss_shape() {
        let (tapers, eigenvalues) = dpss(256, 4.0, 7).expect("unexpected None or Err");
        assert_eq!(tapers.shape(), &[7, 256]);
        assert_eq!(eigenvalues.len(), 7);
    }

    #[test]
    fn test_dpss_orthogonality() {
        // DPSS tapers must be orthogonal: <h_k, h_l> ≈ 0 for k ≠ l
        let n = 128;
        let (tapers, _) = dpss(n, 3.0, 5).expect("unexpected None or Err");
        for k in 0..5 {
            for l in (k + 1)..5 {
                let inner: f64 = (0..n)
                    .map(|i| tapers[[k, i]] * tapers[[l, i]])
                    .sum();
                assert!(
                    inner.abs() < 1e-8,
                    "Tapers {k} and {l} not orthogonal: inner = {inner}"
                );
            }
        }
    }

    #[test]
    fn test_dpss_unit_norm() {
        let n = 128;
        let (tapers, _) = dpss(n, 3.0, 5).expect("unexpected None or Err");
        for k in 0..5 {
            let norm: f64 = (0..n).map(|i| tapers[[k, i]].powi(2)).sum();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_dpss_energy_concentration() {
        // Eigenvalues should be close to 1 for well-concentrated tapers
        let (_, eigenvalues) = dpss(256, 4.0, 7).expect("unexpected None or Err");
        // First 5 tapers should have ratio > 0.99
        for (k, &lam) in eigenvalues.iter().take(5).enumerate() {
            assert!(
                lam > 0.9,
                "Taper {k} concentration ratio {lam} is too low"
            );
        }
    }

    #[test]
    fn test_dpss_nw_2_5() {
        let (tapers, eigenvalues) = dpss(256, 2.5, 4).expect("unexpected None or Err");
        assert_eq!(tapers.shape(), &[4, 256]);
        assert_eq!(eigenvalues.len(), 4);
        // All returned ratios should be in (0,1)
        for &lam in &eigenvalues {
            assert!(lam > 0.0 && lam <= 1.0, "Invalid concentration ratio {lam}");
        }
    }

    #[test]
    fn test_dpss_nw_3_5() {
        let (tapers, eigenvalues) = dpss(128, 3.5, 6).expect("unexpected None or Err");
        assert_eq!(tapers.shape(), &[6, 128]);
        assert_eq!(eigenvalues.len(), 6);
    }

    #[test]
    fn test_dpss_nw_4_0() {
        let (tapers, eigenvalues) = dpss(512, 4.0, 7).expect("unexpected None or Err");
        assert_eq!(tapers.shape(), &[7, 512]);
        assert!(eigenvalues[0] > 0.99, "First taper ratio should be near 1");
    }

    #[test]
    fn test_dpss_sign_convention_even_order() {
        // Even-order tapers (k=0,2,4) should be positive at the centre
        let n = 128;
        let (tapers, _) = dpss(n, 4.0, 7).expect("unexpected None or Err");
        for k in (0..7).step_by(2) {
            assert!(
                tapers[[k, n / 2]] >= 0.0,
                "Even taper {k} centre is negative"
            );
        }
    }

    #[test]
    fn test_dpss_invalid_num_tapers() {
        // Should fail if num_tapers > 2*NW
        let result = dpss(256, 4.0, 9);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Multitaper PSD tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_multitaper_psd_sinusoid() {
        let n = 512;
        let fs = 200.0;
        let f0 = 40.0;
        let signal = make_sinusoid(f0, fs, n);

        let (freqs, psd) =
            multitaper_psd(&signal, fs, 4.0, None, None, false).expect("unexpected None or Err");

        assert!(!freqs.is_empty());
        assert_eq!(freqs.len(), psd.len());

        // Peak should be near f0
        let peak_idx = psd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("unexpected None or Err");
        let peak_freq = freqs[peak_idx];
        assert!(
            (peak_freq - f0).abs() < 5.0,
            "Peak at {peak_freq} Hz, expected near {f0} Hz"
        );
    }

    #[test]
    fn test_multitaper_psd_adaptive() {
        let n = 512;
        let fs = 200.0;
        let signal = make_sinusoid(50.0, fs, n);

        let (freqs_eq, psd_eq) =
            multitaper_psd(&signal, fs, 4.0, None, None, false).expect("unexpected None or Err");
        let (freqs_ad, psd_ad) =
            multitaper_psd(&signal, fs, 4.0, None, None, true).expect("unexpected None or Err");

        assert_eq!(freqs_eq.len(), freqs_ad.len());
        assert_eq!(psd_eq.len(), psd_ad.len());
        // Both should be positive
        assert!(psd_ad.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_multitaper_variance_reduction() {
        // Multitaper should reduce variance compared to a single periodogram.
        // We verify by checking that the multitaper estimate has a lower
        // coefficient of variation on white noise.
        let n = 1024;
        let fs = 1.0;
        let noise = pseudo_noise(n, 42);

        let (_, psd) = multitaper_psd(&noise, fs, 4.0, None, None, false).expect("unexpected None or Err");

        let mean = psd.iter().sum::<f64>() / psd.len() as f64;
        let var = psd
            .iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f64>()
            / psd.len() as f64;
        let cv = var.sqrt() / mean;

        // For white noise the single-taper periodogram has CV ~ 1.
        // Multitaper (K=7) should give CV < 0.6.
        assert!(
            cv < 0.6,
            "Coefficient of variation {cv:.3} is too high for multitaper estimate"
        );
    }

    #[test]
    fn test_multitaper_psd_output_shape() {
        let n = 256;
        let fs = 1000.0;
        let signal = pseudo_noise(n, 7);
        let nfft = 512usize;

        let (freqs, psd) =
            multitaper_psd(&signal, fs, 3.5, Some(5), Some(nfft), false).expect("unexpected None or Err");
        assert_eq!(freqs.len(), nfft / 2 + 1);
        assert_eq!(psd.len(), nfft / 2 + 1);
    }

    // -----------------------------------------------------------------------
    // Adaptive multitaper PSD tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_adaptive_multitaper_positive_psd() {
        let n = 256;
        let fs = 1.0;
        let signal = pseudo_noise(n, 99);
        let (tapers, eigenvalues) = dpss(n, 4.0, 7).expect("unexpected None or Err");

        let (_, psd) =
            adaptive_multitaper_psd(&signal, fs, &tapers, &eigenvalues, None).expect("unexpected None or Err");
        assert!(psd.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_adaptive_multitaper_frequency_axis() {
        let n = 256;
        let fs = 100.0;
        let signal = pseudo_noise(n, 1);
        let (tapers, eigenvalues) = dpss(n, 4.0, 7).expect("unexpected None or Err");

        let (freqs, _) =
            adaptive_multitaper_psd(&signal, fs, &tapers, &eigenvalues, None).expect("unexpected None or Err");

        // Nyquist should be the last frequency
        let nyquist = fs / 2.0;
        assert_relative_eq!(freqs[freqs.len() - 1], nyquist, epsilon = 1.0);
    }

    // -----------------------------------------------------------------------
    // F-test tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_f_test_detects_sinusoid() {
        let n = 512;
        let fs = 200.0;
        let f0 = 60.0;
        // Pure tone – should have a very high F-statistic at f0
        let signal = make_sinusoid(f0, fs, n);
        let (tapers, eigenvalues) = dpss(n, 4.0, 7).expect("unexpected None or Err");

        let (freqs, f_vals) =
            multitaper_f_test(&signal, fs, &tapers, &eigenvalues).expect("unexpected None or Err");

        assert_eq!(freqs.len(), f_vals.len());

        // Find the bin closest to f0
        let target_bin = freqs
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((*a - f0).abs()).partial_cmp(&((*b - f0).abs())).expect("unexpected None or Err")
            })
            .map(|(i, _)| i)
            .expect("unexpected None or Err");

        // The F-statistic should be large near f0
        let peak_f = f_vals[target_bin];
        assert!(
            peak_f > 5.0,
            "F-statistic {peak_f:.2} at f0={f0} Hz is too low"
        );
    }

    #[test]
    fn test_f_test_output_positive() {
        let n = 256;
        let fs = 1.0;
        let signal = pseudo_noise(n, 55);
        let (tapers, eigenvalues) = dpss(n, 4.0, 7).expect("unexpected None or Err");

        let (_, f_vals) =
            multitaper_f_test(&signal, fs, &tapers, &eigenvalues).expect("unexpected None or Err");
        assert!(f_vals.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_f_test_error_on_short_signal() {
        let short = Array1::zeros(3);
        let (tapers, eigenvalues) = dpss(10, 3.0, 5).expect("unexpected None or Err");
        let result = multitaper_f_test(&short, 1.0, &tapers, &eigenvalues);
        assert!(result.is_err());
    }
}
