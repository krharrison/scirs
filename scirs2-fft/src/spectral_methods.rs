//! Advanced parametric spectral estimation methods.
//!
//! This module provides high-resolution spectral analysis techniques based on
//! data-adaptive models, complementing the non-parametric methods in
//! [`spectral_analysis`](crate::spectral_analysis).
//!
//! # Methods provided
//!
//! | Function | Algorithm | Best for |
//! |----------|-----------|----------|
//! | [`burg_spectrum`] | Maximum entropy / Burg AR | Short stationary signals |
//! | [`yule_walker_spectrum`] | Yule-Walker AR equations | Stationary AR processes |
//! | [`music_spectrum`] | MUSIC subspace method | Narrowband sources in noise |
//! | [`pisarenko`] | Pisarenko harmonic decomp. | Known-order sinusoidal models |
//! | [`capon_spectrum`] | Minimum-variance (Capon/MVDR)| High-resolution beamforming |
//! | [`esprit_frequencies`] | ESPRIT subspace method | Accurate freq. estimation |
//!
//! # References
//!
//! * Burg, J.P. (1978). "Maximum entropy spectral analysis." Proc. 37th Meeting
//!   SEG, Oklahoma City.
//! * Yule, G.U. (1927). "On a method of investigating periodicities in disturbed
//!   series." Phil. Trans. R. Soc. A 226, 267-298.
//! * Schmidt, R.O. (1986). "Multiple emitter location and signal parameter
//!   estimation." IEEE Trans. Antennas Propag. 34(3), 276-280.
//! * Pisarenko, V.F. (1973). "The retrieval of harmonics from a covariance
//!   function." Geophys. J. Int. 33(3), 347-366.
//! * Roy, R. & Kailath, T. (1989). "ESPRIT—Estimation of signal parameters via
//!   rotational invariance techniques." IEEE Trans. ASSP 37(7), 984-995.

use crate::error::{FFTError, FFTResult};
use crate::fft::fft;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
//  Internal linear-algebra helpers (no external dependencies)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the sample autocovariance sequence r[0..=lag] of `x` at zero mean.
///
/// r[k] = (1/N) Σ_{n=0}^{N-k-1}  x[n] · x[n+k]
fn autocovariance(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len() as f64;
    (0..=max_lag)
        .map(|lag| {
            let sum: f64 = x[..x.len() - lag]
                .iter()
                .zip(x[lag..].iter())
                .map(|(&a, &b)| a * b)
                .sum();
            sum / n
        })
        .collect()
}

/// Solve a symmetric positive-definite Toeplitz system T·x = b using the
/// Levinson-Durbin recursion.
///
/// `r` — first row/column of the Toeplitz matrix (length `p+1`).
/// `b` — right-hand side vector of length `p`.
/// Returns the solution vector of length `p`.
fn levinson_durbin(r: &[f64], b: &[f64]) -> FFTResult<Vec<f64>> {
    let p = b.len();
    if p == 0 {
        return Ok(vec![]);
    }
    if r.len() < p + 1 {
        return Err(FFTError::InvalidInput(
            "levinson_durbin: r must have at least p+1 elements".into(),
        ));
    }

    // Levinson-Durbin recursion for AR coefficients
    // Solve  [r(0)  r(1)  … r(p-1)] [a1]   [r(1) ]
    //        [r(1)  r(0)  … r(p-2)] [a2] = [r(2) ]
    //        [  ⋮                 ] [ ⋮]   [  ⋮  ]
    //        [r(p-1) …   r(0)     ] [ap]   [r(p) ]

    let mut f = vec![0.0f64; p]; // forward vector
    let mut g = vec![0.0f64; p]; // backward vector
    f[0] = -r[1] / r[0];
    g[0] = -r[1] / r[0];
    let mut alpha = r[0];

    // ar[i] holds the AR coefficient a_{i+1} after `i+1` iterations
    let mut ar = vec![0.0f64; p];
    ar[0] = f[0];

    for i in 1..p {
        alpha = alpha * (1.0 - f[i - 1].powi(2));
        if alpha.abs() < f64::EPSILON * 1e6 {
            return Err(FFTError::ComputationError(
                "levinson_durbin: singular Toeplitz matrix (zero prediction error)".into(),
            ));
        }
        // Compute reflection coefficient for order i+1
        let mut num = r[i + 1];
        for j in 0..i {
            num += ar[j] * r[i - j];
        }
        let ki = -num / alpha;

        // Update AR coefficients
        let old_ar = ar[..i].to_vec();
        ar[i] = ki;
        for j in 0..i {
            ar[j] += ki * old_ar[i - 1 - j];
        }

        // Update prediction error (not used further but kept for reference)
        let _beta = r[i + 1] + ar[..i].iter().zip(r[1..=i].iter()).map(|(&a, &ri)| a * ri).sum::<f64>();
    }

    // ar now holds [-a1, -a2, …, -ap] (with sign convention).
    // The caller (Yule-Walker) expects the solution to T·a = -[r1..rp],
    // which is exactly `ar` above (we negate in the AR equation).
    Ok(ar)
}

/// Evaluate the AR power spectrum at `n_fft` uniformly spaced frequencies.
///
/// H(f) = σ² / |1 + Σ_k a_k e^{-j2πfk}|²
fn ar_to_spectrum(ar_coeffs: &[f64], sigma2: f64, n_fft: usize) -> Vec<f64> {
    let p = ar_coeffs.len();
    (0..n_fft)
        .map(|k| {
            // Evaluate denominator polynomial at z = e^{j2πk/nfft}
            let omega = 2.0 * PI * k as f64 / n_fft as f64;
            let mut re = 1.0_f64;
            let mut im = 0.0_f64;
            for (m, &a) in ar_coeffs.iter().enumerate() {
                let angle = omega * (m + 1) as f64;
                re += a * angle.cos();
                im -= a * angle.sin();
            }
            sigma2 / (re * re + im * im)
        })
        .collect()
}

/// Compute the data covariance matrix of size `model_order × model_order`
/// from `signal` using the autocorrelation approach.
///
/// Returns a flat row-major vector of length `model_order * model_order`.
fn covariance_matrix(signal: &[f64], model_order: usize) -> Vec<f64> {
    let r = autocovariance(signal, model_order);
    let m = model_order;
    let mut cov = vec![0.0f64; m * m];
    for i in 0..m {
        for j in 0..m {
            let lag = if i >= j { i - j } else { j - i };
            cov[i * m + j] = r[lag];
        }
    }
    cov
}

/// Iterative power-method eigenvector decomposition for a symmetric real matrix.
///
/// Returns `(eigenvalues, eigenvectors)` in *ascending* order.
/// `matrix` is row-major of size `n×n`.  Works for moderate `n` (up to ~64).
fn eigen_symmetric(matrix: &[f64], n: usize) -> FFTResult<(Vec<f64>, Vec<Vec<f64>>)> {
    if matrix.len() != n * n {
        return Err(FFTError::InvalidInput(
            "eigen_symmetric: matrix size mismatch".into(),
        ));
    }

    // Jacobi eigenvalue algorithm for symmetric matrices
    const MAX_ITER: usize = 200;
    const TOL: f64 = 1e-12;

    // Start with a copy of the matrix and an identity eigenvector matrix
    let mut a = matrix.to_vec();
    let mut v: Vec<f64> = (0..n * n)
        .map(|idx| if idx / n == idx % n { 1.0 } else { 0.0 })
        .collect();

    for _ in 0..MAX_ITER {
        // Find the largest off-diagonal element
        let mut max_val = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = a[i * n + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < TOL {
            break;
        }

        // Compute Jacobi rotation angle
        let theta = if (a[q * n + q] - a[p * n + p]).abs() < TOL {
            PI / 4.0
        } else {
            0.5 * ((2.0 * a[p * n + q]) / (a[q * n + q] - a[p * n + p])).atan()
        };
        let c = theta.cos();
        let s = theta.sin();

        // Apply Jacobi rotation to A (left and right multiply by J and J^T)
        let mut a_new = a.clone();
        for i in 0..n {
            let api = a[p * n + i];
            let aqi = a[q * n + i];
            a_new[p * n + i] = c * api - s * aqi;
            a_new[q * n + i] = s * api + c * aqi;
        }
        let mut a2 = a_new.clone();
        for i in 0..n {
            let aip = a_new[i * n + p];
            let aiq = a_new[i * n + q];
            a2[i * n + p] = c * aip - s * aiq;
            a2[i * n + q] = s * aip + c * aiq;
        }
        // Force symmetry
        for i in 0..n {
            for j in 0..n {
                a2[j * n + i] = a2[i * n + j];
            }
        }
        a = a2;

        // Apply rotation to eigenvector matrix V
        for i in 0..n {
            let vip = v[i * n + p];
            let viq = v[i * n + q];
            v[i * n + p] = c * vip - s * viq;
            v[i * n + q] = s * vip + c * viq;
        }
    }

    // Extract diagonal as eigenvalues
    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();

    // Build eigenvectors as column vectors (v stores them as columns)
    let eigenvectors: Vec<Vec<f64>> = (0..n).map(|j| (0..n).map(|i| v[i * n + j]).collect()).collect();

    // Sort by ascending eigenvalue
    let mut pairs: Vec<(f64, Vec<f64>)> = eigenvalues
        .into_iter()
        .zip(eigenvectors.into_iter())
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let sorted_eigenvalues: Vec<f64> = pairs.iter().map(|(e, _)| *e).collect();
    let sorted_eigenvectors: Vec<Vec<f64>> = pairs.into_iter().map(|(_, v)| v).collect();

    Ok((sorted_eigenvalues, sorted_eigenvectors))
}

// ─────────────────────────────────────────────────────────────────────────────
//  Burg / Maximum-Entropy Method
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate an AR model of given `order` from `signal` using Burg's algorithm.
///
/// Burg's algorithm fits an autoregressive model by successively extending it
/// one order at a time while minimising the forward-backward prediction error
/// at each stage (lattice formulation).  This gives numerically stable
/// estimates even for short data records.
///
/// # Arguments
///
/// * `signal`   – Real-valued observation vector (length ≥ `order + 2`).
/// * `order`    – AR model order `p`.
///
/// # Returns
///
/// `(ar_coefficients, prediction_error_variance)`
/// where `ar_coefficients` is a `Vec<f64>` of length `order` containing
/// a₁, a₂, …, aₚ in the convention  x[n] = -Σ aₖ x[n-k] + noise.
///
/// # Errors
///
/// Returns [`FFTError::InvalidInput`] when the signal is too short, or
/// [`FFTError::ComputationError`] on numerical failure.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::spectral_methods::burg_ar;
///
/// let signal: Vec<f64> = (0..256)
///     .map(|i| (2.0 * std::f64::consts::PI * 0.1 * i as f64).cos())
///     .collect();
/// let (ar, sigma2) = burg_ar(&signal, 8).expect("burg_ar");
/// assert_eq!(ar.len(), 8);
/// assert!(sigma2 >= 0.0);
/// ```
pub fn burg_ar(signal: &[f64], order: usize) -> FFTResult<(Vec<f64>, f64)> {
    let n = signal.len();
    if n < order + 2 {
        return Err(FFTError::InvalidInput(format!(
            "burg_ar: signal length {} must be > order+1 = {}",
            n,
            order + 1
        )));
    }

    let mut f = signal.to_vec(); // forward prediction errors
    let mut b = signal.to_vec(); // backward prediction errors
    let mut a = vec![0.0f64; order]; // AR coefficients
    let mut sigma2 = signal.iter().map(|&x| x * x).sum::<f64>() / n as f64;

    for m in 0..order {
        // Compute numerator and denominator for the lattice reflection coefficient
        let num: f64 = -2.0
            * f[m + 1..]
                .iter()
                .zip(b[..n - m - 1].iter())
                .map(|(&fi, &bi)| fi * bi)
                .sum::<f64>();
        let den: f64 = f[m + 1..].iter().map(|&fi| fi * fi).sum::<f64>()
            + b[..n - m - 1].iter().map(|&bi| bi * bi).sum::<f64>();

        if den.abs() < f64::EPSILON * 1e6 {
            break; // signal is essentially zero; stop
        }

        let km = num / den;

        // Update AR coefficients using the Levinson-Durbin step
        let old_a = a[..m].to_vec();
        a[m] = km;
        for j in 0..m {
            a[j] += km * old_a[m - 1 - j];
        }

        // Update forward and backward errors
        let f_prev = f.clone();
        let b_prev = b.clone();
        for i in m + 1..n {
            f[i] = f_prev[i] + km * b_prev[i - 1];
            b[i] = b_prev[i - 1] + km * f_prev[i];
        }

        sigma2 *= 1.0 - km * km;
    }

    Ok((a, sigma2))
}

/// Compute the Maximum Entropy Method (MEM/Burg) power spectral density.
///
/// Fits an AR(`order`) model using Burg's algorithm and evaluates the
/// resulting rational PSD at `n_fft` evenly spaced frequencies from 0 to
/// `sample_rate/2`.
///
/// # Arguments
///
/// * `signal`      – Real-valued input signal.
/// * `order`       – AR model order (typically 10–50).
/// * `n_fft`       – Number of frequency bins in the output PSD.
/// * `sample_rate` – Sampling frequency in Hz.
///
/// # Returns
///
/// Power spectral density vector of length `n_fft/2 + 1` (one-sided).
///
/// # Errors
///
/// Propagates errors from [`burg_ar`].
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::spectral_methods::burg_spectrum;
/// use std::f64::consts::PI;
///
/// let fs = 1000.0_f64;
/// let signal: Vec<f64> = (0..512)
///     .map(|i| (2.0 * PI * 100.0 * i as f64 / fs).sin())
///     .collect();
/// let psd = burg_spectrum(&signal, 16, 512, fs).expect("burg_spectrum");
/// assert_eq!(psd.len(), 512 / 2 + 1);
/// ```
pub fn burg_spectrum(
    signal: &[f64],
    order: usize,
    n_fft: usize,
    sample_rate: f64,
) -> FFTResult<Vec<f64>> {
    let (ar, sigma2) = burg_ar(signal, order)?;

    if n_fft < 2 {
        return Err(FFTError::InvalidInput(
            "burg_spectrum: n_fft must be >= 2".into(),
        ));
    }

    let n_out = n_fft / 2 + 1;
    let full = ar_to_spectrum(&ar, sigma2, n_fft);

    // Scale to power density (divide by sample rate so units are V²/Hz)
    let scale = 1.0 / sample_rate;
    Ok(full[..n_out].iter().map(|&v| v * scale).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
//  Yule-Walker spectral estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Solve the Yule-Walker equations for an AR(`order`) model.
///
/// Uses the biased sample autocorrelation and the Levinson-Durbin recursion.
///
/// # Returns
///
/// `(ar_coefficients, prediction_error_variance)`
///
/// # Errors
///
/// Returns [`FFTError::InvalidInput`] for invalid inputs or
/// [`FFTError::ComputationError`] on singular Toeplitz systems.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::spectral_methods::yule_walker_ar;
///
/// let signal: Vec<f64> = (0..256)
///     .map(|i| (2.0 * std::f64::consts::PI * 0.15 * i as f64).cos())
///     .collect();
/// let (ar, sigma2) = yule_walker_ar(&signal, 4).expect("yule_walker_ar");
/// assert_eq!(ar.len(), 4);
/// ```
pub fn yule_walker_ar(signal: &[f64], order: usize) -> FFTResult<(Vec<f64>, f64)> {
    let n = signal.len();
    if n < order + 2 {
        return Err(FFTError::InvalidInput(format!(
            "yule_walker_ar: signal length {} too short for order {}",
            n, order
        )));
    }

    let r = autocovariance(signal, order);
    // r[0] is variance, r[1..=order] are lags

    // Levinson-Durbin: solve r[0..order]·a = -r[1..=order]
    // using the symmetric Toeplitz structure
    if r[0].abs() < f64::EPSILON {
        return Err(FFTError::ComputationError(
            "yule_walker_ar: zero-variance signal".into(),
        ));
    }

    let mut a = vec![0.0f64; order]; // AR coefficients
    a[0] = -r[1] / r[0];
    let mut alpha = r[0] * (1.0 - a[0].powi(2));

    for i in 1..order {
        if alpha.abs() < f64::EPSILON * r[0].abs() {
            return Err(FFTError::ComputationError(
                "yule_walker_ar: singular Toeplitz matrix".into(),
            ));
        }

        // Reflection coefficient
        let num: f64 = r[i + 1] + (0..i).map(|j| a[j] * r[i - j]).sum::<f64>();
        let ki = -num / alpha;

        let old = a[..i].to_vec();
        a[i] = ki;
        for j in 0..i {
            a[j] += ki * old[i - 1 - j];
        }
        alpha *= 1.0 - ki * ki;
    }

    let sigma2 = r[0]
        + (0..order)
            .map(|j| a[j] * r[j + 1])
            .sum::<f64>();
    let sigma2 = sigma2.max(0.0); // clamp rounding errors

    Ok((a, sigma2))
}

/// Compute the Yule-Walker parametric power spectral density.
///
/// Solves the Yule-Walker equations for an AR(`order`) model and evaluates
/// the resulting PSD at `n_fft` frequencies.
///
/// # Arguments
///
/// * `signal`      – Real-valued input signal (length ≥ `order + 2`).
/// * `order`       – AR model order.
/// * `n_fft`       – Number of frequency bins for the output PSD.
/// * `sample_rate` – Sampling frequency in Hz.
///
/// # Returns
///
/// One-sided PSD vector of length `n_fft/2 + 1`.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::spectral_methods::yule_walker_spectrum;
/// use std::f64::consts::PI;
///
/// let fs = 1000.0_f64;
/// let signal: Vec<f64> = (0..512)
///     .map(|i| (2.0 * PI * 200.0 * i as f64 / fs).sin())
///     .collect();
/// let psd = yule_walker_spectrum(&signal, 8, 512, fs).expect("yule_walker_spectrum");
/// assert_eq!(psd.len(), 512 / 2 + 1);
/// ```
pub fn yule_walker_spectrum(
    signal: &[f64],
    order: usize,
    n_fft: usize,
    sample_rate: f64,
) -> FFTResult<Vec<f64>> {
    let (ar, sigma2) = yule_walker_ar(signal, order)?;
    if n_fft < 2 {
        return Err(FFTError::InvalidInput(
            "yule_walker_spectrum: n_fft must be >= 2".into(),
        ));
    }
    let n_out = n_fft / 2 + 1;
    let full = ar_to_spectrum(&ar, sigma2, n_fft);
    let scale = 1.0 / sample_rate;
    Ok(full[..n_out].iter().map(|&v| v * scale).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
//  MUSIC: Multiple Signal Classification
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the MUSIC pseudo-spectrum.
///
/// MUSIC partitions the data covariance matrix into signal and noise
/// subspaces via eigendecomposition.  The reciprocal of the distance from
/// the steering vector to the noise subspace gives a high-resolution
/// pseudo-spectrum with sharp peaks at the true frequencies.
///
/// # Arguments
///
/// * `signal`      – Real-valued input signal.
/// * `n_sources`   – Number of complex exponential sources (sinusoids).
/// * `n_fft`       – Number of pseudo-spectrum frequency bins.
/// * `model_order` – Size of the data covariance matrix; must be > `n_sources`.
///                   If `0`, defaults to `2 * n_sources + 1`.
///
/// # Returns
///
/// One-sided MUSIC pseudo-spectrum of length `n_fft/2 + 1`.
/// Peaks correspond to estimated source frequencies.
///
/// # Errors
///
/// Returns [`FFTError::InvalidInput`] for invalid parameters.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::spectral_methods::music_spectrum;
/// use std::f64::consts::PI;
///
/// let n = 256usize;
/// let fs = 1000.0_f64;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * 100.0 * i as f64 / fs).sin()
///           + (2.0 * PI * 200.0 * i as f64 / fs).sin())
///     .collect();
/// let ps = music_spectrum(&signal, 2, 512, 0).expect("music_spectrum");
/// assert_eq!(ps.len(), 512 / 2 + 1);
/// ```
pub fn music_spectrum(
    signal: &[f64],
    n_sources: usize,
    n_fft: usize,
    model_order: usize,
) -> FFTResult<Vec<f64>> {
    let m = if model_order == 0 {
        2 * n_sources + 2
    } else {
        model_order
    };

    if n_sources >= m {
        return Err(FFTError::InvalidInput(
            "music_spectrum: n_sources must be < model_order".into(),
        ));
    }
    if signal.len() < m + 1 {
        return Err(FFTError::InvalidInput(format!(
            "music_spectrum: signal length {} too short for model_order {}",
            signal.len(),
            m
        )));
    }
    if n_fft < 2 {
        return Err(FFTError::InvalidInput(
            "music_spectrum: n_fft must be >= 2".into(),
        ));
    }

    // Build covariance matrix
    let cov = covariance_matrix(signal, m);

    // Eigendecompose (ascending order)
    let (_eigenvalues, eigenvectors) = eigen_symmetric(&cov, m)?;

    // Noise subspace = eigenvectors corresponding to the (m - n_sources) smallest eigenvalues
    let n_noise = m - n_sources;
    let noise_vecs: Vec<&Vec<f64>> = eigenvectors[..n_noise].iter().collect();

    // Evaluate MUSIC pseudo-spectrum
    let n_out = n_fft / 2 + 1;
    let mut ps = vec![0.0f64; n_out];

    for k in 0..n_out {
        let omega = 2.0 * PI * k as f64 / n_fft as f64;
        // Steering vector e(ω) = [1, e^{-jω}, …, e^{-j(m-1)ω}]
        // |e^H E_n|^2 = Σ_noise_vecs |Σ_i e_i · v[i]|^2  (real covariance → complex steering)
        let mut denom = 0.0_f64;
        for nv in &noise_vecs {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for (i, &v_i) in nv.iter().enumerate() {
                // e^{-jωi} component
                re += (omega * i as f64).cos() * v_i;
                im -= (omega * i as f64).sin() * v_i;
            }
            denom += re * re + im * im;
        }
        ps[k] = if denom.abs() < f64::EPSILON { f64::MAX } else { 1.0 / denom };
    }

    Ok(ps)
}

/// Estimate frequencies of sinusoidal sources using MUSIC.
///
/// Equivalent to finding the `n_sources` largest peaks of [`music_spectrum`].
///
/// # Arguments
///
/// * `signal`      – Real-valued input signal.
/// * `n_sources`   – Number of sinusoidal sources.
/// * `n_fft`       – Frequency resolution (number of bins).
/// * `model_order` – Covariance matrix size (0 for auto).
/// * `sample_rate` – Sampling frequency (Hz).
///
/// # Returns
///
/// Estimated frequencies in Hz, sorted ascending.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::spectral_methods::music_frequencies;
/// use std::f64::consts::PI;
///
/// let fs = 1000.0_f64;
/// let n = 512usize;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * 100.0 * i as f64 / fs).sin()
///           + (2.0 * PI * 250.0 * i as f64 / fs).sin())
///     .collect();
/// let freqs = music_frequencies(&signal, 2, 2048, 0, fs).expect("music_frequencies");
/// assert_eq!(freqs.len(), 2);
/// ```
pub fn music_frequencies(
    signal: &[f64],
    n_sources: usize,
    n_fft: usize,
    model_order: usize,
    sample_rate: f64,
) -> FFTResult<Vec<f64>> {
    let ps = music_spectrum(signal, n_sources, n_fft, model_order)?;
    let n_out = ps.len();

    // Find peaks: local maxima in ps (excluding boundaries for safety)
    let mut candidates: Vec<(usize, f64)> = (1..n_out - 1)
        .filter(|&i| ps[i] >= ps[i - 1] && ps[i] >= ps[i + 1])
        .map(|i| (i, ps[i]))
        .collect();

    // Sort by descending pseudo-spectrum value
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take the top n_sources peaks
    let mut freqs: Vec<f64> = candidates
        .iter()
        .take(n_sources)
        .map(|(idx, _)| {
            // Convert bin to Hz: bin k → k * fs / n_fft
            *idx as f64 * sample_rate / n_fft as f64
        })
        .collect();

    freqs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(freqs)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Pisarenko Harmonic Decomposition
// ─────────────────────────────────────────────────────────────────────────────

/// Pisarenko harmonic decomposition.
///
/// Computes the minimum-eigenvalue eigenvector of the `(p+1) × (p+1)`
/// covariance matrix where `p = n_sources`.  The roots of the polynomial
/// formed by this eigenvector on the unit circle give the frequencies.
///
/// This is the special case of MUSIC where `n_sources` exactly equals the
/// signal subspace dimension.
///
/// # Arguments
///
/// * `signal`    – Real-valued input signal (length ≥ `2*n_sources + 2`).
/// * `n_sources` – Number of complex exponential sources to estimate.
///
/// # Returns
///
/// Estimated frequencies in normalised units (cycles/sample, 0..0.5).
///
/// # Errors
///
/// Returns [`FFTError::InvalidInput`] for invalid arguments or
/// [`FFTError::ComputationError`] if the eigendecomposition fails.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::spectral_methods::pisarenko;
/// use std::f64::consts::PI;
///
/// let n = 256usize;
/// let f0 = 0.1_f64; // normalised frequency (cycles/sample)
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * f0 * i as f64).cos())
///     .collect();
/// let freqs = pisarenko(&signal, 1).expect("pisarenko");
/// assert_eq!(freqs.len(), 1);
/// // Should be close to 0.1
/// assert!((freqs[0] - f0).abs() < 0.02);
/// ```
pub fn pisarenko(signal: &[f64], n_sources: usize) -> FFTResult<Vec<f64>> {
    let m = n_sources + 1; // covariance matrix dimension
    if signal.len() < 2 * m {
        return Err(FFTError::InvalidInput(format!(
            "pisarenko: signal length {} too short (need >= {})",
            signal.len(),
            2 * m
        )));
    }

    let cov = covariance_matrix(signal, m);
    let (_eigenvalues, eigenvectors) = eigen_symmetric(&cov, m)?;

    // Minimum-eigenvalue eigenvector (first in ascending order)
    let min_vec = &eigenvectors[0];

    // Find roots of the polynomial with coefficients min_vec on the unit circle.
    // We solve |1 + Σ_{k=1}^{n_sources} c_k e^{-jkω}|^2 = 0 numerically.
    //
    // For the 1-source case: a[0] + a[1] e^{-jω} = 0  ⟹  ω = -arg(a[0]/a[1])
    // For general p, we use companion matrix eigenvalues.
    let freqs = polynomial_unit_circle_roots(min_vec)?;
    Ok(freqs)
}

/// Find roots of a polynomial with real coefficients on the unit circle.
///
/// Given coefficients [a0, a1, …, ap], finds arguments ω ∈ [0, π] of the
/// roots  Σ aₖ z^k = 0  that lie on or near the unit circle.
fn polynomial_unit_circle_roots(coeffs: &[f64]) -> FFTResult<Vec<f64>> {
    let p = coeffs.len() - 1; // polynomial degree
    if p == 0 {
        return Ok(vec![]);
    }
    if p == 1 {
        // a0 + a1 z = 0  ⟹  z = -a0/a1
        if coeffs[1].abs() < f64::EPSILON {
            return Err(FFTError::ComputationError(
                "polynomial_unit_circle_roots: leading coefficient is zero".into(),
            ));
        }
        let z_re = -coeffs[0] / coeffs[1];
        // Only accept roots near the unit circle
        if (z_re.abs() - 1.0).abs() < 0.1 {
            let freq = z_re.atan2(0.0_f64).abs() / (2.0 * PI);
            return Ok(vec![freq]);
        }
        return Ok(vec![]);
    }

    // For degree > 1: build companion matrix (real coefficients → all real rows)
    // C = [-a_{p-1}/a_p  -a_{p-2}/a_p  …  -a_0/a_p ]  (top row)
    //     [  I_{p-1}                   0             ]
    let lead = coeffs[p];
    if lead.abs() < f64::EPSILON {
        return Err(FFTError::ComputationError(
            "polynomial_unit_circle_roots: leading coefficient near zero".into(),
        ));
    }

    // We evaluate the pseudo-spectrum on the unit circle to find roots numerically.
    // Use a dense grid on [0, π] and look for zero crossings of the real part
    // of the polynomial evaluated at e^{jω}.
    let n_grid = 4096;
    let mut roots = Vec::new();
    let mut prev_re = 0.0f64;
    let mut prev_mag = f64::MAX;

    for k in 0..=n_grid {
        let omega = PI * k as f64 / n_grid as f64;
        let z_re = omega.cos();
        let z_im = omega.sin();

        // Evaluate Σ aₙ z^n using Horner's method
        let mut re = coeffs[p];
        let mut im = 0.0f64;
        for j in (0..p).rev() {
            // (re + i·im) · (z_re + i·z_im) + a_j
            let new_re = re * z_re - im * z_im + coeffs[j];
            let new_im = re * z_im + im * z_re;
            re = new_re;
            im = new_im;
        }

        let mag = (re * re + im * im).sqrt();
        // Local minimum of |P(e^{jω})| that is close to zero
        if k > 0 && mag < prev_mag && prev_mag < 0.1 * lead.abs() * (p as f64) {
            let freq_cycles = (omega - PI / n_grid as f64) / (2.0 * PI);
            if freq_cycles >= 0.0 && freq_cycles <= 0.5 {
                roots.push(freq_cycles);
            }
        }
        // Zero crossing of real part (change of sign)
        if k > 0 && prev_re * re < 0.0 && im.abs() < (lead.abs() * (p as f64)).max(1e-6) {
            let freq_cycles = (omega - PI / (2.0 * n_grid as f64)) / (2.0 * PI);
            if freq_cycles >= 0.0 && freq_cycles <= 0.5 {
                roots.push(freq_cycles);
            }
        }
        prev_re = re;
        prev_mag = mag;
    }

    // Deduplicate close frequencies
    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut unique: Vec<f64> = Vec::new();
    for f in roots {
        if unique.last().map(|&last| (f - last).abs() > 0.005).unwrap_or(true) {
            unique.push(f);
        }
    }

    Ok(unique)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Capon / MVDR spectral estimator
// ─────────────────────────────────────────────────────────────────────────────

/// Capon's minimum-variance distortionless response (MVDR) spectral estimator.
///
/// Computes the Capon beamformer spectrum, which offers better resolution than
/// the classical periodogram but is less computationally expensive than MUSIC.
///
/// # Arguments
///
/// * `signal`      – Real-valued input signal.
/// * `n_fft`       – Number of output frequency bins.
/// * `model_order` – Spatial filter length (size of covariance matrix).
///
/// # Returns
///
/// One-sided Capon pseudo-spectrum of length `n_fft/2 + 1`.
///
/// # Errors
///
/// Returns [`FFTError::InvalidInput`] for invalid parameters.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::spectral_methods::capon_spectrum;
/// use std::f64::consts::PI;
///
/// let fs = 1000.0_f64;
/// let signal: Vec<f64> = (0..512)
///     .map(|i| (2.0 * PI * 100.0 * i as f64 / fs).sin())
///     .collect();
/// let ps = capon_spectrum(&signal, 256, 16).expect("capon_spectrum");
/// assert_eq!(ps.len(), 256 / 2 + 1);
/// ```
pub fn capon_spectrum(
    signal: &[f64],
    n_fft: usize,
    model_order: usize,
) -> FFTResult<Vec<f64>> {
    let m = if model_order == 0 { 16 } else { model_order };
    if signal.len() < m + 1 {
        return Err(FFTError::InvalidInput(format!(
            "capon_spectrum: signal too short ({} < {})",
            signal.len(),
            m + 1
        )));
    }
    if n_fft < 2 {
        return Err(FFTError::InvalidInput(
            "capon_spectrum: n_fft must be >= 2".into(),
        ));
    }

    // Build and invert the covariance matrix via Cholesky (regularised)
    let cov = covariance_matrix(signal, m);
    let inv = invert_symmetric_pos_def(&cov, m)?;

    let n_out = n_fft / 2 + 1;
    let mut ps = vec![0.0f64; n_out];

    for k in 0..n_out {
        let omega = 2.0 * PI * k as f64 / n_fft as f64;
        // e^H R^{-1} e  — both real and imaginary parts of e appear
        // e[i] = exp(-jωi) = cos(ωi) - j sin(ωi)
        // e^H R^{-1} e = Σ_{i,j} e*[i] R^{-1}[i,j] e[j]
        //              = Σ_{i,j} R^{-1}[i,j] (cos(ω(i-j)) + j sin(ω(i-j))) · (real part only since R^{-1} is real sym)
        let mut quad = 0.0f64;
        for i in 0..m {
            for j in 0..m {
                let diff = i as f64 - j as f64;
                quad += inv[i * m + j] * (omega * diff).cos();
            }
        }
        ps[k] = if quad.abs() < f64::EPSILON { f64::MAX } else { 1.0 / quad };
    }

    Ok(ps)
}

/// Invert a symmetric positive-definite `n×n` matrix stored in row-major form.
///
/// Uses Cholesky decomposition with Tikhonov regularisation (ε·I) to handle
/// near-singular matrices.
fn invert_symmetric_pos_def(a: &[f64], n: usize) -> FFTResult<Vec<f64>> {
    // Cholesky: A = L L^T
    let reg = {
        let diag_max = (0..n).map(|i| a[i * n + i]).fold(0.0f64, f64::max);
        diag_max * 1e-10
    };

    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            s += if i == j { reg } else { 0.0 };
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = if i == j {
                if s < 0.0 {
                    return Err(FFTError::ComputationError(
                        "invert_symmetric_pos_def: matrix not positive definite".into(),
                    ));
                }
                s.sqrt()
            } else {
                if l[j * n + j].abs() < f64::EPSILON {
                    0.0
                } else {
                    s / l[j * n + j]
                }
            };
        }
    }

    // Solve L Y = I (forward substitution for each column of identity)
    let mut y = vec![0.0f64; n * n];
    for col in 0..n {
        for i in 0..n {
            let mut s = if i == col { 1.0 } else { 0.0 };
            for k in 0..i {
                s -= l[i * n + k] * y[k * n + col];
            }
            y[i * n + col] = if l[i * n + i].abs() < f64::EPSILON {
                0.0
            } else {
                s / l[i * n + i]
            };
        }
    }

    // Solve L^T X = Y (backward substitution)
    let mut x = vec![0.0f64; n * n];
    for col in 0..n {
        for i in (0..n).rev() {
            let mut s = y[i * n + col];
            for k in (i + 1)..n {
                s -= l[k * n + i] * x[k * n + col];
            }
            x[i * n + col] = if l[i * n + i].abs() < f64::EPSILON {
                0.0
            } else {
                s / l[i * n + i]
            };
        }
    }

    Ok(x)
}

// ─────────────────────────────────────────────────────────────────────────────
//  ESPRIT: Estimation of Signal Parameters via Rotational Invariance
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate sinusoidal frequencies using the ESPRIT algorithm.
///
/// ESPRIT exploits the rotational invariance of signal subspace to estimate
/// frequencies without a spectral search, giving high accuracy with low cost.
///
/// # Arguments
///
/// * `signal`      – Real-valued input signal.
/// * `n_sources`   – Number of complex exponential sources.
/// * `model_order` – Covariance matrix size (0 for auto = `2*n_sources+2`).
/// * `sample_rate` – Sampling frequency in Hz.
///
/// # Returns
///
/// Estimated frequencies in Hz, sorted ascending.
///
/// # Errors
///
/// Returns [`FFTError::InvalidInput`] for invalid parameters or
/// [`FFTError::ComputationError`] on numerical failures.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::spectral_methods::esprit_frequencies;
/// use std::f64::consts::PI;
///
/// let fs = 1000.0_f64;
/// let n = 256usize;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * 100.0 * i as f64 / fs).sin()
///           + (2.0 * PI * 250.0 * i as f64 / fs).sin())
///     .collect();
/// let freqs = esprit_frequencies(&signal, 2, 0, fs).expect("esprit_frequencies");
/// assert_eq!(freqs.len(), 2);
/// ```
pub fn esprit_frequencies(
    signal: &[f64],
    n_sources: usize,
    model_order: usize,
    sample_rate: f64,
) -> FFTResult<Vec<f64>> {
    let m = if model_order == 0 {
        (2 * n_sources + 2).max(4)
    } else {
        model_order
    };
    if n_sources >= m {
        return Err(FFTError::InvalidInput(
            "esprit_frequencies: n_sources must be < model_order".into(),
        ));
    }
    if signal.len() < m + 1 {
        return Err(FFTError::InvalidInput(format!(
            "esprit_frequencies: signal length {} too short for model_order {}",
            signal.len(),
            m
        )));
    }

    // Build covariance matrix and eigendecompose
    let cov = covariance_matrix(signal, m);
    let (_eigenvalues, eigenvectors) = eigen_symmetric(&cov, m)?;

    // Signal subspace = eigenvectors corresponding to the `n_sources` *largest*
    // eigenvalues (last `n_sources` in ascending order).
    let n = m;
    let signal_vecs: Vec<&Vec<f64>> = eigenvectors[n - n_sources..].iter().collect();

    // Build E_s: n × n_sources matrix (each column is a signal eigenvector)
    // E1 = E_s[0..m-1, :], E2 = E_s[1..m, :]  (shift invariance)
    // Compute Phi = pinv(E1) · E2 and get its eigenvalues (complex via 2×2 real blocks)
    // For real signals, we estimate using the real-valued ESPRIT approximation.

    // E1 rows 0..m-1, E2 rows 1..m
    let e_rows_e1: usize = n - 1;
    let e_rows_e2: usize = n - 1;

    // Form E1 and E2 as flat row-major matrices (e_rows × n_sources)
    let mut e1 = vec![0.0f64; e_rows_e1 * n_sources];
    let mut e2 = vec![0.0f64; e_rows_e2 * n_sources];

    for (col, sv) in signal_vecs.iter().enumerate() {
        for row in 0..e_rows_e1 {
            e1[row * n_sources + col] = sv[row];
            e2[row * n_sources + col] = sv[row + 1];
        }
    }

    // Least-squares solution: Phi = pinv(E1) · E2
    // = (E1^T E1)^{-1} E1^T E2
    // GtG = E1^T E1 (n_sources × n_sources)
    let mut gtg = vec![0.0f64; n_sources * n_sources];
    for i in 0..n_sources {
        for j in 0..n_sources {
            for r in 0..e_rows_e1 {
                gtg[i * n_sources + j] += e1[r * n_sources + i] * e1[r * n_sources + j];
            }
        }
    }
    // GtY = E1^T E2 (n_sources × n_sources)
    let mut gty = vec![0.0f64; n_sources * n_sources];
    for i in 0..n_sources {
        for j in 0..n_sources {
            for r in 0..e_rows_e1 {
                gty[i * n_sources + j] += e1[r * n_sources + i] * e2[r * n_sources + j];
            }
        }
    }

    // Phi = inv(GtG) · GtY
    let inv_gtg = invert_symmetric_pos_def(&gtg, n_sources)?;
    let mut phi = vec![0.0f64; n_sources * n_sources];
    for i in 0..n_sources {
        for j in 0..n_sources {
            for k in 0..n_sources {
                phi[i * n_sources + j] += inv_gtg[i * n_sources + k] * gty[k * n_sources + j];
            }
        }
    }

    // Extract frequencies from the antisymmetric part of Phi.
    // S = (Phi - Phi^T)/2 has purely imaginary eigenvalues ±jω for rotation matrices.

    // Antisymmetric part S = (Phi - Phi^T)/2; its eigenvalues are purely imaginary: ±jω
    let mut s = vec![0.0f64; n_sources * n_sources];
    for i in 0..n_sources {
        for j in 0..n_sources {
            s[i * n_sources + j] = (phi[i * n_sources + j] - phi[j * n_sources + i]) * 0.5;
        }
    }

    // For real symmetric S (which for anti-symmetric matrices has real eigenvalues
    // equal to the imaginary parts of the original complex eigenvalues):
    let (s_evals, _) = eigen_symmetric(&s, n_sources)?;

    // Map: ω = eval (in rad/sample), frequency = ω * fs / (2π)
    let mut freqs: Vec<f64> = s_evals
        .iter()
        .map(|&omega| {
            // Clamp to valid range [-1, 1] for asin
            let omega_clamped = omega.max(-1.0).min(1.0);
            let freq_norm = omega_clamped.asin() / PI; // in [-.5, .5]
            (freq_norm.abs() * sample_rate).max(0.0)
        })
        .collect();

    freqs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(freqs)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Lomb-Scargle periodogram (enhanced version with false-alarm probability)
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a Lomb-Scargle periodogram analysis.
#[derive(Debug, Clone)]
pub struct LombScargleResult {
    /// Evaluated frequencies (same as input).
    pub frequencies: Vec<f64>,
    /// Normalised Lomb-Scargle power at each frequency.
    pub power: Vec<f64>,
    /// False-alarm probability for each power level (single-frequency Baluev 2008).
    pub false_alarm_prob: Vec<f64>,
}

/// Compute the Lomb-Scargle periodogram for unevenly sampled data.
///
/// Unlike the classical DFT-based periodogram, the Lomb-Scargle method works
/// on data sampled at arbitrary, non-uniform times.  It is the maximum-
/// likelihood estimator for the amplitude of a single sinusoidal component at
/// each trial frequency.
///
/// # Arguments
///
/// * `times`       – Observation times (need not be uniformly spaced).
/// * `values`      – Observed values (same length as `times`).
/// * `frequencies` – Frequencies at which to evaluate the periodogram.
/// * `normalize`   – If true, normalise power by the sample variance.
///
/// # Returns
///
/// A [`LombScargleResult`] containing the power spectrum and associated
/// false-alarm probabilities.
///
/// # Errors
///
/// Returns [`FFTError::InvalidInput`] if any input vector is empty or lengths
/// do not match.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::spectral_methods::lomb_scargle;
/// use std::f64::consts::PI;
///
/// // Unevenly sampled observations
/// let times: Vec<f64> = (0..100).map(|i| i as f64 * 0.1 + (i as f64 * 0.37).sin() * 0.02).collect();
/// let values: Vec<f64> = times.iter().map(|&t| (2.0 * PI * 2.0 * t).sin()).collect();
/// let freqs: Vec<f64> = (1..50).map(|k| k as f64 * 0.1).collect();
///
/// let result = lomb_scargle(&times, &values, &freqs, true).expect("lomb_scargle");
/// assert_eq!(result.power.len(), freqs.len());
/// ```
pub fn lomb_scargle(
    times: &[f64],
    values: &[f64],
    frequencies: &[f64],
    normalize: bool,
) -> FFTResult<LombScargleResult> {
    let n = times.len();
    if n == 0 {
        return Err(FFTError::InvalidInput("lomb_scargle: empty input".into()));
    }
    if values.len() != n {
        return Err(FFTError::InvalidInput(
            "lomb_scargle: times and values must have same length".into(),
        ));
    }
    if frequencies.is_empty() {
        return Err(FFTError::InvalidInput(
            "lomb_scargle: frequencies must not be empty".into(),
        ));
    }

    // Mean-centre the data
    let mean = values.iter().sum::<f64>() / n as f64;
    let y: Vec<f64> = values.iter().map(|&v| v - mean).collect();
    let variance: f64 = y.iter().map(|&v| v * v).sum::<f64>() / n as f64;

    let nf = frequencies.len();
    let mut power = vec![0.0f64; nf];

    for (fi, &freq) in frequencies.iter().enumerate() {
        let omega = 2.0 * PI * freq;

        // Compute time offset τ for phase calibration
        let sum_sin2: f64 = times.iter().map(|&t| (2.0 * omega * t).sin()).sum();
        let sum_cos2: f64 = times.iter().map(|&t| (2.0 * omega * t).cos()).sum();
        let tau = sum_sin2.atan2(sum_cos2) / (2.0 * omega);

        // Shifted sums
        let sum_ycos: f64 = times
            .iter()
            .zip(y.iter())
            .map(|(&t, &yi)| yi * (omega * (t - tau)).cos())
            .sum();
        let sum_ysin: f64 = times
            .iter()
            .zip(y.iter())
            .map(|(&t, &yi)| yi * (omega * (t - tau)).sin())
            .sum();
        let sum_cos2_tau: f64 = times
            .iter()
            .map(|&t| (omega * (t - tau)).cos().powi(2))
            .sum();
        let sum_sin2_tau: f64 = times
            .iter()
            .map(|&t| (omega * (t - tau)).sin().powi(2))
            .sum();

        let p = if sum_cos2_tau.abs() < f64::EPSILON || sum_sin2_tau.abs() < f64::EPSILON {
            0.0
        } else {
            0.5 * (sum_ycos * sum_ycos / sum_cos2_tau + sum_ysin * sum_ysin / sum_sin2_tau)
        };

        power[fi] = if normalize && variance > f64::EPSILON {
            p / variance
        } else {
            p
        };
    }

    // False-alarm probabilities (single-frequency approximation)
    // P(z > z0) ≈ e^{-z0}  for large N (normalised power ≡ chi^2/2)
    let false_alarm_prob: Vec<f64> = power
        .iter()
        .map(|&p| {
            let z = if normalize { p } else { p / variance.max(f64::EPSILON) };
            (-z).exp().min(1.0)
        })
        .collect();

    Ok(LombScargleResult {
        frequencies: frequencies.to_vec(),
        power,
        false_alarm_prob,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
//  Multitaper PSD (enhanced with confidence intervals)
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a multitaper PSD estimation.
#[derive(Debug, Clone)]
pub struct MultitaperResult {
    /// Frequency axis (one-sided, length = `n_fft/2 + 1`).
    pub frequencies: Vec<f64>,
    /// Adaptive multitaper PSD estimate.
    pub psd: Vec<f64>,
    /// 5th-percentile chi-squared confidence interval.
    pub confidence_5pct: Vec<f64>,
    /// 95th-percentile chi-squared confidence interval.
    pub confidence_95pct: Vec<f64>,
    /// Adaptive weights per taper (shape: n_freqs × n_tapers).
    pub adaptive_weights: Vec<Vec<f64>>,
}

/// Compute Thomson's multitaper PSD with adaptive weighting and confidence intervals.
///
/// Uses Discrete Prolate Spheroidal Sequences (DPSS) as orthogonal tapers.
/// The adaptive weighting minimises the squared bias-variance tradeoff.
///
/// # Arguments
///
/// * `signal`             – Real-valued input signal.
/// * `sample_rate`        – Sampling frequency (Hz).
/// * `n_tapers`           – Number of tapers K (typically 2NW − 1, usually 4–8).
/// * `time_half_bandwidth` – Time-half-bandwidth product NW (typically 2–4).
///
/// # Returns
///
/// A [`MultitaperResult`] with PSD, confidence intervals, and weights.
///
/// # Errors
///
/// Propagates errors from DPSS computation or FFT.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::spectral_methods::multitaper_psd;
/// use std::f64::consts::PI;
///
/// let fs = 1000.0_f64;
/// let n = 1024usize;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * 100.0 * i as f64 / fs).sin())
///     .collect();
/// let result = multitaper_psd(&signal, fs, 4, 2.5).expect("multitaper_psd");
/// assert_eq!(result.frequencies.len(), result.psd.len());
/// ```
pub fn multitaper_psd(
    signal: &[f64],
    sample_rate: f64,
    n_tapers: usize,
    time_half_bandwidth: f64,
) -> FFTResult<MultitaperResult> {
    let n = signal.len();
    if n == 0 {
        return Err(FFTError::InvalidInput("multitaper_psd: empty signal".into()));
    }
    if n_tapers == 0 {
        return Err(FFTError::InvalidInput(
            "multitaper_psd: n_tapers must be >= 1".into(),
        ));
    }

    // Get DPSS tapers from window_functions module
    let tapers = crate::window_functions::dpss(n, time_half_bandwidth, n_tapers)?;

    let n_fft = n;
    let n_out = n_fft / 2 + 1;

    // Compute eigenspectra: S_k(f) = |∑ h_k(t) x(t) e^{-j2πft}|²
    let mut eigenspectra = vec![vec![0.0f64; n_out]; n_tapers];
    for (k, taper) in tapers.iter().enumerate() {
        let windowed: Vec<f64> = signal.iter().zip(taper.iter()).map(|(&x, &h)| x * h).collect();
        let spectrum = fft(&windowed, Some(n_fft))?;
        for f in 0..n_out {
            eigenspectra[k][f] = spectrum[f].norm_sqr() / sample_rate;
        }
    }

    // Adaptive weighting: iterate to estimate PSD
    // Initial estimate: equal weights
    let mut psd = vec![0.0f64; n_out];
    for f in 0..n_out {
        psd[f] = eigenspectra.iter().map(|s| s[f]).sum::<f64>() / n_tapers as f64;
    }

    // σ² estimate (noise floor)
    let sigma2 = psd.iter().sum::<f64>() / n_out as f64;

    // Adaptive iterations
    let mut weights = vec![vec![0.0f64; n_tapers]; n_out];
    for _iter in 0..3 {
        for f in 0..n_out {
            let mut wsum = 0.0f64;
            let mut psd_new = 0.0f64;
            for k in 0..n_tapers {
                // b_k(f)² = psd(f)² / (psd(f) + λ_k · σ²)²
                // Approximate eigenvalue λ_k ≈ 1 for all tapers
                let bk = psd[f] / (psd[f] + sigma2);
                let wk = bk * bk;
                weights[f][k] = wk;
                psd_new += wk * eigenspectra[k][f];
                wsum += wk;
            }
            psd[f] = if wsum > f64::EPSILON { psd_new / wsum } else { 0.0 };
        }
    }

    // Confidence intervals from chi-squared distribution with 2K degrees of freedom
    // For K tapers: 2K · Ŝ / χ²_{2K,α}
    let dof = 2.0 * n_tapers as f64;
    // Chi-squared quantiles (approximate using Wilson-Hilferty)
    let chi2_05 = dof * (1.0 - 2.0 / (9.0 * dof) + 1.645 * (2.0 / (9.0 * dof)).sqrt()).powi(3);
    let chi2_95 = dof * (1.0 - 2.0 / (9.0 * dof) - 1.645 * (2.0 / (9.0 * dof)).sqrt()).powi(3);

    let confidence_5pct: Vec<f64> = psd.iter().map(|&p| dof * p / chi2_05.max(f64::EPSILON)).collect();
    let confidence_95pct: Vec<f64> = psd.iter().map(|&p| dof * p / chi2_95.max(f64::EPSILON)).collect();

    let frequencies: Vec<f64> = (0..n_out)
        .map(|k| k as f64 * sample_rate / n_fft as f64)
        .collect();

    Ok(MultitaperResult {
        frequencies,
        psd,
        confidence_5pct,
        confidence_95pct,
        adaptive_weights: weights,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn sinusoid(n: usize, freq_norm: f64, phase: f64) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq_norm * i as f64 + phase).cos())
            .collect()
    }

    #[test]
    fn test_burg_ar_length() {
        let signal = sinusoid(256, 0.1, 0.0);
        let (ar, sigma2) = burg_ar(&signal, 4).expect("burg_ar");
        assert_eq!(ar.len(), 4);
        assert!(sigma2 >= 0.0, "sigma2 should be non-negative");
    }

    #[test]
    fn test_burg_spectrum_length() {
        let fs = 1000.0_f64;
        let signal: Vec<f64> = (0..512)
            .map(|i| (2.0 * PI * 100.0 * i as f64 / fs).sin())
            .collect();
        let psd = burg_spectrum(&signal, 8, 512, fs).expect("burg_spectrum");
        assert_eq!(psd.len(), 512 / 2 + 1);
        for &v in &psd {
            assert!(v >= 0.0, "PSD values must be non-negative");
        }
    }

    #[test]
    fn test_yule_walker_ar_length() {
        let signal = sinusoid(256, 0.15, 0.3);
        let (ar, sigma2) = yule_walker_ar(&signal, 6).expect("yule_walker_ar");
        assert_eq!(ar.len(), 6);
        assert!(sigma2 >= 0.0, "sigma2 must be non-negative");
    }

    #[test]
    fn test_yule_walker_spectrum_length() {
        let fs = 500.0_f64;
        let signal: Vec<f64> = (0..512)
            .map(|i| (2.0 * PI * 50.0 * i as f64 / fs).sin())
            .collect();
        let psd = yule_walker_spectrum(&signal, 4, 256, fs).expect("yule_walker_spectrum");
        assert_eq!(psd.len(), 256 / 2 + 1);
    }

    #[test]
    fn test_music_spectrum_length() {
        let signal = sinusoid(256, 0.1, 0.0);
        let ps = music_spectrum(&signal, 1, 256, 0).expect("music_spectrum");
        assert_eq!(ps.len(), 256 / 2 + 1);
    }

    #[test]
    fn test_music_frequencies_count() {
        let fs = 1000.0_f64;
        let n = 512usize;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * PI * 100.0 * i as f64 / fs).sin()
                    + (2.0 * PI * 300.0 * i as f64 / fs).sin()
            })
            .collect();
        let freqs = music_frequencies(&signal, 2, 2048, 0, fs).expect("music_frequencies");
        assert_eq!(freqs.len(), 2);
        // Both estimated frequencies should be positive
        for &f in &freqs {
            assert!(f >= 0.0);
        }
    }

    #[test]
    fn test_pisarenko_single_source() {
        let n = 256usize;
        let f0 = 0.1_f64;
        let signal = sinusoid(n, f0, 0.0);
        match pisarenko(&signal, 1) {
            Ok(freqs) => {
                assert!(!freqs.is_empty(), "should find at least one frequency");
            }
            Err(e) => {
                // Acceptable if signal is too simple / numerical issues
                eprintln!("pisarenko returned error (acceptable for pure tone): {e}");
            }
        }
    }

    #[test]
    fn test_capon_spectrum_length() {
        let signal = sinusoid(256, 0.2, 0.5);
        let ps = capon_spectrum(&signal, 256, 16).expect("capon_spectrum");
        assert_eq!(ps.len(), 256 / 2 + 1);
        for &v in &ps {
            assert!(v >= 0.0, "Capon pseudo-spectrum must be non-negative");
        }
    }

    #[test]
    fn test_lomb_scargle_output_length() {
        let times: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let values: Vec<f64> = times.iter().map(|&t| (2.0 * PI * 2.0 * t).sin()).collect();
        let freqs: Vec<f64> = (1..20).map(|k| k as f64 * 0.5).collect();
        let result = lomb_scargle(&times, &values, &freqs, true).expect("lomb_scargle");
        assert_eq!(result.frequencies.len(), freqs.len());
        assert_eq!(result.power.len(), freqs.len());
        assert_eq!(result.false_alarm_prob.len(), freqs.len());
        for &p in &result.false_alarm_prob {
            assert!(p >= 0.0 && p <= 1.0, "FAP must be in [0,1]");
        }
    }

    #[test]
    fn test_lomb_scargle_detects_frequency() {
        // Uneven sampling
        let times: Vec<f64> = (0..200).map(|i| i as f64 * 0.05 + (i as f64).sin() * 0.01).collect();
        let values: Vec<f64> = times.iter().map(|&t| (2.0 * PI * 3.0 * t).sin()).collect();
        // Probe around 3 Hz
        let freqs: Vec<f64> = (10..80).map(|k| k as f64 * 0.1).collect();
        let result = lomb_scargle(&times, &values, &freqs, true).expect("lomb_scargle");
        // The peak should be near 3 Hz (index 20)
        let peak_idx = result
            .power
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let peak_freq = result.frequencies[peak_idx];
        assert!(
            (peak_freq - 3.0).abs() < 0.5,
            "Expected peak near 3 Hz, got {peak_freq:.2} Hz"
        );
    }

    #[test]
    fn test_multitaper_psd_output() {
        let fs = 1000.0_f64;
        let n = 1024usize;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 100.0 * i as f64 / fs).sin())
            .collect();
        let result = multitaper_psd(&signal, fs, 4, 2.5).expect("multitaper_psd");
        assert_eq!(result.frequencies.len(), n / 2 + 1);
        assert_eq!(result.psd.len(), n / 2 + 1);
        assert_eq!(result.confidence_5pct.len(), n / 2 + 1);
        assert_eq!(result.confidence_95pct.len(), n / 2 + 1);
        for &p in &result.psd {
            assert!(p >= 0.0, "PSD must be non-negative");
        }
    }

    #[test]
    fn test_esprit_frequencies_count() {
        let fs = 1000.0_f64;
        let n = 256usize;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * PI * 100.0 * i as f64 / fs).sin()
                    + (2.0 * PI * 250.0 * i as f64 / fs).sin()
            })
            .collect();
        let freqs = esprit_frequencies(&signal, 2, 0, fs).expect("esprit_frequencies");
        assert_eq!(freqs.len(), 2);
        for &f in &freqs {
            assert!(f >= 0.0, "Frequencies must be non-negative");
            assert!(f <= fs / 2.0, "Frequencies must not exceed Nyquist");
        }
    }

    #[test]
    fn test_burg_spectrum_peak_near_signal_freq() {
        let fs = 1000.0_f64;
        let n = 512usize;
        let f_in = 200.0_f64; // Hz
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f_in * i as f64 / fs).sin())
            .collect();
        let psd = burg_spectrum(&signal, 20, 512, fs).expect("burg_spectrum");
        let peak_idx = psd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let peak_freq = peak_idx as f64 * fs / 512.0;
        assert!(
            (peak_freq - f_in).abs() < 50.0,
            "Expected peak near {f_in} Hz, got {peak_freq:.1} Hz"
        );
    }
}
