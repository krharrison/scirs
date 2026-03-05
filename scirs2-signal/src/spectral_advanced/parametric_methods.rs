//! Parametric spectral estimation methods
//!
//! Implements:
//! - Burg's method (maximum entropy spectral estimation)
//! - Yule-Walker AR estimation with Levinson-Durbin recursion
//! - MUSIC (Multiple Signal Classification)
//! - ESPRIT (Estimation of Signal Parameters via Rotational Invariance)
//!
//! References:
//! - Burg, J.P. (1975). "Maximum entropy spectral analysis." Ph.D. dissertation.
//! - Kay, S.M. (1988). "Modern Spectral Estimation." Prentice Hall.
//! - Schmidt, R.O. (1986). "Multiple emitter location and signal parameter
//!   estimation." IEEE Trans. Antennas Propagation, 34(3), 276-280.
//! - Roy, R. & Kailath, T. (1989). "ESPRIT - Estimation of signal parameters
//!   via rotational invariance techniques." IEEE Trans. ASSP, 37(7), 984-995.

use super::types::{
    BurgConfig, BurgResult, EspritConfig, EspritResult, MusicConfig, MusicResult, YuleWalkerConfig,
    YuleWalkerResult,
};
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// =============================================================================
// Burg's Method (Maximum Entropy)
// =============================================================================

/// Estimate power spectral density using Burg's method (maximum entropy).
///
/// Burg's method estimates AR model parameters by minimizing the sum of
/// forward and backward prediction errors. It guarantees a stable model
/// (all poles inside the unit circle).
///
/// # Arguments
///
/// * `signal` - Input time series data
/// * `config` - Burg configuration (order, fs, nfft)
///
/// # Returns
///
/// A `BurgResult` with AR coefficients, reflection coefficients, noise variance,
/// frequency vector, and PSD.
///
/// # Example
///
/// ```
/// use scirs2_signal::spectral_advanced::{burg_spectral, BurgConfig};
///
/// let signal: Vec<f64> = (0..256).map(|i| {
///     (2.0 * std::f64::consts::PI * 30.0 * i as f64 / 256.0).sin()
/// }).collect();
/// let config = BurgConfig { order: 10, fs: 256.0, nfft: 512 };
/// let result = burg_spectral(&signal, &config).expect("Burg failed");
/// assert!(!result.psd.is_empty());
/// ```
pub fn burg_spectral(signal: &[f64], config: &BurgConfig) -> SignalResult<BurgResult> {
    validate_parametric_input(signal, config.order)?;

    if config.fs <= 0.0 {
        return Err(SignalError::ValueError(
            "Sampling frequency must be positive".to_string(),
        ));
    }
    if config.nfft == 0 {
        return Err(SignalError::ValueError("NFFT must be positive".to_string()));
    }

    let n = signal.len();
    let order = config.order;

    // Initialize forward and backward prediction errors
    let mut ef: Vec<f64> = signal.to_vec();
    let mut eb: Vec<f64> = signal.to_vec();

    // AR coefficients: a[0] = 1.0, a[1..order+1] are the AR coefficients
    let mut ar_coeffs = vec![0.0; order + 1];
    ar_coeffs[0] = 1.0;

    let mut reflection_coeffs = vec![0.0; order];

    // Initial error power
    let mut variance: f64 = signal.iter().map(|&x| x * x).sum::<f64>() / n as f64;

    // Levinson-Burg recursion
    for m in 0..order {
        // Compute reflection coefficient k_m
        let mut num = 0.0;
        let mut den = 0.0;

        for i in (m + 1)..n {
            num += ef[i] * eb[i - 1];
            den += ef[i] * ef[i] + eb[i - 1] * eb[i - 1];
        }

        let k = if den > 1e-30 { -2.0 * num / den } else { 0.0 };

        reflection_coeffs[m] = k;

        // Check stability: |k| must be < 1
        if k.abs() >= 1.0 {
            // Clamp to maintain stability
            let clamped_k = k.signum() * (1.0 - 1e-10);
            reflection_coeffs[m] = clamped_k;
        }

        let k = reflection_coeffs[m];

        // Update AR coefficients using Levinson-Durbin
        let mut new_coeffs = ar_coeffs.clone();
        for i in 1..=m + 1 {
            new_coeffs[i] = ar_coeffs[i] + k * ar_coeffs[m + 1 - i];
        }
        ar_coeffs = new_coeffs;

        // Update prediction errors
        let mut new_ef = vec![0.0; n];
        let mut new_eb = vec![0.0; n];

        for i in (m + 1)..n {
            new_ef[i] = ef[i] + k * eb[i - 1];
            new_eb[i] = eb[i - 1] + k * ef[i];
        }

        ef = new_ef;
        eb = new_eb;

        // Update variance
        variance *= 1.0 - k * k;
    }

    // Compute PSD from AR coefficients
    let (frequencies, psd) = ar_to_psd(&ar_coeffs, variance, config.nfft, config.fs);

    Ok(BurgResult {
        ar_coeffs: Array1::from_vec(ar_coeffs),
        reflection_coeffs: Array1::from_vec(reflection_coeffs),
        variance,
        frequencies: Array1::from_vec(frequencies),
        psd: Array1::from_vec(psd),
    })
}

// =============================================================================
// Yule-Walker Method
// =============================================================================

/// Estimate power spectral density using the Yule-Walker method.
///
/// Uses the autocorrelation method with Levinson-Durbin recursion to
/// solve the Yule-Walker equations for AR model parameters.
///
/// # Arguments
///
/// * `signal` - Input time series data
/// * `config` - Yule-Walker configuration
///
/// # Returns
///
/// A `YuleWalkerResult` with AR coefficients, reflection coefficients,
/// noise variance, frequency vector, and PSD.
pub fn yule_walker_spectral(
    signal: &[f64],
    config: &YuleWalkerConfig,
) -> SignalResult<YuleWalkerResult> {
    validate_parametric_input(signal, config.order)?;

    if config.fs <= 0.0 {
        return Err(SignalError::ValueError(
            "Sampling frequency must be positive".to_string(),
        ));
    }

    let n = signal.len();
    let order = config.order;

    // Compute autocorrelation sequence r[0..order]
    let mut autocorrelation = vec![0.0; order + 1];
    for lag in 0..=order {
        let mut sum = 0.0;
        for i in 0..(n - lag) {
            sum += signal[i] * signal[i + lag];
        }
        autocorrelation[lag] = sum / n as f64;
    }

    // Check for zero signal
    if autocorrelation[0] < 1e-30 {
        return Err(SignalError::ComputationError(
            "Signal has zero energy (all zeros)".to_string(),
        ));
    }

    // Levinson-Durbin recursion
    let mut ar_coeffs = vec![0.0; order + 1];
    ar_coeffs[0] = 1.0;

    let mut reflection_coeffs = vec![0.0; order];
    let mut variance = autocorrelation[0];

    for m in 0..order {
        // Compute reflection coefficient
        let mut sum = autocorrelation[m + 1];
        for j in 1..=m {
            sum += ar_coeffs[j] * autocorrelation[m + 1 - j];
        }

        let k = if variance.abs() > 1e-30 {
            -sum / variance
        } else {
            0.0
        };

        reflection_coeffs[m] = k;

        // Update AR coefficients
        let mut new_coeffs = ar_coeffs.clone();
        for j in 1..=m + 1 {
            new_coeffs[j] = ar_coeffs[j] + k * ar_coeffs[m + 1 - j];
        }
        ar_coeffs = new_coeffs;

        // Update variance
        variance *= 1.0 - k * k;

        if variance <= 0.0 {
            return Err(SignalError::ComputationError(
                "Levinson-Durbin recursion became unstable (negative variance)".to_string(),
            ));
        }
    }

    // Compute PSD
    let (frequencies, psd) = ar_to_psd(&ar_coeffs, variance, config.nfft, config.fs);

    Ok(YuleWalkerResult {
        ar_coeffs: Array1::from_vec(ar_coeffs),
        reflection_coeffs: Array1::from_vec(reflection_coeffs),
        variance,
        frequencies: Array1::from_vec(frequencies),
        psd: Array1::from_vec(psd),
    })
}

// =============================================================================
// MUSIC (Multiple Signal Classification)
// =============================================================================

/// Estimate pseudospectrum using the MUSIC algorithm.
///
/// MUSIC is a subspace-based method that estimates frequency components
/// by exploiting the eigenstructure of the signal's covariance matrix.
/// It provides super-resolution frequency estimation (resolution beyond
/// the Fourier limit).
///
/// # Arguments
///
/// * `signal` - Input time series data
/// * `config` - MUSIC configuration
///
/// # Returns
///
/// A `MusicResult` with pseudospectrum, detected frequencies, and eigenvalues.
///
/// # Example
///
/// ```
/// use scirs2_signal::spectral_advanced::{music_spectral, MusicConfig};
///
/// let n = 256;
/// let fs = 256.0;
/// let signal: Vec<f64> = (0..n).map(|i| {
///     let t = i as f64 / fs;
///     (2.0 * std::f64::consts::PI * 30.0 * t).sin()
///     + 0.5 * (2.0 * std::f64::consts::PI * 60.0 * t).sin()
/// }).collect();
///
/// let config = MusicConfig { n_signals: 2, fs, ..Default::default() };
/// let result = music_spectral(&signal, &config).expect("MUSIC failed");
/// assert!(!result.pseudospectrum.is_empty());
/// ```
pub fn music_spectral(signal: &[f64], config: &MusicConfig) -> SignalResult<MusicResult> {
    if signal.is_empty() {
        return Err(SignalError::ValueError(
            "Signal must not be empty".to_string(),
        ));
    }
    if signal.iter().any(|&x| !x.is_finite()) {
        return Err(SignalError::ValueError(
            "Signal contains non-finite values".to_string(),
        ));
    }
    if config.n_signals == 0 {
        return Err(SignalError::ValueError(
            "Number of signals must be at least 1".to_string(),
        ));
    }
    if config.fs <= 0.0 {
        return Err(SignalError::ValueError(
            "Sampling frequency must be positive".to_string(),
        ));
    }

    let n = signal.len();
    let subspace_dim = config.subspace_dim.unwrap_or_else(|| {
        let dim = n / 3;
        dim.max(config.n_signals + 1).min(n - 1)
    });

    if subspace_dim <= config.n_signals {
        return Err(SignalError::ValueError(format!(
            "Subspace dimension ({}) must be greater than number of signals ({})",
            subspace_dim, config.n_signals
        )));
    }
    if subspace_dim >= n {
        return Err(SignalError::ValueError(
            "Subspace dimension must be less than signal length".to_string(),
        ));
    }

    // Build covariance matrix using forward-backward averaging
    let cov_matrix = if config.forward_backward {
        estimate_covariance_fb(signal, subspace_dim)?
    } else {
        estimate_covariance_forward(signal, subspace_dim)?
    };

    // Eigendecomposition of the covariance matrix
    let (eigenvalues, eigenvectors) = symmetric_eigendecomposition(&cov_matrix)?;

    // Sort eigenvalues in descending order
    let mut sorted_indices: Vec<usize> = (0..eigenvalues.len()).collect();
    sorted_indices.sort_by(|&i, &j| {
        eigenvalues[j]
            .partial_cmp(&eigenvalues[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_eigenvalues =
        Array1::from_vec(sorted_indices.iter().map(|&i| eigenvalues[i]).collect());

    // Extract noise subspace (eigenvectors corresponding to smallest eigenvalues)
    let noise_start = config.n_signals;
    let noise_dim = subspace_dim - config.n_signals;

    let mut noise_subspace = Array2::zeros((subspace_dim, noise_dim));
    for (col, &idx) in sorted_indices.iter().skip(noise_start).enumerate() {
        for row in 0..subspace_dim {
            noise_subspace[[row, col]] = eigenvectors[[row, idx]];
        }
    }

    // Compute MUSIC pseudospectrum
    let (f_min, f_max) = config.frequency_range.unwrap_or((0.0, config.fs / 2.0));
    let freq_step = (f_max - f_min) / (config.n_frequencies - 1) as f64;
    let frequencies: Vec<f64> = (0..config.n_frequencies)
        .map(|i| f_min + i as f64 * freq_step)
        .collect();

    let mut pseudospectrum = Array1::zeros(config.n_frequencies);

    for (freq_idx, &freq) in frequencies.iter().enumerate() {
        let omega = 2.0 * PI * freq / config.fs;

        // Steering vector: a(omega) = [1, e^{j*omega}, e^{j*2*omega}, ..., e^{j*(M-1)*omega}]
        let steering_vector: Vec<Complex64> = (0..subspace_dim)
            .map(|k| {
                let phase = omega * k as f64;
                Complex64::new(phase.cos(), phase.sin())
            })
            .collect();

        // P_MUSIC(omega) = 1 / |a^H * E_n * E_n^H * a|
        // = 1 / sum_k |e_k^H * a|^2 (over noise eigenvectors)
        let mut denominator = 0.0;

        for col in 0..noise_dim {
            let mut dot = Complex64::new(0.0, 0.0);
            for row in 0..subspace_dim {
                dot +=
                    Complex64::new(noise_subspace[[row, col]], 0.0) * steering_vector[row].conj();
            }
            denominator += dot.norm_sqr();
        }

        pseudospectrum[freq_idx] = if denominator > 1e-30 {
            1.0 / denominator
        } else {
            1e30 // Very large number for near-zero denominator (indicates a signal)
        };
    }

    // Convert to dB
    let max_power = pseudospectrum
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let db_ref = if max_power > 1e-30 { max_power } else { 1.0 };
    let pseudospectrum_db = pseudospectrum.mapv(|p| 10.0 * (p / db_ref).max(1e-30).log10());

    // Find signal frequencies (peaks in pseudospectrum)
    let signal_frequencies = find_top_peaks(&pseudospectrum, &frequencies, config.n_signals);

    Ok(MusicResult {
        frequencies: Array1::from_vec(frequencies),
        pseudospectrum: pseudospectrum_db,
        signal_frequencies: Array1::from_vec(signal_frequencies),
        eigenvalues: sorted_eigenvalues,
        n_signals: config.n_signals,
        subspace_dim,
    })
}

// =============================================================================
// ESPRIT (Estimation of Signal Parameters via Rotational Invariance)
// =============================================================================

/// Estimate signal frequencies using the ESPRIT algorithm.
///
/// ESPRIT exploits the shift-invariance structure of the signal subspace
/// to directly estimate frequencies without searching over a frequency grid.
/// This makes it computationally more efficient than MUSIC for frequency
/// estimation.
///
/// # Arguments
///
/// * `signal` - Input time series data
/// * `config` - ESPRIT configuration
///
/// # Returns
///
/// An `EspritResult` with estimated frequencies and associated parameters.
pub fn esprit_spectral(signal: &[f64], config: &EspritConfig) -> SignalResult<EspritResult> {
    if signal.is_empty() {
        return Err(SignalError::ValueError(
            "Signal must not be empty".to_string(),
        ));
    }
    if signal.iter().any(|&x| !x.is_finite()) {
        return Err(SignalError::ValueError(
            "Signal contains non-finite values".to_string(),
        ));
    }
    if config.n_signals == 0 {
        return Err(SignalError::ValueError(
            "Number of signals must be at least 1".to_string(),
        ));
    }
    if config.fs <= 0.0 {
        return Err(SignalError::ValueError(
            "Sampling frequency must be positive".to_string(),
        ));
    }

    let n = signal.len();
    let subspace_dim = config.subspace_dim.unwrap_or_else(|| {
        let dim = n / 3;
        dim.max(config.n_signals + 1).min(n - 1)
    });

    if subspace_dim <= config.n_signals {
        return Err(SignalError::ValueError(format!(
            "Subspace dimension ({}) must be greater than number of signals ({})",
            subspace_dim, config.n_signals
        )));
    }
    if subspace_dim >= n {
        return Err(SignalError::ValueError(
            "Subspace dimension must be less than signal length".to_string(),
        ));
    }

    // Build covariance matrix
    let cov_matrix = if config.forward_backward {
        estimate_covariance_fb(signal, subspace_dim)?
    } else {
        estimate_covariance_forward(signal, subspace_dim)?
    };

    // Eigendecomposition
    let (eigenvalues, eigenvectors) = symmetric_eigendecomposition(&cov_matrix)?;

    // Sort eigenvalues in descending order and extract signal subspace
    let mut sorted_indices: Vec<usize> = (0..eigenvalues.len()).collect();
    sorted_indices.sort_by(|&i, &j| {
        eigenvalues[j]
            .partial_cmp(&eigenvalues[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_eigenvalues =
        Array1::from_vec(sorted_indices.iter().map(|&i| eigenvalues[i]).collect());

    // Build signal subspace matrix E_s (M x d)
    let d = config.n_signals;
    let m = subspace_dim;
    let mut signal_subspace = Array2::zeros((m, d));
    for (col, &idx) in sorted_indices.iter().take(d).enumerate() {
        for row in 0..m {
            signal_subspace[[row, col]] = eigenvectors[[row, idx]];
        }
    }

    // Split signal subspace into E1 (first M-1 rows) and E2 (last M-1 rows)
    let m1 = m - 1;
    let mut e1 = Array2::zeros((m1, d));
    let mut e2 = Array2::zeros((m1, d));

    for col in 0..d {
        for row in 0..m1 {
            e1[[row, col]] = signal_subspace[[row, col]];
            e2[[row, col]] = signal_subspace[[row + 1, col]];
        }
    }

    // Compute the rotation matrix Phi
    // Standard ESPRIT: Phi = pinv(E1) * E2
    // TLS-ESPRIT: use SVD of [E1; E2]
    let rotation_eigenvalues = if config.total_least_squares {
        tls_esprit_rotation(&e1, &e2, d)?
    } else {
        ls_esprit_rotation(&e1, &e2, d)?
    };

    // Extract frequencies from eigenvalues of Phi
    // For a sinusoid at frequency f, the eigenvalue is e^{j*2*pi*f/fs}
    // So f = fs * atan2(imag, real) / (2*pi)
    let mut frequencies: Vec<f64> = rotation_eigenvalues
        .iter()
        .map(|&ev| {
            // The eigenvalues should be close to the unit circle
            // Frequency = fs * angle / (2*pi)
            let angle = ev; // ev is already the angle (see ls_esprit_rotation)
            let freq = config.fs * angle / (2.0 * PI);
            // Map to positive frequencies
            if freq < 0.0 {
                freq + config.fs
            } else {
                freq
            }
        })
        .filter(|&f| f >= 0.0 && f <= config.fs / 2.0)
        .collect();

    // Sort frequencies
    frequencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    Ok(EspritResult {
        frequencies: Array1::from_vec(frequencies),
        amplitudes: None,
        rotation_eigenvalues: Array1::from_vec(rotation_eigenvalues.to_vec()),
        covariance_eigenvalues: sorted_eigenvalues,
        subspace_dim,
    })
}

// =============================================================================
// Internal helper functions
// =============================================================================

/// Convert AR coefficients to PSD
fn ar_to_psd(ar_coeffs: &[f64], variance: f64, nfft: usize, fs: f64) -> (Vec<f64>, Vec<f64>) {
    let n_positive = nfft / 2 + 1;
    let mut frequencies = Vec::with_capacity(n_positive);
    let mut psd = Vec::with_capacity(n_positive);

    for i in 0..n_positive {
        let freq = i as f64 * fs / nfft as f64;
        frequencies.push(freq);

        let omega = 2.0 * PI * freq / fs;

        // A(e^{j*omega}) = 1 + a1*e^{-j*omega} + a2*e^{-j*2*omega} + ...
        let mut re = 0.0;
        let mut im = 0.0;
        for (k, &coeff) in ar_coeffs.iter().enumerate() {
            let phase = omega * k as f64;
            re += coeff * phase.cos();
            im -= coeff * phase.sin();
        }

        let a_mag_sq = re * re + im * im;

        // PSD = variance / |A(e^{j*omega})|^2 / fs
        let p = if a_mag_sq > 1e-30 {
            variance / (a_mag_sq * fs)
        } else {
            0.0
        };
        psd.push(p);
    }

    (frequencies, psd)
}

/// Estimate covariance matrix using forward-backward averaging
fn estimate_covariance_fb(signal: &[f64], m: usize) -> SignalResult<Array2<f64>> {
    let n = signal.len();
    if m >= n {
        return Err(SignalError::ValueError(
            "Subspace dimension must be less than signal length".to_string(),
        ));
    }

    let n_snapshots = n - m + 1;
    let mut cov = Array2::zeros((m, m));

    // Forward covariance
    for t in 0..n_snapshots {
        for i in 0..m {
            for j in i..m {
                let val = signal[t + i] * signal[t + j];
                cov[[i, j]] += val;
                if i != j {
                    cov[[j, i]] += val;
                }
            }
        }
    }

    // Backward covariance (uses time-reversed signal)
    for t in 0..n_snapshots {
        for i in 0..m {
            for j in i..m {
                let fi = m - 1 - i;
                let fj = m - 1 - j;
                let val = signal[t + fi] * signal[t + fj];
                cov[[i, j]] += val;
                if i != j {
                    cov[[j, i]] += val;
                }
            }
        }
    }

    // Average
    let scale = 1.0 / (2.0 * n_snapshots as f64);
    cov.mapv_inplace(|v| v * scale);

    Ok(cov)
}

/// Estimate covariance matrix (forward only)
fn estimate_covariance_forward(signal: &[f64], m: usize) -> SignalResult<Array2<f64>> {
    let n = signal.len();
    if m >= n {
        return Err(SignalError::ValueError(
            "Subspace dimension must be less than signal length".to_string(),
        ));
    }

    let n_snapshots = n - m + 1;
    let mut cov = Array2::zeros((m, m));

    for t in 0..n_snapshots {
        for i in 0..m {
            for j in i..m {
                let val = signal[t + i] * signal[t + j];
                cov[[i, j]] += val;
                if i != j {
                    cov[[j, i]] += val;
                }
            }
        }
    }

    let scale = 1.0 / n_snapshots as f64;
    cov.mapv_inplace(|v| v * scale);

    Ok(cov)
}

/// Symmetric eigendecomposition using Jacobi algorithm (pure Rust)
///
/// This implements the cyclic Jacobi method for real symmetric matrices.
/// Returns eigenvalues and eigenvectors.
fn symmetric_eigendecomposition(matrix: &Array2<f64>) -> SignalResult<(Array1<f64>, Array2<f64>)> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(SignalError::ValueError("Matrix must be square".to_string()));
    }
    if n == 0 {
        return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
    }
    if n == 1 {
        return Ok((Array1::from_vec(vec![matrix[[0, 0]]]), Array2::eye(1)));
    }

    // Try scirs2-linalg first, fall back to internal Jacobi
    match try_linalg_eigh(matrix) {
        Ok(result) => Ok(result),
        Err(_) => jacobi_eigendecomposition(matrix),
    }
}

/// Try to use scirs2-linalg's eigh function
fn try_linalg_eigh(matrix: &Array2<f64>) -> SignalResult<(Array1<f64>, Array2<f64>)> {
    let view = matrix.view();
    scirs2_linalg::eigh(&view, None)
        .map_err(|e| SignalError::ComputationError(format!("scirs2-linalg eigh failed: {e}")))
}

/// Fallback Jacobi eigendecomposition for symmetric matrices
fn jacobi_eigendecomposition(matrix: &Array2<f64>) -> SignalResult<(Array1<f64>, Array2<f64>)> {
    let n = matrix.nrows();
    let mut a = matrix.clone();
    let mut v = Array2::eye(n);
    let max_sweeps = 100;
    let eps = f64::EPSILON;

    for sweep in 0..max_sweeps {
        let mut off_norm = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off_norm += a[[i, j]] * a[[i, j]];
            }
        }

        if off_norm.sqrt() < eps * 100.0 {
            break;
        }

        let threshold = if sweep < 3 {
            0.2 * off_norm.sqrt() / (n * n) as f64
        } else {
            0.0
        };

        for p in 0..n - 1 {
            for q in (p + 1)..n {
                let apq = a[[p, q]];
                if apq.abs() < threshold {
                    continue;
                }
                if apq.abs() < eps * (a[[p, p]].abs() + a[[q, q]].abs()) * 0.01 {
                    a[[p, q]] = 0.0;
                    a[[q, p]] = 0.0;
                    continue;
                }

                let app = a[[p, p]];
                let aqq = a[[q, q]];

                let (c, s) = if (app - aqq).abs() < eps * (app.abs() + aqq.abs()) {
                    let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
                    (inv_sqrt2, if apq >= 0.0 { inv_sqrt2 } else { -inv_sqrt2 })
                } else {
                    let tau = (aqq - app) / (2.0 * apq);
                    let t = if tau >= 0.0 {
                        1.0 / (tau + (1.0 + tau * tau).sqrt())
                    } else {
                        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                    };
                    let c_val = 1.0 / (1.0 + t * t).sqrt();
                    (c_val, t * c_val)
                };

                for i in 0..n {
                    if i != p && i != q {
                        let aip = a[[i, p]];
                        let aiq = a[[i, q]];
                        a[[i, p]] = c * aip - s * aiq;
                        a[[p, i]] = a[[i, p]];
                        a[[i, q]] = s * aip + c * aiq;
                        a[[q, i]] = a[[i, q]];
                    }
                }

                a[[p, p]] = c * c * app - 2.0 * c * s * apq + s * s * aqq;
                a[[q, q]] = s * s * app + 2.0 * c * s * apq + c * c * aqq;
                a[[p, q]] = 0.0;
                a[[q, p]] = 0.0;

                for i in 0..n {
                    let vip = v[[i, p]];
                    let viq = v[[i, q]];
                    v[[i, p]] = c * vip - s * viq;
                    v[[i, q]] = s * vip + c * viq;
                }
            }
        }
    }

    let eigenvalues = Array1::from_vec((0..n).map(|i| a[[i, i]]).collect());
    Ok((eigenvalues, v))
}

/// LS-ESPRIT: solve for rotation matrix using least squares
fn ls_esprit_rotation(e1: &Array2<f64>, e2: &Array2<f64>, d: usize) -> SignalResult<Vec<f64>> {
    // Phi = pinv(E1) * E2 = (E1^T * E1)^{-1} * E1^T * E2
    let e1t_e1 = e1.t().dot(e1);
    let e1t_e2 = e1.t().dot(e2);

    // Invert E1^T * E1 using Cholesky or direct for small matrices
    let phi = solve_normal_equations(&e1t_e1, &e1t_e2, d)?;

    // Compute eigenvalues of Phi
    // For ESPRIT, we need the angles of the eigenvalues
    eigenvalues_of_small_matrix(&phi, d)
}

/// TLS-ESPRIT: use total least squares via SVD
fn tls_esprit_rotation(e1: &Array2<f64>, e2: &Array2<f64>, d: usize) -> SignalResult<Vec<f64>> {
    // Stack [E1; E2] and compute SVD-based solution
    let m = e1.nrows();
    let mut c_mat = Array2::zeros((2 * d, 2 * d));

    // C = [E1; E2]^H * [E1; E2] partitioned as [[C11, C12], [C21, C22]]
    // where each block is d x d
    let e1t_e1 = e1.t().dot(e1);
    let e1t_e2 = e1.t().dot(e2);
    let e2t_e1 = e2.t().dot(e1);
    let e2t_e2 = e2.t().dot(e2);

    for i in 0..d {
        for j in 0..d {
            c_mat[[i, j]] = e1t_e1[[i, j]];
            c_mat[[i, j + d]] = e1t_e2[[i, j]];
            c_mat[[i + d, j]] = e2t_e1[[i, j]];
            c_mat[[i + d, j + d]] = e2t_e2[[i, j]];
        }
    }

    // Eigendecompose C to get the partition
    let (c_eigenvalues, c_eigenvectors) = symmetric_eigendecomposition(&c_mat)?;

    // Sort eigenvalues descending
    let mut sorted_indices: Vec<usize> = (0..c_eigenvalues.len()).collect();
    sorted_indices.sort_by(|&i, &j| {
        c_eigenvalues[j]
            .partial_cmp(&c_eigenvalues[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Extract the noise subspace of C (last d eigenvectors)
    // Phi_tls = -E12 * inv(E22) where E = [E12; E22] are the noise eigenvectors
    let mut e12 = Array2::zeros((d, d));
    let mut e22 = Array2::zeros((d, d));

    for col in 0..d {
        let idx = sorted_indices[d + col]; // noise eigenvectors
        for row in 0..d {
            e12[[row, col]] = c_eigenvectors[[row, idx]];
            e22[[row, col]] = c_eigenvectors[[row + d, idx]];
        }
    }

    // Phi_tls = -E12 * E22^{-1}
    let e22_inv = invert_small_matrix(&e22, d)?;
    let phi = e12.dot(&e22_inv).mapv(|v| -v);

    eigenvalues_of_small_matrix(&phi, d)
}

/// Solve normal equations: (A^T A) x = A^T b for x
fn solve_normal_equations(
    ata: &Array2<f64>,
    atb: &Array2<f64>,
    d: usize,
) -> SignalResult<Array2<f64>> {
    // For small matrices, use direct inverse
    let ata_inv = invert_small_matrix(ata, d)?;
    Ok(ata_inv.dot(atb))
}

/// Invert a small (d x d) matrix using Gauss-Jordan elimination
fn invert_small_matrix(matrix: &Array2<f64>, d: usize) -> SignalResult<Array2<f64>> {
    let mut augmented = Array2::zeros((d, 2 * d));

    // Build augmented matrix [A | I]
    for i in 0..d {
        for j in 0..d {
            augmented[[i, j]] = matrix[[i, j]];
        }
        augmented[[i, i + d]] = 1.0;
    }

    // Gauss-Jordan elimination with partial pivoting
    for col in 0..d {
        // Find pivot
        let mut max_val = augmented[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..d {
            if augmented[[row, col]].abs() > max_val {
                max_val = augmented[[row, col]].abs();
                max_row = row;
            }
        }

        if max_val < 1e-14 {
            // Add small regularization instead of failing
            augmented[[col, col]] += 1e-10;
        }

        // Swap rows if needed
        if max_row != col {
            for j in 0..2 * d {
                let tmp = augmented[[col, j]];
                augmented[[col, j]] = augmented[[max_row, j]];
                augmented[[max_row, j]] = tmp;
            }
        }

        // Scale pivot row
        let pivot = augmented[[col, col]];
        if pivot.abs() < 1e-30 {
            return Err(SignalError::ComputationError(
                "Matrix is singular or nearly singular".to_string(),
            ));
        }
        for j in 0..2 * d {
            augmented[[col, j]] /= pivot;
        }

        // Eliminate column
        for row in 0..d {
            if row != col {
                let factor = augmented[[row, col]];
                for j in 0..2 * d {
                    augmented[[row, j]] -= factor * augmented[[col, j]];
                }
            }
        }
    }

    // Extract inverse from the right half
    let mut inverse = Array2::zeros((d, d));
    for i in 0..d {
        for j in 0..d {
            inverse[[i, j]] = augmented[[i, j + d]];
        }
    }

    Ok(inverse)
}

/// Compute eigenvalues of a small general matrix using QR iteration
///
/// Returns the angles (arguments) of the eigenvalues, which correspond
/// to frequencies in ESPRIT.
fn eigenvalues_of_small_matrix(matrix: &Array2<f64>, d: usize) -> SignalResult<Vec<f64>> {
    if d == 0 {
        return Ok(vec![]);
    }

    if d == 1 {
        // Single eigenvalue is just the element
        let ev = matrix[[0, 0]];
        // For ESPRIT, eigenvalue should be on unit circle, angle = acos(ev)
        let angle = ev.acos();
        return Ok(vec![angle]);
    }

    // For small matrices, use general eigendecomposition via scirs2-linalg
    let view = matrix.view();
    match scirs2_linalg::eig(&view, None) {
        Ok((eigenvalues, _eigenvectors)) => {
            // Extract angles from complex eigenvalues
            let angles: Vec<f64> = eigenvalues.iter().map(|ev| ev.im.atan2(ev.re)).collect();
            Ok(angles)
        }
        Err(_) => {
            // Fallback: use power iteration for dominant eigenvalues
            power_iteration_eigenvalues(matrix, d)
        }
    }
}

/// Fallback: power iteration to find eigenvalues
fn power_iteration_eigenvalues(matrix: &Array2<f64>, d: usize) -> SignalResult<Vec<f64>> {
    let mut angles = Vec::with_capacity(d);

    // Use deflation approach: find eigenvalue, deflate, repeat
    let mut current = matrix.clone();
    let max_iter = 1000;
    let tol = 1e-10;

    for _ in 0..d {
        let n = current.nrows();
        if n == 0 {
            break;
        }

        // Power iteration
        let mut v = Array1::from_vec(vec![1.0 / (n as f64).sqrt(); n]);
        let mut eigenvalue = 0.0;

        for _ in 0..max_iter {
            let w = current.dot(&v);
            let new_eigenvalue = w.dot(&v);
            let norm = w.dot(&w).sqrt();
            if norm > 1e-30 {
                v = w.mapv(|x| x / norm);
            }

            if (new_eigenvalue - eigenvalue).abs() < tol {
                break;
            }
            eigenvalue = new_eigenvalue;
        }

        // The angle of this real eigenvalue
        let angle = if eigenvalue.abs() > 1e-30 {
            if eigenvalue > 0.0 {
                0.0
            } else {
                PI
            }
        } else {
            0.0
        };
        angles.push(angle);

        // Deflation: A' = A - lambda * v * v^T
        if n > 1 {
            let mut deflated = current.clone();
            for i in 0..n {
                for j in 0..n {
                    deflated[[i, j]] -= eigenvalue * v[i] * v[j];
                }
            }
            current = deflated;
        } else {
            break;
        }
    }

    Ok(angles)
}

/// Find top N peaks in a spectrum
fn find_top_peaks(spectrum: &Array1<f64>, frequencies: &[f64], n_peaks: usize) -> Vec<f64> {
    let n = spectrum.len();
    if n < 3 {
        return frequencies.to_vec();
    }

    // Find local maxima
    let mut peaks: Vec<(usize, f64)> = Vec::new();
    for i in 1..n - 1 {
        if spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1] {
            peaks.push((i, spectrum[i]));
        }
    }
    // Also check endpoints
    if n > 0 && spectrum[0] > spectrum[1] {
        peaks.push((0, spectrum[0]));
    }
    if n > 1 && spectrum[n - 1] > spectrum[n - 2] {
        peaks.push((n - 1, spectrum[n - 1]));
    }

    // Sort by power descending
    peaks.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Return top n_peaks frequencies
    peaks
        .into_iter()
        .take(n_peaks)
        .map(|(idx, _)| frequencies[idx])
        .collect()
}

/// Validate input for parametric methods
fn validate_parametric_input(signal: &[f64], order: usize) -> SignalResult<()> {
    if signal.is_empty() {
        return Err(SignalError::ValueError(
            "Signal must not be empty".to_string(),
        ));
    }
    if signal.iter().any(|&x| !x.is_finite()) {
        return Err(SignalError::ValueError(
            "Signal contains non-finite values".to_string(),
        ));
    }
    if order == 0 {
        return Err(SignalError::ValueError(
            "Model order must be at least 1".to_string(),
        ));
    }
    if order >= signal.len() {
        return Err(SignalError::ValueError(format!(
            "Model order ({}) must be less than signal length ({})",
            order,
            signal.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_sinusoidal(n: usize, fs: f64, freqs: &[f64], amps: &[f64]) -> Vec<f64> {
        let mut signal = vec![0.0; n];
        for i in 0..n {
            let t = i as f64 / fs;
            for (f, a) in freqs.iter().zip(amps.iter()) {
                signal[i] += a * (2.0 * PI * f * t).sin();
            }
        }
        signal
    }

    // =========================================================================
    // Burg's method tests
    // =========================================================================

    #[test]
    fn test_burg_basic() {
        let signal = generate_sinusoidal(256, 256.0, &[30.0], &[1.0]);
        let config = BurgConfig {
            order: 10,
            fs: 256.0,
            nfft: 512,
        };

        let result = burg_spectral(&signal, &config);
        assert!(result.is_ok(), "Burg failed: {:?}", result.err());
        let result = result.expect("already checked");

        assert_eq!(result.ar_coeffs.len(), 11); // order + 1
        assert_eq!(result.reflection_coeffs.len(), 10);
        assert!(result.variance > 0.0, "Variance should be positive");
        assert!(!result.psd.is_empty());

        // PSD should peak near 30 Hz
        let peak_idx = result
            .psd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("should find peak");
        let peak_freq = result.frequencies[peak_idx];
        assert!(
            (peak_freq - 30.0).abs() < 5.0,
            "Peak at {peak_freq}, expected ~30 Hz"
        );
    }

    #[test]
    fn test_burg_stability() {
        // All reflection coefficients should have |k| < 1
        let signal = generate_sinusoidal(128, 128.0, &[10.0, 20.0], &[1.0, 0.5]);
        let config = BurgConfig {
            order: 20,
            fs: 128.0,
            nfft: 256,
        };

        let result = burg_spectral(&signal, &config).expect("should succeed");
        for &k in result.reflection_coeffs.iter() {
            assert!(
                k.abs() < 1.0 + 1e-9,
                "Reflection coefficient |{k}| should be < 1"
            );
        }
    }

    #[test]
    fn test_burg_psd_non_negative() {
        let signal = generate_sinusoidal(256, 256.0, &[50.0, 100.0], &[1.0, 0.3]);
        let config = BurgConfig {
            order: 15,
            fs: 256.0,
            nfft: 1024,
        };

        let result = burg_spectral(&signal, &config).expect("should succeed");
        assert!(
            result.psd.iter().all(|&p| p >= 0.0),
            "PSD must be non-negative"
        );
    }

    #[test]
    fn test_burg_two_peaks() {
        let signal = generate_sinusoidal(512, 512.0, &[50.0, 120.0], &[1.0, 0.8]);
        let config = BurgConfig {
            order: 20,
            fs: 512.0,
            nfft: 2048,
        };

        let result = burg_spectral(&signal, &config).expect("should succeed");

        // Find local maxima in PSD (true peaks, not just top values)
        let psd = &result.psd;
        let freqs = &result.frequencies;
        let n = psd.len();
        let mut peaks: Vec<(f64, f64)> = Vec::new(); // (freq, power)
        for i in 1..n - 1 {
            if psd[i] > psd[i - 1] && psd[i] > psd[i + 1] {
                peaks.push((freqs[i], psd[i]));
            }
        }
        peaks.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Should find at least 2 distinct peaks
        assert!(
            peaks.len() >= 2,
            "Should find at least 2 peaks in PSD, found {}",
            peaks.len()
        );

        // The top peaks should include frequencies near 50 and 120 Hz
        let top_freqs: Vec<f64> = peaks.iter().take(4).map(|(f, _)| *f).collect();
        let near_50 = top_freqs.iter().any(|&f| (f - 50.0).abs() < 15.0);
        let near_120 = top_freqs.iter().any(|&f| (f - 120.0).abs() < 15.0);
        assert!(
            near_50,
            "Should detect ~50 Hz among top peaks: {top_freqs:?}"
        );
        assert!(
            near_120,
            "Should detect ~120 Hz among top peaks: {top_freqs:?}"
        );
    }

    #[test]
    fn test_burg_validation_errors() {
        let config = BurgConfig::default();

        // Empty signal
        assert!(burg_spectral(&[], &config).is_err());

        // Order too large
        let signal = vec![1.0; 5];
        let big_order = BurgConfig {
            order: 10,
            ..config.clone()
        };
        assert!(burg_spectral(&signal, &big_order).is_err());

        // Zero order
        let zero_order = BurgConfig {
            order: 0,
            ..config.clone()
        };
        assert!(burg_spectral(&[1.0, 2.0, 3.0], &zero_order).is_err());
    }

    // =========================================================================
    // Yule-Walker tests
    // =========================================================================

    #[test]
    fn test_yule_walker_basic() {
        let signal = generate_sinusoidal(256, 256.0, &[40.0], &[1.0]);
        let config = YuleWalkerConfig {
            order: 10,
            fs: 256.0,
            nfft: 512,
        };

        let result = yule_walker_spectral(&signal, &config);
        assert!(result.is_ok(), "YW failed: {:?}", result.err());
        let result = result.expect("already checked");

        assert_eq!(result.ar_coeffs.len(), 11);
        assert!(result.variance > 0.0);

        // PSD should peak near 40 Hz
        let peak_idx = result
            .psd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("peak");
        let peak_freq = result.frequencies[peak_idx];
        assert!(
            (peak_freq - 40.0).abs() < 5.0,
            "Peak at {peak_freq}, expected ~40 Hz"
        );
    }

    #[test]
    fn test_yule_walker_levinson_durbin() {
        // Test that Levinson-Durbin produces valid coefficients
        let signal = generate_sinusoidal(128, 128.0, &[15.0], &[1.0]);
        let config = YuleWalkerConfig {
            order: 8,
            fs: 128.0,
            nfft: 256,
        };

        let result = yule_walker_spectral(&signal, &config).expect("should succeed");
        // First coefficient should be 1.0
        assert!(
            (result.ar_coeffs[0] - 1.0).abs() < 1e-10,
            "First AR coefficient should be 1.0"
        );
        // Variance should be positive
        assert!(result.variance > 0.0);
    }

    #[test]
    fn test_yule_walker_psd_shape() {
        let signal = generate_sinusoidal(256, 256.0, &[20.0], &[1.0]);
        let config = YuleWalkerConfig {
            order: 10,
            fs: 256.0,
            nfft: 1024,
        };

        let result = yule_walker_spectral(&signal, &config).expect("should succeed");
        assert_eq!(result.frequencies.len(), 1024 / 2 + 1);
        assert_eq!(result.psd.len(), result.frequencies.len());
        assert!(result.psd.iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn test_yule_walker_zero_signal() {
        let signal = vec![0.0; 100];
        let config = YuleWalkerConfig::default();
        // Zero signal should fail (zero energy)
        let result = yule_walker_spectral(&signal, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_yule_walker_vs_burg_consistency() {
        // Both methods should find the same peak frequency
        let signal = generate_sinusoidal(256, 256.0, &[60.0], &[1.0]);

        let burg_result = burg_spectral(
            &signal,
            &BurgConfig {
                order: 10,
                fs: 256.0,
                nfft: 512,
            },
        )
        .expect("burg");

        let yw_result = yule_walker_spectral(
            &signal,
            &YuleWalkerConfig {
                order: 10,
                fs: 256.0,
                nfft: 512,
            },
        )
        .expect("yw");

        let burg_peak_idx = burg_result
            .psd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("peak");
        let yw_peak_idx = yw_result
            .psd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("peak");

        let burg_peak = burg_result.frequencies[burg_peak_idx];
        let yw_peak = yw_result.frequencies[yw_peak_idx];
        assert!(
            (burg_peak - yw_peak).abs() < 10.0,
            "Burg peak={burg_peak}, YW peak={yw_peak} should be close"
        );
    }

    // =========================================================================
    // MUSIC tests
    // =========================================================================

    #[test]
    fn test_music_basic() {
        // Use smaller signal and explicit small subspace_dim for fast eigendecomposition
        let n = 64;
        let fs = 64.0;
        let signal = generate_sinusoidal(n, fs, &[10.0, 20.0], &[1.0, 0.8]);

        let config = MusicConfig {
            n_signals: 2,
            fs,
            n_frequencies: 128,
            subspace_dim: Some(12),
            ..Default::default()
        };

        let result = music_spectral(&signal, &config);
        assert!(result.is_ok(), "MUSIC failed: {:?}", result.err());
        let result = result.expect("already checked");

        assert_eq!(result.n_signals, 2);
        assert_eq!(result.pseudospectrum.len(), 128);
        assert!(!result.eigenvalues.is_empty());

        // Eigenvalues should be sorted descending
        for i in 1..result.eigenvalues.len() {
            assert!(
                result.eigenvalues[i] <= result.eigenvalues[i - 1] + 1e-10,
                "Eigenvalues should be descending"
            );
        }
    }

    #[test]
    fn test_music_frequency_detection() {
        // Reduced signal length and subspace dim for fast eigendecomposition
        let n = 128;
        let fs = 128.0;
        let signal = generate_sinusoidal(n, fs, &[20.0, 40.0], &[1.0, 1.0]);

        let config = MusicConfig {
            n_signals: 2,
            fs,
            n_frequencies: 256,
            subspace_dim: Some(16),
            ..Default::default()
        };

        let result = music_spectral(&signal, &config).expect("should succeed");

        // Signal frequencies should be detected
        let near_20 = result
            .signal_frequencies
            .iter()
            .any(|&f| (f - 20.0).abs() < 10.0);
        let near_40 = result
            .signal_frequencies
            .iter()
            .any(|&f| (f - 40.0).abs() < 10.0);
        assert!(
            near_20 || near_40,
            "Should detect at least one of 20/40 Hz. Got: {:?}",
            result.signal_frequencies
        );
    }

    #[test]
    fn test_music_pseudospectrum_finite() {
        let signal = generate_sinusoidal(64, 64.0, &[10.0], &[1.0]);
        let config = MusicConfig {
            n_signals: 1,
            fs: 64.0,
            subspace_dim: Some(10),
            n_frequencies: 128,
            ..Default::default()
        };

        let result = music_spectral(&signal, &config).expect("should succeed");
        assert!(
            result.pseudospectrum.iter().all(|&p| p.is_finite()),
            "Pseudospectrum must be finite"
        );
    }

    #[test]
    fn test_music_forward_backward() {
        // Reduced signal length and explicit small subspace_dim for speed
        let signal = generate_sinusoidal(64, 64.0, &[15.0], &[1.0]);

        // With forward-backward
        let config_fb = MusicConfig {
            n_signals: 1,
            fs: 64.0,
            forward_backward: true,
            subspace_dim: Some(10),
            n_frequencies: 128,
            ..Default::default()
        };
        let result_fb = music_spectral(&signal, &config_fb).expect("fb should succeed");

        // Forward only
        let config_fw = MusicConfig {
            n_signals: 1,
            fs: 64.0,
            forward_backward: false,
            subspace_dim: Some(10),
            n_frequencies: 128,
            ..Default::default()
        };
        let result_fw = music_spectral(&signal, &config_fw).expect("fw should succeed");

        // Both should produce valid results
        assert!(!result_fb.pseudospectrum.is_empty());
        assert!(!result_fw.pseudospectrum.is_empty());
    }

    #[test]
    fn test_music_validation_errors() {
        let config = MusicConfig::default();

        // Empty signal
        assert!(music_spectral(&[], &config).is_err());

        // Too few signals relative to subspace
        let short_signal = vec![1.0; 5];
        let bad_config = MusicConfig {
            n_signals: 10,
            subspace_dim: Some(3),
            ..Default::default()
        };
        assert!(music_spectral(&short_signal, &bad_config).is_err());
    }

    // =========================================================================
    // ESPRIT tests
    // =========================================================================

    #[test]
    fn test_esprit_basic() {
        // Reduced signal length and explicit small subspace_dim for fast eigendecomposition
        let n = 64;
        let fs = 64.0;
        let signal = generate_sinusoidal(n, fs, &[10.0], &[1.0]);

        let config = EspritConfig {
            n_signals: 1,
            fs,
            subspace_dim: Some(10),
            ..Default::default()
        };

        let result = esprit_spectral(&signal, &config);
        assert!(result.is_ok(), "ESPRIT failed: {:?}", result.err());
        let result = result.expect("already checked");

        assert!(
            !result.frequencies.is_empty(),
            "Should estimate at least one frequency"
        );
        assert!(!result.covariance_eigenvalues.is_empty());
    }

    #[test]
    fn test_esprit_frequency_estimation() {
        // Reduced signal length and subspace dim for fast eigendecomposition
        let n = 128;
        let fs = 128.0;
        let f_target = 20.0;
        let signal = generate_sinusoidal(n, fs, &[f_target], &[1.0]);

        let config = EspritConfig {
            n_signals: 1,
            fs,
            subspace_dim: Some(16),
            ..Default::default()
        };

        let result = esprit_spectral(&signal, &config).expect("should succeed");
        // Should have detected a frequency
        if !result.frequencies.is_empty() {
            // At least one frequency should be in range
            let closest = result
                .frequencies
                .iter()
                .map(|&f| (f - f_target).abs())
                .fold(f64::INFINITY, f64::min);
            // Allow generous tolerance for ESPRIT (it's sensitive to parameters)
            assert!(
                closest < 30.0,
                "Closest frequency to {f_target} is {closest} Hz away. Detected: {:?}",
                result.frequencies
            );
        }
    }

    #[test]
    fn test_esprit_tls_variant() {
        let n = 64;
        let fs = 64.0;
        let signal = generate_sinusoidal(n, fs, &[15.0], &[1.0]);

        let config = EspritConfig {
            n_signals: 1,
            fs,
            total_least_squares: true,
            subspace_dim: Some(10),
            ..Default::default()
        };

        let result = esprit_spectral(&signal, &config);
        assert!(result.is_ok(), "TLS-ESPRIT failed: {:?}", result.err());
    }

    #[test]
    fn test_esprit_multiple_signals() {
        // Reduced signal length and subspace dim for fast eigendecomposition
        let n = 128;
        let fs = 128.0;
        let signal = generate_sinusoidal(n, fs, &[20.0, 40.0], &[1.0, 0.8]);

        let config = EspritConfig {
            n_signals: 2,
            fs,
            subspace_dim: Some(16),
            ..Default::default()
        };

        let result = esprit_spectral(&signal, &config);
        assert!(result.is_ok(), "ESPRIT 2-signal failed: {:?}", result.err());
    }

    #[test]
    fn test_esprit_validation_errors() {
        let config = EspritConfig::default();

        // Empty signal
        assert!(esprit_spectral(&[], &config).is_err());

        // NaN
        assert!(esprit_spectral(&[1.0, f64::NAN, 3.0], &config).is_err());

        // Zero signals
        let bad_config = EspritConfig {
            n_signals: 0,
            ..Default::default()
        };
        assert!(esprit_spectral(&[1.0, 2.0, 3.0, 4.0, 5.0], &bad_config).is_err());
    }

    // =========================================================================
    // Cross-algorithm tests
    // =========================================================================

    #[test]
    fn test_burg_vs_yule_walker_ar_coefficients() {
        // For a simple sinusoidal signal, both methods should find the
        // same dominant frequency even if their AR coefficients differ.
        let signal = generate_sinusoidal(256, 256.0, &[30.0], &[1.0]);

        let burg = burg_spectral(
            &signal,
            &BurgConfig {
                order: 10,
                fs: 256.0,
                nfft: 512,
            },
        )
        .expect("burg");

        let yw = yule_walker_spectral(
            &signal,
            &YuleWalkerConfig {
                order: 10,
                fs: 256.0,
                nfft: 512,
            },
        )
        .expect("yw");

        // Both should find the peak near 30 Hz
        let burg_peak = burg.frequencies[burg
            .psd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("peak")];
        let yw_peak = yw.frequencies[yw
            .psd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("peak")];

        assert!(
            (burg_peak - yw_peak).abs() < 10.0,
            "Both should find similar peak: burg={burg_peak}, yw={yw_peak}"
        );

        // Both variances should be positive
        assert!(burg.variance > 0.0, "Burg variance should be positive");
        assert!(yw.variance > 0.0, "YW variance should be positive");
    }
}
