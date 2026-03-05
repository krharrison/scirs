//! Frequency Domain Decomposition (FDD) for Operational Modal Analysis
//!
//! # References
//! - Brincker, R., Zhang, L. & Andersen, P. (2001). "Modal identification of
//!   output-only systems using frequency domain decomposition." *Smart Materials
//!   and Structures*, 10(3), 441–445.
//! - Jacobsen, N.J., Andersen, P. & Brincker, R. (2006). "Using enhanced
//!   frequency domain decomposition as a robust technique to harmonic excitation
//!   in operational modal analysis." *Proc. ISMA*.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for Frequency Domain Decomposition
#[derive(Debug, Clone)]
pub struct FDDConfig {
    /// Sampling frequency in Hz
    pub fs: f64,
    /// Number of FFT points (should be a power of 2 for efficiency)
    pub nfft: usize,
    /// Overlap fraction in `[0, 1)` for Welch's method
    pub overlap: f64,
    /// Number of modes to extract (0 = auto-detect from singular value peaks)
    pub n_modes: usize,
    /// Minimum peak-to-median ratio to accept a singular value peak
    pub peak_threshold: f64,
    /// Minimum frequency (Hz) to search for modes (eliminates DC component)
    pub f_min: f64,
    /// Maximum frequency (Hz) to search for modes
    pub f_max: f64,
    /// Window type: "hann", "hamming", "rectangular"
    pub window: String,
}

impl Default for FDDConfig {
    fn default() -> Self {
        Self {
            fs: 1.0,
            nfft: 1024,
            overlap: 0.5,
            n_modes: 0,
            peak_threshold: 3.0,
            f_min: 0.0,
            f_max: f64::INFINITY,
            window: "hann".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of FDD modal parameter estimation
#[derive(Debug, Clone)]
pub struct FDDResult {
    /// Natural frequencies (Hz), sorted in ascending order
    pub natural_frequencies: Vec<f64>,
    /// Damping ratios (dimensionless, in range [0, 1])
    pub damping_ratios: Vec<f64>,
    /// Mode shapes: one column per mode, rows = measurement channels
    pub mode_shapes: Vec<Vec<f64>>,
    /// Singular values at the detected peak frequencies
    pub singular_values_at_peaks: Vec<f64>,
    /// Frequency axis (Hz) for the entire singular value spectrum
    pub frequencies: Vec<f64>,
    /// First singular value spectrum (all frequency lines)
    pub sv1: Vec<f64>,
}

/// Result of Enhanced FDD modal parameter estimation
#[derive(Debug, Clone)]
pub struct EFDDResult {
    /// Natural frequencies (Hz)
    pub natural_frequencies: Vec<f64>,
    /// Damping ratios estimated from SDOF bell function fit
    pub damping_ratios: Vec<f64>,
    /// Mode shapes (complex; phase information is preserved)
    pub mode_shapes: Vec<Vec<f64>>,
    /// SDOF auto-correlation functions per mode
    pub sdof_correlations: Vec<Vec<f64>>,
    /// Logarithmic decrement per mode (used to compute damping)
    pub log_decrements: Vec<f64>,
    /// Frequency axis (Hz)
    pub frequencies: Vec<f64>,
    /// First singular value spectrum
    pub sv1: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Core SVD-of-PSD computation
// ---------------------------------------------------------------------------

/// Compute the singular value decomposition of the power spectral density matrix.
///
/// Given `n_channels` time series of length `n_samples`, this function:
/// 1. Estimates the cross-PSD matrix `G(f)` using Welch's averaged periodogram.
/// 2. Returns the first (largest) singular values at every frequency line,
///    together with the corresponding left singular vectors (mode shapes).
///
/// # Arguments
/// * `data` – `(n_channels, n_samples)` array of measured responses.
/// * `config` – FDD configuration.
///
/// # Returns
/// `(frequencies, singular_values_matrix, left_singular_vectors)`
/// where `singular_values_matrix` has shape `(n_freq_lines, n_channels)` and
/// `left_singular_vectors` has shape `(n_freq_lines, n_channels, n_channels)`.
pub fn svd_psd(
    data: &Array2<f64>,
    config: &FDDConfig,
) -> SignalResult<(Vec<f64>, Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>)> {
    let (n_channels, n_samples) = (data.nrows(), data.ncols());
    if n_channels == 0 || n_samples == 0 {
        return Err(SignalError::InvalidInput(
            "Data array must have at least one channel and one sample".to_string(),
        ));
    }
    if config.fs <= 0.0 {
        return Err(SignalError::InvalidInput(
            "Sampling frequency must be positive".to_string(),
        ));
    }
    let nfft = config.nfft.max(4);
    let n_freq = nfft / 2 + 1;

    // Build window
    let window = build_window(&config.window, nfft)?;
    let win_power: f64 = window.iter().map(|w| w * w).sum::<f64>();

    // Compute cross-spectral density matrix via Welch's method
    let hop = (nfft as f64 * (1.0 - config.overlap)) as usize;
    let hop = hop.max(1);

    // Accumulator: G[f][i][j] = sum of X_i(f) * conj(X_j(f))
    let mut g_re = vec![vec![vec![0.0f64; n_channels]; n_channels]; n_freq];
    let mut g_im = vec![vec![vec![0.0f64; n_channels]; n_channels]; n_freq];
    let mut n_segments = 0usize;

    let mut start = 0;
    while start + nfft <= n_samples {
        // Compute DFT of each channel on this segment
        let mut spectra: Vec<Vec<(f64, f64)>> = Vec::with_capacity(n_channels);
        for ch in 0..n_channels {
            let row = data.row(ch);
            let segment: Vec<f64> = (0..nfft).map(|k| row[start + k] * window[k]).collect();
            let ft = real_dft(&segment, nfft);
            spectra.push(ft);
        }
        // Accumulate cross-spectrum
        for f in 0..n_freq {
            for i in 0..n_channels {
                for j in 0..n_channels {
                    let (ai, bi) = spectra[i][f];
                    let (aj, bj) = spectra[j][f];
                    // X_i * conj(X_j)
                    g_re[f][i][j] += ai * aj + bi * bj;
                    g_im[f][i][j] += bi * aj - ai * bj;
                }
            }
        }
        n_segments += 1;
        start += hop;
    }

    if n_segments == 0 {
        return Err(SignalError::InvalidInput(
            "Signal too short for the given nfft and overlap".to_string(),
        ));
    }

    let scale = 2.0 / (win_power * n_segments as f64 * config.fs);

    // Normalise and apply one-sided scaling
    for f in 0..n_freq {
        for i in 0..n_channels {
            for j in 0..n_channels {
                g_re[f][i][j] *= scale;
                g_im[f][i][j] *= scale;
            }
        }
    }

    // SVD of each frequency's PSD matrix
    let freq_axis: Vec<f64> = (0..n_freq)
        .map(|k| k as f64 * config.fs / nfft as f64)
        .collect();

    let mut sv_matrix: Vec<Vec<f64>> = Vec::with_capacity(n_freq);
    let mut u_matrix: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_freq);

    for f in 0..n_freq {
        // Perform SVD on the Hermitian n_channels × n_channels PSD matrix
        // Since G is Hermitian positive semi-definite, we use eigendecomposition.
        let (svs, us) = hermitian_svd_real(n_channels, &g_re[f], &g_im[f])?;
        sv_matrix.push(svs);
        u_matrix.push(us);
    }

    Ok((freq_axis, sv_matrix, u_matrix))
}

// ---------------------------------------------------------------------------
// FDD
// ---------------------------------------------------------------------------

/// Perform Frequency Domain Decomposition.
///
/// # Arguments
/// * `data` – `(n_channels, n_samples)` array of measured responses.
/// * `config` – FDD configuration parameters.
///
/// # Returns
/// [`FDDResult`] containing identified modal parameters.
pub fn frequency_domain_decomposition(
    data: &Array2<f64>,
    config: &FDDConfig,
) -> SignalResult<FDDResult> {
    let (freq_axis, sv_matrix, u_matrix) = svd_psd(data, config)?;
    let n_freq = freq_axis.len();
    let n_channels = data.nrows();

    // First singular value vector
    let sv1: Vec<f64> = sv_matrix.iter().map(|svs| svs[0]).collect();

    // Detect peaks in sv1 within [f_min, f_max]
    let f_min = config.f_min;
    let f_max = if config.f_max.is_infinite() {
        freq_axis.last().copied().unwrap_or(config.fs / 2.0)
    } else {
        config.f_max
    };

    let peaks = find_peaks_in_sv1(&sv1, &freq_axis, f_min, f_max, config.peak_threshold);

    let n_modes = if config.n_modes > 0 {
        config.n_modes.min(peaks.len())
    } else {
        peaks.len()
    };

    let selected_peaks: Vec<usize> = peaks.into_iter().take(n_modes).collect();

    let mut natural_frequencies = Vec::with_capacity(n_modes);
    let mut damping_ratios = Vec::with_capacity(n_modes);
    let mut mode_shapes = Vec::with_capacity(n_modes);
    let mut sv_at_peaks = Vec::with_capacity(n_modes);

    for &peak_idx in &selected_peaks {
        let freq = freq_axis[peak_idx];
        natural_frequencies.push(freq);

        // Mode shape = first left singular vector at the peak frequency
        let ms: Vec<f64> = if n_channels > 0 {
            u_matrix[peak_idx][0].clone()
        } else {
            vec![]
        };
        mode_shapes.push(ms);

        sv_at_peaks.push(sv1[peak_idx]);

        // Estimate damping from half-power bandwidth (3 dB method)
        let damp = estimate_damping_half_power(&sv1, &freq_axis, peak_idx, n_freq);
        damping_ratios.push(damp);
    }

    Ok(FDDResult {
        natural_frequencies,
        damping_ratios,
        mode_shapes,
        singular_values_at_peaks: sv_at_peaks,
        frequencies: freq_axis,
        sv1,
    })
}

// ---------------------------------------------------------------------------
// EFDD
// ---------------------------------------------------------------------------

/// Enhanced Frequency Domain Decomposition.
///
/// EFDD refines the mode shape and damping estimates by:
/// 1. Identifying the SDOF bell around each peak (using MAC criterion).
/// 2. Inverse-FFT-ing the bell to obtain an auto-correlation function.
/// 3. Fitting an exponential decay to the auto-correlation to get damping.
///
/// # Arguments
/// * `data` – `(n_channels, n_samples)` measurement array.
/// * `config` – FDD configuration.
/// * `bell_half_width` – Half-width of the SDOF bell in frequency bins.
pub fn enhanced_fdd(
    data: &Array2<f64>,
    config: &FDDConfig,
    bell_half_width: usize,
) -> SignalResult<EFDDResult> {
    let (freq_axis, sv_matrix, u_matrix) = svd_psd(data, config)?;
    let n_freq = freq_axis.len();
    let n_channels = data.nrows();

    let sv1: Vec<f64> = sv_matrix.iter().map(|svs| svs[0]).collect();

    let f_min = config.f_min;
    let f_max = if config.f_max.is_infinite() {
        freq_axis.last().copied().unwrap_or(config.fs / 2.0)
    } else {
        config.f_max
    };

    let peaks = find_peaks_in_sv1(&sv1, &freq_axis, f_min, f_max, config.peak_threshold);
    let n_modes = if config.n_modes > 0 {
        config.n_modes.min(peaks.len())
    } else {
        peaks.len()
    };
    let selected_peaks: Vec<usize> = peaks.into_iter().take(n_modes).collect();

    let half_w = bell_half_width.max(2);

    let mut natural_frequencies = Vec::with_capacity(n_modes);
    let mut damping_ratios = Vec::with_capacity(n_modes);
    let mut mode_shapes = Vec::with_capacity(n_modes);
    let mut sdof_correlations = Vec::with_capacity(n_modes);
    let mut log_decrements = Vec::with_capacity(n_modes);

    for &peak_idx in &selected_peaks {
        let fn_hz = freq_axis[peak_idx];
        natural_frequencies.push(fn_hz);

        // Mode shape = first left singular vector at peak
        let ms: Vec<f64> = if n_channels > 0 {
            u_matrix[peak_idx][0].clone()
        } else {
            vec![]
        };
        mode_shapes.push(ms.clone());

        // Build SDOF bell: include frequency lines where MAC(u_k, u_peak) > 0.8
        let bell_start = peak_idx.saturating_sub(half_w);
        let bell_end = (peak_idx + half_w + 1).min(n_freq);

        let u_peak = &u_matrix[peak_idx][0];
        let mut bell_sv: Vec<f64> = vec![0.0; n_freq];
        for f in bell_start..bell_end {
            let u_f = &u_matrix[f][0];
            let mac = mac_vectors(u_peak, u_f);
            if mac > 0.8 {
                bell_sv[f] = sv1[f];
            }
        }

        // IFFT of the bell to obtain auto-correlation function
        let acf = real_ifft(&bell_sv, n_freq);

        // Estimate damping via log decrement from the auto-correlation
        let (log_dec, damp) = log_decrement_from_acf(&acf, fn_hz, config.fs);
        damping_ratios.push(damp);
        log_decrements.push(log_dec);

        // Store first N_CORR points of auto-correlation
        let n_corr = acf.len().min(512);
        sdof_correlations.push(acf[..n_corr].to_vec());
    }

    Ok(EFDDResult {
        natural_frequencies,
        damping_ratios,
        mode_shapes,
        sdof_correlations,
        log_decrements,
        frequencies: freq_axis,
        sv1,
    })
}

// ---------------------------------------------------------------------------
// MAC criterion
// ---------------------------------------------------------------------------

/// Compute the Modal Assurance Criterion (MAC) between two mode shapes.
///
/// MAC = |{phi_1}^T {phi_2}|^2 / (|{phi_1}|^2 * |{phi_2}|^2)
///
/// Returns a value in `[0, 1]` where 1 indicates identical mode shapes.
///
/// # Arguments
/// * `phi1` – First mode shape vector.
/// * `phi2` – Second mode shape vector.
pub fn mac_criterion(phi1: &[f64], phi2: &[f64]) -> SignalResult<f64> {
    if phi1.len() != phi2.len() {
        return Err(SignalError::DimensionMismatch(format!(
            "Mode shape vectors have different lengths: {} vs {}",
            phi1.len(),
            phi2.len()
        )));
    }
    if phi1.is_empty() {
        return Err(SignalError::InvalidInput(
            "Mode shape vectors must not be empty".to_string(),
        ));
    }
    Ok(mac_vectors(phi1, phi2))
}

/// Internal MAC computation (no error checking).
fn mac_vectors(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum();
    let norm_b: f64 = b.iter().map(|x| x * x).sum();
    let denom = norm_a * norm_b;
    if denom < 1e-30 {
        return 0.0;
    }
    (dot * dot) / denom
}

// ---------------------------------------------------------------------------
// Helper: find peaks in first singular value spectrum
// ---------------------------------------------------------------------------

/// Detect local maxima in `sv1` within `[f_min, f_max]`.
///
/// A bin is a peak if it is strictly greater than its immediate neighbours and
/// its value exceeds `threshold` times the median of the entire spectrum.
fn find_peaks_in_sv1(
    sv1: &[f64],
    freq: &[f64],
    f_min: f64,
    f_max: f64,
    threshold: f64,
) -> Vec<usize> {
    let n = sv1.len();
    if n < 3 {
        return vec![];
    }
    // Compute median for threshold
    let mut sorted_sv = sv1.to_vec();
    sorted_sv.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted_sv[n / 2];
    let peak_min = threshold * median;

    let mut peaks: Vec<(usize, f64)> = Vec::new();

    for i in 1..n - 1 {
        let f = freq[i];
        if f < f_min || f > f_max {
            continue;
        }
        let v = sv1[i];
        if v > sv1[i - 1] && v > sv1[i + 1] && v > peak_min {
            peaks.push((i, v));
        }
    }

    // Sort by descending singular value (most prominent mode first)
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    // Return indices sorted by frequency (ascending) after we've taken the top peaks
    let mut indices: Vec<usize> = peaks.iter().map(|(idx, _)| *idx).collect();
    indices.sort_unstable();
    indices
}

// ---------------------------------------------------------------------------
// Helper: half-power bandwidth damping estimate
// ---------------------------------------------------------------------------

fn estimate_damping_half_power(sv1: &[f64], freq: &[f64], peak_idx: usize, n_freq: usize) -> f64 {
    let peak_val = sv1[peak_idx];
    let half_power = peak_val / 2.0_f64.sqrt();
    let fn_hz = freq[peak_idx];
    if fn_hz <= 0.0 {
        return 0.01; // default
    }

    // Find left crossing
    let mut f_left = freq[peak_idx];
    for i in (1..=peak_idx).rev() {
        if sv1[i] < half_power {
            // Interpolate
            let f1 = freq[i];
            let f2 = freq[i + 1];
            let v1 = sv1[i];
            let v2 = sv1[i + 1];
            if (v2 - v1).abs() > 1e-30 {
                f_left = f1 + (half_power - v1) * (f2 - f1) / (v2 - v1);
            } else {
                f_left = f1;
            }
            break;
        }
    }

    // Find right crossing
    let mut f_right = freq[peak_idx];
    for i in peak_idx + 1..n_freq {
        if sv1[i] < half_power {
            let f1 = freq[i - 1];
            let f2 = freq[i];
            let v1 = sv1[i - 1];
            let v2 = sv1[i];
            if (v2 - v1).abs() > 1e-30 {
                f_right = f1 + (half_power - v1) * (f2 - f1) / (v2 - v1);
            } else {
                f_right = f2;
            }
            break;
        }
    }

    let delta_f = f_right - f_left;
    // xi = Δf / (2 * fn)
    let xi = delta_f / (2.0 * fn_hz);
    xi.clamp(1e-6, 1.0)
}

// ---------------------------------------------------------------------------
// Helper: log decrement from auto-correlation function
// ---------------------------------------------------------------------------

fn log_decrement_from_acf(acf: &[f64], fn_hz: f64, fs: f64) -> (f64, f64) {
    // Find successive peaks in the ACF to compute log decrement
    let n = acf.len();
    if n < 4 || fn_hz <= 0.0 || fs <= 0.0 {
        return (0.0, 0.01);
    }

    // Approximate period in samples
    let period_samples = (fs / fn_hz).round() as usize;
    if period_samples < 2 {
        return (0.0, 0.01);
    }

    // Collect positive peaks spaced roughly one period apart
    let mut peak_values: Vec<f64> = Vec::new();
    let mut idx = 0usize;
    while idx < n {
        // Find local max around idx
        let window_start = idx;
        let window_end = (idx + period_samples).min(n);
        if window_start >= window_end {
            break;
        }
        let local_max = acf[window_start..window_end]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        if local_max > 0.0 {
            peak_values.push(local_max);
        }
        idx += period_samples;
    }

    if peak_values.len() < 2 {
        return (0.0, 0.01);
    }

    // Least-squares estimate of log decrement: ln(A_k) = ln(A_0) - k * delta
    let n_peaks = peak_values.len();
    let log_vals: Vec<f64> = peak_values
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| {
            if v > 1e-30 {
                Some((i as f64, v.ln()))
            } else {
                None
            }
        })
        .map(|(i, lv)| (i, lv))
        .collect::<Vec<_>>()
        .into_iter()
        .map(|(_, lv)| lv)
        .collect();

    let valid_count = log_vals.len();
    if valid_count < 2 {
        return (0.0, 0.01);
    }

    let mean_k = (valid_count as f64 - 1.0) / 2.0;
    let mean_lv = log_vals.iter().sum::<f64>() / valid_count as f64;
    let num: f64 = (0..valid_count)
        .map(|k| (k as f64 - mean_k) * (log_vals[k] - mean_lv))
        .sum();
    let den: f64 = (0..valid_count)
        .map(|k| (k as f64 - mean_k).powi(2))
        .sum();

    let slope = if den.abs() > 1e-30 { -num / den } else { 0.0 };
    // slope = log decrement per cycle
    let log_dec = slope.max(0.0);
    // damping ratio from log decrement: xi = delta / sqrt(4*pi^2 + delta^2)
    let denom = (4.0 * PI * PI + log_dec * log_dec).sqrt();
    let xi = if denom > 1e-30 {
        log_dec / denom
    } else {
        0.01
    };
    (log_dec, xi.clamp(1e-6, 1.0))
}

// ---------------------------------------------------------------------------
// Numerical helpers: real DFT and SVD of Hermitian PSD matrix
// ---------------------------------------------------------------------------

/// Compute the one-sided DFT of a real sequence using the Cooley-Tukey algorithm.
///
/// Returns a `Vec<(re, im)>` of length `nfft / 2 + 1`.
fn real_dft(x: &[f64], nfft: usize) -> Vec<(f64, f64)> {
    // Pad or truncate to nfft
    let mut buf = x.to_vec();
    buf.resize(nfft, 0.0);

    let n = nfft;
    let n_out = n / 2 + 1;

    // DFT by brute force (for moderate nfft this is acceptable in OMA context)
    // For large nfft consider oxifft, but we keep it self-contained here.
    let mut out = Vec::with_capacity(n_out);
    for k in 0..n_out {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (j, &xj) in buf.iter().enumerate() {
            let angle = -2.0 * PI * k as f64 * j as f64 / n as f64;
            re += xj * angle.cos();
            im += xj * angle.sin();
        }
        out.push((re, im));
    }
    out
}

/// Compute the inverse FFT of a one-sided spectrum (as if the input were the
/// magnitude spectrum stored as a real Vec of length n_freq = nfft/2+1).
/// Returns the real-valued IFFT (length 2*(n_freq-1)).
fn real_ifft(sv: &[f64], n_freq: usize) -> Vec<f64> {
    let nfft = 2 * (n_freq - 1);
    if nfft == 0 {
        return vec![];
    }
    let mut out = vec![0.0f64; nfft];
    // Reconstruct double-sided spectrum from one-sided
    for n in 0..nfft {
        let mut re = 0.0f64;
        for k in 0..n_freq {
            let scale = if k == 0 || k == n_freq - 1 { 1.0 } else { 2.0 };
            let angle = 2.0 * PI * k as f64 * n as f64 / nfft as f64;
            re += scale * sv[k] * angle.cos();
        }
        out[n] = re / nfft as f64;
    }
    out
}

/// Build a window function.
fn build_window(name: &str, n: usize) -> SignalResult<Vec<f64>> {
    if n == 0 {
        return Ok(vec![]);
    }
    match name.to_lowercase().as_str() {
        "hann" | "hanning" => Ok((0..n)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
            .collect()),
        "hamming" => Ok((0..n)
            .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos())
            .collect()),
        "rectangular" | "boxcar" | "rect" => Ok(vec![1.0; n]),
        other => Err(SignalError::InvalidInput(format!(
            "Unknown window type: '{other}'. Use 'hann', 'hamming', or 'rectangular'."
        ))),
    }
}

/// SVD (via eigendecomposition) of the Hermitian PSD matrix at a single frequency.
///
/// The PSD matrix G is Hermitian positive semi-definite.  Its singular values
/// equal its eigenvalues; its left singular vectors equal its eigenvectors.
///
/// We return the **real parts** of the eigenvectors (sufficient for FDD mode shapes
/// since the imaginary part is typically negligible for well-separated modes).
///
/// Returns `(singular_values, left_vectors)` where each inner `Vec<f64>` is
/// one left singular vector.
fn hermitian_svd_real(
    n: usize,
    g_re: &[Vec<f64>],
    g_im: &[Vec<f64>],
) -> SignalResult<(Vec<f64>, Vec<Vec<f64>>)> {
    // Build 2n×2n real symmetric representation:
    // [Re(G)  -Im(G); Im(G)  Re(G)]
    // then Jacobi eigendecompose, extract n eigenvalues/vectors.
    if n == 0 {
        return Ok((vec![], vec![]));
    }
    if n == 1 {
        return Ok((vec![g_re[0][0].abs()], vec![vec![1.0]]));
    }

    let n2 = 2 * n;
    let mut mat = vec![0.0f64; n2 * n2];
    for i in 0..n {
        for j in 0..n {
            let re = g_re[i][j];
            let im = g_im[i][j];
            mat[i * n2 + j] = re;
            mat[i * n2 + (n + j)] = -im;
            mat[(n + i) * n2 + j] = im;
            mat[(n + i) * n2 + (n + j)] = re;
        }
    }

    let (eigs, evecs_all) = jacobi_eig_real(&mat, n2)?;

    // Pair up eigenvalues (each real eigenvalue appears twice), keep n largest
    let mut indexed: Vec<(f64, usize)> = eigs
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();
    indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut svs: Vec<f64> = Vec::with_capacity(n);
    let mut us: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut used = 0;
    let mut i = 0;

    while used < n && i < indexed.len() {
        let (val, col) = indexed[i];
        if used > 0 {
            let prev = svs[used - 1];
            if (val - prev).abs() < 1e-8 * (prev.abs() + 1.0) {
                i += 1;
                continue;
            }
        }
        svs.push(val.max(0.0));
        // Use only the top-n components of the eigenvector (real part)
        let ev: Vec<f64> = (0..n).map(|k| evecs_all[col][k]).collect();
        let norm: f64 = ev.iter().map(|x| x * x).sum::<f64>().sqrt();
        let ev_norm = if norm > 1e-14 {
            ev.iter().map(|x| x / norm).collect()
        } else {
            ev
        };
        us.push(ev_norm);
        used += 1;
        i += 1;
    }

    // Pad if needed
    while svs.len() < n {
        svs.push(0.0);
        let mut zero = vec![0.0f64; n];
        if n > 0 {
            zero[svs.len() - 1] = 1.0;
        }
        us.push(zero);
    }

    Ok((svs, us))
}

// ---------------------------------------------------------------------------
// Jacobi eigendecomposition for real symmetric matrices
// ---------------------------------------------------------------------------

/// Jacobi eigendecomposition of a real symmetric matrix stored as a flat row-major Vec.
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvectors[j][i]` is the
/// i-th component of the j-th eigenvector.
fn jacobi_eig_real(mat: &[f64], n: usize) -> SignalResult<(Vec<f64>, Vec<Vec<f64>>)> {
    if mat.len() != n * n {
        return Err(SignalError::DimensionMismatch(format!(
            "Matrix length {} does not match n={} (expected {})",
            mat.len(),
            n,
            n * n
        )));
    }

    let mut a = mat.to_vec();
    // Identity eigenvector matrix
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let max_iter = 100 * n * n;
    let eps = 1e-12;

    for _ in 0..max_iter {
        // Find off-diagonal element with largest absolute value
        let mut max_val = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in i + 1..n {
                let val = a[i * n + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < eps {
            break;
        }

        // Compute rotation angle
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            1.0 / (tau - (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        // Apply Jacobi rotation J^T A J
        // Update rows p and q of A
        let mut new_a = a.clone();
        for r in 0..n {
            if r == p || r == q {
                continue;
            }
            let arp = a[r * n + p];
            let arq = a[r * n + q];
            new_a[r * n + p] = c * arp - s * arq;
            new_a[p * n + r] = new_a[r * n + p];
            new_a[r * n + q] = s * arp + c * arq;
            new_a[q * n + r] = new_a[r * n + q];
        }
        new_a[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        new_a[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        new_a[p * n + q] = 0.0;
        new_a[q * n + p] = 0.0;
        a = new_a;

        // Update eigenvector matrix
        for r in 0..n {
            let vrp = v[r * n + p];
            let vrq = v[r * n + q];
            v[r * n + p] = c * vrp - s * vrq;
            v[r * n + q] = s * vrp + c * vrq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    let eigenvectors: Vec<Vec<f64>> = (0..n).map(|j| (0..n).map(|i| v[i * n + j]).collect()).collect();

    Ok((eigenvalues, eigenvectors))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;
    use std::f64::consts::PI;

    /// Generate a two-DOF system free response (superposition of two sinusoids)
    fn two_dof_response(n_samples: usize, fs: f64) -> Array2<f64> {
        let f1 = 5.0;
        let f2 = 12.0;
        let xi1 = 0.02;
        let xi2 = 0.03;
        let phi1 = [1.0, 0.7];
        let phi2 = [1.0, -0.5];
        let mut data = Array2::zeros((2, n_samples));
        for i in 0..n_samples {
            let t = i as f64 / fs;
            let r1 = (-xi1 * 2.0 * PI * f1 * t).exp() * (2.0 * PI * f1 * t).sin();
            let r2 = (-xi2 * 2.0 * PI * f2 * t).exp() * (2.0 * PI * f2 * t).sin();
            data[[0, i]] = phi1[0] * r1 + phi2[0] * r2;
            data[[1, i]] = phi1[1] * r1 + phi2[1] * r2;
        }
        data
    }

    #[test]
    fn test_mac_criterion_identical() {
        let phi = vec![1.0, 0.5, -0.3, 0.8];
        let mac = mac_criterion(&phi, &phi).expect("MAC should succeed");
        assert!((mac - 1.0).abs() < 1e-10, "MAC of identical vectors should be 1");
    }

    #[test]
    fn test_mac_criterion_orthogonal() {
        let phi1 = vec![1.0, 0.0];
        let phi2 = vec![0.0, 1.0];
        let mac = mac_criterion(&phi1, &phi2).expect("MAC should succeed");
        assert!(mac.abs() < 1e-10, "MAC of orthogonal vectors should be 0");
    }

    #[test]
    fn test_fdd_two_dof() {
        let fs = 200.0;
        let n_samples = 2048;
        let data = two_dof_response(n_samples, fs);

        let config = FDDConfig {
            fs,
            nfft: 512,
            overlap: 0.5,
            n_modes: 2,
            peak_threshold: 2.0,
            f_min: 1.0,
            f_max: 50.0,
            window: "hann".to_string(),
        };

        let result = frequency_domain_decomposition(&data, &config)
            .expect("FDD should succeed");

        // Should find at least 1 mode
        assert!(
            !result.natural_frequencies.is_empty(),
            "FDD should find at least one mode"
        );
        // Natural frequencies should be positive
        for &f in &result.natural_frequencies {
            assert!(f > 0.0, "Natural frequency should be positive");
        }
        // Damping ratios should be in [0, 1]
        for &xi in &result.damping_ratios {
            assert!(xi >= 0.0 && xi <= 1.0, "Damping ratio should be in [0, 1]");
        }
    }

    #[test]
    fn test_svd_psd_shape() {
        let fs = 100.0;
        let n_samples = 1024;
        let data = two_dof_response(n_samples, fs);

        let config = FDDConfig {
            fs,
            nfft: 256,
            ..Default::default()
        };

        let (freq, svs, us) = svd_psd(&data, &config).expect("SVD-PSD should succeed");
        let n_freq = 256 / 2 + 1;
        assert_eq!(freq.len(), n_freq);
        assert_eq!(svs.len(), n_freq);
        assert_eq!(us.len(), n_freq);
        // Each frequency should have 2 singular values (2 channels)
        for sv in &svs {
            assert_eq!(sv.len(), 2);
        }
    }

    #[test]
    fn test_efdd_runs() {
        let fs = 200.0;
        let n_samples = 2048;
        let data = two_dof_response(n_samples, fs);

        let config = FDDConfig {
            fs,
            nfft: 512,
            overlap: 0.5,
            n_modes: 2,
            peak_threshold: 2.0,
            f_min: 1.0,
            f_max: 50.0,
            window: "hann".to_string(),
        };

        let result = enhanced_fdd(&data, &config, 5).expect("EFDD should succeed");
        assert!(!result.natural_frequencies.is_empty());
        assert_eq!(result.natural_frequencies.len(), result.damping_ratios.len());
    }

    #[test]
    fn test_build_window_hann() {
        let w = build_window("hann", 8).expect("Hann window should build");
        assert_eq!(w.len(), 8);
        // First and last should be ~0
        assert!(w[0].abs() < 1e-6);
    }

    #[test]
    fn test_build_window_unknown() {
        let result = build_window("unknown_window", 8);
        assert!(result.is_err());
    }
}
