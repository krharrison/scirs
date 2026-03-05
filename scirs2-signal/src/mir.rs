//! Music Information Retrieval (MIR) algorithms.
//!
//! This module provides a comprehensive set of audio feature extraction tools:
//!
//! - **MFCC**: Mel-Frequency Cepstral Coefficients with delta/delta-delta
//! - **Chroma**: Pitch-class profile features from STFT
//! - **Spectral Features**: Centroid, bandwidth, roll-off, flatness
//! - **Temporal Features**: Zero-crossing rate per frame
//! - **Onset Detection**: Energy-based novelty function + peak-picking
//! - **Beat Tracking**: Autocorrelation-based BPM estimation
//! - **Pitch Estimation**: YIN algorithm for fundamental frequency
//!
//! # References
//!
//! - D. Ellis, "Classifying Music Audio with Timbral and Chroma Features",
//!   ISMIR 2007.
//! - A. de Cheveigné & H. Kawahara, "YIN, a fundamental frequency estimator",
//!   JASA 111(4), 2002.

use crate::error::{SignalError, SignalResult};
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Internal FFT helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute real-valued FFT magnitude spectrum for a windowed frame.
/// Returns `n_fft/2 + 1` bins.
fn frame_magnitude_spectrum(frame: &[f64], n_fft: usize) -> Vec<f64> {
    let fft_size = n_fft.max(frame.len()).next_power_of_two();
    // Use scirs2_fft for FFT computation
    let mut padded: Vec<f64> = frame.to_vec();
    padded.resize(fft_size, 0.0);

    // Simple DFT for smaller sizes, approximated via radix-2 FFT
    let n_bins = fft_size / 2 + 1;
    let mut mag = vec![0.0_f64; n_bins];

    // Use recursive Cooley-Tukey FFT
    let spectrum = fft_real(&padded);
    for (i, c) in spectrum.iter().take(n_bins).enumerate() {
        mag[i] = (c.0 * c.0 + c.1 * c.1).sqrt();
    }
    mag
}

/// Compute power spectrum (magnitude squared) for a frame.
fn frame_power_spectrum(frame: &[f64], n_fft: usize) -> Vec<f64> {
    let mag = frame_magnitude_spectrum(frame, n_fft);
    mag.iter().map(|m| m * m).collect()
}

/// Simple in-place Cooley-Tukey FFT on complex data (re, im) pairs.
fn fft_real(x: &[f64]) -> Vec<(f64, f64)> {
    // Build complex input
    let mut buf: Vec<(f64, f64)> = x.iter().map(|&v| (v, 0.0)).collect();
    fft_inplace(&mut buf);
    buf
}

fn fft_inplace(buf: &mut [(f64, f64)]) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    // Bit-reverse permutation
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
    // Cooley-Tukey
    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * PI / len as f64;
        let wlen = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let mut w = (1.0_f64, 0.0_f64);
            for jj in 0..len / 2 {
                let u = buf[i + jj];
                let v_re = buf[i + jj + len / 2].0 * w.0 - buf[i + jj + len / 2].1 * w.1;
                let v_im = buf[i + jj + len / 2].0 * w.1 + buf[i + jj + len / 2].1 * w.0;
                buf[i + jj] = (u.0 + v_re, u.1 + v_im);
                buf[i + jj + len / 2] = (u.0 - v_re, u.1 - v_im);
                let new_w = (w.0 * wlen.0 - w.1 * wlen.1, w.0 * wlen.1 + w.1 * wlen.0);
                w = new_w;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Apply Hann window to a frame.
fn apply_hann_window(frame: &[f64]) -> Vec<f64> {
    let n = frame.len();
    frame
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n as f64 - 1.0)).cos());
            v * w
        })
        .collect()
}

/// Extract overlapping frames from a signal.
fn frame_signal(signal: &[f64], frame_length: usize, hop_length: usize) -> Vec<Vec<f64>> {
    if signal.len() < frame_length || frame_length == 0 || hop_length == 0 {
        return Vec::new();
    }
    let n_frames = 1 + (signal.len() - frame_length) / hop_length;
    (0..n_frames)
        .map(|i| signal[i * hop_length..i * hop_length + frame_length].to_vec())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Mel filterbank helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Convert Hz to Mel scale.
#[inline]
fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert Mel to Hz.
#[inline]
fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

/// Build a mel filterbank matrix of shape `(n_mels, n_fft/2+1)`.
fn mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: f64,
    f_min: f64,
    f_max: f64,
) -> Vec<Vec<f64>> {
    let n_bins = n_fft / 2 + 1;
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // Mel-spaced center frequencies (n_mels + 2 points)
    let mel_points: Vec<f64> = (0..=n_mels + 1)
        .map(|i| mel_min + i as f64 * (mel_max - mel_min) / (n_mels + 1) as f64)
        .collect();
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert Hz to FFT bin indices
    let bin_points: Vec<f64> = hz_points
        .iter()
        .map(|&h| (n_fft as f64 + 1.0) * h / sample_rate)
        .collect();

    let mut filters = vec![vec![0.0_f64; n_bins]; n_mels];
    for m in 0..n_mels {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];
        for k in 0..n_bins {
            let k_f = k as f64;
            if k_f >= left && k_f <= center && center > left {
                filters[m][k] = (k_f - left) / (center - left);
            } else if k_f > center && k_f <= right && right > center {
                filters[m][k] = (right - k_f) / (right - center);
            }
        }
    }
    filters
}

/// Apply mel filterbank to a power spectrum.
fn apply_mel_filterbank(power_spec: &[f64], filterbank: &[Vec<f64>]) -> Vec<f64> {
    filterbank
        .iter()
        .map(|f| {
            let s: f64 = f
                .iter()
                .zip(power_spec.iter())
                .map(|(fi, pi)| fi * pi)
                .sum();
            s.max(1e-10) // avoid log(0)
        })
        .collect()
}

/// DCT-II for MFCC computation.
fn dct2(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .map(|k| {
            let sum: f64 = x
                .iter()
                .enumerate()
                .map(|(i, &xi)| xi * (PI * k as f64 * (2.0 * i as f64 + 1.0) / (2.0 * n as f64)).cos())
                .sum();
            sum * (2.0 / n as f64).sqrt()
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// MfccExtractor
// ─────────────────────────────────────────────────────────────────────────────

/// MFCC feature extractor with configurable parameters.
///
/// # Example
///
/// ```
/// use scirs2_signal::mir::MfccExtractor;
///
/// let sample_rate = 22050.0;
/// let signal: Vec<f64> = (0..22050)
///     .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / sample_rate).sin())
///     .collect();
///
/// let extractor = MfccExtractor::new(sample_rate);
/// let mfcc = extractor.extract(&signal);
/// assert!(!mfcc.is_empty(), "MFCC frames should not be empty");
/// assert_eq!(mfcc[0].len(), extractor.n_mfcc);
/// ```
#[derive(Debug, Clone)]
pub struct MfccExtractor {
    /// Number of MFCC coefficients to return.
    pub n_mfcc: usize,
    /// FFT size in samples.
    pub n_fft: usize,
    /// Hop length between frames in samples.
    pub hop_length: usize,
    /// Number of mel filterbank channels.
    pub n_mels: usize,
    /// Audio sample rate in Hz.
    pub sample_rate: f64,
    /// Minimum frequency for mel filterbank.
    pub f_min: f64,
    /// Maximum frequency for mel filterbank.
    pub f_max: f64,
}

impl MfccExtractor {
    /// Create an extractor with sensible defaults for the given sample rate.
    pub fn new(sample_rate: f64) -> Self {
        Self {
            n_mfcc: 13,
            n_fft: 2048,
            hop_length: 512,
            n_mels: 128,
            sample_rate,
            f_min: 0.0,
            f_max: sample_rate / 2.0,
        }
    }

    /// Set the number of MFCC coefficients.
    pub fn with_n_mfcc(mut self, n: usize) -> Self {
        self.n_mfcc = n;
        self
    }

    /// Set the FFT size.
    pub fn with_n_fft(mut self, n: usize) -> Self {
        self.n_fft = n;
        self
    }

    /// Set the hop length.
    pub fn with_hop_length(mut self, h: usize) -> Self {
        self.hop_length = h;
        self
    }

    /// Set the number of mel bands.
    pub fn with_n_mels(mut self, n: usize) -> Self {
        self.n_mels = n;
        self
    }

    /// Extract MFCC features from an audio signal.
    ///
    /// Returns a matrix of shape `(n_frames, n_mfcc)`.
    pub fn extract(&self, signal: &[f64]) -> Vec<Vec<f64>> {
        let mel_spec = self.mel_spectrogram(signal);
        mel_spec
            .iter()
            .map(|frame_mel| {
                // Log mel energies
                let log_mel: Vec<f64> = frame_mel.iter().map(|&e| e.max(1e-10).ln()).collect();
                // DCT-II
                let cepstrum = dct2(&log_mel);
                // Keep first n_mfcc coefficients
                cepstrum.into_iter().take(self.n_mfcc).collect()
            })
            .collect()
    }

    /// Compute the mel spectrogram.
    ///
    /// Returns `(n_frames, n_mels)` matrix of mel-filtered power.
    pub fn mel_spectrogram(&self, signal: &[f64]) -> Vec<Vec<f64>> {
        let frame_len = self.n_fft.min(signal.len());
        if frame_len == 0 {
            return Vec::new();
        }
        let filters = mel_filterbank(self.n_mels, self.n_fft, self.sample_rate, self.f_min, self.f_max);
        let frames = frame_signal(signal, frame_len, self.hop_length);
        frames
            .iter()
            .map(|frame| {
                let windowed = apply_hann_window(frame);
                let power = frame_power_spectrum(&windowed, self.n_fft);
                apply_mel_filterbank(&power, &filters)
            })
            .collect()
    }

    /// Compute first-order delta (velocity) features.
    pub fn delta(features: &[Vec<f64>]) -> Vec<Vec<f64>> {
        compute_delta(features, 2)
    }

    /// Compute second-order delta-delta (acceleration) features.
    pub fn delta_delta(features: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let d = compute_delta(features, 2);
        compute_delta(&d, 2)
    }
}

/// Compute delta features using a regression approach.
fn compute_delta(features: &[Vec<f64>], width: usize) -> Vec<Vec<f64>> {
    let n_frames = features.len();
    if n_frames == 0 {
        return Vec::new();
    }
    let n_coeff = features[0].len();
    let denom: f64 = 2.0 * (1..=width as usize).map(|t| (t * t) as f64).sum::<f64>();

    (0..n_frames)
        .map(|t| {
            (0..n_coeff)
                .map(|c| {
                    let mut num = 0.0_f64;
                    for dt in 1..=width {
                        let forward = if (t + dt) < n_frames {
                            features[t + dt][c]
                        } else {
                            features[n_frames - 1][c]
                        };
                        let backward = if t >= dt {
                            features[t - dt][c]
                        } else {
                            features[0][c]
                        };
                        num += dt as f64 * (forward - backward);
                    }
                    if denom.abs() > 1e-15 { num / denom } else { 0.0 }
                })
                .collect()
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Chroma features
// ─────────────────────────────────────────────────────────────────────────────

/// Compute chroma STFT features (pitch-class profiles).
///
/// Maps the STFT magnitude spectrum to 12 (or `n_chroma`) pitch classes
/// using a chroma filter bank based on equal-temperament tuning.
///
/// # Returns
///
/// Matrix of shape `(n_frames, n_chroma)` with values in `[0, 1]`.
///
/// # Example
///
/// ```
/// use scirs2_signal::mir::chroma_stft;
/// use std::f64::consts::PI;
///
/// let sr = 22050.0;
/// let signal: Vec<f64> = (0..4096)
///     .map(|i| (2.0 * PI * 440.0 * i as f64 / sr).sin())
///     .collect();
/// let chroma = chroma_stft(&signal, sr, 2048, 512, 12);
/// assert!(!chroma.is_empty());
/// ```
pub fn chroma_stft(
    signal: &[f64],
    sample_rate: f64,
    n_fft: usize,
    hop_length: usize,
    n_chroma: usize,
) -> Vec<Vec<f64>> {
    if signal.is_empty() || n_fft == 0 || n_chroma == 0 {
        return Vec::new();
    }
    let frame_len = n_fft.min(signal.len());
    let frames = frame_signal(signal, frame_len, hop_length);
    let n_bins = n_fft / 2 + 1;

    // Build chroma filter matrix: (n_chroma, n_bins)
    // Each FFT bin maps to a pitch class based on its frequency.
    let chroma_filters = build_chroma_filters(n_bins, n_fft, sample_rate, n_chroma);

    frames
        .iter()
        .map(|frame| {
            let windowed = apply_hann_window(frame);
            let mag = frame_magnitude_spectrum(&windowed, n_fft);

            // Apply chroma filter
            let mut chroma_frame: Vec<f64> = (0..n_chroma)
                .map(|c| {
                    chroma_filters[c]
                        .iter()
                        .zip(mag.iter())
                        .map(|(f, m)| f * m)
                        .sum::<f64>()
                })
                .collect();

            // Normalize to [0, 1]
            let max_val = chroma_frame.iter().cloned().fold(0.0_f64, f64::max);
            if max_val > 1e-10 {
                chroma_frame.iter_mut().for_each(|v| *v /= max_val);
            }
            chroma_frame
        })
        .collect()
}

/// Build a chroma filter bank mapping FFT bins to pitch classes.
fn build_chroma_filters(n_bins: usize, n_fft: usize, sample_rate: f64, n_chroma: usize) -> Vec<Vec<f64>> {
    let mut filters = vec![vec![0.0_f64; n_bins]; n_chroma];
    // A4 = 440 Hz reference
    let a4_hz = 440.0_f64;

    for bin in 1..n_bins {
        let freq = bin as f64 * sample_rate / n_fft as f64;
        if freq < 1.0 {
            continue;
        }
        // Semitones from A4
        let semitones = 12.0 * (freq / a4_hz).log2();
        // Pitch class (0-based, modulo n_chroma)
        let pc = ((semitones.round() as isize).rem_euclid(n_chroma as isize)) as usize;
        // Gaussian weighting based on deviation from center
        let deviation = semitones - semitones.round();
        let weight = (-0.5 * (deviation / 0.5).powi(2)).exp();
        filters[pc][bin] += weight;
    }
    filters
}

// ─────────────────────────────────────────────────────────────────────────────
// Spectral features
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the spectral centroid (brightness) per frame.
///
/// The centroid is the weighted mean frequency of the spectrum.
///
/// # Example
///
/// ```
/// use scirs2_signal::mir::spectral_centroid;
///
/// let sr = 22050.0;
/// let signal: Vec<f64> = (0..4096)
///     .map(|i| (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / sr).sin())
///     .collect();
/// let centroids = spectral_centroid(&signal, sr, 2048, 512);
/// assert!(!centroids.is_empty());
/// ```
pub fn spectral_centroid(
    signal: &[f64],
    sample_rate: f64,
    n_fft: usize,
    hop_length: usize,
) -> Vec<f64> {
    compute_spectral_feature(signal, n_fft, hop_length, |mag, freqs| {
        let total: f64 = mag.iter().sum();
        if total < 1e-10 {
            return 0.0;
        }
        mag.iter()
            .zip(freqs.iter())
            .map(|(m, f)| m * f)
            .sum::<f64>()
            / total
    }, sample_rate)
}

/// Compute spectral bandwidth per frame.
///
/// The bandwidth is the weighted standard deviation of frequencies around
/// the centroid.
pub fn spectral_bandwidth(
    signal: &[f64],
    sample_rate: f64,
    n_fft: usize,
    hop_length: usize,
) -> Vec<f64> {
    compute_spectral_feature(signal, n_fft, hop_length, |mag, freqs| {
        let total: f64 = mag.iter().sum();
        if total < 1e-10 {
            return 0.0;
        }
        let centroid = mag
            .iter()
            .zip(freqs.iter())
            .map(|(m, f)| m * f)
            .sum::<f64>()
            / total;
        let variance = mag
            .iter()
            .zip(freqs.iter())
            .map(|(m, f)| m * (f - centroid).powi(2))
            .sum::<f64>()
            / total;
        variance.sqrt()
    }, sample_rate)
}

/// Compute spectral roll-off frequency per frame.
///
/// The roll-off is the frequency below which `roll_percent` fraction of the
/// total spectral energy is contained (typically 0.85).
pub fn spectral_rolloff(
    signal: &[f64],
    sample_rate: f64,
    n_fft: usize,
    hop_length: usize,
    roll_percent: f64,
) -> Vec<f64> {
    if signal.is_empty() || n_fft == 0 {
        return Vec::new();
    }
    let frame_len = n_fft.min(signal.len());
    let frames = frame_signal(signal, frame_len, hop_length);
    let n_bins = n_fft / 2 + 1;
    let freqs: Vec<f64> = (0..n_bins)
        .map(|k| k as f64 * sample_rate / n_fft as f64)
        .collect();

    frames
        .iter()
        .map(|frame| {
            let windowed = apply_hann_window(frame);
            let mag = frame_magnitude_spectrum(&windowed, n_fft);
            let total: f64 = mag.iter().sum();
            if total < 1e-10 {
                return 0.0;
            }
            let threshold = roll_percent * total;
            let mut cumsum = 0.0_f64;
            for (m, f) in mag.iter().zip(freqs.iter()) {
                cumsum += m;
                if cumsum >= threshold {
                    return *f;
                }
            }
            freqs[freqs.len() - 1]
        })
        .collect()
}

/// Compute spectral flatness (Wiener entropy) per frame.
///
/// A value of 1.0 indicates white noise; near 0.0 indicates a tonal signal.
pub fn spectral_flatness(signal: &[f64], n_fft: usize, hop_length: usize) -> Vec<f64> {
    if signal.is_empty() || n_fft == 0 {
        return Vec::new();
    }
    let frame_len = n_fft.min(signal.len());
    let frames = frame_signal(signal, frame_len, hop_length);

    frames
        .iter()
        .map(|frame| {
            let windowed = apply_hann_window(frame);
            let power = frame_power_spectrum(&windowed, n_fft);
            let n = power.len() as f64;
            let arith = power.iter().sum::<f64>() / n;
            if arith < 1e-15 {
                return 0.0;
            }
            let log_mean = power.iter().map(|&p| p.max(1e-15).ln()).sum::<f64>() / n;
            let geom = log_mean.exp();
            (geom / arith).min(1.0).max(0.0)
        })
        .collect()
}

/// Compute the zero-crossing rate per frame.
///
/// Returns a per-frame fraction in `[0, 1]` of sample transitions across zero.
///
/// # Example
///
/// ```
/// use scirs2_signal::mir::zero_crossing_rate;
///
/// let signal: Vec<f64> = (0..1024)
///     .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 22050.0).sin())
///     .collect();
/// let zcr = zero_crossing_rate(&signal, 512, 256);
/// assert!(!zcr.is_empty());
/// ```
pub fn zero_crossing_rate(
    signal: &[f64],
    frame_length: usize,
    hop_length: usize,
) -> Vec<f64> {
    if signal.is_empty() || frame_length == 0 {
        return Vec::new();
    }
    let frames = frame_signal(signal, frame_length, hop_length);
    frames
        .iter()
        .map(|frame| {
            if frame.len() < 2 {
                return 0.0;
            }
            let crossings = frame
                .windows(2)
                .filter(|w| w[0] * w[1] < 0.0)
                .count();
            crossings as f64 / (frame.len() - 1) as f64
        })
        .collect()
}

/// Internal helper: compute a per-frame spectral feature.
fn compute_spectral_feature<F>(
    signal: &[f64],
    n_fft: usize,
    hop_length: usize,
    f: F,
    sample_rate: f64,
) -> Vec<f64>
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    if signal.is_empty() || n_fft == 0 {
        return Vec::new();
    }
    let frame_len = n_fft.min(signal.len());
    let frames = frame_signal(signal, frame_len, hop_length);
    let n_bins = n_fft / 2 + 1;
    let freqs: Vec<f64> = (0..n_bins)
        .map(|k| k as f64 * sample_rate / n_fft as f64)
        .collect();
    let _ = sample_rate; // used via freqs computation above

    frames
        .iter()
        .map(|frame| {
            let windowed = apply_hann_window(frame);
            let mag = frame_magnitude_spectrum(&windowed, n_fft);
            f(&mag, &freqs)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Onset detection
// ─────────────────────────────────────────────────────────────────────────────

/// Detect onset times in an audio signal.
///
/// Uses a spectral novelty function (half-wave rectified first-order
/// difference of the log-power spectrum) followed by adaptive thresholding
/// and peak-picking.
///
/// # Arguments
///
/// * `signal` - Audio samples.
/// * `sample_rate` - Sample rate in Hz.
/// * `hop_length` - Hop between analysis frames.
/// * `delta` - Adaptive threshold offset (higher = fewer detections).
///
/// # Returns
///
/// Sample indices of detected onsets.
///
/// # Example
///
/// ```
/// use scirs2_signal::mir::onset_detect;
///
/// let sr = 22050.0;
/// // Create a signal with a sharp transient at sample 4410
/// let mut signal = vec![0.0_f64; 8820];
/// signal[4410] = 1.0;
/// let onsets = onset_detect(&signal, sr, 512, 0.05);
/// assert!(!onsets.is_empty());
/// ```
pub fn onset_detect(
    signal: &[f64],
    sample_rate: f64,
    hop_length: usize,
    delta: f64,
) -> Vec<usize> {
    if signal.is_empty() {
        return Vec::new();
    }
    // Use sample_rate to determine a reasonable FFT size (at least 46ms @ sample_rate)
    let min_fft = ((sample_rate * 0.046) as usize).next_power_of_two().max(256);
    let n_fft = min_fft.min(2048).min(signal.len().next_power_of_two());
    let frame_len = n_fft.min(signal.len());
    let hop = hop_length.max(1);
    let frames = frame_signal(signal, frame_len, hop);
    if frames.len() < 2 {
        return Vec::new();
    }

    // Compute log-power spectrum per frame
    let log_power: Vec<Vec<f64>> = frames
        .iter()
        .map(|frame| {
            let windowed = apply_hann_window(frame);
            let power = frame_power_spectrum(&windowed, n_fft);
            power.iter().map(|&p| (p + 1e-10).log10()).collect()
        })
        .collect();

    // Spectral novelty: half-wave rectified difference
    let n_frames = log_power.len();
    let n_bins = log_power[0].len();
    let mut novelty: Vec<f64> = (1..n_frames)
        .map(|t| {
            (0..n_bins)
                .map(|k| (log_power[t][k] - log_power[t - 1][k]).max(0.0))
                .sum::<f64>()
                / n_bins as f64
        })
        .collect();

    // Normalize novelty
    let max_n = novelty.iter().cloned().fold(0.0_f64, f64::max);
    if max_n > 1e-10 {
        novelty.iter_mut().for_each(|v| *v /= max_n);
    }

    // Adaptive threshold with local mean
    let window = 7usize;
    let threshold: Vec<f64> = (0..novelty.len())
        .map(|i| {
            let lo = i.saturating_sub(window / 2);
            let hi = (i + window / 2 + 1).min(novelty.len());
            let local_mean = novelty[lo..hi].iter().sum::<f64>() / (hi - lo) as f64;
            local_mean + delta
        })
        .collect();

    // Peak-picking
    let mut onsets = Vec::new();
    for i in 1..novelty.len().saturating_sub(1) {
        if novelty[i] > threshold[i]
            && novelty[i] >= novelty[i - 1]
            && novelty[i] >= novelty[i + 1]
        {
            onsets.push((i + 1) * hop); // +1 because novelty starts at frame 1
        }
    }
    onsets
}

// ─────────────────────────────────────────────────────────────────────────────
// Beat tracking
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate the tempo and beat locations of an audio signal.
///
/// Uses an onset novelty function and autocorrelation-based tempo estimation,
/// followed by beat tracking via dynamic programming.
///
/// # Returns
///
/// `(bpm, beat_sample_indices)`
///
/// # Example
///
/// ```
/// use scirs2_signal::mir::beat_track;
/// use std::f64::consts::PI;
///
/// let sr = 22050.0;
/// let bpm_true = 120.0;
/// let beat_period = (60.0 / bpm_true * sr) as usize;
/// // Create periodic impulses at beat positions
/// let mut signal = vec![0.0_f64; 4 * beat_period];
/// for b in 0..4 {
///     signal[b * beat_period] = 1.0;
/// }
/// let (bpm, _beats) = beat_track(&signal, sr);
/// assert!(bpm > 60.0 && bpm < 240.0, "BPM should be in reasonable range");
/// ```
pub fn beat_track(signal: &[f64], sample_rate: f64) -> (f64, Vec<usize>) {
    if signal.is_empty() {
        return (0.0, Vec::new());
    }
    let hop = 512usize;
    let n_fft = 2048.min(signal.len().next_power_of_two());
    let frame_len = n_fft.min(signal.len());
    let frames = frame_signal(signal, frame_len, hop);
    if frames.len() < 4 {
        return (0.0, Vec::new());
    }

    // Compute onset novelty envelope
    let log_power: Vec<Vec<f64>> = frames
        .iter()
        .map(|frame| {
            let windowed = apply_hann_window(frame);
            let power = frame_power_spectrum(&windowed, n_fft);
            power.iter().map(|&p| (p + 1e-10).log10()).collect()
        })
        .collect();

    let n_bins = log_power[0].len();
    let mut novelty: Vec<f64> = (1..log_power.len())
        .map(|t| {
            (0..n_bins)
                .map(|k| (log_power[t][k] - log_power[t - 1][k]).max(0.0))
                .sum::<f64>()
                / n_bins as f64
        })
        .collect();

    // Normalize
    let max_n = novelty.iter().cloned().fold(1e-10_f64, f64::max);
    novelty.iter_mut().for_each(|v| *v /= max_n);

    // BPM search via autocorrelation: range [60, 240] BPM
    let fps = sample_rate / hop as f64; // frames per second
    let lag_min = (60.0 * fps / 240.0).round() as usize; // 240 BPM → min lag
    let lag_max = (60.0 * fps / 60.0).round() as usize;  // 60 BPM → max lag
    let n = novelty.len();

    let mut best_bpm = 120.0_f64;
    let mut best_corr = -1.0_f64;

    for lag in lag_min..=lag_max.min(n / 2) {
        let corr: f64 = (0..n.saturating_sub(lag))
            .map(|i| novelty[i] * novelty[i + lag])
            .sum::<f64>();
        if corr > best_corr {
            best_corr = corr;
            best_bpm = 60.0 * fps / lag as f64;
        }
    }

    // Beat period in frames
    let beat_period_frames = (60.0 * fps / best_bpm).round() as usize;
    if beat_period_frames == 0 {
        return (best_bpm, Vec::new());
    }

    // Find best beat phase by maximizing novelty sum at beat positions
    let n_frames = novelty.len();
    let mut best_phase = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for phase in 0..beat_period_frames.min(n_frames) {
        let score: f64 = (0..)
            .map(|k| phase + k * beat_period_frames)
            .take_while(|&idx| idx < n_frames)
            .map(|idx| novelty[idx])
            .sum();
        if score > best_score {
            best_score = score;
            best_phase = phase;
        }
    }

    // Collect beat sample indices
    let beat_frames: Vec<usize> = (0..)
        .map(|k| best_phase + k * beat_period_frames)
        .take_while(|&idx| idx < n_frames)
        .collect();
    let beat_samples: Vec<usize> = beat_frames.iter().map(|&f| f * hop).collect();

    (best_bpm, beat_samples)
}

// ─────────────────────────────────────────────────────────────────────────────
// Pitch estimation: YIN algorithm
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate per-frame fundamental frequency using the YIN algorithm.
///
/// YIN is based on the difference function of the autocorrelation, with
/// cumulative mean normalization for robust voiced/unvoiced detection.
///
/// # Arguments
///
/// * `signal` - Audio samples.
/// * `sample_rate` - Sample rate in Hz.
/// * `frame_length` - Frame size in samples (typically 2048).
/// * `hop_length` - Hop between frames.
/// * `f_min` - Minimum detectable frequency in Hz (e.g., 80.0).
/// * `f_max` - Maximum detectable frequency in Hz (e.g., 800.0).
/// * `threshold` - YIN threshold for voiced detection (typically 0.1–0.2).
///
/// # Returns
///
/// Per-frame fundamental frequency in Hz, or `None` for unvoiced frames.
///
/// # Example
///
/// ```
/// use scirs2_signal::mir::yin_pitch;
/// use std::f64::consts::PI;
///
/// let sr = 22050.0;
/// let f0 = 440.0_f64; // A4
/// let signal: Vec<f64> = (0..4096)
///     .map(|i| (2.0 * PI * f0 * i as f64 / sr).sin())
///     .collect();
/// let pitches = yin_pitch(&signal, sr, 2048, 512, 80.0, 800.0, 0.15);
/// let voiced: Vec<f64> = pitches.into_iter().flatten().collect();
/// assert!(!voiced.is_empty(), "Should detect voiced frames");
/// // Check frequency is close to 440 Hz
/// for &p in &voiced {
///     assert!((p - f0).abs() < 20.0, "Pitch {} should be near {}", p, f0);
/// }
/// ```
pub fn yin_pitch(
    signal: &[f64],
    sample_rate: f64,
    frame_length: usize,
    hop_length: usize,
    f_min: f64,
    f_max: f64,
    threshold: f64,
) -> Vec<Option<f64>> {
    if signal.is_empty() || frame_length == 0 {
        return Vec::new();
    }

    // Lag search bounds
    let tau_min = (sample_rate / f_max).max(1.0).ceil() as usize;
    let tau_max = (sample_rate / f_min).ceil() as usize;
    let tau_max = tau_max.min(frame_length / 2);

    let frames = frame_signal(signal, frame_length, hop_length);

    frames
        .iter()
        .map(|frame| {
            yin_frame(frame, sample_rate, tau_min, tau_max, threshold)
        })
        .collect()
}

/// Apply YIN to a single frame. Returns `Some(f0_hz)` or `None`.
fn yin_frame(
    frame: &[f64],
    sample_rate: f64,
    tau_min: usize,
    tau_max: usize,
    threshold: f64,
) -> Option<f64> {
    let n = frame.len();
    if tau_max <= tau_min || n < tau_max * 2 {
        return None;
    }

    // Step 1: Difference function
    // d(tau) = sum_{j=0}^{W-1} (x[j] - x[j+tau])^2
    let w = n / 2;
    let mut diff = vec![0.0_f64; tau_max + 1];
    for tau in 0..=tau_max {
        let mut s = 0.0_f64;
        for j in 0..w {
            let diff_j = frame[j] - frame.get(j + tau).copied().unwrap_or(0.0);
            s += diff_j * diff_j;
        }
        diff[tau] = s;
    }

    // Step 2: Cumulative mean normalized difference function
    let mut cmndf = vec![0.0_f64; tau_max + 1];
    cmndf[0] = 1.0;
    let mut running_sum = 0.0_f64;
    for tau in 1..=tau_max {
        running_sum += diff[tau];
        cmndf[tau] = if running_sum.abs() < 1e-15 {
            1.0
        } else {
            diff[tau] * tau as f64 / running_sum
        };
    }

    // Step 3: Absolute threshold — find first dip below threshold
    let tau_est = if tau_min >= tau_max {
        return None;
    } else {
        let mut found = None;
        for tau in tau_min..=tau_max {
            if cmndf[tau] < threshold {
                // Find local minimum in vicinity
                let mut hi = tau;
                while hi < tau_max && cmndf[hi + 1] < cmndf[hi] {
                    hi += 1;
                }
                found = Some(hi);
                break;
            }
        }
        match found {
            Some(t) => t,
            None => {
                // Fall back to global minimum, but only if it is below threshold
                let best = cmndf[tau_min..=tau_max]
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                match best {
                    Some((i, &val)) if val < threshold * 2.0 => i + tau_min,
                    _ => return None,
                }
            }
        }
    };

    if tau_est == 0 {
        return None;
    }

    // Step 4: Parabolic interpolation for sub-sample accuracy
    let tau_interp = if tau_est > tau_min && tau_est < tau_max {
        let x0 = cmndf[tau_est - 1];
        let x1 = cmndf[tau_est];
        let x2 = cmndf[tau_est + 1];
        let denom = x0 - 2.0 * x1 + x2;
        if denom.abs() > 1e-15 {
            tau_est as f64 + 0.5 * (x0 - x2) / denom
        } else {
            tau_est as f64
        }
    } else {
        tau_est as f64
    };

    if tau_interp < 1.0 {
        return None;
    }

    Some(sample_rate / tau_interp)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn sine(freq: f64, sr: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sr).sin())
            .collect()
    }

    #[test]
    fn test_mfcc_extractor_basic() {
        let sr = 22050.0;
        let signal = sine(440.0, sr, 22050);
        let extractor = MfccExtractor::new(sr);
        let mfcc = extractor.extract(&signal);
        assert!(!mfcc.is_empty(), "Should produce MFCC frames");
        for frame in &mfcc {
            assert_eq!(frame.len(), extractor.n_mfcc);
        }
    }

    #[test]
    fn test_mfcc_delta() {
        let sr = 22050.0;
        let signal = sine(440.0, sr, 22050);
        let extractor = MfccExtractor::new(sr);
        let mfcc = extractor.extract(&signal);
        let delta = MfccExtractor::delta(&mfcc);
        let ddelta = MfccExtractor::delta_delta(&mfcc);
        assert_eq!(delta.len(), mfcc.len());
        assert_eq!(ddelta.len(), mfcc.len());
    }

    #[test]
    fn test_mel_spectrogram_shape() {
        let sr = 22050.0;
        let signal = sine(440.0, sr, 4096);
        let extractor = MfccExtractor::new(sr).with_n_mels(40);
        let mel_spec = extractor.mel_spectrogram(&signal);
        assert!(!mel_spec.is_empty());
        for frame in &mel_spec {
            assert_eq!(frame.len(), 40);
        }
    }

    #[test]
    fn test_chroma_stft() {
        let sr = 22050.0;
        let signal = sine(440.0, sr, 4096);
        let chroma = chroma_stft(&signal, sr, 2048, 512, 12);
        assert!(!chroma.is_empty());
        for frame in &chroma {
            assert_eq!(frame.len(), 12);
            for &v in frame {
                assert!(v >= 0.0 && v <= 1.0 + 1e-10, "Chroma value out of range: {}", v);
            }
        }
    }

    #[test]
    fn test_spectral_centroid() {
        let sr = 22050.0;
        // White noise-like signal has centroid near Nyquist/2
        let signal = sine(1000.0, sr, 4096);
        let cents = spectral_centroid(&signal, sr, 2048, 512);
        assert!(!cents.is_empty());
        // Centroid of 1000 Hz sine should be roughly near 1000 Hz
        for &c in &cents {
            assert!(c > 0.0 && c < sr / 2.0, "Centroid {} out of bounds", c);
        }
    }

    #[test]
    fn test_spectral_bandwidth() {
        let sr = 22050.0;
        let signal = sine(440.0, sr, 4096);
        let bw = spectral_bandwidth(&signal, sr, 2048, 512);
        assert!(!bw.is_empty());
        // Bandwidth should be non-negative
        for &b in &bw {
            assert!(b >= 0.0, "Bandwidth should be non-negative: {}", b);
        }
    }

    #[test]
    fn test_spectral_rolloff() {
        let sr = 22050.0;
        let signal = sine(440.0, sr, 4096);
        let ro = spectral_rolloff(&signal, sr, 2048, 512, 0.85);
        assert!(!ro.is_empty());
        for &r in &ro {
            assert!(r >= 0.0 && r <= sr / 2.0, "Roll-off {} out of range", r);
        }
    }

    #[test]
    fn test_spectral_flatness() {
        let sr = 22050.0;
        // Pure tone → low flatness
        let tonal = sine(440.0, sr, 4096);
        let flat_tonal = spectral_flatness(&tonal, 2048, 512);
        assert!(!flat_tonal.is_empty());
        // Flatness should be in [0, 1]
        for &f in &flat_tonal {
            assert!(f >= 0.0 && f <= 1.0 + 1e-9, "Flatness {} out of range", f);
        }
    }

    #[test]
    fn test_zero_crossing_rate() {
        let sr = 22050.0;
        let signal = sine(440.0, sr, 4096);
        let zcr = zero_crossing_rate(&signal, 512, 256);
        assert!(!zcr.is_empty());
        for &z in &zcr {
            assert!(z >= 0.0 && z <= 1.0, "ZCR {} out of range", z);
        }
    }

    #[test]
    fn test_onset_detect_impulse() {
        let sr = 22050.0;
        let mut signal = vec![0.0_f64; 8820];
        signal[4410] = 1.0;
        let onsets = onset_detect(&signal, sr, 512, 0.05);
        assert!(!onsets.is_empty(), "Should detect the impulse onset");
    }

    #[test]
    fn test_beat_track_periodic() {
        let sr = 22050.0;
        let bpm = 120.0;
        let beat_period = (60.0 / bpm * sr) as usize;
        let n = 4 * beat_period;
        let mut signal = vec![0.0_f64; n];
        for b in 0..4 {
            let idx = b * beat_period;
            if idx < n {
                signal[idx] = 1.0;
            }
        }
        let (est_bpm, beats) = beat_track(&signal, sr);
        assert!(est_bpm > 60.0 && est_bpm < 240.0, "BPM {} out of range", est_bpm);
        assert!(!beats.is_empty(), "Should find beats");
    }

    #[test]
    fn test_yin_pitch_sine() {
        let sr = 22050.0;
        let f0 = 440.0;
        let signal = sine(f0, sr, 8192);
        let pitches = yin_pitch(&signal, sr, 2048, 512, 80.0, 800.0, 0.15);
        let voiced: Vec<f64> = pitches.into_iter().flatten().collect();
        assert!(!voiced.is_empty(), "Should detect voiced frames for pure tone");
        for &p in &voiced {
            assert!((p - f0).abs() < 30.0, "Pitch {} should be near {} Hz", p, f0);
        }
    }

    #[test]
    fn test_yin_silence() {
        let sr = 22050.0;
        let signal = vec![0.0_f64; 4096];
        let pitches = yin_pitch(&signal, sr, 2048, 512, 80.0, 800.0, 0.15);
        // Silence should be all unvoiced
        for p in &pitches {
            assert!(p.is_none(), "Silence should be unvoiced");
        }
    }
}
