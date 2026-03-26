//! Mel-Frequency Cepstral Coefficients (MFCC) for WASM
//!
//! Implements the full MFCC processing pipeline:
//! pre-emphasis → framing → windowing → FFT → mel filterbank → log → DCT-II

use crate::error::{WasmError, WasmResult};
use std::f64::consts::PI;

// ─── Config ───────────────────────────────────────────────────────────────────

/// Configuration for MFCC computation
#[derive(Debug, Clone)]
pub struct MfccConfig {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of MFCC coefficients to return
    pub n_mfcc: usize,
    /// Number of mel filterbank channels
    pub n_mels: usize,
    /// FFT size (zero-padded if frame is shorter)
    pub n_fft: usize,
    /// Hop length (samples between successive frames)
    pub hop_length: usize,
    /// Window length (samples per frame)
    pub win_length: usize,
    /// Minimum frequency for mel filterbank (Hz)
    pub f_min: f64,
    /// Maximum frequency for mel filterbank (Hz)
    pub f_max: f64,
    /// Pre-emphasis coefficient
    pub pre_emphasis: f64,
}

impl Default for MfccConfig {
    fn default() -> Self {
        MfccConfig {
            sample_rate: 16000,
            n_mfcc: 13,
            n_mels: 26,
            n_fft: 512,
            hop_length: 160,
            win_length: 400,
            f_min: 0.0,
            f_max: 8000.0,
            pre_emphasis: 0.97,
        }
    }
}

/// Result of MFCC computation
#[derive(Debug, Clone)]
pub struct MfccResult {
    /// MFCC coefficients: n_frames × n_mfcc
    pub coefficients: Vec<Vec<f64>>,
    /// Log mel-spectrogram: n_frames × n_mels
    pub log_mel_spec: Vec<Vec<f64>>,
    /// Number of frames
    pub n_frames: usize,
}

// ─── Processing functions ─────────────────────────────────────────────────────

/// Apply pre-emphasis filter: `y[n] = x[n] - coef * x[n-1]`
pub fn pre_emphasize(signal: &[f64], coef: f64) -> Vec<f64> {
    if signal.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(signal.len());
    out.push(signal[0]);
    for i in 1..signal.len() {
        out.push(signal[i] - coef * signal[i - 1]);
    }
    out
}

/// Split signal into overlapping frames.
///
/// Each frame has length `win_length`; frames advance by `hop_length` samples.
pub fn frame_signal(signal: &[f64], win_length: usize, hop_length: usize) -> Vec<Vec<f64>> {
    if signal.is_empty() || win_length == 0 || hop_length == 0 {
        return Vec::new();
    }
    let n = signal.len();
    if n < win_length {
        return Vec::new();
    }
    let n_frames = 1 + (n - win_length) / hop_length;
    let mut frames = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let start = i * hop_length;
        let frame = signal[start..start + win_length].to_vec();
        frames.push(frame);
    }
    frames
}

/// Hann analysis window: `w[k] = 0.5 * (1 - cos(2*pi*k / N))`
pub fn hann_window(n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f64 / n as f64).cos()))
        .collect()
}

// ─── Internal FFT ─────────────────────────────────────────────────────────────

/// Cooley-Tukey radix-2 FFT (in-place, complex interleaved [re, im, re, im, ...])
fn fft_inplace(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
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
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // Butterfly stages
    let mut len = 2usize;
    while len <= n {
        let angle = -2.0 * PI / len as f64;
        let wr = angle.cos();
        let wi = angle.sin();
        let mut k = 0;
        while k < n {
            let (mut cur_wr, mut cur_wi) = (1.0_f64, 0.0_f64);
            for m in 0..len / 2 {
                let u_re = re[k + m];
                let u_im = im[k + m];
                let v_re = re[k + m + len / 2] * cur_wr - im[k + m + len / 2] * cur_wi;
                let v_im = re[k + m + len / 2] * cur_wi + im[k + m + len / 2] * cur_wr;
                re[k + m] = u_re + v_re;
                im[k + m] = u_im + v_im;
                re[k + m + len / 2] = u_re - v_re;
                im[k + m + len / 2] = u_im - v_im;
                let new_wr = cur_wr * wr - cur_wi * wi;
                cur_wi = cur_wr * wi + cur_wi * wr;
                cur_wr = new_wr;
            }
            k += len;
        }
        len <<= 1;
    }
}

/// Next power of two >= n
fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

/// Compute power spectrum of a windowed frame.
///
/// Applies a Hann window, zero-pads to `n_fft`, runs FFT, returns |FFT|²
/// for the first `n_fft/2 + 1` bins.
pub fn power_spectrum(frame: &[f64], n_fft: usize) -> Vec<f64> {
    let win = hann_window(frame.len());
    let fft_size = next_pow2(n_fft.max(frame.len()));
    let mut re = vec![0.0_f64; fft_size];
    let mut im = vec![0.0_f64; fft_size];

    for (i, (&f, &w)) in frame.iter().zip(win.iter()).enumerate() {
        re[i] = f * w;
    }

    fft_inplace(&mut re, &mut im);

    let n_bins = n_fft / 2 + 1;
    (0..n_bins).map(|k| re[k] * re[k] + im[k] * im[k]).collect()
}

// ─── Mel filterbank ───────────────────────────────────────────────────────────

/// Convert frequency in Hz to mel scale
#[inline]
fn hz_to_mel(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}

/// Convert mel value back to Hz
#[inline]
fn mel_to_hz(m: f64) -> f64 {
    700.0 * (10.0_f64.powf(m / 2595.0) - 1.0)
}

/// Build a mel filterbank matrix of shape `n_mels × (n_fft/2 + 1)`.
///
/// Each row contains triangular filter weights for one mel band.
pub fn mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: u32,
    f_min: f64,
    f_max: f64,
) -> WasmResult<Vec<Vec<f64>>> {
    if n_mels == 0 {
        return Err(WasmError::InvalidParameter(
            "n_mels must be > 0".to_string(),
        ));
    }
    let n_bins = n_fft / 2 + 1;
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max.min(sample_rate as f64 / 2.0));

    // n_mels + 2 equally spaced points in mel domain
    let n_points = n_mels + 2;
    let mel_points: Vec<f64> = (0..n_points)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_points as f64 - 1.0))
        .collect();
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Map Hz points to FFT bin indices
    let bin_points: Vec<f64> = hz_points
        .iter()
        .map(|&f| (n_fft as f64 + 1.0) * f / sample_rate as f64)
        .collect();

    let mut filterbank = vec![vec![0.0_f64; n_bins]; n_mels];
    for m in 0..n_mels {
        let f_m_minus = bin_points[m];
        let f_m = bin_points[m + 1];
        let f_m_plus = bin_points[m + 2];
        for (k, fb_val) in filterbank[m].iter_mut().enumerate() {
            let k_f = k as f64;
            if k_f >= f_m_minus && k_f <= f_m {
                let denom = f_m - f_m_minus;
                *fb_val = if denom.abs() < 1e-15 {
                    0.0
                } else {
                    (k_f - f_m_minus) / denom
                };
            } else if k_f >= f_m && k_f <= f_m_plus {
                let denom = f_m_plus - f_m;
                *fb_val = if denom.abs() < 1e-15 {
                    0.0
                } else {
                    (f_m_plus - k_f) / denom
                };
            }
        }
    }
    Ok(filterbank)
}

/// Apply filterbank and take log: log(mel_energy + 1e-8).
///
/// Returns n_frames × n_mels log-mel spectrogram.
pub fn log_mel_spectrogram(frames: &[Vec<f64>], filterbank: &[Vec<f64>]) -> Vec<Vec<f64>> {
    frames
        .iter()
        .map(|power| {
            filterbank
                .iter()
                .map(|filt| {
                    let energy: f64 = filt.iter().zip(power.iter()).map(|(&h, &p)| h * p).sum();
                    (energy + 1e-8_f64).ln()
                })
                .collect()
        })
        .collect()
}

// ─── DCT-II ───────────────────────────────────────────────────────────────────

/// Type-II DCT: `X[k] = Sum_n x[n] * cos(pi*(2n+1)*k / 2N)` for k = 0 .. n_out-1
pub fn dct_2(x: &[f64], n_out: usize) -> Vec<f64> {
    let n = x.len();
    (0..n_out)
        .map(|k| {
            x.iter()
                .enumerate()
                .map(|(i, &v)| v * (PI * (2 * i + 1) as f64 * k as f64 / (2.0 * n as f64)).cos())
                .sum()
        })
        .collect()
}

// ─── Main entry point ─────────────────────────────────────────────────────────

/// Compute MFCC features for a raw audio signal.
pub fn compute_mfcc(signal: &[f64], config: &MfccConfig) -> WasmResult<MfccResult> {
    if signal.is_empty() {
        return Err(WasmError::InvalidParameter(
            "Signal must not be empty".to_string(),
        ));
    }
    if config.n_mfcc > config.n_mels {
        return Err(WasmError::InvalidParameter(format!(
            "n_mfcc ({}) must be <= n_mels ({})",
            config.n_mfcc, config.n_mels
        )));
    }

    // 1. Pre-emphasis
    let emphasized = pre_emphasize(signal, config.pre_emphasis);

    // 2. Frame
    let frames = frame_signal(&emphasized, config.win_length, config.hop_length);
    if frames.is_empty() {
        return Err(WasmError::InvalidParameter(
            "Signal too short to produce any frames".to_string(),
        ));
    }
    let n_frames = frames.len();

    // 3. Power spectra
    let power_specs: Vec<Vec<f64>> = frames
        .iter()
        .map(|f| power_spectrum(f, config.n_fft))
        .collect();

    // 4. Mel filterbank
    let filterbank = mel_filterbank(
        config.n_mels,
        config.n_fft,
        config.sample_rate,
        config.f_min,
        config.f_max,
    )?;

    // 5. Log mel spectrogram
    let log_mel_spec = log_mel_spectrogram(&power_specs, &filterbank);

    // 6. DCT-II → MFCC
    let coefficients: Vec<Vec<f64>> = log_mel_spec
        .iter()
        .map(|lms| dct_2(lms, config.n_mfcc))
        .collect();

    Ok(MfccResult {
        coefficients,
        log_mel_spec,
        n_frames,
    })
}

// ─── Delta features ───────────────────────────────────────────────────────────

/// Compute delta (first derivative) features via linear regression over a window.
///
/// `width` is the one-sided half-width (default 2); window = 2*width+1 frames.
pub fn mfcc_delta(mfcc: &[Vec<f64>], width: usize) -> Vec<Vec<f64>> {
    if mfcc.is_empty() {
        return Vec::new();
    }
    let n_frames = mfcc.len();
    let n_coef = mfcc[0].len();
    let w = width.max(1);
    let denom: f64 = (1..=w).map(|i| 2 * i * i).sum::<usize>() as f64;

    let mut deltas = vec![vec![0.0_f64; n_coef]; n_frames];
    for (t, delta_row) in deltas.iter_mut().enumerate() {
        for c in 0..n_coef {
            let mut num = 0.0_f64;
            for delta in 1..=w {
                let t_plus = (t + delta).min(n_frames - 1);
                let t_minus = t.saturating_sub(delta);
                num += delta as f64 * (mfcc[t_plus][c] - mfcc[t_minus][c]);
            }
            delta_row[c] = if denom.abs() < 1e-15 {
                0.0
            } else {
                num / denom
            };
        }
    }
    deltas
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_440hz(sample_rate: u32, duration_secs: f64) -> Vec<f64> {
        let n = (sample_rate as f64 * duration_secs) as usize;
        (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / sample_rate as f64).sin())
            .collect()
    }

    #[test]
    fn test_compute_mfcc_correct_dimensions() {
        let config = MfccConfig::default();
        let signal = sine_440hz(config.sample_rate, 1.0);
        let result = compute_mfcc(&signal, &config).expect("MFCC failed");

        // n_frames = 1 + (n - win_length) / hop_length
        let expected_n_frames = 1 + (signal.len() - config.win_length) / config.hop_length;
        assert_eq!(result.n_frames, expected_n_frames);
        assert_eq!(result.coefficients.len(), expected_n_frames);
        for frame_coefs in &result.coefficients {
            assert_eq!(frame_coefs.len(), config.n_mfcc);
        }
    }

    #[test]
    fn test_compute_mfcc_log_mel_spec_dimensions() {
        let config = MfccConfig::default();
        let signal = sine_440hz(config.sample_rate, 1.0);
        let result = compute_mfcc(&signal, &config).expect("MFCC failed");

        for row in &result.log_mel_spec {
            assert_eq!(row.len(), config.n_mels);
        }
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let n_mels = 26;
        let n_fft = 512;
        let sample_rate = 16000;
        let fb =
            mel_filterbank(n_mels, n_fft, sample_rate, 0.0, 8000.0).expect("filterbank failed");
        assert_eq!(fb.len(), n_mels);
        let n_bins = n_fft / 2 + 1;
        for row in &fb {
            assert_eq!(row.len(), n_bins);
        }
    }

    #[test]
    fn test_mel_filterbank_non_negative() {
        let fb = mel_filterbank(26, 512, 16000, 0.0, 8000.0).expect("filterbank");
        for row in &fb {
            for &v in row {
                assert!(v >= 0.0, "negative filterbank weight: {v}");
            }
        }
    }

    #[test]
    fn test_dct_orthogonality() {
        // DCT-II: for a unit impulse at position 0, X[k] = cos(π k / 2N)
        let n = 8;
        let mut x = vec![0.0_f64; n];
        x[0] = 1.0;
        let y = dct_2(&x, n);
        for (k, &v) in y.iter().enumerate() {
            let expected = (PI * k as f64 / (2.0 * n as f64)).cos();
            assert!(
                (v - expected).abs() < 1e-10,
                "DCT[{k}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_pre_emphasize() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let coef = 0.97;
        let out = pre_emphasize(&signal, coef);
        assert_eq!(out[0], 1.0);
        assert!((out[1] - (2.0 - 0.97 * 1.0)).abs() < 1e-12);
        assert!((out[2] - (3.0 - 0.97 * 2.0)).abs() < 1e-12);
    }

    #[test]
    fn test_frame_signal_count() {
        // n=10, win=4, hop=2 → frames at 0,2,4,6 = 4 frames
        let signal: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let frames = frame_signal(&signal, 4, 2);
        assert_eq!(frames.len(), 4);
        assert_eq!(frames[0].len(), 4);
    }

    #[test]
    fn test_hann_window_endpoints() {
        let w = hann_window(8);
        assert!(w[0].abs() < 1e-10, "Hann window should start at 0");
        // The last sample is not exactly 0 for a symmetric window; just check it's small
        assert!(w[7] < 0.25, "Hann window endpoint should be small");
    }

    #[test]
    fn test_mfcc_delta_shape() {
        let config = MfccConfig::default();
        let signal = sine_440hz(config.sample_rate, 0.5);
        let result = compute_mfcc(&signal, &config).expect("MFCC");
        let deltas = mfcc_delta(&result.coefficients, 2);
        assert_eq!(deltas.len(), result.n_frames);
        for row in &deltas {
            assert_eq!(row.len(), config.n_mfcc);
        }
    }
}
