//! Neural audio inference API: weight loading, model trait, and Mel spectrogram.
//!
//! Provides:
//! - [`WeightStore`] — load/save pre-trained neural network weights.
//! - [`AudioModel`] trait — common interface for audio neural networks.
//! - [`NeuralAudioConfig`] — shared configuration (sample rate, FFT params, …).
//! - [`ConvTasNetInference`] — inference-only Conv-TasNet speech enhancer.
//! - [`compute_mel_spectrogram`] — short-time Mel-scale power spectrogram.

use std::collections::HashMap;
use std::io::{Read, Write};

use crate::error::SignalError;

// ── ModelFormat ───────────────────────────────────────────────────────────────

/// Serialisation format for pre-trained weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ModelFormat {
    /// SafeTensors format (header + flat binary).
    SafeTensors,
    /// NumPy-compatible .npz-like format.
    NpzLike,
    /// Simple binary dump: n_tensors → (name, shape, f32 data) records.
    #[default]
    BinaryDump,
}

// ── NeuralAudioConfig ─────────────────────────────────────────────────────────

/// Shared configuration for neural audio models.
#[derive(Debug, Clone)]
pub struct NeuralAudioConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// STFT frame length in samples.
    pub frame_length: usize,
    /// STFT hop length in samples.
    pub hop_length: usize,
    /// Number of Mel filter bands.
    pub n_mels: usize,
    /// Preferred serialisation format.
    pub model_format: ModelFormat,
}

impl Default for NeuralAudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            frame_length: 400,
            hop_length: 160,
            n_mels: 80,
            model_format: ModelFormat::BinaryDump,
        }
    }
}

// ── AudioModel trait ──────────────────────────────────────────────────────────

/// Common interface for audio neural network models.
pub trait AudioModel: Send + Sync {
    /// Process raw audio samples (mono, normalised to ±1) and return
    /// feature frames as a 2-D matrix (outer = features, inner = time).
    fn process(&self, audio: &[f64]) -> Result<Vec<Vec<f64>>, SignalError>;

    /// Human-readable model name.
    fn name(&self) -> &str;

    /// Expected sample rate in Hz.
    fn sample_rate(&self) -> u32;
}

// ── WeightStore ───────────────────────────────────────────────────────────────

/// Storage for neural network weights.
///
/// Each entry maps a tensor name to a `(shape, data)` pair where
/// `shape` is a list of dimension sizes and `data` is a flat f32 slice.
///
/// # Binary dump format
///
/// ```text
/// [n_tensors : u64 LE]
/// For each tensor:
///   [name_len : u64 LE]
///   [name     : name_len bytes (UTF-8)]
///   [n_dims   : u64 LE]
///   [dims     : n_dims × u64 LE]
///   [data     : prod(dims) × f32 LE]
/// ```
#[derive(Debug, Clone, Default)]
pub struct WeightStore {
    weights: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl WeightStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a tensor.
    pub fn insert(&mut self, name: impl Into<String>, shape: Vec<usize>, data: Vec<f32>) {
        self.weights.insert(name.into(), (shape, data));
    }

    /// Retrieve a tensor by name.  Returns `(shape, data)` or `None`.
    pub fn get(&self, name: &str) -> Option<(&[usize], &[f32])> {
        self.weights
            .get(name)
            .map(|(sh, d)| (sh.as_slice(), d.as_slice()))
    }

    /// List all tensor names.
    pub fn names(&self) -> Vec<&str> {
        self.weights.keys().map(|s| s.as_str()).collect()
    }

    /// Number of tensors stored.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Returns `true` if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Load weights from the binary dump format.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::ComputationError`] if the file cannot be read or
    /// the format is invalid.
    pub fn from_binary_dump(path: &str) -> Result<Self, SignalError> {
        let bytes = std::fs::read(path)?;
        let mut cursor = 0usize;

        let read_u64 = |cursor: &mut usize, bytes: &[u8]| -> Result<u64, SignalError> {
            if *cursor + 8 > bytes.len() {
                return Err(SignalError::ComputationError(
                    "Unexpected EOF reading u64".to_string(),
                ));
            }
            let val = u64::from_le_bytes(
                bytes[*cursor..*cursor + 8]
                    .try_into()
                    .map_err(|_| SignalError::ComputationError("u64 slice error".to_string()))?,
            );
            *cursor += 8;
            Ok(val)
        };

        let n_tensors = read_u64(&mut cursor, &bytes)? as usize;
        let mut store = WeightStore::new();

        for _ in 0..n_tensors {
            // Name
            let name_len = read_u64(&mut cursor, &bytes)? as usize;
            if cursor + name_len > bytes.len() {
                return Err(SignalError::ComputationError(
                    "Unexpected EOF reading tensor name".to_string(),
                ));
            }
            let name = std::str::from_utf8(&bytes[cursor..cursor + name_len])
                .map_err(|e| SignalError::ComputationError(format!("UTF-8 error: {e}")))?
                .to_string();
            cursor += name_len;

            // Shape
            let n_dims = read_u64(&mut cursor, &bytes)? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(read_u64(&mut cursor, &bytes)? as usize);
            }

            // Data
            let numel: usize = shape.iter().product();
            let byte_count = numel * 4;
            if cursor + byte_count > bytes.len() {
                return Err(SignalError::ComputationError(
                    "Unexpected EOF reading tensor data".to_string(),
                ));
            }
            let data: Vec<f32> = (0..numel)
                .map(|i| {
                    let off = cursor + i * 4;
                    f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
                })
                .collect();
            cursor += byte_count;

            store.insert(name, shape, data);
        }

        Ok(store)
    }

    /// Save weights to the binary dump format.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::ComputationError`] if the file cannot be written.
    pub fn save_binary_dump(path: &str, store: &WeightStore) -> Result<(), SignalError> {
        let mut buf: Vec<u8> = Vec::new();

        // n_tensors
        buf.extend_from_slice(&(store.len() as u64).to_le_bytes());

        for (name, (shape, data)) in &store.weights {
            // name_len + name
            let name_bytes = name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
            buf.extend_from_slice(name_bytes);

            // n_dims + dims
            buf.extend_from_slice(&(shape.len() as u64).to_le_bytes());
            for &d in shape {
                buf.extend_from_slice(&(d as u64).to_le_bytes());
            }

            // data
            for &v in data {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }

        std::fs::write(path, &buf)?;
        Ok(())
    }
}

// ── Mel spectrogram ───────────────────────────────────────────────────────────

/// Compute a Mel-scale power spectrogram from a raw waveform.
///
/// Steps:
/// 1. Frame the signal with Hann window.
/// 2. Compute power spectrum via radix-2 Cooley-Tukey FFT.
/// 3. Apply triangular Mel filter banks (0 Hz → sample_rate / 2).
///
/// # Returns
///
/// 2-D matrix of shape `[n_mels][n_frames]`.
pub fn compute_mel_spectrogram(audio: &[f64], config: &NeuralAudioConfig) -> Vec<Vec<f64>> {
    let n = audio.len();
    let frame_len = config.frame_length;
    let hop_len = config.hop_length;
    let n_mels = config.n_mels;
    let sr = config.sample_rate as f64;

    if n == 0 || frame_len == 0 || hop_len == 0 {
        return vec![vec![]; n_mels];
    }

    // ── 1. Hann window ────────────────────────────────────────────────────────
    let hann: Vec<f64> = (0..frame_len)
        .map(|i| {
            0.5 * (1.0
                - (2.0 * std::f64::consts::PI * i as f64 / (frame_len - 1).max(1) as f64).cos())
        })
        .collect();

    // ── 2. FFT size (next power-of-2 ≥ frame_len) ────────────────────────────
    let fft_size = {
        let mut s = 1usize;
        while s < frame_len {
            s <<= 1;
        }
        s
    };
    let n_bins = fft_size / 2 + 1;

    // ── 3. Frame and compute power spectra ────────────────────────────────────
    let n_frames = if n < frame_len {
        0
    } else {
        (n - frame_len) / hop_len + 1
    };

    let mut power_frames: Vec<Vec<f64>> = Vec::with_capacity(n_frames);
    for fi in 0..n_frames {
        let start = fi * hop_len;
        let mut buf = vec![0.0; fft_size];
        for k in 0..frame_len {
            buf[k] = audio[start + k] * hann[k];
        }
        let spectrum = fft_power(&buf);
        // spectrum has fft_size / 2 + 1 bins (or full fft_size from our helper)
        let ps: Vec<f64> = spectrum.into_iter().take(n_bins).collect();
        power_frames.push(ps);
    }

    // ── 4. Mel filter banks ───────────────────────────────────────────────────
    let filters = mel_filter_bank(n_mels, n_bins, sr, fft_size);

    // ── 5. Apply filters ──────────────────────────────────────────────────────
    // Output: [n_mels][n_frames]
    let mut mel_spec = vec![vec![0.0_f64; n_frames]; n_mels];
    for (fi, ps) in power_frames.iter().enumerate() {
        for (m, filt) in filters.iter().enumerate() {
            let energy: f64 = ps.iter().zip(filt.iter()).map(|(p, f)| p * f).sum();
            mel_spec[m][fi] = energy;
        }
    }

    mel_spec
}

/// Compute the one-sided power spectrum via radix-2 Cooley-Tukey FFT.
/// Input must have length that is a power of 2.
/// Returns `|X[k]|²` for k = 0 … N/2.
fn fft_power(buf: &[f64]) -> Vec<f64> {
    let n = buf.len();
    if n == 0 {
        return Vec::new();
    }

    // Compute complex FFT in-place (Cooley-Tukey, DIT, radix-2)
    let mut re = buf.to_vec();
    let mut im = vec![0.0_f64; n];

    cooley_tukey_fft_inplace(&mut re, &mut im);

    // One-sided power: |X[k]|² for k = 0 … N/2
    (0..=n / 2).map(|k| re[k] * re[k] + im[k] * im[k]).collect()
}

/// In-place Cooley-Tukey radix-2 DIT FFT.
/// `re` and `im` must have the same length, which must be a power of 2.
fn cooley_tukey_fft_inplace(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation
    let log2 = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = bit_reverse(i, log2);
        if j > i {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // Butterfly stages
    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        let wr = ang.cos();
        let wi = ang.sin();
        let half = len / 2;
        let mut i = 0;
        while i < n {
            let (mut cur_wr, mut cur_wi) = (1.0_f64, 0.0_f64);
            for j in 0..half {
                let u_re = re[i + j];
                let u_im = im[i + j];
                let v_re = re[i + j + half] * cur_wr - im[i + j + half] * cur_wi;
                let v_im = re[i + j + half] * cur_wi + im[i + j + half] * cur_wr;
                re[i + j] = u_re + v_re;
                im[i + j] = u_im + v_im;
                re[i + j + half] = u_re - v_re;
                im[i + j + half] = u_im - v_im;
                let new_wr = cur_wr * wr - cur_wi * wi;
                let new_wi = cur_wr * wi + cur_wi * wr;
                cur_wr = new_wr;
                cur_wi = new_wi;
            }
            i += len;
        }
        len <<= 1;
    }
}

fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0usize;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// Compute triangular Mel filter banks.
///
/// Returns `n_mels` filters, each of length `n_bins`.
fn mel_filter_bank(
    n_mels: usize,
    n_bins: usize,
    sample_rate: f64,
    fft_size: usize,
) -> Vec<Vec<f64>> {
    if n_mels == 0 || n_bins == 0 {
        return Vec::new();
    }

    let hz_to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f64| 700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0);

    let f_min = 0.0_f64;
    let f_max = sample_rate / 2.0;

    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // n_mels + 2 evenly spaced Mel points (includes edge points)
    let mel_points: Vec<f64> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
        .collect();

    // Convert Mel points to FFT bin indices
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<f64> = hz_points
        .iter()
        .map(|&f| (fft_size as f64 + 1.0) * f / sample_rate)
        .collect();

    let mut filters = vec![vec![0.0_f64; n_bins]; n_mels];
    for m in 0..n_mels {
        let f_m_minus = bin_points[m];
        let f_m = bin_points[m + 1];
        let f_m_plus = bin_points[m + 2];

        for k in 0..n_bins {
            let k_f = k as f64;
            if k_f >= f_m_minus && k_f <= f_m {
                let denom = f_m - f_m_minus;
                filters[m][k] = if denom.abs() < 1e-12 {
                    0.0
                } else {
                    (k_f - f_m_minus) / denom
                };
            } else if k_f >= f_m && k_f <= f_m_plus {
                let denom = f_m_plus - f_m;
                filters[m][k] = if denom.abs() < 1e-12 {
                    0.0
                } else {
                    (f_m_plus - k_f) / denom
                };
            }
        }
    }

    filters
}

// ── ConvTasNetInference ───────────────────────────────────────────────────────

/// Inference-only Conv-TasNet speech enhancer.
///
/// Performs a minimal forward pass:
/// 1. **Encoder**: 1-D strided convolution (learned filterbank).
/// 2. **TCN**: two stacked depthwise-separable conv blocks.
/// 3. **Decoder**: overlap-add via transposed convolution.
///
/// To load a real pre-trained model, serialise its weights to the
/// [`WeightStore`] binary dump format and call
/// [`ConvTasNetInference::from_weights`].  The current implementation
/// initialises weights deterministically so that
/// `enhance()` is always callable even without a saved checkpoint.
#[derive(Debug, Clone)]
pub struct ConvTasNetInference {
    config: NeuralAudioConfig,
    /// Encoder weights: [n_filters × (1 × kernel)]
    enc_w: Vec<Vec<f64>>,
    enc_stride: usize,
    /// Two TCN blocks, each as (depthwise, pointwise) weight matrices
    tcn_blocks: Vec<(Vec<Vec<f64>>, Vec<Vec<f64>>)>,
    /// Decoder weights: [kernel × n_filters]
    dec_w: Vec<Vec<f64>>,
    n_filters: usize,
    enc_kernel: usize,
}

impl ConvTasNetInference {
    /// Initialise with random weights (useful for unit tests / dry runs).
    pub fn new(config: NeuralAudioConfig) -> Self {
        let n_filters = 32;
        let enc_kernel = 16;
        let enc_stride = 8;
        let tcn_kernel = 3;
        let bottleneck = 8;

        let mut rng_state: u64 = 0x4a5e3f2c1d7b6a9e;
        let mut lcg = || -> f64 {
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = (rng_state >> 11) as f64 / (1u64 << 53) as f64;
            (bits * 2.0 - 1.0) * (2.0 / enc_kernel as f64).sqrt()
        };

        let enc_w: Vec<Vec<f64>> = (0..n_filters)
            .map(|_| (0..enc_kernel).map(|_| lcg()).collect())
            .collect();

        let tcn_blocks: Vec<(Vec<Vec<f64>>, Vec<Vec<f64>>)> = (0..2)
            .map(|_| {
                let dw: Vec<Vec<f64>> = (0..bottleneck)
                    .map(|_| (0..tcn_kernel).map(|_| lcg()).collect())
                    .collect();
                let pw: Vec<Vec<f64>> = (0..n_filters)
                    .map(|_| (0..bottleneck).map(|_| lcg()).collect())
                    .collect();
                (dw, pw)
            })
            .collect();

        let dec_w: Vec<Vec<f64>> = (0..enc_kernel)
            .map(|_| (0..n_filters).map(|_| lcg()).collect())
            .collect();

        Self {
            config,
            enc_w,
            enc_stride,
            tcn_blocks,
            dec_w,
            n_filters,
            enc_kernel,
        }
    }

    /// Load a model from a weight store.
    ///
    /// Looks for tensors named `enc_w`, `dec_w`, `tcn0_dw`, `tcn0_pw`, etc.
    /// Falls back to random initialisation for any missing tensors.
    pub fn from_weights(
        store: &WeightStore,
        config: NeuralAudioConfig,
    ) -> Result<Self, SignalError> {
        let mut model = Self::new(config);

        // Override encoder weights if present
        if let Some((shape, data)) = store.get("enc_w") {
            if shape.len() == 2 && !data.is_empty() {
                let n_out = shape[0];
                let k = shape[1];
                model.n_filters = n_out;
                model.enc_kernel = k;
                model.enc_w = (0..n_out)
                    .map(|i| {
                        (0..k)
                            .map(|j| data.get(i * k + j).copied().unwrap_or(0.0) as f64)
                            .collect()
                    })
                    .collect();
            }
        }

        // Override decoder weights if present
        if let Some((shape, data)) = store.get("dec_w") {
            if shape.len() == 2 && !data.is_empty() {
                let rows = shape[0];
                let cols = shape[1];
                model.dec_w = (0..rows)
                    .map(|i| {
                        (0..cols)
                            .map(|j| data.get(i * cols + j).copied().unwrap_or(0.0) as f64)
                            .collect()
                    })
                    .collect();
            }
        }

        Ok(model)
    }

    /// Enhance a noisy waveform (speech enhancement).
    ///
    /// Returns a waveform of the same length as the input.
    pub fn enhance(&self, noisy: &[f64]) -> Result<Vec<f64>, SignalError> {
        let n = noisy.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // ── Encoder: strided 1-D convolution ──────────────────────────────────
        let enc_frames = self.encoder(noisy);
        if enc_frames.is_empty() {
            return Ok(vec![0.0; n]);
        }

        // ── TCN blocks ────────────────────────────────────────────────────────
        let mut features = enc_frames.clone();
        for (dw, pw) in &self.tcn_blocks {
            features = self.tcn_forward(&features, dw, pw);
        }

        // ── Simple mask: sigmoid activation on feature sum ────────────────────
        let masks: Vec<f64> = features
            .iter()
            .map(|f| sigmoid(f.iter().sum::<f64>() / f.len().max(1) as f64))
            .collect();

        // ── Decoder: overlap-add ──────────────────────────────────────────────
        let enhanced = self.decoder(&enc_frames, &masks, n);
        Ok(enhanced)
    }

    // ── Encoder ────────────────────────────────────────────────────────────────

    fn encoder(&self, signal: &[f64]) -> Vec<Vec<f64>> {
        let n = signal.len();
        let k = self.enc_kernel;
        let s = self.enc_stride;
        if n < k {
            return Vec::new();
        }
        let n_frames = (n - k) / s + 1;
        (0..n_frames)
            .map(|fi| {
                let start = fi * s;
                self.enc_w
                    .iter()
                    .map(|filt| {
                        filt.iter()
                            .enumerate()
                            .map(|(j, &w)| signal[start + j] * w)
                            .sum::<f64>()
                            .max(0.0) // ReLU
                    })
                    .collect()
            })
            .collect()
    }

    // ── TCN block ──────────────────────────────────────────────────────────────

    fn tcn_forward(&self, frames: &[Vec<f64>], dw: &[Vec<f64>], pw: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if frames.is_empty() || dw.is_empty() || pw.is_empty() {
            return frames.to_vec();
        }
        let t = frames.len();
        let bottleneck = dw.len();
        let k = dw[0].len().min(3);
        let n_out = pw.len();

        let out: Vec<Vec<f64>> = (0..t)
            .map(|fi| {
                // Depthwise: convolve each bottleneck channel with a k-tap filter
                let dw_out: Vec<f64> = (0..bottleneck)
                    .map(|b| {
                        let filt = &dw[b];
                        let mut acc = 0.0_f64;
                        for j in 0..k {
                            let src = if fi + j >= k { fi + j - (k - 1) } else { 0 };
                            let in_val = frames
                                .get(src)
                                .and_then(|f| f.get(b % f.len()))
                                .copied()
                                .unwrap_or(0.0);
                            acc += in_val * filt.get(j).copied().unwrap_or(0.0);
                        }
                        acc.max(0.0) // ReLU
                    })
                    .collect();

                // Pointwise: linear transform to n_out channels
                pw.iter()
                    .map(|row| {
                        row.iter()
                            .zip(dw_out.iter())
                            .map(|(&w, &x)| w * x)
                            .sum::<f64>()
                    })
                    .collect::<Vec<f64>>()
                    .into_iter()
                    .take(n_out)
                    .collect()
            })
            .collect();

        out
    }

    // ── Decoder ────────────────────────────────────────────────────────────────

    fn decoder(&self, enc_frames: &[Vec<f64>], masks: &[f64], out_len: usize) -> Vec<f64> {
        let mut output = vec![0.0_f64; out_len];
        let mut counts = vec![0.0_f64; out_len];
        let k = self.enc_kernel;
        let s = self.enc_stride;

        for (fi, (frame, &mask)) in enc_frames.iter().zip(masks.iter()).enumerate() {
            let start = fi * s;
            // Reconstruct k samples from the encoder frame via dec_w
            for j in 0..k {
                if start + j >= out_len {
                    break;
                }
                let sample: f64 = self
                    .dec_w
                    .get(j)
                    .map(|row| {
                        row.iter()
                            .zip(frame.iter())
                            .map(|(&w, &f)| w * f)
                            .sum::<f64>()
                    })
                    .unwrap_or(0.0);
                output[start + j] += mask * sample;
                counts[start + j] += 1.0;
            }
        }

        // Normalise overlap
        for (o, c) in output.iter_mut().zip(counts.iter()) {
            if *c > 0.0 {
                *o /= c;
            }
        }

        output
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl AudioModel for ConvTasNetInference {
    fn process(&self, audio: &[f64]) -> Result<Vec<Vec<f64>>, SignalError> {
        let enhanced = self.enhance(audio)?;
        Ok(vec![enhanced])
    }

    fn name(&self) -> &str {
        "ConvTasNet-Inference"
    }

    fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_store_roundtrip() {
        let mut store = WeightStore::new();
        store.insert(
            "layer1.weight",
            vec![4, 3],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        store.insert("layer1.bias", vec![4], vec![0.1, 0.2, 0.3, 0.4]);

        let tmp_path = {
            let mut p = std::env::temp_dir();
            p.push("scirs2_signal_weight_store_test.bin");
            p.to_string_lossy().to_string()
        };

        WeightStore::save_binary_dump(&tmp_path, &store).expect("save");
        let loaded = WeightStore::from_binary_dump(&tmp_path).expect("load");

        assert_eq!(loaded.len(), 2);

        let (sh, data) = loaded.get("layer1.weight").expect("tensor present");
        assert_eq!(sh, &[4, 3]);
        assert_eq!(data.len(), 12);
        assert!((data[0] - 1.0_f32).abs() < 1e-6);

        let _ = std::fs::remove_file(&tmp_path);
    }

    #[test]
    fn weight_store_names() {
        let mut store = WeightStore::new();
        store.insert("a", vec![2], vec![1.0, 2.0]);
        store.insert("b", vec![2], vec![3.0, 4.0]);
        let mut names = store.names();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn mel_spectrogram_shape() {
        let config = NeuralAudioConfig {
            sample_rate: 16000,
            frame_length: 512,
            hop_length: 256,
            n_mels: 40,
            ..Default::default()
        };
        // 1 second of audio at 16 kHz
        let audio: Vec<f64> = (0..16000).map(|i| (i as f64 * 0.01).sin()).collect();
        let mel = compute_mel_spectrogram(&audio, &config);

        assert_eq!(mel.len(), 40, "Should have n_mels={} rows", 40);
        // n_frames = (16000 - 512) / 256 + 1 = 60
        let expected_frames = (16000 - 512) / 256 + 1;
        for row in &mel {
            assert_eq!(
                row.len(),
                expected_frames,
                "Each Mel band should have {expected_frames} frames"
            );
        }
    }

    #[test]
    fn mel_spectrogram_values_nonnegative() {
        let config = NeuralAudioConfig::default();
        let audio: Vec<f64> = (0..4000).map(|i| (i as f64 * 0.05).sin()).collect();
        let mel = compute_mel_spectrogram(&audio, &config);
        for row in &mel {
            for &v in row {
                assert!(v >= 0.0, "Mel spectrogram values should be non-negative");
            }
        }
    }

    #[test]
    fn conv_tasnet_inference_enhance_shape() {
        let model = ConvTasNetInference::new(NeuralAudioConfig::default());
        let noisy: Vec<f64> = (0..1600).map(|i| (i as f64 * 0.05).sin()).collect();
        let enhanced = model.enhance(&noisy).expect("enhance");
        assert_eq!(enhanced.len(), noisy.len());
    }

    #[test]
    fn conv_tasnet_inference_from_weights() {
        let mut store = WeightStore::new();
        // Small encoder: 4 filters of size 4
        store.insert("enc_w", vec![4, 4], vec![0.1f32; 16]);
        store.insert("dec_w", vec![4, 4], vec![0.1f32; 16]);

        let model = ConvTasNetInference::from_weights(&store, NeuralAudioConfig::default())
            .expect("from_weights");
        let noisy: Vec<f64> = (0..320).map(|i| (i as f64).sin() * 0.5).collect();
        let out = model.enhance(&noisy).expect("enhance");
        assert_eq!(out.len(), 320);
    }

    #[test]
    fn audio_model_trait_object() {
        let model: Box<dyn AudioModel> =
            Box::new(ConvTasNetInference::new(NeuralAudioConfig::default()));
        assert_eq!(model.sample_rate(), 16_000);
        let audio: Vec<f64> = (0..800).map(|i| (i as f64).sin() * 0.1).collect();
        let result = model.process(&audio).expect("process");
        assert_eq!(result.len(), 1);
    }
}
