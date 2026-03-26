//! Speech enhancement using neural-inspired methods.
//!
//! Implements mask-based, DCCRN-style, and spectral-mapping speech enhancement
//! with support for causal (real-time) operation.

use crate::error::{SignalError, SignalResult};

// ── LCG PRNG ────────────────────────────────────────────────────────────────

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 11) as f64 / (1u64 << 53) as f64
    }

    fn kaiming_uniform(&mut self, fan_in: usize) -> f64 {
        let bound = (3.0_f64 / fan_in.max(1) as f64).sqrt();
        self.next_f64() * 2.0 * bound - bound
    }
}

// ── Configuration ───────────────────────────────────────────────────────────

/// Enhancement method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum EnhancementMethod {
    /// Mask-based: estimate complex ratio mask (cRM) from noisy STFT.
    #[default]
    MaskBased,
    /// DCCRN-style: encoder → LSTM → decoder.
    DCCRN,
    /// Direct spectral mapping from noisy to clean magnitude.
    SpectralMapping,
}

/// Configuration for speech enhancement.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EnhancementConfig {
    /// Enhancement method.
    pub method: EnhancementMethod,
    /// Whether to use causal (no future frames) processing.
    pub causal: bool,
    /// Number of future frames to use when not fully causal (0 = fully causal).
    pub lookahead_frames: usize,
    /// FFT size for STFT.
    pub n_fft: usize,
    /// Hop length for STFT.
    pub hop_length: usize,
    /// Number of encoder channels (DCCRN).
    pub encoder_channels: Vec<usize>,
    /// LSTM hidden size (DCCRN).
    pub lstm_hidden: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for EnhancementConfig {
    fn default() -> Self {
        Self {
            method: EnhancementMethod::MaskBased,
            causal: false,
            lookahead_frames: 2,
            n_fft: 512,
            hop_length: 128,
            encoder_channels: vec![16, 32, 64],
            lstm_hidden: 128,
            seed: 42,
        }
    }
}

// ── STFT / iSTFT helpers ───────────────────────────────────────────────────

/// Complex number as (real, imag) tuple.
type Complex = (f64, f64);

fn hann_window(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()))
        .collect()
}

/// Compute STFT. Returns (n_freq, n_frames) complex spectrogram.
fn stft(signal: &[f64], n_fft: usize, hop: usize) -> Vec<Vec<Complex>> {
    let n = signal.len();
    let n_freq = n_fft / 2 + 1;
    let window = hann_window(n_fft);

    let n_frames = if n >= n_fft {
        (n - n_fft) / hop + 1
    } else if n > 0 {
        1
    } else {
        return vec![vec![(0.0, 0.0); 0]; n_freq];
    };

    let mut spec = vec![vec![(0.0, 0.0); n_frames]; n_freq];

    for frame in 0..n_frames {
        let start = frame * hop;
        for k in 0..n_freq {
            let mut re = 0.0;
            let mut im = 0.0;
            for t in 0..n_fft {
                let idx = start + t;
                let sample = if idx < n { signal[idx] } else { 0.0 };
                let w = sample * window[t];
                let angle = -2.0 * std::f64::consts::PI * k as f64 * t as f64 / n_fft as f64;
                re += w * angle.cos();
                im += w * angle.sin();
            }
            spec[k][frame] = (re, im);
        }
    }

    spec
}

/// Inverse STFT with overlap-add.
fn istft(spec: &[Vec<Complex>], n_fft: usize, hop: usize, output_len: usize) -> Vec<f64> {
    let n_freq = spec.len();
    let n_frames = if n_freq > 0 && !spec[0].is_empty() {
        spec[0].len()
    } else {
        return vec![0.0; output_len];
    };

    let window = hann_window(n_fft);
    let mut output = vec![0.0; output_len];
    let mut window_sum = vec![0.0; output_len];

    for frame in 0..n_frames {
        let start = frame * hop;
        // Reconstruct full-length DFT frame from one-sided spectrum
        for t in 0..n_fft {
            if start + t >= output_len {
                break;
            }
            let mut val = 0.0;
            for k in 0..n_freq {
                let (re, im) = spec[k][frame];
                let angle = 2.0 * std::f64::consts::PI * k as f64 * t as f64 / n_fft as f64;
                val += re * angle.cos() - im * angle.sin();
                // Mirror (conjugate symmetry) for k > 0 and k < n_fft/2
                if k > 0 && k < n_fft / 2 {
                    val += re * angle.cos() - im * angle.sin();
                }
            }
            val /= n_fft as f64;
            output[start + t] += val * window[t];
            window_sum[start + t] += window[t] * window[t];
        }
    }

    // Normalize by window sum
    for i in 0..output_len {
        if window_sum[i] > 1e-8 {
            output[i] /= window_sum[i];
        }
    }

    output
}

/// Complex magnitude.
fn cmag(c: Complex) -> f64 {
    (c.0 * c.0 + c.1 * c.1).sqrt()
}

/// Complex multiply.
fn cmul(a: Complex, b: Complex) -> Complex {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

// ── Dense layer ─────────────────────────────────────────────────────────────

struct Dense {
    weight: Vec<Vec<f64>>,
    bias: Vec<f64>,
}

impl Dense {
    fn new(out_dim: usize, in_dim: usize, rng: &mut Lcg) -> Self {
        let weight = (0..out_dim)
            .map(|_| (0..in_dim).map(|_| rng.kaiming_uniform(in_dim)).collect())
            .collect();
        let bias = (0..out_dim)
            .map(|_| rng.kaiming_uniform(in_dim) * 0.01)
            .collect();
        Self { weight, bias }
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        self.weight
            .iter()
            .zip(self.bias.iter())
            .map(|(w, &b)| {
                let sum: f64 = w.iter().zip(input.iter()).map(|(&wi, &xi)| wi * xi).sum();
                sum + b
            })
            .collect()
    }
}

// ── LSTM layer (simplified single-layer) ────────────────────────────────────

struct LstmLayer {
    /// Combined input-hidden weight: 4*hidden x (input_size + hidden_size)
    w: Vec<Vec<f64>>,
    b: Vec<f64>,
    hidden_size: usize,
}

impl LstmLayer {
    fn new(input_size: usize, hidden_size: usize, rng: &mut Lcg) -> Self {
        let total_in = input_size + hidden_size;
        let gates = 4 * hidden_size;
        let w = (0..gates)
            .map(|_| {
                (0..total_in)
                    .map(|_| rng.kaiming_uniform(total_in))
                    .collect()
            })
            .collect();
        let b = (0..gates)
            .map(|_| rng.kaiming_uniform(total_in) * 0.01)
            .collect();
        Self { w, b, hidden_size }
    }

    /// Process a sequence of frames: (n_frames, input_size) → (n_frames, hidden_size)
    fn forward(&self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n_frames = input.len();
        let hs = self.hidden_size;
        let mut h = vec![0.0; hs];
        let mut c = vec![0.0; hs];
        let mut output = Vec::with_capacity(n_frames);

        for frame in input {
            // Concatenate [frame, h]
            let mut combined = frame.clone();
            combined.extend_from_slice(&h);

            // Compute gates
            let mut gates = vec![0.0; 4 * hs];
            for i in 0..4 * hs {
                let mut val = self.b[i];
                let wlen = self.w[i].len();
                for j in 0..combined.len().min(wlen) {
                    val += self.w[i][j] * combined[j];
                }
                gates[i] = val;
            }

            // Split into i, f, g, o gates
            for j in 0..hs {
                let i_gate = 1.0 / (1.0 + (-gates[j]).exp());
                let f_gate = 1.0 / (1.0 + (-gates[hs + j]).exp());
                let g_gate = gates[2 * hs + j].tanh();
                let o_gate = 1.0 / (1.0 + (-gates[3 * hs + j]).exp());

                c[j] = f_gate * c[j] + i_gate * g_gate;
                h[j] = o_gate * c[j].tanh();
            }

            output.push(h.clone());
        }

        output
    }
}

// ── Mask-based enhancement ──────────────────────────────────────────────────

struct MaskBasedModel {
    /// Maps concatenated (real, imag) spectrogram features to mask.
    encoder1: Dense,
    encoder2: Dense,
    mask_real: Dense,
    mask_imag: Dense,
}

impl MaskBasedModel {
    fn new(n_freq: usize, rng: &mut Lcg) -> Self {
        let in_dim = n_freq * 2; // real + imag
        let hidden = 256.min(n_freq * 2);
        Self {
            encoder1: Dense::new(hidden, in_dim, rng),
            encoder2: Dense::new(hidden, hidden, rng),
            mask_real: Dense::new(n_freq, hidden, rng),
            mask_imag: Dense::new(n_freq, hidden, rng),
        }
    }

    /// Estimate complex ratio mask for each frame.
    fn estimate_mask(&self, spec: &[Vec<Complex>]) -> Vec<Vec<Complex>> {
        let n_freq = spec.len();
        let n_frames = if n_freq > 0 {
            spec[0].len()
        } else {
            return Vec::new();
        };

        let mut masks = vec![vec![(0.0, 0.0); n_frames]; n_freq];

        for frame in 0..n_frames {
            // Build input: [re_0, re_1, ..., im_0, im_1, ...]
            let mut input = Vec::with_capacity(n_freq * 2);
            for k in 0..n_freq {
                input.push(spec[k][frame].0);
            }
            for k in 0..n_freq {
                input.push(spec[k][frame].1);
            }

            let h = self.encoder1.forward(&input);
            let h: Vec<f64> = h.iter().map(|&v| v.max(0.0)).collect(); // ReLU
            let h = self.encoder2.forward(&h);
            let h: Vec<f64> = h.iter().map(|&v| v.max(0.0)).collect();

            let mask_r = self.mask_real.forward(&h);
            let mask_i = self.mask_imag.forward(&h);

            // Apply bounded tanh to mask
            for k in 0..n_freq {
                let mr = if k < mask_r.len() {
                    mask_r[k].tanh()
                } else {
                    0.0
                };
                let mi = if k < mask_i.len() {
                    mask_i[k].tanh()
                } else {
                    0.0
                };
                masks[k][frame] = (mr, mi);
            }
        }

        masks
    }
}

// ── DCCRN-style model ───────────────────────────────────────────────────────

struct DccrnModel {
    /// Encoder: dense layers simulating strided convolutions.
    encoder_layers: Vec<Dense>,
    /// LSTM for sequence modeling.
    lstm: LstmLayer,
    /// Decoder: dense layers simulating transposed convolutions.
    decoder_layers: Vec<Dense>,
    n_freq: usize,
}

impl DccrnModel {
    fn new(n_freq: usize, config: &EnhancementConfig, rng: &mut Lcg) -> Self {
        let mut encoder_layers = Vec::new();
        let mut in_dim = n_freq * 2;

        for &ch in &config.encoder_channels {
            encoder_layers.push(Dense::new(ch, in_dim, rng));
            in_dim = ch;
        }

        let lstm = LstmLayer::new(in_dim, config.lstm_hidden, rng);

        let mut decoder_layers = Vec::new();
        let mut dec_in = config.lstm_hidden;
        for &ch in config.encoder_channels.iter().rev().skip(1) {
            decoder_layers.push(Dense::new(ch, dec_in, rng));
            dec_in = ch;
        }
        // Final decoder maps back to 2*n_freq (real + imag mask)
        decoder_layers.push(Dense::new(n_freq * 2, dec_in, rng));

        Self {
            encoder_layers,
            lstm,
            decoder_layers,
            n_freq,
        }
    }

    fn enhance_spec(
        &self,
        spec: &[Vec<Complex>],
        causal: bool,
        lookahead: usize,
    ) -> Vec<Vec<Complex>> {
        let n_freq = spec.len();
        let n_frames = if n_freq > 0 {
            spec[0].len()
        } else {
            return Vec::new();
        };

        // Build frame features
        let mut frames: Vec<Vec<f64>> = Vec::with_capacity(n_frames);
        for frame in 0..n_frames {
            let mut feat = Vec::with_capacity(n_freq * 2);
            for k in 0..n_freq {
                feat.push(spec[k][frame].0);
            }
            for k in 0..n_freq {
                feat.push(spec[k][frame].1);
            }
            frames.push(feat);
        }

        // Encode
        let mut encoded: Vec<Vec<f64>> = frames;
        for layer in &self.encoder_layers {
            encoded = encoded
                .iter()
                .map(|f| {
                    let out = layer.forward(f);
                    out.iter().map(|&v| v.max(0.0)).collect()
                })
                .collect();
        }

        // LSTM (process entire sequence or causally)
        let lstm_out = if causal {
            // Causal: process frame by frame, only using past + limited lookahead
            let mut causal_frames = Vec::with_capacity(n_frames);
            for i in 0..n_frames {
                let end = (i + 1 + lookahead).min(n_frames);
                let chunk = &encoded[..end];
                let out = self.lstm.forward(chunk);
                if let Some(last) = out.last() {
                    causal_frames.push(last.clone());
                } else {
                    causal_frames.push(vec![0.0; self.lstm.hidden_size]);
                }
            }
            causal_frames
        } else {
            self.lstm.forward(&encoded)
        };

        // Decode
        let mut decoded = lstm_out;
        for layer in &self.decoder_layers {
            decoded = decoded
                .iter()
                .map(|f| {
                    let out = layer.forward(f);
                    out.iter().map(|&v| v.tanh()).collect()
                })
                .collect();
        }

        // Apply estimated mask
        let mut enhanced = vec![vec![(0.0, 0.0); n_frames]; n_freq];
        for frame in 0..n_frames {
            let mask = &decoded[frame];
            for k in 0..n_freq {
                let mr = if k < mask.len() { mask[k] } else { 0.0 };
                let mi = if n_freq + k < mask.len() {
                    mask[n_freq + k]
                } else {
                    0.0
                };
                enhanced[k][frame] = cmul(spec[k][frame], (mr, mi));
            }
        }

        enhanced
    }
}

// ── Spectral mapping model ──────────────────────────────────────────────────

struct SpectralMappingModel {
    layer1: Dense,
    layer2: Dense,
    layer3: Dense,
    n_freq: usize,
}

impl SpectralMappingModel {
    fn new(n_freq: usize, rng: &mut Lcg) -> Self {
        let hidden = 256.min(n_freq * 2);
        Self {
            layer1: Dense::new(hidden, n_freq, rng),
            layer2: Dense::new(hidden, hidden, rng),
            layer3: Dense::new(n_freq, hidden, rng),
            n_freq,
        }
    }

    fn map_spectrum(&self, spec: &[Vec<Complex>]) -> Vec<Vec<Complex>> {
        let n_freq = spec.len();
        let n_frames = if n_freq > 0 {
            spec[0].len()
        } else {
            return Vec::new();
        };

        let mut enhanced = vec![vec![(0.0, 0.0); n_frames]; n_freq];

        for frame in 0..n_frames {
            // Input: magnitude spectrum
            let mag: Vec<f64> = (0..n_freq).map(|k| cmag(spec[k][frame])).collect();

            let h = self.layer1.forward(&mag);
            let h: Vec<f64> = h.iter().map(|&v| v.max(0.0)).collect();
            let h = self.layer2.forward(&h);
            let h: Vec<f64> = h.iter().map(|&v| v.max(0.0)).collect();
            let clean_mag = self.layer3.forward(&h);

            // Use estimated magnitude with original phase
            for k in 0..n_freq {
                let m = cmag(spec[k][frame]);
                let est_m = if k < clean_mag.len() {
                    clean_mag[k].abs()
                } else {
                    m
                };
                if m > 1e-10 {
                    let scale = est_m / m;
                    enhanced[k][frame] = (spec[k][frame].0 * scale, spec[k][frame].1 * scale);
                }
            }
        }

        enhanced
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Speech enhancement using neural-inspired methods.
pub struct SpeechEnhancement {
    config: EnhancementConfig,
}

impl SpeechEnhancement {
    /// Create a new speech enhancement instance.
    pub fn new(config: EnhancementConfig) -> Self {
        Self { config }
    }

    /// Return the configuration.
    pub fn config(&self) -> &EnhancementConfig {
        &self.config
    }
}

/// Enhance a noisy signal using the given configuration.
///
/// Returns an enhanced waveform of the same length as the input.
pub fn enhance(noisy_signal: &[f64], config: &EnhancementConfig) -> SignalResult<Vec<f64>> {
    let n = noisy_signal.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    if config.n_fft == 0 || config.hop_length == 0 {
        return Err(SignalError::InvalidArgument(
            "n_fft and hop_length must be positive".to_string(),
        ));
    }

    let n_fft = config.n_fft;
    let hop = config.hop_length;
    let n_freq = n_fft / 2 + 1;

    // Compute STFT of noisy signal
    let noisy_spec = stft(noisy_signal, n_fft, hop);

    let mut rng = Lcg::new(config.seed);

    // Apply enhancement method
    let enhanced_spec = match config.method {
        EnhancementMethod::MaskBased => {
            let model = MaskBasedModel::new(n_freq, &mut rng);
            let masks = model.estimate_mask(&noisy_spec);
            // Apply complex ratio mask
            let mut enhanced = vec![vec![(0.0, 0.0); noisy_spec[0].len()]; n_freq];
            let n_frames = noisy_spec[0].len();
            for k in 0..n_freq {
                for f in 0..n_frames {
                    enhanced[k][f] = cmul(noisy_spec[k][f], masks[k][f]);
                }
            }
            enhanced
        }
        EnhancementMethod::DCCRN => {
            let model = DccrnModel::new(n_freq, config, &mut rng);
            model.enhance_spec(&noisy_spec, config.causal, config.lookahead_frames)
        }
        EnhancementMethod::SpectralMapping => {
            let model = SpectralMappingModel::new(n_freq, &mut rng);
            model.map_spectrum(&noisy_spec)
        }
        _ => {
            // Fallback: pass-through
            noisy_spec.clone()
        }
    };

    // Inverse STFT
    let output = istft(&enhanced_spec, n_fft, hop, n);

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_signal(freq: f64, sr: u32, dur: f64) -> Vec<f64> {
        let n = (sr as f64 * dur) as usize;
        (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sr as f64).sin())
            .collect()
    }

    #[test]
    fn test_enhancement_output_length_mask() {
        let config = EnhancementConfig {
            method: EnhancementMethod::MaskBased,
            n_fft: 128,
            hop_length: 32,
            seed: 77,
            ..Default::default()
        };
        let noisy = sine_signal(440.0, 8000, 0.1);
        let n = noisy.len();
        let enhanced = enhance(&noisy, &config).expect("enhance should work");
        assert_eq!(enhanced.len(), n, "Output length must match input length");
    }

    #[test]
    fn test_enhancement_output_length_dccrn() {
        let config = EnhancementConfig {
            method: EnhancementMethod::DCCRN,
            n_fft: 128,
            hop_length: 32,
            encoder_channels: vec![16, 32],
            lstm_hidden: 16,
            seed: 77,
            ..Default::default()
        };
        let noisy = sine_signal(440.0, 8000, 0.1);
        let n = noisy.len();
        let enhanced = enhance(&noisy, &config).expect("enhance should work");
        assert_eq!(enhanced.len(), n);
    }

    #[test]
    fn test_enhancement_output_length_spectral_mapping() {
        let config = EnhancementConfig {
            method: EnhancementMethod::SpectralMapping,
            n_fft: 128,
            hop_length: 32,
            seed: 77,
            ..Default::default()
        };
        let noisy = sine_signal(440.0, 8000, 0.1);
        let n = noisy.len();
        let enhanced = enhance(&noisy, &config).expect("enhance should work");
        assert_eq!(enhanced.len(), n);
    }

    #[test]
    fn test_enhancement_causal_mode() {
        let config = EnhancementConfig {
            method: EnhancementMethod::DCCRN,
            causal: true,
            lookahead_frames: 0,
            n_fft: 128,
            hop_length: 32,
            encoder_channels: vec![16],
            lstm_hidden: 16,
            seed: 77,
            ..Default::default()
        };
        let noisy = sine_signal(440.0, 8000, 0.1);
        let enhanced = enhance(&noisy, &config).expect("enhance should work");
        assert_eq!(enhanced.len(), noisy.len());
    }

    #[test]
    fn test_enhancement_empty_input() {
        let config = EnhancementConfig::default();
        let result = enhance(&[], &config).expect("should handle empty");
        assert!(result.is_empty());
    }

    #[test]
    fn test_mask_based_stft_same_shape() {
        let config = EnhancementConfig {
            method: EnhancementMethod::MaskBased,
            n_fft: 128,
            hop_length: 32,
            seed: 42,
            ..Default::default()
        };
        let noisy = sine_signal(440.0, 8000, 0.05);
        let n_freq = config.n_fft / 2 + 1;

        let noisy_spec = stft(&noisy, config.n_fft, config.hop_length);
        assert_eq!(noisy_spec.len(), n_freq);

        // After enhancement, output STFT would have same shape
        let _ = enhance(&noisy, &config).expect("should work");
    }

    #[test]
    fn test_speech_enhancement_struct() {
        let config = EnhancementConfig::default();
        let se = SpeechEnhancement::new(config);
        assert!(!se.config().causal);
    }
}
