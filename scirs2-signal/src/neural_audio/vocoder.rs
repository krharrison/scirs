//! Neural vocoder implementations: WaveNet, MelGAN, HiFi-GAN style synthesis.
//!
//! Provides mel-spectrogram to waveform conversion using neural vocoder
//! architectures. All models use deterministic weight initialization
//! (no pre-trained weights required) for demonstration and testing.

use crate::error::{SignalError, SignalResult};

// ── LCG PRNG (same as conv_tasnet) ─────────────────────────────────────────

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

// ── Vocoder type ────────────────────────────────────────────────────────────

/// Type of neural vocoder architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum VocoderType {
    /// WaveNet-style: dilated causal convolution stack.
    WaveNet,
    /// MelGAN-style: transposed convolution upsampling + residual blocks.
    MelGAN,
    /// HiFi-GAN: multi-receptive field fusion (MRF) blocks.
    #[default]
    HiFiGAN,
}

// ── Configuration ───────────────────────────────────────────────────────────

/// Configuration for the mel spectrogram extraction helper.
#[derive(Debug, Clone)]
pub struct MelConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// FFT window size in samples.
    pub n_fft: usize,
    /// Hop length in samples.
    pub hop_length: usize,
    /// Number of mel bands.
    pub n_mels: usize,
    /// Minimum frequency for mel filter bank (Hz).
    pub f_min: f64,
    /// Maximum frequency for mel filter bank (Hz, 0 = Nyquist).
    pub f_max: f64,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22050,
            n_fft: 1024,
            hop_length: 256,
            n_mels: 80,
            f_min: 0.0,
            f_max: 0.0,
        }
    }
}

/// Configuration for the neural vocoder.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct VocoderConfig {
    /// Vocoder architecture type.
    pub model_type: VocoderType,
    /// Target sample rate in Hz.
    pub sample_rate: u32,
    /// Hop length (samples per mel frame).
    pub hop_length: usize,
    /// Number of mel channels expected as input.
    pub n_mels: usize,
    /// Upsample factors (product must equal hop_length).
    pub upsample_factors: Vec<usize>,
    /// Number of WaveNet residual channels.
    pub wavenet_residual_channels: usize,
    /// WaveNet dilation cycle length (dilations: 1,2,...,2^(cycle_len-1)).
    pub wavenet_dilation_cycle: usize,
    /// Number of WaveNet dilation cycles (repeats).
    pub wavenet_n_cycles: usize,
    /// HiFi-GAN MRF kernel sizes.
    pub hifigan_kernel_sizes: Vec<usize>,
    /// Random seed for deterministic weight init.
    pub seed: u64,
}

impl Default for VocoderConfig {
    fn default() -> Self {
        Self {
            model_type: VocoderType::HiFiGAN,
            sample_rate: 22050,
            hop_length: 256,
            n_mels: 80,
            upsample_factors: vec![8, 8, 2, 2],
            wavenet_residual_channels: 64,
            wavenet_dilation_cycle: 10,
            wavenet_n_cycles: 3,
            hifigan_kernel_sizes: vec![3, 7, 11],
            seed: 42,
        }
    }
}

// ── Internal layer primitives ───────────────────────────────────────────────

/// 1-D convolution weights: (out_ch, in_ch, kernel_size).
struct Conv1d {
    weight: Vec<Vec<Vec<f64>>>,
    bias: Vec<f64>,
    stride: usize,
    dilation: usize,
}

impl Conv1d {
    fn new(
        out_ch: usize,
        in_ch: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        rng: &mut Lcg,
    ) -> Self {
        let fan_in = in_ch * kernel_size;
        let weight = (0..out_ch)
            .map(|_| {
                (0..in_ch)
                    .map(|_| {
                        (0..kernel_size)
                            .map(|_| rng.kaiming_uniform(fan_in))
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let bias = (0..out_ch)
            .map(|_| rng.kaiming_uniform(fan_in) * 0.01)
            .collect();
        Self {
            weight,
            bias,
            stride,
            dilation,
        }
    }

    /// Forward: input shape (in_ch, time) → output shape (out_ch, out_time).
    fn forward(&self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let in_ch = self.weight[0].len();
        let ks = self.weight[0][0].len();
        let out_ch = self.weight.len();
        let in_len = if input.is_empty() { 0 } else { input[0].len() };
        let effective_ks = (ks - 1) * self.dilation + 1;
        let out_len = if in_len >= effective_ks {
            (in_len - effective_ks) / self.stride + 1
        } else {
            0
        };
        let mut output = vec![vec![0.0; out_len]; out_ch];
        for oc in 0..out_ch {
            for t in 0..out_len {
                let mut val = self.bias[oc];
                for ic in 0..in_ch.min(input.len()) {
                    for k in 0..ks {
                        let idx = t * self.stride + k * self.dilation;
                        if idx < in_len {
                            val += self.weight[oc][ic][k] * input[ic][idx];
                        }
                    }
                }
                output[oc][t] = val;
            }
        }
        output
    }

    /// Forward with causal (left) padding so output length == ceil(input_len / stride).
    fn forward_causal(&self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let ks = self.weight[0][0].len();
        let effective_ks = (ks - 1) * self.dilation + 1;
        let pad = effective_ks - 1;
        let padded = pad_left(input, pad);
        self.forward(&padded)
    }
}

/// Transposed 1-D convolution (for upsampling).
struct ConvTranspose1d {
    weight: Vec<Vec<Vec<f64>>>,
    bias: Vec<f64>,
    stride: usize,
}

impl ConvTranspose1d {
    fn new(out_ch: usize, in_ch: usize, kernel_size: usize, stride: usize, rng: &mut Lcg) -> Self {
        let fan_in = in_ch * kernel_size;
        let weight = (0..in_ch)
            .map(|_| {
                (0..out_ch)
                    .map(|_| {
                        (0..kernel_size)
                            .map(|_| rng.kaiming_uniform(fan_in))
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let bias = (0..out_ch)
            .map(|_| rng.kaiming_uniform(fan_in) * 0.01)
            .collect();
        Self {
            weight,
            bias,
            stride,
        }
    }

    /// Forward: input (in_ch, in_len) → output (out_ch, out_len).
    /// out_len = (in_len - 1) * stride + kernel_size
    fn forward(&self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let in_ch = self.weight.len();
        let out_ch = self.weight[0].len();
        let ks = self.weight[0][0].len();
        let in_len = if input.is_empty() { 0 } else { input[0].len() };
        let out_len = if in_len > 0 {
            (in_len - 1) * self.stride + ks
        } else {
            0
        };
        let mut output = vec![vec![0.0; out_len]; out_ch];
        for ic in 0..in_ch.min(input.len()) {
            for t in 0..in_len {
                for oc in 0..out_ch {
                    for k in 0..ks {
                        let idx = t * self.stride + k;
                        if idx < out_len {
                            output[oc][idx] += self.weight[ic][oc][k] * input[ic][t];
                        }
                    }
                }
            }
        }
        for oc in 0..out_ch {
            for t in 0..out_len {
                output[oc][t] += self.bias[oc];
            }
        }
        output
    }
}

fn pad_left(input: &[Vec<f64>], pad: usize) -> Vec<Vec<f64>> {
    input
        .iter()
        .map(|ch| {
            let mut padded = vec![0.0; pad + ch.len()];
            padded[pad..].copy_from_slice(ch);
            padded
        })
        .collect()
}

fn leaky_relu(x: f64, alpha: f64) -> f64 {
    if x >= 0.0 {
        x
    } else {
        alpha * x
    }
}

fn tanh_activate(x: f64) -> f64 {
    x.tanh()
}

// ── WaveNet-style vocoder ───────────────────────────────────────────────────

/// WaveNet-style vocoder with dilated causal convolution stacks.
struct WaveNetVocoder {
    /// Mel conditioning projection: n_mels → residual_channels.
    mel_proj: Conv1d,
    /// Dilated causal conv layers (gated: filter + gate).
    dilated_filter: Vec<Conv1d>,
    dilated_gate: Vec<Conv1d>,
    /// 1x1 residual and skip projections.
    res_proj: Vec<Conv1d>,
    skip_proj: Vec<Conv1d>,
    /// Final output layers.
    output_conv1: Conv1d,
    output_conv2: Conv1d,
    /// Upsample network for mel conditioning.
    upsample: Vec<ConvTranspose1d>,
    residual_channels: usize,
    n_layers: usize,
}

impl WaveNetVocoder {
    fn new(config: &VocoderConfig) -> Self {
        let mut rng = Lcg::new(config.seed);
        let rc = config.wavenet_residual_channels;
        let n_layers = config.wavenet_dilation_cycle * config.wavenet_n_cycles;

        let mel_proj = Conv1d::new(rc, config.n_mels, 1, 1, 1, &mut rng);

        let mut dilated_filter = Vec::with_capacity(n_layers);
        let mut dilated_gate = Vec::with_capacity(n_layers);
        let mut res_proj = Vec::with_capacity(n_layers);
        let mut skip_proj = Vec::with_capacity(n_layers);

        for i in 0..n_layers {
            let dilation = 1 << (i % config.wavenet_dilation_cycle);
            dilated_filter.push(Conv1d::new(rc, rc, 3, 1, dilation, &mut rng));
            dilated_gate.push(Conv1d::new(rc, rc, 3, 1, dilation, &mut rng));
            res_proj.push(Conv1d::new(rc, rc, 1, 1, 1, &mut rng));
            skip_proj.push(Conv1d::new(rc, rc, 1, 1, 1, &mut rng));
        }

        let output_conv1 = Conv1d::new(rc, rc, 1, 1, 1, &mut rng);
        let output_conv2 = Conv1d::new(1, rc, 1, 1, 1, &mut rng);

        // Upsample layers from mel to waveform rate
        let mut upsample = Vec::new();
        for &factor in &config.upsample_factors {
            upsample.push(ConvTranspose1d::new(rc, rc, factor * 2, factor, &mut rng));
        }

        Self {
            mel_proj,
            dilated_filter,
            dilated_gate,
            res_proj,
            skip_proj,
            output_conv1,
            output_conv2,
            upsample,
            residual_channels: rc,
            n_layers,
        }
    }

    fn synthesize(&self, mel: &[Vec<f64>]) -> SignalResult<Vec<f64>> {
        if mel.is_empty() {
            return Ok(Vec::new());
        }
        // mel shape: (n_mels, n_frames)
        // Project mel → (rc, n_frames)
        let mut cond = self.mel_proj.forward(mel);

        // Upsample condition to waveform rate
        for up in &self.upsample {
            cond = up.forward(&cond);
        }

        let out_len = cond[0].len();
        let rc = self.residual_channels;

        // Initialize hidden state with zeros
        let mut hidden = vec![vec![0.0; out_len]; rc];

        // Add conditioning
        for c in 0..rc.min(cond.len()) {
            for t in 0..out_len.min(hidden[c].len()) {
                hidden[c][t] += cond[c][t];
            }
        }

        let mut skip_sum = vec![vec![0.0; out_len]; rc];

        for layer in 0..self.n_layers {
            let filter_out = self.dilated_filter[layer].forward_causal(&hidden);
            let gate_out = self.dilated_gate[layer].forward_causal(&hidden);

            // Gated activation: tanh(filter) * sigmoid(gate)
            let len = filter_out[0].len().min(out_len);
            let mut gated = vec![vec![0.0; len]; rc];
            for c in 0..rc.min(filter_out.len()).min(gate_out.len()) {
                for t in 0..len {
                    let f = tanh_activate(filter_out[c][t]);
                    let g = 1.0 / (1.0 + (-gate_out[c][t]).exp()); // sigmoid
                    gated[c][t] = f * g;
                }
            }

            let residual = self.res_proj[layer].forward(&gated);
            let skip = self.skip_proj[layer].forward(&gated);

            // Residual connection
            for c in 0..rc {
                let rlen = residual.get(c).map_or(0, |v| v.len());
                for t in 0..out_len.min(rlen) {
                    hidden[c][t] += residual[c][t];
                }
            }

            // Accumulate skip
            for c in 0..rc {
                let slen = skip.get(c).map_or(0, |v| v.len());
                for t in 0..out_len.min(slen) {
                    skip_sum[c][t] += skip[c][t];
                }
            }
        }

        // Apply ReLU then 1x1 conv
        for c in 0..rc {
            for t in 0..out_len {
                skip_sum[c][t] = skip_sum[c][t].max(0.0);
            }
        }

        let h = self.output_conv1.forward(&skip_sum);
        let h_relu: Vec<Vec<f64>> = h
            .iter()
            .map(|ch| ch.iter().map(|&v| v.max(0.0)).collect())
            .collect();
        let out = self.output_conv2.forward(&h_relu);

        if out.is_empty() || out[0].is_empty() {
            return Ok(vec![0.0; out_len]);
        }

        // Tanh output
        Ok(out[0].iter().map(|&v| tanh_activate(v)).collect())
    }

    /// Compute the receptive field of the dilated causal conv stack.
    fn receptive_field(&self, dilation_cycle: usize, n_cycles: usize) -> usize {
        // Each layer with kernel 3 and dilation d adds 2*d to the receptive field
        let mut rf = 1usize;
        for cycle in 0..n_cycles {
            for i in 0..dilation_cycle {
                let d = 1usize << i;
                let _ = cycle; // used only for counting
                rf += 2 * d; // kernel=3 → (3-1)*d = 2*d new samples
            }
        }
        rf
    }
}

// ── MelGAN-style vocoder ────────────────────────────────────────────────────

struct MelGANVocoder {
    /// Initial projection.
    pre_conv: Conv1d,
    /// Upsample layers (transposed convolutions).
    upsample_layers: Vec<ConvTranspose1d>,
    /// Residual blocks after each upsample.
    res_blocks: Vec<Vec<Conv1d>>,
    /// Final output conv.
    post_conv: Conv1d,
    channels: usize,
}

impl MelGANVocoder {
    fn new(config: &VocoderConfig) -> Self {
        let mut rng = Lcg::new(config.seed);
        let ch = 256;

        let pre_conv = Conv1d::new(ch, config.n_mels, 7, 1, 1, &mut rng);

        let mut upsample_layers = Vec::new();
        let mut res_blocks = Vec::new();
        let mut cur_ch = ch;

        for &factor in &config.upsample_factors {
            let next_ch = cur_ch / 2;
            let next_ch = next_ch.max(16);
            upsample_layers.push(ConvTranspose1d::new(
                next_ch,
                cur_ch,
                factor * 2,
                factor,
                &mut rng,
            ));
            // 3 residual conv pairs per upsample stage
            let mut block = Vec::new();
            for dil in &[1usize, 3, 9] {
                block.push(Conv1d::new(next_ch, next_ch, 3, 1, *dil, &mut rng));
                block.push(Conv1d::new(next_ch, next_ch, 3, 1, 1, &mut rng));
            }
            res_blocks.push(block);
            cur_ch = next_ch;
        }

        let post_conv = Conv1d::new(1, cur_ch, 7, 1, 1, &mut rng);

        Self {
            pre_conv,
            upsample_layers,
            res_blocks,
            post_conv,
            channels: ch,
        }
    }

    fn synthesize(&self, mel: &[Vec<f64>]) -> SignalResult<Vec<f64>> {
        if mel.is_empty() {
            return Ok(Vec::new());
        }

        let mut x = self.pre_conv.forward_causal(mel);

        // Apply leaky ReLU
        for ch in &mut x {
            for v in ch.iter_mut() {
                *v = leaky_relu(*v, 0.2);
            }
        }

        for (i, up) in self.upsample_layers.iter().enumerate() {
            x = up.forward(&x);

            // Leaky ReLU
            for ch in &mut x {
                for v in ch.iter_mut() {
                    *v = leaky_relu(*v, 0.2);
                }
            }

            // Residual blocks (pairs of dilated conv)
            if i < self.res_blocks.len() {
                let blocks = &self.res_blocks[i];
                for pair in blocks.chunks(2) {
                    let h = pair[0].forward_causal(&x);
                    let h_act: Vec<Vec<f64>> = h
                        .iter()
                        .map(|ch| ch.iter().map(|&v| leaky_relu(v, 0.2)).collect())
                        .collect();
                    let h2 = if pair.len() > 1 {
                        pair[1].forward_causal(&h_act)
                    } else {
                        h_act
                    };
                    // Residual addition
                    let n_ch = x.len().min(h2.len());
                    for c in 0..n_ch {
                        let len = x[c].len().min(h2[c].len());
                        for t in 0..len {
                            x[c][t] += h2[c][t];
                        }
                    }
                }
            }
        }

        // Leaky ReLU + post conv
        for ch in &mut x {
            for v in ch.iter_mut() {
                *v = leaky_relu(*v, 0.2);
            }
        }

        let out = self.post_conv.forward_causal(&x);

        if out.is_empty() || out[0].is_empty() {
            return Ok(Vec::new());
        }

        Ok(out[0].iter().map(|&v| tanh_activate(v)).collect())
    }
}

// ── HiFi-GAN-style vocoder ─────────────────────────────────────────────────

/// Multi-receptive field fusion (MRF) block.
struct MrfBlock {
    /// Residual conv pairs for each kernel size.
    res_convs: Vec<(Conv1d, Conv1d)>,
}

impl MrfBlock {
    fn new(channels: usize, kernel_sizes: &[usize], rng: &mut Lcg) -> Self {
        let res_convs = kernel_sizes
            .iter()
            .map(|&ks| {
                let c1 = Conv1d::new(channels, channels, ks, 1, 1, rng);
                let c2 = Conv1d::new(channels, channels, ks, 1, 1, rng);
                (c1, c2)
            })
            .collect();
        Self { res_convs }
    }

    fn forward(&self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n_ch = input.len();
        let len = if input.is_empty() { 0 } else { input[0].len() };

        // Sum of residual branches
        let mut out = vec![vec![0.0; len]; n_ch];
        let n_branches = self.res_convs.len().max(1);

        for (c1, c2) in &self.res_convs {
            let h: Vec<Vec<f64>> = input
                .iter()
                .map(|ch| ch.iter().map(|&v| leaky_relu(v, 0.1)).collect())
                .collect();
            let h = c1.forward_causal(&h);
            let h: Vec<Vec<f64>> = h
                .iter()
                .map(|ch| ch.iter().map(|&v| leaky_relu(v, 0.1)).collect())
                .collect();
            let h = c2.forward_causal(&h);

            for c in 0..n_ch.min(h.len()) {
                for t in 0..len.min(h[c].len()) {
                    out[c][t] += h[c][t];
                }
            }
        }

        // Average + residual
        for c in 0..n_ch {
            for t in 0..len {
                out[c][t] = out[c][t] / n_branches as f64 + input[c][t];
            }
        }

        out
    }
}

struct HiFiGANVocoder {
    pre_conv: Conv1d,
    upsample_layers: Vec<ConvTranspose1d>,
    mrf_blocks: Vec<MrfBlock>,
    post_conv: Conv1d,
}

impl HiFiGANVocoder {
    fn new(config: &VocoderConfig) -> Self {
        let mut rng = Lcg::new(config.seed);
        let ch = 256;

        let pre_conv = Conv1d::new(ch, config.n_mels, 7, 1, 1, &mut rng);

        let mut upsample_layers = Vec::new();
        let mut mrf_blocks = Vec::new();
        let mut cur_ch = ch;

        for &factor in &config.upsample_factors {
            let next_ch = cur_ch / 2;
            let next_ch = next_ch.max(16);
            upsample_layers.push(ConvTranspose1d::new(
                next_ch,
                cur_ch,
                factor * 2,
                factor,
                &mut rng,
            ));
            mrf_blocks.push(MrfBlock::new(
                next_ch,
                &config.hifigan_kernel_sizes,
                &mut rng,
            ));
            cur_ch = next_ch;
        }

        let post_conv = Conv1d::new(1, cur_ch, 7, 1, 1, &mut rng);

        Self {
            pre_conv,
            upsample_layers,
            mrf_blocks,
            post_conv,
        }
    }

    fn synthesize(&self, mel: &[Vec<f64>]) -> SignalResult<Vec<f64>> {
        if mel.is_empty() {
            return Ok(Vec::new());
        }

        let mut x = self.pre_conv.forward_causal(mel);

        for (i, up) in self.upsample_layers.iter().enumerate() {
            // LeakyReLU before upsample
            for ch in &mut x {
                for v in ch.iter_mut() {
                    *v = leaky_relu(*v, 0.1);
                }
            }

            x = up.forward(&x);

            // MRF block
            if i < self.mrf_blocks.len() {
                x = self.mrf_blocks[i].forward(&x);
            }
        }

        // LeakyReLU + post conv + tanh
        for ch in &mut x {
            for v in ch.iter_mut() {
                *v = leaky_relu(*v, 0.1);
            }
        }

        let out = self.post_conv.forward_causal(&x);

        if out.is_empty() || out[0].is_empty() {
            return Ok(Vec::new());
        }

        Ok(out[0].iter().map(|&v| tanh_activate(v)).collect())
    }
}

// ── Public NeuralVocoder ────────────────────────────────────────────────────

/// Vocoder inner representation.
enum VocoderInner {
    WaveNet(WaveNetVocoder),
    MelGAN(MelGANVocoder),
    HiFiGAN(HiFiGANVocoder),
}

/// Neural vocoder for mel-spectrogram to waveform synthesis.
///
/// Supports WaveNet, MelGAN, and HiFi-GAN architectures using randomly
/// initialized weights (for demonstration; real use requires pre-trained weights).
pub struct NeuralVocoder {
    inner: VocoderInner,
    config: VocoderConfig,
}

impl NeuralVocoder {
    /// Create a new vocoder from configuration.
    pub fn new(config: VocoderConfig) -> Self {
        let inner = match config.model_type {
            VocoderType::WaveNet => VocoderInner::WaveNet(WaveNetVocoder::new(&config)),
            VocoderType::MelGAN => VocoderInner::MelGAN(MelGANVocoder::new(&config)),
            VocoderType::HiFiGAN => VocoderInner::HiFiGAN(HiFiGANVocoder::new(&config)),
            _ => VocoderInner::HiFiGAN(HiFiGANVocoder::new(&config)),
        };
        Self { inner, config }
    }

    /// Synthesize waveform from mel spectrogram.
    ///
    /// `mel_spectrogram` shape: outer = mel bands, inner = time frames.
    /// Returns a 1-D waveform whose length is approximately
    /// `n_frames * hop_length`.
    pub fn synthesize(&self, mel_spectrogram: &[Vec<f64>]) -> SignalResult<Vec<f64>> {
        if mel_spectrogram.is_empty() {
            return Ok(Vec::new());
        }

        let n_mels = mel_spectrogram.len();
        if n_mels != self.config.n_mels {
            return Err(SignalError::DimensionMismatch(format!(
                "Expected {} mel bands, got {}",
                self.config.n_mels, n_mels
            )));
        }

        match &self.inner {
            VocoderInner::WaveNet(v) => v.synthesize(mel_spectrogram),
            VocoderInner::MelGAN(v) => v.synthesize(mel_spectrogram),
            VocoderInner::HiFiGAN(v) => v.synthesize(mel_spectrogram),
        }
    }

    /// Return the configuration.
    pub fn config(&self) -> &VocoderConfig {
        &self.config
    }

    /// Compute the WaveNet receptive field (only meaningful for WaveNet type).
    pub fn receptive_field(&self) -> usize {
        match &self.inner {
            VocoderInner::WaveNet(v) => v.receptive_field(
                self.config.wavenet_dilation_cycle,
                self.config.wavenet_n_cycles,
            ),
            _ => {
                // For MelGAN/HiFiGAN the concept is different; return upsample product
                self.config.upsample_factors.iter().product()
            }
        }
    }
}

// ── Mel spectrogram helper ──────────────────────────────────────────────────

/// Convert a frequency in Hz to the mel scale.
fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert a mel-scale value back to Hz.
fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

/// Compute mel spectrogram from a time-domain signal.
///
/// Returns `(n_mels, n_frames)` where each entry is in log-mel power.
pub fn mel_spectrogram(signal: &[f64], config: &MelConfig) -> Vec<Vec<f64>> {
    let n = signal.len();
    if n == 0 || config.n_fft == 0 || config.hop_length == 0 {
        return vec![vec![]; config.n_mels];
    }

    let n_fft = config.n_fft;
    let hop = config.hop_length;
    let n_mels = config.n_mels;
    let sr = config.sample_rate as f64;
    let f_max = if config.f_max <= 0.0 {
        sr / 2.0
    } else {
        config.f_max
    };
    let f_min = config.f_min;

    // Build mel filter bank
    let n_freqs = n_fft / 2 + 1;
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    let mel_points: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
        .collect();
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<f64> = hz_points.iter().map(|&f| f * n_fft as f64 / sr).collect();

    // Triangular filter bank
    let mut filters = vec![vec![0.0; n_freqs]; n_mels];
    for m in 0..n_mels {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];
        for k in 0..n_freqs {
            let kf = k as f64;
            if kf > left && kf <= center && center > left {
                filters[m][k] = (kf - left) / (center - left);
            } else if kf > center && kf < right && right > center {
                filters[m][k] = (right - kf) / (right - center);
            }
        }
    }

    // Hann window
    let window: Vec<f64> = (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / n_fft as f64).cos()))
        .collect();

    // STFT frames
    let n_frames = if n > n_fft {
        (n - n_fft) / hop + 1
    } else if n > 0 {
        1
    } else {
        0
    };

    let mut mel_spec = vec![vec![0.0; n_frames]; n_mels];

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop;
        // Compute power spectrum via DFT
        let mut power = vec![0.0; n_freqs];
        for k in 0..n_freqs {
            let mut re = 0.0;
            let mut im = 0.0;
            for t in 0..n_fft {
                let idx = start + t;
                let sample = if idx < n { signal[idx] } else { 0.0 };
                let windowed = sample * window[t];
                let angle = -2.0 * std::f64::consts::PI * k as f64 * t as f64 / n_fft as f64;
                re += windowed * angle.cos();
                im += windowed * angle.sin();
            }
            power[k] = re * re + im * im;
        }

        // Apply mel filters
        for m in 0..n_mels {
            let mut energy = 0.0;
            for k in 0..n_freqs {
                energy += filters[m][k] * power[k];
            }
            // Log-mel (floor to avoid log(0))
            mel_spec[m][frame_idx] = (energy.max(1e-10)).ln();
        }
    }

    mel_spec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocoder_output_length_hifigan() {
        let config = VocoderConfig {
            model_type: VocoderType::HiFiGAN,
            n_mels: 8,
            hop_length: 64,
            upsample_factors: vec![8, 4, 2],
            hifigan_kernel_sizes: vec![3, 7],
            wavenet_residual_channels: 16,
            seed: 123,
            ..Default::default()
        };
        let vocoder = NeuralVocoder::new(config);
        let n_frames = 10;
        let mel = vec![vec![0.0; n_frames]; 8];
        let wav = vocoder.synthesize(&mel).expect("synthesize should work");
        // Output length should be at least n_frames * hop_length (with some conv padding)
        assert!(
            wav.len() >= n_frames * 64,
            "Output too short: {} < {}",
            wav.len(),
            n_frames * 64
        );
    }

    #[test]
    fn test_vocoder_output_length_melgan() {
        let config = VocoderConfig {
            model_type: VocoderType::MelGAN,
            n_mels: 8,
            hop_length: 64,
            upsample_factors: vec![8, 4, 2],
            hifigan_kernel_sizes: vec![3, 7],
            wavenet_residual_channels: 16,
            seed: 123,
            ..Default::default()
        };
        let vocoder = NeuralVocoder::new(config);
        let n_frames = 10;
        let mel = vec![vec![0.0; n_frames]; 8];
        let wav = vocoder.synthesize(&mel).expect("synthesize should work");
        assert!(
            wav.len() >= n_frames * 64,
            "MelGAN output too short: {} < {}",
            wav.len(),
            n_frames * 64
        );
    }

    #[test]
    fn test_vocoder_wavenet_receptive_field() {
        let config = VocoderConfig {
            model_type: VocoderType::WaveNet,
            n_mels: 8,
            wavenet_dilation_cycle: 10,
            wavenet_n_cycles: 3,
            wavenet_residual_channels: 8,
            upsample_factors: vec![4, 4],
            seed: 7,
            ..Default::default()
        };
        let vocoder = NeuralVocoder::new(config);
        let rf = vocoder.receptive_field();
        // 3 cycles of dilations 1,2,4,...,512 → each cycle adds 2*(1+2+4+...+512) = 2*1023 = 2046
        // total = 1 + 3*2046 = 6139
        assert_eq!(rf, 6139, "WaveNet receptive field mismatch");
    }

    #[test]
    fn test_melgan_upsample_factor() {
        // Product of upsample factors determines the output length multiplier
        let factors = vec![8, 4, 2];
        let product: usize = factors.iter().product();
        assert_eq!(product, 64);
    }

    #[test]
    fn test_mel_spectrogram_basic() {
        let sr = 8000;
        let dur = 0.1;
        let n = (sr as f64 * dur) as usize;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / sr as f64).sin())
            .collect();
        let config = MelConfig {
            sample_rate: sr,
            n_fft: 256,
            hop_length: 64,
            n_mels: 16,
            ..Default::default()
        };
        let mel = mel_spectrogram(&signal, &config);
        assert_eq!(mel.len(), 16);
        assert!(!mel[0].is_empty(), "Should have at least one frame");
    }

    #[test]
    fn test_vocoder_empty_input() {
        let config = VocoderConfig::default();
        let vocoder = NeuralVocoder::new(config);
        let result = vocoder.synthesize(&[]);
        assert!(result.is_ok());
        assert!(result.expect("should be ok").is_empty());
    }

    #[test]
    fn test_vocoder_wrong_n_mels() {
        let config = VocoderConfig {
            n_mels: 80,
            ..Default::default()
        };
        let vocoder = NeuralVocoder::new(config);
        let mel = vec![vec![0.0; 5]; 40]; // 40 != 80
        let result = vocoder.synthesize(&mel);
        assert!(result.is_err());
    }
}
