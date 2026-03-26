//! Neural audio processing: Conv-TasNet speech enhancement, source separation,
//! inference API, weight store, Mel spectrogram, neural vocoders, speech
//! enhancement, and audio quality metrics.
//!
//! Implements the Conv-TasNet architecture (Luo & Mesgarani 2019) for time-domain
//! audio source separation and speech enhancement, plus a lightweight inference
//! API with [`WeightStore`] serialization and [`compute_mel_spectrogram`].
//!
//! ## Vocoders
//! - [`vocoder::NeuralVocoder`] — WaveNet, MelGAN, HiFi-GAN style mel-to-waveform
//!
//! ## Enhancement
//! - [`enhancement::enhance`] — mask-based, DCCRN, spectral mapping
//!
//! ## Metrics
//! - [`metrics::si_sdr`], [`metrics::stoi`], [`metrics::pesq_estimate`],
//!   [`metrics::snr_improvement`], [`metrics::spectral_convergence`]

pub mod conv_tasnet;
pub mod enhancement;
pub mod inference;
pub mod metrics;
pub mod vocoder;

pub use conv_tasnet::{si_snr_loss, ConvTasNet, ConvTasNetConfig, TcnBlock};
pub use enhancement::{enhance, EnhancementConfig, EnhancementMethod, SpeechEnhancement};
pub use inference::{
    compute_mel_spectrogram, AudioModel, ConvTasNetInference, ModelFormat, NeuralAudioConfig,
    WeightStore,
};
pub use metrics::{pesq_estimate, si_sdr, snr_improvement, spectral_convergence, stoi};
pub use vocoder::{mel_spectrogram, MelConfig, NeuralVocoder, VocoderConfig, VocoderType};
