//! Real-time / streaming signal processing module.
//!
//! This module provides composable, sample-by-sample and chunk-based streaming
//! processors for signal processing pipelines.  Every component maintains
//! internal state so that data can be fed incrementally without having to
//! store the entire signal in memory.
//!
//! ## Sub-modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`ring_buffer`] | Generic circular buffer used for overlap management |
//! | [`filters`] | FIR, IIR (biquad), median, and moving-average filters |
//! | [`stft`] | Streaming STFT with overlap-add ISTFT reconstruction |
//! | [`spectral_analysis`] | Spectral flux, pitch detection (YIN), running PSD |
//! | [`envelope`] | Hilbert, RMS, and peak envelope followers |

pub mod envelope;
pub mod filters;
pub mod ring_buffer;
pub mod spectral_analysis;
pub mod stft;

// Re-export primary types for ergonomic access via `streaming::*`
pub use envelope::{HilbertEnvelope, PeakEnvelopeFollower, RmsEnvelope};
pub use filters::{StreamingFIR, StreamingIIR, StreamingMedianFilter, StreamingMovingAverage};
pub use ring_buffer::RingBuffer;
pub use spectral_analysis::{PitchDetector, SpectralFlux, StreamingPowerSpectrum};
pub use stft::{StreamingISTFT, StreamingSTFT};
