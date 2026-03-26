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

pub mod block_filter;
pub mod envelope;
pub mod filters;
pub mod online_stft;
pub mod overlap_save;
pub mod ring_buffer;
pub mod spectral_analysis;
pub mod stft;
pub mod streaming_omp;
pub mod ws78_block_filter;
pub mod ws78_online_stft;
pub mod ws78_streaming_omp;

// StreamProcessor trait for block-based streaming filters
use crate::error::SignalResult;

/// Trait for block-based streaming signal processors.
pub trait StreamProcessor {
    /// Process one block of input samples, returning the output.
    fn process_block(&mut self, input: &[f64]) -> SignalResult<Vec<f64>>;
    /// Reset internal state.
    fn reset(&mut self);
}

// Re-export primary types for ergonomic access via `streaming::*`
pub use envelope::{HilbertEnvelope, PeakEnvelopeFollower, RmsEnvelope};
pub use filters::{StreamingFIR, StreamingIIR, StreamingMedianFilter, StreamingMovingAverage};
pub use ring_buffer::RingBuffer;
pub use spectral_analysis::{PitchDetector, SpectralFlux, StreamingPowerSpectrum};
pub use stft::{StreamingISTFT, StreamingSTFT};
