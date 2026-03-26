//! Streaming FFT processor for real-time audio and sensor data.
//!
//! Processes a continuous stream of samples through a ring-buffer and emits
//! [`SpectralFrame`]s whenever `hop_size` new samples have arrived.

pub mod processor;

pub use processor::{SpectralFrame, StreamingFftConfig, StreamingFftProcessor, WindowFn};
