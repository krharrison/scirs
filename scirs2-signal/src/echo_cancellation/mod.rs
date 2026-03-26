//! Acoustic Echo Cancellation (AEC) for real-time communications.
//!
//! This module implements a complete AEC pipeline including:
//!
//! - **NLMS adaptive filter** with optional multi-delay partitioning for long
//!   echo paths.
//! - **Double-talk detection** (Geigel DTD, cross-correlation DTD, far-end
//!   activity detection) to freeze adaptation when near-end speech is present.
//! - **ERLE monitoring** (Echo Return Loss Enhancement) to track cancellation
//!   quality over time.
//!
//! # Architecture
//!
//! ```text
//! far-end ──┬──► [NLMS Adaptive Filter] ──► echo_estimate
//!           │                                    │
//!           │                                    ▼
//!           └──► [DTD Detector] ◄──── near_end ─ Σ ──► output (error)
//!                    │
//!                    ▼
//!              freeze/adapt decision
//! ```
//!
//! # Example
//!
//! ```
//! use scirs2_signal::echo_cancellation::{AECConfig, AcousticEchoCanceller};
//!
//! let cfg = AECConfig {
//!     filter_length: 128,
//!     step_size: 0.5,
//!     ..AECConfig::default()
//! };
//! let mut aec = AcousticEchoCanceller::new(cfg).expect("valid config");
//!
//! // Process sample-by-sample
//! let result = aec.process_sample(0.5, 0.3);
//! println!("Output: {}, ERLE: {:.1} dB", result.output, result.erle_db);
//! ```

pub mod canceller;
pub mod coherence_dtd;
pub mod double_talk;
pub mod multi_delay;
pub mod nlms;
pub mod types;

// Re-exports for convenient access.
pub use canceller::AcousticEchoCanceller;
pub use coherence_dtd::{DoubleTalkConfig, DoubleTalkDetector, EchoCanceller};
pub use double_talk::{
    classify_talk_status, CrossCorrelationDetector, FarEndActivityDetector, GeigelDetector,
};
pub use multi_delay::{
    kaiser_window_fir, per_band_coherence, qmf_analysis, qmf_synthesis, MultiDelayAec,
    MultiDelayAecConfig, SubbandFilter,
};
pub use nlms::{NlmsConfig, NlmsFilter, RlsConfig, RlsFilter};
pub use types::{AECConfig, AECResult, AECState, DoubleTalkStatus};
