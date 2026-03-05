//! Video and temporal processing algorithms.
//!
//! This module provides a comprehensive suite of algorithms for analysing and
//! processing video frame sequences, including background subtraction, motion
//! estimation, object tracking, and temporal filtering.
//!
//! # Sub-modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | `background` | Background subtraction (running average, GMM/MOG2, median) |
//! | `motion` | Motion estimation (block matching, phase correlation) |
//! | `tracking` | Object tracking (Mean Shift, CamShift, Kalman, multi-object) |
//! | `temporal` | Temporal filtering, stabilisation, frame interpolation |
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use scirs2_vision::video::background::{RunningAverageBackground, BackgroundConfig};
//! use scirs2_core::ndarray::Array2;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut bg = RunningAverageBackground::new(BackgroundConfig::default())?;
//!     // Process frames in a loop...
//!     let frame: Array2<f64> = Array2::zeros((480, 640));
//!     let mask = bg.apply(&frame)?;
//!     Ok(())
//! }
//! ```

/// Background subtraction algorithms.
pub mod background;

/// Motion estimation algorithms.
pub mod motion;

/// Object tracking algorithms.
pub mod tracking;

/// Temporal filtering and video stabilisation.
pub mod temporal;

// Re-export key types for convenience.
pub use background::{
    adaptive_learning_rate, mask_to_binary, BackgroundConfig, ForegroundLabel, GmmBackground,
    MedianBackground, RunningAverageBackground, ShadowParams,
};

pub use motion::{
    block_match_full, block_match_tss, motion_compensate, phase_correlation, prediction_error,
    MotionField, MotionVector,
};

pub use tracking::{
    BBox, CamShiftTracker, KalmanModel, KalmanTracker, MeanShiftConfig, MeanShiftTracker,
    MultiObjectTracker, MultiTrackerConfig, TrackStatus, TrackedObject,
};

pub use temporal::{
    apply_translation, double_frame_difference, frame_difference, interpolate_linear,
    interpolate_motion_compensated, smooth_trajectory, stabilisation_corrections,
    threshold_difference, TemporalGaussianFilter, TemporalMedianFilter,
};
