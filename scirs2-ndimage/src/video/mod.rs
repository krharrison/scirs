//! Video processing and motion analysis
//!
//! This module provides:
//! - **Background subtraction**: frame-differencing, running-average, GMM (Stauffer-Grimson),
//!   ViBe-style background modelling.
//! - **Motion estimation**: block-matching (EBMA), three-step search, diamond search, temporal
//!   difference, Jain's accumulated-difference image, motion-compensated frame reconstruction.
//! - **Video stabilization**: affine global motion estimation, Kalman-based trajectory smoothing,
//!   per-frame warp application, stable-region crop.

pub mod background_subtraction;
pub mod motion_estimation;
pub mod stabilization;

// ── convenience re-exports ────────────────────────────────────────────────────

pub use background_subtraction::{
    FrameBuffer,
    GaussianMixtureBackground,
    SimpleBackgroundModel,
    subtract_background,
    update_background,
    vibe_model,
};

pub use motion_estimation::{
    BlockMatchResult,
    MotionVector,
    accumulate_difference,
    block_matching,
    diamond_search,
    motion_compensated_frame,
    temporal_difference,
    three_step_search,
    BlockMatchConfig,
};

pub use stabilization::{
    AffineMotion,
    StabilizationConfig,
    VideoStabilizer,
    crop_stable_region,
    estimate_global_motion,
    smooth_motion_trajectory,
    stabilize_frame,
};
