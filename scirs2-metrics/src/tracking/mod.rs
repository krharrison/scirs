//! Multi-object tracking metrics: MOTA, MOTP, IDF1.
//!
//! Implements the CLEAR MOT metrics (Bernardin & Stiefelhagen 2008) for
//! evaluating multi-object tracking systems.

pub mod mot_eval;
pub mod types;

pub use mot_eval::MotEvaluator;
pub use types::{GtTrackStats, MatchAlg, TrackingBox, TrackingMetrics, TrackingMetricsConfig};
