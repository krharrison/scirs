//! Multi-object tracking algorithms.
//!
//! # Algorithms
//!
//! * **SORT** — Simple Online and Realtime Tracking (Bewley et al. 2016)
//! * **ByteTrack** — Two-stage association tracker (Zhang et al. 2022)
//!
//! Both trackers use Kalman filtering for state prediction and the Hungarian
//! algorithm for detection-to-track assignment.

pub mod bytetrack;
pub mod hungarian;
pub mod kalman_box;
pub mod sort;
pub mod types;

pub use bytetrack::ByteTracker;
pub use hungarian::hungarian_assign;
pub use kalman_box::KalmanBoxTracker;
pub use sort::SortTracker;
pub use types::{BoundingBox, ByteTrackConfig, SortConfig, Track, TrackState, TrackerResult};
