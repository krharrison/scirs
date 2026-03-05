//! Stereo vision and depth estimation.
//!
//! Provides a full stereo pipeline:
//!
//! 1. **Rectification** ([`rectification`]): warp a raw stereo pair into a
//!    canonical parallel configuration with horizontal epipolar lines.
//! 2. **Disparity estimation** ([`disparity`]): compute per-pixel horizontal
//!    displacement between the rectified pair using Block Matching or
//!    Semi-Global Matching.
//! 3. **3D Reconstruction** ([`reconstruction`]): back-project the disparity
//!    map to a metric [`PointCloud`].
//! 4. **Calibration** ([`calibration`]): store camera intrinsics, the
//!    stereo extrinsic transform, and derive the Q matrix for reprojection.
//!
//! ## Quick-Start
//!
//! ```
//! use scirs2_vision::stereo::{
//!     StereoCalibration, StereoRectifier, BlockMatching, DisparityMap,
//! };
//!
//! // 1. Define calibration (ideal parallel rig).
//! let cal = StereoCalibration::from_baseline(500.0, 500.0, 32.0, 24.0, 0.10);
//!
//! // 2. Compute rectification maps.
//! let mut rectifier = StereoRectifier::new(cal.clone());
//! rectifier.compute_maps(64, 48);
//!
//! // 3. Match with Block Matching.
//! let bm = BlockMatching::new(5, 0, 16);
//! let left  = vec![128u8; 64 * 48];
//! let right = vec![128u8; 64 * 48];
//! let disp = bm.compute(&left, &right, 64, 48).unwrap();
//!
//! // 4. Reproject to 3D.
//! let pc = cal.reproject_to_3d(&disp);
//! println!("{} 3-D points", pc.len());
//! ```

pub mod calibration;
pub mod disparity;
pub mod reconstruction;
pub mod rectification;

pub use calibration::{CameraIntrinsics, StereoCalibration};
pub use disparity::{BlockMatching, DisparityMap, SemiGlobalMatching};
pub use reconstruction::PointCloud;
pub use rectification::StereoRectifier;
