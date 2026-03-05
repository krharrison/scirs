//! SIFT-like and ORB-like feature detection, binary/float matching, and
//! robust homography estimation.
//!
//! # Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`sift_like`] | Scale-invariant keypoints with 128-D float descriptors |
//! | [`orb_like`] | FAST corners + rBRIEF 256-bit binary descriptors |
//! | [`matching`] | Brute-force, FLANN-like, and ratio-test matching |
//! | [`homography_ransac`] | DLT homography + RANSAC + LM refinement |
//!
//! # Quick start
//!
//! ```rust,no_run
//! use scirs2_vision::features::{
//!     sift_like::{detect_and_describe, SIFTConfig},
//!     orb_like::{detect_and_describe_orb, OrbLikeConfig},
//!     matching::{match_descriptors, match_binary_descriptors, MatchMethod},
//!     homography_ransac::{estimate_homography_ransac, RansacConfig},
//! };
//! use scirs2_core::ndarray::Array2;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a synthetic test image (256×256)
//! let image = Array2::<f64>::from_shape_fn((256, 256), |(r, c)| {
//!     let d = ((r as f64 - 128.0).powi(2) + (c as f64 - 128.0).powi(2)).sqrt();
//!     (-d / 40.0).exp()
//! });
//!
//! // SIFT-like detection and description
//! let config = SIFTConfig {
//!     num_octaves: 3,
//!     max_features: 100,
//!     ..Default::default()
//! };
//! let descs = detect_and_describe(&image, &config)?;
//! println!("Detected {} SIFT-like features", descs.len());
//!
//! // ORB-like detection and description
//! let orb_cfg = OrbLikeConfig { max_features: 100, ..Default::default() };
//! let orb_descs = detect_and_describe_orb(&image, &orb_cfg)?;
//! println!("Detected {} ORB-like features", orb_descs.len());
//!
//! // Match SIFT features against themselves
//! let matches = match_descriptors(&descs, &descs, &MatchMethod::RatioTest { ratio: 0.8 })?;
//!
//! // Match binary ORB features
//! let bin_matches = match_binary_descriptors(&orb_descs, &orb_descs, &MatchMethod::BruteForce)?;
//!
//! // Robust homography from synthetic correspondences
//! let src_pts: Vec<(f64, f64)> = (0..20).map(|i| (i as f64 * 10.0, i as f64 * 5.0)).collect();
//! let dst_pts: Vec<(f64, f64)> = src_pts.iter().map(|&(x, y)| (x + 5.0, y + 3.0)).collect();
//! let rans_cfg = RansacConfig::default();
//! let result = estimate_homography_ransac(&src_pts, &dst_pts, &rans_cfg)?;
//! println!("Homography estimated with {} inliers", result.num_inliers);
//! # Ok(())
//! # }
//! ```

pub mod homography_ransac;
pub mod matching;
pub mod orb_like;
pub mod sift_like;

// Convenience re-exports
pub use homography_ransac::{
    dlt_homography, estimate_homography_ransac, Homography, RansacConfig, RansacHomographyResult,
};
pub use matching::{match_binary_descriptors, match_descriptors, symmetric_filter, MatchMethod};
pub use orb_like::{
    detect_and_describe_orb, hamming_distance, OrbKeypoint, OrbLikeConfig, OrbLikeDescriptor,
    DESC_BITS,
};
pub use sift_like::{detect_and_describe, Keypoint, SIFTConfig, SIFTDescriptor};
