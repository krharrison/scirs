//! Feature descriptor algorithms.
//!
//! # Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`hog`] | Histogram of Oriented Gradients — dense appearance descriptor |
//! | [`brief`] | BRIEF — fast binary descriptor for pre-detected keypoints |
//! | [`orb`] | ORB — FAST corners + steered rBRIEF, multi-scale |
//! | [`sift_like`] | SIFT-like — DoG scale space + 128-D gradient histograms |

pub mod brief;
pub mod hog;
pub mod orb;
pub mod sift_like;

// Convenience re-exports
pub use brief::{
    compute_brief, generate_test_pairs, hamming_distance, BRIEFConfig, BRIEFDescriptor,
    BRIEFPattern,
};
pub use hog::{compute_hog, hog_feature_vector, HOGConfig, HOGDescriptor, HOGNorm};
pub use orb::{detect_orb, ORBConfig, ORBDescriptor, ORBKeypoint};
pub use sift_like::{detect_and_describe, SIFTDescriptor, SIFTKeypoint, SIFTLikeConfig};
