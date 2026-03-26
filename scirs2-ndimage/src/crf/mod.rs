//! Conditional Random Field (CRF) post-processing module
//!
//! Provides a fully-connected dense CRF for label refinement after semantic
//! segmentation, following Krähenbühl & Koltun (2011).

pub mod dense_crf;

pub use dense_crf::{apply_to_segmentation_2d, CrfConfig, DenseCrf};
