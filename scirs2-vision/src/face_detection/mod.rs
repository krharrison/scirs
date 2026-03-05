//! Face detection algorithms.
//!
//! Provides the Viola-Jones cascade classifier for face detection in
//! grayscale images using integral images, Haar-like features, and
//! an AdaBoost cascade.
//!
//! # Modules
//!
//! - [`viola_jones`] – Pure-Rust Viola-Jones face detector.

pub mod viola_jones;

pub use viola_jones::{
    cascade_classify, compute_haar_feature, detect_multiscale, AdaBoostClassifier, AdaBoostStage,
    FaceDetection, HaarFeature, HaarFeatureType, HaarRect, IntegralImage, WeakClassifier,
};
