//! Semantic segmentation module
//!
//! This module provides comprehensive semantic segmentation algorithms including:
//!
//! - **FCN-lite**: Fully Convolutional Network concepts with bilinear upsampling,
//!   segmentation metrics, and dense CRF post-processing.
//! - **DeepLab-lite**: Atrous (dilated) convolution and Atrous Spatial Pyramid Pooling
//!   (ASPP) for multi-scale context aggregation.
//! - **Legacy models**: `DeepLabV3Plus`, `UNet`, `FCN` (from the original semantic module).
//!
//! # Format conventions
//!
//! Arrays use **HWC** order (`Array3<f32>` with shape `[H, W, C]`) for single images
//! and **NHWC** order (`Array4<f32>` with shape `[N, H, W, C]`) for batches.
//! Class logits / probabilities are the last axis (C).

pub mod deeplab;
pub mod fcn;

// Re-export the original semantic segmentation structs / helpers.
mod legacy;

pub use deeplab::{
    aspp_forward, global_average_pooling, ASPPBranch, ASPPConfig, AtrousConv2D, AtrousConv2DConfig,
    ASPP,
};
pub use fcn::{
    bilinear_upsample_mask, compute_segmentation_metrics, dense_crf_post_process, FCNBackbone,
    FCNConfig, FCNOutput, SegmentationMetrics,
};

// Re-export legacy items so existing callers keep working.
pub use legacy::{
    create_cityscapes_classes, create_pascal_voc_classes, DeepLabV3Plus, FCNVariant,
    SegmentationArchitecture, SegmentationClass, SegmentationResult, UNet, FCN,
};
