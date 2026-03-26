//! Core types for SAM-style prompt-based segmentation.

use scirs2_core::ndarray::Array2;

// ---------------------------------------------------------------------------
// Prompt types
// ---------------------------------------------------------------------------

/// The kind of prompt that drives mask prediction.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum PromptType {
    /// A single point with foreground/background label.
    Point {
        /// Horizontal coordinate (column).
        x: usize,
        /// Vertical coordinate (row).
        y: usize,
        /// `true` = foreground, `false` = background.
        is_foreground: bool,
    },
    /// An axis-aligned bounding box.
    BoundingBox {
        /// Left column.
        x1: usize,
        /// Top row.
        y1: usize,
        /// Right column (exclusive).
        x2: usize,
        /// Bottom row (exclusive).
        y2: usize,
    },
    /// A dense mask prompt (e.g. a rough user scribble).
    MaskPrompt {
        /// 2-D mask whose spatial size matches the input image.
        mask: Array2<f64>,
    },
    /// Multiple points with per-point foreground/background labels.
    MultiPoint {
        /// Each entry is `(x, y, is_foreground)`.
        points: Vec<(usize, usize, bool)>,
    },
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the SAM-style segmentation pipeline.
#[derive(Debug, Clone)]
pub struct SAMConfig {
    /// Encoder input resolution (images are conceptually rescaled to this).
    pub image_size: usize,
    /// Embedding dimensionality throughout the pipeline.
    pub embed_dim: usize,
    /// Number of candidate masks produced by the decoder.
    pub num_mask_outputs: usize,
    /// Hidden size of the IoU prediction head.
    pub iou_head_hidden: usize,
    /// Number of encoder down-sampling stages (scales).
    pub encoder_stages: usize,
}

impl Default for SAMConfig {
    fn default() -> Self {
        Self {
            image_size: 1024,
            embed_dim: 256,
            num_mask_outputs: 3,
            iou_head_hidden: 256,
            encoder_stages: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// Output of the mask decoder.
#[derive(Debug, Clone)]
pub struct SegmentationResult {
    /// Predicted masks, one per candidate. Each mask has the same spatial size
    /// as the input image and contains logit values (higher = more likely
    /// foreground).
    pub masks: Vec<Array2<f64>>,
    /// Per-mask IoU predictions (model confidence).
    pub iou_predictions: Vec<f64>,
    /// Per-mask stability scores (IoU between high-threshold and
    /// low-threshold binarisations).
    pub stability_scores: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Prompt wrapper
// ---------------------------------------------------------------------------

/// A user-supplied segmentation prompt together with an optional label.
#[derive(Debug, Clone)]
pub struct SegmentationPrompt {
    /// The prompt geometry.
    pub prompt_type: PromptType,
    /// Optional human-readable label for the prompt.
    pub label: Option<String>,
}

impl SegmentationPrompt {
    /// Create a new prompt without a label.
    pub fn new(prompt_type: PromptType) -> Self {
        Self {
            prompt_type,
            label: None,
        }
    }

    /// Create a new prompt with a label.
    pub fn with_label(prompt_type: PromptType, label: impl Into<String>) -> Self {
        Self {
            prompt_type,
            label: Some(label.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_sam_config_default() {
        let cfg = SAMConfig::default();
        assert_eq!(cfg.image_size, 1024);
        assert_eq!(cfg.embed_dim, 256);
        assert_eq!(cfg.num_mask_outputs, 3);
        assert_eq!(cfg.iou_head_hidden, 256);
        assert_eq!(cfg.encoder_stages, 3);
    }

    #[test]
    fn test_prompt_type_point() {
        let p = PromptType::Point {
            x: 10,
            y: 20,
            is_foreground: true,
        };
        if let PromptType::Point {
            x,
            y,
            is_foreground,
        } = &p
        {
            assert_eq!(*x, 10);
            assert_eq!(*y, 20);
            assert!(*is_foreground);
        } else {
            panic!("expected Point variant");
        }
    }

    #[test]
    fn test_prompt_type_bounding_box() {
        let p = PromptType::BoundingBox {
            x1: 5,
            y1: 10,
            x2: 50,
            y2: 60,
        };
        if let PromptType::BoundingBox { x1, y1, x2, y2 } = &p {
            assert_eq!(*x1, 5);
            assert_eq!(*y1, 10);
            assert_eq!(*x2, 50);
            assert_eq!(*y2, 60);
        } else {
            panic!("expected BoundingBox variant");
        }
    }

    #[test]
    fn test_prompt_type_mask() {
        let mask = Array2::<f64>::zeros((64, 64));
        let p = PromptType::MaskPrompt { mask: mask.clone() };
        if let PromptType::MaskPrompt { mask: m } = &p {
            assert_eq!(m.dim(), (64, 64));
        } else {
            panic!("expected MaskPrompt variant");
        }
    }

    #[test]
    fn test_prompt_type_multipoint() {
        let pts = vec![(1, 2, true), (3, 4, false)];
        let p = PromptType::MultiPoint {
            points: pts.clone(),
        };
        if let PromptType::MultiPoint { points } = &p {
            assert_eq!(points.len(), 2);
        } else {
            panic!("expected MultiPoint variant");
        }
    }

    #[test]
    fn test_segmentation_prompt_new() {
        let sp = SegmentationPrompt::new(PromptType::Point {
            x: 0,
            y: 0,
            is_foreground: true,
        });
        assert!(sp.label.is_none());
    }

    #[test]
    fn test_segmentation_prompt_with_label() {
        let sp = SegmentationPrompt::with_label(
            PromptType::BoundingBox {
                x1: 0,
                y1: 0,
                x2: 10,
                y2: 10,
            },
            "cat",
        );
        assert_eq!(sp.label.as_deref(), Some("cat"));
    }
}
