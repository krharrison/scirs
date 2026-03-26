//! SAM-style prompt-based segmentation.
//!
//! Implements a Segment-Anything-Model inspired pipeline: an image encoder
//! produces multi-scale features, a prompt encoder converts user prompts
//! (points, boxes, masks) into embeddings, and a mask decoder fuses both
//! to produce high-quality segmentation masks with IoU predictions.

mod image_encoder;
mod mask_decoder;
mod prompt_encoder;
mod types;

pub use image_encoder::{PatchEmbedding, SimpleImageEncoder};
pub use mask_decoder::MaskDecoder;
pub use prompt_encoder::{PromptEmbedding, PromptEncoder};
pub use types::{PromptType, SAMConfig, SegmentationPrompt, SegmentationResult};
