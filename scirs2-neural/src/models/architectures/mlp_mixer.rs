//! MLP-Mixer Architecture Implementation
//!
//! This module implements the MLP-Mixer architecture as described in:
//! "MLP-Mixer: An all-MLP Architecture for Vision" (Tolstikhin et al., 2021)
//!
//! MLP-Mixer is an architecture based purely on multi-layer perceptrons (MLPs),
//! that contains two types of mixing layers:
//! - Token-mixing MLPs: Allow communication between different spatial locations
//! - Channel-mixing MLPs: Allow communication between different channels/features
//!
//! # Architecture Overview
//!
//! 1. **Patch Embedding**: Image is split into patches, each linearly projected
//! 2. **Mixer Layers**: Alternating token-mixing and channel-mixing MLPs
//! 3. **Classification Head**: Global average pooling followed by linear classifier
//!
//! # Examples
//!
//! ```rust
//! use scirs2_neural::models::architectures::{MLPMixer, MLPMixerConfig};
//! use scirs2_core::random::SeedableRng;
//!
//! let config = MLPMixerConfig {
//!     image_size: 224,
//!     patch_size: 16,
//!     num_classes: 1000,
//!     hidden_dim: 512,
//!     num_blocks: 8,
//!     token_mlp_dim: 256,
//!     channel_mlp_dim: 2048,
//!     dropout_rate: 0.0,
//!     in_channels: 3,
//! };
//!
//! let mut rng = scirs2_core::random::rngs::SmallRng::seed_from_u64(42);
//! let mixer = MLPMixer::<f32>::new(config, &mut rng).expect("Operation failed");
//! ```

use crate::error::{NeuralError, Result};
use crate::layers::{Dense, Dropout, Layer, LayerNorm};
use scirs2_core::ndarray::{s, Array, Array2, Array3, Axis, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign};
use scirs2_core::random::{Rng, RngExt};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Configuration for the MLP-Mixer model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLPMixerConfig {
    /// Input image size (assumes square images)
    pub image_size: usize,
    /// Size of each patch
    pub patch_size: usize,
    /// Number of output classes
    pub num_classes: usize,
    /// Hidden dimension (channel dimension after patch embedding)
    pub hidden_dim: usize,
    /// Number of Mixer blocks
    pub num_blocks: usize,
    /// Dimension of the token-mixing MLP
    pub token_mlp_dim: usize,
    /// Dimension of the channel-mixing MLP
    pub channel_mlp_dim: usize,
    /// Dropout rate
    pub dropout_rate: f64,
    /// Number of input channels (3 for RGB images)
    pub in_channels: usize,
}

impl Default for MLPMixerConfig {
    fn default() -> Self {
        Self {
            image_size: 224,
            patch_size: 16,
            num_classes: 1000,
            hidden_dim: 512,
            num_blocks: 8,
            token_mlp_dim: 256,
            channel_mlp_dim: 2048,
            dropout_rate: 0.0,
            in_channels: 3,
        }
    }
}

impl MLPMixerConfig {
    /// Create a Mixer-S/32 configuration
    pub fn mixer_s_32(num_classes: usize) -> Self {
        Self {
            image_size: 224,
            patch_size: 32,
            num_classes,
            hidden_dim: 512,
            num_blocks: 8,
            token_mlp_dim: 256,
            channel_mlp_dim: 2048,
            dropout_rate: 0.0,
            in_channels: 3,
        }
    }

    /// Create a Mixer-S/16 configuration
    pub fn mixer_s_16(num_classes: usize) -> Self {
        Self {
            image_size: 224,
            patch_size: 16,
            num_classes,
            hidden_dim: 512,
            num_blocks: 8,
            token_mlp_dim: 256,
            channel_mlp_dim: 2048,
            dropout_rate: 0.0,
            in_channels: 3,
        }
    }

    /// Create a Mixer-B/32 configuration
    pub fn mixer_b_32(num_classes: usize) -> Self {
        Self {
            image_size: 224,
            patch_size: 32,
            num_classes,
            hidden_dim: 768,
            num_blocks: 12,
            token_mlp_dim: 384,
            channel_mlp_dim: 3072,
            dropout_rate: 0.0,
            in_channels: 3,
        }
    }

    /// Create a Mixer-B/16 configuration
    pub fn mixer_b_16(num_classes: usize) -> Self {
        Self {
            image_size: 224,
            patch_size: 16,
            num_classes,
            hidden_dim: 768,
            num_blocks: 12,
            token_mlp_dim: 384,
            channel_mlp_dim: 3072,
            dropout_rate: 0.0,
            in_channels: 3,
        }
    }

    /// Get the number of patches
    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }
}

/// A simple MLP block with GELU activation
///
/// This is the building block for both token-mixing and channel-mixing operations.
/// Structure: Linear -> GELU -> Dropout -> Linear -> Dropout
#[derive(Debug, Clone)]
pub struct MixerMLP<
    F: Float
        + Debug
        + ScalarOperand
        + Send
        + Sync
        + NumAssign
        + scirs2_core::simd_ops::SimdUnifiedOps
        + 'static,
> {
    /// First linear layer
    fc1: Dense<F>,
    /// Second linear layer
    fc2: Dense<F>,
    /// Dropout layer
    dropout: Dropout<F>,
}

impl<
        F: Float
            + Debug
            + ScalarOperand
            + Send
            + Sync
            + NumAssign
            + scirs2_core::simd_ops::SimdUnifiedOps
            + 'static,
    > MixerMLP<F>
{
    /// Create a new MixerMLP
    ///
    /// # Arguments
    /// * `in_features` - Input dimension
    /// * `hidden_features` - Hidden dimension
    /// * `out_features` - Output dimension
    /// * `dropout_rate` - Dropout probability
    /// * `rng` - Random number generator
    pub fn new<R: Rng + Clone + Send + Sync + 'static>(
        in_features: usize,
        hidden_features: usize,
        out_features: usize,
        dropout_rate: f64,
        rng: &mut R,
    ) -> Result<Self> {
        let fc1 = Dense::new(in_features, hidden_features, Some("gelu"), rng)?;
        let fc2 = Dense::new(hidden_features, out_features, None, rng)?;
        let dropout = Dropout::new(dropout_rate, rng)?;

        Ok(Self { fc1, fc2, dropout })
    }

    /// Forward pass through the MLP
    pub fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let x = self.fc1.forward(input)?;
        let x = self.dropout.forward(&x)?;
        let x = self.fc2.forward(&x)?;
        self.dropout.forward(&x)
    }
}

/// A single Mixer block containing token-mixing and channel-mixing
///
/// Each block consists of:
/// 1. Layer normalization
/// 2. Token-mixing MLP (across spatial dimension)
/// 3. Skip connection
/// 4. Layer normalization
/// 5. Channel-mixing MLP (across channel dimension)
/// 6. Skip connection
#[derive(Debug, Clone)]
pub struct MixerBlock<
    F: Float
        + Debug
        + ScalarOperand
        + Send
        + Sync
        + NumAssign
        + scirs2_core::simd_ops::SimdUnifiedOps
        + 'static,
> {
    /// Layer norm before token-mixing
    norm1: LayerNorm<F>,
    /// Token-mixing MLP
    token_mixing: MixerMLP<F>,
    /// Layer norm before channel-mixing
    norm2: LayerNorm<F>,
    /// Channel-mixing MLP
    channel_mixing: MixerMLP<F>,
    /// Number of patches (tokens)
    num_patches: usize,
    /// Hidden dimension
    hidden_dim: usize,
}

impl<
        F: Float
            + Debug
            + ScalarOperand
            + Send
            + Sync
            + NumAssign
            + scirs2_core::simd_ops::SimdUnifiedOps
            + 'static,
    > MixerBlock<F>
{
    /// Create a new MixerBlock
    ///
    /// # Arguments
    /// * `num_patches` - Number of patches (spatial tokens)
    /// * `hidden_dim` - Hidden/channel dimension
    /// * `token_mlp_dim` - Token-mixing MLP hidden dimension
    /// * `channel_mlp_dim` - Channel-mixing MLP hidden dimension
    /// * `dropout_rate` - Dropout probability
    /// * `rng` - Random number generator
    pub fn new<R: Rng + Clone + Send + Sync + 'static>(
        num_patches: usize,
        hidden_dim: usize,
        token_mlp_dim: usize,
        channel_mlp_dim: usize,
        dropout_rate: f64,
        rng: &mut R,
    ) -> Result<Self> {
        let norm1 = LayerNorm::new(hidden_dim, 1e-6, rng)?;
        let token_mixing =
            MixerMLP::new(num_patches, token_mlp_dim, num_patches, dropout_rate, rng)?;
        let norm2 = LayerNorm::new(hidden_dim, 1e-6, rng)?;
        let channel_mixing =
            MixerMLP::new(hidden_dim, channel_mlp_dim, hidden_dim, dropout_rate, rng)?;

        Ok(Self {
            norm1,
            token_mixing,
            norm2,
            channel_mixing,
            num_patches,
            hidden_dim,
        })
    }

    /// Forward pass through the Mixer block
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch_size, num_patches, hidden_dim]
    pub fn forward(&self, input: &Array3<F>) -> Result<Array3<F>> {
        let batch_size = input.shape()[0];

        // Token-mixing: transpose, apply MLP, transpose back
        // Input: [B, S, C] -> [B, C, S] -> MLP -> [B, C, S] -> [B, S, C]

        // First, apply layer norm along the last axis
        let normed1 = self.apply_layer_norm(&self.norm1, input)?;

        // Transpose for token-mixing: [B, S, C] -> [B, C, S]
        let transposed = normed1.permuted_axes([0, 2, 1]);

        // Apply token-mixing MLP
        let mut token_mixed = Array3::zeros(transposed.raw_dim());
        for b in 0..batch_size {
            let slice = transposed.slice(s![b, .., ..]).to_owned().into_dyn();
            let mixed = self.token_mixing.forward(&slice)?;
            let mixed_2d = mixed
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .map_err(|e| {
                    NeuralError::InferenceError(format!("Failed to convert mixed to 2D: {}", e))
                })?;
            token_mixed.slice_mut(s![b, .., ..]).assign(&mixed_2d);
        }

        // Transpose back: [B, C, S] -> [B, S, C]
        let token_mixed = token_mixed.permuted_axes([0, 2, 1]);

        // Skip connection
        let x = input + &token_mixed;

        // Channel-mixing
        let normed2 = self.apply_layer_norm(&self.norm2, &x)?;

        // Apply channel-mixing MLP (operates on last dimension)
        let mut channel_mixed = Array3::zeros(normed2.raw_dim());
        for b in 0..batch_size {
            let slice = normed2.slice(s![b, .., ..]).to_owned().into_dyn();
            let mixed = self.channel_mixing.forward(&slice)?;
            let mixed_2d = mixed
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .map_err(|e| {
                    NeuralError::InferenceError(format!("Failed to convert mixed to 2D: {}", e))
                })?;
            channel_mixed.slice_mut(s![b, .., ..]).assign(&mixed_2d);
        }

        // Skip connection
        Ok(&x + &channel_mixed)
    }

    /// Apply layer norm to a 3D tensor
    fn apply_layer_norm(&self, norm: &LayerNorm<F>, input: &Array3<F>) -> Result<Array3<F>> {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];
        let hidden_dim = input.shape()[2];

        let mut output = Array3::zeros(input.raw_dim());

        for b in 0..batch_size {
            for s in 0..seq_len {
                let slice = input.slice(s![b, s, ..]).to_owned().into_dyn();
                let normed = norm.forward(&slice)?;
                let normed_1d = normed
                    .into_dimensionality::<scirs2_core::ndarray::Ix1>()
                    .map_err(|e| {
                        NeuralError::InferenceError(format!(
                            "Failed to convert normed to 1D: {}",
                            e
                        ))
                    })?;
                output.slice_mut(s![b, s, ..]).assign(&normed_1d);
            }
        }

        Ok(output)
    }
}

/// MLP-Mixer model for image classification
///
/// The model consists of:
/// 1. Patch embedding layer
/// 2. Multiple Mixer blocks
/// 3. Classification head
#[derive(Debug)]
pub struct MLPMixer<
    F: Float
        + Debug
        + ScalarOperand
        + Send
        + Sync
        + NumAssign
        + scirs2_core::simd_ops::SimdUnifiedOps
        + 'static,
> {
    /// Model configuration
    config: MLPMixerConfig,
    /// Patch embedding projection
    patch_embed: Dense<F>,
    /// Mixer blocks
    blocks: Vec<MixerBlock<F>>,
    /// Final layer norm
    norm: LayerNorm<F>,
    /// Classification head
    head: Dense<F>,
}

impl<
        F: Float
            + Debug
            + ScalarOperand
            + Send
            + Sync
            + NumAssign
            + FromPrimitive
            + scirs2_core::simd_ops::SimdUnifiedOps
            + 'static,
    > MLPMixer<F>
{
    /// Create a new MLPMixer model
    ///
    /// # Arguments
    /// * `config` - Model configuration
    /// * `rng` - Random number generator
    pub fn new<R: Rng + Clone + Send + Sync + 'static>(
        config: MLPMixerConfig,
        rng: &mut R,
    ) -> Result<Self> {
        let num_patches = config.num_patches();
        let patch_dim = config.in_channels * config.patch_size * config.patch_size;

        // Patch embedding: flatten patch -> hidden_dim
        let patch_embed = Dense::new(patch_dim, config.hidden_dim, None, rng)?;

        // Create Mixer blocks
        let mut blocks = Vec::with_capacity(config.num_blocks);
        for _ in 0..config.num_blocks {
            blocks.push(MixerBlock::new(
                num_patches,
                config.hidden_dim,
                config.token_mlp_dim,
                config.channel_mlp_dim,
                config.dropout_rate,
                rng,
            )?);
        }

        // Final layer norm
        let norm = LayerNorm::new(config.hidden_dim, 1e-6, rng)?;

        // Classification head
        let head = Dense::new(config.hidden_dim, config.num_classes, None, rng)?;

        Ok(Self {
            config,
            patch_embed,
            blocks,
            norm,
            head,
        })
    }

    /// Extract patches from an image batch
    ///
    /// # Arguments
    /// * `images` - Image batch of shape [B, C, H, W]
    ///
    /// # Returns
    /// Patches of shape [B, num_patches, patch_dim]
    fn extract_patches(&self, images: &Array<F, IxDyn>) -> Result<Array3<F>> {
        let shape = images.shape();
        if shape.len() != 4 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Expected 4D input [B, C, H, W], got {:?}",
                shape
            )));
        }

        let batch_size = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];

        let patch_size = self.config.patch_size;
        let patches_h = height / patch_size;
        let patches_w = width / patch_size;
        let num_patches = patches_h * patches_w;
        let patch_dim = channels * patch_size * patch_size;

        let mut patches = Array3::zeros((batch_size, num_patches, patch_dim));

        for b in 0..batch_size {
            for ph in 0..patches_h {
                for pw in 0..patches_w {
                    let patch_idx = ph * patches_w + pw;
                    let h_start = ph * patch_size;
                    let w_start = pw * patch_size;

                    // Extract and flatten the patch
                    let mut flat_idx = 0;
                    for c in 0..channels {
                        for h in 0..patch_size {
                            for w in 0..patch_size {
                                patches[[b, patch_idx, flat_idx]] =
                                    images[[b, c, h_start + h, w_start + w]];
                                flat_idx += 1;
                            }
                        }
                    }
                }
            }
        }

        Ok(patches)
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    /// * `images` - Image batch of shape [B, C, H, W]
    ///
    /// # Returns
    /// Logits of shape [B, num_classes]
    pub fn forward(&self, images: &Array<F, IxDyn>) -> Result<Array2<F>> {
        let batch_size = images.shape()[0];

        // Extract patches: [B, num_patches, patch_dim]
        let patches = self.extract_patches(images)?;

        // Patch embedding: [B, num_patches, hidden_dim]
        let mut embedded = Array3::zeros((
            batch_size,
            self.config.num_patches(),
            self.config.hidden_dim,
        ));
        for b in 0..batch_size {
            let patch_slice = patches.slice(s![b, .., ..]).to_owned().into_dyn();
            let emb = self.patch_embed.forward(&patch_slice)?;
            let emb_2d = emb
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .map_err(|e| {
                    NeuralError::InferenceError(format!("Failed to convert embedding to 2D: {}", e))
                })?;
            embedded.slice_mut(s![b, .., ..]).assign(&emb_2d);
        }

        // Apply Mixer blocks
        let mut x = embedded;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // Global average pooling over spatial dimension
        // [B, num_patches, hidden_dim] -> [B, hidden_dim]
        let pooled = x.mean_axis(Axis(1)).ok_or_else(|| {
            NeuralError::InferenceError("Failed to compute mean across patches".to_string())
        })?;

        // Apply final layer norm
        let mut normed = Array2::zeros(pooled.raw_dim());
        for b in 0..batch_size {
            let slice = pooled.slice(s![b, ..]).to_owned().into_dyn();
            let n = self.norm.forward(&slice)?;
            let n_1d = n
                .into_dimensionality::<scirs2_core::ndarray::Ix1>()
                .map_err(|e| {
                    NeuralError::InferenceError(format!("Failed to convert normed to 1D: {}", e))
                })?;
            normed.slice_mut(s![b, ..]).assign(&n_1d);
        }

        // Classification head
        let mut output = Array2::zeros((batch_size, self.config.num_classes));
        for b in 0..batch_size {
            let slice = normed.slice(s![b, ..]).to_owned().into_dyn();
            let logits = self.head.forward(&slice)?;
            // Dense layer may return [1, num_classes] for 1D input, so handle both shapes
            if logits.ndim() == 2 && logits.shape()[0] == 1 {
                let logits_1d = logits
                    .into_shape_with_order(scirs2_core::ndarray::IxDyn(&[self.config.num_classes]))
                    .map_err(|e| {
                        NeuralError::InferenceError(format!(
                            "Failed to reshape logits to 1D: {}",
                            e
                        ))
                    })?
                    .into_dimensionality::<scirs2_core::ndarray::Ix1>()
                    .map_err(|e| {
                        NeuralError::InferenceError(format!(
                            "Failed to convert logits to 1D: {}",
                            e
                        ))
                    })?;
                output.slice_mut(s![b, ..]).assign(&logits_1d);
            } else {
                let logits_1d = logits
                    .into_dimensionality::<scirs2_core::ndarray::Ix1>()
                    .map_err(|e| {
                        NeuralError::InferenceError(format!(
                            "Failed to convert logits to 1D: {}",
                            e
                        ))
                    })?;
                output.slice_mut(s![b, ..]).assign(&logits_1d);
            }
        }

        Ok(output)
    }

    /// Get the configuration
    pub fn config(&self) -> &MLPMixerConfig {
        &self.config
    }

    /// Get the number of parameters (approximate)
    pub fn num_parameters(&self) -> usize {
        let num_patches = self.config.num_patches();
        let patch_dim = self.config.in_channels * self.config.patch_size * self.config.patch_size;
        let hidden_dim = self.config.hidden_dim;

        // Patch embedding
        let patch_embed_params = patch_dim * hidden_dim + hidden_dim;

        // Mixer blocks
        let token_mlp_params = (num_patches * self.config.token_mlp_dim
            + self.config.token_mlp_dim)
            + (self.config.token_mlp_dim * num_patches + num_patches);
        let channel_mlp_params = (hidden_dim * self.config.channel_mlp_dim
            + self.config.channel_mlp_dim)
            + (self.config.channel_mlp_dim * hidden_dim + hidden_dim);
        let norm_params = 2 * hidden_dim; // gamma and beta
        let block_params = 2 * norm_params + token_mlp_params + channel_mlp_params;
        let all_blocks_params = self.config.num_blocks * block_params;

        // Head
        let head_params = hidden_dim * self.config.num_classes + self.config.num_classes;

        // Final norm
        let final_norm_params = 2 * hidden_dim;

        patch_embed_params + all_blocks_params + head_params + final_norm_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array4;
    use scirs2_core::random::rngs::SmallRng;
    use scirs2_core::random::SeedableRng;

    #[test]
    fn test_mlp_mixer_config_default() {
        let config = MLPMixerConfig::default();
        assert_eq!(config.image_size, 224);
        assert_eq!(config.patch_size, 16);
        assert_eq!(config.num_patches(), 196); // 14 * 14
    }

    #[test]
    fn test_mlp_mixer_config_variants() {
        let s32 = MLPMixerConfig::mixer_s_32(10);
        assert_eq!(s32.patch_size, 32);
        assert_eq!(s32.hidden_dim, 512);
        assert_eq!(s32.num_patches(), 49); // 7 * 7

        let b16 = MLPMixerConfig::mixer_b_16(100);
        assert_eq!(b16.patch_size, 16);
        assert_eq!(b16.hidden_dim, 768);
        assert_eq!(b16.num_blocks, 12);
    }

    #[test]
    fn test_mixer_mlp() {
        let mut rng = SmallRng::seed_from_u64(42);
        let mlp = MixerMLP::<f32>::new(64, 128, 64, 0.0, &mut rng).expect("Operation failed");

        let input = Array2::<f32>::zeros((10, 64)).into_dyn();
        let output = mlp.forward(&input).expect("Operation failed");

        assert_eq!(output.shape(), &[10, 64]);
    }

    #[test]
    fn test_mixer_block() {
        let mut rng = SmallRng::seed_from_u64(42);
        let block = MixerBlock::<f32>::new(
            16,  // num_patches
            64,  // hidden_dim
            32,  // token_mlp_dim
            128, // channel_mlp_dim
            0.0, // dropout
            &mut rng,
        )
        .expect("Operation failed");

        let input = Array3::<f32>::zeros((2, 16, 64));
        let output = block.forward(&input).expect("Operation failed");

        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_mlp_mixer_small() {
        let mut rng = SmallRng::seed_from_u64(42);

        // Small config for testing
        let config = MLPMixerConfig {
            image_size: 32,
            patch_size: 8,
            num_classes: 10,
            hidden_dim: 32,
            num_blocks: 2,
            token_mlp_dim: 16,
            channel_mlp_dim: 64,
            dropout_rate: 0.0,
            in_channels: 3,
        };

        let mixer = MLPMixer::<f32>::new(config.clone(), &mut rng).expect("Operation failed");

        // Test forward pass
        let images = Array4::<f32>::zeros((2, 3, 32, 32)).into_dyn();
        let output = mixer.forward(&images).expect("Operation failed");

        assert_eq!(output.shape(), &[2, 10]);
    }

    #[test]
    fn test_extract_patches() {
        let mut rng = SmallRng::seed_from_u64(42);

        let config = MLPMixerConfig {
            image_size: 8,
            patch_size: 4,
            num_classes: 2,
            hidden_dim: 16,
            num_blocks: 1,
            token_mlp_dim: 8,
            channel_mlp_dim: 32,
            dropout_rate: 0.0,
            in_channels: 1,
        };

        let mixer = MLPMixer::<f32>::new(config.clone(), &mut rng).expect("Operation failed");

        // Create test image: 1 batch, 1 channel, 8x8
        let mut images = Array4::<f32>::zeros((1, 1, 8, 8));
        for h in 0..8 {
            for w in 0..8 {
                images[[0, 0, h, w]] = (h * 8 + w) as f32;
            }
        }

        let patches = mixer
            .extract_patches(&images.into_dyn())
            .expect("Operation failed");

        // Should have 4 patches (2x2 grid of 4x4 patches)
        assert_eq!(patches.shape(), &[1, 4, 16]);

        // First patch (top-left) should contain values 0-15 from a 4x4 region
        // Actually it contains values from positions (0,0) to (3,3)
        assert_eq!(patches[[0, 0, 0]], 0.0); // Top-left of first patch
    }

    #[test]
    fn test_num_parameters() {
        let config = MLPMixerConfig::mixer_s_16(1000);
        let mut rng = SmallRng::seed_from_u64(42);
        let mixer = MLPMixer::<f32>::new(config, &mut rng).expect("Operation failed");

        let params = mixer.num_parameters();
        assert!(params > 0);
        // Mixer-S/16 should have roughly 18M parameters
        // Our calculation is approximate
        println!("Estimated parameters: {}", params);
    }
}
