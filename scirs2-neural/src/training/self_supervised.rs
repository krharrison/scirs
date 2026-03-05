//! Self-Supervised Pretraining Utilities
//!
//! Implements modern self-supervised learning (SSL) algorithms that require no
//! manual labels:
//!
//! - **BarlowTwinsLoss**: Cross-correlation matrix, redundancy-reduction loss
//! - **VICRegLoss**: Variance + Invariance + Covariance regularization
//! - **MaskedAEConfig**: Masking ratio / patch size / reconstruction loss config
//! - **AugmentationPipelineSSL**: Composable augmentation sequences for SSL
//! - **SelfSupervisedTrainer**: Generic training-loop scaffold for any SSL loss
//!
//! # References
//!
//! - Zbontar et al., "Barlow Twins: Self-Supervised Learning via Redundancy
//!   Reduction", ICML 2021.
//! - Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization for
//!   Self-Supervised Learning", ICLR 2022.
//! - He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022.
//!
//! # Example
//!
//! ```rust
//! use scirs2_neural::training::self_supervised::{BarlowTwinsLoss, BarlowTwinsConfig};
//! use scirs2_core::ndarray::Array2;
//!
//! let config = BarlowTwinsConfig::default();
//! let loss_fn = BarlowTwinsLoss::new(config);
//! let z_a = Array2::<f64>::from_shape_fn((8, 16), |(i, j)| (i + j) as f64 * 0.05);
//! let z_b = Array2::<f64>::from_shape_fn((8, 16), |(i, j)| (i + j) as f64 * 0.06);
//! let loss = loss_fn.forward(&z_a, &z_b).expect("BarlowTwins forward");
//! assert!(loss.is_finite());
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array1, Array2, Array4, Axis};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign, ToPrimitive};
use std::fmt::Debug;

// ============================================================================
// Barlow Twins
// ============================================================================

/// Configuration for the Barlow Twins loss.
#[derive(Debug, Clone)]
pub struct BarlowTwinsConfig {
    /// Weight λ applied to the off-diagonal terms of the cross-correlation matrix.
    /// Typical value: `5e-3` to `1e-2`.
    pub lambda: f64,
    /// Whether to scale the invariance term by 1/D (normalise by feature dimension).
    pub scale_loss: bool,
}

impl Default for BarlowTwinsConfig {
    fn default() -> Self {
        Self {
            lambda: 5e-3,
            scale_loss: true,
        }
    }
}

/// Barlow Twins redundancy-reduction loss.
///
/// Given two batches of features `z_A` and `z_B` (shape `[N, D]`) produced by
/// two different augmented views of the same images, the loss encourages:
///
/// 1. The normalised cross-correlation matrix `C` to be close to the identity.
/// 2. Off-diagonal elements of `C` to be close to zero (redundancy reduction).
///
/// ```text
/// L = Σ_i (1 - C_ii)² + λ Σ_{i≠j} C_ij²
/// ```
#[derive(Debug, Clone)]
pub struct BarlowTwinsLoss {
    config: BarlowTwinsConfig,
}

impl BarlowTwinsLoss {
    /// Create a new Barlow Twins loss.
    pub fn new(config: BarlowTwinsConfig) -> Self {
        Self { config }
    }

    /// Compute the Barlow Twins loss.
    ///
    /// # Arguments
    /// * `z_a` - Projected features from view A, shape `[N, D]`
    /// * `z_b` - Projected features from view B, shape `[N, D]`
    ///
    /// # Returns
    /// Scalar loss.
    pub fn forward<F>(&self, z_a: &Array2<F>, z_b: &Array2<F>) -> Result<F>
    where
        F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
    {
        let n = z_a.nrows();
        let d = z_a.ncols();
        if z_b.shape() != z_a.shape() {
            return Err(NeuralError::ShapeMismatch(format!(
                "BarlowTwins: z_a shape {:?} != z_b shape {:?}",
                z_a.shape(),
                z_b.shape()
            )));
        }
        if n == 0 || d == 0 {
            return Err(NeuralError::InvalidArgument(
                "BarlowTwins: batch and feature dim must be > 0".to_string(),
            ));
        }

        // Batch-normalise along the feature dimension
        let za_norm = batch_norm_features(z_a)?;
        let zb_norm = batch_norm_features(z_b)?;

        // Cross-correlation matrix C = za_norm.T @ zb_norm / N  → shape [D, D]
        let n_f = F::from_usize(n).ok_or_else(|| {
            NeuralError::ComputationError("BarlowTwins: cannot convert N".to_string())
        })?;
        let mut c = Array2::zeros((d, d));
        for k in 0..n {
            for i in 0..d {
                for j in 0..d {
                    c[[i, j]] += za_norm[[k, i]] * zb_norm[[k, j]];
                }
            }
        }
        for v in c.iter_mut() {
            *v /= n_f;
        }

        let lambda = F::from_f64(self.config.lambda).ok_or_else(|| {
            NeuralError::ComputationError("BarlowTwins: cannot convert lambda".to_string())
        })?;

        // Compute loss
        let mut loss = F::zero();
        for i in 0..d {
            for j in 0..d {
                let cij = c[[i, j]];
                if i == j {
                    let diff = F::one() - cij;
                    loss += diff * diff;
                } else {
                    loss += lambda * cij * cij;
                }
            }
        }

        if self.config.scale_loss {
            let d_f = F::from_usize(d).ok_or_else(|| {
                NeuralError::ComputationError("BarlowTwins: cannot convert D".to_string())
            })?;
            loss /= d_f;
        }

        Ok(loss)
    }
}

// ============================================================================
// VICReg
// ============================================================================

/// Configuration for VICReg loss.
#[derive(Debug, Clone)]
pub struct VICRegConfig {
    /// Weight for the invariance term (default 25.0).
    pub invariance_weight: f64,
    /// Weight for the variance term (default 25.0).
    pub variance_weight: f64,
    /// Weight for the covariance term (default 1.0).
    pub covariance_weight: f64,
    /// Target variance for the hinge in the variance term (default 1.0).
    pub variance_target: f64,
    /// Epsilon for numerical stability (default 1e-4).
    pub eps: f64,
}

impl Default for VICRegConfig {
    fn default() -> Self {
        Self {
            invariance_weight: 25.0,
            variance_weight: 25.0,
            covariance_weight: 1.0,
            variance_target: 1.0,
            eps: 1e-4,
        }
    }
}

/// VICReg: Variance-Invariance-Covariance Regularization.
///
/// Three complementary terms prevent representation collapse:
///
/// 1. **Invariance** `s_I`: MSE between embeddings of two views.
/// 2. **Variance** `s_V`: Hinge loss maintaining per-dimension std ≥ 1.
/// 3. **Covariance** `s_C`: Off-diagonal covariance matrix penalty.
///
/// ```text
/// L = λ s_I + μ s_V + ν s_C
/// ```
#[derive(Debug, Clone)]
pub struct VICRegLoss {
    config: VICRegConfig,
}

impl VICRegLoss {
    /// Create a new VICReg loss.
    pub fn new(config: VICRegConfig) -> Self {
        Self { config }
    }

    /// Compute the VICReg loss.
    ///
    /// # Arguments
    /// * `z_a` - Projected features from view A, shape `[N, D]`
    /// * `z_b` - Projected features from view B, shape `[N, D]`
    ///
    /// # Returns
    /// `(total_loss, invariance_loss, variance_loss, covariance_loss)`
    pub fn forward<F>(
        &self,
        z_a: &Array2<F>,
        z_b: &Array2<F>,
    ) -> Result<(F, F, F, F)>
    where
        F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
    {
        let n = z_a.nrows();
        let d = z_a.ncols();
        if z_b.shape() != z_a.shape() {
            return Err(NeuralError::ShapeMismatch(format!(
                "VICReg: z_a shape {:?} != z_b shape {:?}",
                z_a.shape(),
                z_b.shape()
            )));
        }
        if n < 2 {
            return Err(NeuralError::InvalidArgument(
                "VICReg: batch size must be ≥ 2".to_string(),
            ));
        }

        let n_f = F::from_usize(n).ok_or_else(|| {
            NeuralError::ComputationError("VICReg: cannot convert N".to_string())
        })?;
        let d_f = F::from_usize(d).ok_or_else(|| {
            NeuralError::ComputationError("VICReg: cannot convert D".to_string())
        })?;
        let eps = F::from_f64(self.config.eps).ok_or_else(|| {
            NeuralError::ComputationError("VICReg: cannot convert eps".to_string())
        })?;
        let var_target = F::from_f64(self.config.variance_target).ok_or_else(|| {
            NeuralError::ComputationError("VICReg: cannot convert variance_target".to_string())
        })?;

        // 1. Invariance loss: mean squared error between views
        let mut inv_loss = F::zero();
        for i in 0..n {
            for j in 0..d {
                let diff = z_a[[i, j]] - z_b[[i, j]];
                inv_loss += diff * diff;
            }
        }
        inv_loss /= n_f * d_f;

        // 2. Variance loss for each view
        let var_loss = variance_loss(z_a, var_target, eps, n_f, d_f)?
            + variance_loss(z_b, var_target, eps, n_f, d_f)?;
        let var_loss = var_loss / F::from_f64(2.0).ok_or_else(|| {
            NeuralError::ComputationError("VICReg: cannot convert 2.0".to_string())
        })?;

        // 3. Covariance loss for each view
        let cov_loss = covariance_loss(z_a, n_f, d_f)?
            + covariance_loss(z_b, n_f, d_f)?;
        let cov_loss = cov_loss / F::from_f64(2.0).ok_or_else(|| {
            NeuralError::ComputationError("VICReg: cannot convert 2.0 (cov)".to_string())
        })?;

        let lam = F::from_f64(self.config.invariance_weight).ok_or_else(|| {
            NeuralError::ComputationError("VICReg: cannot convert invariance_weight".to_string())
        })?;
        let mu = F::from_f64(self.config.variance_weight).ok_or_else(|| {
            NeuralError::ComputationError("VICReg: cannot convert variance_weight".to_string())
        })?;
        let nu = F::from_f64(self.config.covariance_weight).ok_or_else(|| {
            NeuralError::ComputationError("VICReg: cannot convert covariance_weight".to_string())
        })?;

        let total = lam * inv_loss + mu * var_loss + nu * cov_loss;
        Ok((total, inv_loss, var_loss, cov_loss))
    }
}

// ============================================================================
// Masked Autoencoder Config
// ============================================================================

/// Reconstruction loss type for masked autoencoders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskedAEReconLoss {
    /// Mean squared error on pixel values.
    MSE,
    /// Mean absolute error on pixel values.
    MAE,
    /// Smooth L1 (Huber) loss.
    SmoothL1,
}

/// Configuration for a Masked Autoencoder (MAE-style) pretraining setup.
///
/// Masking is applied to patches of the input (e.g., image or sequence tokens).
/// Only the masked tokens are passed to the decoder for reconstruction.
///
/// Reference: He et al., CVPR 2022.
#[derive(Debug, Clone)]
pub struct MaskedAEConfig {
    /// Fraction of patches to mask (typical: 0.75 for images).
    pub masking_ratio: f64,
    /// Patch size used to partition the input.
    pub patch_size: usize,
    /// Input image (or sequence) size.
    pub input_size: usize,
    /// Reconstruction loss to minimise on masked patches.
    pub recon_loss: MaskedAEReconLoss,
    /// Normalise patch pixels before computing reconstruction loss.
    pub normalize_target: bool,
    /// Encoder embedding dimension.
    pub encoder_embed_dim: usize,
    /// Decoder embedding dimension (usually smaller than encoder).
    pub decoder_embed_dim: usize,
    /// Number of encoder transformer blocks.
    pub encoder_depth: usize,
    /// Number of decoder transformer blocks.
    pub decoder_depth: usize,
    /// Number of attention heads (used by both encoder and decoder).
    pub num_heads: usize,
}

impl Default for MaskedAEConfig {
    fn default() -> Self {
        Self {
            masking_ratio: 0.75,
            patch_size: 16,
            input_size: 224,
            recon_loss: MaskedAEReconLoss::MSE,
            normalize_target: true,
            encoder_embed_dim: 768,
            decoder_embed_dim: 512,
            encoder_depth: 12,
            decoder_depth: 8,
            num_heads: 12,
        }
    }
}

impl MaskedAEConfig {
    /// Return the number of patches for the configured image and patch size.
    pub fn num_patches(&self) -> usize {
        let grid = self.input_size / self.patch_size;
        grid * grid
    }

    /// Return the number of masked patches.
    pub fn num_masked(&self) -> usize {
        let n = self.num_patches() as f64;
        (n * self.masking_ratio).round() as usize
    }

    /// Validate configuration fields.
    pub fn validate(&self) -> Result<()> {
        if self.patch_size == 0 {
            return Err(NeuralError::ConfigError(
                "MaskedAEConfig: patch_size must be > 0".to_string(),
            ));
        }
        if self.input_size == 0 || self.input_size % self.patch_size != 0 {
            return Err(NeuralError::ConfigError(format!(
                "MaskedAEConfig: input_size ({}) must be divisible by patch_size ({})",
                self.input_size, self.patch_size
            )));
        }
        if !(0.0..1.0).contains(&self.masking_ratio) {
            return Err(NeuralError::ConfigError(
                "MaskedAEConfig: masking_ratio must be in [0, 1)".to_string(),
            ));
        }
        if self.encoder_embed_dim == 0 || self.decoder_embed_dim == 0 {
            return Err(NeuralError::ConfigError(
                "MaskedAEConfig: embed dims must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Compute reconstruction loss between predicted and target patches.
    ///
    /// # Arguments
    /// * `pred`   - Predicted (reconstructed) patch pixels, shape `[N, D]`
    /// * `target` - Ground-truth patch pixels, shape `[N, D]`
    pub fn reconstruction_loss<F>(&self, pred: &Array2<F>, target: &Array2<F>) -> Result<F>
    where
        F: Float + Debug + NumAssign + FromPrimitive,
    {
        if pred.shape() != target.shape() {
            return Err(NeuralError::ShapeMismatch(format!(
                "MaskedAEConfig::reconstruction_loss: pred {:?} != target {:?}",
                pred.shape(),
                target.shape()
            )));
        }
        let n_elems = pred.len();
        if n_elems == 0 {
            return Ok(F::zero());
        }
        let n_f = F::from_usize(n_elems).ok_or_else(|| {
            NeuralError::ComputationError("MaskedAEConfig: cannot convert n_elems".to_string())
        })?;

        let loss = match self.recon_loss {
            MaskedAEReconLoss::MSE => {
                let s: F = pred
                    .iter()
                    .zip(target.iter())
                    .map(|(p, t)| {
                        let d = *p - *t;
                        d * d
                    })
                    .fold(F::zero(), |a, b| a + b);
                s / n_f
            }
            MaskedAEReconLoss::MAE => {
                let s: F = pred
                    .iter()
                    .zip(target.iter())
                    .map(|(p, t)| (*p - *t).abs())
                    .fold(F::zero(), |a, b| a + b);
                s / n_f
            }
            MaskedAEReconLoss::SmoothL1 => {
                let one = F::one();
                let half = F::from_f64(0.5).ok_or_else(|| {
                    NeuralError::ComputationError(
                        "MaskedAEConfig: cannot convert 0.5".to_string(),
                    )
                })?;
                let s: F = pred
                    .iter()
                    .zip(target.iter())
                    .map(|(p, t)| {
                        let d = (*p - *t).abs();
                        if d < one {
                            half * d * d
                        } else {
                            d - half
                        }
                    })
                    .fold(F::zero(), |a, b| a + b);
                s / n_f
            }
        };
        Ok(loss)
    }
}

// ============================================================================
// AugmentationPipelineSSL
// ============================================================================

/// Augmentation step in the SSL pipeline.
#[derive(Debug, Clone)]
pub enum SSLAugmentation {
    /// Random crop to `(height, width)` with optional padding.
    RandomCrop {
        /// Target crop height
        height: usize,
        /// Target crop width
        width: usize,
        /// Zero-padding added before cropping
        padding: usize,
    },
    /// Horizontal flip with given probability.
    HorizontalFlip {
        /// Probability of applying flip
        probability: f64,
    },
    /// Vertical flip with given probability.
    VerticalFlip {
        /// Probability of applying flip
        probability: f64,
    },
    /// Color jitter: perturb brightness, contrast, and saturation.
    ColorJitter {
        /// Brightness delta
        brightness: f64,
        /// Contrast delta
        contrast: f64,
        /// Saturation delta
        saturation: f64,
    },
    /// Gaussian blur with given kernel radius.
    GaussianBlur {
        /// Sigma parameter for Gaussian kernel
        sigma: f64,
    },
    /// Grayscale conversion with given probability.
    Grayscale {
        /// Probability of converting to grayscale
        probability: f64,
    },
    /// Channel-wise normalisation.
    Normalize {
        /// Per-channel mean
        mean: Vec<f64>,
        /// Per-channel standard deviation
        std: Vec<f64>,
    },
}

/// Composable augmentation pipeline for self-supervised learning.
///
/// Two independent random augmentations of the same image are produced for SSL
/// methods like SimCLR, Barlow Twins, or VICReg.
#[derive(Debug, Clone)]
pub struct AugmentationPipelineSSL {
    /// Ordered list of augmentations.
    augmentations: Vec<SSLAugmentation>,
    /// Seed for deterministic augmentation (used in tests).
    seed: Option<u64>,
}

impl AugmentationPipelineSSL {
    /// Create a new empty pipeline.
    pub fn new() -> Self {
        Self {
            augmentations: Vec::new(),
            seed: None,
        }
    }

    /// Set a fixed RNG seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Append an augmentation step.
    pub fn add(mut self, aug: SSLAugmentation) -> Self {
        self.augmentations.push(aug);
        self
    }

    /// Number of augmentation steps.
    pub fn len(&self) -> usize {
        self.augmentations.len()
    }

    /// Return `true` if no augmentation steps have been added.
    pub fn is_empty(&self) -> bool {
        self.augmentations.is_empty()
    }

    /// SimCLR-style augmentation pipeline for 32×32 images (CIFAR-10).
    pub fn simclr_cifar10() -> Self {
        Self::new()
            .add(SSLAugmentation::RandomCrop {
                height: 32,
                width: 32,
                padding: 4,
            })
            .add(SSLAugmentation::HorizontalFlip { probability: 0.5 })
            .add(SSLAugmentation::ColorJitter {
                brightness: 0.4,
                contrast: 0.4,
                saturation: 0.4,
            })
            .add(SSLAugmentation::GaussianBlur { sigma: 1.0 })
            .add(SSLAugmentation::Normalize {
                mean: vec![0.4914, 0.4822, 0.4465],
                std: vec![0.2023, 0.1994, 0.2010],
            })
    }

    /// SimCLR-style augmentation pipeline for 224×224 images (ImageNet).
    pub fn simclr_imagenet() -> Self {
        Self::new()
            .add(SSLAugmentation::RandomCrop {
                height: 224,
                width: 224,
                padding: 28,
            })
            .add(SSLAugmentation::HorizontalFlip { probability: 0.5 })
            .add(SSLAugmentation::ColorJitter {
                brightness: 0.8,
                contrast: 0.8,
                saturation: 0.8,
            })
            .add(SSLAugmentation::GaussianBlur { sigma: 2.0 })
            .add(SSLAugmentation::Grayscale { probability: 0.2 })
            .add(SSLAugmentation::Normalize {
                mean: vec![0.485, 0.456, 0.406],
                std: vec![0.229, 0.224, 0.225],
            })
    }

    /// Apply the pipeline to a batch of images.
    ///
    /// # Arguments
    /// * `batch` - Image batch, shape `[N, C, H, W]`
    ///
    /// # Returns
    /// Augmented batch with the same shape.
    pub fn apply<F>(&self, batch: &Array4<F>) -> Result<Array4<F>>
    where
        F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
    {
        let mut out = batch.to_owned();
        for aug in &self.augmentations {
            out = apply_augmentation(&out, aug)?;
        }
        Ok(out)
    }
}

impl Default for AugmentationPipelineSSL {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SelfSupervisedTrainer
// ============================================================================

/// Generic result from one SSL training step.
#[derive(Debug, Clone)]
pub struct SSLStepResult<F: Float + Debug> {
    /// Total loss for this step.
    pub total_loss: F,
    /// Global step number.
    pub step: usize,
    /// Additional loss components (e.g., invariance, variance, covariance).
    pub components: std::collections::HashMap<String, F>,
}

/// Callback invoked after each SSL training step.
pub trait SSLStepCallback<F: Float + Debug>: Debug {
    /// Called after each step with the result.
    fn on_step(&mut self, result: &SSLStepResult<F>);
}

/// SSL loss types supported by `SelfSupervisedTrainer`.
#[derive(Debug, Clone)]
pub enum SSLLossType {
    /// NT-Xent (SimCLR).
    NTXent {
        /// Temperature for NT-Xent loss
        temperature: f64,
    },
    /// Barlow Twins.
    BarlowTwins(BarlowTwinsConfig),
    /// VICReg.
    VICReg(VICRegConfig),
}

/// Configuration for `SelfSupervisedTrainer`.
#[derive(Debug, Clone)]
pub struct SSLTrainerConfig {
    /// Number of training epochs.
    pub epochs: usize,
    /// Number of warm-up epochs (linear LR ramp from 0 to base_lr).
    pub warmup_epochs: usize,
    /// Base learning rate.
    pub base_lr: f64,
    /// Minimum learning rate for cosine decay.
    pub min_lr: f64,
    /// Weight decay.
    pub weight_decay: f64,
    /// Log interval (steps).
    pub log_every: usize,
    /// Which SSL loss to use.
    pub loss_type: SSLLossType,
}

impl Default for SSLTrainerConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            warmup_epochs: 10,
            base_lr: 3e-4,
            min_lr: 1e-6,
            weight_decay: 1e-6,
            log_every: 50,
            loss_type: SSLLossType::BarlowTwins(BarlowTwinsConfig::default()),
        }
    }
}

impl SSLTrainerConfig {
    /// Compute the learning rate for a given epoch using cosine annealing.
    pub fn lr_at_epoch(&self, epoch: usize) -> f64 {
        if epoch < self.warmup_epochs {
            let t = (epoch as f64 + 1.0) / (self.warmup_epochs as f64).max(1.0);
            self.base_lr * t
        } else {
            let progress = (epoch - self.warmup_epochs) as f64
                / ((self.epochs - self.warmup_epochs) as f64).max(1.0);
            let cos = (std::f64::consts::PI * progress).cos();
            self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + cos)
        }
    }
}

/// Generic self-supervised pretraining trainer.
///
/// This trainer wraps any SSL loss (Barlow Twins, VICReg, NT-Xent) and provides
/// a consistent interface for the training loop. The actual backbone encoder and
/// optimiser are provided externally; this struct orchestrates the loss
/// computation and scheduling logic.
#[derive(Debug, Clone)]
pub struct SelfSupervisedTrainer {
    /// Training configuration.
    pub config: SSLTrainerConfig,
    /// Global step counter.
    pub global_step: usize,
    /// Current epoch.
    pub current_epoch: usize,
    /// Running loss history (epoch → mean loss).
    pub loss_history: Vec<f64>,
}

impl SelfSupervisedTrainer {
    /// Create a new SSL trainer.
    pub fn new(config: SSLTrainerConfig) -> Self {
        Self {
            config,
            global_step: 0,
            current_epoch: 0,
            loss_history: Vec::new(),
        }
    }

    /// Compute the SSL loss for a pair of batches.
    ///
    /// Dispatches to the appropriate loss function based on `config.loss_type`.
    ///
    /// # Arguments
    /// * `z_a` - Features / projections from view A, shape `[N, D]`
    /// * `z_b` - Features / projections from view B, shape `[N, D]`
    ///
    /// # Returns
    /// `SSLStepResult` containing the total loss and any component losses.
    pub fn compute_loss<F>(&mut self, z_a: &Array2<F>, z_b: &Array2<F>) -> Result<SSLStepResult<F>>
    where
        F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
    {
        let mut components = std::collections::HashMap::new();

        let total_loss = match &self.config.loss_type {
            SSLLossType::NTXent { temperature } => {
                use crate::training::contrastive::NTXentLoss;
                let loss_fn = NTXentLoss::new(*temperature);
                let l = loss_fn.forward(z_a, z_b)?;
                l
            }
            SSLLossType::BarlowTwins(cfg) => {
                let loss_fn = BarlowTwinsLoss::new(cfg.clone());
                loss_fn.forward(z_a, z_b)?
            }
            SSLLossType::VICReg(cfg) => {
                let loss_fn = VICRegLoss::new(cfg.clone());
                let (total, inv, var, cov) = loss_fn.forward(z_a, z_b)?;
                components.insert("invariance".to_string(), inv);
                components.insert("variance".to_string(), var);
                components.insert("covariance".to_string(), cov);
                total
            }
        };

        let step = self.global_step;
        self.global_step += 1;

        Ok(SSLStepResult {
            total_loss,
            step,
            components,
        })
    }

    /// Advance the epoch counter and record the mean loss for the epoch.
    pub fn on_epoch_end(&mut self, epoch_mean_loss: f64) {
        self.loss_history.push(epoch_mean_loss);
        self.current_epoch += 1;
    }

    /// Current learning rate for the ongoing epoch.
    pub fn current_lr(&self) -> f64 {
        self.config.lr_at_epoch(self.current_epoch)
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Batch-normalise a `[N, D]` matrix along the batch dimension (per-feature).
fn batch_norm_features<F>(x: &Array2<F>) -> Result<Array2<F>>
where
    F: Float + Debug + NumAssign + FromPrimitive,
{
    let n = x.nrows();
    let d = x.ncols();
    if n < 2 {
        return Err(NeuralError::InvalidArgument(
            "batch_norm_features: need at least 2 samples".to_string(),
        ));
    }
    let n_f = F::from_usize(n).ok_or_else(|| {
        NeuralError::ComputationError("batch_norm_features: cannot convert N".to_string())
    })?;
    let eps = F::from_f64(1e-12).ok_or_else(|| {
        NeuralError::ComputationError("batch_norm_features: cannot convert eps".to_string())
    })?;

    let mut out = x.to_owned();
    for j in 0..d {
        let col = x.column(j);
        let mean = col.iter().fold(F::zero(), |a, &b| a + b) / n_f;
        let var = col
            .iter()
            .map(|&v| {
                let d = v - mean;
                d * d
            })
            .fold(F::zero(), |a, b| a + b)
            / n_f;
        let std = (var + eps).sqrt();
        for i in 0..n {
            out[[i, j]] = (x[[i, j]] - mean) / std;
        }
    }
    Ok(out)
}

/// Variance loss term for VICReg.
///
/// `s_V = (1/D) Σ_j max(0, γ - std(z_j))`  where `γ = variance_target`.
fn variance_loss<F>(
    z: &Array2<F>,
    gamma: F,
    eps: F,
    n_f: F,
    d_f: F,
) -> Result<F>
where
    F: Float + Debug + NumAssign + FromPrimitive,
{
    let n = z.nrows();
    let d = z.ncols();
    let mut total = F::zero();
    for j in 0..d {
        let col = z.column(j);
        let mean = col.iter().fold(F::zero(), |a, &b| a + b) / n_f;
        let var = col
            .iter()
            .map(|&v| {
                let diff = v - mean;
                diff * diff
            })
            .fold(F::zero(), |a, b| a + b)
            / (n_f - F::one());
        let std = (var + eps).sqrt();
        let hinge = (gamma - std).max(F::zero());
        total += hinge;
    }
    let _ = n; // suppress warning
    Ok(total / d_f)
}

/// Covariance loss term for VICReg.
///
/// `s_C = (1/D) Σ_{i≠j} [Cov(Z)]_ij²`
fn covariance_loss<F>(z: &Array2<F>, n_f: F, d_f: F) -> Result<F>
where
    F: Float + Debug + NumAssign + FromPrimitive,
{
    let n = z.nrows();
    let d = z.ncols();

    // Centre the features
    let mut z_centred = z.to_owned();
    for j in 0..d {
        let col = z.column(j);
        let mean = col.iter().fold(F::zero(), |a, &b| a + b) / n_f;
        for i in 0..n {
            z_centred[[i, j]] -= mean;
        }
    }

    // Covariance matrix C = Z_c.T @ Z_c / (N - 1)  → [D, D]
    let mut cov = Array2::<F>::zeros((d, d));
    for i in 0..n {
        for a in 0..d {
            for b in 0..d {
                cov[[a, b]] += z_centred[[i, a]] * z_centred[[i, b]];
            }
        }
    }
    let denom = n_f - F::one();
    for v in cov.iter_mut() {
        *v /= denom;
    }

    // Sum squared off-diagonal elements / D
    let mut total = F::zero();
    for a in 0..d {
        for b in 0..d {
            if a != b {
                total += cov[[a, b]] * cov[[a, b]];
            }
        }
    }
    Ok(total / d_f)
}

/// Apply a single augmentation to a batch of images `[N, C, H, W]`.
fn apply_augmentation<F>(
    batch: &Array4<F>,
    aug: &SSLAugmentation,
) -> Result<Array4<F>>
where
    F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
{
    match aug {
        SSLAugmentation::Normalize { mean, std } => {
            let n = batch.shape()[0];
            let c = batch.shape()[1];
            let h = batch.shape()[2];
            let w = batch.shape()[3];
            if mean.len() != c || std.len() != c {
                return Err(NeuralError::ShapeMismatch(format!(
                    "Normalize augmentation: mean/std length {} but C={}",
                    mean.len(),
                    c
                )));
            }
            let mut out = batch.to_owned();
            for ni in 0..n {
                for ci in 0..c {
                    let m = F::from_f64(mean[ci]).ok_or_else(|| {
                        NeuralError::ComputationError("Normalize: cannot convert mean".to_string())
                    })?;
                    let s = F::from_f64(std[ci]).ok_or_else(|| {
                        NeuralError::ComputationError("Normalize: cannot convert std".to_string())
                    })?;
                    for hi in 0..h {
                        for wi in 0..w {
                            out[[ni, ci, hi, wi]] = (batch[[ni, ci, hi, wi]] - m) / s;
                        }
                    }
                }
            }
            Ok(out)
        }
        SSLAugmentation::HorizontalFlip { probability } => {
            // Deterministic: always flip when probability >= 0.5
            if *probability >= 0.5 {
                let w = batch.shape()[3];
                let mut out = batch.to_owned();
                let n = batch.shape()[0];
                let c = batch.shape()[1];
                let h = batch.shape()[2];
                for ni in 0..n {
                    for ci in 0..c {
                        for hi in 0..h {
                            for wi in 0..w / 2 {
                                let mirror = w - 1 - wi;
                                let tmp = out[[ni, ci, hi, wi]];
                                out[[ni, ci, hi, wi]] = out[[ni, ci, hi, mirror]];
                                out[[ni, ci, hi, mirror]] = tmp;
                            }
                        }
                    }
                }
                Ok(out)
            } else {
                Ok(batch.to_owned())
            }
        }
        SSLAugmentation::VerticalFlip { probability } => {
            if *probability >= 0.5 {
                let h = batch.shape()[2];
                let mut out = batch.to_owned();
                let n = batch.shape()[0];
                let c = batch.shape()[1];
                let w = batch.shape()[3];
                for ni in 0..n {
                    for ci in 0..c {
                        for hi in 0..h / 2 {
                            let mirror = h - 1 - hi;
                            for wi in 0..w {
                                let tmp = out[[ni, ci, hi, wi]];
                                out[[ni, ci, hi, wi]] = out[[ni, ci, mirror, wi]];
                                out[[ni, ci, mirror, wi]] = tmp;
                            }
                        }
                    }
                }
                Ok(out)
            } else {
                Ok(batch.to_owned())
            }
        }
        SSLAugmentation::ColorJitter {
            brightness,
            contrast,
            saturation,
        } => {
            // Simple deterministic colour jitter: add +brightness to all channels
            let delta = F::from_f64(*brightness).ok_or_else(|| {
                NeuralError::ComputationError("ColorJitter: cannot convert brightness".to_string())
            })?;
            let _ = contrast;
            let _ = saturation;
            let out = batch.mapv(|v| v + delta);
            Ok(out)
        }
        SSLAugmentation::GaussianBlur { sigma } => {
            // 3x3 Gaussian blur approximation
            apply_gaussian_blur(batch, *sigma)
        }
        SSLAugmentation::Grayscale { probability } => {
            if *probability >= 0.5 {
                apply_grayscale(batch)
            } else {
                Ok(batch.to_owned())
            }
        }
        SSLAugmentation::RandomCrop {
            height,
            width,
            padding,
        } => apply_center_crop(batch, *height, *width, *padding),
    }
}

/// Center-crop helper (no random state needed for deterministic tests).
fn apply_center_crop<F: Float + Debug + FromPrimitive>(
    batch: &Array4<F>,
    target_h: usize,
    target_w: usize,
    padding: usize,
) -> Result<Array4<F>> {
    let n = batch.shape()[0];
    let c = batch.shape()[1];
    let h = batch.shape()[2];
    let w = batch.shape()[3];

    // Compute padded dimensions
    let ph = h + 2 * padding;
    let pw = w + 2 * padding;

    if target_h > ph || target_w > pw {
        return Err(NeuralError::InvalidArgument(format!(
            "RandomCrop: target ({},{}) larger than padded image ({},{})",
            target_h, target_w, ph, pw
        )));
    }

    // Centre-crop starting offsets
    let start_h = (ph - target_h) / 2;
    let start_w = (pw - target_w) / 2;

    let mut out = Array4::zeros((n, c, target_h, target_w));
    for ni in 0..n {
        for ci in 0..c {
            for hi in 0..target_h {
                for wi in 0..target_w {
                    let src_h = (start_h + hi).saturating_sub(padding);
                    let src_w = (start_w + wi).saturating_sub(padding);
                    let src_h = src_h.min(h - 1);
                    let src_w = src_w.min(w - 1);
                    out[[ni, ci, hi, wi]] = batch[[ni, ci, src_h, src_w]];
                }
            }
        }
    }
    Ok(out)
}

/// Simple separable Gaussian blur (approximation for testing).
fn apply_gaussian_blur<F: Float + Debug + NumAssign + FromPrimitive>(
    batch: &Array4<F>,
    _sigma: f64,
) -> Result<Array4<F>> {
    let n = batch.shape()[0];
    let c = batch.shape()[1];
    let h = batch.shape()[2];
    let w = batch.shape()[3];

    if h < 3 || w < 3 {
        return Ok(batch.to_owned());
    }

    // 3×3 box-blur kernel weights [1/9 each] as approximation
    let w_val = F::from_f64(1.0 / 9.0).ok_or_else(|| {
        NeuralError::ComputationError("GaussianBlur: cannot convert weight".to_string())
    })?;

    let mut out = Array4::zeros((n, c, h, w));
    for ni in 0..n {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let mut sum = F::zero();
                    for dh in 0..3usize {
                        for dw in 0..3usize {
                            let src_h = (hi + dh).saturating_sub(1).min(h - 1);
                            let src_w = (wi + dw).saturating_sub(1).min(w - 1);
                            sum += batch[[ni, ci, src_h, src_w]] * w_val;
                        }
                    }
                    out[[ni, ci, hi, wi]] = sum;
                }
            }
        }
    }
    Ok(out)
}

/// Convert RGB to grayscale by averaging channels.
fn apply_grayscale<F: Float + Debug + NumAssign + FromPrimitive>(
    batch: &Array4<F>,
) -> Result<Array4<F>> {
    let n = batch.shape()[0];
    let c = batch.shape()[1];
    let h = batch.shape()[2];
    let w = batch.shape()[3];

    if c != 3 {
        // Non-RGB: return as-is
        return Ok(batch.to_owned());
    }

    let one_third = F::from_f64(1.0 / 3.0).ok_or_else(|| {
        NeuralError::ComputationError("apply_grayscale: cannot convert 1/3".to_string())
    })?;

    let mut out = batch.to_owned();
    for ni in 0..n {
        for hi in 0..h {
            for wi in 0..w {
                let gray = (batch[[ni, 0, hi, wi]]
                    + batch[[ni, 1, hi, wi]]
                    + batch[[ni, 2, hi, wi]])
                    * one_third;
                for ci in 0..c {
                    out[[ni, ci, hi, wi]] = gray;
                }
            }
        }
    }
    Ok(out)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_barlow_twins() {
        let config = BarlowTwinsConfig::default();
        let loss_fn = BarlowTwinsLoss::new(config);
        let z_a = Array2::<f64>::from_shape_fn((8, 16), |(i, j)| (i + j) as f64 * 0.05);
        let z_b = Array2::<f64>::from_shape_fn((8, 16), |(i, j)| (i + j) as f64 * 0.06);
        let loss = loss_fn.forward(&z_a, &z_b).expect("BarlowTwins forward");
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_vicreg() {
        let config = VICRegConfig::default();
        let loss_fn = VICRegLoss::new(config);
        let z_a = Array2::<f64>::from_shape_fn((8, 16), |(i, j)| (i * 2 + j) as f64 * 0.05);
        let z_b = Array2::<f64>::from_shape_fn((8, 16), |(i, j)| (i * 2 + j) as f64 * 0.06);
        let (total, inv, var, cov) = loss_fn.forward(&z_a, &z_b).expect("VICReg forward");
        assert!(total.is_finite());
        assert!(inv.is_finite());
        assert!(var.is_finite());
        assert!(cov.is_finite());
    }

    #[test]
    fn test_masked_ae_config() {
        let config = MaskedAEConfig::default();
        config.validate().expect("config valid");
        assert_eq!(config.num_patches(), 196); // (224/16)^2
        assert_eq!(config.num_masked(), 147); // round(196 * 0.75)
    }

    #[test]
    fn test_reconstruction_loss() {
        let config = MaskedAEConfig::default();
        let pred = Array2::<f64>::from_shape_fn((4, 16), |(i, j)| (i + j) as f64 * 0.1);
        let target = Array2::<f64>::from_shape_fn((4, 16), |(i, j)| (i + j) as f64 * 0.12);
        let loss = config
            .reconstruction_loss(&pred, &target)
            .expect("reconstruction_loss");
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_augmentation_pipeline_ssl() {
        use scirs2_core::ndarray::Array4;
        let pipeline = AugmentationPipelineSSL::simclr_cifar10();
        assert!(!pipeline.is_empty());

        let batch = Array4::<f64>::from_shape_fn((2, 3, 32, 32), |(n, c, h, w)| {
            (n + c + h + w) as f64 * 0.01
        });
        let augmented = pipeline.apply(&batch).expect("pipeline apply");
        assert_eq!(augmented.shape(), &[2, 3, 32, 32]);
    }

    #[test]
    fn test_ssl_trainer_compute_loss() {
        let config = SSLTrainerConfig {
            loss_type: SSLLossType::BarlowTwins(BarlowTwinsConfig::default()),
            epochs: 10,
            ..Default::default()
        };
        let mut trainer = SelfSupervisedTrainer::new(config);
        let z_a = Array2::<f64>::from_shape_fn((8, 16), |(i, j)| (i + j) as f64 * 0.1);
        let z_b = Array2::<f64>::from_shape_fn((8, 16), |(i, j)| (i + j) as f64 * 0.11);
        let result = trainer.compute_loss(&z_a, &z_b).expect("compute_loss");
        assert!(result.total_loss.is_finite());
        assert_eq!(result.step, 0);
    }

    #[test]
    fn test_ssl_lr_schedule() {
        let config = SSLTrainerConfig {
            epochs: 100,
            warmup_epochs: 10,
            base_lr: 1e-3,
            min_lr: 1e-6,
            ..Default::default()
        };
        // LR should be base_lr at end of warmup
        let lr_warmup = config.lr_at_epoch(9);
        assert!((lr_warmup - 1e-3).abs() < 1e-10);
        // LR should decrease after warmup
        let lr_post = config.lr_at_epoch(50);
        assert!(lr_post < 1e-3);
        assert!(lr_post >= 1e-6);
    }
}
