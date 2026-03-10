//! Data augmentation pipeline for neural network training
//!
//! Provides a composable pipeline of image transforms commonly used in
//! computer-vision training workflows, including geometric, photometric,
//! noise-based, and regularisation augmentations such as Mixup and CutMix.
//!
//! # Example
//!
//! ```rust
//! use scirs2_neural::training::augmentation::{AugmentationPipeline, AugmentationType};
//! use scirs2_core::ndarray::Array4;
//!
//! let pipeline = AugmentationPipeline::<f64>::cifar10_train();
//! let batch = Array4::<f64>::zeros((16, 3, 32, 32));
//! let augmented = pipeline.apply(&batch).expect("augmentation failed");
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{s, Array2, Array3, Array4, Axis, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign};
use scirs2_core::random::rngs::SmallRng;
use scirs2_core::random::{Distribution, Normal, RandBeta, Rng, SeedableRng};
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// AugmentationType
// ---------------------------------------------------------------------------

/// Describes a single augmentation transform that can be applied to images.
///
/// Transforms are applied sequentially by [`AugmentationPipeline`].
/// Each variant carries its own hyper-parameters.
#[derive(Debug, Clone)]
pub enum AugmentationType {
    /// Pad the image by `padding` pixels on each side, then randomly crop
    /// to `(height, width)`.
    RandomCrop {
        /// Target crop height
        height: usize,
        /// Target crop width
        width: usize,
        /// Zero-padding added to each border before cropping
        padding: usize,
    },
    /// Flip horizontally with the given probability.
    RandomHorizontalFlip {
        /// Probability of applying the flip (0.0 to 1.0)
        probability: f64,
    },
    /// Flip vertically with the given probability.
    RandomVerticalFlip {
        /// Probability of applying the flip (0.0 to 1.0)
        probability: f64,
    },
    /// Random rotation up to `max_degrees` in either direction.
    RandomRotation {
        /// Maximum rotation angle in degrees (applied symmetrically)
        max_degrees: f64,
    },
    /// Randomly perturb brightness, contrast, and saturation.
    ColorJitter {
        /// Maximum brightness delta
        brightness: f64,
        /// Maximum contrast delta
        contrast: f64,
        /// Maximum saturation delta
        saturation: f64,
    },
    /// Channel-wise normalisation with the given mean and std vectors.
    Normalize {
        /// Per-channel mean values
        mean: Vec<f64>,
        /// Per-channel standard deviation values
        std: Vec<f64>,
    },
    /// Randomly erase a rectangular region of the image.
    RandomErasing {
        /// Probability of applying erasing
        probability: f64,
        /// Range of proportions of erased area vs input image
        scale: (f64, f64),
        /// Range of aspect ratio of erased area
        ratio: (f64, f64),
    },
    /// Additive Gaussian noise.
    GaussianNoise {
        /// Standard deviation of the noise
        std: f64,
    },
    /// Cutout: zero-out `n_holes` square patches of side `length`.
    Cutout {
        /// Number of patches to cut out
        n_holes: usize,
        /// Side length of each square patch
        length: usize,
    },
    /// Mixup (batch-level): linearly interpolate pairs of images and labels.
    Mixup {
        /// Beta distribution parameter (alpha = beta)
        alpha: f64,
    },
    /// CutMix (batch-level): paste a rectangular patch from one sample onto
    /// another.
    CutMix {
        /// Beta distribution parameter (alpha = beta)
        alpha: f64,
    },
    /// Random affine transform (rotation + optional translate + optional scale).
    RandomAffine {
        /// Maximum rotation angle in degrees
        degrees: f64,
        /// Optional (max_dx, max_dy) as fractions of image size
        translate: Option<(f64, f64)>,
        /// Optional (min_scale, max_scale)
        scale: Option<(f64, f64)>,
    },
}

// ---------------------------------------------------------------------------
// AugmentationPipeline
// ---------------------------------------------------------------------------

/// A composable, optionally reproducible pipeline of image augmentations.
///
/// The pipeline stores a list of [`AugmentationType`] transforms and applies
/// them sequentially.  Per-sample transforms operate independently on each
/// image in the batch; batch-level transforms (Mixup, CutMix) should instead
/// be called via the free functions [`apply_mixup`] / [`apply_cutmix`].
#[derive(Debug, Clone)]
pub struct AugmentationPipeline<F> {
    transforms: Vec<AugmentationType>,
    seed: Option<u64>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + FromPrimitive + NumAssign + Debug> AugmentationPipeline<F> {
    // ----- construction -----

    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
            seed: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Append a transform and return `self` (builder pattern).
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, transform: AugmentationType) -> Self {
        self.transforms.push(transform);
        self
    }

    /// Fix the random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    // ----- predefined pipelines -----

    /// Standard CIFAR-10 training augmentation.
    ///
    /// RandomCrop(32, 32, padding=4) + RandomHorizontalFlip(0.5) +
    /// Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    pub fn cifar10_train() -> Self {
        Self::new()
            .add(AugmentationType::RandomCrop {
                height: 32,
                width: 32,
                padding: 4,
            })
            .add(AugmentationType::RandomHorizontalFlip { probability: 0.5 })
            .add(AugmentationType::Normalize {
                mean: vec![0.4914, 0.4822, 0.4465],
                std: vec![0.2470, 0.2435, 0.2616],
            })
    }

    /// Standard ImageNet training augmentation.
    ///
    /// RandomCrop(224, 224, padding=16) + RandomHorizontalFlip(0.5) +
    /// ColorJitter(0.4, 0.4, 0.4) +
    /// Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    pub fn imagenet_train() -> Self {
        Self::new()
            .add(AugmentationType::RandomCrop {
                height: 224,
                width: 224,
                padding: 16,
            })
            .add(AugmentationType::RandomHorizontalFlip { probability: 0.5 })
            .add(AugmentationType::ColorJitter {
                brightness: 0.4,
                contrast: 0.4,
                saturation: 0.4,
            })
            .add(AugmentationType::Normalize {
                mean: vec![0.485, 0.456, 0.406],
                std: vec![0.229, 0.224, 0.225],
            })
    }

    /// Minimal evaluation-time pipeline (normalisation only, ImageNet stats).
    pub fn basic_eval() -> Self {
        Self::new().add(AugmentationType::Normalize {
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
        })
    }

    // ----- application -----

    /// Apply the pipeline to a batch of images (N, C, H, W).
    pub fn apply(&self, images: &Array4<F>) -> Result<Array4<F>> {
        let mut rng = self.make_rng();
        let mut out = images.clone();
        for t in &self.transforms {
            out = apply_transform(&out, t, &mut rng)?;
        }
        Ok(out)
    }

    /// Apply the pipeline to a single image (C, H, W).
    pub fn apply_single(&self, image: &Array3<F>) -> Result<Array3<F>> {
        // Promote to batch dimension, apply, then squeeze
        let shape = image.shape();
        let batch = image
            .clone()
            .into_shape_with_order((1, shape[0], shape[1], shape[2]))
            .map_err(|e| NeuralError::ShapeMismatch(format!("reshape to batch: {e}")))?;
        let result = self.apply(&batch)?;
        let c = result.shape()[1];
        let h = result.shape()[2];
        let w = result.shape()[3];
        result
            .into_shape_with_order((c, h, w))
            .map_err(|e| NeuralError::ShapeMismatch(format!("squeeze batch: {e}")))
    }

    // ----- helpers -----

    fn make_rng(&self) -> SmallRng {
        match self.seed {
            Some(s) => SmallRng::seed_from_u64(s),
            None => SmallRng::from_rng(&mut scirs2_core::random::thread_rng()),
        }
    }
}

impl<F: Float + FromPrimitive + NumAssign + Debug> Default for AugmentationPipeline<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Per-transform dispatch (operates on Array4)
// ---------------------------------------------------------------------------

fn apply_transform<F: Float + FromPrimitive + NumAssign + Debug>(
    images: &Array4<F>,
    transform: &AugmentationType,
    rng: &mut SmallRng,
) -> Result<Array4<F>> {
    match transform {
        AugmentationType::RandomCrop {
            height,
            width,
            padding,
        } => apply_random_crop(images, *height, *width, *padding, rng),
        AugmentationType::RandomHorizontalFlip { probability } => {
            apply_random_hflip(images, *probability, rng)
        }
        AugmentationType::RandomVerticalFlip { probability } => {
            apply_random_vflip(images, *probability, rng)
        }
        AugmentationType::RandomRotation { max_degrees } => {
            apply_random_rotation(images, *max_degrees, rng)
        }
        AugmentationType::ColorJitter {
            brightness,
            contrast,
            saturation,
        } => apply_color_jitter(images, *brightness, *contrast, *saturation, rng),
        AugmentationType::Normalize { mean, std } => apply_normalize(images, mean, std),
        AugmentationType::RandomErasing {
            probability,
            scale,
            ratio,
        } => apply_random_erasing(images, *probability, *scale, *ratio, rng),
        AugmentationType::GaussianNoise { std } => apply_gaussian_noise(images, *std, rng),
        AugmentationType::Cutout { n_holes, length } => {
            apply_cutout(images, *n_holes, *length, rng)
        }
        // Mixup and CutMix are batch+label transforms; applying them here
        // without labels is a no-op.  Users should call the free functions.
        AugmentationType::Mixup { .. } | AugmentationType::CutMix { .. } => Ok(images.clone()),
        AugmentationType::RandomAffine {
            degrees,
            translate,
            scale,
        } => apply_random_affine(images, *degrees, *translate, *scale, rng),
    }
}

// ---------------------------------------------------------------------------
// Transform implementations
// ---------------------------------------------------------------------------

/// Zero-pad then randomly crop each sample.
fn apply_random_crop<F: Float + FromPrimitive + NumAssign + Debug>(
    images: &Array4<F>,
    crop_h: usize,
    crop_w: usize,
    padding: usize,
    rng: &mut SmallRng,
) -> Result<Array4<F>> {
    let (n, c, h, w) = (
        images.shape()[0],
        images.shape()[1],
        images.shape()[2],
        images.shape()[3],
    );
    let ph = h + 2 * padding;
    let pw = w + 2 * padding;

    if crop_h > ph || crop_w > pw {
        return Err(NeuralError::InvalidArgument(format!(
            "Crop size ({crop_h},{crop_w}) exceeds padded image size ({ph},{pw})"
        )));
    }

    let mut out = Array4::<F>::zeros((n, c, crop_h, crop_w));

    for i in 0..n {
        // Build padded image (zero-padded)
        let mut padded = Array3::<F>::zeros((c, ph, pw));
        padded
            .slice_mut(s![.., padding..padding + h, padding..padding + w])
            .assign(&images.slice(s![i, .., .., ..]));

        let top = if ph > crop_h {
            rng.random_range(0..=(ph - crop_h))
        } else {
            0
        };
        let left = if pw > crop_w {
            rng.random_range(0..=(pw - crop_w))
        } else {
            0
        };
        out.slice_mut(s![i, .., .., ..]).assign(&padded.slice(s![
            ..,
            top..top + crop_h,
            left..left + crop_w
        ]));
    }
    Ok(out)
}

/// Flip each sample horizontally with the given probability.
fn apply_random_hflip<F: Float + FromPrimitive + NumAssign + Debug>(
    images: &Array4<F>,
    prob: f64,
    rng: &mut SmallRng,
) -> Result<Array4<F>> {
    let n = images.shape()[0];
    let mut out = images.clone();
    for i in 0..n {
        if rng.random::<f64>() < prob {
            // Reverse the width axis: create a reversed view and assign back.
            let mut view = out.slice(s![i, .., .., ..]).to_owned();
            view.invert_axis(Axis(2));
            out.slice_mut(s![i, .., .., ..]).assign(&view);
        }
    }
    Ok(out)
}

/// Flip each sample vertically with the given probability.
fn apply_random_vflip<F: Float + FromPrimitive + NumAssign + Debug>(
    images: &Array4<F>,
    prob: f64,
    rng: &mut SmallRng,
) -> Result<Array4<F>> {
    let n = images.shape()[0];
    let mut out = images.clone();
    for i in 0..n {
        if rng.random::<f64>() < prob {
            // Reverse the height axis: create a reversed view and assign back.
            let mut view = out.slice(s![i, .., .., ..]).to_owned();
            view.invert_axis(Axis(1));
            out.slice_mut(s![i, .., .., ..]).assign(&view);
        }
    }
    Ok(out)
}

/// Approximate rotation by nearest-neighbour resampling.
fn apply_random_rotation<F: Float + FromPrimitive + NumAssign + Debug>(
    images: &Array4<F>,
    max_degrees: f64,
    rng: &mut SmallRng,
) -> Result<Array4<F>> {
    let (n, c, h, w) = (
        images.shape()[0],
        images.shape()[1],
        images.shape()[2],
        images.shape()[3],
    );
    let mut out = Array4::<F>::zeros((n, c, h, w));
    let cx = (w as f64 - 1.0) / 2.0;
    let cy = (h as f64 - 1.0) / 2.0;

    for i in 0..n {
        let angle_deg = rng.random_range(-max_degrees..=max_degrees);
        let angle_rad = angle_deg * std::f64::consts::PI / 180.0;
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();

        for y_dst in 0..h {
            for x_dst in 0..w {
                let dx = x_dst as f64 - cx;
                let dy = y_dst as f64 - cy;
                let x_src = cos_a * dx + sin_a * dy + cx;
                let y_src = -sin_a * dx + cos_a * dy + cy;
                let xi = x_src.round() as isize;
                let yi = y_src.round() as isize;
                if xi >= 0 && xi < w as isize && yi >= 0 && yi < h as isize {
                    for ch in 0..c {
                        out[[i, ch, y_dst, x_dst]] = images[[i, ch, yi as usize, xi as usize]];
                    }
                }
                // else: remains zero (border fill)
            }
        }
    }
    Ok(out)
}

/// Per-sample colour jitter (brightness, contrast, saturation).
fn apply_color_jitter<F: Float + FromPrimitive + NumAssign + Debug>(
    images: &Array4<F>,
    brightness: f64,
    contrast: f64,
    saturation: f64,
    rng: &mut SmallRng,
) -> Result<Array4<F>> {
    let n = images.shape()[0];
    let mut out = images.clone();

    for i in 0..n {
        // Brightness
        if brightness > 0.0 {
            let factor = 1.0 + rng.random_range(-brightness..=brightness);
            let f_factor =
                F::from(factor).ok_or_else(|| NeuralError::ComputationError("f64->F".into()))?;
            out.slice_mut(s![i, .., .., ..])
                .mapv_inplace(|v| v * f_factor);
        }
        // Contrast
        if contrast > 0.0 {
            let factor = 1.0 + rng.random_range(-contrast..=contrast);
            let f_factor =
                F::from(factor).ok_or_else(|| NeuralError::ComputationError("f64->F".into()))?;
            // Compute mean intensity for this sample
            let sample = out.slice(s![i, .., .., ..]);
            let n_elem = sample.len();
            let mut sum = F::zero();
            for &v in sample.iter() {
                sum += v;
            }
            let mean = if n_elem > 0 {
                sum / F::from(n_elem)
                    .ok_or_else(|| NeuralError::ComputationError("usize->F".into()))?
            } else {
                F::zero()
            };
            out.slice_mut(s![i, .., .., ..])
                .mapv_inplace(|v| (v - mean) * f_factor + mean);
        }
        // Saturation (approximate: blend with per-channel grey)
        if saturation > 0.0 {
            let factor = 1.0 + rng.random_range(-saturation..=saturation);
            let f_factor =
                F::from(factor).ok_or_else(|| NeuralError::ComputationError("f64->F".into()))?;
            let c = images.shape()[1];
            if c >= 3 {
                // Compute luminance per pixel
                let (h, w) = (images.shape()[2], images.shape()[3]);
                let coeffs: [f64; 3] = [0.2989, 0.5870, 0.1140];
                for y in 0..h {
                    for x in 0..w {
                        let mut lum = F::zero();
                        for (ch, &coeff) in coeffs.iter().enumerate() {
                            let fc = F::from(coeff)
                                .ok_or_else(|| NeuralError::ComputationError("coeff->F".into()))?;
                            lum += out[[i, ch, y, x]] * fc;
                        }
                        for ch in 0..3 {
                            let v = out[[i, ch, y, x]];
                            out[[i, ch, y, x]] = lum + (v - lum) * f_factor;
                        }
                    }
                }
            }
        }
        // Clamp [0, 1]
        out.slice_mut(s![i, .., .., ..])
            .mapv_inplace(|v| v.max(F::zero()).min(F::one()));
    }
    Ok(out)
}

/// Channel-wise normalisation: `(pixel - mean) / std`.
fn apply_normalize<F: Float + FromPrimitive + NumAssign + Debug>(
    images: &Array4<F>,
    mean: &[f64],
    std: &[f64],
) -> Result<Array4<F>> {
    let c = images.shape()[1];
    if mean.len() < c || std.len() < c {
        return Err(NeuralError::InvalidArgument(format!(
            "Normalize: expected {} mean/std values, got {}/{}",
            c,
            mean.len(),
            std.len()
        )));
    }
    let mut out = images.clone();
    for ch in 0..c {
        let m = F::from(mean[ch]).ok_or_else(|| NeuralError::ComputationError("mean->F".into()))?;
        let s = F::from(std[ch]).ok_or_else(|| NeuralError::ComputationError("std->F".into()))?;
        if s == F::zero() {
            return Err(NeuralError::InvalidArgument(
                "Normalize: std must be non-zero".into(),
            ));
        }
        out.slice_mut(s![.., ch, .., ..])
            .mapv_inplace(|v| (v - m) / s);
    }
    Ok(out)
}

/// Randomly erase a rectangular patch per sample.
fn apply_random_erasing<F: Float + FromPrimitive + NumAssign + Debug>(
    images: &Array4<F>,
    prob: f64,
    scale: (f64, f64),
    ratio: (f64, f64),
    rng: &mut SmallRng,
) -> Result<Array4<F>> {
    let (n, _c, h, w) = (
        images.shape()[0],
        images.shape()[1],
        images.shape()[2],
        images.shape()[3],
    );
    let mut out = images.clone();
    let total_area = (h * w) as f64;

    for i in 0..n {
        if rng.random::<f64>() >= prob {
            continue;
        }
        let area_frac = rng.random_range(scale.0..=scale.1);
        let target_area = total_area * area_frac;
        let log_ratio_lo = ratio.0.ln();
        let log_ratio_hi = ratio.1.ln();
        let aspect = (rng.random_range(log_ratio_lo..=log_ratio_hi)).exp();
        let eh = ((target_area * aspect).sqrt() as usize).min(h);
        let ew = ((target_area / aspect).sqrt() as usize).min(w);
        if eh == 0 || ew == 0 {
            continue;
        }
        let top = rng.random_range(0..=h.saturating_sub(eh));
        let left = rng.random_range(0..=w.saturating_sub(ew));
        out.slice_mut(s![i, .., top..top + eh, left..left + ew])
            .fill(F::zero());
    }
    Ok(out)
}

/// Additive Gaussian noise.
fn apply_gaussian_noise<F: Float + FromPrimitive + NumAssign + Debug>(
    images: &Array4<F>,
    noise_std: f64,
    rng: &mut SmallRng,
) -> Result<Array4<F>> {
    let dist = Normal::new(0.0, noise_std)
        .map_err(|e| NeuralError::InvalidArgument(format!("Normal distribution: {e}")))?;
    let out = images.mapv(|v| {
        let noise_val: f64 = dist.sample(rng);
        let f_noise = F::from(noise_val).unwrap_or(F::zero());
        v + f_noise
    });
    Ok(out)
}

/// Cutout: zero-fill `n_holes` square patches of side `length`.
fn apply_cutout<F: Float + FromPrimitive + NumAssign + Debug>(
    images: &Array4<F>,
    n_holes: usize,
    length: usize,
    rng: &mut SmallRng,
) -> Result<Array4<F>> {
    let (n, _c, h, w) = (
        images.shape()[0],
        images.shape()[1],
        images.shape()[2],
        images.shape()[3],
    );
    let mut out = images.clone();
    let half = length / 2;

    for i in 0..n {
        for _ in 0..n_holes {
            let cy = rng.random_range(0..h);
            let cx_val = rng.random_range(0..w);
            let y1 = cy.saturating_sub(half);
            let x1 = cx_val.saturating_sub(half);
            let y2 = (cy + half).min(h);
            let x2 = (cx_val + half).min(w);
            out.slice_mut(s![i, .., y1..y2, x1..x2]).fill(F::zero());
        }
    }
    Ok(out)
}

/// Random affine: rotation + optional translate + optional scale.
/// Uses nearest-neighbour resampling.
fn apply_random_affine<F: Float + FromPrimitive + NumAssign + Debug>(
    images: &Array4<F>,
    degrees: f64,
    translate: Option<(f64, f64)>,
    scale: Option<(f64, f64)>,
    rng: &mut SmallRng,
) -> Result<Array4<F>> {
    let (n, c, h, w) = (
        images.shape()[0],
        images.shape()[1],
        images.shape()[2],
        images.shape()[3],
    );
    let mut out = Array4::<F>::zeros((n, c, h, w));
    let cx = (w as f64 - 1.0) / 2.0;
    let cy = (h as f64 - 1.0) / 2.0;

    for i in 0..n {
        let angle_deg = rng.random_range(-degrees..=degrees);
        let angle_rad = angle_deg * std::f64::consts::PI / 180.0;
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();

        let (tx, ty) = match translate {
            Some((max_dx, max_dy)) => {
                let dx = rng.random_range(-max_dx..=max_dx) * w as f64;
                let dy = rng.random_range(-max_dy..=max_dy) * h as f64;
                (dx, dy)
            }
            None => (0.0, 0.0),
        };

        let s_factor = match scale {
            Some((lo, hi)) => rng.random_range(lo..=hi),
            None => 1.0,
        };
        let inv_s = if s_factor.abs() < 1e-12 {
            1.0
        } else {
            1.0 / s_factor
        };

        for y_dst in 0..h {
            for x_dst in 0..w {
                let dx = x_dst as f64 - cx - tx;
                let dy = y_dst as f64 - cy - ty;
                let x_src = (cos_a * dx + sin_a * dy) * inv_s + cx;
                let y_src = (-sin_a * dx + cos_a * dy) * inv_s + cy;
                let xi = x_src.round() as isize;
                let yi = y_src.round() as isize;
                if xi >= 0 && xi < w as isize && yi >= 0 && yi < h as isize {
                    for ch in 0..c {
                        out[[i, ch, y_dst, x_dst]] = images[[i, ch, yi as usize, xi as usize]];
                    }
                }
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Mixup / CutMix (batch + label transforms)
// ---------------------------------------------------------------------------

/// Sample from Beta(alpha, alpha), clamped to [0,1].
fn sample_beta(alpha: f64, rng: &mut SmallRng) -> Result<f64> {
    if alpha <= 0.0 {
        return Ok(0.5);
    }
    let dist = RandBeta::new(alpha, alpha)
        .map_err(|e| NeuralError::InvalidArgument(format!("Beta distribution: {e}")))?;
    let lambda: f64 = dist.sample(rng);
    Ok(lambda.clamp(0.0, 1.0))
}

/// Apply Mixup to a batch of images and one-hot labels.
///
/// Returns `(mixed_images, mixed_labels)`.
///
/// `images`: (N, C, H, W), `labels`: (N, num_classes).
pub fn apply_mixup<F: Float + FromPrimitive + NumAssign + ScalarOperand + Debug>(
    images: &Array4<F>,
    labels: &Array2<F>,
    alpha: f64,
    seed: Option<u64>,
) -> Result<(Array4<F>, Array2<F>)> {
    let n = images.shape()[0];
    if n < 2 {
        return Ok((images.clone(), labels.clone()));
    }
    if labels.shape()[0] != n {
        return Err(NeuralError::ShapeMismatch(format!(
            "apply_mixup: images batch {} != labels batch {}",
            n,
            labels.shape()[0]
        )));
    }
    let mut rng = match seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_rng(&mut scirs2_core::random::thread_rng()),
    };

    let lambda = sample_beta(alpha, &mut rng)?;
    let lam_f = F::from(lambda).ok_or_else(|| NeuralError::ComputationError("lambda->F".into()))?;
    let one_minus =
        F::from(1.0 - lambda).ok_or_else(|| NeuralError::ComputationError("1-lambda->F".into()))?;

    // Shuffle indices
    let mut indices: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        indices.swap(i, j);
    }

    let mut mixed_images = images.clone();
    let mut mixed_labels = labels.clone();

    for (i, &j) in indices.iter().enumerate() {
        // Mix images
        let xi = images.slice(s![i, .., .., ..]);
        let xj = images.slice(s![j, .., .., ..]);
        let mixed = &xi * lam_f + &xj * one_minus;
        mixed_images.slice_mut(s![i, .., .., ..]).assign(&mixed);

        // Mix labels
        let yi = labels.slice(s![i, ..]);
        let yj = labels.slice(s![j, ..]);
        let ml = &yi * lam_f + &yj * one_minus;
        mixed_labels.slice_mut(s![i, ..]).assign(&ml);
    }

    Ok((mixed_images, mixed_labels))
}

/// Apply CutMix to a batch of images and one-hot labels.
///
/// A random rectangular patch from a shuffled partner is pasted onto each
/// image.  Labels are mixed proportionally to the patch area.
///
/// `images`: (N, C, H, W), `labels`: (N, num_classes).
pub fn apply_cutmix<F: Float + FromPrimitive + NumAssign + ScalarOperand + Debug>(
    images: &Array4<F>,
    labels: &Array2<F>,
    alpha: f64,
    seed: Option<u64>,
) -> Result<(Array4<F>, Array2<F>)> {
    let (n, _c, h, w) = (
        images.shape()[0],
        images.shape()[1],
        images.shape()[2],
        images.shape()[3],
    );
    if n < 2 {
        return Ok((images.clone(), labels.clone()));
    }
    if labels.shape()[0] != n {
        return Err(NeuralError::ShapeMismatch(format!(
            "apply_cutmix: images batch {} != labels batch {}",
            n,
            labels.shape()[0]
        )));
    }
    let mut rng = match seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_rng(&mut scirs2_core::random::thread_rng()),
    };

    let lambda = sample_beta(alpha, &mut rng)?;
    let cut_ratio = (1.0 - lambda).sqrt();
    let cut_h = ((h as f64 * cut_ratio) as usize).max(1).min(h);
    let cut_w = ((w as f64 * cut_ratio) as usize).max(1).min(w);

    // Shuffle indices
    let mut indices: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        indices.swap(i, j);
    }

    let mut mixed_images = images.clone();
    let mut mixed_labels = labels.clone();

    for (i, &j) in indices.iter().enumerate() {
        let top = rng.random_range(0..=h.saturating_sub(cut_h));
        let left = rng.random_range(0..=w.saturating_sub(cut_w));

        // Paste patch from j onto i
        let patch = images
            .slice(s![j, .., top..top + cut_h, left..left + cut_w])
            .to_owned();
        mixed_images
            .slice_mut(s![i, .., top..top + cut_h, left..left + cut_w])
            .assign(&patch);

        // Label mixing: proportion of original image kept
        let actual_lambda = 1.0 - (cut_h * cut_w) as f64 / (h * w) as f64;
        let lam_f = F::from(actual_lambda)
            .ok_or_else(|| NeuralError::ComputationError("lambda->F".into()))?;
        let one_minus = F::from(1.0 - actual_lambda)
            .ok_or_else(|| NeuralError::ComputationError("1-lambda->F".into()))?;

        let yi = labels.slice(s![i, ..]);
        let yj = labels.slice(s![j, ..]);
        let ml = &yi * lam_f + &yj * one_minus;
        mixed_labels.slice_mut(s![i, ..]).assign(&ml);
    }

    Ok((mixed_images, mixed_labels))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array4;

    fn sample_batch() -> Array4<f64> {
        Array4::from_shape_fn((4, 3, 8, 8), |(n, c, h, w)| {
            ((n * 1000 + c * 100 + h * 10 + w) as f64) / 2000.0
        })
    }

    fn sample_labels() -> Array2<f64> {
        let mut labels = Array2::<f64>::zeros((4, 5));
        for i in 0..4 {
            labels[[i, i % 5]] = 1.0;
        }
        labels
    }

    // --- AugmentationPipeline construction ---

    #[test]
    fn test_empty_pipeline() {
        let pipe = AugmentationPipeline::<f64>::new();
        assert!(pipe.transforms.is_empty());
        assert!(pipe.seed.is_none());
    }

    #[test]
    fn test_builder_chain() {
        let pipe = AugmentationPipeline::<f64>::new()
            .with_seed(42)
            .add(AugmentationType::RandomHorizontalFlip { probability: 0.5 })
            .add(AugmentationType::GaussianNoise { std: 0.01 });
        assert_eq!(pipe.transforms.len(), 2);
        assert_eq!(pipe.seed, Some(42));
    }

    #[test]
    fn test_default_pipeline() {
        let pipe = AugmentationPipeline::<f64>::default();
        assert!(pipe.transforms.is_empty());
    }

    // --- Predefined pipelines ---

    #[test]
    fn test_cifar10_train_pipeline() {
        let pipe = AugmentationPipeline::<f64>::cifar10_train();
        assert_eq!(pipe.transforms.len(), 3);
        // Check types
        assert!(matches!(
            pipe.transforms[0],
            AugmentationType::RandomCrop { .. }
        ));
        assert!(matches!(
            pipe.transforms[1],
            AugmentationType::RandomHorizontalFlip { .. }
        ));
        assert!(matches!(
            pipe.transforms[2],
            AugmentationType::Normalize { .. }
        ));
    }

    #[test]
    fn test_imagenet_train_pipeline() {
        let pipe = AugmentationPipeline::<f64>::imagenet_train();
        assert_eq!(pipe.transforms.len(), 4);
    }

    #[test]
    fn test_basic_eval_pipeline() {
        let pipe = AugmentationPipeline::<f64>::basic_eval();
        assert_eq!(pipe.transforms.len(), 1);
        assert!(matches!(
            pipe.transforms[0],
            AugmentationType::Normalize { .. }
        ));
    }

    // --- Individual transforms ---

    #[test]
    fn test_random_hflip_always() {
        let images = Array4::from_shape_fn((1, 1, 2, 3), |(_, _, _, x)| x as f64);
        let pipe = AugmentationPipeline::<f64>::new()
            .with_seed(0)
            .add(AugmentationType::RandomHorizontalFlip { probability: 1.0 });
        let result = pipe.apply(&images).expect("apply failed");
        // Original row: [0, 1, 2] -> flipped: [2, 1, 0]
        assert_eq!(result[[0, 0, 0, 0]], 2.0);
        assert_eq!(result[[0, 0, 0, 2]], 0.0);
    }

    #[test]
    fn test_random_hflip_never() {
        let images = sample_batch();
        let pipe = AugmentationPipeline::<f64>::new()
            .with_seed(0)
            .add(AugmentationType::RandomHorizontalFlip { probability: 0.0 });
        let result = pipe.apply(&images).expect("apply failed");
        assert_eq!(result, images);
    }

    #[test]
    fn test_random_vflip_always() {
        let images = Array4::from_shape_fn((1, 1, 3, 2), |(_, _, y, _)| y as f64);
        let pipe = AugmentationPipeline::<f64>::new()
            .with_seed(0)
            .add(AugmentationType::RandomVerticalFlip { probability: 1.0 });
        let result = pipe.apply(&images).expect("apply failed");
        assert_eq!(result[[0, 0, 0, 0]], 2.0);
        assert_eq!(result[[0, 0, 2, 0]], 0.0);
    }

    #[test]
    fn test_random_crop_preserves_shape() {
        let images = Array4::<f64>::ones((2, 3, 8, 8));
        let pipe =
            AugmentationPipeline::<f64>::new()
                .with_seed(1)
                .add(AugmentationType::RandomCrop {
                    height: 6,
                    width: 6,
                    padding: 2,
                });
        let result = pipe.apply(&images).expect("apply failed");
        assert_eq!(result.shape(), &[2, 3, 6, 6]);
    }

    #[test]
    fn test_random_crop_with_zero_padding() {
        let images = Array4::<f64>::ones((1, 1, 4, 4));
        let pipe =
            AugmentationPipeline::<f64>::new()
                .with_seed(99)
                .add(AugmentationType::RandomCrop {
                    height: 2,
                    width: 2,
                    padding: 0,
                });
        let result = pipe.apply(&images).expect("apply failed");
        assert_eq!(result.shape(), &[1, 1, 2, 2]);
        // All ones, so cropped region should also be all ones
        for v in result.iter() {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_random_crop_too_large() {
        let images = Array4::<f64>::ones((1, 1, 4, 4));
        let pipe = AugmentationPipeline::<f64>::new().add(AugmentationType::RandomCrop {
            height: 10,
            width: 10,
            padding: 0,
        });
        let result = pipe.apply(&images);
        assert!(result.is_err());
    }

    #[test]
    fn test_normalize_channel_wise() {
        let images = Array4::from_elem((1, 3, 2, 2), 0.5_f64);
        let pipe = AugmentationPipeline::<f64>::new().add(AugmentationType::Normalize {
            mean: vec![0.5, 0.5, 0.5],
            std: vec![0.5, 0.5, 0.5],
        });
        let result = pipe.apply(&images).expect("apply failed");
        // (0.5 - 0.5)/0.5 = 0.0
        for v in result.iter() {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn test_normalize_zero_std_error() {
        let images = Array4::from_elem((1, 1, 2, 2), 1.0_f64);
        let pipe = AugmentationPipeline::<f64>::new().add(AugmentationType::Normalize {
            mean: vec![0.0],
            std: vec![0.0],
        });
        assert!(pipe.apply(&images).is_err());
    }

    #[test]
    fn test_gaussian_noise_changes_values() {
        let images = Array4::<f64>::zeros((2, 1, 4, 4));
        let pipe = AugmentationPipeline::<f64>::new()
            .with_seed(7)
            .add(AugmentationType::GaussianNoise { std: 1.0 });
        let result = pipe.apply(&images).expect("apply failed");
        // At least some values should be non-zero with high probability
        let any_nonzero = result.iter().any(|&v| v.abs() > 1e-10);
        assert!(
            any_nonzero,
            "Gaussian noise should alter at least some pixels"
        );
    }

    #[test]
    fn test_random_erasing_shape_preserved() {
        let images = sample_batch();
        let pipe =
            AugmentationPipeline::<f64>::new()
                .with_seed(5)
                .add(AugmentationType::RandomErasing {
                    probability: 1.0,
                    scale: (0.02, 0.33),
                    ratio: (0.3, 3.3),
                });
        let result = pipe.apply(&images).expect("apply failed");
        assert_eq!(result.shape(), images.shape());
    }

    #[test]
    fn test_cutout_zeros_patches() {
        let images = Array4::from_elem((1, 1, 8, 8), 1.0_f64);
        let pipe = AugmentationPipeline::<f64>::new()
            .with_seed(3)
            .add(AugmentationType::Cutout {
                n_holes: 1,
                length: 4,
            });
        let result = pipe.apply(&images).expect("apply failed");
        let n_zeros = result.iter().filter(|&&v| v.abs() < 1e-12).count();
        assert!(n_zeros > 0, "Cutout should zero out some pixels");
    }

    #[test]
    fn test_random_rotation_identity() {
        // 0-degree rotation should preserve image
        let images = sample_batch();
        let mut rng = SmallRng::seed_from_u64(0);
        let result = apply_random_rotation(&images, 0.0, &mut rng).expect("rotation failed");
        assert_eq!(result, images);
    }

    #[test]
    fn test_random_rotation_preserves_shape() {
        let images = sample_batch();
        let pipe = AugmentationPipeline::<f64>::new()
            .with_seed(10)
            .add(AugmentationType::RandomRotation { max_degrees: 30.0 });
        let result = pipe.apply(&images).expect("apply failed");
        assert_eq!(result.shape(), images.shape());
    }

    #[test]
    fn test_color_jitter_clamps() {
        let images = Array4::from_elem((1, 3, 4, 4), 0.9_f64);
        let pipe =
            AugmentationPipeline::<f64>::new()
                .with_seed(0)
                .add(AugmentationType::ColorJitter {
                    brightness: 0.5,
                    contrast: 0.5,
                    saturation: 0.5,
                });
        let result = pipe.apply(&images).expect("apply failed");
        for &v in result.iter() {
            assert!(
                (0.0 - 1e-12..=1.0 + 1e-12).contains(&v),
                "ColorJitter should clamp to [0,1], got {v}"
            );
        }
    }

    #[test]
    fn test_random_affine_preserves_shape() {
        let images = sample_batch();
        let pipe =
            AugmentationPipeline::<f64>::new()
                .with_seed(22)
                .add(AugmentationType::RandomAffine {
                    degrees: 15.0,
                    translate: Some((0.1, 0.1)),
                    scale: Some((0.9, 1.1)),
                });
        let result = pipe.apply(&images).expect("apply failed");
        assert_eq!(result.shape(), images.shape());
    }

    // --- apply_single ---

    #[test]
    fn test_apply_single() {
        let image = Array3::from_shape_fn((3, 8, 8), |(c, h, w)| {
            (c * 100 + h * 10 + w) as f64 / 1000.0
        });
        let pipe = AugmentationPipeline::<f64>::new()
            .with_seed(0)
            .add(AugmentationType::RandomHorizontalFlip { probability: 1.0 });
        let result = pipe.apply_single(&image).expect("apply_single failed");
        assert_eq!(result.shape(), image.shape());
    }

    // --- Mixup ---

    #[test]
    fn test_mixup_shapes() {
        let images = sample_batch();
        let labels = sample_labels();
        let (mi, ml) = apply_mixup(&images, &labels, 1.0, Some(42)).expect("mixup failed");
        assert_eq!(mi.shape(), images.shape());
        assert_eq!(ml.shape(), labels.shape());
    }

    #[test]
    fn test_mixup_small_batch() {
        let images = Array4::<f64>::ones((1, 1, 2, 2));
        let labels = Array2::from_elem((1, 3), 1.0_f64);
        let (mi, ml) = apply_mixup(&images, &labels, 1.0, Some(0)).expect("mixup failed");
        // Single sample should be returned unchanged
        assert_eq!(mi, images);
        assert_eq!(ml, labels);
    }

    #[test]
    fn test_mixup_label_shape_mismatch() {
        let images = sample_batch();
        let labels = Array2::<f64>::zeros((2, 5)); // wrong batch size
        let result = apply_mixup(&images, &labels, 1.0, None);
        assert!(result.is_err());
    }

    // --- CutMix ---

    #[test]
    fn test_cutmix_shapes() {
        let images = sample_batch();
        let labels = sample_labels();
        let (mi, ml) = apply_cutmix(&images, &labels, 1.0, Some(42)).expect("cutmix failed");
        assert_eq!(mi.shape(), images.shape());
        assert_eq!(ml.shape(), labels.shape());
    }

    #[test]
    fn test_cutmix_small_batch() {
        let images = Array4::<f64>::ones((1, 1, 4, 4));
        let labels = Array2::from_elem((1, 2), 1.0_f64);
        let (mi, ml) = apply_cutmix(&images, &labels, 1.0, Some(0)).expect("cutmix failed");
        assert_eq!(mi, images);
        assert_eq!(ml, labels);
    }

    #[test]
    fn test_cutmix_label_consistency() {
        let images = sample_batch();
        let labels = sample_labels();
        let (_, ml) = apply_cutmix(&images, &labels, 1.0, Some(99)).expect("cutmix failed");
        // Each row should still sum to ~1.0 (convex combination)
        for i in 0..ml.shape()[0] {
            let row_sum: f64 = ml.slice(s![i, ..]).iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "CutMix label row should sum to 1.0, got {row_sum}"
            );
        }
    }

    // --- Composability ---

    #[test]
    fn test_pipeline_compose_multiple() {
        let images = Array4::from_elem((2, 3, 32, 32), 0.5_f64);
        let pipe = AugmentationPipeline::<f64>::new()
            .with_seed(12)
            .add(AugmentationType::RandomCrop {
                height: 32,
                width: 32,
                padding: 4,
            })
            .add(AugmentationType::RandomHorizontalFlip { probability: 0.5 })
            .add(AugmentationType::GaussianNoise { std: 0.01 })
            .add(AugmentationType::Normalize {
                mean: vec![0.5, 0.5, 0.5],
                std: vec![0.25, 0.25, 0.25],
            });
        let result = pipe.apply(&images).expect("pipeline apply failed");
        assert_eq!(result.shape(), &[2, 3, 32, 32]);
    }

    // --- Reproducibility ---

    #[test]
    fn test_seeded_reproducibility() {
        let images = sample_batch();
        let pipe = AugmentationPipeline::<f64>::new()
            .with_seed(77)
            .add(AugmentationType::RandomHorizontalFlip { probability: 0.5 })
            .add(AugmentationType::GaussianNoise { std: 0.02 });
        let r1 = pipe.apply(&images).expect("first apply failed");
        let r2 = pipe.apply(&images).expect("second apply failed");
        assert_eq!(r1, r2, "Same seed should produce identical results");
    }

    // --- f32 support ---

    #[test]
    fn test_f32_pipeline() {
        let images = Array4::from_elem((2, 3, 8, 8), 0.5_f32);
        let pipe = AugmentationPipeline::<f32>::new()
            .with_seed(1)
            .add(AugmentationType::RandomHorizontalFlip { probability: 0.5 })
            .add(AugmentationType::Normalize {
                mean: vec![0.5, 0.5, 0.5],
                std: vec![0.5, 0.5, 0.5],
            });
        let result = pipe.apply(&images).expect("f32 pipeline failed");
        assert_eq!(result.shape(), &[2, 3, 8, 8]);
    }

    // --- Edge cases ---

    #[test]
    fn test_mixup_cutmix_in_pipeline_noop() {
        // Mixup/CutMix inside a pipeline (without labels) should be a no-op
        let images = sample_batch();
        let pipe = AugmentationPipeline::<f64>::new()
            .with_seed(0)
            .add(AugmentationType::Mixup { alpha: 1.0 })
            .add(AugmentationType::CutMix { alpha: 1.0 });
        let result = pipe.apply(&images).expect("noop pipeline failed");
        assert_eq!(result, images);
    }
}
