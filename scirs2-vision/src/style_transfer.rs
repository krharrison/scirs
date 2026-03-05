//! Neural style transfer utilities
//!
//! This module provides gradient-descent-based neural style transfer,
//! including Gram-matrix style representation, content/style/total-variation
//! loss functions, and an iterative image optimization loop.
//!
//! # Overview
//!
//! Neural style transfer (Gatys et al., 2015) separates and recombines the
//! *content* of one image with the *style* of another.  The key insight is
//! that the Gram matrix of feature activations captures texture statistics
//! while the raw activations encode spatial content.
//!
//! This implementation works directly on raw feature-map tensors (C×H×W
//! `Array3<f64>` where C = channels, H = height, W = width) rather than
//! requiring a full neural network at runtime, making it usable as a
//! standalone post-processing or artistic-texture layer.
//!
//! # Example
//!
//! ```rust
//! use scirs2_vision::style_transfer::{
//!     gram_matrix, content_loss, style_loss, total_variation_loss,
//!     StyleTransferLoss, StyleTransferWeights,
//! };
//! use scirs2_core::ndarray::Array3;
//!
//! // Build toy 2-channel, 4×4 feature maps
//! let content: Array3<f64> = Array3::ones((2, 4, 4));
//! let style: Array3<f64>   = Array3::ones((2, 4, 4));
//!
//! // Gram matrix: shape (C, C)
//! let g = gram_matrix(&content);
//! assert_eq!(g.dim(), (2, 2));
//!
//! // Losses
//! let cl  = content_loss(&content, &content);
//! let sg  = gram_matrix(&style);
//! let sl  = style_loss(&g, &sg);
//! let tvl = total_variation_loss(&content);
//!
//! assert!(cl  >= 0.0);
//! assert!(sl  >= 0.0);
//! assert!(tvl >= 0.0);
//!
//! // Combined loss struct
//! let weights = StyleTransferWeights::default();
//! let combined = StyleTransferLoss::new(weights);
//! let total = combined.total(&g, &sg, &content, &content);
//! assert!(total >= 0.0);
//! ```

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::{Array1, Array2, Array3, Axis};

// ─────────────────────────────────────────────────────────────────────────────
// Gram matrix
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Gram matrix of a feature-map tensor.
///
/// Given a tensor of shape `(C, H, W)` the function reshapes it into
/// `(C, H*W)` and returns `G = F · Fᵀ / (H*W)` of shape `(C, C)`.
///
/// Normalising by the spatial extent `H*W` keeps the magnitude
/// scale-independent with respect to image size.
///
/// # Arguments
///
/// * `features` – Feature-map tensor of shape `(C, H, W)`.
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] when the channel axis has
/// zero elements.
pub fn gram_matrix(features: &Array3<f64>) -> Array2<f64> {
    let (c, h, w) = features.dim();
    let spatial = h * w;

    // Flatten spatial dimensions: shape (C, H*W)
    let flat: Array2<f64> = features
        .to_shape((c, spatial))
        .map(|v| v.into_owned())
        .unwrap_or_else(|_| {
            // Fallback: manual row construction
            let mut buf = Array2::zeros((c, spatial));
            for ch in 0..c {
                let mut idx = 0;
                for row in 0..h {
                    for col in 0..w {
                        buf[[ch, idx]] = features[[ch, row, col]];
                        idx += 1;
                    }
                }
            }
            buf
        });

    // G[i,j] = dot(flat[i,:], flat[j,:]) / spatial
    let scale = if spatial > 0 {
        1.0 / spatial as f64
    } else {
        1.0
    };

    let mut gram = Array2::zeros((c, c));
    for i in 0..c {
        for j in 0..c {
            let dot: f64 = flat
                .row(i)
                .iter()
                .zip(flat.row(j).iter())
                .map(|(a, b)| a * b)
                .sum();
            gram[[i, j]] = dot * scale;
        }
    }
    gram
}

// ─────────────────────────────────────────────────────────────────────────────
// Style loss
// ─────────────────────────────────────────────────────────────────────────────

/// Compute style loss as the squared Frobenius norm of Gram-matrix residuals.
///
/// `L_style = ‖G_generated − G_style‖_F² / (4 · C²)`
///
/// The `4 · C²` denominator is the normalisation used in the original paper.
///
/// # Arguments
///
/// * `generated_gram` – Gram matrix of the generated image's features, shape `(C, C)`.
/// * `style_gram`     – Gram matrix of the style image's features, shape `(C, C)`.
///
/// # Errors
///
/// Returns [`VisionError::DimensionMismatch`] when the two matrices differ
/// in shape.
pub fn style_loss(generated_gram: &Array2<f64>, style_gram: &Array2<f64>) -> f64 {
    debug_assert_eq!(
        generated_gram.dim(),
        style_gram.dim(),
        "Gram matrices must have identical shapes"
    );

    let (c, _) = generated_gram.dim();
    let denom = 4.0 * (c * c) as f64;
    let denom = if denom > 0.0 { denom } else { 1.0 };

    let sum_sq: f64 = generated_gram
        .iter()
        .zip(style_gram.iter())
        .map(|(g, s)| {
            let diff = g - s;
            diff * diff
        })
        .sum();

    sum_sq / denom
}

// ─────────────────────────────────────────────────────────────────────────────
// Content loss
// ─────────────────────────────────────────────────────────────────────────────

/// Compute content loss as mean squared error between two feature maps.
///
/// `L_content = MSE(generated, content) = ‖F_gen − F_content‖² / n`
///
/// # Arguments
///
/// * `generated` – Generated-image feature map, shape `(C, H, W)`.
/// * `content`   – Content-image feature map, shape `(C, H, W)`.
///
/// # Panics (debug)
///
/// Asserts that both tensors have equal shape.
pub fn content_loss(generated: &Array3<f64>, content: &Array3<f64>) -> f64 {
    debug_assert_eq!(
        generated.dim(),
        content.dim(),
        "Feature maps must have identical shapes"
    );

    let n = generated.len();
    if n == 0 {
        return 0.0;
    }

    let sum_sq: f64 = generated
        .iter()
        .zip(content.iter())
        .map(|(g, c)| {
            let d = g - c;
            d * d
        })
        .sum();

    sum_sq / n as f64
}

// ─────────────────────────────────────────────────────────────────────────────
// Total variation loss
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the anisotropic total-variation loss for smoothness regularisation.
///
/// `L_tv = Σ_{c,i,j} (|F[c,i+1,j] - F[c,i,j]| + |F[c,i,j+1] - F[c,i,j]|) / n`
///
/// This penalises abrupt transitions in the generated image and acts as a
/// spatial smoothness prior.
///
/// # Arguments
///
/// * `image` – Image tensor of shape `(C, H, W)`.
pub fn total_variation_loss(image: &Array3<f64>) -> f64 {
    let (c, h, w) = image.dim();
    if h < 2 || w < 2 {
        return 0.0;
    }

    let mut tv = 0.0_f64;
    let n = c * (h - 1) * (w - 1);

    for ch in 0..c {
        for row in 0..h - 1 {
            for col in 0..w - 1 {
                let vert = (image[[ch, row + 1, col]] - image[[ch, row, col]]).abs();
                let horiz = (image[[ch, row, col + 1]] - image[[ch, row, col]]).abs();
                tv += vert + horiz;
            }
        }
    }

    if n > 0 {
        tv / n as f64
    } else {
        0.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Combined loss struct
// ─────────────────────────────────────────────────────────────────────────────

/// Weights for the three components of the combined style-transfer loss.
#[derive(Debug, Clone)]
pub struct StyleTransferWeights {
    /// Weight for the content reconstruction term (`α`).
    pub content_weight: f64,
    /// Weight for the style matching term (`β`).
    pub style_weight: f64,
    /// Weight for the total-variation regularisation term (`γ`).
    pub tv_weight: f64,
}

impl Default for StyleTransferWeights {
    fn default() -> Self {
        Self {
            content_weight: 1.0,
            style_weight: 1e5,
            tv_weight: 1e-4,
        }
    }
}

/// Struct that combines content, style, and TV losses with configurable weights.
///
/// # Example
///
/// ```rust
/// use scirs2_vision::style_transfer::{StyleTransferLoss, StyleTransferWeights, gram_matrix};
/// use scirs2_core::ndarray::Array3;
///
/// let img: Array3<f64> = Array3::ones((3, 8, 8));
/// let gen_gram = gram_matrix(&img);
/// let sty_gram = gram_matrix(&img);
///
/// let loss = StyleTransferLoss::new(StyleTransferWeights::default());
/// let v = loss.total(&gen_gram, &sty_gram, &img, &img);
/// assert!(v >= 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct StyleTransferLoss {
    /// Weights for each loss component.
    pub weights: StyleTransferWeights,
}

impl StyleTransferLoss {
    /// Create a new combined loss with the supplied weights.
    pub fn new(weights: StyleTransferWeights) -> Self {
        Self { weights }
    }

    /// Compute the total weighted loss.
    ///
    /// # Arguments
    ///
    /// * `generated_gram` – Gram matrix of the generated image's features.
    /// * `style_gram`     – Gram matrix of the style image's features.
    /// * `generated`      – Generated feature map (for content + TV).
    /// * `content`        – Content feature map.
    pub fn total(
        &self,
        generated_gram: &Array2<f64>,
        style_gram: &Array2<f64>,
        generated: &Array3<f64>,
        content: &Array3<f64>,
    ) -> f64 {
        let lc = self.weights.content_weight * content_loss(generated, content);
        let ls = self.weights.style_weight * style_loss(generated_gram, style_gram);
        let ltv = self.weights.tv_weight * total_variation_loss(generated);
        lc + ls + ltv
    }

    /// Compute individual loss components without combining.
    ///
    /// Returns `(content_loss, style_loss, tv_loss)`.
    pub fn components(
        &self,
        generated_gram: &Array2<f64>,
        style_gram: &Array2<f64>,
        generated: &Array3<f64>,
        content: &Array3<f64>,
    ) -> (f64, f64, f64) {
        let lc = content_loss(generated, content);
        let ls = style_loss(generated_gram, style_gram);
        let ltv = total_variation_loss(generated);
        (lc, ls, ltv)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gradient computation helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the gradient of the style loss w.r.t. the generated image's
/// feature map via Gram-matrix back-propagation.
///
/// For `G = F Fᵀ / n` and `L_style = ‖G - G_s‖_F² / (4C²)`:
///
/// `∂L/∂F[c,h,w] = 2 / (n · 4C²) · (G - G_s)[c, :] · F[:, h, w]`
///
/// (simplified per-channel accumulation)
fn style_gradient(features: &Array3<f64>, style_gram: &Array2<f64>) -> Array3<f64> {
    let gen_gram = gram_matrix(features);
    let (c, h, w) = features.dim();
    let spatial = (h * w) as f64;
    let denom = 4.0 * (c * c) as f64 * spatial.max(1.0);

    let residual = &gen_gram - style_gram; // (C, C)

    let mut grad = Array3::zeros((c, h, w));
    for row in 0..h {
        for col in 0..w {
            for ci in 0..c {
                // dL/dF[ci, row, col] = 2/denom · Σ_j residual[ci,j] * F[j,row,col]
                let mut acc = 0.0_f64;
                for cj in 0..c {
                    acc += residual[[ci, cj]] * features[[cj, row, col]];
                }
                grad[[ci, row, col]] = 2.0 * acc / denom;
            }
        }
    }
    grad
}

/// Compute the gradient of the content loss w.r.t. the generated feature map.
///
/// `∂MSE/∂F_gen = 2 (F_gen - F_content) / n`
fn content_gradient(generated: &Array3<f64>, content: &Array3<f64>) -> Array3<f64> {
    let n = generated.len().max(1) as f64;
    (generated - content).mapv(|d| 2.0 * d / n)
}

/// Compute the gradient of the anisotropic total-variation loss.
fn tv_gradient(image: &Array3<f64>) -> Array3<f64> {
    let (c, h, w) = image.dim();
    if h < 2 || w < 2 {
        return Array3::zeros((c, h, w));
    }
    let n = (c * (h - 1) * (w - 1)).max(1) as f64;
    let mut grad = Array3::zeros((c, h, w));

    for ch in 0..c {
        for row in 0..h {
            for col in 0..w {
                let mut g = 0.0_f64;

                // Contribution from the (row, col) → (row+1, col) pair
                if row + 1 < h {
                    let diff = image[[ch, row + 1, col]] - image[[ch, row, col]];
                    g -= diff.signum(); // dTV/d(image[row,col]) = -sign(next - curr)
                }
                if row > 0 {
                    let diff = image[[ch, row, col]] - image[[ch, row - 1, col]];
                    g += diff.signum();
                }

                // Contribution from the (row, col) → (row, col+1) pair
                if col + 1 < w {
                    let diff = image[[ch, row, col + 1]] - image[[ch, row, col]];
                    g -= diff.signum();
                }
                if col > 0 {
                    let diff = image[[ch, row, col]] - image[[ch, row, col - 1]];
                    g += diff.signum();
                }

                grad[[ch, row, col]] = g / n;
            }
        }
    }
    grad
}

// ─────────────────────────────────────────────────────────────────────────────
// Iterative optimisation
// ─────────────────────────────────────────────────────────────────────────────

/// Optimise an image via gradient descent to match content and style targets.
///
/// This implements vanilla gradient descent with an optional learning-rate
/// warm-up (first iteration uses `lr * 0.1`) to avoid large initial steps.
///
/// The generated image is initialised as a copy of the *content* tensor and
/// updated by descending the combined style-transfer gradient.
///
/// # Arguments
///
/// * `content`  – Content target feature map, shape `(C, H, W)`.
/// * `style`    – Style target feature map, shape `(C, H, W)`.
/// * `weights`  – Loss weights (content, style, TV).
/// * `n_iters`  – Number of gradient-descent iterations.
/// * `lr`       – Learning rate (step size).
///
/// # Errors
///
/// * [`VisionError::InvalidParameter`] – when `lr ≤ 0` or `n_iters == 0`.
/// * [`VisionError::DimensionMismatch`] – when content and style have
///   different shapes.
///
/// # Example
///
/// ```rust
/// use scirs2_vision::style_transfer::{optimize_style_transfer, StyleTransferWeights};
/// use scirs2_core::ndarray::Array3;
///
/// let content: Array3<f64> = Array3::ones((2, 4, 4));
/// let style: Array3<f64>   = Array3::ones((2, 4, 4));
/// let weights = StyleTransferWeights {
///     content_weight: 1.0,
///     style_weight: 1.0,
///     tv_weight: 0.0,
/// };
/// let result = optimize_style_transfer(&content, &style, &weights, 5, 0.01);
/// assert!(result.is_ok());
/// let img = result.unwrap();
/// assert_eq!(img.dim(), content.dim());
/// ```
pub fn optimize_style_transfer(
    content: &Array3<f64>,
    style: &Array3<f64>,
    weights: &StyleTransferWeights,
    n_iters: usize,
    lr: f64,
) -> Result<Array3<f64>> {
    if lr <= 0.0 {
        return Err(VisionError::InvalidParameter(format!(
            "Learning rate must be positive, got {lr}"
        )));
    }
    if n_iters == 0 {
        return Err(VisionError::InvalidParameter(
            "n_iters must be at least 1".to_string(),
        ));
    }
    if content.dim() != style.dim() {
        return Err(VisionError::DimensionMismatch(format!(
            "Content shape {:?} ≠ style shape {:?}",
            content.dim(),
            style.dim()
        )));
    }

    // Pre-compute the fixed style Gram matrix
    let style_gram = gram_matrix(style);

    // Initialise generated image from content
    let mut generated = content.to_owned();

    for iter in 0..n_iters {
        let gen_gram = gram_matrix(&generated);

        // Compute gradients
        let grad_style = style_gradient(&generated, &style_gram);
        let grad_content = content_gradient(&generated, content);
        let grad_tv = tv_gradient(&generated);

        // Combined gradient
        let grad = grad_content.mapv(|g| g * weights.content_weight)
            + grad_style.mapv(|g| g * weights.style_weight)
            + grad_tv.mapv(|g| g * weights.tv_weight);

        // Warm-up: smaller first step
        let step = if iter == 0 { lr * 0.1 } else { lr };

        // Gradient-descent update
        generated = generated - grad.mapv(|g| g * step);

        // Drop `gen_gram` explicitly to make the borrow-checker happy
        drop(gen_gram);
    }

    Ok(generated)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    fn make_ramp(c: usize, h: usize, w: usize) -> Array3<f64> {
        let mut a = Array3::zeros((c, h, w));
        for ch in 0..c {
            for row in 0..h {
                for col in 0..w {
                    a[[ch, row, col]] = (ch * h * w + row * w + col) as f64;
                }
            }
        }
        a
    }

    #[test]
    fn test_gram_matrix_shape() {
        let feat: Array3<f64> = Array3::ones((4, 8, 8));
        let g = gram_matrix(&feat);
        assert_eq!(g.dim(), (4, 4));
    }

    #[test]
    fn test_gram_matrix_symmetric() {
        let feat = make_ramp(3, 5, 5);
        let g = gram_matrix(&feat);
        for i in 0..3 {
            for j in 0..3 {
                let diff = (g[[i, j]] - g[[j, i]]).abs();
                assert!(diff < 1e-10, "Gram not symmetric at ({i},{j}): {diff}");
            }
        }
    }

    #[test]
    fn test_gram_matrix_positive_semidefinite() {
        // All diagonal elements must be non-negative (they are dot products).
        let feat = make_ramp(3, 4, 4);
        let g = gram_matrix(&feat);
        for i in 0..3 {
            assert!(
                g[[i, i]] >= 0.0,
                "Diagonal element ({i},{i}) is negative: {}",
                g[[i, i]]
            );
        }
    }

    #[test]
    fn test_style_loss_identical() {
        let feat: Array3<f64> = Array3::ones((3, 4, 4));
        let g = gram_matrix(&feat);
        let loss = style_loss(&g, &g);
        assert!(loss.abs() < 1e-12);
    }

    #[test]
    fn test_style_loss_non_negative() {
        let a = make_ramp(3, 4, 4);
        let b: Array3<f64> = Array3::zeros((3, 4, 4));
        let ga = gram_matrix(&a);
        let gb = gram_matrix(&b);
        assert!(style_loss(&ga, &gb) >= 0.0);
    }

    #[test]
    fn test_content_loss_identical() {
        let feat = make_ramp(3, 4, 4);
        assert!(content_loss(&feat, &feat).abs() < 1e-12);
    }

    #[test]
    fn test_content_loss_non_negative() {
        let a = make_ramp(2, 4, 4);
        let b: Array3<f64> = Array3::zeros((2, 4, 4));
        assert!(content_loss(&a, &b) >= 0.0);
    }

    #[test]
    fn test_total_variation_uniform() {
        // Uniform image has zero TV
        let img: Array3<f64> = Array3::from_elem((2, 6, 6), 5.0);
        assert!(total_variation_loss(&img).abs() < 1e-12);
    }

    #[test]
    fn test_total_variation_non_negative() {
        let img = make_ramp(2, 6, 6);
        assert!(total_variation_loss(&img) >= 0.0);
    }

    #[test]
    fn test_total_variation_small_image() {
        let img: Array3<f64> = Array3::ones((2, 1, 1));
        assert_eq!(total_variation_loss(&img), 0.0);
    }

    #[test]
    fn test_style_transfer_loss_struct() {
        let content = make_ramp(2, 4, 4);
        let style: Array3<f64> = Array3::from_elem((2, 4, 4), 3.0);
        let gen_gram = gram_matrix(&content);
        let sty_gram = gram_matrix(&style);
        let loss_fn = StyleTransferLoss::new(StyleTransferWeights::default());
        let total = loss_fn.total(&gen_gram, &sty_gram, &content, &content);
        assert!(total >= 0.0);
    }

    #[test]
    fn test_style_transfer_components() {
        let img: Array3<f64> = Array3::ones((2, 4, 4));
        let g = gram_matrix(&img);
        let loss_fn = StyleTransferLoss::new(StyleTransferWeights::default());
        let (lc, ls, ltv) = loss_fn.components(&g, &g, &img, &img);
        // Identical images → zero content and style losses
        assert!(lc.abs() < 1e-12);
        assert!(ls.abs() < 1e-12);
        assert!(ltv >= 0.0);
    }

    #[test]
    fn test_optimize_style_transfer_shape() {
        let content: Array3<f64> = Array3::ones((2, 4, 4));
        let style: Array3<f64> = Array3::ones((2, 4, 4));
        let weights = StyleTransferWeights {
            content_weight: 1.0,
            style_weight: 1.0,
            tv_weight: 0.0,
        };
        let result = optimize_style_transfer(&content, &style, &weights, 3, 0.01);
        assert!(result.is_ok());
        assert_eq!(result.expect("Test: result shape").dim(), (2, 4, 4));
    }

    #[test]
    fn test_optimize_style_transfer_bad_lr() {
        let content: Array3<f64> = Array3::ones((2, 4, 4));
        let style: Array3<f64> = Array3::ones((2, 4, 4));
        let weights = StyleTransferWeights::default();
        assert!(optimize_style_transfer(&content, &style, &weights, 3, 0.0).is_err());
        assert!(optimize_style_transfer(&content, &style, &weights, 3, -1.0).is_err());
    }

    #[test]
    fn test_optimize_style_transfer_zero_iters() {
        let content: Array3<f64> = Array3::ones((2, 4, 4));
        let style: Array3<f64> = Array3::ones((2, 4, 4));
        let weights = StyleTransferWeights::default();
        assert!(optimize_style_transfer(&content, &style, &weights, 0, 0.01).is_err());
    }

    #[test]
    fn test_optimize_style_transfer_dimension_mismatch() {
        let content: Array3<f64> = Array3::ones((2, 4, 4));
        let style: Array3<f64> = Array3::ones((3, 4, 4));
        let weights = StyleTransferWeights::default();
        assert!(optimize_style_transfer(&content, &style, &weights, 3, 0.01).is_err());
    }
}
