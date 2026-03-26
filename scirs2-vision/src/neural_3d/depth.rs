//! Depth estimation and processing for neural 3D reconstruction
//!
//! Provides:
//! - Simplified MiDaS-style relative depth estimation (encoder-decoder CNN)
//! - Depth completion: sparse-to-dense with bilateral filter propagation
//! - Depth from disparity: `D = f * B / d`
//! - Depth-to-point-cloud conversion
//! - Depth colorization (turbo/viridis colormaps)
//! - Depth edge detection (gradient-based)

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::Array2;

// ─────────────────────────────────────────────────────────────────────────────
// Relative Depth Map
// ─────────────────────────────────────────────────────────────────────────────

/// A relative (inverse) depth map produced by a monocular depth estimator.
///
/// Values represent relative inverse depth -- higher values correspond to
/// closer surfaces.  The map is scale-ambiguous by design.
#[derive(Debug, Clone)]
pub struct RelativeDepthMap {
    /// Per-pixel relative inverse depth.  Shape is `[H, W]`.
    pub data: Array2<f64>,
    /// Minimum value in the map.
    pub min_val: f64,
    /// Maximum value in the map.
    pub max_val: f64,
}

impl RelativeDepthMap {
    /// Create a new relative depth map, computing min/max from the data.
    pub fn from_data(data: Array2<f64>) -> Self {
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        for &v in data.iter() {
            if v.is_finite() {
                if v < min_val {
                    min_val = v;
                }
                if v > max_val {
                    max_val = v;
                }
            }
        }
        if !min_val.is_finite() {
            min_val = 0.0;
            max_val = 0.0;
        }
        Self {
            data,
            min_val,
            max_val,
        }
    }

    /// Normalise values to `[0, 1]` range.
    pub fn normalised(&self) -> Array2<f64> {
        let range = self.max_val - self.min_val;
        if range < 1e-12 {
            Array2::zeros(self.data.dim())
        } else {
            self.data.mapv(|v| (v - self.min_val) / range)
        }
    }

    /// Dimensions `(height, width)`.
    pub fn dim(&self) -> (usize, usize) {
        self.data.dim()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DepthEstimator (simplified MiDaS-style)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the depth estimator.
#[derive(Debug, Clone)]
pub struct DepthEstimatorConfig {
    /// Number of encoder stages (downsampling steps).  Default 3.
    pub num_encoder_stages: usize,
    /// Base number of channels in the first encoder stage.  Default 16.
    pub base_channels: usize,
}

impl Default for DepthEstimatorConfig {
    fn default() -> Self {
        Self {
            num_encoder_stages: 3,
            base_channels: 16,
        }
    }
}

/// A simplified MiDaS-style monocular relative depth estimator.
///
/// Architecture (conceptual):
/// - **Encoder**: progressive 2x downsampling convolutions with increasing channels.
/// - **Decoder**: progressive 2x upsampling with skip connections from the encoder.
/// - **Output**: single-channel relative inverse depth map.
///
/// The weights use a fixed Gaussian-like pattern for structure demonstration.
/// For production use, weights would be loaded from a trained model.
#[derive(Debug, Clone)]
pub struct DepthEstimator {
    /// Estimator configuration.
    pub config: DepthEstimatorConfig,
    /// Encoder kernel weights per stage: `[stage][ky][kx]` (3x3 kernels).
    encoder_kernels: Vec<[[f64; 3]; 3]>,
    /// Decoder kernel weights per stage.
    decoder_kernels: Vec<[[f64; 3]; 3]>,
}

impl DepthEstimator {
    /// Create a new depth estimator with deterministic initial weights.
    pub fn new(config: DepthEstimatorConfig) -> Self {
        let n = config.num_encoder_stages;

        // Simple edge-detection / smoothing kernels for demonstration
        let mut encoder_kernels = Vec::with_capacity(n);
        let mut decoder_kernels = Vec::with_capacity(n);

        for i in 0..n {
            // Encoder: progressively stronger edge kernels
            let sigma = 1.0 + i as f64 * 0.5;
            encoder_kernels.push(gaussian_kernel_3x3(sigma));
            // Decoder: smoothing kernels
            decoder_kernels.push(gaussian_kernel_3x3(sigma * 0.5 + 0.5));
        }

        Self {
            config,
            encoder_kernels,
            decoder_kernels,
        }
    }

    /// Estimate relative inverse depth from a single grayscale image.
    ///
    /// # Arguments
    ///
    /// * `image` - Grayscale image, shape `[H, W]`, values in `[0, 255]`.
    ///
    /// # Returns
    ///
    /// A [`RelativeDepthMap`] of the same spatial dimensions as the input.
    ///
    /// # Errors
    ///
    /// Returns an error if the image is empty or too small for the number of
    /// encoder stages.
    pub fn estimate(&self, image: &Array2<f64>) -> Result<RelativeDepthMap> {
        let (h, w) = image.dim();
        if h < 4 || w < 4 {
            return Err(VisionError::InvalidParameter(
                "Image must be at least 4x4 pixels".to_string(),
            ));
        }

        let min_dim = h.min(w);
        let max_stages = (min_dim as f64).log2().floor() as usize;
        if self.config.num_encoder_stages > max_stages {
            return Err(VisionError::InvalidParameter(format!(
                "Image {}x{} is too small for {} encoder stages (max {})",
                h, w, self.config.num_encoder_stages, max_stages
            )));
        }

        // Encoder: progressively downsample
        let mut encoder_features: Vec<Array2<f64>> =
            Vec::with_capacity(self.config.num_encoder_stages + 1);
        encoder_features.push(image.clone());

        let mut current = image.clone();
        for i in 0..self.config.num_encoder_stages {
            // Convolve with encoder kernel
            let convolved = convolve_3x3(&current, &self.encoder_kernels[i]);
            // Downsample 2x
            let downsampled = downsample_2x(&convolved);
            // ReLU
            let activated = downsampled.mapv(|v| v.max(0.0));
            encoder_features.push(activated.clone());
            current = activated;
        }

        // Decoder: progressively upsample with skip connections
        let mut decoded = current;
        for i in (0..self.config.num_encoder_stages).rev() {
            // Upsample 2x
            let upsampled = upsample_2x(&decoded);

            // Skip connection: add encoder features (resize if needed)
            let skip = &encoder_features[i];
            let (sh, sw) = skip.dim();
            let (uh, uw) = upsampled.dim();

            // Align sizes (take the minimum)
            let oh = sh.min(uh);
            let ow = sw.min(uw);

            let mut combined = Array2::zeros((oh, ow));
            for r in 0..oh {
                for c in 0..ow {
                    combined[[r, c]] = upsampled[[r, c]] + skip[[r, c]] * 0.5;
                }
            }

            // Convolve with decoder kernel
            let convolved = convolve_3x3(&combined, &self.decoder_kernels[i]);
            // ReLU
            decoded = convolved.mapv(|v| v.max(0.0));
        }

        // Resize to original dimensions if needed
        let (dh, dw) = decoded.dim();
        let output = if dh != h || dw != w {
            resize_bilinear(&decoded, h, w)
        } else {
            decoded
        };

        Ok(RelativeDepthMap::from_data(output))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Depth Completion
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for depth completion.
#[derive(Debug, Clone)]
pub struct DepthCompletionConfig {
    /// Spatial sigma for bilateral filter (default 5.0).
    pub sigma_spatial: f64,
    /// Intensity sigma for bilateral filter (default 0.1).
    pub sigma_intensity: f64,
    /// Number of propagation iterations (default 10).
    pub num_iterations: usize,
    /// Filter radius in pixels (default 5).
    pub radius: usize,
}

impl Default for DepthCompletionConfig {
    fn default() -> Self {
        Self {
            sigma_spatial: 5.0,
            sigma_intensity: 0.1,
            num_iterations: 10,
            radius: 5,
        }
    }
}

/// Fill sparse depth measurements into a dense depth map using bilateral
/// filter-based propagation with confidence-weighted interpolation.
///
/// Valid (non-zero) depth values serve as anchors.  In each iteration,
/// zero-valued pixels adopt the bilateral-weighted average of their
/// neighbourhood, with higher confidence for pixels closer to known values.
///
/// # Arguments
///
/// * `sparse_depth` - Sparse depth values, shape `[H, W]`.  Zero means missing.
/// * `guidance`     - Guidance image (e.g. grayscale), shape `[H, W]`.
///   Used for edge-aware weighting in the bilateral filter.
/// * `config`       - Completion parameters.
///
/// # Returns
///
/// Dense depth map of the same shape.  Known values are preserved exactly.
///
/// # Errors
///
/// Returns an error if shapes do not match.
pub fn depth_completion(
    sparse_depth: &Array2<f64>,
    guidance: &Array2<f64>,
    config: &DepthCompletionConfig,
) -> Result<Array2<f64>> {
    let (h, w) = sparse_depth.dim();
    if guidance.dim() != (h, w) {
        return Err(VisionError::DimensionMismatch(
            "sparse_depth and guidance must have the same shape".to_string(),
        ));
    }

    let mut output = sparse_depth.clone();
    let mut confidence = Array2::zeros((h, w));

    // Mark known pixels with confidence 1.0
    for r in 0..h {
        for c in 0..w {
            if sparse_depth[[r, c]] > 0.0 {
                confidence[[r, c]] = 1.0;
            }
        }
    }

    let inv_2_ss2 = 1.0 / (2.0 * config.sigma_spatial * config.sigma_spatial);
    let inv_2_si2 = 1.0 / (2.0 * config.sigma_intensity * config.sigma_intensity);
    let radius = config.radius as i64;

    for _iter in 0..config.num_iterations {
        let prev = output.clone();
        let prev_conf = confidence.clone();

        for r in 0..h {
            for c in 0..w {
                // Skip known pixels (preserve original values)
                if sparse_depth[[r, c]] > 0.0 {
                    continue;
                }

                let guide_center = guidance[[r, c]];
                let mut weighted_depth = 0.0_f64;
                let mut total_weight = 0.0_f64;

                for dr in -radius..=radius {
                    let nr = r as i64 + dr;
                    if nr < 0 || nr >= h as i64 {
                        continue;
                    }
                    let nr = nr as usize;
                    for dc in -radius..=radius {
                        let nc = c as i64 + dc;
                        if nc < 0 || nc >= w as i64 {
                            continue;
                        }
                        let nc = nc as usize;

                        let neighbor_depth = prev[[nr, nc]];
                        let neighbor_conf = prev_conf[[nr, nc]];
                        if neighbor_depth <= 0.0 || neighbor_conf <= 0.0 {
                            continue;
                        }

                        // Spatial weight
                        let dist2 = (dr * dr + dc * dc) as f64;
                        let ws = (-dist2 * inv_2_ss2).exp();

                        // Intensity (guidance) weight
                        let guide_diff = guidance[[nr, nc]] - guide_center;
                        let wi = (-guide_diff * guide_diff * inv_2_si2).exp();

                        let w = ws * wi * neighbor_conf;
                        weighted_depth += neighbor_depth * w;
                        total_weight += w;
                    }
                }

                if total_weight > 1e-12 {
                    output[[r, c]] = weighted_depth / total_weight;
                    confidence[[r, c]] = (total_weight / (total_weight + 1.0)).min(0.99);
                }
            }
        }
    }

    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// Depth from Disparity
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a stereo disparity map to depth using `D = f * B / d`.
///
/// # Arguments
///
/// * `disparity`    - Disparity map `[H, W]`.  Zero or negative values produce zero depth.
/// * `focal_length` - Focal length in pixels.
/// * `baseline`     - Stereo baseline in metres.
///
/// # Returns
///
/// Depth map of the same shape.
///
/// # Errors
///
/// Returns an error if focal length or baseline is non-positive.
pub fn depth_from_disparity(
    disparity: &Array2<f64>,
    focal_length: f64,
    baseline: f64,
) -> Result<Array2<f64>> {
    if focal_length <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "focal_length must be positive".to_string(),
        ));
    }
    if baseline <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "baseline must be positive".to_string(),
        ));
    }

    let fb = focal_length * baseline;
    let depth = disparity.mapv(|d| if d > 0.0 { fb / d } else { 0.0 });

    Ok(depth)
}

// ─────────────────────────────────────────────────────────────────────────────
// Depth to Point Cloud
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a depth map to a 3D point cloud using pinhole camera intrinsics.
///
/// # Arguments
///
/// * `depth` - Depth map `[H, W]` in metres.  Zero or negative values are skipped.
/// * `fx`, `fy` - Focal lengths in pixels.
/// * `cx`, `cy` - Principal point in pixels.
///
/// # Returns
///
/// A vector of `[x, y, z]` 3D points.
pub fn depth_to_point_cloud(
    depth: &Array2<f64>,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
) -> Vec<[f64; 3]> {
    let (h, w) = depth.dim();
    let mut points = Vec::with_capacity(h * w / 4); // estimate ~25% valid

    for r in 0..h {
        for c in 0..w {
            let z = depth[[r, c]];
            if z <= 0.0 || !z.is_finite() {
                continue;
            }
            let x = (c as f64 - cx) * z / fx;
            let y = (r as f64 - cy) * z / fy;
            points.push([x, y, z]);
        }
    }

    points
}

// ─────────────────────────────────────────────────────────────────────────────
// Depth Colorization
// ─────────────────────────────────────────────────────────────────────────────

/// Colorize a depth map using a turbo-style colormap.
///
/// Maps depth values to RGB colours for visualisation.
/// Near objects (small depth) map to warm colours (red/orange),
/// far objects (large depth) map to cool colours (blue/purple).
///
/// # Arguments
///
/// * `depth` - Depth map `[H, W]`.
/// * `min_depth` - Minimum depth for colormap range.
/// * `max_depth` - Maximum depth for colormap range.
///
/// # Returns
///
/// A `Vec<Vec<[u8; 3]>>` of shape `[H][W]` with RGB values.
///
/// # Errors
///
/// Returns an error if `min_depth >= max_depth`.
pub fn depth_colorize(
    depth: &Array2<f64>,
    min_depth: f64,
    max_depth: f64,
) -> Result<Vec<Vec<[u8; 3]>>> {
    if min_depth >= max_depth {
        return Err(VisionError::InvalidParameter(
            "min_depth must be less than max_depth".to_string(),
        ));
    }

    let (h, w) = depth.dim();
    let range = max_depth - min_depth;
    let mut result = vec![vec![[0u8; 3]; w]; h];

    for r in 0..h {
        for c in 0..w {
            let d = depth[[r, c]];
            if d <= 0.0 || !d.is_finite() {
                result[r][c] = [0, 0, 0]; // invalid -> black
                continue;
            }

            let t = ((d - min_depth) / range).clamp(0.0, 1.0);
            result[r][c] = turbo_colormap(t);
        }
    }

    Ok(result)
}

/// Turbo-style colormap: maps a value in [0, 1] to an RGB colour.
///
/// Approximation of the Turbo colormap from Google Research.
fn turbo_colormap(t: f64) -> [u8; 3] {
    // Simplified piecewise-linear approximation of turbo colormap
    let r = if t < 0.25 {
        // Dark blue to blue-green
        t * 4.0 * 0.3
    } else if t < 0.5 {
        // Blue-green to green-yellow
        0.3 + (t - 0.25) * 4.0 * 0.5
    } else if t < 0.75 {
        // Green-yellow to orange
        0.8 + (t - 0.5) * 4.0 * 0.2
    } else {
        // Orange to red
        1.0
    };

    let g = if t < 0.25 {
        t * 4.0 * 0.6
    } else if t < 0.5 {
        0.6 + (t - 0.25) * 4.0 * 0.4
    } else if t < 0.75 {
        1.0 - (t - 0.5) * 4.0 * 0.5
    } else {
        0.5 - (t - 0.75) * 4.0 * 0.5
    };

    let b = if t < 0.25 {
        0.5 + t * 4.0 * 0.5
    } else if t < 0.5 {
        1.0 - (t - 0.25) * 4.0 * 0.7
    } else if t < 0.75 {
        0.3 - (t - 0.5) * 4.0 * 0.2
    } else {
        0.1 - (t - 0.75) * 4.0 * 0.1
    };

    [
        (r.clamp(0.0, 1.0) * 255.0) as u8,
        (g.clamp(0.0, 1.0) * 255.0) as u8,
        (b.clamp(0.0, 1.0) * 255.0) as u8,
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// Depth Edge Detection
// ─────────────────────────────────────────────────────────────────────────────

/// Detect edges in a depth map using gradient magnitude.
///
/// Computes Sobel-like horizontal and vertical gradients, then takes the
/// magnitude.  Large gradient magnitudes correspond to depth discontinuities.
///
/// # Arguments
///
/// * `depth`     - Depth map `[H, W]`.
/// * `threshold` - Gradient magnitude threshold.  Pixels above this are marked as edges.
///
/// # Returns
///
/// Binary edge map of the same shape (1.0 = edge, 0.0 = non-edge).
pub fn depth_edge_detection(depth: &Array2<f64>, threshold: f64) -> Array2<f64> {
    let (h, w) = depth.dim();
    let mut edges = Array2::zeros((h, w));

    if h < 3 || w < 3 {
        return edges;
    }

    for r in 1..h - 1 {
        for c in 1..w - 1 {
            // Sobel gradient in X
            let gx = -depth[[r - 1, c - 1]] - 2.0 * depth[[r, c - 1]] - depth[[r + 1, c - 1]]
                + depth[[r - 1, c + 1]]
                + 2.0 * depth[[r, c + 1]]
                + depth[[r + 1, c + 1]];

            // Sobel gradient in Y
            let gy = -depth[[r - 1, c - 1]] - 2.0 * depth[[r - 1, c]] - depth[[r - 1, c + 1]]
                + depth[[r + 1, c - 1]]
                + 2.0 * depth[[r + 1, c]]
                + depth[[r + 1, c + 1]];

            let magnitude = (gx * gx + gy * gy).sqrt();
            if magnitude > threshold {
                edges[[r, c]] = 1.0;
            }
        }
    }

    edges
}

// ─────────────────────────────────────────────────────────────────────────────
// Image processing helpers (private)
// ─────────────────────────────────────────────────────────────────────────────

/// 3x3 convolution with zero-padding.
fn convolve_3x3(image: &Array2<f64>, kernel: &[[f64; 3]; 3]) -> Array2<f64> {
    let (h, w) = image.dim();
    let mut output = Array2::zeros((h, w));

    for r in 0..h {
        for c in 0..w {
            let mut sum = 0.0;
            for (kr, kernel_row) in kernel.iter().enumerate() {
                for (kc, &kernel_val) in kernel_row.iter().enumerate() {
                    let ir = r as i64 + kr as i64 - 1;
                    let ic = c as i64 + kc as i64 - 1;
                    if ir >= 0 && ir < h as i64 && ic >= 0 && ic < w as i64 {
                        sum += image[[ir as usize, ic as usize]] * kernel_val;
                    }
                }
            }
            output[[r, c]] = sum;
        }
    }

    output
}

/// Downsample an image by a factor of 2 using average pooling.
fn downsample_2x(image: &Array2<f64>) -> Array2<f64> {
    let (h, w) = image.dim();
    let nh = h / 2;
    let nw = w / 2;
    let mut output = Array2::zeros((nh.max(1), nw.max(1)));

    for r in 0..nh {
        for c in 0..nw {
            let sum = image[[r * 2, c * 2]]
                + image[[(r * 2 + 1).min(h - 1), c * 2]]
                + image[[r * 2, (c * 2 + 1).min(w - 1)]]
                + image[[(r * 2 + 1).min(h - 1), (c * 2 + 1).min(w - 1)]];
            output[[r, c]] = sum * 0.25;
        }
    }

    output
}

/// Upsample an image by a factor of 2 using nearest-neighbour interpolation.
fn upsample_2x(image: &Array2<f64>) -> Array2<f64> {
    let (h, w) = image.dim();
    let nh = h * 2;
    let nw = w * 2;
    let mut output = Array2::zeros((nh, nw));

    for r in 0..nh {
        for c in 0..nw {
            output[[r, c]] = image[[r / 2, c / 2]];
        }
    }

    output
}

/// Bilinear resize of a 2D array.
fn resize_bilinear(image: &Array2<f64>, new_h: usize, new_w: usize) -> Array2<f64> {
    let (h, w) = image.dim();
    let mut output = Array2::zeros((new_h, new_w));

    if h == 0 || w == 0 || new_h == 0 || new_w == 0 {
        return output;
    }

    let scale_r = h as f64 / new_h as f64;
    let scale_c = w as f64 / new_w as f64;

    for r in 0..new_h {
        for c in 0..new_w {
            let src_r = r as f64 * scale_r;
            let src_c = c as f64 * scale_c;

            let r0 = (src_r.floor() as usize).min(h - 1);
            let c0 = (src_c.floor() as usize).min(w - 1);
            let r1 = (r0 + 1).min(h - 1);
            let c1 = (c0 + 1).min(w - 1);

            let dr = src_r - r0 as f64;
            let dc = src_c - c0 as f64;

            output[[r, c]] = image[[r0, c0]] * (1.0 - dr) * (1.0 - dc)
                + image[[r0, c1]] * (1.0 - dr) * dc
                + image[[r1, c0]] * dr * (1.0 - dc)
                + image[[r1, c1]] * dr * dc;
        }
    }

    output
}

/// Generate a normalised 3x3 Gaussian kernel.
fn gaussian_kernel_3x3(sigma: f64) -> [[f64; 3]; 3] {
    let mut kernel = [[0.0; 3]; 3];
    let mut total = 0.0;
    let inv_2s2 = 1.0 / (2.0 * sigma * sigma);

    for (r, kernel_row) in kernel.iter_mut().enumerate() {
        for (c, kernel_val) in kernel_row.iter_mut().enumerate() {
            let dr = r as f64 - 1.0;
            let dc = c as f64 - 1.0;
            let val = (-(dr * dr + dc * dc) * inv_2s2).exp();
            *kernel_val = val;
            total += val;
        }
    }

    if total > 1e-15 {
        for row in &mut kernel {
            for v in row.iter_mut() {
                *v /= total;
            }
        }
    }

    kernel
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relative_depth_map_normalised() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .expect("from_shape_vec should succeed");
        let rdm = RelativeDepthMap::from_data(data);
        let norm = rdm.normalised();
        assert!((norm[[0, 0]] - 0.0).abs() < 1e-9);
        assert!((norm[[1, 1]] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_depth_completion_preserves_known() {
        let mut sparse = Array2::zeros((8, 8));
        sparse[[2, 2]] = 5.0;
        sparse[[5, 5]] = 10.0;
        let guidance = Array2::from_elem((8, 8), 128.0);
        let config = DepthCompletionConfig::default();

        let dense =
            depth_completion(&sparse, &guidance, &config).expect("depth_completion should succeed");

        // Known values must be preserved exactly
        assert!(
            (dense[[2, 2]] - 5.0).abs() < 1e-9,
            "known pixel at (2,2) = {}",
            dense[[2, 2]]
        );
        assert!(
            (dense[[5, 5]] - 10.0).abs() < 1e-9,
            "known pixel at (5,5) = {}",
            dense[[5, 5]]
        );

        // Some previously-unknown pixels should now have depth
        let filled_count = dense.iter().filter(|&&v| v > 0.0).count();
        assert!(
            filled_count > 2,
            "should fill more than just known pixels, got {}",
            filled_count
        );
    }

    #[test]
    fn test_depth_completion_shape_mismatch() {
        let sparse = Array2::zeros((4, 4));
        let guidance = Array2::zeros((4, 5));
        let config = DepthCompletionConfig::default();
        assert!(depth_completion(&sparse, &guidance, &config).is_err());
    }

    #[test]
    fn test_depth_from_disparity() {
        // D = f * B / d = 800 * 0.1 / 10 = 8.0
        let disparity = Array2::from_elem((3, 3), 10.0);
        let depth = depth_from_disparity(&disparity, 800.0, 0.1)
            .expect("depth_from_disparity should succeed");
        for &v in depth.iter() {
            assert!((v - 8.0).abs() < 1e-9, "depth = {}", v);
        }
    }

    #[test]
    fn test_depth_from_disparity_zero_disparity() {
        let mut disparity = Array2::from_elem((3, 3), 10.0);
        disparity[[1, 1]] = 0.0;
        let depth = depth_from_disparity(&disparity, 800.0, 0.1)
            .expect("depth_from_disparity should succeed");
        assert!((depth[[1, 1]]).abs() < 1e-9);
    }

    #[test]
    fn test_depth_from_disparity_invalid_params() {
        let disparity = Array2::from_elem((2, 2), 5.0);
        assert!(depth_from_disparity(&disparity, 0.0, 0.1).is_err());
        assert!(depth_from_disparity(&disparity, 800.0, -1.0).is_err());
    }

    #[test]
    fn test_depth_to_point_cloud_known_geometry() {
        let mut depth = Array2::zeros((5, 5));
        // Place a point at the principal point at depth 3.0
        depth[[2, 2]] = 3.0;
        // Place a point at (4, 4) at depth 2.0
        depth[[4, 4]] = 2.0;

        let points = depth_to_point_cloud(&depth, 100.0, 100.0, 2.0, 2.0);
        assert_eq!(points.len(), 2);

        // Point at principal point: x=0, y=0, z=3
        let p0 = &points[0];
        assert!((p0[0]).abs() < 1e-9, "x = {}", p0[0]);
        assert!((p0[1]).abs() < 1e-9, "y = {}", p0[1]);
        assert!((p0[2] - 3.0).abs() < 1e-9, "z = {}", p0[2]);

        // Point at (4, 4): x = (4-2)*2/100 = 0.04, y = (4-2)*2/100 = 0.04, z = 2
        let p1 = &points[1];
        assert!((p1[0] - 0.04).abs() < 1e-9, "x = {}", p1[0]);
        assert!((p1[1] - 0.04).abs() < 1e-9, "y = {}", p1[1]);
        assert!((p1[2] - 2.0).abs() < 1e-9, "z = {}", p1[2]);
    }

    #[test]
    fn test_depth_colorize() {
        let depth = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .expect("from_shape_vec should succeed");
        let colors = depth_colorize(&depth, 0.0, 5.0).expect("depth_colorize should succeed");
        assert_eq!(colors.len(), 2);
        assert_eq!(colors[0].len(), 2);

        // All values should produce non-zero RGB (they are all valid)
        for row in &colors {
            for rgb in row {
                // At least one channel should be > 0
                assert!(rgb[0] > 0 || rgb[1] > 0 || rgb[2] > 0);
            }
        }
    }

    #[test]
    fn test_depth_colorize_invalid_range() {
        let depth = Array2::from_elem((2, 2), 1.0);
        assert!(depth_colorize(&depth, 5.0, 1.0).is_err());
        assert!(depth_colorize(&depth, 1.0, 1.0).is_err());
    }

    #[test]
    fn test_depth_edge_detection() {
        // Create a depth map with a sharp edge
        let mut depth = Array2::zeros((10, 10));
        for r in 0..10 {
            for c in 0..10 {
                depth[[r, c]] = if c < 5 { 1.0 } else { 10.0 };
            }
        }

        let edges = depth_edge_detection(&depth, 1.0);
        assert_eq!(edges.dim(), (10, 10));

        // There should be edges near column 5
        let has_edge = edges.iter().any(|&v| v > 0.5);
        assert!(has_edge, "should detect edge at depth discontinuity");

        // Pixels far from the edge should not be edges
        assert!((edges[[5, 0]]).abs() < 1e-9, "no edge expected at (5,0)");
        assert!((edges[[5, 9]]).abs() < 1e-9, "no edge expected at (5,9)");
    }

    #[test]
    fn test_depth_edge_detection_small_image() {
        let depth = Array2::from_elem((2, 2), 1.0);
        let edges = depth_edge_detection(&depth, 0.1);
        // Too small for Sobel, should be all zeros
        assert!(edges.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_depth_estimator_output_shape() {
        let config = DepthEstimatorConfig {
            num_encoder_stages: 2,
            base_channels: 8,
        };
        let estimator = DepthEstimator::new(config);
        let image = Array2::from_elem((32, 32), 128.0);
        let result = estimator.estimate(&image).expect("estimate should succeed");
        assert_eq!(result.dim(), (32, 32));
    }

    #[test]
    fn test_depth_estimator_too_small() {
        let config = DepthEstimatorConfig {
            num_encoder_stages: 3,
            base_channels: 8,
        };
        let estimator = DepthEstimator::new(config);
        let image = Array2::from_elem((3, 3), 128.0);
        assert!(estimator.estimate(&image).is_err());
    }

    #[test]
    fn test_gaussian_kernel_sums_to_one() {
        let k = gaussian_kernel_3x3(1.0);
        let sum: f64 = k.iter().flat_map(|row| row.iter()).sum();
        assert!((sum - 1.0).abs() < 1e-10, "kernel sum = {}", sum);
    }
}
