//! Neural Radiance Fields (NeRF) implementation
//!
//! Implements the core components of NeRF (Mildenhall et al., 2020):
//! - Positional encoding with configurable frequency levels
//! - MLP with skip connections for density + colour prediction
//! - Volume rendering equation with transmittance accumulation
//! - Ray generation from camera intrinsics and extrinsics
//! - Hierarchical (coarse-to-fine) importance sampling

use crate::camera::CameraExtrinsics;
use crate::camera::CameraIntrinsics;
use crate::error::{Result, VisionError};

// ─────────────────────────────────────────────────────────────────────────────
// Positional Encoding
// ─────────────────────────────────────────────────────────────────────────────

/// Positional encoding: maps a scalar or vector into a higher-dimensional
/// Fourier feature space.
///
/// For a scalar `p`, the encoding is:
/// `[sin(2^0 pi p), cos(2^0 pi p), ..., sin(2^{L-1} pi p), cos(2^{L-1} pi p)]`
///
/// For an `N`-dimensional vector the encoding is applied independently to each
/// component, producing a `2 * L * N`-dimensional output.
#[derive(Debug, Clone)]
pub struct PositionalEncoding {
    /// Number of frequency levels.
    pub num_levels: usize,
}

impl PositionalEncoding {
    /// Create a new positional encoding with `num_levels` frequency bands.
    pub fn new(num_levels: usize) -> Self {
        Self { num_levels }
    }

    /// Output dimensionality for an input of `input_dim` components.
    pub fn output_dim(&self, input_dim: usize) -> usize {
        2 * self.num_levels * input_dim
    }

    /// Encode a single scalar value.
    pub fn encode_scalar(&self, value: f64) -> Vec<f64> {
        let mut encoded = Vec::with_capacity(2 * self.num_levels);
        for l in 0..self.num_levels {
            let freq = std::f64::consts::PI * (1u64 << l) as f64;
            let angle = freq * value;
            encoded.push(angle.sin());
            encoded.push(angle.cos());
        }
        encoded
    }

    /// Encode a vector of values, concatenating per-component encodings.
    pub fn encode(&self, values: &[f64]) -> Vec<f64> {
        let mut encoded = Vec::with_capacity(self.output_dim(values.len()));
        for &v in values {
            encoded.extend(self.encode_scalar(v));
        }
        encoded
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ray
// ─────────────────────────────────────────────────────────────────────────────

/// A single ray defined by an origin and a direction.
#[derive(Debug, Clone)]
pub struct Ray {
    /// Origin of the ray `[x, y, z]`.
    pub origin: [f64; 3],
    /// Unit direction of the ray `[dx, dy, dz]`.
    pub direction: [f64; 3],
}

impl Ray {
    /// Create a new ray. The direction is normalised internally.
    pub fn new(origin: [f64; 3], direction: [f64; 3]) -> Self {
        let len = (direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2])
            .sqrt()
            .max(1e-15);
        Self {
            origin,
            direction: [direction[0] / len, direction[1] / len, direction[2] / len],
        }
    }

    /// Evaluate the ray at parameter `t`: `origin + t * direction`.
    pub fn at(&self, t: f64) -> [f64; 3] {
        [
            self.origin[0] + t * self.direction[0],
            self.origin[1] + t * self.direction[1],
            self.origin[2] + t * self.direction[2],
        ]
    }
}

/// A bundle of rays (e.g. for one image).
#[derive(Debug, Clone)]
pub struct RayBundle {
    /// All rays in the bundle.
    pub rays: Vec<Ray>,
    /// Image width (pixels) that produced these rays.
    pub width: usize,
    /// Image height (pixels) that produced these rays.
    pub height: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Ray generation
// ─────────────────────────────────────────────────────────────────────────────

/// Generate rays for every pixel in an image given camera intrinsics and
/// extrinsics.
///
/// Each ray originates at the camera centre (in world space) and passes
/// through the centre of the corresponding pixel.
///
/// # Arguments
///
/// * `intrinsics` - Camera intrinsic parameters.
/// * `extrinsics` - Camera extrinsic parameters (world-to-camera transform).
/// * `width`      - Image width in pixels.
/// * `height`     - Image height in pixels.
///
/// # Returns
///
/// A [`RayBundle`] containing `width * height` rays.
pub fn generate_rays(
    intrinsics: &CameraIntrinsics,
    extrinsics: &CameraExtrinsics,
    width: usize,
    height: usize,
) -> RayBundle {
    // Camera position in world frame: C = -R^T t
    let rt = transpose3(&extrinsics.rotation);
    let neg_t = [
        -extrinsics.translation[0],
        -extrinsics.translation[1],
        -extrinsics.translation[2],
    ];
    let cam_pos = mat3_vec3_mul(&rt, neg_t);

    let mut rays = Vec::with_capacity(width * height);

    for row in 0..height {
        for col in 0..width {
            // Pixel centre in normalised camera coordinates
            let xn = (col as f64 + 0.5 - intrinsics.cx) / intrinsics.fx;
            let yn = (row as f64 + 0.5 - intrinsics.cy) / intrinsics.fy;

            // Direction in camera frame: (xn, yn, 1)
            let dir_cam = [xn, yn, 1.0];

            // Rotate to world frame: d_world = R^T * d_cam
            let dir_world = mat3_vec3_mul(&rt, dir_cam);

            rays.push(Ray::new(cam_pos, dir_world));
        }
    }

    RayBundle {
        rays,
        width,
        height,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NeRF MLP
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a NeRF model.
#[derive(Debug, Clone)]
pub struct NeRFConfig {
    /// Number of positional encoding levels for position (default 10).
    pub pos_encoding_levels: usize,
    /// Number of positional encoding levels for direction (default 4).
    pub dir_encoding_levels: usize,
    /// Hidden layer width (default 256).
    pub hidden_dim: usize,
    /// Number of hidden layers (default 8).
    pub num_layers: usize,
    /// Layer index for the skip connection (default 4, 0-indexed).
    pub skip_layer: usize,
    /// Number of coarse samples along each ray (default 64).
    pub num_coarse_samples: usize,
    /// Number of fine samples along each ray (default 128).
    pub num_fine_samples: usize,
    /// Near clipping distance (default 2.0).
    pub near: f64,
    /// Far clipping distance (default 6.0).
    pub far: f64,
}

impl Default for NeRFConfig {
    fn default() -> Self {
        Self {
            pos_encoding_levels: 10,
            dir_encoding_levels: 4,
            hidden_dim: 256,
            num_layers: 8,
            skip_layer: 4,
            num_coarse_samples: 64,
            num_fine_samples: 128,
            near: 2.0,
            far: 6.0,
        }
    }
}

/// A simple NeRF MLP model.
///
/// Architecture: 8 dense layers with ReLU, skip connection at layer 4.
///
/// - Input: positional encoding of 3D position
/// - After layer 4: concatenate input encoding (skip connection)
/// - Output of density branch: sigma (1 value)
/// - Colour branch: concat direction encoding -> 1 additional layer -> RGB (3 values)
///
/// Weights are stored as flat `Vec<f64>` for simplicity.
#[derive(Debug, Clone)]
pub struct NeRFModel {
    /// Model configuration.
    pub config: NeRFConfig,
    /// Positional encoding for position.
    pub pos_encoder: PositionalEncoding,
    /// Positional encoding for direction.
    pub dir_encoder: PositionalEncoding,
    /// Weights for each layer: `weights[i]` is a flat row-major matrix.
    weights: Vec<Vec<f64>>,
    /// Biases for each layer.
    biases: Vec<Vec<f64>>,
    /// Density output weight (from last hidden to sigma).
    sigma_weight: Vec<f64>,
    /// Density output bias.
    sigma_bias: f64,
    /// Colour branch weight (from hidden + dir encoding to RGB).
    color_weight: Vec<f64>,
    /// Colour branch bias.
    color_bias: [f64; 3],
}

impl NeRFModel {
    /// Create a new NeRF model with random-ish small weights.
    ///
    /// This is a structural model; for real training one would need a proper
    /// optimiser. The weights are initialised with a simple deterministic
    /// pattern for reproducibility.
    pub fn new(config: NeRFConfig) -> Self {
        let pos_encoder = PositionalEncoding::new(config.pos_encoding_levels);
        let dir_encoder = PositionalEncoding::new(config.dir_encoding_levels);

        let pos_dim = pos_encoder.output_dim(3);
        let _dir_dim = dir_encoder.output_dim(3);
        let hidden = config.hidden_dim;
        let num_layers = config.num_layers;
        let skip = config.skip_layer;

        let mut weights: Vec<Vec<f64>> = Vec::with_capacity(num_layers);
        let mut biases: Vec<Vec<f64>> = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let in_dim = if i == 0 {
                pos_dim
            } else if i == skip {
                hidden + pos_dim // skip connection
            } else {
                hidden
            };
            // Xavier-style init
            let scale = (2.0 / (in_dim + hidden) as f64).sqrt();
            let w: Vec<f64> = (0..in_dim * hidden)
                .map(|j| {
                    let seed = (i * 10007 + j * 31 + 17) as f64;
                    scale * (seed.sin() * 0.5)
                })
                .collect();
            let b = vec![0.0; hidden];
            weights.push(w);
            biases.push(b);
        }

        let sigma_weight: Vec<f64> = (0..hidden).map(|j| 0.01 * (j as f64).cos()).collect();
        let sigma_bias = 0.0;

        let color_input_dim = hidden + dir_encoder.output_dim(3);
        let color_weight: Vec<f64> = (0..color_input_dim * 3)
            .map(|j| 0.01 * (j as f64 * 0.7).sin())
            .collect();
        let color_bias = [0.5, 0.5, 0.5]; // default mid-grey

        Self {
            config,
            pos_encoder,
            dir_encoder,
            weights,
            biases,
            sigma_weight,
            sigma_bias,
            color_weight,
            color_bias,
        }
    }

    /// Forward pass: given a 3D position and a 3D direction, return (sigma, rgb).
    ///
    /// `sigma` is the volume density (non-negative after ReLU).
    /// `rgb` is in `[0, 1]` (after sigmoid).
    pub fn forward(&self, position: &[f64; 3], direction: &[f64; 3]) -> (f64, [f64; 3]) {
        let pos_enc = self.pos_encoder.encode(position);
        let dir_enc = self.dir_encoder.encode(direction);

        let hidden = self.config.hidden_dim;
        let skip = self.config.skip_layer;

        // Forward through hidden layers
        let mut h = pos_enc.clone();
        for i in 0..self.config.num_layers {
            // At skip layer, concatenate the original positional encoding
            if i == skip {
                let mut combined = h.clone();
                combined.extend_from_slice(&pos_enc);
                h = combined;
            }

            let in_dim = h.len();
            let w = &self.weights[i];
            let b = &self.biases[i];

            let mut out = vec![0.0; hidden];
            for o in 0..hidden {
                let mut sum = b[o];
                for k in 0..in_dim {
                    sum += h[k] * w[k * hidden + o];
                }
                // ReLU
                out[o] = sum.max(0.0);
            }
            h = out;
        }

        // Density: linear from hidden -> 1, then ReLU
        let mut sigma = self.sigma_bias;
        for (j, &hj) in h.iter().enumerate() {
            sigma += hj * self.sigma_weight[j];
        }
        sigma = sigma.max(0.0); // ReLU ensures non-negative density

        // Colour: concat hidden + dir_encoding, then linear -> 3, then sigmoid
        let mut color_input = h;
        color_input.extend_from_slice(&dir_enc);

        let cin_dim = color_input.len();
        let mut rgb = [0.0_f64; 3];
        for (c, rgb_val) in rgb.iter_mut().enumerate() {
            let mut sum = self.color_bias[c];
            for (k, &ci) in color_input.iter().enumerate().take(cin_dim) {
                sum += ci * self.color_weight[k * 3 + c];
            }
            // Sigmoid
            *rgb_val = 1.0 / (1.0 + (-sum).exp());
        }

        (sigma, rgb)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Volume Rendering
// ─────────────────────────────────────────────────────────────────────────────

/// Result of volume rendering a single ray.
#[derive(Debug, Clone)]
pub struct VolumeRenderResult {
    /// Accumulated colour `[R, G, B]` in `[0, 1]`.
    pub color: [f64; 3],
    /// Accumulated depth (expected termination distance).
    pub depth: f64,
    /// Accumulated opacity (sum of weights).
    pub opacity: f64,
    /// Per-sample weights (for importance sampling).
    pub weights: Vec<f64>,
}

/// Volume-render a single ray through a density/colour field.
///
/// Implements the discretised volume rendering equation:
///
/// ```text
/// C(r) = sum_i T_i * (1 - exp(-sigma_i * delta_i)) * c_i
/// T_i  = exp(-sum_{j<i} sigma_j * delta_j)
/// ```
///
/// # Arguments
///
/// * `sigmas`  - Volume densities at each sample point along the ray.
/// * `colors`  - RGB colours at each sample point (each `[R, G, B]` in `[0, 1]`).
/// * `deltas`  - Distances between consecutive sample points.
///
/// # Errors
///
/// Returns an error if the input slices have inconsistent lengths.
pub fn volume_render(
    sigmas: &[f64],
    colors: &[[f64; 3]],
    deltas: &[f64],
) -> Result<VolumeRenderResult> {
    let n = sigmas.len();
    if colors.len() != n || deltas.len() != n {
        return Err(VisionError::InvalidParameter(
            "sigmas, colors, and deltas must have the same length".to_string(),
        ));
    }

    if n == 0 {
        return Ok(VolumeRenderResult {
            color: [0.0; 3],
            depth: 0.0,
            opacity: 0.0,
            weights: Vec::new(),
        });
    }

    let mut accumulated_color = [0.0_f64; 3];
    let mut accumulated_depth = 0.0_f64;
    let mut transmittance = 1.0_f64;
    let mut accumulated_opacity = 0.0_f64;
    let mut weights = Vec::with_capacity(n);

    // Running distance from ray origin for depth computation
    let mut t = 0.0_f64;

    for i in 0..n {
        let alpha = 1.0 - (-sigmas[i] * deltas[i]).exp();
        let weight = transmittance * alpha;

        accumulated_color[0] += weight * colors[i][0];
        accumulated_color[1] += weight * colors[i][1];
        accumulated_color[2] += weight * colors[i][2];

        // Depth is the weighted sum of midpoint distances
        let t_mid = t + deltas[i] * 0.5;
        accumulated_depth += weight * t_mid;
        accumulated_opacity += weight;

        weights.push(weight);

        transmittance *= 1.0 - alpha;

        t += deltas[i];

        // Early termination when transmittance is negligible
        if transmittance < 1e-10 {
            // Fill remaining weights with zero
            weights.resize(n, 0.0);
            break;
        }
    }

    Ok(VolumeRenderResult {
        color: accumulated_color,
        depth: accumulated_depth,
        opacity: accumulated_opacity,
        weights,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Hierarchical Sampling
// ─────────────────────────────────────────────────────────────────────────────

/// Hierarchical sampler for NeRF's coarse-to-fine rendering strategy.
///
/// 1. First sample `num_coarse` points uniformly between `near` and `far`.
/// 2. Use the coarse network's weights to build a piecewise-constant PDF.
/// 3. Sample `num_fine` additional points from this PDF (importance sampling).
#[derive(Debug, Clone)]
pub struct HierarchicalSampler {
    /// Number of coarse samples.
    pub num_coarse: usize,
    /// Number of fine (importance) samples.
    pub num_fine: usize,
    /// Near bound.
    pub near: f64,
    /// Far bound.
    pub far: f64,
}

impl HierarchicalSampler {
    /// Create a new hierarchical sampler.
    pub fn new(num_coarse: usize, num_fine: usize, near: f64, far: f64) -> Self {
        Self {
            num_coarse,
            num_fine,
            near,
            far,
        }
    }

    /// Generate uniformly spaced coarse sample distances along a ray.
    ///
    /// Returns `t` values (distances from ray origin) for `num_coarse` bins,
    /// with a small deterministic perturbation for anti-aliasing.
    pub fn coarse_samples(&self) -> Vec<f64> {
        let n = self.num_coarse;
        if n == 0 {
            return Vec::new();
        }
        let step = (self.far - self.near) / n as f64;
        (0..n)
            .map(|i| {
                let lo = self.near + i as f64 * step;
                // Midpoint of each bin (deterministic jitter)
                lo + step * 0.5
            })
            .collect()
    }

    /// Generate fine sample distances via inverse-CDF importance sampling.
    ///
    /// # Arguments
    ///
    /// * `coarse_ts` - The `t` values from the coarse pass.
    /// * `weights`   - The volume-rendering weights from the coarse pass.
    ///
    /// # Returns
    ///
    /// A sorted vector of `num_fine` sample distances.
    ///
    /// # Errors
    ///
    /// Returns an error if `coarse_ts` and `weights` have different lengths.
    pub fn fine_samples(&self, coarse_ts: &[f64], weights: &[f64]) -> Result<Vec<f64>> {
        let n = coarse_ts.len();
        if weights.len() != n {
            return Err(VisionError::InvalidParameter(
                "coarse_ts and weights must have the same length".to_string(),
            ));
        }
        if n == 0 || self.num_fine == 0 {
            return Ok(Vec::new());
        }

        // Build CDF from weights (add small epsilon to avoid zero-weight bins)
        let eps = 1e-5;
        let total: f64 = weights.iter().sum::<f64>() + eps * n as f64;
        let mut cdf = Vec::with_capacity(n + 1);
        cdf.push(0.0);
        let mut cumsum = 0.0;
        for &w in weights {
            cumsum += (w + eps) / total;
            cdf.push(cumsum);
        }
        // Ensure the last CDF value is exactly 1.0
        if let Some(last) = cdf.last_mut() {
            *last = 1.0;
        }

        // Determine bin edges from coarse_ts
        let step = if n > 1 {
            coarse_ts.get(1).copied().unwrap_or(self.far) - coarse_ts[0]
        } else {
            self.far - self.near
        };
        let mut bin_edges: Vec<f64> = coarse_ts.iter().map(|&t| t - step * 0.5).collect();
        bin_edges.push(coarse_ts.last().copied().unwrap_or(self.far) + step * 0.5);

        // Inverse-CDF sampling with uniform stratification
        let mut fine_ts = Vec::with_capacity(self.num_fine);
        for i in 0..self.num_fine {
            let u = (i as f64 + 0.5) / self.num_fine as f64;

            // Binary search in CDF
            let mut lo = 0;
            let mut hi = cdf.len() - 1;
            while lo < hi {
                let mid = (lo + hi) / 2;
                if cdf[mid] < u {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            let idx = lo.saturating_sub(1).min(n - 1);

            // Linear interpolation within the bin
            let cdf_lo = cdf[idx];
            let cdf_hi = cdf[idx + 1];
            let denom = cdf_hi - cdf_lo;
            let frac = if denom > 1e-15 {
                (u - cdf_lo) / denom
            } else {
                0.5
            };

            let t = bin_edges[idx] + frac * (bin_edges[idx + 1] - bin_edges[idx]);
            fine_ts.push(t.clamp(self.near, self.far));
        }

        fine_ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Ok(fine_ts)
    }

    /// Merge coarse and fine samples into one sorted list and return the
    /// inter-sample distances (deltas).
    pub fn merge_and_deltas(&self, coarse_ts: &[f64], fine_ts: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut all: Vec<f64> = coarse_ts.iter().chain(fine_ts.iter()).copied().collect();
        all.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        all.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

        let n = all.len();
        let deltas: Vec<f64> = if n <= 1 {
            vec![self.far - self.near; n]
        } else {
            (0..n)
                .map(|i| {
                    if i + 1 < n {
                        all[i + 1] - all[i]
                    } else {
                        // Last sample: extend to far bound
                        (self.far - all[i]).max(1e-6)
                    }
                })
                .collect()
        };

        (all, deltas)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Camera helpers (look-at, ray generation for NeRF)
// ─────────────────────────────────────────────────────────────────────────────

/// Build camera extrinsics from look-at parameters.
///
/// # Arguments
///
/// * `position` - Camera position in world space.
/// * `target`   - The point the camera is looking at.
/// * `up`       - World "up" direction (typically `[0, 1, 0]` or `[0, -1, 0]`).
///
/// # Returns
///
/// `CameraExtrinsics` encoding the world-to-camera rotation and translation.
///
/// # Errors
///
/// Returns an error if the forward or right vectors are degenerate (zero length).
pub fn camera_from_look_at(
    position: [f64; 3],
    target: [f64; 3],
    up: [f64; 3],
) -> Result<CameraExtrinsics> {
    // Forward = normalise(target - position)
    let fwd = [
        target[0] - position[0],
        target[1] - position[1],
        target[2] - position[2],
    ];
    let fwd_len = vec3_len(fwd);
    if fwd_len < 1e-12 {
        return Err(VisionError::InvalidParameter(
            "position and target are too close".to_string(),
        ));
    }
    let fwd = [fwd[0] / fwd_len, fwd[1] / fwd_len, fwd[2] / fwd_len];

    // Right = normalise(fwd x up)
    let right = cross3(fwd, up);
    let right_len = vec3_len(right);
    if right_len < 1e-12 {
        return Err(VisionError::InvalidParameter(
            "up vector is parallel to the look direction".to_string(),
        ));
    }
    let right = [
        right[0] / right_len,
        right[1] / right_len,
        right[2] / right_len,
    ];

    // Recompute up = right x fwd (ensures orthonormality)
    let true_up = cross3(right, fwd);

    // The rotation matrix R maps world to camera:
    // Camera X = right, Camera Y = true_up, Camera Z = -fwd (OpenGL convention)
    // But vision convention: Z = forward, Y = down
    // We use the standard vision convention: Z forward, X right, Y down
    // So: row0 = right, row1 = -true_up (since Y is down in vision), row2 = fwd
    let rotation = [
        [right[0], right[1], right[2]],
        [-true_up[0], -true_up[1], -true_up[2]],
        [fwd[0], fwd[1], fwd[2]],
    ];

    // Translation: t = -R * position
    let translation = [
        -(rotation[0][0] * position[0]
            + rotation[0][1] * position[1]
            + rotation[0][2] * position[2]),
        -(rotation[1][0] * position[0]
            + rotation[1][1] * position[1]
            + rotation[1][2] * position[2]),
        -(rotation[2][0] * position[0]
            + rotation[2][1] * position[1]
            + rotation[2][2] * position[2]),
    ];

    Ok(CameraExtrinsics::new(rotation, translation))
}

// ─────────────────────────────────────────────────────────────────────────────
// Small vector/matrix helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn vec3_len(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn transpose3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

#[inline]
fn mat3_vec3_mul(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positional_encoding_output_dim() {
        let enc = PositionalEncoding::new(10);
        // 3D input -> 2 * 10 * 3 = 60
        assert_eq!(enc.output_dim(3), 60);

        let enc4 = PositionalEncoding::new(4);
        assert_eq!(enc4.output_dim(3), 24);
    }

    #[test]
    fn test_positional_encoding_values() {
        let enc = PositionalEncoding::new(2);
        let encoded = enc.encode_scalar(1.0);
        assert_eq!(encoded.len(), 4);
        // Level 0: sin(pi * 1), cos(pi * 1) = sin(pi) ~ 0, cos(pi) = -1
        assert!(encoded[0].abs() < 1e-10, "sin(pi) = {}", encoded[0]);
        assert!((encoded[1] + 1.0).abs() < 1e-10, "cos(pi) = {}", encoded[1]);
        // Level 1: sin(2*pi * 1), cos(2*pi * 1) = 0, 1
        assert!(encoded[2].abs() < 1e-10, "sin(2pi) = {}", encoded[2]);
        assert!(
            (encoded[3] - 1.0).abs() < 1e-10,
            "cos(2pi) = {}",
            encoded[3]
        );
    }

    #[test]
    fn test_positional_encoding_vector() {
        let enc = PositionalEncoding::new(3);
        let encoded = enc.encode(&[0.5, 1.0]);
        assert_eq!(encoded.len(), enc.output_dim(2)); // 2 * 3 * 2 = 12
    }

    #[test]
    fn test_volume_render_opaque_object() {
        // An opaque object at the first sample: high density, red colour
        let sigmas = vec![100.0, 0.0, 0.0, 0.0];
        let colors = vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        let deltas = vec![0.1, 0.1, 0.1, 0.1];

        let result =
            volume_render(&sigmas, &colors, &deltas).expect("volume_render should succeed");
        // Should be close to red
        assert!(result.color[0] > 0.99, "R = {}", result.color[0]);
        assert!(result.color[1] < 0.01, "G = {}", result.color[1]);
        assert!(result.color[2] < 0.01, "B = {}", result.color[2]);
    }

    #[test]
    fn test_volume_render_transmittance_sums() {
        // Uniform density -> weights should sum close to 1 for sufficient samples
        let n = 100;
        let sigma = 2.0;
        let delta = 0.05; // total range = 5.0
        let sigmas = vec![sigma; n];
        let colors: Vec<[f64; 3]> = vec![[0.5, 0.5, 0.5]; n];
        let deltas = vec![delta; n];

        let result =
            volume_render(&sigmas, &colors, &deltas).expect("volume_render should succeed");
        let weight_sum: f64 = result.weights.iter().sum();
        // For sufficient range, opacity should be close to 1
        assert!(
            (weight_sum - 1.0).abs() < 0.01,
            "weight_sum = {}",
            weight_sum
        );
    }

    #[test]
    fn test_volume_render_empty() {
        let result =
            volume_render(&[], &[], &[]).expect("volume_render should succeed for empty input");
        assert_eq!(result.color, [0.0; 3]);
        assert_eq!(result.opacity, 0.0);
    }

    #[test]
    fn test_volume_render_mismatched_lengths() {
        assert!(volume_render(&[1.0], &[], &[0.1]).is_err());
    }

    #[test]
    fn test_ray_generation_center_pixel_direction() {
        // Camera at origin looking along Z
        let intrinsics = CameraIntrinsics::ideal(100.0, 100.0, 50.0, 50.0);
        let extrinsics = CameraExtrinsics::identity();

        let bundle = generate_rays(&intrinsics, &extrinsics, 100, 100);
        assert_eq!(bundle.rays.len(), 10000);

        // Center pixel (50, 50) should have direction approximately [0, 0, 1]
        let center_ray = &bundle.rays[50 * 100 + 50];
        assert!(
            center_ray.direction[2] > 0.99,
            "center ray Z = {}",
            center_ray.direction[2]
        );
        assert!(
            center_ray.direction[0].abs() < 0.01,
            "center ray X = {}",
            center_ray.direction[0]
        );
        assert!(
            center_ray.direction[1].abs() < 0.01,
            "center ray Y = {}",
            center_ray.direction[1]
        );
    }

    #[test]
    fn test_ray_at() {
        let ray = Ray::new([1.0, 2.0, 3.0], [1.0, 0.0, 0.0]);
        let pt = ray.at(5.0);
        assert!((pt[0] - 6.0).abs() < 1e-9);
        assert!((pt[1] - 2.0).abs() < 1e-9);
        assert!((pt[2] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_hierarchical_sampler_coarse() {
        let sampler = HierarchicalSampler::new(64, 128, 2.0, 6.0);
        let coarse = sampler.coarse_samples();
        assert_eq!(coarse.len(), 64);
        // All samples should be in [near, far]
        for &t in &coarse {
            assert!((2.0..=6.0).contains(&t), "t = {}", t);
        }
        // Should be sorted
        for i in 1..coarse.len() {
            assert!(coarse[i] >= coarse[i - 1]);
        }
    }

    #[test]
    fn test_hierarchical_sampler_fine() {
        let sampler = HierarchicalSampler::new(8, 16, 2.0, 6.0);
        let coarse = sampler.coarse_samples();
        // Give more weight to the middle bins
        let weights = vec![0.01, 0.01, 0.5, 1.0, 1.0, 0.5, 0.01, 0.01];
        let fine = sampler
            .fine_samples(&coarse, &weights)
            .expect("fine_samples should succeed");
        assert_eq!(fine.len(), 16);
        // All fine samples should be in [near, far]
        for &t in &fine {
            assert!((2.0..=6.0).contains(&t), "t = {}", t);
        }
    }

    #[test]
    fn test_hierarchical_merge_and_deltas() {
        let sampler = HierarchicalSampler::new(4, 4, 0.0, 4.0);
        let coarse = sampler.coarse_samples();
        let fine = vec![0.3, 1.2, 2.1, 3.5];
        let (merged, deltas) = sampler.merge_and_deltas(&coarse, &fine);
        assert_eq!(merged.len(), deltas.len());
        // Merged should be sorted
        for i in 1..merged.len() {
            assert!(merged[i] >= merged[i - 1]);
        }
        // All deltas should be positive
        for &d in &deltas {
            assert!(d > 0.0, "delta = {}", d);
        }
    }

    #[test]
    fn test_nerf_model_forward() {
        let config = NeRFConfig {
            hidden_dim: 32,
            num_layers: 4,
            skip_layer: 2,
            pos_encoding_levels: 4,
            dir_encoding_levels: 2,
            ..NeRFConfig::default()
        };
        let model = NeRFModel::new(config);
        let (sigma, rgb) = model.forward(&[0.5, 0.5, 0.5], &[0.0, 0.0, 1.0]);

        // Sigma should be non-negative (ReLU)
        assert!(sigma >= 0.0, "sigma = {}", sigma);
        // RGB should be in [0, 1] (sigmoid)
        for (c, &val) in rgb.iter().enumerate().take(3) {
            assert!((0.0..=1.0).contains(&val), "rgb[{}] = {}", c, val);
        }
    }

    #[test]
    fn test_camera_from_look_at() {
        // Camera at (0, 0, -5) looking at origin, up = (0, -1, 0)
        let ext = camera_from_look_at([0.0, 0.0, -5.0], [0.0, 0.0, 0.0], [0.0, -1.0, 0.0])
            .expect("camera_from_look_at should succeed");

        // The camera looks along +Z in camera space; in world space it looks
        // from -5 toward 0, so forward = (0, 0, 1).
        // The rotation's third row should be the forward direction.
        assert!(
            (ext.rotation[2][2] - 1.0).abs() < 1e-9,
            "R[2][2] = {}",
            ext.rotation[2][2]
        );
    }

    #[test]
    fn test_camera_from_look_at_degenerate() {
        // Position == target: should fail
        assert!(camera_from_look_at([0.0; 3], [0.0; 3], [0.0, 1.0, 0.0]).is_err());
    }

    #[test]
    fn test_camera_project_unproject_roundtrip() {
        let intrinsics = CameraIntrinsics::ideal(800.0, 800.0, 320.0, 240.0);
        let extrinsics = CameraExtrinsics::identity();

        let bundle = generate_rays(&intrinsics, &extrinsics, 640, 480);
        // Pick a non-center pixel
        let ray = &bundle.rays[100 * 640 + 200];

        // Evaluate at depth = 5
        let pt3d = ray.at(5.0);

        // Project back through intrinsics
        let px = intrinsics
            .project([pt3d[0], pt3d[1], pt3d[2]])
            .expect("project should succeed");

        // Should be close to pixel (200.5, 100.5) -- the centre of pixel (200, 100)
        assert!(
            (px[0] - 200.5).abs() < 0.6,
            "u = {}, expected ~200.5",
            px[0]
        );
        assert!(
            (px[1] - 100.5).abs() < 0.6,
            "v = {}, expected ~100.5",
            px[1]
        );
    }
}
