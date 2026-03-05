//! Scale-Invariant Feature Transform (SIFT-like) feature detector and descriptor
//!
//! This module implements a SIFT-inspired algorithm for detecting and describing
//! local image features that are invariant to scale, rotation, and illumination changes.
//!
//! # Algorithm Overview
//!
//! 1. **Scale-space construction**: Build a Difference-of-Gaussians (DoG) pyramid across
//!    multiple octaves and scales.
//! 2. **Extrema detection**: Find local minima/maxima in the 3D DoG scale-space (x, y, scale).
//! 3. **Keypoint localization**: Refine extrema positions via quadratic (Taylor) interpolation;
//!    reject low-contrast and edge responses.
//! 4. **Orientation assignment**: Build gradient histograms in a neighbourhood window and pick
//!    dominant orientations (including sub-dominant peaks > 80%).
//! 5. **Descriptor computation**: Around each oriented keypoint, partition a 16×16 patch into
//!    4×4 sub-regions, each contributing an 8-bin gradient histogram → 128-D float vector.
//!
//! # References
//!
//! Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints.
//! *International Journal of Computer Vision*, 60(2), 91–110.

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::{s, Array2, Array3, ArrayView2};
use std::f64::consts::{PI, SQRT_2};

// ─── Public types ────────────────────────────────────────────────────────────

/// A detected keypoint in scale-space.
#[derive(Debug, Clone)]
pub struct Keypoint {
    /// Sub-pixel x coordinate (column, left = 0)
    pub x: f64,
    /// Sub-pixel y coordinate (row, top = 0)
    pub y: f64,
    /// Characteristic scale (σ)
    pub scale: f64,
    /// Dominant orientation in radians \[−π, π\]
    pub orientation: f64,
    /// Detector response strength (absolute DoG value)
    pub response: f64,
    /// Octave index (0 = full-resolution)
    pub octave: i32,
}

/// A SIFT-like descriptor paired with its keypoint.
#[derive(Debug, Clone)]
pub struct SIFTDescriptor {
    /// The associated keypoint
    pub keypoint: Keypoint,
    /// 128-dimensional L2-normalised float descriptor
    pub descriptor: Vec<f32>,
}

/// Configuration for the SIFT-like detector / descriptor.
#[derive(Debug, Clone)]
pub struct SIFTConfig {
    /// Number of octaves in the scale pyramid (0 = auto-detect from image size)
    pub num_octaves: usize,
    /// Number of scale-space samples per octave (excluding boundary blurs)
    pub scales_per_octave: usize,
    /// Base σ for the first Gaussian blur
    pub initial_sigma: f64,
    /// Minimum DoG contrast to accept a keypoint
    pub contrast_threshold: f64,
    /// Harris edge-ratio threshold for curvature test  (r in Lowe 2004)
    pub edge_ratio: f64,
    /// Maximum number of features to retain (0 = unlimited)
    pub max_features: usize,
    /// Half-size of the orientation-histogram smoothing window
    pub ori_hist_smooth: usize,
    /// Number of bins in orientation histogram
    pub ori_bins: usize,
    /// Peak ratio threshold for secondary orientations (0.8 in Lowe)
    pub ori_peak_ratio: f64,
}

impl Default for SIFTConfig {
    fn default() -> Self {
        Self {
            num_octaves: 4,
            scales_per_octave: 3,
            initial_sigma: 1.6,
            contrast_threshold: 0.04,
            edge_ratio: 10.0,
            max_features: 0,
            ori_hist_smooth: 2,
            ori_bins: 36,
            ori_peak_ratio: 0.8,
        }
    }
}

// ─── Top-level entry point ────────────────────────────────────────────────────

/// Detect keypoints and compute SIFT-like 128-D descriptors.
///
/// # Arguments
///
/// * `image` – Grayscale image with values in \[0, 1\] or any positive range
/// * `config` – Detector / descriptor parameters
///
/// # Returns
///
/// Sorted (by response, descending) vector of [`SIFTDescriptor`]s.
pub fn detect_and_describe(
    image: &Array2<f64>,
    config: &SIFTConfig,
) -> Result<Vec<SIFTDescriptor>> {
    let (height, width) = image.dim();
    if height < 8 || width < 8 {
        return Err(VisionError::InvalidParameter(
            "Image must be at least 8×8 pixels".to_string(),
        ));
    }

    // 1. Build DoG pyramid
    let pyramid = build_dog_pyramid(image, config)?;

    // 2. Find extrema across all octaves
    let raw_keypoints = find_scale_space_extrema(&pyramid, config)?;

    // 3. Refine & filter keypoints
    let refined = refine_keypoints(raw_keypoints, &pyramid, config)?;

    // 4. Assign orientations (may yield multiple keypoints per extremum)
    let oriented = assign_orientations(refined, &pyramid, config)?;

    // 5. Compute descriptors
    let mut descriptors = compute_descriptors(oriented, &pyramid, config)?;

    // Sort by response strength, keep top-N if requested
    descriptors.sort_unstable_by(|a, b| {
        b.keypoint
            .response
            .partial_cmp(&a.keypoint.response)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if config.max_features > 0 && descriptors.len() > config.max_features {
        descriptors.truncate(config.max_features);
    }

    Ok(descriptors)
}

// ─── Scale-space / DoG pyramid ───────────────────────────────────────────────

/// One level of the Gaussian scale-space pyramid.
#[derive(Debug)]
struct PyramidLevel {
    octave: usize,
    scale_idx: usize,
    sigma: f64,
    image: Array2<f64>,
}

/// The Difference-of-Gaussians pyramid: per octave, a stack of DoG images.
#[derive(Debug)]
struct DogPyramid {
    /// gaussian[octave][scale] – includes guard levels
    gaussian: Vec<Vec<Array2<f64>>>,
    /// dog[octave][scale]
    dog: Vec<Vec<Array2<f64>>>,
    /// sigma[octave][scale]
    sigmas: Vec<Vec<f64>>,
    /// downsample factor for octave o is 2^o
    num_octaves: usize,
    scales_per_octave: usize,
}

fn build_dog_pyramid(image: &Array2<f64>, config: &SIFTConfig) -> Result<DogPyramid> {
    // Number of Gaussian images per octave = scales_per_octave + 3 (2 guard + 1 for continuity)
    let s = config.scales_per_octave;
    let num_gauss = s + 3;

    // Determine number of octaves
    let (h, w) = image.dim();
    let min_dim = h.min(w) as f64;
    let auto_octaves = (min_dim.log2() - 2.0).floor() as usize;
    let num_octaves = if config.num_octaves == 0 {
        auto_octaves.max(1)
    } else {
        config.num_octaves.min(auto_octaves.max(1))
    };

    let sigma0 = config.initial_sigma;

    let mut gaussian: Vec<Vec<Array2<f64>>> = Vec::with_capacity(num_octaves);
    let mut dog: Vec<Vec<Array2<f64>>> = Vec::with_capacity(num_octaves);
    let mut sigmas: Vec<Vec<f64>> = Vec::with_capacity(num_octaves);

    // Pre-blur base image to assumed camera sigma ≈ 0.5, then blur to sigma0
    let assumed_blur = 0.5_f64;
    let delta_sigma = (sigma0 * sigma0 - assumed_blur * assumed_blur)
        .max(0.0)
        .sqrt();
    let mut base = if delta_sigma > 0.0 {
        gaussian_blur(image, delta_sigma)?
    } else {
        image.to_owned()
    };

    for oct in 0..num_octaves {
        let mut oct_gauss: Vec<Array2<f64>> = Vec::with_capacity(num_gauss);
        let mut oct_sigmas: Vec<f64> = Vec::with_capacity(num_gauss);

        // sigma for scale k inside this octave:
        // σ(oct, k) = σ0 · 2^oct · 2^(k/s)
        let oct_base_sigma = sigma0 * 2.0_f64.powi(oct as i32);

        // The base image is already blurred to oct_base_sigma (σ0 for oct=0,
        // 2·σ0 for oct=1, etc.)
        oct_gauss.push(base.clone());
        oct_sigmas.push(oct_base_sigma);

        for k in 1..num_gauss {
            // incremental sigma between consecutive levels
            let prev_sigma = oct_base_sigma * 2.0_f64.powf((k - 1) as f64 / s as f64);
            let next_sigma = oct_base_sigma * 2.0_f64.powf(k as f64 / s as f64);
            let inc_sigma = (next_sigma * next_sigma - prev_sigma * prev_sigma).sqrt();
            let blurred = gaussian_blur(
                oct_gauss
                    .last()
                    .expect("oct_gauss is non-empty after initial push"),
                inc_sigma,
            )?;
            oct_gauss.push(blurred);
            oct_sigmas.push(next_sigma);
        }

        // Build DoG images for this octave
        let mut oct_dog: Vec<Array2<f64>> = Vec::with_capacity(num_gauss - 1);
        for k in 0..(num_gauss - 1) {
            let diff = &oct_gauss[k + 1] - &oct_gauss[k];
            oct_dog.push(diff);
        }

        // Downsample for next octave: take the image at scale s (= 2·σ0 effective)
        let next_base_img = &oct_gauss[s]; // at σ = 2·σ0 effectively
        if oct + 1 < num_octaves {
            base = downsample_2x(next_base_img);
        }

        gaussian.push(oct_gauss);
        dog.push(oct_dog);
        sigmas.push(oct_sigmas);
    }

    Ok(DogPyramid {
        gaussian,
        dog,
        sigmas,
        num_octaves,
        scales_per_octave: s,
    })
}

// ─── Extrema detection ───────────────────────────────────────────────────────

/// Raw extremum found in scale-space.
#[derive(Debug, Clone)]
struct RawExtrema {
    oct: usize,
    scale: usize, // index into dog[oct], 1..=scales_per_octave (interior scales only)
    row: usize,
    col: usize,
    value: f64,
}

fn find_scale_space_extrema(pyramid: &DogPyramid, config: &SIFTConfig) -> Result<Vec<RawExtrema>> {
    let s = pyramid.scales_per_octave;
    // Threshold on DoG contrast (normalised by number of scales)
    let threshold = 0.5 * config.contrast_threshold / s as f64;

    let mut extrema = Vec::new();

    for oct in 0..pyramid.num_octaves {
        let n_dog = pyramid.dog[oct].len(); // = s + 2
        if n_dog < 3 {
            continue;
        }
        // Interior scales: 1 .. n_dog-1
        for scale in 1..(n_dog - 1) {
            let prev = &pyramid.dog[oct][scale - 1];
            let curr = &pyramid.dog[oct][scale];
            let next = &pyramid.dog[oct][scale + 1];

            let (rows, cols) = curr.dim();
            if rows < 3 || cols < 3 {
                continue;
            }

            for r in 1..(rows - 1) {
                for c in 1..(cols - 1) {
                    let val = curr[[r, c]];
                    if val.abs() <= threshold {
                        continue;
                    }

                    // Check 26 neighbours across 3 DoG layers
                    if is_extremum(val, prev, curr, next, r, c) {
                        extrema.push(RawExtrema {
                            oct,
                            scale,
                            row: r,
                            col: c,
                            value: val,
                        });
                    }
                }
            }
        }
    }

    Ok(extrema)
}

#[inline]
fn is_extremum(
    val: f64,
    prev: &Array2<f64>,
    curr: &Array2<f64>,
    next: &Array2<f64>,
    r: usize,
    c: usize,
) -> bool {
    let check_max = |layer: &Array2<f64>| -> bool {
        for dr in 0..3usize {
            for dc in 0..3usize {
                let nr = r + dr - 1;
                let nc = c + dc - 1;
                if layer[[nr, nc]] >= val {
                    return false;
                }
            }
        }
        true
    };
    let check_min = |layer: &Array2<f64>| -> bool {
        for dr in 0..3usize {
            for dc in 0..3usize {
                let nr = r + dr - 1;
                let nc = c + dc - 1;
                if layer[[nr, nc]] <= val {
                    return false;
                }
            }
        }
        true
    };

    if val > 0.0 {
        check_max(prev) && check_max(next) && {
            // curr without centre
            for dr in 0..3usize {
                for dc in 0..3usize {
                    if dr == 1 && dc == 1 {
                        continue;
                    }
                    let nr = r + dr - 1;
                    let nc = c + dc - 1;
                    if curr[[nr, nc]] >= val {
                        return false;
                    }
                }
            }
            true
        }
    } else {
        check_min(prev) && check_min(next) && {
            for dr in 0..3usize {
                for dc in 0..3usize {
                    if dr == 1 && dc == 1 {
                        continue;
                    }
                    let nr = r + dr - 1;
                    let nc = c + dc - 1;
                    if curr[[nr, nc]] <= val {
                        return false;
                    }
                }
            }
            true
        }
    }
}

// ─── Keypoint refinement ─────────────────────────────────────────────────────

/// Keypoint refined to sub-pixel accuracy and octave-space scale.
#[derive(Debug, Clone)]
struct RefinedKeypoint {
    oct: usize,
    scale: usize,
    /// Row in the octave image
    row: f64,
    /// Col in the octave image
    col: f64,
    /// Scale index (fractional)
    scale_f: f64,
    /// Absolute DoG response after refinement
    response: f64,
    /// σ in pixels of the full-resolution image
    sigma: f64,
}

fn refine_keypoints(
    raw: Vec<RawExtrema>,
    pyramid: &DogPyramid,
    config: &SIFTConfig,
) -> Result<Vec<RefinedKeypoint>> {
    let s = pyramid.scales_per_octave as f64;
    let contrast_thresh = config.contrast_threshold / pyramid.scales_per_octave as f64;
    // Edge test: (r+1)^2/r
    let r = config.edge_ratio;
    let edge_thresh = (r + 1.0) * (r + 1.0) / r;

    let max_iter = 5usize;
    let mut refined = Vec::new();

    'outer: for mut ex in raw {
        let (mut r_idx, mut c_idx) = (ex.row as i64, ex.col as i64);
        let mut scale_idx = ex.scale as i64;

        let mut offset = [0.0f64; 3]; // [Δscale, Δrow, Δcol]

        for _iter in 0..max_iter {
            let n_dog = pyramid.dog[ex.oct].len() as i64;
            let (rows, cols) = pyramid.dog[ex.oct][scale_idx as usize].dim();

            // Bounds check – discard if wandered to edge
            if r_idx <= 0
                || r_idx >= (rows as i64 - 1)
                || c_idx <= 0
                || c_idx >= (cols as i64 - 1)
                || scale_idx <= 0
                || scale_idx >= n_dog - 1
            {
                continue 'outer;
            }

            let (r, c, si) = (r_idx as usize, c_idx as usize, scale_idx as usize);

            let d = dog_derivative(pyramid, ex.oct, si, r, c);
            let h = dog_hessian(pyramid, ex.oct, si, r, c);

            // Solve H·x = −d  via Cramer's rule (3×3)
            let x = solve_3x3(&h, &d);
            offset = x;

            // If offset small enough, accept this position
            if offset[0].abs() < 0.5 && offset[1].abs() < 0.5 && offset[2].abs() < 0.5 {
                break;
            }

            // Move to nearest integer location
            scale_idx += offset[0].round() as i64;
            r_idx += offset[1].round() as i64;
            c_idx += offset[2].round() as i64;
        }

        // Final bounds check
        let n_dog = pyramid.dog[ex.oct].len() as i64;
        let (rows, cols) = pyramid.dog[ex.oct][scale_idx as usize].dim();
        if r_idx <= 0
            || r_idx >= (rows as i64 - 1)
            || c_idx <= 0
            || c_idx >= (cols as i64 - 1)
            || scale_idx <= 0
            || scale_idx >= n_dog - 1
        {
            continue;
        }

        let (r, c, si) = (r_idx as usize, c_idx as usize, scale_idx as usize);

        // Contrast check on interpolated response
        let dog_val = pyramid.dog[ex.oct][si][[r, c]];
        let d = dog_derivative(pyramid, ex.oct, si, r, c);
        let response = dog_val + 0.5 * (d[0] * offset[0] + d[1] * offset[1] + d[2] * offset[2]);

        if response.abs() < contrast_thresh {
            continue;
        }

        // Edge rejection via Hessian trace/det ratio
        let h22 = dog_hessian_2d(pyramid, ex.oct, si, r, c);
        let trace = h22[0][0] + h22[1][1];
        let det = h22[0][0] * h22[1][1] - h22[0][1] * h22[1][0];
        if det <= 0.0 || trace * trace / det >= edge_thresh {
            continue;
        }

        // Sub-pixel correction to scale (sigma)
        let scale_f = (si as f64 + offset[0]) / s;
        let sigma = config.initial_sigma * 2.0_f64.powi(ex.oct as i32) * 2.0_f64.powf(scale_f);

        // ex.oct is unchanged (read below)
        ex.scale = si;

        refined.push(RefinedKeypoint {
            oct: ex.oct,
            scale: si,
            row: r as f64 + offset[1],
            col: c as f64 + offset[2],
            scale_f: si as f64 + offset[0],
            response: response.abs(),
            sigma,
        });
    }

    Ok(refined)
}

/// First-order partial derivatives of DoG (scale, row, col)
fn dog_derivative(pyramid: &DogPyramid, oct: usize, si: usize, r: usize, c: usize) -> [f64; 3] {
    let prev = &pyramid.dog[oct][si - 1];
    let curr = &pyramid.dog[oct][si];
    let next = &pyramid.dog[oct][si + 1];

    let (rows, cols) = curr.dim();
    let r1 = r.min(rows - 1);
    let c1 = c.min(cols - 1);

    let ds = (next[[r1, c1]] - prev[[r1, c1]]) * 0.5;
    let dr = (curr[[r1 + 1, c1]] - curr[[r1 - 1, c1]]) * 0.5;
    let dc = (curr[[r1, c1 + 1]] - curr[[r1, c1 - 1]]) * 0.5;
    [ds, dr, dc]
}

/// Second-order Hessian (3×3) of DoG
fn dog_hessian(pyramid: &DogPyramid, oct: usize, si: usize, r: usize, c: usize) -> [[f64; 3]; 3] {
    let prev = &pyramid.dog[oct][si - 1];
    let curr = &pyramid.dog[oct][si];
    let next = &pyramid.dog[oct][si + 1];

    let v = curr[[r, c]];

    let dss = next[[r, c]] - 2.0 * v + prev[[r, c]];
    let drr = curr[[r + 1, c]] - 2.0 * v + curr[[r - 1, c]];
    let dcc = curr[[r, c + 1]] - 2.0 * v + curr[[r, c - 1]];
    let dsr = (next[[r + 1, c]] - next[[r - 1, c]] - prev[[r + 1, c]] + prev[[r - 1, c]]) * 0.25;
    let dsc = (next[[r, c + 1]] - next[[r, c - 1]] - prev[[r, c + 1]] + prev[[r, c - 1]]) * 0.25;
    let drc = (curr[[r + 1, c + 1]] - curr[[r + 1, c - 1]] - curr[[r - 1, c + 1]]
        + curr[[r - 1, c - 1]])
        * 0.25;

    [[dss, dsr, dsc], [dsr, drr, drc], [dsc, drc, dcc]]
}

/// 2D (spatial only) Hessian for edge test
fn dog_hessian_2d(
    pyramid: &DogPyramid,
    oct: usize,
    si: usize,
    r: usize,
    c: usize,
) -> [[f64; 2]; 2] {
    let curr = &pyramid.dog[oct][si];
    let v = curr[[r, c]];
    let drr = curr[[r + 1, c]] - 2.0 * v + curr[[r - 1, c]];
    let dcc = curr[[r, c + 1]] - 2.0 * v + curr[[r, c - 1]];
    let drc = (curr[[r + 1, c + 1]] - curr[[r + 1, c - 1]] - curr[[r - 1, c + 1]]
        + curr[[r - 1, c - 1]])
        * 0.25;
    [[drr, drc], [drc, dcc]]
}

/// Solve 3×3 linear system A·x = −b using Cramer's rule.
fn solve_3x3(a: &[[f64; 3]; 3], b: &[f64; 3]) -> [f64; 3] {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

    if det.abs() < 1e-10 {
        return [0.0; 3];
    }

    let inv_det = -1.0 / det;

    let x0 = b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
        + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2]);

    let x1 = a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
        - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0]);

    let x2 = a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
        - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
        + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

    [x0 * inv_det, x1 * inv_det, x2 * inv_det]
}

// ─── Orientation assignment ───────────────────────────────────────────────────

fn assign_orientations(
    refined: Vec<RefinedKeypoint>,
    pyramid: &DogPyramid,
    config: &SIFTConfig,
) -> Result<Vec<(RefinedKeypoint, f64)>> {
    let mut result: Vec<(RefinedKeypoint, f64)> = Vec::new();
    let n_bins = config.ori_bins;
    let peak_ratio = config.ori_peak_ratio;

    for kp in refined {
        let sigma = kp.sigma;
        let (gauss_rows, gauss_cols) = pyramid.gaussian[kp.oct][kp.scale].dim();

        // The Gaussian image at this scale for gradient computation
        let gauss_img = &pyramid.gaussian[kp.oct][kp.scale];

        let r = kp.row.round() as i64;
        let c = kp.col.round() as i64;

        // Radius of neighbourhood = 3 × 1.5σ
        let radius = (3.0 * 1.5 * sigma).ceil() as i64;

        let mut hist = vec![0.0f64; n_bins];
        let bin_width = 2.0 * PI / n_bins as f64;

        for dr in -radius..=radius {
            for dc in -radius..=radius {
                let nr = r + dr;
                let nc = c + dc;
                if nr <= 0 || nr >= gauss_rows as i64 - 1 || nc <= 0 || nc >= gauss_cols as i64 - 1
                {
                    continue;
                }
                let (nr, nc) = (nr as usize, nc as usize);

                let gx = gauss_img[[nr, nc + 1]] - gauss_img[[nr, nc - 1]];
                let gy = gauss_img[[nr + 1, nc]] - gauss_img[[nr - 1, nc]];
                let mag = (gx * gx + gy * gy).sqrt();
                let angle = gy.atan2(gx);

                // Gaussian weighting
                let dist2 = (dr * dr + dc * dc) as f64;
                let weight = (-dist2 / (2.0 * (1.5 * sigma) * (1.5 * sigma))).exp();

                // Bin the gradient
                let mut bin = ((angle + PI) / bin_width).floor() as i64;
                bin = bin.rem_euclid(n_bins as i64);
                hist[bin as usize] += mag * weight;
            }
        }

        // Smooth histogram by convolution with [1/6, 4/6, 1/6]^2
        smooth_histogram(&mut hist, config.ori_hist_smooth);

        let max_val = hist.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let thresh = max_val * peak_ratio;

        // Find peaks
        for b in 0..n_bins {
            let prev = hist[(b + n_bins - 1) % n_bins];
            let curr = hist[b];
            let next = hist[(b + 1) % n_bins];
            if curr > thresh && curr >= prev && curr >= next {
                // Parabolic interpolation for sub-bin accuracy
                let denom = prev - 2.0 * curr + next;
                let offset = if denom.abs() < 1e-10 {
                    0.0
                } else {
                    0.5 * (prev - next) / denom
                };
                let angle = (b as f64 + offset + 0.5) * bin_width - PI;
                result.push((kp.clone(), angle));
            }
        }
    }

    Ok(result)
}

/// Smooth histogram in-place with multiple passes of [1/4, 1/2, 1/4]
fn smooth_histogram(hist: &mut [f64], passes: usize) {
    let n = hist.len();
    for _ in 0..passes {
        let old = hist.to_owned();
        for i in 0..n {
            hist[i] = 0.25 * old[(i + n - 1) % n] + 0.5 * old[i] + 0.25 * old[(i + 1) % n];
        }
    }
}

// ─── Descriptor computation ───────────────────────────────────────────────────

/// Compute 128-D SIFT descriptor for each oriented keypoint.
fn compute_descriptors(
    oriented: Vec<(RefinedKeypoint, f64)>,
    pyramid: &DogPyramid,
    _config: &SIFTConfig,
) -> Result<Vec<SIFTDescriptor>> {
    // Descriptor geometry: 4×4 spatial cells × 8 orientation bins = 128-D
    const N_CELLS: usize = 4;
    const N_BINS: usize = 8;
    const DESC_LEN: usize = N_CELLS * N_CELLS * N_BINS; // 128

    let mut out = Vec::with_capacity(oriented.len());

    for (kp, orientation) in oriented {
        let gauss_img = &pyramid.gaussian[kp.oct][kp.scale];
        let (rows, cols) = gauss_img.dim();

        let r0 = kp.row.round() as i64;
        let c0 = kp.col.round() as i64;
        let sigma = kp.sigma;

        // Cell width in octave-space pixels = σ × 3 / N_CELLS × 2
        // Lowe uses window = 2 × 3σ, divided into 4 cells → cell_size = 1.5σ
        let cell_size = 1.5 * sigma;
        let half_win = (cell_size * N_CELLS as f64 * SQRT_2 / 2.0).ceil() as i64 + 1;

        let cos_a = orientation.cos();
        let sin_a = orientation.sin();

        let mut raw_desc = vec![0.0f64; DESC_LEN];

        for dr in -half_win..=half_win {
            for dc in -half_win..=half_win {
                // Rotate sample into keypoint frame
                let rot_r = (cos_a * dr as f64 + sin_a * dc as f64) / cell_size;
                let rot_c = (-sin_a * dr as f64 + cos_a * dc as f64) / cell_size;

                // Map to descriptor grid [−2, 2) × [−2, 2)
                let grid_r = rot_r + N_CELLS as f64 / 2.0 - 0.5;
                let grid_c = rot_c + N_CELLS as f64 / 2.0 - 0.5;

                if grid_r < -1.0
                    || grid_r > N_CELLS as f64
                    || grid_c < -1.0
                    || grid_c > N_CELLS as f64
                {
                    continue;
                }

                let nr = r0 + dr;
                let nc = c0 + dc;
                if nr <= 0 || nr >= rows as i64 - 1 || nc <= 0 || nc >= cols as i64 - 1 {
                    continue;
                }
                let (nr, nc) = (nr as usize, nc as usize);

                let gx = gauss_img[[nr, nc + 1]] - gauss_img[[nr, nc - 1]];
                let gy = gauss_img[[nr + 1, nc]] - gauss_img[[nr - 1, nc]];
                let mag = (gx * gx + gy * gy).sqrt();

                // Rotate gradient to keypoint orientation frame
                let angle = gy.atan2(gx) - orientation;
                // Normalise to [0, 2π)
                let angle = angle.rem_euclid(2.0 * PI);

                // Gaussian weighting with σ = N_CELLS/2 = 2
                let dist2 = rot_r * rot_r + rot_c * rot_c;
                let weight = (-dist2 / (2.0 * (N_CELLS as f64 / 2.0).powi(2))).exp();
                let weighted_mag = mag * weight;

                // Bilinearly distribute into descriptor histogram
                let ri = grid_r.floor() as i64;
                let ci = grid_c.floor() as i64;
                let bi = (angle / (2.0 * PI) * N_BINS as f64).floor() as i64;

                let alpha_r = grid_r - ri as f64;
                let alpha_c = grid_c - ci as f64;
                let alpha_b = (angle / (2.0 * PI) * N_BINS as f64) - bi as f64;

                for (dr_tri, wr) in [(0i64, 1.0 - alpha_r), (1, alpha_r)] {
                    let ri2 = ri + dr_tri;
                    if ri2 < 0 || ri2 >= N_CELLS as i64 {
                        continue;
                    }
                    for (dc_tri, wc) in [(0i64, 1.0 - alpha_c), (1, alpha_c)] {
                        let ci2 = ci + dc_tri;
                        if ci2 < 0 || ci2 >= N_CELLS as i64 {
                            continue;
                        }
                        for (db_tri, wb) in [(0i64, 1.0 - alpha_b), (1, alpha_b)] {
                            let bi2 = (bi + db_tri).rem_euclid(N_BINS as i64) as usize;
                            let idx = (ri2 as usize * N_CELLS + ci2 as usize) * N_BINS + bi2;
                            raw_desc[idx] += weighted_mag * wr * wc * wb;
                        }
                    }
                }
            }
        }

        // L2-normalise, clamp at 0.2, re-normalise (Lowe 2004 §6.1)
        let norm: f64 = raw_desc.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for v in raw_desc.iter_mut() {
                *v /= norm;
            }
        }
        for v in raw_desc.iter_mut() {
            if *v > 0.2 {
                *v = 0.2;
            }
        }
        let norm2: f64 = raw_desc.iter().map(|v| v * v).sum::<f64>().sqrt();
        let descriptor: Vec<f32> = if norm2 > 1e-12 {
            raw_desc.iter().map(|v| (*v / norm2) as f32).collect()
        } else {
            vec![0.0f32; DESC_LEN]
        };

        // Back-project to full-resolution coordinates
        let scale_factor = 2.0_f64.powi(kp.oct as i32);
        let x_full = kp.col * scale_factor;
        let y_full = kp.row * scale_factor;

        out.push(SIFTDescriptor {
            keypoint: Keypoint {
                x: x_full,
                y: y_full,
                scale: kp.sigma * scale_factor,
                orientation,
                response: kp.response,
                octave: kp.oct as i32,
            },
            descriptor,
        });
    }

    Ok(out)
}

// ─── Image processing helpers ─────────────────────────────────────────────────

/// Gaussian blur with specified σ using separable 1-D convolution.
pub(crate) fn gaussian_blur(image: &Array2<f64>, sigma: f64) -> Result<Array2<f64>> {
    if sigma < 1e-6 {
        return Ok(image.to_owned());
    }

    // Build 1-D kernel truncated at 3σ
    let radius = (3.0 * sigma).ceil() as usize;
    let kernel_size = 2 * radius + 1;
    let mut kernel = vec![0.0f64; kernel_size];
    let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
    for (i, val) in kernel.iter_mut().enumerate() {
        let x = i as f64 - radius as f64;
        *val = (-x * x * inv_2s2).exp();
    }
    let sum: f64 = kernel.iter().sum();
    for v in kernel.iter_mut() {
        *v /= sum;
    }

    let (rows, cols) = image.dim();
    let mut tmp = Array2::<f64>::zeros((rows, cols));
    let mut out = Array2::<f64>::zeros((rows, cols));

    // Horizontal pass
    for r in 0..rows {
        for c in 0..cols {
            let mut acc = 0.0;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sc = c as i64 + ki as i64 - radius as i64;
                let sc = sc.clamp(0, cols as i64 - 1) as usize;
                acc += image[[r, sc]] * kv;
            }
            tmp[[r, c]] = acc;
        }
    }

    // Vertical pass
    for r in 0..rows {
        for c in 0..cols {
            let mut acc = 0.0;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sr = r as i64 + ki as i64 - radius as i64;
                let sr = sr.clamp(0, rows as i64 - 1) as usize;
                acc += tmp[[sr, c]] * kv;
            }
            out[[r, c]] = acc;
        }
    }

    Ok(out)
}

/// Downsample by factor 2 using simple subsampling (nearest).
fn downsample_2x(image: &Array2<f64>) -> Array2<f64> {
    let (rows, cols) = image.dim();
    let new_rows = (rows / 2).max(1);
    let new_cols = (cols / 2).max(1);
    let mut out = Array2::<f64>::zeros((new_rows, new_cols));
    for r in 0..new_rows {
        for c in 0..new_cols {
            out[[r, c]] = image[[(r * 2).min(rows - 1), (c * 2).min(cols - 1)]];
        }
    }
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Synthetic image with two Gaussian blobs detectable by SIFT-like DoG.
    /// Blob 1: at (40,40) with σ=3 pixels; Blob 2: at (80,120) with σ=2.5 pixels.
    fn synthetic_image(rows: usize, cols: usize) -> Array2<f64> {
        let mut img = Array2::<f64>::zeros((rows, cols));
        for r in 0..rows {
            for c in 0..cols {
                let d1sq = (r as f64 - 40.0).powi(2) + (c as f64 - 40.0).powi(2);
                let d2sq = (r as f64 - 80.0).powi(2) + (c as f64 - 120.0).powi(2);
                img[[r, c]] = (-d1sq / (2.0 * 9.0)).exp() + 0.7 * (-d2sq / (2.0 * 6.25)).exp();
            }
        }
        img
    }

    #[test]
    fn test_gaussian_blur_identity_low_sigma() {
        let img = synthetic_image(32, 32);
        let blurred = gaussian_blur(&img, 0.0).expect("gaussian_blur with sigma=0 should not fail");
        // identity when sigma ~ 0
        for r in 0..32 {
            for c in 0..32 {
                assert!((img[[r, c]] - blurred[[r, c]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_detect_and_describe_runs() {
        let img = synthetic_image(128, 160);
        let config = SIFTConfig {
            num_octaves: 2,
            scales_per_octave: 3,
            contrast_threshold: 0.01,
            max_features: 20,
            ..Default::default()
        };
        let descs = detect_and_describe(&img, &config)
            .expect("detect_and_describe should succeed on valid image");
        // Should find at least the two blobs
        assert!(!descs.is_empty(), "Expected at least 1 descriptor");
        for d in &descs {
            assert_eq!(d.descriptor.len(), 128);
            // Descriptor should be normalised
            let norm: f32 = d.descriptor.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 0.01 || norm < 1e-5, "norm={norm}");
        }
    }

    #[test]
    fn test_too_small_image() {
        let img = Array2::<f64>::zeros((4, 4));
        let result = detect_and_describe(&img, &SIFTConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_downsample_2x() {
        let img = Array2::<f64>::from_shape_fn((8, 8), |(r, c)| (r + c) as f64);
        let ds = downsample_2x(&img);
        assert_eq!(ds.dim(), (4, 4));
        assert!((ds[[0, 0]] - img[[0, 0]]).abs() < 1e-10);
        assert!((ds[[1, 1]] - img[[2, 2]]).abs() < 1e-10);
    }
}
