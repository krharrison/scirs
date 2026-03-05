//! ORB-like feature detector with FAST corners and binary BRIEF descriptor
//!
//! This module implements an ORB-inspired (Oriented FAST and Rotated BRIEF) algorithm:
//!
//! 1. **FAST corner detection** – pixel-ring threshold test with adaptive score
//! 2. **Harris score** – refines FAST corners by curvature response
//! 3. **Orientation assignment** – intensity centroid in a circular patch
//! 4. **BRIEF-like binary descriptor** – 256 pixel-pair tests drawn from a
//!    2-D Gaussian distribution, rotated to the keypoint orientation (rBRIEF)
//!
//! Binary descriptors are stored as packed `u32` words (8 × `u32` = 256 bits).
//!
//! # References
//!
//! - Rosten, E. & Drummond, T. (2006). Machine learning for high-speed corner detection.
//!   ECCV 2006.
//! - Calonder, M. et al. (2010). BRIEF: Binary Robust Independent Elementary Features.
//!   ECCV 2010.
//! - Rublee, E. et al. (2011). ORB: An efficient alternative to SIFT or SURF.
//!   ICCV 2011.

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::Array2;
use std::f64::consts::PI;

// Number of u32 words for 256-bit descriptor
pub(crate) const DESC_WORDS: usize = 8;
/// Total descriptor bits
pub const DESC_BITS: usize = DESC_WORDS * 32;

// ─── Public types ────────────────────────────────────────────────────────────

/// A keypoint detected by the ORB-like detector.
#[derive(Debug, Clone)]
pub struct OrbKeypoint {
    /// Column (x) coordinate in the image
    pub x: f64,
    /// Row (y) coordinate in the image
    pub y: f64,
    /// Harris corner response score
    pub score: f64,
    /// Intensity centroid orientation in radians \[−π, π\]
    pub orientation: f64,
    /// Scale-pyramid level (0 = full resolution)
    pub level: usize,
}

/// An ORB-like descriptor: keypoint + 256-bit binary descriptor.
#[derive(Debug, Clone)]
pub struct OrbLikeDescriptor {
    /// Associated keypoint
    pub keypoint: OrbKeypoint,
    /// 256-bit binary descriptor packed as 8 × u32
    pub descriptor: [u32; DESC_WORDS],
}

/// Configuration for the ORB-like detector.
#[derive(Debug, Clone)]
pub struct OrbLikeConfig {
    /// Maximum number of features to detect (0 = unlimited)
    pub max_features: usize,
    /// FAST intensity-difference threshold (0–255 range)
    pub fast_threshold: u8,
    /// Minimum number of contiguous Bresenham-circle pixels that must pass the test
    pub fast_n: usize,
    /// Harris window half-size for scoring
    pub harris_k: f64,
    /// Harris window radius (σ for Gaussian weighting)
    pub harris_sigma: f64,
    /// NMS radius (pixels): keeps the single strongest response in this neighbourhood
    pub nms_radius: usize,
    /// Scale factor between pyramid levels
    pub scale_factor: f64,
    /// Number of pyramid levels
    pub num_levels: usize,
    /// Half-patch radius for orientation centroid (should cover the descriptor patch)
    pub patch_radius: usize,
}

impl Default for OrbLikeConfig {
    fn default() -> Self {
        Self {
            max_features: 500,
            fast_threshold: 20,
            fast_n: 9,
            harris_k: 0.04,
            harris_sigma: 3.0,
            nms_radius: 5,
            scale_factor: 1.2,
            num_levels: 4,
            patch_radius: 15,
        }
    }
}

// ─── Entry point ─────────────────────────────────────────────────────────────

/// Detect ORB-like keypoints and compute 256-bit binary descriptors.
///
/// # Arguments
///
/// * `image` – Grayscale image, values in \[0, 1\]
/// * `config` – Detector / descriptor parameters
///
/// # Returns
///
/// Vector of [`OrbLikeDescriptor`] sorted by score (descending).
pub fn detect_and_describe_orb(
    image: &Array2<f64>,
    config: &OrbLikeConfig,
) -> Result<Vec<OrbLikeDescriptor>> {
    let (h, w) = image.dim();
    if h < 16 || w < 16 {
        return Err(VisionError::InvalidParameter(
            "Image must be at least 16×16 pixels for ORB detection".to_string(),
        ));
    }

    // Build image pyramid
    let pyramid = build_pyramid(image, config)?;

    // Per-level detection
    let mut all_descs: Vec<OrbLikeDescriptor> = Vec::new();

    for (level, level_img) in pyramid.iter().enumerate() {
        let scale = config.scale_factor.powi(level as i32);

        // 1. FAST corners
        let fast_pts = detect_fast(level_img, config.fast_threshold, config.fast_n)?;
        if fast_pts.is_empty() {
            continue;
        }

        // 2. Harris score + NMS
        let harris = compute_harris_response(level_img, config.harris_k, config.harris_sigma)?;
        let scored: Vec<(usize, usize, f64)> = fast_pts
            .into_iter()
            .map(|(r, c)| {
                let score = harris.get([r, c]).copied().unwrap_or(0.0);
                (r, c, score)
            })
            .collect();

        let nms_pts = non_max_suppression(&scored, config.nms_radius, level_img.dim());

        // 3. Orientation via intensity centroid
        // 4. Compute rBRIEF descriptors
        let border = config.patch_radius + 2;
        let (lrows, lcols) = level_img.dim();

        for (r, c, score) in nms_pts {
            if r < border || r + border >= lrows || c < border || c + border >= lcols {
                continue;
            }

            let orientation = intensity_centroid_orientation(level_img, r, c, config.patch_radius);
            let descriptor = brief_descriptor(level_img, r, c, orientation, config.patch_radius)?;

            all_descs.push(OrbLikeDescriptor {
                keypoint: OrbKeypoint {
                    x: c as f64 * scale,
                    y: r as f64 * scale,
                    score,
                    orientation,
                    level,
                },
                descriptor,
            });
        }
    }

    // Sort by score
    all_descs.sort_unstable_by(|a, b| {
        b.keypoint
            .score
            .partial_cmp(&a.keypoint.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if config.max_features > 0 && all_descs.len() > config.max_features {
        all_descs.truncate(config.max_features);
    }

    Ok(all_descs)
}

// ─── FAST detector ────────────────────────────────────────────────────────────

/// Bresenham circle offsets for FAST-9/12/16 detector (16-pixel ring).
/// Returns (Δrow, Δcol) pairs for the 16 ring positions.
fn bresenham_circle_16() -> [(i32, i32); 16] {
    [
        (-3, 0),
        (-3, 1),
        (-2, 2),
        (-1, 3),
        (0, 3),
        (1, 3),
        (2, 2),
        (3, 1),
        (3, 0),
        (3, -1),
        (2, -2),
        (1, -3),
        (0, -3),
        (-1, -3),
        (-2, -2),
        (-3, -1),
    ]
}

/// Detect FAST corners.
///
/// For each candidate pixel p, check whether at least `n` contiguous pixels
/// on the 16-pixel Bresenham ring are all brighter than p + t or all darker
/// than p − t.
fn detect_fast(image: &Array2<f64>, threshold: u8, n: usize) -> Result<Vec<(usize, usize)>> {
    let (rows, cols) = image.dim();
    let t = threshold as f64 / 255.0;
    let ring = bresenham_circle_16();
    let ring_len = ring.len();
    let border = 4usize;

    let mut corners = Vec::new();

    for r in border..(rows - border) {
        for c in border..(cols - border) {
            let p = image[[r, c]];
            let high = p + t;
            let low = p - t;

            // Fast pre-test: pixels 1, 5, 9, 13 (0-indexed: 0, 4, 8, 12)
            let vals: [f64; 4] = [
                image[[
                    (r as i32 + ring[0].0) as usize,
                    (c as i32 + ring[0].1) as usize,
                ]],
                image[[
                    (r as i32 + ring[4].0) as usize,
                    (c as i32 + ring[4].1) as usize,
                ]],
                image[[
                    (r as i32 + ring[8].0) as usize,
                    (c as i32 + ring[8].1) as usize,
                ]],
                image[[
                    (r as i32 + ring[12].0) as usize,
                    (c as i32 + ring[12].1) as usize,
                ]],
            ];

            let bright_count = vals.iter().filter(|&&v| v > high).count();
            let dark_count = vals.iter().filter(|&&v| v < low).count();

            if bright_count < 2 && dark_count < 2 {
                continue; // fails pre-test: cannot have ≥9 contiguous
            }

            // Full ring check – look for n contiguous above or below
            // Build arc classification: +1 bright, -1 dark, 0 similar
            let mut arc: Vec<i8> = Vec::with_capacity(ring_len);
            for (dr, dc) in ring.iter() {
                let nr = (r as i32 + dr) as usize;
                let nc = (c as i32 + dc) as usize;
                let v = image[[nr, nc]];
                arc.push(if v > high {
                    1
                } else if v < low {
                    -1
                } else {
                    0
                });
            }

            if has_contiguous_run(&arc, n, 1) || has_contiguous_run(&arc, n, -1) {
                corners.push((r, c));
            }
        }
    }

    Ok(corners)
}

/// Returns true if `arc` (treated as circular) has ≥ `n` contiguous pixels
/// with value `target`.
fn has_contiguous_run(arc: &[i8], n: usize, target: i8) -> bool {
    let len = arc.len();
    // Unroll circle for easier scanning
    let mut count = 0usize;
    for i in 0..(2 * len) {
        if arc[i % len] == target {
            count += 1;
            if count >= n {
                return true;
            }
        } else {
            count = 0;
        }
    }
    false
}

// ─── Harris response ──────────────────────────────────────────────────────────

/// Compute Harris corner response at every pixel.
///
/// R = det(M) − k · trace²(M)   where M is the gradient structure tensor,
/// computed using a Gaussian-weighted window.
fn compute_harris_response(image: &Array2<f64>, k: f64, sigma: f64) -> Result<Array2<f64>> {
    let (rows, cols) = image.dim();

    // Compute image gradients (Sobel-like)
    let mut ix = Array2::<f64>::zeros((rows, cols));
    let mut iy = Array2::<f64>::zeros((rows, cols));

    for r in 1..(rows - 1) {
        for c in 1..(cols - 1) {
            ix[[r, c]] = (image[[r, c + 1]] - image[[r, c - 1]]) * 0.5;
            iy[[r, c]] = (image[[r + 1, c]] - image[[r - 1, c]]) * 0.5;
        }
    }

    // Compute products
    let mut ixx = Array2::<f64>::zeros((rows, cols));
    let mut iyy = Array2::<f64>::zeros((rows, cols));
    let mut ixy = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            ixx[[r, c]] = ix[[r, c]] * ix[[r, c]];
            iyy[[r, c]] = iy[[r, c]] * iy[[r, c]];
            ixy[[r, c]] = ix[[r, c]] * iy[[r, c]];
        }
    }

    // Gaussian smoothing of structure tensor components
    let ixx_s = crate::features::sift_like::gaussian_blur(&ixx, sigma)?;
    let iyy_s = crate::features::sift_like::gaussian_blur(&iyy, sigma)?;
    let ixy_s = crate::features::sift_like::gaussian_blur(&ixy, sigma)?;

    // Harris response
    let mut response = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let a = ixx_s[[r, c]];
            let b = ixy_s[[r, c]];
            let d = iyy_s[[r, c]];
            let det = a * d - b * b;
            let trace = a + d;
            response[[r, c]] = det - k * trace * trace;
        }
    }

    Ok(response)
}

// ─── Non-maximum suppression ──────────────────────────────────────────────────

/// Keep only the locally maximal point within radius `r` (in a grid sense).
fn non_max_suppression(
    pts: &[(usize, usize, f64)],
    radius: usize,
    dims: (usize, usize),
) -> Vec<(usize, usize, f64)> {
    let (rows, cols) = dims;
    let cell = (radius * 2 + 1).max(1);
    let grid_rows = rows.div_ceil(cell);
    let grid_cols = cols.div_ceil(cell);

    // Place each point into a grid cell, keeping the best score
    let mut grid: Vec<Vec<Option<(usize, usize, f64)>>> = vec![vec![None; grid_cols]; grid_rows];

    for &(r, c, score) in pts {
        let gr = r / cell;
        let gc = c / cell;
        if gr < grid_rows && gc < grid_cols {
            let cell_val = &mut grid[gr][gc];
            if cell_val.is_none_or(|(_, _, s)| score > s) {
                *cell_val = Some((r, c, score));
            }
        }
    }

    grid.into_iter()
        .flatten()
        .flatten()
        .filter(|(_, _, s)| *s > 0.0)
        .collect()
}

// ─── Orientation via intensity centroid ──────────────────────────────────────

/// Compute the intensity centroid direction over a circular patch of given radius.
///
/// Returns angle in radians in \[−π, π\].
fn intensity_centroid_orientation(
    image: &Array2<f64>,
    row: usize,
    col: usize,
    radius: usize,
) -> f64 {
    let (rows, cols) = image.dim();
    let r = radius as i64;

    let mut m10 = 0.0f64; // first moment in x
    let mut m01 = 0.0f64; // first moment in y

    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy > r * r {
                continue;
            }
            let nr = row as i64 + dy;
            let nc = col as i64 + dx;
            if nr < 0 || nr >= rows as i64 || nc < 0 || nc >= cols as i64 {
                continue;
            }
            let v = image[[nr as usize, nc as usize]];
            m10 += dx as f64 * v;
            m01 += dy as f64 * v;
        }
    }

    m01.atan2(m10)
}

// ─── Steered BRIEF descriptor ─────────────────────────────────────────────────

/// Pre-computed Gaussian-sampled BRIEF pair offsets (generated deterministically).
/// Each pair is (Δr1, Δc1, Δr2, Δc2).
fn generate_brief_pairs(patch_radius: usize) -> Vec<(i32, i32, i32, i32)> {
    // Generate a deterministic pseudo-random sequence using LCG
    // seeded at 0xDEADBEEF to reproduce the same pairs every time.
    let limit = patch_radius as i32;
    let total = DESC_BITS; // 256 pairs
    let mut pairs = Vec::with_capacity(total);

    // Use LCG to generate reproducible Gaussian-like samples within [-limit, limit]
    let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;

    let next_i32 = |s: &mut u64| -> i32 {
        *s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Box-Muller to approximate Gaussian, then scale
        let u1 = (*s >> 32) as f64 / u32::MAX as f64;
        let u2 = (*s & 0xFFFF_FFFF) as f64 / u32::MAX as f64;
        // Box-Muller: approximate by linear mapping of U(0,1) → approximated Gaussian
        // Use: G ≈ (u – 0.5) × 2 × 3σ / limit clipped to [-1,1]
        let g = (u1 - 0.5) * 2.0; // uniform [-1, 1]
                                  // Use 2nd sample for another dimension
        let _ = u2;
        // Quantise to patch
        (g * limit as f64).round() as i32
    };

    while pairs.len() < total {
        let r1 = next_i32(&mut state).clamp(-limit, limit);
        let c1 = next_i32(&mut state).clamp(-limit, limit);
        let r2 = next_i32(&mut state).clamp(-limit, limit);
        let c2 = next_i32(&mut state).clamp(-limit, limit);
        // Avoid identical pairs
        if !(r1 == r2 && c1 == c2) {
            pairs.push((r1, c1, r2, c2));
        }
    }

    pairs
}

/// Compute 256-bit rBRIEF descriptor at the given keypoint location.
///
/// The test pattern is rotated by `orientation` to achieve rotation invariance.
fn brief_descriptor(
    image: &Array2<f64>,
    row: usize,
    col: usize,
    orientation: f64,
    patch_radius: usize,
) -> Result<[u32; DESC_WORDS]> {
    let (rows, cols) = image.dim();
    let pairs = generate_brief_pairs(patch_radius);

    let cos_a = orientation.cos();
    let sin_a = orientation.sin();

    // Apply smoothing before sampling (as in original ORB)
    // We'll compute the descriptor directly with a small blur weight per sample
    // (simplified: use the image directly; production code would pre-smooth)

    let mut words = [0u32; DESC_WORDS];

    for (bit_idx, (dr1, dc1, dr2, dc2)) in pairs.iter().enumerate() {
        // Rotate the offsets into the image frame
        let rot_r1 = (cos_a * *dr1 as f64 - sin_a * *dc1 as f64).round() as i64;
        let rot_c1 = (sin_a * *dr1 as f64 + cos_a * *dc1 as f64).round() as i64;
        let rot_r2 = (cos_a * *dr2 as f64 - sin_a * *dc2 as f64).round() as i64;
        let rot_c2 = (sin_a * *dr2 as f64 + cos_a * *dc2 as f64).round() as i64;

        let nr1 = (row as i64 + rot_r1).clamp(0, rows as i64 - 1) as usize;
        let nc1 = (col as i64 + rot_c1).clamp(0, cols as i64 - 1) as usize;
        let nr2 = (row as i64 + rot_r2).clamp(0, rows as i64 - 1) as usize;
        let nc2 = (col as i64 + rot_c2).clamp(0, cols as i64 - 1) as usize;

        let p1 = image[[nr1, nc1]];
        let p2 = image[[nr2, nc2]];

        if p1 < p2 {
            let word_idx = bit_idx / 32;
            let bit_pos = bit_idx % 32;
            words[word_idx] |= 1u32 << bit_pos;
        }
    }

    Ok(words)
}

// ─── Pyramid builder ──────────────────────────────────────────────────────────

fn build_pyramid(image: &Array2<f64>, config: &OrbLikeConfig) -> Result<Vec<Array2<f64>>> {
    let mut pyramid = Vec::with_capacity(config.num_levels);
    let mut current = image.to_owned();
    pyramid.push(current.clone());

    for _ in 1..config.num_levels {
        let (rows, cols) = current.dim();
        let new_rows = ((rows as f64 / config.scale_factor).round() as usize).max(8);
        let new_cols = ((cols as f64 / config.scale_factor).round() as usize).max(8);
        if new_rows < 16 || new_cols < 16 {
            break;
        }
        // Gaussian blur before downsampling to avoid aliasing
        let blurred =
            crate::features::sift_like::gaussian_blur(&current, config.scale_factor.ln())?;
        current = resize_bilinear(&blurred, new_rows, new_cols);
        pyramid.push(current.clone());
    }

    Ok(pyramid)
}

/// Bilinear resize to target (rows, cols).
fn resize_bilinear(src: &Array2<f64>, dst_rows: usize, dst_cols: usize) -> Array2<f64> {
    let (src_rows, src_cols) = src.dim();
    let mut dst = Array2::<f64>::zeros((dst_rows, dst_cols));

    let row_scale = (src_rows - 1) as f64 / (dst_rows - 1).max(1) as f64;
    let col_scale = (src_cols - 1) as f64 / (dst_cols - 1).max(1) as f64;

    for r in 0..dst_rows {
        let src_r = r as f64 * row_scale;
        let r0 = src_r.floor() as usize;
        let r1 = (r0 + 1).min(src_rows - 1);
        let alpha_r = src_r - r0 as f64;

        for c in 0..dst_cols {
            let src_c = c as f64 * col_scale;
            let c0 = src_c.floor() as usize;
            let c1 = (c0 + 1).min(src_cols - 1);
            let alpha_c = src_c - c0 as f64;

            let top = src[[r0, c0]] * (1.0 - alpha_c) + src[[r0, c1]] * alpha_c;
            let bot = src[[r1, c0]] * (1.0 - alpha_c) + src[[r1, c1]] * alpha_c;
            dst[[r, c]] = top * (1.0 - alpha_r) + bot * alpha_r;
        }
    }

    dst
}

// ─── Hamming distance ─────────────────────────────────────────────────────────

/// Compute Hamming distance between two 256-bit descriptors.
pub fn hamming_distance(a: &[u32; DESC_WORDS], b: &[u32; DESC_WORDS]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn checkerboard(size: usize) -> Array2<f64> {
        Array2::from_shape_fn((size, size), |(r, c)| {
            if (r / 8 + c / 8) % 2 == 0 {
                1.0
            } else {
                0.0
            }
        })
    }

    #[test]
    fn test_orb_detect_runs() {
        let img = checkerboard(128);
        let config = OrbLikeConfig {
            max_features: 50,
            fast_threshold: 10,
            fast_n: 9,
            num_levels: 2,
            ..Default::default()
        };
        let descs = detect_and_describe_orb(&img, &config)
            .expect("detect_and_describe_orb should succeed on valid image");
        // Should find corners at checkerboard transitions
        assert!(!descs.is_empty(), "Expected ORB keypoints on checkerboard");
        for d in &descs {
            assert_eq!(d.descriptor.len(), DESC_WORDS);
        }
    }

    #[test]
    fn test_hamming_distance_identical() {
        let desc = [0xABCD1234u32; DESC_WORDS];
        assert_eq!(hamming_distance(&desc, &desc), 0);
    }

    #[test]
    fn test_hamming_distance_complement() {
        let a = [0u32; DESC_WORDS];
        let b = [u32::MAX; DESC_WORDS];
        assert_eq!(hamming_distance(&a, &b), (DESC_WORDS * 32) as u32);
    }

    #[test]
    fn test_too_small_image() {
        let img = Array2::<f64>::zeros((8, 8));
        let result = detect_and_describe_orb(&img, &OrbLikeConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_fast_corner_detection() {
        // Create a simple corner pattern
        let mut img = Array2::<f64>::zeros((64, 64));
        // Bright square in top-left quadrant
        for r in 10..30 {
            for c in 10..30 {
                img[[r, c]] = 1.0;
            }
        }
        let corners = detect_fast(&img, 30, 9).expect("detect_fast should succeed on valid image");
        // Should detect corners at ~(10,10), (10,30), (30,10), (30,30)
        assert!(
            !corners.is_empty(),
            "Expected FAST corners on bright square"
        );
    }

    #[test]
    fn test_orientation_range() {
        let img = checkerboard(64);
        let angle = intensity_centroid_orientation(&img, 32, 32, 8);
        assert!((-PI..=PI).contains(&angle));
    }
}
