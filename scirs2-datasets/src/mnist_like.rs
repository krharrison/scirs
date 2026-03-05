//! MNIST-like and CIFAR-10-like synthetic image dataset generators.
//!
//! This module generates synthetic benchmark image datasets:
//!
//! - [`MnistLike`]              – Programmatically rendered 28×28 digit images (0–9).
//! - [`Cifar10Like`]            – 32×32 three-channel (RGB) class-pattern images.
//! - [`generate_shapes_dataset`] – 2-D shape classification dataset (circle, square,
//!                                  triangle, ellipse) of arbitrary size.
//!
//! All generators are fully deterministic given a seed and produce values in [0, 1].

use crate::error::{DatasetsError, Result};
use scirs2_core::ndarray::{Array1, Array2, Array3, Array4};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::Distribution;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// Clamp a pixel value to [0.0, 1.0].
#[inline]
fn clamp01(v: f64) -> f64 {
    v.max(0.0).min(1.0)
}

/// Draw a horizontal line segment (inclusive both ends) into a mutable flat pixel buffer
/// (row-major, `cols` columns).  Anti-aliased via simple coverage at endpoints.
fn draw_hline(buf: &mut [f64], cols: usize, row: usize, c0: usize, c1: usize, intensity: f64) {
    let row_off = row * cols;
    for c in c0..=c1 {
        if c < cols {
            buf[row_off + c] = clamp01(buf[row_off + c] + intensity);
        }
    }
}

/// Draw a filled circle (anti-aliased border via smooth step) into a pixel buffer.
fn draw_filled_circle(
    buf: &mut [f64],
    rows: usize,
    cols: usize,
    cy: f64,
    cx: f64,
    radius: f64,
    intensity: f64,
) {
    let r0 = ((cy - radius - 1.0).floor().max(0.0) as usize).min(rows);
    let r1 = ((cy + radius + 1.0).ceil() as usize).min(rows);
    let c0 = ((cx - radius - 1.0).floor().max(0.0) as usize).min(cols);
    let c1 = ((cx + radius + 1.0).ceil() as usize).min(cols);

    for r in r0..r1 {
        for c in c0..c1 {
            let dy = r as f64 - cy;
            let dx = c as f64 - cx;
            let dist = (dy * dy + dx * dx).sqrt();
            // smooth-step coverage: 1 when dist <= radius-0.5, 0 when dist >= radius+0.5
            let coverage = (radius + 0.5 - dist).min(1.0).max(0.0);
            buf[r * cols + c] = clamp01(buf[r * cols + c] + intensity * coverage);
        }
    }
}

/// Draw an unfilled circle (ring) of given `stroke_width` into the buffer.
fn draw_ring(
    buf: &mut [f64],
    rows: usize,
    cols: usize,
    cy: f64,
    cx: f64,
    radius: f64,
    stroke: f64,
    intensity: f64,
) {
    let half = stroke / 2.0;
    let r0 = ((cy - radius - stroke - 1.0).floor().max(0.0) as usize).min(rows);
    let r1 = ((cy + radius + stroke + 1.0).ceil() as usize).min(rows);
    let c0 = ((cx - radius - stroke - 1.0).floor().max(0.0) as usize).min(cols);
    let c1 = ((cx + radius + stroke + 1.0).ceil() as usize).min(cols);

    for r in r0..r1 {
        for c in c0..c1 {
            let dy = r as f64 - cy;
            let dx = c as f64 - cx;
            let dist = (dy * dy + dx * dx).sqrt();
            let d_from_ring = (dist - radius).abs();
            // smooth-step coverage for the stroke band
            let coverage = (half + 0.5 - d_from_ring).min(1.0).max(0.0);
            buf[r * cols + c] = clamp01(buf[r * cols + c] + intensity * coverage);
        }
    }
}

/// Draw a line segment between two float coordinates using anti-aliased Wu algorithm.
fn draw_line(
    buf: &mut [f64],
    rows: usize,
    cols: usize,
    y0: f64,
    x0: f64,
    y1: f64,
    x1: f64,
    stroke: f64,
    intensity: f64,
) {
    // Rasterize using the parametric approach with perpendicular distance
    let dy = y1 - y0;
    let dx = x1 - x0;
    let len = (dy * dy + dx * dx).sqrt();
    if len < 1e-9 {
        // degenerate: draw a small dot
        draw_filled_circle(buf, rows, cols, y0, x0, stroke / 2.0, intensity);
        return;
    }
    let nx = -dy / len; // perpendicular normal
    let ny = dx / len;
    let half_stroke = stroke / 2.0 + 0.5;

    // Bounding box
    let min_y = y0.min(y1) - half_stroke;
    let max_y = y0.max(y1) + half_stroke;
    let min_x = x0.min(x1) - half_stroke;
    let max_x = x0.max(x1) + half_stroke;

    let r0 = (min_y.floor().max(0.0) as usize).min(rows);
    let r1 = (max_y.ceil() as usize).min(rows);
    let c0 = (min_x.floor().max(0.0) as usize).min(cols);
    let c1 = (max_x.ceil() as usize).min(cols);

    for r in r0..r1 {
        for c in c0..c1 {
            let pr = r as f64 - y0;
            let pc = c as f64 - x0;
            // projection onto line direction (clamp to segment)
            let t = ((pr * dy + pc * dx) / (len * len)).max(0.0).min(1.0);
            // closest point on segment
            let closest_r = y0 + t * dy;
            let closest_c = x0 + t * dx;
            let dist = ((r as f64 - closest_r).powi(2) + (c as f64 - closest_c).powi(2)).sqrt();
            let coverage = (stroke / 2.0 + 0.5 - dist).min(1.0).max(0.0);
            buf[r * cols + c] = clamp01(buf[r * cols + c] + intensity * coverage);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-digit programmatic renderers (28×28)
// ─────────────────────────────────────────────────────────────────────────────

/// Render digit `d` (0–9) into a 28×28 buffer using basic geometric primitives.
/// Returns a flat pixel buffer (row-major, 784 elements) with values in [0, 1].
fn render_digit(d: u8) -> Vec<f64> {
    let rows = 28usize;
    let cols = 28usize;
    let mut buf = vec![0.0_f64; rows * cols];

    // Shared metrics
    let cy = 13.5_f64; // vertical centre
    let cx = 13.5_f64; // horizontal centre
    let stroke = 2.0_f64;
    let intensity = 1.0_f64;

    match d {
        0 => {
            // Elliptical ring
            let ry = 10.0_f64;
            let rx = 7.0_f64;
            // Draw using many small lines along the ellipse perimeter
            let steps = 120usize;
            for k in 0..steps {
                let t0 = 2.0 * PI * k as f64 / steps as f64;
                let t1 = 2.0 * PI * (k + 1) as f64 / steps as f64;
                let y0 = cy + ry * t0.sin();
                let x0 = cx + rx * t0.cos();
                let y1 = cy + ry * t1.sin();
                let x1 = cx + rx * t1.cos();
                draw_line(&mut buf, rows, cols, y0, x0, y1, x1, stroke, intensity);
            }
        }
        1 => {
            // Vertical bar slightly right of centre
            draw_line(&mut buf, rows, cols, 4.0, cx + 1.0, 23.0, cx + 1.0, stroke, intensity);
            // Small serif top-left
            draw_line(&mut buf, rows, cols, 4.0, cx - 3.0, 8.0, cx + 1.0, stroke * 0.8, intensity);
        }
        2 => {
            // Top arc (upper semicircle)
            let steps = 60usize;
            for k in 0..steps {
                let t0 = PI + PI * k as f64 / steps as f64;
                let t1 = PI + PI * (k + 1) as f64 / steps as f64;
                draw_line(
                    &mut buf,
                    rows,
                    cols,
                    cy - 5.0 + 5.5 * t0.sin(),
                    cx + 5.5 * t0.cos(),
                    cy - 5.0 + 5.5 * t1.sin(),
                    cx + 5.5 * t1.cos(),
                    stroke,
                    intensity,
                );
            }
            // Diagonal stroke
            draw_line(&mut buf, rows, cols, cy - 0.5, cx + 5.5, 22.0, cx - 6.0, stroke, intensity);
            // Bottom horizontal bar
            draw_line(&mut buf, rows, cols, 22.0, cx - 6.0, 22.0, cx + 6.0, stroke, intensity);
        }
        3 => {
            // Two arcs (upper and lower) facing right
            for half in 0..2usize {
                let arc_cy = if half == 0 { 10.0 } else { 18.0 };
                let steps = 60usize;
                for k in 0..steps {
                    let t0 = -PI / 2.0 + PI * k as f64 / steps as f64;
                    let t1 = -PI / 2.0 + PI * (k + 1) as f64 / steps as f64;
                    draw_line(
                        &mut buf,
                        rows,
                        cols,
                        arc_cy + 5.5 * t0.sin(),
                        cx + 5.5 * t0.cos(),
                        arc_cy + 5.5 * t1.sin(),
                        cx + 5.5 * t1.cos(),
                        stroke,
                        intensity,
                    );
                }
            }
            // Middle horizontal line
            draw_line(&mut buf, rows, cols, 14.0, cx - 3.0, 14.0, cx + 4.0, stroke * 0.7, intensity);
        }
        4 => {
            // Left vertical (upper half)
            draw_line(&mut buf, rows, cols, 4.0, cx - 5.0, 16.0, cx - 5.0, stroke, intensity);
            // Horizontal bar
            draw_line(&mut buf, rows, cols, 16.0, cx - 6.0, 16.0, cx + 6.0, stroke, intensity);
            // Right vertical (full height)
            draw_line(&mut buf, rows, cols, 4.0, cx + 3.0, 23.0, cx + 3.0, stroke, intensity);
        }
        5 => {
            // Top horizontal bar
            draw_line(&mut buf, rows, cols, 4.0, cx - 6.0, 4.0, cx + 5.0, stroke, intensity);
            // Left vertical (upper half)
            draw_line(&mut buf, rows, cols, 4.0, cx - 6.0, 14.0, cx - 6.0, stroke, intensity);
            // Middle bar
            draw_line(&mut buf, rows, cols, 14.0, cx - 6.0, 14.0, cx + 4.0, stroke, intensity);
            // Lower-right arc
            let arc_cy = 18.5_f64;
            let steps = 80usize;
            for k in 0..steps {
                let t0 = -PI * k as f64 / steps as f64;
                let t1 = -PI * (k + 1) as f64 / steps as f64;
                draw_line(
                    &mut buf,
                    rows,
                    cols,
                    arc_cy + 5.5 * t0.sin(),
                    cx + 4.0 + 5.5 * t0.cos(),
                    arc_cy + 5.5 * t1.sin(),
                    cx + 4.0 + 5.5 * t1.cos(),
                    stroke,
                    intensity,
                );
            }
        }
        6 => {
            // Left arc (almost full circle, open top-right)
            let steps = 100usize;
            for k in 0..steps {
                let t0 = PI / 3.0 + (2.0 * PI - PI / 3.0) * k as f64 / steps as f64;
                let t1 = PI / 3.0 + (2.0 * PI - PI / 3.0) * (k + 1) as f64 / steps as f64;
                draw_line(
                    &mut buf,
                    rows,
                    cols,
                    cy + 8.0 * t0.sin(),
                    cx + 6.0 * t0.cos(),
                    cy + 8.0 * t1.sin(),
                    cx + 6.0 * t1.cos(),
                    stroke,
                    intensity,
                );
            }
            // Bottom small loop
            draw_filled_circle(&mut buf, rows, cols, cy + 3.0, cx, 4.5, 0.0);
            draw_ring(&mut buf, rows, cols, cy + 3.0, cx, 4.5, stroke, intensity);
        }
        7 => {
            // Top horizontal bar
            draw_line(&mut buf, rows, cols, 4.0, cx - 6.0, 4.0, cx + 6.0, stroke, intensity);
            // Diagonal down-left
            draw_line(&mut buf, rows, cols, 4.0, cx + 6.0, 23.0, cx - 2.0, stroke, intensity);
        }
        8 => {
            // Two vertically stacked circles
            for (circle_cy, r) in [(10.0, 5.0), (19.0, 6.0)] {
                draw_ring(&mut buf, rows, cols, circle_cy, cx, r, stroke, intensity);
            }
        }
        9 => {
            // Upper circle
            draw_ring(&mut buf, rows, cols, cy - 4.0, cx, 6.0, stroke, intensity);
            // Right vertical line descending from circle
            draw_line(&mut buf, rows, cols, cy + 2.0, cx + 5.5, 23.0, cx + 5.5, stroke, intensity);
        }
        _ => {
            // Fallback: diagonal cross
            draw_line(&mut buf, rows, cols, 4.0, 4.0, 23.0, 23.0, stroke, intensity);
            draw_line(&mut buf, rows, cols, 4.0, 23.0, 23.0, 4.0, stroke, intensity);
        }
    }

    buf
}

// ─────────────────────────────────────────────────────────────────────────────
// MnistLike
// ─────────────────────────────────────────────────────────────────────────────

/// Generator for MNIST-like 28×28 digit images.
///
/// Each digit (0–9) is rendered programmatically using geometric primitives
/// (lines, rings, arcs) and optionally corrupted with Gaussian noise.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::mnist_like::MnistLike;
///
/// let gen = MnistLike::new(0.05);
/// let img = gen.generate_digit(5, 42).expect("digit");
/// assert_eq!(img.shape(), &[28, 28]);
///
/// let (images, labels) = gen.generate_dataset(10, 42).expect("dataset");
/// assert_eq!(images.shape(), &[100, 28, 28]);  // 10 per class × 10 classes
/// assert_eq!(labels.len(), 100);
/// ```
pub struct MnistLike {
    /// Standard deviation of additive Gaussian noise (≥ 0).
    pub noise: f64,
}

impl MnistLike {
    /// Create a new `MnistLike` generator.
    ///
    /// # Arguments
    ///
    /// * `noise` – Gaussian noise std-dev applied on top of the clean rendering (≥ 0).
    pub fn new(noise: f64) -> Self {
        MnistLike { noise }
    }

    /// Generate a single 28×28 image of digit `digit` (0–9).
    ///
    /// # Arguments
    ///
    /// * `digit` – Digit to render (0–9).
    /// * `seed`  – Random seed for noise sampling.
    ///
    /// # Errors
    ///
    /// Returns an error if `digit > 9` or `noise < 0`.
    pub fn generate_digit(&self, digit: u8, seed: u64) -> Result<Array2<f64>> {
        if digit > 9 {
            return Err(DatasetsError::InvalidFormat(format!(
                "MnistLike::generate_digit: digit must be 0–9, got {digit}"
            )));
        }
        if self.noise < 0.0 {
            return Err(DatasetsError::InvalidFormat(
                "MnistLike: noise must be >= 0".to_string(),
            ));
        }

        let buf = render_digit(digit);
        let mut img = Array2::from_shape_vec((28, 28), buf).map_err(|e| {
            DatasetsError::ComputationError(format!("Array2 shape error: {e}"))
        })?;

        if self.noise > 0.0 {
            let mut rng = make_rng(seed);
            let dist = scirs2_core::random::Normal::new(0.0_f64, self.noise).map_err(|e| {
                DatasetsError::ComputationError(format!("Normal dist creation failed: {e}"))
            })?;
            for val in img.iter_mut() {
                *val = clamp01(*val + dist.sample(&mut rng));
            }
        }

        Ok(img)
    }

    /// Generate a dataset of `n_per_class` images per digit class (10 classes).
    ///
    /// Returns `(images, labels)` where
    /// - `images` is `Array3<f64>` of shape `(N, 28, 28)`, N = n_per_class × 10.
    /// - `labels` is `Array1<usize>` of length N with class labels 0–9.
    ///
    /// # Arguments
    ///
    /// * `n_per_class` – Images to generate per digit class (must be ≥ 1).
    /// * `seed`        – Base seed; each sample gets a deterministically derived sub-seed.
    ///
    /// # Errors
    ///
    /// Returns an error if `n_per_class == 0` or `noise < 0`.
    pub fn generate_dataset(
        &self,
        n_per_class: usize,
        seed: u64,
    ) -> Result<(Array3<f64>, Array1<usize>)> {
        if n_per_class == 0 {
            return Err(DatasetsError::InvalidFormat(
                "MnistLike::generate_dataset: n_per_class must be >= 1".to_string(),
            ));
        }
        if self.noise < 0.0 {
            return Err(DatasetsError::InvalidFormat(
                "MnistLike: noise must be >= 0".to_string(),
            ));
        }

        let n_classes = 10usize;
        let n_total = n_per_class * n_classes;
        let mut images = Array3::zeros((n_total, 28, 28));
        let mut labels = Array1::zeros(n_total);

        for class in 0..n_classes {
            for idx in 0..n_per_class {
                let sample_seed = seed
                    .wrapping_add(class as u64 * 1_000_003)
                    .wrapping_add(idx as u64 * 997);
                let img = self.generate_digit(class as u8, sample_seed)?;
                let sample_i = class * n_per_class + idx;
                for r in 0..28 {
                    for c in 0..28 {
                        images[[sample_i, r, c]] = img[[r, c]];
                    }
                }
                labels[sample_i] = class;
            }
        }

        Ok((images, labels))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CIFAR-10-like (32×32×3)
// ─────────────────────────────────────────────────────────────────────────────

/// Class labels for [`Cifar10Like`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cifar10Class {
    /// Airplane – blue-dominant background, light grey shape
    Airplane = 0,
    /// Automobile – red-dominant, rectangular body
    Automobile = 1,
    /// Bird – white body silhouette on sky-blue background
    Bird = 2,
    /// Cat – orange-brown ellipse on light background
    Cat = 3,
    /// Deer – brown shape with antler lines on green background
    Deer = 4,
    /// Dog – medium-brown filled ellipse on light background
    Dog = 5,
    /// Frog – bright green circle on dark green background
    Frog = 6,
    /// Horse – dark shape with legs
    Horse = 7,
    /// Ship – grey hull shape on deep-blue water background
    Ship = 8,
    /// Truck – dark grey rectangular truck on brown road
    Truck = 9,
}

impl Cifar10Class {
    /// Iterate over all 10 classes in index order.
    pub fn all() -> [Cifar10Class; 10] {
        [
            Self::Airplane,
            Self::Automobile,
            Self::Bird,
            Self::Cat,
            Self::Deer,
            Self::Dog,
            Self::Frog,
            Self::Horse,
            Self::Ship,
            Self::Truck,
        ]
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// Generator for CIFAR-10-like 32×32 RGB class-pattern images.
///
/// Each of the 10 classes has a distinct colour palette and shape pattern,
/// making the dataset useful for testing colour-aware classifiers.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::mnist_like::{Cifar10Like, Cifar10Class};
///
/// let gen = Cifar10Like::new(0.02);
/// let img = gen.generate_image(Cifar10Class::Cat, 42).expect("image");
/// assert_eq!(img.shape(), &[32, 32, 3]);
///
/// let (images, labels) = gen.generate_dataset(5, 42).expect("dataset");
/// assert_eq!(images.shape(), &[50, 32, 32, 3]);
/// ```
pub struct Cifar10Like {
    /// Standard deviation of additive Gaussian noise (≥ 0).
    pub noise: f64,
}

impl Cifar10Like {
    /// Create a new `Cifar10Like` generator.
    pub fn new(noise: f64) -> Self {
        Cifar10Like { noise }
    }

    /// Generate a single 32×32×3 image for `class`.
    ///
    /// Returns `Array3<f64>` of shape `(32, 32, 3)` (height, width, channels).
    ///
    /// # Errors
    ///
    /// Returns an error if `noise < 0`.
    pub fn generate_image(&self, class: Cifar10Class, seed: u64) -> Result<Array3<f64>> {
        if self.noise < 0.0 {
            return Err(DatasetsError::InvalidFormat(
                "Cifar10Like: noise must be >= 0".to_string(),
            ));
        }

        let rows = 32usize;
        let cols = 32usize;
        let channels = 3usize;
        // Flat buffer: index = r * cols * channels + c * channels + ch
        let mut buf = vec![0.0_f64; rows * cols * channels];

        let set_px = |buf: &mut Vec<f64>, r: usize, c: usize, rgb: [f64; 3]| {
            if r < rows && c < cols {
                let base = r * cols * channels + c * channels;
                buf[base] = clamp01(rgb[0]);
                buf[base + 1] = clamp01(rgb[1]);
                buf[base + 2] = clamp01(rgb[2]);
            }
        };

        // Helper: fill entire image with a background color
        let fill_bg = |buf: &mut Vec<f64>, rgb: [f64; 3]| {
            for r in 0..rows {
                for c in 0..cols {
                    let base = r * cols * channels + c * channels;
                    buf[base] = rgb[0];
                    buf[base + 1] = rgb[1];
                    buf[base + 2] = rgb[2];
                }
            }
        };

        // Helper: draw filled ellipse in given colour
        let fill_ellipse = |buf: &mut Vec<f64>,
                             cy: f64,
                             cx: f64,
                             ry: f64,
                             rx: f64,
                             rgb: [f64; 3]| {
            let r0 = ((cy - ry - 1.0).floor().max(0.0) as usize).min(rows);
            let r1 = ((cy + ry + 1.0).ceil() as usize).min(rows);
            let c0 = ((cx - rx - 1.0).floor().max(0.0) as usize).min(cols);
            let c1 = ((cx + rx + 1.0).ceil() as usize).min(cols);
            for r in r0..r1 {
                for c in c0..c1 {
                    let dy = (r as f64 - cy) / ry;
                    let dx = (c as f64 - cx) / rx;
                    let dist = dy * dy + dx * dx;
                    if dist <= 1.0 {
                        let base = r * cols * channels + c * channels;
                        buf[base] = clamp01(rgb[0]);
                        buf[base + 1] = clamp01(rgb[1]);
                        buf[base + 2] = clamp01(rgb[2]);
                    }
                }
            }
        };

        // Helper: draw a rectangle
        let fill_rect = |buf: &mut Vec<f64>,
                          r0: usize,
                          r1: usize,
                          c0: usize,
                          c1: usize,
                          rgb: [f64; 3]| {
            for r in r0..r1.min(rows) {
                for c in c0..c1.min(cols) {
                    let base = r * cols * channels + c * channels;
                    buf[base] = clamp01(rgb[0]);
                    buf[base + 1] = clamp01(rgb[1]);
                    buf[base + 2] = clamp01(rgb[2]);
                }
            }
        };

        let cy = 16.0_f64;
        let cx = 16.0_f64;

        match class {
            Cifar10Class::Airplane => {
                fill_bg(&mut buf, [0.5, 0.7, 0.9]); // sky blue
                // Fuselage (horizontal ellipse)
                fill_ellipse(&mut buf, cy, cx, 4.0, 12.0, [0.85, 0.85, 0.9]);
                // Wing (flat wide ellipse)
                fill_ellipse(&mut buf, cy + 1.0, cx, 2.0, 15.0, [0.75, 0.75, 0.85]);
            }
            Cifar10Class::Automobile => {
                fill_bg(&mut buf, [0.8, 0.8, 0.8]); // grey road
                // Body
                fill_rect(&mut buf, 18, 26, 4, 28, [0.8, 0.15, 0.15]);
                // Cabin
                fill_rect(&mut buf, 12, 18, 8, 24, [0.65, 0.1, 0.1]);
                // Wheels (dark circles)
                fill_ellipse(&mut buf, 26.0, 8.0, 3.0, 3.0, [0.15, 0.15, 0.15]);
                fill_ellipse(&mut buf, 26.0, 24.0, 3.0, 3.0, [0.15, 0.15, 0.15]);
            }
            Cifar10Class::Bird => {
                fill_bg(&mut buf, [0.55, 0.75, 0.95]); // sky blue
                // White body ellipse
                fill_ellipse(&mut buf, cy, cx, 7.0, 10.0, [0.95, 0.95, 0.95]);
                // Head circle
                fill_ellipse(&mut buf, cy - 7.0, cx + 6.0, 4.0, 4.0, [0.9, 0.9, 0.9]);
            }
            Cifar10Class::Cat => {
                fill_bg(&mut buf, [0.95, 0.9, 0.8]); // warm white
                // Body ellipse (orange-brown)
                fill_ellipse(&mut buf, cy + 2.0, cx, 8.0, 9.0, [0.8, 0.5, 0.2]);
                // Head
                fill_ellipse(&mut buf, cy - 7.0, cx, 5.0, 6.0, [0.75, 0.45, 0.18]);
                // Ears (triangles approximated as small ellipses)
                fill_ellipse(&mut buf, cy - 12.0, cx - 5.0, 2.0, 2.0, [0.7, 0.4, 0.15]);
                fill_ellipse(&mut buf, cy - 12.0, cx + 5.0, 2.0, 2.0, [0.7, 0.4, 0.15]);
            }
            Cifar10Class::Deer => {
                fill_bg(&mut buf, [0.3, 0.65, 0.3]); // green meadow
                // Brown body
                fill_ellipse(&mut buf, cy + 2.0, cx, 7.0, 8.0, [0.6, 0.35, 0.1]);
                // Head
                fill_ellipse(&mut buf, cy - 7.0, cx, 4.0, 4.0, [0.55, 0.3, 0.08]);
            }
            Cifar10Class::Dog => {
                fill_bg(&mut buf, [0.9, 0.87, 0.8]); // light tan
                // Medium-brown body
                fill_ellipse(&mut buf, cy + 2.0, cx, 8.0, 9.0, [0.55, 0.35, 0.18]);
                // Head
                fill_ellipse(&mut buf, cy - 7.0, cx, 5.0, 6.0, [0.5, 0.3, 0.15]);
                // Snout (lighter ellipse)
                fill_ellipse(&mut buf, cy - 5.0, cx + 3.0, 2.0, 3.0, [0.7, 0.55, 0.35]);
            }
            Cifar10Class::Frog => {
                fill_bg(&mut buf, [0.1, 0.35, 0.1]); // dark green
                // Bright green body circle
                fill_ellipse(&mut buf, cy, cx, 10.0, 11.0, [0.2, 0.85, 0.2]);
                // Eyes (white circles)
                fill_ellipse(&mut buf, cy - 9.0, cx - 5.0, 2.5, 2.5, [0.95, 0.95, 0.1]);
                fill_ellipse(&mut buf, cy - 9.0, cx + 5.0, 2.5, 2.5, [0.95, 0.95, 0.1]);
            }
            Cifar10Class::Horse => {
                fill_bg(&mut buf, [0.6, 0.8, 0.5]); // light green grass
                // Dark body
                fill_ellipse(&mut buf, cy, cx - 2.0, 7.0, 10.0, [0.3, 0.18, 0.05]);
                // Head
                fill_ellipse(&mut buf, cy - 8.0, cx + 7.0, 4.0, 5.0, [0.28, 0.16, 0.04]);
                // Legs
                for leg_cx in [cx - 8.0, cx - 3.0, cx + 2.0, cx + 7.0] {
                    fill_rect(
                        &mut buf,
                        (cy + 7.0) as usize,
                        (cy + 14.0) as usize,
                        (leg_cx - 1.0) as usize,
                        (leg_cx + 1.0) as usize,
                        [0.25, 0.14, 0.03],
                    );
                }
            }
            Cifar10Class::Ship => {
                fill_bg(&mut buf, [0.05, 0.15, 0.5]); // deep blue sea
                // Grey hull (wide trapezoid approximated as rectangle)
                fill_rect(&mut buf, 18, 26, 4, 28, [0.55, 0.55, 0.6]);
                // White superstructure
                fill_rect(&mut buf, 10, 18, 10, 22, [0.9, 0.9, 0.9]);
            }
            Cifar10Class::Truck => {
                fill_bg(&mut buf, [0.55, 0.45, 0.3]); // dusty road
                // Cab (dark grey)
                fill_rect(&mut buf, 10, 24, 4, 14, [0.25, 0.25, 0.28]);
                // Cargo box (medium grey)
                fill_rect(&mut buf, 12, 24, 14, 29, [0.4, 0.4, 0.42]);
                // Wheels
                fill_ellipse(&mut buf, 25.0, 9.0, 3.0, 3.0, [0.12, 0.12, 0.12]);
                fill_ellipse(&mut buf, 25.0, 21.0, 3.0, 3.0, [0.12, 0.12, 0.12]);
            }
        }

        // Add noise
        if self.noise > 0.0 {
            let mut rng = make_rng(seed);
            let dist = scirs2_core::random::Normal::new(0.0_f64, self.noise).map_err(|e| {
                DatasetsError::ComputationError(format!("Normal dist creation failed: {e}"))
            })?;
            for v in buf.iter_mut() {
                *v = clamp01(*v + dist.sample(&mut rng));
            }
        }

        let arr = Array3::from_shape_vec((rows, cols, channels), buf).map_err(|e| {
            DatasetsError::ComputationError(format!("Array3 shape error: {e}"))
        })?;

        Ok(arr)
    }

    /// Generate `n_per_class` images for every one of the 10 classes.
    ///
    /// Returns `(images, labels)` where:
    /// - `images` is `Array4<f64>` of shape `(N, 32, 32, 3)`.
    /// - `labels` is `Array1<usize>` of length N = n_per_class × 10.
    ///
    /// # Errors
    ///
    /// Returns an error if `n_per_class == 0` or `noise < 0`.
    pub fn generate_dataset(
        &self,
        n_per_class: usize,
        seed: u64,
    ) -> Result<(Array4<f64>, Array1<usize>)> {
        if n_per_class == 0 {
            return Err(DatasetsError::InvalidFormat(
                "Cifar10Like::generate_dataset: n_per_class must be >= 1".to_string(),
            ));
        }
        if self.noise < 0.0 {
            return Err(DatasetsError::InvalidFormat(
                "Cifar10Like: noise must be >= 0".to_string(),
            ));
        }

        let classes = Cifar10Class::all();
        let n_total = n_per_class * classes.len();
        let mut images = Array4::zeros((n_total, 32, 32, 3));
        let mut labels = Array1::zeros(n_total);

        for (ci, &class) in classes.iter().enumerate() {
            for idx in 0..n_per_class {
                let sample_seed = seed
                    .wrapping_add(ci as u64 * 1_000_003)
                    .wrapping_add(idx as u64 * 997);
                let img = self.generate_image(class, sample_seed)?;
                let sample_i = ci * n_per_class + idx;
                for r in 0..32 {
                    for c in 0..32 {
                        for ch in 0..3 {
                            images[[sample_i, r, c, ch]] = img[[r, c, ch]];
                        }
                    }
                }
                labels[sample_i] = class.index();
            }
        }

        Ok((images, labels))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shape type
// ─────────────────────────────────────────────────────────────────────────────

/// Supported shape classes for [`generate_shapes_dataset`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeClass {
    /// Class 0 – filled circle.
    Circle = 0,
    /// Class 1 – filled square (axis-aligned).
    Square = 1,
    /// Class 2 – filled equilateral triangle.
    Triangle = 2,
    /// Class 3 – filled ellipse (aspect ratio ≠ 1).
    Ellipse = 3,
}

impl ShapeClass {
    fn from_index(i: usize) -> Option<ShapeClass> {
        match i {
            0 => Some(ShapeClass::Circle),
            1 => Some(ShapeClass::Square),
            2 => Some(ShapeClass::Triangle),
            3 => Some(ShapeClass::Ellipse),
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// generate_shapes_dataset
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a 2-D shape classification dataset.
///
/// Produces `n_samples` grayscale images of shape `(size, size)`, each
/// containing exactly one random shape (circle, square, triangle, or ellipse)
/// placed at a random position with a random scale.  `n_classes` controls how
/// many shape types are included (1–4, in the order Circle → Square →
/// Triangle → Ellipse).
///
/// # Arguments
///
/// * `n_samples` – Total number of images (must be ≥ 1).
/// * `n_classes` – Number of shape classes to use (1–4).
/// * `size`      – Side length of each square image in pixels (must be ≥ 8).
/// * `noise`     – Gaussian noise std-dev added to each pixel (≥ 0).
/// * `seed`      – Random seed.
///
/// # Returns
///
/// `(images, labels)` where
/// - `images` is `Array3<f64>` of shape `(n_samples, size, size)`.
/// - `labels` is `Array1<usize>` of length `n_samples` with class indices 0..n_classes.
///
/// # Errors
///
/// Returns an error if `n_samples == 0`, `n_classes > 4`, `size < 8`, or `noise < 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::mnist_like::generate_shapes_dataset;
///
/// let (imgs, lbls) = generate_shapes_dataset(40, 4, 32, 0.05, 42).expect("shapes");
/// assert_eq!(imgs.shape(), &[40, 32, 32]);
/// assert_eq!(lbls.len(), 40);
/// ```
pub fn generate_shapes_dataset(
    n_samples: usize,
    n_classes: usize,
    size: usize,
    noise: f64,
    seed: u64,
) -> Result<(Array3<f64>, Array1<usize>)> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "generate_shapes_dataset: n_samples must be >= 1".to_string(),
        ));
    }
    if n_classes == 0 || n_classes > 4 {
        return Err(DatasetsError::InvalidFormat(
            "generate_shapes_dataset: n_classes must be in 1..=4".to_string(),
        ));
    }
    if size < 8 {
        return Err(DatasetsError::InvalidFormat(
            "generate_shapes_dataset: size must be >= 8".to_string(),
        ));
    }
    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "generate_shapes_dataset: noise must be >= 0".to_string(),
        ));
    }

    let mut rng = make_rng(seed);

    // Shape index distribution
    let class_dist = scirs2_core::random::Uniform::new(0usize, n_classes).map_err(|e| {
        DatasetsError::ComputationError(format!("Uniform dist failed: {e}"))
    })?;
    // Position and scale distributions (normalised to [0.25, 0.75] of image size)
    let pos_dist = scirs2_core::random::Uniform::new(0.25_f64, 0.75_f64).map_err(|e| {
        DatasetsError::ComputationError(format!("Uniform pos dist failed: {e}"))
    })?;
    // Radius/half-side: 10%–25% of image size
    let radius_dist = scirs2_core::random::Uniform::new(0.10_f64, 0.25_f64).map_err(|e| {
        DatasetsError::ComputationError(format!("Uniform radius dist failed: {e}"))
    })?;
    // Ellipse aspect ratio: 1.5x–3x
    let aspect_dist = scirs2_core::random::Uniform::new(1.5_f64, 3.0_f64).map_err(|e| {
        DatasetsError::ComputationError(format!("Uniform aspect dist failed: {e}"))
    })?;

    let noise_dist_opt = if noise > 0.0 {
        Some(
            scirs2_core::random::Normal::new(0.0_f64, noise).map_err(|e| {
                DatasetsError::ComputationError(format!("Normal dist failed: {e}"))
            })?,
        )
    } else {
        None
    };

    let mut images = Array3::zeros((n_samples, size, size));
    let mut labels = Array1::zeros(n_samples);

    for s in 0..n_samples {
        let class_idx = class_dist.sample(&mut rng);
        let shape = ShapeClass::from_index(class_idx).ok_or_else(|| {
            DatasetsError::ComputationError(format!("Invalid class index: {class_idx}"))
        })?;

        let cy = pos_dist.sample(&mut rng) * size as f64;
        let cx = pos_dist.sample(&mut rng) * size as f64;
        let r = radius_dist.sample(&mut rng) * size as f64;

        let mut buf = vec![0.0_f64; size * size];

        match shape {
            ShapeClass::Circle => {
                draw_filled_circle(&mut buf, size, size, cy, cx, r, 1.0);
            }
            ShapeClass::Square => {
                let half = r;
                let r0 = ((cy - half).floor().max(0.0) as usize).min(size);
                let r1 = ((cy + half).ceil() as usize + 1).min(size);
                let c0 = ((cx - half).floor().max(0.0) as usize).min(size);
                let c1 = ((cx + half).ceil() as usize + 1).min(size);
                for ri in r0..r1 {
                    for ci in c0..c1 {
                        buf[ri * size + ci] = 1.0;
                    }
                }
            }
            ShapeClass::Triangle => {
                // Equilateral triangle with centroid at (cy, cx)
                let h = r * 1.732; // sqrt(3) * r
                // Vertices (rotated so base is horizontal)
                let v0 = (cy - 2.0 * r / 3.0 * 1.732, cx); // apex (top)
                let v1 = (cy + r / 3.0 * 1.732, cx - r); // bottom-left
                let v2 = (cy + r / 3.0 * 1.732, cx + r); // bottom-right
                // Scanline fill
                let ymin = v0.0.min(v1.0).min(v2.0).floor().max(0.0) as usize;
                let ymax = (v0.0.max(v1.0).max(v2.0).ceil() as usize).min(size);
                for row in ymin..ymax {
                    let y = row as f64 + 0.5;
                    // Use barycentric approach: collect x intersections
                    let edges = [(v0, v1), (v1, v2), (v2, v0)];
                    let mut xs = Vec::with_capacity(2);
                    for &(pa, pb) in &edges {
                        let (ya, xa) = pa;
                        let (yb, xb) = pb;
                        if (ya <= y && y < yb) || (yb <= y && y < ya) {
                            let t = (y - ya) / (yb - ya);
                            xs.push(xa + t * (xb - xa));
                        }
                    }
                    if xs.len() >= 2 {
                        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let col_start = (xs[0].floor().max(0.0) as usize).min(size);
                        let col_end = (xs[xs.len() - 1].ceil() as usize + 1).min(size);
                        for ci in col_start..col_end {
                            buf[row * size + ci] = 1.0;
                        }
                    }
                }
                let _ = h; // suppress unused warning
            }
            ShapeClass::Ellipse => {
                let aspect = aspect_dist.sample(&mut rng);
                let ry = r;
                let rx = r / aspect; // narrower horizontal axis
                let r0 = ((cy - ry - 1.0).floor().max(0.0) as usize).min(size);
                let r1 = ((cy + ry + 1.0).ceil() as usize).min(size);
                let c0 = ((cx - rx - 1.0).floor().max(0.0) as usize).min(size);
                let c1 = ((cx + rx + 1.0).ceil() as usize).min(size);
                for ri in r0..r1 {
                    for ci in c0..c1 {
                        let dy = (ri as f64 - cy) / ry;
                        let dx = (ci as f64 - cx) / rx;
                        if dy * dy + dx * dx <= 1.0 {
                            buf[ri * size + ci] = 1.0;
                        }
                    }
                }
            }
        }

        // Add noise
        if let Some(ref ndist) = noise_dist_opt {
            for v in buf.iter_mut() {
                *v = clamp01(*v + ndist.sample(&mut rng));
            }
        }

        for r in 0..size {
            for c in 0..size {
                images[[s, r, c]] = buf[r * size + c];
            }
        }
        labels[s] = class_idx;
    }

    Ok((images, labels))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── MnistLike ────────────────────────────────────────────────────────────

    #[test]
    fn test_digit_shape() {
        let gen = MnistLike::new(0.0);
        for d in 0u8..=9 {
            let img = gen.generate_digit(d, 42).expect("digit failed");
            assert_eq!(img.shape(), &[28, 28], "digit {d} shape mismatch");
        }
    }

    #[test]
    fn test_digit_values_in_range() {
        let gen = MnistLike::new(0.05);
        for d in 0u8..=9 {
            let img = gen.generate_digit(d, d as u64 + 1).expect("digit");
            for &v in img.iter() {
                assert!(
                    (0.0..=1.0).contains(&v),
                    "digit {d} pixel {v} out of [0,1]"
                );
            }
        }
    }

    #[test]
    fn test_digit_foreground_nonzero() {
        let gen = MnistLike::new(0.0);
        for d in 0u8..=9 {
            let img = gen.generate_digit(d, 1).expect("digit");
            let n_fg = img.iter().filter(|&&v| v > 0.1).count();
            assert!(n_fg > 5, "digit {d} should have some foreground pixels");
        }
    }

    #[test]
    fn test_digit_determinism() {
        let gen = MnistLike::new(0.1);
        let img1 = gen.generate_digit(3, 99).expect("d1");
        let img2 = gen.generate_digit(3, 99).expect("d2");
        for (a, b) in img1.iter().zip(img2.iter()) {
            assert!((a - b).abs() < 1e-12, "same seed must produce same result");
        }
    }

    #[test]
    fn test_digit_error_out_of_range() {
        let gen = MnistLike::new(0.0);
        assert!(gen.generate_digit(10, 1).is_err());
    }

    #[test]
    fn test_dataset_shape() {
        let gen = MnistLike::new(0.02);
        let (imgs, lbls) = gen.generate_dataset(5, 42).expect("dataset");
        assert_eq!(imgs.shape(), &[50, 28, 28]);
        assert_eq!(lbls.len(), 50);
    }

    #[test]
    fn test_dataset_labels_all_classes() {
        let gen = MnistLike::new(0.0);
        let (_, lbls) = gen.generate_dataset(3, 7).expect("dataset");
        for class in 0..10usize {
            assert!(
                lbls.iter().any(|&l| l == class),
                "class {class} missing from labels"
            );
        }
    }

    #[test]
    fn test_dataset_error_zero_per_class() {
        let gen = MnistLike::new(0.0);
        assert!(gen.generate_dataset(0, 1).is_err());
    }

    // ── Cifar10Like ──────────────────────────────────────────────────────────

    #[test]
    fn test_cifar_image_shape() {
        let gen = Cifar10Like::new(0.0);
        for class in Cifar10Class::all() {
            let img = gen.generate_image(class, 42).expect("cifar image");
            assert_eq!(img.shape(), &[32, 32, 3], "class {class:?} shape mismatch");
        }
    }

    #[test]
    fn test_cifar_values_in_range() {
        let gen = Cifar10Like::new(0.03);
        for class in Cifar10Class::all() {
            let img = gen.generate_image(class, 1).expect("cifar image");
            for &v in img.iter() {
                assert!(
                    (0.0..=1.0).contains(&v),
                    "class {class:?} pixel {v} out of [0,1]"
                );
            }
        }
    }

    #[test]
    fn test_cifar_dataset_shape() {
        let gen = Cifar10Like::new(0.01);
        let (imgs, lbls) = gen.generate_dataset(4, 42).expect("cifar dataset");
        assert_eq!(imgs.shape(), &[40, 32, 32, 3]);
        assert_eq!(lbls.len(), 40);
    }

    #[test]
    fn test_cifar_dataset_error_zero() {
        let gen = Cifar10Like::new(0.0);
        assert!(gen.generate_dataset(0, 1).is_err());
    }

    // ── generate_shapes_dataset ──────────────────────────────────────────────

    #[test]
    fn test_shapes_shape() {
        let (imgs, lbls) = generate_shapes_dataset(40, 4, 32, 0.0, 42).expect("shapes");
        assert_eq!(imgs.shape(), &[40, 32, 32]);
        assert_eq!(lbls.len(), 40);
    }

    #[test]
    fn test_shapes_values_in_range() {
        let (imgs, _) = generate_shapes_dataset(20, 4, 28, 0.1, 7).expect("shapes noise");
        for &v in imgs.iter() {
            assert!((0.0..=1.0).contains(&v), "pixel {v} out of [0,1]");
        }
    }

    #[test]
    fn test_shapes_labels_valid() {
        let n_classes = 3usize;
        let (_, lbls) = generate_shapes_dataset(30, n_classes, 32, 0.0, 5).expect("shapes lbls");
        for &l in lbls.iter() {
            assert!(l < n_classes, "label {l} >= n_classes {n_classes}");
        }
    }

    #[test]
    fn test_shapes_error_n_samples_zero() {
        assert!(generate_shapes_dataset(0, 4, 32, 0.0, 1).is_err());
    }

    #[test]
    fn test_shapes_error_n_classes_zero() {
        assert!(generate_shapes_dataset(10, 0, 32, 0.0, 1).is_err());
    }

    #[test]
    fn test_shapes_error_n_classes_too_large() {
        assert!(generate_shapes_dataset(10, 5, 32, 0.0, 1).is_err());
    }

    #[test]
    fn test_shapes_error_size_too_small() {
        assert!(generate_shapes_dataset(10, 4, 4, 0.0, 1).is_err());
    }

    #[test]
    fn test_shapes_determinism() {
        let (imgs1, lbls1) = generate_shapes_dataset(20, 4, 32, 0.05, 42).expect("s1");
        let (imgs2, lbls2) = generate_shapes_dataset(20, 4, 32, 0.05, 42).expect("s2");
        for (a, b) in imgs1.iter().zip(imgs2.iter()) {
            assert!((a - b).abs() < 1e-12, "pixel mismatch across runs");
        }
        for (a, b) in lbls1.iter().zip(lbls2.iter()) {
            assert_eq!(a, b, "label mismatch across runs");
        }
    }

    #[test]
    fn test_shapes_foreground_nonzero() {
        let (imgs, _) = generate_shapes_dataset(10, 4, 32, 0.0, 3).expect("shapes fg");
        // Each image should have some foreground
        for s in 0..10 {
            let fg = (0..32)
                .flat_map(|r| (0..32).map(move |c| (r, c)))
                .filter(|&(r, c)| imgs[[s, r, c]] > 0.5)
                .count();
            assert!(fg > 0, "sample {s} has no foreground pixels");
        }
    }
}
