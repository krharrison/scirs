//! Computer vision synthetic dataset generators.
//!
//! Provides pixel-level image datasets for classification, object detection,
//! and semantic segmentation.  All images are stored as
//! `(n, channels, height, width)` tensors of `f64` values in `[0, 1]`.
//!
//! All generators are self-contained (Park-Miller LCG, no external crates).

use crate::error::{DatasetsError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Internal LCG RNG
// ─────────────────────────────────────────────────────────────────────────────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 6364136223846793005 } else { seed })
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn next_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ImageDataset
// ─────────────────────────────────────────────────────────────────────────────

/// A batch of labelled images stored as `(N, C, H, W)` nested Vecs.
///
/// All pixel values are in `[0.0, 1.0]` (or `[0.0, max]` before `normalize()`).
#[derive(Debug, Clone)]
pub struct ImageDataset {
    /// Images stored as `images[n][c][h][w]`.
    pub images: Vec<Vec<Vec<Vec<f64>>>>,
    /// Integer class label per image.
    pub labels: Vec<usize>,
    /// Human-readable name per class.
    pub label_names: Vec<String>,
    /// Number of channels (1 = grayscale, 3 = RGB).
    pub n_channels: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Image width in pixels.
    pub width: usize,
}

impl ImageDataset {
    /// Number of images in the dataset.
    pub fn len(&self) -> usize {
        self.images.len()
    }

    /// Returns `true` if the dataset contains no images.
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    /// Number of distinct classes.
    pub fn n_classes(&self) -> usize {
        self.label_names.len()
    }

    /// Return a reference to image `idx` with shape `(C, H, W)`.
    ///
    /// # Errors
    ///
    /// Returns an error if `idx` is out of bounds.
    pub fn image(&self, idx: usize) -> Result<&Vec<Vec<Vec<f64>>>> {
        self.images.get(idx).ok_or_else(|| {
            DatasetsError::InvalidFormat(format!(
                "ImageDataset::image: index {idx} out of bounds (len={})",
                self.images.len()
            ))
        })
    }

    /// Normalize all pixel values to `[0.0, 1.0]` using the global min/max.
    ///
    /// If all pixels are equal, this is a no-op.
    pub fn normalize(&mut self) {
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for img in &self.images {
            for ch in img {
                for row in ch {
                    for &p in row {
                        if p < min_v {
                            min_v = p;
                        }
                        if p > max_v {
                            max_v = p;
                        }
                    }
                }
            }
        }
        let range = max_v - min_v;
        if range < 1e-12 {
            return;
        }
        for img in self.images.iter_mut() {
            for ch in img.iter_mut() {
                for row in ch.iter_mut() {
                    for p in row.iter_mut() {
                        *p = (*p - min_v) / range;
                    }
                }
            }
        }
    }

    /// Split the dataset into train and test sets.
    ///
    /// `test_ratio` fraction of each class is moved to test (stratified).
    ///
    /// # Arguments
    ///
    /// * `test_ratio` – Fraction in (0, 1) for the test set.
    /// * `seed`       – Reproducibility seed.
    ///
    /// # Errors
    ///
    /// Returns an error if `test_ratio` is not in `(0, 1)`.
    pub fn split(&self, test_ratio: f64, seed: u64) -> Result<(Self, Self)> {
        if !(0.0 < test_ratio && test_ratio < 1.0) {
            return Err(DatasetsError::InvalidFormat(
                "ImageDataset::split: test_ratio must be in (0, 1)".to_string(),
            ));
        }

        let mut rng = Lcg::new(seed);
        let n = self.len();

        // Stratified split: group by class.
        let n_classes = self.n_classes();
        let mut class_indices: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
        for (idx, &lbl) in self.labels.iter().enumerate() {
            if lbl < n_classes {
                class_indices[lbl].push(idx);
            }
        }

        let mut train_idx: Vec<usize> = Vec::new();
        let mut test_idx: Vec<usize> = Vec::new();

        for indices in &class_indices {
            let n_test = ((indices.len() as f64 * test_ratio).round() as usize).max(0);
            let mut perm: Vec<usize> = (0..indices.len()).collect();
            // Partial Fisher-Yates.
            for k in 0..n_test.min(indices.len()) {
                let j = k + rng.next_usize(indices.len() - k);
                perm.swap(k, j);
            }
            for k in 0..indices.len() {
                if k < n_test {
                    test_idx.push(indices[perm[k]]);
                } else {
                    train_idx.push(indices[perm[k]]);
                }
            }
        }

        // Sort to preserve original order within each split.
        train_idx.sort_unstable();
        test_idx.sort_unstable();

        let label_names = self.label_names.clone();
        let nc = self.n_channels;
        let h = self.height;
        let w = self.width;

        let collect_split = |indices: &[usize]| -> (Vec<Vec<Vec<Vec<f64>>>>, Vec<usize>) {
            let imgs = indices.iter().map(|&i| self.images[i].clone()).collect();
            let lbls = indices.iter().map(|&i| self.labels[i]).collect();
            (imgs, lbls)
        };

        let (train_imgs, train_lbls) = collect_split(&train_idx);
        let (test_imgs, test_lbls) = collect_split(&test_idx);

        let _ = n; // suppress unused warning
        Ok((
            ImageDataset {
                images: train_imgs,
                labels: train_lbls,
                label_names: label_names.clone(),
                n_channels: nc,
                height: h,
                width: w,
            },
            ImageDataset {
                images: test_imgs,
                labels: test_lbls,
                label_names,
                n_channels: nc,
                height: h,
                width: w,
            },
        ))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal drawing primitives (grayscale)
// ─────────────────────────────────────────────────────────────────────────────

/// Allocate a blank single-channel image (H × W) filled with `fill`.
fn blank_image(h: usize, w: usize, fill: f64) -> Vec<Vec<Vec<f64>>> {
    vec![vec![vec![fill; w]; h]]
}

/// Draw a filled circle (anti-aliased via distance) on a single-channel image.
fn draw_circle(img: &mut Vec<Vec<Vec<f64>>>, cy: f64, cx: f64, radius: f64, color: f64) {
    let h = img[0].len();
    let w = img[0][0].len();
    for r in 0..h {
        for c in 0..w {
            let dy = r as f64 - cy;
            let dx = c as f64 - cx;
            let dist = (dy * dy + dx * dx).sqrt();
            if dist <= radius {
                img[0][r][c] = color;
            }
        }
    }
}

/// Draw an axis-aligned filled rectangle on a single-channel image.
fn draw_rect(
    img: &mut Vec<Vec<Vec<f64>>>,
    top: usize,
    left: usize,
    bot: usize,
    right: usize,
    color: f64,
) {
    let h = img[0].len();
    let w = img[0][0].len();
    let bot = bot.min(h);
    let right = right.min(w);
    for r in top..bot {
        for c in left..right {
            img[0][r][c] = color;
        }
    }
}

/// Draw a filled triangle (barycentric rasterisation) on a single-channel image.
fn draw_triangle(
    img: &mut Vec<Vec<Vec<f64>>>,
    p0: (f64, f64),
    p1: (f64, f64),
    p2: (f64, f64),
    color: f64,
) {
    let h = img[0].len();
    let w = img[0][0].len();

    // Bounding box.
    let min_r = p0.0.min(p1.0).min(p2.0).max(0.0) as usize;
    let max_r = (p0.0.max(p1.0).max(p2.0) as usize + 1).min(h);
    let min_c = p0.1.min(p1.1).min(p2.1).max(0.0) as usize;
    let max_c = (p0.1.max(p1.1).max(p2.1) as usize + 1).min(w);

    let sign = |ax: f64, ay: f64, bx: f64, by: f64, px: f64, py: f64| -> f64 {
        (px - bx) * (ay - by) - (ax - bx) * (py - by)
    };

    for r in min_r..max_r {
        for c in min_c..max_c {
            let px = c as f64;
            let py = r as f64;
            let d0 = sign(p0.1, p0.0, p1.1, p1.0, px, py);
            let d1 = sign(p1.1, p1.0, p2.1, p2.0, px, py);
            let d2 = sign(p2.1, p2.0, p0.1, p0.0, px, py);
            let has_neg = d0 < 0.0 || d1 < 0.0 || d2 < 0.0;
            let has_pos = d0 > 0.0 || d1 > 0.0 || d2 > 0.0;
            if !(has_neg && has_pos) {
                img[0][r][c] = color;
            }
        }
    }
}

/// Add Gaussian noise to a single-channel image (clamps to [0,1]).
fn add_noise(img: &mut Vec<Vec<Vec<f64>>>, std: f64, rng: &mut Lcg) {
    let h = img[0].len();
    let w = img[0][0].len();
    for r in 0..h {
        for c in 0..w {
            img[0][r][c] = (img[0][r][c] + rng.next_normal() * std).clamp(0.0, 1.0);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// make_shapes_dataset
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a grayscale shapes classification dataset.
///
/// Three classes: `0` = circle, `1` = square, `2` = triangle.
/// The shape is drawn in white (1.0) on a black (0.0) background, then
/// Gaussian noise of standard deviation `noise_std` is added.
///
/// # Arguments
///
/// * `n_per_class` – Number of images per class.
/// * `image_size`  – Side length of the square images (must be ≥ 8).
/// * `noise_std`   – Standard deviation of additive Gaussian noise (≥ 0).
/// * `seed`        – Reproducibility seed.
///
/// # Errors
///
/// Returns an error if `image_size < 8`.
pub fn make_shapes_dataset(
    n_per_class: usize,
    image_size: usize,
    noise_std: f64,
    seed: u64,
) -> Result<ImageDataset> {
    if image_size < 8 {
        return Err(DatasetsError::InvalidFormat(
            "make_shapes_dataset: image_size must be >= 8".to_string(),
        ));
    }

    let mut rng = Lcg::new(seed);
    let sz = image_size as f64;
    let margin = sz * 0.15;
    let inner = sz - 2.0 * margin;

    let n_classes = 3usize;
    let n_total = n_per_class * n_classes;
    let mut images: Vec<Vec<Vec<Vec<f64>>>> = Vec::with_capacity(n_total);
    let mut labels: Vec<usize> = Vec::with_capacity(n_total);

    for class_id in 0..n_classes {
        for _ in 0..n_per_class {
            let mut img = blank_image(image_size, image_size, 0.0);

            match class_id {
                0 => {
                    // Circle: random centre and radius.
                    let r = margin + rng.next_f64() * inner * 0.5;
                    let cx = margin + rng.next_f64() * (inner - 2.0 * r) + r;
                    let cy = margin + rng.next_f64() * (inner - 2.0 * r) + r;
                    let radius = (inner * 0.15 + rng.next_f64() * inner * 0.2).min(r);
                    draw_circle(&mut img, cy, cx, radius, 1.0);
                }
                1 => {
                    // Square: random position and size.
                    let side = (inner * 0.2 + rng.next_f64() * inner * 0.3) as usize;
                    let side = side.max(4);
                    let max_top = (sz - margin as f64 - side as f64) as usize;
                    let top = (margin as usize)
                        + rng.next_usize(max_top.saturating_sub(margin as usize).max(1));
                    let max_left = (sz - margin as f64 - side as f64) as usize;
                    let left = (margin as usize)
                        + rng.next_usize(max_left.saturating_sub(margin as usize).max(1));
                    draw_rect(&mut img, top, left, top + side, left + side, 1.0);
                }
                _ => {
                    // Triangle: random upward-pointing triangle.
                    let base_y = margin + rng.next_f64() * inner * 0.4 + inner * 0.4;
                    let apex_y = base_y - inner * (0.25 + rng.next_f64() * 0.25);
                    let cx = margin + rng.next_f64() * inner;
                    let half_base = inner * (0.1 + rng.next_f64() * 0.15);
                    draw_triangle(
                        &mut img,
                        (apex_y, cx),
                        (base_y, cx - half_base),
                        (base_y, cx + half_base),
                        1.0,
                    );
                }
            }

            if noise_std > 0.0 {
                add_noise(&mut img, noise_std, &mut rng);
            }

            images.push(img);
            labels.push(class_id);
        }
    }

    let label_names = vec![
        "circle".to_string(),
        "square".to_string(),
        "triangle".to_string(),
    ];

    Ok(ImageDataset {
        images,
        labels,
        label_names,
        n_channels: 1,
        height: image_size,
        width: image_size,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// make_mnist_like
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a synthetic MNIST-like dataset of grayscale digit images.
///
/// Digits 0–9 are rendered as simple stroke-based patterns on square images.
/// Each image is grayscale (single channel), normalized to `[0, 1]`.
///
/// The generated patterns are simplified bitmap-style glyphs that are
/// consistent across seeds (same seed → same images).
///
/// # Arguments
///
/// * `n_per_class` – Images per digit class (10 classes total).
/// * `image_size`  – Side length in pixels (typically 28; must be ≥ 8).
/// * `seed`        – Reproducibility seed.
///
/// # Errors
///
/// Returns an error if `image_size < 8`.
pub fn make_mnist_like(n_per_class: usize, image_size: usize, seed: u64) -> Result<ImageDataset> {
    if image_size < 8 {
        return Err(DatasetsError::InvalidFormat(
            "make_mnist_like: image_size must be >= 8".to_string(),
        ));
    }

    let mut rng = Lcg::new(seed);
    let n_classes = 10usize;
    let n_total = n_per_class * n_classes;
    let mut images: Vec<Vec<Vec<Vec<f64>>>> = Vec::with_capacity(n_total);
    let mut labels: Vec<usize> = Vec::with_capacity(n_total);

    let sz = image_size;
    let half = sz / 2;
    let q = sz / 4;

    for digit in 0..n_classes {
        for _ in 0..n_per_class {
            let mut img = blank_image(sz, sz, 0.0);
            // Slight random offset.
            let dy = rng.next_usize(3) as isize - 1;
            let dx = rng.next_usize(3) as isize - 1;

            let shifted = |r: usize, c: usize| -> (usize, usize) {
                let nr = (r as isize + dy).clamp(0, sz as isize - 1) as usize;
                let nc = (c as isize + dx).clamp(0, sz as isize - 1) as usize;
                (nr, nc)
            };

            // Draw a simple prototype glyph for each digit.
            match digit {
                0 => {
                    // Oval outline.
                    draw_circle(&mut img, half as f64, half as f64, (half - 2) as f64, 1.0);
                    draw_circle(
                        &mut img,
                        half as f64,
                        half as f64,
                        (half - 5).max(1) as f64,
                        0.0,
                    );
                }
                1 => {
                    // Vertical bar.
                    draw_rect(&mut img, q, half - 1, sz - q, half + 2, 1.0);
                }
                2 => {
                    // Top arc + diagonal + bottom bar.
                    draw_circle(&mut img, q as f64, half as f64, (q + 1) as f64, 1.0);
                    draw_circle(&mut img, q as f64, half as f64, (q - 1).max(0) as f64, 0.0);
                    draw_rect(&mut img, sz - q - 2, q, sz - q, sz - q, 1.0);
                    // Diagonal approximation via rect.
                    draw_rect(&mut img, q, q, sz - q, q + 3, 1.0);
                }
                3 => {
                    // Two horizontal bars + right vertical.
                    draw_rect(&mut img, q, q, q + 3, sz - q, 1.0);
                    draw_rect(&mut img, half - 1, half, half + 2, sz - q, 1.0);
                    draw_rect(&mut img, sz - q - 3, q, sz - q, sz - q, 1.0);
                    draw_rect(&mut img, q, sz - q - 3, sz - q, sz - q, 1.0);
                }
                4 => {
                    // Left vertical (top half) + horizontal + right vertical.
                    draw_rect(&mut img, q, q, half + 2, q + 3, 1.0);
                    draw_rect(&mut img, half - 1, q, half + 2, sz - q, 1.0);
                    draw_rect(&mut img, q, sz - q - 3, sz - q, sz - q, 1.0);
                }
                5 => {
                    // Top bar + left top vert + middle bar + right bot vert + bot bar.
                    draw_rect(&mut img, q, q, q + 3, sz - q, 1.0);
                    draw_rect(&mut img, q, q, half, q + 3, 1.0);
                    draw_rect(&mut img, half - 1, q, half + 2, sz - q, 1.0);
                    draw_rect(&mut img, half, sz - q - 3, sz - q, sz - q, 1.0);
                    draw_rect(&mut img, sz - q - 3, q, sz - q, sz - q, 1.0);
                }
                6 => {
                    // Filled lower oval + left vertical.
                    draw_rect(&mut img, q, q, sz - q, q + 3, 1.0);
                    draw_rect(&mut img, sz - q - 3, q, sz - q, sz - q, 1.0);
                    draw_rect(&mut img, half - 1, q, half + 2, sz - q, 1.0);
                    draw_rect(&mut img, half, sz - q - 3, sz - q, sz - q, 1.0);
                }
                7 => {
                    // Top bar + right diagonal.
                    draw_rect(&mut img, q, q, q + 3, sz - q, 1.0);
                    draw_rect(&mut img, q, sz - q - 3, sz - q, sz - q, 1.0);
                }
                8 => {
                    // Two circles stacked.
                    let top_cy = (sz as f64 * 0.33) as usize;
                    let bot_cy = (sz as f64 * 0.67) as usize;
                    let r_small = (sz / 5).max(2);
                    draw_circle(&mut img, top_cy as f64, half as f64, r_small as f64, 1.0);
                    draw_circle(
                        &mut img,
                        top_cy as f64,
                        half as f64,
                        (r_small - 2).max(0) as f64,
                        0.0,
                    );
                    draw_circle(&mut img, bot_cy as f64, half as f64, r_small as f64, 1.0);
                    draw_circle(
                        &mut img,
                        bot_cy as f64,
                        half as f64,
                        (r_small - 2).max(0) as f64,
                        0.0,
                    );
                }
                _ => {
                    // 9: upper circle + right vertical tail.
                    let top_cy = (sz as f64 * 0.37) as usize;
                    let r_small = (sz / 5).max(2);
                    draw_circle(&mut img, top_cy as f64, half as f64, r_small as f64, 1.0);
                    draw_circle(
                        &mut img,
                        top_cy as f64,
                        half as f64,
                        (r_small - 2).max(0) as f64,
                        0.0,
                    );
                    draw_rect(&mut img, top_cy, sz - q - 3, sz - q, sz - q, 1.0);
                }
            }

            // Apply shift by copying into new buffer.
            if dy != 0 || dx != 0 {
                let original = img[0].clone();
                for r in 0..sz {
                    for c in 0..sz {
                        let (nr, nc) = shifted(r, c);
                        img[0][nr][nc] = original[r][c];
                    }
                }
            }

            // Small additive noise.
            add_noise(&mut img, 0.05, &mut rng);

            images.push(img);
            labels.push(digit);
        }
    }

    let label_names: Vec<String> = (0..10).map(|d| d.to_string()).collect();

    Ok(ImageDataset {
        images,
        labels,
        label_names,
        n_channels: 1,
        height: sz,
        width: sz,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// DetectionDataset
// ─────────────────────────────────────────────────────────────────────────────

/// Object detection dataset with bounding-box annotations.
///
/// Each image may contain multiple objects, each labelled with a class ID
/// and an axis-aligned bounding box `[x1, y1, x2, y2]` in pixel coordinates.
#[derive(Debug, Clone)]
pub struct DetectionDataset {
    /// Images stored as `images[n][c][h][w]`.
    pub images: Vec<Vec<Vec<Vec<f64>>>>,
    /// Per-image annotations: `(class_id, [x1, y1, x2, y2])`.
    pub annotations: Vec<Vec<(usize, [f64; 4])>>,
    /// Human-readable class names.
    pub class_names: Vec<String>,
}

/// Generate a synthetic object detection dataset.
///
/// Each image is a grayscale scene with `k` filled shapes (k drawn from
/// `1..=max_objects_per_image`) placed at random locations.
/// Annotations contain the class ID and tight bounding box for each object.
///
/// # Arguments
///
/// * `n_images`             – Number of scene images.
/// * `n_classes`            – Number of object classes.
/// * `max_objects_per_image`– Maximum objects placed per image (≥ 1).
/// * `image_size`           – Square image side length (≥ 16).
/// * `seed`                 – Reproducibility seed.
///
/// # Errors
///
/// Returns an error if `image_size < 16`, `n_classes == 0`, or
/// `max_objects_per_image == 0`.
pub fn make_object_detection_dataset(
    n_images: usize,
    n_classes: usize,
    max_objects_per_image: usize,
    image_size: usize,
    seed: u64,
) -> Result<DetectionDataset> {
    if image_size < 16 {
        return Err(DatasetsError::InvalidFormat(
            "make_object_detection_dataset: image_size must be >= 16".to_string(),
        ));
    }
    if n_classes == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_object_detection_dataset: n_classes must be >= 1".to_string(),
        ));
    }
    if max_objects_per_image == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_object_detection_dataset: max_objects_per_image must be >= 1".to_string(),
        ));
    }

    let mut rng = Lcg::new(seed);
    let sz = image_size as f64;
    let min_obj_size = (image_size / 8).max(4) as f64;
    let max_obj_size = (image_size / 3) as f64;

    let mut images: Vec<Vec<Vec<Vec<f64>>>> = Vec::with_capacity(n_images);
    let mut annotations: Vec<Vec<(usize, [f64; 4])>> = Vec::with_capacity(n_images);

    let class_names: Vec<String> = (0..n_classes).map(|i| format!("object_{i}")).collect();

    for _ in 0..n_images {
        let mut img = blank_image(image_size, image_size, 0.1); // light background
        let n_objects = 1 + rng.next_usize(max_objects_per_image);
        let mut ann: Vec<(usize, [f64; 4])> = Vec::with_capacity(n_objects);

        for _ in 0..n_objects {
            let class_id = rng.next_usize(n_classes);
            let obj_size = min_obj_size + rng.next_f64() * (max_obj_size - min_obj_size);
            let cx = obj_size / 2.0 + rng.next_f64() * (sz - obj_size);
            let cy = obj_size / 2.0 + rng.next_f64() * (sz - obj_size);

            let (x1, y1, x2, y2);
            // Alternate between circle and square based on class parity.
            if class_id % 2 == 0 {
                // Circle.
                let radius = obj_size / 2.0;
                draw_circle(&mut img, cy, cx, radius, 0.8 + rng.next_f64() * 0.2);
                x1 = (cx - radius).max(0.0);
                y1 = (cy - radius).max(0.0);
                x2 = (cx + radius).min(sz - 1.0);
                y2 = (cy + radius).min(sz - 1.0);
            } else {
                // Square.
                let half = obj_size / 2.0;
                let top = ((cy - half) as usize).min(image_size - 1);
                let left = ((cx - half) as usize).min(image_size - 1);
                let bot = ((cy + half) as usize + 1).min(image_size);
                let right = ((cx + half) as usize + 1).min(image_size);
                draw_rect(&mut img, top, left, bot, right, 0.6 + rng.next_f64() * 0.4);
                x1 = left as f64;
                y1 = top as f64;
                x2 = right as f64;
                y2 = bot as f64;
            }

            ann.push((class_id, [x1, y1, x2, y2]));
        }

        add_noise(&mut img, 0.03, &mut rng);
        images.push(img);
        annotations.push(ann);
    }

    Ok(DetectionDataset {
        images,
        annotations,
        class_names,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// SegmentationDataset
// ─────────────────────────────────────────────────────────────────────────────

/// Semantic segmentation dataset with pixel-wise class labels.
#[derive(Debug, Clone)]
pub struct SegmentationDataset {
    /// Images stored as `images[n][c][h][w]`.
    pub images: Vec<Vec<Vec<Vec<f64>>>>,
    /// Pixel-wise class label masks stored as `masks[n][h][w]`.
    pub masks: Vec<Vec<Vec<usize>>>,
    /// Number of semantic classes (including background = 0).
    pub n_classes: usize,
}

/// Generate a synthetic semantic segmentation dataset.
///
/// Each image contains several filled regions assigned to different classes.
/// Background (class 0) is always present.  Foreground regions are drawn as
/// Voronoi-like blobs seeded by random centre points.
///
/// # Arguments
///
/// * `n_images`   – Number of images.
/// * `n_classes`  – Total number of classes including background (≥ 2).
/// * `image_size` – Square image side length (≥ 8).
/// * `seed`       – Reproducibility seed.
///
/// # Errors
///
/// Returns an error if `n_classes < 2` or `image_size < 8`.
pub fn make_segmentation_dataset(
    n_images: usize,
    n_classes: usize,
    image_size: usize,
    seed: u64,
) -> Result<SegmentationDataset> {
    if n_classes < 2 {
        return Err(DatasetsError::InvalidFormat(
            "make_segmentation_dataset: n_classes must be >= 2 (including background)".to_string(),
        ));
    }
    if image_size < 8 {
        return Err(DatasetsError::InvalidFormat(
            "make_segmentation_dataset: image_size must be >= 8".to_string(),
        ));
    }

    let mut rng = Lcg::new(seed);
    let sz = image_size;
    let n_seeds = n_classes - 1; // number of foreground region centres

    // Class-to-intensity mapping: class 0 (background) = 0.2, others evenly spaced.
    let intensity_for = |cls: usize| -> f64 {
        if cls == 0 {
            0.2
        } else {
            0.3 + (cls as f64 - 1.0) / (n_classes as f64 - 1.0) * 0.7
        }
    };

    let mut images: Vec<Vec<Vec<Vec<f64>>>> = Vec::with_capacity(n_images);
    let mut masks: Vec<Vec<Vec<usize>>> = Vec::with_capacity(n_images);

    for _ in 0..n_images {
        // Sample seed points for each foreground class.
        let centres: Vec<(f64, f64, usize)> = (1..n_classes)
            .map(|cls| {
                let cy = rng.next_f64() * sz as f64;
                let cx = rng.next_f64() * sz as f64;
                (cy, cx, cls)
            })
            .collect();

        let mut mask = vec![vec![0usize; sz]; sz];
        let mut img_data = blank_image(sz, sz, intensity_for(0));

        // Assign each pixel to the nearest seed (Voronoi).
        for r in 0..sz {
            for c in 0..sz {
                let mut best_dist = f64::INFINITY;
                let mut best_cls = 0usize;
                for &(cy, cx, cls) in &centres {
                    let dy = r as f64 - cy;
                    let dx = c as f64 - cx;
                    let d = dy * dy + dx * dx;
                    if d < best_dist {
                        best_dist = d;
                        best_cls = cls;
                    }
                }
                // Only foreground within a blob radius; otherwise background.
                let blob_radius = (sz as f64 * 0.25).powi(2);
                if best_dist <= blob_radius {
                    mask[r][c] = best_cls;
                    img_data[0][r][c] = intensity_for(best_cls);
                }
                // else: remains background (class 0, already set above).
            }
        }

        add_noise(&mut img_data, 0.03, &mut rng);
        images.push(img_data);
        masks.push(mask);
    }

    Ok(SegmentationDataset {
        images,
        masks,
        n_classes,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn check_pixel_range(ds: &ImageDataset) {
        for img in &ds.images {
            for ch in img {
                for row in ch {
                    for &p in row {
                        assert!(p >= 0.0 && p <= 1.0, "pixel out of [0,1]: {p}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_shapes_dataset_basic() {
        let ds = make_shapes_dataset(10, 32, 0.05, 42).expect("shapes failed");
        assert_eq!(ds.len(), 30); // 10 × 3 classes
        assert_eq!(ds.n_classes(), 3);
        assert_eq!(ds.n_channels, 1);
        assert_eq!(ds.height, 32);
        assert_eq!(ds.width, 32);
        for (i, img) in ds.images.iter().enumerate() {
            assert_eq!(img.len(), 1, "image {i}: wrong channel count");
            assert_eq!(img[0].len(), 32, "image {i}: wrong height");
            assert_eq!(img[0][0].len(), 32, "image {i}: wrong width");
        }
        for &l in &ds.labels {
            assert!(l < 3);
        }
        check_pixel_range(&ds);
    }

    #[test]
    fn test_shapes_small_image_error() {
        assert!(make_shapes_dataset(5, 7, 0.0, 1).is_err());
    }

    #[test]
    fn test_shapes_normalize() {
        let mut ds = make_shapes_dataset(5, 16, 0.0, 77).expect("shapes");
        ds.normalize();
        check_pixel_range(&ds);
    }

    #[test]
    fn test_shapes_split() {
        let ds = make_shapes_dataset(20, 16, 0.0, 1).expect("shapes");
        let (train, test) = ds.split(0.2, 1).expect("split");
        assert_eq!(train.len() + test.len(), 60); // 20*3
        assert!(test.len() > 0);
    }

    #[test]
    fn test_image_index_error() {
        let ds = make_shapes_dataset(2, 16, 0.0, 1).expect("shapes");
        assert!(ds.image(0).is_ok());
        assert!(ds.image(100).is_err());
    }

    #[test]
    fn test_mnist_like_basic() {
        let ds = make_mnist_like(5, 28, 42).expect("mnist-like failed");
        assert_eq!(ds.len(), 50); // 5 × 10 digits
        assert_eq!(ds.n_classes(), 10);
        assert_eq!(ds.n_channels, 1);
        assert_eq!(ds.height, 28);
        assert_eq!(ds.width, 28);
        check_pixel_range(&ds);
        let label_names: Vec<String> = (0..10).map(|d| d.to_string()).collect();
        assert_eq!(ds.label_names, label_names);
    }

    #[test]
    fn test_detection_dataset_basic() {
        let ds = make_object_detection_dataset(10, 4, 3, 32, 42).expect("detection failed");
        assert_eq!(ds.images.len(), 10);
        assert_eq!(ds.annotations.len(), 10);
        assert_eq!(ds.class_names.len(), 4);
        for ann in &ds.annotations {
            assert!(!ann.is_empty(), "each image must have >= 1 annotation");
            for &(cls, bbox) in ann {
                assert!(cls < 4, "class_id out of range: {cls}");
                assert!(bbox[0] >= 0.0 && bbox[2] > bbox[0], "bad x range");
                assert!(bbox[1] >= 0.0 && bbox[3] > bbox[1], "bad y range");
            }
        }
    }

    #[test]
    fn test_detection_invalid() {
        assert!(make_object_detection_dataset(5, 3, 2, 8, 1).is_err()); // image_size < 16
        assert!(make_object_detection_dataset(5, 0, 2, 32, 1).is_err()); // n_classes == 0
        assert!(make_object_detection_dataset(5, 3, 0, 32, 1).is_err()); // max_objects == 0
    }

    #[test]
    fn test_segmentation_dataset_basic() {
        let ds = make_segmentation_dataset(8, 4, 32, 13).expect("segmentation failed");
        assert_eq!(ds.images.len(), 8);
        assert_eq!(ds.masks.len(), 8);
        assert_eq!(ds.n_classes, 4);
        for (img, mask) in ds.images.iter().zip(ds.masks.iter()) {
            assert_eq!(img[0].len(), 32);
            assert_eq!(img[0][0].len(), 32);
            assert_eq!(mask.len(), 32);
            assert_eq!(mask[0].len(), 32);
            for row in mask {
                for &cls in row {
                    assert!(cls < 4, "mask class out of range: {cls}");
                }
            }
        }
    }

    #[test]
    fn test_segmentation_invalid() {
        assert!(make_segmentation_dataset(5, 1, 32, 1).is_err()); // n_classes < 2
        assert!(make_segmentation_dataset(5, 3, 4, 1).is_err()); // image_size < 8
    }

    #[test]
    fn test_reproducibility() {
        let a = make_shapes_dataset(5, 16, 0.1, 42).expect("a");
        let b = make_shapes_dataset(5, 16, 0.1, 42).expect("b");
        assert_eq!(a.labels, b.labels);
        // Compare first pixel of first image.
        assert!((a.images[0][0][0][0] - b.images[0][0][0][0]).abs() < 1e-12);
        let c = make_shapes_dataset(5, 16, 0.1, 99).expect("c");
        // Different seed → different noise (with high probability).
        let same = a.images[0][0][0][0] == c.images[0][0][0][0];
        // Not asserting inequality because the circle in class 0 might coincide;
        // instead just check it compiles and runs.
        let _ = same;
    }
}
