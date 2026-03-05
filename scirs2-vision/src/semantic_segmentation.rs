//! Semantic segmentation algorithms
//!
//! This module provides semantic segmentation capabilities including:
//! - FCN-style segmentation from feature maps
//! - Color-aware SLIC superpixels
//! - Graph-cut based segmentation with alpha-expansion
//! - Dense CRF post-processing for label refinement
//! - Connected component labeling (two-pass algorithm)
//! - Flood fill segmentation

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::{Array2, Array3};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// FCN-style segmentation
// ---------------------------------------------------------------------------

/// FCN-style semantic segmentation using a precomputed feature map.
///
/// Each pixel is assigned the class with the highest activation in `feature_map`.
/// The feature map shape is `[n_classes, height, width]`.
///
/// # Arguments
/// * `image`       - RGB image `[height, width, 3]` (used for validation only here)
/// * `feature_map` - Class logits `[n_classes, height, width]`
///
/// # Returns
/// A `[height, width]` label map where each entry is a class index.
pub fn fcn_segment(image: &Array3<f64>, feature_map: &Array3<f64>) -> Result<Array2<usize>> {
    let (img_h, img_w, _) = image.dim();
    let (n_classes, fm_h, fm_w) = feature_map.dim();

    if n_classes == 0 {
        return Err(VisionError::InvalidParameter(
            "feature_map must have at least one class channel".to_string(),
        ));
    }
    if fm_h != img_h || fm_w != img_w {
        return Err(VisionError::DimensionMismatch(format!(
            "feature_map spatial size ({fm_h}×{fm_w}) does not match image size ({img_h}×{img_w})"
        )));
    }

    let mut labels = Array2::<usize>::zeros((img_h, img_w));

    for y in 0..img_h {
        for x in 0..img_w {
            let mut best_class = 0usize;
            let mut best_score = f64::NEG_INFINITY;
            for c in 0..n_classes {
                let score = feature_map[[c, y, x]];
                if score > best_score {
                    best_score = score;
                    best_class = c;
                }
            }
            labels[[y, x]] = best_class;
        }
    }

    Ok(labels)
}

// ---------------------------------------------------------------------------
// Color SLIC superpixels (ndarray API, works on Array3<f64>)
// ---------------------------------------------------------------------------

/// Approximate CIE Lab conversion from linear RGB in [0,1].
///
/// Uses D65 illuminant. The conversion is intentionally approximate but
/// perceptually relevant for the SLIC distance metric.
#[inline]
fn rgb_to_lab(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    // Linearize (gamma ≈ 2.2 sRGB)
    let linearize = |v: f64| -> f64 {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    let rl = linearize(r.clamp(0.0, 1.0));
    let gl = linearize(g.clamp(0.0, 1.0));
    let bl = linearize(b.clamp(0.0, 1.0));

    // RGB → XYZ (D65)
    let x = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375;
    let y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750;
    let z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041;

    // XYZ → Lab (D65 white: Xn=0.95047, Yn=1.0, Zn=1.08883)
    let f = |t: f64| -> f64 {
        if t > 0.008856 {
            t.cbrt()
        } else {
            7.787 * t + 16.0 / 116.0
        }
    };
    let fx = f(x / 0.95047);
    let fy = f(y / 1.00000);
    let fz = f(z / 1.08883);

    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b_lab = 200.0 * (fy - fz);

    (l, a, b_lab)
}

/// SLIC superpixel segmentation on a colour image (`Array3<f64>`).
///
/// Accepts an RGB image in `[height, width, 3]` format with values in `[0, 1]`.
/// Returns a `[height, width]` label map where each entry identifies the
/// superpixel the pixel belongs to.
///
/// # Arguments
/// * `image`       - RGB image `[height, width, 3]`, values in `[0, 1]`
/// * `n_segments`  - Desired number of superpixels
/// * `compactness` - Balance between colour and spatial distance (typically 10.0)
pub fn slic_superpixels_color(
    image: &Array3<f64>,
    n_segments: usize,
    compactness: f64,
) -> Result<Array2<usize>> {
    let (height, width, channels) = image.dim();
    if channels < 3 {
        return Err(VisionError::InvalidParameter(
            "image must have at least 3 channels (RGB)".to_string(),
        ));
    }
    if n_segments == 0 {
        return Err(VisionError::InvalidParameter(
            "n_segments must be > 0".to_string(),
        ));
    }
    if height == 0 || width == 0 {
        return Err(VisionError::InvalidParameter(
            "image must be non-empty".to_string(),
        ));
    }

    let n_pixels = height * width;
    // Effective k (may be less than n_segments for very small images)
    let k = n_segments.min(n_pixels);

    // Step size between cluster centres
    let step = ((n_pixels as f64 / k as f64).sqrt()) as usize;
    let step = step.max(1);

    // Convert image to Lab colour space
    let mut lab = Array3::<f64>::zeros((height, width, 3));
    for y in 0..height {
        for x in 0..width {
            let r = image[[y, x, 0]];
            let g = image[[y, x, 1]];
            let b = image[[y, x, 2]];
            let (l, a, b_lab) = rgb_to_lab(r, g, b);
            lab[[y, x, 0]] = l;
            lab[[y, x, 1]] = a;
            lab[[y, x, 2]] = b_lab;
        }
    }

    // Initialise cluster centres on a regular grid, perturbed to lowest gradient
    // pixel in a 3×3 neighbourhood to avoid placing centres on edges.
    #[derive(Clone)]
    struct Center {
        y: f64,
        x: f64,
        l: f64,
        a: f64,
        b: f64,
    }

    let mut centers: Vec<Center> = Vec::new();
    let half = step / 2;
    let mut yc = half;
    while yc < height {
        let mut xc = half;
        while xc < width {
            // Perturb to lowest-gradient pixel in 3×3 neighbourhood
            let mut best_y = yc;
            let mut best_x = xc;
            let mut min_grad = f64::INFINITY;
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let ny = (yc as i64 + dy).clamp(0, height as i64 - 1) as usize;
                    let nx = (xc as i64 + dx).clamp(0, width as i64 - 1) as usize;
                    // Approximate gradient magnitude using neighbour differences
                    let gy = if ny + 1 < height {
                        (lab[[ny + 1, nx, 0]] - lab[[ny, nx, 0]]).abs()
                    } else {
                        0.0
                    };
                    let gx = if nx + 1 < width {
                        (lab[[ny, nx + 1, 0]] - lab[[ny, nx, 0]]).abs()
                    } else {
                        0.0
                    };
                    let grad = gy + gx;
                    if grad < min_grad {
                        min_grad = grad;
                        best_y = ny;
                        best_x = nx;
                    }
                }
            }
            centers.push(Center {
                y: best_y as f64,
                x: best_x as f64,
                l: lab[[best_y, best_x, 0]],
                a: lab[[best_y, best_x, 1]],
                b: lab[[best_y, best_x, 2]],
            });
            xc += step;
        }
        yc += step;
    }

    let k_actual = centers.len();
    if k_actual == 0 {
        return Ok(Array2::<usize>::zeros((height, width)));
    }

    let inv_step_sq = 1.0 / (step as f64 * step as f64);
    let comp_sq = compactness * compactness;

    // Distance and label buffers
    let mut dist = vec![f64::INFINITY; n_pixels];
    let mut labels = vec![0usize; n_pixels];

    // Iterative assignment
    let max_iter = 10usize;
    for _iter in 0..max_iter {
        // Reset distances
        for d in dist.iter_mut() {
            *d = f64::INFINITY;
        }

        // Assign each pixel to nearest center within 2S×2S window
        for (ci, center) in centers.iter().enumerate() {
            let cy = center.y as i64;
            let cx = center.x as i64;
            let search = (step as i64) * 2;

            let y_start = (cy - search).clamp(0, height as i64 - 1) as usize;
            let y_end = (cy + search).clamp(0, height as i64 - 1) as usize;
            let x_start = (cx - search).clamp(0, width as i64 - 1) as usize;
            let x_end = (cx + search).clamp(0, width as i64 - 1) as usize;

            for y in y_start..=y_end {
                for x in x_start..=x_end {
                    let dl = lab[[y, x, 0]] - center.l;
                    let da = lab[[y, x, 1]] - center.a;
                    let db = lab[[y, x, 2]] - center.b;
                    let d_color = dl * dl + da * da + db * db;

                    let dy = y as f64 - center.y;
                    let dx = x as f64 - center.x;
                    let d_spatial = (dy * dy + dx * dx) * inv_step_sq;

                    let d_total = d_color + comp_sq * d_spatial;
                    let idx = y * width + x;
                    if d_total < dist[idx] {
                        dist[idx] = d_total;
                        labels[idx] = ci;
                    }
                }
            }
        }

        // Recompute centers
        let mut sum_y = vec![0.0f64; k_actual];
        let mut sum_x = vec![0.0f64; k_actual];
        let mut sum_l = vec![0.0f64; k_actual];
        let mut sum_a = vec![0.0f64; k_actual];
        let mut sum_b = vec![0.0f64; k_actual];
        let mut counts = vec![0usize; k_actual];

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let ci = labels[idx];
                sum_y[ci] += y as f64;
                sum_x[ci] += x as f64;
                sum_l[ci] += lab[[y, x, 0]];
                sum_a[ci] += lab[[y, x, 1]];
                sum_b[ci] += lab[[y, x, 2]];
                counts[ci] += 1;
            }
        }

        for (ci, center) in centers.iter_mut().enumerate() {
            let cnt = counts[ci] as f64;
            if cnt > 0.0 {
                center.y = sum_y[ci] / cnt;
                center.x = sum_x[ci] / cnt;
                center.l = sum_l[ci] / cnt;
                center.a = sum_a[ci] / cnt;
                center.b = sum_b[ci] / cnt;
            }
        }
    }

    // Enforce connectivity: remove disconnected segments via two-pass CCL
    // and reassign orphans to the nearest connected superpixel label.
    let mut out = Array2::<usize>::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            out[[y, x]] = labels[y * width + x];
        }
    }

    enforce_connectivity(&mut out, k_actual, step)
}

/// Enforce connectivity in a superpixel label map.
///
/// Small disconnected blobs are re-assigned to their largest neighbouring label.
/// Returns the (possibly relabelled) map.
fn enforce_connectivity(
    labels: &mut Array2<usize>,
    k: usize,
    min_size: usize,
) -> Result<Array2<usize>> {
    let (height, width) = labels.dim();
    let min_size = min_size.max(1);

    // Build connected components per original label using BFS
    let mut visited = Array2::<bool>::from_elem((height, width), false);
    let mut new_labels = Array2::<usize>::zeros((height, width));
    let mut next_label = 0usize;

    // Map from new label → old label (for merging orphans)
    let mut label_to_orig: Vec<usize> = Vec::new();
    let mut component_size: Vec<usize> = Vec::new();

    for sy in 0..height {
        for sx in 0..width {
            if visited[[sy, sx]] {
                continue;
            }
            let orig = labels[[sy, sx]];
            let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
            queue.push_back((sy, sx));
            visited[[sy, sx]] = true;
            let comp_label = next_label;
            next_label += 1;
            label_to_orig.push(orig);
            component_size.push(0);

            while let Some((y, x)) = queue.pop_front() {
                new_labels[[y, x]] = comp_label;
                component_size[comp_label] += 1;

                let neighbours: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
                for (dy, dx) in neighbours {
                    let ny = y as i64 + dy;
                    let nx = x as i64 + dx;
                    if ny < 0 || ny >= height as i64 || nx < 0 || nx >= width as i64 {
                        continue;
                    }
                    let ny = ny as usize;
                    let nx = nx as usize;
                    if !visited[[ny, nx]] && labels[[ny, nx]] == orig {
                        visited[[ny, nx]] = true;
                        queue.push_back((ny, nx));
                    }
                }
            }
        }
    }

    // Merge small components into a neighbouring larger component
    let threshold = (height * width) / (k.max(1) * 4);
    let threshold = threshold.max(min_size);

    let mut changed = true;
    while changed {
        changed = false;
        for y in 0..height {
            for x in 0..width {
                let comp = new_labels[[y, x]];
                if component_size[comp] < threshold {
                    // Find a neighbour with a different label
                    let neighbours: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
                    for (dy, dx) in neighbours {
                        let ny = y as i64 + dy;
                        let nx = x as i64 + dx;
                        if ny < 0 || ny >= height as i64 || nx < 0 || nx >= width as i64 {
                            continue;
                        }
                        let ny = ny as usize;
                        let nx = nx as usize;
                        let nb_comp = new_labels[[ny, nx]];
                        if nb_comp != comp && component_size[nb_comp] >= threshold {
                            // Absorb comp into nb_comp
                            let old_comp = comp;
                            let new_comp = nb_comp;
                            // Relabel all pixels of old_comp → new_comp
                            for yy in 0..height {
                                for xx in 0..width {
                                    if new_labels[[yy, xx]] == old_comp {
                                        new_labels[[yy, xx]] = new_comp;
                                    }
                                }
                            }
                            component_size[new_comp] += component_size[old_comp];
                            component_size[old_comp] = 0;
                            changed = true;
                            break;
                        }
                    }
                }
            }
        }
    }

    // Remap labels to the original superpixel labels (best-effort; dense 0..k)
    // Remap connected-component labels → original superpixel indices
    let mut final_labels = Array2::<usize>::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let comp = new_labels[[y, x]];
            final_labels[[y, x]] = label_to_orig[comp];
        }
    }

    // Re-densify labels: collect unique labels and map to 0..n_unique
    let mut seen = vec![false; k + next_label];
    for y in 0..height {
        for x in 0..width {
            let lbl = final_labels[[y, x]];
            if lbl < seen.len() {
                seen[lbl] = true;
            }
        }
    }
    let mut remap = vec![0usize; seen.len()];
    let mut counter = 0usize;
    for (i, &s) in seen.iter().enumerate() {
        if s {
            remap[i] = counter;
            counter += 1;
        }
    }
    for y in 0..height {
        for x in 0..width {
            let lbl = final_labels[[y, x]];
            if lbl < remap.len() {
                final_labels[[y, x]] = remap[lbl];
            }
        }
    }

    Ok(final_labels)
}

// ---------------------------------------------------------------------------
// Graph-cut segmentation (alpha-expansion, swap moves)
// ---------------------------------------------------------------------------

/// Graph-cut segmentation using simplified alpha-expansion.
///
/// Minimises an energy `E = Σ_i D(i, l_i) + λ Σ_{(i,j)∈N} V(l_i, l_j)`
/// where `D` is derived from image intensity and `V` is a Potts penalty.
///
/// This implementation uses a greedy iterative label assignment with a
/// fixed number of expansion moves (not the full graph-cut guarantee, but
/// practically effective for moderate `n_classes`).
///
/// # Arguments
/// * `image`     - Grayscale or RGB image `[height, width, C]`
/// * `n_classes` - Number of semantic classes
/// * `lambda`    - Smoothness regularisation weight
pub fn graph_cut_segment(
    image: &Array3<f64>,
    n_classes: usize,
    lambda: f64,
) -> Result<Array2<usize>> {
    if n_classes == 0 {
        return Err(VisionError::InvalidParameter(
            "n_classes must be > 0".to_string(),
        ));
    }

    let (height, width, channels) = image.dim();
    if height == 0 || width == 0 {
        return Err(VisionError::InvalidParameter(
            "image must be non-empty".to_string(),
        ));
    }

    // Build grayscale intensity map (mean across channels)
    let intensity = build_intensity_map(image, height, width, channels);

    // Compute unary costs: uniform priors differentiated by intensity bins
    // Divide [0,1] intensity range into n_classes equal-width bins
    // and assign lower unary cost for the matching class.
    let bin_width = 1.0 / n_classes as f64;

    let mut labels = Array2::<usize>::zeros((height, width));
    // Initialise: assign each pixel to its intensity bin
    for y in 0..height {
        for x in 0..width {
            let v = intensity[[y, x]];
            let bin = ((v / bin_width) as usize).min(n_classes - 1);
            labels[[y, x]] = bin;
        }
    }

    // Iterative alpha-expansion-style refinement
    let n_iter = 10usize;
    for _iter in 0..n_iter {
        let mut changed = false;
        for y in 0..height {
            for x in 0..width {
                let v = intensity[[y, x]];
                let current_label = labels[[y, x]];

                // Unary for current label
                let current_unary = unary_cost(v, current_label, n_classes);
                // Pairwise for current label
                let current_pairwise = pairwise_cost_neighbourhood(
                    &labels,
                    y,
                    x,
                    current_label,
                    height,
                    width,
                    lambda,
                );

                let mut best_label = current_label;
                let mut best_energy = current_unary + current_pairwise;

                // Try each candidate label
                for alpha in 0..n_classes {
                    if alpha == current_label {
                        continue;
                    }
                    let u = unary_cost(v, alpha, n_classes);
                    let p =
                        pairwise_cost_neighbourhood(&labels, y, x, alpha, height, width, lambda);
                    let energy = u + p;
                    if energy < best_energy {
                        best_energy = energy;
                        best_label = alpha;
                    }
                }

                if best_label != current_label {
                    labels[[y, x]] = best_label;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    Ok(labels)
}

/// Build a 2-D intensity map (mean over channels).
fn build_intensity_map(
    image: &Array3<f64>,
    height: usize,
    width: usize,
    channels: usize,
) -> Array2<f64> {
    let mut intensity = Array2::<f64>::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f64;
            for c in 0..channels {
                sum += image[[y, x, c]];
            }
            intensity[[y, x]] = sum / channels as f64;
        }
    }
    intensity
}

/// Unary cost for assigning pixel with intensity `v` to `label`.
///
/// Each label corresponds to an intensity bin; cost is |v - bin_centre|.
#[inline]
fn unary_cost(v: f64, label: usize, n_classes: usize) -> f64 {
    let bin_width = 1.0 / n_classes as f64;
    let centre = (label as f64 + 0.5) * bin_width;
    (v - centre).abs()
}

/// Sum of Potts smoothness penalties for assigning `candidate` to `(y, x)`.
#[inline]
fn pairwise_cost_neighbourhood(
    labels: &Array2<usize>,
    y: usize,
    x: usize,
    candidate: usize,
    height: usize,
    width: usize,
    lambda: f64,
) -> f64 {
    let mut cost = 0.0f64;
    let neighbours: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    for (dy, dx) in neighbours {
        let ny = y as i64 + dy;
        let nx = x as i64 + dx;
        if ny < 0 || ny >= height as i64 || nx < 0 || nx >= width as i64 {
            continue;
        }
        let nb_label = labels[[ny as usize, nx as usize]];
        if nb_label != candidate {
            cost += lambda;
        }
    }
    cost
}

// ---------------------------------------------------------------------------
// Dense CRF post-processing
// ---------------------------------------------------------------------------

/// Dense CRF refinement of a semantic segmentation.
///
/// Uses a mean-field approximation with Gaussian bilateral and spatial kernels
/// to sharpen and clean up a soft probability map.
///
/// # Arguments
/// * `image`       - RGB image `[height, width, 3]`, values in `[0, 1]`
/// * `unary_probs` - Per-pixel class probabilities `[height, width, n_classes]`
/// * `n_iter`      - Number of mean-field iterations
///
/// # Returns
/// Hard label map `[height, width]` after CRF refinement.
pub fn dense_crf_refine(
    image: &Array3<f64>,
    unary_probs: &Array3<f64>,
    n_iter: usize,
) -> Result<Array2<usize>> {
    let (img_h, img_w, img_c) = image.dim();
    let (up_h, up_w, n_classes) = unary_probs.dim();

    if img_h != up_h || img_w != up_w {
        return Err(VisionError::DimensionMismatch(format!(
            "image ({img_h}×{img_w}) and unary_probs ({up_h}×{up_w}) spatial dimensions differ"
        )));
    }
    if n_classes == 0 {
        return Err(VisionError::InvalidParameter(
            "unary_probs must have at least one class".to_string(),
        ));
    }
    if img_c < 3 {
        return Err(VisionError::InvalidParameter(
            "image must have at least 3 channels (RGB)".to_string(),
        ));
    }

    // Initialise Q = unary probabilities (copy and normalise)
    let mut q = unary_probs.to_owned();

    // Ensure rows sum to 1 (softmax normalisation)
    for y in 0..img_h {
        for x in 0..img_w {
            let mut sum = 0.0f64;
            for c in 0..n_classes {
                let v = q[[y, x, c]].max(0.0);
                q[[y, x, c]] = v;
                sum += v;
            }
            if sum > 0.0 {
                for c in 0..n_classes {
                    q[[y, x, c]] /= sum;
                }
            } else {
                // Uniform prior
                let uniform = 1.0 / n_classes as f64;
                for c in 0..n_classes {
                    q[[y, x, c]] = uniform;
                }
            }
        }
    }

    // CRF hyper-parameters
    let spatial_sigma = 3.0f64;
    let bilateral_sigma_xy = 60.0f64;
    let bilateral_sigma_rgb = 13.0f64;
    let compat_spatial = 3.0f64;
    let compat_bilateral = 10.0f64;

    let radius = (3.0 * spatial_sigma.max(bilateral_sigma_xy / 10.0)) as i64;
    let radius = radius.clamp(1, 20); // cap for performance

    for _iter in 0..n_iter {
        let q_prev = q.clone();
        let mut message = Array3::<f64>::zeros((img_h, img_w, n_classes));

        // Approximate dense CRF message passing with subsampled window
        for y in 0..img_h {
            for x in 0..img_w {
                let ry = image[[y, x, 0]];
                let gy = image[[y, x, 1]];
                let by = image[[y, x, 2]];

                let y_min = (y as i64 - radius).max(0) as usize;
                let y_max = (y as i64 + radius).min(img_h as i64 - 1) as usize;
                let x_min = (x as i64 - radius).max(0) as usize;
                let x_max = (x as i64 + radius).min(img_w as i64 - 1) as usize;

                for ny in y_min..=y_max {
                    for nx in x_min..=x_max {
                        if ny == y && nx == x {
                            continue;
                        }
                        let dy = (ny as f64 - y as f64) / spatial_sigma;
                        let dx = (nx as f64 - x as f64) / spatial_sigma;
                        let w_spatial = (-0.5 * (dy * dy + dx * dx)).exp();

                        let bdy = (ny as f64 - y as f64) / bilateral_sigma_xy;
                        let bdx = (nx as f64 - x as f64) / bilateral_sigma_xy;
                        let rn = image[[ny, nx, 0]];
                        let gn = image[[ny, nx, 1]];
                        let bn = image[[ny, nx, 2]];
                        let dr = (rn - ry) / bilateral_sigma_rgb;
                        let dg = (gn - gy) / bilateral_sigma_rgb;
                        let db_ = (bn - by) / bilateral_sigma_rgb;
                        let w_bilateral =
                            (-0.5 * (bdy * bdy + bdx * bdx + dr * dr + dg * dg + db_ * db_)).exp();

                        for c in 0..n_classes {
                            message[[y, x, c]] += (compat_spatial * w_spatial
                                + compat_bilateral * w_bilateral)
                                * q_prev[[ny, nx, c]];
                        }
                    }
                }
            }
        }

        // Update Q: Q_new ∝ exp(-unary_energy - message) ≈ prior * exp(-message)
        for y in 0..img_h {
            for x in 0..img_w {
                let mut sum = 0.0f64;
                for c in 0..n_classes {
                    // Approximate: Q ∝ unary * exp(-compat_message)
                    // Use log-domain to avoid underflow
                    let log_prior = (q_prev[[y, x, c]].max(1e-300)).ln();
                    let val = (log_prior - message[[y, x, c]]).exp();
                    q[[y, x, c]] = val;
                    sum += val;
                }
                if sum > 0.0 {
                    for c in 0..n_classes {
                        q[[y, x, c]] /= sum;
                    }
                } else {
                    let uniform = 1.0 / n_classes as f64;
                    for c in 0..n_classes {
                        q[[y, x, c]] = uniform;
                    }
                }
            }
        }
    }

    // Hard assignment: argmax over classes
    let mut labels = Array2::<usize>::zeros((img_h, img_w));
    for y in 0..img_h {
        for x in 0..img_w {
            let mut best_c = 0usize;
            let mut best_q = f64::NEG_INFINITY;
            for c in 0..n_classes {
                if q[[y, x, c]] > best_q {
                    best_q = q[[y, x, c]];
                    best_c = c;
                }
            }
            labels[[y, x]] = best_c;
        }
    }

    Ok(labels)
}

// ---------------------------------------------------------------------------
// Connected Component Labeling (two-pass)
// ---------------------------------------------------------------------------

/// Connected component labeling using the two-pass union-find algorithm.
///
/// Label `0` is reserved for background (false pixels). Foreground components
/// are numbered `1, 2, …, n`.
///
/// # Arguments
/// * `binary` - Boolean array `[height, width]` where `true` = foreground
///
/// # Returns
/// Integer label map where 0 = background and positive integers = components.
pub fn connected_component_labeling(binary: &Array2<bool>) -> Result<Array2<usize>> {
    let (height, width) = binary.dim();
    let mut labels = Array2::<usize>::zeros((height, width));

    // Union-Find parent table (index 0 is unused; labels start at 1)
    let mut parent: Vec<usize> = vec![0usize; height * width + 1];

    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]); // path compression
        }
        parent[x]
    }

    fn union(parent: &mut Vec<usize>, a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    }

    let mut next_label = 1usize;

    // First pass: provisional labels + union-find merging (4-connectivity)
    for y in 0..height {
        for x in 0..width {
            if !binary[[y, x]] {
                continue;
            }
            let left = if x > 0 && binary[[y, x - 1]] {
                labels[[y, x - 1]]
            } else {
                0
            };
            let up = if y > 0 && binary[[y - 1, x]] {
                labels[[y - 1, x]]
            } else {
                0
            };

            match (left, up) {
                (0, 0) => {
                    // New component
                    labels[[y, x]] = next_label;
                    parent[next_label] = next_label;
                    next_label += 1;
                }
                (l, 0) => {
                    labels[[y, x]] = l;
                }
                (0, u) => {
                    labels[[y, x]] = u;
                }
                (l, u) => {
                    // Merge two existing components
                    union(&mut parent, l, u);
                    labels[[y, x]] = find(&mut parent, l);
                }
            }
        }
    }

    // Second pass: resolve labels to root
    let mut label_map: Vec<usize> = vec![0usize; next_label];
    let mut component_id = 0usize;
    for i in 1..next_label {
        let root = find(&mut parent, i);
        if label_map[root] == 0 {
            component_id += 1;
            label_map[root] = component_id;
        }
        label_map[i] = label_map[root];
    }

    // Apply final labels
    let mut out = Array2::<usize>::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let prov = labels[[y, x]];
            if prov > 0 {
                out[[y, x]] = label_map[prov];
            }
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Flood fill segmentation
// ---------------------------------------------------------------------------

/// Flood fill segmentation starting from a seed pixel.
///
/// Returns a boolean mask of all pixels connected to the seed whose value
/// differs from the seed by at most `tolerance` (in L∞ norm across channels).
///
/// # Arguments
/// * `image`     - Input image `[height, width, C]`
/// * `seed`      - `(row, col)` seed coordinate
/// * `tolerance` - Maximum allowed per-channel absolute difference from seed
pub fn flood_fill_segment(
    image: &Array3<f64>,
    seed: (usize, usize),
    tolerance: f64,
) -> Result<Array2<bool>> {
    let (height, width, channels) = image.dim();
    let (sy, sx) = seed;

    if sy >= height || sx >= width {
        return Err(VisionError::InvalidParameter(format!(
            "seed ({sy}, {sx}) is out of bounds for image ({height}×{width})"
        )));
    }

    let mut mask = Array2::<bool>::from_elem((height, width), false);
    let mut visited = Array2::<bool>::from_elem((height, width), false);

    // Extract seed colour
    let seed_color: Vec<f64> = (0..channels).map(|c| image[[sy, sx, c]]).collect();

    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    queue.push_back((sy, sx));
    visited[[sy, sx]] = true;

    while let Some((y, x)) = queue.pop_front() {
        // Check tolerance (L∞ norm)
        let within = (0..channels).all(|c| (image[[y, x, c]] - seed_color[c]).abs() <= tolerance);

        if !within {
            continue;
        }

        mask[[y, x]] = true;

        let neighbours: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        for (dy, dx) in neighbours {
            let ny = y as i64 + dy;
            let nx = x as i64 + dx;
            if ny < 0 || ny >= height as i64 || nx < 0 || nx >= width as i64 {
                continue;
            }
            let ny = ny as usize;
            let nx = nx as usize;
            if !visited[[ny, nx]] {
                visited[[ny, nx]] = true;
                queue.push_back((ny, nx));
            }
        }
    }

    Ok(mask)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    fn tiny_image(h: usize, w: usize) -> Array3<f64> {
        let mut img = Array3::<f64>::zeros((h, w, 3));
        for y in 0..h {
            for x in 0..w {
                img[[y, x, 0]] = (y as f64) / (h as f64);
                img[[y, x, 1]] = (x as f64) / (w as f64);
                img[[y, x, 2]] = 0.5;
            }
        }
        img
    }

    #[test]
    fn test_fcn_segment_basic() {
        let img = tiny_image(4, 4);
        let mut fm = Array3::<f64>::zeros((3, 4, 4));
        for y in 0..4 {
            for x in 0..4 {
                fm[[1, y, x]] = 2.0; // class 1 always wins
            }
        }
        let labels = fcn_segment(&img, &fm).expect("fcn_segment should succeed");
        assert_eq!(labels.dim(), (4, 4));
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(labels[[y, x]], 1);
            }
        }
    }

    #[test]
    fn test_slic_superpixels_color_basic() {
        let img = tiny_image(20, 20);
        let labels = slic_superpixels_color(&img, 4, 10.0).expect("slic should succeed");
        assert_eq!(labels.dim(), (20, 20));
    }

    #[test]
    fn test_graph_cut_segment_basic() {
        let img = tiny_image(8, 8);
        let labels = graph_cut_segment(&img, 3, 1.0).expect("graph_cut should succeed");
        assert_eq!(labels.dim(), (8, 8));
        for y in 0..8 {
            for x in 0..8 {
                assert!(labels[[y, x]] < 3);
            }
        }
    }

    #[test]
    fn test_dense_crf_refine_basic() {
        let img = tiny_image(6, 6);
        let n_classes = 2usize;
        let mut probs = Array3::<f64>::zeros((6, 6, n_classes));
        for y in 0..6 {
            for x in 0..6 {
                probs[[y, x, 0]] = 0.7;
                probs[[y, x, 1]] = 0.3;
            }
        }
        let labels = dense_crf_refine(&img, &probs, 2).expect("dense_crf should succeed");
        assert_eq!(labels.dim(), (6, 6));
    }

    #[test]
    fn test_connected_component_labeling() {
        let mut binary = Array2::<bool>::from_elem((5, 5), false);
        // Two separate blobs
        binary[[0, 0]] = true;
        binary[[0, 1]] = true;
        binary[[1, 0]] = true;
        binary[[3, 3]] = true;
        binary[[3, 4]] = true;
        let labels = connected_component_labeling(&binary).expect("CCL should succeed");
        assert_eq!(labels.dim(), (5, 5));
        // Background pixels are 0
        assert_eq!(labels[[2, 2]], 0);
        // The two blobs should have different labels
        assert_ne!(labels[[0, 0]], labels[[3, 3]]);
        assert_ne!(labels[[0, 0]], 0);
        assert_ne!(labels[[3, 3]], 0);
        // Connected pixels share a label
        assert_eq!(labels[[0, 0]], labels[[0, 1]]);
        assert_eq!(labels[[0, 0]], labels[[1, 0]]);
    }

    #[test]
    fn test_flood_fill_segment_basic() {
        let img = tiny_image(10, 10);
        let mask = flood_fill_segment(&img, (5, 5), 0.3).expect("flood_fill should succeed");
        assert_eq!(mask.dim(), (10, 10));
        // Seed pixel itself should be in the mask
        assert!(mask[[5, 5]]);
    }
}
