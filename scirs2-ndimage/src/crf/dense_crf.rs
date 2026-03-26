//! Fully-connected (dense) CRF for label post-processing.
//!
//! Implements mean-field inference for the pairwise fully-connected CRF
//! introduced by Krähenbühl & Koltun (2011).  The spatial and bilateral
//! kernels are approximated with separable Gaussian blurs so that inference
//! runs in O(N · L) time instead of O(N² · L).
//!
//! # Model
//! Energy = Σ_i ψ_u(x_i) + Σ_{i<j} ψ_p(x_i, x_j)
//!
//! Unary:    ψ_u(x_i = l) = − log P(x_i = l)   (provided by caller)
//! Pairwise: ψ_p(x_i = l, x_j = l') = μ(l, l') · [w_s · k_s(i,j) + w_b · k_b(i,j)]
//!
//! where μ(l,l') = 1 if l ≠ l' (Potts model).
//!
//! Kernels (both spatial and bilateral) are approximated by iterating
//! 1-D Gaussian blurs along each image dimension.

/// Configuration for the Dense CRF.
#[derive(Debug, Clone)]
pub struct CrfConfig {
    /// Number of mean-field iterations (default: 5)
    pub n_iter: usize,
    /// Weight of the spatial (appearance) kernel (default: 3.0)
    pub spatial_weight: f64,
    /// Weight of the bilateral (colour + position) kernel (default: 10.0)
    pub bilateral_weight: f64,
    /// Position sigma for the spatial kernel (default: 3.0)
    pub spatial_sigma: f64,
    /// Colour sigma for the bilateral kernel (default: 8.0)
    pub color_sigma: f64,
    /// Position sigma for the bilateral kernel (default: 5.0)
    pub bilateral_sigma_pos: f64,
    /// Number of labels (must be set by the caller)
    pub n_labels: usize,
}

impl Default for CrfConfig {
    fn default() -> Self {
        Self {
            n_iter: 5,
            spatial_weight: 3.0,
            bilateral_weight: 10.0,
            spatial_sigma: 3.0,
            color_sigma: 8.0,
            bilateral_sigma_pos: 5.0,
            n_labels: 2,
        }
    }
}

/// Dense fully-connected CRF model.
///
/// Build with [`DenseCrf::new`], supply unary potentials via [`DenseCrf::set_unary`],
/// supply the image with [`DenseCrf::set_image_2d`], and run [`DenseCrf::infer`].
#[derive(Debug, Clone)]
pub struct DenseCrf {
    config: CrfConfig,
    /// Unary potentials: `unary[pixel * n_labels + label]`
    unary: Vec<f64>,
    /// Flattened 2-D image pixels as RGB triplets for bilateral kernel
    image: Vec<[f64; 3]>,
    /// Image width (columns)
    width: usize,
    /// Image height (rows)
    height: usize,
}

impl DenseCrf {
    /// Create a new `DenseCrf` with the given configuration.
    pub fn new(config: CrfConfig) -> Self {
        Self {
            config,
            unary: Vec::new(),
            image: Vec::new(),
            width: 0,
            height: 0,
        }
    }

    /// Set unary potentials (builder pattern).
    ///
    /// `unary` is indexed as `[pixel][label]` (outer: pixel index, inner: label).
    /// Internally it is converted to row-major flat layout.
    pub fn set_unary(mut self, unary: Vec<Vec<f64>>) -> Self {
        let n_labels = self.config.n_labels;
        self.unary = unary
            .into_iter()
            .flat_map(|row| {
                let mut r = row;
                r.resize(n_labels, 0.0);
                r
            })
            .collect();
        self
    }

    /// Set the 2-D image used for the bilateral kernel (builder pattern).
    ///
    /// `image` is indexed `[row][col]` and each pixel is an RGB triplet `[f64; 3]`.
    pub fn set_image_2d(mut self, image: &[Vec<[f64; 3]>]) -> Self {
        self.height = image.len();
        self.width = if self.height > 0 { image[0].len() } else { 0 };
        self.image = image.iter().flat_map(|row| row.iter().cloned()).collect();
        self
    }

    /// Run mean-field inference and return the MAP label for each pixel.
    pub fn infer(&self) -> Vec<usize> {
        let n_pixels = self.unary.len() / self.config.n_labels.max(1);
        let n_labels = self.config.n_labels;

        if n_pixels == 0 || n_labels == 0 {
            return Vec::new();
        }

        // Initialise Q from unary potentials (softmax)
        let mut q = init_q(&self.unary, n_pixels, n_labels);

        for _ in 0..self.config.n_iter {
            // --- Message passing ---
            // Spatial kernel: Gaussian blur over position space only
            let mut msg_spatial = gaussian_filter_2d_per_label(
                &q,
                self.height,
                self.width,
                n_labels,
                self.config.spatial_sigma,
            );

            // Bilateral kernel: approximate by position-only Gaussian blur weighted
            // by colour similarity.  We compute a colour-weighted message.
            let msg_bilateral = bilateral_message_2d(
                &q,
                &self.image,
                self.height,
                self.width,
                n_labels,
                self.config.bilateral_sigma_pos,
                self.config.color_sigma,
            );

            // Weighted combination
            for i in 0..n_pixels * n_labels {
                msg_spatial[i] = self.config.spatial_weight * msg_spatial[i]
                    + self.config.bilateral_weight * msg_bilateral[i];
            }

            // Compatibility (Potts): subtract self term Q[i,l] (since μ(l,l)=0)
            // msg after Potts = Σ_{l'} μ(l,l') msg[i,l']
            //                 = Σ_{l' ≠ l} msg[i,l']
            //                 = Σ_{l'} msg[i,l'] - msg[i,l]
            let mut row_sums = vec![0.0f64; n_pixels];
            for i in 0..n_pixels {
                for l in 0..n_labels {
                    row_sums[i] += msg_spatial[i * n_labels + l];
                }
            }
            let mut compat = vec![0.0f64; n_pixels * n_labels];
            for i in 0..n_pixels {
                for l in 0..n_labels {
                    compat[i * n_labels + l] = row_sums[i] - msg_spatial[i * n_labels + l];
                }
            }

            // Update Q = softmax(-unary - compat)
            for i in 0..n_pixels {
                let base = i * n_labels;
                // Compute unnormalised log probabilities
                let mut log_p: Vec<f64> = (0..n_labels)
                    .map(|l| -(self.unary[base + l] + compat[base + l]))
                    .collect();
                // Numerically stable softmax
                let max_lp = log_p.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut z = 0.0f64;
                for lp in log_p.iter_mut() {
                    *lp = (*lp - max_lp).exp();
                    z += *lp;
                }
                if z < 1e-20 {
                    z = 1.0;
                }
                for l in 0..n_labels {
                    q[base + l] = log_p[l] / z;
                }
            }
        }

        // MAP estimate: argmax label per pixel
        (0..n_pixels)
            .map(|i| {
                let base = i * n_labels;
                let mut best_l = 0usize;
                let mut best_q = q[base];
                for l in 1..n_labels {
                    if q[base + l] > best_q {
                        best_q = q[base + l];
                        best_l = l;
                    }
                }
                best_l
            })
            .collect()
    }
}

/// Convenience function: apply dense CRF to a 2-D segmentation.
///
/// * `unary_log_prob` – unary (negative log probability) map, indexed
///   `[row][col][label]`
/// * `image` – RGB image `[row][col][3]`
/// * `config` – CRF configuration
///
/// Returns a 2-D label map `[row][col]`.
pub fn apply_to_segmentation_2d(
    unary_log_prob: &[Vec<Vec<f64>>],
    image: &[Vec<[f64; 3]>],
    config: &CrfConfig,
) -> Vec<Vec<usize>> {
    let rows = unary_log_prob.len();
    if rows == 0 {
        return Vec::new();
    }
    let cols = unary_log_prob[0].len();

    // Flatten unary: n_pixels × n_labels
    let unary_flat: Vec<Vec<f64>> = unary_log_prob
        .iter()
        .flat_map(|row| row.iter().cloned())
        .collect();

    let crf = DenseCrf::new(config.clone())
        .set_unary(unary_flat)
        .set_image_2d(image);

    let flat_labels = crf.infer();

    // Reshape into 2-D
    let mut result = vec![vec![0usize; cols]; rows];
    for (idx, &lbl) in flat_labels.iter().enumerate() {
        let r = idx / cols;
        let c = idx % cols;
        if r < rows && c < cols {
            result[r][c] = lbl;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Initialise Q by applying softmax to the negative unary potentials.
fn init_q(unary: &[f64], n_pixels: usize, n_labels: usize) -> Vec<f64> {
    let mut q = vec![0.0f64; n_pixels * n_labels];
    for i in 0..n_pixels {
        let base = i * n_labels;
        let mut vals: Vec<f64> = (0..n_labels).map(|l| -unary[base + l]).collect();
        let max_v = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut z = 0.0;
        for v in vals.iter_mut() {
            *v = (*v - max_v).exp();
            z += *v;
        }
        if z < 1e-20 {
            z = 1.0;
        }
        for l in 0..n_labels {
            q[base + l] = vals[l] / z;
        }
    }
    q
}

/// Apply a separable 2-D Gaussian blur to a per-label response map.
///
/// `q` is flat with layout `[pixel * n_labels + label]`.
/// Returns a new flat array of the same size.
fn gaussian_filter_2d_per_label(
    q: &[f64],
    height: usize,
    width: usize,
    n_labels: usize,
    sigma: f64,
) -> Vec<f64> {
    // Build 1-D Gaussian kernel
    let kernel = gaussian_kernel_1d(sigma);
    let k_rad = kernel.len() / 2;

    let mut out = vec![0.0f64; height * width * n_labels];

    // --- Horizontal pass ---
    let mut tmp = vec![0.0f64; height * width * n_labels];
    for r in 0..height {
        for c in 0..width {
            let pixel_out = r * width + c;
            for l in 0..n_labels {
                let mut acc = 0.0f64;
                let mut wt = 0.0f64;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let ci = c as isize + ki as isize - k_rad as isize;
                    if ci >= 0 && ci < width as isize {
                        let pixel_in = r * width + ci as usize;
                        acc += kv * q[pixel_in * n_labels + l];
                        wt += kv;
                    }
                }
                tmp[pixel_out * n_labels + l] = if wt > 0.0 { acc / wt } else { 0.0 };
            }
        }
    }

    // --- Vertical pass ---
    for r in 0..height {
        for c in 0..width {
            let pixel_out = r * width + c;
            for l in 0..n_labels {
                let mut acc = 0.0f64;
                let mut wt = 0.0f64;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let ri = r as isize + ki as isize - k_rad as isize;
                    if ri >= 0 && ri < height as isize {
                        let pixel_in = ri as usize * width + c;
                        acc += kv * tmp[pixel_in * n_labels + l];
                        wt += kv;
                    }
                }
                out[pixel_out * n_labels + l] = if wt > 0.0 { acc / wt } else { 0.0 };
            }
        }
    }

    out
}

/// Compute the bilateral message: for each pixel i and label l,
/// Σ_j k_b(i,j) * Q[j,l].
///
/// We approximate by doing a position-only Gaussian blur per pixel,
/// weighted by the colour affinity exp(−||I_i − I_j||² / (2σ_c²)).
/// For efficiency we iterate over local windows only (radius = 3σ_pos).
fn bilateral_message_2d(
    q: &[f64],
    image: &[[f64; 3]],
    height: usize,
    width: usize,
    n_labels: usize,
    sigma_pos: f64,
    sigma_col: f64,
) -> Vec<f64> {
    let mut out = vec![0.0f64; height * width * n_labels];

    if image.is_empty() {
        // No image available — fall back to spatial-only
        return gaussian_filter_2d_per_label(q, height, width, n_labels, sigma_pos);
    }

    // Window radius: 3 * sigma_pos rounded up, minimum 1
    let radius = ((3.0 * sigma_pos).ceil() as usize).max(1);
    let inv_2sig2_pos = 0.5 / (sigma_pos * sigma_pos);
    let inv_2sig2_col = 0.5 / (sigma_col * sigma_col);

    for r in 0..height {
        for c in 0..width {
            let pi = r * width + c;
            let ii = if pi < image.len() {
                image[pi]
            } else {
                [0.0; 3]
            };

            let r_lo = r.saturating_sub(radius);
            let r_hi = (r + radius + 1).min(height);
            let c_lo = c.saturating_sub(radius);
            let c_hi = (c + width).min(c + radius + 1).min(width);

            let mut acc = vec![0.0f64; n_labels];
            let mut total_w = 0.0f64;

            for rj in r_lo..r_hi {
                for cj in c_lo..c_hi {
                    let pj = rj * width + cj;
                    let ij = if pj < image.len() {
                        image[pj]
                    } else {
                        [0.0; 3]
                    };

                    let dr = (r as f64) - (rj as f64);
                    let dc = (c as f64) - (cj as f64);
                    let dist2_pos = dr * dr + dc * dc;

                    let dcol0 = ii[0] - ij[0];
                    let dcol1 = ii[1] - ij[1];
                    let dcol2 = ii[2] - ij[2];
                    let dist2_col = dcol0 * dcol0 + dcol1 * dcol1 + dcol2 * dcol2;

                    let w = (-inv_2sig2_pos * dist2_pos - inv_2sig2_col * dist2_col).exp();
                    total_w += w;
                    for l in 0..n_labels {
                        acc[l] += w * q[pj * n_labels + l];
                    }
                }
            }

            if total_w > 1e-20 {
                for l in 0..n_labels {
                    out[pi * n_labels + l] = acc[l] / total_w;
                }
            }
        }
    }

    out
}

/// Build a truncated 1-D Gaussian kernel with radius = ceil(3σ).
fn gaussian_kernel_1d(sigma: f64) -> Vec<f64> {
    let radius = ((3.0 * sigma).ceil() as usize).max(1);
    let mut k: Vec<f64> = (0..=2 * radius)
        .map(|i| {
            let x = i as f64 - radius as f64;
            (-0.5 * x * x / (sigma * sigma)).exp()
        })
        .collect();
    // Normalise
    let sum: f64 = k.iter().sum();
    if sum > 0.0 {
        for v in k.iter_mut() {
            *v /= sum;
        }
    }
    k
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny 2×2 image, 2 labels.
    fn make_unary_2labels(n_pixels: usize) -> Vec<Vec<f64>> {
        // Pixel 0,1 strongly favour label 0; pixels 2,3 strongly favour label 1
        (0..n_pixels)
            .map(|i| {
                if i < n_pixels / 2 {
                    vec![0.1, 5.0] // label 0 is much more probable
                } else {
                    vec![5.0, 0.1] // label 1 is much more probable
                }
            })
            .collect()
    }

    #[test]
    fn test_infer_labels_in_valid_range() {
        let config = CrfConfig {
            n_labels: 2,
            n_iter: 3,
            ..Default::default()
        };
        let n_pixels = 4;
        let unary = make_unary_2labels(n_pixels);
        let image: Vec<Vec<[f64; 3]>> = vec![
            vec![[0.0; 3], [0.0; 3]],
            vec![[255.0, 0.0, 0.0], [255.0, 0.0, 0.0]],
        ];
        let crf = DenseCrf::new(config).set_unary(unary).set_image_2d(&image);
        let labels = crf.infer();
        assert_eq!(labels.len(), n_pixels);
        for &l in &labels {
            assert!(l < 2, "label {} out of range [0,2)", l);
        }
    }

    #[test]
    fn test_infer_respects_strong_unary() {
        let config = CrfConfig {
            n_labels: 2,
            n_iter: 5,
            spatial_weight: 0.0,
            bilateral_weight: 0.0,
            ..Default::default()
        };
        // With zero pairwise weights the result is purely determined by unary
        let unary: Vec<Vec<f64>> = vec![
            vec![0.01, 100.0], // pixel 0 → label 0
            vec![0.01, 100.0], // pixel 1 → label 0
            vec![100.0, 0.01], // pixel 2 → label 1
            vec![100.0, 0.01], // pixel 3 → label 1
        ];
        let image: Vec<Vec<[f64; 3]>> = vec![vec![[0.0; 3], [0.0; 3]], vec![[0.0; 3], [0.0; 3]]];
        let crf = DenseCrf::new(config).set_unary(unary).set_image_2d(&image);
        let labels = crf.infer();
        assert_eq!(labels[0], 0);
        assert_eq!(labels[1], 0);
        assert_eq!(labels[2], 1);
        assert_eq!(labels[3], 1);
    }

    #[test]
    fn test_apply_to_segmentation_2d_shape() {
        let rows = 3usize;
        let cols = 4usize;
        let n_labels = 2usize;
        let unary_log_prob: Vec<Vec<Vec<f64>>> = (0..rows)
            .map(|_| {
                (0..cols)
                    .map(|c| {
                        if c < cols / 2 {
                            vec![0.1f64, 5.0]
                        } else {
                            vec![5.0, 0.1]
                        }
                    })
                    .collect()
            })
            .collect();
        let image: Vec<Vec<[f64; 3]>> = vec![vec![[128.0; 3]; cols]; rows];
        let config = CrfConfig {
            n_labels,
            n_iter: 2,
            ..Default::default()
        };
        let result = apply_to_segmentation_2d(&unary_log_prob, &image, &config);
        assert_eq!(result.len(), rows);
        assert_eq!(result[0].len(), cols);
        for row in &result {
            for &l in row {
                assert!(l < n_labels, "label {l} out of range");
            }
        }
    }

    #[test]
    fn test_empty_input() {
        let config = CrfConfig {
            n_labels: 2,
            ..Default::default()
        };
        let crf = DenseCrf::new(config);
        let labels = crf.infer();
        assert!(labels.is_empty());
    }
}
