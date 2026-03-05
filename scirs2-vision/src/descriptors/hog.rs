//! Histogram of Oriented Gradients (HOG) descriptor.
//!
//! Computes HOG features as described in Dalal & Triggs (2005).
//! Each cell accumulates a weighted histogram of gradient orientations;
//! overlapping blocks of cells are concatenated and L2-normalized to
//! form the final feature vector.

use crate::error::VisionError;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Block normalization scheme for HOG descriptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HOGNorm {
    /// L2 normalisation: v / sqrt(||v||² + ε²)
    L2,
    /// L2-Hys: L2 normalisation followed by clipping to 0.2 then re-normalisation
    L2Hys,
    /// L1 normalisation: v / (||v||₁ + ε)
    L1,
    /// L1-sqrt: sqrt(v / (||v||₁ + ε))
    L1Sqrt,
}

/// Configuration for HOG descriptor computation.
#[derive(Debug, Clone)]
pub struct HOGConfig {
    /// Width and height of each cell in pixels.
    pub cell_size: usize,
    /// Number of cells per block edge (block is block_size × block_size cells).
    pub block_size: usize,
    /// Number of orientation bins per cell.
    pub n_bins: usize,
    /// If true, orientations are unsigned (0–180°); otherwise signed (0–360°).
    pub unsigned: bool,
    /// Block normalisation method.
    pub normalize: HOGNorm,
}

impl Default for HOGConfig {
    fn default() -> Self {
        Self {
            cell_size: 8,
            block_size: 2,
            n_bins: 9,
            unsigned: true,
            normalize: HOGNorm::L2Hys,
        }
    }
}

// ─── Descriptor type ─────────────────────────────────────────────────────────

/// HOG descriptor for an image.
///
/// `cells[row][col]` is the n_bins-length histogram for that cell.
#[derive(Debug, Clone)]
pub struct HOGDescriptor {
    /// Per-cell orientation histograms: `cells[row][col][bin]`.
    pub cells: Vec<Vec<Vec<f32>>>,
    /// Configuration used during computation.
    pub config: HOGConfig,
    /// Number of cell rows.
    pub n_cell_rows: usize,
    /// Number of cell cols.
    pub n_cell_cols: usize,
}

// ─── Gradient helpers ────────────────────────────────────────────────────────

/// Compute Sobel gradients (Gx, Gy) at pixel (r, c).
#[inline]
fn sobel_at(image: &[Vec<f32>], r: usize, c: usize, rows: usize, cols: usize) -> (f32, f32) {
    let get = |dr: i32, dc: i32| -> f32 {
        let rr = (r as i32 + dr).clamp(0, rows as i32 - 1) as usize;
        let cc = (c as i32 + dc).clamp(0, cols as i32 - 1) as usize;
        image[rr][cc]
    };
    let gx =
        -get(-1, -1) + get(-1, 1) - 2.0 * get(0, -1) + 2.0 * get(0, 1) - get(1, -1) + get(1, 1);
    let gy =
        -get(-1, -1) - 2.0 * get(-1, 0) - get(-1, 1) + get(1, -1) + 2.0 * get(1, 0) + get(1, 1);
    (gx, gy)
}

// ─── Block normalisation ─────────────────────────────────────────────────────

fn normalize_block(v: &mut [f32], scheme: HOGNorm) {
    const EPS: f32 = 1e-5;
    match scheme {
        HOGNorm::L2 => {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt() + EPS;
            v.iter_mut().for_each(|x| *x /= norm);
        }
        HOGNorm::L2Hys => {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt() + EPS;
            v.iter_mut().for_each(|x| {
                *x = (*x / norm).min(0.2);
            });
            let norm2 = v.iter().map(|x| x * x).sum::<f32>().sqrt() + EPS;
            v.iter_mut().for_each(|x| *x /= norm2);
        }
        HOGNorm::L1 => {
            let norm = v.iter().map(|x| x.abs()).sum::<f32>() + EPS;
            v.iter_mut().for_each(|x| *x /= norm);
        }
        HOGNorm::L1Sqrt => {
            let norm = v.iter().map(|x| x.abs()).sum::<f32>() + EPS;
            v.iter_mut().for_each(|x| *x = (*x / norm).sqrt());
        }
    }
}

// ─── Main API ────────────────────────────────────────────────────────────────

/// Compute HOG descriptor for `image` (row-major, float).
///
/// Returns a [`HOGDescriptor`] containing per-cell histograms.
///
/// # Errors
/// Returns [`VisionError`] if the image is empty or smaller than one cell.
pub fn compute_hog(image: &[Vec<f32>], config: &HOGConfig) -> Result<HOGDescriptor, VisionError> {
    let rows = image.len();
    let cols = image.first().map_or(0, |r| r.len());
    if rows < config.cell_size || cols < config.cell_size {
        return Err(VisionError::InvalidInput(
            "Image too small for HOG cell size".into(),
        ));
    }

    let n_cell_rows = rows / config.cell_size;
    let n_cell_cols = cols / config.cell_size;
    let n_bins = config.n_bins;
    let angle_range = if config.unsigned {
        180.0_f32
    } else {
        360.0_f32
    };

    // Pre-compute per-pixel gradients.
    let mut grad_mag = vec![vec![0.0_f32; cols]; rows];
    let mut grad_ang = vec![vec![0.0_f32; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let (gx, gy) = sobel_at(image, r, c, rows, cols);
            grad_mag[r][c] = (gx * gx + gy * gy).sqrt();
            let ang = gy.atan2(gx).to_degrees();
            // Map to [0, angle_range)
            let ang = if config.unsigned {
                let a = ang % 180.0;
                if a < 0.0 {
                    a + 180.0
                } else {
                    a
                }
            } else {
                let a = ang % 360.0;
                if a < 0.0 {
                    a + 360.0
                } else {
                    a
                }
            };
            grad_ang[r][c] = ang;
        }
    }

    // Accumulate per-cell histograms with soft bin voting.
    let mut cells: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; n_bins]; n_cell_cols]; n_cell_rows];
    #[allow(clippy::needless_range_loop)]
    for cr in 0..n_cell_rows {
        for cc in 0..n_cell_cols {
            let r0 = cr * config.cell_size;
            let c0 = cc * config.cell_size;
            let hist = &mut cells[cr][cc];
            for r in r0..r0 + config.cell_size {
                for c in c0..c0 + config.cell_size {
                    let mag = grad_mag[r][c];
                    let ang = grad_ang[r][c];
                    // Interpolate between two adjacent bins.
                    let bin_f = ang / angle_range * n_bins as f32;
                    let bin_lo = bin_f.floor() as usize % n_bins;
                    let bin_hi = (bin_lo + 1) % n_bins;
                    let frac = bin_f - bin_f.floor();
                    hist[bin_lo] += mag * (1.0 - frac);
                    hist[bin_hi] += mag * frac;
                }
            }
        }
    }

    Ok(HOGDescriptor {
        cells,
        config: config.clone(),
        n_cell_rows,
        n_cell_cols,
    })
}

/// Compute the HOG feature vector for `image`.
///
/// Iterates over overlapping blocks, concatenates and normalises each block's
/// cell histograms, and returns the flattened descriptor.
///
/// # Errors
/// Propagates errors from [`compute_hog`].
pub fn hog_feature_vector(image: &[Vec<f32>], config: &HOGConfig) -> Result<Vec<f32>, VisionError> {
    let desc = compute_hog(image, config)?;
    let block = config.block_size;
    let n_bins = config.n_bins;
    let block_dim = block * block * n_bins;

    // Number of blocks in each direction (stride = 1 cell).
    let n_block_rows = desc.n_cell_rows.saturating_sub(block - 1);
    let n_block_cols = desc.n_cell_cols.saturating_sub(block - 1);

    let mut feature = Vec::with_capacity(n_block_rows * n_block_cols * block_dim);

    for br in 0..n_block_rows {
        for bc in 0..n_block_cols {
            let mut block_vec: Vec<f32> = Vec::with_capacity(block_dim);
            for dr in 0..block {
                for dc in 0..block {
                    block_vec.extend_from_slice(&desc.cells[br + dr][bc + dc]);
                }
            }
            normalize_block(&mut block_vec, config.normalize);
            feature.extend_from_slice(&block_vec);
        }
    }

    Ok(feature)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_image(rows: usize, cols: usize) -> Vec<Vec<f32>> {
        (0..rows)
            .map(|r| (0..cols).map(|c| ((r + c) % 256) as f32 / 255.0).collect())
            .collect()
    }

    #[test]
    fn test_hog_cell_count() {
        let img = synthetic_image(64, 128);
        let cfg = HOGConfig {
            cell_size: 8,
            block_size: 2,
            n_bins: 9,
            ..Default::default()
        };
        let desc = compute_hog(&img, &cfg).expect("compute_hog should succeed on valid image");
        assert_eq!(desc.n_cell_rows, 8);
        assert_eq!(desc.n_cell_cols, 16);
        assert_eq!(desc.cells[0][0].len(), 9);
    }

    #[test]
    fn test_hog_feature_vector_length() {
        let img = synthetic_image(64, 128);
        let cfg = HOGConfig::default();
        let fv = hog_feature_vector(&img, &cfg)
            .expect("hog_feature_vector should succeed on valid image");
        // n_block_rows = 8-1=7, n_block_cols=16-1=15; each block = 2*2*9=36
        assert_eq!(fv.len(), 7 * 15 * 36);
    }

    #[test]
    fn test_hog_too_small() {
        let img = synthetic_image(4, 4);
        let cfg = HOGConfig::default(); // cell_size=8
        assert!(compute_hog(&img, &cfg).is_err());
    }

    #[test]
    fn test_hog_l1_normalisation() {
        let img = synthetic_image(32, 32);
        let cfg = HOGConfig {
            cell_size: 8,
            block_size: 2,
            n_bins: 9,
            normalize: HOGNorm::L1,
            unsigned: true,
        };
        let fv = hog_feature_vector(&img, &cfg)
            .expect("hog_feature_vector should succeed on valid image");
        // All values should be non-negative.
        assert!(fv.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_hog_l1sqrt_normalisation() {
        let img = synthetic_image(32, 32);
        let cfg = HOGConfig {
            normalize: HOGNorm::L1Sqrt,
            ..Default::default()
        };
        let fv = hog_feature_vector(&img, &cfg)
            .expect("hog_feature_vector should succeed on valid image");
        assert!(fv.iter().all(|&v| v >= 0.0));
    }
}
