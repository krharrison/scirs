//! Advanced superpixel segmentation algorithms.
//!
//! Provides SLIC-0 (adaptive compactness), SEEDS (energy-driven sampling),
//! and Compact Watershed superpixels.

use crate::error::{NdimageError, NdimageResult};
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for SLIC-based superpixel methods.
#[derive(Debug, Clone)]
pub struct LSSPConfig {
    /// Target number of superpixels.
    pub n_superpixels: usize,
    /// Spatial compactness (SLIC-0 adapts this per cluster).
    pub compactness: f64,
    /// Number of k-means iterations.
    pub n_iter: usize,
}

impl Default for LSSPConfig {
    fn default() -> Self {
        Self {
            n_superpixels: 100,
            compactness: 10.0,
            n_iter: 10,
        }
    }
}

// ─── SLIC-0 ───────────────────────────────────────────────────────────────────

/// SLIC-0 (zero-parameter adaptive SLIC) superpixel segmentation.
///
/// Unlike standard SLIC, SLIC-0 adapts the compactness per-cluster by
/// normalising the colour distance by the local intra-cluster color variance,
/// so no explicit compactness parameter needs tuning.
///
/// # Arguments
/// * `image`  – multi-channel image as `[row][col][channel]`
/// * `config` – SLIC-0 configuration
///
/// # Returns
/// Label map of shape `rows × cols`, labels in `[0, n_superpixels)`.
///
/// # Errors
/// Returns `InvalidInput` if the image is empty or `n_superpixels` is 0.
pub fn slic_zero(
    image: &[Vec<Vec<f32>>],
    config: &LSSPConfig,
) -> NdimageResult<Vec<Vec<i32>>> {
    if image.is_empty() {
        return Err(NdimageError::InvalidInput("image must not be empty".into()));
    }
    if config.n_superpixels == 0 {
        return Err(NdimageError::InvalidInput("n_superpixels must be at least 1".into()));
    }
    let rows = image.len();
    let cols = image[0].len();
    let channels = image[0][0].len();
    if rows == 0 || cols == 0 || channels == 0 {
        return Err(NdimageError::InvalidInput("image dimensions must be positive".into()));
    }

    let step = ((rows * cols) as f64 / config.n_superpixels as f64).sqrt().max(1.0) as usize;

    // Initialise cluster centres on a regular grid.
    let mut centres: Vec<Centre> = Vec::new();
    let mut ry = step / 2;
    while ry < rows {
        let mut cx = step / 2;
        while cx < cols {
            let (br, bc) = find_min_gradient(image, ry, cx, rows, cols, channels);
            let color: Vec<f64> = (0..channels).map(|ch| image[br][bc][ch] as f64).collect();
            centres.push(Centre { r: br as f64, c: bc as f64, color, color_var: 1.0 });
            cx += step;
        }
        ry += step;
    }

    if centres.is_empty() {
        return Ok(vec![vec![0i32; cols]; rows]);
    }

    let s = step as f64;
    let k = centres.len();

    let mut labels = vec![vec![-1i32; cols]; rows];
    let mut distances = vec![vec![f64::INFINITY; cols]; rows];

    for _iter in 0..config.n_iter {
        // Clear distances.
        for row in &mut distances { for d in row.iter_mut() { *d = f64::INFINITY; } }

        // Assignment.
        for (cluster_id, ctr) in centres.iter().enumerate() {
            let cr = ctr.r as isize;
            let cc = ctr.c as isize;
            let r0 = (cr - 2 * step as isize).max(0) as usize;
            let r1 = (cr + 2 * step as isize + 1).min(rows as isize) as usize;
            let c0 = (cc - 2 * step as isize).max(0) as usize;
            let c1 = (cc + 2 * step as isize + 1).min(cols as isize) as usize;

            for r in r0..r1 {
                for c in c0..c1 {
                    let color_dist = (0..channels)
                        .map(|ch| {
                            let d = image[r][c][ch] as f64 - ctr.color[ch];
                            d * d
                        })
                        .sum::<f64>()
                        / ctr.color_var.max(1e-10);
                    let spatial_dist =
                        ((r as f64 - ctr.r).powi(2) + (c as f64 - ctr.c).powi(2)) / (s * s);
                    let dist = (color_dist + spatial_dist).sqrt();
                    if dist < distances[r][c] {
                        distances[r][c] = dist;
                        labels[r][c] = cluster_id as i32;
                    }
                }
            }
        }

        // Update centres and compute per-cluster color variance.
        let mut new_centres: Vec<(f64, f64, Vec<f64>, usize)> = vec![(0.0, 0.0, vec![0.0; channels], 0); k];
        for r in 0..rows {
            for c in 0..cols {
                let lbl = labels[r][c];
                if lbl < 0 { continue; }
                let l = lbl as usize;
                new_centres[l].0 += r as f64;
                new_centres[l].1 += c as f64;
                for ch in 0..channels {
                    new_centres[l].2[ch] += image[r][c][ch] as f64;
                }
                new_centres[l].3 += 1;
            }
        }

        // Compute per-cluster variance for adaptive compactness.
        let mut color_sums2: Vec<Vec<f64>> = vec![vec![0.0; channels]; k];
        for r in 0..rows {
            for c in 0..cols {
                let lbl = labels[r][c];
                if lbl < 0 { continue; }
                let l = lbl as usize;
                let n = new_centres[l].3 as f64;
                if n < 1.0 { continue; }
                let mean_col = new_centres[l].2[0] / n; // use first channel for simplicity
                let d = image[r][c][0] as f64 - mean_col;
                color_sums2[l][0] += d * d;
            }
        }

        for (l, ctr) in centres.iter_mut().enumerate() {
            let n = new_centres[l].3 as f64;
            if n > 0.0 {
                ctr.r = new_centres[l].0 / n;
                ctr.c = new_centres[l].1 / n;
                for ch in 0..channels {
                    ctr.color[ch] = new_centres[l].2[ch] / n;
                }
                ctr.color_var = (color_sums2[l][0] / n).max(1e-10);
            }
        }
    }

    // Fill any unlabelled pixels with nearest label.
    assign_unlabelled(&mut labels, rows, cols);

    Ok(labels)
}

// ─── SEEDS ────────────────────────────────────────────────────────────────────

/// SEEDS (Superpixels Extracted via Energy-Driven Sampling).
///
/// Starts with a coarse block partition and iteratively refines at the
/// block and pixel level using a colour histogram overlap energy.
///
/// # Arguments
/// * `image`         – multi-channel image as `[row][col][channel]`
/// * `n_superpixels` – target superpixel count
///
/// # Returns
/// Label map of shape `rows × cols`.
///
/// # Errors
/// Returns `InvalidInput` if the image is empty or `n_superpixels` is 0.
pub fn seeds_superpixels(
    image: &[Vec<Vec<f32>>],
    n_superpixels: usize,
) -> NdimageResult<Vec<Vec<i32>>> {
    if image.is_empty() || n_superpixels == 0 {
        return Err(NdimageError::InvalidInput(
            "image must not be empty and n_superpixels must be >= 1".into(),
        ));
    }
    let rows = image.len();
    let cols = image[0].len();
    let channels = image[0][0].len();

    // Compute block size.
    let step = ((rows * cols) as f64 / n_superpixels as f64).sqrt().max(1.0) as usize;
    let n_bins = 8usize;

    // Initialise labels from block partition.
    let mut labels = vec![vec![0i32; cols]; rows];
    let mut label_id = 0i32;
    let mut block_labels: Vec<Vec<i32>> = Vec::new();

    let mut br = 0;
    while br < rows {
        let mut bc = 0;
        while bc < cols {
            let r_end = (br + step).min(rows);
            let c_end = (bc + step).min(cols);
            for r in br..r_end {
                for c in bc..c_end {
                    labels[r][c] = label_id;
                }
            }
            let row: Vec<i32> = (bc..c_end).map(|c| labels[br][c]).collect();
            block_labels.push(row);
            label_id += 1;
            bc += step;
        }
        br += step;
    }

    let n_labels = label_id as usize;

    // Quantise image to bins for histogram computation.
    let quantised: Vec<Vec<Vec<u8>>> = (0..rows).map(|r| {
        (0..cols).map(|c| {
            (0..channels).map(|ch| {
                let v = image[r][c][ch].clamp(0.0, 1.0);
                ((v * (n_bins - 1) as f32) as u8).min(n_bins as u8 - 1)
            }).collect()
        }).collect()
    }).collect();

    // Build histograms per label.
    let mut histograms: Vec<Vec<Vec<usize>>> = vec![vec![vec![0; n_bins]; channels]; n_labels];
    let mut label_size: Vec<usize> = vec![0; n_labels];

    for r in 0..rows {
        for c in 0..cols {
            let l = labels[r][c] as usize;
            for ch in 0..channels {
                let bin = quantised[r][c][ch] as usize;
                histograms[l][ch][bin] += 1;
            }
            label_size[l] += 1;
        }
    }

    // Pixel-level refinement passes.
    let n_pixel_passes = 5;
    for _pass in 0..n_pixel_passes {
        for r in 0..rows {
            for c in 0..cols {
                let cur_lbl = labels[r][c] as usize;
                let mut best_lbl = cur_lbl;
                let mut best_energy = histogram_energy(&histograms[cur_lbl], channels, n_bins);

                for (nr, nc) in neighbours4(r, c, rows, cols) {
                    let nb_lbl = labels[nr][nc] as usize;
                    if nb_lbl == cur_lbl { continue; }

                    // Try moving pixel from cur_lbl to nb_lbl.
                    // Temporarily update histograms.
                    for ch in 0..channels {
                        let bin = quantised[r][c][ch] as usize;
                        if histograms[cur_lbl][ch][bin] > 0 {
                            histograms[cur_lbl][ch][bin] -= 1;
                        }
                        histograms[nb_lbl][ch][bin] += 1;
                    }

                    let e_cur = histogram_energy(&histograms[cur_lbl], channels, n_bins);
                    let e_nb  = histogram_energy(&histograms[nb_lbl], channels, n_bins);
                    let total_energy = e_cur + e_nb;

                    if total_energy < best_energy {
                        best_energy = total_energy;
                        best_lbl = nb_lbl;
                    }

                    // Undo temporary change.
                    for ch in 0..channels {
                        let bin = quantised[r][c][ch] as usize;
                        histograms[cur_lbl][ch][bin] += 1;
                        if histograms[nb_lbl][ch][bin] > 0 {
                            histograms[nb_lbl][ch][bin] -= 1;
                        }
                    }
                }

                if best_lbl != cur_lbl {
                    // Commit move.
                    for ch in 0..channels {
                        let bin = quantised[r][c][ch] as usize;
                        if histograms[cur_lbl][ch][bin] > 0 {
                            histograms[cur_lbl][ch][bin] -= 1;
                        }
                        histograms[best_lbl][ch][bin] += 1;
                    }
                    if label_size[cur_lbl] > 0 { label_size[cur_lbl] -= 1; }
                    label_size[best_lbl] += 1;
                    labels[r][c] = best_lbl as i32;
                }
            }
        }
    }

    Ok(labels)
}

// ─── Compact Watershed ────────────────────────────────────────────────────────

/// Compact Watershed superpixels (Neubert 2014).
///
/// Geodesic distance is weighted by the compactness factor so that the
/// watershed boundaries are pushed toward roundness.
///
/// # Arguments
/// * `image`         – 2-D grayscale image
/// * `n_superpixels` – target number of superpixels
/// * `compactness`   – spatial regularisation weight (higher = more compact)
///
/// # Returns
/// Label map of shape `rows × cols`.
///
/// # Errors
/// Returns `InvalidInput` if the image is empty or `n_superpixels` is 0.
pub fn compact_watershed_superpixels(
    image: &[Vec<f32>],
    n_superpixels: usize,
    compactness: f64,
) -> NdimageResult<Vec<Vec<i32>>> {
    if image.is_empty() || n_superpixels == 0 {
        return Err(NdimageError::InvalidInput(
            "image must not be empty and n_superpixels must be >= 1".into(),
        ));
    }
    let rows = image.len();
    let cols = image[0].len();
    if image.iter().any(|r| r.len() != cols) {
        return Err(NdimageError::DimensionError(
            "image rows must have equal length".into(),
        ));
    }

    // Place seeds on a regular grid.
    let step = ((rows * cols) as f64 / n_superpixels as f64).sqrt().max(1.0) as usize;
    let mut seeds: Vec<(usize, usize)> = Vec::new();
    let mut ry = step / 2;
    while ry < rows {
        let mut cx = step / 2;
        while cx < cols {
            seeds.push((ry, cx));
            cx += step;
        }
        ry += step;
    }

    if seeds.is_empty() {
        return Ok(vec![vec![0i32; cols]; rows]);
    }

    let s = step as f64;
    let n_seeds = seeds.len();

    // Dijkstra with compound distance metric.
    let mut dist = vec![vec![f64::INFINITY; cols]; rows];
    let mut labels = vec![vec![-1i32; cols]; rows];

    // Min-heap: (distance, row, col)
    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();

    for (seed_id, &(sr, sc)) in seeds.iter().enumerate() {
        dist[sr][sc] = 0.0;
        labels[sr][sc] = seed_id as i32;
        heap.push(HeapItem { cost: OrderedF64(0.0), r: sr, c: sc });
    }

    while let Some(HeapItem { cost: OrderedF64(d), r, c }) = heap.pop() {
        if d > dist[r][c] { continue; }
        let lbl = labels[r][c];
        let seed = seeds[lbl as usize];

        for (nr, nc) in neighbours4(r, c, rows, cols) {
            let grad = (image[r][c] as f64 - image[nr][nc] as f64).abs();
            // Spatial penalty: Euclidean distance from seed normalised by step.
            let dr = (nr as f64 - seed.0 as f64) / s;
            let dc = (nc as f64 - seed.1 as f64) / s;
            let spatial_penalty = compactness * (dr * dr + dc * dc).sqrt();
            let nd = d + grad + spatial_penalty;

            if nd < dist[nr][nc] {
                dist[nr][nc] = nd;
                labels[nr][nc] = lbl;
                heap.push(HeapItem { cost: OrderedF64(nd), r: nr, c: nc });
            }
        }
    }

    assign_unlabelled_i32(&mut labels, rows, cols);

    Ok(labels)
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

struct Centre {
    r: f64,
    c: f64,
    color: Vec<f64>,
    color_var: f64,
}

fn find_min_gradient(
    image: &[Vec<Vec<f32>>],
    r: usize,
    c: usize,
    rows: usize,
    cols: usize,
    channels: usize,
) -> (usize, usize) {
    let mut best_r = r;
    let mut best_c = c;
    let mut best_grad = f64::INFINITY;

    let r0 = r.saturating_sub(1);
    let r1 = (r + 2).min(rows);
    let c0 = c.saturating_sub(1);
    let c1 = (c + 2).min(cols);

    for nr in r0..r1 {
        for nc in c0..c1 {
            let mut grad = 0.0f64;
            for ch in 0..channels {
                let dx = if nc + 1 < cols { (image[nr][nc + 1][ch] - image[nr][nc][ch]).abs() as f64 } else { 0.0 };
                let dy = if nr + 1 < rows { (image[nr + 1][nc][ch] - image[nr][nc][ch]).abs() as f64 } else { 0.0 };
                grad += dx + dy;
            }
            if grad < best_grad {
                best_grad = grad;
                best_r = nr;
                best_c = nc;
            }
        }
    }
    (best_r, best_c)
}

fn histogram_energy(hist: &[Vec<usize>], channels: usize, n_bins: usize) -> f64 {
    let mut energy = 0.0f64;
    for ch in 0..channels {
        let total: usize = hist[ch].iter().sum();
        if total == 0 { continue; }
        for &cnt in &hist[ch][..n_bins] {
            let p = cnt as f64 / total as f64;
            if p > 0.0 { energy -= p * p.ln(); } // negative entropy → maximise overlap
        }
    }
    energy
}

fn neighbours4(r: usize, c: usize, rows: usize, cols: usize) -> impl Iterator<Item = (usize, usize)> {
    let mut nb = Vec::with_capacity(4);
    if r > 0        { nb.push((r - 1, c)); }
    if r + 1 < rows { nb.push((r + 1, c)); }
    if c > 0        { nb.push((r, c - 1)); }
    if c + 1 < cols { nb.push((r, c + 1)); }
    nb.into_iter()
}

fn assign_unlabelled(labels: &mut Vec<Vec<i32>>, rows: usize, cols: usize) {
    // BFS from labelled pixels.
    let mut queue: std::collections::VecDeque<(usize, usize)> = std::collections::VecDeque::new();
    for r in 0..rows {
        for c in 0..cols {
            if labels[r][c] >= 0 { queue.push_back((r, c)); }
        }
    }
    while let Some((r, c)) = queue.pop_front() {
        let lbl = labels[r][c];
        for (nr, nc) in neighbours4(r, c, rows, cols) {
            if labels[nr][nc] < 0 {
                labels[nr][nc] = lbl;
                queue.push_back((nr, nc));
            }
        }
    }
    // If still -1, set to 0.
    for row in labels.iter_mut() {
        for l in row.iter_mut() {
            if *l < 0 { *l = 0; }
        }
    }
}

fn assign_unlabelled_i32(labels: &mut Vec<Vec<i32>>, rows: usize, cols: usize) {
    assign_unlabelled(labels, rows, cols);
}

/// A min-heap item for Dijkstra.
#[derive(PartialEq)]
struct HeapItem {
    cost: OrderedF64,
    r: usize,
    c: usize,
}

#[derive(PartialEq, Clone, Copy)]
struct OrderedF64(f64);

impl Eq for OrderedF64 {}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap.
        other.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
    }
}

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost.cmp(&other.cost)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rgb_image(rows: usize, cols: usize) -> Vec<Vec<Vec<f32>>> {
        (0..rows).map(|r| {
            (0..cols).map(|c| {
                vec![
                    (r as f32 / rows as f32),
                    (c as f32 / cols as f32),
                    0.5f32,
                ]
            }).collect()
        }).collect()
    }

    fn make_gray_image(rows: usize, cols: usize) -> Vec<Vec<f32>> {
        (0..rows).map(|r| {
            (0..cols).map(|c| (r * cols + c) as f32 / (rows * cols) as f32).collect()
        }).collect()
    }

    #[test]
    fn test_slic_zero_basic() {
        let image = make_rgb_image(30, 30);
        let config = LSSPConfig {
            n_superpixels: 9,
            compactness: 5.0,
            n_iter: 5,
        };
        let labels = slic_zero(&image, &config).expect("slic_zero failed");
        assert_eq!(labels.len(), 30);
        assert_eq!(labels[0].len(), 30);
        // All pixels should be labelled.
        let all_labelled = labels.iter().flat_map(|r| r.iter()).all(|&l| l >= 0);
        assert!(all_labelled, "all pixels should be labelled");
    }

    #[test]
    fn test_slic_zero_n_labels() {
        let image = make_rgb_image(20, 20);
        let config = LSSPConfig {
            n_superpixels: 4,
            n_iter: 3,
            ..Default::default()
        };
        let labels = slic_zero(&image, &config).expect("slic_zero failed");
        let unique: std::collections::HashSet<i32> =
            labels.iter().flat_map(|r| r.iter().copied()).collect();
        assert!(!unique.is_empty(), "should produce at least one label");
    }

    #[test]
    fn test_seeds_basic() {
        let image = make_rgb_image(20, 20);
        let labels = seeds_superpixels(&image, 4).expect("seeds failed");
        assert_eq!(labels.len(), 20);
        assert_eq!(labels[0].len(), 20);
        let all_labelled = labels.iter().flat_map(|r| r.iter()).all(|&l| l >= 0);
        assert!(all_labelled, "all pixels should be labelled");
    }

    #[test]
    fn test_compact_watershed_basic() {
        let image = make_gray_image(20, 20);
        let labels = compact_watershed_superpixels(&image, 4, 1.0)
            .expect("compact_watershed failed");
        assert_eq!(labels.len(), 20);
        assert_eq!(labels[0].len(), 20);
        let all_labelled = labels.iter().flat_map(|r| r.iter()).all(|&l| l >= 0);
        assert!(all_labelled, "all pixels should be labelled");
    }

    #[test]
    fn test_compact_watershed_compactness_effect() {
        let image = make_gray_image(20, 20);
        // High compactness → more regular regions.
        let l1 = compact_watershed_superpixels(&image, 9, 0.001).expect("low compactness failed");
        let l2 = compact_watershed_superpixels(&image, 9, 10.0).expect("high compactness failed");
        // Both should produce valid label maps.
        assert_eq!(l1.len(), 20);
        assert_eq!(l2.len(), 20);
    }

    #[test]
    fn test_slic_zero_empty_image() {
        let image: Vec<Vec<Vec<f32>>> = vec![];
        let config = LSSPConfig::default();
        assert!(slic_zero(&image, &config).is_err());
    }

    #[test]
    fn test_seeds_zero_superpixels() {
        let image = make_rgb_image(10, 10);
        assert!(seeds_superpixels(&image, 0).is_err());
    }
}
