//! High-level depth completion API: sparse-to-dense depth map filling.
//!
//! Provides `DepthCompleter`, a unified entry-point that dispatches to:
//!
//! * **NearestNeighbor** – BFS from sparse points to fill every empty pixel.
//! * **InvDistWeighted** – weighted average of K nearest valid depth points
//!   (`weight = 1/d²`).
//! * **PropagationFill** – iterative 8-connected neighbourhood averaging until
//!   convergence or `max_iterations`.
//! * **SurfaceNormals** – Sobel-based normal estimation from an RGB guide,
//!   followed by normal-integration anchored at the sparse depth points.
//!
//! All public items use `f32` for consistency with the rest of the vision crate.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Public types specific to this interface
// ---------------------------------------------------------------------------

/// Depth completion method used by `DepthCompleter`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthMethod {
    /// BFS nearest-neighbour fill.
    NearestNeighbor,
    /// Inverse-distance-weighted interpolation (K = 8 nearest valid points).
    InvDistWeighted,
    /// Normal integration from RGB luminance gradients (needs `rgb` argument).
    SurfaceNormals,
    /// Iterative 8-connected propagation until convergence.
    PropagationFill,
}

/// Configuration for `DepthCompleter`.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct DepthCompletionConfig {
    /// Algorithm to use.
    pub method: DepthMethod,
    /// Maximum valid depth value; deeper readings are clamped/ignored.
    pub max_depth: f32,
    /// Minimum valid depth value; shallower readings are clamped/ignored.
    pub min_depth: f32,
    /// Maximum iterations for iterative methods (`PropagationFill`).
    pub iterations: usize,
}

impl Default for DepthCompletionConfig {
    fn default() -> Self {
        Self {
            method: DepthMethod::PropagationFill,
            max_depth: 100.0,
            min_depth: 0.1,
            iterations: 5,
        }
    }
}

/// Result of a depth completion operation.
pub struct DepthResult {
    /// Dense depth map; `dense_depth[row][col]` gives the estimated depth at
    /// that pixel. Pixels that could not be filled retain the value 0.
    pub dense_depth: Vec<Vec<f32>>,
    /// Per-pixel confidence in `[0, 1]`.  Sparse points receive confidence 1.0;
    /// interpolated pixels receive a value derived from the chosen method.
    pub confidence: Vec<Vec<f32>>,
    /// Number of pixels that were filled (were `None` in the input but have a
    /// non-zero value in the output).
    pub filled_pixels: usize,
}

// ---------------------------------------------------------------------------
// DepthCompleter
// ---------------------------------------------------------------------------

/// Depth completer: fills a sparse `Option<f32>` depth map into a dense one.
pub struct DepthCompleter {
    config: DepthCompletionConfig,
}

impl DepthCompleter {
    /// Create a new depth completer with the given configuration.
    pub fn new(config: DepthCompletionConfig) -> Self {
        Self { config }
    }

    /// Complete a sparse depth map.
    ///
    /// # Arguments
    ///
    /// * `sparse_depth` – `sparse_depth[row][col]` is `None` where depth is
    ///   unknown, `Some(d)` where a valid measurement exists.  All rows must
    ///   have the same length.
    /// * `rgb` – optional RGB guide image required for `SurfaceNormals`.
    ///   Each pixel is `[R, G, B]` with values in `[0, 255]`.
    ///
    /// # Returns
    ///
    /// A `DepthResult` whose `dense_depth` has the same shape as `sparse_depth`.
    pub fn complete(
        &self,
        sparse_depth: &[Vec<Option<f32>>],
        rgb: Option<&[Vec<[u8; 3]>]>,
    ) -> DepthResult {
        if sparse_depth.is_empty() {
            return DepthResult {
                dense_depth: Vec::new(),
                confidence: Vec::new(),
                filled_pixels: 0,
            };
        }

        let height = sparse_depth.len();
        let width = sparse_depth[0].len();

        // Initialise dense grid and confidence from sparse data.
        let mut dense = vec![vec![0.0f32; width]; height];
        let mut confidence = vec![vec![0.0f32; width]; height];

        let min_d = self.config.min_depth;
        let max_d = self.config.max_depth;

        for r in 0..height {
            for c in 0..width {
                if let Some(d) = sparse_depth[r][c] {
                    if d >= min_d && d <= max_d {
                        dense[r][c] = d;
                        confidence[r][c] = 1.0;
                    }
                }
            }
        }

        let initial_filled = dense
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&v| v > 0.0)
            .count();

        match self.config.method {
            DepthMethod::NearestNeighbor => {
                fill_nearest_neighbor(&mut dense, &mut confidence, height, width);
            }
            DepthMethod::InvDistWeighted => {
                fill_inv_dist_weighted(&mut dense, &mut confidence, height, width);
            }
            DepthMethod::PropagationFill => {
                fill_propagation(
                    &mut dense,
                    &mut confidence,
                    height,
                    width,
                    self.config.iterations,
                );
            }
            DepthMethod::SurfaceNormals => {
                fill_surface_normals(&mut dense, &mut confidence, height, width, rgb);
            }
        }

        // Count newly filled pixels (were 0.0 before, non-zero after).
        let total_filled = dense
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&v| v > 0.0)
            .count();
        let filled_pixels = total_filled.saturating_sub(initial_filled);

        DepthResult {
            dense_depth: dense,
            confidence,
            filled_pixels,
        }
    }
}

// ---------------------------------------------------------------------------
// NearestNeighbor – BFS from all valid pixels
// ---------------------------------------------------------------------------

fn fill_nearest_neighbor(
    dense: &mut [Vec<f32>],
    confidence: &mut [Vec<f32>],
    height: usize,
    width: usize,
) {
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    let mut dist: Vec<Vec<u32>> = vec![vec![u32::MAX; width]; height];

    // Seed BFS from every known pixel.
    for r in 0..height {
        for c in 0..width {
            if dense[r][c] > 0.0 {
                queue.push_back((r, c));
                dist[r][c] = 0;
            }
        }
    }

    let dirs: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    while let Some((r, c)) = queue.pop_front() {
        let d = dist[r][c];
        for (dr, dc) in &dirs {
            let nr = r as i32 + dr;
            let nc = c as i32 + dc;
            if nr < 0 || nr >= height as i32 || nc < 0 || nc >= width as i32 {
                continue;
            }
            let (nr, nc) = (nr as usize, nc as usize);
            if dist[nr][nc] == u32::MAX {
                dist[nr][nc] = d + 1;
                dense[nr][nc] = dense[r][c]; // inherit from nearest seed
                                             // Confidence decays with distance: conf = 1 / (1 + d)
                let src_conf = confidence[r][c];
                confidence[nr][nc] = src_conf / (1.0 + (d + 1) as f32);
                queue.push_back((nr, nc));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// InvDistWeighted – weighted average of up to K=16 nearest valid pixels
// ---------------------------------------------------------------------------

fn fill_inv_dist_weighted(
    dense: &mut [Vec<f32>],
    confidence: &mut [Vec<f32>],
    height: usize,
    width: usize,
) {
    // Collect positions of all valid pixels (snapshot before mutation).
    let mut valid: Vec<(usize, usize, f32)> = Vec::new();
    for (r, dense_row) in dense.iter().enumerate().take(height) {
        for (c, &d) in dense_row.iter().enumerate().take(width) {
            if d > 0.0 {
                valid.push((r, c, d));
            }
        }
    }

    if valid.is_empty() {
        return;
    }

    const K: usize = 16;
    // Search radius in pixels: start small and expand.
    let max_radius = (height.max(width)) as f32;

    for r in 0..height {
        if dense[r].iter().all(|&d| d > 0.0) {
            // Row is fully filled; skip.
            continue;
        }
        for c in 0..width {
            if dense[r][c] > 0.0 {
                continue; // already known
            }

            // Collect K nearest valid points.
            let mut distances: Vec<(f32, f32)> = valid
                .iter()
                .map(|&(vr, vc, vd)| {
                    let dr = r as f32 - vr as f32;
                    let dc = c as f32 - vc as f32;
                    let dist = (dr * dr + dc * dc).sqrt();
                    (dist, vd)
                })
                .filter(|&(dist, _)| dist > 0.0 && dist <= max_radius)
                .collect();

            if distances.is_empty() {
                continue;
            }

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            distances.truncate(K);

            let mut wsum = 0.0f32;
            let mut dsum = 0.0f32;
            for (dist, depth) in &distances {
                let w = 1.0 / (dist * dist + 1e-6);
                wsum += w;
                dsum += w * depth;
            }

            if wsum > 0.0 {
                dense[r][c] = dsum / wsum;
                // Confidence based on nearest-point distance.
                let min_dist = distances[0].0;
                confidence[r][c] = 1.0 / (1.0 + min_dist);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PropagationFill – iterative 8-connected neighbourhood averaging
// ---------------------------------------------------------------------------

fn fill_propagation(
    dense: &mut [Vec<f32>],
    confidence: &mut [Vec<f32>],
    height: usize,
    width: usize,
    max_iterations: usize,
) {
    for _iter in 0..max_iterations {
        let mut changed = false;
        let old_dense = dense.to_vec();
        let old_conf = confidence.to_vec();

        for r in 0..height {
            for c in 0..width {
                if old_dense[r][c] > 0.0 {
                    continue; // already filled
                }
                // Gather 8-connected neighbours that are filled.
                let mut wsum = 0.0f32;
                let mut dsum = 0.0f32;
                let mut csum = 0.0f32;

                for dr in -1i32..=1 {
                    for dc in -1i32..=1 {
                        if dr == 0 && dc == 0 {
                            continue;
                        }
                        let nr = r as i32 + dr;
                        let nc = c as i32 + dc;
                        if nr < 0 || nr >= height as i32 || nc < 0 || nc >= width as i32 {
                            continue;
                        }
                        let (nr, nc) = (nr as usize, nc as usize);
                        let nd = old_dense[nr][nc];
                        if nd > 0.0 {
                            let w = old_conf[nr][nc].max(1e-6);
                            wsum += w;
                            dsum += w * nd;
                            csum += w;
                        }
                    }
                }

                if wsum > 0.0 {
                    let new_depth = dsum / wsum;
                    // average confidence of contributing neighbours, slight decay
                    let avg_conf = (csum / wsum.max(1e-6_f32)) * 0.9_f32;
                    dense[r][c] = new_depth;
                    confidence[r][c] = avg_conf;
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }

    // Depth consistency check: flag pixels whose depth differs too much from
    // local average (relative threshold 30 %).
    depth_consistency_check(dense, confidence, height, width);
}

/// Flag pixels whose depth deviates more than 30 % from the local 3×3 mean.
fn depth_consistency_check(
    dense: &mut [Vec<f32>],
    confidence: &mut [Vec<f32>],
    height: usize,
    width: usize,
) {
    let snap = dense.to_vec();
    for r in 0..height {
        for c in 0..width {
            let d = snap[r][c];
            if d <= 0.0 {
                continue;
            }
            let mut sum = 0.0f32;
            let mut cnt = 0usize;
            for dr in -1i32..=1 {
                for dc in -1i32..=1 {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr < 0 || nr >= height as i32 || nc < 0 || nc >= width as i32 {
                        continue;
                    }
                    let nd = snap[nr as usize][nc as usize];
                    if nd > 0.0 {
                        sum += nd;
                        cnt += 1;
                    }
                }
            }
            if cnt > 1 {
                let mean = sum / cnt as f32;
                let rel_diff = (d - mean).abs() / mean.max(1e-6);
                if rel_diff > 0.30 {
                    // Reduce confidence proportionally.
                    confidence[r][c] *= (1.0_f32 - rel_diff).max(0.0);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SurfaceNormals – Sobel gradients + normal integration
// ---------------------------------------------------------------------------

fn fill_surface_normals(
    dense: &mut [Vec<f32>],
    confidence: &mut [Vec<f32>],
    height: usize,
    width: usize,
    rgb: Option<&[Vec<[u8; 3]>]>,
) {
    // 1. Compute luminance from RGB (or use a uniform gradient if no RGB given).
    let lum: Vec<Vec<f32>> = match rgb {
        Some(img) if img.len() == height => img
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&[r, g, b]| {
                        // BT.601 luminance
                        0.299 * r as f32 / 255.0
                            + 0.587 * g as f32 / 255.0
                            + 0.114 * b as f32 / 255.0
                    })
                    .collect()
            })
            .collect(),
        _ => vec![vec![0.5f32; width]; height],
    };

    // 2. Sobel gradients (Gx, Gy) of luminance.
    let mut gx = vec![vec![0.0f32; width]; height];
    let mut gy = vec![vec![0.0f32; width]; height];

    for r in 1..(height.saturating_sub(1)) {
        for c in 1..(width.saturating_sub(1)) {
            gx[r][c] = -lum[r - 1][c - 1] - 2.0 * lum[r][c - 1] - lum[r + 1][c - 1]
                + lum[r - 1][c + 1]
                + 2.0 * lum[r][c + 1]
                + lum[r + 1][c + 1];
            gy[r][c] = -lum[r - 1][c - 1] - 2.0 * lum[r - 1][c] - lum[r - 1][c + 1]
                + lum[r + 1][c - 1]
                + 2.0 * lum[r + 1][c]
                + lum[r + 1][c + 1];
        }
    }

    // 3. Compute average depth from sparse anchors.
    let mut total_d = 0.0f32;
    let mut anchor_cnt = 0usize;
    for dense_row in dense.iter().take(height) {
        for &d in dense_row.iter().take(width) {
            if d > 0.0 {
                total_d += d;
                anchor_cnt += 1;
            }
        }
    }
    let anchor_mean = if anchor_cnt > 0 {
        total_d / anchor_cnt as f32
    } else {
        1.0
    };

    // 4. Integrate normals to produce a smooth depth surface.
    //    We use a simplified Frankot-Chellappa scheme: iterative Poisson solve.
    let mut integrated = vec![vec![anchor_mean; width]; height];

    // Copy anchor depths.
    for r in 0..height {
        for c in 0..width {
            if dense[r][c] > 0.0 {
                integrated[r][c] = dense[r][c];
            }
        }
    }

    // Poisson iterations.
    let n_iter = 10usize;
    for _ in 0..n_iter {
        let prev = integrated.clone();
        for r in 1..(height.saturating_sub(1)) {
            for c in 1..(width.saturating_sub(1)) {
                if dense[r][c] > 0.0 {
                    continue; // anchor: do not modify
                }
                // Discrete divergence of the gradient field.
                let lap = prev[r - 1][c] + prev[r + 1][c] + prev[r][c - 1] + prev[r][c + 1]
                    - 4.0 * prev[r][c];
                let rhs = gx[r][c] + gy[r][c];
                integrated[r][c] = prev[r][c] + 0.25 * (lap - rhs);
                integrated[r][c] = integrated[r][c].max(0.0);
            }
        }
    }

    // 5. Write integrated depth into output wherever it was empty.
    for r in 0..height {
        for c in 0..width {
            if dense[r][c] <= 0.0 {
                let d = integrated[r][c];
                if d > 0.0 {
                    dense[r][c] = d;
                    // Confidence based on gradient magnitude (strong edge → lower confidence).
                    let grad_mag = (gx[r][c] * gx[r][c] + gy[r][c] * gy[r][c]).sqrt();
                    confidence[r][c] = 1.0 / (1.0 + grad_mag);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: apply bilateral filter to a dense depth map
// ---------------------------------------------------------------------------

/// Apply a joint bilateral filter to `depth` using spatial sigma
/// `sigma_space` and depth-range sigma `sigma_depth`.
///
/// Pixels with depth ≤ 0 are treated as invalid and are not used as filter
/// centres, but can be filled by valid neighbours.
pub fn apply_bilateral_filter(
    depth: &[Vec<f32>],
    sigma_space: f32,
    sigma_depth: f32,
) -> Vec<Vec<f32>> {
    let height = depth.len();
    if height == 0 {
        return Vec::new();
    }
    let width = depth[0].len();
    let radius = (2.0 * sigma_space).ceil() as usize;
    let mut out = depth.to_vec();

    for r in 0..height {
        for c in 0..width {
            let centre = depth[r][c];
            if centre <= 0.0 {
                continue;
            }

            let mut wsum = 0.0f32;
            let mut dsum = 0.0f32;

            let r0 = r.saturating_sub(radius);
            let r1 = (r + radius + 1).min(height);
            let c0 = c.saturating_sub(radius);
            let c1 = (c + radius + 1).min(width);

            for (nr, depth_row) in depth.iter().enumerate().take(r1).skip(r0) {
                for (nc, &nd) in depth_row.iter().enumerate().take(c1).skip(c0) {
                    if nd <= 0.0 {
                        continue;
                    }
                    let dr = (r as f32 - nr as f32) / sigma_space;
                    let dc = (c as f32 - nc as f32) / sigma_space;
                    let dd = (centre - nd) / sigma_depth;
                    let w = (-(dr * dr + dc * dc + dd * dd) * 0.5).exp();
                    wsum += w;
                    dsum += w * nd;
                }
            }

            if wsum > 0.0 {
                out[r][c] = dsum / wsum;
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Helper: morphological dilation-based hole filling
// ---------------------------------------------------------------------------

/// Fill holes in `depth` using 3×3 morphological dilation.
///
/// Pixels with depth ≤ 0 are considered holes.  Dilation is applied once:
/// each empty pixel receives the maximum depth value of its 3×3 neighbourhood.
pub fn fill_holes_morphological(depth: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let height = depth.len();
    if height == 0 {
        return Vec::new();
    }
    let width = depth[0].len();
    let mut out = depth.to_vec();

    for r in 0..height {
        for c in 0..width {
            if depth[r][c] > 0.0 {
                continue; // not a hole
            }
            let mut max_d = 0.0f32;
            for dr in -1i32..=1 {
                for dc in -1i32..=1 {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr < 0 || nr >= height as i32 || nc < 0 || nc >= width as i32 {
                        continue;
                    }
                    max_d = max_d.max(depth[nr as usize][nc as usize]);
                }
            }
            if max_d > 0.0 {
                out[r][c] = max_d;
            }
        }
    }
    out
}
