//! Dense 3D reconstruction algorithms
//!
//! Implements Semi-Global Matching (SGM) for stereo disparity,
//! Multi-View Stereo (MVS) patch matching, TSDF-based depth fusion,
//! and Poisson surface reconstruction from point normals.

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::{Array2, Array3};

// ─────────────────────────────────────────────────────────────────────────────
// SGM – Semi-Global Matching
// ─────────────────────────────────────────────────────────────────────────────

/// Semi-Global Matching configuration.
#[derive(Debug, Clone)]
pub struct SGMConfig {
    /// Minimum disparity value.
    pub min_disparity: i32,
    /// Maximum disparity value (exclusive).
    pub max_disparity: i32,
    /// Half-size of the census/SAD window.
    pub window_radius: usize,
    /// Penalty P1 for small disparity changes (1 pixel).
    pub p1: f32,
    /// Penalty P2 for larger disparity changes.
    pub p2: f32,
}

impl Default for SGMConfig {
    fn default() -> Self {
        Self {
            min_disparity: 0,
            max_disparity: 64,
            window_radius: 2,
            p1: 10.0,
            p2: 120.0,
        }
    }
}

/// Semi-Global Matching stereo matcher.
pub struct SGM {
    config: SGMConfig,
}

impl SGM {
    /// Create a new SGM matcher with the given configuration.
    pub fn new(config: SGMConfig) -> Self {
        Self { config }
    }

    /// Compute a dense disparity map from rectified left and right images.
    ///
    /// Both images must be single-channel (grayscale) of identical shape.
    /// Returns an `Array2<f32>` of the same dimensions containing disparity values.
    pub fn compute(&self, left: &Array2<f32>, right: &Array2<f32>) -> Result<Array2<f32>> {
        let (rows, cols) = left.dim();
        if right.dim() != (rows, cols) {
            return Err(VisionError::DimensionMismatch(
                "SGM: left and right images must have identical dimensions".to_string(),
            ));
        }
        let cfg = &self.config;
        let nd = (cfg.max_disparity - cfg.min_disparity) as usize;
        if nd == 0 {
            return Err(VisionError::InvalidParameter(
                "SGM: max_disparity must be > min_disparity".to_string(),
            ));
        }

        // Step 1: Matching cost (SAD in a window)
        let cost = self.matching_cost(left, right, rows, cols, nd);

        // Step 2: Aggregate along 4 directions (horizontal ±, vertical ±)
        let aggregated = self.aggregate_cost(&cost, rows, cols, nd);

        // Step 3: Winner-takes-all disparity selection
        let mut disparity = Array2::<f32>::zeros((rows, cols));
        for r in 0..rows {
            for c in 0..cols {
                let mut best_cost = f32::MAX;
                let mut best_d = 0usize;
                for d in 0..nd {
                    let v = aggregated[[r, c, d]];
                    if v < best_cost {
                        best_cost = v;
                        best_d = d;
                    }
                }
                disparity[[r, c]] = best_d as f32 + cfg.min_disparity as f32;
            }
        }

        // Step 4: Sub-pixel refinement (quadratic interpolation)
        let disp_refined = subpixel_refine(&disparity, &aggregated, nd, rows, cols);
        Ok(disp_refined)
    }

    fn matching_cost(
        &self,
        left: &Array2<f32>,
        right: &Array2<f32>,
        rows: usize,
        cols: usize,
        nd: usize,
    ) -> Array3<f32> {
        let cfg = &self.config;
        let wr = cfg.window_radius;
        let mut cost = Array3::<f32>::zeros((rows, cols, nd));
        for r in 0..rows {
            for c in 0..cols {
                for d in 0..nd {
                    let disp = d as i32 + cfg.min_disparity;
                    let mut sad = 0.0f32;
                    let mut count = 0usize;
                    for dr in -(wr as i32)..=(wr as i32) {
                        for dc in -(wr as i32)..=(wr as i32) {
                            let lr = r as i32 + dr;
                            let lc = c as i32 + dc;
                            let rr = lr;
                            let rc = lc - disp;
                            if lr >= 0
                                && lr < rows as i32
                                && lc >= 0
                                && lc < cols as i32
                                && rr >= 0
                                && rr < rows as i32
                                && rc >= 0
                                && rc < cols as i32
                            {
                                sad += (left[[lr as usize, lc as usize]]
                                    - right[[rr as usize, rc as usize]])
                                .abs();
                                count += 1;
                            }
                        }
                    }
                    cost[[r, c, d]] = if count > 0 {
                        sad / count as f32
                    } else {
                        f32::MAX
                    };
                }
            }
        }
        cost
    }

    fn aggregate_cost(
        &self,
        cost: &Array3<f32>,
        rows: usize,
        cols: usize,
        nd: usize,
    ) -> Array3<f32> {
        let p1 = self.config.p1;
        let p2 = self.config.p2;

        // Aggregate along 4 directions and sum
        let mut agg = Array3::<f32>::zeros((rows, cols, nd));

        // Direction: left → right
        {
            let mut path = vec![vec![f32::MAX; nd]; cols];
            for c in 0..cols {
                for d in 0..nd {
                    path[c][d] = cost[[0, c, d]];
                }
            }
            for r in 0..rows {
                let mut prev = vec![f32::MAX; nd];
                for d in 0..nd {
                    prev[d] = cost[[r, 0, d]];
                    agg[[r, 0, d]] += prev[d];
                }
                for c in 1..cols {
                    let min_prev = prev.iter().cloned().fold(f32::MAX, f32::min);
                    let mut cur = vec![0.0f32; nd];
                    for d in 0..nd {
                        let m0 = prev[d];
                        let m1 = if d > 0 { prev[d - 1] + p1 } else { f32::MAX };
                        let m2 = if d + 1 < nd {
                            prev[d + 1] + p1
                        } else {
                            f32::MAX
                        };
                        let m3 = min_prev + p2;
                        let min_val = m0.min(m1).min(m2).min(m3) - min_prev;
                        cur[d] = cost[[r, c, d]] + min_val;
                        agg[[r, c, d]] += cur[d];
                    }
                    prev = cur;
                }
            }
        }

        // Direction: right → left
        {
            for r in 0..rows {
                let mut prev: Vec<f32> = (0..nd).map(|d| cost[[r, cols - 1, d]]).collect();
                for d in 0..nd {
                    agg[[r, cols - 1, d]] += prev[d];
                }
                for c in (0..(cols - 1)).rev() {
                    let min_prev = prev.iter().cloned().fold(f32::MAX, f32::min);
                    let mut cur = vec![0.0f32; nd];
                    for d in 0..nd {
                        let m0 = prev[d];
                        let m1 = if d > 0 { prev[d - 1] + p1 } else { f32::MAX };
                        let m2 = if d + 1 < nd {
                            prev[d + 1] + p1
                        } else {
                            f32::MAX
                        };
                        let m3 = min_prev + p2;
                        let min_val = m0.min(m1).min(m2).min(m3) - min_prev;
                        cur[d] = cost[[r, c, d]] + min_val;
                        agg[[r, c, d]] += cur[d];
                    }
                    prev = cur;
                }
            }
        }

        // Direction: top → bottom
        {
            for c in 0..cols {
                let mut prev: Vec<f32> = (0..nd).map(|d| cost[[0, c, d]]).collect();
                for d in 0..nd {
                    agg[[0, c, d]] += prev[d];
                }
                for r in 1..rows {
                    let min_prev = prev.iter().cloned().fold(f32::MAX, f32::min);
                    let mut cur = vec![0.0f32; nd];
                    for d in 0..nd {
                        let m0 = prev[d];
                        let m1 = if d > 0 { prev[d - 1] + p1 } else { f32::MAX };
                        let m2 = if d + 1 < nd {
                            prev[d + 1] + p1
                        } else {
                            f32::MAX
                        };
                        let m3 = min_prev + p2;
                        let min_val = m0.min(m1).min(m2).min(m3) - min_prev;
                        cur[d] = cost[[r, c, d]] + min_val;
                        agg[[r, c, d]] += cur[d];
                    }
                    prev = cur;
                }
            }
        }

        // Direction: bottom → top
        {
            for c in 0..cols {
                let mut prev: Vec<f32> = (0..nd).map(|d| cost[[rows - 1, c, d]]).collect();
                for d in 0..nd {
                    agg[[rows - 1, c, d]] += prev[d];
                }
                for r in (0..(rows - 1)).rev() {
                    let min_prev = prev.iter().cloned().fold(f32::MAX, f32::min);
                    let mut cur = vec![0.0f32; nd];
                    for d in 0..nd {
                        let m0 = prev[d];
                        let m1 = if d > 0 { prev[d - 1] + p1 } else { f32::MAX };
                        let m2 = if d + 1 < nd {
                            prev[d + 1] + p1
                        } else {
                            f32::MAX
                        };
                        let m3 = min_prev + p2;
                        let min_val = m0.min(m1).min(m2).min(m3) - min_prev;
                        cur[d] = cost[[r, c, d]] + min_val;
                        agg[[r, c, d]] += cur[d];
                    }
                    prev = cur;
                }
            }
        }

        agg
    }
}

fn subpixel_refine(
    disp: &Array2<f32>,
    agg: &Array3<f32>,
    nd: usize,
    rows: usize,
    cols: usize,
) -> Array2<f32> {
    let mut refined = disp.clone();
    for r in 0..rows {
        for c in 0..cols {
            let d = disp[[r, c]] as usize;
            if d > 0 && d + 1 < nd {
                let c0 = agg[[r, c, d - 1]];
                let c1 = agg[[r, c, d]];
                let c2 = agg[[r, c, d + 1]];
                let denom = c0 - 2.0 * c1 + c2;
                if denom.abs() > 1e-6 {
                    refined[[r, c]] = disp[[r, c]] - 0.5 * (c2 - c0) / denom;
                }
            }
        }
    }
    refined
}

// ─────────────────────────────────────────────────────────────────────────────
// MVS – Multi-View Stereo
// ─────────────────────────────────────────────────────────────────────────────

/// A single view for MVS (image + projection matrix).
#[derive(Debug, Clone)]
pub struct MVSView {
    /// Grayscale image as `Array2<f32>`.
    pub image: Array2<f32>,
    /// 3×4 projection matrix P = K [R | t].
    pub proj: Array2<f64>,
}

/// MVS patch-matching configuration.
#[derive(Debug, Clone)]
pub struct MVSConfig {
    /// Half-size of the photometric patch.
    pub patch_radius: usize,
    /// Number of depth hypothesis planes.
    pub num_depth_planes: usize,
    /// Minimum scene depth.
    pub depth_min: f64,
    /// Maximum scene depth.
    pub depth_max: f64,
    /// NCC score threshold for a valid match.
    pub ncc_threshold: f32,
}

impl Default for MVSConfig {
    fn default() -> Self {
        Self {
            patch_radius: 5,
            num_depth_planes: 32,
            depth_min: 0.5,
            depth_max: 10.0,
            ncc_threshold: 0.7,
        }
    }
}

/// Multi-view stereo depth estimator using plane-sweep and NCC.
pub struct MVS {
    config: MVSConfig,
}

impl MVS {
    /// Create a new MVS instance.
    pub fn new(config: MVSConfig) -> Self {
        Self { config }
    }

    /// Estimate a depth map for `reference` using `source` views.
    ///
    /// Returns an `Array2<f64>` where each pixel holds the estimated depth
    /// (or 0.0 if no reliable estimate was found).
    pub fn estimate_depth(&self, reference: &MVSView, sources: &[MVSView]) -> Result<Array2<f64>> {
        if sources.is_empty() {
            return Err(VisionError::InvalidParameter(
                "MVS: at least one source view is required".to_string(),
            ));
        }
        let (rows, cols) = reference.image.dim();
        let mut depth_map = Array2::<f64>::zeros((rows, cols));
        let cfg = &self.config;

        // Depth hypotheses spaced logarithmically
        let depths: Vec<f64> = (0..cfg.num_depth_planes)
            .map(|i| {
                let t = i as f64 / (cfg.num_depth_planes - 1).max(1) as f64;
                cfg.depth_min * (cfg.depth_max / cfg.depth_min).powf(t)
            })
            .collect();

        for r in 0..rows {
            for c in 0..cols {
                let mut best_score = -1.0f32;
                let mut best_depth = 0.0f64;
                for &d in &depths {
                    // For this pixel + depth, compute the 3D point
                    let pt3d = unproject(&reference.proj, r, c, d);
                    // Accumulate NCC over all source views
                    let mut total_ncc = 0.0f32;
                    let mut valid_count = 0usize;
                    for src in sources {
                        if let Some(ncc) = photometric_ncc(
                            &reference.image,
                            &src.image,
                            r,
                            c,
                            &pt3d,
                            &src.proj,
                            cfg.patch_radius,
                        ) {
                            total_ncc += ncc;
                            valid_count += 1;
                        }
                    }
                    if valid_count > 0 {
                        let avg_ncc = total_ncc / valid_count as f32;
                        if avg_ncc > best_score {
                            best_score = avg_ncc;
                            best_depth = d;
                        }
                    }
                }
                if best_score >= cfg.ncc_threshold {
                    depth_map[[r, c]] = best_depth;
                }
            }
        }
        Ok(depth_map)
    }
}

/// Unproject a pixel at `(row, col)` with given `depth` using projection matrix.
fn unproject(proj: &Array2<f64>, row: usize, col: usize, depth: f64) -> [f64; 3] {
    // Simple back-projection assuming P = K [R|t].
    // This is approximate; proper MVS uses the pseudo-inverse.
    let x = col as f64;
    let y = row as f64;
    // Use first two rows of P to find direction
    let fx = proj[[0, 0]];
    let fy = proj[[1, 1]];
    let cx = proj[[0, 2]];
    let cy = proj[[1, 2]];
    let fx = if fx.abs() < 1e-10 { 1.0 } else { fx };
    let fy = if fy.abs() < 1e-10 { 1.0 } else { fy };
    [(x - cx) / fx * depth, (y - cy) / fy * depth, depth]
}

/// Project a 3D point via P and return fractional (col, row).
fn project_pt(proj: &Array2<f64>, pt: &[f64; 3]) -> Option<(f64, f64)> {
    let xw = pt[0];
    let yw = pt[1];
    let zw = pt[2];
    let u = proj[[0, 0]] * xw + proj[[0, 1]] * yw + proj[[0, 2]] * zw + proj[[0, 3]];
    let v = proj[[1, 0]] * xw + proj[[1, 1]] * yw + proj[[1, 2]] * zw + proj[[1, 3]];
    let w = proj[[2, 0]] * xw + proj[[2, 1]] * yw + proj[[2, 2]] * zw + proj[[2, 3]];
    if w.abs() < 1e-10 {
        return None;
    }
    Some((u / w, v / w))
}

/// Compute Normalized Cross-Correlation between a reference patch at `(ref_r, ref_c)`
/// and the patch in `src` centred at the projection of `pt3d`.
fn photometric_ncc(
    reference: &Array2<f32>,
    src: &Array2<f32>,
    ref_r: usize,
    ref_c: usize,
    pt3d: &[f64; 3],
    src_proj: &Array2<f64>,
    patch_radius: usize,
) -> Option<f32> {
    let (ref_rows, ref_cols) = reference.dim();
    let (src_rows, src_cols) = src.dim();
    let (sc, sr) = project_pt(src_proj, pt3d)?;
    let sr_i = sr.round() as i32;
    let sc_i = sc.round() as i32;
    if sr_i < patch_radius as i32
        || sr_i + patch_radius as i32 >= src_rows as i32
        || sc_i < patch_radius as i32
        || sc_i + patch_radius as i32 >= src_cols as i32
    {
        return None;
    }
    let pr = patch_radius as i32;
    // Collect patch values
    let mut ref_vals = Vec::new();
    let mut src_vals = Vec::new();
    for dr in -pr..=pr {
        for dc in -pr..=pr {
            let rr = ref_r as i32 + dr;
            let rc = ref_c as i32 + dc;
            if rr < 0 || rr >= ref_rows as i32 || rc < 0 || rc >= ref_cols as i32 {
                return None;
            }
            ref_vals.push(reference[[rr as usize, rc as usize]]);
            src_vals.push(src[[(sr_i + dr) as usize, (sc_i + dc) as usize]]);
        }
    }
    let n = ref_vals.len() as f32;
    let mean_r = ref_vals.iter().sum::<f32>() / n;
    let mean_s = src_vals.iter().sum::<f32>() / n;
    let mut num = 0.0f32;
    let mut denom_r = 0.0f32;
    let mut denom_s = 0.0f32;
    for (r, s) in ref_vals.iter().zip(src_vals.iter()) {
        let dr = r - mean_r;
        let ds = s - mean_s;
        num += dr * ds;
        denom_r += dr * dr;
        denom_s += ds * ds;
    }
    let denom = (denom_r * denom_s).sqrt();
    if denom < 1e-6 {
        return None;
    }
    Some((num / denom).clamp(-1.0, 1.0))
}

// ─────────────────────────────────────────────────────────────────────────────
// TSDF Depth Fusion
// ─────────────────────────────────────────────────────────────────────────────

/// A volumetric TSDF voxel grid for depth map fusion.
pub struct DepthFusion {
    /// Voxel grid (tsdf values).
    tsdf: Vec<f32>,
    /// Weight grid.
    weights: Vec<f32>,
    /// Grid resolution along each axis.
    resolution: [usize; 3],
    /// Physical extent of the volume (metres) along each axis.
    extent: [f64; 3],
    /// Origin of the volume in world coordinates.
    origin: [f64; 3],
    /// Truncation distance (metres).
    truncation: f64,
}

impl DepthFusion {
    /// Create a new TSDF volume.
    ///
    /// - `resolution`: number of voxels along \[x, y, z\].
    /// - `extent`: physical size in metres along each axis.
    /// - `origin`: world-space origin of the volume.
    /// - `truncation`: truncation distance in metres.
    pub fn new(
        resolution: [usize; 3],
        extent: [f64; 3],
        origin: [f64; 3],
        truncation: f64,
    ) -> Self {
        let total = resolution[0] * resolution[1] * resolution[2];
        Self {
            tsdf: vec![1.0f32; total],
            weights: vec![0.0f32; total],
            resolution,
            extent,
            origin,
            truncation,
        }
    }

    /// Integrate a depth map into the TSDF volume.
    ///
    /// - `depth`: depth image as `Array2<f64>`.
    /// - `proj`: 3×4 camera projection matrix mapping world→image.
    /// - `max_weight`: maximum weight per voxel (prevents over-weighting old data).
    pub fn integrate(
        &mut self,
        depth: &Array2<f64>,
        proj: &Array2<f64>,
        max_weight: f32,
    ) -> Result<()> {
        let (img_rows, img_cols) = depth.dim();
        let [nx, ny, nz] = self.resolution;
        let [ex, ey, ez] = self.extent;
        let [ox, oy, oz] = self.origin;

        let vx = ex / nx as f64;
        let vy = ey / ny as f64;
        let vz = ez / nz as f64;

        for xi in 0..nx {
            for yi in 0..ny {
                for zi in 0..nz {
                    let wx = ox + (xi as f64 + 0.5) * vx;
                    let wy = oy + (yi as f64 + 0.5) * vy;
                    let wz = oz + (zi as f64 + 0.5) * vz;
                    // Project to camera
                    let pt = [wx, wy, wz];
                    let (px, py) = match project_pt(proj, &pt) {
                        Some(p) => p,
                        None => continue,
                    };
                    let pi = px.round() as i32;
                    let pj = py.round() as i32;
                    if pi < 0 || pi >= img_cols as i32 || pj < 0 || pj >= img_rows as i32 {
                        continue;
                    }
                    let measured_depth = depth[[pj as usize, pi as usize]];
                    if measured_depth <= 0.0 {
                        continue;
                    }
                    // Compute signed distance
                    // Camera depth of voxel (approximate z component)
                    let voxel_depth = wz; // simplified: world z as camera depth
                    let sdf = measured_depth - voxel_depth;
                    if sdf < -self.truncation {
                        continue;
                    }
                    let tsdf_val = (sdf / self.truncation).clamp(-1.0, 1.0) as f32;
                    let idx = xi * ny * nz + yi * nz + zi;
                    let old_w = self.weights[idx];
                    let old_tsdf = self.tsdf[idx];
                    let new_w = (old_w + 1.0).min(max_weight);
                    self.tsdf[idx] = (old_w * old_tsdf + tsdf_val) / new_w;
                    self.weights[idx] = new_w;
                }
            }
        }
        Ok(())
    }

    /// Extract a point cloud from the zero-crossing surface of the TSDF.
    ///
    /// Returns world-space points at voxels near the surface (|tsdf| < threshold).
    pub fn extract_surface(&self, surface_threshold: f32) -> Vec<[f64; 3]> {
        let [nx, ny, nz] = self.resolution;
        let [ex, ey, ez] = self.extent;
        let [ox, oy, oz] = self.origin;
        let vx = ex / nx as f64;
        let vy = ey / ny as f64;
        let vz = ez / nz as f64;

        let mut pts = Vec::new();
        for xi in 0..nx {
            for yi in 0..ny {
                for zi in 0..nz {
                    let idx = xi * ny * nz + yi * nz + zi;
                    if self.weights[idx] > 0.0 && self.tsdf[idx].abs() < surface_threshold {
                        let wx = ox + (xi as f64 + 0.5) * vx;
                        let wy = oy + (yi as f64 + 0.5) * vy;
                        let wz = oz + (zi as f64 + 0.5) * vz;
                        pts.push([wx, wy, wz]);
                    }
                }
            }
        }
        pts
    }

    /// Query TSDF value at voxel index `[xi, yi, zi]`.
    pub fn get_tsdf(&self, xi: usize, yi: usize, zi: usize) -> Option<f32> {
        let [_nx, ny, nz] = self.resolution;
        let idx = xi * ny * nz + yi * nz + zi;
        if idx < self.tsdf.len() {
            Some(self.tsdf[idx])
        } else {
            None
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Poisson Surface Reconstruction
// ─────────────────────────────────────────────────────────────────────────────

/// An oriented point (position + normal) for Poisson reconstruction.
#[derive(Debug, Clone)]
pub struct OrientedPoint {
    /// 3D position.
    pub position: [f64; 3],
    /// Outward-facing unit normal.
    pub normal: [f64; 3],
}

/// Simplified Poisson surface reconstruction.
///
/// This implementation approximates the full Poisson equation via
/// a weighted kernel density estimate on a regular grid, suitable
/// for moderate-size point clouds. For production use, a full
/// conjugate-gradient Poisson solver is recommended.
pub struct PoissonSurface {
    /// Grid resolution per axis.
    pub grid_resolution: usize,
    /// Kernel bandwidth (fraction of volume extent).
    pub bandwidth: f64,
    /// Iso-surface extraction threshold.
    pub iso_threshold: f64,
}

impl Default for PoissonSurface {
    fn default() -> Self {
        Self {
            grid_resolution: 32,
            bandwidth: 0.05,
            iso_threshold: 0.5,
        }
    }
}

impl PoissonSurface {
    /// Create a new Poisson surface reconstructor.
    pub fn new(grid_resolution: usize, bandwidth: f64, iso_threshold: f64) -> Self {
        Self {
            grid_resolution,
            bandwidth,
            iso_threshold,
        }
    }

    /// Reconstruct a surface mesh (as a set of vertices) from oriented points.
    ///
    /// Returns a list of `[x, y, z]` vertices on the reconstructed surface.
    pub fn reconstruct(&self, oriented_pts: &[OrientedPoint]) -> Result<Vec<[f64; 3]>> {
        if oriented_pts.is_empty() {
            return Err(VisionError::InvalidParameter(
                "PoissonSurface: at least one oriented point required".to_string(),
            ));
        }
        let n = self.grid_resolution;
        // Bounding box
        let (mut min_x, mut min_y, mut min_z) = (f64::MAX, f64::MAX, f64::MAX);
        let (mut max_x, mut max_y, mut max_z) = (f64::MIN, f64::MIN, f64::MIN);
        for p in oriented_pts {
            min_x = min_x.min(p.position[0]);
            min_y = min_y.min(p.position[1]);
            min_z = min_z.min(p.position[2]);
            max_x = max_x.max(p.position[0]);
            max_y = max_y.max(p.position[1]);
            max_z = max_z.max(p.position[2]);
        }
        let pad = 1.1;
        let cx = (min_x + max_x) * 0.5;
        let cy = (min_y + max_y) * 0.5;
        let cz = (min_z + max_z) * 0.5;
        let half = (((max_x - min_x).max(max_y - min_y).max(max_z - min_z)) * 0.5 * pad).max(1e-6);
        let voxel_size = 2.0 * half / n as f64;
        let sigma = self.bandwidth * 2.0 * half;

        // Build divergence field approximation
        // For each grid point, evaluate the indicator function approximation
        // using the sum of kernel-weighted normal contributions
        let total = n * n * n;
        let mut indicator = vec![0.0f64; total];

        for op in oriented_pts {
            let pi = ((op.position[0] - (cx - half)) / voxel_size) as i32;
            let pj = ((op.position[1] - (cy - half)) / voxel_size) as i32;
            let pk = ((op.position[2] - (cz - half)) / voxel_size) as i32;
            let radius = (3.0 * sigma / voxel_size).ceil() as i32;
            for di in -radius..=radius {
                for dj in -radius..=radius {
                    for dk in -radius..=radius {
                        let xi = pi + di;
                        let yj = pj + dj;
                        let zk = pk + dk;
                        if xi < 0
                            || xi >= n as i32
                            || yj < 0
                            || yj >= n as i32
                            || zk < 0
                            || zk >= n as i32
                        {
                            continue;
                        }
                        let gx = (cx - half) + (xi as f64 + 0.5) * voxel_size;
                        let gy = (cy - half) + (yj as f64 + 0.5) * voxel_size;
                        let gz = (cz - half) + (zk as f64 + 0.5) * voxel_size;
                        let dx = gx - op.position[0];
                        let dy = gy - op.position[1];
                        let dz = gz - op.position[2];
                        let dist2 = dx * dx + dy * dy + dz * dz;
                        let kernel = (-dist2 / (2.0 * sigma * sigma)).exp();
                        // Dot normal with grid→point direction
                        let contribution =
                            kernel * (op.normal[0] * dx + op.normal[1] * dy + op.normal[2] * dz);
                        let idx = xi as usize * n * n + yj as usize * n + zk as usize;
                        indicator[idx] += contribution;
                    }
                }
            }
        }

        // Normalise
        let max_ind = indicator.iter().cloned().fold(0.0f64, f64::max).max(1e-10);
        for v in &mut indicator {
            *v /= max_ind;
        }

        // Extract iso-surface points (cells where indicator crosses threshold)
        let mut surface_pts = Vec::new();
        let thresh = self.iso_threshold;
        for xi in 0..(n - 1) {
            for yj in 0..(n - 1) {
                for zk in 0..(n - 1) {
                    let idx = xi * n * n + yj * n + zk;
                    let v000 = indicator[idx];
                    let v100 = indicator[(xi + 1) * n * n + yj * n + zk];
                    let v010 = indicator[xi * n * n + (yj + 1) * n + zk];
                    let v001 = indicator[xi * n * n + yj * n + (zk + 1)];
                    // Check if any edge crosses threshold
                    let has_crossing = (v000 < thresh) != (v100 < thresh)
                        || (v000 < thresh) != (v010 < thresh)
                        || (v000 < thresh) != (v001 < thresh);
                    if has_crossing {
                        let gx = (cx - half) + (xi as f64 + 0.5) * voxel_size;
                        let gy = (cy - half) + (yj as f64 + 0.5) * voxel_size;
                        let gz = (cz - half) + (zk as f64 + 0.5) * voxel_size;
                        surface_pts.push([gx, gy, gz]);
                    }
                }
            }
        }
        Ok(surface_pts)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgm_basic() {
        let rows = 20;
        let cols = 40;
        let left = Array2::<f32>::zeros((rows, cols));
        let right = Array2::<f32>::zeros((rows, cols));
        let sgm = SGM::new(SGMConfig {
            min_disparity: 0,
            max_disparity: 8,
            window_radius: 1,
            ..Default::default()
        });
        let disp = sgm.compute(&left, &right);
        assert!(disp.is_ok());
        let d = disp.expect("SGM compute should succeed on valid inputs");
        assert_eq!(d.dim(), (rows, cols));
    }

    #[test]
    fn test_sgm_dimension_mismatch() {
        let left = Array2::<f32>::zeros((10, 20));
        let right = Array2::<f32>::zeros((10, 21));
        let sgm = SGM::new(SGMConfig::default());
        assert!(sgm.compute(&left, &right).is_err());
    }

    #[test]
    fn test_depth_fusion_integrate() {
        let depth = Array2::<f64>::from_elem((10, 10), 1.0);
        let mut proj = Array2::<f64>::zeros((3, 4));
        proj[[0, 0]] = 100.0;
        proj[[1, 1]] = 100.0;
        proj[[0, 2]] = 5.0;
        proj[[1, 2]] = 5.0;
        proj[[2, 2]] = 1.0;
        let mut fusion = DepthFusion::new([4, 4, 4], [2.0, 2.0, 2.0], [0.0, 0.0, 0.0], 0.1);
        let result = fusion.integrate(&depth, &proj, 100.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_poisson_surface() {
        let pts: Vec<OrientedPoint> = (0..20)
            .map(|i| {
                let angle = i as f64 * std::f64::consts::TAU / 20.0;
                OrientedPoint {
                    position: [angle.cos(), angle.sin(), 0.0],
                    normal: [angle.cos(), angle.sin(), 0.0],
                }
            })
            .collect();
        let ps = PoissonSurface::new(16, 0.1, 0.3);
        let result = ps.reconstruct(&pts);
        assert!(result.is_ok());
    }
}
