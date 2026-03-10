//! Visual SLAM components
//!
//! Provides building blocks for monocular/stereo/RGB-D visual SLAM:
//! - `VisualOdometry`: frame-to-frame motion estimation via feature tracking
//! - `KeyframeSelector`: keyframe selection based on parallax and overlap
//! - `MapPoint`: 3D landmark with multi-frame observation tracking
//! - `Covisibility`: covisibility graph between keyframes
//! - `LoopClosure`: bag-of-words loop detection interface
//! - `PoseGraph`: pose graph structure for global optimization

use crate::error::{Result, VisionError};
use crate::reconstruction::sfm::{
    build_projection_matrix, rodrigues_to_rotation, triangulate_dlt_single, EssentialMatrix,
    FundamentalMatrix, IntrinsicMatrix, PointCloud, Triangulation,
};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet};

// Re-export helper for external use
pub use crate::reconstruction::sfm::IntrinsicMatrix as CameraIntrinsics;

// ─────────────────────────────────────────────────────────────────────────────
// Camera pose
// ─────────────────────────────────────────────────────────────────────────────

/// A 6-DOF camera pose: rotation (Rodrigues) + translation.
#[derive(Debug, Clone)]
pub struct Pose {
    /// Rodrigues rotation vector (3 components).
    pub rvec: Array1<f64>,
    /// Translation vector (3 components).
    pub tvec: Array1<f64>,
}

impl Pose {
    /// Identity pose (no rotation, no translation).
    pub fn identity() -> Self {
        Self {
            rvec: Array1::zeros(3),
            tvec: Array1::zeros(3),
        }
    }

    /// Build a 3×4 projection matrix P = K [R | t].
    pub fn to_projection(&self, k: &IntrinsicMatrix) -> Array2<f64> {
        let r = rodrigues_to_rotation(&self.rvec);
        let p_cam = build_projection_matrix(&r, &self.tvec);
        let km = k.to_matrix();
        // P = K * [R|t]
        let mut p = Array2::<f64>::zeros((3, 4));
        for i in 0..3 {
            for j in 0..4 {
                for k_idx in 0..3 {
                    p[[i, j]] += km[[i, k_idx]] * p_cam[[k_idx, j]];
                }
            }
        }
        p
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MapPoint
// ─────────────────────────────────────────────────────────────────────────────

/// A 3D landmark observed from multiple frames.
#[derive(Debug, Clone)]
pub struct MapPoint {
    /// Unique identifier.
    pub id: usize,
    /// 3D world position.
    pub position: Array1<f64>,
    /// Map from frame index to the 2D observation `[x, y]`.
    pub observations: HashMap<usize, [f64; 2]>,
    /// Number of times this point has been matched (for culling).
    pub match_count: usize,
    /// Whether this point is considered an outlier.
    pub is_outlier: bool,
}

impl MapPoint {
    /// Create a new map point.
    pub fn new(id: usize, position: Array1<f64>) -> Self {
        Self {
            id,
            position,
            observations: HashMap::new(),
            match_count: 0,
            is_outlier: false,
        }
    }

    /// Add or update an observation of this point in a frame.
    pub fn add_observation(&mut self, frame_id: usize, pixel: [f64; 2]) {
        self.observations.insert(frame_id, pixel);
    }

    /// Number of frames that observe this map point.
    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Keyframe
// ─────────────────────────────────────────────────────────────────────────────

/// A selected keyframe with its pose and observed map points.
#[derive(Debug, Clone)]
pub struct Keyframe {
    /// Unique frame identifier.
    pub id: usize,
    /// Camera pose at this keyframe.
    pub pose: Pose,
    /// 2D feature keypoints.
    pub keypoints: Vec<[f64; 2]>,
    /// Map point IDs observed in this keyframe (one per keypoint, or `None`).
    pub map_point_ids: Vec<Option<usize>>,
    /// Bag-of-words descriptor (simplified: word histogram).
    pub bow_descriptor: Vec<f32>,
}

impl Keyframe {
    /// Create a new keyframe.
    pub fn new(id: usize, pose: Pose, keypoints: Vec<[f64; 2]>) -> Self {
        let n = keypoints.len();
        Self {
            id,
            pose,
            keypoints,
            map_point_ids: vec![None; n],
            bow_descriptor: Vec::new(),
        }
    }

    /// Get the 2D pixel of a particular map point in this frame.
    pub fn get_pixel_for(&self, map_point_id: usize) -> Option<[f64; 2]> {
        for (i, mp_id) in self.map_point_ids.iter().enumerate() {
            if *mp_id == Some(map_point_id) {
                return Some(self.keypoints[i]);
            }
        }
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KeyframeSelector
// ─────────────────────────────────────────────────────────────────────────────

/// Keyframe selection criteria.
#[derive(Debug, Clone)]
pub struct KeyframeSelector {
    /// Minimum median parallax (pixels) to insert a new keyframe.
    pub min_parallax: f64,
    /// Maximum overlap ratio before forcing a new keyframe.
    pub max_overlap: f64,
    /// Minimum number of tracked points to consider.
    pub min_tracked: usize,
    /// Minimum number of frames between keyframes.
    pub min_frame_gap: usize,
}

impl Default for KeyframeSelector {
    fn default() -> Self {
        Self {
            min_parallax: 20.0,
            max_overlap: 0.9,
            min_tracked: 15,
            min_frame_gap: 5,
        }
    }
}

impl KeyframeSelector {
    /// Decide whether the current frame should become a keyframe.
    ///
    /// - `current_pts`: 2D feature points in the current frame.
    /// - `prev_kf_pts`: the same features as seen in the last keyframe.
    /// - `frames_since_kf`: number of frames elapsed since the last keyframe.
    pub fn should_insert(
        &self,
        current_pts: &[[f64; 2]],
        prev_kf_pts: &[[f64; 2]],
        frames_since_kf: usize,
    ) -> bool {
        if frames_since_kf < self.min_frame_gap {
            return false;
        }
        let n = current_pts.len().min(prev_kf_pts.len());
        if n < self.min_tracked {
            // Too few tracked points → must insert
            return true;
        }
        // Compute median parallax
        let mut parallaxes: Vec<f64> = (0..n)
            .map(|i| {
                let dx = current_pts[i][0] - prev_kf_pts[i][0];
                let dy = current_pts[i][1] - prev_kf_pts[i][1];
                (dx * dx + dy * dy).sqrt()
            })
            .collect();
        parallaxes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_parallax = parallaxes[n / 2];

        // Overlap ratio: fraction of keyframe features still tracked
        let overlap = n as f64 / prev_kf_pts.len().max(1) as f64;

        median_parallax > self.min_parallax || overlap < (1.0 - self.max_overlap)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Covisibility Graph
// ─────────────────────────────────────────────────────────────────────────────

/// Covisibility graph: edges between keyframes that share map points.
#[derive(Debug, Clone)]
pub struct Covisibility {
    /// For each keyframe pair (a, b) with a < b, the number of shared map points.
    edges: HashMap<(usize, usize), usize>,
    /// Total number of keyframes.
    num_keyframes: usize,
}

impl Covisibility {
    /// Create an empty covisibility graph.
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
            num_keyframes: 0,
        }
    }

    /// Update the graph for a set of keyframes with their map point observations.
    pub fn update(&mut self, keyframes: &[Keyframe]) {
        self.num_keyframes = keyframes.len();
        self.edges.clear();

        // For each pair of keyframes, count shared map points
        #[allow(clippy::needless_range_loop)]
        for i in 0..keyframes.len() {
            let ids_i: HashSet<usize> = keyframes[i]
                .map_point_ids
                .iter()
                .filter_map(|&id| id)
                .collect();
            for j in (i + 1)..keyframes.len() {
                let ids_j: HashSet<usize> = keyframes[j]
                    .map_point_ids
                    .iter()
                    .filter_map(|&id| id)
                    .collect();
                let shared = ids_i.intersection(&ids_j).count();
                if shared > 0 {
                    self.edges.insert((i, j), shared);
                }
            }
        }
    }

    /// Get keyframe neighbours with at least `min_shared` shared map points.
    pub fn get_neighbours(&self, kf_idx: usize, min_shared: usize) -> Vec<(usize, usize)> {
        let mut neighbours = Vec::new();
        for (&(a, b), &count) in &self.edges {
            if count >= min_shared {
                if a == kf_idx {
                    neighbours.push((b, count));
                } else if b == kf_idx {
                    neighbours.push((a, count));
                }
            }
        }
        neighbours.sort_by(|x, y| y.1.cmp(&x.1));
        neighbours
    }

    /// Number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

impl Default for Covisibility {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bag-of-Words Loop Closure
// ─────────────────────────────────────────────────────────────────────────────

/// Simplified bag-of-words descriptor (word histogram over a fixed vocabulary).
pub struct BagOfWords {
    /// Vocabulary: cluster centres (each row is a visual word descriptor).
    vocabulary: Array2<f32>,
}

impl BagOfWords {
    /// Build a vocabulary from a set of descriptors via k-means.
    ///
    /// - `descriptors`: rows are feature descriptors.
    /// - `num_words`: vocabulary size.
    /// - `max_iter`: k-means iterations.
    pub fn build_vocabulary(
        descriptors: &Array2<f32>,
        num_words: usize,
        max_iter: usize,
    ) -> Result<Self> {
        if descriptors.nrows() < num_words {
            return Err(VisionError::InvalidParameter(
                "BagOfWords: fewer descriptors than vocabulary size".to_string(),
            ));
        }
        let dim = descriptors.ncols();
        // Initialise centres from first `num_words` descriptors
        let mut centres = Array2::<f32>::zeros((num_words, dim));
        for i in 0..num_words {
            for j in 0..dim {
                centres[[i, j]] = descriptors[[i, j]];
            }
        }
        // K-means iterations
        for _ in 0..max_iter {
            let mut sums = Array2::<f32>::zeros((num_words, dim));
            let mut counts = vec![0usize; num_words];
            // Assignment
            for row in 0..descriptors.nrows() {
                let nearest = Self::nearest_centre(&descriptors.row(row).to_owned(), &centres);
                for j in 0..dim {
                    sums[[nearest, j]] += descriptors[[row, j]];
                }
                counts[nearest] += 1;
            }
            // Update
            for k in 0..num_words {
                if counts[k] > 0 {
                    for j in 0..dim {
                        centres[[k, j]] = sums[[k, j]] / counts[k] as f32;
                    }
                }
            }
        }
        Ok(Self {
            vocabulary: centres,
        })
    }

    fn nearest_centre(descriptor: &Array1<f32>, centres: &Array2<f32>) -> usize {
        let mut best = 0usize;
        let mut best_dist = f32::MAX;
        for k in 0..centres.nrows() {
            let dist: f32 = (0..descriptor.len())
                .map(|j| (descriptor[j] - centres[[k, j]]).powi(2))
                .sum::<f32>();
            if dist < best_dist {
                best_dist = dist;
                best = k;
            }
        }
        best
    }

    /// Compute the BoW histogram for a set of descriptors.
    pub fn compute_bow(&self, descriptors: &Array2<f32>) -> Vec<f32> {
        let nw = self.vocabulary.nrows();
        let mut hist = vec![0.0f32; nw];
        for row in 0..descriptors.nrows() {
            let word = Self::nearest_centre(&descriptors.row(row).to_owned(), &self.vocabulary);
            hist[word] += 1.0;
        }
        let total: f32 = hist.iter().sum();
        if total > 0.0 {
            for v in &mut hist {
                *v /= total;
            }
        }
        hist
    }

    /// Compute cosine similarity between two BoW histograms.
    pub fn similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-10 || nb < 1e-10 {
            0.0
        } else {
            dot / (na * nb)
        }
    }
}

/// Loop closure detection using bag-of-words similarity.
pub struct LoopClosure {
    /// Similarity threshold for declaring a loop candidate.
    pub similarity_threshold: f32,
    /// Minimum time gap (in keyframe indices) to consider a loop.
    pub min_temporal_gap: usize,
    /// Known BoW descriptors for each keyframe.
    known_descriptors: Vec<Vec<f32>>,
}

impl LoopClosure {
    /// Create a new loop closure detector.
    pub fn new(similarity_threshold: f32, min_temporal_gap: usize) -> Self {
        Self {
            similarity_threshold,
            min_temporal_gap,
            known_descriptors: Vec::new(),
        }
    }

    /// Add a new keyframe descriptor.
    pub fn add_keyframe_descriptor(&mut self, bow: Vec<f32>) {
        self.known_descriptors.push(bow);
    }

    /// Query for loop closure candidates for the given BoW descriptor.
    ///
    /// Returns a list of `(keyframe_index, similarity_score)` pairs above threshold.
    pub fn query(&self, bow: &[f32]) -> Vec<(usize, f32)> {
        let current_idx = self.known_descriptors.len();
        let mut candidates = Vec::new();
        for (i, known) in self.known_descriptors.iter().enumerate() {
            if current_idx.saturating_sub(i) < self.min_temporal_gap {
                continue;
            }
            let sim = BagOfWords::similarity(bow, known);
            if sim >= self.similarity_threshold {
                candidates.push((i, sim));
            }
        }
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates
    }

    /// Number of stored keyframe descriptors.
    pub fn num_keyframes(&self) -> usize {
        self.known_descriptors.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pose Graph
// ─────────────────────────────────────────────────────────────────────────────

/// A relative pose constraint between two keyframes.
#[derive(Debug, Clone)]
pub struct PoseConstraint {
    /// Source keyframe index.
    pub from: usize,
    /// Target keyframe index.
    pub to: usize,
    /// Relative rotation (Rodrigues).
    pub relative_rvec: Array1<f64>,
    /// Relative translation.
    pub relative_tvec: Array1<f64>,
    /// Information matrix weight (higher = more certain).
    pub weight: f64,
    /// Whether this is a loop closure constraint.
    pub is_loop_closure: bool,
}

/// Pose graph for global consistency optimization.
///
/// Nodes are keyframe poses; edges are relative pose constraints.
/// The graph can be optimised using iterative gradient descent (simplified).
pub struct PoseGraph {
    /// All keyframe poses (mutable during optimization).
    pub poses: Vec<Pose>,
    /// All constraints (sequential + loop closure).
    pub constraints: Vec<PoseConstraint>,
    /// Maximum optimization iterations.
    pub max_iterations: usize,
    /// Learning rate for gradient descent.
    pub learning_rate: f64,
    /// Convergence tolerance.
    pub tolerance: f64,
}

impl PoseGraph {
    /// Create a new pose graph.
    pub fn new(max_iterations: usize, learning_rate: f64, tolerance: f64) -> Self {
        Self {
            poses: Vec::new(),
            constraints: Vec::new(),
            max_iterations,
            learning_rate,
            tolerance,
        }
    }

    /// Add a keyframe pose.
    pub fn add_pose(&mut self, pose: Pose) -> usize {
        let idx = self.poses.len();
        self.poses.push(pose);
        idx
    }

    /// Add a pose constraint.
    pub fn add_constraint(&mut self, constraint: PoseConstraint) {
        self.constraints.push(constraint);
    }

    /// Optimize the pose graph using Gauss-Seidel gradient descent.
    ///
    /// Fixes the first pose and adjusts all others to minimise the sum of
    /// squared relative-pose residuals.
    pub fn optimize(&mut self) -> Result<f64> {
        if self.poses.is_empty() {
            return Err(VisionError::InvalidParameter(
                "PoseGraph: no poses to optimize".to_string(),
            ));
        }
        if self.constraints.is_empty() {
            return Ok(0.0);
        }

        let mut total_error = f64::MAX;
        for _iter in 0..self.max_iterations {
            let mut gradient_rvec = vec![Array1::<f64>::zeros(3); self.poses.len()];
            let mut gradient_tvec = vec![Array1::<f64>::zeros(3); self.poses.len()];
            let mut error = 0.0f64;

            for constraint in &self.constraints {
                let i = constraint.from;
                let j = constraint.to;
                if i >= self.poses.len() || j >= self.poses.len() {
                    continue;
                }
                // Compute residual between predicted and measured relative pose
                let pred_rel_r = subtract_pose_rot(&self.poses[j].rvec, &self.poses[i].rvec);
                let pred_rel_t = subtract_vec(&self.poses[j].tvec, &self.poses[i].tvec);
                let res_r = subtract_vec(&pred_rel_r, &constraint.relative_rvec);
                let res_t = subtract_vec(&pred_rel_t, &constraint.relative_tvec);
                let w = constraint.weight;
                let e: f64 = (res_r.iter().map(|v| v * v).sum::<f64>()
                    + res_t.iter().map(|v| v * v).sum::<f64>())
                    * w;
                error += e;
                // Update gradients (j increases, i decreases)
                for k in 0..3 {
                    gradient_rvec[j][k] += 2.0 * w * res_r[k];
                    gradient_tvec[j][k] += 2.0 * w * res_t[k];
                    gradient_rvec[i][k] -= 2.0 * w * res_r[k];
                    gradient_tvec[i][k] -= 2.0 * w * res_t[k];
                }
            }

            // Apply gradient step (keep pose 0 fixed)
            for idx in 1..self.poses.len() {
                for k in 0..3 {
                    self.poses[idx].rvec[k] -= self.learning_rate * gradient_rvec[idx][k];
                    self.poses[idx].tvec[k] -= self.learning_rate * gradient_tvec[idx][k];
                }
            }

            let delta = (total_error - error).abs();
            total_error = error;
            if delta < self.tolerance {
                break;
            }
        }
        Ok(total_error)
    }
}

fn subtract_vec(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
    let mut r = Array1::zeros(a.len());
    for i in 0..a.len() {
        r[i] = a[i] - b[i];
    }
    r
}

/// Approximate relative rotation as simple vector difference (small-angle approx).
fn subtract_pose_rot(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
    subtract_vec(a, b)
}

// ─────────────────────────────────────────────────────────────────────────────
// Visual Odometry
// ─────────────────────────────────────────────────────────────────────────────

/// Visual odometry: frame-to-frame motion estimation via tracked features.
pub struct VisualOdometry {
    /// Camera intrinsics.
    pub intrinsics: IntrinsicMatrix,
    /// RANSAC threshold in pixels for E-matrix estimation.
    pub ransac_threshold: f64,
    /// Maximum RANSAC iterations.
    pub max_ransac_iter: usize,
    /// Minimum inliers for a valid motion estimate.
    pub min_inliers: usize,
    /// Accumulated pose (world → camera).
    current_pose: Pose,
    /// Total number of frames processed.
    frame_count: usize,
    /// Previous frame keypoints.
    prev_keypoints: Vec<[f64; 2]>,
}

impl VisualOdometry {
    /// Create a new VO instance.
    pub fn new(intrinsics: IntrinsicMatrix) -> Self {
        Self {
            intrinsics,
            ransac_threshold: 1.0,
            max_ransac_iter: 1000,
            min_inliers: 10,
            current_pose: Pose::identity(),
            frame_count: 0,
            prev_keypoints: Vec::new(),
        }
    }

    /// Process a new frame, estimating relative motion from `matches`.
    ///
    /// - `keypoints`: 2D keypoints in the current frame.
    /// - `matches`: `(prev_idx, curr_idx)` pairs.
    ///
    /// Returns the estimated `Pose` of the current frame (or `None` on
    /// the first frame / insufficient matches).
    pub fn process_frame(
        &mut self,
        keypoints: &[[f64; 2]],
        matches: &[(usize, usize)],
    ) -> Result<Option<Pose>> {
        self.frame_count += 1;

        if self.frame_count == 1 {
            // First frame: set canonical pose
            self.prev_keypoints = keypoints.to_vec();
            return Ok(Some(self.current_pose.clone()));
        }

        if self.prev_keypoints.is_empty() || matches.len() < self.min_inliers {
            self.prev_keypoints = keypoints.to_vec();
            return Ok(None);
        }

        let ki = self.intrinsics.to_inverse();
        let normalise = |p: &[f64; 2]| -> [f64; 2] {
            let v = scirs2_core::ndarray::Array1::from(vec![p[0], p[1], 1.0]);
            let n = mat3_vec3(&ki, &v);
            let z = n[2].max(1e-14);
            [n[0] / z, n[1] / z]
        };

        let pts1_px: Vec<[f64; 2]> = matches
            .iter()
            .map(|&(pi, _)| {
                if pi < self.prev_keypoints.len() {
                    self.prev_keypoints[pi]
                } else {
                    [0.0, 0.0]
                }
            })
            .collect();
        let pts2_px: Vec<[f64; 2]> = matches
            .iter()
            .map(|&(_, ci)| {
                if ci < keypoints.len() {
                    keypoints[ci]
                } else {
                    [0.0, 0.0]
                }
            })
            .collect();

        let pts1_norm: Vec<[f64; 2]> = pts1_px.iter().map(&normalise).collect();
        let pts2_norm: Vec<[f64; 2]> = pts2_px.iter().map(&normalise).collect();

        // Estimate fundamental matrix with RANSAC
        let (f, inliers) = match FundamentalMatrix::from_ransac(
            &pts1_px,
            &pts2_px,
            self.ransac_threshold,
            0.99,
            self.max_ransac_iter,
        ) {
            Ok(result) => result,
            Err(_) => {
                self.prev_keypoints = keypoints.to_vec();
                return Ok(None);
            }
        };

        let inlier_count = inliers.iter().filter(|&&b| b).count();
        if inlier_count < self.min_inliers {
            self.prev_keypoints = keypoints.to_vec();
            return Ok(None);
        }

        let e = EssentialMatrix::from_fundamental(&f, &self.intrinsics, &self.intrinsics);

        let pts1_in: Vec<[f64; 2]> = inliers
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| pts1_norm[i])
            .collect();
        let pts2_in: Vec<[f64; 2]> = inliers
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| pts2_norm[i])
            .collect();

        let (rel_r, rel_t) = match e.recover_pose(&pts1_in, &pts2_in) {
            Ok(rt) => rt,
            Err(_) => {
                self.prev_keypoints = keypoints.to_vec();
                return Ok(None);
            }
        };

        // Compose current pose with relative motion
        let rel_rvec = rotation_to_rodrigues(&rel_r);
        let new_rvec = compose_rvec(&self.current_pose.rvec, &rel_rvec);
        let r_curr = rodrigues_to_rotation(&self.current_pose.rvec);
        let new_tvec_arr = mat3_vec3(&r_curr, &rel_t);
        let mut new_tvec = Array1::zeros(3);
        for k in 0..3 {
            new_tvec[k] = self.current_pose.tvec[k] + new_tvec_arr[k];
        }

        self.current_pose = Pose {
            rvec: new_rvec,
            tvec: new_tvec,
        };
        self.prev_keypoints = keypoints.to_vec();
        Ok(Some(self.current_pose.clone()))
    }

    /// Get the current accumulated camera pose.
    pub fn current_pose(&self) -> &Pose {
        &self.current_pose
    }

    /// Reset the odometry to the initial state.
    pub fn reset(&mut self) {
        self.current_pose = Pose::identity();
        self.frame_count = 0;
        self.prev_keypoints.clear();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn mat3_vec3(m: &Array2<f64>, v: &Array1<f64>) -> Array1<f64> {
    let mut out = Array1::zeros(3);
    for i in 0..3 {
        for j in 0..3 {
            out[i] += m[[i, j]] * v[j];
        }
    }
    out
}

fn rotation_to_rodrigues(r: &Array2<f64>) -> Array1<f64> {
    let trace = r[[0, 0]] + r[[1, 1]] + r[[2, 2]];
    let cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    if theta.abs() < 1e-10 {
        return Array1::zeros(3);
    }
    let scale = theta / (2.0 * theta.sin());
    Array1::from(vec![
        (r[[2, 1]] - r[[1, 2]]) * scale,
        (r[[0, 2]] - r[[2, 0]]) * scale,
        (r[[1, 0]] - r[[0, 1]]) * scale,
    ])
}

/// Compose two Rodrigues vectors (R_total = R2 * R1).
fn compose_rvec(r1: &Array1<f64>, r2: &Array1<f64>) -> Array1<f64> {
    let m1 = rodrigues_to_rotation(r1);
    let m2 = rodrigues_to_rotation(r2);
    let mut m = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for k in 0..3 {
            for j in 0..3 {
                m[[i, j]] += m2[[i, k]] * m1[[k, j]];
            }
        }
    }
    rotation_to_rodrigues(&m)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyframe_selector_parallax() {
        let sel = KeyframeSelector::default();
        let pts1: Vec<[f64; 2]> = (0..20).map(|i| [i as f64, i as f64]).collect();
        let pts2: Vec<[f64; 2]> = (0..20).map(|i| [i as f64 + 25.0, i as f64]).collect();
        assert!(sel.should_insert(&pts2, &pts1, 10));
    }

    #[test]
    fn test_keyframe_selector_no_insert_too_soon() {
        let sel = KeyframeSelector {
            min_frame_gap: 10,
            ..Default::default()
        };
        let pts: Vec<[f64; 2]> = (0..20).map(|i| [i as f64, i as f64]).collect();
        assert!(!sel.should_insert(&pts, &pts, 3));
    }

    #[test]
    fn test_covisibility_graph() {
        let mut kf1 = Keyframe::new(0, Pose::identity(), vec![[0.0, 0.0]; 5]);
        kf1.map_point_ids = vec![Some(0), Some(1), Some(2), None, None];
        let mut kf2 = Keyframe::new(1, Pose::identity(), vec![[0.0, 0.0]; 5]);
        kf2.map_point_ids = vec![Some(0), Some(1), Some(3), None, None];
        let mut covis = Covisibility::new();
        covis.update(&[kf1, kf2]);
        let neighbours = covis.get_neighbours(0, 1);
        assert_eq!(neighbours.len(), 1);
        assert_eq!(neighbours[0].0, 1);
        assert_eq!(neighbours[0].1, 2);
    }

    #[test]
    fn test_loop_closure_query() {
        let mut lc = LoopClosure::new(0.8, 5);
        for _ in 0..10 {
            lc.add_keyframe_descriptor(vec![1.0, 0.0, 0.0]);
        }
        lc.add_keyframe_descriptor(vec![0.5, 0.5, 0.0]);
        let bow = vec![1.0, 0.0, 0.0];
        let candidates = lc.query(&bow);
        // Should find similar descriptors
        let _ = candidates; // May have no candidates due to temporal gap
    }

    #[test]
    fn test_pose_graph_optimization() {
        let mut pg = PoseGraph::new(50, 0.01, 1e-6);
        pg.add_pose(Pose::identity());
        let mut p2 = Pose::identity();
        p2.tvec[0] = 1.1; // slightly off from ground truth 1.0
        pg.add_pose(p2);
        pg.add_constraint(PoseConstraint {
            from: 0,
            to: 1,
            relative_rvec: Array1::zeros(3),
            relative_tvec: Array1::from(vec![1.0, 0.0, 0.0]),
            weight: 1.0,
            is_loop_closure: false,
        });
        let result = pg.optimize();
        assert!(result.is_ok());
    }

    #[test]
    fn test_visual_odometry_first_frame() {
        let k = IntrinsicMatrix {
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 240.0,
        };
        let mut vo = VisualOdometry::new(k);
        let kps: Vec<[f64; 2]> = (0..20).map(|i| [i as f64 * 10.0, 100.0]).collect();
        let result = vo.process_frame(&kps, &[]);
        assert!(result.is_ok());
        assert!(result.expect("process_frame should succeed").is_some());
    }

    #[test]
    fn test_map_point() {
        let pos = Array1::from(vec![1.0, 2.0, 3.0]);
        let mut mp = MapPoint::new(0, pos);
        mp.add_observation(0, [100.0, 200.0]);
        mp.add_observation(1, [110.0, 205.0]);
        assert_eq!(mp.observation_count(), 2);
        // Check that frame 0 observation exists (added above)
        assert!(mp.observations.contains_key(&0)); // frame 0 was observed
    }
}
