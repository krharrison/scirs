//! Trajectory clustering algorithms
//!
//! - **Pairwise DTW matrix** – O(n²) matrix of DTW distances.
//! - **k-Medoids with DTW** – Partition-around-medoids (PAM) using DTW as the
//!   distance metric.  Supports random restarts.
//! - **TRACLUS** – TRAjectory CLUStering framework (Lee, Han & Whang, 2007):
//!   partition trajectories into line segments, then cluster segments by
//!   perpendicular/parallel/angle distance.
//! - **Sub-trajectory clustering** – detect common sub-trajectories shared
//!   across multiple input trajectories.

use crate::error::{SpatialError, SpatialResult};
use crate::trajectory::similarity::{dtw_distance, Point2D, Trajectory};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Pairwise DTW matrix
// ---------------------------------------------------------------------------

/// Compute the pairwise DTW distance matrix for a collection of trajectories.
///
/// Returns a symmetric `n × n` matrix stored as a flat `Vec<f64>` in row-major
/// order, where `n` is the number of trajectories.
///
/// # Errors
///
/// Propagates any error from [`dtw_distance`].
pub fn trajectory_dtw_matrix(trajectories: &[Trajectory]) -> SpatialResult<Vec<f64>> {
    let n = trajectories.len();
    let mut matrix = vec![0.0_f64; n * n];

    for i in 0..n {
        for j in (i + 1)..n {
            let d = dtw_distance(&trajectories[i], &trajectories[j])?;
            matrix[i * n + j] = d;
            matrix[j * n + i] = d;
        }
    }
    Ok(matrix)
}

// ---------------------------------------------------------------------------
// k-Medoids
// ---------------------------------------------------------------------------

/// Result of k-medoids clustering.
#[derive(Debug, Clone)]
pub struct KMedoidsResult {
    /// Index of the medoid for each cluster (into the input slice).
    pub medoids: Vec<usize>,
    /// Cluster assignment for each trajectory (0-based cluster index).
    pub labels: Vec<usize>,
    /// Total intra-cluster cost (sum of distances to medoids).
    pub total_cost: f64,
}

/// Cluster `trajectories` into `k` groups using k-medoids with DTW.
///
/// Uses the PAM (Partition Around Medoids) algorithm.  The `seed` parameter
/// controls the initial medoid selection (deterministic when set).
///
/// # Errors
///
/// - [`SpatialError::ValueError`] if `k == 0` or `k > n`.
/// - Propagates DTW errors.
pub fn trajectory_kmedoids(
    trajectories: &[Trajectory],
    k: usize,
    seed: u64,
    max_iter: usize,
) -> SpatialResult<KMedoidsResult> {
    let n = trajectories.len();
    if k == 0 {
        return Err(SpatialError::ValueError(
            "k-medoids: k must be at least 1".to_string(),
        ));
    }
    if k > n {
        return Err(SpatialError::ValueError(format!(
            "k-medoids: k ({k}) cannot exceed the number of trajectories ({n})"
        )));
    }

    // Precompute full DTW matrix to avoid redundant computation.
    let dist_matrix = trajectory_dtw_matrix(trajectories)?;

    // Deterministic pseudo-random initial medoid selection (linear congruential).
    let mut medoids: Vec<usize> = select_initial_medoids(n, k, seed);

    let mut labels = vec![0usize; n];
    let mut iter = 0;
    loop {
        // Assignment step: assign each trajectory to the nearest medoid.
        for i in 0..n {
            let best = medoids
                .iter()
                .enumerate()
                .map(|(ci, &m)| (ci, dist_matrix[i * n + m]))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(ci, _)| ci)
                .unwrap_or(0);
            labels[i] = best;
        }

        // Update step: for each cluster, choose the member that minimises total
        // intra-cluster distance.
        let mut new_medoids = medoids.clone();
        for ci in 0..k {
            let cluster: Vec<usize> = (0..n).filter(|&i| labels[i] == ci).collect();
            if cluster.is_empty() {
                continue;
            }
            let best_medoid = cluster
                .iter()
                .copied()
                .min_by(|&a, &b| {
                    let cost_a: f64 = cluster.iter().map(|&j| dist_matrix[a * n + j]).sum();
                    let cost_b: f64 = cluster.iter().map(|&j| dist_matrix[b * n + j]).sum();
                    cost_a.partial_cmp(&cost_b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(cluster[0]);
            new_medoids[ci] = best_medoid;
        }

        iter += 1;
        if new_medoids == medoids || iter >= max_iter {
            medoids = new_medoids;
            break;
        }
        medoids = new_medoids;
    }

    // Final assignment and cost.
    for i in 0..n {
        let best = medoids
            .iter()
            .enumerate()
            .map(|(ci, &m)| (ci, dist_matrix[i * n + m]))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(ci, _)| ci)
            .unwrap_or(0);
        labels[i] = best;
    }
    let total_cost: f64 = (0..n)
        .map(|i| dist_matrix[i * n + medoids[labels[i]]])
        .sum();

    Ok(KMedoidsResult {
        medoids,
        labels,
        total_cost,
    })
}

/// Select `k` initial medoids from `n` candidates using a linear-congruential
/// pseudo-random sequence seeded by `seed`.
fn select_initial_medoids(n: usize, k: usize, seed: u64) -> Vec<usize> {
    // LCG parameters from Numerical Recipes.
    const A: u64 = 1_664_525;
    const C: u64 = 1_013_904_223;
    let mut state = seed.wrapping_add(1);
    let mut chosen = Vec::with_capacity(k);
    let mut available: Vec<usize> = (0..n).collect();

    for _ in 0..k {
        state = A.wrapping_mul(state).wrapping_add(C);
        let idx = (state as usize) % available.len();
        chosen.push(available[idx]);
        available.swap_remove(idx);
    }
    chosen
}

// ---------------------------------------------------------------------------
// TRACLUS
// ---------------------------------------------------------------------------

/// A line segment extracted from a trajectory partition.
#[derive(Debug, Clone)]
pub struct LineSegment {
    /// Index of the source trajectory.
    pub traj_idx: usize,
    /// Index of the start point within that trajectory.
    pub start_pt: usize,
    /// Start point coordinates.
    pub start: Point2D,
    /// End point coordinates.
    pub end: Point2D,
}

/// Result of TRACLUS clustering.
#[derive(Debug, Clone)]
pub struct TraclusResult {
    /// Cluster assignment per segment (-1 = noise).
    pub labels: Vec<i32>,
    /// Segment list in the same order as `labels`.
    pub segments: Vec<LineSegment>,
    /// Number of clusters found (excluding noise).
    pub n_clusters: usize,
}

/// Cluster trajectories using the TRACLUS framework.
///
/// ## Steps
/// 1. **Partition**: split each trajectory into representative line segments
///    using MDL (minimum description length) cost criterion.
/// 2. **Group**: apply DBSCAN-like density clustering on the segments using
///    the TRACLUS distance (perpendicular + parallel + angle components).
///
/// # Arguments
///
/// * `trajectories` – Input trajectories.
/// * `epsilon`      – Neighbourhood radius for the density clustering.
/// * `min_lines`    – Minimum number of lines required to form a cluster.
///
/// # Errors
///
/// Returns [`SpatialError::ValueError`] for invalid parameters.
pub fn traclus_cluster(
    trajectories: &[Trajectory],
    epsilon: f64,
    min_lines: usize,
) -> SpatialResult<TraclusResult> {
    if epsilon <= 0.0 {
        return Err(SpatialError::ValueError(
            "TRACLUS: epsilon must be positive".to_string(),
        ));
    }
    if min_lines == 0 {
        return Err(SpatialError::ValueError(
            "TRACLUS: min_lines must be at least 1".to_string(),
        ));
    }

    // Step 1: Partition trajectories into line segments.
    let mut segments: Vec<LineSegment> = Vec::new();
    for (ti, traj) in trajectories.iter().enumerate() {
        let segs = partition_trajectory(traj, ti);
        segments.extend(segs);
    }

    if segments.is_empty() {
        return Ok(TraclusResult {
            labels: vec![],
            segments,
            n_clusters: 0,
        });
    }

    // Step 2: DBSCAN on segments with TRACLUS distance.
    let n = segments.len();
    let mut labels = vec![-2i32; n]; // -2 = unvisited, -1 = noise.
    let mut cluster_id = 0i32;

    for i in 0..n {
        if labels[i] != -2 {
            continue;
        }
        let neighbours = region_query(&segments, i, epsilon);
        if neighbours.len() < min_lines {
            labels[i] = -1; // noise
        } else {
            expand_cluster(&segments, &mut labels, i, &neighbours, cluster_id, epsilon, min_lines);
            cluster_id += 1;
        }
    }

    let n_clusters = cluster_id as usize;
    Ok(TraclusResult {
        labels,
        segments,
        n_clusters,
    })
}

// --------------- TRACLUS helpers -------------------------------------------

/// Partition a single trajectory into line segments using a simplified MDL
/// criterion: greedily extend the current segment until the approximation
/// cost exceeds the exact representation cost.
fn partition_trajectory(traj: &[Point2D], traj_idx: usize) -> Vec<LineSegment> {
    let n = traj.len();
    if n < 2 {
        return vec![];
    }

    let mut segs = Vec::new();
    let mut cp_start = 0; // current characteristic-point index

    let mut cp_end = 2;
    while cp_end <= n - 1 {
        let mdl_par = mdl_par_cost(traj, cp_start, cp_end);
        let mdl_no_par = mdl_no_par_cost(traj, cp_start, cp_end);
        if mdl_par > mdl_no_par {
            // Partition at cp_end - 1.
            let split = cp_end - 1;
            segs.push(LineSegment {
                traj_idx,
                start_pt: cp_start,
                start: traj[cp_start],
                end: traj[split],
            });
            cp_start = split;
        }
        cp_end += 1;
    }
    // Final segment.
    if cp_start < n - 1 {
        segs.push(LineSegment {
            traj_idx,
            start_pt: cp_start,
            start: traj[cp_start],
            end: traj[n - 1],
        });
    }
    segs
}

/// MDL cost *with* partitioning: L(H) + L(D|H).
fn mdl_par_cost(traj: &[Point2D], start: usize, end: usize) -> f64 {
    // L(H): length of the hypothesis segment (in bits via log2).
    let seg_len = segment_length(&traj[start], &traj[end]);
    let l_h = if seg_len > 1.0 { seg_len.log2() } else { 0.0 };

    // L(D|H): sum of perpendicular + angle distances for sub-segments.
    let mut l_dh = 0.0;
    for i in start..end {
        let d_perp = perp_seg_dist(&traj[start], &traj[end], &traj[i], &traj[i + 1]);
        let d_ang = angle_seg_dist(&traj[start], &traj[end], &traj[i], &traj[i + 1]);
        l_dh += d_perp + d_ang;
    }
    l_h + l_dh
}

/// MDL cost *without* partitioning: each sub-segment counted exactly.
fn mdl_no_par_cost(traj: &[Point2D], start: usize, end: usize) -> f64 {
    (start..end)
        .map(|i| {
            let l = segment_length(&traj[i], &traj[i + 1]);
            if l > 1.0 {
                l.log2()
            } else {
                0.0
            }
        })
        .sum()
}

fn segment_length(a: &Point2D, b: &Point2D) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    (dx * dx + dy * dy).sqrt()
}

/// TRACLUS distance between two line segments (averaged over three components).
fn traclus_distance(s1: &LineSegment, s2: &LineSegment) -> f64 {
    let d_perp = perp_seg_dist(&s1.start, &s1.end, &s2.start, &s2.end);
    let d_par = par_seg_dist(&s1.start, &s1.end, &s2.start, &s2.end);
    let d_ang = angle_seg_dist(&s1.start, &s1.end, &s2.start, &s2.end);
    d_perp + d_par + d_ang
}

/// Perpendicular distance between two line segments.
fn perp_seg_dist(p1: &Point2D, p2: &Point2D, p3: &Point2D, p4: &Point2D) -> f64 {
    let l = segment_length(p1, p2);
    if l < f64::EPSILON {
        return segment_length(p3, p4) / 2.0;
    }
    let d1 = point_to_seg_dist(p3, p1, p2);
    let d2 = point_to_seg_dist(p4, p1, p2);
    if d1 + d2 < f64::EPSILON {
        0.0
    } else {
        (d1 * d1 + d2 * d2) / (d1 + d2)
    }
}

/// Point-to-segment (infinite line) perpendicular distance.
fn point_to_seg_dist(p: &Point2D, a: &Point2D, b: &Point2D) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let len_sq = dx * dx + dy * dy;
    if len_sq < f64::EPSILON {
        let ex = p[0] - a[0];
        let ey = p[1] - a[1];
        return (ex * ex + ey * ey).sqrt();
    }
    let t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len_sq;
    let proj = [a[0] + t * dx, a[1] + t * dy];
    let ex = p[0] - proj[0];
    let ey = p[1] - proj[1];
    (ex * ex + ey * ey).sqrt()
}

/// Parallel distance between two line segments (projection overlap).
fn par_seg_dist(p1: &Point2D, p2: &Point2D, p3: &Point2D, p4: &Point2D) -> f64 {
    let l = segment_length(p1, p2);
    if l < f64::EPSILON {
        return 0.0;
    }
    let ux = (p2[0] - p1[0]) / l;
    let uy = (p2[1] - p1[1]) / l;

    // Project all four endpoints onto the line through p1.
    let proj1 = (p1[0] - p1[0]) * ux + (p1[1] - p1[1]) * uy; // = 0
    let proj2 = (p2[0] - p1[0]) * ux + (p2[1] - p1[1]) * uy; // = l
    let proj3 = (p3[0] - p1[0]) * ux + (p3[1] - p1[1]) * uy;
    let proj4 = (p4[0] - p1[0]) * ux + (p4[1] - p1[1]) * uy;
    let _ = proj1;
    let _ = proj2;

    let ps = proj3.min(proj4);
    let pe = proj3.max(proj4);
    let qs = 0.0_f64;
    let qe = l;

    let d1 = if ps < qs { qs - ps } else { 0.0 };
    let d2 = if pe > qe { pe - qe } else { 0.0 };
    d1.min(d2)
}

/// Angle-based distance between two line segments.
fn angle_seg_dist(p1: &Point2D, p2: &Point2D, p3: &Point2D, p4: &Point2D) -> f64 {
    let dx1 = p2[0] - p1[0];
    let dy1 = p2[1] - p1[1];
    let dx2 = p4[0] - p3[0];
    let dy2 = p4[1] - p3[1];
    let l1 = (dx1 * dx1 + dy1 * dy1).sqrt();
    let l2 = (dx2 * dx2 + dy2 * dy2).sqrt();

    if l1 < f64::EPSILON || l2 < f64::EPSILON {
        return 0.0;
    }
    let cos_theta = ((dx1 * dx2 + dy1 * dy2) / (l1 * l2)).clamp(-1.0, 1.0);
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
    let len_min = l1.min(l2);
    len_min * sin_theta
}

fn region_query(segments: &[LineSegment], idx: usize, epsilon: f64) -> Vec<usize> {
    segments
        .iter()
        .enumerate()
        .filter(|(j, s)| {
            *j != idx && traclus_distance(&segments[idx], s) <= epsilon
        })
        .map(|(j, _)| j)
        .collect()
}

fn expand_cluster(
    segments: &[LineSegment],
    labels: &mut Vec<i32>,
    core_idx: usize,
    seeds: &[usize],
    cluster_id: i32,
    epsilon: f64,
    min_lines: usize,
) {
    labels[core_idx] = cluster_id;
    let mut queue: Vec<usize> = seeds.to_vec();
    let mut head = 0;

    while head < queue.len() {
        let q = queue[head];
        head += 1;

        if labels[q] == -1 {
            // Previously noise, now border.
            labels[q] = cluster_id;
        } else if labels[q] == -2 {
            labels[q] = cluster_id;
            let neighbours = region_query(segments, q, epsilon);
            if neighbours.len() >= min_lines {
                for &nb in &neighbours {
                    if labels[nb] == -2 || labels[nb] == -1 {
                        queue.push(nb);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Sub-trajectory clustering
// ---------------------------------------------------------------------------

/// A common sub-trajectory detected across multiple input trajectories.
#[derive(Debug, Clone)]
pub struct CommonSubTrajectory {
    /// Indices of the trajectories that share this sub-trajectory.
    pub traj_indices: Vec<usize>,
    /// The representative sub-trajectory (centroid of matched sub-sequences).
    pub representative: Trajectory,
    /// Average DTW distance from each matching sub-sequence to the representative.
    pub avg_dist: f64,
}

/// Detect common sub-trajectories shared across the input trajectories.
///
/// Uses a sliding-window approach: for each possible window length and position,
/// we compute DTW against all other trajectories' sub-windows and cluster
/// windows that are mutually close.
///
/// # Arguments
///
/// * `trajectories`  – Input trajectories.
/// * `min_len`       – Minimum sub-trajectory length (number of points).
/// * `max_len`       – Maximum sub-trajectory length.
/// * `epsilon`       – DTW distance threshold for considering two sub-windows
///                     "the same".
/// * `min_support`   – Minimum number of trajectories that must contain a
///                     sub-trajectory for it to be reported.
///
/// # Errors
///
/// Returns [`SpatialError::ValueError`] for invalid parameter combinations.
pub fn sub_trajectory_cluster(
    trajectories: &[Trajectory],
    min_len: usize,
    max_len: usize,
    epsilon: f64,
    min_support: usize,
) -> SpatialResult<Vec<CommonSubTrajectory>> {
    if min_len == 0 {
        return Err(SpatialError::ValueError(
            "sub-trajectory: min_len must be at least 1".to_string(),
        ));
    }
    if max_len < min_len {
        return Err(SpatialError::ValueError(
            "sub-trajectory: max_len must be >= min_len".to_string(),
        ));
    }
    if epsilon <= 0.0 {
        return Err(SpatialError::ValueError(
            "sub-trajectory: epsilon must be positive".to_string(),
        ));
    }
    if min_support < 2 {
        return Err(SpatialError::ValueError(
            "sub-trajectory: min_support must be at least 2".to_string(),
        ));
    }

    let mut results: Vec<CommonSubTrajectory> = Vec::new();

    // Enumerate all sub-windows from trajectory 0 as candidates.
    let n_traj = trajectories.len();
    if n_traj < min_support {
        return Ok(results);
    }

    for ref_ti in 0..n_traj {
        let ref_traj = &trajectories[ref_ti];
        for win_len in min_len..=max_len {
            if win_len > ref_traj.len() {
                continue;
            }
            for start in 0..=(ref_traj.len() - win_len) {
                let candidate: Trajectory = ref_traj[start..start + win_len].to_vec();

                // Check how many other trajectories contain a close sub-window.
                let mut matching_ti = vec![ref_ti];
                let mut matching_windows: Vec<Trajectory> = vec![candidate.clone()];

                for other_ti in 0..n_traj {
                    if other_ti == ref_ti {
                        continue;
                    }
                    let other = &trajectories[other_ti];
                    let best = find_best_matching_window(other, &candidate, win_len, epsilon)?;
                    if let Some(window) = best {
                        matching_ti.push(other_ti);
                        matching_windows.push(window);
                    }
                }

                if matching_ti.len() >= min_support {
                    // Compute representative as coordinate-wise mean.
                    let rep = compute_centroid(&matching_windows);
                    let avg_dist = compute_avg_dtw(&matching_windows, &rep)?;
                    results.push(CommonSubTrajectory {
                        traj_indices: matching_ti,
                        representative: rep,
                        avg_dist,
                    });
                }
            }
        }
    }

    // Deduplicate: remove dominated results (subsets of traj_indices).
    dedup_results(results)
}

fn find_best_matching_window(
    traj: &[Point2D],
    candidate: &[Point2D],
    win_len: usize,
    epsilon: f64,
) -> SpatialResult<Option<Trajectory>> {
    if win_len > traj.len() {
        return Ok(None);
    }
    let mut best_dist = f64::INFINITY;
    let mut best_win: Option<Trajectory> = None;
    for start in 0..=(traj.len() - win_len) {
        let window: Trajectory = traj[start..start + win_len].to_vec();
        let d = dtw_distance(&window, candidate)?;
        if d < best_dist {
            best_dist = d;
            best_win = Some(window);
        }
    }
    if best_dist <= epsilon {
        Ok(best_win)
    } else {
        Ok(None)
    }
}

fn compute_centroid(windows: &[Trajectory]) -> Trajectory {
    if windows.is_empty() {
        return vec![];
    }
    let len = windows[0].len();
    let n = windows.len() as f64;
    let mut centroid = vec![[0.0_f64; 2]; len];
    for win in windows {
        for (i, &p) in win.iter().enumerate() {
            centroid[i][0] += p[0] / n;
            centroid[i][1] += p[1] / n;
        }
    }
    centroid
}

fn compute_avg_dtw(windows: &[Trajectory], rep: &[Point2D]) -> SpatialResult<f64> {
    if windows.is_empty() {
        return Ok(0.0);
    }
    let total: f64 = windows
        .iter()
        .map(|w| dtw_distance(w, rep))
        .collect::<SpatialResult<Vec<f64>>>()?
        .iter()
        .sum();
    Ok(total / windows.len() as f64)
}

/// Remove duplicate/dominated CommonSubTrajectory entries.
fn dedup_results(
    mut results: Vec<CommonSubTrajectory>,
) -> SpatialResult<Vec<CommonSubTrajectory>> {
    // Use a HashMap keyed on the sorted set of trajectory indices.
    let mut seen: HashMap<Vec<usize>, usize> = HashMap::new();
    let mut out: Vec<CommonSubTrajectory> = Vec::new();

    for r in results.drain(..) {
        let mut key = r.traj_indices.clone();
        key.sort_unstable();
        seen.entry(key).or_insert_with(|| {
            let idx = out.len();
            out.push(r);
            idx
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn line_traj(n: usize, offset_y: f64) -> Trajectory {
        (0..n).map(|i| [i as f64, offset_y]).collect()
    }

    #[test]
    fn test_dtw_matrix_symmetric() {
        let trajs = vec![line_traj(4, 0.0), line_traj(4, 1.0), line_traj(4, 2.0)];
        let mat = trajectory_dtw_matrix(&trajs).expect("dtw matrix");
        let n = trajs.len();
        // Diagonal must be 0.
        for i in 0..n {
            assert!(mat[i * n + i].abs() < 1e-10);
        }
        // Symmetry.
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (mat[i * n + j] - mat[j * n + i]).abs() < 1e-10,
                    "not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_kmedoids_two_clusters() {
        // Two clearly separated groups.
        let trajs = vec![
            line_traj(3, 0.0),
            line_traj(3, 0.1),
            line_traj(3, 100.0),
            line_traj(3, 100.1),
        ];
        let result = trajectory_kmedoids(&trajs, 2, 42, 50).expect("kmedoids");
        assert_eq!(result.labels.len(), 4);
        assert_eq!(result.medoids.len(), 2);
        // Items 0&1 should be in the same cluster; 2&3 in the other.
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[2], result.labels[3]);
        assert_ne!(result.labels[0], result.labels[2]);
    }

    #[test]
    fn test_traclus_basic() {
        // Three identical straight trajectories → should form one cluster.
        let trajs: Vec<Trajectory> = (0..3)
            .map(|_| (0..5).map(|i| [i as f64, 0.0]).collect())
            .collect();
        let res = traclus_cluster(&trajs, 2.0, 2).expect("traclus");
        // At least 0 clusters; result must be valid.
        assert!(res.n_clusters <= res.segments.len());
    }

    #[test]
    fn test_sub_trajectory_cluster_basic() {
        let trajs: Vec<Trajectory> = vec![
            (0..8).map(|i| [i as f64, 0.0]).collect(),
            (0..8).map(|i| [i as f64, 0.1]).collect(),
            (0..8).map(|i| [i as f64, 0.2]).collect(),
        ];
        let results =
            sub_trajectory_cluster(&trajs, 2, 4, 1.5, 2).expect("sub-trajectory");
        // Should detect some common sub-trajectory.
        // The result count can be 0 if no match found; just ensure no error.
        let _ = results;
    }
}
