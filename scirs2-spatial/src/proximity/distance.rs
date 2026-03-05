//! Distance computations between geometric objects
//!
//! Hausdorff distance between point sets and Frechet distance between curves.

use crate::error::{SpatialError, SpatialResult};

/// Result of a Hausdorff distance computation with additional detail
#[derive(Debug, Clone)]
pub struct HausdorffResult {
    /// The (symmetric) Hausdorff distance
    pub distance: f64,
    /// The directed Hausdorff distance from set A to set B
    pub forward_distance: f64,
    /// The directed Hausdorff distance from set B to set A
    pub backward_distance: f64,
    /// Index in set A that realizes the forward distance
    pub forward_index_a: usize,
    /// Index in set B closest to the forward-realizing point
    pub forward_index_b: usize,
    /// Index in set B that realizes the backward distance
    pub backward_index_b: usize,
    /// Index in set A closest to the backward-realizing point
    pub backward_index_a: usize,
}

/// Compute the Hausdorff distance between two point sets with detailed results
///
/// The Hausdorff distance measures the greatest of all the distances from a point
/// in one set to the closest point in the other set. This function returns both
/// directed distances and the indices of the realizing points.
///
/// # Arguments
///
/// * `set_a` - First point set as slices of [x, y] coordinates
/// * `set_b` - Second point set as slices of [x, y] coordinates
///
/// # Returns
///
/// * `HausdorffResult` with distances and indices
///
/// # Examples
///
/// ```
/// use scirs2_spatial::proximity::hausdorff_distance_detailed;
///
/// let set_a = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
/// let set_b = vec![[0.5, 0.5], [1.5, 0.5]];
///
/// let result = hausdorff_distance_detailed(&set_a, &set_b).expect("compute");
/// assert!(result.distance > 0.0);
/// ```
pub fn hausdorff_distance_detailed(
    set_a: &[[f64; 2]],
    set_b: &[[f64; 2]],
) -> SpatialResult<HausdorffResult> {
    if set_a.is_empty() || set_b.is_empty() {
        return Err(SpatialError::ValueError(
            "Both point sets must be non-empty".to_string(),
        ));
    }

    // Directed Hausdorff from A to B
    let (fwd_dist, fwd_ia, fwd_ib) = directed_hausdorff_impl(set_a, set_b);

    // Directed Hausdorff from B to A
    let (bwd_dist, bwd_ib, bwd_ia) = directed_hausdorff_impl(set_b, set_a);

    Ok(HausdorffResult {
        distance: fwd_dist.max(bwd_dist),
        forward_distance: fwd_dist,
        backward_distance: bwd_dist,
        forward_index_a: fwd_ia,
        forward_index_b: fwd_ib,
        backward_index_b: bwd_ib,
        backward_index_a: bwd_ia,
    })
}

/// Compute directed Hausdorff distance from set_a to set_b
/// Returns (distance, index_in_a, index_in_b)
fn directed_hausdorff_impl(set_a: &[[f64; 2]], set_b: &[[f64; 2]]) -> (f64, usize, usize) {
    let mut max_min_dist = f64::NEG_INFINITY;
    let mut best_a = 0;
    let mut best_b = 0;

    for (i, pa) in set_a.iter().enumerate() {
        let mut min_dist = f64::INFINITY;
        let mut closest_b = 0;

        for (j, pb) in set_b.iter().enumerate() {
            let dx = pa[0] - pb[0];
            let dy = pa[1] - pb[1];
            let d = (dx * dx + dy * dy).sqrt();
            if d < min_dist {
                min_dist = d;
                closest_b = j;
            }
        }

        if min_dist > max_min_dist {
            max_min_dist = min_dist;
            best_a = i;
            best_b = closest_b;
        }
    }

    (max_min_dist, best_a, best_b)
}

/// Compute the discrete Frechet distance between two curves
///
/// The Frechet distance is a measure of similarity between curves that takes
/// into account the location and ordering of the points along the curves.
/// This function computes the discrete approximation using the dynamic
/// programming algorithm of Eiter and Mannila (1994).
///
/// The Frechet distance can be intuitively understood as: a person walks along
/// curve P and a dog walks along curve Q, both starting at their respective
/// beginnings and ending at their respective ends. The Frechet distance is
/// the minimum leash length needed for them to complete the walk.
///
/// # Arguments
///
/// * `curve_p` - First curve as a sequence of [x, y] points
/// * `curve_q` - Second curve as a sequence of [x, y] points
///
/// # Returns
///
/// * The discrete Frechet distance
///
/// # Examples
///
/// ```
/// use scirs2_spatial::proximity::discrete_frechet_distance;
///
/// let curve_p = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
/// let curve_q = vec![[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]];
///
/// let dist = discrete_frechet_distance(&curve_p, &curve_q).expect("compute");
/// // Curves are parallel, distance 1 apart
/// assert!((dist - 1.0).abs() < 1e-10);
/// ```
pub fn discrete_frechet_distance(curve_p: &[[f64; 2]], curve_q: &[[f64; 2]]) -> SpatialResult<f64> {
    let n = curve_p.len();
    let m = curve_q.len();

    if n == 0 || m == 0 {
        return Err(SpatialError::ValueError(
            "Both curves must have at least one point".to_string(),
        ));
    }

    // dp[i][j] = discrete Frechet distance between P[0..=i] and Q[0..=j]
    let mut dp = vec![vec![f64::NEG_INFINITY; m]; n];

    // Euclidean distance between two 2D points
    let dist = |p: &[f64; 2], q: &[f64; 2]| -> f64 {
        let dx = p[0] - q[0];
        let dy = p[1] - q[1];
        (dx * dx + dy * dy).sqrt()
    };

    // Base case
    dp[0][0] = dist(&curve_p[0], &curve_q[0]);

    // First column
    for i in 1..n {
        dp[i][0] = dp[i - 1][0].max(dist(&curve_p[i], &curve_q[0]));
    }

    // First row
    for j in 1..m {
        dp[0][j] = dp[0][j - 1].max(dist(&curve_p[0], &curve_q[j]));
    }

    // Fill the DP table
    for i in 1..n {
        for j in 1..m {
            let d = dist(&curve_p[i], &curve_q[j]);
            let prev_min = dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
            dp[i][j] = d.max(prev_min);
        }
    }

    Ok(dp[n - 1][m - 1])
}

/// Compute the discrete Frechet distance between two curves in N-dimensions
///
/// Generalized version supporting arbitrary-dimensional points.
///
/// # Arguments
///
/// * `curve_p` - First curve as a sequence of points (each point is a Vec<f64>)
/// * `curve_q` - Second curve as a sequence of points
///
/// # Returns
///
/// * The discrete Frechet distance
pub fn discrete_frechet_distance_nd(
    curve_p: &[Vec<f64>],
    curve_q: &[Vec<f64>],
) -> SpatialResult<f64> {
    let n = curve_p.len();
    let m = curve_q.len();

    if n == 0 || m == 0 {
        return Err(SpatialError::ValueError(
            "Both curves must have at least one point".to_string(),
        ));
    }

    // Verify consistent dimensions
    let ndim = curve_p[0].len();
    for (i, p) in curve_p.iter().enumerate() {
        if p.len() != ndim {
            return Err(SpatialError::DimensionError(format!(
                "Point {} in curve_p has {} dimensions, expected {}",
                i,
                p.len(),
                ndim
            )));
        }
    }
    for (j, q) in curve_q.iter().enumerate() {
        if q.len() != ndim {
            return Err(SpatialError::DimensionError(format!(
                "Point {} in curve_q has {} dimensions, expected {}",
                j,
                q.len(),
                ndim
            )));
        }
    }

    let dist = |p: &Vec<f64>, q: &Vec<f64>| -> f64 {
        let mut sum = 0.0;
        for d in 0..ndim {
            let diff = p[d] - q[d];
            sum += diff * diff;
        }
        sum.sqrt()
    };

    let mut dp = vec![vec![f64::NEG_INFINITY; m]; n];

    dp[0][0] = dist(&curve_p[0], &curve_q[0]);

    for i in 1..n {
        dp[i][0] = dp[i - 1][0].max(dist(&curve_p[i], &curve_q[0]));
    }

    for j in 1..m {
        dp[0][j] = dp[0][j - 1].max(dist(&curve_p[0], &curve_q[j]));
    }

    for i in 1..n {
        for j in 1..m {
            let d = dist(&curve_p[i], &curve_q[j]);
            let prev_min = dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
            dp[i][j] = d.max(prev_min);
        }
    }

    Ok(dp[n - 1][m - 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hausdorff_identical_sets() {
        let set = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let result = hausdorff_distance_detailed(&set, &set).expect("compute");
        assert!((result.distance).abs() < 1e-10);
    }

    #[test]
    fn test_hausdorff_simple() {
        let set_a = vec![[0.0, 0.0], [1.0, 0.0]];
        let set_b = vec![[0.0, 1.0], [1.0, 1.0]];

        let result = hausdorff_distance_detailed(&set_a, &set_b).expect("compute");
        assert!((result.distance - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hausdorff_asymmetric() {
        let set_a = vec![[0.0, 0.0]];
        let set_b = vec![[1.0, 0.0], [2.0, 0.0]];

        let result = hausdorff_distance_detailed(&set_a, &set_b).expect("compute");

        // Forward: max min dist from A to B = 1.0 (point (0,0) to (1,0))
        assert!((result.forward_distance - 1.0).abs() < 1e-10);

        // Backward: max min dist from B to A = 2.0 (point (2,0) to (0,0))
        assert!((result.backward_distance - 2.0).abs() < 1e-10);

        // Hausdorff = max(1, 2) = 2
        assert!((result.distance - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_hausdorff_empty() {
        let set_a: Vec<[f64; 2]> = vec![];
        let set_b = vec![[0.0, 0.0]];
        assert!(hausdorff_distance_detailed(&set_a, &set_b).is_err());
    }

    #[test]
    fn test_frechet_parallel_curves() {
        let curve_p = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let curve_q = vec![[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]];

        let dist = discrete_frechet_distance(&curve_p, &curve_q).expect("compute");
        assert!((dist - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_frechet_identical_curves() {
        let curve = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];
        let dist = discrete_frechet_distance(&curve, &curve).expect("compute");
        assert!(dist.abs() < 1e-10);
    }

    #[test]
    fn test_frechet_single_point() {
        let curve_p = vec![[0.0, 0.0]];
        let curve_q = vec![[3.0, 4.0]];

        let dist = discrete_frechet_distance(&curve_p, &curve_q).expect("compute");
        assert!((dist - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_frechet_ordering_matters() {
        // Same points but reversed order should give different Frechet distance
        // than unordered Hausdorff
        let curve_p = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let curve_q = vec![[2.0, 1.0], [1.0, 1.0], [0.0, 1.0]]; // reversed

        let dist = discrete_frechet_distance(&curve_p, &curve_q).expect("compute");
        // Person starts at (0,0), dog starts at (2,1) -> distance ~2.24
        // They must traverse monotonically, so the leash must reach from
        // (0,0) to (2,1) at the start
        assert!(dist > 1.0); // Must be more than the parallel distance of 1
    }

    #[test]
    fn test_frechet_empty() {
        let empty: Vec<[f64; 2]> = vec![];
        let curve = vec![[0.0, 0.0]];
        assert!(discrete_frechet_distance(&empty, &curve).is_err());
    }

    #[test]
    fn test_frechet_nd() {
        let curve_p = vec![vec![0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]];
        let curve_q = vec![vec![0.0, 0.0, 1.0], vec![1.0, 0.0, 1.0]];

        let dist = discrete_frechet_distance_nd(&curve_p, &curve_q).expect("compute");
        assert!((dist - 1.0).abs() < 1e-10);
    }
}
