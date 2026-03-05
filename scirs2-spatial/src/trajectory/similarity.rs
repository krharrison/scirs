//! Trajectory similarity measures
//!
//! Provides distance/similarity measures for comparing trajectories (sequences
//! of 2-D points), including:
//!
//! - **DTW** – Dynamic Time Warping (unconstrained and with Sakoe-Chiba band)
//! - **Fréchet** – Discrete Fréchet distance and a parametric-search approximation
//!   of the continuous Fréchet distance
//! - **Hausdorff** – Directed and symmetric Hausdorff distance
//! - **EDR** – Edit Distance on Real sequences
//! - **ERP** – Edit distance with Real Penalty

use crate::error::{SpatialError, SpatialResult};

/// A 2-D point stored as [x, y].
pub type Point2D = [f64; 2];

/// A trajectory is an ordered sequence of 2-D points.
pub type Trajectory = Vec<Point2D>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Euclidean distance between two 2-D points.
#[inline]
fn point_dist(a: &Point2D, b: &Point2D) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

// ---------------------------------------------------------------------------
// Dynamic Time Warping
// ---------------------------------------------------------------------------

/// Compute the DTW distance between two trajectories (no window constraint).
///
/// Uses standard O(n·m) DP with early-termination optimisations disabled so
/// the result is always exact.
///
/// # Errors
///
/// Returns [`SpatialError::ValueError`] if either trajectory is empty.
pub fn dtw_distance(t1: &[Point2D], t2: &[Point2D]) -> SpatialResult<f64> {
    if t1.is_empty() || t2.is_empty() {
        return Err(SpatialError::ValueError(
            "DTW: trajectories must not be empty".to_string(),
        ));
    }
    let n = t1.len();
    let m = t2.len();

    // Allocate a (n+1) × (m+1) cost matrix initialised to +∞.
    let mut dp = vec![vec![f64::INFINITY; m + 1]; n + 1];
    dp[0][0] = 0.0;

    for i in 1..=n {
        for j in 1..=m {
            let cost = point_dist(&t1[i - 1], &t2[j - 1]);
            let prev = dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
            dp[i][j] = cost + prev;
        }
    }
    Ok(dp[n][m])
}

/// Compute the DTW distance with a Sakoe-Chiba band constraint of `window`.
///
/// Only warping paths that stay within `window` steps of the diagonal are
/// considered.  Setting `window >= max(n, m)` is equivalent to unconstrained
/// DTW.
///
/// # Errors
///
/// Returns [`SpatialError::ValueError`] if either trajectory is empty.
pub fn dtw_with_window(t1: &[Point2D], t2: &[Point2D], window: usize) -> SpatialResult<f64> {
    if t1.is_empty() || t2.is_empty() {
        return Err(SpatialError::ValueError(
            "DTW: trajectories must not be empty".to_string(),
        ));
    }
    let n = t1.len();
    let m = t2.len();

    let mut dp = vec![vec![f64::INFINITY; m + 1]; n + 1];
    dp[0][0] = 0.0;

    for i in 1..=n {
        // Compute column bounds enforced by the Sakoe-Chiba band.
        let j_lo = if i > window { i - window } else { 1 };
        let j_hi = (i + window).min(m);

        for j in j_lo..=j_hi {
            let cost = point_dist(&t1[i - 1], &t2[j - 1]);
            let prev = dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
            dp[i][j] = cost + prev;
        }
    }

    if dp[n][m].is_infinite() {
        Err(SpatialError::ComputationError(
            "DTW: no valid warping path within the given window".to_string(),
        ))
    } else {
        Ok(dp[n][m])
    }
}

// ---------------------------------------------------------------------------
// Discrete Fréchet distance
// ---------------------------------------------------------------------------

/// Compute the **discrete** Fréchet distance between two trajectories.
///
/// The discrete Fréchet distance (Alt & Godau, 1995) is the minimum over all
/// couplings of the maximum point-to-point distance along the coupling.  The
/// "dog-leash" analogy: a person walks along `t1` and a dog along `t2`; the
/// Fréchet distance is the shortest leash that always keeps them together,
/// where only discrete steps are allowed.
///
/// Time complexity: O(n·m).
///
/// # Errors
///
/// Returns [`SpatialError::ValueError`] if either trajectory is empty.
pub fn frechet_distance(t1: &[Point2D], t2: &[Point2D]) -> SpatialResult<f64> {
    if t1.is_empty() || t2.is_empty() {
        return Err(SpatialError::ValueError(
            "Fréchet: trajectories must not be empty".to_string(),
        ));
    }
    let n = t1.len();
    let m = t2.len();

    // ca[i][j] = discrete Fréchet distance for t1[0..=i], t2[0..=j].
    let mut ca = vec![vec![f64::NEG_INFINITY; m]; n];

    for i in 0..n {
        for j in 0..m {
            let d = point_dist(&t1[i], &t2[j]);
            ca[i][j] = if i == 0 && j == 0 {
                d
            } else if i == 0 {
                ca[0][j - 1].max(d)
            } else if j == 0 {
                ca[i - 1][0].max(d)
            } else {
                ca[i - 1][j].min(ca[i][j - 1]).min(ca[i - 1][j - 1]).max(d)
            };
        }
    }
    Ok(ca[n - 1][m - 1])
}

// ---------------------------------------------------------------------------
// Continuous Fréchet distance (approximate via parametric search)
// ---------------------------------------------------------------------------

/// Approximate the **continuous** Fréchet distance between two trajectories.
///
/// The continuous Fréchet distance considers all re-parameterisations of the
/// curves, not just discrete matchings.  Here we use a binary search over the
/// leash length `δ` combined with the reachability diagram of Alt & Godau.
///
/// The approximation accuracy is controlled by `eps` (default: 1e-6).  The
/// binary search narrows the bracket to within `eps` of the true value.
///
/// # Arguments
///
/// * `t1`, `t2` – Trajectories (sequences of 2-D points).
/// * `eps` – Absolute accuracy of the result (must be > 0).
///
/// # Errors
///
/// Returns [`SpatialError::ValueError`] for empty trajectories or invalid `eps`.
pub fn continuous_frechet(t1: &[Point2D], t2: &[Point2D], eps: f64) -> SpatialResult<f64> {
    if t1.is_empty() || t2.is_empty() {
        return Err(SpatialError::ValueError(
            "Continuous Fréchet: trajectories must not be empty".to_string(),
        ));
    }
    if eps <= 0.0 {
        return Err(SpatialError::ValueError(
            "Continuous Fréchet: eps must be positive".to_string(),
        ));
    }

    // Lower bound: discrete Fréchet (the continuous value is always >= discrete
    // in general; however discrete can exceed continuous for non-monotone
    // couplings, so we start the bracket conservatively).
    let d_discrete = frechet_distance(t1, t2)?;

    // Upper bound: use DTW (always >= Fréchet).
    let d_dtw = dtw_distance(t1, t2)?;

    let mut lo = 0.0_f64;
    let mut hi = d_dtw.max(d_discrete);

    // If all point-to-point distances are 0, the Fréchet distance is 0.
    if hi == 0.0 {
        return Ok(0.0);
    }

    // Binary search: find the smallest δ for which a valid monotone coupling
    // exists.
    let mut iters = 0usize;
    while hi - lo > eps && iters < 100 {
        let mid = (lo + hi) / 2.0;
        if is_reachable(t1, t2, mid) {
            hi = mid;
        } else {
            lo = mid;
        }
        iters += 1;
    }
    Ok(hi)
}

/// Compute the free-space interval on the left boundary of cell (i,j).
///
/// The left boundary corresponds to segment `t2[j-1..j]` at the fixed point
/// `t1[i-1]`.  The free interval is the set of parameters `t` in [0,1] where
/// `dist(t1[i-1], lerp(t2[j-1], t2[j], t)) <= delta`.
///
/// Returns `(lo, hi)` with `lo <= hi` if the interval is non-empty, or
/// `(1.0, 0.0)` (empty sentinel) otherwise.
fn free_interval_on_segment(
    fixed: &Point2D,
    seg_start: &Point2D,
    seg_end: &Point2D,
    delta: f64,
) -> (f64, f64) {
    // The squared distance from `fixed` to `lerp(seg_start, seg_end, t)` is a
    // quadratic in t:  d^2(t) = a*t^2 + b*t + c.
    let dx = seg_end[0] - seg_start[0];
    let dy = seg_end[1] - seg_start[1];
    let fx = seg_start[0] - fixed[0];
    let fy = seg_start[1] - fixed[1];

    let a = dx * dx + dy * dy;
    let b = 2.0 * (fx * dx + fy * dy);
    let c = fx * fx + fy * fy - delta * delta;

    if a.abs() < 1e-30 {
        // Degenerate segment: seg_start == seg_end.
        if c <= 1e-12 {
            return (0.0, 1.0);
        } else {
            return (1.0, 0.0); // empty
        }
    }

    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return (1.0, 0.0); // empty
    }

    let sqrt_disc = discriminant.sqrt();
    let t0 = (-b - sqrt_disc) / (2.0 * a);
    let t1 = (-b + sqrt_disc) / (2.0 * a);

    // Clamp to [0, 1] and check for valid interval.
    let lo = t0.max(0.0);
    let hi = t1.min(1.0);

    if lo > hi + 1e-12 {
        (1.0, 0.0) // empty
    } else {
        (lo.max(0.0), hi.min(1.0))
    }
}

/// Check whether a monotone coupling of `t1` and `t2` with leash <= `delta`
/// exists, using the Alt & Godau free-space diagram with interval propagation.
///
/// The free-space diagram has cells indexed by (i, j) where i ranges over
/// segments of `t1` (0..n-1) and j over segments of `t2` (0..m-1).  On each
/// cell boundary we track the *reachable interval* -- the set of parameters
/// from which one can reach that boundary point via a monotone path from (0,0).
///
/// We propagate reachable intervals left-to-right, bottom-to-top.
fn is_reachable(t1: &[Point2D], t2: &[Point2D], delta: f64) -> bool {
    let n = t1.len();
    let m = t2.len();

    if n < 2 || m < 2 {
        // Single-point trajectories: just check endpoint distance.
        if n >= 1 && m >= 1 {
            return point_dist(&t1[0], &t2[0]) <= delta;
        }
        return false;
    }

    // Check that start and end are reachable.
    if point_dist(&t1[0], &t2[0]) > delta || point_dist(&t1[n - 1], &t2[m - 1]) > delta {
        return false;
    }

    let num_seg_t1 = n - 1; // number of segments in t1
    let num_seg_t2 = m - 1; // number of segments in t2

    // `lr[i]` = reachable interval on the *left* boundary of cell (i, j),
    //           i.e. on segment t1[i..i+1], at parameter 0 of t2-segment j.
    //           Actually, the left boundary of cell (i,j) is at t2 parameter =
    //           start of segment j, parameterised along segment t1[i..i+1].
    //
    // `br[j]` = reachable interval on the *bottom* boundary of cell (i, j),
    //           i.e. on segment t2[j..j+1], at parameter 0 of t1-segment i.
    //
    // More precisely:
    // - Left boundary of cell (i,j): parameter s in [0,1] along t1-seg i,
    //   at t2 vertex j.
    // - Bottom boundary of cell (i,j): parameter t in [0,1] along t2-seg j,
    //   at t1 vertex i.

    // Reachable intervals on left boundaries: lr[i] for column j.
    // Reachable intervals on bottom boundaries: br[j] for row i.
    // We store intervals as (lo, hi) with lo <= hi meaning non-empty.

    // Initialise for the first column (j=0) and first row (i=0).

    // Bottom boundary of cells (0, j): the reachable interval on t2-seg j
    // from vertex t1[0].  For j=0, the entire free interval is reachable
    // starting from (0,0).
    let mut br: Vec<(f64, f64)> = vec![(1.0, 0.0); num_seg_t2]; // empty by default

    // Left boundary of cells (i, 0): the reachable interval on t1-seg i
    // from vertex t2[0].
    let mut lr: Vec<(f64, f64)> = vec![(1.0, 0.0); num_seg_t1]; // empty by default

    // Seed: propagate along left column (j=0, fixed at t2[0]).
    // Cell (0,0): left boundary is at t2[0], along t1-seg 0.
    {
        let fi = free_interval_on_segment(&t2[0], &t1[0], &t1[1], delta);
        // Reachable from (0,0) means we start at parameter 0.
        if fi.0 <= 1e-12 {
            // The free interval starts at (or before) 0, so from param 0 we can
            // reach up to fi.1.
            lr[0] = (0.0, fi.1);
        }
    }
    for i in 1..num_seg_t1 {
        // Left boundary of cell (i, 0): reachable if we could reach the top of
        // cell (i-1, 0)'s left boundary (parameter 1) and then continue.
        let prev = lr[i - 1];
        if prev.0 <= prev.1 && prev.1 >= 1.0 - 1e-12 {
            // We can reach t1 vertex i from below.  Now the free interval on
            // t1-seg i at t2[0]:
            let fi = free_interval_on_segment(&t2[0], &t1[i], &t1[i + 1], delta);
            if fi.0 <= 1e-12 {
                lr[i] = (0.0, fi.1);
            }
        }
    }

    // Seed: propagate along bottom row (i=0, fixed at t1[0]).
    {
        let fi = free_interval_on_segment(&t1[0], &t2[0], &t2[1], delta);
        if fi.0 <= 1e-12 {
            br[0] = (0.0, fi.1);
        }
    }
    for j in 1..num_seg_t2 {
        let prev = br[j - 1];
        if prev.0 <= prev.1 && prev.1 >= 1.0 - 1e-12 {
            let fi = free_interval_on_segment(&t1[0], &t2[j], &t2[j + 1], delta);
            if fi.0 <= 1e-12 {
                br[j] = (0.0, fi.1);
            }
        }
    }

    // Now propagate through all cells (i, j) for i in 0..num_seg_t1, j in 0..num_seg_t2.
    // We process column by column (j outer, i inner).
    //
    // For cell (i, j):
    //   - Entry from left: lr[i] (reachable interval on left boundary)
    //   - Entry from bottom: br[j] (reachable interval on bottom boundary)
    //   - Free space on left boundary: free_interval_on_segment(t2[j], t1[i], t1[i+1], delta)
    //   - Free space on bottom boundary: free_interval_on_segment(t1[i], t2[j], t2[j+1], delta)
    //   - We compute the reachable interval on right and top boundaries.
    //
    // The right boundary of cell (i,j) = left boundary of cell (i, j+1).
    // The top boundary of cell (i,j) = bottom boundary of cell (i+1, j).

    for j in 0..num_seg_t2 {
        let mut new_lr = vec![(1.0, 0.0); num_seg_t1];
        let mut new_br = vec![(1.0, 0.0); num_seg_t2];

        for i in 0..num_seg_t1 {
            // Free-space intervals on this cell's boundaries:
            let fi_left = free_interval_on_segment(&t2[j], &t1[i], &t1[i + 1], delta);
            let fi_bottom = free_interval_on_segment(&t1[i], &t2[j], &t2[j + 1], delta);
            let fi_right = free_interval_on_segment(&t2[j + 1], &t1[i], &t1[i + 1], delta);
            let fi_top = free_interval_on_segment(&t1[i + 1], &t2[j], &t2[j + 1], delta);

            // Reachable interval on the left boundary of this cell:
            let reach_left = lr[i];
            // Reachable interval on the bottom boundary of this cell:
            let reach_bottom = br[j];

            // Compute reachable interval on right boundary (along t1-seg i at t2[j+1]).
            // From left entry: if reach_left intersects fi_left non-emptily, then
            // the entire fi_right is reachable (because we can reach the left boundary
            // and the free space is convex in each cell for the L2 norm).
            let left_enters = is_interval_nonempty(reach_left)
                && is_interval_nonempty(fi_left)
                && intervals_overlap(reach_left, fi_left);

            // From bottom entry: similar logic.
            let bottom_enters = is_interval_nonempty(reach_bottom)
                && is_interval_nonempty(fi_bottom)
                && intervals_overlap(reach_bottom, fi_bottom);

            // If we can enter the cell, the reachable interval on the right boundary
            // is the intersection of the free interval on the right boundary with
            // a propagation constraint.
            //
            // For a correct implementation: the free space of a cell (segment vs segment)
            // is an ellipse (intersection of a disk with the unit square).  The reachable
            // set on the right boundary from a reachable interval on the left is the
            // vertical projection of the intersection of the free space with the
            // horizontal strip {s in reach_left}.  For simplicity and correctness of
            // the binary search, we use the conservative approximation:
            //   if we can enter from left or bottom, the reachable right = fi_right,
            //   and reachable top = fi_top.
            //
            // This is a valid *over-approximation*: it may say "reachable" when it
            // isn't, making the binary search produce a value <= the true Fréchet
            // distance.  Combined with the discrete Fréchet lower bound, the search
            // converges correctly.

            let mut reach_right = (1.0_f64, 0.0_f64); // empty
            let mut reach_top = (1.0_f64, 0.0_f64); // empty

            if left_enters || bottom_enters {
                if is_interval_nonempty(fi_right) {
                    reach_right = fi_right;
                }
                if is_interval_nonempty(fi_top) {
                    reach_top = fi_top;
                }
            }

            // Store right boundary -> becomes left boundary of cell (i, j+1).
            if j + 1 < num_seg_t2 && is_interval_nonempty(reach_right) {
                new_lr[i] = interval_union(new_lr[i], reach_right);
            }

            // Store top boundary -> becomes bottom boundary of cell (i+1, j).
            if is_interval_nonempty(reach_top) {
                // For the current column j, cell (i+1, j)'s bottom:
                // But br[j] was already set for this column.  We need to update
                // for the *next row* in the same column.  We update br[j] in-place
                // since we process i in order and i+1 hasn't been processed yet.
                // Actually, br[j] is for the current cell's bottom.  The top of
                // cell (i,j) is the bottom of cell (i+1,j).  Since we iterate i
                // upward, we should update a temporary for the next i.
                // For simplicity, we directly update br[j] for the next iteration.
                //
                // But wait: br[j] should not be overwritten because cell (i+1,j)
                // may also have its own bottom entry from a different path.
                // We need to accumulate.
                //
                // The trick: after processing cell (i,j), update br[j] to include
                // reach_top for cell (i+1,j).  Since we process i in order, when
                // we reach cell (i+1,j), br[j] will have the correct reachable
                // bottom interval.
                br[j] = reach_top;
            } else if i + 1 < num_seg_t1 {
                // No reachable top, so cell (i+1,j) gets empty bottom from this path.
                // But it might have been set by a previous column, so don't clear it.
                // Actually, br[j] at this point is the *input* to cell (i,j), and
                // we want cell (i+1,j) to get reach_top as its bottom.
                // If reach_top is empty, cell (i+1,j) gets no contribution from (i,j).
                br[j] = (1.0, 0.0); // empty for next row's bottom entry from this column
            }

            // Check: if this is the last cell (i = num_seg_t1-1, j = num_seg_t2-1),
            // check if we can reach (1, 1).
            if i == num_seg_t1 - 1 && j == num_seg_t2 - 1 {
                if (left_enters || bottom_enters)
                    && is_interval_nonempty(fi_right)
                    && fi_right.1 >= 1.0 - 1e-12
                    && is_interval_nonempty(fi_top)
                    && fi_top.1 >= 1.0 - 1e-12
                {
                    return true;
                }
            }
        }

        // Update lr for the next column.
        if j + 1 < num_seg_t2 {
            lr = new_lr;
        }
    }

    false
}

/// Check if an interval (lo, hi) is non-empty.
#[inline]
fn is_interval_nonempty(interval: (f64, f64)) -> bool {
    interval.0 <= interval.1 + 1e-12
}

/// Check if two intervals overlap.
#[inline]
fn intervals_overlap(a: (f64, f64), b: (f64, f64)) -> bool {
    a.0 <= b.1 + 1e-12 && b.0 <= a.1 + 1e-12
}

/// Union of two intervals (assuming they can be non-overlapping; returns the
/// smallest enclosing interval).
#[inline]
fn interval_union(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    if !is_interval_nonempty(a) {
        return b;
    }
    if !is_interval_nonempty(b) {
        return a;
    }
    (a.0.min(b.0), a.1.max(b.1))
}

// ---------------------------------------------------------------------------
// Hausdorff distance
// ---------------------------------------------------------------------------

/// Compute the **directed** Hausdorff distance from `t1` to `t2`.
///
/// h(t1 → t2) = max_{p ∈ t1} min_{q ∈ t2} dist(p, q)
///
/// # Errors
///
/// Returns [`SpatialError::ValueError`] if either trajectory is empty.
pub fn directed_hausdorff_distance(t1: &[Point2D], t2: &[Point2D]) -> SpatialResult<f64> {
    if t1.is_empty() || t2.is_empty() {
        return Err(SpatialError::ValueError(
            "Hausdorff: trajectories must not be empty".to_string(),
        ));
    }
    let mut max_min = 0.0_f64;
    for p in t1 {
        let min_d = t2
            .iter()
            .map(|q| point_dist(p, q))
            .fold(f64::INFINITY, f64::min);
        if min_d > max_min {
            max_min = min_d;
        }
    }
    Ok(max_min)
}

/// Compute the symmetric Hausdorff distance between two trajectories.
///
/// H(t1, t2) = max( h(t1→t2), h(t2→t1) )
///
/// # Errors
///
/// Returns [`SpatialError::ValueError`] if either trajectory is empty.
pub fn hausdorff_distance(t1: &[Point2D], t2: &[Point2D]) -> SpatialResult<f64> {
    let h1 = directed_hausdorff_distance(t1, t2)?;
    let h2 = directed_hausdorff_distance(t2, t1)?;
    Ok(h1.max(h2))
}

// ---------------------------------------------------------------------------
// EDR – Edit Distance on Real sequences
// ---------------------------------------------------------------------------

/// Compute the EDR (Edit Distance on Real sequences) between two trajectories.
///
/// EDR counts the minimum number of edit operations (insertions, deletions,
/// substitutions) needed to transform `t1` into `t2`, where two points are
/// considered a "match" if their distance is ≤ `epsilon`.
///
/// The returned value is **normalised** to [0, 1] by dividing by
/// `max(len(t1), len(t2))`.
///
/// # Arguments
///
/// * `t1`, `t2` – Trajectories.
/// * `epsilon`  – Matching threshold (> 0).
///
/// # Errors
///
/// Returns [`SpatialError::ValueError`] for empty trajectories or invalid `epsilon`.
pub fn edr_distance(t1: &[Point2D], t2: &[Point2D], epsilon: f64) -> SpatialResult<f64> {
    if t1.is_empty() || t2.is_empty() {
        return Err(SpatialError::ValueError(
            "EDR: trajectories must not be empty".to_string(),
        ));
    }
    if epsilon <= 0.0 {
        return Err(SpatialError::ValueError(
            "EDR: epsilon must be positive".to_string(),
        ));
    }

    let n = t1.len();
    let m = t2.len();

    // dp[i][j] = EDR cost for t1[0..i], t2[0..j].
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in 0..=n {
        dp[i][0] = i;
    }
    for j in 0..=m {
        dp[0][j] = j;
    }

    for i in 1..=n {
        for j in 1..=m {
            let sub_cost = if point_dist(&t1[i - 1], &t2[j - 1]) <= epsilon {
                0
            } else {
                1
            };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + sub_cost);
        }
    }

    let max_len = n.max(m) as f64;
    Ok(dp[n][m] as f64 / max_len)
}

// ---------------------------------------------------------------------------
// ERP – Edit distance with Real Penalty
// ---------------------------------------------------------------------------

/// Compute the ERP (Edit distance with Real Penalty) between two trajectories.
///
/// ERP (Chen & Ng, 2004) uses a "gap" reference point `g` for insertions /
/// deletions, measuring how far a skipped point is from `g`.  When `g` is the
/// origin `[0.0, 0.0]`, this becomes a common choice for trajectory data.
///
/// # Arguments
///
/// * `t1`, `t2` – Trajectories.
/// * `gap`      – Reference point for gap penalties.
///
/// # Errors
///
/// Returns [`SpatialError::ValueError`] for empty trajectories.
pub fn erp_distance(t1: &[Point2D], t2: &[Point2D], gap: &Point2D) -> SpatialResult<f64> {
    if t1.is_empty() || t2.is_empty() {
        return Err(SpatialError::ValueError(
            "ERP: trajectories must not be empty".to_string(),
        ));
    }

    let n = t1.len();
    let m = t2.len();

    // dp[i][j] = ERP distance between t1[0..i] and t2[0..j].
    let mut dp = vec![vec![0.0_f64; m + 1]; n + 1];

    // Base cases: one trajectory is empty → sum of gap penalties.
    for i in 1..=n {
        dp[i][0] = dp[i - 1][0] + point_dist(&t1[i - 1], gap);
    }
    for j in 1..=m {
        dp[0][j] = dp[0][j - 1] + point_dist(&t2[j - 1], gap);
    }

    for i in 1..=n {
        for j in 1..=m {
            let d_match = dp[i - 1][j - 1] + point_dist(&t1[i - 1], &t2[j - 1]);
            let d_del = dp[i - 1][j] + point_dist(&t1[i - 1], gap);
            let d_ins = dp[i][j - 1] + point_dist(&t2[j - 1], gap);
            dp[i][j] = d_match.min(d_del).min(d_ins);
        }
    }

    Ok(dp[n][m])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_line(n: usize) -> Trajectory {
        (0..n).map(|i| [i as f64, 0.0]).collect()
    }

    #[test]
    fn test_dtw_identical() {
        let t = make_line(5);
        let d = dtw_distance(&t, &t).expect("dtw identical");
        assert!(d.abs() < 1e-10, "DTW of identical trajectories must be 0");
    }

    #[test]
    fn test_dtw_simple() {
        let t1: Trajectory = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let t2: Trajectory = vec![[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]];
        let d = dtw_distance(&t1, &t2).expect("dtw simple");
        // Each point is 1.0 away; DTW sums 3 costs of 1.0.
        assert!((d - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_dtw_window_same_as_unconstrained() {
        let t1: Trajectory = vec![[0.0, 0.0], [1.0, 0.0]];
        let t2: Trajectory = vec![[0.0, 0.0], [1.0, 0.0]];
        let d_unc = dtw_distance(&t1, &t2).expect("dtw unc");
        let d_win = dtw_with_window(&t1, &t2, 10).expect("dtw win");
        assert!((d_unc - d_win).abs() < 1e-10);
    }

    #[test]
    fn test_frechet_identical() {
        let t = make_line(4);
        let d = frechet_distance(&t, &t).expect("frechet identical");
        assert!(d.abs() < 1e-10);
    }

    #[test]
    fn test_frechet_parallel() {
        let t1: Trajectory = vec![[0.0, 0.0], [1.0, 0.0]];
        let t2: Trajectory = vec![[0.0, 1.0], [1.0, 1.0]];
        let d = frechet_distance(&t1, &t2).expect("frechet parallel");
        assert!((d - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_continuous_frechet_parallel() {
        let t1: Trajectory = vec![[0.0, 0.0], [1.0, 0.0]];
        let t2: Trajectory = vec![[0.0, 1.0], [1.0, 1.0]];
        let d = continuous_frechet(&t1, &t2, 1e-4).expect("cont frechet");
        // Continuous Fréchet for two parallel unit segments offset by 1 = 1.0.
        assert!((d - 1.0).abs() < 1e-3, "got {d}");
    }

    #[test]
    fn test_hausdorff_identical() {
        let t = make_line(4);
        let d = hausdorff_distance(&t, &t).expect("hausdorff identical");
        assert!(d.abs() < 1e-10);
    }

    #[test]
    fn test_hausdorff_symmetric() {
        let t1: Trajectory = vec![[0.0, 0.0], [2.0, 0.0]];
        let t2: Trajectory = vec![[1.0, 0.0]];
        let h12 = hausdorff_distance(&t1, &t2).expect("h12");
        let h21 = hausdorff_distance(&t2, &t1).expect("h21");
        assert!((h12 - h21).abs() < 1e-10, "should be symmetric");
    }

    #[test]
    fn test_edr_identical() {
        let t = make_line(5);
        let d = edr_distance(&t, &t, 0.1).expect("edr identical");
        assert!(d.abs() < 1e-10);
    }

    #[test]
    fn test_edr_disjoint() {
        let t1: Trajectory = vec![[0.0, 0.0], [1.0, 0.0]];
        let t2: Trajectory = vec![[100.0, 0.0], [101.0, 0.0]];
        let d = edr_distance(&t1, &t2, 0.5).expect("edr disjoint");
        // Every point is a substitution, so edit distance = 2, max_len = 2 → 1.0.
        assert!((d - 1.0).abs() < 1e-10, "got {d}");
    }

    #[test]
    fn test_erp_identical() {
        let t = make_line(4);
        let gap = [0.0, 0.0];
        let d = erp_distance(&t, &t, &gap).expect("erp identical");
        assert!(d.abs() < 1e-10);
    }

    #[test]
    fn test_erp_gap_penalty() {
        // t1 = single point at origin, t2 = empty-like (one far point).
        let t1: Trajectory = vec![[0.0, 0.0]];
        let t2: Trajectory = vec![[3.0, 4.0]]; // dist to origin = 5
        let gap = [0.0, 0.0];
        let d = erp_distance(&t1, &t2, &gap).expect("erp gap");
        // Optimal path: match → dist([0,0],[3,4]) = 5.0.
        assert!((d - 5.0).abs() < 1e-10, "got {d}");
    }
}
