//! Movement pattern detection for trajectories
//!
//! - **Stop detection** – identify segments where the entity is stationary.
//! - **Turn detection** – locate turning points and compute local curvature.
//! - **Speed profile** – velocity and acceleration at each point.
//! - **Convex hull** – convex hull of all trajectory points.
//! - **Turning function** – angular turning-function representation (Arkin et al.).

use crate::error::{SpatialError, SpatialResult};
use crate::trajectory::similarity::Point2D;

// ---------------------------------------------------------------------------
// Stop detection
// ---------------------------------------------------------------------------

/// A detected stationary period (stop) in a trajectory.
#[derive(Debug, Clone)]
pub struct Stop {
    /// Index of the first point in the stop period.
    pub start_idx: usize,
    /// Index of the last point (inclusive) in the stop period.
    pub end_idx: usize,
    /// Centre of mass of the stop region.
    pub centroid: Point2D,
    /// Total displacement within the stop region (max distance from centroid).
    pub radius: f64,
}

/// Detect stationary periods in a trajectory.
///
/// A "stop" is a maximal contiguous sub-sequence where all points remain
/// within `radius` of the sub-sequence's centroid and the sub-sequence
/// spans at least `min_points` points.
///
/// # Arguments
///
/// * `traj`       – Input trajectory.
/// * `radius`     – Maximum allowable displacement from the centroid.
/// * `min_points` – Minimum number of consecutive points to qualify as a stop.
///
/// # Errors
///
/// Returns [`SpatialError::ValueError`] for invalid parameters.
pub fn detect_stops(
    traj: &[Point2D],
    radius: f64,
    min_points: usize,
) -> SpatialResult<Vec<Stop>> {
    if radius <= 0.0 {
        return Err(SpatialError::ValueError(
            "detect_stops: radius must be positive".to_string(),
        ));
    }
    if min_points < 2 {
        return Err(SpatialError::ValueError(
            "detect_stops: min_points must be at least 2".to_string(),
        ));
    }

    let n = traj.len();
    let mut stops: Vec<Stop> = Vec::new();
    if n < min_points {
        return Ok(stops);
    }

    let mut win_start = 0;

    while win_start + min_points <= n {
        // Grow the window as long as all points are within `radius` of the
        // running centroid.
        let mut win_end = win_start;
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;

        loop {
            sum_x += traj[win_end][0];
            sum_y += traj[win_end][1];
            let len = (win_end - win_start + 1) as f64;
            let cx = sum_x / len;
            let cy = sum_y / len;

            // Check all points in the current window against new centroid.
            let max_r = (win_start..=win_end)
                .map(|i| {
                    let dx = traj[i][0] - cx;
                    let dy = traj[i][1] - cy;
                    (dx * dx + dy * dy).sqrt()
                })
                .fold(0.0_f64, f64::max);

            if max_r > radius {
                break;
            }
            win_end += 1;
            if win_end >= n {
                break;
            }
        }

        let window_len = win_end - win_start;
        if window_len >= min_points {
            // Emit a stop.
            let cx = sum_x / window_len as f64;
            let cy = sum_y / window_len as f64;
            let radius_actual = (win_start..win_end)
                .map(|i| {
                    let dx = traj[i][0] - cx;
                    let dy = traj[i][1] - cy;
                    (dx * dx + dy * dy).sqrt()
                })
                .fold(0.0_f64, f64::max);

            stops.push(Stop {
                start_idx: win_start,
                end_idx: win_end - 1,
                centroid: [cx, cy],
                radius: radius_actual,
            });
            // Advance past this stop.
            win_start = win_end;
        } else {
            win_start += 1;
        }
    }

    Ok(stops)
}

// ---------------------------------------------------------------------------
// Turn detection
// ---------------------------------------------------------------------------

/// A turning point detected in a trajectory.
#[derive(Debug, Clone)]
pub struct TurnPoint {
    /// Index of the turning point in the trajectory.
    pub idx: usize,
    /// Signed turning angle in radians (positive = left turn).
    pub angle: f64,
    /// Approximate local curvature (1 / radius of the osculating circle).
    pub curvature: f64,
}

/// Detect turning points in a trajectory.
///
/// A turning point is any interior point whose absolute turning angle exceeds
/// `min_angle_rad`.
///
/// # Arguments
///
/// * `traj`          – Input trajectory (at least 3 points).
/// * `min_angle_rad` – Minimum absolute turning angle (radians, > 0).
///
/// # Errors
///
/// Returns [`SpatialError::ValueError`] for invalid `min_angle_rad`.
pub fn detect_turns(traj: &[Point2D], min_angle_rad: f64) -> SpatialResult<Vec<TurnPoint>> {
    if min_angle_rad <= 0.0 {
        return Err(SpatialError::ValueError(
            "detect_turns: min_angle_rad must be positive".to_string(),
        ));
    }

    let n = traj.len();
    let mut turns = Vec::new();

    for i in 1..n.saturating_sub(1) {
        let (angle, curvature) = turning_angle_and_curvature(&traj[i - 1], &traj[i], &traj[i + 1]);
        if angle.abs() >= min_angle_rad {
            turns.push(TurnPoint {
                idx: i,
                angle,
                curvature,
            });
        }
    }
    Ok(turns)
}

/// Compute the signed turning angle and curvature at point `b` given
/// predecessor `a` and successor `c`.
///
/// Returns `(angle_rad, curvature)`.
fn turning_angle_and_curvature(a: &Point2D, b: &Point2D, c: &Point2D) -> (f64, f64) {
    let v1 = [b[0] - a[0], b[1] - a[1]];
    let v2 = [c[0] - b[0], c[1] - b[1]];

    let l1 = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt();
    let l2 = (v2[0] * v2[0] + v2[1] * v2[1]).sqrt();

    if l1 < f64::EPSILON || l2 < f64::EPSILON {
        return (0.0, 0.0);
    }

    // Signed angle via atan2 of the cross and dot product.
    let cross = v1[0] * v2[1] - v1[1] * v2[0];
    let dot = v1[0] * v2[0] + v1[1] * v2[1];
    let angle = cross.atan2(dot);

    // Menger curvature: κ = 2|sin(angle)| / (l1 + l2) (approximate).
    let curvature = if (l1 + l2) > f64::EPSILON {
        2.0 * cross.abs() / (l1 * l2 * (l1 + l2))
    } else {
        0.0
    };

    (angle, curvature)
}

// ---------------------------------------------------------------------------
// Speed profile
// ---------------------------------------------------------------------------

/// Speed and acceleration at a single point along a trajectory.
#[derive(Debug, Clone)]
pub struct SpeedSample {
    /// Index in the original trajectory.
    pub idx: usize,
    /// Instantaneous speed (distance per time unit) using centred differences.
    pub speed: f64,
    /// Instantaneous acceleration (second derivative of position magnitude).
    pub acceleration: f64,
}

/// Compute the speed and acceleration profile along a trajectory.
///
/// Assumes uniform time sampling (`dt` seconds per step).  Uses centred
/// differences for interior points and one-sided differences at the endpoints.
///
/// # Arguments
///
/// * `traj` – Trajectory.
/// * `dt`   – Time interval between consecutive points (seconds).
///
/// # Errors
///
/// Returns [`SpatialError::ValueError`] if `dt <= 0` or `traj.len() < 2`.
pub fn speed_profile(traj: &[Point2D], dt: f64) -> SpatialResult<Vec<SpeedSample>> {
    if dt <= 0.0 {
        return Err(SpatialError::ValueError(
            "speed_profile: dt must be positive".to_string(),
        ));
    }
    let n = traj.len();
    if n < 2 {
        return Err(SpatialError::ValueError(
            "speed_profile: trajectory must have at least 2 points".to_string(),
        ));
    }

    // Compute displacement magnitudes between consecutive points.
    let disps: Vec<f64> = (0..n - 1)
        .map(|i| {
            let dx = traj[i + 1][0] - traj[i][0];
            let dy = traj[i + 1][1] - traj[i][1];
            (dx * dx + dy * dy).sqrt()
        })
        .collect();

    // Speed at each point (centred or one-sided).
    let speed: Vec<f64> = (0..n)
        .map(|i| {
            if i == 0 {
                disps[0] / dt
            } else if i == n - 1 {
                disps[n - 2] / dt
            } else {
                (disps[i - 1] + disps[i]) / (2.0 * dt)
            }
        })
        .collect();

    // Acceleration at each point.
    let accel: Vec<f64> = (0..n)
        .map(|i| {
            if i == 0 {
                (speed[1] - speed[0]) / dt
            } else if i == n - 1 {
                (speed[n - 1] - speed[n - 2]) / dt
            } else {
                (speed[i + 1] - speed[i - 1]) / (2.0 * dt)
            }
        })
        .collect();

    Ok((0..n)
        .map(|i| SpeedSample {
            idx: i,
            speed: speed[i],
            acceleration: accel[i],
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Convex hull of trajectory points
// ---------------------------------------------------------------------------

/// Compute the convex hull of a trajectory's point set.
///
/// Uses the Graham scan algorithm (O(n log n)).
///
/// # Returns
///
/// Indices of the hull vertices in counter-clockwise order.
///
/// # Errors
///
/// Returns [`SpatialError::ValueError`] if the trajectory is empty.
pub fn convex_hull_trajectory(traj: &[Point2D]) -> SpatialResult<Vec<usize>> {
    if traj.is_empty() {
        return Err(SpatialError::ValueError(
            "convex_hull_trajectory: trajectory must not be empty".to_string(),
        ));
    }
    if traj.len() == 1 {
        return Ok(vec![0]);
    }
    if traj.len() == 2 {
        return Ok(vec![0, 1]);
    }

    // Find the lowest (then leftmost) point as the pivot.
    let pivot = (0..traj.len())
        .min_by(|&a, &b| {
            let pa = &traj[a];
            let pb = &traj[b];
            pa[1]
                .partial_cmp(&pb[1])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| pa[0].partial_cmp(&pb[0]).unwrap_or(std::cmp::Ordering::Equal))
        })
        .expect("min on non-empty range always succeeds");

    // Sort the remaining indices by polar angle from the pivot.
    let mut indices: Vec<usize> = (0..traj.len()).filter(|&i| i != pivot).collect();
    let px = traj[pivot][0];
    let py = traj[pivot][1];

    indices.sort_by(|&a, &b| {
        let ax = traj[a][0] - px;
        let ay = traj[a][1] - py;
        let bx = traj[b][0] - px;
        let by = traj[b][1] - py;
        let angle_a = ay.atan2(ax);
        let angle_b = by.atan2(bx);
        angle_a
            .partial_cmp(&angle_b)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                let da = ax * ax + ay * ay;
                let db = bx * bx + by * by;
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    // Graham scan.
    let mut hull: Vec<usize> = vec![pivot];
    for &idx in &indices {
        // Pop while the last three points make a non-left turn.
        while hull.len() >= 2 {
            let n = hull.len();
            let o = hull[n - 2];
            let a = hull[n - 1];
            let b = idx;
            let cross = cross_product(&traj[o], &traj[a], &traj[b]);
            if cross <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(idx);
    }

    Ok(hull)
}

/// Cross product of vectors (o→a) and (o→b).
#[inline]
fn cross_product(o: &Point2D, a: &Point2D, b: &Point2D) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

// ---------------------------------------------------------------------------
// Turning function
// ---------------------------------------------------------------------------

/// A sample of the turning function at a particular arc-length position.
#[derive(Debug, Clone)]
pub struct TurningFunctionSample {
    /// Normalised arc-length position in [0, 1].
    pub arc_length: f64,
    /// Cumulative turning angle (radians) up to this position.
    pub cumulative_angle: f64,
}

/// Compute the **turning function** representation of a trajectory.
///
/// The turning function φ(s) gives the cumulative turning angle as a function
/// of arc-length s ∈ [0, 1].  It is a classical shape descriptor (Arkin et al.,
/// 1991) often used to compare polylines.
///
/// # Arguments
///
/// * `traj` – Input trajectory (at least 2 points).
///
/// # Errors
///
/// Returns [`SpatialError::ValueError`] if the trajectory has fewer than 2 points.
pub fn turning_function(traj: &[Point2D]) -> SpatialResult<Vec<TurningFunctionSample>> {
    let n = traj.len();
    if n < 2 {
        return Err(SpatialError::ValueError(
            "turning_function: trajectory must have at least 2 points".to_string(),
        ));
    }

    // Compute segment lengths and total arc length.
    let seg_lens: Vec<f64> = (0..n - 1)
        .map(|i| {
            let dx = traj[i + 1][0] - traj[i][0];
            let dy = traj[i + 1][1] - traj[i][1];
            (dx * dx + dy * dy).sqrt()
        })
        .collect();
    let total_len: f64 = seg_lens.iter().sum();
    if total_len < f64::EPSILON {
        // All points coincide.
        return Ok(vec![TurningFunctionSample {
            arc_length: 0.0,
            cumulative_angle: 0.0,
        }]);
    }

    // Compute segment directions and turning angles.
    let mut samples = Vec::with_capacity(n);
    let mut cum_arc = 0.0_f64;
    let mut cum_angle = 0.0_f64;

    // Initial direction.
    let dx0 = traj[1][0] - traj[0][0];
    let dy0 = traj[1][1] - traj[0][1];
    let initial_bearing = dy0.atan2(dx0);

    samples.push(TurningFunctionSample {
        arc_length: 0.0,
        cumulative_angle: 0.0,
    });

    let mut prev_bearing = initial_bearing;

    for i in 1..n - 1 {
        cum_arc += seg_lens[i - 1] / total_len;

        let dx = traj[i + 1][0] - traj[i][0];
        let dy = traj[i + 1][1] - traj[i][1];
        let bearing = dy.atan2(dx);

        // Signed angle difference, normalised to (-π, π].
        let mut delta = bearing - prev_bearing;
        while delta > std::f64::consts::PI {
            delta -= 2.0 * std::f64::consts::PI;
        }
        while delta <= -std::f64::consts::PI {
            delta += 2.0 * std::f64::consts::PI;
        }
        cum_angle += delta;
        prev_bearing = bearing;

        samples.push(TurningFunctionSample {
            arc_length: cum_arc,
            cumulative_angle: cum_angle,
        });
    }

    // Final arc-length position.
    cum_arc += seg_lens[n - 2] / total_len;
    samples.push(TurningFunctionSample {
        arc_length: cum_arc.min(1.0),
        cumulative_angle: cum_angle,
    });

    Ok(samples)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_stops_stationary() {
        // All points at the origin → one big stop.
        let traj: Vec<Point2D> = (0..10).map(|_| [0.0_f64, 0.0_f64]).collect();
        let stops = detect_stops(&traj, 0.5, 3).expect("detect_stops stationary");
        assert!(!stops.is_empty(), "should detect at least one stop");
        assert_eq!(stops[0].start_idx, 0);
    }

    #[test]
    fn test_detect_stops_moving() {
        // Monotone trajectory with large steps → no stops.
        let traj: Vec<Point2D> = (0..20).map(|i| [i as f64 * 10.0, 0.0]).collect();
        let stops = detect_stops(&traj, 0.5, 3).expect("detect_stops moving");
        assert!(stops.is_empty(), "moving trajectory should have no stops");
    }

    #[test]
    fn test_detect_turns_straight() {
        let traj: Vec<Point2D> = (0..5).map(|i| [i as f64, 0.0]).collect();
        let turns = detect_turns(&traj, 0.1).expect("detect_turns straight");
        assert!(turns.is_empty(), "straight line has no turns");
    }

    #[test]
    fn test_detect_turns_right_angle() {
        let traj: Vec<Point2D> = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];
        let turns = detect_turns(&traj, 0.1).expect("detect_turns right angle");
        assert!(!turns.is_empty(), "right-angle turn should be detected");
        assert!(
            (turns[0].angle.abs() - std::f64::consts::FRAC_PI_2).abs() < 1e-9,
            "angle should be π/2"
        );
    }

    #[test]
    fn test_speed_profile_constant_speed() {
        let traj: Vec<Point2D> = (0..5).map(|i| [i as f64, 0.0]).collect();
        let profile = speed_profile(&traj, 1.0).expect("speed_profile");
        for s in &profile {
            assert!(
                (s.speed - 1.0).abs() < 1e-9,
                "speed should be 1.0, got {}",
                s.speed
            );
        }
    }

    #[test]
    fn test_speed_profile_dt_error() {
        let traj: Vec<Point2D> = vec![[0.0, 0.0], [1.0, 0.0]];
        assert!(speed_profile(&traj, -1.0).is_err());
        assert!(speed_profile(&traj, 0.0).is_err());
    }

    #[test]
    fn test_convex_hull_square() {
        let traj: Vec<Point2D> = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5], // interior point
        ];
        let hull = convex_hull_trajectory(&traj).expect("convex hull square");
        // Interior point (0.5, 0.5) must not be on the hull.
        assert!(!hull.contains(&4), "interior point should not be on hull");
        assert_eq!(hull.len(), 4, "hull should have 4 vertices");
    }

    #[test]
    fn test_convex_hull_collinear() {
        let traj: Vec<Point2D> = (0..5).map(|i| [i as f64, 0.0]).collect();
        let hull = convex_hull_trajectory(&traj).expect("convex hull collinear");
        // Collinear points – all are on the hull boundary.
        assert!(hull.len() >= 2);
    }

    #[test]
    fn test_turning_function_straight_line() {
        let traj: Vec<Point2D> = (0..5).map(|i| [i as f64, 0.0]).collect();
        let tf = turning_function(&traj).expect("turning function straight");
        // For a straight line, cumulative angle is always 0.
        for s in &tf {
            assert!(
                s.cumulative_angle.abs() < 1e-9,
                "straight line cumulative angle should be 0, got {}",
                s.cumulative_angle
            );
        }
    }

    #[test]
    fn test_turning_function_90_deg_turn() {
        let traj: Vec<Point2D> = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];
        let tf = turning_function(&traj).expect("turning function 90");
        let last = tf.last().expect("non-empty");
        assert!(
            (last.cumulative_angle.abs() - std::f64::consts::FRAC_PI_2).abs() < 1e-9,
            "cumulative angle should be π/2, got {}",
            last.cumulative_angle.abs()
        );
    }
}
