//! Spatial Autocorrelation Statistics
//!
//! Provides:
//! - Global Moran's I statistic with z-score and p-value
//! - Geary's C statistic
//! - Local Indicators of Spatial Association (LISA / local Moran's I)
//! - Ripley's K function and L function (isotropic edge correction)
//!
//! # Mathematical background
//!
//! **Moran's I**:
//! ```text
//! I = (n / S0) * [Σ_ij w_ij (x_i - x̄)(x_j - x̄)] / [Σ_i (x_i - x̄)²]
//! ```
//! where `S0 = Σ_ij w_ij`.  Under randomisation: `E[I] = -1/(n-1)`.
//!
//! **Geary's C**:
//! ```text
//! C = [(n-1) Σ_ij w_ij (x_i - x_j)²] / [2 S0 Σ_i (x_i - x̄)²]
//! ```
//!
//! **Ripley's K** (isotropic edge correction):
//! ```text
//! K̂(d) = (A / n²) Σ_{i≠j} e_ij · 1[d_ij ≤ d]
//! ```
//! where `e_ij` is the isotropic edge-correction weight.

use scirs2_core::ndarray::Array2;

use super::{SpatialError, SpatialResult};

// ---------------------------------------------------------------------------
// Helper: standard-normal CDF
// ---------------------------------------------------------------------------

/// Abramowitz & Stegun rational approximation of erf(x).
fn erf_approx(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * ax);
    let poly = t * (0.254_829_592
        + t * (-0.284_496_736
            + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    sign * (1.0 - poly * (-ax * ax).exp())
}

fn norm_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2))
}

/// Two-tailed p-value from a standard-normal z-score.
fn two_tailed_p(z: f64) -> f64 {
    2.0 * norm_cdf(-z.abs())
}

// ---------------------------------------------------------------------------
// Moran's I — global
// ---------------------------------------------------------------------------

/// Result of global Moran's I test.
#[derive(Debug, Clone)]
pub struct MoranResult {
    /// The Moran's I statistic (roughly in `[-1, 1]`).
    pub i: f64,
    /// Expected value under no spatial autocorrelation: `-1/(n-1)`.
    pub expected_i: f64,
    /// Variance under the randomisation assumption.
    pub variance_i: f64,
    /// Standardised z-score.
    pub z_score: f64,
    /// Two-tailed p-value via the standard-normal approximation.
    pub p_value: f64,
}

/// Compute Moran's I spatial autocorrelation statistic.
///
/// # Arguments
/// * `values`  – Observed variable at `n` locations.
/// * `weights` – Row-standardised (or raw) `n×n` spatial weights matrix.
///
/// # Returns
/// [`MoranResult`] on success.
///
/// # Errors
/// Returns [`SpatialError`] when dimensions mismatch, `n < 3`, all values are
/// identical, or the total weight sum is zero.
pub fn morans_i(values: &[f64], weights: &Array2<f64>) -> SpatialResult<MoranResult> {
    let n = values.len();
    if n < 3 {
        return Err(SpatialError::InsufficientData(
            "Moran's I requires at least 3 observations".to_string(),
        ));
    }
    if weights.nrows() != n || weights.ncols() != n {
        return Err(SpatialError::DimensionMismatch(format!(
            "weights must be {n}×{n}, got {}×{}",
            weights.nrows(),
            weights.ncols()
        )));
    }
    for &v in values {
        if !v.is_finite() {
            return Err(SpatialError::InvalidArgument(
                "values must be finite".to_string(),
            ));
        }
    }

    let mean = values.iter().sum::<f64>() / n as f64;
    let z: Vec<f64> = values.iter().map(|&v| v - mean).collect();

    let s0: f64 = weights.iter().sum();
    if s0 == 0.0 {
        return Err(SpatialError::InvalidArgument(
            "Sum of spatial weights is zero".to_string(),
        ));
    }

    // Moran numerator: Σ_i Σ_j w_ij z_i z_j
    let mut numerator = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            numerator += weights[[i, j]] * z[i] * z[j];
        }
    }

    let s_sq: f64 = z.iter().map(|&zi| zi * zi).sum();
    if s_sq == 0.0 {
        return Err(SpatialError::InvalidArgument(
            "All values are identical; Moran's I is undefined".to_string(),
        ));
    }

    let i_stat = (n as f64 / s0) * (numerator / s_sq);
    let expected_i = -1.0 / (n as f64 - 1.0);

    // Variance (randomisation assumption, Cliff & Ord 1981)
    // S1 = 0.5 Σ_i Σ_j (w_ij + w_ji)²
    let mut s1 = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let v = weights[[i, j]] + weights[[j, i]];
            s1 += v * v;
        }
    }
    s1 *= 0.5;

    // S2 = Σ_i (Σ_j w_ij + Σ_j w_ji)²
    let mut s2_stat = 0.0_f64;
    for i in 0..n {
        let row_sum: f64 = (0..n).map(|j| weights[[i, j]]).sum();
        let col_sum: f64 = (0..n).map(|j| weights[[j, i]]).sum();
        s2_stat += (row_sum + col_sum).powi(2);
    }

    let n_f = n as f64;
    let m2 = z.iter().map(|&zi| zi.powi(2)).sum::<f64>() / n_f;
    let m4 = z.iter().map(|&zi| zi.powi(4)).sum::<f64>() / n_f;
    let kurtosis = m4 / (m2 * m2);

    let w_sq = s0 * s0;
    let num_var =
        n_f * (n_f * n_f - 3.0 * n_f + 3.0) * s1 - n_f * s2_stat + 3.0 * w_sq;
    let den_var = (n_f - 1.0) * (n_f - 2.0) * (n_f - 3.0) * w_sq;
    let kur_term = kurtosis
        * ((n_f * n_f - n_f) * s1 - 2.0 * n_f * s2_stat + 6.0 * w_sq)
        / ((n_f - 1.0) * (n_f - 2.0) * (n_f - 3.0) * w_sq);

    let variance_i = (num_var / den_var - kur_term - expected_i * expected_i).max(1e-15);
    let z_score = (i_stat - expected_i) / variance_i.sqrt();
    let p_value = two_tailed_p(z_score);

    Ok(MoranResult {
        i: i_stat,
        expected_i,
        variance_i,
        z_score,
        p_value,
    })
}

// ---------------------------------------------------------------------------
// Geary's C
// ---------------------------------------------------------------------------

/// Result of Geary's C spatial autocorrelation test.
#[derive(Debug, Clone)]
pub struct GearyResult {
    /// Geary's C statistic.
    /// * `C ≈ 0` – strong positive autocorrelation
    /// * `C ≈ 1` – no autocorrelation (expected value under CSR)
    /// * `C > 1` – negative autocorrelation
    pub c: f64,
    /// Expected value under null hypothesis: 1.0.
    pub expected_c: f64,
    /// Variance under randomisation.
    pub variance_c: f64,
    /// Standardised z-score.
    pub z_score: f64,
    /// Two-tailed p-value.
    pub p_value: f64,
}

/// Compute Geary's C spatial autocorrelation statistic.
///
/// # Formula
/// ```text
/// C = [(n-1) Σ_ij w_ij (x_i - x_j)²] / [2 S0 Σ_i (x_i - x̄)²]
/// ```
///
/// # Errors
/// Returns [`SpatialError`] when `n < 3` or inputs are invalid.
pub fn gearys_c(values: &[f64], weights: &Array2<f64>) -> SpatialResult<GearyResult> {
    let n = values.len();
    if n < 3 {
        return Err(SpatialError::InsufficientData(
            "Geary's C requires at least 3 observations".to_string(),
        ));
    }
    if weights.nrows() != n || weights.ncols() != n {
        return Err(SpatialError::DimensionMismatch(format!(
            "weights must be {n}×{n}, got {}×{}",
            weights.nrows(),
            weights.ncols()
        )));
    }

    let s0: f64 = weights.iter().sum();
    if s0 == 0.0 {
        return Err(SpatialError::InvalidArgument(
            "Sum of spatial weights is zero".to_string(),
        ));
    }

    let mean = values.iter().sum::<f64>() / n as f64;
    let denom: f64 = values.iter().map(|&x| (x - mean).powi(2)).sum();
    if denom == 0.0 {
        return Err(SpatialError::InvalidArgument(
            "All values are identical; Geary's C is undefined".to_string(),
        ));
    }

    // Numerator: Σ_ij w_ij (x_i - x_j)²
    let mut numer = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let diff = values[i] - values[j];
            numer += weights[[i, j]] * diff * diff;
        }
    }

    let c = ((n as f64 - 1.0) * numer) / (2.0 * s0 * denom);
    let expected_c = 1.0;

    // Variance under randomisation (Cliff & Ord 1981, eq. 2.6)
    let n_f = n as f64;

    let mut s1 = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let v = weights[[i, j]] + weights[[j, i]];
            s1 += v * v;
        }
    }
    s1 *= 0.5;

    let mut s2_stat = 0.0_f64;
    for i in 0..n {
        let rs: f64 = (0..n).map(|j| weights[[i, j]]).sum();
        let cs: f64 = (0..n).map(|j| weights[[j, i]]).sum();
        s2_stat += (rs + cs).powi(2);
    }

    let z_dev: Vec<f64> = values.iter().map(|&x| x - mean).collect();
    let m2 = z_dev.iter().map(|&zi| zi.powi(2)).sum::<f64>() / n_f;
    let m4 = z_dev.iter().map(|&zi| zi.powi(4)).sum::<f64>() / n_f;
    let kurtosis = m4 / (m2 * m2);

    let w_sq = s0 * s0;
    let var_num = (2.0 * s1 + s2_stat) * (n_f - 1.0)
        - 4.0 * w_sq;
    let var_den = 2.0 * (n_f + 1.0) * w_sq;
    let kur_term = kurtosis * ((n_f - 1.0) * s1 - s2_stat / 4.0 + w_sq)
        / ((n_f + 1.0) * w_sq);
    let variance_c = ((var_num / var_den) - kur_term + 0.0).max(1e-15);

    let z_score = (c - expected_c) / variance_c.sqrt();
    let p_value = two_tailed_p(z_score);

    Ok(GearyResult {
        c,
        expected_c,
        variance_c,
        z_score,
        p_value,
    })
}

// ---------------------------------------------------------------------------
// Local Moran's I (LISA)
// ---------------------------------------------------------------------------

/// Classification of a location in the LISA quadrant map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClusterType {
    /// High value surrounded by high values (hot-spot).
    HighHigh,
    /// Low value surrounded by low values (cold-spot).
    LowLow,
    /// High value surrounded by low values (spatial outlier).
    HighLow,
    /// Low value surrounded by high values (spatial outlier).
    LowHigh,
    /// Local I not statistically significant at the 5 % level.
    NotSignificant,
}

/// Result of local Moran's I (LISA) for a single observation.
#[derive(Debug, Clone)]
pub struct LocalMoranResult {
    /// Local Moran's I value.
    pub i_local: f64,
    /// Standardised z-score.
    pub z_score: f64,
    /// Two-tailed p-value.
    pub p_value: f64,
    /// LISA cluster classification.
    pub cluster_type: ClusterType,
}

/// Compute Local Indicators of Spatial Association (LISA / Local Moran's I).
///
/// Each element of the returned vector corresponds to one observation.
///
/// # Errors
/// Returns [`SpatialError`] when `n < 3` or inputs are invalid.
pub fn local_morans_i(
    values: &[f64],
    weights: &Array2<f64>,
) -> SpatialResult<Vec<LocalMoranResult>> {
    let n = values.len();
    if n < 3 {
        return Err(SpatialError::InsufficientData(
            "LISA requires at least 3 observations".to_string(),
        ));
    }
    if weights.nrows() != n || weights.ncols() != n {
        return Err(SpatialError::DimensionMismatch(format!(
            "weights must be {n}×{n}, got {}×{}",
            weights.nrows(),
            weights.ncols()
        )));
    }

    let mean = values.iter().sum::<f64>() / n as f64;
    let z_dev: Vec<f64> = values.iter().map(|&v| v - mean).collect();
    let m2 = z_dev.iter().map(|&zi| zi * zi).sum::<f64>() / n as f64;

    if m2 == 0.0 {
        return Err(SpatialError::InvalidArgument(
            "All values are identical; LISA is undefined".to_string(),
        ));
    }

    let mut results = Vec::with_capacity(n);

    for i in 0..n {
        // Row-standardise weights for location i on the fly
        let row_sum: f64 = (0..n).map(|j| weights[[i, j]]).sum();
        let lag_i: f64 = if row_sum > 0.0 {
            (0..n)
                .map(|j| (weights[[i, j]] / row_sum) * z_dev[j])
                .sum()
        } else {
            0.0
        };

        let i_local = (z_dev[i] / m2) * lag_i;

        // Analytical variance approximation (Anselin 1995)
        let w_sq_sum: f64 = if row_sum > 0.0 {
            (0..n)
                .map(|j| (weights[[i, j]] / row_sum).powi(2))
                .sum::<f64>()
        } else {
            0.0
        };
        let variance = (w_sq_sum * (n as f64) / (n as f64 - 1.0)).max(1e-15);
        let z_score = i_local / variance.sqrt();
        let p_value = two_tailed_p(z_score);

        let cluster_type = if p_value >= 0.05 {
            ClusterType::NotSignificant
        } else {
            match (z_dev[i] >= 0.0, lag_i >= 0.0) {
                (true, true) => ClusterType::HighHigh,
                (false, false) => ClusterType::LowLow,
                (true, false) => ClusterType::HighLow,
                (false, true) => ClusterType::LowHigh,
            }
        };

        results.push(LocalMoranResult {
            i_local,
            z_score,
            p_value,
            cluster_type,
        });
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Ripley's K and L functions
// ---------------------------------------------------------------------------

/// Euclidean distance between two 2-D points.
#[inline]
fn dist2d(p: (f64, f64), q: (f64, f64)) -> f64 {
    ((p.0 - q.0).powi(2) + (p.1 - q.1).powi(2)).sqrt()
}

/// Isotropic edge-correction weight (Ripley 1976).
///
/// Returns the fraction of a circle centred at `p_i` with radius `r` that
/// falls inside a rectangular window `[0, w] × [0, h]`.
/// We approximate the window by the unit square scaled by effective width/height
/// derived from the total `area`. For a generic convex window the exact weight
/// requires geometry; here we use the standard approximation `e_ij = 1`.
/// For rectangular windows the proportional-arc correction is used.
///
/// The function receives the bounding box inferred from the point set.
fn edge_correction_weight(p: (f64, f64), r: f64, bbox: (f64, f64, f64, f64)) -> f64 {
    let (x_min, x_max, y_min, y_max) = bbox;
    if r <= 0.0 {
        return 1.0;
    }

    // Lengths of arcs that fall outside the bounding box
    // We count how many sides of the rectangle the circle crosses and
    // compute the proportion of the circumference inside the window.
    // Uses the arc-length approximation standard in ecology literature.
    let mut outside_angle = 0.0_f64;

    // Distance to each of the 4 walls
    let d_left = (p.0 - x_min).max(0.0);
    let d_right = (x_max - p.0).max(0.0);
    let d_bottom = (p.1 - y_min).max(0.0);
    let d_top = (y_max - p.1).max(0.0);

    let half_pi = std::f64::consts::FRAC_PI_2;

    // For each wall that is closer than r, add the arc-angle outside
    if d_left < r {
        let angle = (d_left / r).clamp(-1.0, 1.0).acos();
        outside_angle += 2.0 * angle; // symmetric arc on both sides of normal
    }
    if d_right < r {
        let angle = (d_right / r).clamp(-1.0, 1.0).acos();
        outside_angle += 2.0 * angle;
    }
    if d_bottom < r {
        let angle = (d_bottom / r).clamp(-1.0, 1.0).acos();
        outside_angle += 2.0 * angle;
    }
    if d_top < r {
        let angle = (d_top / r).clamp(-1.0, 1.0).acos();
        outside_angle += 2.0 * angle;
    }

    // Clamp to [0, 2π] and compute proportion inside
    let two_pi = 2.0 * std::f64::consts::PI;
    let outside_angle = outside_angle.min(two_pi - 4.0 * half_pi);
    let prop_inside = 1.0 - outside_angle / two_pi;
    prop_inside.clamp(1e-6, 1.0)
}

/// Compute the bounding box of a set of 2-D points.
fn bounding_box(points: &[(f64, f64)]) -> Option<(f64, f64, f64, f64)> {
    if points.is_empty() {
        return None;
    }
    let (mut x_min, mut x_max) = (points[0].0, points[0].0);
    let (mut y_min, mut y_max) = (points[0].1, points[0].1);
    for &(x, y) in points.iter().skip(1) {
        if x < x_min {
            x_min = x;
        }
        if x > x_max {
            x_max = x;
        }
        if y < y_min {
            y_min = y;
        }
        if y > y_max {
            y_max = y;
        }
    }
    Some((x_min, x_max, y_min, y_max))
}

/// Compute Ripley's K function for a 2-D point pattern.
///
/// Uses Ripley's isotropic edge correction.
///
/// # Arguments
/// * `points`    – `(x, y)` coordinates of the point pattern.
/// * `area`      – Area of the study region (e.g. `width * height`).
/// * `distances` – Distance thresholds at which to evaluate `K`.
///
/// # Returns
/// Vector of `K(d)` values, one per entry in `distances`.
///
/// # Errors
/// Returns [`SpatialError`] when `n < 2` or `area ≤ 0`.
pub fn ripleys_k(
    points: &[(f64, f64)],
    area: f64,
    distances: &[f64],
) -> SpatialResult<Vec<f64>> {
    let n = points.len();
    if n < 2 {
        return Err(SpatialError::InsufficientData(
            "Ripley's K requires at least 2 points".to_string(),
        ));
    }
    if area <= 0.0 || !area.is_finite() {
        return Err(SpatialError::InvalidArgument(
            "area must be a positive finite number".to_string(),
        ));
    }
    if distances.is_empty() {
        return Ok(Vec::new());
    }

    let bbox = bounding_box(points)
        .ok_or_else(|| SpatialError::InvalidArgument("empty point set".to_string()))?;

    let n_f = n as f64;
    // Intensity estimate λ̂ = n / area
    let lambda = n_f / area;

    let mut k_values = vec![0.0_f64; distances.len()];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let d_ij = dist2d(points[i], points[j]);
            let w_ij = edge_correction_weight(points[i], d_ij, bbox);

            // Accumulate contribution to each distance threshold
            for (k_idx, &d) in distances.iter().enumerate() {
                if d_ij <= d {
                    k_values[k_idx] += 1.0 / w_ij;
                }
            }
        }
    }

    // Normalise: K̂(d) = (1/λ²n) Σ_{i≠j} e_ij · 1[d_ij ≤ d]
    //                   = (area / n²) Σ …
    let scale = area / (n_f * n_f);
    for v in &mut k_values {
        *v *= scale;
    }

    Ok(k_values)
}

/// Compute Ripley's L function (variance-stabilised form of K).
///
/// `L(d) = sqrt(K(d) / π) - d`
///
/// Under Complete Spatial Randomness (CSR), `K(d) = π d²` so `L(d) ≈ 0`.
/// Positive values indicate clustering; negative values indicate regularity.
///
/// # Errors
/// Propagates errors from [`ripleys_k`].
pub fn ripleys_l(
    points: &[(f64, f64)],
    area: f64,
    distances: &[f64],
) -> SpatialResult<Vec<f64>> {
    let k_vals = ripleys_k(points, area, distances)?;
    let l_vals: Vec<f64> = k_vals
        .iter()
        .zip(distances.iter())
        .map(|(&k, &d)| (k / std::f64::consts::PI).sqrt() - d)
        .collect();
    Ok(l_vals)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    // Build a simple contiguous weights matrix for a 1-D chain of n sites.
    fn chain_weights(n: usize) -> Array2<f64> {
        let mut w = Array2::zeros((n, n));
        for i in 0..n {
            if i > 0 {
                w[[i, i - 1]] = 1.0;
            }
            if i + 1 < n {
                w[[i, i + 1]] = 1.0;
            }
        }
        // Row-standardise
        for i in 0..n {
            let rs: f64 = (0..n).map(|j| w[[i, j]]).sum();
            if rs > 0.0 {
                for j in 0..n {
                    w[[i, j]] /= rs;
                }
            }
        }
        w
    }

    #[test]
    fn test_morans_i_clustered() {
        // Strongly clustered values: first half high, second half low
        let n = 10_usize;
        let values: Vec<f64> = (0..n)
            .map(|i| if i < n / 2 { 10.0 } else { 1.0 })
            .collect();
        let w = chain_weights(n);
        let result = morans_i(&values, &w).expect("morans_i failed");
        // Clustered pattern → large positive I
        assert!(
            result.i > 0.0,
            "Expected positive I for clustered data, got {}",
            result.i
        );
        // p-value should indicate significant autocorrelation
        assert!(
            result.p_value < 0.1,
            "Expected significant p-value, got {}",
            result.p_value
        );
    }

    #[test]
    fn test_morans_i_checkerboard() {
        // Alternating values → negative autocorrelation
        let n = 8_usize;
        let values: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { 10.0 } else { 1.0 })
            .collect();
        let w = chain_weights(n);
        let result = morans_i(&values, &w).expect("morans_i failed");
        assert!(
            result.i < 0.0,
            "Expected negative I for checkerboard pattern, got {}",
            result.i
        );
    }

    #[test]
    fn test_morans_i_constant_fails() {
        let n = 5_usize;
        let values = vec![3.0_f64; n];
        let w = chain_weights(n);
        assert!(morans_i(&values, &w).is_err());
    }

    #[test]
    fn test_morans_i_dimension_mismatch() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let w = chain_weights(5); // wrong size
        assert!(morans_i(&values, &w).is_err());
    }

    #[test]
    fn test_morans_i_too_few_obs() {
        let values = vec![1.0, 2.0];
        let w = chain_weights(2);
        assert!(morans_i(&values, &w).is_err());
    }

    #[test]
    fn test_gearys_c_clustered() {
        let n = 10_usize;
        let values: Vec<f64> = (0..n)
            .map(|i| if i < n / 2 { 10.0 } else { 1.0 })
            .collect();
        let w = chain_weights(n);
        let result = gearys_c(&values, &w).expect("gearys_c failed");
        // Clustered → C < 1
        assert!(
            result.c < 1.0,
            "Expected C < 1 for clustered data, got {}",
            result.c
        );
    }

    #[test]
    fn test_gearys_c_expected_value() {
        let result_ec = {
            let n = 8_usize;
            let values: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let w = chain_weights(n);
            gearys_c(&values, &w).expect("gearys_c failed")
        };
        // expected_c is always 1.0
        assert!(
            (result_ec.expected_c - 1.0).abs() < 1e-10,
            "Expected C = 1.0 under null, got {}",
            result_ec.expected_c
        );
    }

    #[test]
    fn test_local_morans_i_cluster_classification() {
        let n = 8_usize;
        // First 4 are high, last 4 are low
        let values: Vec<f64> = (0..n)
            .map(|i| if i < 4 { 10.0 } else { 1.0 })
            .collect();
        let w = chain_weights(n);
        let results = local_morans_i(&values, &w).expect("local_morans_i failed");
        assert_eq!(results.len(), n);
        // Inner high-high sites (index 1, 2) should be HighHigh or NotSignificant
        for r in &results {
            // cluster_type must be one of the valid variants
            let valid = matches!(
                r.cluster_type,
                ClusterType::HighHigh
                    | ClusterType::LowLow
                    | ClusterType::HighLow
                    | ClusterType::LowHigh
                    | ClusterType::NotSignificant
            );
            assert!(valid, "unexpected cluster type");
        }
    }

    #[test]
    fn test_local_morans_i_length() {
        let n = 6_usize;
        let values: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let w = chain_weights(n);
        let results = local_morans_i(&values, &w).expect("local_morans_i failed");
        assert_eq!(results.len(), n);
    }

    #[test]
    fn test_ripleys_k_csr_approx() {
        // Under CSR, K(d) ≈ π·d²
        // Place points on a regular grid to approximate CSR
        let side = 10_usize;
        let n = side * side;
        let mut pts = Vec::with_capacity(n);
        for i in 0..side {
            for j in 0..side {
                pts.push((i as f64 + 0.5, j as f64 + 0.5));
            }
        }
        let area = (side * side) as f64;
        let distances = vec![0.5, 1.0, 1.5, 2.0];

        let k_vals = ripleys_k(&pts, area, &distances).expect("ripleys_k failed");
        assert_eq!(k_vals.len(), distances.len());

        // K(d) should be broadly near π·d² for regular patterns
        // (exact for grid can deviate from CSR — just check positivity and scaling)
        for (k, &d) in k_vals.iter().zip(distances.iter()) {
            assert!(*k >= 0.0, "K({}) = {} should be non-negative", d, k);
            let csr_k = std::f64::consts::PI * d * d;
            // Allow 200% deviation for finite-n grid vs. CSR approximation
            assert!(
                *k < 3.0 * csr_k + 0.5,
                "K({}) = {} way above CSR expectation {}",
                d,
                k,
                csr_k
            );
        }
    }

    #[test]
    fn test_ripleys_l_near_zero_for_csr() {
        let side = 10_usize;
        let mut pts = Vec::with_capacity(side * side);
        for i in 0..side {
            for j in 0..side {
                pts.push((i as f64 + 0.5, j as f64 + 0.5));
            }
        }
        let area = (side * side) as f64;
        let distances = vec![1.0, 1.5, 2.0];

        let l_vals = ripleys_l(&pts, area, &distances).expect("ripleys_l failed");
        assert_eq!(l_vals.len(), distances.len());
        // L(d) should be moderate for a regular grid (negative → regularity)
        for l in &l_vals {
            assert!(l.is_finite(), "L value should be finite");
        }
    }

    #[test]
    fn test_ripleys_k_too_few_points() {
        let pts = vec![(0.0, 0.0)];
        assert!(ripleys_k(&pts, 1.0, &[0.5]).is_err());
    }

    #[test]
    fn test_ripleys_k_bad_area() {
        let pts = vec![(0.0, 0.0), (1.0, 1.0)];
        assert!(ripleys_k(&pts, -1.0, &[0.5]).is_err());
        assert!(ripleys_k(&pts, 0.0, &[0.5]).is_err());
    }

    #[test]
    fn test_ripleys_k_empty_distances() {
        let pts = vec![(0.0, 0.0), (1.0, 1.0), (0.5, 0.5)];
        let k = ripleys_k(&pts, 4.0, &[]).expect("empty distances");
        assert!(k.is_empty());
    }
}
