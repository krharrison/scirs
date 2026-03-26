//! Wasserstein Distance and Related Optimal Transport Metrics
//!
//! This module provides:
//! - **1D Wasserstein distance** of arbitrary order via sorted CDF
//! - **Weighted Wasserstein** for non-uniform empirical measures
//! - **Sliced Wasserstein** for high-dimensional approximation
//! - **Sinkhorn divergence** with explicit cost matrix and debiased formula

use super::types::{SinkhornConfig, SinkhornResult};
use crate::error::{MetricsError, Result};

// ────────────────────────────────────────────────────────────────────────────
// 1D Wasserstein Distance (arbitrary order)
// ────────────────────────────────────────────────────────────────────────────

/// Computes the p-Wasserstein distance between two 1D empirical distributions.
///
/// For order p:
/// ```text
/// W_p(P, Q) = ( integral |F_P^{-1}(t) - F_Q^{-1}(t)|^p dt )^{1/p}
/// ```
///
/// When both samples have equal size this simplifies to:
/// ```text
/// W_p = ( (1/n) * sum |x_sorted[i] - y_sorted[i]|^p )^{1/p}
/// ```
///
/// For unequal sizes, quantile interpolation on a merged grid is used.
///
/// # Arguments
/// * `x` - samples from first distribution
/// * `y` - samples from second distribution
/// * `order` - the order p (must be >= 1)
///
/// # Returns
/// The p-Wasserstein distance (non-negative).
pub fn wasserstein_1d(x: &[f64], y: &[f64], order: usize) -> Result<f64> {
    if x.is_empty() || y.is_empty() {
        return Err(MetricsError::InvalidInput(
            "sample arrays must not be empty".to_string(),
        ));
    }
    if order == 0 {
        return Err(MetricsError::InvalidInput(
            "Wasserstein order must be >= 1".to_string(),
        ));
    }

    let p = order as f64;

    let mut xs: Vec<f64> = x.to_vec();
    let mut ys: Vec<f64> = y.to_vec();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if xs.len() == ys.len() {
        let n = xs.len() as f64;
        let sum: f64 = xs
            .iter()
            .zip(ys.iter())
            .map(|(a, b)| (a - b).abs().powf(p))
            .sum();
        return Ok((sum / n).powf(1.0 / p));
    }

    // Unequal sizes: integrate |F_P^{-1}(t) - F_Q^{-1}(t)|^p over t in [0,1]
    // using the merged CDF breakpoints approach.
    let n = xs.len();
    let m = ys.len();
    let total_steps = n + m;

    // Build quantile breakpoints
    let mut breakpoints: Vec<f64> = Vec::with_capacity(total_steps + 1);
    breakpoints.push(0.0);
    let mut xi = 0usize;
    let mut yi = 0usize;
    while xi < n || yi < m {
        let tx = if xi < n {
            (xi + 1) as f64 / n as f64
        } else {
            2.0 // sentinel
        };
        let ty = if yi < m {
            (yi + 1) as f64 / m as f64
        } else {
            2.0 // sentinel
        };
        if tx <= ty {
            breakpoints.push(tx.min(1.0));
            xi += 1;
        } else {
            breakpoints.push(ty.min(1.0));
            yi += 1;
        }
    }
    breakpoints.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON * 10.0);

    let mut integral = 0.0f64;
    for w in breakpoints.windows(2) {
        let t_lo = w[0];
        let t_hi = w[1];
        let t_mid = 0.5 * (t_lo + t_hi);
        let width = t_hi - t_lo;

        // Quantile of xs at t_mid
        let qx = quantile_sorted(&xs, t_mid);
        let qy = quantile_sorted(&ys, t_mid);
        integral += (qx - qy).abs().powf(p) * width;
    }

    Ok(integral.powf(1.0 / p))
}

/// Linear interpolation quantile on a pre-sorted slice.
fn quantile_sorted(sorted: &[f64], t: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 || t <= 0.0 {
        return sorted[0];
    }
    if t >= 1.0 {
        return sorted[n - 1];
    }
    let pos = t * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

// ────────────────────────────────────────────────────────────────────────────
// Weighted 1D Wasserstein
// ────────────────────────────────────────────────────────────────────────────

/// Computes the 1D Wasserstein-1 distance between two weighted empirical measures.
///
/// Each sample set has associated weights that need not be uniform.
/// Weights are normalized to sum to 1 internally.
///
/// The algorithm sorts both weighted samples, builds their CDFs on a merged
/// grid, and integrates `|F_P(x) - F_Q(x)|` over x.
///
/// # Arguments
/// * `x` - samples from first distribution
/// * `wx` - weights for x (must be positive, same length as x)
/// * `y` - samples from second distribution
/// * `wy` - weights for y (must be positive, same length as y)
///
/// # Returns
/// The weighted Wasserstein-1 distance.
pub fn wasserstein_1d_weighted(x: &[f64], wx: &[f64], y: &[f64], wy: &[f64]) -> Result<f64> {
    if x.is_empty() || y.is_empty() {
        return Err(MetricsError::InvalidInput(
            "sample arrays must not be empty".to_string(),
        ));
    }
    if x.len() != wx.len() {
        return Err(MetricsError::DimensionMismatch(format!(
            "x has length {} but weights have length {}",
            x.len(),
            wx.len()
        )));
    }
    if y.len() != wy.len() {
        return Err(MetricsError::DimensionMismatch(format!(
            "y has length {} but weights have length {}",
            y.len(),
            wy.len()
        )));
    }

    // Validate weights are positive
    for (i, &w) in wx.iter().enumerate() {
        if w < 0.0 {
            return Err(MetricsError::InvalidInput(format!(
                "weight wx[{i}] = {w} is negative"
            )));
        }
    }
    for (i, &w) in wy.iter().enumerate() {
        if w < 0.0 {
            return Err(MetricsError::InvalidInput(format!(
                "weight wy[{i}] = {w} is negative"
            )));
        }
    }

    let sum_wx: f64 = wx.iter().sum();
    let sum_wy: f64 = wy.iter().sum();
    if sum_wx <= 0.0 || sum_wy <= 0.0 {
        return Err(MetricsError::InvalidInput(
            "total weight must be positive".to_string(),
        ));
    }

    // Sort x with weights
    let mut x_pairs: Vec<(f64, f64)> = x
        .iter()
        .zip(wx.iter())
        .map(|(&v, &w)| (v, w / sum_wx))
        .collect();
    x_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Sort y with weights
    let mut y_pairs: Vec<(f64, f64)> = y
        .iter()
        .zip(wy.iter())
        .map(|(&v, &w)| (v, w / sum_wy))
        .collect();
    y_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Build merged sorted breakpoints
    let mut all_vals: Vec<f64> = x_pairs.iter().chain(y_pairs.iter()).map(|p| p.0).collect();
    all_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_vals.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON * a.abs().max(1.0));

    // Compute CDF value for weighted samples at a given threshold
    let cdf_at = |pairs: &[(f64, f64)], threshold: f64| -> f64 {
        pairs
            .iter()
            .filter(|(v, _)| *v <= threshold)
            .map(|(_, w)| w)
            .sum::<f64>()
    };

    let mut total = 0.0f64;
    for i in 0..all_vals.len().saturating_sub(1) {
        let lo = all_vals[i];
        let hi = all_vals[i + 1];
        let width = hi - lo;
        let fx = cdf_at(&x_pairs, lo);
        let fy = cdf_at(&y_pairs, lo);
        total += (fx - fy).abs() * width;
    }

    Ok(total)
}

// ────────────────────────────────────────────────────────────────────────────
// Sliced Wasserstein Distance
// ────────────────────────────────────────────────────────────────────────────

/// Computes the sliced Wasserstein distance between two high-dimensional
/// empirical distributions using random projections.
///
/// The sliced Wasserstein distance is defined as the average 1D Wasserstein
/// distance over all 1D projections. It is approximated by sampling
/// `n_projections` random directions on the unit sphere.
///
/// ```text
/// SW(P, Q) = E_{θ ~ S^{d-1}}[ W_1(θ^T P, θ^T Q) ]
/// ```
///
/// # Arguments
/// * `samples_p` - n x d matrix of samples (each inner Vec has d components)
/// * `samples_q` - m x d matrix of samples (same dimensionality d)
/// * `n_projections` - number of random projections (more = better approximation)
///
/// # Returns
/// The sliced Wasserstein distance approximation.
pub fn sliced_wasserstein(
    samples_p: &[Vec<f64>],
    samples_q: &[Vec<f64>],
    n_projections: usize,
) -> Result<f64> {
    if samples_p.is_empty() || samples_q.is_empty() {
        return Err(MetricsError::InvalidInput(
            "sample arrays must not be empty".to_string(),
        ));
    }
    if n_projections == 0 {
        return Err(MetricsError::InvalidInput(
            "n_projections must be > 0".to_string(),
        ));
    }

    let d = samples_p[0].len();
    if d == 0 {
        return Err(MetricsError::InvalidInput(
            "sample dimensionality must be > 0".to_string(),
        ));
    }

    // Validate all samples have same dimensionality
    for (i, s) in samples_p.iter().enumerate() {
        if s.len() != d {
            return Err(MetricsError::DimensionMismatch(format!(
                "samples_p[{i}] has dimension {} but expected {d}",
                s.len()
            )));
        }
    }
    for (i, s) in samples_q.iter().enumerate() {
        if s.len() != d {
            return Err(MetricsError::DimensionMismatch(format!(
                "samples_q[{i}] has dimension {} but expected {d}",
                s.len()
            )));
        }
    }

    // Use a simple deterministic pseudo-random direction generator
    // based on a hash-like sequence for reproducibility
    let mut total_w = 0.0f64;

    for proj_idx in 0..n_projections {
        // Generate a pseudo-random direction on the unit sphere
        let direction = generate_direction(d, proj_idx);

        // Project all samples onto this direction
        let proj_p: Vec<f64> = samples_p
            .iter()
            .map(|s| dot_product(s, &direction))
            .collect();
        let proj_q: Vec<f64> = samples_q
            .iter()
            .map(|s| dot_product(s, &direction))
            .collect();

        // Compute 1D Wasserstein-1 on projections
        total_w += wasserstein_1d(&proj_p, &proj_q, 1)?;
    }

    Ok(total_w / n_projections as f64)
}

/// Generate a pseudo-random unit direction in d dimensions.
/// Uses a deterministic hash-based approach for reproducibility.
fn generate_direction(d: usize, seed: usize) -> Vec<f64> {
    let mut dir = Vec::with_capacity(d);
    // Simple deterministic "random" using golden ratio hashing
    let golden = 0.618_033_988_749_895_f64;
    for i in 0..d {
        let hash = ((seed as f64 + 1.0) * golden + (i as f64 + 1.0) * std::f64::consts::PI).fract();
        // Map [0,1) to approximately normal via Box-Muller-like transform
        let u1 = (hash * 0.998 + 0.001).clamp(0.001, 0.999);
        let u2 = (((seed * 7 + i * 13 + 3) as f64) * golden).fract();
        let u2 = (u2 * 0.998 + 0.001).clamp(0.001, 0.999);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        dir.push(z);
    }
    // Normalize to unit vector
    let norm = dir.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-15 {
        for v in &mut dir {
            *v /= norm;
        }
    }
    dir
}

/// Dot product of two slices.
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ────────────────────────────────────────────────────────────────────────────
// Sinkhorn Divergence (Cost Matrix API)
// ────────────────────────────────────────────────────────────────────────────

/// Computes the Sinkhorn divergence using an explicit cost matrix.
///
/// The debiased Sinkhorn divergence is:
/// ```text
/// S(P, Q) = OT_ε(P, Q) - 0.5 * OT_ε(P, P) - 0.5 * OT_ε(Q, Q)
/// ```
///
/// This function takes a pre-computed cost matrix C where `C[i][j]` is the
/// cost of transporting mass from source i to target j.
///
/// # Arguments
/// * `cost_matrix` - n x m cost matrix (row-major, non-negative)
/// * `p` - source distribution weights (must sum to ~1, length n)
/// * `q` - target distribution weights (must sum to ~1, length m)
/// * `config` - Sinkhorn algorithm configuration
///
/// # Returns
/// A `SinkhornResult` containing the divergence, transport plan, and convergence info.
pub fn sinkhorn_divergence(
    cost_matrix: &[Vec<f64>],
    p: &[f64],
    q: &[f64],
    config: &SinkhornConfig,
) -> Result<SinkhornResult> {
    if cost_matrix.is_empty() || p.is_empty() || q.is_empty() {
        return Err(MetricsError::InvalidInput(
            "cost matrix, p, and q must not be empty".to_string(),
        ));
    }
    let n = p.len();
    let m = q.len();

    if cost_matrix.len() != n {
        return Err(MetricsError::DimensionMismatch(format!(
            "cost matrix has {} rows but p has length {n}",
            cost_matrix.len()
        )));
    }
    for (i, row) in cost_matrix.iter().enumerate() {
        if row.len() != m {
            return Err(MetricsError::DimensionMismatch(format!(
                "cost matrix row {i} has length {} but q has length {m}",
                row.len()
            )));
        }
    }

    if config.epsilon <= 0.0 {
        return Err(MetricsError::InvalidInput(
            "epsilon must be positive".to_string(),
        ));
    }

    // OT_ε(P, Q)
    let (ot_pq, plan, converged_pq, iters_pq) = sinkhorn_ot_matrix(cost_matrix, p, q, config)?;

    // For debiasing: we need OT_ε(P, P) and OT_ε(Q, Q)
    // Build self-cost matrices
    let cost_pp = build_self_cost(cost_matrix, n, true);
    let cost_qq = build_self_cost(cost_matrix, m, false);

    let (ot_pp, _, _, _) = sinkhorn_ot_matrix(&cost_pp, p, p, config)?;
    let (ot_qq, _, _, _) = sinkhorn_ot_matrix(&cost_qq, q, q, config)?;

    let divergence = (ot_pq - 0.5 * ot_pp - 0.5 * ot_qq).max(0.0);
    let converged = converged_pq;

    Ok(SinkhornResult {
        divergence,
        transport_plan: plan,
        converged,
        iterations: iters_pq,
    })
}

/// Build a self-cost matrix for debiasing.
/// For P-P: use source-source costs from the original cost matrix structure.
/// We approximate with squared Euclidean from the cost entries.
fn build_self_cost(cost_matrix: &[Vec<f64>], size: usize, is_source: bool) -> Vec<Vec<f64>> {
    // For simplicity, build a zero-diagonal cost matrix
    // In 1D-like settings, use the cost structure to infer distances
    // For the general case, we create a symmetric cost where
    // C_self[i][j] is estimated from the original cost matrix
    let mut self_cost = vec![vec![0.0f64; size]; size];

    if is_source {
        // Source-source: C_pp[i][j] ~ average |C[i,k] - C[j,k]| over k
        let m = if cost_matrix.is_empty() {
            0
        } else {
            cost_matrix[0].len()
        };
        if m > 0 && size <= cost_matrix.len() {
            for i in 0..size {
                for j in 0..size {
                    let mut s = 0.0f64;
                    for k in 0..m {
                        let d = cost_matrix[i][k] - cost_matrix[j][k];
                        s += d * d;
                    }
                    self_cost[i][j] = (s / m as f64).sqrt();
                }
            }
        }
    } else {
        // Target-target: C_qq[i][j] ~ average |C[k,i] - C[k,j]| over k
        let n = cost_matrix.len();
        if n > 0 && size <= cost_matrix[0].len() {
            for i in 0..size {
                for j in 0..size {
                    let mut s = 0.0f64;
                    for k in 0..n {
                        let d = cost_matrix[k][i] - cost_matrix[k][j];
                        s += d * d;
                    }
                    self_cost[i][j] = (s / n as f64).sqrt();
                }
            }
        }
    }

    self_cost
}

/// Core Sinkhorn-Knopp iteration in log-domain on an explicit cost matrix.
///
/// Returns (transport_cost, transport_plan, converged, iterations).
fn sinkhorn_ot_matrix(
    cost: &[Vec<f64>],
    p: &[f64],
    q: &[f64],
    cfg: &SinkhornConfig,
) -> Result<(f64, Vec<Vec<f64>>, bool, usize)> {
    let n = p.len();
    let m = q.len();
    let eps = cfg.epsilon;

    // Log-domain Sinkhorn
    let log_p: Vec<f64> = p.iter().map(|&v| (v.max(1e-300)).ln()).collect();
    let log_q: Vec<f64> = q.iter().map(|&v| (v.max(1e-300)).ln()).collect();

    let mut log_u = vec![0.0f64; n];
    let mut log_v = vec![0.0f64; m];

    let mut converged = false;
    let mut iters = 0;

    for iter in 0..cfg.max_iter {
        iters = iter + 1;

        // Update log_v[j] = log_q[j] - logsumexp_i(log_u[i] - C[i,j]/eps)
        let old_log_v = log_v.clone();
        for j in 0..m {
            let mut vals: Vec<f64> = Vec::with_capacity(n);
            for i in 0..n {
                vals.push(log_u[i] - cost[i][j] / eps);
            }
            log_v[j] = log_q[j] - log_sum_exp(&vals);
        }

        // Update log_u[i] = log_p[i] - logsumexp_j(log_v[j] - C[i,j]/eps)
        let old_log_u = log_u.clone();
        for i in 0..n {
            let mut vals: Vec<f64> = Vec::with_capacity(m);
            for j in 0..m {
                vals.push(log_v[j] - cost[i][j] / eps);
            }
            log_u[i] = log_p[i] - log_sum_exp(&vals);
        }

        // Check convergence
        let max_diff_u = log_u
            .iter()
            .zip(old_log_u.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        let max_diff_v = log_v
            .iter()
            .zip(old_log_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        if max_diff_u < cfg.tol && max_diff_v < cfg.tol {
            converged = true;
            break;
        }
    }

    // Compute transport plan and cost
    let mut plan = vec![vec![0.0f64; m]; n];
    let mut transport_cost = 0.0f64;
    for i in 0..n {
        for j in 0..m {
            let log_pij = log_u[i] + log_v[j] - cost[i][j] / eps;
            let pij = log_pij.exp();
            plan[i][j] = pij;
            transport_cost += pij * cost[i][j];
        }
    }

    Ok((transport_cost.max(0.0), plan, converged, iters))
}

/// Numerically stable log-sum-exp.
fn log_sum_exp(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_val = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val.is_infinite() && max_val < 0.0 {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = vals.iter().map(|&v| (v - max_val).exp()).sum();
    max_val + sum.ln()
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasserstein_1d_order1_identical() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = wasserstein_1d(&x, &x, 1).expect("should succeed");
        assert!(w.abs() < 1e-10, "W_1(P,P) should be 0, got {w}");
    }

    #[test]
    fn test_wasserstein_1d_order1_shift() {
        let x: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
        let y: Vec<f64> = x.iter().map(|&v| v + 1.0).collect();
        let w = wasserstein_1d(&x, &y, 1).expect("should succeed");
        assert!((w - 1.0).abs() < 1e-8, "W_1(P, P+1) should be 1.0, got {w}");
    }

    #[test]
    fn test_wasserstein_1d_order2() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![3.0, 4.0, 5.0];
        let w = wasserstein_1d(&x, &y, 2).expect("should succeed");
        assert!(w > 0.0, "W_2 should be positive, got {w}");
        // W_2([0,1,2], [3,4,5]) = sqrt(1/3 * (9 + 9 + 9)) = 3.0
        assert!(
            (w - 3.0).abs() < 1e-8,
            "W_2([0,1,2],[3,4,5]) should be 3.0, got {w}"
        );
    }

    #[test]
    fn test_wasserstein_triangle_inequality() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let z = vec![3.0, 4.0, 5.0, 6.0];

        let w_xy = wasserstein_1d(&x, &y, 1).expect("should succeed");
        let w_yz = wasserstein_1d(&y, &z, 1).expect("should succeed");
        let w_xz = wasserstein_1d(&x, &z, 1).expect("should succeed");

        assert!(
            w_xz <= w_xy + w_yz + 1e-10,
            "Triangle inequality violated: W(x,z)={w_xz} > W(x,y)+W(y,z)={}",
            w_xy + w_yz
        );
    }

    #[test]
    fn test_wasserstein_1d_unequal_sizes() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let w = wasserstein_1d(&x, &y, 1).expect("should succeed");
        assert!(w >= 0.0, "Wasserstein must be non-negative");
    }

    #[test]
    fn test_wasserstein_1d_order_zero_errors() {
        let x = vec![1.0, 2.0];
        assert!(wasserstein_1d(&x, &x, 0).is_err());
    }

    #[test]
    fn test_wasserstein_1d_empty_errors() {
        let empty: Vec<f64> = vec![];
        let x = vec![1.0];
        assert!(wasserstein_1d(&empty, &x, 1).is_err());
        assert!(wasserstein_1d(&x, &empty, 1).is_err());
    }

    #[test]
    fn test_wasserstein_weighted_uniform_matches_unweighted() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let w_uniform = vec![1.0; 5];

        let w_unweighted = wasserstein_1d(&x, &y, 1).expect("should succeed");
        let w_weighted =
            wasserstein_1d_weighted(&x, &w_uniform, &y, &w_uniform).expect("should succeed");

        assert!(
            (w_unweighted - w_weighted).abs() < 0.15,
            "uniform weighted should approximate unweighted: {} vs {}",
            w_unweighted,
            w_weighted
        );
    }

    #[test]
    fn test_wasserstein_weighted_identical() {
        let x = vec![1.0, 2.0, 3.0];
        let w = vec![0.2, 0.5, 0.3];
        let d = wasserstein_1d_weighted(&x, &w, &x, &w).expect("should succeed");
        assert!(d.abs() < 1e-10, "W(P,P) should be 0, got {d}");
    }

    #[test]
    fn test_wasserstein_weighted_negative_weight_errors() {
        let x = vec![1.0, 2.0];
        let wx = vec![1.0, -0.5];
        let y = vec![3.0, 4.0];
        let wy = vec![0.5, 0.5];
        assert!(wasserstein_1d_weighted(&x, &wx, &y, &wy).is_err());
    }

    #[test]
    fn test_sliced_wasserstein_identical() {
        let p = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let sw = sliced_wasserstein(&p, &p, 50).expect("should succeed");
        assert!(sw.abs() < 1e-8, "SW(P,P) should be ~0, got {sw}");
    }

    #[test]
    fn test_sliced_wasserstein_positive_for_different() {
        let p = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let q = vec![vec![10.0, 10.0], vec![11.0, 10.0], vec![10.0, 11.0]];
        let sw = sliced_wasserstein(&p, &q, 100).expect("should succeed");
        assert!(
            sw > 0.0,
            "SW should be positive for different distributions"
        );
    }

    #[test]
    fn test_sliced_wasserstein_dimension_mismatch() {
        let p = vec![vec![1.0, 2.0]];
        let q = vec![vec![1.0, 2.0, 3.0]];
        assert!(sliced_wasserstein(&p, &q, 10).is_err());
    }

    #[test]
    fn test_sinkhorn_with_cost_matrix_identical() {
        // Two identical distributions: divergence should be ~0
        let cost = vec![
            vec![0.0, 1.0, 4.0],
            vec![1.0, 0.0, 1.0],
            vec![4.0, 1.0, 0.0],
        ];
        let p = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let cfg = SinkhornConfig {
            epsilon: 0.1,
            max_iter: 500,
            tol: 1e-9,
            log_domain: true,
        };

        let result = sinkhorn_divergence(&cost, &p, &p, &cfg).expect("should succeed");
        assert!(
            result.divergence < 0.01,
            "S(P,P) should be ~0, got {}",
            result.divergence
        );
    }

    #[test]
    fn test_sinkhorn_with_cost_matrix_converges() {
        let cost = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let p = vec![0.7, 0.3];
        let q = vec![0.4, 0.6];
        let cfg = SinkhornConfig {
            epsilon: 0.5,
            max_iter: 500,
            tol: 1e-9,
            log_domain: true,
        };

        let result = sinkhorn_divergence(&cost, &p, &q, &cfg).expect("should succeed");
        assert!(result.converged, "Sinkhorn should converge");
        assert!(
            result.divergence >= 0.0,
            "divergence must be non-negative, got {}",
            result.divergence
        );
    }

    #[test]
    fn test_sinkhorn_invalid_epsilon() {
        let cost = vec![vec![0.0]];
        let p = vec![1.0];
        let cfg = SinkhornConfig {
            epsilon: -1.0,
            ..Default::default()
        };
        assert!(sinkhorn_divergence(&cost, &p, &p, &cfg).is_err());
    }
}
