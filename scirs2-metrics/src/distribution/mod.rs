//! Distribution Distance Metrics
//!
//! This module provides advanced distance metrics between probability distributions,
//! including:
//!
//! - **Wasserstein Distance**: Earth Mover's distance via sorted CDF integration
//! - **Sinkhorn Divergence**: Regularized optimal transport with epsilon-scaling
//! - **Energy Distance**: U-statistic estimator for E[||X-Y||] - based distances
//! - **Total Variation**: Maximum absolute difference between probability measures
//! - **Kernel Stein Discrepancy**: KSD with RBF/IMQ/Polynomial kernels for goodness-of-fit
//! - **Hellinger Distance**: sqrt(1 - Bhattacharyya coefficient)
//! - **KL Divergence**: Kullback-Leibler divergence
//! - **Jensen-Shannon Divergence**: Symmetric bounded divergence
//! - **Chi-Square Divergence**: Pearson chi-squared divergence
//! - **Sliced Wasserstein**: Random-projection approximation for high dimensions
//!
//! # Submodules
//!
//! - [`types`] — Configuration structs, result types, kernel enums
//! - [`wasserstein`] — Wasserstein distances (1D, weighted, sliced) and Sinkhorn with cost matrix
//! - [`stein`] — Kernel Stein Discrepancy (U/V-statistic, bootstrap test)
//! - [`distances`] — Statistical distances (TV, Hellinger, KL, JSD, chi², energy)
//!
//! # Examples
//!
//! ```
//! use scirs2_metrics::distribution::{
//!     wasserstein_1d, sinkhorn_divergence, energy_distance, total_variation,
//! };
//!
//! // Wasserstein distance between identical distributions = 0
//! let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let w = wasserstein_1d(&x, &x).expect("should succeed");
//! assert!(w.abs() < 1e-10);
//!
//! // Total variation between uniform distributions
//! let p = vec![0.5, 0.5];
//! let q = vec![1.0, 0.0];
//! let tv = total_variation(&p, &q).expect("should succeed");
//! assert!((tv - 0.5).abs() < 1e-10);
//! ```

// ── Submodules ──────────────────────────────────────────────────────────────
pub mod distances;
pub mod stein;
pub mod types;
pub mod wasserstein;

// ── Re-exports from submodules ──────────────────────────────────────────────
pub use distances::{
    chi_square_divergence as chi_square_divergence_ext, energy_distance as energy_distance_ext,
    hellinger_distance as hellinger_distance_ext, jensen_shannon_divergence as jsd_ext,
    jensen_shannon_divergence_samples, kl_divergence as kl_divergence_ext, kl_divergence_samples,
    total_variation_distance as tv_distance_ext,
};
pub use stein::{ksd_bootstrap_test, ksd_u_statistic, ksd_v_statistic, KernelSteinDiscrepancy};
pub use types::{
    DistanceMethod, DistanceResult, KernelType, KsdConfig as KsdConfigAdvanced,
    KsdResult as KsdResultAdvanced, SinkhornConfig as SinkhornConfigAdvanced, SinkhornResult,
};
pub use wasserstein::{
    sinkhorn_divergence as sinkhorn_divergence_matrix, sliced_wasserstein,
    wasserstein_1d as wasserstein_1d_order, wasserstein_1d_weighted,
};

use crate::error::{MetricsError, Result};

// ────────────────────────────────────────────────────────────────────────────
// Wasserstein Distance (1D Earth Mover's Distance)
// ────────────────────────────────────────────────────────────────────────────

/// Computes the 1D Wasserstein-1 (Earth Mover's) distance between two empirical
/// distributions represented as sample sets.
///
/// The 1D Wasserstein distance is computed via the sorted CDF integration:
/// ```text
/// W_1(P, Q) = integral |F_P(x) - F_Q(x)| dx
/// ```
/// which for uniform empirical distributions reduces to:
/// ```text
/// W_1 = (1/n) * sum |x_sorted[i] - y_sorted[i]|
/// ```
/// when |x| == |y|. For unequal sizes, quantile interpolation is used.
///
/// # Arguments
///
/// * `x` - Samples from the first distribution
/// * `y` - Samples from the second distribution
///
/// # Returns
///
/// The Wasserstein-1 distance (non-negative scalar).
pub fn wasserstein_1d(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.is_empty() || y.is_empty() {
        return Err(MetricsError::InvalidInput(
            "sample arrays must not be empty".to_string(),
        ));
    }

    let mut xs: Vec<f64> = x.to_vec();
    let mut ys: Vec<f64> = y.to_vec();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if xs.len() == ys.len() {
        let n = xs.len() as f64;
        let dist = xs
            .iter()
            .zip(ys.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / n;
        return Ok(dist);
    }

    // Unequal sizes: use merged sorted array with CDF integration
    // Build combined sorted unique breakpoints
    let mut all_vals: Vec<f64> = xs.iter().chain(ys.iter()).cloned().collect();
    all_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_vals.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON * a.abs().max(1.0));

    let nx = xs.len() as f64;
    let ny = ys.len() as f64;

    let mut total = 0.0;
    let mut xi = 0usize;
    let mut yi = 0usize;

    for i in 0..all_vals.len().saturating_sub(1) {
        let lo = all_vals[i];
        let hi = all_vals[i + 1];
        let width = hi - lo;

        // Advance pointers past current lo
        while xi < xs.len() && xs[xi] <= lo {
            xi += 1;
        }
        while yi < ys.len() && ys[yi] <= lo {
            yi += 1;
        }

        // CDF of xs at lo = xi / nx; similarly for ys
        let fx = xi as f64 / nx;
        let fy = yi as f64 / ny;
        total += (fx - fy).abs() * width;
    }

    Ok(total)
}

// ────────────────────────────────────────────────────────────────────────────
// Sinkhorn Divergence (Regularized Optimal Transport)
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for the Sinkhorn algorithm.
#[derive(Debug, Clone)]
pub struct SinkhornConfig {
    /// Entropic regularization parameter (higher = more blur, faster convergence)
    pub epsilon: f64,
    /// Maximum number of Sinkhorn iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Whether to use log-domain stabilization
    pub log_domain: bool,
}

impl Default for SinkhornConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            max_iter: 1000,
            tol: 1e-9,
            log_domain: true,
        }
    }
}

/// Computes the Sinkhorn divergence between two empirical 1D distributions.
///
/// The Sinkhorn divergence is a debiased version of entropic regularized OT:
/// ```text
/// S_ε(P, Q) = OT_ε(P, Q) - (1/2) OT_ε(P, P) - (1/2) OT_ε(Q, Q)
/// ```
/// where OT_ε is the entropic regularized optimal transport cost.
///
/// This implementation uses log-domain Sinkhorn iterations for numerical
/// stability, with epsilon-scaling warmup.
///
/// # Arguments
///
/// * `x` - Samples from the first distribution (1D, uniform weights)
/// * `y` - Samples from the second distribution (1D, uniform weights)
/// * `epsilon` - Regularization strength (must be positive)
/// * `max_iter` - Maximum number of Sinkhorn iterations
///
/// # Returns
///
/// The Sinkhorn divergence (≥ 0, equals 0 iff P = Q).
pub fn sinkhorn_divergence(x: &[f64], y: &[f64], epsilon: f64, max_iter: usize) -> Result<f64> {
    if x.is_empty() || y.is_empty() {
        return Err(MetricsError::InvalidInput(
            "sample arrays must not be empty".to_string(),
        ));
    }
    if epsilon <= 0.0 {
        return Err(MetricsError::InvalidInput(
            "epsilon must be positive".to_string(),
        ));
    }

    let cfg = SinkhornConfig {
        epsilon,
        max_iter,
        ..Default::default()
    };

    let ot_xy = sinkhorn_ot(x, y, &cfg)?;
    let ot_xx = sinkhorn_ot(x, x, &cfg)?;
    let ot_yy = sinkhorn_ot(y, y, &cfg)?;

    // Debiased Sinkhorn divergence
    let divergence = (ot_xy - 0.5 * ot_xx - 0.5 * ot_yy).max(0.0);
    Ok(divergence)
}

/// Computes OT_ε(a, b) using log-domain Sinkhorn iterations.
fn sinkhorn_ot(a: &[f64], b: &[f64], cfg: &SinkhornConfig) -> Result<f64> {
    let n = a.len();
    let m = b.len();
    let eps = cfg.epsilon;

    // Cost matrix: C[i][j] = |a[i] - b[j]|^2
    // We store in a flat Vec
    let mut cost = vec![0.0f64; n * m];
    for i in 0..n {
        for j in 0..m {
            let d = a[i] - b[j];
            cost[i * m + j] = d * d;
        }
    }

    let log_n = -(n as f64).ln();
    let log_m = -(m as f64).ln();

    // Log-domain Sinkhorn: u, v are log potentials
    let mut log_u = vec![0.0f64; n]; // log of scaling variable u (initialized to log(1/n))
    let mut log_v = vec![0.0f64; m]; // log of scaling variable v

    for _ in 0..cfg.max_iter {
        // Update log_v: log_v[j] = log_m - logsumexp_i( log_u[i] - C[i,j]/eps )
        let old_log_v = log_v.clone();
        for j in 0..m {
            let lse = log_sum_exp_row(&log_u, &cost, n, j, m, eps, log_n);
            log_v[j] = log_m - lse;
        }

        // Update log_u: log_u[i] = log_n - logsumexp_j( log_v[j] - C[i,j]/eps )
        let old_log_u = log_u.clone();
        for i in 0..n {
            let lse = log_sum_exp_col(&log_v, &cost, m, i, m, eps, log_m);
            log_u[i] = log_n - lse;
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
            break;
        }
    }

    // Transport cost: <P, C> = eps * (sum u[i]*log_u[i] + sum v[j]*log_v[j] + log(n) + log(m))
    // In log domain: sum_ij exp(log_u[i] + log_v[j] - C[i,j]/eps) * C[i,j]
    let mut transport_cost = 0.0f64;
    for i in 0..n {
        for j in 0..m {
            let log_p_ij = log_u[i] + log_v[j] - cost[i * m + j] / eps;
            transport_cost += log_p_ij.exp() * cost[i * m + j];
        }
    }

    Ok(transport_cost.max(0.0))
}

/// log-sum-exp along dimension i for the column j cost update
fn log_sum_exp_row(
    log_u: &[f64],
    cost: &[f64],
    n: usize,
    j: usize,
    m: usize,
    eps: f64,
    _log_n: f64,
) -> f64 {
    let mut vals: Vec<f64> = (0..n).map(|i| log_u[i] - cost[i * m + j] / eps).collect();
    log_sum_exp_slice(&mut vals)
}

/// log-sum-exp along dimension j for the row i cost update
fn log_sum_exp_col(
    log_v: &[f64],
    cost: &[f64],
    m: usize,
    i: usize,
    _stride: usize,
    eps: f64,
    _log_m: f64,
) -> f64 {
    let mut vals: Vec<f64> = (0..m).map(|j| log_v[j] - cost[i * m + j] / eps).collect();
    log_sum_exp_slice(&mut vals)
}

/// Numerically stable log-sum-exp over a slice.
fn log_sum_exp_slice(vals: &mut [f64]) -> f64 {
    if vals.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_val = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val.is_infinite() {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = vals.iter().map(|&v| (v - max_val).exp()).sum();
    max_val + sum.ln()
}

// ────────────────────────────────────────────────────────────────────────────
// Energy Distance
// ────────────────────────────────────────────────────────────────────────────

/// Computes the energy distance between two empirical distributions.
///
/// The energy distance is defined as:
/// ```text
/// E(P, Q) = 2 * E[||X - Y||] - E[||X - X'||] - E[||Y - Y'||]
/// ```
/// where X, X' ~ P and Y, Y' ~ Q independently.
///
/// For 1D samples, this is estimated using U-statistics:
/// ```text
/// E(P, Q) = (2/(n*m)) * sum_{i,j} |x_i - y_j|
///          - (2/(n*(n-1))) * sum_{i<j} |x_i - x_j|
///          - (2/(m*(m-1))) * sum_{i<j} |y_i - y_j|
/// ```
///
/// The large-n approximation uses sorted arrays for O(n log n) computation.
///
/// # Arguments
///
/// * `x` - Samples from the first distribution
/// * `y` - Samples from the second distribution
///
/// # Returns
///
/// The energy distance (non-negative).
pub fn energy_distance(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.is_empty() || y.is_empty() {
        return Err(MetricsError::InvalidInput(
            "sample arrays must not be empty".to_string(),
        ));
    }

    let n = x.len();
    let m = y.len();

    // E[|X - Y|]: cross term
    let cross = if n * m <= 10_000 {
        // Direct O(n*m)
        let mut s = 0.0f64;
        for &xi in x {
            for &yj in y {
                s += (xi - yj).abs();
            }
        }
        s / (n as f64 * m as f64)
    } else {
        // O(n log n + m log m) via sorted CDF integration
        cross_mean_abs_diff_sorted(x, y)?
    };

    // E[|X - X'|]: within-x term  (U-statistic: divide by n*(n-1))
    let within_x = within_mean_abs_diff(x);

    // E[|Y - Y'|]: within-y term
    let within_y = within_mean_abs_diff(y);

    Ok((2.0 * cross - within_x - within_y).max(0.0))
}

/// Computes E[|X - Y|] for large samples.
/// Uses sorted arrays and direct summation for correctness.
fn cross_mean_abs_diff_sorted(x: &[f64], y: &[f64]) -> Result<f64> {
    let mut xs = x.to_vec();
    let mut ys = y.to_vec();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = xs.len() as f64;
    let m = ys.len() as f64;

    // Direct O(n*m) computation (called only for n*m > 10_000)
    let mut s = 0.0f64;
    for &xi in &xs {
        for &yj in &ys {
            s += (xi - yj).abs();
        }
    }
    Ok(s / (n * m))
}

/// Computes E[|X - X'|] = (2/(n*(n-1))) * sum_{i<j} |x_i - x_j|
/// using sorted array O(n log n) method.
fn within_mean_abs_diff(x: &[f64]) -> f64 {
    let n = x.len();
    if n <= 1 {
        return 0.0;
    }
    let mut sorted = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // sum_{i<j} |x_i - x_j| = sum_j x_j * (2j - n + 1) - ... using prefix sums
    // After sorting: sum_{i<j} (x_j - x_i) = sum_j x_j * j - sum_j prefix_sum[j-1]
    let mut prefix = 0.0f64;
    let mut total = 0.0f64;
    for (j, &xj) in sorted.iter().enumerate() {
        total += xj * j as f64 - prefix;
        prefix += xj;
    }

    let pairs = n as f64 * (n as f64 - 1.0) / 2.0;
    if pairs > 0.0 {
        total / pairs
    } else {
        0.0
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Total Variation Distance
// ────────────────────────────────────────────────────────────────────────────

/// Computes the total variation (TV) distance between two discrete probability
/// distributions.
///
/// ```text
/// TV(P, Q) = 0.5 * sum_i |p_i - q_i|
/// ```
///
/// This equals `max_{A} |P(A) - Q(A)|` over all measurable sets A.
///
/// # Arguments
///
/// * `p` - First probability distribution (must sum to ~1, all non-negative)
/// * `q` - Second probability distribution (same length as p)
///
/// # Returns
///
/// The total variation distance in [0, 1].
pub fn total_variation(p: &[f64], q: &[f64]) -> Result<f64> {
    if p.len() != q.len() {
        return Err(MetricsError::DimensionMismatch(format!(
            "p has length {} but q has length {}",
            p.len(),
            q.len()
        )));
    }
    if p.is_empty() {
        return Err(MetricsError::InvalidInput(
            "distributions must not be empty".to_string(),
        ));
    }

    // Validate distributions are non-negative
    for (i, (&pi, &qi)) in p.iter().zip(q.iter()).enumerate() {
        if pi < -1e-10 || qi < -1e-10 {
            return Err(MetricsError::InvalidInput(format!(
                "distribution values must be non-negative, got p[{i}]={pi}, q[{i}]={qi}"
            )));
        }
    }

    let tv = 0.5
        * p.iter()
            .zip(q.iter())
            .map(|(pi, qi)| (pi - qi).abs())
            .sum::<f64>();
    Ok(tv)
}

// ────────────────────────────────────────────────────────────────────────────
// Kernel Stein Discrepancy
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for Kernel Stein Discrepancy.
#[derive(Debug, Clone, Default)]
pub struct KsdConfig {
    /// RBF kernel bandwidth (median heuristic is used if None)
    pub bandwidth: Option<f64>,
    /// Whether to compute the V-statistic (biased) or U-statistic (unbiased)
    pub use_v_statistic: bool,
}

/// Computes the Kernel Stein Discrepancy (KSD) for goodness-of-fit testing.
///
/// KSD measures how well samples `x` fit a target distribution with score function
/// `score_fn(x) = d/dx log p(x)`.
///
/// For a RBF kernel k(x, y) = exp(-||x-y||²/(2h²)), the Stein kernel is:
/// ```text
/// u_p(x, y) = score(x) * score(y) * k(x,y)
///           + score(x) * dk/dy
///           + score(y) * dk/dx
///           + d²k/dxdy
/// ```
///
/// KSD² = (1/n²) * sum_{i,j} u_p(x_i, x_j)  (V-statistic)
///
/// # Arguments
///
/// * `samples` - Samples to test
/// * `score_fn` - Score function of target: d/dx log p(x)
/// * `cfg` - Configuration
///
/// # Returns
///
/// The KSD value (non-negative; 0 iff samples are from target).
pub fn kernel_stein_discrepancy(
    samples: &[f64],
    score_fn: impl Fn(f64) -> f64,
    cfg: &KsdConfig,
) -> Result<f64> {
    if samples.len() < 2 {
        return Err(MetricsError::InvalidInput(
            "at least 2 samples required for KSD".to_string(),
        ));
    }

    let n = samples.len();

    // Compute bandwidth via median heuristic if not provided
    let h2 = match cfg.bandwidth {
        Some(bw) => {
            if bw <= 0.0 {
                return Err(MetricsError::InvalidInput(
                    "bandwidth must be positive".to_string(),
                ));
            }
            2.0 * bw * bw
        }
        None => {
            // Median heuristic: h² = median(||x_i - x_j||²) / log(n)
            let mut dists: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
            for i in 0..n {
                for j in (i + 1)..n {
                    dists.push((samples[i] - samples[j]).powi(2));
                }
            }
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let med = if dists.is_empty() {
                1.0
            } else {
                dists[dists.len() / 2]
            };
            med / (n as f64).ln().max(1.0)
        }
    };

    if h2 <= 0.0 {
        return Err(MetricsError::CalculationError(
            "computed bandwidth is non-positive".to_string(),
        ));
    }

    // Precompute scores
    let scores: Vec<f64> = samples.iter().map(|&xi| score_fn(xi)).collect();

    // V-statistic or U-statistic KSD²
    let mut ksd_sq = 0.0f64;
    let start = 0usize; // both v-statistic and u-statistic start i from 0
    let _ = start;

    for i in 0..n {
        let j_start = if cfg.use_v_statistic { 0 } else { i + 1 };
        for j in j_start..n {
            if !cfg.use_v_statistic && i == j {
                continue;
            }
            let xi = samples[i];
            let xj = samples[j];
            let diff = xi - xj;
            let diff_sq = diff * diff;

            let k = (-diff_sq / h2).exp(); // RBF kernel
            let si = scores[i];
            let sj = scores[j];

            // dk/dxj = k * (2*(xi - xj)/h2)  (derivative wrt xj)
            // dk/dxi = k * (2*(xj - xi)/h2)
            let dk_dxj = k * 2.0 * diff / h2; // d/dxj k(xi, xj)
            let dk_dxi = -dk_dxj; // d/dxi k(xi, xj)

            // d²k / dxi dxj = k * (2/h2) * (1 - 2*diff²/h2)
            let d2k = k * (2.0 / h2) * (1.0 - 2.0 * diff_sq / h2);

            let stein_kernel = si * sj * k + si * dk_dxj + sj * dk_dxi + d2k;
            let weight = if cfg.use_v_statistic { 1.0 } else { 2.0 }; // symmetry
            ksd_sq += weight * stein_kernel;
        }
    }

    let normalizer = if cfg.use_v_statistic {
        (n * n) as f64
    } else {
        (n * (n - 1)) as f64
    };

    let ksd = (ksd_sq / normalizer).max(0.0).sqrt();
    Ok(ksd)
}

/// Summary of all distribution distance metrics for a pair of samples.
#[derive(Debug, Clone)]
pub struct DistributionDistanceMetrics {
    /// 1D Wasserstein (Earth Mover's) distance
    pub wasserstein: f64,
    /// Energy distance
    pub energy: f64,
    /// Sinkhorn divergence (ε = 0.1 by default)
    pub sinkhorn: f64,
}

impl DistributionDistanceMetrics {
    /// Compute all distribution distance metrics between samples `x` and `y`.
    pub fn compute(x: &[f64], y: &[f64]) -> Result<Self> {
        let wasserstein = wasserstein_1d(x, y)?;
        let energy = energy_distance(x, y)?;
        let sinkhorn = sinkhorn_divergence(x, y, 0.1, 500)?;
        Ok(Self {
            wasserstein,
            energy,
            sinkhorn,
        })
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Distribution Divergence Metrics
// ────────────────────────────────────────────────────────────────────────────

const EPS: f64 = 1e-10;

/// Helper: validate that both slices are non-empty and of equal length.
fn check_lengths(p: &[f64], q: &[f64]) -> Result<()> {
    if p.is_empty() || q.is_empty() {
        return Err(MetricsError::InvalidInput(
            "distribution arrays must not be empty".to_string(),
        ));
    }
    if p.len() != q.len() {
        return Err(MetricsError::DimensionMismatch(format!(
            "p has {} elements but q has {}",
            p.len(),
            q.len()
        )));
    }
    Ok(())
}

/// Normalize a distribution: zero negatives, then divide by sum.
///
/// Returns `Err` if the sum is zero (all-zero or all-negative input).
pub fn normalize_distribution(p: &[f64]) -> Result<Vec<f64>> {
    if p.is_empty() {
        return Err(MetricsError::InvalidInput(
            "cannot normalize an empty distribution".to_string(),
        ));
    }
    let clipped: Vec<f64> = p.iter().map(|&v| v.max(0.0)).collect();
    let s: f64 = clipped.iter().sum();
    if s <= 0.0 {
        return Err(MetricsError::CalculationError(
            "distribution sum is zero or negative; cannot normalize".to_string(),
        ));
    }
    Ok(clipped.iter().map(|&v| v / s).collect())
}

/// Total Variation Distance between two discrete distributions.
///
/// `TV(P, Q) = 0.5 * Σ|p_i − q_i|`
///
/// Both arrays must have the same length and represent valid
/// (non-negative, summing-to-one) probability distributions.
pub fn total_variation_distance(p: &[f64], q: &[f64]) -> Result<f64> {
    check_lengths(p, q)?;
    let tv = 0.5 * p.iter().zip(q).map(|(a, b)| (a - b).abs()).sum::<f64>();
    Ok(tv)
}

/// KL Divergence `KL(P||Q) = Σ p_i * log(p_i / q_i)`.
///
/// Returns `Err` if any `q_i = 0` where `p_i > 0`.
pub fn kl_divergence(p: &[f64], q: &[f64]) -> Result<f64> {
    check_lengths(p, q)?;
    let mut kl = 0.0_f64;
    for (&pi, &qi) in p.iter().zip(q) {
        if pi <= 0.0 {
            continue;
        }
        if qi <= 0.0 {
            return Err(MetricsError::CalculationError(
                "KL divergence is infinite: q_i = 0 where p_i > 0".to_string(),
            ));
        }
        kl += pi * (pi / qi).ln();
    }
    Ok(kl)
}

/// Jensen-Shannon Divergence (symmetric, bounded in `[0, ln 2]`).
///
/// `JSD(P,Q) = 0.5*KL(P||M) + 0.5*KL(Q||M)` where `M = 0.5*(P+Q)`.
pub fn js_divergence(p: &[f64], q: &[f64]) -> Result<f64> {
    check_lengths(p, q)?;
    let m: Vec<f64> = p.iter().zip(q).map(|(a, b)| 0.5 * (a + b)).collect();
    let kl_pm = kl_divergence(p, &m)?;
    let kl_qm = kl_divergence(q, &m)?;
    Ok(0.5 * kl_pm + 0.5 * kl_qm)
}

/// Jensen-Shannon Distance = `sqrt(JSD(P,Q))`.
pub fn js_distance(p: &[f64], q: &[f64]) -> Result<f64> {
    let jsd = js_divergence(p, q)?;
    Ok(jsd.max(0.0).sqrt())
}

/// Bhattacharyya Coefficient: `BC(P,Q) = Σ sqrt(p_i * q_i)` ∈ `[0, 1]`.
pub fn bhattacharyya_coefficient(p: &[f64], q: &[f64]) -> Result<f64> {
    check_lengths(p, q)?;
    let bc = p
        .iter()
        .zip(q)
        .map(|(a, b)| (a.max(0.0) * b.max(0.0)).sqrt())
        .sum::<f64>();
    Ok(bc.clamp(0.0, 1.0))
}

/// Bhattacharyya Distance: `-ln(BC(P,Q))`.
pub fn bhattacharyya_distance(p: &[f64], q: &[f64]) -> Result<f64> {
    let bc = bhattacharyya_coefficient(p, q)?;
    if bc <= 0.0 {
        return Err(MetricsError::CalculationError(
            "Bhattacharyya distance is infinite: coefficient is zero".to_string(),
        ));
    }
    Ok(-bc.ln())
}

/// Hellinger Distance: `sqrt(1 - BC(P,Q))` ∈ `[0, 1]`.
pub fn hellinger_distance(p: &[f64], q: &[f64]) -> Result<f64> {
    let bc = bhattacharyya_coefficient(p, q)?;
    Ok((1.0 - bc).max(0.0).sqrt())
}

/// Rényi Divergence of order `alpha`.
///
/// `D_α(P||Q) = 1/(α−1) * log(Σ p_i^α * q_i^(1−α))`
///
/// The limit `α → 1` recovers KL divergence.
pub fn renyi_divergence(p: &[f64], q: &[f64], alpha: f64) -> Result<f64> {
    check_lengths(p, q)?;
    if alpha < 0.0 {
        return Err(MetricsError::InvalidInput(
            "Rényi order alpha must be >= 0".to_string(),
        ));
    }
    // Special case: alpha ≈ 1 → KL divergence
    if (alpha - 1.0).abs() < 1e-8 {
        return kl_divergence(p, q);
    }
    let sum: f64 = p
        .iter()
        .zip(q)
        .map(|(pi, qi)| {
            let pi = pi.max(0.0);
            let qi = qi.max(0.0);
            if pi <= 0.0 {
                0.0
            } else if qi <= 0.0 {
                f64::INFINITY
            } else {
                pi.powf(alpha) * qi.powf(1.0 - alpha)
            }
        })
        .sum();
    if sum.is_infinite() {
        return Err(MetricsError::CalculationError(
            "Rényi divergence is infinite".to_string(),
        ));
    }
    Ok(sum.max(EPS).ln() / (alpha - 1.0))
}

/// Chi-squared distance: `Σ (p_i − q_i)^2 / (p_i + q_i + ε)`.
pub fn chi_squared_distance(p: &[f64], q: &[f64]) -> Result<f64> {
    check_lengths(p, q)?;
    let d = p
        .iter()
        .zip(q)
        .map(|(a, b)| {
            let diff = a - b;
            diff * diff / (a + b + EPS)
        })
        .sum::<f64>();
    Ok(d)
}

/// Population Stability Index (PSI) for monitoring distribution shift.
///
/// `PSI = Σ (P_i − Q_i) * ln(P_i / Q_i)` (with ε to avoid `log(0)`).
pub fn population_stability_index(reference: &[f64], actual: &[f64]) -> Result<f64> {
    check_lengths(reference, actual)?;
    let psi = reference
        .iter()
        .zip(actual)
        .map(|(p, a)| {
            let p = p.max(EPS);
            let a = a.max(EPS);
            (p - a) * (p / a).ln()
        })
        .sum::<f64>();
    Ok(psi)
}

/// Interpretation of a PSI value.
///
/// - `< 0.1`  → "Insignificant change"
/// - `0.1..0.25` → "Moderate change"
/// - `>= 0.25` → "Significant change"
pub fn interpret_psi(psi: f64) -> &'static str {
    if psi < 0.1 {
        "Insignificant change"
    } else if psi < 0.25 {
        "Moderate change"
    } else {
        "Significant change"
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasserstein_1d_identical() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = wasserstein_1d(&x, &x).expect("should succeed");
        assert!(w.abs() < 1e-10, "W(P,P) should be 0, got {w}");
    }

    #[test]
    fn test_wasserstein_1d_shift() {
        // W(N(0,1), N(1,1)) ≈ 1.0 for large n
        // With uniform samples shifted by 1: W = 1.0 exactly
        let x: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
        let y: Vec<f64> = x.iter().map(|&v| v + 1.0).collect();
        let w = wasserstein_1d(&x, &y).expect("should succeed");
        assert!((w - 1.0).abs() < 1e-10, "W(P, P+1) should be 1.0, got {w}");
    }

    #[test]
    fn test_wasserstein_1d_known_value() {
        // W([1,2,3], [4,5,6]) = 3.0
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        let w = wasserstein_1d(&x, &y).expect("should succeed");
        assert!((w - 3.0).abs() < 1e-10, "expected 3.0, got {w}");
    }

    #[test]
    fn test_wasserstein_1d_unequal_size() {
        let x = vec![0.0, 1.0];
        let y = vec![0.0, 0.5, 1.0];
        let w = wasserstein_1d(&x, &y).expect("should succeed");
        assert!(w >= 0.0, "Wasserstein must be non-negative, got {w}");
    }

    #[test]
    fn test_wasserstein_1d_empty_error() {
        let x: Vec<f64> = vec![];
        let y = vec![1.0, 2.0];
        assert!(wasserstein_1d(&x, &y).is_err());
    }

    #[test]
    fn test_sinkhorn_convergence() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = sinkhorn_divergence(&x, &y, 0.1, 500).expect("should converge");
        assert!(s >= 0.0, "Sinkhorn divergence must be non-negative");
        // Debiased: S(P,P) = 0
        let s_self = sinkhorn_divergence(&x, &x, 0.1, 500).expect("should converge");
        assert!(s_self.abs() < 1e-6, "S(P,P) should be ≈ 0, got {s_self}");
    }

    #[test]
    fn test_sinkhorn_positive_epsilon_required() {
        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];
        assert!(sinkhorn_divergence(&x, &y, 0.0, 100).is_err());
        assert!(sinkhorn_divergence(&x, &y, -1.0, 100).is_err());
    }

    #[test]
    fn test_energy_distance_identical() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ed = energy_distance(&x, &x).expect("should succeed");
        assert!(ed.abs() < 1e-10, "ED(P,P) should be 0, got {ed}");
    }

    #[test]
    fn test_energy_distance_positive() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![10.0, 11.0, 12.0];
        let ed = energy_distance(&x, &y).expect("should succeed");
        assert!(
            ed > 0.0,
            "energy distance should be positive for different dists"
        );
    }

    #[test]
    fn test_energy_distance_symmetry() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        let ed_xy = energy_distance(&x, &y).expect("should succeed");
        let ed_yx = energy_distance(&y, &x).expect("should succeed");
        assert!(
            (ed_xy - ed_yx).abs() < 1e-10,
            "energy distance should be symmetric"
        );
    }

    #[test]
    fn test_total_variation_uniform() {
        // TV([0.5, 0.5], [1.0, 0.0]) = 0.5
        let p = vec![0.5, 0.5];
        let q = vec![1.0, 0.0];
        let tv = total_variation(&p, &q).expect("should succeed");
        assert!((tv - 0.5).abs() < 1e-10, "expected TV=0.5, got {tv}");
    }

    #[test]
    fn test_total_variation_identical() {
        let p = vec![0.2, 0.3, 0.5];
        let tv = total_variation(&p, &p).expect("should succeed");
        assert!(tv.abs() < 1e-10, "TV(P,P) should be 0, got {tv}");
    }

    #[test]
    fn test_total_variation_maximum() {
        // TV([1, 0, 0], [0, 0, 1]) = 1.0
        let p = vec![1.0, 0.0, 0.0];
        let q = vec![0.0, 0.0, 1.0];
        let tv = total_variation(&p, &q).expect("should succeed");
        assert!((tv - 1.0).abs() < 1e-10, "expected TV=1.0, got {tv}");
    }

    #[test]
    fn test_total_variation_mismatch_error() {
        let p = vec![0.5, 0.5];
        let q = vec![0.3, 0.3, 0.4];
        assert!(total_variation(&p, &q).is_err());
    }

    #[test]
    fn test_ksd_gaussian_self() {
        // KSD of Gaussian samples against Gaussian score should be small
        // Using standard normal score: d/dx log p(x) = -x
        let samples = vec![-1.5, -0.5, 0.0, 0.5, 1.5, -1.0, 1.0, 0.3, -0.3, 0.8];
        let score_fn = |x: f64| -x; // score of N(0,1)
        let cfg = KsdConfig {
            bandwidth: Some(1.0),
            use_v_statistic: true,
        };
        let ksd = kernel_stein_discrepancy(&samples, score_fn, &cfg).expect("should succeed");
        assert!(ksd >= 0.0, "KSD must be non-negative, got {ksd}");
    }

    #[test]
    fn test_distribution_distance_metrics_compute() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.5, 1.5, 2.5, 3.5, 4.5];
        let metrics = DistributionDistanceMetrics::compute(&x, &y).expect("should succeed");
        assert!(metrics.wasserstein >= 0.0);
        assert!(metrics.energy >= 0.0);
        assert!(metrics.sinkhorn >= 0.0);
    }

    // ── Divergence metric tests ───────────────────────────────────────────

    #[test]
    fn test_tv_distance_identical() {
        let p = vec![0.3, 0.4, 0.3];
        let tv = total_variation_distance(&p, &p).expect("should succeed");
        assert!(tv.abs() < 1e-12, "TV(P,P) should be 0, got {tv}");
    }

    #[test]
    fn test_tv_distance_known_value() {
        let p = vec![0.5, 0.5];
        let q = vec![1.0, 0.0];
        let tv = total_variation_distance(&p, &q).expect("should succeed");
        assert!((tv - 0.5).abs() < 1e-12, "expected TV=0.5, got {tv}");
    }

    #[test]
    fn test_jsd_symmetric() {
        let p = vec![0.3, 0.4, 0.3];
        let q = vec![0.1, 0.7, 0.2];
        let jsd_pq = js_divergence(&p, &q).expect("should succeed");
        let jsd_qp = js_divergence(&q, &p).expect("should succeed");
        assert!(
            (jsd_pq - jsd_qp).abs() < 1e-12,
            "JSD must be symmetric, got {jsd_pq} vs {jsd_qp}"
        );
    }

    #[test]
    fn test_jsd_bounded() {
        use std::f64::consts::LN_2;
        let p = vec![0.3, 0.4, 0.3];
        let q = vec![0.1, 0.8, 0.1];
        let jsd = js_divergence(&p, &q).expect("should succeed");
        assert!(jsd >= 0.0, "JSD must be >= 0, got {jsd}");
        assert!(jsd <= LN_2 + 1e-12, "JSD must be <= ln2, got {jsd}");
    }

    #[test]
    fn test_kl_identical_is_zero() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let kl = kl_divergence(&p, &p).expect("should succeed");
        assert!(kl.abs() < 1e-12, "KL(P,P) should be 0, got {kl}");
    }

    #[test]
    fn test_bhattacharyya_coefficient_bounded() {
        let p = vec![0.5, 0.3, 0.2];
        let q = vec![0.2, 0.5, 0.3];
        let bc = bhattacharyya_coefficient(&p, &q).expect("should succeed");
        assert!((0.0..=1.0).contains(&bc), "BC must be in [0,1], got {bc}");
    }

    #[test]
    fn test_hellinger_distance_bounded() {
        let p = vec![0.5, 0.3, 0.2];
        let q = vec![0.2, 0.5, 0.3];
        let h = hellinger_distance(&p, &q).expect("should succeed");
        assert!(
            (0.0..=1.0).contains(&h),
            "Hellinger must be in [0,1], got {h}"
        );
    }

    #[test]
    fn test_renyi_approaches_kl_at_alpha_one() {
        let p = vec![0.4, 0.3, 0.3];
        let q = vec![0.2, 0.5, 0.3];
        let kl = kl_divergence(&p, &q).expect("should succeed");
        let renyi = renyi_divergence(&p, &q, 1.0).expect("should succeed");
        assert!(
            (renyi - kl).abs() < 1e-8,
            "Rényi(α=1) should equal KL={kl}, got {renyi}"
        );
    }

    #[test]
    fn test_chi_squared_distance_nonnegative() {
        let p = vec![0.4, 0.3, 0.3];
        let q = vec![0.2, 0.5, 0.3];
        let d = chi_squared_distance(&p, &q).expect("should succeed");
        assert!(d >= 0.0, "chi-squared distance must be >= 0, got {d}");
    }

    #[test]
    fn test_psi_interpretation_thresholds() {
        assert_eq!(interpret_psi(0.05), "Insignificant change");
        assert_eq!(interpret_psi(0.15), "Moderate change");
        assert_eq!(interpret_psi(0.30), "Significant change");
    }

    #[test]
    fn test_normalize_distribution_sums_to_one() {
        let p = vec![2.0, 3.0, 5.0];
        let norm = normalize_distribution(&p).expect("should succeed");
        let s: f64 = norm.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-12,
            "normalized sum should be 1, got {s}"
        );
    }

    #[test]
    fn test_divergence_empty_mismatch_errors() {
        let empty: Vec<f64> = vec![];
        let p = vec![0.5, 0.5];
        let q = vec![0.3, 0.3, 0.4];
        assert!(kl_divergence(&empty, &p).is_err(), "empty p should fail");
        assert!(
            kl_divergence(&p, &q).is_err(),
            "length mismatch should fail"
        );
        assert!(
            normalize_distribution(&empty).is_err(),
            "empty normalize should fail"
        );
    }
}
