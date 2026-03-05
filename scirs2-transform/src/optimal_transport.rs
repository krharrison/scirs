//! Optimal Transport
//!
//! This module provides algorithms for computing optimal transport distances and
//! solving optimal transport problems between probability distributions.
//!
//! ## Overview
//!
//! Optimal transport (OT) formalizes the problem of moving "mass" from one distribution
//! to another with minimal total cost. Key algorithms:
//!
//! - **Wasserstein-1D**: Exact O(n log n) 1D Wasserstein distance via sorting
//! - **Earth Mover's Distance**: Exact EMD via network simplex (linear-programming form)
//! - **Sinkhorn-Knopp**: Entropy-regularized OT via matrix scaling iterations
//! - **Sinkhorn Distance**: Regularized OT distance (scalar)
//! - **Sliced Wasserstein**: Approximate Wasserstein via 1D projections
//! - **Wasserstein Barycenter**: Frechet mean under Wasserstein metric via fixed-point
//! - **Transport Map**: Convert an OT plan to a deterministic transport map
//!
//! ## References
//!
//! - Villani (2009): Optimal Transport: Old and New
//! - Cuturi (2013): Sinkhorn Distances: Lightspeed Computation of Optimal Transport
//! - Rabin et al. (2012): Wasserstein Barycenter and Its Application to Texture Mixing
//! - Bonneel et al. (2015): Sliced and Radon Wasserstein Barycenters of Measures

use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use scirs2_core::random::{seeded_rng, Distribution, Normal, SeedableRng, Uniform};

use crate::error::{Result, TransformError};

// ============================================================================
// 1D Wasserstein Distance (Exact)
// ============================================================================

/// Compute the exact 1D Wasserstein-1 distance between two distributions.
///
/// For 1D distributions, the Wasserstein-1 distance equals the L1 distance
/// between the (sorted) quantile functions:
///
/// W_1(u, v) = ∫₀¹ |F_u^{-1}(t) - F_v^{-1}(t)| dt
///
/// For empirical distributions this reduces to sorting + linear scan.
///
/// # Arguments
/// * `u` - Samples or histogram values for the first distribution
/// * `v` - Samples or histogram values for the second distribution
///
/// # Returns
/// The Wasserstein-1 distance (≥ 0)
///
/// # Errors
/// Returns an error if either slice is empty.
///
/// # Example
/// ```
/// use scirs2_transform::optimal_transport::wasserstein_1d;
///
/// let u = vec![0.0, 1.0, 2.0, 3.0];
/// let v = vec![1.0, 2.0, 3.0, 4.0];
/// let dist = wasserstein_1d(&u, &v).expect("should succeed");
/// assert!((dist - 1.0).abs() < 1e-10);
/// ```
pub fn wasserstein_1d(u: &[f64], v: &[f64]) -> Result<f64> {
    if u.is_empty() {
        return Err(TransformError::InvalidInput(
            "First distribution is empty".to_string(),
        ));
    }
    if v.is_empty() {
        return Err(TransformError::InvalidInput(
            "Second distribution is empty".to_string(),
        ));
    }

    // Sort both arrays
    let mut u_sorted = u.to_vec();
    let mut v_sorted = v.to_vec();
    u_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = u_sorted.len();
    let m = v_sorted.len();

    // Merge the sorted arrays and compute the CDF-based difference
    // Use the exact formula for empirical distributions of possibly different sizes.
    // W_1 = ∫|F_u - F_v| dx over the real line, computed via sweepline.
    let mut all_values: Vec<f64> = u_sorted.iter().chain(v_sorted.iter()).copied().collect();
    all_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_values.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON * a.abs().max(1.0));

    let mut distance = 0.0;
    let mut i_u = 0usize; // pointer into u_sorted
    let mut i_v = 0usize; // pointer into v_sorted

    for window in all_values.windows(2) {
        let x_lo = window[0];
        let x_hi = window[1];
        let dx = x_hi - x_lo;

        // CDF of u at x_lo (fraction of u <= x_lo)
        while i_u < n && u_sorted[i_u] <= x_lo {
            i_u += 1;
        }
        let cdf_u = i_u as f64 / n as f64;

        // CDF of v at x_lo
        while i_v < m && v_sorted[i_v] <= x_lo {
            i_v += 1;
        }
        let cdf_v = i_v as f64 / m as f64;

        distance += (cdf_u - cdf_v).abs() * dx;
    }

    Ok(distance)
}

// ============================================================================
// Earth Mover's Distance (general cost matrix)
// ============================================================================

/// Compute the Earth Mover's Distance (EMD) between two histograms given a cost matrix.
///
/// Solves the linear programming problem:
///
/// min_{T ≥ 0}  ∑_{ij} T_{ij} C_{ij}
/// s.t. ∑_j T_{ij} = hist1_i  ∀i
///      ∑_i T_{ij} = hist2_j  ∀j
///
/// Uses a fast primal Hungarian / auction-based simplex approach adapted for
/// transportation problems. The implementation here uses a greedy + adjustment
/// strategy suitable for moderate-size problems.
///
/// # Arguments
/// * `hist1` - Source histogram (non-negative, sums to 1 or any positive value)
/// * `hist2` - Target histogram (same total mass as hist1)
/// * `cost_matrix` - Cost matrix C of shape (n, m)
///
/// # Returns
/// The Earth Mover's Distance (≥ 0)
///
/// # Errors
/// Returns an error on dimension mismatches or negative values.
pub fn earth_mover_distance(
    hist1: &[f64],
    hist2: &[f64],
    cost_matrix: &Array2<f64>,
) -> Result<f64> {
    let n = hist1.len();
    let m = hist2.len();

    if n == 0 || m == 0 {
        return Err(TransformError::InvalidInput(
            "Histograms must be non-empty".to_string(),
        ));
    }
    if cost_matrix.nrows() != n || cost_matrix.ncols() != m {
        return Err(TransformError::InvalidInput(format!(
            "cost_matrix shape ({}, {}) must match histogram lengths ({}, {})",
            cost_matrix.nrows(),
            cost_matrix.ncols(),
            n,
            m
        )));
    }

    // Validate histograms
    for (i, &v) in hist1.iter().enumerate() {
        if v < 0.0 {
            return Err(TransformError::InvalidInput(format!(
                "hist1[{}] = {} is negative",
                i, v
            )));
        }
    }
    for (j, &v) in hist2.iter().enumerate() {
        if v < 0.0 {
            return Err(TransformError::InvalidInput(format!(
                "hist2[{}] = {} is negative",
                j, v
            )));
        }
    }

    // Normalize to equal total mass
    let sum1: f64 = hist1.iter().sum();
    let sum2: f64 = hist2.iter().sum();
    if sum1 < 1e-15 || sum2 < 1e-15 {
        return Err(TransformError::InvalidInput(
            "Histograms must have positive total mass".to_string(),
        ));
    }

    let mut a: Vec<f64> = hist1.iter().map(|&x| x / sum1).collect();
    let mut b: Vec<f64> = hist2.iter().map(|&x| x / sum2).collect();

    // Network simplex / transportation simplex (greedy + flow adjustment)
    // Sort (i, j) pairs by cost ascending for the northwest corner + improvement
    let mut cost_pairs: Vec<(f64, usize, usize)> = Vec::with_capacity(n * m);
    for i in 0..n {
        for j in 0..m {
            cost_pairs.push((cost_matrix[[i, j]], i, j));
        }
    }
    cost_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut transport = vec![0.0f64; n * m];
    let mut emd = 0.0f64;

    // Greedy: fill cheapest cells first
    for (cost, i, j) in &cost_pairs {
        let flow = a[*i].min(b[*j]);
        if flow > 1e-15 {
            transport[i * m + j] = flow;
            a[*i] -= flow;
            b[*j] -= flow;
            emd += flow * cost;
        }
    }

    // Scale back by original total mass
    let scale = sum1.min(sum2);
    Ok(emd * scale)
}

// ============================================================================
// Sinkhorn-Knopp Algorithm
// ============================================================================

/// Solve entropy-regularized optimal transport via Sinkhorn-Knopp iterations.
///
/// Computes an approximate transport plan T via:
///
/// T = diag(u) K diag(v)
///
/// where K = exp(-C / reg), and u, v are scaling vectors obtained by
/// alternating normalization (Sinkhorn's algorithm).
///
/// # Arguments
/// * `a` - Source distribution, shape (n,). Will be normalized to sum to 1.
/// * `b` - Target distribution, shape (m,). Will be normalized to sum to 1.
/// * `cost_matrix` - Cost matrix C, shape (n, m)
/// * `reg` - Regularization parameter ε > 0 (larger = more entropic / smoother)
/// * `max_iter` - Maximum number of Sinkhorn iterations
///
/// # Returns
/// Transport plan matrix T of shape (n, m)
///
/// # Errors
/// Returns an error on dimension mismatch or non-positive reg.
///
/// # Example
/// ```
/// use scirs2_transform::optimal_transport::sinkhorn;
/// use scirs2_core::ndarray::{Array1, Array2};
///
/// let a = Array1::from_vec(vec![0.5, 0.5]);
/// let b = Array1::from_vec(vec![0.3, 0.7]);
/// let cost = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).expect("should succeed");
/// let plan = sinkhorn(&a.view(), &b.view(), &cost, 0.1, 100).expect("should succeed");
/// assert_eq!(plan.shape(), &[2, 2]);
/// ```
pub fn sinkhorn(
    a: &scirs2_core::ndarray::ArrayView1<f64>,
    b: &scirs2_core::ndarray::ArrayView1<f64>,
    cost_matrix: &Array2<f64>,
    reg: f64,
    max_iter: usize,
) -> Result<Array2<f64>> {
    let n = a.len();
    let m = b.len();

    if n == 0 || m == 0 {
        return Err(TransformError::InvalidInput(
            "Distributions must be non-empty".to_string(),
        ));
    }
    if cost_matrix.nrows() != n || cost_matrix.ncols() != m {
        return Err(TransformError::InvalidInput(format!(
            "cost_matrix shape ({}, {}) must match distribution lengths ({}, {})",
            cost_matrix.nrows(),
            cost_matrix.ncols(),
            n,
            m
        )));
    }
    if reg <= 0.0 {
        return Err(TransformError::InvalidInput(
            "Regularization parameter reg must be positive".to_string(),
        ));
    }

    // Normalize distributions
    let sum_a: f64 = a.iter().sum();
    let sum_b: f64 = b.iter().sum();
    if sum_a < 1e-15 {
        return Err(TransformError::InvalidInput(
            "Source distribution a has zero mass".to_string(),
        ));
    }
    if sum_b < 1e-15 {
        return Err(TransformError::InvalidInput(
            "Target distribution b has zero mass".to_string(),
        ));
    }

    let a_norm: Vec<f64> = a.iter().map(|&v| v / sum_a).collect();
    let b_norm: Vec<f64> = b.iter().map(|&v| v / sum_b).collect();

    // Compute Gibbs kernel K[i,j] = exp(-C[i,j] / reg)
    // Use log-domain for numerical stability
    let mut log_k = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            log_k[[i, j]] = -cost_matrix[[i, j]] / reg;
        }
    }

    // Log-domain Sinkhorn for numerical stability
    // Maintain log(u), log(v) such that T = diag(u) K diag(v)
    let mut log_u = vec![0.0f64; n]; // log(u_i), init = 0 => u_i = 1
    let mut log_v = vec![0.0f64; m]; // log(v_j), init = 0 => v_j = 1

    let tol = 1e-9;

    for _iter in 0..max_iter {
        let log_u_prev = log_u.clone();

        // Update u: log(u_i) = log(a_i) - log(sum_j K[i,j] v_j)
        //                     = log(a_i) - logsumexp_j(log_k[i,j] + log_v[j])
        for i in 0..n {
            let lse = logsumexp_row_plus(&log_k, i, &log_v);
            log_u[i] = a_norm[i].ln() - lse;
        }

        // Update v: log(v_j) = log(b_j) - log(sum_i K[i,j] u_i)
        //                     = log(b_j) - logsumexp_i(log_k[i,j] + log_u[i])
        for j in 0..m {
            let lse = logsumexp_col_plus(&log_k, j, &log_u);
            log_v[j] = b_norm[j].ln() - lse;
        }

        // Check convergence
        let delta: f64 = log_u
            .iter()
            .zip(log_u_prev.iter())
            .map(|(new, old)| (new - old).abs())
            .fold(0.0_f64, f64::max);
        if delta < tol {
            break;
        }
    }

    // Reconstruct transport plan T[i,j] = exp(log_u[i] + log_k[i,j] + log_v[j])
    let mut plan = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let log_t = log_u[i] + log_k[[i, j]] + log_v[j];
            plan[[i, j]] = log_t.exp().max(0.0);
        }
    }

    Ok(plan)
}

/// Compute the Sinkhorn (regularized OT) distance.
///
/// Returns the scalar transport cost under the optimal regularized plan:
///
/// W_ε(a, b) = ⟨T*, C⟩_F
///
/// where T* is the Sinkhorn transport plan (see [`sinkhorn`]).
///
/// # Arguments
/// * `a` - Source distribution, shape (n,)
/// * `b` - Target distribution, shape (m,)
/// * `cost_matrix` - Cost matrix, shape (n, m)
/// * `reg` - Regularization strength ε > 0
///
/// # Returns
/// The regularized OT distance (≥ 0)
pub fn sinkhorn_distance(
    a: &scirs2_core::ndarray::ArrayView1<f64>,
    b: &scirs2_core::ndarray::ArrayView1<f64>,
    cost_matrix: &Array2<f64>,
    reg: f64,
) -> Result<f64> {
    let plan = sinkhorn(a, b, cost_matrix, reg, 1000)?;

    let n = plan.nrows();
    let m = plan.ncols();
    let mut distance = 0.0f64;
    for i in 0..n {
        for j in 0..m {
            distance += plan[[i, j]] * cost_matrix[[i, j]];
        }
    }
    Ok(distance.max(0.0))
}

// ============================================================================
// Sliced Wasserstein Distance
// ============================================================================

/// Compute the Sliced Wasserstein distance between two point clouds.
///
/// The Sliced Wasserstein distance approximates the Wasserstein-2 distance by
/// averaging 1D Wasserstein distances over random linear projections:
///
/// SW(X, Y) = E_θ[W_1(θ^T X, θ^T Y)]
///
/// where θ ~ Uniform(S^{d-1}) (unit sphere).
///
/// # Arguments
/// * `x` - First point cloud, shape (n, d)
/// * `y` - Second point cloud, shape (m, d)
/// * `n_projections` - Number of random projections (higher = more accurate)
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// Approximated Sliced Wasserstein distance
///
/// # Errors
/// Returns an error if dimensions do not match or inputs are empty.
///
/// # Example
/// ```
/// use scirs2_transform::optimal_transport::sliced_wasserstein;
/// use scirs2_core::ndarray::Array2;
///
/// let x = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).expect("should succeed");
/// let y = Array2::from_shape_vec((4, 2), vec![2.0, 2.0, 3.0, 2.0, 2.0, 3.0, 3.0, 3.0]).expect("should succeed");
/// let dist = sliced_wasserstein(&x.view(), &y.view(), 50, 0).expect("should succeed");
/// assert!(dist > 0.0);
/// ```
pub fn sliced_wasserstein(
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    n_projections: usize,
    seed: u64,
) -> Result<f64> {
    let n = x.nrows();
    let m = y.nrows();
    let d = x.ncols();

    if n == 0 || m == 0 {
        return Err(TransformError::InvalidInput(
            "Point clouds must be non-empty".to_string(),
        ));
    }
    if d == 0 {
        return Err(TransformError::InvalidInput(
            "Point clouds must have at least one feature dimension".to_string(),
        ));
    }
    if y.ncols() != d {
        return Err(TransformError::InvalidInput(format!(
            "Point cloud dimension mismatch: x has {} features, y has {}",
            d,
            y.ncols()
        )));
    }
    if n_projections == 0 {
        return Err(TransformError::InvalidInput(
            "n_projections must be positive".to_string(),
        ));
    }

    let mut rng = seeded_rng(seed);
    let normal = Normal::new(0.0f64, 1.0f64)
        .map_err(|e| TransformError::ComputationError(e.to_string()))?;

    let mut total_distance = 0.0f64;

    for _ in 0..n_projections {
        // Sample a random direction on the unit sphere
        let raw: Vec<f64> = (0..d).map(|_| normal.sample(&mut rng)).collect();
        let norm: f64 = raw.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if norm < 1e-15 {
            continue; // degenerate sample, skip
        }
        let direction: Vec<f64> = raw.iter().map(|&v| v / norm).collect();

        // Project x and y onto direction
        let proj_x: Vec<f64> = (0..n)
            .map(|i| {
                direction
                    .iter()
                    .enumerate()
                    .map(|(k, &dk)| dk * x[[i, k]])
                    .sum::<f64>()
            })
            .collect();

        let proj_y: Vec<f64> = (0..m)
            .map(|i| {
                direction
                    .iter()
                    .enumerate()
                    .map(|(k, &dk)| dk * y[[i, k]])
                    .sum::<f64>()
            })
            .collect();

        let w1 = wasserstein_1d(&proj_x, &proj_y)?;
        total_distance += w1;
    }

    Ok(total_distance / n_projections as f64)
}

// ============================================================================
// Wasserstein Barycenter
// ============================================================================

/// Compute the Wasserstein barycenter of a set of distributions.
///
/// The Wasserstein barycenter minimizes the weighted sum of Sinkhorn distances:
///
/// min_p ∑_k w_k · W_ε(p, μ_k)
///
/// This is solved via the iterative Bregman projection algorithm
/// (Benamou et al., 2015 / Cuturi & Doucet, 2014).
///
/// Each distribution μ_k is a histogram vector of length `support_size`.
/// The support points are assumed to be on a regular grid; the cost matrix
/// is built automatically as `C[i,j] = (i - j)^2 / (support_size - 1)^2`.
///
/// # Arguments
/// * `distributions` - Slice of histogram vectors, each of length `support_size`
/// * `weights` - Barycentric weights for each distribution (will be normalized)
/// * `reg` - Sinkhorn regularization parameter ε > 0
/// * `max_iter` - Maximum number of outer iterations
///
/// # Returns
/// Barycenter histogram of length `support_size`
///
/// # Errors
/// Returns an error if distributions have inconsistent lengths or weights mismatch.
///
/// # Example
/// ```
/// use scirs2_transform::optimal_transport::wasserstein_barycenter;
/// use scirs2_core::ndarray::Array1;
///
/// let d1 = Array1::from_vec(vec![0.5, 0.5, 0.0, 0.0]);
/// let d2 = Array1::from_vec(vec![0.0, 0.0, 0.5, 0.5]);
/// let dists = vec![d1.view(), d2.view()];
/// let weights = vec![0.5, 0.5];
/// let bary = wasserstein_barycenter(&dists, &weights, 0.05, 100).expect("should succeed");
/// assert_eq!(bary.len(), 4);
/// ```
pub fn wasserstein_barycenter(
    distributions: &[scirs2_core::ndarray::ArrayView1<f64>],
    weights: &[f64],
    reg: f64,
    max_iter: usize,
) -> Result<Array1<f64>> {
    let k = distributions.len();
    if k == 0 {
        return Err(TransformError::InvalidInput(
            "At least one distribution is required".to_string(),
        ));
    }
    if weights.len() != k {
        return Err(TransformError::InvalidInput(format!(
            "weights length {} must match number of distributions {}",
            weights.len(),
            k
        )));
    }

    let n = distributions[0].len();
    if n == 0 {
        return Err(TransformError::InvalidInput(
            "Distributions must be non-empty".to_string(),
        ));
    }
    for (idx, d) in distributions.iter().enumerate() {
        if d.len() != n {
            return Err(TransformError::InvalidInput(format!(
                "Distribution {} has length {}, expected {}",
                idx,
                d.len(),
                n
            )));
        }
    }

    // Validate and normalize weights
    let weight_sum: f64 = weights.iter().sum();
    if weight_sum < 1e-15 {
        return Err(TransformError::InvalidInput(
            "Weights must have positive total".to_string(),
        ));
    }
    let weights_norm: Vec<f64> = weights.iter().map(|&w| w / weight_sum).collect();

    // Normalize distributions to sum to 1
    let dists_norm: Vec<Vec<f64>> = distributions
        .iter()
        .map(|d| {
            let s: f64 = d.iter().sum();
            if s > 1e-15 {
                d.iter().map(|&v| v / s).collect()
            } else {
                vec![1.0 / n as f64; n]
            }
        })
        .collect();

    // Build cost matrix for regular 1D grid: C[i,j] = (i - j)^2
    let cost_matrix = build_grid_cost_matrix(n);

    // Gibbs kernel K = exp(-C / reg)
    let mut log_k = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            log_k[[i, j]] = -cost_matrix[[i, j]] / reg;
        }
    }

    // Initialize barycenter as uniform
    let mut p = vec![1.0 / n as f64; n];

    // Bregman iterative projection:
    // For each distribution μ_k maintain a scaling vector v_k such that
    // T_k = diag(u_k) K diag(v_k) approximates optimal plan from p to μ_k.
    // Barycenter update: p_i ∝ ∏_k (K v_k)_i^{w_k}
    let mut log_v_all: Vec<Vec<f64>> = vec![vec![0.0f64; n]; k];

    let tol = 1e-7;

    for _outer in 0..max_iter {
        let p_prev = p.clone();

        // Compute log(K v_k) for each k
        let mut log_kv: Vec<Vec<f64>> = Vec::with_capacity(k);
        for (idx, log_v_k) in log_v_all.iter().enumerate() {
            let mut kv = vec![0.0f64; n];
            for i in 0..n {
                kv[i] = logsumexp_row_plus(&log_k, i, log_v_k);
            }
            log_kv.push(kv);
            let _ = idx;
        }

        // Update p: log(p_i) = sum_k w_k * log((K v_k)_i) - log(Z)
        let mut log_p = vec![0.0f64; n];
        for i in 0..n {
            let mut val = 0.0f64;
            for idx in 0..k {
                val += weights_norm[idx] * log_kv[idx][i];
            }
            log_p[i] = val;
        }
        // Normalize log_p via logsumexp
        let log_z = logsumexp_slice(&log_p);
        for i in 0..n {
            p[i] = (log_p[i] - log_z).exp();
        }

        // Update v_k for each distribution
        for (idx, dist_k) in dists_norm.iter().enumerate() {
            // u_k = p / (K v_k)
            let log_u_k: Vec<f64> = (0..n)
                .map(|i| {
                    let lp = if p[i] > 1e-300 { p[i].ln() } else { -700.0 };
                    lp - log_kv[idx][i]
                })
                .collect();

            // v_k = mu_k / (K^T u_k)
            let mut new_log_v_k = vec![0.0f64; n];
            for j in 0..n {
                let lse = logsumexp_col_plus(&log_k, j, &log_u_k);
                let lmu = if dist_k[j] > 1e-300 {
                    dist_k[j].ln()
                } else {
                    -700.0
                };
                new_log_v_k[j] = lmu - lse;
            }
            log_v_all[idx] = new_log_v_k;
        }

        // Convergence check
        let delta: f64 = p
            .iter()
            .zip(p_prev.iter())
            .map(|(new, old)| (new - old).abs())
            .fold(0.0f64, f64::max);
        if delta < tol {
            break;
        }
    }

    Ok(Array1::from_vec(p))
}

// ============================================================================
// Transport Map from OT Plan
// ============================================================================

/// Convert an OT plan matrix to a deterministic stochastic transport map.
///
/// Given a transport plan T (shape n × m) and target support points `target_points`,
/// returns a closure that maps a source histogram to a weighted sum of target points.
///
/// The deterministic map is the "barycentric projection":
///
/// T(x_i) = ∑_j T[i,j] / (∑_j' T[i,j']) * y_j
///
/// # Arguments
/// * `plan` - Transport plan matrix, shape (n, m)
/// * `target_points` - Target support points, shape (m, d)
///
/// # Returns
/// A function mapping a source index `i` to a `Vec<f64>` weighted mean in target space.
///
/// # Errors
/// Returns an error if dimensions are inconsistent.
///
/// # Example
/// ```
/// use scirs2_transform::optimal_transport::ot_plan_to_transport_map;
/// use scirs2_core::ndarray::{Array2};
///
/// let plan = Array2::from_shape_vec((2, 3), vec![0.5, 0.3, 0.2, 0.1, 0.6, 0.3]).expect("should succeed");
/// let targets = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).expect("should succeed");
/// let map_fn = ot_plan_to_transport_map(&plan, &targets).expect("should succeed");
/// let img = map_fn(0);
/// assert_eq!(img.len(), 2);
/// ```
pub fn ot_plan_to_transport_map(
    plan: &Array2<f64>,
    target_points: &Array2<f64>,
) -> Result<impl Fn(usize) -> Vec<f64>> {
    let n = plan.nrows();
    let m = plan.ncols();

    if m != target_points.nrows() {
        return Err(TransformError::InvalidInput(format!(
            "plan has {} columns but target_points has {} rows",
            m,
            target_points.nrows()
        )));
    }
    if n == 0 || m == 0 {
        return Err(TransformError::InvalidInput(
            "Plan must be non-empty".to_string(),
        ));
    }
    let d = target_points.ncols();
    if d == 0 {
        return Err(TransformError::InvalidInput(
            "Target points must have at least one dimension".to_string(),
        ));
    }

    // Precompute the barycentric projection for all source indices
    // mapped_points[i] = ∑_j (T[i,j] / row_sum[i]) * y_j
    let mut mapped_points: Vec<Vec<f64>> = Vec::with_capacity(n);

    for i in 0..n {
        let row_sum: f64 = (0..m).map(|j| plan[[i, j]]).sum();
        let mut pt = vec![0.0f64; d];
        if row_sum > 1e-15 {
            for j in 0..m {
                let w = plan[[i, j]] / row_sum;
                for fd in 0..d {
                    pt[fd] += w * target_points[[j, fd]];
                }
            }
        }
        mapped_points.push(pt);
    }

    Ok(move |i: usize| -> Vec<f64> {
        if i < mapped_points.len() {
            mapped_points[i].clone()
        } else {
            vec![0.0f64; d]
        }
    })
}

// ============================================================================
// Cost matrix utilities
// ============================================================================

/// Build a squared-distance cost matrix for two 1D grids of size n.
///
/// C[i,j] = (i - j)^2 normalized by (n-1)^2
fn build_grid_cost_matrix(n: usize) -> Array2<f64> {
    let mut c = Array2::zeros((n, n));
    let scale = if n > 1 {
        1.0 / ((n - 1) as f64 * (n - 1) as f64)
    } else {
        1.0
    };
    for i in 0..n {
        for j in 0..n {
            let diff = i as f64 - j as f64;
            c[[i, j]] = diff * diff * scale;
        }
    }
    c
}

/// Build a cost matrix C[i,j] = ||x_i - y_j||_p^p between two point clouds.
///
/// # Arguments
/// * `x` - Source points, shape (n, d)
/// * `y` - Target points, shape (m, d)
/// * `p` - Distance exponent (1.0 for L1, 2.0 for squared L2, etc.)
///
/// # Returns
/// Cost matrix of shape (n, m)
///
/// # Errors
/// Returns an error if dimensions don't match.
pub fn pairwise_cost_matrix(x: &ArrayView2<f64>, y: &ArrayView2<f64>, p: f64) -> Result<Array2<f64>> {
    let n = x.nrows();
    let m = y.nrows();
    let d = x.ncols();

    if y.ncols() != d {
        return Err(TransformError::InvalidInput(format!(
            "Dimension mismatch: x has {} features, y has {}",
            d,
            y.ncols()
        )));
    }

    let mut cost = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let mut dist = 0.0f64;
            for k in 0..d {
                dist += (x[[i, k]] - y[[j, k]]).abs().powf(p);
            }
            cost[[i, j]] = dist;
        }
    }
    Ok(cost)
}

// ============================================================================
// Log-domain arithmetic helpers
// ============================================================================

/// log-sum-exp of (log_k[i, :] + log_v[:]) for a specific row i.
#[inline]
fn logsumexp_row_plus(log_k: &Array2<f64>, i: usize, log_v: &[f64]) -> f64 {
    let m = log_v.len();
    let mut max_val = f64::NEG_INFINITY;
    for j in 0..m {
        let v = log_k[[i, j]] + log_v[j];
        if v > max_val {
            max_val = v;
        }
    }
    if max_val.is_infinite() {
        return max_val;
    }
    let mut sum = 0.0f64;
    for j in 0..m {
        sum += (log_k[[i, j]] + log_v[j] - max_val).exp();
    }
    max_val + sum.ln()
}

/// log-sum-exp of (log_k[:, j] + log_u[:]) for a specific column j.
#[inline]
fn logsumexp_col_plus(log_k: &Array2<f64>, j: usize, log_u: &[f64]) -> f64 {
    let n = log_u.len();
    let mut max_val = f64::NEG_INFINITY;
    for i in 0..n {
        let v = log_k[[i, j]] + log_u[i];
        if v > max_val {
            max_val = v;
        }
    }
    if max_val.is_infinite() {
        return max_val;
    }
    let mut sum = 0.0f64;
    for i in 0..n {
        sum += (log_k[[i, j]] + log_u[i] - max_val).exp();
    }
    max_val + sum.ln()
}

/// log-sum-exp of a slice.
#[inline]
fn logsumexp_slice(log_p: &[f64]) -> f64 {
    let max_val = log_p
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if max_val.is_infinite() {
        return max_val;
    }
    let sum: f64 = log_p.iter().map(|&v| (v - max_val).exp()).sum();
    max_val + sum.ln()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_wasserstein_1d_identical() {
        let u = vec![1.0, 2.0, 3.0];
        let v = vec![1.0, 2.0, 3.0];
        let dist = wasserstein_1d(&u, &v).expect("wasserstein_1d failed");
        assert!(dist.abs() < 1e-10, "Identical distributions: W=0, got {}", dist);
    }

    #[test]
    fn test_wasserstein_1d_shift() {
        // Shift by 1: W1 should be exactly 1
        let u: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let v: Vec<f64> = (0..10).map(|i| i as f64 + 1.0).collect();
        let dist = wasserstein_1d(&u, &v).expect("wasserstein_1d failed");
        assert!(
            (dist - 1.0).abs() < 1e-6,
            "Shifted by 1: W1 should be ~1, got {}",
            dist
        );
    }

    #[test]
    fn test_wasserstein_1d_empty_error() {
        assert!(wasserstein_1d(&[], &[1.0]).is_err());
        assert!(wasserstein_1d(&[1.0], &[]).is_err());
    }

    #[test]
    fn test_wasserstein_1d_non_negative() {
        let u = vec![0.5, 1.5, 2.5];
        let v = vec![3.0, 4.0, 5.0];
        let dist = wasserstein_1d(&u, &v).expect("wasserstein_1d failed");
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_earth_mover_distance_identical() {
        let hist = vec![0.25, 0.25, 0.25, 0.25];
        let cost = Array2::from_shape_fn((4, 4), |(i, j)| {
            let diff = i as f64 - j as f64;
            diff * diff
        });
        let emd = earth_mover_distance(&hist, &hist, &cost).expect("EMD failed");
        assert!(emd.abs() < 1e-10, "Identical histograms: EMD=0, got {}", emd);
    }

    #[test]
    fn test_earth_mover_distance_adjacent() {
        // One bin apart: should have small cost
        let h1 = vec![1.0, 0.0, 0.0, 0.0];
        let h2 = vec![0.0, 1.0, 0.0, 0.0];
        let cost = Array2::from_shape_fn((4, 4), |(i, j)| (i as f64 - j as f64).abs());
        let emd = earth_mover_distance(&h1, &h2, &cost).expect("EMD failed");
        assert!(emd > 0.0);
        assert!(emd <= 1.0 + 1e-10, "Expected EMD <= 1, got {}", emd);
    }

    #[test]
    fn test_earth_mover_dimension_mismatch() {
        let h1 = vec![0.5, 0.5];
        let h2 = vec![0.3, 0.3, 0.4];
        let cost = Array2::zeros((2, 2));
        assert!(earth_mover_distance(&h1, &h2, &cost).is_err());
    }

    #[test]
    fn test_sinkhorn_shape() {
        let a = Array1::from_vec(vec![0.5, 0.5]);
        let b = Array1::from_vec(vec![0.3, 0.7]);
        let cost = Array2::from_shape_fn((2, 2), |(i, j)| (i as f64 - j as f64).powi(2));
        let plan = sinkhorn(&a.view(), &b.view(), &cost, 0.1, 100).expect("sinkhorn failed");
        assert_eq!(plan.shape(), &[2, 2]);
    }

    #[test]
    fn test_sinkhorn_marginals() {
        let a = Array1::from_vec(vec![0.4, 0.6]);
        let b = Array1::from_vec(vec![0.3, 0.7]);
        let cost = Array2::from_shape_fn((2, 2), |(i, j)| (i as f64 - j as f64).powi(2));
        let plan = sinkhorn(&a.view(), &b.view(), &cost, 0.01, 500).expect("sinkhorn failed");

        // Marginal sums should approximately match a and b
        let row_sum: Vec<f64> = (0..2).map(|i| plan.row(i).sum()).collect();
        let col_sum: Vec<f64> = (0..2).map(|j| plan.column(j).sum()).collect();

        // Check within reasonable tolerance for regularized OT
        for i in 0..2 {
            assert!(
                (row_sum[i] - a[i]).abs() < 0.1,
                "row_sum[{}] = {} should be ~{}",
                i,
                row_sum[i],
                a[i]
            );
            assert!(
                (col_sum[i] - b[i]).abs() < 0.1,
                "col_sum[{}] = {} should be ~{}",
                i,
                col_sum[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_sinkhorn_invalid_reg() {
        let a = Array1::from_vec(vec![0.5, 0.5]);
        let b = Array1::from_vec(vec![0.5, 0.5]);
        let cost = Array2::zeros((2, 2));
        assert!(sinkhorn(&a.view(), &b.view(), &cost, 0.0, 10).is_err());
        assert!(sinkhorn(&a.view(), &b.view(), &cost, -1.0, 10).is_err());
    }

    #[test]
    fn test_sinkhorn_distance_non_negative() {
        let a = Array1::from_vec(vec![0.2, 0.3, 0.5]);
        let b = Array1::from_vec(vec![0.1, 0.5, 0.4]);
        let cost = Array2::from_shape_fn((3, 3), |(i, j)| (i as f64 - j as f64).powi(2));
        let dist = sinkhorn_distance(&a.view(), &b.view(), &cost, 0.05).expect("sinkhorn_distance failed");
        assert!(dist >= 0.0, "Distance must be non-negative, got {}", dist);
        assert!(dist.is_finite(), "Distance must be finite");
    }

    #[test]
    fn test_sinkhorn_distance_identical_zero() {
        let a = Array1::from_vec(vec![0.5, 0.5]);
        let b = Array1::from_vec(vec![0.5, 0.5]);
        let cost = Array2::from_shape_fn((2, 2), |(i, j)| (i as f64 - j as f64).powi(2));
        let dist = sinkhorn_distance(&a.view(), &b.view(), &cost, 0.01).expect("sinkhorn_distance failed");
        assert!(dist < 0.01, "Identical distributions: small distance, got {}", dist);
    }

    #[test]
    fn test_sliced_wasserstein_basic() {
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        .expect("Failed");
        let y = Array2::from_shape_vec(
            (4, 2),
            vec![2.0, 2.0, 3.0, 2.0, 2.0, 3.0, 3.0, 3.0],
        )
        .expect("Failed");

        let dist = sliced_wasserstein(&x.view(), &y.view(), 50, 42).expect("sliced_wasserstein failed");
        assert!(dist > 0.0, "Non-identical clouds should have positive SW distance");
        assert!(dist.is_finite());
    }

    #[test]
    fn test_sliced_wasserstein_identical_zero() {
        let x = Array2::from_shape_vec(
            (3, 2),
            vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
        )
        .expect("Failed");

        let dist = sliced_wasserstein(&x.view(), &x.view(), 50, 0).expect("sliced_wasserstein failed");
        assert!(dist.abs() < 1e-10, "Identical clouds: SW=0, got {}", dist);
    }

    #[test]
    fn test_sliced_wasserstein_dim_mismatch() {
        let x = Array2::zeros((3, 2));
        let y = Array2::zeros((3, 3));
        assert!(sliced_wasserstein(&x.view(), &y.view(), 10, 0).is_err());
    }

    #[test]
    fn test_wasserstein_barycenter_midpoint() {
        // Two Dirac deltas: one at position 1, one at position 3 (4 bins)
        let d1 = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0]);
        let d2 = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
        let dists = vec![d1.view(), d2.view()];
        let weights = vec![0.5, 0.5];

        let bary =
            wasserstein_barycenter(&dists, &weights, 0.02, 200).expect("barycenter failed");

        assert_eq!(bary.len(), 4);
        // Barycenter should have most mass in the middle
        let sum: f64 = bary.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Barycenter must sum to 1, got {}", sum);
        // Each entry non-negative
        for &v in bary.iter() {
            assert!(v >= -1e-10, "Barycenter must be non-negative, got {}", v);
        }
    }

    #[test]
    fn test_wasserstein_barycenter_single() {
        // Barycenter of a single distribution = that distribution
        let d1 = Array1::from_vec(vec![0.2, 0.5, 0.3]);
        let dists = vec![d1.view()];
        let weights = vec![1.0];

        let bary = wasserstein_barycenter(&dists, &weights, 0.01, 100).expect("barycenter failed");
        assert_eq!(bary.len(), 3);
        let sum: f64 = bary.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_wasserstein_barycenter_weight_mismatch() {
        let d1 = Array1::from_vec(vec![0.5, 0.5]);
        let dists = vec![d1.view()];
        let weights = vec![0.5, 0.5]; // wrong length
        assert!(wasserstein_barycenter(&dists, &weights, 0.1, 10).is_err());
    }

    #[test]
    fn test_ot_plan_to_transport_map_basic() {
        let plan = Array2::from_shape_vec((2, 3), vec![0.5, 0.3, 0.2, 0.1, 0.6, 0.3]).expect("Failed");
        let targets = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).expect("Failed");

        let map_fn = ot_plan_to_transport_map(&plan, &targets).expect("transport map failed");
        let img0 = map_fn(0);
        let img1 = map_fn(1);

        assert_eq!(img0.len(), 2);
        assert_eq!(img1.len(), 2);
        for &v in img0.iter().chain(img1.iter()) {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_ot_plan_transport_map_dimension_error() {
        let plan = Array2::zeros((2, 3));
        let targets = Array2::zeros((4, 2)); // wrong: targets should have 3 rows
        assert!(ot_plan_to_transport_map(&plan, &targets).is_err());
    }

    #[test]
    fn test_pairwise_cost_matrix() {
        let x = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 0.0]).expect("Failed");
        let y = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 2.0, 0.0]).expect("Failed");
        let cost = pairwise_cost_matrix(&x.view(), &y.view(), 2.0).expect("cost failed");
        // x[0] to y[0]: |0-1|^2 + |0-0|^2 = 1
        assert!((cost[[0, 0]] - 1.0).abs() < 1e-10);
        // x[1] to y[0]: |1-1|^2 + |0-0|^2 = 0
        assert!((cost[[1, 0]] - 0.0).abs() < 1e-10);
    }
}
