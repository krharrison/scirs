//! Gromov-Wasserstein distance and multi-marginal optimal transport.
//!
//! The Gromov-Wasserstein (GW) distance between two metric measure spaces
//! (X, d_X, μ) and (Y, d_Y, ν) is defined as:
//!
//! ```text
//! GW(X, Y) = min_{T ∈ Π(μ,ν)} ∑_{i,j,k,l} (d_X(i,k) - d_Y(j,l))² T_{ij} T_{kl}
//! ```
//!
//! We solve the entropic regularised version via Frank-Wolfe projected gradient:
//! at each iteration compute the gradient G, then solve a Sinkhorn step with
//! cost G to get the new transport plan T.
//!
//! ## References
//!
//! - Mémoli (2011). Gromov-Wasserstein Distances and the Metric Approach to Object Matching.
//! - Peyré, Cuturi & Solomon (2016). Gromov-Wasserstein Averaging of Kernel and Distance Matrices.
//! - Flamary et al. (2021). POT: Python Optimal Transport. JMLR.

use crate::error::{Result, TransformError};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for Gromov-Wasserstein computation.
#[derive(Debug, Clone)]
pub struct GwConfig {
    /// Entropic regularisation strength (ε > 0).
    pub epsilon: f64,
    /// Maximum number of Frank-Wolfe outer iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the relative change of GW loss.
    pub tol: f64,
    /// Maximum Sinkhorn inner iterations.
    pub sinkhorn_max_iter: usize,
    /// Sinkhorn convergence tolerance.
    pub sinkhorn_tol: f64,
}

impl Default for GwConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            max_iter: 100,
            tol: 1e-9,
            sinkhorn_max_iter: 1000,
            sinkhorn_tol: 1e-9,
        }
    }
}

// ─── GwResult ─────────────────────────────────────────────────────────────────

/// Result of a Gromov-Wasserstein computation.
#[derive(Debug, Clone)]
pub struct GwResult {
    /// Flattened transport plan matrix of shape n×m (row-major).
    pub transport_plan: Vec<f64>,
    /// Gromov-Wasserstein distance (square root of the GW loss).
    pub gw_distance: f64,
    /// Number of outer Frank-Wolfe iterations performed.
    pub n_iter: usize,
}

// ─── GW loss and gradient ─────────────────────────────────────────────────────

/// Compute the GW quadratic loss:
/// L(T) = ∑_{i,j,k,l} (C_X[i,k] - C_Y[j,l])² T[i,j] T[k,l]
///
/// Efficiently expanded as:
/// L(T) = ∑_{ij} h_{ij} T[i,j]
/// where h[i,j] = ∑_{kl} (C_X[i,k]² + C_Y[j,l]² - 2 C_X[i,k] C_Y[j,l]) T[k,l]
fn gw_loss(cost_x: &[Vec<f64>], cost_y: &[Vec<f64>], t: &[f64]) -> f64 {
    let n = cost_x.len();
    let m = cost_y.len();

    // Precompute row sums for C_X^2 and C_Y^2 weighted by T
    // term1[i] = ∑_k C_X[i,k]^2 * (∑_j T[k,j]) = ∑_k C_X[i,k]^2 * p_k
    // where p_k = ∑_j T[k,j]
    let p_marginal: Vec<f64> = (0..n).map(|k| (0..m).map(|j| t[k * m + j]).sum()).collect();
    let q_marginal: Vec<f64> = (0..m).map(|l| (0..n).map(|i| t[i * m + l]).sum()).collect();

    let mut loss = 0.0;
    for i in 0..n {
        for j in 0..m {
            if t[i * m + j] < 1e-300 {
                continue;
            }
            // h[i,j] = ∑_k C_X[i,k]^2 * p_k + ∑_l C_Y[j,l]^2 * q_l - 2 * (C_X T C_Y^T)[i,j]
            let cx2: f64 = (0..n)
                .map(|k| cost_x[i][k] * cost_x[i][k] * p_marginal[k])
                .sum();
            let cy2: f64 = (0..m)
                .map(|l| cost_y[j][l] * cost_y[j][l] * q_marginal[l])
                .sum();
            // (C_X T C_Y^T)[i,j] = ∑_{k,l} C_X[i,k] T[k,l] C_Y[j,l]
            let cross: f64 = (0..n)
                .flat_map(|k| (0..m).map(move |l| cost_x[i][k] * t[k * m + l] * cost_y[j][l]))
                .sum();
            let h_ij = cx2 + cy2 - 2.0 * cross;
            loss += h_ij * t[i * m + j];
        }
    }
    loss
}

/// Compute the GW gradient w.r.t. T:
/// G[i,j] = 2 * ∑_{k,l} (C_X[i,k] - C_Y[j,l])² T[k,l]
///
/// Expanded (same trick):
/// G[i,j] = 2 * (∑_k C_X[i,k]^2 * p_k + ∑_l C_Y[j,l]^2 * q_l - 2 (C_X T C_Y^T)[i,j])
fn gw_gradient(cost_x: &[Vec<f64>], cost_y: &[Vec<f64>], t: &[f64]) -> Vec<Vec<f64>> {
    let n = cost_x.len();
    let m = cost_y.len();

    let p_marginal: Vec<f64> = (0..n).map(|k| (0..m).map(|j| t[k * m + j]).sum()).collect();
    let q_marginal: Vec<f64> = (0..m).map(|l| (0..n).map(|i| t[i * m + l]).sum()).collect();

    // C_X^2 * p: row sums
    let cx2_p: Vec<f64> = (0..n)
        .map(|i| {
            (0..n)
                .map(|k| cost_x[i][k] * cost_x[i][k] * p_marginal[k])
                .sum()
        })
        .collect();
    // C_Y^2 * q: row sums
    let cy2_q: Vec<f64> = (0..m)
        .map(|j| {
            (0..m)
                .map(|l| cost_y[j][l] * cost_y[j][l] * q_marginal[l])
                .sum()
        })
        .collect();

    // C_X T C_Y^T: product of (n×n) * (n×m) * (m×m)^T  — but C_Y is m×m symmetric
    // (C_X T C_Y^T)[i,j] = ∑_{k,l} C_X[i,k] T[k,l] C_Y[j,l]
    let mut cx_t_cy = vec![vec![0.0_f64; m]; n];
    for i in 0..n {
        for l in 0..m {
            // sum over k: C_X[i,k] * T[k,l]
            let xt_il: f64 = (0..n).map(|k| cost_x[i][k] * t[k * m + l]).sum();
            for j in 0..m {
                cx_t_cy[i][j] += xt_il * cost_y[j][l];
            }
        }
    }

    let mut grad = vec![vec![0.0_f64; m]; n];
    for i in 0..n {
        for j in 0..m {
            grad[i][j] = 2.0 * (cx2_p[i] + cy2_q[j] - 2.0 * cx_t_cy[i][j]);
        }
    }
    grad
}

// ─── Sinkhorn (log-domain stabilized) ────────────────────────────────────────

/// Log-domain stabilised Sinkhorn algorithm.
///
/// Solves the regularised OT problem:
/// min_{T ∈ Π(a,b)} ⟨C, T⟩ - ε H(T)
///
/// where H(T) = -Sum\_{ij} `T[ij]` (ln `T[ij]` - 1) is the entropy.
///
/// Works in log-space to avoid numerical underflow for small ε.
///
/// # Arguments
/// * `a` — source histogram (must sum to 1)
/// * `b` — target histogram (must sum to 1)
/// * `cost` — cost matrix (n×m)
/// * `epsilon` — regularisation strength
/// * `max_iter` — maximum iterations
///
/// # Returns
/// Flattened n×m transport plan.
pub fn sinkhorn_log_stabilized(
    a: &[f64],
    b: &[f64],
    cost: &[Vec<f64>],
    epsilon: f64,
    max_iter: usize,
) -> Vec<f64> {
    let n = a.len();
    let m = b.len();

    // Log cost matrix: M[i,j] = -C[i,j] / epsilon
    let log_m: Vec<Vec<f64>> = cost
        .iter()
        .map(|row| row.iter().map(|&c| -c / epsilon).collect())
        .collect();

    let log_a: Vec<f64> = a
        .iter()
        .map(|&x| if x > 0.0 { x.ln() } else { f64::NEG_INFINITY })
        .collect();
    let log_b: Vec<f64> = b
        .iter()
        .map(|&x| if x > 0.0 { x.ln() } else { f64::NEG_INFINITY })
        .collect();

    // Dual variables in log domain
    let mut f = vec![0.0_f64; n]; // log dual for rows
    let mut g = vec![0.0_f64; m]; // log dual for cols

    for _iter in 0..max_iter {
        let f_old = f.clone();

        // Update g: g[j] = log_b[j] - log_sum_exp_i(f[i] + M[i][j])
        for j in 0..m {
            let lse = log_sum_exp_vec((0..n).map(|i| f[i] + log_m[i][j]).collect());
            g[j] = log_b[j] - lse;
        }

        // Update f: f[i] = log_a[i] - log_sum_exp_j(g[j] + M[i][j])
        for i in 0..n {
            let lse = log_sum_exp_vec((0..m).map(|j| g[j] + log_m[i][j]).collect());
            f[i] = log_a[i] - lse;
        }

        // Check convergence
        let diff: f64 = f
            .iter()
            .zip(f_old.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        if diff < 1e-12 {
            break;
        }
    }

    // Build transport plan: T[i,j] = exp(f[i] + g[j] + M[i][j])
    let mut t = vec![0.0_f64; n * m];
    for i in 0..n {
        for j in 0..m {
            let log_t = f[i] + g[j] + log_m[i][j];
            t[i * m + j] = if log_t < -500.0 { 0.0 } else { log_t.exp() };
        }
    }
    t
}

/// Numerically stable log-sum-exp.
fn log_sum_exp_vec(vals: Vec<f64>) -> f64 {
    let finite: Vec<f64> = vals.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_val = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = finite.iter().map(|&v| (v - max_val).exp()).sum();
    max_val + sum.ln()
}

// ─── Gromov-Wasserstein ───────────────────────────────────────────────────────

/// Compute the entropic Gromov-Wasserstein distance between two metric spaces.
///
/// Uses Frank-Wolfe projected gradient: at each iteration compute the gradient
/// matrix G, solve a regularised OT sub-problem (Sinkhorn) with cost G to get
/// the new transport plan T, then take a step along the direction T_new - T.
///
/// # Arguments
/// * `cost_x` — n×n pairwise cost/distance matrix for space X
/// * `cost_y` — m×m pairwise cost/distance matrix for space Y
/// * `p` — probability measure on X (must be non-negative and sum to 1)
/// * `q` — probability measure on Y (must be non-negative and sum to 1)
/// * `config` — algorithm parameters
pub fn gromov_wasserstein(
    cost_x: &[Vec<f64>],
    cost_y: &[Vec<f64>],
    p: &[f64],
    q: &[f64],
    config: &GwConfig,
) -> Result<GwResult> {
    let n = cost_x.len();
    let m = cost_y.len();

    // Input validation
    if n == 0 || m == 0 {
        return Err(TransformError::InvalidInput(
            "Cost matrices must be non-empty".to_string(),
        ));
    }
    if p.len() != n {
        return Err(TransformError::InvalidInput(format!(
            "Measure p has length {} but cost_x has size {n}",
            p.len()
        )));
    }
    if q.len() != m {
        return Err(TransformError::InvalidInput(format!(
            "Measure q has length {} but cost_y has size {m}",
            q.len()
        )));
    }
    if cost_x.iter().any(|row| row.len() != n) {
        return Err(TransformError::InvalidInput(
            "cost_x must be a square matrix".to_string(),
        ));
    }
    if cost_y.iter().any(|row| row.len() != m) {
        return Err(TransformError::InvalidInput(
            "cost_y must be a square matrix".to_string(),
        ));
    }
    let p_sum: f64 = p.iter().sum();
    let q_sum: f64 = q.iter().sum();
    if (p_sum - 1.0).abs() > 1e-6 {
        return Err(TransformError::InvalidInput(format!(
            "Measure p must sum to 1, got {p_sum:.6}"
        )));
    }
    if (q_sum - 1.0).abs() > 1e-6 {
        return Err(TransformError::InvalidInput(format!(
            "Measure q must sum to 1, got {q_sum:.6}"
        )));
    }

    // Initialise transport plan as outer product T = p * q^T
    let mut t: Vec<f64> = (0..n)
        .flat_map(|i| (0..m).map(move |j| p[i] * q[j]))
        .collect();

    let mut prev_loss = f64::INFINITY;

    for iter in 0..config.max_iter {
        // Compute gradient G[i,j] = 2 ∑_{kl} (C_X[i,k] - C_Y[j,l])^2 T[k,l]
        let grad = gw_gradient(cost_x, cost_y, &t);

        // Sinkhorn step: solve regularised OT with cost = grad
        let t_new = sinkhorn_log_stabilized(p, q, &grad, config.epsilon, config.sinkhorn_max_iter);

        // Line search (simple fixed step: Frank-Wolfe with step size 1/(iter+2))
        let step = 2.0 / ((iter + 2) as f64);
        for k in 0..(n * m) {
            t[k] = (1.0 - step) * t[k] + step * t_new[k];
        }

        // Check convergence
        let loss = gw_loss(cost_x, cost_y, &t);
        if iter > 0 && (prev_loss - loss).abs() / (prev_loss.abs() + 1e-300) < config.tol {
            let gw_dist = loss.max(0.0).sqrt();
            return Ok(GwResult {
                transport_plan: t,
                gw_distance: gw_dist,
                n_iter: iter + 1,
            });
        }
        prev_loss = loss;
    }

    let loss = gw_loss(cost_x, cost_y, &t);
    let gw_dist = loss.max(0.0).sqrt();
    Ok(GwResult {
        transport_plan: t,
        gw_distance: gw_dist,
        n_iter: config.max_iter,
    })
}

// ─── Multi-marginal OT ────────────────────────────────────────────────────────

/// Multi-marginal optimal transport via tensor Sinkhorn (up to 3 marginals).
///
/// Solves the entropic multi-marginal OT problem:
/// min_{γ} ∑_I c(I) γ(I) - ε H(γ)
/// subject to each marginal gamma\_k = `marginals[k]`.
///
/// Uses tensor Sinkhorn iterations in log domain.
///
/// # Arguments
/// * `marginals` — slice of probability vectors (each sums to 1)
/// * `cost_fn`   — function mapping a multi-index `I = [i_0, i_1, ...]` to a cost value
/// * `epsilon`   — entropic regularisation
///
/// # Returns
/// Flattened tensor γ in row-major order (product of marginal sizes).
pub fn multi_marginal_ot(
    marginals: &[Vec<f64>],
    cost_fn: impl Fn(&[usize]) -> f64,
    epsilon: f64,
) -> Result<Vec<f64>> {
    let k = marginals.len();
    if k == 0 {
        return Err(TransformError::InvalidInput(
            "At least one marginal is required".to_string(),
        ));
    }
    if k > 3 {
        return Err(TransformError::InvalidInput(
            "multi_marginal_ot supports at most 3 marginals".to_string(),
        ));
    }
    if epsilon <= 0.0 {
        return Err(TransformError::InvalidInput(
            "epsilon must be positive".to_string(),
        ));
    }

    let dims: Vec<usize> = marginals.iter().map(|m| m.len()).collect();
    let total: usize = dims.iter().product();

    // Build log cost tensor
    let mut log_c = vec![0.0_f64; total];
    let indices: Vec<Vec<usize>> = enumerate_indices(&dims);
    for (flat, idx) in indices.iter().enumerate() {
        let c = cost_fn(idx);
        log_c[flat] = -c / epsilon;
    }

    // Log marginals
    let log_marginals: Vec<Vec<f64>> = marginals
        .iter()
        .map(|marg| {
            marg.iter()
                .map(|&x| if x > 0.0 { x.ln() } else { f64::NEG_INFINITY })
                .collect()
        })
        .collect();

    // Dual variables (one per marginal, in log space)
    let mut duals: Vec<Vec<f64>> = dims.iter().map(|&d| vec![0.0_f64; d]).collect();

    let max_iter = 500;
    for _iter in 0..max_iter {
        for target_k in 0..k {
            // Update dual[target_k][i_k] = log_marginals[target_k][i_k]
            //   - logsumexp over other indices of (sum_{other j} duals[j][i_j] + log_c[I])

            let new_dual: Vec<f64> = (0..dims[target_k])
                .map(|ik| {
                    // Collect all terms that have i_{target_k} = ik
                    let terms: Vec<f64> = indices
                        .iter()
                        .enumerate()
                        .filter(|(_, idx)| idx[target_k] == ik)
                        .map(|(flat, idx)| {
                            let other_sum: f64 = (0..k)
                                .filter(|&j| j != target_k)
                                .map(|j| duals[j][idx[j]])
                                .sum();
                            log_c[flat] + other_sum
                        })
                        .collect();
                    log_marginals[target_k][ik] - log_sum_exp_vec(terms)
                })
                .collect();

            duals[target_k] = new_dual;
        }
    }

    // Reconstruct transport tensor
    let mut gamma = vec![0.0_f64; total];
    for (flat, idx) in indices.iter().enumerate() {
        let dual_sum: f64 = (0..k).map(|j| duals[j][idx[j]]).sum();
        let log_val = log_c[flat] + dual_sum;
        gamma[flat] = if log_val < -500.0 { 0.0 } else { log_val.exp() };
    }

    Ok(gamma)
}

/// Enumerate all multi-indices for a product space with given dimensions.
fn enumerate_indices(dims: &[usize]) -> Vec<Vec<usize>> {
    let mut result = vec![vec![]];
    for &d in dims {
        let mut next = Vec::new();
        for idx in &result {
            for i in 0..d {
                let mut new_idx = idx.clone();
                new_idx.push(i);
                next.push(new_idx);
            }
        }
        result = next;
    }
    result
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Three-point metric space: a line [0, 1, 2].
    fn line_space() -> (Vec<Vec<f64>>, Vec<f64>) {
        let cost = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 1.0],
            vec![2.0, 1.0, 0.0],
        ];
        let p = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        (cost, p)
    }

    /// Three-point metric space: an equilateral triangle (all distances = 1).
    fn triangle_space() -> (Vec<Vec<f64>>, Vec<f64>) {
        let cost = vec![
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ];
        let q = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        (cost, q)
    }

    #[test]
    fn test_gw_distance_positive() {
        let (cx, p) = line_space();
        let (cy, q) = triangle_space();
        let config = GwConfig {
            epsilon: 0.05,
            max_iter: 50,
            tol: 1e-6,
            ..Default::default()
        };
        let result = gromov_wasserstein(&cx, &cy, &p, &q, &config).expect("GW should succeed");
        assert!(
            result.gw_distance > 0.0,
            "GW distance should be positive for non-isometric spaces"
        );
    }

    #[test]
    fn test_gw_transport_row_marginals() {
        let (cx, p) = line_space();
        let (cy, q) = triangle_space();
        let config = GwConfig {
            epsilon: 0.1,
            max_iter: 30,
            tol: 1e-6,
            ..Default::default()
        };
        let result = gromov_wasserstein(&cx, &cy, &p, &q, &config).expect("GW should succeed");
        let n = cx.len();
        let m = cy.len();
        let t = &result.transport_plan;

        // Row marginals should be approximately p
        for i in 0..n {
            let row_sum: f64 = (0..m).map(|j| t[i * m + j]).sum();
            assert!(
                (row_sum - p[i]).abs() < 0.05,
                "Row {i} sum = {row_sum:.4}, expected {:.4}",
                p[i]
            );
        }
    }

    #[test]
    fn test_gw_transport_col_marginals() {
        let (cx, p) = line_space();
        let (cy, q) = triangle_space();
        let config = GwConfig {
            epsilon: 0.1,
            max_iter: 30,
            tol: 1e-6,
            ..Default::default()
        };
        let result = gromov_wasserstein(&cx, &cy, &p, &q, &config).expect("GW should succeed");
        let n = cx.len();
        let m = cy.len();
        let t = &result.transport_plan;

        // Column marginals should be approximately q
        for j in 0..m {
            let col_sum: f64 = (0..n).map(|i| t[i * m + j]).sum();
            assert!(
                (col_sum - q[j]).abs() < 0.05,
                "Col {j} sum = {col_sum:.4}, expected {:.4}",
                q[j]
            );
        }
    }

    #[test]
    fn test_sinkhorn_log_stabilized_basic() {
        let a = vec![0.5, 0.5];
        let b = vec![0.5, 0.5];
        let cost = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let t = sinkhorn_log_stabilized(&a, &b, &cost, 0.1, 500);
        // Transport should be concentrated on diagonal
        assert!(t[0] > t[1], "T[0,0] should dominate T[0,1]");
        assert!(t[3] > t[2], "T[1,1] should dominate T[1,0]");
        // Marginals
        let row0: f64 = t[0] + t[1];
        assert!((row0 - 0.5).abs() < 0.01, "Row 0 marginal: {row0:.4}");
    }

    #[test]
    fn test_gw_invalid_input() {
        let cx = vec![vec![0.0]];
        let cy = vec![vec![0.0]];
        // Wrong p length
        let result = gromov_wasserstein(&cx, &cy, &[0.5, 0.5], &[1.0], &GwConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_marginal_ot_2_marginals() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let cost_fn = |idx: &[usize]| -> f64 {
            // Simple L1 distance
            (idx[0] as f64 - idx[1] as f64).abs()
        };
        let gamma = multi_marginal_ot(&[p, q], cost_fn, 0.1).expect("multi-marginal OT");
        assert_eq!(gamma.len(), 4);
        let total: f64 = gamma.iter().sum();
        assert!((total - 1.0).abs() < 0.05, "Total mass: {total:.4}");
    }

    #[test]
    fn test_multi_marginal_ot_3_marginals() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let r = vec![0.5, 0.5];
        let cost_fn = |idx: &[usize]| -> f64 {
            let i = idx[0] as f64;
            let j = idx[1] as f64;
            let k = idx[2] as f64;
            (i - j).abs() + (j - k).abs()
        };
        let gamma = multi_marginal_ot(&[p, q, r], cost_fn, 0.1).expect("3-marginal OT");
        assert_eq!(gamma.len(), 8);
    }

    #[test]
    fn test_multi_marginal_too_many_marginals() {
        let marginals: Vec<Vec<f64>> = vec![vec![1.0]; 4];
        let result = multi_marginal_ot(&marginals, |_| 0.0, 0.1);
        assert!(result.is_err());
    }
}
