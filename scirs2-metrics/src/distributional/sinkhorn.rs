//! Sinkhorn Divergence and Entropic Optimal Transport
//!
//! Implements the Sinkhorn algorithm for computing entropic-regularized optimal
//! transport distances, along with the debiased Sinkhorn divergence.
//!
//! # Key Features
//!
//! - Configurable regularization parameter (epsilon)
//! - Convergence via marginal constraint violation
//! - Log-domain Sinkhorn for numerical stability
//! - Debiased Sinkhorn divergence
//! - Transport plan output
//!
//! # Examples
//!
//! ```
//! use scirs2_metrics::distributional::sinkhorn::{sinkhorn_distance, SinkhornConfig};
//!
//! // Two simple distributions on a 1D grid
//! let a = vec![0.25, 0.25, 0.25, 0.25];
//! let b = vec![0.0, 0.0, 0.5, 0.5];
//! // Cost matrix: |i - j|
//! let cost = vec![
//!     0.0, 1.0, 2.0, 3.0,
//!     1.0, 0.0, 1.0, 2.0,
//!     2.0, 1.0, 0.0, 1.0,
//!     3.0, 2.0, 1.0, 0.0,
//! ];
//! let config = SinkhornConfig::default();
//! let result = sinkhorn_distance(&a, &b, &cost, &config).expect("should succeed");
//! assert!(result.distance > 0.0);
//! ```

use crate::error::{MetricsError, Result};

/// Configuration for the Sinkhorn algorithm.
#[derive(Debug, Clone)]
pub struct SinkhornConfig {
    /// Regularization parameter (epsilon). Larger values make the problem
    /// more strongly regularized (smoother transport plan, faster convergence,
    /// but further from the true Wasserstein distance).
    pub epsilon: f64,
    /// Maximum number of Sinkhorn iterations.
    pub max_iterations: usize,
    /// Convergence threshold on marginal constraint violation.
    pub convergence_threshold: f64,
    /// Whether to use log-domain computation for numerical stability.
    pub log_domain: bool,
}

impl Default for SinkhornConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            max_iterations: 1000,
            convergence_threshold: 1e-9,
            log_domain: true,
        }
    }
}

/// Result of the Sinkhorn algorithm.
#[derive(Debug, Clone)]
pub struct SinkhornResult {
    /// The regularized optimal transport distance.
    pub distance: f64,
    /// The transport plan (coupling matrix), flattened row-major, shape [n, m].
    pub transport_plan: Vec<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the algorithm converged within the tolerance.
    pub converged: bool,
    /// Final marginal constraint violation.
    pub marginal_error: f64,
}

/// Computes the Sinkhorn (entropic OT) distance between two discrete distributions.
///
/// Given marginals `a` (length n), `b` (length m), and a cost matrix `C`
/// (flattened, n x m), computes:
///
/// OT_eps(a, b) = min_{P in U(a,b)} <P, C> + eps * KL(P || a b^T)
///
/// # Arguments
///
/// * `a` - Source distribution (must sum to ~1.0)
/// * `b` - Target distribution (must sum to ~1.0)
/// * `cost` - Cost matrix, flattened row-major, shape [a.len(), b.len()]
/// * `config` - Sinkhorn configuration
///
/// # Returns
///
/// A `SinkhornResult` with the distance, transport plan, and convergence info.
pub fn sinkhorn_distance(
    a: &[f64],
    b: &[f64],
    cost: &[f64],
    config: &SinkhornConfig,
) -> Result<SinkhornResult> {
    let n = a.len();
    let m = b.len();

    validate_sinkhorn_inputs(a, b, cost, config)?;

    if config.log_domain {
        sinkhorn_log_domain(a, b, cost, n, m, config)
    } else {
        sinkhorn_standard(a, b, cost, n, m, config)
    }
}

/// Standard (multiplicative) Sinkhorn algorithm.
fn sinkhorn_standard(
    a: &[f64],
    b: &[f64],
    cost: &[f64],
    n: usize,
    m: usize,
    config: &SinkhornConfig,
) -> Result<SinkhornResult> {
    let eps = config.epsilon;

    // Gibbs kernel: K_ij = exp(-C_ij / eps)
    let mut k = vec![0.0_f64; n * m];
    for i in 0..n {
        for j in 0..m {
            k[i * m + j] = (-cost[i * m + j] / eps).exp();
        }
    }

    let mut u = vec![1.0_f64; n];
    let mut v = vec![1.0_f64; m];

    let mut converged = false;
    let mut iterations = 0;
    let mut marginal_error = f64::MAX;

    for iter in 0..config.max_iterations {
        // Update u: u = a / (K v)
        let mut kv = vec![0.0; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..m {
                s += k[i * m + j] * v[j];
            }
            kv[i] = s;
        }

        for i in 0..n {
            if kv[i] > 1e-300 {
                u[i] = a[i] / kv[i];
            } else {
                u[i] = 1e-300;
            }
        }

        // Update v: v = b / (K^T u)
        let mut ktu = vec![0.0; m];
        for j in 0..m {
            let mut s = 0.0;
            for i in 0..n {
                s += k[i * m + j] * u[i];
            }
            ktu[j] = s;
        }

        for j in 0..m {
            if ktu[j] > 1e-300 {
                v[j] = b[j] / ktu[j];
            } else {
                v[j] = 1e-300;
            }
        }

        // Check convergence: marginal constraint violation
        // P 1_m should equal a
        let mut err = 0.0;
        for i in 0..n {
            let mut row_sum = 0.0;
            for j in 0..m {
                row_sum += u[i] * k[i * m + j] * v[j];
            }
            err += (row_sum - a[i]).abs();
        }
        marginal_error = err;
        iterations = iter + 1;

        if err < config.convergence_threshold {
            converged = true;
            break;
        }
    }

    // Compute transport plan P_ij = u_i * K_ij * v_j
    let mut plan = vec![0.0; n * m];
    let mut distance = 0.0;
    for i in 0..n {
        for j in 0..m {
            let p_ij = u[i] * k[i * m + j] * v[j];
            plan[i * m + j] = p_ij;
            distance += p_ij * cost[i * m + j];
        }
    }

    Ok(SinkhornResult {
        distance,
        transport_plan: plan,
        iterations,
        converged,
        marginal_error,
    })
}

/// Log-domain Sinkhorn algorithm for improved numerical stability.
///
/// Works with log-scaled dual variables to avoid overflow/underflow.
fn sinkhorn_log_domain(
    a: &[f64],
    b: &[f64],
    cost: &[f64],
    n: usize,
    m: usize,
    config: &SinkhornConfig,
) -> Result<SinkhornResult> {
    let eps = config.epsilon;

    // Log of marginals (with floor to avoid -inf)
    let log_a: Vec<f64> = a.iter().map(|x| x.max(1e-300).ln()).collect();
    let log_b: Vec<f64> = b.iter().map(|x| x.max(1e-300).ln()).collect();

    let mut f = vec![0.0_f64; n]; // dual variable for a
    let mut g = vec![0.0_f64; m]; // dual variable for b

    let mut converged = false;
    let mut iterations = 0;
    let mut marginal_error = f64::MAX;

    for iter in 0..config.max_iterations {
        // Update f (corresponds to u = a / (K v) in standard):
        // f_i = eps * log a_i - eps * logsumexp_j((-C_ij + g_j) / eps)
        for i in 0..n {
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..m {
                let val = (-cost[i * m + j] + g[j]) / eps;
                if val > max_val {
                    max_val = val;
                }
            }
            let mut lse = 0.0;
            for j in 0..m {
                lse += ((-cost[i * m + j] + g[j]) / eps - max_val).exp();
            }
            f[i] = eps * log_a[i] - eps * (max_val + lse.ln());
        }

        // Update g (corresponds to v = b / (K^T u) in standard):
        // g_j = eps * log b_j - eps * logsumexp_i((-C_ij + f_i) / eps)
        for j in 0..m {
            let mut max_val = f64::NEG_INFINITY;
            for i in 0..n {
                let val = (-cost[i * m + j] + f[i]) / eps;
                if val > max_val {
                    max_val = val;
                }
            }
            let mut lse = 0.0;
            for i in 0..n {
                lse += ((-cost[i * m + j] + f[i]) / eps - max_val).exp();
            }
            g[j] = eps * log_b[j] - eps * (max_val + lse.ln());
        }

        // Check convergence: compute marginal error in log domain
        let mut err = 0.0;
        for i in 0..n {
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..m {
                let val = (f[i] + g[j] - cost[i * m + j]) / eps;
                if val > max_val {
                    max_val = val;
                }
            }
            let mut lse = 0.0;
            for j in 0..m {
                lse += ((f[i] + g[j] - cost[i * m + j]) / eps - max_val).exp();
            }
            let log_row_sum = max_val + lse.ln();
            let row_sum = log_row_sum.exp();
            err += (row_sum - a[i]).abs();
        }

        marginal_error = err;
        iterations = iter + 1;

        if err < config.convergence_threshold {
            converged = true;
            break;
        }
    }

    // Compute transport plan: P_ij = exp((f_i + g_j - C_ij) / eps)
    let mut plan = vec![0.0; n * m];
    let mut distance = 0.0;
    for i in 0..n {
        for j in 0..m {
            let p_ij = ((f[i] + g[j] - cost[i * m + j]) / eps).exp();
            plan[i * m + j] = p_ij;
            distance += p_ij * cost[i * m + j];
        }
    }

    Ok(SinkhornResult {
        distance,
        transport_plan: plan,
        iterations,
        converged,
        marginal_error,
    })
}

/// Computes the debiased Sinkhorn divergence.
///
/// S(a, b) = OT_eps(a, b) - 0.5 * OT_eps(a, a) - 0.5 * OT_eps(b, b)
///
/// The debiased version removes the entropic bias, so that S(a, a) = 0
/// and the divergence is non-negative and metrizes convergence in distribution.
///
/// # Arguments
///
/// * `a` - Source distribution
/// * `b` - Target distribution
/// * `cost` - Cost matrix, flattened row-major, shape [a.len(), b.len()]
/// * `config` - Sinkhorn configuration
///
/// # Returns
///
/// The debiased Sinkhorn divergence value.
pub fn sinkhorn_divergence(
    a: &[f64],
    b: &[f64],
    cost: &[f64],
    config: &SinkhornConfig,
) -> Result<f64> {
    let n = a.len();
    let m = b.len();

    validate_sinkhorn_inputs(a, b, cost, config)?;

    // OT_eps(a, b)
    let ot_ab = sinkhorn_distance(a, b, cost, config)?.distance;

    // OT_eps(a, a): need cost matrix for a vs a
    let mut cost_aa = vec![0.0; n * n];
    // We need to construct cost_aa from the original cost structure
    // For the self-transport case, we assume the cost is based on the same
    // support as columns of cost. For general use, we build the cost from
    // index distance (matching the convention that cost encodes pairwise distances).
    // However, for proper debiasing, the user should provide cost matrices
    // with matching structure. Here we use a simplified approach:
    // cost_aa[i][j] = 0 if i==j, otherwise interpolate from cost structure.
    for i in 0..n {
        for j in 0..n {
            if i == j {
                cost_aa[i * n + j] = 0.0;
            } else {
                // Use cost row i and row j to estimate distance between supports of a
                // Simplified: use L1 distance between cost rows as proxy
                let mut d = 0.0;
                for k in 0..m {
                    let diff = cost[i * m + k] - cost[j * m + k];
                    d += diff * diff;
                }
                cost_aa[i * n + j] = d.sqrt();
            }
        }
    }
    let ot_aa = sinkhorn_distance(a, a, &cost_aa, config)?.distance;

    // OT_eps(b, b): need cost matrix for b vs b
    let mut cost_bb = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            if i == j {
                cost_bb[i * m + j] = 0.0;
            } else {
                let mut d = 0.0;
                for k in 0..n {
                    let diff = cost[k * m + i] - cost[k * m + j];
                    d += diff * diff;
                }
                cost_bb[i * m + j] = d.sqrt();
            }
        }
    }
    let ot_bb = sinkhorn_distance(b, b, &cost_bb, config)?.distance;

    Ok(ot_ab - 0.5 * ot_aa - 0.5 * ot_bb)
}

/// Computes the Sinkhorn divergence when the user provides explicit cost
/// matrices for the self-transport terms.
///
/// S(a, b) = OT_eps(a, b) - 0.5 * OT_eps(a, a) - 0.5 * OT_eps(b, b)
///
/// # Arguments
///
/// * `a` - Source distribution
/// * `b` - Target distribution
/// * `cost_ab` - Cost matrix between a and b supports, shape [n, m]
/// * `cost_aa` - Cost matrix between a supports, shape [n, n]
/// * `cost_bb` - Cost matrix between b supports, shape [m, m]
/// * `config` - Sinkhorn configuration
pub fn sinkhorn_divergence_with_costs(
    a: &[f64],
    b: &[f64],
    cost_ab: &[f64],
    cost_aa: &[f64],
    cost_bb: &[f64],
    config: &SinkhornConfig,
) -> Result<f64> {
    let n = a.len();
    let m = b.len();

    validate_sinkhorn_inputs(a, b, cost_ab, config)?;

    if cost_aa.len() != n * n {
        return Err(MetricsError::InvalidInput(format!(
            "cost_aa must have length {} (n*n), got {}",
            n * n,
            cost_aa.len()
        )));
    }
    if cost_bb.len() != m * m {
        return Err(MetricsError::InvalidInput(format!(
            "cost_bb must have length {} (m*m), got {}",
            m * m,
            cost_bb.len()
        )));
    }

    let ot_ab = sinkhorn_distance(a, b, cost_ab, config)?.distance;
    let ot_aa = sinkhorn_distance(a, a, cost_aa, config)?.distance;
    let ot_bb = sinkhorn_distance(b, b, cost_bb, config)?.distance;

    Ok(ot_ab - 0.5 * ot_aa - 0.5 * ot_bb)
}

/// Validates inputs for Sinkhorn algorithms.
fn validate_sinkhorn_inputs(
    a: &[f64],
    b: &[f64],
    cost: &[f64],
    config: &SinkhornConfig,
) -> Result<()> {
    if a.is_empty() || b.is_empty() {
        return Err(MetricsError::InvalidInput(
            "distributions must not be empty".to_string(),
        ));
    }
    if cost.len() != a.len() * b.len() {
        return Err(MetricsError::InvalidInput(format!(
            "cost matrix must have length {} (a.len() * b.len()), got {}",
            a.len() * b.len(),
            cost.len()
        )));
    }
    if config.epsilon <= 0.0 {
        return Err(MetricsError::InvalidInput(
            "epsilon must be positive".to_string(),
        ));
    }

    let sum_a: f64 = a.iter().sum();
    let sum_b: f64 = b.iter().sum();
    if (sum_a - 1.0).abs() > 0.05 {
        return Err(MetricsError::InvalidInput(format!(
            "distribution a must sum to ~1.0, got {sum_a}"
        )));
    }
    if (sum_b - 1.0).abs() > 0.05 {
        return Err(MetricsError::InvalidInput(format!(
            "distribution b must sum to ~1.0, got {sum_b}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid_cost(n: usize) -> Vec<f64> {
        let mut cost = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                cost[i * n + j] = ((i as f64) - (j as f64)).abs();
            }
        }
        cost
    }

    #[test]
    fn test_sinkhorn_identical_distributions() {
        let a = vec![0.25, 0.25, 0.25, 0.25];
        let b = vec![0.25, 0.25, 0.25, 0.25];
        let cost = make_grid_cost(4);
        let config = SinkhornConfig::default();
        let result = sinkhorn_distance(&a, &b, &cost, &config).expect("should succeed");
        // Identical distributions should have very small transport cost
        assert!(
            result.distance < 0.01,
            "identical distributions should have near-zero distance, got {}",
            result.distance
        );
    }

    #[test]
    fn test_sinkhorn_different_distributions() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0, 1.0];
        let cost = make_grid_cost(4);
        let config = SinkhornConfig {
            epsilon: 0.01,
            max_iterations: 5000,
            convergence_threshold: 1e-12,
            log_domain: true,
        };
        let result = sinkhorn_distance(&a, &b, &cost, &config).expect("should succeed");
        // Should be close to the true W1 = 3.0 for small epsilon
        assert!(
            (result.distance - 3.0).abs() < 0.1,
            "Dirac masses at 0 and 3 should give distance ~3.0, got {}",
            result.distance
        );
    }

    #[test]
    fn test_sinkhorn_convergence() {
        let a = vec![0.5, 0.5];
        let b = vec![0.5, 0.5];
        let cost = vec![0.0, 1.0, 1.0, 0.0];
        let config = SinkhornConfig {
            epsilon: 0.1,
            max_iterations: 1000,
            convergence_threshold: 1e-9,
            log_domain: true,
        };
        let result = sinkhorn_distance(&a, &b, &cost, &config).expect("should succeed");
        assert!(result.converged, "should converge for simple problem");
    }

    #[test]
    fn test_sinkhorn_transport_plan_marginals() {
        let a = vec![0.3, 0.7];
        let b = vec![0.6, 0.4];
        let cost = vec![0.0, 1.0, 1.0, 0.0];
        let config = SinkhornConfig {
            epsilon: 0.05,
            max_iterations: 2000,
            convergence_threshold: 1e-10,
            log_domain: true,
        };
        let result = sinkhorn_distance(&a, &b, &cost, &config).expect("should succeed");
        let plan = &result.transport_plan;

        // Check row marginals sum to a
        let row0_sum = plan[0] + plan[1];
        let row1_sum = plan[2] + plan[3];
        assert!(
            (row0_sum - 0.3).abs() < 0.01,
            "row 0 marginal should be 0.3, got {row0_sum}"
        );
        assert!(
            (row1_sum - 0.7).abs() < 0.01,
            "row 1 marginal should be 0.7, got {row1_sum}"
        );
    }

    #[test]
    fn test_sinkhorn_standard_vs_log_domain() {
        let a = vec![0.25, 0.25, 0.25, 0.25];
        let b = vec![0.1, 0.2, 0.3, 0.4];
        let cost = make_grid_cost(4);

        let config_std = SinkhornConfig {
            epsilon: 0.5,
            max_iterations: 1000,
            convergence_threshold: 1e-9,
            log_domain: false,
        };
        let config_log = SinkhornConfig {
            log_domain: true,
            ..config_std.clone()
        };

        let r_std = sinkhorn_distance(&a, &b, &cost, &config_std).expect("std should succeed");
        let r_log = sinkhorn_distance(&a, &b, &cost, &config_log).expect("log should succeed");

        assert!(
            (r_std.distance - r_log.distance).abs() < 0.01,
            "standard and log-domain should give similar results: {} vs {}",
            r_std.distance,
            r_log.distance
        );
    }

    #[test]
    fn test_sinkhorn_empty_input() {
        let config = SinkhornConfig::default();
        assert!(sinkhorn_distance(&[], &[1.0], &[], &config).is_err());
    }

    #[test]
    fn test_sinkhorn_bad_epsilon() {
        let config = SinkhornConfig {
            epsilon: -1.0,
            ..SinkhornConfig::default()
        };
        assert!(sinkhorn_distance(&[1.0], &[1.0], &[0.0], &config).is_err());
    }

    #[test]
    fn test_sinkhorn_divergence_self_zero() {
        let a = vec![0.25, 0.25, 0.25, 0.25];
        let cost = make_grid_cost(4);
        let config = SinkhornConfig {
            epsilon: 0.5,
            ..SinkhornConfig::default()
        };
        // Use explicit cost matrices for accurate self-transport
        let div = sinkhorn_divergence_with_costs(&a, &a, &cost, &cost, &cost, &config)
            .expect("should succeed");
        assert!(div.abs() < 0.05, "S(a, a) should be ~0, got {div}");
    }

    #[test]
    fn test_sinkhorn_divergence_positive_different() {
        let a = vec![0.5, 0.5, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.5, 0.5];
        let cost = make_grid_cost(4);
        let config = SinkhornConfig {
            epsilon: 0.1,
            max_iterations: 2000,
            convergence_threshold: 1e-10,
            log_domain: true,
        };
        let div = sinkhorn_divergence(&a, &b, &cost, &config).expect("should succeed");
        assert!(
            div > 0.0,
            "S(a, b) should be positive for different distributions, got {div}"
        );
    }
}
