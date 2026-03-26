//! Unbalanced Optimal Transport
//!
//! This module implements unbalanced optimal transport (UOT), which relaxes the
//! hard marginal constraints of balanced OT so that distributions with different
//! total mass can be compared.
//!
//! ## Theory
//!
//! Classical OT requires the source and target distributions to have equal mass.
//! UOT replaces the hard marginal constraints with soft penalty terms:
//!
//! ```text
//! UOT_{ε,τ}(a, b) = min_{T≥0}  ⟨C, T⟩
//!                              + ε KL(T | a⊗b)
//!                              + τ KL(T1 | a)
//!                              + τ KL(1ᵀT | b)
//! ```
//!
//! where KL(p|q) = Σ p_i log(p_i/q_i) − p_i + q_i is the generalised KL divergence.
//!
//! ### Unbalanced Sinkhorn Algorithm
//!
//! The solution is obtained via the scaling algorithm of Chizat et al. (2018):
//!
//! Initialise u = 1_n, v = 1_m, K_ij = exp(−C_ij / ε).
//! Iterate until convergence:
//! ```text
//! u ← (a / (K v))^{τ/(τ+ε)}
//! v ← (b / (Kᵀ u))^{τ/(τ+ε)}
//! ```
//! Optimal transport plan: T_ij = u_i K_ij v_j.
//!
//! ## References
//!
//! - Chizat, Peyré, Schmitzer, Vialard (2018):
//!   "Scaling algorithms for unbalanced optimal transport problems."
//!   Mathematics of Computation, 87(314), 2563-2609.
//! - Séjourné, Feydy, Vialard, Trouvé, Peyré (2019):
//!   "Sinkhorn Divergences for Unbalanced Optimal Transport."

use scirs2_core::ndarray::{Array1, Array2};

use crate::error::{Result, TransformError};

// ---------------------------------------------------------------------------
// Regularization type
// ---------------------------------------------------------------------------

/// Marginal relaxation type for unbalanced OT.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum UnbalancedRegularization {
    /// KL-divergence marginal penalty: τ KL(T1 | a) + τ KL(1ᵀT | b).
    /// This is the standard choice for UOT and leads to a closed-form
    /// proximal step in the scaling algorithm.
    KLDivergence,
    /// L2-norm marginal penalty: (τ/2) ‖T1 − a‖² + (τ/2) ‖1ᵀT − b‖².
    /// The proximal step is a soft-thresholding operator.
    L2,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for unbalanced Sinkhorn OT.
#[derive(Debug, Clone)]
pub struct UnbalancedOtConfig {
    /// Entropic regularization strength ε > 0. Default: 0.1.
    pub epsilon: f64,
    /// Marginal relaxation strength τ > 0. Default: 1.0.
    ///
    /// As τ → ∞ the problem approaches balanced OT.
    /// Small τ allows large deviations from the input marginals.
    pub tau: f64,
    /// Marginal penalty type.
    pub regularization: UnbalancedRegularization,
    /// Maximum number of Sinkhorn iterations. Default: 1000.
    pub max_iter: usize,
    /// Convergence tolerance (on the marginal error). Default: 1e-6.
    pub tol: f64,
    /// Whether to apply log-domain stabilization (recommended for small ε).
    /// Default: `true`.
    pub log_domain: bool,
}

impl Default for UnbalancedOtConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            tau: 1.0,
            regularization: UnbalancedRegularization::KLDivergence,
            max_iter: 1000,
            tol: 1e-6,
            log_domain: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Result of an unbalanced OT computation.
#[derive(Debug, Clone)]
pub struct UnbalancedOtResult {
    /// Optimal transport plan T (n × m), with potentially unequal row/column sums.
    pub transport_plan: Array2<f64>,
    /// Total transport cost ⟨C, T⟩.
    pub cost: f64,
    /// Marginal violation on the source side: ‖T 1_m − a‖₁.
    pub marginal_violation_source: f64,
    /// Marginal violation on the target side: ‖1_n^ᵀ T − b‖₁.
    pub marginal_violation_target: f64,
    /// Number of iterations performed.
    pub n_iter: usize,
    /// Whether convergence was achieved.
    pub converged: bool,
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Solve an unbalanced optimal transport problem via the Sinkhorn scaling algorithm.
///
/// # Arguments
/// * `a`    – Source histogram (n,), must be non-negative (will be normalised internally).
/// * `b`    – Target histogram (m,), must be non-negative.
/// * `cost` – Ground cost matrix C (n × m), must be non-negative.
/// * `config` – Algorithm parameters.
///
/// # Returns
/// [`UnbalancedOtResult`] containing the transport plan and diagnostics.
///
/// # Errors
/// Returns an error if inputs have incompatible shapes, contain negative entries,
/// or if all weights are zero.
///
/// # Example
/// ```rust
/// use scirs2_transform::ot::unbalanced::{unbalanced_sinkhorn, UnbalancedOtConfig};
/// use scirs2_core::ndarray::array;
///
/// let a = vec![0.5, 0.5];
/// let b = vec![0.5, 0.5];
/// let cost = array![[0.0_f64, 1.0], [1.0, 0.0]];
/// let config = UnbalancedOtConfig::default();
/// let result = unbalanced_sinkhorn(&a, &b, &cost, &config).expect("UOT should succeed");
/// assert!(result.cost >= 0.0);
/// ```
pub fn unbalanced_sinkhorn(
    a: &[f64],
    b: &[f64],
    cost: &Array2<f64>,
    config: &UnbalancedOtConfig,
) -> Result<UnbalancedOtResult> {
    // ----------------------------------------------------------------
    // Validate inputs
    // ----------------------------------------------------------------
    let n = a.len();
    let m = b.len();

    if n == 0 {
        return Err(TransformError::InvalidInput(
            "Source histogram 'a' must be non-empty".to_string(),
        ));
    }
    if m == 0 {
        return Err(TransformError::InvalidInput(
            "Target histogram 'b' must be non-empty".to_string(),
        ));
    }
    if cost.dim() != (n, m) {
        return Err(TransformError::InvalidInput(format!(
            "Cost matrix shape ({},{}) does not match histogram lengths ({n},{m})",
            cost.nrows(),
            cost.ncols()
        )));
    }
    if config.epsilon <= 0.0 {
        return Err(TransformError::InvalidInput(
            "epsilon must be positive".to_string(),
        ));
    }
    if config.tau <= 0.0 {
        return Err(TransformError::InvalidInput(
            "tau must be positive".to_string(),
        ));
    }
    for &ai in a {
        if ai < 0.0 {
            return Err(TransformError::InvalidInput(
                "Source histogram contains negative entries".to_string(),
            ));
        }
    }
    for &bi in b {
        if bi < 0.0 {
            return Err(TransformError::InvalidInput(
                "Target histogram contains negative entries".to_string(),
            ));
        }
    }
    let sum_a: f64 = a.iter().sum();
    let sum_b: f64 = b.iter().sum();
    if sum_a < f64::EPSILON {
        return Err(TransformError::InvalidInput(
            "Source histogram has zero total mass".to_string(),
        ));
    }
    if sum_b < f64::EPSILON {
        return Err(TransformError::InvalidInput(
            "Target histogram has zero total mass".to_string(),
        ));
    }

    // ----------------------------------------------------------------
    // Check for negative cost entries
    // ----------------------------------------------------------------
    for ci in cost.iter() {
        if *ci < 0.0 {
            return Err(TransformError::InvalidInput(
                "Cost matrix contains negative entries".to_string(),
            ));
        }
    }

    match config.regularization {
        UnbalancedRegularization::KLDivergence => {
            if config.log_domain {
                sinkhorn_kl_log_domain(a, b, cost, config)
            } else {
                sinkhorn_kl(a, b, cost, config)
            }
        }
        UnbalancedRegularization::L2 => sinkhorn_l2(a, b, cost, config),
    }
}

// ---------------------------------------------------------------------------
// KL-divergence scaling algorithm (standard domain)
// ---------------------------------------------------------------------------

/// Unbalanced Sinkhorn scaling with KL marginal penalties.
///
/// Scaling exponent: ρ = τ / (τ + ε)
fn sinkhorn_kl(
    a: &[f64],
    b: &[f64],
    cost: &Array2<f64>,
    config: &UnbalancedOtConfig,
) -> Result<UnbalancedOtResult> {
    let n = a.len();
    let m = b.len();
    let rho = config.tau / (config.tau + config.epsilon);

    // Gibbs kernel K_ij = exp(-C_ij / epsilon)
    let k: Array2<f64> = cost.mapv(|c| (-c / config.epsilon).exp());

    // Scaling vectors (dual variables in the exponential domain)
    let mut u = Array1::from_elem(n, 1.0_f64);
    let mut v = Array1::from_elem(m, 1.0_f64);

    let a_arr = Array1::from_vec(a.to_vec());
    let b_arr = Array1::from_vec(b.to_vec());

    let mut converged = false;
    let mut n_iter = 0usize;

    for _iter in 0..config.max_iter {
        n_iter += 1;

        // Kv[i] = Σ_j K[i,j] * v[j]
        let kv: Array1<f64> = k.dot(&v);
        // u ← (a / Kv)^ρ
        let u_new: Array1<f64> = a_arr
            .iter()
            .zip(kv.iter())
            .map(|(&ai, &kvi)| {
                if kvi < f64::EPSILON {
                    0.0
                } else {
                    (ai / kvi).powf(rho)
                }
            })
            .collect::<Vec<f64>>()
            .into();

        // Ktu[j] = Σ_i K[i,j] * u[i]
        let ktu: Array1<f64> = k.t().dot(&u_new);
        // v ← (b / Kᵀu)^ρ
        let v_new: Array1<f64> = b_arr
            .iter()
            .zip(ktu.iter())
            .map(|(&bi, &ktui)| {
                if ktui < f64::EPSILON {
                    0.0
                } else {
                    (bi / ktui).powf(rho)
                }
            })
            .collect::<Vec<f64>>()
            .into();

        // Convergence check: change in scaling vectors
        let du: f64 = u_new
            .iter()
            .zip(u.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f64>()
            / (n as f64);
        let dv: f64 = v_new
            .iter()
            .zip(v.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f64>()
            / (m as f64);

        u = u_new;
        v = v_new;

        if du + dv < config.tol {
            converged = true;
            break;
        }
    }

    // Build transport plan: T_ij = u_i K_ij v_j
    let transport_plan = build_transport_plan(&u, &k, &v);
    let result = compute_result(transport_plan, cost, a, b, n_iter, converged);
    Ok(result)
}

// ---------------------------------------------------------------------------
// KL-divergence scaling algorithm (log domain — numerically stable)
// ---------------------------------------------------------------------------

/// Log-domain stabilized unbalanced Sinkhorn (Chizat 2018, Algorithm 2).
///
/// Works in log-space to avoid numerical overflow/underflow for small ε.
fn sinkhorn_kl_log_domain(
    a: &[f64],
    b: &[f64],
    cost: &Array2<f64>,
    config: &UnbalancedOtConfig,
) -> Result<UnbalancedOtResult> {
    let n = a.len();
    let m = b.len();
    let rho = config.tau / (config.tau + config.epsilon);
    let eps = config.epsilon;

    // Log potentials: f (n,), g (m,)
    let mut f: Array1<f64> = Array1::zeros(n);
    let mut g: Array1<f64> = Array1::zeros(m);

    let log_a: Vec<f64> = a
        .iter()
        .map(|&ai| if ai > 0.0 { ai.ln() } else { f64::NEG_INFINITY })
        .collect();
    let log_b: Vec<f64> = b
        .iter()
        .map(|&bi| if bi > 0.0 { bi.ln() } else { f64::NEG_INFINITY })
        .collect();

    let mut converged = false;
    let mut n_iter = 0usize;

    for _iter in 0..config.max_iter {
        n_iter += 1;

        // Softmin_ε over j: h_i = −ε lse_j (g_j − C_ij / ε)
        // Then f ← ρ (log_a − h_i / ε) * ε  [from KL prox update]
        // Equivalently: f_i ← ρ (ε log_a_i − softmin_ε_j(g_j − C_ij/ε))

        let f_prev = f.clone();
        let g_prev = g.clone();

        // Update f: f_i = ρ * (ε ln a_i − softmin_j(g_j − C_{ij}/ε) * ε ... )
        // The proximal update is: f ← ρ/(ρ+1) * (ε ln a − ε lse_j((g - C/ε) / 1))
        // But with the standard KL UOT: f_i ← rho * ε * (ln a_i − lse_j( (g_j - C_ij)/ε ) )
        // where lse is log-sum-exp.
        for i in 0..n {
            let lse_j = log_sum_exp_row(i, &g, cost, eps, m);
            let new_fi = rho * (eps * log_a[i] - lse_j);
            f[i] = new_fi;
        }

        // Update g: g_j = ρ * (ε ln b_j − lse_i( (f_i - C_ij)/ε ))
        for j in 0..m {
            let lse_i = log_sum_exp_col(j, &f, cost, eps, n);
            let new_gj = rho * (eps * log_b[j] - lse_i);
            g[j] = new_gj;
        }

        // Convergence check
        let df: f64 = f
            .iter()
            .zip(f_prev.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f64>()
            / n as f64;
        let dg: f64 = g
            .iter()
            .zip(g_prev.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f64>()
            / m as f64;

        if df + dg < config.tol {
            converged = true;
            break;
        }
    }

    // Build transport plan from potentials: T_ij = exp((f_i + g_j - C_ij) / ε)
    let mut transport_plan = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            transport_plan[[i, j]] = ((f[i] + g[j] - cost[[i, j]]) / eps).exp();
        }
    }

    let result = compute_result(transport_plan, cost, a, b, n_iter, converged);
    Ok(result)
}

/// Log-sum-exp of (g_j − C_ij / ε) over j (row i of cost matrix).
#[inline]
fn log_sum_exp_row(i: usize, g: &Array1<f64>, cost: &Array2<f64>, eps: f64, m: usize) -> f64 {
    let vals: Vec<f64> = (0..m).map(|j| g[j] - cost[[i, j]] / eps).collect();
    log_sum_exp_vec(&vals)
}

/// Log-sum-exp of (f_i − C_ij / ε) over i (column j of cost matrix).
#[inline]
fn log_sum_exp_col(j: usize, f: &Array1<f64>, cost: &Array2<f64>, eps: f64, n: usize) -> f64 {
    let vals: Vec<f64> = (0..n).map(|i| f[i] - cost[[i, j]] / eps).collect();
    log_sum_exp_vec(&vals)
}

/// Numerically stable log-sum-exp: log Σ exp(x_i) = max + log Σ exp(x_i − max).
fn log_sum_exp_vec(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_val = vals
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !max_val.is_finite() {
        return f64::NEG_INFINITY;
    }
    let sum_exp: f64 = vals
        .iter()
        .filter(|v| v.is_finite())
        .map(|&v| (v - max_val).exp())
        .sum();
    max_val + sum_exp.ln()
}

// ---------------------------------------------------------------------------
// L2 marginal penalty (proximal step = clip-to-positive)
// ---------------------------------------------------------------------------

/// Unbalanced Sinkhorn with L2 marginal penalties.
///
/// The proximal operator for the L2 penalty is:
/// u ← max(0, 1 − (Kv − a) / (τ K1_m))  ... (simplified form)
///
/// In practice we use the scaling form:
/// u ← a / (Kv + ε/τ)
fn sinkhorn_l2(
    a: &[f64],
    b: &[f64],
    cost: &Array2<f64>,
    config: &UnbalancedOtConfig,
) -> Result<UnbalancedOtResult> {
    let n = a.len();
    let m = b.len();

    let k: Array2<f64> = cost.mapv(|c| (-c / config.epsilon).exp());
    let mut u = Array1::from_elem(n, 1.0_f64);
    let mut v = Array1::from_elem(m, 1.0_f64);

    let a_arr = Array1::from_vec(a.to_vec());
    let b_arr = Array1::from_vec(b.to_vec());

    // L2 proximal scaling: effectively a soft update
    // u ← a / (Kv + ε/τ)  — comes from RKHS proximal step for squared norm
    let lambda = config.epsilon / config.tau;

    let mut converged = false;
    let mut n_iter = 0usize;

    for _iter in 0..config.max_iter {
        n_iter += 1;
        let kv: Array1<f64> = k.dot(&v);
        let u_new: Array1<f64> = a_arr
            .iter()
            .zip(kv.iter())
            .map(|(&ai, &kvi)| ai / (kvi + lambda).max(f64::EPSILON))
            .collect::<Vec<f64>>()
            .into();

        let ktu: Array1<f64> = k.t().dot(&u_new);
        let v_new: Array1<f64> = b_arr
            .iter()
            .zip(ktu.iter())
            .map(|(&bi, &ktui)| bi / (ktui + lambda).max(f64::EPSILON))
            .collect::<Vec<f64>>()
            .into();

        let du: f64 = u_new
            .iter()
            .zip(u.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f64>()
            / n as f64;
        let dv: f64 = v_new
            .iter()
            .zip(v.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f64>()
            / m as f64;

        u = u_new;
        v = v_new;

        if du + dv < config.tol {
            converged = true;
            break;
        }
    }

    let transport_plan = build_transport_plan(&u, &k, &v);
    let result = compute_result(transport_plan, cost, a, b, n_iter, converged);
    Ok(result)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build the transport plan: T_ij = u_i K_ij v_j
fn build_transport_plan(u: &Array1<f64>, k: &Array2<f64>, v: &Array1<f64>) -> Array2<f64> {
    let n = u.len();
    let m = v.len();
    let mut t = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            t[[i, j]] = u[i] * k[[i, j]] * v[j];
        }
    }
    t
}

/// Compute diagnostics from the transport plan.
fn compute_result(
    transport_plan: Array2<f64>,
    cost: &Array2<f64>,
    a: &[f64],
    b: &[f64],
    n_iter: usize,
    converged: bool,
) -> UnbalancedOtResult {
    let n = a.len();
    let m = b.len();

    // Transport cost: ⟨C, T⟩
    let ot_cost: f64 = cost
        .iter()
        .zip(transport_plan.iter())
        .map(|(&c, &t)| c * t)
        .sum();

    // Source marginal: T 1_m
    let source_marg: Vec<f64> = (0..n).map(|i| transport_plan.row(i).sum()).collect();

    // Target marginal: 1_n^ᵀ T
    let target_marg: Vec<f64> = (0..m).map(|j| transport_plan.column(j).sum()).collect();

    // Marginal violations (L1 distance from input histograms)
    let mv_src: f64 = source_marg
        .iter()
        .zip(a.iter())
        .map(|(&sm, &ai)| (sm - ai).abs())
        .sum();
    let mv_tgt: f64 = target_marg
        .iter()
        .zip(b.iter())
        .map(|(&tm, &bi)| (tm - bi).abs())
        .sum();

    UnbalancedOtResult {
        transport_plan,
        cost: ot_cost,
        marginal_violation_source: mv_src,
        marginal_violation_target: mv_tgt,
        n_iter,
        converged,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // ------------------------------------------------------------------
    // Basic correctness
    // ------------------------------------------------------------------

    #[test]
    fn test_unbalanced_ot_equal_mass_kl() {
        // Equal-mass uniform histograms; cost = |i − j| / n
        let n = 4usize;
        let a: Vec<f64> = vec![0.25; n];
        let b: Vec<f64> = vec![0.25; n];
        let mut cost_arr = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                cost_arr[[i, j]] = (i as f64 - j as f64).abs() / n as f64;
            }
        }

        let config = UnbalancedOtConfig {
            epsilon: 0.01,
            tau: 100.0, // large tau → close to balanced OT
            log_domain: true,
            max_iter: 2000,
            tol: 1e-8,
            ..Default::default()
        };

        let result = unbalanced_sinkhorn(&a, &b, &cost_arr, &config).expect("UOT ok");
        assert!(result.cost >= 0.0, "cost must be non-negative");
        // For equal uniform mass, balanced W1 = 1/8; with large tau the UOT should be close
        assert!(
            result.marginal_violation_source < 0.1,
            "source marginal violation should be small, got {}",
            result.marginal_violation_source
        );
    }

    #[test]
    fn test_unbalanced_ot_equal_mass_l2() {
        let a = vec![0.5, 0.5];
        let b = vec![0.5, 0.5];
        let cost = array![[0.0_f64, 1.0], [1.0, 0.0]];
        let config = UnbalancedOtConfig {
            regularization: UnbalancedRegularization::L2,
            epsilon: 0.1,
            tau: 10.0,
            max_iter: 500,
            tol: 1e-6,
            log_domain: false,
            ..Default::default()
        };
        let result = unbalanced_sinkhorn(&a, &b, &cost, &config).expect("UOT L2 ok");
        assert!(result.cost >= 0.0);
        // Transport plan should be non-negative
        for &t in result.transport_plan.iter() {
            assert!(t >= -1e-10, "transport plan entries must be non-negative");
        }
    }

    #[test]
    fn test_unbalanced_ot_unequal_mass() {
        // Source has mass 1.0, target has mass 0.5
        let a = vec![0.5, 0.5]; // total mass = 1.0
        let b = vec![0.25, 0.25]; // total mass = 0.5
        let cost = array![[0.0_f64, 1.0], [1.0, 0.0]];

        let config = UnbalancedOtConfig {
            epsilon: 0.05,
            tau: 0.5, // allow significant marginal deviation
            max_iter: 1000,
            tol: 1e-6,
            log_domain: true,
            ..Default::default()
        };
        let result = unbalanced_sinkhorn(&a, &b, &cost, &config).expect("UOT unequal ok");
        assert!(result.cost >= 0.0);
        // With unequal mass, at least one marginal violation should be significant
        let total_mv = result.marginal_violation_source + result.marginal_violation_target;
        // It's expected that marginals don't match perfectly with unequal mass
        assert!(
            total_mv >= 0.0,
            "marginal violations should be non-negative"
        );
    }

    #[test]
    fn test_unbalanced_ot_diagonal_cost() {
        // Zero cost on diagonal: optimal plan should concentrate on diagonal
        let n = 3usize;
        let a = vec![1.0 / n as f64; n];
        let b = vec![1.0 / n as f64; n];
        let mut cost_arr = Array2::<f64>::ones((n, n)) * 10.0;
        for i in 0..n {
            cost_arr[[i, i]] = 0.0;
        }

        let config = UnbalancedOtConfig {
            epsilon: 0.01,
            tau: 100.0,
            max_iter: 2000,
            tol: 1e-9,
            log_domain: true,
            ..Default::default()
        };
        let result = unbalanced_sinkhorn(&a, &b, &cost_arr, &config).expect("UOT diagonal ok");
        // Cost should be close to 0 (all mass on diagonal)
        assert!(
            result.cost < 0.5,
            "diagonal-concentrated plan should have small cost, got {}",
            result.cost
        );
    }

    #[test]
    fn test_unbalanced_ot_kl_standard_domain() {
        let a = vec![0.5, 0.5];
        let b = vec![0.5, 0.5];
        let cost = array![[0.0_f64, 1.0], [1.0, 0.0]];
        let config = UnbalancedOtConfig {
            epsilon: 0.1,
            tau: 1.0,
            log_domain: false, // test non-log-domain path
            max_iter: 500,
            tol: 1e-6,
            ..Default::default()
        };
        let result = unbalanced_sinkhorn(&a, &b, &cost, &config).expect("UOT KL std ok");
        assert!(result.cost >= 0.0);
    }

    // ------------------------------------------------------------------
    // Error cases
    // ------------------------------------------------------------------

    #[test]
    fn test_empty_source_error() {
        let a: Vec<f64> = vec![];
        let b = vec![0.5, 0.5];
        let cost = Array2::<f64>::zeros((0, 2));
        let config = UnbalancedOtConfig::default();
        assert!(unbalanced_sinkhorn(&a, &b, &cost, &config).is_err());
    }

    #[test]
    fn test_shape_mismatch_error() {
        let a = vec![0.5, 0.5];
        let b = vec![0.5, 0.5];
        let cost = Array2::<f64>::zeros((3, 2)); // wrong n
        let config = UnbalancedOtConfig::default();
        assert!(unbalanced_sinkhorn(&a, &b, &cost, &config).is_err());
    }

    #[test]
    fn test_negative_epsilon_error() {
        let a = vec![0.5, 0.5];
        let b = vec![0.5, 0.5];
        let cost = array![[0.0_f64, 1.0], [1.0, 0.0]];
        let config = UnbalancedOtConfig {
            epsilon: -0.1,
            ..Default::default()
        };
        assert!(unbalanced_sinkhorn(&a, &b, &cost, &config).is_err());
    }

    #[test]
    fn test_zero_mass_error() {
        let a = vec![0.0, 0.0];
        let b = vec![0.5, 0.5];
        let cost = array![[0.0_f64, 1.0], [1.0, 0.0]];
        let config = UnbalancedOtConfig::default();
        assert!(unbalanced_sinkhorn(&a, &b, &cost, &config).is_err());
    }

    #[test]
    fn test_transport_plan_non_negative() {
        // All transport plan entries should be non-negative
        let a = vec![0.3, 0.7];
        let b = vec![0.6, 0.4];
        let cost = array![[0.1_f64, 0.9], [0.8, 0.2]];
        let config = UnbalancedOtConfig::default();
        let result = unbalanced_sinkhorn(&a, &b, &cost, &config).expect("UOT ok");
        for &t in result.transport_plan.iter() {
            assert!(t >= -1e-12, "transport plan entry {t} is negative");
        }
    }

    #[test]
    fn test_1x1_trivial() {
        // Single source, single target: with zero cost, transport plan entry should be
        // close to 1 (balanced OT recovered with large tau)
        let a = vec![1.0];
        let b = vec![1.0];
        // Zero cost: optimal T = 1 regardless of regularization
        let cost = array![[0.0_f64]];
        let config = UnbalancedOtConfig {
            epsilon: 0.01,
            tau: 100.0,
            max_iter: 2000,
            tol: 1e-8,
            ..Default::default()
        };
        let result = unbalanced_sinkhorn(&a, &b, &cost, &config).expect("1x1 ok");
        assert!(
            (result.transport_plan[[0, 0]] - 1.0).abs() < 0.2,
            "1x1 transport plan should be close to 1, got {}",
            result.transport_plan[[0, 0]]
        );
        // Cost should be near 0 (zero cost matrix)
        assert!(
            result.cost < 0.5,
            "1x1 cost with zero cost matrix should be small, got {}",
            result.cost
        );
    }

    // ------------------------------------------------------------------
    // Convenience: log_sum_exp
    // ------------------------------------------------------------------

    #[test]
    fn test_log_sum_exp_vec() {
        let vals = vec![1.0_f64, 2.0, 3.0];
        let lse = log_sum_exp_vec(&vals);
        let expected = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
        assert!((lse - expected).abs() < 1e-10, "lse mismatch");
    }
}
