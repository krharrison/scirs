//! Types for Distributed ADMM, PDMM, and EXTRA algorithms.
//!
//! Provides configuration, result, and node structures for
//! consensus-based distributed optimization.
//!
//! # References
//! - Boyd et al. (2011). "Distributed Optimization and Statistical Learning via
//!   the Alternating Direction Method of Multipliers." Foundations and Trends in ML.
//! - Chang et al. (2015). "Multi-Agent Distributed Optimization via Inexact
//!   Consensus ADMM." IEEE Trans. Signal Processing.
//! - Shi et al. (2015). "EXTRA: An Exact First-Order Algorithm for Decentralized
//!   Consensus Optimization." SIAM J. Optim.

/// Configuration for ADMM (Alternating Direction Method of Multipliers).
///
/// ADMM solves the consensus problem: min Σ_i f_i(x_i) s.t. x_i = z
/// using augmented Lagrangian decomposition.
#[derive(Debug, Clone)]
pub struct AdmmConfig {
    /// Augmented Lagrangian penalty parameter ρ > 0.
    pub rho: f64,
    /// Maximum number of ADMM iterations.
    pub max_iter: usize,
    /// Absolute stopping tolerance for primal/dual residuals.
    pub abs_tol: f64,
    /// Relative stopping tolerance.
    pub rel_tol: f64,
    /// Enable warm-starting from a previous solution.
    pub warm_start: bool,
    /// Over-relaxation parameter α ∈ (0,2). Default 1.0 = no relaxation.
    pub over_relaxation: f64,
}

impl Default for AdmmConfig {
    fn default() -> Self {
        Self {
            rho: 1.0,
            max_iter: 1000,
            abs_tol: 1e-4,
            rel_tol: 1e-3,
            warm_start: false,
            over_relaxation: 1.0,
        }
    }
}

/// Configuration for PDMM (Primal-Dual Method of Multipliers).
///
/// PDMM is a decentralized method where each agent communicates only
/// with its immediate neighbours in a graph.
#[derive(Debug, Clone)]
pub struct PdmmConfig {
    /// Step-size for primal and dual updates.
    pub stepsize: f64,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Stopping tolerance for consensus (||x_i - x_j||).
    pub tol: f64,
}

impl Default for PdmmConfig {
    fn default() -> Self {
        Self {
            stepsize: 0.5,
            max_iter: 500,
            tol: 1e-4,
        }
    }
}

/// Configuration for EXTRA (Exact first-order decentralized algorithm).
///
/// EXTRA uses gradient tracking to converge to the exact consensus
/// without diminishing step sizes.
#[derive(Debug, Clone)]
pub struct ExtraConfig {
    /// Fixed step size α (must satisfy α < 2/(L_max(W̃) + ρ_max(∇²F)) for convergence).
    pub alpha: f64,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Stopping tolerance.
    pub tol: f64,
}

impl Default for ExtraConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            max_iter: 500,
            tol: 1e-4,
        }
    }
}

/// Result of a distributed optimization run.
#[derive(Debug, Clone)]
pub struct AdmmResult {
    /// Consensus solution x* (averaged over all agents at convergence).
    pub x: Vec<f64>,
    /// History of primal residuals ||x - z||₂ per iteration.
    pub primal_residual: Vec<f64>,
    /// History of dual residuals ρ||Δz||₂ per iteration.
    pub dual_residual: Vec<f64>,
    /// Whether the algorithm converged within the given tolerances.
    pub converged: bool,
    /// Number of iterations taken.
    pub iterations: usize,
}

/// A single ADMM consensus node (agent).
///
/// Stores the local primal variable, the consensus variable copy,
/// and the corresponding dual variable (scaled Lagrange multiplier).
#[derive(Debug, Clone)]
pub struct ConsensusNode {
    /// Local primal variable x_i.
    pub local_x: Vec<f64>,
    /// Agent's copy of the consensus variable z (shared globally).
    pub local_z: Vec<f64>,
    /// Scaled dual variable y_i = u_i = λ_i/ρ.
    pub dual_y: Vec<f64>,
}

impl ConsensusNode {
    /// Create a new node initialised at zero.
    pub fn new(n_vars: usize) -> Self {
        Self {
            local_x: vec![0.0; n_vars],
            local_z: vec![0.0; n_vars],
            dual_y: vec![0.0; n_vars],
        }
    }

    /// Initialise from a warm-start solution.
    pub fn warm(x0: Vec<f64>) -> Self {
        let n = x0.len();
        Self {
            local_z: x0.clone(),
            dual_y: vec![0.0; n],
            local_x: x0,
        }
    }

    /// Primal residual contribution: x_i - z.
    pub fn primal_residual(&self) -> Vec<f64> {
        self.local_x
            .iter()
            .zip(self.local_z.iter())
            .map(|(xi, zi)| xi - zi)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_admm_config_default() {
        let cfg = AdmmConfig::default();
        assert!((cfg.rho - 1.0).abs() < 1e-15);
        assert_eq!(cfg.max_iter, 1000);
        assert!((cfg.abs_tol - 1e-4).abs() < 1e-15);
        assert!(!cfg.warm_start);
    }

    #[test]
    fn test_pdmm_config_default() {
        let cfg = PdmmConfig::default();
        assert!((cfg.stepsize - 0.5).abs() < 1e-15);
        assert_eq!(cfg.max_iter, 500);
    }

    #[test]
    fn test_extra_config_default() {
        let cfg = ExtraConfig::default();
        assert!((cfg.alpha - 0.05).abs() < 1e-15);
        assert_eq!(cfg.max_iter, 500);
    }

    #[test]
    fn test_consensus_node_new() {
        let node = ConsensusNode::new(3);
        assert_eq!(node.local_x.len(), 3);
        assert!(node.local_x.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_consensus_node_warm() {
        let node = ConsensusNode::warm(vec![1.0, 2.0, 3.0]);
        assert_eq!(node.local_x, vec![1.0, 2.0, 3.0]);
        assert_eq!(node.local_z, vec![1.0, 2.0, 3.0]);
        assert!(node.dual_y.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_primal_residual() {
        let mut node = ConsensusNode::new(2);
        node.local_x = vec![1.0, 2.0];
        node.local_z = vec![0.5, 1.5];
        let res = node.primal_residual();
        assert!((res[0] - 0.5).abs() < 1e-15);
        assert!((res[1] - 0.5).abs() < 1e-15);
    }
}
