//! Types and configuration structs for second-order stochastic optimization.

/// Configuration for L-BFGS-B optimizer (limited-memory BFGS with box constraints).
///
/// Reference: Byrd, Lu, Nocedal & Zhu (1995) "A limited memory algorithm for
/// bound constrained optimization." SIAM J. Sci. Comput. 16(5): 1190–1208.
#[derive(Debug, Clone)]
pub struct LbfgsBConfig {
    /// Number of correction pairs to retain (memory size), default 10.
    pub m: usize,
    /// Maximum number of outer iterations.
    pub max_iter: usize,
    /// Gradient norm convergence tolerance.
    pub tol: f64,
    /// Wolfe sufficient decrease constant (Armijo condition), c₁ ∈ (0, c₂).
    pub c1: f64,
    /// Wolfe curvature condition constant, c₂ ∈ (c₁, 1).
    pub c2: f64,
    /// Maximum number of line-search iterations per outer step.
    pub max_ls_iter: usize,
    /// Initial step size for line search.
    pub alpha_init: f64,
}

impl Default for LbfgsBConfig {
    fn default() -> Self {
        Self {
            m: 10,
            max_iter: 1000,
            tol: 1e-6,
            c1: 1e-4,
            c2: 0.9,
            max_ls_iter: 60,
            alpha_init: 1.0,
        }
    }
}

/// Configuration for SR1 quasi-Newton optimizer.
///
/// Reference: Byrd, Khalfan & Schnabel (1994) "Analysis of a symmetric rank-one
/// trust region method." SIAM J. Optim. 4(1): 1–21.
#[derive(Debug, Clone)]
pub struct Sr1Config {
    /// Number of stored curvature pairs for limited-memory SR1.
    pub m: usize,
    /// Skip threshold: skip SR1 update if |(y - Bs)^T s| < skip_tol·‖s‖·‖y - Bs‖.
    pub skip_tol: f64,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Gradient norm convergence tolerance.
    pub tol: f64,
    /// Initial trust-region radius.
    pub delta_init: f64,
    /// Maximum trust-region radius.
    pub delta_max: f64,
    /// Trust-region acceptance threshold (ratio of actual to predicted reduction).
    pub eta: f64,
}

impl Default for Sr1Config {
    fn default() -> Self {
        Self {
            m: 10,
            skip_tol: 1e-8,
            max_iter: 1000,
            tol: 1e-6,
            delta_init: 1.0,
            delta_max: 100.0,
            eta: 0.1,
        }
    }
}

/// Configuration for Stochastic L-BFGS (S-L-BFGS) optimizer.
///
/// Reference: Moritz, Nishihara & Jordan (2016) "A linearly-convergent stochastic
/// L-BFGS algorithm." AISTATS.
#[derive(Debug, Clone)]
pub struct SlbfgsConfig {
    /// Number of correction pairs to retain (memory size).
    pub m: usize,
    /// Mini-batch size for stochastic gradient estimates.
    pub batch_size: usize,
    /// Whether to use SVRG-style variance reduction.
    pub variance_reduction: bool,
    /// Frequency (in outer iterations) at which to recompute the snapshot gradient.
    pub snapshot_freq: usize,
    /// Large-batch size for computing curvature pairs y_k (should be > batch_size).
    pub curvature_batch_size: usize,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Gradient norm convergence tolerance.
    pub tol: f64,
    /// Step size (learning rate).
    pub lr: f64,
    /// LCG seed for mini-batch selection.
    pub seed: u64,
}

impl Default for SlbfgsConfig {
    fn default() -> Self {
        Self {
            m: 10,
            batch_size: 32,
            variance_reduction: true,
            snapshot_freq: 10,
            curvature_batch_size: 128,
            max_iter: 500,
            tol: 1e-5,
            lr: 0.01,
            seed: 42,
        }
    }
}

/// Result returned by all second-order optimizers.
#[derive(Debug, Clone)]
pub struct OptResult {
    /// Final iterate.
    pub x: Vec<f64>,
    /// Objective value at final iterate.
    pub f_val: f64,
    /// Gradient norm at final iterate.
    pub grad_norm: f64,
    /// Number of outer iterations completed.
    pub n_iter: usize,
    /// Whether the optimizer converged to the specified tolerance.
    pub converged: bool,
}

/// Hessian approximation strategy.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HessianApprox {
    /// Limited-memory BFGS (default L-BFGS).
    Lbfgs,
    /// Symmetric rank-1 update.
    Sr1,
    /// Full BFGS (dense).
    Bfgs,
    /// Diagonal approximation (scaled identity).
    Diagonal,
}
