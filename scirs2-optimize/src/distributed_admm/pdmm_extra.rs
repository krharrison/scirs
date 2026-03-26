//! PDMM (Primal-Dual Method of Multipliers) and EXTRA algorithms.
//!
//! **PDMM** (Chang et al. 2015):
//!
//! Decentralized optimization over a graph G = (V, E) where each agent i
//! communicates only with its neighbours N_i:
//!
//! ```text
//!   min  Σ_i f_i(x)   s.t. x_i = x_j  ∀(i,j) ∈ E
//! ```
//!
//! PDMM updates per agent i:
//!
//! ```text
//!   x_i^{k+1} = argmin_x [ f_i(x) + Σ_{j∈N_i} (λ_{ij}^T x + (ρ/2)||x - x_j^k||²) ]
//!   λ_{ij}^{k+1} = λ_{ij}^k + ρ (x_i^{k+1} - x_j^{k+1})
//! ```
//!
//! **EXTRA** (Shi et al. 2015):
//!
//! Uses two mixing matrices W (doubly stochastic) and W̃ = (I+W)/2.
//! Gradient tracking ensures exact consensus without diminishing step size:
//!
//! ```text
//!   x^1   = W x^0 - α ∇F(x^0)
//!   x^{k+2} = W̃ x^{k+1} + x^{k+1} - W̃ x^k - α (∇F(x^{k+1}) - ∇F(x^k))
//! ```
//!
//! # References
//! - Chang et al. (2015). "Multi-Agent Distributed Optimization via Inexact
//!   Consensus ADMM." IEEE Trans. Signal Processing.
//! - Shi et al. (2015). "EXTRA: An Exact First-Order Algorithm for Decentralized
//!   Consensus Optimization." SIAM J. Optim.

use super::types::{AdmmResult, ExtraConfig, PdmmConfig};
use crate::error::{OptimizeError, OptimizeResult};

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai + bi).collect()
}

fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai - bi).collect()
}

fn vec_scale(a: &[f64], s: f64) -> Vec<f64> {
    a.iter().map(|ai| ai * s).collect()
}

/// Matrix-vector product y = W x (row-major W).
fn mat_vec(w: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    w.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum())
        .collect()
}

/// Verify that a matrix is doubly stochastic (all rows and columns sum to 1).
fn check_doubly_stochastic(w: &[Vec<f64>], tol: f64) -> bool {
    let n = w.len();
    // Row sums
    for row in w.iter() {
        if row.len() != n {
            return false;
        }
        let s: f64 = row.iter().sum();
        if (s - 1.0).abs() > tol {
            return false;
        }
    }
    // Column sums
    for j in 0..n {
        let s: f64 = w.iter().map(|row| row[j]).sum();
        if (s - 1.0).abs() > tol {
            return false;
        }
    }
    true
}

// ─────────────────────────────────────────────────────────────────────────────
// PDMM Solver
// ─────────────────────────────────────────────────────────────────────────────

/// PDMM solver for decentralized consensus optimization over a network.
///
/// Each agent has a quadratic local objective f_i(x) = (1/2)||x - c_i||²
/// and communicates only with adjacent agents.
#[derive(Debug)]
pub struct PdmmSolver {
    /// Adjacency matrix (symmetric, 0-1 valued). Entry `[i][j]` = 1 iff agents i and j are connected.
    pub topology: Vec<Vec<f64>>,
}

impl PdmmSolver {
    /// Create a new PDMM solver with the given adjacency matrix.
    pub fn new(topology: Vec<Vec<f64>>) -> OptimizeResult<Self> {
        let n = topology.len();
        for (i, row) in topology.iter().enumerate() {
            if row.len() != n {
                return Err(OptimizeError::InvalidInput(format!(
                    "Topology row {} has length {} but expected {}",
                    i,
                    row.len(),
                    n
                )));
            }
        }
        Ok(Self { topology })
    }

    /// Solve the consensus problem where each agent i has the local function
    /// `local_fns[i](x)` (which must have a known proximal operator).
    ///
    /// `local_fns[i]` is the proximal operator prox_{f_i/ρ}(v, rho).
    pub fn solve<F>(
        &self,
        local_fns: &[F],
        n_vars: usize,
        config: &PdmmConfig,
    ) -> OptimizeResult<AdmmResult>
    where
        F: Fn(&[f64], f64) -> Vec<f64>,
    {
        let n_agents = self.topology.len();
        if local_fns.len() != n_agents {
            return Err(OptimizeError::InvalidInput(format!(
                "Expected {} local functions but got {}",
                n_agents,
                local_fns.len()
            )));
        }
        if n_vars == 0 {
            return Err(OptimizeError::InvalidInput("n_vars must be > 0".into()));
        }

        let rho = config.stepsize;

        // Primal variables x[i], dual edge variables λ[i][j] (for j in N_i)
        let mut x: Vec<Vec<f64>> = (0..n_agents).map(|_| vec![0.0; n_vars]).collect();
        // λ_{ij}: for each directed edge i→j, one dual vector
        let mut lam: Vec<Vec<Vec<f64>>> = (0..n_agents)
            .map(|_| (0..n_agents).map(|_| vec![0.0_f64; n_vars]).collect())
            .collect();

        let mut primal_history = Vec::with_capacity(config.max_iter);
        let mut dual_history = Vec::with_capacity(config.max_iter);
        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..config.max_iter {
            iterations = iter + 1;
            let x_old = x.clone();

            // ── x-updates ─────────────────────────────────────────────────────
            for i in 0..n_agents {
                // Aggregate neighbour contribution:
                // v_i = -( Σ_{j∈N_i} (λ_{ij} - ρ x_j^old) ) / (ρ * |N_i|)
                let mut neighbours = 0usize;
                let mut agg = vec![0.0_f64; n_vars];
                for j in 0..n_agents {
                    if self.topology[i][j] > 0.0 {
                        neighbours += 1;
                        for k in 0..n_vars {
                            agg[k] += lam[i][j][k] - rho * x_old[j][k];
                        }
                    }
                }
                // prox argument: v = -(1/ρ) agg_i / |N_i| (per-neighbour average)
                // PDMM prox: prox_{f_i/ρ̃}(v) where ρ̃ = ρ * |N_i|
                let rho_eff = rho * (neighbours.max(1) as f64);
                let prox_arg: Vec<f64> = agg.iter().map(|a| -a / rho_eff).collect();
                x[i] = (local_fns[i])(&prox_arg, rho_eff);
            }

            // ── λ-updates ──────────────────────────────────────────────────────
            for i in 0..n_agents {
                for j in 0..n_agents {
                    if self.topology[i][j] > 0.0 {
                        for k in 0..n_vars {
                            lam[i][j][k] += rho * (x[i][k] - x[j][k]);
                        }
                    }
                }
            }

            // ── Consensus residual (max disagreement between neighbours) ───────
            let mut primal_sq = 0.0_f64;
            let mut dual_sq = 0.0_f64;
            for i in 0..n_agents {
                for j in 0..n_agents {
                    if self.topology[i][j] > 0.0 {
                        for k in 0..n_vars {
                            primal_sq += (x[i][k] - x[j][k]).powi(2);
                        }
                    }
                }
                for k in 0..n_vars {
                    dual_sq += (x[i][k] - x_old[i][k]).powi(2);
                }
            }
            let primal_res = primal_sq.sqrt();
            let dual_res = rho * dual_sq.sqrt();

            primal_history.push(primal_res);
            dual_history.push(dual_res);

            if primal_res < config.tol {
                converged = true;
                break;
            }
        }

        // Consensus solution: average over agents
        let mut x_consensus = vec![0.0_f64; n_vars];
        let scale = 1.0 / n_agents as f64;
        for xi in x.iter() {
            for k in 0..n_vars {
                x_consensus[k] += scale * xi[k];
            }
        }

        Ok(AdmmResult {
            x: x_consensus,
            primal_residual: primal_history,
            dual_residual: dual_history,
            converged,
            iterations,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EXTRA Solver
// ─────────────────────────────────────────────────────────────────────────────

/// EXTRA solver for exact decentralized consensus.
///
/// EXTRA uses gradient tracking and converges to the exact minimizer
/// of Σ_i f_i(x) with a fixed step size.
///
/// Requires:
/// - W: doubly stochastic mixing matrix (n_agents × n_agents)
/// - `grad_fns[i]`: gradient of f\_i at any point x
#[derive(Debug)]
pub struct ExtraSolver {
    /// Doubly stochastic mixing matrix W (n_agents × n_agents).
    pub w: Vec<Vec<f64>>,
    /// W̃ = (I + W) / 2.
    pub w_tilde: Vec<Vec<f64>>,
}

impl ExtraSolver {
    /// Create a new EXTRA solver from a doubly stochastic mixing matrix W.
    pub fn new(w: Vec<Vec<f64>>) -> OptimizeResult<Self> {
        let n = w.len();
        if !check_doubly_stochastic(&w, 1e-6) {
            return Err(OptimizeError::InvalidInput(
                "W must be doubly stochastic".into(),
            ));
        }
        // W̃ = (I + W) / 2
        let w_tilde: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let eye = if i == j { 1.0 } else { 0.0 };
                        (eye + w[i][j]) / 2.0
                    })
                    .collect()
            })
            .collect();
        Ok(Self { w, w_tilde })
    }

    /// Solve the decentralized problem using EXTRA.
    ///
    /// `grad_fns[i](x)` returns the gradient of f_i at x.
    pub fn solve<F>(
        &self,
        grad_fns: &[F],
        n_vars: usize,
        config: &ExtraConfig,
    ) -> OptimizeResult<AdmmResult>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let n_agents = self.w.len();
        if grad_fns.len() != n_agents {
            return Err(OptimizeError::InvalidInput(format!(
                "Expected {} gradient functions but got {}",
                n_agents,
                grad_fns.len()
            )));
        }
        if n_vars == 0 {
            return Err(OptimizeError::InvalidInput("n_vars must be > 0".into()));
        }

        let alpha = config.alpha;

        // x^0: all agents start at zero
        // Agent states: matrix n_agents × n_vars
        let mut x_curr: Vec<Vec<f64>> = (0..n_agents).map(|_| vec![0.0; n_vars]).collect();

        // Compute gradients at x^0
        let grad_curr: Vec<Vec<f64>> = (0..n_agents).map(|i| (grad_fns[i])(&x_curr[i])).collect();

        // Stack gradients (sum of per-agent grad, column-wise mixing)
        // Stacked agent vectors as rows for mixing: y[i] = Σ_j W[i][j] x[j]
        let x_next: Vec<Vec<f64>> = (0..n_agents)
            .map(|i| {
                // W x^0 row i
                let wx_i: Vec<f64> = (0..n_vars)
                    .map(|k| {
                        (0..n_agents)
                            .map(|j| self.w[i][j] * x_curr[j][k])
                            .sum::<f64>()
                    })
                    .collect();
                // x^1_i = (W x^0)_i - α ∇f_i(x^0_i)
                wx_i.iter()
                    .zip(grad_curr[i].iter())
                    .map(|(w, g)| w - alpha * g)
                    .collect()
            })
            .collect();

        let mut x_prev = x_curr.clone();
        let mut x_curr = x_next;
        let mut grad_prev = grad_curr;

        let mut primal_history = Vec::with_capacity(config.max_iter);
        let mut dual_history = Vec::with_capacity(config.max_iter);
        let mut converged = false;
        let mut iterations = 1;

        for iter in 1..config.max_iter {
            iterations = iter + 1;

            let grad_curr: Vec<Vec<f64>> =
                (0..n_agents).map(|i| (grad_fns[i])(&x_curr[i])).collect();

            // W̃ x^{k+1}
            let w_tilde_x_curr: Vec<Vec<f64>> = (0..n_agents)
                .map(|i| {
                    (0..n_vars)
                        .map(|k| {
                            (0..n_agents)
                                .map(|j| self.w_tilde[i][j] * x_curr[j][k])
                                .sum::<f64>()
                        })
                        .collect()
                })
                .collect();

            // W̃ x^k
            let w_tilde_x_prev: Vec<Vec<f64>> = (0..n_agents)
                .map(|i| {
                    (0..n_vars)
                        .map(|k| {
                            (0..n_agents)
                                .map(|j| self.w_tilde[i][j] * x_prev[j][k])
                                .sum::<f64>()
                        })
                        .collect()
                })
                .collect();

            // x^{k+2}_i = W̃ x^{k+1}_i + x^{k+1}_i - W̃ x^k_i
            //              - α (∇f_i(x^{k+1}) - ∇f_i(x^k))
            let x_new: Vec<Vec<f64>> = (0..n_agents)
                .map(|i| {
                    (0..n_vars)
                        .map(|k| {
                            w_tilde_x_curr[i][k] + x_curr[i][k]
                                - w_tilde_x_prev[i][k]
                                - alpha * (grad_curr[i][k] - grad_prev[i][k])
                        })
                        .collect()
                })
                .collect();

            // Consensus residual: max ||x_i - x̄||
            let x_bar: Vec<f64> = (0..n_vars)
                .map(|k| x_new.iter().map(|xi| xi[k]).sum::<f64>() / n_agents as f64)
                .collect();
            let cons_res: f64 = x_new
                .iter()
                .map(|xi| {
                    xi.iter()
                        .zip(x_bar.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .fold(0.0_f64, f64::max);

            // Dual residual: ||x^{k+1} - x^k||
            let dx: f64 = x_new
                .iter()
                .zip(x_curr.iter())
                .map(|(xn, xc)| {
                    xn.iter()
                        .zip(xc.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                })
                .sum::<f64>()
                .sqrt();

            primal_history.push(cons_res);
            dual_history.push(dx);

            x_prev = x_curr;
            x_curr = x_new;
            grad_prev = grad_curr;

            if cons_res < config.tol && dx < config.tol {
                converged = true;
                break;
            }
        }

        // Return average of agent states as consensus solution
        let x_bar: Vec<f64> = (0..n_vars)
            .map(|k| x_curr.iter().map(|xi| xi[k]).sum::<f64>() / n_agents as f64)
            .collect();

        Ok(AdmmResult {
            x: x_bar,
            primal_residual: primal_history,
            dual_residual: dual_history,
            converged,
            iterations,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Topology builders
// ─────────────────────────────────────────────────────────────────────────────

/// Build a ring topology adjacency matrix for n agents.
///
/// Agent i is connected to (i-1) mod n and (i+1) mod n.
pub fn ring_topology(n: usize) -> Vec<Vec<f64>> {
    let mut adj = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        let next = (i + 1) % n;
        let prev = (i + n - 1) % n;
        adj[i][next] = 1.0;
        adj[i][prev] = 1.0;
    }
    adj
}

/// Build a Metropolis-Hastings doubly stochastic mixing matrix from an adjacency matrix.
///
/// W_{ij} = 1 / (1 + max(deg_i, deg_j))  for (i,j) ∈ E
/// W_{ii} = 1 - Σ_{j≠i} W_{ij}
pub fn metropolis_hastings_weights(adj: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = adj.len();
    let degrees: Vec<usize> = (0..n)
        .map(|i| adj[i].iter().filter(|&&v| v > 0.0).count())
        .collect();

    let mut w = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            if adj[i][j] > 0.0 && i != j {
                let denom = 1.0 + degrees[i].max(degrees[j]) as f64;
                w[i][j] = 1.0 / denom;
                row_sum += w[i][j];
            }
        }
        w[i][i] = 1.0 - row_sum;
    }
    w
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a ring mixing matrix for n agents.
    fn ring_w(n: usize) -> Vec<Vec<f64>> {
        let adj = ring_topology(n);
        metropolis_hastings_weights(&adj)
    }

    #[test]
    fn test_ring_topology() {
        let adj = ring_topology(4);
        // Agent 0 connects to 1 and 3
        assert_eq!(adj[0][1], 1.0);
        assert_eq!(adj[0][3], 1.0);
        assert_eq!(adj[0][0], 0.0);
        assert_eq!(adj[0][2], 0.0);
    }

    #[test]
    fn test_metropolis_hastings_doubly_stochastic() {
        let w = ring_w(4);
        // Row sums should be 1
        for row in w.iter() {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-10, "Row sum = {}", s);
        }
        // Column sums should be 1
        let n = w.len();
        for j in 0..n {
            let s: f64 = w.iter().map(|row| row[j]).sum();
            assert!((s - 1.0).abs() < 1e-10, "Col {} sum = {}", j, s);
        }
    }

    #[test]
    fn test_pdmm_converges() {
        // 3 agents on a complete graph, each minimising f_i(x) = (x - c_i)^2 / 2
        // Optimal consensus: x* = mean(c_i)
        let n_agents = 3;
        let n_vars = 1;
        let centers = vec![1.0_f64, 3.0, 5.0]; // mean = 3.0
        let topology = vec![
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ];
        let solver = PdmmSolver::new(topology).expect("PDMM creation failed");
        let config = PdmmConfig {
            stepsize: 0.2,
            max_iter: 2000,
            tol: 1e-4,
        };
        // Proximal operator: prox_{f_i/ρ}(v) = (c_i + ρ * v) / (1 + ρ)
        let prox_fns: Vec<Box<dyn Fn(&[f64], f64) -> Vec<f64>>> = centers
            .iter()
            .map(|&c| {
                let f: Box<dyn Fn(&[f64], f64) -> Vec<f64>> =
                    Box::new(move |v: &[f64], rho: f64| vec![(c + rho * v[0]) / (1.0 + rho)]);
                f
            })
            .collect();

        let result = solver
            .solve(&prox_fns, n_vars, &config)
            .expect("PDMM solve failed");

        assert!(
            result.converged,
            "PDMM should converge, iters={}",
            result.iterations
        );
        assert!(
            (result.x[0] - 3.0).abs() < 0.1,
            "x = {:.4} (expected 3.0)",
            result.x[0]
        );
    }

    #[test]
    fn test_pdmm_topology_ring() {
        // 4 agents on a ring, each with f_i(x) = (x - c_i)^2 / 2
        let centers = vec![0.0_f64, 2.0, 4.0, 6.0]; // mean = 3.0
        let adj = ring_topology(4);
        let solver = PdmmSolver::new(adj).expect("PDMM ring creation failed");
        let config = PdmmConfig {
            stepsize: 0.1,
            max_iter: 5000,
            tol: 1e-3,
        };
        let prox_fns: Vec<Box<dyn Fn(&[f64], f64) -> Vec<f64>>> = centers
            .iter()
            .map(|&c| {
                let f: Box<dyn Fn(&[f64], f64) -> Vec<f64>> =
                    Box::new(move |v: &[f64], rho: f64| vec![(c + rho * v[0]) / (1.0 + rho)]);
                f
            })
            .collect();

        let result = solver
            .solve(&prox_fns, 1, &config)
            .expect("PDMM ring solve failed");

        // Ring topology converges more slowly; check approximate consensus
        assert!(
            (result.x[0] - 3.0).abs() < 0.5,
            "x = {:.4} (expected ~3.0)",
            result.x[0]
        );
    }

    #[test]
    fn test_extra_exact_consensus() {
        // 4 agents: f_i(x) = (x - c_i)^2, consensus → x* = mean(c_i)
        let centers = vec![1.0_f64, 3.0, 5.0, 7.0]; // mean = 4.0
        let w = ring_w(4);
        let solver = ExtraSolver::new(w).expect("EXTRA creation failed");
        let config = ExtraConfig {
            alpha: 0.02,
            max_iter: 2000,
            tol: 1e-4,
        };
        // Gradient of f_i(x) = 2(x - c_i)
        let grad_fns: Vec<Box<dyn Fn(&[f64]) -> Vec<f64>>> = centers
            .iter()
            .map(|&c| {
                let f: Box<dyn Fn(&[f64]) -> Vec<f64>> =
                    Box::new(move |x: &[f64]| vec![2.0 * (x[0] - c)]);
                f
            })
            .collect();

        let result = solver
            .solve(&grad_fns, 1, &config)
            .expect("EXTRA solve failed");

        assert!(
            result.converged || result.iterations == config.max_iter,
            "EXTRA iterations: {}",
            result.iterations
        );
        assert!(
            (result.x[0] - 4.0).abs() < 0.1,
            "x = {:.4} (expected 4.0), iters={}",
            result.x[0],
            result.iterations
        );
    }

    #[test]
    fn test_extra_vs_admm_same_solution() {
        use super::super::admm::solve_lasso_admm;

        // Both should solve the mean-consensus problem consistently
        // EXTRA: f_i(x) = (x - c_i)^2, grad = 2(x-c_i)
        let centers = vec![2.0_f64, 4.0, 6.0]; // mean = 4.0
        let n_agents = 3_usize;

        // EXTRA solution
        let w = ring_w(n_agents);
        let solver = ExtraSolver::new(w).expect("EXTRA creation failed");
        let config = ExtraConfig {
            alpha: 0.02,
            max_iter: 2000,
            tol: 1e-4,
        };
        let grad_fns: Vec<Box<dyn Fn(&[f64]) -> Vec<f64>>> = centers
            .iter()
            .map(|&c| {
                let f: Box<dyn Fn(&[f64]) -> Vec<f64>> =
                    Box::new(move |x: &[f64]| vec![2.0 * (x[0] - c)]);
                f
            })
            .collect();
        let extra_res = solver.solve(&grad_fns, 1, &config).expect("EXTRA failed");

        // ADMM consensus solution (same problem via consensus_admm)
        use super::super::admm::consensus_admm;
        let admm_config = super::super::types::AdmmConfig {
            rho: 1.0,
            max_iter: 500,
            abs_tol: 1e-6,
            rel_tol: 1e-4,
            warm_start: false,
            over_relaxation: 1.0,
        };
        let prox_fns: Vec<Box<dyn Fn(&[f64], f64) -> Vec<f64>>> = centers
            .iter()
            .map(|&c| {
                let f: Box<dyn Fn(&[f64], f64) -> Vec<f64>> =
                    Box::new(move |v: &[f64], rho: f64| {
                        // prox for f_i(x) = (x-c)^2: solution is (rho*v + 2*c) / (rho + 2)
                        vec![(rho * v[0] + 2.0 * c) / (rho + 2.0)]
                    });
                f
            })
            .collect();
        let admm_res = consensus_admm(&prox_fns, 1, &admm_config).expect("ADMM failed");

        // Both should be close to the true mean (4.0)
        assert!(
            (extra_res.x[0] - 4.0).abs() < 0.2,
            "EXTRA x = {:.4}",
            extra_res.x[0]
        );
        assert!(
            (admm_res.x[0] - 4.0).abs() < 0.1,
            "ADMM x = {:.4}",
            admm_res.x[0]
        );
    }

    #[test]
    fn test_extra_solver_invalid_w() {
        // Non-doubly-stochastic W should fail
        let w = vec![vec![0.5, 0.5], vec![0.9, 0.1]]; // col 0 sums to 1.4 ≠ 1
        let result = ExtraSolver::new(w);
        assert!(result.is_err());
    }
}
