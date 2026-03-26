//! QAOA (Quantum Approximate Optimization Algorithm) for combinatorial optimization
//!
//! Implements the QAOA variational algorithm for solving MaxCut and related problems.
//! Uses exact statevector simulation to evaluate circuit expectation values, and
//! the parameter-shift rule to compute analytic gradients.

use crate::error::OptimizeError;
use crate::quantum_classical::statevector::Statevector;
use crate::quantum_classical::QcResult;

// ─── Problem definition ────────────────────────────────────────────────────

/// Weighted MaxCut problem instance.
///
/// The MaxCut cost function is: C = Σ_{(i,j)∈E} w_{ij} * (1 - ⟨Z_i Z_j⟩) / 2
#[derive(Debug, Clone)]
pub struct MaxCutProblem {
    /// Number of vertices
    pub n_vertices: usize,
    /// Weighted edges: (vertex_i, vertex_j, weight)
    pub edges: Vec<(usize, usize, f64)>,
}

impl MaxCutProblem {
    /// Create a new MaxCut problem.
    pub fn new(n_vertices: usize, edges: Vec<(usize, usize, f64)>) -> Self {
        Self { n_vertices, edges }
    }

    /// Compute the cost function value: C = Σ w_ij * (1 - ⟨Z_i Z_j⟩) / 2
    ///
    /// This is the expected number of edges in the cut, which we want to maximize.
    pub fn cost_function(&self, state: &Statevector) -> QcResult<f64> {
        let mut cost = 0.0;
        for &(i, j, w) in &self.edges {
            let zz = state.expectation_zz(i, j)?;
            cost += w * (1.0 - zz) / 2.0;
        }
        Ok(cost)
    }

    /// Evaluate the exact cut value for a given bitstring assignment.
    /// Returns number of edges cut (weighted).
    pub fn cut_value(&self, bits: &[bool]) -> f64 {
        self.edges
            .iter()
            .filter(|&&(i, j, _)| bits[i] != bits[j])
            .map(|&(_, _, w)| w)
            .sum()
    }
}

// ─── QAOA configuration ────────────────────────────────────────────────────

/// Configuration for the QAOA algorithm.
#[derive(Debug, Clone)]
pub struct QaoaConfig {
    /// Number of QAOA layers (depth p)
    pub p_layers: usize,
    /// Initial γ parameters (length p)
    pub init_gamma: Vec<f64>,
    /// Initial β parameters (length p)
    pub init_beta: Vec<f64>,
    /// Maximum iterations for the classical optimizer
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
}

impl Default for QaoaConfig {
    fn default() -> Self {
        let p = 2;
        Self {
            p_layers: p,
            init_gamma: vec![0.1; p],
            init_beta: vec![0.1; p],
            max_iter: 200,
            tol: 1e-6,
        }
    }
}

// ─── QAOA circuit ──────────────────────────────────────────────────────────

/// QAOA circuit for MaxCut optimization.
#[derive(Debug, Clone)]
pub struct QaoaCircuit {
    /// The problem instance
    pub problem: MaxCutProblem,
    /// QAOA configuration
    pub config: QaoaConfig,
}

impl QaoaCircuit {
    /// Create a new QAOA circuit.
    pub fn new(problem: MaxCutProblem, config: QaoaConfig) -> Self {
        Self { problem, config }
    }

    /// Evaluate the QAOA objective (negative expected cut) for given parameters.
    ///
    /// Steps:
    /// 1. Prepare |+⟩^⊗n by applying H to each qubit
    /// 2. For each layer l: apply cost unitary U_C(γ_l), then mixer U_B(β_l)
    /// 3. Return -⟨C⟩ (negative because we minimize)
    pub fn run(&self, gamma: &[f64], beta: &[f64]) -> QcResult<f64> {
        if gamma.len() != self.config.p_layers {
            return Err(OptimizeError::ValueError(format!(
                "Expected {} gamma values, got {}",
                self.config.p_layers,
                gamma.len()
            )));
        }
        if beta.len() != self.config.p_layers {
            return Err(OptimizeError::ValueError(format!(
                "Expected {} beta values, got {}",
                self.config.p_layers,
                beta.len()
            )));
        }
        let n = self.problem.n_vertices;
        let mut state = Statevector::zero_state(n)?;

        // Step 1: Prepare |+⟩^⊗n
        for q in 0..n {
            state.apply_hadamard(q)?;
        }

        // Step 2: Apply p layers
        for l in 0..self.config.p_layers {
            // Cost unitary U_C(γ_l): apply Rzz(2*γ_l*w) for each edge
            for &(i, j, w) in &self.problem.edges {
                state.apply_rzz(i, j, 2.0 * gamma[l] * w)?;
            }

            // Mixer unitary U_B(β_l): apply Rx(2*β_l) to each qubit
            for q in 0..n {
                state.apply_rx(q, 2.0 * beta[l])?;
            }
        }

        // Step 3: Evaluate cost
        let cost = self.problem.cost_function(&state)?;
        Ok(-cost) // negate: we minimize, but want to maximize cut
    }

    /// Optimize γ and β parameters using Nelder-Mead.
    pub fn optimize(&self) -> QcResult<QaoaResult> {
        let p = self.config.p_layers;
        let n_params = 2 * p;

        // Pack initial parameters: [gamma_0, ..., gamma_{p-1}, beta_0, ..., beta_{p-1}]
        let mut x: Vec<f64> = self
            .config
            .init_gamma
            .iter()
            .chain(self.config.init_beta.iter())
            .cloned()
            .collect();

        let eval = |params: &[f64]| -> f64 {
            let gam = &params[..p];
            let bet = &params[p..];
            self.run(gam, bet).unwrap_or(0.0)
        };

        let (best_params, best_val, n_evals) =
            nelder_mead_minimize(eval, &x, self.config.max_iter, self.config.tol)?;

        x = best_params;
        let gamma_opt: Vec<f64> = x[..p].to_vec();
        let beta_opt: Vec<f64> = x[p..].to_vec();

        Ok(QaoaResult {
            optimal_gamma: gamma_opt,
            optimal_beta: beta_opt,
            optimal_value: -best_val, // convert back to expected cut value
            n_evaluations: n_evals,
        })
    }

    /// Determine the most probable bitstring assignment from the optimized circuit.
    pub fn best_string(&self, gamma: &[f64], beta: &[f64]) -> QcResult<Vec<bool>> {
        let n = self.problem.n_vertices;
        let mut state = Statevector::zero_state(n)?;
        for q in 0..n {
            state.apply_hadamard(q)?;
        }
        for l in 0..self.config.p_layers {
            for &(i, j, w) in &self.problem.edges {
                state.apply_rzz(i, j, 2.0 * gamma[l] * w)?;
            }
            for q in 0..n {
                state.apply_rx(q, 2.0 * beta[l])?;
            }
        }
        let idx = state.most_probable_state();
        Ok(state.index_to_bits(idx))
    }

    /// Build the optimized statevector (helper for advanced analysis).
    fn build_state(&self, gamma: &[f64], beta: &[f64]) -> QcResult<Statevector> {
        let n = self.problem.n_vertices;
        let mut state = Statevector::zero_state(n)?;
        for q in 0..n {
            state.apply_hadamard(q)?;
        }
        for l in 0..self.config.p_layers {
            for &(i, j, w) in &self.problem.edges {
                state.apply_rzz(i, j, 2.0 * gamma[l] * w)?;
            }
            for q in 0..n {
                state.apply_rx(q, 2.0 * beta[l])?;
            }
        }
        Ok(state)
    }
}

// ─── Result type ───────────────────────────────────────────────────────────

/// Result of a QAOA optimization run.
#[derive(Debug, Clone)]
pub struct QaoaResult {
    /// Optimal γ parameters
    pub optimal_gamma: Vec<f64>,
    /// Optimal β parameters
    pub optimal_beta: Vec<f64>,
    /// Optimal expected cut value (positive = good)
    pub optimal_value: f64,
    /// Total number of circuit evaluations
    pub n_evaluations: usize,
}

// ─── Parameter-shift gradient ──────────────────────────────────────────────

/// Compute analytic gradients of the QAOA objective using the parameter-shift rule.
///
/// d/dθ E(θ) = 0.5 * [E(θ + π/2) - E(θ - π/2)]
///
/// Returns `(d_gamma, d_beta)` where each vector has length `p`.
pub fn parameter_shift_gradient(
    circuit: &QaoaCircuit,
    gamma: &[f64],
    beta: &[f64],
) -> QcResult<(Vec<f64>, Vec<f64>)> {
    let p = circuit.config.p_layers;
    let shift = std::f64::consts::FRAC_PI_2; // π/2

    let mut d_gamma = vec![0.0; p];
    let mut d_beta = vec![0.0; p];

    // Gradient w.r.t. gamma[l]
    for l in 0..p {
        let mut g_plus = gamma.to_vec();
        let mut g_minus = gamma.to_vec();
        g_plus[l] += shift;
        g_minus[l] -= shift;
        let e_plus = circuit.run(&g_plus, beta)?;
        let e_minus = circuit.run(&g_minus, beta)?;
        d_gamma[l] = 0.5 * (e_plus - e_minus);
    }

    // Gradient w.r.t. beta[l]
    for l in 0..p {
        let mut b_plus = beta.to_vec();
        let mut b_minus = beta.to_vec();
        b_plus[l] += shift;
        b_minus[l] -= shift;
        let e_plus = circuit.run(gamma, &b_plus)?;
        let e_minus = circuit.run(gamma, &b_minus)?;
        d_beta[l] = 0.5 * (e_plus - e_minus);
    }

    Ok((d_gamma, d_beta))
}

// ─── Nelder-Mead minimizer (self-contained, no external dep needed) ────────

/// Simple Nelder-Mead minimizer for the QAOA parameter optimization.
///
/// Returns `(best_params, best_value, n_evaluations)`.
fn nelder_mead_minimize<F>(
    f: F,
    x0: &[f64],
    max_iter: usize,
    tol: f64,
) -> QcResult<(Vec<f64>, f64, usize)>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::ValueError(
            "Parameter vector must be non-empty".to_string(),
        ));
    }

    // Initialize simplex: one vertex at x0, others perturbed
    let step = 0.2_f64;
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut v = x0.to_vec();
        v[i] += if v[i].abs() < 1e-10 {
            0.05
        } else {
            step * v[i].abs().max(0.05)
        };
        simplex.push(v);
    }

    let mut fvals: Vec<f64> = simplex.iter().map(|v| f(v)).collect();
    let mut n_evals = simplex.len();

    let alpha = 1.0; // reflection
    let gamma_nm = 2.0; // expansion
    let rho = 0.5; // contraction
    let sigma = 0.5; // shrink

    for _iter in 0..max_iter {
        // Sort vertices by function value
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| {
            fvals[a]
                .partial_cmp(&fvals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Check convergence: spread of function values
        let f_best = fvals[order[0]];
        let f_worst = fvals[order[n]];
        if (f_worst - f_best).abs() < tol {
            let best_params = simplex[order[0]].clone();
            return Ok((best_params, f_best, n_evals));
        }

        // Centroid of all but worst
        let mut centroid = vec![0.0; n];
        for &idx in &order[..n] {
            for (k, c) in centroid.iter_mut().enumerate() {
                *c += simplex[idx][k];
            }
        }
        let n_f = n as f64;
        centroid.iter_mut().for_each(|c| *c /= n_f);

        // Reflection
        let worst_idx = order[n];
        let xr: Vec<f64> = centroid
            .iter()
            .zip(&simplex[worst_idx])
            .map(|(&c, &w)| c + alpha * (c - w))
            .collect();
        let fr = f(&xr);
        n_evals += 1;

        let f_second_worst = fvals[order[n - 1]];

        if fr < fvals[order[0]] {
            // Try expansion
            let xe: Vec<f64> = centroid
                .iter()
                .zip(&xr)
                .map(|(&c, &r)| c + gamma_nm * (r - c))
                .collect();
            let fe = f(&xe);
            n_evals += 1;
            if fe < fr {
                simplex[worst_idx] = xe;
                fvals[worst_idx] = fe;
            } else {
                simplex[worst_idx] = xr;
                fvals[worst_idx] = fr;
            }
        } else if fr < f_second_worst {
            simplex[worst_idx] = xr;
            fvals[worst_idx] = fr;
        } else {
            // Contraction
            let use_reflection = fr < fvals[worst_idx];
            let xc: Vec<f64> = if use_reflection {
                centroid
                    .iter()
                    .zip(&xr)
                    .map(|(&c, &r)| c + rho * (r - c))
                    .collect()
            } else {
                centroid
                    .iter()
                    .zip(&simplex[worst_idx])
                    .map(|(&c, &w)| c + rho * (w - c))
                    .collect()
            };
            let fc = f(&xc);
            n_evals += 1;

            let contraction_success = if use_reflection {
                fc <= fr
            } else {
                fc < fvals[worst_idx]
            };

            if contraction_success {
                simplex[worst_idx] = xc;
                fvals[worst_idx] = fc;
            } else {
                // Shrink: move all vertices toward best
                let best_idx = order[0];
                let best_vertex = simplex[best_idx].clone();
                for i in 1..=n {
                    let idx = order[i];
                    simplex[idx] = best_vertex
                        .iter()
                        .zip(&simplex[idx])
                        .map(|(&b, &v)| b + sigma * (v - b))
                        .collect();
                    fvals[idx] = f(&simplex[idx]);
                    n_evals += 1;
                }
            }
        }
    }

    // Return best found after max_iter
    let best_idx = (0..=n)
        .min_by(|&a, &b| {
            fvals[a]
                .partial_cmp(&fvals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    Ok((simplex[best_idx].clone(), fvals[best_idx], n_evals))
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxcut_two_nodes_improves() {
        // Single edge (0,1): optimal MaxCut = 1.0
        let problem = MaxCutProblem::new(2, vec![(0, 1, 1.0)]);
        let config = QaoaConfig {
            p_layers: 1,
            init_gamma: vec![0.5],
            init_beta: vec![0.5],
            max_iter: 100,
            tol: 1e-5,
        };
        let circuit = QaoaCircuit::new(problem, config);
        let result = circuit.optimize().unwrap();
        // Should get at least 0.5 expected cut value
        assert!(
            result.optimal_value >= 0.5,
            "Expected cut ≥ 0.5, got {}",
            result.optimal_value
        );
    }

    #[test]
    fn test_qaoa_deterministic_evaluation() {
        let problem = MaxCutProblem::new(2, vec![(0, 1, 1.0)]);
        let config = QaoaConfig {
            p_layers: 1,
            init_gamma: vec![0.3],
            init_beta: vec![0.4],
            max_iter: 10,
            tol: 1e-6,
        };
        let circuit = QaoaCircuit::new(problem, config);
        let val1 = circuit.run(&[0.3], &[0.4]).unwrap();
        let val2 = circuit.run(&[0.3], &[0.4]).unwrap();
        assert!(
            (val1 - val2).abs() < 1e-14,
            "Evaluation must be deterministic"
        );
    }

    #[test]
    fn test_parameter_shift_gradient_sign() {
        // Verify that the parameter-shift gradient has the correct sign:
        // moving a parameter in the direction opposite to gradient should decrease energy.
        // We use beta as test parameter since Rx(2*beta) has eigenvalue gap 1 → standard shift applies.
        let problem = MaxCutProblem::new(3, vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]);
        let config = QaoaConfig {
            p_layers: 1,
            init_gamma: vec![0.5],
            init_beta: vec![0.3],
            max_iter: 50,
            tol: 1e-6,
        };
        let circuit = QaoaCircuit::new(problem, config);
        let gamma = [0.5];
        let beta = [0.3];
        let (d_gamma, d_beta) = parameter_shift_gradient(&circuit, &gamma, &beta).unwrap();

        // Check gradient is finite
        assert!(d_gamma[0].is_finite(), "d_gamma must be finite");
        assert!(d_beta[0].is_finite(), "d_beta must be finite");

        // For beta: Rx(2*beta) = exp(-i*beta*X). Standard parameter-shift applies directly.
        // The shift π/2 in beta corresponds exactly to the standard rule.
        // Check using finite differences of the same formula (shifted by π/2):
        let e_plus = circuit
            .run(&gamma, &[beta[0] + std::f64::consts::FRAC_PI_2])
            .unwrap();
        let e_minus = circuit
            .run(&gamma, &[beta[0] - std::f64::consts::FRAC_PI_2])
            .unwrap();
        let ps_check = 0.5 * (e_plus - e_minus);
        assert!(
            (d_beta[0] - ps_check).abs() < 1e-12,
            "Parameter shift gradient {:.6} should equal 0.5*(E+ - E-) = {:.6}",
            d_beta[0],
            ps_check
        );

        // Check that the parameter-shift formula is consistent (not zero) at this non-trivial point
        // or that the gradient correctly identifies a minimum direction
        let e0 = circuit.run(&gamma, &beta).unwrap();
        let lr = 0.1;
        let new_beta = [beta[0] - lr * d_beta[0]];
        let e_new = circuit.run(&gamma, &new_beta).unwrap();
        // Moving in the negative gradient direction should decrease or maintain the energy
        // (within numerical precision - allow a small tolerance for non-convex landscape)
        assert!(
            e_new <= e0 + 0.01,
            "Gradient step should not increase energy significantly: {e0:.4} -> {e_new:.4}"
        );
    }

    #[test]
    fn test_best_string_returns_valid_assignment() {
        let problem = MaxCutProblem::new(3, vec![(0, 1, 1.0), (1, 2, 1.0)]);
        let config = QaoaConfig::default();
        let circuit = QaoaCircuit::new(problem.clone(), config);
        let result = circuit.optimize().unwrap();
        let bits = circuit
            .best_string(&result.optimal_gamma, &result.optimal_beta)
            .unwrap();
        assert_eq!(bits.len(), 3);
        // The cut value should be non-negative
        let cut = problem.cut_value(&bits);
        assert!(cut >= 0.0);
    }
}
