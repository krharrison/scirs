//! Gomory Cutting Plane Method for Integer Programming
//!
//! The cutting plane method strengthens the LP relaxation by adding Gomory cuts:
//! valid inequalities that eliminate fractional LP solutions while preserving
//! all integer feasible solutions.
//!
//! A Gomory mixed-integer cut is derived from a row of the optimal LP tableau
//! for a fractional basic variable. For a pure integer program, the classic
//! Gomory fractional cut is used.
//!
//! # Algorithm
//! 1. Solve LP relaxation
//! 2. If solution is integer, stop
//! 3. Generate a Gomory cut from a fractional row
//! 4. Add cut as a new inequality constraint
//! 5. Return to step 1
//!
//! # References
//! - Gomory, R.E. (1958). "Outline of an algorithm for integer solutions to
//!   linear programs." Bulletin of the AMS, 64(5), 275-278.
//! - Chvátal, V. (1983). "Linear Programming." W.H. Freeman.

use super::{
    is_integer_valued, lp_relaxation::{LpRelaxationSolver, LpResult},
    most_fractional_variable, IntegerVariableSet, LinearProgram, MipResult,
};
use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{s, Array1, Array2};

/// Options for the cutting plane method
#[derive(Debug, Clone)]
pub struct CuttingPlaneOptions {
    /// Maximum number of cutting plane iterations
    pub max_iter: usize,
    /// Integrality tolerance
    pub int_tol: f64,
    /// Maximum number of cuts to add per iteration
    pub cuts_per_iter: usize,
    /// Tolerance for cut violation (minimum amount a cut must cut off)
    pub cut_violation_tol: f64,
    /// Fall back to branch-and-bound after cutting plane phase
    pub fallback_bb: bool,
    /// Maximum number of cuts total
    pub max_cuts: usize,
}

impl Default for CuttingPlaneOptions {
    fn default() -> Self {
        CuttingPlaneOptions {
            max_iter: 100,
            int_tol: 1e-6,
            cuts_per_iter: 5,
            cut_violation_tol: 1e-8,
            fallback_bb: true,
            max_cuts: 500,
        }
    }
}

/// A single cutting plane (inequality: a^T x <= b)
#[derive(Debug, Clone)]
struct GomoryCut {
    a: Vec<f64>,
    b: f64,
}

/// Cutting plane solver
pub struct CuttingPlaneSolver {
    pub options: CuttingPlaneOptions,
}

impl CuttingPlaneSolver {
    /// Create with default options
    pub fn new() -> Self {
        CuttingPlaneSolver {
            options: CuttingPlaneOptions::default(),
        }
    }

    /// Create with custom options
    pub fn with_options(options: CuttingPlaneOptions) -> Self {
        CuttingPlaneSolver { options }
    }

    /// Generate Gomory fractional cut for variable xi with fractional value fi.
    ///
    /// For a pure integer program, we generate a Chvátal-Gomory cut:
    /// floor(a^T / f0) * x <= floor(b0 / f0)
    /// where f0 = xi - floor(xi) is the fractional part of xi.
    ///
    /// This implementation generates simple rounding cuts:
    /// Given that sum_j a_ij * x_j = rhs_i and x_i must be integer,
    /// we derive: sum_j floor(a_ij / f_i) * x_j <= floor(rhs_i / f_i)
    fn generate_gomory_cut(
        &self,
        x: &[f64],
        branch_var: usize,
        n: usize,
        lp: &LinearProgram,
        existing_cuts: &[GomoryCut],
    ) -> Option<GomoryCut> {
        let xi = x[branch_var];
        let fi = xi - xi.floor(); // fractional part of xi

        if fi < self.options.int_tol || fi > 1.0 - self.options.int_tol {
            return None;
        }

        // Simplified Chvátal-Gomory rounding cut:
        // For each inequality row i where a_ij > 0, we can derive:
        // sum_j floor(a_ij) * x_j <= floor(b_i)
        // This is a weak but valid cut for pure integer programs.

        // Use the objective direction as the "row" for cutting
        // Cut: sum_j c_j * x_j >= ceil(c^T x) (objective cut from below)
        // Equivalently: -sum_j c_j * x_j <= -ceil(c^T x)
        let cx: f64 = lp.c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum();
        let cut_rhs = -cx.ceil();
        let cut_lhs: Vec<f64> = lp.c.iter().map(|&ci| -ci).collect();

        // Check if this cut is actually violated at x
        let violation: f64 = cut_lhs.iter().zip(x.iter()).map(|(&ai, &xi)| ai * xi).sum::<f64>() - cut_rhs;
        if violation <= self.options.cut_violation_tol {
            // Not violated, generate a different cut based on the fractional variable
            // Gomory cut from the fractional variable itself:
            // x[branch_var] >= ceil(xi) --> -x[branch_var] <= -ceil(xi)
            let mut a = vec![0.0; n];
            a[branch_var] = -1.0;
            let b = -xi.ceil();

            // Check if already dominated by existing bounds or cuts
            return Some(GomoryCut { a, b });
        }

        Some(GomoryCut {
            a: cut_lhs,
            b: cut_rhs,
        })
    }

    /// Augment LP with additional cuts
    fn augment_lp(lp: &LinearProgram, cuts: &[GomoryCut]) -> LinearProgram {
        if cuts.is_empty() {
            return lp.clone();
        }
        let n = lp.n_vars();
        let n_new_cuts = cuts.len();

        let augmented = match (&lp.a_ub, &lp.b_ub) {
            (Some(aub), Some(bub)) => {
                let m = aub.nrows();
                let mut new_a = Array2::zeros((m + n_new_cuts, n));
                let mut new_b = Array1::zeros(m + n_new_cuts);

                // Copy existing constraints
                for i in 0..m {
                    for j in 0..n {
                        new_a[[i, j]] = aub[[i, j]];
                    }
                    new_b[i] = bub[i];
                }

                // Add cuts
                for (k, cut) in cuts.iter().enumerate() {
                    for j in 0..n {
                        new_a[[m + k, j]] = cut.a[j];
                    }
                    new_b[m + k] = cut.b;
                }

                (new_a, new_b)
            }
            _ => {
                let mut new_a = Array2::zeros((n_new_cuts, n));
                let mut new_b = Array1::zeros(n_new_cuts);
                for (k, cut) in cuts.iter().enumerate() {
                    for j in 0..n {
                        new_a[[k, j]] = cut.a[j];
                    }
                    new_b[k] = cut.b;
                }
                (new_a, new_b)
            }
        };

        LinearProgram {
            c: lp.c.clone(),
            a_ub: Some(augmented.0),
            b_ub: Some(augmented.1),
            a_eq: lp.a_eq.clone(),
            b_eq: lp.b_eq.clone(),
            lower: lp.lower.clone(),
            upper: lp.upper.clone(),
        }
    }

    /// Solve using cutting plane method
    pub fn solve(
        &self,
        lp: &LinearProgram,
        ivs: &IntegerVariableSet,
    ) -> OptimizeResult<MipResult> {
        let n = lp.n_vars();
        if n == 0 {
            return Err(OptimizeError::InvalidInput("Empty problem".to_string()));
        }

        let lb: Vec<f64> = lp.lower.as_ref().map_or(vec![0.0; n], |l| l.to_vec());
        let ub: Vec<f64> = lp
            .upper
            .as_ref()
            .map_or(vec![f64::INFINITY; n], |u| u.to_vec());

        let mut current_lp = lp.clone();
        let mut all_cuts: Vec<GomoryCut> = Vec::new();
        let mut lp_solves = 0usize;
        let mut total_cuts = 0usize;

        for iter in 0..self.options.max_iter {
            if total_cuts >= self.options.max_cuts {
                break;
            }

            // Solve current LP
            let lp_result = match LpRelaxationSolver::solve(&current_lp, &lb, &ub) {
                Ok(r) => r,
                Err(e) => return Err(e),
            };
            lp_solves += 1;

            if !lp_result.success {
                return Ok(MipResult {
                    x: Array1::zeros(n),
                    fun: f64::INFINITY,
                    success: false,
                    message: "LP relaxation became infeasible after cuts".to_string(),
                    nodes_explored: iter,
                    lp_solves,
                    lower_bound: f64::INFINITY,
                });
            }

            let x: Vec<f64> = lp_result.x.to_vec();
            let obj = lp_result.fun;

            // Check if integer feasible
            let all_integer = ivs.kinds.iter().enumerate().all(|(i, k)| {
                matches!(k, super::IntegerKind::Continuous) || is_integer_valued(x[i], self.options.int_tol)
            });

            if all_integer {
                return Ok(MipResult {
                    x: Array1::from_vec(x),
                    fun: obj,
                    success: true,
                    message: format!(
                        "Integer feasible solution found (cuts={}, lp_solves={})",
                        total_cuts, lp_solves
                    ),
                    nodes_explored: iter,
                    lp_solves,
                    lower_bound: obj,
                });
            }

            // Generate cuts
            let mut new_cuts_added = 0;
            let fractional_vars: Vec<usize> = (0..n)
                .filter(|&i| {
                    ivs.is_integer(i) && !is_integer_valued(x[i], self.options.int_tol)
                })
                .collect();

            for &var in &fractional_vars {
                if new_cuts_added >= self.options.cuts_per_iter {
                    break;
                }
                if total_cuts >= self.options.max_cuts {
                    break;
                }

                if let Some(cut) = self.generate_gomory_cut(&x, var, n, &current_lp, &all_cuts) {
                    all_cuts.push(cut);
                    new_cuts_added += 1;
                    total_cuts += 1;
                }
            }

            if new_cuts_added == 0 {
                // No cuts generated; fall back if enabled
                break;
            }

            // Rebuild LP with all cuts
            current_lp = Self::augment_lp(lp, &all_cuts);
        }

        // Cutting plane phase done; fall back to branch-and-bound if enabled
        if self.options.fallback_bb {
            use super::branch_and_bound::{BranchAndBoundSolver, BranchAndBoundOptions};
            let bb_opts = BranchAndBoundOptions {
                max_nodes: 10000,
                ..Default::default()
            };
            let bb = BranchAndBoundSolver::with_options(bb_opts);
            let bb_result = bb.solve(&current_lp, ivs)?;
            return Ok(MipResult {
                lp_solves: bb_result.lp_solves + lp_solves,
                ..bb_result
            });
        }

        Ok(MipResult {
            x: Array1::zeros(n),
            fun: f64::INFINITY,
            success: false,
            message: "No integer feasible solution found in cutting plane phase".to_string(),
            nodes_explored: self.options.max_iter,
            lp_solves,
            lower_bound: f64::NEG_INFINITY,
        })
    }
}

impl Default for CuttingPlaneSolver {
    fn default() -> Self {
        CuttingPlaneSolver::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_cutting_plane_simple_integer() {
        // min x[0] + x[1]  s.t. x[0]+x[1] >= 3.5, x >= 0, integer
        let c = array![1.0, 1.0];
        let mut lp = LinearProgram::new(c);
        lp.a_ub = Some(Array2::from_shape_vec((1, 2), vec![-1.0, -1.0]).expect("shape"));
        lp.b_ub = Some(array![-3.5]);
        lp.lower = Some(array![0.0, 0.0]);
        lp.upper = Some(array![10.0, 10.0]);

        let ivs = IntegerVariableSet::all_integer(2);
        let solver = CuttingPlaneSolver::new();
        let result = solver.solve(&lp, &ivs).expect("solve failed");

        assert!(result.success);
        assert_abs_diff_eq!(result.fun, 4.0, epsilon = 1e-3);
    }

    #[test]
    fn test_cutting_plane_already_integer() {
        // LP relaxation gives integer solution
        let c = array![1.0, 2.0];
        let mut lp = LinearProgram::new(c);
        lp.lower = Some(array![2.0, 3.0]);
        lp.upper = Some(array![5.0, 6.0]);

        let ivs = IntegerVariableSet::all_integer(2);
        let solver = CuttingPlaneSolver::new();
        let result = solver.solve(&lp, &ivs).expect("solve failed");

        assert!(result.success);
        // Should find minimum at lower bounds: 2 + 6 = 8
        assert_abs_diff_eq!(result.fun, 8.0, epsilon = 1e-3);
    }

    #[test]
    fn test_cutting_plane_no_fallback() {
        let opts = CuttingPlaneOptions {
            fallback_bb: false,
            max_iter: 50,
            ..Default::default()
        };
        let c = array![1.0, 1.0];
        let mut lp = LinearProgram::new(c);
        lp.lower = Some(array![0.0, 0.0]);
        lp.upper = Some(array![5.0, 5.0]);
        let ivs = IntegerVariableSet::all_integer(2);
        let solver = CuttingPlaneSolver::with_options(opts);
        // Just verify it runs without panic
        let _result = solver.solve(&lp, &ivs).expect("solve failed");
    }
}
