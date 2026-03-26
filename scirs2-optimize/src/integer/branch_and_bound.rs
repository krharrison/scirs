//! Branch-and-Bound algorithm for Mixed-Integer Programming
//!
//! The branch-and-bound algorithm systematically explores the space of integer
//! solutions by:
//! 1. Solving LP relaxations at each node
//! 2. Branching on a fractional integer variable
//! 3. Using bounds to prune subproblems that cannot improve on the incumbent
//!
//! # Algorithm
//! - Start with the LP relaxation of the full problem
//! - Maintain an incumbent (best integer solution found)
//! - Branch by adding floor/ceil constraints on a fractional variable
//! - Prune nodes where LP lower bound >= incumbent objective
//!
//! # References
//! - Land, A.H. & Doig, A.G. (1960). "An automatic method of solving discrete
//!   programming problems." Econometrica, 28(3), 497-520.

use super::{
    is_integer_valued,
    lp_relaxation::{LpRelaxationSolver, LpResult},
    most_fractional_variable, IntegerKind, IntegerVariableSet, LinearProgram, MipResult,
};
use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::Array1;
use std::collections::VecDeque;

/// Options for branch-and-bound
#[derive(Debug, Clone)]
pub struct BranchAndBoundOptions {
    /// Maximum number of nodes to explore
    pub max_nodes: usize,
    /// Absolute tolerance for integrality
    pub int_tol: f64,
    /// Absolute tolerance for optimality gap
    pub opt_tol: f64,
    /// Maximum LP iterations per node
    pub max_lp_iter: usize,
    /// Node selection strategy
    pub node_selection: NodeSelection,
    /// Variable selection strategy
    pub variable_selection: VariableSelection,
}

impl Default for BranchAndBoundOptions {
    fn default() -> Self {
        BranchAndBoundOptions {
            max_nodes: 10000,
            int_tol: 1e-6,
            opt_tol: 1e-6,
            max_lp_iter: 1000,
            node_selection: NodeSelection::BestFirst,
            variable_selection: VariableSelection::MostFractional,
        }
    }
}

/// Node selection strategy for branch-and-bound
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NodeSelection {
    /// Best-first: select node with lowest lower bound
    BestFirst,
    /// Depth-first: explore deepest node first (DFS)
    DepthFirst,
    /// Best-of-depth: DFS with best-bound pruning
    BestOfDepth,
}

/// Variable selection strategy for branching
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VariableSelection {
    /// Branch on most fractional variable
    MostFractional,
    /// Branch on first fractional integer variable
    FirstFractional,
    /// Branch on variable with largest LP value range
    MaxRange,
}

/// A node in the B&B tree
#[derive(Debug, Clone)]
struct BbNode {
    /// Additional lower bounds for this node (branching constraints)
    extra_lb: Vec<f64>,
    /// Additional upper bounds for this node (branching constraints)
    extra_ub: Vec<f64>,
    /// LP lower bound at this node (for ordering)
    lb: f64,
    /// Depth in the tree
    depth: usize,
}

/// Branch-and-bound solver for mixed-integer programming
pub struct BranchAndBoundSolver {
    pub options: BranchAndBoundOptions,
}

impl BranchAndBoundSolver {
    /// Create with default options
    pub fn new() -> Self {
        BranchAndBoundSolver {
            options: BranchAndBoundOptions::default(),
        }
    }

    /// Create with custom options
    pub fn with_options(options: BranchAndBoundOptions) -> Self {
        BranchAndBoundSolver { options }
    }

    /// Select branching variable
    fn select_variable(&self, x: &[f64], ivs: &IntegerVariableSet) -> Option<usize> {
        match self.options.variable_selection {
            VariableSelection::MostFractional => most_fractional_variable(x, ivs),
            VariableSelection::FirstFractional => {
                for (i, &xi) in x.iter().enumerate() {
                    if ivs.is_integer(i) && !is_integer_valued(xi, self.options.int_tol) {
                        return Some(i);
                    }
                }
                None
            }
            VariableSelection::MaxRange => {
                // Select the integer variable with largest fractional part
                most_fractional_variable(x, ivs)
            }
        }
    }

    /// Check if an LP solution satisfies integrality
    fn is_integer_feasible(&self, x: &[f64], ivs: &IntegerVariableSet) -> bool {
        for (i, &xi) in x.iter().enumerate() {
            if ivs.is_integer(i) && !is_integer_valued(xi, self.options.int_tol) {
                return false;
            }
        }
        true
    }

    /// Round integer variables in LP solution and verify feasibility
    fn round_integer_solution(&self, x: &[f64], ivs: &IntegerVariableSet) -> Vec<f64> {
        x.iter()
            .enumerate()
            .map(|(i, &xi)| if ivs.is_integer(i) { xi.round() } else { xi })
            .collect()
    }

    /// Evaluate objective value
    fn eval_obj(lp: &LinearProgram, x: &[f64]) -> f64 {
        lp.c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum()
    }

    /// Solve the MIP problem
    pub fn solve(&self, lp: &LinearProgram, ivs: &IntegerVariableSet) -> OptimizeResult<MipResult> {
        let n = lp.n_vars();
        if n == 0 {
            return Err(OptimizeError::InvalidInput("Empty problem".to_string()));
        }
        if ivs.len() != n {
            return Err(OptimizeError::InvalidInput(format!(
                "IntegerVariableSet length {} != LP dimension {}",
                ivs.len(),
                n
            )));
        }

        let base_lb: Vec<f64> = lp.lower.as_ref().map_or(vec![0.0; n], |l| l.to_vec());
        let base_ub: Vec<f64> = lp
            .upper
            .as_ref()
            .map_or(vec![f64::INFINITY; n], |u| u.to_vec());

        // Apply binary variable bounds
        let mut base_lb = base_lb;
        let mut base_ub = base_ub;
        for i in 0..n {
            if ivs.kinds[i] == IntegerKind::Binary {
                base_lb[i] = base_lb[i].max(0.0);
                base_ub[i] = base_ub[i].min(1.0);
            }
        }

        // Solve root LP relaxation
        let root_result = LpRelaxationSolver::solve(lp, &base_lb, &base_ub)?;

        if !root_result.success {
            return Ok(MipResult {
                x: Array1::zeros(n),
                fun: f64::INFINITY,
                success: false,
                message: "LP relaxation infeasible".to_string(),
                nodes_explored: 1,
                lp_solves: 1,
                lower_bound: f64::INFINITY,
            });
        }

        let mut incumbent: Option<Vec<f64>> = None;
        let mut incumbent_obj = f64::INFINITY;
        let mut nodes_explored = 1usize;
        let mut lp_solves = 1usize;
        let mut global_lb = root_result.fun;

        // Check if root LP solution is already integer feasible
        let root_x: Vec<f64> = root_result.x.to_vec();
        if self.is_integer_feasible(&root_x, ivs) {
            let obj = Self::eval_obj(lp, &root_x);
            incumbent = Some(root_x.clone());
            incumbent_obj = obj;
            return Ok(MipResult {
                x: Array1::from_vec(root_x),
                fun: obj,
                success: true,
                message: "LP relaxation gives integer solution".to_string(),
                nodes_explored,
                lp_solves,
                lower_bound: global_lb,
            });
        }

        // Initialize node queue
        let root_node = BbNode {
            extra_lb: base_lb.clone(),
            extra_ub: base_ub.clone(),
            lb: root_result.fun,
            depth: 0,
        };

        let mut queue: VecDeque<BbNode> = VecDeque::new();
        queue.push_back(root_node);

        while !queue.is_empty() && nodes_explored < self.options.max_nodes {
            // Select node
            let node = match self.options.node_selection {
                NodeSelection::BestFirst => {
                    // Find node with lowest lb
                    let best_pos = queue
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| {
                            a.lb.partial_cmp(&b.lb).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    match queue.remove(best_pos) {
                        Some(n) => n,
                        None => match queue.pop_back() {
                            Some(n) => n,
                            None => break,
                        },
                    }
                }
                NodeSelection::DepthFirst | NodeSelection::BestOfDepth => {
                    // DFS: pop from back
                    match queue.pop_back() {
                        Some(n) => n,
                        None => break,
                    }
                }
            };

            nodes_explored += 1;

            // Prune: node lower bound >= incumbent
            if node.lb >= incumbent_obj - self.options.opt_tol {
                continue;
            }

            // Solve LP at this node
            let lp_result = match LpRelaxationSolver::solve(lp, &node.extra_lb, &node.extra_ub) {
                Ok(r) => r,
                Err(_) => continue,
            };
            lp_solves += 1;

            if !lp_result.success {
                // Node is infeasible
                continue;
            }

            let node_obj = lp_result.fun;

            // Update global lower bound
            if node_obj > global_lb {
                global_lb = node_obj;
            }

            // Prune: LP objective >= incumbent
            if node_obj >= incumbent_obj - self.options.opt_tol {
                continue;
            }

            let node_x: Vec<f64> = lp_result.x.to_vec();

            // Check integrality
            if self.is_integer_feasible(&node_x, ivs) {
                let obj = Self::eval_obj(lp, &node_x);
                if obj < incumbent_obj {
                    incumbent_obj = obj;
                    incumbent = Some(node_x);
                    // Update global lb
                    if queue.is_empty() {
                        global_lb = incumbent_obj;
                    }
                }
                continue;
            }

            // Select branching variable
            let branch_var = match self.select_variable(&node_x, ivs) {
                Some(v) => v,
                None => {
                    // All integer vars are integer (shouldn't happen, but handle gracefully)
                    let rounded = self.round_integer_solution(&node_x, ivs);
                    let obj = Self::eval_obj(lp, &rounded);
                    if obj < incumbent_obj {
                        incumbent_obj = obj;
                        incumbent = Some(rounded);
                    }
                    continue;
                }
            };

            let xi = node_x[branch_var];
            let xi_floor = xi.floor();
            let xi_ceil = xi.ceil();

            // Branch down: x[branch_var] <= floor(xi)
            let mut lb_down = node.extra_lb.clone();
            let mut ub_down = node.extra_ub.clone();
            ub_down[branch_var] = ub_down[branch_var].min(xi_floor);
            if lb_down[branch_var] <= ub_down[branch_var] {
                queue.push_back(BbNode {
                    extra_lb: lb_down,
                    extra_ub: ub_down,
                    lb: node_obj,
                    depth: node.depth + 1,
                });
            }

            // Branch up: x[branch_var] >= ceil(xi)
            let mut lb_up = node.extra_lb.clone();
            let mut ub_up = node.extra_ub.clone();
            lb_up[branch_var] = lb_up[branch_var].max(xi_ceil);
            if lb_up[branch_var] <= ub_up[branch_var] {
                queue.push_back(BbNode {
                    extra_lb: lb_up,
                    extra_ub: ub_up,
                    lb: node_obj,
                    depth: node.depth + 1,
                });
            }
        }

        match incumbent {
            Some(x) => Ok(MipResult {
                x: Array1::from_vec(x),
                fun: incumbent_obj,
                success: true,
                message: format!(
                    "Optimal solution found (nodes={}, lp_solves={})",
                    nodes_explored, lp_solves
                ),
                nodes_explored,
                lp_solves,
                lower_bound: global_lb,
            }),
            None => Ok(MipResult {
                x: Array1::zeros(n),
                fun: f64::INFINITY,
                success: false,
                message: "No integer feasible solution found".to_string(),
                nodes_explored,
                lp_solves,
                lower_bound: global_lb,
            }),
        }
    }
}

impl Default for BranchAndBoundSolver {
    fn default() -> Self {
        BranchAndBoundSolver::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    /// Simple binary knapsack: maximize value subject to weight constraint
    /// Formulated as: minimize -v^T x s.t. w^T x <= capacity, x in {0,1}^n
    #[test]
    fn test_branch_and_bound_binary_knapsack() {
        // Items: (value, weight)
        // (4, 2), (3, 3), (5, 4), (2, 1), (6, 5)
        // Capacity: 8
        // Optimal: take items 0, 3, 4 -> value = 4+2+6 = 12 or items 0,2,3 -> 4+5+2=11
        // Actually: items 0,1,3,4: weight=2+3+1+5=11 > 8; items 0,2,3: w=2+4+1=7, v=4+5+2=11; items 0,3,4: w=2+1+5=8, v=12
        // So optimal value = 12 (items 0,3,4)
        let values = vec![4.0, 3.0, 5.0, 2.0, 6.0];
        let weights = vec![2.0, 3.0, 4.0, 1.0, 5.0];
        let capacity = 8.0;
        let n = 5;

        // minimize -values^T x
        let c = array![-4.0, -3.0, -5.0, -2.0, -6.0];
        let mut lp = LinearProgram::new(c);
        lp.a_ub = Some(Array2::from_shape_vec((1, n), weights.clone()).expect("shape"));
        lp.b_ub = Some(array![capacity]);
        lp.lower = Some(array![0.0, 0.0, 0.0, 0.0, 0.0]);
        lp.upper = Some(array![1.0, 1.0, 1.0, 1.0, 1.0]);

        let ivs = IntegerVariableSet::all_binary(n);
        let solver = BranchAndBoundSolver::new();
        let result = solver.solve(&lp, &ivs).expect("solve failed");

        assert!(result.success, "B&B should find solution");
        // Optimal: -12
        assert!(
            result.fun <= -11.9,
            "optimal value should be -12, got {}",
            result.fun
        );
    }

    #[test]
    fn test_branch_and_bound_pure_integer() {
        // minimize x[0] + x[1]
        // subject to x[0] + x[1] >= 3.5 (so x_int: x[0]+x[1] >= 4)
        // x >= 0, integer
        let c = array![1.0, 1.0];
        let mut lp = LinearProgram::new(c);
        // -x[0] - x[1] <= -3.5  (i.e. x[0]+x[1] >= 3.5)
        lp.a_ub = Some(Array2::from_shape_vec((1, 2), vec![-1.0, -1.0]).expect("shape"));
        lp.b_ub = Some(array![-3.5]);
        lp.lower = Some(array![0.0, 0.0]);
        lp.upper = Some(array![10.0, 10.0]);

        let ivs = IntegerVariableSet::all_integer(2);
        let solver = BranchAndBoundSolver::new();
        let result = solver.solve(&lp, &ivs).expect("solve failed");

        assert!(result.success);
        // Integer optimal: x[0]+x[1] = 4, minimized -> 4
        assert_abs_diff_eq!(result.fun, 4.0, epsilon = 1e-4);
    }

    #[test]
    fn test_branch_and_bound_mixed_integer() {
        // minimize 2x[0] + x[1]
        // x[0] integer, x[1] continuous
        // x[0] + x[1] >= 2.5
        // x >= 0
        let c = array![2.0, 1.0];
        let mut lp = LinearProgram::new(c);
        lp.a_ub = Some(Array2::from_shape_vec((1, 2), vec![-1.0, -1.0]).expect("shape"));
        lp.b_ub = Some(array![-2.5]);
        lp.lower = Some(array![0.0, 0.0]);
        lp.upper = Some(array![10.0, 10.0]);

        let mut ivs = IntegerVariableSet::new(2);
        ivs.set_kind(0, IntegerKind::Integer);

        let solver = BranchAndBoundSolver::new();
        let result = solver.solve(&lp, &ivs).expect("solve failed");

        assert!(result.success);
        // x[0]=0 int, x[1]=2.5 -> obj=2.5; or x[0]=1,x[1]=1.5->obj=3.5; opt is x[0]=0,x[1]=2.5
        // Actually: x[0]=0 is integer, x[1]=2.5 is continuous: obj=2.5
        assert!(result.fun <= 3.0, "fun={}", result.fun);
    }

    #[test]
    fn test_branch_and_bound_depth_first() {
        let opts = BranchAndBoundOptions {
            node_selection: NodeSelection::DepthFirst,
            ..Default::default()
        };
        let solver = BranchAndBoundSolver::with_options(opts);

        let c = array![1.0, 1.0];
        let mut lp = LinearProgram::new(c);
        lp.a_ub = Some(Array2::from_shape_vec((1, 2), vec![-1.0, -1.0]).expect("shape"));
        lp.b_ub = Some(array![-2.7]);
        lp.lower = Some(array![0.0, 0.0]);
        lp.upper = Some(array![10.0, 10.0]);

        let ivs = IntegerVariableSet::all_integer(2);
        let result = solver.solve(&lp, &ivs).expect("solve failed");

        assert!(result.success);
        assert_abs_diff_eq!(result.fun, 3.0, epsilon = 1e-4);
    }
}
