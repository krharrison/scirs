//! Mixed-Integer Linear Programming via Branch-and-Bound
//!
//! This module provides the top-level [`MilpProblem`] struct that encapsulates
//! the MILP formulation and the [`branch_and_bound`] function that solves it.
//!
//! # Problem formulation
//!
//! ```text
//! minimize    c^T x
//! subject to  A x <= b               (linear inequalities)
//!             lb <= x <= ub           (variable bounds)
//!             x_i ∈ Z  for i in I    (integrality constraints)
//! ```
//!
//! # Algorithm
//!
//! - **LP relaxation** at each node via revised simplex (Phase I + Phase II).
//! - **Variable selection**: most-fractional or strong branching.
//! - **Node selection**: best-first (lowest LP lower bound).
//! - **Pruning**: infeasible nodes, integral nodes, bound-based cutoff.
//!
//! # References
//! - Land, A.H. & Doig, A.G. (1960). "An automatic method of solving discrete
//!   programming problems." Econometrica, 28(3), 497–520.
//! - Wolsey, L.A. (1998). *Integer Programming*. Wiley.

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{Array1, Array2};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// ─────────────────────────────────────────────────────────────────────────────
// MILP Problem Definition
// ─────────────────────────────────────────────────────────────────────────────

/// Mixed-Integer Linear Programming problem.
///
/// Represents:
/// ```text
/// minimize    c^T x
/// subject to  A x <= b
///             lb <= x <= ub
///             x_i ∈ Z  for all i in integer_vars
/// ```
#[derive(Debug, Clone)]
pub struct MilpProblem {
    /// Objective coefficients (length n).
    pub c: Array1<f64>,
    /// Constraint matrix (m × n); represents `A x <= b`.
    pub a: Array2<f64>,
    /// Constraint RHS (length m).
    pub b: Array1<f64>,
    /// Lower bounds for each variable (length n; default 0).
    pub lb: Array1<f64>,
    /// Upper bounds for each variable (length n; default +∞).
    pub ub: Array1<f64>,
    /// Indices of variables that must be integer.
    pub integer_vars: Vec<usize>,
}

impl MilpProblem {
    /// Construct a new MILP problem.
    ///
    /// # Panics
    /// Does not panic; returns [`OptimizeError::InvalidInput`] on invalid dimensions.
    pub fn new(
        c: Array1<f64>,
        a: Array2<f64>,
        b: Array1<f64>,
        lb: Array1<f64>,
        ub: Array1<f64>,
        integer_vars: Vec<usize>,
    ) -> OptimizeResult<Self> {
        let n = c.len();
        let (m, ncols) = a.dim();
        if ncols != n {
            return Err(OptimizeError::InvalidInput(format!(
                "A has {} columns but c has {} entries",
                ncols, n
            )));
        }
        if b.len() != m {
            return Err(OptimizeError::InvalidInput(format!(
                "b has length {} but A has {} rows",
                b.len(),
                m
            )));
        }
        if lb.len() != n || ub.len() != n {
            return Err(OptimizeError::InvalidInput(
                "lb and ub must have the same length as c".to_string(),
            ));
        }
        for &idx in &integer_vars {
            if idx >= n {
                return Err(OptimizeError::InvalidInput(format!(
                    "integer_vars contains out-of-range index {}",
                    idx
                )));
            }
        }
        Ok(MilpProblem {
            c,
            a,
            b,
            lb,
            ub,
            integer_vars,
        })
    }

    /// Number of variables.
    #[inline]
    pub fn n_vars(&self) -> usize {
        self.c.len()
    }

    /// Number of constraints.
    #[inline]
    pub fn n_constraints(&self) -> usize {
        self.b.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Variable selection strategy in B&B
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchingStrategy {
    /// Branch on the variable whose value is closest to 0.5 (most fractional).
    MostFractional,
    /// Branch on the first fractional variable encountered.
    FirstFractional,
    /// Evaluate a small number of candidates and choose the one giving the
    /// best LP bound improvement (strong branching, with a limited trial budget).
    StrongBranching,
}

/// Configuration for the branch-and-bound solver.
#[derive(Debug, Clone)]
pub struct BnbConfig {
    /// Maximum number of B&B nodes to explore.
    pub max_nodes: usize,
    /// Wall-clock time limit in seconds (0 = no limit).
    pub time_limit_secs: f64,
    /// Absolute gap tolerance: stop when `incumbent − lower_bound ≤ abs_gap`.
    pub abs_gap: f64,
    /// Relative gap tolerance: stop when `gap / |incumbent| ≤ rel_gap`.
    pub rel_gap: f64,
    /// Integrality tolerance.
    pub int_tol: f64,
    /// Variable selection strategy.
    pub branching: BranchingStrategy,
    /// Number of candidate variables to evaluate in strong branching.
    pub strong_branching_candidates: usize,
}

impl Default for BnbConfig {
    fn default() -> Self {
        BnbConfig {
            max_nodes: 50_000,
            time_limit_secs: 0.0,
            abs_gap: 1e-6,
            rel_gap: 1e-6,
            int_tol: 1e-6,
            branching: BranchingStrategy::MostFractional,
            strong_branching_candidates: 5,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Result
// ─────────────────────────────────────────────────────────────────────────────

/// Result returned by [`branch_and_bound`].
#[derive(Debug, Clone)]
pub struct MilpResult {
    /// Optimal solution vector (length n).
    pub x: Array1<f64>,
    /// Optimal objective value (`c^T x`).
    pub obj: f64,
    /// Whether an optimal solution was found.
    pub success: bool,
    /// Solver status message.
    pub message: String,
    /// Number of B&B nodes explored.
    pub nodes_explored: usize,
    /// Lower bound on the optimal objective at termination.
    pub lower_bound: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// LP Solver (Revised Simplex with Phase I / Phase II)
// ─────────────────────────────────────────────────────────────────────────────

/// Status returned by the LP solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LpStatus {
    Optimal,
    Infeasible,
    Unbounded,
}

/// LP solution
struct LpSolution {
    x: Vec<f64>,
    obj: f64,
    status: LpStatus,
}

/// Revised-simplex LP solver.
///
/// Solves `min c^T x` subject to:
///   `A_ub x <= b_ub`
///   `lb <= x <= ub`
///
/// We convert to standard form by:
/// - Adding slack variables `s_i >= 0` for each inequality: `A x + s = b_ub`
/// - Shifting `x_j <- x_j - lb[j]` so all variables are non-negative
///   (after adding `ub_shift = ub[j] - lb[j]` as an upper bound)
/// - Adding artificial variables for Phase I (big-M method) to handle RHS < 0
fn solve_lp(
    c: &[f64],         // n
    a_ub: &[Vec<f64>], // m × n
    b_ub: &[f64],      // m
    lb: &[f64],        // n
    ub: &[f64],        // n
    max_iter: usize,
) -> LpSolution {
    let n = c.len();
    let m = a_ub.len();

    if n == 0 {
        return LpSolution {
            x: Vec::new(),
            obj: 0.0,
            status: LpStatus::Optimal,
        };
    }

    // --- Bound feasibility check ----------------------------------------
    for i in 0..n {
        if lb[i] > ub[i] + 1e-10 {
            return LpSolution {
                x: vec![0.0; n],
                obj: f64::INFINITY,
                status: LpStatus::Infeasible,
            };
        }
    }

    // --- Shift x <- x - lb so that all vars start non-negative ----------
    // New variable y = x - lb; bounds: 0 <= y <= ub - lb
    let ub_shifted: Vec<f64> = (0..n).map(|i| ub[i] - lb[i]).collect();
    // Shift b_ub: A(y + lb) <= b => Ay <= b - A*lb
    let b_shifted: Vec<f64> = (0..m)
        .map(|i| {
            b_ub[i]
                - a_ub[i]
                    .iter()
                    .zip(lb.iter())
                    .map(|(&a, &l)| a * l)
                    .sum::<f64>()
        })
        .collect();

    // --- Add slacks: A y + s = b_shifted (s >= 0) -------------------------
    // Handle finite upper bounds on y with surplus variables:
    //   y_j <= ub_j  =>  y_j + t_j = ub_j  (t_j >= 0)
    let n_ub_constrained: usize = ub_shifted.iter().filter(|&&u| u.is_finite()).count();
    let n_total = n + m + n_ub_constrained; // structural + slacks + UB slacks

    // Map structural variable index -> UB slack column index
    let mut ub_slack_for: Vec<Option<usize>> = vec![None; n];
    let mut ub_slack_idx = n + m; // start after structural + inequality slacks
    for j in 0..n {
        if ub_shifted[j].is_finite() {
            ub_slack_for[j] = Some(ub_slack_idx);
            ub_slack_idx += 1;
        }
    }

    // Build full constraint matrix [A | I_m | (UB slack columns)]
    // rows: m (ineq slacks) + n_ub_constrained (UB bound rows)
    let total_rows = m + n_ub_constrained;
    let mut full_a: Vec<Vec<f64>> = vec![vec![0.0; n_total]; total_rows];
    let mut full_b: Vec<f64> = vec![0.0; total_rows];

    // Fill inequality rows
    for i in 0..m {
        for j in 0..n {
            full_a[i][j] = a_ub[i][j];
        }
        // Slack variable for row i is at column n + i
        full_a[i][n + i] = 1.0;
        full_b[i] = b_shifted[i];
    }

    // Fill upper bound rows: y_j + t_j = ub_shifted[j]
    let mut ub_row = m;
    for j in 0..n {
        if let Some(sk) = ub_slack_for[j] {
            full_a[ub_row][j] = 1.0;
            full_a[ub_row][sk] = 1.0;
            full_b[ub_row] = ub_shifted[j];
            ub_row += 1;
        }
    }

    // --- Phase I: find BFS using big-M / artificial variables ------------
    // For rows with b >= 0, the slack/UB-slack is a valid basic variable.
    // For rows with b < 0, we negate the row (so RHS becomes positive) and
    // add an artificial variable (because the negated slack now has coeff -1).

    let mut a_work = full_a.clone();
    let mut b_work = full_b.clone();
    let mut needs_artif = vec![false; total_rows];
    for i in 0..total_rows {
        if b_work[i] < -1e-12 {
            for v in a_work[i].iter_mut() {
                *v = -*v;
            }
            b_work[i] = -b_work[i];
            needs_artif[i] = true;
        }
    }

    let n_artif: usize = needs_artif.iter().filter(|&&v| v).count();
    let n_total_ext = n_total + n_artif; // extended with artificial columns

    // Extend rows to include artificial columns
    let mut artif_col_idx = n_total;
    let mut artif_map: Vec<Option<usize>> = vec![None; total_rows]; // row -> artificial col
    for i in 0..total_rows {
        if needs_artif[i] {
            artif_map[i] = Some(artif_col_idx);
            artif_col_idx += 1;
        }
    }

    // Extend all rows with zero columns for artificials
    for row in a_work.iter_mut() {
        row.resize(n_total_ext, 0.0);
    }
    // Set artificial variable coefficients to +1 for their respective rows
    for i in 0..total_rows {
        if let Some(acol) = artif_map[i] {
            a_work[i][acol] = 1.0;
        }
    }

    // Objective: original c for structural vars, 0 for slacks, big-M for artificials
    let big_m = 1e7_f64;
    let mut big_m_c: Vec<f64> = vec![0.0; n_total_ext];
    for j in 0..n {
        big_m_c[j] = c[j];
    }
    for i in 0..total_rows {
        if let Some(acol) = artif_map[i] {
            big_m_c[acol] = big_m;
        }
    }

    // Initial basis: for rows not needing artificials, use slack/UB-slack;
    // for rows needing artificials, use the artificial variable.
    let mut basis: Vec<usize> = Vec::with_capacity(total_rows);
    let mut ub_row_counter = 0usize;
    for i in 0..m {
        if needs_artif[i] {
            basis.push(artif_map[i].unwrap_or(0));
        } else {
            basis.push(n + i); // slack for ineq row i
        }
    }
    for j in 0..n {
        if let Some(sk) = ub_slack_for[j] {
            let row_idx = m + ub_row_counter;
            if needs_artif[row_idx] {
                basis.push(artif_map[row_idx].unwrap_or(0));
            } else {
                basis.push(sk);
            }
            ub_row_counter += 1;
        }
    }

    // Run revised simplex
    let sol = revised_simplex(&mut a_work, &mut b_work, &big_m_c, &mut basis, max_iter);

    if sol.status == LpStatus::Infeasible || sol.status == LpStatus::Unbounded {
        return LpSolution {
            x: vec![0.0; n],
            obj: f64::INFINITY,
            status: LpStatus::Infeasible,
        };
    }

    // Check that no artificial is in the basis with positive value
    for (i, &bv) in basis.iter().enumerate() {
        if bv >= n_total && b_work[i] > 1e-6 {
            // Artificial variable still in basis -> LP is infeasible
            return LpSolution {
                x: vec![0.0; n],
                obj: f64::INFINITY,
                status: LpStatus::Infeasible,
            };
        }
    }

    // Extract solution: y (shifted) from sol.x (only first n_total variables)
    let y = &sol.x;
    let x: Vec<f64> = (0..n)
        .map(|j| {
            let yj = if j < y.len() { y[j] } else { 0.0 };
            (lb[j] + yj).max(lb[j]).min(ub[j])
        })
        .collect();
    let obj = c
        .iter()
        .zip(x.iter())
        .map(|(&ci, &xi)| ci * xi)
        .sum::<f64>();

    LpSolution {
        x,
        obj,
        status: LpStatus::Optimal,
    }
}

/// Simplified revised simplex method (tableau form).
///
/// Operates on the system `A x = b` with `x >= 0`.
/// `basis` is the initial basic feasible basis (indices of basic variables).
fn revised_simplex(
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
    c: &[f64],
    basis: &mut Vec<usize>,
    max_iter: usize,
) -> LpSolution {
    let m = a.len();
    if m == 0 {
        let n = c.len();
        let mut x = vec![0.0_f64; n];
        // Minimise: set vars to lb (which is 0 after shift)
        return LpSolution {
            x,
            obj: 0.0,
            status: LpStatus::Optimal,
        };
    }
    let n_total = if m > 0 { a[0].len() } else { 0 };

    // Build basis inverse (B^{-1}) as an m×m identity initially,
    // then update with pivot operations.  We work with a full tableau for simplicity.
    // Full tableau: [A | I] -> after entering each column we update.

    // We use the simplex tableau directly:
    // tableau[i][j] = B^{-1} A_j  for structural column j
    // tableau[i][m] = B^{-1} b
    // Reduced costs: c_bar[j] = c[j] - c_B B^{-1} A_j

    let mut tableau: Vec<Vec<f64>> = vec![vec![0.0; n_total + 1]; m];
    for i in 0..m {
        for j in 0..n_total {
            tableau[i][j] = a[i][j];
        }
        tableau[i][n_total] = b[i];
    }

    // Make basis columns identity via row operations (initial BFS)
    for col in 0..m {
        let basic = basis[col];
        // Find the row where this column has 1.0 (it should, by construction)
        // Actually the initial basis columns may not be unit vectors if rows were flipped.
        // Pivot to make basis[col] a unit vector in column col.
        let pivot_row = col; // assume basis variable for row col is in position col
                             // Find pivot in this column among rows
        let pivot_val = tableau[pivot_row][basic];
        if pivot_val.abs() < 1e-12 {
            // Try to find a different row for this basis element
            let mut found = false;
            for i in 0..m {
                if i != pivot_row && tableau[i][basic].abs() > 1e-10 {
                    tableau.swap(pivot_row, i);
                    basis.swap(pivot_row, i);
                    found = true;
                    break;
                }
            }
            if !found {
                continue;
            }
        }
        let pv = tableau[pivot_row][basic];
        if pv.abs() < 1e-12 {
            continue;
        }
        for j in 0..=n_total {
            tableau[pivot_row][j] /= pv;
        }
        for i in 0..m {
            if i == pivot_row {
                continue;
            }
            let factor = tableau[i][basic];
            if factor.abs() < 1e-15 {
                continue;
            }
            for j in 0..=n_total {
                let delta = factor * tableau[pivot_row][j];
                tableau[i][j] -= delta;
            }
        }
    }

    // Simplex iterations
    for _iter in 0..max_iter {
        // Compute reduced costs
        let c_b: Vec<f64> = basis
            .iter()
            .map(|&b| c.get(b).copied().unwrap_or(0.0))
            .collect();

        let mut enter = None;
        let mut min_rc = -1e-8_f64;
        for j in 0..n_total {
            // rc = c[j] - c_B^T B^{-1} A_j = c[j] - c_B^T tableau_col
            let rc = c.get(j).copied().unwrap_or(0.0)
                - c_b
                    .iter()
                    .zip(tableau.iter())
                    .map(|(&cb, row)| cb * row[j])
                    .sum::<f64>();
            if rc < min_rc {
                min_rc = rc;
                enter = Some(j);
            }
        }

        let enter_col = match enter {
            None => break, // optimal
            Some(j) => j,
        };

        // Minimum ratio test
        let mut leave_row = None;
        let mut min_ratio = f64::INFINITY;
        for i in 0..m {
            let coef = tableau[i][enter_col];
            if coef > 1e-10 {
                let ratio = tableau[i][n_total] / coef;
                if ratio < min_ratio {
                    min_ratio = ratio;
                    leave_row = Some(i);
                }
            }
        }

        let pivot_row = match leave_row {
            None => {
                // Unbounded
                let mut x = vec![0.0; n_total];
                for (i, &b) in basis.iter().enumerate() {
                    if b < n_total {
                        x[b] = tableau[i][n_total].max(0.0);
                    }
                }
                return LpSolution {
                    x,
                    obj: f64::NEG_INFINITY,
                    status: LpStatus::Unbounded,
                };
            }
            Some(r) => r,
        };

        // Pivot
        let pv = tableau[pivot_row][enter_col];
        for j in 0..=n_total {
            tableau[pivot_row][j] /= pv;
        }
        for i in 0..m {
            if i == pivot_row {
                continue;
            }
            let factor = tableau[i][enter_col];
            if factor.abs() < 1e-15 {
                continue;
            }
            for j in 0..=n_total {
                let delta = factor * tableau[pivot_row][j];
                tableau[i][j] -= delta;
            }
        }
        basis[pivot_row] = enter_col;
    }

    // Extract solution
    let mut x = vec![0.0_f64; n_total];
    for (i, &b) in basis.iter().enumerate() {
        if b < n_total {
            x[b] = tableau[i][n_total].max(0.0);
        }
    }

    // Update b (for external use)
    for i in 0..m {
        b[i] = tableau[i][n_total];
    }

    let obj: f64 = c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum();
    LpSolution {
        x,
        obj,
        status: LpStatus::Optimal,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// B&B Node
// ─────────────────────────────────────────────────────────────────────────────

/// A node in the branch-and-bound tree.
#[derive(Debug, Clone)]
struct BbNode {
    /// Tightened lower bounds (length n), incorporating branching constraints.
    lb: Vec<f64>,
    /// Tightened upper bounds (length n), incorporating branching constraints.
    ub: Vec<f64>,
    /// LP lower bound at this node (for priority queue ordering).
    lp_lb: f64,
    /// Depth in the tree (for tie-breaking).
    depth: usize,
}

/// Wrapper for priority queue (max-heap, negated to get min-heap by lp_lb).
struct PqEntry {
    neg_lb: f64,
    node: BbNode,
}

impl PartialEq for PqEntry {
    fn eq(&self, other: &Self) -> bool {
        self.neg_lb == other.neg_lb
    }
}

impl Eq for PqEntry {}

impl PartialOrd for PqEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PqEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.neg_lb
            .partial_cmp(&other.neg_lb)
            .unwrap_or(Ordering::Equal)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: check/evaluate integrality
// ─────────────────────────────────────────────────────────────────────────────

fn is_integer_valued_local(v: f64, tol: f64) -> bool {
    (v - v.round()).abs() <= tol
}

fn eval_obj_vec(c: &[f64], x: &[f64]) -> f64 {
    c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// Variable selection
// ─────────────────────────────────────────────────────────────────────────────

fn select_most_fractional(x: &[f64], int_vars: &[usize], tol: f64) -> Option<usize> {
    let mut best = None;
    let mut best_dist = -1.0_f64;
    for &j in int_vars {
        let xi = x[j];
        let frac = (xi - xi.floor()).min(xi.ceil() - xi);
        if frac > tol && frac > best_dist {
            best_dist = frac;
            best = Some(j);
        }
    }
    best
}

fn select_first_fractional(x: &[f64], int_vars: &[usize], tol: f64) -> Option<usize> {
    for &j in int_vars {
        if !is_integer_valued_local(x[j], tol) {
            return Some(j);
        }
    }
    None
}

/// Strong branching: solve 2 mini-LPs for each candidate and pick the one
/// with the best min(down_lb, up_lb).
fn select_strong_branching(
    x: &[f64],
    int_vars: &[usize],
    tol: f64,
    config: &BnbConfig,
    problem: &MilpProblem,
    node_lb: &[f64],
    node_ub: &[f64],
) -> Option<usize> {
    let fractional: Vec<usize> = int_vars
        .iter()
        .copied()
        .filter(|&j| !is_integer_valued_local(x[j], tol))
        .collect();
    if fractional.is_empty() {
        return None;
    }

    // Sort candidates by most-fractional order to limit probing
    let mut candidates: Vec<(usize, f64)> = fractional
        .iter()
        .copied()
        .map(|j| {
            let frac = (x[j] - x[j].floor()).min(x[j].ceil() - x[j]);
            (j, frac)
        })
        .collect();
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    let k = candidates.len().min(config.strong_branching_candidates);
    candidates.truncate(k);

    let n = problem.n_vars();
    let m = problem.n_constraints();
    let a_rows: Vec<Vec<f64>> = (0..m)
        .map(|i| (0..n).map(|j| problem.a[[i, j]]).collect())
        .collect();
    let b_vec: Vec<f64> = problem.b.to_vec();
    let c_vec: Vec<f64> = problem.c.to_vec();

    let mut best_var = None;
    let mut best_score = f64::NEG_INFINITY;

    for &(var, _) in &candidates {
        let xi = x[var];
        let xi_floor = xi.floor();
        let xi_ceil = xi.ceil();

        // Down branch: ub[var] = floor
        let mut lb_d = node_lb.to_vec();
        let mut ub_d = node_ub.to_vec();
        ub_d[var] = ub_d[var].min(xi_floor);
        let down = solve_lp(&c_vec, &a_rows, &b_vec, &lb_d, &ub_d, 500);

        // Up branch: lb[var] = ceil
        let mut lb_u = node_lb.to_vec();
        let mut ub_u = node_ub.to_vec();
        lb_u[var] = lb_u[var].max(xi_ceil);
        let up = solve_lp(&c_vec, &a_rows, &b_vec, &lb_u, &ub_u, 500);

        let down_bound = if down.status == LpStatus::Infeasible {
            f64::INFINITY
        } else {
            down.obj
        };
        let up_bound = if up.status == LpStatus::Infeasible {
            f64::INFINITY
        } else {
            up.obj
        };

        // Score: max of the two bounds (prefer the branch with higher minimum bound)
        let score = down_bound.min(up_bound);
        if score > best_score {
            best_score = score;
            best_var = Some(var);
        }
    }

    best_var
}

// ─────────────────────────────────────────────────────────────────────────────
// Main branch-and-bound entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Solve a Mixed-Integer Linear Program via branch-and-bound.
///
/// # Arguments
/// * `problem` – the [`MilpProblem`] to solve
/// * `config`  – [`BnbConfig`] controlling solver behaviour
///
/// # Returns
/// A [`MilpResult`] containing the solution, objective value, and statistics.
///
/// # Errors
/// Returns [`OptimizeError::InvalidInput`] if `problem` has inconsistent dimensions.
///
/// # Example
/// ```rust,no_run
/// use scirs2_optimize::integer::milp_branch_and_bound::{
///     MilpProblem, BnbConfig, branch_and_bound,
/// };
/// use scirs2_core::ndarray::{array, Array2};
///
/// // maximize 4x0 + 3x1 s.t. 2x0 + 3x1 <= 6, x in {0,1}^2
/// // (as minimization: minimize -4x0 - 3x1)
/// let c  = array![-4.0, -3.0];
/// let a  = Array2::from_shape_vec((1, 2), vec![2.0, 3.0]).expect("valid input");
/// let b  = array![6.0];
/// let lb = array![0.0, 0.0];
/// let ub = array![1.0, 1.0];
/// let prob = MilpProblem::new(c, a, b, lb, ub, vec![0, 1]).expect("valid input");
/// let cfg = BnbConfig::default();
/// let res = branch_and_bound(&prob, &cfg).expect("valid input");
/// assert!(res.success);
/// ```
pub fn branch_and_bound(problem: &MilpProblem, config: &BnbConfig) -> OptimizeResult<MilpResult> {
    let n = problem.n_vars();
    let m = problem.n_constraints();

    if n == 0 {
        return Err(OptimizeError::InvalidInput(
            "Problem has no variables".to_string(),
        ));
    }

    let start_time = std::time::Instant::now();

    let c_vec: Vec<f64> = problem.c.to_vec();
    let b_vec: Vec<f64> = problem.b.to_vec();
    let a_rows: Vec<Vec<f64>> = (0..m)
        .map(|i| (0..n).map(|j| problem.a[[i, j]]).collect())
        .collect();

    let base_lb: Vec<f64> = problem.lb.to_vec();
    let base_ub: Vec<f64> = problem.ub.to_vec();

    // Apply binary variable bounds (treat vars with ub=1 and lb=0 as binary)
    let mut root_lb = base_lb.clone();
    let mut root_ub = base_ub.clone();
    for &j in &problem.integer_vars {
        // Clamp to integer bounds
        root_lb[j] = root_lb[j].ceil();
        root_ub[j] = root_ub[j].floor();
        if root_lb[j] > root_ub[j] {
            // Infeasible by bounds alone
            return Ok(MilpResult {
                x: Array1::zeros(n),
                obj: f64::INFINITY,
                success: false,
                message: format!("Variable {} has empty integer domain", j),
                nodes_explored: 0,
                lower_bound: f64::INFINITY,
            });
        }
    }

    // Solve root LP relaxation
    let root_lp = solve_lp(&c_vec, &a_rows, &b_vec, &root_lb, &root_ub, 5000);

    if root_lp.status == LpStatus::Infeasible {
        return Ok(MilpResult {
            x: Array1::zeros(n),
            obj: f64::INFINITY,
            success: false,
            message: "Root LP relaxation is infeasible".to_string(),
            nodes_explored: 1,
            lower_bound: f64::INFINITY,
        });
    }
    if root_lp.status == LpStatus::Unbounded {
        return Ok(MilpResult {
            x: Array1::zeros(n),
            obj: f64::NEG_INFINITY,
            success: false,
            message: "Root LP relaxation is unbounded".to_string(),
            nodes_explored: 1,
            lower_bound: f64::NEG_INFINITY,
        });
    }

    let mut incumbent: Option<Vec<f64>> = None;
    let mut incumbent_obj = f64::INFINITY;
    let mut nodes_explored = 1usize;
    let mut global_lb = root_lp.obj;

    // Check if root LP is already integer feasible
    let root_x = root_lp.x;
    let all_int = problem
        .integer_vars
        .iter()
        .all(|&j| is_integer_valued_local(root_x[j], config.int_tol));
    if all_int {
        let obj = eval_obj_vec(&c_vec, &root_x);
        return Ok(MilpResult {
            x: Array1::from_vec(root_x),
            obj,
            success: true,
            message: "Root LP relaxation is integer feasible".to_string(),
            nodes_explored,
            lower_bound: obj,
        });
    }

    // Initialize priority queue (best-first by LP lower bound)
    let root_node = BbNode {
        lb: root_lb,
        ub: root_ub,
        lp_lb: root_lp.obj,
        depth: 0,
    };

    let mut pq: BinaryHeap<PqEntry> = BinaryHeap::new();
    pq.push(PqEntry {
        neg_lb: -root_node.lp_lb,
        node: root_node,
    });

    while let Some(PqEntry { node, .. }) = pq.pop() {
        nodes_explored += 1;

        // Check termination conditions
        if nodes_explored > config.max_nodes {
            break;
        }
        if config.time_limit_secs > 0.0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            if elapsed >= config.time_limit_secs {
                break;
            }
        }

        // Prune: node lower bound >= incumbent
        if node.lp_lb >= incumbent_obj - config.abs_gap {
            continue;
        }
        if incumbent_obj.abs() > 1e-10 {
            let gap = (incumbent_obj - node.lp_lb) / incumbent_obj.abs();
            if gap <= config.rel_gap {
                continue;
            }
        }

        // Solve LP at this node
        let lp = solve_lp(&c_vec, &a_rows, &b_vec, &node.lb, &node.ub, 3000);

        if lp.status == LpStatus::Infeasible {
            continue;
        }
        if lp.status == LpStatus::Unbounded {
            continue;
        }

        let lp_obj = lp.obj;

        // Update global lower bound
        if lp_obj > global_lb {
            global_lb = lp_obj;
        }

        // Prune: LP objective >= incumbent
        if lp_obj >= incumbent_obj - config.abs_gap {
            continue;
        }

        let lp_x = lp.x;

        // Check integrality
        let int_feasible = problem
            .integer_vars
            .iter()
            .all(|&j| is_integer_valued_local(lp_x[j], config.int_tol));

        if int_feasible {
            let obj = eval_obj_vec(&c_vec, &lp_x);
            if obj < incumbent_obj {
                incumbent_obj = obj;
                incumbent = Some(lp_x);
                global_lb = global_lb.min(incumbent_obj);
            }
            continue;
        }

        // Select branching variable
        let branch_var = match config.branching {
            BranchingStrategy::MostFractional => {
                select_most_fractional(&lp_x, &problem.integer_vars, config.int_tol)
            }
            BranchingStrategy::FirstFractional => {
                select_first_fractional(&lp_x, &problem.integer_vars, config.int_tol)
            }
            BranchingStrategy::StrongBranching => select_strong_branching(
                &lp_x,
                &problem.integer_vars,
                config.int_tol,
                config,
                problem,
                &node.lb,
                &node.ub,
            ),
        };

        let branch_var = match branch_var {
            Some(v) => v,
            None => {
                // All integer variables are integral (shouldn't happen but handle it)
                let obj = eval_obj_vec(&c_vec, &lp_x);
                if obj < incumbent_obj {
                    incumbent_obj = obj;
                    incumbent = Some(lp_x);
                }
                continue;
            }
        };

        let xi = lp_x[branch_var];
        let xi_floor = xi.floor();
        let xi_ceil = xi.ceil();

        // Down branch: x[branch_var] <= floor(xi)
        {
            let mut lb_d = node.lb.clone();
            let mut ub_d = node.ub.clone();
            ub_d[branch_var] = ub_d[branch_var].min(xi_floor);
            if lb_d[branch_var] <= ub_d[branch_var] + 1e-10 {
                pq.push(PqEntry {
                    neg_lb: -lp_obj,
                    node: BbNode {
                        lb: lb_d,
                        ub: ub_d,
                        lp_lb: lp_obj,
                        depth: node.depth + 1,
                    },
                });
            }
        }

        // Up branch: x[branch_var] >= ceil(xi)
        {
            let mut lb_u = node.lb.clone();
            let mut ub_u = node.ub.clone();
            lb_u[branch_var] = lb_u[branch_var].max(xi_ceil);
            if lb_u[branch_var] <= ub_u[branch_var] + 1e-10 {
                pq.push(PqEntry {
                    neg_lb: -lp_obj,
                    node: BbNode {
                        lb: lb_u,
                        ub: ub_u,
                        lp_lb: lp_obj,
                        depth: node.depth + 1,
                    },
                });
            }
        }
    }

    // Update global lb from remaining queue
    if let Some(PqEntry { node, .. }) = pq.peek() {
        if node.lp_lb > global_lb {
            global_lb = node.lp_lb;
        }
    }

    match incumbent {
        Some(x) => Ok(MilpResult {
            x: Array1::from_vec(x),
            obj: incumbent_obj,
            success: true,
            message: format!(
                "Optimal solution found (nodes={}, gap={:.2e})",
                nodes_explored,
                (incumbent_obj - global_lb).abs()
            ),
            nodes_explored,
            lower_bound: global_lb,
        }),
        None => Ok(MilpResult {
            x: Array1::zeros(n),
            obj: f64::INFINITY,
            success: false,
            message: format!(
                "No integer feasible solution found in {} nodes",
                nodes_explored
            ),
            nodes_explored,
            lower_bound: global_lb,
        }),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    // Helper: create a simple binary knapsack MILP
    // maximize sum(v*x) s.t. sum(w*x) <= cap, x in {0,1}
    fn make_knapsack(values: &[f64], weights: &[f64], cap: f64) -> MilpProblem {
        let n = values.len();
        let c = Array1::from_vec(values.iter().map(|&v| -v).collect());
        let a = Array2::from_shape_vec((1, n), weights.to_vec()).expect("shape");
        let b = array![cap];
        let lb = Array1::zeros(n);
        let ub = Array1::ones(n);
        let int_vars = (0..n).collect();
        MilpProblem::new(c, a, b, lb, ub, int_vars).expect("valid problem")
    }

    #[test]
    fn test_milp_binary_knapsack() {
        let values = vec![4.0, 3.0, 5.0, 2.0, 6.0];
        let weights = vec![2.0, 3.0, 4.0, 1.0, 5.0];
        let prob = make_knapsack(&values, &weights, 8.0);
        let cfg = BnbConfig::default();
        let res = branch_and_bound(&prob, &cfg).expect("failed to create res");
        assert!(res.success, "B&B should find solution");
        // Optimal: items 0,3,4 with total value 12
        assert!(
            res.obj <= -11.9,
            "optimal obj should be ~-12, got {}",
            res.obj
        );
    }

    #[test]
    fn test_milp_pure_integer() {
        // min x + y, x+y >= 3.5, x,y >= 0 integer
        // optimal: x+y = 4, obj = 4
        let c = array![1.0, 1.0];
        let a = Array2::from_shape_vec((1, 2), vec![-1.0, -1.0]).expect("failed to create a");
        let b = array![-3.5];
        let lb = array![0.0, 0.0];
        let ub = array![10.0, 10.0];
        let prob = MilpProblem::new(c, a, b, lb, ub, vec![0, 1]).expect("failed to create prob");
        let cfg = BnbConfig::default();
        let res = branch_and_bound(&prob, &cfg).expect("failed to create res");
        assert!(res.success);
        assert_abs_diff_eq!(res.obj, 4.0, epsilon = 1e-4);
    }

    #[test]
    fn test_milp_lp_optimal_already_integer() {
        // LP relaxation already gives integer solution
        let c = array![1.0, 2.0];
        let a = Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).expect("failed to create a");
        let b = array![10.0];
        let lb = array![2.0, 3.0];
        let ub = array![5.0, 6.0];
        let prob = MilpProblem::new(c, a, b, lb, ub, vec![0, 1]).expect("failed to create prob");
        let cfg = BnbConfig::default();
        let res = branch_and_bound(&prob, &cfg).expect("failed to create res");
        assert!(res.success);
        // min at lb = (2,3) -> obj = 2 + 6 = 8
        assert_abs_diff_eq!(res.obj, 8.0, epsilon = 1e-3);
    }

    #[test]
    fn test_milp_strong_branching() {
        let values = vec![6.0, 5.0, 4.0, 3.0];
        let weights = vec![3.0, 3.0, 2.0, 1.0];
        let prob = make_knapsack(&values, &weights, 6.0);
        let cfg = BnbConfig {
            branching: BranchingStrategy::StrongBranching,
            ..Default::default()
        };
        let res = branch_and_bound(&prob, &cfg).expect("failed to create res");
        assert!(res.success);
        // Best: items 0 (6,3) + 3 (3,1) + 2 (4,2) = 13, weight = 6 or items 0+1+3 = 14 w=7>6
        // items 0+2+3 = 13, w=6 or items 1+2+3=12, w=6
        assert!(res.obj <= -12.9, "obj={} should be <= -13", res.obj);
    }

    #[test]
    fn test_milp_infeasible() {
        // x in {0,1}, x >= 2 -> infeasible
        let c = array![1.0];
        let a = Array2::from_shape_vec((1, 1), vec![-1.0]).expect("failed to create a");
        let b = array![-2.0];
        let lb = array![0.0];
        let ub = array![1.0];
        let prob = MilpProblem::new(c, a, b, lb, ub, vec![0]).expect("failed to create prob");
        let cfg = BnbConfig::default();
        let res = branch_and_bound(&prob, &cfg).expect("failed to create res");
        assert!(!res.success, "should be infeasible");
    }

    #[test]
    fn test_milp_mixed_integer() {
        // min 2x + y; x integer, y continuous; x+y >= 2.5; x,y >= 0
        // optimal: x=0 (integer), y=2.5 -> obj=2.5
        let c = array![2.0, 1.0];
        let a = Array2::from_shape_vec((1, 2), vec![-1.0, -1.0]).expect("failed to create a");
        let b = array![-2.5];
        let lb = array![0.0, 0.0];
        let ub = array![10.0, 10.0];
        let prob = MilpProblem::new(c, a, b, lb, ub, vec![0]).expect("failed to create prob");
        let cfg = BnbConfig::default();
        let res = branch_and_bound(&prob, &cfg).expect("failed to create res");
        assert!(res.success);
        assert!(res.obj <= 3.0 + 1e-4, "obj={}", res.obj);
    }

    #[test]
    fn test_bnb_config_default() {
        let cfg = BnbConfig::default();
        assert_eq!(cfg.max_nodes, 50_000);
        assert_eq!(cfg.branching, BranchingStrategy::MostFractional);
    }

    #[test]
    fn test_milp_problem_new_error_dim() {
        let c = array![1.0, 2.0];
        let a = Array2::from_shape_vec((1, 3), vec![1.0, 1.0, 1.0]).expect("failed to create a"); // 3 cols != 2
        let b = array![5.0];
        let lb = array![0.0, 0.0];
        let ub = array![1.0, 1.0];
        let res = MilpProblem::new(c, a, b, lb, ub, vec![]);
        assert!(res.is_err());
    }
}
