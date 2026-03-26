//! LP relaxation solver for use in branch-and-bound
//!
//! Solves linear programs using a simplex method with big-M phase I.
//! This is an internal solver used by the branch-and-bound and cutting plane methods.

use super::LinearProgram;
use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::Array1;

/// Result from LP solve
#[derive(Debug, Clone)]
pub struct LpResult {
    /// Optimal solution
    pub x: Array1<f64>,
    /// Optimal objective value
    pub fun: f64,
    /// Whether LP is feasible and bounded
    pub success: bool,
    /// Status: 0=optimal, 1=infeasible, 2=unbounded
    pub status: i32,
}

/// Simple LP relaxation solver
pub struct LpRelaxationSolver;

impl LpRelaxationSolver {
    /// Solve a linear program using a simplex-based method.
    ///
    /// Handles:
    ///   min c^T x
    ///   s.t. A_ub x <= b_ub  (inequality constraints)
    ///        A_eq x  = b_eq  (equality constraints)
    ///        lb <= x <= ub
    pub fn solve(
        lp: &LinearProgram,
        extra_lb: &[f64],
        extra_ub: &[f64],
    ) -> OptimizeResult<LpResult> {
        let n = lp.n_vars();
        if n == 0 {
            return Err(OptimizeError::InvalidInput("Empty LP".to_string()));
        }

        // Combine bounds
        let lb: Vec<f64> = (0..n)
            .map(|i| {
                let base = lp.lower.as_ref().map_or(0.0, |l| l[i]);
                if i < extra_lb.len() {
                    base.max(extra_lb[i])
                } else {
                    base
                }
            })
            .collect();

        let ub: Vec<f64> = (0..n)
            .map(|i| {
                let base = lp.upper.as_ref().map_or(f64::INFINITY, |u| u[i]);
                if i < extra_ub.len() {
                    base.min(extra_ub[i])
                } else {
                    base
                }
            })
            .collect();

        // Check bound feasibility
        for i in 0..n {
            if lb[i] > ub[i] + 1e-10 {
                return Ok(LpResult {
                    x: Array1::zeros(n),
                    fun: f64::INFINITY,
                    success: false,
                    status: 1,
                });
            }
        }

        // Gather inequality constraints
        let (a_ub_rows, b_ub_vec) = match (&lp.a_ub, &lp.b_ub) {
            (Some(a), Some(b)) => {
                let m = a.nrows();
                let rows: Vec<Vec<f64>> = (0..m)
                    .map(|i| (0..n).map(|j| a[[i, j]]).collect())
                    .collect();
                let bv: Vec<f64> = b.to_vec();
                (rows, bv)
            }
            _ => (vec![], vec![]),
        };

        // Gather equality constraints
        let (a_eq_rows, b_eq_vec) = match (&lp.a_eq, &lp.b_eq) {
            (Some(a), Some(b)) => {
                let m = a.nrows();
                let rows: Vec<Vec<f64>> = (0..m)
                    .map(|i| (0..n).map(|j| a[[i, j]]).collect())
                    .collect();
                let bv: Vec<f64> = b.to_vec();
                (rows, bv)
            }
            _ => (vec![], vec![]),
        };

        let c_vec: Vec<f64> = lp.c.to_vec();

        let sol = solve_lp_simplex(
            &c_vec, &a_ub_rows, &b_ub_vec, &a_eq_rows, &b_eq_vec, &lb, &ub,
        );

        Ok(sol)
    }
}

/// Internal LP solver using simplex with big-M method.
///
/// min c^T x  s.t.  A_ub x <= b_ub, A_eq x = b_eq, lb <= x <= ub
fn solve_lp_simplex(
    c: &[f64],
    a_ub: &[Vec<f64>],
    b_ub: &[f64],
    a_eq: &[Vec<f64>],
    b_eq: &[f64],
    lb: &[f64],
    ub: &[f64],
) -> LpResult {
    let n = c.len();
    let m_ub = a_ub.len();
    let m_eq = a_eq.len();

    if n == 0 {
        return LpResult {
            x: Array1::zeros(0),
            fun: 0.0,
            success: true,
            status: 0,
        };
    }

    // --- Variable shift: y = x - lb, so y >= 0, y <= ub - lb -----------
    let ub_shifted: Vec<f64> = (0..n)
        .map(|i| {
            if ub[i].is_finite() {
                ub[i] - lb[i]
            } else {
                f64::INFINITY
            }
        })
        .collect();

    // Shift RHS of inequality constraints: A(y+lb) <= b => Ay <= b - A*lb
    let b_ub_shifted: Vec<f64> = (0..m_ub)
        .map(|i| {
            b_ub[i]
                - a_ub[i]
                    .iter()
                    .zip(lb.iter())
                    .map(|(&a, &l)| a * l)
                    .sum::<f64>()
        })
        .collect();

    // Shift RHS of equality constraints: A_eq(y+lb) = b_eq => A_eq y = b_eq - A_eq*lb
    let b_eq_shifted: Vec<f64> = (0..m_eq)
        .map(|i| {
            b_eq[i]
                - a_eq[i]
                    .iter()
                    .zip(lb.iter())
                    .map(|(&a, &l)| a * l)
                    .sum::<f64>()
        })
        .collect();

    // --- Count finite upper bounds for UB slack variables ----------------
    let n_ub_constrained: usize = ub_shifted.iter().filter(|&&u| u.is_finite()).count();

    // Variables: y_0..y_{n-1}, ineq_slack s_0..s_{m_ub-1}, ub_slack t_0..t_{n_ub-1}
    let n_struct = n;
    let n_ineq_slack = m_ub;
    let n_ub_slack = n_ub_constrained;
    let n_total = n_struct + n_ineq_slack + n_ub_slack;

    // Map: structural variable j -> UB slack column index
    let mut ub_slack_col: Vec<Option<usize>> = vec![None; n];
    let mut ub_col_idx = n_struct + n_ineq_slack;
    for j in 0..n {
        if ub_shifted[j].is_finite() {
            ub_slack_col[j] = Some(ub_col_idx);
            ub_col_idx += 1;
        }
    }

    // Total constraint rows: m_ub (inequality) + n_ub (UB) + m_eq (equality)
    let total_rows = m_ub + n_ub_constrained + m_eq;

    // Build full constraint matrix and RHS
    let mut full_a: Vec<Vec<f64>> = vec![vec![0.0; n_total]; total_rows];
    let mut full_b: Vec<f64> = vec![0.0; total_rows];

    // Fill inequality rows: A y + s = b_shifted  (s >= 0)
    for i in 0..m_ub {
        for j in 0..n {
            full_a[i][j] = a_ub[i][j];
        }
        full_a[i][n_struct + i] = 1.0; // slack
        full_b[i] = b_ub_shifted[i];
    }

    // Fill UB bound rows: y_j + t_j = ub_shifted[j]  (t_j >= 0)
    let mut ub_row = m_ub;
    for j in 0..n {
        if let Some(sk_col) = ub_slack_col[j] {
            full_a[ub_row][j] = 1.0;
            full_a[ub_row][sk_col] = 1.0;
            full_b[ub_row] = ub_shifted[j];
            ub_row += 1;
        }
    }

    // Fill equality rows: A_eq y = b_eq_shifted (no slacks, need artificials)
    for i in 0..m_eq {
        let row_idx = m_ub + n_ub_constrained + i;
        for j in 0..n {
            full_a[row_idx][j] = a_eq[i][j];
        }
        full_b[row_idx] = b_eq_shifted[i];
    }

    // --- Handle negative RHS by flipping rows + adding artificials -------
    let mut needs_artif = vec![false; total_rows];
    for i in 0..total_rows {
        if full_b[i] < -1e-12 {
            // Negate the row
            for v in full_a[i].iter_mut() {
                *v = -*v;
            }
            full_b[i] = -full_b[i];
            needs_artif[i] = true;
        }
    }
    // Equality constraint rows always need artificials (no slack in basis)
    for i in 0..m_eq {
        let row_idx = m_ub + n_ub_constrained + i;
        needs_artif[row_idx] = true;
    }

    let n_artif: usize = needs_artif.iter().filter(|&&v| v).count();
    let n_total_ext = n_total + n_artif;

    // Build artificial column map: row -> artificial column index
    let mut artif_col_map: Vec<Option<usize>> = vec![None; total_rows];
    let mut acol = n_total;
    for i in 0..total_rows {
        if needs_artif[i] {
            artif_col_map[i] = Some(acol);
            acol += 1;
        }
    }

    // Extend rows with artificial columns
    for row in full_a.iter_mut() {
        row.resize(n_total_ext, 0.0);
    }
    for i in 0..total_rows {
        if let Some(ac) = artif_col_map[i] {
            full_a[i][ac] = 1.0;
        }
    }

    // Objective: original c for structural vars, 0 for slacks, big-M for artificials
    let big_m = 1e7_f64;
    let mut obj_c: Vec<f64> = vec![0.0; n_total_ext];
    for j in 0..n {
        obj_c[j] = c[j];
    }
    for i in 0..total_rows {
        if let Some(ac) = artif_col_map[i] {
            obj_c[ac] = big_m;
        }
    }

    // Initial basis
    let mut basis: Vec<usize> = Vec::with_capacity(total_rows);
    // Inequality rows: use slack (or artificial if flipped)
    for i in 0..m_ub {
        if needs_artif[i] {
            basis.push(artif_col_map[i].unwrap_or(0));
        } else {
            basis.push(n_struct + i); // ineq slack
        }
    }
    // UB rows: use UB slack (or artificial if flipped -- rare for UB rows)
    let mut ub_row_counter = 0usize;
    for j in 0..n {
        if let Some(sk_col) = ub_slack_col[j] {
            let row_idx = m_ub + ub_row_counter;
            if needs_artif[row_idx] {
                basis.push(artif_col_map[row_idx].unwrap_or(0));
            } else {
                basis.push(sk_col);
            }
            ub_row_counter += 1;
        }
    }
    // Equality rows: always use artificial
    for i in 0..m_eq {
        let row_idx = m_ub + n_ub_constrained + i;
        basis.push(artif_col_map[row_idx].unwrap_or(0));
    }

    // Run simplex
    let simplex_result = run_simplex(&mut full_a, &mut full_b, &obj_c, &mut basis, 20_000);

    if simplex_result == SimplexStatus::Unbounded {
        return LpResult {
            x: Array1::zeros(n),
            fun: f64::NEG_INFINITY,
            success: false,
            status: 2,
        };
    }

    // Check that no artificial is in the basis with positive value
    for (i, &bv) in basis.iter().enumerate() {
        if bv >= n_total && full_b[i] > 1e-6 {
            return LpResult {
                x: Array1::zeros(n),
                fun: f64::INFINITY,
                success: false,
                status: 1,
            };
        }
    }

    // Extract y values (shifted structural variables)
    let mut y = vec![0.0_f64; n];
    for (i, &bv) in basis.iter().enumerate() {
        if bv < n {
            y[bv] = full_b[i].max(0.0);
        }
    }

    // Shift back: x = y + lb, clamped to [lb, ub]
    let x: Vec<f64> = (0..n)
        .map(|j| (lb[j] + y[j]).max(lb[j]).min(ub[j]))
        .collect();

    let fun: f64 = c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum();

    LpResult {
        x: Array1::from_vec(x),
        fun,
        success: true,
        status: 0,
    }
}

#[derive(Debug, PartialEq)]
enum SimplexStatus {
    Optimal,
    Unbounded,
    MaxIter,
}

/// Run the simplex method on the tableau.
///
/// The system is `A x = b`, `x >= 0`, minimize `c^T x`.
/// `basis` holds the indices of the initial basic variables.
/// On return, `a` and `b` are updated in place (tableau form), and `basis` holds the final basis.
fn run_simplex(
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
    c: &[f64],
    basis: &mut Vec<usize>,
    max_iter: usize,
) -> SimplexStatus {
    let m = a.len();
    if m == 0 {
        return SimplexStatus::Optimal;
    }
    let n_total = a[0].len();

    // Build full tableau: [A | b]
    let mut tableau: Vec<Vec<f64>> = vec![vec![0.0; n_total + 1]; m];
    for i in 0..m {
        for j in 0..n_total {
            tableau[i][j] = a[i][j];
        }
        tableau[i][n_total] = b[i];
    }

    // Make basis columns identity via row operations
    for col in 0..m {
        let basic = basis[col];
        let pivot_val = tableau[col][basic];
        if pivot_val.abs() < 1e-12 {
            // Try to find a different row for this basis element
            let mut found = false;
            for i in 0..m {
                if i != col && tableau[i][basic].abs() > 1e-10 {
                    tableau.swap(col, i);
                    basis.swap(col, i);
                    found = true;
                    break;
                }
            }
            if !found {
                continue;
            }
        }
        let pv = tableau[col][basic];
        if pv.abs() < 1e-12 {
            continue;
        }
        for j in 0..=n_total {
            tableau[col][j] /= pv;
        }
        for i in 0..m {
            if i == col {
                continue;
            }
            let factor = tableau[i][basic];
            if factor.abs() < 1e-15 {
                continue;
            }
            for j in 0..=n_total {
                let delta = factor * tableau[col][j];
                tableau[i][j] -= delta;
            }
        }
    }

    // Simplex iterations
    let mut status = SimplexStatus::MaxIter;
    for _iter in 0..max_iter {
        // Compute reduced costs using current basis
        let c_b: Vec<f64> = basis
            .iter()
            .map(|&bv| c.get(bv).copied().unwrap_or(0.0))
            .collect();

        // Find entering variable (most negative reduced cost)
        let mut enter = None;
        let mut min_rc = -1e-8_f64;
        for j in 0..n_total {
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
            None => {
                status = SimplexStatus::Optimal;
                break;
            }
            Some(j) => j,
        };

        // Minimum ratio test
        let mut leave_row = None;
        let mut min_ratio = f64::INFINITY;
        for i in 0..m {
            let coef = tableau[i][enter_col];
            if coef > 1e-10 {
                let ratio = tableau[i][n_total] / coef;
                if ratio < min_ratio - 1e-12 {
                    min_ratio = ratio;
                    leave_row = Some(i);
                } else if (ratio - min_ratio).abs() < 1e-12 {
                    // Bland's rule: prefer smaller index
                    if let Some(prev) = leave_row {
                        if basis[i] < basis[prev] {
                            leave_row = Some(i);
                        }
                    }
                }
            }
        }

        let pivot_row = match leave_row {
            None => {
                status = SimplexStatus::Unbounded;
                break;
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

    // Write back modified b and a
    for i in 0..m {
        b[i] = tableau[i][n_total];
        for j in 0..n_total {
            a[i][j] = tableau[i][j];
        }
    }

    status
}
