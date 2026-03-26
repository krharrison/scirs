//! Core implicit differentiation engine for optimization layers.
//!
//! Given an optimal solution z* satisfying KKT conditions F(z*, θ) = 0,
//! the implicit function theorem yields:
//!
//!   dz*/dθ = -(∂F/∂z)⁻¹ · (∂F/∂θ)
//!
//! This module builds the KKT Jacobian ∂F/∂z, solves the resulting linear
//! system, and supports an active-set variant that restricts differentiation
//! to active inequality constraints.

use crate::error::{OptimizeError, OptimizeResult};

// ─────────────────────────────────────────────────────────────────────────────
// KKT Jacobian construction
// ─────────────────────────────────────────────────────────────────────────────

/// Build the KKT Jacobian matrix ∂F/∂z for the QP:
///
///   min  ½ x'Qx + c'x
///   s.t. Gx ≤ h   (m inequalities)
///        Ax = b    (p equalities)
///
/// The KKT conditions (with slacks absorbed into complementarity) are:
///
///   F₁ = Qx + c + G'diag(λ) + A'ν  = 0   (stationarity, n eqs)
///   F₂ = diag(λ)(Gx - h)           = 0   (complementarity, m eqs)
///   F₃ = Ax - b                      = 0   (primal equality, p eqs)
///
/// The Jacobian w.r.t. z = (x, λ, ν) is the (n+m+p) × (n+m+p) matrix:
///
///   ┌ Q          G'diag(λ)?   A' ┐
///   │ diag(λ)G   diag(Gx-h)  0  │
///   └ A          0            0  ┘
///
/// For the complementarity row the correct linearisation is:
///   ∂F₂/∂x = diag(λ) G
///   ∂F₂/∂λ = diag(Gx - h)   (= diag(s) where s = h - Gx ≥ 0 at optimum)
///
/// # Arguments
/// * `q` – n*n cost matrix (row-major `Vec<Vec<f64>>`).
/// * `g` – m×n inequality constraint matrix.
/// * `a` – p×n equality constraint matrix.
/// * `x` – optimal primal (length n).
/// * `lam` – optimal inequality duals (length m).
/// * `nu` – optimal equality duals (length p).
///
/// # Returns
/// The full KKT Jacobian as a flat (n+m+p) × (n+m+p) row-major matrix.
pub fn compute_kkt_jacobian(
    q: &[Vec<f64>],
    g: &[Vec<f64>],
    a: &[Vec<f64>],
    x: &[f64],
    lam: &[f64],
    nu: &[f64],
) -> Vec<Vec<f64>> {
    let n = x.len();
    let m = lam.len();
    let p = nu.len();
    let dim = n + m + p;

    let mut jac = vec![vec![0.0; dim]; dim];

    // ── Block (0,0): Q  (n×n) ──────────────────────────────────────────
    for i in 0..n {
        for j in 0..n {
            jac[i][j] = if i < q.len() && j < q[i].len() {
                q[i][j]
            } else {
                0.0
            };
        }
    }

    // ── Block (0,1): G' diag(λ)  →  actually for stationarity the
    //     derivative ∂F₁/∂λ_j = G_j (the j-th row of G, transposed).
    //     But we keep the simpler form: column j of block = G[j][:] (transpose).
    //     Stationarity is Qx + c + G'λ + A'ν = 0, so ∂/∂λ_j = G[j][:] transposed.
    for j in 0..m {
        for i in 0..n {
            let g_val = if j < g.len() && i < g[j].len() {
                g[j][i]
            } else {
                0.0
            };
            jac[i][n + j] = g_val;
        }
    }

    // ── Block (0,2): A'  (n×p) ─────────────────────────────────────────
    for j in 0..p {
        for i in 0..n {
            let a_val = if j < a.len() && i < a[j].len() {
                a[j][i]
            } else {
                0.0
            };
            jac[i][n + m + j] = a_val;
        }
    }

    // ── Block (1,0): diag(λ) G  (m×n) ─────────────────────────────────
    for i in 0..m {
        let li = lam[i];
        for j in 0..n {
            let g_val = if i < g.len() && j < g[i].len() {
                g[i][j]
            } else {
                0.0
            };
            jac[n + i][j] = li * g_val;
        }
    }

    // ── Block (1,1): diag(Gx - h)  (m×m) ──────────────────────────────
    // s_i = (Gx - h)_i
    for i in 0..m {
        let mut gx_i = 0.0;
        if i < g.len() {
            for j in 0..n.min(g[i].len()) {
                gx_i += g[i][j] * x[j];
            }
        }
        // Note: h is not passed here; it cancels with the complementarity form.
        // At optimality λ_i (Gx-h)_i = 0, but the Jacobian entry is (Gx-h)_i.
        // We store the slack as-is; caller must account for h offset.
        jac[n + i][n + i] = gx_i; // will be adjusted by caller with -h_i
    }

    // ── Block (2,0): A  (p×n) ──────────────────────────────────────────
    for i in 0..p {
        for j in 0..n {
            let a_val = if i < a.len() && j < a[i].len() {
                a[i][j]
            } else {
                0.0
            };
            jac[n + m + i][j] = a_val;
        }
    }

    // Blocks (1,2), (2,1), (2,2) are zero.
    jac
}

/// Adjust the complementarity diagonal block with the rhs h.
///
/// Call after `compute_kkt_jacobian` to set diag(Gx - h) correctly.
pub fn adjust_complementarity_diagonal(jac: &mut [Vec<f64>], h: &[f64], n: usize) {
    for (i, &h_i) in h.iter().enumerate() {
        jac[n + i][n + i] -= h_i;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear system solver (Gaussian elimination with partial pivoting)
// ─────────────────────────────────────────────────────────────────────────────

/// Solve the linear system `A x = rhs` via Gaussian elimination with partial
/// pivoting.
///
/// Both `mat` (square, row-major) and `rhs` are consumed / mutated.
///
/// # Errors
/// Returns `OptimizeError::ComputationError` if the matrix is singular.
pub fn solve_implicit_system(mat: &[Vec<f64>], rhs: &[f64]) -> OptimizeResult<Vec<f64>> {
    let n = rhs.len();
    if mat.len() != n {
        return Err(OptimizeError::InvalidInput(format!(
            "KKT matrix rows ({}) != rhs length ({})",
            mat.len(),
            n
        )));
    }

    // Build augmented matrix
    let mut aug: Vec<Vec<f64>> = mat
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.push(rhs[i]);
            r
        })
        .collect();

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = aug[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_val < 1e-30 {
            return Err(OptimizeError::ComputationError(
                "Singular KKT matrix in implicit differentiation".to_string(),
            ));
        }

        if max_row != col {
            aug.swap(col, max_row);
        }

        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for j in col..=n {
                let val = aug[col][j];
                aug[row][j] -= factor * val;
            }
        }
    }

    // Back substitution
    let mut solution = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * solution[j];
        }
        let diag = aug[i][i];
        if diag.abs() < 1e-30 {
            return Err(OptimizeError::ComputationError(
                "Zero diagonal in back substitution".to_string(),
            ));
        }
        solution[i] = sum / diag;
    }

    Ok(solution)
}

/// Solve the system `mat * X = rhs_matrix` where rhs_matrix has `k` columns.
/// Returns the solution matrix (n × k) in row-major form.
pub fn solve_implicit_system_multi(
    mat: &[Vec<f64>],
    rhs_cols: &[Vec<f64>],
) -> OptimizeResult<Vec<Vec<f64>>> {
    rhs_cols
        .iter()
        .map(|rhs| solve_implicit_system(mat, rhs))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Active constraint identification
// ─────────────────────────────────────────────────────────────────────────────

/// Identify active inequality constraints: those where h_i - G_i x ≤ tol
/// (i.e., the slack is near zero).
///
/// Returns indices of the active constraints.
pub fn identify_active_constraints(g: &[Vec<f64>], h: &[f64], x: &[f64], tol: f64) -> Vec<usize> {
    let m = h.len();
    let n = x.len();
    let mut active = Vec::new();

    for i in 0..m {
        let mut gx_i = 0.0;
        if i < g.len() {
            for j in 0..n.min(g[i].len()) {
                gx_i += g[i][j] * x[j];
            }
        }
        let slack = h[i] - gx_i; // slack ≥ 0 at feasibility
        if slack.abs() <= tol {
            active.push(i);
        }
    }

    active
}

/// Extract rows from G and h corresponding to the given active indices.
pub fn extract_active_constraints(
    g: &[Vec<f64>],
    h: &[f64],
    active: &[usize],
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let g_active: Vec<Vec<f64>> = active.iter().filter_map(|&i| g.get(i).cloned()).collect();
    let h_active: Vec<f64> = active.iter().filter_map(|&i| h.get(i).copied()).collect();
    (g_active, h_active)
}

// ─────────────────────────────────────────────────────────────────────────────
// Full implicit backward pass
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the full implicit gradient dz*/dθ for a QP.
///
/// Given dl/dx (the upstream gradient from the loss), compute dl/dθ for
/// θ = (Q, c, G, h, A, b) by solving:
///
///   (∂F/∂z)' dz = -[dl/dx, 0, 0]'
///
/// and then computing dl/dθ = (∂F/∂θ)' dz.
///
/// # Arguments
/// * `q` – n×n cost matrix.
/// * `g` – m×n inequality constraint matrix.
/// * `h` – m inequality rhs.
/// * `a` – p×n equality constraint matrix.
/// * `x` – optimal primal solution.
/// * `lam` – optimal inequality duals.
/// * `nu` – optimal equality duals.
/// * `dl_dx` – upstream gradient dl/dx (length n).
///
/// # Returns
/// The implicit gradients for all parameters.
pub fn compute_full_implicit_gradient(
    q: &[Vec<f64>],
    g: &[Vec<f64>],
    h: &[f64],
    a: &[Vec<f64>],
    x: &[f64],
    lam: &[f64],
    nu: &[f64],
    dl_dx: &[f64],
) -> OptimizeResult<super::types::ImplicitGradient> {
    let n = x.len();
    let m = lam.len();
    let p = nu.len();
    let dim = n + m + p;

    // Build the KKT Jacobian
    let mut kkt = compute_kkt_jacobian(q, g, a, x, lam, nu);
    adjust_complementarity_diagonal(&mut kkt, h, n);

    // Transpose the KKT Jacobian (we solve the adjoint system)
    let mut kkt_t = vec![vec![0.0; dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            kkt_t[i][j] = kkt[j][i];
        }
    }

    // RHS = -[dl/dx, 0_m, 0_p]
    let mut rhs = vec![0.0; dim];
    for i in 0..n {
        rhs[i] = -dl_dx[i];
    }

    // Solve: kkt_t * dz = rhs
    let dz = solve_implicit_system(&kkt_t, &rhs)?;

    // Extract components: dz = (dx, dlam, dnu)
    let dx = &dz[..n];
    let dlam = &dz[n..n + m];
    let dnu = &dz[n + m..];

    // ── Compute dl/dθ from dz ──────────────────────────────────────────
    // dl/dc = dx  (from ∂F₁/∂c = I)
    let dl_dc = dx.to_vec();

    // dl/dh = -dlam  (from ∂F₂/∂h = -diag(λ), contracted with dlam gives -λ·dlam,
    //  but more directly: ∂F₂/∂h_i = -λ_i, so dl/dh_i = -λ_i * dlam_i / λ_i = -dlam_i
    //  when λ_i ≠ 0, and 0 otherwise; we use dlam directly.)
    let dl_dh: Vec<f64> = dlam.iter().map(|&v| -v).collect();

    // dl/db = -dnu  (from ∂F₃/∂b = -I)
    let dl_db: Vec<f64> = dnu.iter().map(|&v| -v).collect();

    // dl/dQ = dx * x' (outer product, symmetric part)
    let mut dl_dq = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            // ∂F₁/∂Q_{ij} · x_j, contracted with dx_i
            // = 0.5 * (dx_i * x_j + dx_j * x_i)  for symmetric Q
            dl_dq[i][j] = 0.5 * (dx[i] * x[j] + dx[j] * x[i]);
        }
    }

    // dl/dG: from stationarity (G'λ term) and complementarity (diag(λ)G term)
    let mut dl_dg = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            // From stationarity: ∂F₁_j / ∂G_{ij} = λ_i, contracted with dx_j
            // From complementarity: ∂F₂_i / ∂G_{ij} = λ_i * x_j, contracted with dlam_i
            dl_dg[i][j] = dx[j] * lam[i] + dlam[i] * lam[i] * x[j];
        }
    }

    // dl/dA: from stationarity (A'ν) and primal equality (Ax-b)
    let mut dl_da = vec![vec![0.0; n]; p];
    for i in 0..p {
        for j in 0..n {
            // From stationarity: ∂F₁_j / ∂A_{ij} = ν_i, contracted with dx_j
            // From primal eq: ∂F₃_i / ∂A_{ij} = x_j, contracted with dnu_i
            dl_da[i][j] = dx[j] * nu[i] + dnu[i] * x[j];
        }
    }

    Ok(super::types::ImplicitGradient {
        dl_dq: Some(dl_dq),
        dl_dc,
        dl_dg: Some(dl_dg),
        dl_dh,
        dl_da: if p > 0 { Some(dl_da) } else { None },
        dl_db,
    })
}

/// Compute implicit gradient using only active constraints.
///
/// This is faster than full differentiation when many inequality constraints
/// are inactive, since those constraints have zero dual variables and do not
/// contribute to the gradient.
pub fn compute_active_set_implicit_gradient(
    q: &[Vec<f64>],
    g: &[Vec<f64>],
    h: &[f64],
    a: &[Vec<f64>],
    x: &[f64],
    lam: &[f64],
    nu: &[f64],
    dl_dx: &[f64],
    active_tol: f64,
) -> OptimizeResult<super::types::ImplicitGradient> {
    let m = lam.len();

    // Identify active constraints
    let active = identify_active_constraints(g, h, x, active_tol);
    let (g_active, h_active) = extract_active_constraints(g, h, &active);
    let lam_active: Vec<f64> = active
        .iter()
        .filter_map(|&i| if i < m { Some(lam[i]) } else { None })
        .collect();

    // Solve reduced system
    let grad =
        compute_full_implicit_gradient(q, &g_active, &h_active, a, x, &lam_active, nu, dl_dx)?;

    // Expand gradient back to full dimension
    let m_full = lam.len();
    let n = x.len();

    let mut dl_dh_full = vec![0.0; m_full];
    for (idx, &ai) in active.iter().enumerate() {
        if ai < m_full && idx < grad.dl_dh.len() {
            dl_dh_full[ai] = grad.dl_dh[idx];
        }
    }

    let dl_dg_full = if let Some(ref dg) = grad.dl_dg {
        let mut full = vec![vec![0.0; n]; m_full];
        for (idx, &ai) in active.iter().enumerate() {
            if ai < m_full && idx < dg.len() {
                full[ai] = dg[idx].clone();
            }
        }
        Some(full)
    } else {
        None
    };

    Ok(super::types::ImplicitGradient {
        dl_dq: grad.dl_dq,
        dl_dc: grad.dl_dc,
        dl_dg: dl_dg_full,
        dl_dh: dl_dh_full,
        dl_da: grad.dl_da,
        dl_db: grad.dl_db,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kkt_jacobian_structure_2x2() {
        // Simple 2-var QP with 1 inequality, 0 equalities
        // Q = [[2, 0], [0, 2]], G = [[1, 1]], h = [1]
        let q = vec![vec![2.0, 0.0], vec![0.0, 2.0]];
        let g = vec![vec![1.0, 1.0]];
        let a: Vec<Vec<f64>> = vec![];
        let x = vec![0.25, 0.25];
        let lam = vec![0.5];
        let nu: Vec<f64> = vec![];

        let jac = compute_kkt_jacobian(&q, &g, &a, &x, &lam, &nu);

        // Dimension: 2 + 1 + 0 = 3
        assert_eq!(jac.len(), 3);
        assert_eq!(jac[0].len(), 3);

        // Block (0,0): Q
        assert!((jac[0][0] - 2.0).abs() < 1e-12);
        assert!((jac[1][1] - 2.0).abs() < 1e-12);

        // Block (0,1): G' (transposed)
        assert!((jac[0][2] - 1.0).abs() < 1e-12);
        assert!((jac[1][2] - 1.0).abs() < 1e-12);

        // Block (1,0): diag(λ) G
        assert!((jac[2][0] - 0.5).abs() < 1e-12);
        assert!((jac[2][1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_active_constraint_identification() {
        let g = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let h = vec![1.0, 2.0, 1.5];
        let x = vec![1.0, 0.5]; // Gx = [1.0, 0.5, 1.5], slacks = [0.0, 1.5, 0.0]

        let active = identify_active_constraints(&g, &h, &x, 1e-6);
        assert_eq!(active, vec![0, 2]);
    }

    #[test]
    fn test_solve_implicit_system_simple() {
        // Solve [[2, 1], [1, 3]] x = [5, 7]  →  x = [8/5, 9/5]
        let mat = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let rhs = vec![5.0, 7.0];
        let sol = solve_implicit_system(&mat, &rhs).expect("solve failed");
        assert!((sol[0] - 1.6).abs() < 1e-10);
        assert!((sol[1] - 1.8).abs() < 1e-10);
    }

    #[test]
    fn test_solve_singular_matrix() {
        let mat = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let rhs = vec![3.0, 6.0];
        let result = solve_implicit_system(&mat, &rhs);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_active_constraints() {
        let g = vec![vec![1.0], vec![2.0], vec![3.0]];
        let h = vec![10.0, 20.0, 30.0];
        let active = vec![0, 2];

        let (ga, ha) = extract_active_constraints(&g, &h, &active);
        assert_eq!(ga.len(), 2);
        assert!((ga[0][0] - 1.0).abs() < 1e-12);
        assert!((ga[1][0] - 3.0).abs() < 1e-12);
        assert!((ha[0] - 10.0).abs() < 1e-12);
        assert!((ha[1] - 30.0).abs() < 1e-12);
    }

    #[test]
    fn test_empty_constraints() {
        let q = vec![vec![2.0, 0.0], vec![0.0, 2.0]];
        let g: Vec<Vec<f64>> = vec![];
        let a: Vec<Vec<f64>> = vec![];
        let x = vec![1.0, 2.0];
        let lam: Vec<f64> = vec![];
        let nu: Vec<f64> = vec![];

        let jac = compute_kkt_jacobian(&q, &g, &a, &x, &lam, &nu);
        assert_eq!(jac.len(), 2);
        assert!((jac[0][0] - 2.0).abs() < 1e-12);
    }
}
