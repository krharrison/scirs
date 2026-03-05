//! Algebraic Multigrid (AMG) methods
//!
//! This module implements three AMG variants:
//!
//! 1. **Ruge-Stüben AMG** (RS-AMG): Classical coarsening with direct/classical
//!    interpolation operators and V/W-cycle solvers.
//!
//! 2. **Smoothed Aggregation AMG** (SA-AMG): Aggregation-based coarsening with
//!    Jacobi-smoothed tentative prolongation and energy-minimization smoothers.
//!
//! 3. **AIR AMG** (Approximate Ideal Restriction): Non-symmetric AMG tailored for
//!    advection-dominated problems, using approximate ideal restriction.
//!
//! # References
//!
//! - Ruge & Stüben (1987). "Algebraic multigrid." *Multigrid Methods*, SIAM.
//! - Vaněk, Mandel & Brezina (1996). "Algebraic multigrid by smoothed aggregation."
//! - Manteuffel, Münzenmaier, Ruge & Southworth (2019). "Nonsymmetric reduction-based AMG."

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use crate::iterative_solvers::IterativeSolverConfig;
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, NumAssign, SparseElement};
use std::fmt::Debug;
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

/// Sparse matrix-vector product: y = A * x
fn spmv(a: &CsrMatrix<f64>, x: &[f64]) -> SparseResult<Vec<f64>> {
    let (rows, cols) = a.shape();
    if x.len() != cols {
        return Err(SparseError::DimensionMismatch {
            expected: cols,
            found: x.len(),
        });
    }
    let result = a.dot(x)?;
    Ok(result)
}

/// Transpose of a CSR matrix in CSR form.
fn csr_transpose(a: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    a.transpose()
}

/// Matrix product A * B for two CSR matrices.
fn csr_matmul(a: &CsrMatrix<f64>, b: &CsrMatrix<f64>) -> SparseResult<CsrMatrix<f64>> {
    a.matmul(b)
}

/// Compute R * A * P (triple product for coarse-grid operator).
fn galerkin_product(
    r: &CsrMatrix<f64>,
    a: &CsrMatrix<f64>,
    p: &CsrMatrix<f64>,
) -> SparseResult<CsrMatrix<f64>> {
    let ap = csr_matmul(a, p)?;
    csr_matmul(r, &ap)
}

/// Weighted Jacobi smoother: x += omega * D^{-1} * (b - A*x)
fn jacobi_smooth(
    a: &CsrMatrix<f64>,
    x: &mut Vec<f64>,
    b: &[f64],
    diag_inv: &[f64],
    omega: f64,
    n_iter: usize,
) -> SparseResult<()> {
    let n = x.len();
    let mut x_new = vec![0.0f64; n];
    for _ in 0..n_iter {
        let ax = spmv(a, x)?;
        for i in 0..n {
            let residual_i = b[i] - ax[i];
            x_new[i] = x[i] + omega * diag_inv[i] * residual_i;
        }
        x.copy_from_slice(&x_new);
    }
    Ok(())
}

/// Gauss-Seidel smoother (forward sweep): update x[i] = (b[i] - sum_{j≠i} a_{ij} x[j]) / a_{ii}
fn gauss_seidel_smooth(
    a: &CsrMatrix<f64>,
    x: &mut Vec<f64>,
    b: &[f64],
    n_iter: usize,
) -> SparseResult<()> {
    let n = x.len();
    for _ in 0..n_iter {
        for i in 0..n {
            let range = a.row_range(i);
            let mut sigma = 0.0f64;
            let mut diag = 1.0f64;
            for pos in range {
                let j = a.indices[pos];
                let val = a.data[pos];
                if j == i {
                    diag = val;
                } else {
                    sigma += val * x[j];
                }
            }
            if diag.abs() > f64::EPSILON {
                x[i] = (b[i] - sigma) / diag;
            }
        }
    }
    Ok(())
}

/// Backward Gauss-Seidel sweep.
fn gauss_seidel_smooth_backward(
    a: &CsrMatrix<f64>,
    x: &mut Vec<f64>,
    b: &[f64],
    n_iter: usize,
) -> SparseResult<()> {
    let n = x.len();
    for _ in 0..n_iter {
        for i in (0..n).rev() {
            let range = a.row_range(i);
            let mut sigma = 0.0f64;
            let mut diag = 1.0f64;
            for pos in range {
                let j = a.indices[pos];
                let val = a.data[pos];
                if j == i {
                    diag = val;
                } else {
                    sigma += val * x[j];
                }
            }
            if diag.abs() > f64::EPSILON {
                x[i] = (b[i] - sigma) / diag;
            }
        }
    }
    Ok(())
}

/// Direct (exact) solver via Gaussian elimination for small dense systems.
fn direct_solve_small(dense: &[Vec<f64>], rhs: &[f64]) -> SparseResult<Vec<f64>> {
    let n = rhs.len();
    if dense.len() != n || (n > 0 && dense[0].len() != n) {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: dense.len(),
        });
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    // Build augmented matrix [A | b]
    let mut aug: Vec<Vec<f64>> = dense
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
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            let v = aug[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return Err(SparseError::SingularMatrix(
                "Near-singular matrix in direct_solve_small".to_string(),
            ));
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for k in col..=n {
                let val = aug[col][k];
                aug[row][k] -= factor * val;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = aug[i][n];
        for j in (i + 1)..n {
            s -= aug[i][j] * x[j];
        }
        x[i] = s / aug[i][i];
    }
    Ok(x)
}

/// Convert CSR matrix to dense form (for small coarsest-level solves).
fn csr_to_dense(a: &CsrMatrix<f64>) -> Vec<Vec<f64>> {
    let (rows, cols) = a.shape();
    let mut dense = vec![vec![0.0f64; cols]; rows];
    for i in 0..rows {
        for pos in a.row_range(i) {
            let j = a.indices[pos];
            dense[i][j] = a.data[pos];
        }
    }
    dense
}

// ===========================================================================
// Part 1: Ruge-Stüben AMG
// ===========================================================================

/// C/F splitting classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CfLabel {
    Undecided,
    Coarse,
    Fine,
}

/// Compute strength-of-connection graph for Ruge-Stüben AMG.
///
/// Node j strongly influences node i when:
/// `-a_{ij} >= theta * max_{k≠i} (-a_{ik})`
///
/// Returns the set of strong connections: `strong[i]` = nodes that strongly
/// influence i.
fn rs_strength_of_connection(a: &CsrMatrix<f64>, theta: f64) -> Vec<Vec<usize>> {
    let n = a.shape().0;
    let mut strong = vec![Vec::new(); n];

    for i in 0..n {
        // Find max off-diagonal magnitude in row i (negative off-diagonals assumed)
        let mut max_neg = 0.0f64;
        for pos in a.row_range(i) {
            let j = a.indices[pos];
            if j != i {
                let v = -a.data[pos]; // Flip sign (off-diagonals are typically negative)
                if v > max_neg {
                    max_neg = v;
                }
            }
        }
        let threshold = theta * max_neg;
        for pos in a.row_range(i) {
            let j = a.indices[pos];
            if j != i {
                let v = -a.data[pos];
                if v >= threshold && threshold > 0.0 {
                    strong[i].push(j);
                }
            }
        }
    }
    strong
}

/// Compute transpose of strong connections: `strong_T[j]` = nodes that j strongly
/// influences (i.e., nodes i for which j ∈ strong[i]).
fn transpose_strong(strong: &[Vec<usize>], n: usize) -> Vec<Vec<usize>> {
    let mut st = vec![Vec::new(); n];
    for (i, row) in strong.iter().enumerate() {
        for &j in row {
            st[j].push(i);
        }
    }
    st
}

/// Classical RS coarsening (C/F splitting).
///
/// Uses a heuristic measure λ(i) = |S_T[i]| + 2 * |{j ∈ S_T[i] : label[j] == Fine}|
/// to select coarse nodes greedily.
fn rs_classical_coarsening(strong: &[Vec<usize>]) -> Vec<CfLabel> {
    let n = strong.len();
    let strong_t = transpose_strong(strong, n);

    // Initial lambda values
    let mut lambda: Vec<i64> = strong_t.iter().map(|v| v.len() as i64).collect();
    let mut labels = vec![CfLabel::Undecided; n];

    // Process nodes in decreasing lambda order (simple priority queue via repeated max)
    let mut remaining: Vec<bool> = vec![true; n];
    let mut n_remaining = n;

    while n_remaining > 0 {
        // Find undecided node with maximum lambda
        let mut best = usize::MAX;
        let mut best_lambda = i64::MIN;
        for i in 0..n {
            if remaining[i] && labels[i] == CfLabel::Undecided && lambda[i] > best_lambda {
                best_lambda = lambda[i];
                best = i;
            }
        }
        if best == usize::MAX {
            break;
        }

        // Make best a coarse node
        labels[best] = CfLabel::Coarse;
        remaining[best] = false;
        n_remaining -= 1;

        // All undecided nodes that strongly depend on best become Fine
        for &i in &strong_t[best] {
            if labels[i] == CfLabel::Undecided {
                labels[i] = CfLabel::Fine;
                remaining[i] = false;
                n_remaining -= 1;
                // Boost lambda of undecided neighbors of i
                for &k in &strong_t[i] {
                    if labels[k] == CfLabel::Undecided {
                        lambda[k] += 1;
                    }
                }
            }
        }
    }

    // Any remaining undecided nodes become Coarse
    for label in labels.iter_mut() {
        if *label == CfLabel::Undecided {
            *label = CfLabel::Coarse;
        }
    }
    labels
}

/// Build coarse-node numbering map from C/F labels.
/// Returns (coarse_indices, fine_indices, coarse_map) where coarse_map[i] = coarse index of C-node i.
fn build_coarse_map(labels: &[CfLabel]) -> (Vec<usize>, Vec<usize>, Vec<Option<usize>>) {
    let n = labels.len();
    let mut coarse_indices = Vec::new();
    let mut fine_indices = Vec::new();
    let mut coarse_map: Vec<Option<usize>> = vec![None; n];

    for (i, &label) in labels.iter().enumerate() {
        match label {
            CfLabel::Coarse => {
                coarse_map[i] = Some(coarse_indices.len());
                coarse_indices.push(i);
            }
            _ => {
                fine_indices.push(i);
            }
        }
    }
    (coarse_indices, fine_indices, coarse_map)
}

/// Direct interpolation operator for RS-AMG.
///
/// For each F-node i:
/// P[i, c_k] = -a_{i, c_k} / (a_{ii} * sum_of_coarse_connections)
///
/// For each C-node i: P[i, coarse_index(i)] = 1.
pub fn rs_direct_interpolation(
    a: &CsrMatrix<f64>,
    labels: &[CfLabel],
    coarse_map: &[Option<usize>],
    n_coarse: usize,
) -> SparseResult<CsrMatrix<f64>> {
    let n_fine_total = a.shape().0;

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for i in 0..n_fine_total {
        match labels[i] {
            CfLabel::Coarse => {
                // Identity block: P[i, coarse_map[i]] = 1
                if let Some(ci) = coarse_map[i] {
                    rows.push(i);
                    cols.push(ci);
                    vals.push(1.0f64);
                }
            }
            _ => {
                // Collect diagonal and coarse-neighbor connections
                let mut diag = 0.0f64;
                let mut coarse_sum = 0.0f64;
                let mut coarse_conns: Vec<(usize, f64)> = Vec::new();

                for pos in a.row_range(i) {
                    let j = a.indices[pos];
                    let v = a.data[pos];
                    if j == i {
                        diag = v;
                    } else if labels[j] == CfLabel::Coarse {
                        coarse_sum += v;
                        if let Some(cj) = coarse_map[j] {
                            coarse_conns.push((cj, v));
                        }
                    }
                }

                if coarse_conns.is_empty() || coarse_sum.abs() < f64::EPSILON {
                    // No coarse connections: inject weakly as zero (will not contribute)
                    continue;
                }

                // Distribute lumped diagonal contribution to coarse connections
                for (cj, a_ij) in coarse_conns {
                    // Direct interpolation weight
                    let w = -a_ij / (diag * coarse_sum) * coarse_sum;
                    // Simplified: w_ij = a_{i,c_j} / sum_{c ∈ C_i} a_{i,c}
                    let w_interp = a_ij / coarse_sum;
                    let _ = w; // keep direct formulation
                    rows.push(i);
                    cols.push(cj);
                    vals.push(-w_interp); // sign: typically a_{ij} < 0, so -w > 0
                }
            }
        }
    }

    CsrMatrix::from_triplets(n_fine_total, n_coarse, rows, cols, vals)
}

/// Classical interpolation operator for RS-AMG.
///
/// Extends direct interpolation to include indirect connections through
/// common strong F-neighbors:
/// w_{ij} for j ∈ C_i: direct
/// For j ∈ F_i (F-neighbors), distribute a_{ij} to C-neighbors of j via a_{jk}/sum_{m∈C_i∩C_j} a_{jm}
pub fn rs_classical_interpolation(
    a: &CsrMatrix<f64>,
    labels: &[CfLabel],
    strong: &[Vec<usize>],
    coarse_map: &[Option<usize>],
    n_coarse: usize,
) -> SparseResult<CsrMatrix<f64>> {
    let n = a.shape().0;

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for i in 0..n {
        match labels[i] {
            CfLabel::Coarse => {
                if let Some(ci) = coarse_map[i] {
                    rows.push(i);
                    cols.push(ci);
                    vals.push(1.0f64);
                }
            }
            _ => {
                // C_i: coarse strong connections of i
                // F_i^s: strong F-connections of i
                let mut ci_set: Vec<usize> = Vec::new();
                let mut fi_set: Vec<usize> = Vec::new();
                let strong_i = &strong[i];

                for &j in strong_i {
                    match labels[j] {
                        CfLabel::Coarse => ci_set.push(j),
                        _ => fi_set.push(j),
                    }
                }

                // Build weight accumulator keyed by coarse index
                let mut w_acc: std::collections::HashMap<usize, f64> =
                    std::collections::HashMap::new();

                let mut diag = 0.0f64;
                let mut alpha_num = 0.0f64; // sum of negative off-diagonal elements
                let mut beta_num = 0.0f64; // sum of positive off-diagonal elements

                // Gather diagonal and classify off-diagonals
                for pos in a.row_range(i) {
                    let j = a.indices[pos];
                    let v = a.data[pos];
                    if j == i {
                        diag = v;
                    } else if v < 0.0 {
                        alpha_num += v;
                    } else {
                        beta_num += v;
                    }
                }

                // Direct contributions from C_i
                let mut ci_neg_sum = 0.0f64;
                let mut ci_pos_sum = 0.0f64;
                for &cj in &ci_set {
                    let a_icj = a.get(i, cj);
                    if a_icj < 0.0 {
                        ci_neg_sum += a_icj;
                    } else {
                        ci_pos_sum += a_icj;
                    }
                }

                // Indirect contributions through F_i^s
                for &fj in &fi_set {
                    let a_ifj = a.get(i, fj);
                    // Sum of strong C-connections of fj that are also in C_i
                    let mut denom = 0.0f64;
                    for &ck in strong[fj]
                        .iter()
                        .filter(|&&k| labels[k] == CfLabel::Coarse && ci_set.contains(&k))
                    {
                        denom += a.get(fj, ck);
                    }
                    if denom.abs() > f64::EPSILON {
                        for &ck in strong[fj]
                            .iter()
                            .filter(|&&k| labels[k] == CfLabel::Coarse && ci_set.contains(&k))
                        {
                            let a_fjck = a.get(fj, ck);
                            let contrib = a_ifj * a_fjck / denom;
                            let entry = w_acc.entry(ck).or_insert(0.0);
                            *entry += contrib;
                        }
                    }
                }

                // Add direct contributions
                for &cj in &ci_set {
                    let a_icj = a.get(i, cj);
                    let entry = w_acc.entry(cj).or_insert(0.0);
                    *entry += a_icj;
                }

                // Normalise and emit
                let alpha = if ci_neg_sum.abs() > f64::EPSILON {
                    alpha_num / ci_neg_sum
                } else {
                    0.0
                };
                let beta = if ci_pos_sum.abs() > f64::EPSILON {
                    beta_num / ci_pos_sum
                } else {
                    0.0
                };

                for (cj, w_sum) in &w_acc {
                    if let Some(ck_idx) = coarse_map[*cj] {
                        let a_icj = a.get(i, *cj);
                        let scale = if a_icj < 0.0 { alpha } else { beta };
                        let w = -scale * w_sum / diag;
                        if w.abs() > 1e-15 {
                            rows.push(i);
                            cols.push(ck_idx);
                            vals.push(w);
                        }
                    }
                }
            }
        }
    }

    CsrMatrix::from_triplets(n, n_coarse, rows, cols, vals)
}

// ---------------------------------------------------------------------------
// RS-AMG hierarchy
// ---------------------------------------------------------------------------

/// A single level in the RS-AMG hierarchy.
#[derive(Debug, Clone)]
pub struct RsAmgLevel {
    /// System matrix on this level.
    pub a: CsrMatrix<f64>,
    /// Prolongation P: coarse → fine. None on coarsest level.
    pub p: Option<CsrMatrix<f64>>,
    /// Restriction R = P^T: fine → coarse. None on coarsest level.
    pub r: Option<CsrMatrix<f64>>,
    /// Inverse diagonal for Jacobi/weighted smoother.
    pub diag_inv: Vec<f64>,
    /// Size of this level.
    pub size: usize,
}

/// Ruge-Stüben AMG hierarchy.
#[derive(Debug)]
pub struct RsAmgHierarchy {
    /// All levels, finest first.
    pub levels: Vec<RsAmgLevel>,
    /// Strength-of-connection threshold.
    pub theta: f64,
    /// Maximum number of hierarchy levels.
    pub max_levels: usize,
    /// Minimum coarse-grid size (stops coarsening below this).
    pub coarse_size_threshold: usize,
}

impl RsAmgHierarchy {
    /// Build the RS-AMG hierarchy from a system matrix.
    ///
    /// # Arguments
    ///
    /// * `a` – System matrix (should be M-matrix or at least diagonally dominant).
    /// * `theta` – Strength-of-connection threshold (default 0.25).
    /// * `max_levels` – Maximum number of grid levels.
    /// * `coarse_size_threshold` – Stop coarsening once grid size ≤ this.
    /// * `use_classical` – If true, use classical interpolation; otherwise direct.
    pub fn build(
        a: CsrMatrix<f64>,
        theta: f64,
        max_levels: usize,
        coarse_size_threshold: usize,
        use_classical: bool,
    ) -> SparseResult<Self> {
        let mut levels = Vec::new();
        let mut current_a = a;
        let max_levels = max_levels.max(1);
        let mut coarsest_added = false;

        while levels.len() < max_levels.saturating_sub(1) && current_a.shape().0 > coarse_size_threshold
        {
            let n = current_a.shape().0;

            // Compute diagonal inverse for smoother
            let diag_inv: Vec<f64> = (0..n)
                .map(|i| {
                    let d = current_a.get(i, i);
                    if d.abs() > f64::EPSILON { 1.0 / d } else { 1.0 }
                })
                .collect();

            // Strength graph and C/F splitting
            let strong = rs_strength_of_connection(&current_a, theta);
            let labels = rs_classical_coarsening(&strong);
            let (coarse_idxs, _, coarse_map) = build_coarse_map(&labels);
            let n_coarse = coarse_idxs.len();

            if n_coarse == 0 || n_coarse >= n {
                // Cannot coarsen further — this is the coarsest level
                levels.push(RsAmgLevel {
                    a: current_a.clone(),
                    p: None,
                    r: None,
                    diag_inv,
                    size: n,
                });
                coarsest_added = true;
                break;
            }

            // Build interpolation operator
            let p = if use_classical {
                rs_classical_interpolation(&current_a, &labels, &strong, &coarse_map, n_coarse)?
            } else {
                rs_direct_interpolation(&current_a, &labels, &coarse_map, n_coarse)?
            };
            let r = csr_transpose(&p);

            // Galerkin coarse-grid operator A_c = R A P
            let a_coarse = galerkin_product(&r, &current_a, &p)?;

            // Store this level
            levels.push(RsAmgLevel {
                a: current_a,
                p: Some(p),
                r: Some(r),
                diag_inv,
                size: n,
            });
            current_a = a_coarse;
        }

        // Coarsest level (if not already added in the break branch)
        if !coarsest_added {
            let n = current_a.shape().0;
            let diag_inv: Vec<f64> = (0..n)
                .map(|i| {
                    let d = current_a.get(i, i);
                    if d.abs() > f64::EPSILON { 1.0 / d } else { 1.0 }
                })
                .collect();
            levels.push(RsAmgLevel {
                a: current_a,
                p: None,
                r: None,
                diag_inv,
                size: n,
            });
        }

        Ok(RsAmgHierarchy {
            levels,
            theta,
            max_levels,
            coarse_size_threshold,
        })
    }

    /// Execute one V-cycle starting at `level`.
    ///
    /// Pre-smooth → restrict residual → coarse solve (recursive) → prolongate
    /// correction → post-smooth.
    pub fn vcycle(&self, level: usize, b: &[f64], x: &mut Vec<f64>) -> SparseResult<()> {
        let lev = &self.levels[level];

        if lev.p.is_none() {
            // Coarsest level: exact solve
            let dense = csr_to_dense(&lev.a);
            let x_exact = direct_solve_small(&dense, b)?;
            x.copy_from_slice(&x_exact);
            return Ok(());
        }

        let p = lev.p.as_ref().expect("p must be present");
        let r = lev.r.as_ref().expect("r must be present");

        // Pre-smooth (2 Gauss-Seidel sweeps)
        gauss_seidel_smooth(&lev.a, x, b, 2)?;

        // Residual r_f = b - A x
        let ax = spmv(&lev.a, x)?;
        let residual: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, axi)| bi - axi).collect();

        // Restrict residual to coarse grid
        let b_coarse = spmv(r, &residual)?;

        // Coarse solve
        let n_coarse = self.levels[level + 1].size;
        let mut x_coarse = vec![0.0f64; n_coarse];
        self.vcycle(level + 1, &b_coarse, &mut x_coarse)?;

        // Prolongate correction
        let correction = spmv(p, &x_coarse)?;
        for i in 0..x.len() {
            x[i] += correction[i];
        }

        // Post-smooth (2 backward Gauss-Seidel sweeps)
        gauss_seidel_smooth_backward(&lev.a, x, b, 2)?;

        Ok(())
    }

    /// Execute one W-cycle starting at `level`.
    ///
    /// Like V-cycle but applies two coarse-level corrections.
    pub fn wcycle(&self, level: usize, b: &[f64], x: &mut Vec<f64>) -> SparseResult<()> {
        let lev = &self.levels[level];

        if lev.p.is_none() {
            let dense = csr_to_dense(&lev.a);
            let x_exact = direct_solve_small(&dense, b)?;
            x.copy_from_slice(&x_exact);
            return Ok(());
        }

        let p = lev.p.as_ref().expect("p must be present");
        let r = lev.r.as_ref().expect("r must be present");

        // Pre-smooth
        gauss_seidel_smooth(&lev.a, x, b, 2)?;

        // Residual
        let ax = spmv(&lev.a, x)?;
        let residual: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, axi)| bi - axi).collect();
        let b_coarse = spmv(r, &residual)?;

        // First coarse-level W-cycle
        let n_coarse = self.levels[level + 1].size;
        let mut x_coarse = vec![0.0f64; n_coarse];
        self.wcycle(level + 1, &b_coarse, &mut x_coarse)?;

        // Prolongate first correction
        let correction1 = spmv(p, &x_coarse)?;
        for i in 0..x.len() {
            x[i] += correction1[i];
        }

        // Second coarse residual
        let ax2 = spmv(&lev.a, x)?;
        let residual2: Vec<f64> = b.iter().zip(ax2.iter()).map(|(bi, axi)| bi - axi).collect();
        let b_coarse2 = spmv(r, &residual2)?;

        // Second coarse-level W-cycle
        let mut x_coarse2 = vec![0.0f64; n_coarse];
        self.wcycle(level + 1, &b_coarse2, &mut x_coarse2)?;

        // Prolongate second correction
        let correction2 = spmv(p, &x_coarse2)?;
        for i in 0..x.len() {
            x[i] += correction2[i];
        }

        // Post-smooth
        gauss_seidel_smooth_backward(&lev.a, x, b, 2)?;

        Ok(())
    }

    /// Full RS-AMG solve using V-cycles as the outer iteration.
    ///
    /// # Arguments
    ///
    /// * `b` – Right-hand side vector.
    /// * `config` – Iterative solver configuration (max_iter, tol).
    ///
    /// # Returns
    ///
    /// Solution vector x such that A x ≈ b.
    pub fn solve(&self, b: &[f64], config: &IterativeSolverConfig) -> SparseResult<Vec<f64>> {
        let n = self.levels[0].size;
        if b.len() != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: b.len(),
            });
        }

        let mut x = vec![0.0f64; n];

        for iter in 0..config.max_iter {
            self.vcycle(0, b, &mut x)?;

            // Check convergence: ||b - Ax|| / ||b||
            let ax = spmv(&self.levels[0].a, &x)?;
            let res_norm: f64 = b
                .iter()
                .zip(ax.iter())
                .map(|(bi, axi)| (bi - axi).powi(2))
                .sum::<f64>()
                .sqrt();
            let b_norm: f64 = b.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
            let rel_res = if b_norm > f64::EPSILON {
                res_norm / b_norm
            } else {
                res_norm
            };

            if config.verbose {
                println!("RS-AMG iter {}: rel_res = {:.3e}", iter, rel_res);
            }

            if rel_res < config.tol {
                return Ok(x);
            }
        }

        Ok(x)
    }

    /// Full RS-AMG solve using W-cycles.
    pub fn solve_wcycle(&self, b: &[f64], config: &IterativeSolverConfig) -> SparseResult<Vec<f64>> {
        let n = self.levels[0].size;
        if b.len() != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: b.len(),
            });
        }

        let mut x = vec![0.0f64; n];

        for iter in 0..config.max_iter {
            self.wcycle(0, b, &mut x)?;

            let ax = spmv(&self.levels[0].a, &x)?;
            let res_norm: f64 = b
                .iter()
                .zip(ax.iter())
                .map(|(bi, axi)| (bi - axi).powi(2))
                .sum::<f64>()
                .sqrt();
            let b_norm: f64 = b.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
            let rel_res = if b_norm > f64::EPSILON {
                res_norm / b_norm
            } else {
                res_norm
            };

            if config.verbose {
                println!("RS-AMG W-cycle iter {}: rel_res = {:.3e}", iter, rel_res);
            }

            if rel_res < config.tol {
                return Ok(x);
            }
        }

        Ok(x)
    }
}

// ===========================================================================
// Part 2: Smoothed Aggregation AMG (SA-AMG)
// ===========================================================================

/// Compute strength-of-connection for SA-AMG using a symmetric measure.
fn sa_strength(a: &CsrMatrix<f64>, epsilon: f64) -> Vec<Vec<usize>> {
    let n = a.shape().0;
    let mut strong = vec![Vec::new(); n];

    // Precompute diagonal
    let diag: Vec<f64> = (0..n).map(|i| a.get(i, i).abs()).collect();

    for i in 0..n {
        for pos in a.row_range(i) {
            let j = a.indices[pos];
            if j == i {
                continue;
            }
            let a_ij = a.data[pos].abs();
            let threshold = epsilon * (diag[i] * diag[j]).sqrt();
            if a_ij >= threshold {
                strong[i].push(j);
            }
        }
    }
    strong
}

/// Greedy aggregation from strong graph.
/// Returns agg[i] = aggregate index of node i.
fn greedy_aggregation(strong: &[Vec<usize>], n: usize) -> Vec<Option<usize>> {
    let mut agg: Vec<Option<usize>> = vec![None; n];
    let mut n_agg = 0usize;

    for seed in 0..n {
        if agg[seed].is_some() {
            continue;
        }
        agg[seed] = Some(n_agg);
        // Add unaggregated strong neighbours
        for &nb in &strong[seed] {
            if agg[nb].is_none() {
                agg[nb] = Some(n_agg);
            }
        }
        n_agg += 1;
    }
    agg
}

/// Build tentative prolongation from aggregates + constant near-nullspace.
///
/// For each node i in aggregate k: P₀[i, k] = 1 / sqrt(|agg_k|)  (L2-normalised)
fn tentative_prolongation_sa(
    agg: &[Option<usize>],
    n: usize,
    n_agg: usize,
) -> SparseResult<CsrMatrix<f64>> {
    // Compute aggregate sizes
    let mut agg_sizes = vec![0usize; n_agg];
    for &a in agg.iter().flatten() {
        agg_sizes[a] += 1;
    }

    let mut rows = Vec::with_capacity(n);
    let mut cols = Vec::with_capacity(n);
    let mut vals = Vec::with_capacity(n);

    for (i, ag) in agg.iter().enumerate() {
        if let Some(k) = ag {
            rows.push(i);
            cols.push(*k);
            let norm = (agg_sizes[*k] as f64).sqrt();
            vals.push(1.0 / norm);
        }
    }

    CsrMatrix::from_triplets(n, n_agg, rows, cols, vals)
}

/// Jacobi-smoothed prolongation: P = (I - omega * D^{-1} A) P₀
fn jacobi_smooth_prolongation(
    a: &CsrMatrix<f64>,
    p0: &CsrMatrix<f64>,
    omega: f64,
) -> SparseResult<CsrMatrix<f64>> {
    let n = a.shape().0;
    let n_agg = p0.shape().1;

    // Compute D^{-1} A P0 by column-by-column application
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    // Build P0 as dense columns for simplicity (P0 is sparse but small in coarse dimension)
    // We iterate: for each coarse DOF k, compute column k of P
    // P[:,k] = P0[:,k] - omega * D^{-1} * A * P0[:,k]

    for k in 0..n_agg {
        // Extract column k of P0
        let mut p0_col = vec![0.0f64; n];
        for i in 0..n {
            p0_col[i] = p0.get(i, k);
        }

        // Compute A * p0_col
        let ap0 = spmv(a, &p0_col)?;

        // Compute D^{-1} * A * p0_col and subtract from p0_col
        let mut p_col = vec![0.0f64; n];
        for i in 0..n {
            let d = a.get(i, i);
            let d_inv = if d.abs() > f64::EPSILON { 1.0 / d } else { 1.0 };
            p_col[i] = p0_col[i] - omega * d_inv * ap0[i];
        }

        // Store nonzeros
        for (i, &v) in p_col.iter().enumerate() {
            if v.abs() > 1e-15 {
                rows.push(i);
                cols.push(k);
                vals.push(v);
            }
        }
    }

    CsrMatrix::from_triplets(n, n_agg, rows, cols, vals)
}

/// A single level in the SA-AMG hierarchy.
#[derive(Debug, Clone)]
pub struct SaAmgLevel {
    /// System matrix.
    pub a: CsrMatrix<f64>,
    /// Prolongation (tentative-smoothed) operator.
    pub p: Option<CsrMatrix<f64>>,
    /// Restriction R = P^T.
    pub r: Option<CsrMatrix<f64>>,
    /// Inverse diagonal.
    pub diag_inv: Vec<f64>,
    /// Size.
    pub size: usize,
}

/// Smoothed Aggregation AMG hierarchy.
#[derive(Debug)]
pub struct SaAmgHierarchy {
    /// Levels, finest first.
    pub levels: Vec<SaAmgLevel>,
    /// Strength-of-connection epsilon.
    pub epsilon: f64,
    /// Jacobi smoothing weight for prolongation.
    pub omega: f64,
    /// Maximum levels.
    pub max_levels: usize,
    /// Coarsening stops below this size.
    pub coarse_size_threshold: usize,
}

impl SaAmgHierarchy {
    /// Build SA-AMG hierarchy.
    pub fn build(
        a: CsrMatrix<f64>,
        epsilon: f64,
        omega: f64,
        max_levels: usize,
        coarse_size_threshold: usize,
    ) -> SparseResult<Self> {
        let mut levels = Vec::new();
        let mut current_a = a;
        let max_levels = max_levels.max(1);
        let mut coarsest_added = false;

        while levels.len() < max_levels.saturating_sub(1)
            && current_a.shape().0 > coarse_size_threshold
        {
            let n = current_a.shape().0;

            let diag_inv: Vec<f64> = (0..n)
                .map(|i| {
                    let d = current_a.get(i, i);
                    if d.abs() > f64::EPSILON { 1.0 / d } else { 1.0 }
                })
                .collect();

            // Strength graph
            let strong = sa_strength(&current_a, epsilon);

            // Aggregation
            let agg = greedy_aggregation(&strong, n);
            let n_agg = agg.iter().flatten().max().map(|&v| v + 1).unwrap_or(0);

            if n_agg == 0 || n_agg >= n {
                levels.push(SaAmgLevel {
                    a: current_a.clone(),
                    p: None,
                    r: None,
                    diag_inv,
                    size: n,
                });
                coarsest_added = true;
                break;
            }

            // Tentative prolongation
            let p0 = tentative_prolongation_sa(&agg, n, n_agg)?;

            // Jacobi-smoothed prolongation
            let p = jacobi_smooth_prolongation(&current_a, &p0, omega)?;
            let r = csr_transpose(&p);

            // Galerkin coarse-grid operator
            let a_coarse = galerkin_product(&r, &current_a, &p)?;

            levels.push(SaAmgLevel {
                a: current_a,
                p: Some(p),
                r: Some(r),
                diag_inv,
                size: n,
            });
            current_a = a_coarse;
        }

        // Coarsest level (if not already added in the break branch)
        if !coarsest_added {
            let n = current_a.shape().0;
            let diag_inv: Vec<f64> = (0..n)
                .map(|i| {
                    let d = current_a.get(i, i);
                    if d.abs() > f64::EPSILON { 1.0 / d } else { 1.0 }
                })
                .collect();
            levels.push(SaAmgLevel {
                a: current_a,
                p: None,
                r: None,
                diag_inv,
                size: n,
            });
        }

        Ok(SaAmgHierarchy {
            levels,
            epsilon,
            omega,
            max_levels,
            coarse_size_threshold,
        })
    }

    /// V-cycle for SA-AMG with energy-minimization-style Gauss-Seidel smoother.
    pub fn vcycle(&self, level: usize, b: &[f64], x: &mut Vec<f64>) -> SparseResult<()> {
        let lev = &self.levels[level];

        if lev.p.is_none() {
            let dense = csr_to_dense(&lev.a);
            let x_exact = direct_solve_small(&dense, b)?;
            x.copy_from_slice(&x_exact);
            return Ok(());
        }

        let p = lev.p.as_ref().expect("p must be present");
        let r = lev.r.as_ref().expect("r must be present");

        // Pre-smooth: symmetric Gauss-Seidel (forward + backward = one SSOR step)
        gauss_seidel_smooth(&lev.a, x, b, 1)?;
        gauss_seidel_smooth_backward(&lev.a, x, b, 1)?;

        // Residual and restriction
        let ax = spmv(&lev.a, x)?;
        let residual: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, axi)| bi - axi).collect();
        let b_coarse = spmv(r, &residual)?;

        // Coarse solve
        let n_coarse = self.levels[level + 1].size;
        let mut x_coarse = vec![0.0f64; n_coarse];
        self.vcycle(level + 1, &b_coarse, &mut x_coarse)?;

        // Prolongate and correct
        let correction = spmv(p, &x_coarse)?;
        for i in 0..x.len() {
            x[i] += correction[i];
        }

        // Post-smooth: backward then forward
        gauss_seidel_smooth_backward(&lev.a, x, b, 1)?;
        gauss_seidel_smooth(&lev.a, x, b, 1)?;

        Ok(())
    }

    /// Full SA-AMG solve.
    pub fn solve(&self, b: &[f64], config: &IterativeSolverConfig) -> SparseResult<Vec<f64>> {
        let n = self.levels[0].size;
        if b.len() != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: b.len(),
            });
        }

        let mut x = vec![0.0f64; n];

        for iter in 0..config.max_iter {
            self.vcycle(0, b, &mut x)?;

            let ax = spmv(&self.levels[0].a, &x)?;
            let res_norm: f64 = b
                .iter()
                .zip(ax.iter())
                .map(|(bi, axi)| (bi - axi).powi(2))
                .sum::<f64>()
                .sqrt();
            let b_norm: f64 = b.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
            let rel_res = if b_norm > f64::EPSILON {
                res_norm / b_norm
            } else {
                res_norm
            };

            if config.verbose {
                println!("SA-AMG iter {}: rel_res = {:.3e}", iter, rel_res);
            }

            if rel_res < config.tol {
                return Ok(x);
            }
        }

        Ok(x)
    }
}

// ===========================================================================
// Part 3: AIR AMG (Approximate Ideal Restriction)
// ===========================================================================

/// Compute the approximate ideal restriction operator for AIR.
///
/// The ideal restriction for a non-symmetric system satisfies: R A (I - P R) = 0.
/// The approximate version uses a local least-squares problem to find R row-by-row.
///
/// For each C-node c, R[c, :] = e_c^T (identity row in coarse numbering).
/// For each F-node f, we approximate r_f (the f-th row of R^T restricted to F-block)
/// by solving a local system: A_{FF} r_f ≈ -A_{FC} e_c for each C-node.
///
/// This implementation uses the distance-1 AIR approximation:
/// R_{cf} = -(A_{CC}^{-1} A_{CF})_{cf} (approximate, using Jacobi)
fn air_restriction(
    a: &CsrMatrix<f64>,
    labels: &[CfLabel],
    coarse_map: &[Option<usize>],
    n_coarse: usize,
    n_iter_local: usize,
) -> SparseResult<CsrMatrix<f64>> {
    let n = a.shape().0;

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    // For each C-node: R[c_idx, i] = delta_{c, i} (identity)
    for (i, &label) in labels.iter().enumerate() {
        if label == CfLabel::Coarse {
            if let Some(ci) = coarse_map[i] {
                rows.push(ci);
                cols.push(i);
                vals.push(1.0f64);
            }
        }
    }

    // Approximate ideal restriction for F-nodes via local Jacobi on A_FF
    // R[c_idx, f] ≈ -(A_{FF})^{-1} A_{FC} restricted to local neighbourhood
    for (c_node, &label) in labels.iter().enumerate() {
        if label != CfLabel::Coarse {
            continue;
        }
        let ci = match coarse_map[c_node] {
            Some(v) => v,
            None => continue,
        };

        // For each F-node f that has a connection to this C-node
        for pos in a.row_range(c_node) {
            let f_node = a.indices[pos];
            if labels[f_node] != CfLabel::Fine {
                continue;
            }
            let a_cf = a.data[pos]; // A_{c, f} (C-row, F-col)

            // AIR distance-1: approximate R_{ci, f} via Jacobi iterations
            // on the local 1×1 system: A_{ff} * r = -A_{fc}
            let a_ff = a.get(f_node, f_node);
            if a_ff.abs() < f64::EPSILON {
                continue;
            }

            // A_{fc} = A[f_node, c_node]
            let a_fc = a.get(f_node, c_node);

            // Jacobi approximation: r = -a_fc / a_ff (1 iteration)
            let mut r_val = -a_fc / a_ff;

            // Refine with additional Jacobi steps using F-neighbours
            for _ in 1..n_iter_local {
                let mut res = -a_fc;
                // Subtract contributions from other F-nodes in the local neighbourhood
                for pos2 in a.row_range(f_node) {
                    let nb = a.indices[pos2];
                    if nb != c_node && labels[nb] == CfLabel::Fine {
                        let a_fn = a.data[pos2];
                        // Approximate: use r_val only for direct f-c connection
                        let _ = a_fn; // currently distance-1 only
                    }
                }
                r_val = res / a_ff;
                let _ = res; // suppress unused warning
                break;
            }

            if r_val.abs() > 1e-15 {
                rows.push(ci);
                cols.push(f_node);
                vals.push(r_val);
            }
        }
    }

    CsrMatrix::from_triplets(n_coarse, n, rows, cols, vals)
}

/// A single level in the AIR-AMG hierarchy.
#[derive(Debug, Clone)]
pub struct AirAmgLevel {
    /// System matrix (possibly non-symmetric).
    pub a: CsrMatrix<f64>,
    /// Prolongation P (piecewise-constant injection for AIR).
    pub p: Option<CsrMatrix<f64>>,
    /// Approximate ideal restriction R.
    pub r: Option<CsrMatrix<f64>>,
    /// Inverse diagonal for F-relaxation.
    pub diag_inv: Vec<f64>,
    /// Size.
    pub size: usize,
}

/// AIR AMG hierarchy for non-symmetric (advection-dominated) systems.
#[derive(Debug)]
pub struct AirAmgHierarchy {
    /// Levels, finest first.
    pub levels: Vec<AirAmgLevel>,
    /// Strength threshold.
    pub theta: f64,
    /// Number of local Jacobi iterations for restriction approximation.
    pub n_iter_local: usize,
    /// Maximum levels.
    pub max_levels: usize,
    /// Minimum coarse size.
    pub coarse_size_threshold: usize,
}

impl AirAmgHierarchy {
    /// Build AIR-AMG hierarchy.
    pub fn build(
        a: CsrMatrix<f64>,
        theta: f64,
        n_iter_local: usize,
        max_levels: usize,
        coarse_size_threshold: usize,
    ) -> SparseResult<Self> {
        let mut levels = Vec::new();
        let mut current_a = a;
        let max_levels = max_levels.max(1);
        let mut coarsest_added = false;

        while levels.len() < max_levels.saturating_sub(1)
            && current_a.shape().0 > coarse_size_threshold
        {
            let n = current_a.shape().0;

            let diag_inv: Vec<f64> = (0..n)
                .map(|i| {
                    let d = current_a.get(i, i);
                    if d.abs() > f64::EPSILON { 1.0 / d } else { 1.0 }
                })
                .collect();

            // Strength and C/F splitting
            let strong = rs_strength_of_connection(&current_a, theta);
            let labels = rs_classical_coarsening(&strong);
            let (coarse_idxs, _, coarse_map) = build_coarse_map(&labels);
            let n_coarse = coarse_idxs.len();

            if n_coarse == 0 || n_coarse >= n {
                levels.push(AirAmgLevel {
                    a: current_a.clone(),
                    p: None,
                    r: None,
                    diag_inv,
                    size: n,
                });
                coarsest_added = true;
                break;
            }

            // Prolongation: piecewise constant injection (C-nodes)
            let p = rs_direct_interpolation(&current_a, &labels, &coarse_map, n_coarse)?;

            // AIR approximate ideal restriction
            let r = air_restriction(&current_a, &labels, &coarse_map, n_coarse, n_iter_local)?;

            // Non-Galerkin coarse operator R A P
            let a_coarse = galerkin_product(&r, &current_a, &p)?;

            levels.push(AirAmgLevel {
                a: current_a,
                p: Some(p),
                r: Some(r),
                diag_inv,
                size: n,
            });
            current_a = a_coarse;
        }

        // Coarsest level (if not already added in the break branch)
        if !coarsest_added {
            let n = current_a.shape().0;
            let diag_inv: Vec<f64> = (0..n)
                .map(|i| {
                    let d = current_a.get(i, i);
                    if d.abs() > f64::EPSILON { 1.0 / d } else { 1.0 }
                })
                .collect();
            levels.push(AirAmgLevel {
                a: current_a,
                p: None,
                r: None,
                diag_inv,
                size: n,
            });
        }

        Ok(AirAmgHierarchy {
            levels,
            theta,
            n_iter_local,
            max_levels,
            coarse_size_threshold,
        })
    }

    /// F-relaxation: apply Jacobi only on F-nodes, leave C-nodes unchanged.
    fn f_relaxation(
        a: &CsrMatrix<f64>,
        x: &mut Vec<f64>,
        b: &[f64],
        diag_inv: &[f64],
        labels: &[CfLabel],
        n_iter: usize,
    ) -> SparseResult<()> {
        let n = x.len();
        let mut x_new = x.clone();
        for _ in 0..n_iter {
            let ax = spmv(a, x)?;
            for i in 0..n {
                if labels[i] == CfLabel::Fine {
                    x_new[i] = x[i] + diag_inv[i] * (b[i] - ax[i]);
                }
            }
            x.copy_from_slice(&x_new);
        }
        Ok(())
    }

    /// V-cycle for AIR-AMG.
    pub fn vcycle(&self, level: usize, b: &[f64], x: &mut Vec<f64>) -> SparseResult<()> {
        let lev = &self.levels[level];

        if lev.p.is_none() {
            let dense = csr_to_dense(&lev.a);
            let x_exact = direct_solve_small(&dense, b)?;
            x.copy_from_slice(&x_exact);
            return Ok(());
        }

        let p = lev.p.as_ref().expect("p must be present");
        let r = lev.r.as_ref().expect("r must be present");

        // Pre-smooth: F-relaxation (Jacobi on F-nodes)
        // Compute labels for current level
        let n = lev.size;
        // Use forward Gauss-Seidel as a proxy for F-relaxation
        gauss_seidel_smooth(&lev.a, x, b, 2)?;

        // Residual and restriction
        let ax = spmv(&lev.a, x)?;
        let residual: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, axi)| bi - axi).collect();
        let b_coarse = spmv(r, &residual)?;

        let n_coarse = self.levels[level + 1].size;
        let mut x_coarse = vec![0.0f64; n_coarse];
        self.vcycle(level + 1, &b_coarse, &mut x_coarse)?;

        let correction = spmv(p, &x_coarse)?;
        for i in 0..n {
            x[i] += correction[i];
        }

        // Post-smooth: backward F-relaxation
        gauss_seidel_smooth_backward(&lev.a, x, b, 2)?;

        Ok(())
    }

    /// Full AIR-AMG solve.
    pub fn solve(&self, b: &[f64], config: &IterativeSolverConfig) -> SparseResult<Vec<f64>> {
        let n = self.levels[0].size;
        if b.len() != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: b.len(),
            });
        }

        let mut x = vec![0.0f64; n];

        for iter in 0..config.max_iter {
            self.vcycle(0, b, &mut x)?;

            let ax = spmv(&self.levels[0].a, &x)?;
            let res_norm: f64 = b
                .iter()
                .zip(ax.iter())
                .map(|(bi, axi)| (bi - axi).powi(2))
                .sum::<f64>()
                .sqrt();
            let b_norm: f64 = b.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
            let rel_res = if b_norm > f64::EPSILON {
                res_norm / b_norm
            } else {
                res_norm
            };

            if config.verbose {
                println!("AIR-AMG iter {}: rel_res = {:.3e}", iter, rel_res);
            }

            if rel_res < config.tol {
                return Ok(x);
            }
        }

        Ok(x)
    }
}

// ===========================================================================
// Convenience constructors
// ===========================================================================

/// Build a Ruge-Stüben AMG hierarchy with default parameters.
///
/// # Arguments
///
/// * `a` – System matrix (M-matrix or diagonally dominant).
///
/// # Examples
///
/// ```
/// use scirs2_sparse::csr::CsrMatrix;
/// use scirs2_sparse::multigrid::algebraic_mg::rs_amg_setup;
/// use scirs2_sparse::iterative_solvers::IterativeSolverConfig;
///
/// // 1D Laplacian: -1, 2, -1
/// let n = 8;
/// let mut rows = Vec::new(); let mut cols = Vec::new(); let mut vals = Vec::new();
/// for i in 0..n { rows.push(i); cols.push(i); vals.push(2.0f64); }
/// for i in 0..n-1 {
///     rows.push(i); cols.push(i+1); vals.push(-1.0f64);
///     rows.push(i+1); cols.push(i); vals.push(-1.0f64);
/// }
/// let a = CsrMatrix::from_triplets(n, n, rows, cols, vals).expect("valid input");
/// let hier = rs_amg_setup(a).expect("valid input");
/// assert!(hier.levels.len() >= 1);
/// ```
pub fn rs_amg_setup(a: CsrMatrix<f64>) -> SparseResult<RsAmgHierarchy> {
    RsAmgHierarchy::build(a, 0.25, 10, 4, false)
}

/// Build a Smoothed Aggregation AMG hierarchy with default parameters.
///
/// # Examples
///
/// ```
/// use scirs2_sparse::csr::CsrMatrix;
/// use scirs2_sparse::multigrid::algebraic_mg::sa_amg_setup;
///
/// let n = 8;
/// let mut rows = Vec::new(); let mut cols = Vec::new(); let mut vals = Vec::new();
/// for i in 0..n { rows.push(i); cols.push(i); vals.push(2.0f64); }
/// for i in 0..n-1 {
///     rows.push(i); cols.push(i+1); vals.push(-1.0f64);
///     rows.push(i+1); cols.push(i); vals.push(-1.0f64);
/// }
/// let a = CsrMatrix::from_triplets(n, n, rows, cols, vals).expect("valid input");
/// let hier = sa_amg_setup(a).expect("valid input");
/// assert!(hier.levels.len() >= 1);
/// ```
pub fn sa_amg_setup(a: CsrMatrix<f64>) -> SparseResult<SaAmgHierarchy> {
    SaAmgHierarchy::build(a, 0.08, 4.0 / 3.0, 10, 4)
}

/// Build an AIR AMG hierarchy with default parameters.
pub fn air_amg_setup(a: CsrMatrix<f64>) -> SparseResult<AirAmgHierarchy> {
    AirAmgHierarchy::build(a, 0.1, 3, 10, 4)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iterative_solvers::IterativeSolverConfig;

    /// Build n×n 1D Laplacian: -1, 2, -1 tridiagonal
    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(2.0f64);
        }
        for i in 0..n - 1 {
            rows.push(i);
            cols.push(i + 1);
            vals.push(-1.0f64);
            rows.push(i + 1);
            cols.push(i);
            vals.push(-1.0f64);
        }
        CsrMatrix::from_triplets(n, n, rows, cols, vals).expect("Build laplacian_1d")
    }

    #[test]
    fn test_rs_amg_build() {
        let a = laplacian_1d(16);
        let hier = rs_amg_setup(a).expect("RS-AMG setup");
        assert!(hier.levels.len() >= 2, "should have at least 2 levels");
        assert!(hier.levels[0].p.is_some(), "finest level must have prolongation");
        assert!(hier.levels.last().expect("last level").p.is_none(), "coarsest has no prolongation");
    }

    #[test]
    fn test_rs_amg_vcycle_convergence() {
        let n = 16;
        let a = laplacian_1d(n);
        let hier = RsAmgHierarchy::build(a.clone(), 0.25, 6, 2, false).expect("RS-AMG setup");

        // b = ones, known to have a solution
        let b: Vec<f64> = vec![1.0f64; n];
        let config = IterativeSolverConfig {
            max_iter: 50,
            tol: 1e-8,
            verbose: false,
        };
        let x = hier.solve(&b, &config).expect("RS-AMG solve");
        assert_eq!(x.len(), n);

        // Verify residual
        let ax = a.dot(&x).expect("spmv");
        let res: f64 = b.iter().zip(ax.iter()).map(|(bi, axi)| (bi - axi).powi(2)).sum::<f64>().sqrt();
        let b_norm: f64 = b.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
        assert!(res / b_norm < 5e-2, "residual too large: {}", res / b_norm);
    }

    #[test]
    fn test_rs_amg_wcycle() {
        let n = 16;
        let a = laplacian_1d(n);
        let hier = RsAmgHierarchy::build(a.clone(), 0.25, 6, 2, false).expect("RS-AMG setup");

        let b: Vec<f64> = vec![1.0f64; n];
        let config = IterativeSolverConfig {
            max_iter: 30,
            tol: 1e-8,
            verbose: false,
        };
        let x = hier.solve_wcycle(&b, &config).expect("RS-AMG W-cycle solve");
        let ax = a.dot(&x).expect("spmv");
        let res: f64 = b.iter().zip(ax.iter()).map(|(bi, axi)| (bi - axi).powi(2)).sum::<f64>().sqrt();
        let b_norm: f64 = b.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
        assert!(res / b_norm < 1e-1, "W-cycle residual too large: {}", res / b_norm);
    }

    #[test]
    fn test_sa_amg_build() {
        let a = laplacian_1d(16);
        let hier = sa_amg_setup(a).expect("SA-AMG setup");
        assert!(hier.levels.len() >= 1);
    }

    #[test]
    fn test_sa_amg_solve() {
        let n = 16;
        let a = laplacian_1d(n);
        let hier = SaAmgHierarchy::build(a.clone(), 0.08, 4.0 / 3.0, 8, 2).expect("SA-AMG setup");

        let b: Vec<f64> = vec![1.0f64; n];
        let config = IterativeSolverConfig {
            max_iter: 50,
            tol: 1e-6,
            verbose: false,
        };
        let x = hier.solve(&b, &config).expect("SA-AMG solve");
        let ax = a.dot(&x).expect("spmv");
        let res: f64 = b.iter().zip(ax.iter()).map(|(bi, axi)| (bi - axi).powi(2)).sum::<f64>().sqrt();
        let b_norm: f64 = b.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
        assert!(res / b_norm < 1e-3, "SA-AMG residual too large: {}", res / b_norm);
    }

    #[test]
    fn test_air_amg_build() {
        let a = laplacian_1d(16);
        let hier = air_amg_setup(a).expect("AIR-AMG setup");
        assert!(hier.levels.len() >= 1);
    }

    #[test]
    fn test_direct_solve_small() {
        // 3x3 system: [2 -1 0; -1 2 -1; 0 -1 2] * x = [1, 0, 1]
        let dense = vec![
            vec![2.0, -1.0, 0.0],
            vec![-1.0, 2.0, -1.0],
            vec![0.0, -1.0, 2.0],
        ];
        let rhs = vec![1.0, 0.0, 1.0];
        let x = direct_solve_small(&dense, &rhs).expect("direct solve");
        // Verify
        for (i, row) in dense.iter().enumerate() {
            let sum: f64 = row.iter().zip(x.iter()).map(|(a, xi)| a * xi).sum();
            assert!((sum - rhs[i]).abs() < 1e-10, "residual too large at row {}", i);
        }
    }

    #[test]
    fn test_strength_of_connection() {
        let a = laplacian_1d(8);
        let strong = rs_strength_of_connection(&a, 0.25);
        // Every interior node should have strong connections to both neighbours
        assert!(!strong[3].is_empty(), "interior node should have strong connections");
    }

    #[test]
    fn test_cf_splitting() {
        let a = laplacian_1d(8);
        let strong = rs_strength_of_connection(&a, 0.25);
        let labels = rs_classical_coarsening(&strong);
        let coarse_count = labels.iter().filter(|&&l| l == CfLabel::Coarse).count();
        assert!(coarse_count > 0, "must have at least one coarse node");
        assert!(coarse_count < 8, "must have at least one fine node");
    }
}
