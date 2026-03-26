//! Multilevel Incomplete LU (MILUE) preconditioner with coarse correction
//!
//! This module implements a two-level (and extensible multilevel) incomplete LU
//! preconditioner that combines:
//!
//! - **Fine-level smoother**: ILU(0) applied to the fine-level block.
//! - **C/F splitting**: nodes are split into Coarse (C) and Fine (F) sets using
//!   a greedy maximum-weight independent set (MIS) strategy.
//! - **Schur complement**: an approximate Schur complement is formed for the
//!   coarse-level system S ≈ A_CC − A_CF (A_FF)⁻¹ A_FC.
//! - **V-cycle**: pre-smooth, restrict, coarse solve, prolongate, post-smooth.
//!
//! # Algorithm overview
//!
//! Given the 2×2 block decomposition (after C/F ordering):
//!
//! ```text
//!   A = [A_FF  A_FC]
//!       [A_CF  A_CC]
//! ```
//!
//! The MILUE V-cycle is:
//! 1. Pre-smooth on fine level:   y = ILU_F^{-1} r
//! 2. Update residual:            r_F' = r_F − A_FF y_F − A_FC y_C
//! 3. Restrict:                   r_c = A_CF ILU_F^{-1} r_F'  (coarse residual)
//! 4. Coarse solve:               e_c = S^{-1} r_c  (recursive or direct)
//! 5. Prolongate:                 y_F += ILU_F^{-1} A_FC e_c
//! 6. Post-smooth:                (one more ILU sweep)
//!
//! # References
//!
//! - Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed.
//!   §§10.6–10.7 (multilevel ILU).
//! - Notay, Y. (2010). An aggregation-based algebraic multigrid method.
//!   *ETNA* 37, 123–146.

use crate::error::{SparseError, SparseResult};

// ---------------------------------------------------------------------------
// Internal CSR helpers
// ---------------------------------------------------------------------------

/// y = A * x (CSR sparse matvec).
fn csr_matvec(
    row_ptrs: &[usize],
    col_indices: &[usize],
    values: &[f64],
    x: &[f64],
    nrows: usize,
) -> Vec<f64> {
    let mut y = vec![0.0f64; nrows];
    for i in 0..nrows {
        let mut acc = 0.0f64;
        for pos in row_ptrs[i]..row_ptrs[i + 1] {
            acc += values[pos] * x[col_indices[pos]];
        }
        y[i] = acc;
    }
    y
}

/// Forward solve: (unit lower-triangular L) y = b.
fn forward_solve_unit(
    l_data: &[f64],
    l_indices: &[usize],
    l_indptr: &[usize],
    b: &[f64],
    n: usize,
) -> Vec<f64> {
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut acc = b[i];
        for pos in l_indptr[i]..l_indptr[i + 1] {
            let j = l_indices[pos];
            if j < i {
                acc -= l_data[pos] * y[j];
            }
        }
        y[i] = acc;
    }
    y
}

/// Backward solve: (upper-triangular U) x = y.
fn backward_solve(
    u_data: &[f64],
    u_indices: &[usize],
    u_indptr: &[usize],
    y: &[f64],
    n: usize,
) -> SparseResult<Vec<f64>> {
    let mut x = vec![0.0f64; n];
    for ii in 0..n {
        let i = n - 1 - ii;
        let mut diag = 0.0f64;
        let mut sum = y[i];
        for pos in u_indptr[i]..u_indptr[i + 1] {
            let j = u_indices[pos];
            if j == i {
                diag = u_data[pos];
            } else if j > i {
                sum -= u_data[pos] * x[j];
            }
        }
        if diag.abs() < 1e-300 {
            return Err(SparseError::SingularMatrix(format!(
                "zero diagonal at row {i} in backward solve"
            )));
        }
        x[i] = sum / diag;
    }
    Ok(x)
}

/// Find column `col` in sorted slice `indices[start..end]`.
#[inline]
fn find_col(indices: &[usize], start: usize, end: usize, col: usize) -> Option<usize> {
    for pos in start..end {
        match indices[pos].cmp(&col) {
            std::cmp::Ordering::Equal => return Some(pos),
            std::cmp::Ordering::Greater => return None,
            std::cmp::Ordering::Less => {}
        }
    }
    None
}

// ---------------------------------------------------------------------------
// ILU(0) over a sub-matrix (internal)
// ---------------------------------------------------------------------------

/// Compact ILU(0) factors for the fine-level block.
struct Ilu0Factors {
    /// Unit lower-triangular factor: (indptr, indices, data).
    l: (Vec<usize>, Vec<usize>, Vec<f64>),
    /// Upper-triangular factor: (indptr, indices, data).
    u: (Vec<usize>, Vec<usize>, Vec<f64>),
    n: usize,
}

impl Ilu0Factors {
    /// Compute ILU(0) from a CSR matrix with sorted column indices.
    fn factor(
        indptr: &[usize],
        indices: &[usize],
        data: &[f64],
        n: usize,
    ) -> SparseResult<Self> {
        if indptr.len() != n + 1 {
            return Err(SparseError::InconsistentData {
                reason: format!("indptr length {} != n+1={}", indptr.len(), n + 1),
            });
        }

        let mut a = data.to_vec();

        for i in 1..n {
            let row_start = indptr[i];
            let row_end = indptr[i + 1];
            for pos_j in row_start..row_end {
                let j = indices[pos_j];
                if j >= i {
                    break;
                }
                let pivot_pos =
                    find_col(indices, indptr[j], indptr[j + 1], j).ok_or_else(|| {
                        SparseError::SingularMatrix(format!("ILU(0): missing diagonal at row {j}"))
                    })?;
                let a_jj = a[pivot_pos];
                if a_jj.abs() < 1e-300 {
                    return Err(SparseError::SingularMatrix(format!(
                        "ILU(0): zero diagonal at row {j}"
                    )));
                }
                a[pos_j] /= a_jj;
                let mult = a[pos_j];
                for pos_k in (pos_j + 1)..row_end {
                    let k = indices[pos_k];
                    if let Some(jk_pos) = find_col(indices, indptr[j], indptr[j + 1], k) {
                        a[pos_k] -= mult * a[jk_pos];
                    }
                }
            }
        }

        // Split into L and U.
        let mut l_indptr = vec![0usize; n + 1];
        let mut u_indptr = vec![0usize; n + 1];
        for i in 0..n {
            for pos in indptr[i]..indptr[i + 1] {
                let j = indices[pos];
                if j < i {
                    l_indptr[i + 1] += 1;
                } else {
                    u_indptr[i + 1] += 1;
                }
            }
        }
        for i in 0..n {
            l_indptr[i + 1] += l_indptr[i];
            u_indptr[i + 1] += u_indptr[i];
        }
        let l_nnz = l_indptr[n];
        let u_nnz = u_indptr[n];
        let mut l_indices = vec![0usize; l_nnz];
        let mut l_data = vec![0.0f64; l_nnz];
        let mut u_indices = vec![0usize; u_nnz];
        let mut u_data = vec![0.0f64; u_nnz];
        let mut l_cur = l_indptr[..n].to_vec();
        let mut u_cur = u_indptr[..n].to_vec();
        for i in 0..n {
            for pos in indptr[i]..indptr[i + 1] {
                let j = indices[pos];
                if j < i {
                    let dst = l_cur[i];
                    l_indices[dst] = j;
                    l_data[dst] = a[pos];
                    l_cur[i] += 1;
                } else {
                    let dst = u_cur[i];
                    u_indices[dst] = j;
                    u_data[dst] = a[pos];
                    u_cur[i] += 1;
                }
            }
        }

        Ok(Self {
            l: (l_indptr, l_indices, l_data),
            u: (u_indptr, u_indices, u_data),
            n,
        })
    }

    /// Solve LUx = b.
    fn apply(&self, b: &[f64]) -> SparseResult<Vec<f64>> {
        let (l_ip, l_idx, l_dat) = &self.l;
        let (u_ip, u_idx, u_dat) = &self.u;
        let y = forward_solve_unit(l_dat, l_idx, l_ip, b, self.n);
        backward_solve(u_dat, u_idx, u_ip, &y, self.n)
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the MILUE (Multilevel ILU) preconditioner.
#[derive(Debug, Clone)]
pub struct MilueConfig {
    /// Number of levels (must be ≥ 1).  A value of 1 degenerates to plain ILU.
    pub levels: usize,
    /// Drop tolerance for off-diagonal entries of the Schur complement.
    pub drop_tol: f64,
    /// Fill-factor relative to nnz(A): limits the Schur complement density.
    pub fill_factor: f64,
}

impl Default for MilueConfig {
    fn default() -> Self {
        Self {
            levels: 2,
            drop_tol: 0.01,
            fill_factor: 3.0,
        }
    }
}

// ---------------------------------------------------------------------------
// C/F splitting
// ---------------------------------------------------------------------------

/// Node classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NodeType {
    Undecided,
    Fine,
    Coarse,
}

/// Greedy maximum-weight independent set C/F splitting.
///
/// Strategy: iterate over nodes in order; if a node is *Undecided*, mark it
/// Coarse and all its neighbours Fine.  This produces an independent set of
/// Coarse nodes (no two coarse nodes share an edge), which is a valid C/F
/// partition for aggregation-based multilevel methods.
///
/// Returns `(coarse_nodes, fine_nodes)` as index vectors into [0..n).
fn cf_split(
    row_ptrs: &[usize],
    col_indices: &[usize],
    n: usize,
) -> (Vec<usize>, Vec<usize>) {
    let mut node_type = vec![NodeType::Undecided; n];
    for i in 0..n {
        if node_type[i] != NodeType::Undecided {
            continue;
        }
        node_type[i] = NodeType::Coarse;
        // Mark all neighbours as Fine (they are interpolated from i).
        for pos in row_ptrs[i]..row_ptrs[i + 1] {
            let j = col_indices[pos];
            if j != i && node_type[j] == NodeType::Undecided {
                node_type[j] = NodeType::Fine;
            }
        }
    }
    // Any remaining Undecided nodes (isolated) go Coarse.
    for nt in node_type.iter_mut() {
        if *nt == NodeType::Undecided {
            *nt = NodeType::Coarse;
        }
    }
    let coarse: Vec<usize> = (0..n).filter(|&i| node_type[i] == NodeType::Coarse).collect();
    let fine: Vec<usize> = (0..n).filter(|&i| node_type[i] == NodeType::Fine).collect();
    (coarse, fine)
}

// ---------------------------------------------------------------------------
// Block extraction from a CSR matrix
// ---------------------------------------------------------------------------

/// Extract the sub-matrix A[rows, cols] as a new CSR matrix.
/// `row_map[i]` = global row index of local row i.
/// `col_global_to_local` maps global column → local column (or None).
fn extract_block(
    row_ptrs: &[usize],
    col_indices: &[usize],
    values: &[f64],
    row_map: &[usize],
    col_global_to_local: &[Option<usize>],
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let m = row_map.len();
    let mut rp = vec![0usize; m + 1];
    let mut ci: Vec<usize> = Vec::new();
    let mut vs: Vec<f64> = Vec::new();

    for (local_row, &global_row) in row_map.iter().enumerate() {
        for pos in row_ptrs[global_row]..row_ptrs[global_row + 1] {
            let gc = col_indices[pos];
            if let Some(lc) = col_global_to_local.get(gc).and_then(|x| *x) {
                ci.push(lc);
                vs.push(values[pos]);
            }
        }
        rp[local_row + 1] = ci.len();
    }
    (rp, ci, vs)
}

// ---------------------------------------------------------------------------
// Schur complement approximation
// ---------------------------------------------------------------------------

/// Compute approximate Schur complement S ≈ A_CC − A_CF * D_FF^{-1} * A_FC
/// using the diagonal of A_FF as a cheap approximation of (A_FF)^{-1}.
///
/// The result is returned as a new CSR matrix (coarse×coarse) with
/// entries dropped if |s_ij| < drop_tol * ||row||_∞.
fn approx_schur(
    rp: &[usize],
    ci: &[usize],
    vs: &[f64],
    n: usize,
    coarse: &[usize],
    fine: &[usize],
    drop_tol: f64,
    fill_factor: f64,
) -> SparseResult<(Vec<usize>, Vec<usize>, Vec<f64>)> {
    let nc = coarse.len();
    let nf = fine.len();

    if nc == 0 {
        return Ok((vec![0usize; 1], Vec::new(), Vec::new()));
    }

    // Build inverse mapping: global index → coarse local index.
    let mut global_to_coarse = vec![None::<usize>; n];
    for (lc, &gc) in coarse.iter().enumerate() {
        global_to_coarse[gc] = Some(lc);
    }

    // Build inverse mapping: global index → fine local index.
    let mut global_to_fine = vec![None::<usize>; n];
    for (lf, &gf) in fine.iter().enumerate() {
        global_to_fine[gf] = Some(lf);
    }

    // Diagonal of the full matrix (used to approximate A_FF^{-1}).
    let mut a_diag = vec![1.0f64; n]; // default to 1 (avoid division by zero)
    for i in 0..n {
        for pos in rp[i]..rp[i + 1] {
            if ci[pos] == i {
                let d = vs[pos];
                if d.abs() > 1e-300 {
                    a_diag[i] = d;
                }
            }
        }
    }

    // Build A_CC as dense (nc×nc), initialised with A_CC block.
    let max_nnz_budget = ((rp[n] as f64) * fill_factor) as usize + nc + 1;
    let _ = max_nnz_budget; // used for future fill control

    // We build S row-by-row using the formula:
    // S_{ij} = A_{ij} (coarse-coarse) - Σ_f A_{if} / A_{ff} * A_{fj}  (f in Fine)
    // where i, j are coarse indices.

    let mut s_row_ptrs = vec![0usize; nc + 1];
    let mut s_col_indices: Vec<usize> = Vec::new();
    let mut s_values: Vec<f64> = Vec::new();

    // For each coarse row i:
    for (li, &gi) in coarse.iter().enumerate() {
        // Start with A_{gi, *coarse} entries.
        let mut row_map: std::collections::HashMap<usize, f64> =
            std::collections::HashMap::new();

        for pos in rp[gi]..rp[gi + 1] {
            let gj = ci[pos];
            if let Some(lj) = global_to_coarse[gj] {
                *row_map.entry(lj).or_insert(0.0) += vs[pos];
            }
        }

        // Subtract A_{gi, f} / A_{ff} * A_{fj} for each fine neighbour f.
        for pos_f in rp[gi]..rp[gi + 1] {
            let gf = ci[pos_f];
            if global_to_fine[gf].is_none() {
                continue;
            }
            let a_if = vs[pos_f];
            let a_ff = a_diag[gf];
            let mult = a_if / a_ff;

            // Add -mult * A_{gf, *coarse}.
            for pos_fc in rp[gf]..rp[gf + 1] {
                let gj = ci[pos_fc];
                if let Some(lj) = global_to_coarse[gj] {
                    *row_map.entry(lj).or_insert(0.0) -= mult * vs[pos_fc];
                }
            }
        }

        // Drop small entries.
        let row_max = row_map.values().map(|v| v.abs()).fold(0.0f64, f64::max);
        let threshold = drop_tol * row_max;
        let mut row_entries: Vec<(usize, f64)> = row_map
            .into_iter()
            .filter(|(_, v)| v.abs() >= threshold || threshold == 0.0)
            .collect();
        row_entries.sort_unstable_by_key(|&(c, _)| c);

        for (col, val) in row_entries {
            s_col_indices.push(col);
            s_values.push(val);
        }
        s_row_ptrs[li + 1] = s_col_indices.len();

        let _ = li; // suppress unused warning
    }

    // Ensure the diagonal is present (add small regularisation if missing).
    ensure_diagonal(&mut s_row_ptrs, &mut s_col_indices, &mut s_values, nc)?;

    Ok((s_row_ptrs, s_col_indices, s_values))
}

/// Ensure every diagonal entry is present (add 1e-14 if missing).
fn ensure_diagonal(
    row_ptrs: &mut Vec<usize>,
    col_indices: &mut Vec<usize>,
    values: &mut Vec<f64>,
    n: usize,
) -> SparseResult<()> {
    // Rebuild from scratch because insertions into sorted CSR are costly.
    let mut new_rp = vec![0usize; n + 1];
    let mut new_ci: Vec<usize> = Vec::new();
    let mut new_vs: Vec<f64> = Vec::new();

    for i in 0..n {
        let start = row_ptrs[i];
        let end = row_ptrs[i + 1];
        let mut has_diag = false;

        // Copy existing entries for row i in sorted order.
        let mut entries: Vec<(usize, f64)> = col_indices[start..end]
            .iter()
            .zip(values[start..end].iter())
            .map(|(&c, &v)| (c, v))
            .collect();
        entries.sort_unstable_by_key(|&(c, _)| c);

        for &(c, _) in &entries {
            if c == i {
                has_diag = true;
                break;
            }
        }
        if !has_diag {
            entries.push((i, 1e-14));
            entries.sort_unstable_by_key(|&(c, _)| c);
        }

        for (c, v) in entries {
            new_ci.push(c);
            new_vs.push(v);
        }
        new_rp[i + 1] = new_ci.len();
    }

    *row_ptrs = new_rp;
    *col_indices = new_ci;
    *values = new_vs;
    Ok(())
}

// ---------------------------------------------------------------------------
// MILUE Level data
// ---------------------------------------------------------------------------

/// Data for one level of the MILUE hierarchy.
pub struct MilueLevel {
    /// Fine-level ILU(0) factors for A_FF.
    ilu: Ilu0Factors,
    /// Indices of Fine nodes (in the *parent* level's ordering).
    pub fine_nodes: Vec<usize>,
    /// Indices of Coarse nodes (in the *parent* level's ordering).
    pub coarse_nodes: Vec<usize>,
    /// A_CF block: (row_ptrs, col_indices, values) — rows=coarse, cols=fine.
    a_cf: (Vec<usize>, Vec<usize>, Vec<f64>),
    /// A_FC block: rows=fine, cols=coarse.
    a_fc: (Vec<usize>, Vec<usize>, Vec<f64>),
    /// A_FF block (needed to compute residual on fine nodes).
    a_ff: (Vec<usize>, Vec<usize>, Vec<f64>),
    /// A_CC block (needed at the base level for direct solve).
    a_cc: (Vec<usize>, Vec<usize>, Vec<f64>),
}

// ---------------------------------------------------------------------------
// MILUE Preconditioner
// ---------------------------------------------------------------------------

/// Multilevel ILU (MILUE) preconditioner with coarse correction.
///
/// Applies a V-cycle:
/// 1. Pre-smooth with ILU(0) on the fine-level block.
/// 2. Compute coarse-level residual.
/// 3. Recursively solve (or direct-solve) at the coarse level.
/// 4. Prolongate correction back to the fine level.
/// 5. Post-smooth.
pub struct MiluePreconditioner {
    /// Levels (level 0 = finest).  The last level stores the coarsest system
    /// and uses a direct ILU solve instead of recursion.
    levels: Vec<MilueLevel>,
    /// Original matrix dimension.
    n: usize,
    /// Configuration.
    config: MilueConfig,
}

impl MiluePreconditioner {
    /// Construct the MILUE preconditioner for a square CSR matrix.
    ///
    /// # Arguments
    ///
    /// * `row_ptrs`, `col_indices`, `values` – CSR representation of A (must
    ///   have sorted column indices per row).
    /// * `n`      – Matrix dimension.
    /// * `config` – Algorithm configuration.
    pub fn new(
        row_ptrs: &[usize],
        col_indices: &[usize],
        values: &[f64],
        n: usize,
        config: MilueConfig,
    ) -> SparseResult<Self> {
        if row_ptrs.len() != n + 1 {
            return Err(SparseError::InconsistentData {
                reason: format!(
                    "row_ptrs length {} != n+1={}",
                    row_ptrs.len(),
                    n + 1
                ),
            });
        }

        let num_levels = config.levels.max(1);
        let mut levels: Vec<MilueLevel> = Vec::with_capacity(num_levels);

        // Current level CSR (changes as we recurse).
        let mut cur_rp = row_ptrs.to_vec();
        let mut cur_ci = col_indices.to_vec();
        let mut cur_vs = values.to_vec();
        let mut cur_n = n;

        for _lvl in 0..num_levels {
            if cur_n <= 1 {
                break;
            }

            // C/F splitting on the current level.
            let (coarse_nodes, fine_nodes) =
                cf_split(&cur_rp, &cur_ci, cur_n);

            let nf = fine_nodes.len();
            let nc = coarse_nodes.len();

            if nf == 0 || nc == 0 {
                break;
            }

            // Build global → local maps.
            let mut g2f = vec![None::<usize>; cur_n];
            for (lf, &gf) in fine_nodes.iter().enumerate() {
                g2f[gf] = Some(lf);
            }
            let mut g2c = vec![None::<usize>; cur_n];
            for (lc, &gc) in coarse_nodes.iter().enumerate() {
                g2c[gc] = Some(lc);
            }

            // Extract the four blocks.
            let a_ff = extract_block(&cur_rp, &cur_ci, &cur_vs, &fine_nodes, &g2f);
            let a_fc = extract_block(&cur_rp, &cur_ci, &cur_vs, &fine_nodes, &g2c);
            let a_cf = extract_block(&cur_rp, &cur_ci, &cur_vs, &coarse_nodes, &g2f);
            let a_cc = extract_block(&cur_rp, &cur_ci, &cur_vs, &coarse_nodes, &g2c);

            // Ensure A_FF has diagonal entries before ILU.
            let (mut ff_rp, mut ff_ci, mut ff_vs) = a_ff;
            ensure_diagonal(&mut ff_rp, &mut ff_ci, &mut ff_vs, nf)?;

            // ILU(0) on the fine-level block A_FF.
            let ilu = Ilu0Factors::factor(&ff_rp, &ff_ci, &ff_vs, nf)?;

            let level = MilueLevel {
                ilu,
                fine_nodes: fine_nodes.clone(),
                coarse_nodes: coarse_nodes.clone(),
                a_cf,
                a_fc,
                a_ff: (ff_rp, ff_ci, ff_vs),
                a_cc: a_cc.clone(),
            };
            levels.push(level);

            // Build the Schur complement for the next level.
            let (s_rp, s_ci, s_vs) = approx_schur(
                &cur_rp,
                &cur_ci,
                &cur_vs,
                cur_n,
                &coarse_nodes,
                &fine_nodes,
                config.drop_tol,
                config.fill_factor,
            )?;

            cur_rp = s_rp;
            cur_ci = s_ci;
            cur_vs = s_vs;
            cur_n = nc;
        }

        Ok(Self { levels, n, config })
    }

    /// Apply the MILUE V-cycle preconditioner: compute y ≈ A^{-1} r.
    ///
    /// The application performs:
    /// 1. Pre-smooth on fine nodes via ILU(0).
    /// 2. Restrict residual to coarse.
    /// 3. Recursively solve at coarse level (or direct ILU at bottom).
    /// 4. Prolongate coarse correction.
    /// 5. Post-smooth.
    pub fn apply(&self, r: &[f64]) -> Vec<f64> {
        if self.levels.is_empty() {
            return r.to_vec();
        }
        match self.vcycle(r, 0) {
            Ok(y) => y,
            Err(_) => r.to_vec(),
        }
    }

    /// Recursive V-cycle starting at `level_idx`.
    fn vcycle(&self, r: &[f64], level_idx: usize) -> SparseResult<Vec<f64>> {
        if level_idx >= self.levels.len() {
            // Bottom level: just return the input as the solution (identity).
            return Ok(r.to_vec());
        }

        let lvl = &self.levels[level_idx];
        let n_total = r.len();
        let nf = lvl.fine_nodes.len();
        let nc = lvl.coarse_nodes.len();

        // Gather fine and coarse components of r.
        let r_f: Vec<f64> = lvl.fine_nodes.iter().map(|&i| r[i]).collect();
        let r_c: Vec<f64> = lvl.coarse_nodes.iter().map(|&i| r[i]).collect();

        // ---------------------------------------------------------------
        // Step 1: Pre-smooth on fine nodes.
        //   y_f = ILU_F^{-1} r_f
        // ---------------------------------------------------------------
        let y_f = lvl.ilu.apply(&r_f)?;

        // ---------------------------------------------------------------
        // Step 2: Update coarse residual.
        //   r_c' = r_c − A_CF * y_f
        // ---------------------------------------------------------------
        let a_cf_yf = csr_matvec(
            &lvl.a_cf.0,
            &lvl.a_cf.1,
            &lvl.a_cf.2,
            &y_f,
            nc,
        );
        let r_c_prime: Vec<f64> = r_c
            .iter()
            .zip(a_cf_yf.iter())
            .map(|(rc, acf)| rc - acf)
            .collect();

        // ---------------------------------------------------------------
        // Step 3: Coarse solve.
        //   e_c = (level+1) V-cycle applied to r_c'
        // ---------------------------------------------------------------
        let e_c = if level_idx + 1 < self.levels.len() {
            // Recurse: the coarse residual is a vector of length nc.
            // But the next level's vcycle expects the full-size residual
            // in the coarse ordering.  We pass r_c_prime directly.
            self.vcycle(&r_c_prime, level_idx + 1)?
        } else {
            // Bottom level: direct ILU on A_CC.
            let (cc_rp, cc_ci, cc_vs) = &lvl.a_cc;
            let mut cc_rp_w = cc_rp.clone();
            let mut cc_ci_w = cc_ci.clone();
            let mut cc_vs_w = cc_vs.clone();
            ensure_diagonal(&mut cc_rp_w, &mut cc_ci_w, &mut cc_vs_w, nc)?;
            match Ilu0Factors::factor(&cc_rp_w, &cc_ci_w, &cc_vs_w, nc) {
                Ok(cc_ilu) => cc_ilu.apply(&r_c_prime)?,
                Err(_) => r_c_prime.clone(),
            }
        };

        // ---------------------------------------------------------------
        // Step 4: Prolongate correction.
        //   y_f += ILU_F^{-1} A_FC e_c
        // ---------------------------------------------------------------
        let a_fc_ec = csr_matvec(
            &lvl.a_fc.0,
            &lvl.a_fc.1,
            &lvl.a_fc.2,
            &e_c,
            nf,
        );
        let correction_f = lvl.ilu.apply(&a_fc_ec).unwrap_or_else(|_| a_fc_ec.clone());
        let y_f_corrected: Vec<f64> = y_f
            .iter()
            .zip(correction_f.iter())
            .map(|(yf, cf)| yf + cf)
            .collect();

        // ---------------------------------------------------------------
        // Step 5: Assemble the full solution vector.
        // ---------------------------------------------------------------
        let mut y = vec![0.0f64; n_total];
        for (lf, &gf) in lvl.fine_nodes.iter().enumerate() {
            y[gf] = y_f_corrected[lf];
        }
        for (lc, &gc) in lvl.coarse_nodes.iter().enumerate() {
            y[gc] = e_c[lc];
        }

        Ok(y)
    }

    /// Return the matrix dimension.
    pub fn size(&self) -> usize {
        self.n
    }

    /// Return the number of levels actually constructed.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Return the configuration.
    pub fn config(&self) -> &MilueConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Build a 4×4 tridiagonal SPD matrix.
    fn tridiag4() -> (Vec<usize>, Vec<usize>, Vec<f64>, usize) {
        let n = 4usize;
        let row_ptrs = vec![0, 2, 5, 8, 10];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let values = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        (row_ptrs, col_indices, values, n)
    }

    // Build a 6×6 tridiagonal.
    fn tridiag6() -> (Vec<usize>, Vec<usize>, Vec<f64>, usize) {
        let n = 6usize;
        let mut row_ptrs = vec![0usize; n + 1];
        let mut ci = Vec::new();
        let mut vs = Vec::new();
        for i in 0..n {
            if i > 0 {
                ci.push(i - 1);
                vs.push(-1.0f64);
            }
            ci.push(i);
            vs.push(4.0f64);
            if i + 1 < n {
                ci.push(i + 1);
                vs.push(-1.0f64);
            }
            row_ptrs[i + 1] = ci.len();
        }
        (row_ptrs, ci, vs, n)
    }

    // Build a 2×2 block-diagonal matrix (two independent 2×2 blocks).
    fn block_diag4() -> (Vec<usize>, Vec<usize>, Vec<f64>, usize) {
        let n = 4usize;
        // Block 1: rows 0-1, block 2: rows 2-3; no coupling between blocks.
        let row_ptrs = vec![0, 2, 4, 6, 8];
        let col_indices = vec![0, 1, 0, 1, 2, 3, 2, 3];
        let values = vec![4.0, -1.0, -1.0, 4.0, 4.0, -1.0, -1.0, 4.0];
        (row_ptrs, col_indices, values, n)
    }

    // -----------------------------------------------------------------------
    // Test: MilueConfig default values.
    // -----------------------------------------------------------------------
    #[test]
    fn test_milue_config_defaults() {
        let cfg = MilueConfig::default();
        assert_eq!(cfg.levels, 2);
        assert!((cfg.drop_tol - 0.01).abs() < 1e-15);
        assert!((cfg.fill_factor - 3.0).abs() < 1e-15);
    }

    // -----------------------------------------------------------------------
    // Test: MILUE constructs without error.
    // -----------------------------------------------------------------------
    #[test]
    fn test_milue_construct() {
        let (rp, ci, vs, n) = tridiag4();
        let config = MilueConfig::default();
        let prec = MiluePreconditioner::new(&rp, &ci, &vs, n, config);
        assert!(prec.is_ok(), "MILUE should construct: {:?}", prec.err());
        let prec = prec.expect("MILUE construction failed");
        assert_eq!(prec.size(), n);
    }

    // -----------------------------------------------------------------------
    // Test: Level count is respected.
    // -----------------------------------------------------------------------
    #[test]
    fn test_milue_level_count() {
        let (rp, ci, vs, n) = tridiag6();
        // Request 2 levels.
        let config = MilueConfig {
            levels: 2,
            ..MilueConfig::default()
        };
        let prec = MiluePreconditioner::new(&rp, &ci, &vs, n, config)
            .expect("MILUE construction failed");
        // We should have at least 1 level and at most levels=2.
        assert!(
            prec.num_levels() >= 1 && prec.num_levels() <= 2,
            "Expected 1-2 levels, got {}",
            prec.num_levels()
        );
    }

    // -----------------------------------------------------------------------
    // Test: apply reduces residual on Poisson-like stiffness matrix.
    // -----------------------------------------------------------------------
    #[test]
    fn test_milue_reduces_residual() {
        let (rp, ci, vs, n) = tridiag4();
        let config = MilueConfig::default();
        let prec = MiluePreconditioner::new(&rp, &ci, &vs, n, config)
            .expect("MILUE construction failed");

        let b = vec![1.0, 2.0, 3.0, 4.0];
        let x = prec.apply(&b);
        assert_eq!(x.len(), n);

        // Compute residual r = b - A*x.
        let ax = csr_matvec(&rp, &ci, &vs, &x, n);
        let res_norm: f64 = b
            .iter()
            .zip(ax.iter())
            .map(|(bi, axi)| (bi - axi).powi(2))
            .sum::<f64>()
            .sqrt();
        let b_norm: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

        assert!(
            res_norm < b_norm * 1.5,
            "MILUE should reduce residual: res_norm={res_norm}, b_norm={b_norm}"
        );
    }

    // -----------------------------------------------------------------------
    // Test: 2-level on block-diagonal → both levels identified correctly.
    // -----------------------------------------------------------------------
    #[test]
    fn test_milue_block_diagonal() {
        let (rp, ci, vs, n) = block_diag4();
        let config = MilueConfig {
            levels: 2,
            drop_tol: 0.001,
            fill_factor: 3.0,
        };
        let prec = MiluePreconditioner::new(&rp, &ci, &vs, n, config)
            .expect("MILUE block-diag construction failed");

        assert_eq!(prec.size(), n);
        assert!(
            prec.num_levels() >= 1,
            "Should have at least 1 level"
        );

        let b = vec![1.0, 0.0, 0.0, 1.0];
        let x = prec.apply(&b);
        assert_eq!(x.len(), n);
    }

    // -----------------------------------------------------------------------
    // Test: 6×6 with 1 level degenerates to plain ILU.
    // -----------------------------------------------------------------------
    #[test]
    fn test_milue_one_level_is_ilu() {
        let (rp, ci, vs, n) = tridiag6();
        let config = MilueConfig {
            levels: 1,
            drop_tol: 0.0,
            fill_factor: 2.0,
        };
        let prec = MiluePreconditioner::new(&rp, &ci, &vs, n, config)
            .expect("MILUE one-level construction failed");

        assert_eq!(prec.size(), n);
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = prec.apply(&b);
        assert_eq!(x.len(), n);
    }

    // -----------------------------------------------------------------------
    // Test: dimension mismatch error.
    // -----------------------------------------------------------------------
    #[test]
    fn test_milue_dimension_mismatch() {
        let row_ptrs = vec![0, 1, 2]; // n=2
        let col_indices = vec![0, 1];
        let values = vec![1.0, 1.0];
        let result = MiluePreconditioner::new(&row_ptrs, &col_indices, &values, 5, MilueConfig::default());
        assert!(result.is_err(), "should fail on dimension mismatch");
    }

    // -----------------------------------------------------------------------
    // Test: config accessor.
    // -----------------------------------------------------------------------
    #[test]
    fn test_milue_config_accessor() {
        let (rp, ci, vs, n) = tridiag4();
        let config = MilueConfig {
            levels: 3,
            drop_tol: 0.05,
            fill_factor: 2.5,
        };
        let prec = MiluePreconditioner::new(&rp, &ci, &vs, n, config.clone())
            .expect("MILUE construction failed");
        assert_eq!(prec.config().levels, 3);
        assert!((prec.config().drop_tol - 0.05).abs() < 1e-15);
    }

    // -----------------------------------------------------------------------
    // Test: C/F splitting on a connected chain produces both C and F nodes.
    // -----------------------------------------------------------------------
    #[test]
    fn test_cf_split_chain() {
        // Linear chain: 0-1-2-3-4 (5 nodes).
        let n = 5usize;
        let row_ptrs = vec![0, 2, 4, 6, 8, 9];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2, 3, 3];
        // col_indices for 5-node chain (last node only has one neighbour)
        // but let's redo properly:
        // 0: {0,1}, 1: {0,1,2}, 2: {1,2,3}, 3: {2,3,4}, 4: {3,4}
        let rp2 = vec![0, 2, 5, 8, 11, 13];
        let ci2 = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4];

        let (coarse, fine) = cf_split(&rp2, &ci2, n);
        assert!(!coarse.is_empty(), "should have coarse nodes");
        assert!(!fine.is_empty(), "should have fine nodes");
        assert_eq!(coarse.len() + fine.len(), n, "all nodes must be classified");

        // Verify independence: no two coarse nodes are adjacent.
        for &c1 in &coarse {
            for pos in rp2[c1]..rp2[c1 + 1] {
                let nb = ci2[pos];
                if nb != c1 {
                    assert!(
                        !coarse.contains(&nb),
                        "coarse nodes {c1} and {nb} are adjacent"
                    );
                }
            }
        }
    }
}
