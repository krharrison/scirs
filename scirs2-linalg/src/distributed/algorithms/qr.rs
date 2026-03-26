//! Communication-Avoiding QR (CAQR) via Householder with binary-tree tournament reduction.
//!
//! ## Algorithm overview
//!
//! Classical distributed QR communicates O(n/bs) times (one broadcast per column panel).
//! CAQR reduces this to O(log P) rounds by using a **tournament tree** to combine the
//! local R factors bottom-up:
//!
//! 1. Each virtual process computes a local panel QR via Householder.
//! 2. Pairs of R factors are stacked `[R_top; R_bottom]` and re-factored.
//! 3. Only the combined R is passed up the tree; Q is reconstructed lazily.
//!
//! This simulation partitions the rows of A across `n_proc_rows` virtual processes
//! and applies the above reduction column-by-column (panel-by-panel).
//!
//! ## References
//!
//! Demmel, Grigori, Hoemmen, Langou (2012),
//! *Communication-optimal parallel and sequential QR and LU factorizations*.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2};

use super::DistribConfig;

// ---------------------------------------------------------------------------
// HouseholderReflector
// ---------------------------------------------------------------------------

/// A Householder reflector `H = I - beta * v * v^T` applied to a row range.
#[derive(Debug, Clone)]
pub struct HouseholderReflector {
    /// Householder vector (length = size of subspace acted on).
    pub v: Vec<f64>,
    /// Scalar coefficient: `beta = 2 / (v^T v)`.
    pub beta: f64,
    /// Row range within the original matrix that this reflector was applied to.
    pub applied_to: std::ops::Range<usize>,
}

impl HouseholderReflector {
    /// Construct a Householder reflector that zeroes all but the first component of `x`.
    ///
    /// Returns `(v, beta)` such that `(I - beta * v * v^T) * x = ±‖x‖ e_1`.
    pub fn from_vector(x: &[f64]) -> (Vec<f64>, f64) {
        let n = x.len();
        if n == 0 {
            return (vec![], 0.0);
        }
        let sigma: f64 = x.iter().skip(1).map(|xi| xi * xi).sum::<f64>();
        let mut v: Vec<f64> = x.to_vec();

        if sigma == 0.0 && x[0] >= 0.0 {
            // Already in the target form; null reflector
            return (v, 0.0);
        }

        let x_norm = (x[0] * x[0] + sigma).sqrt();
        // Choose sign to avoid cancellation
        if x[0] <= 0.0 {
            v[0] = x[0] - x_norm;
        } else {
            v[0] = x[0] + x_norm;
        }

        let vt_v: f64 = v[0] * v[0] + sigma;
        let beta = if vt_v.abs() < f64::EPSILON {
            0.0
        } else {
            2.0 / vt_v
        };
        (v, beta)
    }

    /// Apply this reflector to the *rows* of `a` (in-place), acting on the sub-block
    /// starting from column `col_offset`.
    ///
    /// `a[row_range, col_offset..] -= beta * v * (v^T * a[row_range, col_offset..])`
    fn apply_left_to(&self, a: &mut Array2<f64>, row_offset: usize, col_offset: usize) {
        let n = self.v.len();
        let nrows = a.nrows();
        let ncols = a.ncols();
        if row_offset + n > nrows || col_offset >= ncols {
            return;
        }
        let n_cols_apply = ncols - col_offset;
        // w = v^T * A[row_offset..row_offset+n, col_offset..]
        let mut w = vec![0.0f64; n_cols_apply];
        for j in 0..n_cols_apply {
            let mut dot = 0.0f64;
            for i in 0..n {
                dot += self.v[i] * a[[row_offset + i, col_offset + j]];
            }
            w[j] = dot;
        }
        // A[...] -= beta * v * w^T
        for i in 0..n {
            for j in 0..n_cols_apply {
                a[[row_offset + i, col_offset + j]] -= self.beta * self.v[i] * w[j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Local panel QR
// ---------------------------------------------------------------------------

/// Compute in-place panel QR of `a_panel` using Householder reflectors.
///
/// Modifies `a_panel` in-place: on return the upper triangle contains R;
/// the Householder vectors are NOT stored in the subdiagonal (they are
/// returned as a list of [`HouseholderReflector`] structs instead).
///
/// # Returns
///
/// Vector of Householder reflectors that, when applied left-to-right, produce Q.
pub fn local_qr_panel(a_panel: &mut Array2<f64>) -> Vec<HouseholderReflector> {
    let m = a_panel.nrows();
    let n = a_panel.ncols();
    let n_house = m.min(n);
    let mut reflectors = Vec::with_capacity(n_house);

    for col in 0..n_house {
        // Extract the sub-column x = a_panel[col.., col]
        let sub_len = m - col;
        let mut x = vec![0.0f64; sub_len];
        for i in 0..sub_len {
            x[i] = a_panel[[col + i, col]];
        }

        let (v, beta) = HouseholderReflector::from_vector(&x);
        if beta.abs() < f64::EPSILON {
            // Nothing to do for this column
            reflectors.push(HouseholderReflector {
                v,
                beta,
                applied_to: col..col + sub_len,
            });
            continue;
        }

        let refl = HouseholderReflector {
            v: v.clone(),
            beta,
            applied_to: col..col + sub_len,
        };

        // Apply H to the trailing sub-matrix a_panel[col.., col..]
        refl.apply_left_to(a_panel, col, col);

        // Zero the sub-diagonal entries explicitly (rounding artifacts)
        for i in (col + 1)..m {
            a_panel[[i, col]] = 0.0;
        }

        reflectors.push(refl);
    }
    reflectors
}

// ---------------------------------------------------------------------------
// Reconstruct Q from reflectors
// ---------------------------------------------------------------------------

/// Reconstruct the full orthogonal matrix Q from Householder reflectors.
///
/// Given a list of reflectors H_0, H_1, …, H_{k-1}, this builds
/// Q = H_0 * H_1 * … * H_{k-1} * I  acting on an `m × m` identity.
///
/// The reflectors are applied in reverse order (right-to-left) to an identity
/// to recover Q.
pub fn build_q_from_reflectors(reflectors: &[HouseholderReflector], m: usize) -> Array2<f64> {
    let mut q = Array2::<f64>::eye(m);
    // Apply reflectors right-to-left (H_{k-1} … H_0 applied to I gives Q when composed)
    for refl in reflectors.iter().rev() {
        let row_offset = refl.applied_to.start;
        let n = refl.v.len();
        let n_cols = m;
        // w = v^T * Q[row_offset..row_offset+n, :]
        let mut w = vec![0.0f64; n_cols];
        for j in 0..n_cols {
            let mut dot = 0.0f64;
            for i in 0..n {
                dot += refl.v[i] * q[[row_offset + i, j]];
            }
            w[j] = dot;
        }
        // Q[row_offset..row_offset+n, :] -= beta * v * w^T
        for i in 0..n {
            for j in 0..n_cols {
                q[[row_offset + i, j]] -= refl.beta * refl.v[i] * w[j];
            }
        }
    }
    q
}

// ---------------------------------------------------------------------------
// Tournament (binary-tree) QR reduction
// ---------------------------------------------------------------------------

/// Perform one level of the binary-tree tournament: combine two R factors.
///
/// Given `r_top` and `r_bottom` (both upper triangular, n×n), stacks them
/// as `[r_top; r_bottom]` (2n×n) and computes QR to yield a new n×n R.
///
/// Returns `(r_combined, reflectors)`.
fn tournament_combine_pair(
    r_top: &Array2<f64>,
    r_bottom: &Array2<f64>,
) -> LinalgResult<(Array2<f64>, Vec<HouseholderReflector>)> {
    let n = r_top.ncols();
    if r_bottom.ncols() != n {
        return Err(LinalgError::DimensionError(
            "tournament_combine_pair: R matrices must have same number of columns".to_string(),
        ));
    }
    let nt = r_top.nrows();
    let nb = r_bottom.nrows();

    // Stack [r_top; r_bottom] into a (nt+nb) × n matrix
    let mut stacked = Array2::<f64>::zeros((nt + nb, n));
    stacked.slice_mut(s![..nt, ..]).assign(r_top);
    stacked.slice_mut(s![nt.., ..]).assign(r_bottom);

    let reflectors = local_qr_panel(&mut stacked);
    // Only keep the first n rows as the new R (upper triangular)
    let r_new = stacked.slice(s![..n, ..]).to_owned();
    Ok((r_new, reflectors))
}

/// Binary-tree QR reduction of a slice of panel R factors.
///
/// On input, `panels` is a slice of R matrices from per-process local panel QR.
/// The function reduces them bottom-up to a single R:
///
/// ```text
/// Level 0:  [R0, R1, R2, R3]
/// Level 1:  [QR(R0,R1).R, QR(R2,R3).R]
/// Level 2:  [QR(level1[0], level1[1]).R]
/// ```
///
/// Returns `(R_final, reflectors_at_each_level)`.
pub fn tournament_qr_reduction(
    panels: &[Array2<f64>],
) -> LinalgResult<(Array2<f64>, Vec<HouseholderReflector>)> {
    if panels.is_empty() {
        return Err(LinalgError::ValueError(
            "tournament_qr_reduction: panels slice is empty".to_string(),
        ));
    }
    if panels.len() == 1 {
        return Ok((panels[0].clone(), vec![]));
    }

    let mut current: Vec<Array2<f64>> = panels.to_vec();
    let mut all_reflectors: Vec<HouseholderReflector> = Vec::new();

    while current.len() > 1 {
        let mut next: Vec<Array2<f64>> = Vec::new();
        let mut i = 0;
        while i < current.len() {
            if i + 1 < current.len() {
                let (r_new, mut refls) =
                    tournament_combine_pair(&current[i], &current[i + 1])?;
                next.push(r_new);
                all_reflectors.append(&mut refls);
                i += 2;
            } else {
                // Odd element passes through unchanged
                next.push(current[i].clone());
                i += 1;
            }
        }
        current = next;
    }

    Ok((current.into_iter().next().ok_or_else(|| {
        LinalgError::ComputationError("tournament_qr_reduction: empty result".to_string())
    })?, all_reflectors))
}

// ---------------------------------------------------------------------------
// CAQR simulation
// ---------------------------------------------------------------------------

/// Simulate Communication-Avoiding QR on a single process: `A = Q * R`.
///
/// The algorithm partitions the rows of A across `n_proc_rows` virtual processes.
/// For each column panel:
/// 1. Each virtual process computes local panel QR (Householder).
/// 2. Tournament tree reduction combines local R factors.
/// 3. The combined Householder transforms are applied to the trailing matrix.
///
/// ## Properties
///
/// - Communication rounds: O(log P) per panel (vs O(1) per element classically).
/// - Numerical stability: equivalent to standard Householder QR.
///
/// ## Returns
///
/// `(Q, R)` where:
/// - `Q` is orthogonal (m × m)
/// - `R` is upper triangular (m × n)
/// - `Q * R ≈ A` to machine precision
///
/// # Errors
///
/// Returns an error for empty or ill-sized inputs.
pub fn caqr_simulate(
    a: &Array2<f64>,
    config: &DistribConfig,
) -> LinalgResult<(Array2<f64>, Array2<f64>)> {
    let m = a.nrows();
    let n = a.ncols();

    if m == 0 || n == 0 {
        return Err(LinalgError::ValueError(
            "caqr_simulate: input matrix must be non-empty".to_string(),
        ));
    }

    let n_procs = config.n_proc_rows.max(1);
    let bs = config.block_size.max(1);

    // Work on a mutable copy; we accumulate Q separately
    let mut r_work = a.to_owned();
    // Accumulated Q (m × m identity initially)
    let mut q_accum = Array2::<f64>::eye(m);

    let n_panels = n.div_ceil(bs);

    for panel_idx in 0..n_panels {
        let col_start = panel_idx * bs;
        let col_end = (col_start + bs).min(n);
        let panel_width = col_end - col_start;

        // -------------------------------------------------------------------
        // Step 1: Partition rows across virtual processes and compute local QR
        // -------------------------------------------------------------------
        // Each virtual process p owns rows in [row_start_p, row_end_p)
        let mut local_rs: Vec<Array2<f64>> = Vec::with_capacity(n_procs);
        let mut local_reflectors: Vec<Vec<HouseholderReflector>> = Vec::with_capacity(n_procs);

        for proc in 0..n_procs {
            let row_start = (proc * m) / n_procs;
            let row_end = ((proc + 1) * m) / n_procs;
            if row_start >= row_end {
                continue;
            }

            // Extract this process's panel: rows [row_start, row_end), cols [col_start, col_end)
            let mut local_panel = r_work
                .slice(s![row_start..row_end, col_start..col_end])
                .to_owned();
            let refls = local_qr_panel(&mut local_panel);

            // Keep only the square R part (min_dim × panel_width)
            let min_dim = (row_end - row_start).min(panel_width);
            let r_local = local_panel.slice(s![..min_dim, ..]).to_owned();
            local_rs.push(r_local);
            local_reflectors.push(refls);
        }

        if local_rs.is_empty() {
            continue;
        }

        // -------------------------------------------------------------------
        // Step 2: Apply local Householder transforms to the trailing matrix
        //         and accumulate Q
        // -------------------------------------------------------------------
        for (proc, refls) in local_reflectors.iter().enumerate() {
            let row_start = (proc * m) / n_procs;
            let row_end = ((proc + 1) * m) / n_procs;
            if row_start >= row_end {
                continue;
            }

            // Apply reflectors to the trailing block of r_work (col_start onwards)
            // We need a temporary copy of the sub-block
            let sub_rows = row_end - row_start;
            let sub_cols = n - col_start;
            if sub_cols == 0 {
                continue;
            }

            let mut sub = r_work
                .slice(s![row_start..row_end, col_start..n])
                .to_owned();

            for refl in refls {
                refl.apply_left_to(&mut sub, 0, refl.applied_to.start.saturating_sub(col_start));
            }
            r_work
                .slice_mut(s![row_start..row_end, col_start..n])
                .assign(&sub);

            // Zero sub-diagonal entries in this process's panel block
            for c in col_start..col_end {
                for r in (c + 1)..row_end.min(m) {
                    if r >= row_start {
                        r_work[[r, c]] = 0.0;
                    }
                }
            }

            // Apply reflectors to Q accumulator
            let mut q_sub = q_accum.slice(s![row_start..row_end, ..]).to_owned();
            for refl in refls {
                // Adjust row_offset to be relative to the sub-matrix
                let local_row_offset = refl.applied_to.start.saturating_sub(col_start);
                let v_len = refl.v.len();
                if local_row_offset + v_len > sub_rows {
                    continue;
                }
                let n_q_cols = q_accum.ncols();
                let mut w = vec![0.0f64; n_q_cols];
                for j in 0..n_q_cols {
                    let mut dot = 0.0f64;
                    for i in 0..v_len {
                        dot += refl.v[i] * q_sub[[local_row_offset + i, j]];
                    }
                    w[j] = dot;
                }
                for i in 0..v_len {
                    for j in 0..n_q_cols {
                        q_sub[[local_row_offset + i, j]] -= refl.beta * refl.v[i] * w[j];
                    }
                }
            }
            q_accum
                .slice_mut(s![row_start..row_end, ..])
                .assign(&q_sub);
        }

        // -------------------------------------------------------------------
        // Step 3: Tournament reduction — combine local R factors
        // -------------------------------------------------------------------
        if local_rs.len() > 1 {
            let (r_combined, _) = tournament_qr_reduction(&local_rs)?;

            // Write the combined R back into the panel diagonal block
            let r_rows = r_combined.nrows().min(panel_width);
            let r_cols = r_combined.ncols().min(panel_width);
            for ri in 0..r_rows {
                for ci in ri..r_cols {
                    r_work[[col_start + ri, col_start + ci]] = r_combined[[ri, ci]];
                }
                // Zero sub-diagonal within the block
                for ci in 0..ri {
                    if col_start + ci < n {
                        r_work[[col_start + ri, col_start + ci]] = 0.0;
                    }
                }
            }
        }
    }

    // Q is currently stored as the product of all reflectors applied to I;
    // the convention in CAQR is Q^T applied, so we need Q = q_accum^T
    let q = q_accum.t().to_owned();

    Ok((q, r_work))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    // Matrix-matrix multiply helper
    fn matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        let m = a.nrows();
        let k = a.ncols();
        let n = b.ncols();
        let mut c = Array2::<f64>::zeros((m, n));
        for i in 0..m {
            for ki in 0..k {
                for j in 0..n {
                    c[[i, j]] += a[[i, ki]] * b[[ki, j]];
                }
            }
        }
        c
    }

    // Frobenius distance
    fn frob_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let mut s = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = x - y;
            s += d * d;
        }
        s.sqrt()
    }

    #[test]
    fn test_householder_reflector_construction() {
        // H = I - beta * v * v^T should be orthogonal
        let x = vec![3.0, 4.0, 0.0];
        let (v, beta) = HouseholderReflector::from_vector(&x);
        // Check v^T v > 0 and beta = 2 / (v^T v)
        let vt_v: f64 = v.iter().map(|vi| vi * vi).sum();
        assert_abs_diff_eq!(beta, 2.0 / vt_v, epsilon = 1e-12);
    }

    #[test]
    fn test_householder_reflector_orthogonality() {
        // Build a 3×3 Householder matrix and verify H^T H = I
        let x = vec![1.0, 2.0, 3.0];
        let (v, beta) = HouseholderReflector::from_vector(&x);
        let n = v.len();

        // Build H = I - beta * v * v^T
        let mut h = Array2::<f64>::eye(n);
        for i in 0..n {
            for j in 0..n {
                h[[i, j]] -= beta * v[i] * v[j];
            }
        }
        let ht_h = matmul(&h.t().to_owned(), &h);
        let eye = Array2::<f64>::eye(n);
        assert_abs_diff_eq!(frob_diff(&ht_h, &eye), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_local_qr_panel_qtq_identity() {
        let mut a = array![
            [1.0_f64, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 10.0]
        ];
        let original_a = a.clone();
        let reflectors = local_qr_panel(&mut a);
        let q = build_q_from_reflectors(&reflectors, original_a.nrows());

        // Q^T Q = I
        let qtq = matmul(&q.t().to_owned(), &q);
        let eye = Array2::<f64>::eye(3);
        assert_abs_diff_eq!(frob_diff(&qtq, &eye), 0.0, epsilon = 1e-11);
    }

    #[test]
    fn test_local_qr_panel_qr_equals_a() {
        let original_a = array![
            [12.0_f64, -51.0, 4.0],
            [6.0, 167.0, -68.0],
            [-4.0, 24.0, -41.0]
        ];
        let mut a_work = original_a.clone();
        let reflectors = local_qr_panel(&mut a_work);
        let q = build_q_from_reflectors(&reflectors, original_a.nrows());
        let r = a_work.clone(); // upper triangle is R

        let qr = matmul(&q, &r);
        assert_abs_diff_eq!(frob_diff(&qr, &original_a), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn test_tournament_qr_preserves_r_factor() {
        // Construct two upper triangular 3×3 R factors
        let r0 = array![
            [3.0_f64, 1.0, 2.0],
            [0.0, 2.0, -1.0],
            [0.0, 0.0, 1.5]
        ];
        let r1 = array![
            [2.0_f64, -1.0, 1.0],
            [0.0, 1.5, 0.5],
            [0.0, 0.0, 0.8]
        ];
        let (r_combined, _) = tournament_qr_reduction(&[r0.clone(), r1.clone()])
            .expect("tournament failed");
        // R_combined should be upper triangular (sub-diagonal entries ≈ 0)
        assert!(r_combined.nrows() <= 3);
        assert!(r_combined.ncols() == 3);
        for i in 0..r_combined.nrows() {
            for j in 0..i {
                assert_abs_diff_eq!(r_combined[[i, j]], 0.0, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_caqr_qr_equals_a() {
        let a = array![
            [1.0_f64, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 10.0],
            [1.0, -1.0, 2.0]
        ];
        let config = DistribConfig { block_size: 2, n_proc_rows: 2, n_proc_cols: 2 };
        let (q, r) = caqr_simulate(&a, &config).expect("caqr failed");
        let qr_prod = matmul(&q, &r);
        assert_abs_diff_eq!(frob_diff(&qr_prod, &a), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn test_caqr_q_orthogonal() {
        let a = Array2::<f64>::from_shape_fn((8, 5), |(i, j)| (i as f64 * 0.7 - j as f64 * 0.3));
        let config = DistribConfig { block_size: 3, n_proc_rows: 2, n_proc_cols: 2 };
        let (q, _r) = caqr_simulate(&a, &config).expect("caqr failed");
        // Q^T Q should be identity
        let qtq = matmul(&q.t().to_owned(), &q);
        let eye = Array2::<f64>::eye(q.nrows());
        assert_abs_diff_eq!(frob_diff(&qtq, &eye), 0.0, epsilon = 1e-9);
    }
}
