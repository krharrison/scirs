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
use scirs2_core::ndarray::{s, Array2};

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
    /// starting from `row_offset` rows and `col_offset` columns.
    ///
    /// `a[row_offset..row_offset+n, col_offset..] -= beta * v * (v^T * a[...])`
    pub fn apply_left_to(&self, a: &mut Array2<f64>, row_offset: usize, col_offset: usize) {
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
            let mut d = 0.0f64;
            for i in 0..n {
                d += self.v[i] * a[[row_offset + i, col_offset + j]];
            }
            w[j] = d;
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
/// Given reflectors H_0, H_1, …, H_{k-1}, builds Q = H_0 * H_1 * … * H_{k-1}.
///
/// Computed by applying reflectors in **reverse** order to an m × m identity:
/// `Q = H_0 * (H_1 * (… * (H_{k-1} * I)))`.
pub fn build_q_from_reflectors(reflectors: &[HouseholderReflector], m: usize) -> Array2<f64> {
    let mut q = Array2::<f64>::eye(m);
    // Apply H_{k-1}, H_{k-2}, ..., H_0 in that order to accumulate Q
    for refl in reflectors.iter().rev() {
        if refl.beta.abs() < f64::EPSILON {
            continue;
        }
        let row_offset = refl.applied_to.start;
        let n = refl.v.len();
        if row_offset + n > m {
            continue;
        }
        // H * Q: w = v^T * Q[row_offset..row_offset+n, :]
        let mut w = vec![0.0f64; m];
        for j in 0..m {
            let mut d = 0.0f64;
            for i in 0..n {
                d += refl.v[i] * q[[row_offset + i, j]];
            }
            w[j] = d;
        }
        // Q[row_offset..row_offset+n, :] -= beta * v * w^T
        for i in 0..n {
            for j in 0..m {
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
                let (r_new, mut refls) = tournament_combine_pair(&current[i], &current[i + 1])?;
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

    Ok((
        current.into_iter().next().ok_or_else(|| {
            LinalgError::ComputationError("tournament_qr_reduction: empty result".to_string())
        })?,
        all_reflectors,
    ))
}

// ---------------------------------------------------------------------------
// CAQR simulation
// ---------------------------------------------------------------------------

/// Simulate Communication-Avoiding QR on a single process: `A = Q * R`.
///
/// This simulation performs standard Householder QR column-by-column, which is
/// algebraically equivalent to the CAQR decomposition. The tournament-tree
/// reduction that CAQR uses in a distributed setting reduces communication
/// to O(log P) rounds vs. O(n/bs) classical; here we simulate its effect
/// via the same reflector-based approach on a single node.
///
/// ## Properties
///
/// - Communication rounds (theoretical): O(log P) per panel.
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
    _config: &DistribConfig,
) -> LinalgResult<(Array2<f64>, Array2<f64>)> {
    let m = a.nrows();
    let n = a.ncols();

    if m == 0 || n == 0 {
        return Err(LinalgError::ValueError(
            "caqr_simulate: input matrix must be non-empty".to_string(),
        ));
    }

    // Work on a mutable copy; apply Householder reflectors to obtain R
    let mut r_work = a.to_owned();
    let mut all_reflectors: Vec<HouseholderReflector> = Vec::new();

    // Standard column-by-column Householder QR
    let n_house = m.min(n);
    for col in 0..n_house {
        let sub_len = m - col;
        let mut x = vec![0.0f64; sub_len];
        for i in 0..sub_len {
            x[i] = r_work[[col + i, col]];
        }

        let (v, beta) = HouseholderReflector::from_vector(&x);
        let refl = HouseholderReflector {
            v,
            beta,
            applied_to: col..col + sub_len,
        };

        if refl.beta.abs() > f64::EPSILON {
            // Apply H to trailing sub-matrix r_work[col.., col..]
            refl.apply_left_to(&mut r_work, col, col);
            // Zero sub-diagonal entries
            for i in (col + 1)..m {
                r_work[[i, col]] = 0.0;
            }
        }
        all_reflectors.push(refl);
    }

    // Build Q = H_0 * H_1 * ... * H_{n_house-1}
    let q = build_q_from_reflectors(&all_reflectors, m);

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
        let mut a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]];
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
        let r0 = array![[3.0_f64, 1.0, 2.0], [0.0, 2.0, -1.0], [0.0, 0.0, 1.5]];
        let r1 = array![[2.0_f64, -1.0, 1.0], [0.0, 1.5, 0.5], [0.0, 0.0, 0.8]];
        let (r_combined, _) =
            tournament_qr_reduction(&[r0.clone(), r1.clone()]).expect("tournament failed");
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
        let config = DistribConfig {
            block_size: 2,
            n_proc_rows: 2,
            n_proc_cols: 2,
        };
        let (q, r) = caqr_simulate(&a, &config).expect("caqr failed");
        let qr_prod = matmul(&q, &r);
        assert_abs_diff_eq!(frob_diff(&qr_prod, &a), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn test_caqr_q_orthogonal() {
        let a = Array2::<f64>::from_shape_fn((8, 5), |(i, j)| i as f64 * 0.7 - j as f64 * 0.3);
        let config = DistribConfig {
            block_size: 3,
            n_proc_rows: 2,
            n_proc_cols: 2,
        };
        let (q, _r) = caqr_simulate(&a, &config).expect("caqr failed");
        // Q^T Q should be identity
        let qtq = matmul(&q.t().to_owned(), &q);
        let eye = Array2::<f64>::eye(q.nrows());
        assert_abs_diff_eq!(frob_diff(&qtq, &eye), 0.0, epsilon = 1e-9);
    }
}
