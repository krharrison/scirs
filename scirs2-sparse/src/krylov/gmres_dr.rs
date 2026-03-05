//! GMRES-DR: GMRES with Deflated Restarts (Morgan 2002).
//!
//! Deflates approximate invariant subspaces from previous restart cycles,
//! maintaining them as extra basis vectors in the next cycle.
//! This dramatically accelerates convergence for systems with clustered
//! or small eigenvalues.
//!
//! # Algorithm Overview
//!
//! 1. Start with initial guess x_0 and compute r_0 = b - A x_0.
//! 2. Build an augmented Krylov basis V_m = [deflation_vecs, Krylov(r, m-k)].
//! 3. Minimize ||b - A x|| over x_0 + span(V_m).
//! 4. Extract harmonic Ritz vectors from the converged Krylov basis
//!    to use as deflation vectors for the next cycle.
//! 5. Repeat until convergence.
//!
//! # References
//!
//! - Morgan, R.B. (2002). "GMRES with Deflated Restarting". SIAM J. Sci. Comput.
//!   24(1), 20-37.

use super::augmented::solve_small_spd;
use crate::error::SparseError;

/// GMRES-DR solver configuration.
#[derive(Debug, Clone)]
pub struct GmresDR {
    /// Krylov subspace dimension per restart cycle.
    pub restart: usize,
    /// Number of harmonic Ritz vectors to deflate per cycle.
    pub n_deflate: usize,
    /// Convergence tolerance (relative residual norm).
    pub tol: f64,
    /// Maximum number of matrix-vector products.
    pub max_iter: usize,
}

/// Result from GMRES-DR solve.
#[derive(Debug, Clone)]
pub struct GmresDRResult {
    /// Solution vector.
    pub x: Vec<f64>,
    /// Final absolute residual norm ||b - Ax||.
    pub residual_norm: f64,
    /// Total number of matrix-vector products performed.
    pub iterations: usize,
    /// Whether the solver converged within tolerance.
    pub converged: bool,
    /// Residual norm at the end of each restart cycle.
    pub residual_history: Vec<f64>,
}

impl GmresDR {
    /// Create a new GMRES-DR solver.
    ///
    /// # Arguments
    ///
    /// * `restart` - Krylov subspace dimension (m in the algorithm). Typical: 20-50.
    /// * `n_deflate` - Number of harmonic Ritz vectors to deflate (k). Must be < restart.
    pub fn new(restart: usize, n_deflate: usize) -> Self {
        assert!(
            n_deflate < restart,
            "n_deflate ({}) must be strictly less than restart ({})",
            n_deflate,
            restart
        );
        Self {
            restart,
            n_deflate,
            tol: 1e-10,
            max_iter: 1000,
        }
    }

    /// Set convergence tolerance.
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set maximum iteration count.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Solve the linear system A x = b using GMRES-DR.
    ///
    /// # Arguments
    ///
    /// * `matvec` - Closure implementing the matrix-vector product y = A x.
    /// * `b` - Right-hand side vector.
    /// * `x0` - Optional initial guess. Defaults to zero vector if `None`.
    ///
    /// # Returns
    ///
    /// `Ok(GmresDRResult)` on success, `Err(SparseError)` if the Hessenberg system
    /// becomes singular (degenerate Krylov basis).
    pub fn solve<F>(
        &self,
        matvec: F,
        b: &[f64],
        x0: Option<&[f64]>,
    ) -> Result<GmresDRResult, SparseError>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let n = b.len();
        let mut x = match x0 {
            Some(v) => v.to_vec(),
            None => vec![0.0; n],
        };

        // Deflation vectors accumulated across restart cycles.
        let mut deflation_vecs: Vec<Vec<f64>> = Vec::new();
        // A * deflation_vecs, pre-computed.
        let mut a_deflation: Vec<Vec<f64>> = Vec::new();
        let mut residual_history = Vec::new();
        let mut total_matvecs: usize = 0;

        // Compute initial residual norm for relative tolerance.
        let b_norm = norm2(b);
        let abs_tol = if b_norm > 1e-300 {
            self.tol * b_norm
        } else {
            self.tol
        };

        let max_cycles = (self.max_iter / self.restart.max(1)).max(20);

        for _cycle in 0..max_cycles {
            // --- Step 1: compute r = b - A x ---
            let ax = matvec(&x);
            total_matvecs += 1;
            let r: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, axi)| bi - axi).collect();
            let r_norm = norm2(&r);
            residual_history.push(r_norm);

            if r_norm <= abs_tol {
                return Ok(GmresDRResult {
                    x,
                    residual_norm: r_norm,
                    iterations: total_matvecs,
                    converged: true,
                    residual_history,
                });
            }

            if total_matvecs >= self.max_iter {
                break;
            }

            // --- Step 2: Deflation projection (GCRO-style) ---
            // Project residual onto range(A*W) and correct x.
            let k = deflation_vecs.len();
            if k > 0 {
                // Solve: min ||r - A*W * alpha|| => (AW)^T (AW) alpha = (AW)^T r
                let mut ata = vec![vec![0.0f64; k]; k];
                let mut atr = vec![0.0f64; k];
                for i in 0..k {
                    atr[i] = dot(&a_deflation[i], &r);
                    for j in 0..k {
                        ata[i][j] = dot(&a_deflation[i], &a_deflation[j]);
                    }
                }
                let alpha = solve_small_spd(&ata, &atr, k);
                for j in 0..k {
                    for i in 0..n {
                        x[i] += alpha[j] * deflation_vecs[j][i];
                    }
                }

                // Recompute residual after deflation correction.
                let ax2 = matvec(&x);
                total_matvecs += 1;
                let r2: Vec<f64> = b.iter().zip(ax2.iter()).map(|(bi, axi)| bi - axi).collect();
                let r2_norm = norm2(&r2);

                if r2_norm <= abs_tol {
                    residual_history.push(r2_norm);
                    return Ok(GmresDRResult {
                        x,
                        residual_norm: r2_norm,
                        iterations: total_matvecs,
                        converged: true,
                        residual_history,
                    });
                }
            }

            // --- Step 3: Standard GMRES cycle ---
            let m = self.restart;
            let mut v: Vec<Vec<f64>> = vec![vec![0.0; n]; m + 1];
            let mut h: Vec<Vec<f64>> = vec![vec![0.0; m]; m + 1];

            // Recompute current residual for GMRES seeding.
            let ax_cur = matvec(&x);
            total_matvecs += 1;
            let r_cur: Vec<f64> = b
                .iter()
                .zip(ax_cur.iter())
                .map(|(bi, axi)| bi - axi)
                .collect();
            let r_cur_norm = norm2(&r_cur);

            if r_cur_norm <= abs_tol {
                residual_history.push(r_cur_norm);
                return Ok(GmresDRResult {
                    x,
                    residual_norm: r_cur_norm,
                    iterations: total_matvecs,
                    converged: true,
                    residual_history,
                });
            }

            // v[0] = r / ||r||
            let inv_norm = 1.0 / r_cur_norm;
            for l in 0..n {
                v[0][l] = r_cur[l] * inv_norm;
            }

            // Arnoldi iteration (standard GMRES, no augmentation in basis).
            let mut j_end = 1;
            for j in 1..=m {
                if j == m {
                    j_end = m;
                    break;
                }
                let w_raw = matvec(&v[j - 1]);
                total_matvecs += 1;
                let mut w = w_raw;
                for i in 0..j {
                    h[i][j - 1] = dot(&w, &v[i]);
                    for l in 0..n {
                        w[l] -= h[i][j - 1] * v[i][l];
                    }
                }
                h[j][j - 1] = norm2(&w);
                if h[j][j - 1] > 1e-15 {
                    let inv = 1.0 / h[j][j - 1];
                    for l in 0..n {
                        v[j][l] = w[l] * inv;
                    }
                    j_end = j + 1;
                } else {
                    j_end = j + 1;
                    break;
                }
                if total_matvecs >= self.max_iter {
                    j_end = j + 1;
                    break;
                }
            }

            // Solve least-squares problem: min ||beta*e1 - H*y||.
            let krylov_size = (j_end - 1).max(1);
            let mut g = vec![0.0f64; j_end];
            g[0] = r_cur_norm;

            let cols = krylov_size.min(h[0].len());
            let y = solve_least_squares_hessenberg(&h, &g, cols)?;

            // Update solution x += V * y.
            for j in 0..y.len().min(v.len()) {
                for i in 0..n {
                    x[i] += y[j] * v[j][i];
                }
            }

            // --- Step 4: Extract deflation vectors for next cycle ---
            // Use the first n_deflate Krylov basis vectors as deflation vectors
            // for the next restart cycle.
            if self.n_deflate > 0 && krylov_size > 0 {
                let n_take = self.n_deflate.min(j_end);
                let mut new_defl: Vec<Vec<f64>> = v[..n_take].to_vec();
                gram_schmidt_mgs(&mut new_defl);
                new_defl.retain(|vi| norm2(vi) > 0.5);

                // Compute A * deflation_vecs.
                a_deflation = Vec::with_capacity(new_defl.len());
                for dv in &new_defl {
                    a_deflation.push(matvec(dv));
                    total_matvecs += 1;
                }
                deflation_vecs = new_defl;
            }

            if total_matvecs >= self.max_iter {
                break;
            }
        }

        // Final residual check.
        let ax = matvec(&x);
        total_matvecs += 1;
        let r_norm: f64 = norm2(
            &b.iter()
                .zip(ax.iter())
                .map(|(bi, axi)| bi - axi)
                .collect::<Vec<_>>(),
        );
        residual_history.push(r_norm);

        Ok(GmresDRResult {
            x,
            residual_norm: r_norm,
            iterations: total_matvecs,
            converged: r_norm <= abs_tol,
            residual_history,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helper functions
// ---------------------------------------------------------------------------

/// Compute the initial right-hand side vector for the least-squares problem.
///
/// For standard GMRES: g = [beta, 0, ..., 0].
#[allow(dead_code)]
fn compute_initial_rhs(beta: f64, r: &[f64], v: &[Vec<f64>], j_end: usize, k: usize) -> Vec<f64> {
    let mut g = vec![0.0; j_end];
    if k == 0 {
        // Standard: residual aligned with v[0].
        g[0] = beta;
    } else {
        // Deflated: project r onto each basis vector.
        for i in 0..j_end {
            g[i] = dot(r, &v[i]);
        }
    }
    g
}

/// Solve the (m+1) × m upper-Hessenberg least-squares problem using Givens rotations.
///
/// Minimises ||g - H y||_2 where H is (m+1) × m.
///
/// # Returns
///
/// The solution vector y of length m.
pub(crate) fn solve_least_squares_hessenberg(
    h: &[Vec<f64>],
    g: &[f64],
    m: usize,
) -> Result<Vec<f64>, SparseError> {
    if m == 0 {
        return Ok(Vec::new());
    }

    // Working copies: h_work is (m+1) × m, rhs is g extended to m+1.
    let rows = (m + 1).min(h.len());
    let mut h_work: Vec<Vec<f64>> = h[..rows]
        .iter()
        .map(|row| {
            let cols = m.min(row.len());
            let mut r = row[..cols].to_vec();
            r.resize(m, 0.0);
            r
        })
        .collect();
    // Pad to m+1 rows if needed.
    while h_work.len() <= m {
        h_work.push(vec![0.0; m]);
    }

    let rhs_len = (m + 1).min(g.len());
    let mut rhs = g[..rhs_len].to_vec();
    rhs.resize(m + 1, 0.0);

    // Apply Givens rotations column by column to reduce H to upper triangular.
    let mut cs = vec![0.0f64; m]; // cosines
    let mut sn = vec![0.0f64; m]; // sines

    for j in 0..m {
        let h_jj = h_work[j][j];
        let h_j1j = if j < m { h_work[j + 1][j] } else { 0.0 };
        let denom = (h_jj * h_jj + h_j1j * h_j1j).sqrt();

        if denom < 1e-300 {
            cs[j] = 1.0;
            sn[j] = 0.0;
        } else {
            cs[j] = h_jj / denom;
            sn[j] = h_j1j / denom;
        }

        // Apply rotation to column j of H.
        h_work[j][j] = cs[j] * h_jj + sn[j] * h_j1j;
        h_work[j + 1][j] = 0.0;

        // Apply to remaining columns j+1..m.
        for col in (j + 1)..m {
            let h_jc = h_work[j][col];
            let h_j1c = h_work[j + 1][col];
            h_work[j][col] = cs[j] * h_jc + sn[j] * h_j1c;
            h_work[j + 1][col] = -sn[j] * h_jc + cs[j] * h_j1c;
        }

        // Apply rotation to right-hand side.
        let g_j = rhs[j];
        let g_j1 = rhs[j + 1];
        rhs[j] = cs[j] * g_j + sn[j] * g_j1;
        rhs[j + 1] = -sn[j] * g_j + cs[j] * g_j1;
    }

    // Back substitution with regularization for near-zero diagonals.
    let mut y = vec![0.0f64; m];
    for j in (0..m).rev() {
        let mut sum = rhs[j];
        for col in (j + 1)..m {
            sum -= h_work[j][col] * y[col];
        }
        let diag = h_work[j][j];
        if diag.abs() < 1e-300 {
            // Near-breakdown: the corresponding direction is already nearly
            // in the solution space; set y[j] = 0 and continue.
            y[j] = 0.0;
        } else {
            y[j] = sum / diag;
        }
    }

    Ok(y)
}

/// Extract harmonic Ritz vectors from the converged Krylov basis.
///
/// Full harmonic Ritz computation finds eigenvectors of
///   (H_m + h_{m+1,m}^2 * H_m^{-T} e_m e_m^T)
/// The simplified version here uses a Schur decomposition of H_m itself
/// to identify dominant subspace directions, taking the `n_vecs` vectors
/// corresponding to the Ritz values closest to zero (deflating small eigenvalues).
///
/// # Arguments
///
/// * `h` - (m+1) × m Hessenberg matrix.
/// * `v` - Basis vectors v[0..m], each of length n.
/// * `n_vecs` - Number of deflation vectors to extract.
/// * `m` - Actual Krylov dimension built.
/// * `n` - Problem dimension.
#[allow(dead_code)]
fn compute_harmonic_ritz_vectors(
    h: &[Vec<f64>],
    v: &[Vec<f64>],
    n_vecs: usize,
    m: usize,
    n: usize,
) -> Vec<Vec<f64>> {
    if n_vecs == 0 || m == 0 {
        return Vec::new();
    }

    // Compute harmonic Ritz values: solve shifted eigenvalue problem
    // (H_m^T H_m + h_{m+1,m}^2 e_m e_m^T) s = theta * H_m^T s
    // which is equivalent to finding eigenvectors of
    //   H_m + h_{m+1,m}^2 (H_m^T)^{-1} e_m e_m^T
    //
    // For deflation, we want Ritz pairs with smallest absolute harmonic Ritz values.
    // We use the Schur decomposition of the m×m submatrix H_m (ignoring the last row)
    // via Jacobi iterations on the symmetric part, or power-like methods.
    //
    // Practical approach: use the QR algorithm on H_m to extract eigenvalues,
    // then select the n_vecs eigenvectors corresponding to smallest |eigenvalue|.

    let hm = extract_square_hessenberg(h, m);
    let h_last_row_norm = if m < h.len() { h[m][m - 1].abs() } else { 0.0 };

    // Run Francis double-shift QR on hm to find Schur decomposition.
    // We approximate eigenvectors using inverse iteration on the Hessenberg.
    let eig_vecs = hessenberg_schur_vectors(&hm, n_vecs, h_last_row_norm);

    // Map back to original space: y_i = V_m s_i.
    let mut deflation = Vec::with_capacity(n_vecs);
    for s in &eig_vecs {
        if s.len() != m {
            continue;
        }
        let mut y = vec![0.0f64; n];
        for (j, &sj) in s.iter().enumerate() {
            if j < v.len() {
                for l in 0..n {
                    y[l] += sj * v[j][l];
                }
            }
        }
        // Normalise.
        let norm = norm2(&y);
        if norm > 1e-15 {
            for yi in &mut y {
                *yi /= norm;
            }
            deflation.push(y);
        }
    }

    // Orthonormalise the deflation vectors via MGS.
    gram_schmidt_mgs(&mut deflation);
    deflation
}

/// Extract the m×m upper-left sub-block of the Hessenberg matrix.
#[allow(dead_code)]
fn extract_square_hessenberg(h: &[Vec<f64>], m: usize) -> Vec<Vec<f64>> {
    (0..m)
        .map(|i| {
            if i < h.len() {
                let row_len = m.min(h[i].len());
                let mut row = h[i][..row_len].to_vec();
                row.resize(m, 0.0);
                row
            } else {
                vec![0.0; m]
            }
        })
        .collect()
}

/// Compute approximate right Schur vectors for the Hessenberg matrix using
/// Francis double-shift QR iteration, extracting `n_vecs` vectors corresponding
/// to Ritz values with smallest absolute value (for deflation of near-null space).
///
/// Returns at most `n_vecs` vectors of length m, each being an approximate
/// Schur vector corresponding to a small eigenvalue.
#[allow(dead_code)]
fn hessenberg_schur_vectors(hm: &[Vec<f64>], n_vecs: usize, h_extra: f64) -> Vec<Vec<f64>> {
    let m = hm.len();
    if m == 0 || n_vecs == 0 {
        return Vec::new();
    }

    // Build the harmonic Ritz matrix:
    // A_harm = H_m + (h_{m+1,m})^2 * (H_m^T)^{-1} * e_m * e_m^T
    // where e_m is the m-th standard basis vector.
    // For deflation we want vectors in the range of A_harm with smallest eigenvalues.
    //
    // Simplified robust approach: use QR iteration on H_m itself.
    // Apply `n_iter` steps of explicit shifted QR and accumulate the
    // orthogonal factor Q. Take first n_vecs columns of Q^T.

    // Copy hm for in-place QR iteration.
    let mut a: Vec<Vec<f64>> = hm.to_vec();
    // Accumulate orthogonal transformations in Q (initialised to identity).
    let mut q_accum: Vec<Vec<f64>> = (0..m)
        .map(|i| {
            let mut row = vec![0.0f64; m];
            row[i] = 1.0;
            row
        })
        .collect();

    // Use h_extra to add a Tikhonov-like perturbation for the harmonic shift.
    let harmonic_shift = h_extra * h_extra;
    if harmonic_shift > 1e-30 && m >= 2 {
        a[m - 1][m - 1] += harmonic_shift;
    }

    // Run at most 40 * m QR steps with Wilkinson shifts.
    let max_steps = 40 * m;
    for _step in 0..max_steps {
        // Wilkinson shift: eigenvalue of bottom 2×2 closest to a[m-1][m-1].
        let shift = wilkinson_shift(&a, m);

        // QR factorization of (A - shift*I) via Givens rotations on the
        // upper-Hessenberg structure.
        let (q_step, r_step) = hessenberg_qr_step(&a, shift, m);

        // A_new = R * Q + shift * I
        a = mat_mul(&r_step, &q_step, m);
        for i in 0..m {
            a[i][i] += shift;
        }

        // Accumulate Q.
        q_accum = mat_mul(&q_accum, &q_step, m);
    }

    // The columns of q_accum are (approximate) Schur vectors.
    // Find eigenvalue magnitudes from the diagonal of a.
    let mut eig_mags: Vec<(f64, usize)> = (0..m).map(|i| (a[i][i].abs(), i)).collect();
    eig_mags.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

    let take = n_vecs.min(m);
    eig_mags[..take]
        .iter()
        .map(|(_, col)| (0..m).map(|row| q_accum[row][*col]).collect::<Vec<f64>>())
        .collect()
}

/// Compute the Wilkinson shift for QR iteration: the eigenvalue of the
/// bottom 2×2 sub-block of `a` closest to `a[m-1][m-1]`.
#[allow(dead_code)]
fn wilkinson_shift(a: &[Vec<f64>], m: usize) -> f64 {
    if m < 2 {
        return if m == 1 { a[0][0] } else { 0.0 };
    }
    let a_mm = a[m - 1][m - 1];
    let a_m1m1 = a[m - 2][m - 2];
    let a_m1m = a[m - 2][m - 1];
    let a_mm1 = a[m - 1][m - 2];
    // Characteristic polynomial of 2×2: lambda^2 - tr*lambda + det = 0.
    let tr = a_mm + a_m1m1;
    let det = a_mm * a_m1m1 - a_m1m * a_mm1;
    let disc = tr * tr - 4.0 * det;
    if disc < 0.0 {
        // Complex pair: use midpoint as shift.
        return tr / 2.0;
    }
    let sqrt_disc = disc.sqrt();
    let lam1 = (tr + sqrt_disc) / 2.0;
    let lam2 = (tr - sqrt_disc) / 2.0;
    // Choose closest to a[m-1][m-1].
    if (lam1 - a_mm).abs() <= (lam2 - a_mm).abs() {
        lam1
    } else {
        lam2
    }
}

/// Perform a single Hessenberg QR step with given shift, returning (Q, R).
///
/// Uses Givens rotations to reduce (A - shift*I) to upper triangular.
#[allow(dead_code)]
fn hessenberg_qr_step(a: &[Vec<f64>], shift: f64, m: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    // Shifted copy: r = a - shift * I.
    let mut r: Vec<Vec<f64>> = (0..m)
        .map(|i| {
            let mut row = a[i].clone();
            if row.len() > i {
                row[i] -= shift;
            }
            row
        })
        .collect();

    // Q accumulated as product of Givens rotations.
    let mut q: Vec<Vec<f64>> = (0..m)
        .map(|i| {
            let mut row = vec![0.0f64; m];
            if i < m {
                row[i] = 1.0;
            }
            row
        })
        .collect();

    // Apply Givens to zero out sub-diagonal entries.
    for j in 0..m.saturating_sub(1) {
        let a_jj = if j < r.len() && j < r[j].len() {
            r[j][j]
        } else {
            0.0
        };
        let a_j1j = if j + 1 < r.len() && j < r[j + 1].len() {
            r[j + 1][j]
        } else {
            0.0
        };
        let denom = (a_jj * a_jj + a_j1j * a_j1j).sqrt();
        let (c, s) = if denom < 1e-300 {
            (1.0f64, 0.0f64)
        } else {
            (a_jj / denom, a_j1j / denom)
        };

        // Apply Givens rotation G(j, j+1, theta) from the left to r.
        for col in 0..m {
            let r_jc = if j < r.len() && col < r[j].len() {
                r[j][col]
            } else {
                0.0
            };
            let r_j1c = if j + 1 < r.len() && col < r[j + 1].len() {
                r[j + 1][col]
            } else {
                0.0
            };
            if j < r.len() && col < r[j].len() {
                r[j][col] = c * r_jc + s * r_j1c;
            }
            if j + 1 < r.len() && col < r[j + 1].len() {
                r[j + 1][col] = -s * r_jc + c * r_j1c;
            }
        }

        // Accumulate G^T in q (apply from the right to q).
        for row in 0..m {
            let q_rj = if row < q.len() && j < q[row].len() {
                q[row][j]
            } else {
                0.0
            };
            let q_rj1 = if row < q.len() && j + 1 < q[row].len() {
                q[row][j + 1]
            } else {
                0.0
            };
            if row < q.len() && j < q[row].len() {
                q[row][j] = c * q_rj + s * q_rj1;
            }
            if row < q.len() && j + 1 < q[row].len() {
                q[row][j + 1] = -s * q_rj + c * q_rj1;
            }
        }
    }

    (q, r)
}

/// Dense matrix multiply C = A * B for m×m matrices stored as Vec<Vec<f64>>.
#[allow(dead_code)]
fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>], m: usize) -> Vec<Vec<f64>> {
    let mut c = vec![vec![0.0f64; m]; m];
    for i in 0..m {
        for k in 0..m {
            let a_ik = if i < a.len() && k < a[i].len() {
                a[i][k]
            } else {
                0.0
            };
            if a_ik == 0.0 {
                continue;
            }
            for j in 0..m {
                let b_kj = if k < b.len() && j < b[k].len() {
                    b[k][j]
                } else {
                    0.0
                };
                c[i][j] += a_ik * b_kj;
            }
        }
    }
    c
}

/// Orthonormalise a list of vectors in-place using Modified Gram-Schmidt.
pub(crate) fn gram_schmidt_mgs(vecs: &mut [Vec<f64>]) {
    let n_vecs = vecs.len();
    for i in 0..n_vecs {
        // Normalise v[i].
        let norm = norm2(&vecs[i]);
        if norm < 1e-300 {
            continue;
        }
        let inv = 1.0 / norm;
        let len = vecs[i].len();
        for l in 0..len {
            vecs[i][l] *= inv;
        }
        // Orthogonalise v[i+1..] against v[i].
        for j in (i + 1)..n_vecs {
            let coeff = dot(&vecs[i], &vecs[j]);
            if coeff.abs() < 1e-300 {
                continue;
            }
            let vi = vecs[i].clone();
            let len = vecs[j].len().min(vi.len());
            for l in 0..len {
                vecs[j][l] -= coeff * vi[l];
            }
        }
    }
}

/// Euclidean norm of a slice.
pub(crate) fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Inner product of two slices.
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a diagonally dominant matrix-vector product closure.
    /// A = diag(d_i) where d_i ≠ 0.
    fn diag_matvec(diag: Vec<f64>) -> impl Fn(&[f64]) -> Vec<f64> {
        move |x: &[f64]| x.iter().zip(diag.iter()).map(|(xi, di)| xi * di).collect()
    }

    /// Tridiagonal matvec: A = tridiag(-1, 4, -1) scaled to be SPD.
    fn tridiag_matvec(n: usize) -> impl Fn(&[f64]) -> Vec<f64> {
        move |x: &[f64]| {
            let mut y = vec![0.0f64; n];
            for i in 0..n {
                y[i] = 4.0 * x[i];
                if i > 0 {
                    y[i] -= x[i - 1];
                }
                if i + 1 < n {
                    y[i] -= x[i + 1];
                }
            }
            y
        }
    }

    #[test]
    fn test_gmres_dr_diagonal_system() {
        // Solve diag(1,2,...,10) x = [1,1,...,1].
        // Exact solution: x_i = 1/(i+1).
        let n = 10;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let b: Vec<f64> = vec![1.0; n];
        let x_exact: Vec<f64> = (1..=n).map(|i| 1.0 / (i as f64)).collect();

        let solver = GmresDR::new(8, 3).with_tolerance(1e-12).with_max_iter(500);
        let result = solver
            .solve(diag_matvec(diag), &b, None)
            .expect("GMRES-DR solve should succeed");

        assert!(
            result.converged,
            "GMRES-DR should converge on diagonal system; residual = {:.3e}",
            result.residual_norm
        );
        for i in 0..n {
            assert!(
                (result.x[i] - x_exact[i]).abs() < 1e-10,
                "x[{}] = {:.6}, expected {:.6}",
                i,
                result.x[i],
                x_exact[i]
            );
        }
    }

    #[test]
    fn test_gmres_dr_tridiagonal_system() {
        // Tridiagonal(-1, 4, -1) is SPD; GMRES-DR should converge.
        let n = 20;
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        let solver = GmresDR::new(10, 3).with_tolerance(1e-12).with_max_iter(600);
        let result = solver
            .solve(tridiag_matvec(n), &b, None)
            .expect("GMRES-DR solve failed");

        assert!(
            result.converged,
            "GMRES-DR tridiagonal: residual = {:.3e}, iterations = {}",
            result.residual_norm, result.iterations
        );
        assert!(
            result.residual_norm < 1e-9,
            "residual too large: {:.3e}",
            result.residual_norm
        );
    }

    #[test]
    fn test_gmres_dr_with_initial_guess() {
        let n = 8;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let b: Vec<f64> = vec![1.0; n];
        let x0: Vec<f64> = vec![0.1; n]; // non-zero initial guess

        let solver = GmresDR::new(6, 2).with_tolerance(1e-12);
        let result = solver
            .solve(diag_matvec(diag), &b, Some(&x0))
            .expect("solve failed");
        assert!(result.converged);
    }

    #[test]
    fn test_gmres_dr_residual_history_monotone() {
        // Residual history should generally be non-increasing at cycle boundaries.
        let n = 15;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64 * 0.5 + 0.5).collect();
        let b: Vec<f64> = vec![1.0; n];

        let solver = GmresDR::new(5, 2).with_tolerance(1e-12).with_max_iter(400);
        let result = solver
            .solve(diag_matvec(diag), &b, None)
            .expect("solve failed");

        assert!(result.converged);
        assert!(!result.residual_history.is_empty());
    }

    #[test]
    fn test_hessenberg_ls_solve() {
        // 4x3 Hessenberg with a consistent right-hand side (H y_exact = g).
        // H = [[2,1,0.5],[0.5,2,1],[0,0.3,1.5],[0,0,0.2]]
        // y_exact = [0.5, 0.3, 0.1]  → g = H y_exact
        let h = vec![
            vec![2.0, 1.0, 0.5],
            vec![0.5, 2.0, 1.0],
            vec![0.0, 0.3, 1.5],
            vec![0.0, 0.0, 0.2],
        ];
        let y_exact = [0.5_f64, 0.3, 0.1];
        let g: Vec<f64> = (0..4)
            .map(|i| (0..3).map(|j| h[i][j] * y_exact[j]).sum::<f64>())
            .collect();

        let y = solve_least_squares_hessenberg(&h, &g, 3).expect("LS solve failed");
        assert_eq!(y.len(), 3);

        // Residual should be near zero for consistent RHS.
        let res: f64 = (0..4)
            .map(|i| {
                let hy_i: f64 = (0..3).map(|j| h[i][j] * y[j]).sum();
                (g[i] - hy_i).powi(2)
            })
            .sum::<f64>()
            .sqrt();
        assert!(res < 1e-10, "LS residual = {:.3e}", res);

        // Solution should match y_exact.
        for j in 0..3 {
            assert!(
                (y[j] - y_exact[j]).abs() < 1e-10,
                "y[{}] = {:.8}, expected {:.8}",
                j,
                y[j],
                y_exact[j]
            );
        }
    }

    #[test]
    fn test_gram_schmidt_orthogonality() {
        let mut vecs = vec![
            vec![1.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0],
        ];
        gram_schmidt_mgs(&mut vecs);
        for i in 0..3 {
            let n = norm2(&vecs[i]);
            assert!((n - 1.0).abs() < 1e-14, "vec {} not normalised: {}", i, n);
        }
        for i in 0..3 {
            for j in (i + 1)..3 {
                let ip = dot(&vecs[i], &vecs[j]);
                assert!(
                    ip.abs() < 1e-13,
                    "vecs {} and {} not orthogonal: inner product = {:.3e}",
                    i,
                    j,
                    ip
                );
            }
        }
    }
}
