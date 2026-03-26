//! FEAST eigensolver — finds eigenvalues/eigenvectors in a spectral interval [a, b].
//!
//! FEAST uses contour integration around the target interval to project a random
//! subspace onto the eigenspace of interest, then applies Rayleigh-Ritz to extract
//! eigenvalues. Unlike traditional iterative methods, FEAST is a filter-based approach
//! that is highly parallelizable.
//!
//! # Algorithm (Polizzi 2009)
//!
//! 1. Choose m₀ random starting vectors Y (n × m₀).
//! 2. For each FEAST iteration:
//!    a. Contour integration on an ellipse surrounding [a, b]:
//!       - Gauss-Legendre nodes θₖ ∈ [0, π] → complex shift zₖ
//!       - Solve (A - zₖ I) Xₖ = Y for each quadrature point
//!         b. Subspace update: Q = (1/2πi) Σₖ wₖ Xₖ
//!         c. QR decomposition of Q → orthonormal basis V
//!         d. Rayleigh-Ritz projection: A_small = Vᵀ A V
//!         e. Solve small eigenproblem A_small C = C Λ
//!         f. Ritz vectors: Y = V C
//!         g. Check residuals ‖A yᵢ - λᵢ yᵢ‖ < tol
//!
//! # References
//!
//! - Polizzi, E. (2009). Density-matrix-based algorithm for solving eigenvalue problems.
//!   *Physical Review B*, 79(11), 115112.
//! - Kestyn, J., Polizzi, E. & Tang, P.T.P. (2016). FEAST eigensolver for non-Hermitian
//!   problems. *SIAM J. Sci. Comput.*, 38(5), S772–S799.

use crate::error::{LinalgError, LinalgResult};

/// Configuration for the FEAST eigensolver.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct FeastConfig {
    /// Number of Gauss-Legendre quadrature points on the contour. Default: 8.
    pub n_contour_pts: usize,
    /// Subspace size m₀ (should be ≥ expected number of eigenvalues). Default: 10.
    pub subspace_size: usize,
    /// Maximum FEAST iterations. Default: 20.
    pub max_iter: usize,
    /// Convergence tolerance for residuals. Default: 1e-10.
    pub tol: f64,
    /// Target spectral interval [a, b]. Default: (-1.0, 1.0).
    pub interval: (f64, f64),
}

impl Default for FeastConfig {
    fn default() -> Self {
        Self {
            n_contour_pts: 8,
            subspace_size: 10,
            max_iter: 20,
            tol: 1e-10,
            interval: (-1.0, 1.0),
        }
    }
}

/// Result of the FEAST eigensolver.
#[derive(Debug, Clone)]
pub struct FeastResult {
    /// Eigenvalues found within the interval [a, b], sorted ascending.
    pub eigenvalues: Vec<f64>,
    /// Corresponding eigenvectors as columns: `eigenvectors[j][i]` is the i-th component of the j-th eigenvector.
    pub eigenvectors: Vec<Vec<f64>>,
    /// Number of FEAST iterations performed.
    pub n_iter: usize,
    /// Residual norms ‖A vⱼ - λⱼ vⱼ‖ for each eigenpair.
    pub residuals: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Find eigenvalues and eigenvectors of a real symmetric matrix A within the
/// interval [config.interval.0, config.interval.1].
///
/// # Arguments
///
/// * `a` — n×n real symmetric matrix (row-major).
/// * `config` — FEAST configuration.
///
/// # Returns
///
/// [`FeastResult`] containing eigenvalues, eigenvectors, iteration count, and residuals.
pub fn feast_eig(a: &[Vec<f64>], config: &FeastConfig) -> LinalgResult<FeastResult> {
    let n = a.len();
    if n == 0 {
        return Ok(FeastResult {
            eigenvalues: vec![],
            eigenvectors: vec![],
            n_iter: 0,
            residuals: vec![],
        });
    }
    for row in a {
        if row.len() != n {
            return Err(LinalgError::DimensionError(format!(
                "Matrix row has {} elements, expected {n}",
                row.len()
            )));
        }
    }
    let (a_lo, a_hi) = config.interval;
    if a_lo >= a_hi {
        return Err(LinalgError::InvalidInputError(
            "Interval [a, b] must have a < b".to_string(),
        ));
    }
    let m0 = config.subspace_size.min(n);

    // Initialize random subspace Y (n × m0) using a simple deterministic LCG
    let mut lcg = SimpleLcg::new(42);
    let mut y: Vec<Vec<f64>> = (0..m0)
        .map(|_| (0..n).map(|_| lcg.next_normal()).collect())
        .collect();

    // Gauss-Legendre nodes and weights on [-1, 1]
    let (gl_nodes, gl_weights) = gl_nodes_weights(config.n_contour_pts);

    let center = (a_hi + a_lo) / 2.0;
    let radius = (a_hi - a_lo) / 2.0;

    let mut n_iter = 0usize;

    for iter in 0..config.max_iter {
        n_iter = iter + 1;

        // Subspace filter: Q = (1/2πi) ∮ (A - z I)⁻¹ Y dz
        // Discretized on semi-circle contour z(θ) = center + radius * exp(i θ) for θ ∈ [0, π]
        // using Gauss-Legendre on [0, π].
        let mut q_accum: Vec<Vec<f64>> = vec![vec![0.0f64; n]; m0];

        for (k, (&node, &weight)) in gl_nodes.iter().zip(gl_weights.iter()).enumerate() {
            // Map node from [-1,1] to [0, π]
            let theta = std::f64::consts::PI * (node + 1.0) / 2.0;
            let half_pi = std::f64::consts::PI / 2.0;

            // Contour point: z = center + radius * exp(i*theta)
            let z_re = center + radius * theta.cos();
            let z_im = radius * theta.sin();

            // Weight for integration: (π/2) * weight * z'(θ) / (2πi)
            // z'(θ) = radius * i * exp(i*θ) = radius * i * (cos θ + i sin θ)
            //       = radius * (-sin θ + i cos θ)
            // dz = radius * i * exp(i θ) dθ
            // The integration formula is:
            //   Q += (1/2πi) Σₖ wₖ (A - zₖ I)⁻¹ Y * z'_k * (π/2)
            // where the (π/2) comes from the change of variables to [-1,1].
            // z'(θ) = radius * i * exp(i θ):
            //   Real part of z'(θ) = -radius * sin θ
            //   Imag part of z'(θ) =  radius * cos θ
            // So (1/2πi) * z'(θ) * dθ:
            //   = (1/2πi) * radius * (i cos θ - sin θ) * dθ
            //   = radius / (2π) * (cos θ + i (-sin θ) * (1/i))  ... let's compute directly:
            // (1/2πi) * (radius * i * exp(i θ)):
            //   = (radius / 2π) * exp(i θ)
            //   = (radius / 2π) * (cos θ + i sin θ)
            // Real contribution: (radius / 2π) * cos θ
            // Imag contribution: (radius / 2π) * sin θ

            let weight_factor = weight * half_pi; // Jacobian for [-1,1] → [0,π]
            let contour_re = radius * theta.cos() / (2.0 * std::f64::consts::PI);
            let contour_im = radius * theta.sin() / (2.0 * std::f64::consts::PI);
            let w_re = weight_factor * contour_re;
            let w_im = weight_factor * contour_im;

            // For each column of Y, solve (A - z I) x = y (complex linear system)
            // A is real symmetric, z is complex → use augmented 2×2 real block system
            for col in 0..m0 {
                let rhs = &y[col];
                let (x_re, x_im) = solve_complex_shifted(a, z_re, z_im, rhs)?;
                // Accumulate: q_accum[col] += w_re * x_re - w_im * x_im
                // (the imaginary part w_re * x_im + w_im * x_re is discarded for real A)
                // For symmetric real A and conjugate-symmetric contour, Q is real.
                // We take 2 * real part (the contour integral wraps the upper half).
                for i in 0..n {
                    q_accum[col][i] += 2.0 * (w_re * x_re[i] - w_im * x_im[i]);
                }
            }
            let _ = k; // suppress unused warning
        }

        // QR decomposition of Q_accum to get orthonormal basis V
        let (v_cols, _r) = qr_decompose(q_accum);

        // Rayleigh-Ritz: compute A_small = V^T A V  (m0 × m0)
        let a_small = rayleigh_ritz_project(a, &v_cols, m0, n);

        // Solve small eigenproblem A_small C = C Λ via Jacobi or direct
        let (small_evals, small_evecs) = symmetric_eigen_small(&a_small, m0)?;

        // Ritz vectors: y_new[j] = V * c_j
        let mut y_new: Vec<Vec<f64>> = vec![vec![0.0f64; n]; m0];
        for j in 0..m0 {
            for i in 0..n {
                for l in 0..m0 {
                    y_new[j][i] += v_cols[l][i] * small_evecs[j][l];
                }
            }
        }

        // Check convergence: residual ‖A y_j - λ_j y_j‖
        let mut converged = true;
        let mut max_res = 0.0f64;
        for j in 0..m0 {
            let lam = small_evals[j];
            // Only check eigenvectors whose eigenvalue falls in [a_lo, a_hi]
            if lam < a_lo || lam > a_hi {
                continue;
            }
            let ay = matvec(a, &y_new[j], n);
            let res: f64 = (0..n)
                .map(|i| {
                    let r = ay[i] - lam * y_new[j][i];
                    r * r
                })
                .sum::<f64>()
                .sqrt();
            max_res = max_res.max(res);
            if res > config.tol {
                converged = false;
            }
        }

        y = y_new;

        if converged && max_res < config.tol {
            break;
        }
    }

    // Extract eigenvalues/vectors in [a_lo, a_hi]
    // Redo Rayleigh-Ritz with final y (re-orthogonalize first)
    let (v_final, _) = qr_decompose(y);
    let a_small_final = rayleigh_ritz_project(a, &v_final, m0, n);
    let (small_evals_final, small_evecs_final) = symmetric_eigen_small(&a_small_final, m0)?;

    let mut result_pairs: Vec<(f64, Vec<f64>)> = Vec::new();
    for j in 0..m0 {
        let lam = small_evals_final[j];
        if lam >= a_lo && lam <= a_hi {
            let mut ritz_vec = vec![0.0f64; n];
            for i in 0..n {
                for l in 0..m0 {
                    ritz_vec[i] += v_final[l][i] * small_evecs_final[j][l];
                }
            }
            result_pairs.push((lam, ritz_vec));
        }
    }

    // Sort by eigenvalue
    result_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let eigenvalues: Vec<f64> = result_pairs.iter().map(|(e, _)| *e).collect();
    let eigenvectors: Vec<Vec<f64>> = result_pairs.iter().map(|(_, v)| v.clone()).collect();

    // Compute final residuals
    let residuals: Vec<f64> = eigenvalues
        .iter()
        .zip(eigenvectors.iter())
        .map(|(&lam, v)| {
            let ay = matvec(a, v, n);
            (0..n)
                .map(|i| {
                    let r = ay[i] - lam * v[i];
                    r * r
                })
                .sum::<f64>()
                .sqrt()
        })
        .collect();

    Ok(FeastResult {
        eigenvalues,
        eigenvectors,
        n_iter,
        residuals,
    })
}

// ---------------------------------------------------------------------------
// Helper: matrix-vector product
// ---------------------------------------------------------------------------

fn matvec(a: &[Vec<f64>], v: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0.0f64; n];
    for i in 0..n {
        for j in 0..n {
            result[i] += a[i][j] * v[j];
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Rayleigh-Ritz projection
// ---------------------------------------------------------------------------

fn rayleigh_ritz_project(
    a: &[Vec<f64>],
    v_cols: &[Vec<f64>],
    m0: usize,
    n: usize,
) -> Vec<Vec<f64>> {
    // Compute AV first: AV[col][row]
    let mut av: Vec<Vec<f64>> = vec![vec![0.0f64; n]; m0];
    for col in 0..m0 {
        av[col] = matvec(a, &v_cols[col], n);
    }
    // A_small[i][j] = V[:,i]^T * (AV[:,j]) = sum_k V[i][k] * AV[j][k]
    let mut a_small = vec![vec![0.0f64; m0]; m0];
    for i in 0..m0 {
        for j in 0..m0 {
            let dot: f64 = v_cols[i].iter().zip(av[j].iter()).map(|(a, b)| a * b).sum();
            a_small[i][j] = dot;
        }
    }
    a_small
}

// ---------------------------------------------------------------------------
// QR decomposition via modified Gram-Schmidt
// ---------------------------------------------------------------------------

/// Thin QR decomposition. Returns (Q_cols, R) where `Q_cols[j]` is the j-th column of Q.
pub fn qr_decompose(a: Vec<Vec<f64>>) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let m = a.len(); // number of columns
    if m == 0 {
        return (vec![], vec![]);
    }
    let n = a[0].len(); // number of rows

    let mut q = a;
    let mut r = vec![vec![0.0f64; m]; m];

    for j in 0..m {
        // Normalize column j
        let norm: f64 = q[j].iter().map(|x| x * x).sum::<f64>().sqrt();
        r[j][j] = norm;
        if norm < 1e-14 {
            // Degenerate column: replace with a random orthogonal vector
            let mut e = vec![0.0f64; n];
            if j < n {
                e[j] = 1.0;
            }
            q[j] = e;
        } else {
            for xi in &mut q[j] {
                *xi /= norm;
            }
        }
        // Orthogonalize subsequent columns against q[j]
        let qj_clone = q[j].clone();
        for k in j + 1..m {
            let dot: f64 = qj_clone.iter().zip(q[k].iter()).map(|(a, b)| a * b).sum();
            r[j][k] = dot;
            for (qki, qji) in q[k].iter_mut().zip(qj_clone.iter()) {
                *qki -= dot * qji;
            }
        }
    }

    // Re-orthogonalize for numerical stability (one pass of MGS)
    for j in 0..m {
        for k in 0..j {
            let qk_clone = q[k].clone();
            let dot: f64 = qk_clone.iter().zip(q[j].iter()).map(|(a, b)| a * b).sum();
            for (qji, qki) in q[j].iter_mut().zip(qk_clone.iter()) {
                *qji -= dot * qki;
            }
        }
        let norm: f64 = q[j].iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-14 {
            for xi in &mut q[j] {
                *xi /= norm;
            }
        }
    }

    (q, r)
}

// ---------------------------------------------------------------------------
// LU decomposition with partial pivoting
// ---------------------------------------------------------------------------

/// LU decomposition with partial pivoting. Returns (combined LU matrix, permutation).
pub fn lu_decompose(mut a: Vec<Vec<f64>>) -> LinalgResult<(Vec<Vec<f64>>, Vec<usize>)> {
    let n = a.len();
    let mut perm: Vec<usize> = (0..n).collect();

    for k in 0..n {
        // Find pivot
        let mut max_val = a[k][k].abs();
        let mut max_row = k;
        for (i, ai) in a.iter().enumerate().take(n).skip(k + 1) {
            if ai[k].abs() > max_val {
                max_val = ai[k].abs();
                max_row = i;
            }
        }
        if max_val < 1e-300 {
            return Err(LinalgError::SingularMatrixError(format!(
                "Singular matrix at pivot position {k}"
            )));
        }
        // Swap rows
        if max_row != k {
            a.swap(k, max_row);
            perm.swap(k, max_row);
        }
        // Eliminate
        let akk = a[k][k];
        let ak_row: Vec<f64> = a[k].clone();
        for ai in a.iter_mut().take(n).skip(k + 1) {
            let factor = ai[k] / akk;
            ai[k] = factor;
            for j in k + 1..n {
                ai[j] -= factor * ak_row[j];
            }
        }
    }

    Ok((a, perm))
}

/// Solve L U x = b given LU factorization.
pub fn lu_solve(lu: &[Vec<f64>], perm: &[usize], b: &[f64]) -> Vec<f64> {
    let n = lu.len();
    // Apply permutation
    let mut x: Vec<f64> = perm.iter().map(|&i| b[i]).collect();
    // Forward substitution (L is unit lower triangular)
    for i in 0..n {
        for j in 0..i {
            x[i] -= lu[i][j] * x[j];
        }
    }
    // Back substitution (U is upper triangular)
    for i in (0..n).rev() {
        for j in i + 1..n {
            x[i] -= lu[i][j] * x[j];
        }
        x[i] /= lu[i][i];
    }
    x
}

// ---------------------------------------------------------------------------
// Complex shifted system solver
// ---------------------------------------------------------------------------

/// Solve (A - (z_re + i*z_im) I) x = b for real symmetric A.
/// Returns (x_re, x_im) where x = x_re + i*x_im.
///
/// For real symmetric A and complex shift z = z_re + i*z_im:
/// (A - z I) x = b  →  Let x = u + iv:
///   (A - z_re I) u + z_im v = b   (real)
///   (A - z_re I) v - z_im u = 0   (imag)
///
/// This gives the 2n×2n real block system:
///   [[A - z_re I,  z_im I], [-z_im I,  A - z_re I]] [u; v] = [b; 0]
///
/// For efficiency, we solve via the formula:
///   ((A - z_re I)² + z_im² I) u = (A - z_re I) b
///   v = (1/z_im) ((A - z_re I) u - b)  (if z_im ≠ 0)
fn solve_complex_shifted(
    a: &[Vec<f64>],
    z_re: f64,
    z_im: f64,
    b: &[f64],
) -> LinalgResult<(Vec<f64>, Vec<f64>)> {
    let n = a.len();

    if z_im.abs() < 1e-300 {
        // Real shift: just solve (A - z_re I) x = b
        let mut a_shifted: Vec<Vec<f64>> = a.to_vec();
        for (i, row) in a_shifted.iter_mut().enumerate().take(n) {
            row[i] -= z_re;
        }
        let (lu, perm) = lu_decompose(a_shifted)?;
        let x_re = lu_solve(&lu, &perm, b);
        return Ok((x_re, vec![0.0f64; n]));
    }

    // Build M = (A - z_re I)² + z_im² I
    // M[i][j] = Σ_k (A - z_re I)[i][k] * (A - z_re I)[k][j]  +  z_im² * δ_{ij}
    // Compute B = A - z_re I
    let b_mat: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if i == j { a[i][j] - z_re } else { a[i][j] })
                .collect()
        })
        .collect();

    // M = B*B + z_im^2 * I
    let mut m_mat: Vec<Vec<f64>> = vec![vec![0.0f64; n]; n];
    for (i, m_row) in m_mat.iter_mut().enumerate().take(n) {
        for j in 0..n {
            let mut sum = 0.0f64;
            for (k, bk) in b_mat.iter().enumerate().take(n) {
                sum += b_mat[i][k] * bk[j];
            }
            m_row[j] = sum;
        }
        m_row[i] += z_im * z_im;
    }

    // rhs for u: (A - z_re I) b = B * b
    let rhs_u: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|k| b_mat[i][k] * b[k]).sum())
        .collect();

    let (lu, perm) = lu_decompose(m_mat)?;
    let x_re = lu_solve(&lu, &perm, &rhs_u);

    // v = (1/z_im) * (B * u - b)
    let bu: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|k| b_mat[i][k] * x_re[k]).sum::<f64>())
        .collect();
    let x_im: Vec<f64> = (0..n).map(|i| (bu[i] - b[i]) / z_im).collect();

    Ok((x_re, x_im))
}

// ---------------------------------------------------------------------------
// Small symmetric eigensolver (Jacobi iteration)
// ---------------------------------------------------------------------------

/// Compute all eigenvalues and eigenvectors of a small real symmetric matrix
/// using Jacobi iteration.
fn symmetric_eigen_small(a: &[Vec<f64>], n: usize) -> LinalgResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let max_sweeps = 100;
    let tol = 1e-13;

    let mut mat: Vec<Vec<f64>> = a.to_vec();
    // Eigenvectors stored as columns: evecs[j] = j-th eigenvector
    let mut evecs: Vec<Vec<f64>> = (0..n)
        .map(|j| {
            let mut col = vec![0.0f64; n];
            col[j] = 1.0;
            col
        })
        .collect();

    for _sweep in 0..max_sweeps {
        // Find max off-diagonal element
        let mut max_off = 0.0f64;
        let mut p = 0usize;
        let mut q = 1usize;
        for (i, mat_i) in mat.iter().enumerate().take(n) {
            for (j, &mat_ij) in mat_i.iter().enumerate().take(n).skip(i + 1) {
                let abs_ij = mat_ij.abs();
                if abs_ij > max_off {
                    max_off = abs_ij;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < tol {
            break;
        }

        // Compute Jacobi rotation angle
        let theta = (mat[q][q] - mat[p][p]) / (2.0 * mat[p][q]);
        let sign_theta = if theta >= 0.0 { 1.0_f64 } else { -1.0_f64 };
        let t = sign_theta / (theta.abs() + (1.0 + theta * theta).sqrt());
        let cos_phi = 1.0 / (1.0 + t * t).sqrt();
        let sin_phi = t * cos_phi;

        // Update matrix: G^T A G
        // pp, pq, qq elements
        let app = mat[p][p];
        let aqq = mat[q][q];
        let apq = mat[p][q];

        mat[p][p] = app - t * apq;
        mat[q][q] = aqq + t * apq;
        mat[p][q] = 0.0;
        mat[q][p] = 0.0;

        for (r, mat_r) in mat.iter_mut().enumerate().take(n) {
            if r != p && r != q {
                let arp = mat_r[p];
                let arq = mat_r[q];
                mat_r[p] = cos_phi * arp - sin_phi * arq;
                mat_r[q] = sin_phi * arp + cos_phi * arq;
            }
        }
        // Symmetrize
        {
            let col_p: Vec<f64> = (0..n).map(|r| mat[r][p]).collect();
            let col_q: Vec<f64> = (0..n).map(|r| mat[r][q]).collect();
            for r in 0..n {
                if r != p && r != q {
                    mat[p][r] = col_p[r];
                    mat[q][r] = col_q[r];
                }
            }
        }

        // Update eigenvectors
        {
            let (left, right) = evecs.split_at_mut(q);
            let ep = &mut left[p];
            let eq = &mut right[0];
            for (vp, vq) in ep.iter_mut().zip(eq.iter_mut()) {
                let old_p = *vp;
                let old_q = *vq;
                *vp = cos_phi * old_p - sin_phi * old_q;
                *vq = sin_phi * old_p + cos_phi * old_q;
            }
        }
    }

    // Extract eigenvalues from diagonal
    let mut pairs: Vec<(f64, Vec<f64>)> = (0..n).map(|i| (mat[i][i], evecs[i].clone())).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let evals: Vec<f64> = pairs.iter().map(|(e, _)| *e).collect();
    let evecs_sorted: Vec<Vec<f64>> = pairs.into_iter().map(|(_, v)| v).collect();

    Ok((evals, evecs_sorted))
}

// ---------------------------------------------------------------------------
// Gauss-Legendre nodes and weights
// ---------------------------------------------------------------------------

/// Compute Gauss-Legendre quadrature nodes and weights on [-1, 1] using
/// the Golub-Welsch algorithm (companion matrix of Legendre polynomials).
pub fn gl_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (vec![], vec![]);
    }
    if n == 1 {
        return (vec![0.0], vec![2.0]);
    }

    // Build symmetric tridiagonal Jacobi matrix for Legendre polynomials
    // Diagonal is zero; off-diagonal β_k = k / sqrt(4k²-1) for k=1,...,n-1
    let mut diag = vec![0.0f64; n];
    let mut off: Vec<f64> = (1..n)
        .map(|k| {
            let kf = k as f64;
            kf / (4.0 * kf * kf - 1.0).sqrt()
        })
        .collect();

    // Compute eigenvalues of tridiagonal Jacobi matrix using QR iteration
    // (simplified symmetric QR with Wilkinson shift)
    let mut evecs = identity_matrix_f(n);
    qr_tridiag_iter(&mut diag, &mut off, &mut evecs, 1000, 1e-14);

    // Nodes are eigenvalues, weights are 2 * (first component of eigenvector)^2
    let nodes = diag.clone();
    let weights: Vec<f64> = (0..n).map(|i| 2.0 * evecs[0][i] * evecs[0][i]).collect();

    // Sort by node value
    let mut pairs: Vec<(f64, f64)> = nodes.into_iter().zip(weights).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let nodes_sorted: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let weights_sorted: Vec<f64> = pairs.iter().map(|p| p.1).collect();

    (nodes_sorted, weights_sorted)
}

fn identity_matrix_f(n: usize) -> Vec<Vec<f64>> {
    let mut q = vec![vec![0.0f64; n]; n];
    for (i, qi) in q.iter_mut().enumerate().take(n) {
        qi[i] = 1.0;
    }
    q
}

/// QR iteration for symmetric tridiagonal matrix (used for GL nodes).
fn qr_tridiag_iter(
    diag: &mut [f64],
    off: &mut [f64],
    evecs: &mut [Vec<f64>],
    max_iter: usize,
    tol: f64,
) {
    let n = diag.len();
    if n <= 1 {
        return;
    }

    let mut m = n;
    for _ in 0..max_iter {
        // Find the largest unreduced block
        while m > 1 && off[m - 2].abs() < tol * (diag[m - 2].abs() + diag[m - 1].abs()) {
            m -= 1;
        }
        if m <= 1 {
            break;
        }

        // Wilkinson shift
        let d = (diag[m - 2] - diag[m - 1]) / 2.0;
        let shift = diag[m - 1]
            - off[m - 2] * off[m - 2] / (d + d.signum() * (d * d + off[m - 2] * off[m - 2]).sqrt());

        // QR step with shift on subproblem [0..m]
        let mut g = diag[0] - shift;
        let mut h = off[0];

        for k in 0..m - 1 {
            let r = (g * g + h * h).sqrt();
            let cos_r = if r < 1e-300 { 1.0 } else { g / r };
            let sin_r = if r < 1e-300 { 0.0 } else { h / r };

            // Update diag and off
            if k > 0 {
                off[k - 1] = r;
            }
            let dk = diag[k];
            let dk1 = diag[k + 1];
            let ok = if k + 1 < off.len() { off[k + 1] } else { 0.0 };

            diag[k] = cos_r * cos_r * dk + 2.0 * cos_r * sin_r * off[k] + sin_r * sin_r * dk1;
            diag[k + 1] = sin_r * sin_r * dk - 2.0 * cos_r * sin_r * off[k] + cos_r * cos_r * dk1;
            off[k] = cos_r * sin_r * (dk1 - dk) + (cos_r * cos_r - sin_r * sin_r) * off[k];

            g = off[k];
            if k + 1 < m - 1 {
                h = -sin_r * ok;
                off[k + 1] = cos_r * ok;
            }

            // Accumulate eigenvectors
            {
                let (left, right) = evecs.split_at_mut(k + 1);
                let ek = &mut left[k];
                let ek1 = &mut right[0];
                for (vk, vk1) in ek.iter_mut().zip(ek1.iter_mut()) {
                    let old_vk = *vk;
                    let old_vk1 = *vk1;
                    *vk = cos_r * old_vk + sin_r * old_vk1;
                    *vk1 = -sin_r * old_vk + cos_r * old_vk1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Simple LCG for random number generation
// ---------------------------------------------------------------------------

struct SimpleLcg {
    state: u64,
}

impl SimpleLcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        // Map to (0, 1)
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_normal(&mut self) -> f64 {
        // Box-Muller transform
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feast_diagonal_matrix() {
        // 5×5 diagonal matrix with eigenvalues 1, 2, 3, 4, 5
        let n = 5;
        let mut a = vec![vec![0.0f64; n]; n];
        for (i, row) in a.iter_mut().enumerate() {
            row[i] = (i + 1) as f64;
        }

        // Find eigenvalues in [1.5, 3.5] → should find 2 and 3
        let config = FeastConfig {
            n_contour_pts: 8,
            subspace_size: 4,
            max_iter: 30,
            tol: 1e-6,
            interval: (1.5, 3.5),
        };

        let result = feast_eig(&a, &config).expect("FEAST failed");

        // Should find exactly eigenvalues 2 and 3
        assert!(
            !result.eigenvalues.is_empty(),
            "No eigenvalues found in interval"
        );
        for &ev in &result.eigenvalues {
            assert!(
                (1.5..=3.5).contains(&ev),
                "Eigenvalue {ev} outside interval [1.5, 3.5]"
            );
            // Should be close to 2 or 3
            let closest = [2.0f64, 3.0f64]
                .iter()
                .copied()
                .min_by(|a, b| {
                    (a - ev)
                        .abs()
                        .partial_cmp(&(b - ev).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(2.0);
            assert!(
                (ev - closest).abs() < 0.1,
                "Eigenvalue {ev} not close to 2 or 3"
            );
        }
    }

    #[test]
    fn test_gl_nodes_weights_sum() {
        // Sum of GL weights on [-1,1] should equal 2.
        let (nodes, weights) = gl_nodes_weights(5);
        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - 2.0).abs() < 1e-10,
            "GL weights sum is {sum}, expected 2.0"
        );

        // Nodes should be in (-1, 1)
        for n in &nodes {
            assert!(*n > -1.0 && *n < 1.0, "GL node {n} outside (-1, 1)");
        }
    }

    #[test]
    fn test_lu_solve() {
        // Test LU decomposition and solve
        let a = vec![vec![4.0, 3.0], vec![6.0, 3.0]];
        let (lu, perm) = lu_decompose(a).expect("LU failed");
        let b = vec![10.0, 12.0];
        let x = lu_solve(&lu, &perm, &b);
        // Verify: 4x + 3y = 10, 6x + 3y = 12 → x = 1, y = 2
        assert!((x[0] - 1.0).abs() < 1e-10, "x[0] = {}", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-10, "x[1] = {}", x[1]);
    }

    #[test]
    fn test_qr_orthonormality() {
        // Random-ish matrix for QR test
        let cols = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 1.0, 2.0],
            vec![2.0, 3.0, 1.0],
        ];
        let (q_cols, _) = qr_decompose(cols);
        // Check orthonormality
        let m = q_cols.len();
        for i in 0..m {
            let norm: f64 = q_cols[i].iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-10,
                "Column {i} not normalized: {norm}"
            );
            for j in i + 1..m {
                let dot: f64 = q_cols[i]
                    .iter()
                    .zip(q_cols[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                assert!(
                    dot.abs() < 1e-10,
                    "Columns {i} and {j} not orthogonal: dot={dot}"
                );
            }
        }
    }
}
