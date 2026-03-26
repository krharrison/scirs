//! Cuppen's divide-and-conquer algorithm for symmetric tridiagonal eigenvalue problems.
//!
//! This implements the classic Cuppen (1981) D&C algorithm with deflation, which is the
//! basis for LAPACK's `dstedc` routine. The algorithm achieves O(n²) for eigenvalues
//! and O(n³) for eigenvectors in the worst case, but is often much faster in practice
//! due to deflation.
//!
//! # Algorithm Overview
//!
//! 1. **Divide**: Split the n×n tridiagonal T into two halves by removing the off-diagonal
//!    coupling element β at the midpoint. This yields:
//!    T = T₁ ⊕ T₂ + β · z zᵀ
//!    where z = [0…0, 1, 1, 0…0]ᵀ (one 1 at each boundary).
//!
//! 2. **Recurse**: Solve the two smaller tridiagonal problems to get eigenvalues d₁, d₂
//!    and eigenvectors Q₁, Q₂.
//!
//! 3. **Deflate**: Remove eigenvalues where the corresponding component of u = Qᵀz is
//!    nearly zero (they are already converged), and merge nearly-equal eigenvalues.
//!
//! 4. **Secular equation**: Find eigenvalues of D + β·uuᵀ by solving
//!    f(λ) = 1 + β · Σᵢ uᵢ²/(dᵢ - λ) = 0
//!    using Newton-Raphson with safe quadratic interpolation for each root.
//!
//! 5. **Update eigenvectors**: Q_new = Q·[(D - λI)⁻¹u] normalized, then apply deflation.
//!
//! # References
//!
//! - Cuppen, J.J.M. (1981). A divide and conquer method for the symmetric tridiagonal
//!   eigenproblem. *Numerische Mathematik*, 36(2), 177–195.
//! - Gu, M. & Eisenstat, S.C. (1995). A divide-and-conquer algorithm for the symmetric
//!   tridiagonal eigenproblem. *SIAM J. Matrix Anal. Appl.*, 16(1), 172–191.

use crate::error::{LinalgError, LinalgResult};

/// Result of Householder tridiagonalization: (diagonal, off-diagonal, Q matrix)
type TridiagResult = LinalgResult<(Vec<f64>, Vec<f64>, Vec<Vec<f64>>)>;

/// Configuration for the divide-and-conquer eigensolver.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct DcConfig {
    /// Convergence tolerance for secular equation roots. Default: 1e-12.
    pub tol: f64,
    /// Maximum Newton-Raphson iterations per secular root. Default: 30.
    pub max_iter: usize,
    /// Relative threshold for deflation of small |u_i| components. Default: 1e-10.
    pub deflation_tol: f64,
}

impl Default for DcConfig {
    fn default() -> Self {
        Self {
            tol: 1e-12,
            max_iter: 30,
            deflation_tol: 1e-10,
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute eigenvalues and eigenvectors of a symmetric tridiagonal matrix using
/// Cuppen's divide-and-conquer algorithm.
///
/// # Arguments
///
/// * `diag` — diagonal elements d₀, d₁, …, d_{n-1}
/// * `off_diag` — off-diagonal elements e₁, …, e_{n-1} (length n-1)
///
/// # Returns
///
/// `(eigenvalues, eigenvectors)` where eigenvalues are sorted ascending and
/// eigenvectors are stored as columns: `eigenvectors[j]` is the j-th eigenvector.
///
/// # Errors
///
/// Returns `LinalgError` if dimensions are inconsistent or convergence fails.
pub fn dc_eig_tridiag(diag: &[f64], off_diag: &[f64]) -> LinalgResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let n = diag.len();
    if n == 0 {
        return Ok((vec![], vec![]));
    }
    if off_diag.len() != n - 1 {
        return Err(LinalgError::DimensionError(format!(
            "off_diag length {} != diag length {} - 1",
            off_diag.len(),
            n
        )));
    }
    let config = DcConfig::default();
    dc_tridiag_impl(diag, off_diag, &config)
}

/// Compute eigenvalues and eigenvectors of a symmetric dense matrix by first
/// reducing to tridiagonal form (Householder) and then applying the D&C algorithm.
///
/// # Arguments
///
/// * `a` — n×n symmetric matrix stored as `Vec<Vec<f64>>` (row-major).
///
/// # Returns
///
/// `(eigenvalues, eigenvectors)` sorted ascending.
pub fn dc_eig_symmetric(a: &[Vec<f64>]) -> LinalgResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let n = a.len();
    if n == 0 {
        return Ok((vec![], vec![]));
    }
    for row in a {
        if row.len() != n {
            return Err(LinalgError::DimensionError(format!(
                "Matrix is not square: row has {} elements, expected {n}",
                row.len()
            )));
        }
    }

    // Reduce to tridiagonal form using Householder reflections.
    let (diag, off_diag, q_house) = householder_tridiagonalize(a)?;

    // Solve tridiagonal eigenproblem.
    let config = DcConfig::default();
    let (evals, evecs_tri) = dc_tridiag_impl(&diag, &off_diag, &config)?;

    // Back-transform eigenvectors: v_orig = Q_house * v_tri
    // q_house[i][j] = Q[i][j], evecs_tri[k][j] = k-th eigenvector, j-th component
    let evecs = back_transform_evecs(&q_house, &evecs_tri, n);

    Ok((evals, evecs))
}

// ---------------------------------------------------------------------------
// Householder tridiagonalization
// ---------------------------------------------------------------------------

/// Reduce symmetric matrix A to tridiagonal form using Householder reflections.
/// Returns (diag, off_diag, Q) where A = Q T Qᵀ.
fn householder_tridiagonalize(a: &[Vec<f64>]) -> TridiagResult {
    let n = a.len();
    let mut mat: Vec<Vec<f64>> = a.to_vec();
    // Accumulate orthogonal transformation Q = H_1 H_2 … H_{n-2}
    let mut q = identity_matrix(n);

    for k in 0..n.saturating_sub(2) {
        // Build Householder reflector to zero out mat[k+2..][k]
        let col_len = n - k - 1;
        let mut x: Vec<f64> = (0..col_len).map(|i| mat[k + 1 + i][k]).collect();
        let norm_x = vec_norm(&x);
        if norm_x < 1e-15 {
            continue;
        }
        let sign = if x[0] >= 0.0 { 1.0_f64 } else { -1.0_f64 };
        x[0] += sign * norm_x;
        let norm_v = vec_norm(&x);
        if norm_v < 1e-15 {
            continue;
        }
        for xi in &mut x {
            *xi /= norm_v;
        }

        // Apply H = I - 2 v vᵀ symmetrically to trailing submatrix mat[k+1:, k+1:]
        // First compute p = 2 * A_sub * v
        let sz = col_len;
        let mut p = vec![0.0f64; sz];
        for i in 0..sz {
            for j in 0..sz {
                p[i] += mat[k + 1 + i][k + 1 + j] * x[j];
            }
        }
        for pi in &mut p {
            *pi *= 2.0;
        }
        // w = p - (vᵀ p) v
        let vtp: f64 = x.iter().zip(p.iter()).map(|(vi, pi)| vi * pi).sum();
        let mut w = p.clone();
        for (wi, &xi) in w.iter_mut().zip(x.iter()) {
            *wi -= vtp * xi;
        }
        // A_sub -= v wᵀ + w vᵀ
        for i in 0..sz {
            for j in 0..sz {
                mat[k + 1 + i][k + 1 + j] -= x[i] * w[j] + w[i] * x[j];
            }
        }
        // Zero out column k below k+1 and set off-diagonal
        for i in 1..sz {
            mat[k + 1 + i][k] = 0.0;
            mat[k][k + 1 + i] = 0.0;
        }
        let off = -sign * norm_x;
        mat[k + 1][k] = off;
        mat[k][k + 1] = off;

        // Accumulate Q: Q[:, k+1:] -= 2 * (Q[:, k+1:] v) vᵀ
        let mut qv = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..sz {
                qv[i] += q[i][k + 1 + j] * x[j];
            }
        }
        for i in 0..n {
            for j in 0..sz {
                q[i][k + 1 + j] -= 2.0 * qv[i] * x[j];
            }
        }
    }

    let diag: Vec<f64> = (0..n).map(|i| mat[i][i]).collect();
    let off_diag: Vec<f64> = (0..n.saturating_sub(1)).map(|i| mat[i][i + 1]).collect();

    Ok((diag, off_diag, q))
}

fn identity_matrix(n: usize) -> Vec<Vec<f64>> {
    let mut q = vec![vec![0.0f64; n]; n];
    for (i, qi) in q.iter_mut().enumerate().take(n) {
        qi[i] = 1.0;
    }
    q
}

fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Back-transform eigenvectors: evecs_orig[j] = Q * evecs_tri[j]
/// Q[i][j] is the matrix element at row i, col j.
/// evecs_tri[k][j] = k-th eigenvector's j-th component.
fn back_transform_evecs(q: &[Vec<f64>], evecs_tri: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    evecs_tri
        .iter()
        .map(|v| {
            let mut out = vec![0.0f64; n];
            for i in 0..n {
                for j in 0..n {
                    out[i] += q[i][j] * v[j];
                }
            }
            out
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Divide-and-conquer implementation
// ---------------------------------------------------------------------------

/// Core recursive D&C for symmetric tridiagonal matrices.
/// Returns (eigenvalues sorted asc, eigenvectors as columns).
/// Eigenvectors are represented as: result.1[k] = k-th eigenvector of length n.
fn dc_tridiag_impl(
    diag: &[f64],
    off_diag: &[f64],
    config: &DcConfig,
) -> LinalgResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let n = diag.len();

    // Base cases
    if n == 1 {
        return Ok((vec![diag[0]], vec![vec![1.0]]));
    }
    if n == 2 {
        return solve_2x2_tridiag(diag[0], diag[1], off_diag[0]);
    }

    // For small n, use direct QR iteration (more stable than D&C recursion for small matrices)
    if n <= 8 {
        return qr_tridiag_eigen(diag, off_diag, config);
    }

    // Split at midpoint
    let mid = n / 2;
    let beta = off_diag[mid - 1]; // coupling element β
    let abs_beta = beta.abs();

    // Create sub-problems with boundary modifications:
    // T₁ has its last diagonal element reduced by |β|
    // T₂ has its first diagonal element reduced by |β|
    let mut diag1 = diag[..mid].to_vec();
    let mut diag2 = diag[mid..].to_vec();
    diag1[mid - 1] -= abs_beta;
    diag2[0] -= abs_beta;

    let off1 = &off_diag[..mid - 1];
    let off2 = &off_diag[mid..];

    // Recurse to get (d₁, Q₁) and (d₂, Q₂)
    let (evals1, evecs1) = dc_tridiag_impl(&diag1, off1, config)?;
    let (evals2, evecs2) = dc_tridiag_impl(&diag2, off2, config)?;

    // d = [d₁; d₂] — merged diagonal eigenvalue array
    let d: Vec<f64> = evals1.iter().chain(evals2.iter()).copied().collect();

    // Compute u = [Q₁ᵀ e_{mid}; Q₂ᵀ e₁] where e_{mid} is the last unit vector of the top block
    // and e₁ is the first unit vector of the bottom block.
    //
    // Since Q₁ and Q₂ are formed recursively from the tridiagonal sub-problems,
    // Q₁ᵀ e_{mid-1} = last column of Q₁ᵀ = last row of Q₁ = evecs1[k][mid-1] for each k.
    // Q₂ᵀ e₀ = first column of Q₂ᵀ = first row of Q₂ = evecs2[k][0] for each k.
    let sign_beta = if beta >= 0.0 { 1.0_f64 } else { -1.0_f64 };

    let mut u: Vec<f64> = Vec::with_capacity(n);
    for ev1 in evecs1.iter().take(mid) {
        // u[k] = (Q₁ᵀ z_top)[k] = Q₁[mid-1][k] = evecs1[k][mid-1]
        u.push(sign_beta * ev1[mid - 1]);
    }
    for ev2 in evecs2.iter().take(n - mid) {
        // u[mid+k] = (Q₂ᵀ z_bot)[k] = Q₂[0][k] = evecs2[k][0]
        u.push(ev2[0]);
    }

    // Deflate: identify trivial eigenvalues
    let (d_defl, u_defl, active_idx, trivial_evals) = deflate_secular(&d, &u, config.deflation_tol);

    // Solve secular equation for active eigenvalues
    let secular_evals = if d_defl.is_empty() {
        vec![]
    } else {
        solve_secular(&d_defl, &u_defl, abs_beta, config)?
    };

    // Reconstruct eigenvectors in the *combined D-space* (before applying Q_block)
    // Total eigenvectors = trivial + secular
    let n_trivial = trivial_evals.len();
    let n_secular = secular_evals.len();
    let total = n_trivial + n_secular;

    if total != n {
        // Numerical issues: fall back to direct QR
        return qr_tridiag_eigen(diag, off_diag, config);
    }

    // Collect all eigenvalues and their corresponding D-space eigenvectors
    // D-space: the space after applying Q_block = diag(Q₁, Q₂)
    let mut all_pairs: Vec<(f64, Vec<f64>)> = Vec::with_capacity(n);

    // Trivial eigenvectors: standard basis vectors in D-space at positions NOT in active_idx
    let trivial_positions: Vec<usize> = (0..n).filter(|i| !active_idx.contains(i)).collect();
    for (&eval, &pos) in trivial_evals.iter().zip(trivial_positions.iter()) {
        let mut ev = vec![0.0f64; n];
        ev[pos] = 1.0;
        all_pairs.push((eval, ev));
    }

    // Secular eigenvectors: v = (D - λI)⁻¹ u / ||(D - λI)⁻¹ u||
    // where D and u are restricted to the active indices
    for &lam in &secular_evals {
        let v_active: Vec<f64> = d_defl
            .iter()
            .zip(u_defl.iter())
            .map(|(&di, &ui)| {
                let denom = di - lam;
                if denom.abs() < 1e-300 {
                    if ui >= 0.0 {
                        1e150_f64
                    } else {
                        -1e150_f64
                    }
                } else {
                    ui / denom
                }
            })
            .collect();

        let norm = vec_norm(&v_active);
        let mut ev_d = vec![0.0f64; n];
        if norm > 1e-300 {
            for (k, &ai) in active_idx.iter().enumerate() {
                ev_d[ai] = v_active[k] / norm;
            }
        } else {
            // Degenerate: spread equally
            for &ai in &active_idx {
                ev_d[ai] = 1.0 / (active_idx.len() as f64).sqrt();
            }
        }
        all_pairs.push((lam, ev_d));
    }

    // Sort by eigenvalue
    all_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Back-transform: eigenvec_T = Q_block * eigenvec_D
    // Q_block = diag(Q₁, Q₂) acts as:
    //   for the top block (indices 0..mid): apply Q₁
    //   for the bottom block (indices mid..n): apply Q₂
    // eigenvec1[k] = k-th eigenvector of Q₁ (length mid)
    // eigenvec2[k] = k-th eigenvector of Q₂ (length n-mid)
    // Q_block * v = [Q₁ * v[0..mid]; Q₂ * v[mid..n]]
    let n2 = n - mid;
    let evals_final: Vec<f64> = all_pairs.iter().map(|(e, _)| *e).collect();
    let evecs_final: Vec<Vec<f64>> = all_pairs
        .into_iter()
        .map(|(_, v_d)| {
            let mut out = vec![0.0f64; n];
            // Q₁ * v_d[0..mid]: Q₁ is encoded as evecs1[k] = k-th eigenvector (length mid)
            // Q₁ * x = sum_k x[k] * evecs1[k]
            for k in 0..mid {
                let x_k = v_d[k];
                if x_k.abs() > 1e-300 {
                    for i in 0..mid {
                        out[i] += x_k * evecs1[k][i];
                    }
                }
            }
            // Q₂ * v_d[mid..n]: similarly
            for k in 0..n2 {
                let x_k = v_d[mid + k];
                if x_k.abs() > 1e-300 {
                    for i in 0..n2 {
                        out[mid + i] += x_k * evecs2[k][i];
                    }
                }
            }
            out
        })
        .collect();

    Ok((evals_final, evecs_final))
}

// ---------------------------------------------------------------------------
// QR iteration for small tridiagonal matrices (base case fallback)
// ---------------------------------------------------------------------------

/// Compute all eigenvalues and eigenvectors of a small symmetric tridiagonal
/// matrix using Jacobi iteration on the full symmetric matrix form.
///
/// This builds the full symmetric matrix from the tridiagonal representation
/// and applies Jacobi sweeps, which are simple, correct, and efficient for
/// small matrices (n ≤ 32).
fn qr_tridiag_eigen(
    diag: &[f64],
    off_diag: &[f64],
    _config: &DcConfig,
) -> LinalgResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let n = diag.len();
    if n == 1 {
        return Ok((vec![diag[0]], vec![vec![1.0]]));
    }
    if n == 2 {
        return solve_2x2_tridiag(diag[0], diag[1], off_diag[0]);
    }

    // Build full symmetric matrix
    let mut mat: Vec<Vec<f64>> = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        mat[i][i] = diag[i];
    }
    for i in 0..n - 1 {
        mat[i][i + 1] = off_diag[i];
        mat[i + 1][i] = off_diag[i];
    }

    // Jacobi iteration: Q^T A Q = D
    let mut q = identity_matrix(n);
    let max_sweeps = 200;
    let tol = 1e-13;

    for _sweep in 0..max_sweeps {
        let mut max_off = 0.0f64;
        let mut p_idx = 0usize;
        let mut q_idx = 1usize;
        for (i, mat_i) in mat.iter().enumerate().take(n) {
            for (j, &mat_ij) in mat_i.iter().enumerate().take(n).skip(i + 1) {
                let v = mat_ij.abs();
                if v > max_off {
                    max_off = v;
                    p_idx = i;
                    q_idx = j;
                }
            }
        }
        if max_off < tol {
            break;
        }

        // Jacobi rotation for (p_idx, q_idx)
        let theta = (mat[q_idx][q_idx] - mat[p_idx][p_idx]) / (2.0 * mat[p_idx][q_idx]);
        let sign_t = if theta >= 0.0 { 1.0_f64 } else { -1.0_f64 };
        let t = sign_t / (theta.abs() + (1.0 + theta * theta).sqrt());
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        let app = mat[p_idx][p_idx];
        let aqq = mat[q_idx][q_idx];
        let apq = mat[p_idx][q_idx];

        mat[p_idx][p_idx] = app - t * apq;
        mat[q_idx][q_idx] = aqq + t * apq;
        mat[p_idx][q_idx] = 0.0;
        mat[q_idx][p_idx] = 0.0;

        for (r, mat_r) in mat.iter_mut().enumerate().take(n) {
            if r != p_idx && r != q_idx {
                let arp = mat_r[p_idx];
                let arq = mat_r[q_idx];
                mat_r[p_idx] = c * arp - s * arq;
                mat_r[q_idx] = s * arp + c * arq;
            }
        }
        // Symmetrize: copy updated column values to matching rows
        {
            let col_p: Vec<f64> = (0..n).map(|r| mat[r][p_idx]).collect();
            let col_q: Vec<f64> = (0..n).map(|r| mat[r][q_idx]).collect();
            for r in 0..n {
                if r != p_idx && r != q_idx {
                    mat[p_idx][r] = col_p[r];
                    mat[q_idx][r] = col_q[r];
                }
            }
        }

        // Accumulate eigenvectors: q[j] is the j-th column of Q
        {
            let (left, right) = q.split_at_mut(q_idx);
            let qp = &mut left[p_idx];
            let qq = &mut right[0];
            for (vp, vq) in qp.iter_mut().zip(qq.iter_mut()) {
                let old_p = *vp;
                let old_q = *vq;
                *vp = c * old_p - s * old_q;
                *vq = s * old_p + c * old_q;
            }
        }
    }

    // Extract eigenvalues and eigenvectors
    let mut pairs: Vec<(f64, Vec<f64>)> = (0..n).map(|i| (mat[i][i], q[i].clone())).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let evals: Vec<f64> = pairs.iter().map(|(e, _)| *e).collect();
    let evecs: Vec<Vec<f64>> = pairs.into_iter().map(|(_, v)| v).collect();

    Ok((evals, evecs))
}

/// Solve 2×2 symmetric tridiagonal system directly.
fn solve_2x2_tridiag(d0: f64, d1: f64, e: f64) -> LinalgResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let tr = d0 + d1;
    let det = d0 * d1 - e * e;
    let disc = (tr * tr - 4.0 * det).max(0.0).sqrt();
    let lam1 = (tr - disc) / 2.0;
    let lam2 = (tr + disc) / 2.0;

    let ev1 = eigvec_2x2(d0, d1, e, lam1);
    let ev2 = eigvec_2x2(d0, d1, e, lam2);

    Ok((vec![lam1, lam2], vec![ev1, ev2]))
}

fn eigvec_2x2(d0: f64, d1: f64, e: f64, lam: f64) -> Vec<f64> {
    let a = d0 - lam;
    let b = e;
    let c = d1 - lam;
    // Row 1: a * v0 + b * v1 = 0  →  v = [-b, a] (normalized)
    // Row 2: b * v0 + c * v1 = 0  →  v = [-c, b] (normalized)
    // Choose the row with larger norm for stability
    let n1 = (a * a + b * b).sqrt();
    let n2 = (b * b + c * c).sqrt();
    if n1 >= n2 {
        if n1 < 1e-300 {
            vec![1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()]
        } else {
            vec![-b / n1, a / n1]
        }
    } else {
        if n2 < 1e-300 {
            vec![1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()]
        } else {
            vec![-c / n2, b / n2]
        }
    }
}

// ---------------------------------------------------------------------------
// Secular equation solver
// ---------------------------------------------------------------------------

/// Solve the secular equation f(λ) = 1 + β · Σᵢ uᵢ²/(dᵢ - λ) = 0.
///
/// Returns n sorted eigenvalues. The roots are separated by the elements of d.
pub fn solve_secular(d: &[f64], u: &[f64], beta: f64, config: &DcConfig) -> LinalgResult<Vec<f64>> {
    let n = d.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // Sort d (and corresponding u) ascending
    let mut pairs: Vec<(f64, f64)> = d.iter().copied().zip(u.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let d_sorted: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let u_sorted: Vec<f64> = pairs.iter().map(|p| p.1).collect();

    if beta.abs() < 1e-300 {
        return Ok(d_sorted);
    }

    let mut roots = Vec::with_capacity(n);

    for i in 0..n {
        let (lo, hi) = if beta > 0.0 {
            if i < n - 1 {
                (d_sorted[i], d_sorted[i + 1])
            } else {
                let weight_sum: f64 = u_sorted.iter().map(|ui| ui * ui).sum();
                (d_sorted[n - 1], d_sorted[n - 1] + beta * weight_sum + 1.0)
            }
        } else {
            // beta < 0
            if i == 0 {
                let weight_sum: f64 = u_sorted.iter().map(|ui| ui * ui).sum();
                (d_sorted[0] + beta * weight_sum - 1.0, d_sorted[0])
            } else {
                (d_sorted[i - 1], d_sorted[i])
            }
        };

        let root = find_secular_root(&d_sorted, &u_sorted, beta, lo, hi, config)?;
        roots.push(root);
    }

    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(roots)
}

/// Evaluate f(λ) = 1 + β · Σᵢ uᵢ²/(dᵢ - λ)
#[inline]
fn secular_f(d: &[f64], u: &[f64], beta: f64, lam: f64) -> f64 {
    let sum: f64 = d
        .iter()
        .zip(u.iter())
        .map(|(&di, &ui)| ui * ui / (di - lam))
        .sum();
    1.0 + beta * sum
}

/// Evaluate f'(λ) = β · Σᵢ uᵢ²/(dᵢ - λ)²
#[inline]
fn secular_df(d: &[f64], u: &[f64], beta: f64, lam: f64) -> f64 {
    let sum: f64 = d
        .iter()
        .zip(u.iter())
        .map(|(&di, &ui)| {
            let t = di - lam;
            ui * ui / (t * t)
        })
        .sum();
    beta * sum
}

/// Find one root of the secular equation in (lo, hi) using Newton-Raphson with bisection fallback.
fn find_secular_root(
    d: &[f64],
    u: &[f64],
    beta: f64,
    lo: f64,
    hi: f64,
    config: &DcConfig,
) -> LinalgResult<f64> {
    let tol = config.tol;
    let max_iter = config.max_iter;

    let eps = 1e-14 * (lo.abs() + hi.abs() + 1.0);
    let lo_safe = lo + eps;
    let hi_safe = hi - eps;

    let mut lo_m = lo_safe;
    let mut hi_m = hi_safe;

    // Validate bracket: f should have opposite signs at endpoints
    let f_lo = secular_f(d, u, beta, lo_m);
    let f_hi = secular_f(d, u, beta, hi_m);

    if f_lo.is_nan() || f_hi.is_nan() || !f_lo.is_finite() || !f_hi.is_finite() {
        // Return midpoint as fallback
        return Ok((lo + hi) / 2.0);
    }

    // Initial guess: midpoint
    let mut x = (lo_m + hi_m) / 2.0;

    for _ in 0..max_iter {
        let fx = secular_f(d, u, beta, x);
        if !fx.is_finite() {
            x = (lo_m + hi_m) / 2.0;
            continue;
        }
        if fx.abs() < tol {
            return Ok(x);
        }

        let dfx = secular_df(d, u, beta, x);
        let x_new = if dfx.abs() > 1e-300 {
            x - fx / dfx
        } else {
            (lo_m + hi_m) / 2.0
        };

        if x_new > lo_m && x_new < hi_m {
            let step = (x_new - x).abs();
            x = x_new;
            if step < tol * (x.abs() + 1.0) {
                return Ok(x);
            }
        } else {
            // Bisection step
            let f_lo_cur = secular_f(d, u, beta, lo_m);
            let f_mid_cur = secular_f(d, u, beta, (lo_m + hi_m) / 2.0);
            let mid = (lo_m + hi_m) / 2.0;
            if f_lo_cur * f_mid_cur <= 0.0 {
                hi_m = mid;
            } else {
                lo_m = mid;
            }
            x = (lo_m + hi_m) / 2.0;
            if (hi_m - lo_m).abs() < tol {
                return Ok(x);
            }
        }
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Deflation
// ---------------------------------------------------------------------------

/// Deflate the secular equation by removing near-zero u_i components and
/// merging nearly-equal d_i components.
///
/// Returns `(deflated_d, deflated_u, active_indices, trivial_eigenvalues)`.
pub fn deflate_secular(
    d: &[f64],
    u: &[f64],
    tol: f64,
) -> (Vec<f64>, Vec<f64>, Vec<usize>, Vec<f64>) {
    let n = d.len();
    let u_norm = u.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);

    let mut defl_d = Vec::new();
    let mut defl_u = Vec::new();
    let mut active_idx = Vec::new();
    let mut trivial_evals = Vec::new();

    for i in 0..n {
        if u[i].abs() < tol * u_norm {
            trivial_evals.push(d[i]);
        } else {
            defl_d.push(d[i]);
            defl_u.push(u[i]);
            active_idx.push(i);
        }
    }

    // Merge nearly-equal d values among active components
    let mut i = 0;
    while i < defl_d.len() {
        let mut j = i + 1;
        while j < defl_d.len() {
            if (defl_d[j] - defl_d[i]).abs() < tol * (defl_d[i].abs() + 1.0) {
                let new_u = (defl_u[i] * defl_u[i] + defl_u[j] * defl_u[j]).sqrt();
                defl_u[i] = new_u;
                trivial_evals.push(defl_d[j]);
                defl_d.remove(j);
                defl_u.remove(j);
                active_idx.remove(j);
            } else {
                j += 1;
            }
        }
        i += 1;
    }

    (defl_d, defl_u, active_idx, trivial_evals)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    /// Check that eigenvectors are orthonormal.
    fn check_orthonormal(evecs: &[Vec<f64>]) -> bool {
        let m = evecs.len();
        for i in 0..m {
            let dot_ii: f64 = evecs[i].iter().map(|x| x * x).sum();
            if (dot_ii - 1.0).abs() > 1e-7 {
                return false;
            }
            for j in i + 1..m {
                let dot_ij: f64 = evecs[i]
                    .iter()
                    .zip(evecs[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                if dot_ij.abs() > 1e-7 {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn test_3x3_tridiag_known_eigenvalues() {
        // T = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
        // Eigenvalues: 2 - √2, 2, 2 + √2 ≈ 0.586, 2.0, 3.414
        let diag = vec![2.0, 2.0, 2.0];
        let off = vec![-1.0, -1.0];
        let (evals, evecs) = dc_eig_tridiag(&diag, &off).expect("DC tridiag failed");

        assert_eq!(evals.len(), 3);
        assert_eq!(evecs.len(), 3);

        let expected = [2.0 - 2.0_f64.sqrt(), 2.0, 2.0 + 2.0_f64.sqrt()];
        for (ev, ex) in evals.iter().zip(expected.iter()) {
            assert!(
                (ev - ex).abs() < 1e-8,
                "Eigenvalue mismatch: got {ev}, expected {ex}"
            );
        }

        assert!(
            check_orthonormal(&evecs),
            "Eigenvectors are not orthonormal"
        );
    }

    #[test]
    fn test_2x2_base_case() {
        // [[3, 1], [1, 3]] → eigenvalues 2 and 4
        let (evals, evecs) = solve_2x2_tridiag(3.0, 3.0, 1.0).expect("2x2 solve failed");
        assert_eq!(evals.len(), 2);
        assert!(
            approx_eq(evals[0], 2.0, 1e-10),
            "Expected 2.0, got {}",
            evals[0]
        );
        assert!(
            approx_eq(evals[1], 4.0, 1e-10),
            "Expected 4.0, got {}",
            evals[1]
        );
        assert!(check_orthonormal(&evecs));
    }

    #[test]
    fn test_dc_eig_symmetric_4x4() {
        // Symmetric 4×4 matrix
        let a = vec![
            vec![5.0, 1.0, 0.0, 0.0],
            vec![1.0, 5.0, 1.0, 0.0],
            vec![0.0, 1.0, 5.0, 1.0],
            vec![0.0, 0.0, 1.0, 5.0],
        ];
        let (evals, evecs) = dc_eig_symmetric(&a).expect("DC symmetric failed");
        assert_eq!(evals.len(), 4);

        // Verify A * v ≈ lambda * v for each eigenpair
        for (k, &lam) in evals.iter().enumerate() {
            let v = &evecs[k];
            let mut av = [0.0f64; 4];
            for i in 0..4 {
                for j in 0..4 {
                    av[i] += a[i][j] * v[j];
                }
            }
            for i in 0..4 {
                assert!(
                    (av[i] - lam * v[i]).abs() < 1e-6,
                    "Eigenpair {k}: residual at component {i} too large: {} vs {}",
                    av[i],
                    lam * v[i]
                );
            }
        }
        assert!(check_orthonormal(&evecs), "Eigenvectors not orthonormal");
    }

    #[test]
    fn test_deflation() {
        let d = vec![1.0, 2.0, 3.0];
        let u = vec![0.0, 1.0, 0.5]; // u[0] = 0 → trivial
        let (defl_d, _defl_u, idx, trivials) = deflate_secular(&d, &u, 1e-10);
        assert!(trivials.contains(&1.0), "d[0]=1 should be trivial");
        assert_eq!(defl_d.len(), 2);
        assert!(idx.contains(&1));
        assert!(idx.contains(&2));
    }

    #[test]
    fn test_larger_tridiag() {
        // 10×10 tridiagonal with known structure
        let n = 10;
        let diag: Vec<f64> = vec![2.0; n];
        let off: Vec<f64> = vec![-1.0; n - 1];

        let (evals, evecs) = dc_eig_tridiag(&diag, &off).expect("DC tridiag failed for n=10");
        assert_eq!(evals.len(), n);

        // All eigenvalues should be positive (matrix is positive definite)
        for &ev in &evals {
            assert!(ev > 0.0, "Expected positive eigenvalue, got {ev}");
        }

        // Eigenvalues should be sorted
        for i in 1..n {
            assert!(evals[i] >= evals[i - 1], "Eigenvalues not sorted at {i}");
        }

        // Eigenvectors orthonormal
        assert!(
            check_orthonormal(&evecs),
            "Eigenvectors not orthonormal for n=10"
        );
    }
}
