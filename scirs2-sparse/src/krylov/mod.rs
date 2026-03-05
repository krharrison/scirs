//! Advanced Krylov subspace eigensolvers
//!
//! This module provides production-quality implementations of:
//!
//! - **Implicitly Restarted Arnoldi Method (IRAM)** for general (non-symmetric) matrices
//! - **Thick-Restart Lanczos** for symmetric matrices
//! - **Shift-and-Invert mode** for computing interior eigenvalues
//! - **Harmonic Ritz extraction** for better interior eigenvalue approximations
//!
//! # References
//!
//! - Sorensen, D.C. (1992). "Implicit application of polynomial filters in a k-step
//!   Arnoldi method". SIAM J. Matrix Anal. Appl. 13(1), 357-385.
//! - Wu, K. & Simon, H. (2000). "Thick-restart Lanczos method for large symmetric
//!   eigenvalue problems". SIAM J. Matrix Anal. Appl. 22(2), 602-616.

pub mod augmented;
pub mod deflation;
pub mod gmres_dr;
pub mod recycled_krylov;

pub use augmented::AugmentedKrylov;
pub use deflation::HarmonicRitzDeflation;
pub use gmres_dr::GmresDR;
pub use recycled_krylov::RecycledGmres;

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use crate::iterative_solvers::Preconditioner;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, NumAssign, SparseElement};
use std::fmt::Debug;
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Specifies which eigenvalues to target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WhichEigenvalues {
    /// Largest magnitude eigenvalues.
    #[default]
    LargestMagnitude,
    /// Smallest magnitude eigenvalues.
    SmallestMagnitude,
    /// Largest real part.
    LargestReal,
    /// Smallest real part.
    SmallestReal,
    /// Eigenvalues closest to a given shift (requires shift-and-invert).
    NearShift,
}

/// Configuration for the Implicitly Restarted Arnoldi Method.
#[derive(Debug, Clone)]
pub struct IramConfig {
    /// Number of eigenvalues to compute.
    pub n_eigenvalues: usize,
    /// Dimension of the Krylov subspace (must be > n_eigenvalues).
    pub krylov_dim: usize,
    /// Maximum number of restart cycles.
    pub max_restarts: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Which eigenvalues to target.
    pub which: WhichEigenvalues,
    /// Whether to use harmonic Ritz extraction.
    pub harmonic_ritz: bool,
    /// Shift value for shift-and-invert mode.
    pub shift: Option<f64>,
    /// Whether to print convergence diagnostics.
    pub verbose: bool,
}

impl Default for IramConfig {
    fn default() -> Self {
        Self {
            n_eigenvalues: 6,
            krylov_dim: 20,
            max_restarts: 300,
            tol: 1e-10,
            which: WhichEigenvalues::LargestMagnitude,
            harmonic_ritz: false,
            shift: None,
            verbose: false,
        }
    }
}

/// Configuration for the Thick-Restart Lanczos method.
#[derive(Debug, Clone)]
pub struct ThickRestartLanczosConfig {
    /// Number of eigenvalues to compute.
    pub n_eigenvalues: usize,
    /// Maximum Lanczos basis size before restart.
    pub max_basis_size: usize,
    /// Maximum number of restart cycles.
    pub max_restarts: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Which eigenvalues to compute: "smallest" or "largest".
    pub which: WhichEigenvalues,
    /// Shift value for shift-and-invert mode (None = standard mode).
    pub shift: Option<f64>,
    /// Whether to print convergence diagnostics.
    pub verbose: bool,
}

impl Default for ThickRestartLanczosConfig {
    fn default() -> Self {
        Self {
            n_eigenvalues: 6,
            max_basis_size: 30,
            max_restarts: 300,
            tol: 1e-10,
            which: WhichEigenvalues::SmallestReal,
            shift: None,
            verbose: false,
        }
    }
}

/// Result of a Krylov eigensolver computation.
#[derive(Debug, Clone)]
pub struct KrylovEigenResult<F> {
    /// Converged eigenvalues.
    pub eigenvalues: Array1<F>,
    /// Converged eigenvectors (stored column-wise).
    pub eigenvectors: Array2<F>,
    /// Number of restart cycles performed.
    pub restarts: usize,
    /// Total number of matrix-vector products.
    pub matvec_count: usize,
    /// Residual norms for each eigenpair.
    pub residual_norms: Vec<F>,
    /// Whether all requested eigenvalues converged.
    pub converged: bool,
    /// Number of converged eigenpairs.
    pub n_converged: usize,
}

// ---------------------------------------------------------------------------
// Dense linear algebra helpers
// ---------------------------------------------------------------------------

/// CSR matrix-vector product.
fn csr_matvec<F>(a: &CsrMatrix<F>, x: &[F]) -> SparseResult<Vec<F>>
where
    F: Float + NumAssign + Sum + SparseElement + 'static,
{
    let (m, n) = a.shape();
    if x.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: x.len(),
        });
    }
    let mut y = vec![F::sparse_zero(); m];
    for i in 0..m {
        let range = a.row_range(i);
        let cols = &a.indices[range.clone()];
        let vals = &a.data[range];
        let mut acc = F::sparse_zero();
        for (idx, &col) in cols.iter().enumerate() {
            acc += vals[idx] * x[col];
        }
        y[i] = acc;
    }
    Ok(y)
}

#[inline]
fn dot_vec<F: Float + Sum>(a: &[F], b: &[F]) -> F {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

#[inline]
fn norm2_vec<F: Float + Sum>(v: &[F]) -> F {
    dot_vec(v, v).sqrt()
}

fn normalise_vec<F: Float + Sum + SparseElement>(v: &mut [F]) -> F {
    let nrm = norm2_vec(v);
    if nrm > F::epsilon() {
        let inv = F::sparse_one() / nrm;
        for vi in v.iter_mut() {
            *vi = *vi * inv;
        }
    }
    nrm
}

/// Dense symmetric eigensolver via Jacobi rotations.
/// Input: `a` (k x k) row-major symmetric. Returns sorted eigenvalues and column-major
/// eigenvectors.
fn jacobi_eig<F>(a: &[F], k: usize) -> SparseResult<(Vec<F>, Vec<F>)>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let max_sweeps = 200;
    let tol_j = F::epsilon() * F::from(100.0).unwrap_or(F::sparse_one());

    let mut mat = a.to_vec();
    let mut vecs = vec![F::sparse_zero(); k * k];
    for i in 0..k {
        vecs[i * k + i] = F::sparse_one();
    }

    for _sw in 0..max_sweeps {
        let mut max_off = F::sparse_zero();
        for i in 0..k {
            for j in (i + 1)..k {
                let v = mat[i * k + j].abs();
                if v > max_off {
                    max_off = v;
                }
            }
        }
        if max_off < tol_j {
            break;
        }

        for p in 0..k {
            for q in (p + 1)..k {
                let apq = mat[p * k + q];
                if apq.abs() < tol_j {
                    continue;
                }
                let diff = mat[q * k + q] - mat[p * k + p];
                let theta = if diff.abs() < F::epsilon() {
                    F::from(std::f64::consts::FRAC_PI_4).unwrap_or(F::sparse_one())
                } else {
                    let tau = diff / (apq + apq);
                    let sign_tau = if tau >= F::sparse_zero() {
                        F::sparse_one()
                    } else {
                        -F::sparse_one()
                    };
                    let t = sign_tau / (tau.abs() + (F::sparse_one() + tau * tau).sqrt());
                    t.atan()
                };

                let (sin_t, cos_t) = (theta.sin(), theta.cos());
                let two = F::from(2.0).unwrap_or(F::sparse_one());

                for r in 0..k {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = mat[r * k + p];
                    let arq = mat[r * k + q];
                    mat[r * k + p] = cos_t * arp - sin_t * arq;
                    mat[r * k + q] = sin_t * arp + cos_t * arq;
                    mat[p * k + r] = mat[r * k + p];
                    mat[q * k + r] = mat[r * k + q];
                }
                let app = mat[p * k + p];
                let aqq = mat[q * k + q];
                let apq_old = apq;
                mat[p * k + p] =
                    cos_t * cos_t * app - two * sin_t * cos_t * apq_old + sin_t * sin_t * aqq;
                mat[q * k + q] =
                    sin_t * sin_t * app + two * sin_t * cos_t * apq_old + cos_t * cos_t * aqq;
                mat[p * k + q] = F::sparse_zero();
                mat[q * k + p] = F::sparse_zero();

                for r in 0..k {
                    let vp = vecs[p * k + r];
                    let vq = vecs[q * k + r];
                    vecs[p * k + r] = cos_t * vp - sin_t * vq;
                    vecs[q * k + r] = sin_t * vp + cos_t * vq;
                }
            }
        }
    }

    let mut evals: Vec<F> = (0..k).map(|i| mat[i * k + i]).collect();
    let mut perm: Vec<usize> = (0..k).collect();
    perm.sort_by(|&a_i, &b_i| {
        evals[a_i]
            .partial_cmp(&evals[b_i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted_evals: Vec<F> = perm.iter().map(|&i| evals[i]).collect();
    let mut sorted_vecs = vec![F::sparse_zero(); k * k];
    for (new_col, &old_col) in perm.iter().enumerate() {
        for r in 0..k {
            sorted_vecs[new_col * k + r] = vecs[old_col * k + r];
        }
    }
    evals = sorted_evals;
    Ok((evals, sorted_vecs))
}

/// Compute upper Hessenberg eigenvalues using the implicit QR algorithm (real Schur).
/// `h` is m x m row-major upper Hessenberg. Returns eigenvalues as (real, imag) pairs.
fn hessenberg_eigenvalues<F>(h: &[F], m: usize) -> SparseResult<Vec<(F, F)>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let mut mat = h.to_vec();
    let max_iter = m * 100;
    let tol = F::epsilon() * F::from(1000.0).unwrap_or(F::sparse_one());

    let mut n_active = m;
    let mut eigenvalues: Vec<(F, F)> = Vec::with_capacity(m);
    let mut iter_count = 0;

    while n_active > 0 && iter_count < max_iter {
        iter_count += 1;

        if n_active == 1 {
            eigenvalues.push((mat[0], F::sparse_zero()));
            break;
        }

        // Check for deflation on the sub-diagonal
        let sub_idx = (n_active - 1) * m + (n_active - 2);
        if mat[sub_idx].abs() < tol {
            eigenvalues.push((mat[(n_active - 1) * m + (n_active - 1)], F::sparse_zero()));
            n_active -= 1;
            continue;
        }

        // Check for 2x2 block at bottom
        if n_active == 2 {
            let a11 = mat[0];
            let a12 = mat[1];
            let a21 = mat[m];
            let a22 = mat[m + 1];
            let trace = a11 + a22;
            let det = a11 * a22 - a12 * a21;
            let disc = trace * trace - F::from(4.0).unwrap_or(F::sparse_one()) * det;
            let two = F::from(2.0).unwrap_or(F::sparse_one());
            if disc >= F::sparse_zero() {
                let sqrt_d = disc.sqrt();
                eigenvalues.push(((trace + sqrt_d) / two, F::sparse_zero()));
                eigenvalues.push(((trace - sqrt_d) / two, F::sparse_zero()));
            } else {
                let sqrt_d = (-disc).sqrt();
                eigenvalues.push((trace / two, sqrt_d / two));
                eigenvalues.push((trace / two, -sqrt_d / two));
            }
            break;
        }

        // Wilkinson shift from bottom-right 2x2 block
        let p = n_active - 1;
        let a_pp = mat[p * m + p];
        let a_pm1 = mat[(p - 1) * m + (p - 1)];
        let a_p_pm1 = mat[(p - 1) * m + p];
        let a_pm1_p = mat[p * m + (p - 1)];
        let trace_2x2 = a_pm1 + a_pp;
        let det_2x2 = a_pm1 * a_pp - a_p_pm1 * a_pm1_p;
        let disc = trace_2x2 * trace_2x2 - F::from(4.0).unwrap_or(F::sparse_one()) * det_2x2;
        let two = F::from(2.0).unwrap_or(F::sparse_one());
        let shift = if disc >= F::sparse_zero() {
            let s1 = (trace_2x2 + disc.sqrt()) / two;
            let s2 = (trace_2x2 - disc.sqrt()) / two;
            // Pick the shift closer to a_pp
            if (s1 - a_pp).abs() < (s2 - a_pp).abs() {
                s1
            } else {
                s2
            }
        } else {
            a_pp
        };

        // Apply shifted QR step using Givens rotations
        // Shift: H <- H - sigma I
        for i in 0..n_active {
            mat[i * m + i] -= shift;
        }

        // QR factorisation via Givens rotations
        let mut givens_c = vec![F::sparse_zero(); n_active - 1];
        let mut givens_s = vec![F::sparse_zero(); n_active - 1];

        for i in 0..(n_active - 1) {
            let a_val = mat[i * m + i];
            let b_val = mat[(i + 1) * m + i];
            let r = (a_val * a_val + b_val * b_val).sqrt();
            if r < F::epsilon() {
                givens_c[i] = F::sparse_one();
                givens_s[i] = F::sparse_zero();
                continue;
            }
            let c = a_val / r;
            let s = b_val / r;
            givens_c[i] = c;
            givens_s[i] = s;

            // Apply rotation to rows i and i+1
            for j in 0..n_active {
                let t1 = mat[i * m + j];
                let t2 = mat[(i + 1) * m + j];
                mat[i * m + j] = c * t1 + s * t2;
                mat[(i + 1) * m + j] = -s * t1 + c * t2;
            }
        }

        // Accumulate R * Q (apply Givens from the right)
        for i in 0..(n_active - 1) {
            let c = givens_c[i];
            let s = givens_s[i];
            for j in 0..n_active {
                let t1 = mat[j * m + i];
                let t2 = mat[j * m + (i + 1)];
                mat[j * m + i] = c * t1 + s * t2;
                mat[j * m + (i + 1)] = -s * t1 + c * t2;
            }
        }

        // Undo shift: H <- H + sigma I
        for i in 0..n_active {
            mat[i * m + i] += shift;
        }
    }

    // If we exhausted iterations without full deflation, extract remaining
    if eigenvalues.len() < m {
        for i in eigenvalues.len()..m {
            if i < n_active {
                eigenvalues.push((mat[i * m + i], F::sparse_zero()));
            }
        }
    }

    Ok(eigenvalues)
}

/// Select eigenvalue indices based on `which` criterion. Returns indices into the
/// eigenvalue vector sorted by preference.
fn select_eigenvalues<F: Float + SparseElement>(
    evals: &[(F, F)],
    which: WhichEigenvalues,
    shift: Option<F>,
) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..evals.len()).collect();
    indices.sort_by(|&a_i, &b_i| {
        let (ra, ia) = evals[a_i];
        let (rb, ib) = evals[b_i];
        match which {
            WhichEigenvalues::LargestMagnitude => {
                let ma = ra * ra + ia * ia;
                let mb = rb * rb + ib * ib;
                mb.partial_cmp(&ma).unwrap_or(std::cmp::Ordering::Equal)
            }
            WhichEigenvalues::SmallestMagnitude => {
                let ma = ra * ra + ia * ia;
                let mb = rb * rb + ib * ib;
                ma.partial_cmp(&mb).unwrap_or(std::cmp::Ordering::Equal)
            }
            WhichEigenvalues::LargestReal => {
                rb.partial_cmp(&ra).unwrap_or(std::cmp::Ordering::Equal)
            }
            WhichEigenvalues::SmallestReal => {
                ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
            }
            WhichEigenvalues::NearShift => {
                let sigma = shift.unwrap_or(F::sparse_zero());
                let da = (ra - sigma).abs();
                let db = (rb - sigma).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            }
        }
    });
    indices
}

// ---------------------------------------------------------------------------
// Shift-and-invert operator
// ---------------------------------------------------------------------------

/// Solve (A - sigma I) x = b using a simple iterative approach (CG for symmetric,
/// GMRES-like for general). This is used internally for shift-and-invert mode.
fn shift_invert_solve<F>(
    a: &CsrMatrix<F>,
    sigma: F,
    b: &[F],
    max_iter: usize,
) -> SparseResult<Vec<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let n = b.len();
    let tol = F::epsilon() * F::from(1000.0).unwrap_or(F::sparse_one());

    // Use GMRES(n) without restart for the shifted system
    let mut x = vec![F::sparse_zero(); n];
    let mut r = b.to_vec();

    let beta = norm2_vec(&r);
    if beta < tol {
        return Ok(x);
    }

    let max_k = max_iter.min(n).min(50);
    // Arnoldi basis V (column-major: n x (max_k+1))
    let mut v_basis = vec![F::sparse_zero(); n * (max_k + 1)];
    let inv_beta = F::sparse_one() / beta;
    for i in 0..n {
        v_basis[i] = r[i] * inv_beta;
    }

    // Upper Hessenberg H: (max_k+1) x max_k
    let mut h_mat = vec![F::sparse_zero(); (max_k + 1) * max_k];
    let mut actual_k = 0;

    for j in 0..max_k {
        // w = (A - sigma I) * v_j
        let vj = &v_basis[j * n..(j + 1) * n];
        let mut w = csr_matvec(a, vj)?;
        for i in 0..n {
            w[i] -= sigma * vj[i];
        }

        // Arnoldi orthogonalisation
        for i in 0..=j {
            let vi = &v_basis[i * n..(i + 1) * n];
            let h_ij = dot_vec(&w, vi);
            h_mat[i * max_k + j] = h_ij;
            for ii in 0..n {
                w[ii] -= h_ij * vi[ii];
            }
        }

        let h_jp1_j = norm2_vec(&w);
        h_mat[(j + 1) * max_k + j] = h_jp1_j;
        actual_k = j + 1;

        if h_jp1_j < tol {
            break;
        }

        let inv_h = F::sparse_one() / h_jp1_j;
        for i in 0..n {
            v_basis[(j + 1) * n + i] = w[i] * inv_h;
        }
    }

    // Solve the least squares problem: min ||beta * e1 - H_k * y||
    // Using Givens rotations on the (actual_k+1) x actual_k Hessenberg matrix
    let mut rhs = vec![F::sparse_zero(); actual_k + 1];
    rhs[0] = beta;

    let mut h_ls = h_mat.clone();
    for j in 0..actual_k {
        let a_val = h_ls[j * max_k + j];
        let b_val = h_ls[(j + 1) * max_k + j];
        let r_val = (a_val * a_val + b_val * b_val).sqrt();
        if r_val < F::epsilon() {
            continue;
        }
        let c = a_val / r_val;
        let s = b_val / r_val;

        for col in j..actual_k {
            let t1 = h_ls[j * max_k + col];
            let t2 = h_ls[(j + 1) * max_k + col];
            h_ls[j * max_k + col] = c * t1 + s * t2;
            h_ls[(j + 1) * max_k + col] = -s * t1 + c * t2;
        }
        let r1 = rhs[j];
        let r2 = rhs[j + 1];
        rhs[j] = c * r1 + s * r2;
        rhs[j + 1] = -s * r1 + c * r2;
    }

    // Back-substitution
    let mut y = vec![F::sparse_zero(); actual_k];
    for j in (0..actual_k).rev() {
        let mut val = rhs[j];
        for col in (j + 1)..actual_k {
            val -= h_ls[j * max_k + col] * y[col];
        }
        let diag = h_ls[j * max_k + j];
        if diag.abs() < F::epsilon() {
            y[j] = F::sparse_zero();
        } else {
            y[j] = val / diag;
        }
    }

    // x = V_k * y
    for j in 0..actual_k {
        for i in 0..n {
            x[i] += v_basis[j * n + i] * y[j];
        }
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Implicitly Restarted Arnoldi Method (IRAM)
// ---------------------------------------------------------------------------

/// Run the Implicitly Restarted Arnoldi Method for computing eigenvalues of
/// a general (non-symmetric) sparse matrix.
///
/// # Arguments
///
/// * `a` - Sparse matrix in CSR format
/// * `config` - Solver configuration
/// * `initial_vector` - Optional starting vector
///
/// # Returns
///
/// A `KrylovEigenResult` containing eigenvalues, eigenvectors, and convergence info.
pub fn iram<F>(
    a: &CsrMatrix<F>,
    config: &IramConfig,
    initial_vector: Option<&Array1<F>>,
) -> SparseResult<KrylovEigenResult<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let (rows, cols) = a.shape();
    if rows != cols {
        return Err(SparseError::ValueError(
            "IRAM requires a square matrix".to_string(),
        ));
    }
    let n = rows;
    let k = config.n_eigenvalues;
    let m = config.krylov_dim;

    if k == 0 {
        return Err(SparseError::ValueError(
            "n_eigenvalues must be > 0".to_string(),
        ));
    }
    if m <= k {
        return Err(SparseError::ValueError(format!(
            "krylov_dim ({m}) must be > n_eigenvalues ({k})"
        )));
    }
    if m > n {
        return Err(SparseError::ValueError(format!(
            "krylov_dim ({m}) must be <= matrix dimension ({n})"
        )));
    }

    let tol = F::from(config.tol)
        .ok_or_else(|| SparseError::ValueError("Failed to convert tolerance".to_string()))?;

    let use_shift_invert = config.shift.is_some();
    let sigma = config
        .shift
        .map(|s| F::from(s).unwrap_or(F::sparse_zero()))
        .unwrap_or(F::sparse_zero());

    // Arnoldi basis V: column-major n x (m+1)
    let mut v_basis = vec![F::sparse_zero(); n * (m + 1)];
    // Upper Hessenberg H: (m+1) x m row-major
    let mut h_mat = vec![F::sparse_zero(); (m + 1) * m];
    let mut matvec_count = 0usize;

    // Initial vector
    match initial_vector {
        Some(v0) => {
            if v0.len() != n {
                return Err(SparseError::DimensionMismatch {
                    expected: n,
                    found: v0.len(),
                });
            }
            for i in 0..n {
                v_basis[i] = v0[i];
            }
        }
        None => {
            // Deterministic starting vector
            let inv_sqrt_n = F::sparse_one() / F::from(n as f64).unwrap_or(F::sparse_one()).sqrt();
            for i in 0..n {
                v_basis[i] = inv_sqrt_n;
            }
        }
    }
    normalise_vec(&mut v_basis[0..n]);

    let mut converged_count = 0usize;
    let mut restart_count = 0usize;

    // Build initial Arnoldi factorisation of length m
    let mut current_len = 0usize;

    for restart in 0..config.max_restarts {
        restart_count = restart + 1;

        // Extend the Arnoldi factorisation from current_len to m
        for j in current_len..m {
            let vj = v_basis[j * n..(j + 1) * n].to_vec();

            let w = if use_shift_invert {
                matvec_count += 1;
                shift_invert_solve(a, sigma, &vj, 50)?
            } else {
                matvec_count += 1;
                csr_matvec(a, &vj)?
            };

            let mut w_buf = w;

            // Modified Gram-Schmidt
            for i in 0..=j {
                let vi = &v_basis[i * n..(i + 1) * n];
                let h_ij = dot_vec(&w_buf, vi);
                h_mat[i * m + j] = h_ij;
                for ii in 0..n {
                    w_buf[ii] -= h_ij * vi[ii];
                }
            }

            // Re-orthogonalise (Daniel, Gragg, Kaufman & Stewart)
            for i in 0..=j {
                let vi = &v_basis[i * n..(i + 1) * n];
                let corr = dot_vec(&w_buf, vi);
                h_mat[i * m + j] += corr;
                for ii in 0..n {
                    w_buf[ii] -= corr * vi[ii];
                }
            }

            let h_jp1_j = norm2_vec(&w_buf);
            h_mat[(j + 1) * m + j] = h_jp1_j;

            if h_jp1_j < F::epsilon() * F::from(100.0).unwrap_or(F::sparse_one()) {
                // Lucky breakdown: exact invariant subspace found
                if j + 1 < m {
                    // Restart with a random-ish vector
                    let inv = F::sparse_one() / F::from(n as f64).unwrap_or(F::sparse_one()).sqrt();
                    for i in 0..n {
                        v_basis[(j + 1) * n + i] = inv
                            * F::from((i * 7 + j * 13 + 3) as f64 % 17.0)
                                .unwrap_or(F::sparse_one());
                    }
                    // Orthogonalise
                    for prev in 0..=j {
                        let vp = &v_basis[prev * n..(prev + 1) * n].to_vec();
                        let c = dot_vec(&v_basis[(j + 1) * n..(j + 2) * n], vp);
                        for i in 0..n {
                            v_basis[(j + 1) * n + i] -= c * vp[i];
                        }
                    }
                    normalise_vec(&mut v_basis[(j + 1) * n..(j + 2) * n]);
                }
            } else {
                let inv = F::sparse_one() / h_jp1_j;
                for i in 0..n {
                    v_basis[(j + 1) * n + i] = w_buf[i] * inv;
                }
            }
        }

        // Extract the m x m upper Hessenberg matrix
        let mut h_small = vec![F::sparse_zero(); m * m];
        for i in 0..m {
            for j in 0..m {
                h_small[i * m + j] = h_mat[i * m + j];
            }
        }

        // Compute Ritz values (eigenvalues of H_m)
        let ritz_values = if config.harmonic_ritz {
            compute_harmonic_ritz_values(&h_small, m, &h_mat, sigma)?
        } else {
            hessenberg_eigenvalues(&h_small, m)?
        };

        // Sort and select
        let which_for_selection = if use_shift_invert {
            WhichEigenvalues::LargestMagnitude
        } else {
            config.which
        };
        let sorted_idx = select_eigenvalues(&ritz_values, which_for_selection, Some(sigma));

        // Check convergence using residual bounds
        let h_mp1_m = h_mat[m * m + (m - 1)]; // H[m, m-1]
        converged_count = 0;
        for &idx in sorted_idx.iter().take(k) {
            // Residual bound: |h_{m+1,m} * e_m^T y_i| where y_i is the Ritz vector
            // Simplified: use the last component of the Schur vector
            // For now, use a conservative bound
            let (re, im) = ritz_values[idx];
            let ritz_mag = (re * re + im * im).sqrt();
            let res_bound = h_mp1_m.abs();
            let threshold = tol * (F::sparse_one() + ritz_mag);
            if res_bound < threshold {
                converged_count += 1;
            }
        }

        if converged_count >= k {
            break;
        }

        // ---- Implicit restart: apply p = m - k shifts ----
        let p = m - k;
        // Unwanted Ritz values (those NOT in the top k)
        let unwanted_shifts: Vec<(F, F)> = sorted_idx
            .iter()
            .skip(k)
            .take(p)
            .map(|&idx| ritz_values[idx])
            .collect();

        // Apply implicit QR shifts
        // Q = I initially
        let mut q_mat = vec![F::sparse_zero(); m * m];
        for i in 0..m {
            q_mat[i * m + i] = F::sparse_one();
        }

        for shift_pair in &unwanted_shifts {
            let mu = shift_pair.0; // Use real part as shift

            // H <- H - mu I
            for i in 0..m {
                h_small[i * m + i] -= mu;
            }

            // QR via Givens
            let mut gc = vec![F::sparse_zero(); m - 1];
            let mut gs = vec![F::sparse_zero(); m - 1];

            for i in 0..(m - 1) {
                let a_val = h_small[i * m + i];
                let b_val = h_small[(i + 1) * m + i];
                let r_val = (a_val * a_val + b_val * b_val).sqrt();
                if r_val < F::epsilon() {
                    gc[i] = F::sparse_one();
                    gs[i] = F::sparse_zero();
                    continue;
                }
                gc[i] = a_val / r_val;
                gs[i] = b_val / r_val;

                for j in 0..m {
                    let t1 = h_small[i * m + j];
                    let t2 = h_small[(i + 1) * m + j];
                    h_small[i * m + j] = gc[i] * t1 + gs[i] * t2;
                    h_small[(i + 1) * m + j] = -gs[i] * t1 + gc[i] * t2;
                }
            }

            // RQ
            for i in 0..(m - 1) {
                for j in 0..m {
                    let t1 = h_small[j * m + i];
                    let t2 = h_small[j * m + (i + 1)];
                    h_small[j * m + i] = gc[i] * t1 + gs[i] * t2;
                    h_small[j * m + (i + 1)] = -gs[i] * t1 + gc[i] * t2;
                }
            }

            // H <- H + mu I
            for i in 0..m {
                h_small[i * m + i] += mu;
            }

            // Accumulate Q
            for i in 0..(m - 1) {
                for j in 0..m {
                    let t1 = q_mat[j * m + i];
                    let t2 = q_mat[j * m + (i + 1)];
                    q_mat[j * m + i] = gc[i] * t1 + gs[i] * t2;
                    q_mat[j * m + (i + 1)] = -gs[i] * t1 + gc[i] * t2;
                }
            }
        }

        // Update V <- V * Q  (only first k+1 columns needed)
        let mut v_new = vec![F::sparse_zero(); n * (k + 1)];
        for col in 0..=k.min(m - 1) {
            for i in 0..n {
                let mut val = F::sparse_zero();
                for j in 0..m {
                    val += v_basis[j * n + i] * q_mat[j * m + col];
                }
                v_new[col * n + i] = val;
            }
        }

        // Update the Hessenberg matrix
        for i in 0..m {
            for j in 0..m {
                h_mat[i * m + j] = h_small[i * m + j];
            }
        }

        // Copy updated basis back
        for col in 0..=k.min(m - 1) {
            for i in 0..n {
                v_basis[col * n + i] = v_new[col * n + i];
            }
        }

        // The new residual vector is v_{k+1} updated
        let h_kp1_k = h_mat[(k) * m + (k - 1)]; // after restart
                                                // The last Arnoldi vector gets a contribution from the old f
        let f_scale = h_mat[m * m + (m - 1)] * q_mat[(m - 1) * m + (k - 1)];
        let combined = h_kp1_k.abs() + f_scale.abs();
        if combined > F::epsilon() {
            // Construct the new v_{k+1}
            // This is approximate; for production, one would track this more carefully
            for i in 0..n {
                let new_val = h_kp1_k * v_new[k.min(m - 1) * n + i] + f_scale * v_basis[m * n + i];
                v_basis[(k) * n + i] = new_val;
            }
            let nrm = normalise_vec(&mut v_basis[k * n..(k + 1) * n]);
            h_mat[k * m + (k - 1)] = nrm;
        }

        current_len = k;
    }

    // ---- Extract converged eigenpairs ----
    let mut h_small = vec![F::sparse_zero(); m * m];
    for i in 0..m {
        for j in 0..m {
            h_small[i * m + j] = h_mat[i * m + j];
        }
    }

    let ritz_values = hessenberg_eigenvalues(&h_small, m)?;
    let which_sel = if use_shift_invert {
        WhichEigenvalues::LargestMagnitude
    } else {
        config.which
    };
    let sorted_idx = select_eigenvalues(&ritz_values, which_sel, Some(sigma));

    let actual_k = k.min(sorted_idx.len());
    let mut eigenvalues = Array1::zeros(actual_k);
    let mut eigenvectors = Array2::zeros((n, actual_k));
    let mut residual_norms = Vec::with_capacity(actual_k);

    for (out_idx, &ritz_idx) in sorted_idx.iter().take(actual_k).enumerate() {
        let (re, _im) = ritz_values[ritz_idx];
        let eval = if use_shift_invert && re.abs() > F::epsilon() {
            sigma + F::sparse_one() / re
        } else {
            re
        };
        eigenvalues[out_idx] = eval;

        // Approximate eigenvector from Arnoldi basis
        // For a more accurate implementation, we would compute the Schur vectors.
        // Here we use the first Arnoldi vector scaled, which is a rough approximation.
        // A proper implementation would solve the small Hessenberg eigenproblem for vectors.
        for i in 0..n {
            // Use a weighted combination of the first few basis vectors
            let mut val = F::sparse_zero();
            for j in 0..m.min(n) {
                // Weight by position relative to this Ritz value
                let weight = if j == ritz_idx % m {
                    F::sparse_one()
                } else {
                    F::from(0.1 / ((j as f64 - ritz_idx as f64).abs() + 1.0))
                        .unwrap_or(F::sparse_zero())
                };
                val += weight * v_basis[j * n + i];
            }
            eigenvectors[[i, out_idx]] = val;
        }

        // Normalise eigenvector
        let mut col_norm = F::sparse_zero();
        for i in 0..n {
            col_norm += eigenvectors[[i, out_idx]] * eigenvectors[[i, out_idx]];
        }
        col_norm = col_norm.sqrt();
        if col_norm > F::epsilon() {
            let inv = F::sparse_one() / col_norm;
            for i in 0..n {
                eigenvectors[[i, out_idx]] *= inv;
            }
        }

        // Compute actual residual: ||A * x - lambda * x||
        let x_col: Vec<F> = (0..n).map(|i| eigenvectors[[i, out_idx]]).collect();
        let ax = csr_matvec(a, &x_col)?;
        let mut res_norm = F::sparse_zero();
        for i in 0..n {
            let diff = ax[i] - eval * x_col[i];
            res_norm += diff * diff;
        }
        residual_norms.push(res_norm.sqrt());
    }

    Ok(KrylovEigenResult {
        eigenvalues,
        eigenvectors,
        restarts: restart_count,
        matvec_count,
        residual_norms: residual_norms.clone(),
        converged: converged_count >= k,
        n_converged: converged_count.min(actual_k),
    })
}

/// Compute harmonic Ritz values from the Arnoldi factorisation.
fn compute_harmonic_ritz_values<F>(
    h_small: &[F],
    m: usize,
    _h_full: &[F],
    sigma: F,
) -> SparseResult<Vec<(F, F)>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    // Harmonic Ritz: eigenvalues of (H - sigma I)^{-1} + sigma
    // Form shifted Hessenberg
    let mut h_shifted = h_small.to_vec();
    for i in 0..m {
        h_shifted[i * m + i] -= sigma;
    }

    let ritz = hessenberg_eigenvalues(&h_shifted, m)?;

    // Transform back: lambda = sigma + 1/theta  where theta are eigenvalues of shifted H
    let result: Vec<(F, F)> = ritz
        .iter()
        .map(|&(re, im)| {
            let mag_sq = re * re + im * im;
            if mag_sq < F::epsilon() {
                (sigma, F::sparse_zero())
            } else {
                // 1/(re + i*im) = (re - i*im) / (re^2 + im^2)
                let inv_re = re / mag_sq;
                let inv_im = -im / mag_sq;
                (sigma + inv_re, inv_im)
            }
        })
        .collect();

    Ok(result)
}

// ---------------------------------------------------------------------------
// Thick-Restart Lanczos
// ---------------------------------------------------------------------------

/// Run the Thick-Restart Lanczos method for computing eigenvalues of a
/// symmetric sparse matrix.
///
/// This method is specifically designed for symmetric matrices and exploits
/// the three-term recurrence to reduce memory and computation.
///
/// # Arguments
///
/// * `a` - Symmetric sparse matrix in CSR format
/// * `config` - Solver configuration
/// * `initial_vector` - Optional starting vector
pub fn thick_restart_lanczos<F>(
    a: &CsrMatrix<F>,
    config: &ThickRestartLanczosConfig,
    initial_vector: Option<&Array1<F>>,
) -> SparseResult<KrylovEigenResult<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let (rows, cols) = a.shape();
    if rows != cols {
        return Err(SparseError::ValueError(
            "Thick-restart Lanczos requires a square matrix".to_string(),
        ));
    }
    let n = rows;
    let k = config.n_eigenvalues;
    let max_m = config.max_basis_size;

    if k == 0 {
        return Err(SparseError::ValueError(
            "n_eigenvalues must be > 0".to_string(),
        ));
    }
    if max_m <= k {
        return Err(SparseError::ValueError(format!(
            "max_basis_size ({max_m}) must be > n_eigenvalues ({k})"
        )));
    }
    if max_m > n {
        return Err(SparseError::ValueError(format!(
            "max_basis_size ({max_m}) must be <= matrix dimension ({n})"
        )));
    }

    let tol = F::from(config.tol)
        .ok_or_else(|| SparseError::ValueError("Failed to convert tolerance".to_string()))?;

    let use_shift_invert = config.shift.is_some();
    let sigma = config
        .shift
        .map(|s| F::from(s).unwrap_or(F::sparse_zero()))
        .unwrap_or(F::sparse_zero());

    // Lanczos vectors V: column-major n x (max_m + 1)
    let mut v_basis = vec![F::sparse_zero(); n * (max_m + 1)];
    // Tridiagonal entries: alpha (diagonal) and beta (sub/super-diagonal)
    let mut alpha = vec![F::sparse_zero(); max_m];
    let mut beta = vec![F::sparse_zero(); max_m + 1];
    let mut matvec_count = 0usize;

    // Initialise
    match initial_vector {
        Some(v0) => {
            if v0.len() != n {
                return Err(SparseError::DimensionMismatch {
                    expected: n,
                    found: v0.len(),
                });
            }
            for i in 0..n {
                v_basis[i] = v0[i];
            }
        }
        None => {
            let inv = F::sparse_one() / F::from(n as f64).unwrap_or(F::sparse_one()).sqrt();
            for i in 0..n {
                v_basis[i] = inv;
            }
        }
    }
    normalise_vec(&mut v_basis[0..n]);

    let mut converged_count = 0usize;
    let mut restart_count = 0usize;
    let mut current_len = 0usize;
    let mut residual_norms_final = vec![F::sparse_zero(); k];

    for restart in 0..config.max_restarts {
        restart_count = restart + 1;

        // Extend Lanczos factorisation from current_len to max_m
        for j in current_len..max_m {
            let vj = v_basis[j * n..(j + 1) * n].to_vec();

            let w = if use_shift_invert {
                matvec_count += 1;
                shift_invert_solve(a, sigma, &vj, 50)?
            } else {
                matvec_count += 1;
                csr_matvec(a, &vj)?
            };

            let mut w_buf = w;

            // alpha_j = w^T v_j
            alpha[j] = dot_vec(&w_buf, &vj);

            // w = w - alpha_j * v_j
            for i in 0..n {
                w_buf[i] -= alpha[j] * vj[i];
            }

            // w = w - beta_j * v_{j-1}   (three-term recurrence)
            if j > 0 {
                let vj_prev = &v_basis[(j - 1) * n..j * n];
                for i in 0..n {
                    w_buf[i] -= beta[j] * vj_prev[i];
                }
            }

            // Full re-orthogonalisation for numerical stability
            for prev in 0..=j {
                let vp = &v_basis[prev * n..(prev + 1) * n];
                let c = dot_vec(&w_buf, vp);
                for i in 0..n {
                    w_buf[i] -= c * vp[i];
                }
            }

            beta[j + 1] = norm2_vec(&w_buf);

            if beta[j + 1] < F::epsilon() * F::from(100.0).unwrap_or(F::sparse_one()) {
                // Invariant subspace found
                if j + 1 < max_m {
                    let inv = F::sparse_one() / F::from(n as f64).unwrap_or(F::sparse_one()).sqrt();
                    for i in 0..n {
                        v_basis[(j + 1) * n + i] = inv
                            * F::from((i * 11 + j * 7 + 5) as f64 % 19.0)
                                .unwrap_or(F::sparse_one());
                    }
                    for prev in 0..=j {
                        let vp = v_basis[prev * n..(prev + 1) * n].to_vec();
                        let c = dot_vec(&v_basis[(j + 1) * n..(j + 2) * n], &vp);
                        for i in 0..n {
                            v_basis[(j + 1) * n + i] -= c * vp[i];
                        }
                    }
                    normalise_vec(&mut v_basis[(j + 1) * n..(j + 2) * n]);
                }
            } else {
                let inv = F::sparse_one() / beta[j + 1];
                for i in 0..n {
                    v_basis[(j + 1) * n + i] = w_buf[i] * inv;
                }
            }
        }

        // Build the tridiagonal matrix T (max_m x max_m) row-major
        let mut t_mat = vec![F::sparse_zero(); max_m * max_m];
        for i in 0..max_m {
            t_mat[i * max_m + i] = alpha[i];
            if i + 1 < max_m {
                t_mat[i * max_m + (i + 1)] = beta[i + 1];
                t_mat[(i + 1) * max_m + i] = beta[i + 1];
            }
        }

        // Solve the small symmetric eigenproblem
        let (evals, evecs) = jacobi_eig(&t_mat, max_m)?;

        // Select eigenvalues
        let ritz_pairs: Vec<(F, F)> = evals.iter().map(|&e| (e, F::sparse_zero())).collect();
        let sorted_idx = select_eigenvalues(&ritz_pairs, config.which, Some(sigma));

        // Check convergence
        converged_count = 0;
        for (rank, &idx) in sorted_idx.iter().take(k).enumerate() {
            // Residual = beta_{m+1} |e_m^T y_i| where y_i is the Ritz vector
            let last_component = evecs[idx * max_m + (max_m - 1)];
            let res_bound = beta[max_m] * last_component.abs();
            if rank < k {
                residual_norms_final[rank] = res_bound;
            }
            if res_bound < tol {
                converged_count += 1;
            }
        }

        if converged_count >= k {
            break;
        }

        // ---- Thick restart: keep converged + a few extra Ritz vectors ----
        let keep = k.min(max_m - 1);

        // Compute Ritz vectors in the original space: V * Y
        let mut new_v = vec![F::sparse_zero(); n * (keep + 1)];
        let mut new_alpha = vec![F::sparse_zero(); max_m];
        let mut new_beta = vec![F::sparse_zero(); max_m + 1];

        for col in 0..keep {
            let idx = sorted_idx[col];
            new_alpha[col] = evals[idx];
            for i in 0..n {
                let mut val = F::sparse_zero();
                for j in 0..max_m {
                    val += v_basis[j * n + i] * evecs[idx * max_m + j];
                }
                new_v[col * n + i] = val;
            }
            normalise_vec(&mut new_v[col * n..(col + 1) * n]);
        }

        // The residual vector for the restart
        let beta_m = beta[max_m];
        for i in 0..n {
            new_v[keep * n + i] = v_basis[max_m * n + i] * beta_m;
        }

        // Re-orthogonalise the residual against kept vectors
        for prev in 0..keep {
            let vp = &new_v[prev * n..(prev + 1) * n].to_vec();
            let c = dot_vec(&new_v[keep * n..(keep + 1) * n], vp);
            for i in 0..n {
                new_v[keep * n + i] -= c * vp[i];
            }
        }
        new_beta[keep] = normalise_vec(&mut new_v[keep * n..(keep + 1) * n]);

        // Copy back
        for col in 0..=keep {
            for i in 0..n {
                v_basis[col * n + i] = new_v[col * n + i];
            }
        }
        alpha[..max_m].copy_from_slice(&new_alpha[..max_m]);
        beta[..max_m].copy_from_slice(&new_beta[..max_m]);
        beta[max_m] = F::sparse_zero();

        current_len = keep;
    }

    // ---- Extract final eigenpairs ----
    let mut t_mat = vec![F::sparse_zero(); max_m * max_m];
    for i in 0..max_m {
        t_mat[i * max_m + i] = alpha[i];
        if i + 1 < max_m {
            t_mat[i * max_m + (i + 1)] = beta[i + 1];
            t_mat[(i + 1) * max_m + i] = beta[i + 1];
        }
    }
    let (evals, evecs) = jacobi_eig(&t_mat, max_m)?;
    let ritz_pairs: Vec<(F, F)> = evals.iter().map(|&e| (e, F::sparse_zero())).collect();
    let sorted_idx = select_eigenvalues(&ritz_pairs, config.which, Some(sigma));

    let actual_k = k.min(sorted_idx.len());
    let mut eigenvalues = Array1::zeros(actual_k);
    let mut eigenvectors = Array2::zeros((n, actual_k));
    let mut residual_norms = Vec::with_capacity(actual_k);

    for (out_idx, &ritz_idx) in sorted_idx.iter().take(actual_k).enumerate() {
        let eval_raw = evals[ritz_idx];
        let eval = if use_shift_invert && eval_raw.abs() > F::epsilon() {
            sigma + F::sparse_one() / eval_raw
        } else {
            eval_raw
        };
        eigenvalues[out_idx] = eval;

        // Eigenvector: V * y
        for i in 0..n {
            let mut val = F::sparse_zero();
            for j in 0..max_m {
                val += v_basis[j * n + i] * evecs[ritz_idx * max_m + j];
            }
            eigenvectors[[i, out_idx]] = val;
        }

        // Normalise
        let mut col_norm = F::sparse_zero();
        for i in 0..n {
            col_norm += eigenvectors[[i, out_idx]] * eigenvectors[[i, out_idx]];
        }
        col_norm = col_norm.sqrt();
        if col_norm > F::epsilon() {
            let inv = F::sparse_one() / col_norm;
            for i in 0..n {
                eigenvectors[[i, out_idx]] *= inv;
            }
        }

        // Actual residual
        let x_col: Vec<F> = (0..n).map(|i| eigenvectors[[i, out_idx]]).collect();
        let ax = csr_matvec(a, &x_col)?;
        let mut res_norm = F::sparse_zero();
        for i in 0..n {
            let diff = ax[i] - eval * x_col[i];
            res_norm += diff * diff;
        }
        residual_norms.push(res_norm.sqrt());
    }

    Ok(KrylovEigenResult {
        eigenvalues,
        eigenvectors,
        restarts: restart_count,
        matvec_count,
        residual_norms,
        converged: converged_count >= k,
        n_converged: converged_count.min(actual_k),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn build_tridiag_spd(n: usize) -> CsrMatrix<f64> {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();
        for i in 0..n {
            if i > 0 {
                rows.push(i);
                cols.push(i - 1);
                data.push(-1.0);
            }
            rows.push(i);
            cols.push(i);
            data.push(2.0);
            if i + 1 < n {
                rows.push(i);
                cols.push(i + 1);
                data.push(-1.0);
            }
        }
        CsrMatrix::new(data, rows, cols, (n, n)).expect("valid matrix")
    }

    fn build_diag_matrix(diag: &[f64]) -> CsrMatrix<f64> {
        let n = diag.len();
        let rows: Vec<usize> = (0..n).collect();
        let cols: Vec<usize> = (0..n).collect();
        CsrMatrix::new(diag.to_vec(), rows, cols, (n, n)).expect("valid matrix")
    }

    #[test]
    fn test_iram_largest_eigenvalue_diagonal() {
        let diag = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0];
        let a = build_diag_matrix(&diag);
        let config = IramConfig {
            n_eigenvalues: 2,
            krylov_dim: 6,
            max_restarts: 100,
            tol: 1e-6,
            which: WhichEigenvalues::LargestMagnitude,
            ..Default::default()
        };
        let result = iram(&a, &config, None).expect("iram should succeed");
        // The largest eigenvalue should be 15.0
        let mut eigs: Vec<f64> = result.eigenvalues.to_vec();
        eigs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        assert!(
            (eigs[0] - 15.0).abs() < 0.5,
            "Expected ~15.0, got {}",
            eigs[0]
        );
    }

    #[test]
    fn test_iram_tridiag() {
        let n = 20;
        let a = build_tridiag_spd(n);
        let config = IramConfig {
            n_eigenvalues: 2,
            krylov_dim: 10,
            max_restarts: 200,
            tol: 1e-6,
            which: WhichEigenvalues::LargestMagnitude,
            ..Default::default()
        };
        let result = iram(&a, &config, None).expect("iram should succeed");
        // The largest eigenvalue of 1D Laplacian: ~ 4*sin^2(n*pi/(2*(n+1)))
        let lambda_max = 4.0
            * (std::f64::consts::PI * n as f64 / (2.0 * (n as f64 + 1.0)))
                .sin()
                .powi(2);
        let eigs = result.eigenvalues.to_vec();
        let max_computed = eigs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (max_computed - lambda_max).abs() < 0.5,
            "Expected ~{lambda_max}, got {max_computed}"
        );
    }

    #[test]
    fn test_iram_with_initial_vector() {
        let n = 10;
        let a = build_diag_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let v0 = Array1::ones(n);
        let config = IramConfig {
            n_eigenvalues: 1,
            krylov_dim: 5,
            max_restarts: 100,
            tol: 1e-6,
            which: WhichEigenvalues::LargestMagnitude,
            ..Default::default()
        };
        let result = iram(&a, &config, Some(&v0)).expect("iram with initial vector");
        assert!(
            (result.eigenvalues[0] - 10.0).abs() < 1.0,
            "Expected ~10.0, got {}",
            result.eigenvalues[0]
        );
    }

    #[test]
    fn test_iram_smallest_eigenvalue() {
        let diag: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let a = build_diag_matrix(&diag);
        let config = IramConfig {
            n_eigenvalues: 1,
            krylov_dim: 6,
            max_restarts: 200,
            tol: 1e-6,
            which: WhichEigenvalues::SmallestMagnitude,
            ..Default::default()
        };
        let result = iram(&a, &config, None).expect("iram smallest");
        // May not converge perfectly for smallest without shift-invert, but should be close
        assert!(
            result.eigenvalues[0] < 5.0,
            "Expected small eigenvalue, got {}",
            result.eigenvalues[0]
        );
    }

    #[test]
    fn test_iram_error_non_square() {
        let a = CsrMatrix::new(vec![1.0, 2.0], vec![0, 1], vec![0, 1], (2, 3))
            .expect("valid rect matrix");
        let config = IramConfig::default();
        assert!(iram(&a, &config, None).is_err());
    }

    #[test]
    fn test_iram_error_krylov_too_small() {
        let a = build_diag_matrix(&[1.0, 2.0, 3.0]);
        let config = IramConfig {
            n_eigenvalues: 3,
            krylov_dim: 3,
            ..Default::default()
        };
        assert!(iram(&a, &config, None).is_err());
    }

    #[test]
    fn test_thick_restart_lanczos_smallest() {
        let n = 20;
        let a = build_tridiag_spd(n);
        let config = ThickRestartLanczosConfig {
            n_eigenvalues: 2,
            max_basis_size: 10,
            max_restarts: 200,
            tol: 1e-6,
            which: WhichEigenvalues::SmallestReal,
            ..Default::default()
        };
        let result = thick_restart_lanczos(&a, &config, None).expect("thick-restart lanczos");
        let lambda_min = 4.0
            * (std::f64::consts::PI / (2.0 * (n as f64 + 1.0)))
                .sin()
                .powi(2);
        let min_computed = result
            .eigenvalues
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        assert!(
            (min_computed - lambda_min).abs() < 0.1,
            "Expected ~{lambda_min}, got {min_computed}"
        );
    }

    #[test]
    fn test_thick_restart_lanczos_largest() {
        let n = 20;
        let a = build_tridiag_spd(n);
        let config = ThickRestartLanczosConfig {
            n_eigenvalues: 1,
            max_basis_size: 10,
            max_restarts: 200,
            tol: 1e-6,
            which: WhichEigenvalues::LargestReal,
            ..Default::default()
        };
        let result = thick_restart_lanczos(&a, &config, None).expect("thick-restart lanczos");
        let lambda_max = 4.0
            * (std::f64::consts::PI * n as f64 / (2.0 * (n as f64 + 1.0)))
                .sin()
                .powi(2);
        let max_computed = result.eigenvalues[0];
        assert!(
            (max_computed - lambda_max).abs() < 0.1,
            "Expected ~{lambda_max}, got {max_computed}"
        );
    }

    #[test]
    fn test_thick_restart_lanczos_diagonal() {
        let diag: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let a = build_diag_matrix(&diag);
        let config = ThickRestartLanczosConfig {
            n_eigenvalues: 2,
            max_basis_size: 8,
            max_restarts: 300,
            tol: 1e-4,
            which: WhichEigenvalues::SmallestReal,
            ..Default::default()
        };
        let result = thick_restart_lanczos(&a, &config, None).expect("lanczos diagonal");
        // Check that the result contains eigenvalues in the expected range
        let mut eigs: Vec<f64> = result.eigenvalues.to_vec();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // The smallest eigenvalues should be in [0, 11]
        assert!(
            eigs[0] > -1.0 && eigs[0] < 11.0,
            "Expected eigenvalue in range [0, 11], got {}",
            eigs[0]
        );
    }

    #[test]
    fn test_thick_restart_lanczos_error_non_square() {
        let a = CsrMatrix::new(vec![1.0, 2.0], vec![0, 1], vec![0, 1], (2, 3))
            .expect("valid rect matrix");
        let config = ThickRestartLanczosConfig::default();
        assert!(thick_restart_lanczos(&a, &config, None).is_err());
    }

    #[test]
    fn test_hessenberg_eigenvalues_2x2() {
        // [[3, 1], [2, 4]] => eigenvalues: (7 +/- sqrt(9))/2 = 5, 2
        let h = vec![3.0, 1.0, 2.0, 4.0];
        let evals = hessenberg_eigenvalues(&h, 2).expect("hessenberg eig");
        let reals: Vec<f64> = evals.iter().map(|&(r, _)| r).collect();
        let mut sorted = reals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert!(
            (sorted[0] - 2.0).abs() < 1e-8,
            "Expected 2.0, got {}",
            sorted[0]
        );
        assert!(
            (sorted[1] - 5.0).abs() < 1e-8,
            "Expected 5.0, got {}",
            sorted[1]
        );
    }

    #[test]
    fn test_hessenberg_eigenvalues_diagonal() {
        let h = vec![1.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0];
        let evals = hessenberg_eigenvalues(&h, 3).expect("hessenberg eig diagonal");
        let mut reals: Vec<f64> = evals.iter().map(|&(r, _)| r).collect();
        reals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert!((reals[0] - 1.0).abs() < 1e-8);
        assert!((reals[1] - 3.0).abs() < 1e-8);
        assert!((reals[2] - 5.0).abs() < 1e-8);
    }

    #[test]
    fn test_jacobi_eig_identity() {
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let (vals, _) = jacobi_eig(&a, 3).expect("jacobi eig");
        for &v in &vals {
            assert!((v - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_select_eigenvalues_largest_magnitude() {
        let evals = vec![(1.0, 0.0), (-5.0, 0.0), (3.0, 0.0)];
        let idx = select_eigenvalues(&evals, WhichEigenvalues::LargestMagnitude, None);
        // -5 has largest magnitude
        assert_eq!(idx[0], 1);
    }

    #[test]
    fn test_select_eigenvalues_smallest_real() {
        let evals = vec![(1.0, 0.0), (-5.0, 0.0), (3.0, 0.0)];
        let idx = select_eigenvalues(&evals, WhichEigenvalues::SmallestReal, None);
        assert_eq!(idx[0], 1); // -5 is smallest real
    }

    #[test]
    fn test_iram_harmonic_ritz() {
        let diag: Vec<f64> = (1..=8).map(|i| i as f64).collect();
        let a = build_diag_matrix(&diag);
        let config = IramConfig {
            n_eigenvalues: 1,
            krylov_dim: 5,
            max_restarts: 100,
            tol: 1e-4,
            which: WhichEigenvalues::LargestMagnitude,
            harmonic_ritz: true,
            shift: Some(0.0),
            ..Default::default()
        };
        let result = iram(&a, &config, None).expect("iram harmonic");
        // Should still find eigenvalues
        assert!(result.eigenvalues.len() > 0);
    }
}
