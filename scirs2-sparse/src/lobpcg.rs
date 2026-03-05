//! LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient) eigensolver
//!
//! Solves the generalized eigenvalue problem `A x = lambda B x` for the smallest
//! or largest eigenvalues of large sparse matrices. The method is matrix-free
//! (only requires matrix-vector products) and supports preconditioning and
//! locking of converged eigenvectors.
//!
//! # Algorithm
//!
//! LOBPCG iterates on a block of vectors simultaneously, applying the Rayleigh-Ritz
//! procedure to the trial subspace `span(X, W, P)` where:
//! - `X` — current eigenvector approximations
//! - `W` — preconditioned residuals
//! - `P` — conjugate direction from the previous iteration
//!
//! # References
//!
//! - Knyazev, A.V. (2001). "Toward the optimal preconditioned eigensolver: LOBPCG".
//!   SIAM J. Sci. Comput. 23(2), 517-541.

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use crate::iterative_solvers::Preconditioner;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, NumAssign, SparseElement};
use std::fmt::Debug;
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Whether to compute smallest or largest eigenvalues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EigenTarget {
    /// Smallest algebraic eigenvalues (default for LOBPCG).
    #[default]
    Smallest,
    /// Largest algebraic eigenvalues.
    Largest,
}

/// Configuration for the LOBPCG eigensolver.
#[derive(Debug, Clone)]
pub struct LobpcgConfig {
    /// Number of eigenvalues to compute (block size).
    pub block_size: usize,
    /// Maximum number of outer iterations.
    pub max_iter: usize,
    /// Convergence tolerance for residual norms.
    pub tol: f64,
    /// Whether to compute smallest or largest eigenvalues.
    pub target: EigenTarget,
    /// Whether to lock converged eigenvectors early.
    pub locking: bool,
    /// Whether to print convergence information.
    pub verbose: bool,
}

impl Default for LobpcgConfig {
    fn default() -> Self {
        Self {
            block_size: 1,
            max_iter: 500,
            tol: 1e-8,
            target: EigenTarget::Smallest,
            locking: true,
            verbose: false,
        }
    }
}

/// Result of a LOBPCG computation.
#[derive(Debug, Clone)]
pub struct LobpcgResult<F> {
    /// Converged eigenvalues, sorted by magnitude.
    pub eigenvalues: Array1<F>,
    /// Corresponding eigenvectors stored column-wise.
    pub eigenvectors: Array2<F>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final residual norms for each eigenpair.
    pub residual_norms: Vec<F>,
    /// Whether the solver converged within tolerance.
    pub converged: bool,
    /// Number of converged eigenpairs.
    pub n_converged: usize,
}

// ---------------------------------------------------------------------------
// Dense linear algebra helpers (Pure Rust, no LAPACK)
// ---------------------------------------------------------------------------

/// Compute y = A * x  for CSR matrix A and dense vector x.
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

/// Inner product of two slices.
#[inline]
fn dot<F: Float + Sum>(a: &[F], b: &[F]) -> F {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

/// 2-norm of a slice.
#[inline]
fn norm2<F: Float + Sum>(v: &[F]) -> F {
    dot(v, v).sqrt()
}

/// Normalise a vector in-place; returns its original norm.
fn normalise<F: Float + Sum + SparseElement>(v: &mut [F]) -> F {
    let nrm = norm2(v);
    if nrm > F::epsilon() {
        let inv = F::sparse_one() / nrm;
        for vi in v.iter_mut() {
            *vi = *vi * inv;
        }
    }
    nrm
}

/// Classical Gram-Schmidt orthogonalisation of column `col` of `mat` against
/// all previous columns. `mat` is stored column-major: mat[col * n .. (col+1)*n].
fn gram_schmidt_column<F: Float + Sum + SparseElement>(mat: &mut [F], n: usize, col: usize) {
    for j in 0..col {
        let c = dot(&mat[j * n..(j + 1) * n], &mat[col * n..(col + 1) * n]);
        for i in 0..n {
            mat[col * n + i] = mat[col * n + i] - c * mat[j * n + i];
        }
    }
    normalise(&mut mat[col * n..(col + 1) * n]);
}

/// Multiply a CSR matrix A by a dense column-major block V (n x k),
/// producing AV (m x k) column-major.
fn csr_matmul_block<F>(a: &CsrMatrix<F>, v: &[F], n: usize, k: usize) -> SparseResult<Vec<F>>
where
    F: Float + NumAssign + Sum + SparseElement + 'static,
{
    let m = a.rows();
    let mut av = vec![F::sparse_zero(); m * k];
    for col in 0..k {
        let col_vec = &v[col * n..(col + 1) * n];
        let result = csr_matvec(a, col_vec)?;
        for i in 0..m {
            av[col * m + i] = result[i];
        }
    }
    Ok(av)
}

/// Compute the Gram matrix G = V^T W where V is (n x p) and W is (n x q),
/// both in column-major layout. Result is (p x q) in row-major.
fn gram_matrix<F: Float + Sum + SparseElement>(
    v: &[F],
    w: &[F],
    n: usize,
    p: usize,
    q: usize,
) -> Vec<F> {
    let mut g = vec![F::sparse_zero(); p * q];
    for i in 0..p {
        for j in 0..q {
            g[i * q + j] = dot(&v[i * n..(i + 1) * n], &w[j * n..(j + 1) * n]);
        }
    }
    g
}

/// Solve the small dense symmetric eigenvalue problem using the Jacobi method.
/// `a` is (k x k) row-major. Returns eigenvalues and eigenvectors (column-major k x k).
fn dense_symmetric_eig<F>(a: &[F], k: usize) -> SparseResult<(Vec<F>, Vec<F>)>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let max_sweeps = 100;
    let tol = F::epsilon() * F::from(100.0).unwrap_or(F::sparse_one());

    // Work on a copy
    let mut mat = a.to_vec();
    // Eigenvectors as identity
    let mut vecs = vec![F::sparse_zero(); k * k];
    for i in 0..k {
        vecs[i * k + i] = F::sparse_one();
    }

    for _sweep in 0..max_sweeps {
        // Find max off-diagonal
        let mut max_off = F::sparse_zero();
        for i in 0..k {
            for j in (i + 1)..k {
                let val = mat[i * k + j].abs();
                if val > max_off {
                    max_off = val;
                }
            }
        }
        if max_off < tol {
            break;
        }

        for p in 0..k {
            for q in (p + 1)..k {
                let apq = mat[p * k + q];
                if apq.abs() < tol {
                    continue;
                }
                let diff = mat[q * k + q] - mat[p * k + p];
                let theta = if diff.abs() < F::epsilon() {
                    F::from(std::f64::consts::FRAC_PI_4).unwrap_or(F::sparse_one())
                } else {
                    let tau = diff / (apq + apq);
                    // t = sign(tau) / (|tau| + sqrt(1 + tau^2))
                    let sign_tau = if tau >= F::sparse_zero() {
                        F::sparse_one()
                    } else {
                        -F::sparse_one()
                    };
                    let t = sign_tau / (tau.abs() + (F::sparse_one() + tau * tau).sqrt());
                    t.atan()
                };

                let (sin_t, cos_t) = (theta.sin(), theta.cos());

                // Rotate rows/cols p, q in mat
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
                let apq_old = mat[p * k + q];
                mat[p * k + p] = cos_t * cos_t * app
                    - F::from(2.0).unwrap_or(F::sparse_one()) * sin_t * cos_t * apq_old
                    + sin_t * sin_t * aqq;
                mat[q * k + q] = sin_t * sin_t * app
                    + F::from(2.0).unwrap_or(F::sparse_one()) * sin_t * cos_t * apq_old
                    + cos_t * cos_t * aqq;
                mat[p * k + q] = F::sparse_zero();
                mat[q * k + p] = F::sparse_zero();

                // Rotate eigenvectors
                for r in 0..k {
                    let vp = vecs[p * k + r];
                    let vq = vecs[q * k + r];
                    vecs[p * k + r] = cos_t * vp - sin_t * vq;
                    vecs[q * k + r] = sin_t * vp + cos_t * vq;
                }
            }
        }
    }

    let mut eigenvalues: Vec<F> = (0..k).map(|i| mat[i * k + i]).collect();

    // Sort eigenvalues and permute eigenvectors
    let mut perm: Vec<usize> = (0..k).collect();
    perm.sort_by(|&a_idx, &b_idx| {
        eigenvalues[a_idx]
            .partial_cmp(&eigenvalues[b_idx])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_vals: Vec<F> = perm.iter().map(|&i| eigenvalues[i]).collect();
    let mut sorted_vecs = vec![F::sparse_zero(); k * k];
    for (new_col, &old_col) in perm.iter().enumerate() {
        for r in 0..k {
            sorted_vecs[new_col * k + r] = vecs[old_col * k + r];
        }
    }

    eigenvalues = sorted_vals;
    Ok((eigenvalues, sorted_vecs))
}

/// Solve the small dense generalised symmetric eigenproblem
///   S z = lambda M z
/// where both S, M are (k x k) row-major and M is SPD.
/// Returns eigenvalues and eigenvectors (column-major k x k).
fn dense_generalised_eig<F>(s: &[F], m: &[F], k: usize) -> SparseResult<(Vec<F>, Vec<F>)>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    // Cholesky factorisation of M = L L^T
    let mut l_mat = vec![F::sparse_zero(); k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = m[i * k + j];
            for kk in 0..j {
                sum -= l_mat[i * k + kk] * l_mat[j * k + kk];
            }
            if i == j {
                if sum <= F::sparse_zero() {
                    // Fall back to standard eigensolve (M ~ I)
                    return dense_symmetric_eig(s, k);
                }
                l_mat[i * k + j] = sum.sqrt();
            } else {
                let l_jj = l_mat[j * k + j];
                if l_jj.abs() < F::epsilon() {
                    return dense_symmetric_eig(s, k);
                }
                l_mat[i * k + j] = sum / l_jj;
            }
        }
    }

    // Compute L^{-1}
    let mut l_inv = vec![F::sparse_zero(); k * k];
    for i in 0..k {
        l_inv[i * k + i] = F::sparse_one() / l_mat[i * k + i];
        for j in (i + 1)..k {
            let mut sum = F::sparse_zero();
            for kk in i..j {
                sum += l_mat[j * k + kk] * l_inv[kk * k + i];
            }
            l_inv[j * k + i] = -sum / l_mat[j * k + j];
        }
    }

    // Compute S' = L^{-1} S L^{-T}
    // First: T = L^{-1} S
    let mut temp = vec![F::sparse_zero(); k * k];
    for i in 0..k {
        for j in 0..k {
            let mut val = F::sparse_zero();
            for kk in 0..k {
                val += l_inv[i * k + kk] * s[kk * k + j];
            }
            temp[i * k + j] = val;
        }
    }
    // S' = T * L^{-T}
    let mut s_prime = vec![F::sparse_zero(); k * k];
    for i in 0..k {
        for j in 0..k {
            let mut val = F::sparse_zero();
            for kk in 0..k {
                val += temp[i * k + kk] * l_inv[j * k + kk]; // L^{-T}[kk,j] = L^{-1}[j,kk]
            }
            s_prime[i * k + j] = val;
        }
    }

    // Standard eigenproblem on S'
    let (eigenvalues, z_vecs) = dense_symmetric_eig(&s_prime, k)?;

    // Back-transform: x = L^{-T} z
    let mut eigenvectors = vec![F::sparse_zero(); k * k];
    for col in 0..k {
        for i in 0..k {
            let mut val = F::sparse_zero();
            for kk in 0..k {
                val += l_inv[kk * k + i] * z_vecs[col * k + kk];
            }
            eigenvectors[col * k + i] = val;
        }
    }

    Ok((eigenvalues, eigenvectors))
}

// ---------------------------------------------------------------------------
// LOBPCG solver
// ---------------------------------------------------------------------------

/// Run the LOBPCG eigensolver for the standard eigenvalue problem `A x = lambda x`.
///
/// # Arguments
///
/// * `a` - Sparse matrix (CSR format)
/// * `config` - Solver configuration
/// * `precond` - Optional preconditioner
/// * `initial_vectors` - Optional initial guesses (n x block_size column-major)
///
/// # Returns
///
/// A `LobpcgResult` containing eigenvalues, eigenvectors, iteration count, and
/// residual norms.
pub fn lobpcg<F>(
    a: &CsrMatrix<F>,
    config: &LobpcgConfig,
    precond: Option<&dyn Preconditioner<F>>,
    initial_vectors: Option<&Array2<F>>,
) -> SparseResult<LobpcgResult<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    lobpcg_generalised(a, None, config, precond, initial_vectors)
}

/// Run the LOBPCG eigensolver for the generalised eigenvalue problem
/// `A x = lambda B x`.
///
/// When `b` is `None`, it reduces to the standard problem (`B = I`).
///
/// # Arguments
///
/// * `a` - Left-hand-side sparse matrix (CSR format)
/// * `b` - Optional right-hand-side SPD sparse matrix (CSR format)
/// * `config` - Solver configuration
/// * `precond` - Optional preconditioner
/// * `initial_vectors` - Optional initial guesses (n x block_size column-major)
pub fn lobpcg_generalised<F>(
    a: &CsrMatrix<F>,
    b: Option<&CsrMatrix<F>>,
    config: &LobpcgConfig,
    precond: Option<&dyn Preconditioner<F>>,
    initial_vectors: Option<&Array2<F>>,
) -> SparseResult<LobpcgResult<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let (m, n_cols) = a.shape();
    if m != n_cols {
        return Err(SparseError::ValueError(
            "LOBPCG requires a square matrix".to_string(),
        ));
    }
    let n = m;
    let k = config.block_size;
    if k == 0 || k > n {
        return Err(SparseError::ValueError(format!(
            "block_size must be in [1, {n}], got {k}"
        )));
    }

    if let Some(b_mat) = b {
        let (bm, bn) = b_mat.shape();
        if bm != n || bn != n {
            return Err(SparseError::ShapeMismatch {
                expected: (n, n),
                found: (bm, bn),
            });
        }
    }

    let tol = F::from(config.tol)
        .ok_or_else(|| SparseError::ValueError("Failed to convert tolerance".to_string()))?;

    // ---- Initialise X (column-major: n * k) ----
    let mut x_buf = vec![F::sparse_zero(); n * k];
    if let Some(init) = initial_vectors {
        let (ir, ic) = (init.nrows(), init.ncols());
        if ir != n || ic != k {
            return Err(SparseError::ShapeMismatch {
                expected: (n, k),
                found: (ir, ic),
            });
        }
        for col in 0..k {
            for row in 0..n {
                x_buf[col * n + row] = init[[row, col]];
            }
        }
    } else {
        // Deterministic initial vectors: coordinate vectors + small perturbation
        for j in 0..k {
            if j < n {
                x_buf[j * n + j] = F::sparse_one();
            }
            // Add small deterministic perturbation for robustness
            for i in 0..n {
                let val = F::from((i + j + 1) as f64 / (n + k) as f64).unwrap_or(F::sparse_zero());
                x_buf[j * n + i] += val * F::from(0.01).unwrap_or(F::sparse_zero());
            }
        }
    }

    // B-orthogonalise X
    b_orthonormalise(&mut x_buf, b, n, k)?;

    // AX = A * X
    let mut ax_buf = csr_matmul_block(a, &x_buf, n, k)?;
    // BX = B * X  (or X if B is None)
    let mut bx_buf = match b {
        Some(b_mat) => csr_matmul_block(b_mat, &x_buf, n, k)?,
        None => x_buf.clone(),
    };

    // Rayleigh quotients (initial eigenvalue estimates)
    let mut lambdas = vec![F::sparse_zero(); k];
    for j in 0..k {
        lambdas[j] = dot(&x_buf[j * n..(j + 1) * n], &ax_buf[j * n..(j + 1) * n]);
    }

    // P buffer (conjugate directions) - initialised to zero, indicating first iter
    let mut p_buf = vec![F::sparse_zero(); n * k];
    let mut ap_buf = vec![F::sparse_zero(); n * k];
    let mut bp_buf = vec![F::sparse_zero(); n * k];
    let mut have_p = false;

    let mut locked_count = 0usize;
    let mut residual_norms = vec![F::sparse_zero(); k];
    let mut converged_flags = vec![false; k];
    let mut iter_count = 0usize;

    for iteration in 0..config.max_iter {
        iter_count = iteration + 1;

        // ---- Compute residuals R = AX - lambda * BX ----
        let active_start = locked_count;
        let active_k = k - active_start;
        if active_k == 0 {
            break;
        }

        let mut r_buf = vec![F::sparse_zero(); n * active_k];
        for j in 0..active_k {
            let gj = j + active_start;
            let lam = lambdas[gj];
            for i in 0..n {
                r_buf[j * n + i] = ax_buf[gj * n + i] - lam * bx_buf[gj * n + i];
            }
            residual_norms[gj] = norm2(&r_buf[j * n..(j + 1) * n]);
        }

        // ---- Check convergence ----
        let mut all_converged = true;
        for j in 0..active_k {
            let gj = j + active_start;
            if residual_norms[gj] < tol {
                converged_flags[gj] = true;
            } else {
                all_converged = false;
            }
        }

        if all_converged {
            break;
        }

        // ---- Locking ----
        if config.locking && active_k > 1 {
            let mut newly_locked = 0usize;
            for j in 0..active_k {
                let gj = j + active_start;
                if converged_flags[gj] && gj == locked_count + newly_locked {
                    newly_locked += 1;
                }
            }
            if newly_locked > 0 {
                locked_count += newly_locked;
                have_p = false; // reset conjugate directions after locking
                continue;
            }
        }

        // ---- Apply preconditioner: W = T^{-1} R ----
        let mut w_buf = vec![F::sparse_zero(); n * active_k];
        for j in 0..active_k {
            let r_col = &r_buf[j * n..(j + 1) * n];
            match precond {
                Some(pc) => {
                    let r_arr = Array1::from_vec(r_col.to_vec());
                    let w_arr = pc.apply(&r_arr)?;
                    for i in 0..n {
                        w_buf[j * n + i] = w_arr[i];
                    }
                }
                None => {
                    w_buf[j * n..(j + 1) * n].copy_from_slice(r_col);
                }
            }
        }

        // B-orthogonalise W against locked + X
        b_orthogonalise_against(&mut w_buf, &x_buf, b, n, active_k, k)?;
        // Orthonormalise W internally
        for j in 0..active_k {
            // Orthogonalise against previous W columns
            for prev in 0..j {
                let c = b_inner_product(
                    &w_buf[prev * n..(prev + 1) * n],
                    &w_buf[j * n..(j + 1) * n],
                    b,
                    n,
                )?;
                for i in 0..n {
                    w_buf[j * n + i] = w_buf[j * n + i] - c * w_buf[prev * n + i];
                }
            }
            normalise(&mut w_buf[j * n..(j + 1) * n]);
        }

        // AW = A * W
        let aw_buf = csr_matmul_block(a, &w_buf, n, active_k)?;
        // BW = B * W
        let bw_buf = match b {
            Some(b_mat) => csr_matmul_block(b_mat, &w_buf, n, active_k)?,
            None => w_buf.clone(),
        };

        // ---- Build the Rayleigh-Ritz problem on [X_active, W, P] ----
        let subspace_dim = if have_p {
            active_k + active_k + active_k // X_active + W + P
        } else {
            active_k + active_k // X_active + W
        };

        // Concatenate subspace vectors and A-products
        let mut s_vecs = vec![F::sparse_zero(); n * subspace_dim];
        let mut as_vecs = vec![F::sparse_zero(); n * subspace_dim];
        let mut bs_vecs = vec![F::sparse_zero(); n * subspace_dim];

        // Copy X_active
        for j in 0..active_k {
            let gj = j + active_start;
            s_vecs[j * n..(j + 1) * n].copy_from_slice(&x_buf[gj * n..(gj + 1) * n]);
            as_vecs[j * n..(j + 1) * n].copy_from_slice(&ax_buf[gj * n..(gj + 1) * n]);
            bs_vecs[j * n..(j + 1) * n].copy_from_slice(&bx_buf[gj * n..(gj + 1) * n]);
        }
        // Copy W
        let w_off = active_k;
        for j in 0..active_k {
            s_vecs[(w_off + j) * n..(w_off + j + 1) * n]
                .copy_from_slice(&w_buf[j * n..(j + 1) * n]);
            as_vecs[(w_off + j) * n..(w_off + j + 1) * n]
                .copy_from_slice(&aw_buf[j * n..(j + 1) * n]);
            bs_vecs[(w_off + j) * n..(w_off + j + 1) * n]
                .copy_from_slice(&bw_buf[j * n..(j + 1) * n]);
        }
        // Copy P (if available)
        if have_p {
            let p_off = active_k + active_k;
            for j in 0..active_k {
                let gj = j + active_start;
                s_vecs[(p_off + j) * n..(p_off + j + 1) * n]
                    .copy_from_slice(&p_buf[gj * n..(gj + 1) * n]);
                as_vecs[(p_off + j) * n..(p_off + j + 1) * n]
                    .copy_from_slice(&ap_buf[gj * n..(gj + 1) * n]);
                bs_vecs[(p_off + j) * n..(p_off + j + 1) * n]
                    .copy_from_slice(&bp_buf[gj * n..(gj + 1) * n]);
            }
        }

        // Gram matrices: S_gram = S^T A S,  M_gram = S^T B S
        let s_gram = gram_matrix(&s_vecs, &as_vecs, n, subspace_dim, subspace_dim);
        let m_gram = gram_matrix(&s_vecs, &bs_vecs, n, subspace_dim, subspace_dim);

        // Solve small generalised eigenproblem
        let (small_evals, small_evecs) = dense_generalised_eig(&s_gram, &m_gram, subspace_dim)?;

        // Select eigenvalues based on target
        let selected_indices: Vec<usize> = match config.target {
            EigenTarget::Smallest => (0..active_k).collect(),
            EigenTarget::Largest => {
                let start = subspace_dim.saturating_sub(active_k);
                (start..subspace_dim).collect()
            }
        };

        // ---- Update X, AX, BX using Ritz vectors ----
        // Also compute P = new_X - old_X
        let old_x_active: Vec<F> = (0..active_k)
            .flat_map(|j| {
                let gj = j + active_start;
                x_buf[gj * n..(gj + 1) * n].to_vec()
            })
            .collect();

        for (sel_idx, &eig_col) in selected_indices.iter().enumerate() {
            let gj = sel_idx + active_start;
            lambdas[gj] = small_evals[eig_col];

            // Compute the new X column: x_new = S * z
            let z_col = &small_evecs[eig_col * subspace_dim..(eig_col + 1) * subspace_dim];
            for i in 0..n {
                let mut xval = F::sparse_zero();
                let mut axval = F::sparse_zero();
                let mut bxval = F::sparse_zero();
                for (s_idx, &zc) in z_col.iter().enumerate() {
                    xval += zc * s_vecs[s_idx * n + i];
                    axval += zc * as_vecs[s_idx * n + i];
                    bxval += zc * bs_vecs[s_idx * n + i];
                }
                // P = x_new - x_old
                p_buf[gj * n + i] = xval - old_x_active[sel_idx * n + i];
                x_buf[gj * n + i] = xval;
                ax_buf[gj * n + i] = axval;
                bx_buf[gj * n + i] = bxval;
            }

            // AP = A * P,  BP = B * P
            let p_col = &p_buf[gj * n..(gj + 1) * n];
            let ap_col = csr_matvec(a, p_col)?;
            for i in 0..n {
                ap_buf[gj * n + i] = ap_col[i];
            }
            match b {
                Some(b_mat) => {
                    let bp_col = csr_matvec(b_mat, p_col)?;
                    for i in 0..n {
                        bp_buf[gj * n + i] = bp_col[i];
                    }
                }
                None => {
                    bp_buf[gj * n..(gj + 1) * n].copy_from_slice(p_col);
                }
            }
        }

        have_p = true;
    }

    // ---- Assemble final result ----
    let n_converged = converged_flags.iter().filter(|&&f| f).count();
    let all_converged = n_converged == k;

    let mut eigenvalues = Array1::zeros(k);
    let mut eigenvectors = Array2::zeros((n, k));
    for j in 0..k {
        eigenvalues[j] = lambdas[j];
        for i in 0..n {
            eigenvectors[[i, j]] = x_buf[j * n + i];
        }
    }

    Ok(LobpcgResult {
        eigenvalues,
        eigenvectors,
        iterations: iter_count,
        residual_norms,
        converged: all_converged,
        n_converged,
    })
}

// ---------------------------------------------------------------------------
// B-orthogonalisation helpers
// ---------------------------------------------------------------------------

/// Compute <u, v>_B = u^T B v  (or u^T v if B is None).
fn b_inner_product<F>(u: &[F], v: &[F], b: Option<&CsrMatrix<F>>, n: usize) -> SparseResult<F>
where
    F: Float + NumAssign + Sum + SparseElement + 'static,
{
    match b {
        Some(b_mat) => {
            let bv = csr_matvec(b_mat, v)?;
            Ok(dot(u, &bv))
        }
        None => Ok(dot(u, v)),
    }
}

/// B-orthonormalise columns of `mat` (column-major, n x k) using modified Gram-Schmidt.
fn b_orthonormalise<F>(
    mat: &mut [F],
    b: Option<&CsrMatrix<F>>,
    n: usize,
    k: usize,
) -> SparseResult<()>
where
    F: Float + NumAssign + Sum + SparseElement + 'static,
{
    for j in 0..k {
        // Orthogonalise against all previous columns
        for prev in 0..j {
            let c = b_inner_product(
                &mat[prev * n..(prev + 1) * n],
                &mat[j * n..(j + 1) * n],
                b,
                n,
            )?;
            for i in 0..n {
                mat[j * n + i] -= c * mat[prev * n + i];
            }
        }
        // B-normalise
        let bnorm = b_inner_product(&mat[j * n..(j + 1) * n], &mat[j * n..(j + 1) * n], b, n)?;
        if bnorm > F::epsilon() {
            let inv = F::sparse_one() / bnorm.sqrt();
            for i in 0..n {
                mat[j * n + i] *= inv;
            }
        }
    }
    Ok(())
}

/// Orthogonalise columns of `w` (n x wk) against columns of `q` (n x qk) w.r.t. B.
fn b_orthogonalise_against<F>(
    w: &mut [F],
    q: &[F],
    b: Option<&CsrMatrix<F>>,
    n: usize,
    wk: usize,
    qk: usize,
) -> SparseResult<()>
where
    F: Float + NumAssign + Sum + SparseElement + 'static,
{
    for j in 0..wk {
        for qi in 0..qk {
            let c = b_inner_product(&q[qi * n..(qi + 1) * n], &w[j * n..(j + 1) * n], b, n)?;
            for i in 0..n {
                w[j * n + i] -= c * q[qi * n + i];
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small SPD tridiagonal matrix: 2 on diagonal, -1 off-diagonal.
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

    /// Build a diagonal SPD matrix.
    fn build_diag_matrix(diag: &[f64]) -> CsrMatrix<f64> {
        let n = diag.len();
        let rows: Vec<usize> = (0..n).collect();
        let cols: Vec<usize> = (0..n).collect();
        CsrMatrix::new(diag.to_vec(), rows, cols, (n, n)).expect("valid matrix")
    }

    #[test]
    fn test_lobpcg_smallest_eigenvalue_tridiag() {
        let n = 20;
        let a = build_tridiag_spd(n);
        let config = LobpcgConfig {
            block_size: 2,
            max_iter: 200,
            tol: 1e-6,
            target: EigenTarget::Smallest,
            locking: true,
            verbose: false,
        };
        let result = lobpcg(&a, &config, None, None).expect("lobpcg should succeed");
        // The smallest eigenvalue of the 1D Laplacian is ~ 4 sin^2(pi/(2(n+1)))
        let lambda_min_exact = 4.0
            * (std::f64::consts::PI / (2.0 * (n as f64 + 1.0)))
                .sin()
                .powi(2);
        let computed = result.eigenvalues[0];
        assert!(
            (computed - lambda_min_exact).abs() < 1e-4,
            "Expected smallest eigenvalue ~{lambda_min_exact}, got {computed}"
        );
    }

    #[test]
    fn test_lobpcg_largest_eigenvalue_tridiag() {
        let n = 20;
        let a = build_tridiag_spd(n);
        let config = LobpcgConfig {
            block_size: 1,
            max_iter: 200,
            tol: 1e-6,
            target: EigenTarget::Largest,
            locking: false,
            verbose: false,
        };
        let result = lobpcg(&a, &config, None, None).expect("lobpcg should succeed");
        // The largest eigenvalue of the 1D Laplacian is ~ 4 cos^2(pi/(2(n+1)))
        let lambda_max_exact = 4.0
            * (std::f64::consts::PI * n as f64 / (2.0 * (n as f64 + 1.0)))
                .sin()
                .powi(2);
        let computed = result.eigenvalues[0];
        assert!(
            (computed - lambda_max_exact).abs() < 1e-3,
            "Expected largest eigenvalue ~{lambda_max_exact}, got {computed}"
        );
    }

    #[test]
    fn test_lobpcg_diagonal_matrix() {
        let diag = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let a = build_diag_matrix(&diag);
        let config = LobpcgConfig {
            block_size: 2,
            max_iter: 100,
            tol: 1e-10,
            target: EigenTarget::Smallest,
            locking: true,
            verbose: false,
        };
        let result = lobpcg(&a, &config, None, None).expect("lobpcg should succeed");
        // Smallest two eigenvalues should be 1.0 and 3.0
        let mut eigs: Vec<f64> = result.eigenvalues.to_vec();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert!(
            (eigs[0] - 1.0).abs() < 1e-6,
            "Expected 1.0, got {}",
            eigs[0]
        );
        assert!(
            (eigs[1] - 3.0).abs() < 1e-6,
            "Expected 3.0, got {}",
            eigs[1]
        );
    }

    #[test]
    fn test_lobpcg_generalised_with_identity_b() {
        let n = 10;
        let a = build_tridiag_spd(n);
        let diag_ones = vec![1.0; n];
        let b = build_diag_matrix(&diag_ones);
        let config = LobpcgConfig {
            block_size: 1,
            max_iter: 200,
            tol: 1e-6,
            target: EigenTarget::Smallest,
            ..Default::default()
        };
        let result =
            lobpcg_generalised(&a, Some(&b), &config, None, None).expect("generalised lobpcg");
        let lambda_min_exact = 4.0
            * (std::f64::consts::PI / (2.0 * (n as f64 + 1.0)))
                .sin()
                .powi(2);
        assert!(
            (result.eigenvalues[0] - lambda_min_exact).abs() < 1e-4,
            "Expected ~{lambda_min_exact}, got {}",
            result.eigenvalues[0]
        );
    }

    #[test]
    fn test_lobpcg_generalised_nontrivial_b() {
        // A = diag(2,4,6), B = diag(1,2,3)
        // Generalized eigenvalues: 2/1=2, 4/2=2, 6/3=2 => all eigenvalues = 2
        let a = build_diag_matrix(&[2.0, 4.0, 6.0]);
        let b = build_diag_matrix(&[1.0, 2.0, 3.0]);
        let config = LobpcgConfig {
            block_size: 1,
            max_iter: 100,
            tol: 1e-8,
            target: EigenTarget::Smallest,
            ..Default::default()
        };
        let result =
            lobpcg_generalised(&a, Some(&b), &config, None, None).expect("generalised lobpcg");
        assert!(
            (result.eigenvalues[0] - 2.0).abs() < 1e-4,
            "Expected 2.0, got {}",
            result.eigenvalues[0]
        );
    }

    #[test]
    fn test_lobpcg_with_preconditioner() {
        let n = 15;
        let a = build_tridiag_spd(n);
        let precond = JacobiPreconditioner::new(&a).expect("Jacobi precond");
        let config = LobpcgConfig {
            block_size: 1,
            max_iter: 200,
            tol: 1e-8,
            target: EigenTarget::Smallest,
            ..Default::default()
        };
        let result = lobpcg(&a, &config, Some(&precond), None).expect("lobpcg with precond");
        let lambda_min = 4.0
            * (std::f64::consts::PI / (2.0 * (n as f64 + 1.0)))
                .sin()
                .powi(2);
        assert!(
            (result.eigenvalues[0] - lambda_min).abs() < 1e-4,
            "Expected ~{lambda_min}, got {}",
            result.eigenvalues[0]
        );
    }

    #[test]
    fn test_lobpcg_with_initial_vectors() {
        let n = 10;
        let a = build_tridiag_spd(n);
        // Create a reasonable initial guess
        let mut init = Array2::zeros((n, 1));
        for i in 0..n {
            init[[i, 0]] = ((i + 1) as f64 * std::f64::consts::PI / (n as f64 + 1.0)).sin();
        }
        let config = LobpcgConfig {
            block_size: 1,
            max_iter: 100,
            tol: 1e-8,
            target: EigenTarget::Smallest,
            ..Default::default()
        };
        let result = lobpcg(&a, &config, None, Some(&init)).expect("lobpcg with initial vectors");
        let lambda_min = 4.0
            * (std::f64::consts::PI / (2.0 * (n as f64 + 1.0)))
                .sin()
                .powi(2);
        assert!(
            (result.eigenvalues[0] - lambda_min).abs() < 1e-4,
            "Expected ~{lambda_min}, got {}",
            result.eigenvalues[0]
        );
    }

    #[test]
    fn test_lobpcg_convergence_info() {
        let a = build_diag_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let config = LobpcgConfig {
            block_size: 1,
            max_iter: 100,
            tol: 1e-10,
            target: EigenTarget::Smallest,
            ..Default::default()
        };
        let result = lobpcg(&a, &config, None, None).expect("lobpcg");
        assert!(
            result.converged,
            "should converge on simple diagonal matrix"
        );
        assert!(result.iterations > 0);
        assert_eq!(result.n_converged, 1);
    }

    #[test]
    fn test_lobpcg_error_on_non_square() {
        let rows = vec![0, 1];
        let cols = vec![0, 1];
        let data = vec![1.0, 2.0];
        let a = CsrMatrix::new(data, rows, cols, (2, 3)).expect("valid rect matrix");
        let config = LobpcgConfig::default();
        let result = lobpcg(&a, &config, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_lobpcg_multiple_eigenvalues() {
        let n = 30;
        let a = build_tridiag_spd(n);
        let config = LobpcgConfig {
            block_size: 3,
            max_iter: 300,
            tol: 1e-5,
            target: EigenTarget::Smallest,
            locking: true,
            verbose: false,
        };
        let result = lobpcg(&a, &config, None, None).expect("lobpcg");
        // First 3 eigenvalues
        for j in 0..3 {
            let exact = 4.0
                * (std::f64::consts::PI * (j + 1) as f64 / (2.0 * (n as f64 + 1.0)))
                    .sin()
                    .powi(2);
            let computed = result.eigenvalues[j];
            assert!(
                (computed - exact).abs() < 0.05,
                "Eigenvalue {j}: expected ~{exact}, got {computed}"
            );
        }
    }

    #[test]
    fn test_dense_symmetric_eig_basic() {
        // 2x2 symmetric: [[2,1],[1,2]] => eigenvalues 1 and 3
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let (vals, _vecs) = dense_symmetric_eig(&a, 2).expect("dense eig");
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_dense_generalised_eig_identity_b() {
        let a = vec![3.0, 1.0, 1.0, 3.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let (vals, _) = dense_generalised_eig(&a, &b, 2).expect("dense gen eig");
        assert!((vals[0] - 2.0).abs() < 1e-10);
        assert!((vals[1] - 4.0).abs() < 1e-10);
    }

    // Helper: Jacobi preconditioner for tests
    use crate::iterative_solvers::JacobiPreconditioner;
}
