//! Randomized Numerical Linear Algebra (NLA) algorithms
//!
//! This module implements the core randomized NLA algorithms from the
//! Halko-Martinsson-Tropp (2011) framework and extensions:
//!
//! - **Randomized SVD** (full algorithm with power iteration)
//! - **Randomized eigendecomposition** for symmetric PSD matrices
//! - **Nyström approximation** for kernel matrices
//! - **CUR decomposition** via leverage score column/row sampling
//!
//! # References
//!
//! - Halko, N., Martinsson, P.G., Tropp, J.A. (2011). "Finding structure with
//!   randomness: Probabilistic algorithms for constructing approximate matrix
//!   decompositions." SIAM Review 53(2), 217-288.
//! - Mahoney, M.W. & Drineas, P. (2009). "CUR matrix decompositions for improved
//!   data analysis." PNAS.
//! - Williams, C. & Seeger, M. (2001). "Using the Nyström method to speed up
//!   kernel machines." NIPS.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign, One};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{Distribution, Normal};
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Trait alias
// ---------------------------------------------------------------------------

/// Trait alias for randomized NLA float operations.
pub trait RandNlaFloat:
    Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static + std::fmt::Display
{
}
impl<F> RandNlaFloat for F where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static + std::fmt::Display
{
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn matmul_nn<F: RandNlaFloat>(a: &Array2<F>, b: &Array2<F>) -> LinalgResult<Array2<F>> {
    let (m, ka) = (a.nrows(), a.ncols());
    let (kb, n) = (b.nrows(), b.ncols());
    if ka != kb {
        return Err(LinalgError::ShapeError(format!(
            "rand_nla matmul: {} vs {}",
            ka, kb
        )));
    }
    let mut c = Array2::<F>::zeros((m, n));
    for i in 0..m {
        for l in 0..ka {
            let a_il = a[[i, l]];
            if a_il == F::zero() {
                continue;
            }
            for j in 0..n {
                c[[i, j]] += a_il * b[[l, j]];
            }
        }
    }
    Ok(c)
}

fn transpose<F: RandNlaFloat>(a: &Array2<F>) -> Array2<F> {
    let (m, n) = (a.nrows(), a.ncols());
    let mut t = Array2::<F>::zeros((n, m));
    for i in 0..m {
        for j in 0..n {
            t[[j, i]] = a[[i, j]];
        }
    }
    t
}

fn frobenius_norm<F: RandNlaFloat>(a: &Array2<F>) -> F {
    let mut acc = F::zero();
    for &v in a.iter() {
        acc += v * v;
    }
    acc.sqrt()
}

fn lu_factorize<F: RandNlaFloat>(a: &Array2<F>) -> LinalgResult<(Array2<F>, Vec<usize>)> {
    let n = a.nrows();
    let mut lu = a.clone();
    let mut perm: Vec<usize> = (0..n).collect();
    for k in 0..n {
        let mut max_val = F::zero();
        let mut max_row = k;
        for i in k..n {
            let v = lu[[i, k]].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        if max_val < F::epsilon() * F::from(100.0).unwrap_or(F::one()) {
            return Err(LinalgError::SingularMatrixError(
                "rand_nla: near-singular in LU".into(),
            ));
        }
        if max_row != k {
            perm.swap(k, max_row);
            for j in 0..n {
                let tmp = lu[[k, j]];
                lu[[k, j]] = lu[[max_row, j]];
                lu[[max_row, j]] = tmp;
            }
        }
        for i in (k + 1)..n {
            lu[[i, k]] = lu[[i, k]] / lu[[k, k]];
            for j in (k + 1)..n {
                let l_ik = lu[[i, k]];
                let u_kj = lu[[k, j]];
                lu[[i, j]] -= l_ik * u_kj;
            }
        }
    }
    Ok((lu, perm))
}

fn lu_solve<F: RandNlaFloat>(lu: &Array2<F>, perm: &[usize], b: &Array2<F>) -> Array2<F> {
    let n = lu.nrows();
    let nrhs = b.ncols();
    let mut x = Array2::<F>::zeros((n, nrhs));
    for col in 0..nrhs {
        let mut y = vec![F::zero(); n];
        for i in 0..n {
            y[i] = b[[perm[i], col]];
        }
        for i in 0..n {
            for j in 0..i {
                y[i] = y[i] - lu[[i, j]] * y[j];
            }
        }
        let mut z = vec![F::zero(); n];
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum -= lu[[i, j]] * z[j];
            }
            z[i] = sum / lu[[i, i]];
        }
        for i in 0..n {
            x[[i, col]] = z[i];
        }
    }
    x
}

fn mat_inv<F: RandNlaFloat>(a: &Array2<F>) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    let eye = Array2::<F>::eye(n);
    let (lu, perm) = lu_factorize(a)?;
    Ok(lu_solve(&lu, &perm, &eye))
}

// ---------------------------------------------------------------------------
// Randomized range finder (Stage A of HMT framework)
// ---------------------------------------------------------------------------

/// Compute a randomized range finder: an orthonormal basis Q for range(A).
///
/// Algorithm (HMT 2011, Algorithm 4.3 with power iteration):
/// 1. Draw random Gaussian matrix Omega (n × (k + p))
/// 2. Form Y = A * Omega
/// 3. Apply q power iterations: Y = (A*A^T)^q * Y
/// 4. QR factorization of Y → Q
///
/// # Arguments
///
/// * `a` - m×n input matrix
/// * `rank` - target rank
/// * `oversampling` - extra columns for stability (default 10)
/// * `power_iter` - number of power iterations for slow spectral decay (default 0)
/// * `rng` - random number generator
///
/// # Returns
///
/// * Q: m×(rank+oversampling) orthonormal matrix spanning approx range(A)
fn randomized_range_finder<F: RandNlaFloat>(
    a: &ArrayView2<F>,
    rank: usize,
    oversampling: usize,
    power_iter: usize,
    rng: &mut impl Rng,
) -> LinalgResult<Array2<F>> {
    let (m, n) = (a.nrows(), a.ncols());
    let l = (rank + oversampling).min(n).min(m);

    let normal = Normal::new(0.0_f64, 1.0)
        .map_err(|e| LinalgError::ComputationError(format!("Normal dist: {e}")))?;

    // Draw Omega (n×l)
    let mut omega = Array2::<F>::zeros((n, l));
    for v in omega.iter_mut() {
        *v = F::from(normal.sample(rng)).unwrap_or(F::zero());
    }

    // Y = A * Omega (m×l)
    let mut y = matmul_nn(&a.to_owned(), &omega)?;

    // Power iterations: Y = (A*A^T)^q * Y
    let at = transpose(&a.to_owned());
    for _ in 0..power_iter {
        // LU-orthogonalize Y to avoid numerical issues
        let (q, _) = qr_thin(&y)?;
        y = q;

        // y = A * (A^T * y)
        let aty = matmul_nn(&at, &y)?;
        y = matmul_nn(&a.to_owned(), &aty)?;
    }

    // QR factorization of Y
    let (q, _) = qr_thin(&y)?;
    Ok(q)
}

/// Thin QR factorization: returns (Q, R) where Q is m×k orthonormal.
/// Uses Gram-Schmidt with re-orthogonalization for numerical stability.
fn qr_thin<F: RandNlaFloat>(a: &Array2<F>) -> LinalgResult<(Array2<F>, Array2<F>)> {
    let (m, n) = (a.nrows(), a.ncols());
    let k = n.min(m);

    let mut q = Array2::<F>::zeros((m, k));
    let mut r = Array2::<F>::zeros((k, k));

    for j in 0..k {
        // Copy column j
        let mut v = vec![F::zero(); m];
        for i in 0..m {
            v[i] = a[[i, j]];
        }

        // Modified Gram-Schmidt with re-orthogonalization
        for pass in 0..2 {
            let _ = pass;
            for i in 0..j {
                let mut dot = F::zero();
                for l in 0..m {
                    dot += q[[l, i]] * v[l];
                }
                if j == 0 || pass == 0 {
                    r[[i, j]] += dot;
                }
                for l in 0..m {
                    v[l] -= dot * q[[l, i]];
                }
            }
        }

        // Normalize
        let norm = v.iter().fold(F::zero(), |acc, &x| acc + x * x).sqrt();
        if norm < F::from(1e-14).unwrap_or(F::epsilon()) {
            // Linearly dependent: insert zero column
            r[[j, j]] = F::zero();
        } else {
            r[[j, j]] = norm;
            for l in 0..m {
                q[[l, j]] = v[l] / norm;
            }
        }
    }

    Ok((q, r))
}

// ---------------------------------------------------------------------------
// Randomized SVD (full Halko-Martinsson-Tropp algorithm)
// ---------------------------------------------------------------------------

/// Compute a randomized truncated SVD.
///
/// The full algorithm (HMT 2011):
/// 1. Randomized range finder: Q = range_finder(A, rank, oversampling, power_iter)
/// 2. Form B = Q^T * A (small matrix: (rank+p) × n)
/// 3. Compute exact SVD of B: B = Ub * Σ * V^T
/// 4. Reconstruct: U = Q * Ub
///
/// # Arguments
///
/// * `a` - m×n input matrix
/// * `rank` - target rank
/// * `oversampling` - extra columns for numerical stability (default 10)
/// * `power_iter` - power iterations for slowly decaying spectra (default 2)
/// * `rng` - random number generator
///
/// # Returns
///
/// * `(U, s, Vt)` where U is m×rank, s is rank, Vt is rank×n
///   and A ≈ U * diag(s) * Vt
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::random::prelude::*;
/// use scirs2_linalg::randomized::rand_nla::randomized_svd_full;
///
/// let a = array![
///     [1.0_f64, 2.0, 3.0],
///     [4.0, 5.0, 6.0],
///     [7.0, 8.0, 9.0],
/// ];
/// let mut rng = scirs2_core::random::seeded_rng(42);
/// let (u, s, vt) = randomized_svd_full(&a.view(), 2, 2, 1, &mut rng).expect("rand_svd");
/// assert_eq!(u.nrows(), 3);
/// assert_eq!(s.len(), 2);
/// assert_eq!(vt.nrows(), 2);
/// ```
pub fn randomized_svd_full<F: RandNlaFloat>(
    a: &ArrayView2<F>,
    rank: usize,
    oversampling: usize,
    power_iter: usize,
    rng: &mut impl Rng,
) -> LinalgResult<(Array2<F>, Array1<F>, Array2<F>)> {
    let (m, n) = (a.nrows(), a.ncols());
    if rank == 0 {
        return Err(LinalgError::InvalidInputError(
            "randomized_svd_full: rank must be >= 1".into(),
        ));
    }

    let l = (rank + oversampling).min(m).min(n);

    // Stage A: Compute orthonormal basis Q for the range of A
    let q = randomized_range_finder(a, rank, oversampling, power_iter, rng)?;

    // Stage B: Project A into small space: B = Q^T * A (l × n)
    let qt = transpose(&q);
    let b = matmul_nn(&qt, &a.to_owned())?;

    // Exact SVD of B (l × n, small matrix)
    let (ub, sb, vbt) = exact_svd_small(&b)?;

    // Reconstruct U = Q * Ub (m × l)
    let u_full = matmul_nn(&q, &ub)?;

    // Truncate to target rank
    let r = rank.min(l).min(sb.len());

    let mut u_out = Array2::<F>::zeros((m, r));
    let mut s_out = Array1::<F>::zeros(r);
    let mut vt_out = Array2::<F>::zeros((r, n));

    for j in 0..r {
        for i in 0..m {
            u_out[[i, j]] = u_full[[i, j]];
        }
        s_out[j] = sb[j];
        for i in 0..n {
            vt_out[[j, i]] = vbt[[j, i]];
        }
    }

    Ok((u_out, s_out, vt_out))
}

/// Compute exact SVD of a small matrix via Jacobi-based bidiagonalization.
///
/// For small matrices (l ≤ ~100), uses Golub-Reinsch-style computation.
/// Returns (U, s, Vt) in descending singular value order.
fn exact_svd_small<F: RandNlaFloat>(a: &Array2<F>) -> LinalgResult<(Array2<F>, Vec<F>, Array2<F>)> {
    use crate::decomposition::svd;
    let (u, s_array, vt) = svd(&a.view(), true, None)?;
    let s: Vec<F> = s_array.iter().copied().collect();
    Ok((u, s, vt))
}

// ---------------------------------------------------------------------------
// Randomized eigendecomposition for symmetric PSD matrices
// ---------------------------------------------------------------------------

/// Compute a randomized eigendecomposition of a symmetric PSD matrix.
///
/// For a symmetric PSD matrix A ≈ V * D * V^T (k-truncated), uses:
/// 1. Randomized range finder with symmetric power iteration
/// 2. Rayleigh-Ritz procedure: B = Q^T * A * Q, eigen(B) = (Λ, W)
/// 3. Eigenvectors: V = Q * W
///
/// # Arguments
///
/// * `a` - n×n symmetric positive semidefinite matrix
/// * `k` - Number of eigenpairs to compute
/// * `rng` - Random number generator
///
/// # Returns
///
/// * `(eigenvalues, eigenvectors)` in descending eigenvalue order
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::random::prelude::*;
/// use scirs2_linalg::randomized::rand_nla::randomized_eigendecomposition;
///
/// let a = array![[4.0_f64, 0.0], [0.0, 1.0]];
/// let mut rng = scirs2_core::random::seeded_rng(42);
/// let (vals, vecs) = randomized_eigendecomposition(&a.view(), 2, &mut rng).expect("rand_eig");
/// // Eigenvalues should be approximately 4 and 1
/// assert!((vals[0] - 4.0).abs() < 0.5);
/// ```
pub fn randomized_eigendecomposition<F: RandNlaFloat>(
    a: &ArrayView2<F>,
    k: usize,
    rng: &mut impl Rng,
) -> LinalgResult<(Array1<F>, Array2<F>)> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "randomized_eigendecomposition: matrix must be square".into(),
        ));
    }
    if k == 0 {
        return Err(LinalgError::InvalidInputError(
            "randomized_eigendecomposition: k must be >= 1".into(),
        ));
    }

    let oversampling = 10.min(n.saturating_sub(k));
    let power_iter = 2;
    let l = (k + oversampling).min(n);

    // Draw random test matrix Omega (n×l) and compute Y = A * Omega
    let normal = Normal::new(0.0_f64, 1.0)
        .map_err(|e| LinalgError::ComputationError(format!("Normal: {e}")))?;

    let mut omega = Array2::<F>::zeros((n, l));
    for v in omega.iter_mut() {
        *v = F::from(normal.sample(rng)).unwrap_or(F::zero());
    }

    let mut y = matmul_nn(&a.to_owned(), &omega)?;

    // Power iterations (symmetric: Y = (A^2)^{q} * Y)
    for _ in 0..power_iter {
        let (q, _) = qr_thin(&y)?;
        y = matmul_nn(&a.to_owned(), &q)?;
        let (q2, _) = qr_thin(&y)?;
        y = matmul_nn(&a.to_owned(), &q2)?;
    }

    // QR of Y → Q
    let (q, _) = qr_thin(&y)?;

    // Rayleigh-Ritz: B = Q^T * A * Q (l×l symmetric)
    let qt = transpose(&q);
    let aq = matmul_nn(&a.to_owned(), &q)?;
    let b = matmul_nn(&qt, &aq)?;

    // Exact eigendecomposition of small B
    let (eigenvals_b, eigenvecs_b) = symmetric_eigen_small(&b)?;

    // Compute full eigenvectors: V = Q * W
    let v_full = matmul_nn(&q, &eigenvecs_b)?;

    // Truncate and sort (descending)
    let r = k.min(eigenvals_b.len());
    let mut pairs: Vec<(F, usize)> = eigenvals_b
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut vals_out = Array1::<F>::zeros(r);
    let mut vecs_out = Array2::<F>::zeros((n, r));

    for (j, (val, idx)) in pairs.iter().take(r).enumerate() {
        vals_out[j] = *val;
        for i in 0..n {
            vecs_out[[i, j]] = v_full[[i, *idx]];
        }
    }

    Ok((vals_out, vecs_out))
}

/// Compute exact symmetric eigendecomposition of a small matrix.
/// Uses the existing eigen infrastructure.
fn symmetric_eigen_small<F: RandNlaFloat>(a: &Array2<F>) -> LinalgResult<(Vec<F>, Array2<F>)> {
    // Use crate's eig function and take real parts for symmetric matrix
    use crate::eigen::eig;
    let (complex_vals, complex_vecs) = eig(&a.view(), None)?;

    let n = a.nrows();
    let m = complex_vals.len();
    let mut vals = Vec::with_capacity(m);
    let mut vecs = Array2::<F>::zeros((n, m));

    for (j, cv) in complex_vals.iter().enumerate() {
        vals.push(F::from(cv.re.to_f64().unwrap_or(0.0)).unwrap_or(F::zero()));
        for i in 0..n {
            vecs[[i, j]] =
                F::from(complex_vecs[[i, j]].re.to_f64().unwrap_or(0.0)).unwrap_or(F::zero());
        }
    }

    Ok((vals, vecs))
}

// ---------------------------------------------------------------------------
// Nyström approximation
// ---------------------------------------------------------------------------

/// Compute the Nyström approximation of a symmetric PSD kernel matrix.
///
/// The Nyström method (Williams & Seeger 2001) approximates A (n×n PSD) as:
/// ```text
/// A ≈ C * (A_k)^† * C^T
/// ```
/// where:
/// - Omega is a random test matrix (n×k)
/// - C = A * Omega (n×k)
/// - A_k = Omega^T * C = Omega^T * A * Omega (k×k, small)
/// - (A_k)^† is the pseudo-inverse
///
/// # Arguments
///
/// * `a` - n×n symmetric PSD matrix (kernel matrix)
/// * `k` - Rank of approximation
/// * `rng` - Random number generator
///
/// # Returns
///
/// * Rank-k approximation of A (n×n)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::random::prelude::*;
/// use scirs2_linalg::randomized::rand_nla::nystrom_approximation;
///
/// // SPD matrix: Gram matrix of [1,0], [0,1], [1,1]
/// let a = array![
///     [1.0_f64, 0.0, 1.0],
///     [0.0, 1.0, 1.0],
///     [1.0, 1.0, 2.0],
/// ];
/// let mut rng = scirs2_core::random::seeded_rng(42);
/// let a_approx = nystrom_approximation(&a.view(), 2, &mut rng).expect("nystrom");
/// assert_eq!(a_approx.nrows(), 3);
/// assert_eq!(a_approx.ncols(), 3);
/// ```
pub fn nystrom_approximation<F: RandNlaFloat>(
    a: &ArrayView2<F>,
    k: usize,
    rng: &mut impl Rng,
) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "nystrom_approximation: matrix must be square".into(),
        ));
    }
    if k == 0 || k > n {
        return Err(LinalgError::InvalidInputError(format!(
            "nystrom_approximation: k={} must be in [1, {}]",
            k, n
        )));
    }

    let normal = Normal::new(0.0_f64, 1.0)
        .map_err(|e| LinalgError::ComputationError(format!("Normal: {e}")))?;

    // Draw Omega (n×k)
    let mut omega = Array2::<F>::zeros((n, k));
    for v in omega.iter_mut() {
        *v = F::from(normal.sample(rng)).unwrap_or(F::zero());
    }

    // C = A * Omega (n×k)
    let c = matmul_nn(&a.to_owned(), &omega)?;

    // A_k = Omega^T * C = Omega^T * A * Omega (k×k)
    let omega_t = transpose(&omega);
    let a_k = matmul_nn(&omega_t, &c)?;

    // Compute pseudo-inverse of A_k via regularized LU inversion
    // Add small regularization for stability
    let mut a_k_reg = a_k.clone();
    let reg = F::from(1e-10).unwrap_or(F::epsilon());
    for i in 0..k {
        a_k_reg[[i, i]] += reg;
    }

    let a_k_inv = mat_inv(&a_k_reg)?;

    // Approximation: C * A_k^{-1} * C^T
    let c_ak_inv = matmul_nn(&c, &a_k_inv)?;
    let ct = transpose(&c);
    let result = matmul_nn(&c_ak_inv, &ct)?;

    Ok(result)
}

// ---------------------------------------------------------------------------
// CUR decomposition via leverage score sampling
// ---------------------------------------------------------------------------

/// Compute a CUR decomposition via leverage-score column and row sampling.
///
/// The CUR decomposition (Mahoney & Drineas 2009) decomposes A ≈ C * U * R where:
/// - C: c selected columns of A (m×c)
/// - R: r selected rows of A (r×n)
/// - U: a linking matrix (c×r)
///
/// Column selection uses column leverage scores (proportional to squared row norms
/// of the right singular vectors), and row selection uses row leverage scores.
///
/// # Algorithm
///
/// 1. Compute randomized SVD to get approximate V^T (for column leverage)
///    and approximate U (for row leverage)
/// 2. Sample c columns using column leverage scores
/// 3. Sample r rows using row leverage scores
/// 4. Compute U = C^† * A * R^† (linking matrix)
///
/// # Arguments
///
/// * `a` - m×n input matrix
/// * `c` - Number of columns to sample
/// * `r` - Number of rows to sample
/// * `rng` - Random number generator
///
/// # Returns
///
/// * `(C, U, R)` where C is m×c, U is c×r, R is r×n, A ≈ C*U*R
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::random::prelude::*;
/// use scirs2_linalg::randomized::rand_nla::column_sampling_cur;
///
/// let a = array![
///     [1.0_f64, 2.0, 3.0],
///     [4.0, 5.0, 6.0],
///     [7.0, 8.0, 9.0],
/// ];
/// let mut rng = scirs2_core::random::seeded_rng(42);
/// let (c, u, r) = column_sampling_cur(&a.view(), 2, 2, &mut rng).expect("cur");
/// assert_eq!(c.nrows(), 3);
/// assert_eq!(c.ncols(), 2);
/// assert_eq!(r.nrows(), 2);
/// assert_eq!(r.ncols(), 3);
/// ```
pub fn column_sampling_cur<F: RandNlaFloat>(
    a: &ArrayView2<F>,
    c_count: usize,
    r_count: usize,
    rng: &mut impl Rng,
) -> LinalgResult<(Array2<F>, Array2<F>, Array2<F>)> {
    let (m, n) = (a.nrows(), a.ncols());

    if c_count == 0 || c_count > n {
        return Err(LinalgError::InvalidInputError(format!(
            "column_sampling_cur: c_count={} must be in [1, {}]",
            c_count, n
        )));
    }
    if r_count == 0 || r_count > m {
        return Err(LinalgError::InvalidInputError(format!(
            "column_sampling_cur: r_count={} must be in [1, {}]",
            r_count, m
        )));
    }

    // Compute approximate leverage scores via randomized SVD
    let rank = c_count.min(r_count).min(m).min(n);
    let (u_approx, _s, vt_approx) = randomized_svd_full(a, rank, 5.min(m.min(n) - rank), 1, rng)?;

    // Column leverage scores: l_j = ||e_j^T Vt||^2 / rank
    let mut col_scores = Array1::<F>::zeros(n);
    let vt_cols = vt_approx.ncols();
    let vt_rows = vt_approx.nrows();
    for j in 0..vt_cols {
        let mut norm_sq = F::zero();
        for i in 0..vt_rows {
            norm_sq += vt_approx[[i, j]] * vt_approx[[i, j]];
        }
        col_scores[j] = norm_sq;
    }

    // Row leverage scores: l_i = ||e_i^T U||^2 / rank
    let mut row_scores = Array1::<F>::zeros(m);
    let u_cols = u_approx.ncols();
    for i in 0..u_approx.nrows() {
        let mut norm_sq = F::zero();
        for j in 0..u_cols {
            norm_sq += u_approx[[i, j]] * u_approx[[i, j]];
        }
        row_scores[i] = norm_sq;
    }

    // Sample column indices
    let col_indices = sample_by_scores(&col_scores, c_count, rng);
    // Sample row indices
    let row_indices = sample_by_scores(&row_scores, r_count, rng);

    // Build C (m×c): selected columns of A
    let mut c_mat = Array2::<F>::zeros((m, c_count));
    for (j_out, &j_in) in col_indices.iter().enumerate() {
        for i in 0..m {
            c_mat[[i, j_out]] = a[[i, j_in]];
        }
    }

    // Build R (r×n): selected rows of A
    let mut r_mat = Array2::<F>::zeros((r_count, n));
    for (i_out, &i_in) in row_indices.iter().enumerate() {
        for j in 0..n {
            r_mat[[i_out, j]] = a[[i_in, j]];
        }
    }

    // Build intersection W (c×r): rows of C at selected row indices
    // W = A[row_indices, col_indices]
    let mut w = Array2::<F>::zeros((c_count, r_count));
    for (j_out, &j_in) in col_indices.iter().enumerate() {
        for (i_out, &i_in) in row_indices.iter().enumerate() {
            w[[j_out, i_out]] = a[[i_in, j_in]];
        }
    }

    // U = W^† (Moore-Penrose pseudo-inverse of W)
    // Use SVD-based pseudo-inverse: W = Uw * Sw * Vwt, W^† = Vwt^T * Sw^{-1} * Uw^T
    let u_link = pseudoinverse_via_svd(&w, None)?;

    Ok((c_mat, u_link, r_mat))
}

/// Compute Moore-Penrose pseudo-inverse via SVD.
fn pseudoinverse_via_svd<F: RandNlaFloat>(
    a: &Array2<F>,
    tol: Option<F>,
) -> LinalgResult<Array2<F>> {
    let (m, n) = (a.nrows(), a.ncols());
    use crate::decomposition::svd;
    let (u, s, vt) = svd(&a.view(), true, None)?;

    let tol_val = tol.unwrap_or_else(|| {
        let max_dim = m.max(n);
        let max_s = s
            .iter()
            .copied()
            .fold(F::zero(), |acc, x| if x > acc { x } else { acc });
        max_s * F::from(max_dim).unwrap_or(F::one()) * F::epsilon()
    });

    // Compute Vt^T * Σ^{-1} * U^T
    let r = s.len();
    let v = transpose(&vt); // n×r
    let ut = transpose(&u); // r×m

    // Scale columns of V by 1/s_i (for non-zero singular values)
    let mut vs_inv = Array2::<F>::zeros((n, r));
    for j in 0..r {
        if s[j] > tol_val {
            let s_inv = F::one() / s[j];
            for i in 0..n {
                vs_inv[[i, j]] = v[[i, j]] * s_inv;
            }
        }
        // else: leave as zeros (truncated)
    }

    // Pinv = Vs_inv * Ut
    matmul_nn(&vs_inv, &ut)
}

/// Sample indices by importance scores (with replacement, proportional to scores).
fn sample_by_scores<F: RandNlaFloat>(
    scores: &Array1<F>,
    k: usize,
    rng: &mut impl Rng,
) -> Vec<usize> {
    let n = scores.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }

    let total: f64 = scores
        .iter()
        .map(|&v| v.abs().to_f64().unwrap_or(0.0))
        .sum::<f64>()
        .max(f64::EPSILON);

    let mut cdf = vec![0.0f64; n + 1];
    for i in 0..n {
        cdf[i + 1] = cdf[i] + scores[i].abs().to_f64().unwrap_or(0.0) / total;
    }

    let uniform = scirs2_core::random::Uniform::new(0.0_f64, 1.0).unwrap_or_else(|_| {
        scirs2_core::random::Uniform::new(0.0, 1.0 - f64::EPSILON).expect("unexpected None or Err")
    });

    (0..k)
        .map(|_| {
            let u = uniform.sample(rng);
            // Binary search
            let mut lo = 0;
            let mut hi = n;
            while lo < hi {
                let mid = (lo + hi) / 2;
                if cdf[mid + 1] < u {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo.min(n - 1)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Approximation error utility
// ---------------------------------------------------------------------------

/// Estimate the error of a low-rank approximation in Frobenius norm.
///
/// Computes ||A - Q * Q^T * A||_F where Q is an orthonormal matrix.
///
/// # Arguments
///
/// * `a` - Original matrix
/// * `q` - Orthonormal matrix spanning approximate range of A
///
/// # Returns
///
/// * Frobenius norm of the approximation error
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_core::random::prelude::*;
/// use scirs2_linalg::randomized::rand_nla::approximation_error;
///
/// let a = Array2::<f64>::eye(3);
/// let q = Array2::<f64>::eye(3);
/// let err = approximation_error(&a.view(), &q.view()).expect("approx error");
/// assert!(err < 1e-10);
/// ```
pub fn approximation_error<F: RandNlaFloat>(
    a: &ArrayView2<F>,
    q: &ArrayView2<F>,
) -> LinalgResult<F> {
    // Compute Q * Q^T * A and subtract from A
    let qt = transpose(&q.to_owned());
    let qta = matmul_nn(&qt, &a.to_owned())?;
    let proj = matmul_nn(&q.to_owned(), &qta)?;

    let mut diff = a.to_owned();
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            diff[[i, j]] -= proj[[i, j]];
        }
    }

    Ok(frobenius_norm(&diff))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;
    use scirs2_core::random::prelude::*;

    fn make_rng() -> impl Rng {
        scirs2_core::random::seeded_rng(777)
    }

    #[test]
    fn test_randomized_svd_dimensions() {
        let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],];
        let mut rng = make_rng();
        let (u, s, vt) = randomized_svd_full(&a.view(), 2, 2, 1, &mut rng).expect("svd dims");
        assert_eq!(u.nrows(), 3);
        assert_eq!(u.ncols(), 2);
        assert_eq!(s.len(), 2);
        assert_eq!(vt.nrows(), 2);
        assert_eq!(vt.ncols(), 3);
    }

    #[test]
    fn test_randomized_svd_singular_values_positive() {
        let mut a = Array2::<f64>::zeros((5, 4));
        for i in 0..5 {
            for j in 0..4 {
                a[[i, j]] = (i as f64 + 1.0) * (j as f64 + 1.0);
            }
        }
        let mut rng = make_rng();
        let (_u, s, _vt) = randomized_svd_full(&a.view(), 2, 2, 0, &mut rng).expect("svd pos");
        for &sv in s.iter() {
            assert!(sv >= 0.0, "singular value {} should be non-negative", sv);
        }
    }

    #[test]
    fn test_randomized_svd_identity_reconstruction() {
        // For rank-1 matrix, rank-1 SVD should reconstruct perfectly
        let u_true = array![[1.0_f64], [0.0], [0.0]];
        let s_true = 5.0_f64;
        let vt_true = array![[0.0_f64, 1.0, 0.0]];
        let mut a = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                a[[i, j]] = s_true * u_true[[i, 0]] * vt_true[[0, j]];
            }
        }

        let mut rng = make_rng();
        let (u, s, vt) = randomized_svd_full(&a.view(), 1, 5, 2, &mut rng).expect("svd rank1");
        // Reconstruct
        let mut a_rec = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                a_rec[[i, j]] = s[0] * u[[i, 0]] * vt[[0, j]];
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(a_rec[[i, j]].abs(), a[[i, j]].abs(), epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_nystrom_approximation_dimensions() {
        let a = array![[2.0_f64, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0],];
        let mut rng = make_rng();
        let approx = nystrom_approximation(&a.view(), 2, &mut rng).expect("nystrom dims");
        assert_eq!(approx.nrows(), 3);
        assert_eq!(approx.ncols(), 3);
    }

    #[test]
    fn test_nystrom_approximation_identity() {
        // For rank-k matrix, Nyström with k samples should be exact
        let eye = Array2::<f64>::eye(3);
        let mut rng = make_rng();
        let approx = nystrom_approximation(&eye.view(), 3, &mut rng).expect("nystrom identity");
        // Should approximate identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(approx[[i, j]], expected, epsilon = 0.1);
            }
        }
    }

    #[test]
    fn test_column_sampling_cur_dimensions() {
        let a = array![
            [1.0_f64, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ];
        let mut rng = make_rng();
        let (c, u, r) = column_sampling_cur(&a.view(), 2, 2, &mut rng).expect("cur dims");
        assert_eq!(c.nrows(), 3);
        assert_eq!(c.ncols(), 2);
        assert_eq!(u.nrows(), 2);
        assert_eq!(r.ncols(), 4);
        assert_eq!(r.nrows(), 2);
    }

    #[test]
    fn test_approximation_error_perfect_range() {
        // If Q spans the full column space of A, error should be zero
        let a = Array2::<f64>::eye(3);
        let q = Array2::<f64>::eye(3);
        let err = approximation_error(&a.view(), &q.view()).expect("approx error perfect");
        assert_abs_diff_eq!(err, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_approximation_error_subspace() {
        // If Q spans only first column of A (identity), error = sqrt(n-1)
        let a = Array2::<f64>::eye(3);
        let mut q = Array2::<f64>::zeros((3, 1));
        q[[0, 0]] = 1.0;
        let err = approximation_error(&a.view(), &q.view()).expect("approx error subspace");
        // Error = ||A - e1 e1^T||_F = ||diag(0,1,1)||_F = sqrt(2)
        assert_abs_diff_eq!(err, 2.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_randomized_eigen_dimensions() {
        let a = array![[4.0_f64, 0.0], [0.0, 1.0]];
        let mut rng = make_rng();
        let (vals, vecs) = randomized_eigendecomposition(&a.view(), 2, &mut rng).expect("rand_eig");
        assert_eq!(vals.len(), 2);
        assert_eq!(vecs.nrows(), 2);
        assert_eq!(vecs.ncols(), 2);
    }

    #[test]
    fn test_qr_thin_orthonormality() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0],];
        let (q, _r) = qr_thin(&a).expect("qr_thin");
        // Check Q^T Q = I
        let qt = transpose(&q);
        let qtq = matmul_nn(&qt, &q).expect("qtq");
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(qtq[[i, j]], expected, epsilon = 1e-10);
            }
        }
    }
}
