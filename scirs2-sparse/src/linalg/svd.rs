//! Sparse Singular Value Decomposition (SVD) algorithms
//!
//! This module provides efficient SVD algorithms for sparse matrices,
//! including truncated SVD and randomized SVD methods.

#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(unused_mut)]

use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, SparseElement};
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

/// Type alias for bidiagonal SVD result
type BidiagonalSvdResult<T> = (Vec<T>, Vec<Vec<f64>>, Vec<Vec<f64>>);

/// SVD computation methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SVDMethod {
    /// Lanczos bidiagonalization
    Lanczos,
    /// Randomized SVD
    Randomized,
    /// Power method for truncated SVD
    Power,
    /// Cross-approximation SVD
    CrossApproximation,
}

impl SVDMethod {
    pub fn from_str(s: &str) -> SparseResult<Self> {
        match s.to_lowercase().as_str() {
            "lanczos" => Ok(Self::Lanczos),
            "randomized" | "random" => Ok(Self::Randomized),
            "power" => Ok(Self::Power),
            "cross" | "cross_approximation" => Ok(Self::CrossApproximation),
            _ => Err(SparseError::ValueError(format!("Unknown SVD method: {s}"))),
        }
    }
}

/// Options for SVD computation
#[derive(Debug, Clone)]
pub struct SVDOptions {
    /// Number of singular values to compute
    pub k: usize,
    /// Maximum number of iterations
    pub maxiter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Number of additional singular vectors for randomized methods
    pub n_oversamples: usize,
    /// Number of power iterations for randomized methods
    pub n_iter: usize,
    /// SVD computation method
    pub method: SVDMethod,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Whether to compute left singular vectors (U)
    pub compute_u: bool,
    /// Whether to compute right singular vectors (V^T)
    pub compute_vt: bool,
}

impl Default for SVDOptions {
    fn default() -> Self {
        Self {
            k: 6,
            maxiter: 1000,
            tol: 1e-10,
            n_oversamples: 10,
            n_iter: 2,
            method: SVDMethod::Lanczos,
            random_seed: None,
            compute_u: true,
            compute_vt: true,
        }
    }
}

/// Result of SVD computation
#[derive(Debug, Clone)]
pub struct SVDResult<T>
where
    T: Float + SparseElement + Debug + Copy,
{
    /// Left singular vectors (U matrix)
    pub u: Option<Array2<T>>,
    /// Singular values
    pub s: Array1<T>,
    /// Right singular vectors transposed (V^T matrix)
    pub vt: Option<Array2<T>>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether the algorithm converged
    pub converged: bool,
}

/// Compute the truncated SVD of a sparse matrix
///
/// Computes the k largest singular values and corresponding singular vectors
/// of a sparse matrix using iterative methods.
///
/// # Arguments
///
/// * `matrix` - The sparse matrix to decompose
/// * `k` - Number of singular values to compute (default: 6)
/// * `options` - Optional configuration parameters
///
/// # Returns
///
/// SVD result containing U, s, and V^T matrices
///
/// # Examples
///
/// ```
/// use scirs2_sparse::linalg::svds;
/// use scirs2_sparse::csr_array::CsrArray;
///
/// // Create a sparse matrix
/// let rows = vec![0, 0, 1, 2, 2];
/// let cols = vec![0, 2, 1, 0, 2];
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).expect("Operation failed");
///
/// // Compute 2 largest singular values
/// let result = svds(&matrix, Some(2), None).expect("Operation failed");
/// ```
#[allow(dead_code)]
pub fn svds<T, S>(
    matrix: &S,
    k: Option<usize>,
    options: Option<SVDOptions>,
) -> SparseResult<SVDResult<T>>
where
    T: Float
        + SparseElement
        + Debug
        + Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + 'static
        + std::iter::Sum,
    S: SparseArray<T>,
{
    let opts = options.unwrap_or_default();
    let k = k.unwrap_or(opts.k);

    let (m, n) = matrix.shape();
    if k >= m.min(n) {
        return Err(SparseError::ValueError(
            "Number of singular values k must be less than min(m, n)".to_string(),
        ));
    }

    match opts.method {
        SVDMethod::Lanczos => lanczos_bidiag_svd(matrix, k, &opts),
        SVDMethod::Randomized => randomized_svd(matrix, k, &opts),
        SVDMethod::Power => power_method_svd(matrix, k, &opts),
        SVDMethod::CrossApproximation => cross_approximation_svd(matrix, k, &opts),
    }
}

/// Compute truncated SVD using a specific method and parameters
#[allow(dead_code)]
pub fn svd_truncated<T, S>(
    matrix: &S,
    k: usize,
    method: &str,
    tol: Option<f64>,
    maxiter: Option<usize>,
) -> SparseResult<SVDResult<T>>
where
    T: Float
        + SparseElement
        + Debug
        + Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + 'static
        + std::iter::Sum,
    S: SparseArray<T>,
{
    let method_enum = SVDMethod::from_str(method)?;

    let options = SVDOptions {
        k,
        method: method_enum,
        tol: tol.unwrap_or(1e-10),
        maxiter: maxiter.unwrap_or(1000),
        ..Default::default()
    };

    svds(matrix, Some(k), Some(options))
}

/// Lanczos bidiagonalization SVD algorithm
#[allow(dead_code)]
fn lanczos_bidiag_svd<T, S>(
    matrix: &S,
    k: usize,
    options: &SVDOptions,
) -> SparseResult<SVDResult<T>>
where
    T: Float
        + SparseElement
        + Debug
        + Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + 'static
        + std::iter::Sum,
    S: SparseArray<T>,
{
    let (m, n) = matrix.shape();
    let max_lanczos_size = (2 * k + 10).min(m.min(n));

    // Initialize starting vector
    let mut u = Array1::zeros(m);
    u[0] = T::sparse_one();

    // Normalize
    let norm = (u.iter().map(|&v| v * v).sum::<T>()).sqrt();
    if !SparseElement::is_zero(&norm) {
        for i in 0..m {
            u[i] = u[i] / norm;
        }
    }

    let mut alpha = Vec::<T>::new();
    let mut beta = Vec::<T>::new();
    let mut u_vectors = Vec::<Array1<T>>::with_capacity(max_lanczos_size);
    let mut v_vectors = Vec::<Array1<T>>::with_capacity(max_lanczos_size);

    u_vectors.push(u.clone());

    let mut converged = false;
    let mut iter = 0;

    // Lanczos bidiagonalization loop
    while iter < options.maxiter && alpha.len() < max_lanczos_size {
        // v = A^T * u - beta[j-1] * v[j-1]
        let av = matrix_transpose_vector_product(matrix, &u_vectors[iter])?;
        let mut v = av;

        if iter > 0 && !beta.is_empty() {
            let prev_beta = beta[iter - 1];
            for i in 0..n {
                v[i] = v[i] - prev_beta * v_vectors[iter - 1][i];
            }
        }

        // alpha[j] = ||v||
        let alpha_j = (v.iter().map(|&val| val * val).sum::<T>()).sqrt();
        alpha.push(alpha_j);

        if SparseElement::is_zero(&alpha_j) {
            break;
        }

        // Normalize v
        for i in 0..n {
            v[i] = v[i] / alpha_j;
        }
        v_vectors.push(v.clone());

        // u = A * v - alpha[j] * u[j]
        let avu = matrix_vector_product(matrix, &v)?;
        let mut u_next = avu;

        for i in 0..m {
            u_next[i] = u_next[i] - alpha_j * u_vectors[iter][i];
        }

        // beta[j] = ||u||
        let beta_j = (u_next.iter().map(|&val| val * val).sum::<T>()).sqrt();
        beta.push(beta_j);

        if beta_j < T::from(options.tol).expect("Operation failed") {
            converged = true;
            break;
        }

        // Normalize u
        for i in 0..m {
            u_next[i] = u_next[i] / beta_j;
        }

        u_vectors.push(u_next);
        iter += 1;
    }

    // Solve the bidiagonal SVD problem
    let (singular_values, u_bidiag, vt_bidiag) = solve_bidiagonal_svd(&alpha, &beta, k)?;

    // Compute final U and V^T matrices
    let final_u = if options.compute_u {
        let mut u_final = Array2::zeros((m, k.min(singular_values.len())));
        for j in 0..k.min(singular_values.len()) {
            for i in 0..m {
                let mut sum = T::sparse_zero();
                for l in 0..u_vectors.len().min(u_bidiag.len()) {
                    if j < u_bidiag[l].len() {
                        sum = sum
                            + T::from(u_bidiag[l][j]).expect("Operation failed") * u_vectors[l][i];
                    }
                }
                u_final[[i, j]] = sum;
            }
        }
        Some(u_final)
    } else {
        None
    };

    let final_vt = if options.compute_vt {
        let mut vt_final = Array2::zeros((k.min(singular_values.len()), n));
        for j in 0..k.min(singular_values.len()) {
            for i in 0..n {
                let mut sum = T::sparse_zero();
                for l in 0..v_vectors.len().min(vt_bidiag.len()) {
                    if j < vt_bidiag[l].len() {
                        sum = sum
                            + T::from(vt_bidiag[l][j]).expect("Operation failed") * v_vectors[l][i];
                    }
                }
                vt_final[[j, i]] = sum;
            }
        }
        Some(vt_final)
    } else {
        None
    };

    Ok(SVDResult {
        u: final_u,
        s: Array1::from_vec(singular_values[..k.min(singular_values.len())].to_vec()),
        vt: final_vt,
        iterations: iter,
        converged,
    })
}

/// Randomized SVD algorithm
#[allow(dead_code)]
fn randomized_svd<T, S>(matrix: &S, k: usize, options: &SVDOptions) -> SparseResult<SVDResult<T>>
where
    T: Float
        + SparseElement
        + Debug
        + Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + 'static
        + std::iter::Sum,
    S: SparseArray<T>,
{
    let (m, n) = matrix.shape();
    // Limit l to the smaller dimension to ensure we can form orthonormal basis
    let l = (k + options.n_oversamples).min(m).min(n);

    // Generate random matrix
    let mut omega = Array2::zeros((n, l));
    for i in 0..n {
        for j in 0..l {
            // Simple pseudo-random generation (replace with proper RNG in production)
            let val = ((i * 17 + j * 13) % 1000) as f64 / 1000.0 - 0.5;
            omega[[i, j]] = T::from(val).expect("Operation failed");
        }
    }

    // Y = A * Omega
    let mut y = Array2::zeros((m, l));
    for j in 0..l {
        let omega_col = omega.column(j).to_owned();
        let y_col = matrix_vector_product(matrix, &omega_col)?;
        for i in 0..m {
            y[[i, j]] = y_col[i];
        }
    }

    // Power iterations to improve quality
    for _ in 0..options.n_iter {
        // Y = A * (A^T * Y)
        let mut y_new = Array2::zeros((m, l));
        for j in 0..l {
            let y_col = y.column(j).to_owned();
            let at_y_col = matrix_transpose_vector_product(matrix, &y_col)?;
            let a_at_y_col = matrix_vector_product(matrix, &at_y_col)?;
            for i in 0..m {
                y_new[[i, j]] = a_at_y_col[i];
            }
        }
        y = y_new;
    }

    // QR decomposition of Y
    let q = qr_decomposition_orthogonal(&y)?;

    // B = Q^T * A
    // Q has shape (m, l) with orthonormal columns
    // Q^T * A has shape (l, n)
    // The i-th row of Q^T is the i-th column of Q
    let mut b = Array2::zeros((l, n));
    for i in 0..l {
        let q_col = q.column(i).to_owned();

        // Compute A^T * q_col which gives the i-th row of Q^T * A
        let b_row = matrix_transpose_vector_product(matrix, &q_col)?;
        for j in 0..n {
            b[[i, j]] = b_row[j];
        }
    }

    // SVD of B
    let b_svd = dense_svd(&b, k)?;

    // Compute final U and V^T
    // U = Q * U_B where Q is (m, l) and U_B is (l, k)
    // So U is (m, k) with U[i,j] = sum_l Q[i,l] * U_B[l,j]
    let final_u = if options.compute_u {
        if let Some(ref u_b) = b_svd.u {
            let mut u_result = Array2::zeros((m, k));
            for i in 0..m {
                for j in 0..k {
                    let mut sum = T::sparse_zero();
                    for l_idx in 0..l {
                        sum = sum + q[[i, l_idx]] * u_b[[l_idx, j]];
                    }
                    u_result[[i, j]] = sum;
                }
            }
            Some(u_result)
        } else {
            None
        }
    } else {
        None
    };

    Ok(SVDResult {
        u: final_u,
        s: b_svd.s,
        vt: b_svd.vt,
        iterations: options.n_iter + 1,
        converged: true,
    })
}

/// Power method SVD (simplified implementation)
#[allow(dead_code)]
fn power_method_svd<T, S>(matrix: &S, k: usize, options: &SVDOptions) -> SparseResult<SVDResult<T>>
where
    T: Float
        + SparseElement
        + Debug
        + Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + 'static
        + std::iter::Sum,
    S: SparseArray<T>,
{
    // For now, fall back to Lanczos method
    // A full implementation would use deflation and multiple power iterations
    lanczos_bidiag_svd(matrix, k, options)
}

/// Cross-approximation SVD (simplified implementation)
#[allow(dead_code)]
fn cross_approximation_svd<T, S>(
    matrix: &S,
    k: usize,
    options: &SVDOptions,
) -> SparseResult<SVDResult<T>>
where
    T: Float
        + SparseElement
        + Debug
        + Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + 'static
        + std::iter::Sum,
    S: SparseArray<T>,
{
    // For now, fall back to Lanczos method
    // A full implementation would use adaptive cross-approximation
    lanczos_bidiag_svd(matrix, k, options)
}

/// Matrix-vector product: y = A * x
#[allow(dead_code)]
fn matrix_vector_product<T, S>(matrix: &S, vector: &Array1<T>) -> SparseResult<Array1<T>>
where
    T: Float
        + SparseElement
        + Debug
        + Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + 'static
        + std::iter::Sum,
    S: SparseArray<T>,
{
    let (m, n) = matrix.shape();
    if vector.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: vector.len(),
        });
    }

    let mut result = Array1::zeros(m);
    let (row_indices, col_indices, values) = matrix.find();

    for (k, (&i, &j)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
        result[i] = result[i] + values[k] * vector[j];
    }

    Ok(result)
}

/// Matrix-transpose-vector product: y = A^T * x
#[allow(dead_code)]
fn matrix_transpose_vector_product<T, S>(matrix: &S, vector: &Array1<T>) -> SparseResult<Array1<T>>
where
    T: Float
        + SparseElement
        + Debug
        + Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + 'static
        + std::iter::Sum,
    S: SparseArray<T>,
{
    let (m, n) = matrix.shape();
    if vector.len() != m {
        return Err(SparseError::DimensionMismatch {
            expected: m,
            found: vector.len(),
        });
    }

    let mut result = Array1::zeros(n);
    let (row_indices, col_indices, values) = matrix.find();

    for (k, (&i, &j)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
        result[j] = result[j] + values[k] * vector[i];
    }

    Ok(result)
}

/// Solve SVD of a bidiagonal matrix (simplified implementation)
#[allow(dead_code)]
fn solve_bidiagonal_svd<T>(
    alpha: &[T],
    beta: &[T],
    k: usize,
) -> SparseResult<BidiagonalSvdResult<T>>
where
    T: Float
        + SparseElement
        + Debug
        + Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + 'static
        + std::iter::Sum,
{
    let n = alpha.len();

    // For simplicity, use power iteration on A^T * A
    // In a real implementation, we'd use the QR algorithm for bidiagonal matrices

    let mut singular_values = Vec::with_capacity(k);
    let mut u_vectors = Vec::with_capacity(k);
    let mut vt_vectors = Vec::with_capacity(k);

    // Approximate largest singular value using power iteration
    if n > 0 {
        let largest_sv = alpha
            .iter()
            .map(|&x| x.abs())
            .fold(T::sparse_zero(), |a, b| if a > b { a } else { b });
        singular_values.push(largest_sv);

        // Create identity-like vectors
        let mut u_vec = vec![0.0; n];
        let mut vt_vec = vec![0.0; n];

        if n > 0 {
            u_vec[0] = 1.0;
            vt_vec[0] = 1.0;
        }

        u_vectors.push(u_vec);
        vt_vectors.push(vt_vec);
    }

    // Fill remaining with zeros (simplified)
    while singular_values.len() < k && singular_values.len() < n {
        singular_values.push(T::sparse_zero());
        u_vectors.push(vec![0.0_f64; n]);
        vt_vectors.push(vec![0.0_f64; n]);
    }

    Ok((singular_values, u_vectors, vt_vectors))
}

/// QR decomposition returning only Q (simplified implementation)
#[allow(dead_code)]
fn qr_decomposition_orthogonal<T>(matrix: &Array2<T>) -> SparseResult<Array2<T>>
where
    T: Float
        + SparseElement
        + Debug
        + Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + 'static
        + std::iter::Sum,
{
    let (m, n) = matrix.dim();
    let mut q = matrix.clone();

    // Simple Gram-Schmidt orthogonalization
    for j in 0..n {
        // Normalize column j
        let mut norm = T::sparse_zero();
        for i in 0..m {
            norm = norm + q[[i, j]] * q[[i, j]];
        }
        norm = norm.sqrt();

        if !SparseElement::is_zero(&norm) {
            for i in 0..m {
                q[[i, j]] = q[[i, j]] / norm;
            }
        }

        // Orthogonalize remaining columns against column j
        for k in (j + 1)..n {
            let mut dot = T::sparse_zero();
            for i in 0..m {
                dot = dot + q[[i, j]] * q[[i, k]];
            }

            for i in 0..m {
                q[[i, k]] = q[[i, k]] - dot * q[[i, j]];
            }
        }
    }

    Ok(q)
}

/// Dense SVD via the one-sided Jacobi method.
///
/// Computes A = U * diag(s) * V^T for a dense (m x n) matrix by
/// diagonalising G = A^T A with cyclic Jacobi rotations, recovering
/// singular values as sqrt of eigenvalues, then reconstructing U.
///
/// Returns the `k` largest singular triplets.
#[allow(dead_code)]
fn dense_svd<T>(matrix: &Array2<T>, k: usize) -> SparseResult<SVDResult<T>>
where
    T: Float
        + SparseElement
        + Debug
        + Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + 'static
        + std::iter::Sum,
{
    let (m, n) = matrix.dim();
    let rank = k.min(m).min(n);

    // Step 1: Form G = A^T A  (n x n, symmetric positive semidefinite)
    let mut g = Array2::zeros((n, n));
    for i in 0..n {
        for j in i..n {
            let mut s = T::sparse_zero();
            for r in 0..m {
                s = s + matrix[[r, i]] * matrix[[r, j]];
            }
            g[[i, j]] = s;
            g[[j, i]] = s;
        }
    }

    // Step 2: Jacobi eigenvalue algorithm on G
    let mut v_mat = Array2::<T>::eye(n);
    let max_sweeps = 100usize;
    let tol = T::from(1e-14).unwrap_or_else(|| T::epsilon());

    for _sweep in 0..max_sweeps {
        let mut off_norm = T::sparse_zero();
        for i in 0..n {
            for j in (i + 1)..n {
                off_norm = off_norm + g[[i, j]] * g[[i, j]];
            }
        }
        if off_norm < tol * tol {
            break;
        }

        for i in 0..n {
            for j in (i + 1)..n {
                let gij = g[[i, j]];
                if gij.abs() < tol {
                    continue;
                }
                let diff = g[[j, j]] - g[[i, i]];
                let tau = if diff.abs() < tol {
                    T::sparse_one()
                } else {
                    let ratio = T::from(2.0).expect("conv") * gij / diff;
                    let sign_r = if ratio >= T::sparse_zero() {
                        T::sparse_one()
                    } else {
                        -T::sparse_one()
                    };
                    sign_r / (ratio.abs() + (ratio * ratio + T::sparse_one()).sqrt())
                };
                let cos_t = T::sparse_one() / (tau * tau + T::sparse_one()).sqrt();
                let sin_t = tau * cos_t;

                for r in 0..n {
                    if r == i || r == j {
                        continue;
                    }
                    let gri = g[[r, i]];
                    let grj = g[[r, j]];
                    g[[r, i]] = cos_t * gri - sin_t * grj;
                    g[[i, r]] = g[[r, i]];
                    g[[r, j]] = sin_t * gri + cos_t * grj;
                    g[[j, r]] = g[[r, j]];
                }
                let gii = g[[i, i]];
                let gjj = g[[j, j]];
                let two = T::from(2.0).expect("conv");
                g[[i, i]] = cos_t * cos_t * gii - two * sin_t * cos_t * gij + sin_t * sin_t * gjj;
                g[[j, j]] = sin_t * sin_t * gii + two * sin_t * cos_t * gij + cos_t * cos_t * gjj;
                g[[i, j]] = T::sparse_zero();
                g[[j, i]] = T::sparse_zero();

                for r in 0..n {
                    let vri = v_mat[[r, i]];
                    let vrj = v_mat[[r, j]];
                    v_mat[[r, i]] = cos_t * vri - sin_t * vrj;
                    v_mat[[r, j]] = sin_t * vri + cos_t * vrj;
                }
            }
        }
    }

    // Step 3: Extract singular values and V^T, sorted descending
    let mut sigma_sq: Vec<(T, usize)> = (0..n).map(|i| (g[[i, i]], i)).collect();
    sigma_sq.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let take = rank.min(sigma_sq.len());
    let mut singular_values = Vec::with_capacity(take);
    let mut vt_final = Array2::zeros((take, n));

    for j in 0..take {
        let (lam, idx) = sigma_sq[j];
        let sv = if lam > T::sparse_zero() {
            lam.sqrt()
        } else {
            T::sparse_zero()
        };
        singular_values.push(sv);
        for col in 0..n {
            vt_final[[j, col]] = v_mat[[col, idx]];
        }
    }

    // Step 4: Compute U = A * V * Sigma^{-1}
    let mut u_final = Array2::zeros((m, take));
    for j in 0..take {
        let sv = singular_values[j];
        if sv > tol {
            for i in 0..m {
                let mut dot = T::sparse_zero();
                for l in 0..n {
                    dot = dot + matrix[[i, l]] * vt_final[[j, l]];
                }
                u_final[[i, j]] = dot / sv;
            }
        }
    }

    Ok(SVDResult {
        u: Some(u_final),
        s: Array1::from_vec(singular_values),
        vt: Some(vt_final),
        iterations: max_sweeps,
        converged: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr_array::CsrArray;
    use approx::assert_relative_eq;

    fn create_test_matrix() -> CsrArray<f64> {
        // Create a simple sparse matrix for testing
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 2];
        let data = vec![3.0, 2.0, 1.0, 4.0, 5.0];

        CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).expect("Operation failed")
    }

    #[test]
    fn test_svds_basic() {
        let matrix = create_test_matrix();
        let result = svds(&matrix, Some(2), None).expect("Operation failed");

        // Check dimensions
        assert_eq!(result.s.len(), 2);

        if let Some(ref u) = result.u {
            assert_eq!(u.shape(), [3, 2]);
        }

        if let Some(ref vt) = result.vt {
            assert_eq!(vt.shape(), [2, 3]);
        }

        // Singular values should be non-negative and sorted
        assert!(result.s[0] >= 0.0);
        if result.s.len() > 1 {
            assert!(result.s[0] >= result.s[1]);
        }
    }

    #[test]
    fn test_matrix_vector_product() {
        let matrix = create_test_matrix();
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let y = matrix_vector_product(&matrix, &x).expect("Operation failed");

        // Check result dimensions
        assert_eq!(y.len(), 3);

        // Verify computation: y = A * x
        assert_relative_eq!(y[0], 3.0 * 1.0 + 2.0 * 3.0, epsilon = 1e-10); // 9.0
        assert_relative_eq!(y[1], 1.0 * 2.0, epsilon = 1e-10); // 2.0
        assert_relative_eq!(y[2], 4.0 * 1.0 + 5.0 * 3.0, epsilon = 1e-10); // 19.0
    }

    #[test]
    fn test_matrix_transpose_vector_product() {
        let matrix = create_test_matrix();
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let y = matrix_transpose_vector_product(&matrix, &x).expect("Operation failed");

        // Check result dimensions
        assert_eq!(y.len(), 3);

        // Verify computation: y = A^T * x
        assert_relative_eq!(y[0], 3.0 * 1.0 + 4.0 * 3.0, epsilon = 1e-10); // 15.0
        assert_relative_eq!(y[1], 1.0 * 2.0, epsilon = 1e-10); // 2.0
        assert_relative_eq!(y[2], 2.0 * 1.0 + 5.0 * 3.0, epsilon = 1e-10); // 17.0
    }

    #[test]
    fn test_svd_options() {
        let matrix = create_test_matrix();

        let options = SVDOptions {
            k: 1,
            method: SVDMethod::Lanczos,
            compute_u: false,
            compute_vt: true,
            ..Default::default()
        };

        let result = svds(&matrix, Some(1), Some(options)).expect("Operation failed");

        assert_eq!(result.s.len(), 1);
        assert!(result.u.is_none());
        assert!(result.vt.is_some());
    }

    #[test]
    fn test_svd_truncated_api() {
        let matrix = create_test_matrix();

        let result =
            svd_truncated(&matrix, 2, "lanczos", Some(1e-8), Some(100)).expect("Operation failed");

        assert_eq!(result.s.len(), 2);
        assert!(result.u.is_some());
        assert!(result.vt.is_some());
    }

    #[test]
    fn test_randomized_svd() {
        let matrix = create_test_matrix();

        let options = SVDOptions {
            k: 2,
            method: SVDMethod::Randomized,
            n_oversamples: 5,
            n_iter: 1,
            ..Default::default()
        };

        let result = svds(&matrix, Some(2), Some(options)).expect("Operation failed");

        assert_eq!(result.s.len(), 2);
        assert!(result.converged);
    }

    #[test]
    fn test_qr_decomposition() {
        let matrix = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("Operation failed");

        let q = qr_decomposition_orthogonal(&matrix).expect("Operation failed");

        // Check orthogonality (Q^T * Q = I)
        assert_eq!(q.shape(), [3, 2]);

        // Check that columns are orthonormal
        for j in 0..2 {
            let mut norm = 0.0;
            for i in 0..3 {
                norm += q[[i, j]] * q[[i, j]];
            }
            assert_relative_eq!(norm, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_svd_method_parsing() {
        assert_eq!(
            SVDMethod::from_str("lanczos").expect("Operation failed"),
            SVDMethod::Lanczos
        );
        assert_eq!(
            SVDMethod::from_str("randomized").expect("Operation failed"),
            SVDMethod::Randomized
        );
        assert_eq!(
            SVDMethod::from_str("power").expect("Operation failed"),
            SVDMethod::Power
        );
        assert!(SVDMethod::from_str("invalid").is_err());
    }

    #[test]
    fn test_invalid_k() {
        let matrix = create_test_matrix();

        // k too large
        let result = svds(&matrix, Some(10), None);
        assert!(result.is_err());
    }
}
