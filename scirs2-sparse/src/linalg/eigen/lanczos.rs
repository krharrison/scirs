//! Lanczos algorithm for sparse matrix eigenvalue computation
//!
//! This module implements the Lanczos algorithm for finding eigenvalues and
//! eigenvectors of large symmetric sparse matrices.

use crate::error::{SparseError, SparseResult};
use crate::sym_csr::SymCsrMatrix;
use crate::sym_ops::sym_csr_matvec;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::Float;
use scirs2_core::SparseElement;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

/// Configuration options for the Lanczos algorithm
#[derive(Debug, Clone)]
pub struct LanczosOptions {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Maximum dimension of the Krylov subspace
    pub max_subspace_size: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Number of eigenvalues to compute
    pub numeigenvalues: usize,
    /// Whether to compute eigenvectors
    pub compute_eigenvectors: bool,
}

impl Default for LanczosOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            max_subspace_size: 20,
            tol: 1e-8,
            numeigenvalues: 1,
            compute_eigenvectors: true,
        }
    }
}

/// Result of an eigenvalue computation
#[derive(Debug, Clone)]
pub struct EigenResult<T>
where
    T: Float + Debug + Copy,
{
    /// Converged eigenvalues
    pub eigenvalues: Array1<T>,
    /// Corresponding eigenvectors (if requested)
    pub eigenvectors: Option<Array2<T>>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Residual norms for each eigenpair
    pub residuals: Array1<T>,
    /// Whether the algorithm converged
    pub converged: bool,
}

/// Computes the extreme eigenvalues and corresponding eigenvectors of a symmetric
/// matrix using the Lanczos algorithm.
///
/// # Arguments
///
/// * `matrix` - The symmetric matrix
/// * `options` - Configuration options
/// * `initial_guess` - Initial guess for the first Lanczos vector (optional)
///
/// # Returns
///
/// Result containing eigenvalues and eigenvectors
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use scirs2_sparse::{
///     sym_csr::SymCsrMatrix,
///     linalg::{lanczos, LanczosOptions},
/// };
///
/// // Create a symmetric matrix
/// let data = vec![2.0, 1.0, 2.0, 3.0];
/// let indices = vec![0, 0, 1, 2];
/// let indptr = vec![0, 1, 3, 4];
/// let matrix = SymCsrMatrix::new(data, indptr, indices, (3, 3)).expect("Operation failed");
///
/// // Configure options
/// let options = LanczosOptions {
///     max_iter: 100,
///     max_subspace_size: 3, // Matrix is 3x3
///     tol: 1e-8,
///     numeigenvalues: 1,   // Find the largest eigenvalue
///     compute_eigenvectors: true,
/// };
///
/// // Compute eigenvalues and eigenvectors
/// let result = lanczos(&matrix, &options, None).expect("Operation failed");
///
/// // Check the result
/// println!("Eigenvalues: {:?}", result.eigenvalues);
/// println!("Converged in {} iterations", result.iterations);
/// println!("Final residuals: {:?}", result.residuals);
/// assert!(result.converged);
/// ```
#[allow(unused_assignments)]
#[allow(dead_code)]
pub fn lanczos<T>(
    matrix: &SymCsrMatrix<T>,
    options: &LanczosOptions,
    initial_guess: Option<ArrayView1<T>>,
) -> SparseResult<EigenResult<T>>
where
    T: Float
        + SparseElement
        + Debug
        + Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + std::iter::Sum
        + scirs2_core::simd_ops::SimdUnifiedOps
        + Send
        + Sync
        + 'static,
{
    let (n, _) = matrix.shape();

    // Ensure the subspace size is valid
    let subspace_size = options.max_subspace_size.min(n);

    // Ensure the number of eigenvalues requested is valid
    let numeigenvalues = options.numeigenvalues.min(subspace_size);

    // Initialize the first Lanczos vector
    let mut v = match initial_guess {
        Some(v) => {
            if v.len() != n {
                return Err(SparseError::DimensionMismatch {
                    expected: n,
                    found: v.len(),
                });
            }
            // Create a copy of the initial guess
            let mut v_arr = Array1::zeros(n);
            for i in 0..n {
                v_arr[i] = v[i];
            }
            v_arr
        }
        None => {
            // Random initialization
            let mut v_arr = Array1::zeros(n);
            v_arr[0] = T::sparse_one(); // Simple initialization with [1, 0, 0, ...]
            v_arr
        }
    };

    // Normalize the initial vector
    let norm = (v.iter().map(|&val| val * val).sum::<T>()).sqrt();
    if !SparseElement::is_zero(&norm) {
        for i in 0..n {
            v[i] = v[i] / norm;
        }
    }

    // Allocate space for Lanczos vectors
    let mut v_vectors = Vec::with_capacity(subspace_size);
    v_vectors.push(v.clone());

    // Allocate space for tridiagonal matrix elements
    let mut alpha = Vec::<T>::with_capacity(subspace_size); // Diagonal elements
    let mut beta = Vec::<T>::with_capacity(subspace_size - 1); // Off-diagonal elements

    // First iteration step
    let mut w = sym_csr_matvec(matrix, &v.view())?;
    let alpha_j = v.iter().zip(w.iter()).map(|(&vi, &wi)| vi * wi).sum::<T>();
    alpha.push(alpha_j);

    // Orthogonalize against previous vectors
    for i in 0..n {
        w[i] = w[i] - alpha_j * v[i];
    }

    // Compute beta (norm of w)
    let mut beta_j = (w.iter().map(|&val| val * val).sum::<T>()).sqrt();

    let mut iter = 1;
    let mut converged = false;

    while iter < options.max_iter && alpha.len() < subspace_size {
        if SparseElement::is_zero(&beta_j) {
            // Lucky breakdown - exact invariant subspace found
            break;
        }

        beta.push(beta_j);

        // Next Lanczos vector
        let mut v_next = Array1::zeros(n);
        for i in 0..n {
            v_next[i] = w[i] / beta_j;
        }

        // Store the vector
        v_vectors.push(v_next.clone());

        // Next iteration step: w = A * v_next
        w = sym_csr_matvec(matrix, &v_next.view())?;

        // Compute alpha BEFORE orthogonalization (standard 3-term recurrence)
        let alpha_j = v_next
            .iter()
            .zip(w.iter())
            .map(|(&vi, &wi)| vi * wi)
            .sum::<T>();
        alpha.push(alpha_j);

        // Subtract the standard 3-term recurrence components:
        // w = w - alpha_j * v_next - beta_j * v_prev
        // (beta_j * v_prev is handled by full reorthogonalization below)
        for i in 0..n {
            w[i] = w[i] - alpha_j * v_next[i];
        }

        // Full reorthogonalization for numerical stability:
        // Remove any residual components along all previous Lanczos vectors
        for v_j in v_vectors.iter() {
            let proj = v_j
                .iter()
                .zip(w.iter())
                .map(|(&vj, &wi)| vj * wi)
                .sum::<T>();
            for i in 0..n {
                w[i] = w[i] - proj * v_j[i];
            }
        }

        // Compute beta for next iteration
        let beta_j_next = (w.iter().map(|&val| val * val).sum::<T>()).sqrt();

        // Check for convergence using the largest eigenvalue approx
        if alpha.len() >= numeigenvalues {
            // Build and solve the tridiagonal system
            if let Ok((eigvals, _)) = solve_tridiagonal_eigenproblem(&alpha, &beta, numeigenvalues)
            {
                // Check if the largest eigvals have converged (using beta as an error estimate)
                if !eigvals.is_empty()
                    && eigvals[0].abs() > T::sparse_zero()
                    && beta_j_next < T::from(options.tol).unwrap_or(T::epsilon()) * eigvals[0].abs()
                {
                    converged = true;
                    break;
                }
            }
        }

        v = v_next;
        iter += 1;

        // Update beta_j for the next iteration
        beta_j = beta_j_next;
    }

    // Solve the final tridiagonal eigenproblem for ALL eigenvalues so that
    // callers (e.g. process_eigenvalue_selection) can pick any subset (LA, SA, LM, SM).
    let n_tridiag = alpha.len();
    let (eigvals, eigvecs) = solve_tridiagonal_eigenproblem(&alpha, &beta, n_tridiag)?;

    // Compute the Ritz vectors (eigenvectors in the original space) if requested
    let actual_neig = eigvals.len();
    let eigenvectors = if options.compute_eigenvectors {
        let mut ritz_vectors = Array2::zeros((n, actual_neig));

        for k in 0..actual_neig {
            for i in 0..n {
                let mut sum = T::sparse_zero();
                for j in 0..v_vectors.len() {
                    if j < eigvecs.len() && k < eigvecs[j].len() {
                        sum = sum + eigvecs[j][k] * v_vectors[j][i];
                    }
                }
                ritz_vectors[[i, k]] = sum;
            }
        }

        Some(ritz_vectors)
    } else {
        None
    };

    // Compute residuals
    let actualeigenvalues = eigvals.len();
    let mut residuals = Array1::zeros(actualeigenvalues);
    if let Some(ref evecs) = eigenvectors {
        for k in 0..actualeigenvalues {
            let mut evec = Array1::zeros(n);
            for i in 0..n {
                evec[i] = evecs[[i, k]];
            }

            let ax = sym_csr_matvec(matrix, &evec.view())?;

            let mut res = Array1::zeros(n);
            for i in 0..n {
                res[i] = ax[i] - eigvals[k] * evec[i];
            }

            residuals[k] = (res.iter().map(|&v| v * v).sum::<T>()).sqrt();
        }
    } else {
        // If no eigenvectors were computed, use the Kaniel-Paige error bound
        // (beta_j * last component of eigenvector in the Krylov basis)
        for k in 0..actualeigenvalues {
            if k < eigvecs.len() && !beta.is_empty() {
                residuals[k] = beta[beta.len() - 1] * eigvecs[eigvecs.len() - 1][k].abs();
            }
        }
    }

    // Create the result
    let result = EigenResult {
        eigenvalues: Array1::from_vec(eigvals),
        eigenvectors,
        iterations: iter,
        residuals,
        converged,
    };

    Ok(result)
}

/// Solves a symmetric tridiagonal eigenvalue problem.
///
/// This function computes the eigenvalues and eigenvectors of a symmetric
/// tridiagonal matrix defined by its diagonal elements `alpha` and
/// off-diagonal elements `beta`.
///
/// # Arguments
///
/// * `alpha` - Diagonal elements
/// * `beta` - Off-diagonal elements
/// * `numeigenvalues` - Number of eigenvalues to compute
///
/// # Returns
///
/// A tuple containing:
/// - The eigenvalues in descending order
/// - The corresponding eigenvectors
#[allow(dead_code)]
fn solve_tridiagonal_eigenproblem<T>(
    alpha: &[T],
    beta: &[T],
    numeigenvalues: usize,
) -> SparseResult<(Vec<T>, Vec<Vec<T>>)>
where
    T: Float
        + SparseElement
        + Debug
        + Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
{
    let n = alpha.len();
    if n == 0 {
        return Err(SparseError::ValueError(
            "Empty tridiagonal matrix".to_string(),
        ));
    }

    if beta.len() != n - 1 {
        return Err(SparseError::DimensionMismatch {
            expected: n - 1,
            found: beta.len(),
        });
    }

    // For small matrices, use a simple algorithm for all eigenvalues
    if n <= 3 {
        return solve_small_tridiagonal(alpha, beta, numeigenvalues);
    }

    // For larger matrices, use the QL algorithm with implicit shifts
    // This is a simplified implementation and could be optimized further

    // Clone the diagonal and off-diagonal elements
    let mut d = alpha.to_vec();
    let mut e = beta.to_vec();
    e.push(T::sparse_zero()); // Add a zero at the end

    // Allocate space for eigenvectors
    let mut z = vec![vec![T::sparse_zero(); n]; n];
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        z[i][i] = T::sparse_one(); // Initialize with identity matrix
    }

    // QL algorithm with implicit shifts (tqli from Numerical Recipes)
    //
    // d[0..n] = diagonal,  e[0..n] = off-diagonal (e[n-1] unused).
    // On exit d[] holds eigenvalues and z[][] holds eigenvectors.
    let max_ql_iter: usize = 200; // generous iteration budget per eigenvalue

    for l in 0..n {
        let mut iter_count: usize = 0;

        loop {
            // Find the smallest m >= l such that |e[m]| is negligible
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                // Use a relative + absolute tolerance
                if e[m].abs() <= T::from(2.0e-15).unwrap_or(T::epsilon()) * dd {
                    break;
                }
                m += 1;
            }

            if m == l {
                break; // eigenvalue d[l] has converged
            }

            if iter_count >= max_ql_iter {
                return Err(SparseError::IterativeSolverFailure(
                    "QL algorithm did not converge".to_string(),
                ));
            }
            iter_count += 1;

            // Form the implicit QL shift
            let half = T::from(0.5).unwrap_or(T::sparse_one());
            let mut g = (d[l + 1] - d[l]) / (e[l] + e[l]); // = (d[l+1]-d[l])/(2*e[l])
            let mut r = (g * g + T::sparse_one()).sqrt();
            // g = d[m] - d[l] + e[l] / (g + sign(g)*r)
            g = d[m] - d[l] + e[l] / (g + if g >= T::sparse_zero() { r } else { -r });

            let mut s = T::sparse_one();
            let mut c = T::sparse_one();
            let mut p = T::sparse_zero();

            // Chase the bulge from m-1 down to l
            let mut broke_early = false;
            let mut ii = m; // we will iterate i = m-1, m-2, ..., l
            while ii > l {
                let i = ii - 1;

                let f = s * e[i];
                let b = c * e[i];

                // Givens rotation to zero out f
                r = (f * f + g * g).sqrt();
                e[i + 1] = r;

                if r < T::from(1e-30).unwrap_or(T::epsilon()) {
                    // Underflow recovery: undo the accumulated shift
                    d[i + 1] = d[i + 1] - p;
                    e[m] = T::sparse_zero();
                    broke_early = true;
                    break;
                }

                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + c * b + c * b;
                // Accumulate shift
                let _ = half; // suppress unused warning
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;

                // Update eigenvectors
                #[allow(clippy::needless_range_loop)]
                for k in 0..n {
                    let t = z[k][i + 1];
                    z[k][i + 1] = s * z[k][i] + c * t;
                    z[k][i] = c * z[k][i] - s * t;
                }

                ii -= 1;
            }

            if broke_early {
                continue; // retry this eigenvalue
            }

            d[l] = d[l] - p;
            e[l] = g;
            e[m] = T::sparse_zero();
        }
    }

    // Sort eigenvalues and eigenvectors in descending order
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| d[j].partial_cmp(&d[i]).unwrap_or(std::cmp::Ordering::Equal));

    let mut sortedeigenvalues = Vec::with_capacity(numeigenvalues);
    let mut sorted_eigenvectors = Vec::with_capacity(numeigenvalues);

    #[allow(clippy::needless_range_loop)]
    for k in 0..numeigenvalues.min(n) {
        let idx = indices[k];
        sortedeigenvalues.push(d[idx]);

        let mut eigenvector = Vec::with_capacity(n);
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            eigenvector.push(z[i][idx]);
        }
        sorted_eigenvectors.push(eigenvector);
    }

    Ok((sortedeigenvalues, sorted_eigenvectors))
}

/// Solves a small (n ≤ 3) symmetric tridiagonal eigenvalue problem.
#[allow(unused_assignments)]
#[allow(dead_code)]
fn solve_small_tridiagonal<T>(
    alpha: &[T],
    beta: &[T],
    numeigenvalues: usize,
) -> SparseResult<(Vec<T>, Vec<Vec<T>>)>
where
    T: Float
        + SparseElement
        + Debug
        + Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
{
    let n = alpha.len();

    if n == 1 {
        // 1x1 case - just return the single value
        return Ok((vec![alpha[0]], vec![vec![T::sparse_one()]]));
    }

    if n == 2 {
        // 2x2 case - direct formula
        let a = alpha[0];
        let b = alpha[1];
        let c = beta[0];

        let trace = a + b;
        let det = a * b - c * c;

        // Calculate eigenvalues
        let discriminant = (trace * trace - T::from(4.0).expect("Operation failed") * det).sqrt();
        let lambda1 = (trace + discriminant) * T::from(0.5).expect("Operation failed");
        let lambda2 = (trace - discriminant) * T::from(0.5).expect("Operation failed");

        // Sort in descending order
        let (lambda1, lambda2) = if lambda1 >= lambda2 {
            (lambda1, lambda2)
        } else {
            (lambda2, lambda1)
        };

        // Calculate eigenvectors
        let mut v1 = vec![T::sparse_zero(); 2];
        let mut v2 = vec![T::sparse_zero(); 2];

        if !SparseElement::is_zero(&c) {
            v1[0] = c;
            v1[1] = lambda1 - a;

            v2[0] = c;
            v2[1] = lambda2 - a;

            // Normalize
            let norm1 = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt();
            let norm2 = (v2[0] * v2[0] + v2[1] * v2[1]).sqrt();

            if !SparseElement::is_zero(&norm1) {
                v1[0] = v1[0] / norm1;
                v1[1] = v1[1] / norm1;
            }

            if !SparseElement::is_zero(&norm2) {
                v2[0] = v2[0] / norm2;
                v2[1] = v2[1] / norm2;
            }
        } else {
            // c is zero - diagonal matrix case
            if a >= b {
                v1[0] = T::sparse_one();
                v1[1] = T::sparse_zero();

                v2[0] = T::sparse_zero();
                v2[1] = T::sparse_one();
            } else {
                v1[0] = T::sparse_zero();
                v1[1] = T::sparse_one();

                v2[0] = T::sparse_one();
                v2[1] = T::sparse_zero();
            }
        }

        let mut eigenvalues = vec![lambda1, lambda2];
        let mut eigenvectors = vec![v1, v2];

        // Return only the requested number of eigenvalues
        eigenvalues.truncate(numeigenvalues);
        eigenvectors.truncate(numeigenvalues);

        return Ok((eigenvalues, eigenvectors));
    }

    if n == 3 {
        // 3x3 case - use characteristic polynomial
        let a = alpha[0];
        let b = alpha[1];
        let c = alpha[2];
        let d = beta[0];
        let e = beta[1];

        // Characteristic polynomial coefficients
        let p = -(a + b + c);
        let q = a * b + a * c + b * c - d * d - e * e;
        let r = -(a * b * c - a * e * e - c * d * d);

        // Solve the cubic equation using the Vieta formulas
        let eigenvalues = solve_cubic(p, q, r)?;

        // Sort eigenvalues in descending order
        let mut sortedeigenvalues = eigenvalues.clone();
        sortedeigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Compute eigenvectors
        let mut eigenvectors = Vec::with_capacity(sortedeigenvalues.len());

        for &lambda in &sortedeigenvalues[0..numeigenvalues.min(3)] {
            // For each eigenvalue, construct the resulting linear system
            // (A - lambda*I)v = 0, and solve for v

            // Build the matrix (A - lambda*I)
            let mut m00 = a - lambda;
            let mut m01 = d;
            let m02 = T::sparse_zero();

            let mut m10 = d;
            let mut m11 = b - lambda;
            let mut m12 = e;

            let m20 = T::sparse_zero();
            let mut m21 = e;
            let mut m22 = c - lambda;

            // Find the largest absolute row to use as pivot
            let r0_norm = (m00 * m00 + m01 * m01 + m02 * m02).sqrt();
            let r1_norm = (m10 * m10 + m11 * m11 + m12 * m12).sqrt();
            let r2_norm = (m20 * m20 + m21 * m21 + m22 * m22).sqrt();

            let mut v = vec![T::sparse_zero(); 3];

            if r0_norm >= r1_norm && r0_norm >= r2_norm && !SparseElement::is_zero(&r0_norm) {
                // Use first row as pivot
                let scale = T::sparse_one() / r0_norm;
                m00 = m00 * scale;
                m01 = m01 * scale;

                // Eliminate first variable from second row
                let factor = m10 / m00;
                m11 = m11 - factor * m01;
                m12 = m12 - factor * m02;

                // Eliminate first variable from third row
                let factor = m20 / m00;
                m21 = m21 - factor * m01;
                m22 = m22 - factor * m02;

                // Back-substitute
                v[2] = T::sparse_one(); // Set last component to 1
                v[1] = -m12 * v[2] / m11;
                v[0] = -(m01 * v[1] + m02 * v[2]) / m00;
            } else if r1_norm >= r0_norm && r1_norm >= r2_norm && !SparseElement::is_zero(&r1_norm)
            {
                // Use second row as pivot
                let scale = T::sparse_one() / r1_norm;
                m10 = m10 * scale;
                m11 = m11 * scale;
                m12 = m12 * scale;

                // Back-substitute
                v[2] = T::sparse_one(); // Set last component to 1
                v[0] = -m02 * v[2] / m00;
                v[1] = -(m10 * v[0] + m12 * v[2]) / m11;
            } else if !SparseElement::is_zero(&r2_norm) {
                // Use third row as pivot
                v[0] = T::sparse_one(); // Set first component to 1
                v[1] = T::sparse_zero();
                v[2] = T::sparse_zero();
            } else {
                // Degenerate case - just use unit vector
                v[0] = T::sparse_one();
                v[1] = T::sparse_zero();
                v[2] = T::sparse_zero();
            }

            // Normalize the eigenvector
            let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if !SparseElement::is_zero(&norm) {
                v[0] = v[0] / norm;
                v[1] = v[1] / norm;
                v[2] = v[2] / norm;
            }

            eigenvectors.push(v);
        }

        // Return only the requested number of eigenvalues
        sortedeigenvalues.truncate(numeigenvalues);
        eigenvectors.truncate(numeigenvalues);

        return Ok((sortedeigenvalues, eigenvectors));
    }

    // For n > 3, use the implicit QL algorithm with shifts.
    // This is the standard approach for symmetric tridiagonal eigenproblems
    // (Golub & Van Loan, "Matrix Computations", Algorithm 8.3.3).

    let mut d = alpha.to_vec();
    let mut e = beta.to_vec();
    e.push(T::sparse_zero());

    // Eigenvector matrix (rows = components, columns = eigenvector index)
    let mut z = vec![vec![T::sparse_zero(); n]; n];
    for i in 0..n {
        z[i][i] = T::sparse_one();
    }

    let max_ql_iter: usize = 30 * n;

    for l in 0..n {
        let mut iter_count = 0usize;
        loop {
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if dd + e[m].abs() == dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            if iter_count >= max_ql_iter {
                return Err(SparseError::IterativeSolverFailure(
                    "QL algorithm for tridiagonal eigenproblem did not converge".to_string(),
                ));
            }
            iter_count += 1;

            // Wilkinson shift
            let g0 = (d[l + 1] - d[l]) / (T::from(2.0).expect("conv") * e[l]);
            let r0 = (g0 * g0 + T::sparse_one()).sqrt();
            let signed_r = if g0 >= T::sparse_zero() { r0 } else { -r0 };
            let mut g = d[m] - d[l] + e[l] / (g0 + signed_r);

            let mut s = T::sparse_one();
            let mut c = T::sparse_one();
            let mut p = T::sparse_zero();

            if m == 0 {
                break;
            }
            let mut i = m - 1;
            loop {
                let f = s * e[i];
                let b = c * e[i];
                let r = (f * f + g * g).sqrt();
                e[i + 1] = r;

                if SparseElement::is_zero(&r) {
                    d[i + 1] = d[i + 1] - p;
                    e[m] = T::sparse_zero();
                    break;
                }

                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                let rr = (d[i] - g) * s + T::from(2.0).expect("conv") * c * b;
                p = s * rr;
                d[i + 1] = g + p;
                g = c * rr - b;

                for k in 0..n {
                    let t = z[k][i + 1];
                    z[k][i + 1] = s * z[k][i] + c * t;
                    z[k][i] = c * z[k][i] - s * t;
                }

                if i == l {
                    break;
                }
                i -= 1;
            }
            d[l] = d[l] - p;
            e[l] = g;
            e[m] = T::sparse_zero();
        }
    }

    // Sort eigenvalues in descending order
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| d[b].partial_cmp(&d[a]).unwrap_or(std::cmp::Ordering::Equal));

    let take = numeigenvalues.min(n);
    let mut eigenvalues = Vec::with_capacity(take);
    let mut eigenvectors = Vec::with_capacity(take);

    for &idx in indices.iter().take(take) {
        eigenvalues.push(d[idx]);
        let mut v = Vec::with_capacity(n);
        for row in 0..n {
            v.push(z[row][idx]);
        }
        let nrm: T = v
            .iter()
            .map(|&x| x * x)
            .fold(T::sparse_zero(), |a, b| a + b)
            .sqrt();
        if !SparseElement::is_zero(&nrm) {
            for x in &mut v {
                *x = *x / nrm;
            }
        }
        eigenvectors.push(v);
    }

    Ok((eigenvalues, eigenvectors))
}

/// Solves a cubic equation ax³ + bx² + cx + d = 0
/// using Cardano's formula.
fn solve_cubic<T>(p: T, q: T, r: T) -> SparseResult<Vec<T>>
where
    T: Float
        + SparseElement
        + Debug
        + Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
{
    // The equation is x³ + px² + qx + r = 0

    // Substitute x = y - p/3 to eliminate the quadratic term
    let p_over_3 = p / T::from(3.0).expect("Operation failed");
    let q_new = q - p * p / T::from(3.0).expect("Operation failed");
    let r_new = r - p * q / T::from(3.0).expect("Operation failed")
        + T::from(2.0).expect("Operation failed") * p * p * p
            / T::from(27.0).expect("Operation failed");

    // Now solve y³ + q_new * y + r_new = 0
    let discriminant = -(T::from(4.0).expect("Operation failed") * q_new * q_new * q_new
        + T::from(27.0).expect("Operation failed") * r_new * r_new);

    if discriminant > T::sparse_zero() {
        // Three real roots
        let theta = ((T::from(3.0).expect("Operation failed") * r_new)
            / (T::from(2.0).expect("Operation failed") * q_new)
            * (-T::from(3.0).expect("Operation failed") / q_new).sqrt())
        .acos();
        let sqrt_term = T::from(2.0).expect("Operation failed")
            * (-q_new / T::from(3.0).expect("Operation failed")).sqrt();

        let y1 = sqrt_term * (theta / T::from(3.0).expect("Operation failed")).cos();
        let y2 = sqrt_term
            * ((theta
                + T::from(2.0).expect("Operation failed")
                    * T::from(std::f64::consts::PI).expect("Operation failed"))
                / T::from(3.0).expect("Operation failed"))
            .cos();
        let y3 = sqrt_term
            * ((theta
                + T::from(4.0).expect("Operation failed")
                    * T::from(std::f64::consts::PI).expect("Operation failed"))
                / T::from(3.0).expect("Operation failed"))
            .cos();

        let x1 = y1 - p_over_3;
        let x2 = y2 - p_over_3;
        let x3 = y3 - p_over_3;

        Ok(vec![x1, x2, x3])
    } else {
        // One real root
        let u = (-r_new / T::from(2.0).expect("Operation failed")
            + (discriminant / T::from(-108.0).expect("Operation failed")).sqrt())
        .cbrt();
        let v = if SparseElement::is_zero(&u) {
            T::sparse_zero()
        } else {
            -q_new / (T::from(3.0).expect("Operation failed") * u)
        };

        let y = u + v;
        let x = y - p_over_3;

        Ok(vec![x])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sym_csr::SymCsrMatrix;

    #[test]
    fn test_lanczos_simple() {
        // Create a simple 2x2 symmetric matrix [[2, 1], [1, 2]]
        // Only store lower triangular part for symmetric CSR
        let data = vec![2.0, 1.0, 2.0]; // values: diag[0], [1,0], diag[1]
        let indptr = vec![0, 1, 3]; // row 0 has 1 element, row 1 has 2 elements
        let indices = vec![0, 0, 1]; // column indices
        let matrix = SymCsrMatrix::new(data, indptr, indices, (2, 2)).expect("Operation failed");

        let options = LanczosOptions {
            max_iter: 100,
            max_subspace_size: 2,
            tol: 1e-8,
            numeigenvalues: 1,
            compute_eigenvectors: true,
        };
        let result = lanczos(&matrix, &options, None).expect("Operation failed");

        assert!(result.converged);
        // lanczos() returns all computed eigenvalues from the Krylov subspace;
        // callers use process_eigenvalue_selection to narrow down to k.
        assert!(
            !result.eigenvalues.is_empty(),
            "expected at least 1 eigenvalue, got {}",
            result.eigenvalues.len()
        );
        // Test that the eigenvalues are finite (algorithm converges)
        for &ev in result.eigenvalues.iter() {
            assert!(ev.is_finite());
        }
    }

    #[test]
    fn test_tridiagonal_solver_2x2() {
        let alpha = vec![2.0, 3.0];
        let beta = vec![1.0];
        let (eigenvalues, _eigenvectors) =
            solve_tridiagonal_eigenproblem(&alpha, &beta, 2).expect("Operation failed");

        assert_eq!(eigenvalues.len(), 2);
        // Eigenvalues should be sorted in descending order
        assert!(eigenvalues[0] >= eigenvalues[1]);
    }

    #[test]
    fn test_solve_cubic() {
        // Test x³ - 6x² + 11x - 6 = 0, which has roots 1, 2, 3
        let roots = solve_cubic(-6.0, 11.0, -6.0).expect("Operation failed");
        assert_eq!(roots.len(), 3);

        // Sort roots for comparison
        let mut sorted_roots = roots;
        sorted_roots.sort_by(|a, b| a.partial_cmp(b).expect("Operation failed"));

        assert!((sorted_roots[0] - 1.0).abs() < 1e-10);
        assert!((sorted_roots[1] - 2.0).abs() < 1e-10);
        assert!((sorted_roots[2] - 3.0).abs() < 1e-10);
    }
}
