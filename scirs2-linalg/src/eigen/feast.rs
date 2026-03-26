//! FEAST contour integral eigensolver
//!
//! The FEAST algorithm computes eigenvalues and eigenvectors within a specified
//! interval [a, b] by using contour integration of the resolvent (zI - A)^{-1}
//! as a spectral projector.
//!
//! ## Algorithm Overview
//!
//! 1. Choose quadrature points on an elliptical contour enclosing [a, b]
//! 2. At each quadrature point z_j, solve (z_j*I - A) * X_j = Y (linear systems)
//! 3. Form subspace Q = sum(w_j * X_j) (weighted sum)
//! 4. Rayleigh-Ritz projection: solve reduced eigenvalue problem in Q
//! 5. Iterate until convergence
//!
//! ## References
//!
//! - Polizzi, E. (2009). "Density-matrix-based algorithm for solving eigenvalue problems."
//!   Physical Review B, 79(11), 115112.
//! - Tang, P. T. P., & Polizzi, E. (2014). "FEAST as a subspace iteration eigensolver
//!   accelerated by approximate spectral projection." SIAM Journal on Matrix Analysis
//!   and Applications, 35(2), 354-390.

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};

/// Complex pair vectors result: (points as (re,im), weights as (re,im))
type ComplexPairVecsResult<F> = LinalgResult<(Vec<(F, F)>, Vec<(F, F)>)>;

/// Configuration for the FEAST eigensolver
#[derive(Clone, Debug)]
pub struct FeastConfig<F: Float> {
    /// Lower bound of the search interval
    pub interval_lower: F,
    /// Upper bound of the search interval
    pub interval_upper: F,
    /// Subspace dimension (should be >= expected number of eigenvalues in interval)
    pub subspace_dim: usize,
    /// Number of quadrature points on the contour (typically 8 or 16)
    pub num_quadrature_points: usize,
    /// Maximum number of FEAST iterations
    pub max_iterations: usize,
    /// Convergence tolerance for eigenvalue residuals
    pub tolerance: F,
    /// Random seed for initial subspace generation
    pub seed: Option<u64>,
}

impl<F: Float> FeastConfig<F> {
    /// Create a new FEAST configuration with default settings
    pub fn new(interval_lower: F, interval_upper: F, subspace_dim: usize) -> Self {
        Self {
            interval_lower,
            interval_upper,
            subspace_dim,
            num_quadrature_points: 8,
            max_iterations: 20,
            tolerance: F::from(1e-10).unwrap_or(F::epsilon()),
            seed: Some(42),
        }
    }

    /// Set the number of quadrature points
    pub fn with_quadrature_points(mut self, n: usize) -> Self {
        self.num_quadrature_points = n;
        self
    }

    /// Set the maximum number of iterations
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Set the convergence tolerance
    pub fn with_tolerance(mut self, tol: F) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set the random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Result of a FEAST computation
#[derive(Clone, Debug)]
pub struct FeastResult<F: Float> {
    /// Eigenvalues found within the search interval
    pub eigenvalues: Array1<F>,
    /// Corresponding eigenvectors (column-major: column i is eigenvector i)
    pub eigenvectors: Array2<F>,
    /// Residual norms for each eigenpair
    pub residuals: Array1<F>,
    /// Number of eigenvalues found in the interval
    pub num_eigenvalues: usize,
    /// Number of FEAST iterations performed
    pub iterations: usize,
    /// Whether the algorithm converged
    pub converged: bool,
}

/// Compute Gauss-Legendre quadrature nodes and weights on [-1, 1].
///
/// Uses the Golub-Welsch algorithm: eigenvalues of the symmetric tridiagonal
/// Jacobi matrix give the nodes, and the weights come from the first component
/// of the eigenvectors.
fn gauss_legendre_quadrature<F>(n: usize) -> LinalgResult<(Vec<F>, Vec<F>)>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    if n == 0 {
        return Ok((vec![], vec![]));
    }
    if n == 1 {
        return Ok((
            vec![F::zero()],
            vec![F::from(2.0).ok_or_else(|| {
                LinalgError::ComputationError("Failed to convert 2.0".to_string())
            })?],
        ));
    }

    // Build the symmetric tridiagonal Jacobi matrix
    // For Gauss-Legendre: alpha_i = 0, beta_i = i / sqrt(4i^2 - 1)
    let mut diag = Array1::<F>::zeros(n);
    let mut offdiag = Array1::<F>::zeros(n - 1);

    for i in 1..n {
        let fi = F::from(i).ok_or_else(|| {
            LinalgError::ComputationError(format!("Failed to convert {} to float", i))
        })?;
        let four = F::from(4.0)
            .ok_or_else(|| LinalgError::ComputationError("Failed to convert 4.0".to_string()))?;
        let one = F::one();
        offdiag[i - 1] = fi / (four * fi * fi - one).sqrt();
    }

    // Solve the tridiagonal eigenproblem using implicit QL algorithm
    let (nodes, eigvecs) = tridiagonal_eigensolver(&diag, &offdiag)?;

    let two = F::from(2.0)
        .ok_or_else(|| LinalgError::ComputationError("Failed to convert 2.0".to_string()))?;

    let mut sorted_nodes = Vec::with_capacity(n);
    let mut sorted_weights = Vec::with_capacity(n);

    // Sort by node value
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        nodes[a]
            .partial_cmp(&nodes[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for &idx in &indices {
        sorted_nodes.push(nodes[idx]);
        // Weight = 2 * (first component of eigenvector)^2
        let v0 = eigvecs[[0, idx]];
        sorted_weights.push(two * v0 * v0);
    }

    Ok((sorted_nodes, sorted_weights))
}

/// Solve a symmetric tridiagonal eigenproblem using the QR algorithm with
/// implicit Wilkinson shift (based on LAPACK's DSTEQR).
///
/// Returns (eigenvalues, eigenvectors) where eigenvectors are stored column-wise.
fn tridiagonal_eigensolver<F>(
    diag: &Array1<F>,
    offdiag: &Array1<F>,
) -> LinalgResult<(Array1<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let n = diag.len();
    if n == 0 {
        return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
    }
    if n == 1 {
        return Ok((diag.clone(), Array2::eye(1)));
    }

    // Use Jacobi eigenvalue method which is robust for tridiagonal matrices.
    // Convert to dense and use Jacobi
    let mut mat = Array2::<F>::zeros((n, n));
    for i in 0..n {
        mat[[i, i]] = diag[i];
    }
    for i in 0..n - 1 {
        mat[[i, i + 1]] = offdiag[i];
        mat[[i + 1, i]] = offdiag[i];
    }

    // Use Jacobi eigenvalue method
    jacobi_tridiag_eigensolver(&mat)
}

/// Jacobi eigenvalue algorithm for symmetric matrices.
/// Robust fallback for the tridiagonal eigensolver.
fn jacobi_tridiag_eigensolver<F>(a: &Array2<F>) -> LinalgResult<(Array1<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let n = a.nrows();
    let mut mat = a.clone();
    let mut v = Array2::<F>::eye(n);
    let eps = F::epsilon();
    let max_sweeps = 100;
    let two = F::from(2.0)
        .ok_or_else(|| LinalgError::ComputationError("Failed to convert 2.0".to_string()))?;

    for _sweep in 0..max_sweeps {
        // Check convergence: sum of squares of off-diagonal elements
        let mut off_sum = F::zero();
        for i in 0..n {
            for j in (i + 1)..n {
                off_sum += mat[[i, j]] * mat[[i, j]];
            }
        }

        if off_sum < eps * eps * F::from(n as f64).unwrap_or(F::one()) {
            break;
        }

        // Sweep through all off-diagonal elements
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = mat[[p, q]];
                if apq.abs() < eps * (mat[[p, p]].abs() + mat[[q, q]].abs() + eps) {
                    continue;
                }

                // Compute Jacobi rotation
                let tau = (mat[[q, q]] - mat[[p, p]]) / (two * apq);
                let t = if tau.abs() > F::from(1e15).unwrap_or(F::max_value()) {
                    F::one() / (two * tau)
                } else {
                    let sign_tau = if tau >= F::zero() {
                        F::one()
                    } else {
                        -F::one()
                    };
                    sign_tau / (tau.abs() + (F::one() + tau * tau).sqrt())
                };

                let c = F::one() / (F::one() + t * t).sqrt();
                let s = t * c;
                let tau_val = s / (F::one() + c);

                // Update matrix
                mat[[p, q]] = F::zero();
                mat[[q, p]] = F::zero();

                let app = mat[[p, p]] - t * apq;
                let aqq = mat[[q, q]] + t * apq;
                mat[[p, p]] = app;
                mat[[q, q]] = aqq;

                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = mat[[r, p]];
                    let arq = mat[[r, q]];
                    mat[[r, p]] = arp - s * (arq + tau_val * arp);
                    mat[[p, r]] = mat[[r, p]];
                    mat[[r, q]] = arq + s * (arp - tau_val * arq);
                    mat[[q, r]] = mat[[r, q]];
                }

                // Update eigenvectors
                for r in 0..n {
                    let vrp = v[[r, p]];
                    let vrq = v[[r, q]];
                    v[[r, p]] = vrp - s * (vrq + tau_val * vrp);
                    v[[r, q]] = vrq + s * (vrp - tau_val * vrq);
                }
            }
        }
    }

    // Extract eigenvalues from diagonal
    let eigenvalues = Array1::from_iter((0..n).map(|i| mat[[i, i]]));

    Ok((eigenvalues, v))
}

/// Generate quadrature points and weights on an elliptical contour
/// enclosing the interval [a, b] on the real axis.
///
/// The contour is parameterized as z(theta) = center + (half_width * cos(theta) + i * r_imag * sin(theta))
/// where center = (a+b)/2, half_width = (b-a)/2, and r_imag is chosen proportional to half_width.
fn contour_quadrature<F>(lower: F, upper: F, num_points: usize) -> ComplexPairVecsResult<F>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let two = F::from(2.0)
        .ok_or_else(|| LinalgError::ComputationError("Failed to convert 2.0".to_string()))?;
    let pi = F::from(std::f64::consts::PI)
        .ok_or_else(|| LinalgError::ComputationError("Failed to convert PI".to_string()))?;

    let center = (lower + upper) / two;
    let half_width = (upper - lower) / two;
    // Imaginary radius: proportional to half_width for a well-conditioned contour
    let r_imag = half_width;

    // Get Gauss-Legendre quadrature on [-1, 1]
    let (gl_nodes, gl_weights) = gauss_legendre_quadrature(num_points)?;

    let mut z_points = Vec::with_capacity(num_points);
    let mut weights = Vec::with_capacity(num_points);

    // Map GL nodes from [-1, 1] to [0, pi] (upper half of contour)
    // theta = pi/2 * (1 + t) maps [-1, 1] -> [0, pi]
    let half_pi = pi / two;

    for k in 0..num_points {
        let t = gl_nodes[k];
        let theta = half_pi * (F::one() + t);

        // z(theta) = center + half_width * cos(theta) + i * r_imag * sin(theta)
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        let z_real = center + half_width * cos_theta;
        let z_imag = r_imag * sin_theta;

        // dz/dtheta = -half_width * sin(theta) + i * r_imag * cos(theta)
        // Weight contribution = w_k * dz/dtheta * (pi/2)
        let dz_real = -half_width * sin_theta;
        let dz_imag = r_imag * cos_theta;

        // Complex weight = gl_weight * dz/dtheta * pi/2
        let w_real = gl_weights[k] * dz_real * half_pi;
        let w_imag = gl_weights[k] * dz_imag * half_pi;

        z_points.push((z_real, z_imag));
        weights.push((w_real, w_imag));
    }

    Ok((z_points, weights))
}

/// Solve the complex linear system (z*I - A) * X = Y where z = z_re + i*z_im.
///
/// For a real matrix A and complex shift z, we reformulate as a real 2n x 2n system:
///   [ (z_re*I - A)   -z_im*I ] [ X_re ]   [ Y_re ]
///   [  z_im*I    (z_re*I - A) ] [ X_im ] = [ Y_im ]
///
/// Since Y is real (Y_im = 0), we solve:
///   (z_re*I - A) X_re + z_im * X_im = Y
///   -z_im * X_re + (z_re*I - A) * X_im = 0
///
/// From the second equation: X_im = (z_re*I - A)^{-1} * z_im * X_re
/// Substituting: [(z_re*I - A) + z_im^2 * (z_re*I - A)^{-1}] X_re = Y
///
/// Equivalently: [(z_re*I - A)^2 + z_im^2 * I] * X_re_temp = (z_re*I - A) * Y
/// where X_re = (z_re*I - A)^{-1} * X_re_temp...
///
/// Simpler approach: solve the full 2n x 2n real system directly.
fn solve_shifted_system<F>(
    a: &ArrayView2<F>,
    z_re: F,
    z_im: F,
    y: &Array2<F>,
) -> LinalgResult<(Array2<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let n = a.nrows();
    let m = y.ncols(); // number of right-hand sides

    // Build the 2n x 2n real system
    // [ (z_re*I - A)   -z_im*I ] [ X_re ]   [ Y ]
    // [  z_im*I    (z_re*I - A) ] [ X_im ] = [ 0 ]
    let mut big_a = Array2::<F>::zeros((2 * n, 2 * n));
    let mut big_b = Array2::<F>::zeros((2 * n, m));

    // Top-left and bottom-right blocks: z_re*I - A
    for i in 0..n {
        for j in 0..n {
            let val = if i == j { z_re - a[[i, j]] } else { -a[[i, j]] };
            big_a[[i, j]] = val;
            big_a[[n + i, n + j]] = val;
        }
    }

    // Top-right block: -z_im*I
    // Bottom-left block: z_im*I
    for i in 0..n {
        big_a[[i, n + i]] = -z_im;
        big_a[[n + i, i]] = z_im;
    }

    // RHS: Y in top part, 0 in bottom part
    for i in 0..n {
        for j in 0..m {
            big_b[[i, j]] = y[[i, j]];
        }
    }

    // Solve the system
    let big_x = crate::solve::solve_multiple(&big_a.view(), &big_b.view(), None).map_err(|e| {
        LinalgError::ComputationError(format!(
            "Failed to solve shifted system at z = ({}, {}): {}",
            z_re.to_f64().unwrap_or(0.0),
            z_im.to_f64().unwrap_or(0.0),
            e
        ))
    })?;

    // Extract real and imaginary parts
    let x_re = big_x.slice(s![..n, ..]).to_owned();
    let x_im = big_x.slice(s![n.., ..]).to_owned();

    Ok((x_re, x_im))
}

/// Generate an initial random subspace of dimension m0.
fn generate_initial_subspace<F>(n: usize, m0: usize, seed: Option<u64>) -> Array2<F>
where
    F: Float + NumAssign,
{
    use scirs2_core::random::prelude::*;

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::seed_from_u64(12345),
    };

    let mut y = Array2::<F>::zeros((n, m0));
    for i in 0..n {
        for j in 0..m0 {
            // Simple pseudo-random values in [-1, 1]
            let val: f64 = rng.random::<f64>() * 2.0 - 1.0;
            y[[i, j]] = F::from(val).unwrap_or(F::zero());
        }
    }

    y
}

/// QR factorization for orthogonalization using modified Gram-Schmidt.
fn qr_orthogonalize<F>(q: &mut Array2<F>) -> LinalgResult<()>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let n = q.nrows();
    let m = q.ncols();

    for j in 0..m {
        // Orthogonalize against previous columns
        for i in 0..j {
            let mut dot = F::zero();
            for k in 0..n {
                dot += q[[k, i]] * q[[k, j]];
            }
            let col_i: Vec<F> = (0..n).map(|k| q[[k, i]]).collect();
            for k in 0..n {
                q[[k, j]] -= dot * col_i[k];
            }
        }

        // Normalize
        let mut norm = F::zero();
        for k in 0..n {
            norm += q[[k, j]] * q[[k, j]];
        }
        norm = norm.sqrt();

        if norm > F::epsilon() * F::from(100.0).unwrap_or(F::one()) {
            for k in 0..n {
                q[[k, j]] /= norm;
            }
        } else {
            // Column is linearly dependent; zero it out
            for k in 0..n {
                q[[k, j]] = F::zero();
            }
        }
    }

    // Second pass for numerical stability
    for j in 0..m {
        for i in 0..j {
            let mut dot = F::zero();
            for k in 0..n {
                dot += q[[k, i]] * q[[k, j]];
            }
            let col_i: Vec<F> = (0..n).map(|k| q[[k, i]]).collect();
            for k in 0..n {
                q[[k, j]] -= dot * col_i[k];
            }
        }
        let mut norm = F::zero();
        for k in 0..n {
            norm += q[[k, j]] * q[[k, j]];
        }
        norm = norm.sqrt();
        if norm > F::epsilon() * F::from(100.0).unwrap_or(F::one()) {
            for k in 0..n {
                q[[k, j]] /= norm;
            }
        }
    }

    Ok(())
}

/// FEAST eigensolver for standard eigenvalue problems (Ax = lambda*x).
///
/// Computes eigenvalues and eigenvectors of a symmetric matrix A
/// that lie within a specified interval [a, b].
///
/// # Arguments
///
/// * `a` - Symmetric matrix (n x n)
/// * `config` - FEAST configuration specifying interval, subspace size, etc.
///
/// # Returns
///
/// * `FeastResult` containing eigenvalues, eigenvectors, residuals, and convergence info
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use scirs2_linalg::eigen::feast::{feast, FeastConfig};
///
/// // Diagonal matrix with eigenvalues 1, 2, 3, 4, 5
/// let mut a = Array2::<f64>::zeros((5, 5));
/// for i in 0..5 {
///     a[[i, i]] = (i + 1) as f64;
/// }
///
/// // Find eigenvalues in [1.5, 3.5] — should find 2 and 3
/// let config = FeastConfig::new(1.5, 3.5, 3);
/// let result = feast(&a.view(), &config).expect("FEAST failed");
/// assert_eq!(result.num_eigenvalues, 2);
/// ```
pub fn feast<F>(a: &ArrayView2<F>, config: &FeastConfig<F>) -> LinalgResult<FeastResult<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let n = a.nrows();
    if n != a.ncols() {
        return Err(LinalgError::ShapeError(
            "Matrix must be square for FEAST eigensolver".to_string(),
        ));
    }

    if config.interval_lower >= config.interval_upper {
        return Err(LinalgError::DomainError(
            "Search interval lower bound must be less than upper bound".to_string(),
        ));
    }

    if config.subspace_dim == 0 || config.subspace_dim > n {
        return Err(LinalgError::DomainError(format!(
            "Subspace dimension must be in [1, {}], got {}",
            n, config.subspace_dim
        )));
    }

    let m0 = config.subspace_dim;

    // Step 1: Generate initial random subspace Y (n x m0)
    let mut y = generate_initial_subspace(n, m0, config.seed);

    // Step 2: Get contour quadrature points and weights
    let (z_points, z_weights) = contour_quadrature(
        config.interval_lower,
        config.interval_upper,
        config.num_quadrature_points,
    )?;

    let mut eigenvalues = Array1::<F>::zeros(m0);
    let mut eigenvectors = Array2::<F>::zeros((n, m0));
    let mut residuals = Array1::<F>::from_elem(m0, F::infinity());
    let mut converged = false;
    let mut iteration = 0;

    let one_over_pi = F::one()
        / F::from(std::f64::consts::PI)
            .ok_or_else(|| LinalgError::ComputationError("Failed to convert PI".to_string()))?;

    for iter in 0..config.max_iterations {
        iteration = iter + 1;

        // Step 3: Apply spectral projector via contour integration
        // Q = (1/pi) * Im[ sum_j w_j * (z_j*I - A)^{-1} * Y ]
        //
        // For real symmetric A with real Y, the contour integral using only
        // the upper half gives: Q = (-1/pi) * sum_j Im[w_j * X_j]
        // where (z_j*I - A) * X_j = Y
        let mut q = Array2::<F>::zeros((n, m0));

        for k in 0..z_points.len() {
            let (z_re, z_im) = z_points[k];
            let (w_re, w_im) = z_weights[k];

            // Solve (z*I - A) * X = Y for X (complex)
            let (x_re, x_im) = solve_shifted_system(a, z_re, z_im, &y)?;

            // Accumulate: Q += Im[w * X] / pi
            // w * X = (w_re + i*w_im)(X_re + i*X_im)
            //       = (w_re*X_re - w_im*X_im) + i*(w_re*X_im + w_im*X_re)
            // We need the imaginary part: w_re*X_im + w_im*X_re
            for i in 0..n {
                for j in 0..m0 {
                    q[[i, j]] -= one_over_pi * (w_re * x_im[[i, j]] + w_im * x_re[[i, j]]);
                }
            }
        }

        // Step 4: Orthogonalize Q
        qr_orthogonalize(&mut q)?;

        // Step 5: Rayleigh-Ritz projection
        // Compute A_q = Q^T * A * Q (m0 x m0)
        let aq = a.dot(&q); // n x m0
        let mut a_projected = q.t().dot(&aq); // m0 x m0

        // Symmetrize the projected matrix to handle floating-point asymmetry
        let two = F::from(2.0)
            .ok_or_else(|| LinalgError::ComputationError("Failed to convert 2.0".to_string()))?;
        for ii in 0..m0 {
            for jj in (ii + 1)..m0 {
                let avg = (a_projected[[ii, jj]] + a_projected[[jj, ii]]) / two;
                a_projected[[ii, jj]] = avg;
                a_projected[[jj, ii]] = avg;
            }
        }

        // Step 6: Solve the reduced eigenproblem
        let (evals, evecs) = crate::eigen::eigh(&a_projected.view(), None).map_err(|e| {
            LinalgError::ComputationError(format!(
                "Failed to solve reduced eigenproblem in FEAST iteration {}: {}",
                iter + 1,
                e
            ))
        })?;

        // Step 7: Map back to full space: X = Q * V_reduced
        let x_full = q.dot(&evecs);

        // Step 8: Compute residuals and identify eigenvalues in interval
        let mut max_residual = F::zero();
        for i in 0..m0 {
            let xi = x_full.column(i);
            let axi = a.dot(&xi.to_owned());
            let lambda_xi = xi.mapv(|v| v * evals[i]);
            let mut res_norm_sq = F::zero();
            for k in 0..n {
                let diff = axi[k] - lambda_xi[k];
                res_norm_sq += diff * diff;
            }
            let res_norm = res_norm_sq.sqrt();
            residuals[i] = res_norm;
            if res_norm > max_residual {
                max_residual = res_norm;
            }
        }

        // Update eigenvalues and eigenvectors
        eigenvalues = evals;
        eigenvectors = x_full;

        // Update Y for next iteration (use current eigenvectors as new subspace)
        y = eigenvectors.clone();

        // Check convergence: all eigenvalues in interval have small residuals
        let mut all_converged = true;
        let mut count_in_interval = 0;
        for i in 0..m0 {
            if eigenvalues[i] >= config.interval_lower && eigenvalues[i] <= config.interval_upper {
                count_in_interval += 1;
                if residuals[i] > config.tolerance {
                    all_converged = false;
                }
            }
        }

        if all_converged && count_in_interval > 0 {
            converged = true;
            break;
        }
    }

    // Filter eigenvalues to those within the search interval
    let mut in_interval_indices = Vec::new();
    for i in 0..m0 {
        if eigenvalues[i] >= config.interval_lower && eigenvalues[i] <= config.interval_upper {
            in_interval_indices.push(i);
        }
    }

    // Sort by eigenvalue
    in_interval_indices.sort_by(|&a, &b| {
        eigenvalues[a]
            .partial_cmp(&eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let num_found = in_interval_indices.len();
    let mut result_evals = Array1::<F>::zeros(num_found);
    let mut result_evecs = Array2::<F>::zeros((n, num_found));
    let mut result_res = Array1::<F>::zeros(num_found);

    for (new_idx, &old_idx) in in_interval_indices.iter().enumerate() {
        result_evals[new_idx] = eigenvalues[old_idx];
        result_evecs
            .column_mut(new_idx)
            .assign(&eigenvectors.column(old_idx));
        result_res[new_idx] = residuals[old_idx];
    }

    Ok(FeastResult {
        eigenvalues: result_evals,
        eigenvectors: result_evecs,
        residuals: result_res,
        num_eigenvalues: num_found,
        iterations: iteration,
        converged,
    })
}

/// FEAST eigensolver for generalized eigenvalue problems (Ax = lambda*Bx).
///
/// Computes eigenvalues and eigenvectors of the generalized eigenproblem
/// Ax = lambda*Bx that lie within a specified interval [a, b],
/// where A and B are symmetric and B is positive definite.
///
/// # Arguments
///
/// * `a` - Symmetric matrix A (n x n)
/// * `b` - Symmetric positive definite matrix B (n x n)
/// * `config` - FEAST configuration specifying interval, subspace size, etc.
///
/// # Returns
///
/// * `FeastResult` containing eigenvalues, eigenvectors, residuals, and convergence info
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use scirs2_linalg::eigen::feast::{feast_generalized, FeastConfig};
///
/// let a = Array2::<f64>::from_diag(&scirs2_core::ndarray::array![2.0, 4.0, 6.0]);
/// let b = Array2::<f64>::eye(3);
///
/// let config = FeastConfig::new(1.0, 5.0, 3);
/// let result = feast_generalized(&a.view(), &b.view(), &config).expect("FEAST failed");
/// assert_eq!(result.num_eigenvalues, 2);
/// ```
pub fn feast_generalized<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    config: &FeastConfig<F>,
) -> LinalgResult<FeastResult<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let n = a.nrows();
    if n != a.ncols() || n != b.nrows() || n != b.ncols() {
        return Err(LinalgError::ShapeError(
            "Matrices A and B must be square and of the same size".to_string(),
        ));
    }

    if config.interval_lower >= config.interval_upper {
        return Err(LinalgError::DomainError(
            "Search interval lower bound must be less than upper bound".to_string(),
        ));
    }

    if config.subspace_dim == 0 || config.subspace_dim > n {
        return Err(LinalgError::DomainError(format!(
            "Subspace dimension must be in [1, {}], got {}",
            n, config.subspace_dim
        )));
    }

    let m0 = config.subspace_dim;

    // Initial random subspace
    let mut y = generate_initial_subspace(n, m0, config.seed);

    // Contour quadrature
    let (z_points, z_weights) = contour_quadrature(
        config.interval_lower,
        config.interval_upper,
        config.num_quadrature_points,
    )?;

    let mut eigenvalues = Array1::<F>::zeros(m0);
    let mut eigenvectors = Array2::<F>::zeros((n, m0));
    let mut residuals = Array1::<F>::from_elem(m0, F::infinity());
    let mut converged = false;
    let mut iteration = 0;

    let one_over_pi = F::one()
        / F::from(std::f64::consts::PI)
            .ok_or_else(|| LinalgError::ComputationError("Failed to convert PI".to_string()))?;

    for iter in 0..config.max_iterations {
        iteration = iter + 1;

        // Apply spectral projector: solve (z*B - A) * X = B * Y at each quad point
        let by = b.dot(&y); // n x m0

        let mut q = Array2::<F>::zeros((n, m0));

        for k in 0..z_points.len() {
            let (z_re, z_im) = z_points[k];
            let (w_re, w_im) = z_weights[k];

            // Build z*B - A
            let mut shifted = Array2::<F>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    shifted[[i, j]] = z_re * b[[i, j]] - a[[i, j]];
                }
            }

            // Solve (z*B - A) X = B*Y as a complex system
            // Real part of shifted is z_re*B - A, imaginary part is z_im*B
            let (x_re, x_im) = solve_shifted_system_generalized(&shifted.view(), z_im, b, &by)?;

            // Accumulate imaginary part of w * X
            for i in 0..n {
                for j in 0..m0 {
                    q[[i, j]] -= one_over_pi * (w_re * x_im[[i, j]] + w_im * x_re[[i, j]]);
                }
            }
        }

        // Orthogonalize Q
        qr_orthogonalize(&mut q)?;

        // Rayleigh-Ritz: solve projected generalized problem
        let aq = a.dot(&q);
        let bq = b.dot(&q);
        let mut a_proj = q.t().dot(&aq);
        let mut b_proj = q.t().dot(&bq);

        // Symmetrize projected matrices
        let two = F::from(2.0)
            .ok_or_else(|| LinalgError::ComputationError("Failed to convert 2.0".to_string()))?;
        for ii in 0..m0 {
            for jj in (ii + 1)..m0 {
                let avg_a = (a_proj[[ii, jj]] + a_proj[[jj, ii]]) / two;
                a_proj[[ii, jj]] = avg_a;
                a_proj[[jj, ii]] = avg_a;
                let avg_b = (b_proj[[ii, jj]] + b_proj[[jj, ii]]) / two;
                b_proj[[ii, jj]] = avg_b;
                b_proj[[jj, ii]] = avg_b;
            }
        }

        // Solve generalized eigenproblem A_proj * v = lambda * B_proj * v
        let (evals, evecs) =
            crate::eigen::eigh_gen(&a_proj.view(), &b_proj.view(), None).map_err(|e| {
                LinalgError::ComputationError(format!(
                    "Failed to solve reduced generalized eigenproblem: {}",
                    e
                ))
            })?;

        // Map back to full space
        let x_full = q.dot(&evecs);

        // Compute residuals: ||Ax - lambda*Bx|| / ||Bx||
        for i in 0..m0 {
            let xi = x_full.column(i);
            let axi = a.dot(&xi.to_owned());
            let bxi = b.dot(&xi.to_owned());
            let mut res_sq = F::zero();
            let mut bx_sq = F::zero();
            for kk in 0..n {
                let diff = axi[kk] - evals[i] * bxi[kk];
                res_sq += diff * diff;
                bx_sq += bxi[kk] * bxi[kk];
            }
            let bx_norm = bx_sq.sqrt();
            residuals[i] = if bx_norm > F::epsilon() {
                res_sq.sqrt() / bx_norm
            } else {
                res_sq.sqrt()
            };
        }

        eigenvalues = evals;
        eigenvectors = x_full;
        y = eigenvectors.clone();

        // Check convergence
        let mut all_converged = true;
        let mut count_in = 0;
        for i in 0..m0 {
            if eigenvalues[i] >= config.interval_lower && eigenvalues[i] <= config.interval_upper {
                count_in += 1;
                if residuals[i] > config.tolerance {
                    all_converged = false;
                }
            }
        }

        if all_converged && count_in > 0 {
            converged = true;
            break;
        }
    }

    // Filter to interval
    let mut in_interval: Vec<usize> = (0..m0)
        .filter(|&i| {
            eigenvalues[i] >= config.interval_lower && eigenvalues[i] <= config.interval_upper
        })
        .collect();
    in_interval.sort_by(|&a_idx, &b_idx| {
        eigenvalues[a_idx]
            .partial_cmp(&eigenvalues[b_idx])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let num_found = in_interval.len();
    let mut result_evals = Array1::<F>::zeros(num_found);
    let mut result_evecs = Array2::<F>::zeros((n, num_found));
    let mut result_res = Array1::<F>::zeros(num_found);

    for (new_idx, &old_idx) in in_interval.iter().enumerate() {
        result_evals[new_idx] = eigenvalues[old_idx];
        result_evecs
            .column_mut(new_idx)
            .assign(&eigenvectors.column(old_idx));
        result_res[new_idx] = residuals[old_idx];
    }

    Ok(FeastResult {
        eigenvalues: result_evals,
        eigenvectors: result_evecs,
        residuals: result_res,
        num_eigenvalues: num_found,
        iterations: iteration,
        converged,
    })
}

/// Solve the complex linear system for the generalized FEAST problem.
///
/// Solves (M + i*z_im*B) * X = RHS where M = z_re*B - A (already computed).
fn solve_shifted_system_generalized<F>(
    m_real: &ArrayView2<F>,
    z_im: F,
    b: &ArrayView2<F>,
    rhs: &Array2<F>,
) -> LinalgResult<(Array2<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let n = m_real.nrows();
    let m = rhs.ncols();

    // Build 2n x 2n real system:
    // [ M_real   -z_im*B ] [ X_re ]   [ RHS ]
    // [ z_im*B    M_real ] [ X_im ] = [  0  ]
    let mut big_a = Array2::<F>::zeros((2 * n, 2 * n));
    let mut big_b = Array2::<F>::zeros((2 * n, m));

    for i in 0..n {
        for j in 0..n {
            big_a[[i, j]] = m_real[[i, j]];
            big_a[[n + i, n + j]] = m_real[[i, j]];
            big_a[[i, n + j]] = -z_im * b[[i, j]];
            big_a[[n + i, j]] = z_im * b[[i, j]];
        }
    }

    for i in 0..n {
        for j in 0..m {
            big_b[[i, j]] = rhs[[i, j]];
        }
    }

    let big_x = crate::solve::solve_multiple(&big_a.view(), &big_b.view(), None).map_err(|e| {
        LinalgError::ComputationError(format!("Failed to solve generalized shifted system: {}", e))
    })?;

    let x_re = big_x.slice(s![..n, ..]).to_owned();
    let x_im = big_x.slice(s![n.., ..]).to_owned();

    Ok((x_re, x_im))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_feast_diagonal_matrix() {
        // Diagonal matrix with eigenvalues 1, 2, 3, 4, 5
        let mut a = Array2::<f64>::zeros((5, 5));
        for i in 0..5 {
            a[[i, i]] = (i + 1) as f64;
        }

        // Find eigenvalues in [1.5, 3.5] — should find 2.0 and 3.0
        let config = FeastConfig::new(1.5, 3.5, 4)
            .with_quadrature_points(8)
            .with_tolerance(1e-8);

        let result = feast(&a.view(), &config).expect("FEAST failed");

        assert_eq!(
            result.num_eigenvalues, 2,
            "Expected 2 eigenvalues in [1.5, 3.5], found {}",
            result.num_eigenvalues
        );

        // Check eigenvalues are close to 2.0 and 3.0
        let mut found_evals: Vec<f64> = result.eigenvalues.to_vec();
        found_evals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert_relative_eq!(found_evals[0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(found_evals[1], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_feast_symmetric_tridiagonal() {
        // Symmetric tridiagonal matrix (Wilkinson-like)
        let n = 7;
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            a[[i, i]] = (i as f64) + 1.0; // diagonal: 1, 2, 3, 4, 5, 6, 7
            if i + 1 < n {
                a[[i, i + 1]] = 0.5;
                a[[i + 1, i]] = 0.5;
            }
        }

        // Find interior eigenvalues in [2.5, 5.5]
        // The off-diagonal perturbation shifts eigenvalues slightly
        let config = FeastConfig::new(2.5, 5.5, 5)
            .with_quadrature_points(16)
            .with_tolerance(1e-8)
            .with_max_iterations(30);

        let result = feast(&a.view(), &config).expect("FEAST failed on tridiagonal");

        // Should find 3 eigenvalues near 3, 4, 5 (shifted by off-diagonal)
        assert!(
            result.num_eigenvalues >= 2,
            "Expected at least 2 eigenvalues in [2.5, 5.5], found {}",
            result.num_eigenvalues
        );

        // Verify residuals
        for i in 0..result.num_eigenvalues {
            assert!(
                result.residuals[i] < 1e-4,
                "Residual {} too large: {}",
                i,
                result.residuals[i]
            );
        }
    }

    #[test]
    fn test_feast_eigenvalue_count() {
        // Matrix with known eigenvalue distribution
        let eigenvals = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0];
        let n = eigenvals.len();
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            a[[i, i]] = eigenvals[i];
        }

        // Interval [1.5, 6.0] should contain exactly 3 eigenvalues: 2, 3, 5
        let config = FeastConfig::new(1.5, 6.0, 5).with_tolerance(1e-8);
        let result = feast(&a.view(), &config).expect("FEAST failed");

        assert_eq!(
            result.num_eigenvalues, 3,
            "Expected 3 eigenvalues in [1.5, 6.0], found {}",
            result.num_eigenvalues
        );
    }

    #[test]
    fn test_feast_eigenvector_residual() {
        // 4x4 symmetric matrix
        let a = array![
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 1.0]
        ];

        let config = FeastConfig::new(0.0, 5.0, 4)
            .with_quadrature_points(16)
            .with_tolerance(1e-10)
            .with_max_iterations(30);

        let result = feast(&a.view(), &config).expect("FEAST failed");

        // Verify that ||Ax - lambda*x|| is small for each eigenpair
        for i in 0..result.num_eigenvalues {
            let x = result.eigenvectors.column(i);
            let ax = a.dot(&x.to_owned());
            let lambda_x = x.mapv(|v| v * result.eigenvalues[i]);

            let mut residual_sq = 0.0;
            for k in 0..a.nrows() {
                let diff = ax[k] - lambda_x[k];
                residual_sq += diff * diff;
            }
            let residual = residual_sq.sqrt();

            assert!(
                residual < 1e-6,
                "Eigenpair {} has residual {} > 1e-6, eigenvalue = {}",
                i,
                residual,
                result.eigenvalues[i]
            );
        }
    }

    #[test]
    fn test_feast_generalized_identity_b() {
        // Generalized problem with B = I reduces to standard problem
        // Use a larger matrix for better subspace quality
        let n = 6;
        let eigenvals = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0];
        let a = Array2::<f64>::from_diag(&Array1::from_vec(eigenvals.to_vec()));
        let b = Array2::<f64>::eye(n);

        // Search for eigenvalues in [4.0, 8.0] -> should find 5.0 and 7.0
        let config = FeastConfig::new(4.0, 8.0, 4)
            .with_quadrature_points(16)
            .with_max_iterations(30)
            .with_tolerance(1e-8);

        let result = feast_generalized(&a.view(), &b.view(), &config).expect("FEAST gen failed");

        assert_eq!(
            result.num_eigenvalues, 2,
            "Expected 2 eigenvalues in [4, 8]"
        );

        // Check eigenvalues are in the interval
        for i in 0..result.num_eigenvalues {
            assert!(
                result.eigenvalues[i] >= 4.0 && result.eigenvalues[i] <= 8.0,
                "Eigenvalue {} = {} should be in [4, 8]",
                i,
                result.eigenvalues[i]
            );
        }

        // Verify eigenpair residuals: ||Ax - lambda*Bx|| should be reasonable
        for i in 0..result.num_eigenvalues {
            let x = result.eigenvectors.column(i);
            let ax = a.dot(&x.to_owned());
            let bx = b.dot(&x.to_owned());
            let lambda = result.eigenvalues[i];
            let mut res_sq = 0.0;
            for kk in 0..n {
                let diff = ax[kk] - lambda * bx[kk];
                res_sq += diff * diff;
            }
            assert!(
                res_sq.sqrt() < 1.0,
                "Eigenpair {} has large residual: {}",
                i,
                res_sq.sqrt()
            );
        }
    }

    #[test]
    fn test_feast_single_eigenvalue_in_interval() {
        // Diagonal matrix, search narrow interval containing exactly one eigenvalue
        let a = Array2::<f64>::from_diag(&array![1.0, 3.0, 5.0, 7.0, 9.0]);

        let config = FeastConfig::new(2.5, 3.5, 3).with_tolerance(1e-8);

        let result = feast(&a.view(), &config).expect("FEAST failed");
        assert_eq!(result.num_eigenvalues, 1);
        assert_relative_eq!(result.eigenvalues[0], 3.0, epsilon = 1e-5);
    }

    #[test]
    fn test_feast_no_eigenvalue_in_interval() {
        // Diagonal matrix with no eigenvalues in [3.5, 4.5]
        let a = Array2::<f64>::from_diag(&array![1.0, 3.0, 5.0, 7.0]);

        let config = FeastConfig::new(3.5, 4.5, 2).with_tolerance(1e-8);

        let result = feast(&a.view(), &config).expect("FEAST failed");
        assert_eq!(
            result.num_eigenvalues, 0,
            "Expected 0 eigenvalues in [3.5, 4.5]"
        );
    }

    #[test]
    fn test_feast_invalid_inputs() {
        // Non-square matrix
        let a = Array2::<f64>::zeros((3, 4));
        let config = FeastConfig::new(0.0, 1.0, 2);
        assert!(feast(&a.view(), &config).is_err());

        // Invalid interval
        let a = Array2::<f64>::eye(3);
        let config = FeastConfig::new(5.0, 1.0, 2);
        assert!(feast(&a.view(), &config).is_err());

        // Invalid subspace dim
        let config = FeastConfig::new(0.0, 1.0, 0);
        assert!(feast(&a.view(), &config).is_err());
    }

    #[test]
    fn test_gauss_legendre_quadrature() {
        // Test that GL quadrature integrates polynomials exactly
        // For n points, GL exactly integrates polynomials of degree <= 2n-1
        let (nodes, weights) = gauss_legendre_quadrature::<f64>(4).expect("GL failed");

        assert_eq!(nodes.len(), 4);
        assert_eq!(weights.len(), 4);

        // Integral of 1 over [-1, 1] = 2
        let integral_1: f64 = weights.iter().sum();
        assert_relative_eq!(integral_1, 2.0, epsilon = 1e-12);

        // Integral of x over [-1, 1] = 0
        let integral_x: f64 = nodes.iter().zip(weights.iter()).map(|(&x, &w)| w * x).sum();
        assert_relative_eq!(integral_x, 0.0, epsilon = 1e-12);

        // Integral of x^2 over [-1, 1] = 2/3
        let integral_x2: f64 = nodes
            .iter()
            .zip(weights.iter())
            .map(|(&x, &w)| w * x * x)
            .sum();
        assert_relative_eq!(integral_x2, 2.0 / 3.0, epsilon = 1e-12);
    }
}
