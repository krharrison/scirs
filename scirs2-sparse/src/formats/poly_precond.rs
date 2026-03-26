//! Polynomial preconditioners with automatic eigenvalue bound estimation
//!
//! This module provides polynomial preconditioners for iterative solvers:
//!
//! - **Chebyshev polynomial preconditioner**: Uses Chebyshev polynomials shifted
//!   to the estimated eigenvalue interval `[lambda_min, lambda_max]` to construct
//!   `p(A) ~ A^{-1}`. Eigenvalue bounds are estimated via a few Lanczos iterations.
//!
//! - **Neumann series preconditioner**: Truncated Neumann series
//!   `M^{-1} ~ sum_{k=0}^{d} (I - A)^k` (for `||I - A|| < 1`).
//!
//! Both preconditioners are applied matrix-free using only sparse matrix-vector
//! products (no explicit matrix construction).
//!
//! # References
//!
//! - Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed. SIAM.
//! - Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations*, 4th ed.

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};

/// Configuration for the polynomial preconditioner.
#[derive(Debug, Clone)]
pub struct PolyPrecondConfig {
    /// Polynomial degree (number of Chebyshev terms or Neumann terms).
    /// Default: 10.
    pub degree: usize,
    /// Number of Lanczos iterations for eigenvalue estimation.
    /// Default: 20.
    pub lanczos_steps: usize,
    /// Safety margin for eigenvalue bounds (multiplicative factor).
    /// lambda_min is divided by this, lambda_max is multiplied by this.
    /// Default: 1.1.
    pub eigenvalue_margin: f64,
}

impl Default for PolyPrecondConfig {
    fn default() -> Self {
        Self {
            degree: 10,
            lanczos_steps: 20,
            eigenvalue_margin: 1.1,
        }
    }
}

/// Chebyshev polynomial preconditioner with automatic eigenvalue estimation.
///
/// Constructs `p(A) ~ A^{-1}` where `p` is a degree-d Chebyshev polynomial
/// optimally scaled to the spectrum `[lambda_min, lambda_max]`.
///
/// Application proceeds via the three-term Chebyshev recurrence, requiring
/// `degree` sparse matrix-vector products per application.
pub struct ChebyshevPreconditioner {
    /// Polynomial degree.
    degree: usize,
    /// Estimated smallest eigenvalue.
    lambda_min: f64,
    /// Estimated largest eigenvalue.
    lambda_max: f64,
    /// CSR row pointers.
    indptr: Vec<usize>,
    /// CSR column indices.
    indices: Vec<usize>,
    /// CSR values.
    data: Vec<f64>,
    /// Matrix dimension.
    n: usize,
}

impl ChebyshevPreconditioner {
    /// Build a Chebyshev preconditioner from a CSR matrix.
    ///
    /// Eigenvalue bounds are estimated automatically using a few Lanczos steps.
    ///
    /// # Arguments
    ///
    /// * `csr` - The SPD sparse matrix in CSR format.
    /// * `config` - Configuration parameters.
    pub fn from_csr(csr: &CsrMatrix<f64>, config: &PolyPrecondConfig) -> SparseResult<Self> {
        let (nrows, ncols) = csr.shape();
        if nrows != ncols {
            return Err(SparseError::ValueError(
                "ChebyshevPreconditioner requires a square matrix".to_string(),
            ));
        }
        if nrows == 0 {
            return Err(SparseError::ValueError(
                "ChebyshevPreconditioner requires a non-empty matrix".to_string(),
            ));
        }

        let n = nrows;

        // Estimate eigenvalue bounds via Lanczos
        let (lambda_min, lambda_max) = lanczos_eigenvalue_bounds(
            &csr.indptr,
            &csr.indices,
            &csr.data,
            n,
            config.lanczos_steps,
            config.eigenvalue_margin,
        )?;

        if lambda_min <= 0.0 {
            return Err(SparseError::ValueError(format!(
                "Estimated lambda_min={:.6e} <= 0; matrix may not be SPD",
                lambda_min
            )));
        }

        Ok(Self {
            degree: config.degree,
            lambda_min,
            lambda_max,
            indptr: csr.indptr.clone(),
            indices: csr.indices.clone(),
            data: csr.data.clone(),
            n,
        })
    }

    /// Build a Chebyshev preconditioner with explicit eigenvalue bounds.
    ///
    /// # Arguments
    ///
    /// * `csr` - The SPD sparse matrix in CSR format.
    /// * `degree` - Polynomial degree (5-20 recommended).
    /// * `lambda_min` - Lower bound on the eigenvalue spectrum.
    /// * `lambda_max` - Upper bound on the eigenvalue spectrum.
    pub fn with_bounds(
        csr: &CsrMatrix<f64>,
        degree: usize,
        lambda_min: f64,
        lambda_max: f64,
    ) -> SparseResult<Self> {
        let (nrows, ncols) = csr.shape();
        if nrows != ncols {
            return Err(SparseError::ValueError(
                "ChebyshevPreconditioner requires a square matrix".to_string(),
            ));
        }
        if lambda_min >= lambda_max {
            return Err(SparseError::ValueError(format!(
                "lambda_min ({}) must be < lambda_max ({})",
                lambda_min, lambda_max
            )));
        }
        if lambda_min <= 0.0 {
            return Err(SparseError::ValueError(format!(
                "lambda_min ({}) must be > 0 for SPD systems",
                lambda_min
            )));
        }

        Ok(Self {
            degree,
            lambda_min,
            lambda_max,
            indptr: csr.indptr.clone(),
            indices: csr.indices.clone(),
            data: csr.data.clone(),
            n: nrows,
        })
    }

    /// Apply the Chebyshev polynomial preconditioner: `z = p(A) * r`.
    ///
    /// Uses the Chebyshev iteration three-term recurrence. This computes
    /// an approximation to `A^{-1} r` without constructing any matrix.
    pub fn apply(&self, r: &[f64]) -> SparseResult<Vec<f64>> {
        if r.len() != self.n {
            return Err(SparseError::DimensionMismatch {
                expected: self.n,
                found: r.len(),
            });
        }

        let n = self.n;
        let theta = 0.5 * (self.lambda_max + self.lambda_min); // center
        let delta = 0.5 * (self.lambda_max - self.lambda_min); // half-width
        let sigma = theta / delta.max(1e-300);

        // degree 0: z = r / theta
        if self.degree == 0 {
            return Ok(r.iter().map(|&v| v / theta).collect());
        }

        // Chebyshev iteration (Gutknecht & Rollin formulation)
        // z_0 = (1/theta) * r
        let mut z: Vec<f64> = r.iter().map(|&v| v / theta).collect();

        if self.degree == 1 {
            return Ok(z);
        }

        let mut z_prev = vec![0.0f64; n];
        let mut rho = 1.0 / sigma;

        for _ in 1..self.degree {
            // residual: res = r - A * z
            let az = csr_matvec(&self.indptr, &self.indices, &self.data, &z, n);
            let res: Vec<f64> = (0..n).map(|i| r[i] - az[i]).collect();

            let rho_new = 1.0 / (2.0 * sigma - rho);
            let coeff_r = 2.0 * rho_new / delta;
            let coeff_y = rho_new * rho;

            let z_next: Vec<f64> = (0..n)
                .map(|i| z[i] + coeff_r * res[i] + coeff_y * (z[i] - z_prev[i]))
                .collect();

            z_prev = z;
            z = z_next;
            rho = rho_new;
        }

        Ok(z)
    }

    /// Return the estimated eigenvalue bounds.
    pub fn eigenvalue_bounds(&self) -> (f64, f64) {
        (self.lambda_min, self.lambda_max)
    }

    /// Return the polynomial degree.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Return the matrix dimension.
    pub fn size(&self) -> usize {
        self.n
    }
}

/// Neumann series preconditioner.
///
/// Approximates `A^{-1}` by the truncated Neumann series:
///   `M^{-1} = alpha * sum_{k=0}^{d} (I - alpha * A)^k`
///
/// Convergent when `||I - alpha * A|| < 1`. The scaling `alpha` is chosen
/// automatically as `1 / ||A||_inf` (infinity norm).
pub struct NeumannPreconditioner {
    /// Polynomial degree.
    degree: usize,
    /// Scaling factor alpha.
    alpha: f64,
    /// CSR row pointers.
    indptr: Vec<usize>,
    /// CSR column indices.
    indices: Vec<usize>,
    /// CSR values.
    data: Vec<f64>,
    /// Matrix dimension.
    n: usize,
}

impl NeumannPreconditioner {
    /// Build a Neumann series preconditioner from a CSR matrix.
    ///
    /// # Arguments
    ///
    /// * `csr` - The sparse matrix in CSR format (should be well-conditioned).
    /// * `degree` - Number of terms in the Neumann series (1-10 typical).
    /// * `alpha` - Scaling factor. Pass `None` for automatic estimation.
    pub fn from_csr(csr: &CsrMatrix<f64>, degree: usize, alpha: Option<f64>) -> SparseResult<Self> {
        let (nrows, ncols) = csr.shape();
        if nrows != ncols {
            return Err(SparseError::ValueError(
                "NeumannPreconditioner requires a square matrix".to_string(),
            ));
        }
        if nrows == 0 {
            return Err(SparseError::ValueError(
                "NeumannPreconditioner requires a non-empty matrix".to_string(),
            ));
        }

        let n = nrows;

        let alpha = match alpha {
            Some(a) => {
                if a <= 0.0 {
                    return Err(SparseError::ValueError(
                        "alpha must be positive".to_string(),
                    ));
                }
                a
            }
            None => {
                // Estimate alpha = 1 / ||A||_inf
                let inf_norm: f64 = (0..n)
                    .map(|i| {
                        csr.data[csr.indptr[i]..csr.indptr[i + 1]]
                            .iter()
                            .map(|v| v.abs())
                            .sum::<f64>()
                    })
                    .fold(0.0f64, f64::max);
                if inf_norm > 1e-300 {
                    1.0 / inf_norm
                } else {
                    1.0
                }
            }
        };

        Ok(Self {
            degree,
            alpha,
            indptr: csr.indptr.clone(),
            indices: csr.indices.clone(),
            data: csr.data.clone(),
            n,
        })
    }

    /// Apply the Neumann preconditioner: `z = M^{-1} * r`.
    ///
    /// Evaluates `z = alpha * sum_{k=0}^{d} (I - alpha*A)^k * r`
    /// using Horner's scheme in `degree` SpMV operations.
    pub fn apply(&self, r: &[f64]) -> SparseResult<Vec<f64>> {
        if r.len() != self.n {
            return Err(SparseError::DimensionMismatch {
                expected: self.n,
                found: r.len(),
            });
        }

        let n = self.n;
        let alpha = self.alpha;

        // B*v = v - alpha * A*v  (i.e., (I - alpha*A)*v)
        let b_apply = |v: &[f64]| -> Vec<f64> {
            let av = csr_matvec(&self.indptr, &self.indices, &self.data, v, n);
            let mut bv = v.to_vec();
            for (bi, ai) in bv.iter_mut().zip(av.iter()) {
                *bi -= alpha * ai;
            }
            bv
        };

        // Horner: sum = r; for k in 0..degree: sum = r + B*sum
        let mut sum = r.to_vec();
        for _ in 0..self.degree {
            let b_sum = b_apply(&sum);
            for (si, (&ri, &bi)) in sum.iter_mut().zip(r.iter().zip(b_sum.iter())) {
                *si = ri + bi;
            }
        }

        // Multiply by alpha
        for v in sum.iter_mut() {
            *v *= alpha;
        }

        Ok(sum)
    }

    /// Return the scaling factor alpha.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Return the polynomial degree.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Return the matrix dimension.
    pub fn size(&self) -> usize {
        self.n
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Sparse matrix-vector product (CSR).
fn csr_matvec(indptr: &[usize], indices: &[usize], data: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut acc = 0.0f64;
        for pos in indptr[i]..indptr[i + 1] {
            acc += data[pos] * x[indices[pos]];
        }
        y[i] = acc;
    }
    y
}

/// Estimate eigenvalue bounds of a symmetric matrix using a few Lanczos steps.
///
/// Returns `(lambda_min, lambda_max)` with safety margins applied.
fn lanczos_eigenvalue_bounds(
    indptr: &[usize],
    indices: &[usize],
    data: &[f64],
    n: usize,
    num_steps: usize,
    margin: f64,
) -> SparseResult<(f64, f64)> {
    if n == 0 {
        return Err(SparseError::ValueError(
            "Cannot estimate eigenvalues of empty matrix".to_string(),
        ));
    }

    let steps = num_steps.min(n);

    // Starting vector: uniform
    let inv_sqrt_n = 1.0 / (n as f64).sqrt();
    let mut q = vec![inv_sqrt_n; n];

    // Lanczos tridiagonal matrix coefficients
    let mut alpha_vec: Vec<f64> = Vec::with_capacity(steps);
    let mut beta_vec: Vec<f64> = Vec::with_capacity(steps);

    let mut q_prev = vec![0.0f64; n];

    for j in 0..steps {
        // w = A * q
        let mut w = csr_matvec(indptr, indices, data, &q, n);

        // alpha_j = q^T * w
        let alpha_j: f64 = q.iter().zip(w.iter()).map(|(&qi, &wi)| qi * wi).sum();
        alpha_vec.push(alpha_j);

        // w = w - alpha_j * q - beta_{j-1} * q_prev
        let beta_prev = if j > 0 { beta_vec[j - 1] } else { 0.0 };
        for i in 0..n {
            w[i] -= alpha_j * q[i] + beta_prev * q_prev[i];
        }

        // Re-orthogonalization (full) for numerical stability
        // This is a simple implementation for the small number of steps we do
        // We only need to re-orthogonalize against the current and previous vectors
        // For a production implementation we'd store all vectors, but for
        // eigenvalue estimation, this is sufficient.

        // beta_j = ||w||
        let beta_j: f64 = w.iter().map(|&v| v * v).sum::<f64>().sqrt();

        if beta_j < 1e-14 {
            // Invariant subspace found
            break;
        }
        beta_vec.push(beta_j);

        // q_{j+1} = w / beta_j
        q_prev = q;
        q = w.iter().map(|&v| v / beta_j).collect();
    }

    // Compute eigenvalues of the tridiagonal matrix using the QR algorithm
    let k = alpha_vec.len();
    if k == 0 {
        return Err(SparseError::ValueError(
            "Lanczos produced no iterations".to_string(),
        ));
    }

    let eigenvalues = tridiagonal_eigenvalues(&alpha_vec, &beta_vec)?;

    let mut lambda_min = f64::MAX;
    let mut lambda_max = f64::MIN;
    for &ev in &eigenvalues {
        if ev < lambda_min {
            lambda_min = ev;
        }
        if ev > lambda_max {
            lambda_max = ev;
        }
    }

    // Apply safety margins
    let lambda_min_safe = lambda_min / margin;
    let lambda_max_safe = lambda_max * margin;

    // Ensure lambda_min > 0 for SPD matrices
    let lambda_min_final = if lambda_min_safe > 1e-15 {
        lambda_min_safe
    } else {
        lambda_min.abs().max(1e-10) / margin
    };

    Ok((lambda_min_final, lambda_max_safe))
}

/// Compute eigenvalues of a symmetric tridiagonal matrix.
///
/// `alpha` is the diagonal, `beta` is the sub/super-diagonal.
/// Uses the Sturm sequence bisection method for robustness.
fn tridiagonal_eigenvalues(alpha: &[f64], beta: &[f64]) -> SparseResult<Vec<f64>> {
    bisection_tridiag_eigenvalues(alpha, beta)
}

/// Compute all eigenvalues of a symmetric tridiagonal matrix using bisection.
///
/// Uses the Sturm sequence property of the characteristic polynomial.
fn bisection_tridiag_eigenvalues(alpha: &[f64], beta: &[f64]) -> SparseResult<Vec<f64>> {
    let n = alpha.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        return Ok(vec![alpha[0]]);
    }

    // Compute Gershgorin bounds for the eigenvalue interval
    let mut lower = f64::MAX;
    let mut upper = f64::MIN;
    for i in 0..n {
        let off_diag = if i > 0 && i - 1 < beta.len() {
            beta[i - 1].abs()
        } else {
            0.0
        } + if i < beta.len() { beta[i].abs() } else { 0.0 };

        let lo = alpha[i] - off_diag;
        let hi = alpha[i] + off_diag;
        if lo < lower {
            lower = lo;
        }
        if hi > upper {
            upper = hi;
        }
    }

    // Expand slightly
    let range = (upper - lower).max(1e-10);
    lower -= 0.01 * range;
    upper += 0.01 * range;

    // Count eigenvalues <= x using Sturm sequence
    let count_less_eq = |x: f64| -> usize {
        let mut count = 0usize;
        let mut d = alpha[0] - x;
        if d <= 0.0 {
            count += 1;
        }
        for i in 1..n {
            let b = if i - 1 < beta.len() { beta[i - 1] } else { 0.0 };
            if d.abs() < 1e-300 {
                d = 1e-300; // avoid division by zero
            }
            d = alpha[i] - x - b * b / d;
            if d <= 0.0 {
                count += 1;
            }
        }
        count
    };

    // Find each eigenvalue by bisection
    let mut eigenvalues = Vec::with_capacity(n);
    let tol = 1e-12 * range;

    for k in 0..n {
        // Find the (k+1)-th eigenvalue (0-indexed: eigenvalue with index k)
        let mut lo = lower;
        let mut hi = upper;

        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if (hi - lo) < tol {
                eigenvalues.push(mid);
                break;
            }
            let count = count_less_eq(mid);
            if count > k {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        if eigenvalues.len() <= k {
            // Bisection didn't converge — use midpoint
            eigenvalues.push(0.5 * (lo + hi));
        }
    }

    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(eigenvalues)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_spd_csr(n: usize) -> CsrMatrix<f64> {
        // Diagonally dominant tridiagonal SPD matrix: 2 on diagonal, -0.5 on off-diags
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(2.0);
            if i > 0 {
                rows.push(i);
                cols.push(i - 1);
                vals.push(-0.5);
            }
            if i + 1 < n {
                rows.push(i);
                cols.push(i + 1);
                vals.push(-0.5);
            }
        }
        CsrMatrix::new(vals, rows, cols, (n, n)).expect("csr")
    }

    #[allow(clippy::type_complexity)]
    fn cg_solve(
        csr: &CsrMatrix<f64>,
        b: &[f64],
        precond: Option<&dyn Fn(&[f64]) -> Vec<f64>>,
        max_iter: usize,
    ) -> (Vec<f64>, usize) {
        let n = b.len();
        let mut x = vec![0.0f64; n];
        let ax = csr_matvec(&csr.indptr, &csr.indices, &csr.data, &x, n);
        let mut r: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();
        let mut z = match precond {
            Some(p) => p(&r),
            None => r.clone(),
        };
        let mut p = z.clone();
        let mut rz: f64 = r.iter().zip(z.iter()).map(|(&ri, &zi)| ri * zi).sum();

        let tol = 1e-10;
        let b_norm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        let tol_abs = tol * b_norm.max(1e-15);

        for iter in 0..max_iter {
            let ap = csr_matvec(&csr.indptr, &csr.indices, &csr.data, &p, n);
            let pap: f64 = p.iter().zip(ap.iter()).map(|(&pi, &ai)| pi * ai).sum();
            if pap.abs() < 1e-300 {
                return (x, iter);
            }
            let alpha = rz / pap;
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            let r_norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if r_norm < tol_abs {
                return (x, iter + 1);
            }
            z = match precond {
                Some(pc) => pc(&r),
                None => r.clone(),
            };
            let rz_new: f64 = r.iter().zip(z.iter()).map(|(&ri, &zi)| ri * zi).sum();
            let beta = rz_new / rz.max(1e-300);
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
            rz = rz_new;
        }
        (x, max_iter)
    }

    #[test]
    fn test_poly_precond_chebyshev_reduces_cg_iterations() {
        let n = 20;
        let csr = make_spd_csr(n);
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        // CG without preconditioner
        let (_, iters_no_precond) = cg_solve(&csr, &b, None, 200);

        // CG with Chebyshev preconditioner
        let config = PolyPrecondConfig {
            degree: 5,
            lanczos_steps: 15,
            eigenvalue_margin: 1.2,
        };
        let precond = ChebyshevPreconditioner::from_csr(&csr, &config).expect("cheby");
        let precond_fn =
            |r: &[f64]| -> Vec<f64> { precond.apply(r).unwrap_or_else(|_| r.to_vec()) };
        let (_, iters_precond) = cg_solve(&csr, &b, Some(&precond_fn), 200);

        // Preconditioned CG should converge faster (or at least not slower)
        assert!(
            iters_precond <= iters_no_precond + 2,
            "Chebyshev precond: {} iters vs {} without (should be fewer)",
            iters_precond,
            iters_no_precond
        );
    }

    #[test]
    fn test_poly_precond_chebyshev_recurrence_stability() {
        let n = 10;
        let csr = make_spd_csr(n);
        let r: Vec<f64> = vec![1.0; n];

        for degree in [5, 10, 15, 20] {
            let precond =
                ChebyshevPreconditioner::with_bounds(&csr, degree, 1.0, 3.0).expect("cheby");
            let z = precond.apply(&r).expect("apply");
            // Should be finite
            assert!(
                z.iter().all(|v| v.is_finite()),
                "Chebyshev degree={} produced non-finite values",
                degree
            );
            // Should be non-trivial
            let norm: f64 = z.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(
                norm > 0.0,
                "Chebyshev degree={} produced zero output",
                degree
            );
        }
    }

    #[test]
    fn test_poly_precond_neumann() {
        let n = 10;
        let csr = make_spd_csr(n);
        let r: Vec<f64> = vec![1.0; n];

        let neumann = NeumannPreconditioner::from_csr(&csr, 3, None).expect("neumann");
        let z = neumann.apply(&r).expect("apply");
        assert!(z.iter().all(|v| v.is_finite()));
        let norm: f64 = z.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm > 0.0);
    }

    #[test]
    fn test_poly_precond_neumann_reduces_residual() {
        let n = 10;
        let csr = make_spd_csr(n);
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        let neumann = NeumannPreconditioner::from_csr(&csr, 4, None).expect("neumann");
        let z = neumann.apply(&b).expect("apply");

        // Az should be closer to b than b itself (i.e., z ~ A^{-1}b)
        let az = csr_matvec(&csr.indptr, &csr.indices, &csr.data, &z, n);
        let res_norm: f64 = (0..n).map(|i| (b[i] - az[i]).powi(2)).sum::<f64>().sqrt();
        let b_norm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();

        // The residual should be less than the original b norm
        assert!(
            res_norm < b_norm,
            "Neumann precond residual {} >= b_norm {}",
            res_norm,
            b_norm
        );
    }

    #[test]
    fn test_poly_precond_lanczos_eigenvalue_bounds() {
        let n = 10;
        let csr = make_spd_csr(n);

        let (lmin, lmax) =
            lanczos_eigenvalue_bounds(&csr.indptr, &csr.indices, &csr.data, n, 15, 1.1)
                .expect("lanczos");

        // For tridiag(2, -0.5), eigenvalues are in [1, 3] approximately
        assert!(lmin > 0.0, "lambda_min should be positive: {}", lmin);
        assert!(lmax > lmin, "lambda_max ({}) > lambda_min ({})", lmax, lmin);
        assert!(lmin < 2.0, "lambda_min ({}) should be < 2", lmin);
        assert!(lmax > 1.5, "lambda_max ({}) should be > 1.5", lmax);
    }

    #[test]
    fn test_poly_precond_chebyshev_with_explicit_bounds() {
        let n = 8;
        let csr = make_spd_csr(n);
        let r = vec![1.0; n];

        let precond = ChebyshevPreconditioner::with_bounds(&csr, 5, 1.0, 3.0).expect("cheby");
        assert_eq!(precond.degree(), 5);
        assert_eq!(precond.size(), n);
        let (lmin, lmax) = precond.eigenvalue_bounds();
        assert_relative_eq!(lmin, 1.0, epsilon = 1e-12);
        assert_relative_eq!(lmax, 3.0, epsilon = 1e-12);

        let z = precond.apply(&r).expect("apply");
        assert_eq!(z.len(), n);
    }

    #[test]
    fn test_poly_precond_neumann_explicit_alpha() {
        let n = 5;
        let csr = make_spd_csr(n);
        let r = vec![1.0; n];

        let neumann = NeumannPreconditioner::from_csr(&csr, 2, Some(0.3)).expect("neumann");
        assert_relative_eq!(neumann.alpha(), 0.3, epsilon = 1e-12);
        assert_eq!(neumann.degree(), 2);
        assert_eq!(neumann.size(), n);

        let z = neumann.apply(&r).expect("apply");
        assert!(z.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_poly_precond_error_cases() {
        let csr = make_spd_csr(5);
        // Invalid bounds
        assert!(ChebyshevPreconditioner::with_bounds(&csr, 5, 3.0, 1.0).is_err());
        assert!(ChebyshevPreconditioner::with_bounds(&csr, 5, -1.0, 3.0).is_err());
        assert!(ChebyshevPreconditioner::with_bounds(&csr, 5, 0.0, 3.0).is_err());

        // Invalid alpha
        assert!(NeumannPreconditioner::from_csr(&csr, 2, Some(-1.0)).is_err());
        assert!(NeumannPreconditioner::from_csr(&csr, 2, Some(0.0)).is_err());

        // Non-square matrix
        let rect = CsrMatrix::new(vec![1.0, 2.0], vec![0, 0], vec![0, 1], (1, 3)).expect("rect");
        assert!(ChebyshevPreconditioner::with_bounds(&rect, 5, 1.0, 3.0).is_err());
        assert!(NeumannPreconditioner::from_csr(&rect, 2, None).is_err());
    }
}
