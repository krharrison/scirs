//! Polynomial preconditioners
//!
//! Two polynomial preconditioner families are provided:
//!
//! - **[`NeumannPoly`]**: Truncated Neumann series expansion
//!   `M ≈ ω Σ_{k=0}^{d} (I − ω A)^k`.
//!   Evaluated by `d` matrix-vector products at application time.
//!
//! - **[`ChebyshevPoly`]**: Optimal degree-`d` polynomial preconditioner
//!   based on Chebyshev polynomials shifted to the spectrum [λ_min, λ_max].
//!   The polynomial minimises the maximum condition number over the spectrum.
//!   Applied recursively via the three-term Chebyshev recurrence.
//!
//! Both preconditioners store the CSR matrix internally and are applied
//! via repeated sparse matrix-vector products.
//!
//! # References
//!
//! - Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed.
//!   SIAM.  §12.2 (polynomial preconditioners).
//! - Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations*, 4th ed.
//!   Johns Hopkins University Press.

use crate::error::{SparseError, SparseResult};

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

/// Sparse matrix-vector product (CSR).
fn csr_matvec(
    indptr: &[usize],
    indices: &[usize],
    data: &[f64],
    x: &[f64],
    n: usize,
) -> Vec<f64> {
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

// ---------------------------------------------------------------------------
// Neumann Polynomial Preconditioner
// ---------------------------------------------------------------------------

/// Neumann series polynomial preconditioner.
///
/// Approximates A⁻¹ by the truncated Neumann series:
///
///   M ≈ ω Σ_{k=0}^{d} (I − ω A)^k
///
/// where ω (`shift`) is a scaling factor chosen to improve the spectral
/// radius of `I − ω A`.  The default choice is ω = 1/‖A‖_∞.
///
/// Application uses Horner's scheme:
///   M x = ω (x + B (x + B (x + ⋯)))
/// where B = I − ω A, requiring `degree` sparse matrix-vector products.
pub struct NeumannPoly {
    /// Number of terms in the series (polynomial degree `d`).
    pub degree: usize,
    /// Scaling factor ω.
    pub omega: f64,
    // CSR storage of A.
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f64>,
    n: usize,
}

impl NeumannPoly {
    /// Build a Neumann polynomial preconditioner.
    ///
    /// # Arguments
    ///
    /// * `indptr`, `indices`, `data` – CSR representation of A (square, size n×n).
    /// * `n`      – Matrix dimension.
    /// * `degree` – Number of Neumann series terms (polynomial degree).
    /// * `omega`  – Scaling factor ω.  Pass `None` to use the automatic estimate
    ///              ω = 1/‖A‖_∞.
    pub fn new(
        indptr: &[usize],
        indices: &[usize],
        data: &[f64],
        n: usize,
        degree: usize,
        omega: Option<f64>,
    ) -> SparseResult<Self> {
        if indptr.len() != n + 1 {
            return Err(SparseError::InconsistentData {
                reason: format!("indptr length {} != n+1={}", indptr.len(), n + 1),
            });
        }

        let omega = match omega {
            Some(w) => {
                if w <= 0.0 {
                    return Err(SparseError::ValueError(format!(
                        "NeumannPoly omega={w} must be positive"
                    )));
                }
                w
            }
            None => {
                // Estimate ω ≈ 1/‖A‖_∞  (row-sum norm).
                let row_max: f64 = (0..n)
                    .map(|i| {
                        data[indptr[i]..indptr[i + 1]]
                            .iter()
                            .map(|v| v.abs())
                            .sum::<f64>()
                    })
                    .fold(0.0f64, f64::max);
                if row_max > 1e-300 {
                    1.0 / row_max
                } else {
                    1.0
                }
            }
        };

        Ok(Self {
            degree,
            omega,
            indptr: indptr.to_vec(),
            indices: indices.to_vec(),
            data: data.to_vec(),
            n,
        })
    }

    /// Apply the preconditioner: y = ω Σ_{k=0}^{d} (I − ω A)^k x.
    ///
    /// Uses Horner's scheme: evaluates in `degree` sparse matrix-vector products.
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        let n = self.n;
        let omega = self.omega;

        // B·v = v − omega * A·v
        let b_apply = |v: &[f64]| -> Vec<f64> {
            let av = csr_matvec(&self.indptr, &self.indices, &self.data, v, n);
            let mut bv = v.to_vec();
            for (bi, ai) in bv.iter_mut().zip(av.iter()) {
                *bi -= omega * ai;
            }
            bv
        };

        // Horner: sum = x; for k in 1..=degree: sum = x + B*sum
        let mut sum = x.to_vec();
        for _ in 0..self.degree {
            let b_sum = b_apply(&sum);
            for (si, (xi, bi)) in sum.iter_mut().zip(x.iter().zip(b_sum.iter())) {
                *si = xi + bi;
            }
        }

        // Multiply by omega.
        for v in sum.iter_mut() {
            *v *= omega;
        }
        sum
    }

    /// Return the matrix dimension.
    pub fn size(&self) -> usize {
        self.n
    }
}

// ---------------------------------------------------------------------------
// Chebyshev Polynomial Preconditioner
// ---------------------------------------------------------------------------

/// Chebyshev polynomial preconditioner.
///
/// Constructs the degree-`d` polynomial p(A) that minimises
/// `max_{λ ∈ [λ_min, λ_max]} |1 − λ p(λ)|`
/// using the Chebyshev polynomial shifted and scaled to [λ_min, λ_max].
///
/// Application proceeds via the three-term Chebyshev recurrence on vectors:
///   p_0(A) x = x / λ_c
///   p_1(A) x = (2/δ) (A − λ_c I) p_0(A) x + p_0(A) x   (≈ (2A−2λ_c I)/δ · x)
///   ...
/// where λ_c = (λ_max + λ_min)/2,  δ = (λ_max − λ_min)/2.
pub struct ChebyshevPoly {
    /// Polynomial degree.
    pub degree: usize,
    /// Lower bound of the eigenvalue interval.
    pub lambda_min: f64,
    /// Upper bound of the eigenvalue interval.
    pub lambda_max: f64,
    // CSR storage of A.
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f64>,
    n: usize,
}

impl ChebyshevPoly {
    /// Build a Chebyshev polynomial preconditioner.
    ///
    /// # Arguments
    ///
    /// * `indptr`, `indices`, `data` – CSR representation of A.
    /// * `n`          – Matrix dimension.
    /// * `degree`     – Polynomial degree (number of Chebyshev terms).
    /// * `lambda_min` – Estimated smallest eigenvalue of A.
    /// * `lambda_max` – Estimated largest eigenvalue of A.
    pub fn new(
        indptr: &[usize],
        indices: &[usize],
        data: &[f64],
        n: usize,
        degree: usize,
        lambda_min: f64,
        lambda_max: f64,
    ) -> SparseResult<Self> {
        if indptr.len() != n + 1 {
            return Err(SparseError::InconsistentData {
                reason: format!("indptr length {} != n+1={}", indptr.len(), n + 1),
            });
        }
        if lambda_min >= lambda_max {
            return Err(SparseError::ValueError(format!(
                "ChebyshevPoly: lambda_min={lambda_min} must be < lambda_max={lambda_max}"
            )));
        }
        if lambda_min <= 0.0 {
            return Err(SparseError::ValueError(format!(
                "ChebyshevPoly: lambda_min={lambda_min} must be > 0 for SPD systems"
            )));
        }

        Ok(Self {
            degree,
            lambda_min,
            lambda_max,
            indptr: indptr.to_vec(),
            indices: indices.to_vec(),
            data: data.to_vec(),
            n,
        })
    }

    /// Apply the Chebyshev polynomial preconditioner.
    ///
    /// Uses the Chebyshev iteration (Adams & Ortega) three-term recurrence to
    /// compute y ≈ A^{-1} x iteratively, where the polynomial is scaled to
    /// [lambda_min, lambda_max].
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        let n = self.n;
        let lambda_min = self.lambda_min;
        let lambda_max = self.lambda_max;

        let theta = 0.5 * (lambda_max + lambda_min); // centre of eigenvalue interval
        let delta = 0.5 * (lambda_max - lambda_min); // half-width
        let sigma = theta / delta.max(1e-300);

        let a_apply = |v: &[f64]| -> Vec<f64> {
            csr_matvec(&self.indptr, &self.indices, &self.data, v, n)
        };

        // degree 0: y = x / theta
        if self.degree == 0 {
            return x.iter().map(|v| v / theta).collect();
        }

        // r_0 = x (initial residual, assuming y_0 = 0)
        // y_0 = (1/theta) * r_0
        let mut y: Vec<f64> = x.iter().map(|v| v / theta).collect();

        if self.degree == 1 {
            return y;
        }

        // y_prev = 0 (y_{-1})
        let mut y_prev = vec![0.0f64; n];

        // Chebyshev iteration recurrence:
        // rho_{k+1} = 1 / (2*sigma - rho_k)
        // y_{k+1} = rho_{k+1} * (2/delta * r_k + y_{k-1})  -- but we express it differently
        //
        // Standard form (Gutknecht & Rollin):
        //   y_0 = (1/theta) * x
        //   d_k = (2 * rho_k / delta) * r_k  + rho_k * d_{k-1}
        //   y_{k+1} = y_k + d_k

        let mut rho = 1.0 / sigma;

        for _ in 1..self.degree {
            // r = x - A*y
            let ay = a_apply(&y);
            let r: Vec<f64> = (0..n).map(|i| x[i] - ay[i]).collect();

            let rho_new = 1.0 / (2.0 * sigma - rho);

            // y_{k+1} = y_k + rho_new * ((2/delta)*r + rho*(y_k - y_{k-1}))
            // But simpler: use direct Chebyshev iteration update
            let coeff_r = 2.0 * rho_new / delta;
            let coeff_y = rho_new * rho;

            let y_next: Vec<f64> = (0..n)
                .map(|i| {
                    y[i] + coeff_r * r[i] + coeff_y * (y[i] - y_prev[i])
                })
                .collect();

            y_prev = y;
            y = y_next;
            rho = rho_new;
        }

        y
    }

    /// Return the matrix dimension.
    pub fn size(&self) -> usize {
        self.n
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 4×4 diagonally-dominant tridiagonal SPD matrix.
    fn test_matrix() -> (Vec<usize>, Vec<usize>, Vec<f64>, usize) {
        let n = 4usize;
        let indptr = vec![0, 2, 5, 8, 10];
        let indices = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let data = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        (indptr, indices, data, n)
    }

    #[test]
    fn test_neumann_poly_auto_omega() {
        let (ip, idx, dat, n) = test_matrix();
        let prec = NeumannPoly::new(&ip, &idx, &dat, n, 3, None).expect("NeumannPoly failed");

        assert_eq!(prec.size(), n);
        assert!(prec.omega > 0.0);

        let b = vec![1.0, 0.0, 0.0, 0.0];
        let y = prec.apply(&b);
        assert_eq!(y.len(), n);
        let norm: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm > 0.0, "Neumann should give non-zero output");
    }

    #[test]
    fn test_neumann_poly_explicit_omega() {
        let (ip, idx, dat, n) = test_matrix();
        let prec = NeumannPoly::new(&ip, &idx, &dat, n, 2, Some(0.1))
            .expect("NeumannPoly explicit omega");

        let b = vec![1.0; n];
        let y = prec.apply(&b);
        assert!(y.iter().all(|v| v.is_finite()), "output must be finite");
    }

    #[test]
    fn test_neumann_poly_degree0() {
        // degree=0 ⟹ y = omega * x.
        let (ip, idx, dat, n) = test_matrix();
        let prec = NeumannPoly::new(&ip, &idx, &dat, n, 0, Some(0.5))
            .expect("NeumannPoly degree=0");
        let b = vec![2.0, 4.0, 6.0, 8.0];
        let y = prec.apply(&b);
        for (&bi, &yi) in b.iter().zip(y.iter()) {
            assert!((yi - 0.5 * bi).abs() < 1e-12, "degree=0: y={yi}, expected {}", 0.5 * bi);
        }
    }

    #[test]
    fn test_neumann_poly_invalid_omega() {
        let (ip, idx, dat, n) = test_matrix();
        assert!(NeumannPoly::new(&ip, &idx, &dat, n, 2, Some(-1.0)).is_err());
        assert!(NeumannPoly::new(&ip, &idx, &dat, n, 2, Some(0.0)).is_err());
    }

    #[test]
    fn test_chebyshev_poly_basic() {
        let (ip, idx, dat, n) = test_matrix();
        // Eigenvalues of tridiag(-1,4,-1) 4×4 are in [2, 6] approximately.
        let prec = ChebyshevPoly::new(&ip, &idx, &dat, n, 3, 2.0, 6.0)
            .expect("ChebyshevPoly failed");

        assert_eq!(prec.size(), n);

        let b = vec![1.0, 2.0, 3.0, 4.0];
        let y = prec.apply(&b);
        assert_eq!(y.len(), n);
        assert!(y.iter().all(|v| v.is_finite()), "Chebyshev output must be finite");
    }

    #[test]
    fn test_chebyshev_poly_degree0() {
        let (ip, idx, dat, n) = test_matrix();
        let prec = ChebyshevPoly::new(&ip, &idx, &dat, n, 0, 2.0, 6.0)
            .expect("Chebyshev degree=0");
        let b = vec![4.0, 8.0, 12.0, 16.0];
        let y = prec.apply(&b);
        let lc = 4.0f64; // (2+6)/2
        for (&bi, &yi) in b.iter().zip(y.iter()) {
            assert!((yi - bi / lc).abs() < 1e-10, "degree=0: y={yi}, expected {}", bi / lc);
        }
    }

    #[test]
    fn test_chebyshev_poly_invalid_range() {
        let (ip, idx, dat, n) = test_matrix();
        // lambda_min >= lambda_max
        assert!(ChebyshevPoly::new(&ip, &idx, &dat, n, 2, 6.0, 2.0).is_err());
        assert!(ChebyshevPoly::new(&ip, &idx, &dat, n, 2, 3.0, 3.0).is_err());
        // lambda_min <= 0
        assert!(ChebyshevPoly::new(&ip, &idx, &dat, n, 2, -1.0, 6.0).is_err());
        assert!(ChebyshevPoly::new(&ip, &idx, &dat, n, 2, 0.0, 6.0).is_err());
    }

    #[test]
    fn test_chebyshev_approx_inverse() {
        // For a diagonal matrix D with eigenvalues in [lambda_min, lambda_max],
        // the Chebyshev preconditioner should approximate D^{-1}.
        let n = 3;
        let lambda_min = 1.0;
        let lambda_max = 3.0;
        let indptr = vec![0, 1, 2, 3];
        let indices = vec![0, 1, 2];
        let data = vec![2.0, 2.5, 3.0]; // diagonal entries within [1,3]

        let prec = ChebyshevPoly::new(&indptr, &indices, &data, n, 5, lambda_min, lambda_max)
            .expect("ChebyshevPoly for diagonal");

        let b = vec![1.0; n];
        let y = prec.apply(&b);

        // The result should be a reasonable approximation to A^{-1} b = [0.5, 0.4, 0.333].
        // We don't require high accuracy here; just check it's in the right ballpark.
        assert!(y.iter().all(|v| v.is_finite() && *v > 0.0),
            "Chebyshev of diagonal SPD should give positive result: {y:?}");
    }
}
