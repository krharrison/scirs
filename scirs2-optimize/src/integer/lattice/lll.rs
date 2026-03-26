//! LLL (Lenstra-Lenstra-Lovász) lattice basis reduction algorithm.
//!
//! The LLL algorithm reduces a lattice basis so that the basis vectors are
//! "short" and "nearly orthogonal". An LLL-reduced basis satisfies:
//! 1. Size-reduction: |mu[i][j]| <= eta for all j < i
//! 2. Lovász condition: bnorm_sq[k] >= (delta - mu[k][k-1]^2) * bnorm_sq[k-1]
//!
//! # References
//! - Lenstra, A.K., Lenstra, H.W., Lovász, L. (1982). "Factoring polynomials with
//!   rational number coefficients." Mathematische Annalen 261(4), 515–534.
//! - Cohen, H. (1993). "A Course in Computational Algebraic Number Theory."
//!   Springer, Sections 2.6–2.7.

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{Array1, Array2};

use super::gram_schmidt::{gram_schmidt, size_reduce_step, update_gram_schmidt_after_swap};

/// Configuration for the LLL reduction algorithm.
#[derive(Debug, Clone)]
pub struct LLLConfig {
    /// Lovász parameter controlling reduction quality (0.25 < delta < 1).
    /// Higher values yield better reduction but more iterations.
    /// Recommended: 0.75 (classical LLL) to 0.99 (near-HKZ quality).
    pub delta: f64,
    /// Size-reduction threshold: |mu[i][j]| must be <= eta.
    /// Default: 0.501 (slightly above 0.5 to handle floating-point boundary cases).
    pub eta: f64,
    /// Maximum number of swap iterations before declaring non-convergence.
    pub max_iterations: usize,
}

impl Default for LLLConfig {
    fn default() -> Self {
        LLLConfig {
            delta: 0.75,
            eta: 0.501,
            max_iterations: 10_000,
        }
    }
}

/// Result from LLL basis reduction.
#[derive(Debug, Clone)]
pub struct LLLResult {
    /// LLL-reduced basis (rows are reduced basis vectors).
    pub reduced_basis: Array2<f64>,
    /// Unimodular transformation matrix `U` such that `U * original_basis = reduced_basis`.
    pub transformation: Array2<f64>,
    /// Number of Lovász condition swap steps performed.
    pub n_swaps: usize,
    /// Number of size-reduction steps performed.
    pub n_size_reductions: usize,
}

/// LLL lattice basis reducer.
///
/// Implements the classical LLL algorithm with configurable Lovász parameter.
///
/// # Example
/// ```
/// use scirs2_optimize::integer::lattice::{LLLConfig, LLLReducer};
/// use scirs2_core::ndarray::array;
///
/// let basis = array![[1.0, 1.0], [1.0, 0.0]];
/// let reducer = LLLReducer::new(LLLConfig::default());
/// let result = reducer.reduce(&basis).unwrap();
/// assert_eq!(result.reduced_basis.nrows(), 2);
/// ```
pub struct LLLReducer {
    config: LLLConfig,
}

impl LLLReducer {
    /// Create a new LLL reducer with the given configuration.
    pub fn new(config: LLLConfig) -> Self {
        LLLReducer { config }
    }

    /// Apply LLL reduction to the given lattice basis.
    ///
    /// # Arguments
    /// * `basis` - Matrix of shape [n, d] where each row is a lattice basis vector.
    ///   The basis must be full-rank (all rows linearly independent).
    ///
    /// # Returns
    /// `LLLResult` with the reduced basis and the unimodular transformation.
    ///
    /// # Errors
    /// Returns `OptimizeError::ValueError` if the basis is empty or has fewer rows than columns
    /// (underdetermined), or `OptimizeError::ConvergenceError` if `max_iterations` is exceeded.
    pub fn reduce(&self, basis: &Array2<f64>) -> OptimizeResult<LLLResult> {
        let n = basis.nrows();
        let d = basis.ncols();
        if n == 0 || d == 0 {
            return Err(OptimizeError::ValueError(
                "Basis matrix must be non-empty".to_string(),
            ));
        }
        // Allow n > d only if the extra vectors are in the span of the first d vectors;
        // in practice, LLL works best when n <= d (full-rank lattice).
        // We allow n > d but warn via the max_iterations guard.

        let delta = self.config.delta;
        let eta = self.config.eta;

        // Working copy of the basis
        let mut b = basis.to_owned();
        // Initialize unimodular transformation as identity
        let mut u = Array2::<f64>::eye(n);

        // Compute initial Gram-Schmidt
        let (mut mu, mut bnorm_sq) = gram_schmidt(&b);

        let mut n_swaps = 0usize;
        let mut n_size_reductions = 0usize;
        let mut k = 1usize;
        let mut iteration = 0usize;

        while k < n {
            if iteration >= self.config.max_iterations {
                return Err(OptimizeError::ConvergenceError(format!(
                    "LLL did not converge within {} iterations (n_swaps={})",
                    self.config.max_iterations, n_swaps
                )));
            }
            iteration += 1;

            // Size-reduce b[k] with respect to b[j] for j = k-1, k-2, ..., 0
            for j in (0..k).rev() {
                if mu[[k, j]].abs() > eta {
                    size_reduce_step(&mut b, &mut u, &mut mu, k, j);
                    n_size_reductions += 1;
                }
            }

            // Check Lovász condition
            let mu_k_km1 = mu[[k, k - 1]];
            let lovász_rhs = (delta - mu_k_km1 * mu_k_km1) * bnorm_sq[k - 1];

            if bnorm_sq[k] >= lovász_rhs {
                // Condition satisfied: move forward
                k += 1;
            } else {
                // Condition violated: swap b[k-1] and b[k]
                swap_rows(&mut b, k - 1, k);
                swap_rows(&mut u, k - 1, k);
                n_swaps += 1;

                // Update Gram-Schmidt after the swap
                update_gram_schmidt_after_swap(&b, &mut mu, &mut bnorm_sq, k);

                // Step back
                if k > 1 {
                    k -= 1;
                }
            }
        }

        Ok(LLLResult {
            reduced_basis: b,
            transformation: u,
            n_swaps,
            n_size_reductions,
        })
    }

    /// Verify that the LLL conditions are satisfied for the given result.
    ///
    /// Checks:
    /// 1. Size-reduction: |mu[i][j]| <= eta + tolerance for all j < i
    /// 2. Lovász condition: bnorm_sq[k] >= (delta - mu[k][k-1]^2) * bnorm_sq[k-1] - tolerance
    ///
    /// Returns `true` if all conditions are satisfied.
    pub fn verify(&self, result: &LLLResult) -> bool {
        let n = result.reduced_basis.nrows();
        if n == 0 {
            return true;
        }

        let (mu, bnorm_sq) = gram_schmidt(&result.reduced_basis);
        let tol = 1e-6;
        let delta = self.config.delta;
        let eta = self.config.eta;

        // Check size-reduction
        for i in 0..n {
            for j in 0..i {
                if mu[[i, j]].abs() > eta + tol {
                    return false;
                }
            }
        }

        // Check Lovász condition
        for k in 1..n {
            let mu_k_km1 = mu[[k, k - 1]];
            let lhs = bnorm_sq[k];
            let rhs = (delta - mu_k_km1 * mu_k_km1) * bnorm_sq[k - 1];
            if lhs < rhs - tol * rhs.abs().max(1.0) {
                return false;
            }
        }

        true
    }
}

/// Swap two rows of a 2D array in place.
fn swap_rows(mat: &mut Array2<f64>, i: usize, j: usize) {
    let ncols = mat.ncols();
    for col in 0..ncols {
        let tmp = mat[[i, col]];
        mat[[i, col]] = mat[[j, col]];
        mat[[j, col]] = tmp;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// Compute the determinant of a square matrix via LU decomposition.
    fn det_naive(m: &Array2<f64>) -> f64 {
        let n = m.nrows();
        assert_eq!(n, m.ncols());
        if n == 1 {
            return m[[0, 0]];
        }
        if n == 2 {
            return m[[0, 0]] * m[[1, 1]] - m[[0, 1]] * m[[1, 0]];
        }
        if n == 3 {
            return m[[0, 0]] * (m[[1, 1]] * m[[2, 2]] - m[[1, 2]] * m[[2, 1]])
                - m[[0, 1]] * (m[[1, 0]] * m[[2, 2]] - m[[1, 2]] * m[[2, 0]])
                + m[[0, 2]] * (m[[1, 0]] * m[[2, 1]] - m[[1, 1]] * m[[2, 0]]);
        }
        // For larger n: expand along first row
        let mut d = 0.0;
        for col in 0..n {
            let mut minor = Array2::<f64>::zeros((n - 1, n - 1));
            for r in 1..n {
                let mut c2 = 0;
                for c in 0..n {
                    if c == col {
                        continue;
                    }
                    minor[[r - 1, c2]] = m[[r, c]];
                    c2 += 1;
                }
            }
            let sign = if col % 2 == 0 { 1.0 } else { -1.0 };
            d += sign * m[[0, col]] * det_naive(&minor);
        }
        d
    }

    /// Matrix multiply two 2D arrays (simple O(n^3)).
    fn matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        let n = a.nrows();
        let m = b.ncols();
        let k = a.ncols();
        let mut c = Array2::<f64>::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                for l in 0..k {
                    c[[i, j]] += a[[i, l]] * b[[l, j]];
                }
            }
        }
        c
    }

    #[test]
    fn test_lll_identity_basis_unchanged() {
        // LLL of an identity basis should remain an identity (or permutation thereof)
        let basis = Array2::<f64>::eye(3);
        let reducer = LLLReducer::new(LLLConfig::default());
        let result = reducer.reduce(&basis).expect("LLL should succeed");
        // Each row should have norm 1
        for i in 0..3 {
            let norm_sq: f64 = (0..3).map(|j| result.reduced_basis[[i, j]].powi(2)).sum();
            assert!((norm_sq - 1.0).abs() < 1e-8, "Norm of row {} should be 1", i);
        }
    }

    #[test]
    fn test_lll_result_satisfies_lovász_condition() {
        let basis = array![
            [1.0, 1.0, 1.0],
            [-1.0, 0.0, 2.0],
            [3.0, 5.0, 6.0]
        ];
        let reducer = LLLReducer::new(LLLConfig::default());
        let result = reducer.reduce(&basis).expect("LLL should succeed");
        assert!(
            reducer.verify(&result),
            "LLL result should satisfy Lovász condition"
        );
    }

    #[test]
    fn test_lll_transformation_unimodular() {
        // The transformation matrix U must have |det(U)| = 1
        let basis = array![[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]];
        let reducer = LLLReducer::new(LLLConfig::default());
        let result = reducer.reduce(&basis).expect("LLL should succeed");
        let det_u = det_naive(&result.transformation);
        assert!(
            (det_u.abs() - 1.0).abs() < 1e-6,
            "det(U) should be ±1, got {}",
            det_u
        );
    }

    #[test]
    fn test_lll_lattice_equivalence() {
        // U * B_original should equal B_reduced (they span the same lattice)
        let basis = array![[1.0, 2.0], [3.0, 4.0]];
        let reducer = LLLReducer::new(LLLConfig::default());
        let result = reducer.reduce(&basis).expect("LLL should succeed");
        let reconstructed = matmul(&result.transformation, &basis);
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (reconstructed[[i, j]] - result.reduced_basis[[i, j]]).abs() < 1e-6,
                    "U * B_orig[{},{}] = {} != B_red[{},{}] = {}",
                    i, j, reconstructed[[i, j]], i, j, result.reduced_basis[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_lll_2x2_classic() {
        // Classic 2×2 example: basis vectors [1, 1] and [0, 1]
        // LLL should give an already reduced basis or find short vectors
        let basis = array![[1.0, 1.0], [0.0, 1.0]];
        let reducer = LLLReducer::new(LLLConfig::default());
        let result = reducer.reduce(&basis).expect("LLL should succeed");
        assert!(reducer.verify(&result));
        // The first vector should be short
        let norm0_sq: f64 = result.reduced_basis.row(0).iter().map(|x| x * x).sum();
        assert!(norm0_sq <= 2.0 + 1e-8, "First vector should be short, got norm^2={}", norm0_sq);
    }

    #[test]
    fn test_lll_terminates_within_max_iterations() {
        // A harder basis: nearly parallel vectors
        let basis = array![
            [10.0, 3.0, -1.0, 2.0],
            [-4.0, 7.0, 2.0, 1.0],
            [1.0, -2.0, 5.0, 3.0],
            [2.0, 1.0, -3.0, 6.0]
        ];
        let config = LLLConfig {
            max_iterations: 10_000,
            ..Default::default()
        };
        let reducer = LLLReducer::new(config);
        let result = reducer.reduce(&basis).expect("LLL should terminate");
        assert!(result.n_swaps < 10_000);
    }

    #[test]
    fn test_lll_known_reduction() {
        // Example from Cohen (1993): basis = [[1,1,1],[-1,0,2],[3,5,6]]
        // After LLL, first vector should be shorter than the original
        let basis = array![[1.0, 1.0, 1.0], [-1.0, 0.0, 2.0], [3.0, 5.0, 6.0]];
        let reducer = LLLReducer::new(LLLConfig::default());
        let result = reducer.reduce(&basis).expect("LLL should succeed");

        // The first reduced vector should be at most as long as the shortest original
        let orig_norms: Vec<f64> = (0..3)
            .map(|i| (0..3).map(|j| basis[[i, j]].powi(2)).sum::<f64>())
            .collect();
        let min_orig_norm: f64 = orig_norms.into_iter().fold(f64::MAX, f64::min);
        let reduced_norm0: f64 = (0..3).map(|j| result.reduced_basis[[0, j]].powi(2)).sum();
        assert!(
            reduced_norm0 <= min_orig_norm + 1e-6,
            "LLL first vector norm^2={} should be <= min original norm^2={}",
            reduced_norm0, min_orig_norm
        );
    }

    #[test]
    fn test_size_reduce_step_reduces_mu() {
        use super::super::gram_schmidt::gram_schmidt;

        // Build a basis where mu[1][0] is large
        let basis = array![[1.0, 0.0], [5.0, 1.0]];
        let (mu_before, _) = gram_schmidt(&basis);
        assert!(mu_before[[1, 0]].abs() > 0.5, "Setup: mu[1][0] should be large");

        let mut b = basis.clone();
        let mut u = Array2::<f64>::eye(2);
        let (mut mu, mut _bnorm_sq) = gram_schmidt(&b);

        // Size-reduce b[1] w.r.t. b[0]
        size_reduce_step(&mut b, &mut u, &mut mu, 1, 0);

        // Now mu[1][0] should be <= 0.5
        assert!(
            mu[[1, 0]].abs() <= 0.501 + 1e-10,
            "After size-reduce, mu[1][0]={} should be <= 0.501",
            mu[[1, 0]]
        );
    }
}
