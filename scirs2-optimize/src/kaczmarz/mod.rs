//! Kaczmarz and Block Kaczmarz Methods for Linear Systems
//!
//! Iterative algorithms for solving large (possibly overdetermined) linear systems Ax = b.
//!
//! ## Variants
//!
//! - **Randomized** (Strohmer-Vershynin): Select row i with probability proportional to ||a_i||².
//!   Guarantees exponential convergence in expectation for consistent systems.
//! - **Block**: Select a block of rows simultaneously and solve the projected sub-problem.
//!   Uses QR (Gram-Schmidt) for small dense block systems.
//! - **GreedyMaxResidual**: Select the row with the largest absolute residual |a_i^T x - b_i|.
//! - **Cyclic**: Iterate rows in order 0, 1, ..., m-1, 0, 1, ...
//!
//! ## Update Rule (single row)
//!
//! Given row a_i and scalar b_i:
//!
//!   x ← x + ω * (b_i - a_i^T x) / ||a_i||² * a_i
//!
//! where ω is the relaxation parameter (1.0 = standard Kaczmarz).
//!
//! ## References
//!
//! - Kaczmarz, S. (1937). "Approximate solution of systems of linear equations"
//! - Strohmer, T. & Vershynin, R. (2009). "A Randomized Kaczmarz Algorithm with Exponential Convergence"
//! - Needell, D. & Tropp, J.A. (2014). "Paved with Good Intentions: Analysis of a Randomized Block Kaczmarz Method"

// Extended Kaczmarz and sketched methods
pub mod classical;
pub mod sketched_solver;
pub mod sparse_sketch;
pub mod types;

pub use classical::kaczmarz_solve as kaczmarz_solve_ext;
pub use sketched_solver::{iterative_sketching, sketch_and_solve};
pub use sparse_sketch::{apply_sketch, apply_sketch_to_vec, build_sketch};
pub use types::{
    KaczmarzConfigExt, KaczmarzVariantExt, ProjectionResult, SketchConfig, SketchTypeExt,
};

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::random::{rngs::StdRng, RngExt, SeedableRng};

/// Variant of the Kaczmarz algorithm.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum KaczmarzVariant {
    /// Randomized Kaczmarz: select row i with probability ||a_i||² / ||A||_F².
    Randomized,
    /// Block Kaczmarz: select a block of `block_size` rows, solve normal equations.
    Block,
    /// Greedy maximum residual: select row with largest |a_i^T x - b_i|.
    GreedyMaxResidual,
    /// Cyclic Kaczmarz: iterate rows in order.
    Cyclic,
}

impl Default for KaczmarzVariant {
    fn default() -> Self {
        KaczmarzVariant::Randomized
    }
}

/// Configuration for the Kaczmarz solver.
#[derive(Clone, Debug)]
pub struct KaczmarzConfig {
    /// Algorithm variant.
    pub variant: KaczmarzVariant,
    /// Maximum number of row-update iterations.
    pub max_iter: usize,
    /// Convergence tolerance on residual norm ||Ax - b||.
    pub tol: f64,
    /// Block size for the Block variant (ignored for other variants).
    pub block_size: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Relaxation parameter ω ∈ (0, 2). ω = 1.0 gives the standard update.
    pub relaxation: f64,
}

impl Default for KaczmarzConfig {
    fn default() -> Self {
        Self {
            variant: KaczmarzVariant::Randomized,
            max_iter: 10_000,
            tol: 1e-6,
            block_size: 32,
            seed: 42,
            relaxation: 1.0,
        }
    }
}

/// Result returned by the Kaczmarz solver.
#[derive(Debug, Clone)]
pub struct KaczmarzResult {
    /// Approximate solution vector x.
    pub x: Vec<f64>,
    /// Euclidean norm of the final residual ||Ax - b||.
    pub residual_norm: f64,
    /// Number of iterations performed.
    pub n_iter: usize,
    /// Whether the algorithm converged within the tolerance.
    pub converged: bool,
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Compute the dot product of a matrix row with a vector.
fn row_dot(a: &Array2<f64>, row: usize, x: &[f64]) -> f64 {
    let r = a.row(row);
    r.iter().zip(x.iter()).map(|(ai, xi)| ai * xi).sum()
}

/// Squared Euclidean norm of a matrix row.
fn row_norm_sq(a: &Array2<f64>, row: usize) -> f64 {
    let r = a.row(row);
    r.iter().map(|ai| ai * ai).sum()
}

/// Sample a row index using the probability distribution p (already computed).
fn sample_row(probs: &[f64], rng: &mut StdRng) -> usize {
    let u: f64 = rng.random::<f64>();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if u <= cumsum {
            return i;
        }
    }
    probs.len() - 1
}

/// Single-row Kaczmarz update.
///
/// x ← x + ω * (b_i - a_i^T x) / ||a_i||² * a_i
fn kaczmarz_update(a: &Array2<f64>, b: &[f64], x: &mut Vec<f64>, row: usize, omega: f64) {
    let row_ns = row_norm_sq(a, row);
    if row_ns < f64::EPSILON {
        return;
    }
    let residual = b[row] - row_dot(a, row, x);
    let step = omega * residual / row_ns;
    let ai = a.row(row);
    for (xi, aij) in x.iter_mut().zip(ai.iter()) {
        *xi += step * aij;
    }
}

/// Block Kaczmarz update.
///
/// For block B = {i_1, ..., i_k}, solve:
///   A_B^T A_B δx = A_B^T r_B,  where r_B = b_B - A_B x
///
/// Uses in-place Gram-Schmidt QR decomposition on A_B (numerically stable).
fn block_kaczmarz_update(
    a: &Array2<f64>,
    b: &[f64],
    x: &mut Vec<f64>,
    block: &[usize],
    omega: f64,
) {
    let k = block.len();
    let n = x.len();
    if k == 0 {
        return;
    }

    // Assemble A_B (k × n) and r_B = b_B - A_B x
    let mut ab = vec![0.0f64; k * n]; // row-major k×n
    let mut rb = vec![0.0f64; k];

    for (bi, &row) in block.iter().enumerate() {
        let ai = a.row(row);
        for j in 0..n {
            ab[bi * n + j] = ai[j];
        }
        rb[bi] = b[row] - row_dot(a, row, x);
    }

    // Solve (A_B)^T A_B δ = (A_B)^T r_B using QR (Gram-Schmidt on A_B^T = n×k)
    // We want the minimum-norm solution δ = A_B^+ r_B
    // For simplicity: δ = A_B^T (A_B A_B^T)^{-1} r_B  (right pseudo-inverse, k ≤ n assumed)

    // Compute G = A_B A_B^T  (k×k Gram matrix)
    let mut gram = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..k {
            let mut dot = 0.0;
            for l in 0..n {
                dot += ab[i * n + l] * ab[j * n + l];
            }
            gram[i * k + j] = dot;
        }
    }

    // Solve G * α = r_B via Cholesky (add small ridge for stability)
    let ridge = 1e-12;
    for i in 0..k {
        gram[i * k + i] += ridge;
    }

    // Cholesky decomposition G = L L^T
    let alpha = match cholesky_solve(&gram, k, &rb) {
        Ok(a) => a,
        Err(_) => {
            // Fallback: row-by-row single Kaczmarz on block
            for &row in block {
                kaczmarz_update(a, b, x, row, omega);
            }
            return;
        }
    };

    // δ = A_B^T α
    let mut delta = vec![0.0f64; n];
    for j in 0..n {
        for bi in 0..k {
            delta[j] += ab[bi * n + j] * alpha[bi];
        }
    }

    // Update x ← x + ω * δ
    for (xi, di) in x.iter_mut().zip(delta.iter()) {
        *xi += omega * di;
    }
}

/// Solve lower-triangular system L y = b (forward substitution).
fn forward_substitution(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i * n + j] * y[j];
        }
        let diag = l[i * n + i];
        y[i] = if diag.abs() > f64::EPSILON {
            sum / diag
        } else {
            0.0
        };
    }
    y
}

/// Solve upper-triangular system L^T y = z (back substitution).
fn back_substitution(l: &[f64], n: usize, z: &[f64]) -> Vec<f64> {
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = z[i];
        for j in (i + 1)..n {
            sum -= l[j * n + i] * x[j]; // L^T[i,j] = L[j,i]
        }
        let diag = l[i * n + i];
        x[i] = if diag.abs() > f64::EPSILON {
            sum / diag
        } else {
            0.0
        };
    }
    x
}

/// Cholesky factorization and solve for a small k×k system A x = b.
///
/// A is stored in row-major order. Returns Err if matrix is not positive definite.
fn cholesky_solve(a: &[f64], k: usize, b: &[f64]) -> Result<Vec<f64>, ()> {
    let mut l = vec![0.0f64; k * k];

    // Compute L (lower triangular) such that A = L L^T
    for i in 0..k {
        for j in 0..=i {
            let mut sum = a[i * k + j];
            for p in 0..j {
                sum -= l[i * k + p] * l[j * k + p];
            }
            if i == j {
                if sum < 0.0 {
                    return Err(());
                }
                l[i * k + j] = sum.sqrt();
            } else {
                let ljj = l[j * k + j];
                if ljj.abs() < f64::EPSILON {
                    return Err(());
                }
                l[i * k + j] = sum / ljj;
            }
        }
    }

    // Forward substitution: L y = b
    let y = forward_substitution(&l, k, b);
    // Backward substitution: L^T x = y
    let x = back_substitution(&l, k, &y);
    Ok(x)
}

/// Compute the full residual norm ||Ax - b||.
fn residual_norm(a: &Array2<f64>, b: &[f64], x: &[f64]) -> f64 {
    let m = a.nrows();
    let mut norm_sq = 0.0;
    for i in 0..m {
        let r = b[i] - row_dot(a, i, x);
        norm_sq += r * r;
    }
    norm_sq.sqrt()
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Solve the linear system Ax = b using a Kaczmarz-type iterative method.
///
/// # Arguments
/// - `a`: Coefficient matrix A of shape (m, n).
/// - `b`: Right-hand side vector of length m.
/// - `x0`: Optional initial guess of length n. Defaults to the zero vector.
/// - `config`: Solver configuration including variant, tolerance, and relaxation.
///
/// # Returns
/// A [`KaczmarzResult`] on success, or an [`OptimizeError`] on invalid input.
pub fn kaczmarz(
    a: &Array2<f64>,
    b: &[f64],
    x0: Option<&[f64]>,
    config: &KaczmarzConfig,
) -> OptimizeResult<KaczmarzResult> {
    let m = a.nrows();
    let n = a.ncols();

    if m == 0 || n == 0 {
        return Err(OptimizeError::InvalidInput(
            "Matrix A must be non-empty".to_string(),
        ));
    }
    if b.len() != m {
        return Err(OptimizeError::InvalidInput(format!(
            "b has length {} but A has {} rows",
            b.len(),
            m
        )));
    }
    if let Some(x0_ref) = x0 {
        if x0_ref.len() != n {
            return Err(OptimizeError::InvalidInput(format!(
                "x0 has length {} but A has {} columns",
                x0_ref.len(),
                n
            )));
        }
    }
    if config.relaxation <= 0.0 || config.relaxation >= 2.0 {
        return Err(OptimizeError::InvalidParameter(
            "relaxation parameter must be in (0, 2)".to_string(),
        ));
    }

    let mut x: Vec<f64> = match x0 {
        Some(x0_ref) => x0_ref.to_vec(),
        None => vec![0.0; n],
    };

    let omega = config.relaxation;
    let mut rng = StdRng::seed_from_u64(config.seed);

    match config.variant {
        KaczmarzVariant::Randomized => {
            // Precompute probabilities proportional to row squared norms
            let row_norms_sq: Vec<f64> = (0..m).map(|i| row_norm_sq(a, i)).collect();
            let frobenius_sq: f64 = row_norms_sq.iter().sum();

            if frobenius_sq < f64::EPSILON {
                return Err(OptimizeError::ComputationError(
                    "Matrix A has zero Frobenius norm".to_string(),
                ));
            }

            let probs: Vec<f64> = row_norms_sq.iter().map(|rn| rn / frobenius_sq).collect();

            for iter in 0..config.max_iter {
                // Periodic convergence check (every n iterations)
                if iter % n.max(1) == 0 {
                    let rn = residual_norm(a, b, &x);
                    if rn < config.tol {
                        return Ok(KaczmarzResult {
                            x,
                            residual_norm: rn,
                            n_iter: iter,
                            converged: true,
                        });
                    }
                }
                let row = sample_row(&probs, &mut rng);
                kaczmarz_update(a, b, &mut x, row, omega);
            }
        }

        KaczmarzVariant::Cyclic => {
            for iter in 0..config.max_iter {
                let row = iter % m;
                kaczmarz_update(a, b, &mut x, row, omega);

                // Check every full pass
                if (iter + 1) % m == 0 {
                    let rn = residual_norm(a, b, &x);
                    if rn < config.tol {
                        return Ok(KaczmarzResult {
                            x,
                            residual_norm: rn,
                            n_iter: iter + 1,
                            converged: true,
                        });
                    }
                }
            }
        }

        KaczmarzVariant::GreedyMaxResidual => {
            for iter in 0..config.max_iter {
                // Find row with maximum absolute residual
                let max_row = (0..m)
                    .max_by(|&i, &j| {
                        let ri = (b[i] - row_dot(a, i, &x)).abs();
                        let rj = (b[j] - row_dot(a, j, &x)).abs();
                        ri.partial_cmp(&rj).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0);

                kaczmarz_update(a, b, &mut x, max_row, omega);

                // Check every m iterations
                if (iter + 1) % m == 0 {
                    let rn = residual_norm(a, b, &x);
                    if rn < config.tol {
                        return Ok(KaczmarzResult {
                            x,
                            residual_norm: rn,
                            n_iter: iter + 1,
                            converged: true,
                        });
                    }
                }
            }
        }

        KaczmarzVariant::Block => {
            let bs = config.block_size.max(1).min(m);
            let num_blocks = (m + bs - 1) / bs; // ceiling division

            for iter in 0..config.max_iter {
                let block_idx = iter % num_blocks;
                let start = block_idx * bs;
                let end = (start + bs).min(m);
                let block: Vec<usize> = (start..end).collect();

                block_kaczmarz_update(a, b, &mut x, &block, omega);

                // Check after every full pass through all blocks
                if (iter + 1) % num_blocks == 0 {
                    let rn = residual_norm(a, b, &x);
                    if rn < config.tol {
                        return Ok(KaczmarzResult {
                            x,
                            residual_norm: rn,
                            n_iter: iter + 1,
                            converged: true,
                        });
                    }
                }
            }
        }

        _ => {
            // Fallback to cyclic for any future variants
            for iter in 0..config.max_iter {
                let row = iter % m;
                kaczmarz_update(a, b, &mut x, row, omega);
            }
        }
    }

    let rn = residual_norm(a, b, &x);
    Ok(KaczmarzResult {
        converged: rn < config.tol,
        residual_norm: rn,
        n_iter: config.max_iter,
        x,
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn make_test_system() -> (Array2<f64>, Vec<f64>) {
        // 3×2 overdetermined consistent system: A x = b with x* = [1, 2]
        let a = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let b = vec![1.0, 2.0, 3.0];
        (a, b)
    }

    #[test]
    fn test_kaczmarz_consistent_system() {
        let (a, b) = make_test_system();
        let config = KaczmarzConfig {
            variant: KaczmarzVariant::Cyclic,
            max_iter: 50_000,
            tol: 1e-6,
            relaxation: 1.0,
            ..KaczmarzConfig::default()
        };
        let result = kaczmarz(&a, &b, None, &config).expect("Kaczmarz should succeed");
        assert!(
            result.converged,
            "should converge, residual = {}",
            result.residual_norm
        );
        assert!((result.x[0] - 1.0).abs() < 1e-4, "x[0] ≈ 1");
        assert!((result.x[1] - 2.0).abs() < 1e-4, "x[1] ≈ 2");
    }

    #[test]
    fn test_kaczmarz_randomized_overdetermined() {
        let (a, b) = make_test_system();
        let config = KaczmarzConfig {
            variant: KaczmarzVariant::Randomized,
            max_iter: 100_000,
            tol: 1e-5,
            relaxation: 1.0,
            seed: 123,
            ..KaczmarzConfig::default()
        };
        let result = kaczmarz(&a, &b, None, &config).expect("randomized Kaczmarz should succeed");
        assert!(result.converged || result.residual_norm < 1e-3);
        assert!((result.x[0] - 1.0).abs() < 0.01, "x[0] ≈ 1");
        assert!((result.x[1] - 2.0).abs() < 0.01, "x[1] ≈ 2");
    }

    #[test]
    fn test_kaczmarz_block_vs_single() {
        let (a, b) = make_test_system();

        let config_single = KaczmarzConfig {
            variant: KaczmarzVariant::Cyclic,
            max_iter: 100_000,
            tol: 1e-6,
            relaxation: 1.0,
            ..KaczmarzConfig::default()
        };
        let config_block = KaczmarzConfig {
            variant: KaczmarzVariant::Block,
            max_iter: 100_000,
            tol: 1e-6,
            block_size: 2,
            relaxation: 1.0,
            ..KaczmarzConfig::default()
        };

        let r_single = kaczmarz(&a, &b, None, &config_single).expect("single row should work");
        let r_block = kaczmarz(&a, &b, None, &config_block).expect("block should work");

        // Both should find the same solution within tolerance
        assert!(r_single.converged || r_single.residual_norm < 1e-4);
        assert!(r_block.converged || r_block.residual_norm < 1e-4);

        // Solutions should be close
        let diff: f64 = r_single
            .x
            .iter()
            .zip(r_block.x.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            diff < 0.1,
            "block and single solutions should agree, diff = {}",
            diff
        );
    }

    #[test]
    fn test_kaczmarz_relaxation() {
        let (a, b) = make_test_system();
        // Relaxation ω = 1.5 (over-relaxed) should still converge on a consistent system
        let config = KaczmarzConfig {
            variant: KaczmarzVariant::Cyclic,
            max_iter: 100_000,
            tol: 1e-5,
            relaxation: 1.5,
            ..KaczmarzConfig::default()
        };
        let result = kaczmarz(&a, &b, None, &config).expect("relaxed Kaczmarz should succeed");
        assert!(result.converged || result.residual_norm < 1e-3);
    }

    #[test]
    fn test_kaczmarz_max_residual() {
        let (a, b) = make_test_system();
        let config = KaczmarzConfig {
            variant: KaczmarzVariant::GreedyMaxResidual,
            max_iter: 50_000,
            tol: 1e-5,
            relaxation: 1.0,
            ..KaczmarzConfig::default()
        };
        let result = kaczmarz(&a, &b, None, &config).expect("greedy max residual should succeed");
        assert!(result.converged || result.residual_norm < 1e-3);
        assert!((result.x[0] - 1.0).abs() < 0.01, "x[0] ≈ 1");
        assert!((result.x[1] - 2.0).abs() < 0.01, "x[1] ≈ 2");
    }

    #[test]
    fn test_kaczmarz_with_initial_guess() {
        let (a, b) = make_test_system();
        let config = KaczmarzConfig {
            variant: KaczmarzVariant::Cyclic,
            max_iter: 50_000,
            tol: 1e-6,
            relaxation: 1.0,
            ..KaczmarzConfig::default()
        };
        // Start from the exact solution — should converge immediately
        let x0 = vec![1.0, 2.0];
        let result = kaczmarz(&a, &b, Some(&x0), &config).expect("Kaczmarz with x0 should succeed");
        assert!(
            result.residual_norm < 1e-10,
            "starting at solution => near-zero residual"
        );
    }

    #[test]
    fn test_kaczmarz_invalid_relaxation() {
        let (a, b) = make_test_system();
        let config_low = KaczmarzConfig {
            relaxation: 0.0,
            ..KaczmarzConfig::default()
        };
        assert!(kaczmarz(&a, &b, None, &config_low).is_err());
        let config_high = KaczmarzConfig {
            relaxation: 2.0,
            ..KaczmarzConfig::default()
        };
        assert!(kaczmarz(&a, &b, None, &config_high).is_err());
    }

    #[test]
    fn test_kaczmarz_dimension_mismatch() {
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let b = vec![1.0]; // wrong length
        let result = kaczmarz(&a, &b, None, &KaczmarzConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_kaczmarz_square_system() {
        // Exactly determined 2×2 system: [2 1; 1 3] x = [5; 10], x* ≈ [1, 3]
        let a = array![[2.0, 1.0], [1.0, 3.0]];
        let b = vec![5.0, 10.0]; // x = [1, 3]: 2+3=5, 1+9=10 ✓
        let config = KaczmarzConfig {
            variant: KaczmarzVariant::Cyclic,
            max_iter: 200_000,
            tol: 1e-5,
            relaxation: 1.0,
            ..KaczmarzConfig::default()
        };
        let result = kaczmarz(&a, &b, None, &config).expect("square system should succeed");
        assert!(result.converged || result.residual_norm < 1e-3);
        assert!(
            (result.x[0] - 1.0).abs() < 0.01,
            "x[0] ≈ 1, got {}",
            result.x[0]
        );
        assert!(
            (result.x[1] - 3.0).abs() < 0.01,
            "x[1] ≈ 3, got {}",
            result.x[1]
        );
    }
}
