//! Classical, Randomized, Block, Greedy, and REK Kaczmarz solvers
//!
//! All variants operate on a dense matrix A (m x n) and vector b (m).
//! The solver iteratively projects onto hyperplanes defined by individual
//! (or blocks of) rows of the system Ax = b.
//!
//! ## References
//!
//! - Kaczmarz, S. (1937). "Approximate solution of systems of linear equations"
//! - Strohmer, T. & Vershynin, R. (2009). "A Randomized Kaczmarz Algorithm
//!   with Exponential Convergence"
//! - Zouzias, A. & Freris, N. (2013). "Randomized Extended Kaczmarz for
//!   Solving Least Squares"

use super::types::{KaczmarzConfigExt, KaczmarzVariantExt, ProjectionResult};
use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::random::{rngs::StdRng, RngExt, SeedableRng};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Dot product of row `i` of a row-major matrix with vector `x`.
fn row_dot(a: &[Vec<f64>], row: usize, x: &[f64]) -> f64 {
    a[row].iter().zip(x.iter()).map(|(ai, xi)| ai * xi).sum()
}

/// Squared L2 norm of row `i`.
fn row_norm_sq(a: &[Vec<f64>], row: usize) -> f64 {
    a[row].iter().map(|v| v * v).sum()
}

/// Squared L2 norm of column `j`.
fn col_norm_sq(a: &[Vec<f64>], col: usize) -> f64 {
    a.iter()
        .map(|row| {
            let v = if col < row.len() { row[col] } else { 0.0 };
            v * v
        })
        .sum()
}

/// Full residual norm ||Ax - b||.
fn full_residual_norm(a: &[Vec<f64>], b: &[f64], x: &[f64]) -> f64 {
    let mut norm_sq = 0.0;
    for (i, row) in a.iter().enumerate() {
        let ax_i: f64 = row.iter().zip(x.iter()).map(|(ai, xi)| ai * xi).sum();
        let r = b[i] - ax_i;
        norm_sq += r * r;
    }
    norm_sq.sqrt()
}

/// Sample a row index with probability proportional to probs[i].
fn sample_weighted(probs: &[f64], rng: &mut StdRng) -> usize {
    let u: f64 = rng.random::<f64>();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if u <= cumsum {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

/// Single-row Kaczmarz update: x <- x + omega * (b_i - a_i^T x) / ||a_i||^2 * a_i
fn single_row_update(a: &[Vec<f64>], b: &[f64], x: &mut [f64], row: usize, omega: f64) {
    let rns = row_norm_sq(a, row);
    if rns < f64::EPSILON {
        return;
    }
    let residual = b[row] - row_dot(a, row, x);
    let step = omega * residual / rns;
    for (xi, &aij) in x.iter_mut().zip(a[row].iter()) {
        *xi += step * aij;
    }
}

/// Cholesky solve for a small k x k system stored row-major. Returns Err on failure.
fn cholesky_solve_small(gram: &mut [f64], k: usize, rhs: &[f64]) -> Result<Vec<f64>, ()> {
    // Add small ridge for numerical stability
    for i in 0..k {
        gram[i * k + i] += 1e-12;
    }
    // Cholesky L L^T = G
    let mut l = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = gram[i * k + j];
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
    // Forward sub: L y = rhs
    let mut y = vec![0.0; k];
    for i in 0..k {
        let mut s = rhs[i];
        for j in 0..i {
            s -= l[i * k + j] * y[j];
        }
        let d = l[i * k + i];
        y[i] = if d.abs() > f64::EPSILON { s / d } else { 0.0 };
    }
    // Back sub: L^T x = y
    let mut result = vec![0.0; k];
    for i in (0..k).rev() {
        let mut s = y[i];
        for j in (i + 1)..k {
            s -= l[j * k + i] * result[j];
        }
        let d = l[i * k + i];
        result[i] = if d.abs() > f64::EPSILON { s / d } else { 0.0 };
    }
    Ok(result)
}

/// Block Kaczmarz update: project onto a block of rows simultaneously.
fn block_update(a: &[Vec<f64>], b: &[f64], x: &mut [f64], block: &[usize], omega: f64) {
    let k = block.len();
    let n = x.len();
    if k == 0 {
        return;
    }

    // Assemble block residual r_B = b_B - A_B x
    let mut r_b = vec![0.0; k];
    for (bi, &row) in block.iter().enumerate() {
        r_b[bi] = b[row] - row_dot(a, row, x);
    }

    // Gram matrix G = A_B A_B^T  (k x k)
    let mut gram = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..=i {
            let dot: f64 = a[block[i]]
                .iter()
                .zip(a[block[j]].iter())
                .map(|(ai, aj)| ai * aj)
                .sum();
            gram[i * k + j] = dot;
            gram[j * k + i] = dot;
        }
    }

    // Solve G alpha = r_B
    let alpha = match cholesky_solve_small(&mut gram, k, &r_b) {
        Ok(a) => a,
        Err(_) => {
            // Fallback: single-row updates
            for &row in block {
                single_row_update(a, b, x, row, omega);
            }
            return;
        }
    };

    // delta = A_B^T alpha
    let mut delta = vec![0.0; n];
    for j in 0..n {
        for bi in 0..k {
            if j < a[block[bi]].len() {
                delta[j] += a[block[bi]][j] * alpha[bi];
            }
        }
    }

    // x <- x + omega * delta
    for (xi, &di) in x.iter_mut().zip(delta.iter()) {
        *xi += omega * di;
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Solve the linear system Ax = b using a Kaczmarz-type iterative method.
///
/// Supports Classical (cyclic), Randomized, Block, Greedy, and REK variants.
///
/// # Arguments
/// - `a`: Coefficient matrix given as a slice of row vectors (m rows, each of length n).
/// - `b`: Right-hand side vector of length m.
/// - `config`: Solver configuration.
///
/// # Returns
/// A [`ProjectionResult`] on success, or an [`OptimizeError`] on invalid input.
pub fn kaczmarz_solve(
    a: &[Vec<f64>],
    b: &[f64],
    config: &KaczmarzConfigExt,
) -> OptimizeResult<ProjectionResult> {
    let m = a.len();
    if m == 0 {
        return Err(OptimizeError::InvalidInput(
            "Matrix A must be non-empty".to_string(),
        ));
    }
    let n = a[0].len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput(
            "Matrix A must have at least one column".to_string(),
        ));
    }
    if b.len() != m {
        return Err(OptimizeError::InvalidInput(format!(
            "b has length {} but A has {} rows",
            b.len(),
            m
        )));
    }
    if config.relaxation <= 0.0 || config.relaxation >= 2.0 {
        return Err(OptimizeError::InvalidParameter(
            "relaxation must be in (0, 2)".to_string(),
        ));
    }

    let omega = config.relaxation;
    let mut x = vec![0.0; n];
    let mut rng = StdRng::seed_from_u64(config.seed);

    match config.variant {
        KaczmarzVariantExt::Classical => {
            classical_kaczmarz(a, b, &mut x, omega, config.max_iter, config.tol)
        }
        KaczmarzVariantExt::Randomized => {
            randomized_kaczmarz(a, b, &mut x, omega, config.max_iter, config.tol, &mut rng)
        }
        KaczmarzVariantExt::Block => block_kaczmarz(
            a,
            b,
            &mut x,
            omega,
            config.max_iter,
            config.tol,
            config.block_size,
        ),
        KaczmarzVariantExt::Greedy => {
            greedy_kaczmarz(a, b, &mut x, omega, config.max_iter, config.tol)
        }
        KaczmarzVariantExt::REK => {
            rek_kaczmarz(a, b, &mut x, omega, config.max_iter, config.tol, &mut rng)
        }
        _ => {
            // Fallback to classical for future variants
            classical_kaczmarz(a, b, &mut x, omega, config.max_iter, config.tol)
        }
    }
}

// ---------------------------------------------------------------------------
// Variant implementations
// ---------------------------------------------------------------------------

fn classical_kaczmarz(
    a: &[Vec<f64>],
    b: &[f64],
    x: &mut Vec<f64>,
    omega: f64,
    max_iter: usize,
    tol: f64,
) -> OptimizeResult<ProjectionResult> {
    let m = a.len();
    for iter in 0..max_iter {
        let row = iter % m;
        single_row_update(a, b, x, row, omega);

        // Check convergence every full pass
        if (iter + 1) % m == 0 {
            let rn = full_residual_norm(a, b, x);
            if rn < tol {
                return Ok(ProjectionResult {
                    solution: x.clone(),
                    residual_norm: rn,
                    iterations: iter + 1,
                    converged: true,
                });
            }
        }
    }
    let rn = full_residual_norm(a, b, x);
    Ok(ProjectionResult {
        converged: rn < tol,
        residual_norm: rn,
        iterations: max_iter,
        solution: x.clone(),
    })
}

fn randomized_kaczmarz(
    a: &[Vec<f64>],
    b: &[f64],
    x: &mut Vec<f64>,
    omega: f64,
    max_iter: usize,
    tol: f64,
    rng: &mut StdRng,
) -> OptimizeResult<ProjectionResult> {
    let m = a.len();
    let n = x.len();

    // Precompute row probabilities
    let row_norms_sq: Vec<f64> = (0..m).map(|i| row_norm_sq(a, i)).collect();
    let frobenius_sq: f64 = row_norms_sq.iter().sum();
    if frobenius_sq < f64::EPSILON {
        return Err(OptimizeError::ComputationError(
            "Matrix A has zero Frobenius norm".to_string(),
        ));
    }
    let probs: Vec<f64> = row_norms_sq.iter().map(|rn| rn / frobenius_sq).collect();

    for iter in 0..max_iter {
        if iter % n.max(1) == 0 {
            let rn = full_residual_norm(a, b, x);
            if rn < tol {
                return Ok(ProjectionResult {
                    solution: x.clone(),
                    residual_norm: rn,
                    iterations: iter,
                    converged: true,
                });
            }
        }
        let row = sample_weighted(&probs, rng);
        single_row_update(a, b, x, row, omega);
    }
    let rn = full_residual_norm(a, b, x);
    Ok(ProjectionResult {
        converged: rn < tol,
        residual_norm: rn,
        iterations: max_iter,
        solution: x.clone(),
    })
}

fn block_kaczmarz(
    a: &[Vec<f64>],
    b: &[f64],
    x: &mut Vec<f64>,
    omega: f64,
    max_iter: usize,
    tol: f64,
    block_size: usize,
) -> OptimizeResult<ProjectionResult> {
    let m = a.len();
    let bs = block_size.max(1).min(m);
    let num_blocks = (m + bs - 1) / bs;

    for iter in 0..max_iter {
        let block_idx = iter % num_blocks;
        let start = block_idx * bs;
        let end = (start + bs).min(m);
        let block: Vec<usize> = (start..end).collect();

        block_update(a, b, x, &block, omega);

        // Check after each full pass
        if (iter + 1) % num_blocks == 0 {
            let rn = full_residual_norm(a, b, x);
            if rn < tol {
                return Ok(ProjectionResult {
                    solution: x.clone(),
                    residual_norm: rn,
                    iterations: iter + 1,
                    converged: true,
                });
            }
        }
    }
    let rn = full_residual_norm(a, b, x);
    Ok(ProjectionResult {
        converged: rn < tol,
        residual_norm: rn,
        iterations: max_iter,
        solution: x.clone(),
    })
}

fn greedy_kaczmarz(
    a: &[Vec<f64>],
    b: &[f64],
    x: &mut Vec<f64>,
    omega: f64,
    max_iter: usize,
    tol: f64,
) -> OptimizeResult<ProjectionResult> {
    let m = a.len();

    for iter in 0..max_iter {
        // Find row with maximum absolute residual
        let mut max_res = 0.0f64;
        let mut max_row = 0usize;
        for i in 0..m {
            let r = (b[i] - row_dot(a, i, x)).abs();
            if r > max_res {
                max_res = r;
                max_row = i;
            }
        }
        single_row_update(a, b, x, max_row, omega);

        // Periodic convergence check
        if (iter + 1) % m == 0 {
            let rn = full_residual_norm(a, b, x);
            if rn < tol {
                return Ok(ProjectionResult {
                    solution: x.clone(),
                    residual_norm: rn,
                    iterations: iter + 1,
                    converged: true,
                });
            }
        }
    }
    let rn = full_residual_norm(a, b, x);
    Ok(ProjectionResult {
        converged: rn < tol,
        residual_norm: rn,
        iterations: max_iter,
        solution: x.clone(),
    })
}

/// Randomized Extended Kaczmarz (REK).
///
/// Alternates between:
///  1. Column projection: z <- z - (<a_j, z> / ||a_j||^2) * a_j
///     (removes component of z in column space of A)
///  2. Row projection: x <- x + omega * ((b_i - z_i) - a_i^T x) / ||a_i||^2 * a_i
///     (standard Kaczmarz with corrected RHS)
///
/// This converges to the least-squares solution even when Ax = b is inconsistent.
fn rek_kaczmarz(
    a: &[Vec<f64>],
    b: &[f64],
    x: &mut Vec<f64>,
    omega: f64,
    max_iter: usize,
    tol: f64,
    rng: &mut StdRng,
) -> OptimizeResult<ProjectionResult> {
    let m = a.len();
    let n = x.len();

    // z tracks the projection of b onto the left null space of A
    let mut z: Vec<f64> = b.to_vec();

    // Precompute row and column norms
    let row_norms_sq: Vec<f64> = (0..m).map(|i| row_norm_sq(a, i)).collect();
    let col_norms_sq: Vec<f64> = (0..n).map(|j| col_norm_sq(a, j)).collect();

    let frobenius_sq_row: f64 = row_norms_sq.iter().sum();
    let frobenius_sq_col: f64 = col_norms_sq.iter().sum();

    if frobenius_sq_row < f64::EPSILON {
        return Err(OptimizeError::ComputationError(
            "Matrix A has zero Frobenius norm".to_string(),
        ));
    }

    let row_probs: Vec<f64> = row_norms_sq
        .iter()
        .map(|rn| rn / frobenius_sq_row)
        .collect();
    let col_probs: Vec<f64> = if frobenius_sq_col > f64::EPSILON {
        col_norms_sq
            .iter()
            .map(|cn| cn / frobenius_sq_col)
            .collect()
    } else {
        vec![1.0 / n as f64; n]
    };

    for iter in 0..max_iter {
        // Step 1: Column projection on z
        let col_idx = sample_weighted(&col_probs, rng);
        let cns = col_norms_sq[col_idx];
        if cns > f64::EPSILON {
            // Compute <a_{:, col_idx}, z>
            let dot_val: f64 = a
                .iter()
                .zip(z.iter())
                .map(|(row, &zi)| {
                    let aij = if col_idx < row.len() {
                        row[col_idx]
                    } else {
                        0.0
                    };
                    aij * zi
                })
                .sum();
            let scale = dot_val / cns;
            for (i, row) in a.iter().enumerate() {
                let aij = if col_idx < row.len() {
                    row[col_idx]
                } else {
                    0.0
                };
                z[i] -= scale * aij;
            }
        }

        // Step 2: Row projection with corrected RHS (b - z)
        let row_idx = sample_weighted(&row_probs, rng);
        let rns = row_norms_sq[row_idx];
        if rns > f64::EPSILON {
            let target = b[row_idx] - z[row_idx];
            let dot_ax = row_dot(a, row_idx, x);
            let residual_i = target - dot_ax;
            let step = omega * residual_i / rns;
            for (xi, &aij) in x.iter_mut().zip(a[row_idx].iter()) {
                *xi += step * aij;
            }
        }

        // Periodic convergence check
        if (iter + 1) % (m.max(n)) == 0 {
            let rn = full_residual_norm(a, b, x);
            if rn < tol {
                return Ok(ProjectionResult {
                    solution: x.clone(),
                    residual_norm: rn,
                    iterations: iter + 1,
                    converged: true,
                });
            }
        }
    }

    let rn = full_residual_norm(a, b, x);
    Ok(ProjectionResult {
        converged: rn < tol,
        residual_norm: rn,
        iterations: max_iter,
        solution: x.clone(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_consistent_system() -> (Vec<Vec<f64>>, Vec<f64>) {
        // 3x2 overdetermined consistent system: x* = [1, 2]
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let b = vec![1.0, 2.0, 3.0];
        (a, b)
    }

    fn make_inconsistent_system() -> (Vec<Vec<f64>>, Vec<f64>) {
        // Overdetermined inconsistent: 1+2 != 4
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let b = vec![1.0, 2.0, 4.0];
        (a, b)
    }

    #[test]
    fn test_classical_kaczmarz_converges() {
        let (a, b) = make_consistent_system();
        let config = KaczmarzConfigExt {
            variant: KaczmarzVariantExt::Classical,
            max_iter: 50_000,
            tol: 1e-5,
            ..Default::default()
        };
        let result = kaczmarz_solve(&a, &b, &config).expect("should succeed");
        assert!(result.converged, "residual = {}", result.residual_norm);
        assert!((result.solution[0] - 1.0).abs() < 1e-3);
        assert!((result.solution[1] - 2.0).abs() < 1e-3);
    }

    #[test]
    fn test_randomized_kaczmarz_converges() {
        let (a, b) = make_consistent_system();
        let config = KaczmarzConfigExt {
            variant: KaczmarzVariantExt::Randomized,
            max_iter: 100_000,
            tol: 1e-5,
            seed: 123,
            ..Default::default()
        };
        let result = kaczmarz_solve(&a, &b, &config).expect("should succeed");
        assert!(result.converged || result.residual_norm < 1e-3);
        assert!((result.solution[0] - 1.0).abs() < 0.01);
        assert!((result.solution[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_block_kaczmarz_residual_below_tol() {
        let (a, b) = make_consistent_system();
        let config = KaczmarzConfigExt {
            variant: KaczmarzVariantExt::Block,
            max_iter: 50_000,
            tol: 1e-5,
            block_size: 2,
            ..Default::default()
        };
        let result = kaczmarz_solve(&a, &b, &config).expect("should succeed");
        assert!(
            result.residual_norm < 1e-3,
            "residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn test_greedy_fewer_iters_than_classical() {
        let (a, b) = make_consistent_system();
        let tol = 1e-5;
        let config_classical = KaczmarzConfigExt {
            variant: KaczmarzVariantExt::Classical,
            max_iter: 200_000,
            tol,
            ..Default::default()
        };
        let config_greedy = KaczmarzConfigExt {
            variant: KaczmarzVariantExt::Greedy,
            max_iter: 200_000,
            tol,
            ..Default::default()
        };
        let r_classical = kaczmarz_solve(&a, &b, &config_classical).expect("classical");
        let r_greedy = kaczmarz_solve(&a, &b, &config_greedy).expect("greedy");

        // Both should converge
        assert!(r_classical.converged, "classical didn't converge");
        assert!(r_greedy.converged, "greedy didn't converge");

        // Greedy should use fewer or equal iterations
        assert!(
            r_greedy.iterations <= r_classical.iterations,
            "greedy={} classical={}",
            r_greedy.iterations,
            r_classical.iterations
        );
    }

    #[test]
    fn test_rek_handles_inconsistent_systems() {
        let (a, b) = make_inconsistent_system();
        // Least-squares solution: x* = (A^T A)^{-1} A^T b
        // A^T A = [[2,1],[1,2]], A^T b = [5,6]
        // x* = [4/3, 7/3]
        let config = KaczmarzConfigExt {
            variant: KaczmarzVariantExt::REK,
            max_iter: 200_000,
            tol: 1e-4,
            seed: 42,
            relaxation: 0.8,
            ..Default::default()
        };
        let result = kaczmarz_solve(&a, &b, &config).expect("REK should succeed");

        // REK should approximate the LS solution
        let expected_x0 = 4.0 / 3.0;
        let expected_x1 = 7.0 / 3.0;
        assert!(
            (result.solution[0] - expected_x0).abs() < 0.5,
            "x[0]={}, expected ~{}",
            result.solution[0],
            expected_x0
        );
        assert!(
            (result.solution[1] - expected_x1).abs() < 0.5,
            "x[1]={}, expected ~{}",
            result.solution[1],
            expected_x1
        );
    }

    #[test]
    fn test_invalid_relaxation() {
        let (a, b) = make_consistent_system();
        let config = KaczmarzConfigExt {
            relaxation: 0.0,
            ..Default::default()
        };
        assert!(kaczmarz_solve(&a, &b, &config).is_err());
        let config2 = KaczmarzConfigExt {
            relaxation: 2.0,
            ..Default::default()
        };
        assert!(kaczmarz_solve(&a, &b, &config2).is_err());
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![1.0]; // wrong length
        assert!(kaczmarz_solve(&a, &b, &KaczmarzConfigExt::default()).is_err());
    }

    #[test]
    fn test_square_identity_system() {
        let a = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let b = vec![3.0, 5.0, 7.0];
        let config = KaczmarzConfigExt {
            variant: KaczmarzVariantExt::Classical,
            max_iter: 100,
            tol: 1e-10,
            ..Default::default()
        };
        let result = kaczmarz_solve(&a, &b, &config).expect("identity system");
        assert!(result.converged);
        assert!((result.solution[0] - 3.0).abs() < 1e-8);
        assert!((result.solution[1] - 5.0).abs() < 1e-8);
        assert!((result.solution[2] - 7.0).abs() < 1e-8);
    }
}
