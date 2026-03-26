//! Sketch-based solvers for linear systems
//!
//! Two approaches:
//! 1. **Sketch-and-solve**: form the sketched system SA * x = Sb, then solve the
//!    smaller system directly via least-squares (QR or normal equations).
//! 2. **Iterative sketching**: use the sketch as a preconditioner in an iterative
//!    refinement loop, achieving progressively smaller residuals.
//!
//! ## References
//!
//! - Woodruff, D.P. (2014). "Sketching as a Tool for Numerical Linear Algebra"
//! - Pilanci, M. & Wainwright, M.J. (2016). "Iterative Hessian Sketch"

use super::sparse_sketch::{apply_sketch, build_sketch};
use super::types::{ProjectionResult, SketchConfig};
use crate::error::{OptimizeError, OptimizeResult};

/// Solve Ax = b by forming the sketched system SA x = Sb and solving the
/// smaller (sketch_size x n) least-squares problem directly.
///
/// # Arguments
/// - `a`: m x n matrix (slice of row vectors).
/// - `b`: right-hand side vector of length m.
/// - `config`: sketch configuration.
///
/// # Returns
/// A [`ProjectionResult`] with the approximate solution.
pub fn sketch_and_solve(
    a: &[Vec<f64>],
    b: &[f64],
    config: &SketchConfig,
) -> OptimizeResult<ProjectionResult> {
    let m = a.len();
    if m == 0 {
        return Err(OptimizeError::InvalidInput(
            "Matrix A must be non-empty".into(),
        ));
    }
    let n = a[0].len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput(
            "A must have at least one column".into(),
        ));
    }
    if b.len() != m {
        return Err(OptimizeError::InvalidInput(format!(
            "b has length {} but A has {} rows",
            b.len(),
            m
        )));
    }

    let s = config.sketch_size.min(m).max(1);

    // Build sketch matrix and apply
    let sketch = build_sketch(&config.sketch_type, s, m, config.seed, config)?;
    let sa = apply_sketch(&sketch, s, a)?;
    let sb = apply_sketch_vec(&sketch, s, b, m);

    // Now solve the s x n least-squares system SA x = Sb via normal equations:
    // (SA)^T (SA) x = (SA)^T Sb
    let x = solve_normal_equations(&sa, &sb, n)?;

    // Compute full residual
    let residual_norm = full_residual_norm(a, b, &x);

    Ok(ProjectionResult {
        solution: x,
        residual_norm,
        iterations: 1,
        converged: true, // single-shot method
    })
}

/// Iterative sketching: repeatedly solve a sketched problem and refine.
///
/// At each iteration:
///   1. Compute residual r = b - A x
///   2. Sketch the residual system: SA delta = Sr
///   3. Solve the small system for delta
///   4. Update x <- x + delta
///
/// # Arguments
/// - `a`: m x n matrix (slice of row vectors).
/// - `b`: right-hand side vector of length m.
/// - `config`: sketch configuration (uses max_iter and tol).
///
/// # Returns
/// A [`ProjectionResult`] with the refined solution.
pub fn iterative_sketching(
    a: &[Vec<f64>],
    b: &[f64],
    config: &SketchConfig,
) -> OptimizeResult<ProjectionResult> {
    let m = a.len();
    if m == 0 {
        return Err(OptimizeError::InvalidInput(
            "Matrix A must be non-empty".into(),
        ));
    }
    let n = a[0].len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput(
            "A must have at least one column".into(),
        ));
    }
    if b.len() != m {
        return Err(OptimizeError::InvalidInput(format!(
            "b has length {} but A has {} rows",
            b.len(),
            m
        )));
    }

    let s = config.sketch_size.min(m).max(1);
    let mut x = vec![0.0; n];
    let mut seed = config.seed;

    for iter in 0..config.max_iter {
        // Compute residual r = b - Ax
        let r = compute_residual(a, b, &x);
        let rn = l2_norm(&r);
        if rn < config.tol {
            return Ok(ProjectionResult {
                solution: x,
                residual_norm: rn,
                iterations: iter,
                converged: true,
            });
        }

        // Build a fresh sketch each iteration for better convergence
        let sketch = build_sketch(&config.sketch_type, s, m, seed, config)?;
        let sa = apply_sketch(&sketch, s, a)?;
        let sr = apply_sketch_vec(&sketch, s, &r, m);

        // Solve sketched system: SA delta = Sr
        match solve_normal_equations(&sa, &sr, n) {
            Ok(delta) => {
                for (xi, di) in x.iter_mut().zip(delta.iter()) {
                    *xi += di;
                }
            }
            Err(_) => {
                // If the sketched system is singular, just skip this iteration
            }
        }

        // Vary seed to get different sketches
        seed = seed.wrapping_add(1);
    }

    let rn = full_residual_norm(a, b, &x);
    Ok(ProjectionResult {
        converged: rn < config.tol,
        residual_norm: rn,
        iterations: config.max_iter,
        solution: x,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Apply sketch to a vector: result[k] = sum_i sketch[k*m + i] * v[i]
fn apply_sketch_vec(sketch: &[f64], s: usize, v: &[f64], m: usize) -> Vec<f64> {
    let mut result = vec![0.0; s];
    for k in 0..s {
        let mut val = 0.0;
        for i in 0..m.min(v.len()) {
            val += sketch[k * m + i] * v[i];
        }
        result[k] = val;
    }
    result
}

/// Solve (SA)^T (SA) x = (SA)^T Sb via Cholesky on the n x n normal equations.
fn solve_normal_equations(sa: &[Vec<f64>], sb: &[f64], n: usize) -> OptimizeResult<Vec<f64>> {
    let s = sa.len();
    if s == 0 || n == 0 {
        return Err(OptimizeError::ComputationError(
            "Empty sketched system".to_string(),
        ));
    }

    // Form A^T A (n x n) and A^T b (n)
    let mut ata = vec![0.0; n * n];
    let mut atb = vec![0.0; n];

    for k in 0..s {
        for i in 0..n {
            let ai = if i < sa[k].len() { sa[k][i] } else { 0.0 };
            atb[i] += ai * sb[k];
            for j in 0..=i {
                let aj = if j < sa[k].len() { sa[k][j] } else { 0.0 };
                ata[i * n + j] += ai * aj;
            }
        }
    }
    // Symmetrize
    for i in 0..n {
        for j in (i + 1)..n {
            ata[i * n + j] = ata[j * n + i];
        }
    }

    // Cholesky with ridge
    let ridge = 1e-10;
    for i in 0..n {
        ata[i * n + i] += ridge;
    }

    cholesky_solve(&ata, n, &atb)
}

/// Cholesky factorization and solve for an n x n system.
fn cholesky_solve(a: &[f64], n: usize, b: &[f64]) -> OptimizeResult<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for p in 0..j {
                sum -= l[i * n + p] * l[j * n + p];
            }
            if i == j {
                if sum < 0.0 {
                    return Err(OptimizeError::ComputationError(
                        "Cholesky failed: not positive definite".to_string(),
                    ));
                }
                l[i * n + j] = sum.sqrt();
            } else {
                let ljj = l[j * n + j];
                if ljj.abs() < f64::EPSILON {
                    return Err(OptimizeError::ComputationError(
                        "Cholesky failed: zero diagonal".to_string(),
                    ));
                }
                l[i * n + j] = sum / ljj;
            }
        }
    }

    // Forward sub: L y = b
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * y[j];
        }
        let d = l[i * n + i];
        y[i] = if d.abs() > f64::EPSILON { s / d } else { 0.0 };
    }
    // Back sub: L^T x = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in (i + 1)..n {
            s -= l[j * n + i] * x[j];
        }
        let d = l[i * n + i];
        x[i] = if d.abs() > f64::EPSILON { s / d } else { 0.0 };
    }
    Ok(x)
}

fn compute_residual(a: &[Vec<f64>], b: &[f64], x: &[f64]) -> Vec<f64> {
    a.iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let ax_i: f64 = row.iter().zip(x.iter()).map(|(ai, xi)| ai * xi).sum();
            bi - ax_i
        })
        .collect()
}

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|vi| vi * vi).sum::<f64>().sqrt()
}

fn full_residual_norm(a: &[Vec<f64>], b: &[f64], x: &[f64]) -> f64 {
    l2_norm(&compute_residual(a, b, x))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::types::SketchTypeExt;
    use super::*;

    fn make_consistent_system() -> (Vec<Vec<f64>>, Vec<f64>) {
        // 6x2 overdetermined consistent: x* = [1, 2]
        let a = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 0.0],
            vec![0.0, 2.0],
            vec![1.0, -1.0],
        ];
        let b: Vec<f64> = a.iter().map(|row| row[0] * 1.0 + row[1] * 2.0).collect();
        (a, b)
    }

    #[test]
    fn test_sketch_and_solve_gaussian() {
        let (a, b) = make_consistent_system();
        let config = SketchConfig {
            sketch_type: SketchTypeExt::Gaussian,
            sketch_size: 5,
            seed: 42,
            ..Default::default()
        };
        let result = sketch_and_solve(&a, &b, &config).expect("should succeed");
        // Check relative error
        let x_exact = vec![1.0, 2.0];
        let error: f64 = result
            .solution
            .iter()
            .zip(x_exact.iter())
            .map(|(xi, xe)| (xi - xe).powi(2))
            .sum::<f64>()
            .sqrt();
        let norm_exact: f64 = x_exact.iter().map(|v| v * v).sum::<f64>().sqrt();
        let rel_error = error / norm_exact;
        assert!(
            rel_error < 0.5,
            "relative error = {} (solution = {:?})",
            rel_error,
            result.solution
        );
    }

    #[test]
    fn test_sketch_and_solve_count_sketch() {
        let (a, b) = make_consistent_system();
        let config = SketchConfig {
            sketch_type: SketchTypeExt::CountSketch,
            sketch_size: 5,
            seed: 77,
            ..Default::default()
        };
        let result = sketch_and_solve(&a, &b, &config).expect("should succeed");
        // Looser tolerance for count sketch
        assert!(
            result.residual_norm < 5.0,
            "residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn test_iterative_sketching_residual_decreases() {
        let (a, b) = make_consistent_system();
        let config = SketchConfig {
            sketch_type: SketchTypeExt::Gaussian,
            sketch_size: 4,
            seed: 42,
            max_iter: 50,
            tol: 1e-8,
            ..Default::default()
        };
        let result = iterative_sketching(&a, &b, &config).expect("should succeed");
        assert!(
            result.residual_norm < 1e-4,
            "residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn test_sketch_and_solve_srht() {
        let (a, b) = make_consistent_system();
        let config = SketchConfig {
            sketch_type: SketchTypeExt::SRHT,
            sketch_size: 4,
            seed: 42,
            ..Default::default()
        };
        let result = sketch_and_solve(&a, &b, &config).expect("SRHT should work");
        // SRHT may have wider tolerance
        assert!(
            result.residual_norm < 5.0,
            "residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn test_sketch_and_solve_sparse_jl() {
        let (a, b) = make_consistent_system();
        let config = SketchConfig {
            sketch_type: SketchTypeExt::SparseJL,
            sketch_size: 5,
            seed: 42,
            sparse_jl_sparsity: 3,
            ..Default::default()
        };
        let result = sketch_and_solve(&a, &b, &config).expect("SparseJL should work");
        assert!(
            result.residual_norm < 5.0,
            "residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn test_iterative_sketching_identity() {
        // Identity system: should converge quickly
        let a = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let b = vec![3.0, 5.0, 7.0];
        let config = SketchConfig {
            sketch_type: SketchTypeExt::Gaussian,
            sketch_size: 3,
            seed: 42,
            max_iter: 50,
            tol: 1e-8,
            ..Default::default()
        };
        let result = iterative_sketching(&a, &b, &config).expect("identity");
        assert!(result.converged, "residual = {}", result.residual_norm);
        for (i, (&xi, &bi)) in result.solution.iter().zip(b.iter()).enumerate() {
            assert!((xi - bi).abs() < 0.1, "x[{}] = {}, expected {}", i, xi, bi);
        }
    }

    #[test]
    fn test_invalid_input() {
        let a: Vec<Vec<f64>> = vec![];
        let b: Vec<f64> = vec![];
        assert!(sketch_and_solve(&a, &b, &SketchConfig::default()).is_err());
    }
}
