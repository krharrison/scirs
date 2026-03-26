//! Sketched Gradient Descent for Large-Scale Least-Squares
//!
//! Randomized sketching compresses the m×n system (m rows, n columns) into a
//! sketch of dimension (sketch_dim × n), dramatically reducing memory and compute
//! requirements when m is very large.
//!
//! ## Algorithm
//!
//! At each iteration t:
//! 1. Draw sketch matrix S_t of shape (sketch_dim × m).
//! 2. Form sketched system: Ã = S A, b̃ = S b.
//! 3. Compute sketched gradient: g = Ã^T (Ã x - b̃).
//! 4. Update: x ← x - α g.
//!
//! ## Sketch Types
//!
//! - **Gaussian**: Each entry of S is drawn i.i.d. N(0, 1/sketch_dim).
//! - **Hadamard** (SRHT): Structured random Hadamard transform — applied as Walsh-Hadamard
//!   transform followed by random sign-flips and row sampling.
//! - **Uniform** (Rademacher): Entries are uniform ±1 / √sketch_dim.
//! - **CountSketch**: Each column has exactly one non-zero entry ±1. Very sparse and fast.
//!
//! ## References
//!
//! - Mahoney, M.W. (2011). "Randomized Algorithms for Matrices and Data"
//! - Woodruff, D.P. (2014). "Sketching as a Tool for Numerical Linear Algebra"
//! - Drineas, P. et al. (2011). "Faster Least Squares Approximation"

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::Array2;
use scirs2_core::random::{rngs::StdRng, RngExt, SeedableRng};

/// Type of sketch matrix to use for dimensionality reduction.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum SketchType {
    /// Dense Gaussian sketch: S_{ij} ~ N(0, 1/sketch_dim).
    Gaussian,
    /// Subsampled Randomized Hadamard Transform (SRHT).
    Hadamard,
    /// Rademacher (uniform ±1/√sketch_dim) sketch.
    Uniform,
    /// Count sketch: each column of S has exactly one non-zero ±1 entry.
    CountSketch,
}

impl Default for SketchType {
    fn default() -> Self {
        SketchType::Gaussian
    }
}

/// Configuration for sketched least-squares solver.
#[derive(Clone, Debug)]
pub struct SketchedLeastSquaresConfig {
    /// Sketch dimension m_s (number of rows in the sketch matrix, m_s << m).
    pub sketch_dim: usize,
    /// Type of random sketch.
    pub sketch_type: SketchType,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the relative change in x: ||x_{t+1} - x_t|| / (1 + ||x_t||).
    pub tol: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Whether to refresh the sketch at every iteration (recommended for accuracy).
    pub refresh_sketch: bool,
    /// Step size (learning rate) for gradient updates.
    /// When `None`, uses the sketch-based Lipschitz estimate.
    pub step_size: Option<f64>,
}

impl Default for SketchedLeastSquaresConfig {
    fn default() -> Self {
        Self {
            sketch_dim: 512,
            sketch_type: SketchType::Gaussian,
            max_iter: 100,
            tol: 1e-6,
            seed: 42,
            refresh_sketch: true,
            step_size: None,
        }
    }
}

/// Result of sketched least-squares optimization.
#[derive(Debug, Clone)]
pub struct LsqResult {
    /// Approximate solution x minimizing ||Ax - b||².
    pub x: Vec<f64>,
    /// Euclidean norm of the final residual ||Ax - b||.
    pub residual_norm: f64,
    /// Number of iterations performed.
    pub n_iter: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
}

// ─── Sketch construction helpers ─────────────────────────────────────────────

/// Build a Gaussian sketch matrix S of shape (sketch_dim × m).
///
/// Entries S_{ij} ~ N(0, 1/sketch_dim) using Box-Muller transform.
fn build_gaussian_sketch(sketch_dim: usize, m: usize, rng: &mut StdRng) -> Vec<f64> {
    let scale = (1.0 / sketch_dim as f64).sqrt();
    let mut s = Vec::with_capacity(sketch_dim * m);
    // Box-Muller: generate pairs of standard normals
    let mut spare: Option<f64> = None;
    for _ in 0..(sketch_dim * m) {
        let v = match spare.take() {
            Some(z) => z,
            None => {
                // Box-Muller
                loop {
                    let u: f64 = rng.random::<f64>();
                    let v: f64 = rng.random::<f64>();
                    if u > 0.0 {
                        let mag = (-2.0 * u.ln()).sqrt();
                        let angle = std::f64::consts::TAU * v;
                        spare = Some(mag * angle.sin());
                        break mag * angle.cos();
                    }
                }
            }
        };
        s.push(v * scale);
    }
    s
}

/// Build a Rademacher sketch matrix S of shape (sketch_dim × m).
///
/// Entries are uniform ±1 / √sketch_dim.
fn build_rademacher_sketch(sketch_dim: usize, m: usize, rng: &mut StdRng) -> Vec<f64> {
    let scale = 1.0 / (sketch_dim as f64).sqrt();
    (0..sketch_dim * m)
        .map(|_| if rng.random::<bool>() { scale } else { -scale })
        .collect()
}

/// Build a Count sketch matrix of shape (sketch_dim × m).
///
/// Each column j has exactly one non-zero entry at a random row h(j) with sign σ(j) ∈ {±1}.
fn build_count_sketch(sketch_dim: usize, m: usize, rng: &mut StdRng) -> Vec<f64> {
    let mut s = vec![0.0f64; sketch_dim * m];
    for j in 0..m {
        let row = rng.random_range(0..sketch_dim);
        let sign: f64 = if rng.random::<bool>() { 1.0 } else { -1.0 };
        s[row * m + j] = sign;
    }
    s
}

/// Apply the Walsh-Hadamard transform to a slice in-place (length must be a power of 2).
fn walsh_hadamard_transform(x: &mut [f64]) {
    let n = x.len();
    if n <= 1 {
        return;
    }
    // Cooley-Tukey style
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..(i + h) {
                let u = x[j];
                let v = x[j + h];
                x[j] = u + v;
                x[j + h] = u - v;
            }
        }
        h <<= 1;
    }
    // Normalize by 1/sqrt(n)
    let inv_sqrt_n = 1.0 / (n as f64).sqrt();
    for xi in x.iter_mut() {
        *xi *= inv_sqrt_n;
    }
}

/// Build a SRHT sketch matrix of shape (sketch_dim × m).
///
/// Applies D (random sign flips), then H (Hadamard), then samples sketch_dim rows.
/// m is padded to the next power of 2 if needed.
fn build_hadamard_sketch(sketch_dim: usize, m: usize, rng: &mut StdRng) -> (Vec<f64>, usize) {
    // Pad m to next power of 2
    let m_pad = m.next_power_of_two();
    let scale = (m_pad as f64 / sketch_dim as f64).sqrt() / (m_pad as f64).sqrt();

    // Random diagonal sign matrix D (m_pad entries)
    let signs: Vec<f64> = (0..m_pad)
        .map(|_| if rng.random::<bool>() { 1.0 } else { -1.0 })
        .collect();

    // Random row-selection permutation (sample sketch_dim rows without replacement from m_pad)
    let mut perm: Vec<usize> = (0..m_pad).collect();
    // Fisher-Yates partial shuffle for first sketch_dim elements
    for i in 0..sketch_dim.min(m_pad) {
        let j = i + rng.random_range(0..(m_pad - i));
        perm.swap(i, j);
    }
    let selected_rows: Vec<usize> = perm[..sketch_dim.min(m_pad)].to_vec();

    // Build each row of S by applying D then H to canonical basis vectors
    // More practically: S[k, :] = scale * e_{selected_rows[k]}^T H D
    // We represent S as a dense matrix for application to vectors
    // Actually: to apply S to a vector v of length m:
    //   1. Pad v to m_pad with zeros
    //   2. Apply D: u = D v_pad (element-wise multiply)
    //   3. Apply H: w = H u (WHT)
    //   4. Select rows: Sv = scale * w[selected_rows]
    // For the matrix form needed in matrix-matrix products, we build S explicitly.
    let mut s = vec![0.0f64; sketch_dim * m_pad];

    // Build each column of S^T (which is a row of S)
    // We process each basis vector e_j and apply D then H
    for j in 0..m {
        let mut col = vec![0.0f64; m_pad];
        col[j] = signs[j]; // D applied to e_j

        walsh_hadamard_transform(&mut col);

        // Now col = H D e_j; select rows
        for (k, &row_idx) in selected_rows.iter().enumerate() {
            s[k * m_pad + j] = scale * col[row_idx];
        }
    }

    (s, m_pad)
}

// ─── Matrix-vector operations ─────────────────────────────────────────────────

/// Compute S A for sketch matrix S (sketch_dim × m) and A (m × n), giving (sketch_dim × n).
fn sketch_matrix(s: &[f64], sketch_dim: usize, a: &Array2<f64>, m_actual: usize) -> Vec<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let m_s = m_actual.min(m); // rows to use from S (in case of padding)
    let mut sa = vec![0.0f64; sketch_dim * n];

    for k in 0..sketch_dim {
        for j in 0..n {
            let mut val = 0.0;
            for i in 0..m_s {
                val += s[k * m_actual + i] * a[[i, j]];
            }
            sa[k * n + j] = val;
        }
    }
    sa
}

/// Compute S b for sketch matrix S (sketch_dim × m) and vector b (m,), giving (sketch_dim,).
fn sketch_vector(s: &[f64], sketch_dim: usize, b: &[f64], m_actual: usize) -> Vec<f64> {
    let m_use = b.len().min(m_actual);
    let mut sb = vec![0.0f64; sketch_dim];
    for k in 0..sketch_dim {
        let mut val = 0.0;
        for i in 0..m_use {
            val += s[k * m_actual + i] * b[i];
        }
        sb[k] = val;
    }
    sb
}

/// Compute (SA)^T (SA x - Sb) — the sketched gradient with respect to x.
fn sketched_gradient(sa: &[f64], sb: &[f64], x: &[f64], sketch_dim: usize, n: usize) -> Vec<f64> {
    // r = SA x - Sb  (sketch_dim)
    let mut r = vec![0.0f64; sketch_dim];
    for k in 0..sketch_dim {
        let mut dot = 0.0;
        for j in 0..n {
            dot += sa[k * n + j] * x[j];
        }
        r[k] = dot - sb[k];
    }

    // g = (SA)^T r  (n)
    let mut g = vec![0.0f64; n];
    for j in 0..n {
        let mut val = 0.0;
        for k in 0..sketch_dim {
            val += sa[k * n + j] * r[k];
        }
        g[j] = val;
    }
    g
}

/// Estimate a safe step size as 1 / (max diagonal of (SA)^T SA).
fn estimate_step_size(sa: &[f64], sketch_dim: usize, n: usize) -> f64 {
    // Largest eigenvalue of (SA)^T SA is bounded by its maximum diagonal entry * n
    // For a safe choice, use 1 / ||SA||_F^2 (a conservative estimate)
    let norm_sq: f64 = sa.iter().map(|v| v * v).sum();
    if norm_sq < f64::EPSILON {
        1e-4
    } else {
        // Each step size should be < 2 / (largest eigenvalue of (SA)^T SA)
        // A conservative estimate: 1 / (sketch_dim * max_j sum_k sa[k,j]^2)
        let max_col_sq = (0..n)
            .map(|j| (0..sketch_dim).map(|k| sa[k * n + j].powi(2)).sum::<f64>())
            .fold(f64::NEG_INFINITY, f64::max);

        if max_col_sq > f64::EPSILON {
            0.9 / max_col_sq
        } else {
            1e-4
        }
    }
}

/// Compute the full residual norm ||Ax - b||.
fn full_residual_norm(a: &Array2<f64>, b: &[f64], x: &[f64]) -> f64 {
    let m = a.nrows();
    let mut norm_sq = 0.0;
    for i in 0..m {
        let row = a.row(i);
        let ax_i: f64 = row.iter().zip(x.iter()).map(|(aij, xj)| aij * xj).sum();
        let r = ax_i - b[i];
        norm_sq += r * r;
    }
    norm_sq.sqrt()
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Solve the least-squares problem min ||Ax - b||² using sketched gradient descent.
///
/// At each iteration, forms a random sketch S of A and b, computes the sketched
/// gradient g = (SA)^T (SA x - Sb), and performs a gradient step x ← x - α g.
///
/// # Arguments
/// - `a`: Coefficient matrix of shape (m, n) with m >> n typical.
/// - `b`: Right-hand side vector of length m.
/// - `config`: Solver configuration.
///
/// # Returns
/// A [`LsqResult`] with the approximate minimizer.
pub fn sketched_least_squares(
    a: &Array2<f64>,
    b: &[f64],
    config: &SketchedLeastSquaresConfig,
) -> OptimizeResult<LsqResult> {
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
    if config.sketch_dim == 0 {
        return Err(OptimizeError::InvalidParameter(
            "sketch_dim must be positive".to_string(),
        ));
    }

    let sketch_dim = config.sketch_dim.min(m); // Sketch cannot be larger than m

    let mut x = vec![0.0f64; n];
    let mut rng = StdRng::seed_from_u64(config.seed);

    // Precompute sketch once if not refreshing
    let precomputed_sketch: Option<(Vec<f64>, Vec<f64>)> = if !config.refresh_sketch {
        let (s, m_actual) = build_sketch_matrix(&config.sketch_type, sketch_dim, m, &mut rng);
        let sa = sketch_matrix(&s, sketch_dim, a, m_actual);
        let sb = sketch_vector(&s, sketch_dim, b, m_actual);
        Some((sa, sb))
    } else {
        None
    };

    for iter in 0..config.max_iter {
        let (sa, sb) = match &precomputed_sketch {
            Some((sa, sb)) => (sa.clone(), sb.clone()),
            None => {
                let (s, m_actual) =
                    build_sketch_matrix(&config.sketch_type, sketch_dim, m, &mut rng);
                let sa = sketch_matrix(&s, sketch_dim, a, m_actual);
                let sb = sketch_vector(&s, sketch_dim, b, m_actual);
                (sa, sb)
            }
        };

        let alpha = config
            .step_size
            .unwrap_or_else(|| estimate_step_size(&sa, sketch_dim, n));

        let g = sketched_gradient(&sa, &sb, &x, sketch_dim, n);

        // Compute update norm for convergence check
        let update_norm: f64 = g.iter().map(|v| (alpha * v).powi(2)).sum::<f64>().sqrt();
        let x_norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        let rel_change = update_norm / (1.0 + x_norm);

        // Apply update
        for (xi, gi) in x.iter_mut().zip(g.iter()) {
            *xi -= alpha * gi;
        }

        if rel_change < config.tol {
            let rn = full_residual_norm(a, b, &x);
            return Ok(LsqResult {
                x,
                residual_norm: rn,
                n_iter: iter + 1,
                converged: true,
            });
        }
    }

    let rn = full_residual_norm(a, b, &x);
    // Check convergence based on residual norm for consistency
    let converged = rn < config.tol * (1.0 + b.iter().map(|v| v * v).sum::<f64>().sqrt());

    Ok(LsqResult {
        x,
        residual_norm: rn,
        n_iter: config.max_iter,
        converged,
    })
}

/// Build a sketch matrix (flat row-major array) and return (S, m_actual).
///
/// `m_actual` may differ from `m` for Hadamard sketches (padding to power of 2).
fn build_sketch_matrix(
    sketch_type: &SketchType,
    sketch_dim: usize,
    m: usize,
    rng: &mut StdRng,
) -> (Vec<f64>, usize) {
    match sketch_type {
        SketchType::Gaussian => (build_gaussian_sketch(sketch_dim, m, rng), m),
        SketchType::Uniform => (build_rademacher_sketch(sketch_dim, m, rng), m),
        SketchType::CountSketch => (build_count_sketch(sketch_dim, m, rng), m),
        SketchType::Hadamard => build_hadamard_sketch(sketch_dim, m, rng),
        _ => (build_gaussian_sketch(sketch_dim, m, rng), m),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Build a simple overdetermined least-squares test case: A x = b with x* = [1, 2].
    fn make_lsq_problem(noise_scale: f64, rng: &mut StdRng) -> (Array2<f64>, Vec<f64>) {
        let m = 50;
        let n = 2;
        let x_true = vec![1.0, 2.0];

        let mut a_data = vec![0.0f64; m * n];
        let mut b = vec![0.0f64; m];

        for i in 0..m {
            let a0 = (i as f64) / m as f64;
            let a1 = 1.0 - a0;
            a_data[i * n] = a0;
            a_data[i * n + 1] = a1;
            b[i] = a0 * x_true[0] + a1 * x_true[1];
            if noise_scale > 0.0 {
                let u: f64 = rng.random::<f64>() - 0.5;
                b[i] += noise_scale * u;
            }
        }

        let a = Array2::from_shape_vec((m, n), a_data).expect("valid shape");
        (a, b)
    }

    #[test]
    fn test_sketched_ls_gaussian() {
        let mut rng = StdRng::seed_from_u64(0);
        let (a, b) = make_lsq_problem(0.0, &mut rng);

        let config = SketchedLeastSquaresConfig {
            sketch_dim: 30,
            sketch_type: SketchType::Gaussian,
            max_iter: 500,
            tol: 1e-5,
            seed: 42,
            refresh_sketch: true,
            step_size: Some(0.01),
        };

        let result = sketched_least_squares(&a, &b, &config).expect("sketched LS should succeed");
        // Should recover x ≈ [1, 2]
        assert!(
            (result.x[0] - 1.0).abs() < 0.1,
            "x[0] ≈ 1, got {}",
            result.x[0]
        );
        assert!(
            (result.x[1] - 2.0).abs() < 0.1,
            "x[1] ≈ 2, got {}",
            result.x[1]
        );
    }

    #[test]
    fn test_sketched_ls_count_sketch() {
        let mut rng = StdRng::seed_from_u64(0);
        let (a, b) = make_lsq_problem(0.0, &mut rng);

        let config = SketchedLeastSquaresConfig {
            sketch_dim: 30,
            sketch_type: SketchType::CountSketch,
            max_iter: 500,
            tol: 1e-5,
            seed: 77,
            refresh_sketch: true,
            step_size: Some(0.01),
        };

        let result =
            sketched_least_squares(&a, &b, &config).expect("count sketch LS should succeed");
        assert!(
            (result.x[0] - 1.0).abs() < 0.2,
            "x[0] ≈ 1, got {}",
            result.x[0]
        );
        assert!(
            (result.x[1] - 2.0).abs() < 0.2,
            "x[1] ≈ 2, got {}",
            result.x[1]
        );
    }

    #[test]
    fn test_sketched_ls_rademacher() {
        let mut rng = StdRng::seed_from_u64(0);
        let (a, b) = make_lsq_problem(0.0, &mut rng);

        let config = SketchedLeastSquaresConfig {
            sketch_dim: 25,
            sketch_type: SketchType::Uniform,
            max_iter: 500,
            tol: 1e-5,
            seed: 99,
            refresh_sketch: true,
            step_size: Some(0.01),
        };

        let result =
            sketched_least_squares(&a, &b, &config).expect("Rademacher sketch should succeed");
        assert!((result.x[0] - 1.0).abs() < 0.2, "x[0] ≈ 1");
        assert!((result.x[1] - 2.0).abs() < 0.2, "x[1] ≈ 2");
    }

    #[test]
    fn test_sketched_ls_hadamard() {
        let mut rng = StdRng::seed_from_u64(0);
        let (a, b) = make_lsq_problem(0.0, &mut rng);

        let config = SketchedLeastSquaresConfig {
            sketch_dim: 20,
            sketch_type: SketchType::Hadamard,
            max_iter: 500,
            tol: 1e-5,
            seed: 42,
            refresh_sketch: true,
            step_size: Some(0.01),
        };

        let result = sketched_least_squares(&a, &b, &config).expect("SRHT sketch should succeed");
        // SRHT may be less accurate due to padding; allow wider tolerance
        assert!(
            (result.x[0] - 1.0).abs() < 0.5,
            "x[0] ≈ 1, got {}",
            result.x[0]
        );
        assert!(
            (result.x[1] - 2.0).abs() < 0.5,
            "x[1] ≈ 2, got {}",
            result.x[1]
        );
    }

    #[test]
    fn test_sketched_ls_static_sketch() {
        let mut rng = StdRng::seed_from_u64(0);
        let (a, b) = make_lsq_problem(0.0, &mut rng);

        let config = SketchedLeastSquaresConfig {
            sketch_dim: 30,
            sketch_type: SketchType::Gaussian,
            max_iter: 500,
            tol: 1e-5,
            seed: 42,
            refresh_sketch: false, // fixed sketch throughout
            step_size: Some(0.01),
        };

        let result =
            sketched_least_squares(&a, &b, &config).expect("static sketch LS should succeed");
        // Fixed sketch is less powerful but should still reduce residual
        assert!(result.residual_norm < 5.0);
    }

    #[test]
    fn test_sketched_ls_invalid_input() {
        let a = Array2::<f64>::zeros((5, 2));
        let b = vec![1.0; 3]; // wrong length
        let result = sketched_least_squares(&a, &b, &SketchedLeastSquaresConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_sketched_ls_zero_sketch_dim_error() {
        let a = Array2::<f64>::eye(4);
        let b = vec![1.0; 4];
        let config = SketchedLeastSquaresConfig {
            sketch_dim: 0,
            ..SketchedLeastSquaresConfig::default()
        };
        let result = sketched_least_squares(&a, &b, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_sketched_ls_identity_system() {
        // Exact system A = I_4, b = [1,2,3,4], x* = [1,2,3,4]
        let a = Array2::<f64>::eye(4);
        let b = vec![1.0, 2.0, 3.0, 4.0];

        let config = SketchedLeastSquaresConfig {
            sketch_dim: 4,
            sketch_type: SketchType::Gaussian,
            max_iter: 1000,
            tol: 1e-6,
            seed: 42,
            refresh_sketch: true,
            step_size: Some(0.1),
        };

        let result = sketched_least_squares(&a, &b, &config).expect("identity system should work");
        for (i, (&xi, &bi)) in result.x.iter().zip(b.iter()).enumerate() {
            assert!((xi - bi).abs() < 0.5, "x[{}] ≈ {}, got {}", i, bi, xi);
        }
    }
}
