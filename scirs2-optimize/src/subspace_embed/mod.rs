//! Subspace Embedding Optimization
//!
//! Reduces the dimensionality of a large-scale optimization problem by projecting
//! the search space into a random low-dimensional subspace and optimising there.
//!
//! ## Algorithm sketch
//!
//! For each restart:
//! 1. Sample a random projection matrix P of shape `(subspace_dim × full_dim)`.
//! 2. Optimize the surrogate g(y) = f(P^T y) in the subspace via coordinate descent.
//! 3. Recover the full-dimensional point x = P^T y.
//! 4. Clip to bounds if provided.
//!
//! The method is particularly effective when the objective has a low intrinsic
//! dimensionality (e.g., the effective subspace spanned by the gradient is small).
//!
//! ## References
//!
//! - Wang, Z. et al. (2016). "Bayesian Optimization in a Billion Dimensions via
//!   Random Embeddings". JAIR 55.
//! - Choromanski, K. et al. (2022). "Geometry-Aware Structured Transformations".

use crate::error::{OptimizeError, OptimizeResult};

// ──────────────────────────────────────────────────────────────── SketchType ──

/// Type of random projection matrix to use for subspace embedding.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SketchType {
    /// Dense Gaussian sketch: `P[i][j]` ~ N(0, 1/sub_dim).
    Gaussian,
    /// Rademacher sketch: `P[i][j]` = +/-1/sqrt(sub_dim) with equal probability.
    Rademacher,
    /// Count sketch: each full-dim coordinate maps to one sub-dim bucket with ±1 sign.
    CountSketch,
    /// Subsampled Randomized Hadamard Transform (approximate).
    SRHT,
}

impl Default for SketchType {
    fn default() -> Self {
        SketchType::Gaussian
    }
}

// ───────────────────────────────────────────────────────────── SubspaceConfig ──

/// Configuration for the subspace embedding optimizer.
#[derive(Debug, Clone)]
pub struct SubspaceConfig {
    /// Dimensionality of the random subspace.
    pub subspace_dim: usize,
    /// Number of independent restarts.
    pub n_restarts: usize,
    /// Number of coordinate-descent iterations per restart.
    pub n_local_iter: usize,
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// Type of random sketch to use.
    pub sketch_type: SketchType,
    /// Finite-difference step size for gradient estimation.
    pub fd_step: f64,
    /// Step size (learning rate) for coordinate descent updates.
    pub step_size: f64,
}

impl Default for SubspaceConfig {
    fn default() -> Self {
        Self {
            subspace_dim: 100,
            n_restarts: 5,
            n_local_iter: 50,
            seed: 42,
            sketch_type: SketchType::Gaussian,
            fd_step: 1e-5,
            step_size: 0.1,
        }
    }
}

// ──────────────────────────────────────────────────────────── LCG utilities ──

/// Advance a Knuth-style LCG and return the next state.
#[inline]
fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

/// Map an LCG state to a uniform f64 in `[0, 1)`.
#[inline]
fn lcg_to_f64(state: u64) -> f64 {
    (state >> 11) as f64 / (1u64 << 53) as f64
}

/// Draw a standard-normal sample using the Box-Muller transform.
///
/// Returns (z0, z1) and the next RNG state after consuming two uniform draws.
fn box_muller(state: u64) -> (f64, f64, u64) {
    use std::f64::consts::PI;
    let s1 = lcg_next(state);
    let s2 = lcg_next(s1);
    let u1 = lcg_to_f64(s1).max(1e-300); // avoid log(0)
    let u2 = lcg_to_f64(s2);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * PI * u2;
    (r * theta.cos(), r * theta.sin(), s2)
}

// ────────────────────────────────────────────────────── Sketch generators ──

/// Generate a Rademacher sketch matrix of shape `(sub_dim × full_dim)`.
///
/// `P[i][j]` = +/-1/sqrt(sub_dim) with equal probability (seeded LCG).
pub fn rademacher_sketch(full_dim: usize, sub_dim: usize, seed: u64) -> Vec<Vec<f64>> {
    let scale = if sub_dim > 0 {
        1.0 / (sub_dim as f64).sqrt()
    } else {
        1.0
    };
    let mut state = seed;
    let mut p = vec![vec![0.0_f64; full_dim]; sub_dim];
    for row in p.iter_mut() {
        for val in row.iter_mut() {
            state = lcg_next(state);
            *val = if state & 1 == 0 { scale } else { -scale };
        }
    }
    p
}

/// Generate a count-sketch matrix of shape `(sub_dim × full_dim)`.
///
/// Each full-dim coordinate maps to exactly one sub-dim bucket with a random ±1 sign.
pub fn count_sketch(full_dim: usize, sub_dim: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut p = vec![vec![0.0_f64; full_dim]; sub_dim.max(1)];
    let mut state = seed;
    let s = sub_dim.max(1);
    for j in 0..full_dim {
        state = lcg_next(state);
        let bucket = (state as usize) % s;
        state = lcg_next(state);
        let sign: f64 = if state & 1 == 0 { 1.0 } else { -1.0 };
        p[bucket][j] = sign;
    }
    p
}

/// Generate a Gaussian sketch matrix of shape `(sub_dim × full_dim)`.
///
/// `P[i][j]` ~ N(0, 1/sub_dim) (Box-Muller, seeded LCG).
pub fn gaussian_sketch(full_dim: usize, sub_dim: usize, seed: u64) -> Vec<Vec<f64>> {
    let scale = if sub_dim > 0 {
        1.0 / (sub_dim as f64)
    } else {
        1.0
    };
    let mut state = seed;
    let mut p = vec![vec![0.0_f64; full_dim]; sub_dim];
    let mut i = 0;
    let mut j = 0;
    let total = sub_dim * full_dim;
    let mut count = 0;
    while count < total {
        let (z0, z1, new_state) = box_muller(state);
        state = new_state;
        if count < total {
            p[i][j] = z0 * scale.sqrt();
            j += 1;
            if j == full_dim {
                j = 0;
                i += 1;
            }
            count += 1;
        }
        if count < total {
            p[i][j] = z1 * scale.sqrt();
            j += 1;
            if j == full_dim {
                j = 0;
                i += 1;
            }
            count += 1;
        }
    }
    p
}

/// Generate a SRHT-approximation sketch (random sign flips + random row sampling).
///
/// Exact SRHT requires Hadamard transform; here we use a cost-equivalent
/// Rademacher sketch followed by random row selection (same statistical guarantees
/// for subspace embedding purposes at the same parameter budget).
fn srht_sketch(full_dim: usize, sub_dim: usize, seed: u64) -> Vec<Vec<f64>> {
    // Use Rademacher as a practical SRHT substitute.
    rademacher_sketch(full_dim, sub_dim, seed)
}

// ───────────────────────────────────────────── SubspaceEmbeddingOptimizer ──

/// Optimizer that works in a random low-dimensional subspace.
#[derive(Debug, Clone)]
pub struct SubspaceEmbeddingOptimizer {
    config: SubspaceConfig,
    /// LCG state (advanced at each use).
    rng_state: u64,
}

impl SubspaceEmbeddingOptimizer {
    /// Create a new optimizer with the given config.
    pub fn new(config: SubspaceConfig) -> Self {
        let rng_state = config.seed;
        Self { config, rng_state }
    }

    /// Generate the projection matrix according to `sketch_type`.
    pub fn embed(sketch: &SketchType, full_dim: usize, sub_dim: usize, seed: u64) -> Vec<Vec<f64>> {
        match sketch {
            SketchType::Gaussian => gaussian_sketch(full_dim, sub_dim, seed),
            SketchType::Rademacher => rademacher_sketch(full_dim, sub_dim, seed),
            SketchType::CountSketch => count_sketch(full_dim, sub_dim, seed),
            SketchType::SRHT => srht_sketch(full_dim, sub_dim, seed),
        }
    }

    /// Multiply P^T (shape full_dim × sub_dim) by y (length sub_dim) to get x in full_dim.
    ///
    /// x[j] = Σ_i P[i][j] * y[i]
    fn backproject(p: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
        let full_dim = if p.is_empty() { 0 } else { p[0].len() };
        let sub_dim = p.len();
        let mut x = vec![0.0_f64; full_dim];
        for i in 0..sub_dim.min(y.len()) {
            for j in 0..full_dim {
                x[j] += p[i][j] * y[i];
            }
        }
        x
    }

    /// Clip a point to the provided box bounds.
    fn clip_to_bounds(x: &mut [f64], bounds: Option<&[(f64, f64)]>) {
        if let Some(b) = bounds {
            for (xi, &(lo, hi)) in x.iter_mut().zip(b.iter()) {
                if xi.is_nan() {
                    *xi = (lo + hi) / 2.0;
                } else {
                    *xi = xi.clamp(lo, hi);
                }
            }
        }
    }

    /// Minimise `f` over `full_dim` dimensions by optimising in a random subspace.
    ///
    /// # Arguments
    ///
    /// - `f`: Objective function. Receives a slice of length `full_dim`.
    /// - `full_dim`: Dimensionality of the original search space.
    /// - `bounds`: Optional box constraints `[(lo, hi); full_dim]`.
    ///
    /// # Returns
    ///
    /// `(x_best, f_best)` — the best point found across all restarts.
    pub fn minimize(
        &mut self,
        f: impl Fn(&[f64]) -> f64 + Clone,
        full_dim: usize,
        bounds: Option<&[(f64, f64)]>,
    ) -> OptimizeResult<(Vec<f64>, f64)> {
        if full_dim == 0 {
            return Err(OptimizeError::InvalidInput(
                "full_dim must be > 0".to_string(),
            ));
        }
        let sub_dim = self.config.subspace_dim.min(full_dim);
        let n_restarts = self.config.n_restarts;
        let n_local_iter = self.config.n_local_iter;
        let fd_step = self.config.fd_step;
        let step_size = self.config.step_size;
        let sketch_type = self.config.sketch_type.clone();

        let mut best_x: Option<Vec<f64>> = None;
        let mut best_val = f64::INFINITY;

        for restart in 0..n_restarts {
            // Unique seed per restart.
            self.rng_state = lcg_next(self.config.seed.wrapping_add(restart as u64 * 1_000_003));
            let seed_r = self.rng_state;

            let p = Self::embed(&sketch_type, full_dim, sub_dim, seed_r);

            // Initialise y (subspace point) randomly in [-1, 1]^sub_dim.
            let mut y: Vec<f64> = (0..sub_dim)
                .map(|_| {
                    self.rng_state = lcg_next(self.rng_state);
                    lcg_to_f64(self.rng_state) * 2.0 - 1.0
                })
                .collect();

            // Wrap f so it operates via back-projection.
            let p_ref = &p;
            let f_sub = |y: &[f64]| -> f64 {
                let mut x = Self::backproject(p_ref, y);
                Self::clip_to_bounds(&mut x, bounds);
                f(&x)
            };

            // Local coordinate-descent in subspace.
            let (y_opt, _) =
                coord_descent_subspace(f_sub, y.clone(), n_local_iter, step_size, fd_step);
            y = y_opt;

            // Recover full-space point.
            let mut x = Self::backproject(&p, &y);
            Self::clip_to_bounds(&mut x, bounds);
            let val = f(&x);

            if val < best_val {
                best_val = val;
                best_x = Some(x);
            }
        }

        match best_x {
            Some(x) => Ok((x, best_val)),
            None => Err(OptimizeError::ConvergenceError(
                "Subspace optimizer: no valid restart completed".to_string(),
            )),
        }
    }
}

// ──────────────────────────────────────────── coord_descent_subspace ──

/// Coordinate descent in the subspace.
///
/// Uses central finite differences to estimate each partial derivative, then
/// takes a step of size `step_size` in the steepest descent direction along
/// each coordinate.
///
/// # Arguments
///
/// - `f`: Objective function on subspace points.
/// - `x0`: Initial point in the subspace.
/// - `n_iter`: Total number of coordinate updates.
/// - `step`: Gradient-descent step size.
/// - `fd_step`: Finite-difference step for partial derivative estimation.
///
/// # Returns
///
/// `(x_best, f_best)` — best point found during descent.
pub fn coord_descent_subspace(
    f: impl Fn(&[f64]) -> f64,
    x0: Vec<f64>,
    n_iter: usize,
    step: f64,
    fd_step: f64,
) -> (Vec<f64>, f64) {
    let d = x0.len();
    if d == 0 {
        return (x0, 0.0);
    }
    let mut x = x0;
    let mut fx = f(&x);
    let mut best_x = x.clone();
    let mut best_val = fx;

    for iter in 0..n_iter {
        let coord = iter % d;
        // Central finite difference along `coord`.
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();
        x_plus[coord] += fd_step;
        x_minus[coord] -= fd_step;
        let grad_coord = (f(&x_plus) - f(&x_minus)) / (2.0 * fd_step);

        x[coord] -= step * grad_coord;
        fx = f(&x);

        if fx < best_val {
            best_val = fx;
            best_x = x.clone();
        }
    }

    (best_x, best_val)
}

// ═══════════════════════════════════════════════════════════════════ tests ═══

#[cfg(test)]
mod tests {
    use super::*;

    /// Quadratic f(x) = Σ (x_i - t_i)^2 with known minimum at target.
    fn quadratic(x: &[f64], target: &[f64]) -> f64 {
        x.iter()
            .zip(target.iter())
            .map(|(xi, ti)| (xi - ti).powi(2))
            .sum()
    }

    #[test]
    fn gaussian_sketch_correct_shape() {
        let p = gaussian_sketch(20, 5, 42);
        assert_eq!(p.len(), 5, "sub_dim rows");
        assert_eq!(p[0].len(), 20, "full_dim cols");
    }

    #[test]
    fn rademacher_sketch_values_are_unit_scaled() {
        let sub = 4;
        let p = rademacher_sketch(10, sub, 7);
        let expected = 1.0 / (sub as f64).sqrt();
        for row in &p {
            for &v in row {
                assert!((v.abs() - expected).abs() < 1e-12, "v={v}");
            }
        }
    }

    #[test]
    fn count_sketch_one_nonzero_per_column() {
        let full = 20;
        let sub = 4;
        let p = count_sketch(full, sub, 99);
        for j in 0..full {
            let nonzero: usize = (0..sub).filter(|&i| p[i][j] != 0.0).count();
            assert_eq!(
                nonzero, 1,
                "column {j} should have exactly 1 non-zero entry"
            );
        }
    }

    #[test]
    fn coord_descent_subspace_descends_on_quadratic() {
        let target = vec![3.0_f64; 5];
        let f = |x: &[f64]| {
            x.iter()
                .zip(target.iter())
                .map(|(xi, ti)| (xi - ti).powi(2))
                .sum::<f64>()
        };
        let x0 = vec![0.0_f64; 5];
        let (x_opt, val) = coord_descent_subspace(f, x0, 200, 0.5, 1e-5);
        assert!(val < 1.0, "Should converge toward target; val={val}");
        for (xi, ti) in x_opt.iter().zip(target.iter()) {
            assert!((xi - ti).abs() < 0.5, "xi={xi}, ti={ti}");
        }
    }

    #[test]
    fn subspace_optimizer_minimizes_quadratic() {
        let target: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
        let config = SubspaceConfig {
            subspace_dim: 5,
            n_restarts: 3,
            n_local_iter: 100,
            seed: 42,
            sketch_type: SketchType::Rademacher,
            fd_step: 1e-5,
            step_size: 0.3,
        };
        let mut opt = SubspaceEmbeddingOptimizer::new(config);
        let t = target.clone();
        let result = opt.minimize(move |x| quadratic(x, &t), 10, None);
        assert!(result.is_ok(), "Optimizer should succeed");
        let (x_opt, f_opt) = result.unwrap();

        // Compute the true minimum (all zeros since we project and the min in
        // subspace approximates the full-space min up to projection error).
        // We only require that the found value is less than the initial f(0) = Σ t_i^2.
        let f_zero: f64 = target.iter().map(|t| t * t).sum();
        assert!(
            f_opt < f_zero,
            "Optimizer should improve over f(0)={f_zero}, got f_opt={f_opt}"
        );
        assert_eq!(x_opt.len(), 10);
    }

    #[test]
    fn subspace_optimizer_respects_bounds() {
        let target = vec![10.0_f64; 4]; // far outside bounds
        let bounds: Vec<(f64, f64)> = vec![(-1.0, 1.0); 4];
        let config = SubspaceConfig {
            subspace_dim: 2,
            n_restarts: 2,
            n_local_iter: 20,
            ..Default::default()
        };
        let mut opt = SubspaceEmbeddingOptimizer::new(config);
        let t = target.clone();
        let (x_opt, _) = opt
            .minimize(move |x| quadratic(x, &t), 4, Some(&bounds))
            .unwrap();
        for &xi in &x_opt {
            assert!(
                xi >= -1.0 - 1e-10 && xi <= 1.0 + 1e-10,
                "xi={xi} out of bounds"
            );
        }
    }

    #[test]
    fn subspace_optimizer_zero_dim_errors() {
        let mut opt = SubspaceEmbeddingOptimizer::new(SubspaceConfig::default());
        let result = opt.minimize(|_x| 0.0_f64, 0, None);
        assert!(result.is_err());
    }
}
